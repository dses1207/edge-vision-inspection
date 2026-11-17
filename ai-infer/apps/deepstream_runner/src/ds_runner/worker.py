"""worker：後處理 → 站別判定 → FrameResult → publish。

跑在自己的 thread，從 probe 丟來的 queue 取料。這裡是唯一呼叫 libs 與 stations 的地方；
probe 只搬張量。判定再慢、站別再有 bug，都只死這一路，GStreamer 不停。

process() 是純函式路徑，測試直接餵假張量；run() 是 thread 迴圈。

一線多站（2026-09-24）：一條線可以掛主站 + 附加站（例：ALPHA 主模型 + PPE 的 YOLO-World），
每個站別各自一個模型、一個 nvinferserver。TensorJob.station 說這批張量是哪個站別的模型出的；
worker 依它挑後處理器與站別實例。沒帶（舊 probe）就是主站。
"""

from __future__ import annotations

import logging
import os
import queue
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from edgevision_contracts import CoordSystem, FrameResult, ImageMeta, Verdict, VerdictStatus
from edgevision_postprocess import compute_letterbox, compute_letterbox_roi, make_postprocessor
from edgevision_station_sdk import Station, load_station

from ds_runner.config import Binding, LineConfig

logger = logging.getLogger(__name__)

from ds_runner.publish import Publisher, log_publisher  # noqa: F401  （相容舊 import）


@dataclass
class TensorJob:
    camera_id: str
    out0: np.ndarray
    proto: np.ndarray | None
    width: int
    height: int
    image: Any = None            # 選配：要存圖時由 probe 附上
    station: str | None = None   # 哪個站別的模型出的張量；None = 主站
    pts: int | None = None       # 影片測試：這幀的 buf_pts（奈秒）；線上不帶
    frame_pts: int | None = None  # 這幀的 buf_pts（奈秒），線上也帶；只給有 set_frame_time 的站別當判定時間，不進結果


@dataclass
class BindingRuntime:
    """一個 (模型, 站別) 在 worker 裡的東西：後處理器 + 每台相機一個站別實例。"""
    binding: Binding
    postproc: Any
    stations: dict[str, Station] = field(default_factory=dict)
    frames: dict[str, int] = field(default_factory=dict)

    @property
    def class_names(self) -> list[str]:
        return list(self.binding.model.classes)


@dataclass
class Worker:
    """一條 worker 負責一組相機（camera_ids=None 表示全部）。

    同一台相機的幀必須由同一條 worker 依序處理：站別有跨幀狀態（連續幀、冷卻），
    兩條 thread 同時碰同一台會把順序和狀態弄亂。分片由 WorkerPool 負責。
    """
    cfg: LineConfig
    publish: Publisher
    model_version: str = "unknown"
    camera_ids: list[str] | None = None
    runtimes: dict[str, BindingRuntime] = field(default_factory=dict)
    # 串流恢復保護（見 _resume_hold）：同一台相機超過 resume_gap_s 秒沒有新幀，恢復後先停判 resume_hold_s 秒
    resume_gap_s: float = field(default_factory=lambda: float(os.environ.get("RESUME_GAP_S", "2")))
    resume_hold_s: float = field(default_factory=lambda: float(os.environ.get("RESUME_HOLD_S", "5")))
    clock: Callable[[], float] = time.monotonic

    def __post_init__(self) -> None:
        import contextlib
        with contextlib.suppress(Exception):   # 只是效能設定，設不上不影響正確性
            import cv2
            cv2.setNumThreads(1)        # 平行交給分片，OpenCV 自己不要再開執行緒池
        mine = set(self.camera_ids) if self.camera_ids is not None else None
        for b in self.cfg.bindings():
            m = b.model
            rt = BindingRuntime(b, make_postprocessor(b.task, m.classes, imgsz=m.imgsz, conf=m.conf, iou=m.iou))
            # 每台相機一個站別實例：現場值各異、跨幀狀態不可共用
            for cam in self.cfg.active_cameras:
                if mine is not None and cam.id not in mine:
                    continue
                rt.stations[cam.id] = load_station(b.station, self.cfg.params_for(cam, b.station))
                rt.frames[cam.id] = 0
            self.runtimes[b.key] = rt
        self.primary = self.runtimes[self.cfg.station]
        # 主站模型只看 ROI 的相機：座標要從「letterbox 後的 ROI」換回全幅（見 roi.py）。
        # 像素用 mux 畫面尺寸（cfg.frame_size），跟 pipeline 放假物件的框同一套座標。
        from ds_runner.roi import roi_pixels
        fw, fh = self.cfg.frame_size
        self._last_seen: dict[tuple[str, str], float] = {}
        self._last_ng: dict[tuple[str, str], float] = {}
        self.ng_cooldown_s = float(self.cfg.line_params.get("ng_cooldown_s", os.environ.get("NG_COOLDOWN_S", "30")))
        self._hold_until: dict[tuple[str, str], float] = {}
        self.roi_px: dict[str, tuple[int, int, int, int]] = {
            cam.id: roi_pixels(cam.infer_roi, fw, fh) for cam in self.cfg.active_cameras if cam.infer_roi}
        logger.info("[worker] 線路 %s 站別 %s 相機 %s", self.cfg.id,
                    [b.key for b in self.cfg.bindings()], sorted(self.primary.stations))

    # ── 主站的投影（舊介面；probe / pool / 測試都在用）──────────
    @property
    def postproc(self):
        return self.primary.postproc

    @property
    def bbox_only_postproc(self):
        # probe 同步畫框用：與 postproc 同一個實例（postprocessor 無狀態，多執行緒可共用）。
        # probe 呼叫時傳 proto=None 就只解 bbox 不重建 mask —— 快 50 倍，而畫框只需要 bbox。
        return self.primary.postproc

    @property
    def class_names(self) -> list[str]:
        return self.primary.class_names

    @property
    def model_name(self) -> str:
        return self.primary.binding.model.name

    @property
    def task(self) -> str:
        return self.primary.binding.task

    @property
    def stations(self) -> dict[str, Station]:
        return self.primary.stations

    @property
    def frames(self) -> dict[str, int]:
        return self.primary.frames

    def runtime(self, station: str | None) -> BindingRuntime:
        return self.primary if station is None else self.runtimes[station]

    def update_camera_params(self, camera_id: str, params: dict[str, Any], station: str | None = None) -> None:
        """熱套用現場值：單一置換點，不建第二個實例。"""
        self.runtime(station).stations[camera_id].update_params(params)

    def letterbox_for(self, camera_id: str, rt: BindingRuntime, width: int, height: int):
        roi = self.roi_px.get(camera_id) if rt.binding.primary else None
        if roi is None:
            return compute_letterbox(width, height, rt.postproc.imgsz)
        return compute_letterbox_roi(*roi, rt.postproc.imgsz)

    def _resume_hold(self, rt: BindingRuntime, camera_id: str) -> bool:
        """串流斷了又恢復 → 這段時間不判定，回 True。

        ⚠️ 為什麼（2026-10-28 正式環境）：相機到 go2rtc 的連線約每 30 分鐘斷一次、10～15 秒後重連。
           重連後頭幾幀缺參考幀，解出來是停在舊畫面的殘影＋馬賽克（相機自己的時間字都是花的），
           要等下一個關鍵幀才恢復。模型把殘影當成工件 → 判「違規動作」，當天 cam156 的 NG 全落在重連時間點上。
        做法：同一台相機（每個站別各算）超過 resume_gap_s 秒沒幀 → 清掉站別的跨幀狀態，
        接下來 resume_hold_s 秒（涵蓋到下一個關鍵幀，Tapo 約 2 秒一個）只推論不判定。
        影片測試（帶 pts）不會斷流，不套用。
        """
        if self.resume_gap_s <= 0:
            return False
        key, now = (rt.binding.key, camera_id), self.clock()
        last = self._last_seen.get(key)
        self._last_seen[key] = now
        if last is not None and now - last > self.resume_gap_s:
            rt.stations[camera_id].reset()
            self._hold_until[key] = now + self.resume_hold_s
            logger.warning("[worker] %s/%s 停了 %.1f 秒沒有影像，恢復後 %.0f 秒內不判定（重連後的畫面可能是殘影）",
                           camera_id, rt.binding.key, now - last, self.resume_hold_s)
        return now < self._hold_until.get(key, 0.0)

    def _ng_cooldown(self, rt: BindingRuntime, job: TensorJob, verdict: Verdict) -> Verdict:
        """同一台相機（每個站別各算）兩次 NG 至少間隔 ng_cooldown_s 秒；間隔內的 NG 改回 OK（附原因）。

        ⚠️ 為什麼（2026-10-30）：報表改成「一次 NG ＝ 一筆告警 ＝ 一段影片」，去重只做這一道。
           以前站別各自的冷卻很短（B ECHO 照舊程式 5 秒）→ 同一件事每 5 秒一次 NG，一天 1,500 段影片。
           舊系統同樣在錄影之前擋一道「警報觸發間隔」，預設 30 秒。
        秒數：line_params.ng_cooldown_s（每條線可調）> 環境變數 NG_COOLDOWN_S > 30；0 = 不擋。
        影片測試用影片時間（pts）算，結果跟線上一致、不受跑多快影響。
        """
        if verdict.status != VerdictStatus.NG or self.ng_cooldown_s <= 0:
            return verdict
        key = (rt.binding.key, job.camera_id)
        now = job.pts / 1e9 if job.pts is not None else self.clock()
        last = self._last_ng.get(key)
        if last is not None and 0 <= now - last < self.ng_cooldown_s:
            return Verdict.ok("NG 冷卻中", suppressed=verdict.reason, since_last_ng_s=round(now - last, 1))
        self._last_ng[key] = now
        return verdict

    def process(self, job: TensorJob) -> FrameResult:
        rt = self.runtime(job.station)
        lb = self.letterbox_for(job.camera_id, rt, job.width, job.height)
        dets = rt.postproc(job.out0, job.proto, lb, job.width, job.height)
        st = rt.stations[job.camera_id]
        if job.frame_pts is not None and hasattr(st, "set_frame_time"):
            st.set_frame_time(job.frame_pts / 1e9)   # 用畫面時間判定；網路補送的幀不會被擠在同一時刻
        if job.pts is None and self._resume_hold(rt, job.camera_id):
            verdict = Verdict.unknown("串流剛恢復，畫面可能不完整，暫停判定")
        else:
            verdict = st.evaluate(dets, job.width, job.height)
            verdict = self._ng_cooldown(rt, job, verdict)
        rt.frames[job.camera_id] += 1
        return FrameResult(
            camera_id=job.camera_id,
            frame_id=str(rt.frames[job.camera_id]),
            model_name=rt.binding.model.name,
            model_version=self.model_version,
            station=rt.binding.key,
            coord_system=CoordSystem.ORIGINAL,
            image_meta=ImageMeta(width=job.width, height=job.height, letterbox=lb),
            detections=dets,
            verdict=verdict,
            pts=job.pts,
        )

    def run(self, q: queue.Queue[TensorJob], stop: threading.Event) -> None:
        while not stop.is_set():
            try:
                job = q.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                result = self.process(job)
            except Exception:                      # 站別炸了：記錄、丟這幀、繼續
                logger.exception("[worker] %s 判定失敗", job.camera_id)
                continue
            try:
                self.publish(result, job.image)
            except Exception:
                logger.exception("[worker] publish 失敗")


class ShardRouter:
    """probe 拿到的「queue」：依相機把 TensorJob 丟到它所屬 shard 的 queue。

    介面跟 queue.Queue 一樣（put_nowait / put），probe 不用知道有分片；
    put_nowait 滿了拋 queue.Full 讓 probe 照舊計 dropped，並額外按 shard 計數。
    put 會等（不限速的影片測試用，回壓上游、不丟幀）。
    """

    def __init__(self, queues: list[queue.Queue], shard_of: dict[str, int], stats: dict | None = None) -> None:
        self.queues = queues
        self.shard_of = shard_of
        self.stats = stats if stats is not None else {}

    def put_nowait(self, job: TensorJob) -> None:
        shard = self.shard_of[job.camera_id]
        try:
            self.queues[shard].put_nowait(job)
        except queue.Full:
            k = f"dropped_shard{shard}"
            self.stats[k] = self.stats.get(k, 0) + 1
            raise

    def put(self, job: TensorJob) -> None:
        self.queues[self.shard_of[job.camera_id]].put(job)


def shard_cameras(camera_ids: list[str], n_workers: int) -> dict[str, int]:
    """相機 → shard 編號。輪流分配（第 i 台給 i % n），固定且可預測。"""
    n = max(1, min(n_workers, len(camera_ids)))
    return {cam: i % n for i, cam in enumerate(camera_ids)}


class WorkerPool:
    """N 條 worker thread、N 個 queue，每台相機固定歸一條。

    n_workers 預設 min(4, 相機數)；queue 深度預設 20（吸收抖動用，不是緩衝區：
    每格約 4.6 MB 張量，積到滿代表 worker 真的跟不上，那時丟幀比越積越晚好）。
    """

    def __init__(self, cfg: LineConfig, publish: Publisher, *, model_version: str = "unknown",
                 n_workers: int | None = None, queue_size: int = 20, stats: dict | None = None) -> None:
        cams = cfg.camera_ids
        n = n_workers if n_workers else min(4, len(cams))
        self.shard_of = shard_cameras(cams, n)
        n = max(self.shard_of.values()) + 1
        self.queues: list[queue.Queue] = [queue.Queue(maxsize=queue_size) for _ in range(n)]
        self.workers: list[Worker] = []
        for i in range(n):
            mine = [c for c, s in self.shard_of.items() if s == i]
            self.workers.append(Worker(cfg, publish=publish, model_version=model_version, camera_ids=mine))
        self.router = ShardRouter(self.queues, self.shard_of, stats)
        self._threads: list[threading.Thread] = []

    @property
    def bbox_only_postproc(self):
        return self.workers[0].bbox_only_postproc

    def disable_roi(self) -> None:
        """INFER_ROI_ENABLED=0：主站全幅推論，座標還原也回到全幅 letterbox。"""
        for w in self.workers:
            w.roi_px.clear()

    @property
    def class_names(self) -> list[str]:
        return self.workers[0].class_names

    def runtimes(self) -> dict[str, BindingRuntime]:
        """給 probe 用：每個站別的（無狀態）後處理器與類別名。取第 0 條 worker 的即可。"""
        return self.workers[0].runtimes

    def update_camera_params(self, camera_id: str, params: dict[str, Any], station: str | None = None) -> None:
        self.workers[self.shard_of[camera_id]].update_camera_params(camera_id, params, station)

    def start(self, stop: threading.Event) -> None:
        for i, (w, q) in enumerate(zip(self.workers, self.queues)):
            t = threading.Thread(target=w.run, args=(q, stop), daemon=True, name=f"worker-{i}")
            t.start()
            self._threads.append(t)
        logger.info("[worker] %d 條 worker：%s", len(self.workers),
                    {i: sorted(c for c, s in self.shard_of.items() if s == i) for i in range(len(self.workers))})
