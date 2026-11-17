"""線路狀態回報（desired / reported 的 reported 端），HTTP 版。

⭐ 控制端把設定寫進 DB 之後不能假設推送成功（Envoy 式靜默拒絕）。runner 主動回報
「我實際載入的是第幾版」，控制端比對 lines.config_version 與 line_status.reported_version。

monorepo 的 runner 只依賴 HTTP，不帶 psycopg，所以打 output-api：
    POST {CONFIG_API}/api/config/lines/{line_id}/status
    {"reported_version": 3, "apply_state": "ok"|"applying"|"failed",
     "apply_error": null|"…", "cameras_running": 2,
     "stats": {"cameras": {"cam156": 14.9, "cam193": 0.0}, "fps": 118.6, "dropped": 0,
               "sync_ms_avg": 0.61, "worker_queue": 3}}
    → 服務端 upsert line_status（heartbeat_at = now()、stats 存 JSONB）

    cameras_running 是這一段【真的有幀進來】的相機數（2026-09-29 前是設定裡的台數，相機 404 也算）。

    stats 是給健康巡檢看的：
      cameras      每台相機這一段收到的幀/秒（跳幀沒推論的也算）；設定裡有但沒幀的是 0.0
      fps          這一段回報間隔內實際推論的幀/秒（所有相機加總）
      dropped      累計因 worker 跟不上而丟掉的張量數
      sync_ms_avg  probe 同步路徑每幀平均毫秒（解框 + mask 小塊）
      worker_queue 目前最深的 worker queue 長度
      gpu_free_mib / gpu_total_mib  這個程序看得到的 GPU 剩多少（MIG 切片的量；nvidia-smi 在 MIG 下看不到）

    skipped_cameras  {相機: 原因}：設定有問題被跳過的相機（只停那一台，同線其他台照跑；2026-11-03）

    apply_state：啟動時先報 applying，第一幀過推論才轉 ok（watchdog.py）；卡住報 failed + 原因。

回報失敗只記 log，不影響推論（與舊 edgevision/config/reporter.py 同哲學）。
"""

from __future__ import annotations

import json
import logging
import threading
from collections.abc import Callable
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)


def status_url(base_url: str, line_id: str) -> str:
    return f"{base_url.rstrip('/')}/api/config/lines/{line_id}/status"


def _post_json(url: str, payload: dict, timeout: float = 5.0) -> None:
    req = Request(url, data=json.dumps(payload).encode("utf-8"), method="POST",
                  headers={"Content-Type": "application/json"})
    with urlopen(req, timeout=timeout):   # 內網設定服務
        pass


def make_stats_provider(stats: dict, queues: list | None = None,
                        clock: Callable[[], float] | None = None,
                        cameras: list[str] | None = None) -> Callable[[], dict]:
    """把 probe 的累計統計換算成「這段間隔的速率」。純函式工廠，可測。

    stats 是 probe 每幀累加的 dict（frames / dropped / sync_ms / sync_n），
    這裡記住上一次的快照，每次呼叫算差分：fps 與 sync_ms_avg 都是【這一段】的，
    不是啟動以來的平均 —— 健康巡檢要看的是「現在」有沒有變慢，不是歷史。
    """
    import time as _time
    now = clock or _time.monotonic
    last = {"t": now(), "frames": stats.get("frames", 0),
            "sync_ms": stats.get("sync_ms", 0.0), "sync_n": stats.get("sync_n", 0),
            "cam_frames": dict(stats.get("cam_frames", {}))}

    def provider() -> dict:
        t = now()
        frames, sync_ms, sync_n = stats.get("frames", 0), stats.get("sync_ms", 0.0), stats.get("sync_n", 0)
        dt = max(t - last["t"], 1e-6)
        d_frames, d_ms, d_n = frames - last["frames"], sync_ms - last["sync_ms"], sync_n - last["sync_n"]
        cf = dict(stats.get("cam_frames", {}))
        # 相機層級：這一段每台收到幾 fps。設定裡有、但一幀都沒來的相機也要列出來（0.0），
        # 否則「從沒接上」跟「沒回報」分不出來 —— golf 相機 404 就是這種
        cams = {c: round((cf.get(c, 0) - last["cam_frames"].get(c, 0)) / dt, 1)
                for c in (cameras or sorted(cf))}
        last.update(t=t, frames=frames, sync_ms=sync_ms, sync_n=sync_n, cam_frames=cf)
        return {
            "cameras": cams,
            "fps": round(d_frames / dt, 1),
            "dropped": int(stats.get("dropped", 0)),
            "sync_ms_avg": round(d_ms / d_n, 3) if d_n > 0 else None,
            "worker_queue": max((q.qsize() for q in queues), default=0) if queues else 0,
            "mask_err": int(stats.get("mask_err", 0)),
        }

    return provider


class LineReporter:
    def __init__(self, base_url: str | None, line_id: str, config_version: int, *,
                 interval: float = 15.0, post: Callable[[str, dict], None] = _post_json,
                 stats_provider: Callable[[], dict] | None = None) -> None:
        self.base_url = base_url
        self.line_id = line_id
        self.config_version = config_version
        self.interval = interval
        self.cameras_running = 0
        self.state, self.error = "ok", None    # 心跳送的狀態；看門狗會改（applying → ok / failed）
        self.gpu: Callable[[], tuple[int, int] | None] | None = None   # 每次心跳附上 GPU 剩餘記憶體
        # 設定有問題、被跳過的相機 → 原因（config.isolate_bad_cameras）；每次心跳附在 stats.skipped_cameras
        self.skipped_cameras: dict[str, str] = {}
        self._post = post
        # 每次回報時呼叫，回傳要附在 "stats" 的 dict；拋例外就不附，回報照發
        self.stats_provider = stats_provider
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def enabled(self) -> bool:
        return bool(self.base_url)      # file 模式沒有設定服務可回報，直接停用（不是錯誤）

    def set_state(self, state: str, error: str | None = None) -> None:
        self.state, self.error = state, error

    def start(self, cameras_running: int, state: str = "ok") -> None:
        self.cameras_running = cameras_running
        self.state, self.error = state, None
        if not self.enabled:
            logger.info("CONFIG_API 未設定，線路狀態回報停用")
            return
        self.send(state)
        self._thread = threading.Thread(target=self._loop, name="line-reporter", daemon=True)
        self._thread.start()
        logger.info("線路狀態回報已啟動（v%d，每 %.0fs）", self.config_version, self.interval)

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3)

    def report_failure(self, error: str) -> None:
        """載入設定失敗時呼叫：讓控制端看到「這條線套用失敗」而不是靜默沒動。"""
        self.send("failed", error[:500])

    def send(self, state: str, error: str | None = None) -> bool:
        if not self.enabled:
            return False
        payload = {"reported_version": self.config_version, "apply_state": state,
                   "apply_error": error, "cameras_running": self.cameras_running}
        if self.stats_provider is not None:
            try:
                payload["stats"] = self.stats_provider()
                cams = payload["stats"].get("cameras")
                if cams is not None:
                    payload["cameras_running"] = sum(1 for v in cams.values() if v > 0)
            except Exception:   # 統計算不出來不能擋住心跳
                logger.debug("stats_provider 失敗（已忽略）", exc_info=True)
        if self.skipped_cameras:
            payload.setdefault("stats", {})["skipped_cameras"] = self.skipped_cameras
        if self.gpu is not None:
            try:
                mem = self.gpu()
                if mem is not None:
                    payload.setdefault("stats", {}).update(gpu_free_mib=mem[0], gpu_total_mib=mem[1])
            except Exception:
                logger.debug("查 GPU 記憶體失敗（已忽略）", exc_info=True)
        try:
            self._post(status_url(self.base_url, self.line_id), payload)
            return True
        except Exception:
            logger.debug("線路狀態回報失敗（已忽略）", exc_info=True)
            return False

    def _loop(self) -> None:
        while not self._stop.wait(self.interval):
            self.send(self.state, self.error)
