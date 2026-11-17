"""結果發布：FrameResult → Redis Streams。格式與舊 edgevision/storage/events.py 的 EventBus **完全相同**，
既有的消費者（output-api 的 live_consumer、通知、落地）不用改。

三條 stream：
  edgevision:events  業務事件，flat 欄位（type / line_id / station / camera_id / model_name / frame_id / reason / details / ts），不帶 detections
                ⚠️ 2026-09-24 起【只發 NG】。OK（含「冷卻中」「畫面無板」）不進這條 —— 通知消費者只關心 NG，
                每幀都發的話 8 路 15fps 是 120 筆/秒、maxlen 10 萬只夠 14 分鐘；只發 NG 之後同樣的量能撐好幾天。
                查過沒有人依賴 OK：recorder 只吃 ng、落地走 store、前端即時走 live、歷史查 DB。
                line_id / station 是給通知按線分組、按站別分類用的；runner 本來就知道，不必拿 camera_id 反查。
  edgevision:live    即時推播，一個 payload 欄位 = 整個 FrameResult JSON + "type": "result"，maxlen 很小
  edgevision:store   落地用，payload 同 live 但保留 20 萬筆，由 event-processor 以 consumer group 消費

⚠️ 為什麼落地要獨立一條，不共用前兩條：
  events 不帶 detections → 寫不了 DB；
  live 的 maxlen=500，實測 8 路 15fps 時只涵蓋 4.2 秒 → 落地行程重啟一次就漏資料。

fire-and-forget：Redis 掛了只記 log，絕不拖垮推論。連線資訊只讀 REDIS_URL。
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from typing import Any

from edgevision_contracts import FrameResult, VerdictStatus

logger = logging.getLogger(__name__)

Publisher = Callable[[FrameResult, Any], None]


def is_ng(result: FrameResult) -> bool:
    v = result.verdict
    return v is not None and v.status == VerdictStatus.NG


def result_to_event(result: FrameResult, *, line_id: str = "", station: str = "") -> dict[str, str]:
    etype = "ng" if is_ng(result) else "ok"
    v = result.verdict
    return {
        "type": etype,
        "line_id": line_id,
        "station": station,
        "camera_id": result.camera_id or "",
        "model_name": result.model_name or "",
        "frame_id": str(result.frame_id or ""),
        "reason": (v.reason if v else "") or "",
        "details": json.dumps(v.details, ensure_ascii=False) if v and v.details else "{}",
        "ts": f"{time.time():.3f}",
    }


def result_to_live_payload(result: FrameResult) -> str:
    live = json.loads(result.model_dump_json())
    live["type"] = "result"
    return json.dumps(live, ensure_ascii=False)


class RedisPublisher:
    def __init__(self, url: str, *, stream: str = "edgevision:events", live_stream: str = "edgevision:live",
                 store_stream: str = "edgevision:store", maxlen: int = 100_000, live_maxlen: int = 500,
                 store_maxlen: int = 200_000, client: Any = None,
                 line_id: str = "", station: str = "") -> None:
        self.stream, self.live_stream, self.store_stream = stream, live_stream, store_stream
        self.maxlen, self.live_maxlen, self.store_maxlen = maxlen, live_maxlen, store_maxlen
        self.line_id, self.station = line_id, station
        self._redis = client
        if self._redis is None:
            import redis
            self._redis = redis.Redis.from_url(url, socket_connect_timeout=2, socket_timeout=2, decode_responses=True)
        self.published = 0
        self.failed = 0

    def ping(self) -> bool:
        try:
            return bool(self._redis.ping())
        except Exception:   # noqa: BLE001 —— ping 只是診斷，任何錯都當「沒連上」
            return False

    def __call__(self, result: FrameResult, image: Any = None) -> None:
        try:
            # 業務事件只發 NG（理由見檔頭）。OK 幀仍走 live / store。
            if is_ng(result):
                self._redis.xadd(self.stream,
                                 result_to_event(result, line_id=self.line_id,
                                                 station=result.station or self.station),
                                 maxlen=self.maxlen, approximate=True)
            payload = result_to_live_payload(result)
            self._redis.xadd(self.live_stream, {"payload": payload},
                             maxlen=self.live_maxlen, approximate=True)
            # 落地：有偵測物件，或判定是 NG 才發。空的 OK 幀寫進 DB 沒意義。
            # ⚠️ NG 不能只看有沒有框：靠時間判的 NG（「A 之後 N 秒內沒看到 B」這類逾時）當下沒有框，
            #    以前因此不進資料庫 —— 前端查不到、錄好的影片也掛不上去。DB 端寫成一列整幀判定。
            if self.store_maxlen > 0 and (result.detections or is_ng(result)):
                self._redis.xadd(self.store_stream, {"payload": payload},
                                 maxlen=self.store_maxlen, approximate=True)
            self.published += 1
        except Exception:
            self.failed += 1
            if self.failed in (1, 10, 100) or self.failed % 1000 == 0:
                logger.warning("Redis 發布失敗（累計 %d 次，已忽略）", self.failed, exc_info=True)


def log_publisher(result: FrameResult, image: Any = None) -> None:
    if result.verdict and result.verdict.status == VerdictStatus.NG:
        logger.warning("[%s frame %s] NG: %s %s", result.camera_id, result.frame_id,
                       result.verdict.reason, result.verdict.details)


class FilePublisher:
    """影片測試（VIDEO_FILE 模式）：每幀結果寫一行 JSON 到檔案，不送 Redis —— 測試結果不能混進線上的通知與報表。

    t = 從第一筆結果起算的秒數：有 pts（影片的 buf_pts）就用它 = 影片時間，不管照實際速度播還是不限速；
    沒有才退回牆鐘。
    mask_contour 不寫（每幀上百點，檔案會大到沒法看）；框、類別、信心、判定都有。
    """

    def __init__(self, path: str) -> None:
        import threading

        self.path = path
        self._fh = open(path, "w", encoding="utf-8")      # noqa: SIM115 —— 程序結束時 close()
        self._lock = threading.Lock()
        self._t0: float | None = None
        self.written = 0

    def __call__(self, result: FrameResult, image: Any = None) -> None:
        d = result.model_dump(mode="json", exclude={"detections": {"__all__": {"mask_contour", "keypoints"}}})
        now = result.pts / 1e9 if result.pts is not None else result.timestamp
        with self._lock:
            if self._t0 is None:
                self._t0 = now
            d["t"] = round(now - self._t0, 3)
            self._fh.write(json.dumps(d, ensure_ascii=False) + "\n")
            self.written += 1

    def close(self) -> None:
        with self._lock:
            self._fh.close()


def make_publisher(env: dict[str, str], *, line_id: str | None = None, station: str = "") -> Publisher:
    """RESULTS_FILE 有設就寫檔（影片測試）；否則 REDIS_URL 有設就發 Redis（連不上也只降級成 log），沒設就只印 log。

    line_id / station 會寫進每筆 NG 事件；line_id 沒給就用環境變數 LINE_ID。
    """
    if env.get("RESULTS_FILE"):
        logger.info("結果寫檔 → %s（影片測試，不送 Redis）", env["RESULTS_FILE"])
        return FilePublisher(env["RESULTS_FILE"])
    url = env.get("REDIS_URL", "")
    if not url:
        logger.warning("未設定 REDIS_URL，結果只印 log（不通知、不落地）")
        return log_publisher
    try:
        pub = RedisPublisher(url, stream=env.get("REDIS_STREAM", "edgevision:events"),
                             line_id=line_id if line_id is not None else env.get("LINE_ID", ""),
                             station=station,
                             live_stream=env.get("REDIS_STREAM_LIVE", "edgevision:live"),
                             live_maxlen=int(env.get("REDIS_LIVE_MAXLEN", 500)),
                             store_stream=env.get("REDIS_STREAM_STORE", "edgevision:store"),
                             store_maxlen=int(env.get("REDIS_STORE_MAXLEN", 200_000)))
    except Exception:
        logger.warning("Redis 初始化失敗，結果只印 log", exc_info=True)
        return log_publisher
    logger.info("結果發布 → Redis %s（%s）", "已連線" if pub.ping() else "未連上，將持續重試", env.get("REDIS_STREAM", "edgevision:events"))
    return pub
