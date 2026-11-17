"""事件匯流排：把判定結果發布到 Redis Streams，供多個消費者扇出。

為什麼要這層（相對於原本進程內的 queue.Queue）：
    原本 probe → worker 是「同進程、單一消費者」的交棒，queue.Queue 就夠。
    但通知功能要「一筆 NG 事件同時給多個獨立消費者」——
        · 前端 WebSocket 推播（全收）
        · NG 寄信（只收 ng）
        · DB/MinIO 落地（照抽樣）
    而且未來這些消費者會跨進程、甚至跨機器（上 K8s）。這已經超出
    進程內 queue 的能力，需要一個真正的訊息中介。

為什麼選 Redis Streams：
    · consumer group：每組消費者各自獨立進度 + ack，某組掛了重啟能補收
    · 持久化（appendonly）：broker 重啟不丟未消費訊息
    · 運維最輕、團隊最熟；路由需求（照相機/OK-NG）簡單，不需 NATS/Rabbit

⚠️ 設計約束（與 sink.py 同哲學）：發布【不可阻塞推論】。
    probe 是生死線。發布走 fire-and-forget，連不上就靜默降級成
    「只落地不通知」，絕不讓 Redis 抖動拖垮推論。

連線資訊只讀 REDIS_URL，Compose 與 K8s 共用同一份程式 —— 未來連
K8s 上獨立維護的 Redis，只改 REDIS_URL，程式不動。
"""

from __future__ import annotations

import json
import logging
import os
import time

from libs.contracts import FrameResult, VerdictStatus

logger = logging.getLogger(__name__)

# stream 內用來標示「這是什麼事件」的欄位值。消費者靠 type 分流：
#   mailer 只處理 type=ng；前端 type 全收。
EVENT_OK = "ok"
EVENT_NG = "ng"

# 前端即時監看用的獨立 stream。
#
# 為什麼跟 edgevision:events 分開：
#   edgevision:events 是「業務事件」——只帶路由指標(type/camera/frame_id)，
#   給通知、落地、統計這些消費者用，刻意不帶大 payload。
#   但前端畫框需要完整的 detections 座標，且要夠即時(查 DB 太慢)。
#
# 所以另開一條 live stream，帶完整結果，並用很小的 maxlen ——
# 它是「即時視窗」不是「事件紀錄」，過期的沒有意義，不需要留。
#
# ⚠️ 仍然【不傳影像幀】—— 只傳座標與判定，這條原則沒有放寬。
STREAM_LIVE_DEFAULT = "edgevision:live"

# 落地用的 stream（Event Processor 消費）。
#
# 為什麼不共用上面兩條：
#   · edgevision:events 刻意只帶路由指標，沒有 detections —— 寫不了 DB
#   · edgevision:live 帶完整結果，但 maxlen=500。8 路 15fps 時 500 筆只涵蓋【4.2 秒】
#     （實測：1789001912158 → 1789001916364）。落地行程重啟一次就漏資料。
#
# 所以落地要自己一條：帶完整 payload、保留夠久，並用 consumer group + ack，
# 確保「處理過才算數」，重啟能從斷點續傳。
STREAM_STORE_DEFAULT = "edgevision:store"


def result_to_event(result: FrameResult) -> dict[str, str]:
    """FrameResult → Redis Stream 欄位（全部字串，Streams 只吃 flat map）。

    只放「通知/路由需要」的欄位，不放整包 detections（那是落地層的事）。
    下游要細節可用 frame_id 回查 DB。
    """
    v = result.verdict
    if v is not None and v.status == VerdictStatus.NG:
        etype = EVENT_NG
    else:
        etype = EVENT_OK
    return {
        "type": etype,
        "camera_id": result.camera_id or "",
        "model_name": result.model_name or "",
        "frame_id": str(result.frame_id or ""),
        "reason": (v.reason if v else "") or "",
        "details": json.dumps(v.details, ensure_ascii=False) if v and v.details else "{}",
        "ts": f"{time.time():.3f}",
    }


class EventBus:
    """Redis Streams 事件發布器。

    連不上 Redis 時，enabled=False，publish() 變 no-op（靜默降級）——
    符合「通知掛掉不該影響產線」。
    """

    def __init__(
        self,
        url: str | None = None,
        *,
        stream: str | None = None,
        maxlen: int = 100_000,
    ) -> None:
        self.url = url or os.environ.get("REDIS_URL", "")
        self.stream = stream or os.environ.get("REDIS_STREAM", "edgevision:events")
        # maxlen：Stream 近似上限，避免無限成長吃爆記憶體（近似修剪，效能好）
        self.maxlen = maxlen
        # 即時推播 stream：只保留最近數百筆（前端只在乎「現在」）
        self.live_stream = os.environ.get("REDIS_STREAM_LIVE", STREAM_LIVE_DEFAULT)
        self.live_maxlen = int(os.environ.get("REDIS_LIVE_MAXLEN", 500))
        # 落地 stream：預設留 20 萬筆。8 路 15fps ≈ 120 筆/秒 → 約 28 分鐘的緩衝，
        # 足以撐過落地行程重啟或短暫的 DB 抖動。設成 0 可停用（不發布）。
        self.store_stream = os.environ.get("REDIS_STREAM_STORE", STREAM_STORE_DEFAULT)
        self.store_maxlen = int(os.environ.get("REDIS_STORE_MAXLEN", 200_000))
        self._redis = None
        self.enabled = False

        if not self.url:
            logger.warning("未設定 REDIS_URL，事件匯流排停用（只落地不通知）")
            return
        try:
            import redis  # redis-py

            # socket timeout：連線/讀寫都設短超時，Redis 卡住不拖垮呼叫端
            self._redis = redis.Redis.from_url(
                self.url,
                socket_connect_timeout=2,
                socket_timeout=2,
                decode_responses=True,
            )
            self.enabled = True
            logger.info("事件匯流排已連線（stream=%s）", self.stream)
        except ImportError:
            logger.warning("未安裝 redis 套件，事件匯流排停用：pip install redis")
        except Exception:
            logger.warning("Redis 初始化失敗，事件匯流排停用", exc_info=True)

    def ping(self) -> bool:
        if not self.enabled or self._redis is None:
            return False
        try:
            return bool(self._redis.ping())
        except Exception:
            return False

    def publish(self, result: FrameResult) -> bool:
        """發布一筆結果事件。fire-and-forget：失敗回 False 但絕不拋出。"""
        if not self.enabled or self._redis is None:
            return False
        try:
            self._redis.xadd(
                self.stream,
                result_to_event(result),
                maxlen=self.maxlen,
                approximate=True,
            )
            # 即時推播：完整結果給前端畫框（獨立 stream、極短保留）
            # 帶 type 欄位：前端不必靠「有沒有某個 key」判別訊息種類
            import json as _json
            _live = _json.loads(result.model_dump_json())
            _live["type"] = "result"
            _live_json = _json.dumps(_live, ensure_ascii=False)
            self._redis.xadd(
                self.live_stream,
                {"payload": _live_json},
                maxlen=self.live_maxlen,
                approximate=True,
            )
            # 落地：同一份 payload，但保留久得多，且由 consumer group 負責 ack。
            # 只有真的有偵測到東西才發 —— 空幀寫進 DB 沒有意義（insert_detections 也會直接回 0）。
            if self.store_maxlen > 0 and result.detections:
                self._redis.xadd(
                    self.store_stream,
                    {"payload": _live_json},
                    maxlen=self.store_maxlen,
                    approximate=True,
                )
            return True
        except Exception:
            # 發布失敗不影響推論/落地，只記一筆
            logger.debug("事件發布失敗（已忽略）", exc_info=True)
            return False

    def close(self) -> None:
        if self._redis is not None:
            try:
                self._redis.close()
            except Exception:
                pass
