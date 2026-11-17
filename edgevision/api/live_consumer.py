"""即時結果消費者：Redis live stream → WebSocket 廣播。

⭐ 這是 API 抽離之後的關鍵一環。

  同進程模式（舊的 run_mvp，已移除）：推論服務直接呼叫 hub.publish()，不需要本模組。
  獨立模式（output-api）：推論在別的容器，結果經 Redis 過來，
                          由本模組讀出來再廣播給前端。

  兩種模式前端完全無感 —— 收到的 WebSocket 訊息格式一樣。

⚠️ 只讀 live stream（edgevision:live），不碰 edgevision:events：
   前者是「即時視窗」，保留數百筆、可隨意丟棄；
   後者是「業務事件」，有 consumer group 與 ack，給通知/落地用。
   兩者混用會讓前端斷線重連時把積壓的舊事件全補畫一遍。
"""

from __future__ import annotations

import asyncio
import logging
import os

logger = logging.getLogger(__name__)


class LiveConsumer:
    """背景讀 Redis live stream，把結果餵給 WebSocketHub。"""

    def __init__(self, hub, *, url: str | None = None, stream: str | None = None) -> None:
        self.hub = hub
        self.url = url or os.environ.get("REDIS_URL", "")
        self.stream = stream or os.environ.get("REDIS_STREAM_LIVE", "edgevision:live")
        self._task: asyncio.Task | None = None
        self._redis = None

    async def start(self) -> bool:
        if not self.url:
            logger.info("未設定 REDIS_URL，即時推播消費者停用")
            return False
        try:
            import redis.asyncio as aioredis

            self._redis = aioredis.from_url(
                self.url, socket_connect_timeout=3, decode_responses=True
            )
            await self._redis.ping()
        except Exception:
            logger.warning("即時推播消費者連線失敗，停用", exc_info=True)
            self._redis = None
            return False
        self._task = asyncio.create_task(self._loop(), name="live-consumer")
        logger.info("即時推播消費者已啟動（stream=%s）", self.stream)
        return True

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        if self._redis is not None:
            await self._redis.aclose()

    async def _loop(self) -> None:
        # 從「現在」開始讀，不補舊的：前端要的是當下畫面對應的框，
        # 補畫幾秒前的結果只會讓框錯位。
        last_id = "$"
        while True:
            try:
                resp = await self._redis.xread({self.stream: last_id}, count=50, block=2000)
                if not resp:
                    continue
                for _stream, entries in resp:
                    for entry_id, fields in entries:
                        last_id = entry_id
                        payload = fields.get("payload")
                        if payload:
                            # ⚠️ 必須 await，不能 create_task：
                            # create_task 會讓多筆結果併發送出，順序無保證，
                            # 且慢客戶端會讓 task 無限累積 —— 實測造成
                            # WebSocket 端間隔抖到 1.7s（Redis 端只有 370ms），
                            # 超過前端 maxAgeMs=700ms 就沒框可畫 → 框閃爍。
                            await self._broadcast(payload)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.warning("即時推播讀取失敗，2 秒後重試", exc_info=True)
                await asyncio.sleep(2)

    async def _broadcast(self, payload: str) -> None:
        """把結果送進 hub。

        ref_quad 不在這裡逐筆注入 —— 它是靜態設定，連線時送一次就夠
        （hub.connect 已經會送 ref_quads）。逐筆注入要 JSON 解析 +
        重新序列化，在 5fps × N 相機的節奏下是白費的 CPU。
        """
        if not self.hub.client_count:
            return
        await self.hub.broadcast_raw(payload)
