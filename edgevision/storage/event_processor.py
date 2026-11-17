"""Event Processor：Redis `edgevision:store` → PostgreSQL。

⭐ 為什麼需要它

    runner 只負責推論與發布，不碰資料庫 —— 那是刻意的（爆炸半徑：
    DB 抖動絕不能拖垮 GPU 管線）。但這也代表【一定要有人消費】，
    否則結果只會堆在 Redis 裡。

    實際發生過：2026-09-03 15:31 之後 DB 六天沒有新資料，
    而 Redis 積了 10 萬筆事件 —— 推論一直在跑，只是沒人落地。

⚠️ 為什麼讀 edgevision:store 而不是另外兩條

    edgevision:events  只帶路由指標（type/camera/frame_id），沒有 detections → 寫不了 DB
    edgevision:live    帶完整結果但 maxlen=500。實測 8 路 15fps 時 500 筆只涵蓋 4.2 秒，
                  本行程重啟一次就漏資料
    edgevision:store   帶完整 payload、保留 20 萬筆（約 28 分鐘），consumer group + ack

⭐ 只存 NG（2026-10-05）

    8 路 × 15fps = 120 幀/秒，全寫一天 2600 萬列，PG 撐不住。以前是「NG 全存 + 正常幀抽 0.5%」，
    抽樣那部分一天仍有約 20 萬列、佔資料庫九成以上而且沒人查 —— 拿掉了，只存 NG
    （含沒有框的整幀判定 NG，見 Database.insert_detections）。

環境變數：
    REDIS_URL              必填，沒有就不啟動
    REDIS_STREAM_STORE     預設 edgevision:store
    STORE_GROUP            consumer group 名，預設 store
    STORE_CONSUMER         本副本名，預設主機名（多副本要各自不同）
    STORE_BATCH            一次讀幾筆，預設 200
    STORE_BLOCK_MS         沒資料時阻塞多久，預設 2000
    STATS_INTERVAL_S       統計輸出間隔，預設 60；0=關閉
"""

from __future__ import annotations

import json
import logging
import os
import socket
import threading
import time
from dataclasses import dataclass, field

from libs.contracts import Detection, FrameResult, Verdict

from edgevision.storage.events import STREAM_STORE_DEFAULT
from edgevision.storage.sink import is_ng

logger = logging.getLogger(__name__)


# runner 發佈時加在 payload 外層的傳輸標記，不是契約欄位（ai-infer ds_runner/publish.py：
# live / store 共用同一個 payload，帶 "type": "result" 給前端分辨訊息種類）
ENVELOPE_KEYS = frozenset({"type"})


def unknown_fields(data: dict) -> set[str]:
    """payload 裡契約不認識的欄位（含 detections[*] 與 verdict 底下）。純函式，可測。

    ⚠️ 為什麼要查：Pydantic 預設會把不認識的欄位【直接丟掉】，不報錯。
       runner 用較新的 edgevision-contracts、這裡還沒升版時，新欄位就這樣消失 ——
       keypoints、station 都這樣丟過。查到就寫警告，讓版本落差被看見（資料照常寫入）。
    """
    out = {k for k in data if k not in FrameResult.model_fields and k not in ENVELOPE_KEYS}
    for d in data.get("detections") or []:
        if isinstance(d, dict):
            out |= {f"detections.{k}" for k in d if k not in Detection.model_fields}
    v = data.get("verdict")
    if isinstance(v, dict):
        out |= {f"verdict.{k}" for k in v if k not in Verdict.model_fields}
    return out


@dataclass(slots=True)
class ProcessorStats:
    read: int = 0            # 從 stream 讀出來幾筆
    written: int = 0         # 實際寫進 DB 幾列
    skipped: int = 0         # 抽樣決定不寫
    errors: int = 0          # 解析或寫入失敗
    started_at: float = field(default_factory=time.time)

    def snapshot(self) -> dict:
        return {
            "read": self.read,
            "written": self.written,
            "skipped": self.skipped,
            "errors": self.errors,
            "uptime_s": round(time.time() - self.started_at),
        }


class EventProcessor:
    """消費 store stream，把結果寫進 PG。

    ⚠️ ack 的時機：**寫成功之後才 ack**。
       先 ack 再寫的話，寫失敗那筆就永久消失了 —— Redis 已經認為處理完畢。
       失敗不 ack，訊息留在 pending list，下次重啟或別的副本會接手。
    """

    def __init__(
        self,
        *,
        db,                                 # edgevision.storage.db.Database
        redis_url: str | None = None,
        stream: str | None = None,
        group: str | None = None,
        consumer: str | None = None,
        batch: int | None = None,
        block_ms: int | None = None,
    ) -> None:
        self.db = db
        self.url = redis_url or os.environ.get("REDIS_URL", "")
        self.stream = stream or os.environ.get("REDIS_STREAM_STORE", STREAM_STORE_DEFAULT)
        self.group = group or os.environ.get("STORE_GROUP", "store")
        # 多副本時每個 consumer 名字必須不同，否則 pending list 會互相搶。
        # 用主機名（K8s 的 Pod 名）是最省事且天然唯一的選擇。
        self.consumer = consumer or os.environ.get("STORE_CONSUMER") or socket.gethostname()
        self.batch = batch if batch is not None else int(os.environ.get("STORE_BATCH", 200))
        self.block_ms = block_ms if block_ms is not None else int(os.environ.get("STORE_BLOCK_MS", 2000))

        self._redis = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.stats = ProcessorStats()
        self._warned_fields: set[str] = set()   # 已警告過的陌生欄位（見 _warn_unknown）
        self._last_write_at = time.time()      # 給心跳報「多久沒寫 DB」
        self._heartbeat = None

    # ── 連線 ──────────────────────────────────────────────

    def connect(self) -> bool:
        """建立連線並確保 consumer group 存在。失敗回 False，不拋。"""
        if not self.url:
            logger.error("未設定 REDIS_URL —— 落地無法啟動")
            return False
        try:
            import redis

            self._redis = redis.Redis.from_url(
                self.url,
                socket_connect_timeout=5,
                # 讀取要能撐過 block_ms，否則每次阻塞讀都會超時
                socket_timeout=self.block_ms / 1000 + 5,
                decode_responses=True,
            )
            self._redis.ping()
        except Exception:
            logger.exception("Redis 連線失敗")
            return False

        # mkstream=True：stream 還不存在時一併建立，免得首次啟動時
        # 因為 runner 還沒發過任何一筆而拿到 NOGROUP。
        # id="0"：從頭開始消費（積壓的也要處理）。
        try:
            self._redis.xgroup_create(self.stream, self.group, id="0", mkstream=True)
            logger.info("已建立 consumer group %s（從頭消費）", self.group)
        except Exception as exc:
            if "BUSYGROUP" not in str(exc):
                logger.exception("建立 consumer group 失敗")
                return False
            logger.info("consumer group %s 已存在，從上次進度接續", self.group)
        return True

    # ── 單筆處理 ──────────────────────────────────────────

    def _should_store(self, result: FrameResult) -> bool:
        return is_ng(result)

    def _warn_unknown(self, payload: str) -> None:
        """同一組陌生欄位只警告一次（每筆都印會洗版）。"""
        try:
            extra = unknown_fields(json.loads(payload))
        except Exception:
            return
        new = extra - self._warned_fields
        if new:
            self._warned_fields |= new
            logger.warning("payload 有契約不認識的欄位 %s —— 這些資料不會被存下來；"
                           "runner 的 edgevision-contracts 比這裡新，請升 requirements.txt 的版號", sorted(new))

    def handle(self, payload: str) -> int:
        """處理一筆 payload，回傳寫入列數。解析失敗拋出，由呼叫端計數且不 ack。"""
        result = FrameResult.model_validate_json(payload)
        self._warn_unknown(payload)
        if not self._should_store(result):
            self.stats.skipped += 1
            return 0
        n = self.db.insert_detections(result)
        self.stats.written += n
        self._last_write_at = time.time()
        return n

    # ── 主迴圈 ────────────────────────────────────────────

    def run_once(self) -> int:
        """讀一批、處理、ack。回傳這批處理了幾筆（0 = 沒資料）。"""
        assert self._redis is not None
        resp = self._redis.xreadgroup(
            self.group, self.consumer,
            {self.stream: ">"},
            count=self.batch, block=self.block_ms,
        )
        if not resp:
            return 0

        ack_ids: list[str] = []
        for _stream, messages in resp:
            for msg_id, fields in messages:
                self.stats.read += 1
                payload = fields.get("payload")
                if not payload:
                    # 格式不對的訊息：ack 掉，否則它會永遠卡在 pending list 裡重試
                    ack_ids.append(msg_id)
                    self.stats.errors += 1
                    continue
                try:
                    self.handle(payload)
                    ack_ids.append(msg_id)          # ⚠️ 寫成功才 ack
                except Exception:
                    self.stats.errors += 1
                    logger.exception("處理失敗（不 ack，留給下次重試）id=%s", msg_id)

        if ack_ids:
            try:
                self._redis.xack(self.stream, self.group, *ack_ids)
            except Exception:
                logger.exception("ack 失敗 —— 這批會被重複處理")
        return sum(len(m) for _s, m in resp)

    def _metrics(self) -> dict:
        """給心跳的工作指標。⚠️ 在心跳執行緒上跑，要輕 —— 只讀計數器和一次 XLEN。"""
        lag = None
        try:
            if self._redis is not None:
                # stream 總長度減去這個 group 已讀到的位置 ≈ 積壓。
                # XINFO GROUPS 的 lag 欄位（Redis 7+）直接給這個數。
                for g in self._redis.xinfo_groups(self.stream):
                    if g.get("name") == self.group:
                        lag = g.get("lag")
                        break
        except Exception:  # noqa: BLE001 —— 指標拿不到不能讓心跳失敗
            pass
        return {
            "lag": lag,
            "last_write_s": round(time.time() - self._last_write_at),
            "read": self.stats.read, "written": self.stats.written,
            "skipped": self.stats.skipped, "errors": self.stats.errors,
        }

    def run(self) -> None:
        """阻塞式主迴圈，直到 stop()。"""
        # ⭐ 心跳：讓 /healthz 看得到「這個消費者在做事」，不只是「活著」
        from edgevision.storage.heartbeat import Heartbeat
        self._heartbeat = Heartbeat(self.db, service="event-processor")
        self._heartbeat.start(self._metrics)

        stats_every = int(os.environ.get("STATS_INTERVAL_S", 60))
        last_stats = time.time()
        # 連續失敗時退避，避免 DB 掛掉時瘋狂重試打爆 log
        backoff = 0.0
        while not self._stop.is_set():
            try:
                self.run_once()
                backoff = 0.0
            except Exception:
                backoff = min(30.0, (backoff or 1.0) * 2)
                logger.exception("讀取迴圈異常，%.0fs 後重試", backoff)
                self._stop.wait(backoff)
            if stats_every > 0 and time.time() - last_stats >= stats_every:
                logger.info("落地統計 %s", self.stats.snapshot())
                last_stats = time.time()
        if self._heartbeat:
            self._heartbeat.stop()

    # ── 背景執行緒模式（給 output-api 內嵌用）──────────────

    def start(self) -> bool:
        if not self.connect():
            return False
        self._stop.clear()
        self._thread = threading.Thread(target=self.run, name="event-processor", daemon=True)
        self._thread.start()
        logger.info("落地已啟動：%s → PG（group=%s consumer=%s，只存 NG）",
                    self.stream, self.group, self.consumer)
        return True

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=timeout)
        if self._redis is not None:
            try:
                self._redis.close()
            except Exception:
                pass
