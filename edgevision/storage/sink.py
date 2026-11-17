"""結果落地：把偵測結果寫入 PostgreSQL，缺陷圖存入 MinIO。

⚠️ 設計約束：落地【不可阻塞推論】。
    推論以固定節奏產出結果，但寫 DB 與上傳物件都是 I/O 操作，
    延遲不可預測。若同步執行，一次網路抖動就會讓推論掉幀。
    因此採背景執行緒 + 有界佇列：滿了就丟，絕不回壓。

    這符合原則 1（爆炸半徑）—— 儲存出問題不該影響產線推論。

儲存分層：
    事件層（PG）    每次偵測的元資料，長期保留
    證據層（MinIO）缺陷圖，中期保留
    原始層（MinIO）策略性抽樣的原始影像

⚠️ 抽樣策略是關鍵：工業場景良率高，絕大多數影像無缺陷。
   隨機抽樣會存下一堆無用資料，同時漏掉真正有價值的難例。
"""

from __future__ import annotations

import io
import logging
import queue
import random
import threading
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from libs.contracts import FrameResult, VerdictStatus

# 遮罩繪製來自 ai-infer 的 edgevision_postprocess.render（推論那邊算出多邊形，這裡只負責畫）。
# edgevision 的執行環境不一定裝得到那個套件 —— 沒有就退化成「存原圖」，
# 絕不能因為畫不了圖就不存證據。
try:
    from edgevision_postprocess.render import draw_detections
except ImportError:  # pragma: no cover - 取決於部署環境是否併裝 ai-infer
    draw_detections = None

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class SinkStats:
    events_written: int = 0
    images_stored: int = 0
    dropped: int = 0
    errors: int = 0
    started_at: float = field(default_factory=time.time)

    @property
    def drop_ratio(self) -> float:
        total = self.events_written + self.dropped
        return 0.0 if total == 0 else self.dropped / total


def is_ng(result: FrameResult) -> bool:
    """只有 NG 進資料庫（含存證圖）。

    ⚠️ 以前還有「正常幀隨機抽 0.5%」與「低信心抽樣」（SamplingPolicy）。2026-10-05 拿掉：
       OK 幀的抽樣紀錄一天約 20 萬列、佔資料庫九成以上，沒有人查；
       要蒐集訓練背景素材另走資料集流程（LakeFS），不靠這張表。
    """
    return result.verdict is not None and result.verdict.status == VerdictStatus.NG


class ResultSink:
    """結果落地器。背景執行緒消化佇列，主執行緒只負責投遞。"""

    def __init__(
        self,
        *,
        db,                       # edgevision.storage.db.Database
        objects,                  # edgevision.storage.objects.ObjectStore
        queue_size: int = 200,
        jpeg_quality: int = 85,
        draw_overlay: bool = True,
    ) -> None:
        self.db = db
        self.objects = objects
        self.jpeg_quality = jpeg_quality
        self.draw_overlay = draw_overlay
        if draw_overlay and draw_detections is None:
            logger.warning("edgevision_postprocess 不可用，存檔將不畫遮罩與框（存原圖）")

        self._q: queue.Queue[tuple[FrameResult, np.ndarray | None] | None] = queue.Queue(
            maxsize=queue_size
        )
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.stats = SinkStats()

    # ── 生命週期 ──────────────────────────────────────────

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, name="result-sink", daemon=True
        )
        self._thread.start()
        logger.info("結果落地器已啟動（只存 NG）")

    def stop(self, timeout: float = 10.0) -> None:
        self._stop.set()
        try:
            self._q.put_nowait(None)
        except queue.Full:
            pass
        if self._thread:
            self._thread.join(timeout=timeout)
        logger.info("結果落地器已停止（寫入 %d 筆、存圖 %d 張、丟棄 %d）",
                    self.stats.events_written, self.stats.images_stored,
                    self.stats.dropped)

    # ── 對外介面（由推論執行緒呼叫）──────────────────────

    def submit(self, result: FrameResult, image: np.ndarray | None = None) -> bool:
        """投遞一筆結果。佇列滿時丟棄並回 False —— 絕不阻塞推論。"""
        if self._stop.is_set():
            return False
        try:
            # 影像要複製：呼叫端會繼續使用同一個緩衝區
            self._q.put_nowait((result, None if image is None else image.copy()))
            return True
        except queue.Full:
            self.stats.dropped += 1
            return False

    def health(self) -> dict:
        return {
            "events_written": self.stats.events_written,
            "images_stored": self.stats.images_stored,
            "dropped": self.stats.dropped,
            "errors": self.stats.errors,
            "drop_ratio": round(self.stats.drop_ratio, 4),
            "queue_depth": self._q.qsize(),
        }

    # ── 內部 ──────────────────────────────────────────────

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                item = self._q.get(timeout=1.0)
            except queue.Empty:
                continue
            if item is None:
                return
            try:
                self._process(*item)
            except Exception:
                self.stats.errors += 1
                logger.exception("結果落地失敗")

    def _process(self, result: FrameResult, image: np.ndarray | None) -> None:
        if not is_ng(result):
            return                              # OK 幀不進資料庫（見 is_ng）
        image_uri = self._store_image(result, image, "ng") if image is not None else None
        self.db.insert_detections(result, image_uri=image_uri)
        self.stats.events_written += 1

    def _store_image(
        self, result: FrameResult, image: np.ndarray, reason: str
    ) -> str | None:
        """把影像存進 MinIO。

        ⚠️ 先存圖再寫 DB。理由：
            孤兒圖片（有圖沒紀錄）→ 可容忍，定期清理即可
            死連結（有紀錄沒圖）  → 使用者點下載會失敗，不可接受
        選一個「失敗時比較不痛」的順序。
        """
        # 把偵測畫上去再存。直播只能燒錄框（mask 是 worker 非同步算的，追不上 nvdsosd），
        # 存檔沒有這個時序限制 —— 多邊形早就算好在 result.detections 裡了。
        canvas = image
        if self.draw_overlay and draw_detections is not None and result.detections:
            try:
                canvas = draw_detections(image, result.detections)
            except Exception:
                # 畫失敗就存原圖：有圖可看 > 沒有證據
                logger.exception("繪製遮罩失敗，改存原圖：%s", result.frame_id)
                canvas = image

        ok, buf = cv2.imencode(
            ".jpg", canvas, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality]
        )
        if not ok:
            logger.warning("影像編碼失敗：%s", result.frame_id)
            return None

        from datetime import datetime, timezone
        dt = datetime.fromtimestamp(result.timestamp, tz=timezone.utc)
        key = (
            f"defects/{dt:%Y/%m/%d}/{result.camera_id}/"
            f"{dt:%H%M%S}_{result.marker_id or result.frame_id}_{reason}.jpg"
        )

        try:
            self.objects.put(
                bucket="inference-output",
                key=key,
                data=io.BytesIO(buf.tobytes()),
                length=len(buf),
                content_type="image/jpeg",
            )
            self.stats.images_stored += 1
            return f"s3://inference-output/{key}"
        except Exception:
            logger.exception("上傳影像失敗：%s", key)
            return None
