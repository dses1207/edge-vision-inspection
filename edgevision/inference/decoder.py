"""解碼層：從 RTSP 取得影格。

職責邊界（刻意收窄）：
  ✅ 連線、解碼、產出影格與其時間戳
  ✅ 斷線重連、凍結偵測 —— 串流本身的韌性
  ❌ 不做跳幀決策（那是上層的事；舊的 throttle.py 已隨 run_mvp 一起移除）
  ❌ 不做推論、不碰業務邏輯

為何收窄：既有系統把解碼、跳幀、推論、ROI 過濾、業務指標寫在同一個
迴圈裡，換推論方式就得動到解碼。介面分離後，未來換 NVDEC 或 DeepStream
只需替換 Decoder 實作，上層不動。

⚠️ 韌性設計來自既有系統的實戰經驗，不可省略：
  1. -stimeout / timeout —— 缺了它，RTSP 一斷 ffmpeg 會無限阻塞在 read，
     行程還活著但一個位元組都不寫，「檢查行程存不存在」的健檢抓不到。
     曾造成多台相機同時卡死數小時。
  2. 凍結偵測 —— 上游 stall 時仍會送出重複幀，read timeout 抓不到
     （pipe 有資料），worker 會卡在同一畫面。用便宜的取樣雜湊比對。
  3. 指數退避 —— 「連得上但一直掉線」的相機若每次重連都歸零計數，
     退避形同虛設，會在相機端累積殘留 session。
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Iterator

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class Frame:
    """一個解碼後的影格及其身分資訊。

    rtp_timestamp 是前端幀級配對的鍵。取得方式依解碼後端而異，
    取不到時為 None，此時前端須退回 mediaTime 對齊（準確度較低）。
    """

    image: np.ndarray
    frame_id: str
    camera_id: str
    seq: int
    pts: int | None = None
    rtp_timestamp: int | None = None
    # 畫進影像的身分標記。由推論服務在打標時填入 —— 這是前端
    # 配對的鍵（時間戳跨路徑不可用，見 docs/findings/）。
    marker_id: int | None = None
    captured_at: float = field(default_factory=time.time)

    @property
    def shape(self) -> tuple[int, int]:
        h, w = self.image.shape[:2]
        return w, h


class Decoder(ABC):
    """解碼器介面。

    實作可替換：CPU 軟解（開發與 720p 生產）、NVDEC 硬解（高解析度或
    高路數時的備案）。上層只依賴這個介面。
    """

    @abstractmethod
    def frames(self) -> Iterator[Frame]:
        """產出影格。斷線時內部自行重連，呼叫端不需處理。"""

    @abstractmethod
    def close(self) -> None: ...

    @property
    @abstractmethod
    def is_connected(self) -> bool: ...


@dataclass(slots=True)
class DecoderStats:
    """解碼器的運行狀態，供健康檢查與診斷使用。"""

    frames_decoded: int = 0
    reconnects: int = 0
    freezes_detected: int = 0
    last_frame_at: float = 0.0
    connected_since: float = 0.0

    def seconds_since_last_frame(self) -> float:
        if self.last_frame_at == 0.0:
            return float("inf")
        return time.time() - self.last_frame_at


class BackoffPolicy:
    """指數退避。

    只有在串流「穩定運行一段時間後」才重置計數 —— 否則反覆掉線的相機
    每次重連都把計數歸零，退避永遠不會生效。
    """

    def __init__(
        self,
        initial: float = 2.0,
        maximum: float = 60.0,
        stable_reset_after: float = 10.0,
    ) -> None:
        self.initial = initial
        self.maximum = maximum
        self.stable_reset_after = stable_reset_after
        self._failures = 0

    def next_delay(self) -> float:
        delay = min(self.initial * (2 ** self._failures), self.maximum)
        self._failures += 1
        return delay

    def note_stable(self, connected_since: float) -> None:
        """串流已穩定運行足夠久 → 重置計數。"""
        if self._failures and (time.time() - connected_since) > self.stable_reset_after:
            self._failures = 0

    @property
    def failures(self) -> int:
        return self._failures


class FreezeDetector:
    """凍結偵測：畫面連續 N 秒完全沒變 → 判定上游 stall。

    用稀疏取樣的雜湊，成本極低（720p 約取 20×35 個像素）。
    比較的是解碼後的內容，因此能抓到「pipe 有資料但都是重複幀」
    這種 read timeout 抓不到的情況。
    """

    def __init__(self, threshold_seconds: float = 5.0) -> None:
        self.threshold = threshold_seconds
        self._signature: int | None = None
        self._unchanged_since: float = 0.0

    def check(self, image: np.ndarray) -> bool:
        """回傳 True 表示偵測到凍結，呼叫端應重連。"""
        try:
            sig = int(image[::37, ::37, 0].sum())
        except Exception:
            return False

        now = time.time()
        if sig != self._signature:
            self._signature = sig
            self._unchanged_since = now
            return False

        if self._unchanged_since and (now - self._unchanged_since) > self.threshold:
            self.reset()
            return True
        return False

    def reset(self) -> None:
        self._signature = None
        self._unchanged_since = 0.0
