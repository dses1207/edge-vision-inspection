"""偵測結果的資料契約。

這是全系統的地基：推論服務產出、事件處理消費、前端疊框、回流訓練
全部依賴這份 schema。改動前先想清楚下游有誰。

設計要點：
  ・camera_id 必填 —— 不可依賴「哪個 Pod 送來的」推斷相機身分，
    否則未來一個 Pod 服務多路時整個下游要重寫。
  ・rtp_timestamp —— 前端幀級配對用。影像與結果走兩條路徑，
    延遲不同且浮動，只能靠幀身分配對，不能靠抵達時間。
  ・coord_system + image_meta —— 座標系必須顯式記錄。YOLO 推論經過
    letterbox（等比縮放+補邊），不記錄縮放參數，日後回流訓練時
    無法還原框在原圖的位置。
"""

from __future__ import annotations

import time
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field

SCHEMA_VERSION = "1.0"


class CoordSystem(str, Enum):
    """bbox / 輪廓座標所處的座標系。"""

    ORIGINAL = "original"      # 原始影格像素座標
    LETTERBOXED = "letterboxed"  # 模型輸入座標（含縮放與補邊）


class LetterboxParams(BaseModel):
    """letterbox 變換參數，用於在兩個座標系之間換算。

    original -> letterboxed:  (x - offset_x) * scale + pad_x
    letterboxed -> original:  (x - pad_x) / scale + offset_x

    offset_x / offset_y：模型只看原圖的一個 ROI 時（nvdspreprocess process-on-roi），
    ROI 左上角在原圖的像素位置。全幅推論時為 0，公式退化成原本的樣子。
    """

    scale: float
    pad_x: float = 0.0
    pad_y: float = 0.0
    offset_x: float = 0.0
    offset_y: float = 0.0

    def to_original(self, x: float, y: float) -> tuple[float, float]:
        return (x - self.pad_x) / self.scale + self.offset_x, (y - self.pad_y) / self.scale + self.offset_y

    def to_letterboxed(self, x: float, y: float) -> tuple[float, float]:
        return (x - self.offset_x) * self.scale + self.pad_x, (y - self.offset_y) * self.scale + self.pad_y


class ImageMeta(BaseModel):
    """影格的尺寸與變換資訊。"""

    width: int
    height: int
    letterbox: LetterboxParams | None = None


class BBox(BaseModel):
    """軸對齊邊界框，xyxy 格式。"""

    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return max(0.0, self.width) * max(0.0, self.height)

    @property
    def center(self) -> tuple[float, float]:
        return (self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2

    def as_list(self) -> list[float]:
        return [self.x1, self.y1, self.x2, self.y2]

    @classmethod
    def from_list(cls, v: list[float] | tuple[float, ...]) -> BBox:
        return cls(x1=float(v[0]), y1=float(v[1]), x2=float(v[2]), y2=float(v[3]))


class Detection(BaseModel):
    """單一偵測物件。

    分割模型（-seg）會帶 mask_contour 與 mask_area；偵測模型則否。
    業務層判定若依賴遮罩，必須先檢查 has_mask。
    """

    class_id: int
    class_name: str
    confidence: float
    bbox: BBox

    # 分割模型專屬。輪廓為多邊形頂點 [[x, y], ...]，座標系同 coord_system。
    has_mask: bool = False
    mask_area: float | None = None
    mask_contour: list[list[float]] | None = None

    # pose 模型專屬：每個關鍵點 [x, y, conf]，座標系同 coord_system。順序由模型定義（例如區域四角 TL→TR→BR→BL）。
    keypoints: list[list[float]] | None = None

    # 業務層計算的衍生指標掛這裡，不污染核心欄位。
    metrics: dict[str, Any] = Field(default_factory=dict)


class VerdictStatus(str, Enum):
    OK = "ok"
    NG = "ng"
    UNKNOWN = "unknown"   # 判定所需條件不足（例如缺少基準區域）


class Verdict(BaseModel):
    """業務層的判定結果。

    偵測結果（Detection）是模型說了什麼；判定（Verdict）是我們據此
    決定的結論。兩者分開，因為換模型不該影響判定規則的表達方式。
    """

    status: VerdictStatus
    reason: str = ""
    details: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def ok(cls, reason: str = "", **details: Any) -> Verdict:
        return cls(status=VerdictStatus.OK, reason=reason, details=details)

    @classmethod
    def ng(cls, reason: str, **details: Any) -> Verdict:
        return cls(status=VerdictStatus.NG, reason=reason, details=details)

    @classmethod
    def unknown(cls, reason: str, **details: Any) -> Verdict:
        return cls(status=VerdictStatus.UNKNOWN, reason=reason, details=details)


class FrameResult(BaseModel):
    """一個影格的完整推論結果 —— 系統對外的主要資料單位。"""

    schema_version: Literal["1.0"] = SCHEMA_VERSION

    # ── 幀身分 ────────────────────────────────────────────
    camera_id: str
    frame_id: str
    # ⭐ 前端配對的鍵：畫進影像左上角的身分標記。
    # 時間戳無法跨路徑配對（影像與結果走各自的 session，RTP timestamp
    # 基準不同且抖動 200ms），故改用嵌入畫面的 id。
    # 見 docs/findings/2026-08-28-sync-verification.md
    marker_id: int | None = None
    # 來源串流的 RTP timestamp。保留供診斷，不用於配對。
    rtp_timestamp: int | None = None
    pts: int | None = None
    # 牆鐘時間，僅供記錄與查詢，不可用於配對（兩條路徑延遲不同）。
    timestamp: float = Field(default_factory=time.time)

    # ── 模型身分 ──────────────────────────────────────────
    model_name: str
    model_version: str
    # 產生這筆判定的站別（一條線可掛多站：主站 + PPE 等）。None = 舊格式，視為主站。
    station: str | None = None

    # ── 結果 ──────────────────────────────────────────────
    coord_system: CoordSystem = CoordSystem.ORIGINAL
    image_meta: ImageMeta
    detections: list[Detection] = Field(default_factory=list)
    verdict: Verdict | None = None

    # ── 診斷 ──────────────────────────────────────────────
    inference_ms: float | None = None
    postprocess_ms: float | None = None

    def by_class(self, class_name: str) -> list[Detection]:
        return [d for d in self.detections if d.class_name == class_name]
