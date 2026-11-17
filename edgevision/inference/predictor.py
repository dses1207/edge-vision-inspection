"""推論層：把影格送去模型，取回偵測結果。

介面刻意收窄成「影格進、Detection 出」，讓實作可替換：
  ・TritonPredictor —— 生產用，模型集中於 Triton，支援熱切換與動態批次
  ・MockPredictor   —— 開發用，無 GPU 環境驗證串接是否正確

上層（業務層）只依賴 Predictor 介面，不知道模型跑在哪、用什麼格式。
這是換推論引擎時下游零改動的前提。

⚠️ 座標系約定：Predictor 回傳的 bbox 與 mask_contour 一律為【原始影格
座標系】。letterbox 的換算由實作內部處理完畢，不外洩給業務層 ——
否則每個 handler 都要自己記得換算，遲早有人忘記。
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from libs.contracts import BBox, Detection, ImageMeta, LetterboxParams

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class PredictResult:
    """單次推論的產出。"""

    detections: list[Detection]
    image_meta: ImageMeta
    inference_ms: float
    postprocess_ms: float = 0.0


class Predictor(ABC):
    """推論器介面。"""

    @abstractmethod
    def predict(self, image: np.ndarray) -> PredictResult: ...

    @abstractmethod
    def close(self) -> None: ...

    @property
    @abstractmethod
    def model_name(self) -> str: ...

    @property
    @abstractmethod
    def model_version(self) -> str: ...

    @property
    @abstractmethod
    def is_ready(self) -> bool: ...


def compute_letterbox(
    src_w: int, src_h: int, dst: int
) -> LetterboxParams:
    """計算等比縮放 + 置中補邊的參數。

    YOLO 的標準前處理：影像縮到 dst×dst 的正方形，長邊貼齊、短邊補灰。
    這個變換必須被記錄下來，否則模型輸出的座標無法還原回原圖。
    """
    scale = min(dst / src_w, dst / src_h)
    new_w, new_h = src_w * scale, src_h * scale
    return LetterboxParams(
        scale=scale,
        pad_x=(dst - new_w) / 2,
        pad_y=(dst - new_h) / 2,
    )


class MockPredictor(Predictor):
    """假推論器：產生可預測的移動物件，用於驗證整條鏈路。

    存在的理由：開發機無 GPU，但解碼、同步、前端、事件落地這些
    環節都不需要真實模型就能驗證。用 mock 先把管線接通，
    真實模型到位後只換這一個類別。

    產生的物件會隨時間移動，方便肉眼確認「框有沒有跟著畫面動」——
    靜止的框看不出同步問題。
    """

    def __init__(
        self,
        *,
        model_name: str = "seg_alpha",
        model_version: str = "mock",
        imgsz: int = 640,
        latency_ms: float = 60.0,
    ) -> None:
        self._name = model_name
        self._version = model_version
        self._imgsz = imgsz
        self._latency = latency_ms / 1000.0
        self._t0 = time.monotonic()

    @property
    def model_name(self) -> str:
        return self._name

    @property
    def model_version(self) -> str:
        return self._version

    @property
    def is_ready(self) -> bool:
        return True

    def close(self) -> None:
        pass

    def predict(self, image: np.ndarray) -> PredictResult:
        t0 = time.perf_counter()
        h, w = image.shape[:2]
        time.sleep(self._latency)   # 模擬 GPU 推論耗時，讓背壓機制真的被觸發

        phase = (time.monotonic() - self._t0) * 0.35
        cx = w * (0.5 + 0.22 * np.sin(phase))
        cy = h * (0.5 + 0.12 * np.cos(phase * 0.8))
        bw, bh = w * 0.26, h * 0.30

        board = Detection(
            class_id=0, class_name="board", confidence=0.94,
            bbox=BBox(x1=cx-bw/2, y1=cy-bh/2, x2=cx+bw/2, y2=cy+bh/2),
            has_mask=True,
            mask_area=bw * bh * 0.82,
            mask_contour=_rounded_quad(cx, cy, bw, bh),
        )
        hx = cx + bw * 0.42
        hy = cy - bh * 0.18
        hw, hh = w * 0.075, h * 0.11
        hand = Detection(
            class_id=1, class_name="hand", confidence=0.88,
            bbox=BBox(x1=hx-hw/2, y1=hy-hh/2, x2=hx+hw/2, y2=hy+hh/2),
            has_mask=True,
            mask_area=hw * hh * 0.7,
            mask_contour=_rounded_quad(hx, hy, hw, hh),
        )

        elapsed = (time.perf_counter() - t0) * 1000
        return PredictResult(
            detections=[board, hand],
            image_meta=ImageMeta(
                width=w, height=h,
                letterbox=compute_letterbox(w, h, self._imgsz),
            ),
            inference_ms=elapsed,
        )


def _rounded_quad(cx: float, cy: float, w: float, h: float) -> list[list[float]]:
    """產生一個八邊形輪廓，模擬分割遮罩的形狀。"""
    hw, hh = w / 2, h / 2
    k = 0.28
    return [
        [cx - hw + w*k, cy - hh], [cx + hw - w*k, cy - hh],
        [cx + hw,       cy - hh + h*k], [cx + hw,      cy + hh - h*k],
        [cx + hw - w*k, cy + hh], [cx - hw + w*k, cy + hh],
        [cx - hw,       cy + hh - h*k], [cx - hw,      cy - hh + h*k],
    ]
