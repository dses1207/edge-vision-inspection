"""多模型並行推論。

場景:同一影格同時跑多個獨立模型,結果合併給業務層。
例:一幀同時跑「找板」與「找瑕疵」兩個模型,判定時一起看。

設計:實作同一個 Predictor 介面,對推論服務與業務層【完全透明】——
它們拿到的還是一個 PredictResult,不知道背後是幾個模型。這就是
「介面先定死,實作可替換」的價值:加多模型不用改管線、不用改 handler。

⭐ 為什麼不用 Triton Ensemble:
  Ensemble 是「串接」用的(A輸出→B輸入,固定管線),適合
  偵測→分類這種鏈式流程。但「並行跑幾個獨立模型再合併」是
  在推論服務端並行呼叫更直接,且各模型的 batching 仍各自生效
  (每個模型在 Triton 端獨立併批,不受其他模型影響)。

  串接場景則應該用 Triton Ensemble(在 config 裡宣告),
  那是官方原生做法,延遲最低。本類別專門處理並行。
"""

from __future__ import annotations

import logging

from libs.contracts import Detection, ImageMeta
from edgevision.inference.predictor import PredictResult, Predictor

logger = logging.getLogger(__name__)


class MultiPredictor(Predictor):
    """並行呼叫多個 Predictor,合併偵測結果。

    各子 predictor 各自打自己的 Triton 模型 —— 因此各模型的
    dynamic batching 獨立生效,一個模型慢不影響另一個併批。
    """

    def __init__(self, predictors: dict[str, Predictor]) -> None:
        if not predictors:
            raise ValueError("至少需要一個 predictor")
        self._predictors = predictors
        # 併發方式:用各 predictor 的 launch()/collect() 在【同一執行緒】
        # 內併發 —— tritonclient 的 HTTP client 基於 gevent,不能跨執行緒
        # 共用(會拋 greenlet.error)。async_infer 讓多請求同時在飛,
        # 總延遲 ≈ 最慢的模型,而非相加。
        # 非 Triton 的 predictor(如 mock)無 launch,退回序列呼叫。
        self._can_async = all(
            hasattr(p, "launch") and hasattr(p, "collect")
            for p in predictors.values()
        )

    # ── Predictor 介面 ────────────────────────────────────

    @property
    def model_name(self) -> str:
        return "+".join(self._predictors.keys())

    @property
    def model_version(self) -> str:
        # 各子模型版本合起來,保留可追溯性
        return ",".join(
            f"{name}:{p.model_version}" for name, p in self._predictors.items()
        )

    @property
    def is_ready(self) -> bool:
        return all(p.is_ready for p in self._predictors.values())

    def close(self) -> None:
        for p in self._predictors.values():
            p.close()

    def predict(self, image) -> PredictResult:
        """併發跑所有模型,合併偵測結果。

        image_meta 取第一個(所有模型看同一張圖,letterbox 相同)。
        每個 detection 標上來源模型,業務層需要時可區分。
        """
        if self._can_async:
            # 先對所有模型發射請求(不阻塞),再逐一收結果 ——
            # 所有請求同時在飛,總延遲 ≈ 最慢的模型。
            launched = {
                name: p.launch(image)
                for name, p in self._predictors.items()
            }
            results = {}
            for name, ctx in launched.items():
                try:
                    results[name] = self._predictors[name].collect(ctx)
                except Exception:
                    logger.exception("模型 %s 推論失敗,略過其結果", name)
        else:
            # 退回序列(mock 等無 async 能力的 predictor)
            results = {}
            for name, p in self._predictors.items():
                try:
                    results[name] = p.predict(image)
                except Exception:
                    logger.exception("模型 %s 推論失敗,略過其結果", name)

        merged: list[Detection] = []
        image_meta: ImageMeta | None = None
        total_infer = 0.0
        total_post = 0.0

        for name, res in results.items():
            if image_meta is None:
                image_meta = res.image_meta
            for d in res.detections:
                # 標記來源模型,讓業務層能區分「這個框是哪個模型偵測的」
                d.metrics.setdefault("source_model", name)
                merged.append(d)
            total_infer = max(total_infer, res.inference_ms)  # 並行 → 取最大
            total_post += res.postprocess_ms

        if image_meta is None:
            # 所有模型都失敗:回空結果而非拋錯,不中斷推論管線
            h, w = image.shape[:2]
            image_meta = ImageMeta(width=w, height=h)

        return PredictResult(
            detections=merged,
            image_meta=image_meta,
            inference_ms=total_infer,
            postprocess_ms=total_post,
        )
