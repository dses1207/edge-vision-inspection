"""⭐ 前處理一致性契約。

同一張影像：訓練/驗證用的 Ultralytics LetterBox → 推論用的 predictor_triton._preprocess
兩者產出的 tensor 必須逐元素相等。這是最經典也最難查的 ML bug，在這裡釘死。

需要 ultralytics（在 trainer 容器內跑）；開發機沒有就跳過。
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

ultralytics = pytest.importorskip("ultralytics")


def _triton_preprocess(img, imgsz):
    from edgevision.inference.predictor import compute_letterbox
    from edgevision.inference.predictor_triton import TritonPredictor

    p = TritonPredictor.__new__(TritonPredictor)
    p.imgsz = imgsz
    h, w = img.shape[:2]
    return p._preprocess(img, compute_letterbox(w, h, imgsz))


def _ultra_preprocess(img, imgsz):
    from ultralytics.data.augment import LetterBox

    lb = LetterBox((imgsz, imgsz), auto=False, scaleup=True, stride=32)
    out = lb(image=img)                       # BGR HWC，114 補邊
    out = out[..., ::-1].transpose(2, 0, 1)   # → RGB CHW
    return np.ascontiguousarray(out, dtype=np.float32) / 255.0


@pytest.mark.parametrize("shape", [(720, 1280, 3), (480, 640, 3), (1000, 700, 3)])
def test_letterbox_tensor_identical(shape):
    rng = np.random.RandomState(0)
    img = rng.randint(0, 255, size=shape, dtype=np.uint8)
    a = _triton_preprocess(img, 640)
    b = _ultra_preprocess(img, 640)
    a = np.asarray(a).reshape(3, 640, 640)
    assert a.shape == b.shape
    diff = np.abs(a - b)
    # 允許 resize 插值的 1/255 量級差異（兩邊都是 cv2 INTER_LINEAR，理論上為 0）
    assert diff.max() <= 1.5 / 255, f"最大差 {diff.max():.5f}，補邊或縮放不一致"
    assert (diff > 0.5 / 255).mean() < 0.01
