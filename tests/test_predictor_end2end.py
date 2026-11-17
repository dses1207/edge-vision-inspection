"""YOLO26（NMS-free）輸出的 Triton 後處理：(max_det, 6) 直接過濾信心、還原 letterbox 座標。"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from edgevision.inference.predictor import compute_letterbox  # noqa: E402
from edgevision.inference.predictor_triton import TritonPredictor  # noqa: E402


def _pred(class_names, conf=0.25):
    p = TritonPredictor.__new__(TritonPredictor)
    p.imgsz = 640; p.conf = conf; p.iou = 0.45; p._name = "t"
    p._class_names = class_names; p._det_output = "output0"; p._proto_output = None; p._end2end = True
    return p


def test_end2end_rows_filtered_and_scaled():
    p = _pred(["buffalo", "elephant", "rhino", "zebra"])
    w, h = 1280, 720
    lb = compute_letterbox(w, h, 640)          # scale 0.5, pad_y 140
    rows = np.zeros((300, 6), dtype=np.float32)
    rows[0] = [100, 200, 300, 400, 0.9, 1]     # letterbox 座標
    rows[1] = [10, 10, 20, 20, 0.1, 0]         # 低信心 → 濾掉
    dets = p._postprocess({"output0": rows[None]}, lb, w, h)
    assert len(dets) == 1
    d = dets[0]
    assert d.class_name == "elephant" and abs(d.confidence - 0.9) < 1e-6
    # (100,200) → ((100-0)/0.5, (200-140)/0.5) = (200, 120)
    assert abs(d.bbox.x1 - 200) < 1e-3 and abs(d.bbox.y1 - 120) < 1e-3
    assert abs(d.bbox.x2 - 600) < 1e-3 and abs(d.bbox.y2 - 520) < 1e-3


def test_end2end_empty():
    p = _pred(["a"])
    lb = compute_letterbox(640, 640, 640)
    assert p._postprocess({"output0": np.zeros((1, 300, 6), dtype=np.float32)}, lb, 640, 640) == []


def test_grid_path_still_used_when_not_end2end():
    p = _pred(["a", "b"]); p._end2end = False
    lb = compute_letterbox(640, 640, 640)
    grid = np.zeros((1, 6, 8400), dtype=np.float32)
    grid[0, :, 0] = [320, 320, 100, 100, 0.95, 0.1]   # cx,cy,w,h + 2 類分數
    dets = p._postprocess({"output0": grid}, lb, 640, 640)
    assert len(dets) == 1 and dets[0].class_name == "a"


# ── 融合格式（fuse_masks）：output0 (max_det, 38) 已 NMS + masks (max_det, 160, 160)，無 output1 ──

class _FakeClient:
    def __init__(self, outputs):
        self._outputs = outputs

    def get_model_metadata(self, name):
        return {"inputs": [{"name": "images"}], "outputs": self._outputs, "versions": ["1"]}


def test_probe_recognizes_fused_outputs():
    p = _pred(["board", "hand"])
    # masks 也是 4 維：舊規則會把它誤認成 prototypes、把 output0 當 grid
    p._probe_model(_FakeClient([
        {"name": "output0", "shape": [-1, 16, 38]},
        {"name": "masks", "shape": [-1, 16, 160, 160]},
    ]))
    assert p._det_output == "output0" and p._mask_output == "masks"
    assert p._proto_output is None and p._end2end is True
    assert p._output_names == ["output0", "masks"]


def test_probe_raw_seg_unchanged():
    p = _pred(["board", "hand"])
    p._probe_model(_FakeClient([
        {"name": "output0", "shape": [-1, 38, 8400]},
        {"name": "output1", "shape": [-1, 32, 160, 160]},
    ]))
    assert p._proto_output == "output1" and p._mask_output is None and p._end2end is False


def test_fused_rows_with_masks():
    p = _pred(["board", "hand"]); p._mask_output = "masks"
    w, h = 1280, 720
    lb = compute_letterbox(w, h, 640)          # scale 0.5, pad_y 140
    rows = np.zeros((16, 38), dtype=np.float32)
    rows[0, :6] = [100, 200, 300, 400, 0.9, 1]
    rows[1, :6] = [10, 10, 20, 20, 0.1, 0]     # 低信心 → 濾掉
    masks = np.zeros((16, 160, 160), dtype=np.float32)
    masks[0, 50:100, 25:75] = 0.99             # letterbox (100..300, 200..400)，剛好填滿框
    dets = p._postprocess({"output0": rows[None], "masks": masks[None]}, lb, w, h)
    assert len(dets) == 1
    d = dets[0]
    assert d.class_name == "hand" and abs(d.bbox.x1 - 200) < 1e-3 and abs(d.bbox.y2 - 520) < 1e-3
    assert d.has_mask and d.mask_contour
    # 框內 200×200 letterbox 像素 → 原圖 400×400（scale 0.5）
    assert abs(d.mask_area - 400 * 400) / (400 * 400) < 0.05
