"""本地驗證「調配」：模擬 Triton 的 raw tensor 輸出，餵 _postprocess，
確認能正確拼出 bbox + mask —— 不需 GPU/Triton。

背景：DeepStream 路 A（output_tensor_meta 自算 mask）的核心，就是把 Triton 遞出的
raw tensor（output0/output1）用我們的 _postprocess 拼成偵測。DeepStream 遞料那關卡住，
但「拼裝邏輯本身對不對」可以在本地用假料獨立驗證——這就是本檔做的事。

造的假料格式與 seg_alpha 的 Triton 輸出【完全一致】：
  output0: [38, 8400]  (4 box + 2 類(board/hand) + 32 mask係數)
  output1: [32, 160, 160]  (32 prototypes)
"""
from __future__ import annotations

import numpy as np

from edgevision.inference.predictor import compute_letterbox
from edgevision.inference.predictor_triton import TritonPredictor


IMGSZ = 640
CLASS_NAMES = ["board", "hand"]
N_ANCHORS = 8400
N_MASK = 32
PROTO_HW = 160
FEATURES = 4 + len(CLASS_NAMES) + N_MASK  # 4+2+32 = 38


def make_predictor():
    """不連線建立 TritonPredictor（繞過 __init__ 的 Triton 探測），只測 _postprocess。"""
    p = object.__new__(TritonPredictor)
    p.imgsz = IMGSZ
    p.conf = 0.5
    p.iou = 0.45
    p._class_names = CLASS_NAMES
    p._name = "seg_alpha"
    p._det_output = "output0"
    p._proto_output = "output1"
    return p


def make_raw(board_box_xywh, board_coeff, proto):
    """造格式跟 Triton 一模一樣的 output0/output1。

    board_box_xywh: (cx, cy, w, h) 在 640 letterbox 座標
    board_coeff: [32] 該物件的 mask 係數
    proto: [32,160,160] prototypes
    只放一個 board（class 0）、高信心，其餘 anchor 全低分（會被 conf 濾掉）。
    """
    out0 = np.zeros((FEATURES, N_ANCHORS), dtype=np.float32)  # [38, 8400]，跟 Triton 一致
    # anchor 0 放 board：box + class0=0.95 + 係數
    cx, cy, w, h = board_box_xywh
    out0[0, 0], out0[1, 0], out0[2, 0], out0[3, 0] = cx, cy, w, h
    out0[4, 0] = 0.95            # class 0 (board) 高分
    out0[5, 0] = 0.01            # class 1 (hand) 低分
    out0[6:6 + N_MASK, 0] = board_coeff
    return {"output0": out0, "output1": proto.astype(np.float32)}


def test_postprocess_reconstructs_board_bbox():
    """假料放一個 board，_postprocess 應拼出 1 個 board 偵測、bbox 對得上。"""
    p = make_predictor()
    # board 在畫面中央、200x120 大小（640 letterbox 座標）
    cx, cy, w, h = 320.0, 320.0, 200.0, 120.0
    coeff = np.ones(N_MASK, dtype=np.float32) * 0.1
    proto = np.zeros((N_MASK, PROTO_HW, PROTO_HW), dtype=np.float32)
    raw = make_raw((cx, cy, w, h), coeff, proto)

    lb = compute_letterbox(1280, 720, IMGSZ)   # 原圖 720p → 640 letterbox
    dets = p._postprocess(raw, lb, 1280, 720)

    assert len(dets) == 1, f"應偵測到 1 個物件，實得 {len(dets)}"
    d = dets[0]
    assert d.class_name == "board", f"應為 board，實得 {d.class_name}"
    assert d.confidence > 0.9
    # bbox 應在合理範圍（映射回原圖 1280x720，中央附近）
    b = d.bbox
    assert 0 <= b.x1 < b.x2 <= 1280 and 0 <= b.y1 < b.y2 <= 720


def test_postprocess_low_conf_filtered():
    """全低分的假料應被 conf 濾掉，回 0 偵測。"""
    p = make_predictor()
    out0 = np.zeros((FEATURES, N_ANCHORS), dtype=np.float32)
    out0[4:6, :] = 0.1   # 所有類別分數都低於 conf(0.5)
    raw = {"output0": out0, "output1": np.zeros((N_MASK, PROTO_HW, PROTO_HW), np.float32)}
    lb = compute_letterbox(1280, 720, IMGSZ)
    dets = p._postprocess(raw, lb, 1280, 720)
    assert dets == []


def test_mask_reconstruction_produces_contour():
    """mask 調配：造一個會在框內產生前景的 prototype+係數，_build_masks 應算出遮罩輪廓。

    做法：讓 prototype[0] 在框對應區域是大正值，係數只取 prototype[0]，
    sigmoid(coeff@proto) 在該區域 → 接近 1（>0.5 前景），應找得到輪廓。
    """
    p = make_predictor()
    cx, cy, w, h = 320.0, 320.0, 200.0, 120.0
    # prototype[0] 整片設大正值 → sigmoid 後接近 1
    proto = np.zeros((N_MASK, PROTO_HW, PROTO_HW), dtype=np.float32)
    proto[0, :, :] = 10.0
    coeff = np.zeros(N_MASK, dtype=np.float32)
    coeff[0] = 1.0               # 只取 prototype[0]
    raw = make_raw((cx, cy, w, h), coeff, proto)

    lb = compute_letterbox(1280, 720, IMGSZ)
    dets = p._postprocess(raw, lb, 1280, 720)
    assert len(dets) == 1
    d = dets[0]
    # 應該有 mask（has_mask 或 mask_contour 有點）
    assert d.has_mask, "mask 重建應成功產生遮罩"
    assert d.mask_contour is not None and len(d.mask_contour) >= 3, "mask 輪廓應有點"
