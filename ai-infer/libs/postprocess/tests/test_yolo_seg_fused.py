"""fused 格式（NMS + mask 合成都在 ONNX 裡）的後處理。

⭐ 核心保證：同一個偵測，raw 路徑與 fused 路徑要給出【相同】的框、面積、輪廓。
   fused 只是把「係數×prototypes」搬進模型，其餘不變 —— 這裡拿同一組 proto/coeff
   兩邊各跑一次，直接比。
"""
import numpy as np
from edgevision_postprocess import YoloSegPostprocessor, compute_letterbox

CLASS_NAMES = ["board", "hand"]
N_MASK, HW, MAX_DET = 32, 160, 16
FEAT = 4 + len(CLASS_NAMES) + N_MASK
LB = compute_letterbox(1280, 720, 640)


def _raw(box_xyxy, coeff, score=0.95, cls=0):
    """raw 格式：(38, 8400)，一個 anchor 有東西。"""
    out0 = np.zeros((FEAT, 8400), np.float32)
    x1, y1, x2, y2 = box_xyxy
    out0[0, 0], out0[1, 0] = (x1 + x2) / 2, (y1 + y2) / 2
    out0[2, 0], out0[3, 0] = x2 - x1, y2 - y1
    out0[4 + cls, 0] = score
    out0[6:6 + N_MASK, 0] = coeff
    return out0


def _fused(box_xyxy, mask160, score=0.95, cls=0, n_valid=1):
    """fused 格式：(1, MAX_DET, 38) + (1, MAX_DET, 160, 160)，padding 列 score=0。"""
    out0 = np.zeros((1, MAX_DET, FEAT), np.float32)
    masks = np.zeros((1, MAX_DET, HW, HW), np.float32)
    for i in range(n_valid):
        out0[0, i, :4] = box_xyxy
        out0[0, i, 4], out0[0, i, 5] = score, cls
        masks[0, i] = mask160
    return out0, masks


def _proto_and_mask(coeff):
    """造一組 proto，並算出它跟 coeff 合成後的 160×160（就是模型裡那段 sigmoid(coeff@proto)）。"""
    rng = np.random.default_rng(7)
    proto = rng.normal(size=(N_MASK, HW, HW)).astype(np.float32)
    # 讓框內是前景：中央一塊拉高
    proto[0, 40:120, 40:120] += 6.0
    mask = 1.0 / (1.0 + np.exp(-(coeff @ proto.reshape(N_MASK, -1)))).reshape(HW, HW)
    return proto, mask.astype(np.float32)


# ── 格式偵測 ─────────────────────────────────────────────

def test_is_fused_detects_both_layouts():
    assert YoloSegPostprocessor.is_fused(np.zeros((1, MAX_DET, FEAT)))
    assert YoloSegPostprocessor.is_fused(np.zeros((MAX_DET, FEAT)))
    assert not YoloSegPostprocessor.is_fused(np.zeros((1, FEAT, 8400)))
    assert not YoloSegPostprocessor.is_fused(np.zeros((8400, FEAT)))


# ── 基本行為 ─────────────────────────────────────────────

def test_fused_padding_rows_are_ignored():
    """padding 列 score=0，絕不能變成偵測。"""
    pp = YoloSegPostprocessor(CLASS_NAMES)
    out0, masks = _fused((200, 150, 400, 300), np.ones((HW, HW), np.float32), n_valid=2)
    dets = pp(out0, masks, LB, 1280, 720)
    assert len(dets) == 2


def test_fused_all_padding_gives_nothing():
    pp = YoloSegPostprocessor(CLASS_NAMES)
    out0 = np.zeros((1, MAX_DET, FEAT), np.float32)
    assert pp(out0, np.zeros((1, MAX_DET, HW, HW), np.float32), LB, 1280, 720) == []


def test_fused_conf_threshold_applies():
    pp = YoloSegPostprocessor(CLASS_NAMES, conf=0.5)
    out0, masks = _fused((200, 150, 400, 300), np.ones((HW, HW), np.float32), score=0.3)
    assert pp(out0, masks, LB, 1280, 720) == []


def test_fused_box_is_xyxy_letterbox_not_xywh():
    """fused 的 box 已是 xyxy（跟 raw 的 cxcywh 不同），不能再做 cx-w/2。"""
    pp = YoloSegPostprocessor(CLASS_NAMES)
    out0, masks = _fused((200, 150, 400, 300), np.ones((HW, HW), np.float32))
    d = pp(out0, masks, LB, 1280, 720)[0]
    ox1, oy1 = LB.to_original(200, 150); ox2, oy2 = LB.to_original(400, 300)
    assert abs(d.bbox.x1 - ox1) < 1 and abs(d.bbox.y1 - oy1) < 1
    assert abs(d.bbox.x2 - ox2) < 1 and abs(d.bbox.y2 - oy2) < 1


def test_fused_without_masks_still_gives_boxes():
    pp = YoloSegPostprocessor(CLASS_NAMES)
    out0, _ = _fused((200, 150, 400, 300), np.ones((HW, HW), np.float32))
    dets = pp(out0, None, LB, 1280, 720)
    assert len(dets) == 1 and not dets[0].has_mask


# ── ⭐ raw vs fused 一致性 ───────────────────────────────

def test_fused_matches_raw_area_and_contour():
    """同一組 proto/coeff：raw 路徑自己乘、fused 路徑拿現成的 —— 面積與輪廓要一致。"""
    pp = YoloSegPostprocessor(CLASS_NAMES)
    coeff = np.zeros(N_MASK, np.float32); coeff[0] = 1.0
    proto, mask = _proto_and_mask(coeff)
    box = (180.0, 170.0, 460.0, 470.0)

    raw_det = pp(_raw(box, coeff), proto, LB, 1280, 720)[0]
    out0, masks = _fused(box, mask)
    fused_det = pp(out0, masks, LB, 1280, 720)[0]

    assert raw_det.has_mask and fused_det.has_mask
    assert raw_det.mask_area > 0
    assert abs(raw_det.mask_area - fused_det.mask_area) / raw_det.mask_area < 1e-3
    a, b = np.array(raw_det.mask_contour), np.array(fused_det.mask_contour)
    assert len(a) == len(b) and np.abs(a - b).max() < 1.0


def test_fused_mask_crops_match_raw():
    """燒錄用的小塊（decode_with_mask_crops）兩邊也要一致 —— nvdsosd 拉伸後畫面才不會差。"""
    pp = YoloSegPostprocessor(CLASS_NAMES)
    coeff = np.zeros(N_MASK, np.float32); coeff[0] = 1.0
    proto, mask = _proto_and_mask(coeff)
    box = (180.0, 170.0, 460.0, 470.0)

    (_, raw_crop), = pp.decode_with_mask_crops(_raw(box, coeff), proto, LB, 1280, 720)
    out0, masks = _fused(box, mask)
    (_, fused_crop), = pp.decode_with_mask_crops(out0, masks, LB, 1280, 720)

    assert raw_crop is not None and fused_crop is not None
    assert raw_crop.shape == fused_crop.shape
    assert np.abs(raw_crop - fused_crop).max() < 1e-4


def test_fused_degenerate_box_gives_no_mask():
    pp = YoloSegPostprocessor(CLASS_NAMES)
    out0, masks = _fused((300, 300, 300, 300), np.ones((HW, HW), np.float32))
    (_, crop), = pp.decode_with_mask_crops(out0, masks, LB, 1280, 720)
    assert crop is None
