"""假料格式與 seg_alpha 的 Triton 輸出一致：output0 [38, 8400]、output1 [32, 160, 160]。不需 GPU。"""
import numpy as np
from edgevision_postprocess import YoloSegPostprocessor, compute_letterbox

CLASS_NAMES = ["board", "hand"]
N_ANCHORS, N_MASK, PROTO_HW = 8400, 32, 160
FEATURES = 4 + len(CLASS_NAMES) + N_MASK


def make_out0(box_xywh, coeff):
    out0 = np.zeros((FEATURES, N_ANCHORS), dtype=np.float32)
    cx, cy, w, h = box_xywh
    out0[0, 0], out0[1, 0], out0[2, 0], out0[3, 0] = cx, cy, w, h
    out0[4, 0], out0[5, 0] = 0.95, 0.01
    out0[6:6 + N_MASK, 0] = coeff
    return out0


def test_reconstructs_board_bbox():
    pp = YoloSegPostprocessor(CLASS_NAMES)
    proto = np.zeros((N_MASK, PROTO_HW, PROTO_HW), np.float32)
    dets = pp(make_out0((320, 320, 200, 120), np.full(N_MASK, 0.1, np.float32)), proto,
              compute_letterbox(1280, 720, 640), 1280, 720)
    assert len(dets) == 1 and dets[0].class_name == "board" and dets[0].confidence > 0.9
    b = dets[0].bbox
    assert 0 <= b.x1 < b.x2 <= 1280 and 0 <= b.y1 < b.y2 <= 720


def test_low_conf_filtered():
    pp = YoloSegPostprocessor(CLASS_NAMES)
    out0 = np.zeros((FEATURES, N_ANCHORS), np.float32)
    out0[4:6, :] = 0.1
    assert pp(out0, np.zeros((N_MASK, PROTO_HW, PROTO_HW), np.float32),
              compute_letterbox(1280, 720, 640), 1280, 720) == []


def test_mask_reconstruction_produces_contour():
    pp = YoloSegPostprocessor(CLASS_NAMES)
    proto = np.zeros((N_MASK, PROTO_HW, PROTO_HW), np.float32)
    proto[0] = 10.0
    coeff = np.zeros(N_MASK, np.float32)
    coeff[0] = 1.0
    dets = pp(make_out0((320, 320, 200, 120), coeff), proto, compute_letterbox(1280, 720, 640), 1280, 720)
    assert len(dets) == 1 and dets[0].has_mask
    assert dets[0].mask_contour is not None and len(dets[0].mask_contour) >= 3


def test_accepts_batched_and_transposed_layouts():
    pp = YoloSegPostprocessor(CLASS_NAMES)
    out0 = make_out0((320, 320, 200, 120), np.zeros(N_MASK, np.float32))
    lb = compute_letterbox(1280, 720, 640)
    assert len(pp(out0[None, ...], None, lb, 1280, 720)) == 1     # (1, 38, 8400)
    assert len(pp(out0.T, None, lb, 1280, 720)) == 1              # (8400, 38)


def test_mask_area_is_pixel_count_in_original_resolution_within_bbox():
    """舊系統 mask_area = np.sum(mask > 0.5)（原圖解析度、bbox 內）。全前景 prototype → 面積 ≈ bbox 面積。"""
    pp = YoloSegPostprocessor(CLASS_NAMES)
    proto = np.zeros((N_MASK, PROTO_HW, PROTO_HW), np.float32)
    proto[0] = 10.0
    coeff = np.zeros(N_MASK, np.float32)
    coeff[0] = 1.0
    lb = compute_letterbox(1280, 720, 640)                     # scale 0.5, pad_y 140
    d = pp(make_out0((320, 320, 200, 120), coeff), proto, lb, 1280, 720)[0]
    expected = 400 * 240                                        # letterbox 200×120 → 原圖 400×240
    assert abs(d.mask_area - expected) < expected * 0.02
    xs = [p[0] for p in d.mask_contour]
    ys = [p[1] for p in d.mask_contour]
    assert 430 <= min(xs) <= 450 and 830 <= max(xs) <= 850 and 230 <= min(ys) <= 250 and 470 <= max(ys) <= 490


# ── ROI 放大：與「放大全圖」的舊做法比對（2026-09-09 改）

def _legacy_build_masks(pp, proto, coeffs, boxes, lb, w, h):
    """舊做法：每個偵測都把 mask 放大到整張畫面再裁切。留著當精度基準，不再上線。"""
    import cv2
    c, mh, mw = proto.shape
    flat = proto.reshape(c, -1)
    new_w, new_h = round(w * lb.scale), round(h * lb.scale)
    left, top = int(lb.pad_x), int(lb.pad_y)
    out = []
    for coeff, box in zip(coeffs, boxes):
        m = 1.0 / (1.0 + np.exp(-(coeff @ flat)))
        m = cv2.resize(m.reshape(mh, mw), (pp.imgsz, pp.imgsz), interpolation=cv2.INTER_LINEAR)
        m = m[top:top + new_h, left:left + new_w]
        m = cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR)
        ox1, oy1 = lb.to_original(float(box[0]), float(box[1]))
        ox2, oy2 = lb.to_original(float(box[2]), float(box[3]))
        x1, y1 = int(max(0, min(w, ox1))), int(max(0, min(h, oy1)))
        x2, y2 = int(max(0, min(w, ox2))), int(max(0, min(h, oy2)))
        crop = np.zeros((h, w), dtype=bool)
        crop[y1:y2, x1:x2] = True
        binary = (m > 0.5) & crop
        area = float(int(binary.sum()))
        cs, _ = cv2.findContours(binary.astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cs:
            out.append((None, 0.0))
            continue
        lg = max(cs, key=cv2.contourArea)
        ap = cv2.approxPolyDP(lg, 0.002 * cv2.arcLength(lg, True), True)
        out.append((ap.reshape(-1, 2).astype(float).tolist(), area))
    return out


def _blob_proto(cx, cy, r=40):
    """造一個中心在 (cx, cy)、半徑 r 的圓形前景（prototype 座標），係數只取 channel 0。"""
    proto = np.full((N_MASK, PROTO_HW, PROTO_HW), -6.0, np.float32)
    yy, xx = np.mgrid[0:PROTO_HW, 0:PROTO_HW]
    proto[0] = np.where((xx - cx) ** 2 + (yy - cy) ** 2 <= r * r, 6.0, -6.0)
    coeff = np.zeros((1, N_MASK), np.float32)
    coeff[0, 0] = 1.0
    return proto, coeff


def test_roi_matches_legacy_area_and_shape():
    """ROI 放大與舊的放大全圖：面積差 <1%、凸包面積差 <2%。"""
    import cv2
    pp = YoloSegPostprocessor(CLASS_NAMES)
    lb = compute_letterbox(1280, 720, 640)
    proto, coeff = _blob_proto(80, 80, r=36)
    # 框涵蓋整個圓（letterbox 座標）
    boxes = np.array([[160.0, 160.0, 480.0, 480.0]], np.float32)
    new = pp._build_masks(proto, coeff, boxes, lb, 1280, 720)
    old = _legacy_build_masks(pp, proto, coeff, boxes, lb, 1280, 720)
    (pn, an), (po, ao) = new[0], old[0]
    assert pn is not None and po is not None
    assert abs(an - ao) / ao < 0.01, f"面積差過大：{ao} vs {an}"
    hull = lambda p: cv2.contourArea(cv2.convexHull(np.asarray(p, np.float32).reshape(-1, 1, 2)))
    assert abs(hull(pn) - hull(po)) / hull(po) < 0.02, "凸包面積差過大"


def test_roi_contour_stays_inside_bbox():
    """margin 只用於放大，最終輪廓不得超出 bbox（Ultralytics #24272 的邊緣糊出）。"""
    pp = YoloSegPostprocessor(CLASS_NAMES)
    lb = compute_letterbox(1280, 720, 640)
    proto, coeff = _blob_proto(80, 80, r=50)
    boxes = np.array([[280.0, 280.0, 360.0, 360.0]], np.float32)   # 框比圓小，會裁到
    pts, area = pp._build_masks(proto, coeff, boxes, lb, 1280, 720)[0]
    assert pts is not None and area > 0
    x1, y1 = lb.to_original(280.0, 280.0)
    x2, y2 = lb.to_original(360.0, 360.0)
    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
    assert min(xs) >= x1 - 2 and max(xs) <= x2 + 2, "輪廓超出 bbox 左右範圍"
    assert min(ys) >= y1 - 2 and max(ys) <= y2 + 2, "輪廓超出 bbox 上下範圍"


def test_roi_handles_degenerate_boxes():
    """框在畫面外、寬高為零、無前景 —— 都要回 (None, 0.0) 不能拋。"""
    pp = YoloSegPostprocessor(CLASS_NAMES)
    lb = compute_letterbox(1280, 720, 640)
    proto, coeff = _blob_proto(80, 80, r=30)
    for box in ([300.0, 300.0, 300.0, 300.0], [-50.0, -50.0, -10.0, -10.0], [700.0, 700.0, 800.0, 800.0]):
        pts, area = pp._build_masks(proto, coeff, np.array([box], np.float32), lb, 1280, 720)[0]
        assert pts is None and area == 0.0, f"退化框未妥善處理：{box}"


def test_roi_batched_matmul_matches_per_detection():
    """批次矩陣乘的結果與逐個算相同。"""
    pp = YoloSegPostprocessor(CLASS_NAMES)
    lb = compute_letterbox(1280, 720, 640)
    proto, c0 = _blob_proto(60, 60, r=25)
    coeffs = np.repeat(c0, 3, axis=0)
    boxes = np.array([[120.0, 120.0, 360.0, 360.0]] * 3, np.float32)
    res = pp._build_masks(proto, coeffs, boxes, lb, 1280, 720)
    assert len({round(a, 6) for _, a in res}) == 1, "同樣輸入的三個偵測結果應完全相同"


# ── 給 nvdsosd 燒錄用的 mask 小塊 ────────────────────────


def test_mask_crops_align_to_bbox():
    """nvdsosd 會把小塊拉伸到 rect_params，所以小塊必須正好對應框 —— 差半格就是原圖好幾 px。"""
    pp = YoloSegPostprocessor(CLASS_NAMES)
    proto = np.full((N_MASK, PROTO_HW, PROTO_HW), -10.0, np.float32)
    proto[0, 40:80, 20:100] = 10.0                     # proto 座標的前景矩形（rows 40-80, cols 20-100）
    coeff = np.zeros(N_MASK, np.float32)
    coeff[0] = 1.0
    lb = compute_letterbox(640, 640, 640)              # scale 1、無 pad：原圖＝letterbox，ratio 4

    # 框正好蓋住前景 → 整塊都是前景
    pairs = pp.decode_with_mask_crops(make_out0((240, 240, 320, 160), coeff), proto, lb, 640, 640)
    assert len(pairs) == 1
    _, crop = pairs[0]
    assert crop is not None and crop.dtype == np.float32 and crop.ndim == 2
    assert crop.shape == (40, 80)
    assert crop[2:-2, 2:-2].mean() > 0.95

    # 框往左多蓋 80px（= 20 格）→ 左 20 格是背景、其餘前景
    pairs = pp.decode_with_mask_crops(make_out0((200, 240, 400, 160), coeff), proto, lb, 640, 640)
    _, crop = pairs[0]
    assert crop.shape == (40, 100)
    assert crop[:, :15].mean() < 0.05 and crop[:, 25:].mean() > 0.95


def test_mask_crops_boxes_match_call_and_survive_no_proto():
    """框要跟 __call__ 完全一樣（同一套解碼）；沒有 proto（偵測模型）就只回框、mask 為 None。"""
    pp = YoloSegPostprocessor(CLASS_NAMES)
    out0 = make_out0((320, 320, 200, 120), np.zeros(N_MASK, np.float32))
    lb = compute_letterbox(1280, 720, 640)
    dets = pp(out0, None, lb, 1280, 720)
    pairs = pp.decode_with_mask_crops(out0, None, lb, 1280, 720)
    assert [d.bbox for d, _ in pairs] == [d.bbox for d in dets]
    assert all(m is None for _, m in pairs)


def test_mask_crops_nothing_detected():
    pp = YoloSegPostprocessor(CLASS_NAMES)
    out0 = np.zeros((FEATURES, N_ANCHORS), np.float32)
    assert pp.decode_with_mask_crops(out0, np.zeros((N_MASK, PROTO_HW, PROTO_HW), np.float32),
                                     compute_letterbox(1280, 720, 640), 1280, 720) == []
