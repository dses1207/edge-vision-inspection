"""ROI 推論的座標還原：模型輸出（letterbox 後的 ROI 座標）→ 全幅像素。"""
from edgevision_postprocess import compute_letterbox, compute_letterbox_roi


def test_full_frame_is_special_case_of_roi():
    a = compute_letterbox(1280, 720, 640)
    b = compute_letterbox_roi(0, 0, 1280, 720, 640)
    assert (a.scale, a.pad_x, a.pad_y) == (b.scale, b.pad_x, b.pad_y) and (b.offset_x, b.offset_y) == (0, 0)
    assert b.to_original(320, 320) == a.to_original(320, 320)


def test_roi_output_maps_back_to_full_frame_pixels():
    """右下裁切 960×540；模型 640：scale 2/3 → 640×360，pad_y 140。"""
    lb = compute_letterbox_roi(912, 486, 1872, 1026, 640)
    assert abs(lb.scale - 640 / 960) < 1e-9 and lb.pad_x == 0 and lb.pad_y == 140
    # letterbox 左上角 (0, 140) = ROI 左上角 = 全幅 (912, 486)
    assert lb.to_original(0, 140) == (912.0, 486.0)
    # letterbox 右下角 (640, 500) = ROI 右下角 = 全幅 (1872, 1026)
    x, y = lb.to_original(640, 500)
    assert abs(x - 1872) < 1e-6 and abs(y - 1026) < 1e-6
    # 反向一致
    assert lb.to_letterboxed(*lb.to_original(123.0, 234.0)) == (123.0, 234.0)


def test_clip_to_content_keeps_boxes_inside_roi():
    import numpy as np
    from edgevision_postprocess.letterbox import clip_to_content
    lb = compute_letterbox_roi(0, 270, 960, 1080, 640)       # 例：左半下 3/4 → scale 2/3，pad_y 50
    boxes = np.array([[600.0, 100.0, 660.0, 200.0], [-5.0, 40.0, 30.0, 600.0]])
    c = clip_to_content(boxes, lb, 640)
    assert c[0, 2] == 640.0 and lb.to_original(c[0, 2], c[0, 3])[0] == 960.0   # 右緣停在 ROI 邊界
    assert c[1, 0] == 0.0 and c[1, 1] == 50.0 and c[1, 3] == 590.0            # 上下補邊區夾掉
    assert lb.to_original(c[1, 0], c[1, 1]) == (0.0, 270.0)
    assert boxes[0, 2] == 660.0                                               # 不改原陣列


def test_clip_to_content_full_frame_equals_frame_bounds():
    import numpy as np
    from edgevision_postprocess.letterbox import clip_to_content
    lb = compute_letterbox(1920, 1080, 640)
    c = clip_to_content(np.array([[-10.0, 0.0, 700.0, 640.0]]), lb, 640)
    x1, y1 = lb.to_original(c[0, 0], c[0, 1])
    x2, y2 = lb.to_original(c[0, 2], c[0, 3])
    assert (x1, y1, x2, y2) == (0.0, 0.0, 1920.0, 1080.0)
