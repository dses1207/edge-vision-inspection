"""存檔繪製：畫得對、不當掉、不動到原圖。"""

from __future__ import annotations

import numpy as np
from edgevision_contracts import BBox, Detection
from edgevision_postprocess.render import color_for, draw_detections


def _img(w=200, h=120):
    return np.zeros((h, w, 3), dtype=np.uint8)


def _det(class_name="board", contour=None, bbox=(10, 10, 60, 50)):
    return Detection(
        class_id=0, class_name=class_name, confidence=0.9,
        bbox=BBox(x1=bbox[0], y1=bbox[1], x2=bbox[2], y2=bbox[3]),
        has_mask=contour is not None,
        mask_contour=contour,
        mask_area=float(len(contour)) if contour else None,
    )


def test_does_not_mutate_input():
    """呼叫端可能還要用原圖，就地改會有隱形副作用。"""
    img = _img()
    before = img.copy()
    draw_detections(img, [_det(contour=[[10, 10], [60, 10], [60, 50], [10, 50]])])
    assert np.array_equal(img, before)


def test_mask_is_filled():
    img = _img()
    out = draw_detections(img, [_det(contour=[[10, 10], [60, 10], [60, 50], [10, 50]])])
    assert out[30, 35].any(), "多邊形內部應該被填色"


def test_empty_detections_returns_copy():
    img = _img()
    out = draw_detections(img, [])
    assert np.array_equal(out, img) and out is not img


def test_detection_without_mask_still_draws_box():
    """偵測模型（非 -seg）沒有遮罩，仍要畫框。"""
    out = draw_detections(_img(), [_det(contour=None)])
    assert out.any(), "沒有遮罩時仍應畫出框線"


def test_degenerate_contour_is_skipped():
    """少於 3 點構不成多邊形；不該當掉，也不該畫。"""
    out = draw_detections(_img(), [_det(contour=[[10, 10], [20, 20]])])
    assert out[30, 35].sum() == 0, "退化輪廓不應被填色"


def test_out_of_bounds_contour_is_clipped():
    """輪廓來自原圖座標，四捨五入後可能差一兩個像素落在界外。"""
    out = draw_detections(
        _img(), [_det(contour=[[-50, -50], [500, -50], [500, 500], [-50, 500]])]
    )
    assert out.shape == (120, 200, 3)


def test_overlap_blended_once():
    """重疊處只混一次，否則深淺不再代表類別。"""
    sq = [[10, 10], [80, 10], [80, 80], [10, 80]]
    one = draw_detections(_img(), [_det("board", sq)])
    two = draw_detections(_img(), [_det("board", sq), _det("board", sq)])
    assert np.array_equal(one, two)


def test_colors_are_vivid_and_errors_are_red():
    from edgevision_postprocess.colors import ERROR_RGB, box_color_rgb
    assert box_color_rgb("ERROR_Zone", 4) == ERROR_RGB == box_color_rgb("Err_Place", 4)
    assert box_color_rgb("bare hand", 0) == ERROR_RGB
    assert box_color_rgb("PART", 0) != ERROR_RGB                     # 一般類別不用紅色
    assert box_color_rgb("normal", 1) != box_color_rgb("table", 2)  # 同模型不同類 → 不同色
    assert box_color_rgb("x", 3) == box_color_rgb("x", 3)          # 穩定
    for c in [box_color_rgb(f"c{i}", i) for i in range(7)]:
        assert max(c) - min(c) >= 0.4                                # 高飽和，不是灰
    assert color_for("board") == (255, 153, 51)                      # 存檔與直播同一套（BGR）


def test_masks_can_be_disabled():
    c = [[10, 10], [60, 10], [60, 50], [10, 50]]
    out = draw_detections(_img(), [_det(contour=c)], draw_masks=False, draw_labels=False)
    assert out[30, 35].sum() == 0, "關掉遮罩後內部不該被填色"
