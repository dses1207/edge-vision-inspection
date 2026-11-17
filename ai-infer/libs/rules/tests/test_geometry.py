from edgevision_contracts import BBox, Detection
from edgevision_rules import (
    count_touching,
    edge_distance_px,
    oriented_quad,
    overlap_ratio,
    quad_area,
    sort_quad,
)


def det(x1, y1, x2, y2, contour=True):
    d = Detection(class_id=0, class_name="x", confidence=1.0, bbox=BBox(x1=x1, y1=y1, x2=x2, y2=y2))
    if contour:
        d.has_mask = True
        d.mask_contour = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
        d.mask_area = (x2 - x1) * (y2 - y1)
    return d


def test_overlap_ratio_is_target_area_fraction():
    anchor = det(0, 0, 100, 100)
    half_in = det(50, 0, 150, 100)
    assert abs(overlap_ratio(anchor, half_in) - 0.5) < 1e-6
    assert overlap_ratio(anchor, det(200, 200, 300, 300)) == 0.0


def test_edge_distance_zero_when_touching():
    anchor = det(0, 0, 100, 100)
    assert edge_distance_px(anchor, det(90, 40, 120, 60)) == 0.0
    assert 39 < edge_distance_px(anchor, det(140, 40, 160, 60)) < 41


def test_count_touching_unions_overlap_and_distance():
    anchor = det(0, 0, 100, 100)
    near = det(110, 40, 130, 60)      # 不重疊，距離 10
    far = det(300, 300, 320, 320)
    assert count_touching(anchor, [near, far], min_overlap=0.15, max_distance_px=0) == 0
    assert count_touching(anchor, [near, far], min_overlap=0.15, max_distance_px=40) == 1


def test_quad_area_uses_perimeter_order():
    q = sort_quad([(0, 0), (10, 0), (0, 10), (10, 10)])
    assert quad_area(q) == 100.0


def test_oriented_quad_falls_back_to_bbox_without_mask():
    d = det(0, 0, 10, 20, contour=False)
    assert oriented_quad(d) == ((0, 0), (10, 0), (0, 20), (10, 20))


def test_overlap_ratio_falls_back_to_ios_when_anchor_has_no_mask():
    """舊系統：anchor 無遮罩 → bbox IoS（交集 / 較小者），不是 target 面積佔比。"""
    anchor = det(0, 0, 100, 100, contour=False)
    small_half_in = det(90, 0, 110, 20)      # 面積 400，交集 200 → IoS 0.5；若用 target 佔比也是 0.5
    big_overlapping = det(50, 0, 250, 100)   # 面積 20000，交集 5000 → IoS 0.5（較小者是 anchor 10000）；target 佔比會是 0.25
    assert abs(overlap_ratio(anchor, small_half_in) - 0.5) < 1e-6
    assert abs(overlap_ratio(anchor, big_overlapping) - 0.5) < 1e-6


def test_bbox_iou():
    from edgevision_rules import bbox_iou
    assert bbox_iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert bbox_iou((0, 0, 10, 10), (5, 0, 15, 10)) == 50 / 150
    assert bbox_iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0
