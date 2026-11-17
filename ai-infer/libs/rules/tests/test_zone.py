from edgevision_contracts import BBox, Detection
from edgevision_rules import filter_by_roi, point_in_polygon

SQ = [[0.2, 0.2], [0.8, 0.2], [0.8, 0.8], [0.2, 0.8]]


def det(cx, cy):
    return Detection(class_id=0, class_name="x", confidence=1.0, bbox=BBox(x1=cx - 5, y1=cy - 5, x2=cx + 5, y2=cy + 5))


def test_point_in_polygon_ray_casting():
    assert point_in_polygon(0.5, 0.5, SQ) and not point_in_polygon(0.1, 0.5, SQ)


def test_filter_keeps_all_without_detect_or_mask_regions():
    dets = [det(10, 10), det(500, 500)]
    assert filter_by_roi(dets, [{"type": "ref_quad", "points": SQ}], 1000, 1000) == dets


def test_detect_region_requires_center_inside_and_mask_excludes():
    inside, outside = det(500, 500), det(100, 100)
    assert filter_by_roi([inside, outside], [{"type": "detect", "points": SQ}], 1000, 1000) == [inside]
    assert filter_by_roi([inside, outside], [{"type": "mask", "points": SQ}], 1000, 1000) == [outside]


def test_polygons_overlap_inside_apart_touch_cross():
    """移植自舊系統的測試。"""
    from edgevision_rules import polygons_overlap
    sq = [(0, 0), (10, 0), (10, 10), (0, 10)]
    inside = [(2, 2), (8, 2), (8, 8), (2, 8)]
    apart = [(20, 20), (30, 20), (30, 30), (20, 30)]
    touch = [(10, 0), (20, 0), (20, 10), (10, 10)]
    cross = [(4, -5), (6, -5), (6, 15), (4, 15)]
    assert polygons_overlap(sq, inside) and not polygons_overlap(sq, apart)
    assert polygons_overlap(sq, touch) and polygons_overlap(sq, cross)


def test_expand_polygon_moves_vertices_outward():
    from edgevision_rules import expand_polygon
    sq = [(0, 0), (10, 0), (10, 10), (0, 10)]
    out = expand_polygon(sq, 1.0)
    assert out[0][0] < 0 and out[0][1] < 0 and out[2][0] > 10
    assert expand_polygon(sq, 0) == [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]


# ── 每類別各自的 ROI（舊腳本）────────────────────────────


def _det(name, cx, cy):
    from edgevision_contracts import BBox, Detection
    return Detection(class_id=0, class_name=name, confidence=0.9,
                     bbox=BBox(x1=cx - 5, y1=cy - 5, x2=cx + 5, y2=cy + 5))


def test_region_with_classes_only_constrains_those_classes():
    """某類別只在它自己的區域；別的類別不受這個區域限制。"""
    from edgevision_rules import filter_by_roi
    left = [[0.0, 0.0], [0.5, 0.0], [0.5, 1.0], [0.0, 1.0]]
    regions = [{"type": "detect", "points": left, "classes": ["err"]}]
    err_in, err_out, other_out = _det("err", 100, 100), _det("err", 900, 100), _det("other", 900, 100)
    kept = filter_by_roi([err_in, err_out, other_out], regions, 1000, 1000)
    assert kept == [err_in, other_out]


def test_per_class_regions_do_not_leak_into_each_other():
    """兩組類別各有各的區域。"""
    from edgevision_rules import filter_by_roi
    left = [[0.0, 0.0], [0.5, 0.0], [0.5, 1.0], [0.0, 1.0]]
    right = [[0.5, 0.0], [1.0, 0.0], [1.0, 1.0], [0.5, 1.0]]
    regions = [{"type": "detect", "points": left, "classes": ["part"]},
               {"type": "detect", "points": right, "classes": ["hand"]}]
    part_l, part_r, hand_l, hand_r = _det("part", 100, 100), _det("part", 900, 100), _det("hand", 100, 100), _det("hand", 900, 100)
    assert filter_by_roi([part_l, part_r, hand_l, hand_r], regions, 1000, 1000) == [part_l, hand_r]


def test_region_without_classes_keeps_old_behaviour():
    from edgevision_rules import filter_by_roi
    left = [[0.0, 0.0], [0.5, 0.0], [0.5, 1.0], [0.0, 1.0]]
    regions = [{"type": "detect", "points": left}]
    a, b = _det("x", 100, 100), _det("y", 900, 100)
    assert filter_by_roi([a, b], regions, 1000, 1000) == [a]
