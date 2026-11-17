"""判定原語。只有機制，沒有站別的數字；站別把數字帶進來組合。"""
from edgevision_rules.geometry import (
    Point,
    Quad,
    bbox_ios,
    bbox_iou,
    bboxes_intersect,
    count_touching,
    distance,
    edge_distance_px,
    oriented_quad,
    overlap_ratio,
    polygon_area,
    quad_area,
    sort_quad,
)
from edgevision_rules.temporal import Cooldown, EventHistory, PersistCounter, WaitWindow
from edgevision_rules.zone import (
    bbox_polygon,
    expand_polygon,
    filter_by_roi,
    point_in_polygon,
    polygons_overlap,
)

__all__ = [
    "Cooldown",
    "EventHistory",
    "PersistCounter",
    "Point",
    "Quad",
    "WaitWindow",
    "bbox_ios",
    "bbox_iou",
    "bbox_polygon",
    "bboxes_intersect",
    "count_touching",
    "distance",
    "edge_distance_px",
    "expand_polygon",
    "filter_by_roi",
    "oriented_quad",
    "overlap_ratio",
    "point_in_polygon",
    "polygon_area",
    "polygons_overlap",
    "quad_area",
    "sort_quad",
]
