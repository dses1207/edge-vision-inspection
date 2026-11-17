"""幾何工具：分割遮罩的重疊、距離、四角比對。

演算法來自既有系統的實戰驗證，每個選擇都有實測依據，
不要「簡化」成看似等價的做法：

  ・重疊用【凸包】而非原始輪廓
    手壓在板上時，板的遮罩會被挖出凹洞。不補平會判成「沒有重疊」。

  ・重疊算【target 面積佔比】而非 IoU
    手掛在板邊緣時，手的中心常落在板外；用 IoU 會因為板面積遠大於手
    而永遠很小。用「手有多少比例在板上」才有鑑別度。

  ・接觸判定用【重疊 or 距離】的聯集
    板立起來時投影成細長條，面積重疊必然失效。
    實測距離能把「貼著」與「沒碰到」乾淨分開。
    兩個量度互補：板攤平時重疊有效，板立起時距離有效。

  ・四角用【minAreaRect】而非軸對齊 bbox
    板傾斜時，軸對齊的四角會嚴重失真。
"""

from __future__ import annotations

import cv2
import numpy as np
from edgevision_contracts import BBox, Detection

Point = tuple[float, float]
Quad = tuple[Point, Point, Point, Point]


def _contour_array(det: Detection) -> np.ndarray | None:
    if not det.has_mask or not det.mask_contour or len(det.mask_contour) < 3:
        return None
    return np.array(det.mask_contour, dtype=np.float32)


def _hull_or_bbox(det: Detection) -> np.ndarray:
    """取得物件的凸包多邊形；無遮罩時退回 bbox 四角。"""
    c = _contour_array(det)
    if c is not None:
        return cv2.convexHull(c)
    b = det.bbox
    return np.array(
        [[b.x1, b.y1], [b.x2, b.y1], [b.x2, b.y2], [b.x1, b.y2]],
        dtype=np.float32,
    )


def bboxes_intersect(a: BBox, b: BBox) -> bool:
    """快速排除：兩個框完全沒有交集時，不必做昂貴的多邊形運算。"""
    return not (a.x2 <= b.x1 or b.x2 <= a.x1 or a.y2 <= b.y1 or b.y2 <= a.y1)


def bbox_ios(a: BBox, b: BBox) -> float:
    """IoS：交集 / 較小者的面積。anchor 無遮罩時的退路（同舊系統 _bbox_ios）。"""
    ix1, iy1 = max(a.x1, b.x1), max(a.y1, b.y1)
    ix2, iy2 = min(a.x2, b.x2), min(a.y2, b.y2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    smaller = min(a.area, b.area)
    return inter / smaller if smaller > 0 else 0.0


def overlap_ratio(anchor: Detection, target: Detection) -> float:
    """target 有多少比例的自身面積落在 anchor 內。0 表示完全沒疊。

    anchor 有遮罩 → target 框 ∩ anchor 凸包 / target 面積；
    anchor 無遮罩 → 退回 bbox IoS（舊系統 _overlap_ratio 的分支，兩者算法不同，不可合併）。
    """
    if not bboxes_intersect(anchor.bbox, target.bbox):
        return 0.0
    if _contour_array(anchor) is None:
        return bbox_ios(anchor.bbox, target.bbox)

    hull = _hull_or_bbox(anchor)
    tb = target.bbox
    rect = np.array(
        [[tb.x1, tb.y1], [tb.x2, tb.y1], [tb.x2, tb.y2], [tb.x1, tb.y2]],
        dtype=np.float32,
    )
    try:
        inter_area, _ = cv2.intersectConvexConvex(hull, rect)
    except cv2.error:
        return 0.0
    return float(inter_area) / max(tb.area, 1.0)


def edge_distance_px(anchor: Detection, target: Detection) -> float:
    """target 到 anchor 輪廓的最短距離（像素）。重疊或在內部時回 0。

    取 target 框的四角加中心，各自對 anchor 多邊形求距離，取最小值。
    """
    hull = _hull_or_bbox(anchor)
    tb = target.bbox
    cx, cy = tb.center
    probes = [
        (tb.x1, tb.y1), (tb.x2, tb.y1), (tb.x2, tb.y2), (tb.x1, tb.y2), (cx, cy),
    ]
    best = float("inf")
    for px, py in probes:
        # 正值=在內部，負值=在外部，回傳的是有號距離
        d = cv2.pointPolygonTest(hull, (float(px), float(py)), True)
        if d >= 0:
            return 0.0
        best = min(best, -d)
    return best


def count_touching(
    anchor: Detection,
    targets: list[Detection],
    *,
    min_overlap: float = 0.15,
    max_distance_px: float = 0.0,
) -> int:
    """數有多少 target 與 anchor 接觸。

    接觸 = 重疊比 ≥ min_overlap **或** 距離 ≤ max_distance_px。
    max_distance_px = 0 表示停用距離判定。
    """
    n = 0
    for t in targets:
        if t is anchor:
            continue
        if overlap_ratio(anchor, t) >= min_overlap:
            n += 1
        elif max_distance_px > 0:
            d = edge_distance_px(anchor, t)
            if 0 <= d <= max_distance_px:
                n += 1
    return n


def sort_quad(points: list[Point] | np.ndarray) -> Quad:
    """把四個點排成 (左上, 右上, 左下, 右下)。"""
    pts = [(float(p[0]), float(p[1])) for p in points]
    by_y = sorted(pts, key=lambda p: p[1])
    top = sorted(by_y[:2], key=lambda p: p[0])
    bottom = sorted(by_y[2:], key=lambda p: p[0])
    return top[0], top[1], bottom[0], bottom[1]


def oriented_quad(det: Detection) -> Quad:
    """取得物件的旋轉四角。

    有遮罩時用 minAreaRect —— 板傾斜時四角會跟著轉，位移才量得準。
    無遮罩時只能退回軸對齊四角（精度較差）。
    """
    c = _contour_array(det)
    if c is not None:
        box = cv2.boxPoints(cv2.minAreaRect(c))
        return sort_quad(box.tolist())
    b = det.bbox
    return (b.x1, b.y1), (b.x2, b.y1), (b.x1, b.y2), (b.x2, b.y2)


def polygon_area(points: list[Point] | Quad) -> float:
    """多邊形面積（shoelace）。

    ⚠️ 頂點必須依周長順序排列。sort_quad 回傳的是
    (左上, 右上, 左下, 右下) —— 那是「語意順序」不是「周長順序」，
    直接餵進來會沿著對角線繞、算出 0。用 quad_area() 處理四邊形。
    """
    pts = list(points)
    n = len(pts)
    if n < 3:
        return 0.0
    return 0.5 * abs(sum(
        pts[i][0] * pts[(i + 1) % n][1] - pts[(i + 1) % n][0] * pts[i][1]
        for i in range(n)
    ))


def quad_area(quad: Quad) -> float:
    """四邊形面積。輸入為 sort_quad 的語意順序，內部轉成周長順序。"""
    tl, tr, bl, br = quad
    return polygon_area([tl, tr, br, bl])


def distance(a: Point, b: Point) -> float:
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def bbox_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    """兩個 (x1, y1, x2, y2) 框的 IoU。聯集為 0 回 0。"""
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0
