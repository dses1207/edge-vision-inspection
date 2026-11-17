"""區域原語：點在多邊形內、依 ROI 區域過濾偵測。

演算法與舊系統的 ROI 判斷相同：
  ・以 bbox 中心（正規化 0~1）判斷
  ・在任一 mask 區域內 → 排除（優先）
  ・有 detect 區域時，必須落在至少一個 detect 區域內，否則排除
  ・沒有 detect / mask 區域（例如只有 ref_quad）→ 全部保留
"""

from __future__ import annotations

from collections.abc import Sequence

from edgevision_contracts import Detection

Polygon = Sequence[Sequence[float]]


def point_in_polygon(x: float, y: float, polygon: Polygon) -> bool:
    """射線法。與舊系統逐字相同（含 yj==yi 時的 1e-9 防除零）。"""
    inside = False
    n = len(polygon)
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / ((yj - yi) or 1e-9) + xi):
            inside = not inside
        j = i
    return inside


def filter_by_roi(detections: list[Detection], regions: Sequence[dict], width: int, height: int) -> list[Detection]:
    """regions: [{"type": "detect"|"mask"|其他, "points": [[x, y], ...]（正規化）, "classes": [...]（選填）}, ...]

    "classes" 沒給或為空 → 這個區域對所有類別生效（原本的行為）。
    有給 → 只對列出的類別生效：其他類別當作沒有這個區域。
    例：某類別只在它自己的區域內才算數，
    另一組類別各有各的區域；一張圖一個 ROI 不夠用。

    判定用框中心點。一個類別若一個 detect 區域都沒有（所有 detect 都指定了別的類別），
    它就不受 detect 限制，只受 mask 排除。
    """
    detect = [r for r in regions if r.get("type") == "detect" and r.get("points")]
    mask = [r for r in regions if r.get("type") == "mask" and r.get("points")]
    if not detect and not mask:
        return list(detections)

    def applies(r: dict, cls: str) -> bool:
        wanted = r.get("classes")
        return not wanted or cls in wanted

    kept = []
    for d in detections:
        cx = (d.bbox.x1 + d.bbox.x2) / 2 / width
        cy = (d.bbox.y1 + d.bbox.y2) / 2 / height
        if any(point_in_polygon(cx, cy, r["points"]) for r in mask if applies(r, d.class_name)):
            continue
        my_detect = [r["points"] for r in detect if applies(r, d.class_name)]
        if my_detect and not any(point_in_polygon(cx, cy, poly) for poly in my_detect):
            continue
        kept.append(d)
    return kept


def expand_polygon(poly: Polygon, tol_px: float) -> list[tuple[float, float]]:
    """對質心外擴 tol_px（判定寬鬆一點）。tol_px<=0 原樣回傳。與舊系統 _expand_poly 相同。"""
    pts = [(float(p[0]), float(p[1])) for p in poly]
    if not tol_px or tol_px <= 0:
        return pts
    cx = sum(p[0] for p in pts) / len(pts)
    cy = sum(p[1] for p in pts) / len(pts)
    out = []
    for x, y in pts:
        dx, dy = x - cx, y - cy
        d = (dx * dx + dy * dy) ** 0.5 or 1e-9
        out.append((x + dx / d * tol_px, y + dy / d * tol_px))
    return out


def polygons_overlap(a: Polygon, b: Polygon) -> bool:
    """兩凸多邊形是否重疊（分離軸定理 SAT）。共邊視為重疊；X 型交叉（無角在內）也抓得到。

    與舊系統 _polys_overlap 逐字相同。矩形 bbox 與透視梯形區域皆凸，SAT 正確且完整。
    """
    for poly in (a, b):
        n = len(poly)
        for i in range(n):
            x1, y1 = poly[i]
            x2, y2 = poly[(i + 1) % n]
            nx, ny = -(y2 - y1), (x2 - x1)
            if nx == 0 and ny == 0:
                continue
            ap = [x * nx + y * ny for x, y in a]
            bp = [x * nx + y * ny for x, y in b]
            if max(ap) < min(bp) or max(bp) < min(ap):
                return False
    return True


def bbox_polygon(x1: float, y1: float, x2: float, y2: float) -> list[tuple[float, float]]:
    return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
