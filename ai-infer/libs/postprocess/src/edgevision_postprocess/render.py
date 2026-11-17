"""把 Detection 畫到影像上 —— 只給**存檔**用，不在即時路徑上。

⚠️ 為什麼直播畫不了 mask，這裡卻可以：
  直播的框由 nvdsosd 燒錄，meta 必須在 buffer 流到 nvdsosd 之前掛上，
  所以 probe 只能同步解 bbox（bbox_only 快 50 倍）。mask 是 worker 非同步重建的，
  算完時該幀早就流過 nvdsosd 了 —— 不是慢，是**追不上時序**。
  存檔沒有這個限制：worker 已經把多邊形算好放在 Detection.mask_contour 裡，
  這裡只是在它自己的執行緒上把座標畫成像素。

座標契約：mask_contour 與 bbox 都是**原圖座標**（worker 還原 letterbox 後才寫入），
所以這裡直接畫，不做任何座標轉換。
"""

from __future__ import annotations

import cv2
import numpy as np
from edgevision_contracts import Detection

from edgevision_postprocess.colors import box_color_bgr

MASK_ALPHA = 0.45


def color_for(class_name: str, class_id: int | None = None) -> tuple[int, int, int]:
    """與直播同一套色（edgevision_postprocess.colors）。"""
    return box_color_bgr(class_name, class_id)


def draw_detections(
    image: np.ndarray,
    detections: list[Detection],
    *,
    draw_masks: bool = True,
    draw_boxes: bool = True,
    draw_labels: bool = True,
    alpha: float = MASK_ALPHA,
) -> np.ndarray:
    """回傳畫好的**新影像**，不動到傳進來的那張。

    呼叫端（sink）拿到的 image 可能還要拿去做別的事，就地改會有隱形的副作用。
    """
    canvas = image.copy()
    if not detections:
        return canvas

    if draw_masks:
        canvas = _blend_masks(canvas, detections, alpha)

    for det in detections:
        c = color_for(det.class_name, det.class_id)
        if draw_boxes:
            p1 = (round(det.bbox.x1), round(det.bbox.y1))
            p2 = (round(det.bbox.x2), round(det.bbox.y2))
            cv2.rectangle(canvas, p1, p2, c, 2)
        if draw_labels:
            _put_label(canvas, f"{det.class_name} {det.confidence:.2f}",
                       round(det.bbox.x1), round(det.bbox.y1), c)
    return canvas


def _contour_array(det: Detection, w: int, h: int) -> np.ndarray | None:
    """把 mask_contour 轉成 cv2 要的 int32 點陣；點數不足以構成多邊形就回 None。"""
    if not det.has_mask or not det.mask_contour:
        return None
    pts = np.asarray(det.mask_contour, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[0] < 3 or pts.shape[1] < 2:
        return None
    pts = np.round(pts[:, :2]).astype(np.int32)
    # 輪廓來自原圖座標，但 bbox 裁切與四捨五入可能讓端點差一兩個像素而落在界外
    np.clip(pts[:, 0], 0, w - 1, out=pts[:, 0])
    np.clip(pts[:, 1], 0, h - 1, out=pts[:, 1])
    return pts


def _blend_masks(canvas: np.ndarray, detections: list[Detection], alpha: float) -> np.ndarray:
    """半透明填色。

    ⭐ 一次 addWeighted 混合全部，而不是每個 det 混一次：
       逐一混合會讓重疊處（手壓在板上）被混兩次而變得更深，
       深淺就不再代表類別，看起來像另一種東西。
    """
    h, w = canvas.shape[:2]
    overlay = canvas.copy()
    drew = False
    for det in detections:
        pts = _contour_array(det, w, h)
        if pts is None:
            continue
        cv2.fillPoly(overlay, [pts], color_for(det.class_name, det.class_id))
        drew = True
    if not drew:
        return canvas
    blended = cv2.addWeighted(overlay, alpha, canvas, 1.0 - alpha, 0.0)
    # 邊界描實線：半透明填色的邊緣很模糊，實線讓遮罩範圍一眼可見
    for det in detections:
        pts = _contour_array(det, w, h)
        if pts is not None:
            cv2.polylines(blended, [pts], True, color_for(det.class_name, det.class_id), 2)
    return blended


def _put_label(canvas: np.ndarray, text: str, x: int, y: int, color) -> None:
    """標籤畫在框上緣；貼到畫面頂端時改畫在框內，避免被裁掉。"""
    font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
    (tw, th), base = cv2.getTextSize(text, font, scale, thick)
    top = y - th - base
    if top < 0:
        top = y
    x = max(0, min(x, canvas.shape[1] - tw - 1))
    cv2.rectangle(canvas, (x, top), (x + tw, top + th + base), color, -1)
    cv2.putText(canvas, text, (x, top + th), font, scale, (20, 20, 20), thick, cv2.LINE_AA)
