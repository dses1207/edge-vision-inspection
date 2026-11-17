"""NMS：**每個類別各自做**（Ultralytics predict 的預設 agnostic_nms=False）。

之前用 cv2.dnn.NMSBoxes 沒帶類別 → class-agnostic：不同類別的框只要重疊就互相壓掉。
手壓在板上（hand / board）、兩個動作類別互相重疊，
都會被錯誤地砍掉一個 —— 而舊系統（Ultralytics）不會。這是所有站別共用的前處理差異，2026-09-24 修正。
"""

from __future__ import annotations

import cv2
import numpy as np


def nms_keep(boxes_xyxy: np.ndarray, scores: np.ndarray, classes: np.ndarray,
             conf: float, iou: float, *, agnostic: bool = False) -> np.ndarray:
    """回要保留的索引（int 陣列，可能為空）。boxes 為 (N, 4) xyxy。"""
    if len(boxes_xyxy) == 0:
        return np.zeros(0, dtype=int)
    xywh = np.stack([boxes_xyxy[:, 0], boxes_xyxy[:, 1],
                     boxes_xyxy[:, 2] - boxes_xyxy[:, 0], boxes_xyxy[:, 3] - boxes_xyxy[:, 1]], axis=1).tolist()
    sc = [float(s) for s in scores]
    if agnostic:
        keep = cv2.dnn.NMSBoxes(xywh, sc, conf, iou)
    else:
        keep = cv2.dnn.NMSBoxesBatched(xywh, sc, [int(c) for c in classes], conf, iou)
    return np.array(keep, dtype=int).flatten() if len(keep) else np.zeros(0, dtype=int)
