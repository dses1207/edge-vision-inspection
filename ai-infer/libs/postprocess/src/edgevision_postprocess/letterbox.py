"""letterbox 參數計算。YOLO 標準前處理：等比縮到 dst×dst，短邊補灰。這個變換必須記錄，否則座標無法還原。"""

from __future__ import annotations

import numpy as np
from edgevision_contracts import LetterboxParams


def compute_letterbox(src_w: int, src_h: int, dst: int) -> LetterboxParams:
    scale = min(dst / src_w, dst / src_h)
    new_w, new_h = src_w * scale, src_h * scale
    return LetterboxParams(scale=scale, pad_x=(dst - new_w) / 2, pad_y=(dst - new_h) / 2)


def compute_letterbox_roi(x1: float, y1: float, x2: float, y2: float, dst: int) -> LetterboxParams:
    """模型只看原圖的 ROI [x1, y1, x2, y2]（像素）時的變換：先對 ROI 做 letterbox，再記住 ROI 原點。

    主站 nvinferserver 的物件模式（runner 的 roi.py）就是這件事：把 ROI 裁出來、等比縮放、補邊、餵模型。
    模型輸出的座標是「letterbox 後的 ROI」座標，to_original() 會一路換回全幅像素。
    """
    lb = compute_letterbox(round(x2 - x1), round(y2 - y1), dst)
    return LetterboxParams(scale=lb.scale, pad_x=lb.pad_x, pad_y=lb.pad_y, offset_x=float(x1), offset_y=float(y1))


def clip_to_content(boxes: np.ndarray, lb: LetterboxParams, dst: int) -> np.ndarray:
    """把 letterbox 座標的框 (N, 4 xyxy) 夾在「有畫面的區域」內（去掉補邊），回新陣列。

    = Ultralytics scale_boxes 之後的 clip_boxes(img0.shape)：框不會超出模型看到的那張圖。
    全幅時等同夾在 [0, w]×[0, h]；ROI 時等同夾在 ROI 內 —— 原程式先裁圖再推論，
    框本來就不可能超出裁切範圍（測試機實測 ROI 右緣 960，沒夾會出現 x2=990 的框）。
    NMS 之後才夾，同 Ultralytics。
    """
    out = np.array(boxes, dtype=float, copy=True)
    if out.size == 0:
        return out
    out[:, [0, 2]] = out[:, [0, 2]].clip(lb.pad_x, dst - lb.pad_x)
    out[:, [1, 3]] = out[:, [1, 3]].clip(lb.pad_y, dst - lb.pad_y)
    return out
