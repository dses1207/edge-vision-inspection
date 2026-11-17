"""YOLO-pose（與純 detect）的後處理：output0 → Detection（含 keypoints）。

Ultralytics pose 匯出的 output0 shape (B, 4 + nc + n_kpt*3, N)：
    [cx, cy, w, h, cls_0..cls_nc-1, kx_0, ky_0, kc_0, …]
n_kpt = 0 時就是純 detect 模型（4 + nc）。
keypoint 座標與 bbox 一樣在 letterbox 空間，這裡一起還原回原圖；conf 原樣保留。
與舊系統的輸出形狀相同：[[x, y, conf], …]（像素）。
"""

from __future__ import annotations

import logging

import numpy as np
from edgevision_contracts import BBox, Detection, LetterboxParams

from edgevision_postprocess.letterbox import clip_to_content
from edgevision_postprocess.nms import nms_keep

logger = logging.getLogger(__name__)


class YoloPosePostprocessor:
    def __init__(self, class_names: list[str], *, n_kpt: int = 4, imgsz: int = 640,
                 conf: float = 0.5, iou: float = 0.45) -> None:
        self.class_names = list(class_names)
        self.n_kpt = n_kpt
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou

    def __call__(self, out0: np.ndarray, proto: np.ndarray | None, lb: LetterboxParams, w: int, h: int) -> list[Detection]:
        preds = out0
        while preds.ndim > 2:
            preds = preds[0]
        if preds.ndim != 2:
            logger.warning("output0 維度異常: %s", out0.shape)
            return []
        if preds.shape[0] < preds.shape[1]:
            preds = preds.T
        nc = len(self.class_names)
        expected = 4 + nc + self.n_kpt * 3
        if preds.shape[1] != expected:
            logger.warning("output0 特徵數 %d ≠ 預期 %d（4 + %d 類 + %d 點×3）", preds.shape[1], expected, nc, self.n_kpt)

        # 向量化篩選（同 yolo_seg 的理由）：逐列跑 Python 迴圈在 8400 anchors 下要 13.5ms，
        # 20 路一批就 270ms，遠超 frame budget，會讓 probe 同步畫框不可行。
        cls_scores = preds[:, 4:4 + nc]
        cid_all = cls_scores.argmax(axis=1)
        score_all = cls_scores[np.arange(len(cls_scores)), cid_all]
        m = score_all >= self.conf
        if not m.any():
            return []
        sel = preds[m]
        scores = score_all[m].astype(float)
        classes = cid_all[m]
        cx, cy, bw, bh = sel[:, 0], sel[:, 1], sel[:, 2], sel[:, 3]
        boxes = np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], axis=1)
        kpts = (sel[:, 4 + nc:4 + nc + self.n_kpt * 3].reshape(-1, self.n_kpt, 3)
                if self.n_kpt else None)

        keep = nms_keep(boxes, scores, classes, self.conf, self.iou)   # 每類別各自 NMS（同 Ultralytics 預設）
        if len(keep) == 0:
            return []
        boxes = clip_to_content(boxes, lb, self.imgsz)

        results: list[Detection] = []
        for idx in keep:
            bx = boxes[idx]
            x1, y1 = lb.to_original(bx[0], bx[1])
            x2, y2 = lb.to_original(bx[2], bx[3])
            cid = int(classes[idx])
            det = Detection(
                class_id=cid,
                class_name=self.class_names[cid] if cid < nc else f"class_{cid}",
                confidence=float(scores[idx]),
                bbox=BBox(x1=max(0.0, x1), y1=max(0.0, y1), x2=min(float(w), x2), y2=min(float(h), y2)),
            )
            if self.n_kpt and kpts is not None:
                det.keypoints = [[*lb.to_original(float(k[0]), float(k[1])), float(k[2])] for k in kpts[idx]]
            results.append(det)
        return results
