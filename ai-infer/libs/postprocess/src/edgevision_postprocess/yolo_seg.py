"""YOLO-seg（Ultralytics 匯出）的後處理：output0 / output1 → Detection（含遮罩輪廓）。

兩種輸出格式都吃，依 out0 的 shape 自動分辨：

  raw   output0 (B, 38, 8400) + output1 (B, 32, 160, 160)
        anchor 解碼、NMS、係數×prototypes 全在這裡做。原本唯一的格式。

  fused output0 (B, max_det, 38) + masks (B, max_det, 160, 160)   ← 2026-09-24 加
        NMS 與 mask 合成都在 ONNX 裡（edgevision.training.seg_fuse），Triton 直接給每個偵測
        一張 160×160 的機率圖。每列 [x1,y1,x2,y2, score, cls, 32 係數]，padding 到 max_det
        列（score=0）。為什麼：prototypes 每幀 3.28 MB，7 路 463 MB/s，Triton 有 71~78%
        的 CPU 在搬它。合成進模型後傳輸省 61%。

  兩種格式進了 _select 之後走【完全相同】的 ROI 放大 / 找輪廓 / 面積路徑 ——
  fused 只是把「係數×prototypes」換成「直接拿現成的 mask」，數學一樣
  （seg_fuse.check 驗過誤差 0）。面積語意「原圖解析度前景像素數、洞不算」不變。

輸出結構（官方文件確認）：
    output0  偵測 + mask 係數   shape (B, 4+nc+32, N) 或 (B, N, 4+nc+32)
    output1  mask prototypes    shape (B, 32, H/4, W/4)

最終遮罩 = sigmoid(係數 @ prototypes) → 依 bbox 裁切 → 找輪廓 → 換算回原圖座標。
這段從舊 TritonPredictor._postprocess 原樣搬出，只拿掉 Triton client 相依；數值邏輯未動。
"""

from __future__ import annotations

import logging

import cv2
import numpy as np
from edgevision_contracts import BBox, Detection, LetterboxParams

from edgevision_postprocess.letterbox import clip_to_content
from edgevision_postprocess.nms import nms_keep

logger = logging.getLogger(__name__)


class YoloSegPostprocessor:
    """一個模型一個實例（類別名、門檻跟模型走）。無狀態，多執行緒可共用。"""

    def __init__(
        self,
        class_names: list[str],
        *,
        imgsz: int = 640,
        conf: float = 0.5,
        iou: float = 0.45,
        n_mask: int = 32,
    ) -> None:
        self.class_names = list(class_names)
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou
        self.n_mask = n_mask

    @staticmethod
    def is_fused(out0: np.ndarray) -> bool:
        """out0 是不是 fused 格式（(B, max_det, 38)，列數遠小於 8400 且最後一維是 features）。

        raw 是 (B, 38, 8400) 或轉置的 (8400, 38)：不管怎麼排，一定有一維是幾千。
        fused 的兩個非 batch 維都很小（max_det ≤ 64、features = 38）。
        """
        a = out0
        while a.ndim > 2:
            a = a[0]
        return a.ndim == 2 and max(a.shape) <= 512

    def __call__(
        self, out0: np.ndarray, proto: np.ndarray | None, lb: LetterboxParams, w: int, h: int
    ) -> list[Detection]:
        if self.is_fused(out0):
            return self._call_fused(out0, proto, lb, w, h)
        sel = self._select(out0, proto)
        if sel is None:
            return []
        boxes, scores, classes, coeffs, keep, proto_arr = sel
        boxes = clip_to_content(boxes, lb, self.imgsz)
        masks = None
        if proto_arr is not None and coeffs is not None and len(coeffs):
            masks = self._build_masks(proto_arr, coeffs[keep], boxes[keep], lb, w, h)
        return self._to_detections(boxes, scores, classes, keep, masks, lb, w, h)

    # ── fused 格式 ──────────────────────────────────────

    def _select_fused(self, out0: np.ndarray, masks: np.ndarray | None):
        """fused：不用解碼、不用 NMS，只要把 padding 列（score=0）與低於 conf 的濾掉。

        回傳跟 _select 同形狀的 tuple，coeffs 位置放 masks（每列一張 160×160），
        這樣 _to_detections 與 ROI 路徑一行都不用改。
        """
        rows = out0
        while rows.ndim > 2:
            rows = rows[0]
        if rows.ndim != 2 or rows.shape[1] != 4 + len(self.class_names) + self.n_mask:
            logger.warning("fused output0 維度異常: %s", out0.shape)
            return None
        scores = rows[:, 4].astype(float)
        valid = scores >= max(self.conf, 1e-6)       # padding 列 score 恰為 0
        if not valid.any():
            return None
        keep = np.nonzero(valid)[0]
        boxes = rows[:, :4]                            # 已是 xyxy（letterbox 座標）
        classes = rows[:, 5].astype(int)
        m = None
        if masks is not None:
            m = masks
            while m.ndim > 3:
                m = m[0]
        return boxes, scores, classes, m, keep, None

    def _call_fused(self, out0, masks, lb, w, h) -> list[Detection]:
        sel = self._select_fused(out0, masks)
        if sel is None:
            return []
        boxes, scores, classes, m, keep, _ = sel
        boxes = clip_to_content(boxes, lb, self.imgsz)
        out = None
        if m is not None:
            mh, mw = m.shape[-2:]
            ratio = self.imgsz / mh
            out = [self._roi_mask(m[i], boxes[i], lb, w, h, ratio, mh, mw) for i in keep]
        return self._to_detections(boxes, scores, classes, keep, out, lb, w, h)

    def decode_with_mask_crops(
        self, out0: np.ndarray, proto: np.ndarray | None, lb: LetterboxParams, w: int, h: int
    ) -> list[tuple[Detection, np.ndarray | None]]:
        """給 nvdsosd 燒錄用：每個框配一塊【對齊到框、prototype 解析度】的 float mask。

        不放大、不二值化、不找輪廓 —— 那三步是 __call__ 路徑的大頭，而 nvdsosd 的
        display-mask 會在 GPU 上自己把這塊小 mask 拉伸到 rect_params、用 threshold 切、上色
        （實測 DS 7.1：拉伸到框、顏色用 border_color、半透明）。
        probe 同步路徑只剩「每個框算一次 coeffs @ proto 的子區域」，量級是幾十微秒。

        對齊：mask 會被 nvdsosd 拉伸到【正好】rect_params 的範圍，所以小塊必須正好對應框
        （warpAffine 做次像素對齊）。差半格 proto 像素，在 1280 寬的原圖上就是 8px。
        """
        if self.is_fused(out0):
            sel = self._select_fused(out0, proto)
            if sel is None:
                return []
            boxes, scores, classes, m, keep, _ = sel
            boxes = clip_to_content(boxes, lb, self.imgsz)
            dets = self._to_detections(boxes, scores, classes, keep, None, lb, w, h)
            if m is None:
                return [(d, None) for d in dets]
            return list(zip(dets, self._mask_crops_fused(m[keep], dets, lb), strict=True))
        sel = self._select(out0, proto)
        if sel is None:
            return []
        boxes, scores, classes, coeffs, keep, proto_arr = sel
        boxes = clip_to_content(boxes, lb, self.imgsz)
        dets = self._to_detections(boxes, scores, classes, keep, None, lb, w, h)
        if proto_arr is None or coeffs is None or not len(coeffs):
            return [(d, None) for d in dets]
        return list(zip(dets, self._mask_crops(proto_arr, coeffs[keep], dets, lb), strict=True))

    # ── 共用的解碼步驟 ──────────────────────────────────

    def _select(self, out0: np.ndarray, proto: np.ndarray | None):
        """anchor 解碼 + 信心篩選 + NMS。回 (boxes, scores, classes, coeffs, keep, proto_arr)；沒東西回 None。"""
        preds = out0
        while preds.ndim > 2:           # 去 batch 維
            preds = preds[0]
        if preds.ndim != 2:
            logger.warning("output0 維度異常: %s", out0.shape)
            return None
        if preds.shape[0] < preds.shape[1]:   # (features, N) → (N, features)；features 恆小於 anchor 數
            preds = preds.T

        nc = len(self.class_names)
        n_mask = self.n_mask if proto is not None else 0

        # ⚠️ 這一段必須向量化，不能逐列跑 Python 迴圈。
        # 實測（8400 anchors）：迴圈 13.5ms vs 向量化 0.18ms —— 快 76 倍。
        # 20 路一個 batch 的話，迴圈版要 270ms，遠超 frame budget（15fps 只有 66.7ms），
        # 會讓「在 probe 裡同步解 bbox 給 nvdsosd 畫」這條官方路徑完全不可行。
        cls_scores = preds[:, 4:4 + nc]
        cid_all = cls_scores.argmax(axis=1)
        score_all = cls_scores[np.arange(len(cls_scores)), cid_all]
        mask_keep = score_all >= self.conf          # 一次篩掉絕大多數 anchor
        if not mask_keep.any():
            return None

        sel = preds[mask_keep]
        scores = score_all[mask_keep].astype(float)
        classes = cid_all[mask_keep]
        cx, cy, bw, bh = sel[:, 0], sel[:, 1], sel[:, 2], sel[:, 3]
        boxes = np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], axis=1)
        coeffs = sel[:, 4 + nc:4 + nc + n_mask] if n_mask else None

        keep = nms_keep(boxes, scores, classes, self.conf, self.iou)   # 每類別各自 NMS（同 Ultralytics 預設）
        if len(keep) == 0:
            return None

        proto_arr = proto
        if proto_arr is not None:
            while proto_arr.ndim > 3:
                proto_arr = proto_arr[0]
        return boxes, scores, classes, coeffs, keep, proto_arr

    def _to_detections(self, boxes, scores, classes, keep, masks, lb: LetterboxParams, w: int, h: int
                       ) -> list[Detection]:
        nc = len(self.class_names)
        results: list[Detection] = []
        for i, idx in enumerate(keep):
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
            if masks is not None and i < len(masks):
                contour, area = masks[i]
                if contour is not None:
                    det.has_mask = True
                    det.mask_area = area
                    det.mask_contour = contour
            results.append(det)
        return results

    def _mask_crops(self, proto_arr, coeffs, dets: list[Detection], lb: LetterboxParams
                    ) -> list[np.ndarray | None]:
        """每個框一塊 float32 mask（prototype 解析度、次像素對齊到框）。構不成區域的回 None。

        只算框內那一小塊的 coeffs @ proto —— 不算整張 160×160。一個物件約 75×50 格，
        比全圖 25600 格少一個量級；probe 是同步路徑，這裡每一微秒都乘上相機數 × fps。
        """
        c, mh, mw = proto_arr.shape
        ratio = self.imgsz / mh
        out: list[np.ndarray | None] = []
        for coef, d in zip(coeffs, dets, strict=True):
            b = d.bbox
            px1, py1 = lb.to_letterboxed(b.x1, b.y1)
            px2, py2 = lb.to_letterboxed(b.x2, b.y2)
            bx1, by1, bx2, by2 = px1 / ratio, py1 / ratio, px2 / ratio, py2 / ratio   # proto 座標（浮點）
            if bx2 <= bx1 or by2 <= by1:
                out.append(None)
                continue
            # 多留 1 格給 warpAffine 的邊緣內插
            sx1, sy1 = max(0, int(np.floor(bx1)) - 1), max(0, int(np.floor(by1)) - 1)
            sx2, sy2 = min(mw, int(np.ceil(bx2)) + 1), min(mh, int(np.ceil(by2)) + 1)
            if sx2 - sx1 < 1 or sy2 - sy1 < 1:
                out.append(None)
                continue
            sub = proto_arr[:, sy1:sy2, sx1:sx2].reshape(c, -1)
            m = (1.0 / (1.0 + np.exp(-(coef @ sub)))).reshape(sy2 - sy1, sx2 - sx1).astype(np.float32)
            ow, oh = max(1, int(np.ceil(bx2 - bx1))), max(1, int(np.ceil(by2 - by1)))
            # dst(u, v) → src：x = bx1 - sx1 + u·(bx2-bx1)/ow；y 同理。WARP_INVERSE_MAP 就是這個方向。
            aff = np.array([[(bx2 - bx1) / ow, 0.0, bx1 - sx1],
                            [0.0, (by2 - by1) / oh, by1 - sy1]], np.float32)
            crop = cv2.warpAffine(m, aff, (ow, oh), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                                  borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
            out.append(np.ascontiguousarray(crop, dtype=np.float32))
        return out

    def _mask_crops_fused(self, masks: np.ndarray, dets: list[Detection], lb: LetterboxParams
                          ) -> list[np.ndarray | None]:
        """fused 版的 _mask_crops：mask 已經是 160×160 機率圖，只做次像素對齊裁切，不乘。

        跟 _mask_crops 唯一的差別是少了 coef @ sub 那一行，warpAffine 的對齊完全相同。
        """
        mh, mw = masks.shape[-2:]
        ratio = self.imgsz / mh
        out: list[np.ndarray | None] = []
        for m160, d in zip(masks, dets, strict=True):
            b = d.bbox
            px1, py1 = lb.to_letterboxed(b.x1, b.y1)
            px2, py2 = lb.to_letterboxed(b.x2, b.y2)
            bx1, by1, bx2, by2 = px1 / ratio, py1 / ratio, px2 / ratio, py2 / ratio
            if bx2 <= bx1 or by2 <= by1:
                out.append(None)
                continue
            sx1, sy1 = max(0, int(np.floor(bx1)) - 1), max(0, int(np.floor(by1)) - 1)
            sx2, sy2 = min(mw, int(np.ceil(bx2)) + 1), min(mh, int(np.ceil(by2)) + 1)
            if sx2 - sx1 < 1 or sy2 - sy1 < 1:
                out.append(None)
                continue
            m = np.ascontiguousarray(m160[sy1:sy2, sx1:sx2], dtype=np.float32)
            ow, oh = max(1, int(np.ceil(bx2 - bx1))), max(1, int(np.ceil(by2 - by1)))
            aff = np.array([[(bx2 - bx1) / ow, 0.0, bx1 - sx1],
                            [0.0, (by2 - by1) / oh, by1 - sy1]], np.float32)
            crop = cv2.warpAffine(m, aff, (ow, oh), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                                  borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
            out.append(np.ascontiguousarray(crop, dtype=np.float32))
        return out

    # ROI 放大時在 prototype 解析度上多留的邊（proto 像素）。
    # 為什麼要留：在低解析度先硬裁，雙線性插值會把邊緣糊到框外（Ultralytics issue #24272）。
    # 先留邊放大、放大後才做精確 bbox 裁切 —— 同時有「裁切在前」的成本與「裁切在後」的品質。
    ROI_MARGIN = 2

    def _build_masks(self, proto, coeffs, boxes, lb: LetterboxParams, w: int, h: int):
        """由 mask 係數與 prototypes 還原遮罩，回 [(contour_original_coords, area_px), ...]。

        ⭐ 只放大 bbox 那一塊，不放大整張畫面（2026-09-09 改）
          舊做法每個偵測都把 160×160 放大到整張 1280×720 再找輪廓，成本與「畫面面積」成正比；
          改成在 160 上取出 bbox 區域、只放大那一塊，成本與「物件面積」成正比。
          正式機實測：每幀 7.08 → 1.16 ms（快 6.1 倍），面積差平均 0.01%、最大 0.06%。

        輸出語意與舊系統保持一致（判定門檻是拿它的輸出調出來的）：
          ・mask_area = 前景像素數（原圖解析度）—— 洞不算面積。手壓在板上時板遮罩會被挖洞，
            面積因此變小；動作的面積比門檻就是在這種數值上定的，不能改用外輪廓面積
          ・mask_contour = 最大外輪廓經 approxPolyDP(0.002×周長)，座標已是原圖
          ・sigmoid 保留：我們是「放大後才閾值」，而 sigmoid 非線性，
            改成比 logits > 0 會讓邊界差 0.1~0.2% 像素，且實測沒有變快
        """
        c, mh, mw = proto.shape
        # 批次：一次矩陣乘算完所有偵測（比逐個迴圈快，官方 process_mask 也是這樣）
        masks = 1.0 / (1.0 + np.exp(-(np.asarray(coeffs) @ proto.reshape(c, -1))))
        masks = masks.reshape(-1, mh, mw)
        ratio = self.imgsz / mh                      # prototype → letterbox 座標的比例
        out = []
        for m160, box in zip(masks, boxes):
            out.append(self._roi_mask(m160, box, lb, w, h, ratio, mh, mw))
        return out

    def _roi_mask(self, m160, box, lb: LetterboxParams, w: int, h: int, ratio: float, mh: int, mw: int):
        """單一偵測：在 prototype 解析度取 ROI → 只放大這一塊 → 精確裁切 → 面積與輪廓。"""
        ox1, oy1 = lb.to_original(float(box[0]), float(box[1]))
        ox2, oy2 = lb.to_original(float(box[2]), float(box[3]))
        x1, y1 = int(max(0, min(w, ox1))), int(max(0, min(h, oy1)))
        x2, y2 = int(max(0, min(w, ox2))), int(max(0, min(h, oy2)))
        if x2 <= x1 or y2 <= y1:
            return (None, 0.0)

        # 原圖框 → prototype 座標，取出含 margin 的區塊
        px1, py1 = lb.to_letterboxed(float(x1), float(y1))
        px2, py2 = lb.to_letterboxed(float(x2), float(y2))
        sx1 = max(0, int(np.floor(px1 / ratio)) - self.ROI_MARGIN)
        sy1 = max(0, int(np.floor(py1 / ratio)) - self.ROI_MARGIN)
        sx2 = min(mw, int(np.ceil(px2 / ratio)) + self.ROI_MARGIN)
        sy2 = min(mh, int(np.ceil(py2 / ratio)) + self.ROI_MARGIN)
        if sx2 <= sx1 or sy2 <= sy1:
            return (None, 0.0)
        sub = m160[sy1:sy2, sx1:sx2]

        # 這一塊在原圖對應的範圍
        rx1, ry1 = lb.to_original(sx1 * ratio, sy1 * ratio)
        rx2, ry2 = lb.to_original(sx2 * ratio, sy2 * ratio)
        rw, rh = round(rx2 - rx1), round(ry2 - ry1)
        if rw < 1 or rh < 1:
            return (None, 0.0)
        big = cv2.resize(sub, (rw, rh), interpolation=cv2.INTER_LINEAR)
        binary = big > 0.5

        # 放大之後才做精確 bbox 裁切（見 ROI_MARGIN 的說明）
        cx1, cy1 = int(max(0, round(x1 - rx1))), int(max(0, round(y1 - ry1)))
        cx2, cy2 = int(min(rw, round(x2 - rx1))), int(min(rh, round(y2 - ry1)))
        if cx2 <= cx1 or cy2 <= cy1:
            return (None, 0.0)
        keep = np.zeros_like(binary)
        keep[cy1:cy2, cx1:cx2] = True
        binary &= keep

        area = float(int(binary.sum()))
        contours, _ = cv2.findContours(binary.astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return (None, 0.0)
        largest = max(contours, key=cv2.contourArea)
        approx = cv2.approxPolyDP(largest, 0.002 * cv2.arcLength(largest, True), True)
        pts = approx.reshape(-1, 2).astype(float)
        pts[:, 0] += rx1                              # ROI 內座標 → 原圖座標
        pts[:, 1] += ry1
        return (pts.tolist(), area)
