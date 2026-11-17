"""Triton 推論器實作。

與 MockPredictor 實作同一個 Predictor 介面，因此上層（業務層、
推流、前端）完全不需改動 —— 這是「介面先定死，實作可替換」的實例。

⚠️ 分割模型（-seg）的輸出結構（官方文件確認）：
    output0  偵測結果 + mask 係數    shape (B, N, 4+nc+32)
    output1  mask prototypes         shape (B, 32, H/4, W/4)

⚠️ 融合格式（fuse_masks，見 edgevision/training/seg_fuse.py）：NMS 與 mask 合成都在 graph 裡
    output0  已 NMS 的偵測           shape (B, max_det, 6+32)  [x1,y1,x2,y2,score,cls,係數]（letterbox 座標）
    masks    每個偵測一張 mask 機率  shape (B, max_det, H/4, W/4)（sigmoid 後、未裁切）
    沒有 output1。以「有沒有 masks 輸出」辨識，走 end2end 後處理再用 masks 找輪廓。

最終遮罩要靠兩者相乘再依 bbox 裁切。這個後處理刻意放在【推論服務】
而非 Triton 內：
  ・保持 Triton 單純，只做模型推論
  ・findContours 等 CPU 運算不該佔用 GPU 服務的資源
  ・符合「業務邏輯不進 serving 層」的設計原則
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

import cv2
import numpy as np

from libs.contracts import BBox, Detection, ImageMeta
from edgevision.inference.predictor import PredictResult, Predictor, compute_letterbox

logger = logging.getLogger(__name__)

# 融合格式的 mask 輸出名稱（= seg_fuse.MASKS_OUTPUT；不 import training 模組，推論端不該依賴它）
FUSED_MASKS_OUTPUT = "masks"


class TritonPredictor(Predictor):
    """透過 Triton HTTP API 執行推論。

    版本策略：不指定版本，由 Triton 的 config 決定服務哪一版。
    版本切換由 Deployment Controller 透過 load API 的 config override
    處理 —— 推論服務永遠不碰版本號。
    """

    def __init__(
        self,
        *,
        url: str,
        model_name: str,
        imgsz: int = 640,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        class_names: list[str] | None = None,
        timeout: float = 10.0,
    ) -> None:
        try:
            import tritonclient.http as httpclient
        except ImportError as exc:
            raise RuntimeError(
                "需要 tritonclient：pip install 'tritonclient[http]'"
            ) from exc

        self._http = httpclient
        self._url = url
        self._name = model_name
        self.imgsz = imgsz
        self.conf = conf_threshold
        self.iou = iou_threshold
        self._class_names = class_names or []
        self._version = "unknown"

        self._timeout = timeout
        # ⚠️ tritonclient HTTP 底層是 gevent，client 綁定「建立它的執行緒」，
        # 跨執行緒使用會拋 greenlet.error: Cannot switch to a different thread。
        # 推論服務在背景執行緒跑 predict()，故 client 必須 per-thread 惰性建立，
        # 不能在 __init__（主執行緒）建好給別的執行緒用。
        self._local = threading.local()

        # metadata（input/output 名稱）是模型結構，一次探測即可全執行緒共用。
        # 用臨時 client 探測後關掉，避免把主執行緒的 client 留給背景執行緒。
        probe = httpclient.InferenceServerClient(
            url=url, verbose=False,
            network_timeout=timeout, connection_timeout=timeout,
        )
        try:
            self._probe_model(probe)
        finally:
            probe.close()

    @property
    def _client(self):
        """回傳「當前執行緒」專屬的 Triton client（惰性建立）。"""
        c = getattr(self._local, "client", None)
        if c is None:
            c = self._http.InferenceServerClient(
                url=self._url, verbose=False,
                network_timeout=self._timeout, connection_timeout=self._timeout,
                concurrency=4,   # 允許並發請求(MultiPredictor 併發多模型時用)
            )
            self._local.client = c
        return c

    # ── 介面實作 ──────────────────────────────────────────

    @property
    def model_name(self) -> str:
        return self._name

    @property
    def model_version(self) -> str:
        return self._version

    @property
    def is_ready(self) -> bool:
        try:
            return bool(self._client.is_model_ready(self._name))
        except Exception:
            return False

    def close(self) -> None:
        try:
            self._client.close()
        except Exception:
            pass

    def predict(self, image: np.ndarray) -> PredictResult:
        """同步推論:發射 → 等待 → 後處理。"""
        ctx = self._launch(image)
        return self._collect(ctx)

    def launch(self, image: np.ndarray):
        """發射推論請求但不等待,回傳 context。

        供 MultiPredictor 併發多模型用:先對所有模型 launch(),
        再逐一 collect() —— 在【同一執行緒】內完成,避開 tritonclient
        HTTP client 的 gevent 跨執行緒限制。
        """
        return self._launch(image)

    def collect(self, ctx) -> PredictResult:
        """等待並處理 launch() 發出的請求。"""
        return self._collect(ctx)

    def _launch(self, image: np.ndarray) -> dict:
        h, w = image.shape[:2]
        lb = compute_letterbox(w, h, self.imgsz)
        blob = self._preprocess(image, lb)

        inp = self._http.InferInput(self._input_name, blob.shape, "FP32")
        inp.set_data_from_numpy(blob)
        want = list(self._output_names)
        outputs = [self._http.InferRequestedOutput(n) for n in want]

        t = time.perf_counter()
        # async_infer 回傳 future,不阻塞 —— 多模型可同時在飛
        future = self._client.async_infer(self._name, [inp], outputs=outputs)
        return {"future": future, "lb": lb, "w": w, "h": h, "t0": t}

    def _collect(self, ctx: dict) -> PredictResult:
        response = ctx["future"].get_result()
        infer_ms = (time.perf_counter() - ctx["t0"]) * 1000

        t = time.perf_counter()
        raw = {n: response.as_numpy(n) for n in self._output_names}
        detections = self._postprocess(raw, ctx["lb"], ctx["w"], ctx["h"])
        post_ms = (time.perf_counter() - t) * 1000

        return PredictResult(
            detections=detections,
            image_meta=ImageMeta(width=ctx["w"], height=ctx["h"], letterbox=ctx["lb"]),
            inference_ms=infer_ms,
            postprocess_ms=post_ms,
        )

    # ── 內部 ──────────────────────────────────────────────

    def _probe_model(self, client) -> None:
        """讀取模型 metadata，取得輸入輸出名稱與版本。

        不寫死 "images"/"output0" —— 不同匯出設定的名稱可能不同，
        向 Triton 查詢比較可靠。用傳入的（探測用）client，不碰 thread-local。
        """
        meta: dict[str, Any] = client.get_model_metadata(self._name)
        self._input_name = meta["inputs"][0]["name"]

        # 依 shape 辨識偵測輸出與 prototype 輸出,不靠名稱順序 ——
        # 匯出工具產生的輸出順序不保證固定。
        # 偵測輸出是 3 維(B,feat,anchors)或 2 維;prototype 是 4 維(B,32,H,W)。
        out_meta = meta["outputs"]
        self._det_output = None
        self._proto_output = None
        self._mask_output = None
        self._end2end = False
        if any(o["name"] == FUSED_MASKS_OUTPUT for o in out_meta):
            # 融合格式：masks 也是 4 維，不能照下面「4 維 = prototypes」的規則認
            self._mask_output = FUSED_MASKS_OUTPUT
            out_meta = [o for o in out_meta if o["name"] != FUSED_MASKS_OUTPUT]
            self._end2end = True       # output0 已 NMS，跟 YOLO26 同一條後處理
        for o in out_meta:
            shape = o.get("shape", [])
            ndim = len([d for d in shape])
            if ndim >= 4:
                self._proto_output = o["name"]
            elif self._det_output is None:
                self._det_output = o["name"]
                if self._mask_output:
                    continue
                # YOLO26 這類 NMS-free 模型：輸出最後一維固定 6 = [x1,y1,x2,y2,conf,cls]；
                # grid 模型最後一維是 anchor 數(8400)。據此判斷後處理走哪條路。
                self._end2end = bool(shape) and int(shape[-1]) == 6
        if self._det_output is None:
            self._det_output = out_meta[0]["name"]
        # 相容既有欄位
        self._output_names = [n for n in (self._det_output, self._proto_output, self._mask_output) if n]

        versions = meta.get("versions") or ["1"]
        self._version = str(versions[0])

        if not self._class_names:
            self._load_class_names(client)

        logger.info(
            "[%s] Triton 模型就緒 v%s | 輸入 %s | 輸出 %s | 類別 %s",
            self._name, self._version, self._input_name,
            self._output_names, self._class_names or "(未知)",
        )

    def _load_class_names(self, client=None) -> None:
        """從 config.pbtxt 的 metadata 參數讀取類別名稱。

        ⚠️ Ultralytics 匯出時產生的 metadata 必須寫入 config.pbtxt。
        遺漏不會報錯，但類別對應會全錯 —— 這是官方文件特別提醒的陷阱。
        """
        try:
            cfg = (client or self._client).get_model_config(self._name)
            root = cfg.get("config") or cfg
            params = root.get("parameters") or {}
            blob = (params.get("metadata") or {}).get("string_value", "")
            if not blob:
                # HTTP API 有時不回 parameters,退而讀 config.pbtxt 原始檔
                blob = self._read_metadata_from_file()
            if not blob:
                raise ValueError("config 中沒有 metadata 參數")
            import ast
            parsed = ast.literal_eval(blob)
            names = parsed.get("names") if isinstance(parsed, dict) else None
            if isinstance(names, dict):
                self._class_names = [names[k] for k in sorted(names, key=lambda x: int(x))]
            elif isinstance(names, list):
                self._class_names = list(names)
        except Exception as exc:
            logger.warning(
                "[%s] 無法取得類別名稱（%s），將以索引代替。"
                "⚠️ 匯出時 metadata 未寫入 config.pbtxt 會導致類別對應錯誤。",
                self._name, exc,
            )

    def _read_metadata_from_file(self) -> str:
        """從 config.pbtxt 直接讀 metadata。

        Triton HTTP API 的 get_model_config 有時不回傳 parameters,
        但檔案裡有。這是備援路徑。model repository 路徑由環境變數提供。
        """
        import os, re
        repo = os.environ.get("TRITON_MODEL_REPO", "/models")
        cfg_path = os.path.join(repo, self._name, "config.pbtxt")
        try:
            text = open(cfg_path).read()
            m = re.search(r'key:\s*"metadata".*?string_value:\s*"(.*?)"',
                          text, re.DOTALL)
            return m.group(1) if m else ""
        except Exception:
            return ""

    def _preprocess(self, image: np.ndarray, lb) -> np.ndarray:
        """BGR → letterbox → RGB → CHW → 正規化。

        ⚠️ 每一步都必須與【訓練時】完全一致，否則精度會莫名下降且
        不會報錯。這是最經典的 ML bug。
        """
        h, w = image.shape[:2]
        nw, nh = round(w * lb.scale), round(h * lb.scale)
        resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)

        canvas = np.full((self.imgsz, self.imgsz, 3), 114, dtype=np.uint8)
        top, left = int(lb.pad_y), int(lb.pad_x)
        canvas[top:top + nh, left:left + nw] = resized

        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        chw = rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
        return np.expand_dims(chw, 0)

    def _postprocess(self, raw: dict, lb, w: int, h: int) -> list[Detection]:
        out0 = raw[self._det_output]
        proto = raw[self._proto_output] if self._proto_output else None

        # 去掉 batch 維（若有）。max_batch_size=0 時 Triton 不加 batch 維,
        # 輸出可能是 (features, N) 或 (1, features, N),要統一成 2D。
        preds = out0
        while preds.ndim > 2:
            preds = preds[0]
        if preds.ndim != 2:
            logger.warning("[%s] output0 維度異常: %s", self._name, out0.shape)
            return []
        if getattr(self, "_end2end", False):
            mask_name = getattr(self, "_mask_output", None)
            masks = raw.get(mask_name) if mask_name else None
            return self._postprocess_end2end(preds, lb, w, h, masks=masks)
        # Ultralytics 匯出有兩種佈局：(N, features) 或 (features, N)。
        # features 恆小於 anchor 數(8400),據此判斷是否要轉置。
        if preds.shape[0] < preds.shape[1]:
            preds = preds.T

        nc = len(self._class_names) or (preds.shape[1] - 4 - (32 if proto is not None else 0))
        n_mask = 32 if proto is not None else 0

        boxes, scores, classes, coeffs = [], [], [], []
        for row in preds:
            cls_scores = row[4:4 + nc]
            cid = int(np.argmax(cls_scores))
            score = float(cls_scores[cid])
            if score < self.conf:
                continue
            cx, cy, bw, bh = row[:4]
            boxes.append([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2])
            scores.append(score)
            classes.append(cid)
            if n_mask:
                coeffs.append(row[4 + nc:4 + nc + n_mask])

        if not boxes:
            return []

        keep = cv2.dnn.NMSBoxes(
            [[b[0], b[1], b[2] - b[0], b[3] - b[1]] for b in boxes],
            scores, self.conf, self.iou,
        )
        if len(keep) == 0:
            return []
        keep = np.array(keep).flatten()

        masks = None
        if proto is not None and coeffs:
            proto_arr = proto
            while proto_arr.ndim > 3:
                proto_arr = proto_arr[0]
            masks = self._build_masks(
                proto_arr, np.array(coeffs)[keep], np.array(boxes)[keep], lb,
            )

        results: list[Detection] = []
        for i, idx in enumerate(keep):
            bx = boxes[idx]
            x1, y1 = lb.to_original(bx[0], bx[1])
            x2, y2 = lb.to_original(bx[2], bx[3])
            det = Detection(
                class_id=classes[idx],
                class_name=(self._class_names[classes[idx]]
                            if classes[idx] < len(self._class_names)
                            else f"class_{classes[idx]}"),
                confidence=scores[idx],
                bbox=BBox(
                    x1=max(0.0, x1), y1=max(0.0, y1),
                    x2=min(float(w), x2), y2=min(float(h), y2),
                ),
            )
            if masks is not None and i < len(masks):
                contour, area = masks[i]
                if contour is not None:
                    det.has_mask = True
                    det.mask_area = area
                    det.mask_contour = contour
            results.append(det)
        return results

    def _postprocess_end2end(self, preds: np.ndarray, lb, w: int, h: int,
                             masks: np.ndarray | None = None) -> list[Detection]:
        """NMS-free（YOLO26）或融合格式：每列 [x1,y1,x2,y2,conf,cls,...]，letterbox 座標，零填充到 max_det。

        只做信心過濾。融合格式另帶 masks (max_det, H/4, W/4)，與列一一對應。
        """
        if masks is not None:
            while masks.ndim > 3:
                masks = masks[0]
        results: list[Detection] = []
        for i, row in enumerate(preds):
            score = float(row[4])
            if score < self.conf:
                continue
            cid = int(row[5])
            x1, y1 = lb.to_original(float(row[0]), float(row[1]))
            x2, y2 = lb.to_original(float(row[2]), float(row[3]))
            det = Detection(
                class_id=cid,
                class_name=(self._class_names[cid] if cid < len(self._class_names) else f"class_{cid}"),
                confidence=score,
                bbox=BBox(x1=max(0.0, x1), y1=max(0.0, y1), x2=min(float(w), x2), y2=min(float(h), y2)),
            )
            if masks is not None and i < len(masks):
                contour, area = self._mask_contour(masks[i], row[:4], lb)
                if contour is not None:
                    det.has_mask = True
                    det.mask_area = area
                    det.mask_contour = contour
            results.append(det)
        return results

    def _build_masks(self, proto, coeffs, boxes, lb):
        """由 mask 係數與 prototypes 還原遮罩輪廓。

        mask = sigmoid(coeffs @ prototypes) → 依 bbox 裁切 → 找輪廓
        """
        c, mh, mw = proto.shape
        flat = proto.reshape(c, -1)
        out = []
        for coeff, box in zip(coeffs, boxes):
            m = 1.0 / (1.0 + np.exp(-(coeff @ flat)))
            out.append(self._mask_contour(m.reshape(mh, mw), box, lb))
        return out

    def _mask_contour(self, m: np.ndarray, box, lb):
        """一張低解析 mask 機率（H/4×W/4，已 sigmoid）→ 依 bbox 裁切 → 最大輪廓（原圖座標）與面積。"""
        m = cv2.resize(m, (self.imgsz, self.imgsz), interpolation=cv2.INTER_LINEAR)

        # 只保留框內的部分：模型輸出的遮罩會溢出框外
        crop = np.zeros_like(m, dtype=np.uint8)
        x1, y1, x2, y2 = (int(max(0, min(self.imgsz, v))) for v in box)
        crop[y1:y2, x1:x2] = 1
        binary = ((m > 0.5) & (crop > 0)).astype(np.uint8) * 255

        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return None, 0.0

        largest = max(contours, key=cv2.contourArea)
        approx = cv2.approxPolyDP(largest, 0.002 * cv2.arcLength(largest, True), True)
        # 換算回原始影格座標系 —— 業務層只認 original
        pts = [list(lb.to_original(float(p[0][0]), float(p[0][1]))) for p in approx]
        area = float(cv2.contourArea(largest)) / max(lb.scale ** 2, 1e-9)
        return pts, area
