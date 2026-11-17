"""模型轉檔器：best.pt → model.onnx（→ model.plan 由 trtexec 步驟接手）。

分工：
    trainer 容器（本模組）  pt → onnx（Ultralytics，dynamic batch）+ pt vs onnx 一致性
                            + 產出「一致性參考」（PyTorch 對一批圖的偵測）供 TensorRT 比對
    trtexec 容器            onnx → plan（Triton 映像自帶的 trtexec，教訓：engine 必須
                            用 Triton 自己的 binary 建，否則序列化版本不相容）
    finalize 容器用臨時 Triton 載 plan，跑同一批圖 vs 參考 → TensorRT 一致性

⚠️ export 的 batch 必須 >= Triton config max_batch_size，否則載入失敗。
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path

import numpy as np

from libs.contracts.model_meta import ExportSpec

from edgevision.training.parity import ParityResult, compare

logger = logging.getLogger(__name__)

TRT_SHAPES = {"min": 1, "opt": 8}   # max = export.batch


def is_end2end(pt: Path) -> bool:
    """YOLO26 這類 NMS-free 模型（DetectionModel.end2end=True）匯出的 ONNX 直接是 (B, max_det, 6)。"""
    from ultralytics import YOLO

    m = YOLO(str(pt))
    return bool(getattr(m.model, "end2end", False))


def export_onnx(best_pt: Path, target: Path, spec: ExportSpec) -> Path:
    from ultralytics import YOLO

    kw = dict(format="onnx", imgsz=spec.imgsz, dynamic=spec.dynamic, batch=spec.batch,
              opset=spec.opset, simplify=True, verbose=False)
    if spec.end2end:
        kw["max_det"] = spec.max_det
    if spec.fuse_masks:
        # NMS 進 graph（Ultralytics 支援），輸出 (B, max_det, 38)。mask 合成它不做，
        # 由 fuse_seg_masks() 在匯出後接上。max_det 一定要小，理由見 ExportSpec。
        if spec.max_det > 64:
            raise ValueError(f"fuse_masks 時 max_det={spec.max_det} 太大：每偵測一張 mask，"
                             f"300 列是 30 MB/幀。設 16")
        kw.update(nms=True, max_det=spec.max_det)
    out = YOLO(str(best_pt)).export(**kw)
    src = Path(out)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), target)
    logger.info("ONNX 已匯出 %s（imgsz=%d dynamic=%s batch=%d opset=%d）",
                target, spec.imgsz, spec.dynamic, spec.batch, spec.opset)
    return target


def fuse_seg_masks(onnx_path: Path, spec: ExportSpec) -> dict:
    """seg 模型：把 mask 合成接進 ONNX，原地覆蓋。回傳 fuse 資訊（記進 metadata）。

    ⚠️ 要在 onnx_parity 之後跑：parity 用 Ultralytics 載 ONNX 跑它自己的後處理，
       它看不懂融合後的 masks 輸出。順序是匯出 → parity（驗轉檔）→ fuse → check（驗手術）。
    """
    from edgevision.training.seg_fuse import check, fuse

    tmp = onnx_path.with_suffix(".prefuse.onnx")
    shutil.move(str(onnx_path), str(tmp))
    info = fuse(tmp, onnx_path, imgsz=spec.imgsz)
    err = check(tmp, onnx_path)
    if err >= 1e-4:
        raise RuntimeError(f"mask 合成子圖與原始算法不一致：最大誤差 {err:.2e}")
    tmp.unlink()
    logger.info("mask 合成已進 ONNX：max_det=%d，輸出 %s，誤差 %.1e", info["max_det"], info["outputs"], err)
    return {**info, "max_abs_err": err}


def letterbox_square(img, imgsz: int):
    """與 predictor_triton._preprocess 完全相同的 letterbox（round 縮放、int 補邊、114 灰）。

    ⚠️ 一致性比對的兩邊必須看到同一批像素。Ultralytics predict() 對單張圖用「最小矩形」letterbox，
    Triton 端是正方形補邊，直接比會把前處理差異誤判成轉檔差異。所以先把圖做成 640x640 再給兩邊。
    """
    import cv2

    from edgevision.inference.predictor import compute_letterbox

    h, w = img.shape[:2]
    lb = compute_letterbox(w, h, imgsz)
    nw, nh = round(w * lb.scale), round(h * lb.scale)
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((imgsz, imgsz, 3), 114, dtype=np.uint8)
    top, left = int(lb.pad_y), int(lb.pad_x)
    canvas[top:top + nh, left:left + nw] = resized
    return canvas


def prepare_parity_images(images: list[Path], out_dir: Path, *, imgsz: int) -> list[Path]:
    """把一批圖 letterbox 成正方形存到 out_dir，回傳新路徑。兩邊（PyTorch / ONNX / Triton）都用這批。"""
    import cv2

    out_dir.mkdir(parents=True, exist_ok=True)
    out = []
    for p in images:
        img = cv2.imread(str(p))
        if img is None:
            continue
        dst = out_dir / (p.stem + ".png")     # PNG：無損，兩邊讀到的像素一致
        cv2.imwrite(str(dst), letterbox_square(img, imgsz))
        out.append(dst)
    return out


def predict_dets(model, images: list[Path], *, imgsz: int, conf: float, iou: float, device=None) -> dict[str, list[dict]]:
    """統一的偵測表示：{影像名: [{class_id, conf, bbox[x1,y1,x2,y2] 原圖像素}]}"""
    out: dict[str, list[dict]] = {}
    for img in images:
        r = model.predict(str(img), imgsz=imgsz, conf=conf, iou=iou, verbose=False, device=device)[0]
        out[img.name] = [
            {"class_id": int(c), "conf": round(float(s), 4), "bbox": [round(float(v), 2) for v in b]}
            for c, s, b in zip(r.boxes.cls.tolist(), r.boxes.conf.tolist(), r.boxes.xyxy.tolist())
        ]
    return out


def make_parity_reference(best_pt: Path, images: list[Path], parity_dir: Path, *, imgsz: int,
                          conf: float = 0.25, iou: float = 0.45, device=None) -> Path:
    """PyTorch 對「已 letterbox 的圖」的偵測結果 → parity/reference.json。finalize 步驟拿它比 TensorRT。"""
    from ultralytics import YOLO

    dets = predict_dets(YOLO(str(best_pt)), images, imgsz=imgsz, conf=conf, iou=iou, device=device)
    parity_dir.mkdir(parents=True, exist_ok=True)
    ref = parity_dir / "reference.json"
    ref.write_text(json.dumps({"imgsz": imgsz, "conf": conf, "iou": iou, "detections": dets}, indent=2))
    logger.info("一致性參考已寫入 %s（%d 張）", ref, len(images))
    return ref


def onnx_parity(best_pt: Path, onnx_path: Path, images: list[Path], *, imgsz: int,
                parity_cfg: dict, device=None, task: str = "segment") -> ParityResult:
    """pt vs onnx：兩邊都用 Ultralytics 跑（同一套後處理），差異就只來自轉檔。"""
    from ultralytics import YOLO

    conf, iou = 0.25, 0.45
    ref = predict_dets(YOLO(str(best_pt)), images, imgsz=imgsz, conf=conf, iou=iou, device=device)
    cand = predict_dets(YOLO(str(onnx_path), task=task), images, imgsz=imgsz, conf=conf, iou=iou, device="cpu")
    res = compare(ref, cand, iou_match=parity_cfg.get("iou_match", 0.8),
                  min_match_ratio=parity_cfg.get("min_match_ratio", 0.9),
                  conf_tolerance=parity_cfg.get("conf_tolerance", 0.05))
    logger.info("pt vs onnx 一致性：%s（配對 %.3f，信心差 %.3f）", res.reason, res.match_ratio, res.max_conf_diff)
    return res


def trtexec_command(onnx_path: str, plan_path: str, spec: ExportSpec) -> list[str]:
    """給 trtexec 步驟用的指令（在 Triton 映像內執行）。"""
    b = spec.batch
    s = spec.imgsz
    cmd = [
        "/usr/src/tensorrt/bin/trtexec", f"--onnx={onnx_path}", f"--saveEngine={plan_path}",
        f"--minShapes=images:{TRT_SHAPES['min']}x3x{s}x{s}",
        f"--optShapes=images:{min(TRT_SHAPES['opt'], b)}x3x{s}x{s}",
        f"--maxShapes=images:{b}x3x{s}x{s}",
    ]
    if spec.fp16:
        cmd.append("--fp16")
    return cmd
