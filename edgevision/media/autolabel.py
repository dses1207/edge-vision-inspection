"""自動標注引擎（在訓練映像跑：要 torch / ultralytics）。

    registry    Registry 裡的模型（預設現役版）→ 框 / 多邊形 / keypoints。舊系統的「種子模型批次重標」。
    yolo_world  提示詞 → 框。提示詞對應圖片集類別：{"person in safety vest": "vest", ...}
    sam         以圖上現有的框當提示 → 多邊形（框轉 seg）；有 sam3.pt 且給 prompts 時改用文字提示直接分割。

結果一律寫成 auto（AI 預標、沒人看過），每個 shape 記來源；預設不動人看過的圖（labeled / background）。
基礎權重放 MinIO models/_base/（yolov8x-worldv2.pt、clip/ViT-B-32.pt、sam2.1_b.pt、sam3.pt）——
叢集內不一定連得到外網，不能讓 ultralytics 執行時自己下載。
"""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)

BASE_BUCKET, BASE_PREFIX = "models", "_base"
BATCH = 8
MAX_POLY_POINTS = 120


def _fetch_base(objects, name: str, dest: Path) -> Path | None:
    """從 MinIO models/_base/ 取基礎權重；沒有就回 None。"""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return dest
    try:
        objects.download(bucket=BASE_BUCKET, key=f"{BASE_PREFIX}/{name}", dest=dest)
        return dest
    except Exception as exc:
        dest.unlink(missing_ok=True)
        logger.info("models/_base/%s 不可用：%s", name, exc)
        return None


def _poly(xy, w: int, h: int) -> list[list[float]] | None:
    """像素多邊形 → 0–1 正規化，點數過多時簡化（SAM 的輪廓可以上千點）。"""
    import cv2
    import numpy as np

    pts = np.asarray(xy, dtype=np.float32).reshape(-1, 2)
    if len(pts) < 3:
        return None
    if len(pts) > MAX_POLY_POINTS:
        eps = 0.002 * cv2.arcLength(pts.reshape(-1, 1, 2), True)
        while len(pts) > MAX_POLY_POINTS:
            pts = cv2.approxPolyDP(pts.reshape(-1, 1, 2), eps, True).reshape(-1, 2)
            eps *= 1.5
        if len(pts) < 3:
            return None
    return [[min(1.0, max(0.0, float(x) / w)), min(1.0, max(0.0, float(y) / h))] for x, y in pts]


def _src(engine: str, model: str | None, version: int | None, conf: float | None) -> dict:
    return {"kind": "model", "engine": engine, "model": model, "version": version,
            "conf": round(float(conf), 4) if conf is not None else None}


def result_to_shapes(r, names: dict, class_map: dict, *, task: str, engine: str, model: str | None,
                     version: int | None) -> list[dict]:
    """ultralytics Results → shapes。class_map：模型類別名 → 圖片集類別名（不在 map 裡的丟掉）。"""
    h, w = r.orig_shape
    out = []
    boxes = r.boxes
    if boxes is None:
        return out
    xyxy = boxes.xyxy.cpu().numpy()
    confs = boxes.conf.cpu().numpy()
    clss = boxes.cls.cpu().numpy().astype(int)
    polys = r.masks.xy if (r.masks is not None and task == "segment") else None
    kpts = r.keypoints if (r.keypoints is not None and task == "pose") else None
    for i in range(len(xyxy)):
        cls = class_map.get(names.get(int(clss[i]), str(clss[i])))
        if cls is None:
            continue
        src = _src(engine, model, version, confs[i])
        x1, y1, x2, y2 = xyxy[i]
        rect = [[max(0.0, x1 / w), max(0.0, y1 / h)], [min(1.0, x2 / w), min(1.0, y2 / h)]]
        if polys is not None:
            p = _poly(polys[i], w, h)
            if p:
                out.append({"type": "polygon", "class": cls, "points": p, "source": src})
                continue
        if kpts is not None:
            xy = kpts.xy[i].cpu().numpy()
            kc = kpts.conf[i].cpu().numpy() if kpts.conf is not None else None
            pts = [[min(1.0, max(0.0, float(x) / w)), min(1.0, max(0.0, float(y) / h)),
                    0 if (x == 0 and y == 0) else (2 if kc is None or kc[j] >= 0.5 else 1)]
                   for j, (x, y) in enumerate(xy)]
            out.append({"type": "keypoints", "class": cls, "bbox": rect, "points": pts, "source": src})
            continue
        out.append({"type": "rect", "class": cls, "points": rect, "source": src})
    return out


def _decode(data: bytes):
    import cv2
    import numpy as np
    return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)


def select_items(lib, cid: int, p: dict) -> list[dict]:
    if p.get("item_ids"):
        items = lib.items_by_ids([int(i) for i in p["item_ids"]])
        items = [i for i in items if i["collection_id"] == cid]
    else:
        status = p.get("status") or ["unlabeled", "auto"]
        batch_ids = lib.check_batches(cid, p.get("batch_ids"))
        items, after = [], 0
        while True:
            page = lib.list_items(cid, status=status, after_id=after, limit=1000, batch_ids=batch_ids)
            if not page:
                break
            items += page
            after = page[-1]["id"]
    skip = {int(i) for i in (p.get("skip_item_ids") or [])}
    return [i for i in items if i["id"] not in skip]


def run(lib, task, *, progress) -> dict:
    p = task.params
    cid = int(p["collection_id"])
    col = lib.collection(cid)
    engine = p["engine"]
    conf = float(p.get("conf", 0.25))
    class_filter = p.get("class_filter") or None
    overwrite = bool(p.get("overwrite_human", False))
    actor = task.created_by
    items = select_items(lib, cid, p)
    progress(f"自動標注（{engine}）：{len(items)} 張", force=True)
    work = Path(tempfile.mkdtemp(prefix="autolabel-"))
    os.chdir(work)        # ultralytics 的 CLIP 權重找 ./weights/clip/（_load_yolo_world 放好）
    predict, meta = _build(engine, p, col, lib, work, conf)
    written = skipped = 0
    for b in range(0, len(items), BATCH):
        if progress.cancelled():
            raise KeyboardInterrupt
        batch = items[b:b + BATCH]
        imgs = [_decode(lib.blob_bytes(it["blob_sha256"])) for it in batch]
        for it, shapes in zip(batch, predict(batch, imgs)):
            if shapes is None:
                skipped += 1
                continue
            ok = lib.write_auto(it["id"], shapes, actor=actor, task_id=task.task_id,
                                class_filter=class_filter, overwrite_human=overwrite)
            written += int(ok); skipped += int(not ok)
        progress(f"自動標注 {min(b + BATCH, len(items))}/{len(items)}（寫入 {written}）")
    return {"engine": engine, **meta, "items": len(items), "written": written, "skipped": skipped}


def _build(engine: str, p: dict, col: dict, lib, work: Path, conf: float):
    from ultralytics import YOLO

    names_col = col["class_names"]
    if engine == "registry":
        from edgevision.registry.clearml_registry import ClearMLRegistry

        reg = ClearMLRegistry(objects=lib.objects)
        name = p["model_name"]
        r = reg.get(name, int(p["model_version"])) if p.get("model_version") else reg.resolve(name)
        if r is None:
            raise ValueError(f"Registry 找不到 {name} {p.get('model_version') or '現役版'}")
        if not r.meta.artifacts.pytorch:
            raise ValueError(f"{name} v{r.version} 沒有 .pt 權重（自動標注要 PyTorch 權重）")
        b, k = lib.objects.parse_uri(r.meta.artifacts.pytorch)
        pt = work / "model.pt"
        lib.objects.download(bucket=b, key=k, dest=pt)
        model = YOLO(str(pt))
        # 模型類別名 → 圖片集類別名：預設同名對應；class_map 可改名（舊模型類別名與新規範不同時）
        cmap = {n: n for n in r.meta.class_names if n in names_col} | (p.get("class_map") or {})
        if not cmap:
            raise ValueError(f"模型類別 {r.meta.class_names} 與圖片集類別 {names_col} 沒有交集（可用 class_map 對應）")
        # 模型給得出什麼就存什麼（seg 模型 → 多邊形、pose → 關鍵點、其他 → 框）；輸出格式到匯出時才選
        task = r.meta.task if r.meta.task in ("segment", "pose") else "detect"
        imgsz = int(p.get("imgsz") or r.meta.imgsz)

        def predict(batch, imgs):
            res = model.predict(imgs, conf=conf, imgsz=imgsz, verbose=False)
            return [result_to_shapes(x, x.names, cmap, task=task, engine="registry", model=name,
                                     version=r.version) for x in res]
        return predict, {"model": name, "version": r.version, "class_map": cmap}

    if engine == "yolo_world":
        prompts = p.get("prompts") or {}
        if isinstance(prompts, list):
            prompts = {x: x for x in prompts}
        bad = [c for c in prompts.values() if c not in names_col]
        if not prompts or bad:
            raise ValueError(f"prompts 要對應到圖片集類別（{names_col}）；不在類別裡：{bad}")
        weights = p.get("weights") or "yolov8x-worldv2.pt"
        pt = _fetch_base(lib.objects, weights, work / weights)
        clip = _fetch_base(lib.objects, "clip/ViT-B-32.pt", work / "weights" / "clip" / "ViT-B-32.pt")
        if pt is None or clip is None:
            raise ValueError("MinIO models/_base/ 缺 YOLO-World 或 CLIP 權重")
        from ultralytics import YOLOWorld
        model = YOLOWorld(str(pt))
        texts = list(prompts)
        model.set_classes(texts)
        cmap = {t: prompts[t] for t in texts}
        imgsz = int(p.get("imgsz") or 640)

        def predict(batch, imgs):
            res = model.predict(imgs, conf=conf, imgsz=imgsz, verbose=False)
            return [result_to_shapes(x, {i: t for i, t in enumerate(texts)}, cmap, task="detect",
                                     engine="yolo_world", model=weights, version=None) for x in res]
        return predict, {"model": weights, "prompts": prompts}

    if engine == "sam":
        prompts = p.get("prompts") or {}
        if isinstance(prompts, list):
            prompts = {x: x for x in prompts}
        sam3 = _fetch_base(lib.objects, "sam3.pt", work / "sam3.pt") if prompts else None
        if prompts and sam3 is None:
            raise ValueError("文字提示分割需要 SAM3 權重（MinIO models/_base/sam3.pt），目前沒有；"
                             "可改用框提示（不給 prompts，以圖上現有的框產生多邊形）")
        if sam3 is not None:
            from ultralytics.models.sam import SAM3SemanticPredictor
            pred = SAM3SemanticPredictor(overrides={"model": str(sam3), "conf": conf, "task": "segment",
                                                     "mode": "predict", "verbose": False})
            texts = list(prompts)

            def predict(batch, imgs):
                out = []
                for img in imgs:
                    pred.set_image(img)
                    res = pred(text=texts)[0]
                    out.append(result_to_shapes(res, {i: t for i, t in enumerate(texts)}, prompts,
                                                task="segment", engine="sam", model="sam3", version=None))
                return out
            return predict, {"model": "sam3", "prompts": prompts}

        weights = p.get("weights") or "sam2.1_b.pt"
        pt = _fetch_base(lib.objects, weights, work / weights)
        if pt is None:
            raise ValueError(f"MinIO models/_base/{weights} 不存在")
        from ultralytics import SAM
        model = SAM(str(pt))

        def predict(batch, imgs):
            out = []
            for it, img in zip(batch, imgs):
                full = lib.item(it["id"], with_url=False)
                rects = [s for s in full["shapes"] if s["type"] == "rect"]
                if not rects:
                    out.append(None)          # 沒有框可當提示：略過
                    continue
                h, w = img.shape[:2]
                boxes = [[s["points"][0][0] * w, s["points"][0][1] * h, s["points"][1][0] * w, s["points"][1][1] * h]
                         for s in rects]
                res = model.predict(img, bboxes=boxes, verbose=False)[0]
                shapes = []
                polys = res.masks.xy if res.masks is not None else []
                for s, poly in zip(rects, polys):
                    pts = _poly(poly, w, h)
                    if pts:
                        shapes.append({"type": "polygon", "class": s["class"], "points": pts,
                                       "source": _src("sam", weights, None, None)})
                # 框轉多邊形：原本的框換掉，其他形狀保留
                keep = [s for s in full["shapes"] if s["type"] != "rect"]
                out.append(keep + shapes)
            return out
        return predict, {"model": weights, "mode": "boxes"}

    raise ValueError(f"未知的引擎：{engine}（registry | yolo_world | sam）")
