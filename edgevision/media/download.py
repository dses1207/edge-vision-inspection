"""圖片集打包下載：YOLO / YOLO-seg / YOLO-pose / COCO / VOC 的 zip（前四種以外的是舊系統有的）。

跟「匯出」不同：匯出是進 LakeFS 給訓練用（只收人看過的）；這裡是給人帶走的檔案，可以選要不要含 AI 預標。
train / val 用跟匯出同一套 sha256 規則（split_of），同一張圖兩邊都落在同一側。

    yolo       images/{train,val}/<sha>.jpg、labels/{train,val}/<sha>.txt（框；多邊形取外接框）、data.yaml
    yolo-seg   同上，標籤是多邊形（框轉四點）
    yolo-pose  同上，標籤是關鍵點（只出關鍵點；data.yaml 帶 kpt_shape）
    coco       images/{train,val}/、annotations/instances_{train,val}.json（bbox＋segmentation，像素座標）
    voc        JPEGImages/、Annotations/<sha>.xml（bbox，像素座標）、ImageSets/Main/{train,val}.txt
每種都附 items.csv：檔名 ↔ 圖片 id、原始檔名、狀態、批次。
"""

from __future__ import annotations

import csv
import json
import logging
import shutil
import tempfile
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape

from edgevision.media.export import split_of
from edgevision.media.library import BUCKET, _rows
from edgevision.media.shapes import _rect_of, to_yolo_lines

logger = logging.getLogger(__name__)

FORMATS = ("yolo", "yolo-seg", "yolo-pose", "coco", "voc")
STATUSES = ("labeled", "background", "auto", "unlabeled")


def items_for(lib, cid: int, *, statuses: list[str], batch_ids: list[int] | None) -> list[dict]:
    cond, args = "", []
    if batch_ids:
        cond = " AND i.batch_id = ANY(%s)"
        args.append(list(batch_ids))
    with lib.db.cursor() as cur:
        cur.execute(f"""SELECT i.id, i.blob_sha256, i.batch_id, i.status, i.shapes, i.original_name,
                               b.bucket, b.object_key, b.width, b.height
                          FROM media_items i JOIN media_blobs b ON b.sha256 = i.blob_sha256
                         WHERE i.collection_id = %s AND i.status = ANY(%s) {cond} ORDER BY i.id""",
                    (cid, list(statuses), *args))
        return _rows(cur)


def _pts_px(s: dict, w: int, h: int) -> list[tuple[float, float]]:
    if s["type"] == "rect":
        x1, y1, x2, y2 = _rect_of(s["points"])
        pts = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
    elif s["type"] == "polygon":
        pts = [(p[0], p[1]) for p in s["points"]]
    else:
        x1, y1, x2, y2 = _rect_of(s["bbox"])
        pts = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
    return [(round(x * w, 2), round(y * h, 2)) for x, y in pts]


def _bbox_px(s: dict, w: int, h: int) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = _rect_of(s.get("bbox") or s["points"]) if s["type"] == "keypoints" else _rect_of(s["points"])
    return round(x1 * w, 2), round(y1 * h, 2), round(x2 * w, 2), round(y2 * h, 2)


def _area(pts: list[tuple[float, float]]) -> float:
    return abs(sum(pts[i][0] * pts[i - 1][1] - pts[i - 1][0] * pts[i][1] for i in range(len(pts)))) / 2


def coco_json(items: list[dict], names: list[str], files: dict[int, str]) -> dict:
    """純函式：圖（含 shapes、width、height）→ COCO instances。"""
    cats = [{"id": i + 1, "name": n, "supercategory": ""} for i, n in enumerate(names)]
    idx = {n: i + 1 for i, n in enumerate(names)}
    images, anns = [], []
    for it in items:
        w, h = int(it["width"] or 0), int(it["height"] or 0)
        images.append({"id": it["id"], "file_name": files[it["id"]], "width": w, "height": h})
        if it["status"] == "background":
            continue
        for s in it["shapes"]:
            c = idx.get(s.get("class"))
            if c is None:
                continue
            x1, y1, x2, y2 = _bbox_px(s, w, h)
            a = {"id": len(anns) + 1, "image_id": it["id"], "category_id": c, "iscrowd": 0,
                 "bbox": [x1, y1, round(x2 - x1, 2), round(y2 - y1, 2)]}
            if s["type"] == "polygon":
                pts = _pts_px(s, w, h)
                a["segmentation"] = [[v for p in pts for v in p]]
                a["area"] = round(_area(pts), 2)
            else:
                a["segmentation"] = [[x1, y1, x2, y1, x2, y2, x1, y2]]
                a["area"] = round((x2 - x1) * (y2 - y1), 2)
            if s["type"] == "keypoints":
                a["keypoints"] = [v for p in s["points"]
                                  for v in (round(p[0] * w, 2), round(p[1] * h, 2), int(p[2]) if len(p) > 2 else 2)]
                a["num_keypoints"] = sum(1 for p in s["points"] if (int(p[2]) if len(p) > 2 else 2) > 0)
            anns.append(a)
    return {"images": images, "annotations": anns, "categories": cats}


def voc_xml(it: dict, names: list[str], file_name: str) -> str:
    """純函式：一張圖 → Pascal VOC XML（只有框；多邊形取外接框）。"""
    w, h = int(it["width"] or 0), int(it["height"] or 0)
    objs = []
    if it["status"] != "background":
        for s in it["shapes"]:
            if s.get("class") not in names:
                continue
            x1, y1, x2, y2 = _bbox_px(s, w, h)
            objs.append(f"""  <object>
    <name>{escape(s['class'])}</name>
    <pose>Unspecified</pose>
    <truncated>0</truncated>
    <difficult>0</difficult>
    <bndbox><xmin>{x1:g}</xmin><ymin>{y1:g}</ymin><xmax>{x2:g}</xmax><ymax>{y2:g}</ymax></bndbox>
  </object>""")
    return (f"""<annotation>
  <folder>JPEGImages</folder>
  <filename>{escape(file_name)}</filename>
  <size><width>{w}</width><height>{h}</height><depth>3</depth></size>
  <segmented>0</segmented>
""" + "\n".join(objs) + ("\n" if objs else "") + "</annotation>\n")


def _kpt_shape(items: list[dict]) -> list[int] | None:
    """圖片集沒記 kpt_shape 時，從第一個關鍵點形狀推（點數, 3）。"""
    for it in items:
        for s in it["shapes"]:
            if s["type"] == "keypoints":
                return [len(s["points"]), 3]
    return None


def build_zip(lib, cid: int, *, fmt: str, statuses: list[str], batch_ids: list[int] | None, val_pct: int,
              out: Path, progress=None) -> dict:
    if fmt not in FORMATS:
        raise ValueError(f"格式只能是 {FORMATS}")
    col = lib.collection(cid)
    names = col["class_names"]
    items = items_for(lib, cid, statuses=statuses, batch_ids=lib.check_batches(cid, batch_ids))
    if not items:
        raise ValueError(f"沒有符合的圖（狀態 {statuses}）")
    for it in items:
        if isinstance(it["shapes"], str):
            it["shapes"] = json.loads(it["shapes"])
    tmp = Path(tempfile.mkdtemp(prefix="media-dl-"))
    try:
        splits = {it["id"]: ("val" if val_pct >= 100 else split_of(it["blob_sha256"], val_pct)) for it in items}
        files = {it["id"]: f"{it['blob_sha256']}{Path(it['object_key']).suffix or '.jpg'}" for it in items}
        yolo_task = {"yolo": "detect", "yolo-seg": "segment", "yolo-pose": "pose"}.get(fmt)
        for k, it in enumerate(items):
            split = splits[it["id"]]
            img_dir = tmp / ("JPEGImages" if fmt == "voc" else f"images/{split}")
            img_dir.mkdir(parents=True, exist_ok=True)
            lib.objects.download(bucket=it["bucket"], key=it["object_key"], dest=img_dir / files[it["id"]])
            if yolo_task:
                lbl = tmp / "labels" / split / f"{it['blob_sha256']}.txt"
                lbl.parent.mkdir(parents=True, exist_ok=True)
                lines = [] if it["status"] == "background" else to_yolo_lines(it["shapes"], names, yolo_task)
                lbl.write_text("\n".join(lines) + ("\n" if lines else ""))
            elif fmt == "voc":
                a = tmp / "Annotations" / f"{it['blob_sha256']}.xml"
                a.parent.mkdir(parents=True, exist_ok=True)
                a.write_text(voc_xml(it, names, files[it["id"]]), encoding="utf-8")
            if progress and k % 50 == 0:
                progress(f"打包中 {k}/{len(items)}")
        used = sorted(set(splits.values()))
        if yolo_task:
            yaml_lines = ["path: .", *[f"{s}: images/{s}" for s in used], f"nc: {len(names)}", "names:"]
            yaml_lines += [f"  {i}: {json.dumps(n, ensure_ascii=False)}" for i, n in enumerate(names)]
            if yolo_task == "pose" and (ks := col.get("kpt_shape") or _kpt_shape(items)):
                yaml_lines.append(f"kpt_shape: {json.dumps(ks)}")
            (tmp / "data.yaml").write_text("\n".join(yaml_lines) + "\n", encoding="utf-8")
            (tmp / "classes.txt").write_text("\n".join(names) + "\n", encoding="utf-8")
        elif fmt == "coco":
            (tmp / "annotations").mkdir(exist_ok=True)
            for s in used:
                part = [it for it in items if splits[it["id"]] == s]
                (tmp / "annotations" / f"instances_{s}.json").write_text(
                    json.dumps(coco_json(part, names, files), ensure_ascii=False), encoding="utf-8")
        else:
            (tmp / "ImageSets" / "Main").mkdir(parents=True, exist_ok=True)
            for s in used:
                (tmp / "ImageSets" / "Main" / f"{s}.txt").write_text(
                    "".join(f"{it['blob_sha256']}\n" for it in items if splits[it["id"]] == s))
            (tmp / "classes.txt").write_text("\n".join(names) + "\n", encoding="utf-8")
        with (tmp / "items.csv").open("w", newline="", encoding="utf-8-sig") as fh:
            w = csv.writer(fh)
            w.writerow(["file", "split", "item_id", "original_name", "status", "batch_id"])
            for it in items:
                w.writerow([files[it["id"]], splits[it["id"]], it["id"], it["original_name"] or "", it["status"],
                            it["batch_id"] or ""])
        with zipfile.ZipFile(out, "w", zipfile.ZIP_STORED) as z:          # 圖本來就壓過，不再壓
            for p in sorted(tmp.rglob("*")):
                if p.is_file():
                    z.write(p, p.relative_to(tmp).as_posix())
        return {"format": fmt, "items": len(items), "splits": {s: sum(v == s for v in splits.values()) for s in used},
                "size_bytes": out.stat().st_size}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def download_key(cid: int, task_id: str, fmt: str) -> tuple[str, str]:
    return BUCKET, f"downloads/collection-{cid}/{task_id}-{fmt}.zip"


def build_dataset_zip(repo: str, version_id: str, *, out: Path, progress=None) -> dict:
    """LakeFS 資料集版本原樣打包（images/、labels/、data.yaml…）。"""
    from edgevision.training.lakefs_client import lakefs_client, read_object, ref_objects

    client = lakefs_client()
    n = size = 0
    with zipfile.ZipFile(out, "w", zipfile.ZIP_STORED) as z:
        for obj in ref_objects(client, repo, version_id):
            data = read_object(client, repo, version_id, obj.path)
            z.writestr(obj.path, data)
            n += 1
            size += len(data)
            if progress and n % 200 == 0:
                progress(f"打包中 {n} 個檔")
    if n == 0:
        raise ValueError(f"{repo}@{version_id} 是空的")
    return {"format": "dataset", "files": n, "content_bytes": size, "size_bytes": out.stat().st_size}


def dataset_download_key(repo: str, task_id: str) -> tuple[str, str]:
    return BUCKET, f"downloads/dataset-{repo}/{task_id}.zip"
