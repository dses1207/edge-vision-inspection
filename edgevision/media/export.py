"""圖片集 → YOLO 目錄 → LakeFS 資料集版本。

只匯出人看過的圖（labeled / background）。AI 預標（auto）沒人確認，不准進訓練集。
train / val 依 sha256 決定（不是亂數）：同一張圖在每次匯出都落在同一邊，
新版資料集的 val 不會混進上一版訓練過的圖，Gate 的比較才有意義。

標注輸出成什麼格式（task）是匯出的選項，不是圖片集的屬性：圖片集裡的框 / 多邊形 / 關鍵點照原樣存，
匯出時才轉（detect：多邊形取外接框；segment：框轉四點多邊形；pose：只留關鍵點）。
不指定 = detect（框）。
"""

from __future__ import annotations

import logging
import shutil
import tempfile
from pathlib import Path

from edgevision.media.library import MediaLibrary
from edgevision.media.shapes import LABEL_TASKS, conversion_counts, to_yolo_lines

logger = logging.getLogger(__name__)


def split_of(sha256: str, val_pct: int) -> str:
    return "val" if int(sha256[:8], 16) % 100 < val_pct else "train"


def label_task(collection: dict, task: str | None) -> str:
    task = task or "detect"
    if task not in LABEL_TASKS:
        raise ValueError(f"task 只能是 {LABEL_TASKS}")
    return task


def build_yolo_dir(lib: MediaLibrary, collection: dict, items: list[dict], root: Path, *, val_pct: int,
                   task: str | None = None) -> dict:
    """回傳 {train, val, task, converted}；converted = 形狀轉換的數量（conversion_counts）。"""
    names, task = collection["class_names"], label_task(collection, task)
    counts = {"train": 0, "val": 0}
    converted = dict.fromkeys(("polygon_to_box", "keypoints_to_box", "box_to_polygon", "dropped"), 0)
    splits = ["val" if val_pct >= 100 else split_of(it["blob_sha256"], val_pct) for it in items]
    if len(items) > 1 and "val" not in splits:      # 張數少時雜湊可能一張都沒分到 val，訓練會沒得驗
        splits[min(range(len(items)), key=lambda i: items[i]["blob_sha256"])] = "val"
    for it, split in zip(items, splits):
        ext = Path(it["object_key"]).suffix or ".jpg"
        img = root / "images" / split / f"{it['blob_sha256']}{ext}"
        lbl = root / "labels" / split / f"{it['blob_sha256']}.txt"
        img.parent.mkdir(parents=True, exist_ok=True)
        lbl.parent.mkdir(parents=True, exist_ok=True)
        lib.objects.download(bucket=it["bucket"], key=it["object_key"], dest=img)
        lines = [] if it["status"] == "background" else to_yolo_lines(it["shapes"], names, task)
        if it["status"] != "background":
            for k, v in conversion_counts(it["shapes"], names, task).items():
                converted[k] += v
        lbl.write_text("\n".join(lines) + ("\n" if lines else ""))
        counts[split] += 1
    (root / "classes.txt").write_text("\n".join(names) + "\n")
    return counts | {"task": task, "converted": converted}


def export_collection(lib: MediaLibrary, cid: int, *, repo: str, message: str, actor: str,
                      task_id: str | None = None, val_pct: int = 20, batch_ids: list[int] | None = None,
                      task: str | None = None) -> dict:
    """同步執行（呼叫端放背景執行緒）。回傳 publish 的結果 + export_id + task + converted。
    batch_ids = 只匯出這些批次（批次快照）；task = 標注輸出格式（不給 = detect）。"""
    from edgevision.training.ingest import publish

    col = lib.collection(cid)
    task = label_task(col, task)
    if not col["class_names"]:
        raise ValueError("圖片集沒有類別，無法匯出")
    batch_ids = lib.check_batches(cid, batch_ids)
    items = lib.exportable_items(cid, batch_ids=batch_ids)
    if not items:
        raise ValueError("沒有可匯出的圖（只匯出 labeled / background；AI 預標要人確認後才算）")
    export_id = lib.start_export(cid, repo=repo, items=items, actor=actor, task_id=task_id, batch_ids=batch_ids)
    tmp = Path(tempfile.mkdtemp(prefix="media-export-"))
    try:
        built = build_yolo_dir(lib, col, items, tmp, val_pct=val_pct, task=task)
        r = publish(tmp, repo, message, class_names=col["class_names"], split_ratio=0.2,
                    source=f"media-collection:{cid}",
                    extra_metadata={"media_collection": cid, "media_export": export_id,
                                    "media_collection_name": col["name"], "label_task": task,
                                    **({"media_batches": batch_ids} if batch_ids else {})})
        lib.finish_export(export_id, dataset_ref=r["dataset_ref"])
        logger.info("圖片集 %s 匯出 %d 張 → %s", cid, len(items), r["dataset_ref"])
        return {"export_id": export_id, **r, "task": task, "converted": built["converted"]}
    except Exception as exc:
        lib.finish_export(export_id, dataset_ref=None, error=str(exc)[:1000])
        raise
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
