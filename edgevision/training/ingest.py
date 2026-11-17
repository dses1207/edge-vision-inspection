"""YOLO 目錄 → LakeFS 不可變版本。zip 上傳（控制 API）與圖片集匯出（edgevision/media/export.py）共用。"""

from __future__ import annotations

import json
from pathlib import Path


def publish(root: Path, repo: str, message: str, *, class_names: list[str] | None, split_ratio: float,
            source: str, extra_metadata: dict | None = None) -> dict:
    """驗證 + manifest → 上傳 → commit。回傳 dataset_ref 與統計。"""
    from edgevision.training.dataset import finalize
    from edgevision.training.lakefs_client import commit, dataset_uri, ensure_repo, lakefs_client, upload_dir

    prepared = finalize(root, "pending", class_names=class_names, split_ratio=split_ratio)
    (root / "data.yaml").unlink(missing_ok=True)
    client = lakefs_client()
    ensure_repo(client, repo)
    n = upload_dir(client, repo, "main", root)
    m = prepared.manifest
    vid = commit(client, repo, "main", message, metadata={
        "source": source, "n_images": m.n_images, "n_train": m.count("train"), "n_val": m.count("val"),
        "classes": ",".join(prepared.class_names), "files": n,
        # LakeFS metadata 只收字串 → JSON 字串存，get_dataset_version 解回 dict。
        # 不寫的話各類張數只活在記憶體 job 裡，control-api 一重啟就查不回來
        # （稀有類別，送訓練前必須確認得到）。
        "class_distribution": json.dumps(m.class_distribution, ensure_ascii=False),
        **{k: str(v) for k, v in (extra_metadata or {}).items()}})
    return {"dataset_ref": dataset_uri(repo, vid), "files": n, "n_images": m.n_images,
            "n_train": m.count("train"), "n_val": m.count("val"), "class_names": prepared.class_names,
            "class_distribution": m.class_distribution}
