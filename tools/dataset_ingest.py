#!/usr/bin/env python3
"""資料集匯入 LakeFS：標註工具匯出的 zip / 目錄 → 驗格式 → 上傳 → 建立不可變版本 → 印出版本 ID。

    python tools/dataset_ingest.py --src /path/export.zip --repo alpha -m "2026-09 第一批 1200 張"
    → lakefs://alpha/<版本id>

這個「版本 ID」就是訓練請求的 dataset_ref。與 Git 無關。

連線從環境變數：LAKEFS_ENDPOINT / LAKEFS_ACCESS_KEY / LAKEFS_SECRET_KEY（與容器內相同）。
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from edgevision.training.dataset import extract_zip, finalize  # noqa: E402
from edgevision.training.lakefs_client import commit, dataset_uri, ensure_repo, lakefs_client, upload_dir  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="zip 或目錄（images/ labels/ [classes.txt|data.yaml]）")
    ap.add_argument("--repo", required=True, help="LakeFS repository，如 alpha 或 golden")
    ap.add_argument("--branch", default="main")
    ap.add_argument("--prefix", default="", help="repo 內子目錄（可選）")
    ap.add_argument("-m", "--message", required=True)
    ap.add_argument("--classes", default="", help="逗號分隔；來源沒有 classes.txt / data.yaml 時必填")
    ap.add_argument("--all-val", action="store_true", help="黃金測試集：不切分，全部當 val")
    ap.add_argument("--json", action="store_true", help="只印 JSON（給腳本用）")
    a = ap.parse_args()

    src = Path(a.src)
    tmp = Path(tempfile.mkdtemp(prefix="ingest-"))
    try:
        root = extract_zip(src, tmp / "x") if src.is_file() else (shutil.copytree(src, tmp / "x") and tmp / "x")
        names = [c.strip() for c in a.classes.split(",") if c.strip()] or None
        prepared = finalize(root, f"pending", class_names=names,
                            split_ratio=1.0 if a.all_val else 0.2)
        # 上傳的是整理後（已切 train/val、含 manifest.json、data.yaml）的目錄；
        # data.yaml 的 path 是本地路徑，訓練時會重寫，這裡先移除避免誤導
        (root / "data.yaml").unlink(missing_ok=True)

        client = lakefs_client()
        ensure_repo(client, a.repo)
        n = upload_dir(client, a.repo, a.branch, root, prefix=a.prefix)
        m = prepared.manifest
        version_id = commit(client, a.repo, a.branch, a.message, metadata={
            "source": src.name, "n_images": m.n_images, "n_train": m.count("train"), "n_val": m.count("val"),
            "classes": ",".join(prepared.class_names), "files": n,
        })
        uri = dataset_uri(a.repo, version_id) + (f"/{a.prefix.rstrip('/')}" if a.prefix else "")
        info = {"dataset_ref": uri, "repo": a.repo, "version_id": version_id, "files": n,
                "n_images": m.n_images, "n_train": m.count("train"), "n_val": m.count("val"),
                "class_names": prepared.class_names, "class_distribution": m.class_distribution}
        if a.json:
            print(json.dumps(info, ensure_ascii=False))
        else:
            print(f"✅ 已匯入 {n} 個檔案（{m.n_images} 張，train {m.count('train')} / val {m.count('val')}）")
            print(f"   類別：{prepared.class_names}  分布：{m.class_distribution}")
            print(f"   資料集版本 ID：{uri}")
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
