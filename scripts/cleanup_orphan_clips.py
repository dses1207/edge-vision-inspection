#!/usr/bin/env python3
"""清掉沒有索引的事件影片（孤兒片段）。

⚠️ 這是不可逆操作。預設 dry-run，要真的刪必須加 --apply。

為什麼會有孤兒片段
    recorder 把影片上傳 MinIO 之後，回傳的 s3:// URI 沒有人接住寫回 DB
    （`pool.submit(maker.make, ...)` 的結果直接被丟棄）。於是影片躺在
    MinIO 裡，沒有任何 detections 列指向它，前端查到 NG 也播不到。
    2.7 天累積了 3312 個。

    寫回的修正已上線。這支負責清掉先前那批。

怎麼分辨孤兒
    事件影片的 key 是 `{camera_id}/{YYYYMMDD_HHMMSS}_{uuid8}.mp4`
    （加一個同名的 .json side-car）。
    拿它組回 `s3://{bucket}/{key}`，到 detections.clip_uri 查有沒有人引用。
    沒有 → 孤兒。

⚠️ 不會碰存證圖
    存證圖在同一個 bucket，但 key 是 `defects/YYYY/MM/DD/...`。
    本腳本只掃「第一層就是 camera_id」的 key，且副檔名限 .mp4/.json。
    為保險起見另外硬性排除 defects/ 前綴。

用法
    python scripts/cleanup_orphan_clips.py                 # 看看會刪什麼
    python scripts/cleanup_orphan_clips.py --apply         # 真的刪
    python scripts/cleanup_orphan_clips.py --before 2026-09-15 --apply
"""
from __future__ import annotations

import argparse
import logging
import re
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("cleanup-clips")

# 事件影片的 key：{camera_id}/{YYYYMMDD_HHMMSS}_{uuid8}.{mp4,json}
CLIP_KEY_RE = re.compile(r"^(?P<cam>[^/]+)/(?P<ts>\d{8}_\d{6})_(?P<uid>[0-9a-f]{8})\.(?P<ext>mp4|json)$")

# 絕對不碰的前綴（存證圖、資料集等）
PROTECTED_PREFIXES = ("defects/", "datasets/", "models/", "golden/")


def parse_key(key: str) -> dict | None:
    """認得出來就回解析結果，認不出來回 None（= 不是事件影片，不碰）。"""
    if key.startswith(PROTECTED_PREFIXES):
        return None
    m = CLIP_KEY_RE.match(key)
    if not m:
        return None
    d = m.groupdict()
    try:
        d["dt"] = datetime.strptime(d["ts"], "%Y%m%d_%H%M%S")
    except ValueError:
        return None
    return d


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="真的執行刪除（預設只顯示）")
    ap.add_argument("--bucket", default="inference-output")
    ap.add_argument("--before", help="只刪這個日期之前的（YYYY-MM-DD），不給就是全部孤兒")
    ap.add_argument("--limit", type=int, help="最多刪幾個（測試用）")
    args = ap.parse_args()

    before = datetime.strptime(args.before, "%Y-%m-%d") if args.before else None

    from edgevision.storage.db import Database
    from edgevision.storage.objects import ObjectStore

    objects = ObjectStore()
    if not objects.ping():
        log.error("物件儲存無回應")
        return 2
    db = Database()
    if not db.ping():
        log.error("資料庫無回應 —— 沒有 DB 就無法判斷誰是孤兒，拒絕執行")
        return 2

    # ── ① 先把 DB 裡被引用的 clip_uri 全撈出來 ────────────
    # 一次撈完做集合比對，比每個檔案查一次 DB 快得多。
    with db.cursor() as cur:
        cur.execute("SELECT DISTINCT clip_uri FROM detections WHERE clip_uri IS NOT NULL")
        referenced = {r[0] for r in cur.fetchall()}
    log.info("DB 裡有 %s 支影片被引用", f"{len(referenced):,}")

    # ── ② 掃 MinIO ────────────────────────────────────────
    clips: dict[str, list] = {}          # base（不含副檔名）→ [(key, size), ...]
    skipped = 0
    for obj in objects._client.list_objects(args.bucket, recursive=True):
        info = parse_key(obj.object_name)
        if not info:
            skipped += 1
            continue
        if before and info["dt"] >= before:
            continue
        base = obj.object_name.rsplit(".", 1)[0]
        clips.setdefault(base, []).append((obj.object_name, obj.size))

    # ── ③ 分出孤兒 ────────────────────────────────────────
    orphans, kept = {}, 0
    for base, files in clips.items():
        if f"s3://{args.bucket}/{base}.mp4" in referenced:
            kept += 1
        else:
            orphans[base] = files

    total_files = sum(len(f) for f in orphans.values())
    total_bytes = sum(sz for f in orphans.values() for _, sz in f)

    log.info("=" * 60)
    log.info("  bucket            %s", args.bucket)
    log.info("  略過（非事件影片）%s 個物件", f"{skipped:,}")
    log.info("  事件片段共        %s 組", f"{len(clips):,}")
    log.info("  其中有索引        %s 組（保留）", f"{kept:,}")
    log.info("  ⚠️ 孤兒            %s 組 / %s 個檔案 / %.2f GB",
             f"{len(orphans):,}", f"{total_files:,}", total_bytes / 1e9)
    if before:
        log.info("  條件只刪 %s 之前", args.before)
    log.info("=" * 60)

    if orphans:
        log.info("範例（最多 5 組）：")
        for base in sorted(orphans)[:5]:
            log.info("    %s.mp4", base)

    if not orphans:
        log.info("沒有孤兒片段。")
        return 0
    if not args.apply:
        log.info("這是 dry-run。確認無誤後加 --apply 真的執行。")
        return 0

    # ── ④ 刪除 ────────────────────────────────────────────
    # ⚠️ 先刪 .json 再刪 .mp4：json 是索引，有索引沒影片是死連結。
    #    （上傳時順序相反，刪除就要倒過來。）
    deleted = freed = 0
    for i, (base, files) in enumerate(sorted(orphans.items())):
        if args.limit and i >= args.limit:
            log.info("達到 --limit %d，停止", args.limit)
            break
        for key, size in sorted(files, key=lambda kv: not kv[0].endswith(".json")):
            try:
                objects.remove(bucket=args.bucket, key=key)
                deleted += 1
                freed += size
            except Exception:
                log.warning("刪除失敗（略過）%s", key, exc_info=True)
        if deleted and deleted % 500 == 0:
            log.info("  已刪 %s 個檔案（%.2f GB）", f"{deleted:,}", freed / 1e9)

    log.info("=" * 60)
    log.info("  完成：刪除 %s 個檔案，釋放 %.2f GB", f"{deleted:,}", freed / 1e9)
    log.info("=" * 60)
    db.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
