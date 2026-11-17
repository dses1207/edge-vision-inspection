#!/usr/bin/env python3
"""清理偵測資料：PG 的 detections 與 MinIO 的存證圖。

⚠️ 這是不可逆操作。預設 dry-run，要真的刪必須加 --apply。

為什麼需要它
    抽樣策略原本把「低信心」當成罕見難例 100% 保存，實測完全不成立 ——
    存下來的 99.8% 是 ok、只有 0.2% 是 ng，其中大量是被低信心規則抓進來的
    （模型 conf 門檻本來就是 0.5，而 hand 有 38% 天生落在 0.5~0.6）。
    策略已改成只留 NG + 0.5% 背景，這支負責清掉先前照舊策略存下來的東西。

刪除順序很重要
    先刪 MinIO 再刪 PG。反過來的話，PG 沒了就找不到 image_uri，
    MinIO 會留下一堆對不到任何紀錄的孤兒檔案。

用法
    python scripts/cleanup_detections.py                    # 看看會刪什麼
    python scripts/cleanup_detections.py --apply            # 真的刪
    python scripts/cleanup_detections.py --keep-days 30 --apply
    python scripts/cleanup_detections.py --test-only --apply # 只清 itest-* 測試殘留
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# 從 scripts/ 執行時要能 import 到專案根的 edgevision 套件
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("cleanup")

# 要保留的：NG 是存證，不能碰。其餘 ok 的紀錄依 --keep-days 保留近期，更舊的清掉。
KEEP_SQL = "verdict_status = 'ng'"


def build_where(keep_days: int, test_only: bool) -> tuple[str, list]:
    """組出「要刪除的列」的條件。回傳 (where 子句, 參數)。"""
    if test_only:
        # 整合測試留下的資料，跟保留天數無關，一律清掉
        return "camera_id LIKE 'itest-%%' OR camera_id LIKE 'test%%'", []
    # NG 永遠保留；ok 的只留最近 keep_days 天
    return f"NOT ({KEEP_SQL}) AND ts < now() - interval '%s days'", [keep_days]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="真的執行刪除（預設只顯示）")
    ap.add_argument("--keep-days", type=int, default=7,
                    help="非 NG 的紀錄保留幾天（預設 7）。NG 一律保留，不受此影響")
    ap.add_argument("--test-only", action="store_true",
                    help="只清 itest-* / test* 的測試殘留")
    ap.add_argument("--batch", type=int, default=10000, help="每批刪幾列（避免長交易）")
    args = ap.parse_args()

    from edgevision.storage.db import Database
    from edgevision.storage.objects import ObjectStore

    db = Database()
    if not db.ping():
        log.error("資料庫無回應")
        return 2
    objects = ObjectStore()
    has_objects = objects.ping()
    if not has_objects:
        log.warning("物件儲存無回應 —— 只清 PG，MinIO 的圖會變成孤兒檔案")

    where, params = build_where(args.keep_days, args.test_only)

    # ── 先看規模 ──────────────────────────────────────────
    with db.cursor() as cur:
        cur.execute(f"SELECT count(1), count(1) FILTER (WHERE image_uri IS NOT NULL) "
                    f"FROM detections WHERE {where}", params)
        n_rows, n_imgs = cur.fetchone()
        cur.execute("SELECT count(1), pg_size_pretty(pg_total_relation_size('detections')) "
                    "FROM detections")
        total, size = cur.fetchone()

    log.info("=" * 60)
    log.info("  目前 detections 共 %s 列（%s）", f"{total:,}", size)
    log.info("  將刪除 %s 列（%.1f%%），其中 %s 列有存證圖",
             f"{n_rows:,}", 100 * n_rows / max(1, total), f"{n_imgs:,}")
    if args.test_only:
        log.info("  條件：測試殘留（itest-* / test*）")
    else:
        log.info("  條件：非 NG 且超過 %d 天（NG 一律保留）", args.keep_days)
    log.info("=" * 60)

    if not args.apply:
        log.info("這是 dry-run。確認無誤後加 --apply 真的執行。")
        return 0
    if n_rows == 0:
        log.info("沒有符合條件的資料。")
        return 0

    # ── ① 先刪 MinIO ──────────────────────────────────────
    # 順序不能反：PG 沒了就查不到 image_uri，MinIO 會留下對不到紀錄的孤兒檔案。
    deleted_objs = 0
    if has_objects and n_imgs:
        log.info("刪除存證圖…")
        while True:
            with db.cursor() as cur:
                cur.execute(f"SELECT DISTINCT image_uri FROM detections "
                            f"WHERE {where} AND image_uri IS NOT NULL LIMIT %s",
                            [*params, args.batch])
                uris = [r[0] for r in cur.fetchall()]
            if not uris:
                break
            for uri in uris:
                try:
                    bucket, key = objects.parse_uri(uri)
                    objects.remove(bucket=bucket, key=key)
                    deleted_objs += 1
                except Exception:
                    # 檔案可能已經不在（先前清過、或 MinIO 端另有生命週期規則）——
                    # 記下來但繼續，不要讓一個缺檔擋住整批清理
                    log.debug("刪除失敗（略過）%s", uri, exc_info=True)
            # 已刪掉的圖要把欄位清空，否則下一輪還會撈到同一批
            with db.cursor() as cur:
                cur.execute("UPDATE detections SET image_uri = NULL WHERE image_uri = ANY(%s)",
                            [uris])
            log.info("  已刪 %s 張", f"{deleted_objs:,}")

    # ── ② 再刪 PG（分批，避免長交易鎖表）──────────────────
    log.info("刪除資料列…")
    deleted_rows = 0
    while True:
        with db.cursor() as cur:
            cur.execute(f"DELETE FROM detections WHERE ctid IN "
                        f"(SELECT ctid FROM detections WHERE {where} LIMIT %s)",
                        [*params, args.batch])
            n = cur.rowcount
        deleted_rows += n
        if n:
            log.info("  已刪 %s 列", f"{deleted_rows:,}")
        if n < args.batch:
            break

    # ── ③ 回收空間 ────────────────────────────────────────
    # DELETE 只是標記，不會把空間還給作業系統。VACUUM FULL 會重寫整張表 ——
    # 期間會鎖表，所以只在真的刪了很多時才做。
    #
    # ⚠️ 必須另開一條 autocommit 連線：VACUUM 不能在交易區塊裡執行，
    #    而連線池給的 cursor 永遠在交易中（先送 COMMIT 也沒用，psycopg 會立刻開新交易）。
    if deleted_rows > 100_000:
        log.info("執行 VACUUM FULL 回收空間（會短暫鎖表）…")
        try:
            import psycopg
            with psycopg.connect(db.dsn, autocommit=True) as conn:
                conn.execute("VACUUM FULL ANALYZE detections")
        except Exception:
            log.warning("VACUUM 失敗 —— 資料已刪除，但檔案空間尚未歸還作業系統。"
                        "可手動執行：psql -c 'VACUUM FULL ANALYZE detections'", exc_info=True)

    with db.cursor() as cur:
        cur.execute("SELECT count(1), pg_size_pretty(pg_total_relation_size('detections')) "
                    "FROM detections")
        after, size_after = cur.fetchone()

    log.info("=" * 60)
    log.info("  完成：刪除 %s 列、%s 張圖", f"{deleted_rows:,}", f"{deleted_objs:,}")
    log.info("  detections 剩 %s 列（%s）", f"{after:,}", size_after)
    log.info("=" * 60)
    db.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
