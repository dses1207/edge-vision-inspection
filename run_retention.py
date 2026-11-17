"""存證保留：每天跑一次（K8s CronJob edgevision-retention）。

    事件影片（MinIO）保留 CLIP_RETENTION_DAYS 天（預設 7）—— 刪檔由 MinIO bucket 過期規則做，
    這裡只處理資料庫：超過天數的 NG 紀錄把 clip_uri / image_uri 標成失效（紀錄保留）。
    兩邊天數必須一致：init-buckets.sh 與這裡讀同一個 CLIP_RETENTION_DAYS。
"""
import logging
import os

from edgevision.storage.db import Database

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("retention")


def main() -> None:
    days = float(os.environ.get("CLIP_RETENTION_DAYS", 7))
    db = Database()
    n = db.expire_evidence(older_than_days=days)
    log.info("存證連結標成失效：%d 列（超過 %.0f 天；紀錄保留）", n, days)


if __name__ == "__main__":
    main()
