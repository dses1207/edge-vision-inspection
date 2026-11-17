#!/usr/bin/env python3
"""Event Processor：把推論結果從 Redis 落地到 PostgreSQL。

⭐ 為什麼是獨立行程，不塞進 output-api

    兩者的失敗模式不同，綁在一起會互相拖累：
      · 落地卡住（DB 慢）不該讓監看頁面掛掉
      · API 重新部署不該中斷落地
    而且落地要能獨立擴展 —— consumer group 天然支援多副本分攤。

    需要的話仍可內嵌（EventProcessor.start() 是背景執行緒模式），
    但預設走獨立容器。

用法：python run_processor.py
"""
from __future__ import annotations

import logging
import os
import signal
import sys

from edgevision.storage.db import Database
from edgevision.storage.event_processor import EventProcessor

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)-7s %(name)-28s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("processor")


def main() -> None:
    db = Database()
    if not db.ping():
        # 沒有 DB 就沒有落地可言 —— 誠實報錯，不要假裝在跑
        raise SystemExit("資料庫無回應 —— 落地無法啟動（檢查 PG_DSN / POSTGRES_*）")

    proc = EventProcessor(db=db)
    if not proc.connect():
        raise SystemExit("Redis 連線失敗 —— 落地無法啟動（檢查 REDIS_URL）")

    def _bye(_sig, _frm):
        log.info("收到停止訊號，結束中…最終統計 %s", proc.stats.snapshot())
        proc._stop.set()

    signal.signal(signal.SIGTERM, _bye)
    signal.signal(signal.SIGINT, _bye)

    log.info("=" * 64)
    log.info("  event-processor（落地）")
    log.info("  來源      %s", proc.stream)
    log.info("  group     %s / consumer %s", proc.group, proc.consumer)
    log.info("  保存只存 NG（OK 幀不進資料庫）")
    log.info("=" * 64)
    try:
        proc.run()
    finally:
        db.close()
        log.info("已停止。統計 %s", proc.stats.snapshot())


if __name__ == "__main__":
    sys.exit(main())
