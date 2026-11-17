#!/usr/bin/env python3
"""獨立 API 服務（output-api）。

⭐ 為什麼要獨立於推論：
    DeepStream 正式路線下，推論跑在 DeepStream 容器裡，那個容器不該也不適合
    扛一個 web server。API 抽出來之後：
      · 推論容器專心做推論，掛了不影響監看頁面
      · API 可獨立擴展、獨立重啟
      · 前端與設定管理不受推論容器生命週期影響

    與舊的同進程模式（run_mvp.py，已移除）用的是同一份 app，差別只在「狀態從哪來」：
      同進程   → register_service 注入的物件
      獨立     → DB 的 line_status（runner 回報）＋ Redis Streams（結果）

用法：python run_api.py [--port 8080]
"""
from __future__ import annotations

import argparse
import logging
import os
import time
import threading

import uvicorn

from edgevision.api.server import app, bind_storage

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)-7s %(name)-28s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("output-api")


def _connect_storage(db=None, objects=None):
    """還沒連上的那個再試一次。回傳 (db, objects)，連不上的是 None。"""
    try:
        from edgevision.storage.db import Database
        from edgevision.storage.objects import ObjectStore
        if db is None:
            d = Database()
            db = d if d.ping() else None
            if db is None:
                log.warning("資料庫無回應 —— 歷史查詢與設定管理暫不可用，背景重試中")
        if objects is None:
            o = ObjectStore()
            objects = o if o.ping() else None
            if objects is None:
                log.warning("物件儲存無回應 —— 存證下載暫不可用，背景重試中")
    except Exception as exc:
        log.warning("儲存層初始化失敗（%s），背景重試中", exc)
    return db, objects


def _retry_storage(db, objects, interval: float = 10.0) -> None:
    while db is None or objects is None:
        time.sleep(interval)
        db, objects = _connect_storage(db, objects)
        bind_storage(db=db, objects=objects)
    log.info("儲存層已連上（資料庫、物件儲存）")


def main() -> None:
    ap = argparse.ArgumentParser()
    # 不給預設值：port 在 compose/K8s 與 healthcheck 兩處都要用同一個值，
    # 各自帶預設就會不一致（曾發生：服務跑 18080、healthcheck 打 8080，
    # 健檢永遠失敗 → autoheal 每 70 秒重啟一次，重啟了 948 次）。
    _port = os.environ.get("API_PORT")
    if not _port:
        raise SystemExit("必須設定 API_PORT（部署層與健康檢查要用同一個值）")
    ap.add_argument("--port", type=int, default=int(_port))
    ap.add_argument("--host", default="0.0.0.0")
    args = ap.parse_args()

    # 儲存層：查歷史、讀線路狀態、設定管理都要它
    db, objects = _connect_storage()
    bind_storage(db=db, objects=objects)
    if db is None or objects is None:
        # ⚠️ 連不上就在背景重試，連上自動接上。以前只在啟動時試一次：K8s 上 output-api 比 postgres
        #    先起來，之後設定 API 一直回 503「資料庫未連線」，要人手動重啟（2026-10-05 正式叢集首次部署）。
        threading.Thread(target=_retry_storage, args=(db, objects), name="storage-retry", daemon=True).start()

    log.info("=" * 64)
    log.info("  output-api（獨立模式）")
    log.info("  資料庫   %s", "已連線" if db else "未連線")
    log.info("  物件儲存 %s", "已連線" if objects else "未連線")
    log.info("  ▶  監看頁面   http://localhost:%d/", args.port)
    log.info("  ▶  設定管理   http://localhost:%d/api/config/lines", args.port)
    log.info("  ▶  API 文件   http://localhost:%d/docs", args.port)
    log.info("=" * 64)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
