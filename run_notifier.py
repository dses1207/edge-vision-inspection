#!/usr/bin/env python3
"""Notifier：日報 + 健康巡檢（NG 通知是第二階段）。

⭐ 為什麼是一個獨立行程

    它讀 /healthz 判斷別人死活。塞進 output-api 的話，API 掛了就沒人叫 ——
    看門狗不能跟被看的東西住在一起。

⭐ 沒有任何通道也能跑

    NOTIFY_* 沒設就是乾跑：卡片印進 log、notify_log 記 suppressed。
    整條鏈（巡檢 → 狀態機 → 組卡片）在沒有 Teams 的測試機上照樣驗。

環境變數
    REDIS_URL, DATABASE_URL     必填
    OUTPUT_API                  /healthz 與 /api/config/lines 的位址，預設 http://localhost:18080
    NOTIFY_DAILY / NOTIFY_HEALTH / NOTIFY_NG   Apprise URL 清單，分號隔開；空 = 乾跑
                                （Teams 用 workflows://、寄信用 mailto://）
    NOTIFY_DAILY_AT             預設 08:00（TZ 要設 Asia/Taipei）
    HEALTH_INTERVAL_S           巡檢間隔，預設 60
    HEALTH_COOLDOWN_S           同項目冷卻，預設 600
    HEALTH_REPEAT_S             未恢復提醒，預設 14400（4h）
"""
from __future__ import annotations

import logging
import os
import signal
import sys
import threading
import time

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)-7s %(name)-24s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("notifier")


def _require(name: str) -> str:
    v = os.environ.get(name, "").strip()
    if not v:
        raise SystemExit(f"必須設定 {name}")
    return v


def main() -> None:
    redis_url = _require("REDIS_URL")
    _require("DATABASE_URL")
    api = os.environ.get("OUTPUT_API", "http://localhost:18080").rstrip("/")

    from edgevision.storage.db import Database
    from edgevision.storage.heartbeat import Heartbeat
    from edgevision.notify.daily import DailyReporter
    from edgevision.notify.health import HealthMonitor, fetch_healthz
    from edgevision.notify.sender import Sender
    import redis as _redis

    db = Database()
    if not db.ping():
        raise SystemExit("資料庫無回應 —— 沒有 DB 就沒有日報與 health_events，拒絕啟動")
    r = _redis.Redis.from_url(redis_url, socket_connect_timeout=5, socket_timeout=10,
                              decode_responses=True)
    try:
        r.ping()
    except Exception as exc:
        raise SystemExit(f"Redis 無回應 —— 狀態無處可存，拒絕啟動：{exc}") from exc

    sender = Sender()
    if sender.bad:
        raise SystemExit(f"通知 URL 解析失敗，拒絕啟動（.env 打錯字會讓通知靜默消失）：{sender.bad}")
    health = HealthMonitor(
        redis=r, db=db, sender=sender, fetch=lambda: fetch_healthz(api),
        cooldown_s=float(os.environ.get("HEALTH_COOLDOWN_S", 600)),
        repeat_s=float(os.environ.get("HEALTH_REPEAT_S", 14400)))
    daily = DailyReporter(redis=r, db=db, sender=sender, config_api=api,
                          at=os.environ.get("NOTIFY_DAILY_AT", "08:00"))
    interval = float(os.environ.get("HEALTH_INTERVAL_S", 60))

    # 自己也要有心跳 —— 看門狗死了要有人知道（日報是 dead-man switch，這是即時的那個）
    hb = Heartbeat(db, service="notifier")
    hb.start(lambda: {"rounds": health.rounds, "notified": health.notified,
                      "daily_sent": daily.sent,
                      "channels": {t: sender.targets(t) for t in sender.TAGS}})

    log.info("=" * 64)
    log.info("  notifier（通知）")
    log.info("  巡檢每 %.0f 秒讀 %s/healthz；冷卻 %.0fs、未恢復每 %.0fh 提醒",
             interval, api, health.cooldown_s, health.repeat_s / 3600)
    log.info("  日報每日 %02d:%02d（TZ=%s）", daily.hh, daily.mm, os.environ.get("TZ", "未設！"))
    for tag in sender.TAGS:
        log.info("  %-9s → %s", tag, sender.describe(tag))
    log.info("=" * 64)

    stop = threading.Event()

    def _bye(_s, _f):
        log.info("收到停止訊號…")
        stop.set()
    signal.signal(signal.SIGTERM, _bye)
    signal.signal(signal.SIGINT, _bye)

    while not stop.is_set():
        t0 = time.time()
        try:
            for tr in health.run_once():
                log.info("健康 %s：%s → %s%s", tr.item, tr.from_state, tr.to_state,
                         f"（{tr.reason}）" if tr.reason else "")
        except Exception:
            log.exception("巡檢這一輪失敗，下一輪再試")
        try:
            daily.try_send()
        except Exception:
            log.exception("日報失敗 —— 鎖已取得不會重送，明天再來")
        stop.wait(max(1.0, interval - (time.time() - t0)))

    hb.stop()
    db.close()
    log.info("已停止。巡檢 %d 輪、通知 %d 則、日報 %d 則", health.rounds, health.notified, daily.sent)


if __name__ == "__main__":
    sys.exit(main())
