"""消費者服務的心跳：定期把「我活著、我做了什麼」寫進 service_status。

⭐ 為什麼

    docker healthcheck 只答「活著嗎」。一個消費者可以行程在、port 通、
    卻已經 20 天沒做任何事 —— 那是今天發現三個故障共同的型態。
    心跳裡帶工作指標，/healthz 才能答「在做事嗎」。

⭐ 怎麼用（三行）

    hb = Heartbeat(db, service="recorder")
    hb.start(lambda: {"orphans": n_orphans, "made": stats.made})   # 每 15 秒呼叫一次拿指標
    ...
    hb.stop()

    metrics 回呼在心跳執行緒上跑，要輕（只讀計數器，別查 DB、別 I/O）。

⚠️ 心跳失敗不能拖垮服務

    DB 抖一下、連線池滿，心跳寫不進去 —— 那只是「看不到」，不是「壞了」。
    所以這裡吞掉所有例外、記 warning、下一輪再試。服務本體繼續跑。
    （反過來才是災難：因為心跳寫不進去就把 recorder 停掉。）

模式照抄 runner 的 line_status 心跳：15 秒一次，60 秒沒心跳視為失聯
（門檻在 edgevision/api/config_api.py 的 HEARTBEAT_STALE_S，兩邊共用同一條規則）。
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from typing import Callable

logger = logging.getLogger(__name__)

# 跟 runner 的線路心跳同一個節奏。改這裡要一起看 HEARTBEAT_STALE_S（60 秒）——
# 間隔不能超過它的一半，否則一次網路抖動就會被判 stale。
HEARTBEAT_INTERVAL_S = 15.0

MetricsFn = Callable[[], dict]


class Heartbeat:
    """一個背景執行緒，每 HEARTBEAT_INTERVAL_S 秒 upsert 一列 service_status。"""

    def __init__(self, db, *, service: str, instance: str | None = None,
                 interval_s: float = HEARTBEAT_INTERVAL_S) -> None:
        self.db = db
        self.service = service
        # 多副本時用 hostname 區分；單副本就是空字串（一列）
        self.instance = instance if instance is not None else os.environ.get("HEARTBEAT_INSTANCE", "")
        self.interval_s = interval_s
        self._metrics: MetricsFn = dict
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.beats = 0
        self.failures = 0

    def start(self, metrics: MetricsFn | None = None) -> None:
        if metrics is not None:
            self._metrics = metrics
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name=f"heartbeat-{self.service}",
                                        daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        # 一起來就先跳一次，不要等 15 秒 —— 否則服務啟動後前 15 秒 /healthz 看到的是舊心跳
        self.beat()
        while not self._stop.wait(self.interval_s):
            self.beat()

    def beat(self) -> bool:
        """寫一次心跳。回傳成不成功。失敗只記 warning，不拋。"""
        try:
            metrics = self._metrics() or {}
        except Exception:  # noqa: BLE001 —— 指標回呼壞了也不能讓心跳停
            logger.exception("[%s] 指標回呼失敗，這輪心跳不帶指標", self.service)
            metrics = {"metrics_error": True}
        try:
            with self.db.cursor() as cur:
                cur.execute(
                    "INSERT INTO service_status (service, instance, heartbeat_at, started_at, metrics) "
                    "VALUES (%s, %s, now(), now(), %s::jsonb) "
                    "ON CONFLICT (service, instance) DO UPDATE SET "
                    "  heartbeat_at = now(), "
                    "  metrics = EXCLUDED.metrics",
                    (self.service, self.instance, json.dumps(metrics, default=str)),
                )
            self.beats += 1
            return True
        except Exception:  # noqa: BLE001 —— 見模組 docstring：心跳失敗不能拖垮服務
            self.failures += 1
            # 只在第一次和每 20 次記一次，避免 DB 掛掉時每 15 秒刷一行
            if self.failures == 1 or self.failures % 20 == 0:
                logger.warning("[%s] 心跳寫入失敗（累計 %d 次）—— 服務照跑，只是 /healthz 看不到它",
                               self.service, self.failures, exc_info=True)
            return False

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)


def _age_s(seconds: float | None) -> int | None:
    return None if seconds is None else int(seconds)


# ── /healthz 那一端：讀出來、套門檻 ─────────────────────────

# 各服務的「在做事嗎」門檻。回傳 (degraded, reason)。
# ⚠️ 門檻先寬鬆上線，跑一週看實際分佈再收緊 —— 拍腦袋定的門檻不是永遠紅就是永遠綠。
def _judge_event_processor(m: dict) -> tuple[bool, str | None]:
    lag = m.get("lag")
    if isinstance(lag, (int, float)) and lag > 5000:
        return True, f"積壓 {int(lag)} 筆（> 5000）"
    lw = m.get("last_write_s")
    if isinstance(lw, (int, float)) and lw > 120:
        return True, f"{int(lw)} 秒沒寫 DB"
    return False, None


def _judge_recorder(m: dict) -> tuple[bool, str | None]:
    orphans = m.get("orphans", 0)
    if isinstance(orphans, (int, float)) and orphans > 0:
        return True, f"{int(orphans)} 支影片在 DB 找不到對應列（孤兒）"
    return False, None


def _judge_nas_recorder(m: dict) -> tuple[bool, str | None]:
    alive, total = m.get("cameras_alive"), m.get("cameras_total")
    if isinstance(alive, int) and isinstance(total, int) and total and alive < total:
        return True, f"只有 {alive}/{total} 台在錄"
    if m.get("blocked"):
        return True, "空間不足，錄影暫停"
    return False, None


_JUDGES: dict[str, Callable[[dict], tuple[bool, str | None]]] = {
    "event-processor": _judge_event_processor,
    "recorder": _judge_recorder,
    "nas-recorder": _judge_nas_recorder,
}


def service_status(db, *, stale_after_s: int) -> list[dict]:
    """讀 service_status 表，每列判 stale / degraded。給 /healthz 用。

    ⚠️ 只如實呈現，不做任何處置。共用依賴（DB / Redis）抖動會讓所有服務同時
       看似失聯，跟線路心跳一樣的道理 —— 是否重啟由人或更上層判斷。
    """
    if db is None:
        return []
    try:
        with db.cursor() as cur:
            cur.execute(
                "SELECT service, instance, metrics, "
                "       EXTRACT(EPOCH FROM (now() - heartbeat_at)), "
                "       EXTRACT(EPOCH FROM (now() - started_at)) "
                "FROM service_status ORDER BY service, instance"
            )
            rows = cur.fetchall()
    except Exception:  # noqa: BLE001 —— 表還沒建（舊 DB）也不能讓 /healthz 掛
        logger.debug("讀 service_status 失敗", exc_info=True)
        return []

    out = []
    for service, instance, metrics, hb_age, up_age in rows:
        metrics = metrics or {}
        stale = hb_age is None or hb_age > stale_after_s
        degraded, reason = (False, None) if stale else _JUDGES.get(service, lambda _m: (False, None))(metrics)
        out.append({
            "service": service,
            "instance": instance or None,
            "state": "stale" if stale else ("degraded" if degraded else "ok"),
            "reason": "心跳過期" if stale else reason,
            "heartbeat_age_s": _age_s(hb_age),
            "uptime_s": _age_s(up_age),
            "metrics": metrics,
        })
    return out
