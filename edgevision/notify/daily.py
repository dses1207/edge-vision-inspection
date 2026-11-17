"""① 日報：每日 08:00 一則 Teams 卡片。

⭐ 它同時是 dead-man switch

    08:10 沒收到日報 = notifier 本身掛了。不另做「我還活著」的心跳通知 ——
    一則每天固定會來的訊息，比一個沒人看的 healthcheck 更能讓人發現東西壞了。

⭐ 多副本只送一次

    SET NX EX 3600 notify:daily:{date} 搶鎖。搶到的送，沒搶到的跳過。
    K8s replicas>1 不用改。

內容全部來自已存在的表與端點（detections / notify_log / health_events / /api/config/lines），
notifier 自己不存任何統計。
"""

from __future__ import annotations

import logging
import os
import time
from datetime import date, datetime, timedelta
from typing import Any

from edgevision.notify.sender import Card, Sender
from edgevision.notify.health import _log_notify

logger = logging.getLogger(__name__)


def collect(db, *, day: date, lines: list[dict] | None = None) -> dict[str, Any]:
    """撈昨日的數字。純查詢，可對假 cursor 測。"""
    d: dict[str, Any] = {"day": day.isoformat()}
    with db.cursor() as cur:
        # NG：不重複幀數（一幀多列，算幀才是「幾次」）
        cur.execute(
            "SELECT count(DISTINCT frame_id), count(DISTINCT camera_id) "
            "FROM detections WHERE verdict_status='ng' AND ts::date = %s", (day,))
        d["ng_frames"], d["ng_cameras"] = cur.fetchone()

        cur.execute(
            "SELECT camera_id, count(DISTINCT frame_id) AS n FROM detections "
            "WHERE verdict_status='ng' AND ts::date = %s GROUP BY 1 ORDER BY n DESC LIMIT 5", (day,))
        d["ng_by_camera"] = cur.fetchall()

        cur.execute(
            "SELECT coalesce(verdict_reason,'(無)'), count(DISTINCT frame_id) AS n FROM detections "
            "WHERE verdict_status='ng' AND ts::date = %s GROUP BY 1 ORDER BY n DESC LIMIT 5", (day,))
        d["ng_by_reason"] = cur.fetchall()

        # 影片：有 clip_uri 的 NG 幀 vs 沒有的。沒有的應為 0（recorder 拿掉冷卻後一對一）
        cur.execute(
            "SELECT count(DISTINCT clip_uri), "
            "       count(DISTINCT frame_id) FILTER (WHERE clip_uri IS NULL) "
            "FROM detections WHERE verdict_status='ng' AND ts::date = %s", (day,))
        d["clips"], d["ng_without_clip"] = cur.fetchone()

        # 健康：昨日的切換與目前仍 down 的
        cur.execute(
            "SELECT item, from_state, to_state, at, duration_s FROM health_events "
            "WHERE at::date = %s ORDER BY at", (day,))
        d["health_events"] = cur.fetchall()

        # 通知：昨日各狀態筆數
        cur.execute(
            "SELECT status, count(*) FROM notify_log WHERE sent_at::date = %s GROUP BY 1", (day,))
        d["notify"] = dict(cur.fetchall())

    d["lines"] = lines or []
    return d


def build_card(d: dict[str, Any]) -> Card:
    """把數字排成卡片。純函式，可測。"""
    facts: list[tuple[str, str]] = []
    body: list[str] = []

    # NG
    facts.append(("NG", f"{d['ng_frames']} 次，{d['ng_cameras']} 台相機"))
    if d["ng_by_camera"]:
        facts.append(("相機 top", "、".join(f"{c} {n}" for c, n in d["ng_by_camera"])))
    if d["ng_by_reason"]:
        facts.append(("原因", "、".join(f"{r} {n}" for r, n in d["ng_by_reason"])))

    # 影片：沒影片的 NG 應為 0，不是 0 要標出來
    miss = d["ng_without_clip"]
    clip_line = f"{d['clips']} 支"
    if miss:
        clip_line += f"　⚠️ {miss} 個 NG 沒有影片"
    facts.append(("事件影片", clip_line))

    # 健康
    ev = d["health_events"]
    downs = [e for e in ev if e[2] != "ok"]
    recs = [e for e in ev if e[2] == "ok"]
    if not ev:
        facts.append(("健康", "全天無中斷"))
    else:
        facts.append(("健康", f"{len(downs)} 次中斷、{len(recs)} 次恢復"))
        for item, _f, to, at, dur in ev[:6]:
            t = at.strftime("%H:%M") if hasattr(at, "strftime") else str(at)
            if to == "ok":
                body.append(f"🟢 {t} {item} 恢復" + (f"（中斷 {dur // 60} 分）" if dur else ""))
            else:
                body.append(f"{'🔴' if to == 'down' else '🟠'} {t} {item} {to}")
        if len(ev) > 6:
            body.append(f"…另 {len(ev) - 6} 筆")

    # 通知
    n = d["notify"]
    if n:
        facts.append(("通知", "、".join(f"{k} {v}" for k, v in sorted(n.items()))))

    # 版本
    if d["lines"]:
        facts.append(("線路", "、".join(
            f"{l.get('id') or l.get('line_id')} v{l.get('desired_version') or l.get('config_version')}"
            + ("" if l.get("in_sync", True) else " ⚠️未同步")
            for l in d["lines"])))

    return Card(title=f"日報 {d['day']}", level="info", facts=facts, body=body)


class DailyReporter:
    def __init__(self, *, redis, db, sender: Sender, config_api: str | None = None,
                 at: str = "08:00", receiver: str = "daily") -> None:
        self.redis, self.db, self.sender = redis, db, sender
        self.config_api = config_api
        self.hh, self.mm = (int(x) for x in at.split(":"))
        self.receiver = receiver
        self.sent = 0

    def due(self, now: datetime | None = None) -> date | None:
        """現在該不該送。回傳要報的日期（昨天），不該送回 None。

        時間過了 at 且今天還沒送 → 送。判斷「送過沒」靠 Redis 鎖，不靠記憶體，
        重啟後不會重送、多副本不會同送。
        """
        now = now or datetime.now()
        if (now.hour, now.minute) < (self.hh, self.mm):
            return None
        return now.date() - timedelta(days=1)

    def try_send(self, now: datetime | None = None) -> bool:
        day = self.due(now)
        if day is None:
            return False
        key = f"notify:daily:{day.isoformat()}"
        if not self.redis.set(key, "1", nx=True, ex=3600):
            return False                      # 今天已送過（或別的副本在送）
        lines = self._lines()
        d = collect(self.db, day=day, lines=lines)
        card = build_card(d)
        r = self.sender.send(self.receiver, card)
        _log_notify(self.db, kind="daily", event_id=day.isoformat(), receiver=self.receiver, result=r)
        self.sent += 1
        logger.info("日報 %s：NG %d 次、影片 %d 支、健康事件 %d 筆 → %s",
                    day, d["ng_frames"], d["clips"], len(d["health_events"]),
                    "已送" if r.sent else ("乾跑" if r.dry_run else f"失敗 {r.error}"))
        return True

    def _lines(self) -> list[dict]:
        if not self.config_api:
            return []
        try:
            import httpx
            with httpx.Client(timeout=8.0, trust_env=False) as c:
                return c.get(f"{self.config_api.rstrip('/')}/api/config/lines").json()
        except Exception:  # noqa: BLE001 —— 拿不到版本資訊，日報照送
            logger.warning("讀線路清單失敗，日報不含版本", exc_info=True)
            return []
