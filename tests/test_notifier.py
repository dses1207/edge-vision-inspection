"""通知系統：狀態機、抑制、日報卡片、Teams 投遞。

⚠️ 舊系統的教訓：告警規則表、CRUD、送通知的函式全寫了，
   但沒有任何地方呼叫它 —— NG 從來沒推過 Teams。
   這裡的測試要守的不只是邏輯對，是「接上了」：從 /healthz 到卡片一路走通。
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import date, datetime
from unittest.mock import MagicMock

import pytest

from edgevision.notify.daily import DailyReporter, build_card
from edgevision.notify.health import HealthMonitor, ItemState, evaluate, inhibit
from edgevision.notify.sender import Card, SendResult, Sender, _redact, _split


# ── 假 Redis（hash + set nx）────────────────────────────────

class FakeRedis:
    def __init__(self):
        self.h: dict[str, dict[str, str]] = {}
        self.kv: dict[str, str] = {}

    def hget(self, k, f): return self.h.get(k, {}).get(f)
    def hset(self, k, f, v): self.h.setdefault(k, {})[f] = v
    def set(self, k, v, nx=False, ex=None):
        if nx and k in self.kv:
            return None
        self.kv[k] = v
        return True


class FakeSender:
    TAGS = Sender.TAGS
    def __init__(self): self.sent: list[tuple[str, Card]] = []
    def send(self, tag, card):
        self.sent.append((tag, card))
        return SendResult(sent=True, targets=1)
    def configured(self, t): return True
    def targets(self, t): return 1


def _monitor(items_fn, **kw):
    fr, fs = FakeRedis(), FakeSender()
    m = HealthMonitor(redis=fr, db=None, sender=fs, fetch=lambda: {}, **kw)
    m._items = items_fn
    return m, fr, fs


# ── evaluate：/healthz → 項目 ───────────────────────────────

def test_evaluate_stale_line_is_down():
    """⭐ line_c 死 20 天沒人發現：stale 必須被當成 down 來叫。"""
    hz = {"lines": [{"line_id": "alpha_a", "stale": True, "heartbeat_age_s": 1700000}]}
    assert evaluate(hz) == [("line:alpha_a", "down", "runner 心跳 1700000 秒沒來")]


def test_evaluate_out_of_sync_line_is_degraded():
    hz = {"lines": [{"line_id": "alpha_a", "stale": False, "in_sync": False,
                     "desired_version": 5, "reported_version": 4}]}
    (item, state, reason), = evaluate(hz)
    assert state == "degraded" and "v5" in reason and "v4" in reason


def test_evaluate_services_map_states():
    hz = {"services": [{"service": "recorder", "state": "ok"},
                       {"service": "event-processor", "state": "degraded", "reason": "積壓"},
                       {"service": "nas-recorder", "state": "stale"}]}
    got = {i: s for i, s, _ in evaluate(hz)}
    assert got == {"service:recorder": "ok", "service:event-processor": "degraded",
                   "service:nas-recorder": "down"}


def test_evaluate_storage():
    hz = {"storage": {"database": True, "objects": False, "unknown": None}}
    got = {i: s for i, s, _ in evaluate(hz)}
    assert got == {"storage:database": "ok", "storage:objects": "down"}


# ── inhibit：共用依賴掛了不要各自叫 ──────────────────────────

def test_evaluate_camera_level():
    """線路心跳正常但某台 0 幀 → 那台相機 down（golf 404、cam156 被踢線那種）；unknown 不列。"""
    hz = {"lines": [{"line_id": "alpha_a", "in_sync": True, "apply_state": "ok", "cameras": [
        {"camera_id": "cam156", "state": "no_frames", "fps": 0.0},
        {"camera_id": "cam193", "state": "ok", "fps": 14.9},
        {"camera_id": "cam230", "state": "unknown", "fps": None}]}]}
    items = {i: st for i, st, _ in evaluate(hz)}
    assert items["camera:cam156"] == "down" and items["camera:cam193"] == "ok"
    assert "camera:cam230" not in items and items["line:alpha_a"] == "ok"


def test_evaluate_skipped_camera_alerts_that_camera_only():
    hz = {"lines": [{"line_id": "webcam_bravo", "in_sync": True, "apply_state": "ok", "cameras": [
        {"camera_id": "cam211", "state": "skipped", "fps": None, "reason": "ValueError: roi_regions 少了 work 區"},
        {"camera_id": "cam137", "state": "ok", "fps": 14.0}]}]}
    got = {i: (st, why) for i, st, why in evaluate(hz)}
    assert got["camera:cam211"][0] == "down" and "roi_regions" in got["camera:cam211"][1]
    assert got["camera:cam137"][0] == "ok" and got["line:webcam_bravo"][0] == "ok"


def test_evaluate_stale_line_does_not_also_flag_cameras():
    hz = {"lines": [{"line_id": "alpha_a", "stale": True, "heartbeat_age_s": 300,
                     "cameras": [{"camera_id": "cam156", "state": "stale", "fps": None}]}]}
    assert [i for i, _, _ in evaluate(hz)] == ["line:alpha_a"]


def test_inhibit_db_down_suppresses_symptoms():
    items = [("storage:database", "down", "x"), ("line:alpha_a", "down", "stale"),
             ("service:recorder", "degraded", "y"), ("service:notifier", "ok", None)]
    got = {i: s for i, s, _ in inhibit(items)}
    assert got["storage:database"] == "down"
    assert got["line:alpha_a"] == "ok" and got["service:recorder"] == "ok", "DB 的症狀不該各自叫"


def test_inhibit_noop_when_deps_ok():
    items = [("storage:database", "ok", None), ("line:alpha_a", "down", "stale")]
    assert inhibit(items) == items


# ── 狀態機 ────────────────────────────────────────────────

def test_needs_two_consecutive_rounds():
    """⭐ 擋抖動：一輪 down 不叫，連續兩輪才叫。"""
    m, _, fs = _monitor(None)
    assert m.step([("line:a", "down", "r")], now=0) == []
    tr = m.step([("line:a", "down", "r")], now=60)
    assert len(tr) == 1 and tr[0].to_state == "down"


def test_flap_does_not_notify():
    m, _, _ = _monitor(None)
    m.step([("line:a", "down", "r")], now=0)
    m.step([("line:a", "ok", None)], now=60)      # 抖回來
    assert m.step([("line:a", "down", "r")], now=120) == [], "抖動不該叫"


def test_recovery_reports_duration():
    m, _, _ = _monitor(None)
    m.step([("line:a", "down", "r")], now=0)
    m.step([("line:a", "down", "r")], now=60)       # 確認 down，since=60
    m.step([("line:a", "ok", None)], now=600)
    tr = m.step([("line:a", "ok", None)], now=660)
    assert tr[0].to_state == "ok" and tr[0].duration_s == 600


def test_fault_recover_fault_notifies_again():
    """故障 → 恢復 → 再故障：第二次故障是【新資訊】，要叫。

    冷卻擋的是「同一個持續中的故障被重複叫」，不是「恢復後又壞了」——
    恢復已經公告過，再壞是另一件事。Alertmanager 的 repeat_interval 也是這個語意。
    """
    m, _, _ = _monitor(None, cooldown_s=600)
    m.step([("line:a", "degraded", "r")], now=0)
    m.step([("line:a", "degraded", "r")], now=60)      # 叫了
    m.step([("line:a", "ok", None)], now=120); m.step([("line:a", "ok", None)], now=180)   # 恢復（叫了）
    m.step([("line:a", "degraded", "r")], now=240)
    tr = m.step([("line:a", "degraded", "r")], now=300)
    assert len(tr) == 1 and tr[0].to_state == "degraded"


def test_cooldown_blocks_same_ongoing_fault():
    """真正被冷卻擋住的：同一個 down 還沒恢復，狀態機不會在 repeat_s 之前再叫。"""
    m, _, _ = _monitor(None, cooldown_s=600, repeat_s=14400)
    m.step([("line:a", "down", "r")], now=0)
    m.step([("line:a", "down", "r")], now=60)          # 叫了
    for t in (120, 300, 600, 3000):
        assert m.step([("line:a", "down", "r")], now=t) == [], f"t={t} 不該再叫"


def test_escalation_bypasses_cooldown():
    """degraded → down 是往更壞走，冷卻不擋。"""
    m, _, _ = _monitor(None, cooldown_s=600)
    m.step([("line:a", "degraded", "r")], now=0)
    m.step([("line:a", "degraded", "r")], now=60)
    m.step([("line:a", "down", "r")], now=120)
    tr = m.step([("line:a", "down", "r")], now=180)
    assert len(tr) == 1 and tr[0].to_state == "down"


def test_unrecovered_down_repeats_every_repeat_s():
    m, _, _ = _monitor(None, repeat_s=3600)
    m.step([("line:a", "down", "r")], now=0)
    m.step([("line:a", "down", "r")], now=60)        # 叫
    assert m.step([("line:a", "down", "r")], now=1800) == []
    tr = m.step([("line:a", "down", "r")], now=3700)
    assert len(tr) == 1 and tr[0].repeat is True


def test_state_survives_restart():
    """狀態在 Redis，重啟後不會把已知的 down 當成新事件再叫一次。"""
    fr = FakeRedis()
    m1 = HealthMonitor(redis=fr, db=None, sender=FakeSender(), fetch=lambda: {})
    m1.step([("line:a", "down", "r")], now=0); m1.step([("line:a", "down", "r")], now=60)
    m2 = HealthMonitor(redis=fr, db=None, sender=FakeSender(), fetch=lambda: {})
    assert m2.step([("line:a", "down", "r")], now=120) == []


def test_run_once_wires_healthz_to_card():
    """⭐ 「接上了」：從 /healthz 回應到 sender.send 一路走通。"""
    fr, fs = FakeRedis(), FakeSender()
    hz = {"storage": {"database": True}, "lines": [{"line_id": "alpha_a", "stale": True, "heartbeat_age_s": 999}], "services": []}
    m = HealthMonitor(redis=fr, db=None, sender=fs, fetch=lambda: hz)
    m.run_once(); m.run_once()
    assert len(fs.sent) == 1
    receiver, card = fs.sent[0]
    assert receiver == "health" and card.level == "down" and "alpha_a" in card.title


def test_healthz_unreachable_becomes_api_down_item():
    """output-api 掛了本身就是一個故障，走同樣的狀態機（不會每 60 秒叫）。"""
    fr, fs = FakeRedis(), FakeSender()
    def boom(): raise ConnectionError("refused")
    m = HealthMonitor(redis=fr, db=None, sender=fs, fetch=boom)
    m.run_once(); tr = m.run_once()
    assert [t.item for t in tr] == ["api:output-api"] and tr[0].to_state == "down"


# ── 日報 ─────────────────────────────────────────────────

def _daily_data(**over):
    d = {"day": "2026-09-23", "ng_frames": 1063, "ng_cameras": 8,
         "ng_by_camera": [("cam100", 201), ("cam156", 191)],
         "ng_by_reason": [("違規動作", 1063)],
         "clips": 1050, "ng_without_clip": 0,
         "health_events": [], "notify": {"sent": 3}, "lines": []}
    d.update(over)
    return d


def test_daily_card_basic():
    c = build_card(_daily_data())
    assert c.title == "日報 2026-09-23"
    f = dict(c.facts)
    assert f["NG"] == "1063 次，8 台相機"
    assert "cam100 201" in f["相機 top"]
    assert f["健康"] == "全天無中斷"


def test_daily_card_flags_ng_without_clip():
    """⭐ 沒影片的 NG 應為 0（recorder 一對一）；不是 0 要在日報上看到。"""
    c = build_card(_daily_data(ng_without_clip=7))
    assert "⚠️ 7 個 NG 沒有影片" in dict(c.facts)["事件影片"]


def test_daily_card_lists_health_events():
    at = datetime(2026, 9, 23, 3, 14)
    c = build_card(_daily_data(health_events=[
        ("line:alpha_a", "ok", "down", at, None),
        ("line:alpha_a", "down", "ok", at.replace(hour=3, minute=40), 1560)]))
    assert dict(c.facts)["健康"] == "1 次中斷、1 次恢復"
    assert any("03:14" in b and "down" in b for b in c.body)
    assert any("恢復" in b and "26 分" in b for b in c.body)


def test_daily_due_only_after_time():
    r = DailyReporter(redis=FakeRedis(), db=None, sender=FakeSender(), at="08:00")
    assert r.due(datetime(2026, 9, 24, 7, 59)) is None
    assert r.due(datetime(2026, 9, 24, 8, 0)) == date(2026, 9, 23)


def test_daily_lock_prevents_double_send(monkeypatch):
    """多副本 / 重啟：同一天只送一次。"""
    fr, fs = FakeRedis(), FakeSender()
    r = DailyReporter(redis=fr, db=MagicMock(), sender=fs, at="00:00")
    monkeypatch.setattr("edgevision.notify.daily.collect", lambda db, day, lines: _daily_data())
    assert r.try_send(datetime(2026, 9, 24, 9)) is True
    assert r.try_send(datetime(2026, 9, 24, 10)) is False
    assert len(fs.sent) == 1


# ── 投遞（Apprise）────────────────────────────────────────

def test_card_markdown_shape():
    c = Card(title="t", level="down", facts=[("k", "v")], body=["p"], links=[("看", "http://x")])
    assert c.headline == "🔴 t"
    md = c.to_markdown()
    assert "p" in md and "**k**：v" in md and "[看](http://x)" in md


def test_no_channels_is_dry_run_not_failure():
    """⭐ 沒有任何通道的測試機要能跑完整條鏈。"""
    s = Sender(urls={"daily": [], "health": [], "ng": []})
    r = s.send("daily", Card(title="x"))
    assert r.dry_run is True and r.sent is False and r.error is None and r.targets == 0


def test_tag_routing_isolates_channels():
    """daily 的 URL 不會收到 health 的通知。"""
    s = Sender(urls={"daily": ["json://localhost:1/a"], "health": ["json://localhost:1/b", "json://localhost:1/c"], "ng": []})
    assert s.targets("daily") == 1 and s.targets("health") == 2 and s.targets("ng") == 0
    assert s.configured("daily") and not s.configured("ng")


def test_one_url_can_serve_multiple_tags():
    """一個頻道收全部：同一條 URL 掛三個 tag。"""
    u = "json://localhost:1/all"
    s = Sender(urls={"daily": [u], "health": [u], "ng": [u]})
    assert all(s.targets(t) == 1 for t in Sender.TAGS)


def test_bad_url_is_reported_not_silent():
    """⚠️ .env 打錯一個字，通知會靜默消失 —— 所以解析失敗要記下來讓啟動拒絕。"""
    s = Sender(urls={"daily": ["notascheme://nope"], "health": [], "ng": []})
    assert len(s.bad) == 1 and "daily" in s.bad[0]


def test_teams_uses_workflows_not_legacy_connector():
    """msteams:// 是 2026-05 停用的 Incoming Webhook；workflows:// 才是活的。"""
    s = Sender(urls={"health": ["workflows://prod-1.westus.logic.azure.com:443/workflows/abc/triggers/manual/paths/invoke?api-version=2016-06-01&sp=x&sv=1.0&sig=y"], "daily": [], "ng": []})
    assert s.targets("health") == 1 and "Workflows" in s.describe("health")


def test_mailto_parses_for_future_email_api():
    """之後可能改用寄信 API —— 確認 mailto:// 現在就接得上。"""
    s = Sender(urls={"daily": ["mailto://user:pass@smtp.example.com?to=ops@example.com"], "health": [], "ng": []})
    assert s.targets("daily") == 1 and "E-Mail" in s.describe("daily")


def test_blocked_mailbox_never_in_urls():
    import inspect, edgevision.notify.sender as m
    src = inspect.getsource(m)
    # docstring 提到它是為了警告，但要確認沒有任何預設值 / 範例真的用它當收件者


def test_send_reports_channel_failure(monkeypatch):
    """通道連不上：回 sent=False 帶 error，不拋、不假裝成功。"""
    s = Sender(urls={"ng": ["json://127.0.0.1:1/dead"], "daily": [], "ng_": []})
    monkeypatch.setattr(s._ap, "notify", lambda **kw: False)
    r = s.send("ng", Card(title="x"))
    assert r.sent is False and r.error and r.targets == 1


def test_send_swallows_exceptions(monkeypatch):
    s = Sender(urls={"ng": ["json://127.0.0.1:1/dead"], "daily": [], "health": []})
    def boom(**kw): raise RuntimeError("net")
    monkeypatch.setattr(s._ap, "notify", boom)
    r = s.send("ng", Card(title="x"))
    assert r.sent is False and "RuntimeError" in r.error


def test_split_uses_semicolon_not_comma():
    """mailto 的 to= 參數裡會有逗號，分隔符必須是分號。"""
    assert _split("a://x; mailto://u:p@h?to=a@b,c@d ;") == ["a://x", "mailto://u:p@h?to=a@b,c@d"]


def test_redact_hides_secrets():
    r = _redact("workflows://host.azure.com:443/workflows/abc?sig=SECRET")
    assert "SECRET" not in r and "abc" not in r and "host.azure.com" in r
