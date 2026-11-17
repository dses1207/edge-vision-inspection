from edgevision_rules import Cooldown, PersistCounter


def test_persist_requires_consecutive_frames():
    p = PersistCounter(2)
    assert p.update(True) is False
    assert p.update(True) is True
    assert p.update(False) is False
    assert p.update(True) is False, "中斷後要重新累計"


def test_cooldown_with_injected_clock():
    now = [0.0]
    c = Cooldown(120, clock=lambda: now[0])
    assert c.fire() is True
    now[0] = 60
    assert c.fire() is False
    now[0] = 121
    assert c.fire() is True


# ── EventHistory / WaitWindow（舊腳本）────────────────────


def test_event_history_seen_within_and_pruning():
    from edgevision_rules import EventHistory
    h = EventHistory(keep_seconds=30, clock=lambda: 0.0)
    h.record(["hand"], now=0.0)
    h.record([], now=10.0)
    h.record(["part"], now=20.0)
    assert h.seen_within("hand", 25, now=20.0)            # 20 秒前，25 秒窗內
    assert not h.seen_within("hand", 15, now=20.0)        # 15 秒窗外
    h.record([], now=31.0)                                # 0.0 那筆超過 keep_seconds 被修掉
    assert len(h) == 3 and h.last_seen("hand") is None


def test_event_history_first_seen_gives_ordering():
    """順序檢查：A 要在 B / C 之前 —— 比的是各自「最早」出現的時刻。"""
    from edgevision_rules import EventHistory
    h = EventHistory(keep_seconds=10, clock=lambda: 0.0)
    h.record(["half"], now=1.0)
    h.record(["corner"], now=2.0)
    h.record(["half", "finish"], now=3.0)
    assert h.first_seen_within("half", 5, now=5.0) == 1.0     # 最早的 half 在 corner 之前 → 順序錯
    assert h.first_seen_within("corner", 5, now=5.0) == 2.0
    assert h.first_seen_within("finish", 5, now=5.0) == 3.0
    assert h.first_seen_within("half", 2.5, now=5.0) == 3.0   # 窗縮到 2.5 秒就只剩 3.0 那筆


def test_wait_window_satisfied_cancels_and_timeout_fires():
    from edgevision_rules import WaitWindow
    w = WaitWindow(15)
    assert w.update(trigger=False, satisfied=False, now=0) == "idle"
    assert w.update(trigger=True, satisfied=False, now=1) == "armed"
    assert w.update(trigger=True, satisfied=False, now=5) == "waiting"     # A 再出現不重啟計時
    assert w.update(trigger=False, satisfied=True, now=10) == "satisfied"
    assert not w.waiting
    assert w.update(trigger=True, satisfied=False, now=20) == "armed"
    assert w.update(trigger=False, satisfied=False, now=35) == "waiting"    # 剛好 15 秒還不算超時（> 才算）
    assert w.update(trigger=False, satisfied=False, now=35.1) == "timeout"
    assert not w.waiting


def test_wait_window_satisfied_beats_timeout_on_same_frame():
    """在等的時候先看 B，再看逾時。"""
    from edgevision_rules import WaitWindow
    w = WaitWindow(15)
    w.update(trigger=True, satisfied=False, now=0)
    assert w.update(trigger=False, satisfied=True, now=100) == "satisfied"


def test_wait_window_late_ok_false_treats_late_b_as_timeout():
    """late_ok=False：B 在 25 秒之後才來 → 照樣警報（late_ok=True 則算滿足）。"""
    from edgevision_rules import WaitWindow
    w = WaitWindow(25, late_ok=False)
    w.update(trigger=True, satisfied=False, now=0)
    assert w.update(trigger=False, satisfied=True, now=25) == "satisfied"      # 剛好 25 秒還算
    w.update(trigger=True, satisfied=False, now=100)
    assert w.update(trigger=False, satisfied=True, now=125.1) == "timeout"
