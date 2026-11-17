"""消費者心跳 → /healthz services[]。

⚠️ 這守的是「第二層健康」：docker healthcheck 只答活著嗎，這裡答在做事嗎。
   line_c 死 20 天沒人發現、recorder 產出 25 GB 孤兒 —— 都是這一層沒人看。
"""
from __future__ import annotations

import json
from contextlib import contextmanager
from unittest.mock import MagicMock

import pytest

from edgevision.storage.heartbeat import (
    HEARTBEAT_INTERVAL_S,
    Heartbeat,
    service_status,
)


def _db(cursor=None):
    cur = cursor or MagicMock()
    db = MagicMock()

    @contextmanager
    def fc():
        yield cur
    db.cursor = fc
    return db, cur


def _broken_db():
    db = MagicMock()

    @contextmanager
    def boom():
        raise RuntimeError("db down")
        yield
    db.cursor = boom
    return db


# ── 寫入端 ────────────────────────────────────────────────

def test_beat_upserts_with_metrics():
    db, cur = _db()
    hb = Heartbeat(db, service="recorder", instance="")
    hb._metrics = lambda: {"orphans": 3, "made": 10}
    assert hb.beat() is True
    sql, params = cur.execute.call_args[0]
    assert "ON CONFLICT (service, instance)" in sql, "多次心跳要 upsert 同一列，不能一直 INSERT"
    assert params[0] == "recorder"
    assert json.loads(params[2]) == {"orphans": 3, "made": 10}


def test_metrics_callback_failure_does_not_stop_heartbeat():
    """指標回呼壞了，心跳仍要寫 —— 否則 /healthz 會把它判 stale，看起來像整個服務死了。"""
    db, cur = _db()
    hb = Heartbeat(db, service="x", instance="")
    hb._metrics = lambda: 1 / 0
    assert hb.beat() is True
    assert json.loads(cur.execute.call_args[0][1][2]) == {"metrics_error": True}


def test_db_failure_is_swallowed():
    """⚠️ 心跳失敗不能拖垮服務。DB 抖一下只是「看不到」，不是「壞了」。"""
    hb = Heartbeat(_broken_db(), service="x", instance="")
    assert hb.beat() is False          # 不拋
    assert hb.failures == 1


def test_interval_is_under_half_the_stale_threshold():
    """間隔超過 stale 門檻的一半，一次網路抖動就會被誤判失聯。"""
    from edgevision.api.config_api import HEARTBEAT_STALE_S
    assert HEARTBEAT_INTERVAL_S <= HEARTBEAT_STALE_S / 2


def test_instance_defaults_to_env(monkeypatch):
    monkeypatch.setenv("HEARTBEAT_INSTANCE", "pod-7")
    assert Heartbeat(MagicMock(), service="x").instance == "pod-7"


# ── 讀取端：stale / degraded 判定 ─────────────────────────

def _status(rows, stale_after_s=60):
    cur = MagicMock()
    cur.fetchall.return_value = rows
    db, _ = _db(cur)
    return {(r["service"], r["instance"]): r for r in service_status(db, stale_after_s=stale_after_s)}


def test_fresh_heartbeat_with_good_metrics_is_ok():
    r = _status([("recorder", "", {"orphans": 0}, 5.0, 100.0)])
    assert r[("recorder", None)]["state"] == "ok"


def test_old_heartbeat_is_stale_regardless_of_metrics():
    """心跳過期就是 stale，指標再漂亮也沒用 —— 那是舊快照。"""
    r = _status([("recorder", "", {"orphans": 0}, 900.0, 100.0)])
    assert r[("recorder", None)]["state"] == "stale"
    assert r[("recorder", None)]["reason"] == "心跳過期"


def test_null_heartbeat_is_stale():
    r = _status([("recorder", "", {}, None, 100.0)])
    assert r[("recorder", None)]["state"] == "stale"


@pytest.mark.parametrize("metrics,expect_state,expect_in_reason", [
    ({"orphans": 0}, "ok", None),
    ({"orphans": 2}, "degraded", "孤兒"),
])
def test_recorder_orphans_degrade(metrics, expect_state, expect_in_reason):
    """⭐ 孤兒 > 0 就要亮 —— 先前 25 GB 就是因為沒人看得到。"""
    r = _status([("recorder", "", metrics, 5.0, 100.0)])[("recorder", None)]
    assert r["state"] == expect_state
    if expect_in_reason:
        assert expect_in_reason in r["reason"]


@pytest.mark.parametrize("metrics,expect_state", [
    ({"lag": 100, "last_write_s": 5}, "ok"),
    ({"lag": 9000, "last_write_s": 5}, "degraded"),      # 積壓
    ({"lag": 100, "last_write_s": 300}, "degraded"),     # 太久沒寫 DB
    ({"lag": None, "last_write_s": 5}, "ok"),            # 拿不到 lag 不算壞
])
def test_event_processor_thresholds(metrics, expect_state):
    r = _status([("event-processor", "", metrics, 5.0, 100.0)])
    assert r[("event-processor", None)]["state"] == expect_state


@pytest.mark.parametrize("metrics,expect_state", [
    ({"cameras_alive": 8, "cameras_total": 8}, "ok"),
    ({"cameras_alive": 6, "cameras_total": 8}, "degraded"),
    ({"cameras_alive": 8, "cameras_total": 8, "blocked": True}, "degraded"),
    ({"cameras_alive": 0, "cameras_total": 0}, "ok"),     # 沒相機不算壞
])
def test_nas_recorder_thresholds(metrics, expect_state):
    r = _status([("nas-recorder", "", metrics, 5.0, 100.0)])
    assert r[("nas-recorder", None)]["state"] == expect_state


def test_unknown_service_is_never_degraded():
    """沒定義門檻的服務只看心跳，不亂判。"""
    r = _status([("something-new", "", {"whatever": 999}, 5.0, 100.0)])
    assert r[("something-new", None)]["state"] == "ok"


def test_missing_table_returns_empty_not_crash():
    """⚠️ 舊 DB 沒有 service_status 表時 /healthz 不能掛 —— 那是 liveness probe。"""
    assert service_status(_broken_db(), stale_after_s=60) == []


def test_none_db_returns_empty():
    assert service_status(None, stale_after_s=60) == []


# ── 契約：schema 要跟著 ─────────────────────────────────────

def test_healthout_declares_services():
    """⚠️ FastAPI 照 response_model 序列化 —— 沒宣告的欄位不會回給前端。"""
    from edgevision.api.schemas import HealthOut, ServiceStatus
    assert "services" in HealthOut.model_fields
    assert set(ServiceStatus.model_fields) >= {"service", "state", "reason", "heartbeat_age_s", "metrics"}


def test_service_status_state_is_closed_enum():
    from edgevision.api.schemas import ServiceStatus
    with pytest.raises(Exception):
        ServiceStatus(service="x", state="weird")


def test_camera_states_from_runner_stats():
    from edgevision.api.server import camera_states
    st = {"cameras": {"cam156": 0.0, "cam193": 14.9}}
    got = {c["camera_id"]: c["state"] for c in camera_states(["cam156", "cam193", "cam230"], st, stale=False)}
    assert got == {"cam156": "no_frames", "cam193": "ok", "cam230": "unknown"}     # cam230：舊 runner 沒回報
    assert {c["state"] for c in camera_states(["cam156"], st, stale=True)} == {"stale"}
    assert camera_states(["cam156"], {}, stale=False)[0]["state"] == "unknown"   # 沒有 stats

    # 設定有誤被 runner 跳過的相機：skipped + 原因（同線其他台照跑）
    st = {"cameras": {"cam193": 14.9}, "skipped_cameras": {"cam156": "ValueError: roi_regions 少了 work 區"}}
    got = {c["camera_id"]: c for c in camera_states(["cam156", "cam193"], st, stale=False)}
    assert got["cam156"]["state"] == "skipped" and "roi_regions" in got["cam156"]["reason"]
    assert got["cam193"]["state"] == "ok"
