"""控制 API 身分關卡（第二期）：觀察模式放行＋記錄、強制模式 401、服務金鑰、路由樣板、記錄彙整。"""
from datetime import date

import pytest
from fastapi import FastAPI, Header
from fastapi.testclient import TestClient

from edgevision.api import auth, envelope, gate

SVC = "a" * 64


class FakeRecorder(gate.Recorder):
    def __init__(self):
        super().__init__(db=None)

    def rows(self):
        return [(k, v[0]) for k, v in self._buf.items()]


@pytest.fixture
def app(monkeypatch, tmp_path):
    (tmp_path / "alert-feedback").write_text(SVC)
    (tmp_path / "short").write_text("tooshort")                 # 少於 32 字元的不算
    monkeypatch.setattr(auth, "SERVICE_TOKEN_DIR", str(tmp_path))
    monkeypatch.setattr(auth, "_svc_cache", (0.0, {}))
    gate._tok_cache.clear()

    def verify(token):
        if token != "good":
            raise ValueError("bad")
        return auth.Identity(actor="corp\\alice")

    monkeypatch.setattr(auth, "verify_token", verify)
    rec = FakeRecorder()
    a = FastAPI()

    @a.get("/api/tasks/{task_id}")
    def task(task_id: str, x_actor: str | None = Header(default=None), x_channel: str | None = Header(default=None),
             authorization: str | None = Header(default=None)):
        ident, ch = auth.resolve(authorization, x_actor, None, x_channel)
        return {"actor": ident.actor, "type": ident.actor_type, "channel": ch, "unverified": ident.unverified}

    @a.get("/healthz")
    def hz():
        return {"ok": True}

    gate.install(a, lambda: rec)
    envelope.install(a, ("/api/permissions",))
    a.state.rec = rec
    return a


def mode(monkeypatch, enforce: bool):
    monkeypatch.setenv("AUTH_ENABLED", "true" if enforce else "false")
    monkeypatch.setenv("AUTH_ALLOW_HEADER_ACTOR", "false" if enforce else "true")


def test_observe_mode_passes_and_records(app, monkeypatch):
    mode(monkeypatch, enforce=False)
    c = TestClient(app)
    r = c.get("/api/tasks/train-20261110-012430-4999ee", headers={"X-Actor": "someone", "User-Agent": "curl/8"})
    assert r.status_code == 200 and r.json()["unverified"] is True
    (key, n), = app.state.rec.rows()
    day, method, route, actor, channel, reason, ua, client = key
    assert (method, route, actor, reason, ua) == ("GET", "/api/tasks/{task_id}", "someone", "no_token", "curl/8")
    assert day == date.today() and n == 1


def test_observe_counts_same_caller_once(app, monkeypatch):
    mode(monkeypatch, enforce=False)
    c = TestClient(app)
    for tid in ("a1", "b2", "c3"):
        c.get(f"/api/tasks/{tid}", headers={"X-Actor": "x"})
    (key, n), = app.state.rec.rows()
    assert n == 3                                           # 路由樣板相同 → 同一筆


def test_enforce_blocks_without_token(app, monkeypatch):
    mode(monkeypatch, enforce=True)
    c = TestClient(app)
    r = c.get("/api/tasks/123", headers={"X-Actor": "corp\\alice"})
    assert r.status_code == 401 and "X-Trace-Id" in r.headers
    (key, _), = app.state.rec.rows()
    assert key[2] == "/api/tasks/{}"                        # 被擋時還沒路由：id 換成 {}


def test_enforce_bad_token_blocked(app, monkeypatch):
    mode(monkeypatch, enforce=True)
    r = TestClient(app).get("/api/tasks/1", headers={"Authorization": "Bearer forged"})
    assert r.status_code == 401 and app.state.rec.rows()[0][0][5] == "bad_token"


def test_valid_token_passes_without_record(app, monkeypatch):
    mode(monkeypatch, enforce=True)
    r = TestClient(app).get("/api/tasks/1", headers={"Authorization": "Bearer good"})
    assert r.status_code == 200 and r.json()["actor"] == "corp\\alice" and app.state.rec.rows() == []


def test_service_token_is_system_identity(app, monkeypatch):
    """⭐ 服務金鑰 → system:<服務名>，X-Actor 說什麼都不算。"""
    mode(monkeypatch, enforce=True)
    r = TestClient(app).get("/api/tasks/1", headers={"X-Service-Token": SVC, "X-Actor": "corp\\alice",
                                                     "X-Channel": "cron"})
    assert r.status_code == 200
    assert r.json() == {"actor": "system:alert-feedback", "type": "system", "channel": "system", "unverified": False}


def test_bad_or_short_service_token_rejected(app, monkeypatch):
    mode(monkeypatch, enforce=True)
    c = TestClient(app)
    assert c.get("/api/tasks/1", headers={"X-Service-Token": "b" * 64}).status_code == 401
    assert c.get("/api/tasks/1", headers={"X-Service-Token": "tooshort"}).status_code == 401


def test_open_paths_need_nothing(app, monkeypatch):
    mode(monkeypatch, enforce=True)
    assert TestClient(app).get("/healthz").status_code == 200
    assert gate.needs_identity("/api/control/openapi.json", "GET") is False
    assert gate.needs_identity("/static/index.html", "GET") is False
    assert gate.needs_identity("/api/models", "OPTIONS") is False      # CORS 預檢
    assert gate.needs_identity("/api/control/auth/unverified", "GET") is True


def test_cron_channel_no_longer_400():
    """以前告警回流排程送 X-Channel: cron 會被擋成 400，回流根本沒跑。"""
    ident, ch = auth.resolve(None, "system:alert-feedback", "system", "cron")
    assert ch == "cron" and ident.unverified is True


class FakeCur:
    def __init__(self, sink):
        self.sink = sink

    def execute(self, sql, params):
        self.sink.append(params)


class FakeDB:
    def __init__(self):
        self.rows = []

    def cursor(self):
        from contextlib import contextmanager

        @contextmanager
        def cm():
            yield FakeCur(self.rows)
        return cm()


def test_recorder_flush_upserts_and_clears():
    db = FakeDB()
    rec = gate.Recorder(db)
    rec.add("GET", "/api/models", "x", "ui", "no_token", "ua", "192.0.2.84")
    rec.add("GET", "/api/models", "x", "ui", "no_token", "ua", "192.0.2.84")
    assert rec.flush() == 1 and db.rows[0][8] == 2           # count = 2
    assert rec.flush() == 0


def test_recorder_db_failure_does_not_raise():
    class Broken:
        def cursor(self):
            raise RuntimeError("db down")
    rec = gate.Recorder(Broken())
    rec.add("GET", "/api/x", "", "", "no_token", "", "")
    assert rec.flush() == 0


# ── 網址簽名（<video src> 這類帶不了 header 的）──────────────

from edgevision.api import signed_url  # noqa: E402


@pytest.fixture
def signing(monkeypatch, tmp_path):
    k = tmp_path / "key"
    k.write_text("k" * 64)
    monkeypatch.setattr(signed_url, "KEY_FILE", str(k))
    monkeypatch.setattr(signed_url, "_key_cache", (0.0, b""))


@pytest.fixture
def file_app(app, signing):
    @app.get("/api/media/videos/{vid}/stream")
    def stream(vid: int, download: bool = False):
        return {"vid": vid, "download": download}

    app.include_router(gate.build_router({"db": None}))
    return app


def test_signed_url_passes_in_enforce_mode(file_app, monkeypatch):
    """⭐ <video src> 帶簽名就能播，不用 Authorization header；也不記成未驗證。"""
    mode(monkeypatch, enforce=True)
    c = TestClient(file_app)
    url = signed_url.sign("/api/media/videos/7/stream")
    r = c.get(url + "&download=1")                           # 額外 query 不影響簽名
    assert r.status_code == 200 and r.json() == {"vid": 7, "download": True}
    assert file_app.state.rec.rows() == []
    assert c.get("/api/media/videos/7/stream").status_code == 401


def test_signature_bound_to_path_and_expiry(file_app, monkeypatch):
    mode(monkeypatch, enforce=True)
    c = TestClient(file_app)
    url = signed_url.sign("/api/media/videos/7/stream")
    assert c.get(url.replace("/7/", "/8/")).status_code == 401           # 換別支影片不行
    assert c.get(url[:-1] + ("0" if url[-1] != "0" else "1")).status_code == 401
    exp = int(url.split("exp=")[1].split("&")[0])
    assert signed_url.verify("/api/media/videos/7/stream", str(exp), url.split("sig=")[1], now=exp + 1) is False
    assert file_app.state.rec.rows()[0][0][5] == "bad_signature"


def test_only_file_routes_signable(signing):
    assert signed_url.signable("/api/media/recordings/siteb12/files/20261118_0800.mp4")
    assert signed_url.signable("/api/media/items/12/image")
    assert not signed_url.signable("/api/permissions")
    assert not signed_url.signable("/api/media/videos/7")
    with pytest.raises(ValueError):
        signed_url.sign("/api/tasks/1")
    # 簽名不能拿來打一般 API：就算 sig 對得上一個檔案路徑也一樣
    assert signed_url.verify("/api/tasks/1", "9999999999", "x") is False


def test_expiry_is_bucketed_for_cache(signing):
    """同一段時間內同一個檔案網址不變，瀏覽器快取才用得上；有效 ttl～2×ttl。"""
    t = 1_800_000_000
    assert signed_url.sign("/api/media/items/1/image", now=t) == signed_url.sign("/api/media/items/1/image", now=t + 60)
    e = signed_url.expiry(now=t, ttl=7200)
    assert 7200 <= e - t <= 14400


def test_sign_endpoint_envelope(file_app, monkeypatch):
    mode(monkeypatch, enforce=False)
    c = TestClient(file_app)
    r = c.post("/api/control/auth/signed-urls", json={"paths": ["/api/media/items/3/image"]})
    assert r.status_code == 200
    (d,) = r.json()["data"]
    assert d["url"].startswith("/api/media/items/3/image?exp=") and d["expiresAt"].endswith("+08:00")
    r = c.post("/api/control/auth/signed-urls", json={"paths": ["/api/media/items/3/image", "/api/tasks"]})
    assert r.status_code == 422 and r.json()["error"]["details"][0]["field"] == "paths[1]"


def test_missing_key_falls_back(monkeypatch, tmp_path):
    monkeypatch.setattr(signed_url, "KEY_FILE", str(tmp_path / "nope"))
    monkeypatch.setattr(signed_url, "_key_cache", (0.0, b""))
    u = signed_url.sign("/api/media/items/1/image")
    exp, sig = u.split("exp=")[1].split("&sig=")
    assert signed_url.verify("/api/media/items/1/image", exp, sig)
