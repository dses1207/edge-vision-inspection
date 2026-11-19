"""頁面權限 API：身分只認 token、誰能存檔、安全檢查、版本衝突、標準回應外殼。

資料庫用記憶體假實作（FakeStore）；存檔判斷走正式的 plan_save()，SQL 在測試叢集驗。
"""
from datetime import datetime, timezone

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from edgevision.api import auth, envelope
from edgevision.api import permissions as P

NOW = datetime(2026, 11, 18, 1, 0, tzinfo=timezone.utc)


class FakeStore:
    def __init__(self):
        self.users = {"corp\\alice": {"account": "corp\\alice", "name": "", "unit": "", "email": None,
                                         "role": "admin", "first_login": None, "last_login": None}}
        self.b = {"version": 1, "enforce": False, "units": {"U100": {}}, "overrides": {},
                  "guest": {"/monitoring/live": "view", "/reports/summary": "view"}}
        self.audit = []

    def login(self, ident):
        created = ident.account not in self.users
        u = self.users.setdefault(ident.account, {"account": ident.account, "role": "user", "first_login": NOW})
        u.update(name=ident.name, unit=ident.unit, email=ident.email or None, last_login=NOW)
        u["first_login"] = u.get("first_login") or NOW
        return dict(u), created

    def person(self, account):
        return self.users.get(account)

    def bundle(self):
        return dict(self.b, people=[dict(u) for _, u in sorted(self.users.items())])

    def people(self, q, unit, enabled, page, size):
        rows = [dict(u) for _, u in sorted(self.users.items())
                if (not q or q in u["account"] or q in (u.get("name") or "")) and (not unit or u.get("unit") == unit)
                and (enabled is None or u.get("enabled", True) == enabled)]
        return rows[(page - 1) * size: page * size], len(rows)

    def save(self, actor, body):
        cur = self.bundle()
        after = P.plan_save(cur, actor.account, body)
        for acc, role in after["roles"].items():
            self.users[acc]["role"] = role
        self.b = {"version": cur["version"] + 1, "enforce": body.enforce, "units": after["units"],
                  "overrides": after["overrides"], "guest": after["guest"]}
        self.audit.append((actor.account, body.reason))
        return self.bundle()


class FakeSync:
    def __init__(self):
        self.last, self.runs = {"status": "never"}, []

    def synced_at(self):
        return NOW if self.runs else None

    def run(self, full=False, actor=""):
        self.runs.append((full, actor))
        self.last = {"status": "ok", "full": full, "fetched": 1}
        return self.last


TOKENS = {
    "alice": ("CORP\\alice", {"department": "U100"}),
    "hank": ("corp\\legacy", {"department": "U100"}),
    "orgonly": ("corp\\ab12cd", {"organization_code": "P10"}),
    "guestish": ("corp\\zz99zz", {}),
}


@pytest.fixture
def client(monkeypatch):
    def verify(token):
        if token not in TOKENS:
            raise ValueError("bad signature")
        sub, extra = TOKENS[token]
        return auth.Identity(actor=sub, display_name=f"name-{token}", email=f"{token}@example.com",
                             claims={"sub": sub, **extra})

    monkeypatch.setattr(auth, "verify_token", verify)
    store = FakeStore()
    app = FastAPI()
    envelope.install(app, ("/api/permissions",))
    sync = FakeSync()
    app.include_router(P.build_router({"directory_sync": sync}, store_factory=lambda s: store))
    c = TestClient(app, raise_server_exceptions=False)
    c.store, c.sync = store, sync
    return c


def H(tok):
    return {"Authorization": f"Bearer {tok}"}


def put(c, tok, **changes):
    data = c.get("/api/permissions", headers=H(tok)).json()["data"]
    data.update(changes)
    return c.put("/api/permissions", headers=H(tok), json=data)


# ── 身分 ─────────────────────────────────────────────────

def test_no_token_is_401_envelope_even_with_x_actor(client):
    """⭐ X-Actor 不算數：送 admin 的帳號也一樣 401。"""
    r = client.get("/api/permissions", headers={"X-Actor": "corp\\alice"})
    assert r.status_code == 401
    j = r.json()
    assert j["success"] is False and j["data"] is None and j["error"]["code"] == "UNAUTHORIZED"
    assert len(j["traceId"]) == 32 and r.headers["X-Trace-Id"] == j["traceId"]


def test_bad_token_is_401_without_leaking_reason(client):
    r = client.get("/api/permissions", headers=H("forged"))
    assert r.status_code == 401 and "signature" not in r.text


def test_login_takes_everything_from_token_not_body(client):
    """⭐ body 寫什麼都不採信：帳號、單位來自 token。第一次 201，之後 200。"""
    r = client.post("/api/permissions/login", headers=H("hank"),
                    json={"account": "corp\\alice", "unit": "ADMINS"})
    assert r.status_code == 201
    d = r.json()["data"]
    assert d["account"] == "corp\\legacy" and d["unit"] == "U100" and d["role"] == "user"
    assert d["firstLogin"].endswith("+08:00")
    assert client.post("/api/permissions/login", headers=H("hank")).status_code == 200


def test_account_is_lowercased_full_sub(client):
    """大寫的 CORP\\alice 對到種子資料的 corp\\alice（admin），不會變成新人。"""
    r = client.post("/api/permissions/login", headers=H("alice"))
    assert r.status_code == 200 and r.json()["data"]["role"] == "admin"


def test_me_reports_unit_claim_fallback(client):
    j = client.get("/api/permissions/me", headers=H("orgonly")).json()["data"]
    assert j["unit"] == "P10" and j["unitClaim"] == "organization_code" and j["registered"] is False
    assert j["role"] == "user"                                  # 沒登記過 = 第一次登入會建立的角色
    j = client.get("/api/permissions/me", headers=H("hank")).json()["data"]
    assert j["unitClaim"] == "department"


def test_traceparent_is_reused(client):
    tp = "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"
    r = client.get("/api/permissions", headers={**H("hank"), "traceparent": tp})
    assert r.json()["traceId"] == "4bf92f3577b34da6a3ce929d0e0e4736" == r.headers["X-Trace-Id"]


# ── 讀 ───────────────────────────────────────────────────

def test_get_shape(client):
    d = client.get("/api/permissions", headers=H("hank")).json()["data"]
    assert d["version"] == 1 and d["enforce"] is False
    assert d["units"] == {"U100": {}}            # 有設定但一頁都沒開
    assert d["guest"]["/monitoring/live"] == "view"
    assert d["people"][0]["account"] == "corp\\alice"


# ── 寫：誰能存 ───────────────────────────────────────────

def test_normal_user_cannot_put(client):
    client.post("/api/permissions/login", headers=H("hank"))
    r = put(client, "hank", enforce=True)
    assert r.status_code == 403 and r.json()["error"]["code"] == "FORBIDDEN"


def test_unit_edit_on_permissions_page_can_put(client):
    client.post("/api/permissions/login", headers=H("hank"))
    assert put(client, "alice", units={"U100": {"/system/permissions": "edit"}}).status_code == 200
    r = put(client, "hank", enforce=True, units={"U100": {"/system/permissions": "edit"}})
    assert r.status_code == 200 and r.json()["data"]["enforce"] is True and r.json()["data"]["version"] == 3


def test_override_none_revokes(client):
    client.post("/api/permissions/login", headers=H("hank"))
    put(client, "alice", units={"U100": {"/system/permissions": "edit"}},
        overrides={"CORP\\LEGACY": {"/system/permissions": "none"}})
    d = client.get("/api/permissions", headers=H("hank")).json()["data"]
    assert d["overrides"] == {"corp\\legacy": {"/system/permissions": "none"}}   # key 也轉小寫
    assert put(client, "hank", enforce=True).status_code == 403


def test_put_only_takes_role_from_people(client):
    client.post("/api/permissions/login", headers=H("hank"))
    r = put(client, "alice", people=[{"account": "corp\\legacy", "role": "guest", "name": "X", "unit": "HACK"}])
    p = next(x for x in r.json()["data"]["people"] if x["account"] == "corp\\legacy")
    assert p["role"] == "guest" and p["unit"] == "U100"


# ── 寫：安全檢查 ─────────────────────────────────────────

def test_cannot_remove_last_admin_or_lock_self_out(client):
    r = put(client, "alice", people=[{"account": "corp\\alice", "role": "user"}])
    assert r.status_code == 422
    codes = {d["code"] for d in r.json()["error"]["details"]}
    assert {"NO_ADMIN_LEFT", "SELF_LOCKOUT"} <= codes


def test_invalid_values_are_422_with_fields(client):
    r = put(client, "alice", units={"U100": {"/a": "none"}}, overrides={"x": {"/b": "admin"}},
            guest={"nopath": "view"}, people=[{"account": "corp\\alice", "role": "root"}])
    assert r.status_code == 422
    got = {(d["field"], d["code"]) for d in r.json()["error"]["details"]}
    assert ("units.U100./a", "INVALID_LEVEL") in got
    assert ("overrides.x./b", "INVALID_LEVEL") in got
    assert ("guest.nopath", "INVALID_PAGE") in got
    assert ("people[0].role", "INVALID_ROLE") in got


def test_unknown_account_role_is_422(client):
    r = put(client, "alice", people=[{"account": "corp\\nobody", "role": "admin"}])
    assert r.status_code == 422 and r.json()["error"]["details"][0]["code"] == "UNKNOWN_ACCOUNT"


def test_stale_version_is_409(client):
    data = client.get("/api/permissions", headers=H("alice")).json()["data"]
    assert client.put("/api/permissions", headers=H("alice"), json=data).status_code == 200
    r = client.put("/api/permissions", headers=H("alice"), json=data)      # 同一個舊 version 再存
    assert r.status_code == 409 and r.json()["error"]["code"] == "CONFLICT"


def test_wrong_types_are_400_missing_fields_422(client):
    r = client.put("/api/permissions", headers=H("alice"), json={"version": "abc", "enforce": False})
    assert r.status_code == 400 and r.json()["error"]["code"] == "BAD_REQUEST"
    r = client.put("/api/permissions", headers=H("alice"), json={"enforce": False})
    assert r.status_code == 422 and r.json()["error"]["details"][0]["field"] == "version"


def test_unexpected_error_is_500_envelope(client, monkeypatch):
    monkeypatch.setattr(client.store, "bundle", lambda: 1 / 0)
    r = client.get("/api/permissions", headers=H("alice"))
    assert r.status_code == 500
    j = r.json()
    assert j["error"]["code"] == "INTERNAL_ERROR" and "division" not in r.text and j["traceId"]


def test_reason_goes_to_audit(client):
    put(client, "alice", reason="開放 U100 看監看")
    assert client.store.audit[-1] == ("corp\\alice", "開放 U100 看監看")


# ── 規則 ─────────────────────────────────────────────────

@pytest.mark.parametrize("person,page,want", [
    ({"account": "a", "role": "admin", "unit": ""}, "/x", "edit"),
    ({"account": "a", "role": "user", "unit": "U100"}, "/live", "view"),
    ({"account": "a", "role": "user", "unit": "U100"}, "/other", "none"),        # 單位有設定但沒列 → none，不落到訪客
    ({"account": "b", "role": "user", "unit": "U100"}, "/live", "none"),         # 個人例外收回
    ({"account": "a", "role": "user", "unit": "NEW"}, "/guestpage", "view"),    # 單位沒設定 → 訪客範本
    ({"account": "a", "role": "guest", "unit": "U100"}, "/live", "none"),        # guest 不吃單位
    (None, "/guestpage", "view"),                                               # 沒登記、沒單位 → 訪客範本
    ({"account": "c", "unit": "U100"}, "/live", "view"),                          # 沒登記但 token 有單位 → 當 user 吃單位
])
def test_effective_level(person, page, want):
    b = {"units": {"U100": {"/live": "view"}}, "overrides": {"b": {"/live": "none"}}, "guest": {"/guestpage": "view"}}
    assert P.effective_level(person, page, b) == want


def test_old_endpoints_keep_default_error_format():
    """envelope 只套在 /api/permissions：其他路徑驗證錯誤仍是 FastAPI 預設 {"detail": [...]}。"""
    app = FastAPI()
    envelope.install(app, ("/api/permissions",))

    @app.get("/api/other")
    def other(n: int):
        return {"n": n}

    r = TestClient(app).get("/api/other?n=abc")
    assert r.status_code == 422 and "detail" in r.json() and "X-Trace-Id" in r.headers


# ── 員工名冊 ─────────────────────────────────────────────

def _synced(c, acc, **kw):
    c.store.users[acc] = {"account": acc, "name": "N", "unit": "U100", "email": None, "role": "user",
                          "first_login": None, "last_login": None, "enabled": True, **kw}


def test_people_search_paginated(client):
    _synced(client, "corp\\legacy")
    _synced(client, "corp\\left01", enabled=False)
    r = client.get("/api/permissions/people?unit=U100&enabled=true&pageSize=500", headers=H("hank"))
    j = r.json()
    assert r.status_code == 200 and [p["account"] for p in j["data"]] == ["corp\\legacy"]
    assert j["pagination"]["pageSize"] == 100 and j["warnings"][0]["code"] == "PAGE_SIZE_LIMITED"
    assert client.get("/api/permissions/people", headers={"X-Actor": "x"}).status_code == 401


def test_can_make_synced_person_admin_before_first_login(client):
    """⭐ 名冊帶進來、還沒登入過的人也能直接設成 admin（legacy 的情況）。"""
    _synced(client, "corp\\legacy")
    r = put(client, "alice", people=[{"account": "corp\\legacy", "role": "admin"}])
    assert r.status_code == 200
    assert client.store.users["corp\\legacy"]["role"] == "admin"


def test_me_reports_disabled(client):
    _synced(client, "corp\\legacy", enabled=False)
    d = client.get("/api/permissions/me", headers=H("hank")).json()["data"]
    assert d["registered"] is True and d["enabled"] is False


def test_directory_sync_needs_edit(client):
    client.post("/api/permissions/login", headers=H("hank"))
    assert client.post("/api/permissions/directory/sync", headers=H("hank")).status_code == 403
    r = client.post("/api/permissions/directory/sync?full=true", headers=H("alice"))
    assert r.status_code == 200 and r.json()["data"]["last"]["full"] is True
    assert client.sync.runs == [(True, "corp\\alice")]
    s = client.get("/api/permissions/directory", headers=H("hank")).json()["data"]
    assert s["lastSuccessAt"].endswith("+08:00") and s["last"]["status"] == "ok"
