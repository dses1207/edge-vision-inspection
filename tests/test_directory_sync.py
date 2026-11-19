"""員工名冊同步：分頁、增量、只改身分不改角色、停用 = none、失敗不洩漏、沒設定照常。"""
from datetime import datetime, timezone

import pytest

from edgevision.api import directory_sync as D
from edgevision.api import permissions as P


class Resp:
    def __init__(self, code, j):
        self.status_code, self._j = code, j

    def json(self):
        return self._j


class FakeHTTP:
    def __init__(self, users, page_size=2):
        self.users, self.ps, self.calls = users, page_size, []

    def post(self, url, data):
        assert url.endswith("/connect/token") and data["grant_type"] == "client_credentials"
        assert data["scope"] == "directory.read"
        return Resp(200, {"access_token": "t", "expires_in": 86400})

    def get(self, url, params, headers):
        self.calls.append(dict(params))
        assert headers["Authorization"] == "Bearer t"
        p = params["page"]
        items = self.users[(p - 1) * self.ps: p * self.ps]
        return Resp(200, {"items": items, "page": p, "pageSize": self.ps, "total": len(self.users)})


class FakeCur:
    def __init__(self, db):
        self.db, self._one = db, None

    def execute(self, sql, args=None):
        if "INSERT INTO perm_users" in sql:
            created = args["account"] not in self.db.users
            u = self.db.users.setdefault(args["account"], {"role": "user"})
            u.update({k: v for k, v in args.items()})
            self._one = (created,)
        elif "directory_synced_at" in sql and sql.lstrip().startswith("INSERT"):
            self.db.synced = args[0]
        elif "directory_synced_at" in sql:
            self._one = (self.db.synced,) if self.db.synced else None

    def fetchone(self):
        return self._one


class FakeDB:
    def __init__(self):
        self.users, self.synced = {"corp\\alice": {"role": "admin", "name": "Alice"}}, None

    def cursor(self):
        from contextlib import contextmanager

        @contextmanager
        def cm():
            yield FakeCur(self)
        return cm()


USERS = [
    {"sub": "CORP\\alice", "name": "李", "department": "U100", "email": "r@x", "enabled": True},
    {"sub": "corp\\legacy", "name": "林", "department": "U100", "enabled": True, "employeeId": "E1"},
    {"sub": "corp\\left01", "name": "離", "department": "P10", "enabled": False},
    {"sub": "", "name": "沒帳號"},
]


def make(users=USERS):
    db, http = FakeDB(), FakeHTTP(users)
    ds = D.DirectorySync(db, client_factory=lambda: D.DirectoryClient("https://auth", "c", "s", http=http))
    return ds, db, http


def test_full_sync_pages_and_keeps_roles():
    ds, db, http = make()
    out = ds.run()
    assert out["status"] == "ok" and out["full"] is True and out["fetched"] == 3
    assert out["added"] == 2 and out["disabled"] == 1
    assert len(http.calls) == 2 and "updatedSince" not in http.calls[0]          # 4 筆、每頁 2 → 兩頁
    assert db.users["corp\\alice"]["role"] == "admin"                          # ⭐ 角色不動
    assert db.users["corp\\legacy"]["unit"] == "U100" and db.users["corp\\left01"]["enabled"] is False


def test_second_run_is_incremental_with_overlap():
    ds, db, http = make()
    ds.run()
    http.calls.clear()
    ds.run()
    since = datetime.fromisoformat(http.calls[0]["updatedSince"].replace("Z", "+00:00"))
    assert since < datetime.fromisoformat(db.synced)                              # 往前多抓 10 分鐘
    assert ds.run(full=True)["full"] is True


def test_not_configured_and_failure_do_not_raise(monkeypatch, tmp_path):
    monkeypatch.setattr(D, "CRED_DIR", str(tmp_path))
    ds = D.DirectorySync(FakeDB())
    assert ds.run()["status"] == "not_configured"

    class Boom(FakeHTTP):
        def get(self, *a, **k):
            raise ConnectionError("https://internal-host:443 refused")
    db = FakeDB()
    ds = D.DirectorySync(db, client_factory=lambda: D.DirectoryClient("https://auth", "c", "s", http=Boom([])))
    out = ds.run()
    assert out["status"] == "failed" and "internal-host" not in out["error"] and db.synced is None


def test_credentials_from_secret_files(monkeypatch, tmp_path):
    (tmp_path / "client_id").write_text("edgevision-directory\n")
    (tmp_path / "client_secret").write_text("x" * 64)
    monkeypatch.setattr(D, "CRED_DIR", str(tmp_path))
    assert D.credentials() == ("edgevision-directory", "x" * 64)


def test_disabled_person_gets_nothing_even_admin():
    b = {"units": {"U100": {"/live": "view"}}, "overrides": {}, "guest": {"/g": "view"}}
    assert P.effective_level({"account": "a", "role": "admin", "enabled": False}, "/x", b) == "none"
    assert P.effective_level({"account": "a", "role": "user", "unit": "U100", "enabled": False}, "/g", b) == "none"
    assert P.effective_level({"account": "a", "role": "user", "unit": "U100", "enabled": True}, "/live", b) == "view"


def test_disabled_admin_does_not_count_as_last_admin():
    cur_b = {"version": 1, "units": {}, "overrides": {}, "guest": {},
             "people": [{"account": "me", "role": "admin", "enabled": True},
                        {"account": "old", "role": "admin", "enabled": False}]}
    body = P.BundleIn(version=1, enforce=False, people=[P.PersonIn(account="me", role="user")])
    _, details = P.validate(body, cur_b, "me")
    assert "NO_ADMIN_LEFT" in {d["code"] for d in details}
