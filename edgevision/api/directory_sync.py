"""員工名冊同步：AuthServer 的員工名冊 API（AD＋HR）→ perm_users。

為什麼：以前人員要登入過一次才有資料，沒辦法事先給權限；離職、調單位也要等本人再登入才更新。

    名冊 API   GET {AUTH_SERVER}/api/directory/users（分頁，含停用帳號 enabled=false；?updatedSince= 只回有變動的）
    憑證       OAuth2 client_credentials，scope directory.read；client_id / client_secret 在 Secret edgevision-directory-client
              （scripts/register_directory_client.sh 註冊＋建立，不進 git）。沒掛到 → 不同步，其他照常。
    頻率每小時抓一次有變動的（從上次成功的開始時間往前多抓 10 分鐘）；每 24 小時整份重抓一次。
              AuthServer 自己每天 02:00 從 AD 與 HR 同步。

同步只更新「這個人是誰」：姓名、單位（department）、email、在職（enabled）、職稱、工號。
角色與個人例外一律不碰（那是權限頁管理的）；名冊新出現的人角色是 user，跟第一次登入一樣。
停用的人（enabled=false）在 effective_level 一律 none，連 admin 也一樣。
"""

from __future__ import annotations

import logging
import os
import threading
import time
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)
_TZ = ZoneInfo("Asia/Taipei")

CRED_DIR = os.environ.get("DIRECTORY_CLIENT_DIR", "/etc/edgevision/directory-client")
INTERVAL_S = int(os.environ.get("DIRECTORY_SYNC_INTERVAL_S", 3600))
FULL_EVERY_S = int(os.environ.get("DIRECTORY_FULL_SYNC_S", 86400))
PAGE_SIZE = 1000                                     # 名冊 API 上限
OVERLAP = timedelta(minutes=10)
SCOPE = "directory.read"


def credentials() -> tuple[str, str] | None:
    try:
        with open(os.path.join(CRED_DIR, "client_id"), encoding="utf-8") as f:
            cid = f.read().strip()
        with open(os.path.join(CRED_DIR, "client_secret"), encoding="utf-8") as f:
            sec = f.read().strip()
    except FileNotFoundError:
        return None
    return (cid, sec) if cid and sec else None


class DirectoryClient:
    """名冊 API 的 HTTP 端；token 有效 24 小時，快取到過期前 5 分鐘。"""

    def __init__(self, base: str, client_id: str, client_secret: str, verify: bool = False, http=None) -> None:
        import httpx

        self.base, self.cid, self.secret = base.rstrip("/"), client_id, client_secret
        # ⚠️ 同 auth.py：AuthServer 是內網自簽憑證，AUTH_VERIFY_TLS 預設關
        self.http = http or httpx.Client(verify=verify, timeout=60.0)
        self._tok: tuple[float, str] = (0.0, "")

    def token(self) -> str:
        at, tok = self._tok
        if tok and time.time() < at:
            return tok
        r = self.http.post(f"{self.base}/connect/token",
                           data={"grant_type": "client_credentials", "client_id": self.cid,
                                 "client_secret": self.secret, "scope": SCOPE})
        if r.status_code != 200:
            raise RuntimeError(f"拿不到名冊 token（HTTP {r.status_code}）")
        j = r.json()
        self._tok = (time.time() + max(60, int(j.get("expires_in", 3600)) - 300), j["access_token"])
        return self._tok[1]

    def users(self, since: datetime | None):
        page = 1
        while True:
            params: dict = {"page": page, "pageSize": PAGE_SIZE}
            if since:
                params["updatedSince"] = since.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            r = self.http.get(f"{self.base}/api/directory/users", params=params,
                              headers={"Authorization": f"Bearer {self.token()}"})
            if r.status_code == 401:
                self._tok = (0.0, "")
            if r.status_code != 200:
                raise RuntimeError(f"名冊 API 回 HTTP {r.status_code}")
            j = r.json()
            items = j.get("items") or []
            yield from items
            if not items or page * int(j.get("pageSize") or PAGE_SIZE) >= int(j.get("total") or 0):
                return
            page += 1


def row_of(u: dict) -> dict | None:
    """名冊一筆 → perm_users 欄位；沒有 sub 的不收。"""
    from edgevision.api.permissions import norm_account

    acc = norm_account(u.get("sub") or "")
    if not acc or len(acc) > 200:
        return None
    return {"account": acc, "name": (u.get("name") or "")[:200], "unit": (u.get("department") or "")[:100],
            "email": (u.get("email") or None), "enabled": bool(u.get("enabled", True)),
            "unit_name": (u.get("departmentName") or None), "title": (u.get("title") or None),
            "employee_id": (u.get("employeeId") or None)}


UPSERT = """INSERT INTO perm_users (account, name, unit, email, role, enabled, unit_name, title, employee_id,
                                    synced_at, updated_by)
            VALUES (%(account)s, %(name)s, %(unit)s, %(email)s, 'user', %(enabled)s, %(unit_name)s, %(title)s,
                    %(employee_id)s, now(), 'directory')
            ON CONFLICT (account) DO UPDATE SET
                name = COALESCE(NULLIF(EXCLUDED.name, ''), perm_users.name),
                unit = EXCLUDED.unit, email = COALESCE(EXCLUDED.email, perm_users.email),
                enabled = EXCLUDED.enabled, unit_name = EXCLUDED.unit_name, title = EXCLUDED.title,
                employee_id = EXCLUDED.employee_id, synced_at = now()
            RETURNING (xmax = 0) AS created"""


class DirectorySync:
    """背景執行緒；control-api 只有 1 個副本，不用搶鎖。"""

    def __init__(self, db, client_factory=None) -> None:
        self.db = db
        self.client_factory = client_factory or self._default_client
        self.last: dict = {"status": "never"}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_full = 0.0

    @staticmethod
    def _default_client():
        from edgevision.api import auth

        cred = credentials()
        if not cred:
            return None
        verify = os.environ.get("AUTH_VERIFY_TLS", "false").lower() in ("1", "true", "yes")
        return DirectoryClient(os.environ.get("DIRECTORY_API", auth.server_base()), *cred, verify=verify)

    def synced_at(self) -> datetime | None:
        with self.db.cursor() as cur:
            cur.execute("SELECT value FROM perm_settings WHERE key = 'directory_synced_at'")
            r = cur.fetchone()
        return datetime.fromisoformat(r[0]) if r and r[0] else None

    def run(self, full: bool = False, actor: str = "system:directory-sync") -> dict:
        """跑一次。回結果（也存在 self.last）；失敗不丟例外，status=failed。"""
        if not self._lock.acquire(blocking=False):
            return {**self.last, "status": "running"}
        started = datetime.now(timezone.utc)
        try:
            client = self.client_factory()
            if client is None:
                self.last = {"status": "not_configured", "at": started.astimezone(_TZ).isoformat()}
                return self.last
            last_ok = None if full else self.synced_at()
            since = last_ok - OVERLAP if last_ok else None
            rows = [r for r in (row_of(u) for u in client.users(since)) if r]
            added = disabled = 0
            with self.db.cursor() as cur:
                for r in rows:
                    cur.execute(UPSERT, r)
                    added += bool(cur.fetchone()[0])
                    disabled += not r["enabled"]
                cur.execute("""INSERT INTO perm_settings (key, value, updated_by) VALUES ('directory_synced_at', %s, %s)
                               ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value, updated_by = EXCLUDED.updated_by,
                                   updated_at = now()""", (started.isoformat(), actor))
            if since is None:
                self._last_full = time.time()
            self.last = {"status": "ok", "at": started.astimezone(_TZ).isoformat(), "full": since is None,
                         "fetched": len(rows), "added": added, "disabled": disabled,
                         "seconds": round((datetime.now(timezone.utc) - started).total_seconds(), 1)}
            logger.info("名冊同步：%s", self.last)
        except Exception as exc:  # noqa: BLE001 —— 同步失敗只記錄，權限照上次的資料走
            logger.warning("名冊同步失敗：%s", exc)
            msg = str(exc)[:200] if isinstance(exc, RuntimeError) else type(exc).__name__   # 不回內部網址
            self.last = {"status": "failed", "at": started.astimezone(_TZ).isoformat(), "error": msg}
        finally:
            self._lock.release()
        return self.last

    def start(self) -> None:
        if self._thread or os.environ.get("DIRECTORY_SYNC", "true").lower() == "false":
            return

        def loop():
            if self._stop.wait(30):                       # 等服務起穩
                return
            while True:
                self.run(full=time.time() - self._last_full >= FULL_EVERY_S)
                if self._stop.wait(INTERVAL_S):
                    return

        self._thread = threading.Thread(target=loop, name="directory-sync", daemon=True)
        self._thread.start()

    def shutdown(self) -> None:
        self._stop.set()
