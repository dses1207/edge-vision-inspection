"""ipcam-web 的頁面權限：資料存 PostgreSQL（deploy/postgres/init/27-permissions.sql），取代前端伺服器上的 JSON 檔。

    POST /api/permissions/login   登入後呼叫：用 token 建立 / 更新自己（不收 body）
    GET  /api/permissions/me      後端從 token 讀到的身分（對帳、除錯；不回 token）
    GET  /api/permissions         整包權限資料（含 version）
    PUT  /api/permissions         整包寫入（要有 /system/permissions 的 edit）

身分一律來自 SSO access token（Authorization: Bearer），不管 AUTH_ENABLED 怎麼設：
X-Actor 是自我宣告，權限資料若吃它，任何人送一個 header 就能把自己設成 admin。
帳號 = token 的 sub 小寫（corp\\legacy）：多個廠區 AD 分開，去掉前綴會撞名；大小寫 AuthServer 不保證。

人員來源：員工名冊（AuthServer，AD＋HR，edgevision/api/directory_sync.py 每小時同步）＋登入時用 token 更新。
名冊停用的人（enabled=false）一律 none。

判斷規則（第一個有值的算數；前端擋頁面用同一套，後端只拿來決定誰能 PUT）：
    0. 名冊停用 → 每頁 none
    1. 角色 admin → 每頁 edit
    2. 個人例外有這頁 → 例外的等級（可以是 none = 收回）
    3. 角色不是 guest 且單位有設定 → 單位在這頁的等級，沒列算 none
    4. 其他（guest、或單位沒設定）→ 訪客範本，沒列算 none

回應照標準回應外殼（edgevision/api/envelope.py）。
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any, Callable
from zoneinfo import ZoneInfo

from fastapi import APIRouter, Header, Query
from pydantic import BaseModel, ConfigDict, Field

from edgevision.api.envelope import ApiError, ApiResponse, Camel, current_trace_id, ok

logger = logging.getLogger(__name__)

PERM_PAGE = "/system/permissions"
ROLES = ("admin", "user", "guest")
UNIT_LEVELS = ("view", "edit")             # 單位、訪客範本：none 不存
OVERRIDE_LEVELS = ("none", "view", "edit")
# 單位 claim：前端規格寫 department；以前接 AuthServer 看到的是 organization_code。先讀前者，沒有再讀後者
UNIT_CLAIMS = ("department", "organization_code")
_TZ = ZoneInfo("Asia/Taipei")


# ── 規則 ─────────────────────────────────────────────────

def effective_level(person: dict | None, page: str, bundle: dict) -> str:
    """person = {account, role, unit}；沒登記過的人當 user（跟第一次登入建立的一樣），沒有單位就落到訪客範本。"""
    role = (person or {}).get("role") or "user"
    if (person or {}).get("enabled") is False:
        return "none"
    if role == "admin":
        return "edit"
    account = (person or {}).get("account", "")
    ov = bundle.get("overrides", {}).get(account, {})
    if page in ov:
        return ov[page]
    unit = (person or {}).get("unit", "")
    if role != "guest" and unit in bundle.get("units", {}):
        return bundle["units"][unit].get(page, "none")
    return bundle.get("guest", {}).get(page, "none")


def norm_account(sub: str) -> str:
    return (sub or "").strip().lower()


# ── 身分 ─────────────────────────────────────────────────

class Ident(BaseModel):
    account: str
    name: str = ""
    unit: str = ""
    unit_claim: str | None = None
    email: str = ""


def identify(authorization: str | None) -> Ident:
    from edgevision.api import auth

    token = authorization[7:].strip() if authorization and authorization.lower().startswith("bearer ") else ""
    if not token:
        raise ApiError(401, "UNAUTHORIZED", "SSO access token required")
    try:
        ident = auth.verify_token(token)
    except ValueError as exc:
        logger.info("[%s] token 驗證失敗：%s", current_trace_id(), exc)
        raise ApiError(401, "UNAUTHORIZED", "Invalid or expired token") from exc
    claims = ident.claims or {}
    unit, unit_claim = "", None
    for k in UNIT_CLAIMS:
        v = str(claims.get(k) or "").strip()
        if v:
            unit, unit_claim = v, k
            break
    return Ident(account=norm_account(ident.actor), name=ident.display_name or "", unit=unit,
                 unit_claim=unit_claim, email=ident.email or "")


# ── 輸入 / 輸出模型（camelCase） ─────────────────────────

class PersonOut(Camel):
    account: str
    name: str
    unit: str
    email: str | None = None
    role: str
    enabled: bool = True
    unit_name: str | None = None
    title: str | None = None
    employee_id: str | None = None
    first_login: str | None = None
    last_login: str | None = None
    synced_at: str | None = None


class MeOut(Camel):
    account: str
    name: str
    unit: str
    unit_claim: str | None
    email: str
    role: str
    registered: bool
    enabled: bool = True


class BundleOut(Camel):
    version: int
    enforce: bool
    people: list[PersonOut]
    units: dict[str, dict[str, str]]
    overrides: dict[str, dict[str, str]]
    guest: dict[str, str]


class DirectoryStatusOut(Camel):
    last_success_at: str | None
    last: dict[str, Any]


MAX_PAGE_SIZE = 100


class PersonIn(BaseModel):
    model_config = ConfigDict(extra="ignore")      # name、unit 等前端帶回來也不吃
    account: str
    role: str


class BundleIn(BaseModel):
    model_config = ConfigDict(extra="ignore")
    version: int
    enforce: bool
    people: list[PersonIn] = Field(default_factory=list)
    units: dict[str, dict[str, str]] = Field(default_factory=dict)
    overrides: dict[str, dict[str, str]] = Field(default_factory=dict)
    guest: dict[str, str] = Field(default_factory=dict)
    reason: str | None = None                      # 選填：為什麼改（MCP 工具必填，進異動紀錄）


def _iso(dt: datetime | None) -> str | None:
    return dt.astimezone(_TZ).isoformat() if dt else None


def _person(row: dict, with_email: bool = False) -> dict:
    out = {"account": row["account"], "name": row["name"], "unit": row["unit"], "role": row["role"],
           "enabled": row.get("enabled", True) is not False, "unitName": row.get("unit_name"),
           "title": row.get("title"), "employeeId": row.get("employee_id"),
           "firstLogin": _iso(row.get("first_login")), "lastLogin": _iso(row.get("last_login")),
           "syncedAt": _iso(row.get("synced_at"))}
    if with_email:
        out["email"] = row.get("email")
    return out


# ── 資料存取 ─────────────────────────────────────────────

_USER_COLS = ("account, name, unit, email, role, first_login, last_login, enabled, unit_name, title, employee_id,"
              " synced_at")
# 整包裡的 people 只放「有被管到的人」：名冊同步後 perm_users 是全組織，整包回會上千筆。
# 其他人用 GET /api/permissions/people 搜尋；PUT 設角色時照樣認得（save 會把 body 裡的帳號一起讀進來）。
_MANAGED = ("role <> 'user' OR last_login IS NOT NULL"
            " OR account IN (SELECT account FROM perm_user_overrides)")


def _rows(cur) -> list[dict]:
    names = [d[0] for d in cur.description]
    return [dict(zip(names, r)) for r in cur.fetchall()]


class Store:
    """所有 SQL 都在這裡；一個 db.cursor() 區塊 = 一個 transaction（離開時 commit、出錯 rollback）。"""

    def __init__(self, db) -> None:
        self.db = db

    def login(self, ident: Ident) -> tuple[dict, bool]:
        with self.db.cursor() as cur:
            cur.execute(
                f"""INSERT INTO perm_users (account, name, unit, email, role, first_login, last_login, updated_by)
                    VALUES (%s, %s, %s, %s, 'user', now(), now(), %s)
                    ON CONFLICT (account) DO UPDATE SET
                        name = EXCLUDED.name, unit = EXCLUDED.unit, email = EXCLUDED.email, last_login = now(),
                        first_login = COALESCE(perm_users.first_login, now())
                    RETURNING {_USER_COLS}, (xmax = 0) AS created""",
                (ident.account, ident.name, ident.unit, ident.email or None, ident.account))
            row = _rows(cur)[0]
        return row, bool(row.pop("created"))

    def person(self, account: str) -> dict | None:
        with self.db.cursor() as cur:
            cur.execute(f"SELECT {_USER_COLS} FROM perm_users WHERE account = %s", (account,))
            rows = _rows(cur)
        return rows[0] if rows else None

    def bundle(self) -> dict:
        with self.db.cursor() as cur:
            return self._bundle(cur)

    def people(self, q: str, unit: str, enabled: bool | None, page: int, size: int) -> tuple[list[dict], int]:
        where, args = ["TRUE"], []
        if q:
            where.append("(account ILIKE %s OR name ILIKE %s OR employee_id ILIKE %s)")
            args += [f"%{q}%"] * 3
        if unit:
            where.append("unit = %s")
            args.append(unit)
        if enabled is not None:
            where.append("enabled = %s")
            args.append(enabled)
        w = " AND ".join(where)
        with self.db.cursor() as cur:
            cur.execute(f"SELECT count(*) FROM perm_users WHERE {w}", args)
            total = cur.fetchone()[0]
            cur.execute(f"SELECT {_USER_COLS} FROM perm_users WHERE {w} ORDER BY unit, account LIMIT %s OFFSET %s",
                        args + [size, (page - 1) * size])
            return _rows(cur), total

    @staticmethod
    def _bundle(cur, lock: bool = False, extra: tuple[str, ...] = ()) -> dict:
        cur.execute("SELECT key, value FROM perm_settings WHERE key IN ('enforce', 'version')"
                    + (" FOR UPDATE" if lock else ""))
        s = dict(cur.fetchall())
        cur.execute(f"SELECT {_USER_COLS} FROM perm_users WHERE {_MANAGED} OR account = ANY(%s) ORDER BY account",
                    (list(extra),))
        people = _rows(cur)
        cur.execute("SELECT unit FROM perm_units ORDER BY unit")
        units: dict[str, dict[str, str]] = {r[0]: {} for r in cur.fetchall()}
        cur.execute("SELECT unit, page, level FROM perm_unit_pages ORDER BY unit, page")
        for u, p, lv in cur.fetchall():
            units.setdefault(u, {})[p] = lv
        cur.execute("SELECT account, page, level FROM perm_user_overrides ORDER BY account, page")
        overrides: dict[str, dict[str, str]] = {}
        for a, p, lv in cur.fetchall():
            overrides.setdefault(a, {})[p] = lv
        cur.execute("SELECT page, level FROM perm_guest_pages ORDER BY page")
        guest = dict(cur.fetchall())
        return {"version": int(s.get("version", "1")), "enforce": s.get("enforce", "false") == "true",
                "people": people, "units": units, "overrides": overrides, "guest": guest}

    def save(self, actor: Ident, body: BundleIn) -> dict:
        """整包替換。檢查失敗丟 ApiError，transaction 跟著 rollback。"""
        with self.db.cursor() as cur:
            extra = tuple({norm_account(p.account) for p in body.people} | {actor.account})
            cur_b = self._bundle(cur, lock=True, extra=extra)   # 鎖住 version 那列：同時存檔的人排隊
            after = plan_save(cur_b, actor.account, body)

            for acc, role in after["roles"].items():
                cur.execute("UPDATE perm_users SET role = %s, updated_by = %s, updated_at = now() "
                            "WHERE account = %s AND role <> %s", (role, actor.account, acc, role))
            cur.execute("DELETE FROM perm_units")                      # 單位頁面跟著 CASCADE
            for u, pages in after["units"].items():
                cur.execute("INSERT INTO perm_units (unit, updated_by) VALUES (%s, %s)", (u, actor.account))
                for p, lv in pages.items():
                    cur.execute("INSERT INTO perm_unit_pages (unit, page, level) VALUES (%s, %s, %s)", (u, p, lv))
            cur.execute("DELETE FROM perm_user_overrides")
            for a, pages in after["overrides"].items():
                for p, lv in pages.items():
                    cur.execute("INSERT INTO perm_user_overrides (account, page, level) VALUES (%s, %s, %s)",
                                (a, p, lv))
            cur.execute("DELETE FROM perm_guest_pages")
            for p, lv in after["guest"].items():
                cur.execute("INSERT INTO perm_guest_pages (page, level) VALUES (%s, %s)", (p, lv))
            for k, v in (("enforce", "true" if body.enforce else "false"), ("version", str(cur_b["version"] + 1))):
                cur.execute("UPDATE perm_settings SET value = %s, updated_by = %s, updated_at = now() WHERE key = %s",
                            (v, actor.account, k))
            new_b = self._bundle(cur)
            cur.execute("INSERT INTO perm_audit (actor, trace_id, reason, before, after) VALUES (%s, %s, %s, %s, %s)",
                        (actor.account, current_trace_id(), (body.reason or "").strip() or None,
                         json.dumps(_public(cur_b), ensure_ascii=False), json.dumps(_public(new_b), ensure_ascii=False)))
        return new_b


def plan_save(cur_b: dict, actor_account: str, body: BundleIn) -> dict:
    """存檔前的所有判斷（不碰資料庫）：權限 → 版本 → 值與安全檢查。不通過丟 ApiError。"""
    me = next((p for p in cur_b["people"] if p["account"] == actor_account), None)
    if effective_level(me, PERM_PAGE, cur_b) != "edit":
        raise ApiError(403, "FORBIDDEN", "Edit permission on the permissions page is required")
    if body.version != cur_b["version"]:
        raise ApiError(409, "CONFLICT", "Permissions were changed by someone else; reload and try again")
    after, details = validate(body, cur_b, actor_account)
    if details:
        raise ApiError(422, "VALIDATION_FAILED", "Request validation failed", details)
    return after


def _page_ok(p: str) -> bool:
    return isinstance(p, str) and p.startswith("/") and 1 < len(p) <= 200


def validate(body: BundleIn, cur_b: dict, actor_account: str) -> tuple[dict, list[dict]]:
    """回 (存檔後的樣子, 錯誤明細)。也檢查「至少一個 admin」與「存完自己還能改權限」。"""
    details: list[dict] = []

    def bad(field: str, code: str, msg: str):
        details.append({"field": field, "code": code, "message": msg})

    known = {p["account"]: p for p in cur_b["people"]}
    roles: dict[str, str] = {}
    for i, p in enumerate(body.people):
        acc = norm_account(p.account)
        if p.role not in ROLES:
            bad(f"people[{i}].role", "INVALID_ROLE", "角色只能是 admin、user 或 guest")
        elif acc not in known:
            bad(f"people[{i}].account", "UNKNOWN_ACCOUNT", "員工名冊與登入紀錄都沒有這個帳號")
        else:
            roles[acc] = p.role

    units: dict[str, dict[str, str]] = {}
    for u, pages in body.units.items():
        name = u.strip()
        if not name or len(name) > 100:
            bad(f"units.{u}", "INVALID_UNIT", "單位代碼不能空白、最長 100 字")
            continue
        units[name] = {}
        for p, lv in pages.items():
            if not _page_ok(p):
                bad(f"units.{u}.{p}", "INVALID_PAGE", "頁面要是以 / 開頭的路由")
            elif lv not in UNIT_LEVELS:
                bad(f"units.{u}.{p}", "INVALID_LEVEL", "單位等級只能是 view 或 edit")
            else:
                units[name][p] = lv

    overrides: dict[str, dict[str, str]] = {}
    for a, pages in body.overrides.items():
        acc = norm_account(a)
        if not acc or len(acc) > 200:
            bad(f"overrides.{a}", "INVALID_ACCOUNT", "帳號不能空白、最長 200 字")
            continue
        for p, lv in pages.items():
            if not _page_ok(p):
                bad(f"overrides.{a}.{p}", "INVALID_PAGE", "頁面要是以 / 開頭的路由")
            elif lv not in OVERRIDE_LEVELS:
                bad(f"overrides.{a}.{p}", "INVALID_LEVEL", "個人例外等級只能是 none、view 或 edit")
            else:
                overrides.setdefault(acc, {})[p] = lv

    guest: dict[str, str] = {}
    for p, lv in body.guest.items():
        if not _page_ok(p):
            bad(f"guest.{p}", "INVALID_PAGE", "頁面要是以 / 開頭的路由")
        elif lv not in UNIT_LEVELS:
            bad(f"guest.{p}", "INVALID_LEVEL", "訪客等級只能是 view 或 edit")
        else:
            guest[p] = lv

    after = {"roles": roles, "units": units, "overrides": overrides, "guest": guest}
    if details:
        return after, details

    people_after = [dict(p, role=roles.get(p["account"], p["role"])) for p in cur_b["people"]]
    if not any(p["role"] == "admin" and p.get("enabled", True) is not False for p in people_after):
        bad("people", "NO_ADMIN_LEFT", "存完後至少要有一位 admin")
    me = next((p for p in people_after if p["account"] == actor_account), None)
    if effective_level(me, PERM_PAGE, {"units": units, "overrides": overrides, "guest": guest}) != "edit":
        bad("people", "SELF_LOCKOUT", "存完後你自己會失去權限管理頁的 edit")
    return after, details


def _public(b: dict) -> dict:
    return {"version": b["version"], "enforce": b["enforce"], "people": [_person(p) for p in b["people"]],
            "units": b["units"], "overrides": b["overrides"], "guest": b["guest"]}


# ── 路由 ─────────────────────────────────────────────────

_ERR = {401: {"model": ApiResponse[None], "description": "沒有或無效的 SSO token"},
        500: {"model": ApiResponse[None]}}


def build_router(state: dict, store_factory: Callable[[dict], Any] | None = None) -> APIRouter:
    r = APIRouter(prefix="/api/permissions", tags=["permissions"])

    def store():
        if store_factory:
            return store_factory(state)
        if "perm_store" not in state:
            state["perm_store"] = Store(state["db"])
        return state["perm_store"]

    @r.post("/login", status_code=200, responses={200: {"model": ApiResponse[PersonOut]},
                                                   201: {"model": ApiResponse[PersonOut]}, **_ERR})
    def login(authorization: str | None = Header(default=None)):
        """登入成功後呼叫一次：用 token 的帳號、姓名、單位、email 建立或更新自己。第一次 201，之後 200。"""
        ident = identify(authorization)
        row, created = store().login(ident)
        logger.info("[%s] 權限登入：%s（單位 %s，%s）", current_trace_id(), ident.account, ident.unit or "—",
                    "新建" if created else "更新")
        return ok(_person(row, with_email=True), status=201 if created else 200)

    @r.get("/me", responses={200: {"model": ApiResponse[MeOut]}, **_ERR})
    def me(authorization: str | None = Header(default=None)):
        """後端從你的 token 讀到的身分，以及單位來自哪個 claim。不寫資料、不回 token。"""
        ident = identify(authorization)
        row = store().person(ident.account)
        return ok({"account": ident.account, "name": ident.name, "unit": ident.unit, "unitClaim": ident.unit_claim,
                   "email": ident.email, "role": (row or {}).get("role", "user"), "registered": row is not None,
                   "enabled": (row or {}).get("enabled", True) is not False})

    @r.get("", responses={200: {"model": ApiResponse[BundleOut]}, **_ERR})
    def get_all(authorization: str | None = Header(default=None)):
        """整包權限資料（含 version，PUT 時原樣帶回）。"""
        identify(authorization)
        return ok(_public(store().bundle()))

    @r.put("", responses={200: {"model": ApiResponse[BundleOut]},
                          403: {"model": ApiResponse[None], "description": "沒有權限管理頁的 edit"},
                          409: {"model": ApiResponse[None], "description": "version 不符"},
                          422: {"model": ApiResponse[None], "description": "值不合法、沒 admin、自己會被鎖在外面"},
                          **_ERR})
    def put_all(body: BundleIn, authorization: str | None = Header(default=None)):
        """整包寫入：people 只吃 role；enforce、units、overrides、guest 整包替換；version +1；寫異動紀錄。"""
        actor = identify(authorization)
        new_b = store().save(actor, body)
        logger.info("[%s] 權限已存檔：%s → version %d", current_trace_id(), actor.account, new_b["version"])
        return ok(_public(new_b))

    # ── 員工名冊 ─────────────────────────────────────────

    @r.get("/people", responses={200: {"model": ApiResponse[list[PersonOut]]}, **_ERR})
    def people(authorization: str | None = Header(default=None),
               q: str = Query("", max_length=100, description="帳號、姓名或工號（部分符合）"),
               unit: str = Query("", max_length=100, description="單位代碼（完全符合）"),
               enabled: bool | None = Query(None, description="true 只看在職、false 只看停用"),
               page: int = Query(1, ge=1), page_size: int = Query(20, ge=1, alias="pageSize")):
        """搜尋所有人（員工名冊＋登入過的人）。整包 GET 只回有被管到的人（非 user、登入過、有個人例外），
        要幫還沒登入的人設角色就從這裡找，PUT 時把他放進 people。"""
        identify(authorization)
        warnings = None
        if page_size > MAX_PAGE_SIZE:
            page_size = MAX_PAGE_SIZE
            warnings = [{"code": "PAGE_SIZE_LIMITED", "message": f"pageSize 上限 {MAX_PAGE_SIZE}，已自動調整"}]
        rows, total = store().people(q.strip(), unit.strip(), enabled, page, page_size)
        return ok([_person(x) for x in rows], warnings=warnings,
                  pagination={"page": page, "pageSize": page_size, "total": total, "hasNext": page * page_size < total})

    def _sync():
        ds = state.get("directory_sync")
        if ds is None:
            raise ApiError(404, "NOT_FOUND", "Directory sync is not running in this service")
        return ds

    @r.get("/directory", responses={200: {"model": ApiResponse[DirectoryStatusOut]}, **_ERR})
    def directory_status(authorization: str | None = Header(default=None)):
        """名冊同步狀態：上次成功的時間、最近一次的結果（not_configured = 還沒設定名冊 API 的憑證）。"""
        identify(authorization)
        ds = _sync()
        return ok({"lastSuccessAt": _iso(ds.synced_at()), "last": ds.last})

    @r.post("/directory/sync", responses={200: {"model": ApiResponse[DirectoryStatusOut]},
                                          403: {"model": ApiResponse[None], "description": "沒有權限管理頁的 edit"},
                                          409: {"model": ApiResponse[None], "description": "已經在同步"}, **_ERR})
    def directory_sync(authorization: str | None = Header(default=None),
                       full: bool = Query(False, description="true = 整份重抓；預設只抓上次之後有變動的")):
        """立刻同步一次（不用等每小時那次）。要有權限管理頁的 edit。AuthServer 自己每天 02:00 才從 AD/HR 更新。"""
        actor = identify(authorization)
        st = store()
        if effective_level(st.person(actor.account), PERM_PAGE, st.bundle()) != "edit":
            raise ApiError(403, "FORBIDDEN", "Edit permission on the permissions page is required")
        out = _sync().run(full=full, actor=actor.account)
        if out.get("status") == "running":
            raise ApiError(409, "CONFLICT", "Directory sync is already running")
        return ok({"lastSuccessAt": _iso(_sync().synced_at()), "last": out})

    return r
