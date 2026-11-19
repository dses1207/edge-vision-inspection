"""控制 API 的身分關卡（第二期）：每個 /api/* 請求進來先看身分，再交給端點。

為什麼放在最外層：很多讀取端點根本沒呼叫 _actor()，只靠端點各自檢查會漏。

    有效的 SSO token（Authorization: Bearer）        → 放行（human / agent）
    有效的服務金鑰（X-Service-Token）                 → 放行（system:<服務名>）
    有效的網址簽名（?exp=&sig=，只限檔案類路由的 GET）  → 放行（<video src> 這類帶不了 header 的，見 signed_url.py）
    其他（沒帶、X-Actor 自我宣告、token 無效）         → 觀察模式：放行＋記錄；強制模式：401

模式由 auth.enforcing() 決定（AUTH_ENABLED=true 且 AUTH_ALLOW_HEADER_ACTOR=false 才擋）。
記錄按「天 × 方法 × 路由 × X-Actor × 管道 × User-Agent × 來源」彙整，記憶體累計、每 60 秒寫進
auth_unverified_calls（28-auth-unverified.sql）：切強制前看這張表，確定沒有還在用 header 的呼叫者。

    GET  /api/control/auth/unverified    查記錄（標準回應外殼；分頁）
    POST /api/control/auth/signed-urls   幫檔案類網址簽名（給前端自己組網址時用）
"""

from __future__ import annotations

import hashlib
import logging
import re
import threading
import time
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

from fastapi import APIRouter, FastAPI, Query, Request
from fastapi.responses import JSONResponse
from pydantic import Field

from edgevision.api import auth, signed_url
from edgevision.api.envelope import ApiError, ApiResponse, Camel, ok

logger = logging.getLogger(__name__)
_TZ = ZoneInfo("Asia/Taipei")

# 不需要身分的路徑：健康檢查、API 文件、output-api 的靜態頁（頁面裡呼叫的 API 照樣要身分）
_OPEN_EXACT = {"/healthz", "/openapi.json", "/docs", "/api/control/openapi.json", "/api/control/docs"}
_OPEN_PREFIX = ("/static",)
_ID = re.compile(r"/(\d+|[0-9a-f]{8,}(?:-[0-9a-f]{4,})*|[a-z]+-\d{8}-\d{6}-[0-9a-f]{6})(?=/|$)")


def needs_identity(path: str, method: str) -> bool:
    if method == "OPTIONS" or path in _OPEN_EXACT or path.startswith(_OPEN_PREFIX):
        return False
    return path.startswith("/api/")


def route_of(request: Request) -> str:
    """路由樣板（/api/tasks/{task_id}）；還沒路由到（被擋）就把 id 換成 {}，避免一個 id 一筆。"""
    r = request.scope.get("route")
    if r is not None and getattr(r, "path", None):
        return r.path
    return _ID.sub("/{}", request.url.path)


# ── token 驗證快取：同一個 token 60 秒內不重驗（端點的 _actor() 之後還會驗一次，這裡只是關卡）──

_tok_cache: dict[str, tuple[float, bool]] = {}
_tok_lock = threading.Lock()


def _token_ok(token: str) -> bool:
    key = hashlib.sha256(token.encode()).hexdigest()
    now = time.time()
    with _tok_lock:
        hit = _tok_cache.get(key)
        if hit and now - hit[0] < 60:
            return hit[1]
    try:
        auth.verify_token(token)
        good = True
    except Exception:
        good = False
    with _tok_lock:
        if len(_tok_cache) > 5000:
            _tok_cache.clear()
        _tok_cache[key] = (now, good)
    return good


# ── 記錄 ─────────────────────────────────────────────────

class Recorder:
    """記憶體彙整 → 每 60 秒 upsert。DB 寫不進去只記 log，不影響請求。"""

    def __init__(self, db, interval: float = 60.0) -> None:
        self.db, self.interval = db, interval
        self._buf: dict[tuple, list] = {}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def add(self, method: str, route: str, actor: str, channel: str, reason: str, user_agent: str, client: str) -> None:
        now = datetime.now(_TZ)
        key = (now.date(), method, route[:200], actor[:200], channel[:20], reason, user_agent[:200], client[:64])
        with self._lock:
            e = self._buf.get(key)
            if e:
                e[0] += 1
                e[2] = now
            else:
                self._buf[key] = [1, now, now]

    def flush(self) -> int:
        with self._lock:
            buf, self._buf = self._buf, {}
        if not buf:
            return 0
        try:
            with self.db.cursor() as cur:
                for (day, method, route, actor, channel, reason, ua, client), (n, first, last) in buf.items():
                    cur.execute(
                        """INSERT INTO auth_unverified_calls
                               (day, method, route, actor, channel, reason, user_agent, client, count, first_seen, last_seen)
                           VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                           ON CONFLICT (day, method, route, actor, channel, reason, user_agent, client) DO UPDATE SET
                               count = auth_unverified_calls.count + EXCLUDED.count,
                               last_seen = GREATEST(auth_unverified_calls.last_seen, EXCLUDED.last_seen)""",
                        (day, method, route, actor, channel, reason, ua, client, n, first, last))
        except Exception as exc:
            logger.warning("未驗證請求記錄寫入失敗（%d 筆丟棄）：%s", len(buf), exc)
            return 0
        return len(buf)

    def start(self) -> None:
        if self._thread:
            return

        def loop():
            while not self._stop.wait(self.interval):
                self.flush()

        self._thread = threading.Thread(target=loop, name="auth-unverified-flush", daemon=True)
        self._thread.start()

    def shutdown(self) -> None:
        self._stop.set()
        self.flush()


def install(app: FastAPI, recorder_of) -> None:
    """recorder_of() 回 Recorder（或 None = 不記錄）；延遲取得是因為 DB 在 lifespan 才建好。"""

    @app.middleware("http")
    async def _gate(request: Request, call_next):
        import anyio

        path, method = request.url.path, request.method
        if not needs_identity(path, method):
            return await call_next(request)
        h = request.headers
        authz = h.get("authorization", "")
        bearer = authz[7:].strip() if authz.lower().startswith("bearer ") else ""
        svc = h.get("x-service-token", "").strip()
        q = request.query_params
        if svc:
            status = "service" if auth.verify_service_token(svc) else "bad_service_token"
        elif bearer:
            status = "token" if await anyio.to_thread.run_sync(_token_ok, bearer) else "bad_token"
        elif "sig" in q and method in ("GET", "HEAD"):
            status = "signed" if signed_url.verify(path, q.get("exp"), q.get("sig")) else "bad_signature"
        else:
            status = "no_token"

        token = auth.service_token_var.set(svc)
        try:
            if status in ("token", "service", "signed"):
                return await call_next(request)
            if auth.enforcing():
                _record(recorder_of(), request, status, route_of(request))
                return JSONResponse({"detail": "需要有效的 SSO token、服務金鑰或網址簽名"}, status_code=401)
            response = await call_next(request)
            _record(recorder_of(), request, status, route_of(request))
            return response
        finally:
            auth.service_token_var.reset(token)


def _record(rec, request: Request, reason: str, route: str) -> None:
    if rec is None:
        return
    h = request.headers
    fwd = h.get("x-forwarded-for", "").split(",")[0].strip()
    client = fwd or (request.client.host if request.client else "")
    rec.add(request.method, route, h.get("x-actor", "") or "", (h.get("x-channel", "") or "").lower(),
            reason, h.get("user-agent", "") or "", client)


# ── 查詢 ─────────────────────────────────────────────────

class UnverifiedCallOut(Camel):
    day: str
    method: str
    route: str
    actor: str
    channel: str
    reason: str
    user_agent: str
    client: str
    count: int
    first_seen: str
    last_seen: str


class SignIn(Camel):
    paths: list[str] = Field(min_length=1, max_length=100,
                             description="要簽的路徑（不含 query），例如 /api/media/items/123/image")


class SignedUrlOut(Camel):
    path: str
    url: str
    expires_at: str


MAX_PAGE_SIZE = 100


def build_router(state: dict) -> APIRouter:
    r = APIRouter(prefix="/api/control/auth", tags=["auth"])

    @r.get("/unverified", responses={200: {"model": ApiResponse[list[UnverifiedCallOut]]}})
    def unverified(days: int = Query(7, ge=1, le=90), page: int = Query(1, ge=1),
                   page_size: int = Query(20, ge=1, alias="pageSize")):
        """沒帶有效身分的請求（按天彙整，次數多的在前）。切強制模式前用來找還在用 X-Actor 的呼叫者。
        data 是清單；目前模式在 header X-Auth-Mode（observe / enforce）。"""
        warnings = None
        if page_size > MAX_PAGE_SIZE:
            page_size = MAX_PAGE_SIZE
            warnings = [{"code": "PAGE_SIZE_LIMITED", "message": f"pageSize 上限 {MAX_PAGE_SIZE}，已自動調整"}]
        since = date.today() - timedelta(days=days - 1)
        rec = state.get("auth_recorder")
        if rec:
            rec.flush()                                   # 查之前先把記憶體裡的寫進去
        with state["db"].cursor() as cur:
            cur.execute("SELECT count(*) FROM auth_unverified_calls WHERE day >= %s", (since,))
            total = cur.fetchone()[0]
            cur.execute(
                """SELECT day, method, route, actor, channel, reason, user_agent, client, count, first_seen, last_seen
                   FROM auth_unverified_calls WHERE day >= %s
                   ORDER BY day DESC, count DESC, route LIMIT %s OFFSET %s""",
                (since, page_size, (page - 1) * page_size))
            rows = cur.fetchall()
        items = [{"day": d.isoformat(), "method": m, "route": ro, "actor": a, "channel": c, "reason": rs,
                  "userAgent": ua, "client": cl, "count": n,
                  "firstSeen": f.astimezone(_TZ).isoformat(), "lastSeen": l.astimezone(_TZ).isoformat()}
                 for d, m, ro, a, c, rs, ua, cl, n, f, l in rows]
        return ok(items, warnings=warnings,
                  pagination={"page": page, "pageSize": page_size, "total": total, "hasNext": page * page_size < total},
                  headers={"X-Auth-Mode": "enforce" if auth.enforcing() else "observe"})

    @r.post("/signed-urls", responses={200: {"model": ApiResponse[list[SignedUrlOut]]}})
    def signed_urls(body: SignIn):
        """幫 <video src> / <img src> / 下載連結這類瀏覽器直接載入的網址簽名（帶不了 Authorization header）。
        只能簽檔案類路由（錄影檔、影片庫播放與縮圖、圖片集的圖）；簽好的網址 2～4 小時有效，可以再加 ?download=1、?px=。
        清單 API 回的 streamUrl / thumbnailUrl / downloadUrl 已經簽好，不用再叫這支。"""
        bad = [{"field": f"paths[{i}]", "code": "NOT_SIGNABLE", "message": "只能簽檔案類路由"}
               for i, p in enumerate(body.paths) if not signed_url.signable(p)]
        if bad:
            raise ApiError(422, "VALIDATION_FAILED", "Request validation failed", bad)
        out = []
        for p in body.paths:
            u = signed_url.sign(p)
            out.append({"path": p, "url": u, "expiresAt": signed_url.expires_at(u)})
        return ok(out)

    return r
