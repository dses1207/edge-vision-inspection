"""標準 API 回應規格的共用外殼：{success, data, error, traceId}。

範圍：只套在 install() 指定路徑前綴底下的端點（目前是 /api/permissions）。
舊端點維持原本的回應格式 —— 前端與 MCP 依賴它們，之後整體改版時再把前綴加進來。

    traceId   每個請求都有：沿用 W3C traceparent 的 trace-id，否則自己產生 32 碼 hex；
              所有回應（含舊端點）都帶 header X-Trace-Id，新端點的 body traceId 與它相同。
    錯誤端點丟 ApiError(status, code, message, details)；
              參數驗證失敗 → 400 BAD_REQUEST（JSON 壞掉、型別錯）或 422 VALIDATION_FAILED；
              未預期例外 → 500 INTERNAL_ERROR，訊息不含例外原文（原文連同 traceId 進 log）。
"""

from __future__ import annotations

import contextvars
import logging
import re
import uuid
from typing import Any, Generic, TypeVar

from fastapi import FastAPI, Request
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict
from pydantic.alias_generators import to_camel

logger = logging.getLogger(__name__)

trace_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("trace_id", default="")
_TRACEPARENT = re.compile(r"^[0-9a-f]{2}-([0-9a-f]{32})-[0-9a-f]{16}-[0-9a-f]{2}$")


def current_trace_id() -> str:
    return trace_id_var.get() or uuid.uuid4().hex


def trace_id_from(traceparent: str | None) -> str:
    """traceparent 合法且 trace-id 不是全 0 就沿用，否則自己產生。"""
    m = _TRACEPARENT.match((traceparent or "").strip().lower())
    if m and m.group(1) != "0" * 32:
        return m.group(1)
    return uuid.uuid4().hex


# ── OpenAPI 用的外層結構 ─────────────────────────────────

class Camel(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)


class ErrorDetail(Camel):
    field: str
    code: str
    message: str


class ApiErrorBody(Camel):
    code: str
    message: str
    details: list[ErrorDetail] | None = None


T = TypeVar("T")


class ApiResponse(Camel, Generic[T]):
    success: bool
    data: T | None = None
    error: ApiErrorBody | None = None
    trace_id: str


# ── 回應 ─────────────────────────────────────────────────

class ApiError(Exception):
    """端點要回錯誤時丟這個；code 只用規格表或專案清單裡的值。"""

    def __init__(self, status: int, code: str, message: str, details: list[dict] | None = None) -> None:
        super().__init__(message)
        self.status, self.code, self.message, self.details = status, code, message, details


def ok(data: Any, status: int = 200, warnings: list[dict] | None = None,
       pagination: dict | None = None, headers: dict | None = None) -> JSONResponse:
    """pagination 只有列表 API 給：{page, pageSize, total?, hasNext}。"""
    tid = current_trace_id()
    body: dict[str, Any] = {"success": True, "data": data, "error": None}
    if pagination is not None:
        body["pagination"] = pagination
    if warnings:
        body["warnings"] = warnings
    body["traceId"] = tid
    return JSONResponse(body, status_code=status, headers={"X-Trace-Id": tid, **(headers or {})})


def fail(status: int, code: str, message: str, details: list[dict] | None = None) -> JSONResponse:
    tid = current_trace_id()
    err: dict[str, Any] = {"code": code, "message": message}
    if details:
        err["details"] = details
    return JSONResponse({"success": False, "data": None, "error": err, "traceId": tid},
                        status_code=status, headers={"X-Trace-Id": tid})


def _field(loc: tuple) -> str:
    """('body', 'units', 'U100', '/monitoring/live') → units.U100./monitoring/live；串列索引寫成 people[0]。"""
    out = ""
    for p in loc[1:] if loc and loc[0] in ("body", "query", "path", "header") else loc:
        out += f"[{p}]" if isinstance(p, int) else (f".{p}" if out else str(p))
    return out or "body"


_BAD_REQUEST_TYPES = ("json_invalid", "model_attributes_type")


def _is_type_error(t: str) -> bool:
    return t in _BAD_REQUEST_TYPES or t.endswith("_type") or t.endswith("_parsing")


def install(app: FastAPI, prefixes: tuple[str, ...]) -> None:
    """掛 traceId middleware 與錯誤轉換。prefixes 以外的路徑錯誤格式照 FastAPI 預設。"""

    def scoped(request: Request) -> bool:
        return request.url.path.startswith(prefixes)

    @app.middleware("http")
    async def _trace(request: Request, call_next):
        tid = trace_id_from(request.headers.get("traceparent"))
        token = trace_id_var.set(tid)
        try:
            try:
                response = await call_next(request)
            except Exception:
                if not scoped(request):
                    raise
                logger.exception("[%s] 未預期錯誤：%s %s", tid, request.method, request.url.path)
                response = fail(500, "INTERNAL_ERROR", "Unexpected server error")
            response.headers["X-Trace-Id"] = tid
            return response
        finally:
            trace_id_var.reset(token)

    @app.exception_handler(ApiError)
    async def _api_error(request: Request, exc: ApiError):
        return fail(exc.status, exc.code, exc.message, exc.details)

    @app.exception_handler(RequestValidationError)
    async def _validation(request: Request, exc: RequestValidationError):
        if not scoped(request):
            return await request_validation_exception_handler(request, exc)
        errs = exc.errors()
        if any(_is_type_error(str(e.get("type", ""))) for e in errs):
            return fail(400, "BAD_REQUEST", "Malformed request body")
        details = [{"field": _field(tuple(e.get("loc", ()))), "code": str(e.get("type", "invalid")).upper(),
                    "message": str(e.get("msg", ""))} for e in errs]
        return fail(422, "VALIDATION_FAILED", "Request validation failed", details)
