"""MCP server 與憑證快取：工具分級、必填把關、自動續期（不需要真的 AuthServer 或控制 API）。"""
import asyncio
import json
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from edgevision.mcp import credentials as creds_mod  # noqa: E402
from edgevision.mcp.credentials import Credentials, NotLoggedIn  # noqa: E402

pytest.importorskip("mcp")
from edgevision.mcp.server import build_server  # noqa: E402


# ═══ 憑證快取與自動續期 ═══════════════════════════════════

@pytest.fixture
def creds_file(tmp_path, monkeypatch):
    p = tmp_path / "credentials.json"
    monkeypatch.setenv("EDGEVISION_CREDENTIALS", str(p))
    return p


def test_save_is_owner_only(creds_file):
    """憑證含 refresh token（等同 30 天帳號存取權），權限必須是 600。"""
    Credentials(access_token="a", refresh_token="r", expires_at=time.time() + 3600).save()
    assert oct(creds_file.stat().st_mode)[-3:] == "600"


def test_roundtrip(creds_file):
    Credentials(access_token="a", refresh_token="r", expires_at=123.0,
                subject="CORP\\alice", display_name="小明").save()
    got = Credentials.load()
    assert got.subject == "CORP\\alice" and got.display_name == "小明"


def test_expiry_has_margin():
    """剩不到 5 分鐘就視為該換了，避免「檢查時沒過期、送達時過期」。"""
    assert Credentials(access_token="a", expires_at=time.time() + 3600).expired is False
    assert Credentials(access_token="a", expires_at=time.time() + 60).expired is True
    assert Credentials(access_token="a", expires_at=time.time() - 1).expired is True


def test_not_logged_in_message_tells_what_to_do(creds_file):
    with pytest.raises(NotLoggedIn, match="edgevision.mcp.login"):
        creds_mod.get_access_token()


def test_valid_token_used_directly(creds_file, monkeypatch):
    Credentials(access_token="good", refresh_token="r", expires_at=time.time() + 3600).save()
    monkeypatch.setattr(creds_mod, "refresh", lambda c: pytest.fail("不該換發"))
    assert creds_mod.get_access_token() == "good"


def test_expired_token_triggers_refresh(creds_file, monkeypatch):
    """⭐ 過期自動換發，使用者不用管 token。"""
    Credentials(access_token="old", refresh_token="r", expires_at=time.time() - 1).save()
    called = []

    def fake_refresh(c):
        called.append(c.refresh_token)
        return Credentials(access_token="new", refresh_token="r2", expires_at=time.time() + 3600)

    monkeypatch.setattr(creds_mod, "refresh", fake_refresh)
    assert creds_mod.get_access_token() == "new"
    assert called == ["r"]


def test_refresh_without_token_tells_user_to_login(creds_file):
    with pytest.raises(NotLoggedIn, match="login"):
        creds_mod.refresh(Credentials(access_token="a", refresh_token=""))


def test_corrupt_file_is_not_fatal(creds_file):
    creds_file.write_text("{ not json")
    assert Credentials.load() is None


def test_status_reports_login_state(creds_file):
    assert creds_mod.status()["logged_in"] is False
    Credentials(access_token="a", refresh_token="r", expires_at=time.time() + 1800,
                subject="CORP\\alice").save()
    st = creds_mod.status()
    assert st["logged_in"] and st["subject"] == "CORP\\alice"
    assert st["has_refresh_token"] and 1400 < st["access_token_valid_for_s"] <= 1800


# ═══ MCP 工具 ════════════════════════════════════════════

@pytest.fixture(scope="module")
def tools():
    return {t.name: t for t in asyncio.run(build_server().list_tools())}


def _level(t):
    a = t.annotations
    if a and a.read_only_hint:
        return "read"
    return "destructive" if (a and a.destructive_hint) else "write"


def test_tool_levels(tools):
    """三個層級都要標對 —— client 靠 annotations 決定要不要跳確認。"""
    expect = {
        "platform_health": "read", "whoami": "read", "list_models": "read",
        "list_model_versions": "read", "get_model_version": "read", "list_tasks": "read",
        "get_task": "read", "get_task_logs": "read", "list_datasets": "read", "list_configs": "read",
        "get_dataset_version": "read", "ingest_status": "read",
        "submit_training": "write", "cancel_task": "write",
        "create_dataset_upload_url": "write", "ingest_dataset": "write", "mix_datasets": "write",
        "promote_model": "destructive", "rollback_model": "destructive",
        # 圖片庫（edgevision/mcp/media_tools.py）：批次動作與審核開給 agent（第一輪粗審）
        "list_media_collections": "read", "get_media_collection": "read", "list_media_items": "read",
        "create_media_collection": "write", "create_media_upload_url": "write", "import_media": "write",
        "import_from_recording": "write", "autolabel_collection": "write", "export_media_collection": "write",
        "get_review_config": "read", "set_review_config": "write", "create_review_round": "write",
        "list_review_rounds": "read", "get_review_round": "read", "list_round_items": "read",
        "next_review_item": "write", "get_media_item": "read", "review_media_item": "write",
        "close_review_round": "write",
        "update_media_collection": "write", "list_media_batches": "read", "update_media_batch": "write", "delete_media_batch": "destructive",
        "copy_media_items": "write", "remove_media_items": "destructive", "get_media_item_history": "read",
        "list_media_exports": "read",
        "list_campaigns": "read", "get_campaign": "read", "create_campaign": "write", "update_campaign": "write",
        "run_campaign": "write", "prelabel_campaign": "write", "retry_campaign": "write",
        "close_campaign": "write",
        "list_alert_queue": "read", "review_alert": "write", "run_alert_feedback": "write",
        "list_alert_feedback": "read",
        "get_media_video": "read", "update_media_video": "write", "delete_media_video": "destructive",
        "list_recordings": "read", "get_recordings": "read", "create_recording_clip": "write",
        "upload_media_video": "write",
        "get_recording_roster": "read", "set_recording_roster": "write",
        "delete_media_collection": "destructive", "change_collection_classes": "destructive",
        "search_media_items": "read", "set_media_item_meta": "write", "download_media_collection": "write",
        "get_media_download": "read", "list_dataset_versions": "read", "diff_dataset_versions": "read",
        "download_dataset_version": "write", "delete_dataset": "destructive",
        "list_media_videos": "read", "create_video_test": "write", "list_video_tests": "read",
        "get_video_test": "read", "list_media_orgs": "read",
        "get_collection_progress": "read", "get_media_batch": "read", "next_collection_item": "write",
        "lock_media_item": "write", "save_media_draft": "write", "submit_media_annotation": "write",
        "predict_media_item": "read",
        # 控制 API 以外的服務與模型改名（edgevision/mcp/platform_tools.py）
        "rename_model_version": "destructive",
        "list_line_configs": "read", "get_line_config": "read", "list_station_specs": "read", "list_line_config_versions": "read",
        "update_line_config": "destructive", "restore_line_config": "destructive",
        "list_cameras": "read", "list_cameras_in_process": "read", "list_detections": "read",
        "review_detection": "write", "get_detection_stats": "read",
        "list_ptz_cameras": "read", "get_ptz_state": "read", "move_ptz": "write", "goto_ptz": "write",
        "save_ptz_position": "destructive", "ptz_drift": "write", "calibrate_ptz_motor": "destructive",
        "set_ptz_can_pan": "write",
    }
    assert set(tools) == set(expect)
    for name, want in expect.items():
        assert _level(tools[name]) == want, f"{name} 應該是 {want}"


def test_destructive_tools_require_reason(tools):
    """上線與回退必須說明理由 —— 稽核要查得到為什麼換模型。"""
    for name in ("promote_model", "rollback_model"):
        req = tools[name].input_schema["required"]
        assert {"model_name", "version", "reason"} <= set(req)


def test_submit_requires_notes(tools):
    """notes 必填：它會跟著血緣走，是日後回溯「為什麼這樣訓」的唯一線索。"""
    assert "notes" in tools["submit_training"].input_schema["required"]


def test_submit_has_no_auto_promote(tools):
    """⚠️ agent 不能一次做完訓練加上線 —— 上線必須是另一個明確決定。"""
    assert "auto_promote" not in tools["submit_training"].input_schema["properties"]


def test_every_tool_has_description(tools):
    """agent 靠描述挑工具，沒有描述等於沒做。"""
    for name, t in tools.items():
        assert t.description and len(t.description) > 30, f"{name} 描述太短"


def test_destructive_tools_warn_in_description(tools):
    """破壞性工具的描述要講清楚會影響產線。"""
    for name in ("promote_model", "rollback_model"):
        assert "產線" in tools[name].description


# ═══ 呼叫控制 API 的錯誤處理 ═════════════════════════════

def test_not_logged_in_surfaces_clearly(creds_file, monkeypatch):
    from edgevision.mcp import server as srv

    with pytest.raises(srv.ToolError, match="尚未登入"):
        srv._call("GET", "/api/models")


def test_401_tells_user_to_relogin(creds_file, monkeypatch):
    from edgevision.mcp import server as srv

    Credentials(access_token="a", refresh_token="r", expires_at=time.time() + 3600).save()

    class Resp:
        status_code = 401
        text = "unauthorized"
        headers = {"content-type": "application/json"}

        def json(self):
            return {"detail": "bad token"}

    class Client:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def request(self, *a, **k):
            return Resp()

    monkeypatch.setattr("httpx.Client", lambda **k: Client())
    with pytest.raises(srv.ToolError, match="login"):
        srv._call("GET", "/api/models")


def test_headers_carry_mcp_channel(creds_file, monkeypatch):
    """稽核要分得出這是 agent 代為執行的。"""
    from edgevision.mcp import server as srv

    Credentials(access_token="tok", refresh_token="r", expires_at=time.time() + 3600).save()
    h = srv._headers()
    assert h["Authorization"] == "Bearer tok" and h["X-Channel"] == "mcp"


# ═══ 錯誤要讓 agent 看得見、知道下一步 ═══════════════════

def test_errors_are_actionable_not_swallowed(creds_file):
    """⭐ MCP SDK 會把例外吞成 "Error executing tool xxx"，agent 就不知道該做什麼。
    我們用 _safe 包一層，回傳結構化的錯誤與提示。"""
    from edgevision.mcp import server as srv

    @srv._safe
    def boom():
        raise srv.ToolError("尚未登入或憑證已過期。")

    out = boom()
    assert out["ok"] is False
    assert "尚未登入" in out["error"]
    assert "edgevision.mcp.login" in out["hint"]      # 明確告訴 agent 該叫使用者做什麼


def test_unreachable_platform_hints_healthz(creds_file):
    from edgevision.mcp import server as srv

    @srv._safe
    def boom():
        raise srv.ToolError("連不上控制 API（http://x）：timeout")

    assert "healthz" in boom()["hint"]


def test_bad_argument_returns_message(creds_file):
    """參數不合法（例如 notes 沒填）也要讓 agent 看到原因。"""
    from edgevision.mcp import server as srv

    @srv._safe
    def boom():
        raise ValueError("notes 必填：請說明這次訓練的目的與假設")

    out = boom()
    assert out["ok"] is False and "notes 必填" in out["error"]


def test_ingest_requires_message(tools):
    """message 必填：資料集版本的來源與用途要跟著血緣走（同 submit 的 notes）。"""
    assert "message" in tools["ingest_dataset"].input_schema["required"]


def test_remote_server_has_same_dataset_tools():
    """remote.py（正式叢集跑的那份）工具集必須涵蓋 stdio 版 —— 兩份會漂移。"""
    from edgevision.mcp.remote import build_server as build_remote
    remote = {t.name for t in asyncio.run(build_remote().list_tools())}
    stdio = {t.name for t in asyncio.run(build_server().list_tools())}
    assert stdio <= remote | {"whoami"} and "whoami" in remote


# ── API 有的功能 MCP 都要有──────────────

# 刻意不做成工具的端點，每個都要寫理由
_MCP_EXEMPT = {
    ("GET", "/"): "前端頁面",
    ("GET", "/api/image"): "二進位圖檔；偵測紀錄裡有網址",
    ("GET", "/api/frontend-config"): "前端啟動設定",
    ("POST", "/api/config/lines/{}/status"): "runner 回報狀態用，不是給人操作的",
    ("GET", "/api/training/jobs"): "同 list_tasks(type=training)",
    ("POST", "/api/datasets/ingest"): "multipart 上傳；MCP 走 create_dataset_upload_url + ingest_dataset",
    ("POST", "/api/media/collections/{}/images"): "multipart 上傳；MCP 走 create_media_upload_url + import_media",
    ("GET", "/api/media/items/{}/image"): "二進位圖檔；get_media_item 有臨時網址",
    ("GET", "/api/media/recordings/{}/files/{}"): "二進位影片；get_recordings 有下載網址",
    ("GET", "/api/media/videos/{}/stream"): "二進位影片串流",
    ("GET", "/api/media/videos/{}/thumbnail"): "二進位縮圖；list_media_videos 有 thumbnail_url",
    ("GET", "/api/ptz/{}/snapshot"): "二進位 JPEG",
}


def _api_endpoints() -> set[tuple[str, str]]:
    import re
    from unittest import mock

    from edgevision.api import control, server
    from edgevision.ptz import api as ptz
    out = set()
    for app in (control.app, server.app, ptz.app):
        for path, ops in app.openapi()["paths"].items():
            if path.endswith("/healthz"):
                continue
            out |= {(m.upper(), re.sub(r"\{[^}]*\}", "{}", path)) for m in ops}
    return out


def _mcp_endpoints() -> set[tuple[str, str]]:
    import re
    from pathlib import Path
    src = "".join((Path(__file__).parents[1] / "edgevision/mcp" / f).read_text()
                  for f in ("server.py", "remote.py", "media_tools.py", "platform_tools.py"))
    return {(m, re.sub(r"\{[^}]*\}", "{}", p)) for m, p in re.findall(r'call\(\s*"([A-Z]+)",\s*f?"([^"]+)"', src)}


def test_every_api_endpoint_has_mcp_tool():
    """新增 API 端點卻沒補 MCP 工具（或沒寫豁免理由）就會失敗。"""
    missing = _api_endpoints() - _mcp_endpoints() - set(_MCP_EXEMPT)
    assert not missing, f"這些 API 端點沒有 MCP 工具：{sorted(missing)}"


def test_mcp_exemptions_still_exist():
    """豁免清單不能留著已經刪掉的端點。"""
    assert set(_MCP_EXEMPT) <= _api_endpoints()


def test_service_url_routes_by_prefix(monkeypatch):
    from edgevision.mcp.platform_tools import service_url
    monkeypatch.setenv("PTZ_URL", "http://ptz:8097")
    assert service_url("/api/ptz/cam50/status", "http://c") == "http://ptz:8097"
    assert service_url("/api/media/collections", "http://c") == "http://c"


def test_output_api_writes_use_verified_actor(tools):
    """output-api 不驗 token，操作者寫在 body —— 工具不能讓 agent 自己填。"""
    for name in ("update_line_config", "restore_line_config", "review_detection"):
        assert "actor" not in tools[name].input_schema["properties"]


# ── protected resource metadata 補 scopes_supported（要 offline_access 才拿得到 refresh token）──

def _asgi_call(app, path):
    import asyncio

    sent = []

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(msg):
        sent.append(msg)

    asyncio.run(app({"type": "http", "path": path, "method": "GET", "headers": []}, receive, send))
    start = next(m for m in sent if m["type"] == "http.response.start")
    body = b"".join(m.get("body", b"") for m in sent if m["type"] == "http.response.body")
    return start, body


def _json_app(doc, status=200):
    import json as _json

    async def app(scope, receive, send):
        body = _json.dumps(doc).encode()
        await send({"type": "http.response.start", "status": status,
                    "headers": [(b"content-type", b"application/json"), (b"content-length", str(len(body)).encode())]})
        await send({"type": "http.response.body", "body": body[:10], "more_body": True})
        await send({"type": "http.response.body", "body": body[10:]})
    return app


def test_prm_gets_scopes_supported():
    import json as _json

    from edgevision.mcp.remote import PrmScopes
    prm = {"resource": "https://edgevision.example.com/", "authorization_servers": ["https://auth/"]}
    start, body = _asgi_call(PrmScopes(_json_app(prm), ["openid", "offline_access"]),
                             "/.well-known/oauth-protected-resource")
    doc = _json.loads(body)
    assert doc["scopes_supported"] == ["openid", "offline_access"] and doc["resource"] == prm["resource"]
    assert dict(start["headers"])[b"content-length"] == str(len(body)).encode()


def test_prm_other_paths_untouched():
    from edgevision.mcp.remote import PrmScopes
    _, body = _asgi_call(PrmScopes(_json_app({"x": 1}), ["openid"]), "/mcp")
    assert body == b'{"x": 1}'
