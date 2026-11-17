"""身分驗證：token 優先、header 降級、開關行為（不需要真的 AuthServer）。"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from edgevision.api import auth  # noqa: E402


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for k in ("AUTH_ENABLED", "AUTH_ALLOW_HEADER_ACTOR", "AUTH_SERVER", "AUTH_AUDIENCE"):
        monkeypatch.delenv(k, raising=False)
    yield


def test_default_is_header_actor(monkeypatch):
    """預設不啟用驗證：沿用 X-Actor，但標記為未驗證。"""
    ident, ch = auth.resolve(None, "alice", None, "ui")
    assert ident.actor == "alice" and ident.unverified is True
    assert ident.actor_type == "human" and ch == "ui"


def test_mcp_channel_marks_agent():
    ident, ch = auth.resolve(None, "alice", None, "mcp")
    assert ch == "mcp" and ident.actor_type == "agent"


def test_missing_actor_is_anonymous():
    ident, _ = auth.resolve(None, None, None, "api")
    assert ident.actor == "anonymous" and ident.unverified


def test_bad_channel_rejected():
    with pytest.raises(ValueError, match="X-Channel"):
        auth.resolve(None, "alice", None, "cron")


def test_strict_mode_requires_token(monkeypatch):
    """收緊後：沒有 token 就拒絕，不再接受自我宣告的 header。"""
    monkeypatch.setenv("AUTH_ENABLED", "true")
    monkeypatch.setenv("AUTH_ALLOW_HEADER_ACTOR", "false")
    with pytest.raises(ValueError, match="Bearer token"):
        auth.resolve(None, "alice", None, "api")


def test_enabled_but_lenient_still_accepts_header(monkeypatch):
    """過渡期：驗證已啟用，但沒 token 時仍接受 header（既有腳本不會一次全壞）。"""
    monkeypatch.setenv("AUTH_ENABLED", "true")
    monkeypatch.setenv("AUTH_ALLOW_HEADER_ACTOR", "true")
    ident, _ = auth.resolve(None, "alice", None, "api")
    assert ident.actor == "alice" and ident.unverified


def test_token_beats_header(monkeypatch):
    """⭐ 有 token 時，actor 以驗證過的 sub 為準，X-Actor 說什麼都不算。"""
    monkeypatch.setattr(auth, "verify_token", lambda t: auth.Identity(
        actor="CORP\\alice", display_name="李小明", email="ops@example.com"))
    ident, ch = auth.resolve("Bearer abc.def.ghi", "someone-else", None, "ui")
    assert ident.actor == "CORP\\alice"      # 不是 someone-else
    assert ident.unverified is False
    assert ident.short_name == "alice"


def test_token_with_mcp_channel_is_agent(monkeypatch):
    """MCP 呼叫：人是誰由 token 決定，actor_type 記成 agent。"""
    monkeypatch.setattr(auth, "verify_token", lambda t: auth.Identity(actor="CORP\\alice"))
    ident, ch = auth.resolve("Bearer x", None, None, "mcp")
    assert ident.actor == "CORP\\alice" and ident.actor_type == "agent" and ch == "mcp"


def test_bad_token_raises(monkeypatch):
    def boom(t):
        raise ValueError("token 驗證失敗：Signature verification failed")
    monkeypatch.setattr(auth, "verify_token", boom)
    with pytest.raises(ValueError, match="驗證失敗"):
        auth.resolve("Bearer bad", None, None, "api")


def test_short_name_strips_domain():
    assert auth.Identity(actor="CORP\\alice").short_name == "alice"
    assert auth.Identity(actor="alice").short_name == "alice"       # 沒前綴也要正常


def test_jwks_ttl_refetch(monkeypatch):
    """JWKS 過了 TTL 要自動重抓（憑證輪替不必重啟服務）。"""
    calls = []
    monkeypatch.setattr(auth, "_fetch_jwks", lambda: (calls.append(1), "KS")[1])
    monkeypatch.setattr(auth, "_jwks", None)
    monkeypatch.setattr(auth, "_jwks_at", 0.0)
    auth.get_jwks(); auth.get_jwks()
    assert len(calls) == 1                       # TTL 內只抓一次
    monkeypatch.setattr(auth, "_jwks_at", 0.0)   # 模擬過期
    auth.get_jwks()
    assert len(calls) == 2


def test_enabled_flag(monkeypatch):
    assert auth.enabled() is False
    monkeypatch.setenv("AUTH_ENABLED", "true")
    assert auth.enabled() is True
