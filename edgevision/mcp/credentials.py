"""本機憑證快取與自動續期。

問題：access token 只活 1 小時，總不能叫人每小時去改一次 MCP 設定檔。
解法：authorization_code + PKCE 登入一次拿到 refresh token（活 30 天），
      之後每次呼叫前自動檢查、快過期就換新的 —— 使用者不會察覺。

    第一次（約一個月一次）   edgevision-login → 瀏覽器登入 → 存進 ~/.edgevision/credentials.json
    之後每次呼叫（自動）     快取還有效就用；快過期就 refresh；refresh 也過期才叫人重登

這是 gcloud / aws cli / gh 都在用的模式，使用者已經熟悉這個心智模型。

⚠️ 憑證檔權限 600。裡面有 refresh token，等同 30 天的帳號存取權。
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_AUTH_SERVER = "https://auth.example.com"
# 提前多久換新的。留 5 分鐘餘裕，避免「檢查時還沒過期、送達時已過期」。
REFRESH_MARGIN_S = 300


def creds_path() -> Path:
    return Path(os.environ.get("EDGEVISION_CREDENTIALS", Path.home() / ".edgevision" / "credentials.json"))


def auth_server() -> str:
    return os.environ.get("AUTH_SERVER", DEFAULT_AUTH_SERVER).rstrip("/")


def verify_tls() -> bool:
    # 授權伺服器用自簽憑證時可關閉驗證（AUTH_VERIFY_TLS）
    return os.environ.get("AUTH_VERIFY_TLS", "false").lower() in ("1", "true", "yes")


class NotLoggedIn(RuntimeError):
    """沒有憑證或已完全過期。訊息要直接告訴使用者怎麼辦。"""


@dataclass
class Credentials:
    access_token: str
    refresh_token: str = ""
    expires_at: float = 0.0          # epoch 秒
    subject: str = ""                # CORP\alice
    display_name: str = ""
    server: str = ""

    @property
    def expired(self) -> bool:
        return time.time() >= (self.expires_at - REFRESH_MARGIN_S)

    def save(self, path: Path | None = None) -> Path:
        p = path or creds_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        # 先建立空檔並縮權限，再寫入 —— 避免明文憑證有一瞬間是 644
        p.touch(mode=0o600, exist_ok=True)
        os.chmod(p, 0o600)
        p.write_text(json.dumps(asdict(self), ensure_ascii=False, indent=2))
        return p

    @classmethod
    def load(cls, path: Path | None = None) -> "Credentials | None":
        p = path or creds_path()
        if not p.exists():
            return None
        try:
            return cls(**json.loads(p.read_text()))
        except Exception as exc:
            logger.warning("憑證檔損毀（%s）：%s", p, exc)
            return None


def discovery(server: str | None = None) -> dict:
    import httpx

    base = (server or auth_server()).rstrip("/")
    with httpx.Client(verify=verify_tls(), timeout=20.0) as c:
        r = c.get(f"{base}/.well-known/openid-configuration")
        r.raise_for_status()
        return r.json()


def _subject_of(access_token: str) -> tuple[str, str]:
    """從 token 取出身分（不驗簽，只是給人看的顯示用；真正的驗證在伺服器端）。"""
    try:
        import jwt

        c = jwt.decode(access_token, options={"verify_signature": False})
        return str(c.get("sub") or ""), str(c.get("name") or "")
    except Exception:
        return "", ""


def refresh(creds: Credentials) -> Credentials:
    """用 refresh token 換一組新的。失敗就拋 NotLoggedIn。"""
    import httpx

    if not creds.refresh_token:
        raise NotLoggedIn("憑證沒有 refresh token，請執行：python -m edgevision.mcp.login")

    base = creds.server or auth_server()
    try:
        token_ep = discovery(base)["token_endpoint"]
    except Exception as exc:
        raise NotLoggedIn(f"無法連上 AuthServer（{base}）：{exc}") from exc

    client_id = os.environ.get("AUTH_CLIENT_ID", "edgevision-cli")
    with httpx.Client(verify=verify_tls(), timeout=30.0) as c:
        r = c.post(token_ep, data={
            "grant_type": "refresh_token",
            "refresh_token": creds.refresh_token,
            "client_id": client_id,
        })
    if r.status_code != 200:
        raise NotLoggedIn(
            f"換發 token 失敗（{r.status_code}）：refresh token 可能已過期（30 天）。"
            f"請重新執行：python -m edgevision.mcp.login"
        )
    d = r.json()
    sub, name = _subject_of(d["access_token"])
    out = Credentials(
        access_token=d["access_token"],
        refresh_token=d.get("refresh_token") or creds.refresh_token,   # 有些伺服器不輪替
        expires_at=time.time() + int(d.get("expires_in", 3600)),
        subject=sub or creds.subject, display_name=name or creds.display_name, server=base,
    )
    out.save()
    logger.info("已換發新 token（%s，有效至 %s）", out.subject,
                time.strftime("%H:%M:%S", time.localtime(out.expires_at)))
    return out


def get_access_token() -> str:
    """取得可用的 access token。過期就自動換，換不了就叫人重登。

    MCP server 每次要打控制 API 前呼叫這個，使用者完全不用管 token。
    """
    creds = Credentials.load()
    if creds is None:
        raise NotLoggedIn(
            f"尚未登入。請執行：python -m edgevision.mcp.login\n"
            f"（憑證會存在 {creds_path()}，一次登入可用 30 天）"
        )
    if creds.expired:
        creds = refresh(creds)
    return creds.access_token


def status() -> dict:
    """給 edgevision-login --status 與 MCP 的 whoami 用。"""
    creds = Credentials.load()
    if creds is None:
        return {"logged_in": False, "path": str(creds_path())}
    remain = creds.expires_at - time.time()
    return {
        "logged_in": True, "subject": creds.subject, "display_name": creds.display_name,
        "server": creds.server, "path": str(creds_path()),
        "access_token_valid_for_s": max(0, int(remain)),
        "needs_refresh": creds.expired,
        "has_refresh_token": bool(creds.refresh_token),
    }
