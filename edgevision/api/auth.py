"""身分驗證：接 OAuth / OIDC 授權伺服器，用 JWKS 離線驗簽。

為什麼需要：
    控制 API 的每個動作都會寫 audit_log，而 audit_log 存在的意義是「出事時知道誰做的」。
    原本的 X-Actor header 是【自我宣告】—— 填什麼就是什麼，誰都能冒名，
    等於稽核紀錄不可信。接上 OAuth 後 actor 以 token 裡驗證過的 sub 為準。

分工：
    AuthServer   負責登入與發 token（帳號來源由它對接）（我們不碰 LDAP）
    本模組抓 JWKS、驗簽、取出身分

⚠️ 身分格式：sub 的值是 `corp\\alice`（含 domain 前綴的 Windows 帳號），不是純帳號名。
   audit_log 存原值當正規識別 —— 不同網域的帳號可能撞名，
   只存切過的短名會失去唯一性。顯示給人看時才切前綴。

   ⚠️⚠️ 大小寫不保證：實測 AuthServer 發出的是【小寫】corp\\alice，
   但 AD 帳號本身大小寫不敏感，不同流程（互動登入 / refresh / 其他 client）
   可能回傳不同大小寫。所以任何拿 actor 做比對的地方
   （將來的 Supervisor 白名單、擁有者檢查…）一律要用 casefold() 比，
   不可以直接 ==。存檔仍存原值，不要在寫入時改大小寫。

設定全走環境變數（Compose 與 K8s 共用同一份程式）：
    AUTH_ENABLED           false（預設）= 沿用 X-Actor；true = 強制驗 token
    AUTH_SERVER            https://auth.example.com
    AUTH_AUDIENCE          edgevision_api
    AUTH_REQUIRE_AUDIENCE  false（預設）= token 沒帶 aud 也接受（見下方說明）
    AUTH_ALLOW_HEADER_ACTOR true = 沒有 token 時退回 X-Actor（過渡期用）
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

AUDIENCE = os.environ.get("AUTH_AUDIENCE", "edgevision_api")
# ⚠️ 是否強制 token 帶 aud。
#    授權伺服器不支援 RFC 8707 resource 參數時，走 MCP 流程發出的 token 不含 aud。而 aud 在這套架構裡是所有後端服務
#    共用的 edgevision_api，本來就不承擔「這個 token 只能給誰用」的判斷——
#    真正的授權邊界是 issuer + 簽章 + sub。所以預設不強制。
#    要收緊時設 AUTH_REQUIRE_AUDIENCE=true，並讓 AuthServer 對每個服務發不同的 aud。
REQUIRE_AUDIENCE = os.environ.get("AUTH_REQUIRE_AUDIENCE", "false").lower() in ("1", "true", "yes")
CLOCK_SKEW_S = int(os.environ.get("AUTH_CLOCK_SKEW_S", "300"))
# JWKS 快取存活時間。憑證輪替時不必重啟服務就能自動換新金鑰。
JWKS_TTL_S = int(os.environ.get("AUTH_JWKS_TTL_S", "3600"))


def enabled() -> bool:
    return os.environ.get("AUTH_ENABLED", "").lower() in ("1", "true", "yes")


def allow_header_actor() -> bool:
    """沒有 token 時是否退回 X-Actor。

    過渡期預設允許（既有的 test_train.sh、整合測試、文件都還在用 header）。
    正式收緊時設成 false，沒有 token 就一律拒絕。
    """
    return os.environ.get("AUTH_ALLOW_HEADER_ACTOR", "true").lower() in ("1", "true", "yes")


def server_base() -> str:
    return os.environ.get("AUTH_SERVER", "https://auth.example.com").rstrip("/")


@dataclass(slots=True)
class Identity:
    """驗證後的身分。unverified=True 表示來自 X-Actor，未經驗證。"""

    actor: str                      # CORP\alice（正規識別，寫進 audit_log）
    actor_type: str = "human"       # human | agent
    display_name: str = ""          # 中文姓名
    email: str = ""
    employee_id: str = ""
    organization: str = ""          # 組織代碼
    is_supervisor: bool = False
    unverified: bool = False
    claims: dict[str, Any] = field(default_factory=dict)

    @property
    def short_name(self) -> str:
        """切掉 domain 前綴，給人看用。⚠️ 不可當唯一識別（不同網域可能撞名）。"""
        return self.actor.rsplit("\\", 1)[-1]


# ── JWKS（帶 TTL 的快取）────────────────────────────────────

_jwks: Any = None
_jwks_at: float = 0.0


def _fetch_jwks():
    import httpx
    from jwt import PyJWKSet

    base = server_base()
    # ⚠️ AuthServer 用自簽憑證，這種情況可用 AUTH_VERIFY_TLS=false 關閉 TLS 驗證。
    #    正式憑證請設 AUTH_VERIFY_TLS=true。
    verify = os.environ.get("AUTH_VERIFY_TLS", "false").lower() in ("1", "true", "yes")
    with httpx.Client(verify=verify, timeout=30.0) as c:
        disc = c.get(f"{base}/.well-known/openid-configuration")
        disc.raise_for_status()
        jwks_uri = disc.json()["jwks_uri"]
        r = c.get(jwks_uri)
        r.raise_for_status()
        ks = PyJWKSet.from_json(r.text)
    logger.info("JWKS 已載入：%d 把金鑰（%s）", len(ks.keys), jwks_uri)
    return ks


def get_jwks(force: bool = False):
    """取得 JWKS，超過 TTL 自動重抓（憑證輪替時不必重啟服務）。"""
    global _jwks, _jwks_at
    if force or _jwks is None or (time.time() - _jwks_at) > JWKS_TTL_S:
        _jwks = _fetch_jwks()
        _jwks_at = time.time()
    return _jwks


def load_jwks_at_startup() -> bool:
    """啟動時預載。失敗不阻擋啟動（AUTH_ENABLED=false 時本來就不需要）。"""
    if not enabled():
        logger.info("身分驗證未啟用（AUTH_ENABLED=false），actor 沿用 X-Actor header")
        return False
    try:
        get_jwks(force=True)
        return True
    except Exception as exc:
        logger.error("JWKS 載入失敗，token 驗證將無法運作：%s", exc)
        return False


# ── 驗證 ────────────────────────────────────────────────────

def verify_token(token: str) -> Identity:
    """驗簽並取出身分。失敗一律拋 ValueError（呼叫端轉成 401）。"""
    import jwt

    try:
        ks = get_jwks()
    except Exception as exc:
        raise ValueError(f"無法取得 JWKS：{exc}") from exc

    try:
        kid = jwt.get_unverified_header(token).get("kid")
    except Exception as exc:
        raise ValueError(f"token 格式錯誤：{exc}") from exc

    key = next((k for k in ks.keys if k.key_id == kid), None)
    if key is None:
        # kid 對不上可能是剛輪替過金鑰，強制重抓一次再試
        try:
            ks = get_jwks(force=True)
            key = next((k for k in ks.keys if k.key_id == kid), None)
        except Exception:
            pass
    if key is None:
        key = ks.keys[0] if ks.keys else None
    if key is None:
        raise ValueError("JWKS 中沒有可用的簽章金鑰")

    base = server_base()
    # issuer 與簽章一律驗；aud 依設定（見 REQUIRE_AUDIENCE 的說明）。
    kw: dict[str, Any] = {
        "algorithms": ["RS256"],
        "issuer": [base, base + "/"],
        "leeway": CLOCK_SKEW_S,
        "options": {"verify_exp": True, "verify_iss": True, "verify_aud": REQUIRE_AUDIENCE},
    }
    if REQUIRE_AUDIENCE:
        kw["audience"] = AUDIENCE
    try:
        claims = jwt.decode(token, key.key, **kw)
    except Exception as exc:
        raise ValueError(f"token 驗證失敗：{exc}") from exc

    # 有帶 aud 就順手檢查值對不對（不強制有，但有就不能是別人的）
    if not REQUIRE_AUDIENCE and claims.get("aud"):
        auds = claims["aud"]
        auds = [auds] if isinstance(auds, str) else list(auds)
        if AUDIENCE not in auds:
            raise ValueError(f"token 的 aud 不符（收到 {auds}，預期含 {AUDIENCE}）")

    sub = str(claims.get("sub") or "").strip()
    if not sub:
        raise ValueError("token 沒有 sub（無法辨識身分）")

    return Identity(
        actor=sub,
        actor_type="human",
        display_name=str(claims.get("name") or ""),
        email=str(claims.get("email") or ""),
        employee_id=str(claims.get("employee_id") or ""),
        organization=str(claims.get("organization_code") or ""),
        is_supervisor=str(claims.get("is_supervisor", "")).lower() in ("1", "true", "yes"),
        claims=claims,
    )


def resolve(
    authorization: str | None, x_actor: str | None, x_actor_type: str | None, x_channel: str | None,
) -> tuple[Identity, str]:
    """把請求的身分資訊收斂成 (Identity, channel)。

    優先序：
        ① Bearer token   → 驗簽，actor = 驗證過的 sub
        ② X-Actor header → 未驗證身分，僅在 AUTH_ALLOW_HEADER_ACTOR=true 時接受
        ③ 都沒有         → AUTH_ENABLED 時拒絕；否則 anonymous
    """
    channel = (x_channel or "api").lower()
    if channel not in ("ui", "mcp", "api"):
        raise ValueError("X-Channel 必須是 ui | mcp | api")

    token = ""
    if authorization and authorization.lower().startswith("bearer "):
        token = authorization[7:].strip()

    if token:
        ident = verify_token(token)
        # channel=mcp 代表這個動作是 agent 代為執行的（人是誰仍由 token 決定）
        ident.actor_type = "agent" if channel == "mcp" else "human"
        return ident, channel

    if enabled() and not allow_header_actor():
        raise ValueError("需要 Bearer token（AUTH_ENABLED=true 且未允許 header actor）")

    actor = (x_actor or "").strip() or "anonymous"
    actor_type = (x_actor_type or ("agent" if channel == "mcp" else "human")).lower()
    return Identity(actor=actor, actor_type=actor_type, unverified=True), channel
