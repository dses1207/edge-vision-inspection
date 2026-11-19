"""遠端 MCP server（HTTP）：使用者只要一行 URL，不用 clone、不用裝、不用跑腳本。

跟 stdio 模式（server.py）的差別只在「憑證從哪來」：

    stdio （本機）   憑證在 ~/.edgevision/credentials.json，自己 refresh
    http  （遠端）   Claude Code 內建 OAuth：收到 401 → 讀 metadata → 開瀏覽器登入
                     → token 存系統 keychain → 自動 refresh。使用者只打一次 /mcp

使用者的設定就這樣，沒有 token、沒有密碼、沒有路徑：

    { "mcpServers": { "edgevision": { "type": "http", "url": "http://<host>/mcp" } } }

⚠️ 身分怎麼傳下去：
    Claude Code 帶的 Bearer token 由 SDK 的 RequireAuthMiddleware 在傳輸層驗完，
    我們從 auth context 取出驗證過的 sub，原樣轉給控制 API。
    所以 audit_log 記的仍是真實身分，跟 stdio 模式一致。

⚠️ 關於 RFC 8707（resource 參數）：
    規範要求 authorization server 支援，但不是每個授權伺服器都支援。
    我們採寬鬆做法 —— 自己在 TokenVerifier 檢查 aud，不強制 AuthServer 支援。
    代價：所有後端服務共用同一個 edgevision_api，別的服務的 token 我們也會收。
    需要時要收緊時在前面加一層 OAuth proxy，本檔不用改。
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any

logger = logging.getLogger(__name__)

CONTROL_URL = os.environ.get("CONTROL_URL", "http://control-api:8090").rstrip("/")
# 這個 MCP server 對外的位址。K8s 上換成 Ingress host，這裡只改環境變數。
PUBLIC_URL = os.environ.get("MCP_PUBLIC_URL", "http://localhost:8096").rstrip("/")
AUTH_SERVER = os.environ.get("AUTH_SERVER", "https://auth.example.com").rstrip("/")
PORT = int(os.environ.get("MCP_PORT", "8096"))
# 公告在 protected resource metadata 的 scopes_supported：Claude Code 登入時照這份要 scope。
# 要有 offline_access，授權伺服器才發 refresh token —— 沒有它 access token 24 小時一到就得重新登入
# （metadata 沒寫 scopes_supported 的話永遠拿不到 refresh token）。
OAUTH_SCOPES = os.environ.get("MCP_OAUTH_SCOPES", "openid offline_access").split()
PRM_PATH = "/.well-known/oauth-protected-resource"
TIMEOUT = float(os.environ.get("MCP_HTTP_TIMEOUT", "60"))


# ── Token 驗證（重用控制 API 那套 JWKS 驗簽）──────────────

def build_verifier():
    from mcp.server.auth.provider import AccessToken, TokenVerifier

    class OidcTokenVerifier(TokenVerifier):
        """驗授權伺服器發的 JWT。驗簽邏輯與控制 API 共用（edgevision/api/auth.py）。"""

        async def verify_token(self, token: str) -> AccessToken | None:
            import anyio

            from edgevision.api import auth as api_auth

            def _verify():
                try:
                    ident = api_auth.verify_token(token)
                    logger.info("token 驗證通過：%s（%s）", ident.actor, ident.display_name or "—")
                    return ident
                except Exception as exc:
                    # 驗證失敗時記下 token 的結構（不含簽章、不含敏感值），
                    # 否則只看到「missing aud」不知道 AuthServer 到底發了什麼。
                    try:
                        import jwt as _jwt

                        c = _jwt.decode(token, options={"verify_signature": False})
                        shape = {k: c.get(k) for k in ("iss", "aud", "sub", "client_id", "azp", "scope")}
                        logger.info("token 驗證失敗：%s｜token 結構：%s", exc, shape)
                    except Exception:
                        logger.info("token 驗證失敗：%s", exc)
                    return None

            ident = await anyio.to_thread.run_sync(_verify)
            if ident is None:
                return None
            claims = ident.claims or {}
            return AccessToken(
                token=token,
                client_id=str(claims.get("client_id") or claims.get("azp") or "unknown"),
                scopes=str(claims.get("scope", "")).split(),
                expires_at=int(claims["exp"]) if "exp" in claims else None,
                resource=str(claims.get("aud") or ""),
                subject=ident.actor,          # CORP\alice —— 寫進 audit_log 的正規識別
                claims=claims,
            )

    return OidcTokenVerifier()


def current_token() -> str:
    """這次請求已驗證的原始 token：轉給控制 API，讓只認 token 的端點（/api/permissions）也能用。"""
    try:
        from mcp.server.auth.middleware.auth_context import get_access_token

        tok = get_access_token()
        return str(tok.token) if tok and tok.token else ""
    except Exception:
        return ""


def current_subject() -> str:
    """取出這次請求已驗證的身分。SDK 把它放在 contextvar 裡。"""
    try:
        from mcp.server.auth.middleware.auth_context import get_access_token

        tok = get_access_token()
        if tok and tok.subject:
            return str(tok.subject)
    except Exception as exc:
        logger.debug("取不到 auth context：%s", exc)
    return ""


# ── 呼叫控制 API（帶下游身分）──────────────────────────────

class ToolError(Exception):
    """可直接呈現給 agent 的錯誤。"""


def _err(message: str, hint: str = "") -> dict:
    out = {"ok": False, "error": message}
    if hint:
        out["hint"] = hint
    return out


def _call(method: str, path: str, *, json_body: dict | None = None, params: dict | None = None) -> Any:
    """打控制 API。身分用「已驗證的 sub」轉下去，控制 API 才能記對 audit_log。

    ⚠️ 這裡是服務對服務的呼叫，不是使用者直接打控制 API。
       控制 API 信任這個 header 的前提是：MCP server 與控制 API 在同一個信任邊界內
       （同一台機器 / 同一個 K8s namespace，不對外曝露控制 API）。
    """
    import httpx

    subject = current_subject()
    if not subject:
        raise ToolError("無法取得身分（token 未驗證）。請在 Claude Code 執行 /mcp 重新登入。")

    headers = {
        "X-Actor": subject,          # 驗證過的身分（CORP\alice）
        "X-Actor-Type": "agent",     # 這個動作是 agent 代為執行
        "X-Channel": "mcp",
        "Content-Type": "application/json",
    }
    tok = current_token()
    if tok:
        headers["Authorization"] = f"Bearer {tok}"   # 有 token 時控制 API 以 token 的 sub 為準（與 X-Actor 同一人）
    from edgevision.mcp.platform_tools import service_url

    base = service_url(path, CONTROL_URL)
    try:
        with httpx.Client(timeout=TIMEOUT) as c:
            r = c.request(method, f"{base}{path}", headers=headers, json=json_body, params=params)
    except Exception as exc:
        raise ToolError(f"連不上控制 API（{base}）：{exc}") from exc

    if r.status_code >= 400:
        detail = r.text[:500]
        try:
            j = r.json()
            detail = j.get("detail") or _envelope_error(j) or detail   # 舊端點 detail；新端點標準回應外殼 error
        except Exception:
            pass
        raise ToolError(f"控制 API 回 {r.status_code}：{detail}")

    ct = r.headers.get("content-type", "")
    return r.json() if ct.startswith("application/json") else r.text


def _envelope_error(j: dict) -> str:
    """標準回應外殼的錯誤：message ＋ 驗證明細（agent 才知道哪個欄位錯）。"""
    e = j.get("error") if isinstance(j, dict) else None
    if not isinstance(e, dict):
        return ""
    msg = f"{e.get('code', '')}: {e.get('message', '')}"
    if e.get("details"):
        msg += " " + "; ".join(f"{d.get('field')} {d.get('code')} {d.get('message')}" for d in e["details"])
    return msg


def _safe(fn):
    """錯誤轉成結構化回傳，不要被 SDK 吞成 'Error executing tool'。"""
    import functools

    @functools.wraps(fn)
    def wrapper(*a, **kw):
        try:
            return fn(*a, **kw)
        except ToolError as exc:
            m = str(exc)
            hint = ""
            if "身分" in m:
                hint = "在 Claude Code 執行 /mcp，選 edgevision → Authenticate"
            elif "連不上控制 API" in m:
                hint = "平台可能沒在跑，請聯絡維護者"
            return _err(m, hint)
        except ValueError as exc:
            return _err(str(exc))

    return wrapper


class PrmScopes:
    """在 SDK 產生的 protected resource metadata 補上 scopes_supported（只改這一個回應）。

    ⚠️ 不用 AuthSettings.required_scopes：SDK 拿它同時當 scopes_supported 與【強制檢查】，
       token 少一個 scope 就 403 —— 現有 token 都沒帶 scope，一開全部擋掉。
       這裡只公告、不強制：客戶端照公告要 offline_access 換得 refresh token，驗證行為不變。
    """

    def __init__(self, inner, scopes: list[str]):
        self.inner = inner
        self.scopes = scopes

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or not self.scopes or not scope.get("path", "").startswith(PRM_PATH):
            return await self.inner(scope, receive, send)
        start: dict = {}
        chunks: list[bytes] = []

        async def _send(msg):
            if msg["type"] == "http.response.start":
                start.update(msg)
                return
            if msg["type"] == "http.response.body":
                chunks.append(msg.get("body", b""))
                if msg.get("more_body"):
                    return
                body = b"".join(chunks)
                if start.get("status") == 200:
                    try:
                        doc = json.loads(body)
                        doc.setdefault("scopes_supported", self.scopes)
                        body = json.dumps(doc).encode()
                    except ValueError:
                        pass
                headers = [(k, v) for k, v in start.get("headers", []) if k.lower() != b"content-length"]
                headers.append((b"content-length", str(len(body)).encode()))
                await send({**start, "headers": headers})
                await send({"type": "http.response.body", "body": body})
                return
            await send(msg)

        return await self.inner(scope, receive, _send)


def build_server():
    from mcp.server.auth.settings import AuthSettings
    from mcp.server.mcpserver import MCPServer
    from mcp.server.transport_security import TransportSecuritySettings
    from mcp.types import ToolAnnotations
    from pydantic import AnyHttpUrl
    from urllib.parse import urlparse

    auth = AuthSettings(
        issuer_url=AnyHttpUrl(AUTH_SERVER),
        resource_server_url=AnyHttpUrl(PUBLIC_URL),
        # ⚠️ 不強制 resource 檢查：授權伺服器的 aud 若是共用的 edgevision_api，
        #    開了會把所有 token 擋掉。我們在 TokenVerifier 自己驗 aud。
        validate_token_resource=False,
        required_scopes=None,
    )

    # ⚠️ SDK 內建 DNS rebinding 防護：Host header 不在 allowed_hosts 就回 421
    #    Misdirected Request（`transport_security.py`）。預設只認 localhost，
    #    所以任何從別台機器連進來的請求都會被擋 —— 即使 token 驗證已經通過。
    #    實測踩過（2026-09-11）：token 驗證通過但 Claude Code 顯示
    #    「reconnecting failed」，真因就是這個 421，而 SDK 不會把理由回給客戶端。
    #    把對外位址（MCP_PUBLIC_URL）加進白名單；上 K8s 換 Ingress host 時
    #    只要改那個環境變數，這裡自動跟著走。
    _pub = urlparse(PUBLIC_URL)
    _host = _pub.netloc or _pub.path
    _allowed_hosts = [_host, f"{_pub.hostname}:*"] if _pub.hostname else [_host]
    _allowed_origins = [f"{_pub.scheme}://{_host}", f"{_pub.scheme}://{_pub.hostname}:*"]
    # 額外允許的 host（反向代理、K8s Ingress 的別名…），逗號分隔
    for extra in os.environ.get("MCP_ALLOWED_HOSTS", "").split(","):
        extra = extra.strip()
        if extra:
            _allowed_hosts.append(extra)
            _allowed_origins.extend([f"http://{extra}", f"https://{extra}"])

    security = TransportSecuritySettings(
        enable_dns_rebinding_protection=True,
        allowed_hosts=_allowed_hosts,
        allowed_origins=_allowed_origins,
    )
    logger.info("允許的 Host：%s", _allowed_hosts)

    mcp = MCPServer("edgevision", auth=auth, token_verifier=build_verifier())
    # ⚠️ transport_security 不是建構子參數，是 streamable_http_app() 的參數，
    #    所以掛在 server 物件上，由 main() 建 app 時取用。
    mcp._edgevision_security = security

    RO = ToolAnnotations(read_only_hint=True, idempotent_hint=True)
    WRITE = ToolAnnotations(read_only_hint=False, destructive_hint=False, idempotent_hint=False)
    DESTRUCTIVE = ToolAnnotations(read_only_hint=False, destructive_hint=True, idempotent_hint=False)

    # ═══ 🟢 唯讀 ═══════════════════════════════════════════

    @mcp.tool(title="平台狀態", annotations=RO)
    @_safe
    def platform_health() -> dict:
        """控制 API、資料庫、物件儲存、GPU 執行器是否正常，以及有幾個任務在排隊。

        遇到任何「連不上」「卡住」的狀況時先呼叫這個。
        """
        return _call("GET", "/healthz")

    @mcp.tool(title="我是誰", annotations=RO)
    @_safe
    def whoami() -> dict:
        """目前用哪個身分操作平台。所有動作都會以這個身分寫進稽核紀錄。"""
        sub = current_subject()
        return {"subject": sub, "short_name": sub.rsplit("\\", 1)[-1],
                "channel": "mcp", "platform": CONTROL_URL}

    @mcp.tool(title="列出模型", annotations=RO)
    @_safe
    def list_models() -> list[dict]:
        """Registry 裡有哪些模型、各自現役（production）是哪一版、有哪些版本。"""
        return _call("GET", "/api/models")

    @mcp.tool(title="列出模型版本", annotations=RO)
    @_safe
    def list_model_versions(model_name: str) -> list[dict]:
        """某模型的所有版本，含 tag（production / candidate / rejected）與指標。

        Args:
            model_name: 模型名稱，例如 seg_alpha、wildlife_det
        """
        return _call("GET", f"/api/models/{model_name}/versions")

    @mcp.tool(title="模型版本詳情", annotations=RO)
    @_safe
    def get_model_version(model_name: str, version: int) -> dict:
        """單一版本的完整身分：血緣（哪個資料集版本、哪份程式碼、從哪個模型訓的）、
        指標、artifacts 位置。診斷「這個模型怎麼來的」時用這個。
        """
        return _call("GET", f"/api/models/{model_name}/versions/{version}")

    @mcp.tool(title="列出任務", annotations=RO)
    @_safe
    def list_tasks(status: str | None = None, type: str | None = None, limit: int = 20) -> list[dict]:
        """訓練與部署任務清單。

        Args:
            status: queued / working / completed / failed / cancelled
            type: training / deployment
            limit: 最多回幾筆
        """
        params: dict[str, Any] = {"limit": limit}
        if status:
            params["status"] = status
        if type:
            params["type"] = type
        return _call("GET", "/api/tasks", params=params)

    @mcp.tool(title="任務詳情", annotations=RO)
    @_safe
    def get_task(task_id: str) -> dict:
        """單一任務的完整狀態：進度、Gate 結果、指標、一致性驗證、錯誤。

        訓練跑完後要判斷「結果好不好」就看這個。
        """
        return _call("GET", f"/api/tasks/{task_id}")

    @mcp.tool(title="任務 log", annotations=RO)
    @_safe
    def get_task_logs(task_id: str, step: str = "trainer", tail: int = 100) -> str:
        """某個步驟的容器 log。任務失敗時用來找真正的原因。

        Args:
            task_id: 任務 ID
            step: trainer / trtexec / finalize / parity-triton
            tail: 取最後幾行
        """
        return _call("GET", f"/api/tasks/{task_id}/logs", params={"step": step, "tail": tail})

    @mcp.tool(title="列出資料集", annotations=RO)
    @_safe
    def list_datasets() -> list[dict]:
        """LakeFS 裡有哪些資料集 repo、各自有哪些版本（版本 ID 不可變）。

        送訓練時要用這裡的 dataset_ref。
        """
        return _call("GET", "/api/datasets")

    @mcp.tool(title="列出訓練設定", annotations=RO)
    @_safe
    def list_configs() -> list[dict]:
        """平台自帶的範例訓練設定。

        ⚠️ 正式設定應該來自版本控管的訓練設定，這裡只是示範用。
        """
        return _call("GET", "/api/configs")

    @mcp.tool(title="資料集版本詳情", annotations=RO)
    @_safe
    def get_dataset_version(repo: str, version_id: str) -> dict:
        """單一資料集版本的 metadata（n_images、n_train/n_val、classes）。

        送訓練前先看這個確認資料集內容對不對。

        Args:
            repo: 資料集 repo 名稱
            version_id: 版本 ID（list_datasets 回的 id 或完整 commit id）
        """
        return _call("GET", f"/api/datasets/{repo}/versions/{version_id}")

    @mcp.tool(title="資料集匯入進度", annotations=RO)
    @_safe
    def ingest_status(job_id: str) -> dict:
        """查 ingest 工作進度，輪詢到 status 是 done / failed。

        done 之後的 dataset_ref 可直接拿去 submit_training。
        ⚠️ 工作狀態存在記憶體：control-api 重啟就查不到，但 LakeFS 已
        commit 的版本不會消失，可用 list_datasets 確認版本建成與否。
        """
        return _call("GET", f"/api/datasets/ingest/{job_id}")

    # ═══ 🟡 寫入 ═══════════════════════════════════════════

    @mcp.tool(title="取得資料集上傳網址", annotations=WRITE)
    @_safe
    def create_dataset_upload_url(filename: str) -> dict:
        """拿臨時上傳網址（2 小時有效），把資料集 zip 直傳物件儲存。

        zip 幾百 MB 起跳，塞不進工具參數 —— 流程是：
        ① 本工具拿 upload_url 與 source_uri
        ② 使用者執行 `curl -T <zip> '<upload_url>'`（網址要加引號）
        ③ 用 source_uri 呼叫 ingest_dataset

        Args:
            filename: zip 檔名（只取 basename，會自動加時間戳避免覆蓋）
        """
        return _call("POST", "/api/datasets/upload-url", params={"filename": filename})

    @mcp.tool(title="匯入資料集", annotations=WRITE)
    @_safe
    def ingest_dataset(source_uri: str, repo: str, message: str,
                       all_val: bool = False, classes: list[str] | None = None) -> dict:
        """把已上傳的 YOLO 格式 zip 匯入 LakeFS，建立不可變的資料集版本。

        伺服器端直接從物件儲存讀 zip（先用 create_dataset_upload_url 上傳）。
        回傳 job_id，用 ingest_status 輪詢；done 之後拿 dataset_ref 送訓練。

        Args:
            source_uri: s3://bucket/key（create_dataset_upload_url 回的 source_uri）
            repo: 資料集 repo 名稱（小寫英數與 -，3–63 字，不能有底線、大寫；不存在會自動建立）
            message: 這個版本的說明（像 commit message，必填寫清楚來源與用途）
            all_val: true = 整份當驗證集 —— 黃金測試集要設 true
            classes: 類別名稱順序；省略則讀 zip 內的 data.yaml
        """
        if not message.strip():
            raise ValueError("message 必填：說明這份資料的來源與用途（會跟著版本血緣走）")
        body: dict[str, Any] = {"source_uri": source_uri, "repo": repo,
                                "message": message, "all_val": all_val}
        if classes:
            body["classes"] = classes
        return _call("POST", "/api/datasets/ingest-uri", json_body=body)

    @mcp.tool(title="混合資料集", annotations=WRITE)
    @_safe
    def mix_datasets(repo: str, message: str, sources: list[dict], total: int | None = None, seed: int = 0) -> dict:
        """幾個 LakeFS 快照依配方混成一個新快照（訓練快照），送訓練時 dataset_ref 填它。背景任務，回 task_id；
        完成後 get_task 的 result.mix.dataset_ref 就是訓練快照，get_dataset_version 的 metadata.mix 查得到配方。

        用途：各批次要不同比例（例：舊批次少一點、新批次多一點），或微調時舊資料要帶一些、避免遺忘。
        只作用在 train；各來源的 val 全部合起來。同一張圖在多個來源只算一次（取最大次數）。各來源類別必須一致。

        Args:
            repo: 輸出的 LakeFS repo（小寫英數與 -，3–63 字，不能有底線、大寫）
            message: 版本說明（用途、為什麼這樣配；會跟著血緣走）
            sources: [{"ref": "lakefs://repo/版本id", 以下三選一}]
                repeat: 倍數（0.5 = 抽一半、2 = 每張兩次）；choose: 固定張數；proportion: 相對比例（要給 total）
                都不給 = repeat 1（全部取一次）。比例不能跟倍數 / 張數混用
            total: 用 proportion 時的總張數
            seed: 抽樣亂數種子（結果寫死在快照裡）
        """
        if not message.strip():
            raise ValueError("message 必填：說明這份配方的用途")
        body: dict[str, Any] = {"repo": repo, "message": message, "sources": sources, "seed": seed}
        if total is not None:
            body["total"] = total
        return _call("POST", "/api/datasets/mix", json_body=body)

    @mcp.tool(title="建立訓練任務", annotations=WRITE)
    @_safe
    def submit_training(
        model_name: str, dataset_ref: str, notes: str,
        golden_ref: str | None = None, config: dict | None = None, gates: dict | None = None,
        overrides: dict | None = None, parent_task_id: str | None = None,
        clone_reason: str | None = None, config_source: str = "mcp",
    ) -> dict:
        """送出訓練任務。會佔用 GPU 數十分鐘，同時只能跑一個（其餘排隊）。

        ⚠️ notes 必填：寫清楚這次為什麼訓練、想驗證什麼假設。
        它會跟著模型血緣走，是日後回溯的唯一線索。

        ⚠️ 不提供 auto_promote —— 訓練完是 candidate，要上線得另外呼叫 promote_model，
        讓人有機會看過結果再決定。

        Args:
            model_name: 模型名稱
            dataset_ref: 資料集版本 ID，形如 lakefs://alpha/<版本id>
            notes: 這次訓練的目的與假設（必填）
            golden_ref: 黃金測試集版本 ID。省略則用 val 當 Gate 依據（分數偏樂觀）
            config: 完整訓練設定。省略則用平台範例
            gates: 完整 Gate 規則
            overrides: 臨時覆寫，例如 {"train": {"epochs": 2}}
            parent_task_id: 從哪個任務衍生，血緣會接上
            clone_reason: 為什麼從那個任務衍生
            config_source: 設定來源，供血緣追溯
        """
        if not notes.strip():
            raise ValueError("notes 必填：請說明這次訓練的目的與假設")
        body: dict[str, Any] = {
            "model_name": model_name, "dataset_ref": dataset_ref, "notes": notes,
            "config_source": config_source, "auto_promote": False,
        }
        for k, v in (("golden_ref", golden_ref), ("config", config), ("gates", gates),
                     ("overrides", overrides), ("parent_task_id", parent_task_id),
                     ("clone_reason", clone_reason)):
            if v:
                body[k] = v
        return _call("POST", "/api/training/jobs", json_body=body)

    @mcp.tool(title="取消任務", annotations=WRITE)
    @_safe
    def cancel_task(task_id: str) -> dict:
        """取消進行中的任務。已經跑掉的訓練成果會遺失。"""
        return _call("POST", f"/api/tasks/{task_id}/cancel")

    # ═══ 🔴 破壞性 ═════════════════════════════════════════

    @mcp.tool(title="⚠️ 模型上線", annotations=DESTRUCTIVE)
    @_safe
    def promote_model(model_name: str, version: int, reason: str) -> dict:
        """把指定版本設為 production 並部署到 Triton —— **會立刻換掉產線正在用的模型**。

        ⚠️ 呼叫前務必：① 確認通過評估 Gate ② 逐類比對新舊 recall 沒退步
        ③ 讓使用者明確同意 —— 不要自行判斷後直接執行。

        Args:
            model_name: 模型名稱
            version: 要上線的版本號
            reason: 為什麼上線這一版（寫進稽核紀錄）
        """
        if not reason.strip():
            raise ValueError("reason 必填：上線會影響產線，必須說明理由")
        return _call("POST", f"/api/models/{model_name}/promote",
                     json_body={"version": version, "reason": reason})

    @mcp.tool(title="⚠️ 模型回退", annotations=DESTRUCTIVE)
    @_safe
    def rollback_model(model_name: str, version: int, reason: str) -> dict:
        """回退到舊版本並部署到 Triton —— **會立刻換掉產線正在用的模型**。

        回退往已知可用的舊版走，風險較低，新版出問題時應果斷回退。
        但它同樣改變線上服務，仍需使用者同意。

        Args:
            model_name: 模型名稱
            version: 要回退到的版本號
            reason: 為什麼回退（寫進稽核紀錄）
        """
        if not reason.strip():
            raise ValueError("reason 必填：請說明為什麼要回退")
        return _call("POST", f"/api/models/{model_name}/rollback",
                     json_body={"version": version, "reason": reason})

    # 圖片庫（兩個 server 共用一份定義）
    from edgevision.mcp.media_tools import register as _register_media
    _register_media(mcp, _call, _safe, RO, WRITE, DESTRUCTIVE)

    # 控制 API 以外的服務（線路設定、偵測、報表、PTZ）與模型改名
    from edgevision.mcp.platform_tools import register as _register_platform
    _me = current_subject
    _register_platform(mcp, _call, _safe, RO, WRITE, DESTRUCTIVE, _me)

    return mcp


def main() -> None:
    import sys

    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)-7s %(name)s %(message)s", stream=sys.stderr,
    )
    logger.info("遠端 MCP server 啟動：對外 %s，控制 API %s，AuthServer %s",
                PUBLIC_URL, CONTROL_URL, AUTH_SERVER)

    mcp = build_server()

    # ⚠️ mcp.run() 內部自己起 uvicorn，access log 走 uvicorn 預設設定，
    #    實測外部請求【不會】出現在容器 log 裡（只看得到 healthcheck），
    #    除錯時完全看不出「客戶端到底有沒有連進來」。
    #    所以自己包一層，記下來源、路徑、有沒有帶 token —— 這是除錯的第一個問題。
    app = mcp.streamable_http_app(
        transport_security=getattr(mcp, "_edgevision_security", None),
        host="0.0.0.0",
    )

    class _AccessLog:
        def __init__(self, inner):
            self.inner = inner

        async def __call__(self, scope, receive, send):
            if scope["type"] != "http":
                return await self.inner(scope, receive, send)
            headers = {k.decode().lower(): v.decode() for k, v in scope.get("headers", [])}
            path = scope.get("path", "")
            # healthcheck 每 20 秒一次，記了會把真正的請求淹掉
            if path != "/.well-known/oauth-protected-resource":
                client = scope.get("client") or ("?", 0)
                auth_hdr = headers.get("authorization", "")
                bearer = "有 token" if auth_hdr.lower().startswith("bearer ") else "無 token"
                logger.info("→ %s %s 來自 %s:%s｜%s｜UA=%s",
                            scope.get("method", "?"), path, client[0], client[1],
                            bearer, headers.get("user-agent", "—")[:60])
                sys.stderr.flush()

            status_seen = {}

            async def _send(msg):
                if msg["type"] == "http.response.start":
                    status_seen["code"] = msg["status"]
                    if path != "/.well-known/oauth-protected-resource":
                        logger.info("← %s %s", msg["status"], path)
                        sys.stderr.flush()
                await send(msg)

            return await self.inner(scope, receive, _send)

    import uvicorn

    # ⚠️ uvicorn.run() 會【重設】logging 設定，上面 basicConfig 掛的 handler 會失效，
    #    middleware 裡的 logger.info 因此整個消失（實測踩過：以為 middleware 沒生效，
    #    其實是有跑、但 log 被吃掉）。把 uvicorn 的 log config 關掉，沿用我們自己的。
    uvicorn.run(_AccessLog(PrmScopes(app, OAUTH_SCOPES)), host="0.0.0.0", port=PORT, log_config=None)


if __name__ == "__main__":
    main()
