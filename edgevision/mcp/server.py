"""edgevision MCP server：讓 agent 用自然語言操作訓練平台。

定位：
    這是控制 API 的【另一個客戶端】，跟前端 /ui、命令列工具平起平坐。
    它不碰資料庫、不碰 GPU，所有動作都是打 HTTP 給控制 API，
    所以權限、稽核、佇列那些規則自動沿用，不需要在這裡再實作一次。

身分：
    本機 stdio 模式。憑證在 ~/.edgevision/credentials.json，一次 `edgevision-login` 可用 30 天，
    access token 由 credentials.py 自動續期，使用者不用管 token。
    每個呼叫帶 Bearer token 與 X-Channel: mcp → 控制 API 以驗證過的 sub 當 actor 寫進 audit_log。

工具分三級（annotations 標示，client 可據此決定要不要跳確認）：
    🟢 唯讀     read_only_hint=True      查詢，隨便呼叫都沒關係
    🟡 寫入消耗資源但不動線上服務建訓練、取消、匯入資料集
    🔴 破壞性   destructive_hint=True    promote / rollback —— 會換掉產線正在用的模型

⚠️ 🔴 那兩個雖然提供給 agent，但 annotations 已標成破壞性且非冪等，
   MCP client 應該要跳確認。設計文件的立場是「品質檢測場景永遠保留人工放行環節」。
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Literal

logger = logging.getLogger(__name__)

CONTROL_URL = os.environ.get("CONTROL_URL", "http://localhost:8095").rstrip("/")
TIMEOUT = float(os.environ.get("MCP_HTTP_TIMEOUT", "60"))


# ── 與控制 API 溝通 ────────────────────────────────────────

def _headers() -> dict[str, str]:
    """每次呼叫都取一次 token（過期會自動續期）。"""
    from edgevision.mcp.credentials import get_access_token

    return {
        "Authorization": f"Bearer {get_access_token()}",
        "X-Channel": "mcp",          # 稽核看得出這是 agent 代為執行的
        "Content-Type": "application/json",
    }


class ToolError(Exception):
    """可以直接呈現給 agent 的錯誤。

    ⚠️ MCP SDK 會把未預期的例外吞成 "Error executing tool xxx"，
    agent 就看不到「請先登入」這種可行動的訊息。所以我們自己回傳結構化的錯誤，
    而不是讓例外往上拋。
    """


def _err(message: str, hint: str = "") -> dict:
    out = {"ok": False, "error": message}
    if hint:
        out["hint"] = hint
    return out


def _call(method: str, path: str, *, json_body: dict | None = None, params: dict | None = None) -> Any:
    """打控制 API。錯誤一律轉成人看得懂的訊息，讓 agent 能自己判斷下一步。"""
    import httpx

    from edgevision.mcp.credentials import NotLoggedIn

    try:
        headers = _headers()
    except NotLoggedIn as exc:
        raise ToolError(f"尚未登入或憑證已過期。{exc}") from exc

    from edgevision.mcp.platform_tools import service_url

    base = service_url(path, CONTROL_URL)
    url = f"{base}{path}"
    try:
        with httpx.Client(timeout=TIMEOUT) as c:
            r = c.request(method, url, headers=headers, json=json_body, params=params)
    except Exception as exc:
        raise ToolError(f"連不上控制 API（{base}）：{exc}") from exc

    if r.status_code == 401:
        raise ToolError("控制 API 拒絕這個身分（401）。請重新執行：python -m edgevision.mcp.login")
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
    """把工具的錯誤轉成 agent 看得懂的結構化回傳，而不是被 SDK 吞成 'Error executing tool'。"""
    import functools

    @functools.wraps(fn)
    def wrapper(*a, **kw):
        try:
            return fn(*a, **kw)
        except ToolError as exc:
            hint = ""
            m = str(exc)
            if "尚未登入" in m or "401" in m:
                hint = "請在終端機執行：python -m edgevision.mcp.login"
            elif "連不上控制 API" in m:
                hint = f"確認平台是否運行：curl --noproxy '*' {CONTROL_URL}/healthz"
            return _err(m, hint)
        except ValueError as exc:          # 參數不合法（例如 notes 沒填）
            return _err(str(exc))
    return wrapper


def build_server():
    from mcp.server.mcpserver import MCPServer
    from mcp.types import ToolAnnotations

    mcp = MCPServer("edgevision")

    RO = ToolAnnotations(read_only_hint=True, idempotent_hint=True)
    WRITE = ToolAnnotations(read_only_hint=False, destructive_hint=False, idempotent_hint=False)
    DESTRUCTIVE = ToolAnnotations(read_only_hint=False, destructive_hint=True, idempotent_hint=False)

    # ═══ 🟢 唯讀 ═══════════════════════════════════════════

    @mcp.tool(title="平台狀態", annotations=RO)
    @_safe
    def platform_health() -> dict:
        """控制 API、資料庫、物件儲存、GPU 執行器是否正常，以及目前有幾個任務在排隊。

        遇到任何「連不上」「卡住」的狀況時先呼叫這個。
        """
        return _call("GET", "/healthz")

    @mcp.tool(title="我是誰", annotations=RO)
    @_safe
    def whoami() -> dict:
        """目前用哪個身分操作平台、憑證還有多久過期。

        所有動作都會以這個身分寫進稽核紀錄。
        """
        from edgevision.mcp.credentials import status

        return status()

    @mcp.tool(title="列出模型", annotations=RO)
    @_safe
    def list_models() -> list[dict]:
        """Registry 裡有哪些模型、各自現役（production）是哪一版、有哪些版本。"""
        return _call("GET", "/api/models")

    @mcp.tool(title="列出模型版本", annotations=RO)
    @_safe
    def list_model_versions(model_name: str) -> list[dict]:
        """某個模型的所有版本，含 tag（production / candidate / rejected）與指標。

        Args:
            model_name: 模型名稱，例如 seg_alpha、wildlife_det
        """
        return _call("GET", f"/api/models/{model_name}/versions")

    @mcp.tool(title="模型版本詳情", annotations=RO)
    @_safe
    def get_model_version(model_name: str, version: int) -> dict:
        """單一版本的完整身分：血緣（用哪個資料集版本、哪份程式碼、從哪個模型訓的）、
        指標、artifacts 在 MinIO 的位置。

        診斷「這個模型是怎麼來的」時用這個。
        """
        return _call("GET", f"/api/models/{model_name}/versions/{version}")

    @mcp.tool(title="列出任務", annotations=RO)
    @_safe
    def list_tasks(status: str | None = None, type: str | None = None, limit: int = 20) -> list[dict]:
        """訓練與部署任務清單。

        Args:
            status: 篩選狀態 queued / working / completed / failed / cancelled
            type: 篩選類型 training / deployment
            limit: 最多回幾筆
        """
        params = {"limit": limit}
        if status:
            params["status"] = status
        if type:
            params["type"] = type
        return _call("GET", "/api/tasks", params=params)

    @mcp.tool(title="任務詳情", annotations=RO)
    @_safe
    def get_task(task_id: str) -> dict:
        """單一任務的完整狀態：進度、心跳訊息、評估 Gate 結果、指標、一致性驗證、錯誤。

        訓練跑完後要判斷「結果好不好」就看這個。
        """
        return _call("GET", f"/api/tasks/{task_id}")

    @mcp.tool(title="任務 log", annotations=RO)
    @_safe
    def get_task_logs(task_id: str, step: str = "trainer", tail: int = 100) -> str:
        """某個步驟的容器 log。任務失敗時用來找真正的原因。

        Args:
            task_id: 任務 ID
            step: trainer（訓練）/ trtexec（轉檔）/ finalize（收尾）/ parity-triton
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

        ⚠️ 正式的訓練設定應該來自版本控管的訓練設定，不是這裡。
        這個清單只是示範與整合測試用。
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

    # ═══ 🟡 寫入（消耗資源，但不動線上服務）═══════════════

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
        model_name: str,
        dataset_ref: str,
        notes: str,
        golden_ref: str | None = None,
        config: dict | None = None,
        gates: dict | None = None,
        overrides: dict | None = None,
        parent_task_id: str | None = None,
        clone_reason: str | None = None,
        config_source: str = "mcp",
    ) -> dict:
        """送出一個訓練任務。會佔用 GPU 數十分鐘，同時只能跑一個（其餘排隊）。

        ⚠️ notes 是必填：寫清楚這次為什麼要訓練、想驗證什麼假設。
        它會跟著模型血緣走，是日後回溯「這一版為什麼這樣訓」的唯一線索。

        ⚠️ 這個工具【不提供 auto_promote】。訓練完成後模型是 candidate，
        要上線必須另外呼叫 promote_model，讓人有機會看過結果再決定。

        Args:
            model_name: 模型名稱，決定註冊到 Registry 的哪個模型
            dataset_ref: 資料集版本 ID，形如 lakefs://alpha/<版本id>
            notes: 這次訓練的目的與假設（必填）
            golden_ref: 黃金測試集版本 ID。省略則用 val 當 Gate 依據（分數會偏樂觀）
            config: 完整訓練設定。省略則用平台自帶的範例設定
            gates: 完整 Gate 規則。省略則用設定裡指定的
            overrides: 臨時覆寫，例如 {"train": {"epochs": 2}}
            parent_task_id: 從哪個任務衍生（clone），血緣會接上
            clone_reason: 為什麼從那個任務衍生
            config_source: 設定的來源，供血緣追溯
        """
        if not notes.strip():
            raise ValueError("notes 必填：請說明這次訓練的目的與假設")
        body: dict[str, Any] = {
            "model_name": model_name, "dataset_ref": dataset_ref, "notes": notes,
            "config_source": config_source, "auto_promote": False,   # ⚠️ 固定 False
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
        """取消進行中的任務。已經跑掉的訓練成果會遺失。

        Args:
            task_id: 要取消的任務 ID
        """
        return _call("POST", f"/api/tasks/{task_id}/cancel")

    # ═══ 🔴 破壞性（換掉產線正在用的模型）═════════════════

    @mcp.tool(title="⚠️ 模型上線", annotations=DESTRUCTIVE)
    @_safe
    def promote_model(model_name: str, version: int, reason: str) -> dict:
        """把指定版本設為 production 並部署到 Triton —— **會立刻換掉產線正在用的模型**。

        ⚠️ 這是會影響線上檢測的操作。呼叫前務必：
        ① 確認該版本通過了評估 Gate（get_task 或 get_model_version 看得到）
        ② 逐類比對新舊版本的 recall，確認沒有某一類明顯退步
        ③ 讓使用者明確同意 —— 不要自行判斷後直接執行

        Args:
            model_name: 模型名稱
            version: 要上線的版本號
            reason: 為什麼上線這一版（會寫進稽核紀錄）
        """
        if not reason.strip():
            raise ValueError("reason 必填：上線是會影響產線的操作，必須說明理由")
        return _call("POST", f"/api/models/{model_name}/promote",
                     json_body={"version": version, "reason": reason})

    @mcp.tool(title="⚠️ 模型回退", annotations=DESTRUCTIVE)
    @_safe
    def rollback_model(model_name: str, version: int, reason: str) -> dict:
        """回退到舊版本並部署到 Triton —— **會立刻換掉產線正在用的模型**。

        相較於 promote，回退是往「已知可用的舊版」走，風險較低，
        新版出問題時應該果斷回退。但它同樣改變線上服務，仍需使用者同意。

        Args:
            model_name: 模型名稱
            version: 要回退到的版本號
            reason: 為什麼回退（會寫進稽核紀錄）
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
    def _me() -> str:
        from edgevision.mcp.credentials import status
        return status().get("subject") or ""
    _register_platform(mcp, _call, _safe, RO, WRITE, DESTRUCTIVE, _me)

    return mcp


def main() -> None:
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)-7s %(name)s %(message)s",
        stream=__import__("sys").stderr,     # ⚠️ stdio 模式：log 一定要走 stderr，stdout 是協定通道
    )
    build_server().run(transport="stdio")


if __name__ == "__main__":
    main()
