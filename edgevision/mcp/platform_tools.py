"""控制 API 以外的服務的 MCP 工具（stdio 版 server.py 與遠端版 remote.py 共用一份定義）。

    模型改名控制 API   /api/models/{name}/versions/{v}/rename
    線路設定 / 偵測紀錄     output-api /api/config/lines、/api/detections、/api/stats、/api/cameras
    PTZ                   ptz        /api/ptz

原則：API 有的功能 MCP 都要有。tests/test_mcp.py 會比對每個 API 端點都有工具在打，
只有二進位 / 前端專用 / 服務內部回報的端點列在豁免清單。

哪個路徑打哪個服務看 service_url()：Ingress 本來就按路徑分流，所以 stdio 版連 Ingress 主機時全部用 CONTROL_URL 即可；
叢集內的 remote.py 直連各服務，靠 OUTPUT_URL / PTZ_URL 環境變數。
output-api 的寫入端點沒有驗 token，操作者寫在 body 的 actor —— 這裡一律填已驗證的身分，agent 不能自己給。
"""

from __future__ import annotations

import os
from typing import Any, Callable

# 路徑前綴 → 服務位址的環境變數（沒設就跟控制 API 同一個位址，即 Ingress）
_SERVICES = (
    ("/api/ptz", "PTZ_URL"),
    ("/api/config", "OUTPUT_URL"),
    ("/api/detections", "OUTPUT_URL"),
    ("/api/stats", "OUTPUT_URL"),
    ("/api/cameras", "OUTPUT_URL"),
)


def service_url(path: str, control_url: str) -> str:
    for prefix, env in _SERVICES:
        if path.startswith(prefix):
            return (os.environ.get(env) or control_url).rstrip("/")
    return control_url


def _clean(d: dict) -> dict:
    return {k: v for k, v in d.items() if v is not None}


def register(mcp, call: Callable, safe: Callable, RO, WRITE, DESTRUCTIVE, actor: Callable[[], str]) -> None:

    # ── 模型 ─────────────────────────────────────────────

    @mcp.tool(title="模型版本改名", annotations=DESTRUCTIVE)
    @safe
    def rename_model_version(model_name: str, version: int, new_name: str, reason: str,
                             new_version: int | None = None) -> dict:
        """⚠️ 把一個模型版本搬成另一個名字：檔案複製到 models/<new_name>/v<N>/、tag 沿用，
        舊的 Registry 紀錄封存、舊路徑的檔案刪除。現役（production）版本不准改。

        Args:
            new_version: 不給 = 新名字的下一版
            reason: 為什麼改名（必填，進稽核與模型備註）
        """
        if not reason.strip():
            raise ValueError("reason 必填")
        return call("POST", f"/api/models/{model_name}/versions/{version}/rename",
                    json_body=_clean({"new_name": new_name, "new_version": new_version, "reason": reason}))

    # ── 線路設定（output-api /api/config）──────────────────

    @mcp.tool(title="列出線路設定", annotations=RO)
    @safe
    def list_line_configs() -> list[dict]:
        """所有產線線路的摘要：站別、相機數、目前設定版本、runner 心跳與運作狀態。看單一條線用 get_line_config。"""
        return call("GET", "/api/config/lines")

    @mcp.tool(title="站別規格", annotations=RO)
    @safe
    def list_station_specs() -> list[dict]:
        """runner 映像裡有哪些站別：預設模型、必填現場值、門檻欄位（型別與預設值）、ROI 區域用途 roi_roles。
        畫 ROI 或改 thresholds 前先查，roi_regions[].role 要用這裡列的 name。"""
        return call("GET", "/api/config/stations")

    @mcp.tool(title="線路設定詳情", annotations=RO)
    @safe
    def get_line_config(line_id: str) -> dict:
        """一條線的完整設定（模型、ROI、站別參數、相機）與 config_version。改設定前先讀這個。"""
        return call("GET", f"/api/config/lines/{line_id}")

    @mcp.tool(title="線路設定歷史版本", annotations=RO)
    @safe
    def list_line_config_versions(line_id: str, limit: int = 20) -> list[dict]:
        """一條線每次存檔留下的版本：版本號、誰改的、什麼時候、原因。要還原用 restore_line_config。"""
        return call("GET", f"/api/config/lines/{line_id}/versions", params={"limit": limit})

    @mcp.tool(title="修改線路設定", annotations=DESTRUCTIVE)
    @safe
    def update_line_config(line_id: str, line: dict, reason: str, base_version: int) -> dict:
        """⚠️ 整份取代一條線的設定，存檔後 runner 會套用 —— 直接影響產線判定。只在使用者明確要求時用。

        Args:
            line: 完整線路設定（get_line_config 的內容改完整份送回；line.id 要等於 line_id）
            reason: 為什麼改（必填）
            base_version: get_line_config 拿到的 config_version；資料庫已經不是這版會 409（防覆蓋別人的修改）
        """
        if not reason.strip():
            raise ValueError("reason 必填")
        return call("PUT", f"/api/config/lines/{line_id}",
                    json_body={"line": line, "meta": {"actor": actor(), "reason": reason,
                                                      "base_version": base_version}})

    @mcp.tool(title="還原線路設定", annotations=DESTRUCTIVE)
    @safe
    def restore_line_config(line_id: str, version: int, reason: str, base_version: int | None = None) -> dict:
        """⚠️ 把線路設定還原成某個歷史版本（存成新的一版），runner 會套用。只在使用者明確要求時用。"""
        if not reason.strip():
            raise ValueError("reason 必填")
        return call("POST", f"/api/config/lines/{line_id}/restore/{version}",
                    json_body=_clean({"actor": actor(), "reason": reason, "base_version": base_version}))

    # ── 相機與偵測紀錄（output-api）────────────────────────

    @mcp.tool(title="列出相機", annotations=RO)
    @safe
    def list_cameras() -> list[dict]:
        """前端用的相機清單：每台相機屬於哪條線、哪個站別、串流名稱。PTZ 相機另見 list_ptz_cameras。"""
        return call("GET", "/api/cameras")

    @mcp.tool(title="推論中的相機", annotations=RO)
    @safe
    def list_cameras_in_process() -> Any:
        """目前有 runner 正在推論的相機清單（線上真的在跑判定的那些）。設定上啟用但沒在跑的不會出現。"""
        return call("GET", "/api/cameras/in-process")

    @mcp.tool(title="查偵測紀錄", annotations=RO)
    @safe
    def list_detections(camera_id: str | None = None, verdict: str | None = None, station: str | None = None,
                        line_id: str | None = None, reviewed: bool | None = None, limit: int = 50) -> Any:
        """最近的判定結果（OK / NG）與框。verdict：OK / NG；reviewed：只看已判 / 未判。"""
        return call("GET", "/api/detections", params=_clean({
            "camera_id": camera_id, "verdict": verdict, "station": station, "line_id": line_id,
            "reviewed": reviewed, "limit": limit}))

    @mcp.tool(title="判定一筆偵測", annotations=WRITE)
    @safe
    def review_detection(camera_id: str, frame_id: str, result: str) -> dict:
        """人工審核一個 NG 事件。result：true_positive（真的異常）/ false_positive（誤報）。"""
        return call("POST", "/api/detections/review",
                    json_body={"camera_id": camera_id, "frame_id": frame_id, "result": result, "actor": actor()})

    @mcp.tool(title="偵測統計", annotations=RO)
    @safe
    def get_detection_stats(hours: int = 24) -> dict:
        """推論 API 的偵測統計：最近 hours 小時內各相機 / 站別的判定數量（OK、NG）。管理報表的統計用 get_report_stats。"""
        return call("GET", "/api/stats", params={"hours": hours})

    # ── PTZ（ptz 服務）─────────────────────────────────────

    @mcp.tool(title="PTZ 相機清單", annotations=RO)
    @safe
    def list_ptz_cameras() -> list[dict]:
        """PTZ 服務管理的相機：能不能轉（can_pan）、歸位點、中心點、失準偵測基準、目前失準狀態。單台細節用 get_ptz_state。"""
        return call("GET", "/api/ptz/cameras")

    @mcp.tool(title="PTZ 相機狀態", annotations=RO)
    @safe
    def get_ptz_state(camera_id: str) -> dict:
        """一台的目前座標、內建 preset、歸位點、中心點、失準狀態（一次查齊）。"""
        return {
            "status": call("GET", f"/api/ptz/{camera_id}/status"),
            "presets": call("GET", f"/api/ptz/{camera_id}/presets"),
            "target": call("GET", f"/api/ptz/{camera_id}/target"),
            "origin": call("GET", f"/api/ptz/{camera_id}/origin"),
            "drift": call("GET", f"/api/ptz/{camera_id}/drift"),
        }

    @mcp.tool(title="轉動 PTZ", annotations=WRITE)
    @safe
    def move_ptz(camera_id: str, mode: str, pan: float = 0, tilt: float = 0, zoom: float = 0) -> dict:
        """⚠️ 會實際轉動產線相機，畫面跟著動 —— 正在推論的線會受影響。

        Args:
            mode: relative（固定步進，鏡頭自己停）/ absolute（到指定座標）/
                continuous（連續移動，2.5 秒沒重送自動停）/ stop
        """
        body = {"pan": pan, "tilt": tilt, "zoom": zoom}
        if mode == "relative":
            return call("POST", f"/api/ptz/{camera_id}/relative_move", json_body=body)
        if mode == "absolute":
            return call("POST", f"/api/ptz/{camera_id}/absolute_move", json_body=body)
        if mode == "continuous":
            return call("POST", f"/api/ptz/{camera_id}/move", json_body=body)
        if mode == "stop":
            return call("POST", f"/api/ptz/{camera_id}/stop")
        raise ValueError("mode 只能是 relative / absolute / continuous / stop")

    @mcp.tool(title="PTZ 回到定點", annotations=WRITE)
    @safe
    def goto_ptz(camera_id: str, where: str, preset_index: int | None = None) -> dict:
        """⚠️ 會轉動相機。where：target（歸位點）/ origin（軟體中心點）/ preset（內建 preset，要給 preset_index）。"""
        if where == "target":
            return call("POST", f"/api/ptz/{camera_id}/target/goto")
        if where == "origin":
            return call("POST", f"/api/ptz/{camera_id}/origin/goto")
        if where == "preset":
            if preset_index is None:
                raise ValueError("where=preset 要給 preset_index")
            return call("POST", f"/api/ptz/{camera_id}/presets/{preset_index}/goto")
        raise ValueError("where 只能是 target / origin / preset")

    @mcp.tool(title="PTZ 存目前位置", annotations=DESTRUCTIVE)
    @safe
    def save_ptz_position(camera_id: str, what: str, preset_name: str | None = None) -> dict:
        """⚠️ 把相機目前位置存成：target（歸位點，排程每 10 分鐘回這裡）/ origin（中心點）/
        reference（失準偵測基準，存了就進入自動監控）/ preset（相機內建 preset，要給 preset_name）。
        會蓋掉舊的設定，只在使用者明確要求時用。"""
        if what == "target":
            return call("POST", f"/api/ptz/{camera_id}/target")
        if what == "origin":
            return call("POST", f"/api/ptz/{camera_id}/origin")
        if what == "reference":
            return call("POST", f"/api/ptz/{camera_id}/reference")
        if what == "preset":
            if not preset_name:
                raise ValueError("what=preset 要給 preset_name")
            return call("POST", f"/api/ptz/{camera_id}/presets", json_body={"name": preset_name})
        raise ValueError("what 只能是 target / origin / reference / preset")

    @mcp.tool(title="PTZ 失準檢查 / 回正", annotations=WRITE)
    @safe
    def ptz_drift(camera_id: str, action: str, force: bool = False) -> dict:
        """action：check（只量測，不動相機；force=True 忽略跳過條件）/
        recover（⚠️ 立即回正，會轉動相機，背景執行，進度看 get_ptz_state 的 drift）。"""
        if action == "check":
            return call("POST", f"/api/ptz/{camera_id}/drift/check", params={"force": force})
        if action == "recover":
            return call("POST", f"/api/ptz/{camera_id}/drift/recover")
        raise ValueError("action 只能是 check / recover")

    @mcp.tool(title="PTZ 馬達校準", annotations=DESTRUCTIVE)
    @safe
    def calibrate_ptz_motor(camera_id: str) -> dict:
        """⚠️ Tapo 原生馬達校準：相機自己掃描上下左右再回中心（20–40 秒，期間畫面不能用）。"""
        return call("POST", f"/api/ptz/{camera_id}/calibrate_motor")

    @mcp.tool(title="設定能否轉動", annotations=WRITE)
    @safe
    def set_ptz_can_pan(camera_id: str, can_pan: bool | None) -> dict:
        """這台相機能不能轉。null = 用 ONVIF 唯讀自動偵測（不會轉動相機）。"""
        return call("PUT", f"/api/ptz/{camera_id}/can-pan", json_body={"can_pan": can_pan})
