"""API 回應型別 —— 讓 OpenAPI schema 完整，前端才能自動產生型別。

⚠️ 為什麼需要這個檔案

    端點原本一律標 `-> dict` / `-> list[dict]`。FastAPI 照樣能跑，
    /docs 也照樣開得起來，但 OpenAPI 裡那些回應全是無結構的 object ——
    前端跑 openapi-typescript 產出來的型別是 `any`，等於沒有型別。

    請求端有 model（LineIn / CameraIn / …）所以是好的；缺的一直是回應端。

⚠️ 這裡的每個欄位都必須跟實際回傳一致

    這些 model 只用來宣告 OpenAPI schema（response_model），
    FastAPI 會照它序列化 —— 少宣告一個欄位，那個欄位就【不會回給前端】。
    改端點的回傳結構時，這裡要一起改。tests/test_api_schemas.py 會比對。

WebSocket 不在 OpenAPI 規範內，`/ws` 的訊息格式另外用 WsResult 等 model 宣告，
並由 /api/frontend-config 的 ws_schema 指向它們，前端一樣產得出型別。
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


# ── 相機與線路 ────────────────────────────────────────────

class CameraOut(BaseModel):
    """GET /api/cameras 的單筆。前端要顯示什麼、怎麼取畫面，都在這。"""

    id: str
    name: str
    line_id: str
    station: str
    width: int
    height: int
    # 線路失聯（心跳過期 60 秒）時視為 unknown
    state: Literal["online", "unknown"]
    # ⭐ 串流名稱與位址由後端組，前端不必知道 _marked 這個慣例
    view_stream: str
    view_url: str
    # 現場校正值（正規化 0~1），前端畫基準四邊形用
    station_params: dict[str, Any] = Field(default_factory=dict)


class LineSummary(BaseModel):
    """GET /api/config/lines 的單筆：desired 與 reported 的差異一眼看得出來。"""

    id: str
    name: str | None = None
    station: str
    desired_version: int
    line_params: dict[str, Any] = Field(default_factory=dict)
    enabled: bool
    reported_version: int | None = None
    # 心跳斷了就不算同步 —— 那個版本號只是舊快照
    in_sync: bool
    apply_state: str
    cameras_running: int | None = None
    heartbeat_age_s: int | None = None
    cameras_configured: int | None = None
    # runner 失聯（heartbeat_age_s > 60）。true 時上面三個數字都不可信
    stale: bool = False


class CameraHealth(BaseModel):
    """/healthz 的 lines[].cameras[] 單筆。"""

    camera_id: str
    # ok：有畫面；no_frames：線路活著但這台 0 幀；stale：整條線失聯；unknown：舊版 runner 沒回報
    # skipped：這台設定有誤，runner 沒啟動它（reason 寫原因）；同線其他台照跑
    state: Literal["ok", "no_frames", "stale", "unknown", "skipped"]
    fps: float | None = None
    reason: str | None = None


class LineStatus(BaseModel):
    """/healthz 的 lines[] 單筆。"""

    line_id: str
    name: str | None = None
    station: str
    desired_version: int
    reported_version: int | None = None
    in_sync: bool
    apply_state: str
    apply_error: str | None = None
    cameras_running: int | None = None
    heartbeat_age_s: int | None = None
    stale: bool = False
    # 整條線這 15 秒的推論速度（所有相機合計）
    fps: float | None = None
    cameras: list[CameraHealth] = Field(default_factory=list)


class ServiceStatus(BaseModel):
    """/healthz 的 services[] 單筆：消費者的「在做事嗎」。

    state：ok / degraded（活著但指標超門檻）/ stale（60 秒沒心跳）。
    reason 講人話，例如「3 支影片在 DB 找不到對應列（孤兒）」。
    """

    service: str
    instance: str | None = None
    state: Literal["ok", "degraded", "stale"]
    reason: str | None = None
    heartbeat_age_s: int | None = None
    uptime_s: int | None = None
    # 各服務自報的指標，內容依 service 而異（見 04-service-status.sql）
    metrics: dict[str, Any] = Field(default_factory=dict)


class HealthOut(BaseModel):
    """GET /healthz。

    ⚠️ healthy 只反映 API 自己 —— 線路的死活看 lines[]，消費者的死活看 services[]。
    否則推論端出問題會害 API 被 autoheal 重啟。
    """

    healthy: bool
    clients: int = 0
    storage: dict[str, bool] = Field(default_factory=dict)
    mode: Literal["standalone", "in-process"] = "standalone"
    lines: list[LineStatus] = Field(default_factory=list)
    services: list[ServiceStatus] = Field(default_factory=list)


# ── 偵測結果 ──────────────────────────────────────────────

class BBoxOut(BaseModel):
    """原圖像素座標（coord_system=original 時已由後端還原 letterbox）。"""

    x1: float
    y1: float
    x2: float
    y2: float


class DetectionOut(BaseModel):
    class_id: int | None = None
    class_name: str
    confidence: float
    bbox: BBoxOut
    # seg 模型才有：多邊形輪廓，座標系同 bbox
    has_mask: bool = False
    mask_area: float | None = None
    mask_contour: list[list[float]] | None = None
    # pose 模型才有：每點 [x, y, conf]
    keypoints: list[list[float]] | None = None
    # 業務層衍生值（各站別自訂）
    metrics: dict[str, Any] = Field(default_factory=dict)


class VerdictOut(BaseModel):
    status: Literal["ok", "ng"]
    reason: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)


class DetectionRow(BaseModel):
    """GET /api/detections 的單筆（DB 的一列 = 一個物件，不是一幀）。

    整幀判定的 NG（靠時間判、當下沒有框，例如「A 之後 N 秒內沒看到 B」）只有一列，
    class_name / confidence / bbox 都是 null，顯示 verdict_reason 即可。
    """

    id: int
    ts: str
    camera_id: str
    frame_id: str
    marker_id: int | None = None
    model_name: str
    model_version: str
    station: str | None = None
    class_name: str | None = None
    confidence: float | None = None
    # 原圖像素座標 {x1, y1, x2, y2}；整幀判定的 NG 為 null
    bbox: BBoxOut | None = None
    bbox_coord_system: str = "original"
    # {width, height, letterbox?}
    image_meta: dict[str, Any] | None = None
    mask_area: float | None = None
    verdict_status: Literal["ok", "ng"] | None = None
    verdict_reason: str | None = None
    verdict_details: dict[str, Any] | None = None
    # 存證圖（那一幀）。K8s 版目前不產生截圖，一律 null；存證以 clip_uri 為主
    image_uri: str | None = None
    # ⭐ 事件影片（NG 前 8 秒 + 後 5 秒），同一幀的每一列都會帶同一支；用 GET /api/image?uri= 換下載網址。
    #    null = 還沒錄好（約 10~20 秒後出現）
    clip_uri: str | None = None
    # 影片保留 7 天：過期後 clip_uri / image_uri 為 null，這裡記過期時間 —— 顯示「影片已過期」
    evidence_expired_at: str | None = None
    # 人工審核（POST /api/detections/review）：同一幀的每一列一起標
    reviewed: bool = False
    review_result: Literal["true_positive", "false_positive"] | None = None
    reviewed_by: str | None = None
    reviewed_at: str | None = None


class StatsOut(BaseModel):
    """GET /api/stats：最近 hours 小時的列數統計（一列 = 一個物件，不是一幀）。"""

    total: int
    ng: int
    with_image: int
    reviewed: int
    cameras: int


class ImageUrlOut(BaseModel):
    """GET /api/image：s3:// URI 換成的臨時下載網址（約 60 分鐘有效），瀏覽器直接取檔。"""

    url: str


class ReviewIn(BaseModel):
    """POST /api/detections/review：人工審核一個 NG 事件（同一幀的所有列一起標）。"""

    camera_id: str
    frame_id: str
    # true_positive = 確認是真的異常；false_positive = 誤報
    result: Literal["true_positive", "false_positive"]
    actor: str = Field(min_length=1, description="審核者帳號")


class ReviewOut(BaseModel):
    camera_id: str
    frame_id: str
    result: str
    rows: int


# ── WebSocket 訊息（不在 OpenAPI 規範內，但前端一樣要型別）────

class WsFlipRefs(BaseModel):
    """連線時先收到的一則：各相機的基準四邊形。"""

    type: Literal["ref_quads"]
    ref_quads: dict[str, list[list[float]]]


class ImageMetaOut(BaseModel):
    width: int
    height: int
    letterbox: dict[str, Any] | None = None


class WsResult(BaseModel):
    """/ws 的每筆推論結果。

    ⚠️ 這是廣播 —— 所有相機的結果都會送到每個連線，前端自己用 camera_id 過濾。
    """

    type: Literal["result"]
    camera_id: str
    frame_id: str
    # 與畫面上 marker 配對用（overlay.mode=client 時才需要）
    marker_id: int | None = None
    model_name: str | None = None
    model_version: str | None = None
    detections: list[DetectionOut] = Field(default_factory=list)
    verdict: VerdictOut | None = None
    image_meta: ImageMetaOut | None = None
    coord_system: Literal["original", "letterboxed"] = "original"


# ── 前端設定 ──────────────────────────────────────────────

class StreamingCfg(BaseModel):
    base_url: str
    view_suffix: str
    mse_url_template: str
    preferred_transport: str


class ResultsCfg(BaseModel):
    websocket_path: str
    # 結果比這個舊就不該再沿用；由推論頻率推導
    max_age_ms: int
    note: str


class OverlayCfg(BaseModel):
    # server = 框已燒進影像，前端只要播；client = 前端自己疊（才需要 marker）
    mode: Literal["server", "client"]
    note: str


class FrontendConfig(BaseModel):
    """GET /api/frontend-config —— 前端啟動時讀一次，不要輪詢。

    ⚠️ 不要硬編任何位址、後綴或 marker 參數，全部從這裡拿。
    """

    schema_version: str
    streaming: StreamingCfg
    results: ResultsCfg
    overlay: OverlayCfg
    # 只有 overlay.mode = client 時才需要；server 模式下 enabled=false
    marker: dict[str, Any] = Field(default_factory=dict)


class DetectionsPage(BaseModel):
    """GET /api/detections 的回應。

    ⚠️ 一列 = 一個偵測物件，不是一幀 —— 同一幀的多個物件會有相同的 frame_id。
    """

    count: int
    detections: list[DetectionRow] = Field(default_factory=list)
