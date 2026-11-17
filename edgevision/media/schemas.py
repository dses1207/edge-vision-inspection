"""圖片庫 API 的回應 schema（前端靠 OpenAPI 產生型別，每個 2xx 都要有 —— tests/test_api_schemas.py）。"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

Status = Literal["unlabeled", "auto", "labeled", "background"]


class ShapeSourceOut(BaseModel):
    kind: Literal["human", "model"]
    engine: str | None = None
    model: str | None = None
    version: int | None = None
    conf: float | None = None


class ShapeOut(BaseModel):
    id: str | None = Field(None, description="框的 id：修改時保留，審核靠它比對原樣 / 修改 / 新增 / 刪除")
    type: Literal["rect", "polygon", "keypoints"]
    class_: str = Field(alias="class")
    points: list[list[float]] = Field(description="0–1 正規化；rect 兩點（左上、右下）；keypoints 每點 [x, y, v]")
    bbox: list[list[float]] | None = None
    source: ShapeSourceOut | None = None


class ProgressOut(BaseModel):
    unlabeled: int
    auto: int
    labeled: int
    background: int
    total: int


class CollectionOut(BaseModel):
    id: int
    name: str
    description: str
    class_names: list[str]
    kpt_shape: list[int] | None = Field(None, description="關鍵點標注的 [點數, 3]；沒有關鍵點 = null")
    line_id: str | None = None
    camera_id: str | None = None
    org: str | None = Field(None, description="分類；null = 未分類")
    archived: bool
    created_by: str
    created_at: datetime
    updated_at: datetime
    progress: ProgressOut | None = None


class OrgOut(BaseModel):
    org: str | None = Field(description="null = 未分類")
    n_collections: int


class CollectionSummaryOut(CollectionOut):
    n_items: int
    n_done: int = Field(description="labeled + background")


class ItemSummaryOut(BaseModel):
    id: int
    blob_sha256: str
    batch_id: int | None = Field(None, description="屬於哪個批次")
    status: Status
    revision: int
    original_name: str | None = None
    camera_id: str | None = None
    captured_at: datetime | None = None
    updated_by: str | None = None
    updated_at: datetime | None = None
    locked_by: str | None = None
    lock_until: datetime | None = None
    n_shapes: int
    width: int | None = None
    height: int | None = None
    image_url: str | None = None
    tags: list[str] = Field(default_factory=list, description="圖片標籤（PATCH /items/{id}/meta）")
    note: str | None = Field(None, description="圖片備註")


class ItemOut(BaseModel):
    id: int
    collection_id: int
    blob_sha256: str
    status: Status
    shapes: list[ShapeOut]
    revision: int
    image_url: str | None = Field(None, description="臨時下載網址（60 分鐘）")
    width: int | None = None
    height: int | None = None
    class_names: list[str]
    kpt_shape: list[int] | None = None
    original_name: str | None = None
    source_id: int | None = None
    source_offset_s: float | None = None
    camera_id: str | None = None
    captured_at: datetime | None = None
    select_info: dict[str, Any] | None = Field(None, description="怎麼被挑中的（strategy=diverse）：moving / motion / group / group_size；告警回流的圖在 alert")
    tags: list[str] = Field(default_factory=list, description="圖片標籤")
    note: str | None = Field(None, description="圖片備註（跟審核的存疑 note 無關）")
    added_by: str
    added_at: datetime
    updated_by: str | None = None
    updated_at: datetime | None = None
    locked_by: str | None = None
    lock_until: datetime | None = None
    review: "ItemReviewOut | None" = Field(None, description="這張圖所在的審核輪次（不在任何輪次 = null）")


class ItemReviewOut(BaseModel):
    round_id: int
    round_status: Literal["open", "closed"]
    round_mode: Literal["review", "seed"] = Field("review", description="seed = 種子輪：整輪沒有預標、從頭標")
    blind: bool = Field(description="抽查圖：預標已清空，請從頭標")
    priority: float
    reasons: list[str]
    rank: int
    sequence_id: int | None = None
    state: Literal["pending", "done", "doubt"]
    doubt_note: str | None = None
    reviewed_by: str | None = None


class NextOut(BaseModel):
    item: ItemOut | None = Field(description="null = 沒有符合條件的下一張（都標完了，或其他的都被別人領走）")
    remaining: int = Field(description="符合條件的張數（含這張）")


class UploadResultOut(BaseModel):
    filename: str
    item_id: int | None = None
    sha256: str | None = None
    new_in_library: bool | None = Field(None, description="false = 圖片庫裡已有同一張圖（不重存）")
    added: bool | None = Field(None, description="false = 這個圖片集已經有這張")
    error: str | None = None


class UploadImagesOut(BaseModel):
    results: list[UploadResultOut]
    added: int
    batch_id: int = Field(description="這次上傳的批次（新開的，或 batch_id 指定的既有批次）")


class MediaUploadUrlOut(BaseModel):
    upload_url: str
    source_uri: str
    expires_minutes: int
    kind: Literal["zip", "video"]


class TaskCreatedOut(BaseModel):
    task_id: str
    queue_position: int | None = None
    batch_id: int | None = Field(None, description="匯入：這次匯入的批次（新開的，或指定的既有批次）")


class CountOut(BaseModel):
    removed: int | None = None
    added: int | None = None
    batch_id: int | None = Field(None, description="複製：目標圖片集裡的批次（新開的，或指定的既有批次）")


class BatchDeleteOut(BaseModel):
    batch_id: int
    collection_id: int
    removed: int = Field(description="從圖片集移除的張數（圖片庫原圖不動）")


class BatchOut(BaseModel):
    id: int
    collection_id: int
    name: str
    kind: Literal["upload", "zip", "video", "nas", "copy", "alert"]
    params: dict[str, Any] = Field(description="來源參數；回填的舊資料帶 backfill: true。之後用 batch_id 加進來的每一次記在"
                                               " params.appends（kind、actor、at、task_id…）")
    task_id: str | None = Field(None, description="建立這批時的背景匯入任務（上傳 / 複製是同步的，沒有）")
    created_by: str
    created_at: datetime
    n_items: int
    n_unlabeled: int
    n_auto: int = Field(description="AI 預標、還沒人確認")
    n_done: int = Field(description="已確認（labeled + background）")


class LockOut(BaseModel):
    locked_by: str
    lock_until: datetime


class OkOut(BaseModel):
    ok: bool


class RevisionOut(BaseModel):
    item_id: int
    revision: int
    status: Literal["labeled", "background"] | None = None


class HistoryOut(BaseModel):
    id: int
    item_id: int
    revision: int
    status: Status
    shapes: list[ShapeOut]
    action: Literal["submit", "autolabel", "import", "blind"] = Field(
        description="blind = 建輪次時被抽成抽查圖、預標清空（原預測留在輪次的 baseline）")
    actor: str
    task_id: str | None = None
    at: datetime


class PredictOut(BaseModel):
    item_id: int
    model: str
    version: int
    shapes: list[ShapeOut]


class ExportRecordOut(BaseModel):
    id: int
    dataset_repo: str
    dataset_ref: str | None = None
    task_id: str | None = None
    n_items: int
    batch_ids: list[int] | None = Field(None, description="匯出了哪些批次（null = 整個圖片集）")
    status: Literal["running", "completed", "failed"]
    error: str | None = None
    created_by: str
    created_at: datetime


# ── 審核──────────────────────────────

class ReviewClassOut(BaseModel):
    name: str
    color: str
    hotkey: str | None = None
    show_conf: float
    accept_conf: float
    bulk_accept: bool
    rare: bool


class ReviewConfigOut(BaseModel):
    """合併後的完整審核設定（預設 ← 圖片集覆蓋）。前端只讀這份，不自帶預設數字。"""
    classes: list[ReviewClassOut]
    swap_pairs: list[list[str]]
    round: dict[str, Any]
    priority: dict[str, Any]
    sequence: dict[str, Any]
    bulk_accept: dict[str, Any]
    review: dict[str, Any]
    qa: dict[str, Any]
    thumb_px: int
    guide: list[dict[str, Any]]
    override: dict[str, Any] = Field(description="這個圖片集存在資料庫裡的覆蓋（只有跟預設不同的欄位）")


class ReviewerStatOut(BaseModel):
    actor: str
    n: int
    avg_seconds: float | None = None


class RoundProgressOut(BaseModel):
    pending: int
    labeled: int
    background: int
    doubt: int


class RoundOut(BaseModel):
    id: int
    collection_id: int
    name: str
    status: Literal["open", "closed"]
    mode: Literal["review", "seed"] = Field("review", description="seed = 種子輪：整輪沒有預標、從頭標")
    model: dict[str, Any] | None = Field(None, description="預標用的模型 {name, version}")
    batch_ids: list[int] | None = Field(None, description="從哪些批次挑（null = 不限）")
    n_items: int
    n_blind: int
    n_sequences: int
    progress: RoundProgressOut
    by_reason: dict[str, int]
    reviewers: list[ReviewerStatOut]
    config: dict[str, Any] | None = Field(None, description="建立當下的完整設定（清單不回）")
    created_by: str
    created_at: datetime
    closed_by: str | None = None
    closed_at: datetime | None = None


class RoundItemOut(BaseModel):
    id: int
    status: Status
    revision: int
    state: Literal["pending", "done", "doubt"]
    priority: float
    reasons: list[str]
    blind: bool
    sequence_id: int | None = None
    rank: int
    width: int | None = None
    height: int | None = None
    thumb_url: str | None = None
    shapes: list[ShapeOut] = Field(description="抽查圖審完前永遠是空的")
    reviewed_by: str | None = None
    locked_by: str | None = None


class RoundItemsOut(BaseModel):
    items: list[RoundItemOut]
    total: int
    offset: int
    limit: int


class PeekOut(BaseModel):
    id: int
    image_url: str | None = None


class RoundNextOut(BaseModel):
    item: ItemOut | None = Field(description="null = 這輪審完了（或剩下的都被別人領走）")
    remaining: int = Field(description="這輪還沒審的張數（含這張）")
    peek: list[PeekOut] = Field(default_factory=list, description="後面幾張的圖片網址，給前端預載（沒有鎖）")


class ReviewDiffOut(BaseModel):
    accepted: int
    modified: int
    added: int
    deleted: int
    class_changed: int
    per_class: dict[str, dict[str, int]]


class ReviewResultOut(BaseModel):
    item_id: int
    revision: int
    status: Status
    state: Literal["pending", "done", "doubt"]
    diff: ReviewDiffOut | None = None


# ── 標注專案────────────────

class CampaignCollectionOut(BaseModel):
    id: int
    name: str
    n_items: int
    n_done: int = Field(description="已人工確認（labeled + background）")


class CampaignInitialOut(BaseModel):
    batch_ids: list[int] = Field(description="圖片集裡哪幾個批次是初始人工標注")
    model: dict[str, Any] | None = Field(None, description="初始模型 {name, version}（Registry）")


class BatchSnapshotOut(BaseModel):
    ref: str = Field(description="批次快照 lakefs://…")
    sig: str = Field(description="匯出當下的「張數:revision 加總:最大 id」；標注有變才會重新匯出")
    task_id: str | None = None
    iteration: int | None = None


class CampaignBatchOut(BatchOut):
    initial: bool = Field(description="是不是初始人工標注")
    snapshot: BatchSnapshotOut | None = Field(None, description="最新的批次快照（還沒匯出過 = null）")


class CampaignOut(BaseModel):
    id: int
    name: str
    collection_id: int = Field(description="圖片庫的圖片集：專案的圖都在這（初始批次＋之後每一批）")
    model_name: str
    base_model: str
    golden_ref: str | None = Field(None, description="測試集的 LakeFS 快照")
    golden_collection_id: int | None = Field(None, description="測試集圖片集（只是參照）")
    review_collection: CampaignCollectionOut | None = Field(None, description="圖片集（collection_id）的進度")
    golden_collection: CampaignCollectionOut | None = None
    initial: CampaignInitialOut
    initial_batch_ids: list[int] | None = None
    initial_model: dict[str, Any] | None = None
    recipe: dict[str, Any] | None = Field(None, description='混合配方覆蓋 {"batches": {"<batch_id>": {...}}, "total", "seed"}')
    batch_snapshots: dict[str, BatchSnapshotOut] | None = Field(None, description="每個批次最新的快照")
    config: dict[str, Any] = Field(description="這個專案的覆蓋（只有跟預設不同的欄位）")
    state: Literal["idle", "running", "failed", "closed"] = Field(
        description="idle 閒置（等人按 run / prelabel）/ running 這一輪進行中 / failed 停在 stage 那一步（retry 重跑）/ closed 結案")
    stage: Literal["exporting", "mixing", "training", "prelabeling"] | None = Field(
        None, description="running / failed 時在哪一步；閒置 = null")
    step: dict[str, Any] = Field(description="這一步進行到哪；mode = run（跑下一輪）/ prelabel（只預標）")
    iteration: int = Field(description="下一輪（run）會是第幾代（0 = 初始批次＋初始模型）")
    last_error: str | None = None
    created_by: str
    created_at: datetime
    updated_at: datetime


class CampaignIterationOut(BaseModel):
    campaign_id: int
    n: int
    batch_snapshots: dict[str, Any] | None = Field(None, description="這一代用到的批次快照")
    recipe: dict[str, Any] | None = Field(None, description="這一代的混合配方（已展開成 sources）")
    mix_task_id: str | None = None
    dataset_ref: str | None = Field(None, description="訓練快照（混合的結果）")
    export_task_id: str | None = None
    train_task_id: str | None = None
    model_name: str | None = Field(None, description="NULL = 專案的 model_name")
    model_version: int | None = None
    metrics: dict[str, Any] | None = None
    golden_ref: str | None = Field(None, description="這一代訓練評估用的測試集")
    regression: dict[str, Any] | None = Field(None, description="舊版（v0.2）的退步判斷，只剩歷史；新的一代不再寫")
    prelabel_task_id: str | None = Field(None, description="最近一次用這一代模型預標的任務")
    prelabel_task_ids: list[str] | None = Field(None, description="用這一代模型預標過的所有任務（依序）")
    round_id: int | None = Field(None, description="舊版（v0.2）自動建的審核輪次，只剩歷史")
    note: str | None = None
    created_at: datetime


class CampaignEventOut(BaseModel):
    id: int
    at: datetime
    actor: str
    kind: str
    stage: str | None = None
    detail: dict[str, Any]


class CampaignDetailOut(CampaignOut):
    batches: list[CampaignBatchOut] = Field(description="圖片集的所有批次（含是否初始、最新快照）")
    iterations: list[CampaignIterationOut]
    events: list[CampaignEventOut]
    resolved_config: dict[str, Any] = Field(description="合併後的完整設定（預設 ← 覆蓋）")
    latest_model: dict[str, Any] | None = Field(None, description="最新一代的模型 {iteration, name, version}（只預標用它）")


# ── 影片測試（edgevision/media/video_test_api.py）──────────────────

class VideoSourceOut(BaseModel):
    id: int = Field(description="影片測試的 source_id")
    kind: Literal["upload_video", "nas_recording", "recording_clip"] = Field(
        description="upload_video 上傳的影片 / recording_clip 從 NAS 錄影剪的 / nas_recording 抽圖時用過的 NAS 錄影檔")
    name: str = Field(description="名稱（沒取名 = 依來源自動產生）")
    tags: list[str] = Field(default_factory=list)
    note: str | None = None
    uri: str = Field(description="內部位置（s3:// 或 nas://），播放請用 stream_url")
    stream_url: str = Field(description="播放（支援 Range；加 ?download=1 下載）")
    thumbnail_url: str = Field(description="縮圖（webp，可直接放 <img>；加 ?px= 改長邊像素，預設 320）")
    expires_at: datetime | None = Field(None, description="自動刪除的時間；null = 不會自動刪")
    camera_id: str | None = None
    started_at: datetime | None = Field(None, description="錄影的起始時間（NAS 錄影檔 / 剪下的片段）")
    start_s: float | None = Field(None, description="匯入時取的起點（秒）")
    end_s: float | None = None
    actual_start: str | None = Field(None, description="剪片段不重新編碼，起點對齊到前一個關鍵幀；這是實際起點")
    duration_s: float | None = Field(None, description="存在 MinIO 的影片才有（上傳的、剪下的）")
    width: int | None = None
    height: int | None = None
    size_bytes: int | None = None
    created_by: str
    created_at: datetime


class VideoUsageOut(BaseModel):
    collection_id: int
    collection_name: str
    batch_id: int | None = None
    batch_name: str | None = None
    items: int = Field(description="從這支影片抽出、在這個批次的張數")


class VideoTestRefOut(BaseModel):
    task_id: str
    status: str
    created_at: datetime
    line_id: str | None = None


class VideoDetailOut(VideoSourceOut):
    used_by: list[VideoUsageOut] = Field(default_factory=list, description="從它抽圖的圖片集與批次")
    video_tests: list[VideoTestRefOut] = Field(default_factory=list, description="拿它跑過的影片測試")


class VideoDeletedOut(BaseModel):
    id: int
    items_detached: int = Field(description="從它抽出的圖（都還在，只是不再指向這支影片）")
    file_removed: bool = Field(description="MinIO 上的影片檔沒有別人用，一起刪了")


class NasUsageOut(BaseModel):
    total_gb: float
    free_gb: float
    used_gb: float


class RecordingCameraOut(BaseModel):
    camera_id: str
    files: int
    size_gb: float | None = None
    first: datetime | None = Field(None, description="最早一段的開始時間")
    last: datetime | None = Field(None, description="最新一段的開始時間")
    last_write_age_s: float | None = Field(None, description="最後一次寫入距今幾秒；名單上有、這個數字很大 = 沒在錄")
    days: int | None = None
    recording: bool = Field(False, description="錄影名單上有它")


class RecordingsOut(BaseModel):
    nas: NasUsageOut
    cameras: list[RecordingCameraOut]


class RecordingDayOut(BaseModel):
    date: str
    segments: int
    size_gb: float
    first: datetime
    last: datetime


class RecordingSegmentOut(BaseModel):
    file: str
    start: datetime
    end: datetime | None = None
    duration_s: float | None = None
    size_mb: float
    status: Literal["ok", "writing", "broken"] = Field(description="writing = 還在寫（播不了）；broken = 開不了")
    error: str | None = None
    stream_url: str
    download_url: str


class RecordingGapOut(BaseModel):
    start: datetime
    end: datetime
    seconds: int


class RecordingDayDetailOut(BaseModel):
    camera_id: str
    date: str
    segments: list[RecordingSegmentOut]
    gaps: list[RecordingGapOut]
    covered_s: int = Field(description="這天有錄到的總秒數")
    broken: int
    gap_tolerance_s: float


class ClipCreatedOut(BaseModel):
    task_id: str
    name: str


class VideoUploadCreatedOut(BaseModel):
    task_id: str = Field(description="背景任務；完成後 GET /api/tasks/{id} 的 result.video.video_id")
    name: str = Field(description="影片名稱（沒給 = 原檔名）")


class RosterOut(BaseModel):
    camera_id: str
    line_id: str
    name: str | None = None
    camera_enabled: bool = Field(description="相機本身在設定裡是啟用的（停用的相機就算在名單上也不錄）")
    recording: bool = Field(description="在錄影名單上")
    note: str | None = None
    updated_by: str | None = None
    updated_at: datetime | None = None
    daily_gb: float | None = Field(None, description="最近 24 小時實際寫進 NAS 的 GB（名單上的才算）")


class VideoTestSummaryOut(BaseModel):
    task_id: str
    status: str
    status_message: str | None = None
    created_at: str | None = None
    created_by: str | None = None
    line_id: str | None = None
    model_name: str | None = None
    model_version: int | None = None
    camera_id: str | None = None
    source_id: int | None = None
    start: str | None = Field(None, description="NAS 錄影時間段")
    end: str | None = None
    duration_s: float | None = None
    ng_count: int | None = Field(None, description="線上會發警報 / 存證的次數（站別已做完連續幀數與冷卻）")
    ng_by_reason: dict[str, int] | None = None
    unknown_count: int | None = Field(None, description="無法判定的次數（例如整輪沒看到基準物），不算 NG")
    station: str | None = Field(None, description="換過的判定邏輯（站別名）；null = 用這條線現在的")
    mode: Literal["logic", "model"] = Field("logic", description="logic = 邏輯＋模型（照實際速度、出 NG）；"
                                                                "model = 純模型（不限速、不出 NG，ng_count / unknown_count 為 null）")
    runner_s: float | None = Field(None, description="runner 跑了幾秒（不含排隊與準備）")
    speed: float | None = Field(None, description="幾倍速 = duration_s / runner_s")


class VideoTestNgOut(BaseModel):
    t: float | None = Field(None, description="影片第幾秒")
    frame_id: str | None = None
    station: str | None = Field(None, description="哪個站別判的（主站或附加站，例如 ppe_check）")
    model: str | None = None
    reason: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)


class VideoTestClassOut(BaseModel):
    station: str = Field(description="main = 主站；其他 = 附加站（例如 ppe_check）")
    class_name: str | None = None
    boxes: int = Field(description="框數（所有幀加總）")
    frames: int = Field(description="出現在幾幀")
    mean_conf: float = Field(description="平均信心值")


class VideoTestOut(VideoTestSummaryOut):
    params: dict[str, Any]
    frames: dict[str, int] | None = Field(None, description="各站別判定了幾幀")
    detections_by_class: dict[str, int] | None = None
    class_stats: list[VideoTestClassOut] | None = Field(None, description="各（站別, 類別）統計；兩種模式都有，model 模式看這個")
    models: list[dict[str, Any]] | None = Field(None, description="這次載入的模型與版本（只有主站）")
    video: dict[str, Any] | None = Field(None, description="影片來源")
    ng: list[VideoTestNgOut] = Field(default_factory=list)
    unknown: list[VideoTestNgOut] = Field(default_factory=list, description="無法判定的那幾幀（格式同 ng）")
    video_url: str | None = Field(None, description="打框影片（mp4，2 小時有效）")
    results_url: str | None = Field(None, description="每幀結果（JSONL，2 小時有效）")


# ── 圖片集 / 圖片管理（edgevision/media/collection_ops.py）──────────────

class CollectionDeletedOut(BaseModel):
    id: int
    name: str
    items: int = Field(description="一起刪掉的圖（圖片集裡的關聯與標注；圖片庫的原圖不動）")
    batches: int


class ClassChangeOut(BaseModel):
    class_names: list[str] = Field(description="改完的類別（順序不變）")
    renamed: dict[str, str] = Field(default_factory=dict)
    deleted: list[str] = Field(default_factory=list)
    items_changed: int = Field(description="框有改到的圖（每張留一筆歷史，action = class_change）")
    note: str | None = None


class ItemSearchOut(BaseModel):
    id: int
    collection_id: int
    collection_name: str
    batch_id: int | None = None
    blob_sha256: str
    original_name: str | None = None
    status: Status
    camera_id: str | None = None
    captured_at: datetime | None = None
    tags: list[str] = Field(default_factory=list)
    note: str | None = None
    added_at: datetime


class DownloadOut(BaseModel):
    task_id: str
    status: str
    status_message: str | None = None
    format: Literal["yolo", "yolo-seg", "yolo-pose", "coco", "voc", "dataset"] = Field(description="dataset = LakeFS 資料集版本原樣打包")
    collection_id: int | None = None
    repo: str | None = Field(None, description="format=dataset：資料集 repo")
    version_id: str | None = None
    items: int | None = Field(None, description="圖片數（dataset：檔案數）")
    splits: dict[str, int] | None = None
    size_bytes: int | None = None
    url: str | None = Field(None, description="下載網址（完成才有，2 小時有效）")


class ItemMetaOut(BaseModel):
    id: int
    tags: list[str]
    note: str | None = None


# ── 告警回流（edgevision/media/alerts.py）────────────────────────

class AlertBoxOut(BaseModel):
    cls: str = Field(alias="class")
    conf: float | None = None
    x1: float
    y1: float
    x2: float
    y2: float

    model_config = {"populate_by_name": True}


class AlertFeedbackStateOut(BaseModel):
    status: Literal["queued", "added", "skipped", "failed"]
    reason: str | None = None
    collection_id: int | None = None
    batch_id: int | None = None
    item_id: int | None = None
    item_status: str | None = Field(None, description="放進圖片集時的狀態：labeled / auto / unlabeled / already_in_collection")


class AlertEventOut(BaseModel):
    event_id: int = Field(description="事件 id（detections 的第一列）；判定 / 回流都用它")
    camera_id: str
    frame_id: str
    station: str | None = None
    ts: datetime
    model_name: str | None = None
    model_version: str | None = None
    reason: str | None = Field(None, description="線上判 NG 的原因")
    clip_uri: str | None = None
    clip_url: str | None = Field(None, description="證據影片（打了框，2 小時有效；7 天後過期為 null）")
    boxes: list[AlertBoxOut] | None = Field(None, description="當下的框（原圖像素座標）")
    image_size: dict[str, int] | None = Field(None, description="{width, height}：框的座標基準")
    reviewed: bool = False
    review_result: Literal["true_positive", "false_positive"] | None = None
    reviewed_by: str | None = None
    reviewed_at: datetime | None = None
    feedback: AlertFeedbackStateOut | None = Field(None, description="回流結果；null = 還沒回流")


class AlertStationCountOut(BaseModel):
    station: str | None = None
    total: int = Field(description="這天這站的事件總數")
    listed: int = Field(description="列在清單上的（抽到的＋判過的）")
    reviewed: int


class AlertQueueOut(BaseModel):
    day: str
    per_station: int
    stations: list[AlertStationCountOut]
    events: list[AlertEventOut]


class AlertReviewOut(BaseModel):
    event_id: int
    review_result: Literal["true_positive", "false_positive"] | None = None
    rows: int = Field(description="標了 detections 幾列（一幀多個框 = 多列）")


class AlertFeedbackGroupOut(BaseModel):
    model_name: str | None = None
    events: int
    collection_id: int | None = None
    campaign_id: int | None = None
    campaign_name: str | None = None
    created: bool | None = Field(None, description="這次自動建了「告警回流 <模型>」圖片集")
    batch_id: int | None = None
    task_id: str | None = None
    skipped: str | None = Field(None, description="沒有地方放的原因（這批事件記成 skipped）")


class AlertFeedbackRunOut(BaseModel):
    events: int
    groups: list[AlertFeedbackGroupOut]
    dry_run: bool


class AlertFeedbackOut(BaseModel):
    event_id: int
    camera_id: str
    frame_id: str
    station: str | None = None
    event_ts: datetime
    model_name: str | None = None
    model_version: str | None = None
    review_result: str | None = None
    status: Literal["queued", "added", "skipped", "failed"]
    reason: str | None = None
    collection_id: int | None = None
    batch_id: int | None = None
    item_id: int | None = None
    item_status: str | None = None
    task_id: str | None = None
    align: dict[str, Any] | None = Field(None, description="ok（有把握）、lag（證據影片↔NAS 時間差）、lag_agree、clip_t、score（框線吻合度）、nas_t、ts_offset（那一幀比資料庫時間早晚幾秒）、why")
    created_at: datetime
    updated_at: datetime


ItemOut.model_rebuild()       # review 欄位的型別定義在後面
RoundNextOut.model_rebuild()

__all__ = [n for n in dir() if n.endswith("Out")] + ["Any"]
