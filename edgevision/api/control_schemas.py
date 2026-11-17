"""控制 API 的回應模型 —— 只為了 OpenAPI 規格完整，讓前端能 codegen。

⚠️ 這裡的模型是「描述」不是「驗證」：control.py 的端點照舊回 dict（responses={200: {...}}），
   FastAPI 不會用這些模型過濾或改寫回應。改了端點實際回的欄位，這裡要跟著改 ——

以前 16 支 2xx 全是空 schema，前端只能手寫型別，踩過兩次型別錯
（marker_id 當 string、相機 state 當自由字串）—— 前端回饋。
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from libs.contracts import Lineage, ModelArtifacts, TaskView  # noqa: F401 —— TaskView 直接當回應模型


class ControlHealthOut(BaseModel):
    """GET /healthz（控制 API；叢集外打 /healthz 會拿到推論 API 的，這支要從叢集內打）。"""

    ok: bool
    db: bool
    minio: bool
    auth_enabled: bool
    auth_allow_header_actor: bool
    # 以下來自 Orchestrator.health()
    running_tasks: list[str] = Field(default_factory=list)
    runner_ok: bool
    max_concurrent: int
    keep_workspaces: bool
    queued: int


class TrainingJobCreatedOut(BaseModel):
    """POST /api/training/jobs（201）。一律先進 queued，由 Orchestrator 決定馬上跑或排隊。"""

    task_id: str
    status: str
    # 在佇列中的位置（1 = 下一個）；已經在跑或查不到時為 null
    queue_position: int | None = None


class CancelOut(BaseModel):
    """POST /api/tasks/{id}/cancel。取消是非同步的：之後用 GET /api/tasks/{id} 看最終狀態。"""

    task_id: str
    cancel_requested: bool


class DatasetVersionOut(BaseModel):
    """GET /api/datasets 的 versions[] 單筆（main 分支最近 10 個版本，新到舊）。"""

    # 訓練請求的 dataset_ref 就填這個：lakefs://<repo>/<完整版本id>
    dataset_ref: str
    id: str
    message: str
    created: int = Field(description="epoch 秒")
    metadata: dict[str, Any] = Field(default_factory=dict)


class DatasetRepoOut(BaseModel):
    repo: str
    versions: list[DatasetVersionOut] = Field(default_factory=list)
    # 這個 repo 讀不到時的原因（其他 repo 照常列出）
    error: str | None = None


class DatasetVersionPageOut(BaseModel):
    """GET /api/datasets/{repo}/versions：分頁的版本清單（新到舊）。"""
    repo: str
    versions: list[DatasetVersionOut]
    next: str | None = Field(None, description="下一頁的 after（null = 沒有了）")


class DatasetChangeOut(BaseModel):
    type: str = Field(description="added / removed / changed")
    path: str
    size_bytes: int | None = None


class DatasetDiffOut(BaseModel):
    repo: str
    from_ref: str
    to_ref: str
    summary: dict[str, int] = Field(description="added / removed / changed 各幾個檔")
    by_dir: dict[str, dict[str, int]] = Field(description="依第一、二層目錄（例 images/train）分的 added / removed / changed")
    images: dict[str, int] = Field(description="只算圖片檔：added / removed / changed")
    changes: list[DatasetChangeOut] = Field(description="前 limit 筆變動")
    truncated: bool


class DatasetDeletedOut(BaseModel):
    repo: str
    deleted: bool


class UploadUrlOut(BaseModel):
    """POST /api/datasets/upload-url：拿一個臨時 PUT 網址，把資料集 zip 直傳 MinIO。

    上傳完拿 source_uri 去呼叫 POST /api/datasets/ingest-uri。
    """

    upload_url: str = Field(description="curl -T <zip> '<upload_url>' 直傳")
    source_uri: str = Field(description="上傳完成後，ingest-uri 要填的 s3:// 位址")
    expires_minutes: int


class IngestFromUriIn(BaseModel):
    """POST /api/datasets/ingest-uri：伺服器端從物件儲存讀 zip 來 ingest（取代 multipart 上傳）。"""

    source_uri: str = Field(description="s3://bucket/key，通常來自 upload-url 的 source_uri")
    repo: str
    message: str = "上傳"
    all_val: bool = Field(False, description="true = 整份當驗證集（黃金測試集用）")
    classes: list[str] | None = Field(None, description="類別名稱；省略則讀 zip 內的 data.yaml")


class MixSourceIn(BaseModel):
    """混合的一個來源。repeat / choose / proportion 三選一；都不給 = repeat 1（全部取一次）。"""

    ref: str = Field(description="lakefs://<repo>/<版本id>")
    repeat: float | None = Field(None, ge=0, description="倍數；<1 降採樣、>1 升採樣")
    choose: int | None = Field(None, ge=0, description="固定張數（可大於來源張數）")
    proportion: float | None = Field(None, ge=0, description="相對比例（要另給 total；每個來源都要用比例）")


class MixIn(BaseModel):
    """POST /api/datasets/mix：幾個快照＋配方 → 一個新的訓練快照。"""

    repo: str = Field(description="輸出到哪個 LakeFS repo：小寫英數與 -，3–63 字（不能有底線、大寫）")
    message: str = Field(min_length=1, description="版本說明（用途、為什麼這樣配）")
    sources: list[MixSourceIn] = Field(min_length=1)
    total: int | None = Field(None, ge=1, description="用 proportion 時的總張數")
    seed: int = Field(0, description="抽樣亂數種子（結果寫死在快照裡）")


class MixCreatedOut(BaseModel):
    task_id: str = Field(description="背景任務；完成後 GET /api/tasks/{id} 的 result.mix.dataset_ref 就是訓練快照")


class IngestJobOut(BaseModel):
    """POST /api/datasets/ingest（202）與 GET /api/datasets/ingest/{job_id}。

    上傳後輪詢 GET 直到 status 是 done / failed。done 之後的 dataset_ref 可直接拿去開訓練。
    """

    job_id: str
    # downloading 只出現在 ingest-uri 路徑（伺服器端先從 MinIO 拉 zip）
    status: Literal["queued", "downloading", "extracting", "uploading", "done", "failed"]
    repo: str
    message: str | None = None
    n_images: int | None = None
    # status=done 才有
    dataset_ref: str | None = None
    files: int | None = None
    n_train: int | None = None
    n_val: int | None = None
    class_names: list[str] | None = None
    class_distribution: dict[str, int] | None = None
    # status=failed 才有
    error: str | None = None


class TrainConfigOut(BaseModel):
    """GET /api/configs 的單筆：configs/train/*.yaml 的摘要，給前端下拉選單。"""

    model_name: str
    path: str
    task: str | None = None
    base_model: str | None = None
    class_names: list[str] | None = None
    epochs: int | None = None
    dataset: str | None = None
    golden: str | None = None
    # 這個檔解析失敗時的原因（其他檔照常列出）
    error: str | None = None


class ModelSummaryOut(BaseModel):
    """GET /api/models 的單筆。"""

    model_name: str
    # 目前掛 production 標的版本；還沒有人 promote 過為 null
    production: int | None = None
    versions: list[int] = Field(default_factory=list)


class ModelVersionOut(BaseModel):
    """GET /api/models/{name}/versions 的單筆。"""

    version: int
    tags: list[str] = Field(default_factory=list)
    created_at: str
    metrics: dict[str, Any] = Field(default_factory=dict)
    registry_id: str | None = None


class ModelVersionDetailOut(BaseModel):
    """GET /api/models/{name}/versions/{version}。lineage=legacy-unknown 代表匯入的舊模型。"""

    model_name: str
    version: int
    registry_id: str | None = None
    source_uri: str | None = None
    created_at: str
    lineage: Lineage
    metrics: dict[str, Any] = Field(default_factory=dict)
    artifacts: ModelArtifacts
    class_names: list[str] = Field(default_factory=list)
    task: str | None = None
    imgsz: int | None = None


class DeployOut(BaseModel):
    """POST /api/models/{name}/promote 與 /rollback。

    部署是非同步的（建 engine、Triton 載入）：拿 task_id 去 GET /api/tasks/{id} 看進度。
    """

    task_id: str
    model_name: str
    version: int
    status: str


class RenameModelIn(BaseModel):
    """POST /api/models/{name}/versions/{version}/rename。"""

    new_name: str = Field(min_length=1, pattern=r"^[a-z0-9][a-z0-9_]*$")
    new_version: int | None = Field(default=None, ge=1, description="不給 = 新名字的下一版")
    reason: str = Field(min_length=1, description="為什麼改名（進 audit_log 與模型備註）")
