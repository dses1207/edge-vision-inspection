"""訓練任務的請求與狀態契約（控制 API ↔ Orchestrator ↔ trainer 共用）。

觸發訓練只有一條路：POST /api/training/jobs。UI、MCP、腳本都是它的客戶端。
請求本身被完整存進 tasks.params —— clone 功能靠這份重建。
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class TaskStatus(str, Enum):
    QUEUED = "queued"        # 已受理、等 GPU；佇列由 Orchestrator 依 TRAIN_MAX_CONCURRENT 放行
    WORKING = "working"
    INPUT_REQUIRED = "input_required"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskType(str, Enum):
    TRAINING = "training"
    EVALUATION = "evaluation"
    EXPORT = "export"
    DEPLOYMENT = "deployment"
    MEDIA_IMPORT = "media_import"     # 圖片庫匯入：zip / 影片 / NAS 錄影（CPU Job）
    AUTOLABEL = "autolabel"           # 自動標注（GPU Job，跟訓練共用排隊）
    MEDIA_EXPORT = "media_export"     # 圖片集 → LakeFS 資料集（CPU Job）
    DATASET_MIX = "dataset_mix"       # 幾個 LakeFS 快照＋配方 → 一個訓練快照（CPU Job，edgevision/training/mix.py）
    VIDEO_TEST = "video_test"         # 影片 ＋ 模型版本 ＋ 線路邏輯 → 打框影片與 NG 清單（GPU，edgevision/media/video_test.py）


class Channel(str, Enum):
    UI = "ui"
    MCP = "mcp"
    API = "api"


class TrainingJobRequest(BaseModel):
    """建訓練任務。dataset_ref / golden_ref 是【已存在】的資料集版本 ID。"""

    model_name: str = Field(pattern=r"^[a-z0-9_\-]+$", description="與 Triton 模型名相同，如 seg_alpha")
    dataset_ref: str = Field(description="lakefs://<repo>/<版本id> 或 local://<容器內路徑>")
    golden_ref: str | None = Field(default=None, description="黃金測試集版本 ID；省略則用 val 當 Gate 依據")
    # ⭐ 平台無狀態：呼叫端把完整設定送進來，平台不需要事先知道有哪些模型。
    #    訓練內容（各模型的參數與門檻）屬於呼叫端（訓練專區）的 repo，不是平台的。
    config: dict[str, Any] | str = Field(
        default_factory=dict,
        description="完整訓練設定（dict，建議）；或平台內建範例設定的路徑（str，僅供測試/示範）。"
                    "兩者皆空則退回 configs/train/<model_name>.yaml（相容舊行為）",
    )
    gates: dict[str, Any] | None = Field(
        default=None,
        description="完整 Gate 規則（dict）。省略則用 config.gates 指的路徑或平台內建範例",
    )
    overrides: dict[str, Any] = Field(default_factory=dict, description="深層覆寫設定，如 {'train': {'epochs': 2}}")
    config_source: str = Field(
        default="",
        description="設定的來源，供血緣追溯，例如 git@training-repo.git@a3f8c91。平台不解析，原樣記錄",
    )
    parent_task_id: str | None = None
    clone_reason: str | None = None
    auto_promote: bool = Field(default=False, description="通過 Gate 後自動移 production tag 並部署")
    notes: str = ""


class PromoteRequest(BaseModel):
    version: int = Field(ge=1)
    reason: str = ""


class TaskView(BaseModel):
    task_id: str
    type: str
    status: str
    status_message: str | None = None
    created_at: str
    started_at: str | None = None
    finished_at: str | None = None
    last_heartbeat: str | None = None
    created_by: str
    channel: str
    params: dict[str, Any] = Field(default_factory=dict)
    result: dict[str, Any] | None = None
    model_version: str | None = None
    run_report_uri: str | None = None
    error: str | None = None
    parent_task_id: str | None = None
    clone_reason: str | None = None
