"""模型身分契約。

一個模型版本的完整身分：它是誰、從哪來、多好、檔案在哪。
metadata 存 ClearML Registry，檔案存 MinIO，這份 schema 是兩邊共同的語言。

⚠️ lineage 是必填 —— 沒有血緣的模型不准註冊（唯一例外：既有模型遷移，
   標為 legacy-unknown）。這是「新訓練的模型必須可追溯」的強制點。
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

META_SCHEMA_VERSION = "1.0"
LEGACY_UNKNOWN = "legacy-unknown"

# Registry 的 tag 語意。部署 = 移動 production tag。
TAG_PRODUCTION = "production"
TAG_CANDIDATE = "candidate"     # 通過評估 Gate、已註冊、等人放行
TAG_REJECTED = "rejected"       # 轉檔或一致性驗證失敗，留紀錄不上線


class Lineage(BaseModel):
    """血緣：回答「這個模型用什麼資料、哪份程式碼、從哪個模型訓出來的」。"""

    dataset_version: str = Field(description="lakefs://<repo>/<版本id>，或 legacy-unknown")
    golden_version: str | None = None
    code_commit: str = Field(description="訓練程式碼的 git commit，或 legacy-unknown")
    base_model: str = Field(description="起手權重：yolo11m-seg.pt 或 registry://name/vN")
    training_run_id: str | None = Field(default=None, description="ClearML task id")
    task_id: str | None = Field(default=None, description="tasks 表的 task_id")
    parent_task_id: str | None = None
    clone_reason: str | None = None

    @classmethod
    def legacy(cls, base_model: str = LEGACY_UNKNOWN) -> "Lineage":
        return cls(dataset_version=LEGACY_UNKNOWN, code_commit=LEGACY_UNKNOWN, base_model=base_model)


class ModelArtifacts(BaseModel):
    """各格式檔案在 MinIO 的位置。s3://models/<name>/v<N>/..."""

    pytorch: str | None = None
    onnx: str | None = None
    tensorrt: str | None = None
    run_report: str | None = None
    eval_report: str | None = None
    metadata: str | None = None


class ExportSpec(BaseModel):
    """轉檔規格。⚠️ batch 必須 >= Triton config 的 max_batch_size，否則載不進去。"""

    imgsz: int = 640
    dynamic: bool = True
    batch: int = 32
    opset: int = 12
    fp16: bool = False
    # YOLO26 這類 NMS-free 模型：ONNX 直接輸出 (batch, max_det, 6)=[x1,y1,x2,y2,conf,cls]，不需後處理 NMS
    end2end: bool = False
    max_det: int = 300
    # ⭐ seg 模型：NMS + mask 合成都放進 ONNX，Triton 直接輸出每個偵測的 mask。
    #   輸出變成 output0 (B, max_det, 4+2+32) + masks (B, max_det, imgsz/4, imgsz/4)，沒有 output1。
    #   為什麼：prototypes 每幀 3.28 MB，7 路就要搬 463 MB/s，Triton 的 CPU 有 71~78% 在搬它
    #   。合成進模型後傳輸省 61%。
    #   ⚠️ 開這個時 max_det 要小（16）：每偵測一張 160×160，300 列是 30 MB/幀。
    fuse_masks: bool = False
    # pose 模型：每個 instance 的 keypoint 數（每點 x,y,conf）。output0 = 4 + nc + n_kpt×3，config.pbtxt 要靠它算維度
    n_kpt: int = 0


class ModelMetadata(BaseModel):
    schema_version: Literal["1.0"] = META_SCHEMA_VERSION

    model_name: str = Field(description="Registry 與 Triton 共用的名稱，如 seg_alpha")
    version: int = Field(ge=1, description="整數版本，Triton 版本目錄同號")
    task: Literal["segment", "detect", "pose"] = "segment"
    class_names: list[str]
    imgsz: int = 640
    created_at: str
    created_by: str = "system"

    lineage: Lineage
    hyperparameters: dict[str, Any] = Field(default_factory=dict)
    metrics: dict[str, Any] = Field(default_factory=dict)
    artifacts: ModelArtifacts = Field(default_factory=ModelArtifacts)
    export: ExportSpec = Field(default_factory=ExportSpec)
    notes: str = ""

    @property
    def triton_max_batch(self) -> int:
        return self.export.batch if self.export.dynamic else 0
