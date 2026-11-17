"""訓練報告契約 `run_report.json`。

存在的理由：AI agent 要能診斷訓練。散落的 log 與圖片沒辦法診斷，
機器可讀、格式穩定的報告才行。這份 schema 一旦定了就版本化，
改欄位要升 schema_version。

`worst_predictions` 是診斷的關鍵素材 —— 沒有它 agent 無法判斷
「模型錯在哪類情況」。
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

REPORT_SCHEMA_VERSION = "1.0"


class Curves(BaseModel):
    """逐 epoch 曲線。鍵名固定，值為每 epoch 一點。"""

    epoch: list[int] = Field(default_factory=list)
    train_box_loss: list[float] = Field(default_factory=list)
    train_seg_loss: list[float] = Field(default_factory=list)
    train_cls_loss: list[float] = Field(default_factory=list)
    val_box_loss: list[float] = Field(default_factory=list)
    val_seg_loss: list[float] = Field(default_factory=list)
    val_cls_loss: list[float] = Field(default_factory=list)
    map50_box: list[float] = Field(default_factory=list)
    map50_95_box: list[float] = Field(default_factory=list)
    map50_seg: list[float] = Field(default_factory=list)
    map50_95_seg: list[float] = Field(default_factory=list)
    lr: list[float] = Field(default_factory=list)


class ClassMetrics(BaseModel):
    precision: float
    recall: float
    map50: float
    map50_95: float


class FinalMetrics(BaseModel):
    """box 與 seg 各一套；seg 模型兩套都有，detect 模型只有 box。"""

    map50_box: float
    map50_95_box: float
    precision_box: float
    recall_box: float
    map50_seg: float | None = None
    map50_95_seg: float | None = None
    precision_seg: float | None = None
    recall_seg: float | None = None
    per_class_box: dict[str, ClassMetrics] = Field(default_factory=dict)
    per_class_seg: dict[str, ClassMetrics] = Field(default_factory=dict)
    fitness: float | None = None
    best_epoch: int | None = None
    epochs_run: int | None = None
    inference_ms: float | None = Field(default=None, description="val 時每張推論毫秒（PyTorch）")


class WorstPrediction(BaseModel):
    image: str
    precision: float
    recall: float
    f1: float
    tp: int
    fp: int
    fn: int
    error_type: Literal["miss", "false_alarm", "mixed"]


class DatasetStats(BaseModel):
    dataset_version: str
    n_images: int
    n_train: int
    n_val: int
    class_distribution: dict[str, int] = Field(default_factory=dict)
    bbox_size_distribution: dict[str, Any] = Field(default_factory=dict)
    images_per_source: dict[str, int] = Field(default_factory=dict)


class RunReport(BaseModel):
    schema_version: Literal["1.0"] = REPORT_SCHEMA_VERSION

    run_id: str = Field(description="ClearML task id")
    task_id: str | None = Field(default=None, description="tasks 表 task_id")
    model_name: str
    dataset_version: str
    golden_version: str | None = None
    base_model: str
    code_commit: str
    config: dict[str, Any] = Field(description="完整訓練參數（連結進 ClearML 的那份）")

    curves: Curves = Field(default_factory=Curves)
    final_metrics: FinalMetrics
    golden_metrics: FinalMetrics | None = None
    confusion_matrix: list[list[int]] = Field(default_factory=list)
    confusion_labels: list[str] = Field(default_factory=list)
    worst_predictions: list[WorstPrediction] = Field(default_factory=list)
    dataset_stats: DatasetStats

    started_at: str
    finished_at: str
    duration_s: float
    notes: list[str] = Field(default_factory=list)
