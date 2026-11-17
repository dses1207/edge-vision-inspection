"""模型 Registry 介面。

Registry 是「哪一版在線上」的唯一真相：
    ・部署 = 把 production tag 移到另一版
    ・程式碼與設定檔永遠不寫版本號
    ・Triton 只認整數版本，Registry 負責把 tag 翻成版本號 + 檔案位置

介面刻意最小：註冊、解析、移 tag、列版本。實作可換
（ClearML 是設計選型；單元測試用記憶體版）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from libs.contracts import ModelMetadata


@dataclass(slots=True)
class ResolvedModel:
    """解析結果：部署 Controller 需要的最小資訊。"""

    model_name: str
    version: int
    source_uri: str          # s3://models/<name>/v<N>/model.plan（或 .onnx）
    meta: ModelMetadata
    registry_id: str = ""    # ClearML model id


@dataclass(slots=True)
class VersionInfo:
    version: int
    tags: list[str] = field(default_factory=list)
    registry_id: str = ""
    created_at: str = ""
    metrics: dict = field(default_factory=dict)


class Registry(Protocol):
    def next_version(self, model_name: str) -> int: ...

    def register(
        self, meta: ModelMetadata, files: dict[str, Path], *, tags: list[str] | None = None
    ) -> ResolvedModel:
        """上傳檔案到 MinIO、把身分登錄進 Registry。files 的 key 對應 ModelArtifacts 欄位。"""
        ...

    def resolve(self, model_name: str, *, tag: str = "production") -> ResolvedModel | None: ...

    def get(self, model_name: str, version: int) -> ResolvedModel | None: ...

    def promote(self, model_name: str, version: int, *, tag: str = "production") -> ResolvedModel:
        """把 tag 移到指定版本（同名 tag 從其他版本移除）。"""
        ...

    def list_versions(self, model_name: str) -> list[VersionInfo]: ...

    def list_models(self) -> list[str]: ...

    def update_artifacts(self, model_name: str, version: int, **uris: str) -> ResolvedModel: ...

    def rename(self, model_name: str, version: int, new_name: str, new_version: int | None = None,
               *, note: str = "") -> ResolvedModel:
        """搬成另一個名字（new_version 不給 = 新名字的下一版）；現役版本不准改。"""
        ...


# ── MinIO 路徑規範─────────────────────────
MODELS_BUCKET = "models"

ARTIFACT_FILENAMES = {
    "pytorch": "best.pt",
    "onnx": "model.onnx",
    "tensorrt": "model.plan",
    "run_report": "run_report.json",
    "eval_report": "eval_report.json",
    "metadata": "metadata.json",
}


def artifact_key(model_name: str, version: int, kind: str) -> str:
    return f"{model_name}/v{version}/{ARTIFACT_FILENAMES[kind]}"


def artifact_uri(model_name: str, version: int, kind: str) -> str:
    return f"s3://{MODELS_BUCKET}/{artifact_key(model_name, version, kind)}"


def serving_uri(meta: ModelMetadata) -> str:
    """部署要載哪個檔：有 TensorRT 用 TensorRT，否則退回 ONNX。"""
    if meta.artifacts.tensorrt:
        return meta.artifacts.tensorrt
    if meta.artifacts.onnx:
        return meta.artifacts.onnx
    raise ValueError(f"{meta.model_name} v{meta.version} 沒有可服務的檔案（onnx / tensorrt）")
