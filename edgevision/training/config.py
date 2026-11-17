"""訓練設定：全參數化，禁止程式內 hardcode 訓練參數。

⭐ 定位（2026-09-10 釐清）：**本專案是「提供訓練能力的平台」，不是「訓練內容的家」。**
各模型的參數與 Gate 門檻屬於呼叫端（訓練專區）的 repo，由請求送進來：

    請求的 config（完整設定 dict）      ← 正式做法，平台無狀態
    + 請求的 overrides（深層合併）      這次要臨時改的
    + 請求的 gates（Gate 規則 dict）
    = 送進 model.train(**train) 的那份，也是連結進 ClearML 的那份

`configs/` 底下的 YAML 只是**平台自帶的範例**（示範與整合測試用），
不是生產設定的存放處。血緣用 config_source 記錄設定來自哪個 repo 的哪個 commit。

`train` 區塊原樣透傳給 Ultralytics（鍵名就是 Ultralytics 的參數名）。
"""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from libs.contracts.model_meta import ExportSpec

REPO_ROOT = Path(__file__).resolve().parents[2]


class LineageInput(BaseModel):
    parent_task_id: str | None = None
    clone_reason: str | None = None


class TrainConfig(BaseModel):
    model_name: str
    task: str = "segment"
    base_model: str = Field(description="yolo11m-seg.pt（權重目錄內）或 registry://<name>/v<N> 或 s3://...")
    dataset: str = Field(description="lakefs://<repo>/<版本id> 或 local://<路徑>")
    golden: str | None = None
    class_names: list[str] | None = Field(default=None, description="省略則以資料集 data.yaml 為準")
    train: dict[str, Any] = Field(default_factory=dict, description="原樣透傳 model.train(**train)")
    export: ExportSpec = Field(default_factory=ExportSpec)
    gates: str = Field(default="", description="Gate 規則 YAML 路徑；空則 configs/gates/<model_name>.yaml")
    gate_rules: dict[str, Any] | None = Field(
        default=None, description="內嵌的 Gate 規則（呼叫端直接送進來）。有這個就不讀檔案")
    config_source: str = Field(default="", description="設定來源（血緣用），平台原樣記錄")
    parity: dict[str, Any] = Field(default_factory=lambda: {
        "images": 8, "iou_match": 0.8, "min_match_ratio": 0.9, "conf_tolerance": 0.05,
    })
    lineage: LineageInput = Field(default_factory=LineageInput)
    clearml_project: str = Field(default_factory=lambda: os.environ.get("CLEARML_PROJECT", "edgevision"))

    @property
    def imgsz(self) -> int:
        return int(self.train.get("imgsz", 640))

    def gates_path(self) -> Path:
        p = self.gates or f"configs/gates/{self.model_name}.yaml"
        return resolve_path(p)

    def resolve_gate_rules(self) -> dict:
        """Gate 規則：內嵌的優先，其次讀檔案。"""
        if self.gate_rules:
            return self.gate_rules
        from edgevision.training.evaluate import load_rules
        return load_rules(self.gates_path())


def resolve_path(p: str | Path) -> Path:
    p = Path(p)
    if p.is_absolute():
        return p
    # 容器內 repo 掛在 /app，開發機是 REPO_ROOT
    for root in (REPO_ROOT, Path("/app"), Path.cwd()):
        if (root / p).exists():
            return root / p
    return REPO_ROOT / p


def deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_train_config(
    config: dict | str | Path | None, *, model_name: str | None = None,
    overrides: dict | None = None, dataset: str | None = None, golden: str | None = None,
    gates: dict | None = None, config_source: str = "",
) -> TrainConfig:
    """組出這次要用的訓練設定。

    `config` 三種形態（優先序由上而下）：
        dict          呼叫端送完整設定 —— ⭐ 正式做法。平台無狀態，不需要知道有哪些模型。
        str / Path    平台內建的範例設定路徑（示範與整合測試用）
        None          退回 configs/train/<model_name>.yaml（相容舊行為）

    請求裡給的 dataset / golden / gates 優先於設定內容。
    """
    if isinstance(config, dict) and config:
        raw = copy.deepcopy(config)
    else:
        path = config or (f"configs/train/{model_name}.yaml" if model_name else None)
        if not path:
            raise ValueError("需要完整設定（dict）、設定路徑，或 model_name")
        p = resolve_path(path)
        if not p.exists():
            raise FileNotFoundError(f"找不到訓練設定：{p}")
        raw = yaml.safe_load(p.read_text()) or {}
    raw = deep_merge(raw, overrides or {})
    if gates:
        raw["gate_rules"] = gates
    if config_source:
        raw["config_source"] = config_source
    if dataset:
        raw["dataset"] = dataset
    if golden is not None:
        raw["golden"] = golden
    if model_name and raw.get("model_name") not in (None, model_name):
        raise ValueError(f"請求 model_name={model_name} 與設定檔 {raw.get('model_name')} 不一致")
    raw.setdefault("model_name", model_name)
    return TrainConfig.model_validate(raw)


def resolve_base_model(base: str, *, weights_dir: str | None = None, objects=None, work_dir: Path | None = None,
                       registry=None) -> str:
    """把 base_model 字串變成本地檔案路徑。

    ・純檔名（yolo11m-seg.pt）→ 權重目錄（WEIGHTS_DIR，映像建置時預載）
    ・registry://name/vN      → 從 Registry 解析 pytorch artifact 下載
    ・s3://...                → 直接下載
    ・存在的本地路徑          → 原樣
    """
    if Path(base).exists():
        return base
    wd = Path(weights_dir or os.environ.get("WEIGHTS_DIR", "/weights"))
    if not base.startswith(("registry://", "s3://")):
        cand = wd / base
        if cand.exists():
            return str(cand)
        # 交給 Ultralytics 自行下載（需外網；映像應預載避免這條路）
        return base

    if base.startswith("registry://"):
        if registry is None:
            raise ValueError("base_model 用 registry:// 需要 Registry")
        name, _, ver = base[len("registry://"):].partition("/v")
        resolved = registry.get(name, int(ver))
        if resolved is None or not resolved.meta.artifacts.pytorch:
            raise ValueError(f"{base} 找不到 pytorch artifact")
        uri = resolved.meta.artifacts.pytorch
    else:
        uri = base

    if objects is None:
        raise ValueError("下載 base_model 需要 ObjectStore")
    bucket, key = objects.parse_uri(uri)
    target = (work_dir or Path("/tmp")) / "base_model" / Path(key).name
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        target.write_bytes(objects.get(bucket=bucket, key=key))
    return str(target)
