"""訓練設定：YAML 載入、深層覆寫、base_model 解析（不需 GPU）。"""
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from edgevision.training.config import deep_merge, load_train_config, resolve_base_model  # noqa: E402


def test_deep_merge_nested():
    base = {"train": {"epochs": 400, "imgsz": 640}, "export": {"batch": 32}}
    out = deep_merge(base, {"train": {"epochs": 2}})
    assert out["train"] == {"epochs": 2, "imgsz": 640}
    assert out["export"]["batch"] == 32
    assert base["train"]["epochs"] == 400  # 不動原本


def test_load_repo_config_with_overrides():
    cfg = load_train_config(None, model_name="seg_alpha", overrides={"train": {"epochs": 3}},
                            dataset="lakefs://alpha/abc", golden="lakefs://golden/g1")
    assert cfg.model_name == "seg_alpha"
    assert cfg.train["epochs"] == 3 and cfg.train["imgsz"] == 640
    assert cfg.dataset == "lakefs://alpha/abc" and cfg.golden == "lakefs://golden/g1"
    assert cfg.export.batch == 32 and cfg.export.dynamic is True
    assert cfg.gates_path().name == "seg_alpha.yaml"


def test_model_name_mismatch_rejected(tmp_path):
    p = tmp_path / "x.yaml"
    p.write_text(yaml.safe_dump({"model_name": "other", "base_model": "a.pt", "dataset": "local:///d"}))
    with pytest.raises(ValueError):
        load_train_config(p, model_name="seg_alpha")


def test_missing_config():
    with pytest.raises(FileNotFoundError):
        load_train_config("configs/train/nope.yaml", model_name="nope")


def test_resolve_base_model_local_and_weights_dir(tmp_path):
    w = tmp_path / "weights"; w.mkdir()
    (w / "yolo11n-seg.pt").write_bytes(b"x")
    assert resolve_base_model("yolo11n-seg.pt", weights_dir=str(w)) == str(w / "yolo11n-seg.pt")
    local = tmp_path / "my.pt"; local.write_bytes(b"y")
    assert resolve_base_model(str(local)) == str(local)
    # 權重目錄沒有 → 原字串交給 Ultralytics
    assert resolve_base_model("yolo11x-seg.pt", weights_dir=str(w)) == "yolo11x-seg.pt"
    with pytest.raises(ValueError):
        resolve_base_model("registry://seg_alpha/v1", weights_dir=str(w))


# ═══ 平台無狀態：呼叫端送完整設定 ═══════════════════════════

FULL = {
    "model_name": "brand_new_station",       # 平台從沒見過這個模型
    "task": "detect",
    "base_model": "yolo26n.pt",
    "dataset": "lakefs://x/1",
    "class_names": ["a", "b"],
    "train": {"epochs": 50, "imgsz": 960, "batch": 8},
    "export": {"imgsz": 960, "batch": 16, "end2end": True},
}
GATES = {"absolute": {"map50_box": 0.8}, "critical_recall": {"a": 0.9}}


def test_inline_config_needs_no_file_on_platform():
    """⭐ 平台不需要事先有這個模型的 YAML —— 訓練內容屬於呼叫端。"""
    cfg = load_train_config(FULL, model_name="brand_new_station",
                            dataset="lakefs://x/2", gates=GATES,
                            config_source="git@training-repo.git@a3f8c91")
    assert cfg.model_name == "brand_new_station"
    assert cfg.dataset == "lakefs://x/2"          # 請求的 dataset 蓋過設定裡的
    assert cfg.train["epochs"] == 50 and cfg.imgsz == 960
    assert cfg.export.end2end is True and cfg.export.batch == 16
    assert cfg.resolve_gate_rules() == GATES      # 內嵌規則不必讀檔
    assert cfg.config_source.endswith("a3f8c91")  # 血緣記得住設定來源


def test_inline_config_with_overrides():
    cfg = load_train_config(FULL, model_name="brand_new_station",
                            overrides={"train": {"epochs": 2}})
    assert cfg.train["epochs"] == 2 and cfg.train["imgsz"] == 960   # 只蓋到指定的鍵
    assert FULL["train"]["epochs"] == 50                            # 不動呼叫端的原始 dict


def test_inline_gate_rules_beat_file():
    """內嵌規則優先於平台自帶的範例檔。"""
    cfg = load_train_config(None, model_name="seg_alpha", gates={"absolute": {"map50_box": 0.5}})
    assert cfg.resolve_gate_rules()["absolute"]["map50_box"] == 0.5


def test_path_config_still_works_for_platform_samples():
    """平台自帶範例（示範與整合測試）仍可用路徑。"""
    cfg = load_train_config("configs/train/seg_alpha.yaml", model_name="seg_alpha")
    assert cfg.model_name == "seg_alpha"
    assert cfg.resolve_gate_rules()["critical_recall"]["hand"] >= 0.9   # 退回讀檔案


def test_config_without_anything_is_rejected():
    with pytest.raises(ValueError):
        load_train_config(None)
