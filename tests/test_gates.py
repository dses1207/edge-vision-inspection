"""評估 Gate 規則判斷（純邏輯）。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from libs.contracts.run_report import ClassMetrics, FinalMetrics  # noqa: E402
from edgevision.training.evaluate import apply_rules, load_rules  # noqa: E402

RULES = {
    "absolute": {"map50_box": 0.90, "map50_seg": 0.85},
    "no_regression": {"map50_box_delta": -0.01},
    "critical_recall": {"hand": 0.95},
    "latency_ms_max": 50,
    "golden_no_new_miss": True,
}


def _fm(map50=0.95, seg=0.9, hand_recall=0.97, ms=10.0):
    return FinalMetrics(map50_box=map50, map50_95_box=0.7, precision_box=0.9, recall_box=0.9,
                        map50_seg=seg, map50_95_seg=0.6, precision_seg=0.9, recall_seg=0.9,
                        per_class_box={"board": ClassMetrics(precision=1, recall=1, map50=1, map50_95=0.8),
                                       "hand": ClassMetrics(precision=0.9, recall=hand_recall, map50=0.9, map50_95=0.6)},
                        inference_ms=ms)


def test_pass_without_production_baseline():
    r = apply_rules(RULES, _fm())
    assert r.passed, r.summary()
    skipped = {c.name for c in r.checks if c.skipped}
    assert "golden_no_new_miss" in skipped and "vs_production" not in skipped


def test_absolute_fail():
    r = apply_rules(RULES, _fm(map50=0.5))
    assert not r.passed and "absolute.map50_box" in r.summary()


def test_worse_than_production_does_not_block():
    """和現役比只參考（2026-10-14 決定）：分數比現役低也照過，差多少寫進報告。"""
    r = apply_rules(RULES, _fm(map50=0.93), production_metrics={"map50_box": 0.995, "basis": "legacy-val"})
    assert r.passed, r.summary()
    ref = next(c for c in r.checks if c.name == "vs_production")
    assert ref.skipped and "不擋" in ref.detail and "舊驗證集" in ref.detail
    assert r.metrics["vs_production"]["map50_box"]["delta"] == round(0.93 - 0.995, 6)
    assert not any(c.name.startswith("no_regression") for c in r.checks)


def test_critical_recall_and_missing_class():
    assert not apply_rules(RULES, _fm(hand_recall=0.8)).passed
    fm = _fm(); fm.per_class_box.pop("hand")
    r = apply_rules(RULES, fm)
    assert not r.passed and "critical_recall.hand" in r.summary()


def test_latency():
    assert not apply_rules(RULES, _fm(ms=80)).passed


def test_new_miss_on_golden():
    new = [{"image": "a.jpg", "fn": 1, "fp": 0}, {"image": "b.jpg", "fn": 0, "fp": 0}]
    prod = [{"image": "a.jpg", "fn": 0, "fp": 0}, {"image": "b.jpg", "fn": 0, "fp": 0}]
    r = apply_rules(RULES, _fm(), basis="golden", image_metrics=new, production_image_metrics=prod)
    assert not r.passed and "golden_no_new_miss" in r.summary()
    # 現役也漏的圖不算新漏檢
    prod2 = [{"image": "a.jpg", "fn": 2, "fp": 0}, {"image": "b.jpg", "fn": 0, "fp": 0}]
    assert apply_rules(RULES, _fm(), basis="golden", image_metrics=new, production_image_metrics=prod2).passed
    # 非黃金測試集 → 略過
    assert apply_rules(RULES, _fm(), basis="val", image_metrics=new, production_image_metrics=prod).passed


def test_repo_rules_load():
    rules = load_rules(Path(__file__).resolve().parents[1] / "configs/gates/seg_alpha.yaml")
    assert rules["critical_recall"]["hand"] >= 0.9
    assert apply_rules(rules, _fm()).passed
