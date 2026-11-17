"""nvtracker 相關：設定檔內容、以及 tracker 與 BoxCache 的互斥。

不需要 GPU —— 測的是「參數對不對」和「誰負責補跳幀的框」，不是實際追蹤效果。
"""
from pathlib import Path

import pytest
import yaml

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "tracker_nvsort.yml"


@pytest.fixture(scope="module")
def cfg():
    # OpenCV 風格的 %YAML:1.0 指示行 PyYAML 不認，拿掉再解
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8").replace("%YAML:1.0", "", 1))


def test_kalman_state_estimator_is_on(cfg):
    """stateEstimatorType=2（REGULAR）才有 Kalman —— 這是跳幀時框會【移動】而非凍住的唯一原因。
    設成 0（DUMMY）等於退化成 IOU tracker，加了跟沒加一樣。"""
    assert cfg["StateEstimator"]["stateEstimatorType"] == 2


def test_shadow_tracking_outlives_any_reasonable_interval(cfg):
    """shadow tracking 壽命必須 ≥ 跳幀間隔，否則物件在兩次推論之間就被判定終止 → 框消失。
    官方預設 26 幀；我們放寬到能撐過 interval=10 還有餘裕。"""
    assert cfg["TargetManagement"]["maxShadowTrackingAge"] >= 60


def test_probation_short_enough_for_skipped_frames(cfg):
    """新目標要熬過 probationAge 幀才輸出。跳幀時「幀」變稀疏，
    官方預設 5 會讓手伸進畫面後等半秒才出現框。"""
    assert cfg["TargetManagement"]["probationAge"] <= 2


def test_detector_confidence_gate_does_not_eat_our_boxes(cfg):
    """框在 probe 的後處理就用 conf 濾過了；tracker 這一關再濾一次只會吃掉已確定要畫的框。"""
    assert cfg["BaseConfig"]["minDetectorConfidence"] <= 0.05


def test_tracker_replaces_boxcache_not_stacks_with_it():
    """⚠️ 這是最容易搞錯、且症狀是「畫面出現兩份框」的地方。

    跳幀的幀只能有一個東西補框：
      · 沒有 tracker → BoxCache 原地重畫上一次的框
      · 有 tracker   → tracker 用 Kalman 外推，probe 必須【什麼都不做】
    pipeline.py 用 fill_skipped=not tracker_on 保證這個互斥。
    """
    src = (Path(__file__).resolve().parents[1] / "src" / "ds_runner" / "pipeline.py").read_text(encoding="utf-8")
    assert "fill_skipped=not tracker_on" in src


def test_tracker_sits_before_demux():
    """tracker 要的是 batched NvDsBatchMeta，nvstreamdemux 之後就沒有了。
    接線必須是 infer → tracker → out_el（out_el 才是 demux）。"""
    src = (Path(__file__).resolve().parents[1] / "src" / "ds_runner" / "pipeline.py").read_text(encoding="utf-8")
    assert "infer.link(tracker)" in src and "tracker.link(out_el)" in src
