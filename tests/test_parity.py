"""轉檔一致性比對邏輯。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from edgevision.training.parity import compare, iou  # noqa: E402


def d(cls, conf, box):
    return {"class_id": cls, "conf": conf, "bbox": box}


def test_iou():
    assert iou([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0
    assert iou([0, 0, 10, 10], [10, 10, 20, 20]) == 0.0
    assert abs(iou([0, 0, 10, 10], [5, 0, 15, 10]) - 1 / 3) < 1e-6


def test_identical_passes():
    ref = {"a.jpg": [d(0, 0.9, [0, 0, 100, 100]), d(1, 0.8, [200, 200, 240, 260])]}
    r = compare(ref, ref)
    assert r.passed and r.match_ratio == 1.0 and r.max_conf_diff == 0.0


def test_small_shift_and_conf_within_tolerance():
    ref = {"a.jpg": [d(0, 0.90, [0, 0, 100, 100])]}
    cand = {"a.jpg": [d(0, 0.88, [2, 1, 101, 100])]}
    assert compare(ref, cand).passed


def test_missing_detection_fails():
    ref = {"a.jpg": [d(0, 0.9, [0, 0, 100, 100]), d(1, 0.8, [200, 200, 240, 260])]}
    cand = {"a.jpg": [d(0, 0.9, [0, 0, 100, 100])]}
    r = compare(ref, cand)
    assert not r.passed and "配對比例" in r.reason


def test_class_mismatch_fails():
    ref = {"a.jpg": [d(0, 0.9, [0, 0, 100, 100])]}
    cand = {"a.jpg": [d(1, 0.9, [0, 0, 100, 100])]}
    assert not compare(ref, cand).passed


def test_conf_drift_fails():
    ref = {"a.jpg": [d(0, 0.9, [0, 0, 100, 100])]}
    cand = {"a.jpg": [d(0, 0.7, [0, 0, 100, 100])]}
    r = compare(ref, cand)
    assert not r.passed and "信心度差" in r.reason


def test_both_empty_passes_with_note():
    r = compare({"a.jpg": []}, {"a.jpg": []})
    assert r.passed and "無偵測" in r.reason
