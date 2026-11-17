"""標注形狀：驗證、YOLO 互轉、自動標注的類別合併、匯出的 train/val 切分。"""
import pytest

from edgevision.media.export import split_of
from edgevision.media.shapes import from_yolo_lines, merge_by_class, to_yolo_lines, validate_shapes

NAMES = ["board", "hand"]


def test_rect_roundtrip_detect():
    shapes = from_yolo_lines(["1 0.5 0.5 0.2 0.4"], NAMES, "detect")
    assert shapes[0]["type"] == "rect" and shapes[0]["class"] == "hand"
    assert sum(shapes[0]["points"], []) == pytest.approx([0.4, 0.3, 0.6, 0.7])
    assert to_yolo_lines(shapes, NAMES, "detect") == ["1 0.500000 0.500000 0.200000 0.400000"]


def test_polygon_to_detect_uses_bbox_and_rect_to_segment_uses_corners():
    poly = [{"type": "polygon", "class": "board", "points": [[0.1, 0.1], [0.5, 0.1], [0.3, 0.6]]}]
    assert to_yolo_lines(poly, NAMES, "detect") == ["0 0.300000 0.350000 0.400000 0.500000"]
    rect = [{"type": "rect", "class": "hand", "points": [[0.1, 0.2], [0.3, 0.4]]}]
    assert to_yolo_lines(rect, NAMES, "segment") == ["1 0.100000 0.200000 0.300000 0.200000 0.300000 0.400000 0.100000 0.400000"]


def test_pose_roundtrip():
    line = "0 0.5 0.5 0.4 0.4 0.4 0.4 2 0.6 0.4 2 0.6 0.6 1 0.4 0.6 0"
    shapes = from_yolo_lines([line], ["box"], "pose", kpt_shape=[4, 3])
    assert shapes[0]["type"] == "keypoints" and len(shapes[0]["points"]) == 4
    out = to_yolo_lines(shapes, ["box"], "pose")[0].split()
    assert [float(v) for v in out] == pytest.approx([float(v) for v in line.split()])


def test_validate_rejects_unknown_class_and_bad_coords():
    with pytest.raises(ValueError, match="不在圖片集類別"):
        validate_shapes([{"type": "rect", "class": "cat", "points": [[0, 0], [1, 1]]}], NAMES)
    with pytest.raises(ValueError):
        validate_shapes([{"type": "rect", "class": "hand", "points": [[0, 0], [640, 480]]}], NAMES)   # 像素座標
    with pytest.raises(ValueError, match="至少 3 點"):
        validate_shapes([{"type": "polygon", "class": "hand", "points": [[0, 0], [1, 1]]}], NAMES)
    ok = validate_shapes([{"type": "rect", "class": "hand", "points": [[0, 0], [1, 1]]}], NAMES)
    assert ok[0]["source"] == {"kind": "human"}


def test_merge_by_class_only_replaces_filtered_classes():
    old = [{"class": "board", "n": 1}, {"class": "hand", "n": 2}]
    new = [{"class": "board", "n": 3}, {"class": "hand", "n": 4}]
    assert merge_by_class(old, new, ["board"]) == [{"class": "hand", "n": 2}, {"class": "board", "n": 3}]
    assert merge_by_class(old, new, None) == new


def test_split_is_deterministic_by_content():
    sha = "0" * 64
    assert split_of(sha, 20) == "val" and split_of("f" * 64, 20) == "train"
    assert split_of(sha, 20) == split_of(sha, 20)
