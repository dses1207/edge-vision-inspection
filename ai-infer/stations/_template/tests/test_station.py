import pytest
from edgevision_contracts import BBox, Detection, VerdictStatus
from edgevision_station_sdk import MissingCameraParams, load_station


def det(name):
    return Detection(class_id=0, class_name=name, confidence=0.9, bbox=BBox(x1=0, y1=0, x2=10, y2=10))


def test_ng_when_count_reached_for_persist_frames():
    s = load_station("template", {"target_class": "hand", "max_count": 2, "persist_frames": 2})
    assert s.evaluate([det("hand"), det("hand")], 1280, 720).status == VerdictStatus.OK
    assert s.evaluate([det("hand"), det("hand")], 1280, 720).status == VerdictStatus.NG
    assert s.evaluate([det("hand")], 1280, 720).status == VerdictStatus.OK


def test_missing_required_param_fails_at_load():
    with pytest.raises(MissingCameraParams):
        load_station("template", {"target_class": "hand"})
