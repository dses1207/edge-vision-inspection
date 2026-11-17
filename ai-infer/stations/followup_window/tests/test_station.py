import pytest
from edgevision_contracts import BBox, Detection, VerdictStatus
from edgevision_station_sdk import MissingCameraParams, load_station


def det(name):
    return Detection(class_id=0, class_name=name, confidence=0.9, bbox=BBox(x1=0, y1=0, x2=10, y2=10))


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


@pytest.fixture
def make(monkeypatch):
    from edgevision_station_followup_window import FollowupWindowStation
    clock = FakeClock()
    monkeypatch.setattr(FollowupWindowStation, "clock", clock)

    def _make(**th):
        return load_station("followup_window", {"trigger_class": "forklift", "required_class": "spotter",
                                                "thresholds": {"seconds": 10, **th}}), clock
    return _make


def test_timeout_is_ng_even_without_boxes(make):
    s, clock = make()
    assert s.evaluate([det("forklift")], 1280, 720).status == VerdictStatus.OK
    clock.t = 5
    assert s.evaluate([], 1280, 720).details["state"] == "waiting"
    clock.t = 11
    v = s.evaluate([], 1280, 720)
    assert v.status == VerdictStatus.NG and "spotter" in v.reason


def test_satisfied_in_time_is_ok(make):
    s, clock = make()
    s.evaluate([det("forklift")], 1280, 720)
    clock.t = 3
    assert s.evaluate([det("spotter")], 1280, 720).details["state"] == "satisfied"
    clock.t = 20
    assert s.evaluate([], 1280, 720).status == VerdictStatus.OK


def test_late_arrival_counts_only_when_late_ok(make):
    strict, clock = make()
    strict.evaluate([det("forklift")], 1280, 720)
    clock.t = 12
    assert strict.evaluate([det("spotter")], 1280, 720).status == VerdictStatus.NG

    lenient, clock = make(late_ok=True)
    lenient.evaluate([det("forklift")], 1280, 720)
    clock.t = 12
    assert lenient.evaluate([det("spotter")], 1280, 720).status == VerdictStatus.OK


def test_required_params():
    with pytest.raises(MissingCameraParams):
        load_station("followup_window", {"trigger_class": "forklift"})


def test_trigger_counts_once_per_arrival(make):
    s, clock = make()
    s.evaluate([det("forklift")], 1280, 720)                     # 進場，開始計時
    clock.t = 3
    assert s.evaluate([det("forklift"), det("spotter")], 1280, 720).details["state"] == "satisfied"
    for t in (4, 8, 16):                                         # 堆高機還停著：不算新的一次
        clock.t = t
        assert s.evaluate([det("forklift")], 1280, 720).status == VerdictStatus.OK
    clock.t = 30
    s.evaluate([], 1280, 720)
    clock.t = 31                                                 # 離開超過 3 秒後再進來 → 新的一輪
    assert s.evaluate([det("forklift")], 1280, 720).details["state"] == "armed"
