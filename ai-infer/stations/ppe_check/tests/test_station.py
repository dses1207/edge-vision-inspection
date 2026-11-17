import pytest
from edgevision_contracts import BBox, Detection, VerdictStatus
from edgevision_station_sdk import load_station


def det(name, x=100, conf=0.8):
    return Detection(class_id=0, class_name=name, confidence=conf, bbox=BBox(x1=x, y1=100, x2=x + 50, y2=150))


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


@pytest.fixture
def station(monkeypatch):
    from edgevision_station_ppe_check import PpeCheckStation
    clock = FakeClock()
    monkeypatch.setattr(PpeCheckStation, "clock", clock)
    s = load_station("ppe_check", {"thresholds": {"window_size": 4, "min_count": 3, "report_cooldown_seconds": 30}})
    return s, clock


def test_confirms_only_after_consistent_frames(station):
    s, _ = station
    assert s.evaluate([det("no_helmet")], 1280, 720).status == VerdictStatus.OK
    assert s.evaluate([det("no_helmet")], 1280, 720).status == VerdictStatus.OK
    v = s.evaluate([det("no_helmet")], 1280, 720)
    assert v.status == VerdictStatus.NG and v.reason == "未戴安全帽"


def test_flicker_in_different_places_not_confirmed(station):
    s, _ = station
    for x in (0, 400, 800, 1200):
        assert s.evaluate([det("no_helmet", x=x)], 1280, 720).status == VerdictStatus.OK


def test_cooldown_per_reason(station):
    s, clock = station
    for _ in range(3):
        s.evaluate([det("no_helmet")], 1280, 720)
    assert s.evaluate([det("no_helmet")], 1280, 720).status == VerdictStatus.OK       # 冷卻中
    for _ in range(3):
        v = s.evaluate([det("no_helmet"), det("no_vest", x=600)], 1280, 720)
    assert v.status == VerdictStatus.NG and v.reason == "未穿反光背心"                   # 別的原因不受影響
    clock.t = 31
    assert s.evaluate([det("no_helmet")], 1280, 720).status == VerdictStatus.NG


def test_unknown_threshold_rejected():
    with pytest.raises(ValueError):
        load_station("ppe_check", {"thresholds": {"windw": 3}})
