import pytest
from edgevision_contracts import BBox, Detection, VerdictStatus
from edgevision_station_sdk import MissingCameraParams, load_station

ZONE = [[0.5, 0.0], [1.0, 0.0], [1.0, 1.0], [0.5, 1.0]]          # 右半邊


def person(x1, y1, x2, y2, conf=0.9):
    return Detection(class_id=0, class_name="person", confidence=conf, bbox=BBox(x1=x1, y1=y1, x2=x2, y2=y2))


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


@pytest.fixture
def station(monkeypatch):
    from edgevision_station_zone_intrusion import ZoneIntrusionStation
    clock = FakeClock()
    monkeypatch.setattr(ZoneIntrusionStation, "clock", clock)
    s = load_station("zone_intrusion", {"zone": ZONE, "thresholds": {"persist_frames": 2, "cooldown_seconds": 5}})
    return s, clock


def test_ng_after_persist_frames_then_cooldown(station):
    s, clock = station
    inside = [person(800, 100, 900, 400)]
    assert s.evaluate(inside, 1000, 500).status == VerdictStatus.OK
    assert s.evaluate(inside, 1000, 500).status == VerdictStatus.NG
    v = s.evaluate(inside, 1000, 500)
    assert v.status == VerdictStatus.OK and v.reason == "冷卻中"
    clock.t = 6
    assert s.evaluate(inside, 1000, 500).status == VerdictStatus.NG


def test_uses_foot_point_not_box_center(station):
    s, _ = station
    leaning = [person(300, 100, 650, 400)]           # 中心 x=475 在區外；腳點同樣在區外
    reaching = [person(400, 100, 700, 400)]          # 中心 x=550 在區內 → 腳點也在區內
    for _ in range(3):
        assert s.evaluate(leaning, 1000, 500).status == VerdictStatus.OK
    s.evaluate(reaching, 1000, 500)
    assert s.evaluate(reaching, 1000, 500).status == VerdictStatus.NG


def test_low_confidence_ignored(station):
    s, _ = station
    for _ in range(3):
        assert s.evaluate([person(800, 100, 900, 400, conf=0.1)], 1000, 500).status == VerdictStatus.OK


def test_zone_is_required():
    with pytest.raises(MissingCameraParams):
        load_station("zone_intrusion", {})
