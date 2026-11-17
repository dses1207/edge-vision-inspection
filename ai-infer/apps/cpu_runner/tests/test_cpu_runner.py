"""合成場景走完真的 worker：後處理 → 站別 → FrameResult。不推流、不連 Redis。"""
import json

import pytest
from cpu_runner.detector import SceneDetector
from cpu_runner.main import CpuRunner
from cpu_runner.scene import Scene
from ds_runner.config import load_line_config
from edgevision_contracts import VerdictStatus

ZONE = [[0.62, 0.45], [0.95, 0.45], [0.95, 0.95], [0.62, 0.95]]
DEMO_DET = {"name": "demo_det", "classes": ["person", "forklift", "spotter"], "task": "detect", "imgsz": 640}
PPE = {"name": "ppe", "classes": ["helmet", "no_helmet", "vest", "no_vest"], "task": "detect", "imgsz": 640}


class SceneClock:
    """站別的冷卻 / 計時改跟場景時間走：測試一口氣跑完幾十秒的劇本，不用真的等。"""
    t = 0.0

    def __call__(self):
        return self.t


@pytest.fixture
def clock(monkeypatch):
    from edgevision_station_followup_window import FollowupWindowStation
    from edgevision_station_ppe_check import PpeCheckStation
    from edgevision_station_zone_intrusion import ZoneIntrusionStation
    c = SceneClock()
    for cls in (ZoneIntrusionStation, FollowupWindowStation, PpeCheckStation):
        monkeypatch.setattr(cls, "clock", c)
    return c


def run(r, clock, seconds, fps):
    for i in range(int(seconds * fps)):
        clock.t = i / fps
        r.step(clock.t)


def write_line(tmp_path, **over):
    line = {"id": "demo_zone", "station": "zone_intrusion", "config_version": 1, "model": DEMO_DET,
            "line_params": {"target_fps": 10},
            "extra_stations": [{"station": "ppe_check", "interval": 1, "model": PPE}],
            "cameras": [{"id": "cam-01", "width": 1280, "height": 720,
                         "station_params": {"zone": ZONE, "thresholds": {"persist_frames": 3}},
                         "extra_params": {"ppe_check": {}}}]}
    line.update(over)
    p = tmp_path / "line.json"
    p.write_text(json.dumps(line), encoding="utf-8")
    return {"CAMERA_SOURCE": "file", "CAMERA_FILE": str(p)}


def test_scene_detector_tensor_roundtrips_through_postprocess():
    from edgevision_postprocess import compute_letterbox, make_postprocessor
    scene = Scene("cam-01", 1280, 720)
    objs = [o for o in scene.objects(3.0) if o.class_name == "person"]
    lb = compute_letterbox(1280, 720, 640)
    det = SceneDetector(DEMO_DET["classes"], miss_rate=0, jitter_px=0)
    got = make_postprocessor("detect", DEMO_DET["classes"])(det(objs, lb), None, lb, 1280, 720)
    assert len(got) == len(objs)
    for o in objs:
        assert any(abs(d.bbox.x1 - o.box[0]) < 1 and abs(d.bbox.y2 - o.box[3]) < 1 for d in got)


def test_runner_produces_zone_and_ppe_ng(tmp_path, clock):
    env = write_line(tmp_path)
    published = []
    r = CpuRunner(load_line_config(env), {**env, "GO2RTC_RTSP": ""})
    r.publish = lambda res, img: published.append(res)
    run(r, clock, 40, 10)                         # 40 秒的劇本
    ng = {(x.station, x.verdict.reason) for x in published if x.verdict.status == VerdictStatus.NG}
    assert ("zone_intrusion", "person 進入禁區") in ng
    assert any(s == "ppe_check" and "未戴安全帽" in reason for s, reason in ng)


def test_dock_scene_followup_timeout(tmp_path, clock):
    env = write_line(tmp_path, id="demo_dock", station="followup_window", extra_stations=[],
                     line_params={"target_fps": 5, "demo_scene": "dock"},
                     cameras=[{"id": "cam-02", "width": 1280, "height": 720,
                               "station_params": {"trigger_class": "forklift", "required_class": "spotter"}}])
    published = []
    r = CpuRunner(load_line_config(env), {**env, "GO2RTC_RTSP": ""})
    r.publish = lambda res, img: published.append(res)
    run(r, clock, 130, 5)                         # 三個 40 秒週期，第三次引導員缺席
    ng = [x for x in published if x.verdict.status == VerdictStatus.NG]
    assert len(ng) == 1 and "spotter" in ng[0].verdict.reason
