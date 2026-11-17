"""PTZ（舊系統的 PTZ 服務移植）：失準量測、自動回正的決策、歸位、看門狗。用假相機 + 合成畫面，不碰真相機。"""
import time

import numpy as np
import pytest

from edgevision.ptz import settings as S
from edgevision.ptz import vision
from edgevision.ptz.devices import Position
from edgevision.ptz.service import PtzService
from edgevision.ptz.store import PtzCamera


def _scene(seed=0, h=720, w=1280):
    rng = np.random.default_rng(seed)
    img = rng.normal(128, 40, (h // 8, w // 8))
    return np.kron(img, np.ones((8, 8)))          # 有紋理的畫面（相位相關要有特徵）


def _shifted(img, dx, dy):
    return np.roll(np.roll(img, dy, axis=0), dx, axis=1)


def test_phase_corr_recovers_shift():
    base = vision.to_gray_small(_scene())
    moved = vision.to_gray_small(_shifted(_scene(), 5 * 40, 5 * 25))   # 原圖 1280→256（÷5）、720→144（÷5）
    dx, dy, conf = vision.phase_corr(base, moved)
    assert (abs(dx), abs(dy)) == (40, 25) and conf > S.DRIFT_ALIGN_MIN_CONF
    assert vision.phase_corr(base, base)[:2] == (0, 0)
    noise = vision.to_gray_small(np.random.default_rng(9).normal(128, 40, (720, 1280)))
    assert vision.phase_corr(base, noise)[2] < S.DRIFT_ALIGN_MIN_CONF      # 不相干的畫面 → 低信心


def test_ref_roundtrip():
    a = vision.to_gray_small(_scene())
    assert np.allclose(vision.load_ref(vision.dump_ref(a)), a)


class FakeStore:
    def __init__(self):
        self.pos = {}
        self.can_pan = {}

    def camera(self, cam_id):
        return PtzCamera(id=cam_id, ip="192.0.2.84", onvif_port=2020, cred_ref="none") if cam_id == "cam50" else None

    def position(self, cam_id, kind):
        return self.pos.get((cam_id, kind))

    def upsert(self, cam_id, kind, *, pan, tilt, zoom=0.0, has_coord=True, ref_object_key=None, actor=None):
        old = self.pos.get((cam_id, kind), {})
        self.pos[(cam_id, kind)] = {"pan": pan, "tilt": tilt, "zoom": zoom, "has_coord": has_coord,
                                    "ref_object_key": ref_object_key or old.get("ref_object_key"), "updated_by": actor}

    def cameras_with(self, kind):
        return [c for (c, k) in self.pos if k == kind]

    def cameras(self):
        return [PtzCamera(id="cam50", ip="192.0.2.84", onvif_port=2020, cred_ref="none",
                          can_pan=self.can_pan.get("cam50", True)),
                PtzCamera(id="siteb216", ip="192.0.2.137", onvif_port=2020, cred_ref="none")]

    def all_positions(self):
        return {}

    def set_can_pan(self, cam_id, can_pan):
        self.can_pan[cam_id] = can_pan


class FakeObjects:
    def __init__(self):
        self.data = {}

    def put(self, *, bucket, key, data, length, content_type=None):
        self.data[(bucket, key)] = data.read()

    def get(self, *, bucket, key):
        return self.data[(bucket, key)]


class FakeDevice:
    """模擬相機：記座標；畫面 = 基準場景依「相機被撥歪的量 - 已補回的量」平移。"""

    def __init__(self):
        self.p = Position(0.1, 0.2, 0.0)
        self.calls = []
        self.knocked = (0, 0)       # 外力造成的畫面位移（像素，256x144 尺度）

    def status(self):
        return Position(self.p.pan, self.p.tilt, self.p.zoom)

    def continuous_move(self, pan, tilt, zoom):
        self.calls.append(("move", pan, tilt, zoom))

    def stop(self):
        self.calls.append(("stop",))

    def relative_move(self, dx, dy, dz=0.0):
        self.calls.append(("rel", dx, dy))
        # 相對移動讓畫面平移 dx × 增益（方向與舊系統現場驗證過的對應一致：phase_corr 量到 -30 → 移 -30/增益 → 補回）
        kx, ky = self.knocked
        self.knocked = (kx + round(dx * S.DRIFT_ALIGN_GAIN_X), ky + round(dy * S.DRIFT_ALIGN_GAIN_Y))
        self.p = Position(self.p.pan + dx, self.p.tilt + dy, self.p.zoom)

    def absolute_move(self, pan, tilt, zoom):
        self.calls.append(("abs", pan, tilt, zoom))
        self.p = Position(pan, tilt, zoom)


class FakePool:
    def __init__(self, dev):
        self.dev = dev

    def get(self, cam_id):
        if cam_id != "cam50":
            raise LookupError(cam_id)
        return self.dev

    def drop(self, cam_id):
        pass


@pytest.fixture
def svc():
    dev = FakeDevice()
    base = _scene(1)

    def grab(cam_id):
        kx, ky = dev.knocked
        return vision.to_gray_small(_shifted(base, kx * 5, ky * 5))

    calibrated = []
    s = PtzService(FakeStore(), FakeObjects(), pool=FakePool(dev), grab=grab,
                   calibrate=lambda cam: calibrated.append(cam.id), sleep=lambda s: None)
    s._dev_fake, s._calibrated = dev, calibrated
    return s


def test_reference_and_check_drift(svc):
    assert svc.check_drift("cam50") == {"ok": False, "reason": "no_reference"}
    r = svc.save_reference("cam50", actor="u")
    assert r["pan"] == pytest.approx(0.1) and r["ref_object_key"].startswith("ptz/refs/cam50/")
    ok = svc.check_drift("cam50")
    assert ok["ok"] and ok["drift"] is False and ok["shift"] == 0
    svc._dev_fake.knocked = (30, 0)                        # 被外力扳動，但座標沒變（座標在說謊）
    bad = svc.check_drift("cam50")
    assert bad["drift"] is True and bad["shift"] >= S.DRIFT_SHIFT_PX
    svc.mark_control("cam50")                               # 有人剛操作 → 跳過
    assert svc.check_drift("cam50")["skipped"] == "recent_control"
    assert svc.check_drift("cam50", force=True)["drift"] is True


def test_three_consecutive_drifts_trigger_visual_recovery_without_calibration(svc):
    svc.save_reference("cam50", actor="u")
    svc.store.upsert("cam50", "target", pan=0.1, tilt=0.2, zoom=0.0, actor="u")
    svc._dev_fake.knocked = (30, 10)
    started = []
    svc.recover_async = lambda cam: started.append(cam) or True
    for _ in range(S.DRIFT_CONSECUTIVE - 1):
        svc.check_one("cam50")
    assert not started and svc.drift_state["cam50"]["count"] == S.DRIFT_CONSECUTIVE - 1
    svc.check_one("cam50")
    assert started == ["cam50"]
    svc.auto_recover("cam50")                               # 同步跑一次回正
    st = svc.drift_state["cam50"]
    assert st["state"] == "recovered" and st["shift"] <= S.DRIFT_ALIGN_OK_PX
    assert svc._calibrated == []                           # 視覺對齊就到位，不需馬達校準
    assert any(c[0] == "rel" for c in svc._dev_fake.calls)
    assert svc.store.position("cam50", "target") == {"pan": 0.1, "tilt": 0.2, "zoom": 0.0, "has_coord": True,
                                                    "ref_object_key": None, "updated_by": "u"}   # 歸位點不被覆寫


def test_low_confidence_never_triggers_motor_calibration(svc):
    svc.save_reference("cam50", actor="u")
    seeds = iter(range(100, 110))       # 固定雜訊樣本（信心 7–9，門檻 12）：畫面被大物體擋住
    svc._grab = lambda cam: vision.to_gray_small(np.random.default_rng(next(seeds)).normal(128, 40, (720, 1280)))
    for _ in range(5):
        svc.check_one("cam50")
    assert svc.drift_state["cam50"]["count"] == 0           # 低信心不累計
    svc.auto_recover("cam50")
    assert svc.drift_state["cam50"]["state"] == "low_conf_hold" and svc._calibrated == []


def test_recover_goes_home_first_when_coordinates_away(svc):
    svc.save_reference("cam50", actor="u")
    svc.store.upsert("cam50", "target", pan=0.1, tilt=0.2, zoom=0.0, actor="u")
    svc._dev_fake.p = Position(0.6, -0.3, 0.0)             # 被手動撥走、忘了歸位
    svc.auto_recover("cam50")
    assert ("abs", 0.1, 0.2, 0.0) in svc._dev_fake.calls


def test_homing_respects_recent_control_and_calibration(svc):
    t = {"pan": 0.3, "tilt": 0.1, "zoom": 0.0}
    svc.mark_control("cam50")
    assert svc.home_one("cam50", t) == "recent_control"
    svc.last_control.clear()
    svc.calibrating.add("cam50")
    assert svc.home_one("cam50", t) == "calibrating"
    svc.calibrating.clear()
    assert svc.home_one("cam50", t) == "homed" and ("abs", 0.3, 0.1, 0.0) in svc._dev_fake.calls


def test_move_watchdog_stops_camera(svc, monkeypatch):
    monkeypatch.setattr(S, "MOVE_WATCHDOG_SEC", 0.1)
    svc.move("cam50", 0.5, 0, 0)
    assert svc._dev_fake.calls[-1] == ("move", 0.5, 0, 0)       # pan 符號修正在 OnvifDevice 裡（相機怪癖）
    time.sleep(0.3)
    assert svc._dev_fake.calls[-1] == ("stop",)


def test_save_and_goto_target(svc):
    svc._dev_fake.p = Position(0.25, -0.1, 0.4)
    assert svc.save_position("cam50", "target", actor="u") == {"pan": 0.25, "tilt": -0.1, "zoom": 0.4}
    assert svc.save_position("cam50", "origin", actor="u")["zoom"] == 0.0             # 中心點不記 zoom
    svc._dev_fake.p = Position(0, 0, 0)
    svc.goto_position("cam50", "target")
    assert svc._dev_fake.calls[-1] == ("abs", 0.25, -0.1, 0.4)
    with pytest.raises(LookupError):
        svc.status("cam105")


def test_api_routes_and_error_mapping(svc, monkeypatch):
    """API：未知相機 404、沒設歸位點 404、相機連不上 502；改設定的動作寫稽核。"""
    from fastapi.testclient import TestClient

    from edgevision.ptz import api

    audits = []

    class DB:
        def write_audit(self, **kw):
            audits.append(kw)

        def ping(self):
            return True

    monkeypatch.setitem(api._state, "svc", svc)
    monkeypatch.setitem(api._state, "db", DB())
    c = TestClient(api.app)            # 不用 with：不跑 lifespan（不連真資料庫）
    h = {"X-Actor": "tester", "X-Channel": "api"}
    assert c.get("/api/ptz/cam105/status").status_code == 404
    assert c.get("/api/ptz/cam50/status").json() == {"camera_id": "cam50", "pan": 0.1, "tilt": 0.2, "zoom": 0.0}
    assert c.post("/api/ptz/cam50/target/goto").status_code == 404
    assert c.post("/api/ptz/cam50/target", headers=h).json()["pan"] == pytest.approx(0.1)
    assert audits[-1]["action"] == "ptz.target.save" and audits[-1]["actor"] == "tester"
    assert c.post("/api/ptz/cam50/move", json={"pan": 2}).status_code == 422          # 速度超出 -1~1
    svc._dev_fake.status = lambda: (_ for _ in ()).throw(OSError("timed out"))
    assert c.get("/api/ptz/cam50/status").status_code == 502
    assert {x["id"]: x["can_pan"] for x in c.get("/api/ptz/cameras").json()} == {"cam50": True, "siteb216": False}
    assert c.put("/api/ptz/cam50/can-pan", json={"can_pan": False}, headers=h).json() == \
        {"id": "cam50", "can_pan": False, "detected": False}
    assert audits[-1]["action"] == "ptz.can_pan.set"
    monkeypatch.setattr("edgevision.ptz.devices.detect_pan_tilt", lambda cam: True)
    assert c.put("/api/ptz/cam50/can-pan", json={"can_pan": None}, headers=h).json()["detected"] is True
    assert c.put("/api/ptz/cam105/can-pan", json={"can_pan": True}, headers=h).status_code == 404


def test_missing_credentials_is_an_error_not_exit(monkeypatch):
    for k in ("CAM_USER", "CAM_PASS", "CAM_ALPHA_USER", "CAM_ALPHA_PASS"):
        monkeypatch.delenv(k, raising=False)
    with pytest.raises(RuntimeError, match="找不到相機帳密"):
        PtzCamera(id="cam156", ip="192.0.2.84", onvif_port=2020, cred_ref="alpha").credentials()
