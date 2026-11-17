"""設定 API 存檔前的兩道關：站別參數檢查（規格來自 runner 映像）、版本衝突（兩人同時改）。"""
import json

import pytest
from fastapi import HTTPException

from edgevision.api import config_api, station_specs
from edgevision.api.config_api import CameraIn, ChangeMeta, LineIn, LineUpsert, upsert_line
from edgevision.api.station_specs import check_line, load_specs

SPECS = {
    "zone_pose": {"camera_params_required": [],
               "thresholds": {"persist_frames": {"type": "int", "default": 3},
                              "pad_kpt_conf": {"type": "float", "default": 0.3},
                              "enable_persist": {"type": "bool", "default": True}}},
    "part_check": {"camera_params_required": [], "thresholds": {"persist_frames": {"type": "int", "default": 3}}},
    "zone_intrusion": {"camera_params_required": ["ref_quad"], "thresholds": None},
}


def _line(station="part_check", th=None, extra_th=None, **kw):
    cam = CameraIn(id="cam124", ip="192.0.2.190", width=1280, height=720, station_params={"thresholds": th or {}},
                   extra_params={"zone_pose": {"thresholds": extra_th or {}}})
    return LineIn(id="line_c", station=station, cameras=[cam],
                  extra_stations=[{"station": "zone_pose"}], **kw)


def test_good_line_passes():
    assert check_line(_line(th={"persist_frames": 15}, extra_th={"pad_kpt_conf": 0.3}), SPECS) == []


def test_typo_and_wrong_type_caught():
    p = check_line(_line(th={"persist_frame": 15}, extra_th={"persist_frames": "15"}), SPECS)
    assert any("persist_frame" in x and "不認識" in x for x in p)
    assert any("persist_frames 要是 int" in x for x in p)


def test_unknown_station_and_missing_required():
    assert any("不存在" in x for x in check_line(_line(station="part_need_pad"), SPECS))
    flip = LineIn(id="alpha_a", station="zone_intrusion", cameras=[CameraIn(id="cam156", ip="192.0.2.102", width=1280, height=720)])
    assert any("ref_quad" in x for x in check_line(flip, SPECS))

    # 欄位在、內容是空的也算沒給（webcam 77/79 的 roi_regions: [] 就是這種）
    flip.cameras[0].station_params = {"ref_quad": []}
    assert any("ref_quad" in x for x in check_line(flip, SPECS))


def test_disabled_camera_not_checked():
    line = _line(th={"nope": 1})
    line.cameras[0].enabled = False
    assert check_line(line, SPECS) == []


def test_specs_file_missing_means_unchecked(tmp_path):
    assert load_specs(str(tmp_path / "none.json")) is None
    f = tmp_path / "s.json"
    f.write_text(json.dumps({"stations": SPECS}))
    assert set(load_specs(str(f))) == set(SPECS)


def test_upsert_rejects_bad_params_before_touching_db(monkeypatch):
    monkeypatch.setattr(config_api, "load_specs", lambda: SPECS)
    monkeypatch.setattr(config_api, "_db", None)          # 走到資料庫就會 503，證明先被擋
    body = LineUpsert(line=_line(th={"persist_frame": 15}), meta=ChangeMeta(actor="t"))
    with pytest.raises(HTTPException) as e:
        upsert_line("line_c", body)
    assert e.value.status_code == 422 and e.value.detail["problems"]


class _Cur:
    def __init__(self, current):
        self.current, self.sql = current, []

    def execute(self, sql, args=None):
        self.sql.append(sql)

    def fetchone(self):
        return (self.current,) if self.current is not None else None


class _DB:
    def __init__(self, cur):
        self.cur = cur

    def cursor(self):
        db = self

        class _Ctx:
            def __enter__(self):
                return db.cur

            def __exit__(self, *a):
                return False
        return _Ctx()


@pytest.mark.parametrize("current,base", [(5, 4), (None, 3), (2, 0)])
def test_stale_base_version_is_409(monkeypatch, current, base):
    monkeypatch.setattr(config_api, "load_specs", lambda: None)
    monkeypatch.setattr(config_api, "prepare_line_for_save", lambda line, specs, m: ([], [], {}))   # 不查 Registry
    monkeypatch.setattr(config_api, "_db", _DB(_Cur(current)))
    body = LineUpsert(line=_line(), meta=ChangeMeta(actor="t", base_version=base))
    with pytest.raises(HTTPException) as e:
        upsert_line("line_c", body)
    assert e.value.status_code == 409 and e.value.detail["current_version"] == (current or 0)


def test_restore_keeps_current_enabled_state(monkeypatch):
    """還原舊版本不能偷偷把停用中的線打開（快照不含啟用狀態）。"""
    snap = {"line": {"name": "x", "station": "part_check", "line_params": {}},
            "cameras": [{"id": "cam124", "ip": "192.0.2.190", "width": 1280, "height": 720}]}

    class Cur(_Cur):
        def __init__(self):
            super().__init__(None)
            self.rows = [(snap,), (False,)]           # 第一次查快照、第二次查目前的 enabled

        def fetchone(self):
            return self.rows.pop(0)

    monkeypatch.setattr(config_api, "_db", _DB(Cur()))
    seen = {}
    monkeypatch.setattr(config_api, "upsert_line", lambda lid, body: seen.setdefault("line", body.line))
    config_api.restore_version("line_c", 3, ChangeMeta(actor="t"))
    assert seen["line"].enabled is False


ROLE_SPECS = {"bravo_cleanhand": {
    "camera_params_required": ["roi_regions"], "thresholds": None,
    "roi_roles": [{"name": "work", "display_name": "工作區", "classes": ["board", "sheet"], "required": True},
                  {"name": "clean", "display_name": "清潔區", "classes": ["cls_a", "glove"], "required": True},
                  {"name": "presence", "display_name": "人員區", "classes": ["cls_a", "glove"], "required": False}]}}
PTS = [[0.1, 0.1], [0.5, 0.1], [0.5, 0.5]]


def _roi_line(regions):
    cam = CameraIn(id="cam_10_17_147_76", ip="192.0.2.203", width=1280, height=720,
                   station_params={"roi_regions": regions})
    return LineIn(id="webcam_bravo", station="bravo_cleanhand", cameras=[cam])


def test_roi_roles_checked_on_save():
    legacy = [{"type": "detect", "points": PTS, "classes": ["board", "sheet"]}]
    assert check_line(_roi_line(legacy), ROLE_SPECS) == []                     # 舊設定沒標 role：不管
    good = [{"type": "detect", "role": "work", "points": PTS}, {"type": "detect", "role": "clean", "points": PTS},
            {"type": "mask", "points": PTS}]
    assert check_line(_roi_line(good), ROLE_SPECS) == []
    p = check_line(_roi_line([{"type": "detect", "role": "work", "points": PTS},
                              {"type": "detect", "role": "presense", "points": PTS},
                              {"type": "detect", "points": PTS}]), ROLE_SPECS)
    assert any("presense" in x for x in p)
    assert any("第 [3] 塊" in x for x in p)
    assert any("clean（清潔區）" in x for x in p)


def test_station_specs_endpoint(monkeypatch):
    monkeypatch.setattr(config_api, "load_specs", lambda: ROLE_SPECS)
    (s,) = config_api.list_station_specs()
    assert s["name"] == "bravo_cleanhand" and [r["name"] for r in s["roi_roles"]] == ["work", "clean", "presence"]
    monkeypatch.setattr(config_api, "load_specs", lambda: None)
    with pytest.raises(HTTPException) as e:
        config_api.list_station_specs()
    assert e.value.status_code == 503
