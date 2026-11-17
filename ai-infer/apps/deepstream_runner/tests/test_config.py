import json
from unittest import mock

import pytest
from api_shapes import api_line
from ds_runner.config import line_url, load_line_config
from edgevision_station_sdk import MissingCameraParams, StationNotFound


def test_file_source_loads_api_shaped_json(line_file):
    cfg = load_line_config({"CAMERA_SOURCE": "file", "CAMERA_FILE": str(line_file), "LINE_ID": "alpha_a"})
    assert cfg.camera_ids == ["cam156", "cam193"] and cfg.config_version == 3
    assert cfg.frame_size == (1280, 720) and cfg.infer_interval == 5          # 30/5 - 1
    m = cfg.resolve_model()
    assert m.name == "seg_alpha" and m.classes == ["board", "hand"]


def test_api_source_hits_config_endpoint():
    body = json.dumps(api_line()).encode()
    resp = mock.MagicMock()
    resp.read.return_value = body
    resp.__enter__.return_value = resp
    with mock.patch("ds_runner.config.urlopen", return_value=resp) as u:
        cfg = load_line_config({"CAMERA_SOURCE": "api", "CONFIG_API": "http://output-api:8080/", "LINE_ID": "alpha_a"})
    assert u.call_args[0][0].full_url == "http://output-api:8080/api/config/lines/alpha_a"
    assert cfg.id == "alpha_a"
    assert line_url("http://x", "l") == "http://x/api/config/lines/l"


def test_no_implicit_fallback():
    with pytest.raises(ValueError, match="CAMERA_SOURCE"):
        load_line_config({"CAMERA_FILE": "/nonexistent"})


def test_disabled_camera_is_skipped(write_line):
    cams = api_line()["cameras"]
    cams[1]["enabled"] = False
    cfg = load_line_config(write_line(cameras=cams))
    assert cfg.camera_ids == ["cam156"]


def test_missing_param_skips_only_that_camera(write_line):
    """一台缺現場值 → 只跳過那台，同線其他台照常（2026-11-03 之前是整條線不啟動）。"""
    cams = api_line()["cameras"]
    cams[1]["station_params"] = {}
    cfg = load_line_config(write_line(cameras=cams))
    assert cfg.camera_ids == ["cam156"]
    assert "cam193" in cfg.skipped_cameras and "MissingCameraParams" in cfg.skipped_cameras["cam193"]


def test_every_camera_broken_blocks_startup(write_line):
    cams = api_line()["cameras"]
    for c in cams:
        c["station_params"] = {}
    with pytest.raises(MissingCameraParams, match="cam193"):
        load_line_config(write_line(cameras=cams))


def test_mixed_resolution_blocks_startup(write_line):
    cams = api_line()["cameras"]
    cams[1]["width"], cams[1]["height"] = 1920, 1080
    with pytest.raises(ValueError, match="解析度不一致"):
        load_line_config(write_line(cameras=cams))


def test_unknown_station_blocks_startup(write_line):
    with pytest.raises(StationNotFound):
        load_line_config(write_line(station="nope"))


def test_line_id_mismatch_and_disabled_line(line_file, write_line):
    with pytest.raises(ValueError, match="LINE_ID"):
        load_line_config({"CAMERA_SOURCE": "file", "CAMERA_FILE": str(line_file), "LINE_ID": "other"})
    with pytest.raises(ValueError, match="已停用"):
        load_line_config(write_line(enabled=False))


def test_model_classes_must_cover_station(write_line):
    """模型類別沒涵蓋站別需要的 → 載入就失敗（一線多站後 bindings() 在 validate 時就解析模型，不等到 worker 才炸）。"""
    with pytest.raises(ValueError, match="hand"):
        load_line_config(write_line(model={"name": "other", "classes": ["board"]}))


def test_legacy_params_alias_still_accepted(write_line):
    cams = [{"id": "cam156", "width": 1280, "height": 720, "params": api_line()["cameras"][0]["station_params"]}]
    cfg = load_line_config(write_line(cameras=cams, line_params={}))
    assert cfg.active_cameras[0].station_params and cfg.infer_interval is None
