"""設定 API 的延伸欄位（一線多站、模型覆寫、ROI）：輸入檢查。資料庫讀寫在測試機實機驗。"""
import pytest
from edgevision.api.config_api import CameraIn, LineIn
from pydantic import ValidationError

CAM = {"id": "ipcam198", "ip": "192.0.2.109", "stream_path": "/stream", "width": 1920, "height": 1080,
       "cred_ref": "none"}


def test_defaults_keep_old_payloads_valid():
    line = LineIn(id="alpha_a", station="zone_intrusion", cameras=[CameraIn(**CAM)])
    assert line.model is None and line.extra_stations == []
    assert line.cameras[0].infer_roi is None and line.cameras[0].extra_params == {}


def test_full_ipcam_line_accepted():
    cam = CameraIn(**CAM, infer_roi=[0.0, 0.25, 0.5, 1.0], extra_params={"ppe_check": {"thr": 0.5}})
    line = LineIn(id="ipcam_alpha", station="alpha_check", cameras=[cam],
                  model={"name": "alpha_ipcam", "classes": ["a"], "imgsz": 640},
                  extra_stations=[{"station": "ppe_check", "model": {"name": "ppe_alpha"}}])
    assert line.model["name"] == "alpha_ipcam" and line.extra_stations[0]["station"] == "ppe_check"


@pytest.mark.parametrize("roi", [[0.5, 0.2, 0.4, 0.9], [0, 0, 1], [0, 0, 1.2, 1], [-0.1, 0, 1, 1]])
def test_bad_roi_rejected(roi):
    with pytest.raises(ValidationError, match="infer_roi"):
        CameraIn(**CAM, infer_roi=roi)


def test_model_needs_name():
    with pytest.raises(ValidationError, match="name"):
        LineIn(id="x", station="s", model={"imgsz": 640})


def test_extra_station_needs_station():
    with pytest.raises(ValidationError, match="station"):
        LineIn(id="x", station="s", extra_stations=[{"model": {"name": "m"}}])
