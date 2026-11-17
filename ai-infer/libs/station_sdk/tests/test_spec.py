"""站別參數規格：從站別程式讀出，給設定 API 存檔時檢查。"""
import json

from edgevision_station_sdk.spec import build_specs


def test_specs_cover_installed_stations_and_threshold_fields():
    d = json.loads(json.dumps(build_specs(), default=str))      # 要能序列化成 JSON
    s = d["stations"]
    assert {"zone_intrusion", "followup_window", "ppe_check"} <= set(s)
    th = s["ppe_check"]["thresholds"]
    assert th["window_size"]["type"] == "int" and th["window_size"]["default"] == 10
    assert s["zone_intrusion"]["camera_params_required"] == ["zone"]
