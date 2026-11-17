"""跨套件冒煙測試：登記簿裡有哪些站、每站的 station.yaml 都合法、handler 與 entry point 一致。"""
from edgevision_station_sdk import list_stations


def test_all_registered_stations_have_valid_manifest():
    stations = list_stations()
    assert {"template", "zone_intrusion", "followup_window", "ppe_check"} <= set(stations)
    for name, entry in stations.items():
        m = entry.manifest
        assert m.name == name, f"{name}: station.yaml 的 name 與 entry point 不一致"
        assert m.handler == f"{entry.module}:{entry.attr}", f"{name}: handler 與 entry point 不一致"
