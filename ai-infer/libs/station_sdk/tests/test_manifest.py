import pytest
from edgevision_station_sdk import StationManifest
from pydantic import ValidationError


def test_manifest_reports_missing_params():
    m = StationManifest(name="x", model={"name": "m", "task": "segment", "classes": ["a"]},
                        handler="pkg:Cls", camera_params_required=["ref_quad", "roi"])
    assert m.missing_params({"ref_quad": 1}) == ["roi"]


def test_manifest_rejects_unknown_task():
    with pytest.raises(ValidationError):
        StationManifest(name="x", model={"name": "m", "task": "classify", "classes": ["a"]}, handler="p:C")


def test_roi_role_problems():
    from edgevision_station_sdk import RoiRole, roi_role_problems
    roles = [RoiRole(name="work", required=True), RoiRole(name="presence")]
    pts = [[0, 0], [1, 0], [1, 1]]
    assert roi_role_problems([{"type": "detect", "points": pts}], roles) == []          # 舊設定：沒標就不管
    assert roi_role_problems([{"type": "detect", "points": pts}], []) == []             # 站別沒宣告 role
    ok = [{"type": "detect", "role": "work", "points": pts}, {"type": "mask", "points": pts}]
    assert roi_role_problems(ok, roles) == []                                            # mask 不用標
    (p,) = roi_role_problems([{"type": "detect", "role": "presence", "points": pts}], roles)
    assert "work" in p
    probs = roi_role_problems([{"type": "detect", "role": "x", "points": pts},
                               {"type": "detect", "points": pts}], roles)
    assert len(probs) == 3 and "'x'" in probs[0] and "第 [2] 塊" in probs[1]
