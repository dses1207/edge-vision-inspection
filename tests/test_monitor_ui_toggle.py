"""內建監看頁可關（測試叢集讓真正的前端測試時）：/ 回 404，API 照常。"""
from fastapi.testclient import TestClient

from edgevision.api import server


def test_root_closed_when_disabled(monkeypatch):
    monkeypatch.setenv("MONITOR_UI_ENABLED", "false")
    c = TestClient(server.app)
    r = c.get("/")
    assert r.status_code == 404 and "已關閉" in r.json()["message"]
    assert c.get("/api/frontend-config").status_code == 200


def test_root_open_by_default(monkeypatch):
    monkeypatch.delenv("MONITOR_UI_ENABLED", raising=False)
    assert server.monitor_ui_enabled()


def test_review_endpoint_marks_frame(monkeypatch):
    """B3：人工審核一個 NG 事件；找不到回 404。"""
    calls = []

    class DB:
        def review(self, **kw):
            calls.append(kw)
            return 2 if kw["frame_id"] == "42" else 0
    monkeypatch.setattr(server, "_db", DB())
    c = TestClient(server.app)
    r = c.post("/api/detections/review", json={"camera_id": "cam193", "frame_id": "42",
                                               "result": "false_positive", "actor": "op1"})
    assert r.status_code == 200 and r.json()["rows"] == 2
    assert calls[0]["result"] == "false_positive" and calls[0]["actor"] == "op1"
    r = c.post("/api/detections/review", json={"camera_id": "cam193", "frame_id": "9",
                                               "result": "true_positive", "actor": "op1"})
    assert r.status_code == 404
    bad = c.post("/api/detections/review", json={"camera_id": "cam193", "frame_id": "42", "result": "maybe", "actor": "op1"})
    assert bad.status_code == 422


def test_openapi_matches_reality():
    """C1~C5：規格要跟實際回應一致。"""
    spec = server.app.openapi()
    row = spec["components"]["schemas"]["DetectionRow"]["properties"]
    for k in ("reviewed", "review_result", "station", "evidence_expired_at", "image_meta", "verdict_details"):
        assert k in row
    assert "fps" in spec["components"]["schemas"]["LineStatus"]["properties"]
    for path in ("/api/stats", "/api/image"):
        assert spec["paths"][path]["get"]["responses"]["200"]["content"]["application/json"]["schema"]
