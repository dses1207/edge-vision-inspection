"""控制 API 整合測試：健康、請求驗證、任務查詢、模型清單。需 control-api 運行。"""
import os

import httpx
import pytest

pytestmark = pytest.mark.integration
API = os.environ.get("CONTROL_URL", "http://localhost:8090")


@pytest.fixture(scope="module")
def api():
    try:
        r = httpx.get(f"{API}/healthz", timeout=5)
    except Exception:
        pytest.skip(f"control-api 未就緒（{API}）")
    if r.status_code != 200:
        pytest.skip(f"control-api 不健康：{r.text}")
    return httpx.Client(base_url=API, timeout=15, headers={**_auth(), "X-Channel": "api"})


def _auth() -> dict:
    """EDGEVISION_TOKEN（SSO token）或 EDGEVISION_SERVICE_TOKEN（服務金鑰）；都沒有才退回 X-Actor（強制模式會 401）。"""
    if os.environ.get("EDGEVISION_TOKEN"):
        return {"Authorization": f"Bearer {os.environ['EDGEVISION_TOKEN']}"}
    if os.environ.get("EDGEVISION_SERVICE_TOKEN"):
        return {"X-Service-Token": os.environ["EDGEVISION_SERVICE_TOKEN"]}
    return {"X-Actor": "itest"}


def test_health(api):
    h = api.get("/healthz").json()
    assert h["ok"] and h["db"] and h["minio"] and h["runner_ok"]


def test_invalid_job_rejected(api):
    r = api.post("/api/training/jobs", json={"model_name": "no_such_model", "dataset_ref": "local:///nope"})
    assert r.status_code == 400 and "設定" in r.text


def test_bad_channel_rejected(api):
    r = api.post("/api/training/jobs", json={"model_name": "seg_alpha", "dataset_ref": "x"}, headers={"X-Channel": "webhook"})
    assert r.status_code == 400


def test_task_listing_and_404(api):
    assert api.get("/api/tasks?limit=5").status_code == 200
    assert api.get("/api/tasks/does-not-exist").status_code == 404


def test_models_listing(api):
    r = api.get("/api/models")
    assert r.status_code == 200 and isinstance(r.json(), list)
