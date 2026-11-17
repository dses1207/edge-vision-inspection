"""煙霧測試:服務都活著、端到端鏈路在動。

這是最快的健康確認 —— 不驗證細節,只確認「整條鏈沒斷」。
"""
import os

import httpx
import pytest

API = os.environ.get("API_URL", "http://localhost:8080")
TRITON = os.environ.get("TRITON_URL", "http://localhost:8000")


def test_triton_ready():
    r = httpx.get(f"{TRITON}/v2/health/ready", timeout=5)
    assert r.status_code == 200


def test_api_healthy():
    r = httpx.get(f"{API}/healthz", timeout=5)
    d = r.json()
    assert d["healthy"] is True
    assert d["storage"]["database"] is True
    assert d["storage"]["objects"] is True


def test_pipeline_advancing():
    """推論管線真的在處理影格(不是卡住)。"""
    d = httpx.get(f"{API}/healthz", timeout=5).json()
    cam = next(iter(d["cameras"].values()))
    assert cam["frames_seen"] > 0, "解碼沒有產出影格"
    assert cam["frames_inferred"] > 0, "推論沒有執行"
    # 健康檢查標準:最近有處理到影格
    assert cam["healthy"] is True


def test_restream_running():
    d = httpx.get(f"{API}/healthz", timeout=5).json()
    cam = next(iter(d["cameras"].values()))
    if cam.get("restream"):
        assert cam["restream"]["frames_pushed"] > 0


def test_results_queryable():
    r = httpx.get(f"{API}/api/detections?limit=5", timeout=5)
    assert r.status_code == 200
    assert "detections" in r.json()
