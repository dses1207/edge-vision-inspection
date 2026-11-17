"""CORS：CORS_ORIGINS=* 不限來源（兩個 API 都要有，控制 API 以前完全沒設）；白名單外的來源拿不到允許標頭。"""
import importlib

import pytest
from fastapi.testclient import TestClient

from edgevision.api.cors import cors_origins

DEV = "http://192.0.2.30:5173"


def _app(monkeypatch, module: str, origins: str):
    monkeypatch.setenv("CORS_ORIGINS", origins)
    mod = importlib.reload(importlib.import_module(module))
    return TestClient(mod.app)          # 不進 lifespan：不連 DB


@pytest.mark.parametrize("module", ["edgevision.api.control", "edgevision.api.server"])
def test_star_allows_any_origin_including_preflight(monkeypatch, module):
    c = _app(monkeypatch, module, "*")
    r = c.options("/api/config/lines/x", headers={"Origin": DEV, "Access-Control-Request-Method": "PUT"})
    assert r.status_code == 200 and r.headers["access-control-allow-origin"] in (DEV, "*")
    assert "PUT" in r.headers["access-control-allow-methods"]


def test_whitelist_blocks_other_origins(monkeypatch):
    c = _app(monkeypatch, "edgevision.api.control", "https://a.example.com")
    r = c.options("/api/models", headers={"Origin": DEV, "Access-Control-Request-Method": "GET"})
    assert "access-control-allow-origin" not in r.headers


def test_parse():
    assert cors_origins(" https://a , http://b:5173 ,") == ["https://a", "http://b:5173"]
    assert cors_origins("") == []
