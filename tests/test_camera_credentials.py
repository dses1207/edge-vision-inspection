"""cred_ref 解析：none 不帶帳密、指名的 ref 優先、退回預設時要警告、全無報錯。"""
import logging

import pytest

from edgevision.config import camera_source
from edgevision.config.camera_source import CameraConfig, resolve_credentials


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for k in ("CAM_USER", "CAM_PASS", "CAM_VISUAL_USER", "CAM_VISUAL_PASS", "CAM_ALPHA_USER", "CAM_ALPHA_PASS"):
        monkeypatch.delenv(k, raising=False)
    camera_source._warned_fallback.clear()


def test_none_means_no_auth_in_url(monkeypatch):
    monkeypatch.setenv("CAM_USER", "admin")
    monkeypatch.setenv("CAM_PASS", "pw")
    assert resolve_credentials("none") == ("", "")
    assert resolve_credentials("NONE") == ("", "")
    cam = CameraConfig(id="ipcam198", ip="192.0.2.55", width=1920, height=1080, stream_path="/stream", cred_ref="none")
    assert cam.rtsp_url == "rtsp://192.0.2.55/stream"


def test_none_works_without_any_env():
    assert resolve_credentials("none") == ("", "")          # 不需要 CAM_USER 存在


def test_named_ref_wins(monkeypatch):
    monkeypatch.setenv("CAM_USER", "admin")
    monkeypatch.setenv("CAM_VISUAL_USER", "vis")
    monkeypatch.setenv("CAM_VISUAL_PASS", "vpw")
    assert resolve_credentials("visual") == ("vis", "vpw")


def test_fallback_to_default_warns_once(monkeypatch, caplog):
    monkeypatch.setenv("CAM_USER", "admin")
    monkeypatch.setenv("CAM_PASS", "pw")
    with caplog.at_level(logging.WARNING, logger="edgevision.config.camera_source"):
        assert resolve_credentials("alpha") == ("admin", "pw")
        assert resolve_credentials("alpha") == ("admin", "pw")
    warns = [r for r in caplog.records if "cred_ref='alpha'" in r.getMessage()]
    assert len(warns) == 1


def test_empty_ref_falls_back_silently(monkeypatch, caplog):
    monkeypatch.setenv("CAM_USER", "admin")
    with caplog.at_level(logging.WARNING, logger="edgevision.config.camera_source"):
        assert resolve_credentials("") == ("admin", "")
    assert not caplog.records


def test_missing_everything_is_fatal_and_mentions_none():
    with pytest.raises(SystemExit, match="none"):
        resolve_credentials("ipcam")
