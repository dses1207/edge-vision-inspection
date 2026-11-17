"""go2rtc 串流產生器的帳密：none 不帶帳密、指名 ref 優先、退回預設要警告。"""
import importlib.util
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location(
    "gen_go2rtc_streams", Path(__file__).resolve().parents[1] / "tools" / "gen_go2rtc_streams.py")
gen = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gen)


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    for k in ("CAM_USER", "CAM_PASS", "CAM_VISUAL_USER", "CAM_VISUAL_PASS"):
        monkeypatch.delenv(k, raising=False)
    gen._warned.clear()


def test_none_gives_url_without_auth(monkeypatch):
    monkeypatch.setenv("CAM_USER", "admin")
    user, pw = gen.creds("none")
    assert (user, pw) == ("", "")
    assert gen.rtsp_url(user, pw, "192.0.2.55", "/stream") == "rtsp://192.0.2.55:554/stream"


def test_named_ref_and_auth_in_url(monkeypatch):
    monkeypatch.setenv("CAM_VISUAL_USER", "vis")
    monkeypatch.setenv("CAM_VISUAL_PASS", "p")
    user, pw = gen.creds("visual")
    assert gen.rtsp_url(user, pw, "192.0.2.84", "/stream2") == "rtsp://vis:p@192.0.2.84:554/stream2"


def test_fallback_warns_once(monkeypatch, capsys):
    monkeypatch.setenv("CAM_USER", "admin")
    assert gen.creds("alpha") == ("admin", "")
    assert gen.creds("alpha") == ("admin", "")
    assert capsys.readouterr().err.count("cred_ref='alpha'") == 1


def test_missing_all_is_fatal():
    with pytest.raises(SystemExit, match="none"):
        gen.creds("ipcam")


def test_source_opts_disable_onvif_backchannel():
    # IPCAM 相機收到 backchannel 要求時會回錯誤的 SDP（H.265 寫成 H264）
    assert gen.SRC_OPTS.startswith("#backchannel=0")
