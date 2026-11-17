"""名單 API 的 daily_gb（2026-11-03：請求當下列 NAS 目錄 4～9 秒 → 前端逾時；改成背景算、API 只讀記憶體）。"""
import os
import time
from datetime import datetime, timedelta

from edgevision.media import recordings as R
from edgevision.media.frames import LOCAL_TZ


def _seg(d, start: datetime, size: int, mtime: float):
    p = d / start.strftime("%Y%m%d_%H%M%S.mp4")
    p.write_bytes(b"x" * size)
    os.utime(p, (mtime, mtime))


def test_daily_gb_counts_only_last_24h(tmp_path, monkeypatch):
    monkeypatch.setenv("MEDIA_NAS_ROOT", str(tmp_path))
    cam = tmp_path / "siteb12"
    cam.mkdir()
    now = time.time()
    t = datetime.fromtimestamp(now, LOCAL_TZ)
    _seg(cam, t - timedelta(hours=2), 1_000_000_000, now - 3600)
    _seg(cam, t - timedelta(hours=30), 5_000_000_000, now - 29 * 3600)
    assert R.daily_gb("siteb12") == 1.0
    assert R.daily_gb("nope") is None


def test_daily_usage_api_never_touches_nas():
    calls = []
    du = R.DailyUsage(lambda: ["siteb12", "ipcam87"], refresh_s=300,
                      compute=lambda c: (calls.append(c), {"siteb12": 3.2}.get(c))[1])
    assert du.get("siteb12") is None and calls == []          # 還沒算：回 None，不在請求裡算
    du.refresh()
    assert calls == ["siteb12", "ipcam87"] and du.updated_at is not None
    calls.clear()
    assert du.get("siteb12") == 3.2 and du.get("ipcam87") is None and calls == []


def test_daily_usage_one_bad_camera_does_not_break_others():
    def compute(c):
        if c == "bad":
            raise OSError("NAS 掛了")
        return 1.5
    du = R.DailyUsage(lambda: ["bad", "ok"], compute=compute)
    du.refresh()
    assert du.get("bad") is None and du.get("ok") == 1.5
