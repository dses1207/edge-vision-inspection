"""24h 連續錄影的測試。

這裡測的每一條都對應實際踩過的一個坑 ——
那些坑都是舊系統實測踩出來的，靠註解守不住，要有測試。
"""
from __future__ import annotations

import time
from pathlib import Path

import pytest

from edgevision.recorder.continuous import (
    MAX_RETRY_SECONDS,
    MIN_RETRY_SECONDS,
    STABLE_RUN_SECONDS,
    ContinuousSpec,
    cleanup_old,
    dir_size,
)
from edgevision.recorder.prebuffer import BufferSpec, build_command


def _ring(tmp: Path) -> BufferSpec:
    return BufferSpec(camera_id="cam156", rtsp_url="rtsp://h/cam156_marked",
                      out_dir=tmp, segment_seconds=10, segment_wrap=30)


def _cont(tmp: Path) -> BufferSpec:
    return BufferSpec(camera_id="cam156", rtsp_url="rtsp://h/cam156",
                      out_dir=tmp, segment_seconds=600, segment_wrap=0,
                      wallclock_timestamps=True)


# ── 模式切換 ──────────────────────────────────────────────

def test_segment_wrap_zero_means_continuous(tmp_path):
    assert _cont(tmp_path).continuous is True
    assert _ring(tmp_path).continuous is False


def test_continuous_has_no_buffer_seconds(tmp_path):
    """連續錄影沒有「緩衝總長度」的概念，不該回一個誤導的數字。"""
    assert _cont(tmp_path).buffer_seconds == 0
    assert _ring(tmp_path).buffer_seconds == 300


# ──  牆鐘時戳 ────────────────────────────────────────

def test_continuous_uses_wallclock_timestamps(tmp_path):
    """相機靜態時會把幀率壓到 1~2 fps，用來源時戳會讓 10 分鐘的檔橫跨 77 分鐘。"""
    cmd = build_command(_cont(tmp_path))
    assert "-use_wallclock_as_timestamps" in cmd
    assert cmd[cmd.index("-use_wallclock_as_timestamps") + 1] == "1"


def test_wallclock_flag_comes_before_input(tmp_path):
    """它是【輸入端】選項，放在 -i 後面等於沒設。"""
    cmd = build_command(_cont(tmp_path))
    assert cmd.index("-use_wallclock_as_timestamps") < cmd.index("-i")


def test_no_genpts_alongside_wallclock(tmp_path):
    """+genpts 會跟 -use_wallclock_as_timestamps 打架。"""
    assert "+genpts" not in " ".join(build_command(_cont(tmp_path)))


def test_prebuffer_also_uses_wallclock(tmp_path):
    """事件預錄也要牆鐘時戳（2026-10-02）：剪輯用「mtime − 段落長度」推時間軸，
    <cam>_marked 幀率浮動時媒體時間跟牆鐘對不上，起點算錯會剪出空檔。"""
    assert "-use_wallclock_as_timestamps" in build_command(_ring(tmp_path))


# ──  faststart 只給環形緩衝 ──────────────────────────

def test_continuous_has_no_faststart(tmp_path):
    """faststart 等於整檔再改寫一次，在 CIFS 上很貴，而連續錄影沒人要 concat。"""
    assert "+faststart" not in build_command(_cont(tmp_path))


def test_ring_keeps_faststart(tmp_path):
    """環形緩衝的段落要被 concat 成事件影片，需要 moov 在檔頭。"""
    assert "+faststart" in build_command(_ring(tmp_path))


def test_ring_writes_finished_list_but_continuous_does_not(tmp_path):
    """哪幾段寫完由 ffmpeg 的 -segment_list 說；連續錄影沒人剪，不需要。"""
    ring = build_command(_ring(tmp_path))
    assert "-segment_list" in ring and ring[ring.index("-segment_list_type") + 1] == "csv"
    assert "-segment_list" not in build_command(_cont(tmp_path))


# ── 檔名與覆寫 ────────────────────────────────────────────

def test_continuous_names_by_wallclock(tmp_path):
    cmd = build_command(_cont(tmp_path))
    assert "-strftime" in cmd
    assert cmd[-1].endswith("%Y%m%d_%H%M%S.mp4")


def test_continuous_never_wraps(tmp_path):
    """有 -segment_wrap 就會覆寫 —— 24h 錄影絕不能覆寫。"""
    assert "-segment_wrap" not in build_command(_cont(tmp_path))


def test_continuous_records_clean_stream(tmp_path):
    """這批影像要回流訓練，有框就廢了。"""
    assert "_marked" not in build_command(_cont(tmp_path))[
        build_command(_cont(tmp_path)).index("-i") + 1]


def test_both_modes_copy_without_transcoding(tmp_path):
    for spec in (_ring(tmp_path), _cont(tmp_path)):
        cmd = build_command(spec)
        assert cmd[cmd.index("-c") + 1] == "copy"


# ──  保留天數 ────────────────────────────────────────

def _touch(p: Path, *, age_days: float, size: int = 1000) -> Path:
    p.write_bytes(b"x" * size)
    old = time.time() - age_days * 86400
    import os
    os.utime(p, (old, old))
    return p


def test_retention_zero_deletes_nothing(tmp_path):
    """0 = 不刪（與舊系統一致）。誤刪 24h 原始影像是不可逆的。"""
    _touch(tmp_path / "20260101_000000.mp4", age_days=999)
    n, freed = cleanup_old(tmp_path, retention_days=0)
    assert (n, freed) == (0, 0)
    assert len(list(tmp_path.glob("*.mp4"))) == 1


def test_retention_negative_deletes_nothing(tmp_path):
    """設成負數也不該變成「刪光」。"""
    _touch(tmp_path / "old.mp4", age_days=999)
    assert cleanup_old(tmp_path, retention_days=-5) == (0, 0)


def test_retention_deletes_only_expired(tmp_path):
    _touch(tmp_path / "old.mp4", age_days=10, size=500)
    _touch(tmp_path / "new.mp4", age_days=1)
    n, freed = cleanup_old(tmp_path, retention_days=7)
    assert n == 1 and freed == 500
    assert [p.name for p in tmp_path.glob("*.mp4")] == ["new.mp4"]


def test_retention_ignores_non_mp4(tmp_path):
    """別把別人的東西刪掉。"""
    (tmp_path / "notes.txt").write_text("keep me")
    _touch(tmp_path / "old.mp4", age_days=99)
    cleanup_old(tmp_path, retention_days=1)
    assert (tmp_path / "notes.txt").exists()


# ──  看門狗的參數 ────────────────────────────────────

def test_stall_threshold_not_below_ffmpeg_exit():
    """⚠️ 這條測試在守一個很貴的教訓。

    上游斷線時 ffmpeg 自己約 27~30 秒會退出。停滯門檻若設 30 秒會跟它賽跑，
    搶在它前面砍掉行程 —— 一次乾淨重連就變成一個沒有 moov 的壞檔。
    舊系統一天生出 58 個打不開的檔就是這樣來的。
    """
    assert ContinuousSpec().stall_seconds >= 60


def test_retry_backoff_bounds():
    """跑久才斷要立刻重連（缺口最小），一啟動就掛才退避（避免洗版）。"""
    assert MIN_RETRY_SECONDS < MAX_RETRY_SECONDS
    assert MIN_RETRY_SECONDS <= 2.0
    assert STABLE_RUN_SECONDS > 0


def test_dir_size_counts_only_mp4(tmp_path):
    (tmp_path / "a.mp4").write_bytes(b"x" * 100)
    (tmp_path / "b.txt").write_bytes(b"y" * 999)
    assert dir_size(tmp_path) == 100


def test_dir_size_missing_dir_is_zero_not_crash():
    """看門狗會一直呼叫它，NAS 斷掉時不能讓它爆掉。"""
    assert dir_size(Path("/nonexistent/nope")) == 0


# ── 預設值對照規格 ───────────────────────────────────────

@pytest.mark.parametrize("field,expected", [
    ("segment_seconds", 600),     # 10 分鐘一段
    ("retention_days", 0),        # 不刪，與舊系統一致
    ("min_free_gb", 20.0),
    ("stall_seconds", 90.0),
])
def test_defaults_match_spec(field, expected):
    assert getattr(ContinuousSpec(), field) == expected


# ── 相機篩選（REC_CAMERAS，2026-10-08 先只錄 IPCAM）─────────

def test_filter_cameras_prefix_and_exact():
    from run_nas_recorder import filter_cameras
    cams = ["ipcam87", "ipcam179", "siteb12", "siteb197", "cam156", "cam137"]
    # 正式叢集的目標設定：IPCAM 兩組前綴，ALPHA（cam156 起的幾台）要被排除
    assert filter_cameras(cams, "ipcam*,siteb*") == ["ipcam87", "ipcam179", "siteb12", "siteb197"]
    # 精確比對：單錄一台
    assert filter_cameras(cams, "siteb12") == ["siteb12"]
    # 空 spec = 全錄（維持舊行為，compose 版沒設這個變數）
    assert filter_cameras(cams, "") == cams
    assert filter_cameras(cams, "  ") == cams


def test_filter_cameras_siteb_prefix_does_not_eat_cam():
    """`siteb*` 不能吃到 cam156/cam137 —— 錄錯幾十台就是容量災難。"""
    from run_nas_recorder import filter_cameras
    assert filter_cameras(["cam156", "camt1"], "siteb*") == []


def test_filter_cameras_no_match_is_empty_not_all():
    """篩到一台不剩要回空清單（main 會拒絕啟動），不准默默退回全錄。"""
    from run_nas_recorder import filter_cameras
    assert filter_cameras(["cam156"], "ipcam*") == []


# ── 存活檢查（2026-11-09 掃 NAS 超時，每 20 分鐘被重啟）─────

class _Rec:
    def __init__(self, last_progress):
        self.last_progress = last_progress


def test_alive_when_any_camera_recent():
    """一台壞（147_80 那種）不能讓整個服務被判死。"""
    from run_nas_recorder import recording_alive
    assert recording_alive({"bad": _Rec(0.0), "ok": _Rec(990.0)}, 300, now=1000.0) is True


def test_not_alive_when_every_camera_stale():
    from run_nas_recorder import recording_alive
    assert recording_alive({"a": _Rec(100.0), "b": _Rec(0.0)}, 300, now=1000.0) is False


def test_empty_roster_counts_as_alive():
    """名單空著是合法的待命，不該被重啟。"""
    from run_nas_recorder import recording_alive
    assert recording_alive({}, 300, now=1000.0) is True


def test_touch_alive_failure_does_not_crash(tmp_path):
    from run_nas_recorder import touch_alive
    touch_alive(tmp_path / "alive")
    assert (tmp_path / "alive").exists()
    touch_alive(tmp_path / "missing" / "alive")


def test_watch_records_progress_only_when_output_changes(tmp_path, monkeypatch):
    from edgevision.recorder import continuous as c

    sizes = iter([10, 10, 20, 20])
    monkeypatch.setattr(c, "dir_size", lambda _p: next(sizes))
    rec = c.ContinuousRecorder(_cont(tmp_path), ContinuousSpec())
    monkeypatch.setattr(rec._stop, "wait", lambda _t: False)

    class _Proc:
        polls = iter([None, None, None, 0])

        def poll(self):
            return next(self.polls)

    assert rec.last_progress == 0.0
    assert rec._watch(_Proc()) is False
    assert rec.last_progress > 0.0


def test_terminate_prefers_sigterm_so_moov_gets_written():
    """停滯重連要 SIGTERM（ffmpeg 收尾寫 moov），不肯走才 SIGKILL。2026-11-03 cam137 壞檔。"""
    import subprocess
    import sys

    from edgevision.recorder.continuous import terminate
    p = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    assert terminate(p, 5) is True and p.returncode == -15
    stubborn = subprocess.Popen([sys.executable, "-c",
                                 "import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); print('r', flush=True); time.sleep(60)"],
                                stdout=subprocess.PIPE, text=True)
    stubborn.stdout.readline()
    assert terminate(stubborn, 0.5) is False and stubborn.returncode == -9
