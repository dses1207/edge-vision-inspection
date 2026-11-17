"""錄影服務：預錄緩衝與事件片段。不需要 ffmpeg 或真串流。

測的是那些「錯了會靜默壞掉」的地方：
  · ring buffer 繞回後的排序（照檔名排會讓影片時間軸亂掉）
  · 冷卻要擋在錄影之前（舊系統的 bug 是只擋存圖）
  · 空間檢查（舊系統完全沒有，寫到磁碟滿為止）
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from edgevision.recorder.clip import ClipSpec
from edgevision.recorder.prebuffer import (
    BufferSpec, build_command, check_space, list_segments, segments_covering,
)


def _seg(d: Path, name: str, mtime: float) -> Path:
    p = d / name
    p.write_bytes(b"x")
    os.utime(p, (mtime, mtime))
    return p


# ── ring buffer 排序 ──────────────────────────────────────

def test_segments_sorted_by_time_not_name(tmp_path):
    """⚠️ ring buffer 會繞回：seg029 之後是 seg000。

    照檔名排會把最新的 seg000 排到最前面，concat 出來的影片時間軸是亂的 ——
    而且不會有任何錯誤，只是片段內容前後顛倒。
    """
    now = time.time()
    _seg(tmp_path, "seg028.mp4", now - 30)
    _seg(tmp_path, "seg029.mp4", now - 20)
    _seg(tmp_path, "seg000.mp4", now - 10)     # 繞回，檔名最小但時間最新
    assert [p.name for p in list_segments(tmp_path)] == [
        "seg028.mp4", "seg029.mp4", "seg000.mp4"]


def test_segments_covering_takes_enough_history(tmp_path):
    """往回撈要多撈一段 —— 事件發生在某段中間，只撈剛好會缺開頭。"""
    now = time.time()
    for i, age in enumerate([50, 40, 30, 20, 10]):
        _seg(tmp_path, f"seg{i:03d}.mp4", now - age)
    got = segments_covering(tmp_path, seconds_back=25, now=now)
    # 25 秒前落在 seg002(30s) 與 seg003(20s) 之間 → 必須含 seg002
    assert "seg002.mp4" in [p.name for p in got]
    assert got == sorted(got, key=lambda p: p.stat().st_mtime), "沒照時間排"


def test_segments_covering_empty_dir(tmp_path):
    assert segments_covering(tmp_path, seconds_back=10) == []


# ── ffmpeg 參數 ───────────────────────────────────────────

def test_command_has_critical_flags(tmp_path):
    """幾個參數不是可有可無的，少一個就會出事。"""
    cmd = " ".join(build_command(BufferSpec(
        camera_id="cam156", rtsp_url="rtsp://x/cam156", out_dir=tmp_path)))
    assert "-rtsp_transport tcp" in cmd, "UDP 丟包會讓分段破圖，且事後無法修"
    assert "-c copy" in cmd, "轉碼的話 8 路要吃掉好幾核"
    assert "-an" in cmd, "相機的 PCM 音軌用不到且拖慢 remux"
    assert "-reset_timestamps 1" in cmd, "不重設的話 concat 出來時間軸是錯的"
    assert "+faststart" in cmd, "moov 不在檔頭的話寫一半被砍就打不開"


def test_buffer_seconds_is_product():
    s = BufferSpec(camera_id="c", rtsp_url="r", out_dir=Path("/tmp"),
                   segment_seconds=10, segment_wrap=30)
    assert s.buffer_seconds == 300


def test_build_command_records_given_stream_verbatim(tmp_path):
    """錄哪條串流是 run_recorder 決定的（RECORD_SUFFIX，預設 _marked：框和 mask 燒好的那條）。

    build_command 不做任何改寫 —— 給它 _marked 就錄 _marked。
    「錄乾淨版之後再疊框」已經證實做不到（side-car 沒逐幀座標、PG 不存多邊形），別再改回去。
    """
    cmd = build_command(BufferSpec(camera_id="cam156",
                                   rtsp_url="rtsp://go2rtc:8554/cam156_marked",
                                   out_dir=tmp_path))
    assert "rtsp://go2rtc:8554/cam156_marked" in cmd


# ── 沒有冷卻（這是刻意的）────────────────────────────────

def test_no_cooldown_in_recorder():
    """⚠️ 守住一個結構保證：進到 stream 的每個 NG 都要有影片，跟 DB 一對一。

    去重是判定層的事（zone_intrusion station 內建 streak + 冷卻）。recorder 曾經
    自己再疊一層 30 秒冷卻 —— 同一件事擋兩次，而且第二道擋掉的【不留紀錄】，
    結果是「DB 有 NG、沒影片」。有人想加回來，這條測試會先擋。
    """
    import edgevision.recorder.clip as clip
    import inspect
    assert not hasattr(clip, "Cooldown"), "Cooldown 類別不該存在於 recorder"
    src = inspect.getsource(clip.ClipMaker)
    assert "cooldown" not in src.lower(), "ClipMaker 不該有任何冷卻邏輯"


def test_every_ng_gets_a_clip(tmp_path):
    """連續兩個 NG（間隔 1 秒）都要進到製作流程，不能被擋掉。"""
    from unittest.mock import patch
    from edgevision.recorder.clip import ClipMaker
    (tmp_path / "cam156").mkdir()
    maker = ClipMaker(objects=None, buffer_root=tmp_path)
    calls = []
    with patch.object(maker, "_build_and_upload", side_effect=lambda *a, **k: calls.append(a[1]) or "s3://x"), \
         patch("edgevision.recorder.clip.wait_for_segments", return_value=[(tmp_path / "cam156", 2000.0)]), \
         patch("edgevision.recorder.clip.readable", side_effect=lambda segs: segs), \
         patch("edgevision.recorder.clip.time.sleep"):
        maker.make("cam156", 1000.0, {})
        maker.make("cam156", 1001.0, {})     # 舊版會被 30 秒冷卻擋掉
    assert calls == [1000.0, 1001.0]


# ── 空間檢查 ──────────────────────────────────────────────

def test_check_space_returns_free(tmp_path):
    """⚠️ 舊系統沒有任何 disk_usage/statvfs 呼叫，寫到 NAS 滿為止。"""
    ok, free = check_space(tmp_path, min_free_gb=0.001)
    assert ok and free > 0


def test_check_space_rejects_when_low(tmp_path):
    ok, _ = check_space(tmp_path, min_free_gb=10 ** 9)
    assert not ok


def test_check_space_survives_bad_path():
    ok, free = check_space(Path("/nonexistent-xyz"), min_free_gb=1)
    assert not ok and free == 0.0


# ── 片段範圍 ──────────────────────────────────────────────

def test_clip_spec_defaults_match_legacy():
    """前 8 後 5 是舊系統的值，現場驗證過的節奏，不要憑感覺改。"""
    s = ClipSpec()
    assert (s.pre_seconds, s.post_seconds, s.total) == (8.0, 5.0, 13.0)


def test_probe_duration_survives_missing_ffprobe(tmp_path):
    """⚠️ 映像裡可能只有 ffmpeg 沒有 ffprobe（實測 edgevision:mvp 就是）。

    漏接 FileNotFoundError 的話整個片段製作會拋例外，
    而真正的原因（少一個執行檔）從 log 完全看不出來。
    """
    from edgevision.recorder.clip import _probe_duration

    f = tmp_path / "x.mp4"
    f.write_bytes(b"not a real mp4")
    assert _probe_duration(f) is None       # 不拋例外


# ── 輪替覆寫導致空片段（實測 261B）────────────────────────────


def test_offset_clamped_when_material_shorter_than_start(tmp_path, monkeypatch):
    """起點超過素材長度時要夾回去，不能原樣丟給 ffmpeg。

    緩衝是 30 段輪替覆寫：concat/probe 期間 segs[0] 可能已被蓋成新的一段，
    mtime 一變 timeline_start 就往後跳，offset 變成遠超素材長度。
    ffmpeg 這時仍 returncode=0，寫出只有容器沒有影格的 261B 檔 —— 能上傳、不能播。
    """
    from edgevision.recorder import clip as clipmod

    seen: dict = {}

    def fake_concat(segs, out):
        out.write_bytes(b"x")
        return True

    def fake_probe_duration(path):
        return 50.0          # 素材只有 50 秒

    def fake_trim(src, out, *, start, duration):
        seen["start"] = start
        out.write_bytes(b"y")
        return True

    monkeypatch.setattr(clipmod, "concat_segments", fake_concat)
    monkeypatch.setattr(clipmod, "_probe_duration", fake_probe_duration)
    monkeypatch.setattr(clipmod, "trim", fake_trim)

    seg = tmp_path / "seg000.mp4"
    seg.write_bytes(b"z")

    maker = clipmod.ClipMaker.__new__(clipmod.ClipMaker)
    maker.spec = clipmod.ClipSpec()
    maker.stats = clipmod.ClipStats()
    maker.bucket = "b"
    maker.objects = type("O", (), {"put": lambda self, **kw: None})()

    # first_end 落後於事件很久 → offset 會被算成遠大於 50 秒
    maker._build_and_upload("cam50", event_ts=10_000.0, payload={},
                            segs=[seg], first_end=1_000.0)

    assert seen["start"] < 50.0, f"offset 應被夾在素材長度內，實際 {seen['start']}"


def test_trim_rejects_frameless_output(tmp_path, monkeypatch):
    """ffmpeg 回 0 且檔案非空，但沒有影格時 trim 要回 False。"""
    from edgevision.recorder import clip as clipmod

    def fake_run(*a, **kw):
        out = Path(a[0][-1])
        out.write_bytes(b"\x00" * 261)      # 只有容器、沒有 mdat 內容
        return type("R", (), {"returncode": 0, "stderr": ""})()

    monkeypatch.setattr(clipmod.subprocess, "run", fake_run)
    monkeypatch.setattr(clipmod, "_probe_nb_frames", lambda p: 0)

    ok = clipmod.trim(tmp_path / "in.mp4", tmp_path / "out.mp4",
                      start=99.0, duration=13.0)
    assert ok is False, "沒有影格的輸出必須判定失敗"


# ── 2026-10-02：約三成 NG 沒影片的修正 ────────────────────────

def _seg(d, name, mtime):
    import os
    p = d / name
    p.write_bytes(b"x")
    os.utime(p, (mtime, mtime))
    return p


def test_wait_for_segments_skips_the_one_still_being_written(tmp_path):
    """最新那段 ffmpeg 還在寫、打不開，不能拿來剪。"""
    from edgevision.recorder.clip import wait_for_segments
    _seg(tmp_path, "seg001.mp4", 100.0)
    _seg(tmp_path, "seg002.mp4", 110.0)
    _seg(tmp_path, "seg003.mp4", 120.0)          # 還在寫
    got = wait_for_segments(tmp_path, until=105.0, max_wait=0)
    assert [p.name for p, _ in got] == ["seg001.mp4", "seg002.mp4"]


def test_wait_for_segments_waits_until_post_event_is_written(tmp_path):
    """事件後 5 秒所在的那段要寫完才剪；等不到就在期限到時用現有的。"""
    import threading
    import time as _t
    from edgevision.recorder.clip import wait_for_segments
    _seg(tmp_path, "seg001.mp4", 100.0)
    _seg(tmp_path, "seg002.mp4", 110.0)          # 還在寫
    threading.Timer(0.3, lambda: _seg(tmp_path, "seg003.mp4", 120.0)).start()   # seg002 寫完
    t0 = _t.time()
    got = wait_for_segments(tmp_path, until=109.0, max_wait=5, poll=0.05)
    assert [p.name for p, _ in got][-1] == "seg002.mp4" and _t.time() - t0 < 3


def test_prebuffer_switches_back_to_marked(tmp_path):
    """服務啟動時 runner 在重啟 → 暫錄原始串流；_marked 恢復後要自己切回去，不能永遠沒框。"""
    from edgevision.recorder.prebuffer import BufferSpec, PreRecordBuffer
    answers = iter([("rtsp://x/cam50", False), ("rtsp://x/cam50_marked", True)])
    b = PreRecordBuffer(BufferSpec(camera_id="cam50", rtsp_url="rtsp://x/cam50_marked", out_dir=tmp_path),
                        pick_url=lambda: next(answers), recheck_seconds=0.01)

    class P:
        def __init__(self):
            self.sig = None

        def poll(self):
            return None if self.sig is None else 0

        def send_signal(self, s):
            self.sig = s
    b._proc = P()
    b.on_preferred = False
    next(answers)                                # 第一次啟動已用掉「沒畫面」
    b._watch_for_preferred()
    assert b._proc.sig is not None               # 收到 SIGTERM → 迴圈會用 _marked 重啟


def test_unreadable_segment_is_dropped(tmp_path):
    """還在寫的分段（moov 還沒寫進去）不能拿去接 —— 它的 mtime 不一定最新。"""
    from unittest.mock import patch
    from edgevision.recorder.clip import readable
    a, b = _seg(tmp_path, "seg001.mp4", 100.0), _seg(tmp_path, "seg002.mp4", 110.0)
    with patch("edgevision.recorder.clip._probe_duration", side_effect=lambda p: 10.0 if p == a else None):
        assert readable([(a, 100.0), (b, 110.0)]) == [(a, 100.0)]


def test_encoder_prefers_gpu_then_falls_back(monkeypatch):
    """先 GPU（NVENC）、失敗退回 CPU；GPU 失敗後一段時間內直接用 CPU，不每支白試。"""
    from edgevision.recorder import clip
    monkeypatch.delenv("CLIP_ENCODER", raising=False)
    monkeypatch.setattr(clip, "_nvenc_down_until", 0.0)
    assert clip.encoder_order(now=1000) == ["nvenc", "x264"]
    monkeypatch.setattr(clip, "_nvenc_down_until", 1600.0)
    assert clip.encoder_order(now=1000) == ["x264"]
    assert clip.encoder_order(now=1700) == ["nvenc", "x264"]
    assert clip.encoder_order("x264") == ["x264"]


def test_trim_falls_back_to_cpu_when_nvenc_fails(tmp_path, monkeypatch):
    from unittest.mock import patch
    from edgevision.recorder import clip
    monkeypatch.setattr(clip, "_nvenc_down_until", 0.0)
    out = tmp_path / "c.mp4"
    calls = []

    class R:
        def __init__(self, rc):
            self.returncode, self.stderr = rc, "No capable devices found"

    def fake_run(cmd, **kw):
        calls.append("h264_nvenc" if "h264_nvenc" in cmd else "libx264")
        if calls[-1] == "libx264":
            out.write_bytes(b"x")
        return R(1 if calls[-1] == "h264_nvenc" else 0)
    with patch.object(clip.subprocess, "run", side_effect=fake_run), \
         patch.object(clip, "_probe_nb_frames", return_value=30):
        assert clip.trim(tmp_path / "j.mp4", out, start=1.0, duration=13)
    assert calls == ["h264_nvenc", "libx264"] and clip._nvenc_down_until > 0


# ── 2026-11-05：約 17% 事件影片片尾少一段（faststart 收尾讓 mtime 判斷失準） ──────────

def _list(d, name, segs, mtime=None):
    import os
    p = d / name
    p.write_text("".join(f"{s},{i * 10:.6f},{i * 10 + 10:.6f}\n" for i, s in enumerate(segs)))
    if mtime is not None:
        os.utime(p, (mtime, mtime))
    return p


def test_finished_list_beats_mtime_order(tmp_path):
    """剛寫完的 seg002 被 faststart 改寫、mtime 比還在寫的 seg003 晚。
    以前把 mtime 最新的 seg002 當成還在寫 → 等待提早結束、片尾少一段；現在看 ffmpeg 的清單。"""
    from edgevision.recorder.clip import wait_for_segments
    from edgevision.recorder.prebuffer import SEGMENT_LIST, completed_segments
    _seg(tmp_path, "seg001.mp4", 110.0)
    _seg(tmp_path, "seg003.mp4", 120.0)          # 還在寫
    _seg(tmp_path, "seg002.mp4", 120.4)          # 剛寫完，faststart 改寫 → mtime 反而最新
    _list(tmp_path, SEGMENT_LIST, ["seg001.mp4", "seg002.mp4"])
    assert [p.name for p in completed_segments(tmp_path)] == ["seg001.mp4", "seg002.mp4"]
    got = wait_for_segments(tmp_path, until=120.0, max_wait=0)
    assert [p.name for p, _ in got] == ["seg001.mp4", "seg002.mp4"]


def test_half_written_list_line_is_ignored(tmp_path):
    """ffmpeg 每次整份重寫清單，可能讀到寫一半的最後一行。"""
    from edgevision.recorder.prebuffer import SEGMENT_LIST, completed_segments
    _seg(tmp_path, "seg001.mp4", 110.0)
    _seg(tmp_path, "seg002.mp4", 120.0)
    (tmp_path / SEGMENT_LIST).write_text("seg001.mp4,0.000000,10.000000\nseg00")
    assert [p.name for p in completed_segments(tmp_path)] == ["seg001.mp4"]


def test_segments_from_before_a_restart_stay_usable(tmp_path):
    """相機斷線 → ffmpeg 重啟、清單從頭寫。上一個行程寫完的段落仍可用；
    已被新行程覆寫（mtime 比舊清單新）而且還沒寫完的不算。"""
    from edgevision.recorder.prebuffer import SEGMENT_LIST, completed_segments, rotate_segment_list
    _seg(tmp_path, "seg010.mp4", 90.0)
    _seg(tmp_path, "seg011.mp4", 100.0)
    _list(tmp_path, SEGMENT_LIST, ["seg010.mp4", "seg011.mp4"], mtime=100.0)
    rotate_segment_list(tmp_path)                # 新行程啟動前
    _seg(tmp_path, "seg000.mp4", 200.0)          # 新行程寫完的
    _seg(tmp_path, "seg001.mp4", 210.0)          # 新行程正在寫
    _list(tmp_path, SEGMENT_LIST, ["seg000.mp4"], mtime=200.0)
    assert [p.name for p in completed_segments(tmp_path)] == ["seg010.mp4", "seg011.mp4", "seg000.mp4"]

    _seg(tmp_path, "seg010.mp4", 215.0)          # 新行程繞一圈覆寫到 seg010、還沒寫完
    assert "seg010.mp4" not in [p.name for p in completed_segments(tmp_path)]


def test_wrapped_name_being_rewritten_is_not_done(tmp_path):
    """檔名輪流覆寫：seg000 上一輪寫完記在清單上，現在又在重錄（mtime 比清單新）→ 不算寫完。
    2026-11-05 上線後實測：環繞滿一圈（5 分鐘）後短片又出現，就是這個。"""
    from edgevision.recorder.clip import wait_for_segments
    from edgevision.recorder.prebuffer import SEGMENT_LIST, completed_segments
    _seg(tmp_path, "seg001.mp4", 110.0)
    _seg(tmp_path, "seg002.mp4", 120.0)
    _list(tmp_path, SEGMENT_LIST, ["seg000.mp4", "seg001.mp4", "seg002.mp4"], mtime=120.5)
    _seg(tmp_path, "seg000.mp4", 125.0)          # 下一輪正在重錄
    assert [p.name for p in completed_segments(tmp_path)] == ["seg001.mp4", "seg002.mp4"]
    got = wait_for_segments(tmp_path, until=124.0, max_wait=0)
    assert [p.name for p, _ in got][-1] == "seg002.mp4"


def test_ring_list_keeps_one_less_than_wrap():
    cmd = build_command(BufferSpec(camera_id="c", rtsp_url="rtsp://x/c", out_dir=Path("/b/c")))
    i = cmd.index("-segment_list_size")
    assert int(cmd[i + 1]) == int(cmd[cmd.index("-segment_wrap") + 1]) - 1


def test_short_clip_is_counted(tmp_path, monkeypatch):
    """長度不足照樣上傳，但記進 stats.short —— 短片率要看得到，不用人工抽查。"""
    from edgevision.recorder import clip as clipmod
    monkeypatch.setattr(clipmod, "concat_segments", lambda segs, out: out.write_bytes(b"j") or True)
    monkeypatch.setattr(clipmod, "trim", lambda src, out, **kw: out.write_bytes(b"c") or True)
    monkeypatch.setattr(clipmod, "_probe_duration", lambda p: 4.0 if p.name == "clip.mp4" else 30.0)
    seg = _seg(tmp_path, "seg000.mp4", 1000.0)
    maker = clipmod.ClipMaker.__new__(clipmod.ClipMaker)
    maker.spec, maker.stats, maker.bucket = clipmod.ClipSpec(), clipmod.ClipStats(), "b"
    maker.objects = type("O", (), {"put": lambda self, **kw: None})()
    assert maker._build_and_upload("cam50", event_ts=995.0, payload={}, segs=[seg], first_end=1000.0)
    assert maker.stats.short == 1 and maker.stats.snapshot()["short"] == 1


class _FakeGo2rtc:
    """假的 go2rtc /api/streams：每次被問就回下一個 bytes_recv；記錄打過哪些網址。"""

    def __init__(self, seq):
        self.seq, self.urls = list(seq), []

    def __call__(self, url, timeout=10):
        self.urls.append(url)
        v = self.seq.pop(0) if self.seq else None
        return {"producers": [] if v is None else [{"bytes_recv": v}], "consumers": []}


def test_probe_uses_stream_api_not_snapshot(monkeypatch):
    """探測只讀 /api/streams 的 bytes_recv，不打 frame.jpeg（那會在 go2rtc 留下永遠不清的 consumer）。"""
    import run_recorder as rr
    fake = _FakeGo2rtc([1000, 90000])
    monkeypatch.setattr(rr, "_get_json", fake)
    assert rr._stream_has_data("cam100", "cam100_marked", window=0) is True
    assert all("/api/streams?src=cam100_marked" in u for u in fake.urls)
    assert not any("frame" in u for u in fake.urls)


def test_probe_no_new_bytes_means_no_picture(monkeypatch):
    import run_recorder as rr
    monkeypatch.setattr(rr, "_get_json", _FakeGo2rtc([1000, 1000] * 3))
    monkeypatch.setattr(rr.time, "sleep", lambda s: None)
    assert rr._stream_has_data("cam100", "cam100_marked", window=0) is False


def test_probe_no_producer_means_no_picture(monkeypatch):
    import run_recorder as rr
    monkeypatch.setattr(rr, "_get_json", _FakeGo2rtc([None] * 6))
    monkeypatch.setattr(rr.time, "sleep", lambda s: None)
    assert rr._stream_has_data("cam215", "cam215_marked", window=0) is False


def test_probe_trickle_of_control_packets_is_not_picture(monkeypatch):
    """runner 卡死時 _marked 每 5 秒仍多 240 bytes（2026-11-09 實測），不能算有畫面。"""
    import run_recorder as rr
    monkeypatch.setattr(rr, "_get_json", _FakeGo2rtc([1000, 1240] * 3))
    monkeypatch.setattr(rr.time, "sleep", lambda s: None)
    assert rr._stream_has_data("cam100", "cam100_marked", window=0) is False
