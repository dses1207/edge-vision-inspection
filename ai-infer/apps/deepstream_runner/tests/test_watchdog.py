"""啟動看門狗與 nvinferserver batch（2026-10-14 正式 ipcam_golf：GPU 切片被吃光，管線卡住卻回報 ok）。"""
from ds_runner.pipeline import nvinfer_batch
from ds_runner.status import LineReporter
from ds_runner.watchdog import EXIT_STALLED, StartupWatchdog, diagnose


def test_nvinfer_batch_is_capped_at_camera_count():
    assert nvinfer_batch(32, 1) == 1        # ppe：engine 32、線上 1 台 → 不要預先配 32 份 1280² 緩衝
    assert nvinfer_batch(4, 19) == 4        # 相機比 engine 多：由啟動檢查擋，這裡不放大
    assert nvinfer_batch(8, 8) == 8


def test_diagnose():
    gpu = (0, 48000)
    assert diagnose({"cam_frames": {"c": 3}}, 120, gpu)[0] == "ok"
    v, why = diagnose({"mux_batches": 50}, 120, gpu)
    assert v == "stalled" and "推論" in why and "GPU 剩 0 MiB" in why
    assert diagnose({"src_err": 4}, 120, gpu)[0] == "camera"
    v, why = diagnose({}, 120, None)
    assert v == "stalled" and "查不到" in why


class FakeReporter:
    def __init__(self):
        self.state, self.error, self.sent = "applying", None, []

    def set_state(self, state, error=None):
        self.state, self.error = state, error

    def send(self, state, error=None):
        self.sent.append((state, error))


def make(stats, t):
    exits, rep = [], FakeReporter()
    wd = StartupWatchdog(stats, rep, 120, clock=lambda: t[0], gpu=lambda: (12, 48000), exit_fn=exits.append)
    return wd, rep, exits


def test_first_frame_turns_applying_into_ok():
    stats, t = {}, [0.0]
    wd, rep, exits = make(stats, t)
    assert not wd.check()
    stats["cam_frames"] = {"ipcam235": 1}
    assert wd.check() and rep.state == "ok" and not exits


def test_stalled_inference_reports_failed_and_exits():
    stats, t = {"mux_batches": 900}, [0.0]
    wd, rep, exits = make(stats, t)
    t[0] = 119
    assert not wd.check()
    t[0] = 121
    assert wd.check()
    assert rep.state == "failed" and "GPU 剩 12 MiB" in rep.error and rep.sent[-1][0] == "failed"
    assert exits == [EXIT_STALLED]


def test_camera_down_is_not_a_runner_failure():
    stats, t = {"src_err": 3}, [0.0]
    wd, rep, exits = make(stats, t)
    t[0] = 200
    assert wd.check() and rep.state == "ok" and not exits


def test_reporter_heartbeat_keeps_state_and_adds_gpu():
    posts = []
    r = LineReporter("http://x", "l", 2, post=lambda u, p: posts.append(p))
    r.gpu = lambda: (100, 48000)
    r.start(cameras_running=1, state="applying")
    r.set_state("failed", "卡住")
    r.send(r.state, r.error)
    r.stop()
    assert posts[0]["apply_state"] == "applying"
    assert posts[-1]["apply_state"] == "failed" and posts[-1]["apply_error"] == "卡住"
    assert posts[-1]["stats"]["gpu_free_mib"] == 100


def test_model_version_must_match_registry():
    from ds_runner.pipeline import check_model_version
    assert check_model_version({"versions": ["1"]}, model_name="ppe", version=1) == []
    assert check_model_version({"versions": ["1"]}, model_name="ppe", version=None) == []
    assert "還沒跟上" in check_model_version({"versions": ["1"]}, model_name="ppe", version=2)[0]


# ── 執行中看門狗（go2rtc / redis 被趕走後管線長時間卡住，Pod 一直 Running）──

def make_run(stats, t):
    from ds_runner.watchdog import RunWatchdog
    exits, rep = [], FakeReporter()
    return RunWatchdog(stats, rep, 180, clock=lambda: t[0], exit_fn=exits.append), rep, exits


def test_run_watchdog_waits_for_first_frame():
    stats, t = {}, [0.0]
    wd, _, exits = make_run(stats, t)
    t[0] = 10_000
    assert not wd.check() and not exits          # 一幀都還沒有：啟動看門狗管


def test_run_watchdog_exits_when_frames_stop_without_camera_errors():
    stats, t = {"frames": 100}, [0.0]
    wd, rep, exits = make_run(stats, t)
    assert not wd.check()
    t[0] = 100; stats["frames"] = 200
    assert not wd.check()                         # 還在動
    t[0] = 279
    assert not wd.check()
    t[0] = 281
    assert wd.check()
    assert exits == [EXIT_STALLED] and rep.state == "failed" and "200" in rep.error


def test_run_watchdog_tolerates_camera_reconnects():
    stats, t = {"frames": 100, "src_err": 0}, [0.0]
    wd, _, exits = make_run(stats, t)
    wd.check()
    t[0] = 150; stats["src_err"] = 2               # 相機斷了在重連
    t[0] = 500
    assert not wd.check() and not exits
    stats["frames"] = 101                          # 恢復後重新計時
    assert not wd.check()
    t[0] = 690
    assert wd.check() and exits == [EXIT_STALLED]


def _cam_dog(stats, cams, stall=90.0):
    from ds_runner.watchdog import CameraStallWatchdog
    t = [0.0]
    return CameraStallWatchdog(stats, cams, stall, clock=lambda: t[0], max_s=360.0), t


def test_one_stuck_camera_is_reported_others_untouched():
    stats = {"frames": 10, "cam_frames": {"a": 5, "b": 5}}
    d, t = _cam_dog(stats, ["a", "b"])
    assert d.check() == []
    t[0] = 50.0
    stats["cam_frames"]["b"] = 9
    assert d.check() == []
    t[0] = 95.0
    stats["cam_frames"]["b"] = 12
    assert d.check() == [("a", 95.0)]


def test_single_camera_line_is_still_rebuilt():
    """工序W只有 cam215：「全部都停就不管」會漏掉它。"""
    stats = {"frames": 10, "cam_frames": {"cam215": 5}}
    d, t = _cam_dog(stats, ["cam215"])
    d.check()
    t[0] = 91.0
    assert [c for c, _ in d.check()] == ["cam215"]


def test_all_cameras_stopped_is_left_to_run_watchdog():
    stats = {"frames": 10, "cam_frames": {"a": 5, "b": 5}}
    d, t = _cam_dog(stats, ["a", "b"])
    d.check()
    t[0] = 200.0
    assert d.check() == []


def test_backoff_doubles_and_resets_on_new_frames():
    stats = {"frames": 10, "cam_frames": {"a": 5, "b": 5}}
    d, t = _cam_dog(stats, ["a", "b"])
    d.check()
    def tick(now):
        t[0] = now
        stats["cam_frames"]["b"] += 1
        return [c for c, _ in d.check()]
    assert tick(90.0) == ["a"]
    assert tick(179.0) == []          # 門檻已變 180
    assert tick(270.0) == ["a"]
    stats["cam_frames"]["a"] = 6      # 救回來
    assert tick(280.0) == []
    assert tick(371.0) == ["a"]       # 門檻回到 90


def test_camera_that_never_produced_counts_as_stuck():
    stats = {"frames": 10, "cam_frames": {"b": 5}}
    d, t = _cam_dog(stats, ["a", "b"])
    t[0] = 50.0
    stats["cam_frames"]["b"] = 8
    d.check()
    t[0] = 91.0
    stats["cam_frames"]["b"] = 11
    assert [c for c, _ in d.check()] == ["a"]


def test_nothing_before_first_frame_and_zero_disables():
    d, t = _cam_dog({"frames": 0, "cam_frames": {}}, ["a"])
    t[0] = 999.0
    assert d.check() == []
    d, t = _cam_dog({"frames": 10, "cam_frames": {"a": 1}}, ["a"], stall=0.0)
    t[0] = 999.0
    assert d.check() == []

