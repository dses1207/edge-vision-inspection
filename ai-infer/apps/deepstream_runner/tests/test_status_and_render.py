import pytest
from ds_runner.config import load_line_config
from ds_runner.pipeline import render_nvinfer_config, resolve_interval
from ds_runner.status import LineReporter, status_url


def test_reporter_posts_reported_version_to_status_endpoint():
    sent = []
    r = LineReporter("http://output-api:8080", "alpha_a", 3, post=lambda url, payload: sent.append((url, payload)))
    r.cameras_running = 2
    assert r.send("ok") is True
    r.report_failure("boom")
    assert sent[0][0] == "http://output-api:8080/api/config/lines/alpha_a/status"
    assert sent[0][1] == {"reported_version": 3, "apply_state": "ok", "apply_error": None, "cameras_running": 2}
    assert sent[1][1]["apply_state"] == "failed" and sent[1][1]["apply_error"] == "boom"
    assert status_url("http://x/", "l") == "http://x/api/config/lines/l/status"


def test_reporter_disabled_without_config_api_and_survives_post_errors():
    assert LineReporter(None, "l", 1).send("ok") is False

    def boom(url, payload):
        raise ConnectionError
    assert LineReporter("http://x", "l", 1, post=boom).send("ok") is False


def test_render_template_and_interval_precedence(line_file):
    cfg = load_line_config({"CAMERA_SOURCE": "file", "CAMERA_FILE": str(line_file)})
    tmpl = 'model_name: "${MODEL_NAME}"\nurl: "${TRITON_GRPC}"\ninterval: ${INFER_INTERVAL}\n'
    out = render_nvinfer_config(tmpl, model_name="seg_alpha", triton_grpc="localhost:8001", interval=resolve_interval(cfg, {}))
    assert 'model_name: "seg_alpha"' in out and "interval: 5" in out
    assert resolve_interval(cfg, {"INFER_INTERVAL": "0"}) == 0
    cfg.line_params = {}
    assert resolve_interval(cfg, {}) == 0, "沒設定 = 每幀推論"


def test_triton_model_version_unknown_without_endpoint_or_on_error():
    from ds_runner.pipeline import triton_model_version
    assert triton_model_version(None, "seg_alpha") == "unknown"
    assert triton_model_version("127.0.0.1:1", "seg_alpha") == "unknown"     # 連不上也不炸


def test_render_outputs_follow_model_task():
    from ds_runner.pipeline import render_nvinfer_config
    tmpl = 'outputs: [\n${OUTPUT_LAYERS}\n]\n${MODEL_NAME} ${TRITON_GRPC} ${INFER_INTERVAL}'
    seg = render_nvinfer_config(tmpl, model_name="m", triton_grpc="t", interval=0, task="segment")
    pose = render_nvinfer_config(tmpl, model_name="m", triton_grpc="t", interval=0, task="pose")
    assert '{name: "output1"}' in seg and '{name: "output1"}' not in pose and '{name: "output0"}' in pose


# ── 心跳附上推論負載（給健康巡檢）────────────────────────────


def test_stats_provider_reports_interval_rates_not_lifetime_averages():
    """fps / sync_ms_avg 是「這一段間隔」的差分：健康巡檢要看現在有沒有變慢，不是啟動以來的平均。"""
    import queue

    from ds_runner.status import make_stats_provider

    t = [100.0]
    stats = {"frames": 1000, "sync_ms": 500.0, "sync_n": 1000, "dropped": 3, "mask_err": 0}
    q = queue.Queue()
    q.put(1); q.put(2)
    prov = make_stats_provider(stats, [q, queue.Queue()], clock=lambda: t[0])

    # 15 秒後多了 1800 幀（8 台 × 15fps）、同步路徑多花 1080ms
    t[0] = 115.0
    stats.update(frames=2800, sync_ms=1580.0, sync_n=2800, dropped=5)
    s = prov()
    assert s["fps"] == 120.0
    assert s["sync_ms_avg"] == 0.6
    assert s["dropped"] == 5 and s["worker_queue"] == 2

    # 沒有新幀 → fps 0、sync_ms_avg 不能除以零
    t[0] = 130.0
    s = prov()
    assert s["fps"] == 0.0 and s["sync_ms_avg"] is None


def test_stats_provider_reports_per_camera_fps_including_silent_ones():
    """相機層級：設定裡有、但一幀都沒來的相機（golf 404、cam156 被踢線）要是 0.0，不是缺席。"""
    from ds_runner.status import make_stats_provider

    t = [0.0]
    stats = {"cam_frames": {"cam156": 100}}
    prov = make_stats_provider(stats, clock=lambda: t[0], cameras=["cam156", "cam193", "ipcam235"])
    t[0] = 15.0
    stats["cam_frames"] = {"cam156": 100 + 225, "cam193": 150}      # cam156 15fps、cam193 10fps、ipcam235 沒來
    assert prov()["cameras"] == {"cam156": 15.0, "cam193": 10.0, "ipcam235": 0.0}


def test_reporter_counts_only_cameras_with_frames():
    from ds_runner.status import LineReporter

    sent = []
    rep = LineReporter("http://api", "ipcam_golf", 1, post=lambda url, p: sent.append(p),
                       stats_provider=lambda: {"cameras": {"ipcam235": 0.0, "cam193": 9.9}})
    rep.start(2)                              # 設定裡 2 台
    rep.stop()
    assert sent[-1]["cameras_running"] == 1   # 真的有幀的只有 1 台


def test_reporter_attaches_stats_and_survives_provider_errors():
    from ds_runner.status import LineReporter

    sent = []
    rep = LineReporter("http://api", "alpha_a", 3, post=lambda url, p: sent.append(p),
                       stats_provider=lambda: {"fps": 1.0})
    assert rep.send("ok") and sent[-1]["stats"] == {"fps": 1.0}

    def boom():
        raise RuntimeError("x")
    rep.stats_provider = boom
    assert rep.send("ok") and "stats" not in sent[-1]     # 統計壞了心跳照發


# ── 啟動閘門：engine 參數要跟設定一致 ────────────────────────


def _meta(h=640, feats=38, seg=True):
    outs = [{"name": "output0", "shape": [-1, feats, -1]}]
    if seg:
        outs.append({"name": "output1", "shape": [-1, 32, -1, -1]})
    return {"name": "m", "versions": ["1"], "inputs": [{"name": "images", "shape": [-1, 3, h, h]}], "outputs": outs}


def test_check_model_matches_accepts_matching_seg_model():
    from ds_runner.pipeline import check_model_matches
    assert check_model_matches(_meta(), model_name="seg_alpha", imgsz=640, n_classes=2, task="segment") == []


def test_check_model_matches_flags_imgsz_and_class_count():
    from ds_runner.pipeline import check_model_matches
    p = check_model_matches(_meta(h=640, feats=10, seg=False), model_name="delta", imgsz=1024, n_classes=6, task="detect")
    assert any("imgsz=1024" in x for x in p)                       # engine 640、設定 1024
    p = check_model_matches(_meta(h=1024, feats=9, seg=False), model_name="delta", imgsz=1024, n_classes=6, task="detect")
    assert any("類別數對不上" in x for x in p)                     # 4+6=10 ≠ 9
    p = check_model_matches(_meta(seg=False), model_name="seg_alpha", imgsz=640, n_classes=2, task="segment")
    assert any("output1" in x for x in p)                          # seg 缺 prototypes


def test_verify_models_exits_on_mismatch_and_reports(write_line, monkeypatch):
    import ds_runner.pipeline as pl
    cfg = load_line_config(write_line())
    monkeypatch.setattr(pl, "triton_model_metadata", lambda http, name: _meta(h=1024))
    with pytest.raises(SystemExit, match="拒絕啟動"):
        pl.verify_models_against_triton(cfg, {"TRITON_HTTP": "x", "MODEL_NAME": "seg_alpha"})
    pl.verify_models_against_triton(cfg, {"TRITON_HTTP": "x", "MODEL_NAME": "seg_alpha", "MODEL_CHECK": "0"})   # 可關
    monkeypatch.setattr(pl, "triton_model_metadata", lambda http, name: _meta())
    pl.verify_models_against_triton(cfg, {"TRITON_HTTP": "x", "MODEL_NAME": "seg_alpha"})                     # 一致就過


def test_reporter_sends_skipped_cameras_every_heartbeat():
    sent = []
    rep = LineReporter("http://api", "webcam_bravo", 3, post=lambda url, p: sent.append(p))
    rep.skipped_cameras = {"cam211": "ValueError: roi_regions 少了 work 區"}
    rep.send("ok")
    assert sent[-1]["apply_state"] == "ok"                       # 線路本身是好的，壞的只有那一台
    assert sent[-1]["stats"]["skipped_cameras"] == {"cam211": "ValueError: roi_regions 少了 work 區"}


def test_bad_config_waits_instead_of_exiting(monkeypatch):
    """設定沒填完（例：沒畫 ROI）→ 不結束程序（以前 exit(2) → K8s 重啟兩百多次），待命、持續回報 failed，
    重讀到可用的設定就照常往下走。"""
    import ds_runner.pipeline as pl
    sent: list[str] = []
    monkeypatch.setattr(pl.LineReporter, "report_failure", lambda self, err: sent.append(err))
    import types
    good = types.SimpleNamespace(config_version=7)
    attempts = iter([ValueError("還是沒 ROI"), ValueError("還是沒 ROI"), good])

    def fake_load(env):
        r = next(attempts)
        if isinstance(r, Exception):
            raise r
        return r
    monkeypatch.setattr(pl, "load_line_config", fake_load)
    now = [0.0]

    def sleep(s):
        now[0] += s
    cfg = pl.wait_for_line_config({"LINE_ID": "x"}, None, ValueError("沒 ROI"), retry_s=60,
                                  heartbeat_s=15, sleep=sleep, clock=lambda: now[0])
    assert cfg is good
    assert now[0] == 180                       # 重讀三次（60s 一次），沒有提早放棄
    assert len(sent) >= 10 and all("ROI" in e for e in sent)   # 待命期間每 15 秒回報，心跳不斷


def test_video_test_still_fails_fast_on_bad_config(monkeypatch):
    import ds_runner.pipeline as pl
    monkeypatch.setattr(pl.LineReporter, "report_failure", lambda self, err: None)
    monkeypatch.setattr(pl, "load_line_config", lambda env: (_ for _ in ()).throw(ValueError("沒 ROI")))
    monkeypatch.setattr(pl, "wait_for_line_config", lambda *a, **k: pytest.fail("影片測試不該待命"))
    monkeypatch.setenv("VIDEO_FILE", "/x.mp4")
    with pytest.raises(SystemExit) as e:
        pl.main()
    assert e.value.code == 2
