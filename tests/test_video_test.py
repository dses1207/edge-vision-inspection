"""影片測試：切影片清單、線路設定改寫、摘要、API 參數、Orchestrator 步驟順序。"""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from edgevision.media.video_test import build_line, class_stats, concat_list, summarize, work_dir
from edgevision.media.video_test_api import VideoTestIn
from libs.contracts.training import TaskType
from tests.test_orchestrator import FakeRunner, MemStore, _orch


def test_concat_list_inpoint_outpoint():
    txt = concat_list([(Path("/nas/raw/siteb216/a.mp4"), 30.0, None), (Path("/nas/raw/siteb216/b's.mp4"), 0.0, 12.5)])
    assert txt.splitlines() == ["file '/nas/raw/siteb216/a.mp4'", "inpoint 30.000",
                                "file '/nas/raw/siteb216/b'\\''s.mp4'", "outpoint 12.500"]


def _meta(name, classes, version):
    return SimpleNamespace(meta=SimpleNamespace(model_name=name, task="segment", imgsz=640, class_names=classes),
                           version=version, source_uri=f"s3://models/{name}/v{version}/model.plan")


class FakeRegistry:
    def __init__(self):
        self.calls = []

    def get(self, name, version):
        self.calls.append((name, version))
        return _meta(name, ["board", "hand"], version)

    def resolve(self, name):
        self.calls.append((name, None))
        return _meta(name, ["person"], 1)


LINE = {"id": "siteb_echo", "station": "echo_check", "config_version": 4,
        "model": {"name": "siteb_echo", "version": 1, "classes": ["old"], "conf": 0.4},
        "extra_stations": [{"station": "ppe_check", "model": {"name": "ppe", "version": 1}}],
        "cameras": [{"id": "siteb160", "enabled": True, "station_params": {"roi_regions": []}},
                    {"id": "siteb61", "enabled": False}]}


def test_build_line_swaps_model_version_and_keeps_params():
    reg = FakeRegistry()
    line, models = build_line(LINE, camera_id=None, model_name=None, model_version=3, registry=reg)
    assert [c["id"] for c in line["cameras"]] == ["siteb160"]              # 只留一台（啟用中的第一台）
    m = line["model"]
    assert m["version"] == 3 and m["classes"] == ["board", "hand"] and m["conf"] == 0.4   # 部署參數保留
    assert line["extra_stations"] == []                                 # 只測主站：線上掛的 ppe 不帶進來
    assert [(n, v) for n, v, _ in models] == [("siteb_echo", 3)] and ("ppe", None) not in reg.calls
    other, _ = build_line(LINE, camera_id=None, model_name=None, model_version=None, registry=reg,
                          prefer_camera="siteb12")                    # 別台的影片：用這條線第一台的參數
    assert other["cameras"][0]["id"] == "siteb160"
    with pytest.raises(ValueError, match="沒有相機 siteb61"):
        build_line(LINE, camera_id="siteb61", model_name=None, model_version=None, registry=reg)


def test_build_line_station_override():
    """換判定邏輯：相機與錄影照舊，站別參數不沿用舊邏輯的（改用給的或站別預設）。"""
    reg = FakeRegistry()
    line, _ = build_line(LINE, camera_id=None, model_name=None, model_version=None, registry=reg,
                         station="siteb_bravo_v2")
    assert line["station"] == "siteb_bravo_v2" and line["cameras"][0]["station_params"] == {}
    line, _ = build_line(LINE, camera_id=None, model_name=None, model_version=None, registry=reg,
                         station="siteb_bravo_v2", station_params={"thresholds": {"press_frames": 3}})
    assert line["cameras"][0]["station_params"] == {"thresholds": {"press_frames": 3}}
    line, _ = build_line(LINE, camera_id=None, model_name=None, model_version=None, registry=reg)
    assert line["station"] == "echo_check" and line["cameras"][0]["station_params"] == {"roi_regions": []}


def test_summarize_counts_ng_per_station():
    rows = [{"t": 0.0, "frame_id": "1", "station": None, "model_name": "siteb_echo",
             "detections": [{"class_name": "hand"}], "verdict": {"status": "ok"}},
            {"t": 1.2, "frame_id": "2", "station": None, "model_name": "siteb_echo",
             "detections": [{"class_name": "hand"}, {"class_name": "board"}],
             "verdict": {"status": "ng", "reason": "未完成確認", "details": {"n": 1}}},
            {"t": 1.2, "frame_id": "2", "station": "ppe", "model_name": "ppe", "detections": [],
             "verdict": {"status": "ng", "reason": "未戴安全帽"}},
            {"t": 2.0, "frame_id": "3", "station": None, "model_name": "siteb_echo", "detections": [],
             "verdict": {"status": "unknown", "reason": "沒看到目標"}}]
    s = summarize(rows)
    assert s["frames"] == {"main": 3, "ppe": 1} and s["ng_count"] == 2
    assert s["unknown_count"] == 1 and s["unknown"][0]["reason"] == "沒看到目標"
    assert s["ng_by_reason"] == {"main：未完成確認": 1, "ppe：未戴安全帽": 1}
    assert s["detections_by_class"] == {"hand": 2, "board": 1}
    assert s["ng"][0] == {"t": 1.2, "frame_id": "2", "station": None, "model": "siteb_echo", "reason": "未完成確認",
                          "details": {"n": 1}}


def test_model_mode_has_class_stats_and_no_ng():
    """純模型測試：不照實際速度、站別判定不準 → 不出 NG，改看各（站別, 類別）的框數 / 出現幀數 / 平均信心值。"""
    rows = [{"station": None, "detections": [{"class_name": "glove", "confidence": 0.8},
                                             {"class_name": "glove", "confidence": 0.6},
                                             {"class_name": "tap", "confidence": 0.5}],
             "verdict": {"status": "ng", "reason": "x"}},
            {"station": None, "detections": [{"class_name": "glove", "confidence": 0.4}], "verdict": {"status": "ok"}},
            {"station": "ppe", "detections": [{"class_name": "glove", "confidence": 0.9}], "verdict": {"status": "ok"}}]
    assert class_stats(rows) == [
        {"station": "main", "class_name": "glove", "boxes": 3, "frames": 2, "mean_conf": 0.6},
        {"station": "main", "class_name": "tap", "boxes": 1, "frames": 1, "mean_conf": 0.5},
        {"station": "ppe", "class_name": "glove", "boxes": 1, "frames": 1, "mean_conf": 0.9}]
    s = summarize(rows, "model")
    assert s["mode"] == "model" and s["ng_count"] is None and s["unknown_count"] is None and s["ng"] == []
    assert s["frames"] == {"main": 2, "ppe": 1} and len(s["class_stats"]) == 3
    assert summarize(rows)["ng_count"] == 1                             # logic 照舊


def test_input_needs_exactly_one_video_source():
    VideoTestIn(line_id="siteb_echo", source_id=5)
    VideoTestIn(line_id="siteb_echo", camera_id="siteb160", start="2026-10-16T08:00:00", end="2026-10-16T08:05:00")
    with pytest.raises(ValueError):
        VideoTestIn(line_id="siteb_echo")
    with pytest.raises(ValueError):
        VideoTestIn(line_id="siteb_echo", source_id=5, start="2026-10-16T08:00:00")
    with pytest.raises(ValueError):
        VideoTestIn(line_id="siteb_echo", start="2026-10-16T08:00:00", end="2026-10-16T08:05:00")   # 少相機
    with pytest.raises(ValueError, match="太長"):                                                  # 上限 10 分鐘
        VideoTestIn(line_id="siteb_echo", camera_id="siteb160", start="2026-10-16T08:00:00", end="2026-10-16T08:10:01")
    with pytest.raises(ValueError, match="太長"):
        VideoTestIn(line_id="siteb_echo", source_id=5, start_s=10, end_s=700)
    VideoTestIn(line_id="siteb_echo", camera_id="siteb160", start="2026-10-16T08:00:00", end="2026-10-16T08:10:00")


def test_orchestrator_runs_four_steps_and_cleans_up(tmp_path, monkeypatch):
    store = MemStore()
    runner = FakeRunner(store)
    o = _orch(store, runner, tmp_path)
    img = tmp_path / "image"
    img.write_text("registry/edgevision-runner:abc\n")
    o.runner_image_file = img
    o.media_nas_pvc = "nas-recordings"
    monkeypatch.setattr(o, "_wait_triton", lambda url, tid: True)
    tid = store.create(type=TaskType.VIDEO_TEST, created_by="u", channel="api",
                       params={"line_id": "siteb_echo", "source_id": 3})
    wd = work_dir(tmp_path, tid)
    wd.mkdir(parents=True)
    (wd / "run.json").write_text(json.dumps({"model_name": "siteb_echo", "model_version": 3}))
    (wd / "line.json").write_text(json.dumps({"id": "siteb_echo"}))
    o._run_video_test(tid)
    assert runner.started == ["prepare", "vt-triton", "runner", "finalize"]
    r = runner.specs["runner"]
    assert r.image == "registry/edgevision-runner:abc" and r.command == [] and r.gpu
    assert r.env["VIDEO_FILE"] == f"/work/video-tests/{tid}/video.mp4" and r.env["CAMERA_SOURCE"] == "file"
    assert r.env["TRITON_GRPC"] == f"edgevision-vt-{tid}:8001" and r.env["MODEL_NAME"] == "siteb_echo"
    assert "REDIS_URL" not in r.env                                     # 測試結果不能混進線上通知
    assert runner.specs["vt-triton"].ports == [8000, 8001]
    assert runner.specs["prepare"].volumes == {o.work_volume: "/work", "nas-recordings": "/nas"}
    assert f"edgevision-vt-{tid}" in runner.removed and not wd.exists()      # 臨時 Triton 與工作檔都清掉
    assert TaskType.VIDEO_TEST.value in __import__("edgevision.orchestrator.orchestrator", fromlist=["GPU_TYPES"]).GPU_TYPES
    assert "VIDEO_PACE" not in r.env                                    # logic（預設）：照實際速度播


def test_model_mode_runs_runner_unpaced_and_records_time(tmp_path, monkeypatch):
    store = MemStore()
    runner = FakeRunner(store)
    o = _orch(store, runner, tmp_path)
    (tmp_path / "image").write_text("registry/edgevision-runner:abc\n")
    o.runner_image_file = tmp_path / "image"
    monkeypatch.setattr(o, "_wait_triton", lambda url, tid: True)
    tid = store.create(type=TaskType.VIDEO_TEST, created_by="u", channel="api",
                       params={"line_id": "siteb_echo", "source_id": 3, "mode": "model"})
    wd = work_dir(tmp_path, tid)
    wd.mkdir(parents=True)
    (wd / "run.json").write_text(json.dumps({"model_name": "siteb_echo", "model_version": 3}))
    (wd / "line.json").write_text(json.dumps({"id": "siteb_echo"}))
    seen = {}
    real = o._run_step

    def spy(task_id, spec):
        if spec.name == "finalize":                                     # finalize 讀得到 runner 花的秒數
            seen.update(json.loads((wd / "run.json").read_text()))
        return real(task_id, spec)
    monkeypatch.setattr(o, "_run_step", spy)
    o._run_video_test(tid)
    assert runner.specs["runner"].env["VIDEO_PACE"] == "0" and "runner_s" in seen


def test_runner_image_prefers_line_override(tmp_path):
    o = _orch(MemStore(), FakeRunner(MemStore()), tmp_path)
    o.runner_image_file = tmp_path / "image"
    (tmp_path / "image").write_text("registry/edgevision-runner:old\n")
    (tmp_path / "image.siteb_bravo").write_text("registry/edgevision-runner:new\n")
    assert o.runner_image("siteb_bravo") == "registry/edgevision-runner:new"   # 這條線在試新版：影片測試跟線上同一版
    assert o.runner_image("siteb_echo") == o.runner_image() == "registry/edgevision-runner:old"
    # 影片測試沒有自己的版本（2026-11-04 拿掉）：留著舊的 image.video-test 也不理它
    (tmp_path / "image.video-test").write_text("registry/edgevision-runner:vt\n")
    assert o.runner_image("siteb_echo") == "registry/edgevision-runner:old"


def test_orchestrator_fails_without_runner_image(tmp_path):
    store = MemStore()
    runner = FakeRunner(store)
    o = _orch(store, runner, tmp_path)
    o.runner_image_file = tmp_path / "missing"
    tid = store.create(type=TaskType.VIDEO_TEST, created_by="u", channel="api",
                       params={"line_id": "siteb_echo", "source_id": 3})
    o._run_video_test(tid)
    assert runner.started == [] and store.tasks[tid]["status"] == "failed"


def _vt(store, status="queued"):
    from libs.contracts.training import TaskStatus
    return store.create(type=TaskType.VIDEO_TEST, created_by="u", channel="api",
                        params={"line_id": "siteb_echo", "source_id": 3}, status=TaskStatus(status))


def test_video_tests_side_by_side_and_one_with_training(tmp_path):
    """影片測試彼此可以同時跑（video_max_concurrent）；訓練在跑時最多 video_max_with_training 個
    （10/21 實測：訓練≈24 GB＋影片測試≈12.5 GB 放得下，＋2 個會爆 48 GB 切片）；自動標注照舊互斥。"""
    from libs.contracts.training import TaskStatus
    store = MemStore()
    o = _orch(store, FakeRunner(store), tmp_path)
    o.video_max_concurrent, o.video_max_with_training = 2, 1
    VT, TR, AL = TaskType.VIDEO_TEST.value, TaskType.TRAINING.value, TaskType.AUTOLABEL.value
    v1 = _vt(store, "working")
    assert o._has_slot(VT) and o._has_slot(TR)                     # 1 個影片測試：訓練也能一起跑
    assert not o._has_slot(AL)                                     # 自動標注沒量過：照舊等
    v2 = _vt(store, "working")
    assert not o._has_slot(VT)                                     # 2/2 滿了
    assert not o._has_slot(TR)                                     # 2 個影片測試＋訓練會爆：訓練等
    store.tasks[v2]["status"] = "completed"
    tr = store.create(type=TaskType.TRAINING, created_by="u", channel="api", params={}, status=TaskStatus.WORKING)
    assert not o._has_slot(VT)                                     # 訓練＋1 個影片測試：滿了
    store.tasks[v1]["status"] = "completed"
    assert o._has_slot(VT)                                         # 訓練＋0 個：可以再跑 1 個
    assert not o._has_slot(TR)                                     # 訓練一次一個
    store.tasks[tr]["status"] = "completed"
    store.create(type=TaskType.AUTOLABEL, created_by="u", channel="api", params={}, status=TaskStatus.WORKING)
    assert not o._has_slot(VT)                                     # 自動標注在跑：影片測試等
    o.video_max_with_training = 0
    store.tasks[[k for k, v in store.tasks.items() if v["type"] == AL][0]]["status"] = "completed"
    store.create(type=TaskType.TRAINING, created_by="u", channel="api", params={}, status=TaskStatus.WORKING)
    assert not o._has_slot(VT)                                     # 設 0 = 訓練時不跑影片測試（舊行為）


def test_queue_message_names_what_is_on_gpu(tmp_path):
    """排隊訊息寫出 GPU 上正在跑誰、前面是誰，不只「前面還有 N 個」。"""
    from libs.contracts.training import TaskStatus
    store = MemStore()
    o = _orch(store, FakeRunner(store), tmp_path)
    o.video_max_with_training = 0
    tr = store.create(type=TaskType.TRAINING, created_by="u", channel="api",
                      params={"request": {"model_name": "ppe"}}, status=TaskStatus.WORKING)
    v1, v2 = _vt(store), _vt(store)
    o._drain_queue()
    m1, m2 = store.tasks[v1]["status_message"], store.tasks[v2]["status_message"]
    assert f"GPU 上：{tr}（訓練，ppe）" in m1 and "前面還有 0 個排隊" in m1
    assert "前面還有 1 個排隊" in m2 and f"下一個是 {v1}（影片測試，siteb_echo）" in m2


def test_drain_keeps_queue_order(tmp_path, monkeypatch):
    """照排隊順序放行：最前面的是訓練、影片測試 2 個在跑（訓練要等到剩 1 個）→ 後面的影片測試也不能插隊
    （訓練才不會一直被擠掉）。"""
    from libs.contracts.training import TaskStatus
    store = MemStore()
    o = _orch(store, FakeRunner(store), tmp_path)
    o.video_max_concurrent, o.video_max_with_training = 3, 1
    spawned = []
    monkeypatch.setattr(o, "_spawn", lambda tid, target: spawned.append(tid))
    _vt(store, "working")
    _vt(store, "working")
    tr = store.create(type=TaskType.TRAINING, created_by="u", channel="api", params={}, status=TaskStatus.QUEUED)
    v2 = _vt(store)
    o._drain_queue()
    assert spawned == [] and store.tasks[v2]["status"] == "queued"
    store.tasks[tr]["status"] = "cancelled"                        # 前面沒有訓練了 → 影片測試並排跑
    o._drain_queue()
    assert spawned == [v2]


def test_video_test_passes_training_that_only_waits_for_training_slot(tmp_path, monkeypatch):
    """前面的訓練只是在等訓練名額（另一個訓練在跑）→ 影片測試可以先跑，上限照 video_max_with_training；
    前面是自動標注、或訓練在等影片測試變少 → 不能插。"""
    from libs.contracts.training import TaskStatus
    store = MemStore()
    o = _orch(store, FakeRunner(store), tmp_path)
    o.video_max_concurrent, o.video_max_with_training = 2, 1
    spawned = []
    monkeypatch.setattr(o, "_spawn", lambda tid, target: spawned.append(tid))
    tr1 = store.create(type=TaskType.TRAINING, created_by="u", channel="api", params={}, status=TaskStatus.WORKING)
    store.create(type=TaskType.TRAINING, created_by="u", channel="api", params={}, status=TaskStatus.QUEUED)
    v1, v2 = _vt(store), _vt(store)
    o._drain_queue()
    assert spawned == [v1] and store.tasks[v2]["status"] == "queued"           # 只放 1 個（訓練在跑）
    assert o._has_slot(TaskType.TRAINING.value) is False                      # 第二個訓練照樣等訓練名額
    store.tasks[tr1]["status"] = "completed"
    assert o._has_slot(TaskType.TRAINING.value)                               # 前一個跑完：1 個影片測試不擋訓練

    store2 = MemStore()
    o2 = _orch(store2, FakeRunner(store2), tmp_path)
    spawned2 = []
    monkeypatch.setattr(o2, "_spawn", lambda tid, target: spawned2.append(tid))
    store2.create(type=TaskType.TRAINING, created_by="u", channel="api", params={}, status=TaskStatus.WORKING)
    store2.create(type=TaskType.AUTOLABEL, created_by="u", channel="api", params={}, status=TaskStatus.QUEUED)
    _vt(store2)
    o2._drain_queue()
    assert spawned2 == []                                                     # 前面是自動標注：不插


def test_reconcile_resumes_video_test_and_stale_cleans_up(tmp_path, monkeypatch):
    """服務重啟：影片測試要接手（以前漏掉 → 卡 working、臨時 Triton 一直佔 GPU）；逾時判失敗時清掉臨時 Triton。"""
    store = MemStore()
    runner = FakeRunner(store)
    o = _orch(store, runner, tmp_path)
    spawned = []
    monkeypatch.setattr(o, "_spawn", lambda tid, target: spawned.append((tid, target.__name__)))
    v = _vt(store, "working")
    o.reconcile()
    assert spawned == [(v, "_run_video_test")]

    from edgevision.orchestrator.runner import RunHandle
    left = {f"edgevision-vt-{v}", f"edgevision-job-{v}-runner"}
    monkeypatch.setattr(runner, "find", lambda name: RunHandle(ref=name, name=name) if name in left else None)
    (work_dir(tmp_path, v)).mkdir(parents=True)
    o._cleanup_video_test(v)
    assert set(runner.removed) == left and not work_dir(tmp_path, v).exists()


def test_build_line_model_mode_needs_no_logic_or_roi():
    """mode=model：不用線上判定邏輯、不要 ROI（線路還沒畫 ROI 的新線也能測模型）。"""
    from edgevision.media.video_test import MODEL_ONLY_PARAMS, MODEL_ONLY_STATION
    reg = FakeRegistry()
    line, _ = build_line(LINE, camera_id=None, model_name=None, model_version=None, registry=reg,
                         station="golf_check", station_params={"roi_regions": [{"type": "detect"}]},
                         mode="model")
    assert line["station"] == MODEL_ONLY_STATION
    assert line["cameras"][0]["station_params"] == MODEL_ONLY_PARAMS
    assert line["extra_stations"] == []


# ── 壞段（沒有 moov）要跳過，不能拖垮整個任務（2026-11-03 cam137 14:39）──

def _mp4(path, boxes):
    import struct
    path.write_bytes(b"".join(struct.pack(">I4s", 8 + len(body), kind) + body for kind, body in boxes))
    return path


def test_has_moov(tmp_path):
    from edgevision.media.video_test import has_moov
    ok = _mp4(tmp_path / "ok.mp4", [(b"ftyp", b"isom"), (b"free", b""), (b"mdat", b"x" * 100), (b"moov", b"m")])
    bad = _mp4(tmp_path / "bad.mp4", [(b"ftyp", b"isom"), (b"free", b""), (b"mdat", b"x" * 100)])
    trunc = tmp_path / "trunc.mp4"
    trunc.write_bytes(ok.read_bytes()[:-1])       # moov 寫到一半
    assert has_moov(ok)
    assert not has_moov(bad)
    assert not has_moov(tmp_path / "missing.mp4")
    assert not has_moov(trunc)
    (tmp_path / "empty.mp4").write_bytes(b"")
    assert not has_moov(tmp_path / "empty.mp4")


def test_resolve_parts_skips_segments_without_moov(tmp_path):
    from edgevision.media.video_test import resolve_parts
    d = tmp_path / "cam"
    d.mkdir()
    good = [(b"ftyp", b"isom"), (b"mdat", b"x"), (b"moov", b"m")]
    _mp4(d / "20261103_142947.mp4", good)
    _mp4(d / "20261103_143941.mp4", [(b"ftyp", b"isom"), (b"mdat", b"x")])
    _mp4(d / "20261103_144922.mp4", good)
    parts, _cam, desc = resolve_parts(None, {"camera_id": "cam", "start": "2026-11-03T14:35:00",
                                            "end": "2026-11-03T14:55:00"}, tmp_path)
    assert [p.name for p, _, _ in parts] == ["20261103_142947.mp4", "20261103_144922.mp4"]
    assert desc["skipped_segments"] == ["20261103_143941.mp4"]


def test_resolve_parts_all_broken_is_a_clear_error(tmp_path):
    import pytest

    from edgevision.media.video_test import resolve_parts
    d = tmp_path / "cam"
    d.mkdir()
    _mp4(d / "20261103_143941.mp4", [(b"ftyp", b"isom"), (b"mdat", b"x")])
    with pytest.raises(ValueError, match="打不開"):
        resolve_parts(None, {"camera_id": "cam", "start": "2026-11-03T14:40:00",
                             "end": "2026-11-03T14:45:00"}, tmp_path)
