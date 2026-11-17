"""餵假張量走完 worker 的 process()：後處理 → 站別 → FrameResult。不需 pyds、不需 GPU。"""
import numpy as np
from ds_runner.config import load_line_config
from ds_runner.probe import split_outputs
from ds_runner.worker import TensorJob, Worker
from edgevision_contracts import VerdictStatus

N_ANCHORS, N_MASK, PROTO_HW = 8400, 32, 160


def board_tensor():
    out0 = np.zeros((4 + 2 + N_MASK, N_ANCHORS), np.float32)
    out0[:4, 0] = (320, 320, 200, 120)
    out0[4, 0] = 0.95
    return out0, np.zeros((N_MASK, PROTO_HW, PROTO_HW), np.float32)


def test_worker_builds_one_station_per_camera_and_processes(line_file):
    cfg = load_line_config({"CAMERA_SOURCE": "file", "CAMERA_FILE": str(line_file)})
    published = []
    w = Worker(cfg, publish=lambda r, img: published.append(r), model_version="1")
    assert set(w.stations) == {"cam156", "cam193"}
    assert w.stations["cam156"] is not w.stations["cam193"]

    out0, proto = board_tensor()
    r = w.process(TensorJob("cam156", out0, proto, 1280, 720))
    assert r.camera_id == "cam156" and r.frame_id == "1" and r.model_name == "seg_alpha"
    assert [d.class_name for d in r.detections] == ["board"]
    assert r.verdict is not None and r.verdict.status == VerdictStatus.OK   # 平放的板，沒手


def test_worker_hot_updates_params_without_rebuilding(line_file):
    cfg = load_line_config({"CAMERA_SOURCE": "file", "CAMERA_FILE": str(line_file)})
    w = Worker(cfg, publish=lambda r, img: None)
    before = w.stations["cam156"]
    w.update_camera_params("cam156", {"zone": [[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]]})
    assert w.stations["cam156"] is before


def test_split_outputs_picks_proto_by_shape():
    out0, proto = board_tensor()
    got0, gotp = split_outputs([(proto, list(proto.shape)), (out0, list(out0.shape))])
    assert got0 is out0 and gotp is proto


# ── fused 格式（NMS + mask 合成在 ONNX 裡）──────────────────

def _meta(outputs: dict[str, list[int]]) -> dict:
    return {"inputs": [{"name": "images", "shape": [-1, 3, 640, 640]}],
            "outputs": [{"name": n, "shape": s} for n, s in outputs.items()]}


def test_check_model_accepts_fused_seg_engine():
    """fused：output0 (B, 16, 38) + masks (B, 16, 160, 160)，沒有 output1 —— 必須通過。"""
    from ds_runner.pipeline import check_model_matches
    meta = _meta({"output0": [-1, 16, 38], "masks": [-1, 16, 160, 160]})
    assert check_model_matches(meta, model_name="seg_alpha", imgsz=640, n_classes=2, task="segment") == []


def test_check_model_still_rejects_raw_seg_without_output1():
    from ds_runner.pipeline import check_model_matches
    meta = _meta({"output0": [-1, 38, 8400]})
    probs = check_model_matches(meta, model_name="seg_alpha", imgsz=640, n_classes=2, task="segment")
    assert any("output1" in p for p in probs)


def test_check_model_fused_catches_class_mismatch():
    """fused 的 features 在【最後】一維；類別數對不上仍要擋。"""
    from ds_runner.pipeline import check_model_matches
    meta = _meta({"output0": [-1, 16, 39], "masks": [-1, 16, 160, 160]})   # 4+3+32
    probs = check_model_matches(meta, model_name="seg_alpha", imgsz=640, n_classes=2, task="segment")
    assert any("類別數" in p for p in probs)


def test_check_model_fused_catches_mask_size_mismatch():
    from ds_runner.pipeline import check_model_matches
    meta = _meta({"output0": [-1, 16, 38], "masks": [-1, 16, 128, 128]})
    probs = check_model_matches(meta, model_name="seg_alpha", imgsz=640, n_classes=2, task="segment")
    assert any("masks 尺寸" in p for p in probs)


def test_seg_output_layers_picks_by_triton_outputs():
    from ds_runner.pipeline import seg_output_layers
    assert seg_output_layers({"output0": {}, "masks": {}}) == ["output0", "masks"]
    assert seg_output_layers({"output0": {}, "output1": {}}) == ["output0", "output1"]


def test_render_uses_given_output_layers():
    """呼叫端問過 Triton 後給的層名要進 nvinferserver 設定，不能被 task 預設蓋掉。"""
    from ds_runner.pipeline import render_nvinfer_config
    tmpl = 'unique_id: ${UNIQUE_ID}\n${OUTPUT_LAYERS}\nmodel_name: "${MODEL_NAME}"\nurl: "${TRITON_GRPC}"\ninterval: ${INFER_INTERVAL}\n'
    out = render_nvinfer_config(tmpl, model_name="seg_alpha", triton_grpc="t:8001", interval=0,
                                task="segment", output_layers=["output0", "masks"])
    assert 'name: "masks"' in out and 'name: "output1"' not in out
    # 沒給就是 raw
    out = render_nvinfer_config(tmpl, model_name="seg_alpha", triton_grpc="t:8001", interval=0, task="segment")
    assert 'name: "output1"' in out


def test_split_outputs_handles_fused_shapes():
    """fused：masks 是 3 維含 160，跟 proto 同一個判準 —— 分得出來即可，語意由後處理決定。"""
    out0 = np.zeros((16, 38), np.float32)
    masks = np.zeros((16, 160, 160), np.float32)
    got0, gotm = split_outputs([(out0, list(out0.shape)), (masks, list(masks.shape))])
    assert got0 is out0 and gotm is masks


# ── nvinferserver max_batch 跟著 engine 走 ─────────────────

def test_render_max_batch_from_caller():
    """⚠️ 樣板曾寫死 32，IPCAM engine 改 max 4 後 nvinferserver 啟動就死。"""
    from ds_runner.pipeline import render_nvinfer_config
    tmpl = ('max_batch_size: ${MAX_BATCH}\nunique_id: ${UNIQUE_ID}\n${OUTPUT_LAYERS}\n'
            'model_name: "${MODEL_NAME}"\nurl: "${TRITON_GRPC}"\ninterval: ${INFER_INTERVAL}\n')
    out = render_nvinfer_config(tmpl, model_name="foxtrot", triton_grpc="t:8001", interval=0, task="detect", max_batch=4)
    assert "max_batch_size: 4" in out


def test_real_template_has_no_hardcoded_batch():
    from pathlib import Path

    from ds_runner.pipeline import DEFAULT_TEMPLATE
    t = Path(DEFAULT_TEMPLATE).read_text()
    assert "${MAX_BATCH}" in t and "max_batch_size: 32" not in t


def test_triton_max_batch_falls_back_without_triton():
    from ds_runner.pipeline import triton_max_batch
    assert triton_max_batch(None, "x") == 32
    assert triton_max_batch(None, "x", default=4) == 4


# ── 串流恢復保護（2026-10-28 正式環境：相機每 30 分重連，重連後的殘影被判成違規動作）──

def test_worker_holds_verdicts_after_stream_gap(line_file):
    cfg = load_line_config({"CAMERA_SOURCE": "file", "CAMERA_FILE": str(line_file)})
    t = [100.0]
    w = Worker(cfg, publish=lambda r, img: None, resume_gap_s=2, resume_hold_s=5, clock=lambda: t[0])
    resets = []
    w.stations["cam156"].reset = lambda: resets.append(1)
    out0, proto = board_tensor()
    def job(cam="cam156"):
        return TensorJob(cam, out0, proto, 1280, 720)
    assert w.process(job()).verdict.status == VerdictStatus.OK
    t[0] += 0.5
    assert w.process(job()).verdict.status == VerdictStatus.OK           # 正常間隔
    t[0] += 12                                                            # 斷了 12 秒
    r = w.process(job())
    assert r.verdict.status == VerdictStatus.UNKNOWN and resets == [1]
    assert r.detections                                                   # 推論照跑，只是不判
    for _ in range(9):                                                    # 保護期內幀照常進來
        t[0] += 0.5
        assert w.process(job()).verdict.status == VerdictStatus.UNKNOWN
    t[0] += 0.6
    assert w.process(job()).verdict.status == VerdictStatus.OK            # 保護期過了
    t[0] += 1
    assert w.process(job("cam193")).verdict.status == VerdictStatus.OK     # 別台不受影響


def test_worker_resume_hold_skips_video_files_and_can_be_disabled(line_file):
    cfg = load_line_config({"CAMERA_SOURCE": "file", "CAMERA_FILE": str(line_file)})
    t = [0.0]
    out0, proto = board_tensor()
    w = Worker(cfg, publish=lambda r, img: None, resume_gap_s=2, clock=lambda: t[0])
    w.process(TensorJob("cam156", out0, proto, 1280, 720, pts=1))
    t[0] += 60
    assert w.process(TensorJob("cam156", out0, proto, 1280, 720, pts=2)).verdict.status == VerdictStatus.OK
    w = Worker(cfg, publish=lambda r, img: None, resume_gap_s=0, clock=lambda: t[0])
    w.process(TensorJob("cam156", out0, proto, 1280, 720))
    t[0] += 60
    assert w.process(TensorJob("cam156", out0, proto, 1280, 720)).verdict.status == VerdictStatus.OK


# ── NG 冷卻（2026-10-30：一次 NG ＝ 一筆告警 ＝ 一段影片，去重只在這一道）──

class _AlwaysNG:
    def evaluate(self, dets, w, h):
        from edgevision_contracts import Verdict
        return Verdict.ng("測試異常")

    def reset(self):
        pass


def test_worker_ng_cooldown_per_camera(line_file):
    cfg = load_line_config({"CAMERA_SOURCE": "file", "CAMERA_FILE": str(line_file)})
    t = [0.0]
    w = Worker(cfg, publish=lambda r, img: None, clock=lambda: t[0])
    assert w.ng_cooldown_s == 30
    w.stations["cam156"] = _AlwaysNG()
    w.stations["cam193"] = _AlwaysNG()
    out0, proto = board_tensor()
    def job(cam):
        return TensorJob(cam, out0, proto, 1280, 720)
    got = []
    for _ in range(31):                                          # 0～30 秒、每秒一幀都判 NG
        got.append(w.process(job("cam156")).verdict)
        t[0] += 1
    ng = [v for v in got if v.status == VerdictStatus.NG]
    assert len(ng) == 2                                          # 第 0 秒與第 30 秒
    assert got[1].status == VerdictStatus.OK and got[1].details["suppressed"] == "測試異常"
    assert w.process(job("cam193")).verdict.status == VerdictStatus.NG    # 別台各算各的


def test_worker_ng_cooldown_from_line_params_and_video_pts(line_file):
    cfg = load_line_config({"CAMERA_SOURCE": "file", "CAMERA_FILE": str(line_file)})
    cfg.line_params["ng_cooldown_s"] = 0
    w = Worker(cfg, publish=lambda r, img: None)
    w.stations["cam156"] = _AlwaysNG()
    out0, proto = board_tensor()
    assert all(w.process(TensorJob("cam156", out0, proto, 1280, 720)).verdict.status == VerdictStatus.NG
               for _ in range(3))                                # 0 = 不擋
    cfg.line_params["ng_cooldown_s"] = 10
    w = Worker(cfg, publish=lambda r, img: None, clock=lambda: 0.0)
    w.stations["cam156"] = _AlwaysNG()
    pts = [0, 5, 10, 11]                                         # 影片時間（秒）；牆鐘不動
    st = [w.process(TensorJob("cam156", out0, proto, 1280, 720, pts=int(p * 1e9))).verdict.status for p in pts]
    assert st == [VerdictStatus.NG, VerdictStatus.OK, VerdictStatus.NG, VerdictStatus.OK]
