"""主站 ROI 推論（假物件模式）：nvinferserver 設定改寫、跳幀排程、座標還原。不需 GPU。"""
import json

import pytest
from api_shapes import api_line
from ds_runner.config import LineConfig, load_line_config
from ds_runner.pipeline import DEFAULT_TEMPLATE, render_nvinfer_config
from ds_runner.roi import ROI_GIE_ID, InjectSchedule, roi_pixels, use_roi_objects
from ds_runner.worker import Worker, WorkerPool


def test_roi_pixels_rounds_and_clamps():
    assert roi_pixels([0.475, 0.45, 0.975, 0.95], 1920, 1080) == (912, 486, 1872, 1026)
    assert roi_pixels([0.0, 0.25, 0.5, 1.0], 1280, 720) == (0, 180, 640, 720)
    x1, y1, x2, y2 = roi_pixels([0.999, 0.999, 1.0, 1.0], 100, 100)
    assert x2 - x1 >= 2 and y2 - y1 >= 2 and x2 <= 100 and y2 <= 100


def test_use_roi_objects_switches_primary_to_clip_mode_on_real_template():
    text = render_nvinfer_config(DEFAULT_TEMPLATE.read_text(encoding="utf-8"),
                                 model_name="m", triton_grpc="t", interval=3, task="detect")
    out = use_roi_objects(text)
    ic = out.split("input_control {")[1].split("}")[0]
    assert "PROCESS_MODE_CLIP_OBJECTS" in ic and f"operate_on_gie_id: {ROI_GIE_ID}" in ic
    assert "operate_on_class_ids: [0]" in ic and "interval: 0" in ic      # 跳幀交給 injector
    assert "PROCESS_MODE_FULL_FRAME" not in out
    # 前處理（等比、對稱補邊、1/255）與輸出段都不動 —— 裁出來的 ROI 跟全幅走同一套 preprocess
    assert "maintain_aspect_ratio: 1" in out and "symmetric_padding: 1" in out
    assert "output_tensor_meta: true" in out and out.count("input_control {") == 1


def test_use_roi_objects_rejects_config_without_input_control():
    with pytest.raises(ValueError, match="input_control"):
        use_roi_objects("infer_config {}\n")


def test_roi_gie_id_does_not_collide_with_station_gie_ids():
    assert ROI_GIE_ID > 16          # nvinferserver unique_id 是 1..站別數


@pytest.mark.parametrize(("interval", "expect"), [(0, [1, 1, 1, 1]), (1, [1, 0, 1, 0]), (2, [1, 0, 0, 1])])
def test_inject_schedule_matches_nvinferserver_interval(interval, expect):
    s = InjectSchedule(interval)
    assert [int(s.tick()) for _ in expect] == expect


def test_infer_roi_validation():
    d = api_line()
    d["cameras"][0]["infer_roi"] = [0.5, 0.2, 0.4, 0.9]        # x1 >= x2
    with pytest.raises(ValueError, match="infer_roi"):
        LineConfig.model_validate(d)


def test_worker_uses_roi_letterbox_for_primary_only(tmp_path):
    d = api_line()
    d["cameras"][0]["infer_roi"] = [0.0, 0.25, 0.5, 1.0]        # cam156 裁左半下 3/4；cam193 整幅
    d["extra_stations"] = [{"station": "ppe_check", "model": {"name": "ppe", "classes": ["helmet", "vest", "person", "no_helmet", "mouth", "no_vest"]}}]
    for cam in d["cameras"]:
        cam["extra_params"] = {"ppe_check": {}}
    p = tmp_path / "line.json"
    p.write_text(json.dumps(d))
    cfg = load_line_config({"CAMERA_SOURCE": "file", "CAMERA_FILE": str(p)})
    w = Worker(cfg, publish=lambda r, img: None)
    lb69 = w.letterbox_for("cam156", w.primary, 1280, 720)
    lb70 = w.letterbox_for("cam193", w.primary, 1280, 720)
    assert (lb69.offset_x, lb69.offset_y) == (0, 180) and lb69.to_original(0, lb69.pad_y) == (0.0, 180.0)
    assert (lb70.offset_x, lb70.offset_y) == (0, 0)
    ppe = w.letterbox_for("cam156", w.runtimes["ppe_check"], 1280, 720)
    assert (ppe.offset_x, ppe.offset_y) == (0, 0)             # 附加站永遠整幅


def test_pool_disable_roi_falls_back_to_full_frame(tmp_path):
    d = api_line()
    d["cameras"][0]["infer_roi"] = [0.0, 0.25, 0.5, 1.0]
    p = tmp_path / "line.json"
    p.write_text(json.dumps(d))
    cfg = load_line_config({"CAMERA_SOURCE": "file", "CAMERA_FILE": str(p)})
    pool = WorkerPool(cfg, publish=lambda r, img: None, n_workers=2)
    assert any(w.roi_px for w in pool.workers)
    pool.disable_roi()                                        # INFER_ROI_ENABLED=0
    assert not any(w.roi_px for w in pool.workers)
    w = next(w for w in pool.workers if "cam156" in w.primary.stations)
    lb = w.letterbox_for("cam156", w.primary, 1280, 720)
    assert (lb.offset_x, lb.offset_y) == (0, 0)
