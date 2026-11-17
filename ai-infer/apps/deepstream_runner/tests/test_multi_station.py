"""一線多站：主站 + 附加站（各自模型、各自 nvinferserver）。不需 pyds、不需 GPU。"""
import json

import numpy as np
import pytest
from api_shapes import api_line
from ds_runner.config import LineConfig, load_line_config
from ds_runner.worker import TensorJob, Worker
from edgevision_contracts import VerdictStatus
from edgevision_station_sdk import MissingCameraParams

PPE_DEMO = {"name": "ppe", "classes": ["helmet", "vest", "person", "no_helmet", "mouth", "no_vest"]}


def line_with_ppe(tmp_path, **extra):
    d = api_line()
    d["extra_stations"] = [{"station": "ppe_check", "interval": 4, "model": PPE_DEMO, **extra}]
    for cam in d["cameras"]:
        cam["extra_params"] = {"ppe_check": {}}
    p = tmp_path / "line.json"
    p.write_text(json.dumps(d))
    return load_line_config({"CAMERA_SOURCE": "file", "CAMERA_FILE": str(p)})


def test_bindings_primary_first_then_extras_with_gie_ids(tmp_path):
    cfg = line_with_ppe(tmp_path)
    b = cfg.bindings()
    assert [x.key for x in b] == ["zone_intrusion", "ppe_check"]
    assert [x.gie_id for x in b] == [1, 2] and b[0].primary and not b[1].primary
    assert b[1].task == "detect" and b[1].interval == 4 and b[1].model.name == "ppe"
    assert b[0].interval is None                                   # 主站的間隔由 pipeline.resolve_interval 決定


def test_extra_station_required_params_are_validated_per_camera(tmp_path):
    """附加站宣告的必填現場值也要齊（template 站要 target_class / max_count）。"""
    d = api_line()
    d["extra_stations"] = [{"station": "template", "model": {"name": "tpl", "classes": ["board"]}}]
    p = tmp_path / "line.json"
    p.write_text(json.dumps(d))
    with pytest.raises(MissingCameraParams):
        load_line_config({"CAMERA_SOURCE": "file", "CAMERA_FILE": str(p)})


def test_duplicate_station_names_rejected():
    d = api_line()
    d["extra_stations"] = [{"station": "zone_intrusion"}]
    with pytest.raises(ValueError, match="站別重複"):
        LineConfig.model_validate(d)


def test_worker_routes_job_by_station_and_tags_result(tmp_path):
    cfg = line_with_ppe(tmp_path)
    w = Worker(cfg, publish=lambda r, img: None, model_version="1")
    assert set(w.runtimes) == {"zone_intrusion", "ppe_check"}
    assert set(w.runtimes["ppe_check"].stations) == {"cam156", "cam193"}

    # PPE 模型（6 類，類別名來自線路設定）：detect → output0 (4+6, 8400)；第 3 類 = no_helmet
    out0 = np.zeros((4 + 6, 8400), np.float32)
    out0[:4, 0] = (320, 320, 100, 100)
    out0[4 + 3, 0] = 0.9
    r = w.process(TensorJob("cam156", out0, None, 1280, 720, station="ppe_check"))
    assert r.station == "ppe_check" and r.model_name == "ppe"
    assert [d.class_name for d in r.detections] == ["no_helmet"]
    assert r.verdict.status == VerdictStatus.OK                     # 視窗 20 次只出現 1 次，不確認

    # 沒帶 station = 主站，行為與以前相同
    seg = np.zeros((4 + 2 + 32, 8400), np.float32)
    seg[:4, 0] = (320, 320, 200, 120)
    seg[4, 0] = 0.95
    r2 = w.process(TensorJob("cam156", seg, np.zeros((32, 160, 160), np.float32), 1280, 720))
    assert r2.station == "zone_intrusion" and r2.model_name == "seg_alpha" and r2.frame_id == "1"


def test_render_unique_id_per_binding():
    from ds_runner.pipeline import render_nvinfer_config
    tmpl = "unique_id: ${UNIQUE_ID}\n${MODEL_NAME} ${TRITON_GRPC} ${INFER_INTERVAL}\n${OUTPUT_LAYERS}"
    out = render_nvinfer_config(tmpl, model_name="m", triton_grpc="t", interval=4, task="detect", unique_id=2)
    assert "unique_id: 2" in out and "interval" not in out


def test_publisher_uses_station_from_result():
    from ds_runner.publish import RedisPublisher
    from edgevision_contracts import BBox, Detection, FrameResult, ImageMeta, Verdict

    class R:
        def __init__(self): self.calls = []
        def ping(self): return True
        def xadd(self, stream, fields, maxlen=None, approximate=None): self.calls.append((stream, fields))

    r = R()
    pub = RedisPublisher("redis://x", client=r, line_id="ipcam_alpha", station="alpha_check")
    det = Detection(class_id=0, class_name="no_helmet", confidence=0.9, bbox=BBox(x1=0, y1=0, x2=10, y2=10))
    res = FrameResult(camera_id="c", frame_id="1", model_name="ppe_alpha", model_version="1", station="ppe_check",
                      image_meta=ImageMeta(width=1280, height=720), detections=[det], verdict=Verdict.ng("未戴安全帽"))
    pub(res)
    assert r.calls[0][0] == "edgevision:events" and r.calls[0][1]["station"] == "ppe_check"


def test_hidden_classes_per_model(tmp_path):
    """每顆模型各自的 hidden_classes（不畫但照樣判定）；沒給 = 全畫。"""
    cfg = line_with_ppe(tmp_path, model={**PPE_DEMO, "hidden_classes": ["mouth"]})
    b = cfg.bindings()
    assert b[0].model.hidden_classes == [] and b[1].model.hidden_classes == ["mouth"]
