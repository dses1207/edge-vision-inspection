"""NAS 錄影列表 / 檔名防護、打包格式（COCO / VOC）、匯入任務哪些要掛 NAS。"""
import os
import time
from datetime import datetime, timedelta

import cv2
import numpy as np
import pytest

from edgevision.media import recordings as R
from edgevision.media.download import coco_json, voc_xml
from edgevision.media.frames import LOCAL_TZ


def _mp4(path, seconds: int, fps: int = 5):
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (160, 90))
    for k in range(seconds * fps):
        vw.write(np.full((90, 160, 3), k % 255, np.uint8))
    vw.release()


@pytest.fixture()
def nas(tmp_path, monkeypatch):
    monkeypatch.setenv("MEDIA_NAS_ROOT", str(tmp_path))
    d = tmp_path / "siteb12"
    d.mkdir()
    day = datetime(2026, 10, 18, 0, 0, tzinfo=LOCAL_TZ)
    starts = [day - timedelta(seconds=20), day + timedelta(seconds=10), day + timedelta(seconds=40),
              day + timedelta(seconds=200)]                      # 40→200 中間缺一段
    for t in starts:
        _mp4(d / f"{t:%Y%m%d_%H%M%S}.mp4", 30)
    (d / f"{day + timedelta(seconds=300):%Y%m%d_%H%M%S}.mp4").write_bytes(b"not a video")   # 壞檔
    old = time.time() - 3600
    for p in d.iterdir():
        os.utime(p, (old, old))                                  # 都不是「還在寫」
    (d / "notes.txt").write_text("x")                            # 不是錄影檔，不列
    return tmp_path, day


def test_list_day_segments_gaps_broken(nas):
    _, day = nas
    out = R.list_day(None, "siteb12", day.date())
    files = [s["file"] for s in out["segments"]]
    assert files[0] == f"{day - timedelta(seconds=20):%Y%m%d_%H%M%S}.mp4"        # 前一天最後一段跨過午夜
    assert [s["status"] for s in out["segments"]] == ["ok", "ok", "ok", "ok", "broken"]
    assert out["broken"] == 1 and abs(out["segments"][1]["duration_s"] - 30) < 0.5
    g = out["gaps"][0]
    assert g["start"] == day + timedelta(seconds=70) and g["end"] == day + timedelta(seconds=200)
    days = R.list_days("siteb12")
    assert [d["date"] for d in days] == ["2026-10-18", "2026-10-17"] and days[0]["segments"] == 4
    cams = R.cameras()
    assert cams[0]["camera_id"] == "siteb12" and cams[0]["files"] == 5 and cams[0]["days"] == 2


def test_file_path_guards(nas):
    root, day = nas
    name = f"{day + timedelta(seconds=10):%Y%m%d_%H%M%S}.mp4"
    assert R.file_path("siteb12", name) == root / "siteb12" / name
    for cam, f in (("siteb12", "../../etc/passwd"), ("siteb12", "notes.txt"), ("../x", name)):
        with pytest.raises(ValueError):
            R.file_path(cam, f)
    with pytest.raises(FileNotFoundError):
        R.file_path("siteb12", "20990101_000000.mp4")


def test_no_nas(tmp_path, monkeypatch):
    monkeypatch.setenv("MEDIA_NAS_ROOT", str(tmp_path / "missing"))
    with pytest.raises(R.NasUnavailable):
        R.cameras()


def _item(**kw):
    return {"id": 7, "status": "labeled", "width": 200, "height": 100,
            "shapes": [{"type": "rect", "class": "mat", "points": [[0.1, 0.2], [0.5, 0.6]]},
                       {"type": "polygon", "class": "glove", "points": [[0, 0], [0.5, 0], [0.5, 0.5]]},
                       {"type": "rect", "class": "gone", "points": [[0, 0], [1, 1]]}]} | kw


def test_coco_json():
    j = coco_json([_item()], ["glove", "mat"], {7: "abc.jpg"})
    assert j["categories"] == [{"id": 1, "name": "glove", "supercategory": ""}, {"id": 2, "name": "mat", "supercategory": ""}]
    assert j["images"] == [{"id": 7, "file_name": "abc.jpg", "width": 200, "height": 100}]
    pad, glove = j["annotations"]
    assert pad["category_id"] == 2 and pad["bbox"] == [20.0, 20.0, 80.0, 40.0] and pad["area"] == 3200.0
    assert glove["category_id"] == 1 and glove["segmentation"] == [[0.0, 0.0, 100.0, 0.0, 100.0, 50.0]]
    assert glove["area"] == 2500.0 and glove["bbox"] == [0.0, 0.0, 100.0, 50.0]
    assert coco_json([_item(status="background")], ["mat"], {7: "a.jpg"})["annotations"] == []


def test_voc_xml():
    x = voc_xml(_item(), ["glove", "mat"], "a&b.jpg")
    assert "<filename>a&amp;b.jpg</filename>" in x and "<width>200</width>" in x
    assert x.count("<object>") == 2 and "<xmin>20</xmin><ymin>20</ymin><xmax>100</xmax><ymax>60</ymax>" in x
    assert "gone" not in x


def test_nas_import_kinds_mount_nas(tmp_path):
    from edgevision.orchestrator.orchestrator import NAS_IMPORT_KINDS
    from libs.contracts.training import TaskType
    from tests.test_orchestrator import FakeRunner, MemStore, _orch
    assert set(NAS_IMPORT_KINDS) == {"nas", "alert", "clip"}
    store = MemStore()
    runner = FakeRunner(store)
    o = _orch(store, runner, tmp_path)
    o.media_nas_pvc = "nas-recordings"
    for kind in ("alert", "clip", "zip"):
        tid = store.create(type=TaskType.MEDIA_IMPORT, created_by="u", channel="api", params={"kind": kind})
        o._run_media_import(tid)
        vols = runner.specs["media-import"].volumes
        assert ("nas-recordings" in vols) == (kind != "zip"), kind
    o.media_nas_pvc = None
    tid = store.create(type=TaskType.MEDIA_IMPORT, created_by="u", channel="api", params={"kind": "alert"})
    o._run_media_import(tid)
    assert store.tasks[tid]["status"] == "failed"                 # 沒 NAS 的環境直接說清楚，不跑


def test_label_task_is_export_option():
    from edgevision.media.download import FORMATS, _kpt_shape
    from edgevision.media.shapes import conversion_counts, to_yolo_lines
    it = _item()
    names = ["glove", "mat"]
    assert conversion_counts(it["shapes"], names, "detect") == {"polygon_to_box": 1, "keypoints_to_box": 0,
                                                                 "box_to_polygon": 0, "dropped": 0}
    assert conversion_counts(it["shapes"], names, "segment")["box_to_polygon"] == 1
    assert conversion_counts(it["shapes"], names, "pose")["dropped"] == 2          # 沒有關鍵點 → 都不出
    assert to_yolo_lines(it["shapes"], names, "pose") == []
    kp = {"type": "keypoints", "class": "mat", "bbox": [[0, 0], [1, 1]], "points": [[.1, .1, 2]] * 4}
    assert _kpt_shape([{"shapes": [kp]}]) == [4, 3] and _kpt_shape([it]) is None
    assert "yolo-pose" in FORMATS
