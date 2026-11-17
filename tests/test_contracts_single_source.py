"""契約只有一份：主 repo 的 libs.contracts 偵測契約就是 ai-infer 的 edgevision-contracts 套件。

以前是兩份手抄，抄漏過 keypoints 與 FrameResult.station，兩次都是靜默資料遺失
（event-processor 解析時 Pydantic 丟掉不認識的欄位）。現在改成同一份程式碼 + 陌生欄位警告。
"""
import json
import logging
import sys
from pathlib import Path

import edgevision_contracts

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from edgevision.storage.event_processor import EventProcessor, unknown_fields  # noqa: E402
from libs.contracts import Detection, FrameResult  # noqa: E402


def test_same_classes_not_a_copy():
    assert FrameResult is edgevision_contracts.FrameResult
    assert Detection is edgevision_contracts.Detection
    assert "station" in FrameResult.model_fields


def _payload(**extra) -> dict:
    return {
        "camera_id": "cam50", "frame_id": "f1", "model_name": "m", "model_version": "1",
        "image_meta": {"width": 640, "height": 480},
        "detections": [{"class_id": 2, "class_name": "part", "confidence": 0.9,
                        "bbox": {"x1": 0, "y1": 0, "x2": 10, "y2": 10}}],
        **extra,
    }


def test_unknown_fields_found_at_every_level():
    p = _payload(new_top=1, station="zone_pose", type="result")   # type 是 runner 的傳輸標記，不算
    p["detections"][0]["new_det"] = 2
    p["verdict"] = {"status": "ng", "new_v": 3}
    assert unknown_fields(p) == {"new_top", "detections.new_det", "verdict.new_v"}


def test_handle_warns_once_and_keeps_station(caplog):
    written = []

    class DB:
        def insert_detections(self, result):
            written.append(result)
            return len(result.detections)

    ep = EventProcessor.__new__(EventProcessor)
    from edgevision.storage.event_processor import ProcessorStats
    ep.stats, ep._warned_fields, ep.db = ProcessorStats(), set(), DB()
    payload = json.dumps(_payload(station="zone_pose", brand_new=True, verdict={"status": "ng", "reason": "x"}))
    with caplog.at_level(logging.WARNING):
        ep.handle(payload)
        ep.handle(payload)
    assert written[0].station == "zone_pose"
    assert sum("brand_new" in r.message for r in caplog.records) == 1
