"""fixture 回放：拿現場錄下的偵測結果 + 預期判定，離線驗證站別。不需 GPU、不需相機。

fixture 格式（JSON）：
{
  "width": 1280, "height": 720,
  "camera_params": {"ref_quad": [...]},
  "frames": [
    {"detections": [<Detection>...], "expect": "ok" | "ng"},
    ...
  ]
}
frames 依序餵入同一個實例，所以連續幀去抖這種跨幀行為也能驗。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from edgevision_contracts import Detection, Verdict

from edgevision_station_sdk.loader import load_station


@dataclass
class FrameOutcome:
    index: int
    expected: str
    verdict: Verdict

    @property
    def passed(self) -> bool:
        return self.verdict.status.value == self.expected


def load_fixture(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def replay(station_name: str, fixture: dict) -> list[FrameOutcome]:
    station = load_station(station_name, fixture.get("camera_params", {}))
    w, h = fixture["width"], fixture["height"]
    out = []
    for i, frame in enumerate(fixture["frames"]):
        dets = [Detection.model_validate(d) for d in frame["detections"]]
        out.append(FrameOutcome(i, frame["expect"], station.evaluate(dets, w, h)))
    return out


def assert_fixture(station_name: str, path: str | Path) -> None:
    """給 pytest 用：任何一幀判錯就 fail，訊息帶幀號、預期、實際與 details。"""
    failures = [o for o in replay(station_name, load_fixture(path)) if not o.passed]
    if failures:
        lines = [f"frame {o.index}: 預期 {o.expected}，實得 {o.verdict.status.value} {o.verdict.details}" for o in failures]
        raise AssertionError(f"{Path(path).name} 有 {len(failures)} 幀判錯：\n" + "\n".join(lines))
