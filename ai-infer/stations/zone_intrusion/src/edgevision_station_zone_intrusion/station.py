"""示範站：禁區闖入。

規則：target_classes 的物件「腳點」（框底邊中點）落在 zone 多邊形內，連續 persist_frames 幀 → NG；
NG 之後 cooldown_seconds 秒內同一台相機不再報（跨幀狀態一律交給 edgevision_rules 的原語）。

用腳點而不是框中心：人站在禁區線外、上半身探進去時框中心會在區內，腳點不會 —— 判定要跟現場的地面標線一致。
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

from edgevision_contracts import Detection, Verdict
from edgevision_rules import Cooldown, PersistCounter, filter_by_roi, point_in_polygon


class ZoneIntrusionStation:
    clock: Callable[[], float] = time.monotonic

    def __init__(self, camera_params: dict[str, Any]) -> None:
        self._apply(camera_params)

    def _inside(self, d: Detection, width: int, height: int) -> bool:
        fx = (d.bbox.x1 + d.bbox.x2) / 2 / width
        fy = d.bbox.y2 / height
        return point_in_polygon(fx, fy, self.zone)

    def evaluate(self, detections: list[Detection], width: int, height: int) -> Verdict:
        detections = filter_by_roi(detections, self.roi_regions, width, height)
        intruders = [d for d in detections
                     if d.class_name in self.target_classes and d.confidence >= self.min_conf
                     and self._inside(d, width, height)]
        confirmed = self._persist.update(bool(intruders))
        detail = {"intruders": len(intruders), "streak": self._persist.streak}
        if not confirmed:
            return Verdict.ok("", **detail)
        if not self._cooldown.fire():
            return Verdict.ok("冷卻中", **detail)
        names = sorted({d.class_name for d in intruders})
        return Verdict.ng(f"{'、'.join(names)} 進入禁區", **detail)

    def classes_used(self) -> set[str]:
        return set(self.target_classes)

    def reset(self) -> None:
        self._persist.reset()
        self._cooldown.reset()

    def update_params(self, camera_params: dict[str, Any]) -> None:
        self._apply(camera_params)

    def _apply(self, p: dict[str, Any]) -> None:
        self.zone = [tuple(map(float, pt)) for pt in p["zone"]]
        if len(self.zone) < 3:
            raise ValueError("zone 至少要 3 個點")
        th = p.get("thresholds") or {}
        self.target_classes = set(th.get("target_classes", ["person"]))
        self.min_conf = float(th.get("min_conf", 0.4))
        self.roi_regions = list(p.get("roi_regions") or [])
        self._persist = PersistCounter(int(th.get("persist_frames", 3)))
        self._cooldown = Cooldown(float(th.get("cooldown_seconds", 10.0)), clock=type(self).clock)
