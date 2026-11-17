"""示範站：觸發後限時確認（「A 出現後 N 秒內要看到 B」）。

例：堆高機（trigger_class）駛入畫面後 10 秒內要看到引導員（required_class），否則 NG。

這類規則的 NG 是「時間到了 B 沒來」，NG 那一幀畫面上可能一個框都沒有 ——
所以 NG 判斷不能看「有沒有框」，下游（publish、recorder、存檔）一律以 Verdict 為準。
計時交給 edgevision_rules.WaitWindow；late_ok 決定 B 逾時後才來算不算數。

「一次進場算一次」：A 要先消失 rearm_gap_seconds 秒以上，再出現才會重新計時。
否則 A 停在畫面上時，B 一來一走就會被當成新的一輪；偵測漏個一兩幀也不會被誤當成重新進場。
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

from edgevision_contracts import Detection, Verdict
from edgevision_rules import Cooldown, WaitWindow, filter_by_roi


class FollowupWindowStation:
    clock: Callable[[], float] = time.monotonic

    def __init__(self, camera_params: dict[str, Any]) -> None:
        self._apply(camera_params)

    def _seen(self, detections: list[Detection], cls: str) -> bool:
        return any(d.class_name == cls and d.confidence >= self.min_conf for d in detections)

    def _arrived(self, detections: list[Detection]) -> bool:
        """A 這幀在、而且之前已經消失超過 rearm_gap_seconds（或從沒出現過）→ 新的一次進場。"""
        now = type(self).clock()
        if not self._seen(detections, self.trigger_class):
            return False
        new = self._last_trigger is None or now - self._last_trigger > self.rearm_gap
        self._last_trigger = now
        return new

    def evaluate(self, detections: list[Detection], width: int, height: int) -> Verdict:
        detections = filter_by_roi(detections, self.roi_regions, width, height)
        state = self._window.update(trigger=self._arrived(detections),
                                    satisfied=self._seen(detections, self.required_class))
        detail = {"state": state, "elapsed": self._window.elapsed()}
        if state != WaitWindow.TIMEOUT:
            return Verdict.ok("", **detail)
        if not self._cooldown.fire():
            return Verdict.ok("冷卻中", **detail)
        return Verdict.ng(f"{self.trigger_class} 出現後 {self.seconds:g} 秒內未見 {self.required_class}", **detail)

    def classes_used(self) -> set[str]:
        return {self.trigger_class, self.required_class}

    def reset(self) -> None:
        self._last_trigger = None
        self._window.reset()
        self._cooldown.reset()

    def update_params(self, camera_params: dict[str, Any]) -> None:
        self._apply(camera_params)

    def _apply(self, p: dict[str, Any]) -> None:
        th = p.get("thresholds") or {}
        self.trigger_class = str(p["trigger_class"])
        self.required_class = str(p["required_class"])
        self.seconds = float(th.get("seconds", 10.0))
        self.min_conf = float(th.get("min_conf", 0.4))
        self.rearm_gap = float(th.get("rearm_gap_seconds", 3.0))
        self._last_trigger: float | None = None
        self.roi_regions = list(p.get("roi_regions") or [])
        clock = type(self).clock
        self._window = WaitWindow(self.seconds, clock=clock, late_ok=bool(th.get("late_ok", False)))
        self._cooldown = Cooldown(float(th.get("cooldown_seconds", 0.0)), clock=clock)
