"""示範站：防護具違規 → NG。示範「附加站」：一條線除了主站，再掛一個吃第二模型的站。

步驟：
  1. 只留 violations 列的類別、過各自的信心門檻
  2. 滑動視窗確認：最近 window_size 次判定裡，有 ≥ min_count 次在相近位置（IoU > iou_threshold）看到同類別 → 確認
     （開放詞彙模型單幀很吵，靠時間上的一致性濾掉閃一下的誤判）
  3. 確認後依「原因」冷卻：同一原因 report_cooldown_seconds 內只報一次；同幀同原因取信心最高

附加站通常不是每幀都跑（線路設定的 interval），視窗數的是 evaluate 次數，不是影格數。
"""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field, fields
from typing import Any

from edgevision_contracts import Detection, Verdict
from edgevision_rules import Cooldown, bbox_iou, filter_by_roi

DEFAULT_VIOLATIONS = {"no_helmet": "未戴安全帽", "no_vest": "未穿反光背心"}


@dataclass(frozen=True, slots=True)
class PpeThresholds:
    violations: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_VIOLATIONS))   # 類別 → 通報原因
    min_conf: dict[str, float] = field(default_factory=dict)
    default_conf: float = 0.3
    window_size: int = 10
    min_count: int = 5
    iou_threshold: float = 0.2
    report_cooldown_seconds: float = 60.0

    @classmethod
    def from_params(cls, overrides: dict | None) -> PpeThresholds:
        if not overrides:
            return cls()
        known = {f.name for f in fields(cls)}
        unknown = set(overrides) - known
        if unknown:
            raise ValueError(f"thresholds 有不認識的欄位 {sorted(unknown)}，可用：{sorted(known)}")
        return cls(**overrides)


class PpeCheckStation:
    clock: Callable[[], float] = time.monotonic

    def __init__(self, camera_params: dict[str, Any]) -> None:
        self._apply(camera_params)

    def _extract(self, detections: list[Detection]) -> list[tuple[tuple[float, float, float, float], str, float]]:
        th = self.th
        return [((d.bbox.x1, d.bbox.y1, d.bbox.x2, d.bbox.y2), d.class_name, d.confidence)
                for d in detections
                if d.class_name in th.violations and d.confidence >= th.min_conf.get(d.class_name, th.default_conf)]

    def _confirm(self, current: list) -> list:
        self._window.append(current)
        confirmed = []
        for box, label, conf in current:
            count = sum(1 for frame in self._window
                        if any(lbl == label and bbox_iou(box, hb) > self.th.iou_threshold for hb, lbl, _ in frame))
            if count >= self.th.min_count:
                confirmed.append((label, conf, count))
        return confirmed

    def evaluate(self, detections: list[Detection], width: int, height: int) -> Verdict:
        detections = filter_by_roi(detections, self.roi_regions, width, height)
        current = self._extract(detections)
        confirmed = self._confirm(current)
        detail = {"candidates": len(current), "confirmed": [(l, round(c, 3), n) for l, c, n in confirmed]}
        best: dict[str, tuple[str, float]] = {}
        for label, conf, _ in confirmed:
            reason = self.th.violations[label]
            if self._report_cd[reason].ready() and (reason not in best or conf > best[reason][1]):
                best[reason] = (label, conf)
        if not best:
            return Verdict.ok("違規已通報過（冷卻中）" if confirmed else "", **detail)
        for reason in best:
            self._report_cd[reason].fire()
        return Verdict.ng("、".join(best), reported={r: {"label": l, "confidence": round(c, 3)}
                                                      for r, (l, c) in best.items()}, **detail)

    def classes_used(self) -> set[str]:
        return set(self.th.violations)

    def reset(self) -> None:
        self._window.clear()
        for cd in self._report_cd.values():
            cd.reset()

    def update_params(self, camera_params: dict[str, Any]) -> None:
        self._apply(camera_params)

    def _apply(self, p: dict[str, Any]) -> None:
        self.th = PpeThresholds.from_params(p.get("thresholds"))
        self.roi_regions = list(p.get("roi_regions") or [])
        self._window: deque[list] = deque(maxlen=self.th.window_size)
        clock = type(self).clock
        self._report_cd = {r: Cooldown(self.th.report_cooldown_seconds, clock=clock)
                           for r in set(self.th.violations.values())}
