"""範本站：畫面裡 target_class 的數量 ≥ max_count，且連續 persist_frames 幀 → NG。

刻意簡單，用來示範一個站別的最小組成：
  ・建構子只吃 camera_params（現場值）
  ・evaluate 純計算，回 Verdict，details 放診斷用的數字
  ・跨幀狀態用 edgevision_rules 的原語，不自己數
  ・update_params 熱套用後把狀態歸零
"""

from __future__ import annotations

from typing import Any

from edgevision_contracts import Detection, Verdict
from edgevision_rules import PersistCounter


class TemplateStation:
    def __init__(self, camera_params: dict[str, Any]) -> None:
        self._apply(camera_params)

    def evaluate(self, detections: list[Detection], width: int, height: int) -> Verdict:
        n = sum(1 for d in detections if d.class_name == self.target_class)
        condition = n >= self.max_count
        confirmed = self._persist.update(condition)
        detail = {"count": n, "max_count": self.max_count, "streak": self._persist.streak}
        if confirmed:
            return Verdict.ng(f"{self.target_class} 達 {n} 個", **detail)
        return Verdict.ok("", **detail)

    def classes_used(self) -> set[str]:
        """依目前參數，判定會用到的模型類別名（runner 啟動時拿去比對模型的真實類別）。"""
        return {self.target_class}

    def reset(self) -> None:
        self._persist.reset()

    def update_params(self, camera_params: dict[str, Any]) -> None:
        self._apply(camera_params)

    def _apply(self, p: dict[str, Any]) -> None:
        self.target_class = str(p["target_class"])
        self.max_count = int(p["max_count"])
        self._persist = PersistCounter(int(p.get("persist_frames", 1)))
