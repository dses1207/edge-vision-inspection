"""站別必須長成的形狀。只約束進出格式，中間邏輯完全自由。"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from edgevision_contracts import Detection, Verdict


@runtime_checkable
class Station(Protocol):
    """一個站別 = 一個純計算物件。每台相機一個實例（狀態如連續幀計數不可跨相機共用）。

    建構子簽章固定：``Station(camera_params: dict)``。
    camera_params 是這台相機的現場值（ref_quad、ROI、門檻覆寫…），
    需要哪些鍵由 station.yaml 的 camera_params_required 宣告，載入時先驗過才會建構。
    """

    def evaluate(self, detections: list[Detection], width: int, height: int) -> Verdict:
        """一幀的偵測結果 → OK / NG。不碰網路、DB、檔案。"""
        ...

    def reset(self) -> None:
        """清掉跨幀狀態（串流重連、相機停用再啟用時呼叫）。"""
        ...

    def update_params(self, camera_params: dict[str, Any]) -> None:
        """熱套用新的現場值，不重建實例。實例自己決定要不要順便 reset()。"""
        ...

    def classes_used(self) -> set[str]:
        """依目前現場值，判定會用到的模型類別名（例：hand_class、violations 的 key）。

        runner 啟動時拿去比對模型的真實類別（加上 roi_regions[].classes），有不存在的就拒絕啟動 ——
        打錯字不會變成「永遠偵測不到」。判定邏輯只講角色，類別名一律走現場值（有預設），不寫死在程式裡。
        """
        ...
