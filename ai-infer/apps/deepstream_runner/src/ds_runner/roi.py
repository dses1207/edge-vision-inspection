"""主站 ROI 推論：讓模型只看每台相機的一塊 ROI（原程式的「裁切後再推論」）。

為什麼需要：舊程式有些站別先裁出畫面的一塊（例如左下 3/4）再放大餵模型，
模型也是在那個畫面上訓練的。餵全幅物件會變小，偵測品質不一樣。

做法（「假物件」模式，2026-09-29 取代 nvdspreprocess）：
  ① 主站 nvinferserver 的 sink pad 掛 injector probe：每幀替每台相機放一個 NvDsObjectMeta，
     框 = 該相機的 ROI（沒設 ROI 的相機給整幅），unique_component_id = ROI_GIE_ID
  ② 主站 nvinferserver 改成 PROCESS_MODE_CLIP_OBJECTS、只吃 ROI_GIE_ID 的物件 ——
     它自己裁框、等比縮放、對稱補邊、正規化（跟全幅模式同一段 preprocess 設定），
     tensor meta 掛在【那個物件】的 obj_user_meta_list 上
  ③ 畫框 probe 從假物件身上讀張量，讀完把假物件從幀上刪掉（下游 tracker / nvdsosd 看不到它）
  ④ 座標還原：模型輸出是「letterbox 後的 ROI」座標，compute_letterbox_roi 帶著 ROI 原點換回全幅

為什麼不用 nvdspreprocess：它的張量掛在 ROI meta（NvDsPreProcessBatchMeta / NvDsRoiMeta）上，
pyds 1.2.0 沒有這兩個 binding，probe 讀不到 —— 測試機實測 ROI 線 0 幀有結果。
物件模式的 tensor meta 走 obj_user_meta_list，pyds 讀得到。

只影響主站。附加站（PPE）照常全幅。
"""

from __future__ import annotations

import re

# 假 ROI 物件的 unique_component_id。不能跟任何 nvinferserver 的 unique_id（1, 2, … 站別數）撞到。
ROI_GIE_ID = 99
UNTRACKED_OBJECT_ID = 0xFFFFFFFFFFFFFFFF


def roi_pixels(norm: list[float], width: int, height: int) -> tuple[int, int, int, int]:
    """正規化 [x1, y1, x2, y2] → 像素，夾在畫面內，寬高至少 2。"""
    x1 = max(0, min(width - 2, round(norm[0] * width)))
    y1 = max(0, min(height - 2, round(norm[1] * height)))
    x2 = max(x1 + 2, min(width, round(norm[2] * width)))
    y2 = max(y1 + 2, min(height, round(norm[3] * height)))
    return x1, y1, x2, y2


_INPUT_CONTROL = re.compile(r"^input_control \{.*?^\}", re.DOTALL | re.MULTILINE)


def use_roi_objects(nvinfer_config: str, roi_gie_id: int = ROI_GIE_ID) -> str:
    """把渲染好的 nvinferserver 設定改成「只對 ROI 假物件裁切推論」。純函式，可測。

    interval 設 0：跳幀改由 injector 決定（沒放假物件的幀就不推論），
    不讓 nvinferserver 在物件模式下再套一次自己的 interval 語意。
    """
    block = ("input_control {\n"
             "  process_mode: PROCESS_MODE_CLIP_OBJECTS\n"
             f"  operate_on_gie_id: {roi_gie_id}\n"
             "  operate_on_class_ids: [0]\n"
             "  interval: 0\n"
             "}")
    out, n = _INPUT_CONTROL.subn(block, nvinfer_config, count=1)
    if n != 1:
        raise ValueError("nvinferserver 設定裡找不到 input_control 區塊")
    return out


class InjectSchedule:
    """每 interval+1 個 batch 放一次假物件（= nvinferserver 全幅模式的 interval 語意）。純邏輯，可測。"""

    def __init__(self, interval: int) -> None:
        self.every = max(0, int(interval)) + 1
        self._n = 0

    def tick(self) -> bool:
        hit = self._n % self.every == 0
        self._n += 1
        return hit


def make_roi_injector(rects: dict[int, tuple[int, int, int, int]], *, interval: int = 0,
                      roi_gie_id: int = ROI_GIE_ID, stats: dict | None = None):
    """主站 nvinferserver sink pad 上的 probe：每台相機放一個 ROI 假物件。

    rects：mux source_id → (x1, y1, x2, y2)，mux 畫面像素。沒列到的來源不放（= 不推論）。
    ⚠️ probe 在 critical path 上：取不到 meta 就跳過，絕不拋。
    """
    import pyds
    from gi.repository import Gst

    sched = InjectSchedule(interval)

    def probe(pad, info, _user):
        buf = info.get_buffer()
        if not buf or not sched.tick():
            return Gst.PadProbeReturn.OK
        batch = pyds.gst_buffer_get_nvds_batch_meta(hash(buf))
        l_frame = batch.frame_meta_list
        while l_frame:
            fmeta = pyds.NvDsFrameMeta.cast(l_frame.data)
            rect = rects.get(fmeta.source_id)
            if rect is not None:
                try:
                    obj = pyds.nvds_acquire_obj_meta_from_pool(batch)
                    x1, y1, x2, y2 = rect
                    r = obj.rect_params
                    r.left, r.top, r.width, r.height = float(x1), float(y1), float(x2 - x1), float(y2 - y1)
                    r.border_width = 0
                    r.has_bg_color = 0
                    obj.unique_component_id = roi_gie_id
                    obj.class_id = 0
                    obj.confidence = 1.0
                    obj.object_id = UNTRACKED_OBJECT_ID   # 不讓 nvinferserver 依 object_id 快取、跳過重推
                    pyds.nvds_add_obj_meta_to_frame(fmeta, obj, None)
                except Exception:   # noqa: BLE001
                    if stats is not None:
                        stats["roi_inject_err"] = stats.get("roi_inject_err", 0) + 1
            try:
                l_frame = l_frame.next
            except StopIteration:
                break
        return Gst.PadProbeReturn.OK

    return probe
