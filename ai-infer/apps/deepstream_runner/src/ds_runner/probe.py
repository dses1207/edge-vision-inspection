"""probe：掛在 nvinferserver 的 src pad 上，每個 batch 被 GStreamer 呼叫一次。

做兩件事：
  ① 【同步】解 bbox → 掛成 NvDsObjectMeta，讓下游 nvdsosd 自動畫框
  ② 【非同步】把原始張量丟 queue，worker 去做 mask 重建與站別判定

⭐ 為什麼 bbox 要同步、mask 與判定要非同步：

  nvdsosd 只認 NvDsBatchMeta 裡的 meta，不管誰寫進去（官方 deepstream-ssd-parser
  就是在 Python probe 裡解析並掛 meta）。但 meta 必須在 buffer 流到 nvdsosd 之前
  就掛好 —— 如果等 worker thread 算完再回來掛，那一幀早就流過去了。

  所以「要畫的東西」必須同步解。實測支持這個切法（8400 anchors、20 路一批）：
      只解 bbox（向量化）   0.4ms/台 → 20 台 8.3ms   ✅ 佔 15fps 預算的 12%
      含 mask 重建         21.6ms/台 → 20 台 432ms   ❌ 遠超預算
  mask 重建佔了絕大多數時間，而它不影響畫框 —— 留在 worker 即可。

  站別判定（OK/NG）同理留在 worker：它產出的是「這一幀的結論」，不是要畫的框。
  代價是 NG 標示會比框晚幾幀，這是刻意接受的取捨。

⚠️ probe 在 GStreamer 的 critical path 上，這裡的任何阻塞都會卡住整條管線。
   queue 滿就丟幀，絕不等待。

pyds 只在這個檔與 pipeline.py 出現；libs / stations 永遠不 import 它。
"""

from __future__ import annotations

import ctypes
import queue
from typing import Any

import numpy as np
from edgevision_postprocess.colors import box_color_rgb

from ds_runner.worker import TensorJob


def _layer_to_numpy(pyds, layer) -> tuple[np.ndarray, list[int]]:
    dims = layer.inferDims
    shape = [dims.d[i] for i in range(dims.numDims)]
    n = 1
    for s in shape:
        n *= s
    ptr = ctypes.cast(pyds.get_ptr(layer.buffer), ctypes.POINTER(ctypes.c_float))
    return np.ctypeslib.as_array(ptr, shape=(n,)).copy().reshape(shape), shape


def _collect_tensors(pyds, l_user, by_gie: dict[int, list]) -> None:
    """把一條 user meta 串列裡的 NvDsInferTensorMeta 依 unique_id（哪個 nvinferserver）收進 by_gie。

    幀（全幅模式）與物件（ROI 假物件模式）的 user meta 串列格式相同，共用這段。
    """
    while l_user:
        um = pyds.NvDsUserMeta.cast(l_user.data)
        if um.base_meta.meta_type == pyds.NVDSINFER_TENSOR_OUTPUT_META:
            tmeta = pyds.NvDsInferTensorMeta.cast(um.user_meta_data)
            arrays = by_gie.setdefault(int(tmeta.unique_id), [])
            for i in range(tmeta.num_output_layers):
                arrays.append(_layer_to_numpy(pyds, pyds.get_nvds_LayerInfo(tmeta, i)))
        try:
            l_user = l_user.next
        except StopIteration:
            break


def split_outputs(arrays: list[tuple[np.ndarray, list[int]]], proto_hw: int = 160):
    """從 output layers 分出 (out0, proto)：proto 是 3 維且含 160 的那個。純函式，可測。

    兩種 seg 格式都認：
      raw    output0 (38, 8400)      + output1 (32, 160, 160)     → proto 是 prototypes
      fused  output0 (max_det, 38)   + masks   (max_det, 160, 160) → proto 是現成的 mask
    第二個位置的語意由後處理依 out0 的 shape 決定（YoloSegPostprocessor.is_fused），
    這裡只負責把「那個 3 維含 160 的」跟「另一個」分開，不解讀。
    """
    out0 = proto = None
    for arr, shape in arrays:
        if len(shape) >= 3 and proto_hw in shape:
            proto = arr
        else:
            out0 = arr
    return out0, proto


class BoxCache:
    """每路相機「上一次推論的框」。

    nvinferserver 設了 interval 之後，只有被推論的那一幀帶 tensor meta，其餘幀沒有東西可畫，
    燒錄畫面會變成一閃一閃（實測 interval=5 時 40 幀只有 4 幀有框）。
    沒推論的幀就沿用上一次的框；超過 max_stale 幀沒有新結果就不畫（來源斷線時別留殘影）。
    """

    def __init__(self, max_stale: int = 30) -> None:
        self.max_stale = max_stale
        self._dets: dict[int, list] = {}
        self._age: dict[int, int] = {}

    def update(self, source_id: int, dets: list) -> list:
        self._dets[source_id] = dets
        self._age[source_id] = 0
        return dets

    def get(self, source_id: int) -> list:
        """沒有新結果的幀呼叫：回上一次的框（age +1），太舊回空。"""
        if source_id not in self._dets:
            return []
        self._age[source_id] += 1
        if self._age[source_id] > self.max_stale:
            self._dets.pop(source_id, None)
            self._age.pop(source_id, None)
            return []
        return self._dets[source_id]


class DrawItem:
    """一個要畫的東西：框（Detection）+ 選配的 mask 小塊（prototype 解析度、已對齊框的 float32）。"""
    __slots__ = ("det", "mask")

    def __init__(self, det, mask=None) -> None:
        self.det, self.mask = det, mask


MASK_THRESHOLD = 0.5


def _attach_mask(obj, mask: np.ndarray) -> None:
    """把 float mask 小塊塞進 obj_meta.mask_params；nvdsosd display-mask=1 會在 GPU 拉伸到框、二值化、上色。

    實測（DS 7.1 / pyds 1.2.0，離線 videotestsrc 驗證）：
      · size 要先設好（bytes），alloc_mask_array() 才會配出對應大小的可寫 float32 view
      · mask 被拉伸到【正好】rect_params 的範圍；顏色用 rect_params.border_color，半透明
      · 釋放跟著 obj_meta 回池子走，probe 不用管
    """
    mh, mw = mask.shape
    mp = obj.mask_params
    mp.width, mp.height = int(mw), int(mh)
    mp.threshold = MASK_THRESHOLD
    mp.size = int(mw * mh * 4)
    arr = mp.alloc_mask_array()
    np.copyto(arr.reshape(-1), mask.reshape(-1))


def visible(items: list[DrawItem], hidden) -> list[DrawItem]:
    """拿掉不畫的類別（模型設定 hidden_classes）。只影響畫框；判定吃的是整份張量，不經過這裡。"""
    return [it for it in items if it.det.class_name not in hidden] if hidden else items


# 框色：與存檔（edgevision_postprocess.render）同一套，錯誤類紅色、其他高飽和色（見 edgevision_postprocess.colors）


def _draw_boxes(pyds, batch_meta, frame_meta, items, class_names, stats: dict | None = None) -> int:
    """把 Detection 掛成 NvDsObjectMeta，下游 nvdsosd 會自動畫框與標籤（有 mask 小塊就連 mask 一起）。

    這是官方模式（deepstream-ssd-parser）：probe 解析完直接掛 meta，
    nvdsosd 不做推論、只讀 NvDsBatchMeta 繪製。

    items 可以是 DrawItem（框 + 選配 mask）或裸 Detection（只畫框）。
    ⚠️ obj_meta 的池子有上限，取不到就停止（回傳實際畫了幾個），不要硬取。
    """
    n = 0
    for it in items:
        d = it.det if isinstance(it, DrawItem) else it
        mask = it.mask if isinstance(it, DrawItem) else None
        try:
            obj = pyds.nvds_acquire_obj_meta_from_pool(batch_meta)
        except Exception:   # noqa: BLE001 —— meta 池取不到就停止畫，probe 絕不拋
            break
        if obj is None:
            break
        b = d.bbox
        r = obj.rect_params
        r.left, r.top = float(b.x1), float(b.y1)
        r.width, r.height = float(b.x2 - b.x1), float(b.y2 - b.y1)
        r.border_width = 3
        cr, cg, cb = box_color_rgb(d.class_name, d.class_id)
        r.border_color.set(cr, cg, cb, 1.0)
        r.has_bg_color = 0

        obj.class_id = d.class_id
        obj.confidence = float(d.confidence)
        obj.object_id = 0xFFFFFFFFFFFFFFFF        # UNTRACKED_OBJECT_ID：沒有 tracker
        # 這兩個欄位 nvdsosd 不用，但下游（例如 nvmsgconv）會讀
        obj.obj_label = d.class_name

        t = obj.text_params
        t.display_text = f"{d.class_name} {d.confidence * 100:.0f}%"
        t.x_offset, t.y_offset = int(b.x1), max(0, int(b.y1) - 22)
        t.font_params.font_name = "Serif"
        t.font_params.font_size = 12
        t.font_params.font_color.set(1.0, 1.0, 1.0, 1.0)
        t.set_bg_clr = 1
        t.text_bg_clr.set(cr, cg, cb, 0.85)

        if mask is not None:
            try:
                _attach_mask(obj, mask)
            except Exception:   # noqa: BLE001 —— mask 掛不上就只畫框，probe 絕不拋
                if stats is not None:
                    stats["mask_err"] = stats.get("mask_err", 0) + 1

        pyds.nvds_add_obj_meta_to_frame(frame_meta, obj, None)
        n += 1
    return n


def make_probe(q: queue.Queue[TensorJob], src_to_cam: dict[int, str], width: int, height: int,
               stats: dict, *, bbox_only=None, class_names=None, draw: bool = False,
               draw_masks: bool = False, fill_skipped: bool = True, bindings: dict[int, Any] | None = None,
               roi_by_source: dict[int, tuple[int, int, int, int]] | None = None,
               roi_gie_id: int | None = None, with_pts: bool = False, block_on_full: bool = False):
    """建立 probe。

    draw=True 時，probe 會用 bbox_only（只解框、不重建 mask 的後處理器）同步算出
    框並掛成 obj meta —— 這是「燒錄」路線的核心：框與畫面必然同幀。
    draw=False 時退回原本行為（只搬張量），供不需要輸出影像的線路使用。

    draw_masks=True 時，同步路徑改走 decode_with_mask_crops：每個框多算一塊
    prototype 解析度的 mask 小塊掛進 mask_params，nvdsosd（display-mask=1）在 GPU 拉伸上色。
    這跟原本「mask 太貴不能同步」不矛盾 —— 貴的是放大到原圖 + 找輪廓，那些現在都不在 CPU 做。
    分割以外的模型（沒有 proto）自動退回只畫框。

    fill_skipped 決定【沒被推論的幀】怎麼辦：
      True （沒有 tracker）→ BoxCache 把上一次的框原地重畫。框不閃，但會「凍住」再跳。
      False（有 nvtracker）→ probe 什麼都不做，交給 tracker 用 Kalman 外推。框會平滑移動。
                             ⚠️ 這時候絕不能再重畫快取 —— tracker 已經補了一份，會疊成兩個框。

    bindings：一線多站。{nvinferserver unique_id: 物件(.key .postproc .class_names .primary .hidden)}。
      .hidden：不畫的類別名（模型設定 hidden_classes）—— 只從畫框濾掉，張量照樣整份丟給 worker 判定。
      每個 nvinferserver 各自掛 tensor meta（NvDsInferTensorMeta.unique_id 分辨），
      這裡逐個解框、掛 meta（hidden 以外都畫）、丟 TensorJob(station=key) 給 worker。
      沒給就是單站：{1: bbox_only/class_names}，行為與以前完全相同。

    with_pts：TensorJob 帶這幀的 buf_pts（影片測試；線上不帶，線上的結果格式不變）。
    block_on_full：queue 滿了就等（回壓上游、不丟幀）。只給不限速的影片測試用 ——
      線上絕不能等：等就是卡住 streaming thread，畫面跟著延遲。

    roi_gie_id：主站走 ROI 假物件模式（見 roi.py）時給。主站的 tensor meta 不在幀上，
      而在 unique_component_id == roi_gie_id 的假物件上；這裡讀出來併進同一個 by_gie，
      然後把假物件刪掉 —— 下游 tracker / nvdsosd 不該看到它。
    """
    import gi
    gi.require_version("Gst", "1.0")
    from time import perf_counter

    import pyds
    from gi.repository import Gst

    if bindings is None:
        from types import SimpleNamespace
        bindings = {1: SimpleNamespace(key=None, postproc=bbox_only, class_names=class_names, primary=True)}

    cache = BoxCache(max_stale=int(stats.get("box_max_stale", 30)))
    lbs: dict[int, Any] = {}            # gie → 全幅 letterbox（尺寸固定，算一次就好）
    lbs_roi: dict[int, Any] = {}        # source_id → 主站 ROI 的 letterbox（只有主站裁 ROI）
    if draw:
        from edgevision_postprocess import compute_letterbox, compute_letterbox_roi
        for gie, b in bindings.items():
            if b.postproc is not None:
                lbs[gie] = compute_letterbox(width, height, b.postproc.imgsz)
                if b.primary and roi_by_source:
                    for sid_, roi in roi_by_source.items():
                        lbs_roi[sid_] = compute_letterbox_roi(*roi, b.postproc.imgsz)

    def probe(pad, info, _user):
        buf = info.get_buffer()
        if not buf:
            return Gst.PadProbeReturn.OK
        batch = pyds.gst_buffer_get_nvds_batch_meta(hash(buf))
        l_frame = batch.frame_meta_list
        while l_frame:
            fmeta = pyds.NvDsFrameMeta.cast(l_frame.data)
            by_gie: dict[int, list] = {}
            _collect_tensors(pyds, fmeta.frame_user_meta_list, by_gie)
            if roi_gie_id is not None:
                fakes = []
                l_obj = fmeta.obj_meta_list
                while l_obj:
                    obj = pyds.NvDsObjectMeta.cast(l_obj.data)
                    if obj.unique_component_id == roi_gie_id:
                        _collect_tensors(pyds, obj.obj_user_meta_list, by_gie)
                        fakes.append(obj)
                    try:
                        l_obj = l_obj.next
                    except StopIteration:
                        break
                for obj in fakes:               # 走完串列再刪，邊走邊刪會斷鏈
                    pyds.nvds_remove_obj_meta_from_frame(fmeta, obj)
            sid = fmeta.source_id
            cam = src_to_cam.get(sid)
            if cam is not None:
                # 相機層級健康：這台相機真的有幀進來（跳幀沒推論的也算）。斷線的來源不會出現在 batch 裡。
                cf = stats.setdefault("cam_frames", {})
                cf[cam] = cf.get(cam, 0) + 1
            for gie, b in bindings.items():
                out0, proto = split_outputs(by_gie.get(gie, []))
                can_draw = draw and b.postproc is not None
                if out0 is None:
                    # 這一幀這個模型沒推論（interval 跳過）：沿用上一次的框，畫面才不會一閃一閃。
                    # 有 tracker 時跳過這段 —— tracker 會自己補，重畫會變兩份框。
                    if can_draw and fill_skipped:
                        cached = cache.get((sid, gie))
                        if cached:
                            try:
                                stats["drawn_cached"] = stats.get("drawn_cached", 0) + _draw_boxes(
                                    pyds, batch, fmeta, cached, b.class_names, stats)
                            except Exception:   # noqa: BLE001 —— 畫框失敗只計數，probe 絕不拋
                                stats["draw_err"] = stats.get("draw_err", 0) + 1
                    continue
                if b.primary:
                    # 有推論結果的幀計數：心跳的 fps 由這個差分算（不管有沒有畫框）
                    stats["frames"] = stats.get("frames", 0) + 1
                if cam is None:
                    continue
                # ① 同步：解 bbox 掛 meta。draw_masks 時多掛一塊 mask 小塊（不放大、不找輪廓，
                #    那些才是慢 50 倍的部分 —— 放大與上色交給 nvdsosd 在 GPU 做）
                if can_draw:
                    t0 = perf_counter()
                    lb = lbs_roi.get(sid, lbs[gie]) if b.primary else lbs[gie]
                    try:
                        if draw_masks and proto is not None and hasattr(b.postproc, "decode_with_mask_crops"):
                            items = [DrawItem(d, m) for d, m in
                                     b.postproc.decode_with_mask_crops(out0, proto, lb, width, height)]
                        else:
                            items = [DrawItem(d) for d in b.postproc(out0, None, lb, width, height)]
                        items = visible(items, getattr(b, "hidden", None))
                        cache.update((sid, gie), items)
                        stats["drawn"] = stats.get("drawn", 0) + _draw_boxes(
                            pyds, batch, fmeta, items, b.class_names, stats)
                    except Exception:   # noqa: BLE001 —— 畫框失敗只計數，probe 絕不拋
                        stats["draw_err"] = stats.get("draw_err", 0) + 1
                    # 同步路徑的累計毫秒 / 幀數：部署後拿來算每幀平均，確認 mask 沒把 probe 拖慢
                    stats["sync_ms"] = stats.get("sync_ms", 0.0) + (perf_counter() - t0) * 1000.0
                    stats["sync_n"] = stats.get("sync_n", 0) + 1
                # ② 非同步：原始張量丟給 worker 做 mask 與判定
                job = TensorJob(cam, out0, proto, width, height, station=b.key,
                                pts=int(fmeta.buf_pts) if with_pts else None, frame_pts=int(fmeta.buf_pts))
                if block_on_full:
                    q.put(job)
                else:
                    try:
                        q.put_nowait(job)
                    except queue.Full:
                        stats["dropped"] = stats.get("dropped", 0) + 1
            try:
                l_frame = l_frame.next
            except StopIteration:
                break
        return Gst.PadProbeReturn.OK

    return probe
