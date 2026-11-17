"""管線骨架：nvurisrcbin ×N → nvstreammux → nvinferserver(gRPC→Triton) → [demux → osd → 編碼 → 推流]。

輸出兩條路：
  · 判定結果 → Redis（worker thread，非同步）
  · 監看影像 → 框燒進畫面 → 推回 go2rtc 的 <cam>_marked（RESTREAM=1 時才建）

⭐ 為什麼框是「燒錄」不是「前端疊加」：
  影像與偵測結果走兩條路徑，延遲不同且會浮動，前端疊加必然對不準
  。框燒進畫面後兩者天生同幀。
  代價是前端不能開關框線 —— 已確認不需要。
  乾淨影像仍存在（NAS 連續錄影、MinIO 事件存證），監看畫面燒錄不損失任何資料。

一條線路一個進程。這裡沒有任何站別名詞：站名來自 LineConfig，站別實例由 worker 透過 SDK 建立。

環境變數：
  CAMERA_SOURCE=file|api、CAMERA_FILE / CONFIG_API、LINE_ID   → 見 config.py
  MODEL_NAME、TRITON_GRPC      填進 nvinferserver 設定樣板（模型名跟 Triton 的 model repo 走，不在線路設定裡）
  NVINFER_TEMPLATE             樣板路徑；NVINFER_CONFIG 產出路徑
  INFER_INTERVAL               推論間隔覆寫；沒設就用 line_params 的 source_fps/target_fps 算，都沒有 → 0（每幀推論）
  GO2RTC_RTSP                  預設 rtsp://localhost:8554；推 <cam>_marked 到這裡（前端看的那台 go2rtc）
  GO2RTC_SOURCE_RTSP           選填，拉相機原始串流的 go2rtc；沒設 = GO2RTC_RTSP。
                               推論站點在別的節點時，填那個節點本機的 go2rtc，原始畫面才不用跨節點繞一圈
  TRITON_HTTP                  選填，例如 localhost:8000；有設就查模型版本寫進 FrameResult.model_version，並在啟動時比對 engine 尺寸 / 類別數
  MODEL_CHECK                  1=啟動時比對 Triton 上的 engine 與 line config 的 imgsz / 類別數，不符就不啟動（預設 1）
  REDIS_URL / REDIS_STREAM / REDIS_STREAM_LIVE   結果發布（格式同舊 EventBus）；沒設就只印 log
  WORKER_THREADS               worker 分片數，預設 min(4, 相機數)；每台相機固定歸一條
  QUEUE_SIZE                   每個 shard 的 queue 深度，預設 20（吸收抖動，不是緩衝區）
  STATS_INTERVAL_S             probe 統計輸出間隔秒數，預設 30；0=關閉
  STARTUP_TIMEOUT_S            啟動後幾秒內要有幀過推論，否則回報 failed 並結束（見 watchdog.py；預設 120；0=關閉）
  NG_COOLDOWN_S                同一台相機兩次 NG 至少間隔秒數（預設 30；line_params.ng_cooldown_s 優先；0=不擋，見 worker._ng_cooldown）
  RESUME_GAP_S / RESUME_HOLD_S 同一台相機幾秒沒新幀算斷流（預設 2）；恢復後幾秒內不判定（預設 5；GAP 0=關閉，見 worker._resume_hold）
  RUN_STALL_S                  跑起來後推論幾秒沒新幀（相機也沒報錯）就回報 failed 並結束（RunWatchdog；預設 180；0=關閉）
  CAM_STALL_S                  單台相機幾秒沒新幀就只重建那一台（CameraStallWatchdog；預設 90；0=關閉）
  BOX_DUMP                     除錯：把每幀最終要畫的框寫成 JSONL 到這個路徑
  RESTREAM_ENABLED             1=建立打框推流分支（預設 1）；0=只推論不輸出影像
  DRAW_MASKS                   1=mask 也燒進 <cam>_marked（預設 1，nvdsosd 在 GPU 畫）；0=只燒框
  INFER_ROI_ENABLED            1=相機有 infer_roi 就讓主站只推論 ROI（假物件模式，見 roi.py；預設 1）；0=忽略 infer_roi 全幅推論
  RESTREAM_BITRATE             編碼位元率，預設 2000000
  RESTREAM_SUFFIX              推流名稱後綴，預設 _marked
  TRACKER_ENABLED              1=在 infer 後插入 nvtracker（跳幀時用 Kalman 外推補框）；預設 0
  TRACKER_CONFIG               tracker 低階設定檔；預設 /app/tracker_nvsort.yml
  TRACKER_WIDTH / TRACKER_HEIGHT  tracker 內部運作解析度（須為 32 倍數），預設 640 / 384

影片測試（檔案模式，VIDEO_FILE 有設就啟用；線路設定只能有一台相機）：
  VIDEO_FILE                   要測的影片（本機路徑）。取代相機串流；預設照影片的實際速度播 ——
                               站別的「連續 N 秒 / 冷卻」用的是牆鐘，播快了判定就跟線上不一樣
  VIDEO_OUT                    打框結果寫成這個 mp4（跟線上 _marked 同一條燒錄分支，只是終點換成檔案）
  RESULTS_FILE                 每幀結果寫成 JSONL（見 publish.FilePublisher），不送 Redis
  VIDEO_PACE                   1（預設）= 照實際速度播；0 = 純模型測試：能跑多快跑多快、一幀都不丟
                               （probe 的 queue 滿了就等 = 回壓），站別判定照跑但不準（牆鐘），只看框
  播完（EOS）→ 等 worker 把結果寫完 → 正常結束（exit 0）；來源或管線出錯 → exit 1
"""

from __future__ import annotations

import os

# 必須在 numpy / cv2 被 import 之前設好（Dockerfile 也有設，這裡是給本機直接跑時的保險）
for _k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

import itertools
import logging
import sys
import threading
import time
from pathlib import Path

from ds_runner.config import LineConfig, load_line_config
from ds_runner.probe import make_probe
from ds_runner.publish import make_publisher
from ds_runner.roi import ROI_GIE_ID, make_roi_injector, roi_pixels, use_roi_objects
from ds_runner.status import LineReporter, make_stats_provider
from ds_runner.watchdog import CameraStallWatchdog, RunWatchdog, StartupWatchdog, gpu_memory_mib
from ds_runner.worker import TensorJob, Worker, WorkerPool  # noqa: F401

logger = logging.getLogger(__name__)

DEFAULT_TEMPLATE = Path(__file__).resolve().parents[2] / "configs" / "nvinferserver.tmpl.txt"


OUTPUT_LAYERS = {                     # Ultralytics 匯出的輸出層名；依 station.yaml 的 model.task 決定
    "segment": ["output0", "output1"],   # 偵測+mask 係數、prototypes（raw 格式）
    "segment_fused": ["output0", "masks"],   # NMS + mask 合成在 ONNX 裡（edgevision.training.seg_fuse）
    "pose": ["output0"],
    "detect": ["output0"],
}


def seg_output_layers(outputs_meta: dict) -> list[str]:
    """seg 模型依 Triton 實際有的輸出決定要哪兩層：有 masks 就是 fused，否則 raw。

    不靠設定檔多一個 flag —— engine 長什麼樣 Triton 最清楚，問它就好；
    設定跟 engine 不一致的話，check_model_matches 會擋。
    """
    return OUTPUT_LAYERS["segment_fused" if "masks" in outputs_meta else "segment"]


def render_nvinfer_config(template: str, *, model_name: str, triton_grpc: str, interval: int, task: str = "segment",
                          unique_id: int = 1, output_layers: list[str] | None = None, max_batch: int = 32) -> str:
    """一線多站時每個 nvinferserver 各渲染一份：unique_id 不同（probe 靠 NvDsInferTensorMeta.unique_id 分流）。

    output_layers：seg 模型有兩種輸出格式（raw / fused），要拿哪兩層由呼叫端問過 Triton 後給；
    沒給就用 task 的預設（raw）。
    """
    outputs = ",\n".join(f'      {{name: "{n}"}}' for n in (output_layers or OUTPUT_LAYERS[task]))
    out = (template.replace("${MAX_BATCH}", str(max_batch))
           .replace("${MODEL_NAME}", model_name)
                   .replace("${TRITON_GRPC}", triton_grpc)
                   .replace("${INFER_INTERVAL}", str(interval))
                   .replace("${OUTPUT_LAYERS}", outputs)
                   .replace("${UNIQUE_ID}", str(unique_id)))
    leftover = [t for t in ("${MODEL_NAME}", "${TRITON_GRPC}", "${INFER_INTERVAL}", "${OUTPUT_LAYERS}", "${UNIQUE_ID}",
                            "${MAX_BATCH}") if t in out]
    if leftover:
        raise ValueError(f"nvinferserver 樣板仍有未填的變數：{leftover}")
    return out


def resolve_interval(cfg: LineConfig, env: dict[str, str]) -> int:
    """優先序：環境變數覆寫 > line_params 算出來 > 0（每幀都推論；決定 2026-09-04，要跳幀才在線路設定給 target_fps）。

    連續幀去抖數的是「推論幀」，這個值決定 persist_frames=2 實際代表幾百毫秒。
    """
    if env.get("INFER_INTERVAL"):
        return int(env["INFER_INTERVAL"])
    if cfg.infer_interval is not None:
        return cfg.infer_interval
    return 0


def _mk(Gst, pipeline, kind: str, name: str):
    e = Gst.ElementFactory.make(kind, name)
    if e is None:
        raise SystemExit(f"找不到 GStreamer 元件 {kind}（打框推流需要它）")
    pipeline.add(e)
    return e


def _try_set(element, prop: str, value) -> bool:
    """設定可選屬性 —— 不同 DeepStream 版本的編碼器屬性名不一致，設不上就跳過。"""
    try:
        element.set_property(prop, value)
        return True
    except TypeError:
        return False


def make_box_dump_probe(path: str):
    """除錯用：把【真正要畫的框】逐幀寫成 JSONL。掛在 demux 之前、tracker 之後。

    這是唯一無爭議的量測點 —— 不管框是 probe 解出來的、BoxCache 補的、還是 tracker 外推的，
    到這裡都已經是 NvDsObjectMeta，座標就是最終畫上畫面的座標。
    （用錄影逐幀比對不可靠：實測 rtspclientsink 輸出的 pts 是 50/100ms 混雜，
     ffmpeg 取出的幀與編碼器的幀不是一對一。）
    """
    import json

    import gi
    import pyds
    gi.require_version("Gst", "1.0")
    from gi.repository import Gst

    fh = open(path, "w", encoding="utf-8")   # noqa: SIM115 —— 隨程序生命週期
    n = [0]

    def probe(_pad, info, _u):
        buf = info.get_buffer()
        if not buf:
            return Gst.PadProbeReturn.OK
        batch = pyds.gst_buffer_get_nvds_batch_meta(hash(buf))
        l_f = batch.frame_meta_list
        while l_f:
            fm = pyds.NvDsFrameMeta.cast(l_f.data)
            boxes = []
            l_o = fm.obj_meta_list
            while l_o:
                om = pyds.NvDsObjectMeta.cast(l_o.data)
                r = om.rect_params
                oid = int(om.object_id)
                boxes.append([round(r.left, 1), round(r.top, 1), round(r.width, 1), round(r.height, 1),
                              int(om.class_id), -1 if oid >= 2**63 else oid])
                try:
                    l_o = l_o.next
                except StopIteration:
                    break
            fh.write(json.dumps({"i": n[0], "src": int(fm.source_id),
                                 "fnum": int(fm.frame_num), "boxes": boxes}) + "\n")
            n[0] += 1
            try:
                l_f = l_f.next
            except StopIteration:
                break
        if n[0] % 50 == 0:
            fh.flush()
        return Gst.PadProbeReturn.OK

    return probe


DEFAULT_TRACKER_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "tracker_nvsort.yml"
TRACKER_LL_LIB = "/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so"


def build_tracker(Gst, pipeline, env: dict[str, str]):
    """nvtracker：跳幀時把框補回來。

    ⭐ 為什麼 tracker 吃得下我們 probe 掛的 meta：
      我們用 postprocess{other{}}，框是 Python probe 自己解析後掛上去的，不是 nvinfer 內建
      parser 產的。官方對 tracker 輸入的定義只有「a frame batch + the detected object info」，
      並明講它是 "adds/updates the data in the existing NvDsBatchMeta" ——
      讀的是 batch meta，跟 nvinfer 沒有私有通道，也沒有 gie-unique-id 的對應要求。

      唯一條件是 probe 必須在 tracker【上游】。probe 掛在 nvinferserver 的 src pad，
      排成 infer → tracker 後這點天然成立：buffer 流到 tracker 前 meta 已經在了。

    ⚠️ 必須在 nvstreamdemux【之前】—— tracker 要的是 batched NvDsBatchMeta，demux 之後就沒有了。
    """
    tracker = _mk(Gst, pipeline, "nvtracker", "tracker")
    cfg_path = Path(env.get("TRACKER_CONFIG", str(DEFAULT_TRACKER_CONFIG)))
    if not cfg_path.is_file():
        raise SystemExit(f"找不到 tracker 設定檔：{cfg_path}（TRACKER_ENABLED=1 就必須有）")
    tracker.set_property("ll-lib-file", TRACKER_LL_LIB)
    tracker.set_property("ll-config-file", str(cfg_path))
    # tracker 內部運作解析度，須為 32 的倍數。NvSORT 不碰像素，這個值對它影響很小。
    tracker.set_property("tracker-width", int(env.get("TRACKER_WIDTH", 640)))
    tracker.set_property("tracker-height", int(env.get("TRACKER_HEIGHT", 384)))
    _try_set(tracker, "gpu-id", 0)
    _try_set(tracker, "display-tracking-id", 0)   # 畫面上不顯示 ID：標籤已有類別與信心度，再加 ID 太吵
    logger.info("nvtracker 已啟用：%s (%sx%s)", cfg_path.name,
                env.get("TRACKER_WIDTH", 640), env.get("TRACKER_HEIGHT", 384))
    return tracker


ENCODERS = ("auto", "nvenc", "x264")


def encoder_kind(env: dict[str, str]) -> str:
    """ENCODER 設定值：auto（預設）/ nvenc / x264。不認識 → SystemExit。"""
    kind = env.get("ENCODER", "auto")
    if kind not in ENCODERS:
        raise SystemExit(f"ENCODER={kind!r} 不認識，可用：{ENCODERS}")
    return kind


def encoder_spec(env: dict[str, str], w: int, h: int,
                 kind: str | None = None) -> tuple[str, str, list[tuple[str, object]]]:
    """推流編碼器的選擇：(caps 字串, 元件名, 屬性清單)。純函式，可測。

    kind（不給就看 ENCODER）：
      nvenc：nvv4l2h264enc，吃 NVMM 記憶體的 NV12。auto 的第一選擇。
      x264 ：x264enc 軟體編碼，吃【系統記憶體】的 I420 —— 前面的 nvvideoconvert 看到 caps
             沒有 memory:NVMM 就會自己把幀從 GPU 拷出來（官方 deepstream-app enc-type=1 同一招）。
    為什麼要有 x264：GeForce 驅動限制【整台機器】8 個 NVENC session（NVENC Application Note：
    "On non-qualified GPUs ... limited to 8 per system"），一台相機的 _marked 就是一個 session。
    上限看卡（專業卡 RTX PRO / L 系列沒這限制），部署端不知道是哪張卡 → auto 在 runner 裡試：
    先開 NVENC，編碼器報錯就把那一台改 x264 重建（見 build_and_run 的 rebuild_branch）。
    x264 一定要 zerolatency + ultrafast：預設 preset 會攢幀、吃到好幾核，pipeline 直接被拖慢。
    """
    kind = kind or encoder_kind(env)
    if kind == "auto":
        kind = "nvenc"
    bitrate = int(env.get("RESTREAM_BITRATE", 2_000_000))
    if kind == "x264":
        props: list[tuple[str, object]] = [
            ("bitrate", bitrate // 1000),                 # x264enc 的單位是 kbit/s
            ("tune", 0x4),                                # zerolatency
            ("speed-preset", int(env.get("X264_PRESET", 1))),   # 1 = ultrafast
            ("key-int-max", 30), ("bframes", 0), ("byte-stream", True),
            ("threads", int(env.get("X264_THREADS", 2))), # 不設會開滿核心，跟 worker 搶
        ]
        return f"video/x-raw, format=I420, width={w}, height={h}", "x264enc", props
    props = [("bitrate", bitrate), ("iframeinterval", 30), ("insert-sps-pps", True),
             ("idrinterval", 30), ("preset-level", 1), ("num-B-Frames", 0)]
    return f"video/x-raw(memory:NVMM), format=NV12, width={w}, height={h}", "nvv4l2h264enc", props


def branch_index(obj, pipeline) -> tuple[int, str, bool] | None:
    """錯誤訊息的來源往上找，是推流分支元件（r<role>_<N>）就回 (N, role, 還在管線裡嗎)。

    role 例：enc（編碼器）、sink（rtspclientsink，內部元件的錯誤也會歸到它）。
    還在管線裡 = False：分支已經被換掉，這是舊元件拆掉前排進 bus 的錯誤，要忽略 ——
    測試機實測 NVENC 失敗會連發兩則，第二則到的時候舊編碼器已移出管線，當成未知錯誤就整條退出了。
    """
    while obj is not None:
        parent = obj.get_parent()
        name = obj.get_name() or ""
        if (parent is pipeline or parent is None) and name.startswith("r") and "_" in name:
            role, _, n = name[1:].rpartition("_")
            if n.isdigit() and role in BRANCH_ROLES:
                return int(n), role, parent is pipeline
        obj = parent
    return None


BRANCH_ROLES = ("q", "conv1", "caps1", "osd", "conv2", "caps2", "enc", "parse", "sink")


def _disable_sync(_bin, _sub, el) -> None:
    """rtspclientsink 內部元件建立時把 sync 關掉（見 build_restream_branch）。"""
    if el.find_property("sync") is not None:
        el.set_property("sync", False)


def build_restream_branch(Gst, pipeline, demux, idx: int, cam_id: str, env: dict[str, str],
                          go2rtc: str, kind: str = "nvenc", pad=None) -> list:
    """一台相機的打框推流分支：osd → 轉換 → 編碼 → 推回 go2rtc。

    元件順序照官方 SOP（deepstream-test1-rtsp-out）：
        demux → queue(leaky) → nvvideoconvert → caps(RGBA) → nvdsosd
              → nvvideoconvert → caps(NV12) → nvv4l2h264enc → h264parse → rtspclientsink
    kind=x264 時最後三格換成 caps(I420, 系統記憶體) → x264enc（見 encoder_spec）。
    元件名 r<role>_<idx>（role 見 BRANCH_ROLES）：出錯時 branch_index 靠名字找回是哪一台。
    pad：重建分支時沿用 demux 原本的 src_<idx>（不再 request 新的）。回傳整串元件（重建時要拆）。

    三個官方 SOP 標註的坑，都在這裡處理了：
      ① queue 必須 leaky —— 沒人看時不能回壓推論分支（"a slow/absent RTSP client
         never backpressures tee1 and starves the inference branch"）
      ② nvdsosd 需要 RGBA，編碼器需要 NV12 —— 前後各一次 nvvideoconvert + capsfilter
      ③ caps 要寫死 width/height，否則會繼承 mux 尺寸（mux 為了餵模型可能被縮小）
    """
    suffix = env.get("RESTREAM_SUFFIX", "_marked")
    w, h = int(env["_FRAME_W"]), int(env["_FRAME_H"])
    video_out = env.get("VIDEO_OUT") if env.get("VIDEO_FILE") else None
    paced = env.get("VIDEO_PACE", "1") != "0"

    q = _mk(Gst, pipeline, "queue", f"rq_{idx}")
    if video_out:
        # 影片測試：一幀都不能丟（寫檔沒有「沒人看」的問題），而且要回壓 —— 下面的 identity 照時間放行，
        # 整條管線（含推論）就跟著照影片速度走。queue 留短，推論才不會跑在畫面前面太多。
        q.set_property("max-size-buffers", 4)
    else:
        q.set_property("leaky", 2)             # downstream：丟舊的，永不回壓
        q.set_property("max-size-buffers", 2)
    q.set_property("max-size-bytes", 0)
    q.set_property("max-size-time", 0)
    pace = None
    if video_out and paced:                    # VIDEO_PACE=0：不限速，由 x264 / 推論自己的速度決定
        pace = _mk(Gst, pipeline, "identity", f"rpace_{idx}")
        pace.set_property("sync", True)

    conv1 = _mk(Gst, pipeline, "nvvideoconvert", f"rconv1_{idx}")
    caps1 = _mk(Gst, pipeline, "capsfilter", f"rcaps1_{idx}")
    caps1.set_property("caps", Gst.Caps.from_string(
        f"video/x-raw(memory:NVMM), format=RGBA, width={w}, height={h}"))

    osd = _mk(Gst, pipeline, "nvdsosd", f"rosd_{idx}")
    # ⚠️ display-bbox 必須明確開啟 —— 官方 property 表的示例值是 0，
    # 不開的話 nvdsosd 會【完全忽略】meta 裡的框：probe 照跑、不拋例外、像素沒變。
    _try_set(osd, "display-bbox", 1)
    _try_set(osd, "display-text", 1)
    _try_set(osd, "process-mode", 1)           # 1=GPU
    # mask 也交給 nvdsosd 在 GPU 畫：probe 把 prototype 解析度的小塊塞進 obj_meta.mask_params，
    # 拉伸 / 二值化 / 上色都在這裡做（實測 DS 7.1 可行，見 probe._attach_mask）。
    if env.get("DRAW_MASKS", "1") == "1":
        _try_set(osd, "display-mask", 1)

    caps_str, enc_kind, enc_props = encoder_spec(env, w, h, kind)
    conv2 = _mk(Gst, pipeline, "nvvideoconvert", f"rconv2_{idx}")
    caps2 = _mk(Gst, pipeline, "capsfilter", f"rcaps2_{idx}")
    caps2.set_property("caps", Gst.Caps.from_string(caps_str))

    enc = _mk(Gst, pipeline, enc_kind, f"renc_{idx}")
    for prop, val in enc_props:
        if not _try_set(enc, prop, val) and prop == "bitrate":
            raise SystemExit(f"[{cam_id}] {enc_kind} 設不了 bitrate")
    if enc_kind != "nvv4l2h264enc":
        logger.warning("[%s] 推流走 %s 軟體編碼（CPU）", cam_id, enc_kind)

    parse = _mk(Gst, pipeline, "h264parse", f"rparse_{idx}")
    if video_out:
        mux = _mk(Gst, pipeline, "mp4mux", f"rmp4_{idx}")
        _try_set(mux, "faststart", True)       # moov 放前面：瀏覽器拿預簽網址邊下邊播
        sink = _mk(Gst, pipeline, "filesink", f"rsink_{idx}")
        sink.set_property("location", video_out)
        sink.set_property("sync", False)       # 照時間放行交給 identity（不限速時根本不等）
        pad = pad or demux.request_pad_simple(f"src_{idx}")
        if pad is None or pad.link(q.get_static_pad("sink")) != Gst.PadLinkReturn.OK:
            raise SystemExit(f"[{cam_id}] demux 接 queue 失敗")
        # pace 只在照實際速度播時才有（VIDEO_PACE=0 不限速 → None，不串）
        chain = [e for e in (q, pace, conv1, caps1, osd, conv2, caps2, enc, parse, mux, sink) if e is not None]
        for a, b in itertools.pairwise(chain):
            if not a.link(b):
                raise SystemExit(f"[{cam_id}] 串接失敗：{a.get_name()} → {b.get_name()}")
        return chain
    sink = _mk(Gst, pipeline, "rtspclientsink", f"rsink_{idx}")
    sink.set_property("location", f"{go2rtc}/{cam_id}{suffix}")
    sink.set_property("protocols", 4)          # TCP：UDP 丟包會讓畫面破圖
    sink.set_property("latency", 0)
    # ⚠️ 推流不照時間戳等待（RESTREAM_SYNC=1 可恢復）。rtspclientsink 自己沒有 sync 屬性，
    #    它內部動態建立的送出元件預設會「等到該播的時間才送」。多台相機各自的時間戳基準不同，
    #    偏在未來的那幾台每一幀都在等，前面的 leaky queue 只好一直丟 —— 2026-10-02 測試叢集
    #    8 台同一條線，有的只剩每秒 0~3 幀、有的完整 15 幀（終點不同步時全部恢復）。
    #    推給 go2rtc 不需要照時間送：時間戳跟著 RTP 走，播放端自己會依時間播。
    if env.get("RESTREAM_SYNC", "0") != "1":
        sink.connect("deep-element-added", _disable_sync)

    pad = pad or demux.request_pad_simple(f"src_{idx}")
    if pad is None:
        raise SystemExit(f"無法取得 demux src_{idx}")
    if pad.link(q.get_static_pad("sink")) != Gst.PadLinkReturn.OK:
        raise SystemExit(f"[{cam_id}] demux 接 queue 失敗")
    chain = [q, conv1, caps1, osd, conv2, caps2, enc, parse, sink]
    for a, b in itertools.pairwise(chain):
        if not a.link(b):
            raise SystemExit(f"[{cam_id}] 串接失敗：{a.get_name()} → {b.get_name()}")
    return chain


def build_and_run(cfg: LineConfig, worker: Worker, env: dict[str, str], reporter: LineReporter | None = None) -> None:
    """worker 參數只拿 publish / model_version 當樣板；實際跑的是 WorkerPool 分片出來的 N 條。"""
    import gi
    gi.require_version("Gst", "1.0")
    from gi.repository import GLib, Gst

    go2rtc = env.get("GO2RTC_RTSP", "rtsp://localhost:8554")
    source = env.get("GO2RTC_SOURCE_RTSP") or go2rtc
    video_file = env.get("VIDEO_FILE") or None
    # 純模型測試（VIDEO_PACE=0）：不照影片時間播，probe 也不准丟幀（見檔頭）
    unpaced = bool(video_file) and env.get("VIDEO_PACE", "1") == "0"
    if video_file and len(cfg.active_cameras) != 1:
        raise SystemExit(f"影片測試只能有一台相機（線路設定有 {cfg.camera_ids}）")
    width, height = cfg.frame_size
    Gst.init(None)                      # 先於任何 thread：GLib 的 TLS 要在主執行緒初始化
    stop = threading.Event()
    stats: dict = {}
    # worker 分片：每台相機固定一條 thread（站別的跨幀狀態不能被兩條 thread 碰）。
    # WORKER_THREADS 預設 min(4, 相機數)；QUEUE_SIZE 是每個 shard 的深度，預設 20。
    pool = WorkerPool(cfg, publish=worker.publish, model_version=worker.model_version,
                      n_workers=int(env.get("WORKER_THREADS", 0)) or None,
                      queue_size=int(env.get("QUEUE_SIZE", 20)), stats=stats)
    pool.start(stop)
    q = pool.router                     # probe 只認 put_nowait，分片對它透明
    if reporter is not None:
        # 心跳附上推論負載（fps / dropped / sync_ms / queue 深度），給健康巡檢看
        reporter.stats_provider = make_stats_provider(stats, pool.queues, cameras=cfg.camera_ids)

    # nvinferserver 設定：樣板 + 模型名 / Triton 位址 / 推論間隔。
    # 一線多站 → 每個站別一個 nvinferserver、一份設定（unique_id 1, 2, …）。
    # 主站的模型名沿用 MODEL_NAME 環境變數（部署時可覆寫），附加站用 station.yaml / line config 的。
    template = Path(env.get("NVINFER_TEMPLATE", str(DEFAULT_TEMPLATE))).read_text(encoding="utf-8")
    primary_interval = resolve_interval(cfg, env)
    cams = cfg.active_cameras
    # 主站 ROI 推論（假物件模式，見 roi.py）：任一相機有 infer_roi 就啟用。來源 id = mux 的 sink_<idx>。
    roi_by_source = {i: roi_pixels(c.infer_roi, width, height) for i, c in enumerate(cams) if c.infer_roi}
    use_roi = bool(roi_by_source) and env.get("INFER_ROI_ENABLED", "1") == "1"
    if not use_roi:
        roi_by_source = {}
        pool.disable_roi()             # worker 的座標還原也要回到全幅，否則框會被平移
    roi_interval = 0
    nvinfer_paths: list[Path] = []
    for b in cfg.bindings():
        path = Path(env.get("NVINFER_CONFIG", "/tmp/nvinferserver.txt")) if b.primary \
            else Path(f"/tmp/nvinferserver_{b.gie_id}.txt")
        mname = env["MODEL_NAME"] if b.primary else b.model.name
        # seg 模型：engine 是 raw 還是 fused 只有 Triton 知道。問一次，決定要抓 output1 還是 masks。
        # 問不到（TRITON_HTTP 沒設 / 連不上）就退回 raw —— 跟先前行為一致；真的不對 verify 會擋。
        layers = None
        if b.task == "segment":
            meta = triton_model_metadata(env.get("TRITON_HTTP"), mname)
            if meta is not None:
                layers = seg_output_layers({o.get("name"): o for o in meta.get("outputs", [])})
                logger.info("%s 輸出層：%s", mname, layers)
        interval = primary_interval if b.interval is None else b.interval
        text = render_nvinfer_config(
            template,
            model_name=mname,
            triton_grpc=env["TRITON_GRPC"],
            interval=interval,
            task=b.task, unique_id=b.gie_id, output_layers=layers,
            max_batch=nvinfer_batch(triton_max_batch(env.get("TRITON_HTTP"), mname), len(cams)),
        )
        if b.primary and use_roi:
            text = use_roi_objects(text)           # 主站只對 injector 放的 ROI 假物件裁切推論
            roi_interval = interval                # 跳幀改由 injector 決定
        path.write_text(text, encoding="utf-8")
        nvinfer_paths.append(path)
    if use_roi:
        logger.info("主站 ROI 推論：%d/%d 台相機裁 ROI（假物件模式）", len(roi_by_source), len(cams))

    pipeline = Gst.Pipeline()
    mux = Gst.ElementFactory.make("nvstreammux", "mux")
    mux.set_property("batch-size", len(cams))      # batch = 相機數，多路匯流成一批
    mux.set_property("width", width)
    mux.set_property("height", height)
    mux.set_property("batched-push-timeout", 40000)
    mux.set_property("live-source", 0 if video_file else 1)

    def _count_batch(_pad, _info):                 # 看門狗用：影像有沒有進到推論前（分辨「推論卡住」跟「沒影像」）
        stats["mux_batches"] = stats.get("mux_batches", 0) + 1
        return Gst.PadProbeReturn.OK
    mux.get_static_pad("src").add_probe(Gst.PadProbeType.BUFFER, _count_batch)
    infers = []
    for i, path in enumerate(nvinfer_paths):
        el = Gst.ElementFactory.make("nvinferserver", "infer" if i == 0 else f"infer{i + 1}")
        el.set_property("config-file-path", str(path))
        infers.append(el)
    infer = infers[-1]                  # 下游（tracker / demux / probe）接在最後一個之後：所有 tensor meta 都到齊了
    restream = env.get("RESTREAM_ENABLED", "1") == "1"
    pipeline.add(mux)
    for el in infers:
        pipeline.add(el)
    if restream:
        # demux 拆回各相機：src_<idx> 的 idx 與 mux 的 sink_<idx>、probe 的 source_id 一致
        out_el = Gst.ElementFactory.make("nvstreamdemux", "demux")
        pipeline.add(out_el)
    else:
        out_el = Gst.ElementFactory.make("fakesink", "sink")
        out_el.set_property("sync", 1 if video_file and not unpaced else 0)   # 檔案模式：照影片時間播（見檔頭）
        pipeline.add(out_el)

    src_to_cam = {}

    def add_source(idx: int, cam_id: str):
        """建一台相機的 nvurisrcbin 並接到 mux 的 sink_<idx>。重建單一來源時也走這裡。"""
        src = Gst.ElementFactory.make("nvurisrcbin", f"src_{idx}")
        if video_file:
            src.set_property("uri", Path(video_file).resolve().as_uri())
        else:
            src.set_property("uri", f"{source}/{cam_id}")
            src.set_property("rtsp-reconnect-interval", 10)
        pipeline.add(src)

        def on_pad(_bin, pad, sink_idx=idx):
            if not pad.get_name().startswith("vsrc"):
                return
            sp = mux.get_static_pad(f"sink_{sink_idx}") or mux.request_pad_simple(f"sink_{sink_idx}")
            if sp is None:
                logger.error("[src_%d] 拿不到 mux sink_%d", sink_idx, sink_idx)
                return
            if sp.is_linked():                     # 舊來源殘留的連結（重建時）
                peer = sp.get_peer()
                if peer is not None:
                    peer.unlink(sp)
            if pad.link(sp) != Gst.PadLinkReturn.OK:
                logger.error("[src_%d] 連不上 mux sink_%d", sink_idx, sink_idx)

        src.connect("pad-added", on_pad)
        return src

    for idx, cam in enumerate(cams):                  # sink_<idx> 的 idx 就是 source_id，probe 靠它分流
        src_to_cam[idx] = cam.id
        add_source(idx, cam.id)
    mux.link(infers[0])
    if use_roi:
        # 每台相機一個假物件：有 ROI 的框 ROI，沒有的框整幅（同一條線可以混）
        rects = {i: roi_by_source.get(i, (0, 0, width, height)) for i in range(len(cams))}
        infers[0].get_static_pad("sink").add_probe(
            Gst.PadProbeType.BUFFER, make_roi_injector(rects, interval=roi_interval, stats=stats), None)
    for a, b in itertools.pairwise(infers):   # 主站 → 附加站 …，每個都在全幅上推論（operate_on_gie_id: -1）
        if not a.link(b):
            raise SystemExit(f"nvinferserver 串接失敗：{a.get_name()} → {b.get_name()}")
    # infer → [tracker] → demux/fakesink
    # tracker 一定在 demux 之前：它要的是 batched meta。probe 掛在 infer 的 src pad，
    # 也就是 tracker 上游 —— 我們的框在 tracker 收到 buffer 前就已經在 batch meta 裡了。
    tracker_on = env.get("TRACKER_ENABLED", "0") == "1"
    tracker = None
    if tracker_on:
        tracker = build_tracker(Gst, pipeline, env)
        if not infer.link(tracker) or not tracker.link(out_el):
            raise SystemExit("nvtracker 串接失敗（infer → tracker → out）")
    else:
        infer.link(out_el)
    # 每台相機的推流分支：{idx: (編碼器種類, 元件串)}。auto 先開 NVENC，滿了由 on_msg 改 x264。
    branches: dict[int, tuple[str, list]] = {}
    enc_mode = encoder_kind(env)
    has_x264 = Gst.ElementFactory.find("x264enc") is not None
    if enc_mode == "auto" and not has_x264:
        logger.warning("映像沒有 x264enc：ENCODER=auto 只能用 NVENC，NVENC 滿了那台就沒有 _marked 輸出")
    if restream:
        env["_FRAME_W"], env["_FRAME_H"] = str(width), str(height)
        first = "x264" if enc_mode == "x264" else "nvenc"
        for idx, cam in enumerate(cams):
            branches[idx] = (first, build_restream_branch(Gst, pipeline, out_el, idx, cam.id, env, go2rtc, first))

    # ⭐ probe 掛在 infer 的 src pad —— 也就是 demux【之前】。
    # meta 必須在 buffer 流到 nvdsosd 前掛好，而且 demux 之後每個 buffer 已非 batched，
    # 在那裡加 meta 不會被渲染（實測踩過）。
    bbox_only = worker.bbox_only_postproc if restream else None
    from types import SimpleNamespace
    probe_bindings = {
        rt.binding.gie_id: SimpleNamespace(key=None if rt.binding.primary else rt.binding.key,
                                           postproc=rt.postproc if restream else None,
                                           class_names=rt.class_names, primary=rt.binding.primary,
                                           hidden=frozenset(rt.binding.model.hidden_classes))
        for rt in pool.runtimes().values()
    }
    infer.get_static_pad("src").add_probe(
        Gst.PadProbeType.BUFFER,
        make_probe(q, src_to_cam, width, height, stats,
                   bbox_only=bbox_only, class_names=worker.class_names, draw=restream,
                   draw_masks=restream and env.get("DRAW_MASKS", "1") == "1",
                   bindings=probe_bindings, roi_by_source=roi_by_source or None,
                   roi_gie_id=ROI_GIE_ID if use_roi else None,
                   # tracker 開著時，跳幀的框由 tracker 用 Kalman 外推補上（框會【移動】）。
                   # 此時 probe 不可以再把上一次的框原地重畫 —— 那會變成兩份框疊在一起。
                   fill_skipped=not tracker_on,
                   # 影片測試：每幀帶影片 PTS（JSONL 的 t 用它，跑快也對得上影片秒數）；
                   # 不限速時 queue 滿了要等，不能丟幀
                   with_pts=bool(video_file), block_on_full=unpaced),
        None,
    )

    # 除錯：把最終要畫的框 dump 成 JSONL（BOX_DUMP=<path>）。
    # ⚠️ 一定要在上面那個 probe【之後】才 add_probe —— 同一個 pad 上的 probe 依加入順序執行，
    #    先掛的會在畫框之前跑，dump 出來全是空的（實測踩過）。
    # tracker 開著時掛在 tracker 的 src pad，才看得到 Kalman 外推補的框。
    if env.get("BOX_DUMP"):
        (tracker or infer).get_static_pad("src").add_probe(
            Gst.PadProbeType.BUFFER, make_box_dump_probe(env["BOX_DUMP"]), None)
        logger.info("框 dump 已啟用 → %s", env["BOX_DUMP"])

    loop = GLib.MainLoop()

    # 定期印 probe 統計：跳幀時「框是誰畫的」只能從這裡看出來。
    #   drawn        有推論的幀畫的框數
    #   drawn_cached BoxCache 補的框數（tracker 開著時應為 0 —— 補框是 tracker 的事）
    #   dropped      queue 滿而丟棄的張量（worker 跟不上）
    stats_every = int(env.get("STATS_INTERVAL_S", 30))
    if stats_every > 0:
        def _dump_stats():
            if stats:
                logger.info("probe 統計 %s", {k: v for k, v in sorted(stats.items()) if not k.startswith("_")})
            return True
        GLib.timeout_add_seconds(stats_every, _dump_stats)

    bus = pipeline.get_bus()
    bus.add_signal_watch()

    # ── 單一相機出錯只重建那一台，不拖垮整條線 ─────────────────────
    # ⚠️ 為什麼：原本收到任何 ERROR 就 loop.quit()。只要 8 台裡任何一台 RTSP 重連時
    #    nvurisrcbin 接不上（cb_newpad_rtspsrc: Failed to populate and link video elements），
    #    整條 8 台一起退出、靠 restart 拉回 —— 測試機實測 4.5 天重啟 24 次，每次全線瞎 15~60 秒。
    #    line_c 在 9/4 就是死在同一個錯誤（當時沒有 restart，死了 20 天）。
    #    做法照 NVIDIA deepstream_rt_src_add_del：拆掉那一台（放掉 mux 接口、移出 pipeline）、
    #    延遲後重建。錯誤不是來自來源（推論 / 編碼…）的話維持原行為，整條退出。
    resetting: set[int] = set()
    reset_backoff: dict[int, tuple[float, float]] = {}   # idx → (上次出錯時間, 上次延遲)

    def source_index(obj) -> int | None:
        """錯誤訊息的來源往上找，是 pipeline 直屬的 src_<N> 就回 N。"""
        while obj is not None:
            parent = obj.get_parent()
            name = obj.get_name() or ""
            if parent is pipeline and name.startswith("src_") and name[4:].isdigit():
                return int(name[4:])
            obj = parent
        return None

    def rebuild_source(idx: int):
        old = pipeline.get_by_name(f"src_{idx}")
        if old is not None:
            old.set_state(Gst.State.NULL)
            sp = mux.get_static_pad(f"sink_{idx}")
            if sp is not None:
                sp.send_event(Gst.Event.new_flush_stop(False))
                mux.release_request_pad(sp)
            pipeline.remove(old)
        src = add_source(idx, src_to_cam[idx])
        src.sync_state_with_parent()
        resetting.discard(idx)
        stats["src_reset"] = stats.get("src_reset", 0) + 1
        logger.warning("[src_%d] %s 已重建（累計重建 %d 次）", idx, src_to_cam[idx], stats["src_reset"])
        return False                                   # GLib 只跑一次

    def schedule_rebuild(idx: int, reason: str, *, count_err: bool = True):
        if idx in resetting:
            return                                     # 同一台已經在排隊重建，別疊
        resetting.add(idx)
        # 退避：連續出錯就拉長間隔（相機真的死掉時不要狂刷），上限 60 秒；成功跑 5 分鐘後歸零
        now = time.time()
        last, delay = reset_backoff.get(idx, (0.0, 0.0))
        delay = 2.0 if now - last > 300 else min(60.0, max(2.0, delay * 2))
        reset_backoff[idx] = (now, delay)
        # ⚠️ 無幀重建不算 src_err：RunWatchdog 看到 src_err 增加會當成相機問題繼續等，
        #    單台線路就失去「救不回來時重啟整個程序」這條最後防線。
        key = "src_err" if count_err else "src_stall"
        stats[key] = stats.get(key, 0) + 1
        logger.error("[src_%d] %s 串流出錯，只重建這一台（%.0f 秒後）：%s", idx, src_to_cam[idx], delay, reason)
        GLib.timeout_add(int(delay * 1000), rebuild_source, idx)

    # ── 推流分支出錯：只重建那一台的輸出，推論不受影響 ───────────────────
    # auto：NVENC 開不起來（GeForce 整台 8 路用完，或其他原因）→ 那一台改 x264。
    # 其他分支錯誤（例 go2rtc 重啟、rtspclientsink 斷線）→ 同一種編碼器重建，跟來源一樣退避。
    # 換分支照 GStreamer 動態改管線的做法：在 demux 的 src_<idx> 掛 BLOCK probe，
    # 擋住後才拆舊接新，避免 demux 正往被拆的元件推資料。
    branch_resetting: set[int] = set()
    branch_backoff: dict[int, tuple[float, float]] = {}

    def swap_branch(idx: int, kind: str):
        pad = out_el.get_static_pad(f"src_{idx}")
        _old_kind, old = branches[idx]

        def do_swap():
            peer = pad.get_peer()
            if peer is not None:
                pad.unlink(peer)
            for el in old:
                el.set_state(Gst.State.NULL)
                pipeline.remove(el)
            chain = build_restream_branch(Gst, pipeline, out_el, idx, src_to_cam[idx], env, go2rtc, kind, pad=pad)
            for el in chain:
                el.sync_state_with_parent()
            branches[idx] = (kind, chain)
            branch_resetting.discard(idx)
            stats["branch_reset"] = stats.get("branch_reset", 0) + 1
            stats["x264_branches"] = sum(1 for k, _ in branches.values() if k == "x264")
            logger.warning("[%s] 推流分支已重建：%s（x264 共 %d 台）", src_to_cam[idx], kind, stats["x264_branches"])
            pad.remove_probe(probe_id[0])
            return False

        probe_id = [0]

        def on_blocked(_pad, _info):
            GLib.idle_add(do_swap)
            return Gst.PadProbeReturn.OK          # 維持擋住，do_swap 換好再移除

        probe_id[0] = pad.add_probe(Gst.PadProbeType.BLOCK_DOWNSTREAM, on_blocked)
        return False

    def schedule_branch_rebuild(idx: int, role: str, reason: str):
        if idx in branch_resetting:
            return
        branch_resetting.add(idx)
        kind, _ = branches[idx]
        if role == "enc" and kind == "nvenc" and enc_mode == "auto" and has_x264:
            logger.warning("[%s] NVENC 開不起來，這台改 x264 軟體編碼：%s", src_to_cam[idx], reason)
            GLib.idle_add(swap_branch, idx, "x264")
            return
        now = time.time()
        last, delay = branch_backoff.get(idx, (0.0, 0.0))
        delay = 2.0 if now - last > 300 else min(60.0, max(2.0, delay * 2))
        branch_backoff[idx] = (now, delay)
        logger.error("[%s] 推流分支出錯（%s），%.0f 秒後重建：%s", src_to_cam[idx], role, delay, reason)
        GLib.timeout_add(int(delay * 1000), swap_branch, idx, kind)

    outcome = {"eos": False}

    def on_msg(_b, m):
        if m.type == Gst.MessageType.ERROR:
            err, dbg = m.parse_error()
            if video_file:                     # 影片測試：沒有「重連」這回事，出錯就結束
                logger.error("影片測試管線出錯：%s | %s", err, dbg)
                loop.quit()
                return
            idx = source_index(m.src)
            if idx is not None and idx in src_to_cam:
                schedule_rebuild(idx, f"{err} | {dbg}")
                return
            hit = branch_index(m.src, pipeline) if branches else None
            if hit is not None and hit[0] in branches:
                if not hit[2]:
                    logger.info("[%s] 已換掉的推流分支送來的舊錯誤，忽略：%s", src_to_cam[hit[0]], err)
                    return
                schedule_branch_rebuild(hit[0], hit[1], f"{err} | {dbg}")
                return
            logger.error("GStreamer ERROR: %s | %s", err, dbg)
            loop.quit()
        elif m.type == Gst.MessageType.EOS:
            outcome["eos"] = True
            loop.quit()

    bus.connect("message", on_msg)
    logger.info(">>> 線路 %s 站別 %s 相機 %s %dx%d config_version=%s interval=%s tracker=%s",
                cfg.id, cfg.station, cfg.camera_ids, width, height, cfg.config_version,
                resolve_interval(cfg, env), "on" if tracker_on else "off")
    # 心跳與看門狗都要在 PLAYING 之前起：GPU 不夠時 set_state 本身就可能卡住，之後的程式碼跑不到
    if reporter:
        reporter.gpu = gpu_memory_mib
        reporter.start(cameras_running=len(cams), state="applying")   # 版本已載入；第一幀過推論才轉 ok
    watchdog = StartupWatchdog(stats, reporter, float(env.get("STARTUP_TIMEOUT_S", 120)))
    watchdog.start()
    run_watchdog = RunWatchdog(stats, reporter, float(env.get("RUN_STALL_S", 180)))
    run_watchdog.start()
    # 影片檔沒有斷線重連這回事；重建要動 pipeline，所以排在 GLib 主迴圈上（主迴圈卡住歸 RunWatchdog 管）
    cam_stall_s = 0.0 if video_file else float(env.get("CAM_STALL_S", 90))
    if cam_stall_s > 0:
        cam_to_src = {c: i for i, c in src_to_cam.items()}
        cam_watchdog = CameraStallWatchdog(stats, list(cam_to_src), cam_stall_s)

        def _check_camera_stall():
            for cam, idle in cam_watchdog.check():
                schedule_rebuild(cam_to_src[cam], f"{idle:.0f} 秒沒有新畫面（相機來源沒報錯，重連卡住）", count_err=False)
            return True

        GLib.timeout_add_seconds(5, _check_camera_stall)
    pipeline.set_state(Gst.State.PLAYING)
    try:
        loop.run()
        if video_file and outcome["eos"]:
            drain_queues(pool.queues)          # 最後幾幀還在 worker 手上，等它們判定完、寫進結果
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        watchdog.stop()
        run_watchdog.stop()
        if reporter:
            reporter.stop()
        pipeline.set_state(Gst.State.NULL)
        logger.info("dropped frames: %s", stats.get("dropped", 0))
        close = getattr(worker.publish, "close", None)
        if close:
            close()
    return outcome["eos"]


def drain_queues(queues, *, timeout_s: float = 30.0, settle_s: float = 1.0) -> None:
    """等 worker 把 queue 清空（再多等 settle_s 讓正在處理的那一幀寫完）。"""
    deadline = time.time() + timeout_s
    while time.time() < deadline and any(not q.empty() for q in queues):
        time.sleep(0.1)
    time.sleep(settle_s)


def triton_model_metadata(triton_http: str | None, model_name: str) -> dict | None:
    """GET /v2/models/{name}：inputs / outputs 的 shape、versions。連不上回 None。"""
    if not triton_http:
        return None
    import json
    from urllib.request import urlopen
    try:
        with urlopen(f"http://{triton_http}/v2/models/{model_name}", timeout=3) as r:
            return json.loads(r.read().decode())
    except Exception:
        logger.warning("查不到 Triton 模型 metadata（%s）", model_name, exc_info=True)
        return None


def triton_max_batch(triton_http: str | None, model_name: str, default: int = 32) -> int:
    """GET /v2/models/{name}/config 的 max_batch_size —— nvinferserver 的 max_batch_size 不能超過它。

    ⚠️ 為什麼要問：樣板曾寫死 32。IPCAM 的 engine 為了塞進 16 GB 卡改成 max 4 之後，
       nvinferserver 啟動就死（failed to specify input dims triton backend），而且錯誤訊息完全
       沒提 batch。engine 的上限由匯出決定，Triton 最清楚，問它。
    問不到回 default（跟先前寫死的行為一致）；Triton 回 0（不 batch）當 1。
    """
    if not triton_http:
        return default
    import json
    from urllib.request import urlopen
    try:
        with urlopen(f"http://{triton_http}/v2/models/{model_name}/config", timeout=3) as r:
            mb = int(json.loads(r.read().decode()).get("max_batch_size", default))
        return max(1, mb)
    except Exception:
        logger.warning("查不到 Triton 模型 config（%s），nvinferserver max_batch 用 %d", model_name, default, exc_info=True)
        return default


def nvinfer_batch(triton_max: int, n_cams: int) -> int:
    """nvinferserver 的 max_batch_size = min(engine 上限, 相機數)。

    ⚠️ 2026-10-15 正式叢集 GPU 切片被吃光的主因：以前直接用 engine 上限。nvinferserver 依它預先配置
       輸入張量與 output_buffer_pool_size（64）份輸出緩衝 —— ppe 是 1280×1280、max batch 32，
       一條只有 1 台相機的線就白白佔掉約 3 GB，掛了 12 條線就把 48 GB 的 MIG 切片塞滿，
       第 13 個程序連 CUDA context 都建不起來，管線卡住、不報錯。
       streammux 一批最多就是相機數（ROI 模式也是每台一個假物件），多配的永遠用不到。
    """
    return max(1, min(triton_max, n_cams))


def check_model_version(meta: dict, *, model_name: str, version: int | None) -> list[str]:
    """Triton 上載入的版本要包含 Registry 現役版。沒附版本（舊設定 / 測試）不檢查。"""
    if version is None:
        return []
    loaded = {str(v) for v in meta.get("versions") or []}
    if loaded and str(version) not in loaded:
        return [(f"{model_name}: 現役是 v{version}，但這個 Triton 載入的是 {sorted(loaded)}"
                 "（多節點時：這個站點還沒跟上 promote，見 site_deploy）")]
    return []


def triton_model_version(triton_http: str | None, model_name: str) -> str:
    """問 Triton 目前服務的版本。查不到回 unknown，不阻擋啟動。"""
    meta = triton_model_metadata(triton_http, model_name)
    versions = (meta or {}).get("versions") or []
    return str(max(int(v) for v in versions)) if versions else "unknown"


def check_model_matches(meta: dict, *, model_name: str, imgsz: int, n_classes: int, task: str) -> list[str]:
    """比對 Triton 上的 engine 跟 line config 的模型參數。回問題清單（空 = 一致）。

    ⚠️ 為什麼要有這道閘門：engine 是固定尺寸 / 固定類別數的東西。設定寫 imgsz=1024 但 engine 是 640
    建的、或類別數對不上，runner 照樣會跑、照樣有框、只是全部是錯的 —— 沒有任何錯誤訊息。
    寧可啟動就死（MODEL_CHECK=0 可關）。

    輸入 images: [-1, 3, H, W]（H = W = imgsz）
    output0 第二維：detect = 4+nc；segment = 4+nc+32；pose = 4+nc+kpt*3（只檢查 ≥ 4+nc）
    """
    problems: list[str] = []
    inputs = {i.get("name"): i for i in meta.get("inputs", [])}
    outputs = {o.get("name"): o for o in meta.get("outputs", [])}
    img = inputs.get("images")
    if img is None:
        problems.append(f"{model_name}: 沒有名為 images 的輸入（有：{sorted(inputs)}）")
    else:
        shape = [int(x) for x in img.get("shape", [])]
        hw = shape[-2:] if len(shape) >= 2 else []
        if hw and any(v not in (-1, imgsz) for v in hw):
            problems.append(f"{model_name}: engine 輸入 {shape} 不是 imgsz={imgsz}（line config / station 寫的）")
    out0 = outputs.get("output0")
    fused = task == "segment" and "masks" in outputs
    if out0 is None:
        problems.append(f"{model_name}: 沒有 output0（有：{sorted(outputs)}）")
    else:
        shape = [int(x) for x in out0.get("shape", [])]
        feats = [v for v in shape[1:] if v > 0]
        if fused:
            # fused：(B, max_det, features)，features 是【最後】一維，max_det 可能比它小
            f = shape[-1] if shape else None
        else:
            f = min(feats) if feats else None      # (B, features, anchors)：features 是較小的那個正數
        expect = {"detect": 4 + n_classes, "segment": 4 + n_classes + 32}.get(task)
        if f is not None and expect is not None and f != expect:
            problems.append(f"{model_name}: output0 features={f}，但 {task} + {n_classes} 類應為 {expect}（類別數對不上）")
        if f is not None and task == "pose" and f < 4 + n_classes:
            problems.append(f"{model_name}: output0 features={f} < 4+{n_classes}（類別數對不上）")
    if task == "segment" and not fused and "output1" not in outputs:
        problems.append(f"{model_name}: segment 模型缺 output1（prototypes）；"
                        f"若是 fused 匯出則應有 masks（有：{sorted(outputs)}）")
    if fused:
        m = [int(x) for x in outputs["masks"].get("shape", [])]
        if len(m) >= 3 and m[-1] > 0 and m[-1] != imgsz // 4:
            problems.append(f"{model_name}: masks 尺寸 {m[-2:]} ≠ imgsz/4={imgsz // 4}")
    return problems


def verify_models_against_triton(cfg: LineConfig, env: dict[str, str]) -> None:
    """啟動閘門：每個 binding 的模型都去 Triton 對一次。對不上 → SystemExit。MODEL_CHECK=0 跳過。"""
    if env.get("MODEL_CHECK", "1") != "1":
        return
    triton_http = env.get("TRITON_HTTP")
    if not triton_http:
        logger.warning("沒設 TRITON_HTTP，跳過模型參數比對")
        return
    problems: list[str] = []
    for b in cfg.bindings():
        name = env["MODEL_NAME"] if b.primary else b.model.name
        meta = triton_model_metadata(triton_http, name)
        if meta is None:
            problems.append(f"{name}: Triton 查不到（沒 load？名字錯？）")
            continue
        problems += check_model_matches(meta, model_name=name, imgsz=b.model.imgsz,
                                        n_classes=len(b.model.classes), task=b.task)
        # 版本也要對：設定 API 附的是 Registry 現役版。多個推論站點時，別的節點的 Triton 可能還沒跟上
        # promote（那邊要現場建 engine）—— 用舊版 engine 配新版類別，框的名字會全錯，寧可不啟動
        problems += check_model_version(meta, model_name=name, version=b.model.version)
        # streammux 一批 = 相機數；超過 engine 的 max batch 時 nvinferserver 啟動就死，錯誤訊息還不提 batch
        n_cams = len(cfg.active_cameras)
        mb = triton_max_batch(triton_http, name, default=n_cams)
        if n_cams > mb:
            problems.append(f"{name}: 這條線 {n_cams} 台相機，但 engine max batch 只有 {mb}（重新匯出 --max-batch）")
    if problems:
        raise SystemExit("模型檢查沒過（沒載入或參數不符），拒絕啟動：\n  " + "\n  ".join(problems))
    logger.info("模型參數比對通過：%s", [(env["MODEL_NAME"] if b.primary else b.model.name, b.model.imgsz, len(b.model.classes)) for b in cfg.bindings()])


def wait_for_line_config(env: dict[str, str], base: str | None, first_error: Exception, *,
                         retry_s: float, heartbeat_s: float = 15.0, sleep=time.sleep,
                         clock=time.monotonic) -> LineConfig:
    """設定載入失敗（例：相機沒畫 ROI）→ 不套用、原地待命，不結束程序。

    以前直接 sys.exit(2)：K8s 把 Pod 拉起來 → 又載入失敗 → 又結束，一天重啟兩百多次（CrashLoopBackOff）。
    設定沒填完不是故障，不該讓它重啟。待命期間：
      - 每 heartbeat_s 秒回報一次 failed（心跳不能斷，否則控制端只會看到 stale，看不到原因）
      - 每 retry_s 秒重讀一次設定；讀得到就照常啟動（設定 API 一時連不上也靠這個恢復）
    使用者存了新設定 → 設定版本變了 → 調和器換掉 Deployment，新 Pod 一啟動就讀到新設定，不必等重讀。
    """
    reporter = LineReporter(base, env.get("LINE_ID", "?"), 0)   # 版本號未知填 0
    err = f"{type(first_error).__name__}: {first_error}"
    logger.error("線路設定載入失敗，不套用、待命（每 %.0f 秒重讀一次設定）：%s", retry_s, err)
    reporter.report_failure(err)
    next_retry = clock() + retry_s
    while True:
        sleep(heartbeat_s)
        if clock() < next_retry:
            reporter.report_failure(err)
            continue
        next_retry = clock() + retry_s
        try:
            cfg = load_line_config(env)
        except Exception as exc:   # noqa: BLE001
            msg = f"{type(exc).__name__}: {exc}"
            if msg != err:
                logger.error("線路設定仍無法套用：%s", msg)
                err = msg
            reporter.report_failure(err)
            continue
        logger.info("線路設定可以套用了（config_version=%s），開始啟動", cfg.config_version)
        return cfg


def main() -> None:
    logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(message)s")
    env = dict(os.environ)
    base = env.get("CONFIG_API") if env.get("CAMERA_SOURCE") == "api" else None
    try:
        cfg = load_line_config(env)                    # 站別不存在 / 全部相機都設定有誤才失敗；壞一台只跳過那台
    except Exception as exc:   # noqa: BLE001 —— 任何載入錯誤都要回報成 failed
        if env.get("VIDEO_FILE"):
            # 影片測試是一次性任務：設定有誤就直接失敗，讓任務看得到錯誤
            LineReporter(base, env.get("LINE_ID", "?"), 0).report_failure(f"{type(exc).__name__}: {exc}")
            logger.error("線路設定載入失敗：%s", exc)
            sys.exit(2)
        cfg = wait_for_line_config(env, base, exc, retry_s=float(env.get("CONFIG_RETRY_S", "60")))
    reporter = LineReporter(base, cfg.id, cfg.config_version)
    reporter.skipped_cameras = cfg.skipped_cameras   # 被跳過的相機每次心跳都報，健康巡檢逐台告警
    try:
        verify_models_against_triton(cfg, env)         # engine 尺寸 / 類別數要跟設定一致，否則不啟動
    except SystemExit as exc:
        reporter.report_failure(str(exc)[:500])
        logger.error("%s", exc)
        raise
    # REDIS_URL 有設 → 發到 edgevision:events（只有 NG）/ edgevision:live / edgevision:store；line_id/station 進事件供通知分組
    worker = Worker(cfg, publish=make_publisher(env, line_id=cfg.id, station=cfg.station),
                    model_version=triton_model_version(env.get("TRITON_HTTP"), env["MODEL_NAME"]))
    finished = build_and_run(cfg, worker, env, reporter)
    if env.get("VIDEO_FILE") and not finished:
        sys.exit(1)                                    # 影片測試沒播完（來源 / 管線出錯）


if __name__ == "__main__":
    main()
