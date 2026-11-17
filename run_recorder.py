#!/usr/bin/env python3
"""錄影服務：預錄緩衝 + 事件片段。

⭐ 它解決什麼

    現在如果發生 NG，你查得到「cam156 在 16:07:02 判定 NG、原因是違規動作、
    框在哪裡」—— 但【看不到當時的畫面】。推論結果經 Redis 傳的只有座標
    （1.5 KB/筆），影像留在 runner 容器裡，沒人存下來。

    這支服務從 go2rtc 另外拉一條乾淨串流，持續滾動錄影；
    收到 NG 事件時，把「前 8 秒 + 後 5 秒」剪出來存進 MinIO。

⚠️ 為什麼不是讓 runner 存

    runner 在 GPU 管線上，任何 I/O 都可能拖累推論（probe 是生死線）。
    而且它是 ai-infer 的映像，與這個 repo 分開部署。
    錄影是獨立的失敗域：它掛了推論照跑，推論掛了它還在錄。

環境變數
    REDIS_URL               必填，事件來源
    CONFIG_API              必填，相機清單（GET /api/config/lines）
    GO2RTC_RTSP             必填，例如 rtsp://localhost:8554
    PRE_RECORD_BUFFER_DIR   必填，預錄緩衝位置
    MINIO_ENDPOINT / MINIO_ROOT_USER / MINIO_ROOT_PASSWORD
    CLIP_PRE_SECONDS        事件前幾秒，預設 8
    CLIP_POST_SECONDS       事件後幾秒，預設 5
    SEGMENT_SECONDS         分段長度，預設 10
    SEGMENT_WRAP            分段數，預設 30（10×30 = 5 分鐘緩衝）
    MIN_FREE_GB             低於這個空間就不錄，預設 20
    STATS_INTERVAL_S        統計輸出間隔，預設 60
"""
from __future__ import annotations

import json
import logging
import os
import signal
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import quote
from pathlib import Path
from urllib.request import ProxyHandler, build_opener

from edgevision.recorder.clip import ClipMaker, ClipSpec
from edgevision.recorder.prebuffer import BufferSpec, PreRecordBuffer, check_space

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)-7s %(name)-24s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("recorder")


def _require(name: str) -> str:
    v = os.environ.get(name, "").strip()
    if not v:
        raise SystemExit(f"必須設定 {name}")
    return v


# ⚠️ 明確不走 proxy。
#
# 機台的 docker 會自動把 http_proxy 注入每個容器（~/.docker/config.json 的
# proxies 設定）。urlopen 預設吃那個環境變數，於是連 output-api 這種
# 同一個 compose 網路裡的服務也被送去 proxy → HTTP 502 Proxy Error。
# no_proxy 對 urllib 的支援不可靠（服務名不是 IP 也不是網域字尾），
# 所以直接建一個不含 ProxyHandler 的 opener。
_OPENER = build_opener(ProxyHandler({}))


def _get_json(url: str, timeout: int = 10):
    with _OPENER.open(url, timeout=timeout) as r:
        return json.loads(r.read().decode())


def _producer_bytes(api: str, src: str) -> int | None:
    """go2rtc 這條串流的 producer 目前收到多少 bytes（沒有 producer → None）。只讀 API，不建 consumer。"""
    d = _get_json(f"{api}/api/streams?src={quote(src)}", timeout=5)
    prods = (d or {}).get("producers") or []
    vals = [p.get("bytes_recv") for p in prods if isinstance(p.get("bytes_recv"), int)]
    return sum(vals) if vals else None


def _stream_has_data(camera_id: str, src: str, *, window: float | None = None) -> bool:
    """這條串流現在有沒有影像：看 go2rtc producer 收到的 bytes 在幾秒內有沒有增加。

    ⚠️ 以前打 /api/frame.jpeg：串流沒畫面時 go2rtc 一直等關鍵幀，我們 12 秒逾時掛斷，
       它卻不知道（沒東西可送就發現不了），那個等待中的 consumer 永遠不清 ——
       2026-11-09 正式累積約 5 萬個、go2rtc 吃到 26GB。改看 API 的 bytes_recv，不建任何 consumer。
    只用在 <cam>_marked（runner 推進來的，一定有 producer）；原始串流是 lazy producer，不適用。
    抓不到不代表相機壞了，可能只是 runner 沒在推 —— 這時候退回原始串流。
    """
    api = os.environ.get("GO2RTC_API", "http://localhost:1984")
    window = float(os.environ.get("PROBE_WINDOW_S", 3)) if window is None else window
    # ⚠️ 不能只看「有增加」：runner 卡死時 _marked 仍有零星控制封包（2026-11-09 實測 5 秒 +240 bytes）。
    min_bytes = int(os.environ.get("PROBE_MIN_BYTES", 4096))
    # ⚠️ 要重試、要記原因：一次失敗就退回原始串流的代價太高（這台之後錄的都沒框），寧可多等幾秒。
    last = ""
    for attempt in range(3):
        try:
            b0 = _producer_bytes(api, src)
            if b0 is None:
                last = "go2rtc 沒有這條串流的 producer（runner 沒在推？）"
            else:
                time.sleep(window)
                b1 = _producer_bytes(api, src)
                if b1 is not None and b1 - b0 >= min_bytes:
                    return True
                last = f"{window:.0f} 秒內新資料不到 {min_bytes} bytes（串流沒有畫面）"
        except Exception as e:  # noqa: BLE001 —— 探測失敗要記原因，不能吞掉
            last = f"{type(e).__name__}: {e}"
        if attempt < 2:
            time.sleep(4)
    log.warning("[%s] 探測 %s 三次都失敗：%s", camera_id, src, last)
    return False


# 影片存了但 DB 沒對應列的次數（每相機）。持續大於 0 就是兩條消費鏈之間有落差。
_ORPHANS: dict[str, int] = {}


def _make_and_attach(maker, db, camera_id: str, event_ts: float, payload: dict) -> None:
    """做片段，然後【把 URI 寫回 DB】。

    ⚠️ 這一步以前漏掉了：片段有上傳 MinIO，但回傳的 URI 沒人接，
       於是那些影片變成孤兒檔案 —— 躺在 MinIO 裡，沒有任何紀錄指向它們，
       前端查到 NG 也播不到影片（實際產生過 3312 個這樣的片段）。

    ⚠️ 寫 DB 失敗不能讓片段消失

       影片已經上傳成功了，DB 寫不進去只是「查不到」，不是「沒有」。
       所以這裡吞掉例外並記 error，不重試也不刪檔 ——
       事後可以靠 camera_id + 檔名時間戳補回去。
    """
    try:
        uri = maker.make(camera_id, event_ts, payload)
    except Exception:
        log.exception("[%s] 片段製作失敗", camera_id)
        return
    if not uri:
        return                      # 沒有預錄緩衝或製作失敗，make() 已記過原因

    frame_id = payload.get("frame_id") or ""
    if db is None or not frame_id:
        # 沒有 DB（或事件沒帶 frame_id）就只有檔案，沒有索引。
        # 這不是錯誤，但要講出來 —— 否則又是一批查不到的孤兒檔案。
        log.warning("[%s] 片段 %s 沒有寫回 DB（%s）—— 前端查不到這支影片",
                    camera_id, uri, "未設定資料庫" if db is None else "事件沒帶 frame_id")
        return
    try:
        n = db.attach_clip(camera_id=camera_id, frame_id=frame_id,
                           clip_uri=uri, event_ts=event_ts)
        if n:
            log.info("[%s] 片段已掛上 %d 列偵測紀錄", camera_id, n)
        else:
            # ⚠️ 這支影片是孤兒：MinIO 有、DB 沒有任何一列指向它。
            #    NG 一律落地（ng_rate=1.0），所以正常情況這裡不該發生；
            #    會發生代表 event-processor 落後、掛了、或兩邊拿到的 frame_id 不一致。
            #    要看得見 —— 先前就是因為這裡只記 debug，孤兒累積 25 GB 才被發現。
            _ORPHANS[camera_id] = _ORPHANS.get(camera_id, 0) + 1
            log.warning("[%s] 影片 %s 在 DB 找不到 frame_id=%s 的列 —— 孤兒（累計 %d）",
                        camera_id, uri, frame_id, _ORPHANS[camera_id])
    except Exception:
        log.exception("[%s] 片段 %s 寫回 DB 失敗 —— 影片還在，只是查不到", camera_id, uri)


def load_cameras(config_api: str) -> list[str]:
    """從設定 API 拿相機清單 —— DB 是真相來源，不在這裡另外維護一份。"""
    base = config_api.rstrip("/")
    out: list[str] = []
    for ln in _get_json(f"{base}/api/config/lines"):
        if not ln.get("enabled"):
            continue
        detail = _get_json(f"{base}/api/config/lines/{ln['id']}")
        out += [c["id"] for c in detail.get("cameras", []) if c.get("enabled")]
    return out


def main() -> None:
    redis_url = _require("REDIS_URL")
    config_api = _require("CONFIG_API")
    go2rtc = _require("GO2RTC_RTSP").rstrip("/")
    buf_root = Path(_require("PRE_RECORD_BUFFER_DIR"))

    min_free = float(os.environ.get("MIN_FREE_GB", 20))
    ok, free = check_space(buf_root.parent if buf_root.parent.exists() else Path("/"),
                           min_free_gb=min_free)
    if not ok:
        # ⚠️ 空間不足就【不要啟動】。舊系統沒有任何空間檢查，寫到磁碟滿為止。
        raise SystemExit(f"可用空間不足：{free} GB < {min_free} GB —— 拒絕啟動")
    log.info("可用空間 %s GB", free)

    cameras = load_cameras(config_api)
    if not cameras:
        raise SystemExit("設定 API 沒有任何啟用的相機")

    from edgevision.storage.objects import ObjectStore
    objects = ObjectStore()
    if not objects.ping():
        raise SystemExit("物件儲存無回應 —— 片段無處可存，拒絕啟動")

    # ⭐ DB 是【選用】的，但沒有它前端就查不到影片。
    #
    # 為什麼不強制：錄影的核心價值是「把畫面存下來」，那只需要 MinIO。
    # DB 掛掉時仍然該繼續錄（影片之後可以靠檔名補索引），
    # 不該因為查詢功能不可用就停止存證。
    db = None
    try:
        from edgevision.storage.db import Database
        _db = Database()
        if _db.ping():
            db = _db
        else:
            log.warning("資料庫無回應 —— 片段仍會存進 MinIO，但【前端查不到影片】")
    except Exception:
        log.exception("資料庫連線失敗 —— 片段仍會存進 MinIO，但【前端查不到影片】")

    # ① 預錄緩衝：每台相機一個 ffmpeg
    seg_s = int(os.environ.get("SEGMENT_SECONDS", 10))
    wrap = int(os.environ.get("SEGMENT_WRAP", 30))

    # ⭐ 錄【已經燒好框的那條】。
    #
    # runner 的燒錄分支已經把框畫進 <cam>_marked 了 —— 那是現成的成品，
    # 直接錄它就有框。原本我錄原始串流想「之後再疊框」，但 side-car JSON
    # 只有觸發那一幀的判定，沒有逐幀座標，「之後重畫」根本做不到。
    #
    # 代價是框燒死了不能改。接受 —— 存證影片的用途是「事後看發生什麼事」，
    # 要乾淨像素做訓練的話，那是資料集的事（走 LakeFS），不是存證的事。
    #
    # RECORD_SUFFIX="" 可切回原始串流（例如某條線沒開燒錄）。
    suffix = os.environ.get("RECORD_SUFFIX", "_marked")

    buffers = {}
    for cam in cameras:
        # ⭐ 首選 <cam>_marked（有框），沒畫面才退回原始串流。
        #    以前只在服務啟動時判斷一次：那時 runner 剛好在重啟，這台之後錄的就【永遠】沒框。
        #    現在每次 ffmpeg 重連都重新判斷，退回期間定期檢查，恢復就切回去（PreRecordBuffer）。
        def pick(cam=cam):
            if suffix and _stream_has_data(cam, f"{cam}{suffix}"):
                return f"{go2rtc}/{cam}{suffix}", True
            if suffix:
                log.warning("[%s] %s%s 沒有影像（runner 沒在跑？），暫錄原始串流 —— 這段沒有框",
                            cam, cam, suffix)
            return f"{go2rtc}/{cam}", not suffix
        spec = BufferSpec(camera_id=cam, rtsp_url=f"{go2rtc}/{cam}{suffix}",
                          out_dir=buf_root / cam,
                          segment_seconds=seg_s, segment_wrap=wrap)
        b = PreRecordBuffer(spec, pick_url=pick if suffix else None)
        b.start()
        buffers[cam] = b

    # ② 事件片段
    maker = ClipMaker(
        objects=objects, buffer_root=buf_root,
        spec=ClipSpec(pre_seconds=float(os.environ.get("CLIP_PRE_SECONDS", 8)),
                      post_seconds=float(os.environ.get("CLIP_POST_SECONDS", 5))))

    log.info("=" * 60)
    log.info("  recorder（錄影服務）")
    log.info("  相機      %d 台：%s", len(cameras), ", ".join(cameras))
    log.info("  錄影來源  %s", f"<cam>{suffix}（已燒框；沒畫面時暫錄原始串流，恢復自動切回）"
             if suffix else "<cam>（原始，無框）")
    log.info("  預錄緩衝  %d 秒（%d × %ds）→ %s", seg_s * wrap, wrap, seg_s, buf_root)
    log.info("  事件片段前 %.0fs + 後 %.0fs（無冷卻：每個 NG 事件一支，與 DB 一對一）",
             maker.spec.pre_seconds, maker.spec.post_seconds)
    log.info("  影片索引  %s", "clip_uri 會寫回 detections（前端查得到）" if db
             else "⚠️ 沒有 DB —— 影片存得了但前端查不到")
    log.info("=" * 60)

    # ⭐ 心跳：孤兒數、片段數、錯誤數 → service_status → /healthz services[]
    hb = None
    if db is not None:
        from edgevision.storage.heartbeat import Heartbeat
        hb = Heartbeat(db, service="recorder")
        hb.start(lambda: {"orphans": sum(_ORPHANS.values()),
                          **{k: v for k, v in maker.stats.snapshot().items() if k != "uptime_s"},
                          "cameras": len(buffers),
                          "prebuffer_alive": sum(1 for b in buffers.values() if b.alive)})

    stop = threading.Event()

    def _bye(_s, _f):
        log.info("收到停止訊號…")
        stop.set()
    signal.signal(signal.SIGTERM, _bye)
    signal.signal(signal.SIGINT, _bye)

    # ③ 消費 NG 事件。用 consumer group —— 多副本時不重複做同一段。
    import redis as _redis
    r = _redis.Redis.from_url(redis_url, socket_connect_timeout=5,
                              socket_timeout=10, decode_responses=True)
    stream = os.environ.get("REDIS_STREAM", "edgevision:events")
    group = os.environ.get("RECORDER_GROUP", "recorder")
    consumer = os.environ.get("RECORDER_CONSUMER") or os.uname().nodename
    try:
        # id=">" 從現在開始 —— 積壓的舊事件不該補錄（畫面早就不在緩衝裡了）
        r.xgroup_create(stream, group, id="$", mkstream=True)
        log.info("已建立 consumer group %s（從現在開始）", group)
    except Exception as exc:
        if "BUSYGROUP" not in str(exc):
            raise
        log.info("consumer group %s 已存在，接續", group)

    # 片段製作會 sleep 等後段，不能擋住讀取迴圈
    pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="clip")
    stats_every = int(os.environ.get("STATS_INTERVAL_S", 60))
    last_stats = time.time()

    while not stop.is_set():
        try:
            resp = r.xreadgroup(group, consumer, {stream: ">"}, count=50, block=2000)
            for _s, msgs in resp or []:
                for msg_id, f in msgs:
                    if f.get("type") == "ng":
                        cam = f.get("camera_id", "")
                        if cam in buffers:
                            pool.submit(_make_and_attach, maker, db, cam,
                                        float(f.get("ts", time.time())),
                                        {"reason": f.get("reason", ""),
                                         "frame_id": f.get("frame_id", ""),
                                         "details": f.get("details", "{}")})
                    r.xack(stream, group, msg_id)
        except Exception:
            log.exception("事件讀取失敗，5 秒後重試")
            stop.wait(5)
        if stats_every > 0 and time.time() - last_stats >= stats_every:
            log.info("片段 %s | 孤兒 %s | 預錄 %s", maker.stats.snapshot(),
                     sum(_ORPHANS.values()) or 0,
                     {c: b.health()["segments"] for c, b in buffers.items()})
            last_stats = time.time()

    log.info("停止預錄…")
    pool.shutdown(wait=False)
    for b in buffers.values():
        b.stop()
    if hb is not None:
        hb.stop()
    if db is not None:
        db.close()
    log.info("已停止。%s", maker.stats.snapshot())


if __name__ == "__main__":
    sys.exit(main())
