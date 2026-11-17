"""告警回流：線上 NG 告警 → 人判 → NAS 原始幀＋當下的框 → 標注專案圖片集的新批次。

    線上告警（detections 的 NG 列）
      → 待判清單：一個事件一筆（同相機、同一幀、同一站），每站每天抽 N 個（queue）
      → 人判：真異常 / 誤報（review；寫回 detections.reviewed / review_result）
      → 每天回流（plan_feedback → 匯入任務 kind=alert）：
          依事件的模型找進行中的標注專案 → 它的圖片集開一個批次「告警回流 <日期>」
          每個事件：NAS 原始錄影找到告警那一幀（locate）→ 存圖 → 當下的框當標注
      → 標注專案照常接手（預標 → 審核 → 匯出 → 訓練）

對幀（locate）—— 告警時間只準到 ±1 秒，框是那一幀的，差 1 秒框就偏掉（實測框整個錯位），所以不能只靠時間：
    ① 證據影片（打了框的 <cam>_marked）跟 NAS 原始錄影逐幀比畫面 → 兩條時間軸的差（lag）
    ② 證據影片裡「框畫在哪一幀」：對齊後兩邊相減只剩疊加的框線，資料庫那幾個框的外框線差異最大的那一幀就是
    ③ NAS 上同一時刻那一幀 = 告警的乾淨原圖
    證據影片已過期 / 對不準 → 依設定（align.fallback）放進去當預標或跳過。

預設值都在 alert_feedback_defaults.yaml。
"""

from __future__ import annotations

import json
import logging
import os
from datetime import date, datetime, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from edgevision.media.library import _row, _rows

logger = logging.getLogger(__name__)

DEFAULTS_PATH = Path(__file__).with_name("alert_feedback_defaults.yaml")
SYSTEM_ACTOR = "edgevision-alert-feedback"
RESULTS = ("true_positive", "false_positive")


@lru_cache(maxsize=1)
def defaults() -> dict:
    return yaml.safe_load(DEFAULTS_PATH.read_text(encoding="utf-8"))


def _day_range(day: date):
    from edgevision.media.frames import LOCAL_TZ
    t0 = datetime(day.year, day.month, day.day, tzinfo=LOCAL_TZ)
    return t0, t0 + timedelta(days=1)


# ── 事件 ─────────────────────────────────────────────────

_EVENT_SQL = """
    SELECT min(id) AS event_id, camera_id, frame_id, station, min(ts) AS ts,
           max(model_name) AS model_name, max(model_version) AS model_version,
           max(verdict_reason) AS reason, max(clip_uri) AS clip_uri,
           bool_or(reviewed) AS reviewed, max(review_result) AS review_result,
           max(reviewed_by) AS reviewed_by, max(reviewed_at) AS reviewed_at
      FROM detections
     WHERE verdict_status = 'ng' AND ts >= %s AND ts < %s {extra}
     GROUP BY camera_id, frame_id, station
"""


def line_cameras(db, line_id: str) -> list[str]:
    with db.cursor() as cur:
        cur.execute("SELECT id FROM cameras WHERE line_id = %s ORDER BY id", (line_id,))
        return [r[0] for r in cur.fetchall()]


def queue(db, *, day: date, per_station: int, line_id: str | None = None, camera_id: str | None = None,
          station: str | None = None) -> dict:
    """待判清單：這天的 NG 事件，每站抽 per_station 個（依事件 id 的雜湊排序 —— 同一天重複查是同一批）。

    判過的一律列出（不管有沒有被抽到），所以清單不會因為判了幾筆就換人。
    """
    t0, t1 = _day_range(day)
    extra, args = "", [t0, t1]
    if camera_id:
        extra += " AND camera_id = %s"
        args.append(camera_id)
    elif line_id:
        extra += " AND camera_id = ANY(%s)"
        args.append(line_cameras(db, line_id))
    if station:
        extra += " AND station = %s"
        args.append(station)
    sql = (f"WITH ev AS ({_EVENT_SQL.format(extra=extra)}),"
           " r AS (SELECT ev.*, row_number() OVER (PARTITION BY station ORDER BY md5(event_id::text)) AS rn,"
           "              count(*) OVER (PARTITION BY station) AS n_station FROM ev)"
           " SELECT r.*, f.status AS fb_status, f.reason AS fb_reason, f.collection_id AS fb_collection_id,"
           "        f.batch_id AS fb_batch_id, f.item_id AS fb_item_id, f.item_status AS fb_item_status"
           "   FROM r LEFT JOIN alert_feedback f ON f.event_id = r.event_id"
           "  WHERE r.rn <= %s OR r.reviewed ORDER BY r.station, r.ts")
    with db.cursor() as cur:
        cur.execute(sql, (*args, per_station))
        rows = _rows(cur)
    stations: dict[str, dict] = {}
    for r in rows:
        s = stations.setdefault(r["station"] or "", {"station": r["station"], "total": int(r["n_station"]),
                                                     "listed": 0, "reviewed": 0})
        s["listed"] += 1
        s["reviewed"] += int(bool(r["reviewed"]))
    return {"day": day.isoformat(), "per_station": per_station, "stations": list(stations.values()),
            "events": rows}


def event_boxes(db, ev: dict) -> tuple[list[dict], dict | None]:
    """事件那一幀的所有框（原圖像素座標）與影像尺寸。"""
    with db.cursor() as cur:
        cur.execute("SELECT class_name, confidence, bbox, image_meta FROM detections"
                    " WHERE camera_id = %s AND frame_id = %s AND station IS NOT DISTINCT FROM %s"
                    "   AND verdict_status = 'ng' AND ts BETWEEN %s - interval '2 minutes' AND %s + interval '2 minutes'"
                    "   AND class_name IS NOT NULL ORDER BY id",
                    (ev["camera_id"], ev["frame_id"], ev["station"], ev["ts"], ev["ts"]))
        rows = cur.fetchall()
    meta = None
    boxes = []
    for cls, conf, bbox, im in rows:
        meta = meta or (im if isinstance(im, dict) else json.loads(im or "null"))
        b = bbox if isinstance(bbox, dict) else json.loads(bbox)
        boxes.append({"class": cls, "conf": round(float(conf), 4) if conf is not None else None,
                      "x1": b["x1"], "y1": b["y1"], "x2": b["x2"], "y2": b["y2"]})
    size = {"width": meta["width"], "height": meta["height"]} if meta else None
    return boxes, size


def get_event(db, event_id: int) -> dict | None:
    with db.cursor() as cur:
        cur.execute("SELECT camera_id, frame_id, station, ts FROM detections WHERE id = %s AND verdict_status = 'ng'",
                    (event_id,))
        r = _row(cur)
    if r is None:
        return None
    with db.cursor() as cur:
        cur.execute(f"WITH ev AS ({_EVENT_SQL.format(extra=' AND camera_id = %s AND frame_id = %s')})"
                    " SELECT * FROM ev WHERE station IS NOT DISTINCT FROM %s",
                    (r["ts"] - timedelta(minutes=2), r["ts"] + timedelta(minutes=2), r["camera_id"], r["frame_id"],
                     r["station"]))
        return _row(cur)


def review(db, event_id: int, *, result: str | None, actor: str) -> dict:
    """判一個事件（同一幀、同一站的每一列一起標）。result=None = 取消判定。已回流的不能再改。"""
    if result is not None and result not in RESULTS:
        raise ValueError(f"result 只能是 {RESULTS} 或 null")
    ev = get_event(db, event_id)
    if ev is None:
        raise LookupError(f"沒有告警事件 {event_id}")
    with db.cursor() as cur:
        cur.execute("SELECT status FROM alert_feedback WHERE event_id = %s", (event_id,))
        f = cur.fetchone()
        if f and f[0] in ("queued", "added"):
            raise PermissionError(f"事件 {event_id} 已回流（{f[0]}），不能再改判定")
        cur.execute("UPDATE detections SET reviewed = %s, review_result = %s, reviewed_by = %s, reviewed_at = now()"
                    " WHERE camera_id = %s AND frame_id = %s AND station IS NOT DISTINCT FROM %s"
                    "   AND verdict_status = 'ng' AND ts BETWEEN %s - interval '2 minutes' AND %s + interval '2 minutes'",
                    (result is not None, result, actor if result else None, ev["camera_id"], ev["frame_id"],
                     ev["station"], ev["ts"], ev["ts"]))
        n = cur.rowcount
    return {"event_id": event_id, "review_result": result, "rows": n}


# ── 回流規劃（控制 API 呼叫；建任務在 API 層）───────────────────

def pending_events(db, *, lookback_days: int, now: datetime | None = None) -> list[dict]:
    """最近 N 天判過、還沒回流（或上次失敗）的事件。"""
    now = now or datetime.now().astimezone()
    with db.cursor() as cur:
        cur.execute(f"WITH ev AS ({_EVENT_SQL.format(extra=' AND reviewed')})"
                    " SELECT ev.* FROM ev LEFT JOIN alert_feedback f ON f.event_id = ev.event_id"
                    " WHERE ev.review_result IS NOT NULL AND (f.event_id IS NULL OR f.status = 'failed')"
                    " ORDER BY ev.ts", (now - timedelta(days=lookback_days), now))
        return _rows(cur)


def target_collection(db, model_name: str) -> dict | None:
    """這個模型進行中（未結案）的標注專案的待審圖片集。"""
    with db.cursor() as cur:
        cur.execute("SELECT id AS campaign_id, name AS campaign_name, collection_id,"
                    " config->'dataset'->>'task' AS label_task FROM media_campaigns"
                    " WHERE model_name = %s AND state <> 'closed' ORDER BY id DESC LIMIT 1", (model_name,))
        return _row(cur)


def claim(db, events: list[dict], *, task_id: str, collection_id: int, batch_id: int) -> None:
    with db.cursor() as cur:
        for e in events:
            cur.execute(
                """INSERT INTO alert_feedback (event_id, camera_id, frame_id, station, event_ts, model_name,
                       model_version, review_result, status, collection_id, batch_id, task_id)
                   VALUES (%s,%s,%s,%s,%s,%s,%s,%s,'queued',%s,%s,%s)
                   ON CONFLICT (event_id) DO UPDATE SET status = 'queued', reason = NULL, collection_id = EXCLUDED.collection_id,
                       batch_id = EXCLUDED.batch_id, task_id = EXCLUDED.task_id, review_result = EXCLUDED.review_result,
                       updated_at = now()""",
                (e["event_id"], e["camera_id"], e["frame_id"], e["station"], e["ts"], e["model_name"],
                 e["model_version"], e["review_result"], collection_id, batch_id, task_id))


def mark(db, event_id: int, status: str, *, reason: str | None = None, item_id: int | None = None,
         item_status: str | None = None, align: dict | None = None) -> None:
    with db.cursor() as cur:
        cur.execute("UPDATE alert_feedback SET status = %s, reason = %s, item_id = %s, item_status = %s, align = %s,"
                    " updated_at = now() WHERE event_id = %s",
                    (status, reason, item_id, item_status, json.dumps(align, default=str) if align else None, event_id))


def release_queued(db, task_id: str) -> int:
    """任務結束時還是 queued 的（任務掛掉 / 取消）→ failed，下次回流重試。"""
    with db.cursor() as cur:
        cur.execute("UPDATE alert_feedback SET status = 'failed', reason = '回流任務中斷', updated_at = now()"
                    " WHERE task_id = %s AND status = 'queued'", (task_id,))
        return cur.rowcount


def mark_skipped(db, events: list[dict], reason: str) -> None:
    """沒地方放（例如模型沒有進行中的專案、也建不了圖片集）：記下來，下次不再重試。"""
    with db.cursor() as cur:
        for e in events:
            cur.execute(
                """INSERT INTO alert_feedback (event_id, camera_id, frame_id, station, event_ts, model_name,
                       model_version, review_result, status, reason)
                   VALUES (%s,%s,%s,%s,%s,%s,%s,%s,'skipped',%s)
                   ON CONFLICT (event_id) DO UPDATE SET status = 'skipped', reason = EXCLUDED.reason, updated_at = now()""",
                (e["event_id"], e["camera_id"], e["frame_id"], e["station"], e["ts"], e["model_name"],
                 e["model_version"], e["review_result"], reason))


def history(db, *, limit: int = 200, status: str | None = None, task_id: str | None = None) -> list[dict]:
    sql, args = "SELECT * FROM alert_feedback WHERE true", []
    if status:
        sql += " AND status = %s"
        args.append(status)
    if task_id:
        sql += " AND task_id = %s"
        args.append(task_id)
    sql += " ORDER BY updated_at DESC LIMIT %s"
    args.append(limit)
    with db.cursor() as cur:
        cur.execute(sql, args)
        return _rows(cur)


# ── 框 → 標注 ─────────────────────────────────────────────

def boxes_to_shapes(boxes: list[dict], size: dict, *, model: str, version: str | None,
                    class_names: list[str]) -> tuple[list[dict], list[str]]:
    """原圖像素座標的框 → 圖片集的 rect（0–1）。圖片集沒有的類別丟掉，回 (shapes, 丟掉的類別)。"""
    w, h = float(size["width"]), float(size["height"])
    ver = int(version) if version and str(version).isdigit() else None
    shapes, dropped = [], []
    for i, b in enumerate(boxes):
        if b["class"] not in class_names:
            dropped.append(b["class"])
            continue

        def c(v, m):
            return min(1.0, max(0.0, v / m))
        shapes.append({"id": f"alert-{i}", "type": "rect", "class": b["class"],
                       "points": [[c(b["x1"], w), c(b["y1"], h)], [c(b["x2"], w), c(b["y2"], h)]],
                       "source": {"kind": "model", "engine": "online", "model": model, "version": ver,
                                  "conf": b.get("conf")}})
    return shapes, sorted(set(dropped))


def item_status(review_result: str, *, aligned: bool, shapes: list[dict], dropped: list[str], task: str,
                cfg: dict) -> str | None:
    """放進圖片集的狀態。None = 不帶標注（unlabeled，交給預標）。"""
    if task != "detect":
        return None                       # 線上只存框（沒有遮罩 / 關鍵點）：要訓練分割 / 姿態時交給預標
    if not shapes:
        return None                       # 整幀判定的 NG（當下沒有框）：沒有標注可帶
    want = cfg["feedback"]["status"].get(review_result, "auto")
    if not aligned or dropped:
        return "auto"                     # 對不準 / 有類別對不上：當預標，進審核
    return want


# ── 對幀 ─────────────────────────────────────────────────

def _small(img, width: int = 320):
    """彩色小圖（uint8）。框線是飽和色，灰階下黃框疊在白色物件上幾乎看不出來，所以比對用彩色。"""
    import cv2
    h = max(1, width * img.shape[0] // img.shape[1])
    return cv2.resize(img, (width, h), interpolation=cv2.INTER_AREA)


def _z(img):
    import cv2
    import numpy as np
    g = cv2.cvtColor(cv2.resize(img, (64, max(1, 64 * img.shape[0] // img.shape[1])), interpolation=cv2.INTER_AREA),
                     cv2.COLOR_BGR2GRAY).astype(np.float32)
    return (g - g.mean()) / (g.std() + 1e-6)


def locate(clip: list[tuple[float, Any]], nas: list[tuple[float, Any]], boxes: list[dict], size: dict | None,
           cfg: dict, *, ts: float | None = None) -> dict:
    """證據影片（有框）＋NAS 原始錄影（同一段時間）→ 告警那一幀在 NAS 上的時間。

    clip / nas：[(時間秒, 彩色小圖 ndarray，見 _small)]，兩者時間軸各自為準（NAS 用絕對 epoch 秒）。
    ts：資料庫記的告警時間（epoch）。只在它前後 align.max_ts_offset_s 秒內挑（實測 NAS 上那一幀在 ts 前 0.1–0.2 秒）。
    回 {"ok": bool, "nas_t": float | None, "lag": …, "lag_agree": …, "clip_t": …, "score": …, "why": …}

    分數相近的幾幀不算「對不準」：分數高 = 框線跟資料庫的框吻合 = 那幾幀的框都對（畫面那段時間沒怎麼動）。
    """
    import cv2
    import numpy as np

    a = cfg["align"]
    out: dict = {"ok": False, "nas_t": None}
    if not clip or not nas:
        return out | {"why": "沒有證據影片" if not clip else "NAS 這段沒有錄影"}
    tn = np.array([t for t, _ in nas])
    nz = np.stack([_z(g) for _, g in nas])
    lags = []
    for i in range(0, len(clip), max(1, int(a["lag_sample_every"]))):
        e = ((nz - _z(clip[i][1])) ** 2).mean(axis=(1, 2))
        lags.append(tn[int(e.argmin())] - clip[i][0])
    lags = np.array(lags)
    lag = float(np.median(lags))
    agree = float(np.mean(np.abs(lags - lag) <= a["lag_agree_s"]))
    out |= {"lag": round(lag, 3), "lag_agree": round(agree, 3)}
    if agree < a["lag_min_agree"]:
        return out | {"why": f"證據影片跟 NAS 對不起來（一致率 {agree:.0%}）"}
    if not boxes or not size:
        return out | {"why": "這一幀沒有框，無法確定是哪一幀"}

    H, W = clip[0][1].shape[:2]
    masks = []
    for b in boxes:
        m = np.zeros((H, W), np.uint8)
        p1 = (int(b["x1"] / size["width"] * W), int(b["y1"] / size["height"] * H))
        p2 = (int(b["x2"] / size["width"] * W), int(b["y2"] / size["height"] * H))
        cv2.rectangle(m, p1, p2, 1, 1)
        if m.any():
            masks.append(m.astype(bool))
    if not masks:
        return out | {"why": "框太小"}
    vals, cts = [], []
    for t, g in clip:
        j = int(np.abs(tn - (t + lag)).argmin())
        if nas[j][1].shape != g.shape:
            continue
        # 對齊後兩邊相減只剩疊加的框線；看資料庫那幾個框的外框線上差異多大
        diff = np.abs(g.astype(np.int16) - nas[j][1].astype(np.int16)).max(axis=2).astype(np.float32)
        base = float(np.median(diff))
        vals.append([float(diff[m].mean()) - base for m in masks])
        cts.append(t)
    if not vals:
        return out | {"why": "畫面尺寸對不上"}
    v = np.array(vals)
    # 每個框扣掉它自己在各幀的中位數：不會動的框（墊子）每幀都一樣高，扣掉後只剩會動的框決定是哪一幀；
    # 再取平均 —— 某個框線上沒畫也不會把整幀拉到 0
    excess = (v - np.median(v, axis=0)).mean(axis=1)
    scores = [(sc, t) for sc, t in zip(excess.tolist(), cts)
              if ts is None or abs(t + lag - ts) <= a["max_ts_offset_s"]]
    if not scores:
        return out | {"why": f"證據影片沒拍到告警前後 {a['max_ts_offset_s']} 秒（影片太短）"}
    best, t_best = max(scores)
    nas_t = t_best + lag
    gap = float(np.abs(tn - nas_t).min())
    out |= {"clip_t": round(t_best, 3), "score": round(best, 2), "nas_t": nas_t, "nas_gap": round(gap, 3),
            "ts_offset": round(nas_t - ts, 3) if ts is not None else None}
    if best < a["box_min_score"]:
        return out | {"why": f"找不到框畫在哪一幀（分數 {best:.1f}）"}
    if gap > a["nas_max_gap_s"]:
        return out | {"why": f"NAS 在那一刻缺幀（最近差 {gap:.2f}s）"}
    return out | {"ok": True}


def decode_all(path: Path, *, start_s: float = 0.0, end_s: float | None = None, small: bool = True):
    """解出每一幀 (相對開頭秒數, 影像)；small=True 回小圖（省記憶體）。"""
    import av

    from edgevision.media.frames import _t
    with av.open(str(path)) as c:
        st = c.streams.video[0]
        st.thread_type = "AUTO"
        if start_s > 1:
            c.seek(int(start_s / st.time_base), stream=st, any_frame=False, backward=True)
        for fr in c.decode(st):
            if fr.pts is None:
                continue
            t = _t(fr, st)
            if t < start_s - 1e-6:
                continue
            if end_s is not None and t > end_s:
                break
            img = fr.to_ndarray(format="bgr24")
            yield t, (_small(img) if small else img)


def nas_window(root: Path, camera_id: str, ts: datetime, before_s: float, after_s: float):
    """NAS 錄影在 [ts-before, ts+after] 的每一幀 (epoch 秒, 小圖) 與來源 [(path, 段起點)]。"""
    from edgevision.media.frames import nas_segments
    t0, t1 = ts - timedelta(seconds=before_s), ts + timedelta(seconds=after_s)
    try:
        segs = nas_segments(root, camera_id, t0, t1)
    except FileNotFoundError:
        return [], []
    frames = []
    for path, s0 in segs:
        a = max(0.0, (t0 - s0).total_seconds())
        b = (t1 - s0).total_seconds()
        try:
            for t, g in decode_all(path, start_s=a, end_s=b):
                frames.append((s0.timestamp() + t, g))
        except Exception as exc:  # noqa: BLE001 —— 還在寫的那一段（moov 未寫）打不開
            logger.warning("NAS %s 讀不了：%s", path.name, exc)
    frames.sort(key=lambda x: x[0])
    return frames, segs


def nas_frame(segs: list[tuple[Path, datetime]], t_epoch: float):
    """NAS 上最接近 t_epoch 的那一幀（原解析度 BGR）。"""
    from edgevision.media.frames import decode_at
    cand = [(p, s0) for p, s0 in segs if s0.timestamp() <= t_epoch + 0.5]
    if not cand:
        return None
    path, s0 = cand[-1]
    off = max(0.0, t_epoch - s0.timestamp())
    for _t, img in decode_at(path, [off]):
        return img
    return None


# ── 匯入任務（edgevision.media.jobs import，kind=alert）────────────────

def import_alerts(lib, cid: int, p: dict, *, actor: str, task_id: str, progress, batch_id: int | None) -> dict:
    """params.events 逐一：NAS 找那一幀 → 存圖 → 框當標注。一個事件失敗不影響其他。"""
    import tempfile

    import cv2

    from edgevision.media.frames import LOCAL_TZ

    cfg = defaults()
    root = Path(os.environ.get("MEDIA_NAS_ROOT", "/nas/raw"))
    col = lib.collection(cid)
    events = p["events"]
    counts = {"added": 0, "labeled": 0, "auto": 0, "unlabeled": 0, "skipped": 0, "failed": 0, "already": 0}
    for k, ev in enumerate(events):
        if progress.cancelled():
            raise KeyboardInterrupt
        progress(f"告警回流 {k + 1}/{len(events)}：{ev['camera_id']} {ev['ts']}")
        eid = int(ev["event_id"])
        try:
            ts = datetime.fromisoformat(ev["ts"]) if isinstance(ev["ts"], str) else ev["ts"]
            nas, segs = nas_window(root, ev["camera_id"], ts, cfg["align"]["before_s"], cfg["align"]["after_s"])
            if not nas:
                mark(lib.db, eid, "skipped", reason=f"NAS 沒有 {ev['camera_id']} 在 {ts:%m-%d %H:%M:%S} 的錄影")
                counts["skipped"] += 1
                continue
            clip = []
            if ev.get("clip_uri"):
                with tempfile.TemporaryDirectory(prefix="alert-") as td:
                    b, key = lib.objects.parse_uri(ev["clip_uri"])
                    local = Path(td) / "clip.mp4"
                    try:
                        lib.objects.download(bucket=b, key=key, dest=local)
                        clip = list(decode_all(local))
                    except Exception as exc:              # noqa: BLE001 —— 過期 / 被清掉
                        logger.info("證據影片 %s 拿不到：%s", ev["clip_uri"], exc)
            boxes, size = ev.get("boxes") or [], ev.get("size")
            loc = locate(clip, nas, boxes, size, cfg, ts=ts.timestamp())
            if loc["ok"]:
                t_pick = loc["nas_t"]
            elif cfg["align"]["fallback"] == "auto":
                t_pick = ts.timestamp()
            else:
                mark(lib.db, eid, "skipped", reason=f"對不到那一幀：{loc.get('why')}", align=loc)
                counts["skipped"] += 1
                continue
            img = nas_frame(segs, t_pick)
            if img is None:
                mark(lib.db, eid, "skipped", reason="NAS 取不到那一幀", align=loc)
                counts["skipped"] += 1
                continue
            ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            if not ok:
                raise RuntimeError("JPEG 編碼失敗")
            captured = datetime.fromtimestamp(t_pick, tz=LOCAL_TZ)
            ref = lib.put_image(buf.tobytes(), filename=f"{ev['camera_id']}_{captured:%Y%m%d_%H%M%S_%f}"[:-3] + ".jpg",
                                actor=actor)
            shapes, dropped = boxes_to_shapes(boxes, size, model=ev["model_name"], version=ev.get("model_version"),
                                              class_names=col["class_names"]) if size else ([], [])
            status = item_status(ev["review_result"], aligned=loc["ok"], shapes=shapes, dropped=dropped,
                                 task=p.get("label_task") or "detect", cfg=cfg)
            info = {"alert": {"event_id": eid, "camera_id": ev["camera_id"], "frame_id": ev["frame_id"],
                              "station": ev.get("station"), "reason": ev.get("reason"),
                              "review_result": ev["review_result"], "model": ev["model_name"],
                              "model_version": ev.get("model_version"), "clip_uri": ev.get("clip_uri"),
                              "aligned": loc["ok"], "why": loc.get("why"), "dropped_classes": dropped}}
            item_id, created = lib.add_item(
                cid, ref.sha256, actor=actor, camera_id=ev["camera_id"], captured_at=captured,
                original_name=f"告警 {ev['camera_id']} {captured:%m-%d %H:%M:%S}", select_info=info,
                shapes=shapes if status else None, status=status, task_id=task_id, batch_id=batch_id)
            st = status or "unlabeled"
            mark(lib.db, eid, "added", item_id=item_id, item_status=st if created else "already_in_collection",
                 align={k_: v for k_, v in loc.items() if k_ != "nas_t"} | {"nas_t": t_pick})
            counts["added"] += int(created)
            counts["already"] += int(not created)
            if created:
                counts[st] += 1
        except Exception as exc:                          # 一個事件壞掉不影響其他
            logger.exception("告警事件 %s 回流失敗", eid)
            mark(lib.db, eid, "failed", reason=str(exc)[:500])
            counts["failed"] += 1
    return {"events": len(events), **counts}


# ── 每天的回流（CronJob alert-feedback 呼叫）──────────────────

def main() -> int:
    """python -m edgevision.media.alerts run：呼叫控制 API 執行一次回流（建批次、任務都在 API 裡，跟手動按同一條路）。"""
    import sys
    import urllib.request

    if sys.argv[1:2] != ["run"]:
        print("用法：python -m edgevision.media.alerts run", file=sys.stderr)
        return 2
    url = os.environ.get("CONTROL_API", "http://control-api:8090").rstrip("/") + "/api/media/alerts/feedback/run"
    req = urllib.request.Request(url, data=b"{}", method="POST", headers={
        "Content-Type": "application/json", "X-Actor": SYSTEM_ACTOR, "X-Actor-Type": "system", "X-Channel": "cron"})
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))      # 叢集內，不走 proxy
    with opener.open(req, timeout=300) as r:
        out = json.loads(r.read())
    logging.basicConfig(level=logging.INFO)
    logger.info("告警回流：%d 個事件，%s", out["events"],
                "；".join(f"{g['model_name']} {g['events']} 個 → " + (f"圖片集 {g['collection_id']} 任務 {g['task_id']}"
                                                                   if g.get("task_id") else f"略過：{g.get('skipped')}")
                         for g in out["groups"]) or "沒有要回流的")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
