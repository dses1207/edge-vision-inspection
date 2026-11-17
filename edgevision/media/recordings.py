"""NAS 24h 錄影：列出、播放 / 下載、剪成影片、錄影名單。

    NAS（nas-recorder 寫）  <MEDIA_NAS_ROOT>/<camera>/YYYYMMDD_HHMMSS.mp4（檔名 = 分段開始的本地時間）
      → 列出：某台相機有哪幾天、每段起訖、缺口、壞檔、用量（list_days / list_day / usage）
      → 播放 / 下載：控制 API 直接回檔（支援 Range，<video> 可以拖）
      → 剪片段：起訖時間 → 影片庫的一筆（kind = recording_clip，見 edgevision/media/jobs.py run_clip）
    錄影名單（rec_roster 表）：哪些相機要錄；nas-recorder 每 CAMERA_REFRESH_S 秒重讀一次（REC_ROSTER=db）。

每段的長度要開檔才知道（moov 在檔尾），開過的記在 nas_segments 表；檔案大小 / 修改時間變了才重開。
最新那一段 ffmpeg 還在寫（moov 未寫）→ status = writing，不算壞檔。
"""

from __future__ import annotations

import logging
import os
import re
import threading
import time
from datetime import date, datetime, timedelta
from pathlib import Path

from edgevision.media.frames import LOCAL_TZ, nas_segment_start
from edgevision.media.library import _rows

logger = logging.getLogger(__name__)

FILE_RE = re.compile(r"^\d{8}_\d{6}\.mp4$")
CAM_RE = re.compile(r"^[A-Za-z0-9_\-]{1,64}$")
WRITING_S = 120          # 修改時間在這幾秒內的最新一段 = 還在寫
GAP_TOLERANCE_S = 15     # 檔名時間只準到幾秒（ffmpeg 開新段時才命名），差超過這個才算缺口


class NasUnavailable(RuntimeError):
    pass


def root() -> Path:
    r = Path(os.environ.get("MEDIA_NAS_ROOT", "/nas/raw"))
    if not r.is_dir():
        raise NasUnavailable(f"這個環境沒有 NAS 錄影（{r} 不存在）")
    return r


def cam_dir(camera_id: str) -> Path:
    if not CAM_RE.match(camera_id):
        raise ValueError(f"相機名稱不合法：{camera_id!r}")
    return root() / camera_id


def file_path(camera_id: str, name: str) -> Path:
    """只允許 <root>/<camera>/YYYYMMDD_HHMMSS.mp4：不給跳出目錄。"""
    if not FILE_RE.match(name):
        raise ValueError(f"檔名不合法：{name!r}")
    p = cam_dir(camera_id) / name
    if not p.is_file():
        raise FileNotFoundError(f"NAS 沒有 {camera_id}/{name}")
    return p


def _files(camera_id: str) -> list[tuple[Path, datetime]]:
    d = cam_dir(camera_id)
    if not d.is_dir():
        return []
    out = []
    for e in os.scandir(d):
        if e.is_file() and FILE_RE.match(e.name):
            out.append((Path(e.path), nas_segment_start(Path(e.name))))
    return sorted(out, key=lambda x: x[1])


def cameras() -> list[dict]:
    """NAS 上有錄影目錄的相機（不開檔，只看檔名與大小）。"""
    out = []
    for e in sorted(os.scandir(root()), key=lambda e: e.name):
        if not (e.is_dir() and CAM_RE.match(e.name)):
            continue
        files = _files(e.name)
        if not files:
            out.append({"camera_id": e.name, "files": 0})
            continue
        size = latest_mtime = 0
        for p, _ in files:
            st = p.stat()
            size += st.st_size
            latest_mtime = max(latest_mtime, st.st_mtime)
        out.append({"camera_id": e.name, "files": len(files), "size_gb": round(size / 1e9, 2),
                    "first": files[0][1], "last": files[-1][1],
                    "last_write_age_s": round(time.time() - latest_mtime, 1),
                    "days": len({t.date() for _, t in files})})
    return out


def usage() -> dict:
    st = os.statvfs(root())
    total, free = st.f_blocks * st.f_frsize, st.f_bavail * st.f_frsize
    return {"total_gb": round(total / 1e9, 1), "free_gb": round(free / 1e9, 1), "used_gb": round((total - free) / 1e9, 1)}


def list_days(camera_id: str) -> list[dict]:
    """這台相機有錄影的日子：段數、大小、第一段 / 最後一段的開始時間（不開檔）。"""
    days: dict[date, dict] = {}
    for p, t in _files(camera_id):
        d = days.setdefault(t.date(), {"date": t.date().isoformat(), "segments": 0, "size_bytes": 0,
                                       "first": t, "last": t})
        d["segments"] += 1
        d["size_bytes"] += p.stat().st_size
        d["last"] = t
    out = []
    for d in sorted(days.values(), key=lambda x: x["date"], reverse=True):
        d["size_gb"] = round(d.pop("size_bytes") / 1e9, 2)
        out.append(d)
    return out


def _probe(path: Path) -> tuple[float | None, str | None]:
    """(長度秒, 錯誤)。開不了 = 壞檔（或還在寫）。"""
    import av
    try:
        with av.open(str(path)) as c:
            st = c.streams.video[0]
            if c.duration:
                return float(c.duration / 1e6), None
            if st.duration:
                return float(st.duration * st.time_base), None
            return None, "讀不到長度"
    except Exception as exc:  # 壞檔 / 還在寫（moov 未寫）
        return None, str(exc)[:300]


def _cached(db, camera_id: str, names: list[str]) -> dict[str, dict]:
    if db is None or not names:
        return {}
    with db.cursor() as cur:
        cur.execute("SELECT file, size_bytes, mtime, duration_s, error FROM nas_segments"
                    " WHERE camera_id = %s AND file = ANY(%s)", (camera_id, names))
        return {r["file"]: r for r in _rows(cur)}


def _remember(db, camera_id: str, name: str, start: datetime, size: int, mtime: float, dur: float | None,
              err: str | None) -> None:
    if db is None:
        return
    with db.cursor() as cur:
        cur.execute(
            """INSERT INTO nas_segments (camera_id, file, start_at, size_bytes, mtime, duration_s, error, probed_at)
               VALUES (%s,%s,%s,%s,%s,%s,%s,now())
               ON CONFLICT (camera_id, file) DO UPDATE SET size_bytes = EXCLUDED.size_bytes, mtime = EXCLUDED.mtime,
                   duration_s = EXCLUDED.duration_s, error = EXCLUDED.error, probed_at = now()""",
            (camera_id, name, start, size, mtime, dur, err))


def list_day(db, camera_id: str, day: date, *, gap_tolerance_s: float = GAP_TOLERANCE_S) -> dict:
    """某天的每一段（起訖、長度、大小、狀態）與缺口。前一天最後一段跨到今天的也算進來。"""
    files = _files(camera_id)
    t0 = datetime(day.year, day.month, day.day, tzinfo=LOCAL_TZ)
    t1 = t0 + timedelta(days=1)
    idx = [i for i, (_, t) in enumerate(files) if t0 <= t < t1]
    if idx and idx[0] > 0:
        idx.insert(0, idx[0] - 1)                       # 前一天最後一段可能跨過午夜
    picked = [files[i] for i in idx]
    cache = _cached(db, camera_id, [p.name for p, _ in picked])
    latest = files[-1][0].name if files else None
    segs = []
    for p, start in picked:
        st = p.stat()
        c = cache.get(p.name)
        if c and c["size_bytes"] == st.st_size and abs((c["mtime"] or 0) - st.st_mtime) < 1:
            dur, err = c["duration_s"], c["error"]
        else:
            dur, err = _probe(p)
            writing = p.name == latest and time.time() - st.st_mtime < WRITING_S
            if not writing:                              # 還在寫的不記，下次再開
                _remember(db, camera_id, p.name, start, st.st_size, st.st_mtime, dur, err)
        writing = p.name == latest and time.time() - st.st_mtime < WRITING_S
        status = "writing" if writing else ("ok" if dur is not None else "broken")
        end = start + timedelta(seconds=dur) if dur else (datetime.fromtimestamp(st.st_mtime, LOCAL_TZ)
                                                          if writing else None)
        segs.append({"file": p.name, "start": start, "end": end, "duration_s": round(dur, 2) if dur else None,
                     "size_mb": round(st.st_size / 1e6, 1), "status": status, "error": err if status == "broken" else None})
    segs = [s for s in segs if (s["end"] or s["start"]) > t0 and s["start"] < t1]
    gaps = []
    prev_end = t0
    for s in segs:
        if s["start"] > prev_end and (s["start"] - prev_end).total_seconds() > gap_tolerance_s:
            gaps.append({"start": prev_end, "end": s["start"], "seconds": round((s["start"] - prev_end).total_seconds())})
        if s["end"]:
            prev_end = max(prev_end, s["end"])
    now = datetime.now(LOCAL_TZ)
    day_end = min(t1, now)
    if prev_end < day_end and (day_end - prev_end).total_seconds() > gap_tolerance_s and segs:
        gaps.append({"start": prev_end, "end": day_end, "seconds": round((day_end - prev_end).total_seconds())})
    covered = sum((min(s["end"], t1) - max(s["start"], t0)).total_seconds() for s in segs if s["end"])
    return {"camera_id": camera_id, "date": day.isoformat(), "segments": segs, "gaps": gaps,
            "covered_s": round(covered), "broken": sum(s["status"] == "broken" for s in segs),
            "gap_tolerance_s": gap_tolerance_s}


def segments_between(camera_id: str, start: datetime, end: datetime) -> list[tuple[Path, datetime]]:
    from edgevision.media.frames import nas_segments
    return nas_segments(root(), camera_id, start, end)


# ── 錄影名單 ─────────────────────────────────────────────

def roster(db) -> list[dict]:
    """所有相機（cameras 表）＋要不要錄＋最近 24 小時 NAS 實際寫了多少。"""
    with db.cursor() as cur:
        cur.execute("""SELECT c.id AS camera_id, c.line_id, c.name, c.enabled AS camera_enabled,
                              coalesce(r.enabled, false) AS recording, r.note, r.updated_by, r.updated_at
                         FROM cameras c LEFT JOIN rec_roster r ON r.camera_id = c.id ORDER BY c.line_id, c.id""")
        return _rows(cur)


def set_roster(db, camera_id: str, *, enabled: bool, actor: str, note: str | None = None) -> dict:
    with db.cursor() as cur:
        cur.execute("SELECT 1 FROM cameras WHERE id = %s", (camera_id,))
        if cur.fetchone() is None:
            raise LookupError(f"沒有相機 {camera_id}")
        cur.execute("""INSERT INTO rec_roster (camera_id, enabled, note, updated_by, updated_at)
                       VALUES (%s,%s,%s,%s,now())
                       ON CONFLICT (camera_id) DO UPDATE SET enabled = EXCLUDED.enabled,
                           note = coalesce(EXCLUDED.note, rec_roster.note), updated_by = EXCLUDED.updated_by,
                           updated_at = now()""", (camera_id, enabled, note, actor))
    return next(r for r in roster(db) if r["camera_id"] == camera_id)


def roster_cameras(db) -> list[str]:
    """nas-recorder 用：名單上要錄、而且相機本身是啟用的。"""
    with db.cursor() as cur:
        cur.execute("SELECT r.camera_id FROM rec_roster r JOIN cameras c ON c.id = r.camera_id"
                    " WHERE r.enabled AND c.enabled ORDER BY r.camera_id")
        return [r[0] for r in cur.fetchall()]


def daily_gb(camera_id: str, *, now: float | None = None) -> float | None:
    """最近 24 小時這台寫了多少 GB（用檔案修改時間算，不開檔）。慢（要列 NAS 目錄），只給背景用 —— 見 DailyUsage。"""
    now = now or time.time()
    try:
        files = _files(camera_id)
    except (NasUnavailable, ValueError):
        return None
    if not files:
        return None
    since = datetime.fromtimestamp(now - 25 * 3600, LOCAL_TZ)
    total = 0
    for p, start in files:
        if start is not None and start < since:
            continue                    # 檔名開始時間早於 25 小時前：不可能是最近 24 小時寫的
        st = p.stat()
        if now - st.st_mtime <= 86400:
            total += st.st_size
    return round(total / 1e9, 2)


class DailyUsage:
    """名單 API 的 daily_gb 由背景每 refresh_s 秒算一次；API 只讀記憶體，永遠不在請求當下碰 NAS。

    ⚠️ 為什麼（2026-11-03）：慢的是列目錄 —— SMB 列一個 3,000 檔的目錄 0.8～1.2 秒，全部相機 4 秒以上，
       新 Pod 快取冷的第一次 8.7 秒，前端逾時（「推論 API 無回應」）。stat 反而是 0 秒（有快取）。
       請求時快取（5 分鐘過期）也不行：過期後第一個請求照樣要等。只是估容量用，晚 5 分鐘無妨。
    第一輪還沒算完前回 None（前端顯示「計算中」即可）。
    """

    def __init__(self, cameras_fn, *, refresh_s: float | None = None, compute=None) -> None:
        self.cameras_fn = cameras_fn
        self.refresh_s = refresh_s if refresh_s is not None else float(os.environ.get("REC_DAILY_REFRESH_S", "300"))
        self.compute = compute or daily_gb
        self.values: dict[str, float | None] = {}
        self.updated_at: float | None = None
        self._lock = threading.Lock()
        self._started = False

    def refresh(self) -> None:
        vals = {}
        for cam in self.cameras_fn():
            try:
                vals[cam] = self.compute(cam)
            except Exception:                  # noqa: BLE001 —— 一台壞了不影響其他台
                logger.exception("daily_gb %s 算不出來", cam)
                vals[cam] = None
        self.values, self.updated_at = vals, time.time()

    def start(self) -> None:
        with self._lock:
            if self._started or self.refresh_s <= 0:
                return
            self._started = True

        def run():
            while True:
                try:
                    self.refresh()
                    time.sleep(self.refresh_s)
                except Exception:              # noqa: BLE001
                    # 開機時 DB 還沒接上（state["db"] 晚於路由註冊才放進去）、NAS 暫時掛了 → 10 秒後重試，
                    # 不要乾等一整輪（2026-11-03 上線第一輪 KeyError: 'db'，前 5 分鐘全是 None）
                    logger.warning("daily_gb 背景計算失敗，10 秒後重試", exc_info=True)
                    time.sleep(min(10.0, self.refresh_s))
        threading.Thread(target=run, name="rec-daily-usage", daemon=True).start()

    def get(self, camera_id: str) -> float | None:
        return self.values.get(camera_id)
