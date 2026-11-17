"""影片 → 幀。上傳的影片、NAS 24h 錄影共用（固定間隔 sample；挑圖 edgevision/media/select.py 也用這裡的解碼）。

用 PyAV 依【媒體時間】取樣（每 interval 秒一張），不是每 N 幀：相機掉幀、可變幀率時間隔才會對。
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterator
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)

# NAS 24h 錄影檔名＝分段開始的本地時間（edgevision/recorder/prebuffer.CONTINUOUS_PATTERN）
_NAS_RE = re.compile(r"(\d{8})_(\d{6})\.mp4$")
LOCAL_TZ = ZoneInfo("Asia/Taipei")


@dataclass(slots=True)
class Frame:
    offset_s: float
    jpeg: bytes
    width: int
    height: int


def probe(path: Path) -> tuple[float | None, tuple[int, int] | None]:
    import av

    with av.open(str(path)) as c:
        st = c.streams.video[0]
        dur = float(c.duration / 1e6) if c.duration else (
            float(st.duration * st.time_base) if st.duration else None)
        return dur, (st.codec_context.width, st.codec_context.height)


def _t(fr, st) -> float:
    return float(fr.pts * st.time_base) - (float(st.start_time * st.time_base) if st.start_time else 0.0)


def keyframe_interval(path: Path, *, probe_s: float = 60.0) -> float | None:
    """只 demux（不解碼）前 probe_s 秒，回關鍵幀間隔中位數（秒）。"""
    import av

    with av.open(str(path)) as c:
        st = c.streams.video[0]
        ts = []
        for pkt in c.demux(st):
            if pkt.pts is None:
                continue
            t = float(pkt.pts * st.time_base)
            if pkt.is_keyframe:
                ts.append(t)
            if ts and t - ts[0] > probe_s:
                break
    if len(ts) < 3:
        return None
    gaps = sorted(b - a for a, b in zip(ts, ts[1:]))
    return gaps[len(gaps) // 2]


def decode_every(path: Path, *, interval_s: float, start_s: float = 0.0, end_s: float | None = None):
    """依媒體時間每 interval_s 秒產出一張 BGR 影像 (t, ndarray)。

    關鍵幀間隔 ≤ interval_s 時只解關鍵幀（HEVC 1080p 全解很貴：12 小時 10fps 是 43 萬幀）；
    否則全解、照時間挑。
    """
    import av

    interval_s = max(0.1, float(interval_s))
    gop = keyframe_interval(path)
    key_only = gop is not None and gop <= interval_s + 0.05
    tol = gop / 2 if key_only else 0.0      # 只解關鍵幀時，取最接近排程時間的那一張
    next_t = start_s
    with av.open(str(path)) as c:
        st = c.streams.video[0]
        st.thread_type = "AUTO"
        if key_only:
            st.codec_context.skip_frame = "NONKEY"
        if start_s > 1:
            c.seek(int(start_s / st.time_base), stream=st, any_frame=False, backward=True)
        for fr in c.decode(st):
            if fr.pts is None:
                continue
            t = _t(fr, st)
            if end_s is not None and t > end_s:
                break
            if t + 1e-6 < next_t - tol:
                continue
            yield round(t, 3), fr.to_ndarray(format="bgr24")
            next_t += interval_s                 # 照排程格點走，不累積漂移
            while next_t - tol <= t + 1e-6:
                next_t += interval_s


def decode_at(path: Path, offsets: list[float], *, gap_reseek_s: float = 5.0):
    """只解碼指定時間點（遞增）。相鄰兩點差超過 gap_reseek_s 才重新 seek，否則往下解。回 (offset, ndarray)。"""
    import av

    if not offsets:
        return
    with av.open(str(path)) as c:
        st = c.streams.video[0]
        st.thread_type = "AUTO"
        i, n = 0, len(offsets)
        while i < n:
            target = offsets[i]
            c.seek(int(max(0.0, target - 0.5) / st.time_base), stream=st, any_frame=False, backward=True)
            for fr in c.decode(st):
                if fr.pts is None:
                    continue
                t = _t(fr, st)
                if t + 0.02 < offsets[i]:
                    continue
                yield offsets[i], fr.to_ndarray(format="bgr24")
                i += 1
                if i >= n or offsets[i] - t > gap_reseek_s:
                    break
            else:
                logger.warning("%s 讀到結尾仍缺 %d 個時間點", path.name, n - i)
                return


def sample(path: Path, *, interval_s: float, start_s: float = 0.0, end_s: float | None = None,
           max_frames: int = 2000, quality: int = 95) -> Iterator[Frame]:
    """固定間隔取樣（strategy=interval）：依媒體時間每 interval_s 秒取一幀（JPEG）。start_s / end_s 是相對影片開頭的秒數。"""
    import cv2

    n = 0
    for t, img in decode_every(path, interval_s=interval_s, start_s=start_s, end_s=end_s):
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not ok:
            continue
        yield Frame(offset_s=t, jpeg=buf.tobytes(), width=img.shape[1], height=img.shape[0])
        n += 1
        if n >= max_frames:
            logger.warning("達到單次上限 %d 張，停止取樣", max_frames)
            return


def nas_segment_start(path: Path) -> datetime | None:
    m = _NAS_RE.search(path.name)
    if not m:
        return None
    return datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S").replace(tzinfo=LOCAL_TZ)


def nas_segments(root: Path, camera: str, start: datetime, end: datetime,
                 *, segment_max_s: int = 3600) -> list[tuple[Path, datetime]]:
    """找出與 [start, end] 重疊的 NAS 錄影段（依檔名時間）。目錄：<root>/<camera>/YYYYmmdd_HHMMSS.mp4"""
    d = root / camera
    if not d.is_dir():
        raise FileNotFoundError(f"NAS 沒有 {camera} 的錄影目錄（{d}）")
    segs = sorted((p, t) for p in d.glob("*.mp4") if (t := nas_segment_start(p)))
    out = []
    for i, (p, t0) in enumerate(segs):
        t1 = segs[i + 1][1] if i + 1 < len(segs) else t0 + timedelta(seconds=segment_max_s)
        if t1 > start and t0 < end:
            out.append((p, t0))
    return out


def grab_frame(src: str | Path, at_s: float, *, timeout_s: float = 20.0):
    """取 at_s 秒（相對影片開頭）那一幀的 BGR 影像；影片比 at_s 短就回最後一幀，一幀都解不出來回 None。

    src 可以是本地路徑或 http(s) 網址（MinIO 預簽）：PyAV 用 Range 讀，不必整支下載；moov 在檔尾也能 seek。
    """
    import av

    last = None
    with av.open(str(src), timeout=timeout_s) as c:
        st = c.streams.video[0]
        st.thread_type = "AUTO"
        if at_s > 0.5:
            c.seek(int((at_s - 0.5) / st.time_base), stream=st, any_frame=False, backward=True)
        for fr in c.decode(st):
            if fr.pts is None:
                continue
            last = fr
            if _t(fr, st) + 0.02 >= at_s:
                break
    return None if last is None else last.to_ndarray(format="bgr24")
