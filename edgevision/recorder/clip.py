"""事件片段：NG 發生時，把「前 N 秒 + 後 M 秒」剪出來存進 MinIO。

流程
    NG 事件 → 從預錄緩衝撈前段 → 等 M 秒讓後段錄完
            → concat → 依實際時間裁切 → 上傳 MinIO → 回傳 s3:// URI

    ⭐ 回傳的 URI 【必須有人接住並寫回 DB】（run_recorder._make_and_attach 在做）。
       沒寫回去的話影片就是孤兒檔案：躺在 MinIO 裡，沒有任何紀錄指向它，
       前端查到 NG 也播不到。這個洞真的發生過，產出 3312 個孤兒片段。

⚠️ 這裡【沒有】冷卻，而且不要加回來

    去重是判定層的事，不是錄影層的事。zone_intrusion 的 station 內建 streak +
    冷卻，runner 發到 stream 的每個 NG 已經是去重過的獨立事件
    （實測 cam63 相鄰 NG 間隔 134~557 秒，沒有一個落在 30 秒內）。

    recorder 曾經自己再疊一層 30 秒冷卻。那是「同一件事擋兩次」——
    而且第二道閘門擋掉的事件【不留任何紀錄】，於是出現「DB 有 NG、卻沒影片」：
    前端查到異常、點進去是空的。舊系統的註解講的
    正是這個坑（他們的修法是只留一道閘門）。2026-09-24 拿掉。

    ⭐ 結構保證：進到 stream 的每個 NG 事件都會有一支影片，跟 DB 一對一。
       要防灌爆，去判定層調 streak / 冷卻，別在這裡擋。

⚠️ 錄的是【有框】的那條（<cam>_marked）

    事件影片給【人】看：事後回放「當時發生什麼事」，框和 mask 燒死在畫面裡
    正是要的。要乾淨像素做訓練是另一回事 —— 那是事件截圖與 24h 錄影的用途
    。

    曾經想「錄乾淨版、之後再疊框」：做不到。side-car JSON 只有觸發那一幀的
    判定，沒有逐幀座標，PG 也不存 mask 多邊形。
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from edgevision.recorder.prebuffer import completed_segments

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ClipSpec:
    """事件片段的時間範圍。預設對齊舊系統（前 8 後 5）。"""

    pre_seconds: float = 8.0
    post_seconds: float = 5.0

    @property
    def total(self) -> float:
        return self.pre_seconds + self.post_seconds


def concat_segments(segments: list[Path], out: Path) -> bool:
    """把多個分段接成一個檔。

    ⚠️ 用 concat demuxer（-f concat）不是 concat protocol：
       後者只對 MPEG-TS 有效，對 mp4 會產出只能播第一段的檔案。

    ⚠️ -c copy：這些段本來就同一組編碼參數，不需要轉碼。
    """
    if not segments:
        return False
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False,
                                     encoding="utf-8") as f:
        for p in segments:
            # concat demuxer 的路徑要跳脫單引號
            f.write(f"file '{str(p.resolve()).replace(chr(39), chr(39) * 3)}'\n")
        listfile = f.name
    try:
        r = subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
             "-f", "concat", "-safe", "0", "-i", listfile,
             "-c", "copy", "-movflags", "+faststart", str(out)],
            capture_output=True, text=True, timeout=120)
        if r.returncode != 0:
            logger.warning("concat 失敗：%s", (r.stderr or "")[-300:])
            return False
        return out.exists() and out.stat().st_size > 0
    except subprocess.TimeoutExpired:
        logger.warning("concat 逾時（%d 段）", len(segments))
        return False
    finally:
        Path(listfile).unlink(missing_ok=True)


def _probe_nb_frames(path: Path) -> int | None:
    """影片有幾個影格。判斷「剪出來是不是空的」用這個，不能用檔案大小。"""
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v",
             "-count_packets", "-show_entries", "stream=nb_read_packets",
             "-of", "csv=p=0", str(path)],
            capture_output=True, text=True, timeout=30)
        return int((r.stdout or "").strip() or 0)
    except Exception:
        return None


# 編碼器順序：先 GPU（NVENC）、失敗退回 CPU（libx264）—— 同舊系統 alert_event_manager。
# GPU 失敗（沒分配 GPU、或消費級卡 NVENC 路數被 runner 用滿）後 NVENC_RETRY_S 秒內直接用 CPU，
# 不要每支片段都先白試一次。CLIP_ENCODER=x264 / nvenc 可強制。
ENCODERS = {
    "nvenc": ["-c:v", "h264_nvenc", "-preset", "p4", "-rc", "vbr", "-cq", "26"],
    "x264": ["-c:v", "libx264", "-preset", "ultrafast", "-crf", "23"],
}
NVENC_RETRY_S = 600.0
_nvenc_down_until = 0.0


def encoder_order(mode: str | None = None, now: float | None = None) -> list[str]:
    """這次要依序嘗試的編碼器。純函式（讀模組狀態），可測。"""
    mode = (mode or os.environ.get("CLIP_ENCODER", "auto")).lower()
    if mode in ("nvenc", "x264"):
        return [mode]
    now = time.time() if now is None else now
    return ["x264"] if now < _nvenc_down_until else ["nvenc", "x264"]


def trim(src: Path, out: Path, *, start: float, duration: float) -> bool:
    """從 src 的第 start 秒起剪 duration 秒，【重新編碼】（先 GPU、不行退回 CPU）。

    ⚠️ 不能 -c copy：串流複製只能從關鍵幀開始。<cam>_marked 是 runner 燒框後推回來的，
       幀率跟著推論負載浮動，實測關鍵幀可能 10 秒才一個 —— 起點落在兩個關鍵幀之間，
       copy 剪出來是空檔或只剩一兩秒（2026-10-02 測試叢集約三成 NG 因此沒有影片）。
       重新編碼起點精準到幀。
    ⚠️ -ss 放在 -i 前面：轉碼時輸入端 seek 是精準的，而且不用從頭解碼。
    """
    global _nvenc_down_until
    for enc in encoder_order():
        try:
            r = subprocess.run(
                ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                 "-ss", f"{max(0.0, start):.3f}", "-i", str(src), "-t", f"{duration:.3f}",
                 "-an", *ENCODERS[enc], "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)],
                capture_output=True, text=True, timeout=180)
        except subprocess.TimeoutExpired:
            logger.warning("裁切逾時（%s）", enc)
            continue
        if r.returncode != 0:
            if enc == "nvenc":
                _nvenc_down_until = time.time() + NVENC_RETRY_S
                logger.warning("GPU 編碼失敗，%.0f 分鐘內改用 CPU：%s",
                               NVENC_RETRY_S / 60, (r.stderr or "")[-200:].strip())
                continue
            logger.warning("裁切失敗：%s", (r.stderr or "")[-300:])
            return False
        if not out.exists():
            return False
        # size > 0 擋不住空片段（只有 ftyp+moov 的 261 B 檔）；用有沒有影格判斷才是真的
        if (_probe_nb_frames(out) or 0) < 1:
            logger.warning("裁切結果沒有影格（起點 %.3fs 可能超過素材長度）", start)
            return False
        return True
    return False


def wait_for_segments(cam_dir: Path, *, until: float, max_wait: float,
                      poll: float = 1.0) -> list[tuple[Path, float]]:
    """等到【已寫完】的分段涵蓋到 until 這個時刻，回傳 [(分段, mtime)]（時間順序）。

    ⚠️ 以前直接拿目錄裡全部分段：最新那段 ffmpeg 還在寫、打不開，
       事件後 5 秒正好在那段裡 → 片段缺後段，起點也容易超出素材。
    ⚠️ mtime 當下就抓：緩衝是輪替覆寫的，晚一點讀可能已經是另一段的時間。
    ⚠️「寫完沒」看 ffmpeg 的 -segment_list（completed_segments），不看 mtime 排第幾 ——
       2026-11-05 以前這裡會被 faststart 收尾的同秒 mtime 騙到，提早結束、片尾少一整段。
    超過 max_wait 就用現有的（寧可短一點，也要有片段）。
    """
    deadline = time.time() + max_wait
    while True:
        segs = [(p, p.stat().st_mtime) for p in completed_segments(cam_dir)]
        if (segs and segs[-1][1] >= until) or time.time() >= deadline:
            return segs
        time.sleep(poll)


def readable(segs: list[tuple[Path, float]]) -> list[tuple[Path, float]]:
    """只留讀得開的分段（保險：清單說寫完了，檔案仍可能壞，例如磁碟滿或被砍）。

    ⚠️「mtime 最新的那段 = 還在寫」不保證成立：上一段收尾時 faststart 會把 moov 搬到檔頭、
       整檔改寫，它的 mtime 反而比剛開始寫的下一段【晚】。結果還在寫的那段被當成已完成，
       concat 直接失敗（moov atom not found），那次 NG 就沒有影片（2026-10-02 實測）。
       逐段用 ffprobe 確認（剪輯只會用到兩三段，成本很小）。
    """
    return [(p, m) for p, m in segs if _probe_duration(p)]


@dataclass
class ClipStats:
    made: int = 0
    short: int = 0                 # 做出來但比規格短（素材不夠：斷流、緩衝被覆寫…）
    skipped_no_segments: int = 0
    errors: int = 0
    started_at: float = field(default_factory=time.time)

    def snapshot(self) -> dict:
        return {"made": self.made, "short": self.short,
                "no_segments": self.skipped_no_segments, "errors": self.errors,
                "uptime_s": round(time.time() - self.started_at)}


class ClipMaker:
    """把 NG 事件變成 MinIO 裡的一段影片 + 一份 side-car JSON。"""

    def __init__(self, *, objects, buffer_root: Path,
                 spec: ClipSpec | None = None,
                 bucket: str | None = None) -> None:
        self.objects = objects
        self.buffer_root = buffer_root
        self.spec = spec or ClipSpec()
        self.bucket = bucket or os.environ.get("CLIP_BUCKET", "inference-output")
        self.stats = ClipStats()

    def make(self, camera_id: str, event_ts: float, payload: dict) -> str | None:
        """做一段事件片段。回傳 MinIO URI，沒做成回 None。

        沒有冷卻檢查 —— 每個進來的 NG 都錄（理由見模組 docstring）。
        """
        cam_dir = self.buffer_root / camera_id
        if not cam_dir.is_dir():
            self.stats.skipped_no_segments += 1
            logger.warning("[%s] 沒有預錄緩衝目錄 —— 預錄服務沒在跑？", camera_id)
            return None

        # 等後段錄完：要有一個【寫完的】分段結束在 event_ts + post_seconds 之後。
        # 分段 10 秒一切，最多多等一段的時間。
        seg_s = float(os.environ.get("SEGMENT_SECONDS", 10))
        end_needed = event_ts + self.spec.post_seconds
        all_segs = wait_for_segments(cam_dir, until=end_needed,
                                     max_wait=max(0.0, end_needed - time.time()) + seg_s + 5)
        # 從涵蓋起點的那段開始取：結束時間（mtime）在起點之後的都要
        start_needed = event_ts - self.spec.pre_seconds
        picked = readable([(p, m) for p, m in all_segs if m > start_needed])
        if not picked:
            self.stats.skipped_no_segments += 1
            logger.warning("[%s] 緩衝區沒有可用分段", camera_id)
            return None
        segs = [p for p, _ in picked]
        first_end = picked[0][1]

        try:
            return self._build_and_upload(camera_id, event_ts, payload, segs,
                                          first_end=first_end)
        except Exception:
            self.stats.errors += 1
            logger.exception("[%s] 事件片段製作失敗", camera_id)
            return None

    def _build_and_upload(self, camera_id: str, event_ts: float,
                          payload: dict, segs: list[Path],
                          *, first_end: float) -> str | None:
        with tempfile.TemporaryDirectory(prefix="clip-") as td:
            tmp = Path(td)
            joined = tmp / "joined.mp4"
            if not concat_segments(segs, joined):
                self.stats.errors += 1
                return None

            # 第一段的起始時刻 = 它的 mtime 減去它自己的長度。
            # 用 mtime 是因為 ffmpeg 寫完一段才更新它 —— 那是「這段的結束時間」。
            # （first_end 由呼叫端在 concat 之前就抓好，理由見上面）
            seg_len = _probe_duration(segs[0]) or float(
                os.environ.get("SEGMENT_SECONDS", 10))
            timeline_start = first_end - seg_len
            offset = max(0.0, (event_ts - self.spec.pre_seconds) - timeline_start)

            # 夾住 offset：素材只有這麼長，起點超過就一定剪出空檔。
            # 寧可給「結尾往前 total 秒」這個次佳片段，也不要一個打不開的檔案。
            total_len = _probe_duration(joined) or (len(segs) * seg_len)
            if offset >= total_len:
                logger.warning(
                    "[%s] 起點 %.1fs 超過素材長度 %.1fs（緩衝可能已輪替覆寫），"
                    "改剪結尾", camera_id, offset, total_len)
                offset = max(0.0, total_len - self.spec.total)

            clip = tmp / "clip.mp4"
            if not trim(joined, clip, start=offset, duration=self.spec.total):
                self.stats.errors += 1
                return None
            # 長度不足照樣上傳（有比沒有好），但要記下來：短片率是看得到的指標，不用人工抽查
            got = _probe_duration(clip)
            if got is not None and got < self.spec.total - 1.0:
                self.stats.short += 1
                logger.warning("[%s] 事件片段只有 %.1fs（規格 %.0fs）：素材 %.1fs、起點 %.1fs、%d 段",
                               camera_id, got, self.spec.total, total_len, offset, len(segs))

            from datetime import datetime
            # 檔名用本地時間（同舊系統 datetime.now()）。之前用 UTC，檔名比牆鐘慢 8 小時，
            # 在 MinIO 看起來像「沒有新檔」。side-car JSON 的 event_ts 仍是 epoch，不受影響。
            dt = datetime.fromtimestamp(event_ts).astimezone()
            base = f"{camera_id}/{dt:%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:8]}"

            # ⚠️ 先上傳影片再上傳 JSON：JSON 是索引，有索引沒影片是死連結。
            with clip.open("rb") as fh:
                self.objects.put(bucket=self.bucket, key=f"{base}.mp4",
                                 data=fh, length=clip.stat().st_size,
                                 content_type="video/mp4")
            meta = json.dumps({
                "camera_id": camera_id, "event_ts": event_ts,
                "pre_seconds": self.spec.pre_seconds,
                "post_seconds": self.spec.post_seconds,
                "clip": f"{base}.mp4", **payload,
            }, ensure_ascii=False).encode()
            import io
            self.objects.put(bucket=self.bucket, key=f"{base}.json",
                             data=io.BytesIO(meta), length=len(meta),
                             content_type="application/json")

            self.stats.made += 1
            uri = f"s3://{self.bucket}/{base}.mp4"
            logger.info("[%s] 事件片段 %s（%.1fs，%d 段）", camera_id, uri,
                        self.spec.total, len(segs))
            return uri


def _probe_duration(path: Path) -> float | None:
    """問 ffprobe 這個檔多長。取不到回 None（呼叫端用 segment_seconds 當預設）。

    ⚠️ 要接 FileNotFoundError —— 映像裡可能只有 ffmpeg 沒有 ffprobe
       （實測 edgevision:mvp 就是這樣）。漏接的話整個片段製作會拋例外，
       而真正的原因（少一個執行檔）完全看不出來。
    """
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=nw=1:nk=1", str(path)],
            capture_output=True, text=True, timeout=10)
        return float(r.stdout.strip()) if r.returncode == 0 and r.stdout.strip() else None
    except (subprocess.TimeoutExpired, ValueError, FileNotFoundError, OSError):
        return None
