"""預錄緩衝：讓「事件前 N 秒」成為可能。

⭐ 問題

    事件是「發生了才知道」。等判定 NG，那一刻【之前】的畫面已經過去了 ——
    但最有價值的正是事件前那幾秒（工件怎麼被翻的）。

⭐ 解法：滾動 ring buffer（照抄舊系統，這個設計是對的）

    ffmpeg -f segment -segment_time 10 -segment_wrap 30 -c copy
           ↓
    seg000.mp4 ~ seg029.mp4  →  寫滿 30 個從頭覆寫（= 最近 5 分鐘）

    三個優點：
      · -c copy 不轉碼，CPU 幾乎零成本
      · -segment_wrap 由 ffmpeg 原地覆寫 —— 不需要我們寫清理邏輯，
        【沒有清理邏輯就沒有清理 bug】，這是最漂亮的地方
      · 空間有硬上限：30 個檔算得出來（720p H.264 約 56MB/相機），不會失控

⚠️ 從 go2rtc 拉，不直連相機

    每台相機全系統只開一條上游連線。直連相機的話，錄影 + 推論 + 前端
    會各開一條，相機的併發連線數很快就爆（舊系統踩過）。

⚠️ 預設拉的是 _marked（runner 燒好框與 mask 的那條），不是原始串流

    要哪條由 run_recorder.py 的 RECORD_SUFFIX 決定（預設 "_marked"）。
    曾經想錄原始串流「之後再疊框」—— 做不到：side-car JSON 只有觸發那一幀的判定，
    沒有逐幀座標，PG 也不存 mask 多邊形。存證的用途是「事後看發生什麼事」，
    框和 mask 燒死在畫面裡正是要的；要乾淨像素做訓練是資料集（LakeFS）的事。
    只有 runner 沒在推 _marked 時才退回原始串流（那段就沒有框）。
"""

from __future__ import annotations

import dataclasses

import logging
import os
import re
import shutil
import signal
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# ffmpeg 的分段檔名樣式。%03d 由 -segment_wrap 決定上限。
SEGMENT_PATTERN = "seg%03d.mp4"
_SEG_RE = re.compile(r"seg(\d+)\.mp4$")

# 環形緩衝的「已寫完清單」：ffmpeg 每關閉一段才把它寫進去（-segment_list），還在寫的那段不會出現。
# 重啟 ffmpeg 前把上一份改名成 PREV（新行程會從頭重寫清單，舊的那幾段不能就此看不到）。
SEGMENT_LIST = "segments.csv"
SEGMENT_LIST_PREV = "segments.prev.csv"

# 連續錄影（24h → NAS）的檔名樣式。ffmpeg -strftime 1 會把它展開成本地時間。
# ⚠️ 本地時間不是 UTC —— 檔名比牆鐘慢 8 小時會讓人以為錄影停了（踩過）。
#    容器要設 TZ=Asia/Taipei。
CONTINUOUS_PATTERN = "%Y%m%d_%H%M%S.mp4"
_CONTINUOUS_RE = re.compile(r"(\d{8})_(\d{6})\.mp4$")


@dataclass(frozen=True, slots=True)
class BufferSpec:
    """一台相機的錄影設定。**事件預錄與 24h 連續錄影共用這一份。**

    兩種模式由 segment_wrap 區分：

      · segment_wrap > 0 → **環形緩衝**（事件預錄）。
        segment_seconds × segment_wrap = 緩衝總長度，預設 10 × 30 = 300 秒。
        ffmpeg 寫滿從頭覆寫，檔名 seg000~seg029。

      · segment_wrap = 0 → **連續錄影**（24h → NAS）。
        不覆寫、不限量，檔名用牆鐘時間 YYYYMMDD_HHMMSS.mp4。
        清理另外做（見 edgevision/recorder/continuous.py），不靠 ffmpeg。

    ⚠️ 不要為了 24h 錄影另寫一份 ffmpeg 包裝。兩者要處理的事情幾乎一樣
       （go2rtc 拉流、-c copy、-f segment、斷線重啟），分兩份就會有一邊
       修了坑另一邊沒修 —— 舊系統的舊系統錄影程式就是那樣長到 500 行的。
    """

    camera_id: str
    rtsp_url: str
    out_dir: Path
    segment_seconds: int = 10
    segment_wrap: int = 30
    # ⭐ 媒體時間 = 牆鐘時間。連續錄影與事件預錄【都要】，理由見 build_command 的註解。
    wallclock_timestamps: bool = True

    @property
    def continuous(self) -> bool:
        """是不是連續錄影模式（不覆寫、檔名帶時間）。"""
        return self.segment_wrap <= 0

    @property
    def buffer_seconds(self) -> int:
        """環形緩衝的總長度。連續錄影模式沒有這個概念，回 0。"""
        return 0 if self.continuous else self.segment_seconds * self.segment_wrap


def build_command(spec: BufferSpec) -> list[str]:
    """組 ffmpeg 指令。純函式，可測。

    ⚠️ 幾個參數不是可有可無的：

      -rtsp_transport tcp   UDP 丟包會讓分段檔破圖，而破掉的 mp4 事後無法修
      -an                   相機有 PCM 音軌，存證用不到，而且它會拖慢 remux
      -c copy               不轉碼。轉碼的話 8 路就要吃掉好幾核
      -reset_timestamps 1   每段從 0 開始計時，否則 concat 出來的片段時間軸是錯的

    兩種模式的差異見下方註解。
    """
    pre: list[str] = [
        "ffmpeg", "-hide_banner", "-loglevel", "warning",
        "-rtsp_transport", "tcp",
    ]

    # ⭐ -use_wallclock_as_timestamps：只有連續錄影要，而且【必須】要。
    #
    # 這批相機開了智慧串流：靜態場景會把幀率壓到 1~2 fps。
    # -f segment 是按【媒體時間】切段的，-strftime 是按【牆鐘】命名的 ——
    # 用相機給的時戳，「10 分鐘的檔」實際會橫跨 77 分鐘：
    # 檔名出現大洞、回放變快轉、時間涵蓋率只剩 63~82%。
    #
    # 這行讓媒體時間恆等於牆鐘。⚠️ 不要同時加 +genpts，兩者會打架。
    # （出處：舊系統錄影程式的實測）
    #
    # 預錄緩衝【也要】（2026-10-02 修正，以前說不需要是錯的）：剪事件片段時用
    # 「段落 mtime − 段落長度」推算時間軸，段落長度是媒體時間。<cam>_marked 是 runner
    # 推回來的畫面，幀率跟著推論負載浮動（實測有的只有 3 fps）→ 媒體時間跟牆鐘對不上，
    # 起點算錯、剪出空檔 —— 測試叢集約三成 NG 因此沒有影片。
    if spec.wallclock_timestamps:
        pre += ["-use_wallclock_as_timestamps", "1"]

    cmd = pre + [
        "-i", spec.rtsp_url,
        "-an",
        "-c", "copy",
        "-f", "segment",
        "-segment_time", str(spec.segment_seconds),
        "-reset_timestamps", "1",
    ]

    if spec.continuous:
        # 連續錄影：不覆寫，檔名帶本地時間。
        #
        # ⚠️ 不要 -movflags +faststart。
        #    faststart 會在檔案寫完後把 moov 搬到檔頭 —— 等於整檔再改寫一次。
        #    在 CIFS（NAS）上那是很貴的動作，而連續錄影的段落沒人要 concat，
        #    moov 在檔尾完全可以播。
        #    （預錄緩衝相反：它的段落要被 concat，所以那邊要 faststart。）
        #
        # ⚠️ 這也是為什麼收尾需要 40 秒寬限期：moov 在檔尾，
        #    被 SIGKILL 打斷的那一段就是永遠打不開的檔。見 PreRecordBuffer.stop()。
        cmd += ["-strftime", "1", str(spec.out_dir / CONTINUOUS_PATTERN)]
    else:
        # 環形緩衝：ffmpeg 原地覆寫，不需要我們寫清理邏輯
        #（沒有清理邏輯就沒有清理 bug）。
        #
        # ⭐ -segment_list：哪幾段【已經寫完】由 ffmpeg 自己說，不要用 mtime 猜（見 completed_segments）。
        cmd += [
            "-segment_wrap", str(spec.segment_wrap),
            "-segment_list", str(spec.out_dir / SEGMENT_LIST),
            "-segment_list_type", "csv",
            # 只留 wrap-1 筆：檔名 30 個輪流覆寫，留滿 30 筆的話【正在重錄】那個檔名
            # 還掛在清單上（上一輪寫完時記的）→ 又被當成已寫完（2026-11-05 上線後實測）。
            "-segment_list_size", str(max(1, spec.segment_wrap - 1)),
            "-movflags", "+faststart",
            str(spec.out_dir / SEGMENT_PATTERN),
        ]
    return cmd


def rotate_segment_list(out_dir: Path) -> None:
    """（重新）啟動 ffmpeg 前呼叫：新行程會把清單從頭重寫，上一個行程寫完的那幾段要留在 PREV 裡才看得到。"""
    cur = out_dir / SEGMENT_LIST
    if cur.exists():
        cur.replace(out_dir / SEGMENT_LIST_PREV)


def _listed(path: Path) -> list[str]:
    """清單裡的分段檔名。ffmpeg 每次整份重寫，可能讀到寫一半的最後一行 → 只收完整的檔名。"""
    import csv
    try:
        with path.open(newline="", encoding="utf-8", errors="replace") as f:
            rows = list(csv.reader(f))
    except OSError:
        return []
    return [r[0] for r in rows if r and len(r) >= 3 and _SEG_RE.fullmatch(r[0])]


def list_segments(out_dir: Path) -> list[Path]:
    """回傳緩衝區內的分段，【照寫入時間排序】而不是檔名。

    ⚠️ 不能照檔名排：ring buffer 會繞回，seg029 之後是 seg000，
       照檔名排會把最新的那幾個排到最前面，concat 出來的影片時間軸是亂的。
    """
    segs = [p for p in out_dir.glob("seg*.mp4") if _SEG_RE.search(p.name)]
    return sorted(segs, key=lambda p: p.stat().st_mtime)


def completed_segments(out_dir: Path) -> list[Path]:
    """已經寫完的分段（照時間排序）。

    ⭐ 以 ffmpeg 的 -segment_list 為準：它關閉一段才把那段寫進清單。
    ⚠️ 以前用「mtime 最新的那段 = 還在寫」：不成立。faststart 收尾會把剛寫完那段整檔改寫，
       它的 mtime 跟正在寫的下一段同一秒甚至更晚（2026-11-05 正式 cam26 實測同秒）。
       結果剛寫完的那段被當成還在寫、等待提早結束，事件影片少掉最後一整段
       （正式 ALPHA 抽查約 17% 片段只有 3.6～12.3 秒）。
    ⚠️ 檔名輪流覆寫：清單上的檔名可能已經在重錄 → 清單只留 wrap-1 筆，且 mtime 比清單新的不算。
    上一個 ffmpeg 行程寫完的段落記在 PREV；之後被新行程覆寫過的（mtime 比 PREV 新）不算。
    兩份清單都沒有（舊版緩衝目錄）才退回舊判斷。
    """
    cur, prev = out_dir / SEGMENT_LIST, out_dir / SEGMENT_LIST_PREV
    if not cur.exists() and not prev.exists():
        return list_segments(out_dir)[:-1]
    done = set()
    if cur.exists():
        # ffmpeg 先關檔（含 faststart 改寫）才重寫清單 → 寫完的段 mtime 一定不晚於清單。
        # 比清單新的 = 已經被下一輪覆寫、正在錄（保險；主要靠清單只留 wrap-1 筆）。
        listed_at = cur.stat().st_mtime
        for name in _listed(cur):
            p = out_dir / name
            if p.exists() and p.stat().st_mtime <= listed_at:
                done.add(name)
    if prev.exists():
        cutoff = prev.stat().st_mtime + 1.0
        for name in _listed(prev):
            p = out_dir / name
            if name not in done and p.exists() and p.stat().st_mtime <= cutoff:
                done.add(name)
    return [p for p in list_segments(out_dir) if p.name in done]


def segments_covering(out_dir: Path, *, seconds_back: float,
                      now: float | None = None) -> list[Path]:
    """撈出「往回 seconds_back 秒」需要的分段，時間順序排好。

    多撈一段：事件發生在某一段的中間，只撈剛好的秒數會缺開頭。
    寧可多剪一點，concat 之後再依實際時間裁切。
    """
    now = now or time.time()
    deadline = now - seconds_back
    segs = list_segments(out_dir)
    keep: list[Path] = []
    for p in reversed(segs):            # 由新到舊
        keep.append(p)
        if p.stat().st_mtime <= deadline:
            break                       # 這一段已涵蓋起點，含它就夠
    return list(reversed(keep))


class PreRecordBuffer:
    """單台相機的預錄緩衝。跑一個 ffmpeg 子行程，掛了自動重啟。

    ⚠️ 為什麼要自己管重啟：相機會斷線（實測每 30 分鐘一次）。
       ffmpeg 遇到 RTSP 斷線會直接結束，
       沒有人重啟的話那台相機的緩衝就永遠空了 —— 而且【不會有任何錯誤】，
       只是事件發生時撈不到前段。
    """

    def __init__(self, spec: BufferSpec, *, restart_delay: float = 5.0,
                 pick_url=None, recheck_seconds: float = 120.0) -> None:
        """pick_url：每次（重新）啟動 ffmpeg 前呼叫，回傳 (網址, 是不是首選來源)。
        不給就固定用 spec.rtsp_url。用在「首選 <cam>_marked、沒畫面才退回原始串流」：
        以前只在服務啟動時判斷一次，那時 runner 剛好在重啟，這台之後錄的就永遠沒框。
        現在每次重連都重新判斷；退回原始串流期間每 recheck_seconds 秒檢查一次，首選恢復就切回去。
        """
        self.spec = spec
        self.restart_delay = restart_delay
        self.pick_url = pick_url
        self.recheck_seconds = recheck_seconds
        self.on_preferred = True
        self._proc: subprocess.Popen | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.restarts = 0
        self.started_at = 0.0

    def start(self) -> None:
        self.spec.out_dir.mkdir(parents=True, exist_ok=True)
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._supervise, name=f"prebuf-{self.spec.camera_id}", daemon=True)
        self._thread.start()

    def _supervise(self) -> None:
        while not self._stop.is_set():
            spec = self.spec
            if self.pick_url is not None:
                url, self.on_preferred = self.pick_url()
                spec = dataclasses.replace(spec, rtsp_url=url)
            cmd = build_command(spec)
            if not spec.continuous:
                rotate_segment_list(spec.out_dir)
            logger.info("[%s] 預錄啟動（%d 秒緩衝）%s", spec.camera_id, spec.buffer_seconds,
                        "" if self.on_preferred else " —— 首選來源沒畫面，暫用備用來源")
            self.started_at = time.time()
            try:
                self._proc = subprocess.Popen(
                    cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
                if not self.on_preferred:
                    self._watch_for_preferred()
                _, err = self._proc.communicate()
            except Exception:
                logger.exception("[%s] 預錄行程異常", self.spec.camera_id)
                err = ""
            if self._stop.is_set():
                return
            self.restarts += 1
            # 只印最後幾行 —— ffmpeg 斷線時會刷一大片同樣的訊息
            tail = "\n".join((err or "").strip().splitlines()[-3:])
            logger.warning("[%s] 預錄結束（第 %d 次重啟）%s",
                           self.spec.camera_id, self.restarts,
                           f"：{tail}" if tail else "")
            self._stop.wait(self.restart_delay)

    def _watch_for_preferred(self) -> None:
        """用備用來源錄的期間，定期看首選來源恢復了沒；恢復就讓 ffmpeg 收尾，迴圈會換回首選。"""
        while self._proc and self._proc.poll() is None and not self._stop.is_set():
            if self._stop.wait(self.recheck_seconds):
                return
            _url, preferred = self.pick_url()
            if preferred:
                logger.warning("[%s] 首選來源恢復，切回去（備用來源那幾段沒有框）", self.spec.camera_id)
                self._proc.send_signal(signal.SIGTERM)
                return

    def stop(self, timeout: float = 15.0) -> None:
        """停止。

        ⚠️ 用 SIGTERM 不用 SIGKILL：mp4 要寫 moov 才能打開，
           被 SIGKILL 的那一段就是永遠打不開的檔（舊系統在 CIFS 上踩過，
           8 支同時收尾超過 docker 預設的 10 秒寬限期）。
        """
        self._stop.set()
        p = self._proc
        if p and p.poll() is None:
            p.send_signal(signal.SIGTERM)
            try:
                p.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                logger.warning("[%s] 預錄未在 %.0f 秒內收尾，強制結束（最後一段可能損毀）",
                               self.spec.camera_id, timeout)
                p.kill()
        if self._thread:
            self._thread.join(timeout=5)

    @property
    def alive(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def health(self) -> dict:
        segs = list_segments(self.spec.out_dir)
        return {
            "camera_id": self.spec.camera_id,
            "alive": self.alive,
            "restarts": self.restarts,
            "segments": len(segs),
            "buffer_seconds": self.spec.buffer_seconds,
            "uptime_s": round(time.time() - self.started_at) if self.started_at else 0,
            "disk_mb": round(sum(p.stat().st_size for p in segs) / 1e6, 1),
        }


def check_space(path: Path, *, min_free_gb: float) -> tuple[bool, float]:
    """寫入前檢查空間。回傳 (夠不夠, 剩餘 GB)。

    ⚠️ 舊系統的推論服務沒有任何 disk_usage / statvfs 呼叫，
       唯一的容量 API 是純顯示用、不會擋錄影。這跟我們踩過的
       docker log 爆磁碟（698G）是同一類問題的不同面向。
    """
    try:
        free_gb = shutil.disk_usage(path).free / 1e9
    except OSError:
        logger.exception("無法檢查 %s 的空間", path)
        return False, 0.0
    return free_gb >= min_free_gb, round(free_gb, 1)


def buffer_dir(camera_id: str) -> Path:
    """預錄緩衝的位置。

    程式只讀 PRE_RECORD_BUFFER_DIR —— 位置由部署層決定：
    Compose 給本機磁碟、K8s 給 emptyDir 或 PVC，程式不必知道。
    """
    base = os.environ.get("PRE_RECORD_BUFFER_DIR")
    if not base:
        raise SystemExit("必須設定 PRE_RECORD_BUFFER_DIR（預錄緩衝的位置）")
    return Path(base) / camera_id
