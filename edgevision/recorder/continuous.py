"""24 小時連續錄影 → NAS。

⭐ 它跟事件錄影（prebuffer.py）的關係

    兩者都是「從 go2rtc 拉流、-c copy、-f segment、掛了自動重啟」，
    所以共用 `BufferSpec` 與 `build_command`（用 segment_wrap=0 切換成連續模式）。
    這裡多的是連續錄影【特有】的四件事：

      ① 輸出停滯看門狗 —— ffmpeg 不會告訴你它卡住了
      ② 重連退避 —— 區分「跑很久才斷」與「一啟動就掛」
      ③ 保留天數清理 —— 沒有 -segment_wrap 幫忙覆寫了
      ④ 掛載檢查 —— 寫到主機硬碟而不自知是最貴的錯

⚠️ 為什麼錄【乾淨】串流，跟事件影片相反

    事件影片給人看（要框），這個給模型用（不能有框）。
    舊系統把框畫進警報截圖，導致那些畫面 100% 無法回收成訓練資料 ——

⚠️ 保留天數預設 0 = 不刪，寫到滿

    與舊系統一致。唯一的保護是空間下限：低於 REC_MIN_FREE_GB
    就不再開新段。那不是清理，只是不讓 NAS 被寫到 100% ——
    SMB 掛載寫滿會整個卡死，連帶拖垮其他在用這個掛載點的東西。

出處：舊系統錄影程式的實測經驗。
"""

from __future__ import annotations

import logging
import os
import signal
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from edgevision.recorder.prebuffer import BufferSpec, build_command

logger = logging.getLogger(__name__)

# ── 這些數字都是實測來的，改之前先看註解 ──────────────────

# 輸出停滯多久算「卡住了」。
# ⚠️ 不要往下調到 30 秒 —— 見 ContinuousRecorder._supervise 的說明。
DEFAULT_STALL_SECONDS = 90.0

# 曾經穩定跑超過這麼久，就當作「短暫斷線」而不是「設定壞了」
STABLE_RUN_SECONDS = 20.0

# 重連延遲：短暫斷線用最小值，一啟動就掛則指數退避到最大值
MIN_RETRY_SECONDS = 2.0
MAX_RETRY_SECONDS = 15.0

# 停滯被判死時，先 SIGTERM 讓 ffmpeg 把當前段的 moov 寫完，等這麼久還不走才 SIGKILL。
# 2026-11-03：原本直接 SIGKILL，每次停滯就留一個打不開的段（cam137 14:39 那段讓影片測試整個失敗）。
STALL_TERM_SECONDS = 20.0


def terminate(proc: subprocess.Popen, timeout: float) -> bool:
    """SIGTERM → 等 timeout 秒 → 還沒走才 SIGKILL。回傳是否乾淨收尾（沒用到 SIGKILL）。

    連續錄影的段落 moov 在檔尾：被 SIGKILL 的那一段永遠打不開。
    """
    if proc.poll() is not None:
        return True
    proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=timeout)
        return True
    except subprocess.TimeoutExpired:
        proc.kill()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass
        return False


@dataclass(frozen=True, slots=True)
class ContinuousSpec:
    """連續錄影的設定。相機相關的部分在 BufferSpec 裡。"""

    segment_seconds: int = 600
    retention_days: int = 0          # 0 = 不刪（與舊系統一致）
    min_free_gb: float = 20.0
    stall_seconds: float = DEFAULT_STALL_SECONDS


def dir_size(path: Path) -> int:
    """目錄下所有 mp4 的總位元組數。看門狗靠它判斷「有沒有在長」。"""
    try:
        return sum(p.stat().st_size for p in path.glob("*.mp4"))
    except OSError:
        return 0


def is_mounted(path: Path) -> bool:
    """這個路徑是不是一個掛載點。

    ⚠️ 為什麼要檢查：bind-mount 是【容器啟動當下】綁定的。
       先起容器、後掛 NAS 的話，容器看到的是主機上那個空目錄，
       然後很開心地把 24 小時的錄影寫進主機硬碟 —— 而且不會有任何錯誤，
       直到主機磁碟爆掉。（舊系統錄影程式特別警告過。）
    """
    try:
        return os.path.ismount(path) or _is_network_fs(path)
    except OSError:
        return False


def _is_network_fs(path: Path) -> bool:
    """容器裡 bind-mount 進來的 NAS 不會被 ismount() 認出來
    （它在容器的 mount namespace 裡不是掛載點的邊界），
    所以另外看檔案系統型別是不是網路檔案系統。
    """
    try:
        out = subprocess.run(["stat", "-f", "-c", "%T", str(path)],
                             capture_output=True, text=True, timeout=5)
        return out.stdout.strip().lower() in {"smb2", "cifs", "nfs", "fuseblk"}
    except Exception:  # noqa: BLE001 —— stat 不在就當作測不出來
        return False


def cleanup_old(out_dir: Path, *, retention_days: int, now: float | None = None) -> tuple[int, int]:
    """刪掉超過保留天數的檔案。回傳（刪幾個, 釋放幾 bytes）。

    retention_days = 0 代表不刪 —— 直接回 (0, 0)，不要誤刪。
    """
    if retention_days <= 0:
        return 0, 0
    deadline = (now or time.time()) - retention_days * 86400
    n = freed = 0
    for p in out_dir.glob("*.mp4"):
        try:
            st = p.stat()
            if st.st_mtime < deadline:
                p.unlink()
                n += 1
                freed += st.st_size
        except OSError:
            logger.debug("刪除失敗（略過）%s", p, exc_info=True)
    return n, freed


class ContinuousRecorder:
    """單台相機的 24h 連續錄影。跑一個 ffmpeg 子行程，掛了或卡住都會重來。

    ⚠️ 跟 PreRecordBuffer 最大的差別是【看門狗】。
       上游斷線時 ffmpeg 不一定會退出 —— 它可能就卡在那裡不 crash、不產出，
       只看 proc.poll() 會永遠等不到。舊系統實測有過單台 68 分鐘的無聲缺口。
    """

    def __init__(self, buf: BufferSpec, cfg: ContinuousSpec,
                 *, on_blocked=None) -> None:
        self.buf = buf
        self.cfg = cfg
        self._on_blocked = on_blocked      # 空間不足時回報用
        self._proc: subprocess.Popen | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.restarts = 0
        self.stalls = 0
        self.blocked = 0
        self.started_at = 0.0
        self.last_progress = 0.0           # 上次看到輸出變化的時間；0 = 還沒寫出東西

    def start(self) -> None:
        self.buf.out_dir.mkdir(parents=True, exist_ok=True)
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._supervise, name=f"nasrec-{self.buf.camera_id}", daemon=True)
        self._thread.start()

    # ── 主迴圈 ────────────────────────────────────────────

    def _supervise(self) -> None:
        delay = MIN_RETRY_SECONDS
        while not self._stop.is_set():
            # ① 空間不足就不要開新的 ffmpeg。
            #    NAS 寫到 100% 會讓 SMB 掛載整個卡死，比少錄一段嚴重得多。
            if not self._space_ok():
                self.blocked += 1
                self._stop.wait(60)
                continue

            # ② 順手清掉過期的（retention_days=0 時這是 no-op）
            n, freed = cleanup_old(self.buf.out_dir, retention_days=self.cfg.retention_days)
            if n:
                logger.info("[%s] 清掉 %d 個過期片段（%.1f GB）",
                            self.buf.camera_id, n, freed / 1e9)

            ran = self._run_once()

            if self._stop.is_set():
                return
            self.restarts += 1

            # ③ 退避：跑很久才斷 = 相機端的例行斷線（每 30 分鐘一次，
            #    ，立刻重連把缺口縮到最小。
            #    一啟動就掛 = 設定或網路有問題，退避避免洗版。
            if ran >= STABLE_RUN_SECONDS:
                delay = MIN_RETRY_SECONDS
            else:
                delay = min(delay * 2, MAX_RETRY_SECONDS)
            self._stop.wait(delay)

    def _space_ok(self) -> bool:
        try:
            free_gb = os.statvfs(self.buf.out_dir).f_bavail * \
                os.statvfs(self.buf.out_dir).f_frsize / 1e9
        except OSError:
            logger.exception("[%s] 無法檢查空間", self.buf.camera_id)
            return False
        if free_gb < self.cfg.min_free_gb:
            logger.warning("[%s] 可用空間 %.1f GB < %.1f GB —— 暫停錄影，60 秒後重試",
                           self.buf.camera_id, free_gb, self.cfg.min_free_gb)
            if self._on_blocked:
                self._on_blocked(self.buf.camera_id, free_gb)
            return False
        return True

    def _run_once(self) -> float:
        """跑一次 ffmpeg，直到它結束或被看門狗殺掉。回傳實際跑了幾秒。"""
        cmd = build_command(self.buf)
        logger.info("[%s] 連續錄影啟動（%d 秒一段）→ %s",
                    self.buf.camera_id, self.buf.segment_seconds, self.buf.out_dir)
        self.started_at = began = time.time()
        try:
            self._proc = subprocess.Popen(
                cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        except Exception:
            logger.exception("[%s] 無法啟動 ffmpeg", self.buf.camera_id)
            return 0.0

        stalled = self._watch(self._proc)
        ran = time.time() - began

        err = ""
        try:
            _, err = self._proc.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            self._proc.kill()

        if self._stop.is_set():
            return ran
        tail = "\n".join((err or "").strip().splitlines()[-3:])
        logger.warning("[%s] 錄影中斷（跑了 %.0f 秒，第 %d 次重啟）%s%s",
                       self.buf.camera_id, ran, self.restarts + 1,
                       "【輸出停滯】" if stalled else "",
                       f"：{tail}" if tail else "")
        return ran

    def _watch(self, proc: subprocess.Popen) -> bool:
        """盯著 ffmpeg，卡住就殺掉。回傳「是不是因為停滯被殺的」。

        ⚠️ 門檻是 90 秒，【不要往下調到 30 秒】。

           上游斷線時 ffmpeg 自己約 27~30 秒會退出。門檻設 30 秒會跟它賽跑，
           搶在它前面把行程砍掉 —— 於是一次乾淨的重連變成一個沒有 moov 的壞檔。
           舊系統就是這樣一天生出 58 個打不開的檔。
           （出處：舊系統錄影程式的 record_loop 註解）
        """
        last_size = dir_size(self.buf.out_dir)
        last_grow = time.time()
        grace = self.cfg.stall_seconds * 2      # 起步階段還沒開檔，不算停滯

        while proc.poll() is None:
            if self._stop.wait(5):
                return False
            size = dir_size(self.buf.out_dir)
            # 目錄變小也算活著 —— 那是清理在刪舊檔，不是停滯
            if size != last_size:
                last_size, last_grow = size, time.time()
                self.last_progress = last_grow
                continue
            idle = time.time() - last_grow
            if idle < self.cfg.stall_seconds:
                continue
            # 還在起步（沒開過檔）就多給一點時間
            if last_size == 0 and time.time() - self.started_at < grace:
                continue
            self.stalls += 1
            logger.error("[%s] 輸出停滯 %.0f 秒 —— 強制重連", self.buf.camera_id, idle)
            if not terminate(proc, STALL_TERM_SECONDS):
                logger.warning("[%s] ffmpeg 未在 %.0f 秒內收尾，已強制結束（這一段可能打不開）",
                               self.buf.camera_id, STALL_TERM_SECONDS)
            return True
        return False

    # ── 停止 ──────────────────────────────────────────────

    def stop(self, timeout: float = 35.0) -> None:
        """停止。

        ⚠️ SIGTERM 不是 SIGKILL，而且要等夠久。

           連續錄影的段落是 moov-at-end（我們刻意不用 +faststart，見 build_command）。
           ffmpeg 收到 SIGTERM 要把當前段的 moov 寫完才算收尾 ——
           8 支同時在 CIFS 上做這件事，docker 預設的 10 秒寬限期不夠，
           被 SIGKILL 的那一段就是永遠打不開的檔。
           compose 那邊要配合設 stop_grace_period: 40s。
        """
        self._stop.set()
        p = self._proc
        if p and not terminate(p, timeout):
            logger.warning("[%s] 未在 %.0f 秒內收尾，強制結束（最後一段可能損毀）",
                           self.buf.camera_id, timeout)
        if self._thread:
            self._thread.join(timeout=5)

    @property
    def alive(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def health(self) -> dict:
        files = sorted(self.buf.out_dir.glob("*.mp4"), key=lambda p: p.stat().st_mtime)
        newest = files[-1] if files else None
        return {
            "camera_id": self.buf.camera_id,
            "alive": self.alive,
            "restarts": self.restarts,
            "stalls": self.stalls,
            "blocked": self.blocked,
            "files": len(files),
            "disk_gb": round(sum(p.stat().st_size for p in files) / 1e9, 2),
            "newest": newest.name if newest else None,
            "newest_age_s": round(time.time() - newest.stat().st_mtime) if newest else None,
        }
