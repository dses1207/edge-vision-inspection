"""啟動看門狗：管線起來了卻一幀都沒推論出來 → 回報 failed（附原因）並結束程序，讓控制端看得到。

⚠️ 為什麼（2026-10-14 正式叢集 ipcam_golf）：GPU 切片被吃光，新程序的 nvinferserver / 解碼器拿不到記憶體，
   管線卡住但不送 GStreamer ERROR —— 程序沒死、心跳照送 apply_state=ok（或連心跳都沒有），
   apply_error 一路是 None，前端只看到 cameras_running=0，分不出是相機壞還是我們壞。

判斷（啟動後 STARTUP_TIMEOUT_S 秒內，任一相機有幀過推論 probe 就算成功、看門狗結束）：
  ・影像有進 streammux、推論 probe 一幀都沒有        → 推論卡住（多半 GPU 記憶體不足）  → failed + 結束
  ・streammux 沒有影像、相機來源也沒報過錯           → 解碼卡住或相機無回應              → failed + 結束
  ・streammux 沒有影像、相機來源一直報錯（在重連）   → 相機問題，不是 runner 的錯      → 維持 ok，只記 log
結束程序（exit 3）而不是乾等：K8s 會重啟並累計次數（看得到），GPU 記憶體騰出來後自己恢復。

這個執行緒不靠 GLib 主迴圈 —— 主迴圈可能就是卡住的那個。
"""

from __future__ import annotations

import ctypes
import logging
import os
import threading
import time
from collections.abc import Callable

logger = logging.getLogger(__name__)

EXIT_STALLED = 3


def gpu_memory_mib() -> tuple[int, int] | None:
    """(剩餘, 總共) MiB —— 這個程序看得到的 GPU（MIG 切片就是切片的量）。

    用 primary context（DeepStream 用 runtime API，本來就建在它上面），不額外佔記憶體。
    MIG 下 nvidia-smi 看不到記憶體（權限不足），只能這樣問。查不到回 None。
    """
    try:
        cu = ctypes.CDLL("libcuda.so.1")
        if cu.cuInit(0) != 0:
            return None
        dev, ctx = ctypes.c_int(), ctypes.c_void_p()
        if cu.cuDeviceGet(ctypes.byref(dev), 0) != 0 or cu.cuDevicePrimaryCtxRetain(ctypes.byref(ctx), dev) != 0:
            return None
        try:
            if cu.cuCtxPushCurrent_v2(ctx) != 0:
                return None
            free, total = ctypes.c_size_t(), ctypes.c_size_t()
            rc = cu.cuMemGetInfo_v2(ctypes.byref(free), ctypes.byref(total))
            cu.cuCtxPopCurrent_v2(ctypes.byref(ctypes.c_void_p()))
            return (free.value >> 20, total.value >> 20) if rc == 0 else None
        finally:
            cu.cuDevicePrimaryCtxRelease_v2(dev)
    except (OSError, AttributeError):
        return None


def diagnose(stats: dict, timeout_s: float, gpu: tuple[int, int] | None) -> tuple[str, str | None]:
    """逾時當下的判斷。回 (verdict, 原因)：verdict = ok | camera | stalled。純函式，可測。"""
    if any(v > 0 for v in (stats.get("cam_frames") or {}).values()):
        return "ok", None
    mem = f"（GPU 剩 {gpu[0]} MiB / 共 {gpu[1]} MiB）" if gpu else "（GPU 記憶體查不到）"
    if stats.get("mux_batches", 0) > 0:
        return "stalled", (f"管線啟動 {timeout_s:.0f} 秒：影像有進來，但推論一幀都沒有輸出 —— "
                           f"nvinferserver 卡住，多半是 GPU 記憶體不足{mem}")
    if stats.get("src_err", 0) > 0:
        return "camera", f"管線啟動 {timeout_s:.0f} 秒沒有影像：相機來源一直報錯（在重連），相機或網路問題"
    return "stalled", (f"管線啟動 {timeout_s:.0f} 秒：沒有任何影像解碼出來，相機也沒報錯 —— "
                       f"解碼器卡住（多半是 GPU 記憶體不足{mem}）或相機連線沒回應")


class StartupWatchdog:
    def __init__(self, stats: dict, reporter, timeout_s: float, *,
                 clock: Callable[[], float] = time.monotonic,
                 gpu: Callable[[], tuple[int, int] | None] = gpu_memory_mib,
                 exit_fn: Callable[[int], None] = os._exit, poll_s: float = 2.0) -> None:
        self.stats, self.reporter, self.timeout_s = stats, reporter, timeout_s
        self.clock, self.gpu, self.exit_fn, self.poll_s = clock, gpu, exit_fn, poll_s
        self.t0 = clock()
        self._stop = threading.Event()

    def check(self) -> bool:
        """跑一次；回 True 表示看門狗任務結束（成功、相機問題或已結束程序）。"""
        if any(v > 0 for v in (self.stats.get("cam_frames") or {}).values()):
            if self.reporter is not None:
                self.reporter.set_state("ok")
            logger.info("啟動確認：%.0f 秒內有幀過推論", self.clock() - self.t0)
            return True
        if self.clock() - self.t0 < self.timeout_s:
            return False
        verdict, why = diagnose(self.stats, self.timeout_s, self.gpu())
        if verdict == "camera":
            logger.warning("%s", why)
            if self.reporter is not None:
                self.reporter.set_state("ok")      # 相機壞不是套用失敗；cameras_running=0 已經表達了
            return True
        logger.error("%s；結束程序讓 K8s 重啟", why)
        if self.reporter is not None:
            self.reporter.set_state("failed", why)
            self.reporter.send("failed", why)
        self.exit_fn(EXIT_STALLED)
        return True

    def start(self) -> None:
        if self.timeout_s <= 0:
            return
        def run():
            while not self._stop.wait(self.poll_s):
                if self.check():
                    return
        threading.Thread(target=run, name="startup-watchdog", daemon=True).start()

    def stop(self) -> None:
        self._stop.set()


class RunWatchdog:
    """執行中看門狗：跑起來之後推論停了（幀數不再增加）→ 回報 failed 並結束程序，讓 K8s 重啟。

    ⚠️ 為什麼：某個節點硬碟滿、go2rtc 與 redis 被趕走，七個 runner 同一秒
       推流失敗後整條管線卡住 —— 不再出 log、不送 ERROR，程序沒死、Pod 一直是 Running，K8s 不會重啟。
       節點恢復後也接不回去，長時間沒有推論、告警、存證，直到人工重啟。
       啟動看門狗只管「起來後有沒有第一幀」，管不到這種跑到一半卡住的。

    判斷（第一幀過推論後才開始看）：推論幀數 stall_s 秒沒增加 →
      ・這段期間相機來源有報錯（在重連）→ 相機或網路問題，不是 runner 的錯，繼續等
      ・相機沒報錯                       → 管線卡住 → failed + 結束（exit 3）
    只看 probe 的總幀數，不靠 GLib 主迴圈（卡住的可能就是它）。
    """

    def __init__(self, stats: dict, reporter, stall_s: float, *,
                 clock: Callable[[], float] = time.monotonic,
                 exit_fn: Callable[[int], None] = os._exit, poll_s: float = 5.0) -> None:
        self.stats, self.reporter, self.stall_s = stats, reporter, stall_s
        self.clock, self.exit_fn, self.poll_s = clock, exit_fn, poll_s
        self._last = None                   # (幀數, 相機錯誤數, 時間)：最後一次看到幀數增加的時候
        self._camera_logged = False
        self._stop = threading.Event()

    def check(self) -> bool:
        """跑一次；回 True 表示已結束程序。"""
        frames, src_err, now = self.stats.get("frames", 0), self.stats.get("src_err", 0), self.clock()
        if frames == 0:
            return False                    # 還沒有第一幀：啟動看門狗的事
        if self._last is None or frames != self._last[0]:
            self._last, self._camera_logged = (frames, src_err, now), False
            return False
        idle = now - self._last[2]
        if idle < self.stall_s:
            return False
        if src_err != self._last[1]:
            if not self._camera_logged:
                logger.warning("推論已 %.0f 秒沒有新幀，相機來源在報錯重連 —— 相機或網路問題，繼續等", idle)
                self._camera_logged = True
            return False
        why = f"執行中推論停了 {idle:.0f} 秒（幀數停在 {frames}），相機也沒報錯 —— 管線卡住"
        logger.error("%s；結束程序讓 K8s 重啟", why)
        if self.reporter is not None:
            self.reporter.set_state("failed", why)
            self.reporter.send("failed", why)
        self.exit_fn(EXIT_STALLED)
        return True

    def start(self) -> None:
        if self.stall_s <= 0:
            return
        def run():
            while not self._stop.wait(self.poll_s):
                if self.check():
                    return
        threading.Thread(target=run, name="run-watchdog", daemon=True).start()

    def stop(self) -> None:
        self._stop.set()


class CameraStallWatchdog:
    """單台相機沒有新幀 → 回報給呼叫端，把那一台重建（不動其他台）。

    ⚠️ 為什麼（2026-11-09 正式 ALPHA 5 台、工序W cam215）：相機定時斷線後 nvurisrcbin 自己重連，偶爾進入
       「連得上、畫面進不來、等 10 秒再重來」的迴圈（log 一直是 Resetting source -1, attempts: 1）。
       它不送 GStreamer ERROR，既有的「出錯就重建」不會觸發；多台的線總幀數還在長，RunWatchdog 也不會動。
       cam215 就這樣停了 4 天。

    判斷：某台的 cam_frames 超過門檻沒增加就回報。沒救回來門檻加倍（上限 max_s），有新幀就歸零。
    多台線路【全部】同時停 = 不是單台的問題（推論卡住），交給 RunWatchdog；單台線路無從比較，一樣重建，
    救不回來時 RunWatchdog 仍會在 RUN_STALL_S 後重啟整個程序。
    check 是純邏輯；實際重建要在 GLib 主迴圈做，由 pipeline 排程呼叫。
    """

    def __init__(self, stats: dict, cameras: list[str], stall_s: float, *,
                 clock: Callable[[], float] = time.monotonic, max_s: float = 600.0) -> None:
        self.stats, self.cameras, self.stall_s, self.max_s = stats, list(cameras), stall_s, max(max_s, stall_s)
        self.clock = clock
        now = clock()
        self._seen = {c: (0, now, stall_s) for c in self.cameras}     # 相機 → (幀數, 上次有進展, 目前門檻)

    def check(self) -> list[tuple[str, float]]:
        """回這次該重建的 [(相機, 幾秒沒新幀)]。"""
        if self.stall_s <= 0 or not self.stats.get("frames", 0):
            return []                                                 # 還沒第一幀：啟動看門狗的事
        now, counts = self.clock(), self.stats.get("cam_frames") or {}
        stalled = []
        for cam in self.cameras:
            n, since, wait = self._seen[cam]
            cur = counts.get(cam, 0)
            if cur != n:
                self._seen[cam] = (cur, now, self.stall_s)
            elif now - since >= wait:
                stalled.append((cam, now - since))
        if len(self.cameras) > 1 and len(stalled) == len(self.cameras):
            return []
        for cam, _ in stalled:
            n, _since, wait = self._seen[cam]
            self._seen[cam] = (n, now, min(self.max_s, wait * 2))
        return stalled

