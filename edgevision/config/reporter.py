"""線路狀態回報（reported 端）。

⭐ 為什麼一定要 runner 主動回報：
    控制端把設定寫進 DB（desired）之後，【不能假設推送成功】。
    Envoy 有記錄在案的失效模式：控制平面推送新設定、資料平面拒絕了卻
    繼續跑舊設定，兩邊都以為沒事 —— 這叫「靜默拒絕」，最難查。

    解法是 desired／reported 調和：runner 回報「我實際載入的是第幾版」，
    控制端比對 lines.config_version(desired) vs line_status.reported_version，
    差異即為「還沒生效」。ZoneMinder 用純 SQL 兩張表就做出這個模式
    （Monitors=desired / Monitor_Status=actual），幾十路的規模不需要
    Device Shadow 那套服務。

⚠️ 心跳的判讀要小心：共用依賴（DB／Triton）抖動會讓【所有】線同時
   看似失聯。watchdog 不可因此觸發全體重啟 —— ZoneMinder 的 zmwatch
   就因為 DB 斷線重啟了整個 fleet，事後才補上防護。
"""

from __future__ import annotations

import logging
import os
import threading

logger = logging.getLogger(__name__)


class LineReporter:
    """定期把「這條線實際跑的版本與狀態」寫回 DB。

    連不上 DB 時靜默降級（回報失敗不該拖垮推論），與 EventBus 同哲學。
    """

    def __init__(self, line_id: str, config_version: int, *, interval: float = 15.0) -> None:
        self.line_id = line_id
        self.config_version = config_version
        self.interval = interval
        self.cameras_running = 0
        self._db = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

        if os.environ.get("CAMERA_SOURCE", "file").lower() != "api":
            # file 模式沒有 DB 可回報，直接停用（不是錯誤）
            logger.info("CAMERA_SOURCE≠api，線路狀態回報停用")
            return
        try:
            from edgevision.storage.db import Database

            self._db = Database()
        except Exception:
            logger.warning("線路狀態回報初始化失敗，停用", exc_info=True)

    @property
    def enabled(self) -> bool:
        return self._db is not None

    def start(self, cameras_running: int = 0) -> None:
        self.cameras_running = cameras_running
        if not self.enabled:
            return
        # 先送一次 applying → ok，讓控制端立刻看到這條線接手了新版本
        self._write(state="ok")
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="line-reporter", daemon=True)
        self._thread.start()
        logger.info("線路狀態回報已啟動（v%d，每 %.0fs）", self.config_version, self.interval)

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3)

    def report_failure(self, error: str) -> None:
        """載入設定失敗時呼叫 —— 讓控制端看得到「這條線套用失敗」而不是靜默沒動。"""
        self._write(state="failed", error=error[:500])

    def _loop(self) -> None:
        while not self._stop.wait(self.interval):
            self._write(state="ok")

    def _write(self, *, state: str, error: str | None = None) -> None:
        if self._db is None:
            return
        try:
            with self._db.cursor() as cur:
                cur.execute(
                    "INSERT INTO line_status "
                    "  (line_id, reported_version, apply_state, apply_error, "
                    "   heartbeat_at, cameras_running, updated_at) "
                    "VALUES (%s, %s, %s, %s, now(), %s, now()) "
                    "ON CONFLICT (line_id) DO UPDATE SET "
                    "  reported_version = EXCLUDED.reported_version, "
                    "  apply_state = EXCLUDED.apply_state, "
                    "  apply_error = EXCLUDED.apply_error, "
                    "  heartbeat_at = now(), "
                    "  cameras_running = EXCLUDED.cameras_running, "
                    "  updated_at = now()",
                    (self.line_id, self.config_version, state, error, self.cameras_running),
                )
        except Exception:
            # 回報失敗不影響推論
            logger.debug("線路狀態回報失敗（已忽略）", exc_info=True)
