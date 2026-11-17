"""時間性原語：連續幀去抖、冷卻、事件歷史（回看 / 順序）、等待視窗（A 之後 N 秒內要有 B）。

站別只決定「幾幀」「幾秒」這些數字（那是現場定的），機制放這裡共用。
所有類別都不碰時鐘以外的任何 I/O；時鐘可注入，測試不用 sleep。

後兩個來自舊腳本裡反覆出現的寫法：
  ・「B 出現時，回看前 N 秒有沒有 A」                → EventHistory.seen_within
  ・「C 出現時，前 N 秒 A、B 要齊且順序對」            → EventHistory.first_seen_within
  ・「A 之後 N 秒內要看到 B，否則 NG」                  → WaitWindow
原程式各自用 deque + time.time() 手刻，各有各的 bug；機制收在這裡一份。
"""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable, Iterable


class PersistCounter:
    """連續 N 幀條件成立才算數（去抖）。

    用法：每幀呼叫 update(condition)，回傳「這一幀是否達到持續門檻」。
    條件一斷，計數歸零 —— 單幀雜訊永遠到不了門檻。
    """

    def __init__(self, frames: int) -> None:
        if frames < 1:
            raise ValueError("frames 必須 >= 1")
        self.frames = frames
        self._streak = 0

    @property
    def streak(self) -> int:
        return self._streak

    def update(self, condition: bool) -> bool:
        if not condition:
            self._streak = 0
            return False
        self._streak += 1
        return self._streak >= self.frames

    def reset(self) -> None:
        self._streak = 0


class Cooldown:
    """觸發後 N 秒內不再觸發。

    用法：條件成立時呼叫 fire()，回 True 表示這次算數（並開始冷卻），False 表示還在冷卻中。
    """

    def __init__(self, seconds: float, clock: Callable[[], float] = time.monotonic) -> None:
        self.seconds = seconds
        self._clock = clock
        self._last: float | None = None

    def ready(self) -> bool:
        return self._last is None or (self._clock() - self._last) >= self.seconds

    def fire(self) -> bool:
        if not self.ready():
            return False
        self._last = self._clock()
        return True

    def reset(self) -> None:
        self._last = None


class EventHistory:
    """帶時間戳的事件歷史：每幀 record() 這幀出現了哪些事件名，之後可以回看。

    只保留最近 keep_seconds 秒（超過的在下一次 record / 查詢時修掉），記憶體有上限。
    「事件」是站別定義的字串（通常是類別名，也可以是「part_in_roi」這種組合條件）。
    """

    def __init__(self, keep_seconds: float, clock: Callable[[], float] = time.monotonic) -> None:
        if keep_seconds <= 0:
            raise ValueError("keep_seconds 必須 > 0")
        self.keep_seconds = keep_seconds
        self._clock = clock
        self._entries: deque[tuple[float, frozenset[str]]] = deque()

    def record(self, names: Iterable[str], now: float | None = None) -> None:
        t = self._clock() if now is None else now
        self._entries.append((t, frozenset(names)))
        self._prune(t)

    def _prune(self, now: float) -> None:
        cutoff = now - self.keep_seconds
        while self._entries and self._entries[0][0] < cutoff:
            self._entries.popleft()

    def seen_within(self, name: str, seconds: float, now: float | None = None) -> bool:
        """最近 seconds 秒內有沒有出現過 name。"""
        return self.first_seen_within(name, seconds, now) is not None

    def first_seen_within(self, name: str, seconds: float, now: float | None = None) -> float | None:
        """最近 seconds 秒內 name **最早**出現的時刻；沒有回 None。拿來比先後順序。"""
        t = self._clock() if now is None else now
        cutoff = t - seconds
        for ts, names in self._entries:          # 由舊到新
            if ts < cutoff:
                continue
            if ts > t:
                break
            if name in names:
                return ts
        return None

    def last_seen(self, name: str) -> float | None:
        for ts, names in reversed(self._entries):
            if name in names:
                return ts
        return None

    def clear(self) -> None:
        self._entries.clear()

    def __len__(self) -> int:
        return len(self._entries)


class WaitWindow:
    """「A 發生後，N 秒內要看到 B」：B 來了視窗關閉（滿足）；N 秒內沒來 → 逾時（通常就是 NG）。

    用法：每幀呼叫 update(trigger=這幀有 A, satisfied=這幀有 B)，回傳這一幀的結果：
      "idle"      沒在等，也沒有 A
      "armed"     這幀看到 A，開始等（已經在等的話 A 再出現不會重啟計時 —— 照舊腳本）
      "waiting"   在等，還沒逾時
      "satisfied" 在等的時候 B 來了，視窗關閉
      "timeout"   等超過 N 秒 B 沒來，視窗關閉
    late_ok 決定「B 在 N 秒之後才來」算什麼：
      True  → 只要視窗還開著、B 來了就算滿足（先看 B、再看逾時）
      False → 超過 N 秒之後 B 來也算逾時（B 逾時後才出現照樣警報）
    """

    IDLE, ARMED, WAITING, SATISFIED, TIMEOUT = "idle", "armed", "waiting", "satisfied", "timeout"

    def __init__(self, seconds: float, clock: Callable[[], float] = time.monotonic, *,
                 late_ok: bool = True) -> None:
        if seconds <= 0:
            raise ValueError("seconds 必須 > 0")
        self.seconds = seconds
        self.late_ok = late_ok
        self._clock = clock
        self._since: float | None = None

    @property
    def waiting(self) -> bool:
        return self._since is not None

    def elapsed(self, now: float | None = None) -> float | None:
        if self._since is None:
            return None
        return (self._clock() if now is None else now) - self._since

    def update(self, *, trigger: bool, satisfied: bool, now: float | None = None) -> str:
        t = self._clock() if now is None else now
        if self._since is not None:
            expired = t - self._since > self.seconds
            if satisfied and (self.late_ok or not expired):
                self._since = None
                return self.SATISFIED
            if expired:
                self._since = None
                return self.TIMEOUT
            return self.WAITING
        if trigger:
            self._since = t
            return self.ARMED
        return self.IDLE

    def reset(self) -> None:
        self._since = None
