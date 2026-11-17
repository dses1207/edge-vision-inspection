"""② 健康巡檢：每 60 秒讀 /healthz，狀態機只在【變化】時通知。

⭐ 資料來源只有一個：/healthz

    lines[]（runner 心跳、版本同步）、services[]（消費者在不在做事）、
    storage（DB / MinIO）都已經在那裡算好了。這裡不重新探測任何東西 ——
    一個真相來源，notifier 保持薄。之後要加巡檢項目，去改 /healthz。

⭐ 狀態機（照舊系統監控服務的意圖，規則來自 Alertmanager）

    · 連續 2 輪一致才切換 —— 擋抖動。一次網路抖動不該發一則 🔴 再發一則 🟢
    · 切換才通知：🟠 degraded、🔴 down、🟢 recovered（附中斷了幾分鐘）
    · 未恢復的 down 每 4 小時提醒一次 —— 不然半夜掛了早上沒人記得
    · 同一項目 10 分鐘冷卻
    · 抑制：DB down → 壓掉所有 stale / degraded（它們是 DB 掛的【症狀】，不是獨立故障）

    狀態存 Redis，重啟不丟：hash health:state {item: json}。

⚠️ 為什麼 line stale 是 down 而不是 degraded

    runner 心跳 60 秒沒來，那條線就是沒在推論 —— 對產線來說跟掛了一樣。
    line_c 死了 20 天沒人發現，正是因為沒有東西把「stale」當成「down」來叫。
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from edgevision.notify.sender import Card, Sender

logger = logging.getLogger(__name__)

STATES = ("ok", "degraded", "down")
_RANK = {s: i for i, s in enumerate(STATES)}

# ── 從 /healthz 算出每個項目的狀態 ─────────────────────────

Item = tuple[str, str, str]      # (item, state, reason)


def evaluate(healthz: dict) -> list[Item]:
    """把 /healthz 的回應攤平成 [(item, state, reason)]。純函式，可測。"""
    out: list[Item] = []

    for k, ok in (healthz.get("storage") or {}).items():
        if ok is None:
            continue
        out.append((f"storage:{k}", "ok" if ok else "down", None if ok else f"{k} 無回應"))

    for ln in healthz.get("lines") or []:
        lid = ln.get("line_id", "?")
        if ln.get("stale"):
            out.append((f"line:{lid}", "down", f"runner 心跳 {ln.get('heartbeat_age_s')} 秒沒來"))
        elif ln.get("apply_state") == "failed":
            out.append((f"line:{lid}", "down", f"設定套用失敗：{(ln.get('apply_error') or '')[:80]}"))
        elif not ln.get("in_sync", True):
            out.append((f"line:{lid}", "degraded",
                        f"設定 v{ln.get('desired_version')} 未生效（runner 在 v{ln.get('reported_version')}）"))
        else:
            out.append((f"line:{lid}", "ok", None))
        # 相機層級：線路心跳正常（沒 stale / failed）時才逐台看；線路斷了由線路那一項叫，不重複。
        # unknown（舊 runner 沒回報）不列入 —— 狀態機對沒出現的項目保持原狀。
        if not ln.get("stale") and ln.get("apply_state") != "failed":
            for cam in ln.get("cameras") or []:
                st = cam.get("state")
                if st == "no_frames":
                    out.append((f"camera:{cam['camera_id']}", "down", f"{lid} 的這台相機沒有畫面（0 fps）"))
                elif st == "skipped":
                    out.append((f"camera:{cam['camera_id']}", "down",
                                f"{lid} 的這台相機設定有誤，沒有啟動：{(cam.get('reason') or '')[:120]}"))
                elif st == "ok":
                    out.append((f"camera:{cam['camera_id']}", "ok", None))

    for sv in healthz.get("services") or []:
        name = sv.get("service", "?")
        st = sv.get("state")
        state = "down" if st == "stale" else ("degraded" if st == "degraded" else "ok")
        out.append((f"service:{name}", state, sv.get("reason")))

    return out


def inhibit(items: list[Item]) -> list[Item]:
    """抑制：共用依賴掛了，它的症狀不要各自叫。

    DB down → 線路 stale、消費者 stale/degraded 全是它造成的，只留 DB 那一則。
    （ZoneMinder 的教訓：一個依賴抖動觸發全體重啟。這裡只是不重複叫，不做處置。）
    """
    down_deps = {i for i, s, _ in items if i.startswith("storage:") and s == "down"}
    if not down_deps:
        return items
    kept = []
    for item, state, reason in items:
        if item.startswith("storage:") or state == "ok":
            kept.append((item, state, reason))
        else:
            kept.append((item, "ok", None))      # 視為被抑制，不算故障
    return kept


# ── 狀態機 ────────────────────────────────────────────────

@dataclass
class ItemState:
    state: str = "ok"                # 已確認的狀態
    pending: str | None = None       # 觀察到但還沒連續 2 輪的
    pending_n: int = 0
    since: float = 0.0               # 進入非 ok 的時間
    notified_at: float = 0.0         # 上次通知（冷卻、4h 提醒用）
    reason: str | None = None

    def to_json(self) -> str:
        return json.dumps(self.__dict__)

    @classmethod
    def from_json(cls, s: str | None) -> "ItemState":
        if not s:
            return cls()
        try:
            return cls(**json.loads(s))
        except Exception:  # noqa: BLE001 —— 壞掉的舊狀態就當全新
            return cls()


@dataclass
class Transition:
    item: str
    from_state: str
    to_state: str
    reason: str | None
    duration_s: int | None = None    # 恢復時：中斷了幾秒
    repeat: bool = False             # 是「4h 提醒」不是新切換


class HealthMonitor:
    """一輪 = 讀 /healthz → 算狀態 → 抑制 → 狀態機 → 通知變化 → 落 health_events。"""

    def __init__(self, *, redis, db, sender: Sender,
                 fetch: Callable[[], dict],
                 confirm_rounds: int = 2,
                 cooldown_s: float = 600.0,
                 repeat_s: float = 14400.0,
                 receiver: str = "health",
                 key: str = "health:state") -> None:
        self.redis, self.db, self.sender, self.fetch = redis, db, sender, fetch
        self.confirm_rounds, self.cooldown_s, self.repeat_s = confirm_rounds, cooldown_s, repeat_s
        self.receiver, self.key = receiver, key
        self.rounds = 0
        self.notified = 0

    # 狀態存 Redis hash，重啟不丟
    def _load(self, item: str) -> ItemState:
        return ItemState.from_json(self.redis.hget(self.key, item))

    def _save(self, item: str, st: ItemState) -> None:
        self.redis.hset(self.key, item, st.to_json())

    def step(self, items: list[Item], now: float | None = None) -> list[Transition]:
        """餵一輪觀察，回傳要通知的切換。純狀態邏輯，不做 I/O（Redis 除外）。"""
        now = now or time.time()
        out: list[Transition] = []
        for item, observed, reason in items:
            st = self._load(item)

            # 連續 N 輪一致才確認
            if observed == st.state:
                st.pending, st.pending_n = None, 0
            else:
                if st.pending == observed:
                    st.pending_n += 1
                else:
                    st.pending, st.pending_n = observed, 1
                if st.pending_n >= self.confirm_rounds:
                    prev = st.state
                    st.state, st.pending, st.pending_n = observed, None, 0
                    st.reason = reason
                    if observed == "ok":
                        dur = int(now - st.since) if st.since else None
                        st.since = 0.0
                        out.append(Transition(item, prev, "ok", reason, duration_s=dur))
                    else:
                        if prev == "ok":
                            st.since = now
                        # 冷卻：同項目 10 分鐘內不重複叫（除非是往更壞走）
                        if now - st.notified_at >= self.cooldown_s or _RANK[observed] > _RANK[prev]:
                            out.append(Transition(item, prev, observed, reason))
                            st.notified_at = now

            # 未恢復的 down 每 repeat_s 提醒
            if st.state == "down" and st.pending is None and now - st.notified_at >= self.repeat_s:
                out.append(Transition(item, "down", "down", st.reason,
                                      duration_s=int(now - st.since) if st.since else None, repeat=True))
                st.notified_at = now

            self._save(item, st)
        return out

    def run_once(self) -> list[Transition]:
        self.rounds += 1
        try:
            hz = self.fetch()
        except Exception as exc:  # noqa: BLE001
            # /healthz 本身打不到 = output-api 掛了。當成一個項目餵進狀態機，
            # 一樣走連續 2 輪 + 冷卻，不要每 60 秒叫一次。
            logger.warning("讀 /healthz 失敗：%s", exc)
            hz = {"storage": {}, "lines": [], "services": [],
                  "_api_down": f"{type(exc).__name__}: {exc}"}
        items = inhibit(evaluate(hz))
        if "_api_down" in hz:
            items.append(("api:output-api", "down", hz["_api_down"][:120]))
        else:
            items.append(("api:output-api", "ok", None))

        transitions = self.step(items)
        for t in transitions:
            self._record(t)
            self._notify(t)
        return transitions

    def _record(self, t: Transition) -> None:
        if t.repeat or self.db is None:
            return
        try:
            with self.db.cursor() as cur:
                cur.execute(
                    "INSERT INTO health_events (item, from_state, to_state, duration_s) "
                    "VALUES (%s, %s, %s, %s)", (t.item, t.from_state, t.to_state, t.duration_s))
        except Exception:  # noqa: BLE001 —— 記不進去不能擋通知
            logger.warning("health_events 寫入失敗", exc_info=True)

    def _notify(self, t: Transition) -> None:
        if t.to_state == "ok":
            mins = f"，中斷 {t.duration_s // 60} 分鐘" if t.duration_s else ""
            card = Card(title=f"{t.item} 已恢復{mins}", level="recovered")
        elif t.repeat:
            mins = t.duration_s // 60 if t.duration_s else "?"
            card = Card(title=f"{t.item} 仍然 down（已 {mins} 分鐘）", level="down",
                        body=[t.reason or ""])
        else:
            card = Card(title=f"{t.item} {t.to_state}", level=t.to_state,
                        body=[t.reason or ""])
        r = self.sender.send(self.receiver, card)
        self.notified += 1
        _log_notify(self.db, kind="health", event_id=t.item, receiver=self.receiver, result=r)


def _log_notify(db, *, kind: str, event_id: str, receiver: str, result) -> None:
    if db is None:
        return
    status = "sent" if result.sent else ("suppressed" if result.dry_run else "failed")
    try:
        with db.cursor() as cur:
            cur.execute(
                "INSERT INTO notify_log (kind, event_id, receiver, status, error) VALUES (%s,%s,%s,%s,%s)",
                (kind, event_id, receiver, status, result.error))
    except Exception:  # noqa: BLE001
        logger.warning("notify_log 寫入失敗", exc_info=True)


def fetch_healthz(base_url: str) -> dict:
    """讀 /healthz。host network 下 base_url 是 localhost:18080。"""
    import httpx
    with httpx.Client(timeout=8.0, trust_env=False) as c:
        r = c.get(f"{base_url.rstrip('/')}/healthz")
        r.raise_for_status()
        return r.json()
