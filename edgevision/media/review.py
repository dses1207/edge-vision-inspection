"""標注審核：設定、輪次、優先順序、送出差異。

    設定   review_defaults.yaml（全域預設，唯一放數字的地方）← media_collections.review_config（只存覆蓋）
    輪次挑最該看的 N 張（優先分數）→ 抽 blind_ratio 當抽查圖（清掉預標）→ 一張一張審
    送出跟建立輪次時的框（baseline）比：原樣 / 修改 / 新增 / 刪除，記在 media_round_items.diff

純函式（合併設定、算優先分數、比差異）沒有 I/O，單元測試直接測；ReviewRounds 是資料層（SQL 只寫在這裡與 library.py）。
"""

from __future__ import annotations

import copy
import json
import math
import random
import uuid
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from edgevision.media.library import EXPORTABLE, Conflict, MediaLibrary, NotFound, _row, _rows

DEFAULTS_PATH = Path(__file__).with_name("review_defaults.yaml")
ROUND_STATES = ("pending", "done", "doubt")
PRIORITY_SIGNALS = ("rare", "low_conf", "empty", "diverse")


# ── 設定：只定義欄位與範圍，不帶預設值（預設全在 review_defaults.yaml）──────────

class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ClassDefaults(_Strict):
    show_conf: float = Field(ge=0, le=1)
    accept_conf: float = Field(ge=0, le=1)
    bulk_accept: bool
    rare: bool


class RoundCfg(_Strict):
    size: int = Field(ge=1)
    size_min: int = Field(ge=1)
    size_max: int = Field(ge=1)
    blind_ratio: float = Field(ge=0, le=1)
    blind_ratio_max: float = Field(ge=0, le=1)
    max_per_group: int = Field(ge=1)

    @model_validator(mode="after")
    def _ranges(self):
        if not self.size_min <= self.size <= self.size_max:
            raise ValueError(f"round.size 要在 size_min～size_max 之間（{self.size_min}～{self.size_max}）")
        if self.blind_ratio > self.blind_ratio_max:
            raise ValueError("round.blind_ratio 不能超過 blind_ratio_max")
        return self


class PriorityCfg(_Strict):
    reason_min: float = Field(ge=0, le=1)
    weights: dict[str, float]
    diverse_group_size: int = Field(ge=1)

    @model_validator(mode="after")
    def _known(self):
        bad = set(self.weights) - set(PRIORITY_SIGNALS)
        if bad:
            raise ValueError(f"priority.weights 不認得 {sorted(bad)}（可用 {PRIORITY_SIGNALS}）")
        if any(w < 0 for w in self.weights.values()):
            raise ValueError("priority.weights 不能是負的")
        return self


class SequenceCfg(_Strict):
    frames_before: int = Field(ge=0, le=100)
    frames_after: int = Field(ge=0, le=100)
    on_target_ratio: float = Field(ge=0, le=1)


class BulkCfg(_Strict):
    default_conf: float = Field(ge=0, le=1)
    preview_n: int = Field(ge=0, le=100)


class ReviewUiCfg(_Strict):
    require_clean_view: bool
    clean_view_ms: int = Field(ge=0, le=10000)
    min_box_px: int = Field(ge=0, le=100)
    idle_s: int = Field(ge=1)
    lock_s: int = Field(ge=30, le=3600)
    lock_renew_s: int = Field(ge=5)
    draft_debounce_ms: int = Field(ge=0)
    offline_retry_s: int = Field(ge=1)
    offline_retry_max_s: int = Field(ge=1)
    peek: int = Field(ge=0, le=10)

    @model_validator(mode="after")
    def _renew(self):
        if self.lock_renew_s >= self.lock_s:
            raise ValueError("review.lock_renew_s 要小於 lock_s，否則鎖會在續期前過期")
        return self


class QaCfg(_Strict):
    iou: float = Field(gt=0, le=1)


class GuideExample(_Strict):
    url: str
    good: bool
    caption: str = ""


class GuideEntry(_Strict):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)
    cls: str = Field(alias="class")
    text: str
    examples: list[GuideExample] = []


class ClassOverride(_Strict):
    color: str | None = None
    hotkey: str | None = None
    show_conf: float | None = Field(None, ge=0, le=1)
    accept_conf: float | None = Field(None, ge=0, le=1)
    bulk_accept: bool | None = None
    rare: bool | None = None


class ReviewDefaults(_Strict):
    """review_defaults.yaml 的形狀（每個欄位必填）。"""
    class_defaults: ClassDefaults
    palette: list[str] = Field(min_length=1)
    swap_pairs: list[list[str]]
    round: RoundCfg
    priority: PriorityCfg
    sequence: SequenceCfg
    bulk_accept: BulkCfg
    review: ReviewUiCfg
    qa: QaCfg
    thumb_px: int = Field(ge=64, le=2048)
    guide: list[GuideEntry]


@lru_cache(maxsize=1)
def load_defaults(path: str | None = None) -> dict:
    """讀全域預設並驗證（缺欄位 / 超出範圍直接報錯，不猜值）。"""
    raw = yaml.safe_load(Path(path or DEFAULTS_PATH).read_text(encoding="utf-8"))
    return ReviewDefaults.model_validate(raw).model_dump(by_alias=True)


def deep_merge(base: dict, over: dict) -> dict:
    """over 蓋 base（dict 遞迴合併，其他型別整個換掉）；over 裡的 None = 刪掉這個覆蓋、回到預設。"""
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if v is None:
            out.pop(k, None)
        elif isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def prune_none(d: dict) -> dict:
    """覆蓋裡值為 None 的鍵拿掉（PATCH 傳 null = 回到預設）；空的 dict 也拿掉。"""
    out = {}
    for k, v in d.items():
        if v is None:
            continue
        if isinstance(v, dict):
            v = prune_none(v)
            if not v:
                continue
        out[k] = v
    return out


def resolve_config(class_names: list[str], override: dict | None, *, defaults: dict | None = None) -> dict:
    """預設 ← 圖片集覆蓋 → 前端拿到的完整設定（classes 依圖片集類別展開成清單）。"""
    d = copy.deepcopy(defaults or load_defaults())
    over = copy.deepcopy(override or {})
    cls_over = over.pop("classes", {}) or {}
    unknown = set(cls_over) - set(class_names)
    if unknown:
        raise ValueError(f"classes 覆蓋了不在圖片集裡的類別：{sorted(unknown)}（圖片集類別 {class_names}）")
    for name, o in cls_over.items():
        ClassOverride.model_validate(o)
    merged = deep_merge(d, over)
    cd = merged.pop("class_defaults")
    palette = merged.pop("palette")
    classes = []
    for i, name in enumerate(class_names):
        c = {"name": name, "color": palette[i % len(palette)], "hotkey": str(i + 1) if i < 9 else None, **cd}
        c.update({k: v for k, v in (cls_over.get(name) or {}).items() if v is not None})
        if c["accept_conf"] < c["show_conf"]:
            raise ValueError(f"類別 {name} 的 accept_conf 不能小於 show_conf")
        classes.append(c)
    merged["classes"] = classes
    for pair in merged["swap_pairs"]:
        if len(pair) != 2 or any(p not in class_names for p in pair):
            raise ValueError(f"swap_pairs 每組要兩個圖片集裡的類別：{pair}")
    for g in merged["guide"]:
        if g["class"] not in class_names:
            raise ValueError(f"guide 的類別 {g['class']!r} 不在圖片集裡")
    # 整份再驗一次（覆蓋可能把範圍弄壞，例如 size 超過 size_max）
    check = {k: v for k, v in merged.items() if k != "classes"} | {"class_defaults": cd, "palette": palette}
    ReviewDefaults.model_validate(check)
    return merged


def class_cfg(cfg: dict) -> dict[str, dict]:
    return {c["name"]: c for c in cfg["classes"]}


# ── 框的 id 與差異 ────────────────────────────────────────────

def new_shape_id() -> str:
    return uuid.uuid4().hex[:12]


def ensure_ids(shapes: list[dict]) -> tuple[list[dict], bool]:
    """沒有 id 的框補上（舊資料、自動標注寫的框）。回傳 (shapes, 有沒有改)。"""
    changed = False
    out = []
    for s in shapes:
        if not s.get("id"):
            s = {**s, "id": new_shape_id()}
            changed = True
        out.append(s)
    return out, changed


def _same_geom(a: dict, b: dict, tol: float = 1e-4) -> bool:
    if a.get("type") != b.get("type") or a.get("class") != b.get("class"):
        return False
    pa, pb = a.get("points") or [], b.get("points") or []
    if len(pa) != len(pb):
        return False
    return all(abs(x - y) <= tol for p, q in zip(pa, pb) for x, y in zip(p[:2], q[:2]))


def shape_diff(baseline: list[dict], submitted: list[dict]) -> dict:
    """依框 id 比：原樣接受 / 修改（位置或類別）/ 新增 / 刪除。per_class 用送出後的類別（刪除用原類別）。"""
    base = {s["id"]: s for s in baseline if s.get("id")}
    per: dict[str, Counter] = {}

    def bump(cls: str, k: str) -> None:
        per.setdefault(cls, Counter())[k] += 1

    seen = set()
    for s in submitted:
        b = base.get(s.get("id"))
        if b is None:
            bump(s["class"], "added")
            continue
        seen.add(s["id"])
        if _same_geom(b, s):
            bump(s["class"], "accepted")
        else:
            bump(s["class"], "modified")
            if b.get("class") != s.get("class"):
                bump(s["class"], "class_changed")
    for sid, b in base.items():
        if sid not in seen:
            bump(b["class"], "deleted")
    total = Counter()
    for c in per.values():
        total.update(c)
    keys = ("accepted", "modified", "added", "deleted")
    return {**{k: total.get(k, 0) for k in keys}, "class_changed": total.get("class_changed", 0),
            "per_class": {cls: {k: c.get(k, 0) for k in (*keys, "class_changed")} for cls, c in per.items()}}


def mark_sources(baseline: list[dict], submitted: list[dict]) -> list[dict]:
    """送出的框：原樣接受的保留模型來源；改過或新增的記成人工（匯出不在乎，但歷史看得出誰畫的）。"""
    base = {s["id"]: s for s in baseline if s.get("id")}
    out = []
    for s in submitted:
        b = base.get(s.get("id"))
        if b is not None and _same_geom(b, s):
            s = {**s, "source": b.get("source") or s.get("source") or {"kind": "human"}}
        else:
            s = {**s, "source": {"kind": "human"}}
        out.append(s)
    return out


# ── 優先分數 ──────────────────────────────────────────────────

def _conf(s: dict) -> float | None:
    return (s.get("source") or {}).get("conf")


def priority_signals(item: dict, cfg: dict) -> dict[str, float]:
    """一張圖的審核訊號（各 0–1）。只看模型框；人標過的圖不會進輪次。"""
    cc = class_cfg(cfg)
    shapes = [s for s in item.get("shapes") or []
              if s.get("class") in cc and (_conf(s) is None or _conf(s) >= cc[s["class"]]["show_conf"])]
    sig = {k: 0.0 for k in PRIORITY_SIGNALS}
    if any(cc[s["class"]]["rare"] for s in shapes):
        sig["rare"] = 1.0
    lows = []
    for s in shapes:
        c, k = _conf(s), cc[s["class"]]
        if c is not None and c < k["accept_conf"] and k["accept_conf"] > k["show_conf"]:
            mid = (k["show_conf"] + k["accept_conf"]) / 2
            half = (k["accept_conf"] - k["show_conf"]) / 2
            lows.append(1.0 - abs(c - mid) / half * 0.5)      # 中間 = 1，兩端 = 0.5
    if lows:
        sig["low_conf"] = max(lows)
    if not shapes:
        sig["empty"] = 1.0
    gs = (item.get("select_info") or {}).get("group_size")
    if gs:
        sig["diverse"] = min(1.0, math.log1p(gs) / math.log1p(cfg["priority"]["diverse_group_size"]))
    return sig


def priority_of(item: dict, cfg: dict) -> tuple[float, list[str]]:
    sig = priority_signals(item, cfg)
    w = cfg["priority"]["weights"]
    score = sum(w.get(k, 0.0) * v for k, v in sig.items())
    total = sum(w.values()) or 1.0
    reasons = [k for k in PRIORITY_SIGNALS if sig[k] >= cfg["priority"]["reason_min"] and w.get(k, 0) > 0]
    return round(score / total, 4), reasons


def pick_round(items: list[dict], cfg: dict, *, size: int, blind_ratio: float, seed: int) -> list[dict]:
    """依優先分數挑 size 張（同一相似群最多 max_per_group 張），再隨機抽 blind_ratio 當抽查圖。

    回傳 [{"item": it, "priority", "reasons", "blind", "rank"}]，rank 0 = 最先審。
    """
    scored = []
    for it in items:
        p, reasons = priority_of(it, cfg)
        scored.append((p, it["id"], it, reasons))
    scored.sort(key=lambda t: (-t[0], t[1]))
    per_group: Counter = Counter()
    cap = cfg["round"]["max_per_group"]
    chosen, overflow = [], []
    for p, _, it, reasons in scored:
        g = (it.get("select_info") or {}).get("group")
        key = (it.get("source_id"), g) if g is not None else None
        if key is not None and per_group[key] >= cap:
            overflow.append((p, it, reasons))
            continue
        if key is not None:
            per_group[key] += 1
        chosen.append((p, it, reasons))
        if len(chosen) >= size:
            break
    if len(chosen) < size:                 # 群上限擋掉太多：依分數補滿
        chosen += overflow[: size - len(chosen)]
    n_blind = int(round(len(chosen) * blind_ratio))
    rng = random.Random(seed)
    blind_idx = set(rng.sample(range(len(chosen)), n_blind)) if n_blind else set()
    return [{"item": it, "priority": p, "reasons": reasons, "blind": i in blind_idx, "rank": i}
            for i, (p, it, reasons) in enumerate(chosen)]


def visible_shapes(shapes: list[dict], cfg: dict) -> list[dict]:
    """給前端的框：信心 < show_conf 的模型框不給（人畫的框沒有信心，一律給）。"""
    cc = class_cfg(cfg)
    out = []
    for s in shapes:
        c = _conf(s)
        k = cc.get(s.get("class"))
        if c is not None and k is not None and c < k["show_conf"]:
            continue
        out.append(s)
    return out


# ── 資料層 ────────────────────────────────────────────────────

class ReviewRounds:
    """輪次與審核紀錄。圖本身的狀態 / 框 / 鎖仍由 MediaLibrary 管（送出走 lib.submit，同一套 revision 與鎖）。"""

    def __init__(self, lib: MediaLibrary) -> None:
        self.lib = lib
        self.db = lib.db

    # 設定 ------------------------------------------------------------

    def config(self, cid: int) -> dict:
        col = self.lib.collection(cid)
        return resolve_config(col["class_names"], col.get("review_config") or {})

    def override(self, cid: int) -> dict:
        return self.lib.collection(cid).get("review_config") or {}

    def patch_config(self, cid: int, patch: dict) -> tuple[dict, dict, dict]:
        """部分更新覆蓋（null = 回到預設）。先驗證合併結果，過了才寫。回傳 (改前覆蓋, 改後覆蓋, 完整設定)。"""
        col = self.lib.collection(cid)
        before = col.get("review_config") or {}
        after = prune_none(deep_merge(before, patch))
        full = resolve_config(col["class_names"], after)
        with self.db.cursor() as cur:
            cur.execute("UPDATE media_collections SET review_config = %s, updated_at = now() WHERE id = %s",
                        (json.dumps(after), cid))
        return before, after, full

    # 輪次 ------------------------------------------------------------

    def create_round(self, cid: int, *, actor: str, name: str | None = None, size: int | None = None,
                     blind_ratio: float | None = None, mode: str = "review",
                     batch_ids: list[int] | None = None) -> dict:
        """mode=review：依優先分數挑 size 張、抽 blind_ratio 當抽查圖。
        mode=seed（種子輪）：可選的圖全收（上限 size_max）、整輪都不給預標（有框也清掉），從頭標。"""
        if mode not in ("review", "seed"):
            raise ValueError("mode 只能是 review / seed")
        cfg = self.config(cid)
        rc = cfg["round"]
        batch_ids = self.lib.check_batches(cid, batch_ids)
        items = self._eligible(cid, batch_ids)
        if not items:
            raise ValueError("沒有可以審的圖（未標 / AI 預標、不在其他進行中的輪次、沒被鎖住"
                             + ("、在指定批次裡）" if batch_ids else "）"))
        if mode == "seed":
            size = min(len(items), rc["size_max"]) if size is None else size
            blind_ratio = 0.0
            if not 1 <= size <= rc["size_max"]:
                raise ValueError(f"size 要在 1～{rc['size_max']} 之間")
        else:
            size = rc["size"] if size is None else size
            blind_ratio = rc["blind_ratio"] if blind_ratio is None else blind_ratio
            if not rc["size_min"] <= size <= rc["size_max"]:
                raise ValueError(f"size 要在 {rc['size_min']}～{rc['size_max']} 之間")
            if not 0 <= blind_ratio <= rc["blind_ratio_max"]:
                raise ValueError(f"blind_ratio 要在 0～{rc['blind_ratio_max']} 之間")
        cfg["round"] = {**rc, "size": size, "blind_ratio": blind_ratio}
        with self.db.cursor() as cur:
            cur.execute("SELECT count(*) FROM media_review_rounds WHERE collection_id = %s", (cid,))
            n_prev = int(cur.fetchone()[0])
        name = name or f"第 {n_prev + 1} 輪"
        with self.db.cursor() as cur:
            cur.execute("INSERT INTO media_review_rounds (collection_id, name, config, created_by, mode, batch_ids)"
                        " VALUES (%s,%s,%s,%s,%s,%s) RETURNING id",
                        (cid, name, json.dumps(cfg), actor, mode, json.dumps(batch_ids) if batch_ids else None))
            rid = int(cur.fetchone()[0])
        picked = pick_round(items, cfg, size=size, blind_ratio=blind_ratio, seed=rid)
        models: Counter = Counter()
        for p in picked:
            it = p["item"]
            shapes, changed = ensure_ids(it["shapes"] or [])
            if changed:                    # 只補 id，不算一次修改（不動 revision、不留歷史）
                with self.db.cursor() as cur:
                    cur.execute("UPDATE media_items SET shapes = %s WHERE id = %s", (json.dumps(shapes), it["id"]))
            for s in shapes:
                src = s.get("source") or {}
                if src.get("kind") == "model" and src.get("model"):
                    models[(src["model"], src.get("version"))] += 1
            p["baseline"] = shapes
            if p["blind"] or (mode == "seed" and shapes):
                self.lib._write(it["id"], [], "unlabeled", actor=actor, action="blind",
                                expect_revision=it["revision"])
        with self.db.cursor() as cur:
            cur.executemany(
                "INSERT INTO media_round_items (round_id, item_id, rank, priority, reasons, blind, baseline)"
                " VALUES (%s,%s,%s,%s,%s,%s,%s)",
                [(rid, p["item"]["id"], p["rank"], p["priority"], json.dumps(p["reasons"]), p["blind"],
                  json.dumps(p["baseline"])) for p in picked])
            by_reason = Counter(r for p in picked for r in p["reasons"])
            comp = {"n_items": len(picked), "n_blind": sum(p["blind"] for p in picked),
                    "by_reason": dict(by_reason), "n_eligible": len(items)}
            model = None
            if models:
                (mn, mv), _ = models.most_common(1)[0]
                model = {"name": mn, "version": mv}
            cur.execute("UPDATE media_review_rounds SET composition = %s, model = %s WHERE id = %s",
                        (json.dumps(comp), json.dumps(model) if model else None, rid))
        return self.round(rid)

    def _eligible(self, cid: int, batch_ids: list[int] | None = None) -> list[dict]:
        """能進新輪次的圖：未標 / AI 預標、不在任何進行中的輪次、沒被別人鎖住（batch_ids：只從這些批次挑）。"""
        cond, params = "", []
        if batch_ids:
            cond = "AND i.batch_id = ANY(%s)"; params.append(list(batch_ids))
        with self.db.cursor() as cur:
            cur.execute(
                f"""SELECT i.id, i.revision, i.shapes, i.select_info, i.source_id FROM media_items i
                   WHERE i.collection_id = %s AND i.status IN ('unlabeled', 'auto') {cond}
                     AND (i.locked_by IS NULL OR i.lock_until < now())
                     AND NOT EXISTS (SELECT 1 FROM media_round_items ri JOIN media_review_rounds r ON r.id = ri.round_id
                                     WHERE ri.item_id = i.id AND r.status = 'open')
                   ORDER BY i.id""", (cid, *params))
            return _rows(cur)

    def round_row(self, rid: int) -> dict:
        with self.db.cursor() as cur:
            cur.execute("SELECT * FROM media_review_rounds WHERE id = %s", (rid,))
            r = _row(cur)
        if r is None:
            raise NotFound(f"輪次 {rid} 不存在")
        return r

    def round(self, rid: int) -> dict:
        r = self.round_row(rid)
        with self.db.cursor() as cur:
            cur.execute(
                """SELECT ri.state, i.status, count(*) FROM media_round_items ri JOIN media_items i ON i.id = ri.item_id
                   WHERE ri.round_id = %s GROUP BY ri.state, i.status""", (rid,))
            rows = cur.fetchall()
            cur.execute(
                """SELECT reviewed_by, count(*), avg(seconds_spent) FROM media_round_items
                   WHERE round_id = %s AND state = 'done' AND reviewed_by IS NOT NULL
                   GROUP BY reviewed_by ORDER BY count(*) DESC""", (rid,))
            revs = cur.fetchall()
        prog = {"pending": 0, "labeled": 0, "background": 0, "doubt": 0}
        for state, status, n in rows:
            if state == "done":
                prog[status if status in EXPORTABLE else "labeled"] += int(n)
            else:
                prog[state] += int(n)
        comp = r["composition"] or {}
        return {"id": r["id"], "collection_id": r["collection_id"], "name": r["name"], "status": r["status"],
                "mode": r.get("mode") or "review", "batch_ids": r.get("batch_ids"),
                "model": r["model"], "n_items": comp.get("n_items", 0), "n_blind": comp.get("n_blind", 0),
                "n_sequences": 0, "progress": prog, "by_reason": comp.get("by_reason", {}),
                "reviewers": [{"actor": a, "n": int(n), "avg_seconds": round(float(s), 1) if s is not None else None}
                              for a, n, s in revs],
                "config": r["config"], "created_by": r["created_by"], "created_at": r["created_at"],
                "closed_by": r["closed_by"], "closed_at": r["closed_at"]}

    def rounds(self, cid: int) -> list[dict]:
        with self.db.cursor() as cur:
            cur.execute("SELECT id FROM media_review_rounds WHERE collection_id = %s ORDER BY id DESC", (cid,))
            ids = [int(r[0]) for r in cur.fetchall()]
        out = []
        for rid in ids:
            d = self.round(rid)
            d.pop("config", None)
            out.append(d)
        return out

    def close_round(self, rid: int, *, actor: str) -> dict:
        with self.db.cursor() as cur:
            cur.execute("UPDATE media_review_rounds SET status = 'closed', closed_by = %s, closed_at = now()"
                        " WHERE id = %s AND status = 'open' RETURNING id", (actor, rid))
            ok = cur.fetchone()
        if not ok:
            self.round_row(rid)            # 不存在 → NotFound
            raise Conflict("這一輪已經關閉")
        return self.round(rid)

    def _open(self, rid: int) -> dict:
        r = self.round_row(rid)
        if r["status"] != "open":
            raise Conflict("這一輪已經關閉，不能再審")
        return r

    # 清單與下一張 ------------------------------------------------------

    def round_items(self, rid: int, *, state: str | None = None, reason: str | None = None,
                    cls: str | None = None, blind: bool | None = None, order: str = "priority",
                    offset: int = 0, limit: int = 100) -> dict:
        r = self.round_row(rid)
        cfg = r["config"]
        conds, params = ["ri.round_id = %s"], [rid]
        if state:
            if state not in ROUND_STATES:
                raise ValueError(f"state 只能是 {ROUND_STATES}")
            conds.append("ri.state = %s"); params.append(state)
        if reason:
            conds.append("ri.reasons ? %s"); params.append(reason)
        if blind is not None:
            conds.append("ri.blind = %s"); params.append(blind)
        if cls:
            conds.append("EXISTS (SELECT 1 FROM jsonb_array_elements(i.shapes) e WHERE e->>'class' = %s)")
            params.append(cls)
        order_sql = {"priority": "ri.rank", "time": "i.captured_at NULLS LAST, i.id",
                     "conf": "(SELECT min((e->'source'->>'conf')::float) FROM jsonb_array_elements(i.shapes) e) "
                             "NULLS LAST, ri.rank"}.get(order)
        if order_sql is None:
            raise ValueError("order 只能是 priority / time / conf")
        where = " AND ".join(conds)
        with self.db.cursor() as cur:
            cur.execute(f"SELECT count(*) FROM media_round_items ri JOIN media_items i ON i.id = ri.item_id"
                        f" WHERE {where}", params)
            total = int(cur.fetchone()[0])
            cur.execute(
                f"""SELECT ri.*, i.status, i.revision, i.shapes, i.blob_sha256, i.locked_by, i.lock_until,
                           b.width, b.height FROM media_round_items ri
                    JOIN media_items i ON i.id = ri.item_id JOIN media_blobs b ON b.sha256 = i.blob_sha256
                    WHERE {where} ORDER BY {order_sql} OFFSET %s LIMIT %s""", (*params, offset, limit))
            rows = _rows(cur)
        items = []
        for x in rows:
            items.append({
                "id": x["item_id"], "status": x["status"], "revision": x["revision"], "state": x["state"],
                "priority": x["priority"], "reasons": x["reasons"], "blind": x["blind"], "sequence_id": None,
                "rank": x["rank"], "width": x["width"], "height": x["height"],
                "thumb_url": self.thumb_url(x["blob_sha256"], cfg["thumb_px"]),
                "shapes": visible_shapes(x["shapes"] or [], cfg),
                "reviewed_by": x["reviewed_by"], "locked_by": x["locked_by"] if x["lock_until"] and
                _future(x["lock_until"]) else None})
        return {"items": items, "total": total, "offset": offset, "limit": limit}

    def next_item(self, rid: int, *, actor: str, after_item_id: int | None = None,
                  peek: int = 0) -> tuple[dict | None, int, list[dict]]:
        """依優先順序的下一張待審（跳過別人鎖住的），並鎖給自己。回 (item, 剩幾張, 預載清單)。"""
        r = self._open(rid)
        after_rank = -1
        if after_item_id is not None:
            with self.db.cursor() as cur:
                cur.execute("SELECT rank FROM media_round_items WHERE round_id = %s AND item_id = %s",
                            (rid, after_item_id))
                x = cur.fetchone()
                after_rank = int(x[0]) if x else -1
        sql = """SELECT ri.item_id FROM media_round_items ri JOIN media_items i ON i.id = ri.item_id
                 WHERE ri.round_id = %s AND ri.state = 'pending' AND ri.rank > %s
                   AND (i.locked_by IS NULL OR i.locked_by = %s OR i.lock_until < now())
                 ORDER BY ri.rank LIMIT %s"""
        chosen = None
        with self.db.cursor() as cur:
            cur.execute("SELECT count(*) FROM media_round_items WHERE round_id = %s AND state = 'pending'", (rid,))
            remaining = int(cur.fetchone()[0])
        for start in (after_rank, -1):      # 往後找，到底從頭繞一圈
            with self.db.cursor() as cur:
                cur.execute(sql, (rid, start, actor, 20))
                cands = [int(x[0]) for x in cur.fetchall()]
            for iid in cands:
                try:
                    self.lib.lock(iid, actor=actor, seconds=r["config"]["review"]["lock_s"])
                    chosen = iid
                    break
                except Conflict:            # 剛好被別人搶走
                    continue
            if chosen is not None:
                break
        if chosen is None:
            return None, remaining, []
        item = self.item(chosen)
        peeks = []
        if peek:
            with self.db.cursor() as cur:
                cur.execute(sql, (rid, item["review"]["rank"], actor, peek))
                pid = [int(x[0]) for x in cur.fetchall()]
            for iid in pid:
                it = self.lib.item(iid)
                peeks.append({"id": iid, "image_url": it["image_url"]})
        return item, remaining, peeks

    def membership(self, item_id: int) -> dict | None:
        """這張圖所在的輪次：優先進行中的，否則最近的一輪。"""
        with self.db.cursor() as cur:
            cur.execute(
                """SELECT ri.*, r.status AS round_status, r.mode AS round_mode, r.config FROM media_round_items ri
                   JOIN media_review_rounds r ON r.id = ri.round_id WHERE ri.item_id = %s
                   ORDER BY (r.status = 'open') DESC, r.id DESC LIMIT 1""", (item_id,))
            return _row(cur)

    def item(self, item_id: int) -> dict:
        """lib.item ＋ review 區塊；框依 show_conf 過濾（抽查圖的框在建立輪次時已清空）。"""
        it = self.lib.item(item_id)
        m = self.membership(item_id)
        if m is None:
            it["review"] = None
            return it
        it["shapes"] = visible_shapes(it["shapes"] or [], m["config"])
        it["review"] = {"round_id": m["round_id"], "round_status": m["round_status"],
                        "round_mode": m.get("round_mode") or "review", "blind": m["blind"],
                        "priority": m["priority"], "reasons": m["reasons"], "rank": m["rank"],
                        "sequence_id": None, "state": m["state"], "doubt_note": m["doubt_note"],
                        "reviewed_by": m["reviewed_by"]}
        return it

    # 送出 ------------------------------------------------------------

    def review(self, item_id: int, *, actor: str, action: str, revision: int, shapes: list[dict],
               seconds_spent: float | None = None, clean_view_seen: bool | None = None,
               note: str | None = None) -> dict:
        m = self.membership(item_id)
        if m is None:
            raise ValueError("這張圖不在任何審核輪次裡（用 /items/{id}/submit）")
        if m["round_status"] != "open":
            raise Conflict("這一輪已經關閉，不能再審")
        if action == "doubt":
            if not (note or "").strip():
                raise ValueError("有疑問要寫備註（note）")
            with self.db.cursor() as cur:
                cur.execute("UPDATE media_round_items SET state = 'doubt', doubt_note = %s, reviewed_by = %s,"
                            " reviewed_at = now() WHERE round_id = %s AND item_id = %s",
                            (note.strip(), actor, m["round_id"], item_id))
            self.lib.unlock(item_id, actor=actor)
            it = self.lib.item(item_id, with_url=False)
            return {"item_id": item_id, "revision": it["revision"], "status": it["status"], "state": "doubt",
                    "diff": None}
        if action not in ("submit", "background"):
            raise ValueError("action 只能是 submit / background / doubt")
        status = "labeled" if action == "submit" else "background"
        shapes, _ = ensure_ids(list(shapes or []))
        baseline = [] if (m["blind"] or m.get("round_mode") == "seed") else (m["baseline"] or [])
        shapes = mark_sources(baseline, shapes)
        out = self.lib.submit(item_id, shapes, revision=revision, actor=actor, status=status)
        diff = shape_diff(baseline, shapes)
        with self.db.cursor() as cur:
            cur.execute(
                """UPDATE media_round_items SET state = 'done', reviewed_by = %s, reviewed_at = now(),
                       seconds_spent = %s, clean_view_seen = %s, diff = %s, doubt_note = NULL
                   WHERE round_id = %s AND item_id = %s""",
                (actor, seconds_spent, clean_view_seen, json.dumps(diff), m["round_id"], item_id))
        return {**out, "state": "done", "diff": diff}

    # 縮圖 ------------------------------------------------------------

    def thumb_url(self, sha256: str, px: int) -> str | None:
        """長邊 px 的 webp 縮圖網址；沒有就現做一張存起來（之後直接用）。"""
        blob = self.lib.blob(sha256)
        if blob is None:
            return None
        key = (blob.get("thumbs") or {}).get(str(px))
        if key is None:
            key = self._make_thumb(blob, px)
        return self.lib.objects.presigned_url(bucket=blob["bucket"], key=key)

    def _make_thumb(self, blob: dict, px: int) -> str:
        import io

        import cv2
        import numpy as np

        data = self.lib.objects.get(bucket=blob["bucket"], key=blob["object_key"])
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"blob {blob['sha256']} 解不開")
        h, w = img.shape[:2]
        s = px / max(h, w)
        if s < 1:
            img = cv2.resize(img, (max(1, round(w * s)), max(1, round(h * s))), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".webp", img, [cv2.IMWRITE_WEBP_QUALITY, 80])
        if not ok:
            raise ValueError("縮圖編碼失敗")
        sha = blob["sha256"]
        key = f"library/thumbs/{sha[:2]}/{sha}_{px}.webp"
        b = buf.tobytes()
        self.lib.objects.put(bucket=blob["bucket"], key=key, data=io.BytesIO(b), length=len(b),
                             content_type="image/webp")
        with self.db.cursor() as cur:
            cur.execute("UPDATE media_blobs SET thumbs = thumbs || %s WHERE sha256 = %s",
                        (json.dumps({str(px): key}), sha))
        return key


def _future(ts) -> bool:
    from datetime import datetime, timezone
    return ts > datetime.now(timezone.utc)
