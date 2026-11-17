"""標注專案：照配方一輪一輪訓練。

專案只管「訓練＋用新模型預標」，其他（加批次、審核、上線）都用現成功能，專案不碰：

    人按「跑下一輪」
      → 匯出（有改過的批次各自一個快照）→ 混合（依配方 → 訓練快照）→ 訓練 → 用新模型預標還沒人審的圖 → 閒置
    人用圖片庫加批次、用圖片集審核（要不要開審核輪次是人的事）→ 再按「跑下一輪」…

    人按「只預標」：用最新一代的模型預標還沒人審的圖（加了新批次、還不想訓練時用）

每一步都呼叫既有零件（匯出 / 混合 / 訓練 / 預標），這裡只負責「依序呼叫、記錄走到哪」。
不會自己開始下一輪：調和迴圈（CampaignEngine.reconcile_all）只追蹤人按下去之後正在跑的任務，重啟後從原地接著追。

    state  idle（閒置，等人按）/ running（這一輪進行中）/ failed（停在這一步，retry 從這一步重跑）/
           closed（結案：圖片集釋放給別的專案；reopen 可接回）
    stage  running / failed 時在哪一步（exporting / mixing / training / prelabeling）；閒置 = null
    step   這一步內部進行到哪（task id 等）；step.mode = run（跑下一輪）/ prelabel（只預標）
"""

from __future__ import annotations

import copy
import json
import logging
import threading
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal, Protocol

import yaml
from pydantic import BaseModel, ConfigDict, Field

from edgevision.media.library import Conflict, NotFound, _row, _rows
from edgevision.media.review import deep_merge, prune_none

logger = logging.getLogger(__name__)

DEFAULTS_PATH = Path(__file__).with_name("campaign_defaults.yaml")
STAGES = ("exporting", "mixing", "training", "prelabeling")
RUN_FLOW = {"exporting": "mixing", "mixing": "training", "training": "prelabeling", "prelabeling": None}
MODES = ("run", "prelabel")
SYSTEM_ACTOR = "edgevision-campaign"
TASK_DONE = ("completed",)
TASK_BAD = ("failed", "cancelled", "rejected")
RECIPE_KEYS = ("repeat", "choose", "proportion")
PRELABEL_STATUS = ["unlabeled", "auto"]          # 還沒人審的圖（預標過的會被新模型重新預標）
# 舊版（v0.2）欄位：資料庫還在、不再寫，輸出時拿掉
LEGACY_COLS = ("waiting_reason", "waiting_detail", "round_id", "pending_source", "golden_sync")


# ── 設定（只定義欄位與範圍；預設全在 campaign_defaults.yaml）────────────────

class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DatasetCfg(_Strict):
    # 前綴；實際 repo 是 <前綴>-batch-<批次> / <前綴>-train，所以比 LakeFS 的 63 字上限再留 13 字
    repo: str = Field(pattern=r"^([a-z0-9][a-z0-9-]{2,49})?$", description="小寫英數與 -，3–50 字；空 = campaign-<專案 id>")
    val_pct: int = Field(ge=0, le=100)
    task: Literal["detect", "segment", "pose"] | None = Field(
        None, description="批次快照的標注格式＝訓練的 task；空 = 建專案時從初始模型帶（帶不到 = detect）")


class PrelabelCfg(_Strict):
    conf: float = Field(gt=0, lt=1)


class CampaignDefaults(_Strict):
    reconcile_s: int = Field(ge=5, le=3600)
    dataset: DatasetCfg
    prelabel: PrelabelCfg
    train: dict[str, Any]
    export: dict[str, Any]
    iteration_gates: dict[str, Any]


@lru_cache(maxsize=1)
def load_defaults(path: str | None = None) -> dict:
    raw = yaml.safe_load(Path(path or DEFAULTS_PATH).read_text(encoding="utf-8"))
    return CampaignDefaults.model_validate(raw).model_dump()


def resolve_config(override: dict | None, *, defaults: dict | None = None) -> dict:
    """預設 ← 專案覆蓋，整份驗證（欄位打錯 / 超出範圍 → ValueError）。"""
    merged = deep_merge(copy.deepcopy(defaults or load_defaults()), override or {})
    return CampaignDefaults.model_validate(merged).model_dump()


# ── 純邏輯 ────────────────────────────────────────────────────

def merge_recipe(cur: dict | None, patch: dict) -> dict:
    """配方覆蓋：{"batches": {"<batch_id>": {"repeat": 0.5}}, "total": …, "seed": …}；值給 null = 回到預設。"""
    out = copy.deepcopy(cur or {})
    for k, v in patch.items():
        if k == "batches":
            b = out.setdefault("batches", {})
            for bid, spec in (v or {}).items():
                if spec is None:
                    b.pop(str(bid), None)
                    continue
                spec = {kk: vv for kk, vv in spec.items() if vv is not None}
                if set(spec) - set(RECIPE_KEYS) or len(spec) > 1:
                    raise ValueError(f"批次 {bid} 的配方只能給 {RECIPE_KEYS} 其中一個")
                if any(float(x) < 0 for x in spec.values()):
                    raise ValueError(f"批次 {bid} 的配方不能是負數")
                b[str(bid)] = spec
        elif k in ("total", "seed"):
            if v is None:
                out.pop(k, None)
            else:
                out[k] = int(v)
        else:
            raise ValueError(f"配方沒有 {k!r} 這個欄位（batches / total / seed）")
    return out


def recipe_sources(snapshots: dict, recipe: dict | None) -> list[dict]:
    """批次快照＋配方覆蓋 → 混合的 sources（每批預設 repeat 1 = 全部取一次）。"""
    over = (recipe or {}).get("batches") or {}
    return [{"ref": s["ref"], "batch_id": int(bid), **over.get(str(bid), {})}
            for bid, s in sorted(snapshots.items(), key=lambda kv: int(kv[0]))]


def transient_db_error(exc: BaseException) -> bool:
    """暫時性資料庫錯誤：連線斷了，或部署時 migrate_db 加欄位、舊程序的 SELECT * 快取計畫失效
    （cached plan must not change result type，SQLSTATE 0A000；10/15 部署 v0.2 時真的把 siteb12 專案標成失敗）。"""
    import psycopg
    return isinstance(exc, (psycopg.OperationalError, psycopg.errors.FeatureNotSupported))


def _public(c: dict) -> dict:
    return {k: v for k, v in c.items() if k not in LEGACY_COLS}


# ── 動作介面（control-api 注入真的；測試注入假的）─────────────────

class Actions(Protocol):
    def autolabel(self, cid: int, params: dict, actor: str) -> str: ...
    def export(self, cid: int, *, repo: str, message: str, val_pct: int, actor: str,
               batch_ids: list[int] | None = None, task: str | None = None) -> str: ...
    def mix(self, params: dict, actor: str) -> str: ...
    def submit_training(self, body: dict, actor: str) -> str: ...
    def task(self, task_id: str) -> dict | None: ...


# ── 資料層＋推進 ───────────────────────────────────────────────

class CampaignEngine:
    def __init__(self, lib, actions: Actions, *, audit=None) -> None:
        self.lib = lib
        self.db = lib.db
        self.actions = actions
        self.audit = audit
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # 讀取 ---------------------------------------------------------------

    def get(self, cid: int) -> dict:
        with self.db.cursor() as cur:
            cur.execute("SELECT * FROM media_campaigns WHERE id = %s", (cid,))
            c = _row(cur)
        if c is None:
            raise NotFound(f"標注專案 {cid} 不存在")
        return c

    def config(self, c: dict) -> dict:
        return resolve_config(c.get("config") or {})

    def detail(self, cid: int) -> dict:
        c = self.get(cid)
        with self.db.cursor() as cur:
            cur.execute("SELECT * FROM media_campaign_iterations WHERE campaign_id = %s ORDER BY n", (cid,))
            its = _rows(cur)
            cur.execute("SELECT id, at, actor, kind, stage, detail FROM media_campaign_events WHERE campaign_id = %s"
                        " ORDER BY id DESC LIMIT 30", (cid,))
            evs = _rows(cur)
        return {**_public(c), **self._roles(c, with_batches=True), "iterations": its, "events": evs,
                "resolved_config": self.config(c), "latest_model": self._model_ref(cid)}

    def _roles(self, c: dict, *, with_batches: bool = False) -> dict:
        """各角色的實體：圖片集、測試集圖片集（只是參照）、初始批次＋初始模型、（詳情才帶）所有批次＋快照。"""
        def col(cid):
            if cid is None:
                return None
            try:
                x = self.lib.collection(cid)
            except NotFound:
                return None
            p = x["progress"]
            return {"id": x["id"], "name": x["name"], "n_items": p["total"],
                    "n_done": p.get("labeled", 0) + p.get("background", 0)}
        snaps = c.get("batch_snapshots") or {}
        out = {"review_collection": col(c["collection_id"]), "golden_collection": col(c.get("golden_collection_id")),
               "initial": {"batch_ids": c.get("initial_batch_ids") or [], "model": c.get("initial_model")}}
        if with_batches:
            try:
                bs = self.lib.batches(c["collection_id"])
            except NotFound:
                bs = []
            init = set(c.get("initial_batch_ids") or [])
            out["batches"] = [{**b, "initial": b["id"] in init, "snapshot": snaps.get(str(b["id"]))} for b in bs]
        return out

    def list(self) -> list[dict]:
        with self.db.cursor() as cur:
            cur.execute("SELECT * FROM media_campaigns ORDER BY id DESC")
            cs = _rows(cur)
        return [{**_public(c), **self._roles(c)} for c in cs]

    def owner_of_collection(self, collection_id: int) -> dict | None:
        """這個圖片集屬於哪個專案、扮演什麼角色（review / golden）；都不是 → None。"""
        with self.db.cursor() as cur:
            cur.execute("SELECT id, name, collection_id, golden_collection_id FROM media_campaigns"
                        " WHERE collection_id = %s OR golden_collection_id = %s"
                        " ORDER BY state = 'closed', id DESC LIMIT 1",
                        (collection_id, collection_id))
            r = _row(cur)
        if r is None:
            return None
        return {"campaign_id": r["id"], "campaign_name": r["name"],
                "role": "review" if r["collection_id"] == collection_id else "golden"}

    # 寫入 ---------------------------------------------------------------

    _JSON_COLS = ("step", "config", "initial_batch_ids", "initial_model", "recipe", "batch_snapshots")
    _ITER_JSON = ("metrics", "batch_snapshots", "recipe", "prelabel_task_ids")

    def _event(self, cid: int, kind: str, *, actor: str = SYSTEM_ACTOR, stage: str | None = None,
               detail: dict | None = None) -> None:
        with self.db.cursor() as cur:
            cur.execute("INSERT INTO media_campaign_events (campaign_id, actor, kind, stage, detail)"
                        " VALUES (%s,%s,%s,%s,%s)", (cid, actor, kind, stage, json.dumps(detail or {}, default=str)))
        logger.info("標注專案 %s：%s %s %s", cid, kind, stage or "", detail or "")

    def _set(self, cid: int, **f) -> None:
        sets, params = [], []
        for k, v in f.items():
            sets.append(f"{k} = %s")
            params.append(json.dumps(v, default=str) if k in self._JSON_COLS and v is not None else v)
        with self.db.cursor() as cur:
            cur.execute(f"UPDATE media_campaigns SET {', '.join(sets)}, updated_at = now() WHERE id = %s",
                        (*params, cid))

    def _iter_upsert(self, cid: int, n: int, **f) -> None:
        with self.db.cursor() as cur:
            cur.execute("INSERT INTO media_campaign_iterations (campaign_id, n) VALUES (%s,%s)"
                        " ON CONFLICT (campaign_id, n) DO NOTHING", (cid, n))
            if f:
                sets = ", ".join(f"{k} = %s" for k in f)
                vals = [json.dumps(v, default=str) if k in self._ITER_JSON and v is not None else v
                        for k, v in f.items()]
                cur.execute(f"UPDATE media_campaign_iterations SET {sets} WHERE campaign_id = %s AND n = %s",
                            (*vals, cid, n))

    def _iteration(self, cid: int, n: int) -> dict | None:
        with self.db.cursor() as cur:
            cur.execute("SELECT * FROM media_campaign_iterations WHERE campaign_id = %s AND n = %s", (cid, n))
            return _row(cur)

    @staticmethod
    def _check_model(m: dict | None) -> dict:
        if not m or not m.get("name") or m.get("version") is None:
            raise ValueError("initial_model 要給 {name, version}（Registry 的模型與版本）")
        return {"name": str(m["name"]), "version": int(m["version"])}

    def _check_golden_collection(self, collection_id: int | None, golden_collection_id: int | None,
                                 *, cid: int | None = None) -> None:
        if golden_collection_id is None:
            return
        if golden_collection_id == collection_id:
            raise ValueError("測試集不能跟圖片集是同一個")
        self.lib.collection(golden_collection_id)
        with self.db.cursor() as cur:
            cur.execute("SELECT id, name FROM media_campaigns WHERE collection_id = %s AND id IS DISTINCT FROM %s"
                        " AND state <> 'closed'",
                        (golden_collection_id, cid))
            r = _row(cur)
        if r:
            raise Conflict(f"圖片集 {golden_collection_id} 是專案 {r['id']}「{r['name']}」的圖片集，"
                           "不能拿來當測試集（訓練圖會混進測試集）")

    def _check_collection_free(self, collection_id: int, *, cid: int | None = None) -> None:
        """一個圖片集只給一個專案用：兩個專案一起預標、匯出同一批圖會互相蓋掉；也不能是別的專案的測試集。
        結案的專案不算。"""
        with self.db.cursor() as cur:
            cur.execute("SELECT id, name, collection_id FROM media_campaigns"
                        " WHERE (collection_id = %s OR golden_collection_id = %s) AND state <> 'closed'"
                        " AND id IS DISTINCT FROM %s ORDER BY id LIMIT 1",
                        (collection_id, collection_id, cid))
            r = _row(cur)
        if r:
            role = "圖片集" if r["collection_id"] == collection_id else "測試集"
            raise Conflict(f"圖片集 {collection_id} 已是專案 {r['id']}「{r['name']}」的{role}；"
                           "一個圖片集只給一個進行中的專案用（舊專案不用了先結案；不給 collection_id 會自動建一個新的）")

    def _collection_of_batches(self, batch_ids: list[int]) -> int:
        ids = sorted({int(b) for b in batch_ids})
        with self.db.cursor() as cur:
            cur.execute("SELECT id, collection_id FROM media_batches WHERE id = ANY(%s)", (ids,))
            found = {int(r[0]): int(r[1]) for r in cur.fetchall()}
        missing = [b for b in ids if b not in found]
        if missing:
            raise NotFound(f"批次 {missing} 不存在")
        cols = sorted(set(found.values()))
        if len(cols) > 1:
            raise ValueError(f"初始批次分屬不同圖片集 {cols}；一個專案只用一個圖片集")
        return cols[0]

    def resolve_collection(self, *, collection_id: int | None, initial_batch_ids: list[int] | None) -> int | None:
        """專案用哪個圖片集：明講 → 它；沒講但給了初始批次 → 批次所在的圖片集；都沒有 → None（要自動建一個）。"""
        if collection_id is not None:
            self.lib.collection(collection_id)
            return collection_id
        if initial_batch_ids:
            return self._collection_of_batches(initial_batch_ids)
        return None

    def create(self, *, name: str, model_name: str, base_model: str, actor: str,
               initial_model: dict, collection_id: int | None = None, new_collection: dict | None = None,
               initial_batch_ids: list[int] | None = None, golden_ref: str | None = None,
               golden_collection_id: int | None = None, config: dict | None = None,
               recipe: dict | None = None) -> dict:
        """建專案（全部是引用），建好是閒置：人按「跑下一輪」或「只預標」才會動。

        圖片集三種來源：collection_id 明講；只給 initial_batch_ids → 批次所在的圖片集；
        都沒給 → 依 new_collection（名稱、task、類別等，API 層從初始模型帶）自動建一個空的，之後用圖片庫的匯入加批次。
        """
        init_model = self._check_model(initial_model)
        collection_id = self.resolve_collection(collection_id=collection_id, initial_batch_ids=initial_batch_ids)
        if collection_id is None:
            if not new_collection:
                raise ValueError("要給 collection_id 或 initial_batch_ids，或讓後端自動建圖片集（new_collection）")
            if recipe and (recipe.get("batches") or {}):
                raise ValueError("新圖片集還沒有批次，recipe 先不要指定批次")
            self._check_golden_collection(None, golden_collection_id)
        else:
            self._check_collection_free(collection_id)
            self._check_golden_collection(collection_id, golden_collection_id)
        init_batches = (self.lib.check_batches(collection_id, initial_batch_ids) or []) if collection_id else []
        cfg_over = prune_none(config or {})
        resolve_config(cfg_over)                       # 驗證
        rec = merge_recipe(None, recipe or {})
        if collection_id:
            self.lib.check_batches(collection_id, [int(b) for b in rec.get("batches", {})])
        with self.db.cursor() as cur:
            cur.execute("SELECT 1 FROM media_campaigns WHERE name = %s", (name,))
            if cur.fetchone():
                raise Conflict(f"專案名稱 {name!r} 已存在")
        created_collection = None
        if collection_id is None:                      # 其他檢查都過了才建，失敗不留空的圖片集
            created_collection = self.lib.create_collection(actor=actor, **new_collection)
            collection_id = int(created_collection["id"])
        with self.db.cursor() as cur:
            cur.execute(
                """INSERT INTO media_campaigns (name, collection_id, model_name, base_model, golden_ref,
                       golden_collection_id, initial_batch_ids, initial_model, recipe, config, stage, state,
                       iteration, created_by)
                   VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,NULL,'idle',1,%s) RETURNING id""",
                (name, collection_id, model_name, base_model, golden_ref, golden_collection_id,
                 json.dumps(init_batches), json.dumps(init_model), json.dumps(rec), json.dumps(cfg_over), actor))
            cid = int(cur.fetchone()[0])
        self._iter_upsert(cid, 0, model_name=init_model["name"], model_version=init_model["version"],
                          note="初始人工標注＋初始模型")
        self._event(cid, "created", actor=actor,
                    detail={"initial_model": init_model, "initial_batch_ids": init_batches,
                            "collection_id": collection_id, "collection_created": created_collection is not None})
        return self.detail(cid)

    def update(self, cid: int, *, actor: str, golden_ref: str | None = None, clear_golden: bool = False,
               golden_collection_id: int | None = None, clear_golden_collection: bool = False,
               config: dict | None = None, recipe: dict | None = None,
               initial_batch_ids: list[int] | None = None, initial_model: dict | None = None) -> dict:
        c = self.get(cid)
        self._not_closed(c)
        f: dict[str, Any] = {}
        if clear_golden:
            f["golden_ref"] = None
        elif golden_ref is not None:
            f["golden_ref"] = golden_ref
        if clear_golden_collection:
            f["golden_collection_id"] = None
        elif golden_collection_id is not None and golden_collection_id != c.get("golden_collection_id"):
            self._check_golden_collection(c["collection_id"], golden_collection_id, cid=cid)
            f["golden_collection_id"] = golden_collection_id
        if config is not None:
            after = prune_none(deep_merge(c.get("config") or {}, config))
            resolve_config(after)
            f["config"] = after
        if recipe is not None:
            rec = merge_recipe(c.get("recipe"), recipe)
            self.lib.check_batches(c["collection_id"], [int(b) for b in rec.get("batches", {})])
            f["recipe"] = rec
        if initial_batch_ids is not None:
            f["initial_batch_ids"] = self.lib.check_batches(c["collection_id"], initial_batch_ids) or []
        if initial_model is not None:
            f["initial_model"] = self._check_model(initial_model)
            self._iter_upsert(cid, 0, model_name=f["initial_model"]["name"],
                              model_version=f["initial_model"]["version"])
        if f:
            self._set(cid, **f)
            self._event(cid, "updated", actor=actor, stage=c["stage"], detail={k: v for k, v in f.items()})
        return self.detail(cid)

    # 人的動作 -------------------------------------------------------------

    def _start(self, cid: int, *, actor: str, mode: str) -> dict:
        c = self.get(cid)
        self._not_closed(c)
        if c["state"] != "idle":
            raise Conflict("這個專案正在跑（或失敗待重試），閒置時才能開始", state=c["state"], stage=c["stage"])
        first = "exporting" if mode == "run" else "prelabeling"
        self._set(cid, state="running", stage=first, step={"mode": mode}, last_error=None)
        self._event(cid, "run" if mode == "run" else "prelabel", actor=actor, stage=first,
                    detail={"iteration": c["iteration"]} if mode == "run" else {"model": self._model_ref(cid)})
        self.tick(cid)
        return self.detail(cid)

    def run(self, cid: int, *, actor: str) -> dict:
        """跑下一輪：匯出 → 混合 → 訓練 → 用新模型預標 → 閒置（記成一代）。"""
        return self._start(cid, actor=actor, mode="run")

    def prelabel(self, cid: int, *, actor: str) -> dict:
        """只預標：用最新一代的模型預標還沒人審的圖。"""
        return self._start(cid, actor=actor, mode="prelabel")

    def retry(self, cid: int, *, actor: str) -> dict:
        c = self.get(cid)
        if c["state"] != "failed":
            raise Conflict("只有失敗的專案能重試", state=c["state"])
        self._set(cid, state="running", step={"mode": self._mode(c)}, last_error=None)
        self._event(cid, "retry", actor=actor, stage=c["stage"])
        self.tick(cid)
        return self.detail(cid)

    @staticmethod
    def _not_closed(c: dict) -> None:
        if c["state"] == "closed":
            raise Conflict("專案已結案（要改先 reopen）", state="closed")

    def close(self, cid: int, *, actor: str) -> dict:
        """結案：圖片集釋放給別的專案。圖片集、批次、快照、模型、每一代的紀錄都留著。跑到一半不能結案。"""
        c = self.get(cid)
        self._not_closed(c)
        if c["state"] == "running":
            raise Conflict("這一輪還在跑，跑完（或失敗）再結案", state=c["state"], stage=c["stage"])
        self._set(cid, state="closed", stage=None, step={})
        self._event(cid, "closed", actor=actor, detail={"state": c["state"], "stage": c["stage"]})
        return self.detail(cid)

    def reopen(self, cid: int, *, actor: str) -> dict:
        """接回結案的專案（回到閒置）；圖片集這期間被別的專案拿走 → 409。"""
        c = self.get(cid)
        if c["state"] != "closed":
            raise Conflict("不是結案狀態", state=c["state"])
        self._check_collection_free(c["collection_id"], cid=cid)
        self._check_golden_collection(c["collection_id"], c.get("golden_collection_id"), cid=cid)
        self._set(cid, state="idle", stage=None, step={}, last_error=None)
        self._event(cid, "reopened", actor=actor)
        return self.detail(cid)

    # 推進 ---------------------------------------------------------------

    def reconcile_all(self) -> None:
        """所有進行中的專案各推一步。用 advisory lock 確保多副本時只有一個在推。"""
        with self.db.cursor() as cur:
            cur.execute("SELECT pg_try_advisory_lock(hashtext('media_campaign_reconcile'))")
            got = cur.fetchone()[0]
            if not got:
                return
            try:
                cur.execute("SELECT id FROM media_campaigns WHERE state = 'running' ORDER BY id")
                ids = [int(r[0]) for r in cur.fetchall()]
                for cid in ids:
                    try:
                        self.tick(cid)
                    except Exception as exc:          # 一個專案壞掉不影響其他
                        if transient_db_error(exc):   # 下一輪調和自然恢復，不要把專案標失敗
                            logger.warning("標注專案 %s 這輪略過（暫時性資料庫錯誤）：%s", cid, exc)
                            continue
                        logger.exception("標注專案 %s 推進失敗", cid)
                        self._fail(cid, f"內部錯誤：{exc}")
            finally:
                cur.execute("SELECT pg_advisory_unlock(hashtext('media_campaign_reconcile'))")

    def tick(self, cid: int) -> None:
        """推一步（同一個專案重複呼叫是安全的：每一步都先看 step 記錄走到哪）。"""
        for _ in range(len(STAGES) * 2):               # 同一次最多連走幾個不用等的步驟
            c = self.get(cid)
            if c["state"] != "running":
                return
            if c["stage"] not in STAGES:
                self._fail(cid, f"步驟 {c['stage']!r} 已不存在（舊版專案）")
                return
            before = (c["stage"], c["state"], json.dumps(c["step"], sort_keys=True, default=str))
            getattr(self, f"_stage_{c['stage']}")(c, self.config(c))
            c2 = self.get(cid)
            after = (c2["stage"], c2["state"], json.dumps(c2["step"], sort_keys=True, default=str))
            if after == before:
                return

    @staticmethod
    def _mode(c: dict) -> str:
        """step.mode；沒記（v0.2 留下來的預標）= 只預標。"""
        m = (c.get("step") or {}).get("mode")
        return m if m in MODES else "prelabel"

    def _fail(self, cid: int, msg: str) -> None:
        c = self.get(cid)
        self._set(cid, state="failed", last_error=msg[:2000])
        self._event(cid, "failed", stage=c["stage"], detail={"error": msg[:500]})

    def _done_step(self, c: dict, **extra) -> None:
        """這一步做完：跑下一輪 → 接下一步，最後一步做完記成一代、回閒置；只預標 → 回閒置。"""
        mode = self._mode(c)
        nxt = RUN_FLOW[c["stage"]] if mode == "run" else None
        if nxt:
            self._set(c["id"], stage=nxt, step={"mode": mode}, **extra)
            self._event(c["id"], "advanced", stage=nxt, detail={"from": c["stage"]})
            return
        f: dict[str, Any] = {"state": "idle", "stage": None, "step": {}, **extra}
        if mode == "run":
            f["iteration"] = c["iteration"] + 1
        self._set(c["id"], **f)
        self._event(c["id"], "iteration_done" if mode == "run" else "prelabel_done",
                    detail={"iteration": c["iteration"]} if mode == "run" else {})

    def _task_state(self, tid: str) -> tuple[str, dict | None]:
        t = self.actions.task(tid)
        if t is None:
            return "missing", None
        st = t.get("status")
        if st in TASK_DONE:
            return "done", t
        if st in TASK_BAD:
            return "bad", t
        return "running", t

    def _wait_task(self, c: dict, key: str) -> dict | None:
        """step[key] 的任務：完成回 task；還在跑回 None；失敗就把專案標失敗並回 None。"""
        st, t = self._task_state(c["step"][key])
        if st == "running":
            return None
        if st in ("bad", "missing"):
            self._fail(c["id"], f"{key} {c['step'][key]}：{self._task_error(t)}")
            return None
        return t

    @staticmethod
    def _task_error(t: dict | None) -> str:
        return (t or {}).get("error") or (t or {}).get("status_message") or "任務不見了（control-api 重啟？）"

    def _repo(self, c: dict, cfg: dict) -> str:
        return cfg["dataset"]["repo"] or f"campaign-{c['id']}"

    def _latest(self, cid: int) -> tuple[int, str, int] | None:
        """最新一代有模型的：(代, 模型名, 版本)。"""
        c = self.get(cid)
        with self.db.cursor() as cur:
            cur.execute("SELECT n, model_name, model_version FROM media_campaign_iterations WHERE campaign_id = %s"
                        " AND model_version IS NOT NULL ORDER BY n DESC LIMIT 1", (cid,))
            r = cur.fetchone()
        if r:
            return int(r[0]), (r[1] or c["model_name"]), int(r[2])
        m = c.get("initial_model")
        return (0, m["name"], int(m["version"])) if m else None

    def _model_ref(self, cid: int) -> dict | None:
        """最新一代的模型（「只預標」用它）。"""
        x = self._latest(cid)
        return {"iteration": x[0], "name": x[1], "version": x[2]} if x else None

    def _n_unreviewed(self, collection_id: int) -> int:
        with self.db.cursor() as cur:
            cur.execute("SELECT count(*) FROM media_items WHERE collection_id = %s AND status = ANY(%s)",
                        (collection_id, PRELABEL_STATUS))
            return int(cur.fetchone()[0])

    def _batch_signatures(self, collection_id: int) -> dict[str, str]:
        """每個有已確認圖的批次：「張數:revision 加總:最大 id」—— 標注有變才會不一樣。"""
        with self.db.cursor() as cur:
            cur.execute("""SELECT batch_id, count(*), coalesce(sum(revision), 0), max(id) FROM media_items
                           WHERE collection_id = %s AND batch_id IS NOT NULL AND status IN ('labeled', 'background')
                           GROUP BY batch_id""", (collection_id,))
            return {str(r[0]): f"{r[1]}:{r[2]}:{r[3]}" for r in cur.fetchall()}

    @staticmethod
    def _label_task(c: dict, cfg: dict) -> str:
        """批次快照的標注格式＝訓練的 task：專案設定（建專案時從初始模型帶）；沒有 = detect。"""
        return cfg["dataset"].get("task") or "detect"

    # 各步驟 -------------------------------------------------------------

    def _stage_exporting(self, c: dict, cfg: dict) -> None:
        """有新確認的批次各自匯出成批次快照（每批一個 LakeFS repo，可以同時跑）。
        task 換了 → 每一批都用新格式重匯，混合時不會一半框一半多邊形。"""
        n, step = c["iteration"], c["step"]
        snaps = dict(c.get("batch_snapshots") or {})
        task = self._label_task(c, cfg)
        if "exports" not in step:
            sigs = self._batch_signatures(c["collection_id"])
            if not sigs:
                self._fail(c["id"], "圖片集裡沒有任何已確認的圖，無法匯出")
                return
            exports = {}
            for bid, sig in sorted(sigs.items(), key=lambda kv: int(kv[0])):
                old = snaps.get(bid) or {}
                if old.get("sig") == sig and old.get("task", task) == task:      # 舊快照沒記 task = 當時的預設
                    continue
                b = self.lib.batch(int(bid))
                repo = f"{self._repo(c, cfg)}-batch-{bid}"
                tid = self.actions.export(c["collection_id"], repo=repo, val_pct=cfg["dataset"]["val_pct"],
                                          batch_ids=[int(bid)], actor=SYSTEM_ACTOR, task=task,
                                          message=f"標注專案 {c['name']} 第 {n} 代：批次「{b['name']}」（{bid}）的已確認圖")
                exports[bid] = {"task": tid, "sig": sig, "repo": repo, "label_task": task}
            self._set(c["id"], step={**step, "exports": exports})
            self._event(c["id"], "export_started", stage="exporting",
                        detail={"batches": {k: v["task"] for k, v in exports.items()},
                                "unchanged": sorted(set(sigs) - set(exports), key=int)})
            return
        for bid, e in step["exports"].items():
            st, t = self._task_state(e["task"])
            if st == "running":
                return
            if st != "done":
                self._fail(c["id"], f"批次 {bid} 匯出 {e['task']}：{self._task_error(t)}")
                return
            ref = ((t.get("result") or {}).get("export") or {}).get("dataset_ref")
            if not ref:
                self._fail(c["id"], f"批次 {bid} 匯出 {e['task']} 沒有回 dataset_ref")
                return
            snaps[bid] = {"ref": ref, "sig": e["sig"], "task_id": e["task"], "iteration": n,
                          "task": e.get("label_task", task)}
        self._iter_upsert(c["id"], n, batch_snapshots=snaps)
        self._done_step(c, batch_snapshots=snaps)

    def _stage_mixing(self, c: dict, cfg: dict) -> None:
        """批次快照＋配方 → 訓練快照（預設每批全部取一次；PATCH recipe 調比例）。"""
        n, step = c["iteration"], c["step"]
        if "task" not in step:
            from edgevision.training.mix import validate_recipe
            sources = recipe_sources(c.get("batch_snapshots") or {}, c.get("recipe"))
            rec = c.get("recipe") or {}
            try:
                validate_recipe(sources, rec.get("total"))
            except ValueError as exc:
                self._fail(c["id"], f"配方不合法：{exc}（PATCH recipe 修正後 retry）")
                return
            desc = "、".join(f"批次 {s['batch_id']}" + "".join(f" {k}={s[k]}" for k in RECIPE_KEYS if k in s)
                            for s in sources)
            params = {"repo": f"{self._repo(c, cfg)}-train",
                      "sources": [{k: v for k, v in s.items() if k != "batch_id"} for s in sources],
                      "total": rec.get("total"), "seed": rec.get("seed", 0),
                      "message": f"標注專案 {c['name']} 第 {n} 代訓練快照：{desc}"}
            tid = self.actions.mix(params, SYSTEM_ACTOR)
            self._iter_upsert(c["id"], n, recipe={"sources": sources, "total": rec.get("total"),
                                                  "seed": rec.get("seed", 0)}, mix_task_id=tid)
            self._set(c["id"], step={**step, "task": tid})
            self._event(c["id"], "mix_started", stage="mixing", detail={"task_id": tid, "sources": sources})
            return
        t = self._wait_task(c, "task")
        if t is None:
            return
        ref = ((t.get("result") or {}).get("mix") or {}).get("dataset_ref")
        if not ref:
            self._fail(c["id"], f"混合任務 {step['task']} 沒有回 dataset_ref")
            return
        self._iter_upsert(c["id"], n, dataset_ref=ref)
        self._done_step(c)

    def _stage_training(self, c: dict, cfg: dict) -> None:
        n, step = c["iteration"], c["step"]
        if "train_task" not in step:
            ds = (self._iteration(c["id"], n) or {}).get("dataset_ref")
            if not ds:
                self._fail(c["id"], f"第 {n} 代沒有訓練快照")
                return
            body = {
                "model_name": c["model_name"], "dataset_ref": ds, "golden_ref": c["golden_ref"],
                "notes": f"標注專案 {c['name']}（id {c['id']}）第 {n} 代：訓練快照 {ds}（配方見該快照 metadata.mix），"
                         f"從 {c['base_model']} 重新訓練。用途：下一批預標；上線另由人判斷。",
                "config": {"model_name": c["model_name"], "task": self._label_task(c, cfg),
                           "base_model": c["base_model"],
                           "dataset": ds, "train": cfg["train"], "export": cfg["export"],
                           "class_names": self.lib.collection(c["collection_id"])["class_names"]},
                "gates": cfg["iteration_gates"], "config_source": f"campaign:{c['id']}",
            }
            tid = self.actions.submit_training(body, SYSTEM_ACTOR)
            self._iter_upsert(c["id"], n, train_task_id=tid, golden_ref=c["golden_ref"], model_name=c["model_name"])
            self._set(c["id"], step={**step, "train_task": tid})
            self._event(c["id"], "training_started", stage="training", detail={"task_id": tid, "dataset_ref": ds})
            return
        t = self._wait_task(c, "train_task")
        if t is None:
            return
        mv = t.get("model_version")
        metrics = ((t.get("result") or {}).get("trainer") or {}).get("metrics") or {}
        if not mv:
            self._fail(c["id"], f"訓練任務 {step['train_task']} 完成但沒有模型版本")
            return
        self._iter_upsert(c["id"], n, model_version=int(mv), metrics=metrics)
        self._event(c["id"], "trained", stage="training", detail={"model_version": int(mv)})
        self._done_step(c)

    def _stage_prelabeling(self, c: dict, cfg: dict) -> None:
        """用最新一代的模型（跑下一輪 = 剛訓練好的那個）預標圖片集裡還沒人審的圖；沒有就略過。"""
        step = c["step"]
        if "task" not in step:
            latest = self._latest(c["id"])
            if latest is None:
                self._fail(c["id"], "沒有任何模型可以預標（initial_model 沒設？）")
                return
            n_model, name, version = latest
            n_todo = self._n_unreviewed(c["collection_id"])
            if n_todo == 0:
                self._event(c["id"], "prelabel_skipped", stage="prelabeling",
                            detail={"reason": "圖片集裡沒有還沒審的圖", "model": {"name": name, "version": version}})
                self._done_step(c)
                return
            params = {"engine": "registry", "model_name": name, "model_version": version,
                      "conf": cfg["prelabel"]["conf"], "status": PRELABEL_STATUS}
            tid = self.actions.autolabel(c["collection_id"], params, SYSTEM_ACTOR)
            it = self._iteration(c["id"], n_model) or {}
            self._iter_upsert(c["id"], n_model, prelabel_task_id=tid,
                              prelabel_task_ids=[*(it.get("prelabel_task_ids") or []), tid])
            self._set(c["id"], step={**step, "task": tid, "model_iteration": n_model})
            self._event(c["id"], "prelabel_started", stage="prelabeling",
                        detail={"task_id": tid, "n_items": n_todo, "model_iteration": n_model, **params})
            return
        if self._wait_task(c, "task") is None:
            return
        self._done_step(c)

    # 背景迴圈 -----------------------------------------------------------

    def start_background(self) -> None:
        if self._thread is not None:
            return

        def loop():
            while not self._stop.is_set():
                try:
                    self.reconcile_all()
                except Exception:
                    logger.exception("標注專案調和迴圈出錯")
                self._stop.wait(load_defaults()["reconcile_s"])

        self._thread = threading.Thread(target=loop, name="campaign-reconciler", daemon=True)
        self._thread.start()

    def shutdown(self) -> None:
        self._stop.set()
