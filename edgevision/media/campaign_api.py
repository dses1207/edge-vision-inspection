"""標注專案 API，掛在 /api/media 底下，由 api.build_router 呼叫。

    POST /campaigns、GET /campaigns、GET /campaigns/{id}、PATCH /campaigns/{id}
    POST /campaigns/{id}/run | prelabel | retry | close | reopen

專案只引用圖片庫（圖片集、批次）、LakeFS（測試集、快照）、Registry（模型）；圖要加進來就用圖片庫的匯入、
審核用圖片集審核，專案不管。人按 run（跑下一輪）/ prelabel（只預標）才會動。
"""

from __future__ import annotations

import logging
from typing import Any, Callable

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from edgevision.media import schemas as S
from edgevision.media.campaign import CampaignEngine
from edgevision.media.shapes import LABEL_TASKS

logger = logging.getLogger(__name__)

SYSTEM_IDENT = ("edgevision-campaign", "system", "api")


class StateActions:
    """把 control-api 既有的動作（同一套驗證、任務、稽核）接給 CampaignEngine。"""

    def __init__(self, state: dict) -> None:
        self.state = state

    def _ma(self) -> dict:
        return self.state["media_actions"]

    def autolabel(self, cid: int, params: dict, actor: str) -> str:
        m = self._ma()
        return m["autolabel"](SYSTEM_IDENT, cid, m["models"]["AutolabelIn"].model_validate(params))

    def export(self, cid: int, *, repo: str, message: str, val_pct: int, actor: str,
               batch_ids: list[int] | None = None, task: str | None = None) -> str:
        m = self._ma()
        body = m["models"]["ExportIn"].model_validate({"repo": repo, "message": message, "val_pct": val_pct,
                                                       "batch_ids": batch_ids, "task": task})
        return m["export"](SYSTEM_IDENT, cid, body)

    def mix(self, params: dict, actor: str) -> str:
        return self.state["start_dataset_mix"](params, actor=SYSTEM_IDENT[0], actor_type=SYSTEM_IDENT[1],
                                               channel=SYSTEM_IDENT[2])

    def submit_training(self, body: dict, actor: str) -> str:
        from libs.contracts.training import TrainingJobRequest
        req = TrainingJobRequest.model_validate({k: v for k, v in body.items() if v is not None})
        out = self.state["submit_training"](req, actor=SYSTEM_IDENT[0], actor_type=SYSTEM_IDENT[1],
                                            channel=SYSTEM_IDENT[2])
        return out["task_id"]

    def task(self, task_id: str) -> dict | None:
        t = self.state["store"].get(task_id)
        return t.model_dump() if t is not None else None


def engine_of(state: dict) -> CampaignEngine:
    if "media_campaign" not in state:
        from edgevision.media.library import MediaLibrary
        if "media" not in state:
            state["media"] = MediaLibrary(state["db"], state["objects"])
        audit = state["db"].write_audit if hasattr(state.get("db"), "write_audit") else None
        state["media_campaign"] = CampaignEngine(state["media"], StateActions(state), audit=audit)
    return state["media_campaign"]


class ModelRefIn(BaseModel):
    name: str = Field(description="Registry 模型名（例 siteb_bravo_seed）")
    version: int


RECIPE_DOC = ('混合配方：{"batches": {"<batch_id>": {"repeat": 0.5} | {"choose": 300} | {"proportion": 1}}, '
              '"total": 用 proportion 時的總張數, "seed": 0}；沒寫到的批次 = 全部取一次；值給 null = 拿掉')


class CampaignIn(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    collection_id: int | None = Field(None, description="圖片庫的圖片集：專案的圖都在這（初始批次＋之後每一批）。"
                                                       "不給 → 有 initial_batch_ids 就用批次所在的圖片集；"
                                                       "都沒給 → 自動建一個空的（名稱＝專案名、類別與 task 從初始模型帶）。"
                                                       "一個圖片集只給一個專案用")
    initial_model: ModelRefIn = Field(description="初始模型（Registry）：第 0 代；第一次「只預標」用它")
    initial_batch_ids: list[int] | None = Field(None, description="圖片集裡哪幾個批次是初始人工標注")
    line_id: str | None = Field(None, description="自動建圖片集時記在圖片集上的產線（指定 collection_id 時不用）")
    camera_id: str | None = Field(None, description="自動建圖片集時記在圖片集上的相機")
    model_name: str = Field(pattern=r"^[a-z0-9_\-]+$", description="每一代註冊成這個模型的新版本（建議用產線最後要用的名字）")
    base_model: str = Field(description="每一代從這個權重重新訓練，例 s3://models/_base/yolo26m.pt")
    golden_ref: str | None = Field(None, description="測試集：LakeFS 快照；可之後再設")
    golden_collection_id: int | None = Field(None, description="測試集圖片集（只是參照，前端顯示進度用）")
    config: dict[str, Any] | None = Field(None, description="覆蓋 campaign_defaults.yaml 的欄位")
    recipe: dict[str, Any] | None = Field(None, description=RECIPE_DOC)


class CampaignPatch(BaseModel):
    golden_ref: str | None = Field(None, description="換測試集（之後每一代訓練都在它上面評估）")
    clear_golden: bool = False
    golden_collection_id: int | None = Field(None, description="測試集圖片集（參照）")
    clear_golden_collection: bool = False
    config: dict[str, Any] | None = Field(None, description="部分更新覆蓋；值給 null = 回到預設")
    recipe: dict[str, Any] | None = Field(None, description=RECIPE_DOC)
    initial_batch_ids: list[int] | None = Field(None, description="改初始批次（更正用）")
    initial_model: ModelRefIn | None = Field(None, description="改初始模型（更正用；第 0 代紀錄一起改）")


def register(r: APIRouter, *, state: dict, lib: Callable, who: Callable, audit: Callable, call: Callable,
             H: dict) -> None:
    def eng() -> CampaignEngine:
        return engine_of(state)

    def act(fn, *a, **kw):
        try:
            return call(fn, *a, **kw)
        except PermissionError as exc:
            raise HTTPException(403, str(exc)) from exc

    @r.post("/campaigns", status_code=201, responses={201: {"model": S.CampaignDetailOut}})
    def create_campaign(body: CampaignIn, x_actor: str | None = H["x_actor"],
                        x_actor_type: str | None = H["x_actor_type"], x_channel: str | None = H["x_channel"],
                        authorization: str | None = H["authorization"]):
        """建標注專案（建好是閒置）。之後人按 run：匯出（每批一個快照）→ 混合 → 訓練 → 用新模型預標 → 閒置。"""
        ident = who(x_actor, x_actor_type, x_channel, authorization)
        d = body.model_dump()
        line_id, camera_id = d.pop("line_id"), d.pop("camera_id")
        if d["collection_id"] is None and not d["initial_batch_ids"]:
            d["new_collection"] = _new_collection(body.name, body.initial_model, line_id, camera_id)
        if not ((d.get("config") or {}).get("dataset") or {}).get("task"):
            if t := _model_task(body.initial_model):         # 標注格式＝訓練 task 跟著初始模型
                cfg = dict(d.get("config") or {})
                cfg["dataset"] = {**(cfg.get("dataset") or {}), "task": t}
                d["config"] = cfg
        out = act(eng().create, actor=ident[0], **d)
        audit(ident, "media.campaign.create", str(out["id"]),
              {**body.model_dump(), "collection_id": out["collection_id"]})
        return out

    def _model_task(m: ModelRefIn) -> str | None:
        """初始模型的 task（Registry）；連不上 / 找不到 = None（之後當 detect）。"""
        try:
            r_ = state["registry_factory"]().get(m.name, m.version)
        except Exception:  # noqa: BLE001
            logger.warning("讀不到初始模型 %s v%s 的 task，當 detect", m.name, m.version)
            return None
        t = getattr(getattr(r_, "meta", None), "task", None)
        return t if t in LABEL_TASKS else None

    def _new_collection(name: str, m: ModelRefIn, line_id: str | None, camera_id: str | None) -> dict:
        """自動建的圖片集：類別跟初始模型一樣（預標出來的類別才對得上）。"""
        try:
            r_ = state["registry_factory"]().get(m.name, m.version)
        except Exception as exc:
            raise HTTPException(502, f"Registry 連不上，無法讀初始模型的類別：{exc}") from exc
        if r_ is None:
            raise HTTPException(404, f"Registry 找不到初始模型 {m.name} v{m.version}")
        meta = r_.meta
        n_kpt = getattr(getattr(meta, "export", None), "n_kpt", 0) or 0
        return {"name": name, "class_names": list(meta.class_names),
                "kpt_shape": [n_kpt, 3] if meta.task == "pose" and n_kpt else None,
                "description": f"標注專案「{name}」自動建立（初始模型 {m.name} v{m.version}）",
                "line_id": line_id, "camera_id": camera_id}

    @r.get("/campaigns", responses={200: {"model": list[S.CampaignOut]}})
    def list_campaigns():
        """所有標注專案（含各角色：圖片集、測試集圖片集、初始批次＋初始模型）。"""
        return eng().list()

    @r.get("/campaigns/{cid}", responses={200: {"model": S.CampaignDetailOut}})
    def get_campaign(cid: int):
        """專案詳情：狀態與進行到哪一步、所有批次與快照、每一代的配方 / 訓練快照 / 模型 / 指標 / 預標任務、
        最新一代模型（latest_model，只預標用它）、最近事件。"""
        return act(eng().detail, cid)

    @r.patch("/campaigns/{cid}", responses={200: {"model": S.CampaignDetailOut}})
    def patch_campaign(cid: int, body: CampaignPatch, x_actor: str | None = H["x_actor"],
                       x_actor_type: str | None = H["x_actor_type"], x_channel: str | None = H["x_channel"],
                       authorization: str | None = H["authorization"]):
        ident = who(x_actor, x_actor_type, x_channel, authorization)
        d = body.model_dump()
        out = act(eng().update, cid, actor=ident[0], **d)
        audit(ident, "media.campaign.update", str(cid), body.model_dump())
        return out

    def _simple(name: str, fn_name: str, doc: str):
        def handler(cid: int, x_actor: str | None = H["x_actor"], x_actor_type: str | None = H["x_actor_type"],
                    x_channel: str | None = H["x_channel"], authorization: str | None = H["authorization"]):
            ident = who(x_actor, x_actor_type, x_channel, authorization)
            out = act(getattr(eng(), fn_name), cid, actor=ident[0])
            audit(ident, f"media.campaign.{name}", str(cid), {"stage": out["stage"], "state": out["state"]})
            return out
        handler.__doc__ = doc
        handler.__name__ = f"campaign_{name}"
        r.post(f"/campaigns/{{cid}}/{name}", responses={200: {"model": S.CampaignDetailOut},
                                                        409: {"description": "目前狀態不能做這件事"}})(handler)

    _simple("run", "run", "跑下一輪（閒置時）：匯出有改過的批次 → 依配方混合 → 訓練 → 用新模型預標還沒人審的圖 → 閒置，記成一代。")
    _simple("prelabel", "prelabel", "只預標（閒置時）：用最新一代的模型預標圖片集裡還沒人審的圖（加了新批次、還不想訓練時用）。")
    _simple("retry", "retry", "失敗時從失敗的那一步重跑（不從頭）。")
    _simple("close", "close", "結案（跑到一半不行）：圖片集釋放給別的專案，資料都留著。")
    _simple("reopen", "reopen", "接回結案的專案（回到閒置）；圖片集已被別的專案用 → 409。")
