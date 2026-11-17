"""告警回流 API（掛在 /api/media 底下；做法見 edgevision/media/alerts.py）。

    待判  GET  /alerts/queue                 某天的 NG 事件，每站抽 N 個（含證據影片網址、當下的框）
    判    POST /alerts/{event_id}/review     真異常 / 誤報（null = 取消）
    回流  POST /alerts/feedback/run          判過、還沒回流的 → 各模型的標注專案圖片集開新批次（每天 CronJob 呼叫）
    紀錄  GET  /alerts/feedback              每個事件回流的結果（放進哪個圖片集、哪張圖、怎麼對到那一幀）
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Literal

from fastapi import HTTPException, Query
from pydantic import BaseModel, Field

from edgevision.media import alerts as A
from edgevision.media import schemas as S


class AlertReviewIn(BaseModel):
    result: Literal["true_positive", "false_positive"] | None = Field(
        description="true_positive = 真異常、false_positive = 誤報；null = 取消判定（已回流的不能改）")


class FeedbackRunIn(BaseModel):
    lookback_days: int | None = Field(None, ge=1, le=60, description="看最近幾天判過的；不給 = 設定預設")
    dry_run: bool = Field(False, description="只回會怎麼分，不建任務")


def _today() -> date:
    from edgevision.media.frames import LOCAL_TZ
    return datetime.now(LOCAL_TZ).date()


def _iso(v):
    return v.isoformat() if isinstance(v, datetime) else v


def register(r, *, state: dict, lib, who, audit, call, H, new_batch, new_task) -> None:
    from libs.contracts.training import TaskType

    def db():
        return state["db"]

    def _url(uri: str | None) -> str | None:
        if not uri:
            return None
        try:
            b, k = state["objects"].parse_uri(uri)
            return state["objects"].presigned_url(bucket=b, key=k, expires_minutes=120)
        except Exception:                                  # noqa: BLE001 —— 連結壞了就不給網址，清單照樣回
            return None

    @r.get("/alerts/queue", responses={200: {"model": S.AlertQueueOut}})
    def alert_queue(day: date | None = Query(None, description="哪一天（本地）；不給 = 今天"),
                    line_id: str | None = None, camera_id: str | None = None, station: str | None = None,
                    per_station: int | None = Query(None, ge=1, le=500, description="每站抽幾個；不給 = 設定預設"),
                    with_boxes: bool = Query(True, description="附上當下的框（前端疊在證據影片上）")):
        """待判清單：一個事件 = 同一台相機、同一幀、同一站的告警。每站每天抽 per_station 個，同一天重複查是同一批；
        判過的一律列出。證據影片（clip_url，2 小時有效）是打了框的；框另外附（原圖像素座標＋影像尺寸）。"""
        cfg = A.defaults()
        q = A.queue(db(), day=day or _today(), per_station=per_station or cfg["queue"]["per_station"],
                    line_id=line_id, camera_id=camera_id, station=station)
        for e in q["events"]:
            e["clip_url"] = _url(e.get("clip_uri"))
            if with_boxes:
                e["boxes"], e["image_size"] = A.event_boxes(db(), e)
            e["feedback"] = ({"status": e.pop("fb_status"), "reason": e.pop("fb_reason"),
                              "collection_id": e.pop("fb_collection_id"), "batch_id": e.pop("fb_batch_id"),
                              "item_id": e.pop("fb_item_id"), "item_status": e.pop("fb_item_status")}
                             if e.get("fb_status") else None)
            for k in ("fb_status", "fb_reason", "fb_collection_id", "fb_batch_id", "fb_item_id", "fb_item_status",
                      "rn", "n_station"):
                e.pop(k, None)
        return q

    @r.post("/alerts/{event_id}/review", responses={200: {"model": S.AlertReviewOut}})
    def alert_review(event_id: int, body: AlertReviewIn, x_actor: str | None = H["x_actor"],
                     x_actor_type: str | None = H["x_actor_type"], x_channel: str | None = H["x_channel"],
                     authorization: str | None = H["authorization"]):
        """判一個告警事件。判過的會在下一次每日回流時放進該模型標注專案的圖片集（框直接當標注）。"""
        ident = who(x_actor, x_actor_type, x_channel, authorization)
        try:
            out = A.review(db(), event_id, result=body.result, actor=ident[0])
        except LookupError as exc:
            raise HTTPException(404, str(exc)) from exc
        except PermissionError as exc:
            raise HTTPException(409, str(exc)) from exc
        audit(ident, "media.alert.review", str(event_id), body.model_dump())
        return out

    def _collection_for(model_name: str, actor: str, date_s: str) -> tuple[dict | None, str | None]:
        """模型 → 放哪個圖片集：進行中的標注專案；沒有就「告警回流 <模型>」（沒有就建，類別從 Registry 帶）。"""
        cfg = A.defaults()
        t = A.target_collection(db(), model_name)
        if t:
            return {"collection_id": int(t["collection_id"]), "campaign_id": t["campaign_id"],
                    "campaign_name": t["campaign_name"], "label_task": t.get("label_task") or "detect"}, None
        name = cfg["feedback"]["fallback_collection"].format(model=model_name)
        hit = next((c for c in lib().list_collections(include_archived=True) if c["name"] == name), None)
        try:
            res = state["registry_factory"]().resolve(model_name)
        except Exception as exc:                           # noqa: BLE001
            res, reg_err = None, f"Registry 連不上：{exc}"
        else:
            reg_err = None if res is not None else f"{model_name} 沒有現役版本，無法建圖片集"
        # 沒有專案時：線上框能不能直接當標注，看出告警的模型是不是框模型（分割 / 姿態模型的線上紀錄只有框）
        label_task = getattr(getattr(res, "meta", None), "task", None) or "detect"
        if hit:
            return {"collection_id": int(hit["id"]), "campaign_id": None, "campaign_name": None,
                    "label_task": label_task}, None
        if res is None:
            return None, reg_err
        c = lib().create_collection(name=name, actor=actor, class_names=list(res.meta.class_names),
                                    description=f"告警回流：{model_name} 沒有進行中的標注專案時放這裡（{date_s} 自動建立）")
        return {"collection_id": int(c["id"]), "campaign_id": None, "campaign_name": None, "created": True,
                "label_task": label_task}, None

    @r.post("/alerts/feedback/run", status_code=202, responses={202: {"model": S.AlertFeedbackRunOut}})
    def alert_feedback_run(body: FeedbackRunIn, x_actor: str | None = H["x_actor"],
                           x_actor_type: str | None = H["x_actor_type"], x_channel: str | None = H["x_channel"],
                           authorization: str | None = H["authorization"]):
        """回流：最近 N 天判過、還沒回流的事件 → 依模型分組 → 各自的圖片集開一個批次「告警回流 <日期>」＋匯入任務。
        每天由 CronJob 呼叫一次；手動呼叫也可以（已回流的不會重複）。"""
        ident = who(x_actor, x_actor_type, x_channel, authorization)
        cfg = A.defaults()
        evs = A.pending_events(db(), lookback_days=body.lookback_days or cfg["feedback"]["lookback_days"])
        groups: dict[str, list[dict]] = {}
        for e in evs:
            groups.setdefault(e["model_name"], []).append(e)
        today = _today().isoformat()
        out = []
        for model_name, es in groups.items():
            g = {"model_name": model_name, "events": len(es)}
            if body.dry_run:
                t = A.target_collection(db(), model_name)
                out.append(g | {"collection_id": t["collection_id"] if t else None,
                                "campaign_id": t["campaign_id"] if t else None})
                continue
            target, err = _collection_for(model_name, ident[0], today)
            if target is None:
                A.mark_skipped(db(), es, err)
                out.append(g | {"skipped": err})
                continue
            cid = target["collection_id"]
            payload = []
            for e in es:
                boxes, size = A.event_boxes(db(), e)
                payload.append({k: _iso(e[k]) for k in ("event_id", "camera_id", "frame_id", "station", "ts",
                                                        "model_name", "model_version", "reason", "clip_uri",
                                                        "review_result")} | {"boxes": boxes, "size": size})
            b = new_batch(ident, cid, "alert", cfg["feedback"]["batch_name"].format(date=today),
                          {"events": len(es), "model_name": model_name})
            params = {"collection_id": cid, "kind": "alert", "batch_id": b["id"], "events": payload,
                      "label_task": target["label_task"]}
            tid = new_task(TaskType.MEDIA_IMPORT, ident, params)
            lib().set_batch_task(b["id"], tid)
            A.claim(db(), es, task_id=tid, collection_id=cid, batch_id=b["id"])
            audit(ident, "media.alert.feedback", str(cid), {"model_name": model_name, "events": len(es),
                                                            "batch_id": b["id"]}, task_id=tid)
            state["orchestrator"].submit_media_import(tid)
            out.append(g | target | {"batch_id": b["id"], "task_id": tid})
        return {"events": len(evs), "groups": out, "dry_run": body.dry_run}

    @r.get("/alerts/feedback", responses={200: {"model": list[S.AlertFeedbackOut]}})
    def alert_feedback(status: Literal["queued", "added", "skipped", "failed"] | None = None,
                       task_id: str | None = None, limit: int = Query(200, ge=1, le=2000)):
        """每個事件回流的結果。align：怎麼對到 NAS 那一幀（lag = 證據影片↔NAS 的時間差、score = 框線吻合度、ok = 有把握）。"""
        return A.history(db(), limit=limit, status=status, task_id=task_id)
