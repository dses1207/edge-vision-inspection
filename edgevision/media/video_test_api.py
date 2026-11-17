"""影片測試 API（掛在 /api/media 底下；做法見 edgevision/media/video_test.py）。

    選影片  GET  /videos                    影片庫（在 edgevision/media/recordings_api.py）
    跑      POST /video-tests               影片 ＋ 線路（判定邏輯）＋ 模型版本（不給 = 現役）→ 任務
    看結果  GET  /video-tests               歷次結果（可依線路 / 模型 / 影片篩）
            GET  /video-tests/{task_id}     摘要、NG 清單、打框影片與每幀結果的下載網址
"""

from __future__ import annotations

import json

from fastapi import HTTPException, Query
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from edgevision.media import schemas as S


class VideoTestIn(BaseModel):
    line_id: str = Field(description="用哪條線的判定邏輯與參數")
    model_name: str | None = Field(None, description="主站模型；不給 = 這條線現在用的模型")
    model_version: int | None = Field(None, description="不給 = 現役（production）版")
    camera_id: str | None = Field(None, description="用這條線哪台相機的現場參數（ROI 等）；不給 = 影片的相機在這條線就用它，否則第一台。"
                                                   "NAS 錄影時是錄影的相機")
    source_id: int | None = Field(None, description="圖片庫的影片（GET /videos 的 id）")
    start_s: float | None = Field(None, ge=0, description="source_id：從影片第幾秒開始；不給 = 匯入時的起點")
    end_s: float | None = Field(None, gt=0, description="source_id：到第幾秒；不給 = 匯入時的終點或到結尾")
    start: str | None = Field(None, description="不用圖片庫、直接拿 NAS 錄影：本地時間 ISO，如 2026-10-16T08:00:00")
    end: str | None = None
    station: str | None = Field(None, description="換成別的判定邏輯（站別名，例 siteb_bravo_v2）；不給 = 這條線現在的。"
                                                  "新邏輯上線前先拿錄影試")
    station_params: dict | None = Field(None, description="換站別時的現場參數（thresholds / roi_regions）；不給 = 換站別就用站別預設，"
                                                         "沒換就用相機現在的")
    mode: Literal["logic", "model"] = Field("logic", description="logic = 邏輯＋模型：照影片實際速度播，出 NG 清單；"
                                                                "model = 純模型：能跑多快跑多快、不丟幀，不看判定，"
                                                                "出各類別框數 / 出現幀數 / 平均信心值（比模型版本用）；"
                                                                "不載入線上判定邏輯、不需要 ROI，station / station_params 不採用")

    @model_validator(mode="after")
    def _one_source(self):
        nas = bool(self.start or self.end)
        if bool(self.source_id) == nas:
            raise ValueError("影片二選一：source_id（圖片庫），或 camera_id + start + end（NAS 錄影）")
        if nas and not (self.camera_id and self.start and self.end):
            raise ValueError("NAS 錄影要給 camera_id、start、end")
        # 長度上限先在這裡擋（不用等排隊、取片才失敗）；source_id 沒給範圍的由 prepare 擋
        from datetime import datetime
        from edgevision.media.video_test import MAX_DURATION_S
        span = ((datetime.fromisoformat(self.end) - datetime.fromisoformat(self.start)).total_seconds() if nas
                else (self.end_s - (self.start_s or 0)) if self.end_s is not None else None)
        if span is not None and span > MAX_DURATION_S:
            raise ValueError(f"影片 {span:.0f} 秒太長（上限 {MAX_DURATION_S:.0f} 秒，{MAX_DURATION_S / 60:.0f} 分鐘）；請縮短時間段")
        return self


def _summary(row: dict) -> dict:
    vt = ((row.get("result") or {}).get("video_test") or {})
    prep = ((row.get("result") or {}).get("prepare") or {})
    p = row.get("params") or {}
    return {
        "task_id": row["task_id"], "status": row["status"], "status_message": row.get("status_message"),
        "created_at": row["created_at"].isoformat() if row.get("created_at") else None,
        "created_by": row.get("created_by"), "line_id": p.get("line_id"),
        "model_name": vt.get("model_name") or prep.get("model_name") or p.get("model_name"),
        "model_version": vt.get("model_version") or prep.get("model_version") or p.get("model_version"),
        "camera_id": vt.get("camera_id") or prep.get("camera_id") or p.get("camera_id"),
        "source_id": p.get("source_id"), "start": p.get("start"), "end": p.get("end"),
        "duration_s": vt.get("duration_s") or prep.get("duration_s"),
        "ng_count": vt.get("ng_count"), "ng_by_reason": vt.get("ng_by_reason"),
        "unknown_count": vt.get("unknown_count"), "station": p.get("station"),
        "mode": p.get("mode") or "logic", "runner_s": vt.get("runner_s"), "speed": vt.get("speed"),
    }


def register(r, *, state: dict, lib, who, audit, call, H) -> None:
    from libs.contracts.training import TaskType

    @r.post("/video-tests", status_code=202, responses={202: {"model": S.TaskCreatedOut}})
    def create_video_test(body: VideoTestIn, x_actor: str | None = H["x_actor"],
                          x_actor_type: str | None = H["x_actor_type"], x_channel: str | None = H["x_channel"],
                          authorization: str | None = H["authorization"]):
        """影片 ＋ 線路判定邏輯 ＋ 模型版本 → 打框影片、每幀結果、NG 清單（GPU 背景任務，跟訓練共用排隊）。

        mode=logic（預設）照影片實際速度播（站別的連續秒數 / 冷卻才跟線上一樣），10 分鐘的影片大約要跑 10 分鐘。
        mode=model 純模型：不限速、不出 NG，改出各類別統計，用來比兩版模型看得到什麼。
        """
        from libs.contracts.training import TaskStatus

        ident = who(x_actor, x_actor_type, x_channel, authorization)
        if body.source_id:
            with state["db"].cursor() as cur:
                cur.execute("SELECT kind FROM media_sources WHERE id = %s", (body.source_id,))
                row = cur.fetchone()
            if row is None or row[0] not in ("upload_video", "nas_recording", "recording_clip"):
                raise HTTPException(404, f"圖片庫沒有影片 {body.source_id}（GET /api/media/videos）")
        reg = state["registry_factory"]()
        if body.model_name or body.model_version:
            name = body.model_name
            if not name:
                raise HTTPException(400, "給了 model_version 就要給 model_name")
            r_ = reg.get(name, body.model_version) if body.model_version else reg.resolve(name)
            if r_ is None:
                raise HTTPException(404, f"Registry 找不到 {name} {('v' + str(body.model_version)) if body.model_version else '現役版'}")
        params = body.model_dump(exclude_none=True)
        tid = state["store"].create(type=TaskType.VIDEO_TEST, created_by=ident[0], channel=ident[2], params=params,
                                    status=TaskStatus.QUEUED, status_message="已受理")
        audit(ident, "media.video_test", body.line_id, params, task_id=tid)
        state["orchestrator"].submit(tid)
        return {"task_id": tid, "queue_position": state["store"].queue_position(tid)}

    @r.get("/video-tests", responses={200: {"model": list[S.VideoTestSummaryOut]}})
    def list_video_tests(line_id: str | None = None, model_name: str | None = None, source_id: int | None = None,
                         limit: int = Query(50, ge=1, le=500)):
        """歷次影片測試（新的在前）。同一段影片換模型版本各跑一次，就能並排比 NG。"""
        sql = ("SELECT task_id, status, status_message, created_at, created_by, params, result FROM tasks"
               " WHERE type = %s")
        args: list = [TaskType.VIDEO_TEST.value]
        if line_id:
            sql += " AND params->>'line_id' = %s"
            args.append(line_id)
        if source_id:
            sql += " AND params->>'source_id' = %s"
            args.append(str(source_id))
        sql += " ORDER BY created_at DESC LIMIT %s"
        args.append(limit * 4 if model_name else limit)
        with state["db"].cursor() as cur:
            cur.execute(sql, args)
            cols = [d[0] for d in cur.description]
            rows = [_summary(dict(zip(cols, x))) for x in cur.fetchall()]
        if model_name:
            rows = [x for x in rows if x["model_name"] == model_name][:limit]
        return rows

    @r.get("/video-tests/{task_id}", responses={200: {"model": S.VideoTestOut}})
    def get_video_test(task_id: str, ng_limit: int = Query(500, ge=0, le=10000)):
        """一次測試的完整結果：摘要、NG 清單（時間 = 影片第幾秒；mode=model 沒有）、各類別統計、
        打框影片與每幀結果的下載網址（2 小時有效）。"""
        with state["db"].cursor() as cur:
            cur.execute("SELECT task_id, type, status, status_message, created_at, created_by, params, result"
                        " FROM tasks WHERE task_id = %s", (task_id,))
            row = cur.fetchone()
            cols = [d[0] for d in cur.description] if row else []
        if row is None:
            raise HTTPException(404, f"沒有任務 {task_id}")
        d = dict(zip(cols, row))
        if d["type"] != TaskType.VIDEO_TEST.value:
            raise HTTPException(404, f"{task_id} 不是影片測試")
        res = d.get("result") or {}
        if isinstance(res, str):
            res = json.loads(res)
            d["result"] = res
        vt = res.get("video_test") or {}
        out = _summary(d) | {"params": d["params"], "frames": vt.get("frames"),
                             "detections_by_class": vt.get("detections_by_class"),
                             "class_stats": vt.get("class_stats"),
                             "models": vt.get("models") or (res.get("prepare") or {}).get("models"),
                             "video": vt.get("video") or (res.get("prepare") or {}).get("video"),
                             "ng": (vt.get("ng") or [])[:ng_limit], "unknown": (vt.get("unknown") or [])[:ng_limit]}
        objects = state["objects"]
        for k, uri in (("video_url", vt.get("video_uri")), ("results_url", vt.get("results_uri"))):
            if uri:
                b, key = objects.parse_uri(uri)
                out[k] = objects.presigned_url(bucket=b, key=key, expires_minutes=120)
        return out
