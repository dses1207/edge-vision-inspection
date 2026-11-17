"""NAS 24h 錄影與影片庫 API（掛在 /api/media 底下；做法見 edgevision/media/recordings.py）。

    錄影
      GET  /recordings                           NAS 用量＋有錄影的相機（段數、大小、最後寫入）
      GET  /recordings/{cam}/days                這台有錄影的日子
      GET  /recordings/{cam}?day=                某天每一段的起訖、狀態（ok / writing / broken）、缺口
      GET  /recordings/{cam}/files/{file}        播放（Range）；?download=1 下載
      POST /recordings/{cam}/clips               剪一段（起訖時間＋名稱）→ 影片庫的一筆（背景任務）
    錄影名單
      GET  /recordings/roster                    所有相機：要不要錄、最近 24 小時寫了多少
      PUT  /recordings/roster/{cam}              開 / 關（nas-recorder 5 分鐘內跟上）
    影片庫
      POST   /videos                             上傳的影片只存進影片庫（不抽幀、不需要圖片集；背景任務）
      GET    /videos                             清單（名稱、標籤、播放網址、到期時間）
      GET    /videos/{id}                        單筆（含被哪些圖片集 / 批次 / 影片測試用過）
      PATCH  /videos/{id}                        改名、標籤、備註
      DELETE /videos/{id}                        刪除（圖片集裡的圖不受影響，只是不再指向這支影片）
      GET    /videos/{id}/stream                 播放（NAS 檔直接回、存在 MinIO 的轉到預簽網址；都支援 Range）
      GET    /videos/{id}/thumbnail?px=          縮圖（webp，第一次要才做、存起來；見 edgevision/media/video_thumbs.py）
"""

from __future__ import annotations

import logging
import os
from datetime import date, datetime
from typing import Literal

from fastapi import HTTPException, Query
from fastapi.responses import FileResponse, RedirectResponse, Response
from pydantic import BaseModel, Field

from edgevision.media import recordings as R
from edgevision.media import schemas as S
from edgevision.media import video_thumbs as VT
from edgevision.media.library import _row, _rows

logger = logging.getLogger(__name__)

VIDEO_KINDS = ("upload_video", "nas_recording", "recording_clip")
MAX_CLIP_S = int(os.environ.get("MEDIA_MAX_CLIP_S", 3600))
BINARY = {"type": "string", "format": "binary"}


class ClipIn(BaseModel):
    start: str = Field(description="本地時間 ISO，如 2026-10-19T08:00:00")
    end: str
    name: str | None = Field(None, max_length=200, description="影片名稱；不給 = 「<相機> <起> – <迄>」")
    tags: list[str] = Field(default_factory=list)
    note: str | None = Field(None, max_length=2000)


class RosterIn(BaseModel):
    enabled: bool
    note: str | None = Field(None, max_length=500)


class VideoUploadIn(BaseModel):
    source_uri: str = Field(description="POST /api/media/upload-url 回傳的 source_uri（影片已用 PUT 傳上去）")
    name: str | None = Field(None, max_length=200, description="影片名稱；不給 = 原檔名")
    tags: list[str] = Field(default_factory=list)
    note: str | None = Field(None, max_length=2000)
    camera_id: str | None = Field(None, description="哪台相機拍的（選填，影片清單可以用它篩）")


class VideoPatch(BaseModel):
    name: str | None = Field(None, max_length=200)
    tags: list[str] | None = Field(None, description="整組換掉")
    note: str | None = Field(None, max_length=2000)


def _local(s: str) -> datetime:
    from edgevision.media.frames import LOCAL_TZ
    t = datetime.fromisoformat(s)
    return t if t.tzinfo else t.replace(tzinfo=LOCAL_TZ)


def register(r, *, state: dict, lib, who, audit, call, H, new_task) -> None:
    from libs.contracts.training import TaskType

    def db():
        return state["db"]

    daily = R.DailyUsage(lambda: R.roster_cameras(db()))
    if os.path.isdir(os.environ.get("MEDIA_NAS_ROOT", "/nas/raw")):
        daily.start()                           # 有 NAS 的環境一開機就先算，第一個請求就有值

    def nas(fn, *a, **kw):
        try:
            return fn(*a, **kw)
        except R.NasUnavailable as exc:
            raise HTTPException(503, str(exc)) from exc
        except FileNotFoundError as exc:
            raise HTTPException(404, str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc

    # ── 錄影 ─────────────────────────────────────────────

    @r.get("/recordings", responses={200: {"model": S.RecordingsOut}})
    def recordings():
        """NAS 用量＋有錄影目錄的相機。recording = 錄影名單上有它；last_write_age_s 太大 = 名單上有但沒在寫。"""
        cams = nas(R.cameras)
        on = {x["camera_id"]: x["recording"] for x in R.roster(db())}
        for c in cams:
            c["recording"] = on.get(c["camera_id"], False)
        return {"nas": nas(R.usage), "cameras": cams}

    @r.get("/recordings/roster", responses={200: {"model": list[S.RosterOut]}})
    def get_roster():
        """所有相機（設定裡的）＋要不要 24h 錄到 NAS＋最近 24 小時實際寫了多少 GB（估容量用）。"""
        daily.start()                           # 第一次被叫才起背景（測試環境沒 NAS 也不會一直跑）
        rows = R.roster(db())
        for x in rows:
            x["daily_gb"] = daily.get(x["camera_id"]) if x["recording"] else None
        return rows

    @r.put("/recordings/roster/{cam}", responses={200: {"model": S.RosterOut}})
    def put_roster(cam: str, body: RosterIn, x_actor: str | None = H["x_actor"],
                   x_actor_type: str | None = H["x_actor_type"], x_channel: str | None = H["x_channel"],
                   authorization: str | None = H["authorization"]):
        """開 / 關這台的 24h 錄影。nas-recorder 每 5 分鐘重讀名單，開了之後最多 5 分鐘開始寫、再 10 分鐘才有第一個完整的檔。
        ⚠️ 原始畫質一台一天約 10–25 GB，開之前先看 GET /recordings 的 free_gb。"""
        ident = who(x_actor, x_actor_type, x_channel, authorization)
        try:
            out = R.set_roster(db(), cam, enabled=body.enabled, actor=ident[0], note=body.note)
        except LookupError as exc:
            raise HTTPException(404, str(exc)) from exc
        audit(ident, "media.recordings.roster", cam, body.model_dump())
        return out

    @r.get("/recordings/{cam}/days", responses={200: {"model": list[S.RecordingDayOut]}})
    def recording_days(cam: str):
        """這台有錄影的日子（新到舊）：段數、大小、第一段 / 最後一段的開始時間。不開檔，很快。"""
        return nas(R.list_days, cam)

    @r.get("/recordings/{cam}", responses={200: {"model": S.RecordingDayDetailOut}})
    def recording_day(cam: str, day: date = Query(description="本地日期 YYYY-MM-DD"),
                      gap_tolerance_s: float = Query(R.GAP_TOLERANCE_S, ge=1, le=3600,
                                                     description="相鄰兩段差超過幾秒才算缺口（檔名時間只準到幾秒）")):
        """某天每一段：起訖、長度、大小、狀態（ok / writing 還在寫 / broken 開不了）、播放網址；以及缺口。
        第一次查某天要逐段開檔量長度（約幾秒），之後有快取。"""
        out = nas(R.list_day, db(), cam, day, gap_tolerance_s=gap_tolerance_s)
        for s in out["segments"]:
            s["stream_url"] = f"/api/media/recordings/{cam}/files/{s['file']}"
            s["download_url"] = s["stream_url"] + "?download=1"
        return out

    @r.get("/recordings/{cam}/files/{file}", response_class=FileResponse,
           responses={200: {"content": {"video/mp4": {"schema": BINARY}}},
                      206: {"description": "Range 部分內容", "content": {"video/mp4": {"schema": BINARY}}}})
    def recording_file(cam: str, file: str, download: bool = False):
        """錄影檔本身（支援 Range，<video> 可以拖）。⚠️ 還在寫的那一段（status=writing）播不了：moov 在檔尾，寫完才有。
        ⚠️ siteb12 / ipcam87 是 H.265：Edge、Safari、有硬體解碼的 Chrome 可以播，其他瀏覽器可能只有聲音或黑畫面 —— 用下載。"""
        p = nas(R.file_path, cam, file)
        return FileResponse(p, media_type="video/mp4",
                            filename=f"{cam}_{file}" if download else None,
                            content_disposition_type="attachment" if download else "inline")

    @r.post("/recordings/{cam}/clips", status_code=202, responses={202: {"model": S.ClipCreatedOut}})
    def create_clip(cam: str, body: ClipIn, x_actor: str | None = H["x_actor"],
                    x_actor_type: str | None = H["x_actor_type"], x_channel: str | None = H["x_channel"],
                    authorization: str | None = H["authorization"]):
        """從 NAS 錄影剪一段存進影片庫（背景任務，回 task_id；完成後 GET /api/tasks/{id} 的 result.clip.video_id）。
        不重新編碼：起點對齊到前一個關鍵幀，實際起點記在影片的 params.actual_start。剪好的影片不會自動刪（expires_at = null）。"""
        ident = who(x_actor, x_actor_type, x_channel, authorization)
        try:
            start, end = _local(body.start), _local(body.end)
        except ValueError as exc:
            raise HTTPException(400, f"時間格式錯誤：{exc}") from exc
        if end <= start:
            raise HTTPException(400, "end 要晚於 start")
        if (end - start).total_seconds() > MAX_CLIP_S:
            raise HTTPException(400, f"一次最多剪 {MAX_CLIP_S // 60} 分鐘")
        segs = nas(R.segments_between, cam, start, end)
        if not segs:
            raise HTTPException(404, f"NAS 上 {cam} 在 {start:%m-%d %H:%M:%S}–{end:%H:%M:%S} 沒有錄影")
        name = body.name or f"{cam} {start:%m-%d %H:%M:%S}–{end:%H:%M:%S}"
        params = {"kind": "clip", "camera_id": cam, "start": start.isoformat(), "end": end.isoformat(), "name": name,
                  "tags": body.tags, "note": body.note}
        tid = new_task(TaskType.MEDIA_IMPORT, ident, params)
        audit(ident, "media.recordings.clip", cam, params, task_id=tid)
        state["orchestrator"].submit_media_import(tid)
        return {"task_id": tid, "name": name}

    # ── 影片庫 ───────────────────────────────────────────

    _SELECT = ("SELECT s.id, s.kind, s.uri, s.name, s.tags, s.note, s.expires_at, s.camera_id, s.started_at, s.params,"
               "       s.created_by, s.created_at, s.blob_sha256, b.duration_s, b.width, b.height, b.size_bytes"
               "  FROM media_sources s LEFT JOIN media_blobs b ON b.sha256 = s.blob_sha256")

    def _shape(x: dict) -> dict:
        p = x.pop("params") or {}
        x["start_s"], x["end_s"] = p.get("start_s"), p.get("end_s")
        x["actual_start"] = p.get("actual_start")
        x["name"] = x["name"] or _default_name(x)
        x["stream_url"] = f"/api/media/videos/{x['id']}/stream"
        x["thumbnail_url"] = f"/api/media/videos/{x['id']}/thumbnail"
        x.pop("blob_sha256", None)
        return x

    def _default_name(x: dict) -> str:
        if x["kind"] == "nas_recording" and x.get("started_at"):
            return f"{x.get('camera_id') or ''} {x['started_at'].astimezone():%m-%d %H:%M} NAS 錄影".strip()
        return os.path.basename(x["uri"]) or f"影片 {x['id']}"

    @r.post("/videos", status_code=202, responses={202: {"model": S.VideoUploadCreatedOut}})
    def upload_video(body: VideoUploadIn, x_actor: str | None = H["x_actor"],
                     x_actor_type: str | None = H["x_actor_type"], x_channel: str | None = H["x_channel"],
                     authorization: str | None = H["authorization"]):
        """上傳的影片只存進影片庫：不抽幀、不需要圖片集（要抽幀到圖片集用 POST /collections/{cid}/import）。

        流程：① POST /upload-url?filename=xxx.mp4 ② PUT 影片到 upload_url ③ 本端點帶 source_uri。
        背景任務（回 task_id）；完成後 GET /api/tasks/{id} 的 result.video.video_id。
        同一支影片（內容相同）已經在影片庫 → 不重複新增，result.video.duplicate = true、video_id 是原本那筆。"""
        from edgevision.media.api import UPLOAD_PREFIX
        from edgevision.media.library import VIDEO_EXTS
        from edgevision.media.jobs import upload_display_name

        ident = who(x_actor, x_actor_type, x_channel, authorization)
        if not body.source_uri.startswith(f"s3://raw/{UPLOAD_PREFIX}/"):
            raise HTTPException(400, "source_uri 要是 upload-url 回傳的上傳位置（s3://raw/uploads/media/…）")
        if os.path.splitext(body.source_uri)[1].lower() not in VIDEO_EXTS:
            raise HTTPException(400, f"只收影片（{sorted(VIDEO_EXTS)}）")
        if body.name is not None and not body.name.strip():
            raise HTTPException(400, "名稱不能是空白")
        if body.camera_id:
            with db().cursor() as cur:
                cur.execute("SELECT 1 FROM cameras WHERE id = %s", (body.camera_id,))
                if cur.fetchone() is None:
                    raise HTTPException(404, f"沒有相機 {body.camera_id}")
        name = (body.name or "").strip() or upload_display_name(body.source_uri)
        params = {"kind": "video_upload", **body.model_dump(), "name": name}
        tid = new_task(TaskType.MEDIA_IMPORT, ident, params)
        audit(ident, "media.video.upload", body.source_uri, params, task_id=tid)
        state["orchestrator"].submit_media_import(tid)
        return {"task_id": tid, "name": name}

    @r.get("/videos", responses={200: {"model": list[S.VideoSourceOut]}})
    def list_videos(camera_id: str | None = None, kind: Literal["upload_video", "nas_recording", "recording_clip"] | None = None,
                    tag: str | None = None, q: str | None = Query(None, description="名稱關鍵字"),
                    limit: int = Query(100, ge=1, le=1000)):
        """影片庫：上傳的影片、剪下的錄影片段、NAS 抽圖時用過的錄影檔。影片測試用 id 當 source_id。"""
        sql, args = _SELECT + " WHERE s.kind = ANY(%s)", [list(VIDEO_KINDS)]
        if kind:
            sql += " AND s.kind = %s"
            args.append(kind)
        if camera_id:
            sql += " AND s.camera_id = %s"
            args.append(camera_id)
        if tag:
            sql += " AND s.tags ? %s"
            args.append(tag)
        if q:
            sql += " AND (s.name ILIKE %s OR s.uri ILIKE %s)"
            args += [f"%{q}%", f"%{q}%"]
        sql += " ORDER BY s.id DESC LIMIT %s"
        args.append(limit)
        with db().cursor() as cur:
            cur.execute(sql, args)
            return [_shape(x) for x in _rows(cur)]

    def _video(vid: int) -> dict:
        with db().cursor() as cur:
            cur.execute(_SELECT + " WHERE s.id = %s AND s.kind = ANY(%s)", (vid, list(VIDEO_KINDS)))
            x = _row(cur)
        if x is None:
            raise HTTPException(404, f"影片庫沒有 {vid}")
        return x

    @r.get("/videos/{vid}", responses={200: {"model": S.VideoDetailOut}})
    def get_video(vid: int):
        """單筆＋被誰用過：從它抽圖的圖片集與批次（張數）、拿它跑過的影片測試。"""
        x = _shape(_video(vid))
        with db().cursor() as cur:
            cur.execute("""SELECT i.collection_id, c.name AS collection_name, i.batch_id, b.name AS batch_name,
                                  count(*) AS items
                             FROM media_items i JOIN media_collections c ON c.id = i.collection_id
                             LEFT JOIN media_batches b ON b.id = i.batch_id
                            WHERE i.source_id = %s GROUP BY 1, 2, 3, 4 ORDER BY 1, 3""", (vid,))
            x["used_by"] = _rows(cur)
            cur.execute("SELECT task_id, status, created_at, params->>'line_id' AS line_id FROM tasks"
                        " WHERE type = %s AND params->>'source_id' = %s ORDER BY created_at DESC LIMIT 100",
                        (TaskType.VIDEO_TEST.value, str(vid)))
            x["video_tests"] = _rows(cur)
        return x

    @r.patch("/videos/{vid}", responses={200: {"model": S.VideoSourceOut}})
    def patch_video(vid: int, body: VideoPatch, x_actor: str | None = H["x_actor"],
                    x_actor_type: str | None = H["x_actor_type"], x_channel: str | None = H["x_channel"],
                    authorization: str | None = H["authorization"]):
        """改名、換標籤、備註（只送要改的）。"""
        ident = who(x_actor, x_actor_type, x_channel, authorization)
        _video(vid)
        f = body.model_dump(exclude_unset=True)
        if "name" in f and f["name"] is not None and not f["name"].strip():
            raise HTTPException(400, "名稱不能是空白")
        if "tags" in f:
            f["tags"] = sorted({t.strip() for t in (f["tags"] or []) if t.strip()})
        if f:
            import json
            sets = ", ".join(f"{k} = %s" for k in f)
            vals = [json.dumps(v) if k == "tags" else v for k, v in f.items()]
            with db().cursor() as cur:
                cur.execute(f"UPDATE media_sources SET {sets} WHERE id = %s", (*vals, vid))
            audit(ident, "media.video.update", str(vid), f)
        return _shape(_video(vid))

    @r.delete("/videos/{vid}", responses={200: {"model": S.VideoDeletedOut}})
    def delete_video(vid: int, x_actor: str | None = H["x_actor"], x_actor_type: str | None = H["x_actor_type"],
                     x_channel: str | None = H["x_channel"], authorization: str | None = H["authorization"]):
        """從影片庫刪除。從它抽出來的圖都還在（只是不再指向這支影片）；存在 MinIO 的影片檔沒有別人用就一起刪。
        NAS 錄影檔本身不會被刪（那是 24h 錄影的原檔）。"""
        ident = who(x_actor, x_actor_type, x_channel, authorization)
        x = _video(vid)
        removed_file = False
        with db().cursor() as cur:
            cur.execute("SELECT count(*) FROM media_items WHERE source_id = %s", (vid,))
            n_items = int(cur.fetchone()[0])
            cur.execute("DELETE FROM media_sources WHERE id = %s", (vid,))
            if x["blob_sha256"]:
                cur.execute("SELECT 1 FROM media_sources WHERE blob_sha256 = %s LIMIT 1", (x["blob_sha256"],))
                orphan = cur.fetchone() is None
                if orphan:
                    cur.execute("SELECT 1 FROM media_items WHERE blob_sha256 = %s LIMIT 1", (x["blob_sha256"],))
                    orphan = cur.fetchone() is None
                if orphan:
                    blob = lib().blob(x["blob_sha256"])
                    cur.execute("DELETE FROM media_blobs WHERE sha256 = %s", (x["blob_sha256"],))
                    try:
                        lib().objects.remove(bucket=blob["bucket"], key=blob["object_key"])
                    except Exception as exc:  # noqa: BLE001 —— 檔案刪不掉只是多佔空間，紀錄已經刪了
                        logger.warning("影片檔 %s 刪不掉：%s", blob["object_key"], exc)
                    removed_file = True
        try:
            lib().objects.remove_prefix(bucket=VT.BUCKET, prefix=VT.thumb_prefix(vid))
        except Exception as exc:  # noqa: BLE001 —— 縮圖刪不掉只是多佔一點空間
            logger.warning("影片 %s 的縮圖刪不掉：%s", vid, exc)
        audit(ident, "media.video.delete", str(vid), {"kind": x["kind"], "uri": x["uri"], "items_detached": n_items})
        return {"id": vid, "items_detached": n_items, "file_removed": removed_file}

    @r.get("/videos/{vid}/stream", response_model=None,
           responses={200: {"content": {"video/mp4": {"schema": BINARY}}},
                      307: {"description": "轉到 MinIO 預簽網址（支援 Range）"}})
    def video_stream(vid: int, download: bool = False):
        """播放影片庫的一筆。NAS 錄影直接回檔（Range）；上傳 / 剪下的轉到 MinIO 預簽網址（2 小時，Range 由 MinIO 處理）。"""
        x = _video(vid)
        if x["kind"] == "nas_recording" and x["uri"].startswith("nas://raw/"):
            cam, _, file = x["uri"].removeprefix("nas://raw/").partition("/")
            return recording_file(cam, file, download=download)
        if not x["blob_sha256"]:
            raise HTTPException(404, "這支影片沒有保存原檔")
        blob = lib().blob(x["blob_sha256"])
        url = lib().objects.presigned_url(bucket=blob["bucket"], key=blob["object_key"], expires_minutes=120)
        return RedirectResponse(url, status_code=307)

    @r.get("/videos/{vid}/thumbnail", response_class=Response,
           responses={200: {"content": {"image/webp": {"schema": BINARY}}},
                      404: {"description": "影片不存在，或影片檔已不在（NAS 超過保留期）"},
                      422: {"description": "影片解不開（壞檔、還在寫的 NAS 段）"}})
    def video_thumbnail(vid: int, px: int = Query(320, ge=64, le=1280, description="長邊像素")):
        """影片縮圖（webp，經 control-api 代傳，瀏覽器不用連得到 MinIO）。取第 1 秒那一幀（有匯入起點就從起點算）。
        第一次要的時候才做（要解一下影片，約 0.5～2 秒），做完存起來，之後直接讀；內容不會變，可長期快取。
        清單的 thumbnail_url 可以直接放進 <img>。"""
        x = _video(vid)
        try:
            data = VT.thumbnail(lib(), x, px, R.file_path)
        except VT.VideoUnavailable as exc:
            raise HTTPException(404, str(exc)) from exc
        except R.NasUnavailable as exc:
            raise HTTPException(503, str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        return Response(data, media_type="image/webp", headers={"Cache-Control": "private, max-age=31536000, immutable"})
