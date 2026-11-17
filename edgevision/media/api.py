"""圖片庫 / 標注 / 自動標注 API（掛在控制 API 的 /api/media 底下）。

    圖片集      POST/GET /collections、GET/PATCH /collections/{id}、GET /collections/{id}/progress
    進圖片庫    POST /collections/{id}/images（multipart，少量）
                POST /upload-url → 直傳 MinIO → POST /collections/{id}/import（zip / 影片）
                POST /collections/{id}/from-recording（NAS 24h 錄影：相機 + 時間段）
    圖片        GET /collections/{id}/items、POST /collections/{id}/items/remove、POST /collections/{id}/copy
    標注        GET /items/{id}、POST|DELETE /items/{id}/lock、PUT /items/{id}/draft、POST /items/{id}/submit、
                GET /items/{id}/history、GET /collections/{id}/next（續標）
    自動標注    POST /collections/{id}/autolabel（批次，背景任務）、POST /items/{id}/predict（單張，Triton 即時）
    匯出        POST /collections/{id}/export → LakeFS 資料集版本、GET /collections/{id}/exports
    影片測試    GET /videos、POST|GET /video-tests、GET /video-tests/{task_id}（edgevision/media/video_test_api.py）
    審核        review-config、輪次、/rounds/{id}/next、POST /items/{id}/review（edgevision/media/review_api.py）

背景任務（匯入 / 自動標注 / 匯出）都是 tasks 表裡的任務：GET /api/tasks/{task_id} 看進度，可取消。
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any, Callable, Literal

from fastapi import APIRouter, File, Header, HTTPException, Query, UploadFile
from pydantic import BaseModel, Field

from edgevision.media import schemas as S
from edgevision.training.lakefs_client import repo_name_error

from edgevision.media.library import (
    IMAGE_EXTS,
    VIDEO_EXTS,
    Conflict,
    MediaLibrary,
    NotFound,
    read_upload,
)

logger = logging.getLogger(__name__)

UPLOAD_PREFIX = "uploads/media"
MAX_DIRECT_UPLOAD = 50 << 20          # 單檔 50 MB；再大走 upload-url 直傳


class CollectionIn(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    class_names: list[str] = Field(default_factory=list)
    description: str = ""
    kpt_shape: list[int] | None = None
    line_id: str | None = None
    camera_id: str | None = None
    org: str | None = Field(None, max_length=100, description="分類（自由文字，例「倉庫 A」）；不給 = 未分類")


class CollectionPatch(BaseModel):
    name: str | None = None
    description: str | None = None
    class_names: list[str] | None = Field(default=None, description="只能往後加，不能改既有順序")
    archived: bool | None = None
    line_id: str | None = None
    camera_id: str | None = None
    org: str | None = Field(None, max_length=100, description="改分類；空字串 = 清掉（變成未分類）")


class SelectIn(BaseModel):
    """strategy=diverse 的挑圖參數（不用任何模型；見 edgevision/media/select.py）。座標 0–1 [x1, y1, x2, y2]。"""
    target: int = Field(2000, ge=1, le=20000, description="最後要幾張（去重後不到這個數就全收）")
    dense_interval_s: float = Field(1.0, ge=0.2, le=60, description="密集看：每幾秒看一張（不存）")
    dedup_bits: int = Field(12, ge=0, le=256, description="去重：256 bit 指紋差異少於幾 bit（且 dedup_area）算重複；越大丟越多")
    dedup_area: float = Field(0.01, ge=0, le=1, description="去重：變化面積比例小於這個（且 dedup_bits）算重複；越大丟越多")
    diff_level: int = Field(20, ge=1, le=255, description="去重：灰階差多少算「有變化」的像素")
    motion_threshold: float = Field(0.004, ge=0, le=1, description="前景像素比例 ≥ 這個算「有變動」")
    still_ratio: float = Field(0.1, ge=0, le=1, description="保留給沒變動畫面的名額比例")
    ignore_regions: list[list[float]] | None = Field(
        None, description="比對時塗掉的區域；省略＝左上角時間戳 [[0,0,0.30,0.11]]；[] = 不遮")
    roi: list[float] | None = Field(None, description="只看這一塊（例如作業台）；省略＝整張")

    def params(self) -> dict:
        from edgevision.media.select import validate_rect
        d = self.model_dump(exclude_none=True)
        if self.roi is not None:
            d["roi"] = validate_rect(self.roi, "roi")
        if self.ignore_regions is not None:
            d["ignore_regions"] = [validate_rect(r, "ignore_regions") for r in self.ignore_regions]
        return d


class ImportIn(BaseModel):
    source_uri: str = Field(description="upload-url 回傳的 source_uri")
    # ⚠️ 沒有「存證片段」：存證影片錄的是燒錄了框 / mask 的串流，擷取出來的圖帶框，不能當訓練資料
    #    （舊系統警報截圖無法回收就是這個原因）。要某次 NG 的畫面 → 同相機同時間走 from-recording（NAS 乾淨錄影）。
    kind: Literal["zip", "video"] = "zip"
    with_labels: bool = Field(False, description="zip 內含 YOLO 標注（labels/ + classes.txt 或 data.yaml）")
    strategy: Literal["diverse", "interval"] = Field(
        "diverse", description="影片：diverse＝去重＋有變動優先＋分群挑代表（select）；interval＝每 interval_s 秒一張")
    select: SelectIn = Field(default_factory=SelectIn)
    interval_s: float = Field(1.0, gt=0, le=3600, description="strategy=interval：每幾秒取一張")
    max_frames: int = Field(2000, ge=1, le=20000, description="strategy=interval：上限")
    camera_id: str | None = None
    batch_name: str | None = Field(None, max_length=200, description="這批圖的批次名稱（例「第二批」）；省略 = 「<來源> <時間>」")
    batch_id: int | None = Field(None, description="加進這個既有批次（同一個圖片集）；不給 = 開新的一批。跟 batch_name 擇一")


class FromRecordingIn(BaseModel):
    camera_id: str
    start: str = Field(description="本地時間 ISO，如 2026-10-13T08:00:00")
    end: str
    strategy: Literal["diverse", "interval"] = Field(
        "diverse", description="diverse＝去重＋有變動優先＋分群挑代表（select）；interval＝每 interval_s 秒一張")
    select: SelectIn = Field(default_factory=SelectIn)
    interval_s: float = Field(2.0, gt=0, le=3600, description="strategy=interval：每幾秒取一張")
    max_frames: int = Field(2000, ge=1, le=20000, description="strategy=interval：上限")
    batch_name: str | None = Field(None, max_length=200, description="這批圖的批次名稱（例「第二批」）；省略 = 「<來源> <時間>」")
    batch_id: int | None = Field(None, description="加進這個既有批次（同一個圖片集）；不給 = 開新的一批。跟 batch_name 擇一")


class ItemIdsIn(BaseModel):
    item_ids: list[int] = Field(min_length=1)


class CopyIn(ItemIdsIn):
    target_collection_id: int
    with_labels: bool = False
    batch_name: str | None = Field(None, max_length=200, description="這批圖的批次名稱（例「第二批」）；省略 = 「<來源> <時間>」")
    batch_id: int | None = Field(None, description="加進這個既有批次（同一個圖片集）；不給 = 開新的一批。跟 batch_name 擇一")


class BatchPatch(BaseModel):
    name: str = Field(min_length=1, max_length=200)


class DraftIn(BaseModel):
    shapes: list[dict[str, Any]]
    revision: int


class SubmitIn(DraftIn):
    status: Literal["labeled", "background"] = "labeled"


class AutolabelIn(BaseModel):
    engine: Literal["registry", "yolo_world", "sam"]
    model_name: str | None = Field(None, description="registry 引擎：Registry 模型名")
    model_version: int | None = Field(None, description="預設現役（production）版")
    prompts: dict[str, str] | list[str] | None = Field(None, description="yolo_world / sam3：提示詞 → 圖片集類別")
    class_map: dict[str, str] | None = Field(None, description="registry：模型類別名 → 圖片集類別名")
    conf: float = Field(0.25, gt=0, lt=1)
    imgsz: int | None = None
    class_filter: list[str] | None = Field(None, description="只換這些類別，其他保留；不給 = 整張換")
    item_ids: list[int] | None = None
    skip_item_ids: list[int] | None = Field(None, description="例如種子圖")
    status: list[Literal["unlabeled", "auto", "labeled", "background"]] | None = Field(
        None, description="要跑哪些狀態的圖，預設 unlabeled + auto")
    overwrite_human: bool = Field(False, description="是否改寫人看過的圖（labeled / background）")
    weights: str | None = Field(None, description="yolo_world / sam 的基礎權重檔名（models/_base/）")
    batch_ids: list[int] | None = Field(None, description="只跑這些批次的圖（不給 = 整個圖片集）")


class ExportIn(BaseModel):
    repo: str = Field(description="LakeFS repo（資料集名稱）：小寫英數與 -，3–63 字（不能有底線、大寫）")
    message: str = "圖片集匯出"
    val_pct: int = Field(20, ge=0, le=100, description="驗證集比例；100 = 全部當驗證（黃金測試集）")
    batch_ids: list[int] | None = Field(None, description="只匯出這些批次（批次快照）；不給 = 整個圖片集")
    task: Literal["detect", "segment", "pose"] | None = Field(
        None, description="標注輸出格式：detect 框（多邊形取外接框）/ segment 多邊形（框轉四點）/ pose 關鍵點（其他形狀不出）。"
                          "不給 = detect。結果的 export.converted 列出轉換了幾個形狀")


class PredictIn(BaseModel):
    model_name: str = Field(description="Triton 上的模型（Registry 現役版）")
    conf: float = Field(0.25, gt=0, lt=1)
    class_map: dict[str, str] | None = None


def build_router(state: dict, actor_fn: Callable) -> APIRouter:
    """state：控制 API 的 _state（db、objects、store、orchestrator、registry_factory）。"""
    r = APIRouter(prefix="/api/media", tags=["media"])

    def lib() -> MediaLibrary:
        if "media" not in state:
            state["media"] = MediaLibrary(state["db"], state["objects"])
        return state["media"]

    def who(x_actor, x_actor_type, x_channel, authorization):
        return actor_fn(x_actor, x_actor_type, x_channel, authorization)

    def audit(ident, action: str, target: str, params: dict, result: str = "success", task_id: str | None = None,
              error: str | None = None):
        actor, actor_type, channel = ident
        state["db"].write_audit(actor=actor, actor_type=actor_type, channel=channel, action=action, target=target,
                                params=params, result=result, task_id=task_id, error=error)

    def call(fn, *a, **kw):
        try:
            return fn(*a, **kw)
        except NotFound as exc:
            raise HTTPException(404, str(exc)) from exc
        except Conflict as exc:
            raise HTTPException(409, exc.detail) from exc
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc

    H = dict(x_actor=Header(default=None), x_actor_type=Header(default=None), x_channel=Header(default=None),
             authorization=Header(default=None))

    def new_task(type_, ident, params: dict, *, queued: bool = False) -> str:
        from libs.contracts.training import TaskStatus
        st = TaskStatus.QUEUED if queued else TaskStatus.WORKING
        return state["store"].create(type=type_, created_by=ident[0], channel=ident[2], params=params,
                                     status=st, status_message="已受理")

    # 固定路徑（/items/search）要排在 /items/{item_id} 前面，所以先掛
    from edgevision.media.collection_ops_api import register as _register_collection_ops
    _register_collection_ops(r, state=state, lib=lib, who=who, audit=audit, call=call, H=H, new_task=new_task)

    # ── 圖片集 ───────────────────────────────────────────

    @r.post("/collections", status_code=201, responses={201: {"model": S.CollectionOut}})
    def create_collection(body: CollectionIn, x_actor: str | None = H["x_actor"],
                          x_actor_type: str | None = H["x_actor_type"], x_channel: str | None = H["x_channel"],
                          authorization: str | None = H["authorization"]):
        ident = who(x_actor, x_actor_type, x_channel, authorization)
        c = call(lib().create_collection, actor=ident[0], **body.model_dump())
        audit(ident, "media.collection.create", str(c["id"]), body.model_dump())
        return c

    @r.get("/collections", responses={200: {"model": list[S.CollectionSummaryOut]}})
    def list_collections(include_archived: bool = False,
                         org: str | None = Query(None, description="只列這個分類；給空字串 = 只列未分類的")):
        return call(lib().list_collections, include_archived=include_archived, org=org)

    @r.get("/collections/orgs", responses={200: {"model": list[S.OrgOut]}})
    def list_orgs(include_archived: bool = False):
        """用過的分類與各有幾個圖片集（下拉選單用；新分類直接在建立 / 修改圖片集時填）。"""
        return lib().orgs(include_archived=include_archived)

    @r.get("/collections/{cid}", responses={200: {"model": S.CollectionOut}})
    def get_collection(cid: int):
        return call(lib().collection, cid)

    @r.patch("/collections/{cid}", responses={200: {"model": S.CollectionOut}})
    def patch_collection(cid: int, body: CollectionPatch, x_actor: str | None = H["x_actor"],
                         x_actor_type: str | None = H["x_actor_type"], x_channel: str | None = H["x_channel"],
                         authorization: str | None = H["authorization"]):
        ident = who(x_actor, x_actor_type, x_channel, authorization)
        c = call(lib().update_collection, cid, **body.model_dump(exclude_none=True))
        audit(ident, "media.collection.update", str(cid), body.model_dump(exclude_none=True))
        return c

    @r.get("/collections/{cid}/progress", responses={200: {"model": S.ProgressOut}})
    def progress(cid: int):
        call(lib().collection, cid)
        return lib().progress(cid)

    @r.get("/collections/{cid}/items", responses={200: {"model": list[S.ItemSummaryOut]}})
    def list_items(cid: int, status: list[str] | None = Query(None), after_id: int = 0,
                   limit: int = Query(100, ge=1, le=1000), with_urls: bool = False,
                   batch_id: list[int] | None = Query(None, description="只列這些批次的圖（可重複給）"),
                   tag: str | None = Query(None, description="只列有這個標籤的圖")):
        items = lib().list_items(cid, status=status, after_id=after_id, limit=limit, batch_ids=batch_id, tag=tag)
        if with_urls:
            for it in items:
                it["image_url"] = lib().image_url(lib().blob(it["blob_sha256"]))
        return items

    # ── 批次（每次把圖放進圖片集 = 一批）───────────────────

    @r.get("/collections/{cid}/batches", responses={200: {"model": list[S.BatchOut]}})
    def list_batches(cid: int):
        """這個圖片集的所有批次，含各狀態張數（未標 / AI 預標 / 已確認）。"""
        return call(lib().batches, cid)

    @r.get("/batches/{bid}", responses={200: {"model": S.BatchOut}})
    def get_batch(bid: int):
        return call(lib().batch, bid)

    @r.patch("/batches/{bid}", responses={200: {"model": S.BatchOut}})
    def patch_batch(bid: int, body: BatchPatch, x_actor: str | None = H["x_actor"],
                    x_actor_type: str | None = H["x_actor_type"], x_channel: str | None = H["x_channel"],
                    authorization: str | None = H["authorization"]):
        """改批次名稱。"""
        ident = who(x_actor, x_actor_type, x_channel, authorization)
        out = call(lib().update_batch, bid, name=body.name)
        audit(ident, "media.batch.update", str(bid), body.model_dump())
        return out

    @r.delete("/batches/{bid}", responses={200: {"model": S.BatchDeleteOut}})
    def delete_batch(bid: int, x_actor: str | None = H["x_actor"],
                     x_actor_type: str | None = H["x_actor_type"], x_channel: str | None = H["x_channel"],
                     authorization: str | None = H["authorization"]):
        """刪整個批次：這批的圖從圖片集移除、批次刪掉（圖片庫的原圖不動）。
        匯入還在跑、或是進行中專案的初始批次 → 409。"""
        ident = who(x_actor, x_actor_type, x_channel, authorization)
        out = call(lib().delete_batch, bid)
        audit(ident, "media.batch.delete", str(bid), out)
        return out

    def new_batch(ident, cid: int, kind: str, name: str | None, params: dict, batch_id: int | None = None) -> dict:
        """batch_id 給了 = 加進既有批次；否則開新的一批。回傳的 b["appended"] 標明是哪一種。"""
        if batch_id is not None:
            if name is not None:
                raise HTTPException(400, "batch_id（加進既有批次）跟 batch_name（開新批次）只能給一個；要改舊批次名稱用 PATCH /batches/{id}")
            b = call(lib().append_batch, cid, batch_id, kind=kind, actor=ident[0], params=params)
            audit(ident, "media.batch.append", str(cid), {"batch_id": batch_id, "kind": kind})
            return b | {"appended": True}
        b = call(lib().create_batch, cid, kind=kind, actor=ident[0], name=name, params=params)
        audit(ident, "media.batch.create", str(cid), {"batch_id": b["id"], "name": b["name"], "kind": kind})
        return b | {"appended": False}

    def link_task(b: dict, tid: str) -> None:
        (lib().add_append_task if b["appended"] else lib().set_batch_task)(b["id"], tid)

    # ── 進圖片庫 ─────────────────────────────────────────

    @r.post("/collections/{cid}/images", status_code=201, responses={201: {"model": S.UploadImagesOut}})
    def upload_images(cid: int, files: list[UploadFile] = File(...),
                      batch_name: str | None = Query(None, max_length=200, description="批次名稱；省略 = 「上傳 <時間>」"),
                      batch_id: int | None = Query(None, description="加進這個既有批次；不給 = 開新的一批。跟 batch_name 擇一"),
                      x_actor: str | None = H["x_actor"],
                      x_actor_type: str | None = H["x_actor_type"], x_channel: str | None = H["x_channel"],
                      authorization: str | None = H["authorization"]):
        """少量圖片直接上傳（每檔 ≤50 MB），這次上傳 = 一個新批次，或用 batch_id 加進既有批次。大量或 zip 請用 upload-url + import。"""
        ident = who(x_actor, x_actor_type, x_channel, authorization)
        call(lib().collection, cid)
        bid = new_batch(ident, cid, "upload", batch_name, {"files": len(files)}, batch_id)["id"]
        out = []
        for f in files:
            name = Path(f.filename or "image.jpg").name
            if Path(name).suffix.lower() not in IMAGE_EXTS:
                out.append({"filename": name, "error": "不是支援的圖片格式"})
                continue
            try:
                ref = lib().put_image(read_upload(f.file, MAX_DIRECT_UPLOAD), filename=name, actor=ident[0])
                item_id, created = lib().add_item(cid, ref.sha256, actor=ident[0], original_name=name, batch_id=bid)
                out.append({"filename": name, "item_id": item_id, "sha256": ref.sha256,
                            "new_in_library": ref.created, "added": created})
            except ValueError as exc:
                out.append({"filename": name, "error": str(exc)})
        audit(ident, "media.upload", str(cid), {"files": [o["filename"] for o in out], "batch_id": bid,
                                                 "added": sum(1 for o in out if o.get("added"))})
        return {"results": out, "added": sum(1 for o in out if o.get("added")), "batch_id": bid}

    @r.post("/upload-url", responses={200: {"model": S.MediaUploadUrlOut}})
    def upload_url(filename: str = Query(...), x_actor: str | None = H["x_actor"],
                   x_actor_type: str | None = H["x_actor_type"], x_channel: str | None = H["x_channel"],
                   authorization: str | None = H["authorization"]):
        """臨時 PUT 網址（2 小時）：zip 或影片直傳 MinIO，再用回傳的 source_uri 呼叫 import。"""
        ident = who(x_actor, x_actor_type, x_channel, authorization)
        base = Path(filename).name
        ext = Path(base).suffix.lower()
        if ext != ".zip" and ext not in VIDEO_EXTS:
            raise HTTPException(400, f"只收 .zip 或影片（{sorted(VIDEO_EXTS)}）")
        key = f"{UPLOAD_PREFIX}/{time.strftime('%Y%m%d_%H%M%S')}_{base}"
        url = state["objects"].presigned_put_url(bucket="raw", key=key, expires_minutes=120)
        audit(ident, "media.upload_url", f"s3://raw/{key}", {"filename": base})
        return {"upload_url": url, "source_uri": f"s3://raw/{key}", "expires_minutes": 120,
                "kind": "zip" if ext == ".zip" else "video"}

    @r.post("/collections/{cid}/import", status_code=202, responses={202: {"model": S.TaskCreatedOut}})
    def import_source(cid: int, body: ImportIn, x_actor: str | None = H["x_actor"],
                      x_actor_type: str | None = H["x_actor_type"], x_channel: str | None = H["x_channel"],
                      authorization: str | None = H["authorization"]):
        from libs.contracts.training import TaskType
        ident = who(x_actor, x_actor_type, x_channel, authorization)
        call(lib().collection, cid)
        if not body.source_uri.startswith(f"s3://raw/{UPLOAD_PREFIX}/"):
            raise HTTPException(400, "source_uri 要是 upload-url 回傳的上傳位置（s3://raw/uploads/media/…）")
        try:
            params = {"collection_id": cid, **body.model_dump(), "select": body.select.params()}
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        b = new_batch(ident, cid, body.kind, body.batch_name, {"source_uri": body.source_uri,
                                                                "with_labels": body.with_labels}, body.batch_id)
        params["batch_id"] = b["id"]
        tid = new_task(TaskType.MEDIA_IMPORT, ident, params)
        link_task(b, tid)
        audit(ident, "media.import", str(cid), params, task_id=tid)
        state["orchestrator"].submit_media_import(tid)
        return {"task_id": tid, "batch_id": b["id"]}

    @r.post("/collections/{cid}/from-recording", status_code=202, responses={202: {"model": S.TaskCreatedOut}})
    def from_recording(cid: int, body: FromRecordingIn, x_actor: str | None = H["x_actor"],
                       x_actor_type: str | None = H["x_actor_type"], x_channel: str | None = H["x_channel"],
                       authorization: str | None = H["authorization"]):
        """NAS 24h 錄影 → 圖片集（相機 + 時間段）。預設 strategy=diverse：去重、有變動優先、分群挑 select.target 張。"""
        ident = who(x_actor, x_actor_type, x_channel, authorization)
        tid = start_from_recording(ident, cid, body)
        return {"task_id": tid, "batch_id": state["store"].get(tid).params["batch_id"]}

    def start_from_recording(ident, cid: int, body: FromRecordingIn) -> str:
        from datetime import datetime

        from libs.contracts.training import TaskType
        call(lib().collection, cid)
        try:
            if datetime.fromisoformat(body.end) <= datetime.fromisoformat(body.start):
                raise HTTPException(400, "end 要晚於 start")
        except ValueError as exc:
            raise HTTPException(400, f"時間格式錯誤：{exc}") from exc
        try:
            params = {"collection_id": cid, "kind": "nas", **body.model_dump(), "select": body.select.params()}
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        b = new_batch(ident, cid, "nas", body.batch_name, {"camera_id": body.camera_id, "start": body.start,
                                                            "end": body.end}, body.batch_id)
        params["batch_id"] = b["id"]
        tid = new_task(TaskType.MEDIA_IMPORT, ident, params)
        link_task(b, tid)
        audit(ident, "media.from_recording", str(cid), params, task_id=tid)
        state["orchestrator"].submit_media_import(tid)
        return tid

    @r.post("/collections/{cid}/items/remove", responses={200: {"model": S.CountOut}})
    def remove_items(cid: int, body: ItemIdsIn, x_actor: str | None = H["x_actor"],
                     x_actor_type: str | None = H["x_actor_type"], x_channel: str | None = H["x_channel"],
                     authorization: str | None = H["authorization"]):
        """從圖片集移除（圖片庫的原圖不動）。"""
        ident = who(x_actor, x_actor_type, x_channel, authorization)
        n = lib().remove_items(cid, body.item_ids)
        audit(ident, "media.items.remove", str(cid), {"item_ids": body.item_ids, "removed": n})
        return {"removed": n}

    @r.post("/collections/{cid}/copy", responses={200: {"model": S.CountOut}})
    def copy_items(cid: int, body: CopyIn, x_actor: str | None = H["x_actor"],
                   x_actor_type: str | None = H["x_actor_type"], x_channel: str | None = H["x_channel"],
                   authorization: str | None = H["authorization"]):
        """把這個圖片集（cid）的圖複製到 target_collection_id（只多一列關聯，不複製檔案）；在目標圖片集成為一個新批次（或用 batch_id 加進目標圖片集的既有批次）。"""
        ident = who(x_actor, x_actor_type, x_channel, authorization)
        call(lib().collection, cid)
        src = {i["id"] for i in lib().items_by_ids(body.item_ids) if i["collection_id"] == cid}
        bad = [i for i in body.item_ids if i not in src]
        if bad:
            raise HTTPException(400, f"圖片 {bad[:20]} 不在圖片集 {cid} 裡")
        b = new_batch(ident, body.target_collection_id, "copy", body.batch_name,
                      {"from_collection_id": cid, "n": len(body.item_ids), "with_labels": body.with_labels},
                      body.batch_id)
        n = call(lib().copy_items, body.item_ids, body.target_collection_id, actor=ident[0],
                 with_labels=body.with_labels, batch_id=b["id"])
        audit(ident, "media.items.copy", str(cid), body.model_dump() | {"added": n, "batch_id": b["id"]})
        return {"added": n, "batch_id": b["id"]}

    # ── 標注 ─────────────────────────────────────────────

    @r.get("/items/{item_id}", responses={200: {"model": S.ItemOut}})
    def get_item(item_id: int):
        """圖片網址、框、狀態、revision、鎖；在審核輪次裡的圖多一個 review 區塊（抽查圖不給預標）。"""
        from edgevision.media.review_api import item_with_review
        return call(item_with_review, state, lib, item_id)

    @r.post("/items/{item_id}/lock", responses={200: {"model": S.LockOut}, 409: {"description": "別人正在標"}})
    def lock(item_id: int, seconds: int = Query(300, ge=30, le=3600), x_actor: str | None = H["x_actor"],
             x_actor_type: str | None = H["x_actor_type"], x_channel: str | None = H["x_channel"],
             authorization: str | None = H["authorization"]):
        ident = who(x_actor, x_actor_type, x_channel, authorization)
        return call(lib().lock, item_id, actor=ident[0], seconds=seconds)

    @r.delete("/items/{item_id}/lock", responses={200: {"model": S.OkOut}})
    def unlock(item_id: int, x_actor: str | None = H["x_actor"], x_actor_type: str | None = H["x_actor_type"],
               x_channel: str | None = H["x_channel"], authorization: str | None = H["authorization"]):
        ident = who(x_actor, x_actor_type, x_channel, authorization)
        lib().unlock(item_id, actor=ident[0])
        return {"ok": True}

    @r.put("/items/{item_id}/draft", responses={200: {"model": S.RevisionOut}, 409: {"description": "revision 不符或被別人鎖住"}})
    def save_draft(item_id: int, body: DraftIn, x_actor: str | None = H["x_actor"],
                   x_actor_type: str | None = H["x_actor_type"], x_channel: str | None = H["x_channel"],
                   authorization: str | None = H["authorization"]):
        """自動存檔（前端每幾秒或切圖時呼叫）。不留歷史、不寫稽核（太頻繁）；revision 不符回 409。"""
        ident = who(x_actor, x_actor_type, x_channel, authorization)
        return call(lib().save_draft, item_id, body.shapes, revision=body.revision, actor=ident[0])

    @r.post("/items/{item_id}/submit", responses={200: {"model": S.RevisionOut}, 409: {"description": "revision 不符或被別人鎖住"}})
    def submit(item_id: int, body: SubmitIn, x_actor: str | None = H["x_actor"],
               x_actor_type: str | None = H["x_actor_type"], x_channel: str | None = H["x_channel"],
               authorization: str | None = H["authorization"]):
        ident = who(x_actor, x_actor_type, x_channel, authorization)
        out = call(lib().submit, item_id, body.shapes, revision=body.revision, actor=ident[0], status=body.status)
        audit(ident, "media.annotation.submit", str(item_id), {"status": body.status, "n": len(body.shapes),
                                                               "revision": out["revision"]})
        return out

    @r.get("/items/{item_id}/history", responses={200: {"model": list[S.HistoryOut]}})
    def history(item_id: int, limit: int = Query(20, ge=1, le=200)):
        return lib().history(item_id, limit=limit)

    @r.get("/collections/{cid}/next", responses={200: {"model": S.NextOut}})
    def next_item(cid: int, status: list[str] | None = Query(None), after_id: int | None = None,
                  x_actor: str | None = H["x_actor"], x_actor_type: str | None = H["x_actor_type"],
                  x_channel: str | None = H["x_channel"], authorization: str | None = H["authorization"]):
        """續標：從自己上次停的地方往後找下一張（預設未標 + AI 預標），順便領取。item 為 null = 沒有了。"""
        ident = who(x_actor, x_actor_type, x_channel, authorization)
        call(lib().collection, cid)
        st = status or ["unlabeled", "auto"]
        it = lib().next_item(cid, actor=ident[0], status=st, after_id=after_id)
        prog = lib().progress(cid)
        return {"item": it, "remaining": sum(prog.get(s, 0) for s in st)}

    # ── 自動標注 ─────────────────────────────────────────

    @r.post("/collections/{cid}/autolabel", status_code=202, responses={202: {"model": S.TaskCreatedOut}})
    def autolabel(cid: int, body: AutolabelIn, x_actor: str | None = H["x_actor"],
                  x_actor_type: str | None = H["x_actor_type"], x_channel: str | None = H["x_channel"],
                  authorization: str | None = H["authorization"]):
        """批次自動標注（GPU 背景任務，跟訓練共用排隊）。結果寫成 auto，預設不動人看過的圖。"""
        ident = who(x_actor, x_actor_type, x_channel, authorization)
        tid = start_autolabel(ident, cid, body)
        return {"task_id": tid, "queue_position": state["store"].queue_position(tid)}

    def start_autolabel(ident, cid: int, body: AutolabelIn) -> str:
        from libs.contracts.training import TaskType
        col = call(lib().collection, cid)
        if not col["class_names"]:
            raise HTTPException(400, "圖片集還沒有類別")
        if body.engine == "registry" and not body.model_name:
            raise HTTPException(400, "registry 引擎要給 model_name")
        if body.engine == "yolo_world" and not body.prompts:
            raise HTTPException(400, "yolo_world 要給 prompts（提示詞 → 圖片集類別）")
        call(lib().check_batches, cid, body.batch_ids)
        params = {"collection_id": cid, **body.model_dump(exclude_none=True)}
        tid = new_task(TaskType.AUTOLABEL, ident, params, queued=True)
        audit(ident, "media.autolabel", str(cid), params, task_id=tid)
        state["orchestrator"].submit(tid)
        return tid

    @r.post("/items/{item_id}/predict", responses={200: {"model": S.PredictOut}})
    def predict(item_id: int, body: PredictIn):
        """單張即時預標：用 Triton 上的現役模型（毫秒級），只回結果、不寫入（前端放進草稿再由人修）。"""
        it = call(lib().item, item_id, with_url=False)
        reg = state["registry_factory"]()
        prod = reg.resolve(body.model_name)
        if prod is None:
            raise HTTPException(404, f"{body.model_name} 沒有現役版本")
        shapes = predict_triton(lib().blob_bytes(it["blob_sha256"]), model_name=body.model_name,
                                version=prod.version, class_names=prod.meta.class_names, imgsz=prod.meta.imgsz,
                                conf=body.conf, task=prod.meta.task, targets=it["class_names"],
                                class_map=body.class_map)
        return {"item_id": item_id, "model": body.model_name, "version": prod.version, "shapes": shapes}

    # ── 匯出 ─────────────────────────────────────────────

    @r.post("/collections/{cid}/export", status_code=202, responses={202: {"model": S.TaskCreatedOut}})
    def export(cid: int, body: ExportIn, x_actor: str | None = H["x_actor"],
               x_actor_type: str | None = H["x_actor_type"], x_channel: str | None = H["x_channel"],
               authorization: str | None = H["authorization"]):
        """人看過的圖（labeled / background）→ LakeFS 資料集版本。AI 預標沒人確認的不匯出。"""
        ident = who(x_actor, x_actor_type, x_channel, authorization)
        return {"task_id": start_export(ident, cid, body)}

    def start_export(ident, cid: int, body: ExportIn) -> str:
        from libs.contracts.training import TaskType
        if err := repo_name_error(body.repo):
            raise HTTPException(400, err)
        call(lib().collection, cid)
        call(lib().check_batches, cid, body.batch_ids)
        params = {"collection_id": cid, **body.model_dump()}
        tid = new_task(TaskType.MEDIA_EXPORT, ident, params)
        audit(ident, "media.export", str(cid), params, task_id=tid)
        state["orchestrator"].submit_media_export(tid)     # CPU Job（edgevision.media.jobs export）；control-api 重啟不會打斷
        return tid

    @r.get("/collections/{cid}/exports", responses={200: {"model": list[S.ExportRecordOut]}})
    def exports(cid: int):
        return lib().exports(cid)

    from edgevision.media.video_test_api import register as _register_video_test
    _register_video_test(r, state=state, lib=lib, who=who, audit=audit, call=call, H=H)

    from edgevision.media.recordings_api import register as _register_recordings
    _register_recordings(r, state=state, lib=lib, who=who, audit=audit, call=call, H=H, new_task=new_task)

    from edgevision.media.alerts_api import register as _register_alerts
    _register_alerts(r, state=state, lib=lib, who=who, audit=audit, call=call, H=H, new_batch=new_batch,
                     new_task=new_task)

    from edgevision.media.review_api import register as _register_review
    _register_review(r, state=state, lib=lib, who=who, audit=audit, call=call, H=H)

    # 標注專案（edgevision/media/campaign.py）直接呼叫這些動作，不經 HTTP；任務、稽核跟從 API 呼叫時一模一樣
    state["media_actions"] = {"from_recording": start_from_recording, "autolabel": start_autolabel,
                              "export": start_export, "models": {"FromRecordingIn": FromRecordingIn,
                                                                 "AutolabelIn": AutolabelIn, "ExportIn": ExportIn}}
    from edgevision.media.campaign_api import register as _register_campaign
    _register_campaign(r, state=state, lib=lib, who=who, audit=audit, call=call, H=H)

    return r


def predict_triton(image_bytes: bytes, *, model_name: str, version: int, class_names: list[str], imgsz: int,
                   conf: float, task: str, targets: list[str], class_map: dict | None) -> list[dict]:
    """Triton 推論 → shapes（0–1 正規化）。模型類別對應到圖片集類別（同名或 class_map），對不上的丟掉。"""
    import cv2
    import numpy as np

    from edgevision.inference.predictor_triton import TritonPredictor

    url = os.environ.get("TRITON_URL", "triton:8000").removeprefix("http://")
    img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "圖片無法解碼")
    h, w = img.shape[:2]
    p = TritonPredictor(url=url, model_name=model_name, imgsz=imgsz, conf_threshold=conf, class_names=class_names)
    try:
        dets = p.predict(img).detections
    finally:
        p.close() if hasattr(p, "close") else None
    cmap = {n: n for n in class_names if n in targets} | (class_map or {})
    src = {"kind": "model", "engine": "registry", "model": model_name, "version": version}
    out = []
    for d in dets:
        cls = cmap.get(d.class_name)
        if cls is None:
            continue
        b = d.bbox
        rect = [[max(0.0, b.x1 / w), max(0.0, b.y1 / h)], [min(1.0, b.x2 / w), min(1.0, b.y2 / h)]]
        s = src | {"conf": round(float(d.confidence), 4)}
        if task == "segment" and d.mask_contour and len(d.mask_contour) >= 3:
            out.append({"type": "polygon", "class": cls, "source": s,
                        "points": [[min(1.0, max(0.0, x / w)), min(1.0, max(0.0, y / h))] for x, y, *_ in d.mask_contour]})
        elif task == "pose" and d.keypoints:
            out.append({"type": "keypoints", "class": cls, "bbox": rect, "source": s,
                        "points": [[min(1.0, max(0.0, k[0] / w)), min(1.0, max(0.0, k[1] / h)),
                                    2 if len(k) < 3 or k[2] >= 0.5 else 1] for k in d.keypoints]})
        else:
            out.append({"type": "rect", "class": cls, "points": rect, "source": s})
    return out
