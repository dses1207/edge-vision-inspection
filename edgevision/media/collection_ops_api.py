"""圖片集 / 圖片管理 API（掛在 /api/media 底下；做法見 edgevision/media/collection_ops.py）。

    DELETE /collections/{cid}?confirm=<名稱>     刪圖片集（被標注專案用著 → 409）
    PATCH  /collections/{cid}/classes             類別改名 / 刪除（順序不開放調整）
    GET    /items/search?q=&sha256=&tag=          跨圖片集找圖
    PATCH  /items/{item_id}/meta                  圖片標籤 / 備註
    POST   /collections/{cid}/download            打包成 zip（YOLO / YOLO-seg / COCO / VOC；背景任務）
    GET    /downloads/{task_id}                   打包進度與下載網址
"""

from __future__ import annotations

from typing import Literal

from fastapi import HTTPException, Query
from pydantic import BaseModel, Field

from edgevision.media import collection_ops as O
from edgevision.media import schemas as S


class ClassChangeIn(BaseModel):
    rename: dict[str, str] | None = Field(None, description='{"舊名": "新名"}')
    delete: list[str] | None = Field(None, description="要刪的類別（這個類別的框全部拿掉）")


class DownloadIn(BaseModel):
    format: Literal["yolo", "yolo-seg", "yolo-pose", "coco", "voc"] = Field(
        description="yolo 框（多邊形取外接框）/ yolo-seg 多邊形（框轉四點）/ yolo-pose 關鍵點 / coco 框＋多邊形 / voc 框")
    statuses: list[Literal["labeled", "background", "auto", "unlabeled"]] = Field(
        default_factory=lambda: ["labeled", "background"],
        description="要哪些狀態的圖；預設只要人確認過的（同匯出）。加 auto = 含 AI 預標")
    batch_ids: list[int] | None = Field(None, description="只要這些批次")
    val_pct: int = Field(20, ge=0, le=100, description="驗證集比例（依 sha256 分，同匯出規則）；0 = 全部 train")


class ItemMetaIn(BaseModel):
    tags: list[str] | None = Field(None, description="整組換掉；不給 = 不改")
    note: str | None = Field(None, max_length=2000, description="不給 = 不改")
    clear_note: bool = False


def register(r, *, state: dict, lib, who, audit, call, H, new_task) -> None:
    from libs.contracts.training import TaskType

    @r.delete("/collections/{cid}", responses={200: {"model": S.CollectionDeletedOut},
                                               409: {"description": "被標注專案用著"}})
    def delete_collection(cid: int, confirm: str = Query(description="圖片集名稱（防手滑）"),
                          x_actor: str | None = H["x_actor"], x_actor_type: str | None = H["x_actor_type"],
                          x_channel: str | None = H["x_channel"], authorization: str | None = H["authorization"]):
        """刪圖片集：裡面的圖（關聯與標注）、批次、審核輪次、匯出紀錄一起刪；圖片庫的原圖、已匯出的 LakeFS 快照不動。
        被標注專案用著（待審或測試集，含已結案）→ 409。只是不想看到請用封存（PATCH archived=true）。"""
        ident = who(x_actor, x_actor_type, x_channel, authorization)
        out = call(O.delete_collection, lib(), cid, confirm=confirm)
        audit(ident, "media.collection.delete", str(cid), out)
        return out

    @r.patch("/collections/{cid}/classes", responses={200: {"model": S.ClassChangeOut}})
    def change_classes(cid: int, body: ClassChangeIn, x_actor: str | None = H["x_actor"],
                       x_actor_type: str | None = H["x_actor_type"], x_channel: str | None = H["x_channel"],
                       authorization: str | None = H["authorization"]):
        """類別改名 / 刪除。改名：所有框（含草稿）、審核設定、專案的退步判斷類別一起改。刪除：該類別的框全部拿掉
        （拿光框的已確認圖變 background、AI 預標的變回未標）。每張改到的圖留一筆歷史。
        順序不開放調整（快照與模型依順序編號）；新增類別照舊用 PATCH /collections/{cid} 往後加。"""
        ident = who(x_actor, x_actor_type, x_channel, authorization)
        out = call(O.change_classes, lib(), cid, rename=body.rename, delete=body.delete, actor=ident[0])
        audit(ident, "media.collection.classes", str(cid), body.model_dump() | {"items_changed": out["items_changed"]})
        return out

    @r.get("/items/search", responses={200: {"model": list[S.ItemSearchOut]}})
    def search_items(q: str | None = Query(None, description="檔名關鍵字（不分大小寫）"),
                     sha256: str | None = Query(None, description="圖片 sha256（前 8 碼以上）"),
                     tag: str | None = None, collection_id: int | None = None,
                     limit: int = Query(100, ge=1, le=1000)):
        """跨圖片集找圖。同一張圖在幾個圖片集就回幾筆（各自的標注狀態）。"""
        return call(O.search_items, state["db"], q=q, sha256=sha256, tag=tag, collection_id=collection_id,
                    limit=limit)

    @r.patch("/items/{item_id}/meta", responses={200: {"model": S.ItemMetaOut}})
    def item_meta(item_id: int, body: ItemMetaIn, x_actor: str | None = H["x_actor"],
                  x_actor_type: str | None = H["x_actor_type"], x_channel: str | None = H["x_channel"],
                  authorization: str | None = H["authorization"]):
        """圖片的標籤與備註（跟標注無關：不算修訂、不進匯出、不擋別人標）。列表可用 ?tag= 篩。"""
        ident = who(x_actor, x_actor_type, x_channel, authorization)
        try:
            out = O.set_item_meta(state["db"], item_id, tags=body.tags, note=body.note, clear_note=body.clear_note)
        except LookupError as exc:
            raise HTTPException(404, str(exc)) from exc
        audit(ident, "media.item.meta", str(item_id), body.model_dump(exclude_unset=True))
        return out

    @r.post("/collections/{cid}/download", status_code=202, responses={202: {"model": S.TaskCreatedOut}})
    def download(cid: int, body: DownloadIn, x_actor: str | None = H["x_actor"],
                 x_actor_type: str | None = H["x_actor_type"], x_channel: str | None = H["x_channel"],
                 authorization: str | None = H["authorization"]):
        """把圖片集打包成 zip（背景任務）。完成後 GET /downloads/{task_id} 拿下載網址。
        yolo：框（多邊形取外接框）；yolo-seg：多邊形（框轉四點）；coco：bbox＋segmentation；voc：bbox。都附 items.csv。"""
        ident = who(x_actor, x_actor_type, x_channel, authorization)
        call(lib().collection, cid)
        call(lib().check_batches, cid, body.batch_ids)
        params = {"collection_id": cid, **body.model_dump()}
        tid = new_task(TaskType.MEDIA_EXPORT, ident, params)
        audit(ident, "media.collection.download", str(cid), params, task_id=tid)
        state["orchestrator"].submit_media_export(tid)
        return {"task_id": tid, "queue_position": None}

    @r.get("/downloads/{task_id}", responses={200: {"model": S.DownloadOut}})
    def get_download(task_id: str):
        """打包狀態；完成時 url = 下載網址（2 小時有效，過期再查一次就會換新的）。"""
        t = state["store"].get(task_id)
        if t is None or not (t.params or {}).get("format"):
            raise HTTPException(404, f"沒有打包任務 {task_id}")
        res = ((t.result or {}).get("download") or {})
        url = None
        if res.get("uri"):
            b, k = state["objects"].parse_uri(res["uri"])
            url = state["objects"].presigned_url(bucket=b, key=k, expires_minutes=120)
        return {"task_id": task_id, "status": t.status, "status_message": t.status_message,
                "format": t.params["format"], "collection_id": t.params.get("collection_id"),
                "repo": t.params.get("repo"), "version_id": t.params.get("version_id"),
                "items": res.get("items"), "splits": res.get("splits"), "size_bytes": res.get("size_bytes"),
                "url": url}
