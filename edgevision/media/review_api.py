"""標注審核 API，掛在 /api/media 底下，由 api.build_router 呼叫。

    設定    GET/PATCH /collections/{id}/review-config
    輪次    POST/GET /collections/{id}/rounds、GET /rounds/{id}、GET /rounds/{id}/items、
            GET /rounds/{id}/next、POST /rounds/{id}/close
    圖片    GET /items/{id}/image[?px=]（代傳原圖 / 縮圖）
    送出    POST /items/{id}/review（通過 / 背景 / 有疑問，記錄跟預標的差異與花了幾秒）
"""

from __future__ import annotations

import mimetypes
from typing import Any, Callable, Literal

from fastapi import APIRouter, HTTPException, Query, Response
from pydantic import BaseModel, Field

from edgevision.media import schemas as S
from edgevision.media.review import ReviewRounds


class RoundIn(BaseModel):
    name: str | None = Field(None, max_length=200, description="省略 = 第 N 輪")
    size: int | None = Field(None, ge=1, description="這一輪幾張；省略 = 設定的 round.size（範圍見 size_min / size_max）")
    blind_ratio: float | None = Field(None, ge=0, le=1, description="抽查比例；省略 = 設定的 round.blind_ratio")
    batch_ids: list[int] | None = Field(None, description="只從這些批次挑圖；省略 = 整個圖片集")


class ReviewIn(BaseModel):
    revision: int
    action: Literal["submit", "background", "doubt"]
    shapes: list[dict[str, Any]] = Field(default_factory=list, description="要留下的框（沒接受的候選框不要送）")
    seconds_spent: float | None = Field(None, ge=0, le=86400)
    clean_view_seen: bool | None = None
    note: str | None = Field(None, max_length=2000, description="action=doubt 必填")


def register(r: APIRouter, *, state: dict, lib: Callable, who: Callable, audit: Callable, call: Callable,
             H: dict) -> None:
    def rv() -> ReviewRounds:
        if "media_review" not in state:
            state["media_review"] = ReviewRounds(lib())
        return state["media_review"]

    def _cfg_out(cid: int) -> dict:
        return {**call(rv().config, cid), "override": rv().override(cid)}

    @r.get("/collections/{cid}/review-config", responses={200: {"model": S.ReviewConfigOut}})
    def get_review_config(cid: int):
        """合併後的完整審核設定（全域預設 ← 這個圖片集的覆蓋）。"""
        return _cfg_out(cid)

    @r.patch("/collections/{cid}/review-config", responses={200: {"model": S.ReviewConfigOut}})
    def patch_review_config(cid: int, body: dict[str, Any], x_actor: str | None = H["x_actor"],
                            x_actor_type: str | None = H["x_actor_type"], x_channel: str | None = H["x_channel"],
                            authorization: str | None = H["authorization"]):
        """部分更新這個圖片集的覆蓋（只寫要改的欄位；值給 null = 回到預設）。驗證不過整筆不寫。

        例：`{"round": {"size": 300}, "classes": {"tap": {"bulk_accept": false, "rare": true}}}`
        """
        ident = who(x_actor, x_actor_type, x_channel, authorization)
        before, after, _ = call(rv().patch_config, cid, body)
        audit(ident, "media.review_config.patch", str(cid), {"patch": body, "before": before, "after": after})
        return _cfg_out(cid)

    @r.post("/collections/{cid}/rounds", status_code=201, responses={201: {"model": S.RoundOut}})
    def create_round(cid: int, body: RoundIn, x_actor: str | None = H["x_actor"],
                     x_actor_type: str | None = H["x_actor_type"], x_channel: str | None = H["x_channel"],
                     authorization: str | None = H["authorization"]):
        """建一輪：從未標 / AI 預標、不在其他進行中輪次的圖裡，依優先分數挑 size 張，抽 blind_ratio 當抽查圖。"""
        ident = who(x_actor, x_actor_type, x_channel, authorization)
        out = call(rv().create_round, cid, actor=ident[0], **body.model_dump())
        audit(ident, "media.round.create", str(cid), body.model_dump() | {"round_id": out["id"],
                                                                        "n_items": out["n_items"]})
        return out

    @r.get("/collections/{cid}/rounds", responses={200: {"model": list[S.RoundOut]}})
    def list_rounds(cid: int):
        call(lib().collection, cid)
        return rv().rounds(cid)

    @r.get("/rounds/{rid}", responses={200: {"model": S.RoundOut}})
    def get_round(rid: int):
        return call(rv().round, rid)

    @r.get("/rounds/{rid}/items", responses={200: {"model": S.RoundItemsOut}})
    def round_items(rid: int, state: Literal["pending", "done", "doubt"] | None = None,
                    reason: str | None = None, class_: str | None = Query(None, alias="class"),
                    blind: bool | None = None, order: Literal["priority", "time", "conf"] = "priority",
                    offset: int = Query(0, ge=0), limit: int = Query(100, ge=1, le=500)):
        """縮圖牆清單（含縮圖網址與框；抽查圖審完前不給框）。"""
        return call(rv().round_items, rid, state=state, reason=reason, cls=class_, blind=blind, order=order,
                    offset=offset, limit=limit)

    @r.get("/rounds/{rid}/next", responses={200: {"model": S.RoundNextOut}})
    def round_next(rid: int, after_id: int | None = None, peek: int = Query(0, ge=0, le=10),
                   x_actor: str | None = H["x_actor"], x_actor_type: str | None = H["x_actor_type"],
                   x_channel: str | None = H["x_channel"], authorization: str | None = H["authorization"]):
        """依優先順序的下一張待審，並鎖給自己（after_id：從這張之後找）。item 為 null = 這輪審完了。"""
        ident = who(x_actor, x_actor_type, x_channel, authorization)
        item, remaining, peeks = call(rv().next_item, rid, actor=ident[0], after_item_id=after_id, peek=peek)
        return {"item": item, "remaining": remaining, "peek": peeks}

    @r.post("/rounds/{rid}/close", responses={200: {"model": S.RoundOut}})
    def close_round(rid: int, x_actor: str | None = H["x_actor"], x_actor_type: str | None = H["x_actor_type"],
                    x_channel: str | None = H["x_channel"], authorization: str | None = H["authorization"]):
        """關閉這一輪（之後只能看）。沒審的圖回到可選池，下一輪還會被挑。"""
        ident = who(x_actor, x_actor_type, x_channel, authorization)
        out = call(rv().close_round, rid, actor=ident[0])
        audit(ident, "media.round.close", str(rid), {"progress": out["progress"]})
        return out

    @r.get("/items/{item_id}/image", response_class=Response,
           responses={200: {"content": {t: {"schema": {"type": "string", "format": "binary"}}
                                        for t in ("image/jpeg", "image/png", "image/webp")}}})
    def item_image(item_id: int, px: int | None = Query(None, ge=32, le=4096)):
        """圖片本身（經 control-api 代傳，瀏覽器不用連得到 MinIO）。px = 長邊像素的 webp 縮圖（第一次會現做並存起來）。
        內容以 sha256 定址、永不變，可長期快取。"""
        it = call(lib().item, item_id, with_url=False)
        blob = lib().blob(it["blob_sha256"])
        if blob is None:
            raise HTTPException(404, "圖片不存在")
        if px:
            key = (blob.get("thumbs") or {}).get(str(px)) or rv()._make_thumb(blob, px)
            ctype = "image/webp"
        else:
            key = blob["object_key"]
            ctype = mimetypes.guess_type(key)[0] or "application/octet-stream"
        data = lib().objects.get(bucket=blob["bucket"], key=key)
        return Response(data, media_type=ctype, headers={"Cache-Control": "private, max-age=31536000, immutable"})

    @r.post("/items/{item_id}/review", responses={200: {"model": S.ReviewResultOut},
                                                   409: {"description": "revision 不符、被別人鎖住、或輪次已關閉"}})
    def review_item(item_id: int, body: ReviewIn, x_actor: str | None = H["x_actor"],
                    x_actor_type: str | None = H["x_actor_type"], x_channel: str | None = H["x_channel"],
                    authorization: str | None = H["authorization"]):
        """審核送出：submit（→ labeled）/ background（→ background，shapes 要空）/ doubt（不改圖，記備註、釋放鎖）。"""
        ident = who(x_actor, x_actor_type, x_channel, authorization)
        out = call(rv().review, item_id, actor=ident[0], action=body.action, revision=body.revision,
                   shapes=body.shapes, seconds_spent=body.seconds_spent, clean_view_seen=body.clean_view_seen,
                   note=body.note)
        audit(ident, "media.annotation.review", str(item_id),
              {"action": body.action, "n": len(body.shapes), "revision": out["revision"],
               "diff": {k: v for k, v in (out["diff"] or {}).items() if k != "per_class"}})
        return out


def item_with_review(state: dict, lib: Callable, item_id: int) -> dict:
    """GET /items/{id} 用：加上 review 區塊（不在輪次裡就跟原本一樣）。"""
    if "media_review" not in state:
        state["media_review"] = ReviewRounds(lib())
    return state["media_review"].item(item_id)


__all__ = ["register", "item_with_review", "HTTPException"]
