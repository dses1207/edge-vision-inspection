"""圖片庫 / 圖片集 / 標注的資料層。

API、背景任務（匯入、切幀、自動標注、匯出）、MCP 都經過這一層，SQL 只寫在這裡。
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import mimetypes
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO

from edgevision.media.shapes import validate_shapes

logger = logging.getLogger(__name__)

BUCKET = "raw"
IMAGE_PREFIX = "library/images"
VIDEO_PREFIX = "library/videos"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".ts"}

STATUSES = ("unlabeled", "auto", "labeled", "background")
EXPORTABLE = ("labeled", "background")
DEFAULT_LOCK_S = 300


class Conflict(Exception):
    """revision 不一致或被別人鎖住。API 轉成 409。"""

    def __init__(self, message: str, **detail: Any) -> None:
        super().__init__(message)
        self.detail = {"message": message, **detail}


class NotFound(Exception):
    pass


def blob_key(sha256: str, ext: str, kind: str = "image") -> str:
    prefix = IMAGE_PREFIX if kind == "image" else VIDEO_PREFIX
    return f"{prefix}/{sha256[:2]}/{sha256}{ext.lower()}"


def probe_image(data: bytes) -> tuple[int, int]:
    """讀圖寬高；解不開就不是圖（擋掉亂傳的檔）。"""
    import cv2
    import numpy as np

    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError("無法解碼的圖片")
    return int(img.shape[1]), int(img.shape[0])


def _rows(cur) -> list[dict]:
    cols = [c.name for c in cur.description]
    return [dict(zip(cols, r)) for r in cur.fetchall()]


def _row(cur) -> dict | None:
    r = cur.fetchone()
    return dict(zip([c.name for c in cur.description], r)) if r else None


@dataclass(slots=True)
class BlobRef:
    sha256: str
    width: int | None
    height: int | None
    created: bool          # False = 同一張圖已在圖片庫


def _org(v: str | None) -> str | None:
    """分類正規化：去頭尾空白；空的 = 未分類（NULL）。"""
    v = (v or "").strip()
    if len(v) > 100:
        raise ValueError("分類最多 100 個字")
    return v or None


class MediaLibrary:
    def __init__(self, db, objects) -> None:
        self.db = db
        self.objects = objects

    # ── 圖片庫（blob）───────────────────────────────────

    def put_image(self, data: bytes, *, filename: str, actor: str) -> BlobRef:
        """存一張圖：內容 sha256 當身分，已存在就不重傳。"""
        sha = hashlib.sha256(data).hexdigest()
        with self.db.cursor() as cur:
            cur.execute("SELECT width, height FROM media_blobs WHERE sha256 = %s", (sha,))
            hit = cur.fetchone()
        if hit:
            return BlobRef(sha, hit[0], hit[1], created=False)
        ext = Path(filename).suffix.lower()
        if ext not in IMAGE_EXTS:
            ext = ".jpg"
        w, h = probe_image(data)
        key = blob_key(sha, ext)
        ctype = mimetypes.types_map.get(ext, "application/octet-stream")
        self.objects.put(bucket=BUCKET, key=key, data=io.BytesIO(data), length=len(data), content_type=ctype)
        with self.db.cursor() as cur:
            cur.execute(
                "INSERT INTO media_blobs (sha256, kind, bucket, object_key, content_type, size_bytes, width, height,"
                " created_by) VALUES (%s,'image',%s,%s,%s,%s,%s,%s,%s) ON CONFLICT (sha256) DO NOTHING",
                (sha, BUCKET, key, ctype, len(data), w, h, actor))
        return BlobRef(sha, w, h, created=True)

    def put_video(self, path: Path, *, actor: str, duration_s: float | None = None,
                  size: tuple[int, int] | None = None) -> str:
        h = hashlib.sha256()
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(8 << 20), b""):
                h.update(chunk)
        sha = h.hexdigest()
        with self.db.cursor() as cur:
            cur.execute("SELECT 1 FROM media_blobs WHERE sha256 = %s", (sha,))
            if cur.fetchone():
                return sha
        ext = path.suffix.lower() if path.suffix.lower() in VIDEO_EXTS else ".mp4"
        key = blob_key(sha, ext, "video")
        ctype = mimetypes.types_map.get(ext, "video/mp4")
        with path.open("rb") as fh:
            self.objects.put(bucket=BUCKET, key=key, data=fh, length=path.stat().st_size, content_type=ctype)
        with self.db.cursor() as cur:
            cur.execute(
                "INSERT INTO media_blobs (sha256, kind, bucket, object_key, content_type, size_bytes, width, height,"
                " duration_s, created_by) VALUES (%s,'video',%s,%s,%s,%s,%s,%s,%s,%s) ON CONFLICT (sha256) DO NOTHING",
                (sha, BUCKET, key, ctype, path.stat().st_size, size[0] if size else None,
                 size[1] if size else None, duration_s, actor))
        return sha

    def blob(self, sha256: str) -> dict | None:
        with self.db.cursor() as cur:
            cur.execute("SELECT * FROM media_blobs WHERE sha256 = %s", (sha256,))
            return _row(cur)

    def blob_bytes(self, sha256: str) -> bytes:
        b = self.blob(sha256)
        if b is None:
            raise NotFound(f"blob {sha256} 不存在")
        return self.objects.get(bucket=b["bucket"], key=b["object_key"])

    def image_url(self, blob: dict, *, minutes: int = 60) -> str:
        return self.objects.presigned_url(bucket=blob["bucket"], key=blob["object_key"], expires_minutes=minutes)

    def add_source(self, *, kind: str, uri: str, actor: str, blob_sha256: str | None = None,
                   camera_id: str | None = None, started_at=None, params: dict | None = None) -> int:
        with self.db.cursor() as cur:
            cur.execute(
                "INSERT INTO media_sources (kind, uri, blob_sha256, camera_id, started_at, params, created_by)"
                " VALUES (%s,%s,%s,%s,%s,%s,%s) RETURNING id",
                (kind, uri, blob_sha256, camera_id, started_at, json.dumps(params or {}), actor))
            return int(cur.fetchone()[0])

    # ── 圖片集 ───────────────────────────────────────────

    def create_collection(self, *, name: str, actor: str, class_names: list[str] | None = None,
                          description: str = "", kpt_shape: list[int] | None = None,
                          line_id: str | None = None, camera_id: str | None = None,
                          org: str | None = None) -> dict:
        """圖片集不綁輸出格式（框 / 多邊形 / 關鍵點在匯出時選）；kpt_shape 只給關鍵點標注用。
        org：分類（自由文字，給人找圖片集用）；空白 = 未分類。"""
        with self.db.cursor() as cur:
            cur.execute("SELECT 1 FROM media_collections WHERE name = %s", (name,))
            if cur.fetchone():
                raise Conflict(f"圖片集名稱 {name!r} 已存在")
            cur.execute(
                "INSERT INTO media_collections (name, description, class_names, kpt_shape, line_id, camera_id,"
                " org, created_by) VALUES (%s,%s,%s,%s,%s,%s,%s,%s) RETURNING id",
                (name, description, json.dumps(class_names or []),
                 json.dumps(kpt_shape) if kpt_shape else None, line_id, camera_id, _org(org), actor))
            cid = int(cur.fetchone()[0])
        return self.collection(cid)

    def update_collection(self, cid: int, **fields) -> dict:
        """改名稱 / 說明 / 類別 / 分類 / 封存。類別只能【往後加】：改動既有順序會讓已標的類別編號全錯。
        org 給空字串 = 清掉分類（變成未分類）。"""
        cur_c = self.collection(cid)
        sets, params = [], []
        if "class_names" in fields and fields["class_names"] is not None:
            new = list(fields["class_names"])
            old = cur_c["class_names"]
            if new[:len(old)] != old:
                raise ValueError(f"類別只能往後加，不能改既有順序或刪除（現有 {old}）")
            sets.append("class_names = %s"); params.append(json.dumps(new))
        for k in ("name", "description", "archived", "line_id", "camera_id"):
            if fields.get(k) is not None:
                sets.append(f"{k} = %s"); params.append(fields[k])
        if fields.get("org") is not None:
            sets.append("org = %s"); params.append(_org(fields["org"]))
        if not sets:
            return cur_c
        with self.db.cursor() as cur:
            cur.execute(f"UPDATE media_collections SET {', '.join(sets)}, updated_at = now() WHERE id = %s",
                        (*params, cid))
        return self.collection(cid)

    def collection(self, cid: int) -> dict:
        with self.db.cursor() as cur:
            cur.execute("SELECT * FROM media_collections WHERE id = %s", (cid,))
            c = _row(cur)
        if c is None:
            raise NotFound(f"圖片集 {cid} 不存在")
        c["progress"] = self.progress(cid)
        return c

    def list_collections(self, *, include_archived: bool = False, org: str | None = None) -> list[dict]:
        """org：只列這個分類；給空字串 = 只列未分類的。"""
        conds, args = ([] if include_archived else ["NOT c.archived"]), []
        if org is not None:
            if _org(org) is None:
                conds.append("c.org IS NULL")
            else:
                conds.append("c.org = %s"); args.append(_org(org))
        where = ("WHERE " + " AND ".join(conds)) if conds else ""
        with self.db.cursor() as cur:
            cur.execute(
                f"""SELECT c.*, count(i.id) AS n_items,
                           count(i.id) FILTER (WHERE i.status IN ('labeled','background')) AS n_done
                    FROM media_collections c LEFT JOIN media_items i ON i.collection_id = c.id
                    {where} GROUP BY c.id ORDER BY c.updated_at DESC""", args)
            return _rows(cur)

    def orgs(self, *, include_archived: bool = False) -> list[dict]:
        """用過的分類與各有幾個圖片集（前端下拉用）；未分類的 org = null，排最後。"""
        where = "" if include_archived else "WHERE NOT archived"
        with self.db.cursor() as cur:
            cur.execute(f"SELECT org, count(*) FROM media_collections {where}"
                        " GROUP BY org ORDER BY org NULLS LAST")
            return [{"org": c, "n_collections": int(n)} for c, n in cur.fetchall()]

    def progress(self, cid: int) -> dict:
        with self.db.cursor() as cur:
            cur.execute("SELECT status, count(*) FROM media_items WHERE collection_id = %s GROUP BY status", (cid,))
            counts = {s: 0 for s in STATUSES} | {r[0]: int(r[1]) for r in cur.fetchall()}
        counts["total"] = sum(counts[s] for s in STATUSES)
        return counts

    # ── 批次──────

    BATCH_KINDS = ("upload", "zip", "video", "nas", "copy", "alert")
    _KIND_LABEL = {"upload": "上傳", "zip": "zip 匯入", "video": "影片", "nas": "NAS 錄影", "copy": "複製", "alert": "告警回流"}

    def create_batch(self, cid: int, *, kind: str, actor: str, name: str | None = None,
                     params: dict | None = None, task_id: str | None = None) -> dict:
        """name 省略 = 「<來源> <月-日時:分>」。"""
        if kind not in self.BATCH_KINDS:
            raise ValueError(f"批次種類只能是 {self.BATCH_KINDS}")
        self.collection(cid)
        if name is not None and not name.strip():
            raise ValueError("批次名稱不能是空白")
        with self.db.cursor() as cur:
            cur.execute(
                """INSERT INTO media_batches (collection_id, name, kind, params, task_id, created_by)
                   VALUES (%s, coalesce(%s, %s || ' ' || to_char(now() AT TIME ZONE 'Asia/Taipei', 'MM-DD HH24:MI')),
                           %s, %s, %s, %s) RETURNING id""",
                (cid, name.strip() if name else None, self._KIND_LABEL[kind], kind, json.dumps(params or {}, default=str),
                 task_id, actor))
            bid = int(cur.fetchone()[0])
        return self.batch(bid)

    def append_batch(self, cid: int, batch_id: int, *, kind: str, actor: str, params: dict | None = None) -> dict:
        """加進既有批次（不開新批次）。批次要屬於這個圖片集；每次加入記在 params.appends。

        標注專案用批次簽章（張數 / revision / 最大 id）判斷要不要重新匯出，舊批次多了圖下一代會自動重匯，不會漏。
        """
        if kind not in self.BATCH_KINDS:
            raise ValueError(f"批次種類只能是 {self.BATCH_KINDS}")
        b = self.batch(batch_id)
        if b["collection_id"] != cid:
            raise ValueError(f"批次 {batch_id} 不屬於圖片集 {cid}")
        entry = {"kind": kind, "actor": actor, **(params or {})}
        with self.db.cursor() as cur:
            cur.execute("""UPDATE media_batches SET params = jsonb_set(coalesce(params, '{}'::jsonb), '{appends}',
                                  coalesce(params->'appends', '[]'::jsonb)
                                  || jsonb_build_array(%s::jsonb || jsonb_build_object('at', now())))
                           WHERE id = %s""", (json.dumps(entry, default=str), batch_id))
        return self.batch(batch_id)

    def add_append_task(self, batch_id: int, task_id: str) -> None:
        """加進既有批次的匯入任務：記在最後一筆 appends（批次的 task_id 保持建立時那個）。"""
        with self.db.cursor() as cur:
            cur.execute("""UPDATE media_batches SET params = jsonb_set(params,
                                  ARRAY['appends', (jsonb_array_length(params->'appends') - 1)::text, 'task_id'],
                                  to_jsonb(%s::text)) WHERE id = %s""", (task_id, batch_id))

    def set_batch_task(self, batch_id: int, task_id: str) -> None:
        with self.db.cursor() as cur:
            cur.execute("UPDATE media_batches SET task_id = %s WHERE id = %s", (task_id, batch_id))

    def _batch_rows(self, where: str, params: tuple) -> list[dict]:
        with self.db.cursor() as cur:
            cur.execute(
                f"""SELECT b.*, count(i.id) AS n_items,
                           count(i.id) FILTER (WHERE i.status = 'unlabeled') AS n_unlabeled,
                           count(i.id) FILTER (WHERE i.status = 'auto') AS n_auto,
                           count(i.id) FILTER (WHERE i.status IN ('labeled','background')) AS n_done
                    FROM media_batches b LEFT JOIN media_items i ON i.batch_id = b.id
                    WHERE {where} GROUP BY b.id ORDER BY b.id""", params)
            return _rows(cur)

    def batch(self, batch_id: int) -> dict:
        rows = self._batch_rows("b.id = %s", (batch_id,))
        if not rows:
            raise NotFound(f"批次 {batch_id} 不存在")
        return rows[0]

    def batches(self, cid: int) -> list[dict]:
        self.collection(cid)
        return self._batch_rows("b.collection_id = %s", (cid,))

    def update_batch(self, batch_id: int, *, name: str) -> dict:
        if not name.strip():
            raise ValueError("批次名稱不能是空白")
        self.batch(batch_id)
        with self.db.cursor() as cur:
            cur.execute("UPDATE media_batches SET name = %s WHERE id = %s", (name.strip(), batch_id))
        return self.batch(batch_id)

    def delete_batch(self, batch_id: int) -> dict:
        """刪整個批次：這批的圖從圖片集移除、批次本身刪掉（圖片庫裡的原圖不動）。

        擋兩種：匯入還在跑（刪了之後任務會往不存在的批次塞圖）、進行中的標注專案拿它當初始批次。
        """
        b = self.batch(batch_id)
        appends = (b.get("params") or {}).get("appends") or []
        tasks = [t for t in [b.get("task_id"), *(a.get("task_id") for a in appends)] if t]
        with self.db.cursor() as cur:
            cur.execute("SELECT task_id FROM tasks WHERE task_id = ANY(%s) AND status IN ('queued', 'working')",
                        (tasks,))
            running = [r[0] for r in cur.fetchall()]
            if running:
                raise Conflict(f"批次 {batch_id} 的匯入還在跑（{', '.join(running)}），跑完或取消後再刪", tasks=running)
            cur.execute("SELECT id, name FROM media_campaigns WHERE state <> 'closed'"
                        " AND initial_batch_ids @> to_jsonb(ARRAY[%s::bigint]) ORDER BY id", (batch_id,))
            used = _rows(cur)
            if used:
                raise Conflict(f"批次 {batch_id} 是專案 {used[0]['id']}「{used[0]['name']}」的初始批次；"
                               "先從專案拿掉或把專案結案再刪", campaigns=[c["id"] for c in used])
            cur.execute("DELETE FROM media_items WHERE batch_id = %s", (batch_id,))
            removed = cur.rowcount
            cur.execute("DELETE FROM media_batches WHERE id = %s", (batch_id,))
        return {"batch_id": batch_id, "collection_id": b["collection_id"], "removed": removed}

    def check_batches(self, cid: int, batch_ids: list[int] | None) -> list[int] | None:
        """篩選用：批次都要屬於這個圖片集。None / [] = 不篩。"""
        if not batch_ids:
            return None
        ids = sorted({int(b) for b in batch_ids})
        with self.db.cursor() as cur:
            cur.execute("SELECT id FROM media_batches WHERE collection_id = %s AND id = ANY(%s)", (cid, ids))
            found = {int(r[0]) for r in cur.fetchall()}
        missing = [b for b in ids if b not in found]
        if missing:
            raise ValueError(f"批次 {missing} 不屬於圖片集 {cid}")
        return ids

    # ── 圖片集裡的圖 ─────────────────────────────────────

    def add_item(self, cid: int, sha256: str, *, actor: str, source_id: int | None = None,
                 source_offset_s: float | None = None, camera_id: str | None = None, captured_at=None,
                 original_name: str | None = None, shapes: list[dict] | None = None,
                 status: str | None = None, task_id: str | None = None,
                 select_info: dict | None = None, batch_id: int | None = None) -> tuple[int, bool]:
        """把圖放進圖片集。同一張圖已在集裡就回既有的 id（第二個值 False，批次不變）。帶標注的匯入直接記成 labeled。"""
        with self.db.cursor() as cur:
            cur.execute(
                "INSERT INTO media_items (collection_id, blob_sha256, source_id, source_offset_s, camera_id,"
                " captured_at, original_name, added_by, select_info, batch_id)"
                " VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)"
                " ON CONFLICT (collection_id, blob_sha256) DO NOTHING RETURNING id",
                (cid, sha256, source_id, source_offset_s, camera_id, captured_at, original_name, actor,
                 json.dumps(select_info) if select_info else None, batch_id))
            r = cur.fetchone()
            if r is None:
                cur.execute("SELECT id FROM media_items WHERE collection_id = %s AND blob_sha256 = %s", (cid, sha256))
                return int(cur.fetchone()[0]), False
            item_id = int(r[0])
            cur.execute("UPDATE media_collections SET updated_at = now() WHERE id = %s", (cid,))
        if shapes is not None or status:
            self._write(item_id, shapes or [], status or ("labeled" if shapes else "background"),
                        actor=actor, action="import", task_id=task_id)
        return item_id, True

    def copy_items(self, item_ids: list[int], target_cid: int, *, actor: str, with_labels: bool = False,
                   batch_id: int | None = None) -> int:
        """複製到別的圖片集：只多一列關聯，不複製檔案。batch_id = 目標圖片集的批次。"""
        target = self.collection(target_cid)
        n = 0
        for it in self.items_by_ids(item_ids):
            shapes = None
            if with_labels and it["status"] in EXPORTABLE:
                shapes = [s for s in it["shapes"] if s.get("class") in target["class_names"]]
            _, created = self.add_item(target_cid, it["blob_sha256"], actor=actor, source_id=it["source_id"],
                                       source_offset_s=it["source_offset_s"], camera_id=it["camera_id"],
                                       captured_at=it["captured_at"], original_name=it["original_name"],
                                       shapes=shapes, status=it["status"] if shapes is not None else None,
                                       select_info=it.get("select_info"),      # 挑圖資訊（審核排序用）也帶過去
                                       batch_id=batch_id)
            n += int(created)
        return n

    def remove_items(self, cid: int, item_ids: list[int]) -> int:
        """從圖片集移除（只刪關聯；圖片庫裡的原圖不動）。"""
        with self.db.cursor() as cur:
            cur.execute("DELETE FROM media_items WHERE collection_id = %s AND id = ANY(%s)", (cid, item_ids))
            return cur.rowcount

    def items_by_ids(self, item_ids: list[int]) -> list[dict]:
        with self.db.cursor() as cur:
            cur.execute("SELECT * FROM media_items WHERE id = ANY(%s) ORDER BY id", (item_ids,))
            return _rows(cur)

    def list_items(self, cid: int, *, status: list[str] | None = None, after_id: int = 0,
                   limit: int = 100, batch_ids: list[int] | None = None, tag: str | None = None) -> list[dict]:
        cond, params = "", []
        if status:
            cond += " AND i.status = ANY(%s)"; params.append(status)
        if batch_ids:
            cond += " AND i.batch_id = ANY(%s)"; params.append(list(batch_ids))
        if tag:
            cond += " AND i.tags ? %s"; params.append(tag)
        with self.db.cursor() as cur:
            cur.execute(
                f"""SELECT i.id, i.blob_sha256, i.batch_id, i.status, i.revision, i.original_name, i.camera_id,
                           i.captured_at,
                           i.updated_by, i.updated_at, i.locked_by, i.lock_until, i.tags, i.note,
                           jsonb_array_length(i.shapes) AS n_shapes, b.width, b.height
                    FROM media_items i JOIN media_blobs b ON b.sha256 = i.blob_sha256
                    WHERE i.collection_id = %s AND i.id > %s {cond}
                    ORDER BY i.id LIMIT %s""",
                (cid, after_id, *params, limit))
            return _rows(cur)

    def item(self, item_id: int, *, with_url: bool = True) -> dict:
        with self.db.cursor() as cur:
            cur.execute(
                """SELECT i.*, b.width, b.height, b.bucket, b.object_key, c.class_names, c.kpt_shape
                   FROM media_items i JOIN media_blobs b ON b.sha256 = i.blob_sha256
                   JOIN media_collections c ON c.id = i.collection_id WHERE i.id = %s""", (item_id,))
            it = _row(cur)
        if it is None:
            raise NotFound(f"圖片 {item_id} 不存在")
        if with_url:
            it["image_url"] = self.objects.presigned_url(bucket=it["bucket"], key=it["object_key"])
        return it

    # ── 標注：鎖、草稿、送出、續標 ──────────────────────

    def lock(self, item_id: int, *, actor: str, seconds: int = DEFAULT_LOCK_S) -> dict:
        """領取這張圖。別人持有中（且未逾時）→ Conflict；自己持有就延長。"""
        with self.db.cursor() as cur:
            cur.execute(
                """UPDATE media_items SET locked_by = %s, lock_until = now() + make_interval(secs => %s)
                   WHERE id = %s AND (locked_by IS NULL OR locked_by = %s OR lock_until < now())
                   RETURNING locked_by, lock_until""", (actor, seconds, item_id, actor))
            r = cur.fetchone()
            if r:
                return {"locked_by": r[0], "lock_until": r[1]}
            cur.execute("SELECT locked_by, lock_until FROM media_items WHERE id = %s", (item_id,))
            r = cur.fetchone()
        if r is None:
            raise NotFound(f"圖片 {item_id} 不存在")
        raise Conflict(f"{r[0]} 正在標這張", locked_by=r[0], lock_until=r[1].isoformat() if r[1] else None)

    def unlock(self, item_id: int, *, actor: str) -> None:
        with self.db.cursor() as cur:
            cur.execute("UPDATE media_items SET locked_by = NULL, lock_until = NULL WHERE id = %s AND locked_by = %s",
                        (item_id, actor))

    def save_draft(self, item_id: int, shapes: list[dict], *, revision: int, actor: str) -> dict:
        """自動存檔：覆蓋目前標注（不留歷史），狀態不變（未標的變成…仍未標，直到送出）。"""
        it = self.item(item_id, with_url=False)
        shapes = validate_shapes(shapes, it["class_names"])
        self.lock(item_id, actor=actor)
        with self.db.cursor() as cur:
            cur.execute(
                """UPDATE media_items SET shapes = %s, revision = revision + 1, updated_by = %s, updated_at = now()
                   WHERE id = %s AND revision = %s RETURNING revision""",
                (json.dumps(shapes), actor, item_id, revision))
            r = cur.fetchone()
        if r is None:
            raise Conflict("這張圖已被改過，請重新載入", current_revision=self.item(item_id, with_url=False)["revision"])
        return {"item_id": item_id, "revision": int(r[0])}

    def submit(self, item_id: int, shapes: list[dict], *, revision: int, actor: str,
               status: str = "labeled") -> dict:
        """送出：狀態 labeled（有物件）或 background（確認沒東西），留歷史、釋放鎖。"""
        if status not in EXPORTABLE:
            raise ValueError(f"送出狀態只能是 {EXPORTABLE}")
        it = self.item(item_id, with_url=False)
        shapes = validate_shapes(shapes, it["class_names"])
        if status == "background" and shapes:
            raise ValueError("background 表示確認沒有物件，shapes 必須是空的")
        if status == "labeled" and not shapes:
            raise ValueError("沒有任何標注：若確認這張沒有物件，請用 status=background")
        if it["locked_by"] and it["locked_by"] != actor and it["lock_until"] and _future(it["lock_until"]):
            raise Conflict(f"{it['locked_by']} 正在標這張", locked_by=it["locked_by"])
        rev = self._write(item_id, shapes, status, actor=actor, action="submit", expect_revision=revision)
        self.unlock(item_id, actor=actor)
        return {"item_id": item_id, "revision": rev, "status": status}

    def _write(self, item_id: int, shapes: list[dict], status: str, *, actor: str, action: str,
               expect_revision: int | None = None, task_id: str | None = None) -> int:
        """寫標注 + 歷史，同一個交易。"""
        with self.db.cursor() as cur:
            cond = "AND revision = %s" if expect_revision is not None else ""
            extra = (expect_revision,) if expect_revision is not None else ()
            cur.execute(
                f"""UPDATE media_items SET shapes = %s, status = %s, revision = revision + 1,
                       updated_by = %s, updated_at = now() WHERE id = %s {cond} RETURNING revision""",
                (json.dumps(shapes), status, actor, item_id, *extra))
            r = cur.fetchone()
            if r is None:
                raise Conflict("這張圖已被改過，請重新載入")
            rev = int(r[0])
            cur.execute(
                "INSERT INTO media_annotation_history (item_id, revision, status, shapes, action, actor, task_id)"
                " VALUES (%s,%s,%s,%s,%s,%s,%s)",
                (item_id, rev, status, json.dumps(shapes), action, actor, task_id))
        return rev

    def write_auto(self, item_id: int, shapes: list[dict], *, actor: str, task_id: str,
                   class_filter: list[str] | None = None, overwrite_human: bool = False) -> bool:
        """自動標注寫回。預設不動人看過的（labeled / background）；別人正在標的也跳過。回傳有沒有寫。"""
        from edgevision.media.shapes import merge_by_class

        it = self.item(item_id, with_url=False)
        if it["status"] in EXPORTABLE and not overwrite_human:
            return False
        if it["locked_by"] and it["lock_until"] and _future(it["lock_until"]):
            return False
        merged = merge_by_class(it["shapes"], shapes, class_filter)
        if overwrite_human and it["status"] in EXPORTABLE:
            status = it["status"] if merged else "background"
        else:
            status = "auto"
        self._write(item_id, merged, status, actor=actor, action="autolabel", task_id=task_id,
                    expect_revision=it["revision"])
        return True

    def next_item(self, cid: int, *, actor: str, status: list[str] | None = None,
                  after_id: int | None = None, lock: bool = True) -> dict | None:
        """續標：從上次停的地方（或 after_id）往後找下一張符合狀態、沒被別人鎖住的；到底就從頭繞一圈。"""
        status = status or ["unlabeled", "auto"]
        if after_id is None:
            with self.db.cursor() as cur:
                cur.execute("SELECT item_id FROM media_label_cursors WHERE collection_id = %s AND actor = %s",
                            (cid, actor))
                r = cur.fetchone()
                after_id = int(r[0]) if r and r[0] else 0
        sql = """SELECT id FROM media_items WHERE collection_id = %s AND status = ANY(%s) AND id {op} %s
                   AND (locked_by IS NULL OR locked_by = %s OR lock_until < now())
                 ORDER BY id LIMIT 1"""
        with self.db.cursor() as cur:
            cur.execute(sql.format(op=">"), (cid, status, after_id, actor))
            r = cur.fetchone()
            if r is None:
                cur.execute(sql.format(op=">="), (cid, status, 0, actor))
                r = cur.fetchone()
        if r is None:
            return None
        item_id = int(r[0])
        if lock:
            try:
                self.lock(item_id, actor=actor)
            except Conflict:          # 剛好被別人搶走：往後再找一張
                return self.next_item(cid, actor=actor, status=status, after_id=item_id, lock=lock)
        self.set_cursor(cid, actor=actor, item_id=item_id)
        return self.item(item_id)

    def set_cursor(self, cid: int, *, actor: str, item_id: int) -> None:
        with self.db.cursor() as cur:
            cur.execute(
                """INSERT INTO media_label_cursors (collection_id, actor, item_id) VALUES (%s,%s,%s)
                   ON CONFLICT (collection_id, actor) DO UPDATE SET item_id = EXCLUDED.item_id, updated_at = now()""",
                (cid, actor, item_id))

    def history(self, item_id: int, *, limit: int = 20) -> list[dict]:
        with self.db.cursor() as cur:
            cur.execute("SELECT * FROM media_annotation_history WHERE item_id = %s ORDER BY id DESC LIMIT %s",
                        (item_id, limit))
            return _rows(cur)

    # ── 匯出 ─────────────────────────────────────────────

    def exportable_items(self, cid: int, *, batch_ids: list[int] | None = None) -> list[dict]:
        cond, params = "", []
        if batch_ids:
            cond = "AND i.batch_id = ANY(%s)"; params.append(list(batch_ids))
        with self.db.cursor() as cur:
            cur.execute(
                f"""SELECT i.id, i.blob_sha256, i.batch_id, i.revision, i.status, i.shapes, b.bucket, b.object_key
                   FROM media_items i JOIN media_blobs b ON b.sha256 = i.blob_sha256
                   WHERE i.collection_id = %s AND i.status = ANY(%s) {cond} ORDER BY i.id""",
                (cid, list(EXPORTABLE), *params))
            return _rows(cur)

    def start_export(self, cid: int, *, repo: str, items: list[dict], actor: str, task_id: str | None,
                     batch_ids: list[int] | None = None) -> int:
        with self.db.cursor() as cur:
            cur.execute(
                "INSERT INTO media_exports (collection_id, dataset_repo, task_id, items, n_items, created_by, batch_ids)"
                " VALUES (%s,%s,%s,%s,%s,%s,%s) RETURNING id",
                (cid, repo, task_id, json.dumps([{"item_id": i["id"], "sha256": i["blob_sha256"],
                                                   "revision": i["revision"], "batch_id": i.get("batch_id")}
                                                  for i in items]),
                 len(items), actor, json.dumps(batch_ids) if batch_ids else None))
            return int(cur.fetchone()[0])

    def finish_export(self, export_id: int, *, dataset_ref: str | None, error: str | None = None) -> None:
        with self.db.cursor() as cur:
            cur.execute("UPDATE media_exports SET dataset_ref = %s, status = %s, error = %s WHERE id = %s",
                        (dataset_ref, "failed" if error else "completed", error, export_id))

    def exports(self, cid: int) -> list[dict]:
        with self.db.cursor() as cur:
            cur.execute("SELECT id, dataset_repo, dataset_ref, task_id, n_items, batch_ids, status, error, created_by,"
                        " created_at FROM media_exports WHERE collection_id = %s ORDER BY id DESC", (cid,))
            return _rows(cur)


def _future(ts) -> bool:
    from datetime import datetime, timezone
    return ts > datetime.now(timezone.utc)


def read_upload(fh: BinaryIO, limit: int) -> bytes:
    data = fh.read(limit + 1)
    if len(data) > limit:
        raise ValueError(f"單檔超過 {limit // (1 << 20)} MB，請改用 upload-url 直傳")
    return data
