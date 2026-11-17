"""圖片集與圖片的管理動作：刪圖片集、類別改名 / 刪除、跨圖片集搜尋、圖片標籤與備註。

類別順序不開放調整：每個批次快照、混合出來的訓練快照、訓練出來的模型都依固定的類別順序編號，
順序一換，混在一起訓練時編號就錯位。前端要換顯示順序自己排。
"""

from __future__ import annotations

import json
import re

from edgevision.media.library import Conflict, _rows

SHA_RE = re.compile(r"^[0-9a-f]{8,64}$")


def delete_collection(lib, cid: int, *, confirm: str) -> dict:
    """刪圖片集（連同裡面的圖的關聯、標注、批次、審核輪次、匯出紀錄）。圖片庫的原圖與已經匯出的 LakeFS 快照不動。

    被標注專案用著（待審或測試集，含已結案的）→ 409：專案的紀錄還指向它。confirm 要等於圖片集名稱。
    """
    c = lib.collection(cid)
    if confirm != c["name"]:
        raise ValueError(f"要刪除請把 confirm 設成圖片集名稱「{c['name']}」")
    with lib.db.cursor() as cur:
        cur.execute("SELECT id, name, state FROM media_campaigns WHERE collection_id = %s OR golden_collection_id = %s",
                    (cid, cid))
        used = _rows(cur)
    if used:
        raise Conflict(f"圖片集 {cid} 被標注專案用著，不能刪：" + "、".join(f"{u['id']}「{u['name']}」" for u in used),
                       campaigns=used)
    with lib.db.cursor() as cur:
        cur.execute("SELECT count(*) FROM media_items WHERE collection_id = %s", (cid,))
        n_items = int(cur.fetchone()[0])
        cur.execute("SELECT count(*) FROM media_batches WHERE collection_id = %s", (cid,))
        n_batches = int(cur.fetchone()[0])
        cur.execute("DELETE FROM media_collections WHERE id = %s", (cid,))
    return {"id": cid, "name": c["name"], "items": n_items, "batches": n_batches}


def _rename_cfg(cfg: dict, rename: dict[str, str], delete: set[str]) -> dict:
    """審核設定的覆蓋（classes / swap_pairs / guide）跟著改名或拿掉。"""
    cfg = json.loads(json.dumps(cfg or {}))
    if isinstance(cfg.get("classes"), dict):
        cfg["classes"] = {rename.get(k, k): v for k, v in cfg["classes"].items() if k not in delete}
    if isinstance(cfg.get("swap_pairs"), list):
        cfg["swap_pairs"] = [[rename.get(a, a) for a in p] for p in cfg["swap_pairs"] if not set(p) & delete]
    if isinstance(cfg.get("guide"), list):
        cfg["guide"] = [g | {"class": rename.get(g.get("class"), g.get("class"))} for g in cfg["guide"]
                        if g.get("class") not in delete]
    return cfg


def change_classes(lib, cid: int, *, rename: dict[str, str] | None, delete: list[str] | None, actor: str) -> dict:
    """類別改名 / 刪除（順序不變；刪掉的位置後面的往前補）。

    改名：所有圖的框（含草稿）、審核設定、用這個圖片集的標注專案的退步判斷類別一起改。
    刪除：這個類別的框全部拿掉；拿光框的「已確認」圖變成 background（確認過、沒有物件），AI 預標的變回未標。
    每張改到的圖留一筆歷史（action = class_change）。已匯出的快照、訓練好的模型還是舊名稱。
    """
    rename = {k: v.strip() for k, v in (rename or {}).items() if k != v.strip()}
    delete_set = set(delete or [])
    c = lib.collection(cid)
    old = list(c["class_names"])
    missing = sorted((set(rename) | delete_set) - set(old))
    if missing:
        raise ValueError(f"圖片集沒有這些類別：{missing}（現有 {old}）")
    if set(rename) & delete_set:
        raise ValueError(f"同一個類別不能又改名又刪除：{sorted(set(rename) & delete_set)}")
    if any(not v for v in rename.values()):
        raise ValueError("新名稱不能是空白")
    new = [rename.get(n, n) for n in old if n not in delete_set]
    if len(set(new)) != len(new):
        raise ValueError(f"改完會有重複的類別：{new}")
    if not rename and not delete_set:
        return {"class_names": old, "items_changed": 0}
    if c.get("kpt_shape") and delete_set:
        raise ValueError("有關鍵點設定（kpt_shape）的圖片集不能刪類別")

    changed = 0
    with lib.db.cursor() as cur:
        cur.execute("SELECT id, shapes, status FROM media_items WHERE collection_id = %s AND shapes <> '[]'::jsonb",
                    (cid,))
        rows = cur.fetchall()
    for iid, shapes, status in rows:
        shapes = shapes if isinstance(shapes, list) else json.loads(shapes)
        out = [s | {"class": rename.get(s.get("class"), s.get("class"))} for s in shapes
               if s.get("class") not in delete_set]
        if out == shapes:
            continue
        st = status
        if not out and status == "labeled":
            st = "background"
        elif not out and status == "auto":
            st = "unlabeled"
        lib._write(int(iid), out, st, actor=actor, action="class_change")
        changed += 1
    with lib.db.cursor() as cur:
        cur.execute("UPDATE media_collections SET class_names = %s, review_config = %s, updated_at = now()"
                    " WHERE id = %s",
                    (json.dumps(new), json.dumps(_rename_cfg(c.get("review_config") or {}, rename, delete_set)), cid))
    return {"class_names": new, "renamed": rename, "deleted": sorted(delete_set), "items_changed": changed,
            "note": "已匯出的快照、訓練好的模型、歷史紀錄裡的框仍是舊名稱"}


def search_items(db, *, q: str | None = None, sha256: str | None = None, tag: str | None = None,
                 collection_id: int | None = None, limit: int = 100) -> list[dict]:
    """跨圖片集找圖：檔名關鍵字（不分大小寫）、sha256（前 8 碼以上）、標籤。"""
    cond, args = [], []
    if q:
        cond.append("lower(i.original_name) LIKE %s")
        args.append(f"%{q.lower()}%")
    if sha256:
        sha256 = sha256.lower()
        if not SHA_RE.match(sha256):
            raise ValueError("sha256 至少要給前 8 碼（16 進位）")
        cond.append("i.blob_sha256 LIKE %s")
        args.append(sha256 + "%")
    if tag:
        cond.append("i.tags ? %s")
        args.append(tag)
    if not cond:
        raise ValueError("至少給 q、sha256、tag 其中一個")
    if collection_id:
        cond.append("i.collection_id = %s")
        args.append(collection_id)
    with db.cursor() as cur:
        cur.execute(f"""SELECT i.id, i.collection_id, c.name AS collection_name, i.batch_id, i.blob_sha256,
                               i.original_name, i.status, i.camera_id, i.captured_at, i.tags, i.note, i.added_at
                          FROM media_items i JOIN media_collections c ON c.id = i.collection_id
                         WHERE {' AND '.join(cond)} ORDER BY i.id DESC LIMIT %s""", (*args, limit))
        return _rows(cur)


def set_item_meta(db, item_id: int, *, tags: list[str] | None = None, note: str | None = None,
                  clear_note: bool = False) -> dict:
    """圖片的標籤 / 備註：跟標注無關，不算修訂、不進匯出。tags 整組換掉。"""
    sets, args = [], []
    if tags is not None:
        sets.append("tags = %s")
        args.append(json.dumps(sorted({t.strip() for t in tags if t.strip()})))
    if clear_note:
        sets.append("note = NULL")
    elif note is not None:
        sets.append("note = %s")
        args.append(note)
    with db.cursor() as cur:
        if sets:
            cur.execute(f"UPDATE media_items SET {', '.join(sets)} WHERE id = %s", (*args, item_id))
        cur.execute("SELECT id, tags, note FROM media_items WHERE id = %s", (item_id,))
        r = cur.fetchone()
    if r is None:
        raise LookupError(f"沒有圖 {item_id}")
    return {"id": r[0], "tags": r[1], "note": r[2]}
