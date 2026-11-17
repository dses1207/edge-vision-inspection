"""批次，對真的 PostgreSQL + MinIO 跑（沒設 MEDIA_TEST_DSN 就略過）。

每次把圖放進圖片集 = 一批；預標、建審核輪、匯出都能按批次篩。資料庫要先跑過 deploy/postgres/init 到 19-media-batches.sql。
"""
import io
import os
import uuid
import zipfile

import cv2
import numpy as np
import pytest

pytestmark = pytest.mark.skipif(not os.environ.get("MEDIA_TEST_DSN"), reason="需要 MEDIA_TEST_DSN")
CLASSES = ["board", "sheet", "glove", "tap", "mat"]


def _jpeg(seed: int) -> bytes:
    img = np.random.default_rng(seed).integers(0, 255, (48, 64, 3), dtype=np.uint8)
    return cv2.imencode(".jpg", img)[1].tobytes()


class FakeOrch:
    def submit(self, tid): pass
    def submit_media_import(self, tid): pass
    def submit_media_export(self, tid): pass


@pytest.fixture(scope="module")
def env():
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from edgevision.media.api import build_router
    from edgevision.storage.db import Database
    from edgevision.storage.objects import ObjectStore
    from edgevision.training.tracker import TaskStore

    os.environ["DATABASE_URL"] = os.environ["MEDIA_TEST_DSN"]
    db, objects = Database(), ObjectStore()
    try:
        objects._client.make_bucket("raw")
    except Exception:
        pass
    state = {"db": db, "objects": objects, "store": TaskStore(db), "orchestrator": FakeOrch()}
    app = FastAPI()
    app.include_router(build_router(state, lambda a, *_: (a or "alice", "human", "api")))
    return TestClient(app), state


def _upload(c, cid, seeds, batch_name=None):
    files = [("files", (f"{s}.jpg", _jpeg(s), "image/jpeg")) for s in seeds]
    params = {"batch_name": batch_name} if batch_name else None
    r = c.post(f"/api/media/collections/{cid}/images", files=files, params=params)
    assert r.status_code == 201, r.text
    return r.json()


def test_batches_end_to_end(env, monkeypatch):
    c, state = env
    from edgevision.media import jobs
    from edgevision.media.autolabel import select_items

    name = f"bt-{uuid.uuid4().hex[:6]}"
    cid = c.post("/api/media/collections", json={"name": name, "class_names": CLASSES}).json()["id"]
    other = c.post("/api/media/collections", json={"name": name + "-src", "class_names": CLASSES}).json()["id"]

    # ① 上傳 = 一批（有名字）；同一張圖再傳 → 新批次但沒新增圖，原本那張的批次不變
    u1 = _upload(c, cid, [1, 2, 3], "初始")
    b1 = u1["batch_id"]
    u2 = _upload(c, cid, [3, 4])
    b2 = u2["batch_id"]
    assert u2["added"] == 1
    bs = {b["id"]: b for b in c.get(f"/api/media/collections/{cid}/batches").json()}
    assert bs[b1]["name"] == "初始" and bs[b1]["kind"] == "upload" and bs[b1]["n_items"] == 3
    assert bs[b2]["name"].startswith("上傳 ") and bs[b2]["n_items"] == 1

    # ② zip 匯入（背景任務）：批次先建好、記著任務；圖帶著 batch_id
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("classes.txt", "\n".join(CLASSES) + "\n")
        z.writestr("images/z1.jpg", _jpeg(10))
        z.writestr("labels/z1.txt", "4 0.5 0.5 0.2 0.2\n")
        z.writestr("images/z2.jpg", _jpeg(11))
    key = f"uploads/media/bt_{uuid.uuid4().hex[:6]}.zip"
    state["objects"].put(bucket="raw", key=key, data=io.BytesIO(buf.getvalue()), length=len(buf.getvalue()))
    r = c.post(f"/api/media/collections/{cid}/import",
               json={"source_uri": f"s3://raw/{key}", "kind": "zip", "with_labels": True, "batch_name": "第一批"}).json()
    b3, tid = r["batch_id"], r["task_id"]
    assert jobs.run_import(tid) == 0
    assert state["store"].get(tid).result["import"]["batch_id"] == b3
    b = c.get(f"/api/media/batches/{b3}").json()
    assert (b["name"], b["kind"], b["task_id"], b["n_items"], b["n_done"], b["n_unlabeled"]) == (
        "第一批", "zip", tid, 2, 1, 1)

    # NAS 錄影：批次先建好（任務由 Job 跑，這裡只看批次與參數）
    r = c.post(f"/api/media/collections/{cid}/from-recording",
               json={"camera_id": "siteb12", "start": "2026-10-16T08:00:00", "end": "2026-10-16T09:00:00"}).json()
    nb = c.get(f"/api/media/batches/{r['batch_id']}").json()
    assert nb["kind"] == "nas" and nb["task_id"] == r["task_id"] and nb["params"]["camera_id"] == "siteb12"
    assert state["store"].get(r["task_id"]).params["batch_id"] == r["batch_id"]

    # ③ 從別的圖片集複製 = 目標的一批；來源不對的圖擋掉
    src = _upload(c, other, [20, 21, 1])
    src_ids = [x["item_id"] for x in src["results"]]
    r = c.post(f"/api/media/collections/{other}/copy",
               json={"item_ids": src_ids, "target_collection_id": cid, "batch_name": "第二批"}).json()
    b4 = r["batch_id"]
    assert r["added"] == 2                                            # 圖 1 已在目標裡 → 跳過
    assert c.get(f"/api/media/batches/{b4}").json()["params"]["from_collection_id"] == other
    assert c.post(f"/api/media/collections/{cid}/copy",
                  json={"item_ids": src_ids, "target_collection_id": other}).status_code == 400

    # ④ 改名、列圖篩批次
    assert c.patch(f"/api/media/batches/{b4}", json={"name": "第二批（siteb12 10/16）"}).json()["name"] == "第二批（siteb12 10/16）"
    assert c.patch(f"/api/media/batches/{b4}", json={"name": ""}).status_code == 422
    items = c.get(f"/api/media/collections/{cid}/items", params={"batch_id": [b3, b4]}).json()
    assert len(items) == 4 and {i["batch_id"] for i in items} == {b3, b4}

    # ⑤ 預標只跑指定批次；別的圖片集的批次 → 400
    lib = state["media"]
    picked = select_items(lib, cid, {"batch_ids": [b4]})
    assert {i["batch_id"] for i in picked} == {b4} and len(picked) == 2
    bad = c.post(f"/api/media/collections/{cid}/autolabel",
                 json={"engine": "registry", "model_name": "m", "batch_ids": [src["batch_id"]]})
    assert bad.status_code == 400

    # ⑥ 審核輪只從指定批次挑，輪次記著 batch_ids
    c.patch(f"/api/media/collections/{cid}/review-config", json={"round": {"size_min": 1}})
    rd = c.post(f"/api/media/collections/{cid}/rounds", json={"size": 10, "blind_ratio": 0, "batch_ids": [b4]}).json()
    assert rd["batch_ids"] == [b4] and rd["n_items"] == 2
    rd_items = c.get(f"/api/media/rounds/{rd['id']}/items").json()["items"]
    assert {i["id"] for i in rd_items} == {i["id"] for i in picked}

    # ⑦ 匯出只匯出指定批次（批次快照）：LakeFS metadata 與匯出紀錄都記批次
    seen = {}

    def fake_publish(root, repo, message, **kw):
        seen.update(kw, n=len(list((root / "images").rglob("*.jpg"))))
        return {"dataset_ref": f"lakefs://{repo}/abc", "repo": repo}
    monkeypatch.setattr("edgevision.training.ingest.publish", fake_publish)
    from edgevision.media.export import export_collection
    with pytest.raises(ValueError):
        export_collection(lib, cid, repo="x", message="m", actor="t", batch_ids=[src["batch_id"]])
    out = export_collection(lib, cid, repo="bt", message="m", actor="t", batch_ids=[b1, b3])
    assert seen["n"] == 1 and seen["extra_metadata"]["media_batches"] == [b1, b3]   # 只有 z1 是已確認
    rec = c.get(f"/api/media/collections/{cid}/exports").json()[0]
    assert rec["id"] == out["export_id"] and rec["batch_ids"] == [b1, b3] and rec["n_items"] == 1
    assert c.post(f"/api/media/collections/{cid}/export",
                  json={"repo": "bt", "batch_ids": [src["batch_id"]]}).status_code == 400


def test_append_to_existing_batch(env):
    """batch_id = 加進既有批次：上傳、zip 匯入、NAS 錄影、複製都可以；每次加入記在 params.appends。"""
    c, state = env
    from edgevision.media import jobs

    name = f"ap-{uuid.uuid4().hex[:6]}"
    cid = c.post("/api/media/collections", json={"name": name, "class_names": CLASSES}).json()["id"]
    other = c.post("/api/media/collections", json={"name": name + "-o", "class_names": CLASSES}).json()["id"]
    b = _upload(c, cid, [101, 102], "第一批")["batch_id"]

    # 上傳加進同一批
    files = [("files", (f"{s}.jpg", _jpeg(s), "image/jpeg")) for s in (103, 104)]
    r = c.post(f"/api/media/collections/{cid}/images", files=files, params={"batch_id": b})
    assert r.status_code == 201 and r.json()["batch_id"] == b, r.text
    got = c.get(f"/api/media/batches/{b}").json()
    assert got["n_items"] == 4 and got["name"] == "第一批" and got["params"]["appends"][0]["kind"] == "upload"
    assert len(c.get(f"/api/media/collections/{cid}/batches").json()) == 1          # 沒有多開一批

    # 擋：batch_id 與 batch_name 同時給、別的圖片集的批次、不存在的批次
    assert c.post(f"/api/media/collections/{cid}/images", files=files,
                  params={"batch_id": b, "batch_name": "x"}).status_code == 400
    ob = _upload(c, other, [105])["batch_id"]
    assert c.post(f"/api/media/collections/{cid}/images", files=files, params={"batch_id": ob}).status_code == 400
    assert c.post(f"/api/media/collections/{cid}/images", files=files, params={"batch_id": 99999999}).status_code == 404

    # zip 匯入加進同一批：批次的 task_id 保持原樣，任務記在 appends
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("images/a1.jpg", _jpeg(106))
    key = f"uploads/media/ap_{uuid.uuid4().hex[:6]}.zip"
    state["objects"].put(bucket="raw", key=key, data=io.BytesIO(buf.getvalue()), length=len(buf.getvalue()))
    r = c.post(f"/api/media/collections/{cid}/import",
               json={"source_uri": f"s3://raw/{key}", "kind": "zip", "batch_id": b}).json()
    assert r["batch_id"] == b
    assert jobs.run_import(r["task_id"]) == 0
    got = c.get(f"/api/media/batches/{b}").json()
    assert got["n_items"] == 5 and got["task_id"] is None
    assert [a.get("task_id") for a in got["params"]["appends"]] == [None, r["task_id"]]

    # NAS 錄影、複製也能加進來
    r = c.post(f"/api/media/collections/{cid}/from-recording",
               json={"camera_id": "siteb12", "start": "2026-10-16T08:00:00", "end": "2026-10-16T09:00:00",
                     "batch_id": b}).json()
    assert r["batch_id"] == b and state["store"].get(r["task_id"]).params["batch_id"] == b
    oid = c.get(f"/api/media/collections/{other}/items").json()[0]["id"]
    r = c.post(f"/api/media/collections/{other}/copy", json={"item_ids": [oid], "target_collection_id": cid,
                                                             "batch_id": b}).json()
    assert r == {"added": 1, "batch_id": b} or (r["added"] == 1 and r["batch_id"] == b)
    got = c.get(f"/api/media/batches/{b}").json()
    assert got["n_items"] == 6 and [a["kind"] for a in got["params"]["appends"]] == ["upload", "zip", "nas", "copy"]


def test_delete_batch(env):
    """刪批次：這批的圖從圖片集移除、批次刪掉，別批不動；匯入還在跑 / 進行中專案的初始批次 → 409。"""
    c, state = env
    name = f"del-{uuid.uuid4().hex[:6]}"
    cid = c.post("/api/media/collections", json={"name": name, "class_names": CLASSES}).json()["id"]
    b1 = _upload(c, cid, [201, 202], "要刪的")["batch_id"]
    b2 = _upload(c, cid, [203], "留著的")["batch_id"]

    r = c.delete(f"/api/media/batches/{b1}")
    assert r.status_code == 200 and r.json() == {"batch_id": b1, "collection_id": cid, "removed": 2}, r.text
    assert [b["id"] for b in c.get(f"/api/media/collections/{cid}/batches").json()] == [b2]
    assert [i["batch_id"] for i in c.get(f"/api/media/collections/{cid}/items").json()] == [b2]
    assert c.delete(f"/api/media/batches/{b1}").status_code == 404

    # 匯入還在跑
    r = c.post(f"/api/media/collections/{cid}/from-recording",
               json={"camera_id": "siteb12", "start": "2026-10-16T08:00:00", "end": "2026-10-16T09:00:00"}).json()
    r2 = c.delete(f"/api/media/batches/{r['batch_id']}")
    assert r2.status_code == 409 and r["task_id"] in r2.text

    # 進行中專案的初始批次
    with state["db"].cursor() as cur:
        cur.execute("INSERT INTO media_campaigns (name, collection_id, model_name, base_model, stage, initial_batch_ids,"
                    " created_by) VALUES (%s, %s, 'm', 'yolo26m.pt', 'training', %s, 'alice') RETURNING id",
                    (name, cid, f"[{b2}]"))
        camp = cur.fetchone()[0]
    assert c.delete(f"/api/media/batches/{b2}").status_code == 409
    with state["db"].cursor() as cur:
        cur.execute("UPDATE media_campaigns SET state = 'closed' WHERE id = %s", (camp,))
    assert c.delete(f"/api/media/batches/{b2}").status_code == 200


def test_collection_org(env):
    """分類：建立時給、修改、清掉（空字串）；清單依分類篩；分類清單含各幾個圖片集。"""
    c, _ = env
    tag = uuid.uuid4().hex[:6]
    cat = f"B 工序B {tag}"
    a = c.post("/api/media/collections", json={"name": f"cat-a-{tag}", "org": f"  {cat} "}).json()
    b = c.post("/api/media/collections", json={"name": f"cat-b-{tag}"}).json()
    assert a["org"] == cat and b["org"] is None                      # 去頭尾空白；沒給 = 未分類
    assert c.patch(f"/api/media/collections/{b['id']}", json={"org": cat}).json()["org"] == cat
    got = {x["id"] for x in c.get("/api/media/collections", params={"org": cat}).json()}
    assert got == {a["id"], b["id"]}
    cats = {x["org"]: x["n_collections"] for x in c.get("/api/media/collections/orgs").json()}
    assert cats[cat] == 2
    assert c.patch(f"/api/media/collections/{b['id']}", json={"org": ""}).json()["org"] is None
    assert b["id"] in {x["id"] for x in c.get("/api/media/collections", params={"org": ""}).json()}
    assert c.get(f"/api/media/collections/{a['id']}").json()["org"] == cat
