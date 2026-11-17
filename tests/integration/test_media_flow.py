"""圖片庫整條流程，對真的 PostgreSQL + MinIO 跑（沒設 MEDIA_TEST_DSN 就略過）。

    MEDIA_TEST_DSN=postgresql://…/edgevision_media_dev MINIO_ENDPOINT=… MINIO_ROOT_USER=… MINIO_ROOT_PASSWORD=… \
        pytest tests/integration/test_media_flow.py

資料庫要先跑過 deploy/postgres/init/02-schema.sql、12-media-library.sql、14-media-select.sql。
"""
import io
import os
import uuid
import zipfile

import cv2
import numpy as np
import pytest

pytestmark = pytest.mark.skipif(not os.environ.get("MEDIA_TEST_DSN"), reason="需要 MEDIA_TEST_DSN")


def _jpeg(seed: int, size=(64, 48)) -> bytes:
    rng = np.random.default_rng(seed)
    img = rng.integers(0, 255, (size[1], size[0], 3), dtype=np.uint8)
    return cv2.imencode(".jpg", img)[1].tobytes()


class FakeOrch:
    def __init__(self):
        self.submitted, self.imports = [], []

    def submit(self, tid):
        self.submitted.append(tid)

    def submit_media_import(self, tid):
        self.imports.append(tid)

    def submit_media_export(self, tid):
        self.imports.append(tid)


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
    actors = {"alice": ("alice", "human", "api"), "bob": ("bob", "human", "api")}

    def actor_fn(x_actor, *_):
        return actors[x_actor or "alice"]

    app = FastAPI()
    app.include_router(build_router(state, actor_fn))
    return TestClient(app), state


def _hdr(who):
    return {"X-Actor": who}


def test_full_labeling_flow(env):
    c, state = env
    name = f"it-{uuid.uuid4().hex[:6]}"
    col = c.post("/api/media/collections", json={"name": name, "class_names": ["board", "hand"]},
                 headers=_hdr("alice")).json()
    cid = col["id"]
    assert c.post("/api/media/collections", json={"name": name}, headers=_hdr("alice")).status_code == 409

    # ① 上傳：同一張圖傳兩次只存一份、只進集合一次；非圖片擋掉
    a, b = _jpeg(1), _jpeg(2)
    r = c.post(f"/api/media/collections/{cid}/images", headers=_hdr("alice"),
               files=[("files", ("a.jpg", a, "image/jpeg")), ("files", ("b.jpg", b, "image/jpeg")),
                      ("files", ("a_copy.jpg", a, "image/jpeg")), ("files", ("x.jpg", b"not an image", "image/jpeg"))]).json()
    assert r["added"] == 2
    assert [x.get("added") for x in r["results"][:3]] == [True, True, False]
    assert "error" in r["results"][3]
    prog = c.get(f"/api/media/collections/{cid}/progress").json()
    assert prog["unlabeled"] == 2 and prog["total"] == 2

    # ② 續標：alice 拿到第一張並領取；bob 拿到的是另一張（第一張被鎖）
    n1 = c.get(f"/api/media/collections/{cid}/next", headers=_hdr("alice")).json()
    it1 = n1["item"]
    it2 = c.get(f"/api/media/collections/{cid}/next", headers=_hdr("bob")).json()["item"]
    assert n1["remaining"] == 2
    assert it1["id"] != it2["id"] and it1["locked_by"] == "alice" and it2["locked_by"] == "bob"
    assert it1["image_url"].startswith("http")
    assert c.post(f"/api/media/items/{it1['id']}/lock", headers=_hdr("bob")).status_code == 409

    # ③ 草稿：revision 遞增；用舊 revision 存 → 409（不會互蓋）
    shapes = [{"type": "rect", "class": "hand", "points": [[0.1, 0.1], [0.4, 0.5]]}]
    d = c.put(f"/api/media/items/{it1['id']}/draft", json={"shapes": shapes, "revision": 0}, headers=_hdr("alice"))
    assert d.status_code == 200 and d.json()["revision"] == 1
    assert c.put(f"/api/media/items/{it1['id']}/draft", json={"shapes": shapes, "revision": 0},
                 headers=_hdr("alice")).status_code == 409
    got = c.get(f"/api/media/items/{it1['id']}").json()
    assert got["status"] == "unlabeled" and got["shapes"][0]["class"] == "hand"     # 草稿不改狀態

    # 類別不在清單、或 labeled 卻沒標注 → 400
    assert c.put(f"/api/media/items/{it1['id']}/draft", headers=_hdr("alice"),
                 json={"shapes": [{"type": "rect", "class": "cat", "points": [[0, 0], [1, 1]]}],
                       "revision": 1}).status_code == 400
    assert c.post(f"/api/media/items/{it2['id']}/submit", json={"shapes": [], "revision": 0},
                  headers=_hdr("bob")).status_code == 400

    # ④ 送出：labeled / background；留歷史、釋放鎖
    s1 = c.post(f"/api/media/items/{it1['id']}/submit", json={"shapes": shapes, "revision": 1},
                headers=_hdr("alice")).json()
    assert s1["status"] == "labeled" and s1["revision"] == 2
    s2 = c.post(f"/api/media/items/{it2['id']}/submit", json={"shapes": [], "revision": 0, "status": "background"},
                headers=_hdr("bob")).json()
    assert s2["status"] == "background"
    hist = c.get(f"/api/media/items/{it1['id']}/history").json()
    assert hist[0]["action"] == "submit" and hist[0]["actor"] == "alice"
    assert c.get(f"/api/media/items/{it1['id']}").json()["locked_by"] is None
    done = c.get(f"/api/media/collections/{cid}/next", headers=_hdr("alice")).json()
    assert done == {"item": None, "remaining": 0}                                                 # 全標完

    # ⑤ 自動標注不動人看過的圖
    lib = state["media"]
    assert lib.write_auto(it1["id"], [], actor="bot", task_id="t") is False

    # ⑥ 複製到別的集合：不複製檔案，帶標注
    col2 = c.post("/api/media/collections", json={"name": name + "-b", "class_names": ["hand"]},
                  headers=_hdr("alice")).json()
    assert c.post(f"/api/media/collections/{cid}/copy", headers=_hdr("alice"),
                  json={"item_ids": [it1["id"]], "target_collection_id": col2["id"], "with_labels": True}).json()["added"] == 1
    p2 = c.get(f"/api/media/collections/{col2['id']}/progress").json()
    assert p2["labeled"] == 1

    # ⑦ 匯出前置：只有人看過的、train/val 依內容雜湊
    from edgevision.media.export import build_yolo_dir
    items = lib.exportable_items(cid)
    assert len(items) == 2
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as td:
        counts = build_yolo_dir(lib, lib.collection(cid), items, Path(td), val_pct=20)
        assert counts["val"] >= 1 and counts["train"] + counts["val"] == 2
        labels = sorted(p.read_text() for p in Path(td).glob("labels/*/*.txt"))
        assert labels == ["", "1 0.250000 0.300000 0.300000 0.400000\n"]
        assert counts["task"] == "detect" and counts["converted"]["box_to_polygon"] == 0
    with tempfile.TemporaryDirectory() as td:                            # 輸出格式是匯出的選項，不是圖片集的屬性
        counts = build_yolo_dir(lib, lib.collection(cid), items, Path(td), val_pct=20, task="segment")
        labels = sorted(p.read_text() for p in Path(td).glob("labels/*/*.txt"))
        assert labels[1].split()[0] == "1" and len(labels[1].split()) == 9    # 框 → 四點多邊形
        assert counts["task"] == "segment" and counts["converted"]["box_to_polygon"] == 1
    with pytest.raises(ValueError):
        build_yolo_dir(lib, lib.collection(cid), items, Path("/tmp"), val_pct=20, task="obb")

    # ⑧ 類別只能往後加
    assert c.patch(f"/api/media/collections/{cid}", json={"class_names": ["hand", "board"]},
                   headers=_hdr("alice")).status_code == 400
    assert c.patch(f"/api/media/collections/{cid}", json={"class_names": ["board", "hand", "glove"]},
                   headers=_hdr("alice")).json()["class_names"] == ["board", "hand", "glove"]


def test_zip_import_with_labels_and_video(env, tmp_path):
    """匯入任務（jobs.run_import）：帶標注的 zip、影片切幀。"""
    c, state = env
    from edgevision.media import jobs

    name = f"it-zip-{uuid.uuid4().hex[:6]}"
    cid = c.post("/api/media/collections", json={"name": name}, headers=_hdr("alice")).json()["id"]
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("ds/classes.txt", "board\nhand\n")
        z.writestr("ds/images/train/p1.jpg", _jpeg(10))
        z.writestr("ds/labels/train/p1.txt", "1 0.1 0.1 0.5 0.1 0.3 0.6\n")
        z.writestr("ds/images/val/p2.jpg", _jpeg(11))          # 沒標籤檔 → 保持未標
    objects = state["objects"]
    key = f"uploads/media/test_{uuid.uuid4().hex[:6]}.zip"
    data = buf.getvalue()
    objects.put(bucket="raw", key=key, data=io.BytesIO(data), length=len(data))
    tid = c.post(f"/api/media/collections/{cid}/import", headers=_hdr("alice"),
                 json={"source_uri": f"s3://raw/{key}", "kind": "zip", "with_labels": True}).json()["task_id"]
    assert jobs.run_import(tid) == 0
    t = state["store"].get(tid)
    assert t.status == "completed" and t.result["import"]["added"] == 2 and t.result["import"]["with_labels"] == 1
    col = c.get(f"/api/media/collections/{cid}").json()
    assert col["class_names"] == ["board", "hand"]                # 集合沒類別 → 採用 zip 的
    assert col["progress"]["labeled"] == 1 and col["progress"]["unlabeled"] == 1

    # 影片：每 0.5 秒一張，原影片進圖片庫
    vid = tmp_path / "clip.mp4"
    w = cv2.VideoWriter(str(vid), cv2.VideoWriter_fourcc(*"mp4v"), 10, (64, 48))
    for i in range(30):                                            # 3 秒
        w.write(np.full((48, 64, 3), i * 8, dtype=np.uint8))
    w.release()
    vkey = f"uploads/media/test_{uuid.uuid4().hex[:6]}.mp4"
    objects.put(bucket="raw", key=vkey, data=vid.open("rb"), length=vid.stat().st_size)
    tid = c.post(f"/api/media/collections/{cid}/import", headers=_hdr("alice"),
                 json={"source_uri": f"s3://raw/{vkey}", "kind": "video", "strategy": "interval",
                       "interval_s": 0.5}).json()["task_id"]
    assert jobs.run_import(tid) == 0
    r = state["store"].get(tid).result["import"]
    assert 5 <= r["frames"] <= 7, r
    items = c.get(f"/api/media/collections/{cid}/items").json()
    assert sum(1 for i in items if (i["original_name"] or "").startswith(vkey.rsplit("/", 1)[-1])) == r["added"]


def test_video_import_diverse_selection(env, tmp_path):
    """strategy=diverse（預設）：去重＋有變動優先＋分群；每張記 select_info，結果有挑圖報告。"""
    c, state = env
    from edgevision.media import jobs

    cid = c.post("/api/media/collections", json={"name": f"it-sel-{uuid.uuid4().hex[:6]}"},
                 headers=_hdr("alice")).json()["id"]
    vid = tmp_path / "ipcam.mp4"
    w = cv2.VideoWriter(str(vid), cv2.VideoWriter_fourcc(*"mp4v"), 10, (320, 180))
    for i in range(600):                                           # 60 秒：前 30 秒空場景、後 30 秒有東西在動
        img = np.full((180, 320, 3), 100, np.uint8)
        cv2.putText(img, f"10:00:{i // 10:02d}", (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        if i >= 300:
            x = 40 + (i - 300) // 2
            cv2.rectangle(img, (x, 60), (x + 30, 150), (30, 200, 30), -1)
        w.write(img)
    w.release()
    vkey = f"uploads/media/test_{uuid.uuid4().hex[:6]}.mp4"
    state["objects"].put(bucket="raw", key=vkey, data=vid.open("rb"), length=vid.stat().st_size)
    bad = c.post(f"/api/media/collections/{cid}/import", headers=_hdr("alice"),
                 json={"source_uri": f"s3://raw/{vkey}", "kind": "video", "select": {"roi": [0.5, 0, 0.2, 1]}})
    assert bad.status_code == 400                                  # roi 座標錯
    tid = c.post(f"/api/media/collections/{cid}/import", headers=_hdr("alice"),
                 json={"source_uri": f"s3://raw/{vkey}", "kind": "video", "select": {"target": 6}}).json()["task_id"]
    assert jobs.run_import(tid) == 0
    r = state["store"].get(tid).result["import"]
    sel = r["select"]
    assert r["strategy"] == "diverse" and sel["sampled"] >= 58 and sel["after_dedup"] < sel["sampled"]
    assert 1 <= r["added"] <= 6 and sel["selected"] == r["added"] and sel["selected_moving"] >= 1
    items = c.get(f"/api/media/collections/{cid}/items").json()
    assert len(items) == r["added"]
    one = c.get(f"/api/media/items/{items[0]['id']}").json()
    assert one["select_info"]["strategy"] == "diverse" and one["select_info"]["group_size"] >= 1
