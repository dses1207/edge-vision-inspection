"""錄影 / 影片庫 / 錄影名單 / 圖片集管理 / 打包下載，經 API 對真的 PostgreSQL + MinIO 跑。

NAS 是 tmp 目錄（合成的 mp4）；背景任務（剪片段、打包）直接呼叫任務函式（就是 K8s Job 跑的那段）。
沒設 MEDIA_TEST_DSN 就略過。資料庫要先跑過 deploy/postgres/init 全部。
"""
import io
import json
import os
import uuid
import zipfile
from datetime import datetime, timedelta

import cv2
import numpy as np
import pytest

pytestmark = pytest.mark.skipif(not os.environ.get("MEDIA_TEST_DSN"), reason="需要 MEDIA_TEST_DSN")


def _mp4(path, seconds: int, fps: int = 5):
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (160, 90))
    for k in range(seconds * fps):
        vw.write(np.full((90, 160, 3), (k * 3) % 255, np.uint8))
    vw.release()


def _jpeg(seed: int) -> bytes:
    img = np.random.default_rng(seed).integers(0, 255, (100, 200, 3), dtype=np.uint8)
    return cv2.imencode(".jpg", img)[1].tobytes()


class Orch:
    def __init__(self):
        self.imports, self.exports = [], []

    def submit_media_import(self, tid):
        self.imports.append(tid)

    def submit_media_export(self, tid):
        self.exports.append(tid)


@pytest.fixture()
def env(tmp_path, monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from edgevision.media.api import build_router
    from edgevision.media.library import MediaLibrary
    from edgevision.storage.db import Database
    from edgevision.storage.objects import ObjectStore
    from edgevision.training.tracker import TaskStore

    os.environ["DATABASE_URL"] = os.environ["MEDIA_TEST_DSN"]
    monkeypatch.setenv("MEDIA_NAS_ROOT", str(tmp_path / "nas"))
    db, objects = Database(), ObjectStore()
    try:
        objects._client.make_bucket("raw")
    except Exception:  # noqa: BLE001
        pass
    state = {"db": db, "objects": objects, "store": TaskStore(db), "orchestrator": Orch()}
    app = FastAPI()
    app.include_router(build_router(state, lambda x_actor, *_: (x_actor or "corp\\alice", "human", "api")))
    return TestClient(app), state, MediaLibrary(db, objects), tmp_path


def test_recordings_roster_clip_and_video_library(env):
    from edgevision.media.frames import LOCAL_TZ
    from edgevision.media.jobs import run_clip

    c, state, lib, tmp = env
    cam = f"rc{uuid.uuid4().hex[:6]}"
    line = f"ln-{cam}"
    with state["db"].cursor() as cur:
        cur.execute("INSERT INTO lines (id, name, station) VALUES (%s,%s,'s')", (line, line))
        cur.execute("INSERT INTO cameras (id, line_id, ip) VALUES (%s,%s,'192.0.2.84')", (cam, line))

    # 錄影名單
    r = c.put(f"/api/media/recordings/roster/{cam}", json={"enabled": True, "note": "測試"})
    assert r.status_code == 200 and r.json()["recording"] and r.json()["note"] == "測試", r.text
    assert c.put("/api/media/recordings/roster/no-such-cam", json={"enabled": True}).status_code == 404
    from edgevision.media.recordings import roster_cameras
    assert cam in roster_cameras(state["db"])
    c.put(f"/api/media/recordings/roster/{cam}", json={"enabled": False})
    assert cam not in roster_cameras(state["db"])

    # NAS：兩段，中間沒缺
    d = tmp / "nas" / cam
    d.mkdir(parents=True)
    day = datetime(2026, 10, 18, 9, 0, tzinfo=LOCAL_TZ)
    for k in range(2):
        _mp4(d / f"{day + timedelta(seconds=20 * k):%Y%m%d_%H%M%S}.mp4", 20)
    old = datetime.now().timestamp() - 3600
    for p in d.iterdir():
        os.utime(p, (old, old))
    rec = c.get("/api/media/recordings").json()
    me = next(x for x in rec["cameras"] if x["camera_id"] == cam)
    assert me["files"] == 2 and rec["nas"]["total_gb"] > 0
    assert c.get(f"/api/media/recordings/{cam}/days").json()[0]["segments"] == 2
    dd = c.get(f"/api/media/recordings/{cam}", params={"day": "2026-10-18"}).json()
    assert [s["status"] for s in dd["segments"]] == ["ok", "ok"] and dd["gaps"][0]["end"].startswith("2026-10-18T09:00:00")
    url = dd["segments"][0]["stream_url"]
    part = c.get(url, headers={"Range": "bytes=0-99"})
    assert part.status_code == 206 and len(part.content) == 100
    assert "attachment" in c.get(dd["segments"][0]["download_url"]).headers["content-disposition"]
    assert c.get(f"/api/media/recordings/{cam}/files/..%2F..%2Fetc").status_code in (400, 404)
    assert c.get(f"/api/media/recordings/{cam}", params={"day": "2026-10-18"}).json() == dd   # 第二次走快取

    # 剪片段 → 影片庫
    body = {"start": "2026-10-18T09:00:10", "end": "2026-10-18T09:00:30", "name": "測試片段", "tags": ["golden"]}
    assert c.post(f"/api/media/recordings/{cam}/clips", json={**body, "end": "2026-10-18T09:00:05"}).status_code == 400
    r = c.post(f"/api/media/recordings/{cam}/clips", json=body)
    assert r.status_code == 202, r.text
    tid = r.json()["task_id"]
    assert tid in state["orchestrator"].imports
    assert run_clip(lib, state["store"], tid) == 0, state["store"].get(tid).status_message
    vid = state["store"].get(tid).result["clip"]["video_id"]
    v = c.get(f"/api/media/videos/{vid}").json()
    assert v["kind"] == "recording_clip" and v["name"] == "測試片段" and v["tags"] == ["golden"]
    assert v["expires_at"] is None and 18 <= v["duration_s"] <= 22 and v["used_by"] == []
    assert any(x["id"] == vid for x in c.get("/api/media/videos", params={"tag": "golden"}).json())
    p = c.patch(f"/api/media/videos/{vid}", json={"name": "改名", "tags": ["b", "a", "a"]}).json()
    assert p["name"] == "改名" and p["tags"] == ["a", "b"]
    s = c.get(f"/api/media/videos/{vid}/stream", follow_redirects=False)
    assert s.status_code == 307 and "X-Amz-Signature" in s.headers["location"]
    from edgevision.media import video_thumbs as VT
    assert v["thumbnail_url"] == f"/api/media/videos/{vid}/thumbnail"
    th = c.get(v["thumbnail_url"], params={"px": 64})                     # 影片在 MinIO：PyAV 走預簽網址 Range 讀
    assert th.status_code == 200 and th.headers["content-type"] == "image/webp" and "immutable" in th.headers["cache-control"]
    img = cv2.imdecode(np.frombuffer(th.content, np.uint8), cv2.IMREAD_COLOR)
    assert max(img.shape[:2]) == 64
    assert c.get(v["thumbnail_url"], params={"px": 64}).content == th.content      # 第二次讀存好的
    assert lib.objects.exists_prefix(bucket=VT.BUCKET, prefix=VT.thumb_prefix(vid))
    assert c.get(v["thumbnail_url"], params={"px": 9999}).status_code == 422
    dl = c.delete(f"/api/media/videos/{vid}").json()
    assert dl == {"id": vid, "items_detached": 0, "file_removed": True}
    assert not lib.objects.exists_prefix(bucket=VT.BUCKET, prefix=VT.thumb_prefix(vid))   # 縮圖跟著刪

    # 上傳影片只進影片庫（不抽幀、不需要圖片集）
    from edgevision.media.jobs import run_video_upload
    up = c.post("/api/media/upload-url", params={"filename": "上傳測試.mp4"}).json()
    src = tmp / "up.mp4"
    _mp4(src, 3)
    bucket, key = up["source_uri"].removeprefix("s3://").split("/", 1)
    with src.open("rb") as fh:
        lib.objects.put(bucket=bucket, key=key, data=fh, length=src.stat().st_size, content_type="video/mp4")
    assert c.post("/api/media/videos", json={"source_uri": "s3://raw/elsewhere/x.mp4"}).status_code == 400
    assert c.post("/api/media/videos", json={"source_uri": up["source_uri"], "camera_id": "no-such"}).status_code == 404
    r = c.post("/api/media/videos", json={"source_uri": up["source_uri"], "tags": ["golden", " "], "camera_id": cam})
    assert r.status_code == 202 and r.json()["name"] == "上傳測試.mp4", r.text
    tid = r.json()["task_id"]
    assert tid in state["orchestrator"].imports
    assert run_video_upload(lib, state["store"], tid) == 0, state["store"].get(tid).status_message
    res = state["store"].get(tid).result["video"]
    v = c.get(f"/api/media/videos/{res['video_id']}").json()
    assert v["kind"] == "upload_video" and v["name"] == "上傳測試.mp4" and v["tags"] == ["golden"]
    assert v["camera_id"] == cam and v["used_by"] == [] and 2 <= v["duration_s"] <= 4 and not res["duplicate"]
    assert not lib.objects.exists_prefix(bucket=bucket, prefix=key)                     # 暫存上傳檔已刪
    assert c.get(v["thumbnail_url"]).status_code == 200
    with state["db"].cursor() as cur:
        cur.execute("SELECT count(*) FROM media_items WHERE source_id = %s", (v["id"],))
        assert cur.fetchone()[0] == 0                                                   # 沒有抽幀
    # 同一支再傳一次 → 不重複新增
    up2 = c.post("/api/media/upload-url", params={"filename": "again.mp4"}).json()
    b2, k2 = up2["source_uri"].removeprefix("s3://").split("/", 1)
    with src.open("rb") as fh:
        lib.objects.put(bucket=b2, key=k2, data=fh, length=src.stat().st_size, content_type="video/mp4")
    t2 = c.post("/api/media/videos", json={"source_uri": up2["source_uri"]}).json()["task_id"]
    assert run_video_upload(lib, state["store"], t2) == 0
    assert state["store"].get(t2).result["video"] == {**res, "duplicate": True, "size_bytes": res["size_bytes"]}
    # 不是影片 → 任務失敗
    up3 = c.post("/api/media/upload-url", params={"filename": "fake.mp4"}).json()
    b3, k3 = up3["source_uri"].removeprefix("s3://").split("/", 1)
    lib.objects.put(bucket=b3, key=k3, data=io.BytesIO(b"not a video"), length=11, content_type="video/mp4")
    t3 = c.post("/api/media/videos", json={"source_uri": up3["source_uri"]}).json()["task_id"]
    assert run_video_upload(lib, state["store"], t3) != 0
    c.delete(f"/api/media/videos/{v['id']}")
    assert c.get(f"/api/media/videos/{vid}").status_code == 404


def test_collection_ops_and_download(env):
    from edgevision.media.campaign import CampaignEngine
    from edgevision.media.jobs import run_download

    c, state, lib, _ = env
    name = f"ops-{uuid.uuid4().hex[:6]}"
    cid = c.post("/api/media/collections", json={"name": name, "class_names": ["mat", "glove", "sheet"]}).json()["id"]
    files = [("files", (f"{name}-{i}.jpg", _jpeg(hash((name, i)) % 10 ** 6), "image/jpeg")) for i in range(4)]
    up = c.post(f"/api/media/collections/{cid}/images", files=files).json()
    ids = [x["item_id"] for x in up["results"]]
    shapes = [{"type": "rect", "class": "mat", "points": [[0.1, 0.1], [0.5, 0.5]]},
              {"type": "polygon", "class": "glove", "points": [[0.6, 0.6], [0.9, 0.6], [0.9, 0.9]]}]
    for iid in ids[:3]:
        lib._write(iid, shapes, "labeled", actor="t", action="submit")
    lib._write(ids[3], [shapes[1]], "auto", actor="t", action="autolabel")

    # 類別改名 / 刪除
    assert c.patch(f"/api/media/collections/{cid}/classes", json={"rename": {"mat": "glove"}}).status_code == 400
    r = c.patch(f"/api/media/collections/{cid}/classes", json={"rename": {"mat": "mat"}, "delete": ["glove"]})
    assert r.status_code == 200, r.text
    assert r.json()["class_names"] == ["mat", "sheet"] and r.json()["items_changed"] == 4
    it = lib.item(ids[0], with_url=False)
    assert [s["class"] for s in it["shapes"]] == ["mat"] and it["status"] == "labeled"
    assert lib.item(ids[3], with_url=False)["status"] == "unlabeled"           # 預標的框拿光 → 回未標
    assert lib.history(ids[0])[0]["action"] == "class_change"

    # 標籤 / 備註 / 搜尋
    m = c.patch(f"/api/media/items/{ids[1]}/meta", json={"tags": ["hard", "hard", "night"], "note": "反光"}).json()
    assert m == {"id": ids[1], "tags": ["hard", "night"], "note": "反光"}
    assert [x["id"] for x in c.get(f"/api/media/collections/{cid}/items", params={"tag": "hard"}).json()] == [ids[1]]
    found = c.get("/api/media/items/search", params={"q": f"{name}-2"}).json()
    assert [x["id"] for x in found] == [ids[2]] and found[0]["collection_name"] == name
    sha = lib.item(ids[2], with_url=False)["blob_sha256"]
    assert c.get("/api/media/items/search", params={"sha256": sha[:10]}).json()[0]["id"] == ids[2]
    assert c.get("/api/media/items/search", params={"sha256": "zz"}).status_code == 400
    assert c.get("/api/media/items/search").status_code == 400

    # 打包四種格式
    for fmt in ("yolo", "yolo-seg", "coco", "voc"):
        r = c.post(f"/api/media/collections/{cid}/download", json={"format": fmt, "val_pct": 0})
        assert r.status_code == 202, r.text
        tid = r.json()["task_id"]
        assert run_download(lib, state["store"], tid) == 0, state["store"].get(tid).status_message
        g = c.get(f"/api/media/downloads/{tid}").json()
        assert g["status"] == "completed" and g["items"] == 3 and g["url"]          # 預設只含人確認過的
        b, k = state["objects"].parse_uri(state["store"].get(tid).result["download"]["uri"])
        z = zipfile.ZipFile(io.BytesIO(state["objects"].get(bucket=b, key=k)))
        names = z.namelist()
        assert "items.csv" in names
        if fmt == "yolo":
            lbl = next(n for n in names if n.startswith("labels/train/"))
            assert z.read(lbl).decode().split()[0] == "0" and "data.yaml" in names
        elif fmt == "yolo-seg":
            lbl = next(n for n in names if n.startswith("labels/train/"))
            assert len(z.read(lbl).decode().split()) == 9                        # 框 → 四點多邊形
        elif fmt == "coco":
            j = json.loads(z.read("annotations/instances_train.json"))
            assert len(j["images"]) == 3 and j["categories"][0]["name"] == "mat"
        else:
            assert sum(n.startswith("Annotations/") for n in names) == 3
    assert c.post(f"/api/media/collections/{cid}/download", json={"format": "pdf"}).status_code == 422

    # 刪圖片集：專案用著 → 409；名稱不對 → 400；之後才刪得掉
    eng = CampaignEngine(lib, None)
    p = eng.create(name=name, collection_id=cid, model_name="m", base_model="b", actor="t",
                   initial_model={"name": "seed", "version": 1})
    assert c.delete(f"/api/media/collections/{cid}", params={"confirm": "x"}).status_code == 400
    r = c.delete(f"/api/media/collections/{cid}", params={"confirm": name})
    assert r.status_code == 409 and "標注專案" in r.text
    with state["db"].cursor() as cur:
        cur.execute("DELETE FROM media_campaign_events WHERE campaign_id = %s", (p["id"],))
        cur.execute("DELETE FROM media_campaign_iterations WHERE campaign_id = %s", (p["id"],))
        cur.execute("DELETE FROM media_campaigns WHERE id = %s", (p["id"],))
    r = c.delete(f"/api/media/collections/{cid}", params={"confirm": name})
    assert r.status_code == 200 and r.json()["items"] == 4, r.text
    assert c.get(f"/api/media/collections/{cid}").status_code == 404
