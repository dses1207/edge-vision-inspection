"""告警回流整條，對真的 PostgreSQL + MinIO 跑：

    detections 的 NG 列 → 待判清單 → 判 → 回流（建批次、任務）→ 匯入任務（NAS 找那一幀、框當標注）

NAS 錄影與證據影片是合成的（同一個會動的畫面；證據影片每幀畫當下的框）。
沒設 MEDIA_TEST_DSN 就略過。資料庫要先跑過 deploy/postgres/init 全部。
"""
import json
import os
import uuid
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import numpy as np
import pytest

pytestmark = pytest.mark.skipif(not os.environ.get("MEDIA_TEST_DSN"), reason="需要 MEDIA_TEST_DSN")

FPS, W, H = 10, 640, 360
CLASSES = ["board", "sheet", "glove", "tap", "mat"]
PAD = {"x1": 400, "y1": 250, "x2": 500, "y2": 330}               # 不會動的框


def _scene(t: float) -> np.ndarray:
    rng = np.random.default_rng(1)
    img = cv2.GaussianBlur(rng.integers(0, 255, (H, W, 3)).astype(np.uint8), (0, 0), 4)
    x = round(20 + 15 * t) % (W - 100)
    cv2.rectangle(img, (x, 120), (x + 80, 200), (240, 240, 240), -1)
    return img


def _box(t: float) -> dict:
    x = round(20 + 15 * t) % (W - 100)
    return {"x1": x, "y1": 120, "x2": x + 80, "y2": 200}


def _write(path: Path, t0: float, n: int, *, overlay: bool = False) -> None:
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    for k in range(n):
        t = round(t0 + k / FPS, 3)
        img = _scene(t)
        if overlay:                                              # 線上每幀都畫當下所有的框
            for b in (_box(t), PAD):
                cv2.rectangle(img, (b["x1"], b["y1"]), (b["x2"], b["y2"]), (0, 255, 255), 2)
        vw.write(img)
    vw.release()


@pytest.fixture()
def env(tmp_path, monkeypatch):
    from edgevision.media.library import MediaLibrary
    from edgevision.storage.db import Database
    from edgevision.storage.objects import ObjectStore

    os.environ["DATABASE_URL"] = os.environ["MEDIA_TEST_DSN"]
    db, objects = Database(), ObjectStore()
    for b in ("raw", "inference-output"):
        try:
            objects._client.make_bucket(b)
        except Exception:                                         # noqa: BLE001
            pass
    monkeypatch.setenv("MEDIA_NAS_ROOT", str(tmp_path / "nas"))
    return db, objects, MediaLibrary(db, objects), tmp_path


def _ng(db, cam, frame_id, ts, boxes, *, model="m_alert", station="st_alert", clip_uri=None):
    with db.cursor() as cur:
        ids = []
        for cls, b in boxes:
            cur.execute(
                """INSERT INTO detections (ts, camera_id, frame_id, model_name, model_version, class_name, confidence,
                       bbox, image_meta, verdict_status, verdict_reason, clip_uri, station)
                   VALUES (%s,%s,%s,%s,'2',%s,0.9,%s,%s,'ng','未完成確認',%s,%s) RETURNING id""",
                (ts, cam, frame_id, model, cls, json.dumps(b), json.dumps({"width": W, "height": H}), clip_uri,
                 station))
            ids.append(cur.fetchone()[0])
    return min(ids)


def test_queue_review_feedback_and_import(env):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from edgevision.media import alerts as A
    from edgevision.media.api import build_router
    from edgevision.media.campaign import CampaignEngine
    from edgevision.media.jobs import _Progress
    from edgevision.training.tracker import TaskStore

    db, objects, lib, tmp = env
    cam = f"t{uuid.uuid4().hex[:6]}"
    model = f"m_{uuid.uuid4().hex[:6]}"
    from edgevision.media.frames import LOCAL_TZ
    day = datetime.now(LOCAL_TZ).replace(hour=9, minute=0, second=0, microsecond=0) - timedelta(days=1)

    # NAS：09:00:00 起 60 秒；告警在 09:00:30（畫面第 30 秒）
    nas_dir = tmp / "nas" / cam
    nas_dir.mkdir(parents=True)
    _write(nas_dir / f"{day:%Y%m%d_%H%M%S}.mp4", 0.0, 60 * FPS)
    ng_t = 30.0
    clip = tmp / "clip.mp4"
    _write(clip, ng_t - 8.0, 13 * FPS, overlay=True)
    key = f"{cam}/{uuid.uuid4().hex}.mp4"
    with clip.open("rb") as fh:
        objects.put(bucket="inference-output", key=key, data=fh, length=clip.stat().st_size, content_type="video/mp4")
    ts = day + timedelta(seconds=ng_t + 0.6)                          # 資料庫時間比畫面晚一點（寫入延遲）
    e1 = _ng(db, cam, "100", ts, [("glove", _box(ng_t)), ("mat", PAD)],
             model=model, clip_uri=f"s3://inference-output/{key}")
    e2 = _ng(db, cam, "200", day + timedelta(seconds=50), [("glove", _box(50))], model=model)   # 沒證據影片
    e3 = _ng(db, cam, "300", day + timedelta(hours=3), [("glove", _box(1))], model=model)       # NAS 沒錄到

    # 專案：這個模型的進行中標注專案 → 回流進它的圖片集
    col = lib.create_collection(name=f"af-{cam}", actor="t", class_names=CLASSES)
    eng = CampaignEngine(lib, None)
    eng.create(name=f"af-{cam}", collection_id=col["id"], model_name=model, base_model="b", actor="t",
               initial_model={"name": "seed", "version": 1})

    class Orch:
        def __init__(self):
            self.cpu = []

        def submit_media_import(self, tid):
            self.cpu.append(tid)

    state = {"db": db, "objects": objects, "store": TaskStore(db), "orchestrator": Orch()}
    app = FastAPI()
    app.include_router(build_router(state, lambda x_actor, *_: (x_actor or "corp\\alice", "human", "api")))
    c = TestClient(app)

    q = c.get("/api/media/alerts/queue", params={"day": day.date().isoformat(), "camera_id": cam}).json()
    ev = {e["event_id"]: e for e in q["events"]}
    assert set(ev) == {e1, e2, e3} and q["stations"][0]["total"] == 3
    assert ev[e1]["clip_url"] and {b["class"] for b in ev[e1]["boxes"]} == {"glove", "mat"}
    assert ev[e1]["image_size"] == {"width": W, "height": H} and ev[e1]["feedback"] is None
    assert len(c.get("/api/media/alerts/queue", params={"day": day.date().isoformat(), "camera_id": cam,
                                                         "per_station": 1}).json()["events"]) == 1

    for e in (e1, e2, e3):
        r = c.post(f"/api/media/alerts/{e}/review", json={"result": "true_positive"})
        assert r.status_code == 200 and r.json()["rows"] >= 1, r.text
    assert c.post("/api/media/alerts/999999999/review", json={"result": "false_positive"}).status_code == 404

    run = c.post("/api/media/alerts/feedback/run", json={"lookback_days": 3}).json()
    g = next(x for x in run["groups"] if x["model_name"] == model)
    assert g["events"] == 3 and g["collection_id"] == col["id"] and g["campaign_name"] == f"af-{cam}"
    tid = g["task_id"]
    assert tid in state["orchestrator"].cpu
    assert c.post(f"/api/media/alerts/{e1}/review", json={"result": None}).status_code == 409   # 已排進回流
    again = c.post("/api/media/alerts/feedback/run", json={"lookback_days": 3}).json()
    assert all(x["model_name"] != model for x in again["groups"])                           # 不會重複

    # 匯入任務（就是 K8s Job 跑的那段）
    t = state["store"].get(tid)
    res = A.import_alerts(lib, col["id"], t.params, actor="edgevision", task_id=tid,
                          progress=_Progress(state["store"], tid), batch_id=t.params["batch_id"])
    assert res["added"] == 2 and res["skipped"] == 1, res
    fb = {f["event_id"]: f for f in A.history(db, task_id=tid)}
    assert fb[e1]["status"] == "added" and fb[e1]["item_status"] == "labeled" and fb[e1]["align"]["ok"]
    assert abs(fb[e1]["align"]["nas_t"] - (day.timestamp() + ng_t)) < 0.06                 # 對到告警那一幀
    assert fb[e2]["status"] == "added" and fb[e2]["item_status"] == "auto"                  # 沒證據影片 → 預標
    assert fb[e3]["status"] == "skipped" and "NAS 沒有" in fb[e3]["reason"]

    it = lib.item(fb[e1]["item_id"], with_url=False)
    assert it["status"] == "labeled" and it["batch_id"] == t.params["batch_id"]
    gl = next(s for s in it["shapes"] if s["class"] == "glove")
    b = _box(ng_t)
    assert gl["points"] == [[b["x1"] / W, b["y1"] / H], [b["x2"] / W, b["y2"] / H]]
    assert gl["source"]["model"] == model and it["select_info"]["alert"]["event_id"] == e1
    assert lib.batch(t.params["batch_id"])["kind"] == "alert"
    q2 = c.get("/api/media/alerts/queue", params={"day": day.date().isoformat(), "camera_id": cam}).json()
    assert {e["event_id"]: e["feedback"]["status"] for e in q2["events"]} == {e1: "added", e2: "added", e3: "skipped"}
