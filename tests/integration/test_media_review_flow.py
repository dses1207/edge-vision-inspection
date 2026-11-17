"""審核整條流程（輪次、抽查、下一張、送出差異、關閉），對真的 PostgreSQL + MinIO 跑（沒設 MEDIA_TEST_DSN 就略過）。

    MEDIA_TEST_DSN=postgresql://…/edgevision_review_dev MINIO_ENDPOINT=… MINIO_ROOT_USER=… MINIO_ROOT_PASSWORD=… \
        pytest tests/integration/test_media_review_flow.py

資料庫要先跑過 deploy/postgres/init/02-schema.sql、12-media-library.sql、14-media-select.sql、16-media-review.sql。
"""
import os
import uuid

import cv2
import numpy as np
import pytest

pytestmark = pytest.mark.skipif(not os.environ.get("MEDIA_TEST_DSN"), reason="需要 MEDIA_TEST_DSN")

CLASSES = ["board", "sheet", "glove", "tap", "mat"]


def _jpeg(seed: int, size=(320, 180)) -> bytes:
    rng = np.random.default_rng(seed)
    img = rng.integers(0, 255, (size[1], size[0], 3), dtype=np.uint8)
    return cv2.imencode(".jpg", img)[1].tobytes()


def _rect(cls, x1, y1, x2, y2, conf):
    return {"type": "rect", "class": cls, "points": [[x1, y1], [x2, y2]],
            "source": {"kind": "model", "engine": "registry", "model": "seed", "version": 1, "conf": conf}}


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
    state = {"db": db, "objects": objects, "store": TaskStore(db), "orchestrator": None}
    actors = {"alice": ("alice", "human", "api"), "bob": ("bob", "human", "api")}

    def actor_fn(x_actor, *_):
        return actors[x_actor or "alice"]

    app = FastAPI()
    app.include_router(build_router(state, actor_fn))
    return TestClient(app), state


def H(who="alice"):
    return {"X-Actor": who}


@pytest.fixture()
def seeded(env):
    """12 張圖：0–2 有 press（rare）、3–5 低信心 glove、6–8 只有 pad、9–11 什麼都沒有。"""
    c, state = env
    name = f"rv-{uuid.uuid4().hex[:6]}"
    cid = c.post("/api/media/collections", json={"name": name, "class_names": CLASSES}, headers=H()).json()["id"]
    files = [("files", (f"{i}.jpg", _jpeg(100 + i + hash(name) % 1000), "image/jpeg")) for i in range(12)]
    res = c.post(f"/api/media/collections/{cid}/images", headers=H(), files=files).json()
    ids = [x["item_id"] for x in res["results"]]
    lib = state["media"]
    pad = _rect("mat", .6, .2, .8, .4, .97)
    for i, iid in enumerate(ids):
        if i < 3:
            shapes = [pad, _rect("tap", .65, .25, .7, .3, .9)]
        elif i < 6:
            shapes = [pad, _rect("glove", .1, .1, .2, .2, .25)]
        elif i < 9:
            shapes = [pad]
        else:
            shapes = []
        assert lib.write_auto(iid, shapes, actor="bot", task_id="t-seed")
    return c, state, cid, ids


def test_review_config_defaults_and_override(seeded):
    c, _, cid, _ = seeded
    base = c.get(f"/api/media/collections/{cid}/review-config").json()
    assert base["override"] == {} and [x["name"] for x in base["classes"]] == CLASSES

    bad = c.patch(f"/api/media/collections/{cid}/review-config", json={"round": {"size": 10 ** 7}}, headers=H())
    assert bad.status_code == 400
    assert c.get(f"/api/media/collections/{cid}/review-config").json()["override"] == {}   # 驗證不過整筆不寫

    ok = c.patch(f"/api/media/collections/{cid}/review-config", headers=H(),
                 json={"classes": {"tap": {"rare": True, "bulk_accept": False}}, "round": {"size_min": 1}}).json()
    assert ok["override"] == {"classes": {"tap": {"rare": True, "bulk_accept": False}}, "round": {"size_min": 1}}
    press = next(x for x in ok["classes"] if x["name"] == "tap")
    assert press["rare"] and not press["bulk_accept"]
    reset = c.patch(f"/api/media/collections/{cid}/review-config", json={"round": {"size_min": None}},
                    headers=H()).json()
    assert reset["override"] == {"classes": {"tap": {"rare": True, "bulk_accept": False}}}


def test_round_flow(seeded):
    c, state, cid, ids = seeded
    c.patch(f"/api/media/collections/{cid}/review-config", headers=H(),
            json={"classes": {"tap": {"rare": True}}, "round": {"size_min": 1}})

    rd = c.post(f"/api/media/collections/{cid}/rounds", json={"size": 10, "blind_ratio": 0.2}, headers=H())
    assert rd.status_code == 201, rd.text
    rd = rd.json()
    rid = rd["id"]
    assert rd["n_items"] == 10 and rd["n_blind"] == 2 and rd["status"] == "open"
    assert rd["model"] == {"name": "seed", "version": 1}
    assert rd["config"]["round"]["size"] == 10                       # 建立當下的設定存進輪次
    assert rd["by_reason"].get("rare", 0) >= 1

    # 縮圖牆：有縮圖網址；抽查圖不給框
    lst = c.get(f"/api/media/rounds/{rid}/items").json()
    assert lst["total"] == 10
    assert all(x["thumb_url"] and x["thumb_url"].startswith("http") for x in lst["items"])
    blinds = [x for x in lst["items"] if x["blind"]]
    assert len(blinds) == 2 and all(x["shapes"] == [] for x in blinds)
    assert all(not x["blind"] or x["status"] == "unlabeled" for x in lst["items"])
    # 第一名一定是 press（rare 權重最高）——除非它被抽成抽查圖
    first = lst["items"][0]
    assert "rare" in first["reasons"]

    # 單張 GET：抽查圖也拿不到預標
    b = c.get(f"/api/media/items/{blinds[0]['id']}").json()
    assert b["review"]["blind"] is True and b["shapes"] == []

    # 下一張：依順序、上鎖；bob 拿到不同張
    n1 = c.get(f"/api/media/rounds/{rid}/next?peek=2", headers=H("alice")).json()
    n2 = c.get(f"/api/media/rounds/{rid}/next", headers=H("bob")).json()
    it1, it2 = n1["item"], n2["item"]
    assert it1["id"] == first["id"] and it1["locked_by"] == "alice"
    assert it2["id"] != it1["id"] and n1["remaining"] == 10 and len(n1["peek"]) == 2
    assert all(s.get("id") for s in it1["shapes"])                   # 建輪次時補了框 id

    # 送出：原樣一個、拖一個、加一個
    shapes = [dict(s) for s in it1["shapes"]]
    if not it1["review"]["blind"]:
        shapes[0] = {**shapes[0], "points": [[shapes[0]["points"][0][0], shapes[0]["points"][0][1]], [0.85, 0.45]]}
    shapes.append({"type": "rect", "class": "glove", "points": [[0.3, 0.3], [0.4, 0.4]], "id": "new-1"})
    rv = c.post(f"/api/media/items/{it1['id']}/review", headers=H("alice"),
                json={"revision": it1["revision"], "action": "submit", "shapes": shapes, "seconds_spent": 7.5,
                      "clean_view_seen": True})
    assert rv.status_code == 200, rv.text
    rv = rv.json()
    assert rv["status"] == "labeled" and rv["state"] == "done"
    assert rv["diff"]["added"] == 1
    if not it1["review"]["blind"]:
        assert rv["diff"]["modified"] == 1 and rv["diff"]["accepted"] == len(it1["shapes"]) - 1

    # bob 沒寫備註的「有疑問」→ 400；寫了 → doubt，圖不變、鎖放掉
    assert c.post(f"/api/media/items/{it2['id']}/review", headers=H("bob"),
                  json={"revision": it2["revision"], "action": "doubt"}).status_code == 400
    d = c.post(f"/api/media/items/{it2['id']}/review", headers=H("bob"),
               json={"revision": it2["revision"], "action": "doubt", "note": "看不出有沒有完成動作"}).json()
    assert d["state"] == "doubt" and d["status"] == it2["status"]
    assert c.get(f"/api/media/items/{it2['id']}").json()["locked_by"] is None

    # 舊 revision 送出 → 409
    assert c.post(f"/api/media/items/{it1['id']}/review", headers=H("alice"),
                  json={"revision": it1["revision"], "action": "background"}).status_code == 409

    st = c.get(f"/api/media/rounds/{rid}").json()
    assert st["progress"] == {"pending": 8, "labeled": 1, "background": 0, "doubt": 1}
    assert st["reviewers"][0]["actor"] == "alice" and st["reviewers"][0]["avg_seconds"] == 7.5
    assert c.get(f"/api/media/rounds/{rid}/items?state=doubt").json()["total"] == 1

    # 第二輪：不會挑到第一輪（進行中）裡的圖
    r2 = c.post(f"/api/media/collections/{cid}/rounds", json={"size": 5, "blind_ratio": 0}, headers=H()).json()
    assert r2["n_items"] == 2                                       # 12 − 10
    # 關閉後不能再審、也不能再拿下一張
    assert c.post(f"/api/media/rounds/{rid}/close", headers=H()).json()["status"] == "closed"
    assert c.post(f"/api/media/rounds/{rid}/close", headers=H()).status_code == 409
    assert c.get(f"/api/media/rounds/{rid}/next", headers=H()).status_code == 409
    nxt = c.get(f"/api/media/items/{ids[0]}").json()
    assert nxt["review"] is not None
    # 第一輪關了，沒審完的回到可選池：第三輪挑得到
    c.post(f"/api/media/rounds/{r2['id']}/close", headers=H())
    r3 = c.post(f"/api/media/collections/{cid}/rounds", json={"size": 20, "blind_ratio": 0}, headers=H()).json()
    assert r3["n_items"] == 11                                      # 只有 it1 已 labeled 不能再選
    hist = c.get(f"/api/media/items/{blinds[0]['id']}/history").json()
    assert any(h["action"] == "blind" for h in hist)
