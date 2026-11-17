"""標注專案 v0.3 整個循環，對真的 PostgreSQL + MinIO 跑：

    建專案（閒置）→ 只預標（初始模型）→ 人在圖片集審 → 跑下一輪：匯出（每批一個快照）→ 混合 → 訓練 → 用新模型預標
    → 閒置 → 用圖片庫加一批 → 只預標（最新一代）→ 審 → 跑下一輪（只重匯有改的批次）…

預標 / 匯出 / 混合 / 訓練換成假的動作（測的是推進邏輯，不是 GPU / LakeFS）；圖片、批次、審核都是真的。
沒設 MEDIA_TEST_DSN 就略過。資料庫要先跑過 deploy/postgres/init 全部。
"""
import itertools
import os
import uuid

import cv2
import numpy as np
import pytest

from edgevision.media.library import Conflict

pytestmark = pytest.mark.skipif(not os.environ.get("MEDIA_TEST_DSN"), reason="需要 MEDIA_TEST_DSN")

CLASSES = ["glove", "tap", "mat"]
_seq = itertools.count()
BOX = [{"type": "rect", "class": "mat", "points": [[.1, .1], [.5, .5]]}]


def _jpeg(seed: int) -> bytes:
    img = np.random.default_rng(seed).integers(0, 255, (90, 160, 3), dtype=np.uint8)
    return cv2.imencode(".jpg", img)[1].tobytes()


class FakeActions:
    """任務一建就是 working；測試用 finish() 讓它完成，完成時才產生效果（寫預標 / 回快照）。"""

    def __init__(self, lib):
        self.lib = lib
        self.tasks: dict[str, dict] = {}
        self.metrics: list[dict] = []          # 依序給每次訓練的指標

    def _new(self, kind, **info):
        tid = f"fake-{kind}-{next(_seq)}"
        self.tasks[tid] = {"task_id": tid, "status": "working", "kind": kind, **info}
        return tid

    def autolabel(self, cid, params, actor):
        return self._new("autolabel", cid=cid, params=params)

    def export(self, cid, *, repo, message, val_pct, actor, batch_ids=None, task=None):
        return self._new("export", cid=cid, repo=repo, batch_ids=batch_ids, label_task=task)

    def mix(self, params, actor):
        return self._new("mix", params=params)

    def submit_training(self, body, actor):
        return self._new("train", body=body)

    def task(self, tid):
        return self.tasks.get(tid)

    def pending(self, kind):
        return [t for t in self.tasks.values() if t["kind"] == kind and t["status"] == "working"]

    def finish(self, kind, *, fail=False):
        for t in self.pending(kind):
            if fail:
                t.update(status="failed", error="假的失敗")
                continue
            if kind == "autolabel":
                items = self.lib.list_items(t["cid"], status=["unlabeled", "auto"], limit=1000)
                for it in items:
                    self.lib.write_auto(it["id"], [{**BOX[0], "source": {"kind": "model", "model": "m",
                                                                              "version": 1, "conf": .9}}],
                                            actor="t", task_id=t["task_id"])
            elif kind == "export":
                t["result"] = {"export": {"dataset_ref": f"lakefs://{t['repo']}/v{next(_seq)}"}}
            elif kind == "mix":
                t["result"] = {"mix": {"dataset_ref": f"lakefs://{t['params']['repo']}/v{next(_seq)}"}}
            elif kind == "train":
                t["model_version"] = str(next(_seq))
                t["result"] = {"trainer": {"metrics": self.metrics.pop(0)}}
            t["status"] = "completed"


@pytest.fixture()
def env():
    from edgevision.media.campaign import CampaignEngine
    from edgevision.media.library import MediaLibrary
    from edgevision.storage.db import Database
    from edgevision.storage.objects import ObjectStore

    os.environ["DATABASE_URL"] = os.environ["MEDIA_TEST_DSN"]
    db, objects = Database(), ObjectStore()
    try:
        objects._client.make_bucket("raw")
    except Exception:
        pass
    lib = MediaLibrary(db, objects)
    fa = FakeActions(lib)
    return CampaignEngine(lib, fa), fa, lib


def add_batch(lib, cid, name, n, *, labeled=False):
    """用圖片庫的方式加一批（同 API 的上傳：先建批次再放圖）。"""
    b = lib.create_batch(cid, kind="upload", actor="t", name=name)
    for i in range(n):
        ref = lib.put_image(_jpeg(hash((cid, name, i)) % 10 ** 6), filename=f"{name}{i}.jpg", actor="t")
        lib.add_item(cid, ref.sha256, actor="t", batch_id=b["id"],
                     shapes=BOX if labeled else None, status="labeled" if labeled else None)
    return b["id"]


def review_all(lib, cid, actor="alice"):
    """人在圖片集裡審（不開輪次）：還沒審的圖全部送出。"""
    for it in lib.list_items(cid, status=["unlabeled", "auto"], limit=1000):
        it = lib.item(it["id"], with_url=False)
        shapes = it["shapes"] or [{"type": "rect", "class": "glove", "points": [[.2, .2], [.3, .3]]}]
        lib.submit(it["id"], shapes, revision=it["revision"], actor=actor)


def _project(lib):
    name = f"cp-{uuid.uuid4().hex[:6]}"
    cid = lib.create_collection(name=name, actor="t", class_names=CLASSES)["id"]
    b0 = add_batch(lib, cid, "初始", 2, labeled=True)
    return name, cid, b0


def test_full_loop(env):
    eng, fa, lib = env
    name, cid, b0 = _project(lib)
    b1 = add_batch(lib, cid, "第一批", 5)
    with pytest.raises(ValueError):                                     # 初始模型必填
        eng.create(name=name, collection_id=cid, model_name="m", base_model="b", actor="t", initial_model={})
    pid = eng.create(name=name, collection_id=cid, model_name="m", base_model="s3://models/_base/yolo26m.pt",
                     actor="corp\\alice", initial_model={"name": "m_seed", "version": 1},
                     initial_batch_ids=[b0])["id"]
    d = eng.detail(pid)
    assert (d["state"], d["stage"], d["iteration"]) == ("idle", None, 1) and "waiting_reason" not in d
    assert d["initial"] == {"batch_ids": [b0], "model": {"name": "m_seed", "version": 1}}
    assert [(b["id"], b["initial"]) for b in d["batches"]] == [(b0, True), (b1, False)]
    assert d["iterations"][0]["model_name"] == "m_seed"
    assert d["latest_model"] == {"iteration": 0, "name": "m_seed", "version": 1}
    eng.reconcile_all()
    assert not fa.tasks                                                # 閒置：不會自己動

    # ① 只預標：用初始模型（m_seed v1，不是專案的 model_name）→ 回閒置，記在第 0 代
    eng.prelabel(pid, actor="t")
    (al,) = fa.pending("autolabel")
    assert (al["params"]["model_name"], al["params"]["model_version"], al["params"]["conf"]) == ("m_seed", 1, 0.10)
    with pytest.raises(Conflict):                                      # 跑的時候不能再按
        eng.run(pid, actor="t")
    fa.finish("autolabel")
    eng.reconcile_all()
    c = eng.get(pid)
    assert (c["state"], c["stage"], c["iteration"]) == ("idle", None, 1)
    assert eng.detail(pid)["iterations"][0]["prelabel_task_ids"] == [al["task_id"]]

    # ② 人在圖片集審（不開輪次，專案不管）
    review_all(lib, cid)

    # ③ 跑下一輪：初始批次與第一批各自一個快照（各自的 repo）
    eng.update(pid, actor="t", recipe={"batches": {str(b1): {"repeat": 2}}})
    eng.run(pid, actor="t")
    c = eng.get(pid)
    assert (c["state"], c["stage"], c["step"]["mode"]) == ("running", "exporting", "run")
    ex = {t["batch_ids"][0]: t for t in fa.pending("export")}
    assert set(ex) == {b0, b1} and ex[b1]["repo"] == f"campaign-{pid}-batch-{b1}"
    assert {t["label_task"] for t in ex.values()} == {"detect"}         # 沒設 → 預設
    fa.finish("export")

    # ④ 混合：照配方（第一批 ×2）
    eng.tick(pid)
    (mx,) = fa.pending("mix")
    assert mx["params"]["repo"] == f"campaign-{pid}-train"
    assert sorted(s.get("repeat", 1) for s in mx["params"]["sources"]) == [1, 2]
    fa.finish("mix")

    # ⑤ 訓練：dataset_ref = 訓練快照
    eng.tick(pid)
    (tr,) = fa.pending("train")
    it1 = eng.detail(pid)["iterations"][1]
    assert tr["body"]["dataset_ref"] == it1["dataset_ref"] and it1["dataset_ref"].startswith(f"lakefs://campaign-{pid}-train/")
    assert set(it1["batch_snapshots"]) == {str(b0), str(b1)} and it1["recipe"]["sources"][1]["repeat"] == 2
    assert tr["body"]["gates"]["absolute"]["map50_box"] == 0.2
    assert tr["body"]["config"]["task"] == "detect"
    fa.metrics.append({"basis": "val", "map50_box": 0.6, "per_class_recall_box": {"tap": 0.8, "mat": 1.0}})
    fa.finish("train")

    # ⑥ 預標：圖都審完了 → 略過 → 回閒置，記成第 1 代
    eng.tick(pid)
    c = eng.get(pid)
    assert (c["state"], c["stage"], c["iteration"]) == ("idle", None, 2) and not fa.pending("autolabel")
    d = eng.detail(pid)
    mv1 = int(fa.tasks[tr["task_id"]]["model_version"])
    assert d["iterations"][1]["model_version"] == mv1 and d["iterations"][1]["metrics"]["map50_box"] == 0.6
    assert d["iterations"][1]["regression"] is None
    assert d["latest_model"] == {"iteration": 1, "name": "m", "version": mv1}

    # ⑦ 用圖片庫加一批 → 只預標：用第 1 代的模型 → 記在第 1 代
    b2 = add_batch(lib, cid, "第二批", 3)
    eng.reconcile_all()
    assert not fa.pending("autolabel")                                  # 加批次不會自己預標
    eng.prelabel(pid, actor="t")
    (al2,) = fa.pending("autolabel")
    assert (al2["params"]["model_name"], al2["params"]["model_version"]) == ("m", mv1)
    fa.finish("autolabel")
    eng.tick(pid)
    assert eng.get(pid)["state"] == "idle"
    assert eng.detail(pid)["iterations"][1]["prelabel_task_ids"] == [al2["task_id"]]
    review_all(lib, cid)

    # ⑧ 跑下一輪：只有第二批是新的 → 只匯出它；訓練完有一張還沒審 → 用新模型預標
    b3 = add_batch(lib, cid, "第三批", 1)
    eng.run(pid, actor="t")
    (ex2,) = fa.pending("export")
    assert ex2["batch_ids"] == [b2]
    fa.finish("export")
    eng.tick(pid)
    fa.finish("mix")
    eng.tick(pid)
    fa.metrics.append({"basis": "val", "map50_box": 0.65, "per_class_recall_box": {"tap": 0.7, "mat": 1.0}})
    fa.finish("train")
    eng.tick(pid)                                                      # press 掉了也照走，不判退步
    c = eng.get(pid)
    assert (c["state"], c["stage"], c["iteration"]) == ("running", "prelabeling", 2)
    (al3,) = fa.pending("autolabel")
    mv2 = int(eng.detail(pid)["iterations"][2]["model_version"])
    assert al3["params"]["model_version"] == mv2
    fa.finish("autolabel")
    eng.reconcile_all()
    c = eng.get(pid)
    assert (c["state"], c["iteration"]) == ("idle", 3)
    assert eng.detail(pid)["iterations"][2]["prelabel_task_ids"] == [al3["task_id"]]
    review_all(lib, cid)

    # ⑨ 中途把標注格式改成 segment → 每一批都用新格式重匯（不會一代混兩種格式），訓練 task 跟著改
    eng.update(pid, actor="t", config={"dataset": {"task": "segment"}})
    eng.run(pid, actor="t")
    ex3 = {t["batch_ids"][0]: t["label_task"] for t in fa.pending("export")}
    assert set(ex3) == {b0, b1, b2, b3} and set(ex3.values()) == {"segment"}
    fa.finish("export")
    eng.tick(pid)
    assert {s["task"] for s in eng.detail(pid)["iterations"][3]["batch_snapshots"].values()} == {"segment"}
    fa.finish("mix")
    eng.tick(pid)
    (tr3,) = fa.pending("train")
    assert tr3["body"]["config"]["task"] == "segment"


def test_failure_retry_and_close(env):
    eng, fa, lib = env
    name, cid, b0 = _project(lib)
    pid = eng.create(name=name, collection_id=cid, model_name="m", base_model="b", actor="t",
                     initial_model={"name": "m_seed", "version": 1}, initial_batch_ids=[b0])["id"]
    with pytest.raises(Conflict, match="不是結案"):
        eng.reopen(pid, actor="t")
    add_batch(lib, cid, "第一批", 1, labeled=True)
    eng.run(pid, actor="t")
    with pytest.raises(Conflict, match="還在跑"):
        eng.close(pid, actor="t")
    fa.finish("export", fail=True)
    eng.tick(pid)
    c = eng.get(pid)
    assert (c["state"], c["stage"]) == ("failed", "exporting") and "假的失敗" in c["last_error"]
    with pytest.raises(Conflict):                                      # 失敗中不能開新的一輪
        eng.prelabel(pid, actor="t")
    eng.retry(pid, actor="t")
    assert len(fa.pending("export")) == 2 and eng.get(pid)["step"]["mode"] == "run"     # 從這一步重跑
    fa.finish("export")
    # 用比例但漏了一批 → 混合前就擋下、專案失敗並說明
    eng.update(pid, actor="t", recipe={"batches": {str(b0): {"proportion": 1}}, "total": 10})
    eng.tick(pid)
    c = eng.get(pid)
    assert (c["state"], c["stage"]) == ("failed", "mixing")
    # 沒有已確認的圖 → 匯出就失敗
    name2 = f"cp-{uuid.uuid4().hex[:6]}"
    cid2 = lib.create_collection(name=name2, actor="t", class_names=CLASSES)["id"]
    p2 = eng.create(name=name2, collection_id=cid2, model_name="m", base_model="b", actor="t",
                    initial_model={"name": "m_seed", "version": 1})["id"]
    assert eng.run(p2, actor="t")["state"] == "failed"
    # 沒有還沒審的圖 → 只預標直接回閒置
    assert eng.close(p2, actor="t")["state"] == "closed"
    with pytest.raises(Conflict, match="已結案"):
        eng.run(p2, actor="t")
    assert eng.reopen(p2, actor="t")["state"] == "idle"
    assert eng.prelabel(p2, actor="t")["state"] == "idle" and not fa.pending("autolabel")


def test_api_and_real_actions_wiring():
    """經過 API：建專案、詳情、配方、權限；StateActions 用真的 media 動作（任務真的建進 tasks 表）。"""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from edgevision.media.api import build_router
    from edgevision.media.campaign_api import engine_of
    from edgevision.storage.db import Database
    from edgevision.storage.objects import ObjectStore
    from edgevision.training.tracker import TaskStore

    class Orch:
        def __init__(self):
            self.cpu, self.gpu = [], []

        def submit_media_import(self, tid):
            self.cpu.append(tid)

        def submit_media_export(self, tid):
            self.cpu.append(tid)

        def submit(self, tid):
            self.gpu.append(tid)

    os.environ["DATABASE_URL"] = os.environ["MEDIA_TEST_DSN"]
    db = Database()
    state = {"db": db, "objects": ObjectStore(), "store": TaskStore(db), "orchestrator": Orch()}
    app = FastAPI()
    app.include_router(build_router(state, lambda x_actor, *_: (x_actor or "corp\\alice", "human", "api")))
    c = TestClient(app)
    name = f"cpa-{uuid.uuid4().hex[:6]}"
    cid = c.post("/api/media/collections", json={"name": name, "class_names": CLASSES}).json()["id"]
    files = [("files", (f"{i}.jpg", _jpeg(hash((name, i)) % 10 ** 6), "image/jpeg")) for i in range(3)]
    b1 = c.post(f"/api/media/collections/{cid}/images", files=files, params={"batch_name": "第一批"}).json()["batch_id"]

    base = {"name": name, "collection_id": cid, "model_name": "m", "base_model": "s3://models/_base/yolo26m.pt",
            "initial_model": {"name": "m_seed", "version": 1}}
    assert c.post("/api/media/campaigns", json={**base, "config": {"seed": {"size": 1}}}).status_code == 400
    assert c.post("/api/media/campaigns", json={**base, "initial_batch_ids": [999999]}).status_code == 400
    r = c.post("/api/media/campaigns", json=base)
    assert r.status_code == 201, r.text
    pid = r.json()["id"]

    assert r.json()["state"] == "idle" and not state["orchestrator"].gpu
    d = c.post(f"/api/media/campaigns/{pid}/prelabel").json()     # → 真的 start_autolabel：建任務、送 orchestrator
    assert (d["state"], d["stage"]) == ("running", "prelabeling")
    tid = d["step"]["task"]
    assert state["orchestrator"].gpu == [tid]
    t = state["store"].get(tid)
    assert t.params["model_name"] == "m_seed" and t.params["model_version"] == 1
    assert d["batches"][0]["id"] == b1 and d["batches"][0]["snapshot"] is None

    assert c.post(f"/api/media/campaigns/{pid}/run").status_code == 409          # 預標還在跑
    assert c.post(f"/api/media/campaigns/{pid}/continue").status_code in (404, 405)   # v0.3 拿掉
    p = c.patch(f"/api/media/campaigns/{pid}", json={"golden_ref": "lakefs://golden/v1",
                                                    "recipe": {"batches": {str(b1): {"repeat": 0.5}}}}).json()
    assert p["golden_ref"] == "lakefs://golden/v1" and p["recipe"] == {"batches": {str(b1): {"repeat": 0.5}}}
    assert c.patch(f"/api/media/campaigns/{pid}", json={"recipe": {"batches": {"999999": {"repeat": 1}}}}).status_code == 400
    p = c.patch(f"/api/media/campaigns/{pid}", json={"initial_model": {"name": "m_seed", "version": 2}}).json()
    assert p["initial_model"]["version"] == 2 and p["iterations"][0]["model_version"] == 2
    assert p["latest_model"] == {"iteration": 0, "name": "m_seed", "version": 2}
    assert any(c_["id"] == pid for c_ in c.get("/api/media/campaigns").json())


def test_collection_sources_and_one_project_per_collection(env):
    """圖片集三種來源（明講 / 初始批次推出 / 自動建）＋一個圖片集只給一個專案。"""
    eng, _, lib = env
    kw = {"model_name": "m", "base_model": "b", "actor": "t", "initial_model": {"name": "m_seed", "version": 1}}
    name, cid, b0 = _project(lib)
    c1 = eng.create(name=name, initial_batch_ids=[b0], **kw)            # 只給批次 → 批次所在的圖片集
    assert c1["collection_id"] == cid and c1["initial_batch_ids"] == [b0]
    with pytest.raises(Conflict, match="只給一個進行中的專案"):
        eng.create(name=name + "-2", collection_id=cid, **kw)
    with pytest.raises(Conflict, match="只給一個進行中的專案"):
        eng.create(name=name + "-3", initial_batch_ids=[b0], **kw)

    name2, cid2, b2 = _project(lib)
    with pytest.raises(ValueError, match="分屬不同圖片集"):
        eng.create(name=name2, initial_batch_ids=[b0, b2], **kw)
    with pytest.raises(Conflict, match="不能拿來當測試集"):               # 別人的待審圖片集不能當測試集
        eng.create(name=name2, collection_id=cid2, golden_collection_id=cid, **kw)
    eng.create(name=name2, collection_id=cid2, **kw)
    with pytest.raises(Conflict, match="不能拿來當測試集"):
        eng.update(c1["id"], actor="t", golden_collection_id=cid2)

    # 結案：圖片集釋放；接回時圖片集已被別人用 → 409；結案中不能改、不能跑
    assert eng.close(c1["id"], actor="corp\\alice")["state"] == "closed"
    with pytest.raises(Conflict, match="已結案"):
        eng.update(c1["id"], actor="t", golden_ref="lakefs://g/v1")
    with pytest.raises(Conflict, match="已結案"):
        eng.prelabel(c1["id"], actor="t")
    eng.tick(c1["id"])
    assert eng.get(c1["id"])["state"] == "closed"                      # 調和不會推結案的專案
    c1b = eng.create(name=name + "-new", collection_id=cid, **kw)
    with pytest.raises(Conflict, match="只給一個進行中的專案"):
        eng.reopen(c1["id"], actor="corp\\alice")
    eng.close(c1b["id"], actor="corp\\alice")
    r = eng.reopen(c1["id"], actor="corp\\alice")
    assert (r["state"], r["stage"]) == ("idle", None)

    with pytest.raises(ValueError, match="自動建圖片集"):
        eng.create(name=f"cp-{uuid.uuid4().hex[:6]}", **kw)
    n3 = f"cp-{uuid.uuid4().hex[:6]}"
    c3 = eng.create(name=n3, new_collection={"name": n3, "class_names": CLASSES,
                                             "line_id": "siteb_bravo", "camera_id": "siteb12"}, **kw)
    col = lib.collection(c3["collection_id"])
    assert col["name"] == n3 and col["class_names"] == CLASSES and "task" not in col
    assert col["line_id"] == "siteb_bravo" and c3["state"] == "idle" and c3["initial_batch_ids"] == []
    n4 = f"cp-{uuid.uuid4().hex[:6]}"
    with pytest.raises(Conflict, match="已存在"):                        # 專案名重複：不留下空的圖片集
        eng.create(name=n3, new_collection={"name": n4, "class_names": CLASSES}, **kw)
    assert not any(x["name"] == n4 for x in lib.list_collections())


def test_api_auto_creates_collection_from_initial_model():
    from types import SimpleNamespace

    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from edgevision.media.api import build_router
    from edgevision.storage.db import Database
    from edgevision.storage.objects import ObjectStore
    from edgevision.training.tracker import TaskStore

    class Reg:
        def get(self, name, version):
            if name != "m_seed":
                return None
            return SimpleNamespace(meta=SimpleNamespace(task="segment", class_names=["board", "mat"],
                                                        export=SimpleNamespace(n_kpt=0)))

    os.environ["DATABASE_URL"] = os.environ["MEDIA_TEST_DSN"]
    db = Database()
    state = {"db": db, "objects": ObjectStore(), "store": TaskStore(db), "orchestrator": None,
             "registry_factory": Reg}
    app = FastAPI()
    app.include_router(build_router(state, lambda x_actor, *_: (x_actor or "corp\\alice", "human", "api")))
    c = TestClient(app)
    name = f"cpn-{uuid.uuid4().hex[:6]}"
    base = {"name": name, "model_name": "m", "base_model": "s3://models/_base/yolo26m.pt",
            "initial_model": {"name": "m_seed", "version": 1}, "line_id": "siteb_bravo", "camera_id": "siteb12"}
    assert c.post("/api/media/campaigns", json={**base, "initial_model": {"name": "nope", "version": 1}}).status_code == 404
    r = c.post("/api/media/campaigns", json=base)
    assert r.status_code == 201, r.text
    col = c.get(f"/api/media/collections/{r.json()['collection_id']}").json()
    assert col["name"] == name and col["class_names"] == ["board", "mat"] and col["camera_id"] == "siteb12"
    assert r.json()["config"]["dataset"]["task"] == "segment"          # 標注格式跟著初始模型
    r2 = c.post("/api/media/campaigns", json={**base, "name": name + "-2", "collection_id": col["id"]})
    assert r2.status_code == 409 and "只給一個進行中的專案" in r2.text
