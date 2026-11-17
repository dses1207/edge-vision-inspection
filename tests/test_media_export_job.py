"""圖片集匯出 job（edgevision.media.jobs export）：參數從 tasks.params 來，成功 / 失敗寫回任務。"""
from types import SimpleNamespace

from edgevision.media import jobs


class Store:
    def __init__(self, params):
        self.t = SimpleNamespace(params=params, created_by="corp\\alice")
        self.result, self.status, self.msg = {}, None, ""

    def get(self, tid): return self.t
    def heartbeat(self, tid, msg=None): pass
    def set_result(self, tid, patch): self.result.update(patch)

    def finish(self, tid, status, message=None, error=None, **kw):
        self.status, self.msg = status.value, message


P = {"collection_id": 4, "repo": "siteb-bravo", "message": "m", "val_pct": 20}


def test_run_export_success(monkeypatch):
    store, seen = Store(P), {}
    monkeypatch.setattr(jobs, "_deps", lambda: ("LIB", store))

    def fake_export(lib, cid, **kw):
        seen.update(lib=lib, cid=cid, **kw)
        return {"export_id": 1, "dataset_ref": "lakefs://siteb-bravo/abc"}
    monkeypatch.setattr("edgevision.media.export.export_collection", fake_export)
    assert jobs.main(["export", "--task-id", "t1"]) == jobs.EXIT_OK
    assert store.status == "completed" and store.result["export"]["dataset_ref"].endswith("abc")
    assert seen == {"lib": "LIB", "cid": 4, "repo": "siteb-bravo", "message": "m", "actor": "corp\\alice",
                    "task_id": "t1", "val_pct": 20, "batch_ids": None, "task": None}


def test_run_export_failure(monkeypatch):
    store = Store(P)
    monkeypatch.setattr(jobs, "_deps", lambda: ("LIB", store))

    def boom(*a, **kw):
        raise ValueError("沒有可匯出的圖")
    monkeypatch.setattr("edgevision.media.export.export_collection", boom)
    assert jobs.main(["export", "--task-id", "t1"]) == jobs.EXIT_ERROR
    assert store.status == "failed" and "沒有可匯出的圖" in store.msg
