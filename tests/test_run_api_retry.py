"""output-api 啟動時連不上資料庫要在背景重試、連上自動接上（K8s 上它常比 postgres 先起來）。"""
import run_api


def test_retry_until_both_connected(monkeypatch):
    seq = iter([(None, "obj"), ("db", "obj")])
    bound = []
    monkeypatch.setattr(run_api, "_connect_storage", lambda db, objects: next(seq))
    monkeypatch.setattr(run_api, "bind_storage", lambda db, objects: bound.append((db, objects)))
    monkeypatch.setattr(run_api.time, "sleep", lambda s: None)
    run_api._retry_storage(None, None)
    assert bound[-1] == ("db", "obj") and len(bound) == 2
