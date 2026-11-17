"""取消任務：排隊中的直接收掉、執行中的標記請求、結束的拒絕（2026-10-21：以前排隊中的取消不了）。"""
import os

import pytest

pytestmark = pytest.mark.skipif(not os.environ.get("MEDIA_TEST_DSN"), reason="需要 MEDIA_TEST_DSN")


@pytest.fixture(scope="module")
def store():
    from edgevision.storage.db import Database
    from edgevision.training.tracker import TaskStore
    return TaskStore(Database(os.environ["MEDIA_TEST_DSN"]))


def _new(store, status):
    from libs.contracts.training import TaskType
    return store.create(type=TaskType.TRAINING, created_by="itest", channel="api", params={}, status=status)


def test_cancel_queued_task_finishes_it(store):
    from libs.contracts.training import TaskStatus
    tid = _new(store, TaskStatus.QUEUED)
    assert store.request_cancel(tid)
    t = store.get(tid)
    assert t.status == "cancelled" and t.finished_at
    assert store.claim(tid) is False          # 排程器不會再把它拿出來跑


def test_cancel_working_task_only_flags(store):
    from libs.contracts.training import TaskStatus
    tid = _new(store, TaskStatus.WORKING)
    assert store.request_cancel(tid)
    assert store.get(tid).status == "working" and store.cancel_requested(tid)


def test_cancel_finished_task_rejected(store):
    from libs.contracts.training import TaskStatus
    tid = _new(store, TaskStatus.WORKING)
    store.finish(tid, TaskStatus.COMPLETED)
    assert not store.request_cancel(tid)
