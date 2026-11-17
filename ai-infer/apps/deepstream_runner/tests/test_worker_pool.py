"""分片：每台相機固定一條 worker、順序保住；queue 滿只丟自己那個 shard。"""
import queue

import numpy as np
import pytest
from ds_runner.config import load_line_config
from ds_runner.worker import ShardRouter, TensorJob, WorkerPool, shard_cameras


def test_shard_assignment_is_round_robin_and_stable():
    cams = ["cam156", "cam193", "cam230", "cam26", "cam63", "cam100", "cam137", "cam174"]
    s = shard_cameras(cams, 4)
    assert s == {"cam156": 0, "cam193": 1, "cam230": 2, "cam26": 3, "cam63": 0, "cam100": 1, "cam137": 2, "cam174": 3}
    assert shard_cameras(["a", "b"], 4) == {"a": 0, "b": 1}          # worker 數不超過相機數


def test_pool_builds_one_station_per_camera_in_its_own_shard(line_file):
    cfg = load_line_config({"CAMERA_SOURCE": "file", "CAMERA_FILE": str(line_file)})
    pool = WorkerPool(cfg, publish=lambda r, img: None, n_workers=2)
    assert len(pool.workers) == 2 and len(pool.queues) == 2
    assert set(pool.workers[0].stations) == {"cam156"} and set(pool.workers[1].stations) == {"cam193"}
    assert pool.bbox_only_postproc is pool.workers[0].bbox_only_postproc


def test_router_dispatches_by_camera_and_counts_drops_per_shard():
    stats = {}
    q0, q1 = queue.Queue(maxsize=1), queue.Queue(maxsize=1)
    r = ShardRouter([q0, q1], {"cam156": 0, "cam193": 1}, stats)
    job = lambda c: TensorJob(c, np.zeros((1, 1)), None, 1280, 720)
    r.put_nowait(job("cam156"))
    r.put_nowait(job("cam193"))
    with pytest.raises(queue.Full):
        r.put_nowait(job("cam156"))                                    # shard0 滿
    assert stats == {"dropped_shard0": 1} and q1.qsize() == 1          # shard1 不受影響


def test_router_put_blocks_instead_of_dropping():
    """不限速的影片測試：滿了就等 worker 消化（回壓），不丟幀、不計 dropped。"""
    import threading
    stats = {}
    q0 = queue.Queue(maxsize=1)
    r = ShardRouter([q0], {"cam156": 0}, stats)
    r.put(TensorJob("cam156", np.zeros((1, 1)), None, 1280, 720, pts=1))
    threading.Timer(0.2, q0.get).start()               # worker 0.2 秒後拿走一筆
    r.put(TensorJob("cam156", np.zeros((1, 1)), None, 1280, 720, pts=2))
    assert q0.get_nowait().pts == 2 and stats == {}


def test_pool_processes_in_order_per_camera(line_file):
    """同一台相機的幀依序進同一條 worker：站別狀態（連續幀）才可靠。"""
    import threading
    import time
    cfg = load_line_config({"CAMERA_SOURCE": "file", "CAMERA_FILE": str(line_file)})
    seen = []
    pool = WorkerPool(cfg, publish=lambda r, img: seen.append((r.camera_id, r.frame_id)), n_workers=2)
    stop = threading.Event()
    pool.start(stop)
    out0 = np.zeros((38, 8400), np.float32)
    for _ in range(3):
        pool.router.put_nowait(TensorJob("cam156", out0, None, 1280, 720))
        pool.router.put_nowait(TensorJob("cam193", out0, None, 1280, 720))
    time.sleep(1.0)
    stop.set()
    assert [f for c, f in seen if c == "cam156"] == ["1", "2", "3"]
    assert [f for c, f in seen if c == "cam193"] == ["1", "2", "3"]
