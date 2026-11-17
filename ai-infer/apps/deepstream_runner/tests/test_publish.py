import json

from ds_runner.publish import RedisPublisher, make_publisher, result_to_event
from edgevision_contracts import BBox, Detection, FrameResult, ImageMeta, Verdict


class FakeRedis:
    def __init__(self):
        self.calls = []

    def ping(self):
        return True

    def xadd(self, stream, fields, maxlen=None, approximate=None):
        self.calls.append((stream, fields, maxlen))


def result(ng=True):
    det = Detection(class_id=0, class_name="board", confidence=0.9, bbox=BBox(x1=0, y1=0, x2=10, y2=10))
    return FrameResult(camera_id="cam156", frame_id="7", model_name="seg_alpha", model_version="1",
                       image_meta=ImageMeta(width=1280, height=720), detections=[det],
                       verdict=Verdict.ng("違規動作", hits=1) if ng else Verdict.ok())


def test_publishes_events_and_live_in_legacy_format():
    r = FakeRedis()
    pub = RedisPublisher("redis://x", client=r, line_id="alpha_a", station="zone_intrusion")
    pub(result())
    (s1, f1, m1), (s2, f2, m2), (s3, f3, m3) = r.calls
    assert s1 == "edgevision:events" and f1["type"] == "ng" and f1["camera_id"] == "cam156" and f1["frame_id"] == "7"
    assert f1["line_id"] == "alpha_a" and f1["station"] == "zone_intrusion"       # 通知按線分組要用
    assert json.loads(f1["details"]) == {"hits": 1} and m1 == 100_000
    assert s2 == "edgevision:live" and m2 == 500
    live = json.loads(f2["payload"])
    assert live["type"] == "result" and live["detections"][0]["class_name"] == "board"
    # 落地 stream：payload 與 live 相同，但保留久得多（live 的 500 筆只夠 4.2 秒）
    assert s3 == "edgevision:store" and m3 == 200_000 and f3["payload"] == f2["payload"]
    assert result_to_event(result(ng=False))["type"] == "ok"


def test_ok_frames_skip_events_stream_but_still_go_live_and_store():
    """⚠️ edgevision:events 只發 NG（2026-09-24 定案）。

    OK 幀每秒上百筆，會把 10 萬筆的 maxlen 在 14 分鐘內沖掉，通知消費者重啟就漏 NG。
    查過沒有人依賴 OK 事件：recorder 只吃 ng、落地走 store、前端即時走 live。
    """
    r = FakeRedis()
    pub = RedisPublisher("redis://x", client=r)
    pub(result(ng=False))
    streams = [c[0] for c in r.calls]
    assert "edgevision:events" not in streams
    assert streams == ["edgevision:live", "edgevision:store"]


def test_make_publisher_takes_line_id_from_env_when_not_given(monkeypatch):
    import ds_runner.publish as pubmod
    captured = {}

    class FakePublisher:
        def __init__(self, url, **kw):
            captured.update(kw)

        def ping(self):
            return True

    monkeypatch.setattr(pubmod, "RedisPublisher", FakePublisher)
    pubmod.make_publisher({"REDIS_URL": "redis://x", "LINE_ID": "alpha_a"}, station="zone_intrusion")
    assert captured["line_id"] == "alpha_a" and captured["station"] == "zone_intrusion"


def test_empty_frame_skips_store_stream():
    """沒偵測到東西的幀不進落地 stream —— 寫進 DB 沒意義，只會佔掉保留空間。

    live 照發（前端要知道「這一幀沒東西」）；events 不發（OK 幀已全面不進 events，見上一個測試）。
    """
    r = FakeRedis()
    pub = RedisPublisher("redis://x", client=r)
    empty = FrameResult(camera_id="cam156", frame_id="8", model_name="seg_alpha", model_version="1",
                        image_meta=ImageMeta(width=1280, height=720), detections=[],
                        verdict=Verdict.ok())
    pub(empty)
    assert [c[0] for c in r.calls] == ["edgevision:live"]


def test_ng_without_detections_still_goes_to_store():
    """靠時間判的 NG（「A 之後 N 秒內沒看到 B」這類）當下沒有框，也要進落地 stream，DB 才有這筆 NG、影片才掛得上去。"""
    r = FakeRedis()
    pub = RedisPublisher("redis://x", client=r)
    ng = FrameResult(camera_id="ipcam87", frame_id="9", model_name="bravo", model_version="1",
                     image_meta=ImageMeta(width=1920, height=1080), detections=[],
                     verdict=Verdict(status="ng", reason="逾時未確認"))
    pub(ng)
    assert [c[0] for c in r.calls] == ["edgevision:events", "edgevision:live", "edgevision:store"]


def test_store_stream_can_be_disabled():
    """store_maxlen=0 停用落地發布（例如純測試線路，不想污染 DB）。"""
    r = FakeRedis()
    pub = RedisPublisher("redis://x", client=r, store_maxlen=0)
    pub(result())
    assert "edgevision:store" not in [c[0] for c in r.calls]


def test_publish_failure_never_raises():
    class Boom(FakeRedis):
        def xadd(self, *a, **k):
            raise ConnectionError
    pub = RedisPublisher("redis://x", client=Boom())
    pub(result())
    assert pub.failed == 1 and pub.published == 0


def test_make_publisher_falls_back_to_log_without_url():
    assert make_publisher({}).__name__ == "log_publisher"


def test_file_publisher_for_video_test(tmp_path):
    # 影片測試：RESULTS_FILE 優先於 Redis，每幀一行；t 從第一筆起算；mask 輪廓不寫
    out = tmp_path / "r.jsonl"
    pub = make_publisher({"RESULTS_FILE": str(out), "REDIS_URL": "redis://x"})
    a, b = result(), result(ng=False)
    a.detections[0].mask_contour = [[1, 2], [3, 4], [5, 6]]
    b.timestamp = a.timestamp + 1.5
    pub(a)
    pub(b)
    pub.close()
    rows = [json.loads(x) for x in out.read_text().splitlines()]
    assert [r["t"] for r in rows] == [0.0, 1.5]
    assert rows[0]["verdict"]["status"] == "ng" and rows[1]["verdict"]["status"] == "ok"
    assert "mask_contour" not in rows[0]["detections"][0] and rows[0]["detections"][0]["class_name"] == "board"


def test_file_publisher_uses_video_pts_when_present(tmp_path):
    # 不限速的影片測試：牆鐘跟影片秒數對不上 → 有 pts（奈秒）就用影片時間
    out = tmp_path / "r.jsonl"
    pub = make_publisher({"RESULTS_FILE": str(out)})
    a, b = result(), result(ng=False)
    a.pts, b.pts = 2_000_000_000, 4_500_000_000
    b.timestamp = a.timestamp + 0.01                  # 跑很快：牆鐘只過了 10ms
    pub(a)
    pub(b)
    pub.close()
    assert [json.loads(x)["t"] for x in out.read_text().splitlines()] == [0.0, 2.5]


def test_drain_queues_waits_for_workers():
    import queue
    import threading

    from ds_runner.pipeline import drain_queues

    q = queue.Queue()
    for i in range(3):
        q.put(i)
    threading.Thread(target=lambda: [q.get() for _ in range(3)], daemon=True).start()
    drain_queues([q], timeout_s=5, settle_s=0)
    assert q.empty()
