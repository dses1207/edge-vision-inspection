"""事件影片的 URI 要寫回 DB —— 否則影片是孤兒檔案。

⚠️ 這個洞真的發生過：片段有上傳 MinIO，但 make() 的回傳值沒人接
   （run_recorder 是 `pool.submit(maker.make, ...)`，結果直接丟棄），
   於是 2.7 天產出 3312 個沒有任何紀錄指向的影片。

這裡測的是「影片存得到 = 前端查得到」這條鏈不能再斷。
"""
from __future__ import annotations

from contextlib import contextmanager
from unittest.mock import MagicMock

import pytest

from edgevision.storage.db import Database
from run_recorder import _make_and_attach


# ── attach_clip 的 SQL ────────────────────────────────────

def _db_with_cursor(rowcount: int = 1):
    db = Database.__new__(Database)          # 不跑 __init__，不連線
    cur = MagicMock()
    cur.rowcount = rowcount

    @contextmanager
    def fake_cursor():
        yield cur
    db.cursor = fake_cursor
    return db, cur


def test_attach_clip_returns_row_count():
    """一次 NG 在 detections 裡是多列（一列 = 一個物件），全部都要掛上。"""
    db, _ = _db_with_cursor(rowcount=3)
    n = db.attach_clip(camera_id="cam156", frame_id="f1",
                       clip_uri="s3://b/x.mp4", event_ts=1000.0)
    assert n == 3


def test_attach_clip_scopes_by_camera_and_time():
    """⚠️ frame_id 沒有索引，單靠它 UPDATE 會全表掃描。

    帶上 camera_id + ts 範圍才能走 idx_detections_camera_ts。
    """
    db, cur = _db_with_cursor()
    db.attach_clip(camera_id="cam156", frame_id="f1",
                   clip_uri="s3://b/x.mp4", event_ts=1000.0)
    sql = " ".join(cur.execute.call_args[0][0].split())
    assert "camera_id = %s" in sql
    assert "ts BETWEEN %s AND %s" in sql


def test_attach_clip_window_is_symmetric():
    """片段要等後段錄完才上傳，寫回來時那筆 detection 已是一兩分鐘前的事。"""
    db, cur = _db_with_cursor()
    db.attach_clip(camera_id="cam156", frame_id="f1",
                   clip_uri="s3://b/x.mp4", event_ts=1000.0, window_s=120)
    lo, hi = cur.execute.call_args[0][1][3:5]
    assert (hi - lo).total_seconds() == 240


def test_attach_clip_does_not_overwrite():
    """已經掛過影片的列不該被後來的事件覆蓋。"""
    db, cur = _db_with_cursor()
    db.attach_clip(camera_id="cam156", frame_id="f1",
                   clip_uri="s3://b/x.mp4", event_ts=1000.0)
    assert "clip_uri IS NULL" in " ".join(cur.execute.call_args[0][0].split())


# ── 寫回流程的每條路徑 ───────────────────────────────────

class _Maker:
    def __init__(self, uri): self.uri = uri
    def make(self, cam, ts, payload): return self.uri


class _DB:
    def __init__(self, n=1, boom=False):
        self.n, self.boom, self.calls = n, boom, []

    def attach_clip(self, **kw):
        self.calls.append(kw)
        if self.boom:
            raise RuntimeError("DB 掛了")
        return self.n


def test_uri_is_written_back():
    """⭐ 核心：片段做成了就要寫回 DB。"""
    db = _DB()
    _make_and_attach(_Maker("s3://b/x.mp4"), db, "cam156", 1000.0, {"frame_id": "f1"})
    assert db.calls == [{"camera_id": "cam156", "frame_id": "f1",
                         "clip_uri": "s3://b/x.mp4", "event_ts": 1000.0}]


def test_no_clip_means_no_db_write():
    """冷卻中或沒有預錄緩衝時 make() 回 None，不該去碰 DB。"""
    db = _DB()
    _make_and_attach(_Maker(None), db, "cam156", 1000.0, {"frame_id": "f1"})
    assert db.calls == []


def test_db_failure_does_not_lose_the_clip():
    """影片已經上傳成功了，DB 寫不進去只是「查不到」，不是「沒有」。

    所以不能讓例外逸出 —— 那會讓 ThreadPoolExecutor 吞掉錯誤而且無人知曉。
    """
    _make_and_attach(_Maker("s3://b/x.mp4"), _DB(boom=True),
                     "cam156", 1000.0, {"frame_id": "f1"})   # 不該拋


def test_maker_failure_does_not_propagate():
    class Boom:
        def make(self, *a): raise RuntimeError("ffmpeg 掛了")
    _make_and_attach(Boom(), _DB(), "cam156", 1000.0, {"frame_id": "f1"})


def test_zero_rows_is_not_an_error():
    """那一幀可能根本沒進 DB（抽樣只留 NG + 0.5% 背景）。"""
    _make_and_attach(_Maker("s3://b/x.mp4"), _DB(n=0),
                     "cam156", 1000.0, {"frame_id": "f1"})


@pytest.mark.parametrize("db,payload", [
    (None, {"frame_id": "f1"}),        # 沒有 DB
    (_DB(), {}),                       # 事件沒帶 frame_id
])
def test_unindexable_clip_warns_but_survives(db, payload, caplog):
    """這兩種情況影片存得了但查不到 —— 要警告，不能靜默。"""
    with caplog.at_level("WARNING"):
        _make_and_attach(_Maker("s3://b/x.mp4"), db, "cam156", 1000.0, payload)
    assert "查不到" in caplog.text


# ── 契約：查詢與回應型別要帶 clip_uri ────────────────────

def test_query_selects_clip_uri():
    """欄位存在、也寫進去了，但查詢沒 SELECT 它的話前端還是拿不到。"""
    import inspect
    assert "clip_uri" in inspect.getsource(Database.recent_detections)


def test_response_schema_declares_clip_uri():
    """⚠️ FastAPI 照 response_model 序列化 —— 沒宣告的欄位【不會回給前端】。"""
    from edgevision.api.schemas import DetectionRow
    assert "clip_uri" in DetectionRow.model_fields
