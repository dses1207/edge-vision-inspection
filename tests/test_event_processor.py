"""Event Processor 單元測試 —— 用假的 Redis 與 DB，不需要外部服務。

重點測的是三件會造成【資料靜默遺失】的行為：
  ① 寫失敗絕不能 ack（ack 掉就永久消失）
  ② 格式壞掉的訊息一定要 ack（否則永遠卡在 pending list 重試）
  ③ 只存 NG：OK 幀不進資料庫，但一樣要 ack
"""
from __future__ import annotations

import pytest

from libs.contracts import BBox, Detection, FrameResult, ImageMeta, Verdict

from edgevision.storage.event_processor import EventProcessor


def _result(ng: bool = False, conf: float = 0.94) -> FrameResult:
    return FrameResult(
        camera_id="cam156", frame_id="1", model_name="seg_alpha", model_version="1",
        image_meta=ImageMeta(width=1280, height=720),
        detections=[
            Detection(class_id=0, class_name="board", confidence=conf,
                      bbox=BBox(x1=10, y1=10, x2=100, y2=100)),
        ],
        verdict=Verdict.ng("動作") if ng else Verdict.ok(),
    )


class FakeDB:
    """記下寫了幾次；可設定成拋錯。"""

    def __init__(self, fail: bool = False) -> None:
        self.fail = fail
        self.written: list[FrameResult] = []

    def insert_detections(self, result, *, image_uri=None) -> int:
        if self.fail:
            raise RuntimeError("DB 掛了")
        self.written.append(result)
        return len(result.detections)


class FakeRedis:
    """只實作 processor 用到的三個方法。"""

    def __init__(self, messages) -> None:
        self._messages = messages          # [(id, {"payload": ...}), ...]
        self.acked: list[str] = []

    def xreadgroup(self, _group, _consumer, _streams, count=10, block=0):
        if not self._messages:
            return []
        batch, self._messages = self._messages[:count], self._messages[count:]
        return [("edgevision:store", batch)]

    def xack(self, _stream, _group, *ids):
        self.acked.extend(ids)
        return len(ids)


def _proc(db, messages) -> EventProcessor:
    p = EventProcessor(db=db, redis_url="redis://fake")
    p._redis = FakeRedis(messages)
    return p


def test_writes_and_acks_on_success():
    db = FakeDB()
    p = _proc(db, [("1-0", {"payload": _result(ng=True).model_dump_json()})])
    p.run_once()
    assert len(db.written) == 1
    assert p._redis.acked == ["1-0"]
    assert p.stats.written == 1 and p.stats.errors == 0


def test_write_failure_does_not_ack():
    """⚠️ 最關鍵的一條：ack 掉就等於資料永久消失。

    不 ack 的訊息留在 pending list，重啟或別的副本會接手。
    """
    db = FakeDB(fail=True)
    p = _proc(db, [("1-0", {"payload": _result(ng=True).model_dump_json()})])
    p.run_once()
    assert p._redis.acked == []          # 沒有 ack
    assert p.stats.errors == 1


def test_malformed_message_is_acked():
    """格式壞掉的一定要 ack —— 不然它會永遠卡在 pending list 重試，擋住後面的。"""
    db = FakeDB()
    p = _proc(db, [("1-0", {"wrong_field": "x"})])
    p.run_once()
    assert p._redis.acked == ["1-0"]
    assert p.stats.errors == 1
    assert db.written == []


def test_unparseable_payload_is_not_acked():
    """payload 存在但解不開：可能是版本不相容，留著人工處理，不要默默丟掉。"""
    db = FakeDB()
    p = _proc(db, [("1-0", {"payload": "{不是合法 JSON"})])
    p.run_once()
    assert p._redis.acked == []
    assert p.stats.errors == 1


def test_empty_read_returns_zero():
    p = _proc(FakeDB(), [])
    assert p.run_once() == 0


# ── 只存 NG ────────────────────────────────────────────────

def test_ng_always_stored():
    db = FakeDB()
    msgs = [(f"{i}-0", {"payload": _result(ng=True).model_dump_json()}) for i in range(20)]
    p = _proc(db, msgs)
    p.run_once()
    assert len(db.written) == 20


def test_ok_frames_not_stored_but_acked():
    """OK 幀不進資料庫（2026-10-05 拿掉抽樣），但一定要 ack —— 已經處理完了，只是決定不寫。"""
    db = FakeDB()
    msgs = [(f"{i}-0", {"payload": _result(conf=c).model_dump_json()}) for i, c in enumerate([0.94, 0.52] * 50)]
    p = _proc(db, msgs)
    p.batch = 100
    p.run_once()
    assert db.written == [] and p.stats.skipped == 100
    assert len(p._redis.acked) == 100


def test_consumer_name_defaults_to_hostname():
    """多副本時 consumer 名必須不同，否則 pending list 會互搶。"""
    import socket
    p = EventProcessor(db=FakeDB(), redis_url="redis://fake")
    assert p.consumer == socket.gethostname()


# ── 心跳過期判定 ──────────────────────────────────────────
# 放這裡是因為它跟落地一樣，屬於「維運看板不能騙人」這類問題。

def test_stale_detection():
    """runner 消失後，line_status 留下的快照不能再被當成現況。

    實際發生過：line_c 的 runner 已消失 5.7 天，/healthz 仍顯示
    apply_state=ok、cameras_running=6、in_sync=true。
    """
    from edgevision.api.config_api import HEARTBEAT_STALE_S, _is_stale

    assert _is_stale(None) is True            # 從來沒回報過
    assert _is_stale(491831) is True          # line_c 的實際值（5.7 天）
    assert _is_stale(HEARTBEAT_STALE_S + 1) is True
    assert _is_stale(1) is False              # alpha_a 的實際值
    assert _is_stale(HEARTBEAT_STALE_S) is False


def test_stale_threshold_covers_missed_heartbeats():
    """runner 每 15 秒回報一次 —— 門檻要容許漏掉幾次，不能一漏就判死。"""
    from edgevision.api.config_api import HEARTBEAT_STALE_S

    assert HEARTBEAT_STALE_S >= 45, "至少要能容忍連續漏掉 2 次心跳"
