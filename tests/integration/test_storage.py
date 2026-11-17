"""儲存層整合測試:結果真的寫進 PG 與 MinIO。"""
import io
import numpy as np

from libs.contracts import BBox, Detection, FrameResult, ImageMeta, Verdict
from edgevision.storage.sink import ResultSink, is_ng


def _make_result(i: int, ng: bool = False) -> FrameResult:
    return FrameResult(
        camera_id="test-cam", frame_id=f"test:{i}", marker_id=90000 + i,
        model_name="test", model_version="1",
        image_meta=ImageMeta(width=1280, height=720),
        detections=[
            Detection(class_id=0, class_name="board", confidence=0.94,
                      bbox=BBox(x1=100, y1=100, x2=400, y2=350),
                      has_mask=True, mask_area=60000),
            Detection(class_id=1, class_name="hand", confidence=0.45 if ng else 0.88,
                      bbox=BBox(x1=380, y1=150, x2=460, y2=240)),
        ],
        verdict=Verdict.ng("動作", hits=1) if ng else Verdict.ok(),
    )


def test_minio_reachable(minio):
    assert minio.ping()


def test_db_reachable(db):
    assert db.ping()


def test_write_and_query(db, minio):
    """落地一批結果,查得回來,NG 篩選正確,圖存得進。

    用唯一 camera_id 隔離本測試的資料 —— 真實推論服務可能同時在寫入,
    不能靠全域計數差值判斷。
    """
    import time, uuid
    cam = f"itest-{uuid.uuid4().hex[:8]}"

    sink = ResultSink(db=db, objects=minio)
    sink.start()
    img = np.random.randint(0, 255, (720, 1280, 3), np.uint8)
    for i in range(6):
        r = _make_result(i, ng=(i % 3 == 0))
        r.camera_id = cam   # 用唯一相機隔離
        sink.submit(r, img)
    time.sleep(3)
    stats = sink.health()
    sink.stop()

    assert stats["errors"] == 0
    assert stats["events_written"] == 2      # 只存 NG：i = 0、3
    assert stats["images_stored"] == 2

    # 只查自己這個相機的資料 —— 不受其他寫入干擾
    mine = db.recent_detections(camera_id=cam, limit=50)
    assert len(mine) == 4    # 2 個 NG 幀 × 2 物件；OK 幀不進資料庫

    ng = db.recent_detections(camera_id=cam, verdict="ng", limit=10)
    assert len(ng) >= 2
    assert all(r["verdict_status"] == "ng" for r in ng)
    assert any(r["image_uri"] for r in ng)


def test_only_ng_is_kept():
    """只存 NG（2026-10-05 拿掉正常幀抽樣）。"""
    assert is_ng(_make_result(0, ng=True))
    assert not is_ng(_make_result(1, ng=False))


def test_frame_level_ng_without_detections_is_stored(db):
    """靠時間判的 NG（當下沒有框）也要進資料庫一列：類別與框為 NULL，判定原因照寫。
    以前沒框就不寫 → 前端查不到、錄影服務錄好的影片掛不上去（2026-10-02）。"""
    import uuid
    cam = f"test-frame-ng-{uuid.uuid4().hex[:6]}"
    r = FrameResult(camera_id=cam, frame_id="42", model_name="m", model_version="1",
                    station="bravo_check", image_meta=ImageMeta(width=640, height=480),
                    detections=[], verdict=Verdict(status="ng", reason="逾時未確認"))
    assert db.insert_detections(r) == 1
    row = db.recent_detections(camera_id=cam, limit=5)[0]
    assert row["class_name"] is None and row["bbox"] is None
    assert row["verdict_status"] == "ng" and row["station"] == "bravo_check"
    ok = FrameResult(camera_id=cam, frame_id="43", model_name="m", model_version="1",
                     image_meta=ImageMeta(width=640, height=480), detections=[])
    assert db.insert_detections(ok) == 0           # 沒東西的 OK 幀照舊不寫


def test_expire_evidence_keeps_row_but_drops_links(db):
    """影片保留 7 天：過期後紀錄保留、clip_uri 清掉、記下 evidence_expired_at（2026-10-05）。"""
    import uuid
    cam = f"test-expire-{uuid.uuid4().hex[:6]}"
    r = FrameResult(camera_id=cam, frame_id="1", model_name="m", model_version="1",
                    image_meta=ImageMeta(width=640, height=480), detections=[],
                    verdict=Verdict(status="ng", reason="x"))
    db.insert_detections(r)
    db.attach_clip(camera_id=cam, frame_id="1", clip_uri="s3://inference-output/x.mp4",
                   event_ts=__import__("time").time())
    with db.cursor() as cur:
        cur.execute("UPDATE detections SET ts = now() - interval '8 days' WHERE camera_id = %s", (cam,))
    assert db.expire_evidence(older_than_days=7, camera_id=cam) == 1
    row = db.recent_detections(camera_id=cam, limit=5)[0]
    assert row["clip_uri"] is None and row["evidence_expired_at"] is not None
    assert row["verdict_status"] == "ng"                      # 紀錄還在
    assert db.expire_evidence(older_than_days=7, camera_id=cam) == 0  # 可重複執行


def test_review_marks_all_rows_of_frame(db):
    """B3：審核以幀為單位，同一幀的每一列一起標；可改判。"""
    import uuid
    cam = f"test-review-{uuid.uuid4().hex[:6]}"
    r = _make_result(0, ng=True); r.camera_id = cam
    db.insert_detections(r)
    assert db.review(camera_id=cam, frame_id=r.frame_id, result="false_positive", actor="op1") == 2
    assert db.review(camera_id=cam, frame_id=r.frame_id, result="true_positive", actor="op2") == 2
    rows = db.recent_detections(camera_id=cam, limit=5)
    assert all(x["reviewed"] and x["review_result"] == "true_positive" and x["reviewed_by"] == "op2" for x in rows)
    assert db.recent_detections(camera_id=cam, reviewed=False, limit=5) == []
