from edgevision_contracts import (
    BBox,
    Detection,
    FrameResult,
    ImageMeta,
    LetterboxParams,
    Verdict,
    VerdictStatus,
)


def test_letterbox_roundtrip():
    lb = LetterboxParams(scale=0.5, pad_x=0.0, pad_y=140.0)
    x, y = lb.to_letterboxed(100.0, 200.0)
    assert lb.to_original(x, y) == (100.0, 200.0)


def test_frame_result_serialises_with_verdict():
    det = Detection(class_id=0, class_name="board", confidence=0.9, bbox=BBox(x1=0, y1=0, x2=10, y2=10))
    fr = FrameResult(
        camera_id="cam156", frame_id="1", model_name="seg_alpha", model_version="1",
        image_meta=ImageMeta(width=1280, height=720), detections=[det], verdict=Verdict.ng("違規動作"),
    )
    data = fr.model_dump()
    assert data["verdict"]["status"] == VerdictStatus.NG.value
    assert fr.by_class("board") == [det]
