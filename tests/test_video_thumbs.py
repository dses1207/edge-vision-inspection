"""影片庫縮圖（edgevision/media/video_thumbs.py）：取幀、取第幾秒、快取、檔案不在 / 解不開。"""
from datetime import datetime, timezone

import cv2
import numpy as np
import pytest

from edgevision.media import video_thumbs as VT
from edgevision.media.frames import grab_frame


def _mp4(path, seconds: int, fps: int = 5, size=(320, 180)):
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    for k in range(seconds * fps):
        vw.write(np.full((size[1], size[0], 3), min(255, k * 10), np.uint8))   # 亮度隨時間變，看得出取到哪一幀
    vw.release()
    return path


class _NoSuchKey(Exception):
    code = "NoSuchKey"


class FakeObjects:
    def __init__(self):
        self.store, self.gets = {}, 0

    def get(self, *, bucket, key):
        self.gets += 1
        if (bucket, key) not in self.store:
            raise _NoSuchKey(key)
        return self.store[(bucket, key)]

    def put(self, *, bucket, key, data, length, content_type):
        self.store[(bucket, key)] = data.read()

    def presigned_url(self, **kw):
        raise AssertionError("NAS 錄影不該走 MinIO")


class FakeLib:
    def __init__(self):
        self.objects = FakeObjects()

    def blob(self, sha):
        return None


def _video(**kw):
    return {"id": 7, "kind": "nas_recording", "uri": "nas://raw/siteb12/20261018_000000.mp4", "blob_sha256": None,
            "params": {}, "duration_s": None, "created_at": datetime(2026, 11, 16, tzinfo=timezone.utc), **kw}


def test_grab_frame_seeks_and_falls_back_to_last(tmp_path):
    p = _mp4(tmp_path / "a.mp4", 4)
    early, late = grab_frame(p, 0.0), grab_frame(p, 2.0)
    assert early.shape == (180, 320, 3) and late.mean() > early.mean() + 50       # 第 2 秒比開頭亮
    assert grab_frame(p, 99.0).mean() >= late.mean()                               # 超過長度 → 最後一幀


def test_pick_offset():
    assert VT.pick_offset(_video()) == 1.0
    assert VT.pick_offset(_video(duration_s=0.8)) == pytest.approx(0.4)           # 太短 → 取中間
    assert VT.pick_offset(_video(params={"start_s": 30})) == 31.0                  # 有匯入起點就從起點算


def test_thumbnail_made_once_then_cached(tmp_path):
    _mp4(tmp_path / "20261018_000000.mp4", 3, size=(1280, 720))
    lib, calls = FakeLib(), []

    def nas_path(cam, file):
        calls.append((cam, file))
        return tmp_path / file

    data = VT.thumbnail(lib, _video(), 320, nas_path)
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    assert img.shape[:2] == (180, 320)                                             # 長邊 320、比例不變
    assert (VT.BUCKET, VT.thumb_key(_video(), 320)) in lib.objects.store
    assert VT.thumbnail(lib, _video(), 320, nas_path) == data and len(calls) == 1  # 第二次直接讀，不再開影片
    assert VT.thumb_key(_video(), 320).startswith(VT.thumb_prefix(7))


def test_thumbnail_missing_file_and_broken_file(tmp_path):
    def gone(cam, file):
        raise FileNotFoundError(file)

    with pytest.raises(VT.VideoUnavailable):
        VT.thumbnail(FakeLib(), _video(), 320, gone)
    with pytest.raises(VT.VideoUnavailable):                                       # 上傳的影片卻沒有原檔
        VT.thumbnail(FakeLib(), _video(kind="upload_video", uri="s3://raw/x.mp4"), 320, gone)
    (tmp_path / "bad.mp4").write_bytes(b"not a video")
    with pytest.raises(ValueError):
        VT.thumbnail(FakeLib(), _video(), 320, lambda c, f: tmp_path / "bad.mp4")


def test_thumbnail_other_storage_errors_propagate():
    lib = FakeLib()
    lib.objects.get = lambda **kw: (_ for _ in ()).throw(ConnectionError("minio down"))
    with pytest.raises(ConnectionError):
        VT.thumbnail(lib, _video(), 320, lambda c, f: None)


def test_upload_display_name_strips_upload_stamp():
    from edgevision.media.jobs import upload_display_name
    assert upload_display_name("s3://raw/uploads/media/20261116_093012_產線A 測試.mp4") == "產線A 測試.mp4"
    assert upload_display_name("s3://raw/uploads/media/clip.mp4") == "clip.mp4"            # 沒有時間前綴照留
    assert upload_display_name("s3://raw/uploads/media/20261116_093012_") == "20261116_093012_"
