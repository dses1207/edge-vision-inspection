"""影格標記的回歸測試。

標記是前端配對的唯一依據，任何改動都必須通過這些案例。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest

from edgevision.marker import decode_marker, encode_marker, MARKER_HEIGHT, MARKER_WIDTH
from edgevision.marker.frame_marker import FrameMarker, ID_MASK, stamp


def blank(h=720, w=1280):
    return np.zeros((h, w, 3), dtype=np.uint8)


def noisy(h=720, w=1280, seed=0):
    return np.random.default_rng(seed).integers(0, 255, (h, w, 3), dtype=np.uint8)


@pytest.mark.parametrize("fid", [0, 1, 42, 255, 1234, 65535, 999999, ID_MASK])
def test_roundtrip(fid):
    img = noisy()
    stamp(img, fid)
    assert decode_marker(img) == fid


def test_marker_size_is_small():
    """標記不該佔用太多畫面。"""
    ratio = MARKER_WIDTH * MARKER_HEIGHT / (1280 * 720)
    assert ratio < 0.01, f"標記佔畫面 {ratio:.1%}，過大"


def test_rejects_unmarked_frames():
    """沒有標記的畫面不可解出 id —— 誤讀比漏讀危險。"""
    assert decode_marker(noisy(seed=1)) is None
    assert decode_marker(blank()) is None
    assert decode_marker(np.full((720, 1280, 3), 128, np.uint8)) is None


def test_rejects_corrupted_marker():
    """標記被破壞時應回 None 而非錯誤的 id。"""
    from edgevision.marker.frame_marker import MARKER_WIDTH, MARKER_HEIGHT
    img = noisy()
    stamp(img, 12345)
    # marker 在右下角，翻轉它資料區的幾個位元（用尺寸常數定位，不寫死座標）
    ih, iw = img.shape[:2]
    oy, ox = ih - MARKER_HEIGHT, iw - MARKER_WIDTH
    img[oy + 4:oy + 10, ox + 60:ox + 130] = 255 - img[oy + 4:oy + 10, ox + 60:ox + 130]
    assert decode_marker(img) != 12345


def test_rejects_too_small_image():
    assert decode_marker(np.zeros((10, 10, 3), np.uint8)) is None


def test_id_wraps_at_mask():
    """id 超過位元寬度時應回繞，不可溢位。"""
    img = noisy()
    stamp(img, ID_MASK + 5)
    assert decode_marker(img) == 4


def test_sequential_ids_are_unique():
    """連號的 id 必須各自可辨識 —— 這是配對正確的前提。"""
    seen = set()
    for fid in range(1000, 1050):
        img = noisy(seed=fid)
        stamp(img, fid)
        got = decode_marker(img)
        assert got == fid
        seen.add(got)
    assert len(seen) == 50


def test_marker_generator():
    m = FrameMarker(start=100)
    img = blank()
    assert m.stamp(img) == 100
    assert decode_marker(img) == 100
    assert m.stamp(img) == 101
    assert decode_marker(img) == 101


def test_survives_jpeg_like_quantization():
    """模擬有損壓縮：加入雜訊後仍應可讀。"""
    import cv2
    img = noisy(seed=3)
    stamp(img, 54321)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 60])
    assert ok
    restored = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    assert decode_marker(restored) == 54321
