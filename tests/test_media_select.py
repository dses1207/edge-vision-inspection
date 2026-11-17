"""挑圖（edgevision/media/select.py）：不用模型的去重 / 變動 / 分群。用合成的「IPCAM」影片驗證。

合成影片（10 fps、120 秒）：
    左上角時間戳每秒變（會被預設 ignore_regions 遮掉）
    0–40s   空場景（完全不動）
    40–80s  一個「人」（方塊）在畫面中移動
    80–120s 空場景，但背景換成另一個樣子（例如換了料）
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

av = pytest.importorskip("av")
cv2 = pytest.importorskip("cv2")

from edgevision.media.frames import decode_at, decode_every, keyframe_interval  # noqa: E402
from edgevision.media.select import (  # noqa: E402
    Candidate, SelectParams, choose, extract, hamming_to_all, scan, validate_rect,
)

W, H, FPS, DUR = 640, 360, 10, 120


def _frame(t: float) -> np.ndarray:
    img = np.full((H, W, 3), (90, 110, 120), np.uint8)
    cv2.rectangle(img, (100, 150), (540, 330), (40, 160, 200), -1)          # 作業台
    if t >= 80:
        cv2.rectangle(img, (300, 180), (500, 300), (200, 60, 200), -1)      # 換了料
    if 40 <= t < 80:
        x = int(120 + (t - 40) * 9)
        cv2.rectangle(img, (x, 120), (x + 60, 300), (60, 200, 60), -1)      # 人
    cv2.putText(img, f"2026-10-14 10:{int(t) // 60:02d}:{int(t) % 60:02d}", (8, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)          # 時間戳（左上）
    return img


@pytest.fixture(scope="module")
def video(tmp_path_factory) -> Path:
    p = tmp_path_factory.mktemp("sel") / "ipcam.mp4"
    with av.open(str(p), "w") as c:
        st = c.add_stream("mpeg4", rate=FPS)
        st.width, st.height, st.pix_fmt = W, H, "yuv420p"
        st.codec_context.gop_size = FPS * 2           # 2 秒一個關鍵幀
        st.codec_context.bit_rate = 2_000_000
        for i in range(FPS * DUR):
            fr = av.VideoFrame.from_ndarray(_frame(i / FPS), format="bgr24")
            for pkt in st.encode(fr):
                c.mux(pkt)
        for pkt in st.encode():
            c.mux(pkt)
    return p


def test_decode_every_and_at(video):
    assert keyframe_interval(video) == pytest.approx(2.0, abs=0.15)
    ts = [t for t, _ in decode_every(video, interval_s=5.0)]
    # 間隔 5s ≥ GOP 2s → 只解關鍵幀，取最接近排程格點的那張（誤差 ≤ GOP/2），不累積漂移
    assert len(ts) == DUR // 5 and all(abs(t - 5 * i) <= 1.0 + 1e-6 for i, t in enumerate(ts))
    # 間隔 ≥ GOP → 只解關鍵幀
    ts2 = [t for t, _ in decode_every(video, interval_s=2.0, start_s=10, end_s=20)]
    assert ts2[0] == pytest.approx(10.0, abs=0.15) and len(ts2) in (5, 6)
    got = [(t, img.shape) for t, img in decode_at(video, [3.0, 3.5, 61.2, 100.0])]
    assert [t for t, _ in got] == [3.0, 3.5, 61.2, 100.0] and got[0][1] == (H, W, 3)


def test_timestamp_ignored_static_scene_dedups(video):
    p = SelectParams(target=50, dense_interval_s=1.0)
    cands, st = scan([(video, 0.0, 39.0)], p)
    # 0–39s 畫面除了時間戳完全一樣 → 遮掉時間戳後只剩 1 張
    assert st.sampled >= 39 and st.candidates == 1 and st.moving == 0


def test_without_ignoring_timestamp_static_scene_looks_moving(video):
    # 時間戳每秒變 → 不遮的話靜止畫面也有前景像素（這就是為什麼預設要遮）
    _, st_masked = scan([(video, 0.0, 39.0)], SelectParams(dedup_bits=0))
    cands, _ = scan([(video, 0.0, 39.0)], SelectParams(ignore_regions=[], dedup_bits=0))
    assert st_masked.moving == 0
    assert max(c.motion for c in cands) > 0


def test_full_pipeline_prefers_motion(video):
    p = SelectParams(target=12, dense_interval_s=1.0, still_ratio=0.25)
    cands, st = scan([(video, 0.0, None)], p)
    assert st.sampled >= DUR - 1
    assert st.candidates < st.sampled / 2                       # 大部分是重複
    moving_t = [c.offset_s for c in cands if c.moving]
    assert moving_t and all(39 <= t <= 82 for t in moving_t)    # 只有人在的時段算「有變動」
    picked = choose(cands, p)
    assert len(picked) <= 12
    assert sum(not c.moving for c in picked) >= 1                # 沒變動的也有名額（空場景）
    assert any(c.offset_s >= 80 for c in picked)                 # 換料後的場景有被挑到
    assert sum(c.group_size for c in picked if c.moving) == sum(c.moving for c in cands)
    assert [c.offset_s for c in picked] == sorted(c.offset_s for c in picked)
    out = list(extract(video, [c.offset_s for c in picked]))
    assert len(out) == len(picked) and out[0][1][:2] == b"\xff\xd8"


def test_roi_excludes_motion_outside(video):
    # 只看右下角（人不會走到）→ 沒有變動
    p = SelectParams(target=10, roi=[0.85, 0.85, 1.0, 1.0])
    cands, st = scan([(video, 30.0, 70.0)], p)
    assert st.moving == 0


def test_choose_fills_from_other_side():
    def c(i, moving):
        f = np.zeros(8, np.float32); f[i % 8] = 1 + i
        return Candidate(src=0, offset_s=float(i), motion=0.1 if moving else 0.0, feat=f, moving=moving)
    only_still = [c(i, False) for i in range(20)]
    assert len(choose(only_still, SelectParams(target=5, still_ratio=0.1))) == 5     # 沒有變動的就全用沒變動的
    mixed = [c(i, True) for i in range(3)] + [c(i + 3, False) for i in range(20)]
    picked = choose(mixed, SelectParams(target=10, still_ratio=0.1))
    assert sum(x.moving for x in picked) == 3 and len(picked) == 10                 # 有變動的不夠 → 補沒變動的


def test_hamming_and_rect_validation():
    a = np.zeros(32, np.uint8)
    kept = np.stack([a, np.full(32, 255, np.uint8)])
    assert hamming_to_all(a, kept).tolist() == [0, 256]
    assert validate_rect([0, 0, 0.5, 0.5], "roi") == [0.0, 0.0, 0.5, 0.5]
    for bad in ([0, 0, 1], [0.5, 0, 0.4, 1], [0, 0, 1.2, 1]):
        with pytest.raises(ValueError):
            validate_rect(bad, "roi")
