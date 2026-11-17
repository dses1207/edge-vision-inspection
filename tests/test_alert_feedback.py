"""告警回流：對幀（locate）、框 → 標注、放進圖片集的狀態。"""
import cv2
import numpy as np

from edgevision.media import alerts as A

FPS = 10
W, H = 320, 180


def _scene(t: float) -> np.ndarray:
    """會動的畫面：底圖固定紋理，一塊方塊隨時間移動（像手）。"""
    rng = np.random.default_rng(0)
    img = (rng.integers(0, 255, (H, W, 3))).astype(np.uint8)
    img = cv2.GaussianBlur(img, (0, 0), 3)
    x = round(10 + 9 * t)                                # t ≤ 30 不會出畫面、不重複（重複的畫面本來就對不出時間）
    cv2.rectangle(img, (x, 60), (x + 40, 100), (255, 255, 255), -1)
    return img


def _box(t: float) -> dict:
    x = round(10 + 9 * t)
    return {"class": "glove", "conf": 0.9, "x1": x, "y1": 60, "x2": x + 40, "y2": 100}


def _frames(t0: float, t1: float, *, overlay_at: float | None = None, clock: float = 0.0):
    """t0..t1 的每一幀（灰階小圖）；overlay_at：證據影片 —— 每幀都畫當下的框（線上 runner 每幀都畫）。"""
    out = []
    n = int(round((t1 - t0) * FPS))
    for k in range(n):
        t = round(t0 + k / FPS, 3)
        img = _scene(t)
        if overlay_at is not None:
            b = _box(t)
            cv2.rectangle(img, (int(b["x1"]), int(b["y1"])), (int(b["x2"]), int(b["y2"])), (0, 255, 255), 2)
        out.append((round(t - t0 + clock, 3), A._small(img)))
    return out


def test_locate_finds_the_ng_frame():
    cfg = A.defaults()
    ng_t = 20.0                                          # 告警那一幀的「真實」時間
    nas = [(t + 1_000_000.0, g) for t, g in _frames(5.0, 30.0)]          # NAS：畫面第 5 秒 = epoch 1e6
    clip = _frames(ng_t - 8.0, ng_t + 5.0, overlay_at=ng_t)              # 證據影片：從告警前 8 秒起
    loc = A.locate(clip, nas, [_box(ng_t)], {"width": W, "height": H}, cfg, ts=1_000_000.0 + ng_t - 5.0 + 0.6)
    assert loc["ok"], loc
    assert abs(loc["nas_t"] - (1_000_000.0 + ng_t - 5.0)) < 0.05      # 對到同一幀（10fps，差不到半幀）
    assert loc["lag_agree"] > 0.9 and loc["score"] >= cfg["align"]["box_min_score"]
    # 證據影片沒拍到告警那一刻（影片太短 / 預錄不夠）：不能硬挑
    short = _frames(ng_t - 8.0, ng_t - 3.0, overlay_at=ng_t)
    r = A.locate(short, nas, [_box(ng_t)], {"width": W, "height": H}, cfg, ts=1_000_000.0 + ng_t - 5.0)
    assert not r["ok"] and "沒拍到" in r["why"]


def test_locate_reports_why_when_it_cannot():
    cfg = A.defaults()
    nas = [(t + 1e6, g) for t, g in _frames(5.0, 30.0)]
    assert A.locate([], nas, [_box(20)], {"width": W, "height": H}, cfg)["why"] == "沒有證據影片"
    clip = _frames(12.0, 25.0, overlay_at=20.0)
    assert not A.locate(clip, [], [_box(20)], {"width": W, "height": H}, cfg)["ok"]
    r = A.locate(clip, nas, [], None, cfg)                          # 整幀判定的 NG：沒框就不知道是哪一幀
    assert not r["ok"] and "沒有框" in r["why"]
    other = [(t + 1e6, A._small(np.full((H, W, 3), (t * 37) % 255, np.uint8))) for t, _ in _frames(5.0, 30.0)]
    assert not A.locate(clip, other, [_box(20)], {"width": W, "height": H}, cfg)["ok"]   # 別台相機的錄影


def test_boxes_to_shapes_and_status():
    cfg = A.defaults()
    boxes = [{"class": "sheet", "conf": 0.9, "x1": 192, "y1": 108, "x2": 960, "y2": 540},
             {"class": "hand", "conf": 0.5, "x1": 0, "y1": 0, "x2": 10, "y2": 10}]
    shapes, dropped = A.boxes_to_shapes(boxes, {"width": 1920, "height": 1080}, model="siteb_bravo", version="2",
                                        class_names=["board", "sheet"])
    assert dropped == ["hand"] and len(shapes) == 1
    s = shapes[0]
    assert s["points"] == [[0.1, 0.1], [0.5, 0.5]] and s["source"] == {
        "kind": "model", "engine": "online", "model": "siteb_bravo", "version": 2, "conf": 0.9}
    st = A.item_status
    assert st("true_positive", aligned=True, shapes=shapes, dropped=[], task="detect", cfg=cfg) == "labeled"
    assert st("false_positive", aligned=True, shapes=shapes, dropped=[], task="detect",
              cfg=cfg) == cfg["feedback"]["status"]["false_positive"]
    assert st("true_positive", aligned=False, shapes=shapes, dropped=[], task="detect", cfg=cfg) == "auto"
    assert st("true_positive", aligned=True, shapes=shapes, dropped=["hand"], task="detect", cfg=cfg) == "auto"
    assert st("true_positive", aligned=True, shapes=[], dropped=[], task="detect", cfg=cfg) is None
    assert st("true_positive", aligned=True, shapes=shapes, dropped=[], task="segment", cfg=cfg) is None
