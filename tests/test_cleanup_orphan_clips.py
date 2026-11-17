"""孤兒片段清理的 key 比對測試。

⚠️ 這支腳本會【永久刪除】MinIO 的物件，而存證圖跟事件影片在同一個 bucket。
   認錯一個 key 就是刪掉存證。所以比對規則必須有測試守住。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from cleanup_orphan_clips import parse_key  # noqa: E402


@pytest.mark.parametrize("key", [
    "cam156/20260915_100000_a1b2c3d4.mp4",
    "cam156/20260915_100000_a1b2c3d4.json",          # side-car
    "alpha_a_cam118/20260101_235959_deadbeef.mp4",     # 相機 id 含底線
])
def test_recognises_clip_keys(key):
    assert parse_key(key) is not None


@pytest.mark.parametrize("key,why", [
    ("defects/2026/09/15/cam156/100000_f1_ng.jpg", "⭐ 存證圖 —— 刪掉就沒了"),
    ("defects/2026/09/15/cam156/100000_f1_ng.mp4", "defects 下的 mp4 也不能碰"),
    ("datasets/golden/seg-alpha/img001.jpg", "黃金測試集"),
    ("models/seg_alpha/v3/model.plan", "模型檔"),
    ("golden/x.jpg", "黃金集另一種擺法"),
])
def test_never_touches_protected_paths(key, why):
    assert parse_key(key) is None, why


@pytest.mark.parametrize("key,why", [
    ("cam156/randomfile.mp4", "沒有時間戳"),
    ("cam156/20260915_100000.mp4", "缺 uuid 後綴"),
    ("cam156/20260915_100000_a1b2c3d4.txt", "副檔名不是 mp4/json"),
    ("cam156/sub/20260915_100000_a1b2c3d4.mp4", "多一層目錄"),
    ("20260915_100000_a1b2c3d4.mp4", "沒有相機層"),
    ("cam156/20261399_995959_a1b2c3d4.mp4", "無效日期"),
    ("cam156/20260915_100000_ZZZZZZZZ.mp4", "uuid 不是 hex"),
])
def test_unknown_shapes_are_left_alone(key, why):
    """認不出來就不碰 —— 寧可留下垃圾，不可誤刪。"""
    assert parse_key(key) is None, why


def test_parsed_fields_are_usable():
    """--before 過濾靠解析出來的時間，格式要對。"""
    info = parse_key("cam156/20260915_143000_a1b2c3d4.mp4")
    assert info["cam"] == "cam156"
    assert info["ext"] == "mp4"
    assert info["dt"].strftime("%Y-%m-%d %H:%M:%S") == "2026-09-15 14:30:00"


def test_key_matches_what_clipmaker_actually_writes():
    """⚠️ 契約測試：比對規則必須跟 ClipMaker 產生的 key 一致。

    ClipMaker 改了命名而這裡沒跟上的話，清理會變成「什麼都認不出來」
    （安全但無效）或更糟。這裡從原始碼確認格式沒被改掉。
    """
    import inspect

    from edgevision.recorder.clip import ClipMaker
    src = inspect.getsource(ClipMaker)
    assert '{camera_id}/{dt:%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:8]}' in src, \
        "ClipMaker 的 key 格式變了 —— cleanup_orphan_clips.py 的 CLIP_KEY_RE 要一起改"
