"""審核的純邏輯：設定合併 / 驗證、優先分數、挑圖、框的差異（edgevision/media/review.py）。"""
import pytest

from edgevision.media.review import (
    deep_merge,
    load_defaults,
    mark_sources,
    pick_round,
    priority_of,
    prune_none,
    resolve_config,
    shape_diff,
    visible_shapes,
)

CLASSES = ["board", "sheet", "glove", "tap", "mat"]
SITEB12 = {"classes": {"tap": {"rare": True, "bulk_accept": False, "accept_conf": 0.4},
                    "glove": {"bulk_accept": False, "accept_conf": 0.4}},
        "swap_pairs": [["glove", "tap"]]}


def rect(cls, x1, y1, x2, y2, conf=None, sid=None):
    s = {"type": "rect", "class": cls, "points": [[x1, y1], [x2, y2]],
         "source": {"kind": "model", "conf": conf} if conf is not None else {"kind": "human"}}
    if sid:
        s["id"] = sid
    return s


# ── 設定 ──────────────────────────────────────────────

def test_defaults_file_is_complete_and_valid():
    d = load_defaults()
    assert d["round"]["size"] >= d["round"]["size_min"]
    assert set(d["priority"]["weights"]) <= {"rare", "low_conf", "empty", "diverse"}


def test_no_override_equals_defaults():
    d = load_defaults()
    cfg = resolve_config(CLASSES, {})
    assert cfg["round"] == d["round"] and cfg["review"] == d["review"]
    assert [c["name"] for c in cfg["classes"]] == CLASSES
    assert [c["hotkey"] for c in cfg["classes"]] == ["1", "2", "3", "4", "5"]
    assert all(c["show_conf"] == d["class_defaults"]["show_conf"] for c in cfg["classes"])


def test_class_override_only_touches_that_class():
    cfg = resolve_config(CLASSES, SITEB12)
    cc = {c["name"]: c for c in cfg["classes"]}
    assert cc["tap"]["bulk_accept"] is False and cc["tap"]["rare"] is True
    assert cc["mat"]["bulk_accept"] is True and cc["mat"]["rare"] is False
    assert cfg["swap_pairs"] == [["glove", "tap"]]


def test_override_out_of_range_rejected():
    with pytest.raises(ValueError):
        resolve_config(CLASSES, {"round": {"size": 100000}})          # 超過 size_max
    with pytest.raises(ValueError):
        resolve_config(CLASSES, {"review": {"lock_renew_s": 999}})    # 續期比鎖還久
    with pytest.raises(ValueError):
        resolve_config(CLASSES, {"classes": {"hand": {"rare": True}}})  # 不在圖片集的類別
    with pytest.raises(ValueError):
        resolve_config(CLASSES, {"round": {"sizee": 3}})              # 打錯欄位名不能默默忽略
    with pytest.raises(ValueError):
        resolve_config(CLASSES, {"swap_pairs": [["glove", "hand"]]})


def test_null_in_patch_resets_to_default():
    over = prune_none(deep_merge({"round": {"size": 300}, "thumb_px": 640}, {"round": {"size": None}}))
    assert over == {"thumb_px": 640}
    assert resolve_config(CLASSES, over)["round"]["size"] == load_defaults()["round"]["size"]


# ── 優先分數與挑圖 ─────────────────────────────────────

def test_rare_class_ranks_first():
    cfg = resolve_config(CLASSES, SITEB12)
    pad_only = {"id": 1, "shapes": [rect("mat", .6, .2, .8, .4, conf=.95)]}
    with_tap = {"id": 2, "shapes": [rect("mat", .6, .2, .8, .4, conf=.95), rect("tap", .7, .3, .75, .35, conf=.9)]}
    p1, r1 = priority_of(pad_only, cfg)
    p2, r2 = priority_of(with_tap, cfg)
    assert p2 > p1 and "rare" in r2 and "rare" not in r1


def test_low_conf_and_empty_signals():
    cfg = resolve_config(CLASSES, SITEB12)
    _, r = priority_of({"id": 1, "shapes": [rect("glove", .1, .1, .2, .2, conf=.25)]}, cfg)
    assert "low_conf" in r
    _, r = priority_of({"id": 2, "shapes": []}, cfg)
    assert r == ["empty"]
    # 低於 show_conf 的框當作沒有
    _, r = priority_of({"id": 3, "shapes": [rect("glove", .1, .1, .2, .2, conf=.05)]}, cfg)
    assert "empty" in r


def test_pick_round_caps_group_and_blind_ratio():
    cfg = resolve_config(CLASSES, {"round": {"max_per_group": 2}})
    items = [{"id": i, "shapes": [], "source_id": 1, "select_info": {"group": 0 if i < 10 else i, "group_size": 5}}
             for i in range(40)]
    picked = pick_round(items, cfg, size=20, blind_ratio=0.1, seed=7)
    assert len(picked) == 20
    assert sum(1 for p in picked if p["item"]["select_info"]["group"] == 0) == 2
    assert sum(p["blind"] for p in picked) == 2
    assert [p["rank"] for p in picked] == list(range(20))
    # 同 seed 抽到同一批抽查圖（可重現）
    again = pick_round(items, cfg, size=20, blind_ratio=0.1, seed=7)
    assert [p["blind"] for p in picked] == [p["blind"] for p in again]


def test_pick_round_fills_from_overflow_when_groups_exhausted():
    cfg = resolve_config(CLASSES, {"round": {"max_per_group": 1, "size_min": 1}})
    items = [{"id": i, "shapes": [], "source_id": 1, "select_info": {"group": 0}} for i in range(5)]
    assert len(pick_round(items, cfg, size=3, blind_ratio=0, seed=1)) == 3


def test_visible_shapes_hides_below_show_conf_keeps_human():
    cfg = resolve_config(CLASSES, {})
    shapes = [rect("mat", 0, 0, .1, .1, conf=.05), rect("mat", 0, 0, .1, .1, conf=.5), rect("mat", 0, 0, .1, .1)]
    assert len(visible_shapes(shapes, cfg)) == 2


# ── 差異 ──────────────────────────────────────────────

def test_shape_diff_counts():
    base = [rect("tap", .1, .1, .2, .2, .9, "a"), rect("mat", .5, .5, .9, .9, .95, "b"),
            rect("glove", .3, .3, .4, .4, .3, "c")]
    sub = [rect("tap", .1, .1, .2, .2, .9, "a"),             # 原樣
           rect("mat", .5, .5, .9, .92, .95, "b"),             # 拖了邊
           rect("glove", .6, .6, .7, .7, None, "new1")]        # 新增；c 被刪
    d = shape_diff(base, sub)
    assert (d["accepted"], d["modified"], d["added"], d["deleted"]) == (1, 1, 1, 1)
    assert d["per_class"]["glove"]["added"] == 1 and d["per_class"]["glove"]["deleted"] == 1


def test_class_change_counts_as_modified():
    base = [rect("glove", .1, .1, .2, .2, .5, "a")]
    d = shape_diff(base, [rect("tap", .1, .1, .2, .2, .5, "a")])
    assert d["modified"] == 1 and d["class_changed"] == 1 and d["per_class"]["tap"]["class_changed"] == 1


def test_mark_sources_keeps_model_only_for_untouched():
    base = [rect("mat", .5, .5, .9, .9, .95, "b"), rect("tap", .1, .1, .2, .2, .9, "a")]
    out = mark_sources(base, [rect("mat", .5, .5, .9, .9, .95, "b"), rect("tap", .1, .1, .25, .2, .9, "a")])
    assert out[0]["source"]["kind"] == "model" and out[1]["source"] == {"kind": "human"}
