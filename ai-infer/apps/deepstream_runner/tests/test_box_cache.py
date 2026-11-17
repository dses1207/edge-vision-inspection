from ds_runner.probe import BoxCache


def test_skipped_frames_reuse_last_boxes_until_stale():
    c = BoxCache(max_stale=3)
    assert c.get(0) == []                      # 還沒推論過
    c.update(0, ["a"])
    assert c.get(0) == ["a"] and c.get(0) == ["a"] and c.get(0) == ["a"]   # 3 幀內沿用
    assert c.get(0) == []                      # 第 4 幀太舊，不畫（來源斷線別留殘影）
    c.update(0, ["b"])
    assert c.get(0) == ["b"]


def test_cache_is_per_source():
    c = BoxCache()
    c.update(0, ["a"]); c.update(1, ["b"])
    assert c.get(0) == ["a"] and c.get(1) == ["b"] and c.get(2) == []


def test_hidden_classes_are_not_drawn():
    """hidden_classes 只從畫框濾掉；沒設定時原樣回傳。"""
    from types import SimpleNamespace as NS

    from ds_runner.probe import DrawItem, visible
    items = [DrawItem(NS(class_name=n)) for n in ("glove", "mat", "glove")]
    assert [it.det.class_name for it in visible(items, frozenset({"glove"}))] == ["mat"]
    assert visible(items, frozenset()) is items and visible(items, None) is items
