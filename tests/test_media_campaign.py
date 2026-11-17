"""標注專案的純邏輯：設定、配方（edgevision/media/campaign.py）。"""
import pytest

from edgevision.media.campaign import load_defaults, merge_recipe, recipe_sources, resolve_config, transient_db_error


def test_defaults_valid_and_override_validated():
    d = load_defaults()
    assert d["prelabel"]["conf"] == 0.10 and d["train"]["epochs"] == 150
    assert resolve_config({"prelabel": {"conf": 0.25}})["prelabel"]["conf"] == 0.25
    for gone in ({"transitions": {"exporting": "confirm"}}, {"regression": {"tolerance": 0.02}},
                 {"round": {"complete_ratio": 1}}, {"approvers": ["x"]}, {"seed": {"size": 50}}):
        with pytest.raises(ValueError):
            resolve_config(gone)                                       # v0.3 / v0.2 拿掉的欄位
    with pytest.raises(ValueError):
        resolve_config({"prelabel": {"conff": 0.2}})                   # 欄位打錯
    with pytest.raises(ValueError):
        resolve_config({"prelabel": {"conf": 0}})                      # 要 > 0


def test_recipe_merge_and_sources():
    r = merge_recipe(None, {"batches": {"4": {"repeat": 0.5}, 7: {"choose": 300}}, "seed": 1})
    assert r == {"batches": {"4": {"repeat": 0.5}, "7": {"choose": 300}}, "seed": 1}
    r = merge_recipe(r, {"batches": {"4": None, "7": {"choose": None, "repeat": 2}}, "seed": None})
    assert r == {"batches": {"7": {"repeat": 2}}}
    for bad in ({"batches": {"4": {"repeat": 1, "choose": 2}}}, {"batches": {"4": {"weight": 1}}},
                {"batches": {"4": {"repeat": -1}}}, {"ratio": 1}):
        with pytest.raises(ValueError):
            merge_recipe(None, bad)
    snaps = {"7": {"ref": "lakefs://b7/x"}, "4": {"ref": "lakefs://b4/y"}}
    assert recipe_sources(snaps, r) == [{"ref": "lakefs://b4/y", "batch_id": 4},
                                        {"ref": "lakefs://b7/x", "batch_id": 7, "repeat": 2}]


def test_transient_db_error():
    import psycopg
    assert transient_db_error(psycopg.errors.FeatureNotSupported("cached plan must not change result type"))
    assert transient_db_error(psycopg.OperationalError("server closed the connection"))
    assert not transient_db_error(ValueError("x")) and not transient_db_error(psycopg.errors.UndefinedColumn("c"))
