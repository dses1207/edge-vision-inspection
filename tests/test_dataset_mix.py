"""混合（edgevision/training/mix.py）：配方檢查、抽樣計畫、寫出快照、訓練端展開重複取樣。"""
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from edgevision.training import mix as M
from edgevision.training.dataset import expand_repeats, source_for

CLASSES = ["board", "mat"]


def _img(path: Path, seed: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.random.default_rng(seed).integers(0, 255, (24, 32, 3), dtype=np.uint8))


def _ds(root: Path, train: list[int], val: list[int], names=CLASSES) -> str:
    for split, seeds in (("train", train), ("val", val)):
        for s in seeds:
            _img(root / "images" / split / f"i{s}.jpg", s)
            (root / "labels" / split).mkdir(parents=True, exist_ok=True)
            (root / "labels" / split / f"i{s}.txt").write_text("1 0.5 0.5 0.2 0.2\n")
    (root / "classes.txt").write_text("\n".join(names) + "\n")
    return f"local://{root}"


def test_validate_recipe():
    ok = [{"ref": "a"}, {"ref": "b", "repeat": 0.5}, {"ref": "c", "choose": 10}]
    M.validate_recipe(ok, None)
    for bad, total in [([{"ref": "a", "repeat": 1, "choose": 2}], None),
                       ([{"ref": "a", "proportion": 1}, {"ref": "b", "repeat": 1}], 10),
                       ([{"ref": "a", "proportion": 1}], None),
                       ([{"ref": "a"}], 10),
                       ([{"ref": "a"}, {"ref": "a"}], None),
                       ([], None)]:
        with pytest.raises(ValueError):
            M.validate_recipe(bad, total)


def _imgs(ref, train, val=()):
    return M.SourceImages(ref=ref, train={s: (Path(f"{s}.jpg"), None) for s in train},
                          val={s: (Path(f"{s}.jpg"), None) for s in val})


def test_plan_counts_val_union_and_dedup():
    a = _imgs("a", [f"a{i}" for i in range(10)], ["v1"])
    b = _imgs("b", [f"b{i}" for i in range(4)] + ["a0", "v1"])       # a0 兩邊都有；v1 在 a 是 val
    src = [{"ref": "a", "repeat": 0.5}, {"ref": "b", "repeat": 2.5}]
    p = M.plan(src, [a, b], total=None, seed=0)
    assert p["val"] == ["v1"] and "v1" not in p["train"]               # val 優先、不進 train
    sa, sb = p["sources"]
    assert (sa["n_train"], sa["target"], sa["n_selected"]) == (10, 5, 5)
    assert (sb["n_train"], sb["target"]) == (5, 13)                   # b 去掉 v1 剩 5 張 → 2.5 倍 = 12.5 → 13
    assert all(c in (2, 3) for k, c in p["train"].items() if k.startswith("b"))
    assert p["train"]["a0"] >= 2                                       # 兩個來源都有 → 取大的
    assert p == M.plan(src, [a, b], total=None, seed=0)                # 同 seed 同結果


def test_plan_choose_and_proportion():
    a, b = _imgs("a", [f"a{i}" for i in range(10)]), _imgs("b", [f"b{i}" for i in range(10)])
    p = M.plan([{"ref": "a", "choose": 3}, {"ref": "b", "choose": 25}], [a, b], total=None, seed=0)
    assert sum(c for k, c in p["train"].items() if k.startswith("a")) == 3
    assert sum(c for k, c in p["train"].items() if k.startswith("b")) == 25
    p = M.plan([{"ref": "a", "proportion": 1}, {"ref": "b", "proportion": 3}], [a, b], total=40, seed=0)
    assert [s["target"] for s in p["sources"]] == [10, 30]


def test_mix_end_to_end_and_expand(tmp_path, monkeypatch):
    r1 = _ds(tmp_path / "s1", train=[1, 2, 3, 4], val=[9])
    r2 = _ds(tmp_path / "s2", train=[5, 6], val=[])
    seen = {}

    def fake_publish(root, repo, message, **kw):
        import shutil
        shutil.copytree(root, tmp_path / "published")
        seen.update(kw)
        return {"dataset_ref": f"lakefs://{repo}/abc", "n_images": 0}
    monkeypatch.setattr("edgevision.training.ingest.publish", fake_publish)
    out = M.mix([{"ref": r1, "repeat": 0.5}, {"ref": r2, "repeat": 3}], repo="mix-t", message="m", seed=0)
    assert out["dataset_ref"] == "lakefs://mix-t/abc"
    meta = json.loads(seen["extra_metadata"]["mix"])
    assert meta["n_train_unique"] == 4 and meta["n_train_images"] == 8 and meta["n_val"] == 1
    pub = tmp_path / "published"
    assert len(list((pub / "images" / "train").iterdir())) == 4         # 每張只存一份
    rep = json.loads((pub / M.REPEATS_FILE).read_text())
    assert sorted(rep.values()) == [3, 3]

    # 訓練端拉這個快照：展開成 8 張 train，標籤跟著複製，sha 不變（測試集重疊檢查照用）
    prepared = source_for(f"local://{pub}").prepare(tmp_path / "train")
    assert prepared.manifest.count("train") == 8 and prepared.manifest.count("val") == 1
    assert len({i["sha256"] for i in prepared.manifest.images if i["split"] == "train"}) == 4
    assert not (prepared.root / M.REPEATS_FILE).exists()


def test_mix_rejects_class_mismatch(tmp_path):
    r1 = _ds(tmp_path / "s1", train=[1], val=[])
    r2 = _ds(tmp_path / "s2", train=[2], val=[], names=["mat", "board"])
    with pytest.raises(ValueError, match="類別不一致"):
        M.mix([{"ref": r1}, {"ref": r2}], repo="x", message="m")


def test_expand_repeats_missing_file(tmp_path):
    (tmp_path / "images" / "train").mkdir(parents=True)
    (tmp_path / M.REPEATS_FILE).write_text(json.dumps({"nope.jpg": 2}))
    with pytest.raises(ValueError):
        expand_repeats(tmp_path)
    assert expand_repeats(tmp_path / "none") == 0


def test_repo_name_follows_lakefs_rule():
    """LakeFS 只收 ^[a-z0-9][a-z0-9-]{2,62}$：送出時就擋（正式 siteb_golf_mix_test 曾跑到一半才失敗）。"""
    from edgevision.media.campaign import resolve_config
    from edgevision.training.lakefs_client import repo_name_error

    assert repo_name_error("siteb-golf-mix-test") is None
    for bad in ("siteb_golf_mix_test", "B-mix", "ab", "-siteb", "a" * 64):
        assert repo_name_error(bad)
    resolve_config({"dataset": {"repo": "siteb-bravo"}})
    with pytest.raises(ValueError):
        resolve_config({"dataset": {"repo": "siteb_bravo"}})        # 專案的 repo 前綴同一套規則
