"""資料集整理：切分、標籤驗證、manifest、zip（用合成資料，不需 GPU / 服務）。"""
import sys
import zipfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from edgevision.training.dataset import LocalDirSource, extract_zip, finalize, source_for, validate_labels  # noqa: E402
from tools.make_synthetic_seg_dataset import write_dataset, zip_dataset  # noqa: E402


def test_flat_dataset_is_split_and_manifested(tmp_path):
    src = write_dataset(tmp_path / "src", 10, seed=3)
    ds = LocalDirSource(f"local://{src}").prepare(tmp_path / "work")
    assert ds.class_names == ["board", "hand"]
    m = ds.manifest
    assert m.n_images == 10 and m.count("val") == 2 and m.count("train") == 8
    assert m.class_distribution["board"] == 10 and m.class_distribution["hand"] >= 10
    assert (ds.root / "manifest.json").exists() and ds.data_yaml.exists()
    assert len(m.hashes()) == 10
    assert m.bbox_size_distribution["n"] == sum(m.class_distribution.values())


def test_split_is_deterministic(tmp_path):
    src = write_dataset(tmp_path / "src", 12, seed=5)
    a = LocalDirSource(f"local://{src}", seed=0).prepare(tmp_path / "a")
    b = LocalDirSource(f"local://{src}", seed=0).prepare(tmp_path / "b")
    assert [i["path"] for i in a.manifest.images] == [i["path"] for i in b.manifest.images]


def test_zip_source(tmp_path):
    src = write_dataset(tmp_path / "src", 6, seed=1)
    z = zip_dataset(src, tmp_path / "ds.zip")
    ds = source_for(f"local://{z}").prepare(tmp_path / "work")
    assert ds.manifest.n_images == 6


def test_zip_nested_dir(tmp_path):
    src = write_dataset(tmp_path / "src" / "export-2026", 4, seed=1)
    z = tmp_path / "nested.zip"
    with zipfile.ZipFile(z, "w") as zf:
        for p in src.rglob("*"):
            if p.is_file():
                zf.write(p, f"export-2026/{p.relative_to(src).as_posix()}")
    root = extract_zip(z, tmp_path / "x")
    assert (root / "images").exists()


def test_bad_labels_rejected(tmp_path):
    src = write_dataset(tmp_path / "src", 3, seed=1)
    bad = next((src / "labels").glob("*.txt"))
    bad.write_text("7 0.1 0.1 0.2 0.2 0.3 0.3\n")   # 類別超出範圍
    with pytest.raises(ValueError):
        LocalDirSource(f"local://{src}").prepare(tmp_path / "work")


def test_validate_labels_reports_problems(tmp_path):
    root = tmp_path
    (root / "labels" / "train").mkdir(parents=True)
    (root / "labels" / "val").mkdir(parents=True)
    (root / "labels" / "train" / "a.txt").write_text("0 0.1 0.1 0.5\n1 1.5 0.2 0.3 0.4 0.5 0.6\n")
    probs = validate_labels(root, 2)
    assert any("座標數" in p for p in probs) and any("正規化" in p for p in probs)


def test_missing_class_names(tmp_path):
    src = write_dataset(tmp_path / "src", 3, seed=1)
    (src / "classes.txt").unlink()
    with pytest.raises(ValueError):
        finalize(src, "v")
    assert finalize(src, "v", class_names=["board", "hand"]).class_names == ["board", "hand"]


def test_unknown_source():
    with pytest.raises(ValueError):
        source_for("ftp://nope")


def test_fake_image_rejected(tmp_path):
    """副檔名是 .jpg、內容是「404 page not found」的檔案不能進訓練集（2026-10-06 前端回報）。"""
    src = write_dataset(tmp_path / "src", 3, seed=1)
    img = next(p for p in (src / "images").iterdir() if p.suffix.lower() in (".jpg", ".png"))
    img.write_bytes(b"404 page not found\n")
    with pytest.raises(ValueError, match="不是有效圖片"):
        LocalDirSource(f"local://{src}").prepare(tmp_path / "work")
