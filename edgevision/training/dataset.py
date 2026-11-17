"""資料集來源：把「資料集版本 ID」變成本地可訓練的目錄 + data.yaml + manifest。

    DatasetSource 介面（可抽換）
      ├─ LocalDirSource   local://<路徑>          開發、P2 階段先跑通用
      └─ LakeFSSource     lakefs://<repo>/<版本id> 正式：不可變版本，可重現

資料集目錄格式 = 標註工具匯出的 YOLO 格式：
    images/{train,val}/*.jpg   labels/{train,val}/*.txt   classes.txt 或 data.yaml
    seg 標籤每行：<class> x1 y1 x2 y2 ... xn yn（正規化 polygon）
    若沒有 train/val 子目錄（只有 images/ labels/），依 split_ratio 切分（固定 seed，可重現）。

manifest.json：每張圖 sha256、類別分布、bbox 尺寸分布 —— run_report 的 dataset_stats
與黃金測試集零重疊檢查都靠它。
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import re
import shutil
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import yaml

logger = logging.getLogger(__name__)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


@dataclass(slots=True)
class Manifest:
    dataset_version: str
    class_names: list[str]
    images: list[dict] = field(default_factory=list)      # {path, split, sha256, n_labels}
    class_distribution: dict[str, int] = field(default_factory=dict)
    bbox_size_distribution: dict[str, float | int] = field(default_factory=dict)
    images_per_source: dict[str, int] = field(default_factory=dict)

    @property
    def n_images(self) -> int:
        return len(self.images)

    def count(self, split: str) -> int:
        return sum(1 for i in self.images if i["split"] == split)

    def hashes(self) -> set[str]:
        return {i["sha256"] for i in self.images}

    def to_dict(self) -> dict:
        return {
            "dataset_version": self.dataset_version, "class_names": self.class_names,
            "n_images": self.n_images, "n_train": self.count("train"), "n_val": self.count("val"),
            "class_distribution": self.class_distribution,
            "bbox_size_distribution": self.bbox_size_distribution,
            "images_per_source": self.images_per_source, "images": self.images,
        }


@dataclass(slots=True)
class PreparedDataset:
    version: str
    root: Path
    data_yaml: Path
    class_names: list[str]
    manifest: Manifest


class DatasetSource(Protocol):
    def prepare(self, target: Path) -> PreparedDataset: ...


# ── 共用：目錄整理、切分、manifest ─────────────────────────

def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_class_names(root: Path) -> list[str] | None:
    # data.yaml / dataset.yaml 優先，否則任何含 names 的 *.yaml（Ultralytics 公開資料集用自己的檔名）
    cands = [root / "data.yaml", root / "dataset.yaml"] + sorted(root.glob("*.yaml"))
    for cand in cands:
        if cand.exists():
            d = yaml.safe_load(cand.read_text()) or {}
            names = d.get("names")
            if isinstance(names, dict):
                return [names[k] for k in sorted(names, key=int)]
            if isinstance(names, list):
                return list(names)
    ct = root / "classes.txt"
    if ct.exists():
        return [l.strip() for l in ct.read_text().splitlines() if l.strip()]
    return None


def _label_for(img: Path, images_dir: Path, labels_dir: Path) -> Path:
    rel = img.relative_to(images_dir).with_suffix(".txt")
    return labels_dir / rel


def normalize_layout(root: Path, *, split_ratio: float = 0.2, seed: int = 0) -> None:
    """確保 images/{train,val} 與 labels/{train,val} 存在；平面結構就切分。"""
    images, labels = root / "images", root / "labels"
    if not images.exists():
        raise FileNotFoundError(f"資料集缺 images/：{root}")
    if (images / "train").exists() or (images / "val").exists():
        # 已切分（空目錄上傳到 LakeFS 不會保留，缺的那邊補回來）
        for d in (images / "train", images / "val", labels / "train", labels / "val"):
            d.mkdir(parents=True, exist_ok=True)
        return
    files = sorted(p for p in images.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if not files:
        raise ValueError(f"images/ 沒有影像：{root}")
    rng = random.Random(seed)
    rng.shuffle(files)
    n_val = max(1, int(len(files) * split_ratio)) if len(files) > 1 else 0
    for i, img in enumerate(files):
        split = "val" if i < n_val else "train"
        (images / split).mkdir(exist_ok=True)
        (labels / split).mkdir(parents=True, exist_ok=True)
        shutil.move(str(img), images / split / img.name)
        lbl = labels / img.with_suffix(".txt").name
        if lbl.exists():
            shutil.move(str(lbl), labels / split / lbl.name)
    logger.info("資料集平面結構已切分：train=%d val=%d（seed=%d）", len(files) - n_val, n_val, seed)


# 圖片檔頭（magic bytes）。副檔名可以亂取，檔頭騙不了人。
_IMAGE_MAGIC = (
    b"\xff\xd8\xff",            # JPEG
    b"\x89PNG\r\n\x1a\n",       # PNG
    b"BM",                        # BMP
    b"II*\x00", b"MM\x00*",       # TIFF
)


def _is_image(p: Path) -> bool:
    with p.open("rb") as fh:
        head = fh.read(12)
    return head.startswith(_IMAGE_MAGIC) or (head[:4] == b"RIFF" and head[8:12] == b"WEBP")


def validate_images(root: Path) -> list[str]:
    """圖片要真的是圖片（檢查檔頭）。回傳問題清單。

    ⚠️ 以前完全不驗：前端探測時把 19 bytes 的「404 page not found」當 jpg 塞進 zip，
       ingest 照樣回 done、n_images 算進去 —— 壞資料會靜默進到訓練集（2026-10-06 前端回報）。
    """
    problems: list[str] = []
    for split in ("train", "val"):
        d = root / "images" / split
        if not d.is_dir():
            continue
        for img in sorted(p for p in d.iterdir() if p.suffix.lower() in IMAGE_EXTS):
            if not _is_image(img):
                problems.append(f"{split}/{img.name} 不是有效的圖片檔（{img.stat().st_size} bytes）")
    return problems


def validate_labels(root: Path, n_classes: int) -> list[str]:
    """檢查標籤格式：類別索引範圍、polygon 點數（seg 至少 3 點）、座標正規化。回傳問題清單。"""
    problems: list[str] = []
    for split in ("train", "val"):
        for lbl in (root / "labels" / split).glob("*.txt"):
            for ln, line in enumerate(lbl.read_text().splitlines(), 1):
                parts = line.split()
                if not parts:
                    continue
                try:
                    cls = int(parts[0]); coords = [float(x) for x in parts[1:]]
                except ValueError:
                    problems.append(f"{lbl.name}:{ln} 非數字"); continue
                if not 0 <= cls < n_classes:
                    problems.append(f"{lbl.name}:{ln} 類別 {cls} 超出 0..{n_classes-1}")
                if len(coords) < 4 or len(coords) % 2:
                    problems.append(f"{lbl.name}:{ln} 座標數 {len(coords)} 不合法")
                if any(c < -0.001 or c > 1.001 for c in coords):
                    problems.append(f"{lbl.name}:{ln} 座標未正規化")
    return problems


def build_manifest(root: Path, version: str, class_names: list[str]) -> Manifest:
    m = Manifest(dataset_version=version, class_names=class_names)
    dist = {n: 0 for n in class_names}
    areas: list[float] = []
    for split in ("train", "val"):
        idir, ldir = root / "images" / split, root / "labels" / split
        if not idir.exists():
            continue
        for img in sorted(p for p in idir.iterdir() if p.suffix.lower() in IMAGE_EXTS):
            lbl = _label_for(img, idir, ldir)
            n = 0
            if lbl.exists():
                for line in lbl.read_text().splitlines():
                    parts = line.split()
                    if len(parts) < 5:
                        continue
                    n += 1
                    cls = int(parts[0])
                    if 0 <= cls < len(class_names):
                        dist[class_names[cls]] += 1
                    xs, ys = [float(v) for v in parts[1::2]], [float(v) for v in parts[2::2]]
                    if xs and ys:
                        areas.append((max(xs) - min(xs)) * (max(ys) - min(ys)))
            src = re.split(r"[_\-]", img.stem)[0] if img.stem else "unknown"
            m.images.append({"path": f"images/{split}/{img.name}", "split": split,
                             "sha256": sha256_file(img), "n_labels": n})
            m.images_per_source[src] = m.images_per_source.get(src, 0) + 1
    m.class_distribution = dist
    if areas:
        areas.sort()
        m.bbox_size_distribution = {
            "n": len(areas), "min": areas[0], "p50": areas[len(areas) // 2],
            "p90": areas[int(len(areas) * 0.9)], "max": areas[-1],
            "tiny_ratio": sum(1 for a in areas if a < 0.001) / len(areas),
        }
    return m


def write_data_yaml(root: Path, class_names: list[str]) -> Path:
    d = {"path": str(root), "train": "images/train", "val": "images/val",
         "names": {i: n for i, n in enumerate(class_names)}}
    p = root / "data.yaml"
    p.write_text(yaml.safe_dump(d, allow_unicode=True, sort_keys=False))
    return p


def finalize(root: Path, version: str, *, class_names: list[str] | None = None,
             split_ratio: float = 0.2, seed: int = 0) -> PreparedDataset:
    normalize_layout(root, split_ratio=split_ratio, seed=seed)
    names = class_names or _read_class_names(root)
    if not names:
        raise ValueError(f"無法得知類別名稱：{root} 缺 data.yaml / classes.txt，且設定未給 class_names")
    bad = validate_images(root)
    if bad:
        raise ValueError(f"有 {len(bad)} 個圖片檔不是有效圖片（前 5 筆）：" + "; ".join(bad[:5]))
    problems = validate_labels(root, len(names))
    if problems:
        raise ValueError("標籤格式有誤（前 5 筆）：" + "; ".join(problems[:5]))
    data_yaml = write_data_yaml(root, names)
    manifest = build_manifest(root, version, names)
    (root / "manifest.json").write_text(json.dumps(manifest.to_dict(), ensure_ascii=False, indent=2))
    logger.info("資料集就緒 %s：%d 張（train %d / val %d），類別 %s",
                version, manifest.n_images, manifest.count("train"), manifest.count("val"), names)
    return PreparedDataset(version=version, root=root, data_yaml=data_yaml,
                           class_names=names, manifest=manifest)


def expand_repeats(root: Path) -> int:
    """混合快照（edgevision/training/mix.py）的 train_repeats.json：{檔名: 次數}。訓練前把次數 > 1 的圖展開成多份
    （<名>__r2.jpg …，硬連結、標籤複製），Ultralytics 才會照次數取樣。快照本身每張只存一份。回傳多出幾張。"""
    f = root / "train_repeats.json"
    if not f.exists():
        return 0
    n = 0
    for name, k in json.loads(f.read_text()).items():
        img = root / "images" / "train" / name
        if not img.exists():
            raise ValueError(f"train_repeats.json 指到不存在的圖：{name}")
        lbl = root / "labels" / "train" / (img.stem + ".txt")
        for j in range(2, int(k) + 1):
            dst = img.with_name(f"{img.stem}__r{j}{img.suffix}")
            try:
                os.link(img, dst)
            except OSError:
                shutil.copyfile(img, dst)
            if lbl.exists():
                shutil.copyfile(lbl, lbl.with_name(f"{img.stem}__r{j}.txt"))
            n += 1
    f.unlink()
    logger.info("混合快照：依 train_repeats.json 展開 %d 張重複取樣", n)
    return n


def extract_zip(zip_path: Path, target: Path) -> Path:
    """解壓標註工具匯出的 zip；若 zip 內多包一層目錄，自動下探。"""
    target.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(target)
    root = target
    for _ in range(3):
        if (root / "images").exists():
            break
        subs = [p for p in root.iterdir() if p.is_dir()]
        if len(subs) == 1:
            root = subs[0]
    return root


# ── 來源實作 ──────────────────────────────────────────────

class LocalDirSource:
    """local://<路徑>：目錄或 zip。開發與 P2 階段用；版本 ID 就是路徑（不可重現，僅供跑通）。"""

    def __init__(self, uri: str, *, class_names: list[str] | None = None, seed: int = 0) -> None:
        self.path = Path(uri[len("local://"):] if uri.startswith("local://") else uri)
        self.class_names = class_names
        self.seed = seed

    def prepare(self, target: Path) -> PreparedDataset:
        if not self.path.exists():
            raise FileNotFoundError(f"資料集不存在：{self.path}")
        target.mkdir(parents=True, exist_ok=True)
        if self.path.is_file() and self.path.suffix == ".zip":
            root = extract_zip(self.path, target)
        else:
            root = target / "data"
            if root.exists():
                shutil.rmtree(root)
            shutil.copytree(self.path, root)
        expand_repeats(root)
        return finalize(root, f"local://{self.path}", class_names=self.class_names, seed=self.seed)


class LakeFSSource:
    """lakefs://<repo>/<版本id>[/<prefix>]：從不可變版本拉整個資料集。"""

    def __init__(self, uri: str, *, class_names: list[str] | None = None, seed: int = 0) -> None:
        if not uri.startswith("lakefs://"):
            raise ValueError(f"不是 lakefs URI：{uri}")
        rest = uri[len("lakefs://"):]
        parts = rest.split("/", 2)
        if len(parts) < 2:
            raise ValueError(f"格式應為 lakefs://<repo>/<版本id>：{uri}")
        self.repo, self.ref = parts[0], parts[1]
        self.prefix = parts[2].rstrip("/") + "/" if len(parts) > 2 and parts[2] else ""
        self.uri = f"lakefs://{self.repo}/{self.ref}" + (f"/{self.prefix.rstrip('/')}" if self.prefix else "")
        self.class_names = class_names
        self.seed = seed

    def prepare(self, target: Path) -> PreparedDataset:
        from edgevision.training.lakefs_client import lakefs_client, ref_objects, read_object

        client = lakefs_client()
        root = target / "data"
        if root.exists():
            shutil.rmtree(root)
        root.mkdir(parents=True)
        n = 0
        for obj in ref_objects(client, self.repo, self.ref, prefix=self.prefix):
            rel = obj.path[len(self.prefix):] if self.prefix else obj.path
            if not rel or rel.endswith("/"):
                continue
            dst = root / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(read_object(client, self.repo, self.ref, obj.path))
            n += 1
        if n == 0:
            raise ValueError(f"{self.uri} 沒有任何物件（版本 ID 或 prefix 錯誤？）")
        logger.info("LakeFS 已拉取 %d 個物件：%s", n, self.uri)
        expand_repeats(root)
        return finalize(root, self.uri, class_names=self.class_names, seed=self.seed)


def source_for(uri: str, **kw) -> DatasetSource:
    if uri.startswith("lakefs://"):
        return LakeFSSource(uri, **kw)
    if uri.startswith("local://") or os.path.exists(uri):
        return LocalDirSource(uri, **kw)
    raise ValueError(f"不支援的資料集來源：{uri}（lakefs:// 或 local://）")
