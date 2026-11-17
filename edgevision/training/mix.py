"""混合：幾個 LakeFS 快照＋配方 → 一個新的 LakeFS 快照。

    送進訓練的永遠是一個快照，圖片清單定案，不靠訓練當下抽樣 —— 重訓不用依賴亂數與程式版本一致。

配方（每個來源三選一，同 MosaicML Streaming）：
    repeat     倍數；<1 降採樣、>1 升採樣（1.5 = 每張 1 次，再抽一半的圖多 1 次）
    choose     固定張數（可大於來源張數 = 升採樣）
    proportion 相對比例，要另給 total（總張數）；同一份配方用了比例，就每個來源都要用比例
    都不給 = repeat 1（全部取一次）

規則：
    - 只作用在 train；各來源的 val 全部聯集、不抽樣。同一張圖（sha256）只要在任何來源是 val，就只放 val。
    - 同一張圖出現在多個來源 → 只算一次，取最大的次數。
    - 各來源類別清單必須完全一樣（含順序），不一致直接拒絕。
    - 圖片檔每張只存一份（檔名 = sha256 前 16 碼）；次數 > 1 記在 train_repeats.json，
      訓練端拉快照時才展開成多份（edgevision/training/dataset.py expand_repeats）。
"""

from __future__ import annotations

import json
import logging
import math
import random
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

REPEATS_FILE = "train_repeats.json"
MIX_FILE = "mix.json"


@dataclass
class SourceImages:
    ref: str
    train: dict[str, tuple[Path, Path | None]]     # sha256 → (圖, 標籤或 None)
    val: dict[str, tuple[Path, Path | None]]


def validate_recipe(sources: list[dict], total: int | None) -> None:
    """配方本身的檢查（API 收到時就擋，不用等任務跑）。"""
    if not sources:
        raise ValueError("至少要一個來源")
    refs = [s["ref"] for s in sources]
    if len(set(refs)) != len(refs):
        raise ValueError("來源重複")
    for s in sources:
        given = [k for k in ("repeat", "choose", "proportion") if s.get(k) is not None]
        if len(given) > 1:
            raise ValueError(f"{s['ref']}：repeat / choose / proportion 只能給一個")
        if s.get("repeat") is not None and s["repeat"] < 0:
            raise ValueError(f"{s['ref']}：repeat 不能是負數")
        if s.get("choose") is not None and s["choose"] < 0:
            raise ValueError(f"{s['ref']}：choose 不能是負數")
        if s.get("proportion") is not None and s["proportion"] < 0:
            raise ValueError(f"{s['ref']}：proportion 不能是負數")
    props = [s.get("proportion") is not None for s in sources]
    if any(props):
        if not all(props):
            raise ValueError("用了 proportion 就每個來源都要用 proportion（比例與倍數 / 張數不能混用）")
        if not total or total < 1:
            raise ValueError("用 proportion 要給 total（總張數）")
        if sum(s["proportion"] for s in sources) <= 0:
            raise ValueError("proportion 加總要大於 0")
    elif total is not None:
        raise ValueError("total 只在用 proportion 時有意義")


def _target(s: dict, n: int, sources: list[dict], total: int | None) -> int:
    if s.get("proportion") is not None:
        return _half_up(total * s["proportion"] / sum(x["proportion"] for x in sources))
    if s.get("choose") is not None:
        return int(s["choose"])
    return _half_up((1.0 if s.get("repeat") is None else float(s["repeat"])) * n)


def _half_up(x: float) -> int:
    """四捨五入（Python 的 round 是銀行家捨入，12.5 → 12，不直覺）。"""
    return int(math.floor(x + 0.5))


def plan(sources: list[dict], images: list[SourceImages], *, total: int | None, seed: int) -> dict:
    """純計算：每張 train 圖取幾次、val 有哪些、各來源的統計。"""
    val = set().union(*(set(im.val) for im in images))
    counts: dict[str, int] = {}
    stats = []
    for i, (s, im) in enumerate(zip(sources, images)):
        shas = sorted(sh for sh in im.train if sh not in val)
        want = _target(s, len(shas), sources, total) if shas else 0
        full, rem = divmod(want, len(shas)) if shas else (0, 0)
        rng = random.Random(f"{seed}:{i}")
        order = shas[:]
        rng.shuffle(order)
        mine = {sh: full + (1 if k < rem else 0) for k, sh in enumerate(order)}
        for sh, c in mine.items():
            if c > 0:
                counts[sh] = max(counts.get(sh, 0), c)
        stats.append({"ref": s["ref"], **{k: s[k] for k in ("repeat", "choose", "proportion") if s.get(k) is not None},
                      "n_train": len(shas), "n_val": len(im.val), "target": want,
                      "n_selected": sum(1 for c in mine.values() if c > 0)})
    return {"train": counts, "val": sorted(val), "sources": stats,
            "n_train_images": sum(counts.values()), "n_train_unique": len(counts), "n_val": len(val)}


def _collect(prepared) -> SourceImages:
    """拉下來的快照 → sha256 → (圖, 標籤)。來源若本身是混合快照，展開的重複檔會因 sha 相同自動合併。"""
    root = prepared.root
    out = {"train": {}, "val": {}}
    for rec in prepared.manifest.images:
        img = root / rec["path"]
        lbl = root / "labels" / rec["split"] / Path(rec["path"]).with_suffix(".txt").name
        out[rec["split"]].setdefault(rec["sha256"], (img, lbl if lbl.exists() else None))
    return SourceImages(ref=prepared.version, train=out["train"], val=out["val"])


def build(root: Path, images: list[SourceImages], p: dict) -> None:
    """依計畫寫出 YOLO 目錄（每張圖一份）＋ train_repeats.json。"""
    where: dict[str, tuple[Path, Path | None]] = {}
    for im in images:
        for d in (im.train, im.val):
            for sh, v in d.items():
                where.setdefault(sh, v)
    repeats = {}
    for split, shas in (("train", sorted(p["train"])), ("val", p["val"])):
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)
        for sh in shas:
            img, lbl = where[sh]
            name = sh[:16] + img.suffix.lower()
            shutil.copyfile(img, root / "images" / split / name)
            if lbl is not None:
                shutil.copyfile(lbl, root / "labels" / split / (sh[:16] + ".txt"))
            if split == "train" and p["train"][sh] > 1:
                repeats[name] = p["train"][sh]
    if repeats:
        (root / REPEATS_FILE).write_text(json.dumps(repeats, indent=0, sort_keys=True))


def mix(sources: list[dict], *, repo: str, message: str, seed: int = 0, total: int | None = None,
        work: Path | None = None) -> dict:
    """拉來源 → 檢查類別 → 計畫 → 寫目錄 → publish。回傳 publish 結果＋計畫統計。"""
    from edgevision.training.dataset import source_for
    from edgevision.training.ingest import publish

    validate_recipe(sources, total)
    tmp = Path(work or tempfile.mkdtemp(prefix="mix-"))
    try:
        prepared = [source_for(s["ref"], seed=seed).prepare(tmp / f"src{i}") for i, s in enumerate(sources)]
        names = prepared[0].class_names
        for s, pr in zip(sources, prepared):
            if pr.class_names != names:
                raise ValueError(f"類別不一致：{sources[0]['ref']} 是 {names}，{s['ref']} 是 {pr.class_names}")
        images = [_collect(pr) for pr in prepared]
        p = plan(sources, images, total=total, seed=seed)
        if not p["train"]:
            raise ValueError("混合後 train 是空的（配方把所有來源都取 0 張？）")
        out = tmp / "out"
        build(out, images, p)
        summary = {"seed": seed, "total": total, "sources": p["sources"], "n_train_images": p["n_train_images"],
                   "n_train_unique": p["n_train_unique"], "n_val": p["n_val"]}
        (out / MIX_FILE).write_text(json.dumps(summary, ensure_ascii=False, indent=1))
        (out / "classes.txt").write_text("\n".join(names) + "\n")
        r = publish(out, repo, message, class_names=names, split_ratio=0.2, source="mix",
                    extra_metadata={"mix": json.dumps(summary, ensure_ascii=False)})
        logger.info("混合 %d 個來源 → %s（train %d 次 / %d 張、val %d）", len(sources), r["dataset_ref"],
                    p["n_train_images"], p["n_train_unique"], p["n_val"])
        return {**r, "mix": summary}
    finally:
        if work is None:
            shutil.rmtree(tmp, ignore_errors=True)
