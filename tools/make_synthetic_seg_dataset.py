#!/usr/bin/env python3
"""產生合成的 YOLO-seg 資料集（board / hand 兩類），格式與標註工具匯出的 zip 相同。

用途：整體聯通性測試 —— 不需要真實資料就能把「資料 → 訓練 → Gate → 轉檔 → 註冊 → 部署」
整條鏈跑通。真實的 ALPHA dataset 到位後，只是把 zip 換掉。

    python tools/make_synthetic_seg_dataset.py --out /tmp/synth --n 60 --seed 1 [--zip]

輸出：images/*.jpg  labels/*.txt（正規化 polygon）  classes.txt
"""

from __future__ import annotations

import argparse
import math
import random
import shutil
import zipfile
from pathlib import Path

import cv2
import numpy as np

CLASSES = ["board", "hand"]


def _poly_rect(cx, cy, w, h, angle):
    c, s = math.cos(angle), math.sin(angle)
    pts = []
    for dx, dy in ((-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2), (-w / 2, h / 2)):
        pts.append((cx + dx * c - dy * s, cy + dx * s + dy * c))
    return pts


def _poly_ellipse(cx, cy, rx, ry, n=12):
    return [(cx + rx * math.cos(2 * math.pi * i / n), cy + ry * math.sin(2 * math.pi * i / n)) for i in range(n)]


def make_image(rng: random.Random, W: int, H: int):
    img = np.full((H, W, 3), rng.randint(60, 120), dtype=np.uint8)
    img += rng.randint(0, 20) * (np.random.RandomState(rng.randint(0, 9999)).rand(H, W, 3) > 0.5).astype(np.uint8)
    labels = []
    # board：一片綠色大板
    cx, cy = rng.uniform(0.35, 0.65) * W, rng.uniform(0.35, 0.65) * H
    bw, bh = rng.uniform(0.3, 0.5) * W, rng.uniform(0.2, 0.35) * H
    poly = _poly_rect(cx, cy, bw, bh, rng.uniform(-0.3, 0.3))
    cv2.fillPoly(img, [np.array(poly, dtype=np.int32)], (rng.randint(60, 90), rng.randint(140, 190), rng.randint(60, 90)))
    labels.append((0, poly))
    # hand：一到兩個膚色橢圓
    for _ in range(rng.randint(1, 2)):
        hx, hy = rng.uniform(0.15, 0.85) * W, rng.uniform(0.15, 0.85) * H
        rx, ry = rng.uniform(0.05, 0.09) * W, rng.uniform(0.07, 0.12) * H
        poly = _poly_ellipse(hx, hy, rx, ry)
        cv2.fillPoly(img, [np.array(poly, dtype=np.int32)], (rng.randint(120, 160), rng.randint(150, 190), rng.randint(200, 240)))
        labels.append((1, poly))
    img = cv2.GaussianBlur(img, (3, 3), 0)
    return img, labels


def write_dataset(out: Path, n: int, seed: int, W: int = 640, H: int = 480, prefix: str = "syn") -> Path:
    rng = random.Random(seed)
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(parents=True, exist_ok=True)
    for i in range(n):
        img, labels = make_image(rng, W, H)
        name = f"{prefix}{seed}_{i:04d}"
        cv2.imwrite(str(out / "images" / f"{name}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        lines = []
        for cls, poly in labels:
            coords = " ".join(f"{min(max(x / W, 0), 1):.5f} {min(max(y / H, 0), 1):.5f}" for x, y in poly)
            lines.append(f"{cls} {coords}")
        (out / "labels" / f"{name}.txt").write_text("\n".join(lines) + "\n")
    (out / "classes.txt").write_text("\n".join(CLASSES) + "\n")
    return out


def zip_dataset(out: Path, zip_path: Path) -> Path:
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(out.rglob("*")):
            if p.is_file():
                zf.write(p, p.relative_to(out).as_posix())
    return zip_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--zip", action="store_true", help="另外打包成 <out>.zip")
    ap.add_argument("--clean", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    if a.clean and out.exists():
        shutil.rmtree(out)
    write_dataset(out, a.n, a.seed)
    print(f"✅ 合成資料集 {out}：{a.n} 張，類別 {CLASSES}")
    if a.zip:
        z = zip_dataset(out, out.with_suffix(".zip"))
        print(f"✅ 已打包 {z}")


if __name__ == "__main__":
    main()
