"""場景偵測器：把場景真值包成「YOLO detect 模型的原始輸出」，讓後處理照真的跑。

輸出跟 Triton 上的 detect engine 一樣：(4 + 類別數, N) 的 float32，
前 4 列是 letterbox 座標系的 cx, cy, w, h，後面每列是一個類別的分數。
worker 拿到後走同一套 edgevision_postprocess（解碼、NMS、還原 letterbox），站別看到的 Detection 跟正式環境同形狀。

刻意加一點雜訊（抖動、漏檢、信心浮動）：判定層的去抖、滑動視窗、冷卻要有東西可擋，demo 才看得出它們在做事。
"""

from __future__ import annotations

import hashlib

import numpy as np
from edgevision_contracts import LetterboxParams

from cpu_runner.scene import Obj

N_ANCHORS = 64


class SceneDetector:
    def __init__(self, class_names: list[str], *, seed: str = "", miss_rate: float = 0.04,
                 jitter_px: float = 3.0) -> None:
        self.class_names = list(class_names)
        self.index = {c: i for i, c in enumerate(self.class_names)}
        self.miss_rate, self.jitter_px = miss_rate, jitter_px
        self.rng = np.random.default_rng(int(hashlib.sha1(seed.encode()).hexdigest()[:8], 16))

    def __call__(self, objects: list[Obj], lb: LetterboxParams) -> np.ndarray:
        out = np.zeros((4 + len(self.class_names), N_ANCHORS), np.float32)
        k = 0
        for o in objects:
            ci = self.index.get(o.class_name)
            if ci is None or k >= N_ANCHORS:
                continue                                   # 這顆模型沒有這個類別
            if self.rng.random() < self.miss_rate:
                continue                                   # 漏檢
            x1, y1, x2, y2 = (v + self.rng.normal(0, self.jitter_px) for v in o.box)
            x1, x2 = x1 * lb.scale + lb.pad_x, x2 * lb.scale + lb.pad_x
            y1, y2 = y1 * lb.scale + lb.pad_y, y2 * lb.scale + lb.pad_y
            out[:4, k] = ((x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1)
            out[4 + ci, k] = float(np.clip(self.rng.normal(0.85, 0.06), 0.5, 0.99))
            k += 1
        return out
