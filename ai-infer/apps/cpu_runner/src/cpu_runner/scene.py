"""合成場景：沒有相機、沒有模型也能把整條鏈路跑起來。

每台相機一個 Scene，依時間 t（秒）決定畫面上有什麼（Actor），兩件事都從同一份狀態算：
  ・render(t)   → 畫面（BGR）：送進直播、存證錄影
  ・objects(t)  → 真值框：交給 SceneDetector 假裝成模型輸出
畫面與真值同源，所以「框對不對得上畫面」在 demo 裡永遠成立 —— 要驗的是後面的判定、事件、錄影、審核。

劇本是決定性的（seed 由相機編號算），每次啟動看到的事件時間點一樣，方便寫文件、截圖、比對。
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

import cv2
import numpy as np

Box = tuple[float, float, float, float]


@dataclass(frozen=True)
class Obj:
    """一個真值物件：類別名 + 原圖像素框。"""
    class_name: str
    box: Box


@dataclass
class Walker:
    """沿著折線來回走的人。helmet / vest 決定 PPE 類別；period 秒走完一趟。"""
    path: list[tuple[float, float]]      # 正規化座標（腳點）
    period: float
    phase: float = 0.0
    helmet: bool = True
    vest: bool = False
    height: float = 0.32                 # 身高（佔畫面高的比例）

    def foot(self, t: float) -> tuple[float, float]:
        u = ((t + self.phase) % self.period) / self.period
        u = 1 - abs(2 * u - 1)            # 0→1→0 來回
        seg = u * (len(self.path) - 1)
        i = min(int(seg), len(self.path) - 2)
        f = seg - i
        (x0, y0), (x1, y1) = self.path[i], self.path[i + 1]
        return x0 + (x1 - x0) * f, y0 + (y1 - y0) * f


@dataclass
class Visit:
    """週期性出現的東西（堆高機、引導員）：每 every 秒出現一次，停留 stay 秒，從 start 秒後開始。"""
    class_name: str
    every: float
    start: float
    stay: float
    at: tuple[float, float, float, float]   # 正規化框
    skip_every: int = 0                     # 每 N 次缺席一次（製造 NG）；0 = 從不缺席

    def present(self, t: float) -> bool:
        if t < self.start:
            return False
        k, r = divmod(t - self.start, self.every)
        if self.skip_every and int(k) % self.skip_every == self.skip_every - 1:
            return False
        return r < self.stay


class Scene:
    def __init__(self, camera_id: str, width: int, height: int, *, kind: str = "zone",
                 zone: list[list[float]] | None = None) -> None:
        self.camera_id, self.w, self.h, self.kind = camera_id, width, height, kind
        self.zone = zone or [[0.62, 0.45], [0.95, 0.45], [0.95, 0.95], [0.62, 0.95]]
        seed = int(hashlib.sha1(camera_id.encode()).hexdigest()[:8], 16)
        rng = np.random.default_rng(seed)
        self._noise = rng.integers(0, 12, size=(height, width, 1), dtype=np.uint8)
        self.walkers: list[Walker] = []
        self.visits: list[Visit] = []
        if kind == "zone":
            # 兩個人在走道上來回；一個會走進禁區，另一個沒戴安全帽
            self.walkers = [
                Walker(path=[(0.10, 0.80), (0.45, 0.78), (0.80, 0.82)], period=24, helmet=True),
                Walker(path=[(0.15, 0.62), (0.50, 0.60)], period=18, phase=5, helmet=False),
                Walker(path=[(0.30, 0.92), (0.55, 0.90)], period=30, phase=11, helmet=True, vest=True),
            ]
        elif kind == "dock":
            # 堆高機每 40 秒進場一次；引導員每三次缺席一次 → followup_window 逾時 NG
            self.visits = [
                Visit("forklift", every=40, start=5, stay=15, at=(0.15, 0.40, 0.45, 0.85)),
                Visit("spotter", every=40, start=9, stay=10, at=(0.60, 0.42, 0.68, 0.85), skip_every=3),
            ]
        else:
            raise ValueError(f"不認識的場景 {kind!r}（zone / dock）")

    # ── 真值 ──────────────────────────────────────────────
    def _walker_boxes(self, wk: Walker, t: float) -> tuple[Box, Box]:
        fx, fy = wk.foot(t)
        hh = wk.height * self.h
        bw = hh * 0.36
        x, y = fx * self.w, fy * self.h
        body = (x - bw / 2, y - hh, x + bw / 2, y)
        head_r = bw * 0.38
        head = (x - head_r, y - hh - head_r * 0.2, x + head_r, y - hh + head_r * 1.8)
        return body, head

    def objects(self, t: float) -> list[Obj]:
        out: list[Obj] = []
        for wk in self.walkers:
            body, head = self._walker_boxes(wk, t)
            out.append(Obj("person", body))
            out.append(Obj("helmet" if wk.helmet else "no_helmet", head))
            out.append(Obj("vest" if wk.vest else "no_vest", (body[0], body[1] + (body[3] - body[1]) * 0.2,
                                                             body[2], body[1] + (body[3] - body[1]) * 0.6)))
        for v in self.visits:
            if v.present(t):
                x1, y1, x2, y2 = v.at
                out.append(Obj(v.class_name, (x1 * self.w, y1 * self.h, x2 * self.w, y2 * self.h)))
        return out

    # ── 畫面 ──────────────────────────────────────────────
    def render(self, t: float) -> np.ndarray:
        img = np.full((self.h, self.w, 3), (70, 74, 78), np.uint8)
        img[int(self.h * 0.35):] = (92, 96, 100)                          # 地板
        img = cv2.add(img, np.broadcast_to(self._noise, img.shape).copy())
        for i in range(0, self.w, 80):                                     # 地板格線
            cv2.line(img, (i, int(self.h * 0.35)), (i - 200, self.h), (84, 88, 92), 1)
        if self.kind == "zone":
            pts = (np.array(self.zone) * [self.w, self.h]).astype(np.int32)
            overlay = img.copy()
            cv2.fillPoly(overlay, [pts], (40, 40, 200))
            img = cv2.addWeighted(overlay, 0.25, img, 0.75, 0)
            cv2.polylines(img, [pts], True, (40, 40, 220), 3)
            cv2.putText(img, "RESTRICTED", tuple(pts[0] + [10, 30]), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (40, 40, 220), 2)
        for v in self.visits:
            if v.present(t):
                x1, y1, x2, y2 = (int(a * b) for a, b in zip(v.at, (self.w, self.h, self.w, self.h)))
                if v.class_name == "forklift":
                    cv2.rectangle(img, (x1, y1 + (y2 - y1) // 3), (x2, y2 - 20), (0, 140, 255), -1)
                    cv2.rectangle(img, (x1 + 20, y1), (x1 + (x2 - x1) // 2, y1 + (y2 - y1) // 3), (40, 40, 40), 4)
                    cv2.line(img, (x2, y2 - 30), (x2 + 60, y2 - 30), (180, 180, 180), 6)
                    for cx in (x1 + 40, x2 - 40):
                        cv2.circle(img, (cx, y2 - 15), 18, (20, 20, 20), -1)
                else:
                    self._draw_person(img, (x1, y1, x2, y2), helmet=True, vest=True)
        for wk in sorted(self.walkers, key=lambda w: w.foot(t)[1]):        # 遠的先畫
            body, _ = self._walker_boxes(wk, t)
            self._draw_person(img, tuple(int(v) for v in body), helmet=wk.helmet, vest=wk.vest,
                              stride=math.sin((t + wk.phase) * 6))
        cv2.putText(img, f"{self.camera_id}  SYNTHETIC DEMO  t={t:7.1f}s", (16, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (230, 230, 230), 2)
        return img

    @staticmethod
    def _draw_person(img, box, *, helmet: bool, vest: bool, stride: float = 0.0) -> None:
        x1, y1, x2, y2 = box
        w, h = x2 - x1, y2 - y1
        cx = (x1 + x2) // 2
        head_r = max(4, int(w * 0.38))
        torso = (x1 + w // 6, y1 + int(h * 0.12), x2 - w // 6, y1 + int(h * 0.6))
        cv2.rectangle(img, torso[:2], torso[2:], (0, 220, 120) if vest else (150, 90, 40), -1)
        leg = int(w * 0.25 * stride)
        cv2.line(img, (cx, torso[3]), (cx - w // 4 + leg, y2), (60, 50, 40), max(3, w // 6))
        cv2.line(img, (cx, torso[3]), (cx + w // 4 - leg, y2), (60, 50, 40), max(3, w // 6))
        cv2.circle(img, (cx, y1), head_r, (150, 180, 220), -1)
        if helmet:
            cv2.ellipse(img, (cx, y1 - head_r // 4), (head_r + 2, head_r), 0, 180, 360, (0, 215, 255), -1)
