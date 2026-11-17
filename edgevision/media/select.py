"""從長錄影挑「值得標」的畫面 —— 不用任何 AI 模型，只用傳統影像處理（2026-10-14 決定：現役模型不可信）。

固定間隔取樣在 24h 錄影上會拿到大量幾乎一樣的畫面（產線大部分時間沒在動）。這裡分四步：

    ① 密集取樣每 dense_interval_s 秒看一張（只在記憶體裡算特徵，不存圖）
    ② 去重跟【所有已留下的】比：感知雜湊（dHash 16×16 = 256 bit）差 < dedup_bits【且】
                 「變化面積」< dedup_area 才算重複。變化面積＝模糊後 64×36 灰階逐像素差 > diff_level 的比例：
                 壓縮雜訊（P 幀累積、I 幀重置）只造成小差異，真的有東西出現是大差異且集中。
                 dHash 單獨用不夠（作業台上換一疊料只差幾 bit）；縮圖向量距離也不行（siteb12 實測靜止畫面
                 因壓縮雜訊就飄到 0.08，有人進場才 0.12，分不開）
    ③ 有沒有在動背景相減（OpenCV MOG2）：前景像素佔關注區的比例 ≥ motion_threshold 算「有變動」
    ④ 挑多樣縮圖 + 色彩直方圖當特徵，k-means 分群、每群取最接近中心的一張
                 有變動的畫面優先；保留 still_ratio 的名額給沒變動的（避免只看得到「有人在動」的情境）

比對前先把 ignore_regions（預設：左上角相機時間戳）塗掉 —— 時間戳每秒都變，不遮的話每張都「不一樣」、
每張都「有變動」。roi 有給就只看那一塊（例如作業台；走道有人經過不算）。

座標一律 0–1 正規化 [x1, y1, x2, y2]（跟標注同一套）。
挑中的幀第二輪才解碼存檔（依時間排序 seek），候選只留特徵、不佔磁碟。
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterator

import numpy as np

logger = logging.getLogger(__name__)

# IPCAM 相機的時間戳在左上角（siteb12 / ipcam87 實測約 0–27% 寬、0–9% 高；留一點邊）
DEFAULT_IGNORE = [[0.0, 0.0, 0.30, 0.11]]
_WORK_W, _WORK_H = 320, 180          # 背景相減 / 特徵用的工作解析度
_HASH = 16                            # dHash 16×16 → 256 bit
_TINY_W, _TINY_H = 64, 36             # 去重比變化面積用


@dataclass(slots=True)
class SelectParams:
    target: int = 2000                        # 最後要幾張（使用者 10/14：至少 2000；去重後不到就全收）
    dense_interval_s: float = 1.0             # ① 每幾秒看一張
    dedup_bits: int = 12                      # ② 256 bit 裡差異少於幾 bit
    dedup_area: float = 0.01                  # ②【且】變化面積小於這個比例，才算重複
    diff_level: int = 20                      # ② 灰階差多少（0–255）算「有變化」的像素
    motion_threshold: float = 0.004           # ③ 前景像素比例
    still_ratio: float = 0.1                  # ④ 保留給沒變動畫面的名額比例
    ignore_regions: list[list[float]] = field(default_factory=lambda: [list(r) for r in DEFAULT_IGNORE])
    roi: list[float] | None = None

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__slots__}


@dataclass(slots=True)
class Candidate:
    src: int                  # 第幾個來源檔
    offset_s: float           # 在該檔的媒體時間
    motion: float             # 前景比例（暖機期間為 None → 以 0 計）
    feat: np.ndarray          # 分群特徵
    tiny: np.ndarray | None = None   # 去重用的模糊小灰階
    moving: bool = False
    group: int = -1           # 分群編號（挑中後才有）
    group_size: int = 0


@dataclass(slots=True)
class SelectStats:
    sampled: int = 0
    duplicates: int = 0
    candidates: int = 0
    moving: int = 0
    selected: int = 0
    selected_moving: int = 0
    by_hour: Counter = field(default_factory=Counter)

    def to_dict(self) -> dict:
        return {"sampled": self.sampled, "after_dedup": self.candidates, "duplicates_removed": self.duplicates,
                "moving": self.moving, "selected": self.selected, "selected_moving": self.selected_moving,
                "selected_still": self.selected - self.selected_moving,
                "by_hour": dict(sorted(self.by_hour.items()))}


# ── 單張影像的特徵 ───────────────────────────────────────

def _rect_px(r: list[float], w: int, h: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = r
    return (max(0, int(x1 * w)), max(0, int(y1 * h)), min(w, int(np.ceil(x2 * w))), min(h, int(np.ceil(y2 * h))))


def validate_rect(r: list[float], name: str) -> list[float]:
    if len(r) != 4 or not all(0.0 <= v <= 1.0 for v in r) or r[2] <= r[0] or r[3] <= r[1]:
        raise ValueError(f"{name} 要是 0–1 的 [x1, y1, x2, y2]，且 x2 > x1、y2 > y1：{r}")
    return [float(v) for v in r]


class FrameAnalyzer:
    """把一張 BGR 影像變成：遮罩後的工作圖、dHash、分群特徵；並維護背景模型算前景比例。"""

    def __init__(self, p: SelectParams) -> None:
        import cv2

        self.p = p
        self._bg = cv2.createBackgroundSubtractorMOG2(history=120, varThreshold=25, detectShadows=False)
        self._kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self._mask: np.ndarray | None = None
        self._n = 0

    def _work(self, bgr: np.ndarray) -> np.ndarray:
        """遮掉 ignore_regions、裁出 roi、縮到工作解析度。"""
        import cv2

        img = bgr
        h, w = img.shape[:2]
        if self.p.ignore_regions:
            img = img.copy()
            for r in self.p.ignore_regions:
                x1, y1, x2, y2 = _rect_px(r, w, h)
                img[y1:y2, x1:x2] = 0
        if self.p.roi:
            x1, y1, x2, y2 = _rect_px(self.p.roi, w, h)
            img = img[y1:y2, x1:x2]
        return cv2.resize(img, (_WORK_W, _WORK_H), interpolation=cv2.INTER_AREA)

    def _valid_mask(self, w: int, h: int) -> np.ndarray:
        """工作圖上哪些像素算數（roi 內、ignore 外）；前景比例的分母。"""
        if self._mask is None:
            m = np.ones((_WORK_H, _WORK_W), np.uint8)
            rx1, ry1, rx2, ry2 = self.p.roi or [0.0, 0.0, 1.0, 1.0]
            for r in self.p.ignore_regions:
                # ignore 區換算到 roi 內的座標
                ix1, iy1 = (r[0] - rx1) / (rx2 - rx1), (r[1] - ry1) / (ry2 - ry1)
                ix2, iy2 = (r[2] - rx1) / (rx2 - rx1), (r[3] - ry1) / (ry2 - ry1)
                if ix2 <= 0 or iy2 <= 0 or ix1 >= 1 or iy1 >= 1:
                    continue
                x1, y1, x2, y2 = _rect_px([max(0, ix1), max(0, iy1), min(1, ix2), min(1, iy2)], _WORK_W, _WORK_H)
                m[y1:y2, x1:x2] = 0
            self._mask = m
        return self._mask

    def tiny_mask(self) -> np.ndarray:
        import cv2
        return cv2.resize(self._valid_mask(_WORK_W, _WORK_H), (_TINY_W, _TINY_H), interpolation=cv2.INTER_NEAREST) > 0

    def analyze(self, bgr: np.ndarray) -> tuple[np.ndarray, float | None, np.ndarray, np.ndarray]:
        import cv2

        small = self._work(bgr)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        valid = self._valid_mask(_WORK_W, _WORK_H)

        # ② dHash：相鄰像素比亮度（對整體亮度變化不敏感）
        g = cv2.resize(gray, (_HASH + 1, _HASH), interpolation=cv2.INTER_AREA)
        bits = np.packbits((g[:, 1:] > g[:, :-1]).ravel())

        # ③ 背景相減
        fg = self._bg.apply(gray)
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, self._kernel)
        self._n += 1
        denom = int(valid.sum()) or 1
        motion = None if self._n <= 5 else float(((fg > 0) & (valid > 0)).sum()) / denom   # 前 5 張背景還沒學好

        # ④ 分群特徵：32×18 灰階（去均值、單位長度）+ HSV 色相/飽和度直方圖
        thumb = cv2.resize(gray, (32, 18), interpolation=cv2.INTER_AREA).astype(np.float32).ravel()
        thumb -= thumb.mean()
        thumb /= (np.linalg.norm(thumb) + 1e-6)
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], valid, [16, 4], [0, 180, 0, 256]).ravel()
        hist /= (np.linalg.norm(hist) + 1e-6)
        feat = np.concatenate([thumb, 0.5 * hist]).astype(np.float32)
        tiny = cv2.GaussianBlur(cv2.resize(gray, (_TINY_W, _TINY_H), interpolation=cv2.INTER_AREA), (3, 3), 0)
        return bits, motion, feat, tiny


def hamming_to_all(bits: np.ndarray, kept: np.ndarray) -> np.ndarray:
    """bits（32 bytes）跟 kept（N×32）逐列的漢明距離。"""
    if kept.size == 0:
        return np.empty(0, np.int32)
    return np.unpackbits(np.bitwise_xor(kept, bits), axis=1).sum(axis=1)


# ── 挑選 ─────────────────────────────────────────────────

def _pick_diverse(cands: list[Candidate], k: int, seed: int = 0) -> list[Candidate]:
    """k-means 分 k 群，每群取最接近中心的一張；候選不夠 k 張就全收。"""
    import cv2

    if k <= 0 or not cands:
        return []
    if len(cands) <= k:
        for c in cands:
            c.group_size = 1
        return list(cands)
    X = np.stack([c.feat for c in cands]).astype(np.float32)
    cv2.setRNGSeed(seed)
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-4)
    _, labels, centers = cv2.kmeans(X, k, None, crit, 3, cv2.KMEANS_PP_CENTERS)
    labels = labels.ravel()
    out = []
    for g in range(k):
        idx = np.flatnonzero(labels == g)
        if idx.size == 0:
            continue
        d = np.linalg.norm(X[idx] - centers[g], axis=1)
        c = cands[int(idx[int(np.argmin(d))])]
        c.group, c.group_size = g, int(idx.size)
        out.append(c)
    return out


def choose(cands: list[Candidate], p: SelectParams) -> list[Candidate]:
    """有變動的優先，still_ratio 留給沒變動的；一邊不夠就讓給另一邊。"""
    moving = [c for c in cands if c.moving]
    still = [c for c in cands if not c.moving]
    n_still = min(len(still), int(round(p.target * p.still_ratio)))
    n_move = min(len(moving), p.target - n_still)
    n_still = min(len(still), p.target - n_move)          # 有變動的不夠 → 補沒變動的
    picked = _pick_diverse(moving, n_move) + _pick_diverse(still, n_still)
    # 群組編號在兩邊各自從 0 起算；沒變動那邊加前綴避免撞號
    for c in picked:
        if not c.moving and c.group >= 0:
            c.group += 100_000
    return sorted(picked, key=lambda c: (c.src, c.offset_s))


# ── 兩輪讀檔 ─────────────────────────────────────────────

def changed_area(a: np.ndarray, b: np.ndarray, mask: np.ndarray, n_valid: int, level: int) -> float:
    return float(((np.abs(a.astype(np.int16) - b.astype(np.int16)) > level) & mask).sum()) / n_valid


def _is_duplicate(bits, tiny, kept_arr, cands, mask, n_valid, p: SelectParams) -> bool:
    """dHash 先篩（便宜），近的才比變化面積。"""
    if kept_arr.size == 0:
        return False
    ham = hamming_to_all(bits, kept_arr)
    for j in np.flatnonzero(ham < p.dedup_bits):
        if changed_area(tiny, cands[int(j)].tiny, mask, n_valid, p.diff_level) < p.dedup_area:
            return True
    return False


def scan(paths: list[tuple[Path, float, float | None]], p: SelectParams, *,
         progress: Callable[[str], None] | None = None,
         cancelled: Callable[[], bool] | None = None) -> tuple[list[Candidate], SelectStats]:
    """第一輪：每 dense_interval_s 秒算一次特徵，去重後留下候選。paths = [(檔案, start_s, end_s)]。"""
    from edgevision.media.frames import decode_every

    an = FrameAnalyzer(p)
    st = SelectStats()
    tmask = an.tiny_mask()
    n_valid = int(tmask.sum()) or 1
    kept_bits: list[np.ndarray] = []
    kept_arr = np.empty((0, _HASH * _HASH // 8), np.uint8)
    cands: list[Candidate] = []
    for si, (path, start_s, end_s) in enumerate(paths):
        for t, img in decode_every(path, interval_s=p.dense_interval_s, start_s=start_s, end_s=end_s):
            if cancelled and cancelled():
                raise KeyboardInterrupt
            bits, motion, feat, tiny = an.analyze(img)
            st.sampled += 1
            if len(kept_bits) != kept_arr.shape[0]:
                kept_arr = np.stack(kept_bits)
            if _is_duplicate(bits, tiny, kept_arr, cands, tmask, n_valid, p):
                st.duplicates += 1
            else:
                kept_bits.append(bits)
                c = Candidate(src=si, offset_s=t, motion=motion or 0.0, feat=feat, tiny=tiny,
                              moving=(motion or 0.0) >= p.motion_threshold)
                cands.append(c)
                st.moving += int(c.moving)
            if progress:
                progress(f"挑圖第一輪：看了 {st.sampled} 張、留下 {len(cands)} 張（{Path(path).name} {t:.0f}s）")
    st.candidates = len(cands)
    return cands, st


def extract(path: Path, offsets: list[float], *, quality: int = 95) -> Iterator[tuple[float, bytes, int, int]]:
    """第二輪：只解碼挑中的時間點（依序 seek），回 (offset, jpeg, w, h)。"""
    import cv2

    from edgevision.media.frames import decode_at

    for t, img in decode_at(path, sorted(offsets)):
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if ok:
            yield t, buf.tobytes(), img.shape[1], img.shape[0]
