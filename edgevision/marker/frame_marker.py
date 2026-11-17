"""影格身分標記：把 frame id 畫進畫面，讓身分隨像素一起傳遞。

為什麼需要（見 docs/findings/2026-08-28-sync-verification.md）：
  影像與偵測結果走兩條路徑，各自的 RTP timestamp 從獨立基準起算，
  偏移抖動 200ms，無法配對。captureTime 與 RTCP SR 也都證實不可用。

  把 id 畫進畫面後，它就是畫面內容的一部分 —— 不管中間經過幾層
  轉發或重新編碼，只要畫面到得了，id 就在那裡。

設計取捨：
  ・純黑純白 —— H.264 量化後仍能穩定區分，中間灰階容易糊掉
  ・每 bit 用一個方格 —— 方格夠大，壓縮的邊緣模糊不會吃掉中心
  ・取樣中心區域 —— 避開方格邊界，那裡最容易被壓縮糊化
  ・CRC 校驗 —— 讀錯時能發現，寧可回 None 也不要給錯的 id
"""

from __future__ import annotations

import numpy as np

# 版面配置。
#
# 前導同步樣式（SYNC）：固定的 1010 交替樣式，用途有二 ——
#   ① 確認標記真的存在（沒有標記的畫面不會誤判出 id）
#   ② 提供黑白參考點，可據此自適應決定閾值，不必寫死 127.5
#
# CRC 加大到 8 bit：實測在低碼率（0.2-0.5Mbps）+ ultrafast 編碼下，
# 4 bit CRC 的碰撞率不足以擋掉誤讀（16 種值，隨機碰撞 1/16）。
# 誤讀比漏讀危險 —— 漏讀只是沒框，誤讀會把框畫到錯的幀上。
SYNC_BITS = 4
SYNC_PATTERN = [1, 0, 1, 0]
ID_BITS = 20
CRC_BITS = 8
TOTAL_BITS = SYNC_BITS + ID_BITS + CRC_BITS      # 32

CELL = 10                            # 每格邊長（像素）
GUARD = 2                            # 四周留白，隔開畫面內容
MARKER_WIDTH = TOTAL_BITS * CELL + GUARD * 2      # 244
MARKER_HEIGHT = CELL + GUARD * 2                  # 14

ID_MASK = (1 << ID_BITS) - 1


def _crc8(value: int) -> int:
    """CRC-8（多項式 x^8+x^2+x+1，即 0x07）。

    用於偵測讀取錯誤。8 bit 使隨機碰撞率降到 1/256，
    配合 SYNC 樣式檢查後，誤讀機率可忽略。
    """
    reg = 0
    for i in range(ID_BITS - 1, -1, -1):
        reg ^= ((value >> i) & 1) << 7
        for _ in range(1):
            if reg & 0x80:
                reg = ((reg << 1) ^ 0x07) & 0xFF
            else:
                reg = (reg << 1) & 0xFF
    return reg & 0xFF


def encode_marker(frame_id: int) -> np.ndarray:
    """產生標記圖塊（BGR）。回傳 shape 為 (MARKER_HEIGHT, MARKER_WIDTH, 3)。"""
    value = frame_id & ID_MASK
    bits = list(SYNC_PATTERN)
    bits += [(value >> i) & 1 for i in range(ID_BITS - 1, -1, -1)]
    bits += [(_crc8(value) >> i) & 1 for i in range(CRC_BITS - 1, -1, -1)]

    tile = np.zeros((MARKER_HEIGHT, MARKER_WIDTH, 3), dtype=np.uint8)
    for i, b in enumerate(bits):
        x0 = GUARD + i * CELL
        tile[GUARD:GUARD + CELL, x0:x0 + CELL] = 255 if b else 0
    return tile


def stamp(image: np.ndarray, frame_id: int) -> np.ndarray:
    """就地把標記蓋在影像右下角。回傳同一個陣列，方便串接。

    右下角而非左上角：畫面左上/右上常疊時間戳等 OSD，會被 marker 擋住。
    右下角偏移 = (影像高 - marker 高, 影像寬 - marker 寬)，decode 用同一算法。
    """
    tile = encode_marker(frame_id)
    h, w = tile.shape[:2]
    ih, iw = image.shape[:2]
    y0, x0 = ih - h, iw - w
    image[y0:y0 + h, x0:x0 + w] = tile
    return image


def decode_marker(image: np.ndarray) -> int | None:
    """從影像右下角讀回 frame id。CRC 不符或尺寸不足時回 None。

    取樣策略：每格只取中心 60% 的區域求平均 —— 方格邊界在 H.264
    壓縮後最容易糊化，避開它才穩定。

    右下角原點：marker 貼在 (影像高 - marker 高, 影像寬 - marker 寬)，
    與 stamp() 同一算法，故先算出這個偏移，再套原本的格內取樣。
    """
    ih, iw = image.shape[:2]
    if ih < MARKER_HEIGHT or iw < MARKER_WIDTH:
        return None

    oy = ih - MARKER_HEIGHT            # marker 左上角在影像中的 y 偏移
    ox = iw - MARKER_WIDTH             # marker 左上角在影像中的 x 偏移

    inset = max(1, CELL // 5)          # 邊緣各縮 20%
    y0 = oy + GUARD + inset
    y1 = oy + GUARD + CELL - inset

    levels: list[float] = []
    for i in range(TOTAL_BITS):
        x0 = ox + GUARD + i * CELL + inset
        x1 = ox + GUARD + (i + 1) * CELL - inset
        patch = image[y0:y1, x0:x1]
        if patch.size == 0:
            return None
        levels.append(float(patch.mean()))

    # 自適應閾值：用前導的 1010 樣式當黑白參考。
    # 壓縮會壓縮動態範圍，寫死 127.5 在低碼率下會判錯。
    whites = [levels[i] for i, b in enumerate(SYNC_PATTERN) if b == 1]
    blacks = [levels[i] for i, b in enumerate(SYNC_PATTERN) if b == 0]
    if not whites or not blacks:
        return None
    hi, lo = sum(whites) / len(whites), sum(blacks) / len(blacks)

    # 對比不足代表這裡根本不是標記（或已被壓爛），直接放棄。
    if hi - lo < 40:
        return None
    threshold = (hi + lo) / 2

    bits = [1 if v > threshold else 0 for v in levels]

    # SYNC 樣式必須完全吻合，否則不是有效標記
    if bits[:SYNC_BITS] != SYNC_PATTERN:
        return None

    value = 0
    for b in bits[SYNC_BITS:SYNC_BITS + ID_BITS]:
        value = (value << 1) | b
    crc = 0
    for b in bits[SYNC_BITS + ID_BITS:]:
        crc = (crc << 1) | b

    return value if crc == _crc8(value) else None


class FrameMarker:
    """標記器：維護遞增的 frame id，並提供打標與還原。"""

    def __init__(self, start: int = 0) -> None:
        self._next = start & ID_MASK

    def next_id(self) -> int:
        fid = self._next
        self._next = (self._next + 1) & ID_MASK
        return fid

    def stamp(self, image: np.ndarray, frame_id: int | None = None) -> int:
        """打標並回傳使用的 id。未指定時自動取號。"""
        fid = self.next_id() if frame_id is None else (frame_id & ID_MASK)
        stamp(image, fid)
        return fid

    @staticmethod
    def read(image: np.ndarray) -> int | None:
        return decode_marker(image)
