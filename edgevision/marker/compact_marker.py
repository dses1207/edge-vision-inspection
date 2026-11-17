"""精簡版影格標記（v2）：13 格，單一 display_meta 畫得完。

⚠️ 為什麼要 v2（見 docs/findings）：
    v1（frame_marker.py）是 32 格：4 SYNC + 20 ID + 8 CRC。
    在 DeepStream 路線下畫不出來 —— 實測確認
    **NvDsDisplayMeta 的 rect_params 陣列長度就是 16**，
    要 33 個矩形（含黑底）必須分 3 批，而分批在實務上不可靠。

設計依據（查證後大幅精簡，見 research）：
    v1 的規格是為了對抗「光學失真 + 類比雜訊 + 幾何未知」——
    那是 VITC／AprilTag／QR 的世界。我們一項都沒有：
      · canvas 直接讀 buffer，沒有鏡頭、沒有透視
      · 全程數位路徑，沒有類比雜訊
      · marker 位置由我們固定，不需要 finder pattern
    唯一真實的威脅只有 H.264 壓縮。

    · ID 10 bit：1024 幀 ≈ 34 秒 @30fps，對「短窗配對」綽綽有餘。
      依 RFC 3550 的模數原則（RTP sequence number 也只有 16 bit）：
      模數空間要遠大於最大合理位移，34 秒 vs 10 秒窗 = 3.4 倍餘裕。
    · 1 bit parity 取代 8 bit CRC：LTC 也只用 1 個 parity bit
      （VITC 用 8-bit checksum 是因為它跑在類比訊號上）。
      真正的驗證靠「時序窗檢查」—— 解出的 id 必須落在
      lastSeen 之後的合理範圍內。這比 CRC 更強，因為它用了
      CRC 用不到的外部資訊（我們知道上一幀是誰）。
    · Gray code：相鄰 frame id 只差 1 bit，讀錯時誤差通常是 ±1 幀
      而非任意跳躍 —— 對短窗配對而言，優雅降級比「偵測後丟棄」有用。

    · CELL 維持 10px：H.264 的 deblocking 每側最多污染 3 個 sample，
      10 - 6 = 4px 乾淨中心。取樣只取中心，不要平均整格。
"""

from __future__ import annotations

SYNC_BITS = 2
SYNC_PATTERN = [1, 0]          # 一白一黑：同時提供每幀的黑白參考電平
ID_BITS = 12                   # 4096 幀 ≈ 137 秒 @30fps
#
# ⚠️ 為什麼從 10 bit 加到 12：
#   10 bit = 1024 幀 = 34 秒繞回一次，而前端的結果 buffer 保留約 30 秒 ——
#   兩者太接近，繞回時舊 key 可能還在 buffer 裡造成碰撞。
#   12 bit = 137 秒，是 buffer 保留時間的 4.5 倍，安全。
#   再大就超過 rect 上限（13 bit → 17 個矩形 > 16）。
PARITY_BITS = 1
TOTAL_BITS = SYNC_BITS + ID_BITS + PARITY_BITS      # 13

CELL = 10
GUARD = 2
MARKER_WIDTH = TOTAL_BITS * CELL + GUARD * 2        # 154
MARKER_HEIGHT = CELL + GUARD * 2                    # 14

ID_MASK = (1 << ID_BITS) - 1
# 底色 + 13 格 = 14 個矩形，單一 display_meta（上限 16）畫得完
MAX_RECTS_PER_META = 16


def _gray(n: int) -> int:
    """二進位 → Gray code。相鄰值只差 1 bit。"""
    return n ^ (n >> 1)


def _ungray(g: int) -> int:
    """Gray code → 二進位。"""
    n = g
    shift = 1
    while shift < ID_BITS:
        n ^= n >> shift
        shift <<= 1
    return n


def marker_bits(frame_id: int) -> list[int]:
    """算出 13 個 bit：SYNC + Gray(ID) + parity。"""
    value = _gray(frame_id & ID_MASK)
    bits = list(SYNC_PATTERN)
    bits += [(value >> i) & 1 for i in range(ID_BITS - 1, -1, -1)]
    bits.append(sum(bits[SYNC_BITS:]) & 1)      # 偶同位
    return bits


def decode_bits(bits: list[int]) -> int | None:
    """13 個 bit → frame id。SYNC 或 parity 不符時回 None。"""
    if len(bits) != TOTAL_BITS or bits[:SYNC_BITS] != SYNC_PATTERN:
        return None
    body = bits[SYNC_BITS:SYNC_BITS + ID_BITS]
    if (sum(body) & 1) != bits[-1]:
        return None
    g = 0
    for b in body:
        g = (g << 1) | b
    return _ungray(g)


def stamp_osd(pyds, batch_meta, frame_meta, frame_id: int,
              width: int, height: int) -> bool:
    """用 nvdsosd 把 marker 畫進這一幀的右下角。

    ⚠️ 兩個實測踩過的坑：
      ① nvdsosd 必須設 display-bbox=1，否則【完全忽略】display_meta 的
         矩形：probe 照跑、不拋例外、但像素一點都沒變。
      ② rect_params 陣列長度是 16，存取 rect_params[16] 會拋
         list index out of range。故總矩形數必須 ≤ 16。
    """
    bits = marker_bits(frame_id)
    bx = width - MARKER_WIDTH
    by = height - MARKER_HEIGHT

    # 黑底（含 GUARD 留白）+ 每格明確上色，共 14 個矩形
    rects = [(bx, by, MARKER_WIDTH, MARKER_HEIGHT, 0.0)]
    rects += [
        (bx + GUARD + i * CELL, by + GUARD, CELL, CELL, 1.0 if b else 0.0)
        for i, b in enumerate(bits)
    ]
    if len(rects) > MAX_RECTS_PER_META:
        return False

    dm = pyds.nvds_acquire_display_meta_from_pool(batch_meta)
    if dm is None:
        return False
    dm.num_rects = len(rects)
    for idx, (x, y, w, h, gray) in enumerate(rects):
        r = dm.rect_params[idx]
        r.left, r.top, r.width, r.height = int(x), int(y), int(w), int(h)
        r.border_width = 0
        r.has_bg_color = 1
        r.bg_color.set(gray, gray, gray, 1.0)
    pyds.nvds_add_display_meta_to_frame(frame_meta, dm)
    return True


def decode_marker(image) -> int | None:
    """從影像右下角讀回 frame id（numpy 影像，給測試與 QA 用）。

    取樣只取每格中心，不平均整格 —— deblocking 會污染格子邊緣。
    門檻用該幀 SYNC 格的黑白電平動態決定，不寫死 127.5：
    這能自動吸收 QP 變動與亮度漂移。
    """
    ih, iw = image.shape[:2]
    if ih < MARKER_HEIGHT or iw < MARKER_WIDTH:
        return None
    oy, ox = ih - MARKER_HEIGHT, iw - MARKER_WIDTH

    inset = max(1, CELL // 3)       # 只取中心約 1/3
    y0, y1 = oy + GUARD + inset, oy + GUARD + CELL - inset

    levels = []
    for i in range(TOTAL_BITS):
        x0 = ox + GUARD + i * CELL + inset
        x1 = ox + GUARD + (i + 1) * CELL - inset
        patch = image[y0:y1, x0:x1]
        if patch.size == 0:
            return None
        levels.append(float(patch.mean()))

    hi, lo = levels[0], levels[1]   # SYNC = [白, 黑]
    if hi - lo < 40:
        return None
    threshold = (hi + lo) / 2
    return decode_bits([1 if v > threshold else 0 for v in levels])
