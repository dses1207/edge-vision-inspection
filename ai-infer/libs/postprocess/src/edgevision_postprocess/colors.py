"""框的顏色 —— 直播（probe 掛給 nvdsosd）與存檔（render.py）共用這一份，兩邊看起來才一致。

規則（醒目，不要灰）：
  1. 錯誤 / 違規類（名稱含 error、Err 開頭、bare hand）→ 紅色，一眼看出異常
  2. 指定色：board / hand / mat（固定配色，換模型也不變）
  3. 其他 → 高飽和亮色盤依 class_id 取，同一個模型的同一類永遠同色；色盤刻意不含紅色，不跟錯誤混淆
"""
from __future__ import annotations

RGB = tuple[float, float, float]

ERROR_RGB: RGB = (1.0, 0.13, 0.13)
NAMED_RGB: dict[str, RGB] = {
    "board": (0.2, 0.6, 1.0),
    "hand": (1.0, 0.5, 0.0),
    "mat": (0.2, 1.0, 0.35),
}
PALETTE_RGB: list[RGB] = [
    (1.0, 0.92, 0.0),    # 黃
    (0.0, 0.9, 1.0),     # 青
    (1.0, 0.2, 0.9),     # 洋紅
    (0.55, 1.0, 0.1),    # 萊姆綠
    (1.0, 0.6, 0.0),     # 橘
    (0.3, 0.55, 1.0),    # 天藍
    (1.0, 0.45, 0.7),    # 粉
]


def is_error_class(class_name: str) -> bool:
    n = (class_name or "").lower()
    return "error" in n or n.startswith("err") or n == "bare hand"


def box_color_rgb(class_name: str, class_id: int | None = None) -> RGB:
    """RGB 0~1（nvdsosd 用）。純函式，可測。"""
    if is_error_class(class_name):
        return ERROR_RGB
    if class_name in NAMED_RGB:
        return NAMED_RGB[class_name]
    if class_id is None or class_id < 0:
        class_id = sum(map(ord, class_name or ""))      # 沒有 class_id 時用名字算，仍然穩定
    return PALETTE_RGB[class_id % len(PALETTE_RGB)]


def box_color_bgr(class_name: str, class_id: int | None = None) -> tuple[int, int, int]:
    """BGR 0~255（OpenCV 存檔用）。"""
    r, g, b = box_color_rgb(class_name, class_id)
    return int(b * 255), int(g * 255), int(r * 255)
