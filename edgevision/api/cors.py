"""CORS：推論 API 與控制 API 共用同一套設定（環境變數 CORS_ORIGINS）。

  CORS_ORIGINS=*                         不限來源（開發 / 內網環境；正式對外前改成白名單）
                                         
  CORS_ORIGINS=https://a,http://b:5173   白名單（逗號分隔；完整 origin = 協定 + 主機 + 埠）
  沒設只有同源可用

⚠️ 開 OAuth（cookie 帶身分）後建議改回白名單：不限來源時任何網頁都能借已登入使用者的身分打寫入端點。
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)


def cors_origins(raw: str | None = None) -> list[str]:
    raw = os.environ.get("CORS_ORIGINS", "") if raw is None else raw
    return [o.strip() for o in raw.split(",") if o.strip()]


def apply_cors(app, name: str) -> None:
    from fastapi.middleware.cors import CORSMiddleware
    origins = cors_origins()
    if not origins:
        logger.info("[%s] 未設定 CORS_ORIGINS —— 僅同源可用", name)
        return
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"] if "*" in origins else origins,
        # 搭配 "*" 時 Starlette 會回填請求的 origin（瀏覽器不收 "*" + credentials）
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["*"],
    )
    logger.info("[%s] CORS 已啟用：%s", name, "不限來源" if "*" in origins else ", ".join(origins))
