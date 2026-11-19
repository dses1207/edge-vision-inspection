"""瀏覽器直接載入的網址（<video src>、<img src>、下載連結）用的短效簽名。

為什麼要有：這類請求是瀏覽器自己發的，帶不了 Authorization header，切強制模式後會全部 401。
做法：前端用帶 token 的 API 拿到網址時，網址已經附上 ?exp=<到期秒>&sig=<HMAC>；關卡看到合法簽名就放行。

    簽的是「路徑＋到期時間」，不含其他 query（?download=1、?px= 照樣可以加）。
    只有 SIGNABLE 這幾種路由能簽：簽名只能拿來看檔案，不能拿來打一般 API。
    到期時間對齊 TTL 的整數倍：同一段時間內同一個檔案的網址不變，瀏覽器快取才用得上。
    金鑰在 Secret edgevision-url-signing（deploy_test.sh 自動產生，不進 git）；沒掛到時每個行程自己隨機產生一把
    （control-api 只有 1 個副本，只是重啟後舊網址失效）。
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import os
import re
import secrets
import time
from datetime import datetime
from urllib.parse import quote
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)
_TZ = ZoneInfo("Asia/Taipei")

KEY_FILE = os.environ.get("AUTH_URL_SIGNING_KEY_FILE", "/etc/edgevision/url-signing/key")
TTL_S = int(os.environ.get("AUTH_SIGNED_URL_TTL_S", 7200))

# 能簽的路由：只有回檔案 / 圖片 / 影片的
SIGNABLE = (
    re.compile(r"^/api/media/recordings/[^/]+/files/[^/]+$"),
    re.compile(r"^/api/media/videos/\d+/(stream|thumbnail)$"),
    re.compile(r"^/api/media/items/\d+/image$"),
)

_key_cache: tuple[float, bytes] = (0.0, b"")
_fallback = secrets.token_bytes(32)


def signable(path: str) -> bool:
    return any(p.match(path) for p in SIGNABLE)


def _key() -> bytes:
    global _key_cache
    at, k = _key_cache
    if k and time.time() - at < 60:
        return k
    try:
        with open(KEY_FILE, encoding="utf-8") as f:
            k = f.read().strip().encode()
        if len(k) < 32:
            raise ValueError("too short")
    except (FileNotFoundError, ValueError):
        if not _key_cache[1]:
            logger.warning("沒有網址簽名金鑰（%s），改用行程內隨機金鑰：重啟後舊網址失效", KEY_FILE)
        k = _fallback
    _key_cache = (time.time(), k)
    return k


def _mac(path: str, exp: int) -> str:
    return hmac.new(_key(), f"{path}\n{exp}".encode(), hashlib.sha256).hexdigest()[:32]


def expiry(now: float | None = None, ttl: int | None = None) -> int:
    """對齊 ttl 的下一個整數倍再加一個 ttl：有效 ttl～2×ttl 秒。"""
    ttl = ttl or TTL_S
    now = time.time() if now is None else now
    return (int(now) // ttl + 2) * ttl


def sign(path: str, now: float | None = None) -> str:
    """回附上簽名的網址（相對路徑）。path 不能簽時丟 ValueError。"""
    if not signable(path):
        raise ValueError(f"這個路徑不能簽：{path}")
    exp = expiry(now)
    return f"{quote(path)}?exp={exp}&sig={_mac(path, exp)}"


def sign_or_keep(path: str) -> str:
    """回應裡的網址欄位用：能簽就簽，不能簽原樣回。"""
    return sign(path) if signable(path) else path


def expires_at(url: str) -> str | None:
    m = re.search(r"[?&]exp=(\d+)", url)
    return datetime.fromtimestamp(int(m.group(1)), _TZ).isoformat() if m else None


def verify(path: str, exp: str | None, sig: str | None, now: float | None = None) -> bool:
    if not (exp and sig and exp.isdigit() and signable(path)):
        return False
    if int(exp) < (time.time() if now is None else now):
        return False
    return hmac.compare_digest(_mac(path, int(exp)), sig)
