"""影片庫縮圖：GET /api/media/videos/{id}/thumbnail 用（edgevision/media/recordings_api.py）。

第一次要的時候才做：取一幀 → 長邊 px 的 webp → 存 MinIO，之後直接讀。跟圖片縮圖（edgevision/media/review.py thumb_url）同一套想法，
差在 NAS 錄影沒有 blob，所以縮圖不記在 DB，用固定的物件路徑（影片 id＋建立時間＋px）找：
影片內容建立後不會變，路徑本身就是快取鍵；帶建立時間是怕測試叢集重建 DB 後 id 重複、撞到舊的縮圖。
"""

from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import Callable

from edgevision.media.library import BUCKET

logger = logging.getLogger(__name__)

THUMB_PREFIX = "library/thumbs/videos"
DEFAULT_AT_S = 1.0          # 開頭常是黑畫面或還沒對焦，取第 1 秒


class VideoUnavailable(Exception):
    """影片檔不在了（NAS 超過保留期、MinIO 沒有原檔）。"""


def thumb_key(video: dict, px: int) -> str:
    return f"{THUMB_PREFIX}/{video['id']}_{int(video['created_at'].timestamp())}_{px}.webp"


def thumb_prefix(vid: int) -> str:
    """這支影片所有縮圖的共同前綴（刪影片時一起清）。"""
    return f"{THUMB_PREFIX}/{vid}_"


def pick_offset(video: dict) -> float:
    """取第幾秒：匯入時有指定起點（start_s）就從那裡算；影片太短就取中間。"""
    start = float((video.get("params") or {}).get("start_s") or 0.0)
    dur = video.get("duration_s")
    at = DEFAULT_AT_S
    if dur:
        at = min(at, max(0.0, (float(dur) - start) / 2))
    return start + at


def encode(img, px: int, *, quality: int = 80) -> bytes:
    import cv2

    h, w = img.shape[:2]
    s = px / max(h, w)
    if s < 1:
        img = cv2.resize(img, (max(1, round(w * s)), max(1, round(h * s))), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".webp", img, [cv2.IMWRITE_WEBP_QUALITY, quality])
    if not ok:
        raise ValueError("縮圖編碼失敗")
    return buf.tobytes()


def _missing(exc: Exception) -> bool:
    return getattr(exc, "code", None) in ("NoSuchKey", "NoSuchObject")


def video_source(lib, video: dict, nas_path: Callable[[str, str], Path]) -> str:
    """影片在哪：NAS 錄影 → 本地路徑；上傳 / 剪下的 → 叢集內的 MinIO 預簽網址（不繞對外位址）。"""
    uri = video["uri"]
    if video["kind"] == "nas_recording" and uri.startswith("nas://raw/"):
        cam, _, file = uri.removeprefix("nas://raw/").partition("/")
        try:
            return str(nas_path(cam, file))
        except FileNotFoundError as exc:
            raise VideoUnavailable(f"NAS 上找不到 {uri}（可能已超過保留期）") from exc
    if not video.get("blob_sha256"):
        raise VideoUnavailable("這支影片沒有保存原檔")
    blob = lib.blob(video["blob_sha256"])
    if blob is None:
        raise VideoUnavailable("這支影片沒有保存原檔")
    return lib.objects.presigned_url(bucket=blob["bucket"], key=blob["object_key"], expires_minutes=10, internal=True)


def thumbnail(lib, video: dict, px: int, nas_path: Callable[[str, str], Path]) -> bytes:
    """回 webp 位元組。已做過就直接讀 MinIO；沒有就現做、存起來。

    video 要有 id、kind、uri、blob_sha256、params、duration_s、created_at（recordings_api 的 _SELECT 那幾欄）。
    解不出畫面 → ValueError；檔案不在 → VideoUnavailable。
    """
    from edgevision.media.frames import grab_frame

    key = thumb_key(video, px)
    try:
        return lib.objects.get(bucket=BUCKET, key=key)
    except Exception as exc:  # noqa: BLE001 —— 只有「物件不存在」才往下做，其他錯照丟
        if not _missing(exc):
            raise
    src = video_source(lib, video, nas_path)
    try:
        img = grab_frame(src, pick_offset(video))
    except Exception as exc:  # noqa: BLE001 —— PyAV 的錯誤種類很多（壞檔、還在寫、編碼不支援），對外都是「解不開」
        logger.warning("影片 %s 取縮圖失敗：%s", video["id"], exc)
        raise ValueError("影片解不開，做不出縮圖") from exc
    if img is None:
        raise ValueError("影片裡沒有可用的畫面")
    data = encode(img, px)
    lib.objects.put(bucket=BUCKET, key=key, data=io.BytesIO(data), length=len(data), content_type="image/webp")
    return data
