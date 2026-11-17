"""失準偵測的影像部分：抓畫面（go2rtc 截圖）、縮成灰階小圖、FFT 相位相關求整張平移量。

跟舊系統唯一不同：舊系統每次直接開 RTSP 抓一幀。Tapo 同時連線數少，多開一條會擠掉推論那條，
所以改向 go2rtc 要 frame.jpeg —— go2rtc 本來就為 runner 拉著這條串流，截圖不多佔相機連線。
"""

from __future__ import annotations

import io

import numpy as np

from edgevision.ptz.settings import GO2RTC_API

REF_H, REF_W = 144, 256


def to_gray_small(img: np.ndarray) -> np.ndarray:
    """→ 144x256 灰階、去均值、加 Hanning 窗（相位相關前的標準處理，減少邊緣效應）。"""
    if img.ndim == 3:
        img = (0.114 * img[..., 0] + 0.587 * img[..., 1] + 0.299 * img[..., 2])     # BGR → 灰階
    ys = max(1, img.shape[0] // REF_H)
    xs = max(1, img.shape[1] // REF_W)
    s = img[::ys, ::xs][:REF_H, :REF_W].astype(np.float64)
    wy = np.hanning(s.shape[0])[:, None]
    wx = np.hanning(s.shape[1])[None, :]
    return (s - s.mean()) * wy * wx


def phase_corr(a: np.ndarray, b: np.ndarray) -> tuple[int, int, float]:
    """回 (dx, dy, conf)：整張畫面平移量（像素）與峰值銳利度（信心；雜訊時峰值鈍、conf 低）。"""
    from numpy.fft import fft2, ifft2

    R = fft2(a) * np.conj(fft2(b))
    R /= np.abs(R) + 1e-9
    r = np.abs(ifft2(R))
    pk = np.unravel_index(np.argmax(r), r.shape)
    dy, dx = int(pk[0]), int(pk[1])
    if dy > a.shape[0] // 2:
        dy -= a.shape[0]
    if dx > a.shape[1] // 2:
        dx -= a.shape[1]
    return dx, dy, float(r.max() / (r.mean() + 1e-12))


def grab_jpeg(cam_id: str, *, timeout: float = 10.0) -> bytes:
    """go2rtc 的原始串流（不是 _marked：要乾淨畫面，框會干擾相位相關）截一張。"""
    import httpx

    r = httpx.get(f"{GO2RTC_API}/api/frame.jpeg", params={"src": cam_id}, timeout=timeout)
    r.raise_for_status()
    if not r.content:
        raise RuntimeError(f"go2rtc 沒有回傳 {cam_id} 的畫面")
    return r.content


def grab_gray(cam_id: str) -> np.ndarray:
    import cv2

    img = cv2.imdecode(np.frombuffer(grab_jpeg(cam_id), np.uint8), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError(f"{cam_id} 的截圖無法解碼")
    return to_gray_small(img)


def dump_ref(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    np.savez_compressed(buf, ref=arr)
    return buf.getvalue()


def load_ref(data: bytes) -> np.ndarray:
    return np.load(io.BytesIO(data))["ref"]
