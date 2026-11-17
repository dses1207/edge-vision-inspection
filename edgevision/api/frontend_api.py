"""給前端的能力探索端點（capability discovery）。

⭐ 這一層存在的理由，跟一般 API 不同：

    一般 API 傳的是「資料」（有哪些偵測、哪些線路）。
    這裡傳的是「怎麼解讀資料的規則」——串流位址怎麼組、
    marker 怎麼解碼、結果多久算過期。

    這些先前全部是隱性知識，只寫在我們自己的 web/index.html 裡。
    外來前端只能讀那份原始碼逆向工程，而且我們一改規格
    （例如 marker ID 從 10 bit 改 12 bit）他們就得跟著改版。

    把它變成 API 之後：前端從這裡讀參數，規格改了不用改前端。
    這是 Milestone / ONVIF 的 capability discovery 做法。

⚠️ 這裡回的東西都是「靜態設定」，前端啟動時讀一次即可，不要輪詢。
"""

from __future__ import annotations

import logging
import os

from fastapi import APIRouter, HTTPException

from edgevision.marker import compact_marker as marker

from edgevision.api.schemas import CameraOut, FrontendConfig

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["frontend"])

_db = None


def bind_db(db) -> None:
    global _db
    _db = db


def _stream_base() -> str:
    """go2rtc 的對外位址。

    ⚠️ 不能用容器內的服務名（go2rtc:1984）—— 那是給後端連的，
    瀏覽器解析不到。要用對外可達的位址，由部署層給。
    """
    return os.environ.get("GO2RTC_PUBLIC_URL", "").rstrip("/")


@router.get("/frontend-config", response_model=FrontendConfig)
def frontend_config() -> dict:
    """前端啟動時讀一次，拿到所有「怎麼接」的資訊。"""
    base = _stream_base()
    return {
        "schema_version": "1.0",

        # ── 影像串流 ──────────────────────────────────
        "streaming": {
            # go2rtc 的對外 base URL。前端用它組 WebSocket/HLS 位址。
            "base_url": base,
            # ⚠️ 監看要看「打了 marker 的那條」，不是原始串流 ——
            # marker 是幀級配對的鍵，原始串流上沒有。
            "view_suffix": "_marked",
            # go2rtc 的 MSE 端點格式（{src} 由前端代入 camera_id + view_suffix）
            "mse_url_template": f"{base.replace('http', 'ws', 1)}/api/ws?src={{src}}" if base else "",
            # 建議的傳輸方式。MSE 不需要 ICE、不需要額外 port，
            # 多路 grid 時比 WebRTC 穩
            "preferred_transport": "mse",
        },

        # ── 偵測結果推播 ──────────────────────────────
        "results": {
            "websocket_path": "/ws",
            # 結果比這個舊就不該再沿用。由推論頻率推導，見下方 note。
            "max_age_ms": _max_age_ms(),
            "note": "WebSocket 是廣播，前端要自己用 camera_id 過濾",
        },

        # ── 疊加模式：告訴前端「框是誰畫的」──────────────
        # server: 框已燒進影像（nvdsosd），前端只要播，不用疊加也不用配對
        # client: 影像乾淨，前端要自己疊框（需要 marker 或時間戳配對）
        "overlay": {
            "mode": os.environ.get("OVERLAY_MODE", "server"),
            "note": "server = 框已燒在影像裡，前端只要播放；"
                    "client = 前端需自行疊加（此時才需要下方的 marker 規格）",
        },

        # ── marker 規格：只有 overlay.mode = client 時才需要 ────
        # 這是本端點最重要的部分。先前這些常數只存在於 index.html，
        # 外來前端得逐位元逆向工程；規格一改就得跟著改版。
        "marker": {
            "enabled": os.environ.get("OVERLAY_MODE", "server") == "client",
            "position": "bottom-right",
            "sync_pattern": marker.SYNC_PATTERN,
            "sync_bits": marker.SYNC_BITS,
            "id_bits": marker.ID_BITS,
            "parity_bits": marker.PARITY_BITS,
            "total_cells": marker.TOTAL_BITS,
            "cell_px": marker.CELL,
            "guard_px": marker.GUARD,
            "width_px": marker.MARKER_WIDTH,
            "height_px": marker.MARKER_HEIGHT,
            "gray_coded": True,
            "id_wraps_at": 1 << marker.ID_BITS,
            "decode_hints": {
                # 這三條都是實測踩出來的，不寫出來前端會重踩一次
                "sample_center_fraction": 1 / 3,
                "adaptive_threshold": "用 sync 格的黑白電平算門檻，不要寫死 127.5",
                "disable_image_smoothing": "canvas 的 imageSmoothingEnabled 預設 true，"
                                           "會把 marker 插值糊掉造成間歇性解碼失敗",
            },
        },
    }


def _max_age_ms() -> int:
    """結果的有效期：由推論頻率推導，不要讓前端硬編。

    取所有啟用線路裡最慢的 target_fps —— 前端是廣播接收，
    用最慢的那條當門檻才不會讓慢的線路一直沒框。
    """
    default = 700
    if _db is None:
        return default
    try:
        with _db.cursor() as cur:
            cur.execute(
                "SELECT min((line_params->>'target_fps')::float) "
                "FROM lines WHERE enabled = TRUE "
                "  AND line_params ? 'target_fps'"
            )
            row = cur.fetchone()
            fps = row[0] if row else None
        if fps and fps > 0:
            # 給 5 幀的餘裕，且不低於 500ms。
            #
            # 實測（15fps、兩台相機）：單台結果間隔中位 94ms、最大 293ms。
            # 「幀數餘裕」不夠用 —— 兩台相機在 Redis stream 交錯寫入，
            # 單台的實際間隔遠大於 1/fps，抖動時還會更久。
            # 下限 500ms 是實測最大值（293ms）的 1.7 倍。
            return max(500, int(5000 / fps))
    except Exception:
        logger.debug("推導 max_age_ms 失敗，用預設值", exc_info=True)
    return default


@router.get("/cameras", response_model=list[CameraOut])
def cameras() -> list[dict]:
    """相機清單：前端要顯示什麼、怎麼取得畫面，都在這裡。

    ⚠️ 取代舊的 /api/cameras —— 舊版讀進程內的 _services，
    獨立部署（output-api）時永遠回空 dict。這版從 DB 讀（資料庫是真相來源）。
    """
    if _db is None:
        raise HTTPException(503, "資料庫未連線")
    base = _stream_base()
    with _db.cursor() as cur:
        cur.execute(
            "SELECT c.id, c.name, c.line_id, l.station, c.width, c.height, "
            "       c.enabled, c.station_params, "
            "       s.apply_state, "
            "       EXTRACT(EPOCH FROM (now() - s.heartbeat_at))::int "
            "FROM cameras c "
            "JOIN lines l ON l.id = c.line_id "
            "LEFT JOIN line_status s ON s.line_id = c.line_id "
            # 線路停用的相機不列（B1：以前只看相機自己的 enabled，停用線的相機也列出來、view_url 指向不存在的串流）
            "WHERE c.enabled = TRUE AND l.enabled = TRUE ORDER BY c.id"
        )
        out = []
        for r in cur.fetchall():
            cam_id, params = r[0], (r[7] or {})
            hb = r[9]
            out.append({
                "id": cam_id,
                "name": r[1] or cam_id,
                "line_id": r[2],
                "station": r[3],
                "width": r[4],
                "height": r[5],
                # 線路失聯（心跳過期）時相機視為不可用
                "state": "online" if (hb is not None and hb < 60
                                      and r[8] == "ok") else "unknown",
                # ⭐ 串流位址由後端組，前端不用知道 _marked 這個慣例
                "view_stream": f"{cam_id}_{'marked'}",
                "view_url": (f"{base.replace('http', 'ws', 1)}/api/ws?src={cam_id}_marked"
                             if base else ""),
                # 現場校正值：前端畫基準四邊形用（正規化 0~1）
                "station_params": params,
            })
    return out
