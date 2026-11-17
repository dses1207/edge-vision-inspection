"""API 服務：對外提供 WebSocket 結果推送、健康檢查、相機設定管理與前端頁面。

⭐ 兩種部署形態，同一份程式：

  ① 同進程（Python 輕量路線）
     （已停用）舊的 run_mvp.py 曾直接 uvicorn.run(app)，推論服務用 register_service 注入，
     健康狀態直接從進程內物件讀。

  ② 獨立容器（DeepStream 正式路線）⭐
     推論跑在 DeepStream 容器裡，API 進程【引用不到】那些物件。
     此時健康狀態改從 DB 的 line_status 表讀（runner 定期回報），
     結果推送改從 Redis Streams 消費。

  兩者的差別只在「狀態從哪來」，端點與前端完全不用改。
  判斷方式：有沒有人呼叫過 register_service —— 有就是同進程，沒有就查 DB。
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from libs.contracts import FrameResult
from edgevision.inference.decoder import Frame

# 儲存層為選配：未設定連線時 API 仍可運作（只是沒有歷史查詢）
_db = None
_objects = None


def bind_storage(db=None, objects=None) -> None:
    """注入儲存層。由啟動腳本呼叫。"""
    global _db, _objects
    _db = db
    _objects = objects
    # 設定管理端點共用同一個 DB 連線池
    from edgevision.api import config_api, frontend_api

    config_api.bind_db(db)
    frontend_api.bind_db(db)

from edgevision.api.schemas import DetectionsPage, HealthOut, ImageUrlOut, ReviewIn, ReviewOut, StatsOut

logger = logging.getLogger(__name__)

WEB_DIR = Path(__file__).resolve().parents[2] / "web"


class WebSocketHub:
    """管理前端連線，把結果廣播出去。

    設計要點：發布端是推論執行緒（同步），消費端是事件迴圈（非同步），
    中間用 call_soon_threadsafe 銜接。廣播失敗只影響該連線，
    不可拋回推論端。
    """

    def __init__(self) -> None:
        self._clients: set[WebSocket] = set()
        self._loop: asyncio.AbstractEventLoop | None = None
        # 逐相機的基準四邊形：多相機時每台 ref_quad 不同，用 camera_id 分開存。
        self._ref_quads: dict[str, list[list[float]]] = {}

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    def set_ref_quad(self, camera_id: str, points: list[list[float]] | None) -> None:
        """基準四邊形，讓前端畫出來核對是否準確。逐相機存。"""
        if points:
            self._ref_quads[camera_id] = points
        else:
            self._ref_quads.pop(camera_id, None)

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        self._clients.add(ws)
        logger.info("前端已連線（目前 %d 個）", len(self._clients))
        # 一連上就先送各相機的基準（逐相機），前端依自己看的 cam 取用
        if self._ref_quads:
            # type 欄位讓前端不必靠「有沒有某個 key」來判別訊息種類 ——
            # 那在只有兩種訊息時能動，加第三種就會亂。
            await ws.send_text(json.dumps({
                "type": "ref_quads", "ref_quads": self._ref_quads,
            }))

    def disconnect(self, ws: WebSocket) -> None:
        self._clients.discard(ws)
        logger.info("前端已離線（剩餘 %d 個）", len(self._clients))

    # 由推論執行緒呼叫（同步）
    def publish(self, result: FrameResult, frame: Frame) -> None:
        if self._loop is None or not self._clients:
            return
        data = json.loads(result.model_dump_json())
        data["type"] = "result"
        ref = self._ref_quads.get(result.camera_id)
        if ref:
            data["ref_quad"] = ref
        payload = json.dumps(data)
        self._loop.call_soon_threadsafe(
            lambda: asyncio.create_task(self._broadcast(payload))
        )

    async def _broadcast(self, payload: str) -> None:
        """廣播給所有客戶端。

        併發送出而非逐一 await —— 逐一的話一個慢客戶端會拖累後面所有人，
        在 5fps 的節奏下累積成可見的間隔抖動（框會忽有忽無）。
        """
        clients = list(self._clients)
        if not clients:
            return
        results = await asyncio.gather(
            *(ws.send_text(payload) for ws in clients), return_exceptions=True
        )
        for ws, r in zip(clients, results):
            if isinstance(r, Exception):
                self._clients.discard(ws)

    @property
    def client_count(self) -> int:
        return len(self._clients)

    # ── 獨立部署模式用（結果來自 Redis 而非進程內推論）──────
    def ref_quad_of(self, camera_id: str | None) -> list | None:
        return self._ref_quads.get(camera_id) if camera_id else None

    async def broadcast_raw(self, payload: str) -> None:
        """直接廣播已序列化的結果。供 LiveConsumer 使用。

        與 publish() 的差別：publish 是同步的（推論執行緒呼叫，
        需要 call_soon_threadsafe 銜接），這個本來就在事件迴圈裡。
        """
        await self._broadcast(payload)


hub = WebSocketHub()
_services: dict[str, object] = {}


def _load_ref_quads_from_db() -> None:
    """從 cameras.station_params 載入各相機的 ref_quad，供前端畫基準四邊形。

    同進程模式由推論服務呼叫 set_ref_quad；獨立模式推論在別的容器，
    只好自己從設定的真相來源（資料庫）拿。
    """
    if _db is None:
        return
    try:
        with _db.cursor() as cur:
            cur.execute(
                "SELECT id, station_params FROM cameras "
                "WHERE enabled = TRUE AND station_params ? 'ref_quad'"
            )
            n = 0
            for cam_id, params in cur.fetchall():
                pts = (params or {}).get("ref_quad")
                if pts and len(pts) >= 4:
                    hub.set_ref_quad(cam_id, pts)
                    n += 1
        if n:
            logger.info("已從設定載入 %d 台相機的 ref_quad", n)
    except Exception:
        logger.warning("載入 ref_quad 失敗（前端將不顯示基準四邊形）", exc_info=True)


def register_service(camera_id: str, service) -> None:
    """⚠️ 同進程模式的注入點 —— 目前【沒有任何呼叫者】。

    唯一的使用者是舊的 run_mvp.py（PyAV 軟解 + ffmpeg 推流的 Python 管線），
    已於 2026-09-10 隨舊架構一起移除。現行部署一律是獨立模式：
    推論在 DeepStream 容器裡，結果經 Redis 過來。

    留著不刪是因為 _services 空的時候所有讀取點都會走「查 DB」那條分支，
    行為正確且零成本；真要拆掉會動到健康檢查與相機狀態共 5 處讀取，
    風險大於收益。若之後確定不會再有同進程模式，再一起清掉。
    """
    _services[camera_id] = service


_live_consumer = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _live_consumer
    hub.bind_loop(asyncio.get_running_loop())

    # 獨立部署模式：推論在別的容器，結果經 Redis 過來。
    # 同進程模式不需要（推論直接呼叫 hub.publish），但啟動它也無害 ——
    # 那時沒有別的進程在發 live 事件，consumer 只是閒置。
    from edgevision.api.live_consumer import LiveConsumer

    _live_consumer = LiveConsumer(hub)
    await _live_consumer.start()

    # 獨立模式下沒有推論服務會呼叫 set_ref_quad，改從 DB 的現場校正值載入
    if not _services:
        _load_ref_quads_from_db()

    logger.info("API 服務已就緒")
    yield

    if _live_consumer is not None:
        await _live_consumer.stop()
    for svc in _services.values():
        try:
            svc.stop()
        except Exception:
            logger.exception("停止服務時發生錯誤")


app = FastAPI(title="edgevision", lifespan=lifespan)


# ⚠️ 用 responses= 而非 response_model=：本端點回 JSONResponse 以自控 status code，
#    掛 response_model 會讓 FastAPI 重新序列化並【過濾掉沒宣告的欄位】。
#    responses= 只影響 OpenAPI schema，不碰執行路徑。
@app.get("/healthz", responses={200: {"model": HealthOut}})
def healthz() -> JSONResponse:
    """健康檢查。供 liveness probe 使用。

    同進程：直接讀推論服務的 health()。
    獨立部署：讀 line_status 表 —— runner 定期回報心跳與版本。
    """
    storage = {
        "database": _db.ping() if _db else None,
        "objects": _objects.ping() if _objects else None,
    }
    if _services:
        cams = {cid: svc.health() for cid, svc in _services.items()}
        ok = all(c["healthy"] for c in cams.values())
        body = {"mode": "in-process", "cameras": cams}
    else:
        lines = _line_status()
        # 沒有任何線路回報 ≠ API 壞了。API 本身活著就回 200，
        # 線路健康與否由 lines 欄位表達 —— 否則 API 會因為推論端的問題被重啟。
        ok = True
        # ⭐ services[]：消費者（event-processor / recorder / nas-recorder）的第二層健康。
        #    docker healthcheck 只答「活著嗎」，這裡答「在做事嗎」——
        #    line_c 死 20 天沒人發現、recorder 產出 25 GB 孤兒，都是這一層沒人看。
        from edgevision.api.config_api import HEARTBEAT_STALE_S
        from edgevision.storage.heartbeat import service_status
        body = {"mode": "standalone", "lines": lines,
                "services": service_status(_db, stale_after_s=HEARTBEAT_STALE_S)}
    return JSONResponse(
        {"healthy": ok, "clients": hub.client_count, "storage": storage, **body},
        status_code=200 if ok else 503,
    )


def camera_states(camera_ids: list[str], stats: dict, stale: bool) -> list[dict]:
    """相機層級狀態。純函式，可測。

    runner 心跳帶 stats.cameras = {相機: 這 15 秒的 fps}：
      ok         fps > 0
      no_frames  心跳正常、但這台 0 幀（相機 404、被相機踢線、斷線重連中）
      unknown    舊版 runner 沒回報這台（不告警）
      stale      整條線心跳過期 —— 由線路那一項告警，相機不重複叫
      skipped    這台設定有問題，runner 沒啟動它（stats.skipped_cameras 帶原因）；同線其他台照跑
    ⚠️ 為什麼要這層：線路心跳只代表 runner 活著。golf 相機一直 404、cam156 被踢線，
       線路層級照樣全綠（2026-09-29 正式機）。
    """
    fps = (stats or {}).get("cameras") or {}
    skipped = (stats or {}).get("skipped_cameras") or {}
    out = []
    for cid in camera_ids:
        if stale:
            state, v = "stale", None
        elif cid in skipped:
            out.append({"camera_id": cid, "state": "skipped", "fps": None, "reason": skipped[cid]})
            continue
        elif cid not in fps:
            state, v = "unknown", None
        else:
            v = fps[cid]
            state = "ok" if v and v > 0 else "no_frames"
        out.append({"camera_id": cid, "state": state, "fps": v})
    return out


def _line_status() -> list[dict]:
    """從 DB 讀各線路的 desired vs reported —— 這是抽離後的狀態來源。

    ⚠️ 心跳過期不等於「這條線掛了」：共用依賴（DB／Triton）抖動會讓所有線
    同時看似失聯。這裡只如實呈現，是否重啟由人或更上層判斷。
    """
    if _db is None:
        return []
    try:
        with _db.cursor() as cur:
            cur.execute(
                "SELECT l.id, l.name, l.station, l.config_version, "
                "       s.reported_version, s.apply_state, s.apply_error, "
                "       s.cameras_running, "
                "       EXTRACT(EPOCH FROM (now() - s.heartbeat_at))::int, s.stats "
                "FROM lines l LEFT JOIN line_status s ON s.line_id = l.id "
                "WHERE l.enabled = TRUE ORDER BY l.id"
            )
            rows = cur.fetchall()
            cur.execute("SELECT line_id, id FROM cameras WHERE enabled = TRUE ORDER BY id")
            cams_by_line: dict[str, list[str]] = {}
            for lid, cid in cur.fetchall():
                cams_by_line.setdefault(lid, []).append(cid)
            # 心跳過期的判定與 config_api 共用同一條線，兩個端點不能互相矛盾
            from edgevision.api.config_api import _is_stale

            out = []
            for r in rows:
                desired, reported = r[3], r[4]
                stale = _is_stale(r[8])
                stats = r[9] or {}
                out.append({
                    "line_id": r[0], "name": r[1], "station": r[2],
                    "desired_version": desired,
                    "reported_version": reported,
                    # ⭐ 這就是「改了但還沒生效」的可見化
                    # 心跳斷了就不算同步 —— 那個版本號只是舊快照
                    "in_sync": (not stale) and reported is not None and reported >= desired,
                    "apply_state": "stale" if stale else (r[5] or "unknown"),
                    "apply_error": r[6],
                    "cameras_running": 0 if stale else r[7],
                    "heartbeat_age_s": r[8],
                    "stale": stale,
                    "fps": None if stale else stats.get("fps"),
                    "cameras": camera_states(cams_by_line.get(r[0], []), stats, stale),
                })
            return out
    except Exception:
        logger.exception("讀取線路狀態失敗")
        return []


@app.get("/api/cameras/in-process")
def cameras_in_process() -> dict:
    """同進程模式的相機狀態（讀 register_service 注入的物件）。

    ⚠️ 獨立部署時會回空 —— 正式的相機清單請用 /api/cameras（frontend_api），
    那個從 DB 讀，兩種部署形態都可用。這支保留給本機除錯。
    """
    return {
        cid: {
            "health": svc.health(),
            "latest": (
                svc.latest_result.model_dump() if svc.latest_result else None
            ),
        }
        for cid, svc in _services.items()
    }


@app.websocket("/ws")
async def ws_results(ws: WebSocket) -> None:
    await hub.connect(ws)
    try:
        while True:
            # 前端目前不送訊息，這裡只是維持連線並偵測斷線
            await ws.receive_text()
    except WebSocketDisconnect:
        pass
    except Exception:
        logger.debug("WebSocket 連線異常結束", exc_info=True)
    finally:
        hub.disconnect(ws)


DETECTIONS_LIMIT_MAX = 500


@app.get("/api/detections", responses={200: {"model": DetectionsPage}})
def list_detections(
    camera_id: str | None = None,
    verdict: str | None = None,
    station: str | None = None,
    line_id: str | None = None,
    reviewed: bool | None = None,
    limit: int = 100,
) -> JSONResponse:
    """查詢歷史偵測紀錄（新到舊）。

    - camera_id / line_id：只看某台相機 / 某條線的相機
    - station：只看某一站的判定（一條線掛多站時用，例如只看 PPE）
    - verdict=ng：只看異常；reviewed=false：只看還沒審核的
    - limit：預設 100、**上限 500**（超過會被夾到 500）。要看更多請縮小範圍（line_id / camera_id）
    """
    if _db is None:
        return JSONResponse({"error": "未設定資料庫"}, status_code=503)
    limit = max(1, min(limit, DETECTIONS_LIMIT_MAX))
    rows = _db.recent_detections(camera_id=camera_id, verdict=verdict, station=station,
                                 line_id=line_id, reviewed=reviewed, limit=limit)
    for r in rows:
        for k in ("ts", "evidence_expired_at", "reviewed_at"):
            if r.get(k) is not None:
                r[k] = r[k].isoformat()
    return JSONResponse({"count": len(rows), "detections": rows})


@app.post("/api/detections/review", responses={200: {"model": ReviewOut}})
def review_detection(body: ReviewIn) -> JSONResponse:
    """人工審核一個 NG 事件：確認是真異常（true_positive）或誤報（false_positive）。

    以「幀」為單位（camera_id + frame_id）：同一幀的每一列一起標。可重複標，以最後一次為準。
    找不到這一幀的 NG → 404。
    """
    if _db is None:
        return JSONResponse({"error": "未設定資料庫"}, status_code=503)
    n = _db.review(camera_id=body.camera_id, frame_id=body.frame_id, result=body.result, actor=body.actor)
    if n == 0:
        return JSONResponse({"error": f"找不到 {body.camera_id} 第 {body.frame_id} 幀的 NG 紀錄"}, status_code=404)
    return JSONResponse({"camera_id": body.camera_id, "frame_id": body.frame_id, "result": body.result, "rows": n})


@app.get("/api/stats", responses={200: {"model": StatsOut}})
def stats(hours: int = 24) -> JSONResponse:
    if _db is None:
        return JSONResponse({"error": "未設定資料庫"}, status_code=503)
    return JSONResponse(_db.stats_summary(hours=max(1, min(hours, 720))))


@app.get("/api/image", responses={200: {"model": ImageUrlOut}})
def image_url(uri: str) -> JSONResponse:
    """把 s3:// URI 換成可直接下載的臨時連結。

    讓前端直接向 MinIO 取檔，不經過 API server 轉手 ——
    大檔案不會佔用 API 的頻寬與記憶體。
    """
    if _objects is None:
        return JSONResponse({"error": "未設定物件儲存"}, status_code=503)
    try:
        bucket, key = _objects.parse_uri(uri)
        return JSONResponse({"url": _objects.presigned_url(bucket=bucket, key=key)})
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)


def monitor_ui_enabled() -> bool:
    """內建監看頁開關（MONITOR_UI_ENABLED，預設開）。

    測試叢集讓真正的前端測試時關掉（2026-10-05）：每開一個監看頁就多一條 go2rtc 影像連線，
    開發用頁面跟正式前端搶 go2rtc 資源沒有意義。關掉只影響 / 與 /static，API、/healthz 照常。
    """
    return os.environ.get("MONITOR_UI_ENABLED", "true").lower() not in ("0", "false", "no", "off")


@app.get("/", response_model=None)
def index() -> FileResponse | JSONResponse:
    if not monitor_ui_enabled():
        return JSONResponse({"message": "內建監看頁已關閉（MONITOR_UI_ENABLED=false）；API 照常，文件在 /docs"},
                            status_code=404)
    # no-cache：監看頁很小，不值得快取；且快取會害使用者載到舊版前端
    # （曾發生：改了傳輸方式，但某台瀏覽器吃快取仍跑舊版，行為不一致難除錯）。
    return FileResponse(
        WEB_DIR / "index.html",
        headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
    )


if WEB_DIR.exists() and monitor_ui_enabled():
    app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")


# ── 掛載子路由與中介層 ──────────────────────────────────────
# 掛在檔尾：確保 app 與所有既有端點都已定義完成。

from edgevision.api.config_api import router as _config_router  # noqa: E402
from edgevision.api.frontend_api import router as _frontend_router  # noqa: E402

app.include_router(_config_router)
app.include_router(_frontend_router)

# CORS：與控制 API 共用設定，見 edgevision/api/cors.py（CORS_ORIGINS=* 不限來源 / 逗號白名單 / 沒設只限同源）
from edgevision.api.cors import apply_cors  # noqa: E402

apply_cors(app, "推論 API")
