"""PTZ API（獨立服務 ptz，Ingress /api/ptz）。舊系統的 PTZ 服務的端點全部保留，改成 REST 路徑。

    GET  /api/ptz/cameras                       相機清單（can_pan = 能不能轉）+ 各自的歸位點 / 中心點 / 基準 / 失準狀態
    PUT  /api/ptz/{cam}/can-pan                 設定能不能轉（null = ONVIF 唯讀自動偵測）
    GET  /api/ptz/{cam}/status                  目前座標
    POST /api/ptz/{cam}/move                    連續移動（2.5 秒看門狗：前端長按每 1.5 秒重送一次）
    POST /api/ptz/{cam}/stop
    POST /api/ptz/{cam}/relative_move           固定步進（鏡頭自己停）
    POST /api/ptz/{cam}/absolute_move
    GET  /api/ptz/{cam}/presets                 相機內建 preset
    POST /api/ptz/{cam}/presets                 存目前位置成 preset
    POST /api/ptz/{cam}/presets/{index}/goto
    GET|POST /api/ptz/{cam}/target              歸位點：讀 / 存目前座標（排程每 10 分鐘回這裡）
    POST /api/ptz/{cam}/target/goto
    GET|POST /api/ptz/{cam}/origin              軟體中心點
    POST /api/ptz/{cam}/origin/goto             （舊系統的 goto_center）
    POST /api/ptz/{cam}/reference               存失準偵測基準（目前畫面 + 座標）
    GET  /api/ptz/{cam}/drift                   失準狀態
    POST /api/ptz/{cam}/drift/check             只量測，不移動（force=true 忽略跳過條件）
    POST /api/ptz/{cam}/drift/recover           立即回正（背景執行，進度看 GET drift）
    POST /api/ptz/{cam}/calibrate_motor         Tapo 原生馬達校準（掃描 20–40 秒）
    GET  /api/ptz/{cam}/snapshot                目前畫面（JPEG，經 go2rtc）

會改相機設定的動作（preset、歸位點、中心點、基準、校準、回正）寫 audit_log；手動移動太頻繁，不寫。
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import FastAPI, Header, HTTPException, Query
from fastapi.responses import Response
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

_state: dict = {}


class Vector(BaseModel):
    pan: float = Field(0.0, ge=-1, le=1)
    tilt: float = Field(0.0, ge=-1, le=1)
    zoom: float = Field(0.0, ge=-1, le=1)


class PositionOut(BaseModel):
    pan: float | None
    tilt: float | None
    zoom: float | None = None


class StoredPositionOut(PositionOut):
    has_coord: bool | None = None
    updated_by: str | None = None
    updated_at: datetime | None = None


class StatusOut(PositionOut):
    camera_id: str


class OkOut(BaseModel):
    ok: bool
    message: str | None = None


class PresetOut(BaseModel):
    index: int
    token: str
    name: str


class PresetIn(BaseModel):
    name: str = Field(min_length=1, max_length=60)


class ReferenceOut(BaseModel):
    pan: float | None
    tilt: float | None
    ref_object_key: str


class DriftOut(BaseModel):
    camera_id: str
    monitored: bool = Field(description="有基準 = 在失準偵測範圍內")
    recovering: bool
    detail: dict = Field(description="count / state / dx / dy / shift / conf / ts")


class DriftCheckOut(BaseModel):
    ok: bool
    reason: str | None = None
    skipped: str | None = None
    dx: int | None = None
    dy: int | None = None
    shift: float | None = None
    conf: float | None = None
    drift: bool | None = None


class CameraOut(BaseModel):
    id: str
    name: str
    line_id: str
    ip: str
    onvif_port: int
    can_pan: bool = Field(False, description="有沒有雲台（能不能轉）；固定鏡頭 false，前端不要顯示控制")
    target: StoredPositionOut | None = None
    origin: StoredPositionOut | None = None
    reference: StoredPositionOut | None = None
    drift: dict | None = None
    recovering: bool = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    from edgevision.api import auth as _auth
    from edgevision.ptz.service import PtzService
    from edgevision.ptz.store import PtzStore
    from edgevision.storage.db import Database
    from edgevision.storage.objects import ObjectStore

    _auth.load_jwks_at_startup()
    db = Database()
    svc = PtzService(PtzStore(db), ObjectStore())
    _state.update(db=db, svc=svc)
    # ⚠️ 兩個叢集若控制【同一批實體相機】：兩邊都跑背景迴圈（定時歸位、自動回正）會互搶。
    #    只讓一邊開；另一邊 overlay 設 PTZ_BACKGROUND=false（手動 API 照常）
    if os.environ.get("PTZ_BACKGROUND", "true").lower() in ("1", "true", "yes"):
        svc.start_background()
    else:
        logger.warning("PTZ_BACKGROUND=false：不跑定時歸位與失準偵測（只有手動 API）")
    yield
    svc.shutdown()
    db.close()


app = FastAPI(title="edgevision PTZ", lifespan=lifespan,
              openapi_url="/api/ptz/openapi.json", docs_url="/api/ptz/docs", redoc_url=None)

from edgevision.api.cors import apply_cors  # noqa: E402

apply_cors(app, "PTZ API")


def _svc():
    return _state["svc"]


def _actor(x_actor, x_actor_type, x_channel, authorization) -> tuple[str, str, str]:
    from edgevision.api import auth as _auth
    try:
        ident, ch = _auth.resolve(authorization, x_actor, x_actor_type, x_channel)
    except ValueError as exc:
        raise HTTPException(400 if "X-Channel" in str(exc) else 401, str(exc))
    return ident.actor, ident.actor_type, ch


def _audit(ident, action: str, cam: str, params: dict, result: str = "success", error: str | None = None):
    try:
        _state["db"].write_audit(actor=ident[0], actor_type=ident[1], channel=ident[2], action=action,
                                 target=cam, params=params, result=result, error=error)
    except Exception:
        logger.exception("audit 寫入失敗")


def _do(fn, *a, **kw):
    """相機 / 裝置錯誤 → HTTP：未知相機或沒設定 404、相機連不上 502。"""
    try:
        return fn(*a, **kw)
    except HTTPException:
        raise
    except LookupError as exc:
        raise HTTPException(404, str(exc)) from exc
    except ImportError as exc:
        raise HTTPException(501, f"缺套件：{exc}") from exc
    except Exception as exc:
        logger.warning("PTZ 呼叫失敗：%s", exc)
        raise HTTPException(502, f"相機操作失敗：{exc}") from exc


def _cam(cam: str):
    c = _svc().store.camera(cam)
    if c is None:
        raise HTTPException(404, f"未知的攝影機：{cam}")
    return c


HDR = dict(x_actor=Header(default=None), x_actor_type=Header(default=None), x_channel=Header(default=None),
           authorization=Header(default=None))


@app.get("/healthz")
def healthz():
    return {"ok": True, "db": _state["db"].ping() if "db" in _state else False}


@app.get("/api/ptz/cameras", response_model=list[CameraOut])
def cameras():
    svc = _svc()
    pos = svc.store.all_positions()
    out = []
    for c in svc.store.cameras():
        p = pos.get(c.id, {})
        out.append(CameraOut(id=c.id, name=c.name, line_id=c.line_id, ip=c.ip, onvif_port=c.onvif_port,
                             can_pan=c.can_pan,
                             target=p.get("target"), origin=p.get("origin"), reference=p.get("reference"),
                             drift=svc.drift_state.get(c.id), recovering=c.id in svc.calibrating))
    return out


class CanPanIn(BaseModel):
    can_pan: bool | None = Field(None, description="true / false 直接設；null = 用 ONVIF 唯讀自動偵測（不會轉動相機）")


class CanPanOut(BaseModel):
    id: str
    can_pan: bool
    detected: bool = Field(description="這次是不是自動偵測出來的")


@app.put("/api/ptz/{cam}/can-pan", response_model=CanPanOut)
def set_can_pan(cam: str, body: CanPanIn, x_actor: str | None = HDR["x_actor"],
                x_actor_type: str | None = HDR["x_actor_type"], x_channel: str | None = HDR["x_channel"],
                authorization: str | None = HDR["authorization"]):
    """設定這台相機能不能轉（新加的相機預設 false）。"""
    from edgevision.ptz.devices import detect_pan_tilt

    ident = _actor(x_actor, x_actor_type, x_channel, authorization)
    c = _cam(cam)
    value = body.can_pan if body.can_pan is not None else detect_pan_tilt(c)
    _svc().store.set_can_pan(cam, value)
    _audit(ident, "ptz.can_pan.set", cam, {"can_pan": value, "detected": body.can_pan is None})
    return {"id": cam, "can_pan": value, "detected": body.can_pan is None}


@app.get("/api/ptz/{cam}/status", response_model=StatusOut)
def status(cam: str):
    _cam(cam)
    p = _do(_svc().status, cam)
    return {"camera_id": cam, "pan": p.pan, "tilt": p.tilt, "zoom": p.zoom}


@app.post("/api/ptz/{cam}/move", response_model=OkOut)
def move(cam: str, v: Vector):
    _cam(cam)
    _do(_svc().move, cam, v.pan, v.tilt, v.zoom)
    return {"ok": True}


@app.post("/api/ptz/{cam}/stop", response_model=OkOut)
def stop(cam: str):
    _cam(cam)
    _do(_svc().stop, cam)
    return {"ok": True}


@app.post("/api/ptz/{cam}/relative_move", response_model=OkOut)
def relative_move(cam: str, v: Vector):
    _cam(cam)
    _do(_svc().relative_move, cam, v.pan, v.tilt, v.zoom)
    return {"ok": True}


@app.post("/api/ptz/{cam}/absolute_move", response_model=OkOut)
def absolute_move(cam: str, v: Vector):
    _cam(cam)
    _do(_svc().absolute_move, cam, v.pan, v.tilt, v.zoom)
    return {"ok": True}


@app.get("/api/ptz/{cam}/presets", response_model=list[PresetOut])
def presets(cam: str):
    _cam(cam)
    return _do(_svc().presets, cam)


@app.post("/api/ptz/{cam}/presets", response_model=OkOut)
def set_preset(cam: str, body: PresetIn, x_actor: str | None = HDR["x_actor"],
               x_actor_type: str | None = HDR["x_actor_type"], x_channel: str | None = HDR["x_channel"],
               authorization: str | None = HDR["authorization"]):
    ident = _actor(x_actor, x_actor_type, x_channel, authorization)
    _cam(cam)
    tok = _do(_svc().set_preset, cam, body.name)
    _audit(ident, "ptz.preset.set", cam, {"name": body.name, "token": tok})
    return {"ok": True, "message": tok}


@app.post("/api/ptz/{cam}/presets/{index}/goto", response_model=OkOut)
def goto_preset(cam: str, index: int):
    _cam(cam)
    tok = _do(_svc().goto_preset, cam, index)
    return {"ok": True, "message": tok}


def _get_pos(cam: str, kind: str):
    _cam(cam)
    p = _svc().store.position(cam, kind)
    if p is None:
        raise HTTPException(404, f"{cam} 尚未設定{'歸位點' if kind == 'target' else '中心點'}")
    return p


@app.get("/api/ptz/{cam}/target", response_model=StoredPositionOut)
def get_target(cam: str):
    return _get_pos(cam, "target")


@app.post("/api/ptz/{cam}/target", response_model=PositionOut)
def save_target(cam: str, x_actor: str | None = HDR["x_actor"], x_actor_type: str | None = HDR["x_actor_type"],
                x_channel: str | None = HDR["x_channel"], authorization: str | None = HDR["authorization"]):
    """把相機目前座標存成歸位點（排程每 10 分鐘強定位回這裡；自動回正的目的地）。"""
    ident = _actor(x_actor, x_actor_type, x_channel, authorization)
    _cam(cam)
    r = _do(_svc().save_position, cam, "target", actor=ident[0])
    _audit(ident, "ptz.target.save", cam, r)
    return r


@app.post("/api/ptz/{cam}/target/goto", response_model=PositionOut)
def goto_target(cam: str):
    _cam(cam)
    return _do(_svc().goto_position, cam, "target")


@app.get("/api/ptz/{cam}/origin", response_model=StoredPositionOut)
def get_origin(cam: str):
    return _get_pos(cam, "origin")


@app.post("/api/ptz/{cam}/origin", response_model=PositionOut)
def save_origin(cam: str, x_actor: str | None = HDR["x_actor"], x_actor_type: str | None = HDR["x_actor_type"],
                x_channel: str | None = HDR["x_channel"], authorization: str | None = HDR["authorization"]):
    ident = _actor(x_actor, x_actor_type, x_channel, authorization)
    _cam(cam)
    r = _do(_svc().save_position, cam, "origin", actor=ident[0])
    _audit(ident, "ptz.origin.save", cam, r)
    return r


@app.post("/api/ptz/{cam}/origin/goto", response_model=PositionOut)
def goto_origin(cam: str):
    _cam(cam)
    return _do(_svc().goto_position, cam, "origin")


@app.post("/api/ptz/{cam}/reference", response_model=ReferenceOut)
def save_reference(cam: str, x_actor: str | None = HDR["x_actor"], x_actor_type: str | None = HDR["x_actor_type"],
                   x_channel: str | None = HDR["x_channel"], authorization: str | None = HDR["authorization"]):
    """把目前畫面 + 座標存成失準偵測基準（請在相機位於正確歸位視角時做）。存了之後這台就進入自動監控。"""
    ident = _actor(x_actor, x_actor_type, x_channel, authorization)
    _cam(cam)
    r = _do(_svc().save_reference, cam, actor=ident[0])
    _audit(ident, "ptz.reference.save", cam, r)
    return r


@app.get("/api/ptz/{cam}/drift", response_model=DriftOut)
def drift(cam: str):
    _cam(cam)
    svc = _svc()
    return {"camera_id": cam, "monitored": svc.store.position(cam, "reference") is not None,
            "recovering": cam in svc.calibrating, "detail": dict(svc.drift_state.get(cam, {}))}


@app.post("/api/ptz/{cam}/drift/check", response_model=DriftCheckOut)
def drift_check(cam: str, force: bool = Query(False)):
    """只量測目前是否失準，不移動相機。"""
    _cam(cam)
    return _do(_svc().check_drift, cam, force=force)


@app.post("/api/ptz/{cam}/drift/recover", response_model=OkOut)
def drift_recover(cam: str, x_actor: str | None = HDR["x_actor"], x_actor_type: str | None = HDR["x_actor_type"],
                  x_channel: str | None = HDR["x_channel"], authorization: str | None = HDR["authorization"]):
    """立即回正（視覺對齊優先，對不上才原生校準）。背景執行，進度看 GET drift。"""
    ident = _actor(x_actor, x_actor_type, x_channel, authorization)
    _cam(cam)
    if _svc().store.position(cam, "reference") is None:
        raise HTTPException(400, "尚未設定監控基準（POST /api/ptz/{cam}/reference）")
    started = _svc().recover_async(cam)
    _audit(ident, "ptz.drift.recover", cam, {"started": started})
    return {"ok": True, "message": "已開始回正" if started else "已在回正中"}


@app.post("/api/ptz/{cam}/calibrate_motor", response_model=OkOut)
def calibrate(cam: str, x_actor: str | None = HDR["x_actor"], x_actor_type: str | None = HDR["x_actor_type"],
              x_channel: str | None = HDR["x_channel"], authorization: str | None = HDR["authorization"]):
    """Tapo 原生馬達校準：相機自己掃描上下左右再回中心（20–40 秒）。帳密錯連續幾次會被鎖約 30 分鐘。"""
    ident = _actor(x_actor, x_actor_type, x_channel, authorization)
    c = _cam(cam)
    svc = _svc()
    svc.mark_control(cam)
    try:
        svc._calibrate(c)
    except ImportError as exc:
        raise HTTPException(501, f"缺 pytapo：{exc}") from exc
    except Exception as exc:
        low = str(exc).lower()
        _audit(ident, "ptz.calibrate", cam, {}, result="failed", error=str(exc))
        if "authentication" in low or "suspension" in low or "invalid" in low:
            raise HTTPException(401, f"相機本機 API 認證失敗或暫時鎖定：{exc}") from exc
        raise HTTPException(502, f"校準失敗：{exc}") from exc
    _audit(ident, "ptz.calibrate", cam, {})
    return {"ok": True, "message": "已觸發校準，相機掃描中（約 20–40 秒後回中）"}


@app.get("/api/ptz/{cam}/snapshot", response_class=Response,
         responses={200: {"content": {"image/jpeg": {}}, "description": "目前畫面"}})
def snapshot(cam: str):
    from edgevision.ptz.vision import grab_jpeg
    _cam(cam)
    return Response(_do(grab_jpeg, cam), media_type="image/jpeg")
