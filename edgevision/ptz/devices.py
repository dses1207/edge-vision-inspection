"""相機控制：ONVIF（移動 / 座標 / preset）與 Tapo 本機 API（馬達校準）。

ONVIF 連線第一次用時建立並快取（建連線要抓 WSDL、GetProfiles，約 1–2 秒；冷連線的第一個 move 會慢、會亂序）。
"""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass

from edgevision.ptz.settings import CONTINUOUS_PAN_SIGN
from edgevision.ptz.store import PtzCamera

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class Position:
    pan: float
    tilt: float
    zoom: float
    has_pan_tilt: bool = True


class OnvifDevice:
    """一台相機的 ONVIF 連線（onvif-zeep）。"""

    def __init__(self, cam: PtzCamera) -> None:
        from onvif import ONVIFCamera

        user, pw = cam.credentials()
        self.cam = cam
        self._camera = ONVIFCamera(cam.ip, cam.onvif_port, user, pw)
        self.media = self._camera.create_media_service()
        self.ptz = self._camera.create_ptz_service()
        self.profiles = self.media.GetProfiles()
        self.token = self.profiles[0].token

    def status(self) -> Position:
        pos = self.ptz.GetStatus({"ProfileToken": self.token}).Position
        pt, z = pos.PanTilt, pos.Zoom
        return Position(pan=float(pt.x) if pt is not None else 0.0, tilt=float(pt.y) if pt is not None else 0.0,
                        zoom=float(z.x) if z is not None else 0.0, has_pan_tilt=pt is not None)

    def continuous_move(self, pan: float, tilt: float, zoom: float) -> None:
        req = self.ptz.create_type("ContinuousMove")
        req.ProfileToken = self.token
        req.Velocity = {"PanTilt": {"x": pan * CONTINUOUS_PAN_SIGN, "y": tilt}, "Zoom": {"x": zoom}}
        self.ptz.ContinuousMove(req)

    def stop(self) -> None:
        req = self.ptz.create_type("Stop")
        req.ProfileToken = self.token
        req.PanTilt = True
        req.Zoom = True
        self.ptz.Stop(req)

    def relative_move(self, dx: float, dy: float, dz: float = 0.0) -> None:
        """固定步進，鏡頭自己會停。本批 TP-Link 不支援 Zoom 相對位移。"""
        req = self.ptz.create_type("RelativeMove")
        req.ProfileToken = self.token
        req.Translation = {"PanTilt": {"x": dx, "y": dy}, "Zoom": {"x": dz}}
        self.ptz.RelativeMove(req)

    def absolute_move(self, pan: float, tilt: float, zoom: float) -> None:
        req = self.ptz.create_type("AbsoluteMove")
        req.ProfileToken = self.token
        req.Position = {"PanTilt": {"x": pan, "y": tilt}, "Zoom": {"x": zoom}}
        self.ptz.AbsoluteMove(req)

    def presets(self) -> list[dict]:
        return [{"index": i + 1, "token": p.token, "name": getattr(p, "Name", None) or p.token}
                for i, p in enumerate(self.ptz.GetPresets({"ProfileToken": self.token}) or [])]

    def set_preset(self, name: str) -> str:
        return str(self.ptz.SetPreset({"ProfileToken": self.token, "PresetName": name}))

    def goto_preset(self, index: int) -> str:
        presets = self.ptz.GetPresets({"ProfileToken": self.token}) or []
        if not 1 <= index <= len(presets):
            raise LookupError(f"找不到預設點 {index}（共 {len(presets)} 個）")
        tok = presets[index - 1].token
        self.ptz.GotoPreset({"ProfileToken": self.token, "PresetToken": tok})
        return tok


class DevicePool:
    """ONVIF 連線快取（每台一條，建立後重用）。"""

    def __init__(self, store) -> None:
        self.store = store
        self._devs: dict[str, OnvifDevice] = {}
        self._lock = threading.Lock()

    def get(self, cam_id: str) -> OnvifDevice:
        with self._lock:
            d = self._devs.get(cam_id)
            if d is not None:
                return d
        cam = self.store.camera(cam_id)
        if cam is None:
            raise LookupError(f"未知的攝影機：{cam_id}")
        d = OnvifDevice(cam)
        with self._lock:
            self._devs[cam_id] = d
        return d

    def drop(self, cam_id: str) -> None:
        """連線出錯時丟掉快取，下次重建（相機重開機後舊 session 會失效）。"""
        with self._lock:
            self._devs.pop(cam_id, None)


def tapo_secret() -> tuple[tuple[str, str] | None, str]:
    """Tapo 本機 API（443）帳密：TP-Link 雲端帳號，不一定等於 RTSP / ONVIF 帳密。K8s Secret ptz-tapo 掛成環境變數。
    ⚠️ 連續登入失敗會被鎖約 30 分鐘（Temporary Suspension）—— 帳密要先確認正確再用。"""
    u, p = os.environ.get("TAPO_USERNAME", ""), os.environ.get("TAPO_PASSWORD", "")
    return ((u, p) if u and p else None), os.environ.get("TAPO_CLOUD_PASSWORD", "")


def calibrate_motor(cam: PtzCamera) -> None:
    """觸發相機原生馬達校準（manual_cali）：相機自己掃描上下左右再回中心，把座標重新錨定（約 20–40 秒）。
    被外力扳動、座標漂掉後唯一可靠的重置。成功即回、失敗 raise。"""
    from pytapo import Tapo

    account, cloud = tapo_secret()
    user, pw = account or cam.credentials()
    Tapo(cam.ip, user, pw, cloudPassword=cloud).calibrateMotor()
    logger.info("[校準] %s 已觸發 manual_cali（相機掃描中）", cam.id)


def detect_pan_tilt(cam: PtzCamera) -> bool:
    """ONVIF 唯讀查相機有沒有雲台（GetNodes 有 PanTilt 空間）；不移動相機。連不上 ONVIF → False（固定鏡頭多半沒開 ONVIF）。"""
    import socket

    from onvif import ONVIFCamera

    try:            # 固定鏡頭多半根本沒開 ONVIF 埠；先短連線試一下，不然 onvif-zeep 會重試到一分多鐘
        socket.create_connection((cam.ip, cam.onvif_port), timeout=3).close()
    except OSError:
        return False
    try:
        user, pw = cam.credentials()
        nodes = ONVIFCamera(cam.ip, cam.onvif_port, user, pw).create_ptz_service().GetNodes()
    except Exception as exc:  # noqa: BLE001 —— 各種連線 / SOAP 錯誤都當作沒有 ONVIF 雲台
        logger.info("[偵測] %s ONVIF 連不上，當作不能轉：%s", cam.id, exc)
        return False
    return any(getattr(n.SupportedPTZSpaces, "ContinuousPanTiltVelocitySpace", None)
               or getattr(n.SupportedPTZSpaces, "AbsolutePanTiltPositionSpace", None) for n in nodes or [])
