"""PTZ 的資料存取：相機（cameras 表）與座標（ptz_positions）。"""

from __future__ import annotations

from dataclasses import dataclass

from edgevision.ptz.settings import DEFAULT_ONVIF_PORT


@dataclass(slots=True)
class PtzCamera:
    id: str
    ip: str
    onvif_port: int
    cred_ref: str
    name: str = ""
    line_id: str = ""
    can_pan: bool = False

    def credentials(self) -> tuple[str, str]:
        from edgevision.config.camera_source import resolve_credentials
        try:
            return resolve_credentials(self.cred_ref)
        except SystemExit as exc:      # runner 缺帳密就該停機；PTZ 是 API，回清楚的錯誤就好（不能讓 SystemExit 變 500）
            raise RuntimeError(str(exc)) from None


KINDS = ("target", "origin", "reference")
_CAM_COLS = "id, ip, onvif_port, cred_ref, name, line_id, can_pan"


def _camera(r) -> PtzCamera:
    return PtzCamera(id=r[0], ip=r[1], onvif_port=r[2] or DEFAULT_ONVIF_PORT, cred_ref=r[3] or "",
                     name=r[4] or "", line_id=r[5] or "", can_pan=bool(r[6]))


class PtzStore:
    def __init__(self, db) -> None:
        self.db = db

    def camera(self, cam_id: str) -> PtzCamera | None:
        with self.db.cursor() as cur:
            cur.execute(f"SELECT {_CAM_COLS} FROM cameras WHERE id = %s", (cam_id,))
            r = cur.fetchone()
        return _camera(r) if r else None

    def cameras(self) -> list[PtzCamera]:
        with self.db.cursor() as cur:
            cur.execute(f"SELECT {_CAM_COLS} FROM cameras ORDER BY id")
            return [_camera(r) for r in cur.fetchall()]

    def set_can_pan(self, cam_id: str, can_pan: bool) -> None:
        # 不動 updated_at：那是線路設定的修改時間；誰改的記在稽核
        with self.db.cursor() as cur:
            cur.execute("UPDATE cameras SET can_pan = %s WHERE id = %s", (can_pan, cam_id))

    def position(self, cam_id: str, kind: str) -> dict | None:
        with self.db.cursor() as cur:
            cur.execute("SELECT pan, tilt, zoom, has_coord, ref_object_key, updated_by, updated_at"
                        " FROM ptz_positions WHERE camera_id = %s AND kind = %s", (cam_id, kind))
            r = cur.fetchone()
        if not r:
            return None
        return {"pan": r[0], "tilt": r[1], "zoom": r[2] or 0.0, "has_coord": r[3], "ref_object_key": r[4],
                "updated_by": r[5], "updated_at": r[6]}

    def upsert(self, cam_id: str, kind: str, *, pan: float, tilt: float, zoom: float = 0.0,
               has_coord: bool = True, ref_object_key: str | None = None, actor: str | None = None) -> None:
        with self.db.cursor() as cur:
            cur.execute(
                """INSERT INTO ptz_positions (camera_id, kind, pan, tilt, zoom, has_coord, ref_object_key, updated_by)
                   VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
                   ON CONFLICT (camera_id, kind) DO UPDATE SET pan = EXCLUDED.pan, tilt = EXCLUDED.tilt,
                     zoom = EXCLUDED.zoom, has_coord = EXCLUDED.has_coord,
                     ref_object_key = COALESCE(EXCLUDED.ref_object_key, ptz_positions.ref_object_key),
                     updated_by = EXCLUDED.updated_by, updated_at = now()""",
                (cam_id, kind, pan, tilt, zoom, has_coord, ref_object_key, actor))

    def cameras_with(self, kind: str) -> list[str]:
        with self.db.cursor() as cur:
            cur.execute("SELECT camera_id FROM ptz_positions WHERE kind = %s ORDER BY camera_id", (kind,))
            return [r[0] for r in cur.fetchall()]

    def all_positions(self) -> dict[str, dict[str, dict]]:
        out: dict[str, dict[str, dict]] = {}
        with self.db.cursor() as cur:
            cur.execute("SELECT camera_id, kind, pan, tilt, zoom, has_coord, updated_by, updated_at FROM ptz_positions")
            for cam, kind, pan, tilt, zoom, hc, by, at in cur.fetchall():
                out.setdefault(cam, {})[kind] = {"pan": pan, "tilt": tilt, "zoom": zoom, "has_coord": hc,
                                                 "updated_by": by, "updated_at": at}
        return out
