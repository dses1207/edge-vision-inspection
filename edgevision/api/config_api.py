"""相機與線路設定的管理端點。

⭐ 這裡是「改設定」的唯一入口。每次變更做四件事，缺一不可：
    ① 寫表（desired）
    ② config_version += 1
    ③ 存【完整設定物件】的版本快照 —— 供還原
    ④ 寫 audit_log —— 供稽核

⚠️ ③ 和 ④ 是兩件事，不能互相取代：
    audit_log 記「誰改了什麼」，但 NetBox 的教訓是 before/after 的
    changelog【不足以還原一個 UPDATE】（有依賴關係時會重建出不同 ID）。
    所以要另存整個設定物件的完整 JSON。

⭐ schema 驗證在這一層做 —— 這正是選 API 而非 Git YAML 的實質優勢之一
   （GitOps 的已知限制：「沒有直接的方式驗證被 commit 的檔案」）。
   現場人員填 IP 與座標，這道防線在這裡。
"""

from __future__ import annotations

import ipaddress
import json
import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, field_validator

from edgevision.api.model_info import ModelInfo, enrich_line, prepare_line_for_save
from edgevision.api.schemas import LineSummary
from edgevision.api.station_specs import check_line, load_specs

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/config", tags=["config"])

# 由 server.py 注入（與 bind_storage 同一套）
_db = None


def bind_db(db) -> None:
    global _db
    _db = db


def _require_db():
    if _db is None:
        raise HTTPException(503, "資料庫未連線，設定管理不可用")
    return _db


# ── 輸入驗證 ────────────────────────────────────────────────
class CameraIn(BaseModel):
    id: str = Field(min_length=1, max_length=64)
    ip: str
    stream_path: str = ""
    width: int = Field(gt=0, le=8192)
    height: int = Field(gt=0, le=8192)
    cred_ref: str = ""
    name: str = ""
    enabled: bool = True
    station_params: dict = Field(default_factory=dict)
    # 主站只推論這塊（正規化 [x1, y1, x2, y2]）；None = 整幅。語意見 06-line-extras.sql / ai-infer config.py
    infer_roi: list[float] | None = None
    # 附加站的現場值 {站別: {...}}；主站仍用 station_params
    extra_params: dict[str, dict] = Field(default_factory=dict)

    @field_validator("infer_roi")
    @classmethod
    def _valid_roi(cls, v: list[float] | None) -> list[float] | None:
        # 同 runner 的檢查：填錯不會有症狀，模型就是看錯地方
        if v is not None and (len(v) != 4 or not all(0.0 <= x <= 1.0 for x in v)
                              or v[0] >= v[2] or v[1] >= v[3]):
            raise ValueError(f"infer_roi 必須是正規化 [x1, y1, x2, y2] 且 x1<x2、y1<y2：{v}")
        return v

    @field_validator("ip")
    @classmethod
    def _valid_ip(cls, v: str) -> str:
        try:
            ipaddress.ip_address(v)
        except ValueError:
            raise ValueError(f"不是合法的 IP 位址：{v}")
        return v

    @field_validator("station_params")
    @classmethod
    def _valid_params(cls, v: dict) -> dict:
        """現場校正值的座標要在畫面範圍內（正規化 0~1）。

        座標填錯不會有任何症狀 —— 判定就是默默算錯。這裡擋掉。
        """
        for key, pts in v.items():
            if not isinstance(pts, list):
                continue
            for p in pts:
                if isinstance(p, (list, tuple)) and len(p) >= 2:
                    x, y = p[0], p[1]
                    if not (0 <= x <= 1 and 0 <= y <= 1):
                        raise ValueError(
                            f"{key} 的座標 ({x}, {y}) 超出畫面範圍，"
                            "現場校正值須為正規化座標 0~1"
                        )
        return v


class LineIn(BaseModel):
    id: str = Field(min_length=1, max_length=64)
    name: str = ""
    station: str = Field(min_length=1)
    line_params: dict = Field(default_factory=dict)
    enabled: bool = True
    cameras: list[CameraIn] = Field(default_factory=list)
    # 主站模型 {name, conf, iou, hidden_classes}；name 不填 = 站別預設模型。classes / imgsz / task 是模型自己的屬性，不存（GET 時從 Registry 附上）
    model: dict | None = None
    # 附加站 [{station, model?, interval?}]：每個一個 nvinferserver（例：PPE）
    extra_stations: list[dict] = Field(default_factory=list)

    @field_validator("model")
    @classmethod
    def _valid_model(cls, v: dict | None) -> dict | None:
        if v is not None and not v.get("name"):
            raise ValueError("model 要有 name（Triton 上的模型名）")
        return v

    @field_validator("extra_stations")
    @classmethod
    def _valid_extra(cls, v: list[dict]) -> list[dict]:
        for e in v:
            if not isinstance(e, dict) or not e.get("station"):
                raise ValueError(f"extra_stations 每一項都要有 station：{e}")
        return v

    @field_validator("line_params")
    @classmethod
    def _valid_line_params(cls, v: dict) -> dict:
        tf, sf = v.get("target_fps"), v.get("source_fps")
        if tf is not None:
            if not (0 < float(tf) <= 120):
                raise ValueError(f"target_fps 超出合理範圍：{tf}")
            if sf is None:
                raise ValueError(
                    "設了 target_fps 就必須同時給 source_fps —— "
                    "跳幀間隔靠兩者相除，缺一個算不出來"
                )
        if sf is not None and not (0 < float(sf) <= 240):
            raise ValueError(f"source_fps 超出合理範圍：{sf}")
        if tf is not None and sf is not None and float(tf) > float(sf):
            raise ValueError(f"target_fps({tf}) 不能大於 source_fps({sf})")
        cd = v.get("ng_cooldown_s")
        if cd is not None and not (0 <= float(cd) <= 3600):
            raise ValueError(f"ng_cooldown_s 要在 0～3600 秒之間：{cd}（同一台相機兩次 NG 至少間隔幾秒；0 = 不冷卻）")
        return v


class ChangeMeta(BaseModel):
    """誰改的、為什麼改 —— 進 audit_log 與版本快照。"""

    actor: str = Field(min_length=1, description="操作者")
    reason: str = ""
    # 這次修改是基於第幾版（GET 拿到的 config_version）。有帶就檢查：資料庫已經不是這版 → 409，
    # 避免兩個人同時改同一條線、後存的把先存的整份蓋掉。新增線路帶 0。不帶 = 不檢查（舊工具相容）。
    base_version: int | None = Field(default=None, ge=0)


class LineUpsert(BaseModel):
    line: LineIn
    meta: ChangeMeta


# runner 每 15 秒回報一次心跳。超過這個秒數沒回報，就認定它已經不在了 ——
# 不管 line_status 裡最後留下的 apply_state 是什麼。
#
# ⚠️ 為什麼要這條：line_status 是「最後一次回報的快照」，runner 被 docker rm 掉時
#    不會、也沒機會把狀態改成 stale。實際發生過：line_c 的 runner 已經消失 5.7 天，
#    /healthz 與設定頁仍顯示 apply_state=ok、cameras_running=6、in_sync=true。
#    維運看板照著這個數字看，會以為那條線活得好好的。
HEARTBEAT_STALE_S = 60


def _is_stale(heartbeat_age_s: int | None) -> bool:
    return heartbeat_age_s is None or heartbeat_age_s > HEARTBEAT_STALE_S


_LINE_COLS = "id, name, station, config_version, line_params, enabled, model, extra_stations"
_CAM_COLS = ("id, ip, stream_path, width, height, cred_ref, name, enabled, station_params, "
             "infer_roi, extra_params")


def _cam_dict(c) -> dict:
    return {"id": c[0], "ip": c[1], "stream_path": c[2], "width": c[3], "height": c[4],
            "cred_ref": c[5], "name": c[6], "enabled": c[7], "station_params": c[8],
            "infer_roi": c[9], "extra_params": c[10] or {}}


# ── 讀 ──────────────────────────────────────────────────────
@router.get("/stations")
def list_station_specs() -> list[dict]:
    """正在跑的 runner 映像有哪些站別，各自要哪些現場值、門檻欄位、ROI 區域用途（roi_roles）。

    前端畫 ROI 時的「用途」下拉選單從這裡來：roi_roles 每項 name 存進 roi_regions[].role，
    classes 是選了用途後預填的類別，required 的一定要畫。
    """
    specs = load_specs()
    if specs is None:
        raise HTTPException(503, "叢集沒有站別參數規格（ConfigMap edgevision-station-specs），查不到站別清單")
    return [{"name": name, **spec} for name, spec in sorted(specs.items())]


@router.get("/lines", response_model=list[LineSummary])
def list_lines() -> list[dict]:
    """所有線路 + 各自的 desired／reported 版本差異。"""
    db = _require_db()
    with db.cursor() as cur:
        cur.execute(
            "SELECT l.id, l.name, l.station, l.config_version, l.line_params, l.enabled, "
            "       s.reported_version, s.apply_state, s.cameras_running, "
            "       EXTRACT(EPOCH FROM (now() - s.heartbeat_at))::int, "
            "       (SELECT count(*) FROM cameras c WHERE c.line_id = l.id AND c.enabled) "
            "FROM lines l LEFT JOIN line_status s ON s.line_id = l.id ORDER BY l.id"
        )
        out = []
        for r in cur.fetchall():
            stale = _is_stale(r[9])
            out.append({
                "id": r[0], "name": r[1], "station": r[2],
                "desired_version": r[3], "line_params": r[4], "enabled": r[5],
                "reported_version": r[6],
                # 心跳斷了就不能再宣稱同步 —— 那個版本號是舊快照，不代表現在
                "in_sync": (not stale) and r[6] is not None and r[6] >= r[3],
                "apply_state": "stale" if stale else (r[7] or "unknown"),
                # 同理：runner 不在了，「有幾台相機在跑」只能是 0
                "cameras_running": 0 if stale else r[8],
                "heartbeat_age_s": r[9],
                "cameras_configured": r[10],
                "stale": stale,
            })
        return out


class CameraConfigOut(BaseModel):
    """GET /api/config/lines/{id} 的 cameras[]（跟 PUT 的 CameraIn 同欄位，輸出不做輸入驗證）。"""

    id: str
    name: str | None = None
    ip: str
    stream_path: str | None = None
    width: int
    height: int
    cred_ref: str | None = None
    enabled: bool = True
    station_params: dict = Field(default_factory=dict)
    infer_roi: list[float] | None = None
    extra_params: dict = Field(default_factory=dict)


class LineDetailOut(BaseModel):
    """GET /api/config/lines/{id}：整條線（含相機）。PUT 時把它放進 {line: ..., meta: ...} 送回。"""

    id: str
    name: str | None = None
    station: str
    config_version: int
    line_params: dict = Field(default_factory=dict)
    enabled: bool
    model: dict | None = None
    extra_stations: list[dict] = Field(default_factory=list)
    cameras: list[CameraConfigOut] = Field(default_factory=list)


class VersionOut(BaseModel):
    """GET /api/config/lines/{id}/versions 的單筆（新到舊）。"""

    config_version: int
    ts: str
    changed_by: str | None = None
    reason: str | None = None


@router.get("/lines/{line_id}", response_model=LineDetailOut)
def get_line(line_id: str) -> dict:
    db = _require_db()
    with db.cursor() as cur:
        cur.execute(f"SELECT {_LINE_COLS} FROM lines WHERE id = %s", (line_id,))
        row = cur.fetchone()
        if row is None:
            raise HTTPException(404, f"找不到線路 {line_id}")
        cur.execute(f"SELECT {_CAM_COLS} FROM cameras WHERE line_id = %s ORDER BY id", (line_id,))
        cams = [_cam_dict(c) for c in cur.fetchall()]
    line = {
        "id": row[0], "name": row[1], "station": row[2],
        "config_version": row[3], "line_params": row[4], "enabled": row[5],
        "model": row[6], "extra_stations": row[7] or [],
        "cameras": cams,
    }
    # model 附上 Registry 的 version / task / imgsz / classes（runner 與調和器讀的就是這份）
    return enrich_line(line, load_specs(), ModelInfo(db))


@router.get("/lines/{line_id}/versions", response_model=list[VersionOut])
def list_versions(line_id: str, limit: int = 20) -> list[dict]:
    """版本快照清單，供還原。"""
    db = _require_db()
    with db.cursor() as cur:
        cur.execute(
            "SELECT config_version, ts, changed_by, change_reason "
            "FROM config_snapshots WHERE line_id = %s "
            "ORDER BY config_version DESC LIMIT %s", (line_id, limit)
        )
        return [
            {"config_version": r[0], "ts": r[1].isoformat(),
             "changed_by": r[2], "reason": r[3]}
            for r in cur.fetchall()
        ]


# ── 寫 ──────────────────────────────────────────────────────
@router.put("/lines/{line_id}")
def upsert_line(line_id: str, body: LineUpsert) -> dict:
    """建立或更新一條線路（含其相機）。

    整條線一起寫：相機清單是線路設定的一部分，分開寫會出現
    「版本號對得上但相機還是舊的」這種難查的中間狀態。
    """
    if body.line.id != line_id:
        raise HTTPException(400, f"路徑的 {line_id} 與內容的 {body.line.id} 不一致")
    line, meta = body.line, body.meta
    # 存檔前檢查站別參數：存進去 30 秒內 runner 就會套用，打錯的參數會讓 runner 反覆重啟
    specs = load_specs()
    if specs is not None:
        problems = check_line(line, specs)
        if problems:
            raise HTTPException(422, {"message": "站別參數不正確，沒有存檔", "problems": problems})
    # 模型自己的屬性（classes / imgsz / task）不存，以 Registry 為準；站別參數明寫的類別名要是模型真的有的
    problems, model_notes, model_infos = prepare_line_for_save(line, specs, ModelInfo(_db))
    if problems:
        raise HTTPException(422, {"message": "模型或類別名不正確，沒有存檔", "problems": problems})
    db = _require_db()

    with db.cursor() as cur:
        if meta.base_version is not None:
            cur.execute("SELECT config_version FROM lines WHERE id = %s FOR UPDATE", (line.id,))
            row = cur.fetchone()
            current = row[0] if row else 0
            if current != meta.base_version:
                raise HTTPException(409, {
                    "message": f"線路 {line.id} 已經被改成 v{current}（你是基於 v{meta.base_version} 改的），"
                               "請重新載入後再改",
                    "current_version": current})
        cur.execute(
            "INSERT INTO lines (id, name, station, line_params, enabled, model, extra_stations, updated_by) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s) "
            "ON CONFLICT (id) DO UPDATE SET "
            "  name = EXCLUDED.name, station = EXCLUDED.station, "
            "  line_params = EXCLUDED.line_params, enabled = EXCLUDED.enabled, "
            "  model = EXCLUDED.model, extra_stations = EXCLUDED.extra_stations, "
            "  config_version = lines.config_version + 1, "
            "  updated_by = EXCLUDED.updated_by, updated_at = now() "
            "RETURNING config_version",
            (line.id, line.name or line.id, line.station,
             json.dumps(line.line_params), line.enabled,
             json.dumps(line.model) if line.model is not None else None,
             json.dumps(line.extra_stations), meta.actor),
        )
        version = cur.fetchone()[0]

        # 不在清單裡的相機視為移出這條線（停用而非刪除，保留歷史）
        keep = [c.id for c in line.cameras]
        if keep:
            cur.execute(
                "UPDATE cameras SET enabled = FALSE, updated_by = %s, updated_at = now() "
                "WHERE line_id = %s AND id <> ALL(%s)", (meta.actor, line.id, keep)
            )
        for c in line.cameras:
            cur.execute(
                "INSERT INTO cameras (id, line_id, name, ip, stream_path, width, height, "
                "                     cred_ref, station_params, infer_roi, extra_params, enabled, updated_by) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) "
                "ON CONFLICT (id) DO UPDATE SET "
                "  line_id = EXCLUDED.line_id, name = EXCLUDED.name, ip = EXCLUDED.ip, "
                "  stream_path = EXCLUDED.stream_path, width = EXCLUDED.width, "
                "  height = EXCLUDED.height, cred_ref = EXCLUDED.cred_ref, "
                "  station_params = EXCLUDED.station_params, "
                "  infer_roi = EXCLUDED.infer_roi, extra_params = EXCLUDED.extra_params, "
                "  enabled = EXCLUDED.enabled, "
                "  updated_by = EXCLUDED.updated_by, updated_at = now()",
                (c.id, line.id, c.name, c.ip, c.stream_path, c.width, c.height,
                 c.cred_ref, json.dumps(c.station_params),
                 json.dumps(c.infer_roi) if c.infer_roi is not None else None,
                 json.dumps(c.extra_params), c.enabled, meta.actor),
            )

        _snapshot_and_audit(cur, line.id, version, meta.actor, meta.reason or "更新線路設定")

    logger.info("線路 %s 已更新為 v%d（by %s）", line.id, version, meta.actor)
    return {
        "line_id": line.id, "config_version": version,
        # 寫入 ≠ 生效：runner 調和器約 30 秒內重啟這條線（畫面與推論中斷約 30 秒～1 分鐘），
        # runner 載入後 reported_version 才會跟上
        "applied": False,
        "params_checked": specs is not None,
        # 這條線實際會用的模型（Registry production 版）：類別、imgsz、task 都以這裡為準
        "models": model_infos,
        "model_notes": model_notes,
        "note": "設定已寫入。約 30 秒內 runner 會自動重啟並載入這一版（該線畫面與推論中斷約 30 秒～1 分鐘）；"
                "以 GET /api/config/lines 看 reported_version 是否跟上、in_sync 是否為 true。"
                + ("" if specs is not None else "⚠️ 叢集沒有站別參數規格（edgevision-station-specs），這次沒有檢查站別參數。"),
    }


@router.post("/lines/{line_id}/restore/{version}")
def restore_version(line_id: str, version: int, meta: ChangeMeta) -> dict:
    """把線路還原到某個版本快照。

    ⚠️ 還原用的是【完整設定物件】而非欄位差異 —— NetBox 的教訓是
    before/after 字典不足以還原 UPDATE。所以這裡直接套用整份快照。
    """
    db = _require_db()
    with db.cursor() as cur:
        cur.execute(
            "SELECT snapshot FROM config_snapshots "
            "WHERE line_id = %s AND config_version = %s", (line_id, version)
        )
        row = cur.fetchone()
        if row is None:
            raise HTTPException(404, f"找不到 {line_id} 的 v{version} 快照")
        snap = row[0]
        # 快照不含啟用狀態（啟用 / 停用是「開關」，不是設定內容）→ 維持線路現在的狀態。
        # 以前沒帶這個欄位，LineIn 預設 enabled=True：還原停用中的線會偷偷把它打開，runner 跟著跑起來。
        cur.execute("SELECT enabled FROM lines WHERE id = %s", (line_id,))
        now = cur.fetchone()
        enabled = bool(now[0]) if now else True

    payload = LineUpsert(
        line=LineIn(
            id=line_id,
            name=snap["line"].get("name", ""),
            station=snap["line"]["station"],
            line_params=snap["line"].get("line_params", {}),
            enabled=enabled,
            # 舊快照（2026-09-29 以前）沒有這兩個欄位 → 預設值，等同當時的行為
            model=snap["line"].get("model"),
            extra_stations=snap["line"].get("extra_stations", []),
            cameras=[CameraIn(**c) for c in snap["cameras"]],
        ),
        meta=ChangeMeta(actor=meta.actor, base_version=meta.base_version,
                        reason=meta.reason or f"還原至 v{version}"),
    )
    return upsert_line(line_id, payload)


# ── runner 回報（reported 端）────────────────────────────────
class StatusIn(BaseModel):
    """runner 回報「我實際載入的是第幾版」。欄位對應 line_status 表。"""

    reported_version: int = Field(ge=0)
    apply_state: str = Field(pattern="^(ok|applying|failed)$")
    apply_error: str | None = None
    cameras_running: int = Field(default=0, ge=0)
    # 推論統計（ai-infer ds_runner/status.py）：cameras = {相機: fps}、fps、dropped…；
    # /healthz 與健康巡檢用 cameras 做相機層級判斷。舊 runner 不帶 → None。
    stats: dict | None = None


@router.post("/lines/{line_id}/status")
def report_status(line_id: str, body: StatusIn) -> dict:
    """monorepo 的 runner 只走 HTTP、不帶 psycopg，用這個端點取代 edgevision/config/reporter.py 直接寫 DB。

    行為與 reporter._write 相同：upsert line_status，heartbeat_at = now()。
    runner 管線起來後送一次 ok，之後每 15 秒心跳；設定載入失敗送 failed 並帶錯誤訊息。
    """
    db = _require_db()
    with db.cursor() as cur:
        cur.execute("SELECT 1 FROM lines WHERE id = %s", (line_id,))
        if cur.fetchone() is None:
            raise HTTPException(404, f"找不到線路 {line_id}")
        cur.execute(
            "INSERT INTO line_status "
            "  (line_id, reported_version, apply_state, apply_error, "
            "   heartbeat_at, cameras_running, stats, updated_at) "
            "VALUES (%s, %s, %s, %s, now(), %s, %s, now()) "
            "ON CONFLICT (line_id) DO UPDATE SET "
            "  reported_version = EXCLUDED.reported_version, "
            "  apply_state = EXCLUDED.apply_state, "
            "  apply_error = EXCLUDED.apply_error, "
            "  heartbeat_at = now(), "
            "  cameras_running = EXCLUDED.cameras_running, "
            "  stats = EXCLUDED.stats, "
            "  updated_at = now()",
            (line_id, body.reported_version, body.apply_state,
             (body.apply_error or None) and body.apply_error[:500], body.cameras_running,
             json.dumps(body.stats) if body.stats is not None else None),
        )
        cur.execute("SELECT config_version FROM lines WHERE id = %s", (line_id,))
        desired = cur.fetchone()[0]
    return {"line_id": line_id, "reported_version": body.reported_version,
            "desired_version": desired, "in_sync": body.reported_version >= desired}


def _snapshot_and_audit(cur, line_id: str, version: int, actor: str, reason: str) -> None:
    """存完整快照 + 寫 audit_log。兩者都要，理由見模組檔頭。"""
    cur.execute(
        "SELECT id, name, station, line_params, model, extra_stations FROM lines WHERE id = %s", (line_id,)
    )
    l = cur.fetchone()
    cur.execute(f"SELECT {_CAM_COLS} FROM cameras WHERE line_id = %s ORDER BY id", (line_id,))
    snapshot = {
        "line": {"id": l[0], "name": l[1], "station": l[2], "line_params": l[3],
                 "model": l[4], "extra_stations": l[5] or []},
        "cameras": [_cam_dict(c) for c in cur.fetchall()],
    }
    cur.execute(
        "INSERT INTO config_snapshots (line_id, config_version, snapshot, changed_by, change_reason) "
        "VALUES (%s, %s, %s, %s, %s) RETURNING id",
        (line_id, version, json.dumps(snapshot, ensure_ascii=False), actor, reason),
    )
    snap_id = cur.fetchone()[0]
    cur.execute(
        "INSERT INTO audit_log (actor, actor_type, channel, action, target, params, result) "
        "VALUES (%s, 'human', 'api', 'camera_config_update', %s, %s, 'success')",
        (actor, line_id,
         json.dumps({"config_version": version, "snapshot_id": snap_id,
                     "reason": reason}, ensure_ascii=False)),
    )
