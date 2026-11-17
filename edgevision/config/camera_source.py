"""相機與線路設定的載入。

runner 啟動時需要知道的外部事實：接哪幾台相機、每台的 IP／串流路徑、
影像規格、現場校正值。這些的真相在資料庫，
本模組把「從哪裡拿」抽象掉，讓 runner 只管「拿到什麼」。

⚠️ 兩個來源，必須用 CAMERA_SOURCE 明確指定，【禁止隱式 fallback】：

    CAMERA_SOURCE=file   讀掛載的 JSON（過渡期／無 DB 的開發環境）
    CAMERA_SOURCE=api    讀資料庫（正式路線）

  為什麼禁止「檔案有就用檔案、沒有才打 API」這種 fallback：
  那會讓「到底讀到哪一份設定」變得不可預測 —— 現場改了 DB 卻沒生效，
  查半天才發現它讀的是某個殘留的舊檔案。寧可啟動失敗，不要靜默用錯設定。

⚠️ 帳密不存在設定裡，設定只有「引用名」(cred_ref)，由 resolve_credentials()
   在各環境解析（Compose 讀 env_file、K8s 讀 Secret）—— 不把 Secret 結構
   洩進資料庫。
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class CameraConfig:
    """一台相機跑起來需要的全部資訊。

    ⚠️ 解析度不給預設值：猜錯會讓所有座標算錯，而且不會有任何症狀
    （框畫在錯的位置、letterbox 還原錯、存進 DB 的 bbox 全歪）。
    寧可缺設定時啟動失敗，也不要靜默跑錯。
    """

    id: str
    ip: str
    width: int          # 必填
    height: int         # 必填
    stream_path: str = ""
    cred_ref: str = ""
    name: str = ""
    enabled: bool = True
    # 現場校正值：ref_quad、ROI、門檻覆寫…
    # 對應 station.yaml 的 camera_params_required
    station_params: dict = field(default_factory=dict)

    @property
    def rtsp_url(self) -> str:
        """直連相機的 RTSP URL（帳密由 resolve_credentials 帶入）。"""
        user, pw = resolve_credentials(self.cred_ref)
        auth = f"{user}:{pw}@" if user else ""
        return f"rtsp://{auth}{self.ip}{self.stream_path}"


@dataclass(slots=True)
class LineConfig:
    """一條線路（= 一個 runner 進程）的完整設定。"""

    id: str
    station: str
    cameras: list[CameraConfig]
    # 線路層級參數，例如 target_fps。station.yaml 給預設、這裡覆寫
    line_params: dict = field(default_factory=dict)
    # ⭐ desired 版本號。runner 載入後要回報自己跑的是這一版
    config_version: int = 1
    name: str = ""

    def param(self, key: str, default=None):
        return self.line_params.get(key, default)


# ── 帳密解析：設定只存引用名，值從環境變數來 ────────────────────
# cred_ref 寫這個值 = 相機不用帳密（有些相機的 RTSP 不需要驗證）
NO_AUTH_REF = "none"
_warned_fallback: set[str] = set()


def resolve_credentials(cred_ref: str) -> tuple[str, str]:
    """把 cred_ref 解析成 (帳號, 密碼)。回 ("", "") 表示不帶帳密。

    Compose：env_file 提供 CAM_USER / CAM_PASS（或 CAM_<REF>_USER/_PASS）
    K8s：    Secret 掛成同名環境變數

    兩邊都只是「環境變數」，所以程式碼一份即可 —— 這與 db.py 讀
    DATABASE_URL、objects.py 讀 MINIO_* 是同一套哲學。

    規則：
      · cred_ref = "none"            → 不帶帳密
      · CAM_<REF>_USER 有設          → 用它
      · 沒設（或 cred_ref 空）       → 退回 CAM_USER / CAM_PASS。指名了 ref 卻退回時【印警告】——
        以前是默默換成預設那組，不用帳密的相機就被塞進別人的帳密
      · 連 CAM_USER 都沒有          → 報錯
    """
    ref = (cred_ref or "").strip()
    if ref.lower() == NO_AUTH_REF:
        return "", ""
    if ref:
        prefix = f"CAM_{ref.upper()}"
        user = os.environ.get(f"{prefix}_USER")
        pw = os.environ.get(f"{prefix}_PASS")
        if user:
            return user, pw or ""

    user = os.environ.get("CAM_USER")
    pw = os.environ.get("CAM_PASS")
    if not user:
        # 不回空字串矇混：沒帳密就是連不上，與其讓 RTSP 在深處
        # 報一個看不懂的錯，不如在這裡講清楚缺什麼
        want = f"CAM_{ref.upper()}_USER / _PASS 或 " if ref else ""
        raise SystemExit(
            f"找不到相機帳密（cred_ref={ref or '未指定'}）。"
            f"請設定 {want}CAM_USER / CAM_PASS；相機不用帳密請把 cred_ref 設成 {NO_AUTH_REF!r}。"
        )
    if ref and ref not in _warned_fallback:
        _warned_fallback.add(ref)
        logger.warning("cred_ref=%r 沒有 CAM_%s_USER，改用 CAM_USER；相機不用帳密請設 cred_ref=%r",
                       ref, ref.upper(), NO_AUTH_REF)
    return user, pw or ""


# ── 來源一：檔案（過渡期）────────────────────────────────────
_FILE_CANDIDATES = [
    Path("/data/camera-package"),
    Path(__file__).resolve().parents[2] / "camera-package",
]


def _pkg_dir() -> Path:
    for p in _FILE_CANDIDATES:
        if (p / "config" / "cameras.json").exists():
            return p
    return _FILE_CANDIDATES[-1]


def _load_from_file(line_id: str, camera_ids: list[str] | None) -> LineConfig:
    """從掛載的 JSON 讀。過渡期用，schema 定案後由匯入腳本轉進 DB。"""
    path = _pkg_dir() / "config" / "cameras.json"
    if not path.exists():
        raise SystemExit(f"CAMERA_SOURCE=file 但找不到設定檔：{path}")
    raw = json.loads(path.read_text())

    # file 模式沒有解析度欄位（舊 JSON 格式沒有），要求由環境變數明確給定。
    # 不給預設值 —— 見 CameraConfig 的說明。
    _w, _h = os.environ.get("CAMERA_WIDTH"), os.environ.get("CAMERA_HEIGHT")
    if not _w or not _h:
        raise SystemExit(
            "CAMERA_SOURCE=file 時必須設定 CAMERA_WIDTH / CAMERA_HEIGHT。"
            "\n舊的 JSON 格式沒有這兩個欄位，而猜錯解析度會讓所有座標算錯且無症狀。"
            "\n（子碼流通常是 1280x720，但請確認後明確設定，不要讓程式替你猜。）"
        )
    width, height = int(_w), int(_h)

    stream_path = os.environ.get("CAMERA_STREAM_PATH")
    if not stream_path:
        raise SystemExit(
            "CAMERA_SOURCE=file 時必須設定 CAMERA_STREAM_PATH（如 /stream2）。"
            "\n接錯碼流會讓解析度與推論成本都不對。"
        )

    wanted = camera_ids or []
    cams: list[CameraConfig] = []
    for c in raw:
        cam_id = f"cam{c['id']}"
        if wanted and cam_id not in wanted:
            continue
        # 現場校正值：把 roi_config 整包帶進 station_params，
        # 判定模組要什麼自己挑（ref_quad、其他 ROI…）
        params: dict = {}
        for r in (c.get("roi_config") or {}).get("regions") or []:
            if r.get("type") and r.get("points"):
                params.setdefault(r["type"], r["points"])
        url = c.get("connection_url") or ""
        cams.append(
            CameraConfig(
                id=cam_id,
                ip=c.get("ip") or _ip_from_url(url),
                width=width, height=height,
                stream_path=stream_path,
                name=c.get("name", ""),
                station_params=params,
            )
        )
    if wanted:
        cams.sort(key=lambda c: wanted.index(c.id))
    station = os.environ.get("STATION")
    if not station:
        raise SystemExit(
            "CAMERA_SOURCE=file 時必須設定 STATION（判定模組名，如 zone_intrusion）。"
            "\n站別決定跑哪套判定邏輯，不該有預設值。"
        )
    return LineConfig(id=line_id, station=station, cameras=cams)


def _ip_from_url(url: str) -> str:
    if "@" in url:
        url = url.split("@", 1)[1]
    return url.split("//")[-1].split("/")[0].split(":")[0]


# ── 來源二：資料庫（正式路線）────────────────────────────────
def _load_from_api(line_id: str) -> LineConfig:
    """從資料庫讀 —— 這是正式路線的真相來源。

    runner 本機只知道 LINE_ID 與連線設定，相機資料一律用 ID 拉
    。
    """
    from edgevision.storage.db import Database

    db = Database()
    with db.cursor() as cur:
        cur.execute(
            "SELECT name, station, line_params, config_version "
            "FROM lines WHERE id = %s AND enabled = TRUE",
            (line_id,),
        )
        row = cur.fetchone()
        if row is None:
            raise SystemExit(f"設定中找不到線路 {line_id}（或已停用）")
        name, station, line_params, config_version = row

        cur.execute(
            "SELECT id, ip, stream_path, width, height, cred_ref, name, station_params "
            "FROM cameras WHERE line_id = %s AND enabled = TRUE ORDER BY id",
            (line_id,),
        )
        cams = [
            CameraConfig(
                id=r[0], ip=r[1], stream_path=r[2] or "",
                width=r[3], height=r[4], cred_ref=r[5] or "",
                name=r[6] or "", station_params=r[7] or {},
            )
            for r in cur.fetchall()
        ]
    if not cams:
        raise SystemExit(f"線路 {line_id} 沒有啟用中的相機")
    return LineConfig(
        id=line_id, name=name or "", station=station,
        cameras=cams, line_params=line_params or {},
        config_version=config_version,
    )


# ── 對外入口 ────────────────────────────────────────────────
def load_line_config(line_id: str, camera_ids: list[str] | None = None) -> LineConfig:
    """載入一條線路的設定。來源由 CAMERA_SOURCE 決定，無隱式 fallback。

    camera_ids 僅在 file 模式有意義（過渡期用環境變數指定要哪幾台）；
    api 模式下「這條線有哪些相機」是 DB 說了算。
    """
    source = os.environ.get("CAMERA_SOURCE", "file").strip().lower()
    if source == "file":
        cfg = _load_from_file(line_id, camera_ids)
    elif source == "api":
        cfg = _load_from_api(line_id)
    else:
        raise SystemExit(f"CAMERA_SOURCE 只能是 file 或 api，得到：{source!r}")

    logger.info(
        "設定已載入（來源=%s）：線路 %s／站別 %s／%d 台相機／版本 v%d",
        source, cfg.id, cfg.station, len(cfg.cameras), cfg.config_version,
    )
    return cfg
