"""線路設定：runner 啟動時唯一需要從外面拿的東西。

runner 本機只知道「身分」（LINE_ID）與「去哪拿設定」（CAMERA_SOURCE + 位址）。
相機 IP / 帳密只有 go2rtc 需要；runner 隔著 go2rtc 拉串流，只認相機編號。

兩種來源，用 CAMERA_SOURCE 明確切換，**禁止隱式 fallback**：
  api   正式：GET {CONFIG_API}/api/config/lines/{LINE_ID}（output-api 的設定端點）
  file  過渡期 / 無 DB 開發：讀一個 JSON 檔（CAMERA_FILE），**內容就是上面 GET 的回應**

schema（api 回應 = file 內容，同一份）：
{
  "id": "alpha_a", "name": "ALPHA A 線", "station": "zone_intrusion",
  "config_version": 3, "enabled": true,
  "line_params": {"source_fps": 30, "target_fps": 5},        # 選填；決定推論間隔
  "cameras": [
    {"id": "cam156", "ip": "…", "stream_path": "/stream2", "width": 1280, "height": 720,
     "cred_ref": "default", "name": "ALPHA2", "enabled": true,
     "station_params": {"ref_quad": [[…]], "roi_regions": [...]}},
    …
  ]
}
runner 只用：id / station / config_version / line_params / cameras[].{id,width,height,enabled,station_params}。
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

import yaml
from edgevision_station_sdk import MissingCameraParams, get_station_entry, load_station
from pydantic import BaseModel, Field, model_validator

logger = logging.getLogger(__name__)


class CameraConfig(BaseModel):
    id: str
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    enabled: bool = True
    station_params: dict[str, Any] = Field(default_factory=dict)
    # 一線多站時，附加站別各自的現場值：{"ppe_check": {...}}。主站仍用 station_params。
    extra_params: dict[str, dict[str, Any]] = Field(default_factory=dict)
    # 主站模型只看這個 ROI（正規化 [x1, y1, x2, y2]）：主站 nvinferserver 以物件模式裁出來再餵模型（見 roi.py）。
    # 原程式「裁切後再推論」的線要設；None = 整幅。
    infer_roi: list[float] | None = None

    @model_validator(mode="after")
    def _check_roi(self) -> CameraConfig:
        r = self.infer_roi
        if r is not None and (len(r) != 4 or not all(0.0 <= v <= 1.0 for v in r) or r[0] >= r[2] or r[1] >= r[3]):
            raise ValueError(f"相機 {self.id!r} 的 infer_roi 必須是正規化 [x1, y1, x2, y2] 且 x1<x2、y1<y2：{r}")
        return self

    @model_validator(mode="before")
    @classmethod
    def _accept_params_alias(cls, data: Any) -> Any:
        if isinstance(data, dict) and "params" in data and "station_params" not in data:
            data = {**data, "station_params": data["params"]}
        return data

    @property
    def params(self) -> dict[str, Any]:
        return self.station_params


class ModelConfig(BaseModel):
    """classes / imgsz / task / version 是模型自己的屬性：設定 API 從 Registry 附上（線路設定不存）。
    conf / iou 是部署參數，留在線路設定。"""
    name: str
    classes: list[str] = Field(default_factory=list)
    imgsz: int = 640
    conf: float = 0.5
    iou: float = 0.45
    task: str | None = None          # 沒附（舊設定 API）→ 用站別 station.yaml 的 task
    version: int | None = None       # Registry production 版號（啟動時比對 Triton 載入的版本）
    # 不畫進影像的類別：照樣推論、照樣送去判定（NG 不受影響），只是直播 _marked 與事件影片上看不到框
    hidden_classes: list[str] = Field(default_factory=list)


class ExtraStation(BaseModel):
    """一條線的附加站別：自己的模型、自己的推論間隔。例：ALPHA 主站 + PPE（YOLO-World 每 5 幀）。

    每個附加站對應 runner 裡多一個 nvinferserver（unique_id 依序 2, 3, …）；
    現場值放 cameras[].extra_params[station]。
    """
    station: str
    model: ModelConfig | None = None
    interval: int | None = None      # 每 N+1 幀推論一次；None = 跟主站相同


@dataclass(frozen=True)
class Binding:
    """runner 內部用：一個模型 + 一個站別 = 一個 nvinferserver。主站 gie_id=1，附加站依序往後。"""
    key: str                          # = 站別名，也是 TensorJob / FrameResult.station 的值
    station: str
    model: ModelConfig
    task: str
    gie_id: int
    primary: bool
    interval: int | None = None       # 附加站的覆寫；主站由 pipeline.resolve_interval 決定


class LineConfig(BaseModel):
    id: str
    name: str = ""
    station: str
    config_version: int = 0
    enabled: bool = True
    line_params: dict[str, Any] = Field(default_factory=dict)
    model: ModelConfig | None = None
    cameras: list[CameraConfig] = Field(min_length=1)
    extra_stations: list[ExtraStation] = Field(default_factory=list)
    # runner 自己填：設定有問題、被跳過的相機 → 原因（isolate_bad_cameras）。設定 API 不會送這個欄位
    skipped_cameras: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check(self) -> LineConfig:
        active = self.active_cameras
        if not active:
            raise ValueError(f"線路 {self.id!r} 沒有啟用中的相機")
        ids = [c.id for c in active]
        if len(ids) != len(set(ids)):
            raise ValueError(f"相機編號重複：{ids}")
        names = [self.station, *(e.station for e in self.extra_stations)]
        if len(names) != len(set(names)):
            raise ValueError(f"線路 {self.id!r} 的站別重複：{names}")
        sizes = {(c.width, c.height) for c in active}
        if len(sizes) > 1:
            # nvstreammux 把所有路縮到同一尺寸；不同解析度的相機混在一條線會讓座標還原錯，且沒有症狀
            raise ValueError(f"線路 {self.id!r} 的相機解析度不一致：{sorted(sizes)}，請分線")
        return self

    # ── 給 runner 用的投影 ───────────────────────────────
    @property
    def line_id(self) -> str:
        return self.id

    @property
    def active_cameras(self) -> list[CameraConfig]:
        return [c for c in self.cameras if c.enabled]

    @property
    def camera_ids(self) -> list[str]:
        return [c.id for c in self.active_cameras]

    @property
    def frame_size(self) -> tuple[int, int]:
        c = self.active_cameras[0]
        return c.width, c.height

    @property
    def infer_interval(self) -> int | None:
        """nvinferserver 的 interval（每 N+1 幀推論一次）。由 line_params 的 source_fps / target_fps 算；沒給回 None。"""
        sf, tf = self.line_params.get("source_fps"), self.line_params.get("target_fps")
        if not sf or not tf:
            return None
        return max(0, round(float(sf) / float(tf)) - 1)

    @staticmethod
    def resolve_model_for(station: str, override: ModelConfig | None) -> ModelConfig:
        """模型由線路設定指定名字；類別、imgsz、task 是設定 API 從 Registry 附上的（線路設定不存）。

        站別不宣告「必要類別」—— 它會用到哪些類別名，由 camera_problems() 依現場值逐台比對。
        """
        if override is None:
            raise ValueError(f"站別 {station!r} 沒有模型：設定 API 沒附上 model（站別也沒有預設模型？）")
        if not override.classes:
            raise ValueError(f"模型 {override.name!r} 沒有類別：設定 API 查不到它在 Registry 的現役版本"
                             "（模型沒登錄 / 沒有 production，或 control-api 連不上且沒有快取）")
        return override

    def resolve_model(self) -> ModelConfig:
        return self.resolve_model_for(self.station, self.model)

    @staticmethod
    def task_for(station: str, model: ModelConfig) -> str:
        """後處理（detect / segment / pose）跟模型走：模型自己記的 task 優先，沒有才用站別宣告的。"""
        return model.task or get_station_entry(station).manifest.model.task

    @property
    def model_task(self) -> str:
        return self.task_for(self.station, self.resolve_model())

    def bindings(self) -> list[Binding]:
        """主站 + 附加站，依 gie_id 排序。主站永遠是第一個、gie_id=1。"""
        out = [Binding(key=self.station, station=self.station, model=self.resolve_model(),
                       task=self.model_task, gie_id=1, primary=True)]
        for i, e in enumerate(self.extra_stations, start=2):
            m = self.resolve_model_for(e.station, e.model)
            out.append(Binding(key=e.station, station=e.station, model=m, task=self.task_for(e.station, m),
                               gie_id=i, primary=False, interval=e.interval))
        return out

    def params_for(self, camera: CameraConfig, station: str) -> dict[str, Any]:
        """某台相機給某個站別的現場值：主站用 station_params，附加站用 extra_params[站別]。"""
        return camera.station_params if station == self.station else camera.extra_params.get(station, {})

    def camera_problems(self) -> dict[str, Exception]:
        """逐台、逐站別檢查現場值：缺必填、站別參數建不起來、引用模型沒有的類別。回 {相機: 第一個錯誤}。

        給 isolate_bad_cameras() 決定哪幾台要跳過。
        """
        out: dict[str, Exception] = {}
        bindings = self.bindings()
        for c in self.active_cameras:
            for b in bindings:
                p = self.params_for(c, b.station)
                missing = get_station_entry(b.station).manifest.missing_params(p)
                if missing:
                    out[c.id] = MissingCameraParams(f"站別 {b.station!r} 缺現場值 {missing}")
                    break
                try:
                    used = set(load_station(b.station, p).classes_used()) | roi_classes(p)
                except Exception as exc:   # noqa: BLE001 —— 站別自己的參數檢查（例：ROI 少了類別）
                    out[c.id] = exc
                    break
                bad = sorted(used - set(b.model.classes))
                if bad:
                    out[c.id] = ClassNameMismatch(
                        f"站別 {b.station!r} 引用了模型 {b.model.name!r} 沒有的類別 {bad}；"
                        f"模型的類別是 {list(b.model.classes)}")
                    break
        return out

    def isolate_bad_cameras(self) -> LineConfig:
        """設定有問題的相機【只跳過那一台】，其他台照常啟動；回傳拿掉壞相機的設定（skipped_cameras 記原因）。

        ⚠️ 為什麼（2026-11-03）：以前一台相機的現場值有錯（缺 ROI、ROI 少類別、打錯類別名）
           整條線就拒絕啟動，同線其他正常的相機跟著停判。一條線的相機共用一條管線只是為了批次推論，
           判定本來就一台一個站別實例，啟動檢查也應該一台一台算。
        全部相機都有問題 → 這條線沒東西可跑，照舊拋錯（例外型別沿用第一台的）。
        """
        bad = self.camera_problems()
        if not bad:
            return self
        reasons = {cid: f"{type(e).__name__}: {e}" for cid, e in bad.items()}
        if len(bad) == len(self.active_cameras):
            first = next(iter(bad.values()))
            msg = f"線路 {self.id!r} 每台啟用中的相機設定都有問題：{reasons}"
            try:
                err = type(first)(msg)
            except Exception:   # noqa: BLE001 —— 例外型別的建構子不吃單一字串
                err = ValueError(msg)
            raise err from first
        for cid, why in reasons.items():
            logger.error("[config] 相機 %s 設定有問題，這台不啟動（同線其他相機照常）：%s", cid, why)
        cams = [c.model_copy(update={"enabled": False}) if c.id in bad else c for c in self.cameras]
        return self.model_copy(update={"cameras": cams, "skipped_cameras": {**self.skipped_cameras, **reasons}})


def roi_classes(params: dict[str, Any]) -> set[str]:
    """roi_regions 標了只對哪些類別生效（filter_by_roi 用）—— 這些也是類別名引用。"""
    return {c for r in params.get("roi_regions") or [] for c in (r.get("classes") or [])}


class ClassNameMismatch(ValueError):
    pass


def line_url(base_url: str, line_id: str) -> str:
    return f"{base_url.rstrip('/')}/api/config/lines/{line_id}"


def _from_file(path: str | Path) -> dict:
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    return yaml.safe_load(text) if p.suffix in (".yaml", ".yml") else json.loads(text)


def _from_api(base_url: str, line_id: str, timeout: float = 10.0) -> dict:
    req = Request(line_url(base_url, line_id), headers={"Accept": "application/json"})
    with urlopen(req, timeout=timeout) as r:   # 內網設定服務，URL 由環境變數組成
        return json.loads(r.read().decode("utf-8"))


def load_line_config(env: dict[str, str] | None = None) -> LineConfig:
    """從環境變數決定來源並載入。回傳前已驗證站別存在；現場值不齊 / 類別名對不上的相機被跳過（skipped_cameras）。"""
    e = os.environ if env is None else env
    source = e.get("CAMERA_SOURCE")
    if source == "file":
        raw = _from_file(e["CAMERA_FILE"])
    elif source == "api":
        raw = _from_api(e["CONFIG_API"], e["LINE_ID"])
    else:
        raise ValueError("CAMERA_SOURCE 必須是 file 或 api（不做隱式 fallback）")
    cfg = LineConfig.model_validate(raw)
    if "LINE_ID" in e and cfg.id != e["LINE_ID"]:
        raise ValueError(f"設定的 id={cfg.id!r} 與環境 LINE_ID={e['LINE_ID']!r} 不符")
    if not cfg.enabled:
        raise ValueError(f"線路 {cfg.id!r} 已停用")
    return cfg.isolate_bad_cameras()   # 有問題的相機只跳過那一台；全壞才整條不啟動
