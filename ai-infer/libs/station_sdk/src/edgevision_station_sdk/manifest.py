"""station.yaml：手冊封面。宣告這站預設吃哪個模型、每台相機要提供哪些現場值。

⭐ 不宣告模型類別：模型有哪些類別是模型自己的事（Registry 的 class_names，線路設定 API 帶入）。
   站別判定用到哪些類別名，由站別實例的 classes_used() 依現場值回報，runner 啟動時拿去比對模型。
   （2026-10-14 前這裡寫死 model.classes 兼當「必要類別」，逼得換了類別名的新模型要在設定裡謊報類別名。）
設定線：runner 啟動時比對 camera_params_required 與 cameras.station_params 是否齊全。

roi_roles（選填）：這站的 ROI 有哪幾種用途（例：工作區 / 清潔區 / 人員區）。
  每塊 roi_regions 用 "role": "<name>" 標明用途，站別照 role 取用，不再靠類別或面積猜。
  規格匯出給設定 API（存檔檢查）與前端（畫 ROI 時的用途下拉選單，classes 是選了之後預填的類別）。
  向下相容：一塊都沒標 role 的舊設定照舊（站別自己的舊規則）；只要有一塊標了，就整台都要標（見 roi_problems）。
"""

from __future__ import annotations

from importlib import resources
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field

MANIFEST_FILENAME = "station.yaml"


class ModelSpec(BaseModel):
    name: str
    task: Literal["detect", "segment", "pose"]


class RoiRole(BaseModel):
    name: str                                      # 存進 roi_regions[].role 的值（英文、固定）
    display_name: str = ""                         # 前端顯示
    classes: list[str] = Field(default_factory=list)   # 預填的類別（模型類別名）
    required: bool = False                         # 標了 role 的設定裡一定要有這種區域
    description: str = ""


def roi_role_problems(regions: list, roles: list[RoiRole]) -> list[str]:
    """roi_regions 的 role 標得對不對。沒有任何一塊標 role → []（舊設定，站別自己處理）。

    規則：有標 role 時，每塊 detect 區域都要標（mask 是排除區，可不標）；role 要是這站宣告過的；required 的都要有。
    """
    if not roles:
        return []
    regions = [r for r in regions or [] if isinstance(r, dict)]
    if not any(r.get("role") for r in regions):
        return []
    names = [r.name for r in roles]
    out = []
    unknown = sorted({str(r["role"]) for r in regions if r.get("role") and r["role"] not in names})
    if unknown:
        out.append(f"roi_regions 的 role {unknown} 這個站別沒有，可用：{names}")
    untagged = [i for i, r in enumerate(regions) if r.get("type") == "detect" and not r.get("role")]
    if untagged:
        out.append(f"roi_regions 第 {[i + 1 for i in untagged]} 塊 detect 區沒標 role（有標 role 時每塊 detect 都要標）")
    present = {r.get("role") for r in regions if r.get("points")}
    missing = [f"{r.name}（{r.display_name or r.name}）" for r in roles if r.required and r.name not in present]
    if missing:
        out.append(f"roi_regions 少了必要的區域 {missing}")
    return out


class StationManifest(BaseModel):
    name: str
    display_name: str = ""
    model: ModelSpec
    handler: str                                   # "package.module:ClassName"，與 entry point 一致
    camera_params_required: list[str] = Field(default_factory=list)
    roi_roles: list[RoiRole] = Field(default_factory=list)
    owner: str = ""

    def roi_problems(self, camera_params: dict) -> list[str]:
        return roi_role_problems(camera_params.get("roi_regions") or [], self.roi_roles)

    def missing_params(self, camera_params: dict) -> list[str]:
        return [k for k in self.camera_params_required if k not in camera_params]


def load_manifest_file(path: str | Path) -> StationManifest:
    with open(path, encoding="utf-8") as f:
        return StationManifest.model_validate(yaml.safe_load(f))


def load_manifest_from_package(module_name: str) -> StationManifest:
    """從已安裝的站別套件裡讀 station.yaml（檔案放在套件目錄內，跟 wheel 一起出貨）。"""
    ref = resources.files(module_name).joinpath(MANIFEST_FILENAME)
    with resources.as_file(ref) as p:
        return load_manifest_file(p)
