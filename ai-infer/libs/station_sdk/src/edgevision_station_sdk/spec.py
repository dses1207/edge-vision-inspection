"""站別參數規格：把每站「接受哪些現場值」匯出成 JSON，給 EdgeVision 的設定 API 在存檔時檢查。

    python -m edgevision_station_sdk.spec > station-specs.json

⭐ 為什麼要有：設定存進資料庫後，runner 調和器 30 秒內就自動套用。門檻欄位打錯一個字
   （persist_frame），站別的 from_params 會在 runner 啟動時拋例外 → runner 反覆重啟。
   這份規格讓設定 API 在【存檔時】就擋下來，而不是等 runner 炸。

規格從站別程式本身讀出來（不另外手寫一份）：
  - station.yaml：預設模型名、task、camera_params_required、roi_roles（模型類別以 Registry 為準，不在這裡）
  - 站別模組裡的門檻 dataclass（有 from_params 的那個）：欄位名、型別、預設值
ai-infer CI 從【runner 映像】跑這支，寫進叢集的 ConfigMap edgevision-station-specs —— 規格永遠跟正在跑的 runner 同一版。
"""
from __future__ import annotations

import dataclasses
import json
import sys
import typing

from edgevision_station_sdk.loader import list_stations

SPEC_VERSION = 1


def _type_name(tp) -> str:
    if isinstance(tp, str):                   # from __future__ import annotations → 字串
        return tp
    return getattr(tp, "__name__", str(tp))


def threshold_fields(station_cls: type) -> dict[str, dict] | None:
    """站別模組裡的門檻 dataclass 欄位 {名: {type, default}}；找不到回 None（= 不檢查門檻欄位）。"""
    mod = sys.modules.get(station_cls.__module__)
    found = [obj for obj in vars(mod).values()
             if isinstance(obj, type) and dataclasses.is_dataclass(obj) and hasattr(obj, "from_params")] if mod else []
    if len(found) != 1:
        return None                            # 0 個或多個：不猜
    th = found[0]
    try:
        hints = typing.get_type_hints(th)
    except Exception:   # noqa: BLE001 —— 型別註記解析失敗就退回欄位上的原始註記，不影響匯出
        hints = {}
    out = {}
    for f in dataclasses.fields(th):
        default = f.default if f.default is not dataclasses.MISSING else None
        out[f.name] = {"type": _type_name(hints.get(f.name, f.type)), "default": default}
    return out


def build_specs() -> dict:
    stations = {}
    for name, entry in sorted(list_stations().items()):
        m = entry.manifest
        stations[name] = {
            "display_name": m.display_name,
            "model": {"name": m.model.name, "task": m.model.task},
            "camera_params_required": list(m.camera_params_required),
            "roi_roles": [r.model_dump() for r in m.roi_roles],
            "thresholds": threshold_fields(entry.cls),
        }
    return {"spec_version": SPEC_VERSION, "stations": stations}


if __name__ == "__main__":
    json.dump(build_specs(), sys.stdout, ensure_ascii=False, indent=1, default=str)
    sys.stdout.write("\n")
