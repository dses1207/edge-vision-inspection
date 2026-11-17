"""借閱櫃台：按名字找站別。

站別在自己的 pyproject.toml 登記：
    [project.entry-points."edgevision.stations"]
    zone_intrusion = "edgevision_station_zone_intrusion:ZoneIntrusionStation"

pip / uv 安裝時把這行寫進 site-packages 的登記簿；這裡用 importlib.metadata 查。
判讀室（apps）從頭到尾不 import 任何一站，只認名字。找不到就明確報錯，不猜。
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import metadata
from typing import Any

from edgevision_station_sdk.manifest import StationManifest, load_manifest_from_package
from edgevision_station_sdk.protocol import Station

ENTRY_POINT_GROUP = "edgevision.stations"


class StationNotFound(LookupError):
    pass


class MissingCameraParams(ValueError):
    pass


class InvalidRoiRegions(ValueError):
    """roi_regions 的 role 標錯（不認識的 role、有的標有的沒標、少了必要的區域）。"""


@dataclass(frozen=True)
class StationEntry:
    name: str
    module: str
    attr: str
    cls: type

    @property
    def manifest(self) -> StationManifest:
        return load_manifest_from_package(self.module)


def _entry_points():
    eps = metadata.entry_points()
    if hasattr(eps, "select"):                    # Python ≥ 3.10
        return list(eps.select(group=ENTRY_POINT_GROUP))
    return list(eps.get(ENTRY_POINT_GROUP, []))   # pragma: no cover


def list_stations() -> dict[str, StationEntry]:
    out: dict[str, StationEntry] = {}
    for ep in _entry_points():
        out[ep.name] = StationEntry(name=ep.name, module=ep.value.split(":")[0], attr=ep.attr, cls=ep.load())
    return out


def get_station_entry(name: str) -> StationEntry:
    for ep in _entry_points():
        if ep.name == name:
            return StationEntry(name=ep.name, module=ep.value.split(":")[0], attr=ep.attr, cls=ep.load())
    known = sorted(ep.name for ep in _entry_points())
    raise StationNotFound(f"找不到站別 {name!r}；映像裡登記的有：{known}")


def load_station(name: str, camera_params: dict[str, Any] | None = None) -> Station:
    """按名字建一個站別實例（每台相機呼叫一次）。缺必要現場值 → 啟動失敗，不用預設值硬跑。"""
    params = dict(camera_params or {})
    entry = get_station_entry(name)
    manifest = entry.manifest
    missing = manifest.missing_params(params)
    if missing:
        raise MissingCameraParams(f"站別 {name!r} 需要現場值 {missing}，但相機參數只有 {sorted(params)}")
    bad_roi = manifest.roi_problems(params)
    if bad_roi:
        raise InvalidRoiRegions(f"站別 {name!r}：{'；'.join(bad_roi)}")
    station = entry.cls(params)
    if not isinstance(station, Station):
        raise TypeError(f"{entry.module}:{entry.attr} 沒有實作 Station 介面（evaluate / reset / update_params）")
    return station
