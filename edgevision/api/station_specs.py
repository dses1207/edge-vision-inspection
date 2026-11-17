"""存檔前檢查線路設定的站別參數 —— 規格來自正在跑的 runner 映像（ai-infer edgevision_station_sdk.spec）。

⭐ 為什麼要擋在存檔時：設定一存進資料庫，runner 調和器 30 秒內就自動套用。
   門檻欄位打錯一個字、站別名稱寫錯、少了必要的現場值，站別程式在 runner 啟動時才拋例外 →
   runner 反覆重啟、那條線整條停擺。在設定 API 擋下來，錯誤訊息直接回給改設定的人。

規格檔：ConfigMap edgevision-station-specs 掛進來的 JSON（ai-infer CI 每次換 runner 映像時一起更新），
路徑 STATION_SPECS_FILE。檔案不存在 → 不檢查（回 None，讓呼叫端在回應裡註明「未檢查」），不擋存檔。
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_PATH = "/etc/edgevision/station-specs/station-specs.json"
_cache: dict = {"mtime": None, "specs": None}


def load_specs(path: str | None = None) -> dict | None:
    """讀規格（檔案改了才重讀：ConfigMap 更新後 kubelet 會換掉掛載的檔案）。沒有檔案回 None。"""
    p = Path(path or os.environ.get("STATION_SPECS_FILE", DEFAULT_PATH))
    try:
        mtime = p.stat().st_mtime
    except OSError:
        return None
    if _cache["mtime"] != mtime:
        try:
            _cache["specs"] = json.loads(p.read_text(encoding="utf-8")).get("stations", {})
            _cache["mtime"] = mtime
        except Exception:
            logger.exception("站別參數規格讀不了：%s", p)
            return None
    return _cache["specs"]


def _type_ok(value, tname: str) -> bool:
    if tname == "bool":
        return isinstance(value, bool)
    if tname == "int":
        return isinstance(value, int) and not isinstance(value, bool)
    if tname == "float":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if tname == "str":
        return isinstance(value, str)
    return True                                  # 複合型別（list / dict / 自訂）不細查


def roi_role_problems(regions, roles: list[dict]) -> list[str]:
    """roi_regions[].role 標得對不對 —— 規則與 ai-infer edgevision_station_sdk.roi_role_problems 相同（runner 載入時會再檢查一次）。

    沒有任何一塊標 role → 不管（舊設定）。有標：每塊 detect 都要標（mask 不用）、role 要是站別宣告過的、required 的都要有。
    """
    if not roles or not isinstance(regions, list):
        return []
    regions = [r for r in regions if isinstance(r, dict)]
    if not any(r.get("role") for r in regions):
        return []
    names = [r["name"] for r in roles]
    out = []
    unknown = sorted({str(r["role"]) for r in regions if r.get("role") and r["role"] not in names})
    if unknown:
        out.append(f"roi_regions 的 role {unknown} 這個站別沒有，可用：{names}")
    untagged = [i + 1 for i, r in enumerate(regions) if r.get("type") == "detect" and not r.get("role")]
    if untagged:
        out.append(f"roi_regions 第 {untagged} 塊 detect 區沒標 role（有標 role 時每塊 detect 都要標）")
    present = {r.get("role") for r in regions if r.get("points")}
    missing = [f"{r['name']}（{r.get('display_name') or r['name']}）" for r in roles
               if r.get("required") and r["name"] not in present]
    if missing:
        out.append(f"roi_regions 少了必要的區域 {missing}")
    return out


def _check_params(where: str, station: str, params: dict, spec: dict, *, required: bool) -> list[str]:
    out = [f"{where}：站別 {station} 的 {p}" for p in roi_role_problems(params.get("roi_regions"), spec.get("roi_roles") or [])]
    if required:
        # 欄位在、但內容是空的（roi_regions: []）也算沒給：runner 會跳過那台，存檔當下就講清楚
        missing = [k for k in spec.get("camera_params_required", []) if params.get(k) in (None, [], {}, "")]
        if missing:
            out.append(f"{where}：站別 {station} 必須提供 {missing}")
    fields = spec.get("thresholds")
    th = params.get("thresholds") or {}
    if fields is None or not isinstance(th, dict):
        return out                               # 這站沒有可檢查的門檻定義
    unknown = sorted(set(th) - set(fields))
    if unknown:
        out.append(f"{where}：站別 {station} 的 thresholds 不認識 {unknown}，可用：{sorted(fields)}")
    for k, v in th.items():
        if k in fields and not _type_ok(v, fields[k]["type"]):
            out.append(f"{where}：thresholds.{k} 要是 {fields[k]['type']}，收到 {v!r}")
    return out


def check_line(line, specs: dict) -> list[str]:
    """LineIn → 問題清單（空 = 沒問題）。純函式，可測。"""
    problems: list[str] = []
    known = sorted(specs)

    def spec_of(name: str, where: str) -> dict | None:
        if name not in specs:
            problems.append(f"{where}：站別 {name!r} 不存在（目前 runner 映像有：{known}）")
            return None
        return specs[name]

    main = spec_of(line.station, "線路 station")
    extras = {}
    for e in line.extra_stations:
        s = spec_of(e.get("station", ""), "extra_stations")
        if s is not None:
            extras[e["station"]] = s
    for c in line.cameras:
        if not c.enabled:
            continue
        if main is not None:
            problems += _check_params(f"相機 {c.id}", line.station, c.station_params or {}, main, required=True)
        for st, s in extras.items():
            p = (c.extra_params or {}).get(st) or {}
            problems += _check_params(f"相機 {c.id}（附加站 {st}）", st, p, s, required=False)
    return problems
