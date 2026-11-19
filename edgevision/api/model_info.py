"""模型自己的屬性（類別、imgsz、task）一律以模型為準：線路設定不存、不讓人填，讀設定時從 Registry 附上。

⭐ 為什麼：模型只輸出類別「編號」，runner 用設定裡的 classes 翻成名字 —— 偵測紀錄、WebSocket、框的標籤都吃它。
   以前線路設定自己存一份：靠人手抄，抄錯或被逼著寫假名沒人發現，
   而且模型 promote 換版後，存著的那份就過期了，runner 照舊名字翻新模型的編號。
   模型的屬性只該有一個來源：Registry（訓練 / 補登錄時記下的 ModelMetadata，Triton 引擎也是照它建的）。

規則：
  ・PUT：model / extra_stations[].model 只收 name 與部署參數（conf、iou、hidden_classes）；classes / imgsz / task 帶了也不存（回應註明）
         hidden_classes：照樣推論、照樣判定，只是不畫進直播與事件影片（runner probe 畫框前濾掉）；
                         要是模型真的有的類別名（打錯字就等於沒藏）
         模型不在 Registry → 擋（沒登錄就沒有類別來源，runner 跑不起來）
         站別參數明寫的類別名（*_class、violations / targets 的 key、class_map 的值、roi_regions[].classes）
         不在模型類別裡 → 擋（打錯字；吃預設值的部分由 runner 依站別 classes_used() 檢查）
  ・GET：依 name 查 Registry production 版，附上 version、task、imgsz、classes（runner 與調和器讀這份）
  ・Registry 連不上：用最後一次查到的值（model_info_cache 表），不讓 control-api 掛掉連帶所有線起不來
"""
from __future__ import annotations

import json
import logging
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable

logger = logging.getLogger(__name__)

INTRINSIC = ("classes", "imgsz", "task", "version")   # 模型自己的屬性：線路設定不存
UNREACHABLE = object()                               # Registry 連不上（跟「模型不在 Registry」= None 分開）
TTL_S = 20.0                                         # 調和器每 30 秒讀所有線；同一顆模型 20 秒內不重查

_mem: dict[str, tuple[float, dict | None]] = {}


def fetch_registry(name: str, *, base: str | None = None, timeout: float = 5.0):
    """模型 production 版的 {version, task, imgsz, classes}；不在 Registry / 沒有 production → None；連不上 → UNREACHABLE。"""
    base = (base or os.environ.get("CONTROL_URL", "http://control-api:8090")).rstrip("/")
    q = urllib.parse.quote(name, safe="")
    # 服務金鑰（Secret edgevision-service-tokens 的 output-api）：控制 API 強制驗身分後，沒帶就會 401
    hdr = {"X-Service-Token": os.environ["EDGEVISION_SERVICE_TOKEN"]} if os.environ.get("EDGEVISION_SERVICE_TOKEN") else {}

    def get(url):
        return urllib.request.urlopen(urllib.request.Request(url, headers=hdr), timeout=timeout)

    try:
        with get(f"{base}/api/models/{q}/versions") as r:
            versions = json.load(r)
        prod = [v["version"] for v in versions if "production" in (v.get("tags") or [])]
        if not prod:
            return None
        with get(f"{base}/api/models/{q}/versions/{max(prod)}") as r:
            d = json.load(r)
        if not d.get("class_names") or not d.get("imgsz"):
            logger.warning("Registry 的 %s v%s 缺 class_names / imgsz", name, d.get("version"))
            return UNREACHABLE
        return {"version": d["version"], "task": d.get("task"), "imgsz": d["imgsz"], "classes": list(d["class_names"])}
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        logger.warning("查 Registry 失敗 %s：HTTP %s", name, e.code)
        return UNREACHABLE
    except (OSError, ValueError, KeyError) as e:
        logger.warning("查 Registry 失敗 %s：%s", name, e)
        return UNREACHABLE


class ModelInfo:
    """Registry 查詢 + 記憶體 TTL + 資料庫「最後一次查到的值」。db=None 時只用記憶體（測試）。"""

    def __init__(self, db=None, fetch: Callable[[str], object] = fetch_registry, clock=time.monotonic) -> None:
        self.db, self.fetch, self.clock = db, fetch, clock

    def get(self, name: str):
        """回 info dict、None（不在 Registry）或 UNREACHABLE（連不上且沒有快取）。"""
        hit = _mem.get(name)
        if hit and self.clock() - hit[0] < TTL_S:
            return hit[1]
        info = self.fetch(name)
        if info is UNREACHABLE:
            cached = self._load(name)
            if cached is not None:
                return cached
            return UNREACHABLE
        _mem[name] = (self.clock(), info)
        if info is not None:
            self._save(name, info)
        return info

    def _load(self, name: str) -> dict | None:
        if self.db is None:
            return None
        try:
            with self.db.cursor() as cur:
                cur.execute("SELECT info FROM model_info_cache WHERE name = %s", (name,))
                r = cur.fetchone()
            return r[0] if r else None
        except Exception:
            logger.exception("讀 model_info_cache 失敗")
            return None

    def _save(self, name: str, info: dict) -> None:
        if self.db is None:
            return
        try:
            with self.db.cursor() as cur:
                cur.execute("INSERT INTO model_info_cache (name, info, updated_at) VALUES (%s, %s, now()) "
                            "ON CONFLICT (name) DO UPDATE SET info = EXCLUDED.info, updated_at = now() "
                            "WHERE model_info_cache.info IS DISTINCT FROM EXCLUDED.info", (name, json.dumps(info)))
        except Exception:
            logger.exception("寫 model_info_cache 失敗")


def strip_intrinsic(model: dict | None) -> tuple[dict | None, list[str]]:
    """拿掉模型自己的屬性；回 (乾淨的 model, 被拿掉的欄位)。"""
    if not model:
        return model, []
    dropped = [k for k in INTRINSIC if k in model]
    return {k: v for k, v in model.items() if k not in INTRINSIC}, dropped


def _slots(line: dict | object):
    """(位置說明, 站別, model) —— 主站 + 附加站。line 可以是 dict（GET）或 LineIn（PUT）。"""
    get = (lambda k, d=None: line.get(k, d)) if isinstance(line, dict) else (lambda k, d=None: getattr(line, k, d))
    yield "線路 model", get("station"), get("model"), None
    for i, e in enumerate(get("extra_stations") or []):
        yield f"附加站 {e.get('station')}", e.get("station", ""), e.get("model"), i


def _default_name(station: str, specs: dict | None) -> str | None:
    return ((specs or {}).get(station) or {}).get("model", {}).get("name")


def enrich_line(line: dict, specs: dict | None, models: ModelInfo) -> dict:
    """GET 用：每個 model 附上 Registry 的 version / task / imgsz / classes。查不到就不附（runner 會拒絕啟動並寫原因）。"""
    for _, station, model, idx in list(_slots(line)):
        clean, _ = strip_intrinsic(model)
        name = (clean or {}).get("name") or _default_name(station, specs)
        if not name:
            continue
        info = models.get(name)
        out = {**(clean or {}), "name": name, **(info if isinstance(info, dict) else {})}
        if idx is None:
            line["model"] = out
        else:
            line["extra_stations"][idx] = {**line["extra_stations"][idx], "model": out}
    return line


def explicit_class_refs(params: dict) -> set[str]:
    """站別參數裡明寫的類別名（命名慣例跟 ai-infer 站別一致）。"""
    out = {c for r in params.get("roi_regions") or [] if isinstance(r, dict) for c in (r.get("classes") or [])}
    th = params.get("thresholds") or {}
    if not isinstance(th, dict):
        return out
    for k, v in th.items():
        if k.endswith("_class") and isinstance(v, str):
            out.add(v)
        elif k in ("violations", "targets") and isinstance(v, dict):
            out |= set(v)
        elif k == "class_map" and isinstance(v, dict):
            out |= {x for x in v.values() if isinstance(x, str)}
    return out


def prepare_line_for_save(line, specs: dict | None, models: ModelInfo) -> tuple[list[str], list[str], dict]:
    """PUT 用：就地拿掉 model 的模型屬性；回 (problems, notes, {位置: Registry info})。"""
    problems: list[str] = []
    notes: list[str] = []
    infos: dict[str, dict] = {}
    classes_of: dict[str, set[str]] = {}
    for where, station, model, idx in list(_slots(line)):
        clean, dropped = strip_intrinsic(model)
        if dropped:
            notes.append(f"{where}：{dropped} 不用設定，以模型（Registry）為準，已忽略")
        if idx is None:
            line.model = clean
        else:
            line.extra_stations[idx]["model"] = clean
        name = (clean or {}).get("name") or _default_name(station, specs)
        if not name:
            problems.append(f"{where}：站別 {station} 沒有指定模型（model.name）")
            continue
        hidden = (clean or {}).get("hidden_classes")
        if hidden is not None and (not isinstance(hidden, list) or not all(isinstance(c, str) for c in hidden)):
            problems.append(f"{where}：hidden_classes 要是類別名的清單：{hidden!r}")
            hidden = None
        info = models.get(name)
        if info is None:
            problems.append(f"{where}：模型 {name} 不在 Registry（沒有 production 版），先登錄模型")
        elif info is UNREACHABLE:
            notes.append(f"{where}：Registry 連不上，模型 {name} 沒有比對")
        else:
            infos[station] = {"name": name, **info}
            classes_of[station] = set(info["classes"])
            bad = sorted(set(hidden or []) - classes_of[station])
            if bad:
                problems.append(f"{where}：hidden_classes 有模型沒有的類別 {bad}，模型 {name} 的類別是 {info['classes']}")

    main = line.station
    for c in line.cameras:
        if not c.enabled:
            continue
        pairs = [(main, c.station_params or {})] + [
            (st, (c.extra_params or {}).get(st) or {}) for st in classes_of if st != main]
        for st, p in pairs:
            if st not in classes_of:
                continue
            bad = sorted(explicit_class_refs(p) - classes_of[st])
            if bad:
                problems.append(f"相機 {c.id}（{st}）：參數引用了模型沒有的類別 {bad}，"
                                f"模型 {infos[st]['name']} 的類別是 {infos[st]['classes']}")
    return problems, notes, infos
