"""模型自己的屬性（classes / imgsz / task）以模型為準：線路設定不存，GET 時從 Registry 附上（edgevision/api/model_info.py）。"""
import pytest
from fastapi import HTTPException

from edgevision.api import config_api, model_info
from edgevision.api.config_api import CameraIn, ChangeMeta, LineIn, LineUpsert, upsert_line
from edgevision.api.model_info import (
    UNREACHABLE,
    ModelInfo,
    enrich_line,
    explicit_class_refs,
    prepare_line_for_save,
    strip_intrinsic,
)

PPE = {"version": 1, "task": "detect", "imgsz": 1280, "classes": ["vest_ng", "helmet_ng", "helmet_ok", "goggles_ng"]}
ALT = {"version": 1, "task": "detect", "imgsz": 1024, "classes": ["load_part", "check_ok"]}
REG = {"ppe": PPE, "siteb_golf": ALT}
SPECS = {"siteb_golf": {"model": {"name": "siteb_golf"}}, "ppe_check": {"model": {"name": "ppe"}}}
TH = {"violations": {"vest_ng": "未穿反光背心", "helmet_ng": "未戴安全帽", "goggles_ng": "未戴護目鏡"}}


@pytest.fixture(autouse=True)
def _fresh_cache():
    model_info._mem.clear()


def models(reg=REG):
    return ModelInfo(fetch=lambda n: reg.get(n))


def siteb_golf(ppe_model=None, ppe_th=None, model=None):
    cam = CameraIn(id="siteb216", ip="192.0.2.185", width=1920, height=1080,
                   extra_params={"ppe_check": {"thresholds": ppe_th or {}}})
    extra = {"station": "ppe_check", "interval": 4}
    if ppe_model is not None:
        extra["model"] = ppe_model
    return LineIn(id="siteb_golf", station="siteb_golf", cameras=[cam], model=model, extra_stations=[extra])


def test_save_drops_model_attributes_and_does_not_block_on_them():
    # 舊寫法帶了 classes（甚至假名）/ imgsz：不擋，直接丟掉，以模型為準
    line = siteb_golf(ppe_model={"name": "ppe", "conf": 0.25, "imgsz": 640, "classes": ["clothes", "bare hand"]},
                      ppe_th=TH)
    problems, notes, infos = prepare_line_for_save(line, SPECS, models())
    assert problems == []
    assert line.extra_stations[0]["model"] == {"name": "ppe", "conf": 0.25}          # 存進去的只剩部署參數
    assert any("已忽略" in n for n in notes)
    assert infos["ppe_check"]["classes"] == PPE["classes"] and infos["siteb_golf"]["name"] == "siteb_golf"


def test_param_class_names_still_checked_against_model():
    th = {"violations": {"clothes": "未穿反光背心", "bare hand": "未戴安全帽"}}      # 舊的假名
    problems, _, _ = prepare_line_for_save(siteb_golf(ppe_model={"name": "ppe"}, ppe_th=th), SPECS, models())
    assert any("bare hand" in p and "vest_ng" in p for p in problems)


def test_hidden_classes_kept_and_checked_against_model():
    """hidden_classes 是部署參數（存）；名字要是模型的類別，格式不對也擋。"""
    line = siteb_golf(ppe_model={"name": "ppe", "hidden_classes": ["helmet_ok"]}, ppe_th=TH,
                      model={"name": "siteb_golf", "hidden_classes": ["load_part"]})
    problems, _, _ = prepare_line_for_save(line, SPECS, models())
    assert problems == [] and line.extra_stations[0]["model"]["hidden_classes"] == ["helmet_ok"]
    problems, _, _ = prepare_line_for_save(siteb_golf(ppe_model={"name": "ppe", "hidden_classes": ["glove"]},
                                                      ppe_th=TH), SPECS, models())
    assert any("hidden_classes" in p and "glove" in p for p in problems)
    problems, _, _ = prepare_line_for_save(siteb_golf(ppe_model={"name": "ppe", "hidden_classes": "helmet_ok"},
                                                      ppe_th=TH), SPECS, models())
    assert any("清單" in p for p in problems)


def test_unregistered_model_blocks_and_unreachable_does_not():
    assert any("不在 Registry" in p for p in prepare_line_for_save(
        siteb_golf(ppe_model={"name": "ppe_x"}, ppe_th=TH), SPECS, models())[0])
    problems, notes, _ = prepare_line_for_save(siteb_golf(ppe_model={"name": "ppe"}, ppe_th=TH), SPECS,
                                               ModelInfo(fetch=lambda n: UNREACHABLE))
    assert problems == [] and any("沒有比對" in n for n in notes)


def test_get_attaches_registry_attributes():
    line = {"id": "siteb_golf", "station": "siteb_golf", "model": None,
            "extra_stations": [{"station": "ppe_check", "interval": 4,
                                "model": {"name": "ppe", "conf": 0.25, "classes": ["stale"]}}]}
    out = enrich_line(line, SPECS, models())
    assert out["model"] == {"name": "siteb_golf", **ALT}                                   # 沒寫 → 站別預設模型
    assert out["extra_stations"][0]["model"] == {"name": "ppe", "conf": 0.25, **PPE}       # 舊的 classes 被模型的蓋掉
    assert out["extra_stations"][0]["interval"] == 4


def test_registry_down_uses_last_known_from_db():
    class _Cur:
        rows: dict = {}
        def execute(self, sql, args):
            self.sql, self.args = sql, args
            if sql.startswith("INSERT"):
                _Cur.rows[args[0]] = __import__("json").loads(args[1])
        def fetchone(self):
            return (_Cur.rows[self.args[0]],) if self.args[0] in _Cur.rows else None

    class _DB:
        def cursor(self):
            import contextlib
            return contextlib.nullcontext(_Cur())

    t = [0.0]
    up = ModelInfo(_DB(), fetch=lambda n: REG.get(n), clock=lambda: t[0])
    assert up.get("ppe") == PPE                                   # 查到 → 寫進快取
    t[0] = 100.0                                                  # 超過記憶體 TTL
    down = ModelInfo(_DB(), fetch=lambda n: UNREACHABLE, clock=lambda: t[0])
    assert down.get("ppe") == PPE                                 # Registry 掛了 → 最後一次查到的值
    assert down.get("never_seen") is UNREACHABLE


def test_strip_and_refs():
    assert strip_intrinsic({"name": "m", "classes": [1], "imgsz": 1, "task": "d", "conf": 0.5}) == (
        {"name": "m", "conf": 0.5}, ["classes", "imgsz", "task"])
    p = {"roi_regions": [{"type": "detect", "classes": ["cls_a"], "points": []}, {"type": "mask"}],
         "thresholds": {"hand_class": "cls_a", "rack1_class": "cls_b", "min_conf": {"x": 0.5},
                        "targets": {"cls_c": {}}, "class_map": {"corner": "cls_d"}, "persist_frames": 3}}
    assert explicit_class_refs(p) == {"cls_a", "cls_b", "cls_c", "cls_d"}


def test_upsert_rejects_before_touching_db(monkeypatch):
    monkeypatch.setattr(config_api, "load_specs", lambda: SPECS)
    monkeypatch.setattr(config_api, "ModelInfo", lambda db: models())
    monkeypatch.setattr(config_api, "_db", None)
    th = {"violations": {"helmet_ngg": "未戴安全帽"}}
    body = LineUpsert(line=siteb_golf(ppe_model={"name": "ppe"}, ppe_th=th), meta=ChangeMeta(actor="t"))
    with pytest.raises(HTTPException) as e:
        upsert_line("siteb_golf", body)
    assert e.value.status_code == 422 and "helmet_ngg" in str(e.value.detail["problems"])
