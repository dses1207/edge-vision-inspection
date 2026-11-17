"""評估 Gate：擋下不合格模型，不過就 fail。

規則來自 configs/gates/<model>.yaml，擋的有四條：
    absolute            絕對門檻（在黃金測試集上量）
    critical_recall     關鍵類別 recall 下限
    latency_ms_max      推論延遲
    golden_no_new_miss  黃金測試集不得出現「現役抓到、新模型漏掉」

和現役比分數（舊的 no_regression）不擋（2026-10-14 決定：分數不代表一切；
而且現役舊模型的指標是舊驗證集量的，跟新資料集不能比）。有現役指標時照樣算出差多少，
寫進報告給人看；規則裡還留著 no_regression 也只當參考。

Gate 的評估對象一律是【黃金測試集】—— 沒給黃金測試集就退回用 val，並在報告註明
（val 參與過早停與模型挑選，分數會偏樂觀）。

純邏輯部分（規則判斷）與 GPU 部分（跑 val）分開，規則判斷可在開發機單元測試。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from libs.contracts.run_report import FinalMetrics

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class GateCheck:
    name: str
    passed: bool
    detail: str
    skipped: bool = False


@dataclass(slots=True)
class GateResult:
    passed: bool
    checks: list[GateCheck] = field(default_factory=list)
    basis: str = "golden"       # golden | val
    metrics: dict[str, Any] = field(default_factory=dict)
    image_metrics: list[dict] = field(default_factory=list)

    def summary(self) -> str:
        failed = [c for c in self.checks if not c.passed and not c.skipped]
        return "PASS" if self.passed else "FAIL：" + "; ".join(f"{c.name}（{c.detail}）" for c in failed)

    def to_dict(self) -> dict:
        return {
            "passed": self.passed, "basis": self.basis, "summary": self.summary(),
            "checks": [{"name": c.name, "passed": c.passed, "skipped": c.skipped, "detail": c.detail}
                       for c in self.checks],
            "metrics": self.metrics, "image_metrics": self.image_metrics,
        }


def load_rules(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"找不到 Gate 規則：{path}")
    return yaml.safe_load(path.read_text()) or {}


def _flat(m: FinalMetrics) -> dict[str, float | None]:
    return {
        "map50_box": m.map50_box, "map50_95_box": m.map50_95_box,
        "precision_box": m.precision_box, "recall_box": m.recall_box,
        "map50_seg": m.map50_seg, "map50_95_seg": m.map50_95_seg,
        "precision_seg": m.precision_seg, "recall_seg": m.recall_seg,
        "inference_ms": m.inference_ms,
    }


def apply_rules(
    rules: dict, metrics: FinalMetrics, *, basis: str = "golden",
    production_metrics: dict | None = None,
    image_metrics: list[dict] | None = None,
    production_image_metrics: list[dict] | None = None,
) -> GateResult:
    checks: list[GateCheck] = []
    flat = _flat(metrics)

    # ① 絕對門檻
    for key, thr in (rules.get("absolute") or {}).items():
        v = flat.get(key)
        if v is None:
            checks.append(GateCheck(f"absolute.{key}", True, "指標不存在，略過", skipped=True)); continue
        checks.append(GateCheck(f"absolute.{key}", v >= thr, f"{v:.4f} >= {thr}"))

    # ② 和現役比：只參考、不擋
    vs_production = {}
    for metric in ("map50_box", "map50_95_box", "map50_seg", "map50_95_seg"):
        v, cur = flat.get(metric), (production_metrics or {}).get(metric)
        if v is not None and cur is not None:
            vs_production[metric] = {"new": v, "production": cur, "delta": round(v - cur, 6)}
    if vs_production:
        basis_note = "（現役為舊驗證集指標）" if (production_metrics or {}).get("basis") == "legacy-val" else ""
        checks.append(GateCheck("vs_production", True, "僅參考、不擋" + basis_note + "：" + "，".join(
            f"{m} {d['new']:.4f}（現役 {d['production']:.4f}，{d['delta']:+.4f}）" for m, d in vs_production.items()),
            skipped=True))

    # ③ 關鍵類別 recall（seg 模型看 box 的逐類 recall，偵測有無不需 mask）
    for cls, thr in (rules.get("critical_recall") or {}).items():
        cm = metrics.per_class_box.get(cls)
        if cm is None:
            checks.append(GateCheck(f"critical_recall.{cls}", False, "類別不在評估結果中（資料集缺該類？）")); continue
        checks.append(GateCheck(f"critical_recall.{cls}", cm.recall >= thr, f"recall {cm.recall:.4f} >= {thr}"))

    # ④ 延遲
    lat = rules.get("latency_ms_max")
    if lat is not None:
        v = flat.get("inference_ms")
        if v is None:
            checks.append(GateCheck("latency_ms_max", True, "無延遲數據，略過", skipped=True))
        else:
            checks.append(GateCheck("latency_ms_max", v <= lat, f"{v:.1f}ms <= {lat}ms"))

    # ⑤ 黃金測試集不得出現新漏檢
    if rules.get("golden_no_new_miss"):
        if not image_metrics or not production_image_metrics or basis != "golden":
            checks.append(GateCheck("golden_no_new_miss", True, "無現役逐圖結果或非黃金測試集，略過", skipped=True))
        else:
            prod = {r["image"]: r for r in production_image_metrics}
            new_miss = [r["image"] for r in image_metrics
                        if r["fn"] > 0 and r["image"] in prod and prod[r["image"]]["fn"] == 0]
            checks.append(GateCheck("golden_no_new_miss", not new_miss,
                                    f"新漏檢 {len(new_miss)} 張" + (f"：{new_miss[:3]}" if new_miss else "")))

    passed = all(c.passed for c in checks)
    out_metrics = {k: v for k, v in flat.items() if v is not None}
    if vs_production:
        out_metrics["vs_production"] = vs_production
    return GateResult(passed=passed, checks=checks, basis=basis,
                      metrics=out_metrics,
                      image_metrics=image_metrics or [])


def evaluate_model(
    model_path: str, *, data_yaml: Path, imgsz: int, dataset,
    rules: dict | None = None, rules_path: Path | None = None,
    basis: str, production_metrics: dict | None, production_image_metrics: list[dict] | None,
    device=None, work_dir: Path | None = None,
) -> tuple[GateResult, FinalMetrics, Any]:
    """GPU 部分：跑 val + 逐圖指標，再套規則。"""
    from ultralytics import YOLO

    from edgevision.training.train import image_metrics as _image_metrics, metrics_from_val, validate

    if rules is None:
        if rules_path is None:
            raise ValueError("需要 rules（內嵌規則）或 rules_path")
        rules = load_rules(rules_path)
    results = validate(model_path, data_yaml, imgsz=imgsz, device=device, project=work_dir)
    names = {int(k): v for k, v in results.names.items()}
    fm = metrics_from_val(results, names)
    rows = _image_metrics(YOLO(model_path), dataset, split="val", imgsz=imgsz, device=device)
    gate = apply_rules(rules, fm, basis=basis, production_metrics=production_metrics,
                       image_metrics=rows, production_image_metrics=production_image_metrics)
    logger.info("評估 Gate（%s）：%s", basis, gate.summary())
    return gate, fm, results
