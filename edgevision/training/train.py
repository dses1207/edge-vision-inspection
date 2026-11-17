"""訓練 Job：YAML 參數 → Ultralytics 訓練 → best.pt + run_report.json。

ClearML 整合：
    ・先 Task.init，Ultralytics 的 ClearML callback 看到 current_task 就沿用它
    ・task.connect(設定) → 參數可在 UI 覆寫 → clone 功能的基礎
    ・曲線、逐類指標、best.pt 由 callback 自動記錄

前處理一致性：訓練用 Ultralytics 的 LetterBox，推論用 predictor_triton._preprocess，
兩者在 tests/contract/test_preprocess_parity.py 釘死相等。
"""

from __future__ import annotations

import csv
import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from libs.contracts.run_report import (
    ClassMetrics, Curves, DatasetStats, FinalMetrics, RunReport, WorstPrediction,
)

from edgevision.training.config import TrainConfig
from edgevision.training.dataset import PreparedDataset

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class TrainOutput:
    best_pt: Path
    last_pt: Path
    run_dir: Path
    results_csv: Path
    epochs_run: int
    best_epoch: int | None
    clearml_task_id: str
    started_at: str
    finished_at: str
    duration_s: float
    metrics: FinalMetrics


class CancelledError(RuntimeError):
    pass


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def code_commit() -> str:
    return os.environ.get("CODE_COMMIT") or os.environ.get("GIT_COMMIT") or "unknown"


# ── 指標抽取（Ultralytics Metrics → 契約）────────────────────

def _class_metrics(metric, names: dict[int, str]) -> dict[str, ClassMetrics]:
    out: dict[str, ClassMetrics] = {}
    try:
        idx = list(metric.ap_class_index)
        for k, ci in enumerate(idx):
            p, r, ap50, ap = metric.class_result(k)
            out[names[int(ci)]] = ClassMetrics(precision=float(p), recall=float(r),
                                               map50=float(ap50), map50_95=float(ap))
    except Exception as exc:  # 某些版本沒有 class_result
        logger.debug("逐類指標抽取失敗：%s", exc)
    return out


def metrics_from_val(results, names: dict[int, str], *, epochs_run: int | None = None,
                     best_epoch: int | None = None) -> FinalMetrics:
    box = results.box
    seg = getattr(results, "seg", None)
    speed = getattr(results, "speed", {}) or {}
    fm = FinalMetrics(
        map50_box=float(box.map50), map50_95_box=float(box.map),
        precision_box=float(box.mp), recall_box=float(box.mr),
        per_class_box=_class_metrics(box, names),
        fitness=float(getattr(results, "fitness", 0.0) or 0.0) if hasattr(results, "fitness") else None,
        epochs_run=epochs_run, best_epoch=best_epoch,
        inference_ms=float(speed.get("inference")) if speed.get("inference") is not None else None,
    )
    if seg is not None:
        fm.map50_seg = float(seg.map50); fm.map50_95_seg = float(seg.map)
        fm.precision_seg = float(seg.mp); fm.recall_seg = float(seg.mr)
        fm.per_class_seg = _class_metrics(seg, names)
    return fm


def confusion_from_val(results, names: dict[int, str]) -> tuple[list[list[int]], list[str]]:
    cm = getattr(results, "confusion_matrix", None)
    if cm is None or getattr(cm, "matrix", None) is None:
        return [], []
    mat = cm.matrix
    labels = [names[i] for i in sorted(names)] + ["background"]
    return [[int(v) for v in row] for row in mat.tolist()], labels


def curves_from_csv(path: Path) -> Curves:
    c = Curves()
    if not path.exists():
        return c
    with path.open() as fh:
        for row in csv.DictReader(fh):
            row = {k.strip(): v.strip() for k, v in row.items() if k}

            def f(key: str) -> float | None:
                v = row.get(key)
                try:
                    return float(v) if v not in (None, "") else None
                except ValueError:
                    return None

            c.epoch.append(int(float(row.get("epoch", len(c.epoch) + 1))))
            for attr, key in [
                ("train_box_loss", "train/box_loss"), ("train_seg_loss", "train/seg_loss"),
                ("train_cls_loss", "train/cls_loss"), ("val_box_loss", "val/box_loss"),
                ("val_seg_loss", "val/seg_loss"), ("val_cls_loss", "val/cls_loss"),
                ("map50_box", "metrics/mAP50(B)"), ("map50_95_box", "metrics/mAP50-95(B)"),
                ("map50_seg", "metrics/mAP50(M)"), ("map50_95_seg", "metrics/mAP50-95(M)"),
                ("lr", "lr/pg0"),
            ]:
                v = f(key)
                if v is not None:
                    getattr(c, attr).append(v)
    return c


def best_epoch_from_curves(c: Curves) -> int | None:
    series = c.map50_95_seg or c.map50_95_box
    if not series:
        return None
    return c.epoch[max(range(len(series)), key=lambda i: series[i])]


# ── 逐圖指標（自算，版本無關）→ worst_predictions ───────────

def _labels_boxes(lbl: Path, w: int, h: int) -> list[dict]:
    out = []
    if not lbl.exists():
        return out
    for line in lbl.read_text().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        xs = [float(v) * w for v in parts[1::2]]
        ys = [float(v) * h for v in parts[2::2]]
        out.append({"class_id": int(parts[0]), "bbox": [min(xs), min(ys), max(xs), max(ys)]})
    return out


def image_metrics(model, dataset: PreparedDataset, *, split: str = "val", imgsz: int = 640,
                  conf: float = 0.25, iou_thr: float = 0.5, device=None, limit: int | None = None) -> list[dict]:
    """每張圖 tp/fp/fn/precision/recall/f1（bbox IoU≥0.5 配對）。"""
    from edgevision.training.parity import match_detections

    idir = dataset.root / "images" / split
    ldir = dataset.root / "labels" / split
    images = sorted(p for p in idir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"})
    if limit:
        images = images[:limit]
    rows = []
    for img in images:
        res = model.predict(str(img), imgsz=imgsz, conf=conf, verbose=False, device=device)[0]
        h, w = res.orig_shape
        preds = [{"class_id": int(c), "conf": float(s), "bbox": [float(v) for v in b]}
                 for c, s, b in zip(res.boxes.cls.tolist(), res.boxes.conf.tolist(), res.boxes.xyxy.tolist())]
        gts = _labels_boxes(ldir / img.with_suffix(".txt").name, w, h)
        pairs = match_detections(gts, preds, iou_match=iou_thr)
        tp = len(pairs); fp = len(preds) - tp; fn = len(gts) - tp
        p = tp / (tp + fp) if tp + fp else (1.0 if fn == 0 else 0.0)
        r = tp / (tp + fn) if tp + fn else 1.0
        f1 = 2 * p * r / (p + r) if p + r else 0.0
        rows.append({"image": f"images/{split}/{img.name}", "tp": tp, "fp": fp, "fn": fn,
                     "precision": round(p, 4), "recall": round(r, 4), "f1": round(f1, 4)})
    return rows


def worst_predictions(rows: list[dict], n: int = 20) -> list[WorstPrediction]:
    bad = [r for r in rows if r["fp"] or r["fn"]]
    bad.sort(key=lambda r: (r["f1"], -(r["fp"] + r["fn"])))
    out = []
    for r in bad[:n]:
        et = "mixed" if r["fp"] and r["fn"] else ("miss" if r["fn"] else "false_alarm")
        out.append(WorstPrediction(**r, error_type=et))
    return out


# ── 訓練主流程 ───────────────────────────────────────────

def run_training(
    cfg: TrainConfig, dataset: PreparedDataset, *, base_model_path: str, work_dir: Path,
    task_id: str, on_epoch: Callable[[int, int, dict], None] | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> TrainOutput:
    from clearml import Task
    from ultralytics import YOLO

    run_dir = work_dir / "run"
    started = _now(); t0 = time.time()

    # ClearML task：先建，Ultralytics callback 會沿用（官方行為）
    # ⚠️ MinIO 這類非 AWS 端點，ClearML 的 S3 URI 要帶 host:port：s3://minio:9000/experiments/
    output_uri = os.environ.get("CLEARML_OUTPUT_URI") or f"s3://{os.environ.get('MINIO_ENDPOINT', 'minio:9000')}/experiments/"
    clearml_task = Task.init(
        project_name=cfg.clearml_project, task_name=f"{cfg.model_name}-{task_id}",
        task_type=Task.TaskTypes.training, tags=[cfg.model_name, "edgevision"],
        output_uri=output_uri, reuse_last_task_id=False,
        auto_connect_frameworks={"pytorch": False, "matplotlib": False},
    )
    connected = clearml_task.connect(dict(cfg.train), name="edgevision_train")
    clearml_task.connect({
        "model_name": cfg.model_name, "task_id": task_id, "dataset": dataset.version,
        "golden": cfg.golden or "", "base_model": cfg.base_model, "code_commit": code_commit(),
        "parent_task_id": cfg.lineage.parent_task_id or "", "clone_reason": cfg.lineage.clone_reason or "",
    }, name="edgevision_lineage")
    train_args = dict(connected)   # UI 覆寫（遠端重跑）會反映在這裡

    model = YOLO(base_model_path)
    state = {"epochs": int(train_args.get("epochs", 100))}

    def _on_epoch_end(trainer):
        epoch = trainer.epoch + 1
        m = {k: float(v) for k, v in (trainer.metrics or {}).items() if isinstance(v, (int, float))}
        if on_epoch:
            on_epoch(epoch, state["epochs"], m)
        if cancel_check and cancel_check():
            raise CancelledError("收到取消請求")

    model.add_callback("on_train_epoch_end", _on_epoch_end)
    logger.info("開始訓練 %s：base=%s data=%s args=%s", cfg.model_name, base_model_path,
                dataset.data_yaml, train_args)
    results = model.train(data=str(dataset.data_yaml), project=str(run_dir.parent),
                          name=run_dir.name, exist_ok=True, plots=True, **train_args)

    trainer = model.trainer
    best = Path(trainer.best); last = Path(trainer.last)
    if not best.exists():
        raise RuntimeError("訓練結束但找不到 best.pt")
    results_csv = Path(trainer.save_dir) / "results.csv"
    curves = curves_from_csv(results_csv)
    epochs_run = len(curves.epoch) or int(trainer.epoch + 1)
    names = {int(k): v for k, v in model.names.items()}
    fm = metrics_from_val(results, names, epochs_run=epochs_run, best_epoch=best_epoch_from_curves(curves))
    finished = _now()
    return TrainOutput(best_pt=best, last_pt=last, run_dir=Path(trainer.save_dir), results_csv=results_csv,
                       epochs_run=epochs_run, best_epoch=fm.best_epoch, clearml_task_id=clearml_task.id,
                       started_at=started, finished_at=finished, duration_s=time.time() - t0, metrics=fm)


def build_report(
    cfg: TrainConfig, out: TrainOutput, dataset: PreparedDataset, *, task_id: str,
    val_results=None, golden_metrics: FinalMetrics | None = None,
    worst: list[WorstPrediction] | None = None, notes: list[str] | None = None,
) -> RunReport:
    names = {i: n for i, n in enumerate(dataset.class_names)}
    cm, labels = confusion_from_val(val_results, names) if val_results is not None else ([], [])
    m = dataset.manifest
    return RunReport(
        run_id=out.clearml_task_id, task_id=task_id, model_name=cfg.model_name,
        dataset_version=dataset.version, golden_version=cfg.golden, base_model=cfg.base_model,
        code_commit=code_commit(), config=dict(cfg.train),
        curves=curves_from_csv(out.results_csv), final_metrics=out.metrics,
        golden_metrics=golden_metrics, confusion_matrix=cm, confusion_labels=labels,
        worst_predictions=worst or [],
        dataset_stats=DatasetStats(
            dataset_version=dataset.version, n_images=m.n_images, n_train=m.count("train"),
            n_val=m.count("val"), class_distribution=m.class_distribution,
            bbox_size_distribution=m.bbox_size_distribution, images_per_source=m.images_per_source,
        ),
        started_at=out.started_at, finished_at=out.finished_at, duration_s=round(out.duration_s, 1),
        notes=notes or [],
    )


def validate(model_path: str, data_yaml: Path, *, imgsz: int, device=None, batch: int = 8, project: Path | None = None):
    """跑一次 val，回傳 Ultralytics metrics 物件。"""
    from ultralytics import YOLO

    kw = dict(data=str(data_yaml), imgsz=imgsz, batch=batch, plots=False, verbose=False)
    if device is not None:
        kw["device"] = device
    if project is not None:
        kw.update(project=str(project), name="val", exist_ok=True)
    return YOLO(model_path).val(**kw)
