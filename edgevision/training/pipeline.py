"""訓練 pipeline 的步驟 CLI。Orchestrator 拉起的容器就是跑這個。

    python -m edgevision.training.pipeline trainer  --task-id X   （GPU，trainer 映像）
    python -m edgevision.training.pipeline finalize --task-id X   （CPU，edgevision 映像）

trainer 步驟：拉資料 → 訓練 → run_report → 評估 Gate → 匯出 ONNX + 一致性 → 註冊（tag=candidate）
finalize 步驟：上傳 model.plan → TensorRT 一致性（臨時 Triton）→ 更新 Registry → 完成任務

每步冪等：狀態進 tasks 表，重跑同一 task 不會重複註冊（版本已存在會擋）。
退出碼：0 成功、2 取消、3 Gate / 一致性未過（任務 failed 但不是程式錯）、1 其他錯誤。
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

from libs.contracts import Lineage, ModelMetadata, TrainingJobRequest
from libs.contracts.model_meta import TAG_CANDIDATE, TAG_PRODUCTION, TAG_REJECTED
from libs.contracts.training import TaskStatus

from edgevision.training.tracker import TaskStore

logger = logging.getLogger("pipeline")

EXIT_OK, EXIT_ERROR, EXIT_CANCELLED, EXIT_REJECTED = 0, 1, 2, 3


def _work_dir(task_id: str) -> Path:
    root = Path(os.environ.get("TRAIN_WORK_DIR", "/work"))
    d = root / task_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _images_of(ds, split: str, n: int) -> list[Path]:
    d = ds.root / "images" / split
    return sorted(p for p in d.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"})[:n]


# ═══════════════════════════════════════════════════════════
# trainer 步驟
# ═══════════════════════════════════════════════════════════

def step_trainer(task_id: str) -> int:
    from edgevision.controller.deployer import ModelSpec, write_triton_config
    from edgevision.registry.clearml_registry import ClearMLRegistry
    from edgevision.storage.objects import ObjectStore
    from edgevision.training.config import load_train_config, resolve_base_model
    from edgevision.training.dataset import source_for
    from edgevision.training.evaluate import evaluate_model
    from edgevision.training.export import export_onnx, make_parity_reference, onnx_parity, prepare_parity_images
    from edgevision.training.train import (
        CancelledError, build_report, image_metrics, run_training, worst_predictions,
    )

    store = TaskStore()
    task = store.get(task_id)
    if task is None:
        logger.error("task %s 不存在", task_id); return EXIT_ERROR
    req = TrainingJobRequest.model_validate(task.params["request"])
    work = _work_dir(task_id)
    objects = ObjectStore()
    registry = ClearMLRegistry(objects=objects)

    def hb(msg: str) -> None:
        store.heartbeat(task_id, msg)

    try:
        hb("trainer：載入設定")
        cfg = load_train_config(req.config or None, model_name=req.model_name, overrides=req.overrides,
                                dataset=req.dataset_ref, golden=req.golden_ref,
                                gates=req.gates, config_source=req.config_source)
        cfg.lineage.parent_task_id = req.parent_task_id
        cfg.lineage.clone_reason = req.clone_reason
        version = registry.next_version(cfg.model_name)
        store.set_result(task_id, {"trainer": {"version": version, "config": cfg.model_dump(mode="json")}})

        # ① 資料
        hb("trainer：準備資料集")
        seed = int(cfg.train.get("seed", 0))
        ds = source_for(cfg.dataset, class_names=cfg.class_names, seed=seed).prepare(work / "dataset")
        golden = None
        if cfg.golden:
            golden = source_for(cfg.golden, class_names=ds.class_names, seed=seed).prepare(work / "golden")
            from edgevision.training.dataset import write_data_yaml
            # 黃金測試集全部當 val（train 指向同一目錄只是滿足 Ultralytics 的欄位要求）
            import yaml as _y
            d = {"path": str(golden.root), "train": "images/val", "val": "images/val",
                 "names": {i: n for i, n in enumerate(golden.class_names)}}
            golden.data_yaml.write_text(_y.safe_dump(d, allow_unicode=True, sort_keys=False))
            overlap = ds.manifest.hashes() & golden.manifest.hashes()
            if overlap:
                raise ValueError(f"黃金測試集與訓練資料重疊 {len(overlap)} 張（hash 相同），Gate 會失真，拒絕訓練")
        if cfg.class_names and list(cfg.class_names) != list(ds.class_names):
            raise ValueError(f"設定 class_names={cfg.class_names} 與資料集 {ds.class_names} 不一致")

        # ② 訓練
        base_path = resolve_base_model(cfg.base_model, objects=objects, work_dir=work, registry=registry)
        epochs_total = int(cfg.train.get("epochs", 0))

        def on_epoch(epoch: int, total: int, m: dict) -> None:
            hb(f"trainer：epoch {epoch}/{total} mAP50(B)={m.get('metrics/mAP50(B)', 0):.3f}")
            store.set_result(task_id, {"trainer": {"epoch": epoch, "epochs": total}})

        hb(f"trainer：訓練中（{epochs_total} epochs）")
        out = run_training(cfg, ds, base_model_path=base_path, work_dir=work, task_id=task_id,
                           on_epoch=on_epoch, cancel_check=lambda: store.cancel_requested(task_id))
        device = cfg.train.get("device", None)

        # ③ 評估 Gate（黃金測試集；沒有就用 val）
        hb("trainer：評估 Gate")
        prod = registry.resolve(cfg.model_name, tag=TAG_PRODUCTION)
        prod_metrics = prod.meta.metrics if prod else None
        prod_image_metrics = None
        if prod and prod.meta.artifacts.eval_report:
            try:
                b, k = objects.parse_uri(prod.meta.artifacts.eval_report)
                prod_image_metrics = json.loads(objects.get(bucket=b, key=k)).get("image_metrics")
            except Exception as exc:
                logger.warning("讀不到現役 eval_report：%s", exc)
        basis_ds = golden or ds
        gate, gate_metrics, val_results = evaluate_model(
            str(out.best_pt), data_yaml=basis_ds.data_yaml, rules=cfg.resolve_gate_rules(), imgsz=cfg.imgsz,
            dataset=basis_ds, basis="golden" if golden else "val", production_metrics=prod_metrics,
            production_image_metrics=prod_image_metrics, device=device, work_dir=work,
        )

        # ④ run_report
        from ultralytics import YOLO
        rows = image_metrics(YOLO(str(out.best_pt)), ds, split="val", imgsz=cfg.imgsz, device=device)
        notes = [] if golden else ["未提供黃金測試集，Gate 以 val 為依據（分數偏樂觀）"]
        report = build_report(cfg, out, ds, task_id=task_id, val_results=val_results,
                              golden_metrics=gate_metrics if golden else None,
                              worst=worst_predictions(rows), notes=notes)
        (work / "run_report.json").write_text(report.model_dump_json(indent=2))
        (work / "eval_report.json").write_text(json.dumps(gate.to_dict(), ensure_ascii=False, indent=2))

        if not gate.passed:
            # 擋下：報告留在 experiments/ 供診斷，不進 Registry
            _upload_reports(objects, task_id, work)
            store.finish(task_id, TaskStatus.FAILED, message=f"評估 Gate 未通過：{gate.summary()}",
                         error=gate.summary(), run_report_uri=f"s3://experiments/{task_id}/run_report.json")
            return EXIT_REJECTED

        # ⑤ 匯出 ONNX + 一致性
        hb("trainer：匯出 ONNX")
        from edgevision.training.export import is_end2end
        cfg.export.end2end = is_end2end(out.best_pt)
        onnx_path = export_onnx(out.best_pt, work / "model.onnx", cfg.export)
        n_img = int(cfg.parity.get("images", 8))
        # 兩邊都吃同一批「已 letterbox 成正方形」的圖，差異才只剩轉檔本身
        parity_imgs = prepare_parity_images(_images_of(basis_ds, "val", n_img), work / "parity" / "images", imgsz=cfg.imgsz)
        pres = onnx_parity(out.best_pt, onnx_path, parity_imgs, imgsz=cfg.imgsz, parity_cfg=cfg.parity, device=device, task=cfg.task)
        store.set_result(task_id, {"trainer": {"onnx_parity": pres.to_dict()}})
        if cfg.export.fuse_masks:
            # parity 過了才動 graph：parity 驗的是轉檔，fuse 的 check 驗的是手術
            from edgevision.training.export import fuse_seg_masks
            store.set_result(task_id, {"trainer": {"fuse_masks": fuse_seg_masks(onnx_path, cfg.export)}})
        if not pres.passed:
            _upload_reports(objects, task_id, work)
            store.finish(task_id, TaskStatus.FAILED, message=f"pt→onnx 一致性未過：{pres.reason}", error=pres.reason)
            return EXIT_REJECTED
        make_parity_reference(out.best_pt, parity_imgs, work / "parity", imgsz=cfg.imgsz, device=device)

        # 給 trtexec / 臨時 Triton 用的 repo 結構
        spec = ModelSpec(task=cfg.task, imgsz=cfg.imgsz, class_names=list(ds.class_names),
                         backend="tensorrt", max_batch_size=cfg.export.batch if cfg.export.dynamic else 0,
                         end2end=cfg.export.end2end, max_det=cfg.export.max_det,
                         fuse_masks=cfg.export.fuse_masks)
        write_triton_config(work / "repo", cfg.model_name, spec)
        (work / "repo" / cfg.model_name / "1").mkdir(parents=True, exist_ok=True)

        # ⑥ 註冊（candidate）
        hb("trainer：註冊模型")
        meta = ModelMetadata(
            model_name=cfg.model_name, version=version, task=cfg.task, class_names=list(ds.class_names),
            imgsz=cfg.imgsz, created_at=_now(), created_by=task.created_by,
            lineage=Lineage(dataset_version=ds.version, golden_version=golden.version if golden else None,
                            code_commit=report.code_commit, base_model=cfg.base_model,
                            training_run_id=out.clearml_task_id, task_id=task_id,
                            parent_task_id=req.parent_task_id, clone_reason=req.clone_reason),
            hyperparameters=dict(cfg.train), metrics=_metrics_dict(gate_metrics, out.metrics, gate.basis),
            notes=(req.notes + (f"｜設定來源：{cfg.config_source}" if cfg.config_source else "")).strip("｜ ") or req.notes,
            export=cfg.export,
        )
        resolved = registry.register(meta, {
            "pytorch": out.best_pt, "onnx": onnx_path,
            "run_report": work / "run_report.json", "eval_report": work / "eval_report.json",
        }, tags=[TAG_CANDIDATE])
        store.set_result(task_id, {"trainer": {
            "registry_id": resolved.registry_id, "clearml_task_id": out.clearml_task_id,
            "gate": gate.to_dict() | {"image_metrics": None}, "metrics": meta.metrics,
            "epochs_run": out.epochs_run, "duration_s": round(out.duration_s, 1),
        }})
        with_uri = resolved.meta.artifacts.run_report
        store.heartbeat(task_id, f"trainer：完成，v{version} 已註冊（candidate）")
        # model_version / run_report_uri 先寫上，finalize 再改狀態
        with store.db.cursor() as cur:
            cur.execute("UPDATE tasks SET model_version = %s, run_report_uri = %s WHERE task_id = %s",
                        (str(version), with_uri, task_id))
        return EXIT_OK

    except CancelledError:
        store.finish(task_id, TaskStatus.CANCELLED, message="已取消")
        return EXIT_CANCELLED
    except Exception as exc:
        logger.error("trainer 失敗：%s\n%s", exc, traceback.format_exc())
        store.finish(task_id, TaskStatus.FAILED, message=f"trainer 錯誤：{exc}", error=str(exc)[:2000])
        return EXIT_ERROR


def _metrics_dict(gate_metrics, val_metrics, basis: str) -> dict:
    g = gate_metrics
    return {
        "basis": basis, "map50_box": g.map50_box, "map50_95_box": g.map50_95_box,
        "precision_box": g.precision_box, "recall_box": g.recall_box,
        "map50_seg": g.map50_seg, "map50_95_seg": g.map50_95_seg,
        "inference_ms": g.inference_ms,
        "per_class_recall_box": {k: v.recall for k, v in g.per_class_box.items()},
        "val_map50_box": val_metrics.map50_box, "val_map50_seg": val_metrics.map50_seg,
    }


def _upload_reports(objects, task_id: str, work: Path) -> None:
    import io
    for name in ("run_report.json", "eval_report.json"):
        p = work / name
        if p.exists():
            data = p.read_bytes()
            objects.put(bucket="experiments", key=f"{task_id}/{name}", data=io.BytesIO(data),
                        length=len(data), content_type="application/json")


# ═══════════════════════════════════════════════════════════
# finalize 步驟
# ═══════════════════════════════════════════════════════════

def step_finalize(task_id: str) -> int:
    """model.plan 上傳 + TensorRT 一致性（臨時 Triton）+ 收尾。"""
    from edgevision.registry.clearml_registry import ClearMLRegistry
    from edgevision.storage.objects import ObjectStore
    from edgevision.training.parity import compare

    store = TaskStore()
    task = store.get(task_id)
    if task is None:
        logger.error("task %s 不存在", task_id); return EXIT_ERROR
    tr = (task.result or {}).get("trainer") or {}
    version = int(tr.get("version") or task.model_version or 0)
    cfg = tr.get("config") or {}
    model_name = cfg.get("model_name") or task.params["request"]["model_name"]
    work = _work_dir(task_id)
    objects = ObjectStore()
    registry = ClearMLRegistry(objects=objects)

    try:
        plan = work / "repo" / model_name / "1" / "model.plan"
        if not plan.exists():
            raise FileNotFoundError(f"trtexec 沒有產出 {plan}")

        # TensorRT 一致性：臨時 Triton 的 URL 由 Orchestrator 以環境變數帶入
        parity_url = os.environ.get("PARITY_TRITON_URL", "")
        ref_path = work / "parity" / "reference.json"
        if parity_url and ref_path.exists():
            store.heartbeat(task_id, "finalize：TensorRT 一致性驗證")
            pres = _trt_parity(parity_url, model_name, work, ref_path, cfg)
            store.set_result(task_id, {"finalize": {"trt_parity": pres.to_dict()}})
            if not pres.passed:
                registry.set_tag(model_name, version, TAG_REJECTED, exclusive=False)
                registry.remove_tag(model_name, version, TAG_CANDIDATE)
                store.finish(task_id, TaskStatus.FAILED, message=f"TensorRT 一致性未過：{pres.reason}", error=pres.reason)
                return EXIT_REJECTED
        else:
            store.set_result(task_id, {"finalize": {"trt_parity": {"skipped": True}}})
            logger.warning("略過 TensorRT 一致性（PARITY_TRITON_URL 未設或無參考）")

        store.heartbeat(task_id, "finalize：上傳 model.plan")
        from edgevision.registry.base import MODELS_BUCKET, artifact_key, artifact_uri
        with plan.open("rb") as fh:
            objects.put(bucket=MODELS_BUCKET, key=artifact_key(model_name, version, "tensorrt"),
                        data=fh, length=plan.stat().st_size)
        resolved = registry.update_artifacts(model_name, version, tensorrt=artifact_uri(model_name, version, "tensorrt"))
        store.set_result(task_id, {"finalize": {"tensorrt": resolved.meta.artifacts.tensorrt}})
        store.finish(task_id, TaskStatus.COMPLETED,
                     message=f"完成：{model_name} v{version} 已註冊（candidate），等待放行",
                     model_version=str(version))
        return EXIT_OK
    except Exception as exc:
        logger.error("finalize 失敗：%s\n%s", exc, traceback.format_exc())
        store.finish(task_id, TaskStatus.FAILED, message=f"finalize 錯誤：{exc}", error=str(exc)[:2000])
        return EXIT_ERROR


def _trt_parity(url: str, model_name: str, work: Path, ref_path: Path, cfg: dict):
    import time

    import cv2
    import httpx

    from edgevision.inference.predictor_triton import TritonPredictor
    from edgevision.training.parity import compare

    ref = json.loads(ref_path.read_text())
    # 等臨時 Triton 就緒
    deadline = time.time() + 120
    while time.time() < deadline:
        try:
            if httpx.get(f"{url}/v2/models/{model_name}/ready", timeout=3).status_code == 200:
                break
        except Exception:
            pass
        time.sleep(2)
    else:
        raise RuntimeError(f"臨時 Triton {url} 未就緒（model {model_name}）")

    pred = TritonPredictor(url=url.replace("http://", ""), model_name=model_name, imgsz=int(ref["imgsz"]),
                           conf_threshold=float(ref["conf"]), iou_threshold=float(ref["iou"]),
                           class_names=cfg.get("class_names") or None)
    cand: dict[str, list[dict]] = {}
    for name in ref["detections"]:
        img = cv2.imread(str(work / "parity" / "images" / name))
        r = pred.predict(img)
        cand[name] = [{"class_id": d.class_id, "conf": d.confidence,
                       "bbox": [d.bbox.x1, d.bbox.y1, d.bbox.x2, d.bbox.y2]} for d in r.detections]
    pc = cfg.get("parity") or {}
    return compare(ref["detections"], cand, iou_match=pc.get("iou_match", 0.8),
                   min_match_ratio=pc.get("min_match_ratio", 0.9), conf_tolerance=pc.get("conf_tolerance", 0.05))


# ═══════════════════════════════════════════════════════════

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="訓練 pipeline 步驟")
    ap.add_argument("step", choices=["trainer", "finalize"])
    ap.add_argument("--task-id", required=True)
    args = ap.parse_args(argv)
    logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO").upper(),
                        format="%(asctime)s %(levelname)-7s %(name)-20s %(message)s", datefmt="%H:%M:%S")
    return {"trainer": step_trainer, "finalize": step_finalize}[args.step](args.task_id)


if __name__ == "__main__":
    sys.exit(main())
