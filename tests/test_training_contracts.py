"""訓練側契約：模型身分、run_report、任務請求（純 schema，不需服務）。"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from libs.contracts import Lineage, ModelMetadata, RunReport, TrainingJobRequest  # noqa: E402
from libs.contracts.run_report import DatasetStats, FinalMetrics  # noqa: E402


def _meta(**kw):
    base = dict(model_name="seg_alpha", version=2, class_names=["board", "hand"], created_at="2026-09-02T00:00:00Z",
                lineage=Lineage(dataset_version="lakefs://alpha/abc", code_commit="deadbeef", base_model="yolo11m-seg.pt"))
    base.update(kw)
    return ModelMetadata(**base)


def test_metadata_requires_lineage():
    with pytest.raises(Exception):
        ModelMetadata(model_name="x", version=1, class_names=["a"], created_at="t")  # 缺 lineage


def test_metadata_roundtrip_and_batch():
    m = _meta()
    m2 = ModelMetadata.model_validate_json(m.model_dump_json())
    assert m2 == m
    assert m.triton_max_batch == 32
    m.export.dynamic = False
    assert m.triton_max_batch == 0


def test_legacy_lineage():
    l = Lineage.legacy()
    assert l.dataset_version == "legacy-unknown" and l.code_commit == "legacy-unknown"


def test_run_report_minimal():
    fm = FinalMetrics(map50_box=0.9, map50_95_box=0.7, precision_box=0.9, recall_box=0.9)
    r = RunReport(run_id="c1", model_name="seg_alpha", dataset_version="lakefs://alpha/a", base_model="b",
                  code_commit="c", config={"epochs": 1}, final_metrics=fm,
                  dataset_stats=DatasetStats(dataset_version="lakefs://alpha/a", n_images=2, n_train=1, n_val=1),
                  started_at="s", finished_at="f", duration_s=1.0)
    assert r.schema_version == "1.0"
    assert RunReport.model_validate_json(r.model_dump_json()).final_metrics.map50_box == 0.9


def test_job_request_validation():
    ok = TrainingJobRequest(model_name="seg_alpha", dataset_ref="lakefs://alpha/abc")
    assert ok.auto_promote is False and ok.overrides == {}
    with pytest.raises(Exception):
        TrainingJobRequest(model_name="Seg ALPHA", dataset_ref="x")  # 名稱不合法（Triton 模型名）
