"""推論站點（B 推論搬到 gpu-node-2）：站點設定、runner 排程、promote 同步到站點的 Triton。"""
import json

import pytest

from edgevision.controller.deployer import DeployResult
from edgevision.orchestrator.runner import StepSpec, build_job_manifest
from edgevision.orchestrator.runner_reconciler import desired_runners, runner_deployment
from edgevision.orchestrator.site_deploy import deploy_to_site, plan_key, site_export_spec
from edgevision.orchestrator.sites import load_sites, parse_sites, site_for_line
from libs.contracts.model_meta import ExportSpec
from tests.test_orchestrator import FakeRunner, MemStore, _orch

B = {"siteb": {"lines": ["siteb_*"], "triton": "triton-siteb", "gpu_claim": "inference-gpu-siteb",
             "node_selector": {"camera-net": "siteb"}, "triton_repo": "s3://minio:9000/models/_triton/sm89",
             "max_batch": 8, "go2rtc_source": "rtsp://192.0.2.75:8554/"}}


def test_parse_and_match(tmp_path):
    sites = parse_sites(B)
    s = site_for_line(sites, "siteb_alpha")
    assert s.name == "siteb" and s.arch == "sm89" and s.triton_http == "triton-siteb:8000"
    assert site_for_line(sites, "ipcam_alpha") is None
    assert load_sites(str(tmp_path / "none.json")) == {}            # 沒檔案 = 只有預設站點
    empty = parse_sites({"siteb": {**B["siteb"], "lines": []}})              # 清空 lines = 回退，要合法
    assert site_for_line(empty, "siteb_alpha") is None and empty["siteb"].arch == "sm89"
    with pytest.raises(ValueError, match="lines"):
        parse_sites({"siteb": {k: v for k, v in B["siteb"].items() if k != "lines"}})
    with pytest.raises(ValueError, match="triton_repo"):
        parse_sites({"x": {**B["siteb"], "triton_repo": "/models"}})
    with pytest.raises(ValueError, match="go2rtc_source"):
        parse_sites({"x": {**B["siteb"], "go2rtc_source": "http://192.0.2.75:1984"}})
    assert s.go2rtc_source == "rtsp://192.0.2.75:8554"                 # 尾巴的 / 去掉（runner 接 /<cam>）
    with pytest.raises(ValueError, match="多個站點"):
        site_for_line(parse_sites({"a": B["siteb"], "b": {**B["siteb"], "lines": ["siteb_alpha"]}}), "siteb_alpha")


def _line(lid):
    return {"id": lid, "model": {"name": "m", "version": 1}, "cameras": [{"enabled": True}], "config_version": 3}


def _env(dep):
    return {e["name"]: e["value"] for e in dep["spec"]["template"]["spec"]["containers"][0]["env"]}


def test_reconciler_places_site_lines():
    lines = [{"id": "siteb_alpha"}, {"id": "ipcam_alpha"}]
    details = {d["id"]: _line(d["id"]) for d in lines}
    out = desired_runners(lines, details, image="img", gpu_claim="inference-gpu", namespace="edgevision",
                          sites=parse_sites(B))
    ct, ipcam = out["runner-siteb-alpha"], out["runner-ipcam-alpha"]
    pod = ct["spec"]["template"]["spec"]
    assert pod["nodeSelector"] == {"camera-net": "siteb"}
    assert pod["resourceClaims"][0]["resourceClaimName"] == "inference-gpu-siteb"
    assert _env(ct)["TRITON_GRPC"] == "triton-siteb:8001" and _env(ct)["TRITON_HTTP"] == "triton-siteb:8000"
    assert _env(ipcam)["TRITON_GRPC"] == "triton:8001" and "nodeSelector" not in ipcam["spec"]["template"]["spec"]
    # 原始畫面從站點本機的 go2rtc 拉；打框畫面照樣推回主 go2rtc（前端只連那台）
    assert _env(ct)["GO2RTC_SOURCE_RTSP"] == "rtsp://192.0.2.75:8554"
    assert _env(ct)["GO2RTC_RTSP"] == _env(ipcam)["GO2RTC_RTSP"] == "rtsp://go2rtc:8554"
    assert "GO2RTC_SOURCE_RTSP" not in _env(ipcam)


def test_default_site_runner_unchanged():
    # 預設站點的 runner 跟加站點功能之前一模一樣（spec-hash 不變 → 上線時不會重啟非 B 的線）
    a = runner_deployment("ipcam_alpha", "m", image="i", gpu_claim="g", namespace="n", config_version=1)
    b = runner_deployment("ipcam_alpha", "m", image="i", gpu_claim="g", namespace="n", config_version=1,
                          triton="triton", node_selector=None)
    assert a == b and "nodeSelector" not in a["spec"]["template"]["spec"]


def test_job_manifest_site_gpu_and_node():
    spec = StepSpec(name="engine", image="t", command=["true"], gpu=True, gpu_claim="inference-gpu-siteb",
                    node_selector={"camera-net": "siteb"})
    pod = build_job_manifest(spec, namespace="edgevision", gpu_claim="training-gpu", gpu_claim_template=None,
                             runtime_class="nvidia", priority_class=None, service_account=None)["spec"]["template"]["spec"]
    assert pod["resourceClaims"][0]["resourceClaimName"] == "inference-gpu-siteb"
    assert pod["nodeSelector"] == {"camera-net": "siteb"}


def test_site_batch_cap():
    site = parse_sites(B)["siteb"]
    assert site_export_spec(ExportSpec(imgsz=1280, batch=32), site).batch == 8
    assert site_export_spec(ExportSpec(batch=4), site).batch == 4
    assert plan_key(site, "ppe", 1) == "_engines/sm89/ppe/v1/model.plan"


# ── 站點部署（建 engine → 上架）──

class Objects:
    def presigned_url(self, **kw): return "GET-URL"
    def presigned_put_url(self, **kw): return "PUT-URL"


class Repo:
    def __init__(self, versions): self.v = dict(versions)
    def versions(self, m): return self.v.get(m, [])
    def has_version(self, m, ver): return ver in self.v.get(m, [])


class Ctrl:
    def __init__(self, versions=None):
        self.repo, self.objects, self.deployed, self.rolled = Repo(versions or {}), Objects(), [], []
    def deploy(self, *, triton_model, source_uri, version, spec):
        self.deployed.append((triton_model, source_uri, version, spec.max_batch_size))
        return DeployResult(True, triton_model, version, "ok", 1.0)
    def rollback(self, *, triton_model, to_version):
        self.rolled.append((triton_model, to_version))
        return DeployResult(True, triton_model, to_version, "ok", 1.0)
    def close(self): pass


def _resolved(name="ppe", version=2, batch=32):
    from libs.contracts.model_meta import Lineage, ModelArtifacts, ModelMetadata
    meta = ModelMetadata(model_name=name, version=version, task="detect", class_names=["a", "b"], imgsz=1280,
                         created_at="x", lineage=Lineage(dataset_version="d", code_commit="c", base_model="b"),
                         artifacts=ModelArtifacts(onnx=f"s3://models/{name}/v{version}/model.onnx"),
                         export=ExportSpec(imgsz=1280, batch=batch, fp16=True))
    class R: pass
    r = R(); r.meta, r.version, r.source_uri = meta, version, f"s3://models/{name}/v{version}/model.plan"
    return r


def test_deploy_to_site_builds_on_site_gpu(tmp_path):
    store = MemStore(); runner = FakeRunner(store)
    o = _orch(store, runner, tmp_path)
    tid = store.create(type=__import__("libs.contracts.training", fromlist=["TaskType"]).TaskType.DEPLOYMENT,
                       created_by="u", channel="api", params={})
    site, ctrl = parse_sites(B)["siteb"], Ctrl({"ppe": [1]})
    assert deploy_to_site(o, tid, site, ctrl, _resolved())
    spec = runner.specs["engine-siteb-ppe-v2"]
    assert spec.gpu_claim == "inference-gpu-siteb" and spec.node_selector == {"camera-net": "siteb"}
    assert "maxShapes=images:8x3x1280x1280" in spec.command[-1] and "--fp16" in spec.command[-1]
    assert "GET-URL" not in spec.command[-1] and spec.env["PLAN_URL"] == "PUT-URL"   # 簽名不進指令列
    # 上架用站點自己的 plan（不是 Registry 記的主站點 plan），Triton config 的 batch 跟 engine 一致
    assert ctrl.deployed == [("ppe", "s3://models/_engines/sm89/ppe/v2/model.plan", 2, 8)]


def test_deploy_to_site_skips_unused_and_reuses_existing(tmp_path):
    store = MemStore(); runner = FakeRunner(store)
    o = _orch(store, runner, tmp_path)
    site = parse_sites(B)["siteb"]
    assert not deploy_to_site(o, "t", site, Ctrl(), _resolved())                 # 站點沒在用 → 不動
    ctrl = Ctrl({"ppe": [1, 2]})
    assert deploy_to_site(o, "t", site, ctrl, _resolved()) and ctrl.rolled == [("ppe", 2)]
    assert runner.started == []                                                   # 已有這版 → 不用重建


def test_promote_failure_on_site_rolls_everything_back(tmp_path):
    class Prev: version = 1
    class Reg:
        def __init__(self): self.promoted = []
        def resolve(self, name, tag="production"): return Prev()
        def promote(self, name, v, tag="production"): self.promoted.append(v)
        def get(self, name, v): return _resolved(version=v)
    class Dep:
        def __init__(self): self.calls = []
        def deploy_from_registry(self, registry, name, version=None, tag="production"):
            self.calls.append(version); return DeployResult(True, name, version, "ok", 1.0)
        def close(self): pass
    store = MemStore(); runner = FakeRunner(store, {"engine-siteb-ppe-v2": 1})
    reg, dep, ctrl = Reg(), Dep(), Ctrl({"ppe": [1]})
    o = _orch(store, runner, tmp_path, reg, dep)
    o._site_deployer_factory = lambda site: ctrl
    o._sites_loader = lambda: parse_sites(B)
    tid = store.create(type=__import__("libs.contracts.training", fromlist=["TaskType"]).TaskType.DEPLOYMENT,
                       created_by="u", channel="api", params={"model_name": "ppe", "version": 2})
    o._run_deployment(tid)
    assert store.tasks[tid]["status"] == "failed" and "已退回 v1" in store.tasks[tid]["status_message"]
    assert reg.promoted == [2, 1] and dep.calls == [2, 1] and ctrl.rolled == [("ppe", 1)]


