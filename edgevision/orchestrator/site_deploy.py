"""把模型部署到其他推論站點的 Triton（站點見 sites.py）。

promote / rollback 在主 Triton 換好之後，對【已經在用這顆模型】的每個站點：
    ① 該站點的 repo 已有這一版 → 直接釘版本、載入（回退的秒級路徑）
    ② 沒有 → 在【那個站點的 GPU】上開一個 Job 建 engine（engine 綁 GPU 架構，不能拿主站點的 plan）
             Registry 的 ONNX ─(預簽下載)→ trtexec ─(預簽上傳)→ models/_engines/<架構>/<模型>/v<版>/model.plan
       → 那個站點的 Triton 照一般流程上架（寫 config、explicit load、等就緒）

Job 只用 Triton 映像（有 trtexec 與 python3），檔案進出靠 control-api 簽的預簽網址 —— 不掛任何卷，
所以排到哪個節點都行（訓練工作區是 gpu-node-1 上的 RWO 卷，別的節點掛不到）。

⚠️ 不碰 models/<模型>/v<版>/model.plan：那是 Registry 記的主站點 engine。
"""
from __future__ import annotations

import dataclasses
import logging

from edgevision.orchestrator.runner import StepSpec
from edgevision.orchestrator.sites import Site

logger = logging.getLogger(__name__)

ENGINE_PREFIX = "_engines"

# 在 Triton 映像裡跑：下載 ONNX → trtexec → 上傳 plan。網址走環境變數，不進指令列（log 裡看不到簽名）
_FETCH = ("python3 -c \"import os,urllib.request as u;"
          "u.urlretrieve(os.environ['ONNX_URL'],'/tmp/model.onnx')\"")
_PUSH = ("python3 -c \"import os,urllib.request as u;"
         "u.urlopen(u.Request(os.environ['PLAN_URL'],data=open('/tmp/model.plan','rb').read(),method='PUT'))\"")


def plan_key(site: Site, model: str, version: int) -> str:
    return f"{ENGINE_PREFIX}/{site.arch}/{model}/v{version}/model.plan"


def site_export_spec(export, site: Site):
    """模型的匯出規格，batch 不超過站點上限（小卡：宣告 32 也只建到 site.max_batch）。"""
    if site.max_batch and export.batch > site.max_batch:
        return export.model_copy(update={"batch": site.max_batch})
    return export


def engine_step(site: Site, model: str, version: int, *, task_id: str, triton_image: str, spec,
                onnx_url: str, plan_url: str) -> StepSpec:
    from edgevision.training.export import trtexec_command
    trt = " ".join(trtexec_command("/tmp/model.onnx", "/tmp/model.plan", spec))
    return StepSpec(
        name=f"engine-{site.name}-{model}-v{version}", image=triton_image,
        command=["bash", "-c", f"set -e; {_FETCH}; {trt}; {_PUSH}; echo engine 已上傳"],
        env={"ONNX_URL": onnx_url, "PLAN_URL": plan_url, "NO_PROXY": "*", "no_proxy": "*"},
        gpu=True, gpu_claim=site.gpu_claim, node_selector=dict(site.node_selector),
        container_name=f"edgevision-job-{task_id}-{site.name}-{model}",
    )


def deploy_to_site(orch, task_id: str, site: Site, ctrl, resolved, *, only_if_present: bool = True) -> bool:
    """一個站點、一顆模型的一個版本。回 True = 部署了；False = 這個站點沒在用這顆模型（跳過）。失敗拋例外。

    orch：Orchestrator（借它的 _run_step 跑 Job、triton_image）；ctrl：該站點的 TritonController。
    """
    from edgevision.controller.deployer import ModelSpec
    from edgevision.storage.objects import ObjectStore

    model, version, meta = resolved.meta.model_name, resolved.version, resolved.meta
    if only_if_present and not ctrl.repo.versions(model):
        return False
    if ctrl.repo.has_version(model, version):
        r = ctrl.rollback(triton_model=model, to_version=version)
        if not r.ok:
            raise RuntimeError(f"站點 {site.name}：{model} v{version} 載入失敗：{r.message}")
        logger.info("站點 %s：%s v%d 已在 repo，直接載入", site.name, model, version)
        return True
    if not meta.artifacts.onnx:
        raise RuntimeError(f"站點 {site.name}：{model} v{version} 在 Registry 沒有 ONNX，沒辦法在這張卡上建 engine")
    spec = site_export_spec(meta.export, site)
    bucket, key = ObjectStore.parse_uri(meta.artifacts.onnx)
    pk = plan_key(site, model, version)
    objects = ctrl.objects
    step = engine_step(site, model, version, task_id=task_id, triton_image=orch.triton_image, spec=spec,
                       onnx_url=objects.presigned_url(bucket=bucket, key=key, expires_minutes=180, internal=True),
                       plan_url=objects.presigned_put_url(bucket="models", key=pk, expires_minutes=180,
                                                          internal=True))
    code = orch._run_step(task_id, step)
    if code != 0:
        raise RuntimeError(f"站點 {site.name}：{model} v{version} 建 engine 失敗（exit {code}，log 見任務步驟 {step.name}）")
    mspec = dataclasses.replace(ModelSpec.from_meta(meta), max_batch_size=spec.batch if spec.dynamic else 0)
    r = ctrl.deploy(triton_model=model, source_uri=f"s3://models/{pk}", version=version, spec=mspec)
    if not r.ok:
        raise RuntimeError(f"站點 {site.name}：{model} v{version} 上架失敗：{r.message}")
    logger.info("站點 %s：%s v%d 已上線（%s，batch≤%d）", site.name, model, version, site.arch, spec.batch)
    return True


def main(argv: list[str] | None = None) -> int:
    """新站點第一次上架：在 control-api 容器裡跑
        python -m edgevision.orchestrator.site_deploy <站點> <模型> [<模型> …] [--actor 名字]
    每顆模型照 Registry 現役版，在該站點的 GPU 上建 engine 並上架。進度與 log 在任務（type=deployment）裡。
    """
    import argparse
    import json

    from libs.contracts.training import TaskType

    ap = argparse.ArgumentParser()
    ap.add_argument("site")
    ap.add_argument("models", nargs="+")
    ap.add_argument("--actor", default="claude")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    from edgevision.api.control import build_state
    st = build_state()
    orch, store = st["orchestrator"], st["store"]
    task = store.create(type=TaskType.DEPLOYMENT, created_by=a.actor, channel="api",
                        params={"sync_site": a.site, "models": a.models})
    print("task", task, flush=True)
    out = orch.sync_site(task, a.site, a.models)
    print(json.dumps(out, ensure_ascii=False, indent=1))
    return 0 if all(v.endswith(" ok") for v in out.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
