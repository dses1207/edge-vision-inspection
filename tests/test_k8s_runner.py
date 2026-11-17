"""K8sRunner：StepSpec → Job / Service，以及 Job 狀態 → exit code。不需要叢集（假的 API 物件）。"""
from types import SimpleNamespace as NS

import pytest
from edgevision.orchestrator.runner import (K8sRunner, RunHandle, StepSpec, build_job_manifest,
                                       build_service_manifest)

KW = dict(namespace="edgevision", gpu_claim=None, gpu_claim_template=None, runtime_class="nvidia",
          priority_class="edgevision-training", service_account=None)


def test_trainer_step_to_job_with_shared_gpu_claim():
    spec = StepSpec(name="trainer", image="reg/edgevision-train:1", command=["python", "-m", "x"],
                    env={"TASK_ID": "t1"}, gpu=True, volumes={"edgevision-train-work": "/work"},
                    container_name="edgevision-job-T1_x-trainer", shm_size="8g")
    job = build_job_manifest(spec, **{**KW, "gpu_claim": "inference-gpu"})
    pod = job["spec"]["template"]["spec"]
    c = pod["containers"][0]
    assert job["metadata"]["name"] == "edgevision-job-t1-x-trainer"          # DNS-1123
    assert job["spec"]["backoffLimit"] == 0 and pod["restartPolicy"] == "Never"
    assert pod["resourceClaims"] == [{"name": "gpu", "resourceClaimName": "inference-gpu"}]
    assert c["resources"] == {"claims": [{"name": "gpu"}]} and pod["runtimeClassName"] == "nvidia"
    assert pod["priorityClassName"] == "edgevision-training"
    assert {"name": "dshm", "mountPath": "/dev/shm"} in c["volumeMounts"]
    assert {"name": "dshm", "emptyDir": {"medium": "Memory", "sizeLimit": "8Gi"}} in pod["volumes"]
    assert {"name": "v0", "persistentVolumeClaim": {"claimName": "edgevision-train-work"}} in pod["volumes"]
    assert c["env"] == [{"name": "TASK_ID", "value": "t1"}]
    # 指令進 args、不給 command：保留映像 ENTRYPOINT（訓練映像靠它產 clearml.conf）
    assert c["args"] == ["python", "-m", "x"] and "command" not in c


def test_gpu_step_needs_claim_or_template_and_cpu_step_has_none():
    with pytest.raises(ValueError, match="K8S_GPU_CLAIM"):
        build_job_manifest(StepSpec(name="t", image="i", command=["x"], gpu=True), **KW)
    job = build_job_manifest(StepSpec(name="finalize", image="i", command=["x"]),
                             **{**KW, "gpu_claim_template": "training-gpu"})
    pod = job["spec"]["template"]["spec"]
    assert "resourceClaims" not in pod and "runtimeClassName" not in pod
    job = build_job_manifest(StepSpec(name="t", image="i", command=["x"], gpu=True),
                             **{**KW, "gpu_claim_template": "training-gpu"})
    assert job["spec"]["template"]["spec"]["resourceClaims"][0]["resourceClaimTemplateName"] == "training-gpu"


def test_parity_step_gets_same_name_service():
    spec = StepSpec(name="parity-triton", image="triton", command=["tritonserver"], gpu=True,
                    container_name="edgevision-parity-abc", ports=[8000])
    job = build_job_manifest(spec, **{**KW, "gpu_claim": "g"})
    svc = build_service_manifest(job, spec.ports)
    assert svc["metadata"]["name"] == "edgevision-parity-abc"       # finalize 連 http://edgevision-parity-abc:8000
    assert svc["spec"]["selector"] == {"edgevision.job": "edgevision-parity-abc"}
    assert svc["spec"]["ports"][0]["port"] == 8000


def _runner(status, exit_code=0):
    batch = NS(read_namespaced_job_status=lambda n, ns: NS(status=status))
    pod = NS(metadata=NS(name="p1"),
             status=NS(container_statuses=[NS(state=NS(terminated=NS(exit_code=exit_code)))]))
    core = NS(list_namespaced_pod=lambda ns, label_selector: NS(items=[pod]),
              read_namespaced_pod=lambda n, ns: pod)
    return K8sRunner(namespace="edgevision", batch_api=batch, core_api=core)


def test_wait_maps_job_status_to_exit_code():
    h = RunHandle(ref="j", name="j")
    assert _runner(NS(succeeded=1, failed=None)).wait(h, poll_s=0) == 0
    assert _runner(NS(succeeded=None, failed=1), exit_code=3).wait(h, poll_s=0) == 3
