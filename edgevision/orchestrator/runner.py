"""JobRunner：執行「一個步驟 = 一個容器」的介面與實作。

StepSpec 是與執行環境無關的描述（映像、指令、環境變數、掛載、要不要 GPU），
DockerRunner 把它翻成 docker run，K8sRunner 翻成 Job / Pod。步驟定義本身不變。
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Protocol

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class StepSpec:
    name: str                       # trainer | trtexec | parity-triton | finalize
    image: str
    command: list[str]
    env: dict[str, str] = field(default_factory=dict)
    gpu: bool = False
    volumes: dict[str, str] = field(default_factory=dict)   # {volume 名或主機路徑: 容器內路徑}
    network: str | None = None
    container_name: str | None = None
    labels: dict[str, str] = field(default_factory=dict)
    entrypoint: list[str] | None = None
    shm_size: str | None = None      # 訓練 dataloader 需要較大的 /dev/shm
    # 別的步驟要用名稱連進來的埠（parity-triton 的 8000）。Docker 靠容器名 DNS 不用宣告；
    # K8s 的 Job Pod 沒有穩定名稱，K8sRunner 會照這個建一個同名 Service。
    ports: list[int] = field(default_factory=list)
    # 指定這一步用哪張 GPU、跑在哪個節點（推論站點建 engine：engine 綁 GPU 架構，要在那張卡上建）。
    # 不給 = K8sRunner 的預設（K8S_GPU_CLAIM / 模板）。Docker 版忽略。
    gpu_claim: str | None = None
    node_selector: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class RunHandle:
    ref: str                         # 容器 id / Job 名
    name: str
    started_at: float = field(default_factory=time.time)


class JobRunner(Protocol):
    def start(self, spec: StepSpec) -> RunHandle: ...
    def wait(self, handle: RunHandle, *, cancel_check: Callable[[], bool] | None = None,
             poll_s: float = 5.0) -> int: ...
    def logs(self, handle: RunHandle, *, tail: int | None = None) -> str: ...
    def stop(self, handle: RunHandle) -> None: ...
    def remove(self, handle: RunHandle) -> None: ...
    def is_running(self, ref: str) -> bool: ...
    def find(self, name: str) -> RunHandle | None: ...


# ═══════════════════════════════════════════════════════════

class DockerRunner:
    """docker SDK 實作。控制 API 容器掛 /var/run/docker.sock 即可。"""

    def __init__(self, *, default_network: str | None = None, gpu_ids: str | None = None) -> None:
        try:
            import docker
        except ImportError as exc:
            raise RuntimeError("需要 docker：pip install docker") from exc
        self._docker = docker
        self.client = docker.from_env()
        self.network = default_network or os.environ.get("DOCKER_NETWORK", "edgevision_edgevision")
        # 空字串 = 全部 GPU；"0" / "0,1" = 指定卡（K8s 對應 resources.limits nvidia.com/gpu）
        self.gpu_ids = gpu_ids if gpu_ids is not None else os.environ.get("TRAIN_GPU_IDS", "")

    def _device_requests(self):
        DR = self._docker.types.DeviceRequest
        if self.gpu_ids.strip():
            return [DR(device_ids=[g.strip() for g in self.gpu_ids.split(",") if g.strip()], capabilities=[["gpu"]])]
        return [DR(count=-1, capabilities=[["gpu"]])]

    def start(self, spec: StepSpec) -> RunHandle:
        name = spec.container_name or f"edgevision-job-{spec.name}-{int(time.time())}"
        # 同名殘留（上次中斷）先清掉
        old = self.find(name)
        if old:
            logger.warning("清掉殘留容器 %s", name)
            self.remove(old)
        kw = dict(
            image=spec.image, command=spec.command, name=name, detach=True,
            environment=spec.env, network=spec.network or self.network,
            volumes={src: {"bind": dst, "mode": "rw"} for src, dst in spec.volumes.items()},
            labels={"edgevision.step": spec.name, **spec.labels},
        )
        if spec.entrypoint is not None:
            kw["entrypoint"] = spec.entrypoint
        if spec.gpu:
            kw["device_requests"] = self._device_requests()
        if spec.shm_size:
            kw["shm_size"] = spec.shm_size
        c = self.client.containers.run(**kw)
        logger.info("步驟 %s 已啟動：%s（%s）", spec.name, name, c.short_id)
        return RunHandle(ref=c.id, name=name)

    def wait(self, handle: RunHandle, *, cancel_check: Callable[[], bool] | None = None,
             poll_s: float = 5.0) -> int:
        c = self.client.containers.get(handle.ref)
        while True:
            c.reload()
            if c.status in ("exited", "dead"):
                return int(c.attrs["State"].get("ExitCode", 1))
            if cancel_check and cancel_check():
                logger.info("收到取消，停止 %s", handle.name)
                self.stop(handle)
                return 2
            time.sleep(poll_s)

    def logs(self, handle: RunHandle, *, tail: int | None = None) -> str:
        try:
            c = self.client.containers.get(handle.ref)
            return c.logs(tail=tail or "all").decode("utf-8", "replace")
        except Exception as exc:
            return f"<無法取得 log：{exc}>"

    def stop(self, handle: RunHandle) -> None:
        try:
            self.client.containers.get(handle.ref).stop(timeout=20)
        except Exception as exc:
            logger.debug("stop %s：%s", handle.name, exc)

    def remove(self, handle: RunHandle) -> None:
        try:
            self.client.containers.get(handle.ref).remove(force=True)
        except Exception as exc:
            logger.debug("remove %s：%s", handle.name, exc)

    def is_running(self, ref: str) -> bool:
        try:
            c = self.client.containers.get(ref)
            return c.status in ("running", "created", "restarting")
        except Exception:
            return False

    def find(self, name: str) -> RunHandle | None:
        try:
            c = self.client.containers.get(name)
            return RunHandle(ref=c.id, name=name)
        except Exception:
            return None

    def ping(self) -> bool:
        try:
            return bool(self.client.ping())
        except Exception:
            return False


# ═══════════════════════════════════════════════════════════

def _k8s_name(name: str) -> str:
    """Job / Service 名稱要是 DNS-1123：小寫英數與 -，最長 63。"""
    import re
    n = re.sub(r"[^a-z0-9-]+", "-", name.lower()).strip("-")
    return n[:63].rstrip("-") or "edgevision-step"


def _shm_quantity(size: str) -> str:
    """docker 的 "8g" → K8s 的 "8Gi"。"""
    size = size.strip()
    units = {"k": "Ki", "m": "Mi", "g": "Gi", "t": "Ti"}
    if size and size[-1].lower() in units and size[:-1].isdigit():
        return size[:-1] + units[size[-1].lower()]
    return size


def build_job_manifest(spec: StepSpec, *, namespace: str, gpu_claim: str | None,
                       gpu_claim_template: str | None, runtime_class: str | None,
                       priority_class: str | None, service_account: str | None,
                       env_from: list[dict] | None = None) -> dict:
    """StepSpec → batch/v1 Job（dict）。純函式，可測。

    對應關係：
      image / command / entrypoint / env → containers[0]
      volumes {名稱: 路徑}               → PVC（名稱就是 claimName；K8s 上 TRAIN_WORK_VOLUME 設成 PVC 名）
      gpu=True                           → DRA：resourceClaims + runtimeClassName（沒有 device plugin）
          gpu_claim           共用一個既有的 ResourceClaim（單卡叢集：訓練跟推論用同一張卡）
          gpu_claim_template  每個 Job 各自申請（有專用訓練卡時）
      shm_size                           → emptyDir(Memory) 掛 /dev/shm
      backoffLimit 0                     → 失敗不自動重跑：Orchestrator 自己管重試與續跑
    """
    name = _k8s_name(spec.container_name or f"edgevision-job-{spec.name}-{int(time.time())}")
    labels = {"app.kubernetes.io/part-of": "edgevision", "edgevision.step": _k8s_name(spec.name), "edgevision.job": name,
              **{k: _k8s_name(str(v)) for k, v in spec.labels.items()}}
    container: dict = {"name": "step", "image": spec.image,
                       "env": [{"name": k, "value": str(v)} for k, v in spec.env.items()]}
    if spec.entrypoint is not None:
        container["command"] = list(spec.entrypoint)
        container["args"] = list(spec.command)
    else:
        # Docker 的 command 是 CMD，K8s 的 command 卻是 ENTRYPOINT：寫成 command 會跳過映像入口。
        # 訓練映像的 /entrypoint.sh 要產 clearml.conf（MinIO 憑證），跳過就 ClearML 上傳失敗。
        container["args"] = list(spec.command)
    if env_from:
        container["envFrom"] = env_from
    if spec.ports:
        container["ports"] = [{"containerPort": int(p)} for p in spec.ports]
    mounts, volumes = [], []
    for i, (src, dst) in enumerate(spec.volumes.items()):
        vname = f"v{i}"
        volumes.append({"name": vname, "persistentVolumeClaim": {"claimName": src}})
        mounts.append({"name": vname, "mountPath": dst})
    if spec.shm_size:
        volumes.append({"name": "dshm",
                        "emptyDir": {"medium": "Memory", "sizeLimit": _shm_quantity(spec.shm_size)}})
        mounts.append({"name": "dshm", "mountPath": "/dev/shm"})
    if mounts:
        container["volumeMounts"] = mounts
    pod: dict = {"restartPolicy": "Never", "containers": [container]}
    if volumes:
        pod["volumes"] = volumes
    if service_account:
        pod["serviceAccountName"] = service_account
    if priority_class:
        pod["priorityClassName"] = priority_class
    if spec.node_selector:
        pod["nodeSelector"] = dict(spec.node_selector)
    if spec.gpu:
        if spec.gpu_claim:
            pod["resourceClaims"] = [{"name": "gpu", "resourceClaimName": spec.gpu_claim}]
        elif gpu_claim:
            pod["resourceClaims"] = [{"name": "gpu", "resourceClaimName": gpu_claim}]
        elif gpu_claim_template:
            pod["resourceClaims"] = [{"name": "gpu", "resourceClaimTemplateName": gpu_claim_template}]
        else:
            raise ValueError("步驟要 GPU，但沒設 K8S_GPU_CLAIM 或 K8S_GPU_CLAIM_TEMPLATE")
        container["resources"] = {"claims": [{"name": "gpu"}]}
        if runtime_class:
            pod["runtimeClassName"] = runtime_class
    return {
        "apiVersion": "batch/v1", "kind": "Job",
        "metadata": {"name": name, "namespace": namespace, "labels": labels},
        "spec": {"backoffLimit": 0, "template": {"metadata": {"labels": labels}, "spec": pod}},
    }


def build_service_manifest(job: dict, ports: list[int]) -> dict:
    """給需要被連的步驟（parity-triton）一個同名 Service，選到那個 Job 的 Pod。"""
    md = job["metadata"]
    return {
        "apiVersion": "v1", "kind": "Service",
        "metadata": {"name": md["name"], "namespace": md["namespace"], "labels": md["labels"]},
        "spec": {"selector": {"edgevision.job": md["name"]},
                 "ports": [{"name": f"p{p}", "port": int(p), "targetPort": int(p)} for p in ports]},
    }


class K8sRunner:
    """JobRunner 的 K8s 實作：一個步驟 = 一個 Job。control-api 用 ServiceAccount 在自己的 namespace 建 Job。

    設定（環境變數）：
      K8S_NAMESPACE            預設讀 ServiceAccount 掛載的 namespace 檔，再不行 edgevision
      K8S_GPU_CLAIM            GPU 步驟共用的既有 ResourceClaim（單卡：跟推論同一張）
      K8S_GPU_CLAIM_TEMPLATE   GPU 步驟各自申請用的 ResourceClaimTemplate（有專用訓練卡時）
      K8S_RUNTIME_CLASS        GPU Pod 的 runtimeClassName，預設 nvidia
      K8S_PRIORITY_CLASS       預設 edgevision-training（比推論低，資源緊時先犧牲訓練）
      K8S_JOB_SERVICE_ACCOUNT  步驟 Pod 用的 ServiceAccount（預設 namespace 的 default）
      K8S_JOB_ENV_FROM         逗號分隔的 ConfigMap / Secret 名（configmap:x,secret:y），整包掛進步驟
    """

    def __init__(self, *, namespace: str | None = None, batch_api=None, core_api=None) -> None:
        if batch_api is None or core_api is None:
            from kubernetes import client, config
            try:
                config.load_incluster_config()
            except Exception:
                config.load_kube_config()
            batch_api = batch_api or client.BatchV1Api()
            core_api = core_api or client.CoreV1Api()
        self.batch, self.core = batch_api, core_api
        env = os.environ
        ns_file = Path("/var/run/secrets/kubernetes.io/serviceaccount/namespace")
        self.namespace = namespace or env.get("K8S_NAMESPACE") or (
            ns_file.read_text().strip() if ns_file.exists() else "edgevision")
        self.gpu_claim = env.get("K8S_GPU_CLAIM") or None
        self.gpu_claim_template = env.get("K8S_GPU_CLAIM_TEMPLATE") or None
        self.runtime_class = env.get("K8S_RUNTIME_CLASS", "nvidia") or None
        self.priority_class = env.get("K8S_PRIORITY_CLASS", "edgevision-training") or None
        self.service_account = env.get("K8S_JOB_SERVICE_ACCOUNT") or None
        self.env_from = []
        for item in (env.get("K8S_JOB_ENV_FROM") or "").split(","):
            kind, _, ref = item.strip().partition(":")
            if kind == "configmap" and ref:
                self.env_from.append({"configMapRef": {"name": ref}})
            elif kind == "secret" and ref:
                self.env_from.append({"secretRef": {"name": ref}})

    # ── JobRunner ────────────────────────────────────────
    def start(self, spec: StepSpec) -> RunHandle:
        job = build_job_manifest(spec, namespace=self.namespace, gpu_claim=self.gpu_claim,
                                 gpu_claim_template=self.gpu_claim_template, runtime_class=self.runtime_class,
                                 priority_class=self.priority_class, service_account=self.service_account,
                                 env_from=self.env_from or None)
        name = job["metadata"]["name"]
        if self.find(name):
            logger.warning("清掉殘留 Job %s", name)
            self.remove(RunHandle(ref=name, name=name))
            self._wait_gone(name)
        self.batch.create_namespaced_job(self.namespace, job)
        if spec.ports:
            self.core.create_namespaced_service(self.namespace, build_service_manifest(job, spec.ports))
        logger.info("步驟 %s 已建立 Job %s", spec.name, name)
        return RunHandle(ref=name, name=name)

    def wait(self, handle: RunHandle, *, cancel_check: Callable[[], bool] | None = None,
             poll_s: float = 5.0) -> int:
        while True:
            try:
                st = self.batch.read_namespaced_job_status(handle.ref, self.namespace).status
            except Exception as exc:
                logger.warning("讀 Job %s 失敗：%s", handle.ref, exc)
                return 1
            if st.succeeded:
                return 0
            if st.failed:
                return self._exit_code(handle.ref)
            if cancel_check and cancel_check():
                logger.info("收到取消，停止 %s", handle.name)
                self.stop(handle)
                return 2
            time.sleep(poll_s)

    def logs(self, handle: RunHandle, *, tail: int | None = None) -> str:
        pod = self._pod(handle.ref)
        if pod is None:
            return f"<找不到 {handle.ref} 的 Pod>"
        try:
            kw = {"tail_lines": tail} if tail else {}
            return self.core.read_namespaced_pod_log(pod, self.namespace, **kw)
        except Exception as exc:
            return f"<無法取得 log：{exc}>"

    def stop(self, handle: RunHandle) -> None:
        self.remove(handle)

    def remove(self, handle: RunHandle) -> None:
        for fn in (lambda: self.batch.delete_namespaced_job(handle.ref, self.namespace,
                                                            propagation_policy="Background"),
                   lambda: self.core.delete_namespaced_service(handle.ref, self.namespace)):
            try:
                fn()
            except Exception as exc:     # 不存在就算了（Service 只有 parity 有）
                logger.debug("remove %s：%s", handle.ref, exc)

    def is_running(self, ref: str) -> bool:
        try:
            st = self.batch.read_namespaced_job_status(ref, self.namespace).status
        except Exception:
            return False
        return not (st.succeeded or st.failed)

    def find(self, name: str) -> RunHandle | None:
        name = _k8s_name(name)
        try:
            self.batch.read_namespaced_job(name, self.namespace)
            return RunHandle(ref=name, name=name)
        except Exception:
            return None

    def ping(self) -> bool:
        try:
            self.batch.list_namespaced_job(self.namespace, limit=1)
            return True
        except Exception:
            return False

    # ── 內部 ─────────────────────────────────────────────
    def _pod(self, job: str) -> str | None:
        try:
            pods = self.core.list_namespaced_pod(self.namespace, label_selector=f"edgevision.job={job}").items
        except Exception:
            return None
        return pods[-1].metadata.name if pods else None

    def _exit_code(self, job: str) -> int:
        pod = self._pod(job)
        try:
            p = self.core.read_namespaced_pod(pod, self.namespace) if pod else None
            term = p.status.container_statuses[0].state.terminated if p else None
            return int(term.exit_code) if term and term.exit_code else 1
        except Exception:
            return 1

    def _wait_gone(self, name: str, timeout: float = 60.0) -> None:
        t0 = time.time()
        while time.time() - t0 < timeout and self.find(name):
            time.sleep(1)


def save_logs(runner: JobRunner, handle: RunHandle, log_dir: Path, step: str) -> Path:
    log_dir.mkdir(parents=True, exist_ok=True)
    p = log_dir / f"{step}.log"
    p.write_text(runner.logs(handle))
    return p
