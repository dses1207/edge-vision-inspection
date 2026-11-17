"""runner 調和器：讓叢集上的 runner 跟資料庫的線路表一致（K8s controller / reconciliation loop）。

    資料庫（設定 API）= 想要的狀態叢集上 app=runner 的 Deployment = 實際狀態
            │                                          │
            └──────── 每 RECONCILE_INTERVAL 秒比一次 ───┘
                 啟用中、叢集沒有        → 建 runner
                 停用 / 刪掉、叢集還有    → 收掉 runner
                 設定版本或映像變了       → 換掉 Deployment（Pod 範本上的版本標註變 → K8s 自己滾動重啟）
                 用到的模型 promote 換版  → 同上（類別、imgsz 以模型為準，runner 只在啟動時讀一次）
                 都一樣                  → 不動

⭐ 為什麼要有：設定（線路、相機、幀數、門檻、ROI）存在資料庫、不經過 git，CI 不會被觸發；
   runner 只在啟動時讀一次設定。以前改完設定要人手動重啟 runner，新增 / 停用線路要等下次部署。

⭐ 為什麼靠「Pod 範本上的設定版本標註」重啟：K8s 只在 Pod 範本變了才滾動重啟，
   設定存在別處（ConfigMap、資料庫）它不會知道。把版本號寫進範本是 Helm checksum/config 的同一招。

runner 跑哪一版映像：ConfigMap edgevision-runner。
    image                = 預設，所有線都用（ai-infer 的 CI 建好新映像後改這個值，下一輪所有 runner 換過去）
    image.<線路 id>       = 只有這條線用（例 image.siteb_bravo）：新站別先在一條線試，其他線不重啟；
                           試完把預設改成同一版、刪掉這一格（刪的時候映像沒變，那條線也不會重啟）
映像只有這一個來源，不會被誰的 kubectl set image 蓋來蓋去。

哪條線跑在哪一台 GPU 節點：推論站點（sites.py，ConfigMap edgevision-inference-sites）。
不在任何站點的線用預設（Triton = triton、GPU = K8S_GPU_CLAIM）；站點的線改連它自己的 Triton、
用它的 GPU、釘在它的節點。站點設定每一輪重讀，改了就照 spec-hash 換掉那幾條線。

每一輪順便同步 go2rtc 的相機清單（edgevision/orchestrator/go2rtc_sync.py；GO2RTC_SYNC=0 關掉）。

安全規則：設定 API、ConfigMap 或站點設定讀失敗 → 這一輪什麼都不做（絕不因為「讀不到」就把 runner 全收掉）。
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlopen

from edgevision.orchestrator.sites import Site, load_sites, site_for_line

logger = logging.getLogger(__name__)

MANAGED_BY = "runner-reconciler"
ANN_VERSION = "edgevision.example.com/config-version"
ANN_HASH = "edgevision.example.com/spec-hash"
ANN_MODELS = "edgevision.example.com/model-versions"
RUNNER_IMAGE_CONFIGMAP = "edgevision-runner"
LINE_IMAGE_PREFIX = "image."


def parse_runner_images(data: dict | None) -> tuple[str, dict[str, str]]:
    """ConfigMap edgevision-runner 的 data → (預設映像, {線路 id: 這條線專用的映像})。"""
    data = data or {}
    default = (data.get("image") or "").strip()
    if not default:
        raise RuntimeError(f"ConfigMap {RUNNER_IMAGE_CONFIGMAP} 沒有 image")
    per_line = {k[len(LINE_IMAGE_PREFIX):]: v.strip() for k, v in data.items()
                if k.startswith(LINE_IMAGE_PREFIX) and (v or "").strip()}
    return default, per_line


def deployment_name(line_id: str) -> str:
    return "runner-" + line_id.replace("_", "-")


def runner_deployment(line_id: str, model: str, *, image: str, gpu_claim: str, namespace: str,
                      config_version: int = 0, encoder: str | None = None, model_versions: str = "",
                      triton: str = "triton", node_selector: dict | None = None,
                      go2rtc_source: str | None = None) -> dict:
    """一條線的 runner Deployment。純函式，可測。

    encoder 不給 = runner 自己判斷（ENCODER=auto：先開 NVENC，開不起來那台改 x264）。
    triton / gpu_claim / node_selector：推論站點（預設站點是 triton + K8S_GPU_CLAIM、不釘節點）。
    go2rtc_source：站點節點本機的 go2rtc，原始畫面從那裡拉（不用跨節點繞到主 go2rtc 再繞回來）；
                   打框畫面照樣推回主 go2rtc。不給 = 不加這個環境變數（預設站點的 Deployment 一個字都不變）。
    """
    env = {
        "LINE_ID": line_id, "CAMERA_SOURCE": "api", "CONFIG_API": "http://output-api:18080",
        "MODEL_NAME": model, "TRITON_HTTP": f"{triton}:8000", "TRITON_GRPC": f"{triton}:8001",
        "GO2RTC_RTSP": "rtsp://go2rtc:8554", "REDIS_URL": "redis://redis:6379/0",
        "RESTREAM_ENABLED": "1", "LOG_LEVEL": "INFO",
        # DRA / CDI 注入 GPU 時預設可能只給 compute：runner 要 NVDEC 解相機、NVENC 燒錄推流
        "NVIDIA_DRIVER_CAPABILITIES": "all",
        "NO_PROXY": "*", "no_proxy": "*",
    }
    if go2rtc_source:
        env["GO2RTC_SOURCE_RTSP"] = go2rtc_source
    if encoder:
        env["ENCODER"] = encoder
    labels = {"app": "runner", "edgevision.line": line_id, "app.kubernetes.io/part-of": "edgevision"}
    dep = {
        "apiVersion": "apps/v1", "kind": "Deployment",
        "metadata": {"name": deployment_name(line_id), "namespace": namespace,
                     "labels": labels | {"app.kubernetes.io/managed-by": MANAGED_BY},
                     "annotations": {}},
        "spec": {
            "replicas": 1,
            "strategy": {"type": "Recreate"},        # 同一條線不要新舊兩份同時推同一路 _marked
            "selector": {"matchLabels": {"app": "runner", "edgevision.line": line_id}},
            "template": {
                # 設定版本寫在 Pod 範本上：版本一變，K8s 自己滾動重啟這條線（見檔頭）
                "metadata": {"labels": labels, "annotations": {ANN_VERSION: str(config_version)}
                             | ({ANN_MODELS: model_versions} if model_versions else {})},
                "spec": {
                    "priorityClassName": "edgevision-inference",
                    "runtimeClassName": "nvidia",
                    "resourceClaims": [{"name": "gpu", "resourceClaimName": gpu_claim}],
                    "containers": [{
                        "name": "runner", "image": image,
                        "env": [{"name": k, "value": v} for k, v in env.items()],
                        "resources": {"claims": [{"name": "gpu"}],
                                      "requests": {"cpu": "250m", "memory": "512Mi"}},
                        "volumeMounts": [{"name": "dshm", "mountPath": "/dev/shm"}],
                    }],
                    "volumes": [{"name": "dshm", "emptyDir": {"medium": "Memory", "sizeLimit": "256Mi"}}],
                },
            },
        },
    }
    if node_selector:
        dep["spec"]["template"]["spec"]["nodeSelector"] = dict(node_selector)
    dep["metadata"]["annotations"][ANN_HASH] = spec_hash(dep)
    return dep


def spec_hash(dep: dict) -> str:
    """Deployment 我們在意的部分（spec）的雜湊：叢集上那份的標註跟這個一樣，就不用動。"""
    return hashlib.sha256(json.dumps(dep["spec"], sort_keys=True).encode()).hexdigest()[:16]


@dataclass(frozen=True)
class Action:
    kind: str                 # create / update / delete
    name: str
    reason: str
    body: dict | None = None


def plan(desired: dict[str, dict], existing: dict[str, str | None], hold: frozenset[str] = frozenset()) -> list[Action]:
    """想要的 {Deployment 名: manifest} vs 叢集上的 {Deployment 名: spec-hash 標註} → 動作清單。純函式，可測。
    hold：這一輪資訊不全的線（例：查不到模型版本）→ 叢集上那份原樣不動，不建、不換、不收。"""
    actions: list[Action] = []
    for name, dep in sorted(desired.items()):
        if name in hold:
            continue
        want = dep["metadata"]["annotations"][ANN_HASH]
        if name not in existing:
            actions.append(Action("create", name, "新線路（或 runner 被刪了）", dep))
        elif existing[name] != want:
            actions.append(Action("update", name, "設定版本 / 映像 / 部署設定變了", dep))
    for name in sorted(set(existing) - set(desired) - hold):
        actions.append(Action("delete", name, "線路停用或已刪除"))
    return actions


def model_versions(line: dict) -> str | None:
    """這條線用到的模型與現役版本（設定 API 從 Registry 附上的 model.version）："seg_alpha:3,ppe:1"。
    有任何一顆查不到版本 → None。"""
    models = [line.get("model") or {}] + [e.get("model") or {} for e in line.get("extra_stations") or []]
    if any(not m.get("name") or m.get("version") is None for m in models):
        return None
    return ",".join(f"{m['name']}:{m['version']}" for m in models)


def desired_runners(lines: list[dict], details: dict[str, dict], *, image: str, gpu_claim: str,
                    namespace: str, hold: set[str] | None = None,
                    sites: dict[str, Site] | None = None,
                    line_images: dict[str, str] | None = None) -> dict[str, dict]:
    """設定 API 的線路表 → {Deployment 名: manifest}。沒有 model.name 的線跳過（記 log，不建錯的 runner）。
    line_images：個別線專用的映像（沒列的線用 image）。
    查不到模型版本的線放進 hold（呼叫端傳入的 set）：這一輪不動它，免得 Registry 一時連不上就把所有線重啟一輪。"""
    out: dict[str, dict] = {}
    for meta in lines:
        lid = meta["id"]
        if not meta.get("enabled", True):
            continue
        line = details.get(lid) or {}
        model = (line.get("model") or {}).get("name")
        if not model:
            logger.error("線路 %s 沒有 model.name，不建 runner（請在線路設定補上模型名）", lid)
            continue
        if not any(c.get("enabled", True) for c in line.get("cameras", [])):
            logger.warning("線路 %s 沒有啟用中的相機，不建 runner", lid)
            continue
        mv = model_versions(line)
        if mv is None and hold is not None:
            logger.warning("線路 %s 查不到模型現役版本（Registry 沒登錄或連不上），這一輪不動它", lid)
            hold.add(deployment_name(lid))
        site = site_for_line(sites or {}, lid)
        dep = runner_deployment(lid, model, image=(line_images or {}).get(lid) or image, namespace=namespace,
                                config_version=int(line.get("config_version") or meta.get("desired_version") or 0),
                                model_versions=mv or "",
                                gpu_claim=site.gpu_claim if site else gpu_claim,
                                triton=site.triton if site else "triton",
                                node_selector=site.node_selector if site else None,
                                go2rtc_source=site.go2rtc_source if site else None)
        out[dep["metadata"]["name"]] = dep
    return out


class Reconciler:
    def __init__(self, *, config_api: str, namespace: str, gpu_claim: str, apps_api=None, core_api=None,
                 go2rtc_sync=None) -> None:
        if apps_api is None or core_api is None:
            from kubernetes import client, config
            try:
                config.load_incluster_config()
            except Exception:
                config.load_kube_config()
            apps_api = apps_api or client.AppsV1Api()
            core_api = core_api or client.CoreV1Api()
        self.apps, self.core = apps_api, core_api
        self.config_api = config_api.rstrip("/")
        self.namespace, self.gpu_claim = namespace, gpu_claim
        self.go2rtc_sync = go2rtc_sync        # None = 不管 go2rtc 相機清單（edgevision/orchestrator/go2rtc_sync.py）

    def _get(self, path: str):
        with urlopen(f"{self.config_api}{path}", timeout=10) as r:
            return json.load(r)

    def runner_images(self) -> tuple[str, dict[str, str]]:
        cm = self.core.read_namespaced_config_map(RUNNER_IMAGE_CONFIGMAP, self.namespace)
        return parse_runner_images(cm.data)

    def existing(self) -> dict[str, str | None]:
        deps = self.apps.list_namespaced_deployment(self.namespace, label_selector="app=runner")
        return {d.metadata.name: (d.metadata.annotations or {}).get(ANN_HASH) for d in deps.items}

    def once(self) -> list[Action]:
        """跑一輪。讀不到任何一個來源就整輪放棄（不做半套）。"""
        try:
            image, line_images = self.runner_images()
            lines = self._get("/api/config/lines")
            # 停用的線也讀：go2rtc 清單含停用的相機（啟用 / 停用時 go2rtc 不用重啟）；runner 只建啟用的
            details = {m["id"]: self._get(f"/api/config/lines/{m['id']}") for m in lines}
            existing = self.existing()
            sites = load_sites()
            desired_hold: set[str] = set()
            desired = desired_runners(lines, details, image=image, gpu_claim=self.gpu_claim,
                                      namespace=self.namespace, hold=desired_hold, sites=sites,
                                      line_images=line_images)
        except Exception as exc:
            logger.warning("這一輪略過（讀不到設定 API / 叢集 / 站點設定）：%s", exc)
            return []
        hold = desired_hold
        if self.go2rtc_sync is not None:
            self.go2rtc_sync.once(list(details.values()), sites)
        actions = plan(desired, existing, frozenset(hold))
        for a in actions:
            try:
                self.apply(a)
                logger.warning("[%s] %s：%s", a.name, {"create": "建立", "update": "更新", "delete": "收掉"}[a.kind], a.reason)
            except Exception:
                logger.exception("[%s] %s 失敗，下一輪再試", a.name, a.kind)
        return actions

    def apply(self, a: Action) -> None:
        if a.kind == "create":
            self.apps.create_namespaced_deployment(self.namespace, a.body)
        elif a.kind == "update":
            cur = self.apps.read_namespaced_deployment(a.name, self.namespace)
            body = dict(a.body)
            body["metadata"] = dict(body["metadata"], resourceVersion=cur.metadata.resource_version)
            self.apps.replace_namespaced_deployment(a.name, self.namespace, body)
        elif a.kind == "delete":
            self.apps.delete_namespaced_deployment(a.name, self.namespace)

    def run(self, interval: float) -> None:
        logger.info("runner 調和器啟動：每 %.0f 秒比對 %s 與 namespace %s", interval, self.config_api, self.namespace)
        while True:
            self.once()
            time.sleep(interval)


def main() -> None:
    logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"),
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")
    ns_file = Path("/var/run/secrets/kubernetes.io/serviceaccount/namespace")
    namespace = os.environ.get("K8S_NAMESPACE") or (ns_file.read_text().strip() if ns_file.exists() else "edgevision")
    r = Reconciler(config_api=os.environ.get("CONFIG_API", "http://output-api:18080"), namespace=namespace,
                   gpu_claim=os.environ.get("K8S_GPU_CLAIM", "inference-gpu"))
    if os.environ.get("GO2RTC_SYNC", "1") != "0":
        from edgevision.orchestrator.go2rtc_sync import Go2rtcSync
        r.go2rtc_sync = Go2rtcSync(core_api=r.core, apps_api=r.apps, namespace=namespace,
                                   relay=os.environ.get("GO2RTC_RELAY") or None)
    r.run(
        float(os.environ.get("RECONCILE_INTERVAL", 30)))
