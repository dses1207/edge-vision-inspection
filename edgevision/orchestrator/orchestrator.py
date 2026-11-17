"""Orchestrator：task → 步驟序列 → 狀態、心跳、對帳、放行。

步驟（訓練任務）：
    ① trainer        GPU，trainer 映像資料 → 訓練 → Gate → ONNX → 註冊(candidate)
    ② trtexec        GPU，Triton 映像      onnx → plan（Triton 自帶 trtexec）
    ③ parity-triton  GPU，Triton 映像臨時服務，只載這一版
    ④ finalize       CPU，edgevision 映像       TensorRT 一致性 → 上傳 plan → 完成
    ⑤ (auto_promote) Registry 移 tag → 部署 Controller → Triton 熱載入

每步完成寫進 tasks.result.steps，重啟後從未完成的步驟續跑（冪等）。
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import threading
import time
from pathlib import Path

from libs.contracts import TrainingJobRequest
from libs.contracts.model_meta import TAG_PRODUCTION
from libs.contracts.training import TaskStatus, TaskType, TaskView

from edgevision.orchestrator.runner import JobRunner, RunHandle, StepSpec, save_logs
from edgevision.training.tracker import TaskStore

logger = logging.getLogger(__name__)

# 用 GPU 的任務類型：共用同一個排隊、同一個訓練切片（例如一張卡切出的 MIG 切片）
GPU_TYPES = [TaskType.TRAINING.value, TaskType.AUTOLABEL.value, TaskType.VIDEO_TEST.value]
# 一次 TRAIN_MAX_CONCURRENT 個的任務（訓練 / 自動標注）。
# 影片測試（臨時 Triton + runner）彼此可以同時跑，上限 VIDEO_TEST_MAX_CONCURRENT；
# 訓練在跑時最多 VIDEO_TEST_MAX_WITH_TRAINING 個（10/21 實測：影片測試一個約 12.5 GB、訓練最多約 24 GB，
# 訓練＋1 個 ≈ 37.5 GB 放得下，＋2 個 ≈ 50 GB 會爆）。自動標注沒量過，照舊跟影片測試互斥。
EXCLUSIVE_TYPES = [TaskType.TRAINING.value, TaskType.AUTOLABEL.value]
TYPE_LABELS = {TaskType.TRAINING.value: "訓練", TaskType.AUTOLABEL.value: "預標", TaskType.VIDEO_TEST.value: "影片測試"}

# 傳給步驟容器的環境變數（前綴比對）。機密只在這條路上流動，不落檔。
PASS_ENV_PREFIXES = ("MINIO_", "DATABASE_URL", "CLEARML_", "LAKEFS_", "TRITON_", "LOG_LEVEL",
                     "CODE_COMMIT", "TRAIN_", "NO_PROXY", "no_proxy", "AWS_", "VIDEO_TEST_")


# 訓練容器出外網（ultralytics 下載 AMP 檢查用的 yolo26n.pt 等）：TRAIN_HTTPS_PROXY 有設才帶 proxy。
# 叢集 Pod 不會自動吃 proxy（叢集管理者），其他容器一律 NO_PROXY=*。帶 proxy 時 NO_PROXY 改成
# TRAIN_NO_PROXY（內網網段）＋ 傳進容器的服務位址（minio、lakefs、clearml-apiserver…短名不在網段清單裡）。
DEFAULT_TRAIN_NO_PROXY = ("localhost,127.0.0.1,192.0.2.31/8,192.0.2.31/16,192.0.2.31/16,192.0.2.31/16,"
                          ".svc,.cluster.local,.example.com")


def proxy_env(env: dict[str, str]) -> dict[str, str]:
    """env = 要傳給容器的變數。回傳要蓋上去的 proxy 變數；沒設 TRAIN_HTTPS_PROXY 回空 dict。"""
    proxy = env.get("TRAIN_HTTPS_PROXY", "").strip()
    if not proxy:
        return {}
    hosts = [h.strip() for h in env.get("TRAIN_NO_PROXY", DEFAULT_TRAIN_NO_PROXY).split(",") if h.strip()]
    for k, v in sorted(env.items()):
        if not (k.endswith(("_URL", "_ENDPOINT", "_HOST")) and v):
            continue
        rest = v.split("://", 1)[-1]                       # http://minio:9000/x、postgres://u:p@db:5432、minio:9000
        host = rest.split("/", 1)[0].rsplit("@", 1)[-1].split(":", 1)[0]
        if host and host not in hosts:
            hosts.append(host)
    no = ",".join(hosts)
    return {"HTTP_PROXY": proxy, "HTTPS_PROXY": proxy, "http_proxy": proxy, "https_proxy": proxy,
            "NO_PROXY": no, "no_proxy": no}


# 要讀 NAS 24h 錄影的匯入：NAS 時段抽圖、告警回流（取告警那一幀）、剪錄影片段
NAS_IMPORT_KINDS = ("nas", "alert", "clip")


class Orchestrator:
    def __init__(self, runner: JobRunner, store: TaskStore, *, registry_factory=None,
                 deployer_factory=None, audit=None, site_deployer_factory=None, sites_loader=None) -> None:
        self.runner = runner
        self.store = store
        self._registry_factory = registry_factory
        self._deployer_factory = deployer_factory
        # 其他推論站點（sites.py）：site → 那個站點 Triton 的 TritonController。沒給 = 只有主 Triton
        self._site_deployer_factory = site_deployer_factory
        self._sites_loader = sites_loader
        self._audit = audit                     # callable(**kw) → db.write_audit
        self._threads: dict[str, threading.Thread] = {}
        self._stop = threading.Event()

        env = os.environ
        self.trainer_image = env.get("TRAINER_IMAGE", "edgevision/trainer:dev")
        self.edgevision_image = env.get("EDGEVISION_IMAGE", "edgevision:mvp")
        self.triton_image = env.get("TRITON_IMAGE", "nvcr.io/nvidia/tritonserver:24.09-py3")
        self.work_volume = env.get("TRAIN_WORK_VOLUME", "edgevision_train-work")
        self.work_dir = Path(env.get("TRAIN_WORK_DIR", "/work"))
        self.heartbeat_timeout = int(env.get("TASK_HEARTBEAT_TIMEOUT", "1800"))
        # 同時可執行的訓練任務數。單卡預設 1：訓練會吃滿 GPU，兩個一起跑只會互相拖慢或 OOM。
        # （ClearML agent 預設也是一次一個任務，理由相同。）
        self.max_concurrent = max(1, int(env.get("TRAIN_MAX_CONCURRENT", "1")))
        # 影片測試彼此可以同時跑幾個；訓練在跑時最多幾個（0 = 訓練時不跑影片測試）
        self.video_max_concurrent = max(1, int(env.get("VIDEO_TEST_MAX_CONCURRENT", "2")))
        self.video_max_with_training = max(0, int(env.get("VIDEO_TEST_MAX_WITH_TRAINING", "1")))
        # 工作區保留幾個任務份（資料集 + 權重，一次動輒數 GB）。0 = 不清。
        # ⚠️ 這台機器有過 698G log 撐爆磁碟的前科，長跑的東西一律要有回收機制。
        self.keep_workspaces = int(env.get("TRAIN_KEEP_WORKSPACES", "5"))
        # 除錯用：失敗的步驟容器不刪，才能進去看它拿到什麼環境。平時關閉避免容器堆積。
        self.keep_failed_containers = env.get("TRAIN_KEEP_FAILED_CONTAINERS", "").lower() in ("1", "true", "yes")
        self.trt_parity = env.get("TRT_PARITY", "true").lower() in ("1", "true", "yes")
        # 圖片庫從 NAS 24h 錄影擷取：匯入 Job 掛這個 PVC 到 /nas（沒設 = 不支援 NAS 來源，例如測試叢集）
        self.media_nas_pvc = env.get("MEDIA_NAS_PVC", "").strip() or None
        # 影片測試用線上同一個 runner 映像：跟 runner-reconciler 讀同一個 ConfigMap（edgevision-runner），掛成檔案
        self.runner_image_file = Path(env.get("RUNNER_IMAGE_FILE", "/etc/edgevision/runner/image"))

    # ── 對外 ─────────────────────────────────────────────

    def submit(self, task_id: str) -> None:
        """交付 GPU 任務（訓練 / 自動標注）。GPU 有空位就直接跑，否則留在 queued 由 _drain_queue 放行。"""
        t = self.store.get(task_id)
        type_ = t.type if t else TaskType.TRAINING.value
        # 前面還有人排隊就照順序（不插隊）；輪到時由 _drain_queue 放行
        if self.store.queue_position(task_id) in (None, 1) and self._has_slot(type_) and self.store.claim(task_id):
            self._spawn(task_id, self._target_for(type_))
        else:
            self._drain_queue()          # 前面的訓練只是在等訓練名額 → 影片測試可能可以先跑
            logger.info("task %s 進入佇列（執行中 %d/%d）", task_id,
                        self.store.running_count(types=GPU_TYPES), self.max_concurrent)

    def _target_for(self, type_: str):
        return {TaskType.AUTOLABEL.value: self._run_autolabel,
                TaskType.VIDEO_TEST.value: self._run_video_test}.get(type_, self._run_training)

    def _has_slot(self, type_: str | None = None) -> bool:
        """影片測試：沒有自動標注在跑；影片測試未滿 video_max_concurrent（訓練在跑時未滿 video_max_with_training）。
        訓練：獨佔任務未滿 max_concurrent，而且影片測試不超過 video_max_with_training 個。
        自動標注：獨佔任務未滿 max_concurrent，而且沒有影片測試在跑。"""
        excl = self.store.running_count(types=EXCLUSIVE_TYPES)
        training = self.store.running_count(type=TaskType.TRAINING.value)
        vt = self.store.running_count(type=TaskType.VIDEO_TEST.value)
        if type_ == TaskType.VIDEO_TEST.value:
            if excl > training:                        # 有自動標注在跑
                return False
            cap = self.video_max_concurrent if training == 0 else min(self.video_max_concurrent,
                                                                     self.video_max_with_training)
            return vt < cap
        if type_ == TaskType.AUTOLABEL.value:
            return vt == 0 and excl < self.max_concurrent
        return vt <= self.video_max_with_training and excl < self.max_concurrent

    @staticmethod
    def _task_label(t: TaskView) -> str:
        """排隊訊息裡怎麼稱呼一個任務：task id（種類，模型或線路）。"""
        p = t.params or {}
        req = p.get("request") or {}
        what = (p.get("model_name") or req.get("model_name") or (req.get("config") or {}).get("model_name")
                or p.get("line_id"))
        return f"{t.task_id}（{TYPE_LABELS.get(t.type, t.type)}{'，' + what if what else ''}）"

    def _update_queue_messages(self) -> None:
        """排隊中的 GPU 任務：寫清楚前面還有幾個、GPU 上正在跑誰（不然只看到「前面還有 N 個」不知道卡在什麼）。"""
        queued = sorted((t for t in self.store.queued() if t.type in GPU_TYPES), key=lambda t: t.created_at)
        if not queued:
            return
        running = [t for t in self.store.working() if t.type in GPU_TYPES]
        on_gpu = "、".join(self._task_label(t) for t in running) or "沒有（等排程放行）"
        for i, t in enumerate(queued):
            ahead = f"前面還有 {i} 個排隊" + (f"，下一個是 {self._task_label(queued[i - 1])}" if i else "")
            msg = f"排隊中（{ahead}；GPU 上：{on_gpu}）"
            if t.status_message != msg:
                self.store.heartbeat(t.task_id, msg)

    def _drain_queue(self) -> None:
        """照排隊順序放行：最前面那個有空位就跑，沒有就停（不讓後面的插隊，訓練才不會被影片測試一直擠掉）；
        唯一例外見 _video_test_may_pass。
        由 _monitor 定期呼叫，任務結束時也呼叫一次。"""
        while True:
            queued = sorted((t for t in self.store.queued() if t.type in GPU_TYPES), key=lambda t: t.created_at)
            if not queued:
                return
            pick = queued[0] if self._has_slot(queued[0].type) else self._video_test_may_pass(queued)
            if pick is None:
                self._update_queue_messages()
                return
            if self.store.claim(pick.task_id):
                self._spawn(pick.task_id, self._target_for(pick.type))

    def _video_test_may_pass(self, queued: list[TaskView]) -> TaskView | None:
        """排在前面的都是訓練、而且只是因為訓練名額滿了在等（不是在等影片測試變少）→ 後面的影片測試可以先跑，
        上限照 video_max_with_training，所以前一個訓練跑完時，下一個訓練照樣能開始，不會被拖到。
        10/21：兩個 ppe 訓練（各 200 epochs）排在前面，6 個 10 分鐘的影片測試要白等 4 小時。
        前面有自動標注就不插（它要等影片測試全部跑完）。"""
        excl = self.store.running_count(types=EXCLUSIVE_TYPES)
        vt = self.store.running_count(type=TaskType.VIDEO_TEST.value)
        if excl < self.max_concurrent or vt > self.video_max_with_training:
            return None
        for t in queued:
            if t.type == TaskType.VIDEO_TEST.value:
                return t if self._has_slot(t.type) else None
            if t.type != TaskType.TRAINING.value:
                return None
        return None

    def submit_media_import(self, task_id: str) -> None:
        """圖片庫匯入：CPU Job，不佔 GPU 排隊，馬上跑。"""
        self._spawn(task_id, self._run_media_import)

    def submit_media_export(self, task_id: str) -> None:
        """圖片集匯出：CPU Job，不佔 GPU 排隊，馬上跑。"""
        self._spawn(task_id, self._run_media_export)

    def submit_dataset_mix(self, task_id: str) -> None:
        """混合成訓練快照：CPU Job，不佔 GPU 排隊，馬上跑。"""
        self._spawn(task_id, self._run_dataset_mix)

    def submit_deployment(self, task_id: str) -> None:
        self._spawn(task_id, self._run_deployment)

    def cancel(self, task_id: str) -> bool:
        return self.store.request_cancel(task_id)

    def start_background(self) -> None:
        """啟動對帳 + 心跳監看。"""
        self.reconcile()
        t = threading.Thread(target=self._monitor, name="orch-monitor", daemon=True)
        t.start()

    def shutdown(self) -> None:
        self._stop.set()

    def health(self) -> dict:
        return {"running_tasks": [k for k, t in self._threads.items() if t.is_alive()],
                "runner_ok": getattr(self.runner, "ping", lambda: True)(),
                "max_concurrent": self.max_concurrent, "video_max_concurrent": self.video_max_concurrent,
                "video_max_with_training": self.video_max_with_training,
                "keep_workspaces": self.keep_workspaces,
                "queued": sum(len(self.store.queued(type=t)) for t in GPU_TYPES)}

    # ── 內部：執行緒管理 ──────────────────────────────────

    def _spawn(self, task_id: str, target) -> None:
        if task_id in self._threads and self._threads[task_id].is_alive():
            return
        t = threading.Thread(target=self._guard, args=(task_id, target), name=f"task-{task_id}", daemon=True)
        self._threads[task_id] = t
        t.start()

    def _guard(self, task_id: str, target) -> None:
        try:
            target(task_id)
        except Exception as exc:
            logger.exception("task %s 執行緒異常", task_id)
            self.store.finish(task_id, TaskStatus.FAILED, message=f"Orchestrator 錯誤：{exc}", error=str(exc)[:2000])
        finally:
            # 這個任務讓出 GPU，換下一個排隊的上場；順手回收舊工作區
            try:
                self.cleanup_workspaces()
            except Exception:
                logger.exception("清理工作區失敗")
            try:
                self._drain_queue()
            except Exception:
                logger.exception("drain queue 失敗")

    # ── 工作區回收 ───────────────────────────────────────

    def cleanup_workspaces(self) -> list[str]:
        """只保留最近 N 個任務的工作區，其餘刪掉（log 留著，佔用極小）。

        安全條件：只刪【已結束】任務的目錄，且該目錄不屬於任何 working/queued 任務。
        產出物早就在 MinIO（模型、報告）與 ClearML（曲線），工作區純屬中間檔。
        """
        if self.keep_workspaces <= 0 or not self.work_dir.exists():
            return []
        active = {t.task_id for t in self.store.working()} | {t.task_id for t in self.store.queued()}
        dirs = [d for d in self.work_dir.iterdir() if d.is_dir() and d.name not in active]
        dirs.sort(key=lambda d: d.stat().st_mtime, reverse=True)
        removed = []
        for d in dirs[self.keep_workspaces:]:
            try:
                logs = d / "logs"
                for child in d.iterdir():
                    if child == logs:
                        continue
                    shutil.rmtree(child, ignore_errors=True) if child.is_dir() else child.unlink(missing_ok=True)
                removed.append(d.name)
            except Exception as exc:
                logger.warning("清理工作區 %s 失敗：%s", d, exc)
        if removed:
            logger.info("已清理 %d 個工作區（保留最近 %d 個 + 全部 log）：%s",
                        len(removed), self.keep_workspaces, ", ".join(removed[:5]))
        return removed

    # ── 內部：步驟共用 ──────────────────────────────────

    def _pass_env(self) -> dict[str, str]:
        return {k: v for k, v in os.environ.items() if k.startswith(PASS_ENV_PREFIXES)}

    def _step_status(self, task_id: str, step: str) -> str | None:
        t = self.store.get(task_id)
        return (((t.result or {}).get("steps") or {}).get(step) or {}).get("status") if t else None

    def _step_done(self, task_id: str, step: str) -> bool:
        return self._step_status(task_id, step) == "done"

    def _steps_resumable(self, steps: dict) -> bool:
        """記成 running 的步驟，容器（K8s Job）還在 —— 不管還在跑還是已經跑完 —— 就接手。
        10/21：訓練 Job 剛好在 control-api 部署重啟時跑完，以前只認「還在跑」→ 判成容器不見、把跑完的訓練標失敗。"""
        return any(v.get("status") == "running" and self.runner.find(v.get("container", ""))
                   for v in steps.values())

    def _mark_step(self, task_id: str, step: str, status: str, **extra) -> None:
        self.store.set_result(task_id, {"steps": {step: {"status": status, "at": time.time(), **extra}}})

    def _run_step(self, task_id: str, spec: StepSpec) -> int:
        """跑一步：起容器 → 等待（可取消）→ 存 log → 記狀態。回傳 exit code。"""
        if self._step_done(task_id, spec.name):
            logger.info("task %s 步驟 %s 已完成，略過", task_id, spec.name)
            return 0
        # 重啟續跑：同名容器還在跑就接著等；記成 running 而容器已經跑完（重啟期間結束）→ 接手讀結果，不重跑
        handle = self.runner.find(spec.container_name) if spec.container_name else None
        if handle and (self.runner.is_running(handle.ref) or self._step_status(task_id, spec.name) == "running"):
            logger.info("task %s 步驟 %s 容器還在（執行中或剛跑完），接手", task_id, spec.name)
        else:
            handle = self.runner.start(spec)
            self.store.start(task_id, runner_ref=handle.ref)
            self._mark_step(task_id, spec.name, "running", container=handle.name)
        self.store.heartbeat(task_id, f"{spec.name}：執行中")
        code = self.runner.wait(handle, cancel_check=lambda: self.store.cancel_requested(task_id))
        log_path = save_logs(self.runner, handle, self.work_dir / task_id / "logs", spec.name)
        # 失敗的容器可選擇保留，方便檢查它拿到的環境與檔案（除錯用；預設仍清掉避免堆積）
        if code == 0 or not self.keep_failed_containers:
            self.runner.remove(handle)
        else:
            logger.info("步驟 %s 失敗，保留容器 %s 供檢查（TRAIN_KEEP_FAILED_CONTAINERS）", spec.name, handle.name)
        self._mark_step(task_id, spec.name, "done" if code == 0 else "failed", exit_code=code, log=str(log_path))
        logger.info("task %s 步驟 %s 結束（exit %d）", task_id, spec.name, code)
        return code

    def _volumes(self) -> dict[str, str]:
        return {self.work_volume: "/work"}

    # ── 訓練任務 ─────────────────────────────────────────

    def _run_training(self, task_id: str) -> None:
        task = self.store.get(task_id)
        if task is None:
            return
        req = TrainingJobRequest.model_validate(task.params["request"])
        env = self._pass_env() | {"TASK_ID": task_id, "TRAIN_WORK_DIR": "/work"}
        wd = f"/work/{task_id}"

        # ① trainer（只有這步要出外網）
        code = self._run_step(task_id, StepSpec(
            name="trainer", image=self.trainer_image,
            command=["python", "-m", "edgevision.training.pipeline", "trainer", "--task-id", task_id],
            env=env | proxy_env(env), gpu=True, volumes=self._volumes(), container_name=f"edgevision-job-{task_id}-trainer",
            shm_size="8g",
        ))
        if code != 0:
            self._fail_if_working(task_id, f"trainer 步驟失敗（exit {code}）", cancelled=(code == 2))
            return

        # ② trtexec：onnx → plan（Triton 自帶 binary）
        t = self.store.get(task_id)
        cfg = ((t.result or {}).get("trainer") or {}).get("config") or {}
        model_name = cfg.get("model_name", req.model_name)
        export = cfg.get("export") or {}
        from libs.contracts.model_meta import ExportSpec
        from edgevision.training.export import trtexec_command
        spec = ExportSpec.model_validate(export) if export else ExportSpec()
        plan_dir = f"{wd}/repo/{model_name}/1"
        code = self._run_step(task_id, StepSpec(
            name="trtexec", image=self.triton_image,
            command=["bash", "-c", f"mkdir -p {plan_dir} && " + " ".join(
                trtexec_command(f"{wd}/model.onnx", f"{plan_dir}/model.plan", spec))],
            env={"NO_PROXY": "*"}, gpu=True, volumes=self._volumes(),
            container_name=f"edgevision-job-{task_id}-trtexec",
        ))
        if code != 0:
            self._fail_if_working(task_id, f"trtexec 轉檔失敗（exit {code}）", cancelled=(code == 2))
            return

        # ③④ 臨時 Triton + finalize
        parity: RunHandle | None = None
        fin_env = dict(env)
        if self.trt_parity:
            parity = self.runner.start(StepSpec(
                name="parity-triton", image=self.triton_image,
                command=["tritonserver", f"--model-repository={wd}/repo", "--model-control-mode=none",
                         "--log-verbose=0", "--allow-grpc=false"],
                env={"NO_PROXY": "*"}, gpu=True, volumes=self._volumes(),
                container_name=f"edgevision-parity-{task_id}", ports=[8000],   # K8s：同名 Service 讓 finalize 連得到
            ))
            fin_env["PARITY_TRITON_URL"] = f"http://{parity.name}:8000"
        try:
            code = self._run_step(task_id, StepSpec(
                name="finalize", image=self.edgevision_image,
                command=["python", "-m", "edgevision.training.pipeline", "finalize", "--task-id", task_id],
                env=fin_env, gpu=False, volumes=self._volumes(),
                container_name=f"edgevision-job-{task_id}-finalize",
            ))
        finally:
            if parity:
                save_logs(self.runner, parity, self.work_dir / task_id / "logs", "parity-triton")
                self.runner.stop(parity); self.runner.remove(parity)
        if code != 0:
            self._fail_if_working(task_id, f"finalize 失敗（exit {code}）", cancelled=(code == 2))
            return

        # ⑤ 自動放行
        if req.auto_promote:
            t = self.store.get(task_id)
            version = int(t.model_version or 0)
            self._promote(task_id, model_name, version, actor=t.created_by, channel=t.channel,
                          reason="auto_promote", parent=task_id)

    def _fail_if_working(self, task_id: str, message: str, *, cancelled: bool = False) -> None:
        t = self.store.get(task_id)
        if t and t.status == TaskStatus.WORKING.value:
            self.store.finish(task_id, TaskStatus.CANCELLED if cancelled else TaskStatus.FAILED,
                              message=message, error=None if cancelled else message)

    # ── 圖片庫：匯入 / 自動標注 ─────────────────────────

    def _run_media_import(self, task_id: str) -> None:
        t = self.store.get(task_id)
        if t is None:
            return
        volumes = {}
        if t.params.get("kind") in NAS_IMPORT_KINDS:
            if not self.media_nas_pvc:
                self.store.finish(task_id, TaskStatus.FAILED, message="這個環境沒有 NAS 錄影（MEDIA_NAS_PVC 未設）",
                                  error="MEDIA_NAS_PVC not set")
                return
            volumes[self.media_nas_pvc] = "/nas"
        code = self._run_step(task_id, StepSpec(
            name="media-import", image=self.edgevision_image,
            command=["python", "-m", "edgevision.media.jobs", "import", "--task-id", task_id],
            env=self._pass_env() | {"TASK_ID": task_id, "MEDIA_NAS_ROOT": "/nas/raw", "PYTHONPATH": "/app"},
            gpu=False, volumes=volumes, container_name=f"edgevision-job-{task_id}-media-import",
        ))
        if code != 0:
            self._fail_if_working(task_id, f"匯入失敗（exit {code}）", cancelled=(code == 2))

    def _run_autolabel(self, task_id: str) -> None:
        code = self._run_step(task_id, StepSpec(
            name="autolabel", image=self.trainer_image,
            command=["python", "-m", "edgevision.media.jobs", "autolabel", "--task-id", task_id],
            env=self._pass_env() | {"TASK_ID": task_id}, gpu=True,
            container_name=f"edgevision-job-{task_id}-autolabel", shm_size="4g",
        ))
        if code != 0:
            self._fail_if_working(task_id, f"自動標注失敗（exit {code}）", cancelled=(code == 2))

    # ── 影片測試（edgevision/media/video_test.py）──────────────────

    def runner_image(self, line_id: str | None = None) -> str | None:
        """runner 映像（ConfigMap edgevision-runner）：這條線有專用的 image.<線路 id> 就用它，否則預設 image。

        影片測試也走這裡：一律用那條線線上正在跑的同一版（判定邏輯在映像裡，CI 換到哪版就測哪版）。
        ⚠️ 2026-11-04 拿掉影片測試自己的 image.video-test：它不跟 CI 走，10/21 釘住後沒人更新，
           新站別 bravo_cleanhand 上線了影片測試卻找不到 —— 兩邊分開維護一定會脫節。
           要先試還沒上線的 runner，請用測試叢集。
        """
        files = ([self.runner_image_file.with_name(f"image.{line_id}")] if line_id else []) + [self.runner_image_file]
        for f in files:
            try:
                if image := f.read_text().strip():
                    return image
            except OSError:
                continue
        return None

    def _run_video_test(self, task_id: str) -> None:
        """prepare → 臨時 Triton（只載這次的模型版本）→ runner 檔案模式 → finalize。工作檔用完就清。"""
        from edgevision.media.video_test import work_dir

        t = self.store.get(task_id)
        if t is None:
            return
        image = self.runner_image((t.params or {}).get("line_id"))
        if not image:
            self.store.finish(task_id, TaskStatus.FAILED, message=f"找不到 runner 映像（{self.runner_image_file}）",
                              error="runner image not configured")
            return
        wd = work_dir(Path("/work"), task_id)
        env = self._pass_env() | {"TASK_ID": task_id, "TRAIN_WORK_DIR": "/work", "PYTHONPATH": "/app",
                                  "MEDIA_NAS_ROOT": "/nas/raw"}
        volumes = self._volumes()
        if self.media_nas_pvc:
            volumes = volumes | {self.media_nas_pvc: "/nas"}      # 圖片庫的影片也可能是 NAS 錄影
        elif not t.params.get("source_id"):
            self.store.finish(task_id, TaskStatus.FAILED, message="這個環境沒有 NAS 錄影（MEDIA_NAS_PVC 未設）",
                              error="MEDIA_NAS_PVC not set")
            return
        triton: RunHandle | None = None
        try:
            code = self._run_step(task_id, StepSpec(
                name="prepare", image=self.edgevision_image,
                command=["python", "-m", "edgevision.media.video_test", "prepare", "--task-id", task_id],
                env=env, gpu=False, volumes=volumes, container_name=f"edgevision-job-{task_id}-prepare"))
            if code != 0:
                self._fail_if_working(task_id, f"準備失敗（exit {code}）", cancelled=(code == 2))
                return
            run = json.loads((self.work_dir / "video-tests" / task_id / "run.json").read_text(encoding="utf-8"))
            # 服務重啟後接手（reconcile）：臨時 Triton 還在就沿用，不再起一個同名的
            found = self.runner.find(f"edgevision-vt-{task_id}")
            triton = found if found and self.runner.is_running(found.ref) else self.runner.start(StepSpec(
                name="vt-triton", image=self.triton_image,
                command=["tritonserver", f"--model-repository={wd}/repo", "--model-control-mode=none",
                         "--log-verbose=0"],
                env={"NO_PROXY": "*"}, gpu=True, volumes=self._volumes(),
                container_name=f"edgevision-vt-{task_id}", ports=[8000, 8001]))
            self.store.heartbeat(task_id, "臨時 Triton 啟動中")
            if not self._wait_triton(f"http://{triton.name}:8000", task_id):
                self._fail_if_working(task_id, "臨時 Triton 沒有就緒（看任務步驟 vt-triton 的 log）")
                return
            pace = {"VIDEO_PACE": "0"} if (t.params or {}).get("mode") == "model" else {}   # 純模型：不限速、不丟幀
            t_runner = time.time()
            prev = self.runner.find(f"edgevision-job-{task_id}-runner")
            resumed = bool(prev and self.runner.is_running(prev.ref))       # 接手的不知道從何時開始跑，不記秒數
            code = self._run_step(task_id, StepSpec(
                name="runner", image=image, command=[],          # 空 = 映像預設指令（必經官方 entrypoint）
                env={"LINE_ID": json.loads((self.work_dir / "video-tests" / task_id / "line.json")
                                           .read_text(encoding="utf-8"))["id"],
                     "CAMERA_SOURCE": "file", "CAMERA_FILE": f"{wd}/line.json",
                     "VIDEO_FILE": f"{wd}/video.mp4", "VIDEO_OUT": f"{wd}/annotated.mp4",
                     "RESULTS_FILE": f"{wd}/results.jsonl", "MODEL_NAME": run["model_name"],
                     "TRITON_HTTP": f"{triton.name}:8000", "TRITON_GRPC": f"{triton.name}:8001",
                     "RESTREAM_ENABLED": "1", "ENCODER": "x264", "STATS_INTERVAL_S": "0",
                     "NVIDIA_DRIVER_CAPABILITIES": "all", "NO_PROXY": "*", "no_proxy": "*", "LOG_LEVEL": "INFO"}
                    | pace,
                gpu=True, volumes=self._volumes(), shm_size="256Mi",
                container_name=f"edgevision-job-{task_id}-runner"))
            if code != 0:
                self._fail_if_working(task_id, f"runner 失敗（exit {code}）", cancelled=(code == 2))
                return
            # runner 花了幾秒（finalize 算倍速）：寫回 run.json
            if not resumed:
                run_path = self.work_dir / "video-tests" / task_id / "run.json"
                run["runner_s"] = round(time.time() - t_runner, 1)
                run_path.write_text(json.dumps(run, ensure_ascii=False), encoding="utf-8")
            code = self._run_step(task_id, StepSpec(
                name="finalize", image=self.edgevision_image,
                command=["python", "-m", "edgevision.media.video_test", "finalize", "--task-id", task_id],
                env=env, gpu=False, volumes=self._volumes(), container_name=f"edgevision-job-{task_id}-finalize"))
            if code != 0:
                self._fail_if_working(task_id, f"收尾失敗（exit {code}）", cancelled=(code == 2))
        finally:
            if triton:
                save_logs(self.runner, triton, self.work_dir / task_id / "logs", "vt-triton")
                self.runner.stop(triton); self.runner.remove(triton)
            shutil.rmtree(self.work_dir / "video-tests" / task_id, ignore_errors=True)   # 影片與 engine 都大，結果已上傳

    def _cleanup_video_test(self, task_id: str) -> None:
        """失敗收掉的影片測試：臨時 Triton 與還在的步驟容器一併清掉（不然一直佔著 GPU 切片）。"""
        for name in (f"edgevision-vt-{task_id}", f"edgevision-job-{task_id}-runner", f"edgevision-job-{task_id}-prepare",
                     f"edgevision-job-{task_id}-finalize"):
            try:
                h = self.runner.find(name)
                if h:
                    self.runner.stop(h)
                    self.runner.remove(h)
                    logger.warning("影片測試 %s 收掉殘留的 %s", task_id, name)
            except Exception:
                logger.exception("清 %s 失敗", name)
        shutil.rmtree(self.work_dir / "video-tests" / task_id, ignore_errors=True)

    def _wait_triton(self, url: str, task_id: str, timeout_s: float = 600) -> bool:
        import urllib.request
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            if self.store.cancel_requested(task_id):
                return False
            try:
                with urllib.request.urlopen(f"{url}/v2/health/ready", timeout=3) as r:
                    if r.status == 200:
                        return True
            except OSError:
                pass
            time.sleep(3)
        return False

    def _run_media_export(self, task_id: str) -> None:
        code = self._run_step(task_id, StepSpec(
            name="media-export", image=self.edgevision_image,
            command=["python", "-m", "edgevision.media.jobs", "export", "--task-id", task_id],
            env=self._pass_env() | {"TASK_ID": task_id, "PYTHONPATH": "/app"},
            gpu=False, container_name=f"edgevision-job-{task_id}-media-export",
        ))
        if code != 0:
            self._fail_if_working(task_id, f"匯出失敗（exit {code}）", cancelled=(code == 2))

    def _run_dataset_mix(self, task_id: str) -> None:
        code = self._run_step(task_id, StepSpec(
            name="dataset-mix", image=self.edgevision_image,
            command=["python", "-m", "edgevision.media.jobs", "mix", "--task-id", task_id],
            env=self._pass_env() | {"TASK_ID": task_id, "PYTHONPATH": "/app"},
            gpu=False, container_name=f"edgevision-job-{task_id}-dataset-mix",
        ))
        if code != 0:
            self._fail_if_working(task_id, f"混合失敗（exit {code}）", cancelled=(code == 2))

    # ── 部署任務（promote / rollback）────────────────────

    def _run_deployment(self, task_id: str) -> None:
        t = self.store.get(task_id)
        if t is None:
            return
        p = t.params
        self._promote(task_id, p["model_name"], int(p["version"]), actor=t.created_by,
                      channel=t.channel, reason=p.get("reason", ""), parent=None, own_task=True)

    def _promote(self, task_id: str, model_name: str, version: int, *, actor: str, channel: str,
                 reason: str, parent: str | None, own_task: bool = False) -> None:
        """Registry 移 tag → 部署 Controller 載入 → 稽核。"""
        dep_task = task_id if own_task else self.store.create(
            type=TaskType.DEPLOYMENT, created_by=actor, channel=channel,
            params={"model_name": model_name, "version": version, "reason": reason},
            parent_task_id=parent, clone_reason=None)
        registry = self._registry_factory()
        deployer = self._deployer_factory()
        try:
            self.store.start(dep_task)
            self.store.heartbeat(dep_task, f"promote：{model_name} → v{version}")
            prev = registry.resolve(model_name, tag=TAG_PRODUCTION)
            registry.promote(model_name, version, tag=TAG_PRODUCTION)
            r = deployer.deploy_from_registry(registry, model_name, version=version)
            if not r.ok:
                # 部署失敗：tag 移回去，避免 Registry 與 Triton 不一致
                if prev:
                    registry.promote(model_name, prev.version, tag=TAG_PRODUCTION)
                raise RuntimeError(f"部署失敗：{r.message}")
            try:
                sites_done = self._deploy_sites(dep_task, registry, model_name, version)
            except Exception as exc:
                # 其他站點換不過去：全部退回上一版（Registry、主 Triton、已換好的站點），不要各站點版本不一
                if prev:
                    registry.promote(model_name, prev.version, tag=TAG_PRODUCTION)
                    deployer.deploy_from_registry(registry, model_name, version=prev.version)
                    try:
                        self._deploy_sites(dep_task, registry, model_name, prev.version)
                    except Exception:
                        logger.exception("站點退回 v%s 失敗", prev.version)
                raise RuntimeError(f"其他推論站點部署失敗，已退回 v{prev.version if prev else '?'}：{exc}") from exc
            self.store.set_result(dep_task, {"deploy": {"ok": True, "version": version,
                                                        "previous": prev.version if prev else None,
                                                        "elapsed_ms": r.elapsed_ms, "sites": sites_done}})
            self.store.finish(dep_task, TaskStatus.COMPLETED, message=f"{model_name} v{version} 已上線",
                              model_version=str(version))
            self._write_audit(actor, channel, "promote", model_name,
                              {"version": version, "previous": prev.version if prev else None, "reason": reason},
                              "success", dep_task)
        except Exception as exc:
            logger.exception("promote 失敗")
            self.store.finish(dep_task, TaskStatus.FAILED, message=f"promote 失敗：{exc}", error=str(exc)[:2000])
            self._write_audit(actor, channel, "promote", model_name, {"version": version, "reason": reason},
                              "failed", dep_task, error=str(exc))
        finally:
            try:
                deployer.close()
            except Exception:
                pass

    def _deploy_sites(self, task_id: str, registry, model_name: str, version: int, *,
                      only: list[str] | None = None, only_if_present: bool = True) -> list[str]:
        """其他推論站點跟上這一版（見 site_deploy.py）。回傳換好的站點名；任一站點失敗拋例外。"""
        if self._site_deployer_factory is None or self._sites_loader is None:
            return []
        from edgevision.orchestrator.site_deploy import deploy_to_site
        sites = self._sites_loader()
        resolved = None
        done: list[str] = []
        for name, site in sites.items():
            if only is not None and name not in only:
                continue
            ctrl = self._site_deployer_factory(site)
            try:
                if only_if_present and not ctrl.repo.versions(model_name):
                    continue                       # 這個站點沒在用這顆模型
                resolved = resolved or registry.get(model_name, version)
                if resolved is None:
                    raise RuntimeError(f"Registry 找不到 {model_name} v{version}")
                self.store.heartbeat(task_id, f"站點 {name}：{model_name} v{version}")
                if deploy_to_site(self, task_id, site, ctrl, resolved, only_if_present=only_if_present):
                    done.append(name)
            finally:
                ctrl.close()
        return done

    def sync_site(self, task_id: str, site_name: str, models: list[str]) -> dict[str, str]:
        """新站點第一次上架（或補模型）：把這些模型的現役版放上那個站點。回 {模型: 結果}。"""
        registry = self._registry_factory()
        out: dict[str, str] = {}
        self.store.start(task_id)
        for m in models:
            prod = registry.resolve(m, tag=TAG_PRODUCTION)
            if prod is None:
                out[m] = "Registry 沒有現役版"
                continue
            try:
                self._deploy_sites(task_id, registry, m, prod.version, only=[site_name], only_if_present=False)
                out[m] = f"v{prod.version} ok"
            except Exception as exc:
                logger.exception("站點 %s 上架 %s 失敗", site_name, m)
                out[m] = f"失敗：{exc}"
        ok = all(v.endswith(" ok") for v in out.values())
        self.store.set_result(task_id, {"sync_site": site_name, "models": out})
        self.store.finish(task_id, TaskStatus.COMPLETED if ok else TaskStatus.FAILED,
                          message=f"站點 {site_name} 上架：{out}")
        return out

    def _write_audit(self, actor, channel, action, target, params, result, task_id, error=None) -> None:
        if self._audit:
            try:
                self._audit(actor=actor, actor_type="agent" if channel == "mcp" else "human", channel=channel,
                            action=action, target=target, params=params, result=result, error=error, task_id=task_id)
            except Exception:
                logger.exception("audit 寫入失敗")

    # ── 對帳與監看 ───────────────────────────────────────

    def reconcile(self) -> None:
        """啟動時：DB 說 working 的任務，容器還在就接手，不在就標 failed；queued 的重新排隊。"""
        for t in self.store.working():
            if t.type == TaskType.TRAINING.value:
                steps = (t.result or {}).get("steps") or {}
                alive = self._steps_resumable(steps)
                if alive or not steps:
                    logger.info("對帳：task %s 續跑（%s）", t.task_id, "容器仍在" if alive else "尚未開始")
                    # 直接接手：它已經是 working（佔著名額），走 submit 會因為 claim 只收 queued 而什麼都不做，
                    # 卡到心跳逾時才被判失敗（10/21 前訓練重啟續跑其實從沒成功過）
                    self._spawn(t.task_id, self._run_training)
                else:
                    self.store.finish(t.task_id, TaskStatus.FAILED,
                                      message="Orchestrator 重啟時發現步驟容器已不存在",
                                      error="reconcile: container missing")
            elif t.type in (TaskType.MEDIA_IMPORT.value, TaskType.AUTOLABEL.value, TaskType.MEDIA_EXPORT.value,
                            TaskType.DATASET_MIX.value):
                steps = (t.result or {}).get("steps") or {}
                alive = self._steps_resumable(steps)
                if alive:
                    logger.info("對帳：task %s 容器仍在，接手等待", t.task_id)
                    self._spawn(t.task_id, {TaskType.MEDIA_IMPORT.value: self._run_media_import,
                                            TaskType.AUTOLABEL.value: self._run_autolabel,
                                            TaskType.MEDIA_EXPORT.value: self._run_media_export,
                                            TaskType.DATASET_MIX.value: self._run_dataset_mix}[t.type])
                else:
                    self.store.finish(t.task_id, TaskStatus.FAILED, message="Orchestrator 重啟時任務容器已不存在",
                                      error="reconcile: container missing")
            elif t.type == TaskType.VIDEO_TEST.value:
                # 每一步都能接手（_run_step 跳過做完的、接回還在跑的容器；臨時 Triton 還在就沿用），
                # 以前漏了這一類：重啟後任務卡在 working、臨時 Triton 一直佔著 GPU（10/16 留下一個掛了 4 天）
                logger.info("對帳：影片測試 %s 接手", t.task_id)
                self._spawn(t.task_id, self._run_video_test)
            elif t.type == TaskType.DEPLOYMENT.value:
                self.store.finish(t.task_id, TaskStatus.FAILED, message="Orchestrator 重啟，任務中斷",
                                  error="reconcile: interrupted")
        self._drain_queue()

    def _monitor(self) -> None:
        while not self._stop.wait(60):
            try:
                self._drain_queue()
                for t in self.store.stale(seconds=self.heartbeat_timeout):
                    steps = (t.result or {}).get("steps") or {}
                    alive = any(self.runner.is_running(self.runner.find(v.get("container", "")).ref)
                                for v in steps.values()
                                if v.get("status") == "running" and self.runner.find(v.get("container", "")))
                    if not alive:
                        self.store.finish(t.task_id, TaskStatus.FAILED,
                                          message=f"心跳逾時（>{self.heartbeat_timeout}s）且容器不在",
                                          error="heartbeat timeout")
                        if t.type == TaskType.VIDEO_TEST.value:
                            self._cleanup_video_test(t.task_id)
            except Exception:
                logger.exception("monitor 例外")
