"""Orchestrator 步驟序列、失敗傳播、冪等續跑、佇列（假 runner + 記憶體 task store，不需 docker / DB）。"""
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from libs.contracts.training import TaskStatus, TaskType, TaskView  # noqa: E402
from edgevision.orchestrator.orchestrator import Orchestrator  # noqa: E402
from edgevision.orchestrator.runner import RunHandle, StepSpec  # noqa: E402


class MemStore:
    """TaskStore 的記憶體版：只實作 Orchestrator 用到的方法。"""

    def __init__(self):
        self.tasks: dict[str, dict] = {}

    def create(self, *, type, created_by, channel, params, status=None, status_message=None, **kw):
        tid = f"t{len(self.tasks) + 1}"
        self.tasks[tid] = dict(task_id=tid, type=type.value,
                               status=(status.value if status is not None else "working"),
                               status_message=status_message,
                               created_at="c", created_by=created_by, channel=channel, params=params,
                               result={}, model_version=None, error=None,
                               parent_task_id=kw.get("parent_task_id"), clone_reason=None)
        return tid

    def get(self, tid):
        t = self.tasks.get(tid)
        return TaskView(**t) if t else None

    def start(self, tid, runner_ref=None): pass
    def heartbeat(self, tid, msg=None):
        if msg: self.tasks[tid]["status_message"] = msg

    def set_result(self, tid, patch):
        r = self.tasks[tid]["result"]
        for k, v in patch.items():
            if isinstance(v, dict) and isinstance(r.get(k), dict):
                for k2, v2 in v.items():
                    if isinstance(v2, dict) and isinstance(r[k].get(k2), dict):
                        r[k][k2].update(v2)
                    else:
                        r[k][k2] = v2
            else:
                r[k] = v

    def finish(self, tid, status, message=None, error=None, model_version=None, run_report_uri=None):
        t = self.tasks[tid]; t["status"] = status.value; t["status_message"] = message; t["error"] = error
        if model_version: t["model_version"] = model_version

    def request_cancel(self, tid):
        self.tasks[tid]["result"]["cancel_requested"] = True; return True

    def cancel_requested(self, tid):
        return bool(self.tasks[tid]["result"].get("cancel_requested"))

    def working(self):
        return [self.get(k) for k, v in self.tasks.items() if v["status"] == "working"]

    def stale(self, seconds): return []

    # ── 佇列 ──
    def running_count(self, type=None, types=None):
        ok = (lambda t: t in types) if types else (lambda t: type is None or t == type)
        return sum(1 for v in self.tasks.values() if v["status"] == "working" and ok(v["type"]))

    def queued(self, type=None):
        return [self.get(k) for k, v in self.tasks.items()
                if v["status"] == "queued" and (type is None or v["type"] == type)]

    def queue_position(self, task_id):
        t = self.tasks.get(task_id)
        if not t or t["status"] != "queued":
            return None
        ids = [k for k, v in self.tasks.items() if v["status"] == "queued"]
        return ids.index(task_id) + 1

    def claim(self, task_id):
        t = self.tasks.get(task_id)
        if not t or t["status"] != "queued":
            return False
        t["status"] = "working"; t["status_message"] = "開始執行"
        return True

    def claim_next_queued(self, type=None, types=None):
        ok = (lambda t: t in types) if types else (lambda t: type is None or t == type)
        for k, v in self.tasks.items():
            if v["status"] == "queued" and ok(v["type"]):
                v["status"] = "working"; v["status_message"] = "排隊結束，開始執行"
                return self.get(k)
        return None


class FakeRunner:
    """記錄步驟順序；exit_codes 決定每步結果；模擬 trainer 寫入 result。"""

    def __init__(self, store: MemStore, exit_codes: dict[str, int] | None = None):
        self.store = store; self.codes = exit_codes or {}
        self.started: list[str] = []; self.running: dict[str, StepSpec] = {}; self.removed = []
        self.specs: dict[str, StepSpec] = {}

    def start(self, spec):
        self.started.append(spec.name); self.specs[spec.name] = spec
        h = RunHandle(ref=f"id-{spec.name}", name=spec.container_name or spec.name)
        self.running[h.ref] = spec
        return h

    def wait(self, handle, cancel_check=None, poll_s=5.0):
        spec = self.running[handle.ref]
        tid = spec.env.get("TASK_ID") or handle.name.split("-")[2]
        if spec.name == "trainer" and self.codes.get("trainer", 0) == 0:
            self.store.set_result(tid, {"trainer": {"version": 7, "config": {"model_name": "seg_alpha", "export": {}}}})
            self.store.tasks[tid]["model_version"] = "7"
        if spec.name == "finalize" and self.codes.get("finalize", 0) == 0:
            self.store.finish(tid, TaskStatus.COMPLETED, message="done", model_version="7")
        code = self.codes.get(spec.name, 0)
        if code == 3:
            self.store.finish(tid, TaskStatus.FAILED, message="gate fail", error="gate")
        return code

    def logs(self, handle, tail=None): return "log"
    def stop(self, handle): pass
    def remove(self, handle): self.removed.append(handle.name)
    def is_running(self, ref): return False
    def find(self, name): return None
    def ping(self): return True


def _orch(store, runner, tmp_path, registry=None, deployer=None, audit=None):
    o = Orchestrator(runner, store, registry_factory=lambda: registry, deployer_factory=lambda: deployer, audit=audit)
    o.work_dir = tmp_path
    return o


def _task(store, auto_promote=False):
    return store.create(type=TaskType.TRAINING, created_by="u", channel="api",
                        params={"request": {"model_name": "seg_alpha", "dataset_ref": "lakefs://alpha/a",
                                            "auto_promote": auto_promote}})


def test_full_sequence(tmp_path):
    store = MemStore(); runner = FakeRunner(store)
    o = _orch(store, runner, tmp_path)
    tid = _task(store)
    o._run_training(tid)
    assert runner.started == ["trainer", "trtexec", "parity-triton", "finalize"]
    assert store.tasks[tid]["status"] == "completed"
    steps = store.tasks[tid]["result"]["steps"]
    assert steps["trainer"]["status"] == "done" and steps["finalize"]["status"] == "done"
    assert "edgevision-parity-" + tid in runner.removed   # 臨時 Triton 一定被清掉


def test_gate_failure_stops_pipeline(tmp_path):
    store = MemStore(); runner = FakeRunner(store, {"trainer": 3})
    o = _orch(store, runner, tmp_path)
    tid = _task(store)
    o._run_training(tid)
    assert runner.started == ["trainer"]
    assert store.tasks[tid]["status"] == "failed"


def test_trtexec_failure_marks_failed(tmp_path):
    store = MemStore(); runner = FakeRunner(store, {"trtexec": 1})
    o = _orch(store, runner, tmp_path)
    tid = _task(store)
    o._run_training(tid)
    assert runner.started == ["trainer", "trtexec"]
    assert store.tasks[tid]["status"] == "failed" and "trtexec" in store.tasks[tid]["status_message"]


def test_resume_skips_done_steps(tmp_path):
    store = MemStore(); runner = FakeRunner(store)
    o = _orch(store, runner, tmp_path)
    tid = _task(store)
    store.set_result(tid, {"steps": {"trainer": {"status": "done"}, "trtexec": {"status": "done"}},
                          "trainer": {"version": 7, "config": {"model_name": "seg_alpha", "export": {}}}})
    o._run_training(tid)
    assert runner.started == ["parity-triton", "finalize"]


def test_auto_promote_calls_registry_and_deployer(tmp_path):
    class Reg:
        def __init__(self): self.promoted = []
        def resolve(self, name, tag="production"): return None
        def promote(self, name, v, tag="production"): self.promoted.append((name, v, tag))
    class Dep:
        def __init__(self): self.calls = []
        def deploy_from_registry(self, registry, name, version=None, tag="production"):
            self.calls.append((name, version))
            from edgevision.controller.deployer import DeployResult
            return DeployResult(True, name, version, "ok", 12.0)
        def close(self): pass
    audits = []
    store = MemStore(); runner = FakeRunner(store); reg = Reg(); dep = Dep()
    o = _orch(store, runner, tmp_path, reg, dep, audit=lambda **kw: audits.append(kw))
    tid = _task(store, auto_promote=True)
    o._run_training(tid)
    assert reg.promoted == [("seg_alpha", 7, "production")] and dep.calls == [("seg_alpha", 7)]
    dep_tasks = [t for t in store.tasks.values() if t["type"] == "deployment"]
    assert dep_tasks and dep_tasks[0]["status"] == "completed" and dep_tasks[0]["parent_task_id"] == tid
    assert audits and audits[0]["action"] == "promote" and audits[0]["result"] == "success"


def test_deploy_failure_restores_previous_tag(tmp_path):
    class Prev: version = 3
    class Reg:
        def __init__(self): self.promoted = []
        def resolve(self, name, tag="production"): return Prev()
        def promote(self, name, v, tag="production"): self.promoted.append(v)
    class Dep:
        def deploy_from_registry(self, registry, name, version=None, tag="production"):
            from edgevision.controller.deployer import DeployResult
            return DeployResult(False, name, version, "load 失敗", 1.0)
        def close(self): pass
    store = MemStore(); runner = FakeRunner(store); reg = Reg()
    o = _orch(store, runner, tmp_path, reg, Dep())
    tid = store.create(type=TaskType.DEPLOYMENT, created_by="u", channel="ui",
                       params={"model_name": "seg_alpha", "version": 4, "reason": "r"})
    o._run_deployment(tid)
    assert reg.promoted == [4, 3]              # 先移到 4，失敗後移回 3
    assert store.tasks[tid]["status"] == "failed"


# ═══ 佇列（併發控制）═══════════════════════════════════════

def _queued_task(store, n=1):
    from libs.contracts.training import TaskStatus
    ids = []
    for _ in range(n):
        ids.append(store.create(type=TaskType.TRAINING, created_by="u", channel="api",
                                params={"request": {"model_name": "seg_alpha", "dataset_ref": "lakefs://alpha/a"}},
                                status=TaskStatus.QUEUED, status_message="已受理，等待 GPU"))
    return ids


class BlockingRunner(FakeRunner):
    """wait() 卡住直到 release()，用來製造「GPU 佔用中」的狀態。"""

    def __init__(self, store):
        super().__init__(store)
        self.gate = threading.Event()

    def wait(self, handle, cancel_check=None, poll_s=5.0):
        self.gate.wait(timeout=10)
        return super().wait(handle, cancel_check, poll_s)


def test_first_runs_second_queues(tmp_path):
    store = MemStore(); runner = BlockingRunner(store)
    o = _orch(store, runner, tmp_path)
    a, b = _queued_task(store, 2)
    o.submit(a); time.sleep(0.2)
    o.submit(b)
    assert store.tasks[a]["status"] == "working"
    assert store.tasks[b]["status"] == "queued"          # 只有一個空位，第二個排隊
    assert "排隊中" in store.tasks[b]["status_message"]
    assert o.health()["queued"] == 1
    runner.gate.set()
    for _ in range(60):                                   # 等第一個跑完，第二個自動接上
        if store.tasks[b]["status"] != "queued":
            break
        time.sleep(0.1)
    assert store.tasks[b]["status"] in ("working", "completed")


def _wait_status(store, tid, want, timeout=5.0):
    end = time.time() + timeout
    while time.time() < end:
        if store.tasks[tid]["status"] in want:
            return store.tasks[tid]["status"]
        time.sleep(0.05)
    return store.tasks[tid]["status"]


def test_slot_free_runs_immediately(tmp_path):
    store = MemStore(); runner = FakeRunner(store)
    o = _orch(store, runner, tmp_path)
    (a,) = _queued_task(store, 1)
    o.submit(a)                                           # 非同步：背景執行緒跑
    assert _wait_status(store, a, {"completed"}) == "completed"
    assert runner.started[0] == "trainer"


def test_max_concurrent_two(tmp_path):
    store = MemStore(); runner = BlockingRunner(store)
    o = _orch(store, runner, tmp_path); o.max_concurrent = 2
    a, b, c = _queued_task(store, 3)
    for t in (a, b, c):
        o.submit(t); time.sleep(0.1)
    assert store.tasks[a]["status"] == "working"
    assert store.tasks[b]["status"] == "working"
    assert store.tasks[c]["status"] == "queued"
    runner.gate.set()


def test_reconcile_drains_queue(tmp_path):
    """重啟後 queued 的任務要能重新排上。"""
    store = MemStore(); runner = FakeRunner(store)
    o = _orch(store, runner, tmp_path)
    (a,) = _queued_task(store, 1)
    o.reconcile()
    assert _wait_status(store, a, {"completed"}) == "completed"


def test_reconcile_adopts_step_that_finished_during_restart(tmp_path, monkeypatch):
    """訓練 Job 剛好在 control-api 重啟時跑完（10/21 正式）：容器還在、已結束 → 接手讀結果，不判失敗、不重跑。"""
    store = MemStore(); runner = FakeRunner(store)
    o = _orch(store, runner, tmp_path)
    tid = _task(store)
    name = f"edgevision-job-{tid}-trainer"
    store.set_result(tid, {"steps": {"trainer": {"status": "running", "container": name}}})
    adopted = RunHandle(ref="id-trainer", name=name)
    runner.running[adopted.ref] = StepSpec(name="trainer", image="x", command=[], env={"TASK_ID": tid})
    monkeypatch.setattr(runner, "find", lambda n: adopted if n == name else None)   # 在，但 is_running = False
    spawned = []
    monkeypatch.setattr(o, "_spawn", lambda t, target: spawned.append(t))
    o.reconcile()
    assert spawned == [tid] and store.tasks[tid]["status"] == "working"            # 以前：failed（container missing）
    o._run_training(tid)
    assert "trainer" not in runner.started                                         # 沒重跑 trainer
    assert store.tasks[tid]["result"]["steps"]["trainer"]["status"] == "done"


def test_reconcile_still_fails_when_container_gone(tmp_path, monkeypatch):
    store = MemStore(); runner = FakeRunner(store)
    o = _orch(store, runner, tmp_path)
    tid = _task(store)
    store.set_result(tid, {"steps": {"trainer": {"status": "running", "container": f"edgevision-job-{tid}-trainer"}}})
    o.reconcile()
    assert store.tasks[tid]["status"] == "failed"


# ═══ 工作區回收 ═══════════════════════════════════════════

def _mk_ws(root, name, *, files=("dataset/a.jpg", "model.onnx", "logs/trainer.log")):
    for f in files:
        p = root / name / f
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("x" * 100)
    return root / name


def test_cleanup_keeps_recent_and_logs(tmp_path):
    import os as _os
    store = MemStore(); o = _orch(store, FakeRunner(store), tmp_path)
    o.keep_workspaces = 2
    for i, name in enumerate(["old1", "old2", "new1", "new2"]):
        d = _mk_ws(tmp_path, name)
        _os.utime(d, (1000 + i * 100, 1000 + i * 100))
    removed = o.cleanup_workspaces()
    assert set(removed) == {"old1", "old2"}
    assert not (tmp_path / "old1" / "model.onnx").exists()
    assert (tmp_path / "old1" / "logs" / "trainer.log").exists()   # log 一定留著
    assert (tmp_path / "new2" / "model.onnx").exists()             # 最近的不動


def test_cleanup_skips_active_tasks(tmp_path):
    store = MemStore(); o = _orch(store, FakeRunner(store), tmp_path)
    o.keep_workspaces = 0                      # 0 = 停用
    _mk_ws(tmp_path, "any")
    assert o.cleanup_workspaces() == []
    o.keep_workspaces = 1
    tid = _task(store)                          # working 中的任務
    _mk_ws(tmp_path, tid); _mk_ws(tmp_path, "done1"); _mk_ws(tmp_path, "done2")
    removed = o.cleanup_workspaces()
    assert tid not in removed                   # 執行中的工作區絕不刪


def test_keep_failed_containers_flag(tmp_path):
    """除錯開關：失敗的容器可保留供檢查，成功的一律清掉。"""
    store = MemStore(); runner = FakeRunner(store, {"trainer": 1})
    o = _orch(store, runner, tmp_path)
    o.keep_failed_containers = True
    tid = _task(store)
    o._run_training(tid)
    assert runner.removed == []                       # 失敗的留著

    store2 = MemStore(); runner2 = FakeRunner(store2)
    o2 = _orch(store2, runner2, tmp_path); o2.keep_failed_containers = True
    o2._run_training(_task(store2))
    assert len(runner2.removed) >= 3                  # 成功的照樣清掉


def test_autolabel_shares_gpu_queue_with_training(tmp_path):
    """自動標注跟訓練共用訓練切片：訓練在跑時，自動標注要排隊，跑完再接上（而且跑的是自動標注步驟）。"""
    from libs.contracts.training import TaskStatus
    store = MemStore(); runner = BlockingRunner(store)
    o = _orch(store, runner, tmp_path)
    (a,) = _queued_task(store, 1)
    b = store.create(type=TaskType.AUTOLABEL, created_by="u", channel="api",
                     params={"collection_id": 1, "engine": "registry", "model_name": "seg_alpha"},
                     status=TaskStatus.QUEUED, status_message="已受理")
    o.submit(a); time.sleep(0.2)
    o.submit(b)
    assert store.tasks[b]["status"] == "queued"
    runner.gate.set()
    for _ in range(60):
        if store.tasks[b]["status"] != "queued":
            break
        time.sleep(0.1)
    assert store.tasks[b]["status"] != "queued"
    for _ in range(50):
        if "autolabel" in runner.started:
            break
        time.sleep(0.1)
    assert "autolabel" in runner.started
    spec = runner.specs["autolabel"]
    assert spec.gpu and spec.command[-3:] == ["autolabel", "--task-id", b]


def test_media_export_runs_as_cpu_job_without_gpu_slot(tmp_path):
    """圖片集匯出是 CPU Job：GPU 被訓練佔滿也馬上跑，不排隊；跑的是 edgevision.media.jobs export。"""
    store = MemStore(); runner = BlockingRunner(store)
    o = _orch(store, runner, tmp_path)
    (a,) = _queued_task(store, 1)
    o.submit(a); time.sleep(0.2)                                   # 訓練佔住唯一的 GPU 空位
    e = store.create(type=TaskType.MEDIA_EXPORT, created_by="u", channel="api",
                     params={"collection_id": 1, "repo": "r", "message": "m", "val_pct": 20})
    o.submit_media_export(e)
    for _ in range(50):
        if "media-export" in runner.started:
            break
        time.sleep(0.05)
    spec = runner.specs["media-export"]
    assert spec.command == ["python", "-m", "edgevision.media.jobs", "export", "--task-id", e]
    assert spec.gpu is False and spec.env["TASK_ID"] == e
    runner.gate.set()


def test_reconcile_media_export_container_gone_fails(tmp_path):
    """control-api / orchestrator 重啟：匯出容器已不在 → failed（重送即可）；在的話接手等（同匯入）。"""
    store = MemStore(); runner = FakeRunner(store)
    o = _orch(store, runner, tmp_path)
    e = store.create(type=TaskType.MEDIA_EXPORT, created_by="u", channel="api",
                     params={"collection_id": 1, "repo": "r", "message": "m", "val_pct": 20})
    store.tasks[e]["result"] = {"steps": {"media-export": {"status": "running", "container": "c"}}}
    o.reconcile()
    assert store.tasks[e]["status"] == "failed"
    assert "容器已不存在" in store.tasks[e]["status_message"]


def test_dataset_mix_runs_as_cpu_job(tmp_path):
    """混合是 CPU Job：不排 GPU 隊，跑 edgevision.media.jobs mix；重啟時容器不在 → failed。"""
    store = MemStore(); runner = BlockingRunner(store)
    o = _orch(store, runner, tmp_path)
    (a,) = _queued_task(store, 1)
    o.submit(a); time.sleep(0.2)
    m = store.create(type=TaskType.DATASET_MIX, created_by="u", channel="api",
                     params={"repo": "r", "message": "m", "sources": [{"ref": "lakefs://x/1"}], "seed": 0})
    o.submit_dataset_mix(m)
    for _ in range(50):
        if "dataset-mix" in runner.started:
            break
        time.sleep(0.05)
    spec = runner.specs["dataset-mix"]
    assert spec.command == ["python", "-m", "edgevision.media.jobs", "mix", "--task-id", m] and spec.gpu is False
    runner.gate.set()

    store2 = MemStore(); o2 = _orch(store2, FakeRunner(store2), tmp_path)
    m2 = store2.create(type=TaskType.DATASET_MIX, created_by="u", channel="api", params={})
    store2.tasks[m2]["result"] = {"steps": {"dataset-mix": {"status": "running", "container": "c"}}}
    o2.reconcile()
    assert store2.tasks[m2]["status"] == "failed"


def test_trainer_gets_proxy_and_service_hosts_in_no_proxy(tmp_path, monkeypatch):
    monkeypatch.setenv("TRAIN_HTTPS_PROXY", "http://192.0.2.118:15629")
    monkeypatch.delenv("TRAIN_NO_PROXY", raising=False)
    monkeypatch.setenv("MINIO_ENDPOINT", "minio:9000")
    monkeypatch.setenv("LAKEFS_ENDPOINT", "http://lakefs:8000")
    monkeypatch.setenv("CLEARML_API_HOST", "http://clearml-apiserver:8008")
    monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@postgres:5432/edgevision")
    store = MemStore(); runner = FakeRunner(store)
    o = _orch(store, runner, tmp_path)
    o._run_training(_task(store))
    env = runner.specs["trainer"].env
    assert env["HTTPS_PROXY"] == env["https_proxy"] == "http://192.0.2.118:15629"
    no = env["no_proxy"].split(",")
    assert env["NO_PROXY"] == env["no_proxy"] and "*" not in no
    assert {"minio", "lakefs", "clearml-apiserver", "postgres", "192.0.2.31/16", ".svc"} <= set(no)
    assert runner.specs["finalize"].env.get("HTTPS_PROXY") is None          # 只有 trainer 出外網


def test_trainer_without_proxy_setting_keeps_no_proxy_star(tmp_path, monkeypatch):
    monkeypatch.delenv("TRAIN_HTTPS_PROXY", raising=False)
    monkeypatch.setenv("NO_PROXY", "*")
    store = MemStore(); runner = FakeRunner(store)
    o = _orch(store, runner, tmp_path)
    o._run_training(_task(store))
    env = runner.specs["trainer"].env
    assert "HTTPS_PROXY" not in env and env["NO_PROXY"] == "*"
