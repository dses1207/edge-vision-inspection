"""控制 API：訓練與部署的唯一入口。UI、MCP、腳本都是它的客戶端。

    POST /api/training/jobs          建訓練任務 → task_id
    GET  /api/training/jobs          列任務
    GET  /api/tasks/{id}             狀態、進度、心跳、結果
    GET  /api/tasks/{id}/logs        步驟 log
    POST /api/tasks/{id}/cancel      取消
    GET  /api/datasets               LakeFS repo 與最新版本
    POST /api/datasets/ingest        上傳 zip → LakeFS 建版本（multipart）
    POST /api/datasets/mix           幾個快照＋配方 → 一個訓練快照（背景任務，edgevision/training/mix.py）
    GET  /api/configs                可用的訓練設定（configs/train/*.yaml）
    GET  /ui                         測試前端
    GET  /api/models                 Registry 模型清單
    GET  /api/models/{name}/versions 各版本、tag、指標
    POST /api/models/{name}/promote  放行：production tag → 該版 + 部署到 Triton
    POST /api/models/{name}/rollback 回退（同 promote，語意不同，稽核分開記）
    GET  /healthz
    /api/media/*                     圖片庫、標注、自動標注、匯出（edgevision/media/api.py）

誰做的：X-Actor（預設 anonymous）、X-Actor-Type（human|agent）、X-Channel（ui|mcp|api）。
每個會改狀態的請求都寫 audit_log。
"""

from __future__ import annotations

import json
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

import shutil
import tempfile
import threading
import time

from fastapi import FastAPI, File, Form, Header, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from libs.contracts import PromoteRequest, TrainingJobRequest
from libs.contracts.training import TaskStatus, TaskType
from edgevision.training.lakefs_client import repo_name_error

logger = logging.getLogger(__name__)

_state: dict = {}


def _actor(actor: str | None, actor_type: str | None, channel: str | None,
           authorization: str | None = None) -> tuple[str, str, str]:
    """收斂請求的身分。有 Bearer token 就以驗證過的 sub 為準，否則退回 X-Actor。

    ⚠️ X-Actor 是【自我宣告】，誰都能冒名。正式收緊時把 AUTH_ALLOW_HEADER_ACTOR 設成 false，
       沒有 token 就一律拒絕。詳見 edgevision/api/auth.py。
    """
    from edgevision.api import auth as _auth

    try:
        ident, ch = _auth.resolve(authorization, actor, actor_type, channel)
    except ValueError as exc:
        # channel 不合法是 400；身分問題是 401
        raise HTTPException(400 if "X-Channel" in str(exc) else 401, str(exc))
    return ident.actor, ident.actor_type, ch


def build_state() -> dict:
    """組裝相依：DB、Registry、Deployer、Runner、Orchestrator。全部從環境變數。"""
    from edgevision.controller.deployer import TritonController
    from edgevision.orchestrator import DockerRunner, K8sRunner, Orchestrator
    from edgevision.registry.clearml_registry import ClearMLRegistry
    from edgevision.storage.db import Database
    from edgevision.storage.objects import ObjectStore
    from edgevision.training.tracker import TaskStore

    db = Database()
    objects = ObjectStore()
    store = TaskStore(db)

    def registry_factory():
        return ClearMLRegistry(objects=objects)

    def deployer_factory():
        return TritonController(
            triton_url=_norm_url(_require_env("TRITON_URL")),
            model_repo=os.environ.get("TRITON_MODEL_REPO", "/models"), objects=objects,
        )

    def site_deployer_factory(site):
        # 別的節點的 Triton 要跨節點從 MinIO 拉 plan：等就緒放寬到 120 秒
        return TritonController(triton_url=f"http://{site.triton_http}", model_repo=site.triton_repo,
                                objects=objects, ready_timeout=120.0)

    from edgevision.orchestrator.sites import load_sites

    # JOB_RUNNER=k8s：訓練步驟建成 K8s Job（叢集上沒有 docker.sock，DockerRunner 一建就失敗、整個 API 起不來）
    runner = K8sRunner() if os.environ.get("JOB_RUNNER", "docker").lower() == "k8s" else DockerRunner()
    orch = Orchestrator(runner, store, registry_factory=registry_factory,
                        deployer_factory=deployer_factory, audit=db.write_audit,
                        site_deployer_factory=site_deployer_factory, sites_loader=load_sites)
    return {"db": db, "objects": objects, "store": store, "registry_factory": registry_factory,
            "orchestrator": orch, "runner": runner}


def _require_env(name: str) -> str:
    """必填環境變數：缺了就在啟動/使用時明確報錯，不給預設值（避免兩處各自預設而不一致）。"""
    v = os.environ.get(name, "").strip()
    if not v:
        raise RuntimeError(f"未設定環境變數 {name}")
    return v


def _norm_url(u: str) -> str:
    return u if u.startswith("http") else f"http://{u}"


@asynccontextmanager
async def lifespan(app: FastAPI):
    from edgevision.api import auth as _auth
    _auth.load_jwks_at_startup()
    _state.update(build_state())
    _state["orchestrator"].start_background()
    _state["submit_training"] = submit_training_job
    _state["start_dataset_mix"] = start_dataset_mix
    # 標注專案的調和迴圈；CAMPAIGN_RECONCILE=false 可關
    if os.environ.get("CAMPAIGN_RECONCILE", "true").lower() != "false":
        from edgevision.media.campaign_api import engine_of
        engine_of(_state).start_background()
    logger.info("控制 API 就緒")
    yield
    if "media_campaign" in _state:
        _state["media_campaign"].shutdown()
    _state["orchestrator"].shutdown()
    _state["db"].close()


# 規格與文件放在 /api/control/ 底下：K8s 上推論 API 與控制 API 共用一個網域，Ingress 的「其餘路徑」導到推論 API，
# 預設的 /openapi.json、/docs 會被推論 API 蓋住。舊網址（直連 :8095 時）用下面兩個相容路由保留。
app = FastAPI(title="edgevision 控制 API", lifespan=lifespan,
              openapi_url="/api/control/openapi.json", docs_url="/api/control/docs", redoc_url=None)


# CORS：以前控制 API 完全沒設，前端從別的 origin 呼叫模型 / 訓練端點一律被瀏覽器擋（2026-09-30 發現）
from edgevision.api.control_schemas import (
    CancelOut,
    ControlHealthOut,
    DatasetDeletedOut,
    DatasetDiffOut,
    DatasetRepoOut,
    DatasetVersionOut,
    DatasetVersionPageOut,
    DeployOut,
    IngestFromUriIn,
    IngestJobOut,
    MixCreatedOut,
    MixIn,
    UploadUrlOut,
    ModelSummaryOut,
    ModelVersionDetailOut,
    ModelVersionOut,
    RenameModelIn,
    TaskView,
    TrainConfigOut,
    TrainingJobCreatedOut,
)
from edgevision.api.cors import apply_cors  # noqa: E402

apply_cors(app, "控制 API")

# 圖片庫 / 標注 / 自動標注
from edgevision.media.api import build_router as _media_router  # noqa: E402

app.include_router(_media_router(_state, lambda *a: _actor(*a)))


@app.get("/openapi.json", include_in_schema=False)
def _openapi_compat():
    return JSONResponse(app.openapi())


@app.get("/docs", include_in_schema=False)
def _docs_compat():
    return RedirectResponse("/api/control/docs")


# ── 健康 ─────────────────────────────────────────────────

@app.get("/healthz", responses={200: {"model": ControlHealthOut}})
def healthz():
    db_ok = _state["db"].ping()
    minio_ok = _state["objects"].ping()
    orch = _state["orchestrator"].health()
    ok = db_ok and minio_ok and orch["runner_ok"]
    from edgevision.api import auth as _auth
    return JSONResponse({"ok": ok, "db": db_ok, "minio": minio_ok,
                         "auth_enabled": _auth.enabled(),
                         "auth_allow_header_actor": _auth.allow_header_actor(), **orch},
                        status_code=200 if ok else 503)


# ── 訓練任務 ─────────────────────────────────────────────

@app.post("/api/training/jobs", status_code=201, responses={201: {"model": TrainingJobCreatedOut}})
def create_training_job(
    req: TrainingJobRequest,
    x_actor: str | None = Header(default=None), x_actor_type: str | None = Header(default=None),
    x_channel: str | None = Header(default=None), authorization: str | None = Header(default=None),
):
    actor, actor_type, channel = _actor(x_actor, x_actor_type, x_channel, authorization)
    return submit_training_job(req, actor=actor, actor_type=actor_type, channel=channel)


def submit_training_job(req: TrainingJobRequest, *, actor: str, actor_type: str, channel: str) -> dict:
    """送訓練（API 與標注專案共用）：設定先驗證、寫稽核、進排隊。設定無效 → HTTPException 400。"""
    store = _state["store"]; db = _state["db"]

    # clone：從父任務複製請求，再套本次覆寫
    if req.parent_task_id:
        parent = store.get(req.parent_task_id)
        if parent is None:
            raise HTTPException(404, f"parent_task_id {req.parent_task_id} 不存在")
        base = TrainingJobRequest.model_validate(parent.params.get("request", {}))
        merged = base.model_dump()
        patch = req.model_dump(exclude_unset=True)
        from edgevision.training.config import deep_merge
        merged["overrides"] = deep_merge(base.overrides, req.overrides)
        for k, v in patch.items():
            if k not in ("overrides",):
                merged[k] = v
        req = TrainingJobRequest.model_validate(merged)

    # 早期失敗：設定檔要存在、能載入（不碰 GPU）
    try:
        from edgevision.training.config import load_train_config
        cfg = load_train_config(req.config or None, model_name=req.model_name, overrides=req.overrides,
                                dataset=req.dataset_ref, golden=req.golden_ref,
                                gates=req.gates, config_source=req.config_source)
    except Exception as exc:
        db.write_audit(actor=actor, actor_type=actor_type, channel=channel, action="training.create",
                       target=req.model_name, params=req.model_dump(), result="rejected", error=str(exc))
        raise HTTPException(400, f"設定無效：{exc}")

    # 一律先進 queued，由 Orchestrator 依 TRAIN_MAX_CONCURRENT 決定馬上跑還是排隊
    task_id = store.create(type=TaskType.TRAINING, created_by=actor, channel=channel,
                           params={"request": req.model_dump(), "config_resolved": cfg.model_dump(mode="json")},
                           dataset_version=req.dataset_ref, base_model=cfg.base_model,
                           parent_task_id=req.parent_task_id, clone_reason=req.clone_reason,
                           status=TaskStatus.QUEUED, status_message="已受理，等待 GPU")
    db.write_audit(actor=actor, actor_type=actor_type, channel=channel, action="training.create",
                   target=req.model_name, params=req.model_dump(), result="success", task_id=task_id)
    _state["orchestrator"].submit(task_id)
    t = store.get(task_id)
    return {"task_id": task_id, "status": t.status if t else TaskStatus.QUEUED.value,
            "queue_position": store.queue_position(task_id)}


@app.get("/api/training/jobs", responses={200: {"model": list[TaskView]}})
def list_training_jobs(status: str | None = None, limit: int = Query(50, le=500)):
    return [t.model_dump() for t in _state["store"].list(type=TaskType.TRAINING.value, status=status, limit=limit)]


@app.get("/api/tasks", responses={200: {"model": list[TaskView]}})
def list_tasks(type: str | None = None, status: str | None = None, limit: int = Query(50, le=500)):
    return [t.model_dump() for t in _state["store"].list(type=type, status=status, limit=limit)]


@app.get("/api/tasks/{task_id}", responses={200: {"model": TaskView}})
def get_task(task_id: str):
    t = _state["store"].get(task_id)
    if t is None:
        raise HTTPException(404, "task 不存在")
    return t.model_dump()


@app.get("/api/tasks/{task_id}/logs", response_class=PlainTextResponse)
def task_logs(task_id: str, step: str = "trainer", tail: int = Query(200, le=5000)):
    work = Path(os.environ.get("TRAIN_WORK_DIR", "/work")) / task_id / "logs" / f"{step}.log"
    if work.exists():
        lines = work.read_text(errors="replace").splitlines()
        return "\n".join(lines[-tail:])
    # 還在跑：直接從容器拿
    runner = _state["runner"]
    h = runner.find(f"edgevision-job-{task_id}-{step}")
    if h:
        return runner.logs(h, tail=tail)
    raise HTTPException(404, "沒有這個步驟的 log")


@app.post("/api/tasks/{task_id}/cancel", responses={200: {"model": CancelOut}})
def cancel_task(task_id: str, x_actor: str | None = Header(default=None),
                x_actor_type: str | None = Header(default=None), x_channel: str | None = Header(default=None),
                authorization: str | None = Header(default=None)):
    actor, actor_type, channel = _actor(x_actor, x_actor_type, x_channel, authorization)
    ok = _state["orchestrator"].cancel(task_id)
    _state["db"].write_audit(actor=actor, actor_type=actor_type, channel=channel, action="task.cancel",
                             target=task_id, params={}, result="success" if ok else "rejected", task_id=task_id)
    if not ok:
        raise HTTPException(409, "task 不在排隊中或執行中")
    return {"task_id": task_id, "cancel_requested": True}


# ── 資料集（LakeFS）與設定 ──────────────────────────────

@app.get("/api/datasets", responses={200: {"model": list[DatasetRepoOut]}})
def list_datasets():
    """每個 repo 的最新版本 ID（main 分支 head）與 commit 訊息。"""
    import lakefs
    from edgevision.training.lakefs_client import lakefs_client
    client = lakefs_client()
    out = []
    for repo in lakefs.repositories(client=client):
        item = {"repo": repo.id, "versions": []}
        try:
            r = lakefs.repository(repo.id, client=client)
            for c in r.branch("main").log(max_amount=10):
                item["versions"].append({
                    "dataset_ref": f"lakefs://{repo.id}/{c.id}", "id": c.id[:12], "message": c.message,
                    "created": c.creation_date, "metadata": c.metadata or {},
                })
        except Exception as exc:
            item["error"] = str(exc)
        out.append(item)
    return out


_ingest_jobs: dict[str, dict] = {}


def _run_ingest(job_id: str, zip_path: Path, repo: str, message: str, all_val: bool, classes: list[str] | None):
    from edgevision.training.dataset import extract_zip
    from edgevision.training.ingest import publish
    job = _ingest_jobs[job_id]
    tmp = Path(tempfile.mkdtemp(prefix="ingest-"))
    try:
        job["status"] = "extracting"
        root = extract_zip(zip_path, tmp / "x")
        job["status"] = "uploading"
        r = publish(root, repo, message, class_names=classes, split_ratio=1.0 if all_val else 0.2,
                    source=zip_path.name)
        job.update(status="done", **r)
    except Exception as exc:
        logger.exception("ingest 失敗")
        job.update(status="failed", error=str(exc)[:1000])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        zip_path.unlink(missing_ok=True)


@app.post("/api/datasets/ingest", status_code=202, responses={202: {"model": IngestJobOut}})
def ingest_dataset(
    file: UploadFile = File(...), repo: str = Form(...), message: str = Form("上傳"),
    all_val: bool = Form(False), classes: str = Form(""),
    x_actor: str | None = Header(default=None), x_actor_type: str | None = Header(default=None),
    x_channel: str | None = Header(default=None), authorization: str | None = Header(default=None),
):
    """上傳標註工具匯出的 zip → LakeFS 建不可變版本。大檔在背景處理，用 GET /api/datasets/ingest/{id} 查。"""
    actor, actor_type, channel = _actor(x_actor, x_actor_type, x_channel, authorization)
    if err := repo_name_error(repo):
        raise HTTPException(400, err)
    tmp = Path(tempfile.mkdtemp(prefix="upload-")) / (file.filename or "dataset.zip")
    with tmp.open("wb") as fh:
        shutil.copyfileobj(file.file, fh)
    from edgevision.training.tracker import new_task_id
    job_id = new_task_id("ingest")
    _ingest_jobs[job_id] = {"job_id": job_id, "status": "queued", "repo": repo, "message": message}
    _state["db"].write_audit(actor=actor, actor_type=actor_type, channel=channel, action="dataset.ingest",
                             target=repo, params={"file": file.filename, "message": message, "all_val": all_val},
                             result="success", task_id=job_id)
    cls = [c.strip() for c in classes.split(",") if c.strip()] or None
    threading.Thread(target=_run_ingest, args=(job_id, tmp, repo, message, all_val, cls), daemon=True).start()
    return _ingest_jobs[job_id]


@app.get("/api/datasets/ingest/{job_id}", responses={200: {"model": IngestJobOut}})
def ingest_status(job_id: str):
    j = _ingest_jobs.get(job_id)
    if j is None:
        raise HTTPException(404, "ingest job 不存在（注意：工作狀態存在記憶體，"
                                 "control-api 重啟就消失；LakeFS 已 commit 的版本不受影響，"
                                 "可用 GET /api/datasets 確認版本有沒有建成）")
    return j


# 資料集 zip 上傳的暫存區：raw bucket 的 uploads/ 前綴。
# ingest 成功後 zip 會刪掉（LakeFS 版本才是真相，暫存不留兩份）。
_UPLOAD_BUCKET = "raw"
_UPLOAD_PREFIX = "uploads/datasets"


@app.post("/api/datasets/upload-url", responses={200: {"model": UploadUrlOut}})
def dataset_upload_url(filename: str = Query(..., description="zip 檔名，只取 basename"),
                       x_actor: str | None = Header(default=None), x_actor_type: str | None = Header(default=None),
                       x_channel: str | None = Header(default=None), authorization: str | None = Header(default=None)):
    """拿臨時 PUT 網址直傳 MinIO —— 資料集 zip 幾百 MB 起跳，不要過 API 的手。

    上傳完成後用回傳的 source_uri 呼叫 POST /api/datasets/ingest-uri。
    """
    actor, actor_type, channel = _actor(x_actor, x_actor_type, x_channel, authorization)
    base = Path(filename).name or "dataset.zip"
    if not base.endswith(".zip"):
        raise HTTPException(400, "只收 .zip")
    # 檔名加時間戳：兩個人同名上傳不互相覆蓋
    key = f"{_UPLOAD_PREFIX}/{time.strftime('%Y%m%d_%H%M%S')}_{base}"
    url = _state["objects"].presigned_put_url(bucket=_UPLOAD_BUCKET, key=key, expires_minutes=120)
    _state["db"].write_audit(actor=actor, actor_type=actor_type, channel=channel,
                             action="dataset.upload_url", target=f"s3://{_UPLOAD_BUCKET}/{key}",
                             params={"filename": base}, result="success")
    return {"upload_url": url, "source_uri": f"s3://{_UPLOAD_BUCKET}/{key}", "expires_minutes": 120}


@app.post("/api/datasets/ingest-uri", status_code=202, responses={202: {"model": IngestJobOut}})
def ingest_dataset_from_uri(body: IngestFromUriIn,
                            x_actor: str | None = Header(default=None), x_actor_type: str | None = Header(default=None),
                            x_channel: str | None = Header(default=None), authorization: str | None = Header(default=None)):
    """伺服器端從物件儲存讀 zip 來 ingest —— MCP 走這條（zip 塞不進 JSON 參數）。

    流程：POST /api/datasets/upload-url 拿網址 → curl -T 上傳 → 本端點 → 輪詢 ingest/{job_id}。
    """
    actor, actor_type, channel = _actor(x_actor, x_actor_type, x_channel, authorization)
    if err := repo_name_error(body.repo):
        raise HTTPException(400, err)
    from edgevision.storage.objects import ObjectStore
    try:
        bucket, key = ObjectStore.parse_uri(body.source_uri)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    if not key:
        raise HTTPException(400, "source_uri 缺物件路徑")

    from edgevision.training.tracker import new_task_id
    job_id = new_task_id("ingest")
    _ingest_jobs[job_id] = {"job_id": job_id, "status": "queued", "repo": body.repo, "message": body.message}
    _state["db"].write_audit(actor=actor, actor_type=actor_type, channel=channel, action="dataset.ingest",
                             target=body.repo, params={"source_uri": body.source_uri, "message": body.message,
                                                       "all_val": body.all_val}, result="success", task_id=job_id)

    def _download_then_ingest():
        job = _ingest_jobs[job_id]
        tmp = Path(tempfile.mkdtemp(prefix="upload-")) / Path(key).name
        try:
            job["status"] = "downloading"
            _state["objects"].download(bucket=bucket, key=key, dest=tmp)
        except Exception as exc:
            logger.exception("ingest 下載失敗")
            job.update(status="failed", error=f"從 {body.source_uri} 下載失敗：{exc}"[:1000])
            shutil.rmtree(tmp.parent, ignore_errors=True)
            return
        _run_ingest(job_id, tmp, body.repo, body.message, body.all_val, body.classes or None)
        if job.get("status") == "done":
            # ingest 成功才清暫存 zip；失敗留著，修完參數可直接重試不用重傳
            try:
                _state["objects"].remove(bucket=bucket, key=key)
            except Exception:
                logger.warning("清不掉上傳暫存 %s（不影響結果）", body.source_uri)

    threading.Thread(target=_download_then_ingest, daemon=True).start()
    return _ingest_jobs[job_id]


@app.post("/api/datasets/mix", status_code=202, responses={202: {"model": MixCreatedOut}})
def mix_datasets(body: MixIn, x_actor: str | None = Header(default=None),
                 x_actor_type: str | None = Header(default=None), x_channel: str | None = Header(default=None),
                 authorization: str | None = Header(default=None)):
    """幾個 LakeFS 快照依配方混成一個新快照（訓練快照），訓練時 dataset_ref 填它（背景 CPU 任務）。

    每個來源 repeat（倍數）/ choose（張數）/ proportion（比例＋total）三選一，都不給 = 全部取一次。
    只作用在 train；val 全部聯集。配方、各來源實際張數、seed 記在快照 metadata（get_dataset_version 的 metadata.mix）。
    """
    from edgevision.training.mix import validate_recipe
    actor, actor_type, channel = _actor(x_actor, x_actor_type, x_channel, authorization)
    if err := repo_name_error(body.repo):
        raise HTTPException(400, err)
    sources = [s.model_dump(exclude_none=True) for s in body.sources]
    for s in sources:
        if not s["ref"].startswith("lakefs://"):
            raise HTTPException(400, f"來源要是 lakefs:// 快照：{s['ref']}")
    try:
        validate_recipe(sources, body.total)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    params = {"repo": body.repo, "message": body.message, "sources": sources, "total": body.total, "seed": body.seed}
    return {"task_id": start_dataset_mix(params, actor=actor, actor_type=actor_type, channel=channel)}


def start_dataset_mix(params: dict, *, actor: str, actor_type: str, channel: str) -> str:
    """建混合任務（API 與標注專案共用：同一套檢查、任務、稽核）。"""
    from edgevision.training.mix import validate_recipe
    from libs.contracts.training import TaskStatus, TaskType
    validate_recipe(params["sources"], params.get("total"))
    tid = _state["store"].create(type=TaskType.DATASET_MIX, created_by=actor, channel=channel, params=params,
                                 status=TaskStatus.WORKING, status_message="已受理")
    _state["db"].write_audit(actor=actor, actor_type=actor_type, channel=channel, action="dataset.mix",
                             target=params["repo"], params=params, result="success", task_id=tid)
    _state["orchestrator"].submit_dataset_mix(tid)
    return tid


@app.get("/api/datasets/{repo}/versions/{version_id}", responses={200: {"model": DatasetVersionOut}})
def get_dataset_version(repo: str, version_id: str):
    """單一資料集版本的 metadata（n_images、train/val、classes）—— 送訓練前確認用。"""
    import lakefs
    from edgevision.training.lakefs_client import dataset_uri, lakefs_client
    try:
        c = lakefs.repository(repo, client=lakefs_client()).ref(version_id).get_commit()
    except Exception as exc:
        raise HTTPException(404, f"找不到 {repo}@{version_id}：{exc}") from exc
    meta = dict(c.metadata or {})
    for k in ("class_distribution", "mix", "media_batches"):     # JSON 字串存的欄位解回來（mix = 混合配方與來源）
        if k in meta:
            try:
                meta[k] = json.loads(meta[k])
            except (TypeError, ValueError):
                pass  # 舊版本或手動塞的字串：原樣回，不要 500
    return {"dataset_ref": dataset_uri(repo, c.id), "id": c.id[:12], "message": c.message,
            "created": c.creation_date, "metadata": meta}


@app.get("/api/datasets/{repo}/versions", responses={200: {"model": DatasetVersionPageOut}})
def list_dataset_versions(repo: str, limit: int = Query(50, ge=1, le=500),
                          after: str | None = Query(None, description="上一頁回的 next")):
    """這個 repo 的版本清單（main 分支，新到舊，分頁）。GET /api/datasets 只附最近 10 版。"""
    import lakefs
    from edgevision.training.lakefs_client import lakefs_client
    try:
        r = lakefs.repository(repo, client=lakefs_client())
        kw = {"after": after} if after else {}
        commits = list(r.branch("main").log(max_amount=limit + 1, **kw))
    except Exception as exc:
        raise HTTPException(404, f"找不到資料集 {repo}：{exc}") from exc
    page = commits[:limit]
    return {"repo": repo, "next": page[-1].id if len(commits) > limit else None,
            "versions": [{"dataset_ref": f"lakefs://{repo}/{c.id}", "id": c.id[:12], "message": c.message,
                          "created": c.creation_date, "metadata": c.metadata or {}} for c in page]}


@app.get("/api/datasets/{repo}/diff", responses={200: {"model": DatasetDiffOut}})
def diff_dataset_versions(repo: str, from_ref: str = Query(alias="from", description="舊版本 id"),
                          to_ref: str = Query(alias="to", description="新版本 id"),
                          limit: int = Query(200, ge=0, le=5000, description="回幾筆明細")):
    """兩個版本差在哪：新增 / 刪除 / 改變的檔案數（整體、依目錄、只算圖片），加前 limit 筆明細。"""
    import lakefs
    from edgevision.training.lakefs_client import lakefs_client
    try:
        ref = lakefs.repository(repo, client=lakefs_client()).ref(from_ref)
        changes = list(ref.diff(to_ref))
    except Exception as exc:
        raise HTTPException(404, f"比不了 {repo} {from_ref}↔{to_ref}：{exc}") from exc
    from collections import Counter, defaultdict
    summary: Counter = Counter()
    images: Counter = Counter()
    by_dir: dict[str, Counter] = defaultdict(Counter)
    out = []
    for ch in changes:
        t = str(getattr(ch, "type", "changed")).split(".")[-1].lower()
        path = ch.path
        summary[t] += 1
        by_dir["/".join(path.split("/")[:2]) if "/" in path else "."][t] += 1
        if Path(path).suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
            images[t] += 1
        if len(out) < limit:
            out.append({"type": t, "path": path, "size_bytes": getattr(ch, "size_bytes", None)})
    return {"repo": repo, "from_ref": from_ref, "to_ref": to_ref, "summary": dict(summary),
            "by_dir": {k: dict(v) for k, v in sorted(by_dir.items())}, "images": dict(images), "changes": out,
            "truncated": len(changes) > limit}


@app.post("/api/datasets/{repo}/versions/{version_id}/download", status_code=202,
          responses={202: {"model": MixCreatedOut}})
def download_dataset_version(repo: str, version_id: str, x_actor: str | None = Header(default=None),
                             x_actor_type: str | None = Header(default=None), x_channel: str | None = Header(default=None),
                             authorization: str | None = Header(default=None)):
    """把一個資料集版本打包成 zip（原樣：images/、labels/、data.yaml…；背景任務）。
    完成後 GET /api/media/downloads/{task_id} 拿下載網址。"""
    from edgevision.training.lakefs_client import lakefs_client, ref_exists
    actor, actor_type, channel = _actor(x_actor, x_actor_type, x_channel, authorization)
    if not ref_exists(lakefs_client(), repo, version_id):
        raise HTTPException(404, f"找不到資料集版本 {repo}@{version_id}")
    params = {"format": "dataset", "repo": repo, "version_id": version_id, "collection_id": None}
    tid = _state["store"].create(type=TaskType.MEDIA_EXPORT, created_by=actor, channel=channel, params=params,
                                 status=TaskStatus.WORKING, status_message="已受理")
    _state["db"].write_audit(actor=actor, actor_type=actor_type, channel=channel, action="dataset.download",
                             target=f"{repo}@{version_id}", params=params, result="success", task_id=tid)
    _state["orchestrator"].submit_media_export(tid)
    return {"task_id": tid, "queue_position": None}


def dataset_references(db, repo: str) -> list[str]:
    """還有誰指向這個 repo：訓練任務、標注專案（測試集 / 每代訓練快照 / 批次快照）。有就不能刪。"""
    like = f"lakefs://{repo}/%"
    refs = []
    with db.cursor() as cur:
        cur.execute("SELECT task_id FROM tasks WHERE dataset_version LIKE %s OR params->>'dataset_ref' LIKE %s"
                    " OR params->>'golden_ref' LIKE %s LIMIT 20", (like, like, like))
        refs += [f"任務 {r[0]}" for r in cur.fetchall()]
        cur.execute("SELECT id, name FROM media_campaigns WHERE golden_ref LIKE %s OR batch_snapshots::text LIKE %s",
                    (like, f"%lakefs://{repo}/%"))
        refs += [f"標注專案 {r[0]}「{r[1]}」" for r in cur.fetchall()]
        cur.execute("SELECT DISTINCT campaign_id FROM media_campaign_iterations WHERE dataset_ref LIKE %s"
                    " OR golden_ref LIKE %s OR batch_snapshots::text LIKE %s", (like, like, f"%lakefs://{repo}/%"))
        refs += [f"標注專案 {r[0]} 的某一代" for r in cur.fetchall()]
    return refs


@app.delete("/api/datasets/{repo}", responses={200: {"model": DatasetDeletedOut},
                                                409: {"description": "還有任務 / 模型 / 標注專案指向它"}})
def delete_dataset(repo: str, confirm: str = Query(description="repo 名稱（防手滑）"),
                   x_actor: str | None = Header(default=None), x_actor_type: str | None = Header(default=None),
                   x_channel: str | None = Header(default=None), authorization: str | None = Header(default=None)):
    """刪整個資料集 repo。LakeFS 的版本是不可變的，不能只刪其中一版 —— 不要的版本就是不用它。
    有訓練任務（＝模型的來源）或標注專案指向它 → 409：模型的來源紀錄要能追溯。"""
    import lakefs
    from edgevision.training.lakefs_client import lakefs_client
    actor, actor_type, channel = _actor(x_actor, x_actor_type, x_channel, authorization)
    if confirm != repo:
        raise HTTPException(400, f"要刪除請把 confirm 設成 {repo}")
    refs = dataset_references(_state["db"], repo)
    if refs:
        raise HTTPException(409, f"{repo} 還被用著，不能刪：" + "、".join(refs[:10]))
    try:
        lakefs.repository(repo, client=lakefs_client()).delete()
    except Exception as exc:
        raise HTTPException(404, f"刪不了 {repo}：{exc}") from exc
    _state["db"].write_audit(actor=actor, actor_type=actor_type, channel=channel, action="dataset.delete",
                             target=repo, params={}, result="success")
    return {"repo": repo, "deleted": True}


@app.get("/api/configs", responses={200: {"model": list[TrainConfigOut]}})
def list_configs():
    """可用的訓練設定：configs/train/*.yaml 的摘要，給前端下拉選單。"""
    import yaml
    from edgevision.training.config import resolve_path
    out = []
    for p in sorted(resolve_path("configs/train").glob("*.yaml")):
        try:
            d = yaml.safe_load(p.read_text()) or {}
            out.append({"model_name": d.get("model_name", p.stem), "path": f"configs/train/{p.name}",
                        "task": d.get("task"), "base_model": d.get("base_model"),
                        "class_names": d.get("class_names"), "epochs": (d.get("train") or {}).get("epochs"),
                        "dataset": d.get("dataset"), "golden": d.get("golden")})
        except Exception as exc:
            out.append({"model_name": p.stem, "path": str(p), "error": str(exc)})
    return out


# ── 測試前端 ─────────────────────────────────────────────
_UI_DIR = Path(__file__).resolve().parents[2] / "web" / "train"
if _UI_DIR.exists():
    app.mount("/ui", StaticFiles(directory=str(_UI_DIR), html=True), name="ui")

# 標注審核前端
_REVIEW_DIR = Path(__file__).resolve().parents[2] / "web" / "review"
if _REVIEW_DIR.exists():
    app.mount("/review", StaticFiles(directory=str(_REVIEW_DIR), html=True), name="review")


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse("/ui/")


# ── 模型與部署 ───────────────────────────────────────────

@app.get("/api/models", responses={200: {"model": list[ModelSummaryOut]}})
def list_models():
    reg = _state["registry_factory"]()
    out = []
    for name in reg.list_models():
        prod = reg.resolve(name, tag="production")
        out.append({"model_name": name, "production": prod.version if prod else None,
                    "versions": [v.version for v in reg.list_versions(name)]})
    return out


@app.get("/api/models/{name}/versions", responses={200: {"model": list[ModelVersionOut]}})
def model_versions(name: str):
    reg = _state["registry_factory"]()
    vs = reg.list_versions(name)
    if not vs:
        raise HTTPException(404, f"模型 {name} 不存在")
    return [{"version": v.version, "tags": v.tags, "created_at": v.created_at, "metrics": v.metrics,
             "registry_id": v.registry_id} for v in vs]


@app.get("/api/models/{name}/versions/{version}", responses={200: {"model": ModelVersionDetailOut}})
def model_version(name: str, version: int):
    from edgevision.registry.clearml_registry import describe
    r = _state["registry_factory"]().get(name, version)
    if r is None:
        raise HTTPException(404, "版本不存在")
    return describe(r)


@app.post("/api/models/{name}/versions/{version}/rename", responses={200: {"model": ModelVersionDetailOut}})
def rename_model_version(name: str, version: int, body: RenameModelIn, x_actor: str | None = Header(default=None),
                         x_actor_type: str | None = Header(default=None), x_channel: str | None = Header(default=None),
                         authorization: str | None = Header(default=None)):
    """Registry 沒有改名：複製成新名字的一版，舊的封存。現役版本不准改。"""
    from edgevision.registry.clearml_registry import describe
    actor, actor_type, channel = _actor(x_actor, x_actor_type, x_channel, authorization)
    params = {"version": version, "new_name": body.new_name, "new_version": body.new_version,
              "reason": body.reason}
    try:
        r = _state["registry_factory"]().rename(name, version, body.new_name, body.new_version, note=body.reason)
    except KeyError as exc:
        raise HTTPException(404, str(exc).strip("'")) from exc
    except ValueError as exc:
        _state["db"].write_audit(actor=actor, actor_type=actor_type, channel=channel, action="model.rename",
                                 target=name, params=params, result="rejected", error=str(exc))
        raise HTTPException(409, str(exc)) from exc
    _state["db"].write_audit(actor=actor, actor_type=actor_type, channel=channel, action="model.rename",
                             target=name, params={**params, "new_version": r.version}, result="success")
    return describe(r)


def _deploy(name: str, body: PromoteRequest, action: str, headers) -> dict:
    actor, actor_type, channel = headers
    store = _state["store"]; db = _state["db"]
    reg = _state["registry_factory"]()
    target = reg.get(name, body.version)
    if target is None:
        raise HTTPException(404, f"{name} v{body.version} 不存在")
    task_id = store.create(type=TaskType.DEPLOYMENT, created_by=actor, channel=channel,
                           params={"model_name": name, "version": body.version, "reason": body.reason,
                                   "action": action})
    db.write_audit(actor=actor, actor_type=actor_type, channel=channel, action=f"{action}.request",
                   target=name, params={"version": body.version, "reason": body.reason}, result="success",
                   task_id=task_id)
    _state["orchestrator"].submit_deployment(task_id)
    return {"task_id": task_id, "model_name": name, "version": body.version, "status": "working"}


@app.post("/api/models/{name}/promote", responses={200: {"model": DeployOut}})
def promote(name: str, body: PromoteRequest, x_actor: str | None = Header(default=None),
            x_actor_type: str | None = Header(default=None), x_channel: str | None = Header(default=None),
            authorization: str | None = Header(default=None)):
    return _deploy(name, body, "promote", _actor(x_actor, x_actor_type, x_channel, authorization))


@app.post("/api/models/{name}/rollback", responses={200: {"model": DeployOut}})
def rollback(name: str, body: PromoteRequest, x_actor: str | None = Header(default=None),
             x_actor_type: str | None = Header(default=None), x_channel: str | None = Header(default=None),
             authorization: str | None = Header(default=None)):
    return _deploy(name, body, "rollback", _actor(x_actor, x_actor_type, x_channel, authorization))
