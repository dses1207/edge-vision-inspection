"""tasks 表存取：訓練 / 評估 / 轉檔 / 部署任務的狀態、心跳、對帳。

⚠️ 狀態【必須】持久化（schema 註解已說明）：Orchestrator 重啟後，
   容器其實還在跑，沒有 DB 就變殭屍任務。

三個角色都用這個模組：
    控制 API   建 task、查 task
    Orchestrator 更新狀態、對帳
    trainer 容器回報進度與心跳（每 epoch）
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from datetime import datetime, timezone
from typing import Any

from libs.contracts.training import TaskStatus, TaskType, TaskView

from edgevision.storage.db import Database

logger = logging.getLogger(__name__)


def _deep_merge(base: dict, patch: dict) -> dict:
    out = dict(base)
    for k, v in patch.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def new_task_id(prefix: str = "t") -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"{prefix}-{ts}-{uuid.uuid4().hex[:6]}"


class TaskStore:
    def __init__(self, db: Database | None = None) -> None:
        self.db = db or Database()

    # ── 建立 ─────────────────────────────────────────────

    def create(
        self, *, type: TaskType, created_by: str, channel: str, params: dict,
        dataset_version: str | None = None, base_model: str | None = None,
        parent_task_id: str | None = None, clone_reason: str | None = None,
        task_id: str | None = None, status: TaskStatus = TaskStatus.WORKING,
        status_message: str | None = None,
    ) -> str:
        task_id = task_id or new_task_id(type.value[:5])
        with self.db.cursor() as cur:
            cur.execute(
                """
                INSERT INTO tasks (task_id, type, status, status_message, created_by, owner, channel,
                                   dataset_version, base_model, params,
                                   parent_task_id, clone_reason, last_heartbeat)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,now())
                """,
                (task_id, type.value, status.value, status_message, created_by, created_by,
                 channel, dataset_version, base_model, json.dumps(params),
                 parent_task_id, clone_reason),
            )
        logger.info("task %s 建立（%s，by %s via %s）", task_id, type.value, created_by, channel)
        return task_id

    # ── 更新 ─────────────────────────────────────────────

    def start(self, task_id: str, *, runner_ref: str | None = None) -> None:
        with self.db.cursor() as cur:
            cur.execute(
                "UPDATE tasks SET started_at = COALESCE(started_at, now()), last_heartbeat = now(),"
                " k8s_job_name = COALESCE(%s, k8s_job_name) WHERE task_id = %s",
                (runner_ref, task_id),
            )

    def heartbeat(self, task_id: str, message: str | None = None) -> None:
        with self.db.cursor() as cur:
            if message is None:
                cur.execute("UPDATE tasks SET last_heartbeat = now() WHERE task_id = %s", (task_id,))
            else:
                cur.execute(
                    "UPDATE tasks SET last_heartbeat = now(), status_message = %s WHERE task_id = %s",
                    (message[:500], task_id),
                )

    def set_result(self, task_id: str, patch: dict) -> None:
        """深層合併寫入 result JSON（各步驟各寫自己的鍵，不互相覆蓋）。

        ⚠️ PG 的 jsonb `||` 是淺層合併：{"finalize":{"a":1}} || {"finalize":{"b":2}} 會把 a 丟掉。
        所以用 SELECT ... FOR UPDATE 讀回、Python 深層合併、再寫回，同一交易內完成。
        """
        with self.db.cursor() as cur:
            cur.execute("SELECT result FROM tasks WHERE task_id = %s FOR UPDATE", (task_id,))
            row = cur.fetchone()
            current = (row[0] if row and row[0] else {}) or {}
            merged = _deep_merge(current, patch)
            cur.execute(
                "UPDATE tasks SET result = %s::jsonb, last_heartbeat = now() WHERE task_id = %s",
                (json.dumps(merged), task_id),
            )

    def finish(
        self, task_id: str, status: TaskStatus, *, message: str | None = None,
        error: str | None = None, model_version: str | None = None,
        run_report_uri: str | None = None,
    ) -> None:
        with self.db.cursor() as cur:
            cur.execute(
                """
                UPDATE tasks SET status = %s, status_message = COALESCE(%s, status_message),
                       error = %s, finished_at = now(), last_heartbeat = now(),
                       model_version = COALESCE(%s, model_version),
                       run_report_uri = COALESCE(%s, run_report_uri)
                WHERE task_id = %s
                """,
                (status.value, message, error, model_version, run_report_uri, task_id),
            )
        logger.info("task %s → %s%s", task_id, status.value, f"（{error}）" if error else "")

    def request_cancel(self, task_id: str) -> bool:
        """標記取消請求。trainer 每 epoch 檢查；Orchestrator 負責殺容器。
        還在排隊的直接收掉（沒有東西要殺）—— 以前只認 working，排隊中的取消不了（2026-10-21）。"""
        with self.db.cursor() as cur:
            cur.execute(
                "UPDATE tasks SET status = 'cancelled', status_message = '排隊中取消', finished_at = now()"
                " WHERE task_id = %s AND status = 'queued'",
                (task_id,),
            )
            if cur.rowcount > 0:
                logger.info("task %s 排隊中取消", task_id)
                return True
            cur.execute(
                "UPDATE tasks SET result = COALESCE(result, '{}'::jsonb) || '{\"cancel_requested\": true}'::jsonb"
                " WHERE task_id = %s AND status = 'working'",
                (task_id,),
            )
            return cur.rowcount > 0

    def cancel_requested(self, task_id: str) -> bool:
        t = self.get(task_id)
        return bool(t and (t.result or {}).get("cancel_requested"))

    # ── 查詢 ─────────────────────────────────────────────

    def get(self, task_id: str) -> TaskView | None:
        with self.db.cursor() as cur:
            cur.execute("SELECT * FROM tasks WHERE task_id = %s", (task_id,))
            row = cur.fetchone()
            if not row:
                return None
            cols = [c.name for c in cur.description]
        return self._view(dict(zip(cols, row)))

    def list(self, *, type: str | None = None, status: str | None = None, limit: int = 50) -> list[TaskView]:
        conds, params = [], []
        if type:
            conds.append("type = %s"); params.append(type)
        if status:
            conds.append("status = %s"); params.append(status)
        where = f"WHERE {' AND '.join(conds)}" if conds else ""
        with self.db.cursor() as cur:
            cur.execute(f"SELECT * FROM tasks {where} ORDER BY created_at DESC LIMIT %s", (*params, limit))
            cols = [c.name for c in cur.description]
            return [self._view(dict(zip(cols, r))) for r in cur.fetchall()]

    def stale(self, *, seconds: int) -> list[TaskView]:
        """心跳逾時仍 working 的任務。"""
        with self.db.cursor() as cur:
            cur.execute(
                "SELECT * FROM tasks WHERE status = 'working'"
                " AND last_heartbeat < now() - make_interval(secs => %s)",
                (seconds,),
            )
            cols = [c.name for c in cur.description]
            return [self._view(dict(zip(cols, r))) for r in cur.fetchall()]

    # ── 佇列（單機版排程；上 K8s 換成 Kueue，這層拿掉）─────────
    #
    # 業界共識：佇列 + worker，一個 worker 一次一個任務（ClearML agent、Kueue 都是這個形狀）。
    # 這裡用 PostgreSQL 的 SELECT ... FOR UPDATE SKIP LOCKED —— 資料庫佇列的標準寫法：
    # 多個 Orchestrator 同時搶時不會拿到同一筆，也不會互相卡住。

    def running_count(self, *, type: str | None = None, types: list[str] | None = None) -> int:
        with self.db.cursor() as cur:
            if types:
                cur.execute("SELECT count(*) FROM tasks WHERE status = 'working' AND type = ANY(%s)", (list(types),))
            elif type:
                cur.execute("SELECT count(*) FROM tasks WHERE status = 'working' AND type = %s", (type,))
            else:
                cur.execute("SELECT count(*) FROM tasks WHERE status = 'working'")
            return int(cur.fetchone()[0])

    def claim(self, task_id: str) -> bool:
        """把指定任務從 queued 轉成 working。已被別人搶走（非 queued）就回 False。

        submit 與 _drain_queue 都走這條路 —— 狀態轉換只有一個地方，
        running_count 才不會因為「執行緒已起、DB 還是 queued」而少算。
        """
        with self.db.cursor() as cur:
            cur.execute(
                "UPDATE tasks SET status = 'working', status_message = %s, last_heartbeat = now(),"
                " started_at = COALESCE(started_at, now())"
                " WHERE task_id = %s AND status = 'queued'",
                ("開始執行", task_id),
            )
            return cur.rowcount > 0

    def claim_next_queued(self, *, type: str | None = None, types: list[str] | None = None) -> TaskView | None:
        """把最舊的 queued 任務轉成 working 並回傳。沒有就回 None。"""
        if types:
            cond, params = "AND type = ANY(%s)", (list(types),)
        else:
            cond = "AND type = %s" if type else ""
            params = (type,) if type else ()
        with self.db.cursor() as cur:
            cur.execute(
                f"""
                SELECT task_id FROM tasks
                WHERE status = 'queued' {cond}
                ORDER BY created_at
                FOR UPDATE SKIP LOCKED
                LIMIT 1
                """,
                params,
            )
            row = cur.fetchone()
            if not row:
                return None
            task_id = row[0]
            cur.execute(
                "UPDATE tasks SET status = 'working', status_message = %s, last_heartbeat = now()"
                " WHERE task_id = %s",
                ("排隊結束，開始執行", task_id),
            )
        logger.info("task %s 出列開始執行", task_id)
        return self.get(task_id)

    def queued(self, *, type: str | None = None) -> list[TaskView]:
        return self.list(type=type, status=TaskStatus.QUEUED.value, limit=1000)

    def queue_position(self, task_id: str) -> int | None:
        """排在第幾個（1 = 下一個）。非 queued 回 None。"""
        with self.db.cursor() as cur:
            cur.execute(
                """
                SELECT count(*) + 1 FROM tasks
                WHERE status = 'queued'
                  AND created_at < (SELECT created_at FROM tasks WHERE task_id = %s)
                """,
                (task_id,),
            )
            row = cur.fetchone()
            cur.execute("SELECT status FROM tasks WHERE task_id = %s", (task_id,))
            st = cur.fetchone()
        if not st or st[0] != TaskStatus.QUEUED.value:
            return None
        return int(row[0])

    def working(self) -> list[TaskView]:
        return self.list(status=TaskStatus.WORKING.value, limit=1000)

    @staticmethod
    def _view(row: dict[str, Any]) -> TaskView:
        def iso(v):
            return v.isoformat() if isinstance(v, datetime) else (str(v) if v else None)

        return TaskView(
            task_id=row["task_id"], type=row["type"], status=row["status"],
            status_message=row.get("status_message"),
            created_at=iso(row["created_at"]) or "", started_at=iso(row.get("started_at")),
            finished_at=iso(row.get("finished_at")), last_heartbeat=iso(row.get("last_heartbeat")),
            created_by=row["created_by"], channel=row["channel"],
            params=row.get("params") or {}, result=row.get("result"),
            model_version=row.get("model_version"), run_report_uri=row.get("run_report_uri"),
            error=row.get("error"), parent_task_id=row.get("parent_task_id"),
            clone_reason=row.get("clone_reason"),
        )


def store_from_env() -> TaskStore:
    if not os.environ.get("DATABASE_URL"):
        raise RuntimeError("未設定 DATABASE_URL")
    return TaskStore()
