"""Job Orchestrator：把 task 變成一串容器步驟，追蹤狀態、心跳、對帳。

    控制 API → tasks 表 → Orchestrator.submit(task_id)
        → JobRunner（DockerRunner 現在 / K8sRunner 未來）依序跑：
            trainer（GPU）→ trtexec（GPU）→ [臨時 Triton] finalize（CPU）
        → auto_promote 時：Registry 移 tag → 部署 Controller → Triton

換到 K8s 只換 JobRunner 實作：步驟定義（映像、指令、環境變數、掛載）不變。
"""

from edgevision.orchestrator.orchestrator import Orchestrator
from edgevision.orchestrator.runner import DockerRunner, JobRunner, K8sRunner, StepSpec

__all__ = ["Orchestrator", "DockerRunner", "JobRunner", "K8sRunner", "StepSpec"]
