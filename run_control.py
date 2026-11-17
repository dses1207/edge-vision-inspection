#!/usr/bin/env python3
"""控制 API + Orchestrator 啟動點。設定全從環境變數（Compose env_file ↔ K8s ConfigMap/Secret）。"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import uvicorn

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)-7s %(name)-24s %(message)s", datefmt="%H:%M:%S",
)

if __name__ == "__main__":
    uvicorn.run("edgevision.api.control:app", host="0.0.0.0", port=int(os.environ.get("CONTROL_PORT", "8090")),
                log_level=os.environ.get("LOG_LEVEL", "info").lower())
