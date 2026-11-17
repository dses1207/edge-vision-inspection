#!/usr/bin/env python3
"""PTZ 服務啟動點（舊系統的 PTZ 服務移植；edgevision/ptz/）。設定全從環境變數。"""
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
    uvicorn.run("edgevision.ptz.api:app", host="0.0.0.0", port=int(os.environ.get("PTZ_PORT", "8097")),
                log_level=os.environ.get("LOG_LEVEL", "info").lower())
