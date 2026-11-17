#!/bin/bash
# 檢查必要環境變數後啟動 runner。nvinferserver 設定由 pipeline.py 以樣板產生。
set -e
: "${MODEL_NAME:?需要 MODEL_NAME（Triton model name，例如 seg_alpha）}"
: "${TRITON_GRPC:?需要 TRITON_GRPC（例如 localhost:8001）}"
: "${CAMERA_SOURCE:?需要 CAMERA_SOURCE=file|api}"
exec ds-runner
