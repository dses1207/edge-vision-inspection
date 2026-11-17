# 推論服務映像
#
# 包含：
#   ・ffmpeg —— 影格編碼與 RTSP 推流（打標後重新編碼用）
#   ・PyAV   —— RTSP 解碼（能取得 pts，cv2 拿不到）
#   ・OpenCV headless —— 幾何運算，不需 GUI 相依
#
# 刻意不含：ultralytics / torch。推論走 Triton，模型不在此映像內 ——
# 換模型不需重建映像，這是模型熱切換的前提。

# 基底映像可改走自己的 Docker Hub 代理（避開限流）：--build-arg PY_IMAGE=<proxy>/library/python:3.12-slim
ARG PY_IMAGE=library/python:3.12-slim
FROM ${PY_IMAGE} AS base

ARG HTTP_PROXY
ARG HTTPS_PROXY
ARG NO_PROXY
ENV HTTP_PROXY=${HTTP_PROXY} HTTPS_PROXY=${HTTPS_PROXY} NO_PROXY=${NO_PROXY} \
    http_proxy=${HTTP_PROXY} https_proxy=${HTTPS_PROXY} no_proxy=${NO_PROXY}

# apt 加 retry：Debian 鏡像偶發抓不到套件索引，重試可避免 build 隨機失敗
RUN echo 'Acquire::Retries "5";' > /etc/apt/apt.conf.d/80-retries \
    && apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# 相依套件單獨一層，改程式碼時不必重裝
COPY requirements.txt .
COPY ai-infer/libs/contracts ./ai-infer/libs/contracts
RUN pip install --no-cache-dir -r requirements.txt

# 測試套件：預裝進映像，讓 pytest 在容器內免現裝即可跑
#   —— 測試容器接內部網路(minio/pg/triton)時本就連不到 PyPI，
#      預裝是唯一乾淨解。體積小(pytest+onnx ~20MB)，不影響生產。
RUN pip install --no-cache-dir pytest onnx

COPY edgevision/ ./edgevision/
COPY libs/ ./libs/
COPY configs/ ./configs/
# 全部入口都進映像：錄影 / 通知 / NAS 錄影以前靠 compose 把主機程式碼掛進 /app 才跑得動，
# K8s 上沒有那個主機目錄（2026-09-30 搬 K8s 時發現）
COPY run_*.py ./
# 模型部署工具（K8s 上用 Job 跑：plan + manifest → MinIO → Triton）
# 舊模型補登錄 Registry（deploy_engine 那條路不登錄，上線後要補）
COPY scripts/ ./scripts/

# 執行期不需要 proxy（容器內只連內網）
ENV HTTP_PROXY= HTTPS_PROXY= http_proxy= https_proxy=

# ⚠️ 這個映像有多個入口，由 compose / K8s 指定 command 決定跑哪個：
#     run_api.py          output-api（監看、設定管理、WebSocket）
#     run_control.py      control-api（訓練、模型註冊）
#     run_processor.py    event-processor（Redis → PG 落地）
#     run_recorder.py     recorder（預錄 + NG 事件影片）
#     run_notifier.py     notifier（日報 + 健康巡檢）
#     run_nas_recorder.py nas-recorder（24h 連續錄影）
#     python -m edgevision.mcp.remote   MCP server
#
# 映像層【不設 HEALTHCHECK】—— 各服務的 port 與健康定義都不同：
#   · run_processor.py 根本不是 web 服務（它的健康是「consumer group 還在不在」）
#   · run_api.py 實際跑在 API_PORT（18080），不是 8080
# 這裡寫死一個打 :8080/healthz 的檢查曾經害 autoheal 每 70 秒重啟一次、
# 累計 948 次，症狀是前端 WebSocket 一直斷、偵測框狂跳，查了很久才找到。
# 健康檢查一律由 compose / K8s 各自宣告。

CMD ["python", "run_api.py"]
