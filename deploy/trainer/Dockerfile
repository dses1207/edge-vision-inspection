# trainer 映像：訓練 Job 用（K8s 上就是 Job 的映像）
#
# 基底用 NGC pytorch（驗證過的組合）。
# 內含：ultralytics（訓練/匯出 ONNX）、clearml（實驗記錄）、lakefs（資料集版本）、
#       onnxruntime（pt vs onnx 一致性）、專案的 edgevision/ libs/ configs/。
# 不含 trtexec：engine 由 Triton 映像自帶的 trtexec 建（Orchestrator 的 trtexec 步驟）。
#
# ⚠️ 相依地獄的正解：裝完 ultralytics 後強制釘 numpy 1.26 + opencv 4.8，
#    並刪掉 cv2/typing 的 DictValue stub；最後一個 import 閘門過不了就 build 失敗，
#    不要把壞映像帶到訓練時才發現。

# 基底可換（build arg）：外網拉不到 NGC pytorch 時，改用機台已有的 Triton 映像
#   TRAINER_BASE_IMAGE=nvcr.io/nvidia/tritonserver:24.09-py3 TRAINER_INSTALL_TORCH=1
ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:24.09-py3
FROM ${BASE_IMAGE}

ARG INSTALL_TORCH=0
ARG HTTP_PROXY
ARG HTTPS_PROXY
ARG NO_PROXY
ARG GIT_COMMIT=unknown
ENV HTTP_PROXY=${HTTP_PROXY} HTTPS_PROXY=${HTTPS_PROXY} NO_PROXY=${NO_PROXY} \
    http_proxy=${HTTP_PROXY} https_proxy=${HTTPS_PROXY} no_proxy=${NO_PROXY} \
    PYTHONUNBUFFERED=1 \
    CODE_COMMIT=${GIT_COMMIT} \
    WEIGHTS_DIR=/weights \
    YOLO_CONFIG_DIR=/tmp/ultralytics

WORKDIR /app

# 基底沒有 torch 時（例如用 Triton 映像）由 PyPI 安裝（wheel 自帶 CUDA 函式庫）
RUN if [ "$INSTALL_TORCH" = "1" ]; then pip install --no-cache-dir torch torchvision; fi

# pip 走內網 proxy 偶發 502。對策：
#   ・BuildKit cache mount 保留 pip 下載快取 → 失敗重跑只補缺的 wheel，不從頭
#   ・拆成多個小層 → 已成功的層被 Docker 快取
#   ・單次 timeout 短、重試次數多 → 抖一下就換下一次，不白等
# ultralytics 會拉進 opencv-python 5.x（要 numpy 2）與基底的 numpy1 / cv2 打架
# → 裝完後移除所有 opencv 變體，只留釘版的 headless 4.8 + numpy 1.26。
# cv2/typing 的 DictValue stub 用 find 刪（不能靠 import cv2 找路徑：壞掉時 import 本身就失敗）。
# 離線來源：外網不可靠時，在能上網的機器 `pip download` 整組 wheel 放進 repo 的 wheels/，
# build 時帶 TRAINER_PIP_EXTRA="--no-index --find-links=/wheels"（wheels/.keep 保證 COPY 不會失敗）
ARG PIP_EXTRA=""
COPY wheels/ /wheels/
ENV PIP_DEFAULT_TIMEOUT=30 PIP_RETRIES=25
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install $PIP_EXTRA "ultralytics>=8.3,<9"
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install $PIP_EXTRA "clearml>=1.16" boto3 "lakefs>=0.8" onnx onnxruntime onnxslim
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install $PIP_EXTRA "pydantic>=2.9" "psycopg[binary,pool]>=3.2" "minio>=7.2" pyyaml "httpx>=0.27" \
        "tritonclient[http]>=2.50" pytest
RUN --mount=type=cache,target=/root/.cache/pip \
    (pip uninstall -y opencv-python opencv-python-headless opencv-contrib-python opencv-contrib-python-headless || true) \
    && pip install $PIP_EXTRA --force-reinstall --no-deps "numpy==1.26.4" "opencv-python-headless==4.8.1.78" \
    && find / -path "*-packages/cv2/typing/__init__.py" -not -path "/proc/*" -delete 2>/dev/null; \
    python -c "import numpy, cv2, torch, ultralytics, clearml, lakefs, onnxruntime; \
               print('import gate ok', numpy.__version__, cv2.__version__, torch.__version__, ultralytics.__version__)"

# 預載起手權重：訓練時不需要外網（K8s 訓練池通常沒有）
# 離線時把 .pt 一起放在 wheels/；有本地檔 Ultralytics 就不會去 GitHub 下載
RUN mkdir -p /weights && (cp /wheels/*.pt /weights/ 2>/dev/null || true) && cd /weights \
    && python -c "from ultralytics import YOLO; YOLO('yolo11n-seg.pt'); YOLO('yolo11m-seg.pt')" \
    && ls -la /weights

# 偵測資料契約：libs/contracts 是指向 edgevision-contracts 套件的 shim（同主映像 requirements.txt，版號跟著升）。
# --no-deps：只要它本身（純 Python、相依 pydantic 已在上面），不能讓它動到上面釘住的 numpy / opencv。
# 漏裝時 import edgevision.training 就失敗（2026-10-13 重建訓練映像時發現：10/1 改成 shim 後這裡沒跟上）。
COPY ai-infer/libs/contracts /tmp/edgevision-contracts
RUN pip install --no-deps /tmp/edgevision-contracts \
    && python -c "import edgevision_contracts.detection"

# 自動標注（edgevision/media/autolabel.py）：YOLO-World 換提示詞要 CLIP 文字模型（ultralytics 執行時才 pip 裝，叢集內不一定有外網）。
# CLIP 權重（ViT-B-32.pt，338 MB）不進映像，放 MinIO models/_base/clip/，任務開始時取。
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install $PIP_EXTRA --no-deps ftfy "git+https://github.com/ultralytics/CLIP.git" \
    && python -c "import clip, ftfy; print('clip ok')"

COPY edgevision/ ./edgevision/
COPY libs/ ./libs/
COPY configs/ ./configs/
COPY tools/ ./tools/
COPY pyproject.toml .
# ⚠️ 根目錄 .dockerignore 排除 deploy/ 與 tests/，故進入點放 tools/；tests 於測試時掛載
RUN cp tools/trainer_entrypoint.sh /entrypoint.sh && chmod +x /entrypoint.sh \
    && python -c "import edgevision.training.pipeline; print('code import gate ok')"

# 執行期不走 proxy（只連內網服務）
ENV HTTP_PROXY= HTTPS_PROXY= http_proxy= https_proxy= NO_PROXY=* no_proxy=*

ENTRYPOINT ["/entrypoint.sh"]
CMD ["python", "-m", "edgevision.training.pipeline", "--help"]
