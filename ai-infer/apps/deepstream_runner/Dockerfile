# deepstream-runner 映像：DeepStream 7.1 基底 + pyds + 整個 workspace（libs / 全部 stations / 這個 app）
#
# 三層都裝進同一個映像；體積幾乎全是基底（約 8 GB），我們的程式碼只有幾百 KB。
# 一條線路一個容器，用 LINE_ID / CAMERA_SOURCE 決定跑哪站、接哪些相機。
#
# build 要在 workspace 根執行：docker build -f apps/deepstream_runner/Dockerfile -t deepstream-runner:<sha> .
# 基底版本用 build-arg 選：
#   4080 / Ada（CUDA 12.x）：DS_VERSION=7.1 PYDS_VERSION=1.2.0 PY_TAG=cp310（預設）
#   Blackwell（driver 595 / CUDA 13）：DS_VERSION=8.0 PYDS_VERSION=1.2.2 PY_TAG=cp312，配 Triton 25.08
# 兩者程式碼相同，只有基底、pyds wheel、entrypoint 路徑不同。
ARG DS_VERSION=7.1
FROM nvcr.io/nvidia/deepstream:${DS_VERSION}-triton-multiarch
ARG DS_VERSION
ARG PYDS_VERSION=1.2.0
ARG PY_TAG=cp310

# proxy：不寫進 ENV。Docker 的 ~/.docker/config.json "proxies" 會在 build 與 run 時自動注入，
# 硬塞空字串反而讓 GLib 的 proxy resolver 報 assertion。需要時用 --build-arg 傳給 RUN 就好。
ARG HTTP_PROXY
ARG HTTPS_PROXY
ARG NO_PROXY

# 官方要求：執行期才有 NVDEC / NVENC 函式庫
ENV NVIDIA_DRIVER_CAPABILITIES=${NVIDIA_DRIVER_CAPABILITIES},video

# x264enc（ENCODER=x264 用）：官方映像因 GPL 授權刻意不帶 gst-plugins-ugly。
# GeForce 整台機器只有 8 個 NVENC session，第 9 條線起要靠它（見 pipeline.encoder_spec）。
# ⚠️ 兩個坑（2026-09-29 實測，deepstream-runner:<tag> 一啟動 nvinferserver / nvstreammux 都 "No such element"）：
#   ① build 階段沒有 GPU 驅動（libcuda），任何 gst-inspect 都會把 DS 自家依賴 libcuda 的 plugin
#      記成 blacklisted 寫進 /root/.cache/gstreamer-1.0 的 registry；執行期 GStreamer 信任這份 cache
#      不重掃 → 元件消失。所以 build 時檢查完 x264enc 之後【一定要刪 cache】，讓第一次啟動（有 GPU）重掃。
#   ② 官方映像的做法是「dpkg 登記為已安裝、但 .so 檔被刪掉」→ 一般 apt install 什麼都不做，要 --reinstall。
#      不用 NVIDIA 的 user_additional_install.sh（它會重裝十幾個套件）；只重裝 plugins-ugly + libx264，同版本不升級。
#   ③ libx264 的套件名帶 ABI 版號：22.04（DS 7.1）是 libx264-163、24.04（DS 8.0）是 libx264-164 → 用 apt-cache 找，不寫死。
# 預設關閉（WITH_X264=0）直到測試機驗證過；要 x264 的映像用 --build-arg WITH_X264=1。
ARG WITH_X264=0
RUN if [ "$WITH_X264" = "1" ]; then \
      apt-get update \
      && X264_PKG=$(apt-cache search --names-only '^libx264-[0-9]+$' | awk '{print $1}' | head -1) \
      && echo "x264 套件：$X264_PKG" \
      && apt-get install -y --no-install-recommends --reinstall gstreamer1.0-plugins-ugly "$X264_PKG" \
      && gst-inspect-1.0 x264enc > /dev/null \
      && rm -rf /var/lib/apt/lists/* /root/.cache/gstreamer-1.0; \
    fi

# pyds 不在 PyPI：DS 7.1 = Python 3.10 = pyds 1.2.0 cp310；DS 8.0 = Python 3.12 = pyds 1.2.2 cp312
RUN WHL=pyds-${PYDS_VERSION}-${PY_TAG}-${PY_TAG}-linux_x86_64.whl \
    && curl -sL -o /tmp/$WHL \
      https://github.com/NVIDIA-AI-IOT/deepstream_python_apps/releases/download/v${PYDS_VERSION}/$WHL \
    && pip3 install --no-cache-dir --break-system-packages /tmp/$WHL 2>/dev/null || pip3 install --no-cache-dir /tmp/$WHL \
    && rm -f /tmp/$WHL

# ⚠️ numpy(OpenBLAS)/OpenCV 預設會開滿核心的執行緒池，mask 那種小矩陣不值得平行，
# 結果是幾十條 thread 忙等（實測 32 核機器吃到 29 核、270 條 thread）。
# 平行交給 worker 分片，函式庫自己一律單執行緒。
ENV OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
WORKDIR /app
# DeepStream 容器的 Python 是系統的 /usr/bin/python3，不是虛擬環境；映像裡不用 uv。
# 第三方相依由 uv.lock 匯出成 requirements.txt（見該檔頭）→ 系統 pip 安裝，成為可快取層。
COPY apps/deepstream_runner/requirements.txt /tmp/requirements.txt
RUN pip3 install --no-cache-dir --break-system-packages -r /tmp/requirements.txt 2>/dev/null \
    || pip3 install --no-cache-dir -r /tmp/requirements.txt
COPY libs/ ./libs/
COPY stations/ ./stations/
COPY apps/ ./apps/
# pyds 只在有 GPU 時才 import 得到（要 libcuda），建置階段不檢查它
# 全部 libs 與【全部】stations 用 glob，新增一站不用改這裡（映像含所有手冊，啟動參數決定翻哪本）
RUN (pip3 install --no-cache-dir --no-deps --break-system-packages ./libs/* ./stations/* ./apps/deepstream_runner 2>/dev/null \
     || pip3 install --no-cache-dir --no-deps ./libs/* ./stations/* ./apps/deepstream_runner) \
    && python3 -c "import ds_runner, edgevision_station_zone_intrusion; from edgevision_station_sdk import list_stations; print('stations:', sorted(list_stations()))"

ENV NVINFER_TEMPLATE=/app/nvinferserver.tmpl.txt NVINFER_CONFIG=/app/nvinferserver.txt \
    CAMERA_FILE=/etc/edgevision/line.json TRACKER_CONFIG=/app/tracker_nvsort.yml
# nvinferserver 設定由 pipeline.py 以樣板產生（MODEL_NAME / TRITON_GRPC / 推論間隔）
COPY apps/deepstream_runner/configs/nvinferserver.tmpl.txt /app/nvinferserver.tmpl.txt
# nvtracker 低階設定（TRACKER_ENABLED=1 才用到）；改自官方 config_tracker_NvSORT.yml
COPY apps/deepstream_runner/configs/tracker_nvsort.yml /app/tracker_nvsort.yml
COPY apps/deepstream_runner/entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh
# ⭐ 一定要經過 DeepStream 官方 entrypoint：它跑 /opt/nvidia/entrypoint.d/*（驅動檢查等）。
# 跳過它，nvinferserver 對 Triton 註冊 CUDA shared memory 會被拒（實測 4080：invalid args）。
# /opt/nvidia/deepstream/deepstream 是指向當前版本目錄的 symlink，兩個版本都適用
ENTRYPOINT ["/opt/nvidia/deepstream/deepstream/entrypoint.sh", "/app/entrypoint.sh"]
