# cpu-runner 映像（demo）：Python + ffmpeg + 整個 workspace（libs / stations / 兩個 app）。
# 跟 deepstream_runner 一樣「全部站別都裝進去，啟動參數決定跑哪站」，只是不需要 DeepStream / GPU。
# build context = repo 根目錄：docker build -f ai-infer/apps/cpu_runner/Dockerfile .
FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

COPY ai-infer/apps/cpu_runner/requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

COPY ai-infer/libs/ ./libs/
COPY ai-infer/stations/ ./stations/
COPY ai-infer/apps/ ./apps/
RUN pip install --no-deps ./libs/* ./stations/* ./apps/deepstream_runner ./apps/cpu_runner \
    && python -c "from edgevision_station_sdk import list_stations; print('stations:', sorted(list_stations()))"

CMD ["cpu-runner"]
