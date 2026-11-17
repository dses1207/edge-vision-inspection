#!/usr/bin/env bash
# trainer 容器進入點：把環境變數轉成 ClearML SDK 設定檔，再執行指令。
#
# 為什麼要產檔：ClearML SDK 連 apiserver 可以純靠環境變數，但「S3 相容儲存（MinIO）的
# endpoint」只能寫在 clearml.conf 的 sdk.aws.s3.credentials 裡。K8s 上同樣用 env → 這支腳本
# 產檔，或直接以 ConfigMap 掛 /root/clearml.conf 蓋掉它。
set -euo pipefail

CONF="${CLEARML_CONFIG_FILE:-/root/clearml.conf}"
if [ ! -f "$CONF" ]; then
  MINIO_HOST="${MINIO_ENDPOINT:-minio:9000}"
  SECURE="${MINIO_SECURE:-false}"
  cat > "$CONF" <<EOF
api {
  api_server: "${CLEARML_API_HOST:-http://clearml-apiserver:8008}"
  web_server: "${CLEARML_WEB_HOST:-http://clearml-webserver:80}"
  files_server: "${CLEARML_FILES_HOST:-http://clearml-fileserver:8081}"
  credentials { "access_key": "${CLEARML_API_ACCESS_KEY:-}", "secret_key": "${CLEARML_API_SECRET_KEY:-}" }
}
sdk {
  aws {
    s3 {
      key: "${MINIO_ROOT_USER:-}"
      secret: "${MINIO_ROOT_PASSWORD:-}"
      region: "us-east-1"
      credentials: [
        {
          host: "${MINIO_HOST}"
          key: "${MINIO_ROOT_USER:-}"
          secret: "${MINIO_ROOT_PASSWORD:-}"
          multipart: false
          secure: ${SECURE}
          verify: false
        }
      ]
    }
  }
  development {
    default_output_uri: "${CLEARML_OUTPUT_URI:-s3://${MINIO_HOST}/experiments/}"
    vcs_repo_detect_async: false
  }
}
EOF
fi
export CLEARML_CONFIG_FILE="$CONF"
exec "$@"
