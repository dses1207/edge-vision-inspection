#!/bin/bash
# CI 部署（CI 呼叫；手動也能跑）。
#   KUBECONFIG=… SHA=<edgevision-service 標籤> [TRAIN_TAG=<edgevision-train 標籤>] [OVERLAY=<環境>] bash deploy/ci/deploy_test.sh
#   OVERLAY 預設 example。
#
#   ⓪ 資料庫結構變更（migrate_db.sh）
#   ① kustomize overlay → apply（apply.sh）
#   ② runner：交給叢集上的 runner-reconciler（這裡只確保 ConfigMap edgevision-runner 存在）
#   ③ 等所有 Deployment 就緒，/healthz 要 healthy
set -euo pipefail
cd "$(dirname "$0")/../.."
: "${SHA:?要設 SHA（edgevision-service 映像標籤）}"
NS=edgevision

current_image() {   # 叢集上某 Deployment 第一個容器（或某個 env）的映像
  kubectl -n $NS get deploy "$1" -o jsonpath="$2" 2>/dev/null || true
}

# 訓練映像：CI 會建 edgevision-train:<sha> 並傳 TRAIN_TAG=$SHA；手動跑沒給就沿用 control-api 現在的 TRAINER_IMAGE
if [ -z "${TRAIN_TAG:-}" ]; then
  TRAIN_TAG=$(current_image control-api '{.spec.template.spec.containers[0].env[?(@.name=="TRAINER_IMAGE")].value}' | sed 's/.*://')
  TRAIN_TAG=${TRAIN_TAG:-$SHA}
fi
echo "edgevision-service:$SHA  edgevision-train:$TRAIN_TAG"
bash deploy/ci/migrate_db.sh            # 資料庫結構先補上，再換服務映像

# 服務金鑰（控制 API 身分關卡認 X-Service-Token）：缺哪把補哪把，已有的不換（換了要重啟所有呼叫端）。
# 金鑰只經 stdin 給 kubectl，不印、不進命令列參數、不進 git。要輪替：刪掉 Secret 再部署一次。
ensure_service_tokens() {
  local have
  have=$(kubectl -n $NS get secret edgevision-service-tokens -o jsonpath='{.data}' 2>/dev/null || true)
  kubectl -n $NS get secret edgevision-service-tokens >/dev/null 2>&1 \
    || kubectl -n $NS create secret generic edgevision-service-tokens >/dev/null
  for svc in alert-feedback output-api; do
    if ! grep -q "\"$svc\"" <<<"$have"; then
      printf '{"stringData":{"%s":"%s"}}' "$svc" "$(openssl rand -hex 32)" \
        | kubectl -n $NS patch secret edgevision-service-tokens --type merge --patch-file /dev/stdin >/dev/null
      echo "服務金鑰：新增 $svc"
    fi
  done
  # 網址簽名金鑰（<video src> 用）：分開放，否則會被當成一把服務金鑰
  kubectl -n $NS get secret edgevision-url-signing >/dev/null 2>&1 || {
    printf '{"apiVersion":"v1","kind":"Secret","metadata":{"name":"edgevision-url-signing"},"stringData":{"key":"%s"}}' \
      "$(openssl rand -hex 32)" | kubectl -n $NS create -f - >/dev/null
    echo "網址簽名金鑰：新增"
  }
}
ensure_service_tokens
EDGEVISION_TAG=$SHA TRAIN_TAG=$TRAIN_TAG bash deploy/k8s/apply.sh "${OVERLAY:-example}"

# runner：不在這裡產生 —— 叢集上的 runner-reconciler 每 30 秒照線路表自己建 / 收 / 換。
# 它讀 ConfigMap edgevision-runner 的 image；第一次部署沒有就建一個（沿用叢集上現有 runner 的映像，或 RUNNER_IMAGE）。
# 之後由 ai-infer 的 CI 更新這個值，這裡不覆蓋。
if ! kubectl -n $NS get configmap edgevision-runner >/dev/null 2>&1; then
  RUNNER_IMAGE=${RUNNER_IMAGE:-$(kubectl -n $NS get deploy -l app=runner -o jsonpath='{.items[0].spec.template.spec.containers[0].image}' 2>/dev/null || true)}
  if [ -n "$RUNNER_IMAGE" ]; then
    kubectl -n $NS create configmap edgevision-runner --from-literal=image="$RUNNER_IMAGE"
    echo "建立 ConfigMap edgevision-runner：$RUNNER_IMAGE"
  else
    echo "⚠️ 沒有 ConfigMap edgevision-runner、叢集上也沒有 runner：設 RUNNER_IMAGE 再跑，否則調和器不會建 runner" >&2
  fi
fi

# 等就緒
for d in $(kubectl -n $NS get deploy -o jsonpath='{.items[*].metadata.name}'); do
  kubectl -n $NS rollout status "deploy/$d" --timeout=600s
done
kubectl -n $NS exec deploy/output-api -- python3 -c '
import json, sys, urllib.request as u
d = json.load(u.urlopen("http://localhost:18080/healthz"))
for l in d.get("lines", []):
    bad = [c.get("id") or c.get("camera_id") for c in l.get("cameras", []) if c.get("state") != "ok"]
    print(l.get("line_id"), l.get("fps"), "fps，異常相機", bad or "無")
print("healthy", d.get("healthy"))
sys.exit(0 if d.get("healthy") else 1)'
