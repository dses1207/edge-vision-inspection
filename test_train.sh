#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════
# 訓練鏈路整體聯通性測試（在 GPU 機台跑；需 infra + deploy + train-infra + control 都在）
#
#   ./test_train.sh            全部：服務就緒 → 資料進 LakeFS → API 建訓練任務 → 等完成
#                              → 檢查 Registry/MinIO → promote → Triton 切版 → rollback
#   ./test_train.sh services   只檢查服務就緒
#   ./test_train.sh unit       容器內單元 + 契約測試（含前處理一致性，需 trainer 映像）
#   ./test_train.sh integration 整合測試（Registry / LakeFS / 控制 API）
#   ./test_train.sh e2e        端到端（合成資料、epochs=1、yolo11n-seg，數分鐘）
#
# 技術棧與正式一致（LakeFS / ClearML / Ultralytics / trtexec / Triton），只有資料與 epoch 是小的。
# ══════════════════════════════════════════════════════════
set -uo pipefail
cd "$(dirname "$0")"

MODE="${1:-all}"
COMPOSE="docker compose -f deploy/docker-compose.yml"
NET="${DOCKER_NETWORK:-edgevision_edgevision}"
IMAGE="${EDGEVISION_IMAGE:-edgevision:mvp}"
TRAINER_IMAGE="${TRAINER_IMAGE:-edgevision/trainer:dev}"
MODEL="${E2E_MODEL:-seg_alpha}"
EPOCHS="${E2E_EPOCHS:-1}"
BASE="${E2E_BASE_MODEL:-yolo11n-seg.pt}"
PASS=0; FAIL=0
hr(){ printf '%.0s─' {1..64}; echo; }
head_(){ echo; hr; echo "  $1"; hr; }
ok(){ echo "  ✅ $1"; }
bad(){ echo "  ❌ $1"; }

# 讀 .env（ClearML / LakeFS 憑證等）—— 只在本 shell，供容器內測試使用
if [ -f deploy/.env ]; then set -a; . deploy/.env; set +a; fi
CONTROL="${CONTROL_URL:-http://localhost:${CONTROL_PORT:-8090}}"

# 容器內執行（掛原始碼，走內部網路）
in_container() {  # image [docker args...] -- cmd...
    local img="$1"; shift
    local dargs=()
    while [ $# -gt 0 ] && [ "$1" != "--" ]; do dargs+=("$1"); shift; done
    [ "${1:-}" = "--" ] && shift
    docker run --rm --network "$NET" \
        -v "$PWD/edgevision:/app/edgevision:ro" -v "$PWD/libs:/app/libs:ro" -v "$PWD/tests:/app/tests:ro" \
        -v "$PWD/tools:/app/tools:ro" -v "$PWD/configs:/app/configs:ro" -v "$PWD/pyproject.toml:/app/pyproject.toml:ro" \
        --env-file deploy/.env \
        -e NO_PROXY='*' -e no_proxy='*' \
        -e MINIO_ENDPOINT=minio:9000 -e TRITON_URL=http://triton:8000 \
        -e DATABASE_URL="postgresql://${POSTGRES_USER}:${POSTGRES_PASSWORD}@postgres:5432/${POSTGRES_DB}" \
        -e CLEARML_API_HOST=http://clearml-apiserver:8008 -e LAKEFS_ENDPOINT=http://lakefs:8000 \
        -e CONTROL_URL=http://control-api:8090 -e CLEARML_PROJECT=edgevision-itest -e EDGEVISION_TOKEN -e EDGEVISION_SERVICE_TOKEN \
        -w /app "${dargs[@]}" "$img" "$@"
}

# ── 服務就緒 ──────────────────────────────────────────────
check_services() {
    head_ "服務就緒檢查"
    local all=0
    chk(){ if eval "$2" >/dev/null 2>&1; then ok "$1"; else bad "$1"; all=1; fi; }
    chk "MinIO          localhost:9000"          "curl -sf -m 3 http://localhost:9000/minio/health/live"
    chk "PostgreSQL     tasks 表"                 "$COMPOSE exec -T postgres psql -U $POSTGRES_USER -d $POSTGRES_DB -c 'select 1 from tasks limit 1'"
    chk "Triton         /v2/health/ready"         "curl -sf -m 3 http://localhost:8000/v2/health/ready"
    chk "ClearML api    /debug.ping"              "curl -sf -m 3 http://localhost:${CLEARML_API_PORT:-8008}/debug.ping"
    chk "ClearML web    :${CLEARML_WEB_PORT:-8180}" "curl -sf -m 3 -o /dev/null http://localhost:${CLEARML_WEB_PORT:-8180}/"
    chk "LakeFS         /api/v1/healthcheck"      "curl -sf -m 3 http://localhost:${LAKEFS_PORT:-8010}/api/v1/healthcheck"
    chk "Control API    /healthz"                 "curl -sf -m 5 $CONTROL/healthz"
    chk "trainer 映像   $TRAINER_IMAGE"          "docker image inspect $TRAINER_IMAGE"
    chk "GPU            nvidia-smi"               "nvidia-smi -L"
    [ $all -eq 0 ] && PASS=$((PASS+1)) || FAIL=$((FAIL+1))
    return $all
}

# ── 單元 + 契約（trainer 映像內，有 ultralytics）────────────
run_unit() {
    head_ "單元 + 契約測試（trainer 映像內）"
    in_container "$TRAINER_IMAGE" --entrypoint python -- -m pytest \
        tests/test_training_contracts.py tests/test_train_config.py tests/test_gates.py tests/test_parity.py \
        tests/test_dataset.py tests/test_orchestrator.py tests/test_config_gen.py \
        tests/contract/test_preprocess_parity.py -q --no-header -p no:cacheprovider
    [ $? -eq 0 ] && PASS=$((PASS+1)) || FAIL=$((FAIL+1))
}

# ── 整合（Registry / LakeFS / 控制 API）───────────────────
run_integration() {
    head_ "整合測試（Registry / LakeFS / 控制 API）"
    in_container "$IMAGE" -- python -m pytest \
        tests/integration/test_registry.py tests/integration/test_lakefs.py tests/integration/test_control_api.py \
        -v --no-header -p no:cacheprovider
    [ $? -eq 0 ] && PASS=$((PASS+1)) || FAIL=$((FAIL+1))
}

# ── 端到端 ────────────────────────────────────────────────
# 身分：EDGEVISION_TOKEN（SSO token）或 EDGEVISION_SERVICE_TOKEN（服務金鑰）；都沒有才退回 X-Actor（強制模式會 401）
auth_hdr() {
    if [ -n "${EDGEVISION_TOKEN:-}" ]; then printf 'Authorization: Bearer %s' "$EDGEVISION_TOKEN"
    elif [ -n "${EDGEVISION_SERVICE_TOKEN:-}" ]; then printf 'X-Service-Token: %s' "$EDGEVISION_SERVICE_TOKEN"
    else printf 'X-Actor: test_train.sh'; fi
}
api() { curl -sf -m 30 -H 'Content-Type: application/json' -H "$(auth_hdr)" -H 'X-Channel: api' "$@"; }
jq_(){ python3 -c "import sys,json
v=json.load(sys.stdin)
for k in sys.argv[1].split('.'): v=v[k]
print(v)" "$1"; }

run_e2e() {
    head_ "端到端：合成資料 → LakeFS → API 訓練 → Gate → 轉檔 → 註冊 → promote → Triton → rollback"
    local work; work="$(mktemp -d)"; local fail0=$FAIL
    echo "  工作目錄 $work"

    # ① 合成資料集（訓練 + 黃金測試集用不同 seed，保證零重疊）
    in_container "$IMAGE" -v "$work:/out" -- python tools/make_synthetic_seg_dataset.py --out /out/train --n "${E2E_N:-40}" --seed 11 >/dev/null || { bad "產生合成資料失敗"; FAIL=$((FAIL+1)); return; }
    in_container "$IMAGE" -v "$work:/out" -- python tools/make_synthetic_seg_dataset.py --out /out/golden --n 12 --seed 23 >/dev/null
    ok "合成資料集已產生"

    # ② 進 LakeFS，拿資料集版本 ID（不是 Git）
    local ds gd
    ds=$(in_container "$IMAGE" -v "$work:/out" -- python tools/dataset_ingest.py --src /out/train --repo alpha -m "e2e 合成 $(date +%F)" --json | jq_ dataset_ref) || { bad "LakeFS 匯入失敗"; FAIL=$((FAIL+1)); return; }
    gd=$(in_container "$IMAGE" -v "$work:/out" -- python tools/dataset_ingest.py --src /out/golden --repo golden -m "e2e golden" --all-val --json | jq_ dataset_ref)
    ok "資料集版本 ID：$ds"
    ok "黃金測試集版本 ID：$gd"

    # ③ 透過控制 API 建訓練任務（唯一入口）
    local tid
    tid=$(api -X POST "$CONTROL/api/training/jobs" -d "{
        \"model_name\": \"$MODEL\", \"dataset_ref\": \"$ds\", \"golden_ref\": \"$gd\",
        \"overrides\": {\"base_model\": \"$BASE\", \"train\": {\"epochs\": $EPOCHS, \"batch\": 8, \"workers\": 2},
                        \"gates\": \"configs/gates/e2e.yaml\"},
        \"notes\": \"test_train.sh e2e\"}" | jq_ task_id) || { bad "建任務失敗"; FAIL=$((FAIL+1)); return; }
    ok "任務已建立：$tid"

    # ④ 等完成（心跳訊息即時顯示）
    local status="working" last="" t0=$SECONDS
    while [ "$status" = "working" ]; do
        sleep 10
        local j; j=$(api "$CONTROL/api/tasks/$tid") || { bad "查任務失敗"; break; }
        status=$(echo "$j" | jq_ status); local msg; msg=$(echo "$j" | jq_ status_message)
        [ "$msg" != "$last" ] && { echo "    [$((SECONDS-t0))s] $status — $msg"; last="$msg"; }
        [ $((SECONDS-t0)) -gt "${E2E_TIMEOUT:-2400}" ] && { bad "逾時"; break; }
    done
    if [ "$status" != "completed" ]; then
        bad "任務狀態 $status"; echo "--- trainer log（尾 60 行）---"; api "$CONTROL/api/tasks/$tid/logs?step=trainer&tail=60"; echo
        FAIL=$((FAIL+1)); return
    fi
    local ver; ver=$(api "$CONTROL/api/tasks/$tid" | jq_ model_version)
    ok "訓練任務完成，註冊為 $MODEL v$ver"

    # ⑤ Registry 與 MinIO 檢查
    local v; v=$(api "$CONTROL/api/models/$MODEL/versions/$ver")
    echo "$v" | jq_ lineage.dataset_version | grep -q "$ds" && ok "血緣 dataset_version 正確" || { bad "血緣錯"; FAIL=$((FAIL+1)); }
    for kind in pytorch onnx tensorrt run_report eval_report; do
        echo "$v" | jq_ "artifacts.$kind" | grep -q "s3://models/$MODEL/v$ver/" && ok "artifact $kind 在 MinIO 規範路徑" || { bad "缺 artifact $kind"; FAIL=$((FAIL+1)); }
    done
    api "$CONTROL/api/models/$MODEL/versions" | grep -q '"candidate"' && ok "tag=candidate（等人放行）" || bad "沒有 candidate tag"

    # ⑥ promote → Triton 切版
    local before; before=$(curl -s -X POST localhost:8000/v2/repository/index | python3 -c "import sys,json;print([m.get('version') for m in json.load(sys.stdin) if m['name']=='$MODEL' and m.get('state')=='READY'])")
    local dt; dt=$(api -X POST "$CONTROL/api/models/$MODEL/promote" -d "{\"version\": $ver, \"reason\": \"e2e\"}" | jq_ task_id)
    for i in $(seq 1 30); do sleep 3; s=$(api "$CONTROL/api/tasks/$dt" | jq_ status); [ "$s" != "working" ] && break; done
    [ "$s" = "completed" ] && ok "promote 任務完成" || { bad "promote 失敗：$(api "$CONTROL/api/tasks/$dt" | jq_ status_message)"; FAIL=$((FAIL+1)); }
    curl -sf "localhost:8000/v2/models/$MODEL/versions/$ver/ready" >/dev/null && ok "Triton 正在服務 $MODEL v$ver（之前 $before）" || { bad "Triton 未服務 v$ver"; FAIL=$((FAIL+1)); }
    api "$CONTROL/api/models" | grep -qE "\"production\": ?$ver[,}]" && ok "Registry production=v$ver" || bad "Registry production 不是 v$ver"

    # ⑦ rollback（回到之前的 production；沒有就略過）
    local prev; prev=$(echo "$before" | tr -d "[]' " | cut -d, -f1)
    if [ -n "$prev" ] && [ "$prev" != "None" ] && [ "$prev" != "$ver" ]; then
        dt=$(api -X POST "$CONTROL/api/models/$MODEL/rollback" -d "{\"version\": $prev, \"reason\": \"e2e rollback\"}" | jq_ task_id)
        for i in $(seq 1 20); do sleep 3; s=$(api "$CONTROL/api/tasks/$dt" | jq_ status); [ "$s" != "working" ] && break; done
        curl -sf "localhost:8000/v2/models/$MODEL/versions/$prev/ready" >/dev/null && ok "rollback 成功：Triton 回到 v$prev" || { bad "rollback 失敗"; FAIL=$((FAIL+1)); }
    else
        echo "  ⚪ 沒有先前的 production 版本，略過 rollback"
    fi

    # ⑧ 稽核
    $COMPOSE exec -T postgres psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -tc \
        "select count(*) from audit_log where task_id in ('$tid','$dt') or action like 'training.%' and ts > now() - interval '1 hour'" | grep -qE '[1-9]' \
        && ok "audit_log 有紀錄" || { bad "audit_log 沒紀錄"; FAIL=$((FAIL+1)); }
    [ "$FAIL" -eq "$fail0" ] && PASS=$((PASS+1))
    docker run --rm -v "$work:/w" alpine rm -rf /w/train /w/golden >/dev/null 2>&1; rm -rf "$work"
}

case "$MODE" in
    services)    check_services ;;
    unit)        run_unit ;;
    integration) run_integration ;;
    e2e)         check_services && run_e2e ;;
    all)         check_services && { run_unit; run_integration; run_e2e; } ;;
    *) echo "用法: ./test_train.sh [services|unit|integration|e2e|all]"; exit 1 ;;
esac

head_ "總結"
echo "  通過群組: $PASS   失敗群組: $FAIL"
hr
[ $FAIL -eq 0 ]
