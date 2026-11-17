#!/bin/bash
# 既有資料庫的結構變更：每次部署前把 deploy/postgres/init/*.sql 全部依序再跑一遍。
#
# ⭐ 為什麼可以每次全跑：init 目錄的每個檔都必須【可重複執行】（CREATE ... IF NOT EXISTS、
#    ALTER TABLE ... ADD COLUMN IF NOT EXISTS），已經有的跳過、新加的補上，不需要另外記錄跑過哪些。
#    tests/test_db_init_idempotent.py 會擋下不可重複執行的寫法。
# ⭐ 為什麼要有：postgres 只在資料庫【第一次建立】時執行 init 目錄。之後新增的檔（例如加欄位）
#    不會自己生效，以前要手動 kubectl exec 跑 ALTER TABLE。
# 順序：要在換服務映像【之前】跑 —— 新程式會用到新欄位；加欄位對舊程式無害。
#
#   KUBECONFIG=… bash deploy/ci/migrate_db.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
NS=${NS:-edgevision}

if ! kubectl -n "$NS" get pod postgres-0 >/dev/null 2>&1; then
  echo "還沒有 postgres（第一次部署）→ 略過；資料庫建立時 postgres 會自己跑 init 目錄"
  exit 0
fi
kubectl -n "$NS" wait --for=condition=Ready pod/postgres-0 --timeout=300s >/dev/null
for f in deploy/postgres/init/*.sql; do
  if ! out=$(kubectl -n "$NS" exec -i postgres-0 -- \
        sh -c 'psql -q -v ON_ERROR_STOP=1 -U "$POSTGRES_USER" -d "$POSTGRES_DB"' < "$f" 2>&1); then
    echo "$out" >&2; echo "❌ $f 執行失敗，停止部署" >&2; exit 1
  fi
  grep -v NOTICE <<<"$out" || true
  echo "✓ $(basename "$f")"
done
