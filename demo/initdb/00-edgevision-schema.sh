#!/bin/sh
# 套用平台的正式 schema（deploy/postgres/init，跟 K8s 用的是同一份），再由 99-demo-seed.sql 灌示範資料。
set -e
for f in /edgevision-init/*.sql; do
  echo "→ $f"
  psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" -f "$f"
done
