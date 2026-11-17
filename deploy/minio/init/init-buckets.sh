#!/bin/sh
# 建立 bucket 結構。
#
# 一次建好【所有階段】會用到的 bucket —— 基礎建設不該分次做，
# 否則之後每加一個功能就要動基礎設施。
#

set -e

MC="mc --config-dir /tmp/.mc"
$MC alias set edgevision "http://minio:9000" "$MINIO_ROOT_USER" "$MINIO_ROOT_PASSWORD" >/dev/null

echo "建立 bucket 結構…"

# ── 資料層 ──────────────────────────────────────────────
# raw：原始影像。⚠️ 只寫不改，這是可重現性的基礎
$MC mb -p edgevision/raw           2>/dev/null || true
# curated：篩選過、待標註的候選
$MC mb -p edgevision/curated       2>/dev/null || true
# datasets：版本化的訓練資料集（第二階段由 LakeFS 管理）
$MC mb -p edgevision/datasets      2>/dev/null || true

# ── 模型層 ──────────────────────────────────────────────
# models：模型 artifact。集中放 MinIO，不放共用資料夾
$MC mb -p edgevision/models        2>/dev/null || true

# ── 推論產出 ────────────────────────────────────────────
# inference-output：缺陷圖、影片片段
$MC mb -p edgevision/inference-output 2>/dev/null || true

# ── 實驗產出（第二階段）──────────────────────────────────
$MC mb -p edgevision/experiments   2>/dev/null || true

# ── 備份目標 ────────────────────────────────────────────
$MC mb -p edgevision/backups       2>/dev/null || true

echo ""
echo "設定保留政策…"
# 事件影片保留 CLIP_RETENTION_DAYS 天（預設 7，2026-10-05 決定，同舊系統）。
# 資料庫的 NG 紀錄保留更久，影片過期時由 edgevision-retention CronJob 把連結標成失效（run_retention.py）。
# ⚠️ 先清掉舊規則再加：ilm rule add 每跑一次就多一條（測試叢集曾疊出兩條 90 天）。
DAYS=${CLIP_RETENTION_DAYS:-7}
$MC ilm rule rm --all --force edgevision/inference-output >/dev/null 2>&1 || true
$MC ilm rule add --expire-days "$DAYS" edgevision/inference-output 2>/dev/null \
  && echo "  inference-output: $DAYS 天後過期" \
  || echo "  inference-output: 保留政策設定略過"

echo ""
echo "目前 bucket："
$MC ls edgevision

echo ""
echo "✅ MinIO 初始化完成"
