#!/bin/bash
# 套用 EdgeVision 到 K8s（kustomize overlay + 映像標籤）。
#
#   KUBECONFIG=~/.kube/edgevision.kubeconfig \
#   EDGEVISION_TAG=<EdgeVision git sha> TRAIN_TAG=<sha> bash deploy/k8s/apply.sh test [--dry-run]
#   ⚠️ TRAIN_TAG 要明確給：訓練映像不是每個 commit 都有（查 registry 的 edgevision-train tags）
#
# 前置（第一次）：edgevision-secrets（make_secrets.py）與 go2rtc-streams（README）要先在叢集上。
set -euo pipefail
cd "$(dirname "$0")"
ENV=${1:?用法：apply.sh <overlay> [--dry-run]}
DRY=${2:-}
: "${EDGEVISION_TAG:?要設 EDGEVISION_TAG（edgevision-service 映像標籤）}"
TRAIN_TAG=${TRAIN_TAG:-$EDGEVISION_TAG}
NS=edgevision

for s in edgevision-secrets go2rtc-streams; do
  kubectl -n $NS get secret $s >/dev/null 2>&1 || { echo "缺 Secret $s（第一次部署要先建：make_secrets.py、go2rtc 清單）" >&2; [ -n "$DRY" ] || exit 1; }
done

out=$(kubectl kustomize --load-restrictor LoadRestrictionsNone "overlays/$ENV" \
      | sed -e "s/EDGEVISION_TAG/$EDGEVISION_TAG/g" -e "s/TRAIN_TAG/$TRAIN_TAG/g")
if grep -q "GPU-SET-IN-OVERLAY" <<<"$out"; then
  echo "overlay $ENV 沒有設定推論卡 UUID（ResourceClaim inference-gpu）" >&2; exit 1
fi
# 映像要真的存在才套用：TRAIN_TAG 沒給會預設成 EDGEVISION_TAG，但訓練映像只在訓練程式變動時才重建，
# 正式叢集曾因此指向不存在的 edgevision-train:<sha>，一直到第一次送訓練才發現（2026-10-12）。
if [ -z "${SKIP_IMAGE_CHECK:-}" ]; then
  missing=0
  # 私有 registry → 用 docker 已登入的帳號查（CI 要先登入 registry）。dockerhub 代理的不查。
  for img in $(grep -oE 'ghcr\.io/your-org/edgevision/[a-z0-9._/-]+:[A-Za-z0-9._-]+' <<<"$out" | sort -u); do
    docker manifest inspect "$img" >/dev/null 2>&1 || { echo "映像不存在或查不到（沒登入 registry？）：$img" >&2; missing=1; }
  done
  [ $missing = 0 ] || { echo "中止：先確認標籤（訓練映像請明確給 TRAIN_TAG）；確定要略過檢查設 SKIP_IMAGE_CHECK=1" >&2; exit 1; }
fi
# -n $NS：沒寫 namespace 的資源進 $NS，寫了別的 namespace 的 kubectl 直接拒絕。
# ⚠️ 不能省：CI 的 kubeconfig 預設 namespace 是 default，曾經每次部署都把 overlay 加的資源
#    套進 default（Ingress 撞名，edgevision-minio 回 503）。
if [ "$DRY" = "--dry-run" ]; then
  kubectl -n $NS apply --dry-run=server -f - <<<"$out"
else
  kubectl -n $NS apply -f - <<<"$out"
fi
