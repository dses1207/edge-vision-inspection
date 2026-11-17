#!/usr/bin/env bash
# 算「改了什麼，要測什麼」。輸出要跑 pytest 的路徑清單（空白分隔）；all 表示全部。
#
#   libs/**            → all（共用書改了，每本手冊重校）
#   stations/<X>/**    → stations/<X>
#   apps/<A>/**        → apps/<A>
#   tests/** 或根設定  → all
#   其他（docs、README）→ 空（只 lint）
#
# 用法：tools/ci/affected.sh <base-ref>     例：tools/ci/affected.sh origin/main
set -euo pipefail
base="${1:-origin/main}"
changed=$(git diff --name-only "$base"...HEAD -- . 2>/dev/null || git diff --name-only "$base" -- .)
targets=()
for f in $changed; do
  case "$f" in
    libs/*|tests/*|pyproject.toml|uv.lock|tools/ci/*) echo all; exit 0 ;;
    stations/*/*) targets+=("$(echo "$f" | cut -d/ -f1-2)") ;;
    apps/*/*)     targets+=("$(echo "$f" | cut -d/ -f1-2)") ;;
  esac
done
printf '%s\n' "${targets[@]:-}" | sort -u | tr '\n' ' '
