#!/bin/sh
# 啟動前把範本中的環境變數展開成實際設定。
#
# go2rtc 不支援設定檔內的 ${VAR} 展開（實測確認），故在此預處理。
# 用 sed 而非 envsubst —— go2rtc 的映像是精簡版，沒有 gettext 套件。
set -e

TEMPLATE=/config/go2rtc.template.yaml
STREAMS=/config/streams.generated.yaml
OUTPUT=/tmp/go2rtc.yaml

[ -f "$TEMPLATE" ] || { echo "找不到範本：$TEMPLATE" >&2; exit 1; }

# ⚠️ 相機清單是必要的，缺了不能悄悄跑起來。
#
# 實際踩過：範本裡寫死 cam156，DB 裡那條線卻有好幾台。go2rtc 只認得 cam156，
# runner 照 DB 去連 cam193 起的幾台全部落空，每 10 秒重試一次無限重連
# （attempts 累積到 410 次），而且沒有任何一行 log 指出「這條串流不存在」。
# 更糟的是 streams.generated.yaml 早就產好躺在旁邊，只是從沒有人讀它。
#
# 所以這裡寧可啟動失敗也不要少了相機 —— 沒有相機的 go2rtc 一點用都沒有，
# 讓它起來只會製造上面那種查不出原因的無限重連。
if [ ! -f "$STREAMS" ]; then
    echo "找不到相機清單：$STREAMS" >&2
    echo "它是 cameras 表的投影，用下列指令產生：" >&2
    echo "  python3 ai-infer/tools/gen_go2rtc_streams.py --api http://localhost:18080 \\" >&2
    echo "      -o deploy/go2rtc/streams.generated.yaml" >&2
    exit 1
fi

# 只替換明確列出的變數，避免誤傷設定檔中其他的 $ 符號
sed \
  -e "s|\${CAM_USER}|${CAM_USER}|g" \
  -e "s|\${CAM_PASS}|${CAM_PASS}|g" \
  -e "s|\${WEBRTC_IP}|${WEBRTC_IP}|g" \
  -e "s|\${GO2RTC_BASE_PATH}|${GO2RTC_BASE_PATH:-}|g" \
  "$TEMPLATE" > "$OUTPUT"

# 相機清單接在範本的 streams: 之後。
# 它已經帶著明碼帳密（產生時就從環境變數解析好了），不需要再 sed。
cat "$STREAMS" >> "$OUTPUT"

# 驗證：確認沒有未展開的變數殘留。
# 只檢查非註解行 —— 註解裡提到 ${VAR} 是說明文字，不是待展開的變數。
LEFTOVER=$(grep -v '^\s*#' "$OUTPUT" | grep -o '\${[A-Za-z_][A-Za-z0-9_]*}' | sort -u || true)
if [ -n "$LEFTOVER" ]; then
    echo "⚠️ 設定中仍有未展開的變數：" >&2
    echo "$LEFTOVER" >&2
    exit 1
fi

echo "設定已產生：$(grep -cE '^  cam[0-9]+:' "$OUTPUT") 台相機（帳密來自環境變數）"
exec go2rtc -config "$OUTPUT"
