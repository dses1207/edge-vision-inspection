# 架構

[English](ARCHITECTURE.md)

## 一、整體流程

系統做三件事，串成一個迴圈：

1. 即時判定：相機畫面進來 → GPU 推論 → 站別規則判 OK / NG → 結果發到事件匯流排。
2. 留證據、給人看：NG 進資料庫、剪出事件前後的影片、推播與通知；人在審核佇列判「真的 / 誤判」。
3. 變得更準：判過的告警回流成標注資料 → 訓練 → 通過 Gate 才登錄 → 移 production tag → Triton 熱載入。

```
  ┌──────────── 即時判定 ────────────┐   ┌───────── 留證據、給人看 ─────────┐
  相機 → go2rtc → runner → Redis Streams → event-processor → Postgres
                    │                  → recorder → MinIO（影片）
                    └→ <cam>_marked    → output-api（WebSocket / 查詢 / 判定）
                                        → notifier
  ┌──────────── 變得更準 ────────────────────────────────────────────┐
  審核結果 → 告警回流 → 圖片庫 / 標注 → lakeFS → 訓練 → Gate → Registry → Triton
```

設定（哪條線、哪些相機、哪個站別、哪顆模型、門檻）是宣告式的：寫進資料庫，
調和器把它變成實際跑著的 runner，runner 回報自己套用到哪一版。

## 二、順著流程講細節

### 1. 影像進來：go2rtc

- 每台相機在 go2rtc 有兩條串流：`<cam>`（拉相機原始畫面）與 `<cam>_marked`（空來源，等 runner 推燒好框的畫面回來）。
- 串流清單由 runner 調和器從相機表產生（`edgevision/orchestrator/go2rtc_sync.py`），手改會被蓋掉，以資料庫為準。
- 瀏覽器看的是 `_marked`：框在伺服器端燒進畫面，前端不需要做任何疊加（理由見 `docs/findings/2026-09-04-overlay-decision.md`）。

### 2. 推論與判定：runner（`ai-infer/`）

一條線一個 runner（K8s 上是一個 Deployment），啟動參數只有 `LINE_ID`：

```
config.py   向設定 API 拿線路設定 → 驗站別存在、現場值齊全、引用的類別名模型真的有
pipeline.py RTSP → nvstreammux → nvinferserver（gRPC 到 Triton）→ nvtracker（選用）→ nvdsosd 燒框 → 推回 go2rtc
probe.py    只搬張量、同步解 bbox 給 nvdsosd 畫；判定丟到 worker 的 queue
worker.py   後處理（edgevision_postprocess）→ 站別 evaluate() → FrameResult
publish.py  FrameResult → Redis Streams 三條：events（只發 NG）、live（即時）、store（落地）
status.py   每 15 秒心跳：套用的設定版本、各相機 fps、GPU 剩餘記憶體
watchdog.py 串流全斷、Triton 掛掉時自我重啟，不靠人
```

- 站別是外掛：每個站別是獨立套件，用 entry point `edgevision.stations` 登記；runner 只認名字（`load_station(name)`）。
  機制（連續幀、冷卻、等待視窗、區域）在 `edgevision_rules`，數字留在站別。
- 一線多站：主站之外可以掛附加站（各自一顆模型、各自的推論間隔），例如主模型判動作、第二顆模型判防護具。
- 串流恢復保護：同一台相機超過幾秒沒幀，恢復後先停判一小段（重連後頭幾幀是殘影）。

demo 用 `apps/cpu_runner` 取代 GStreamer 與 Triton，`worker / publish / status` 是同一份程式。

### 3. 落地與查詢：event-processor、output-api（`edgevision/storage`、`edgevision/api`）

- event-processor 以 consumer group 消費 `edgevision:store`，寫 `detections`。只存有偵測或 NG 的幀；
  時間性 NG（當下沒有框）寫成一列整幀判定。
- output-api：
  - 設定 API（`/api/config/lines`）：寫入時驗證站別參數規格（規格由 runner 映像產生）、類別名對 Registry；
    每次寫入產生版本快照，可還原。讀取時從 Registry 附上模型的類別、imgsz、task。
  - 偵測紀錄、統計、判定回饋；`/ws` 推即時結果；`/healthz` 列出每條線「期望版本 vs 回報版本」。

### 4. 存證影片：recorder（`edgevision/recorder`、`run_recorder.py`）

- 每台相機一個 ffmpeg 持續錄 `<cam>_marked` 成環狀分段（預錄緩衝），NG 事件來了就剪「前 N 秒 + 後 M 秒」上傳 MinIO，
  並把影片位址寫回對應的偵測紀錄。
- 沒有冷卻：每個 NG 一支影片，跟資料庫一對一。去重是判定層的事。
- 保留期：MinIO 生命週期規則刪檔，`run_retention.py` 把資料庫的影片欄位標成失效（紀錄保留）。
- 24 小時連續錄影另有 `run_nas_recorder.py`，寫 NAS，跟事件錄影互不影響。

### 5. 通知：notifier（`edgevision/notify`）

日報與健康巡檢；投遞走 Apprise，通道就是一串 URL（Teams、Email…），加通道不改程式。

### 6. 審核與回流：media（`edgevision/media`）

- 告警佇列：人判「真的 / 誤判」。
- 告警回流：判過的告警 → 從錄影取原始幀 + 當下的框 → 進標注專案的新批次。
- 圖片庫 / 標注 / 審核輪次 / 標注專案：一輪一輪擴充資料集；自動標注（YOLO-World、SAM）先預標，人只改錯的。
- 影片測試：選一段錄影 + 一個模型版本 + 一套站別邏輯 → 產出打框影片與 NG 清單，上線前先試。

### 7. 訓練到上線：control-api、Orchestrator、Registry（`edgevision/training`、`edgevision/orchestrator`、`edgevision/registry`、`edgevision/controller`）

```
POST /api/training/jobs
  → tasks 表 → Orchestrator 依序拉起步驟容器（Docker 或 K8s Job）
     ① trainer        資料（lakeFS 版本）→ 訓練 → 評估 Gate → 匯出 ONNX + 一致性 → 註冊 candidate
     ② trtexec        ONNX → TensorRT plan（用 Triton 映像自帶的 trtexec，避開版本不合）
     ③ parity-triton  臨時 Triton 只載這一版
     ④ finalize       TensorRT 輸出 vs ONNX 一致性 → 上傳 plan → 完成
  → promote：Registry 把 production tag 移到這一版 → Deployer 讓 Triton 熱載入
```

- 每步冪等、狀態進 `tasks`，重啟後從沒完成的那步續跑。
- Gate 在 `configs/gates/<model>.yaml`：絕對門檻、關鍵類別 recall、延遲上限、黃金測試集不得新漏。
- Registry 是「哪一版在線上」的唯一真相；程式和設定檔都不寫版本號。

### 8. 讓 agent 也能操作：MCP（`edgevision/mcp`）

API 的每個端點都有對應的 MCP 工具（`tests/test_mcp.py` 比對，漏了就擋），stdio 版給本機、遠端版走 OAuth。

### 9. 部署：K8s（`deploy/k8s`）

- kustomize：`base/` 是所有服務，`overlays/<環境>` 換網域、映像、GPU 選擇器。
- GPU 用 DRA（ResourceClaim）：推論、訓練各自一個 claim，可以切 MIG。
- 多站點推論：某些線的 runner 排到別的節點、連那個節點的 Triton（`inference-sites.yaml`），Triton 直接從 MinIO 讀模型庫。
- runner 調和器（`run_runner_reconciler.py`）：每 30 秒把資料庫的線路設定對到 Deployment；設定版本或模型版本變了就換 Pod。
