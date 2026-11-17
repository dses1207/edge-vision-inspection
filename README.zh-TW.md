# EdgeVision

[English](README.md)

EdgeVision 是攝影機視覺檢測系統的後端。它負責幾十路 RTSP 相機的 GPU 推論，依每條產線的規則判定，
每個告警都錄一段存證影片，審核過的告警再回到標注與訓練。新模型要通過驗收 Gate 才會上線，
上線後有問題可以從模型 Registry 直接退回上一版。

這個 repo 只有平台本身，原本服務的產線判定規則和訓練好的模型都不在這裡。
取而代之的是三個通用示範站別（禁區闖入、觸發後限時確認、防護具檢查）和一台合成相機，
整條流程在筆電上就能跑，不需要 GPU。

```
相機 -> go2rtc -> runner（DeepStream + Triton）-> 站別判定 -> Redis Streams
                    | 框燒進 <cam>_marked                    |
                    v                                        +-> event-processor -> Postgres
                go2rtc -> WebRTC 給瀏覽器                    +-> recorder -> MinIO（NG 前後影片）
                                                             +-> notifier（Apprise）
                                                             +-> output-api（WebSocket / REST）

審核 -> 告警回流 -> 圖片庫 / 標注 -> lakeFS 資料集
  ^                                      |
  +-- Triton 熱載入 <- Registry 移 tag <- Gate <- 訓練任務（Orchestrator：Docker / K8s）
```

這張圖的每一步在 [docs/ARCHITECTURE.zh-TW.md](docs/ARCHITECTURE.zh-TW.md) 有詳細說明。

## 執行 demo

需要 Docker，大約五分鐘，只用 CPU。

```bash
cd demo
cp .env.example .env
docker compose up -d --build
```

| 網址 | 內容 |
|---|---|
| http://localhost:1984/stream.html?src=cam-01_marked | 已燒框的直播，畫面上有各站別這一幀的判定 |
| http://localhost:1984/stream.html?src=cam-02_marked | 卸貨區：堆高機進場後 10 秒內要看到引導員 |
| http://localhost:8080/docs | output-api：線路設定、偵測紀錄、統計、審核 |
| http://localhost:8080/healthz | 每條線的設定版本（期望與回報）、心跳、fps |
| http://localhost:9001 | MinIO console，存證影片在 `inference-output/` 底下 |

demo 跑的 worker、站別、發布、心跳、錄影、事件落地和 API 都跟正式環境是同一套，
只換掉影像來源和模型。合成場景負責畫出畫面，「場景偵測器」把場景的真值轉成 YOLO 模型的原始輸出張量，
所以後處理、NMS、letterbox 還原都是真的在跑。詳見 [`demo/`](demo/) 和
[`ai-infer/apps/cpu_runner`](ai-infer/apps/cpu_runner/src/cpu_runner/main.py)。

## Repo 結構

| 路徑 | 內容 |
|---|---|
| [`ai-infer/`](ai-infer/) | 推論 monorepo（uv workspace）。`libs/` 是共用的資料契約、後處理、規則元件與站別 SDK；`stations/` 每個判定規則一個套件；`apps/deepstream_runner` 是正式環境的 runner，`apps/cpu_runner` 是 demo 用的 runner |
| [`edgevision/api`](edgevision/api) | output-api（有版本與驗證的線路設定、偵測紀錄、即時 WebSocket）與 control-api（訓練、資料集、模型、任務） |
| [`edgevision/orchestrator`](edgevision/orchestrator) | 任務編排（在 Docker 或 K8s 上跑各步驟容器）、runner reconciler（資料庫裡的線路轉成 runner Deployment）、go2rtc 串流同步 |
| [`edgevision/training`](edgevision/training) | 匯入、訓練、Gate 評估、匯出 ONNX、TensorRT 一致性檢查、註冊，每一步都是可重跑的 CLI |
| [`edgevision/registry`](edgevision/registry)、[`controller`](edgevision/controller) | 模型 Registry（production tag 就是線上版本）與 Triton 部署器 |
| [`edgevision/media`](edgevision/media) | 圖片庫、標注、審核輪次、迭代標注、自動標注、告警回流、影片測試 |
| [`edgevision/recorder`](edgevision/recorder)、`run_nas_recorder.py` | 從預錄環形緩衝切出事件影片；24 小時連續錄影到 NAS |
| [`edgevision/notify`](edgevision/notify) | 每日報表與健康巡檢，透過 Apprise 發送 |
| [`edgevision/mcp`](edgevision/mcp) | MCP server（stdio 與遠端 OAuth）。每個 API 端點同時是 agent 工具，由一致性測試確保兩邊同步 |
| [`edgevision/ptz`](edgevision/ptz) | 透過 ONVIF 控制 PTZ、預置位、偏移偵測 |
| [`deploy/docker-compose.yml`](deploy/docker-compose.yml)、[`test_train.sh`](test_train.sh) | 單台 GPU 主機的完整堆疊，含訓練（Triton、ClearML、lakeFS、trainer），以及訓練鏈路的端到端測試：資料進 lakeFS、訓練、Gate、TensorRT、Registry、promote、rollback |
| [`deploy/k8s`](deploy/k8s) | kustomize base 與範例 overlay：DRA GPU 配發、多廠區推論、Triton 從 S3 讀模型庫 |
| [`deploy/postgres/init`](deploy/postgres/init) | Schema 與可重跑的 migration |
| [`docs/`](docs/) | 架構、設計筆記、有日期的實測紀錄 |

## 設計取捨

框在伺服器端燒進影像，不在瀏覽器上疊。我試過用 RTP 時間戳讓 WebRTC 串流和另一條結果通道逐幀對齊，
實測做不到（[實測紀錄](docs/findings/2026-08-28-sync-verification.md)、[決策](docs/findings/2026-09-04-overlay-decision.md)）。

一條產線對應一個 runner、一個設定版本。線路和相機存在 Postgres，由 reconciler 轉成 Deployment。
runner 在心跳裡回報實際套用的設定版本，所以 `/healthz` 能並排看到期望版本和回報版本
（[`server.py`](edgevision/api/server.py)）。

站別是外掛，完全不知道 runner 的存在。一個站別就是一個小 class，提供 `evaluate`、`reset`、
`update_params`、`classes_used`，透過 Python entry point 註冊，runner 只認得它的名字。
持續計數、冷卻、等待窗、區域這些共用機制放在 `edgevision_rules`，門檻數字放在站別裡。

類別名稱以模型為準。線路設定不存類別清單，讀取時才從 Registry 帶入；
runner 啟動前會檢查站別用到的每個類別名稱是否真的存在於部署中的模型。

NG 是判定結果，不是一個框。「A 發生後 10 秒內沒看到 B」這類時間規則，可能在完全沒有框的幀上觸發，
所以發布、儲存和存證錄影都以判定結果為準。

去重只在判定層做一次。recorder 收到 NG 就錄，不另外設冷卻，
因此「資料庫有 NG 但沒有影片」一定是 bug。

更多內容見 [docs/DESIGN-NOTES.zh-TW.md](docs/DESIGN-NOTES.zh-TW.md) 與 [docs/ARCHITECTURE.zh-TW.md](docs/ARCHITECTURE.zh-TW.md)。

## 開發

```bash
# 推論 workspace（uv）
cd ai-infer && uv sync --all-packages && uv run pytest -q && uv run ruff check .

# 平台服務
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest -q tests --ignore=tests/integration
```

要寫新站別，可以複製 [`ai-infer/stations/_template`](ai-infer/stations/_template)，或從示範站別改起：
[`zone_intrusion`](ai-infer/stations/zone_intrusion)、[`followup_window`](ai-infer/stations/followup_window)、
[`ppe_check`](ai-infer/stations/ppe_check)。

## 第三方軟體

訓練與自動標注步驟會在 trainer 映像裡於執行期呼叫 [Ultralytics](https://github.com/ultralytics/ultralytics)
（AGPL-3.0）。repo 沒有內含它，服務本身和 demo 也不需要它。
