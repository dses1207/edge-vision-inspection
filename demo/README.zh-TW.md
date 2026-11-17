# demo：不需要 GPU 的整條鏈路

[English](README.md)

```bash
cp .env.example .env          # port 撞到就改 .env 最下面那幾行
docker compose up -d --build
docker compose logs -f runner-demo-zone    # 看 NG 一筆一筆出來
```

## 裡面有什麼

| 服務 | 是什麼 | 正式環境對應 |
|---|---|---|
| `runner-demo-zone` / `runner-demo-dock` | cpu-runner，一條線一個 | deepstream_runner（GPU），由 runner 調和器建立 |
| `go2rtc` | 收 runner 推來的 `<cam>` 與 `<cam>_marked` | 同（原始串流改為拉相機） |
| `output-api` | 線路設定 API、偵測查詢、WebSocket、`/healthz` | 同 |
| `event-processor` | `edgevision:store` → Postgres | 同 |
| `recorder` | 預錄緩衝 + NG 片段 → MinIO，影片位址寫回偵測紀錄 | 同 |
| `postgres` / `redis` / `minio` | 資料庫、事件匯流排、物件儲存 | 同 |

沒有起的：Triton、control-api（訓練 / Registry / 資料集）、ClearML、lakeFS、notifier、PTZ。
模型類別資訊由 `initdb/99-demo-seed.sql` 預填進 `model_info_cache`（設定 API 連不到 Registry 時用的快取），
所以 output-api 的 log 會一直出現「查 Registry 失敗」，這是預期的。

## 兩條示範線

### `demo_zone`（cam-01，倉庫走道）
- 主站 `zone_intrusion`：人的腳點進入紅色禁區、連續 3 幀 → NG，冷卻 15 秒。
- 附加站 `ppe_check`：第二顆模型（`ppe`），沒戴安全帽在 10 次判定內出現 ≥5 次 → NG，同原因冷卻 30 秒。
  `helmet / vest / no_vest` 設成 `hidden_classes`：照樣推論，只是不畫。

### `demo_dock`（cam-02，卸貨區）
- `followup_window`：堆高機進場後 10 秒內要看到引導員。劇本裡每三次有一次引導員缺席 → 逾時 NG。
  這種 NG 當下畫面上沒有任何框，影片與紀錄照樣對得上。

## 改設定看它生效

```bash
# 把禁區改成左半邊：設定 API 存檔 → 版本 +1 → runner 偵測到版本變了就重啟套用
curl -s localhost:8080/api/config/lines/demo_zone | python3 -m json.tool | head -40
```

`/healthz` 的 `desired_version` / `reported_version` 會先不一致，runner 重啟套用後再對齊。

## 合成場景怎麼做的

`ai-infer/apps/cpu_runner/src/cpu_runner/scene.py` 用時間 t 決定畫面上有什麼，同時產生畫面與真值框；
`detector.py` 把真值框轉成 YOLO detect 模型的原始輸出（letterbox 座標 + 類別分數），加一點抖動、漏檢與信心浮動，
讓後面的去抖、滑動視窗、冷卻有東西可擋。劇本是決定性的，每次啟動事件時間點都一樣。
