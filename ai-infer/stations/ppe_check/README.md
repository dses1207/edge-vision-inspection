# ppe_check（示範站）

吃「第二個模型」的偵測，示範一條線掛主站 + 附加站（線路設定 `extra_stations`）。
單幀很吵的開放詞彙模型，靠滑動視窗的時間一致性確認，再依原因冷卻。

| 現場值 | 說明 |
|---|---|
| `thresholds.violations` | 類別 → 通報原因，預設 `{"no_helmet": "未戴安全帽", "no_vest": "未穿反光背心"}` |
| `thresholds.min_conf` / `default_conf` | 各類別信心門檻 / 預設 0.3 |
| `thresholds.window_size` / `min_count` | 視窗 10 次、≥5 次才確認 |
| `thresholds.iou_threshold` | 視窗內「同一個東西」的 IoU 門檻，預設 0.2 |
| `thresholds.report_cooldown_seconds` | 同原因冷卻，預設 60 |
