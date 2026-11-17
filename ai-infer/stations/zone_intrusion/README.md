# zone_intrusion（示範站）

指定類別（預設 `person`）的腳點進入禁區多邊形、連續 `persist_frames` 幀 → NG，之後冷卻 `cooldown_seconds` 秒。

| 現場值 | 必填 | 說明 |
|---|---|---|
| `zone` | 是 | 禁區多邊形，正規化座標 `[[x, y], ...]` |
| `thresholds.target_classes` | | 預設 `["person"]` |
| `thresholds.min_conf` | | 預設 0.4 |
| `thresholds.persist_frames` | | 預設 3（去抖） |
| `thresholds.cooldown_seconds` | | 預設 10 |
| `roi_regions` | | 通用 ROI（detect / mask 區域） |
