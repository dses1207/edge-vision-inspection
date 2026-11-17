# followup_window（示範站）

看到 `trigger_class` 後 `seconds` 秒內必須看到 `required_class`，逾時 → NG。示範「時間性 NG」：NG 那一幀可能沒有任何框。

| 現場值 | 必填 | 說明 |
|---|---|---|
| `trigger_class` | 是 | 觸發計時的類別 |
| `required_class` | 是 | 計時內必須出現的類別 |
| `thresholds.seconds` | | 預設 10 |
| `thresholds.late_ok` | | 逾時後才出現算不算滿足，預設 false |
| `thresholds.min_conf` | | 預設 0.4 |
| `thresholds.rearm_gap_seconds` | | 觸發類別要消失多久才算下一次進場，預設 3 |
| `thresholds.cooldown_seconds` | | 預設 0（不冷卻） |
