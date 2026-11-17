-- ════════════════════════════════════════════════════════
-- line_status.stats：runner 心跳附帶的推論統計（2026-09-29）
-- ════════════════════════════════════════════════════════
--
-- runner（ai-infer ds_runner/status.py）每 15 秒心跳都帶 stats，但以前 StatusIn 沒宣告這個欄位，
-- 一直被默默丟掉。現在存下來，/healthz 與健康巡檢用它做【相機層級】判斷：
--   stats.cameras = {"cam156": 14.9, "ipcam235": 0.0}   每台相機這一段的 fps；0.0 = 心跳正常但這台沒幀
--   stats.fps / dropped / sync_ms_avg / worker_queue  整條線的推論負載
-- 舊版 runner 沒有 stats.cameras → 那些相機顯示 unknown，不告警。
--
-- 可重複執行；既有資料庫手動跑一次：
--   docker exec -i edgevision-postgres psql -U edgevision -d edgevision < deploy/postgres/init/07-line-status-stats.sql

ALTER TABLE line_status ADD COLUMN IF NOT EXISTS stats JSONB;
