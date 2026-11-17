-- 標注專案 v0.2：資料來源統一，專案只引用。
--   initial_batch_ids  圖片集裡哪幾個批次是初始人工標注
--   initial_model      初始模型 {name, version}（Registry）
--   recipe             混合配方覆蓋 {"batches": {"<batch_id>": {"repeat"|"choose"|"proportion": …}}, "total", "seed"}
--   batch_snapshots    每個批次最新的快照 {"<batch_id>": {ref, sig, task_id, iteration}}
-- 階段改成 prelabeling → reviewing → exporting → mixing → training → evaluating；
-- 拿掉 seed_select / seed_labeling / sourcing（pending_source、golden_sync 欄位留著不用）。
ALTER TABLE media_campaigns ADD COLUMN IF NOT EXISTS initial_batch_ids JSONB;
ALTER TABLE media_campaigns ADD COLUMN IF NOT EXISTS initial_model JSONB;
ALTER TABLE media_campaigns ADD COLUMN IF NOT EXISTS recipe JSONB;
ALTER TABLE media_campaigns ADD COLUMN IF NOT EXISTS batch_snapshots JSONB;

-- 每一代：用到的批次快照、混合配方（已展開成 sources）、混合任務；dataset_ref = 訓練快照
ALTER TABLE media_campaign_iterations ADD COLUMN IF NOT EXISTS batch_snapshots JSONB;
ALTER TABLE media_campaign_iterations ADD COLUMN IF NOT EXISTS recipe JSONB;
ALTER TABLE media_campaign_iterations ADD COLUMN IF NOT EXISTS mix_task_id TEXT;
