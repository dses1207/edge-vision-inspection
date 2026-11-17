-- 標注專案 v0.3：專案只管「照配方訓練一輪＋用新模型預標」，人按才跑。
--   state  idle（閒置）/ running / failed / closed；拿掉 waiting、paused（不再自己推進、不再等人審、不再判退步）
--   stage  running / failed 時在哪一步（exporting / mixing / training / prelabeling）；閒置 = NULL
--   每一代可以預標很多次（加一批就預標一次）→ prelabel_task_ids 記全部；prelabel_task_id 留著 = 最近一次
-- waiting_reason、waiting_detail、round_id、pending_source、golden_sync 欄位留著不用；iterations.regression 只剩歷史。
-- 每次部署都會重跑：每一句都只動舊版留下來的資料，第二次跑什麼都不會改。
ALTER TABLE media_campaigns ALTER COLUMN stage DROP NOT NULL;
ALTER TABLE media_campaign_iterations ADD COLUMN IF NOT EXISTS prelabel_task_ids JSONB;
ALTER TABLE media_campaigns DROP CONSTRAINT IF EXISTS media_campaigns_state_check;

-- 舊版在「評估」停下的：那一代已經訓練完（有模型）→ 算做完，下一輪是下一代
UPDATE media_campaigns c SET iteration = c.iteration + 1
 WHERE c.stage = 'evaluating' AND c.state <> 'closed'
   AND EXISTS (SELECT 1 FROM media_campaign_iterations i
                WHERE i.campaign_id = c.id AND i.n = c.iteration AND i.model_version IS NOT NULL);
-- 舊版在等人（審核 / 按繼續 / 退步 / 新批次）、暫停、或停在已不存在的步驟 → 閒置
UPDATE media_campaigns SET state = 'idle', stage = NULL, step = '{}'::jsonb,
       waiting_reason = NULL, waiting_detail = NULL, round_id = NULL
 WHERE state <> 'closed' AND (state IN ('waiting', 'paused') OR stage IN ('reviewing', 'evaluating'));
UPDATE media_campaigns SET stage = NULL, step = '{}'::jsonb, waiting_reason = NULL, waiting_detail = NULL
 WHERE state = 'closed' AND (stage IS NOT NULL OR waiting_detail IS NOT NULL);
-- 拿掉的設定欄位（設定驗證是嚴格的，留著一讀就報錯）；prelabel.conf 舊版 0 = 自動，現在要 > 0
UPDATE media_campaigns SET config = config - 'round' - 'transitions' - 'regression' - 'approvers'
 WHERE config ?| array['round', 'transitions', 'regression', 'approvers'];
UPDATE media_campaigns SET config = config #- '{prelabel,conf}'
 WHERE (config #>> '{prelabel,conf}')::numeric <= 0;
UPDATE media_campaign_iterations SET prelabel_task_ids = jsonb_build_array(prelabel_task_id)
 WHERE prelabel_task_ids IS NULL AND prelabel_task_id IS NOT NULL;

ALTER TABLE media_campaigns ADD CONSTRAINT media_campaigns_state_check
    CHECK (state IN ('idle', 'running', 'failed', 'closed'));
