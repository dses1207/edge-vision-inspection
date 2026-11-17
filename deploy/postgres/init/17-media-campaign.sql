-- 標注專案
--
-- 一個專案 = 一個任務從 0 到上線：挑種子 → 種子標注 → 訓練 → 評估 → 預標 → 審核 → 訓練 …
-- 狀態全在這裡：control-api 的調和迴圈每隔幾秒看一次，重啟後從原地接著跑。
CREATE TABLE IF NOT EXISTS media_campaigns (
    id              BIGSERIAL   PRIMARY KEY,
    name            TEXT        NOT NULL UNIQUE,
    collection_id   BIGINT      NOT NULL REFERENCES media_collections(id) ON DELETE RESTRICT,
    model_name      TEXT        NOT NULL,       -- 每一代都註冊成這個模型的新版本（candidate）
    base_model      TEXT        NOT NULL,       -- 每一代都從這個權重重新訓練（例 s3://models/_base/yolo26m.pt）
    golden_ref      TEXT,                       -- 測試集：任一份 LakeFS 資料集版本（可換、可空）
    config          JSONB       NOT NULL DEFAULT '{}'::jsonb,   -- 只存覆蓋；預設在 edgevision/media/campaign_defaults.yaml
    stage           TEXT        NOT NULL,       -- seed_select | seed_labeling | training | evaluating | prelabeling | reviewing | sourcing
    state           TEXT        NOT NULL DEFAULT 'running'
                    CHECK (state IN ('running', 'waiting', 'failed', 'paused')),
    waiting_reason  TEXT,                       -- labeling | confirm | regression | sourcing（state=waiting 時）
    waiting_detail  JSONB,
    step            JSONB       NOT NULL DEFAULT '{}'::jsonb,   -- 這個階段進行到哪（例 {"export_task": …, "train_task": …}）
    iteration       INTEGER     NOT NULL DEFAULT 0,             -- 第幾代（0 = 種子）
    round_id        BIGINT,                     -- 目前在審的輪次
    pending_source  JSONB,                      -- 取料：人指定的相機 / 時段 / 挑圖參數（用完清空）
    last_error      TEXT,
    created_by      TEXT        NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- 每一代：資料集版本 → 訓練 → 模型版本 → 評估（記用了哪一版測試集，退步只在同一版之間比）→ 預標 → 審核輪次
CREATE TABLE IF NOT EXISTS media_campaign_iterations (
    campaign_id     BIGINT      NOT NULL REFERENCES media_campaigns(id) ON DELETE CASCADE,
    n               INTEGER     NOT NULL,
    dataset_ref     TEXT,
    export_task_id  TEXT,
    train_task_id   TEXT,
    model_version   INTEGER,
    metrics         JSONB,
    golden_ref      TEXT,
    regression      JSONB,                      -- 跟上一代比的結果 {"regressed": [...], "compared_to": n}
    prelabel_task_id TEXT,
    round_id        BIGINT,
    note            TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (campaign_id, n)
);

-- 事件：階段轉換、誰按了繼續、失敗原因（稽核另寫 audit_log）
CREATE TABLE IF NOT EXISTS media_campaign_events (
    id              BIGSERIAL   PRIMARY KEY,
    campaign_id     BIGINT      NOT NULL REFERENCES media_campaigns(id) ON DELETE CASCADE,
    at              TIMESTAMPTZ NOT NULL DEFAULT now(),
    actor           TEXT        NOT NULL,
    kind            TEXT        NOT NULL,
    stage           TEXT,
    detail          JSONB       NOT NULL DEFAULT '{}'::jsonb
);
CREATE INDEX IF NOT EXISTS idx_media_campaign_events ON media_campaign_events (campaign_id, id DESC);

-- 種子輪：整輪不給預標、全部從頭標
ALTER TABLE media_review_rounds ADD COLUMN IF NOT EXISTS mode TEXT NOT NULL DEFAULT 'review';
