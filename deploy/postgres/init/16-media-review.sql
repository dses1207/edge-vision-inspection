-- 標注審核：輪次、輪次裡的圖、圖片集的審核設定覆蓋
--
-- 審核設定：平常全用預設（edgevision/media/review_defaults.yaml），這欄只存「這個圖片集要覆蓋的欄位」。
ALTER TABLE media_collections ADD COLUMN IF NOT EXISTS review_config JSONB NOT NULL DEFAULT '{}'::jsonb;

-- 縮圖（審核縮圖牆）：{"480": "library/thumbs/ab/<sha>_480.webp"}，依長邊像素分
ALTER TABLE media_blobs ADD COLUMN IF NOT EXISTS thumbs JSONB NOT NULL DEFAULT '{}'::jsonb;

-- 一輪審核：挑最該看的 N 張；config 是建立當下合併後的完整設定（之後設定再改，舊輪次仍知道當時參數）
CREATE TABLE IF NOT EXISTS media_review_rounds (
    id              BIGSERIAL   PRIMARY KEY,
    collection_id   BIGINT      NOT NULL REFERENCES media_collections(id) ON DELETE CASCADE,
    name            TEXT        NOT NULL,
    status          TEXT        NOT NULL DEFAULT 'open' CHECK (status IN ('open', 'closed')),
    config          JSONB       NOT NULL,
    model           JSONB,                      -- 預標用的模型 {"name", "version"}（從框的 source 推得）
    composition     JSONB       NOT NULL DEFAULT '{}'::jsonb,  -- 建立時各原因幾張、抽查幾張
    created_by      TEXT        NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    closed_by       TEXT,
    closed_at       TIMESTAMPTZ
);
CREATE INDEX IF NOT EXISTS idx_media_rounds_coll ON media_review_rounds (collection_id, id DESC);

-- 輪次裡的一張圖。baseline = 建立輪次當下的框（含 id）：送出時拿來比「原樣 / 修改 / 新增 / 刪除」；
-- 抽查圖（blind）的 baseline 是模型的預測，圖本身的框在建立輪次時清空（前端、API 都拿不到預標）。
CREATE TABLE IF NOT EXISTS media_round_items (
    round_id        BIGINT      NOT NULL REFERENCES media_review_rounds(id) ON DELETE CASCADE,
    item_id         BIGINT      NOT NULL REFERENCES media_items(id) ON DELETE CASCADE,
    rank            INTEGER     NOT NULL,       -- 審核順序（0 起，優先分數高的在前）
    priority        REAL        NOT NULL,
    reasons         JSONB       NOT NULL DEFAULT '[]'::jsonb,
    blind           BOOLEAN     NOT NULL DEFAULT FALSE,
    baseline        JSONB       NOT NULL DEFAULT '[]'::jsonb,
    state           TEXT        NOT NULL DEFAULT 'pending' CHECK (state IN ('pending', 'done', 'doubt')),
    reviewed_by     TEXT,
    reviewed_at     TIMESTAMPTZ,
    seconds_spent   REAL,
    clean_view_seen BOOLEAN,
    diff            JSONB,                      -- {"accepted", "modified", "added", "deleted", "per_class": {...}}
    doubt_note      TEXT,
    PRIMARY KEY (round_id, item_id)
);
CREATE INDEX IF NOT EXISTS idx_media_round_items_rank ON media_round_items (round_id, state, rank);
CREATE INDEX IF NOT EXISTS idx_media_round_items_item ON media_round_items (item_id);
