-- ════════════════════════════════════════════════════════
-- 圖片庫 / 圖片集 / 標注
--
-- 從舊系統移植，但刻意改了三件事：
--   ① 標注全放 PostgreSQL（標注不要拆兩個資料庫存，先寫一邊再寫另一邊一定會不同步）
--   ② 每張圖有明確狀態（舊系統只有 is_annotated，分不出 AI 標的和人確認過的）
--   ③ 草稿存伺服器（舊系統存瀏覽器 localStorage，換電腦半成品就沒了）
-- ════════════════════════════════════════════════════════

-- 一張實體圖 / 一支實體影片。主鍵是內容 sha256：同一張圖重複上傳只存一份，檔案只寫不改。
CREATE TABLE IF NOT EXISTS media_blobs (
    sha256          TEXT        PRIMARY KEY,
    kind            TEXT        NOT NULL CHECK (kind IN ('image', 'video')),
    bucket          TEXT        NOT NULL,
    object_key      TEXT        NOT NULL,
    content_type    TEXT,
    size_bytes      BIGINT      NOT NULL,
    width           INTEGER,
    height          INTEGER,
    duration_s      REAL,                       -- 影片才有
    created_by      TEXT        NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- 一個來源檔：上傳的影片、NAS 錄影段、上傳的 zip。幀記得自己從哪來（存證影片有燒錄的框，不收）。
CREATE TABLE IF NOT EXISTS media_sources (
    id              BIGSERIAL   PRIMARY KEY,
    kind            TEXT        NOT NULL,       -- upload_video | nas_recording | zip
    uri             TEXT        NOT NULL,       -- s3://… 或 nas://raw/<cam>/<file>.mp4
    blob_sha256     TEXT        REFERENCES media_blobs(sha256),
    camera_id       TEXT,
    started_at      TIMESTAMPTZ,                -- 錄影起始時間（幀的拍攝時間 = started_at + offset）
    params          JSONB       NOT NULL DEFAULT '{}'::jsonb,
    created_by      TEXT        NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- 圖片集：一組圖 + 一套類別。標注在這一層（同一張圖在不同圖片集可以有不同類別、不同標注）。
CREATE TABLE IF NOT EXISTS media_collections (
    id              BIGSERIAL   PRIMARY KEY,
    name            TEXT        NOT NULL UNIQUE,
    description     TEXT        NOT NULL DEFAULT '',
    class_names     JSONB       NOT NULL DEFAULT '[]'::jsonb,
    kpt_shape       JSONB,                      -- 關鍵點標注：[點數, 3]（輸出成框 / 多邊形 / 關鍵點是匯出時選，不綁在圖片集）
    line_id         TEXT,
    camera_id       TEXT,
    org             TEXT,                       -- 分類（給人找用；NULL = 未分類），見 25-media-collection-org.sql
    archived        BOOLEAN     NOT NULL DEFAULT FALSE,
    created_by      TEXT        NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- 圖片集裡的一張圖 + 它的標注。
--   status：unlabeled（沒人碰）| auto（AI 預標、沒人看）| labeled（人工標完）| background（人看過、確實沒東西）
--   只有 labeled / background 會匯出成訓練資料。
--   revision：每次寫入 +1；存草稿要帶目前的 revision，不一致回 409（兩人同時改不會互蓋）。
--   locked_by / lock_until：開啟即領取，逾時自動失效（不用人解鎖）。
CREATE TABLE IF NOT EXISTS media_items (
    id              BIGSERIAL   PRIMARY KEY,
    collection_id   BIGINT      NOT NULL REFERENCES media_collections(id) ON DELETE CASCADE,
    blob_sha256     TEXT        NOT NULL REFERENCES media_blobs(sha256) ON DELETE RESTRICT,
    source_id       BIGINT      REFERENCES media_sources(id) ON DELETE SET NULL,
    source_offset_s REAL,
    camera_id       TEXT,
    captured_at     TIMESTAMPTZ,
    original_name   TEXT,
    added_by        TEXT        NOT NULL,
    added_at        TIMESTAMPTZ NOT NULL DEFAULT now(),

    status          TEXT        NOT NULL DEFAULT 'unlabeled'
                    CHECK (status IN ('unlabeled', 'auto', 'labeled', 'background')),
    shapes          JSONB       NOT NULL DEFAULT '[]'::jsonb,
    revision        INTEGER     NOT NULL DEFAULT 0,
    updated_by      TEXT,
    updated_at      TIMESTAMPTZ,
    locked_by       TEXT,
    lock_until      TIMESTAMPTZ,

    UNIQUE (collection_id, blob_sha256)
);
CREATE INDEX IF NOT EXISTS idx_media_items_coll_status ON media_items (collection_id, status, id);
CREATE INDEX IF NOT EXISTS idx_media_items_blob        ON media_items (blob_sha256);

-- 標注歷史：送出、自動標注、匯入時各留一份（存草稿不留，太頻繁）。
CREATE TABLE IF NOT EXISTS media_annotation_history (
    id              BIGSERIAL   PRIMARY KEY,
    item_id         BIGINT      NOT NULL REFERENCES media_items(id) ON DELETE CASCADE,
    revision        INTEGER     NOT NULL,
    status          TEXT        NOT NULL,
    shapes          JSONB       NOT NULL,
    action          TEXT        NOT NULL,       -- submit | autolabel | import
    actor           TEXT        NOT NULL,
    task_id         TEXT,
    at              TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_media_hist_item ON media_annotation_history (item_id, revision DESC);

-- 續標：每人在每個圖片集最後停在哪張。
CREATE TABLE IF NOT EXISTS media_label_cursors (
    collection_id   BIGINT      NOT NULL REFERENCES media_collections(id) ON DELETE CASCADE,
    actor           TEXT        NOT NULL,
    item_id         BIGINT,
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (collection_id, actor)
);

-- 匯出成資料集：哪個圖片集的哪些圖、各自哪一版標注 → LakeFS 哪個版本。模型血緣可一路追到原圖。
--   items：[{item_id, sha256, revision}]。blob 被任何一筆匯出引用就不可清。
CREATE TABLE IF NOT EXISTS media_exports (
    id              BIGSERIAL   PRIMARY KEY,
    collection_id   BIGINT      REFERENCES media_collections(id) ON DELETE SET NULL,
    dataset_repo    TEXT        NOT NULL,
    dataset_ref     TEXT,                       -- lakefs://repo/commit（完成後才有）
    task_id         TEXT,
    items           JSONB       NOT NULL DEFAULT '[]'::jsonb,
    n_items         INTEGER     NOT NULL DEFAULT 0,
    status          TEXT        NOT NULL DEFAULT 'running',  -- running | completed | failed
    error           TEXT,
    created_by      TEXT        NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);
