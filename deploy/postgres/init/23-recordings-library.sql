-- NAS 24h 錄影的 API（edgevision/media/recordings.py）、錄影名單、影片庫與圖片的補充欄位。

-- 每段錄影開過一次就記下長度（moov 在檔尾，要開檔才知道）；檔案大小 / 修改時間變了才重開
CREATE TABLE IF NOT EXISTS nas_segments (
    camera_id       TEXT        NOT NULL,
    file            TEXT        NOT NULL,
    start_at        TIMESTAMPTZ NOT NULL,
    size_bytes      BIGINT      NOT NULL,
    mtime           DOUBLE PRECISION,
    duration_s      REAL,                       -- NULL = 開不了（壞檔）
    error           TEXT,
    probed_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (camera_id, file)
);

-- 錄影名單：哪些相機要 24h 錄到 NAS。nas-recorder（REC_ROSTER=db）每 CAMERA_REFRESH_S 秒重讀
CREATE TABLE IF NOT EXISTS rec_roster (
    camera_id       TEXT        PRIMARY KEY,
    enabled         BOOLEAN     NOT NULL DEFAULT FALSE,
    note            TEXT,
    updated_by      TEXT,
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);
-- 首次建立時沿用原本寫死在部署設定的名單（REC_CAMERAS）；之後以這張表為準，不會被重設
INSERT INTO rec_roster (camera_id, enabled, note, updated_by)
SELECT id, TRUE, '原 REC_CAMERAS 名單', 'migration' FROM cameras WHERE id = ANY(string_to_array(current_setting('edgevision.rec_cameras', true), ','))
ON CONFLICT (camera_id) DO NOTHING;

-- 影片庫：名稱、標籤、備註、到期時間（NULL = 不會自動刪）
ALTER TABLE media_sources ADD COLUMN IF NOT EXISTS name TEXT;
ALTER TABLE media_sources ADD COLUMN IF NOT EXISTS tags JSONB NOT NULL DEFAULT '[]'::jsonb;
ALTER TABLE media_sources ADD COLUMN IF NOT EXISTS note TEXT;
ALTER TABLE media_sources ADD COLUMN IF NOT EXISTS expires_at TIMESTAMPTZ;
CREATE INDEX IF NOT EXISTS idx_media_sources_kind ON media_sources (kind, id DESC);

-- 圖片：標籤與備註（跟標注無關，不進匯出、不算修訂）
ALTER TABLE media_items ADD COLUMN IF NOT EXISTS tags JSONB NOT NULL DEFAULT '[]'::jsonb;
ALTER TABLE media_items ADD COLUMN IF NOT EXISTS note TEXT;
CREATE INDEX IF NOT EXISTS idx_media_items_tags ON media_items USING gin (tags);
CREATE INDEX IF NOT EXISTS idx_media_items_name ON media_items (lower(original_name) text_pattern_ops);
