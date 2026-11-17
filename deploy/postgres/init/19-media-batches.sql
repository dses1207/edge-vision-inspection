-- 批次：每次把圖放進圖片集（上傳 / zip / 影片 / NAS / 從別的圖片集複製）= 一個批次。
-- 圖記得自己屬於哪一批；預標、建審核輪、匯出都可以按批次篩。同一張圖已在圖片集裡 → 跳過、不改它的批次。
CREATE TABLE IF NOT EXISTS media_batches (
    id              BIGSERIAL   PRIMARY KEY,
    collection_id   BIGINT      NOT NULL REFERENCES media_collections(id) ON DELETE CASCADE,
    name            TEXT        NOT NULL,
    kind            TEXT        NOT NULL CHECK (kind IN ('upload', 'zip', 'video', 'nas', 'copy')),
    params          JSONB       NOT NULL DEFAULT '{}'::jsonb,   -- 來源參數（相機 / 時段、來源圖片集…）；回填的帶 backfill: true
    task_id         TEXT,                                       -- 背景匯入任務（同步上傳 / 複製沒有）
    created_by      TEXT        NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_media_batches_coll ON media_batches (collection_id, id);

ALTER TABLE media_items ADD COLUMN IF NOT EXISTS batch_id BIGINT REFERENCES media_batches(id) ON DELETE SET NULL;
CREATE INDEX IF NOT EXISTS idx_media_items_batch ON media_items (collection_id, batch_id);
ALTER TABLE media_exports ADD COLUMN IF NOT EXISTS batch_ids JSONB;          -- 這次匯出了哪些批次（NULL = 整個圖片集）
ALTER TABLE media_review_rounds ADD COLUMN IF NOT EXISTS batch_ids JSONB;    -- 這一輪從哪些批次挑（NULL = 不限）

-- 回填：還沒有批次的圖，依「同圖片集＋同匯入者＋同來源種類＋前後間隔 < 10 分鐘」分成一批。可重複執行（只動 batch_id 是 NULL 的）。
DO $$
DECLARE g RECORD; bid BIGINT;
BEGIN
  FOR g IN
    WITH x AS (
      SELECT i.id, i.collection_id, i.added_by, i.added_at,
             CASE s.kind WHEN 'zip' THEN 'zip' WHEN 'nas_recording' THEN 'nas' WHEN 'upload_video' THEN 'video'
                         ELSE 'upload' END AS kind
      FROM media_items i LEFT JOIN media_sources s ON s.id = i.source_id
      WHERE i.batch_id IS NULL),
    y AS (
      SELECT *, lag(added_at) OVER (PARTITION BY collection_id, added_by, kind ORDER BY added_at, id) AS prev FROM x),
    z AS (
      SELECT *, sum(CASE WHEN prev IS NULL OR added_at - prev > interval '10 minutes' THEN 1 ELSE 0 END)
                  OVER (PARTITION BY collection_id, added_by, kind ORDER BY added_at, id) AS grp FROM y)
    SELECT collection_id, added_by, kind, grp, min(added_at) AS t0, array_agg(id) AS ids
    FROM z GROUP BY collection_id, added_by, kind, grp ORDER BY collection_id, t0
  LOOP
    INSERT INTO media_batches (collection_id, name, kind, params, created_by, created_at)
    VALUES (g.collection_id,
            (CASE g.kind WHEN 'zip' THEN 'zip 匯入' WHEN 'nas' THEN 'NAS 錄影' WHEN 'video' THEN '影片' ELSE '上傳' END)
              || ' ' || to_char(g.t0 AT TIME ZONE 'Asia/Taipei', 'MM-DD HH24:MI'),
            g.kind, '{"backfill": true}'::jsonb, g.added_by, g.t0)
    RETURNING id INTO bid;
    UPDATE media_items SET batch_id = bid WHERE id = ANY(g.ids);
  END LOOP;
END $$;
