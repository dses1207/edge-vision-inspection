-- 圖片集的 org（2026-10-20）：一個圖片集一個值（自由文字），清單可依它篩。給人找圖片集用，不影響標注、匯出、訓練。
-- NULL = 未設定。最早叫 category（同一天改名，當時還沒有任何圖片集填過）→ 已有 category 欄位的資料庫直接改名。
-- 每次部署都會整批重跑（deploy/ci/migrate_db.sh），所以要能重複執行。
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'media_collections' AND column_name = 'category')
       AND NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'media_collections' AND column_name = 'org') THEN
        ALTER TABLE media_collections RENAME COLUMN category TO org;
    END IF;
END $$;
ALTER TABLE media_collections ADD COLUMN IF NOT EXISTS org TEXT;
DROP INDEX IF EXISTS media_collections_category;
CREATE INDEX IF NOT EXISTS media_collections_org ON media_collections (org);
