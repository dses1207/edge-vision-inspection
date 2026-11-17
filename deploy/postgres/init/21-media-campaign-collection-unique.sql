-- 結案（state = closed）：不再推進，圖片集釋放給別的專案。
ALTER TABLE media_campaigns DROP CONSTRAINT IF EXISTS media_campaigns_state_check;
ALTER TABLE media_campaigns ADD CONSTRAINT media_campaigns_state_check
    CHECK (state IN ('idle', 'running', 'waiting', 'failed', 'paused', 'closed'));   -- idle：v0.3（26-*.sql 再收緊）

-- 一個圖片集只給一個進行中（未結案）的專案當待審圖片集：兩個專案一起預標、審核、匯出同一批圖會互相蓋掉。
-- API 先擋並回清楚的訊息；這個索引防兩個請求同時建的競態。
DROP INDEX IF EXISTS media_campaigns_collection_uniq;
CREATE UNIQUE INDEX IF NOT EXISTS media_campaigns_collection_active_uniq
    ON media_campaigns (collection_id) WHERE state <> 'closed';
