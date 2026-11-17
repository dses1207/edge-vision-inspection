-- 標注專案的角色要從資料看得出來：
--   collection_id         待審圖片集：專案匯入、模型預標、人審核（整個系統唯一要審的東西）
--   golden_collection_id  測試集圖片集：從頭人工標；全部標完自動匯出成 LakeFS 版本 → 寫回 golden_ref
--   第 0 代初始人工標注資料集（dataset_ref）＋初始模型（model_name / model_version）
-- 不屬於任何專案的圖片集不會出現在審核前端。
ALTER TABLE media_campaigns ADD COLUMN IF NOT EXISTS golden_collection_id BIGINT
    REFERENCES media_collections(id) ON DELETE SET NULL;
ALTER TABLE media_campaigns ADD COLUMN IF NOT EXISTS golden_sync JSONB;   -- 測試集匯出進度 {signature, task_id, dataset_ref, error}

-- 每一代的模型名（NULL = 專案的 model_name；第 0 代的初始模型常是另一個名字，例 siteb_bravo_seed）
ALTER TABLE media_campaign_iterations ADD COLUMN IF NOT EXISTS model_name TEXT;
