-- 挑圖（strategy=diverse，edgevision/media/select.py）：每張圖記錄它是怎麼被挑中的
--   {"strategy": "diverse", "moving": true, "motion": 0.031, "group": 87, "group_size": 23}
--   group_size = 這張代表了多少張相似的畫面。之後「下一張」依此排序（有變動、代表性高的先標）。
ALTER TABLE media_items ADD COLUMN IF NOT EXISTS select_info JSONB;
