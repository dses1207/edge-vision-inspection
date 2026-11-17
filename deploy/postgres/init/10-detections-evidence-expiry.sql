-- ════════════════════════════════════════════════════════
-- 存證過期：影片 / 截圖過期後，紀錄保留、連結標成失效（2026-10-05）
-- ════════════════════════════════════════════════════════
--
-- MinIO 的事件影片保留 7 天（bucket 過期規則，見 deploy/minio/init/init-buckets.sh），
-- NG 紀錄要留更久（統計、追溯）。影片被 MinIO 刪掉後，紀錄若還指著它就是死連結：
-- 前端點了才發現打不開。每天 edgevision-retention 排程（run_retention.py）把超過保留天數的
-- clip_uri / image_uri 清成 NULL，並記下 evidence_expired_at —— 前端看到它就顯示「影片已過期」。
-- 舊系統的清理容器只刪影片、不管資料庫，留下的就是這種死連結。
--
-- 可重複執行。

ALTER TABLE detections ADD COLUMN IF NOT EXISTS evidence_expired_at TIMESTAMPTZ;
