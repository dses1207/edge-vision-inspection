-- ════════════════════════════════════════════════════════
-- detections.station：這筆判定是哪一站做的（2026-10-01）
-- ════════════════════════════════════════════════════════
--
-- 一條線可掛多站（主站 + 附加站，例：主模型的動作判定 + 防護具），
-- 同一台相機、甚至同一個模型會產生不同站的 NG。runner 早就在 FrameResult.station 帶了，
-- 但主 repo 的契約副本沒有這個欄位 → event-processor 解析時默默丟掉，DB 分不出站別。
-- 契約改成只有一份（ai-infer 的 edgevision-contracts 套件）後補上這欄。
-- NULL = 舊資料或舊格式（視為該線主站）。
--
-- 可重複執行；既有資料庫手動跑一次：
--   kubectl -n edgevision exec -i postgres-0 -- psql -U edgevision -d edgevision < deploy/postgres/init/08-detections-station.sql

ALTER TABLE detections ADD COLUMN IF NOT EXISTS station TEXT;
