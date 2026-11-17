-- ════════════════════════════════════════════════════════
-- 線路與相機的延伸欄位：一線多站、主站模型覆寫、主站 ROI 推論（2026-09-29）
-- ════════════════════════════════════════════════════════
--
-- 為什麼要：從舊腳本轉過來的設定用到 runner 早就支援的四個欄位，
-- 但資料庫與設定 API 不認得 —— 那幾條線只能用 CAMERA_SOURCE=file 跑，進不了整套系統
-- （前端、/healthz、錄影都看不到）。欄位語意以 ai-infer ds_runner/config.py 為準：
--
--   lines.model            主站模型 {name, conf, iou}；NULL = 站別預設模型（classes / imgsz 自 15 起不存，見 15-model-intrinsic.sql）
--                          （但 station.yaml 不帶 imgsz/conf/iou，1024 尺寸的模型一定要給）
--   lines.extra_stations   附加站 [{station, model?, interval?}]，每個一個 nvinferserver（例：PPE）
--   cameras.infer_roi      主站只推論這塊 [x1, y1, x2, y2]（正規化 0~1）；NULL = 整幅
--   cameras.extra_params   附加站的現場值 {站別: {...}}；主站仍用 station_params
--
-- 可重複執行（IF NOT EXISTS）：新庫由 docker-entrypoint-initdb.d 自動跑；
-- 既有資料庫（測試機）手動跑一次：
--   docker exec -i edgevision-postgres psql -U edgevision -d edgevision < deploy/postgres/init/06-line-extras.sql

ALTER TABLE lines   ADD COLUMN IF NOT EXISTS model          JSONB;
ALTER TABLE lines   ADD COLUMN IF NOT EXISTS extra_stations JSONB NOT NULL DEFAULT '[]'::jsonb;
ALTER TABLE cameras ADD COLUMN IF NOT EXISTS infer_roi      JSONB;
ALTER TABLE cameras ADD COLUMN IF NOT EXISTS extra_params   JSONB NOT NULL DEFAULT '{}'::jsonb;
