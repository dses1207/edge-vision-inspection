-- 模型自己的屬性（classes / imgsz / task）以模型為準，線路設定不存（edgevision/api/model_info.py）。
--   GET /api/config/lines/{id} 依 model.name 查 Registry production 版附上；runner 與調和器讀的就是那份。
--   2026-10-14 前線路設定自己存一份 classes / imgsz：靠人手抄、promote 換版後就過期。

-- Registry 連不上時用「最後一次查到的值」，不讓 control-api 掛掉連帶所有線起不來
CREATE TABLE IF NOT EXISTS model_info_cache (
    name       TEXT PRIMARY KEY,
    info       JSONB NOT NULL,          -- {version, task, imgsz, classes}
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- 舊資料：拿掉線路設定裡存著的模型屬性（不動 config_version：GET 的結果不變，runner 不需要重啟）
UPDATE lines SET model = model - 'classes' - 'imgsz' - 'task'
 WHERE model IS NOT NULL AND jsonb_typeof(model) = 'object' AND model ?| ARRAY['classes', 'imgsz', 'task'];

UPDATE lines SET extra_stations = (
    SELECT jsonb_agg(CASE WHEN jsonb_typeof(e -> 'model') = 'object'
                          THEN jsonb_set(e, '{model}', (e -> 'model') - 'classes' - 'imgsz' - 'task')
                          ELSE e END ORDER BY ord)
      FROM jsonb_array_elements(extra_stations) WITH ORDINALITY AS t(e, ord))
 WHERE jsonb_typeof(extra_stations) = 'array' AND EXISTS (
    SELECT 1 FROM jsonb_array_elements(extra_stations) e
     WHERE jsonb_typeof(e -> 'model') = 'object' AND (e -> 'model') ?| ARRAY['classes', 'imgsz', 'task']);
