-- 示範資料：兩條線、兩台合成相機、兩顆「模型」的類別資訊。
--
-- 模型類別正式環境由 Registry（control-api）提供；demo 不起 Registry，
-- 設定 API 查不到時會退回 model_info_cache（最後一次查到的值）—— 這裡直接預填那份快取。

INSERT INTO model_info_cache (name, info) VALUES
  ('demo_det', '{"version": 1, "task": "detect", "imgsz": 640, "classes": ["person", "forklift", "spotter"]}'),
  ('ppe',      '{"version": 1, "task": "detect", "imgsz": 640, "classes": ["helmet", "no_helmet", "vest", "no_vest"]}')
ON CONFLICT (name) DO NOTHING;

-- 線一：禁區闖入（主站）+ 防護具（附加站，吃第二顆模型，每 2 幀跑一次）
INSERT INTO lines (id, name, station, line_params, model, extra_stations, updated_by) VALUES
  ('demo_zone', '示範：倉庫走道', 'zone_intrusion',
   '{"target_fps": 10, "demo_scene": "zone"}',
   '{"name": "demo_det", "conf": 0.5}',
   '[{"station": "ppe_check", "interval": 1, "model": {"name": "ppe", "conf": 0.5, "hidden_classes": ["helmet", "vest", "no_vest"]}}]',
   'demo-seed'),
-- 線二：堆高機進場後 10 秒內要有引導員
  ('demo_dock', '示範：卸貨區', 'followup_window',
   '{"target_fps": 10, "demo_scene": "dock"}',
   '{"name": "demo_det", "conf": 0.5}',
   '[]',
   'demo-seed')
ON CONFLICT (id) DO NOTHING;

INSERT INTO cameras (id, line_id, name, ip, stream_path, width, height, station_params, extra_params, updated_by) VALUES
  ('cam-01', 'demo_zone', '走道東側', '127.0.0.1', '/synthetic', 1280, 720,
   '{"zone": [[0.62, 0.45], [0.95, 0.45], [0.95, 0.95], [0.62, 0.95]], "thresholds": {"persist_frames": 3, "cooldown_seconds": 15}}',
   '{"ppe_check": {"thresholds": {"violations": {"no_helmet": "未戴安全帽"}, "report_cooldown_seconds": 30}}}',
   'demo-seed'),
  ('cam-02', 'demo_dock', '卸貨口', '127.0.0.1', '/synthetic', 1280, 720,
   '{"trigger_class": "forklift", "required_class": "spotter", "thresholds": {"seconds": 10}}',
   '{}',
   'demo-seed')
ON CONFLICT (id) DO NOTHING;
