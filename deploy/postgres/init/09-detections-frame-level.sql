-- ════════════════════════════════════════════════════════
-- 整幀判定的 NG：沒有偵測物件也要留一筆（2026-10-02）
-- ════════════════════════════════════════════════════════
--
-- 有些 NG 是靠【時間】判的，判定當下畫面上沒有框：「A 之後 N 秒內沒看到 B」這類，
-- 逾時那一幀本來就沒有框。以前 runner 沒框就不送去落地、這張表也要求每列都有類別與框，
-- 結果這些 NG 不進資料庫 —— 前端查不到，錄影服務錄好的影片也掛不上去（變孤兒）。
--
-- 整幀判定寫成一列：class_name / confidence / bbox 為 NULL，verdict_* 照寫。
-- 不填假類別（例如 '(無偵測)'）：依類別統計會多出不存在的類別，訓練資料回流也可能誤當成標註。
--
-- 可重複執行（DROP NOT NULL 對已經可為空的欄位不會報錯）。

ALTER TABLE detections ALTER COLUMN class_name DROP NOT NULL;
ALTER TABLE detections ALTER COLUMN confidence DROP NOT NULL;
ALTER TABLE detections ALTER COLUMN bbox       DROP NOT NULL;
