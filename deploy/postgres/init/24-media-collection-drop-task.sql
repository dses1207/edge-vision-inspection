-- 圖片集不綁輸出格式（框 / 多邊形 / 關鍵點在匯出時選，見 edgevision/media/export.py）。
-- 程式已不讀這個欄位；2026-10-19 拿掉前，非 detect 的只有舊系統搬來的 12 個 segment 圖片集
-- （id 10 12 13 16 18 19 20 23 24 25 26 29），匯出時要選 task=segment。
ALTER TABLE media_collections DROP COLUMN IF EXISTS task;
