-- ════════════════════════════════════════════════════════
-- PTZ
--
-- 相機本身沿用 cameras 表（帳密走 cred_ref，不明文存 DB）。
-- 這裡只存 PTZ 自己的座標：
--   target     歸位點：排程每 10 分鐘強定位回來；使用者手設的 ground truth，自動回正【永不覆寫】
--   origin     軟體中心點（goto_center）
--   reference  失準偵測基準：當下畫面（灰階小圖，存 MinIO）+ 當時座標
-- ════════════════════════════════════════════════════════

-- Tapo 的 ONVIF 埠是 2020（不是標準 80）；NULL = 用預設 2020
ALTER TABLE cameras ADD COLUMN IF NOT EXISTS onvif_port INTEGER;

CREATE TABLE IF NOT EXISTS ptz_positions (
    camera_id       TEXT        NOT NULL REFERENCES cameras(id) ON DELETE CASCADE,
    kind            TEXT        NOT NULL CHECK (kind IN ('target', 'origin', 'reference')),
    pan             DOUBLE PRECISION,
    tilt            DOUBLE PRECISION,
    zoom            DOUBLE PRECISION NOT NULL DEFAULT 0,
    has_coord       BOOLEAN     NOT NULL DEFAULT TRUE,   -- reference 存的當下讀不到座標 → false
    ref_object_key  TEXT,                                -- reference 的灰階小圖（raw/ptz/refs/…npz）
    updated_by      TEXT,
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (camera_id, kind)
);

-- 能不能轉（雲台）。有沒有雲台要用 ONVIF 查；固定鏡頭通常沒有 ONVIF。
-- 只在第一次加欄位時依線別帶入；之後新相機預設 false，用 PUT /api/ptz/{cam}/can-pan 設定（可自動偵測）。
DO $$
BEGIN
  IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                 WHERE table_schema = current_schema() AND table_name = 'cameras' AND column_name = 'can_pan') THEN
    ALTER TABLE cameras ADD COLUMN IF NOT EXISTS can_pan BOOLEAN NOT NULL DEFAULT FALSE;
    -- （依現場設定 can_pan；預設 false）
  END IF;
END $$;
