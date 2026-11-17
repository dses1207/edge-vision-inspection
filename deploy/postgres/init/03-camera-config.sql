-- 相機與線路設定
--
--
-- ⚠️ 這裡是「相機設定的真相來源」。runner 本機不存相機資料，
--    啟動時用 LINE_ID 向這裡拉。
--
-- 為什麼放 DB 而不是 Git YAML：
--   · 操作者是產線現場人員，救火時走 PR 會被繞過 → Git 不再是真相
--   · 只有結構化入口（API）才能算出 delta → 才可能做到「改 ROI 不停線」
--   · API 有 schema 驗證，Git 沒有
--   · Milestone / Genetec / ZoneMinder / Shinobi / ThingsBoard 一致採 DB

-- ════════════════════════════════════════════════════════
-- 線路：一條線 = 一個 runner 進程 = 一個站別
-- ════════════════════════════════════════════════════════
CREATE TABLE IF NOT EXISTS lines (
    id              TEXT        PRIMARY KEY,     -- runner 啟動時帶的 LINE_ID
    name            TEXT        NOT NULL,

    -- 站別名，對應 station.yaml 的 name。
    -- 「這條線跑哪個判定模組」因此也是純資料操作，不必改程式。
    station         TEXT        NOT NULL,

    -- 目前部署的映像；與模型部署狀態分屬兩件事，不要混用
    image_sha       TEXT,

    -- ⭐ 設定版本號（單調遞增）。runner 載入後回報自己實際跑的版本，
    -- 控制端比對 desired(此欄) vs reported(line_status) 得出 delta。
    -- 這是為了避開 Envoy 式「靜默拒絕」：推送成功不等於生效。
    config_version  BIGINT      NOT NULL DEFAULT 1,

    -- 線路層級參數（例如 target_fps）。station.yaml 給預設值，這裡覆寫。
    line_params     JSONB       NOT NULL DEFAULT '{}'::jsonb,

    enabled         BOOLEAN     NOT NULL DEFAULT TRUE,
    updated_by      TEXT,
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- ════════════════════════════════════════════════════════
-- 相機：一台相機屬於一條線（一對多）
-- ════════════════════════════════════════════════════════
--
-- ⚠️ 一相機一線一站是已定案的前提。若日後真需要
--    「一台相機餵多條線」，加一張 line_cameras 關聯表、把 station_params
--    搬過去即可，本表不動。現在不做。
CREATE TABLE IF NOT EXISTS cameras (
    id              TEXT        PRIMARY KEY,     -- 相機編號，如 cam156
    line_id         TEXT        NOT NULL REFERENCES lines(id) ON DELETE RESTRICT,

    name            TEXT,
    factory         TEXT,
    production_line TEXT,

    -- 連線資訊
    ip              TEXT        NOT NULL,
    stream_path     TEXT,                        -- 子碼流路徑，如 /stream2

    -- ⭐ 帳密「引用名」，不存值。由設定載入層在 Compose(env_file 變數名)
    -- 與 K8s(Secret key) 各自解析 —— 不把 Secret 結構洩進資料庫。
    cred_ref        TEXT,

    -- 影像規格。先前寫死在 ds_pipeline.py（1280x720），改由此處提供。
    width           INTEGER     NOT NULL DEFAULT 1280,
    height          INTEGER     NOT NULL DEFAULT 720,

    -- ⭐ 現場校正值（ref_quad / ROI / 門檻覆寫…）。
    -- 對應 station.yaml 的 camera_params_required：程式宣告需要什麼，
    -- 這裡提供實際的值，runner 啟動時比對，缺了就啟動失敗。
    station_params  JSONB       NOT NULL DEFAULT '{}'::jsonb,

    enabled         BOOLEAN     NOT NULL DEFAULT TRUE,
    updated_by      TEXT,
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_cameras_line    ON cameras (line_id);
CREATE INDEX IF NOT EXISTS idx_cameras_enabled ON cameras (line_id) WHERE enabled = TRUE;

-- ════════════════════════════════════════════════════════
-- 線路狀態回報：reported（對應 lines.config_version 的 desired）
-- ════════════════════════════════════════════════════════
--
-- ⭐ runner 必須主動回報「我實際載入的是第幾版」，控制端不得假設推送成功。
--    這個模式 ZoneMinder 用純 SQL 兩張表就做出來了（Monitors=desired /
--    Monitor_Status=actual），幾十路的規模不需要 Device Shadow 那套服務。
CREATE TABLE IF NOT EXISTS line_status (
    line_id             TEXT        PRIMARY KEY REFERENCES lines(id) ON DELETE CASCADE,

    -- runner 實際跑的版本。< lines.config_version 表示尚未套用
    reported_version    BIGINT,
    -- ok | applying | failed
    apply_state         TEXT        NOT NULL DEFAULT 'ok',
    apply_error         TEXT,

    -- 心跳。過期即視為該線失聯
    -- ⚠️ 判斷失聯時要能分辨「這條線掛了」與「共用依賴(DB/Triton)掛了」——
    --    後者會讓所有線同時看似失聯，不該觸發全體重啟（ZoneMinder 的教訓）
    heartbeat_at        TIMESTAMPTZ,
    cameras_running     INTEGER,
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- ════════════════════════════════════════════════════════
-- 設定版本快照：可還原（audit_log 不足以還原）
-- ════════════════════════════════════════════════════════
--
-- ⚠️ 為什麼不能只靠 audit_log：NetBox 的教訓 —— before/after 的
--    changelog 字典【不足以還原一個 UPDATE】，有依賴關係時 rollback
--    會重建出不同 ID 造成掛載問題。故存「整個設定物件的完整 JSON」。
--
-- ⚠️ 本表是【永久資料】，不套用一般 log 的清理政策。
--    NetBox 升級後 changelog retention 預設變 90 天(4.4+ 變零)，使用者歷史全失；
--    Milestone 官方亦明載 audit log 不屬於 configuration backup、還原時不還原。
--    工廠可能有法規稽核需求，這份不能被清掉。
CREATE TABLE IF NOT EXISTS config_snapshots (
    id              BIGSERIAL   PRIMARY KEY,
    ts              TIMESTAMPTZ NOT NULL DEFAULT now(),

    line_id         TEXT        NOT NULL,
    config_version  BIGINT      NOT NULL,

    -- 整條線的完整設定（line 本身 + 其下所有 cameras），可直接還原
    snapshot        JSONB       NOT NULL,

    changed_by      TEXT,
    change_reason   TEXT,
    -- 對應的 audit_log 條目，兩者互為索引
    audit_id        BIGINT
);

CREATE INDEX IF NOT EXISTS idx_snapshots_line ON config_snapshots (line_id, config_version DESC);

COMMENT ON TABLE lines            IS '線路設定（desired）。一條線=一個 runner 進程=一個站別';
COMMENT ON TABLE cameras          IS '相機設定（desired）。真相來源，runner 啟動時用 line_id 拉';
COMMENT ON TABLE line_status      IS '線路回報狀態（reported）。與 lines.config_version 比對得出 delta';
COMMENT ON TABLE config_snapshots IS '設定版本快照，供還原。永久保留，不套 log 清理政策';
