-- 通知系統（Notifier）的兩張表。都是【紀錄】，不是設定 —— 設定走環境變數。
--
-- ⚠️ 為什麼沒有 notify_rules 表
--   舊系統把告警規則表、CRUD、送通知的函式全寫了，
--   但沒有任何地方呼叫它 —— NG 從來沒寄過信、沒推過 Teams。
--   這次路由固定寫死（幾個通知頻道，環境變數），反而是保證「做了就一定接上」。

-- 每一則通知的去向與結果。日報統計「昨日送出 / 壓下 / 失敗 / 死信」用。
CREATE TABLE IF NOT EXISTS notify_log (
    id          BIGSERIAL   PRIMARY KEY,
    -- daily | health | ng
    kind        TEXT        NOT NULL,
    -- 對應的事件識別：health 是項目名、ng 是 stream id、daily 是日期
    event_id    TEXT        NOT NULL DEFAULT '',
    -- teams-daily | teams-health | teams-ng
    receiver    TEXT        NOT NULL,
    -- sent | suppressed | failed | dead
    status      TEXT        NOT NULL,
    error       TEXT,
    sent_at     TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_notify_log_sent ON notify_log (sent_at DESC);

-- 健康狀態的每一次切換。日報「昨日中斷事件清單」與事後追查用。
-- 只記切換，不記每輪巡檢 —— 每 60 秒一輪、十幾個項目，全記的話一天幾萬列，沒人要看。
CREATE TABLE IF NOT EXISTS health_events (
    id          BIGSERIAL   PRIMARY KEY,
    -- 例如 line:alpha_a、service:recorder、storage:database
    item        TEXT        NOT NULL,
    from_state  TEXT        NOT NULL,
    to_state    TEXT        NOT NULL,
    at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    -- 恢復時填：從上次進入非 ok 到現在幾秒。進入故障時為 NULL
    duration_s  INTEGER
);
CREATE INDEX IF NOT EXISTS idx_health_events_at ON health_events (at DESC);
