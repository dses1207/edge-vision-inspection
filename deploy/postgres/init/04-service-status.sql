-- 消費者服務的心跳與工作指標。
--
-- ⭐ 為什麼需要這張表
--
--   docker healthcheck 只答「活著嗎」。今天三個故障全是「活著但沒用」：
--     · line_c runner 死了 20 天，line_status 還寫 ok（沒人看心跳）
--     · recorder 產出 25 GB 孤兒影片，healthcheck 綠燈（它做的事是錯的）
--     · clip_uri 從沒被寫過，每個服務都 healthy（鏈路是斷的）
--
--   第二層問題是「在做事嗎」——該產出的東西有沒有產出。這張表讓每個消費者
--   定期回報「我做了什麼」，/healthz 讀出來判 stale / degraded。
--
-- 模式照抄 line_status（runner 的心跳）：每 15 秒 upsert，60 秒沒心跳視為失聯。
-- 多一個 metrics jsonb 讓各服務報自己的指標，不用每加一個服務就改 schema。

CREATE TABLE IF NOT EXISTS service_status (
    -- event-processor | recorder | nas-recorder | ...
    service         TEXT        NOT NULL,
    -- 同一服務多副本時區分（hostname）。單副本就是一列
    instance        TEXT        NOT NULL DEFAULT '',

    heartbeat_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    started_at      TIMESTAMPTZ NOT NULL DEFAULT now(),

    -- 各服務自報的工作指標。/healthz 依 service 名稱套對應門檻。
    --   event-processor: {"lag": 積壓筆數, "last_write_s": 距上次寫 DB 秒數, "written": 累計}
    --   recorder:        {"orphans": 孤兒影片數, "made": 累計片段, "errors": 累計錯誤}
    --   nas-recorder:    {"free_gb": 剩餘空間, "stalls": 停滯次數, "cameras_alive": N, "cameras_total": M}
    metrics         JSONB       NOT NULL DEFAULT '{}',

    PRIMARY KEY (service, instance)
);

-- /healthz 每次都掃整張表；表很小（每服務一列），不需要更多索引。
