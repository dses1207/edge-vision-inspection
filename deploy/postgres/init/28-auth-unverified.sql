-- 控制 API 身分關卡（edgevision/api/gate.py）的觀察記錄：沒帶有效 SSO token / 服務金鑰的請求，按天彙整。
-- 切強制模式（AUTH_ALLOW_HEADER_ACTOR=false）前看這張表，確定沒有還在用 X-Actor 的呼叫者。
-- 可重複執行（deploy/ci/migrate_db.sh 每次部署全跑）。
CREATE TABLE IF NOT EXISTS auth_unverified_calls (
    day         DATE         NOT NULL,
    method      VARCHAR(10)  NOT NULL,
    route       VARCHAR(200) NOT NULL,              -- 路由樣板，例如 /api/tasks/{task_id}
    actor       VARCHAR(200) NOT NULL DEFAULT '',   -- X-Actor（自我宣告，只當線索）
    channel     VARCHAR(20)  NOT NULL DEFAULT '',
    reason      VARCHAR(20)  NOT NULL,              -- no_token | bad_token | bad_service_token
    user_agent  VARCHAR(200) NOT NULL DEFAULT '',
    client      VARCHAR(64)  NOT NULL DEFAULT '',   -- X-Forwarded-For 第一個，沒有就是連線來源
    count       BIGINT       NOT NULL DEFAULT 0,
    first_seen  TIMESTAMPTZ  NOT NULL,
    last_seen   TIMESTAMPTZ  NOT NULL,
    PRIMARY KEY (day, method, route, actor, channel, reason, user_agent, client)
);
CREATE INDEX IF NOT EXISTS auth_unverified_calls_day ON auth_unverified_calls (day DESC);
