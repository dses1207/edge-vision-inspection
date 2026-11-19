-- 權限管理（ipcam-web 的頁面權限）：原本暫存在前端伺服器的 JSON 檔，K8s 多副本 / pod 重建會不見。
-- 判斷規則在前端（admin → 個人例外 → 單位 → 訪客範本），後端存資料並自己判斷「誰能改權限」。
-- 帳號 = SSO token 的 sub 小寫（corp\legacy）：多個廠區 AD 分開，去掉前綴會撞名；大小寫 AuthServer 不保證。
-- 可重複執行（deploy/ci/migrate_db.sh 每次部署全跑）。

CREATE TABLE IF NOT EXISTS perm_users (
    account      VARCHAR(200) PRIMARY KEY,              -- corp\legacy（小寫）
    name         VARCHAR(200) NOT NULL DEFAULT '',      -- 每次登入用 token 更新
    unit         VARCHAR(100) NOT NULL DEFAULT '',      -- 每次登入用 token 更新
    email        VARCHAR(200),
    role         VARCHAR(10)  NOT NULL DEFAULT 'user' CHECK (role IN ('admin', 'user', 'guest')),
    first_login  TIMESTAMPTZ,
    last_login   TIMESTAMPTZ,
    updated_by   VARCHAR(200),
    updated_at   TIMESTAMPTZ  NOT NULL DEFAULT now()
);

-- 單位「有設定」= perm_units 有這一筆（就算一頁都沒開）
CREATE TABLE IF NOT EXISTS perm_units (
    unit        VARCHAR(100) PRIMARY KEY,
    updated_by  VARCHAR(200),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS perm_unit_pages (
    unit   VARCHAR(100) NOT NULL REFERENCES perm_units(unit) ON DELETE CASCADE,
    page   VARCHAR(200) NOT NULL,
    level  VARCHAR(4)   NOT NULL CHECK (level IN ('view', 'edit')),   -- none 不存
    PRIMARY KEY (unit, page)
);

-- 個人例外：none 要存（= 收回）
CREATE TABLE IF NOT EXISTS perm_user_overrides (
    account  VARCHAR(200) NOT NULL,
    page     VARCHAR(200) NOT NULL,
    level    VARCHAR(4)   NOT NULL CHECK (level IN ('none', 'view', 'edit')),
    PRIMARY KEY (account, page)
);

CREATE TABLE IF NOT EXISTS perm_guest_pages (
    page   VARCHAR(200) PRIMARY KEY,
    level  VARCHAR(4)   NOT NULL CHECK (level IN ('view', 'edit'))
);

-- enforce：權限控管開關；version：每次整包寫入 +1，擋兩個管理者同時存檔互相覆蓋
CREATE TABLE IF NOT EXISTS perm_settings (
    key         VARCHAR(50)  PRIMARY KEY,
    value       VARCHAR(200) NOT NULL,
    updated_by  VARCHAR(200),
    updated_at  TIMESTAMPTZ  NOT NULL DEFAULT now()
);

-- 異動紀錄：每次存檔一筆，before/after 整包
CREATE TABLE IF NOT EXISTS perm_audit (
    id        BIGSERIAL PRIMARY KEY,
    actor     VARCHAR(200) NOT NULL,
    at        TIMESTAMPTZ  NOT NULL DEFAULT now(),
    trace_id  VARCHAR(32),
    reason    TEXT,
    before    JSONB,
    after     JSONB
);

ALTER TABLE perm_audit ADD COLUMN IF NOT EXISTS reason TEXT;

-- 11/19 員工名冊同步（edgevision/api/directory_sync.py）：人員改由 AuthServer 名冊帶入，不用等本人登入
ALTER TABLE perm_users ADD COLUMN IF NOT EXISTS enabled     BOOLEAN NOT NULL DEFAULT TRUE;  -- 名冊停用 = 每頁 none
ALTER TABLE perm_users ADD COLUMN IF NOT EXISTS unit_name   VARCHAR(200);                   -- 部門名稱（顯示用）
ALTER TABLE perm_users ADD COLUMN IF NOT EXISTS title       VARCHAR(200);
ALTER TABLE perm_users ADD COLUMN IF NOT EXISTS employee_id VARCHAR(50);
ALTER TABLE perm_users ADD COLUMN IF NOT EXISTS synced_at   TIMESTAMPTZ;                    -- 最後一次從名冊更新
CREATE INDEX IF NOT EXISTS perm_users_unit ON perm_users (unit);

-- 初始資料（已有就不動）
INSERT INTO perm_settings (key, value, updated_by) VALUES ('enforce', 'false', 'seed') ON CONFLICT (key) DO NOTHING;
INSERT INTO perm_settings (key, value, updated_by) VALUES ('version', '1', 'seed') ON CONFLICT (key) DO NOTHING;
-- 第一位 admin：沒有這筆就沒有人進得了權限管理頁
INSERT INTO perm_users (account, name, role, updated_by) VALUES ('corp\alice', '', 'admin', 'seed')
    ON CONFLICT (account) DO NOTHING;
-- U100 先建成「有設定、還沒開頁」：頁面清單在前端，由 admin 第一次在權限管理頁整包存入
-- 單位與訪客範本只在「還沒人存過」時放：之後被管理者刪掉也不再補回（migrate_db.sh 每次部署都會重跑本檔）
INSERT INTO perm_units (unit, updated_by)
    SELECT 'U100', 'seed' WHERE NOT EXISTS (SELECT 1 FROM perm_audit)
ON CONFLICT (unit) DO NOTHING;
INSERT INTO perm_guest_pages (page, level)
    SELECT p, 'view' FROM (VALUES ('/monitoring/live'), ('/reports/summary')) v(p)
    WHERE NOT EXISTS (SELECT 1 FROM perm_audit)
ON CONFLICT (page) DO NOTHING;

-- 11/19 加第二位 admin（使用者指定）。只做一次：異動紀錄有這筆就不再跑，之後被改成別的角色也不會被蓋回。
-- 還沒登入過也先建好人；已經登入過（role=user）就升級。version +1，避免前端拿舊資料存檔把它蓋掉。
WITH done AS (SELECT 1 FROM perm_audit WHERE actor = 'seed' AND reason = '加第二位 admin：corp\legacy'),
up AS (
    INSERT INTO perm_users (account, name, role, updated_by)
        SELECT 'corp\legacy', '', 'admin', 'seed' WHERE NOT EXISTS (SELECT 1 FROM done)
    ON CONFLICT (account) DO UPDATE SET role = 'admin', updated_by = 'seed', updated_at = now()
    RETURNING account),
ver AS (
    UPDATE perm_settings SET value = (value::int + 1)::text, updated_by = 'seed', updated_at = now()
    WHERE key = 'version' AND EXISTS (SELECT 1 FROM up) RETURNING value)
INSERT INTO perm_audit (actor, reason, after)
    SELECT 'seed', '加第二位 admin：corp\legacy', jsonb_build_object('account', account, 'role', 'admin') FROM up;
