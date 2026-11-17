-- 建立各服務所需的 database。
--
-- 一個 PostgreSQL 實例、多個 database：本機開發省資源，
-- K8s 上可依需求拆成獨立實例（連線字串從環境變數讀，程式不用改）。

-- 第二階段：LakeFS 的 metadata
SELECT 'CREATE DATABASE lakefs'
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'lakefs')\gexec

-- 第二階段：ClearML（若採用 PG 後端）
SELECT 'CREATE DATABASE clearml'
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'clearml')\gexec
