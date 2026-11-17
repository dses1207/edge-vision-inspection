-- edgevision 主要資料表
--

-- ════════════════════════════════════════════════════════
-- 偵測事件
-- ════════════════════════════════════════════════════════
CREATE TABLE IF NOT EXISTS detections (
    id              BIGSERIAL PRIMARY KEY,
    ts              TIMESTAMPTZ NOT NULL DEFAULT now(),

    -- 幀身分
    camera_id       TEXT        NOT NULL,
    frame_id        TEXT        NOT NULL,
    -- 畫進影像的身分標記。前端配對用，也是回溯特定影格的鍵。
    marker_id       BIGINT,
    rtp_timestamp   BIGINT,

    -- 模型身分：讓「換模型後誤判率有無改善」這種問題可回答
    model_name      TEXT        NOT NULL,
    model_version   TEXT        NOT NULL,

    -- 偵測結果
    class_name      TEXT        NOT NULL,
    confidence      REAL        NOT NULL,
    bbox            JSONB       NOT NULL,
    -- ⚠️ 座標系必須顯式記錄。YOLO 推論經過 letterbox（等比縮放+補邊），
    -- 不記錄縮放參數，日後回流訓練時無法還原框在原圖的位置。
    bbox_coord_system TEXT      NOT NULL DEFAULT 'original',
    image_meta      JSONB,
    mask_area       REAL,

    -- 業務判定
    verdict_status  TEXT,
    verdict_reason  TEXT,
    verdict_details JSONB,

    -- 證據
    image_uri       TEXT,
    clip_uri        TEXT,

    -- 閉環：人工複判結果是回流訓練的燃料
    reviewed        BOOLEAN     NOT NULL DEFAULT FALSE,
    review_result   TEXT,       -- true_positive | false_positive
    reviewed_by     TEXT,
    reviewed_at     TIMESTAMPTZ,
    fed_back        BOOLEAN     NOT NULL DEFAULT FALSE
);

CREATE INDEX IF NOT EXISTS idx_detections_ts        ON detections (ts DESC);
CREATE INDEX IF NOT EXISTS idx_detections_camera_ts ON detections (camera_id, ts DESC);
CREATE INDEX IF NOT EXISTS idx_detections_verdict   ON detections (verdict_status) WHERE verdict_status = 'ng';
-- 難例快查：低信心度的樣本是回流訓練最有價值的資料
CREATE INDEX IF NOT EXISTS idx_detections_lowconf   ON detections (confidence) WHERE confidence < 0.6;
CREATE INDEX IF NOT EXISTS idx_detections_pending   ON detections (reviewed) WHERE reviewed = FALSE;


-- ════════════════════════════════════════════════════════
-- 非同步任務
--
-- ⚠️ 狀態【必須】持久化。只放記憶體的話，Orchestrator 重啟後
-- 所有進行中的任務就查無此人 —— 但 K8s Job 其實還在跑，
-- 形成殭屍任務。
-- ════════════════════════════════════════════════════════
CREATE TABLE IF NOT EXISTS tasks (
    task_id         TEXT        PRIMARY KEY,
    type            TEXT        NOT NULL,   -- training | evaluation | export | deployment
    status          TEXT        NOT NULL,   -- working | input_required | completed | failed | cancelled
    status_message  TEXT,

    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    started_at      TIMESTAMPTZ,
    finished_at     TIMESTAMPTZ,
    -- 心跳：判斷任務是否卡死。行程活著但沒進度也算失效。
    last_heartbeat  TIMESTAMPTZ,

    created_by      TEXT        NOT NULL,
    owner           TEXT        NOT NULL,
    channel         TEXT        NOT NULL,   -- ui | mcp | api

    -- 輸入：完整記錄，clone 功能靠這些欄位重建
    dataset_id      TEXT,
    dataset_version TEXT,
    base_model      TEXT,
    params          JSONB       NOT NULL DEFAULT '{}'::jsonb,

    -- 輸出
    result          JSONB,
    model_version   TEXT,
    run_report_uri  TEXT,
    error           TEXT,

    -- 血脈：實驗系譜，讓「在之前的基礎上優化」可追溯
    parent_task_id  TEXT REFERENCES tasks(task_id),
    clone_reason    TEXT,

    k8s_job_name    TEXT
);

CREATE INDEX IF NOT EXISTS idx_tasks_status  ON tasks (status) WHERE status = 'working';
CREATE INDEX IF NOT EXISTS idx_tasks_created ON tasks (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_tasks_parent  ON tasks (parent_task_id);


-- ════════════════════════════════════════════════════════
-- 操作稽核
--
-- 記錄「誰、用什麼方式、做了什麼」。
-- actor_type 區分 human/agent 是關鍵 —— 出事時要知道是人做的
-- 還是 agent 做的。
--
-- ⚠️ 保留期比資料長：資料三個月，稽核至少一年。
-- 出事時這是唯一的線索。
-- ════════════════════════════════════════════════════════
CREATE TABLE IF NOT EXISTS audit_log (
    id          BIGSERIAL   PRIMARY KEY,
    ts          TIMESTAMPTZ NOT NULL DEFAULT now(),

    actor       TEXT        NOT NULL,
    actor_type  TEXT        NOT NULL,   -- human | agent
    channel     TEXT        NOT NULL,   -- ui | mcp | api
    client_info TEXT,

    action      TEXT        NOT NULL,
    target      TEXT,
    params      JSONB,

    result      TEXT        NOT NULL,   -- success | failed | rejected
    error       TEXT,
    task_id     TEXT
);

CREATE INDEX IF NOT EXISTS idx_audit_ts     ON audit_log (ts DESC);
CREATE INDEX IF NOT EXISTS idx_audit_actor  ON audit_log (actor, ts DESC);
CREATE INDEX IF NOT EXISTS idx_audit_action ON audit_log (action, ts DESC);
