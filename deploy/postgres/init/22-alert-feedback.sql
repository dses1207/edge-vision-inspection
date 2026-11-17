-- 告警回流（edgevision/media/alerts.py）：人判過的線上 NG 告警 → NAS 原始幀＋當下的框 → 標注專案圖片集的新批次。
--
-- 事件 = detections 同一台相機、同一幀、同一站的那幾列；event_id = 其中最小的 detections.id。
-- 一個事件只回流一次：這張表記結果（queued → added / skipped / failed），failed 下次會重試。
CREATE TABLE IF NOT EXISTS alert_feedback (
    event_id        BIGINT      PRIMARY KEY,
    camera_id       TEXT        NOT NULL,
    frame_id        TEXT        NOT NULL,
    station         TEXT,
    event_ts        TIMESTAMPTZ NOT NULL,
    model_name      TEXT,
    model_version   TEXT,
    review_result   TEXT,                       -- true_positive | false_positive（送出當下）
    status          TEXT        NOT NULL CHECK (status IN ('queued', 'added', 'skipped', 'failed')),
    reason          TEXT,                       -- skipped / failed 的原因
    collection_id   BIGINT,
    batch_id        BIGINT,
    item_id         BIGINT,
    item_status     TEXT,                       -- 放進圖片集時的狀態（labeled / auto / unlabeled）
    task_id         TEXT,
    align           JSONB,                      -- 怎麼對到 NAS 那一幀（lag、分數、是否有把握）
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_alert_feedback_task ON alert_feedback (task_id);
CREATE INDEX IF NOT EXISTS idx_alert_feedback_status ON alert_feedback (status, created_at DESC);

-- ⚠️ 不在 detections 加索引：大表，CREATE INDEX 會擋住線上寫入。找判過的告警只看最近 N 天，走既有的 ts 索引。

-- 批次多一種來源：告警回流
ALTER TABLE media_batches DROP CONSTRAINT IF EXISTS media_batches_kind_check;
ALTER TABLE media_batches ADD CONSTRAINT media_batches_kind_check
    CHECK (kind IN ('upload', 'zip', 'video', 'nas', 'copy', 'alert'));
