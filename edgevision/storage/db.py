"""PostgreSQL 存取層。

只提供本專案實際需要的操作，不做通用 ORM —— 查詢模式固定，
手寫 SQL 比 ORM 更好掌控索引使用與效能。

連線資訊從環境變數讀（DATABASE_URL），Compose 與 K8s 共用同一份程式。
"""

from __future__ import annotations

import json
import logging
import os
from contextlib import contextmanager
from typing import Any, Iterator

from libs.contracts import FrameResult

logger = logging.getLogger(__name__)


class Database:
    """PostgreSQL 連線池封裝。

    用連線池而非單一連線：落地器在背景執行緒寫入，查詢 API 在
    事件迴圈讀取，兩者並行。
    """

    def __init__(self, dsn: str | None = None, *, min_size: int = 1, max_size: int = 5) -> None:
        try:
            from psycopg_pool import ConnectionPool
        except ImportError as exc:
            raise RuntimeError("需要 psycopg[pool]：pip install 'psycopg[binary,pool]'") from exc

        self.dsn = dsn or os.environ.get("DATABASE_URL", "")
        if not self.dsn:
            raise ValueError("未設定 DATABASE_URL")

        self._pool = ConnectionPool(
            self.dsn, min_size=min_size, max_size=max_size, open=True, timeout=10,
        )
        logger.info("資料庫連線池已建立（%d-%d）", min_size, max_size)

    def close(self) -> None:
        self._pool.close()

    @contextmanager
    def cursor(self) -> Iterator[Any]:
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                yield cur

    def ping(self) -> bool:
        try:
            with self.cursor() as cur:
                cur.execute("SELECT 1")
                return cur.fetchone() is not None
        except Exception:
            return False

    # ── 寫入 ──────────────────────────────────────────────

    def insert_detections(self, result: FrameResult, *, image_uri: str | None = None) -> int:
        """把一個影格的所有偵測寫入。

        一幀多個物件 → 多列。每列都帶完整的幀身分與模型身分，
        查詢時不必 join，且單列就能追溯到來源影像。
        沒有物件但判定是 NG → 一列整幀判定（class_name / confidence / bbox 為 NULL）。
        """
        verdict = result.verdict
        is_ng = verdict is not None and verdict.status.value == "ng"
        if not result.detections and not is_ng:
            return 0                     # 沒東西、也不是 NG：不用留

        def row(d):
            return (
                result.camera_id,
                result.frame_id,
                result.marker_id,
                result.rtp_timestamp,
                result.model_name,
                result.model_version,
                d.class_name if d else None,
                d.confidence if d else None,
                json.dumps(d.bbox.model_dump()) if d else None,
                result.coord_system.value,
                json.dumps(result.image_meta.model_dump()),
                d.mask_area if d else None,
                verdict.status.value if verdict else None,
                verdict.reason if verdict else None,
                json.dumps(verdict.details) if verdict else None,
                image_uri,
                result.station,
            )

        # 一個物件一列。整幀判定的 NG（靠時間判、當下沒有框）寫一列，類別與框為 NULL ——
        # 否則它不進資料庫：前端查不到、錄好的影片也掛不上去（見 09-detections-frame-level.sql）
        rows = [row(d) for d in result.detections] if result.detections else [row(None)]

        with self.cursor() as cur:
            cur.executemany(
                """
                INSERT INTO detections (
                    camera_id, frame_id, marker_id, rtp_timestamp,
                    model_name, model_version,
                    class_name, confidence, bbox, bbox_coord_system,
                    image_meta, mask_area,
                    verdict_status, verdict_reason, verdict_details,
                    image_uri, station
                ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                """,
                rows,
            )
        return len(rows)

    def attach_clip(self, *, camera_id: str, frame_id: str, clip_uri: str,
                    event_ts: float, window_s: float = 120.0) -> int:
        """把事件影片的 URI 掛到對應的偵測列上。回傳更新了幾列。

        ⭐ 為什麼是「幾列」不是「一列」

            一次 NG 在 detections 裡是【多列】（一列 = 一個偵測物件），
            但影片是整幀的。所以同一個 frame_id 的每一列都掛同一支影片 ——
            前端查到任何一列都能播到那段影片，不必先做聚合。

        ⚠️ 為什麼要帶 camera_id 與時間窗

            frame_id 沒有索引（見 02-schema.sql），單靠它 UPDATE 會全表掃描。
            加上 camera_id + ts 範圍就能走 idx_detections_camera_ts。
            窗口給 120 秒是因為片段要等後段錄完（post_seconds + ffmpeg 收尾）
            才會上傳，寫回來時那筆 detection 已經是一兩分鐘前的事了。

        ⚠️ 回傳 0 不是錯誤

            事件可能來自抽樣沒存進 DB 的那一幀（抽樣策略只留 NG + 0.5% 背景）。
            呼叫端記 debug 就好，不要當失敗處理。
        """
        from datetime import datetime, timedelta, timezone

        centre = datetime.fromtimestamp(event_ts, tz=timezone.utc)
        lo = centre - timedelta(seconds=window_s)
        hi = centre + timedelta(seconds=window_s)
        with self.cursor() as cur:
            cur.execute(
                """
                UPDATE detections SET clip_uri = %s
                WHERE camera_id = %s AND frame_id = %s
                  AND ts BETWEEN %s AND %s
                  AND clip_uri IS NULL
                """,
                (clip_uri, camera_id, frame_id, lo, hi),
            )
            return cur.rowcount

    def write_audit(
        self,
        *,
        actor: str,
        actor_type: str,
        channel: str,
        action: str,
        result: str,
        target: str | None = None,
        params: dict | None = None,
        error: str | None = None,
        task_id: str | None = None,
        client_info: str | None = None,
    ) -> None:
        """記錄操作稽核。誰、用什麼方式、做了什麼、結果如何。"""
        with self.cursor() as cur:
            cur.execute(
                """
                INSERT INTO audit_log (
                    actor, actor_type, channel, client_info,
                    action, target, params, result, error, task_id
                ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                """,
                (actor, actor_type, channel, client_info, action, target,
                 json.dumps(params or {}), result, error, task_id),
            )

    def expire_evidence(self, *, older_than_days: float, camera_id: str | None = None) -> int:
        """超過保留天數的存證連結標成失效：clip_uri / image_uri 清成 NULL、記下 evidence_expired_at。

        紀錄本身保留（統計、追溯）。影片在 MinIO 端由 bucket 過期規則刪除，天數要跟這裡一致；
        不清的話資料庫留著指向已刪檔案的死連結。回傳這次標了幾列。可重複執行。
        camera_id：只處理這台（測試用 —— 不帶就是整張表，曾在開發機誤清 10 萬列舊連結）。
        """
        with self.cursor() as cur:
            cur.execute(
                """
                UPDATE detections
                   SET clip_uri = NULL, image_uri = NULL, evidence_expired_at = now()
                 WHERE ts < now() - make_interval(secs => %s)
                   AND (clip_uri IS NOT NULL OR image_uri IS NOT NULL)
                   AND (%s::text IS NULL OR camera_id = %s)
                """,
                (older_than_days * 86400, camera_id, camera_id),
            )
            return cur.rowcount

    # ── 查詢 ──────────────────────────────────────────────

    def recent_detections(
        self,
        *,
        camera_id: str | None = None,
        verdict: str | None = None,
        station: str | None = None,
        line_id: str | None = None,
        reviewed: bool | None = None,
        limit: int = 100,
    ) -> list[dict]:
        conditions, params = [], []
        if line_id:
            # detections 沒有 line_id 欄位：用 cameras 表對應（相機屬於哪條線）
            conditions.append("camera_id IN (SELECT id FROM cameras WHERE line_id = %s)")
            params.append(line_id)
        if reviewed is not None:
            conditions.append("reviewed = %s")
            params.append(reviewed)
        if camera_id:
            conditions.append("camera_id = %s")
            params.append(camera_id)
        if station:
            conditions.append("station = %s")
            params.append(station)
        if verdict:
            conditions.append("verdict_status = %s")
            params.append(verdict)
        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""

        with self.cursor() as cur:
            cur.execute(
                f"""
                SELECT id, ts, camera_id, frame_id, marker_id,
                       model_name, model_version, class_name, confidence,
                       bbox, bbox_coord_system, image_meta, mask_area,
                       verdict_status, verdict_reason, verdict_details,
                       image_uri, clip_uri, reviewed, review_result, reviewed_by, reviewed_at,
                       station, evidence_expired_at
                FROM detections {where}
                ORDER BY ts DESC LIMIT %s
                """,
                (*params, limit),
            )
            cols = [c.name for c in cur.description]
            return [dict(zip(cols, row)) for row in cur.fetchall()]

    def review(self, *, camera_id: str, frame_id: str, result: str, actor: str) -> int:
        """人工審核一個 NG 事件：同一幀的每一列一起標（一幀多個物件 = 多列）。回傳標了幾列。
        可重複標（改判），以最後一次為準。"""
        with self.cursor() as cur:
            cur.execute(
                """
                UPDATE detections
                   SET reviewed = TRUE, review_result = %s, reviewed_by = %s, reviewed_at = now()
                 WHERE camera_id = %s AND frame_id = %s AND verdict_status = 'ng'
                """,
                (result, actor, camera_id, frame_id),
            )
            return cur.rowcount

    def stats_summary(self, *, hours: int = 24) -> dict:
        with self.cursor() as cur:
            cur.execute(
                """
                SELECT
                    count(*)                                          AS total,
                    count(*) FILTER (WHERE verdict_status = 'ng')     AS ng,
                    count(*) FILTER (WHERE image_uri IS NOT NULL)     AS with_image,
                    count(*) FILTER (WHERE reviewed)                  AS reviewed,
                    count(DISTINCT camera_id)                         AS cameras
                FROM detections
                WHERE ts > now() - make_interval(hours => %s)
                """,
                (hours,),
            )
            cols = [c.name for c in cur.description]
            return dict(zip(cols, cur.fetchone()))
