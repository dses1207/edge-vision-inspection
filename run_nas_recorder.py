#!/usr/bin/env python3
"""24 小時連續錄影 → NAS。

⭐ 它跟 run_recorder.py（事件錄影）是兩個服務，不是一個

    分開跑的理由：這支掛 NAS。NAS 掛不上、寫滿、SMB 卡死的時候，
    不該把事件錄影一起拖下水 —— 那是 NG 存證，比 24h 原始影像重要。
    兩者唯一共用的是 edgevision/recorder/prebuffer.py 的 ffmpeg 包裝。

⚠️ 錄的是【乾淨】串流，不是 <cam>_marked

    這批影像的用途是回流訓練，有框的畫面無法當訓練資料。
    （事件影片相反 —— 那是給人看的，框燒死在畫面裡正是要的。）

⚠️ 預設不啟動

    compose 裡放在 profiles: [nas]，要 `--profile nas up -d` 才會起。
    

⚠️ 只能跑一個副本（上 K8s 後是 replicas: 1）

    load_cameras() 拿的是「config API 回的【所有】enabled 相機」，不是分片。
    兩個副本 = 兩邊都錄全部相機 → 寫進同一個 {profile}/{camera}/ 目錄、
    檔名都是時間戳 → 互相覆蓋，而且【不會有任何錯誤】。

    8 台相機走 -c copy 負載很輕，一個行程吃得下。真要多副本得先做線路分片，
    但那是為了不存在的問題增加協調複雜度。

環境變數
    CONFIG_API           必填，相機清單來源
    GO2RTC_RTSP          必填，例如 rtsp://localhost:8554
    REC_OUT_DIR          必填，NAS 掛載點（容器裡是 /out）
    REC_ROSTER           db = 錄哪幾台以資料庫的錄影名單（rec_roster，API：PUT /api/media/recordings/roster/{cam}）
                         為準，每 CAMERA_REFRESH_S 秒重讀；資料庫連不上時沿用目前在錄的（啟動時退回 REC_CAMERAS）
    REC_CAMERAS          相機篩選，逗號分隔；`siteb*` 結尾星號=前綴、其餘精確比對，
                         空=全錄。例如只錄部分線（ipcam*,siteb*）——
                         幾十台一起錄會把 NAS 容量預估整個打翻
    REC_SEGMENT_SECONDS  分段長度，預設 600（10 分鐘）
    REC_RETENTION_DAYS   保留天數，預設 0 = 不刪（與舊系統一致）
    REC_MIN_FREE_GB      低於此空間就暫停錄影，預設 20
    REC_STALL_SECONDS    輸出停滯多久算卡住，預設 90（不要調低，見 continuous.py）
    REC_REQUIRE_MOUNT    1=拒絕在非掛載點上啟動，預設 1
    CAMERA_REFRESH_S     多久重抓一次相機清單，預設 300
    STATS_INTERVAL_S     統計輸出間隔，預設 300
    REC_ALIVE_FILE       存活檔路徑（放本機、不放 NAS），預設 /tmp/nas-recorder.alive
    REC_ALIVE_WINDOW_S   這麼久內至少一台有寫入才算活著，預設 300
"""
from __future__ import annotations

import logging
import os
import signal
import sys
import threading
import time
from pathlib import Path

from edgevision.recorder.continuous import ContinuousRecorder, ContinuousSpec, is_mounted
from edgevision.recorder.prebuffer import BufferSpec

# run_recorder.py 已經有「從設定 API 拿相機清單」與「不走 proxy 的 opener」，
# 不要再寫一份 —— 兩份會漂移。
from run_recorder import load_cameras

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)-7s %(name)-24s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("nas-recorder")


def _require(name: str) -> str:
    v = os.environ.get(name, "").strip()
    if not v:
        raise SystemExit(f"必須設定 {name}")
    return v


def filter_cameras(cameras: list[str], spec: str) -> list[str]:
    """REC_CAMERAS 篩選。逗號分隔：`siteb*` 這種結尾星號是前綴，其餘精確比對。

    空字串＝不篩（維持原本「全錄」行為）。
    注意 `siteb*` 吃不到 `ipcam124`，兩組前綴要分開列。
    """
    spec = spec.strip()
    if not spec:
        return list(cameras)
    pats = [p.strip() for p in spec.split(",") if p.strip()]

    def hit(cam: str) -> bool:
        return any(cam.startswith(p[:-1]) if p.endswith("*") else cam == p
                   for p in pats)

    return [c for c in cameras if hit(c)]


def recording_alive(recorders, window_s: float, now: float | None = None) -> bool:
    """最近 window_s 秒內有沒有至少一台真的寫出東西。名單空著待命也算活著。"""
    if not recorders:
        return True
    now = time.time() if now is None else now
    return any(now - r.last_progress < window_s for r in recorders.values())


def touch_alive(path: Path) -> None:
    """更新存活檔給 livenessProbe 看。

    ⚠️ 存活檢查不准掃 NAS：/out/raw 只增不刪，2026-11-09 檔案多到 find 每次超過
       15 秒，initialDelay 900 + 5 次失敗 → 整個服務每 20 分鐘被重啟一次。
    """
    try:
        path.touch()
    except OSError:
        log.warning("無法更新存活檔 %s", path, exc_info=True)


def main() -> None:
    config_api = _require("CONFIG_API")
    go2rtc = _require("GO2RTC_RTSP").rstrip("/")
    out_root = Path(_require("REC_OUT_DIR"))

    cfg = ContinuousSpec(
        segment_seconds=int(os.environ.get("REC_SEGMENT_SECONDS", 600)),
        retention_days=int(os.environ.get("REC_RETENTION_DAYS", 0)),
        min_free_gb=float(os.environ.get("REC_MIN_FREE_GB", 20)),
        stall_seconds=float(os.environ.get("REC_STALL_SECONDS", 90)),
    )

    # ⚠️ 先檢查掛載，再做任何事。
    #
    # bind-mount 是容器啟動當下綁定的。先起容器、後掛 NAS 的話，
    # 容器看到的是主機上那個空目錄，然後把 24 小時的錄影寫進主機硬碟 ——
    # 不會有任何錯誤，直到主機磁碟爆掉。
    if not out_root.exists():
        raise SystemExit(f"{out_root} 不存在 —— NAS 沒掛上？拒絕啟動")
    if os.environ.get("REC_REQUIRE_MOUNT", "1") == "1" and not is_mounted(out_root):
        raise SystemExit(
            f"{out_root} 不是掛載點 —— NAS 沒掛上就啟動會把錄影寫進本機硬碟。"
            f"拒絕啟動。（確定要這樣請設 REC_REQUIRE_MOUNT=0）")

    cam_spec = os.environ.get("REC_CAMERAS", "")
    use_roster = os.environ.get("REC_ROSTER", "").strip().lower() == "db"
    roster_db = None
    if use_roster:
        try:
            from edgevision.storage.db import Database
            roster_db = Database()
        except Exception:
            log.exception("錄影名單：資料庫連不上，先用 REC_CAMERAS")

    def wanted() -> list[str] | None:
        """要錄的相機；None = 這次查不到（沿用現狀）。"""
        if roster_db is not None:
            try:
                from edgevision.media.recordings import roster_cameras
                return roster_cameras(roster_db)
            except Exception:
                log.exception("讀錄影名單失敗")
                return None
        return filter_cameras(load_cameras(config_api), cam_spec)

    cameras = wanted()
    if cameras is None:
        cameras = filter_cameras(load_cameras(config_api), cam_spec)
    if not cameras and not use_roster:
        raise SystemExit(
            f"REC_CAMERAS={cam_spec!r} 篩完一台不剩 —— 篩選寫錯了？" if cam_spec.strip()
            else "設定 API 沒有任何啟用的相機")
    if not cameras:
        log.warning("錄影名單是空的 —— 先待命，名單有相機就開始錄")

    profile = os.environ.get("REC_PROFILE", "720p_h264")

    def _blocked(cam: str, free_gb: float) -> None:
        log.error("[%s] 空間不足（剩 %.1f GB）—— 這台暫停錄影", cam, free_gb)

    recorders: dict[str, ContinuousRecorder] = {}
    for cam in cameras:
        recorders[cam] = _spawn(cam, go2rtc, out_root, profile, cfg, _blocked)

    log.info("=" * 64)
    log.info("  nas-recorder（24h 連續錄影）")
    log.info("  相機      %d 台：%s", len(cameras), ", ".join(cameras))
    if use_roster:
        log.info("  名單資料庫錄影名單（REC_ROSTER=db），每 %s 秒重讀", os.environ.get("CAMERA_REFRESH_S", 300))
    elif cam_spec.strip():
        log.info("  篩選      REC_CAMERAS=%s", cam_spec.strip())
    log.info("  來源      <cam>（乾淨串流，無框 —— 這批要拿來訓練）")
    log.info("  輸出      %s/%s/<cam>/YYYYMMDD_HHMMSS.mp4", out_root, profile)
    log.info("  分段      %d 秒", cfg.segment_seconds)
    log.info("  保留      %s", f"{cfg.retention_days} 天"
             if cfg.retention_days > 0 else "不刪（寫到滿，與舊系統一致）")
    log.info("  空間下限  %.0f GB（低於此暫停，不是清理）", cfg.min_free_gb)
    log.info("=" * 64)

    # ⭐ 心跳：幾台在錄、剩多少空間、停滯幾次 → /healthz services[]
    #    DB 是選用的（NAS 錄影不依賴它），連不上就沒有心跳，服務照跑。
    hb = None
    try:
        from edgevision.storage.db import Database
        from edgevision.storage.heartbeat import Heartbeat
        _db = Database()
        if _db.ping():
            def _metrics() -> dict:
                hs = [r.health() for r in recorders.values()]
                try:
                    st = os.statvfs(out_root); free_gb = round(st.f_bavail * st.f_frsize / 1e9, 1)
                except OSError:
                    free_gb = None
                return {"cameras_alive": sum(1 for h in hs if h["alive"]),
                        "cameras_total": len(hs),
                        "free_gb": free_gb,
                        "blocked": any(r.blocked for r in recorders.values()) and free_gb is not None and free_gb < cfg.min_free_gb,
                        "stalls": sum(h["stalls"] for h in hs),
                        "restarts": sum(h["restarts"] for h in hs),
                        "disk_gb": round(sum(h["disk_gb"] for h in hs), 1)}
            hb = Heartbeat(_db, service="nas-recorder")
            hb.start(_metrics)
        else:
            log.warning("資料庫無回應 —— 沒有心跳，/healthz 看不到這個服務（錄影不受影響）")
    except Exception:
        log.warning("心跳未啟動（錄影不受影響）", exc_info=True)

    stop = threading.Event()

    def _bye(_s, _f):
        log.info("收到停止訊號 —— 等各段寫完 moov（最多 35 秒）…")
        stop.set()
    signal.signal(signal.SIGTERM, _bye)
    signal.signal(signal.SIGINT, _bye)

    refresh_every = int(os.environ.get("CAMERA_REFRESH_S", 300))
    stats_every = int(os.environ.get("STATS_INTERVAL_S", 300))
    last_refresh = last_stats = time.time()
    alive_file = Path(os.environ.get("REC_ALIVE_FILE", "/tmp/nas-recorder.alive"))
    alive_window = float(os.environ.get("REC_ALIVE_WINDOW_S", 300))

    while not stop.is_set():
        stop.wait(10)
        now = time.time()

        if recording_alive(recorders, alive_window, now):
            touch_alive(alive_file)

        # 相機清單變了就增減，不要整個重啟 —— 重啟等於所有相機都缺一段
        if refresh_every > 0 and now - last_refresh >= refresh_every:
            last_refresh = now
            try:
                got = wanted()
            except Exception:
                log.exception("重抓相機清單失敗，沿用現有 %d 台", len(recorders))
                got = None
            current = set(recorders) if got is None else set(got)
            for cam in current - set(recorders):
                log.info("[%s] 新增相機 —— 開始錄影", cam)
                recorders[cam] = _spawn(cam, go2rtc, out_root, profile, cfg, _blocked)
            for cam in set(recorders) - current:
                log.info("[%s] 相機已停用 —— 停止錄影", cam)
                recorders.pop(cam).stop()

        if stats_every > 0 and now - last_stats >= stats_every:
            last_stats = now
            total = sum(r.health()["disk_gb"] for r in recorders.values())
            dead = [c for c, r in recorders.items() if not r.alive]
            log.info("錄影中 %d/%d 台｜已寫 %.1f GB｜停滯 %d 次%s",
                     len(recorders) - len(dead), len(recorders), total,
                     sum(r.stalls for r in recorders.values()),
                     f"｜⚠️ 沒在錄：{', '.join(dead)}" if dead else "")

    log.info("停止所有錄影…")
    if hb is not None:
        hb.stop()
    for r in recorders.values():
        r.stop()
    log.info("已停止。")


def _spawn(cam: str, go2rtc: str, out_root: Path, profile: str,
           cfg: ContinuousSpec, on_blocked) -> ContinuousRecorder:
    """開一台相機的連續錄影。

    ⭐ segment_wrap=0 是切換鍵：BufferSpec 看到它就切成連續模式
       （不覆寫、檔名帶時間、加 -use_wallclock_as_timestamps）。
    """
    buf = BufferSpec(
        camera_id=cam,
        rtsp_url=f"{go2rtc}/{cam}",          # ⚠️ 乾淨串流，不是 _marked
        out_dir=out_root / profile / cam,
        segment_seconds=cfg.segment_seconds,
        segment_wrap=0,
        wallclock_timestamps=True,
    )
    rec = ContinuousRecorder(buf, cfg, on_blocked=on_blocked)
    rec.start()
    return rec


if __name__ == "__main__":
    sys.exit(main())
