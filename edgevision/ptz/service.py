"""PTZ 的邏輯：手動控制（含看門狗）、歸位點 / 中心點、失準偵測與全自動回正、定時歸位。

移植自舊系統的 PTZ 服務（motion / homing / drift / scheduler），行為與參數不變。
相機永遠固定在一個視角：推論的 ROI（cameras.station_params）是對著那個視角設的，相機被撥歪 ROI 就失效。

失準偵測原理：
  歸位時存一張「基準小圖」+ 當時座標。之後定時抓目前畫面，用 FFT 相位相關求整張畫面平移量。
  「相機自認在基準座標，畫面卻大幅平移」→ 座標在說謊 = 被外力扳動。
  連續數次確認、近期無人操作 → 先視覺對齊（相對移動閉環）；對不上才原生馬達校準 + 回歸位點 + 再對齊。
"""

from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from edgevision.ptz import settings as S
from edgevision.ptz import vision
from edgevision.ptz.devices import DevicePool, calibrate_motor

logger = logging.getLogger(__name__)


class PtzService:
    def __init__(self, store, objects, *, pool: DevicePool | None = None, grab=vision.grab_gray,
                 calibrate=calibrate_motor, sleep=time.sleep) -> None:
        self.store = store
        self.objects = objects
        self.pool = pool or DevicePool(store)
        self._grab = grab
        self._calibrate = calibrate
        self._sleep = sleep
        # 共用狀態（舊系統 PTZ 程式）
        self.last_control: dict[str, float] = {}      # 最近人工操作時間 → 排程 / 失準偵測避讓
        self.calibrating: set[str] = set()            # 正在回正的相機
        self.drift_state: dict[str, dict] = {}        # cam → {count, state, dx, dy, shift, conf, ts}
        self.lock = threading.Lock()
        self._watchdogs: dict[str, threading.Timer] = {}
        self._wd_lock = threading.Lock()
        self._refs: dict[str, tuple[str, object]] = {}   # cam → (object_key, ndarray) 基準小圖快取
        self._stop = threading.Event()

    # ── 手動控制 ─────────────────────────────────────────

    def mark_control(self, cam_id: str) -> None:
        self.last_control[cam_id] = time.time()

    def _dev(self, cam_id: str):
        return self.pool.get(cam_id)

    def _call(self, cam_id: str, fn):
        """ONVIF 呼叫失敗時丟掉快取的連線再試一次（相機重開機後舊 session 會失效）。"""
        try:
            return fn(self._dev(cam_id))
        except LookupError:
            raise
        except Exception as exc:
            logger.warning("[ONVIF] %s 呼叫失敗，重建連線再試一次：%s", cam_id, exc)
            self.pool.drop(cam_id)
            return fn(self._dev(cam_id))

    def move(self, cam_id: str, pan: float, tilt: float, zoom: float) -> None:
        """連續移動 + 看門狗：即使 stop 遺失或亂序，最多動 MOVE_WATCHDOG_SEC 秒就被強制停。"""
        self.mark_control(cam_id)
        self._call(cam_id, lambda d: d.continuous_move(pan, tilt, zoom))
        self._arm_watchdog(cam_id)

    def stop(self, cam_id: str) -> None:
        self._cancel_watchdog(cam_id)
        self._call(cam_id, lambda d: d.stop())

    def relative_move(self, cam_id: str, dx: float, dy: float, dz: float = 0.0, *, manual: bool = True) -> None:
        if manual:
            self.mark_control(cam_id)
        self._call(cam_id, lambda d: d.relative_move(dx, dy, dz))

    def absolute_move(self, cam_id: str, pan: float, tilt: float, zoom: float, *, manual: bool = True) -> None:
        if manual:
            self.mark_control(cam_id)
        self._call(cam_id, lambda d: d.absolute_move(pan, tilt, zoom))

    def status(self, cam_id: str):
        return self._call(cam_id, lambda d: d.status())

    def presets(self, cam_id: str) -> list[dict]:
        return self._call(cam_id, lambda d: d.presets())

    def set_preset(self, cam_id: str, name: str) -> str:
        return self._call(cam_id, lambda d: d.set_preset(name))

    def goto_preset(self, cam_id: str, index: int) -> str:
        self.mark_control(cam_id)
        return self._call(cam_id, lambda d: d.goto_preset(index))

    def _arm_watchdog(self, cam_id: str) -> None:
        def fire():
            try:
                self._call(cam_id, lambda d: d.stop())
                logger.info("[看門狗] %s 逾時未收到 stop，已自動停止", cam_id)
            except Exception as exc:
                logger.warning("[看門狗] %s 自動停止失敗：%s", cam_id, exc)

        with self._wd_lock:
            old = self._watchdogs.pop(cam_id, None)
            if old:
                old.cancel()
            t = threading.Timer(S.MOVE_WATCHDOG_SEC, fire)
            t.daemon = True
            self._watchdogs[cam_id] = t
            t.start()

    def _cancel_watchdog(self, cam_id: str) -> None:
        with self._wd_lock:
            t = self._watchdogs.pop(cam_id, None)
        if t:
            t.cancel()

    # ── 歸位點 / 中心點 ──────────────────────────────────

    def save_position(self, cam_id: str, kind: str, *, actor: str) -> dict:
        """把相機目前座標存成 target（歸位點）或 origin（中心點；中心點不記 zoom）。"""
        p = self.status(cam_id)
        zoom = p.zoom if kind == "target" else 0.0
        self.store.upsert(cam_id, kind, pan=p.pan, tilt=p.tilt, zoom=zoom, actor=actor)
        return {"pan": p.pan, "tilt": p.tilt, "zoom": zoom}

    def goto_position(self, cam_id: str, kind: str) -> dict:
        pos = self.store.position(cam_id, kind)
        if pos is None:
            raise LookupError(f"{cam_id} 尚未設定{'歸位點' if kind == 'target' else '中心點'}")
        zoom = pos["zoom"] if kind == "target" else 0.0
        self.absolute_move(cam_id, pos["pan"], pos["tilt"], zoom)
        return {"pan": pos["pan"], "tilt": pos["tilt"], "zoom": zoom}

    # ── 失準偵測：基準 ───────────────────────────────────

    def save_reference(self, cam_id: str, *, actor: str) -> dict:
        """抓目前畫面 + 目前座標存成監控基準（請在相機位於正確歸位視角時做）。
        灰階小圖存 MinIO（每次新 key，舊的留著可追），座標與 key 存 DB。"""
        ref = self._grab(cam_id)
        pan = tilt = None
        try:
            p = self.status(cam_id)
            if p.has_pan_tilt:
                pan, tilt = p.pan, p.tilt
        except Exception as exc:
            logger.warning("[失準] %s 存基準時讀不到座標：%s", cam_id, exc)
        data = vision.dump_ref(ref)
        key = f"{S.REF_PREFIX}/{cam_id}/{time.strftime('%Y%m%d_%H%M%S')}.npz"
        import io
        self.objects.put(bucket=S.REF_BUCKET, key=key, data=io.BytesIO(data), length=len(data),
                         content_type="application/octet-stream")
        self.store.upsert(cam_id, "reference", pan=pan if pan is not None else 0.0,
                          tilt=tilt if tilt is not None else 0.0, has_coord=pan is not None,
                          ref_object_key=key, actor=actor)
        self._refs[cam_id] = (key, ref)
        self._set_state(cam_id, count=0, state="monitoring", dx=0, dy=0, shift=0.0, conf=None)
        return {"pan": pan, "tilt": tilt, "ref_object_key": key}

    def load_reference(self, cam_id: str) -> dict | None:
        pos = self.store.position(cam_id, "reference")
        if not pos or not pos.get("ref_object_key"):
            return None
        key = pos["ref_object_key"]
        cached = self._refs.get(cam_id)
        if cached and cached[0] == key:
            arr = cached[1]
        else:
            arr = vision.load_ref(self.objects.get(bucket=S.REF_BUCKET, key=key))
            self._refs[cam_id] = (key, arr)
        return {"ref": arr, "pan": float(pos["pan"] or 0.0), "tilt": float(pos["tilt"] or 0.0),
                "has_coord": bool(pos["has_coord"])}

    # ── 失準偵測：量測 ───────────────────────────────────

    def _set_state(self, cam_id: str, **kw) -> None:
        with self.lock:
            st = self.drift_state.setdefault(cam_id, {})
            st.update(kw)
            st["ts"] = time.time()

    def _coord_near(self, cam_id: str, pan: float, tilt: float, tol: float = S.DRIFT_ATHOME_TOL) -> bool:
        """目前座標是否在 (pan, tilt) 附近。讀不到座標時保守回 True（寧可不動，交給相位相關判斷）。"""
        try:
            p = self.status(cam_id)
            if not p.has_pan_tilt:
                return True
            return abs(p.pan - pan) <= tol and abs(p.tilt - tilt) <= tol
        except Exception:
            return True

    def check_drift(self, cam_id: str, *, force: bool = False) -> dict:
        """只量測（不移動相機）。force=True 忽略「最近有人操作 / 不在家」的跳過條件。"""
        ref = self.load_reference(cam_id)
        if ref is None:
            return {"ok": False, "reason": "no_reference"}
        if not force:
            if time.time() - self.last_control.get(cam_id, 0) < S.DRIFT_CONTROL_QUIET:
                return {"ok": True, "skipped": "recent_control"}
            if ref["has_coord"] and not self._coord_near(cam_id, ref["pan"], ref["tilt"]):
                return {"ok": True, "skipped": "not_at_home"}
        try:
            cur = self._grab(cam_id)
        except Exception as exc:
            return {"ok": False, "reason": f"grab_failed:{exc}"}
        dx, dy, conf = vision.phase_corr(ref["ref"], cur)
        shift = round((dx * dx + dy * dy) ** 0.5, 1)
        return {"ok": True, "dx": dx, "dy": dy, "shift": shift, "conf": round(conf, 1),
                "drift": shift >= S.DRIFT_SHIFT_PX}

    # ── 失準偵測：回正 ───────────────────────────────────

    def _home_destination(self, cam_id: str):
        """回正的目的地：優先歸位點（使用者手設的穩定錨點），其次中心點。不用會漂的基準座標當目的地。"""
        t = self.store.position(cam_id, "target")
        if t:
            return {"pan": t["pan"], "tilt": t["tilt"], "zoom": t["zoom"]}, "target"
        o = self.store.position(cam_id, "origin")
        if o:
            return {"pan": o["pan"], "tilt": o["tilt"], "zoom": 0.0}, "origin"
        return None, None

    def _update_reference_coord(self, cam_id: str) -> None:
        """回正成功後：基準小圖不變，基準座標更新成目前座標（視覺回正用相對移動補回畫面，座標會偏離原基準）。"""
        ref = self.store.position(cam_id, "reference")
        if not ref:
            return
        try:
            p = self.status(cam_id)
            if p.has_pan_tilt:
                self.store.upsert(cam_id, "reference", pan=p.pan, tilt=p.tilt, has_coord=True,
                                  ref_object_key=ref["ref_object_key"], actor="ptz:auto")
                logger.info("[失準] %s 已更新基準座標 → (%.3f, %.3f)", cam_id, p.pan, p.tilt)
        except Exception as exc:
            logger.warning("[失準] %s 更新基準座標失敗：%s", cam_id, exc)

    def _visual_align(self, cam_id: str) -> dict | None:
        """視覺閉環：反覆抓圖、依位移換算相對移動，直到對齊或迭代用盡。"""
        ref = self.load_reference(cam_id)
        if ref is None:
            return None
        res = None
        for i in range(S.DRIFT_ALIGN_MAX_ITER):
            try:
                cur = self._grab(cam_id)
            except Exception as exc:
                return {"shift": 9999.0, "error": str(exc), "iters": i}
            dx, dy, conf = vision.phase_corr(ref["ref"], cur)
            shift = (dx * dx + dy * dy) ** 0.5
            res = {"dx": dx, "dy": dy, "shift": round(shift, 1), "conf": round(conf, 1), "iters": i + 1}
            if shift <= S.DRIFT_ALIGN_OK_PX:
                return res
            if conf < S.DRIFT_ALIGN_MIN_CONF:
                res["low_conf"] = True        # 峰值不可信 → 不在雜訊上移動（免發散、免錨點污染）
                return res
            cpan = max(-0.25, min(0.25, dx / S.DRIFT_ALIGN_GAIN_X))
            ctilt = max(-0.25, min(0.25, dy / S.DRIFT_ALIGN_GAIN_Y))
            self.relative_move(cam_id, cpan, ctilt, 0.0, manual=False)
            self._sleep(2.0)
        return res

    def _wait_motion_stable(self, cam_id: str, *, timeout: float = 60, poll: float = 2.0,
                            stable_needed: int = 3, eps: float = 0.004) -> bool:
        """輪詢座標直到連續數次幾乎不動（等原生校準掃描結束 / AbsoluteMove 到位）。"""
        last, stable, t0 = None, 0, time.time()
        while time.time() - t0 < timeout:
            self._sleep(poll)
            try:
                p = self.status(cam_id)
                cur = (p.pan, p.tilt) if p.has_pan_tilt else None
            except Exception:
                cur = None
            if cur and last and abs(cur[0] - last[0]) < eps and abs(cur[1] - last[1]) < eps:
                stable += 1
                if stable >= stable_needed:
                    return True
            else:
                stable = 0
            last = cur
        return False

    @staticmethod
    def _res_fields(res: dict) -> dict:
        return {k: res.get(k) for k in ("dx", "dy", "shift", "conf") if k in res}

    def auto_recover(self, cam_id: str) -> None:
        """⓪ 座標不在歸位點 → 先 AbsoluteMove 回去（相位相關只在畫面重疊時有效）
           ① 視覺對齊（免掃描，小 / 中偏移一兩次到位）
           ①b 低信心 = 這次量測不可信（畫面有大面積移動物），放掉這輪，不觸發馬達掃描
           ② 對不上 → 原生校準 → 回歸位點 → 再視覺對齊 + 複驗"""
        with self.lock:
            if cam_id in self.calibrating:
                return
            self.calibrating.add(cam_id)
        try:
            self._set_state(cam_id, state="recovering")
            dest, kind = self._home_destination(cam_id)
            if dest and not self._coord_near(cam_id, dest["pan"], dest["tilt"]):
                logger.info("[失準] %s 座標不在歸位點（%s），先 AbsoluteMove 回去再微調", cam_id, kind)
                self._set_state(cam_id, state="homing")
                self.absolute_move(cam_id, dest["pan"], dest["tilt"], dest["zoom"], manual=False)
                self._wait_motion_stable(cam_id, timeout=15)
            res = self._visual_align(cam_id)
            if res and res.get("shift", 9999) <= S.DRIFT_SHIFT_PX and not res.get("low_conf"):
                if res.get("conf", 0) >= S.DRIFT_REF_UPDATE_MIN_CONF:
                    self._update_reference_coord(cam_id)
                self._set_state(cam_id, state="recovered", count=0, **self._res_fields(res))
                logger.info("[失準] %s 視覺對齊完成：%s", cam_id, res)
                return
            if res and res.get("low_conf"):
                self._set_state(cam_id, state="low_conf_hold", count=0, **self._res_fields(res))
                logger.info("[失準] %s 信心不足（%s），判定為畫面雜訊，不做原生校準", cam_id, res.get("conf"))
                return
            logger.info("[失準] %s 視覺對齊不足（%s），改走原生校準", cam_id, res)
            cam = self.store.camera(cam_id)
            self._calibrate(cam)
            self._sleep(5)
            self._wait_motion_stable(cam_id, timeout=S.DRIFT_RECOVER_WAIT + 30)
            dest, _ = self._home_destination(cam_id)
            if dest:
                self.absolute_move(cam_id, dest["pan"], dest["tilt"], dest["zoom"], manual=False)
                self._wait_motion_stable(cam_id, timeout=15)
                res = self._visual_align(cam_id) or self.check_drift(cam_id, force=True)
            else:
                logger.warning("[失準] %s 沒有歸位點 / 中心點可回，停在校準中心（請先設歸位點）", cam_id)
                res = self.check_drift(cam_id, force=True)
            recovered = res.get("shift", 9999) <= S.DRIFT_SHIFT_PX
            # 歸位點是使用者手設的 ground truth，回正永不覆寫（舊系統曾因此「越校越歪」）
            if recovered and res.get("conf", 0) >= S.DRIFT_REF_UPDATE_MIN_CONF:
                self._update_reference_coord(cam_id)
            self._set_state(cam_id, state="recovered" if recovered else "drift_unresolved", count=0,
                            **self._res_fields(res))
            logger.info("[失準] %s 校準 + 回歸位點 + 視覺對齊後：%s", cam_id, res)
        except Exception as exc:
            self._set_state(cam_id, state="recover_failed", error=str(exc)[:300])
            logger.exception("[失準] %s 自動回正失敗", cam_id)
        finally:
            with self.lock:
                self.calibrating.discard(cam_id)
            self.last_control[cam_id] = time.time()       # 回正後別立刻又檢查

    def recover_async(self, cam_id: str) -> bool:
        if cam_id in self.calibrating:
            return False
        threading.Thread(target=self.auto_recover, args=(cam_id,), name=f"ptz-recover-{cam_id}", daemon=True).start()
        return True

    def check_one(self, cam_id: str) -> None:
        """排程檢查一台：低信心的量測不計入連續失準（峰值不可信時 shift 常在門檻上緣遊走）。"""
        if cam_id in self.calibrating:
            return
        res = self.check_drift(cam_id)
        if not res.get("ok") or "skipped" in res or "reason" in res:
            return
        with self.lock:
            st = self.drift_state.setdefault(cam_id, {"count": 0})
            trusted = res["drift"] and res["conf"] >= S.DRIFT_ALIGN_MIN_CONF
            st["count"] = st.get("count", 0) + 1 if trusted else 0
            if not trusted:
                st["state"] = "monitoring" if not res["drift"] else "low_conf_hold"
            st.update(dx=res["dx"], dy=res["dy"], shift=res["shift"], conf=res["conf"], ts=time.time())
            cnt, active = st["count"], len(self.calibrating)
        if cnt >= S.DRIFT_CONSECUTIVE and time.time() - self.last_control.get(cam_id, 0) >= S.DRIFT_CONTROL_QUIET:
            if active >= S.DRIFT_MAX_RECOVERIES:
                logger.info("[失準] %s 待回正（已達同時回正上限 %d），下輪再試", cam_id, S.DRIFT_MAX_RECOVERIES)
                return
            self.recover_async(cam_id)

    # ── 背景迴圈 ─────────────────────────────────────────

    def start_background(self) -> None:
        threading.Thread(target=self._drift_loop, name="ptz-drift", daemon=True).start()
        threading.Thread(target=self._homing_loop, name="ptz-homing", daemon=True).start()
        threading.Thread(target=self._prewarm, name="ptz-prewarm", daemon=True).start()
        logger.info("PTZ 背景：每 %ds 歸位有歸位點的相機；失準偵測每台約每 %ds 一次（併發 %d、同時回正上限 %d）",
                    S.SCHEDULE_INTERVAL_SEC, S.DRIFT_CHECK_INTERVAL, S.DRIFT_WORKERS, S.DRIFT_MAX_RECOVERIES)

    def shutdown(self) -> None:
        self._stop.set()

    def _prewarm(self) -> None:
        """預熱有歸位點的相機 ONVIF 連線，避免第一次移動因冷連線慢、亂序。"""
        for cam_id in self.store.cameras_with("target"):
            try:
                self.pool.get(cam_id)
            except Exception as exc:
                logger.warning("[預熱] %s 連線失敗（第一次移動會較慢）：%s", cam_id, exc)

    def home_one(self, cam_id: str, t: dict) -> str:
        if cam_id in self.calibrating:
            return "calibrating"
        if time.time() - self.last_control.get(cam_id, 0) < S.DRIFT_CONTROL_QUIET:
            return "recent_control"
        self.absolute_move(cam_id, t["pan"], t["tilt"], t["zoom"], manual=False)
        return "homed"

    def _homing_loop(self) -> None:
        while not self._stop.wait(S.SCHEDULE_INTERVAL_SEC):
            try:
                cams = self.store.cameras_with("target")
            except Exception as exc:
                logger.warning("[排程] 讀歸位點清單失敗：%s", exc)
                continue
            for cam_id in cams:
                try:
                    t = self.store.position(cam_id, "target")
                    r = self.home_one(cam_id, t)
                    if r != "homed":
                        logger.info("[排程] %s 略過本輪（%s）", cam_id, r)
                except Exception as exc:
                    logger.warning("[排程] 歸位 %s 失敗：%s", cam_id, exc)

    def _drift_loop(self) -> None:
        """有界併發 + 錯開排程：每台平均每 DRIFT_CHECK_INTERVAL 檢查一次，分散到整個週期，同時最多 DRIFT_WORKERS 台。"""
        if self._stop.wait(25):
            return
        pool = ThreadPoolExecutor(max_workers=S.DRIFT_WORKERS)
        next_check: dict[str, float] = {}
        inflight = {"n": 0}
        ilock = threading.Lock()

        def wrap(cam_id):
            try:
                self.check_one(cam_id)
            except Exception as exc:
                logger.warning("[失準] %s 檢查錯誤：%s", cam_id, exc)
            finally:
                with ilock:
                    inflight["n"] -= 1

        while not self._stop.is_set():
            try:
                cams = self.store.cameras_with("reference")
            except Exception as exc:
                logger.warning("[失準] 讀監控相機清單失敗：%s", exc)
                cams = list(next_check)
            now = time.time()
            new = [c for c in cams if c not in next_check]
            step = S.DRIFT_CHECK_INTERVAL / max(1, len(cams))
            for i, c in enumerate(new):
                next_check[c] = now + i * step
            for c in list(next_check):
                if c not in cams:
                    del next_check[c]
            for cam_id in sorted(next_check, key=next_check.get):
                if next_check[cam_id] > now:
                    break
                if cam_id in self.calibrating:
                    next_check[cam_id] += S.DRIFT_CHECK_INTERVAL
                    continue
                with ilock:
                    if inflight["n"] >= S.DRIFT_WORKERS:
                        break                 # 飽和：留待下個 tick，不推進 → 不漏檢
                    inflight["n"] += 1
                pool.submit(wrap, cam_id)
                next_check[cam_id] += S.DRIFT_CHECK_INTERVAL
            self._stop.wait(1.0)
        pool.shutdown(wait=False)
