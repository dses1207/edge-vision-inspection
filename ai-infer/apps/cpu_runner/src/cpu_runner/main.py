"""CPU 判讀室：一條線一個行程，跟 deepstream_runner 同樣的設定來源、同樣的 worker、同樣的發布與心跳。

    deepstream_runner（正式）                 cpu_runner（demo）
    ─────────────────────────                 ──────────────────────────
    RTSP → nvstreammux                         Scene.render(t)          合成畫面
    nvinferserver → Triton（GPU）              SceneDetector            真值 → YOLO 原始輸出張量
    probe → TensorJob → worker ─────────────── 同一個 ds_runner.worker.Worker（後處理 → 站別 → FrameResult）
    publish → Redis Streams ────────────────── 同一個 ds_runner.publish
    nvdsosd 燒錄框 → go2rtc <cam>_marked       OpenCV 畫框 → ffmpeg 推 go2rtc <cam>_marked
    LineReporter 心跳 ──────────────────────── 同一個 ds_runner.status.LineReporter

設定改了（config_version 變）就結束行程，交給容器重啟重新載入 —— 跟正式環境的 runner 調和器同一個語意：
設定是宣告式的，runner 不做就地熱改以外的事。

環境變數：
  CAMERA_SOURCE / CONFIG_API / LINE_ID / CAMERA_FILE   同 deepstream_runner
  REDIS_URL                                            結果發布
  GO2RTC_RTSP          推流目標，預設 rtsp://go2rtc:8554（沒設 = 不推流，只判定）
  DEMO_SCENE           zone | dock；沒設用 line_params.demo_scene，再沒有用 zone
  DEMO_FPS             覆寫 line_params.target_fps（預設 10）
  CONFIG_POLL_S        多久檢查一次設定版本，預設 15
"""

from __future__ import annotations

import json
import logging
import os
import signal
import subprocess
import threading
import time
import urllib.request

import cv2
import numpy as np
from ds_runner.config import LineConfig, line_url, load_line_config
from ds_runner.publish import make_publisher
from ds_runner.status import LineReporter
from ds_runner.worker import TensorJob, Worker
from edgevision_contracts import VerdictStatus
from edgevision_postprocess import compute_letterbox, draw_detections

from cpu_runner.detector import SceneDetector
from cpu_runner.scene import Scene

logger = logging.getLogger("cpu_runner")


class RtspPush:
    """一路 ffmpeg：stdin 吃 BGR 原始幀 → H.264 → RTSP 推到 go2rtc。ffmpeg 掛了就重開，不拖垮判定。"""

    def __init__(self, url: str, width: int, height: int, fps: int) -> None:
        self.url, self.w, self.h, self.fps = url, width, height, fps
        self.proc: subprocess.Popen | None = None
        self._next_retry = 0.0

    def _start(self) -> None:
        cmd = ["ffmpeg", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
               "-s", f"{self.w}x{self.h}", "-r", str(self.fps), "-i", "-",
               "-c:v", "libx264", "-preset", "ultrafast", "-tune", "zerolatency", "-pix_fmt", "yuv420p",
               "-g", str(self.fps * 2), "-f", "rtsp", "-rtsp_transport", "tcp", self.url]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        logger.info("推流 → %s", self.url)

    def write(self, frame: np.ndarray) -> None:
        if self.proc is None or self.proc.poll() is not None:
            if time.monotonic() < self._next_retry:
                return
            self._next_retry = time.monotonic() + 5
            self._start()
        try:
            self.proc.stdin.write(frame.tobytes())
        except (BrokenPipeError, OSError):
            logger.warning("推流中斷：%s（5 秒後重試）", self.url)
            self.proc = None

    def close(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.stdin.close()
            self.proc.terminate()


def banner(frame: np.ndarray, verdicts: dict[str, object]) -> np.ndarray:
    """右上角列出各站別這一幀的判定；NG 用紅底。"""
    y = 20
    for station, v in verdicts.items():
        ng = v is not None and v.status == VerdictStatus.NG
        text = f"{station}: {'NG ' + v.reason if ng else 'OK'}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        x = frame.shape[1] - tw - 24
        cv2.rectangle(frame, (x - 8, y - 4), (x + tw + 8, y + th + 10), (0, 0, 200) if ng else (40, 120, 40), -1)
        cv2.putText(frame, text, (x, y + th + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y += th + 22
    return frame


def remote_version(env: dict[str, str]) -> int | None:
    if env.get("CAMERA_SOURCE") != "api":
        return None
    try:
        with urllib.request.urlopen(line_url(env["CONFIG_API"], env["LINE_ID"]), timeout=5) as r:
            return int(json.load(r).get("config_version", 0))
    except Exception:   # noqa: BLE001 —— 查不到就下次再查，不影響判定
        return None


class CpuRunner:
    def __init__(self, cfg: LineConfig, env: dict[str, str]) -> None:
        self.cfg, self.env = cfg, env
        self.fps = int(env.get("DEMO_FPS") or cfg.line_params.get("target_fps") or 10)
        kind = env.get("DEMO_SCENE") or cfg.line_params.get("demo_scene") or "zone"
        self.publish = make_publisher(env, line_id=cfg.id, station=cfg.station)
        self.worker = Worker(cfg, publish=self.publish, model_version=str(cfg.resolve_model().version or "demo"))
        self.bindings = cfg.bindings()
        w, h = cfg.frame_size
        self.scenes = {c.id: Scene(c.id, w, h, kind=kind, zone=c.station_params.get("zone"))
                       for c in cfg.active_cameras}
        self.detectors = {(c.id, b.key): SceneDetector(b.model.classes, seed=f"{c.id}/{b.key}")
                          for c in cfg.active_cameras for b in self.bindings}
        self.letterbox = {b.key: compute_letterbox(w, h, b.model.imgsz) for b in self.bindings}
        rtsp = env.get("GO2RTC_RTSP", "rtsp://go2rtc:8554").rstrip("/")
        self.push = {} if not rtsp else {
            (c.id, s): RtspPush(f"{rtsp}/{c.id}{s}", w, h, self.fps) for c in cfg.active_cameras for s in ("", "_marked")}
        self.hidden = {b.key: set(b.model.hidden_classes) for b in self.bindings}
        self.last = {(c.id, b.key): None for c in cfg.active_cameras for b in self.bindings}
        self.frames = 0
        self.reporter = LineReporter(env.get("CONFIG_API") if env.get("CAMERA_SOURCE") == "api" else None,
                                     cfg.id, cfg.config_version,
                                     stats_provider=lambda: {"cameras": {c: float(self.fps) for c in self.scenes}})

    def _interval(self, b) -> int:
        return (b.interval if b.interval is not None else 0) if not b.primary else 0

    def step(self, t: float) -> None:
        for cam, scene in self.scenes.items():
            frame = scene.render(t)
            objs = scene.objects(t)
            for b in self.bindings:
                if self.frames % (self._interval(b) + 1):
                    continue
                out0 = self.detectors[(cam, b.key)](objs, self.letterbox[b.key])
                job = TensorJob(cam, out0, None, scene.w, scene.h, image=frame,
                                station=None if b.primary else b.key)
                try:
                    result = self.worker.process(job)
                except Exception:
                    logger.exception("[%s] %s 判定失敗", cam, b.key)
                    continue
                self.last[(cam, b.key)] = result
                self.publish(result, frame)
                if result.verdict and result.verdict.status == VerdictStatus.NG:
                    logger.info("NG %s/%s：%s", cam, b.key, result.verdict.reason)
            if self.push:
                marked = frame
                for b in self.bindings:
                    r = self.last[(cam, b.key)]
                    if r is not None:
                        dets = [d for d in r.detections if d.class_name not in self.hidden[b.key]]
                        marked = draw_detections(marked, dets, draw_masks=False)
                marked = banner(marked, {b.key: (self.last[(cam, b.key)] or None) and self.last[(cam, b.key)].verdict
                                         for b in self.bindings})
                self.push[(cam, "")].write(frame)
                self.push[(cam, "_marked")].write(marked)
        self.frames += 1

    def run(self, stop: threading.Event) -> None:
        self.reporter.start(len(self.scenes))
        poll = float(self.env.get("CONFIG_POLL_S", 15))
        t0 = next_poll = time.monotonic()
        period = 1.0 / self.fps
        logger.info("線路 %s 開跑：站別 %s，相機 %s，%d fps", self.cfg.id,
                    [b.key for b in self.bindings], sorted(self.scenes), self.fps)
        while not stop.is_set():
            now = time.monotonic()
            self.step(now - t0)
            if now >= next_poll:
                next_poll = now + poll
                v = remote_version(self.env)
                if v is not None and v != self.cfg.config_version:
                    logger.info("設定版本 %d → %d，結束行程讓容器重啟後重新載入", self.cfg.config_version, v)
                    break
            stop.wait(max(0.0, period - (time.monotonic() - now)))
        self.reporter.stop()
        for p in self.push.values():
            p.close()


def main() -> None:
    logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"),
                        format="%(asctime)s %(levelname)-7s %(name)s %(message)s", datefmt="%H:%M:%S")
    env = dict(os.environ)
    while True:                      # 設定 API 還沒起來 / 線路還沒建好：等，不要一直被重啟
        try:
            cfg = load_line_config(env)
            break
        except Exception as e:   # noqa: BLE001 —— 任何原因都等一下再試，交給 log 說明
            logger.warning("載入線路設定失敗，10 秒後重試：%s", e)
            time.sleep(10)
    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    CpuRunner(cfg, env).run(stop)


if __name__ == "__main__":
    main()
