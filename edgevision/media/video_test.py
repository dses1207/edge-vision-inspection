"""影片測試：一段影片 ＋ 一個模型（版本）＋ 一條線的判定邏輯 → 打好框的影片、每幀結果、NG 清單。

兩種模式（mode）：
    logic（預設）  邏輯＋模型：照影片實際速度播（站別的「連續 N 秒 / 冷卻」用牆鐘，播快了判定就跟線上不一樣），出 NG 清單
    model          純模型：能跑多快跑多快、一幀都不丟（runner VIDEO_PACE=0），不採用站別判定，
                   改出各類別的框數 / 出現幀數 / 平均信心值 —— 比兩版模型看得到什麼、哪裡漏框

    選影片圖片庫的影片來源（media_sources：上傳的影片 / NAS 錄影），或直接給 NAS 相機 + 時間段
    選模型  Registry 的任何一版（不給 = 現役版）；只跑主站 —— 線上掛的附加站（例如 ppe）不帶進測試
    看結果打框 mp4、每幀 JSONL、NG 清單與摘要（存 MinIO experiments/video-tests/<任務>/，每次一份，不覆蓋）

怎麼跑（GPU 任務，跟訓練 / 自動標注共用排隊；Orchestrator._run_video_test）：
    ① prepare   edgevision 映像切出影片、取線路設定（只留一台相機、換成指定的模型版本）、把 engine 放進臨時 repo
    ② triton    Triton 映像臨時服務，只載這次要用的模型版本（不碰線上的 Triton）
    ③ runner    runner 映像檔案模式（VIDEO_FILE）：跟線上同一份推論 / 判定 / 燒框程式；logic 照影片實際速度播，
                             model 不限速
    ④ finalize  edgevision 映像上傳結果、算摘要、寫進任務結果

⭐ 為什麼用線上的 runner 而不是另寫一套 Python 推論：前處理、ROI 裁切、推論間隔、後處理、燒框都跟線上一樣，
   測出來的就是上線後會看到的。logic 照實際速度播是因為站別的「連續 N 秒 / 冷卻」用牆鐘。
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import urllib.parse
import urllib.request
from collections import Counter
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

RESULT_BUCKET = "experiments"
RESULT_PREFIX = "video-tests"
WORK_SUBDIR = "video-tests"
# 影片長度上限（兩種模式一樣）：10/21 使用者定 10 分鐘（一小時太長；打框影片一段 1 小時約 800 MB，曾把 MinIO 撐滿）。
# NAS 錄影起點對齊到前一個關鍵幀，實際會多幾秒 → 多給 MAX_DURATION_SLACK_S 秒寬限。
MAX_DURATION_S = float(os.environ.get("VIDEO_TEST_MAX_S", "600"))
MAX_DURATION_SLACK_S = 15
MODES = ("logic", "model")

# mode=model 不用線上的判定邏輯：換成一個「什麼都不判、不要 ROI、不要求任何類別」的站別，
# 純看模型框。借 ppe_check 清空 violations（classes_used = 空集合、camera_params_required 空），
# 不用另做站別、runner 映像不必改。否則線路還沒畫 ROI / 站別類別跟新模型對不上時，模型都測不了。
MODEL_ONLY_STATION = "ppe_check"
MODEL_ONLY_PARAMS = {"thresholds": {"violations": {}, "min_conf": {}}, "roi_regions": []}

EXIT_OK, EXIT_ERROR = 0, 1


def work_dir(root: Path, task_id: str) -> Path:
    return root / WORK_SUBDIR / task_id


def result_prefix(task_id: str) -> str:
    return f"{RESULT_PREFIX}/{task_id}"


# ── ① prepare ───────────────────────────────────────────

def concat_list(parts: list[tuple[Path, float, float | None]]) -> str:
    """ffmpeg concat 清單：每段一個檔，inpoint / outpoint 是那個檔內的秒數。純函式，可測。"""
    lines = []
    for path, start_s, end_s in parts:
        lines.append("file '" + str(path).replace("'", "'\\''") + "'")
        if start_s > 0:
            lines.append(f"inpoint {start_s:.3f}")
        if end_s is not None:
            lines.append(f"outpoint {end_s:.3f}")
    return "\n".join(lines) + "\n"


def has_moov(path: Path) -> bool:
    """mp4 有沒有索引（moov）。只讀每個頂層 box 的檔頭（seek），不讀內容，NAS 上也便宜。

    沒有 moov 的段：錄影被 SIGKILL 打斷的壞檔，或最新那段 ffmpeg 還在寫。ffmpeg 開不了，
    一段混進 concat 整個任務就失敗（2026-11-03 cam137 14:39）。
    """
    import struct
    try:
        size = path.stat().st_size
        with path.open("rb") as f:
            pos = 0
            while pos + 8 <= size:
                f.seek(pos)
                head = f.read(16)
                if len(head) < 8:
                    return False
                box, kind = struct.unpack(">I4s", head[:8])
                if box == 1 and len(head) == 16:
                    box = struct.unpack(">Q", head[8:16])[0]
                elif box == 0:          # 延伸到檔尾
                    box = size - pos
                if box < 8:
                    return False
                if kind == b"moov":
                    return pos + box <= size      # moov 寫到一半也不算
                pos += box
    except OSError:
        return False
    return False


def cut_video(parts: list[tuple[Path, float, float | None]], dst: Path) -> None:
    """切出 / 接起一段（不重新編碼；起點對齊到前一個關鍵幀）。"""
    lst = dst.with_suffix(".txt")
    lst.write_text(concat_list(parts), encoding="utf-8")
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(lst),
           "-map", "0:v:0", "-c", "copy", "-movflags", "+faststart", str(dst)]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=1800, check=False)
    if r.returncode != 0 or not dst.exists() or dst.stat().st_size == 0:
        raise RuntimeError(f"切影片失敗：{r.stderr.strip()[-500:]}")


def resolve_parts(lib, p: dict, nas_root: Path) -> tuple[list[tuple[Path, float, float | None]], str | None, dict]:
    """要測的影片 → ([(檔案, 起, 迄)], 相機, 描述)。上傳的影片會先下載到 dl_dir。"""
    from edgevision.media.frames import LOCAL_TZ, nas_segments

    if p.get("source_id"):
        with lib.db.cursor() as cur:
            cur.execute("SELECT id, kind, uri, blob_sha256, camera_id, started_at, params FROM media_sources WHERE id = %s",
                        (int(p["source_id"]),))
            row = cur.fetchone()
        if row is None:
            raise ValueError(f"圖片庫沒有影片來源 {p['source_id']}")
        sid, kind, uri, sha, cam, started_at, sparams = row
        sparams = sparams or {}
        start_s = float(p["start_s"]) if p.get("start_s") is not None else float(sparams.get("start_s") or 0.0)
        end_s = p.get("end_s") if p.get("end_s") is not None else sparams.get("end_s")
        if kind in ("upload_video", "recording_clip"):
            if not sha:
                raise ValueError(f"影片來源 {sid} 沒有保存原檔")
            blob = lib.blob(sha)
            local = Path(p["_dl_dir"]) / Path(blob["object_key"]).name
            lib.objects.download(bucket=blob["bucket"], key=blob["object_key"], dest=local)
        elif kind == "nas_recording" and uri.startswith("nas://raw/"):
            local = nas_root / uri.removeprefix("nas://raw/")
            if not local.exists():
                raise ValueError(f"NAS 上找不到 {uri}（可能已超過保留期）")
        else:
            raise ValueError(f"影片來源 {sid} 是 {kind}，不是影片")
        desc = {"source_id": sid, "kind": kind, "uri": uri, "started_at": started_at.isoformat() if started_at else None}
        return [(local, start_s, float(end_s) if end_s is not None else None)], cam, desc

    if p.get("camera_id") and p.get("start") and p.get("end"):
        cam = p["camera_id"]
        start = datetime.fromisoformat(p["start"])
        end = datetime.fromisoformat(p["end"])
        start = start if start.tzinfo else start.replace(tzinfo=LOCAL_TZ)
        end = end if end.tzinfo else end.replace(tzinfo=LOCAL_TZ)
        if end <= start:
            raise ValueError("end 要晚於 start")
        segs = nas_segments(nas_root, cam, start, end)
        if not segs:
            raise ValueError(f"NAS 上 {cam} 在 {start}–{end} 沒有錄影")
        skipped = [path.name for path, _ in segs if not has_moov(path)]
        segs = [(path, t0) for path, t0 in segs if path.name not in skipped]
        if not segs:
            raise ValueError(f"NAS 上 {cam} 在 {start}–{end} 的錄影都打不開（壞檔或還在寫）：{', '.join(skipped)}")
        if skipped:
            logger.warning("跳過打不開的錄影段（壞檔或還在寫）：%s", ", ".join(skipped))
        parts = []
        for i, (path, t0) in enumerate(segs):
            s = max(0.0, (start - t0).total_seconds())
            e = (end - t0).total_seconds()
            nxt = segs[i + 1][1] if i + 1 < len(segs) else None
            parts.append((path, s, None if nxt is not None and end >= nxt else e))
        return parts, cam, {"camera_id": cam, "start": start.isoformat(), "end": end.isoformat(),
                            "segments": [s.name for s, _ in segs], "skipped_segments": skipped}

    raise ValueError("要給 source_id（圖片庫的影片），或 NAS 的 camera_id + start + end")


def fetch_line(line_id: str, base: str | None = None) -> dict:
    base = (base or os.environ.get("CONFIG_API", "http://output-api:18080")).rstrip("/")
    with urllib.request.urlopen(f"{base}/api/config/lines/{urllib.parse.quote(line_id, safe='')}", timeout=10) as r:
        return json.load(r)


def model_block(model: dict | None, meta, version: int) -> dict:
    """線路設定的 model 區塊：保留部署參數（conf / iou…），模型屬性換成指定版本的。"""
    keep = {k: v for k, v in (model or {}).items() if k not in ("classes", "imgsz", "task", "version")}
    return {**keep, "name": meta.model_name, "version": version, "task": meta.task, "imgsz": meta.imgsz,
            "classes": list(meta.class_names)}


def build_line(line: dict, *, camera_id: str | None, model_name: str | None, model_version: int | None,
               registry, prefer_camera: str | None = None, station: str | None = None,
               station_params: dict | None = None, mode: str = "logic") -> tuple[dict, list[tuple[str, int, object]]]:
    """線上的線路設定 → 這次測試用的（只留一台相機、只留主站、主站換成指定的模型版本）。

    相機（決定 ROI 等現場參數）：camera_id 明講就用它（不在這條線 → 錯）；
    沒講 → 影片的相機（prefer_camera）在這條線就用它，否則這條線第一台 —— 拿別台的影片試這條線的邏輯是正常用法。

    station：換成別的判定邏輯（新邏輯上線前先拿錄影試）；換了就不沿用相機現在的站別參數（那是舊邏輯的），
    改用 station_params（不給 = 站別預設）。
    mode=model：不管 station / station_params，一律換成不判定的站別（只看模型），不需要 ROI。

    回 (設定, [(模型名, 版本, Registry 解析結果)])：這次要放進臨時 Triton 的模型。
    """
    cams = [c for c in line.get("cameras") or [] if c.get("enabled", True)]
    if not cams:
        raise ValueError(f"線路 {line.get('id')} 沒有啟用中的相機（判定參數要跟著相機走）")
    cam = next((c for c in cams if c["id"] == camera_id), None) if camera_id else None
    if camera_id and cam is None:
        raise ValueError(f"線路 {line.get('id')} 沒有相機 {camera_id}（有：{[c['id'] for c in cams]}）")
    cam = cam or next((c for c in cams if c["id"] == prefer_camera), None) or cams[0]
    if mode == "model":
        station, station_params = MODEL_ONLY_STATION, MODEL_ONLY_PARAMS
        cam = {**cam, "station_params": station_params, "extra_params": {}}
    elif station and station != line.get("station"):
        cam = {**cam, "station_params": station_params or {}}
    elif station_params is not None:
        cam = {**cam, "station_params": station_params}
    out = {**line, "station": station or line.get("station"), "cameras": [cam], "enabled": True}

    models: list[tuple[str, int, object]] = []

    def resolve(name: str, version: int | None):
        r = registry.get(name, int(version)) if version else registry.resolve(name)
        if r is None:
            raise ValueError(f"Registry 找不到 {name} {('v' + str(version)) if version else '現役版'}")
        if not r.source_uri:
            raise ValueError(f"{name} v{r.version} 沒有可部署的模型檔（plan / onnx）")
        models.append((name, r.version, r))
        return r

    primary = (line.get("model") or {}).get("name")
    name = model_name or primary
    if not name:
        raise ValueError(f"線路 {line.get('id')} 沒有模型")
    r = resolve(name, model_version)
    out["model"] = model_block(line.get("model"), r.meta, r.version)
    # 只測主站：附加站是線上另外掛的（例：ipcam 線的 ppe），測的不是它 —— 帶進來會混進它的 NG、多吃臨時 GPU，
    # 也會讓「同一段影片、同一版模型」的結果隨別的模型上線而變（2026-10-21 ppe v2 上線後每個測試都在跑 ppe）
    out["extra_stations"] = []
    return out, models


def stage_models(repo: Path, models: list[tuple[str, int, object]], objects) -> None:
    """臨時 Triton 的 repo：<模型>/<版本>/model.plan + config.pbtxt（跟線上同一份設定產生器）。"""
    from edgevision.controller.deployer import LocalModelRepo, ModelSpec, render_triton_config

    lr = LocalModelRepo(repo)
    for name, version, r in models:
        spec = ModelSpec.from_meta(r.meta)
        lr.stage_file(objects, r.source_uri, name, version,
                      "model.onnx" if spec.backend == "onnxruntime" else "model.plan")
        lr.write_config(name, render_triton_config(name, spec, version=version))


def prepare(task_id: str) -> int:
    from libs.contracts.training import TaskStatus

    from edgevision.media.frames import probe
    from edgevision.media.jobs import _deps
    from edgevision.registry.clearml_registry import ClearMLRegistry

    lib, store = _deps()
    t = store.get(task_id)
    p = dict(t.params)
    wd = work_dir(Path(os.environ.get("TRAIN_WORK_DIR", "/work")), task_id)
    try:
        wd.mkdir(parents=True, exist_ok=True)
        (wd / "dl").mkdir(exist_ok=True)
        store.heartbeat(task_id, "準備：取影片")
        parts, cam, video_desc = resolve_parts(lib, p | {"_dl_dir": str(wd / "dl")},
                                               Path(os.environ.get("MEDIA_NAS_ROOT", "/nas/raw")))
        video = wd / "video.mp4"
        cut_video(parts, video)
        dur, size = probe(video)
        if dur and dur > MAX_DURATION_S + MAX_DURATION_SLACK_S:
            raise ValueError(f"影片 {dur:.0f} 秒太長（上限 {MAX_DURATION_S:.0f} 秒）；請縮短時間段")
        for f in (wd / "dl").iterdir():
            f.unlink()

        store.heartbeat(task_id, "準備：線路設定與模型")
        registry = ClearMLRegistry(objects=lib.objects)
        line, models = build_line(fetch_line(p["line_id"]), camera_id=p.get("camera_id") if p.get("source_id") else None,
                                  model_name=p.get("model_name"), model_version=p.get("model_version"),
                                  registry=registry, prefer_camera=cam, station=p.get("station"),
                                  station_params=p.get("station_params"), mode=p.get("mode") or "logic")
        (wd / "line.json").write_text(json.dumps(line, ensure_ascii=False, default=str), encoding="utf-8")
        stage_models(wd / "repo", models, lib.objects)
        run = {"model_name": line["model"]["name"], "model_version": line["model"]["version"],
               "camera_id": line["cameras"][0]["id"], "duration_s": dur, "size": size, "video": video_desc,
               "models": [{"name": n, "version": v} for n, v, _ in models]}
        (wd / "run.json").write_text(json.dumps(run, ensure_ascii=False), encoding="utf-8")
        store.set_result(task_id, {"prepare": run})
        store.heartbeat(task_id, f"準備完成：{dur or 0:.0f} 秒影片、{run['model_name']} v{run['model_version']}")
        return EXIT_OK
    except Exception as exc:
        logger.exception("影片測試準備失敗")
        store.finish(task_id, TaskStatus.FAILED, message=f"準備失敗：{exc}", error=str(exc)[:2000])
        return EXIT_ERROR


# ── ④ finalize ──────────────────────────────────────────

def class_stats(rows: list[dict]) -> list[dict]:
    """每個（站別, 類別）：框數、出現在幾幀、平均信心值。純函式，可測。

    站別分開算：一條線可掛多站（主站 + ppe），各自是不同模型，同名類別不能混在一起。
    """
    boxes: Counter = Counter()
    frames: Counter = Counter()
    conf: Counter = Counter()
    for r in rows:
        st = r.get("station") or "main"
        seen = set()
        for d in r.get("detections") or []:
            k = (st, d.get("class_name"))
            boxes[k] += 1
            conf[k] += float(d.get("confidence") or 0.0)
            seen.add(k)
        for k in seen:
            frames[k] += 1
    return [{"station": st, "class_name": c, "boxes": boxes[(st, c)], "frames": frames[(st, c)],
             "mean_conf": round(conf[(st, c)] / boxes[(st, c)], 4)}
            for st, c in sorted(boxes, key=lambda k: (k[0] != "main", k[0], -boxes[k], str(k[1])))]


def summarize(rows: list[dict], mode: str = "logic") -> dict:
    """每幀結果 → 摘要與 NG 清單。純函式，可測。

    NG 就是線上會發警報 / 存證的那一幀（站別自己已經做完連續幀數、冷卻），不另外合併。
    UNKNOWN（無法判定，例如整輪沒看到基準物）另列，不算 NG。
    mode=model：不照實際速度播，站別判定不準（牆鐘），不出 NG / UNKNOWN（都是 None），只看各類別統計。
    """
    frames = Counter()
    ng, unknown = [], []
    classes = Counter()
    for r in rows:
        st = r.get("station") or "main"
        frames[st] += 1
        for d in r.get("detections") or []:
            classes[d.get("class_name")] += 1
        v = r.get("verdict") or {}
        if v.get("status") in ("ng", "unknown"):
            (ng if v["status"] == "ng" else unknown).append(
                {"t": r.get("t"), "frame_id": r.get("frame_id"), "station": r.get("station"),
                 "model": r.get("model_name"), "reason": v.get("reason"), "details": v.get("details") or {}})
    out = {"mode": mode, "frames": dict(frames), "detections_by_class": dict(classes.most_common()),
           "class_stats": class_stats(rows)}
    if mode == "model":
        return out | {"ng_count": None, "unknown_count": None, "ng_by_reason": None, "ng": [], "unknown": []}
    return out | {"ng_count": len(ng), "unknown_count": len(unknown), "unknown": unknown,
                  "ng_by_reason": dict(Counter(f"{n['station'] or 'main'}：{n['reason']}" for n in ng)), "ng": ng}


def _put(objects, path: Path, key: str, content_type: str) -> str:
    with path.open("rb") as fh:
        return objects.put(bucket=RESULT_BUCKET, key=key, data=fh, length=path.stat().st_size,
                           content_type=content_type)


def finalize(task_id: str) -> int:
    from libs.contracts.training import TaskStatus

    from edgevision.media.jobs import _deps

    lib, store = _deps()
    wd = work_dir(Path(os.environ.get("TRAIN_WORK_DIR", "/work")), task_id)
    mode = (store.get(task_id).params or {}).get("mode") or "logic"
    try:
        rows_path, video = wd / "results.jsonl", wd / "annotated.mp4"
        if not rows_path.exists():
            raise RuntimeError("runner 沒有產出結果檔（看任務步驟 runner 的 log）")
        rows = [json.loads(x) for x in rows_path.read_text(encoding="utf-8").splitlines() if x.strip()]
        summary = summarize(rows, mode)
        prefix = result_prefix(task_id)
        out = {"results_uri": _put(lib.objects, rows_path, f"{prefix}/results.jsonl", "application/x-ndjson")}
        if video.exists() and video.stat().st_size > 0:
            out["video_uri"] = _put(lib.objects, video, f"{prefix}/annotated.mp4", "video/mp4")
        run = json.loads((wd / "run.json").read_text(encoding="utf-8"))
        if run.get("runner_s") and run.get("duration_s"):
            run["speed"] = round(run["duration_s"] / run["runner_s"], 2)      # 幾倍速（runner 這一步，不含排隊與準備）
        store.set_result(task_id, {"video_test": {**run, **out, **summary}})
        n = sum(summary["frames"].values())
        msg = (f"完成：{n} 幀、{sum(c['boxes'] for c in summary['class_stats'])} 個框" if mode == "model"
               else f"完成：{n} 幀、NG {summary['ng_count']} 次")
        store.finish(task_id, TaskStatus.COMPLETED, message=msg,
                     model_version=str(run["model_version"]))
        return EXIT_OK
    except Exception as exc:
        logger.exception("影片測試收尾失敗")
        store.finish(task_id, TaskStatus.FAILED, message=f"收尾失敗：{exc}", error=str(exc)[:2000])
        return EXIT_ERROR


def main(argv: list[str] | None = None) -> int:
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["prepare", "finalize"])
    ap.add_argument("--task-id", required=True)
    a = ap.parse_args(argv)
    return {"prepare": prepare, "finalize": finalize}[a.step](a.task_id)


if __name__ == "__main__":
    sys.exit(main())
