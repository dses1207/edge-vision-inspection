"""圖片庫的背景任務（K8s Job）。

    python -m edgevision.media.jobs import    --task-id <id>   匯入：zip / 影片 / NAS 錄影 / 告警回流（edgevision 映像，CPU）
    python -m edgevision.media.jobs autolabel --task-id <id>   自動標注（訓練映像，GPU；引擎見 edgevision/media/autolabel.py）
    python -m edgevision.media.jobs export    --task-id <id>   匯出：圖片集 → LakeFS 資料集版本（edgevision 映像，CPU）

參數在 tasks.params（控制 API 建任務時寫入），進度寫心跳、結果寫 tasks.result —— 跟訓練任務同一套，
MCP 的 get_task / get_task_logs / cancel_task 直接可用。
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import shutil
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

from libs.contracts.training import TaskStatus

logger = logging.getLogger(__name__)

EXIT_OK, EXIT_ERROR, EXIT_CANCELLED = 0, 1, 2


class _Progress:
    """每 N 秒才寫一次心跳：幾千張圖逐張寫 DB 會拖慢匯入。"""

    def __init__(self, store, task_id: str, every_s: float = 5.0) -> None:
        self.store, self.task_id, self.every_s, self._t = store, task_id, every_s, 0.0

    def __call__(self, msg: str, *, force: bool = False) -> None:
        if force or time.time() - self._t >= self.every_s:
            self.store.heartbeat(self.task_id, msg)
            self._t = time.time()

    def cancelled(self) -> bool:
        return self.store.cancel_requested(self.task_id)


def _deps():
    from edgevision.media.library import MediaLibrary
    from edgevision.storage.db import Database
    from edgevision.storage.objects import ObjectStore
    from edgevision.training.tracker import TaskStore

    db = Database()
    return MediaLibrary(db, ObjectStore()), TaskStore(db)


# ── 匯入 ─────────────────────────────────────────────────

def import_zip(lib, cid: int, zip_path: Path, *, actor: str, task_id: str, with_labels: bool, progress,
               batch_id: int | None = None) -> dict:
    from edgevision.media.library import IMAGE_EXTS
    from edgevision.media.shapes import from_yolo_lines
    from edgevision.training.dataset import _read_class_names, extract_zip

    col = lib.collection(cid)
    work = Path(tempfile.mkdtemp(prefix="media-zip-"))
    try:
        root = extract_zip(zip_path, work / "x")
        images_dir = root / "images" if (root / "images").is_dir() else root
        labels_dir = root / "labels"
        zip_names = _read_class_names(root) if with_labels else None
        names = col["class_names"]
        if with_labels:
            if not zip_names:
                raise ValueError("帶標注匯入需要 classes.txt 或 data.yaml（類別名稱）")
            if not names:
                lib.update_collection(cid, class_names=zip_names)
                names = zip_names
            missing = [n for n in zip_names if n not in names]
            if missing:
                raise ValueError(f"zip 的類別 {missing} 不在圖片集類別 {names} 裡（先把類別加進圖片集）")
        source_id = lib.add_source(kind="zip", uri=zip_path.name, actor=actor, params={"with_labels": with_labels})
        files = sorted(p for p in images_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTS and p.is_file())
        added = dup = labeled = bad = 0
        for i, img in enumerate(files):
            if progress.cancelled():
                raise KeyboardInterrupt
            try:
                ref = lib.put_image(img.read_bytes(), filename=img.name, actor=actor)
            except ValueError:
                bad += 1
                continue
            shapes = None
            if with_labels:
                lbl = labels_dir / img.relative_to(images_dir).with_suffix(".txt")
                if lbl.is_file():
                    lines = lbl.read_text().splitlines()
                    shapes = from_yolo_lines(lines, zip_names, "pose" if col.get("kpt_shape") else "detect",
                                             col.get("kpt_shape"))
                    labeled += 1
            _, created = lib.add_item(cid, ref.sha256, actor=actor, source_id=source_id, original_name=img.name,
                                      shapes=shapes, status=None if shapes is None else
                                      ("labeled" if shapes else "background"), task_id=task_id, batch_id=batch_id)
            added += int(created); dup += int(not created)
            progress(f"匯入 {i + 1}/{len(files)}")
        return {"files": len(files), "added": added, "already_in_collection": dup, "with_labels": labeled,
                "not_images": bad}
    finally:
        shutil.rmtree(work, ignore_errors=True)


def import_video(lib, cid: int, video: Path, *, actor: str, kind: str, uri: str, interval_s: float,
                 max_frames: int, progress, camera_id: str | None = None, started_at: datetime | None = None,
                 start_s: float = 0.0, end_s: float | None = None, batch_id: int | None = None) -> dict:
    from datetime import timedelta

    from edgevision.media.frames import probe, sample

    dur, size = probe(video)
    sha = lib.put_video(video, actor=actor, duration_s=dur, size=size) if kind == "upload_video" else None
    source_id = lib.add_source(kind=kind, uri=uri, actor=actor, blob_sha256=sha, camera_id=camera_id,
                               started_at=started_at, params={"interval_s": interval_s, "start_s": start_s,
                                                              "end_s": end_s})
    added = dup = 0
    for fr in sample(video, interval_s=interval_s, start_s=start_s, end_s=end_s, max_frames=max_frames):
        if progress.cancelled():
            raise KeyboardInterrupt
        ref = lib.put_image(fr.jpeg, filename=f"{Path(uri).stem}_{fr.offset_s:09.3f}.jpg", actor=actor)
        captured = started_at + timedelta(seconds=fr.offset_s) if started_at else None
        _, created = lib.add_item(cid, ref.sha256, actor=actor, source_id=source_id, source_offset_s=fr.offset_s,
                                  camera_id=camera_id, captured_at=captured,
                                  original_name=f"{Path(uri).name}@{fr.offset_s:.1f}s", batch_id=batch_id)
        added += int(created); dup += int(not created)
        progress(f"切幀 {Path(uri).name}：{added + dup} 張（{fr.offset_s:.0f}s）")
    return {"frames": added + dup, "added": added, "already_in_collection": dup, "duration_s": dur}


def import_selected(lib, cid: int, files: list[dict], *, actor: str, select: dict, progress,
                    batch_id: int | None = None) -> dict:
    """strategy=diverse：密集看 → 去重 → 有沒有在動 → 分群挑代表（edgevision/media/select.py，不用模型）。

    files：[{path, uri, kind, camera_id, started_at, start_s, end_s}]；只有挑中的幀會存進圖片庫。
    """
    from datetime import timedelta

    from edgevision.media.frames import LOCAL_TZ, probe
    from edgevision.media.select import SelectParams, choose, extract, scan

    sp = SelectParams(**select)
    progress(f"挑圖：{len(files)} 個檔、目標 {sp.target} 張、每 {sp.dense_interval_s}s 看一張", force=True)
    cands, st = scan([(f["path"], f.get("start_s", 0.0), f.get("end_s")) for f in files], sp,
                     progress=progress, cancelled=progress.cancelled)
    picked = choose(cands, sp)
    st.selected, st.selected_moving = len(picked), sum(c.moving for c in picked)

    added = dup = 0
    for si, f in enumerate(files):
        mine = {c.offset_s: c for c in picked if c.src == si}
        if not mine:
            continue
        path = f["path"]
        sha = None
        if f["kind"] == "upload_video":
            dur, size = probe(path)
            sha = lib.put_video(path, actor=actor, duration_s=dur, size=size)
        source_id = lib.add_source(kind=f["kind"], uri=f["uri"], actor=actor, blob_sha256=sha,
                                   camera_id=f.get("camera_id"), started_at=f.get("started_at"),
                                   params={"strategy": "diverse", "select": sp.to_dict(),
                                           "start_s": f.get("start_s", 0.0), "end_s": f.get("end_s")})
        for t, jpeg, _w, _h in extract(path, list(mine)):
            if progress.cancelled():
                raise KeyboardInterrupt
            c = mine[t]
            ref = lib.put_image(jpeg, filename=f"{Path(f['uri']).stem}_{t:09.3f}.jpg", actor=actor)
            captured = f["started_at"] + timedelta(seconds=t) if f.get("started_at") else None
            st.by_hour[captured.astimezone(LOCAL_TZ).strftime("%m-%d %H時") if captured
                       else f"{int(t // 600) * 10}–{int(t // 600) * 10 + 10} 分"] += 1
            _, created = lib.add_item(
                cid, ref.sha256, actor=actor, source_id=source_id, source_offset_s=t, camera_id=f.get("camera_id"),
                captured_at=captured, original_name=f"{Path(f['uri']).name}@{t:.1f}s",
                select_info={"strategy": "diverse", "moving": c.moving, "motion": round(c.motion, 5),
                             "group": c.group, "group_size": c.group_size}, batch_id=batch_id)
            added += int(created); dup += int(not created)
            progress(f"挑圖第二輪：存了 {added + dup}/{len(picked)} 張")
    return {"strategy": "diverse", "files": len(files), "select": st.to_dict(), "params": sp.to_dict(),
            "added": added, "already_in_collection": dup, "frames": added + dup}


def run_import(task_id: str) -> int:
    from edgevision.storage.objects import ObjectStore

    lib, store = _deps()
    t = store.get(task_id)
    p = t.params
    if p.get("kind") == "clip":
        return run_clip(lib, store, task_id)
    if p.get("kind") == "video_upload":
        return run_video_upload(lib, store, task_id)
    cid, kind, actor = int(p["collection_id"]), p["kind"], t.created_by
    bid = int(p["batch_id"]) if p.get("batch_id") else None
    progress = _Progress(store, task_id)
    work = Path(tempfile.mkdtemp(prefix="media-import-"))
    try:
        progress(f"匯入（{kind}）開始", force=True)
        if kind in ("zip", "video"):
            bucket, key = ObjectStore.parse_uri(p["source_uri"])
            local = work / Path(key).name
            lib.objects.download(bucket=bucket, key=key, dest=local)
            if kind == "zip":
                result = import_zip(lib, cid, local, actor=actor, task_id=task_id,
                                    with_labels=bool(p.get("with_labels")), progress=progress, batch_id=bid)
            elif p.get("strategy", "interval") == "diverse":
                result = import_selected(lib, cid, [{"path": local, "uri": p["source_uri"], "kind": "upload_video",
                                                     "camera_id": p.get("camera_id")}],
                                         actor=actor, select=p.get("select") or {}, progress=progress, batch_id=bid)
            else:
                result = import_video(lib, cid, local, actor=actor, kind="upload_video", uri=p["source_uri"],
                                      interval_s=float(p.get("interval_s", 1.0)),
                                      max_frames=int(p.get("max_frames", 2000)), progress=progress,
                                      camera_id=p.get("camera_id"), batch_id=bid)
            if p.get("cleanup_upload", True):
                lib.objects.remove(bucket=bucket, key=key)       # 暫存上傳：影片已存進圖片庫、zip 已拆開
        elif kind == "nas":
            result = import_nas(lib, cid, p, actor=actor, progress=progress, batch_id=bid)
        elif kind == "alert":
            from edgevision.media.alerts import import_alerts
            result = import_alerts(lib, cid, p, actor=actor, task_id=task_id, progress=progress, batch_id=bid)
        else:
            raise ValueError(f"未知的匯入類型：{kind}")
        store.set_result(task_id, {"import": {**result, "batch_id": bid}})
        store.finish(task_id, TaskStatus.COMPLETED, message=f"匯入完成：新增 {result.get('added', 0)} 張")
        return EXIT_OK
    except KeyboardInterrupt:
        store.finish(task_id, TaskStatus.CANCELLED, message="已取消（已匯入的保留）")
        return EXIT_CANCELLED
    except Exception as exc:
        logger.exception("匯入失敗")
        store.finish(task_id, TaskStatus.FAILED, message=f"匯入失敗：{exc}", error=str(exc)[:2000])
        return EXIT_ERROR
    finally:
        shutil.rmtree(work, ignore_errors=True)
        if kind == "alert":                            # 中途掛掉 / 取消：還沒處理到的事件下次重試
            from edgevision.media.alerts import release_queued
            release_queued(lib.db, task_id)


def run_clip(lib, store, task_id: str) -> int:
    """NAS 錄影剪一段 → 影片庫（media_sources kind=recording_clip，檔案存 MinIO）。不重新編碼。"""
    import os

    from edgevision.media.frames import probe
    from edgevision.media.video_test import cut_video, resolve_parts

    t = store.get(task_id)
    p = t.params
    work = Path(tempfile.mkdtemp(prefix="media-clip-"))
    try:
        store.heartbeat(task_id, "剪片段中")
        parts, cam, desc = resolve_parts(lib, {"camera_id": p["camera_id"], "start": p["start"], "end": p["end"]},
                                         Path(os.environ.get("MEDIA_NAS_ROOT", "/nas/raw")))
        out = work / "clip.mp4"
        cut_video(parts, out)
        dur, size = probe(out)
        sha = lib.put_video(out, actor=t.created_by, duration_s=dur, size=size)
        blob = lib.blob(sha)
        sid = lib.add_source(kind="recording_clip", uri=f"s3://{blob['bucket']}/{blob['object_key']}",
                             actor=t.created_by, blob_sha256=sha, camera_id=cam,
                             started_at=datetime.fromisoformat(p["start"]),
                             params={"start": p["start"], "end": p["end"], "segments": desc["segments"],
                                     "task_id": task_id, "copy": True})
        with lib.db.cursor() as cur:
            import json
            cur.execute("UPDATE media_sources SET name = %s, tags = %s, note = %s WHERE id = %s",
                        (p.get("name"), json.dumps(p.get("tags") or []), p.get("note"), sid))
        store.set_result(task_id, {"clip": {"video_id": sid, "duration_s": dur, "size_bytes": out.stat().st_size,
                                            "segments": desc["segments"]}})
        store.finish(task_id, TaskStatus.COMPLETED, message=f"已存進影片庫（id {sid}，{dur or 0:.0f} 秒）")
        return EXIT_OK
    except Exception as exc:
        logger.exception("剪片段失敗")
        store.finish(task_id, TaskStatus.FAILED, message=f"剪片段失敗：{exc}", error=str(exc)[:2000])
        return EXIT_ERROR
    finally:
        shutil.rmtree(work, ignore_errors=True)


_UPLOAD_STAMP = re.compile(r"^\d{8}_\d{6}_")


def upload_display_name(source_uri: str) -> str:
    """upload-url 的暫存檔名是「YYYYmmdd_HHMMSS_<原檔名>」→ 拿掉時間前綴，留原檔名當影片名稱。"""
    return _UPLOAD_STAMP.sub("", Path(source_uri).name) or Path(source_uri).name


def store_video(lib, local: Path, *, actor: str, name: str, tags: list[str], note: str | None,
                camera_id: str | None, task_id: str | None = None) -> dict:
    """一支影片 → 影片庫（media_sources kind=upload_video，檔案存 MinIO），不抽幀、不碰圖片集。

    同一支影片（內容相同）已經在影片庫 → 不新增，回原本那筆（duplicate=True）。
    """
    from edgevision.media.frames import probe

    try:
        dur, size = probe(local)
    except Exception as exc:  # noqa: BLE001 —— PyAV 的錯誤種類很多，對外都是「不是影片」
        raise ValueError(f"{name} 讀不出影片：{exc}") from exc
    sha = lib.put_video(local, actor=actor, duration_s=dur, size=size)
    with lib.db.cursor() as cur:
        cur.execute("SELECT id FROM media_sources WHERE kind = 'upload_video' AND blob_sha256 = %s ORDER BY id LIMIT 1",
                    (sha,))
        row = cur.fetchone()
    if row:
        return {"video_id": int(row[0]), "duplicate": True, "duration_s": dur}
    blob = lib.blob(sha)
    sid = lib.add_source(kind="upload_video", uri=f"s3://{blob['bucket']}/{blob['object_key']}", actor=actor,
                         blob_sha256=sha, camera_id=camera_id, params={"task_id": task_id, "library_only": True})
    with lib.db.cursor() as cur:
        cur.execute("UPDATE media_sources SET name = %s, tags = %s, note = %s WHERE id = %s",
                    (name, json.dumps(sorted({t.strip() for t in tags if t.strip()})), note, sid))
    return {"video_id": sid, "duplicate": False, "duration_s": dur}


def run_video_upload(lib, store, task_id: str) -> int:
    """上傳的影片只存進影片庫（POST /api/media/videos）。暫存上傳檔處理完就刪。"""
    from edgevision.storage.objects import ObjectStore

    t = store.get(task_id)
    p = t.params
    work = Path(tempfile.mkdtemp(prefix="media-video-"))
    try:
        store.heartbeat(task_id, "下載上傳的影片")
        bucket, key = ObjectStore.parse_uri(p["source_uri"])
        local = work / Path(key).name
        lib.objects.download(bucket=bucket, key=key, dest=local)
        store.heartbeat(task_id, "存進影片庫")
        out = store_video(lib, local, actor=t.created_by, name=p.get("name") or upload_display_name(key),
                          tags=p.get("tags") or [], note=p.get("note"), camera_id=p.get("camera_id"), task_id=task_id)
        lib.objects.remove(bucket=bucket, key=key)
        store.set_result(task_id, {"video": {**out, "size_bytes": local.stat().st_size}})
        msg = (f"影片庫已經有這支影片（id {out['video_id']}），沒有重複新增" if out["duplicate"]
               else f"已存進影片庫（id {out['video_id']}，{out['duration_s'] or 0:.0f} 秒）")
        store.finish(task_id, TaskStatus.COMPLETED, message=msg)
        return EXIT_OK
    except Exception as exc:
        logger.exception("影片上傳失敗")
        store.finish(task_id, TaskStatus.FAILED, message=f"影片上傳失敗：{exc}", error=str(exc)[:2000])
        return EXIT_ERROR
    finally:
        shutil.rmtree(work, ignore_errors=True)


def import_nas(lib, cid: int, p: dict, *, actor: str, progress, batch_id: int | None = None) -> dict:
    """NAS 24h 錄影：相機 + 時間段 → 挑圖（strategy=diverse）或每 interval 秒一張。錄影檔名是分段開始的本地時間。"""
    import os

    from edgevision.media.frames import LOCAL_TZ, nas_segments

    root = Path(os.environ.get("MEDIA_NAS_ROOT", "/nas/raw"))
    cam = p["camera_id"]
    start = datetime.fromisoformat(p["start"])
    end = datetime.fromisoformat(p["end"])
    start = start if start.tzinfo else start.replace(tzinfo=LOCAL_TZ)
    end = end if end.tzinfo else end.replace(tzinfo=LOCAL_TZ)
    segs = nas_segments(root, cam, start, end)
    if not segs:
        raise ValueError(f"NAS 上 {cam} 在 {start}–{end} 沒有錄影")
    if p.get("strategy", "interval") == "diverse":
        files = [{"path": path, "uri": f"nas://raw/{cam}/{path.name}", "kind": "nas_recording", "camera_id": cam,
                  "started_at": t0, "start_s": max(0.0, (start - t0).total_seconds()),
                  "end_s": (end - t0).total_seconds()} for path, t0 in segs]
        return {"segments": len(segs)} | import_selected(lib, cid, files, actor=actor, select=p.get("select") or {},
                                                          progress=progress, batch_id=batch_id)
    total = {"segments": len(segs), "frames": 0, "added": 0, "already_in_collection": 0}
    budget = int(p.get("max_frames", 2000))
    for path, t0 in segs:
        if budget <= 0:
            break
        r = import_video(lib, cid, path, actor=actor, kind="nas_recording", uri=f"nas://raw/{cam}/{path.name}",
                         interval_s=float(p.get("interval_s", 2.0)), max_frames=budget, progress=progress,
                         camera_id=cam, started_at=t0,
                         start_s=max(0.0, (start - t0).total_seconds()), end_s=(end - t0).total_seconds(),
                         batch_id=batch_id)
        for k in ("frames", "added", "already_in_collection"):
            total[k] += r[k]
        budget -= r["frames"]
    return total


# ── 自動標注 ─────────────────────────────────────────────

def run_autolabel(task_id: str) -> int:
    from edgevision.media.autolabel import run

    lib, store = _deps()
    progress = _Progress(store, task_id)
    try:
        result = run(lib, store.get(task_id), progress=progress)
        store.set_result(task_id, {"autolabel": result})
        store.finish(task_id, TaskStatus.COMPLETED,
                     message=f"自動標注完成：寫入 {result['written']} 張、略過 {result['skipped']} 張")
        return EXIT_OK
    except KeyboardInterrupt:
        store.finish(task_id, TaskStatus.CANCELLED, message="已取消（已寫入的保留）")
        return EXIT_CANCELLED
    except Exception as exc:
        logger.exception("自動標注失敗")
        store.finish(task_id, TaskStatus.FAILED, message=f"自動標注失敗：{exc}", error=str(exc)[:2000])
        return EXIT_ERROR


# ── 匯出 ─────────────────────────────────────────────────

def run_export(task_id: str) -> int:
    """全有或全無：export_collection 最後才 commit，中途失敗不會留下半套資料集版本。"""
    from edgevision.media.export import export_collection

    lib, store = _deps()
    t = store.get(task_id)
    p = t.params
    if p.get("format"):
        return run_download(lib, store, task_id)
    try:
        store.heartbeat(task_id, "匯出中")
        res = export_collection(lib, int(p["collection_id"]), repo=p["repo"], message=p["message"],
                                actor=t.created_by, task_id=task_id, val_pct=int(p["val_pct"]),
                                batch_ids=p.get("batch_ids"), task=p.get("task"))
        store.set_result(task_id, {"export": res})
        store.finish(task_id, TaskStatus.COMPLETED, message=f"已建立 {res['dataset_ref']}")
        return EXIT_OK
    except Exception as exc:
        logger.exception("圖片集匯出失敗")
        store.finish(task_id, TaskStatus.FAILED, message=f"匯出失敗：{exc}", error=str(exc)[:2000])
        return EXIT_ERROR


def run_download(lib, store, task_id: str) -> int:
    """圖片集打包成 zip（YOLO / YOLO-seg / COCO / VOC）→ MinIO，結果給下載網址（edgevision/media/download.py）。"""
    from edgevision.media.download import build_dataset_zip, build_zip, dataset_download_key, download_key

    p = store.get(task_id).params
    work = Path(tempfile.mkdtemp(prefix="media-download-"))
    try:
        out = work / "dl.zip"
        if p["format"] == "dataset":                   # LakeFS 資料集版本原樣打包
            res = build_dataset_zip(p["repo"], p["version_id"], out=out, progress=_Progress(store, task_id))
            res["items"] = res["files"]
            bucket, key = dataset_download_key(p["repo"], task_id)
        else:
            cid = int(p["collection_id"])
            res = build_zip(lib, cid, fmt=p["format"], statuses=p["statuses"], batch_ids=p.get("batch_ids"),
                            val_pct=int(p.get("val_pct", 20)), out=out, progress=_Progress(store, task_id))
            bucket, key = download_key(cid, task_id, p["format"])
        with out.open("rb") as fh:
            lib.objects.put(bucket=bucket, key=key, data=fh, length=out.stat().st_size, content_type="application/zip")
        store.set_result(task_id, {"download": res | {"uri": f"s3://{bucket}/{key}"}})
        store.finish(task_id, TaskStatus.COMPLETED, message=f"打包完成：{res['items']} 張，{res['size_bytes'] / 1e6:.0f} A")
        return EXIT_OK
    except Exception as exc:
        logger.exception("打包下載失敗")
        store.finish(task_id, TaskStatus.FAILED, message=f"打包失敗：{exc}", error=str(exc)[:2000])
        return EXIT_ERROR
    finally:
        shutil.rmtree(work, ignore_errors=True)


# ── 混合（LakeFS 快照＋配方 → 訓練快照）─────────────────────

def run_mix(task_id: str) -> int:
    """全有或全無：最後才 commit。"""
    from edgevision.training.mix import mix

    _lib, store = _deps()
    p = store.get(task_id).params
    try:
        store.heartbeat(task_id, "混合中：拉來源快照")
        res = mix(p["sources"], repo=p["repo"], message=p["message"], seed=int(p.get("seed", 0)),
                  total=p.get("total"))
        store.set_result(task_id, {"mix": res})
        store.finish(task_id, TaskStatus.COMPLETED, message=f"已建立 {res['dataset_ref']}")
        return EXIT_OK
    except Exception as exc:
        logger.exception("混合失敗")
        store.finish(task_id, TaskStatus.FAILED, message=f"混合失敗：{exc}", error=str(exc)[:2000])
        return EXIT_ERROR


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["import", "autolabel", "export", "mix"])
    ap.add_argument("--task-id", required=True)
    a = ap.parse_args(argv)
    return {"import": run_import, "autolabel": run_autolabel, "export": run_export,
            "mix": run_mix}[a.action](a.task_id)


if __name__ == "__main__":
    sys.exit(main())
