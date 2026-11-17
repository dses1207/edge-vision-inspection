#!/usr/bin/env python3
"""把現有的相機 JSON 設定匯入資料庫（一次性遷移）。

用途：從「掛載 JSON」過渡到「DB as truth」。跑過一次之後，
設定就以 DB 為準，CAMERA_SOURCE 切成 api。

用法：
    python scripts/import_cameras.py --line-id alpha_a --station zone_intrusion \
        --cameras cam156,cam193 [--dry-run]

⚠️ 這是過渡工具，不是常態流程。日後改相機設定走 API（會寫 audit_log
   與版本快照），不要再用這支腳本直接寫表。
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from edgevision.config.camera_source import _load_from_file  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--line-id", required=True, help="線路 ID，runner 啟動時帶的就是它")
    ap.add_argument("--line-name", default="", help="線路顯示名")
    ap.add_argument("--station", default="zone_intrusion", help="站別名，對應 station.yaml")
    ap.add_argument("--cameras", default="", help="要匯入哪幾台（逗號分隔），空=全部")
    ap.add_argument("--cred-ref", default="default", help="帳密引用名（值不進 DB）")
    ap.add_argument("--source-fps", type=float, default=None,
                    help="相機來源 fps（設了 --target-fps 就必須給，跳幀間隔靠兩者相除）")
    ap.add_argument("--target-fps", type=float, default=None, help="線路層級的跳幀目標")
    ap.add_argument("--dry-run", action="store_true", help="只印不寫")
    args = ap.parse_args()

    cam_ids = [c.strip() for c in args.cameras.split(",") if c.strip()] or None
    cfg = _load_from_file(args.line_id, cam_ids)
    if not cfg.cameras:
        print("找不到任何相機，中止", file=sys.stderr)
        return 1

    line_params = {}
    if args.target_fps is not None:
        # 與 API 的驗證規則一致：不給預設值，缺了就誠實抱錯
        if args.source_fps is None:
            print("設了 --target-fps 就必須同時給 --source-fps"
                  "（跳幀間隔靠兩者相除，缺一個算不出來）", file=sys.stderr)
            return 1
        if args.target_fps > args.source_fps:
            print(f"--target-fps({args.target_fps}) 不能大於 "
                  f"--source-fps({args.source_fps})", file=sys.stderr)
            return 1
        line_params["target_fps"] = args.target_fps
        line_params["source_fps"] = args.source_fps
    elif args.source_fps is not None:
        line_params["source_fps"] = args.source_fps

    print(f"線路 {args.line_id}（{args.station}）共 {len(cfg.cameras)} 台相機：")
    for c in cfg.cameras:
        keys = ", ".join(c.station_params.keys()) or "無"
        print(f"  {c.id:8} ip={c.ip:16} {c.width}x{c.height}  現場值: {keys}")
    if line_params:
        print(f"  線路參數: {line_params}")

    if args.dry_run:
        print("\n--dry-run，未寫入")
        return 0

    from edgevision.storage.db import Database

    db = Database()
    with db.cursor() as cur:
        cur.execute(
            "INSERT INTO lines (id, name, station, line_params, updated_by) "
            "VALUES (%s, %s, %s, %s, 'import_script') "
            "ON CONFLICT (id) DO UPDATE SET "
            "  name = EXCLUDED.name, station = EXCLUDED.station, "
            "  line_params = EXCLUDED.line_params, "
            "  config_version = lines.config_version + 1, "
            "  updated_by = 'import_script', updated_at = now()",
            (args.line_id, args.line_name or args.line_id, args.station,
             json.dumps(line_params)),
        )
        for c in cfg.cameras:
            cur.execute(
                "INSERT INTO cameras (id, line_id, name, ip, stream_path, width, height, "
                "                     cred_ref, station_params, updated_by) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, 'import_script') "
                "ON CONFLICT (id) DO UPDATE SET "
                "  line_id = EXCLUDED.line_id, name = EXCLUDED.name, ip = EXCLUDED.ip, "
                "  stream_path = EXCLUDED.stream_path, width = EXCLUDED.width, "
                "  height = EXCLUDED.height, cred_ref = EXCLUDED.cred_ref, "
                "  station_params = EXCLUDED.station_params, "
                "  updated_by = 'import_script', updated_at = now()",
                (c.id, args.line_id, c.name, c.ip, c.stream_path, c.width, c.height,
                 args.cred_ref, json.dumps(c.station_params)),
            )
        # 版本快照：整條線的完整設定，供還原
        cur.execute("SELECT config_version FROM lines WHERE id = %s", (args.line_id,))
        version = cur.fetchone()[0]
        snapshot = {
            "line": {"id": args.line_id, "name": args.line_name or args.line_id,
                     "station": args.station, "line_params": line_params},
            "cameras": [
                {"id": c.id, "ip": c.ip, "stream_path": c.stream_path,
                 "width": c.width, "height": c.height, "cred_ref": args.cred_ref,
                 "station_params": c.station_params}
                for c in cfg.cameras
            ],
        }
        cur.execute(
            "INSERT INTO config_snapshots (line_id, config_version, snapshot, "
            "                              changed_by, change_reason) "
            "VALUES (%s, %s, %s, 'import_script', '從 JSON 一次性匯入')",
            (args.line_id, version, json.dumps(snapshot, ensure_ascii=False)),
        )
        cur.execute(
            "INSERT INTO audit_log (actor, actor_type, channel, action, target, params, result) "
            "VALUES ('import_script', 'human', 'api', 'camera_config_import', %s, %s, 'success')",
            (args.line_id, json.dumps({"cameras": [c.id for c in cfg.cameras],
                                       "version": version}, ensure_ascii=False)),
        )

    print(f"\n✓ 已匯入（config_version=v{version}，含版本快照與 audit_log）")
    print(f"  接著把 runner 的 CAMERA_SOURCE 設為 api、LINE_ID 設為 {args.line_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
