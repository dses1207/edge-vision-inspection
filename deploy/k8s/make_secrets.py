#!/usr/bin/env python3
"""產生 edgevision-secrets（K8s Secret）：資料庫 / MinIO / ClearML / LakeFS 的密碼與金鑰，隨機產生；相機帳密從 .env 拿。

    python3 deploy/k8s/make_secrets.py --cam-env deploy/.env | kubectl apply -f -

⚠️ 只在第一次部署跑。資料庫、LakeFS 初始化後密碼就寫死在資料裡了，之後重產會讓全部服務連不上。
   所以叢集上已經有 edgevision-secrets 時預設拒絕（--force 才覆蓋，而且要知道自己在幹嘛）。
⚠️ 輸出含明碼，直接 pipe 給 kubectl，不要存檔、不要進 git（Secret 不進 git）。
"""

from __future__ import annotations

import argparse
import base64
import json
import re
import secrets
import subprocess
import sys


def token(n: int = 32) -> str:
    return secrets.token_urlsafe(n)[:n]


def cam_creds(env_file: str | None) -> dict[str, str]:
    """只拿 CAM_* 開頭的鍵（相機帳密）。不 source 整個 .env —— 值裡可能有 shell 特殊字元。"""
    out: dict[str, str] = {}
    if not env_file:
        return out
    for line in open(env_file, encoding="utf-8"):
        m = re.match(r"^(CAM_[A-Z0-9_]+)=(.*)$", line.rstrip("\n"))
        if m and not m.group(1).endswith("_IP"):
            out[m.group(1)] = m.group(2).strip().strip('"')
    return out


def build(cam: dict[str, str], *, pg_user: str = "edgevision", pg_db: str = "edgevision") -> dict[str, str]:
    pg_pw, minio_pw = token(24), token(24)
    data = {
        "POSTGRES_PASSWORD": pg_pw,
        "DATABASE_URL": f"postgresql://{pg_user}:{pg_pw}@postgres:5432/{pg_db}",
        "LAKEFS_DATABASE_URL": f"postgres://{pg_user}:{pg_pw}@postgres:5432/lakefs?sslmode=disable",
        "MINIO_ROOT_USER": "edgevision",
        "MINIO_ROOT_PASSWORD": minio_pw,
        "CLEARML_API_ACCESS_KEY": secrets.token_hex(12).upper(),
        "CLEARML_API_SECRET_KEY": token(40),
        "LAKEFS_ACCESS_KEY": "AKIA" + secrets.token_hex(8).upper(),
        "LAKEFS_SECRET_KEY": token(40),
        "LAKEFS_ENCRYPT_SECRET": token(32),
    }
    data.update(cam)
    return data


def secret_exists(name: str, namespace: str) -> bool:
    r = subprocess.run(["kubectl", "-n", namespace, "get", "secret", name], capture_output=True)
    return r.returncode == 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cam-env", help="從這個 .env 拿 CAM_* 相機帳密（例 deploy/.env）")
    ap.add_argument("--namespace", default="edgevision")
    ap.add_argument("--force", action="store_true", help="叢集上已有 edgevision-secrets 也照樣產生（會讓既有資料庫連不上）")
    a = ap.parse_args()
    if not a.force and secret_exists("edgevision-secrets", a.namespace):
        sys.exit("叢集上已經有 edgevision-secrets，不重產（資料庫密碼已寫死在資料裡）。真的要覆蓋請加 --force")
    data = build(cam_creds(a.cam_env))
    obj = {"apiVersion": "v1", "kind": "Secret", "type": "Opaque",
           "metadata": {"name": "edgevision-secrets", "namespace": a.namespace,
                        "labels": {"app.kubernetes.io/part-of": "edgevision"}},
           "data": {k: base64.b64encode(v.encode()).decode() for k, v in data.items()}}
    json.dump(obj, sys.stdout, ensure_ascii=False)
    print(f"# edgevision-secrets：{len(data)} 個鍵（{', '.join(sorted(data))}）", file=sys.stderr)


if __name__ == "__main__":
    main()
