"""demo 的 MinIO 初始化：等 MinIO 起來 → 建 bucket → 存證影片設保留天數。

同 deploy/minio/init/init-buckets.sh 的 bucket 結構；改用 Python 是因為 demo 用的 MinIO 映像沒有 shell。
"""
import os
import sys
import time

from minio import Minio
from minio.commonconfig import ENABLED, Filter
from minio.lifecycleconfig import Expiration, LifecycleConfig, Rule

BUCKETS = ["raw", "curated", "datasets", "models", "inference-output", "experiments", "backups"]

client = Minio(os.environ.get("MINIO_ENDPOINT", "minio:9000"), access_key=os.environ["MINIO_ROOT_USER"],
               secret_key=os.environ["MINIO_ROOT_PASSWORD"], secure=False)
for i in range(60):
    try:
        client.list_buckets()
        break
    except Exception as e:  # noqa: BLE001 —— 還沒起來就等
        print(f"等 MinIO（{e.__class__.__name__}）…", flush=True)
        time.sleep(2)
else:
    sys.exit("MinIO 一直沒起來")

for b in BUCKETS:
    if not client.bucket_exists(b):
        client.make_bucket(b)
        print("建立 bucket", b)
days = int(os.environ.get("CLIP_RETENTION_DAYS", 7))
client.set_bucket_lifecycle("inference-output", LifecycleConfig(
    [Rule(ENABLED, rule_filter=Filter(prefix=""), rule_id="expire-clips", expiration=Expiration(days=days))]))
print(f"✅ MinIO 初始化完成（inference-output {days} 天後過期）")
