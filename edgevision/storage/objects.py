"""MinIO / S3 物件儲存存取層。

介面刻意保持最小：put / get / presigned_url。
換成其他 S3 相容儲存時，這一層是唯一要動的地方。
"""

from __future__ import annotations

import logging
import os
from datetime import timedelta
from typing import BinaryIO

logger = logging.getLogger(__name__)


class ObjectStore:
    """MinIO 客戶端封裝。"""

    def __init__(
        self,
        *,
        endpoint: str | None = None,
        access_key: str | None = None,
        secret_key: str | None = None,
        secure: bool | None = None,
    ) -> None:
        try:
            from minio import Minio
        except ImportError as exc:
            raise RuntimeError("需要 minio：pip install minio") from exc

        # 不給預設值：位址猜錯不會報錯，只會靜默連到不存在的地方，
        # 或更糟——連到別的環境。部署層一定要明確給（見 deploy/.env.example）。
        self.endpoint = endpoint or os.environ.get("MINIO_ENDPOINT", "")
        if not self.endpoint:
            raise ValueError("未設定 MINIO_ENDPOINT")
        self._client = Minio(
            self.endpoint,
            access_key=access_key or os.environ.get("MINIO_ROOT_USER", ""),
            secret_key=secret_key or os.environ.get("MINIO_ROOT_PASSWORD", ""),
            secure=(
                secure if secure is not None
                else os.environ.get("MINIO_SECURE", "false").lower() == "true"
            ),
        )
        logger.info("物件儲存已連線：%s", self.endpoint)

        # 瀏覽器用的下載網址要用【對外】位址簽名：MINIO_ENDPOINT 是叢集內部名稱（minio:9000），
        # 瀏覽器連不到 —— K8s 上前端拿到影片網址卻播不出來（2026-10-05）。
        # S3 簽名包含 Host，所以不能事後換主機名，必須用對外位址的 client 直接簽。
        # region 寫死：沒給的話 presign 會先連線去問 bucket 在哪個 region，而 API 不一定連得到對外位址。
        public = os.environ.get("MINIO_PUBLIC_URL", "").strip()
        self._public = None
        if public:
            from urllib.parse import urlparse
            u = urlparse(public if "://" in public else f"http://{public}")
            self._public = Minio(
                u.netloc,
                access_key=access_key or os.environ.get("MINIO_ROOT_USER", ""),
                secret_key=secret_key or os.environ.get("MINIO_ROOT_PASSWORD", ""),
                secure=u.scheme == "https",
                region=os.environ.get("MINIO_REGION", "us-east-1"),
            )
            logger.info("下載網址用對外位址簽名：%s", public)

    def ping(self) -> bool:
        try:
            list(self._client.list_buckets())
            return True
        except Exception:
            return False

    def put(
        self,
        *,
        bucket: str,
        key: str,
        data: BinaryIO,
        length: int,
        content_type: str = "application/octet-stream",
    ) -> str:
        self._client.put_object(bucket, key, data, length, content_type=content_type)
        return f"s3://{bucket}/{key}"

    def get(self, *, bucket: str, key: str) -> bytes:
        resp = self._client.get_object(bucket, key)
        try:
            return resp.read()
        finally:
            resp.close()
            resp.release_conn()

    def remove(self, *, bucket: str, key: str) -> None:
        """刪除單一物件。"""
        self._client.remove_object(bucket, key)

    def remove_prefix(self, *, bucket: str, prefix: str) -> int:
        """刪除某前綴下的所有物件,回傳刪除數。

        用於清理:測試殘留、過期版本、孤兒檔案。
        """
        n = 0
        for obj in self._client.list_objects(bucket, prefix=prefix, recursive=True):
            self._client.remove_object(bucket, obj.object_name)
            n += 1
        return n

    def list_prefixes(self, *, bucket: str, prefix: str) -> list[str]:
        """prefix 底下一層的「子目錄」名稱（不含 prefix 與結尾 /）。"""
        out = []
        for obj in self._client.list_objects(bucket, prefix=prefix, recursive=False):
            if obj.is_dir:
                out.append(obj.object_name[len(prefix):].rstrip("/"))
        return out

    def exists_prefix(self, *, bucket: str, prefix: str) -> bool:
        return any(True for _ in self._client.list_objects(bucket, prefix=prefix, recursive=True))

    def copy(self, *, src_bucket: str, src_key: str, bucket: str, key: str) -> None:
        """伺服器端複製（資料不經過本程序），模型檔從 Registry 搬到 Triton repo 用。"""
        from minio.commonconfig import CopySource

        self._client.copy_object(bucket, key, CopySource(src_bucket, src_key))

    def presigned_url(
        self, *, bucket: str, key: str, expires_minutes: int = 60, internal: bool = False
    ) -> str:
        """產生臨時下載連結。

        讓前端直接向 MinIO 取檔，不必經過 API server 轉手 ——
        大檔案不會塞爆 API 的頻寬與記憶體。有設 MINIO_PUBLIC_URL 就用對外位址簽（見 __init__）。
        """
        return (self._client if internal else (self._public or self._client)).presigned_get_object(
            bucket, key, expires=timedelta(minutes=expires_minutes)
        )

    def presigned_put_url(
        self, *, bucket: str, key: str, expires_minutes: int = 60, internal: bool = False
    ) -> str:
        """產生臨時上傳連結（HTTP PUT）。

        用途：資料集 zip 可達數 GB，塞不進 API/MCP 的 JSON 參數，
        讓使用者拿這個網址 `curl -T` 直傳 MinIO。
        與下載連結同理：簽名含 Host，必須用對外位址的 client 簽（見 __init__）。
        internal=True：給叢集內的 Job 用（minio:9000 簽），不繞對外網址。
        """
        return (self._client if internal else (self._public or self._client)).presigned_put_object(
            bucket, key, expires=timedelta(minutes=expires_minutes)
        )

    def download(self, *, bucket: str, key: str, dest) -> int:
        """把物件串流寫到本地檔案，回傳位元組數。

        跟 get() 的差別：get() 整包進記憶體，資料集 zip 可達數 GB，
        伺服器端 ingest 要用這個。
        """
        resp = self._client.get_object(bucket, key)
        n = 0
        try:
            with open(dest, "wb") as fh:
                for chunk in resp.stream(8 * 1024 * 1024):
                    fh.write(chunk)
                    n += len(chunk)
            return n
        finally:
            resp.close()
            resp.release_conn()

    @staticmethod
    def parse_uri(uri: str) -> tuple[str, str]:
        """s3://bucket/path/to/key → (bucket, key)"""
        if not uri.startswith("s3://"):
            raise ValueError(f"不是 s3 URI：{uri}")
        bucket, _, key = uri[5:].partition("/")
        return bucket, key
