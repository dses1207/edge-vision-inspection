"""整合測試的共用設定。

整合測試需要實際運行的服務(MinIO、PostgreSQL、Triton、go2rtc)。
與單元測試分開:單元測試隨時可跑,整合測試需要 `docker compose up`。

環境變數(有預設值,對應本機 Compose):
    MINIO_ENDPOINT   localhost:9000
    DATABASE_URL     postgresql://.../edgevision
    TRITON_URL       http://localhost:8000
    API_URL          http://localhost:8080
"""
import os
import socket

import pytest

MINIO_ENDPOINT = os.environ.get("MINIO_ENDPOINT", "localhost:9000")
TRITON_URL = os.environ.get("TRITON_URL", "http://localhost:8000")
API_URL = os.environ.get("API_URL", "http://localhost:8080")

os.environ.setdefault("MINIO_ENDPOINT", MINIO_ENDPOINT)
os.environ.setdefault("MINIO_ROOT_USER", "edgevision")
os.environ.setdefault("MINIO_ROOT_PASSWORD", "edgevision-dev-password")
os.environ.setdefault(
    "DATABASE_URL",
    "postgresql://edgevision:edgevision-dev-password@localhost:5432/edgevision",
)


def _reachable(host: str, port: int, timeout: float = 2.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout):
            return True
    except OSError:
        return False


def require(host: str, port: int, name: str):
    """服務未就緒時跳過該測試,而非失敗 —— 缺服務不是程式的錯。"""
    if not _reachable(host, port):
        pytest.skip(f"{name} 未就緒（{host}:{port}），請先 docker compose up")


def _host_port(url: str, default_port: int) -> tuple[str, int]:
    """從 endpoint 拆 host:port,容器內外都適用。"""
    u = url.replace("http://", "").replace("https://", "")
    if "/" in u:
        u = u.split("/")[0]
    if ":" in u:
        h, p = u.rsplit(":", 1)
        return h, int(p)
    return u, default_port


@pytest.fixture(scope="session")
def minio():
    h, p = _host_port(MINIO_ENDPOINT, 9000)
    require(h, p, "MinIO")
    from edgevision.storage.objects import ObjectStore
    return ObjectStore()


@pytest.fixture(scope="session")
def db():
    dsn = os.environ.get("DATABASE_URL", "")
    # postgresql://user:pw@host:port/db
    host = "localhost"; port = 5432
    if "@" in dsn:
        hostpart = dsn.split("@")[1].split("/")[0]
        if ":" in hostpart:
            host, port = hostpart.rsplit(":", 1); port = int(port)
    require(host, port, "PostgreSQL")
    from edgevision.storage.db import Database
    d = Database()
    yield d
    d.close()
