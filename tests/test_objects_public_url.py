"""下載網址要用對外位址簽：MINIO_ENDPOINT 是叢集內部名稱，瀏覽器連不到（2026-10-05）。"""
from urllib.parse import urlparse

from edgevision.storage.objects import ObjectStore


def _store(monkeypatch, public=""):
    monkeypatch.setenv("MINIO_ENDPOINT", "minio:9000")
    monkeypatch.setenv("MINIO_ROOT_USER", "u")
    monkeypatch.setenv("MINIO_ROOT_PASSWORD", "p" * 12)
    monkeypatch.setenv("MINIO_PUBLIC_URL", public)
    return ObjectStore()


def test_presign_uses_public_host_without_network(monkeypatch):
    url = _store(monkeypatch, "https://edgevision-minio-test.example.com").presigned_url(
        bucket="inference-output", key="cam193/x.mp4")
    u = urlparse(url)
    assert u.scheme == "https" and u.netloc == "edgevision-minio-test.example.com"
    assert u.path == "/inference-output/cam193/x.mp4" and "X-Amz-Signature" in u.query


def test_without_public_url_falls_back_to_internal(monkeypatch):
    s = _store(monkeypatch)
    s._client._region_map["inference-output"] = "us-east-1"     # 不連線
    assert urlparse(s.presigned_url(bucket="inference-output", key="k")).netloc == "minio:9000"
