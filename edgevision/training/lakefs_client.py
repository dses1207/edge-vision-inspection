"""LakeFS 存取。

連線只讀環境變數：
    LAKEFS_ENDPOINT      http://lakefs:8000
    LAKEFS_ACCESS_KEY    LAKEFS_SECRET_KEY
Compose 與 K8s 同一份程式。

名詞對照（LakeFS 借用 Git 名詞，但管的是資料）：
    repository = 一個資料集的儲存空間     branch = 工作線
    commit     = 資料集的一個不可變版本    commit id = 我們的「資料集版本 ID」
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Iterator

logger = logging.getLogger(__name__)

DATASETS_BUCKET = "datasets"

# LakeFS 對 repository 名稱的限制（RepositoryCreation.name）：小寫英數與 -，3–63 字，開頭不能是 -。
REPO_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9-]{2,62}$")
REPO_NAME_RULE = "小寫英數與 -，3–63 字，開頭不能是 -（不能有底線、大寫）"


def repo_name_error(name: str) -> str | None:
    """repo 名稱不合 LakeFS 規則就回錯誤訊息（API 送出時就擋，不要等任務跑起來才失敗）。"""
    if REPO_NAME_RE.fullmatch(name or ""):
        return None
    return f"repo 名稱 {name!r} 不合 LakeFS 規則：{REPO_NAME_RULE}"


def lakefs_client():
    try:
        from lakefs.client import Client
    except ImportError as exc:
        raise RuntimeError("需要 lakefs：pip install lakefs") from exc
    endpoint = os.environ.get("LAKEFS_ENDPOINT", "http://lakefs:8000")
    key = os.environ.get("LAKEFS_ACCESS_KEY", "")
    secret = os.environ.get("LAKEFS_SECRET_KEY", "")
    if not key or not secret:
        raise RuntimeError("未設定 LAKEFS_ACCESS_KEY / LAKEFS_SECRET_KEY")
    return Client(host=endpoint, username=key, password=secret)


def healthy(endpoint: str | None = None, timeout: float = 3.0) -> bool:
    import httpx

    url = (endpoint or os.environ.get("LAKEFS_ENDPOINT", "http://lakefs:8000")).rstrip("/")
    try:
        return httpx.get(f"{url}/api/v1/healthcheck", timeout=timeout).status_code in (200, 204)
    except Exception:
        return False


def ensure_repo(client, name: str, *, storage_namespace: str | None = None):
    """建 repository（存在則沿用）。實體檔案落在 MinIO datasets/lakefs/<name>/。"""
    import lakefs

    ns = storage_namespace or f"s3://{DATASETS_BUCKET}/lakefs/{name}"
    repo = lakefs.repository(name, client=client)
    try:
        repo.create(storage_namespace=ns, default_branch="main", exist_ok=True)
    except TypeError:  # 舊版 SDK 沒有 exist_ok
        try:
            repo.create(storage_namespace=ns, default_branch="main")
        except Exception as exc:
            if "exists" not in str(exc).lower():
                raise
    return repo


def upload_dir(client, repo_name: str, branch: str, local: Path, *, prefix: str = "", replace: bool = True) -> int:
    """把本地目錄整個上傳到 branch（未 commit）。回傳檔案數。

    replace=True（預設）：先刪掉 branch 上該 prefix 的舊物件，讓「一個版本 = 這份資料的完整內容」。
    否則只新增/覆蓋，上一版多出來的檔案會一直留著（黃金測試集去重就靠這個才正確）。
    """
    import lakefs

    branch_obj = lakefs.repository(repo_name, client=client).branch(branch)
    if replace:
        old = [o.path for o in branch_obj.objects(prefix=prefix)]
        for path in old:
            branch_obj.object(path).delete()
        if old:
            logger.info("%s/%s 已清除 %d 個舊物件（prefix=%r）", repo_name, branch, len(old), prefix)
    n = 0
    for p in sorted(local.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(local).as_posix()
        branch_obj.object(f"{prefix}{rel}").upload(data=p.read_bytes(), mode="wb")
        n += 1
    return n


def commit(client, repo_name: str, branch: str, message: str, metadata: dict | None = None) -> str:
    """建立不可變版本，回傳版本 ID。"""
    import lakefs

    br = lakefs.repository(repo_name, client=client).branch(branch)
    try:
        ref = br.commit(message=message, metadata={k: str(v) for k, v in (metadata or {}).items()})
    except Exception as exc:
        # 內容與上一版完全相同 → LakeFS 拒絕建空 commit。同內容 = 同版本，回傳現有版本 ID（冪等）。
        if "no changes" in str(exc).lower():
            head = br.get_commit().id
            logger.info("%s/%s 內容無變更，沿用現有版本 %s", repo_name, branch, head[:12])
            return head
        raise
    return ref.get_commit().id if hasattr(ref, "get_commit") else ref.id


def ref_objects(client, repo_name: str, ref: str, *, prefix: str = "") -> Iterator:
    import lakefs

    return lakefs.repository(repo_name, client=client).ref(ref).objects(prefix=prefix)


def read_object(client, repo_name: str, ref: str, path: str) -> bytes:
    import lakefs

    with lakefs.repository(repo_name, client=client).ref(ref).object(path).reader(mode="rb") as fh:
        return fh.read()


def ref_exists(client, repo_name: str, ref: str) -> bool:
    import lakefs

    try:
        lakefs.repository(repo_name, client=client).ref(ref).get_commit()
        return True
    except Exception:
        return False


def dataset_uri(repo_name: str, version_id: str) -> str:
    return f"lakefs://{repo_name}/{version_id}"
