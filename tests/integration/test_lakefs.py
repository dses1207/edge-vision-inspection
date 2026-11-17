"""LakeFS 整合測試：建 repo → 上傳 → 建版本 → 從版本 ID 讀回 → 分支隔離。需 LakeFS + MinIO。"""
import os

import pytest

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def client():
    from edgevision.training.lakefs_client import healthy, lakefs_client
    if not healthy():
        pytest.skip(f"LakeFS 未就緒（{os.environ.get('LAKEFS_ENDPOINT')}）")
    return lakefs_client()


def test_version_roundtrip(client, tmp_path):
    from edgevision.training.lakefs_client import commit, ensure_repo, read_object, ref_exists, ref_objects, upload_dir
    from edgevision.training.dataset import LakeFSSource
    from tools.make_synthetic_seg_dataset import write_dataset

    ensure_repo(client, "itest")
    src = write_dataset(tmp_path / "src", 4, seed=9)
    n = upload_dir(client, "itest", "main", src)
    assert n == 4 * 2 + 1
    v1 = commit(client, "itest", "main", "itest v1", {"n": 4})
    assert ref_exists(client, "itest", v1)
    paths = sorted(o.path for o in ref_objects(client, "itest", v1, prefix="images/"))
    assert len(paths) == 4
    assert read_object(client, "itest", v1, "classes.txt").decode().startswith("board")

    # 改資料再建版本 → 舊版本不變
    import lakefs
    lakefs.repository("itest", client=client).branch("main").object(paths[0]).delete()
    v2 = commit(client, "itest", "main", "itest v2 刪一張")
    assert len(list(ref_objects(client, "itest", v2, prefix="images/"))) == 3
    assert len(list(ref_objects(client, "itest", v1, prefix="images/"))) == 4

    # DatasetSource 依版本 ID 拉回並整理
    ds = LakeFSSource(f"lakefs://itest/{v1}").prepare(tmp_path / "work")
    assert ds.manifest.n_images == 4 and ds.version == f"lakefs://itest/{v1}"
    with pytest.raises(Exception):
        LakeFSSource("lakefs://itest/0000000000000000000000000000000000000000").prepare(tmp_path / "w2")
