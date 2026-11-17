"""Registry（ClearML + MinIO）整合測試：註冊 → 解析 → 移 tag → 版本遞增。需 ClearML server 與 MinIO。"""
import io
import os

import pytest

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def registry(minio):
    import httpx
    api = os.environ.get("CLEARML_API_HOST", "http://localhost:8008")
    try:
        assert httpx.get(f"{api}/debug.ping", timeout=3).status_code == 200
    except Exception:
        pytest.skip(f"ClearML apiserver 未就緒（{api}）")
    from edgevision.registry.clearml_registry import ClearMLRegistry
    reg = ClearMLRegistry(objects=minio, project=os.environ.get("CLEARML_PROJECT", "edgevision-itest"))
    yield reg
    minio.remove_prefix(bucket="models", prefix="itest_reg/")


def _meta(version, tmp_path):
    from libs.contracts import Lineage, ModelMetadata
    pt = tmp_path / f"best{version}.pt"; pt.write_bytes(b"fake-pt-" * 100)
    onnx = tmp_path / f"m{version}.onnx"; onnx.write_bytes(b"fake-onnx-" * 100)
    meta = ModelMetadata(model_name="itest_reg", version=version, class_names=["board", "hand"],
                         created_at="2026-09-02T00:00:00Z",
                         lineage=Lineage(dataset_version="lakefs://alpha/itest", code_commit="abc", base_model="x.pt"),
                         metrics={"map50_box": 0.9 + version / 100})
    return meta, {"pytorch": pt, "onnx": onnx}


def test_register_resolve_promote(registry, minio, tmp_path):
    from libs.contracts.model_meta import TAG_CANDIDATE, TAG_PRODUCTION
    v = registry.next_version("itest_reg")
    meta, files = _meta(v, tmp_path)
    r = registry.register(meta, files, tags=[TAG_CANDIDATE])
    assert r.version == v and r.registry_id
    # 檔案在 MinIO 規範路徑
    assert minio.get(bucket="models", key=f"itest_reg/v{v}/best.pt").startswith(b"fake-pt")
    assert minio.get(bucket="models", key=f"itest_reg/v{v}/metadata.json")
    got = registry.get("itest_reg", v)
    assert got and got.meta.lineage.dataset_version == "lakefs://alpha/itest"
    assert got.source_uri.endswith("model.onnx")
    # promote → resolve production
    registry.promote("itest_reg", v, tag=TAG_PRODUCTION)
    prod = registry.resolve("itest_reg", tag=TAG_PRODUCTION)
    assert prod and prod.version == v
    assert TAG_CANDIDATE not in [t for vi in registry.list_versions("itest_reg") if vi.version == v for t in vi.tags]
    # 第二版：production 移動且只在一版
    v2 = registry.next_version("itest_reg"); assert v2 == v + 1
    meta2, files2 = _meta(v2, tmp_path)
    registry.register(meta2, files2)
    registry.promote("itest_reg", v2)
    assert registry.resolve("itest_reg").version == v2
    tagged = [vi.version for vi in registry.list_versions("itest_reg") if TAG_PRODUCTION in vi.tags]
    assert tagged == [v2]
    # 回退
    registry.promote("itest_reg", v)
    assert registry.resolve("itest_reg").version == v
    # 重複註冊被擋
    with pytest.raises(ValueError):
        registry.register(meta, files)
    # update_artifacts
    r3 = registry.update_artifacts("itest_reg", v, tensorrt=f"s3://models/itest_reg/v{v}/model.plan")
    assert r3.meta.artifacts.tensorrt.endswith("model.plan") and r3.source_uri.endswith("model.plan")


def test_rename_moves_files_and_archives_old(registry, minio, tmp_path):
    """改名 = 複製成新名字的一版、tag 沿用、舊紀錄封存＋舊檔刪除；現役版本不准改。"""
    from libs.contracts.model_meta import TAG_CANDIDATE, TAG_PRODUCTION
    v = registry.next_version("itest_reg")
    meta, files = _meta(v, tmp_path)
    registry.register(meta, files, tags=[TAG_CANDIDATE])
    try:
        nv = registry.next_version("itest_reg_renamed")
        r = registry.rename("itest_reg", v, "itest_reg_renamed", note="itest")
        assert (r.model_name, r.version) == ("itest_reg_renamed", nv)
        assert minio.get(bucket="models", key=f"itest_reg_renamed/v{nv}/best.pt").startswith(b"fake-pt")
        assert r.meta.artifacts.onnx == f"s3://models/itest_reg_renamed/v{nv}/model.onnx"
        assert TAG_CANDIDATE in next(x.tags for x in registry.list_versions("itest_reg_renamed") if x.version == nv)
        assert registry.get("itest_reg", v) is None
        assert not minio.exists_prefix(bucket="models", prefix=f"itest_reg/v{v}/")
        # 現役不准改
        registry.promote("itest_reg_renamed", nv, tag=TAG_PRODUCTION)
        with pytest.raises(ValueError):
            registry.rename("itest_reg_renamed", nv, "itest_reg_other")
    finally:
        minio.remove_prefix(bucket="models", prefix="itest_reg_renamed/")
