"""Triton model repository 的兩種放法：本地目錄 / MinIO（s3://），以及 TritonController 在兩者上的行為。"""
import io

import pytest

from edgevision.controller.deployer import (LocalModelRepo, ModelSpec, S3ModelRepo, TritonController,
                                       open_model_repo)


class FakeObjects:
    """ObjectStore 的記憶體版：只實作 deployer 用到的方法。"""

    def __init__(self):
        self.store: dict[tuple[str, str], bytes] = {}

    parse_uri = staticmethod(lambda uri: tuple(uri[5:].split("/", 1)))

    def get(self, *, bucket, key):
        return self.store[(bucket, key)]

    def put(self, *, bucket, key, data, length, content_type=""):
        self.store[(bucket, key)] = data.read()

    def copy(self, *, src_bucket, src_key, bucket, key):
        self.store[(bucket, key)] = self.store[(src_bucket, src_key)]

    def exists_prefix(self, *, bucket, prefix):
        return any(b == bucket and k.startswith(prefix) for b, k in self.store)

    def list_prefixes(self, *, bucket, prefix):
        return sorted({k[len(prefix):].split("/")[0] for b, k in self.store
                       if b == bucket and k.startswith(prefix) and "/" in k[len(prefix):]})

    def remove_prefix(self, *, bucket, prefix):
        keys = [k for k in self.store if k[0] == bucket and k[1].startswith(prefix)]
        for k in keys:
            del self.store[k]
        return len(keys)


@pytest.mark.parametrize("uri,bucket,prefix", [
    ("s3://minio:9000/triton-repo", "triton-repo", ""),
    ("s3://minio:9000/triton-repo/", "triton-repo", ""),
    ("s3://minio:9000/models/triton/blackwell", "models", "triton/blackwell/"),
    ("s3://http://minio:9000/triton-repo/a", "triton-repo", "a/"),
    ("s3://https://minio.example:443/b", "b", ""),
])
def test_s3_uri_parse(uri, bucket, prefix):
    assert S3ModelRepo.parse(uri) == (bucket, prefix)


@pytest.mark.parametrize("uri", ["s3://triton-repo/x", "s3://minio:9000"])
def test_s3_uri_needs_host_port_and_bucket(uri):
    with pytest.raises(ValueError):
        S3ModelRepo.parse(uri)


def test_open_model_repo_picks_backend(tmp_path):
    assert isinstance(open_model_repo(tmp_path, None), LocalModelRepo)
    assert isinstance(open_model_repo("/models", None), LocalModelRepo)
    assert isinstance(open_model_repo("s3://minio:9000/triton-repo", FakeObjects()), S3ModelRepo)


def _ctl(repo_location, objects, keep=2):
    c = TritonController.__new__(TritonController)
    c.objects = objects
    c.repo = open_model_repo(repo_location, objects)
    c.keep_versions = keep
    return c


@pytest.mark.parametrize("kind", ["local", "s3"])
def test_stage_prune_and_rollback_check(kind, tmp_path):
    objs = FakeObjects()
    loc = tmp_path / "repo" if kind == "local" else "s3://minio:9000/triton-repo/blackwell"
    c = _ctl(loc, objs)
    spec = ModelSpec(class_names=["board", "hand"], backend="tensorrt")
    for v in (1, 2, 3):
        objs.store[("models", f"seg/v{v}/model.plan")] = f"plan{v}".encode()
        c._stage_model("seg", f"s3://models/seg/v{v}/model.plan", v, spec)
        c._prune("seg", keep_around=v)

    assert sorted(c.repo.versions("seg")) == [2, 3]          # keep_versions=2
    assert c.repo.has_version("seg", 3) and not c.repo.has_version("seg", 1)
    if kind == "s3":
        assert objs.store[("triton-repo", "blackwell/seg/3/model.plan")] == b"plan3"
        cfg = objs.store[("triton-repo", "blackwell/seg/config.pbtxt")].decode()
    else:
        assert (loc / "seg" / "3" / "model.plan").read_bytes() == b"plan3"
        cfg = (loc / "seg" / "config.pbtxt").read_text()
    assert 'name: "seg"' in cfg and "versions: [ 3 ]" in cfg   # 版本釘進檔案


@pytest.mark.parametrize("kind", ["local", "s3"])
def test_pin_version_rewrites_only_version_policy(kind, tmp_path):
    objs = FakeObjects()
    loc = tmp_path / "repo" if kind == "local" else "s3://minio:9000/models/_triton/sm89"
    c = _ctl(loc, objs, keep=3)
    spec = ModelSpec(class_names=["board", "hand"], backend="tensorrt", max_batch_size=8)
    for v in (1, 2):
        objs.store[("models", f"seg/v{v}/model.plan")] = b"x"
        c._stage_model("seg", f"s3://models/seg/v{v}/model.plan", v, spec)
    assert "versions: [ 2 ]" in c.repo.read_config("seg")
    c._pin_version("seg", 1)
    cfg = c.repo.read_config("seg")
    assert "versions: [ 1 ]" in cfg and "versions: [ 2 ]" not in cfg
    assert cfg.count("version_policy") == 1
    assert "max_queue_delay_microseconds" in cfg and "'board'" in cfg     # 其他設定原樣保留


def test_load_does_not_override_config():
    """帶 config override 會整份取代 config.pbtxt（metadata / queue delay 消失），load 只能空 body。"""
    sent = {}

    class _Resp:
        status_code = 200
        text = ""

    class _Client:
        def post(self, url, json=None):
            sent["url"], sent["json"] = url, json
            return _Resp()

    c = TritonController.__new__(TritonController)
    c.triton_url = "http://triton:8000"
    c._client = _Client()
    c._load("seg", 3)
    assert sent["url"].endswith("/v2/repository/models/seg/load")
    assert not (sent["json"] or {}).get("parameters")
