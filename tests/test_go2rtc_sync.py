"""go2rtc 相機清單同步：相機表 → Secret → 清單變了才換 go2rtc Pod。"""
import base64
import sys
from pathlib import Path
from types import SimpleNamespace as NS

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from edgevision.orchestrator.go2rtc_sync import (  # noqa: E402
    ANN_STREAMS, SECRET_KEY, Go2rtcSync, content_hash, redact, render, targets,
)
from edgevision.orchestrator.sites import parse_sites  # noqa: E402

SITES = parse_sites({"siteb": {"lines": ["siteb_*"], "triton": "triton-siteb", "gpu_claim": "g", "triton_repo": "s3://m/sm89",
                            "go2rtc_source": "rtsp://192.0.2.75:8554"}})


def _details():
    return [
        {"id": "line_c", "enabled": True, "cameras": [
            {"id": "cam65", "ip": "192.0.2.112", "cred_ref": "visual", "stream_path": "/stream2", "enabled": True},
            {"id": "cam87", "ip": "192.0.2.137", "cred_ref": "visual", "enabled": False}]},     # 停用也註冊
        {"id": "webcam_golf", "enabled": False, "cameras": [
            {"id": "cam_192_0_2_199", "ip": "192.0.2.199", "cred_ref": "none", "stream_path": "/stream"}]},
        {"id": "siteb_alpha", "enabled": True, "cameras": [
            {"id": "siteb123", "ip": "192.0.2.156", "cred_ref": "camgrp", "stream_path": "/stream2"}]},
    ]


def test_targets_direct_relay_and_site(monkeypatch):
    monkeypatch.setenv("CAM_USER", "u")
    monkeypatch.setenv("CAM_PASS", "p")
    main, ct = targets(_details(), SITES)
    assert (main.deployment, main.secret, ct.deployment, ct.secret) == (
        "go2rtc", "go2rtc-streams", "go2rtc-siteb", "go2rtc-siteb-streams")
    s = main.streams
    assert s["cam65"] == "rtsp://u:p@192.0.2.112:554/stream2#backchannel=0#video=copy#audio="
    assert s["cam87"].endswith("192.0.2.137:554/stream2#backchannel=0#video=copy#audio=")      # 沒 stream_path = /stream2
    assert s["cam_192_0_2_199"].startswith("rtsp://192.0.2.199:554/stream#")              # cred_ref=none 不帶帳密
    assert s["siteb123"] == "rtsp://192.0.2.75:8554/siteb123"                                       # 站點的線：主 go2rtc 轉拉
    assert all(s[k + "_marked"] is None for k in ("cam65", "cam87", "siteb123"))                  # 打框串流來源留空
    assert not any(k.endswith("_main") for k in s)
    assert ct.streams == {"siteb123": "rtsp://u:p@192.0.2.156:554/stream2#backchannel=0#video=copy#audio="}

    # 測試叢集：全部從正式機轉拉，不需要帳密
    monkeypatch.delenv("CAM_USER")
    (relay,) = targets(_details()[:2], {}, relay="rtsp://192.0.2.170:8554/")
    assert relay.streams["cam65"] == "rtsp://192.0.2.170:8554/cam65"


def test_render_and_redacted_hash(monkeypatch):
    monkeypatch.setenv("CAM_USER", "u")
    monkeypatch.setenv("CAM_PASS", "secret")
    text = render(targets(_details()[:1], {})[0].streams)
    assert '  cam65:\n    - "rtsp://u:secret@' in text and "\n  cam65_marked:\n" in text
    assert "secret" not in redact(text)
    monkeypatch.setenv("CAM_PASS", "other")
    assert content_hash(render(targets(_details()[:1], {})[0].streams)) == content_hash(text)  # hash 跟密碼無關


class FakeK8s:
    def __init__(self, secret_text="", ann=None, missing=()):
        self.secret = {"data": {SECRET_KEY: base64.b64encode(secret_text.encode()).decode()}}
        self.ann = dict(ann or {})
        self.missing = set(missing)
        self.secret_patches, self.dep_patches = [], []

    def read_namespaced_deployment(self, name, ns):
        if name in self.missing:
            raise _NotFound()
        return NS(spec=NS(template=NS(metadata=NS(annotations=self.ann))))

    def read_namespaced_secret(self, name, ns):
        return NS(data=self.secret["data"])

    def patch_namespaced_secret(self, name, ns, body):
        self.secret_patches.append(name)
        self.secret["data"] = body["data"]

    def patch_namespaced_deployment(self, name, ns, body):
        self.dep_patches.append(name)
        self.ann.update(body["spec"]["template"]["metadata"]["annotations"])


class _NotFound(Exception):
    status = 404


def test_sync_writes_secret_and_restarts_only_on_change(monkeypatch):
    monkeypatch.setenv("CAM_USER", "u")
    monkeypatch.setenv("CAM_PASS", "p")
    k = FakeK8s("  cam65:\n", missing={"go2rtc-siteb"})
    s = Go2rtcSync(core_api=k, apps_api=k, namespace="edgevision")
    assert s.once(_details(), SITES) == ["go2rtc"]                 # 站點那台不存在（測試叢集）→ 略過
    assert k.secret_patches == ["go2rtc-streams"] and k.dep_patches == ["go2rtc"]
    assert "cam65_marked" in base64.b64decode(k.secret["data"][SECRET_KEY]).decode()
    assert s.once(_details(), SITES) == []                         # 沒變：不寫、不換
    assert len(k.secret_patches) == 1 and len(k.dep_patches) == 1

    # Secret 已是新的但上次換 Pod 失敗（範本 hash 不對）→ 這輪補換，不再寫 Secret
    k.ann[ANN_STREAMS] = "old"
    assert s.once(_details(), SITES) == ["go2rtc"] and len(k.secret_patches) == 1


def test_sync_skips_on_no_cameras_or_missing_credentials(monkeypatch):
    k = FakeK8s("  cam65:\n")
    s = Go2rtcSync(core_api=k, apps_api=k, namespace="edgevision")
    assert s.once([{"id": "x", "cameras": []}], {}) == []          # 一台都沒有：不清空
    monkeypatch.delenv("CAM_USER", raising=False)
    assert s.once(_details()[:1], {}) == []                        # 帳密解析失敗：不動
    assert k.secret_patches == [] and k.dep_patches == []
