"""推流編碼器選擇：nvenc 預設、x264 溢出用。"""
import pytest
from ds_runner.pipeline import encoder_spec


def test_default_is_nvenc_on_nvmm_nv12():
    caps, kind, props = encoder_spec({}, 1920, 1080)
    assert kind == "nvv4l2h264enc"
    assert caps == "video/x-raw(memory:NVMM), format=NV12, width=1920, height=1080"
    assert ("bitrate", 2_000_000) in props and ("num-B-Frames", 0) in props


def test_x264_uses_system_memory_i420_zerolatency_ultrafast():
    caps, kind, props = encoder_spec({"ENCODER": "x264", "RESTREAM_BITRATE": "3000000"}, 1280, 720)
    assert kind == "x264enc"
    assert caps == "video/x-raw, format=I420, width=1280, height=720"   # 沒有 memory:NVMM → nvvideoconvert 拷回 CPU
    p = dict(props)
    assert p["bitrate"] == 3000                # kbit/s
    assert p["tune"] == 0x4 and p["speed-preset"] == 1 and p["bframes"] == 0
    assert p["threads"] == 2                   # 不設會開滿核心


def test_x264_threads_and_preset_overridable():
    p = dict(encoder_spec({"ENCODER": "x264", "X264_THREADS": "4", "X264_PRESET": "2"}, 640, 480)[2])
    assert p["threads"] == 4 and p["speed-preset"] == 2


def test_unknown_encoder_is_fatal():
    with pytest.raises(SystemExit):
        encoder_spec({"ENCODER": "nvenc2"}, 640, 480)


def test_auto_is_default_and_starts_with_nvenc():
    from ds_runner.pipeline import encoder_kind
    assert encoder_kind({}) == "auto"
    assert encoder_spec({"ENCODER": "auto"}, 640, 480)[1] == "nvv4l2h264enc"
    assert encoder_spec({"ENCODER": "auto"}, 640, 480, "x264")[1] == "x264enc"   # 退回 x264 時明確指定


class _El:
    def __init__(self, name, parent=None):
        self.name, self.parent = name, parent

    def get_name(self):
        return self.name

    def get_parent(self):
        return self.parent


def test_branch_index_finds_camera_and_role():
    from ds_runner.pipeline import branch_index
    pipe = _El("pipeline0")
    assert branch_index(_El("renc_7", pipe), pipe) == (7, "enc", True)
    inner = _El("udpsink0", _El("rtpbin", _El("rsink_12", pipe)))   # rtspclientsink 內部元件
    assert branch_index(inner, pipe) == (12, "sink", True)
    assert branch_index(_El("renc_7"), pipe) == (7, "enc", False)     # 已移出管線的舊元件
    assert branch_index(_El("src_3", pipe), pipe) is None            # 來源不是推流分支
    assert branch_index(_El("rfoo_3", pipe), pipe) is None
