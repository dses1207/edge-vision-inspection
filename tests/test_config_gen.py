"""config.pbtxt 產生器測試(純邏輯,不需服務)。"""
import sys
import pytest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from edgevision.controller.deployer import LocalModelRepo, TritonController, ModelSpec


def _gen(tmp_path, spec, *, version=None):
    (tmp_path / "m").mkdir()
    ctl = TritonController.__new__(TritonController)
    ctl.repo = LocalModelRepo(tmp_path)
    ctl._write_config("m", spec, version=version)
    return (tmp_path / "m" / "config.pbtxt").read_text()


def test_batch_and_instance(tmp_path):
    cfg = _gen(tmp_path, ModelSpec(max_batch_size=32, instance_count=2,
                                   class_names=["board", "hand"]))
    assert "max_batch_size: 32" in cfg
    assert "dynamic_batching" in cfg
    assert "max_queue_delay_microseconds: 5000" in cfg
    assert "count: 2" in cfg
    # batch>0:dims 不含 batch 維
    assert "dims: [ 3, 640, 640 ]" in cfg
    assert "dims: [ 1, 3, 640, 640 ]" not in cfg


def test_no_batch(tmp_path):
    cfg = _gen(tmp_path, ModelSpec(max_batch_size=0, class_names=["board", "hand"]))
    assert "max_batch_size: 0" in cfg
    assert "dynamic_batching" not in cfg
    assert "instance_group" not in cfg
    # 無 batch:dims 含 batch=1
    assert "dims: [ 1, 3, 640, 640 ]" in cfg


def test_single_instance_no_group(tmp_path):
    """count=1 時不寫 instance_group(用 Triton 預設)。"""
    cfg = _gen(tmp_path, ModelSpec(max_batch_size=32, instance_count=1,
                                   class_names=["a", "b"]))
    assert "instance_group" not in cfg


def test_class_names_in_metadata(tmp_path):
    cfg = _gen(tmp_path, ModelSpec(class_names=["board", "hand", "scratch"]))
    assert "0: 'board'" in cfg
    assert "2: 'scratch'" in cfg
    # 類別數影響 output0 特徵維度: 4 + 3類 + 32mask = 39
    assert "dims: [ 39, 8400 ]" in cfg


def test_detect_task_no_mask(tmp_path):
    """偵測模型(非 seg)沒有 mask prototype 輸出。"""
    cfg = _gen(tmp_path, ModelSpec(task="detect", class_names=["a", "b"]))
    assert "output1" not in cfg   # 沒有 prototype
    # 4 + 2類 + 0mask = 6
    assert "dims: [ 6, 8400 ]" in cfg


# ── version_policy：這一段是為了擋住一個沉默過的 bug ──────────

def test_version_policy_is_written_to_file(tmp_path):
    """⭐ 必須寫進【檔案】,不能只靠 load API 的 config override。

    override 只活在記憶體裡。Triton 一重啟就重讀 config.pbtxt,
    沒有 version_policy 就套用預設的 latest,服務版本【靜默】跳到最新那個。

    實際發生過:seg_alpha 部署 v1,repo 裡另有 v2/v3,重啟後自動切到 v3,
    前端框全消失且 log 無任何錯誤(DB 的 2.9M 筆偵測全是 model_version=1,
    v2/v3 從未產出過任何結果)。
    """
    cfg = _gen(tmp_path, ModelSpec(), version=1)
    assert "version_policy" in cfg
    assert "specific" in cfg
    assert "versions: [ 1 ]" in cfg


def test_version_policy_omitted_when_no_version(tmp_path):
    """不指定版本就不寫 —— 維持 Triton 預設,只適用於單版本模型。"""
    cfg = _gen(tmp_path, ModelSpec())
    assert "version_policy" not in cfg


def test_stage_model_passes_version_through(tmp_path):
    """⚠️ 真正的迴歸點:_stage_model 必須把 version 傳給設定產生器。

    原本的 bug 不在 write_triton_config(它根本沒這個參數),
    而在整條鏈路沒有人負責把版本持久化。
    """
    import inspect
    from edgevision.controller.deployer import TritonController

    src = inspect.getsource(TritonController._stage_model)
    assert "version=version" in src, "_stage_model 沒有把版本傳下去 —— 重啟後會跳版"


def test_fused_masks_writes_masks_not_output1(tmp_path):
    """⭐ fuse_masks：NMS + mask 合成在 ONNX 裡，Triton 輸出 output0 (max_det, 38) + masks，沒有 output1。

    沒有 output1 是重點 —— 那正是要省下的 3.28 MB/幀傳輸。
    """
    cfg = _gen(tmp_path, ModelSpec(task="segment", class_names=["board", "hand"],
                                   fuse_masks=True, max_det=16, max_batch_size=32))
    assert "output1" not in cfg
    assert 'name: "masks"' in cfg
    assert "dims: [ 16, 38 ]" in cfg          # output0: (max_det, 4+2+32)
    assert "dims: [ 16, 160, 160 ]" in cfg    # masks:  (max_det, imgsz/4, imgsz/4)


def test_fused_masks_no_batch_layout(tmp_path):
    cfg = _gen(tmp_path, ModelSpec(task="segment", class_names=["board", "hand"],
                                   fuse_masks=True, max_det=16, max_batch_size=0))
    assert "dims: [ 1, 16, 38 ]" in cfg and "dims: [ 1, 16, 160, 160 ]" in cfg


@pytest.mark.parametrize("names", [["hand", "part", "tray", "part_drop", "hand_bare"], ["a"], [str(i) for i in range(12)]])
def test_fused_output0_width_ignores_class_count(tmp_path, names):
    """合成後 output0 是 NMS 之後的 [x1,y1,x2,y2,conf,cls]+32 = 38，跟類別數無關。
    2026-11-04：寫成 4+類別數+32，5 類變 41 → wash_pickup_seg 臨時 Triton 載不起來。"""
    cfg = _gen(tmp_path, ModelSpec(task="segment", class_names=names,
                                   fuse_masks=True, max_det=32, max_batch_size=32))
    assert "dims: [ 32, 38 ]" in cfg


def test_fused_masks_default_off_keeps_output1(tmp_path):
    """預設關閉：既有 seg 模型的 config 一個字都不能變。"""
    cfg = _gen(tmp_path, ModelSpec(task="segment", class_names=["board", "hand"]))
    assert 'name: "output1"' in cfg and "masks" not in cfg


@pytest.mark.parametrize("imgsz,anchors", [(640, 8400), (1024, 21504), (1280, 33600)])
def test_anchor_count_follows_imgsz(tmp_path, imgsz, anchors):
    """⚠️ 曾寫死 8400：1024 的模型（IPCAM 五個）全部 load 失敗，output0 dims 不符。"""
    cfg = _gen(tmp_path, ModelSpec(task="detect", class_names=["a", "b"], imgsz=imgsz))
    assert f"dims: [ 6, {anchors} ]" in cfg
