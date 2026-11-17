"""部署鏈路整合測試:版本切換與回退。

需要 Triton 運行,且測試會自己建假 ONNX 模型上傳 MinIO。
"""
import io
import os
import time

import numpy as np
import pytest

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def onnx_models(minio):
    """建兩個假 seg ONNX(v1 框偏左、v2 框偏右)並上傳。"""
    onnx = pytest.importorskip("onnx")
    from onnx import helper, TensorProto

    def build(version: int, cx: float) -> bytes:
        N, feat = 8400, 38
        out0 = np.zeros((1, feat, N), dtype=np.float32)
        out0[0, 0, 0] = cx; out0[0, 1, 0] = 320
        out0[0, 2, 0] = 260; out0[0, 3, 0] = 300; out0[0, 4, 0] = 0.95
        out0[0, 0, 1] = cx + 120; out0[0, 1, 1] = 300
        out0[0, 2, 1] = 80; out0[0, 3, 1] = 90; out0[0, 5, 1] = 0.88
        proto = (np.random.RandomState(version).randn(1, 32, 160, 160) * 0.1).astype(np.float32)
        c0 = helper.make_node("Constant", [], ["output0"],
                              value=helper.make_tensor("v0", TensorProto.FLOAT, out0.shape, out0.flatten()))
        c1 = helper.make_node("Constant", [], ["output1"],
                              value=helper.make_tensor("v1", TensorProto.FLOAT, proto.shape, proto.flatten()))
        sh = helper.make_node("Shape", ["images"], ["_s"])
        g = helper.make_graph([sh, c0, c1], f"seg_v{version}",
            [helper.make_tensor_value_info("images", TensorProto.FLOAT, [1,3,640,640])],
            [helper.make_tensor_value_info("output0", TensorProto.FLOAT, list(out0.shape)),
             helper.make_tensor_value_info("output1", TensorProto.FLOAT, list(proto.shape))])
        m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 13)])
        m.ir_version = 9
        import onnx as _o, tempfile
        f = tempfile.mktemp(suffix=".onnx")
        _o.save(m, f)
        return open(f, "rb").read()

    for v, cx in [(1, 200.0), (2, 440.0)]:
        data = build(v, cx)
        minio.put(bucket="models", key=f"itest-seg/v{v}/model.onnx",
                  data=io.BytesIO(data), length=len(data))
    yield ["itest-seg"]
    # 清理:測試不留垃圾在 MinIO
    minio.remove_prefix(bucket="models", prefix="itest-seg/")


@pytest.fixture(scope="module")
def controller(minio):
    from edgevision.controller.deployer import TritonController
    repo = os.environ.get("TRITON_MODEL_REPO", "/models")
    c = TritonController(
        triton_url=os.environ.get("TRITON_URL", "http://localhost:8000"),
        model_repo=repo, objects=minio,
    )
    yield c
    # 清理
    try:
        import httpx
        httpx.post(f"{c.triton_url}/v2/repository/models/itest-seg/unload", timeout=10)
    except Exception:
        pass
    c.close()


def test_deploy_v1(controller, onnx_models):
    r = controller.deploy(triton_model="itest-seg",
                          source_uri="s3://models/itest-seg/v1/model.onnx", version=1)
    assert r.ok, r.message
    assert controller.current_version("itest-seg") == 1


def test_switch_to_v2(controller, onnx_models):
    r = controller.deploy(triton_model="itest-seg",
                          source_uri="s3://models/itest-seg/v2/model.onnx", version=2)
    assert r.ok, r.message
    assert controller.current_version("itest-seg") == 2


def test_rollback_to_v1(controller, onnx_models):
    r = controller.rollback(triton_model="itest-seg", to_version=1)
    assert r.ok, r.message
    assert controller.current_version("itest-seg") == 1


def test_version_changes_inference(controller, onnx_models):
    """核心驗證:切版本,推論結果真的跟著變。"""
    from edgevision.inference.predictor_triton import TritonPredictor
    url = os.environ.get("TRITON_URL", "http://localhost:8000").replace("http://", "")
    img = np.zeros((720, 1280, 3), np.uint8)

    results = {}
    for v in [1, 2]:
        controller.deploy(triton_model="itest-seg",
                          source_uri=f"s3://models/itest-seg/v{v}/model.onnx", version=v)
        time.sleep(0.3)
        p = TritonPredictor(url=url, model_name="itest-seg", imgsz=640, conf_threshold=0.5)
        res = p.predict(img)
        boards = [d for d in res.detections if d.class_name == "board"]
        assert boards, f"v{v} 應偵測到 board"
        results[v] = boards[0].bbox.center[0]
        # 類別對應正確
        assert {d.class_name for d in res.detections} == {"board", "hand"}
        p.close()

    # v2 的框應該比 v1 偏右
    assert results[2] > results[1], f"版本切換未改變輸出: v1={results[1]}, v2={results[2]}"
