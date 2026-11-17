"""多模型並行 + batching config 整合測試。"""
import io
import os
import time

import numpy as np
import pytest

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def two_models(minio):
    """用假 ONNX 部署兩個獨立模型 model-a / model-b。"""
    onnx = pytest.importorskip("onnx")
    from onnx import helper, TensorProto
    import tempfile

    def build(cx):
        out0 = np.zeros((1, 38, 8400), np.float32)
        out0[0,0,0]=cx; out0[0,1,0]=320; out0[0,2,0]=260; out0[0,3,0]=300; out0[0,4,0]=0.95
        out0[0,0,1]=cx+120; out0[0,1,1]=300; out0[0,2,1]=80; out0[0,3,1]=90; out0[0,5,1]=0.88
        proto=(np.random.RandomState(1).randn(1,32,160,160)*0.1).astype(np.float32)
        c0=helper.make_node("Constant",[],["output0"],value=helper.make_tensor("v0",TensorProto.FLOAT,out0.shape,out0.flatten()))
        c1=helper.make_node("Constant",[],["output1"],value=helper.make_tensor("v1",TensorProto.FLOAT,proto.shape,proto.flatten()))
        sh=helper.make_node("Shape",["images"],["_s"])
        g=helper.make_graph([sh,c0,c1],"m",
            [helper.make_tensor_value_info("images",TensorProto.FLOAT,[1,3,640,640])],
            [helper.make_tensor_value_info("output0",TensorProto.FLOAT,list(out0.shape)),
             helper.make_tensor_value_info("output1",TensorProto.FLOAT,list(proto.shape))])
        m=helper.make_model(g,opset_imports=[helper.make_opsetid("",13)]); m.ir_version=9
        f=tempfile.mktemp(suffix=".onnx"); onnx.save(m,f); return open(f,"rb").read()

    from edgevision.controller.deployer import TritonController, ModelSpec
    ctl=TritonController(triton_url=os.environ.get("TRITON_URL","http://localhost:8000"),
                         model_repo=os.environ.get("TRITON_MODEL_REPO","/models"),objects=minio)
    for name,cx in [("mm-a",200.0),("mm-b",440.0)]:
        data=build(cx)
        minio.put(bucket="models",key=f"{name}/v1/model.onnx",data=io.BytesIO(data),length=len(data))
        ctl.deploy(triton_model=name,source_uri=f"s3://models/{name}/v1/model.onnx",version=1,
                   spec=ModelSpec(class_names=["board","hand"]))
    yield ["mm-a","mm-b"]
    import httpx
    for n in ["mm-a","mm-b"]:
        try: httpx.post(f"{ctl.triton_url}/v2/repository/models/{n}/unload",timeout=10)
        except Exception: pass
        minio.remove_prefix(bucket="models", prefix=f"{n}/")
    ctl.close()


def test_parallel_merges_results(two_models):
    """兩模型並行,結果合併、來源可區分。"""
    from edgevision.inference.predictor_triton import TritonPredictor
    from edgevision.inference.predictor_multi import MultiPredictor
    url=os.environ.get("TRITON_URL","http://localhost:8000").replace("http://","")
    multi=MultiPredictor({
        "mm-a":TritonPredictor(url=url,model_name="mm-a",imgsz=640,conf_threshold=0.5),
        "mm-b":TritonPredictor(url=url,model_name="mm-b",imgsz=640,conf_threshold=0.5),
    })
    res=multi.predict(np.zeros((720,1280,3),np.uint8))
    multi.close()
    # 兩模型各 2 偵測 → 合併 4
    assert len(res.detections)==4
    sources={d.metrics.get("source_model") for d in res.detections}
    assert sources=={"mm-a","mm-b"}
    # 版本可追溯
    assert "mm-a" in multi.model_version and "mm-b" in multi.model_version



