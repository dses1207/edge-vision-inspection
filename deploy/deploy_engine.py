"""把 ai-infer/tools/export_engine.py 產出的 engine（model.plan + manifest.json）部署上 Triton。

在 edgevision-control-api 容器內執行（有 /models 掛載與 MINIO / TRITON 環境變數）：
    docker cp /data/engines edgevision-control-api:/tmp/engines
    docker cp deploy/deploy_engine.py edgevision-control-api:/app/deploy/deploy_engine.py
    docker exec edgevision-control-api python /app/deploy/deploy_engine.py /tmp/engines            # 全部
    docker exec edgevision-control-api python /app/deploy/deploy_engine.py /tmp/engines delta echo    # 指定

每個 <dir>/<name>/ 要有 model.plan 與 manifest.json（task / imgsz / class_names / max_batch_size；
fused seg 另加 end2end: true、max_det、fuse_masks: true；pose 另加 n_kpt）。
流程同 deploy_pose_demo.py：plan → MinIO models/<name>/v<version>/model.plan → TritonController.deploy()
（寫 config.pbtxt、explicit load、等 ready）。

⚠️ 這條路【不登錄 Registry】。上線後要跑 tools/register_legacy_models.py 補登錄，
   否則 MCP 看不到、rollback 不能用，第一次重訓還會覆蓋 models/<name>/v1/ 的現役檔案。
"""
import json
import os
import sys
from pathlib import Path

from edgevision.controller.deployer import ModelSpec, TritonController
from edgevision.storage.objects import ObjectStore

root = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/engines")
only = set(sys.argv[2:])
version = int(os.environ.get("ENGINE_VERSION", "1"))

store = ObjectStore(
    endpoint=os.environ["MINIO_ENDPOINT"],
    access_key=os.environ.get("MINIO_ROOT_USER", "edgevision"),
    secret_key=os.environ.get("MINIO_ROOT_PASSWORD", "edgevision-dev-password"),
    secure=False,
)
ctrl = TritonController(triton_url=os.environ["TRITON_URL"],
                        model_repo=os.environ.get("TRITON_MODEL_REPO", "/models"), objects=store)

failed = []
for d in sorted(p for p in root.iterdir() if p.is_dir() and (p / "manifest.json").is_file()):
    name = d.name
    if only and name not in only:
        continue
    m = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
    plan = d / "model.plan"
    if not plan.is_file():
        print(f"!!! {name}: 沒有 model.plan，跳過")
        failed.append(name)
        continue
    key = f"{name}/v{version}/model.plan"
    with plan.open("rb") as f:
        uri = store.put(bucket="models", key=key, data=f, length=plan.stat().st_size)
    # fused seg（NMS + mask 合成在 ONNX 裡，edgevision/training/seg_fuse.py）：輸出是 output0 (max_det, 38) + masks，
    # 不是 raw 的 (38, anchors) + output1。manifest 沒寫就是 raw，行為同以前。
    spec = ModelSpec(task=m["task"], imgsz=int(m["imgsz"]), class_names=list(m["class_names"]),
                     backend="tensorrt", max_batch_size=int(m.get("max_batch_size", 32)), instance_count=1,
                     end2end=bool(m.get("end2end", False)), max_det=int(m.get("max_det", 300)),
                     fuse_masks=bool(m.get("fuse_masks", False)),
                     # pose：輸出 = 4 + nc + n_kpt×3。沒寫 n_kpt 會把姿態模型當偵測模型寫 config，Triton 載入就對不上
                     n_kpt=int(m.get("n_kpt", 0)))
    res = ctrl.deploy(triton_model=name, source_uri=uri, version=version, spec=spec)
    print(f">>> {name}: ok={res.ok} imgsz={m['imgsz']} classes={len(m['class_names'])} msg={res.message} ({res.elapsed_ms:.0f}ms)")
    if not res.ok:
        failed.append(name)
print("失敗:", failed or "無")
sys.exit(1 if failed else 0)
