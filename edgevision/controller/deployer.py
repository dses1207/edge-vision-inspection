"""部署 Controller：把模型從倉庫載入到 Triton。

在架構中的角色：

    ClearML Registry（tag=production → v15）
            │ Controller 查詢
            ▼
    ① 從 MinIO 下載 model.onnx/plan 到 Triton 的 model repository
    ② 呼叫 Triton load API，用 config override 指定要服務的版本
    ③ 確認載入成功，清理過舊版本

為什麼需要這一層：
  ・ClearML 用 tag 表示「哪一版是 production」，Triton 只認數字版本
    —— 兩者之間需要橋接。
  ・Registry 的檔案佈局（models/<name>/vN/model.plan）不是 Triton repository 的佈局
    （<name>/<N>/model.plan + config.pbtxt），要有人整理過去。

model repository 兩種放法（TRITON_MODEL_REPO 決定，Triton 的 --model-repository 要指向同一處）：
  ・本地目錄 /models   ：RWO PVC，control-api 與 Triton 必須同節點（現行）
  ・s3://host:port/bucket/prefix：Triton 官方支援直讀 MinIO，load 時自己下載 → 拆掉儲存的節點綁定
    。這裡用 ObjectStore（MINIO_ENDPOINT）寫入，URI 的 host:port 只給 Triton 看。

版本策略：用 load API 的 config override 指定 specific version，
讓 Triton 只服務我們要的那一版。這樣「現在跑哪一版」的真相在
Registry（tag），推論服務程式碼永遠不碰版本號。
"""

from __future__ import annotations

import io
import logging
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class DeployResult:
    ok: bool
    model: str
    version: int
    message: str
    elapsed_ms: float


@dataclass(slots=True)
class ModelSpec:
    """模型的部署規格,決定 config.pbtxt 怎麼產生。

    設定驅動 —— 要調 batch/instance 改這裡,不改程式碼。
    """

    task: str = "segment"          # segment | detect | pose
    imgsz: int = 640
    class_names: list[str] = field(default_factory=lambda: ["board", "hand"])
    backend: str = "onnxruntime"   # onnxruntime | tensorrt

    # ── 批次(dynamic batching)──────────────────────────
    # 最多把幾個請求併成一批。0 = 不 batch。
    # ⚠️ 模型 export 時的 batch 參數必須 >= 此值,否則載入失敗。
    #    Ultralytics: yolo export ... dynamic=True batch=32
    max_batch_size: int = 32
    # 湊批的等待窗(微秒)。到時間或湊滿都會觸發推論。
    # 太大 → 延遲高;太小 → 湊不滿,batch 效益差。
    max_queue_delay_us: int = 5000

    # ── 並行(instance group)────────────────────────────
    # 同一模型在 GPU 上載入幾份副本。每份吃一份顯存,但能真正並行。
    # 大量相機時靠增加此值分流,不是靠加大 batch。
    instance_count: int = 1

    # ── 輸出佈局 ───────────────────────────────────────
    # YOLOv8/11：grid 輸出 (features, 8400)，後處理要做 NMS
    # YOLO26  ：end2end 輸出 (max_det, 6)=[x1,y1,x2,y2,conf,cls]，不需 NMS
    end2end: bool = False
    max_det: int = 300
    # seg 模型 NMS + mask 合成都在 ONNX 裡：output0 (max_det, 38) + masks (max_det, H/4, W/4)，無 output1
    fuse_masks: bool = False
    # pose 模型：每個 instance 的 keypoint 數（每點 x,y,conf 三個值）。detect/segment 為 0
    n_kpt: int = 0

    @classmethod
    def from_meta(cls, meta, *, backend: str | None = None, instance_count: int = 1) -> "ModelSpec":
        """從 Registry 的 ModelMetadata 產生部署規格：類別、imgsz、batch 都以模型身分為準。"""
        be = backend or ("tensorrt" if meta.artifacts.tensorrt else "onnxruntime")
        return cls(task=meta.task, imgsz=meta.imgsz, class_names=list(meta.class_names),
                   backend=be, max_batch_size=meta.triton_max_batch, instance_count=instance_count,
                   end2end=meta.export.end2end, max_det=meta.export.max_det,
                   fuse_masks=getattr(meta.export, "fuse_masks", False),
                   n_kpt=getattr(meta.export, "n_kpt", 0))

    @property
    def num_classes(self) -> int:
        # 從 class_names 推導,避免兩個欄位不同步
        return len(self.class_names)

    @property
    def n_mask(self) -> int:
        return 32 if self.task == "segment" else 0

    @property
    def features(self) -> int:
        return 4 + self.num_classes + self.n_mask + (self.n_kpt * 3 if self.task == "pose" else 0)


def write_triton_config(repo: Path, model: str, spec: "ModelSpec",
                        *, version: int | None = None) -> Path:
    """產生 config.pbtxt 寫進本地 repository（內容見 render_triton_config）。"""
    path = repo / model / "config.pbtxt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_triton_config(model, spec, version=version))
    return path


def render_triton_config(model: str, spec: "ModelSpec", *, version: int | None = None) -> str:
    """產生 config.pbtxt 內容。

    ⚠️ 四個容易錯的地方(官方文件確認):

    1. metadata 必須寫入,否則類別對應會錯。

    2. batch 維度規則:max_batch_size > 0 時,input/output 的 dims
       【不含 batch 維】—— Triton 會自動加上。設 0 時才要自己寫 [1,...]。
       寫錯會導致模型載入失敗或 shape 不符。

    3. TensorRT engine 的 batch 上限由 export 時決定。config 的
       max_batch_size 不能超過 export 的 batch,否則載入報錯。

    4. ⭐ version_policy 必須寫進【檔案】,不能只靠 load API 的 config override。
       override 只活在記憶體裡:Triton 一重啟就重讀 config.pbtxt,
       沒寫 version_policy 就套用預設的 latest —— 服務版本會【靜默】跳到最新的那個。

       實際發生過:seg_alpha 部署的是 v1,但 repo 裡另有 v2/v3。
       重啟 Triton 後自動切到 v3,前端框全部消失,而 log 沒有任何錯誤
       (DB 顯示 2.9M 筆偵測全是 model_version=1,v2/v3 從未產出過任何結果)。

       給 version 就釘住它;不給則不寫(維持 Triton 預設,僅適用於單版本模型)。
    """
    engine_line = (
        'backend: "onnxruntime"' if spec.backend == "onnxruntime"
        else 'platform: "tensorrt_plan"'
    )

    names_dict = "{" + ", ".join(
        f"{i}: '{n}'" for i, n in enumerate(spec.class_names)
    ) + "}"
    metadata = "{'names': " + names_dict + "}"

    imgsz, feat, n_mask = spec.imgsz, spec.features, spec.n_mask
    # ⚠️ fuse_masks 的 output0 是 NMS 之後的 [x1,y1,x2,y2,conf,cls]+32 係數 = 38，跟類別數無關。
    #    以前寫成 feat（4+類別數+32）：2 類剛好也是 38 沒露餡，5 類變 41 → Triton 載不起來
    #    （2026-11-04 wash_pickup_seg v4–v6 finalize 連續失敗）。
    fused_w = 4 + 2 + n_mask
    proto_h = proto_w = imgsz // 4
    # ⚠️ anchor 數跟 imgsz 走：三個 stride（8/16/32）的格子數加總。640 → 8400、1024 → 21504。
    #    曾經寫死 8400：IPCAM 五個 1024 模型全部載不起來（tensor 'output0' dims 不符）。
    anchors = sum((imgsz // s) ** 2 for s in (8, 16, 32))

    lines = [f'name: "{model}"', engine_line]

    if spec.max_batch_size > 0:
        # 有 batch:dims 不含 batch 維,Triton 自動加
        lines.append(f'max_batch_size: {spec.max_batch_size}')
        img_dims = f'[ 3, {imgsz}, {imgsz} ]'
        if spec.fuse_masks:
            out0_dims = f'[ {spec.max_det}, {fused_w} ]'
        elif spec.end2end:
            out0_dims = f'[ {spec.max_det}, 6 ]'
        else:
            out0_dims = f'[ {feat}, {anchors} ]'
        out1_dims = f'[ {n_mask}, {proto_h}, {proto_w} ]'
        masks_dims = f'[ {spec.max_det}, {proto_h}, {proto_w} ]'
    else:
        # 無 batch:dims 含明確的 batch=1
        lines.append('max_batch_size: 0')
        img_dims = f'[ 1, 3, {imgsz}, {imgsz} ]'
        if spec.fuse_masks:
            out0_dims = f'[ 1, {spec.max_det}, {fused_w} ]'
        elif spec.end2end:
            out0_dims = f'[ 1, {spec.max_det}, 6 ]'
        else:
            out0_dims = f'[ 1, {feat}, {anchors} ]'
        out1_dims = f'[ 1, {n_mask}, {proto_h}, {proto_w} ]'
        masks_dims = f'[ 1, {spec.max_det}, {proto_h}, {proto_w} ]'

    lines.append(
        f'input [{{ name: "images" data_type: TYPE_FP32 dims: {img_dims} }}]'
    )
    outputs = [f'{{ name: "output0" data_type: TYPE_FP32 dims: {out0_dims} }}']
    if n_mask and spec.fuse_masks:
        # ⭐ 合成後沒有 output1：Triton 不再搬 prototypes，每偵測直接一張 mask
        outputs.append(
            f'{{ name: "masks" data_type: TYPE_FP32 dims: {masks_dims} }}'
        )
    elif n_mask:
        outputs.append(
            f'{{ name: "output1" data_type: TYPE_FP32 dims: {out1_dims} }}'
        )
    lines.append('output [\n  ' + ',\n  '.join(outputs) + '\n]')

    # dynamic batching:把多個請求併批。多支程式共用同一模型時,
    # 各自的請求會在此匯集成一批,一次推論。
    if spec.max_batch_size > 0:
        lines.append(
            'dynamic_batching {\n'
            f'  max_queue_delay_microseconds: {spec.max_queue_delay_us}\n'
            '}'
        )

    # instance group:同一模型在 GPU 上載入幾份副本,真正並行。
    # 大量相機時靠這個分流,不是靠加大 batch。
    if spec.instance_count > 1:
        lines.append(
            'instance_group [\n'
            f'  {{ count: {spec.instance_count} kind: KIND_GPU }}\n'
            ']'
        )

    # ⭐ 釘住服務版本 —— 見上面第 4 點。這一段的有無決定 Triton 重啟後跑哪一版。
    if version is not None:
        lines.append(
            'version_policy {\n'
            f'  specific {{ versions: [ {version} ] }}\n'
            '}'
        )

    lines.append(
        f'parameters {{ key: "metadata" value {{ string_value: "{metadata}" }} }}'
    )
    return "\n".join(lines) + "\n"


# ── model repository 的兩種放法 ─────────────────────────────

class LocalModelRepo:
    """本地目錄（PVC）。"""

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)

    def __str__(self) -> str:
        return str(self.root)

    def stage_file(self, objects, source_uri: str, model: str, version: int, filename: str) -> int:
        vdir = self.root / model / str(version)
        vdir.mkdir(parents=True, exist_ok=True)
        bucket, key = objects.parse_uri(source_uri)
        data = objects.get(bucket=bucket, key=key)
        (vdir / filename).write_bytes(data)
        return len(data)

    def write_config(self, model: str, text: str) -> None:
        path = self.root / model / "config.pbtxt"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)

    def read_config(self, model: str) -> str:
        return (self.root / model / "config.pbtxt").read_text()

    def has_version(self, model: str, version: int) -> bool:
        return (self.root / model / str(version)).exists()

    def versions(self, model: str) -> list[int]:
        mdir = self.root / model
        if not mdir.is_dir():
            return []
        return [int(d.name) for d in mdir.iterdir() if d.is_dir() and d.name.isdigit()]

    def remove_version(self, model: str, version: int) -> None:
        shutil.rmtree(self.root / model / str(version), ignore_errors=True)


class S3ModelRepo:
    """MinIO 上的 repository：s3://host:port/bucket/prefix（Triton 的寫法，host:port 這裡不用）。"""

    def __init__(self, uri: str, objects) -> None:
        self.uri = uri
        self.objects = objects
        self.bucket, self.prefix = self.parse(uri)

    def __str__(self) -> str:
        return self.uri

    @staticmethod
    def parse(uri: str) -> tuple[str, str]:
        """s3://[http(s)://]host:port/bucket/prefix → (bucket, "prefix/")。Triton 自訂 endpoint 一定帶 host:port。"""
        rest = uri[len("s3://"):]
        for scheme in ("http://", "https://"):
            if rest.startswith(scheme):
                rest = rest[len(scheme):]
        parts = [p for p in rest.split("/") if p]
        if len(parts) < 2 or ":" not in parts[0]:
            raise ValueError(f"TRITON_MODEL_REPO 要寫成 s3://host:port/bucket[/prefix]：{uri}")
        bucket = parts[1]
        prefix = "/".join(parts[2:])
        return bucket, (prefix + "/" if prefix else "")

    def _key(self, *parts) -> str:
        return self.prefix + "/".join(str(p) for p in parts)

    def stage_file(self, objects, source_uri: str, model: str, version: int, filename: str) -> int:
        src_bucket, src_key = objects.parse_uri(source_uri)
        objects.copy(src_bucket=src_bucket, src_key=src_key, bucket=self.bucket,
                     key=self._key(model, version, filename))
        return -1                                   # 伺服器端複製，不經手位元組

    def write_config(self, model: str, text: str) -> None:
        data = text.encode()
        self.objects.put(bucket=self.bucket, key=self._key(model, "config.pbtxt"),
                         data=io.BytesIO(data), length=len(data), content_type="text/plain")

    def read_config(self, model: str) -> str:
        return self.objects.get(bucket=self.bucket, key=self._key(model, "config.pbtxt")).decode()

    def has_version(self, model: str, version: int) -> bool:
        return self.objects.exists_prefix(bucket=self.bucket, prefix=self._key(model, version) + "/")

    def versions(self, model: str) -> list[int]:
        names = self.objects.list_prefixes(bucket=self.bucket, prefix=self._key(model) + "/")
        return [int(n) for n in names if n.isdigit()]

    def remove_version(self, model: str, version: int) -> None:
        self.objects.remove_prefix(bucket=self.bucket, prefix=self._key(model, version) + "/")


def open_model_repo(location: Path | str, objects):
    loc = str(location)
    return S3ModelRepo(loc, objects) if loc.startswith("s3://") else LocalModelRepo(loc)



class TritonController:
    """透過 Triton HTTP API 管理模型的載入與版本。"""

    def __init__(
        self,
        *,
        triton_url: str,
        model_repo: Path | str,
        objects,                      # ObjectStore
        keep_versions: int = 3,
        ready_timeout: float = 15.0,
    ) -> None:
        self.triton_url = triton_url.rstrip("/")
        # 載入後等就緒的秒數。別的節點上的 Triton 要跨節點從 MinIO 拉 plan，15 秒不一定夠
        self.ready_timeout = ready_timeout
        self.repo = open_model_repo(model_repo, objects)
        self.objects = objects
        self.keep_versions = keep_versions
        self._client = httpx.Client(timeout=30.0)

    def close(self) -> None:
        self._client.close()

    # ── 主流程 ────────────────────────────────────────────

    def deploy(
        self,
        *,
        triton_model: str,
        source_uri: str,
        version: int,
        spec: "ModelSpec | None" = None,
    ) -> DeployResult:
        """把指定版本部署上線。

        source_uri: s3://models/seg-alpha/v2/model.onnx
        version:    要在 Triton 中服務的版本號
        spec:       部署規格(batch/instance/類別等)。未給則用預設。
        """
        spec = spec or ModelSpec()
        t0 = time.perf_counter()
        try:
            # ① 確保檔案在 model repository
            self._stage_model(triton_model, source_uri, version, spec)
            # ② 載入指定版本
            self._load(triton_model, version)
            # ③ 確認就緒
            if not self._wait_ready(triton_model, version):
                raise RuntimeError("載入後模型未就緒")
            # ④ 清理過舊版本（保留最近 N 版供快速回退）
            self._prune(triton_model, keep_around=version)

            elapsed = (time.perf_counter() - t0) * 1000
            logger.info("[%s] v%d 部署成功（%.0fms）", triton_model, version, elapsed)
            return DeployResult(True, triton_model, version, "部署成功", elapsed)

        except Exception as exc:
            elapsed = (time.perf_counter() - t0) * 1000
            logger.exception("[%s] v%d 部署失敗", triton_model, version)
            return DeployResult(False, triton_model, version, str(exc), elapsed)

    def deploy_from_registry(self, registry, model_name: str, *, tag: str = "production",
                             version: int | None = None) -> DeployResult:
        """Registry 說哪版就部署哪版。

        version 給了就部署該版（回退用），否則解析 tag。
        """
        resolved = registry.get(model_name, version) if version else registry.resolve(model_name, tag=tag)
        if resolved is None:
            return DeployResult(False, model_name, version or 0,
                                f"Registry 找不到 {model_name}（tag={tag}, version={version}）", 0)
        spec = ModelSpec.from_meta(resolved.meta)
        return self.deploy(triton_model=model_name, source_uri=resolved.source_uri,
                           version=resolved.version, spec=spec)

    def rollback(self, *, triton_model: str, to_version: int, **kw) -> DeployResult:
        """回退到舊版本。

        若該版本檔案仍在 repository（在保留範圍內）→ 秒級完成，不用重新下載。
        """
        logger.info("[%s] 回退到 v%d", triton_model, to_version)
        if self.repo.has_version(triton_model, to_version):
            t0 = time.perf_counter()
            try:
                self._pin_version(triton_model, to_version)
                self._load(triton_model, to_version)
                if not self._wait_ready(triton_model, to_version):
                    raise RuntimeError("回退後模型未就緒")
                elapsed = (time.perf_counter() - t0) * 1000
                logger.info("[%s] 已回退到 v%d（%.0fms，檔案已在本地）",
                            triton_model, to_version, elapsed)
                return DeployResult(True, triton_model, to_version, "回退成功", elapsed)
            except Exception as exc:
                return DeployResult(False, triton_model, to_version, str(exc), 0)
        # 檔案已被清掉，需重新部署(需要 source_uri 與 spec)
        logger.info("[%s] v%d 檔案已清理，重新下載", triton_model, to_version)
        return self.deploy(triton_model=triton_model, version=to_version, **kw)

    def current_version(self, triton_model: str) -> int | None:
        """查詢目前 Triton 正在服務的版本。"""
        r = self._client.post(
            f"{self.triton_url}/v2/repository/index",
            json={},
        )
        r.raise_for_status()
        for entry in r.json():
            if entry.get("name") == triton_model and entry.get("state") == "READY":
                return int(entry.get("version", 0))
        return None

    # ── 內部 ──────────────────────────────────────────────

    def _stage_model(
        self, model: str, source_uri: str, version: int, spec: "ModelSpec",
    ) -> None:
        """把 Registry 的模型檔放進 repository，產生 config.pbtxt。"""
        filename = "model.onnx" if spec.backend == "onnxruntime" else "model.plan"

        # 一律重新下載覆蓋：同版本號的 plan 內容會變（換 GPU 重建 engine 時版本仍是 v1），
        # 「檔案在就跳過」會讓 Triton 永遠載到舊卡的 plan（compute capability 對不上，10/6 踩過）。
        # 回退的秒級路徑在 rollback() 直接 _load，不經過這裡，不受影響。
        n = self.repo.stage_file(self.objects, source_uri, model, version, filename)
        logger.info("[%s] v%d 已放進 repository %s（%s）", model, version, self.repo,
                    f"{n} bytes" if n >= 0 else "伺服器端複製")

        self._write_config(model, spec, version=version)

    def _write_config(self, model: str, spec: "ModelSpec", *, version: int | None = None) -> None:
        self.repo.write_config(model, render_triton_config(model, spec, version=version))

    def _pin_version(self, model: str, version: int) -> None:
        """改寫 repository 裡 config.pbtxt 釘住的版本（回退的快速路徑：檔案在，只換版本）。"""
        import re

        text = self.repo.read_config(model)
        block = "version_policy {\n" f"  specific {{ versions: [ {version} ] }}\n" "}"
        new, n = re.subn(r"version_policy\s*\{.*?\}\s*\}", block, text, flags=re.DOTALL)
        if n == 0:
            new = text.rstrip("\n") + "\n" + block + "\n"
        self.repo.write_config(model, new)

    def _load(self, model: str, version: int) -> None:
        """呼叫 load API，讓 Triton 照 repository 的 config.pbtxt 重新載入。

        ⚠️ 不能帶 config override：Triton 會拿 override【整份取代】config.pbtxt
           （剩下的靠 auto-complete 補），metadata（類別名稱）與 dynamic_batching 的
           max_queue_delay 都會不見，直到 Triton 重啟重讀檔案才恢復（2026-10-13 測試叢集實測：
           promote 上去的模型 parameters={}、queue delay 5000→0）。
           服務版本由檔案的 version_policy 決定：呼叫前 config.pbtxt 已釘好版本
           （deploy → _stage_model 寫檔；rollback 快速路徑 → _pin_version），
           也就同時解決「重啟後跳回 latest」（見 write_triton_config 第 4 點）。
        """
        r = self._client.post(f"{self.triton_url}/v2/repository/models/{model}/load", json={})
        if r.status_code != 200:
            raise RuntimeError(f"load 失敗 HTTP {r.status_code}: {r.text[:200]}")

    def _wait_ready(self, model: str, version: int, timeout: float | None = None) -> bool:
        deadline = time.time() + (timeout or self.ready_timeout)
        while time.time() < deadline:
            r = self._client.get(
                f"{self.triton_url}/v2/models/{model}/versions/{version}/ready"
            )
            if r.status_code == 200:
                return True
            time.sleep(0.3)
        return False

    def _prune(self, model: str, keep_around: int) -> None:
        """清理過舊版本，保留最近 keep_versions 版（含目前版本）。"""
        versions = sorted(self.repo.versions(model), reverse=True)
        for v in versions[self.keep_versions:]:
            if v == keep_around:
                continue
            self.repo.remove_version(model, v)
            logger.info("[%s] 清理舊版本 v%d（MinIO 仍保留）", model, v)
