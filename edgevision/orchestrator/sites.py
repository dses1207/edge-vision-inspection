"""推論站點：哪些線的推論跑在哪一台 GPU 節點。

    預設站點（不在任何站點裡的線）：Triton = triton、GPU = K8S_GPU_CLAIM（gpu-node-1 的 MIG 切片）
    其他站點：自己一套 Triton ＋ 自己的 GPU ＋ 釘在自己的節點；那幾條線的 runner 跟它同節點
    （9/24 實驗：runner 與 Triton 之間搬張量是最大瓶頸，不可跨節點）

⭐ 設定是部署拓撲，跟 git 走：ConfigMap edgevision-inference-sites（kustomize 產生）掛成檔案，
   runner 調和器每一輪重讀，control-api 每次 promote 重讀 —— 改站點不用重啟任何服務。

sites.json：
    {"siteb": {"lines": ["siteb_*"],                       # 線路 id，可用萬用字元
            "triton": "triton-siteb",                   # Service 名，runner 連 <triton>:8000 / 8001
            "gpu_claim": "inference-gpu-siteb",         # 這個站點的 GPU（ResourceClaim）
            "node_selector": {"camera-net": "siteb"},
            "triton_repo": "s3://minio:9000/models/_triton/sm89",   # engine 綁 GPU 架構，最後一段就是架構
            "max_batch": 8,                          # 選填：這張卡建 engine 的 batch 上限（卡小，模型宣告 32 也只建到 8）
            "go2rtc_source": "rtsp://192.0.2.75:8554"}}   # 選填：runner 從哪台 go2rtc 拉原始畫面（站點節點本機那台）；
                                                     #       不填 = 主 go2rtc。打框畫面一律推回主 go2rtc（前端只連那台）
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from fnmatch import fnmatchcase
from pathlib import Path

DEFAULT_PATH = "/etc/edgevision/sites/sites.json"


@dataclass(frozen=True)
class Site:
    name: str
    lines: tuple[str, ...]
    triton: str
    gpu_claim: str
    triton_repo: str
    node_selector: dict = field(default_factory=dict)
    max_batch: int | None = None
    go2rtc_source: str | None = None

    def matches(self, line_id: str) -> bool:
        return any(fnmatchcase(line_id, p) for p in self.lines)

    @property
    def arch(self) -> str:
        return self.triton_repo.rstrip("/").rsplit("/", 1)[-1]

    @property
    def triton_http(self) -> str:
        return f"{self.triton}:8000"


def parse_sites(data: dict) -> dict[str, Site]:
    out = {}
    for name, d in sorted((data or {}).items()):
        if not isinstance(d.get("lines"), list):           # 可以是空的：清空 = 這個站點暫時不跑任何線（回退）
            raise ValueError(f"站點 {name} 缺 lines（清單）")
        for k in ("triton", "gpu_claim", "triton_repo"):
            if not d.get(k):
                raise ValueError(f"站點 {name} 缺 {k}")
        if d.get("go2rtc_source") and not str(d["go2rtc_source"]).startswith("rtsp://"):
            raise ValueError(f"站點 {name} 的 go2rtc_source 要是 rtsp://…")
        if not str(d["triton_repo"]).startswith("s3://"):
            raise ValueError(f"站點 {name} 的 triton_repo 要是 s3://…（各節點的 Triton 直讀 MinIO）")
        out[name] = Site(name=name, lines=tuple(d["lines"]), triton=d["triton"], gpu_claim=d["gpu_claim"],
                         triton_repo=d["triton_repo"], node_selector=dict(d.get("node_selector") or {}),
                         max_batch=int(d["max_batch"]) if d.get("max_batch") else None,
                         go2rtc_source=str(d["go2rtc_source"]).rstrip("/") if d.get("go2rtc_source") else None)
    return out


def load_sites(path: str | None = None) -> dict[str, Site]:
    """檔案不存在 = 只有預設站點（測試叢集、單節點）。檔案壞掉 → 拋例外（呼叫端整輪放棄，不要猜）。"""
    p = Path(path or os.environ.get("INFERENCE_SITES_FILE", DEFAULT_PATH))
    if not p.is_file():
        return {}
    return parse_sites(json.loads(p.read_text(encoding="utf-8") or "{}"))


def site_for_line(sites: dict[str, Site], line_id: str) -> Site | None:
    hits = [s for s in sites.values() if s.matches(line_id)]
    if len(hits) > 1:
        raise ValueError(f"線路 {line_id} 同時符合多個站點：{[s.name for s in hits]}")
    return hits[0] if hits else None
