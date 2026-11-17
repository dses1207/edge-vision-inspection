#!/usr/bin/env python3
"""手動 / 除錯用：印出 runner 調和器「現在會建出的」runner Deployment（不套用）。

⭐ 平常不用跑這支：叢集上的 runner-reconciler 每 30 秒自動讓 runner 跟線路表一致
   （edgevision/orchestrator/runner_reconciler.py）。這支跟它用同一份產生邏輯，給人看差異或救急用：

    kubectl -n edgevision port-forward svc/output-api 18080:18080 &
    python3 deploy/k8s/render_runners.py --api http://localhost:18080 \\
        --image ghcr.io/your-org/edgevision/edgevision-runner:<tag> [--lines alpha_a] | kubectl diff -f -

編碼器預設交給 runner 自己判斷（ENCODER=auto）；--x264 強制整條線走 CPU。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from urllib.request import urlopen

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from edgevision.orchestrator.runner_reconciler import desired_runners, runner_deployment  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--api", required=True, help="設定 API（output-api），例 http://localhost:18080")
    ap.add_argument("--image", required=True)
    ap.add_argument("--lines", default="", help="只印這些線（逗號分隔）；空 = 全部啟用中的線")
    ap.add_argument("--x264", default="", help="強制這些線用 x264（逗號分隔）")
    ap.add_argument("--gpu-claim", default="inference-gpu")
    ap.add_argument("--namespace", default="edgevision")
    a = ap.parse_args()
    only = {x.strip() for x in a.lines.split(",") if x.strip()}
    x264 = {x.strip() for x in a.x264.split(",") if x.strip()}
    api = a.api.rstrip("/")
    with urlopen(f"{api}/api/config/lines", timeout=10) as r:
        metas = json.load(r)
    details = {}
    for m in metas:
        if m.get("enabled", True):
            with urlopen(f"{api}/api/config/lines/{m['id']}", timeout=10) as r:
                details[m["id"]] = json.load(r)
    items = []
    for name, dep in desired_runners(metas, details, image=a.image, gpu_claim=a.gpu_claim,
                                     namespace=a.namespace).items():
        lid = dep["metadata"]["labels"]["edgevision.line"]
        if only and lid not in only:
            continue
        if lid in x264:
            line = details[lid]
            dep = runner_deployment(lid, line["model"]["name"], image=a.image, gpu_claim=a.gpu_claim,
                                    namespace=a.namespace, config_version=int(line.get("config_version") or 0),
                                    encoder="x264")
        items.append(dep)
        print(f"# {lid} → {name}", file=sys.stderr)
    json.dump({"apiVersion": "v1", "kind": "List", "items": items}, sys.stdout, ensure_ascii=False)


if __name__ == "__main__":
    main()
