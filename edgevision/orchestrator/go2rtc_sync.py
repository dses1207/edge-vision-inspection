"""go2rtc 相機清單同步：讓 go2rtc 的串流跟資料庫的相機表一致（runner 調和器每一輪順便做）。

    資料庫（設定 API 的線路 + 相機）          Secret go2rtc-streams（go2rtc 啟動時讀，見 deploy/go2rtc/entrypoint.sh）
            │                                          │
            └──────── 每一輪產生一次、跟 Secret 比 ──────┘
                 一樣              → 不動
                 不一樣            → 寫 Secret，改 go2rtc Pod 範本上的 streams-hash → K8s 換一個 go2rtc Pod

⭐ 為什麼不用 go2rtc 的 API（PUT /api/streams）即時加：<id>_marked 必須存在而且「來源留空」，
   runner 才能把打好框的畫面推進去（推到不存在的串流會被拒）；API 加串流一定要給 src，空的做不到。
   只能寫進設定檔、讓 go2rtc 重新啟動。換 Pod 時所有畫面斷幾秒再自動連回，所以只在清單真的變了才換。

⭐ 以前：相機清單是人手動跑 ai-infer/tools/gen_go2rtc_streams.py → 建 Secret → 重啟 go2rtc。
   前端從 API 加的相機沒人做這三步，go2rtc 沒有那幾條串流 → 前端沒直播、runner 拉不到畫面一直重連
   。

每台相機兩條（跟手動產生的一致，不產生 _main：前端看 _marked，主碼流多佔相機連線）：
    <id>          原始畫面，runner 推論用
    <id>_marked   來源留空，runner 推打框畫面進來，前端看這條

來源怎麼決定：
    線路在有 go2rtc_source 的推論站點（例 B 在 gpu-node-2）→ 主 go2rtc 從站點那台轉拉；
        站點那台 go2rtc（Deployment go2rtc-<站點>、Secret go2rtc-<站點>-streams）直連相機
    GO2RTC_RELAY（測試叢集：連不到相機網段）→ 全部從正式機 go2rtc 轉拉
    其他 → 直連相機（帳密由 cred_ref 從環境變數解析，規則同 edgevision/config/camera_source.resolve_credentials）

停用的線路 / 相機也照樣註冊：go2rtc 沒人看就不連相機，不花資源；這樣啟用 / 停用不會讓 go2rtc 重啟。

安全規則：設定 API 讀失敗、一台相機都沒有、帳密解析失敗 → 這一輪不動（絕不因為讀不到就把清單清空）。
"""
from __future__ import annotations

import base64
import hashlib
import logging
import re
from dataclasses import dataclass

from edgevision.config.camera_source import resolve_credentials
from edgevision.orchestrator.sites import Site, site_for_line

logger = logging.getLogger(__name__)

ANN_STREAMS = "edgevision.example.com/streams-hash"
SECRET_KEY = "streams.generated.yaml"
MARKED = "_marked"
# backchannel=0：go2rtc 預設帶 ONVIF 雙向語音標頭，IPCAM 相機收到會回錯的 SDP（見 gen_go2rtc_streams.py）
SRC_OPTS = "#backchannel=0#video=copy#audio="
_AUTH = re.compile(r"(rtsp://)[^@/\"]*@")


@dataclass(frozen=True)
class Target:
    """一台 go2rtc：Deployment 名、Secret 名、要放哪些串流（名字 → 來源；None = 留空）。"""
    deployment: str
    secret: str
    streams: dict[str, str | None]


def direct_url(cam: dict) -> str:
    user, pw = resolve_credentials(cam.get("cred_ref", ""))
    auth = f"{user}:{pw}@" if user else ""
    return f"rtsp://{auth}{cam['ip']}:554{cam.get('stream_path') or '/stream2'}{SRC_OPTS}"


def targets(details: list[dict], sites: dict[str, Site], *, relay: str | None = None) -> list[Target]:
    """線路詳情（含相機）→ 主 go2rtc ＋ 每個有 go2rtc_source 的站點各一台。純函式（帳密讀環境變數）。"""
    main: dict[str, str | None] = {}
    per_site: dict[str, dict[str, str | None]] = {}
    for line in sorted(details, key=lambda d: d["id"]):
        site = site_for_line(sites, line["id"])
        for cam in line.get("cameras") or []:
            cid = cam["id"]
            if not cam.get("ip") or cid in main:
                continue
            if site is not None and site.go2rtc_source:
                main[cid] = f"{site.go2rtc_source}/{cid}"
                per_site.setdefault(site.name, {})[cid] = direct_url(cam)
            elif relay:
                main[cid] = f"{relay.rstrip('/')}/{cid}"
            else:
                main[cid] = direct_url(cam)
            main[cid + MARKED] = None
    out = [Target("go2rtc", "go2rtc-streams", main)]
    out += [Target(f"go2rtc-{name}", f"go2rtc-{name}-streams", s) for name, s in sorted(per_site.items())]
    return out


def render(streams: dict[str, str | None]) -> str:
    """接在範本 streams: 後面的那一段（entrypoint.sh 直接 cat 上去）。"""
    lines = ["  # 由 runner 調和器從相機表產生（edgevision/orchestrator/go2rtc_sync.py），手動改會被蓋掉"]
    for name in sorted(streams):
        src = streams[name]
        lines.append(f"  {name}:" if src is None else f'  {name}:\n    - "{src}"')
    return "\n".join(lines) + "\n"


def redact(text: str) -> str:
    return _AUTH.sub(r"\1***@", text)


def content_hash(text: str) -> str:
    """去掉帳密再算（hash 會寫在 Pod 範本上，看得到的地方不放跟密碼有關的東西）。"""
    return hashlib.sha256(redact(text).encode()).hexdigest()[:16]


class Go2rtcSync:
    def __init__(self, *, core_api, apps_api, namespace: str, relay: str | None = None) -> None:
        self.core, self.apps, self.namespace, self.relay = core_api, apps_api, namespace, relay

    def once(self, details: list[dict], sites: dict[str, Site]) -> list[str]:
        """回這一輪換掉的 go2rtc Deployment 名。"""
        try:
            tgs = targets(details, sites, relay=self.relay)
        except (Exception, SystemExit) as exc:   # resolve_credentials 找不到帳密是 SystemExit
            logger.warning("go2rtc 清單這一輪略過（產生失敗）：%s", exc)
            return []
        if not tgs[0].streams:
            logger.warning("go2rtc 清單這一輪略過：一台相機都沒有（不清空）")
            return []
        changed = []
        for t in tgs:
            try:
                if self._sync(t):
                    changed.append(t.deployment)
            except Exception:
                logger.exception("[%s] 同步相機清單失敗，下一輪再試", t.deployment)
        return changed

    def _sync(self, t: Target) -> bool:
        try:
            dep = self.apps.read_namespaced_deployment(t.deployment, self.namespace)
        except Exception as exc:
            if getattr(exc, "status", None) == 404:
                logger.warning("[%s] 叢集沒有這個 Deployment，略過", t.deployment)
                return False
            raise
        text = render(t.streams)
        want = content_hash(text)
        sec = self.core.read_namespaced_secret(t.secret, self.namespace)
        cur = base64.b64decode((sec.data or {}).get(SECRET_KEY, "")).decode()
        if cur != text:
            self.core.patch_namespaced_secret(t.secret, self.namespace, {
                "metadata": {"annotations": {ANN_STREAMS: want}},
                "data": {SECRET_KEY: base64.b64encode(text.encode()).decode()}})
            old = set(_names(cur))
            logger.warning("[%s] 相機清單更新：+%s -%s", t.deployment,
                           sorted(set(t.streams) - old), sorted(old - set(t.streams)))
        anns = (dep.spec.template.metadata.annotations or {}) if dep.spec.template.metadata else {}
        if anns.get(ANN_STREAMS) == want:
            return False
        # Pod 範本變 → K8s 換 Pod（Recreate：hostNetwork 綁埠，舊的先停）；新 Pod 讀新的 Secret
        self.apps.patch_namespaced_deployment(t.deployment, self.namespace, {
            "spec": {"template": {"metadata": {"annotations": {ANN_STREAMS: want}}}}})
        logger.warning("[%s] 換 Pod 套用新的相機清單（%s）", t.deployment, want)
        return True


def _names(text: str) -> list[str]:
    return re.findall(r"^  ([A-Za-z0-9_\-]+):", text, re.M)
