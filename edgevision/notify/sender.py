"""投遞層：Card → Apprise → 任何通道。

⭐ 為什麼用 Apprise

    通道是【設定】不是【程式】。現在只有 Teams，之後會有寄信 API ——
    加通道只改 .env 的 URL 清單，程式碼一行不動：

        NOTIFY_DAILY=workflows://...;mailto://user:pass@smtp.example.com?to=ops@example.com

    Apprise 支援 100+ 種服務（Teams、Email、LINE、Slack、SMS…），URL 格式見
    https://github.com/caronc/apprise/wiki

⭐ 三個 tag 就是三個頻道

    daily / health / ng。每條 URL 掛在一個 tag 上，notify(tag=...) 只送那一組。
    同一條 URL 可以掛多個 tag（例如一個頻道收全部）。

⚠️ Teams 要用 workflows://，不是 msteams://

    msteams:// 是舊的 Incoming Webhook connector，2026-05 全面停用（MC1181996）。
    workflows:// 對應「工作流程 → 透過 Webhook 要求接收時張貼到頻道」。

⚠️ 沒設任何 URL 時不是錯誤，是【乾跑】

    測試機不一定有 Teams 頻道。清單空 → 內容印進 log、回 dry_run=True、
    不算失敗。整條鏈在沒有通道的環境照跑，只差最後一步。

"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# 狀態 → 標題前的圖示。Apprise 各通道對顏色支援不一，用 emoji 最穩。
_ICON = {"ok": "🟢", "recovered": "🟢", "degraded": "🟠", "warning": "🟠",
         "down": "🔴", "ng": "🔴", "info": "ℹ️"}


@dataclass
class Card:
    """一則通知。跟通道無關的中性表示；daily / health 只組這個。"""

    title: str
    level: str = "info"                          # ok/recovered/degraded/warning/down/ng/info
    facts: list[tuple[str, str]] = field(default_factory=list)   # 兩欄表格
    body: list[str] = field(default_factory=list)                # 段落
    links: list[tuple[str, str]] = field(default_factory=list)   # (文字, URL)

    @property
    def headline(self) -> str:
        return f"{_ICON.get(self.level, '')} {self.title}".strip()

    def to_markdown(self) -> str:
        """Apprise 的 Teams / Email plugin 都吃 Markdown；不支援的通道會自動降成純文字。"""
        lines: list[str] = []
        lines += self.body
        if self.facts:
            if lines:
                lines.append("")
            lines += [f"**{k}**：{v}" for k, v in self.facts]
        if self.links:
            lines.append("")
            lines += [f"[{t}]({u})" for t, u in self.links]
        return "\n\n".join(lines) if lines else ""


@dataclass
class SendResult:
    sent: bool
    error: str | None = None
    dry_run: bool = False
    targets: int = 0          # 這個 tag 底下有幾條通道


class Sender:
    """三個 tag，各自一組 Apprise URL。"""

    TAGS = ("daily", "health", "ng")
    ENV = {"daily": "NOTIFY_DAILY", "health": "NOTIFY_HEALTH", "ng": "NOTIFY_NG"}

    def __init__(self, *, urls: dict[str, list[str]] | None = None) -> None:
        import apprise
        self._ap = apprise.Apprise()
        self.bad: list[str] = []
        if urls is None:
            urls = {t: _split(os.environ.get(env, "")) for t, env in self.ENV.items()}
        for tag, lst in urls.items():
            for u in lst:
                # ⚠️ 加失敗（URL 格式錯、不支援的 scheme）要講出來，
                #    否則 .env 打錯一個字，通知就靜默消失。
                if not self._ap.add(u, tag=tag):
                    self.bad.append(f"{tag}: {_redact(u)}")
                    logger.error("[%s] 無法解析通知 URL：%s", tag, _redact(u))

    def targets(self, tag: str) -> int:
        return len(list(self._ap.find(tag=tag)))

    def configured(self, tag: str) -> bool:
        return self.targets(tag) > 0

    def describe(self, tag: str) -> str:
        """給啟動橫幅：這個 tag 送到哪些服務。"""
        names = [s.service_name for s in self._ap.find(tag=tag)]
        return "、".join(names) if names else "⚠️ 未設 → 乾跑（只印 log）"

    def send(self, tag: str, card: Card) -> SendResult:
        n = self.targets(tag)
        if n == 0:
            logger.info("[%s] 乾跑（未設通道）：%s\n%s", tag, card.headline, card.to_markdown()[:500])
            return SendResult(sent=False, dry_run=True, targets=0)
        try:
            # Apprise 自己處理各通道的重試與逾時；回 False = 至少一個通道失敗
            ok = self._ap.notify(title=card.headline, body=card.to_markdown(),
                                 body_format="markdown", tag=tag)
        except Exception as exc:  # noqa: BLE001 —— 投遞失敗不能拋到迴圈外
            return SendResult(sent=False, error=f"{type(exc).__name__}: {exc}", targets=n)
        if not ok:
            logger.warning("[%s] 至少一個通道投遞失敗（共 %d 條）", tag, n)
            return SendResult(sent=False, error="apprise notify() 回 False", targets=n)
        return SendResult(sent=True, targets=n)


def _split(s: str) -> list[str]:
    """多條 URL 用分號隔開（逗號會出現在 mailto 的 to= 參數裡）。"""
    return [u.strip() for u in s.split(";") if u.strip()]


def _redact(u: str) -> str:
    """log 裡不要印 secret：只留 scheme 與 host。"""
    scheme, _, rest = u.partition("://")
    host = rest.split("/")[0].split("@")[-1].split("?")[0]
    return f"{scheme}://…{host}…"
