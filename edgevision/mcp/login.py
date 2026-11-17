"""edgevision-login：瀏覽器登入授權伺服器，把憑證存進本機快取。

    python -m edgevision.mcp.login              登入（開瀏覽器）
    python -m edgevision.mcp.login --status     看目前登入狀態
    python -m edgevision.mcp.login --logout     刪掉本機憑證

一次登入可用 30 天（refresh token 效期），期間 MCP 會自動續期 access token，
使用者不用管。30 天後再跑一次即可。

流程是 OAuth 2.1 的 authorization_code + PKCE（AuthServer 強制 PKCE）：
    ① 產生 code_verifier / challenge，開瀏覽器到 AuthServer
    ② 使用者在授權伺服器登入
    ③ AuthServer 回呼到 localhost 的臨時伺服器，帶 authorization code
    ④ 用 code + verifier 換 access_token + refresh_token，存檔

⚠️ 必須帶 offline_access scope 才會拿到 refresh token，否則一小時後就要重登。
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import http.server
import json
import os
import secrets
import socket
import sys
import threading
import time
import urllib.parse
import webbrowser

from edgevision.mcp.credentials import (
    Credentials, NotLoggedIn, _subject_of, auth_server, creds_path, discovery, status, verify_tls,
)

CLIENT_ID = os.environ.get("AUTH_CLIENT_ID", "edgevision-cli")
SCOPES = "openid profile offline_access edgevision_api"


def _pkce() -> tuple[str, str]:
    verifier = base64.urlsafe_b64encode(secrets.token_bytes(64)).rstrip(b"=").decode()
    challenge = base64.urlsafe_b64encode(
        hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()
    return verifier, challenge


# ⚠️ OAuth 的 redirect URI 必須【事先註冊】在 AuthServer，不能用隨機 port。
#    所以固定一組候選，註冊 client 時把這幾個都加進 redirectUris。
CALLBACK_PORTS = [int(p) for p in os.environ.get(
    "AUTH_CALLBACK_PORTS", "8765,8766,8767,8768,8769").split(",")]


def _free_port() -> int:
    """從已註冊的候選 port 裡找一個沒被占用的。"""
    for port in CALLBACK_PORTS:
        with socket.socket() as s:
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    raise SystemExit(
        f"❌ 候選 port 都被占用了（{CALLBACK_PORTS}）。"
        f"請關掉占用的程式，或用 AUTH_CALLBACK_PORTS 指定其他 port（需同步在 AuthServer 註冊）")


class _Catcher(http.server.BaseHTTPRequestHandler):
    """接住 AuthServer 的回呼，取出 authorization code。"""

    result: dict = {}

    def do_GET(self):  # noqa: N802
        q = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
        _Catcher.result = {k: v[0] for k, v in q.items()}
        ok = "code" in _Catcher.result
        body = (
            "<h2>✅ 登入成功</h2><p>可以關掉這個分頁，回到終端機。</p>" if ok else
            f"<h2>❌ 登入失敗</h2><pre>{_Catcher.result.get('error_description', _Catcher.result)}</pre>"
        )
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(f"<html><body style='font-family:sans-serif;padding:40px'>{body}</body></html>".encode())

    def log_message(self, *a):  # 不要把 HTTP log 印到終端機
        pass


def login(timeout: int = 300) -> Credentials:
    import httpx

    base = auth_server()
    try:
        disc = discovery(base)
    except Exception as exc:
        raise SystemExit(f"❌ 連不上 AuthServer（{base}）：{exc}")

    port = _free_port()
    redirect_uri = f"http://localhost:{port}/callback"
    verifier, challenge = _pkce()
    state = secrets.token_urlsafe(16)

    params = {
        "client_id": CLIENT_ID, "response_type": "code", "redirect_uri": redirect_uri,
        "scope": SCOPES, "state": state,
        "code_challenge": challenge, "code_challenge_method": "S256",
    }
    url = f"{disc['authorization_endpoint']}?{urllib.parse.urlencode(params)}"

    srv = http.server.HTTPServer(("127.0.0.1", port), _Catcher)
    threading.Thread(target=srv.serve_forever, daemon=True).start()

    print(f"→ 開啟瀏覽器登入")
    print(f"  若沒有自動開啟，請手動貼上：\n  {url}\n")
    try:
        webbrowser.open(url)
    except Exception:
        pass

    deadline = time.time() + timeout
    while not _Catcher.result and time.time() < deadline:
        time.sleep(0.3)
    srv.shutdown()

    if not _Catcher.result:
        raise SystemExit(f"❌ 等待登入逾時（{timeout} 秒）")
    if "code" not in _Catcher.result:
        raise SystemExit(f"❌ 登入失敗：{_Catcher.result.get('error_description', _Catcher.result)}")
    if _Catcher.result.get("state") != state:
        raise SystemExit("❌ state 不符，可能遭到攔截，已中止")

    with httpx.Client(verify=verify_tls(), timeout=30.0) as c:
        r = c.post(disc["token_endpoint"], data={
            "grant_type": "authorization_code", "code": _Catcher.result["code"],
            "redirect_uri": redirect_uri, "client_id": CLIENT_ID, "code_verifier": verifier,
        })
    if r.status_code != 200:
        raise SystemExit(f"❌ 換發 token 失敗（{r.status_code}）：{r.text[:300]}")

    d = r.json()
    sub, name = _subject_of(d["access_token"])
    creds = Credentials(
        access_token=d["access_token"], refresh_token=d.get("refresh_token", ""),
        expires_at=time.time() + int(d.get("expires_in", 3600)),
        subject=sub, display_name=name, server=base,
    )
    path = creds.save()

    print(f"✅ 已登入：{name or sub}（{sub}）")
    print(f"   憑證存於 {path}（權限 600）")
    if creds.refresh_token:
        print(f"   一次登入可用 30 天，期間 MCP 會自動續期，你不用管 token")
    else:
        print(f"   ⚠️ 沒有拿到 refresh token —— 一小時後要重新登入。"
              f"請確認 client 有授權 offline_access scope")
    return creds


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="edgevision-login", description="登入授權伺服器，供 MCP 與 CLI 使用")
    ap.add_argument("--status", action="store_true", help="顯示目前登入狀態")
    ap.add_argument("--logout", action="store_true", help="刪除本機憑證")
    ap.add_argument("--timeout", type=int, default=300, help="等待瀏覽器登入的秒數")
    a = ap.parse_args(argv)

    if a.logout:
        p = creds_path()
        if p.exists():
            p.unlink()
            print(f"✅ 已刪除 {p}")
        else:
            print(f"（沒有憑證可刪：{p}）")
        return 0

    if a.status:
        st = status()
        if not st["logged_in"]:
            print(f"未登入（{st['path']}）\n請執行：python -m edgevision.mcp.login")
            return 1
        mins = st["access_token_valid_for_s"] // 60
        print(f"已登入：{st['display_name'] or st['subject']}（{st['subject']}）")
        print(f"  AuthServer : {st['server']}")
        print(f"  access token: {'需換發' if st['needs_refresh'] else f'還有 {mins} 分鐘'}")
        print(f"  refresh token: {'有（30 天內自動續期）' if st['has_refresh_token'] else '無 ⚠️'}")
        return 0

    try:
        login(timeout=a.timeout)
        return 0
    except NotLoggedIn as exc:
        print(f"❌ {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
