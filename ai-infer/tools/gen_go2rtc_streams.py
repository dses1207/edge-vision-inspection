#!/usr/bin/env python3
"""從設定服務產 go2rtc 的 streams 區塊：go2rtc 的相機清單完全是 cameras 表的投影。

    python tools/gen_go2rtc_streams.py --api http://localhost:18080 -o deploy/go2rtc/streams.generated.yaml

每台輸出三條：
  <id>        子碼流，推論用，passthrough
  <id>_main   主碼流
  <id>_marked ⭐ 來源留空 —— runner 把「已燒好框」的影像 RTSP PUBLISH 進來，前端看這條。

⚠️ _marked 必須宣告且【來源留空】。官方文件：「資料可以推送到已存在的串流，
   包括那些來源為空的串流」——
     · 完全不定義 → go2rtc 拒收，推流端收到 broken pipe
     · 定義成從自己拉流 → 形成迴圈，同樣失敗
   兩種錯法都實測過。
帳密由 cred_ref 解析（規則同 EdgeVision edgevision/config/camera_source.resolve_credentials）：
  cred_ref = "none" → 網址不帶帳密（不需要帳密的相機）
  CAM_<REF>_USER / CAM_<REF>_PASS → 找不到退回 CAM_USER / CAM_PASS，並在 stderr 警告；都沒有就報錯不產生。
產出的檔案含明碼帳密，只能留在機台上，不進 git。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from urllib.request import urlopen

NO_AUTH_REF = "none"
# ⚠️ backchannel=0：go2rtc 預設會帶 ONVIF 雙向語音的 Require 標頭。有些相機收到後回的 SDP
#    把 H.265 的視訊寫成 H264 + PCMU 音訊，下游拿到的就是宣告錯誤、解不出來的串流
#    （2026-09-29 正式機實測；不帶這個標頭時 SDP 是正確的 H265/90000）。我們從不回送聲音，一律關掉。
SRC_OPTS = "#backchannel=0#video=copy#audio="
_warned: set[str] = set()


def creds(cred_ref: str) -> tuple[str, str]:
    """回 (帳號, 密碼)；("", "") = 不帶帳密。純函式（只讀環境變數），可測。"""
    raw = (cred_ref or "").strip()
    if raw.lower() == NO_AUTH_REF:
        return "", ""
    ref = raw.upper()
    user = os.environ.get(f"CAM_{ref}_USER") if ref else None
    pw = os.environ.get(f"CAM_{ref}_PASS") if ref else None
    if user:
        return user, pw or ""
    user, pw = os.environ.get("CAM_USER"), os.environ.get("CAM_PASS")
    if not user:
        raise SystemExit(f"找不到相機帳密：需要 CAM_{ref}_USER / CAM_{ref}_PASS 或 CAM_USER / CAM_PASS"
                         f"（相機不用帳密請把 cred_ref 設成 {NO_AUTH_REF!r}）")
    if ref and ref not in _warned:
        _warned.add(ref)
        print(f"⚠️ cred_ref={raw!r} 沒有 CAM_{ref}_USER，改用 CAM_USER；相機不用帳密請設 cred_ref={NO_AUTH_REF!r}",
              file=sys.stderr)
    return user, pw or ""


def rtsp_url(user: str, pw: str, ip: str, path: str) -> str:
    auth = f"{user}:{pw}@" if user else ""
    return f"rtsp://{auth}{ip}:554{path}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", required=True)
    ap.add_argument("-o", "--out")
    ap.add_argument("--main-stream", default="/stream1", help="主碼流路徑（觀看用），空字串 = 不產生")

    ap.add_argument("--marked-suffix", default="_marked",

                    help="打框推流的名稱後綴；空字串=不產生")
    ap.add_argument("--exclude", default="", help="略過這些相機 id（逗號分隔），例如範本裡已手寫的 cam156")
    ap.add_argument("--relay", default="",
                    help="從另一台 go2rtc 轉拉（例 rtsp://192.0.2.170:8554），不直接連相機、不需帳密。"
                         "測試叢集連不到 IPCAM 網段、ALPHA 相機有連線數上限時用；不產生 _main")
    a = ap.parse_args()
    exclude = {x.strip() for x in a.exclude.split(",") if x.strip()}

    with urlopen(f"{a.api.rstrip('/')}/api/config/lines", timeout=10) as r:
        lines = json.load(r)
    out = []
    n = 0
    for meta in lines:
        if not meta.get("enabled", True):
            continue
        with urlopen(f"{a.api.rstrip('/')}/api/config/lines/{meta['id']}", timeout=10) as r:
            line = json.load(r)
        out.append(f"  # ── 線路 {line['id']}（{line.get('name', '')}，站別 {line['station']}）──")
        for c in line["cameras"]:
            if not c.get("enabled", True) or c["id"] in exclude:
                continue
            if a.relay:
                # 轉拉：對相機來說只多了「來源 go2rtc」那一個既有連線，不會多吃相機的連線數
                out.append(f'  {c["id"]}:\n    - "{a.relay.rstrip("/")}/{c["id"]}"')
                if a.marked_suffix:
                    out.append(f'  {c["id"]}{a.marked_suffix}:')
                n += 1
                continue
            user, pw = creds(c.get("cred_ref", ""))
            sub = c.get("stream_path") or "/stream2"
            out.append(f'  {c["id"]}:\n    - "{rtsp_url(user, pw, c["ip"], sub)}{SRC_OPTS}"')
            if a.main_stream:
                out.append(f'  {c["id"]}_main:\n    - "{rtsp_url(user, pw, c["ip"], a.main_stream)}{SRC_OPTS}"')
            if a.marked_suffix:
                # 來源留空：等 runner PUBLISH 進來（見檔頭說明）
                out.append(f'  {c["id"]}{a.marked_suffix}:')
            n += 1
    text = "\n".join(out) + "\n"
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"已產生 {n} 台相機的串流到 {a.out}", file=sys.stderr)
    else:
        sys.stdout.write(text)


if __name__ == "__main__":
    main()
