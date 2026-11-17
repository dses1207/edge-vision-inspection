#!/usr/bin/env python3
"""從 Pydantic model 產出 WebSocket 訊息的 JSON Schema，供前端 codegen。

⚠️ 為什麼需要這支腳本

    REST 端點的型別前端可以直接 `npx openapi-typescript <host>/openapi.json` 拿到。
    但 WebSocket 不在 OpenAPI 規範內（那是 HTTP 的規格），`/ws` 產不出來。

    業界大多數專案（Home Assistant、Frigate、Grafana live、Supabase Realtime）
    就是在這裡放棄，改成前端手寫 interface —— 然後後端一改欄位，
    前端靜默拿到錯的東西，在下游某處才爆掉。

    我們不必這樣：訊息型別本來就有 Pydantic model（edgevision/api/schemas.py），
    JSON Schema 是免費的，兩種 codegen 都吃它。

⚠️ 為什麼要把 $ref 內聯

    json-schema-to-zod 不會跟著 $ref 走，遇到就產出 z.any() ——
    detections 會變成 z.array(z.any())，等於沒驗證到。
    所以這裡先把 $defs 就地展開再輸出。
    （openapi-typescript 沒這個問題，但用同一份檔案比較單純。）

用法
    python scripts/gen_ws_schema.py -o ws.schema.json

    # 編譯期型別（自動完成、拼錯會紅字）
    npx json-schema-to-typescript ws.schema.json -o ws.d.ts

    # 執行期驗證（收到不對的當場炸，不要靜默 as）
    npx json-schema-to-zod -i ws.schema.json -o ws.zod.ts

兩者都是 generated，都不用手寫。改了 edgevision/api/schemas.py 就重跑一次。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from edgevision.api.schemas import WsFlipRefs, WsResult  # noqa: E402

# /ws 會送的所有訊息種類。加新訊息時在這裡登記。
WS_MESSAGES = (WsFlipRefs, WsResult)


def inline_refs(node: Any, defs: dict[str, Any]) -> Any:
    """把 $ref 就地展開。

    json-schema-to-zod 不跟 $ref，不展開的話巢狀型別全變 z.any()。
    重複展開同一個 def 沒關係 —— JSON Schema 允許，產出的 Zod 也正確。
    """
    if isinstance(node, dict):
        ref = node.get("$ref")
        if ref:
            name = ref.rsplit("/", 1)[-1]
            if name not in defs:
                raise KeyError(f"$ref 指向不存在的定義：{ref}")
            # 深拷貝再展開，避免共用同一個 dict 造成無限遞迴
            return inline_refs(json.loads(json.dumps(defs[name])), defs)
        return {k: inline_refs(v, defs) for k, v in node.items() if k != "$defs"}
    if isinstance(node, list):
        return [inline_refs(v, defs) for v in node]
    return node


def seal(node: Any) -> Any:
    """對「有明確欄位的物件」加上 additionalProperties: false。

    ⚠️ 不加的話產出的 interface 每個都會帶 `[k: string]: unknown` 索引簽章，
       意思是「任何 key 都合法」—— 前端拼錯欄位名不會報錯，型別檢查形同虛設。

    只封閉有 properties 的物件。像 metrics / details / letterbox 那種
    宣告成 dict[str, Any] 的自由欄位沒有 properties，保持開放才是對的。
    """
    if isinstance(node, dict):
        out = {k: seal(v) for k, v in node.items()}
        if out.get("type") == "object" and out.get("properties"):
            out.setdefault("additionalProperties", False)
        return out
    if isinstance(node, list):
        return [seal(v) for v in node]
    return node


def build() -> dict[str, Any]:
    """組成一份 discriminated union —— 前端 switch (msg.type) 就能收窄型別。"""
    variants = []
    for model in WS_MESSAGES:
        s = model.model_json_schema()
        defs = s.pop("$defs", {})
        v = seal(inline_refs(s, defs))
        v["title"] = model.__name__          # 產出的 TS interface 名稱
        variants.append(v)
    return {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "title": "WsMessage",
        "description": (
            "output-api /ws 推播的訊息。用 type 欄位判別種類 —— "
            "不要靠「有沒有某個 key」。由 scripts/gen_ws_schema.py 從 "
            "edgevision/api/schemas.py 產生，不要手改。"
        ),
        "oneOf": variants,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--out", help="輸出路徑；不給就印到 stdout")
    args = ap.parse_args()

    schema = build()
    text = json.dumps(schema, ensure_ascii=False, indent=2)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
        print(f"✓ {args.out}（{len(WS_MESSAGES)} 種訊息："
              f"{', '.join(m.__name__ for m in WS_MESSAGES)}）", file=sys.stderr)
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
