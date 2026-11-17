"""API 回應型別：確保 OpenAPI schema 完整，且沒有欄位被 response_model 吃掉。

⚠️ 這次改動最大的風險

    掛上 response_model 之後，FastAPI 會【照 model 重新序列化】——
    model 少宣告一個欄位，那個欄位就直接不會回給前端，而且不會有任何錯誤。
    前端會看到一個安靜消失的欄位。

    所以這裡逐一比對「端點實際回什麼」與「model 宣告了什麼」。
"""
from __future__ import annotations

import pytest

from edgevision.api import schemas


def _fields(model) -> set[str]:
    return set(model.model_fields)


# ── 欄位完整性 ────────────────────────────────────────────

def test_camera_out_covers_actual_response():
    """比對 frontend_api.cameras() 實際塞進 dict 的每一個 key。"""
    actual = {"id", "name", "line_id", "station", "width", "height",
              "state", "view_stream", "view_url", "station_params"}
    missing = actual - _fields(schemas.CameraOut)
    assert not missing, f"CameraOut 少了 {missing} —— 這些欄位不會回給前端"


def test_line_summary_covers_actual_response():
    """比對 config_api.list_lines() 實際塞進 dict 的每一個 key。"""
    actual = {"id", "name", "station", "desired_version", "line_params", "enabled",
              "reported_version", "in_sync", "apply_state", "cameras_running",
              "heartbeat_age_s", "cameras_configured", "stale"}
    missing = actual - _fields(schemas.LineSummary)
    assert not missing, f"LineSummary 少了 {missing}"


def test_line_status_covers_healthz_lines():
    """比對 server._line_status() 實際塞進 dict 的每一個 key。"""
    actual = {"line_id", "name", "station", "desired_version", "reported_version",
              "in_sync", "apply_state", "apply_error", "cameras_running",
              "heartbeat_age_s", "stale"}
    missing = actual - _fields(schemas.LineStatus)
    assert not missing, f"LineStatus 少了 {missing}"


def test_detection_out_matches_contract():
    """回應型別不能比契約少 —— 少了就是前端拿不到。

    keypoints 就是活生生的例子：ai-infer 有、主 repo 契約沒有，
    結果 pose 模型的關鍵點被 Pydantic 靜默丟棄（見 test_contracts_parity）。
    """
    from libs.contracts import Detection

    core = {"class_id", "class_name", "confidence", "bbox",
            "has_mask", "mask_area", "mask_contour", "keypoints", "metrics"}
    assert core <= _fields(Detection), "契約本身少了欄位"
    missing = core - _fields(schemas.DetectionOut)
    assert not missing, f"DetectionOut 少了 {missing}"


def test_stale_is_declared_everywhere_it_is_returned():
    """stale 是 2026-09-10 新增的欄位，兩個端點都會回，兩個 model 都要有。

    漏掉的話前端永遠看不到「這條線失聯了」，只會看到一組舊快照數字。
    """
    assert "stale" in _fields(schemas.LineSummary)
    assert "stale" in _fields(schemas.LineStatus)


# ── OpenAPI 產得出型別 ────────────────────────────────────

def test_openapi_has_response_schemas():
    """前端跑 openapi-typescript 時，回應不能是無結構的 object。"""
    from edgevision.api.server import app

    spec = app.openapi()
    schemas_in_spec = spec.get("components", {}).get("schemas", {})
    for name in ("CameraOut", "LineSummary", "HealthOut", "FrontendConfig"):
        assert name in schemas_in_spec, f"{name} 沒有進 OpenAPI —— 前端會產出 any"


def test_key_endpoints_declare_response_type():
    """這幾個端點是前端一定會打的，回應必須有型別。"""
    from edgevision.api.server import app

    spec = app.openapi()
    for path in ("/api/cameras", "/api/frontend-config", "/api/config/lines", "/healthz"):
        ok = spec["paths"][path]["get"]["responses"]["200"]
        ref = str(ok.get("content", {}).get("application/json", {}).get("schema", {}))
        assert "$ref" in ref or "items" in ref, f"{path} 的回應沒有型別（會產出 any）"


def test_ws_message_types_exist():
    """WebSocket 不在 OpenAPI 規範內，但前端一樣要型別 —— 至少 model 要在。"""
    assert _fields(schemas.WsResult) >= {"type", "camera_id", "frame_id", "detections"}
    assert _fields(schemas.WsFlipRefs) >= {"type", "ref_quads"}


# ── WebSocket schema 產生器 ────────────────────────────────

def test_ws_schema_builds():
    """gen_ws_schema.py 產得出合法的 discriminated union。"""
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from gen_ws_schema import build

    s = build()
    assert s["title"] == "WsMessage"
    titles = {v["title"] for v in s["oneOf"]}
    assert titles == {"WsFlipRefs", "WsResult"}


def test_ws_schema_has_no_dangling_refs():
    """⚠️ $ref 必須全部內聯 —— json-schema-to-zod 不跟 $ref，
    留著的話 detections 會產出 z.array(z.any())，等於沒驗證到。
    """
    import json
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from gen_ws_schema import build

    assert '"$ref"' not in json.dumps(build()), "還有 $ref 沒展開"


def test_ws_result_detections_fully_expanded():
    """detections 必須展開到 bbox 這一層，不能只是 array of object。"""
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from gen_ws_schema import build

    result = next(v for v in build()["oneOf"] if v["title"] == "WsResult")
    det = result["properties"]["detections"]["items"]["properties"]
    assert "class_name" in det and "confidence" in det
    assert "x1" in det["bbox"]["properties"], "bbox 沒展開"


def test_all_ws_message_types_registered():
    """新增 /ws 訊息種類時，WS_MESSAGES 要一起登記，否則前端產不出它的型別。"""
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from gen_ws_schema import WS_MESSAGES

    from edgevision.api import schemas

    declared = {n for n in dir(schemas) if n.startswith("Ws")}
    registered = {m.__name__ for m in WS_MESSAGES}
    assert declared == registered, f"schemas.py 有 {declared}，但只登記了 {registered}"


def test_ws_schema_seals_known_objects():
    """⚠️ 有明確欄位的物件必須 additionalProperties: false。

    不封閉的話，產出的 interface 會帶 `[k: string]: unknown` 索引簽章 ——
    任何 key 都合法，前端把 camera_id 拼成 camrea_id 不會報錯，
    型別檢查形同虛設（實測：加了才會出現 "Did you mean to write 'camera_id'?"）。
    """
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from gen_ws_schema import build

    result = next(v for v in build()["oneOf"] if v["title"] == "WsResult")
    assert result.get("additionalProperties") is False
    det = result["properties"]["detections"]["items"]
    assert det.get("additionalProperties") is False
    assert det["properties"]["bbox"].get("additionalProperties") is False


def test_free_form_dicts_stay_open():
    """metrics / details 是各站別自訂的自由欄位，不能封閉 —— 封了就塞不進東西。"""
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from gen_ws_schema import build

    result = next(v for v in build()["oneOf"] if v["title"] == "WsResult")
    metrics = result["properties"]["detections"]["items"]["properties"]["metrics"]
    assert metrics.get("additionalProperties") is not False


def test_control_api_every_2xx_has_schema():
    """前端回饋：控制 API 的 2xx 回應都要有 schema，前端才能 codegen，
    不用手寫型別（手寫踩過 marker_id 當 string、相機 state 當自由字串兩次）。"""
    from edgevision.api.control import app as control_app
    spec = control_app.openapi()
    empty = []
    for path, methods in spec["paths"].items():
        for method, op in methods.items():
            for code, resp in op.get("responses", {}).items():
                if not code.startswith("2"):
                    continue
                content = resp.get("content", {})
                schema = next((c["schema"] for c in content.values() if c.get("schema")), None)   # 圖片 = binary
                if not schema:
                    empty.append(f"{method.upper()} {path} {code}")
    assert not empty, "空 schema：" + ", ".join(empty)
