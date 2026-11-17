"""跟 output-api GET /api/config/lines/{id} 一模一樣形狀的假資料。"""

CAM156_REF = [[0.30, 0.35], [0.55, 0.40], [0.55, 0.05], [0.30, 0.05]]


def api_line(**over):
    """跟 output-api GET /api/config/lines/{id} 一模一樣的形狀。"""
    line = {
        "id": "alpha_a", "name": "ALPHA A 線", "station": "zone_intrusion", "config_version": 3, "enabled": True,
        "line_params": {"source_fps": 30, "target_fps": 5},
        # 模型類別由設定 API 從 Registry 帶入（站別不再宣告類別）
        "model": {"name": "seg_alpha", "classes": ["board", "hand"], "task": "segment"},
        "cameras": [
            {"id": "cam156", "ip": "192.0.2.102", "stream_path": "/stream2", "width": 1280, "height": 720,
             "cred_ref": "default", "name": "ALPHA2", "enabled": True, "station_params": {"zone": CAM156_REF, "thresholds": {"target_classes": ["hand"]}}},
            {"id": "cam193", "ip": "192.0.2.155", "stream_path": "/stream2", "width": 1280, "height": 720,
             "cred_ref": "default", "name": "ALPHA3", "enabled": True, "station_params": {"zone": CAM156_REF, "thresholds": {"target_classes": ["hand"]}}},
        ],
    }
    line.update(over)
    return line


