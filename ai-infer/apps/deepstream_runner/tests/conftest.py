import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))   # pytest importlib 模式：同目錄的 api_shapes 要手動掛路徑
from api_shapes import api_line


@pytest.fixture
def line_file(tmp_path):
    p = tmp_path / "line.json"
    p.write_text(json.dumps(api_line()))
    return p


@pytest.fixture
def write_line(tmp_path):
    def _w(**over):
        p = tmp_path / "line.json"
        p.write_text(json.dumps(api_line(**over)))
        return {"CAMERA_SOURCE": "file", "CAMERA_FILE": str(p)}
    return _w
