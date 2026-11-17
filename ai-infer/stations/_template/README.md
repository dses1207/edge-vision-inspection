# 站別範本

複製這個目錄開新站，五個地方要改：

1. `pyproject.toml`：`name`、`description`、entry point 那一行（左邊站名、右邊 `模組:類別`）。
2. `src/edgevision_station_template/` 改名成 `edgevision_station_<站名>/`，`station.yaml` 填模型、類別、必要現場值。
3. `station.py`：只要 `__init__(camera_params)`、`evaluate()`、`reset()`、`update_params()` 四個方法形狀對，中間隨便寫。
4. `tests/`：行為測試 + `fixtures/*.json`（現場錄下的偵測結果與預期判定）。
5. 在 workspace 根 `pyproject.toml` 的 `[tool.uv.sources]` 加一行，`apps/deepstream_runner/pyproject.toml` 的 dependencies 加一行，然後 `uv sync`。

這個範本本身是可跑的（類別計數站），有測試和 fixture，先跑一次確認環境正常再改。
