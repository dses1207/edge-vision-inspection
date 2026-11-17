# ai-infer：推論與判定的 monorepo（uv workspace）

[English](README.md)

一個 workspace 只放程式碼：共用庫、各站判定、判讀室。模型歸 Triton 管，相機與現場值歸資料庫管。
全部裝進同一個 runner 映像，一條線一個容器，啟動參數決定跑哪站、接哪些相機。

```
ai-infer/
├── libs/
│   ├── contracts/     edgevision_contracts    Detection / Verdict / FrameResult，runner 與平台共用的唯一一份契約
│   ├── postprocess/   edgevision_postprocess  YOLO detect / seg / pose 張量 → Detection；letterbox 還原；畫框
│   ├── rules/         edgevision_rules        幾何（重疊、距離、多邊形）+ 時間性（連續幀、冷卻、事件歷史、等待視窗）
│   └── station_sdk/   edgevision_station_sdk  Station 介面、station.yaml、entry point 載入、fixture 回放、參數規格
├── stations/          一站一套件，站與站禁止互相 import
│   ├── _template/        範本（類別計數），複製它開新站
│   ├── zone_intrusion/   示範：禁區闖入（腳點在多邊形內、去抖、冷卻）
│   ├── followup_window/  示範：A 出現後 N 秒內要看到 B（時間性 NG，NG 當下可能沒有框）
│   └── ppe_check/        示範：附加站，第二顆模型 + 滑動視窗確認 + 依原因冷卻
├── apps/
│   ├── deepstream_runner/  正式判讀室：GStreamer / DeepStream 管線 + Triton（GPU）
│   └── cpu_runner/         demo 判讀室：合成畫面 + 場景偵測器，worker 之後與正式版同一份程式
└── tools/ci/affected.sh    算「改了什麼要測什麼」
```

## 三層怎麼分

| 層 | 放什麼 | 判斷標準 |
|---|---|---|
| `libs/` | 換一站還會一字不改再用的東西 | 函式簽章裡沒有任何站別名詞 |
| `stations/` | 只對這一站有意義的數字和概念 | 門檻有現場實測依據；概念離開這站沒人用 |
| `apps/` | 把站別拿來用的程式 | 沒有任何一行寫站名；站名來自設定，靠 `load_station(name)` 拿 |

灰色地帶的規則：機制進 libs，數字留 stations。連續幀去抖的機制是 `edgevision_rules.PersistCounter`，「3 幀」這個數字在站別參數。

## 一個站別長什麼樣

```python
class MyStation:
    def __init__(self, camera_params: dict): ...          # 只吃現場值
    def evaluate(self, detections, width, height) -> Verdict: ...   # 純計算，跨幀狀態用 edgevision_rules
    def classes_used(self) -> set[str]: ...                # 會用到的類別名，runner 啟動時對模型比對
    def reset(self) -> None: ...                           # 串流中斷恢復時清狀態
    def update_params(self, camera_params: dict) -> None: ...      # 熱套用
```

`station.yaml` 宣告預設模型、必要現場值；`pyproject.toml` 用 entry point 登記：

```toml
[project.entry-points."edgevision.stations"]
zone_intrusion = "edgevision_station_zone_intrusion:ZoneIntrusionStation"
```

測試：行為測試 + `tests/fixtures/*.json`（偵測結果序列與每幀預期判定，`assert_fixture` 回放）。

## 開發

```bash
uv sync --all-packages
uv run pytest -q              # 全部（不需要 GPU；標了 gpu 的測試預設不跑）
uv run ruff check .
tools/ci/affected.sh origin/main   # 只測受影響的套件
```

新增一站後，在根目錄 `pyproject.toml` 的 `[tool.uv.sources]` 與 runner 的 dependencies 各加一行，然後 `uv lock`。
