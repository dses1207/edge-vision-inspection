# ai-infer: inference and verdict monorepo (uv workspace)

[繁體中文](README.zh-TW.md)

The workspace holds only code: shared libraries, per-station verdict logic, and the runners. Models belong to Triton; cameras and site values belong to the database.
Everything is installed into one runner image, one container per line. Startup parameters decide which stations run and which cameras they connect to.

```
ai-infer/
├── libs/
│   ├── contracts/     edgevision_contracts    Detection / Verdict / FrameResult, the single contract shared by the runner and the platform
│   ├── postprocess/   edgevision_postprocess  YOLO detect / seg / pose tensors → Detection; letterbox reversal; box drawing
│   ├── rules/         edgevision_rules        geometry (overlap, distance, polygons) + temporal (consecutive frames, cooldown, event history, wait windows)
│   └── station_sdk/   edgevision_station_sdk  Station interface, station.yaml, entry point loading, fixture replay, parameter spec
├── stations/          one package per station; stations must not import each other
│   ├── _template/        template (class counting); copy it to start a new station
│   ├── zone_intrusion/   example: restricted-zone intrusion (foot point inside polygon, debounce, cooldown)
│   ├── followup_window/  example: B must appear within N seconds after A (temporal NG; there may be no box at the NG moment)
│   └── ppe_check/        example: attached station, second model + sliding-window confirmation + per-reason cooldown
├── apps/
│   ├── deepstream_runner/  production runner: GStreamer / DeepStream pipeline + Triton (GPU)
│   └── cpu_runner/         demo runner: synthetic video + scene detector; from the worker on, same code as production
└── tools/ci/affected.sh    works out what to test for a given change
```

## How the three layers split

| Layer | What goes here | Rule of thumb |
|---|---|---|
| `libs/` | Code another station would reuse without changing a character | No station-specific nouns in any function signature |
| `stations/` | Numbers and concepts that only make sense for this station | Thresholds are backed by on-site measurements; nobody outside this station uses the concept |
| `apps/` | Code that uses the stations | No line names a station; station names come from config via `load_station(name)` |

Gray areas: the mechanism goes in libs, the numbers stay in stations. Consecutive-frame debouncing is `edgevision_rules.PersistCounter`; the value "3 frames" is a station parameter.

## What a station looks like

```python
class MyStation:
    def __init__(self, camera_params: dict): ...          # site values only
    def evaluate(self, detections, width, height) -> Verdict: ...   # pure computation; cross-frame state via edgevision_rules
    def classes_used(self) -> set[str]: ...                # class names used; runner checks them against the model at startup
    def reset(self) -> None: ...                           # clear state when a stream recovers from an interruption
    def update_params(self, camera_params: dict) -> None: ...      # hot apply
```

`station.yaml` declares the default model and required site values. `pyproject.toml` registers the station through an entry point:

```toml
[project.entry-points."edgevision.stations"]
zone_intrusion = "edgevision_station_zone_intrusion:ZoneIntrusionStation"
```

Tests: behavior tests plus `tests/fixtures/*.json` (a sequence of detection results with the expected verdict for each frame, replayed with `assert_fixture`).

## Development

```bash
uv sync --all-packages
uv run pytest -q              # everything (no GPU needed; tests marked gpu are skipped by default)
uv run ruff check .
tools/ci/affected.sh origin/main   # test only the affected packages
```

After adding a station, add one line to `[tool.uv.sources]` in the root `pyproject.toml` and one to the runner's dependencies, then run `uv lock`.
