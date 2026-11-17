# demo: the full pipeline without a GPU

[繁體中文](README.zh-TW.md)

```bash
cp .env.example .env          # if a port clashes, change the last few lines of .env
docker compose up -d --build
docker compose logs -f runner-demo-zone    # watch NGs come out one by one
```

## What is inside

| Service | What it is | Production counterpart |
|---|---|---|
| `runner-demo-zone` / `runner-demo-dock` | cpu-runner, one per line | deepstream_runner (GPU), created by the runner reconciler |
| `go2rtc` | receives `<cam>` and `<cam>_marked` pushed by the runner | same (the raw stream is pulled from the camera instead) |
| `output-api` | line config API, detection queries, WebSocket, `/healthz` | same |
| `event-processor` | `edgevision:store` → Postgres | same |
| `recorder` | pre-record buffer + NG clips → MinIO, video location written back to the detection record | same |
| `postgres` / `redis` / `minio` | database, event bus, object storage | same |

Not started: Triton, control-api (training / Registry / datasets), ClearML, lakeFS, notifier, PTZ.
`initdb/99-demo-seed.sql` pre-fills model class information into `model_info_cache` (the cache the config API falls back to when it cannot reach the Registry), so the output-api log keeps reporting that the Registry lookup failed. This is expected.

## Two demo lines

### `demo_zone` (cam-01, warehouse aisle)

- Main station `zone_intrusion`: a person's foot point inside the red restricted zone for 3 consecutive frames → NG, then a 15-second cooldown.
- Attached station `ppe_check`: a second model (`ppe`). No helmet in ≥5 of 10 evaluations → NG, then a 30-second cooldown for the same reason.
  `helmet / vest / no_vest` are set as `hidden_classes`: they are still inferred, just not drawn.

### `demo_dock` (cam-02, unloading area)

- `followup_window`: after a forklift enters, a spotter must be seen within 10 seconds. In the script the spotter is missing one time in three → timeout NG.
  When this kind of NG fires there are no boxes on screen, yet the video and the record still match.

## Change the config and watch it take effect

```bash
# Move the restricted zone to the left half: config API saves → version +1 → runner sees the version change and restarts to apply it
curl -s localhost:8080/api/config/lines/demo_zone | python3 -m json.tool | head -40
```

In `/healthz`, `desired_version` and `reported_version` differ at first and line up again once the runner has restarted and applied the change.

## How the synthetic scene works

`ai-infer/apps/cpu_runner/src/cpu_runner/scene.py` uses the time t to decide what is in the frame, producing the frame and the ground-truth boxes together.
`detector.py` converts the ground-truth boxes into the raw output of a YOLO detect model (letterbox coordinates + class scores) and adds some jitter, missed detections and confidence noise, so the debounce, sliding window and cooldown downstream have something to filter. The script is deterministic: events happen at the same times on every start.
