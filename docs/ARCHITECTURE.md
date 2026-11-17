# Architecture

[繁體中文](ARCHITECTURE.zh-TW.md)

## 1. Overall flow

The system does three things, and they form a loop:

1. Real-time verdicts: camera frames come in, go through GPU inference, the station rules decide OK / NG, and the result goes onto the event bus.
2. Evidence and human review: NG results are written to the database, a clip around the event is cut, and alerts and notifications go out. A person marks each alert in the review queue as real or a false alarm.
3. Getting more accurate: reviewed alerts flow back as labeled data, then into training. A model is registered only if it passes the Gate; then the production tag moves and Triton hot-reloads it.

```
  ┌──────────── Real-time verdicts ──┐   ┌──────── Evidence and review ────┐
  camera → go2rtc → runner → Redis Streams → event-processor → Postgres
                    │                  → recorder → MinIO (video)
                    └→ <cam>_marked    → output-api (WebSocket / query / review)
                                        → notifier
  ┌──────────── Getting more accurate ───────────────────────────────┐
  review results → alert feedback → image library / labeling → lakeFS → training → Gate → Registry → Triton
```

Configuration (which line, which cameras, which station, which model, which thresholds) is declarative. It is written to the database, a reconciler turns it into running runners, and each runner reports which version it has applied.

## 2. Details, step by step

### 1. Video in: go2rtc

- Each camera has two streams in go2rtc: `<cam>` (pulls the raw camera feed) and `<cam>_marked` (an empty source that waits for the runner to push back frames with boxes drawn in).
- The runner reconciler generates the stream list from the camera table (`edgevision/orchestrator/go2rtc_sync.py`). Manual edits get overwritten; the database is the source of truth.
- Browsers watch `_marked`. Boxes are burned into the frames on the server, so the frontend does no overlay work at all (reasoning in `docs/findings/2026-09-04-overlay-decision.md`).

### 2. Inference and verdicts: runner (`ai-infer/`)

One runner per line (one Deployment on K8s). Its only startup parameter is `LINE_ID`:

```
config.py   fetch line config from the config API → check the station exists, site values are complete, referenced class names exist in the model
pipeline.py RTSP → nvstreammux → nvinferserver (gRPC to Triton) → nvtracker (optional) → nvdsosd draws boxes → push back to go2rtc
probe.py    only moves tensors and decodes bboxes synchronously for nvdsosd to draw; verdicts go to the worker queue
worker.py   post-processing (edgevision_postprocess) → station evaluate() → FrameResult
publish.py  FrameResult → three Redis Streams: events (NG only), live (real time), store (persistence)
status.py   heartbeat every 15 s: applied config version, per-camera fps, free GPU memory
watchdog.py restarts itself when all streams drop or Triton goes down, no human needed
```

- Stations are plugins. Each station is a separate package registered under the entry point `edgevision.stations`; the runner only knows names (`load_station(name)`). The mechanisms (consecutive frames, cooldown, wait windows, zones) live in `edgevision_rules`; the numbers stay in the station.
- Several stations per line: besides the main station, a line can attach extra stations, each with its own model and inference interval. For example, the main model judges actions and a second model checks protective equipment.
- Stream recovery guard: if a camera produces no frames for more than a few seconds, verdicts pause for a short time after it recovers (the first frames after a reconnect are ghost images).

The demo replaces GStreamer and Triton with `apps/cpu_runner`. `worker / publish / status` are the same code.

### 3. Persistence and queries: event-processor, output-api (`edgevision/storage`, `edgevision/api`)

- event-processor consumes `edgevision:store` through a consumer group and writes `detections`. It only stores frames that have detections or an NG. A temporal NG (no box at that moment) is written as a single whole-frame verdict row.
- output-api:
  - Config API (`/api/config/lines`): on write, it validates station parameters against the parameter spec (generated from the runner image) and checks class names against the Registry. Every write creates a version snapshot that can be restored. On read, it attaches the model's classes, imgsz and task from the Registry.
  - Detection records, statistics and review feedback; `/ws` pushes real-time results; `/healthz` lists desired version vs reported version for every line.

### 4. Evidence video: recorder (`edgevision/recorder`, `run_recorder.py`)

- One ffmpeg per camera continuously records `<cam>_marked` into ring segments (a pre-record buffer). When an NG event arrives, it cuts N seconds before plus M seconds after, uploads the clip to MinIO, and writes the video location back to the matching detection record.
- No cooldown: one video per NG, one-to-one with the database. Deduplication belongs to the verdict layer.
- Retention: MinIO lifecycle rules delete the files, and `run_retention.py` marks the video field in the database as expired (the record is kept).
- 24-hour continuous recording is handled separately by `run_nas_recorder.py`, which writes to a NAS and does not interfere with event recording.

### 5. Notifications: notifier (`edgevision/notify`)

Daily reports and health checks. Delivery goes through Apprise; a channel is just a URL (Teams, email, and so on), so adding a channel needs no code change.

### 6. Review and feedback: media (`edgevision/media`)

- Alert queue: a person marks alerts as real or false alarm.
- Alert feedback: reviewed alerts → original frame from the recording plus the boxes at that moment → a new batch in a labeling project.
- Image library, labeling, review rounds and labeling projects grow the dataset round by round. Auto-labeling (YOLO-World, SAM) pre-labels images, and people only fix the mistakes.
- Video testing: pick a recording, a model version and a set of station logic, and get back a video with boxes drawn plus a list of NGs. Used to try changes before they go live.

### 7. From training to production: control-api, Orchestrator, Registry (`edgevision/training`, `edgevision/orchestrator`, `edgevision/registry`, `edgevision/controller`)

```
POST /api/training/jobs
  → tasks table → Orchestrator starts step containers in order (Docker or K8s Job)
     ① trainer        data (lakeFS version) → train → evaluate Gate → export ONNX + parity → register candidate
     ② trtexec        ONNX → TensorRT plan (uses the trtexec bundled in the Triton image to avoid version mismatches)
     ③ parity-triton  temporary Triton that loads only this version
     ④ finalize       TensorRT output vs ONNX parity → upload plan → done
  → promote: Registry moves the production tag to this version → Deployer has Triton hot-reload it
```

- Every step is idempotent and its state goes into `tasks`, so after a restart the job resumes from the first unfinished step.
- Gates live in `configs/gates/<model>.yaml`: absolute thresholds, recall on critical classes, a latency cap, and no new misses on the golden test set.
- The Registry is the single source of truth for which version is live. Neither code nor config files contain version numbers.

### 8. Letting agents operate it: MCP (`edgevision/mcp`)

Every API endpoint has a matching MCP tool (`tests/test_mcp.py` compares the two and fails if one is missing). The stdio version is for local use; the remote version uses OAuth.

### 9. Deployment: K8s (`deploy/k8s`)

- kustomize: `base/` contains all services; `overlays/<environment>` swaps domains, images and GPU selectors.
- GPUs are allocated with DRA (ResourceClaim): inference and training each have their own claim, and MIG partitioning is possible.
- Multi-site inference: runners for some lines are scheduled on other nodes and connect to that node's Triton (`inference-sites.yaml`). Triton reads the model repository directly from MinIO.
- Runner reconciler (`run_runner_reconciler.py`): every 30 seconds it reconciles the line configs in the database with Deployments, and replaces the Pod when the config version or model version changes.
