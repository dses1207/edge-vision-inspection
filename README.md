# EdgeVision

[繁體中文](README.zh-TW.md)

EdgeVision is the backend of a camera-based visual inspection system. It runs GPU inference on dozens of
RTSP cameras, applies per-line judgement rules, records an evidence clip for every alarm, and sends
reviewed alarms back into labeling and training. New models only reach production after passing
evaluation gates, and a bad release can be rolled back from the model registry.

This repository contains the platform only. The production-line rules and the trained models it was built
for are not included. Three generic demo stations (zone intrusion, follow-up window, PPE check) and a
synthetic camera take their place, and the whole pipeline runs on a laptop without a GPU.

```
 camera -> go2rtc -> runner (DeepStream + Triton) -> station plugin -> Redis Streams
                       |  boxes burned into <cam>_marked                  |
                       v                                                  +-> event-processor -> Postgres
                   go2rtc -> WebRTC to browsers                           +-> recorder -> MinIO (NG clips)
                                                                          +-> notifier (Apprise)
                                                                          +-> output-api (WebSocket / REST)

 review queue -> alert feedback -> media library / labeling -> lakeFS dataset
      ^                                                             |
      +-- Triton hot reload <- registry (promote) <- gates <- training job (Orchestrator, Docker / K8s)
```

The flow above is explained step by step in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Running the demo

Needs Docker and about five minutes. CPU only.

```bash
cd demo
cp .env.example .env
docker compose up -d --build
```

| URL | What it shows |
|---|---|
| http://localhost:1984/stream.html?src=cam-01_marked | Live stream with boxes burned in and the per-station verdict |
| http://localhost:1984/stream.html?src=cam-02_marked | Loading dock: after a forklift arrives, a spotter must appear within 10 s |
| http://localhost:8080/docs | output-api: line config, detections, stats, review |
| http://localhost:8080/healthz | Desired vs. reported config version, heartbeat and fps for every line |
| http://localhost:9001 | MinIO console; evidence clips are under `inference-output/` |

The demo uses the same worker, station plugins, publisher, heartbeat, recorder, event processor and API as
production. Only the frame source and the model are replaced. A synthetic scene renders the frames, and a
"scene detector" converts its ground truth into raw YOLO output tensors, so post-processing, NMS and
letterbox restoration still run for real. See [`demo/`](demo/) and
[`ai-infer/apps/cpu_runner`](ai-infer/apps/cpu_runner/src/cpu_runner/main.py).

## Repository layout

| Path | Contents |
|---|---|
| [`ai-infer/`](ai-infer/) | Inference monorepo (uv workspace). `libs/` holds shared contracts, post-processing, rule primitives and the station SDK; `stations/` has one package per judgement rule; `apps/deepstream_runner` is the production runner and `apps/cpu_runner` the demo runner |
| [`edgevision/api`](edgevision/api) | output-api (versioned and validated line config, detections, live WebSocket) and control-api (training, datasets, models, tasks) |
| [`edgevision/orchestrator`](edgevision/orchestrator) | Job orchestrator that runs step containers on Docker or K8s, the runner reconciler (lines in the database become runner Deployments), and go2rtc stream sync |
| [`edgevision/training`](edgevision/training) | Ingest, train, evaluate gates, export ONNX, check TensorRT parity, register. Each step is an idempotent CLI |
| [`edgevision/registry`](edgevision/registry), [`controller`](edgevision/controller) | Model registry (the production tag is what runs) and the Triton deployer |
| [`edgevision/media`](edgevision/media) | Media library, labeling, review rounds, iterative labeling campaigns, auto-labeling, alert feedback, video tests |
| [`edgevision/recorder`](edgevision/recorder), `run_nas_recorder.py` | Event clips cut from a pre-record ring buffer, and 24 h continuous recording to a NAS |
| [`edgevision/notify`](edgevision/notify) | Daily report and health patrol, sent through Apprise |
| [`edgevision/mcp`](edgevision/mcp) | MCP server (stdio and remote OAuth). Every API endpoint is also an agent tool; a parity test keeps the two in sync |
| [`edgevision/ptz`](edgevision/ptz) | PTZ control over ONVIF, home positions, drift detection |
| [`deploy/docker-compose.yml`](deploy/docker-compose.yml), [`test_train.sh`](test_train.sh) | Full stack on a single GPU host, including training (Triton, ClearML, lakeFS, trainer), plus an end-to-end training test: data to lakeFS, train, gates, TensorRT, registry, promote, rollback |
| [`deploy/k8s`](deploy/k8s) | Kustomize base and an example overlay: DRA GPU claims, multi-site inference, Triton loading its model repository from S3 |
| [`deploy/postgres/init`](deploy/postgres/init) | Schema and idempotent migrations |
| [`docs/`](docs/) | Architecture, design notes, and dated engineering findings |

## Design decisions

Boxes are burned into the video on the server instead of being drawn in the browser. I tried to keep a
WebRTC stream and a separate result channel frame-accurate using RTP timestamps, measured it, and it did not
work ([finding](docs/findings/2026-08-28-sync-verification.md), [decision](docs/findings/2026-09-04-overlay-decision.md)).

One production line maps to one runner and one config version. Lines and cameras live in Postgres, and the
reconciler turns them into Deployments. Each runner reports the config version it actually applied in its
heartbeat, so `/healthz` shows desired and reported versions side by side ([`server.py`](edgevision/api/server.py)).

Stations are plugins that know nothing about the runner. A station is a small class with `evaluate`,
`reset`, `update_params` and `classes_used`, registered through a Python entry point; the runner only knows
its name. Reusable mechanisms such as persistence counters, cooldowns, wait windows and zones live in
`edgevision_rules`. The thresholds live in the station.

The model owns its class names. Line configs never store class lists; they are attached from the registry
when the config is read, and every class a station refers to is checked against the deployed model before a
runner starts.

NG is a verdict, not a box. Time-based rules such as "B did not happen within 10 s of A" can fire on a frame
with no boxes at all, so publishing, storage and evidence recording are all keyed on the verdict.

De-duplication happens in one place, the judgement layer. The recorder records every NG it receives, so an
NG in the database without a clip is always a bug and never an intentional cooldown.

More detail in [docs/DESIGN-NOTES.md](docs/DESIGN-NOTES.md) and [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Development

```bash
# inference workspace (uv)
cd ai-infer && uv sync --all-packages && uv run pytest -q && uv run ruff check .

# platform services
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest -q tests --ignore=tests/integration
```

To write a new station, copy [`ai-infer/stations/_template`](ai-infer/stations/_template) or start from one
of the demo stations: [`zone_intrusion`](ai-infer/stations/zone_intrusion),
[`followup_window`](ai-infer/stations/followup_window), [`ppe_check`](ai-infer/stations/ppe_check).

Most code comments are in Traditional Chinese. They usually explain why a decision was made, often with the
measurement or incident behind it. The engineering findings under `docs/findings/` are also in Chinese.

## Third-party software

The training and auto-labeling steps call [Ultralytics](https://github.com/ultralytics/ultralytics)
(AGPL-3.0) at runtime inside the trainer image. It is not vendored in this repository and is not needed by
the services or the demo.
