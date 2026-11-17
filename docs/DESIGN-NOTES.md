# Design notes: lessons learned the hard way

[繁體中文](DESIGN-NOTES.zh-TW.md)

Every item here actually happened and was measured; none of it is inference. The details are in code comments (search for `⚠️ 為什麼`, which means "why").

## Streaming and video

### Burn the boxes into the video on the server

The frontend gets video over WebRTC and results over a separate channel, then aligns them by timestamp. We measured this and it does not line up: go2rtc opens an independent session per consumer, each RTP timestamp starts from its own origin, and the offset jitters by 200 ms ([measurements](findings/2026-08-28-sync-verification.md)).
Embedding a frame marker in the video makes alignment possible, but decoding the marker on every frame in the browser is expensive and depends on MSE. We chose to have the runner burn the boxes into `<cam>_marked` with nvdsosd, so the frontend does nothing ([decision](findings/2026-09-04-overlay-decision.md)).

### When several frontends stall together, it is not necessarily bandwidth

go2rtc advertises the IP of every network interface as an ICE candidate. The browser may try an unreachable one first and only moves on after a timeout.
Advertising a single IP that browsers can reach (`WEBRTC_IP`) fixes it.

### Cameras disconnect and reconnect regularly, and the first frames after a reconnect are ghost images

When the decoder is missing reference frames it holds the old picture with blocky artifacts, and the model still judges it. On one day, every NG fell on a reconnect time.
When the worker sees that a camera has had no frames for a while, it pauses verdicts for a few seconds after recovery and clears the station's cross-frame state.

## Inference

### nvinferserver allocates batch memory for the engine's max batch

If the engine is exported with a large max batch, Triton allocates that size for every instance, and a few models can use up a 48 GB MIG slice.
Export max batch to match the real camera count. At startup the runner checks that the number of cameras on the line is ≤ the engine's max batch, and there is a GPU memory watchdog.

### Moving seg prototypes costs more than the inference

Each frame's 32×160×160 prototypes are copied from GPU to host and then sent over gRPC. With 7 streams that is 463 MB/s, and 70% of Triton's CPU time went there.
Moving mask composition into the ONNX graph (it outputs already-cropped masks) cut the transfer by 60% (`edgevision/training/seg_fuse.py`).
The cost is that max_det must stay small: each detection carries one 160×160 mask, so 300 rows is 30 MB per frame.

### A TensorRT engine is tied to the GPU architecture and TensorRT version

Always build engines with the trtexec bundled in the Triton image, not in the training image. The model repository has one directory per GPU architecture (`_triton/sm89`, `sm120`).

## Configuration and data

### Configuration is declarative, and the runner reports the version it actually applied

"I changed the config" and "production is running that config" are two different things. Each line has a desired and a reported version, and `/healthz` shows at a glance which line has not caught up.

### A model's own properties can have only one source

Class names used to be stored a second time in the line config, copied by hand. After a model version change that copy went stale, and the runner kept mapping the new model's class IDs with the old names, without raising any error.
The line config no longer stores classes; they are attached from the Registry on read. Before startup, the runner checks every class name each station references, camera by camera.

### NG is a verdict, not a box

For a rule like "B not seen within 10 seconds after A", the NG frame may contain no boxes at all. Previously only frames with boxes were persisted, so these NGs could not be queried and no video could be attached.
Publishing, persistence and recording all key off the Verdict.

### There can be only one gate

The recording layer once added its own cooldown. The result was NGs in the database with no video, and no way to tell whether that was by design or a fault.
Deduplication belongs to the verdict layer (consecutive frames and cooldown both live in the station). The recording layer takes everything it receives, the two sides stay one-to-one, and any mismatch is a bug.

### Every value is a parameter

Images per round, sampling ratio, thresholds, frame counts: the code only provides defaults, real values live in config, and the frontend only reads the config the backend returns.

## Operations

### Container logs need a size limit

A container log without rotation once grew to several hundred GB and took down a whole test machine. Every service sets `max-size` / `max-file`.

### Disk pressure makes K8s evict Pods, so the runner has to recover on its own

The node disk filled up, go2rtc and redis were evicted, and all runners disconnected at once. They could not reconnect after recovery, and there was no inference for a day and a half.
The runner now has a watchdog: if all streams drop or a dependency is down for a while, it restarts itself without anyone stepping in.

### No API gateway; Ingress routes by path

A home-grown gateway tends to grow into a second backend. Path prefixes map to services, and each service validates the JWT itself.

### Deploy only images that actually exist

The training image is not rebuilt on every commit. With its tag defaulting to the main image's tag, it once pointed to an image that did not exist, and nobody noticed until the first training job was submitted.
`apply.sh` checks that every image tag exists before applying.
