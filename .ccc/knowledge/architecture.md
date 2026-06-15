# Krabby Architecture — Canonical Reference

> Single source of truth for how the Krabby system is designed.
> Agent-specific lenses (engineer, principal, …) point here rather
> than restating it (T-023 DRY). Grounded against code + `docs/` as of
> 2026-06-03 (branch `jdp/m11-real2sim`). When code and this doc
> disagree, **code wins** — fix the doc.
>
> Confidence convention: data-flow and module boundaries below are
> verified against real files. Exact numeric constants (observation
> dims, reward weights, gains) are reported as the explorers found
> them but tagged _(verify in code)_ — treat as approximate until you
> read the cited source.

---

## 1. What Krabby is

A locomotion stack for the **Krabby hexapod** ("krabby-uno") — a
six-legged robot, 18 actuated DoF (6 legs × 3 joints). Compute is a
**Jetson Orin**; actuation is **3× Arduino Mega 2560**, each driving 6
motors through **BTS7960 H-bridges**. Sensing is a **ZED 2i** RGB-D +
IMU (primary), with an optional side **MaixSense A075V** depth module.

The repo spans the full path **from bare OS to running locomotion**,
plus two research pipelines that feed it.

### Three pillars

1. **Robot runtime** — the live control stack on the robot
   (firmware → HAL → controller/policy). §2.
2. **parkour** — IsaacLab/IsaacSim RL training that produces the
   locomotion policy checkpoints the runtime consumes. §3.
3. **real2sim** — offline reconstruction of real spaces into
   simulation-ready scenes (the active M11 milestone). §4.

The connective tissue is the **HAL** (Hardware Abstraction Layer): one
ZMQ contract that both the real robot and the simulator implement, so
the same inference client drives either. §2.2.

---

## 2. Robot runtime stack

### 2.1 End-to-end data flow

```
            ┌─ gamepad (Pro Controller, SDL2 ~50 Hz) ─┐
command ───>│   OR                                    │──> HalClient ──┐
source      └─ policy (ParkourInferenceClient ~100 Hz)┘   (PUSH/req)   │
                                                                       v
                                                          HalServer (PULL)
                                                                       │
                                              JetsonHalServer.apply_*  │
                                                                       v
                                          KrabbyMCUSDK (pyserial @115200)
                                                                       │
                                          3× Arduino Mega (primary/L/R)
                                                                       v
                                          BTS7960 H-bridges → 18 motors
observation <── HalClient(SUB,CONFLATE) <── HalServer(PUB) <── grab(): joints+IMU+RGB-D
```

- **Command path:** ZMQ PUSH/PULL, endpoint `inproc://hal_commands` or
  `tcp://*:6002`. Payload `JointCommand` (18 target joint positions).
- **Observation path:** ZMQ PUB/SUB, topic `b"observation"`, endpoint
  `inproc://hal_observation` or `tcp://*:6001`. Payload
  `HardwareObservations` (joint pos/vel, base quat/ang-vel/lin-vel,
  contacts, optional camera RGB/depth, optional depth-scan features).
  Latest-only semantics (server `SNDHWM=1`, client `CONFLATE=1`).
- In production both HAL server and inference client run in **one
  process** over `inproc://` (lowest latency); the gamepad path uses
  **TCP 6001/6002** so the client can be a separate process/host.

### 2.2 HAL — the abstraction seam

`hal/` ships as wheel packages with clean boundaries:

| Package | Path | Role |
|---|---|---|
| `krabby-hal-client` | `hal/client/` | SUB observations, PUSH commands. No hardware coupling. |
| `krabby-hal-server` (base) | `hal/server/` | Abstract `HalServerBase` — PUB/PULL transport, leaves `get_observation` / `apply_joint_command` abstract. |
| `krabby-hal-server-jetson` | `hal/server/jetson/` | `JetsonHalServer` — real sensors (ZED, IMU) + `KrabbyMCUSDK`. |
| (isaac variant) | `hal/server/isaac/` | Sim server — same contract inside IsaacLab. |
| `krabby-hal-tools` | `hal/tools/` | `hal_dump`, multi-stream display, GStreamer debug. |

The **same `HardwareObservations`/`JointCommand` contract** is honored
by the Jetson server (real robot) and the Isaac server (sim). This is
why a policy trained in sim can run on hardware unmodified — the seam
is the HAL message schema. Authoritative: `docs/HAL_GUIDE.md`.

### 2.3 Firmware

- **Three boards, elected roles:** `primary` (FRONT, the one on host
  USB), `left`, `right` followers. Primary forwards follower telemetry
  over Serial2/Serial3; role persists in EEPROM.
- **V protocol:** send `V\n` → firmware replies with per-board
  version/branch/sha (`-` for a missing board). Used by
  `krabby firmware show`.
- **Telemetry line:** `<ROLE>; <JOINT> <9 values>; …` — position,
  pot raw, current raw, enable L/R, PWM L/R, safety flag. Parsed by
  `firmware.interfaces.joint_telemetry.JointTelemetry`.
- **Firmware store:** S3 bucket `krabby-firmware-public/`, per-branch
  `latest.json` / `builds.json`, builds keyed `YYYYMMDD-HHMMSS-<sha7>`.
  Flashed via `krabby firmware update`; downloads cached under
  `~/.cache/krabby-firmware/`.
- **Pinout:** `firmware/MOTOR_HEADER_PINOUT.md` (`KRABBY_PIN_REV=3`
  default = Uno v0.2). Latest flashed firmware ≈ v0.2.9 (M14).

### 2.4 Controller + CLI

- **Gamepad mode:** `InputController` (pygame/SDL2) →
  `GamepadToKrabbyHALMapper` → `JointCommand` → HAL.
  Wired in `controller/control_loop.py`.
- **Inference mode:** `ParkourInferenceClient`
  (`compute/parkour/inference_client.py`) extends `HalClient`; loop is
  poll → `HWObservationsToParkourMapper` → policy forward (CUDA) →
  `ParkourLocomotionToHWMapper` → `put_joint_command`. Mappers live in
  `compute/parkour/mappers/`.
- **Launcher:** `krabby/` CLI. `krabby install` pulls the locomotion
  image from ECR + sets up udev/dialout/systemd; `krabby run` launches
  the container with GPU + `/dev` passthrough. Mode selection in
  `krabby/run.py`: presence of `--checkpoint` / `--entrypoint` ⇒
  inference, else gamepad. Entrypoint `hal.server.jetson.main`.

---

## 3. parkour — RL training

**Goal:** train locomotion/parkour policies in IsaacLab → export a
TorchScript `.pt` checkpoint → robot runs it via
`krabby run -- --checkpoint <path>`.

- **Stack split:** `parkour_isaaclab/` = base env + MDP machinery
  (envs, managers, terrains, actuators); `parkour_tasks/` = task
  registrations + per-robot configs. Trainer is `rsl-rl-lib`. Targets
  both the `crab_hex` hexapod and Unitree Go2.
- **Gym IDs:** `Isaac-Crab-Hex-Teacher-v0`, `…-Student-v0` (+
  Play/Eval), registered under
  `parkour_tasks/crab_hexapod_task/config/crab_hex/`.
- **Teacher → Student → deploy:** teacher trains on privileged obs;
  student distills to **depth-camera-only** observations (RMA-style);
  export emits `policy.pt` (+ depth encoder for student). Exporter:
  `parkour/scripts/rsl_rl/exporter.py`.
- **Staged curriculum:** flat-walk → bridge → moderate parkour →
  student distill, driven by env-var config mutations (prevents
  curriculum thrashing). See `crab_hex_env_cfg.py`.
- **Policy shape** _(verify in code — `docs/crab-hexapod-policy-config.md`)_:
  18-D action (6 legs × {hip revolute, hip-femur prismatic,
  femur-tibia prismatic}); observation ≈ proprioception + IMU +
  commands + history + terrain scan; privileged latents (mass,
  friction, gains) teacher-only.

---

## 4. real2sim — scene reconstruction (active, M11)

> **Canonical process doc:** the operator-facing, phase-by-phase M11
> scene-processing process lives at
> `real2sim/knowledge/scene-processing/` (T0 ingress → T1 scouting/spine
> → T2 view-selection → T3a/b/c reconstruction → T4 ranking;
> `RECIPES.md` points there). This §4 is the *system map*; that doc set
> is the *how-to-run-it*. Tracked under `EPI-SCN-M11-PROCESS-DOCS`.

**Goal:** handheld phone/action-cam video → simulation-ready scene
(USD/Blender) with collision-grade mesh + camera poses, no LIDAR / no
ground-truth scale. Output is intended to seed IsaacSim environments.

**Pipeline (scripts in `real2sim/`):**

```
video ──extract_frames.sh──> frames/
      ──run_colmap_sparse.sh (or MASt3R / VGGT / SLAM3R)──> sparse poses
      ──colmap_to_cameras_json.py──> cameras.json
      ──[MAtCha reconstruction]──> tetra mesh
      ──orient_mesh.py──> floor at z=0, +z up (RANSAC + gravity prior)
      ──cull_mesh.py──> coverage/below-floor cull
      ──decimate_oriented.py──> ~200k-tri collision mesh
      ──project_color.py──> vertex RGB from source frames
      ──build_blender_scene.py──> scene.blend ──> USD for IsaacSim
```

- **Interchangeable front-ends:** COLMAP (tuned, CPU mapper-bound),
  MASt3R-SLAM / VGGT / SLAM3R (neural, GPU, fast). All emit a common
  `cameras.json`, so the back half of the pipeline is front-end
  agnostic.
- **Manifest system** (`manifest_lib.py`, `backfill_manifests.py`):
  schema-v1 JSON per reconstruction variant — frames used, MAtCha
  config, execution host/GPU/duration, outputs, post-processing flags.
  Drives the rating UI and reproducibility.
- **Camera-model gotcha:** DJI Action 3 needs
  `SIMPLE_RADIAL_FISHEYE`, not `PINHOLE`. COLMAP version must match
  across hosts (3.10 mapper can't read a 3.11.1 DB).
- **real2sim ↔ parkour link:** as of now there is **no automated
  hand-off** — training terrain is procedurally authored, not yet fed
  by reconstructed geometry. real2sim output is the manual bridge. If
  you wire these together, update this line.

---

## 5. Repo map (top level)

| Dir | What |
|---|---|
| `firmware/` | Arduino Mega firmware, V protocol, S3 store, flash tooling. |
| `hal/` | Hardware Abstraction Layer wheel packages (client/server/jetson/isaac/tools). |
| `controller/` | Gamepad input + control loop + mappers. |
| `compute/` | Production inference (`compute/parkour/` = client, mappers, policy). |
| `krabby/` | `krabby` launcher CLI (install/run/firmware). |
| `parkour/` | IsaacLab RL training (teacher/student, terrains). |
| `real2sim/` | M11 reconstruction pipeline. |
| `environments/` | Reconstructed scene outputs (USD/mesh/manifest per scene). |
| `experiments/` | Reproducibility provenance (e.g. m11 eval runs + decision matrix). |
| `hardware/` | Mechanical/electrical CAD, wiring, servo specs. |
| `data_collection/` | HAL data collector (rosbag2/mcap recording off the ZMQ stream). |
| `teleop/` | Edge relay (Jetson) + portal web UI (WebRTC remote drive). |
| `bench/` | Bench watchdog systemd service (polls ECR, smoke-tests firmware, alerts). |
| `images/` | Docker images: locomotion (prod), isaacsim, testing x86/arm, + M11 recon images. |
| `compute/`,`scripts/`,`tests/` | Inference, host bootstrap scripts, test suites. |

Authoritative layout: `docs/FOLDER_LAYOUT.md`. Glossary:
`docs/TECHNOLOGY_AND_TERMINOLOGY.md`.

---

## 6. Build & deploy

- **Docker images** (`images/`): production **`krabby-locomotion`**
  (Jetson/ARM), **`krabby-isaacsim`** (x86 sim), testing x86/arm, plus
  the M11 recon images (scene-reconstruction-base, matcha, mast3r,
  slam3r, vggt). Inventory: `docs/DOCKER_DEPENDENCIES.md`.
- **ECR channels:** `release-latest` (stable) vs `mainline-latest`
  (dev). `krabby install` pulls `release-latest`.
- **CI** (`.github/`): firmware publish (→ S3), packages publish (→
  PyPI wheels), locomotion image (→ ECR, daily scheduled build).
- **Bench watchdog** (`bench/`): polls ECR digest ~60 s, pulls + runs
  `krabby firmware show` smoke test, alerts SMTP/GitHub on failure.
- **Jetson deploy:** JetPack 6.1/6.2; `--runtime=nvidia --privileged
  -v /dev:/dev`; ZED SDK + resources mounted. See
  `docs/JETSON_DEPLOYMENT.md`, `docs/PYTORCH_GPU_SUPPORT.md` (sm_87
  Orin / sm_120 RTX 50-series; cu128 wheels).

---

## 7. Milestone arc

| M | Focus | Status |
|---|---|---|
| M11 | real2sim scene reconstruction | **Active** (this branch). |
| M12 | Hardware assembly (Jetson + 3× Mega + H-bridges + ZED) | Done (per README). |
| M14 | Jetson bench bringup — firmware updater, locomotion image, `krabby` CLI, bench watchdog | Done ~2026-05-22; firmware v0.2.9. Blocker noted: no trained checkpoint yet. |

(Numbers like exact firmware version / dims are point-in-time — confirm
against `docs/m14-bringup-report.md` and code before relying on them.)

---

## 8. Networking — service reachability

**Local services are reached via DNS `krabby.organl.com`, not raw IPs
or `localhost`.** Any service krabby hosts locally (dashboards, viewers,
debug servers, `ccc-combine`, the real2sim verify viewer, etc.) is
addressed through that name — prefer it over hardcoded addresses so the
binding survives host/IP changes. This is the self-hosted counterpart to
the "bind by discovery, not by hostname" stance krabby takes toward the
*fleet's* shared services (sherpa's `consumption-map.md` covers the
fleet side).

---

## 9. First files to read

Runtime: `docs/RUNTIME_ARCHITECTURE.md`, `docs/HAL_GUIDE.md`,
`hal/server/jetson/main.py`, `firmware/SETUP.md`,
`firmware/MOTOR_HEADER_PINOUT.md`, `controller/control_loop.py`,
`krabby/run.py`, `compute/parkour/inference_client.py`.

Training: `parkour/README.md`, `docs/crab-hexapod-policy-config.md`,
`parkour/scripts/rsl_rl/{train,exporter}.py`.

real2sim: `real2sim/README.md`,
`docs/M11-SCENE-RECONSTRUCTION-NOTES.md`,
`real2sim/{colmap_to_cameras_json,orient_mesh,cull_mesh,manifest_lib}.py`.

Cross-cutting: `docs/FOLDER_LAYOUT.md`,
`docs/TECHNOLOGY_AND_TERMINOLOGY.md`, `docs/DOCKER_DEPENDENCIES.md`,
top-level `Makefile`.
