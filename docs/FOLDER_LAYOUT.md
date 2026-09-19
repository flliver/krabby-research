# Project Folder Layout

This document is a map of the `krabby-research` repository (paths below are relative to the repo root). It describes the structure as it exists today and is extended as components are added.

## Overview

The repository holds two kinds of code that share one robot definition:

- **Simulation and training** (`parkour/`, `assets/`, `tests/`): the Isaac Lab environment, the crab-hexapod RL task packages, the generated robot plant, and the experiment campaigns that produced the checkpoints of record.
- **On-robot runtime** (`hal/`, `compute/`, `controller/`, `firmware/`, `images/`, `krabby/`, `fleet/`, `teleop/`, `bench/`, `data_collection/`, `scripts/`): the Hardware Abstraction Layer, production inference, bring-up scripts, MCU firmware, and the containers and fleet tooling that run them on the Jetson.

**Key distinction**:
- **Game loop** = the core control logic (poll HAL → build observation → run inference → send `JointCommand`)
  - Typical stack: **`compute.parkour.inference_client.ParkourInferenceClient`** + **`hal.client.HalClient`** against a running Jetson HAL (`python -m hal.server.jetson.main` in the locomotion image)

All containers use inproc ZMQ for communication within the same process:
- **Production container** (`images/locomotion/`): Bundles **`compute/parkour/`** and **`hal/server/jetson/`** (Jetson HAL server) for the robot (Jetson/ARM). Uses wheels: `krabby-hal-client`, `krabby-hal-server`, `krabby-hal-server-jetson`
- **IsaacSim container** (`images/isaacsim/`): Combines inference (`compute/parkour/`) and HAL server (`krabby-hal-server-isaac`) for simulation (x86). Uses wheels: `krabby-hal-client`, `krabby-hal-server`, `krabby-hal-server-isaac`
- **Testing containers** (`images/testing/x86/` and `images/testing/arm/`): Containers for running tests and development. Uses wheels: `krabby-hal-client`, `krabby-hal-server`, `krabby-hal-server-isaac`

## Directory Structure

```
krabby-research/
├── assets/                           # Robot plant (USD) and the generator that produces it
│   ├── crab.usda                     # MAIN asset: A15+B geometry (15 deg outer-mount splay, outer yaw axes 2.5 in from body ends)
│   ├── crab_simple.usda              # HISTORICAL: hand-authored 2026-08-09 baseline Cube model (sha-pinned, not loaded by any task)
│   ├── variants/                     # Splay/axis variants incl. the LEGACY golden crab_simple__splay00_axis5p5in.usda + MANIFEST.md (generated)
│   ├── scripts/generate_crab.py      # Generator: plain run = crab.usda; --legacy-golden; --all-variants
│   ├── scripts/                      # Also: extract_leg_profiles.py, Isaac Sim script-editor probes (squat.py, simple_walk.py)
│   ├── crab_hex_ref.usd / .urdf      # Reference model used by the joystick/HAL task (Isaac-CrabHex-Joystick-v0), NOT the training plant
│   ├── crab_hex.usd / .usda          # Earlier hand-built models (Blender pipeline; see Krabby-Uno-USD-pipeline.md)
│   └── Krabby-Uno.blend              # Blender source
│
├── parkour/                          # Isaac Lab training / evaluation code
│   ├── scripts/rsl_rl/               # train.py, play.py, evaluation.py, runner_factory.py, crab_on_policy_runner.py
│   ├── scripts/                      # curriculum_metrics.py, publish_weights.py, verify_usda_equivalence.py, ...
│   ├── parkour_isaaclab/             # Environment code: envs/ (mdp, parkour commands, events), terrains/, actuators/, managers/
│   ├── parkour_tasks/parkour_tasks/  # Task packages
│   │   ├── crab_hex_forward_task/    # Crab hexapod FORWARD-walk task (see below)
│   │   └── extreme_parkour_task/     # Upstream parkour tasks: config/go2, config/hex (joystick/HAL play cfg)
│   ├── assets/                       # Scenes and published weights used by the parkour scripts
│   ├── parkour_test/                 # Camera / terrain-generator tests
│   ├── logs/                         # RSL-RL runs + gait evals (git-ignored training artifacts)
│   └── outputs/                      # Hydra outputs (git-ignored)
│
├── hal/                              # Hardware Abstraction Layer
│   ├── __init__.py                   # Minimal stub (packages installed via wheels or editable mode)
│   │
│   ├── client/                       # HAL client package (package: krabby-hal-client)
│   │   ├── __init__.py               # Re-exports HalClient, HalClientConfig
│   │   ├── client.py                 # HalClient (ZMQ logic black-boxed)
│   │   ├── config.py                 # HalClientConfig
│   │   ├── observation/              # Observation types (NavigationCommand only)
│   │   ├── data_structures/          # Hardware data structures (hardware.py: HardwareObservations, JointCommand)
│   │   └── pyproject.toml
│   │
│   ├── server/                       # HAL server base package (package: krabby-hal-server)
│   │   ├── __init__.py               # Re-exports HalServerBase, HalServerConfig
│   │   ├── server.py                 # HalServerBase (ZMQ logic black-boxed)
│   │   ├── config.py                 # HalServerConfig
│   │   ├── robot_definition*.py      # Robot definitions (krabby hex, krabby quad, unitree go2)
│   │   ├── sensor_interface.py, gstreamer_runtime.py, streaming_map.py   # Sensor / streaming plumbing (docs/SENSOR_INTERFACE.md)
│   │   ├── pyproject.toml
│   │   │
│   │   ├── isaac/                    # IsaacSim HAL server (package: krabby-hal-server-isaac)
│   │   │   ├── __init__.py           # Re-exports IsaacSimHalServer
│   │   │   ├── hal_server.py         # IsaacSimHalServer (extends HalServerBase)
│   │   │   ├── main.py               # Entry point (console script: krabby-hal-server-isaac)
│   │   │   └── pyproject.toml
│   │   │
│   │   └── jetson/                   # Jetson HAL server (package: krabby-hal-server-jetson)
│   │       ├── __init__.py           # Re-exports JetsonHalServer
│   │       ├── hal_server.py         # JetsonHalServer (extends HalServerBase)
│   │       ├── main.py               # Console entry (python -m hal.server.jetson.main)
│   │       ├── zed_camera.py, maixsense_a075v.py, ...   # Camera / IMU backends
│   │       └── pyproject.toml
│   │
│   └── tools/                        # HAL debugging tools (package: krabby-hal-tools)
│       ├── hal_dump.py               # CLI tool (console script: hal-dump)
│       ├── joystick_teleoperation.py, multi_stream_display.py
│       └── pyproject.toml
│
├── compute/                          # Inference and computation logic
│   ├── parkour/                      # Parkour inference implementation (used in production)
│   │   ├── inference_client.py       # ParkourInferenceClient
│   │   ├── policy_interface.py       # Parkour policy inference interface
│   │   ├── model_definition.py, modules/, utils/
│   │   ├── parkour_types.py          # Parkour-specific types (ParkourObservation, ParkourModelIO, InferenceResponse)
│   │   └── mappers/                  # Data mappers (hardware ↔ model)
│   │       ├── hardware_to_model.py  # HardwareObservations → ParkourObservation
│   │       └── model_to_hardware.py  # InferenceResponse → JointCommand
│   └── testing/                      # Inference test runner + mock HAL server
│
├── controller/                       # On-robot scripts (gamepad / bring-up) that use the HAL client
│   └── control_loop.py, cli/, input/, mappers/, scripts/{isaac,jetson}
│
├── firmware/                         # MCU firmware (arduino/), MCU port + CLI/GUI tooling
├── krabby/                           # Robot host agent (install, enroll, run, telemetry)
├── fleet/                            # Fleet service, portal, infra, CLI (see fleet/README.md)
├── teleop/                           # WebRTC teleop: edge/ (robot side) and portal/ (docs/TELEOP.md)
├── bench/                            # Bench watchdog package (krabby_bench) + systemd units
├── data_collection/                  # HAL data collector (rosbag2 / mcap; docs/DATA_COLLECTOR.md)
├── hardware/                         # Mechanical iterations (Uno-v0.1, Uno-v0.2, old/)
│
├── tests/                            # Test suite (pytest.ini at the root; its norecursedirs skips any experiments/, logs/ or outputs/ dir, so tests never live in campaign trees)
│   ├── helpers.py                    # Test helpers (create_dummy_hw_obs, etc.)
│   ├── unit/                         # Unit tests: crab-hex rewards/metrics/plant (test_crab_hex_*.py), hal/, controller/,
│   │                                 #   firmware/, teleop/, bench/, krabby/, data_collection/, fixtures/, test_experiment_layout.py (campaign keep-set + bundled heads), test_policy_of_record.py (policy/ manifest)
│   └── integration/                  # Integration tests: HAL (Isaac/Jetson), game loop, timing, crab-hex phase configs
│
├── docs/                             # Documentation (list below)
│
├── images/                           # Dockerfiles and container configs
│   ├── locomotion/                   # Production container (Jetson: inference + HAL server, inproc ZMQ)
│   ├── isaacsim/                     # IsaacSim container (inference + HAL server, inproc ZMQ)
│   └── testing/{x86,arm}/            # Testing containers
│
└── scripts/                          # Deployment and utility scripts
    ├── run_isaac_hal_server.sh       # Run Isaac Sim HAL server in Docker
    ├── run_teleop_portal_x86_docker.sh
    ├── demo_isaacsim_teacher.sh
    ├── jetson/                       # bootstrap, install-docker, jetson-reset, setup-docker-gpu, run_jetson_hal_server_host.sh
    └── wheel-build/                  # Parkour wheel build helpers
```

## Crab hexapod forward-walk task

`parkour/parkour_tasks/parkour_tasks/crab_hex_forward_task/` is the FORWARD-walk task package (renamed from `crab_hexapod_task` on 2026-09-09; a sideways sibling, `crab_hex_sideways_task`, comes next). Gym ids are unchanged: `Isaac-Crab-Hex-Flat-Walk-v0`, `-Teacher-v0`, `-Student-v0` (plus `-Play-v0` variants); the RSL-RL experiment directories are unchanged too: `parkour/logs/rsl_rl/crab_hex_flat_walk|crab_hex_teacher|crab_hex_student`.

```
crab_hex_forward_task/
├── README.md                         # Task usage (train / play / eval commands)
├── config/crab_hex/                  # Env, scene, student and termination cfgs; crab_hex_phases.py (KRABBY_PHASE presets, PLANTS table)
│   └── agents/                       # RSL-RL PPO / student runner cfgs (rsl_rl_ppo_cfg.py, crab_hex_rl_cfg.py, parkour_mdp_cfg.py)
├── mdp/                              # Rewards, observations, actions, RSI, curriculums, exposure knobs;
│                                     #   crab_hex_dimensions.py = source of truth for the plant geometry
├── scripts/                          # eval_crab_hex_gait.py, run_gait_eval_suite.py, plot_crab_hex_gait.py, demo/verify probes
│   └── gait_eval/                    # metrics.py, report.py, schedule.py
├── sensors/                          # parkour_hex_contact_sensor.py
├── policy/                           # POLICY OF RECORD: the shipped head + the stage heads it was trained through
│   ├── manifest.yaml                 #   (1a_formation -> 2a_elements -> 2b_clearance -> 2c_teacher -> 3a_student, one .pt each);
│   ├── <stage>/model_N.pt            #   manifest = source of truth, README generated (experiments/tools/bundle_policy.py --sync / --check)
│   ├── POLICY_SUMMARY.md             #   what each stage was trained on: goals, active rewards / terrain / knobs, reward + terrain catalogues
│   └── mdp_pins.yaml                 #   (prose hand-written, tables generated); pins = Isaac config numbers (--pin-mdp, identity-test validated)
└── experiments/                      # Campaigns (see below) + shared eval/, lit reviews, old-runs/, tools/
    ├── eval/                         # Scenario manifests (scenarios_v1/v2/morph.yaml) + baselines/v1/ (moved from <task>/eval/ 2026-09-09)
    ├── lit-review-*.md               # Literature reviews (hexapod reward stability, plasticity cliff, continuous spin; moved from docs/ 2026-09-09)
    ├── old-runs/<timestamp>/         # May-2026 bundled stage checkpoints; the pre-generator USD snapshot sits beside the two flat-walk bundles (2026-05-19, 2026-05-23; byte-identical), the later bundles reference 2026-05-23's (moved from <task>/runs/ 2026-09-09)
    └── tools/                        # run_phases.py, launch_phases.sh, heartbeat_phases.sh, bundle_experiment.py, bundle_policy.py
```

### Experiments and the keep-set

Every campaign lives at `crab_hex_forward_task/experiments/<YYYY-MM-DD_HHMM_name>/` (moved from the former `sim_fine_tuning/` on 2026-09-09; 29 dated campaigns today). The campaign trees live there whole, but git tracks only the **keep-set** per campaign: the records (REPORT / CHANGELOG / CHARTER / RESULTS / PLAN markdown, `state.json`, `heads.json`, csv, driver py/sh), the eval summaries (`run_meta.json`, `scenario_metrics.json`, `summary.md`, gait PNGs -- copied under `<campaign>/evals/` for campaigns that evaluated into `parkour/logs`), the reference RSI banks, and at most ONE checkpoint of record per campaign under `<campaign>/head/` (17 of the 29 campaigns bundle one; the others declare `head: {none: ...}` or `head: {ref: <campaign>}` in `bundle.yaml`) (a sha-verified copy plus README, declared in `<campaign>/bundle.yaml`). Raw artifacts -- intermediate checkpoints, videos, raw eval NPZ, per-episode metrics JSON, console logs, tfevents -- stay on disk and are git-ignored by `experiments/.gitignore`. `experiments/README.md` (the campaign index) is GENERATED by `experiments/tools/bundle_experiment.py --index`; do not hand-write it. `parkour/logs/` and `parkour/outputs/` stay git-ignored. Three non-campaign entries also sit at the top of `experiments/`: `eval/` (the gait-eval scenario manifests `scenarios_v1/v2/morph.yaml` plus the committed `baselines/v1/` reports; moved from `crab_hex_forward_task/eval/` on 2026-09-09), the three `lit-review-*.md` literature reviews (moved from `docs/` the same day), and `old-runs/<timestamp>/`, the seven May-2026 stage-baseline bundles (moved from `crab_hex_forward_task/runs/` the same day: one checkpoint of record each, the paired USD snapshot where one was bundled, and the student's `exported_deploy/` exports; tracked whole as stage releases with no `bundle.yaml`, their `.pt` files re-included by `experiments/.gitignore`). Never add an `__init__.py` anywhere under `experiments/`: `parkour_tasks/__init__.py` runs Isaac's `import_packages`, which walks every subpackage at gym registration and would import the campaign trees (guarded by `tests/unit/test_experiment_layout.py::test_no_package_marker_under_experiments`).

### Policy of record

`crab_hex_forward_task/policy/` is the task's shipped lineage: the current head (`3a_student/model_24995.pt`, the depth student baked 2026-09-09) and the four stage heads it was trained through (`1a_formation`, `2a_elements`, `2b_clearance`, `2c_teacher`), each sha-pinned in `policy/manifest.yaml` with its source run, producing campaign and evals. It changes only on a user bake decision (edit the manifest, `bundle_policy.py --sync`, commit); campaign heads under `experiments/<campaign>/head/` are the provenance copies and stay where they are. `policy/POLICY_SUMMARY.md` explains the training phase by phase (goals, active rewards / terrain / other configuration, reward and terrain catalogues); its tables are generated by `bundle_policy.py --sync` from the presets, the manifest and `policy/mdp_pins.yaml` (Isaac config numbers extracted with `--pin-mdp` from the identity-test dumps). Guarded by `tests/unit/test_policy_of_record.py` and `tests/integration/test_crab_hex_phase_configs.py::test_mdp_pins_match_dumps`.

Provenance records for the current plant and pipeline: `experiments/2026-08-20_1506_hardware_morphology/RESULTS.md` (measured robot → generated plant), `experiments/2026-09-02_1446_leg_mount_morphology/{PLAN,RESULTS}.md` (splay/axis variants), `experiments/2026-09-04_1105_morph_x_exposure/REPORT.md` (A15+B chosen), `experiments/2026-09-06_2130_a15b_lineage/CHANGELOG.md` (charter + bake), `experiments/2026-09-07_1330_phase_pipeline/{CHARTER,REPORT}.md`.

## Assets (robot plant)

- `assets/crab.usda` -- the MAIN asset: the "A15+B" geometry (15 deg outward splay of the front/rear leg mounts, outer yaw axes re-hinged to 2.5 in from the body ends; mid legs unchanged). Training and the gait harness use it by default; no environment variable is needed.
- `assets/variants/crab_simple__splay00_axis5p5in.usda` -- the LEGACY golden: the 2026-08-20 build (splay 0, axes 5.5 in), kept byte-pinned. Every head trained on the 2026-08-20 build before the a15b lineage was trained on it -- campaigns 2026-08-20_1506 .. 2026-09-03_1156 plus the golden/base arms of the 2026-09-02 and 2026-09-04 morphology campaigns (select with `KRABBY_PLANT=legacy_golden` or `--plant legacy_golden`). Era-A heads (campaigns 2026-08-07 .. 2026-08-17 and the `experiments/eval/baselines/v1/` checkpoints) predate the rebuild: they trained on the hand-authored `crab_simple.usda` of their day and no longer load on the current task; the May-2026 `old-runs/` bundles use the snapshot bundled there.
- `assets/crab_simple.usda` -- HISTORICAL, not a plant: the hand-authored Cube model of the 2026-08-09 campaign baseline (reverted 2026-09-09, sha-pinned). Kept as a reference of the robot before the measured-hardware rebuild; no task loads it.
- `assets/variants/*.usda` + `assets/variants/MANIFEST.md` -- the splay/axis variants (B, A10, A15, A20, A10+B, A20+B, and the A15+B file byte-identical to `crab.usda`); the manifest is generated.
- `assets/scripts/generate_crab.py` -- the generator. A plain run writes the main asset; `--legacy-golden` regenerates the legacy golden variant file; `--all-variants` regenerates `assets/variants/` and its manifest. Geometry constants come from `crab_hex_forward_task/mdp/crab_hex_dimensions.py`; the byte-pin tests are `tests/unit/test_crab_hex_usd_generation.py`.
- Plant selection: `KRABBY_PLANT=<name>` for training or `--plant <name>` on the gait harness; an explicitly exported `KRABBY_HEX_USD_PATH` wins over both. The joystick/HAL task (`extreme_parkour_task/config/hex`) reads the same `KRABBY_HEX_USD_PATH` but defaults to the container path `/workspace/assets/crab_hex_ref.usd` (i.e. `assets/crab_hex_ref.usd` inside the Isaac container; `scripts/run_isaac_hal_server.sh` pins the same path), so set the plant per command rather than in a shell profile. Details: [crab-hexapod-plant.md](crab-hexapod-plant.md).

## Documentation (`docs/`)

- [crab-hexapod-plant.md](crab-hexapod-plant.md) -- the robot plant: geometry of record, generator, variants, plant selection, which heads need which plant.
- [crab-hex-forward-policy-config.md](crab-hex-forward-policy-config.md) -- policy / MDP configuration of the crab hexapod forward-walk task and the phase pipeline.
- [GLOSSARY.md](GLOSSARY.md) -- Krabby glossary.
- [TECHNOLOGY_AND_TERMINOLOGY.md](TECHNOLOGY_AND_TERMINOLOGY.md) -- technology stack and terminology.
- [RUNTIME_ARCHITECTURE.md](RUNTIME_ARCHITECTURE.md) -- runtime architecture diagram (HAL, inference, containers).
- [HAL_GUIDE.md](HAL_GUIDE.md) -- Hardware Abstraction Layer guide.
- [SENSOR_INTERFACE.md](SENSOR_INTERFACE.md) -- GStreamer multi-sensor interface (HAL).
- [DATA_COLLECTOR.md](DATA_COLLECTOR.md) -- HAL data collector (rosbag2 / mcap).
- [JETSON_DEPLOYMENT.md](JETSON_DEPLOYMENT.md) -- Jetson deployment guide.
- [MAIXSENSE_A075V_SETUP.md](MAIXSENSE_A075V_SETUP.md) -- MaixSense-A075V camera setup on the Jetson.
- [DOCKER_DEPENDENCIES.md](DOCKER_DEPENDENCIES.md) -- Docker base images and dependencies.
- [PUBLISHING.md](PUBLISHING.md) -- publishing packages to PyPI.
- [TELEOP.md](TELEOP.md) -- teleop (WebRTC remote viewing).
- [m14-bringup-report.md](m14-bringup-report.md) -- M14 bring-up report (Jetson Orin bench).
- [lit-review-hexapod-reward-stability.md](../parkour/parkour_tasks/parkour_tasks/crab_hex_forward_task/experiments/lit-review-hexapod-reward-stability.md) -- literature review: reward design for hexapod stability (lives in the task's `experiments/` since 2026-09-09).
- [lit-review-plasticity-cliff.md](../parkour/parkour_tasks/parkour_tasks/crab_hex_forward_task/experiments/lit-review-plasticity-cliff.md) -- literature review: the plasticity cliff at curriculum stage transitions (lives in the task's `experiments/` since 2026-09-09).
- [lit-review-continuous-spin.md](../parkour/parkour_tasks/parkour_tasks/crab_hex_forward_task/experiments/lit-review-continuous-spin.md) -- literature review: the continuous-spin problem (lives in the task's `experiments/` since 2026-09-09).
- `2309.14341v1.pdf`, `2312.02976v2.pdf` -- reference papers.

## Key Points

### HAL Package Structure (Wheel-based)

The HAL packages are organized with a clean directory structure that matches the import namespace:

- **`hal/client/`**: HAL client package (package name: `krabby-hal-client`, installed via wheel)
  - `hal/client/client.py`: HalClient implementation (ZMQ black-boxed)
  - `hal/client/config.py`: HalClientConfig
  - `hal/client/__init__.py`: Re-exports `HalClient`, `HalClientConfig` for cleaner imports
  - `hal/client/observation/`: Observation types (NavigationCommand only - generic HAL type)
  - `hal/client/data_structures/`: Hardware data structures (`HardwareObservations`, `JointCommand`)

**Model-specific types** (ParkourObservation, ParkourModelIO, InferenceResponse, etc.) are in `compute/parkour/parkour_types.py`.

**Mappers** for converting between hardware and model formats are in `compute/parkour/mappers/`.

- **`hal/server/`**: HAL server base package (package name: `krabby-hal-server`, installed via wheel)
  - `hal/server/server.py`: HalServerBase implementation (ZMQ black-boxed)
  - `hal/server/config.py`: HalServerConfig
  - `hal/server/__init__.py`: Re-exports `HalServerBase`, `HalServerConfig` for cleaner imports

- **`hal/server/isaac/`**: IsaacSim HAL server package (package name: `krabby-hal-server-isaac`, installed via wheel)
  - `hal/server/isaac/hal_server.py`: IsaacSimHalServer (extends HalServerBase)
  - `hal/server/isaac/main.py`: Entry point (console script: `krabby-hal-server-isaac`)
  - `hal/server/isaac/__init__.py`: Re-exports `IsaacSimHalServer` for cleaner imports

- **`hal/server/jetson/`**: Jetson HAL server package (package name: `krabby-hal-server-jetson`, installed via wheel)
  - `hal/server/jetson/hal_server.py`: JetsonHalServer (extends HalServerBase)
  - `hal/server/jetson/zed_camera.py`: ZED camera integration for depth sensing
  - `hal/server/jetson/sensor_backend_jetson.py`: `JETSON_SENSOR_CATALOG`, `JetsonSensorInterface`
  - `hal/server/jetson/main.py`: Console entry (`python -m hal.server.jetson.main`)
  - `hal/server/jetson/__init__.py`: Re-exports `JetsonHalServer` for cleaner imports

- **`hal/tools/`**: HAL debugging tools package (package name: `krabby-hal-tools`, installed via wheel)
  - `hal/tools/hal_dump.py`: CLI tool (console script: `hal-dump`)

**Import Patterns:**
```python
# HAL client/server
from hal.client import HalClient, HalClientConfig
from hal.server import HalServerBase, HalServerConfig
from hal.server.isaac import IsaacSimHalServer
from hal.server.jetson import JetsonHalServer

# Generic HAL types
from hal.client.observation.types import NavigationCommand
from hal.client.data_structures.hardware import (
    HardwareObservations,
    JointCommand,
)

# Model-specific types (Parkour)
from compute.parkour.parkour_types import (
    ParkourObservation,
    ParkourModelIO,
    InferenceResponse,
)

# Mappers
from compute.parkour.mappers.hardware_to_model import HWObservationsToParkourMapper
from compute.parkour.mappers.model_to_hardware import ParkourLocomotionToHWMapper
```

### Single Source of Truth with Editable Installs

The HAL components use a **single source of truth** approach with a clean directory structure:

- **Source files** are located directly in `hal/client/`, `hal/server/`, `hal/server/isaac/`, `hal/server/jetson/`, and `hal/tools/` directories
- **Directory structure matches import namespace**: `hal/client/` → `from hal.client import ...`
- **No redundant nesting**: Clean paths like `hal/client/client.py` instead of `hal/krabby-hal-client/hal/client/client.py`
- **Editable installs for development**: Run `make install-editable` to install packages in editable mode
  - This allows you to edit files in `hal/client/`, `hal/server/`, etc. and see changes immediately
  - No need to rebuild wheels during development
- **Wheel builds for distribution**: Run `make build-wheels` to create distributable wheels
- **Production/Docker**: Install wheels from `hal/*/dist/*.whl` (each package has its own `dist/` directory)

**Development workflow:**
```bash
# Install packages in editable mode (one-time setup)
cd hal/client && pip install -e .
cd ../server && pip install -e .
cd isaac && pip install -e .
cd ../jetson && pip install -e .
cd ../../tools && pip install -e .

# Or use make if available:
make install-editable

# Now you can edit files in hal/client/, hal/server/, etc. and changes are immediately available
# No need to rebuild or reinstall

# To build wheels for distribution/Docker
cd hal/client && python -m build
cd ../server && python -m build
# etc.
```

### Other Components
- **`compute/parkour/`**: Production inference logic (used in production container)
- **`hal/server/jetson/`**: Jetson HAL server (sensors, observations, `python -m hal.server.jetson.main`)
- **`images/locomotion/`**: Production container that runs on the robot (uses wheels)
- **`images/isaacsim/`**: IsaacSim container for simulation (uses wheels)
- **`images/testing/`**: Testing containers for x86 and ARM platforms (uses wheels)
