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
│   │   ├── crab_hex_sideways_task/   # Crab hexapod SIDEWAYS-walk task, thin subclass of the forward task (see below)
│   │   ├── crab_hex_omni_task/       # Crab hexapod OMNIDIRECTIONAL-walk task, thin subclass of the forward task (see below)
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
├── firmware/                         # MCU firmware (arduino/); host sim index: firmware/sim/DOCUMENTATION.md
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
│   │                                 #   firmware/, teleop/, bench/, krabby/, data_collection/, fixtures/ (candidates_v1_2026-09-19.yaml -- the pre-rule-v2 candidates.yaml frozen byte-for-byte for the v1-rule/legacy state-machine tests; candidates_v2_2026-09-22.yaml / candidates_clock_2026-09-23.yaml -- later frozen ladder specs; gait_eval_compile_snapshot.json + its generator gen_gait_eval_compile_snapshot.py -- the step-0 compile-identity snapshot of every committed gait-eval scenario manifest; gait_eval_travel_yaw_golden.json + gen_gait_eval_travel_yaw_golden.py -- travel-yaw golden; omni_ring_golden.npz + omni_ring_golden_forward_decorator.npz + gen_omni_ring_golden.py -- ring-terrain goldens; crab_hex_base_settle.json -- the golden plant's settled base pose (2026-09-02); omni_track_ratio/ -- README + seed3_{1a,1b}_track_ratio.log for test_crab_hex_omni_gate; fixtures/.gitignore re-includes *.log against the root *.log rule), test_experiment_layout.py (campaign keep-set + bundled heads), test_policy_of_record.py (forward policy/ manifest), test_policy_of_record_omni.py (omni policy/ manifest)
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

`parkour/parkour_tasks/parkour_tasks/crab_hex_forward_task/` is the FORWARD-walk task package (renamed from `crab_hexapod_task` on 2026-09-09; a sideways sibling, `crab_hex_sideways_task`, exists since 2026-09-11 -- see below). Gym ids are unchanged: `Isaac-Crab-Hex-Flat-Walk-v0`, `-Teacher-v0`, `-Student-v0` (plus `-Play-v0` variants); the RSL-RL experiment directories are unchanged too: `parkour/logs/rsl_rl/crab_hex_flat_walk|crab_hex_teacher|crab_hex_student`.

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
    └── tools/                        # run_phases.py, launch_phases.sh, heartbeat_phases.sh, bundle_experiment.py, bundle_policy.py, policy_summary.py
```

### Experiments and the keep-set

Every campaign lives at `crab_hex_forward_task/experiments/<YYYY-MM-DD_HHMM_name>/` (moved from the former `sim_fine_tuning/` on 2026-09-09; 30 dated campaigns today). The campaign trees live there whole, but git tracks only the **keep-set** per campaign: the records (REPORT / CHANGELOG / CHARTER / RESULTS / PLAN markdown, `state.json`, `heads.json`, csv, driver py/sh), the eval summaries (`run_meta.json`, `scenario_metrics.json`, `summary.md`, gait PNGs -- copied under `<campaign>/evals/` for campaigns that evaluated into `parkour/logs`), the reference RSI banks, and at most ONE checkpoint of record per campaign under `<campaign>/head/` (17 of the 30 campaigns bundle one; the others declare `head: {none: ...}` or `head: {ref: <campaign>}` in `bundle.yaml`) (a sha-verified copy plus README, declared in `<campaign>/bundle.yaml`). Raw artifacts -- intermediate checkpoints, videos, raw eval NPZ, per-episode metrics JSON, console logs, tfevents -- stay on disk and are git-ignored by `experiments/.gitignore`. `experiments/README.md` (the campaign index) is GENERATED by `experiments/tools/bundle_experiment.py --index`; do not hand-write it. `parkour/logs/` and `parkour/outputs/` stay git-ignored. Three non-campaign entries also sit at the top of `experiments/`: `eval/` (the gait-eval scenario manifests `scenarios_v1/v2/morph.yaml` plus the committed `baselines/v1/` reports; moved from `crab_hex_forward_task/eval/` on 2026-09-09), the three `lit-review-*.md` literature reviews (moved from `docs/` the same day), and `old-runs/<timestamp>/`, the seven May-2026 stage-baseline bundles (moved from `crab_hex_forward_task/runs/` the same day: one checkpoint of record each, the paired USD snapshot where one was bundled, and the student's `exported_deploy/` exports; tracked whole as stage releases with no `bundle.yaml`, their `.pt` files re-included by `experiments/.gitignore`). Never add an `__init__.py` anywhere under `experiments/`: `parkour_tasks/__init__.py` runs Isaac's `import_packages`, which walks every subpackage at gym registration and would import the campaign trees (guarded by `tests/unit/test_experiment_layout.py::test_no_package_marker_under_experiments`).

### Policy of record

`crab_hex_forward_task/policy/` is the task's shipped lineage: the current head (`3a_student/model_24995.pt`, the depth student baked 2026-09-09) and the four stage heads it was trained through (`1a_formation`, `2a_elements`, `2b_clearance`, `2c_teacher`), each sha-pinned in `policy/manifest.yaml` with its source run, producing campaign and evals. It changes only on a user bake decision (edit the manifest, `bundle_policy.py --sync`, commit); campaign heads under `experiments/<campaign>/head/` are the provenance copies and stay where they are. `policy/POLICY_SUMMARY.md` explains the training phase by phase (goals, active rewards / terrain / other configuration, reward and terrain catalogues); its tables are generated by `bundle_policy.py --sync` from the presets, the manifest and `policy/mdp_pins.yaml` (Isaac config numbers extracted with `--pin-mdp` from the identity-test dumps). Guarded by `tests/unit/test_policy_of_record.py` and `tests/integration/test_crab_hex_phase_configs.py::test_mdp_pins_match_dumps`.

Provenance records for the current plant and pipeline: `experiments/2026-08-20_1506_hardware_morphology/RESULTS.md` (measured robot → generated plant), `experiments/2026-09-02_1446_leg_mount_morphology/{PLAN,RESULTS}.md` (splay/axis variants), `experiments/2026-09-04_1105_morph_x_exposure/REPORT.md` (A15+B chosen), `experiments/2026-09-06_2130_a15b_lineage/CHANGELOG.md` (charter + bake), `experiments/2026-09-07_1330_phase_pipeline/{CHARTER,REPORT}.md`.

## Crab hexapod sideways-walk task

`parkour/parkour_tasks/parkour_tasks/crab_hex_sideways_task/` (exists since 2026-09-11) is a thin
subclass of `crab_hex_forward_task/`: same plant (A15+B), same MDP terms and hyper-parameters, but
the travel axis is the robot's body **+y** instead of +x ("left side leads", spawn yaw -pi/2). It
reuses `crab_hex_forward_task/mdp/*` unchanged and re-declares only the cfg-valued attributes that
need the travel direction (scene, commands, rewards) -- see the task's own README for the full
delta table.

```
crab_hex_sideways_task/
├── README.md                         # Task usage + the delta table vs. the forward task
├── config/crab_hex/                  # Env, scene cfgs; crab_hex_side_phases.py (KRABBY_SIDE_PHASE presets)
│   └── agents/                       # RSL-RL PPO / student runner cfgs
├── mdp/                              # crab_hex_side_mirror.py (fore-aft mirror-symmetry augmentation)
└── experiments/                      # Campaigns; eval/scenarios_side.yaml; tools/ (driver copies)
```

Preset variable: `KRABBY_SIDE_PHASE` (never `KRABBY_PHASE` in the same process -- setting both
raises at import). Gym ids: `Isaac-Crab-Hex-Side-Walk-v0`, `-Side-Teacher-v0`, `-Side-Student-v0`
(plus `-Play-v0` variants). RSL-RL experiment directories:
`parkour/logs/rsl_rl/crab_hex_side_{flat_walk,teacher,student}`; eval root:
`parkour/logs/rsl_rl/gait_eval/side_phases`. Same keep-set / campaign-dir conventions as the
forward task's `experiments/` (above), including: never add an `__init__.py` anywhere under
`experiments/` (same reason -- `parkour_tasks/__init__.py`'s `import_packages` would import the
campaign trees at gym registration).

## Crab hexapod omnidirectional-walk task

`parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task/` (exists since 2026-09-16) is a thin
subclass of `crab_hex_forward_task/`: same plant (A15+B), same MDP terms and hyper-parameters, but
travel and facing are two INDEPENDENT world-frame commands (any direction, any absolute heading)
instead of one body-frame axis, sampled across four command-regime slots (stand / move / turn /
move+turn), and the obstacle course is a ring of randomly placed blocks around the spawn instead of
a corridor. It reuses `crab_hex_forward_task/mdp/*` unchanged and re-declares only the cfg-valued
attributes the world-frame commands need (scene, commands, observations, rewards, terminations,
parkour term, RSI, mirror) -- see the task's own README for the full delta table.

```
crab_hex_omni_task/
├── README.md                         # Task usage + the delta table vs. the forward task + "Policy of record (F19, baked 2026-10-24)"
├── config/crab_hex/                  # Env, scene cfgs; crab_hex_omni_phases.py (KRABBY_OMNI_PHASE presets); omni_knobs.py
│   └── agents/                       # RSL-RL PPO runner cfgs (teacher + the depth-student runner, Task 4 2026-10-16)
├── mdp/                              # crab_hex_omni_command(_math).py; omni observations/rewards/mirror/RSI; omni_parkour_event.py, omni_exposure.py, omni_events.py, omni_terrain_knobs.py (ring + field geometry); stroke (crab_hex_omni_stroke_{math,parity,rewards}.py), motion (crab_hex_omni_motion_rewards.py), restance (crab_hex_omni_restance.py), yaw window (crab_hex_omni_yaw_window.py), clearance (omni_clearance.py)
├── eval_categories.py                # Per-category eval plan (2026-09-21, D4): trained_categories/expand_rule/composite_guards -- Isaac-free, loaded by path; task-root (outside mdp/config), so it hashes into eval_sha, not code_sha
├── policy/                           # POLICY OF RECORD: the shipped head + the stage heads it was trained through
│   ├── manifest.yaml                 #   (F19 lineage, baked 2026-10-24: f19_1_s1 -> f19_2_c4 -> f19_3_c5 -> f19_4_c6 -> f19_5_c6x_clockfree -> f19_6_p2a_field -> f19_7_p3a_payload90 -> f19_8_p3b_payload180 = current T*; + f19_9_student (ONNX exports only) and f19_stand_net/ (stand network of record); heads, obs_contract, construction, stages, stand_network, superseded (S1_s18, SH_T17, SN_Amask_s17, W), followups blocks); manifest = source of truth, README generated (experiments/tools/bundle_policy.py --sync / --check)
│   ├── <stage>/model_N.pt            #   one .pt per RL stage
│   └── 1_fb6k .. 6_p2a_field/, stand_net/  # SUPERSEDED S1 s18 lineage and SH_T17 / A-mask s17 stand nets, files kept in place (superseded[].files)
└── experiments/                      # Campaigns (index: experiments/README.md); eval/scenarios_omni{,_body}.yaml + scenarios_gpu0_sidesweep.yaml; lit reviews; tools/
    ├── 2026-09-16_1805_omni_formation/   # First training campaign (CLOSED at its 2b decision point): CHARTER.md, RSI_BANK.md, replay_reward_economy.py
    ├── 2026-09-19_facing_tracking_ladder/ # The facing/tracking candidate ladder: CHARTER.md, PLAN_AMENDMENTS.md, candidates.yaml (the spec), resolution.json (D00). Written by the driver at launch, not present before it: ledger.json + ledger_overrides.jsonl, REPORT.md, CHANGELOG.md, notify.log, state.json/heads.json. Per-category evaluation plan (2026-09-21, D9) adds, once the migration runs: resolution.<eval_sha[:12]>.json (D00s/D00bs, superseding D00 at the new eval) and, only on a failed acceptance, resolution.<eval_sha[:12]>.<kind>.rejected.json (kind = D00s|D00bs, overwritten each recomputation, never cited)
    ├── 2026-10-11_stand_network/scripts/, 2026-10-13_stand_height/scripts/  # stand-network trainers (sn_dagger / sh_dagger), eval launchers (sn_/sh_eval_rows.sh), table tests and video recorders (sh_record_video.sh/.py: --plan mix, --terrain field_max) -- the viewer + eval tools for the stand net of record
    ├── 2026-10-16_task4_pipeline/    # M18 Task 4 (CLOSED): the campaign that produced the policy of record; pipeline.yaml, pipeline.lock.json, F19/ + R18/ runs, REPORT_baseline_vs_final.md, generated README.md, head/
    └── tools/                        # run_phases.py + launch/heartbeat (the phase chain); run_candidates.py + ledger_cli.py + resolution.py + launch_candidates.sh (the candidate ladder); bundle_policy.py / bundle_experiment.py (policy folder + campaign bundles); record_categories.py, replay_*.py, score_reference.py (viewer / CPU tools); pipeline/ (the Task 4 pipeline: pipeline.py, distill.sh, report.py, gate_read.py, gate_row.sh, select_student.py, ...)
```

Outputs of the stand-network, phase-2 and Task 4 runs land under `parkour/outputs/omni_plan/{task4,stand_network,stand_height,phase2}` (git-ignored).

### Policy of record

`crab_hex_omni_task/policy/` is the task's shipped lineage: the current head T*
(`f19_8_p3b_payload180/model_25992.pt`, seed 19, payload 0-180 kg, produced by the M18 Task 4 pipeline and baked
2026-10-24, commit de81f98a) and the stage heads it was trained through (`f19_1_s1`, `f19_2_c4`, `f19_3_c5`,
`f19_4_c6`, `f19_5_c6x_clockfree` = the phase-1 head, `f19_6_p2a_field` = the phase-2 head, `f19_7_p3a_payload90`),
each sha-pinned in `policy/manifest.yaml` with its source run, preset (`phase`: `1b` for `f19_1_s1`, `1a` for
c4..c6x, `2a` for p2a / p3a / p3b), overlay (`overlay_ref` into the pipeline-generated `candidates.yaml`, not a
`KRABBY_OMNI_PHASE` preset of its own -- there is no phase-1c / 2b / 2c lineage here) and evals (the phase-2 and
payload stages also their obstacle / payload rows). Beside them: the depth student `f19_9_student/` (a
`kind: distill` stage with `bundled: false` -- its 122 MB `.pt` is cited by sha256 and source path because it is
over GitHub's 100 MiB limit; the folder holds the ONNX exports `policy.onnx` + `depth_latest.onnx` +
`omni_student_io.json`), and the `stand_network:` block (`f19_stand_net/model_5000.pt`, SH_T4F19, walker = T*,
plus its warm-start ancestor `model_4000.pt`). It changes only on a user bake decision (edit the manifest,
`bundle_policy.py --sync`, commit); campaign records under `experiments/<campaign>/` are the provenance. Unlike
the forward policy, the manifest also carries an `obs_contract` block (the clock observation columns, which stage
was trained with `KRABBY_OMNI_CLOCK_HIDE` on, and the exact eval-env knobs the heads trained under), a `followups`
block (open items recorded at bake time, e.g. `move_turn`), a `construction` block (the runner-side PPO knobs
every stage trained with, less a stage's own `construction_delta` -- the phase-2 `KRABBY_ACTION_RATE_W -0.05`;
only `KRABBY_STD_PARAM` / `KRABBY_INIT_STD` reach eval and play), a `payload:` block per payload stage, and a
`superseded` block of earlier heads of record: entries whose files left the folder (W: source runs, sha256 and the
last commit that holds them) and entries whose files STAY in place under `superseded[].files` (the seed-18
lineage S1 s18 = `1_fb6k` .. `6_p2a_field`, and SH_T17 / A-mask s17 in `stand_net/`). No `--pin-mdp` /
`POLICY_SUMMARY.md` here: every stage is an overlay on a `KRABBY_OMNI_PHASE` preset recorded in a campaign's
`candidates.yaml`, not its own preset. Guarded by `tests/unit/test_policy_of_record_omni.py`.

Preset variable: `KRABBY_OMNI_PHASE` (never `KRABBY_PHASE` or `KRABBY_SIDE_PHASE` in the same
process -- setting any two raises at import). Gym ids: `Isaac-Crab-Hex-Omni-Walk-v0`,
`-Omni-Teacher-v0`, `-Omni-Student-v0` (plus `-Play-v0` variants; the student ids since the Task 4 pipeline,
2026-10-16). RSL-RL experiment directories: `parkour/logs/rsl_rl/crab_hex_omni_{flat_walk,teacher,student}`;
eval root: `parkour/logs/rsl_rl/gait_eval/omni_phases`; Task 4, stand-network and phase-2 outputs under
`parkour/outputs/omni_plan/`. Same keep-set / campaign-dir conventions as the
forward task's `experiments/` (above), including: never add an `__init__.py` anywhere under
`experiments/` (same reason -- `parkour_tasks/__init__.py`'s `import_packages` would import the
campaign trees at gym registration). The ring-terrain generator lives outside this task package,
in the shared import graph: `parkour/parkour_isaaclab/terrains/omni/`. Two drivers share
`experiments/tools/`: `run_phases.py` runs the preset chain `1a`..`2c` (launch/heartbeat via
`launch_phases.sh` / `heartbeat_phases.sh`) and REFUSES a campaign `state.json` that carries a
`ladder` key, while `run_candidates.py` (with `ledger_cli.py`, `resolution.py` and
`launch_candidates.sh`) runs the candidate ladder, whose campaign dir adds `candidates.yaml`,
`resolution.json`, `ledger.json` and `ledger_overrides.jsonl` to the usual
`state.json`/`heads.json`/`REPORT.md`/`CHANGELOG.md` records. `record_categories.py` beside them is
a VIEWER, not a driver: it replays a head through one follow-cam episode per suite command hold
(both hands, the command term's own travel/facing/achieved arrows on) composed exactly as the
per-category eval composes it, and writes mp4s plus a sanity JSON under `parkour/logs/videos/` --
nothing in the ladder reads its output, and it is outside both `code_sha` and `eval_sha`. Its defaults
reproduce the 2026-09-22 C1d recording (a superseded head); to record the policy of record pass
`--checkpoint policy/f19_8_p3b_payload180/model_25992.pt --eval-run <T*'s suite__omni stamp>`.

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
