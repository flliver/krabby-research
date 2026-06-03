# Crab hexapod task (`crab_hexapod_task`)

This package adds a **Krabby hexapod** parkour task on top of Isaac Lab’s extreme parkour stack.  
The goal of this README is that anyone can clone the repo, create a Python env similar to yours, and **train + play** the hexapod policy.

The examples below assume:

- `**$KRABBY_ROOT=/home/sanjay/Projects/krabby`**
- `**krabby-research`** lives at `**$KRABBY_ROOT/krabby-research`**
- **Isaac Lab** lives at `**$KRABBY_ROOT/IsaacLab`**
- Your Isaac Lab conda env is called `**env_isaaclab`**

Adjust paths and the conda env name if your layout is different.

---

## 1. Environment setup (once per machine)

All commands in this README assume:

Installation: [https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/binaries_installation.html](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/binaries_installation.html)

```bash
conda activate env_isaaclab
```

Then install and point Python at the **krabby-research** copies of `parkour` and `parkour_tasks`:

```bash
export KRABBY_ROOT=/home/sanjay/Projects/krabby

conda activate env_isaaclab

cd "$KRABBY_ROOT/krabby-research/parkour"
pip install -e .

cd "$KRABBY_ROOT/krabby-research/parkour/parkour_tasks"
pip install -e .

export PYTHONPATH="$KRABBY_ROOT/krabby-research/parkour/parkour_tasks:$KRABBY_ROOT/krabby-research/parkour:${PYTHONPATH}"
```

Isaac Lab itself is launched via:

```bash
cd "$KRABBY_ROOT/IsaacLab"
./isaaclab.sh -p ...
```

### Hexapod asset (canonical)

This task uses **only** `[krabby-research/assets/crab_simple.usda](../../../assets/crab_simple.usda)`. The scene config resolves that path automatically from the repo layout (see `_crab_simple_usd_path()` in `crab_hex_scene_cfg.py`).

Optional override (Docker or non-standard layouts):

```bash
export KRABBY_HEX_USD_PATH="$KRABBY_ROOT/krabby-research/assets/crab_simple.usda"
```

You can point `KRABBY_HEX_USD_PATH` at a flattened `.usd` export for deployment; the default authoring file is `crab_simple.usda`.

**Spawn height:** The USD root `krabby` is offset **+1 m** in the file; `[_crab_simple_robot_cfg()](config/crab_hex/crab_hex_scene_cfg.py)` sets articulation spawn `z` from `KRABBY_HEX_SPAWN_Z` (default `**1.05`** m). Use the same value for train, play, and stance checks. If the robot **floats then slams**, **lower** slightly; if **hips scrape** or the root **interpenetrates**, **raise** in ~**0.02** m steps on flat ground.

**Default joint pose (rad):** body–hip yaw splay **±0.6** on front/rear legs, **±0.25** on middle legs (ML **+0.25**, MR **−0.25**); `Hip_Femur` **0.30**; `Femur_Tibia` left legs **−0.07**, right legs **+0.10**. Tune in `crab_hex_scene_cfg.py` if the passive stance is wrong.

---

## 2. How stages differ

**Current priority:** use bundled **student** `9800` ([Appendix G](#appendix-g--stage-3-student-distillation--2026-05-26)) for deploy/play on the student MDP; **Stage 4 `full` parkour** is next ([Stage 4](#stage-4--full-parkour-todo)).

> **Do not jump to `full` parkour yet.** Skipping to `**full`** (0.25 / ±4.8, diff 0–1) from bridge, 2b1, or 2b2 usually thrashes or collapses to in-place shuffling. Finish validating the **student** baseline on 2b2-mixed play before Stage 4.

Each stage resumes the previous bundled checkpoint. Same policy network throughout. Commands: [§4](#4-training-and-playing-the-hexapod). Config files: [§3](#3-config-reference).

### At a glance


| Stage         | Task / mode             | Resume from        | Terrain               | Rewards                                                         | Actions     | Success in play              |
| ------------- | ----------------------- | ------------------ | --------------------- | --------------------------------------------------------------- | ----------- | ---------------------------- |
| **1 flat**    | `Flat-Walk-v0`          | scratch            | 100% flat             | gait + command tracking                                         | 0.24 / ±1   | stable flat walk             |
| **2a bridge** | `Teacher-v0` + `bridge` | `6000`             | easy mixed, frozen    | velocity/posture; **no goal**                                   | 0.24 / ±1   | flat + light gaps            |
| **2b1**       | `Teacher-v0` + `2b1`    | `6099`             | **same as 2a**        | weak goal/yaw added                                             | 0.24 / ±1   | mixed walk; steps still hard |
| **2b2**       | `Teacher-v0` + `2b2`    | `6198`             | 50/50, curriculum on  | lift-first teacher stack ([§4.2b](#42b-2b2-teacher-sweet-spot)) | 0.24 / ±1   | lift over holes/steps        |
| **3 student** | `Student-v0`            | teacher `6300`     | student MDP (2b2 mix) | distillation                                                    | student cfg | obstacle-walk on student MDP |
| **4 full**    | `Teacher-v0` (`full`)   | 2b2 `6300` (later) | full parkour          | Go2-style full teacher                                          | 0.25 / ±4.8 | **deferred**                 |


**Transition cheat sheet:** **1→2a** teacher env + easy mix, still command-following · **2a→2b1** rewards only (same terrain) · **2b1→2b2** terrain + lift rewards · **2b2→3** privileged → depth obs · **2b2→4** large MDP jump — do not skip student.

---

### Stage 1 — flat walk

**What changes**

- **MDP / terrain:** `Flat-Walk-v0`; 100% flat; difficulty 0.1–0.25; curriculum off; reduced domain randomization.
- **Rewards:** `CrabHexFlatWalkRewardsCfg` — command tracking + gait shaping (air time, tibia deviation, idle-foot penalties). See [§3](#3-config-reference).
- **Actions / PPO:** scale **0.24**, clip **±1**; `lin_vel_x` **(0.30, 0.65)**; 20k iters, save every 100.

**What stays the same:** flat-walk task (no `KRABBY_HEX_TEACHER_MODE`); no parkour goals.

**Why this stage exists:** learn a reliable hex gait before any teacher or parkour MDP.

**What good play looks like:** walks forward on flat; alternating feet; stays upright.

**Bundled checkpoint:** [Appendix C](#appendix-c--stage-1-flat-walk--2026-05-23-baseline) `runs/2026-05-23_10-15-21/model_6000.pt`

→ **1 → 2a:** move from flat-walk env into teacher env, but keep easy terrain and command-following rewards.

---

### Stage 2a — bridge

**What changes**

- **MDP / terrain:** `Teacher-v0` + `KRABBY_HEX_TEACHER_MODE=bridge`; ~82% flat / ~18% parkour; shallow gaps; difficulty 0.08–0.30; **frozen** terrain levels; `lin_vel_x` **(0.45, 0.85)**.
- **Rewards:** `CrabHexTeacherBridgeRewardsCfg` — velocity + forward progress + posture; **goal_vel / yaw = 0**.
- **Actions / PPO:** 0.24 / ±1; **100** iters, LR **3e-5**; resume flat `6000`.

**What stays the same:** bridge-lite physics (no push/mass/COM DR); same action scale as flat.

**Why this stage exists:** transfer the flat gait into the teacher stack on mostly flat ground without parkour goal pressure.

**What good play looks like:** stable forward walk on flat + light tiles; some heading drift OK.

**Bundled checkpoint:** [Appendix D](#appendix-d--stage-2a-teacher-bridge--2026-05-25-baseline) `runs/2026-05-25_22-26-06/model_6099.pt`

→ **2a → 2b1:** same terrain and physics; only weak parkour goal/yaw rewards added.

---

### Stage 2b1 — hybrid walk

**What changes**

- **MDP / terrain:** *(none vs 2a)* — same easy mixed, frozen terrain.
- **Rewards:** `CrabHexStage2BPhase1RewardsCfg` — bridge core + `goal_vel` **0.75**, `yaw` **0.2** + body regularizers.
- **Actions / PPO:** 0.24 / ±1; **100** iters, LR **3e-5**; resume bridge `6099`.

**What stays the same:** terrain, actions, commands, bridge-lite DR.

**Why this stage exists:** introduce weak parkour goal signals before harder terrain (optional — see [footnote](#footnote--curriculum-staging-and-future-full-teacher)).

**What good play looks like:** good on flat/light tiles; steps and hurdles still hard (expected).

**Bundled checkpoint:** [Appendix E](#appendix-e--stage-2b1-hybrid-walk--2026-05-25-baseline) `runs/2026-05-25_23-57-58/model_6198.pt`

→ **2b1 → 2b2:** harder 50/50 terrain + curriculum on + lift-first reward stack; this becomes the **distillation teacher**.

---

### Stage 2b2 — teacher-ready obstacle walk

**What changes**

- **MDP / terrain:** 50% flat / 50% parkour; difficulty **0.20–0.70** with curriculum **on**; moderate steps/gaps/hurdles; moderate push/mass/COM DR.
- **Rewards:** `CrabHexStage2BPhase2RewardsCfg` — lift-first stack; bridge velocity aux **zeroed**. Additional lift delta: `reward_swing_vertical_vel` **0.8**, `penalty_swing_min_clearance` **−0.4**, `reward_recover_from_stall` **0.2**. Full weights + sweet-spot gates: [§4.2b](#42b-2b2-teacher-sweet-spot).
- **Actions / PPO:** 0.24 / ±1; up to **10k** iters, LR **1e-4**, save every 100; **stop at sweet-spot** (bundled @ **6300**); resume 2b1 `6198` only.

**What stays the same:** action scale; same teacher task; same policy network.

**Why this stage exists:** produce a **2b2 teacher** that lifts over holes/steps for student distillation.

**What good play looks like:** steady forward walk; **lifts legs from holes**; some stumble OK; do not use `6400+` from the same log.

**Bundled checkpoint:** [Appendix F](#appendix-f--stage-2b2-teacher-ready-baseline--2026-05-26) `runs/2026-05-26_21-46-37/model_6300.pt`

→ **2b2 → 3:** same teacher MDP for rollouts; student learns from depth + proprio ([§3.1](#31-teacher-vs-student)).

---

### Stage 3 — student distillation

**What changes**

- **MDP / terrain:** `Student-v0`; depth + proprioception (no privileged terrain scan). **Terrain matches 2b2 teacher** (50/50 flat/parkour, difficulty **0.20–0.70**, 2b2 geometry, `lin_vel_x` **(0.45, 0.85)**, bridge-lite DR).
- **Rewards:** distillation losses — match teacher actions/values ([§4.4](#44-student-distillation)).
- **Actions / PPO:** student action cfg (**0.24 / ±1**, action delay on); teacher policy loaded from 2b2 `6300`.

**What stays the same:** teacher checkpoint `6300`; teacher and student both roll out in the **same** student env MDP.

**Why this stage exists:** deployable policy without privileged observations.

**What good play looks like:** forward walk and hurdle crossing on **2b2-mixed** student terrain (not dedicated flat-walk).

**Bundled checkpoint:** [Appendix G](#appendix-g--stage-3-student-distillation--2026-05-26) `runs/2026-05-26_22-57-01/model_9800.pt`

→ **3 → 4:** large jump in actions, terrain, and rewards; resume from 2b2 teacher `6300` for `full`, not from student `9800`.

---

### Stage 4 — full parkour (TODO)

**What changes**

- **MDP / terrain:** full Go2 sub-terrain mix; difficulty **0–1**; curriculum on; full domain randomization.
- **Rewards:** `CrabHexRewardsCfg` — `goal_vel` **2.25**, collision **−6**, etc.
- **Actions / PPO:** **0.25 / ±4.8**; LR **2e-4**; resume 2b2 `6300`.

**What stays the same:** teacher task family; policy network (in principle).

**Why this stage exists:** eventual Go2-style full parkour teacher — **not started**.

**What good play looks like:** deferred.

**Bundled checkpoint:** none. Plan: [Appendix footnote](#footnote--curriculum-staging-and-future-full-teacher).

---

## 3. Config reference

Scene, rewards, and code pointers. Stage differences: [§2](#2-how-stages-differ).

- **Gym registrations:** `config/crab_hex/__init__.py` — `Flat-Walk-v0`, `Teacher-v0`, `Student-v0`, `*-Play-v0`.
- **Scene / robot:** `crab_hex_scene_cfg.py` — `crab_simple.usda`, spawn `KRABBY_HEX_SPAWN_Z`, contact sensor on `.*_Footpad`.
- **Env / curriculum:** `crab_hex_env_cfg.py` — `KRABBY_HEX_TEACHER_MODE` selects bridge / 2b1 / 2b2 / `full`; terrain helpers `_apply_crab_hex_stage_2b_`*.
- **Rewards / actions:** `parkour_mdp_cfg.py` — config classes per stage ([§2](#2-how-stages-differ)); math in `parkour_isaaclab/envs/mdp/rewards.py`.
- **2b2 full reward weights:** [§4.2b](#42b-2b2-teacher-sweet-spot) only (not duplicated here).

### 3.1 Teacher vs student


|                  | **Teacher**                                                    | **Student**                                                     |
| ---------------- | -------------------------------------------------------------- | --------------------------------------------------------------- |
| **Tasks**        | `Flat-Walk-v0` (stage 1); `Teacher-v0` (2a–2b2, future `full`) | `Student-v0`                                                    |
| **Observations** | Privileged (terrain scan, dynamics, etc.)                      | Depth + proprioception                                          |
| **Training**     | PPO on teacher MDP                                             | Distillation from 2b2 `6300` ([§4.4](#44-student-distillation)) |


Set `KRABBY_HEX_TEACHER_MODE=2b2` when loading the teacher for student rollouts.

### RSL-RL runner factory

**Committed:** `parkour/scripts/rsl_rl/runner_factory.py`  
`train.py`, `play.py`, and `evaluation.py` call `agent_cfg_to_train_dict()` and `make_on_policy_runner()` instead of `agent_cfg.to_dict()` directly. That fixes corrupted scalars from multi-inherit `configclass` `to_dict()` (e.g. `num_steps_per_env` becoming an `obs_groups` dict).

**Crab runner (committed):**


| File                                                       | Purpose                                                 |
| ---------------------------------------------------------- | ------------------------------------------------------- |
| `scripts/rsl_rl/crab_on_policy_runner.py`                  | `OnPolicyRunnerCrabHex` → loads `CrabHexActorCriticRMA` |
| `scripts/rsl_rl/modules/crab_actor_critic_with_encoder.py` | RMA policy with **clamped** Gaussian action std         |


Go2 uses stock `OnPolicyRunnerWithExtractor` / `ActorCriticRMA` via the same factory.

**Local only (gitignored):** `crab_hexapod_task/tempscripts/` — optional diagnostics (`audit_crab_joint_drives.py`, `verify_crab_simple_usda.py`, `diagnose_obs_action_alignment.py`, `diagnose_forward_rollout.py`).

**Crab routing (`make_on_policy_runner`)** — uses `OnPolicyRunnerCrabHex` when any of:


| Condition                                      | Example                                    |
| ---------------------------------------------- | ------------------------------------------ |
| `runner_class_name == "OnPolicyRunnerCrabHex"` | `crab_hex_rl_cfg.py`                       |
| `policy.class_name == "CrabHexActorCriticRMA"` | Flat-walk / teacher / student              |
| `estimator.num_prop == 75`                     | `CrabHexParkourObservations` (Go2: **53**) |


**Related (committed):** `config/crab_hex/agents/crab_hex_rl_cfg.py`, `crab_hexapod_task/mdp/observations.py`, `modules/on_policy_runner_with_extractor.py`.

---

## 4. Training and playing the hexapod

All commands in this section assume:

```bash
export KRABBY_ROOT=/home/sanjay/Projects/krabby
export KRABBY_HEX_SPAWN_Z=1.05
conda activate env_isaaclab
export PYTHONPATH="$KRABBY_ROOT/krabby-research/parkour/parkour_tasks:$KRABBY_ROOT/krabby-research/parkour:${PYTHONPATH}"
# Optional if the default path resolver finds crab_simple.usda:
# export KRABBY_HEX_USD_PATH="$KRABBY_ROOT/krabby-research/assets/crab_simple.usda"
```

**Where checkpoints are written:** `[parkour/scripts/rsl_rl/train.py](../../../scripts/rsl_rl/train.py)` sets the log root to `abspath("logs/rsl_rl/<experiment_name>")`, i.e. it is **relative to the shell’s current working directory**. There is no separate `--log_root` flag.

- **Recommended (checkpoints under `krabby-research`):** `cd` into `**$KRABBY_ROOT/krabby-research/parkour`**, then run `**$KRABBY_ROOT/IsaacLab/isaaclab.sh`** with an absolute `-p` path to `train.py` / `play.py`. Artifacts land in `**krabby-research/parkour/logs/rsl_rl/...**`.
- **Resume training** must use the **same** `cd` as the original run, because `--resume` / `--load_run` resolve under that directory’s `logs/rsl_rl/<experiment_name>/`. With `--load_run` + `--checkpoint model_XXXX.pt`, paths resolve under that run folder (bare filenames like `model_6000.pt` work when `cd` is `krabby-research/parkour`).
- **Play** with an explicit `**--checkpoint`** uses that file path for inference; cwd does not change which weights load. You may still see a line like `Loading experiment from directory: ...` that reflects cwd-based `log_root_path`—when you pass `--checkpoint`, the run uses the checkpoint path you gave.

**Alternative:** if you `cd "$KRABBY_ROOT/IsaacLab"` and run `./isaaclab.sh`, checkpoints go under `**IsaacLab/logs/rsl_rl/...`** instead (same script, different cwd).

### 4.0 Training commands (curriculum)

Stage differences: [§2](#2-how-stages-differ). Bundled checkpoints: appendices C–G.

```bash
export KRABBY_ROOT=/home/sanjay/Projects/krabby
RUNS_DIR="$KRABBY_ROOT/krabby-research/parkour/parkour_tasks/parkour_tasks/crab_hexapod_task/runs"
USD="$RUNS_DIR/2026-05-23_10-15-21/crab_simple_2026-05-23_10-15-21.usda"
```

**Actions (stages 1–2b2):** scale **0.24**, clip **±1**. `**full`:** 0.25 / ±4.8. **Play:** `KRABBY_HEX_TEACHER_MODE` must match training ([§4.3](#43-play-a-bundled-checkpoint)). Ad-hoc log checkpoints use direct `play.py` one-liners (examples in [§4.3](#43-play-a-bundled-checkpoint)).

**Aliases:** `stage2b1` → `2b1`, `stage2b2` → `2b2`.

**Stage 1 — flat walk** (`logs/rsl_rl/crab_hex_flat_walk/`):

```bash
cd "$KRABBY_ROOT/krabby-research/parkour"
"$KRABBY_ROOT/IsaacLab/isaaclab.sh" -p "$KRABBY_ROOT/krabby-research/parkour/scripts/rsl_rl/train.py" \
  --task Isaac-Crab-Hex-Flat-Walk-v0 \
  --headless --num_envs 256 --seed 1 --max_iterations 20000
```

Checkpoints save every **100** iterations. Bundled baseline: [Appendix C](#appendix-c--stage-1-flat-walk--2026-05-23-baseline).

**Stage 2a — bridge** (`export KRABBY_HEX_TEACHER_MODE=bridge`):

```bash
export KRABBY_ROOT=/home/sanjay/Projects/krabby
export KRABBY_HEX_TEACHER_MODE=bridge
conda activate env_isaaclab

FLAT_CKPT="$KRABBY_ROOT/krabby-research/parkour/parkour_tasks/parkour_tasks/crab_hexapod_task/runs/2026-05-23_10-15-21/model_6000.pt"

cd "$KRABBY_ROOT/krabby-research/parkour"
"$KRABBY_ROOT/IsaacLab/isaaclab.sh" -p "$KRABBY_ROOT/krabby-research/parkour/scripts/rsl_rl/train.py" \
  --task Isaac-Crab-Hex-Teacher-v0 \
  --headless --num_envs 256 --seed 1 \
  --resume --checkpoint "$FLAT_CKPT" \
  --max_iterations 100
```

PPO: **100** iters, LR `3e-5` → `**6099`**. Provenance: [Appendix D](#appendix-d--stage-2a-teacher-bridge--2026-05-25-baseline).

**Stage 2b1** (`export KRABBY_HEX_TEACHER_MODE=2b1`):

```bash
export KRABBY_HEX_TEACHER_MODE=2b1
conda activate env_isaaclab
BRIDGE_CKPT="$KRABBY_ROOT/krabby-research/parkour/parkour_tasks/parkour_tasks/crab_hexapod_task/runs/2026-05-25_22-26-06/model_6099.pt"
cd "$KRABBY_ROOT/krabby-research/parkour"
"$KRABBY_ROOT/IsaacLab/isaaclab.sh" -p scripts/rsl_rl/train.py \
  --task Isaac-Crab-Hex-Teacher-v0 --headless --num_envs 256 --seed 1 \
  --resume --checkpoint "$BRIDGE_CKPT" --max_iterations 100
```

**Stage 2b2** — resume 2b1 `6198` only; rewards and gates: [§4.2b](#42b-2b2-teacher-sweet-spot). **Stop early** at sweet-spot; do not promote the last saved iter.

```bash
export KRABBY_HEX_TEACHER_MODE=2b2
CKPT2B1="$KRABBY_ROOT/krabby-research/parkour/parkour_tasks/parkour_tasks/crab_hexapod_task/runs/2026-05-25_23-57-58/model_6198.pt"
cd "$KRABBY_ROOT/krabby-research/parkour"
"$KRABBY_ROOT/IsaacLab/isaaclab.sh" -p scripts/rsl_rl/train.py \
  --task Isaac-Crab-Hex-Teacher-v0 --headless --num_envs 256 --seed 1 \
  --resume --checkpoint "$CKPT2B1"
```

**Stage 3 — student:** [§4.4](#44-student-distillation). **Stage 4 — `full`:** [Appendix footnote](#footnote--curriculum-staging-and-future-full-teacher).

**TensorBoard / metrics:** [§4.2](#42-teacher-training-utilities-logs-tensorboard-resume). Prioritize play + `crab_failure`, `mean_episode_length`, `reward_forward_progress_along_command`, and (2b2) `reward_obstacle_clearance`.

**Stance check (no checkpoint):** [§4.1](#41-zero-agent-stance-check-no-policy) (`zero_agent.py`).

### 4.1 Zero-agent stance check (no policy)

`[parkour/scripts/zero_agent.py](../../../scripts/zero_agent.py)` runs any registered `parkour_tasks` env with **all-zero actions** (hold default joint targets from `crab_hex_scene_cfg.py`; no checkpoint). Use this to verify **spawn height**, **default pose**, and `**CRAB_HEX_VIEWER`** before training.

**Recommended for a flat stance check** (easy terrain, hex camera, one env):

```bash
export KRABBY_ROOT=/home/sanjay/Projects/krabby
export KRABBY_HEX_SPAWN_Z=1.05
conda activate env_isaaclab

cd "$KRABBY_ROOT/krabby-research/parkour"
"$KRABBY_ROOT/IsaacLab/isaaclab.sh" -p "$KRABBY_ROOT/krabby-research/parkour/scripts/zero_agent.py" \
  --task Isaac-Crab-Hex-Teacher-Play-v0 \
  --num_envs 1
```

- Default `--task` is `Isaac-Crab-Hex-Teacher-Play-v0`; use `Isaac-Crab-Hex-Teacher-v0` to match the training MDP (parkour terrain mix). For stage 1 only, use `Isaac-Crab-Hex-Flat-Walk-Play-v0`.
- Add `--headless` for no GUI (physics only).
- The script must be launched via `isaaclab.sh -p` (do not run `zero_agent.py` directly — you will get `Permission denied`).
- Passive stability is **not** the same as a trained policy: the robot only holds the configured default pose under gravity. A few tens of seconds upright is normal; long collapse means retune spawn or joint defaults in `crab_hex_scene_cfg.py`.

### 4.1a Crab verification scripts (headless)

Optional checks in `[scripts/](scripts/)`. Run from `krabby-research/parkour` via `isaaclab.sh -p` (same as `zero_agent.py`).


| Script                                                                     | What it does                                                                                                                                                                                                                                                                                                                                                                                        |
| -------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `[verify_crab_contact_physics.py](scripts/verify_crab_contact_physics.py)` | Spawns the flat-walk env, steps with zero actions, then prints a runtime audit: whether `.*_Footpad` bodies resolve on `contact_forces`, per-link masses (~**104 kg** total expected), foot contact flags, and friction/material notes. Writes JSON to `logs/rsl_rl/crab_hex_flat_walk/diagnostics/contact_physics_audit.json` by default (`--output` to override). Use after USD or spawn changes. |
| `[verify_crab_joint_drive.py](scripts/verify_crab_joint_drive.py)`         | Drives each of the **18** revolute joints one at a time (± action) and reports whether the joint moves (position delta, torque, velocity). Gravity off by default for a clean actuation test. Exits with code **1** if any joint fails. Use after actuator or joint limit changes in `crab_hex_scene_cfg.py`.                                                                                       |


```bash
cd "$KRABBY_ROOT/krabby-research/parkour"
SCRIPTS=parkour_tasks/parkour_tasks/crab_hexapod_task/scripts

"$KRABBY_ROOT/IsaacLab/isaaclab.sh" -p "$SCRIPTS/verify_crab_contact_physics.py" --headless

"$KRABBY_ROOT/IsaacLab/isaaclab.sh" -p "$SCRIPTS/verify_crab_joint_drive.py" --headless
```

### 4.2 Teacher training utilities (logs, TensorBoard, resume)

**Curriculum path:** use [§4.0](#40-training-commands-curriculum) with the correct `KRABBY_HEX_TEACHER_MODE` and bundled resume checkpoints. The example below is a **generic** long teacher run (default/full MDP) — not the recommended path until stage 4.

The teacher uses privileged observations (terrain, dynamics, etc.) and trains with `scripts/rsl_rl/train.py`.

**Example: 256 envs, 10 000 PPO iterations (full teacher MDP, ≈ 6 h on an RTX 5080–class GPU):**

```bash
cd "$KRABBY_ROOT/krabby-research/parkour"
"$KRABBY_ROOT/IsaacLab/isaaclab.sh" -p "$KRABBY_ROOT/krabby-research/parkour/scripts/rsl_rl/train.py" \
  --task Isaac-Crab-Hex-Teacher-v0 \
  --headless \
  --num_envs 256 \
  --seed 1 \
  --max_iterations 10000 
```

The script logs runs under:

```text
krabby-research/parkour/logs/rsl_rl/crab_hex_teacher/<TIMESTAMP>/
```

That is `**$KRABBY_ROOT/krabby-research/parkour/logs/rsl_rl/crab_hex_teacher/<TIMESTAMP>/**` on disk when you launch from `krabby-research/parkour` as above (the job also prints `Logging experiment in directory: ...` at startup).

Inside each timestamped folder you will see:

- `model_0.pt`, `model_100.pt`, …, `model_9900.pt`, `**model_9999.pt**` (checkpoints)
- `events.out.tfevents.*` (TensorBoard)
- `params/agent.yaml`, `params/env.yaml` (frozen configs)

**TensorBoard:** Run it with `**conda activate env_isaaclab`**. The default `(base)` Python often fails TensorBoard with `ModuleNotFoundError: No module named 'pkg_resources'`.

Point `--logdir` at the parent `**logs/rsl_rl`** directory that matches **how you trained** (same rule as checkpoints: relative to the shell’s current working directory):

```bash
conda activate env_isaaclab
# If you trained from krabby-research/parkour (recommended above):
tensorboard --logdir "$KRABBY_ROOT/krabby-research/parkour/logs/rsl_rl" --port 6006 --bind_all
# If you trained from IsaacLab instead:
# tensorboard --logdir "$KRABBY_ROOT/IsaacLab/logs/rsl_rl" --port 6006 --bind_all
```

Open **[http://localhost:6006/](http://localhost:6006/)** (or another `--port` if 6006 is in use). A single `--logdir` on `**logs/rsl_rl`** lists every experiment underneath (e.g. teacher and student runs). Scalars keep updating while training is running; quit TensorBoard with **Ctrl+C** in that terminal.

In **Scalars**, search for `**mean_reward`** / `**Train/`** and `**Episode_Reward/`** (per-term curves such as `reward_tracking_goal_vel`, `reward_collision`, matching the training log).

You can stop a long run early and still use the last `model_<iter>.pt` that was saved.  
To resume from a specific checkpoint (use the **same** `cd` as training so `--load_run` resolves correctly):

```bash
cd "$KRABBY_ROOT/krabby-research/parkour"
"$KRABBY_ROOT/IsaacLab/isaaclab.sh" -p "$KRABBY_ROOT/krabby-research/parkour/scripts/rsl_rl/train.py" \
  --task Isaac-Crab-Hex-Teacher-v0 \
  --headless \
  --num_envs 256 \
  --seed 1 \
  --resume \
  --load_run <TIMESTAMP_DIR_NAME> \
  --checkpoint model_9200.pt \
  --max_iterations 800
```

Here `max_iterations` means “run this many **more** PPO iterations starting from the loaded `iter`,” so `9200 + 800 = 10000`.

### 4.2b 2b2 teacher sweet-spot

**Goal:** a stable **2b2 teacher** for student distillation — robust obstacle walking with visible leg lift, not maximal raw speed.

**Play criteria:**

- Crosses gaps, hurdles, and steps with commitment (not freezing at edges).
- Some heading drift and occasional falls are OK.
- **Reject:** leg thrash then fall; standing still; hole stuck / pulling without lift.

**MDP (unchanged from 2b2):** action scale **0.24**, clip **±1**; **50/50** flat/parkour; terrain curriculum **0.20–0.70**; moderate domain randomization — not full 0–1 terrain or 0.25/±4.8.

**Rewards (`CrabHexStage2BPhase2RewardsCfg`):**


| Term                                                         | Weight                              |
| ------------------------------------------------------------ | ----------------------------------- |
| `reward_foot_clearance`                                      | **+2.0**                            |
| `reward_obstacle_clearance`                                  | **+1.8**                            |
| `reward_swing_vertical_vel`                                  | **+0.8**                            |
| `reward_recover_from_stall`                                  | **+0.2**                            |
| `penalty_swing_min_clearance`                                | **−0.4**                            |
| `reward_forward_progress_along_command`                      | **+0.25**                           |
| `reward_tracking_goal_vel`                                   | **+1.0**                            |
| `reward_tracking_yaw`                                        | **+0.3**                            |
| `penalty_low_forward_speed_when_commanded`                   | **−0.8**                            |
| `reward_collision`                                           | **−3.0**                            |
| `reward_feet_stumble` / `reward_feet_edge`                   | **−0.8** each                       |
| `reward_orientation` / `reward_lin_vel_z`                    | **−1.0** each                       |
| `reward_hip_pos`                                             | **−0.5**                            |
| `reward_ang_vel_xy`                                          | **−0.05**                           |
| `reward_action_rate`                                         | **−0.1**                            |
| `reward_dof_error`                                           | **−0.04**                           |
| `reward_torques` / `reward_dof_acc` / `reward_delta_torques` | **−1e-5** / **−2.5e-7** / **−1e-7** |


Bridge velocity-primary aux (`track_lin_vel_xy_exp`, flat speed, `reward_tracking_yaw_on_parkour`, etc.) are **zeroed** in 2b2.

**Training protocol:** resume **2b1** `6198` only; `max_iterations=10000`, `save_interval=100`. Play + TensorBoard at each saved checkpoint; **stop** at the first sweet-spot iter — later iters in the same log often regress (lift collapses, hole stuck).

**Sweet-spot gates** (TensorBoard + play; star gate must match visible lift):


| Metric                                                 | Target                                                               |
| ------------------------------------------------------ | -------------------------------------------------------------------- |
| `Episode_Reward/reward_obstacle_clearance`             | **> 0.15** (star gate)                                               |
| `Episode_Termination/crab_failure`                     | **< 20%**                                                            |
| `Train/mean_episode_length`                            | **≥ 750–800**                                                        |
| `Episode_Reward/reward_forward_progress_along_command` | **> 0.15–0.20** (ideal; bundled teacher is **~0.11** with good play) |
| `Metrics/base_parkour/current_goal_idx`                | **> 0.7–0.9**                                                        |
| Play confirms lift at holes/steps                      | **Required**                                                         |


**Bundled teacher:** `runs/2026-05-26_21-46-37/model_6300.pt` — provenance [Appendix F](#appendix-f--stage-2b2-teacher-ready-baseline--2026-05-26).

### 4.3 Play a bundled checkpoint

Use the one-liners below. Set `KRABBY_HEX_USD_PATH`, `KRABBY_HEX_SPAWN_Z=1.05`, and `PYTHONPATH`; teacher stages also set `KRABBY_HEX_TEACHER_MODE`.

```bash
export KRABBY_ROOT=/home/sanjay/Projects/krabby
conda activate env_isaaclab
RUNS_DIR="$KRABBY_ROOT/krabby-research/parkour/parkour_tasks/parkour_tasks/crab_hexapod_task/runs"
USD="$RUNS_DIR/2026-05-23_10-15-21/crab_simple_2026-05-23_10-15-21.usda"
export KRABBY_HEX_USD_PATH="$USD"
export KRABBY_HEX_SPAWN_Z=1.05
export PYTHONPATH="$KRABBY_ROOT/krabby-research/parkour/parkour_tasks:$KRABBY_ROOT/krabby-research/parkour:${PYTHONPATH:-}"
cd "$KRABBY_ROOT/krabby-research/parkour"
```


| Stage     | Task                                                                | One-liner checkpoint path           |
| --------- | ------------------------------------------------------------------- | ----------------------------------- |
| 1 flat    | `Isaac-Crab-Hex-Flat-Walk-Play-v0`                                  | `2026-05-23_10-15-21/model_6000.pt` |
| 2a bridge | `Isaac-Crab-Hex-Teacher-Play-v0` + `KRABBY_HEX_TEACHER_MODE=bridge` | `2026-05-25_22-26-06/model_6099.pt` |
| 2b1       | `Isaac-Crab-Hex-Teacher-Play-v0` + `KRABBY_HEX_TEACHER_MODE=2b1`    | `2026-05-25_23-57-58/model_6198.pt` |
| 2b2       | `Isaac-Crab-Hex-Teacher-Play-v0` + `KRABBY_HEX_TEACHER_MODE=2b2`    | `2026-05-26_21-46-37/model_6300.pt` |
| 3 student | `Isaac-Crab-Hex-Student-Play-v0`                                    | `2026-05-26_22-57-01/model_9800.pt` |


Flat walk uses `Isaac-Crab-Hex-Flat-Walk-Play-v0`; teacher stages use `Isaac-Crab-Hex-Teacher-Play-v0`; student uses `Isaac-Crab-Hex-Student-Play-v0` (2b2-mixed terrain). Optional **100% flat** diagnostic adds `export KRABBY_HEX_PLAY_FLAT=1` for student play (not the training distribution). For log-folder checkpoints, set `KRABBY_HEX_TEACHER_MODE` to match training and use `play.py`:

```bash
cd "$KRABBY_ROOT/krabby-research/parkour"
export KRABBY_HEX_TEACHER_MODE=2b2   # match checkpoint
"$KRABBY_ROOT/IsaacLab/isaaclab.sh" -p scripts/rsl_rl/play.py \
  --task Isaac-Crab-Hex-Teacher-Play-v0 \
  --num_envs 1 --real-time \
  --checkpoint "$KRABBY_ROOT/krabby-research/parkour/logs/rsl_rl/crab_hex_teacher/<TIMESTAMP>/model_XXXX.pt"
```

Use `Isaac-Crab-Hex-Teacher-v0` for the exact training MDP; `*-Play-v0` for follow-cam and debug vis (`KRABBY_HEX_PLAY_HARD=1` for harder play terrain).

### 4.4 Student distillation

Uses [§3.1](#31-teacher-vs-student) student MDP. **Prerequisite:** [Appendix F](#appendix-f--stage-2b2-teacher-ready-baseline--2026-05-26) `model_6300.pt`.

**Conceptual distillation loss (first version):**

```yaml
distill:
  enabled: true
  teacher_checkpoint: runs/2026-05-26_21-46-37/model_6300.pt
  loss_action_kl_weight: 0.5      # match teacher action distribution
  loss_value_mse_weight: 0.2      # optional value match
  loss_policy_mse_weight: 1.0     # keep RL losses
```

**Minimal student train** (terrain is 2b2-aligned in config; use **256** envs on a 16 GB GPU):

```bash
export KRABBY_ROOT=/home/sanjay/Projects/krabby
export KRABBY_HEX_SPAWN_Z=1.05
TEACHER_CKPT="$KRABBY_ROOT/krabby-research/parkour/parkour_tasks/parkour_tasks/crab_hexapod_task/runs/2026-05-26_21-46-37/model_6300.pt"

cd "$KRABBY_ROOT/krabby-research/parkour"
"$KRABBY_ROOT/IsaacLab/isaaclab.sh" -p "$KRABBY_ROOT/krabby-research/parkour/scripts/rsl_rl/train.py" \
  --task Isaac-Crab-Hex-Student-v0 \
  --headless \
  --num_envs 256 \
  --seed 1 \
  --checkpoint "$TEACHER_CKPT"
```

Student logs: `krabby-research/parkour/logs/rsl_rl/crab_hex_student/<TIMESTAMP>/` (same `--logdir` as teacher — [§4.2](#42-teacher-training-utilities-logs-tensorboard-resume)).

**Bundled student:** `runs/2026-05-26_22-57-01/model_9800.pt` — provenance [Appendix G](#appendix-g--stage-3-student-distillation--2026-05-26). Selected by play on 2b2-mixed student MDP (not last log iter).

**Play bundled student (recommended):**

```bash
export KRABBY_ROOT=/home/sanjay/Projects/krabby
conda activate env_isaaclab
RUNS_DIR="$KRABBY_ROOT/krabby-research/parkour/parkour_tasks/parkour_tasks/crab_hexapod_task/runs"
USD="$RUNS_DIR/2026-05-23_10-15-21/crab_simple_2026-05-23_10-15-21.usda"
export KRABBY_HEX_USD_PATH="$USD"
export KRABBY_HEX_SPAWN_Z=1.05
export PYTHONPATH="$KRABBY_ROOT/krabby-research/parkour/parkour_tasks:$KRABBY_ROOT/krabby-research/parkour:${PYTHONPATH:-}"
cd "$KRABBY_ROOT/krabby-research/parkour"
"$KRABBY_ROOT/IsaacLab/isaaclab.sh" -p scripts/rsl_rl/play.py \
  --task Isaac-Crab-Hex-Student-Play-v0 \
  --num_envs 1 \
  --real-time \
  --checkpoint "$RUNS_DIR/2026-05-26_22-57-01/model_9800.pt"
```

Ad-hoc log checkpoints:

```bash
cd "$KRABBY_ROOT/krabby-research/parkour"
"$KRABBY_ROOT/IsaacLab/isaaclab.sh" -p scripts/rsl_rl/play.py \
  --task Isaac-Crab-Hex-Student-Play-v0 \
  --num_envs 1 \
  --real-time \
  --checkpoint "$KRABBY_ROOT/krabby-research/parkour/logs/rsl_rl/crab_hex_student/<TIMESTAMP>/model_XXXX.pt"
```

Use `Isaac-Crab-Hex-Student-v0` for the exact training MDP; `*-Play-v0` uses `CRAB_HEX_VIEWER` follow-cam and command/parkour debug vis (same as teacher play). Per-100-iter metrics CSV: `logs/rsl_rl/crab_hex_student/student_metrics_per100.csv`.

**Play bundled student with Pro Controller (gamepad velocity teleop):**

Connect a gamepad to the machine running Isaac Sim (USB or Bluetooth). Pairing on Linux: [controller/scripts/jetson/CONNECT_PRO_CONTROLLER.md](../../../../controller/scripts/jetson/CONNECT_PRO_CONTROLLER.md). List devices: `python -m controller.input --list`.

```bash
export KRABBY_ROOT=/home/sanjay/Projects/krabby
conda activate env_isaaclab
RUNS_DIR="$KRABBY_ROOT/krabby-research/parkour/parkour_tasks/parkour_tasks/crab_hexapod_task/runs"
USD="$RUNS_DIR/2026-05-23_10-15-21/crab_simple_2026-05-23_10-15-21.usda"
export KRABBY_HEX_USD_PATH="$USD"
export KRABBY_HEX_SPAWN_Z=1.05
export PYTHONPATH="$KRABBY_ROOT/krabby-research/parkour/parkour_tasks:$KRABBY_ROOT/krabby-research/parkour:${PYTHONPATH:-}"
cd "$KRABBY_ROOT/krabby-research/parkour"
"$KRABBY_ROOT/IsaacLab/isaaclab.sh" -p \
  parkour_tasks/parkour_tasks/crab_hexapod_task/scripts/demo_crab_hex_student.py \
  --task Isaac-Crab-Hex-Student-Play-v0 \
  --num_envs 1 \
  --real-time \
  --checkpoint "$RUNS_DIR/2026-05-26_22-57-01/model_9800.pt"
```

**Controls:** left stick **Y** = forward speed **(0.45–0.85 m/s)**; right stick **X** = heading (parkour turns the robot). Input uses `**krabby-research/controller`** (pygame SDL2 Pro Controller mapping), not Isaac’s Carb gamepad. Install once: `pip install -e "$KRABBY_ROOT/krabby-research/controller"`. Verify pad: `python -m controller.input --list`. For **manual joint teleop** without the policy, use `krabby-uno-sim --hex` ([isaacsim_demo_runbook.md](../../../../controller/scripts/isaac/isaacsim_demo_runbook.md)).

---

## 5. Reference: existing quadruped (Unitree Go2) rewards and training

The crab hexapod task is built by following the conventions of the **extreme parkour Unitree Go2** task that ships with Isaac Lab.

- **Gym registrations (Go2 teacher / student / eval / play):**  
`IsaacLab/Isaaclab_Parkour/parkour_tasks/parkour_tasks/extreme_parkour_task/config/go2/__init__.py`  
(e.g. `Isaac-Extreme-Parkour-Teacher-Unitree-Go2-v0`, `Isaac-Extreme-Parkour-Teacher-Unitree-Go2-Play-v0`, etc.)
- **Go2 MDP / rewards / actions:**  
`IsaacLab/Isaaclab_Parkour/parkour_tasks/parkour_tasks/extreme_parkour_task/config/go2/agents/parkour_mdp_cfg.py`  
which in turn uses the same reward functions in  
`krabby-research/parkour/parkour_isaaclab/envs/mdp/rewards.py`.

### 5.1 Go2 teacher training (extreme parkour)

From inside the Isaac Lab checkout:

```bash
cd "$KRABBY_ROOT/IsaacLab"
conda activate env_isaaclab

./isaaclab.sh -p ./Isaaclab_Parkour/scripts/rsl_rl/train.py \
  --task Isaac-Extreme-Parkour-Teacher-Unitree-Go2-v0 \
  --headless \
  --num_envs 4096 \
  --seed 1
```

This writes checkpoints under:

```text
Isaaclab_Parkour/logs/rsl_rl/unitree_go2_parkour_teacher/<TIMESTAMP>/
```

### 5.2 Go2 play (extreme parkour teacher play env)

You can visualize a trained Go2 teacher policy on parkour terrain using the Go2 **PLAY** env:

```bash
cd "$KRABBY_ROOT/IsaacLab"
./isaaclab.sh -p ./Isaaclab_Parkour/scripts/rsl_rl/play.py \
  --task Isaac-Extreme-Parkour-Teacher-Unitree-Go2-Play-v0 \
  --num_envs 1 \
  --real-time \
  --checkpoint ./Isaaclab_Parkour/logs/rsl_rl/unitree_go2_parkour_teacher/<TIMESTAMP>/model_XXXX.pt
```

The hexapod task mirrors this layout (Gym registrations, env cfgs, reward wiring, and train/play commands), so anyone familiar with the Go2 extreme parkour examples should find the crab hexapod task immediately recognizable.  
Training uses `**crab_simple.usda**` only; set `**KRABBY_HEX_USD_PATH**` only if your checkout or container layout is non-standard. RSL-RL checkpoints for the commands in **§4** are kept under `**krabby-research/parkour/logs/rsl_rl/`** by running from that directory as documented there.

---

## Appendix

Baseline **provenance and metrics** only — stage differences: [§2](#2-how-stages-differ); train/play: [§4.0](#40-training-commands-curriculum) / [§4.3](#43-play-a-bundled-checkpoint).


| Appendix                                                       | Stage                                                                    | Run dir               | Ckpt   |
| -------------------------------------------------------------- | ------------------------------------------------------------------------ | --------------------- | ------ |
| [A](#appendix-a--general-lessons--first-successful-run)        | Lessons (pre-curriculum)                                                 | —                     | —      |
| [B](#appendix-b--stage-1-flat-walk--2026-05-19-legacy)         | 1 flat (legacy)                                                          | `2026-05-19_12-06-10` | `4000` |
| [C](#appendix-c--stage-1-flat-walk--2026-05-23-baseline)       | 1 flat (current)                                                         | `2026-05-23_10-15-21` | `6000` |
| [D](#appendix-d--stage-2a-teacher-bridge--2026-05-25-baseline) | 2a bridge                                                                | `2026-05-25_22-26-06` | `6099` |
| [E](#appendix-e--stage-2b1-hybrid-walk--2026-05-25-baseline)   | 2b1                                                                      | `2026-05-25_23-57-58` | `6198` |
| [F](#appendix-f--stage-2b2-teacher-ready-baseline--2026-05-26) | 2b2 teacher                                                              | `2026-05-26_21-46-37` | `6300` |
| [G](#appendix-g--stage-3-student-distillation--2026-05-26)     | 3 student                                                                | `2026-05-26_22-57-01` | `9800` |
| —                                                              | [4 `full` (TODO)](#footnote--curriculum-staging-and-future-full-teacher) | —                     | —      |


### Appendix A — General lessons — first successful run

- **Focus on USD, not reward tuning to start:** Removed overlapping reward experiments until `crab_simple.usda` and spawn were credible. Reward tuning can come incrementally after the asset and default stance are trustworthy.
- **Explicit masses in USD:** Per-link weights (~**104 kg** total for the current `crab_simple.usda`; earlier ~**25 kg** baseline also in logs) instead of relying on PhysX auto-mass. Retrain when additional payload is modeled.
- **Foot rubber at the feet:** Separate `*_Footpad` colliders with `FootRubber` for ground contact (not full-shank tibia collision).
- **Stable stance:** Body–hip yaw splay **±0.6** on front and rear legs; spawn `z` **1.05** m (`KRABBY_HEX_SPAWN_Z`).
- **Simpler flat-walk reward weights:** Small `CrabHexFlatWalkRewardsCfg` set for easier experimentation.
- **Velocity in observations:** Base linear velocity (`root_lin_vel_xy`) included in proprioceptive observations.

### Appendix B — Stage 1 flat walk — 2026-05-19 legacy

This commit captures the best flat-walk baseline found during the 2026-05-19 tuning pass and documents why the current flat-walk settings were chosen.

The checked-in baseline artifacts are stored under:

```text
parkour_tasks/parkour_tasks/crab_hexapod_task/runs/2026-05-19_12-06-10/
```

It contains:

- `crab_simple_2026-05-19_12-06-10.usda` - the USD snapshot used for this baseline.
- `model_4000.pt` - the baseline flat-walk policy checkpoint.
- `README.md` - short provenance and frozen flat-walk settings.

*The artifacts in this folder may be deleted in a future cleanup: later runs (especially [Appendix C](#appendix-c--stage-1-flat-walk--2026-05-23-baseline)) supersede this baseline for training and play. Keep this bundle only if you want to compare run-to-run improvements against 2026-05-19.*

Key changes and why they were made:

- **USD and checkpoint bundle:** The known-good `crab_simple.usda` snapshot and `model_4000.pt` are stored under `runs/2026-05-19_12-06-10/` so the play baseline is reproducible even if later assets or training logs change.
- **Explicit USD override in play:** The one-liner commands set `KRABBY_HEX_USD_PATH` so the bundled checkpoint plays against the bundled USD, not whichever asset happens to be current in `assets/`.
- **Flat-walk command range:** `lin_vel_x = (0.25, 0.60)` keeps the speed request high enough for visible progress while avoiding the earlier overly aggressive forward shortcut.
- **Forward progress reward:** `reward_forward_progress_along_command = 0.50` was selected as the best balance so far. Larger values encouraged faster motion but began to reintroduce north/south drift; smaller values made the gait too conservative.
- **Velocity tracking kept primary:** `track_lin_vel_xy_exp = 1.0` stays active so the policy is rewarded for matching commanded body-frame planar velocity instead of just moving roughly forward.
- **Lateral drift penalty:** `penalty_lin_vel_y = -3.0` keeps body-frame sideways velocity small without over-constraining gait exploration.
- **Air-time reward:** `reward_feet_air_time_positive = 0.25` nudges the policy toward clearer swing/step behavior rather than an all-feet shuffling gait.
- **Collision and feet-slide terms disabled for flat walk:** `reward_collision = 0.0` and `feet_slide = 0.0` remain available but are not part of this baseline because the drift/speed tradeoff was better controlled by velocity, progress, and air-time terms.
- **Stance defaults:** The current stance keeps body-hip yaw splay at **±0.6**, hip-femur at **0.30**, and mirrored knee defaults (left **−0.07**, right **+0.10**) to balance the passive zero-action stance without removing body-hip splay.

Run: [§4.3](#43-play-a-bundled-checkpoint) with `2026-05-19_12-06-10` USD + `model_4000.pt`.

Metrics @ `4000`: `track_lin_vel_xy_exp` ~**0.87**; `crab_failure` < **1%**.

### Appendix C — Stage 1 flat walk — 2026-05-23 baseline

**Log:** `logs/rsl_rl/crab_hex_flat_walk/2026-05-23_10-15-21/`. **Artifacts:** `runs/2026-05-23_10-15-21/` (`model_6000.pt`, paired USD, README).

**Tuning vs [B](#appendix-b--stage-1-flat-walk--2026-05-19-legacy):** `lin_vel_x` **(0.30, 0.65)**; stronger velocity tracking (**1.25**); forward progress **0.60**; gait helpers (`penalty_tibia_deviation_in_stance`, idle-foot / excess-contact penalties); ML/MR hip splay **±0.25**. Reward detail: [Stage 1](#stage-1--flat-walk) in [§2](#2-how-stages-differ).

**Metrics @ `6000`:** `track_lin_vel_xy_exp` ≈ **1.00**; `fwd_progress` ≈ **0.28**; `crab_failure` ≈ **6%**; `ep_len` ≈ **985**.

**Play:**

```bash
export KRABBY_ROOT=/home/sanjay/Projects/krabby
conda activate env_isaaclab
RUNS_DIR="$KRABBY_ROOT/krabby-research/parkour/parkour_tasks/parkour_tasks/crab_hexapod_task/runs"
export KRABBY_HEX_USD_PATH="$RUNS_DIR/2026-05-23_10-15-21/crab_simple_2026-05-23_10-15-21.usda"
export KRABBY_HEX_SPAWN_Z=1.05
export PYTHONPATH="$KRABBY_ROOT/krabby-research/parkour/parkour_tasks:$KRABBY_ROOT/krabby-research/parkour:${PYTHONPATH:-}"
cd "$KRABBY_ROOT/krabby-research/parkour"
"$KRABBY_ROOT/IsaacLab/isaaclab.sh" -p scripts/rsl_rl/play.py \
  --task Isaac-Crab-Hex-Flat-Walk-Play-v0 \
  --num_envs 1 --real-time \
  --checkpoint "$RUNS_DIR/2026-05-23_10-15-21/model_6000.pt"
```

### Appendix D — Stage 2a teacher bridge — 2026-05-25 baseline

**Log:** `logs/rsl_rl/crab_hex_teacher/2026-05-25_22-26-06/`. **Artifacts:** `runs/2026-05-25_22-26-06/model_6099.pt`. Resume flat `6000` → **100** iters.

**Play @6099:** stable forward walk on flat + light tiles; some heading drift OK.

**Metrics @ `6099`:** `track_lin_vel_xy` ~**1.66**; `crab_failure` ~**7%**; `ep_len` ~**920**; `error_vel_yaw` ~**1.85**.

**Play:**

```bash
export KRABBY_ROOT=/home/sanjay/Projects/krabby
conda activate env_isaaclab
RUNS_DIR="$KRABBY_ROOT/krabby-research/parkour/parkour_tasks/parkour_tasks/crab_hexapod_task/runs"
USD="$RUNS_DIR/2026-05-23_10-15-21/crab_simple_2026-05-23_10-15-21.usda"
export KRABBY_HEX_USD_PATH="$USD"
export KRABBY_HEX_SPAWN_Z=1.05
export KRABBY_HEX_TEACHER_MODE=bridge
export PYTHONPATH="$KRABBY_ROOT/krabby-research/parkour/parkour_tasks:$KRABBY_ROOT/krabby-research/parkour:${PYTHONPATH:-}"
cd "$KRABBY_ROOT/krabby-research/parkour"
"$KRABBY_ROOT/IsaacLab/isaaclab.sh" -p scripts/rsl_rl/play.py \
  --task Isaac-Crab-Hex-Teacher-Play-v0 \
  --num_envs 1 --real-time \
  --checkpoint "$RUNS_DIR/2026-05-25_22-26-06/model_6099.pt"
```

### Appendix E — Stage 2b1 hybrid walk — 2026-05-25 baseline

**Log:** `logs/rsl_rl/crab_hex_teacher/2026-05-25_23-57-58/`. **Artifacts:** `runs/2026-05-25_23-57-58/model_6198.pt`. Resume bridge `6099` → **100** iters. Same terrain as 2a; weak goal/yaw on ([Stage 2b1](#stage-2b1--hybrid-walk)).

**Play @6198:** good on flat/light tiles; steps/hurdles still hard (expected).

**Metrics @ `6198`:** `crab_failure` ~**4.8%**; `ep_len` ~**943**; `mean_reward` ~**2.68**.

**Play:**

```bash
export KRABBY_ROOT=/home/sanjay/Projects/krabby
conda activate env_isaaclab
RUNS_DIR="$KRABBY_ROOT/krabby-research/parkour/parkour_tasks/parkour_tasks/crab_hexapod_task/runs"
USD="$RUNS_DIR/2026-05-23_10-15-21/crab_simple_2026-05-23_10-15-21.usda"
export KRABBY_HEX_USD_PATH="$USD"
export KRABBY_HEX_SPAWN_Z=1.05
export KRABBY_HEX_TEACHER_MODE=2b1
export PYTHONPATH="$KRABBY_ROOT/krabby-research/parkour/parkour_tasks:$KRABBY_ROOT/krabby-research/parkour:${PYTHONPATH:-}"
cd "$KRABBY_ROOT/krabby-research/parkour"
"$KRABBY_ROOT/IsaacLab/isaaclab.sh" -p scripts/rsl_rl/play.py \
  --task Isaac-Crab-Hex-Teacher-Play-v0 \
  --num_envs 1 --real-time \
  --checkpoint "$RUNS_DIR/2026-05-25_23-57-58/model_6198.pt"
```

### Appendix F — Stage 2b2 teacher-ready baseline — 2026-05-26

**Log:** `logs/rsl_rl/crab_hex_teacher/2026-05-26_21-46-37/`. **Artifacts:** `runs/2026-05-26_21-46-37/model_6300.pt`. Resume 2b1 `6198` → **~106** iters; selected `**6300`** after play (`6400/6500` kept for reference).

**Why `6300`:** best play after the additional lift-focused 2b2 delta (`reward_swing_vertical_vel` **0.8**, `penalty_swing_min_clearance` **−0.4**, `reward_recover_from_stall` **0.2**); visibly lifts out of holes better while preserving usable forward motion. Full reward stack, gates, and training protocol: [§4.2b](#42b-2b2-teacher-sweet-spot).

**Metrics @ `6300`:** `crab_failure` ~**23.6%**; `ep_len` ~**731**; `obstacle_clearance` ~**0.189**; `foot_clearance` ~**0.086**; `goal_idx` ~**0.94**; `fwd_progress` ~**0.059**.

**Play @6300:** steady forward walk; lifts legs from holes; some stumble. **Use for student distillation** ([§4.4](#44-student-distillation)).

**Play:**

```bash
export KRABBY_ROOT=/home/sanjay/Projects/krabby
conda activate env_isaaclab
RUNS_DIR="$KRABBY_ROOT/krabby-research/parkour/parkour_tasks/parkour_tasks/crab_hexapod_task/runs"
USD="$RUNS_DIR/2026-05-23_10-15-21/crab_simple_2026-05-23_10-15-21.usda"
export KRABBY_HEX_USD_PATH="$USD"
export KRABBY_HEX_SPAWN_Z=1.05
export KRABBY_HEX_TEACHER_MODE=2b2
export PYTHONPATH="$KRABBY_ROOT/krabby-research/parkour/parkour_tasks:$KRABBY_ROOT/krabby-research/parkour:${PYTHONPATH:-}"
cd "$KRABBY_ROOT/krabby-research/parkour"
"$KRABBY_ROOT/IsaacLab/isaaclab.sh" -p scripts/rsl_rl/play.py \
  --task Isaac-Crab-Hex-Teacher-Play-v0 \
  --num_envs 1 --real-time \
  --checkpoint "$RUNS_DIR/2026-05-26_21-46-37/model_6300.pt"
```

*Superseded: `runs/2026-05-26_11-30-18/` (2b2-v2 rewards, no foot/swing-vz terms) — reference only.*

### Appendix G — Stage 3 student distillation — 2026-05-26

**Log:** `logs/rsl_rl/crab_hex_student/2026-05-26_22-57-01/`. **Artifacts:** `runs/2026-05-26_22-57-01/model_9800.pt`. Train from teacher [Appendix F](#appendix-f--stage-2b2-teacher-ready-baseline--2026-05-26) `model_6300.pt`; iter counter starts at **6300**; **256** envs recommended on ~16 GB GPU.

**Why `9800`:** among late training checkpoints, `**9800*`* had the best combined TensorBoard tradeoff — highest late-run `**ep_len**` (~~**758**), strong `**goal_idx`** (~~**0.50**), and acceptable `**crab_failure`** (~**28.7%**). Later iters (e.g. **12000**, **12400**) regressed on `**goal_idx`** and/or episode length; play on 2b2-mixed terrain matched that ranking (walking + hurdle crossing).

**Metrics @ `9800`:** `crab_failure` ~**28.7%**; `ep_len` ~**758**; `goal_idx` ~**0.50**; `depth_actor_loss` ~**1.85**.

**Play @9800:** use **2b2-mixed** student terrain via the direct `play.py` command below.

**Play:**

```bash
export KRABBY_ROOT=/home/sanjay/Projects/krabby
conda activate env_isaaclab
RUNS_DIR="$KRABBY_ROOT/krabby-research/parkour/parkour_tasks/parkour_tasks/crab_hexapod_task/runs"
USD="$RUNS_DIR/2026-05-23_10-15-21/crab_simple_2026-05-23_10-15-21.usda"
export KRABBY_HEX_USD_PATH="$USD"
export KRABBY_HEX_SPAWN_Z=1.05
export PYTHONPATH="$KRABBY_ROOT/krabby-research/parkour/parkour_tasks:$KRABBY_ROOT/krabby-research/parkour:${PYTHONPATH:-}"
cd "$KRABBY_ROOT/krabby-research/parkour"
"$KRABBY_ROOT/IsaacLab/isaaclab.sh" -p scripts/rsl_rl/play.py \
  --task Isaac-Crab-Hex-Student-Play-v0 \
  --num_envs 1 --real-time \
  --checkpoint "$RUNS_DIR/2026-05-26_22-57-01/model_9800.pt"
```

---

### Footnote — curriculum staging and future full teacher

**2b1 is the most optional stage** (same terrain as 2a; reward change only). Skipping stages often caused thrashing or no lift — see [§2](#2-how-stages-differ). **Minimal redo:** Flat → **2a** → **2b2** (try skipping **2b1**; add back if 6099→2b2 regresses). Checkpoints **6099**, **6198**, **6300** are **resume anchors** when a 2b2 reward rebalance goes wrong.

#### TODO — Stage 4: full Go2-style teacher (`full`)

**After** validating bundled student [Appendix G](#appendix-g--stage-3-student-distillation--2026-05-26). Resume **2b2 teacher** `6300` ([Appendix F](#appendix-f--stage-2b2-teacher-ready-baseline--2026-05-26)), not student `9800`; unset `KRABBY_HEX_TEACHER_MODE` or `=full`; actions **0.25** / ±4.8; `CrabHexRewardsCfg`; full terrain diff **0–1**; PPO LR **2e-4**. Details: [Stage 4](#stage-4--full-parkour-todo) and `crab_hex_env_cfg.py`.

---

### Large bundled artifacts (checkpoints & USD)

The `runs/` appendices include **large binary files** (PyTorch checkpoints and paired `crab_simple` USD snapshots). Teacher checkpoints are typically **~10–12 MB** each; the student baseline `model_9800.pt` is **~48 MB** because it also stores the **depth encoder** and **depth actor** in addition to the teacher policy copy and optimizer state. If storing these in GitHub becomes a problem, keep the README and one-liner play commands in the repo and host the weights elsewhere (e.g. object storage, Git LFS, or copies under `logs/rsl_rl/` on your machine). Document the download path in the appendix `runs/<RUN_DIR>/README.md` and point the one-liners at your local copy.