# Crab hexapod task (`crab_hexapod_task`)

This package adds a **Krabby hexapod** parkour task on top of Isaac Lab’s extreme parkour stack.  
The goal of this README is that anyone can clone the repo, create a Python env similar to yours, and **train + play** the hexapod policy.

The examples below assume:

- `**$KRABBY_ROOT=/home/sanjay/Projects/krabby`**
- `**krabby-research**` lives at `**$KRABBY_ROOT/krabby-research**`
- **Isaac Lab** lives at `**$KRABBY_ROOT/IsaacLab`**
- Your Isaac Lab conda env is called `**env_isaaclab**`

Adjust paths and the conda env name if your layout is different.

---

## 1. Environment setup (once per machine)

All commands in this README assume:

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

**Spawn height:** The USD root `krabby` is offset **+1 m** in the file; `[_crab_simple_robot_cfg()](config/crab_hex/crab_hex_scene_cfg.py)` sets articulation spawn `z` from `KRABBY_HEX_SPAWN_Z` (default **`1.05`** m). Use the same value for train, play, and stance checks. If the robot **floats then slams**, **lower** slightly; if **hips scrape** or the root **interpenetrates**, **raise** in ~**0.02** m steps on flat ground.

**Default joint pose (rad):** body–hip yaw splay **±0.6** on front/rear legs (middle legs **0**); `Hip_Femur` **0.30**; `Femur_Tibia` left legs **−0.07**, right legs **+0.10**. Tune in `crab_hex_scene_cfg.py` if the passive stance is wrong.

---

## 2. Where hexapod configs live (scene, rewards, actions)

- **Gym registrations (flat walk + teacher + student):**  
`parkour_tasks/crab_hexapod_task/config/crab_hex/__init__.py`  
  - `Isaac-Crab-Hex-Flat-Walk-v0` → `CrabHexFlatWalkEnvCfg` (stage-1 flat walk pretrain)  
  - `Isaac-Crab-Hex-Flat-Walk-Play-v0` → `CrabHexFlatWalkEnvCfgPLAY` (viewer/debug for flat walk)  
  - `Isaac-Crab-Hex-Teacher-v0` → `CrabHexTeacherEnvCfg`  
  - `Isaac-Crab-Hex-Teacher-Play-v0` → `CrabHexTeacherEnvCfgPLAY` (viewer/debug-play variant)  
  - `Isaac-Crab-Hex-Student-v0` → `CrabHexStudentEnvCfg`
- **Scene / robot / sensors:**  
`parkour_tasks/crab_hexapod_task/config/crab_hex/crab_hex_scene_cfg.py`  
  - Loads `**crab_simple.usda`** by default (or `KRABBY_HEX_USD_PATH` if set); USD default prim `krabby` composes under `**{ENV_REGEX_NS}/Robot**`. Base **rigid** link is `**body`** under `**chassis**` (paths may appear as `.../Robot/krabby/chassis/body` or flattened under `Robot` depending on composition).
  - `**contact_forces**` uses `**ParkourHexContactSensorCfg**` (see `crab_hexapod_task/sensors/parkour_hex_contact_sensor.py`) with `**prim_path="{ENV_REGEX_NS}/Robot/.*"**` so chassis and all leg links with `PhysxContactReportAPI` are aggregated (Go2 uses stock `**ContactSensorCfg**` on `**Robot/.***`).
  - Sets PhysX articulation solver (16 / 4 pos/vel iterations), `**enabled_self_collisions=True**` on the articulation root (same idea as Go2 parkour scene). If large-`num_envs` training FPS drops badly, you can set `**enabled_self_collisions=False**` in `_crab_simple_robot_cfg()` as a last resort. Reset pose, 18-DOF revolute actuators, height scanner and (student) depth camera on `**{ENV_REGEX_NS}/Robot/chassis/body**`
- **Env configuration (teacher + student MDP):**  
`parkour_tasks/crab_hexapod_task/config/crab_hex/crab_hex_env_cfg.py`  
  - **Defaults match Go2** `[UnitreeGo2TeacherParkourEnvCfg](../extreme_parkour_task/config/go2/parkour_teacher_cfg.py)` / `[UnitreeGo2StudentParkourEnvCfg](../extreme_parkour_task/config/go2/parkour_student_cfg.py)`: same `CommandsCfg`, `sim.dt`, domain randomization (push, friction, mass/COM noise), sensor periods, etc.  
  - **Crab teacher-specific:** after `super().__post_init__()`, `CrabHexTeacherEnvCfg` sets `**sim.physx.enable_external_forces_every_iteration = True`** (TGS velocity stability), tightens `**commands.base_velocity.ranges.lin_vel_x**` to `**(0.45, 0.85)**` (Go2 default is `(0.3, 0.8)`), and wires event terms that target the chassis to `**body**` instead of Go2’s `**base**` (`SceneEntityCfg("robot", body_names="body")` for external wrench, mass, COM).  
  - **Env counts:** teacher `**num_envs=6144`**, student `**num_envs=192**` (same as Go2 defaults).  
  - **Optional easier train terrain:** set `**export KRABBY_HEX_TRAIN_EASY=1`** before `train.py` to use a lower difficulty band and **50%** `parkour_flat` (same mix idea as play-easy); teacher does **not** enable this by default.  
  - `CrabHexTeacherEnvCfgPLAY` is the **play/eval** variant:
    - Viewer follow-cam (`CRAB_HEX_VIEWER` in `crab_hex_env_cfg.py` — front 3/4, tracks `robot` root; same on `CrabHexTeacherEnvCfg`)  
    - Longer episodes (~60 s)  
    - Parkour + base velocity debug visualization  
    - **Default terrain:** easier / flat-heavy mix (difficulty **0.15–0.55**, **50%** `parkour_flat`) unless `**export KRABBY_HEX_PLAY_HARD=1`**, which restores a harder band (**0.7–1.0**, no flat). You can still force easy with `**export KRABBY_HEX_PLAY_EASY=1`**.
- **Rewards and actions for the hexapod:**  
`parkour_tasks/crab_hexapod_task/config/crab_hex/agents/parkour_mdp_cfg.py`
  - `CrabHexRewardsCfg` matches **Go2 teacher** weights for the same terms and uses the scene `**contact_forces`** sensor (same pattern as Go2) for collision / feet rewards. Low-level math lives in `parkour_isaaclab/envs/mdp/rewards.py`.
    - **Contacts / feet:** `reward_collision` (**-6**) on `body` / `.*_Hip` / `.*_Femur`; `reward_feet_edge` / `reward_feet_stumble` (**-1** each) on `.*_Footpad` via `contact_forces`.
    - **Regularization (Go2 parity):** `reward_torques` (**-1e-5**), `reward_dof_error` (**-0.04**), `reward_hip_pos` (**-0.5**), `reward_ang_vel_xy` (**-0.05**), `reward_action_rate` (**-0.1**), `reward_dof_acc` (**-2.5e-7**), `reward_delta_torques` (**-1e-7**).
    - **Base motion:** `reward_lin_vel_z` (**-1**), `reward_orientation` (**-1**).
    - **Task:** `reward_tracking_goal_vel` (**+2.25**), `reward_tracking_yaw` (**+0.5**).
  - **Observations:** proprioceptive history includes **one contact flag per footpad** (`.*_Footpad`, six bodies for crab vs four feet on Go2). Observation width is still **six** foot channels; only the resolved body names changed from `*_Tibia` to `*_Footpad`.
  - **Terminations:** `CrabHexTerminationsCfg` — `total_terminates` (parkour timeout / goal / fall) plus `**crab_failure`** (tilt + hip ground contact); see `parkour_mdp_cfg.py` for thresholds.
  - **Actions:** shared Go2 `[ActionsCfg](../extreme_parkour_task/config/go2/parkour_mdp_cfg.py)` — delayed joint position with `joint_names=[".*"]`, scale **0.25**, clip **±4.8**; teacher vs student delay/history comes from each env’s `__post_init__` like Go2.
  - **Student rewards:** `[CrabHexStudentRewardsCfg](config/crab_hex/agents/parkour_mdp_cfg.py)` follows Go2 `[StudentRewardsCfg](../extreme_parkour_task/config/go2/parkour_mdp_cfg.py)`: a single `**reward_collision`** term with **weight 0**, using hex bodies on `contact_forces`.
  - **Flat-walk rewards (stage 1):** `[CrabHexFlatWalkRewardsCfg](config/crab_hex/agents/parkour_mdp_cfg.py)` — walk before parkour:
    - **Command tracking:** `track_lin_vel_xy_exp` (**+1.0**), `track_ang_vel_z_exp` (**+1.0**), `reward_forward_progress_along_command` (**+0.50**, `max_speed_scale` **1.65**).
    - **Stability:** `reward_orientation` (**-0.7**), `reward_lin_vel_z` (**-0.15**); `reward_ang_vel_xy`, `reward_dof_error` at **0**.
    - **Gait:** `reward_feet_air_time_positive` on `.*_Footpad` (**+0.25**, threshold **0.05** s).
    - **Collision:** `reward_collision` weight **0** (hip/femur/chassis terms available but disabled for flat walk).
  - **Flat-walk env:** `CrabHexFlatWalkEnvCfg` — **100%** `parkour_flat`, difficulty **0.1–0.25**, terrain curriculum **off**, `lin_vel_x = (0.25, 0.60)`, straight heading, reduced domain randomization (no push / mass / COM noise), `crab_failure` with `limit_angle` **0.5** rad and hip contact **500 N**.

The **low-level** math for each reward term (errors, masks, exponents, etc.) is implemented in  
`parkour_isaaclab/envs/mdp/rewards.py`; `CrabHexRewardsCfg` just chooses which ones to use, with what weights and bodies/joints.

### Design notes (reward intuition)

- `reward_collision` (**-6.0**): penalty for undesired contacts on chassis / hips / femurs (not footpads).
- `reward_feet_edge` / `reward_feet_stumble` (**-1.0** each): edge and stumble shaping on **footpad** contacts (`.*_Footpad`).
- Torque / DOF / action / acceleration terms: discourage jitter and violent motions (`reward_dof_error` pulls toward **articulation default joint pose**, consistent with spawn defaults).
- `reward_lin_vel_z` / `reward_orientation`: penalize bounce and tilt (terrain-dependent scaling matches Go2).
- `reward_hip_pos` (**-0.5**): hip yaw regularization only (joint subset).
- `reward_tracking_goal_vel` (**+2.25**) / `reward_tracking_yaw` (**+0.5**): main parkour progress and heading alignment (goal-vel reward clamps / stabilizes small commanded speeds in `rewards.py`).

### Flat-walk training metrics (what “good” looks like)

On the current USD + spawn (**1.05** m), a working flat-walk policy often shows:

- **`Episode_Reward/reward_forward_progress_along_command`** ≈ **0.16–0.22** on the current best flat-walk settings — robot moves along the commanded heading without the earlier high-progress drift shortcut.
- **`Train/mean_reward`** rising to **~10+** with **long episode length** (~800–1000 steps).
- **`Episode_Termination/crab_failure`** ≈ **3–5%**.
- **`Episode_Reward/track_lin_vel_xy_exp`** can stay **near zero** early on—the policy may satisfy `reward_forward_progress` without tightly matching commanded speed. Watch this term if you need precise velocity tracking later.
- **`Metrics/base_velocity/error_vel_xy`** is a **cumulative** tracking error over the command resampling window; high values with visible walking are common until speed tracking improves.

### RSL-RL runner factory

**Committed:** `parkour/scripts/rsl_rl/runner_factory.py`  
`train.py`, `play.py`, and `evaluation.py` call `agent_cfg_to_train_dict()` and `make_on_policy_runner()` instead of `agent_cfg.to_dict()` directly. That fixes corrupted scalars from multi-inherit `configclass` `to_dict()` (e.g. `num_steps_per_env` becoming an `obs_groups` dict).

**Crab runner (committed):**

| File | Purpose |
|------|---------|
| `scripts/rsl_rl/crab_on_policy_runner.py` | `OnPolicyRunnerCrabHex` → loads `CrabHexActorCriticRMA` |
| `scripts/rsl_rl/modules/crab_actor_critic_with_encoder.py` | RMA policy with **clamped** Gaussian action std |

Go2 uses stock `OnPolicyRunnerWithExtractor` / `ActorCriticRMA` via the same factory.

**Local only (gitignored):** `crab_hexapod_task/tempscripts/` — optional diagnostics (`audit_crab_joint_drives.py`, `verify_crab_simple_usda.py`, `diagnose_obs_action_alignment.py`, `diagnose_forward_rollout.py`).

**Crab routing (`make_on_policy_runner`)** — uses `OnPolicyRunnerCrabHex` when any of:

| Condition | Example |
|-----------|---------|
| `runner_class_name == "OnPolicyRunnerCrabHex"` | `crab_hex_rl_cfg.py` |
| `policy.class_name == "CrabHexActorCriticRMA"` | Flat-walk / teacher / student |
| `estimator.num_prop == 75` | `CrabHexParkourObservations` (Go2: **53**) |

**Related (committed):** `config/crab_hex/agents/crab_hex_rl_cfg.py`, `crab_hexapod_task/mdp/observations.py`, `modules/on_policy_runner_with_extractor.py`.

---

## 3. Training and playing the hexapod

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

- **Recommended (checkpoints under `krabby-research`):** `cd` into `**$KRABBY_ROOT/krabby-research/parkour`**, then run `**$KRABBY_ROOT/IsaacLab/isaaclab.sh**` with an **absolute** `-p` path to `train.py` / `play.py`. Artifacts land in `**krabby-research/parkour/logs/rsl_rl/...`**.
- **Resume training** must use the **same** `cd` as the original run, because `--resume` / `--load_run` resolve under that directory’s `logs/rsl_rl/<experiment_name>/`.
- **Play** with an explicit `**--checkpoint`** uses that file path for inference; cwd does not change which weights load. You may still see a line like `Loading experiment from directory: ...` that reflects cwd-based `log_root_path`—when you pass `--checkpoint`, the run uses the checkpoint path you gave.

**Alternative:** if you `cd "$KRABBY_ROOT/IsaacLab"` and run `./isaaclab.sh`, checkpoints go under `**IsaacLab/logs/rsl_rl/...`** instead (same script, different cwd).

### 3.0 Two-stage curriculum: flat walk → parkour teacher

Train basic walking on flat terrain first, then fine-tune on the full parkour teacher MDP. Checkpoints use the **same** policy network as the teacher (observation space unchanged), so stage 2 loads stage-1 weights with `--resume`.

**Stage 1 — flat walk** (`logs/rsl_rl/crab_hex_flat_walk/`):

```bash
cd "$KRABBY_ROOT/krabby-research/parkour"
"$KRABBY_ROOT/IsaacLab/isaaclab.sh" -p "$KRABBY_ROOT/krabby-research/parkour/scripts/rsl_rl/train.py" \
  --task Isaac-Crab-Hex-Flat-Walk-v0 \
  --headless --num_envs 256 --seed 1 --max_iterations 20000
```

Checkpoints save every **100** iterations (`model_0.pt`, `model_100.pt`, …). Reference flat-walk run (~**104 kg** USD): `logs/rsl_rl/crab_hex_flat_walk/2026-05-17_23-25-09.WORKING_VERSION_1.104kg/`.

**Play flat walk** (viewer):

```bash
cd "$KRABBY_ROOT/krabby-research/parkour"
"$KRABBY_ROOT/IsaacLab/isaaclab.sh" -p "$KRABBY_ROOT/krabby-research/parkour/scripts/rsl_rl/play.py" \
  --task Isaac-Crab-Hex-Flat-Walk-Play-v0 \
  --num_envs 1 --real-time \
  --checkpoint "$KRABBY_ROOT/krabby-research/parkour/logs/rsl_rl/crab_hex_flat_walk/2026-05-17_23-25-09.WORKING_VERSION_1.104kg/model_900.pt"
```

Replace the timestamp and `model_*.pt` with your run folder and latest checkpoint.

**Stage 2a — parkour warm-up** (resume flat checkpoint; easier mixed terrain):

Use an **absolute** `--checkpoint` path to the flat-walk run (flat and teacher use different `experiment_name` log folders):

```bash
export KRABBY_HEX_TRAIN_EASY=1
cd "$KRABBY_ROOT/krabby-research/parkour"
FLAT_CKPT="$KRABBY_ROOT/krabby-research/parkour/logs/rsl_rl/crab_hex_flat_walk/<FLAT_WALK_TIMESTAMP>/model_2999.pt"
"$KRABBY_ROOT/IsaacLab/isaaclab.sh" -p "$KRABBY_ROOT/krabby-research/parkour/scripts/rsl_rl/train.py" \
  --task Isaac-Crab-Hex-Teacher-v0 \
  --headless --num_envs 256 --seed 1 \
  --resume --checkpoint "$FLAT_CKPT" \
  --max_iterations 3000
```

**Stage 2b — full parkour** (unset easy flag; continue from stage-2a checkpoint):

```bash
unset KRABBY_HEX_TRAIN_EASY
cd "$KRABBY_ROOT/krabby-research/parkour"
"$KRABBY_ROOT/IsaacLab/isaaclab.sh" -p "$KRABBY_ROOT/krabby-research/parkour/scripts/rsl_rl/train.py" \
  --task Isaac-Crab-Hex-Teacher-v0 \
  --headless --num_envs 256 --seed 1 \
  --resume --load_run <STAGE_2A_TIMESTAMP> --checkpoint model_2999.pt \
  --max_iterations 7000
```

**TensorBoard (stage 1):**

```bash
tensorboard --logdir "$KRABBY_ROOT/krabby-research/parkour/logs/rsl_rl/crab_hex_flat_walk" --port 6006
```

Prioritize `Episode_Reward/reward_forward_progress_along_command`, `Train/mean_reward`, `Train/mean_episode_length`, and `Episode_Termination/crab_failure`. Use `track_lin_vel_xy_exp` and `Metrics/base_velocity/error_vel_xy` when tuning speed tracking.

**Flat-walk stance check (no checkpoint):**

```bash
"$KRABBY_ROOT/IsaacLab/isaaclab.sh" -p "$KRABBY_ROOT/krabby-research/parkour/scripts/zero_agent.py" \
  --task Isaac-Crab-Hex-Flat-Walk-Play-v0 --num_envs 1
```

### 3.1 Zero-agent stance check (no policy)

[`parkour/scripts/zero_agent.py`](../../../scripts/zero_agent.py) runs any registered `parkour_tasks` env with **all-zero actions** (hold default joint targets from `crab_hex_scene_cfg.py`; no checkpoint). Use this to verify **spawn height**, **default pose**, and **`CRAB_HEX_VIEWER`** before training.

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

- Default `--task` is `Isaac-Crab-Hex-Teacher-Play-v0`; use `Isaac-Crab-Hex-Teacher-v0` to match the training MDP (parkour terrain mix).
- Add `--headless` for no GUI (physics only).
- The script must be launched via `isaaclab.sh -p` (do not run `zero_agent.py` directly — you will get `Permission denied`).
- Passive stability is **not** the same as a trained policy: the robot only holds the configured default pose under gravity. A few tens of seconds upright is normal; long collapse means retune spawn or joint defaults in `crab_hex_scene_cfg.py`.

### 3.1a Crab verification scripts (headless)

Optional checks in [`scripts/`](scripts/). Run from `krabby-research/parkour` via `isaaclab.sh -p` (same as `zero_agent.py`).

| Script | What it does |
|--------|----------------|
| [`verify_crab_contact_physics.py`](scripts/verify_crab_contact_physics.py) | Spawns the flat-walk env, steps with zero actions, then prints a runtime audit: whether `.*_Footpad` bodies resolve on `contact_forces`, per-link masses (~**104 kg** total expected), foot contact flags, and friction/material notes. Writes JSON to `logs/rsl_rl/crab_hex_flat_walk/diagnostics/contact_physics_audit.json` by default (`--output` to override). Use after USD or spawn changes. |
| [`verify_crab_joint_drive.py`](scripts/verify_crab_joint_drive.py) | Drives each of the **18** revolute joints one at a time (± action) and reports whether the joint moves (position delta, torque, velocity). Gravity off by default for a clean actuation test. Exits with code **1** if any joint fails. Use after actuator or joint limit changes in `crab_hex_scene_cfg.py`. |

```bash
cd "$KRABBY_ROOT/krabby-research/parkour"
SCRIPTS=parkour_tasks/parkour_tasks/crab_hexapod_task/scripts

"$KRABBY_ROOT/IsaacLab/isaaclab.sh" -p "$SCRIPTS/verify_crab_contact_physics.py" --headless

"$KRABBY_ROOT/IsaacLab/isaaclab.sh" -p "$SCRIPTS/verify_crab_joint_drive.py" --headless
```

### 3.2 Train the teacher (parkour only, or stage 2 after flat walk)

The teacher uses privileged observations (terrain, dynamics, etc.) and trains with `scripts/rsl_rl/train.py`.

**Example: 256 envs, 10 000 PPO iterations (≈ 6 h on an RTX 5080–class GPU):**

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

Point `--logdir` at the parent `**logs/rsl_rl**` directory that matches **how you trained** (same rule as checkpoints: relative to the shell’s current working directory):

```bash
conda activate env_isaaclab
# If you trained from krabby-research/parkour (recommended above):
tensorboard --logdir "$KRABBY_ROOT/krabby-research/parkour/logs/rsl_rl" --port 6006 --bind_all
# If you trained from IsaacLab instead:
# tensorboard --logdir "$KRABBY_ROOT/IsaacLab/logs/rsl_rl" --port 6006 --bind_all
```

Open **[http://localhost:6006/](http://localhost:6006/)** (or another `--port` if 6006 is in use). A single `--logdir` on `**logs/rsl_rl`** lists every experiment underneath (e.g. teacher and student runs). Scalars keep updating while training is running; quit TensorBoard with **Ctrl+C** in that terminal.

In **Scalars**, search for `**mean_reward`** / `**Train/**` and `**Episode_Reward/**` (per-term curves such as `reward_tracking_goal_vel`, `reward_collision`, matching the training log).

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

### 3.3 Play back the trained teacher (viewer)

To **watch** the trained hexapod on parkour terrain in the Isaac Sim GUI, use `scripts/rsl_rl/play.py` with either the **training** env or the **play** env:

- Training-style env (exact MDP used for training):

```bash
cd "$KRABBY_ROOT/krabby-research/parkour"
"$KRABBY_ROOT/IsaacLab/isaaclab.sh" -p "$KRABBY_ROOT/krabby-research/parkour/scripts/rsl_rl/play.py" \
  --task Isaac-Crab-Hex-Teacher-v0 \
  --num_envs 1 \
  --real-time \
  --checkpoint "$KRABBY_ROOT/krabby-research/parkour/logs/rsl_rl/crab_hex_teacher/<TIMESTAMP>/model_9999.pt"
```

- Play-style env (longer episodes, follow-cam, parkour + command debug; default **easier** terrain mix unless `KRABBY_HEX_PLAY_HARD=1`):

```bash
cd "$KRABBY_ROOT/krabby-research/parkour"
"$KRABBY_ROOT/IsaacLab/isaaclab.sh" -p "$KRABBY_ROOT/krabby-research/parkour/scripts/rsl_rl/play.py" \
  --task Isaac-Crab-Hex-Teacher-Play-v0 \
  --num_envs 1 \
  --real-time \
  --checkpoint "$KRABBY_ROOT/krabby-research/parkour/logs/rsl_rl/crab_hex_teacher/<TIMESTAMP>/model_9999.pt"
```

Replace `<TIMESTAMP>` with the folder name printed in your training log (for example `2026-05-07_17-23-43`).

### 3.4 Train (and later distill) the student

The student uses the same hexapod env but with **student** observations (e.g. depth input) and a **distillation** algorithm. A minimal training command is:

```bash
cd "$KRABBY_ROOT/krabby-research/parkour"
"$KRABBY_ROOT/IsaacLab/isaaclab.sh" -p "$KRABBY_ROOT/krabby-research/parkour/scripts/rsl_rl/train.py" \
  --task Isaac-Crab-Hex-Student-v0 \
  --headless \
  --num_envs 1024 \
  --seed 1
```

Student training uses the same layout under `krabby-research/parkour`:

```text
krabby-research/parkour/logs/rsl_rl/crab_hex_student/<TIMESTAMP>/
```

(i.e. `**$KRABBY_ROOT/krabby-research/parkour/logs/rsl_rl/crab_hex_student/<TIMESTAMP>/**` when launched from `krabby-research/parkour` as above.)

**TensorBoard** for the student is the same as for the teacher: use `**tensorboard --logdir "$KRABBY_ROOT/krabby-research/parkour/logs/rsl_rl"`** (see **§3.1**).

You can play a student checkpoint with:

```bash
cd "$KRABBY_ROOT/krabby-research/parkour"
"$KRABBY_ROOT/IsaacLab/isaaclab.sh" -p "$KRABBY_ROOT/krabby-research/parkour/scripts/rsl_rl/play.py" \
  --task Isaac-Crab-Hex-Student-v0 \
  --num_envs 1 \
  --real-time \
  --checkpoint "$KRABBY_ROOT/krabby-research/parkour/logs/rsl_rl/crab_hex_student/<TIMESTAMP>/model_XXXX.pt"
```

---

## 4. Reference: existing quadruped (Unitree Go2) rewards and training

The crab hexapod task is built by following the conventions of the **extreme parkour Unitree Go2** task that ships with Isaac Lab.

- **Gym registrations (Go2 teacher / student / eval / play):**  
`IsaacLab/Isaaclab_Parkour/parkour_tasks/parkour_tasks/extreme_parkour_task/config/go2/__init__.py`  
(e.g. `Isaac-Extreme-Parkour-Teacher-Unitree-Go2-v0`, `Isaac-Extreme-Parkour-Teacher-Unitree-Go2-Play-v0`, etc.)
- **Go2 MDP / rewards / actions:**  
`IsaacLab/Isaaclab_Parkour/parkour_tasks/parkour_tasks/extreme_parkour_task/config/go2/agents/parkour_mdp_cfg.py`  
which in turn uses the same reward functions in  
`krabby-research/parkour/parkour_isaaclab/envs/mdp/rewards.py`.

### 4.1 Go2 teacher training (extreme parkour)

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

### 4.2 Go2 play (extreme parkour teacher play env)

You can visualize a trained Go2 teacher policy on parkour terrain using the Go2 **PLAY** env:

```bash
cd "$KRABBY_ROOT/IsaacLab"
./isaaclab.sh -p ./Isaaclab_Parkour/scripts/rsl_rl/play.py \
  --task Isaac-Extreme-Parkour-Teacher-Unitree-Go2-Play-v0 \
  --num_envs 1 \
  --real-time \
  --checkpoint ./Isaaclab_Parkour/logs/rsl_rl/unitree_go2_parkour_teacher/<TIMESTAMP>/model_XXXX.pt
```

The hexapod task mirrors this layout (Gym registrations, env cfgs, reward wiring, and train/play scripts), so anyone familiar with the Go2 extreme parkour examples should find the crab hexapod task immediately recognizable.  
Training uses `**crab_simple.usda**` only; set `**KRABBY_HEX_USD_PATH**` only if your checkout or container layout is non-standard. RSL-RL checkpoints for the commands in **§3** are kept under `**krabby-research/parkour/logs/rsl_rl/`** by running from that directory as documented there.

---

## Appendix

### Appendix A - Learning from the first successful run

- **Focus on USD, not reward tuning to start:** Removed overlapping reward experiments until `crab_simple.usda` and spawn were credible. Reward tuning can come incrementally after the asset and default stance are trustworthy.
- **Explicit masses in USD:** Per-link weights (~**104 kg** total for the current `crab_simple.usda`; earlier ~**25 kg** baseline also in logs) instead of relying on PhysX auto-mass. Retrain when additional payload is modeled.
- **Foot rubber at the feet:** Separate `*_Footpad` colliders with `FootRubber` for ground contact (not full-shank tibia collision).
- **Stable stance:** Body–hip yaw splay **±0.6** on front and rear legs; spawn `z` **1.05** m (`KRABBY_HEX_SPAWN_Z`).
- **Simpler flat-walk reward weights:** Small `CrabHexFlatWalkRewardsCfg` set for easier experimentation.
- **Velocity in observations:** Base linear velocity (`root_lin_vel_xy`) included in proprioceptive observations. 

### Appendix B - 2026-05-19 Flat-Walk Tuning Summary

This commit captures the best flat-walk baseline found during the 2026-05-19 tuning pass and documents why the current flat-walk settings were chosen.

The checked-in baseline artifacts are stored under:

```text
parkour_tasks/parkour_tasks/crab_hexapod_task/runs/2026-05-19_12-06-10/
```

It contains:

- `crab_simple_2026-05-19_12-06-10.usda` - the USD snapshot used for this baseline.
- `model_4000.pt` - the baseline flat-walk policy checkpoint.
- `README.md` - short provenance and frozen flat-walk settings.

Key changes and why they were made:

- **USD and checkpoint bundle:** The known-good `crab_simple.usda` snapshot and `model_4000.pt` are stored under `runs/2026-05-19_12-06-10/` so the play baseline is reproducible even if later assets or training logs change.
- **Explicit USD override in play:** The helper script sets `KRABBY_HEX_USD_PATH` so the bundled checkpoint plays against the bundled USD, not whichever asset happens to be current in `assets/`.
- **Flat-walk command range:** `lin_vel_x = (0.25, 0.60)` keeps the speed request high enough for visible progress while avoiding the earlier overly aggressive forward shortcut.
- **Forward progress reward:** `reward_forward_progress_along_command = 0.50` was selected as the best balance so far. Larger values encouraged faster motion but began to reintroduce north/south drift; smaller values made the gait too conservative.
- **Velocity tracking kept primary:** `track_lin_vel_xy_exp = 1.0` stays active so the policy is rewarded for matching commanded body-frame planar velocity instead of just moving roughly forward.
- **Lateral drift penalty:** `penalty_lin_vel_y = -3.0` keeps body-frame sideways velocity small without over-constraining gait exploration.
- **Air-time reward:** `reward_feet_air_time_positive = 0.25` nudges the policy toward clearer swing/step behavior rather than an all-feet shuffling gait.
- **Collision and feet-slide terms disabled for flat walk:** `reward_collision = 0.0` and `feet_slide = 0.0` remain available but are not part of this baseline because the drift/speed tradeoff was better controlled by velocity, progress, and air-time terms.
- **Stance defaults:** The current stance keeps body-hip yaw splay at **±0.6**, hip-femur at **0.30**, and mirrored knee defaults (left **−0.07**, right **+0.10**) to balance the passive zero-action stance without removing body-hip splay.

Run the bundled baseline by passing both the USD and checkpoint explicitly:

```bash
export KRABBY_ROOT=/home/sanjay/Projects/krabby
conda activate env_isaaclab
RUNS_DIR="$KRABBY_ROOT/krabby-research/parkour/parkour_tasks/parkour_tasks/crab_hexapod_task/runs"

"$RUNS_DIR/play_crab_hex_flat_walk_baseline.sh" \
  "$RUNS_DIR/2026-05-19_12-06-10/crab_simple_2026-05-19_12-06-10.usda" \
  "$RUNS_DIR/2026-05-19_12-06-10/model_4000.pt"
```

The script sets:

- `KRABBY_HEX_USD_PATH` to the bundled USD.
- `KRABBY_HEX_SPAWN_Z=1.05` unless already set.
- `PYTHONPATH` for the local `parkour` and `parkour_tasks` packages.
- `Isaac-Crab-Hex-Flat-Walk-Play-v0`, which uses flat/easy terrain for this flat-walk checkpoint.

Stop large headless training jobs before GUI play to avoid GPU memory pressure.

Reference metrics around the bundled `model_4000.pt`:

- `track_lin_vel_xy_exp` around **0.87**.
- `track_ang_vel_z_exp` around **0.92**.
- `Metrics/base_velocity/error_vel_xy` around **0.16**.
- `Metrics/base_velocity/error_vel_yaw` around **0.37**.
- `Episode_Termination/crab_failure` below **1%**.
