# Phase pipeline campaign (paradigm restore) — CHARTER and verification record

Approved plan: `/home/nickmagus/.claude/plans/wiggly-gathering-sloth.md` (user, 2026-09-07).
Successor to the A15+B lineage retrain (`sim_fine_tuning/2026-09-06_2130_a15b_lineage/`), whose
20k head `parkour/logs/rsl_rl/crab_hex_flat_walk/2026-09-07_04-38-50/model_19996.pt` is the policy
of record and becomes the **end of phase 2c**.

## Goal (user)

Fold the fine-tuning curriculum back into the repository's three-phase training paradigm:
**phase 1 pure student**, **phase 2 teacher-student**, **phase 3 student distillation**, with
sub-phases matching the baked curriculum, phase 3 extending past the 20k head.

**Decisions (user):** 1a = 0–5k; 2a / 2b / 2c = 5–10k / 10–15k / 15–20k; phase 2 on
`Isaac-Crab-Hex-Teacher-v0` modes `2a/2b/2c`, phase 3 on `Isaac-Crab-Hex-Student-v0`;
3a distils on the phase-2c MDP (difficulty 0.20–0.70), 3b on 0.70–0.90.

## What changed (code)

| Piece | File | Notes |
|---|---|---|
| Phase presets | `config/crab_hex/crab_hex_phases.py` (new) | `PHASES` = frozen env-var stacks of the recorded windows (`run_lineage.window_stack`); legacy golden presets; `activate_phase` (setdefault), `phase_env`, `is_student_phase` (+ `KRABBY_STUDENT_MDP=1`) |
| Activation | `config/crab_hex/__init__.py` | `activate_phase()` before any cfg import (USD path is read at import) |
| Shared knobs | `crab_hex_env_cfg.py` `apply_flat_walk_knobs(cfg, include_reward_anneal)` | the flat-walk `__post_init__` body, extracted verbatim (`self`→`cfg`) |
| Phase-2 teacher modes | `_crab_hex_teacher_mode()` accepts `2a/2b/2c`; `CrabHexTeacherEnvCfg._crab_hex_apply_teacher_mode` | flat-walk rewards + terminations, `full` actions, knobs; PLAY keeps the train MDP |
| Flat-walk immunity | `CrabHexFlatWalkEnvCfg.__post_init__` | pins mode `full` so a phase preset's exported mode cannot apply the knobs twice |
| Phase-3 student mirror | `CrabHexStudentEnvCfg.__post_init__` | teacher terrain generator (0.08 m / 40 cols, not simplified), flat-walk terminations, `full` actions, DR on the chassis, `enable_external_forces_every_iteration`, knobs without reward anneal; legacy 2b2 mirror when no phase |
| Runner | `agents/rsl_rl_ppo_cfg.py` | phase modes mirror the flat-walk runner (LR 3e-4, noise 1.5, clip 1.0, mirror loss via `_apply_flat_walk_symmetry`) |
| Driver | `sim_fine_tuning/tools/run_phases.py` + `launch_phases.sh` + `heartbeat_phases.sh` | train env = `KRABBY_PHASE` + `KRABBY_PLANT` only; evals with explicit keys (+ `KRABBY_STUDENT_MDP=1`, `--task Isaac-Crab-Hex-Student-v0` for phase 3); REPORT / CHANGELOG / state / heads; pauses after 2c, 3a, 3b |
| Tests | `tests/unit/test_crab_hex_phases.py` (20), `tests/integration/test_crab_hex_phase_configs.py` + `crab_hex_phase_cfg_dump.py` | see below |
| Docs | crab task README §2 / §3.1 / §4.0 / §4.4; `docs/crab-hex-forward-policy-config.md` | phase table, commands, env-var rows |

## Verification record

### Unit (pure, no Isaac) — 2026-09-07
`tests/unit/test_crab_hex_phases.py`: 20 passed. Presets 1a–2c equal `run_lineage.window_stack(w)`
for A15+B; legacy golden presets equal the exposure campaign's `lineage_stack` / `REPLAY_SCHEDULE`;
`activate_phase` keeps explicit overrides; plants resolve to existing USDs; phase-3 presets carry no
reward or `KRABBY_PHASEOUT` keys (tracking-shaping knobs removed from the student MDP set).

### Config identity inside Isaac Sim — 2026-09-07 13:12
`RUN_CRAB_HEX_CFG_IDENTITY=1 pytest tests/integration/test_crab_hex_phase_configs.py`: **10 passed**
(six headless boots, `class_to_dict` comparisons):
- `Teacher-v0` mode 2a/2b/2c env cfg == `Flat-Walk-v0` env cfg under the same preset; runner cfgs
  equal modulo `experiment_name`.
- `KRABBY_PHASE=1a/2a/2b/2c` reproduce the recorded `params/env.yaml` + `agent.yaml` of the lineage
  windows 0–3 (`2026-09-06_21-32-46`, `2026-09-06_23-59-54`, `2026-09-07_02-19-49`,
  `2026-09-07_04-38-50`) modulo runtime-resolved fields (seed / num_envs / device / prim-path
  namespace / generator-propagated sub-terrain size and scales). This is also the regression proof
  that extracting `apply_flat_walk_knobs` changed nothing.
- `KRABBY_PHASE=2c` == the raw window-3 `KRABBY_*` stack.
- 3a student == 2c teacher on commands, events (minus the camera-placement event), terminations,
  actions, horizon, decimation, sim dt, external-forces flag, terrain, plant USD. Before the mirror
  fix the student differed on terrain resolution (0.1 m / 20 cols / simplified), `limit_angle`
  (1.5 vs 0.5), action clip (±1 vs ±4.8) and DR body names (`base` vs `body`) — all corrected.
- Play variants keep the train MDP (Teacher-Play == Flat-Walk-Play under 2c).

### Smoke 2b (200 iterations, Teacher-v0 mode 2b, resume A15+B 10k head) — 2026-09-07 13:24
Run `logs/rsl_rl/crab_hex_teacher/2026-09-07_13-24-28`: `params/env.yaml` byte-identical to the
lineage window-2 run; `agent.yaml` differs only in `max_iterations` (200) and `experiment_name`;
the 32-row reward-term table and the 10 telemetry key groups in the training log are identical to
the window-2 log. Result block: `smoke/REPORT.md`.

### Smoke 3a (200 iterations, Student-v0, from the 20k head) — 2026-09-07 13:30 PASS
Run `logs/rsl_rl/crab_hex_student/2026-09-07_13-30-34`: `DistillationWithExtractor`; the depth actor
is initialised from the teacher actor ("No saved depth actor, copying"); the iteration counter
continued 19996 → 20195; `depth_actor_loss` 10.76 → 3.77 and `yaw_loss` 0.93 → 0.87 over 200
iterations; plant A15+B and 40 s episodes in `params/env.yaml`; all six record checks PASS.
Throughput 618 steps/s ≈ 7.2 s per iteration at 192 envs (≈ 10 h per 5k phase); GPU memory
13.7 GB of 16 GB with the teacher terrain generator (0.08 m / 40 cols) under the depth ray-caster.
Result block: `smoke/REPORT.md`.

### Phase 3a → 3b from the baked 2c head — launched 2026-09-07 14:00
`launch_phases.sh --campaign-dir sim_fine_tuning/2026-09-07_1330_phase_pipeline --plant A15+B
--phases 3a,3b --seed 3 --from-checkpoint .../2026-09-07_04-38-50/model_19996.pt`; pause +
push-notify after 3a (evals: slow canary, step onset, recal2b2w obstacle eval on the student task
with `KRABBY_STUDENT_MDP=1`), then 3b on relaunch.

## Budget and order
Smokes (~1 h) → phase 3a then 3b from the baked 2c head (≈ 9.5 h each at 192 envs; pause +
push-notify after each) → seed-2 replay 1a→3b (≈ 13 h + 19 h) → bake decision (user's).
GPU serial (one run saturates it); no live kills beyond plant-soundness / NaN / stall.

### PAUSED 2026-09-07 18:17 (user restart)
Phase 3a stopped at iteration 22000 of 24995 (`model_22000.pt` verified); resume via the `--continue` flag of `run_phases.py` (command in CHANGELOG / state.json).

### 2026-09-08 05:52 — phase 3a run 1 INVALID (runner clip): root cause + fix
Student runner `clip_actions` was None (flat-walk/phase-2: 1.0). Teacher on the student env: 0/100 → 0.70 completion after the fix. Coverage holes closed: distillation survival checks (driver), runner clip parity (identity test), `--policy-role` diagnostic flag in the gait harness. 3a rerun from the 20k head (~10 h + evals).
Rerun confirmed in the walking regime by iteration 20 (episode length ~1.5k steps, failure 0.36 by it 60). Open: teacher 0.70/30 on the student env vs 0.81/19 on flat-walk (same head/knobs) — quantify after 3a evals.

### 2026-09-08 16:02 — phase 3a rerun COMPLETE: student ≈ teacher (flat 0.79 vs 0.81, step 0.71 vs 0.68, obstacles 0.64 vs 0.66). PAUSED before 3b.

### 2026-09-09 12:19 — seed-3 pipeline 3a→3b COMPLETE
3b head model_29994: flat 0.81/19, step 0.68/32, obstacles 0.62/38, hard band 0.52/48 (teacher 0.81/19, 0.68/32, 0.66/34, 0.45/55; 3a 0.79/21, 0.71/29, 0.64/36, 0.51/49). Students ≈ teacher on every band; 3b adds nothing measurable over 3a. Open: seed-2 replay (user go), bake choice (user).

### 2026-09-09 12:26 — CLOSED (BAKED): phase-3 head of record = 3a `parkour/logs/rsl_rl/crab_hex_student/2026-09-08_05-54-01/model_24995.pt`; 3b not baked; seed-2 replay waived (user).
