<!-- paths-note -->
> **Paths note (2026-09-09):** this campaign moved from `sim_fine_tuning/2026-09-07_1330_phase_pipeline/` to `parkour/parkour_tasks/parkour_tasks/crab_hex_forward_task/experiments/2026-09-07_1330_phase_pipeline/`. Absolute paths below (`/home/.../sim_fine_tuning/...`, `parkour/logs/rsl_rl/...`) name raw artifacts that stay on disk untracked; the tracked checkpoint of record is `head/` (see `bundle.yaml`) and the eval summaries are in place / under `evals/`.

# Phase pipeline campaign `2026-09-07_1330_phase_pipeline`

Plant **A15+B** (`/home/nickmagus/krabby/krabby-research/assets/variants/crab_simple__splay15_axis2p5in.usda`), seed 3, phases 3a, 3b. Driver: `sim_fine_tuning/tools/run_phases.py`; presets: `crab_hex_phases.py` (`KRABBY_PHASE` / `KRABBY_PLANT`).

| phase | task / mode | iterations | resume from | notes |
|---|---|---|---|---|
| 3a | `Isaac-Crab-Hex-Student-v0` | 5000 | 2c | phase 3 student distillation on the 2c MDP |
| 3b | `Isaac-Crab-Hex-Student-v0` | 5000 | 3a | phase 3 student distillation, harder band 0.70-0.90 |

## Phase 3a — Isaac-Crab-Hex-Student-v0 — seed3 — 2026-09-08 05:34 (continued after a pause)

- status **ok** | checkpoint `2026-09-07_23-13-07/model_24995.pt` | resume `model_19996.pt` | 5000 its | wall 10.59 h
- distillation: iterations logged 2996 | depth_actor_loss first 7.660 -> last 2.277 | yaw_loss first 0.783 -> last 0.705
- slow canary (morph manifest): tripod 0.065 | completion 0.000 | tracking None | falls 100/100 | pitch-fwd share 0.020
- step onset (morph manifest, shallow 0.05-0.2): completion 0.000 | falls 100/100 | pitch-fwd share 0.000
- obstacle eval (recal2b2w 0.20-0.70): completion 0.000 | tripod 0.030 | falls 100/100
- check PASS: plant is A15+B (params/env.yaml usd path)
- check PASS: episode_length_s 40 in params
- check PASS: experiment crab_hex_student
- check PASS: distillation algorithm (DistillationWithExtractor)
- check PASS: depth losses logged
- check PASS: iteration counter continued from the teacher head


> **2026-09-08 05:52 — the phase 3a block above is INVALID** (student runner `clip_actions` unset → unclipped raw actions on the full action space; every rollout fell). Root cause, confirmation and fix in CHANGELOG. 3a rerun follows.

## Phase 3a — Isaac-Crab-Hex-Student-v0 — seed3 — 2026-09-08 15:51

- status **ok** | checkpoint `2026-09-08_05-54-01/model_24995.pt` | resume `model_19996.pt` | 5000 its | wall 9.87 h
- distillation: iterations logged 5000 | depth_actor_loss first 10.941 -> last 3.569 | yaw_loss first 1.428 -> last 1.161
- student rollouts: mean episode length first50 1141.290 -> last50 1557.071 steps | crab_failure last50 0.357
- slow canary (morph manifest): tripod 0.512 | completion 0.790 | tracking 0.650 | falls 21/100 | pitch-fwd share 1.000
- step onset (morph manifest, shallow 0.05-0.2): completion 0.710 | falls 29/100 | pitch-fwd share 0.966
- obstacle eval (recal2b2w 0.20-0.70): completion 0.640 | tripod 0.490 | falls 36/100
- check PASS: plant is A15+B (params/env.yaml usd path)
- check PASS: episode_length_s 40 in params
- check PASS: experiment crab_hex_student
- check PASS: distillation algorithm (DistillationWithExtractor)
- check PASS: depth losses logged
- check PASS: iteration counter continued from the teacher head
- check PASS: student rollouts survive (mean episode length last 50 its >= 600 steps)
- check PASS: student failure share last 50 its < 0.9

## Phase 3a (valid rerun) vs the 2c teacher — 2026-09-08 16:02

Student `crab_hex_student/2026-09-08_05-54-01/model_24995.pt` (depth actor + encoder, 5000 distillation iterations from the 20k head, action clip fixed) on `Isaac-Crab-Hex-Student-v0` + `KRABBY_STUDENT_MDP=1`; teacher numbers from the lineage record on the flat-walk task (same plant, knobs, env_seed 1).

| scenario | student 3a: completion / falls / tripod / tracking | 2c teacher: completion / falls / tripod / tracking |
|---|---|---|
| slow flat canary | **0.79** / 21 / 0.51 / 0.65 | 0.81 / 19 / 0.50 / 0.65 |
| step onset (shallow 0.05–0.2) | **0.71** / 29 / 0.48 / 0.58 | 0.68 / 32 / 0.48 / 0.58 |
| obstacle course (recal2b2w 0.20–0.70) | **0.64** / 36 / 0.49 / — | 0.66 / 34 / 0.49 / — |

Training: mean episode length 1141 (first 50 its) → 1557 (last 50); crab_failure 0.36 at the end (teacher ≈ 0.34 on this MDP); depth_actor_loss 10.9 → 3.6; yaw_loss 1.43 → 1.16. All eight record checks PASS. The depth student reproduces the privileged teacher within eval noise on every scenario, so the phase-2c competence transferred to depth + proprioception. **Paused for the go/no-go before 3b** (difficulty band 0.70–0.90).

## Phase 3b — Isaac-Crab-Hex-Student-v0 — seed3 — 2026-09-09 12:04

- status **ok** | checkpoint `2026-09-09_02-06-51/model_29994.pt` | resume `model_24995.pt` | 5000 its | wall 9.87 h
- distillation: iterations logged 5000 | depth_actor_loss first 5.745 -> last 3.201 | yaw_loss first 1.116 -> last 1.087
- student rollouts: mean episode length first50 1044.378 -> last50 1366.781 steps | crab_failure last50 0.439
- slow canary (morph manifest): tripod 0.507 | completion 0.810 | tracking 0.649 | falls 19/100 | pitch-fwd share 0.947
- step onset (morph manifest, shallow 0.05-0.2): completion 0.680 | falls 32/100 | pitch-fwd share 0.969
- obstacle eval (recal2b2w 0.20-0.70): completion 0.620 | tripod 0.481 | falls 38/100
- check PASS: plant is A15+B (params/env.yaml usd path)
- check PASS: episode_length_s 40 in params
- check PASS: experiment crab_hex_student
- check PASS: distillation algorithm (DistillationWithExtractor)
- check PASS: depth losses logged
- check PASS: iteration counter continued from the teacher head
- check PASS: student rollouts survive (mean episode length last 50 its >= 600 steps)
- check PASS: student failure share last 50 its < 0.9

## Seed-3 pipeline complete — phase-3 heads vs the 2c teacher — 2026-09-09 12:19

Evals: flat canary and step onset (morph manifest `slow__A15pB` / `step__A15pB`), obstacle course recal2b2w 0.20–0.70 (`flat_walk_slow_v2`), and the hard band recal2b2w 0.70–0.90 (same scenario, `KRABBY_FLAT_TERRAIN_DIFF=0.70:0.90`; eval logs under `logs/rsl_rl/gait_eval/phases/2026-09-07_1330_phase_pipeline_hardband/`). Students on `Isaac-Crab-Hex-Student-v0` + `KRABBY_STUDENT_MDP=1`; teacher on the flat-walk task; 100 episodes, env_seed 1. Cells: completion / falls / tripod median.

| head | flat canary | step onset | obstacles 0.20–0.70 | hard band 0.70–0.90 |
|---|---|---|---|---|
| 2c teacher `crab_hex_flat_walk/2026-09-07_04-38-50/model_19996.pt` (privileged) | 0.81 / 19 / 0.50 | 0.68 / 32 / 0.48 | 0.66 / 34 / 0.49 | 0.45 / 55 / 0.43 |
| 3a student `crab_hex_student/2026-09-08_05-54-01/model_24995.pt` (depth, 5k on 0.20–0.70) | 0.79 / 21 / 0.51 | 0.71 / 29 / 0.48 | 0.64 / 36 / 0.49 | 0.51 / 49 / 0.46 |
| 3b student `crab_hex_student/2026-09-09_02-06-51/model_29994.pt` (depth, +5k on 0.70–0.90) | 0.81 / 19 / 0.51 | 0.68 / 32 / 0.46 | 0.62 / 38 / 0.48 | 0.52 / 48 / 0.45 |

Reading: both depth students reproduce the privileged teacher within eval noise on every band (differences of 2–6 falls in 100); the students are not worse than the teacher on the hard band (0.51–0.52 vs 0.45). Phase 3b's extra 5k iterations on the 0.70–0.90 band changed nothing measurable, including on that band itself (0.52 vs 0.51). Distillation transferred the 2c competence to depth + proprioception; it did not add competence beyond the teacher, as expected for pure action-matching.

**Open (user):** (1) seed-2 replay 1a→3b to confirm the pipeline reproduces the 2c head and its student; (2) which phase-3 head to bake — 3a and 3b are equivalent on every eval (3a is 5k iterations cheaper; 3b is the end of the paradigm's pipeline).

## BAKE — 2026-09-09 12:26 (user decision: "bake 3a as the phase-3 head and stop there")

**Phase-3 head of record (A15+B, depth student): `parkour/logs/rsl_rl/crab_hex_student/2026-09-08_05-54-01/model_24995.pt`** — phase 3a, 5000 distillation iterations
from the phase-2c policy of record `parkour/logs/rsl_rl/crab_hex_flat_walk/2026-09-07_04-38-50/model_19996.pt`
on the 2c MDP (recal2b2w 0.20–0.70, walking slots, 40 s episodes, DR, RSI 0.2). Evaluate/play it on
`Isaac-Crab-Hex-Student-v0` / `-Student-Play-v0` with `KRABBY_STUDENT_MDP=1` (or `KRABBY_PHASE=3a`).
Numbers of record (completion / falls / tripod, 100 episodes, env_seed 1): flat canary 0.79 / 21 / 0.51,
step onset 0.71 / 29 / 0.48, obstacles 0.20–0.70 0.64 / 36 / 0.49, hard band 0.70–0.90 0.51 / 49 / 0.46
(teacher: 0.81 / 19, 0.68 / 32, 0.66 / 34, 0.45 / 55).

**Baked pipeline (paradigm of record):** 1a (0–5k, `Flat-Walk-v0`) → 2a / 2b / 2c (5–20k,
`Teacher-v0` modes) → 3a (20k → 25k, `Student-v0`), all via `KRABBY_PHASE` + `KRABBY_PLANT=A15+B`.
Phase 3b (25k → 30k on 0.70–0.90) was run and found equivalent to 3a on every eval; it stays available
as a preset but is **not** part of the baked pipeline.

**Required for phase 3 (in code since 2026-09-08):** `CrabHexStudentPPORunnerCfg.clip_actions = 1.0`
(raw-action clip in the vec-env wrapper, as the teacher runners) — without it every rollout falls.

**Waived (user):** seed-2 replay of 1a→3b (pipeline reproducibility across seeds is unverified for
phase 3; the phase-1/2 presets are verified against the recorded seed-3 lineage by config identity).

## CAMPAIGN CLOSE — 2026-09-09 12:26
Seed-3 pipeline 3a→3b complete, 3a baked, 3b not baked, seed-2 replay waived. Code: phase presets,
teacher modes, student mirror, driver, identity + unit tests, docs (see CHARTER). Nothing pushed.
