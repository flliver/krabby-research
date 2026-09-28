# Gait eval baselines — v1 (Milestone 18, Task 0, AC 0c)

Produced by `scripts/run_gait_eval_suite.py` (default manifest `experiments/eval/scenarios_v1.yaml`, run from `krabby-research/parkour`) against
`experiments/eval/scenarios_v1.yaml`. Every later task (Task 1 reward shaping, Task 4 stage gates) is scored
against these numbers.

**Baseline checkpoints are the current-geometry ones, not the grant text's `model_6300.pt` /
`model_9800.pt`.** Those two predate this session's cam-mechanism migration — they drove
`*_Body_Hip_RevoluteJoint` directly, whereas the current asset drives
`*_Body_CamShaft_RevoluteJoint` with the hip passive and kinematically slaved. Both are still
18-dim, so they *load*, and would silently produce meaningless gait rather than erroring. See
`experiments/eval/scenarios_v1.yaml`'s header comment for the full note.

- Teacher: `logs/rsl_rl/crab_hex_teacher/2026-08-05_17-36-56/model_21100.pt` (2b2 stage)
- Student: `logs/rsl_rl/crab_hex_student/2026-08-06_14-22-48/model_29098.pt`
- Flat-walk teacher: `logs/rsl_rl/crab_hex_flat_walk/2026-08-04_23-34-31/model_19999.pt`
  Note: the `flat_walk_forward` row was measured on `2026-08-04_23-34-31/model_19999.pt` (sha256 `68eb4e25…`, see its `run_meta.json`).

## Headline: tripod score is low everywhere, consistently

| scenario | episodes | tripod_score (median, p25–p75) | tippy_tap_fraction | slip_ratio | schedule_completion_rate |
|---|---|---|---|---|---|
| `teacher_2b2_forward` | 10 | 0.052 (0.020–0.078) | see run | see run | 0.70 |
| `student_2b2_forward` | 10 | 0.079 (0.032–0.108) | see run | see run | 0.90 |
| `student_flat_forward` | 10 | 0.080 (0.026–0.105) | see run | see run | 0.80 |
| `flat_walk_forward` | 10 | 0.074 (0.065–0.092) | see run | see run | 0.90 |
| `student_flat_probe_vy` | 5 | 0.054 (0.011–0.096) | — | — | 1.00 |
| `student_flat_probe_yaw` | 5 | 0.069 (0.037–0.074) | — | — | 1.00 |

(Full metrics, including `tippy_tap_fraction`/`slip_ratio`/per-hold breakdowns, are in each
scenario's `scenario_metrics.json` and `summary.md`.)

Every stage of the curriculum -- flat-walk through the 2b2 student -- scores in the same low
0.05-0.08 tripod range. That is a direct, machine-measured confirmation of the "tippy-tap"
micro-stepping pathology the grant overview describes qualitatively: rapid, poorly-phased
stepping rather than clean alternating-tripod support. Visual confirmation in each scenario's
`gait_diagram_episode_00.png` (dense, fragmented contact bars rather than a checkerboard).

`teacher_2b2_forward`'s `gait_diagram_episode_00.png` additionally shows a second, distinct
pathology: `RL_Footpad` is in near-continuous contact for the whole episode (barely lifts) while
the stance-slip trace repeatedly spikes to 2-4 m/s -- a foot that stays "in contact" by the
threshold definition while actually skating. This is exactly the failure mode the `slip_ratio`
metric (§ "Other metrics") was added to catch beyond the grant's original spec, since a
never-lifting foot cannot register on air time or break tripod alternation.

## Off-axis probes: recorded, not gated

`student_flat_probe_vy` and `student_flat_probe_yaw` ran and produced numbers, but per
`run_meta.json`'s `policy_sees_command_channel`, the policy is structurally blind to `vy` (zeroed
in the observation) and to `wz` (never in the observation vector at all; only reachable through
the `delta_yaw` hint the yaw probe injects). Do not read their tracking-error numbers as a
capability measurement -- see `eval_crab_hex_gait.py`'s module docstring and
`scenarios_v1.yaml`'s per-scenario notes for the full reasoning. This is effectively the grant's
"suspect 2" verdict, reached from the observation code rather than from measurement.

## Reproduce

**Not reproducible on the current tree.** These are era-A heads (2026-08-04..06) trained on the hand-authored
cam-shaft `assets/crab_simple.usda` of the day, before the 2026-08-20 measured-hardware rebuild, with
1151-wide observations (`run_meta.json: obs_dim_actual`). The rebuild changed the plant, the action layout and
the observation width (1149 today), so `runner.load` rejects them with a size mismatch (critic input 1151 vs 1149)
on every plant, including `--plant legacy_golden`. The reports here are the frozen AC 0c record; the manifest
`experiments/eval/scenarios_v1.yaml` stays sha-pinned to these checkpoints, but re-running the suite against them is
not possible.

Each scenario is a separate `isaaclab.sh -p` process (only one Isaac Sim process fits the GPU at
a time). `--repeat 2` reruns a scenario with different seeds to check the tripod-score
determinism/noise floor before trusting a threshold tighter than that spread.
