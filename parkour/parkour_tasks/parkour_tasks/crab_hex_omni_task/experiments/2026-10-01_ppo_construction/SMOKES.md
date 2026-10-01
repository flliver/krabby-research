# GPU smokes, rung 0b and rung 1 (plan step 2 items 4–6; CHARTER §5, §6, §9)

**Date:** 2026-10-01, 09:38–10:30. **Tree:** worktree `claude/ppo-construction` (uncommitted, on top of
`fd255444`); the pre-change control runs used the main checkout at `fd255444` read-only. Nothing was committed or
pushed.

**Raw notes and scripts:** `/tmp/claude-1000/-home-nickmagus-krabby-krabby-research/f09fa754-dd17-413f-8553-6b22fdb41cf7/scratchpad/smokes/`
(`logs/`, `cmp_*.txt`, `check_*.json`, `audit_*.txt`, `check_lr_epochkl.txt`, `fork_c4.txt`).

**Safety:** one Isaac process at a time (checked before every launch); every process under
`systemd-run --user --scope -p MemoryMax=48G -p MemorySwapMax=0` and `timeout -s TERM`; no SIGKILL; detached with
`setsid nohup`. No Kit, OOM or cache incident.

## How the smokes were launched

**Direct smokes** (`smoke.sh TREE NAME SEED ITERS NUM_ENVS OVERLAY_FILE [RESUME]`) reproduce
`run_phases.train()`'s command for base `1b` with only `--num_envs` changed:

```
cd <tree>/parkour && systemd-run --user --scope -q -p MemoryMax=48G -p MemorySwapMax=0 timeout -s TERM 2400 \
  env OMNI_KIT_ACCEPT_EULA=yes TERM=xterm KRABBY_OMNI_PHASE=1b KRABBY_PLANT=A15+B PYTHONUNBUFFERED=1 \
      [PYTHONPATH=$WT/parkour:$WT/parkour/parkour_tasks]   # worktree runs only
      <the chain's stage overlay, verbatim from candidates.yaml (overlay.py)> \
  /home/nickmagus/krabby/isaac_venv/bin/python <tree>/parkour/scripts/rsl_rl/train.py \
  --task Isaac-Crab-Hex-Omni-Walk-v0 --headless --num_envs 64 --seed 11 --max_iterations 20 --run_name <name>
```

No other `KRABBY_*` was in the environment. Main-tree runs log to `parkour/logs/rsl_rl/crab_hex_omni_flat_walk/`
of the main checkout; worktree runs log to the worktree's own `parkour/logs/rsl_rl/crab_hex_omni_flat_walk/`.
The worktree code is what ran: every worktree run logs to the worktree's log root and writes the new `PPO/*` tags
(absent from the main-tree runs), and every knob-on run prints `[PPO] construction: …` and `[KNOB] … consumed`.

**Driver-path smokes** (`driver_train_smoke.py REFERENCE ITERS SCRATCH_DIR [RESUME] [STAGE]`) call the
worktree driver itself: `run_candidates.reference_recipe` → `stage_env` (the overlay the driver passes to
`train`) → `run_phases.train(c, PHASES[base], …, extra_env=…, run_name=…)`, on a scratch `Campaign` dir (never
the campaign's). The command, environment, `--num_envs` and the `[KNOB] KRABBY_NUM_ENVS` line are the driver's.
Only the poll cadence was shortened (`POLL_S` 120 → 10 s).

**Reads.**
- `cmp_runs.py`: params diff (ignoring run name and log paths) plus every TB scalar tag, step by step.
- `check_on.py`: the campaign's own `verdict.py` wiring read (`wiring_read`, `consumed_lines`, `dump_checks`,
  `step4a_wiring_fails`), with the iteration-200 window moved to the last logged iteration
  (`W_200 = (0, 19)`, `WIRING_AT = 19`). The checks are the same; only the read point moves, since a 20-iteration
  run never reaches iteration 200, where `--check-run --at 200` would return 2 (wait).
- `audit.py`: the driver's own post-window audit, `run_candidates.overlay_checks`. That is
  `OVERLAY_DUMP_CHECKS` on `params/{env,agent}.yaml`, plus `construction_log_checks` on the training log.

## 1. Knob-off identity with the F11 control (CHARTER §9) — PASS

Overlay: `C_lr__C0` without `KRABBY_PPO_HAND_DIAG`; seed 11, 64 envs, 20 iterations.

| Run | Tree | Run dir (basename) |
|---|---|---|
| a1 | main `fd255444` | `2026-10-01_09-39-01_ppoc_smk_a1_main_s11` |
| a2 | main `fd255444` | `2026-10-01_09-40-08_ppoc_smk_a2_main_s11` |
| b | worktree, knob-off | `2026-10-01_09-41-13_ppoc_smk_b_wt_s11` |
| b2 | worktree, knob-off (repeat, to measure the worktree's own spread) | `2026-10-01_09-44-19_ppoc_smk_b2_wt_s11` |
| c | worktree, `KRABBY_PPO_HAND_DIAG=1`, 30 iterations | `2026-10-01_09-43-20_ppoc_smk_c_wt_diag_s11` |

**The pre-change tree is not bit-reproducible on every tag.** a1 vs a2: `agent.yaml` and `env.yaml` are identical,
and 112/116 tags are bit-identical through iteration 19. Four differ:
- `Loss/priv_reg`: max |Δ| 8.6e-7, from iteration 2;
- three wall-clock tags: `Perf/collection time`, `Perf/learning_time`, `Perf/total_fps`.

So (b) is judged against that spread, not against bit identity.

| Comparison | agent.yaml | env.yaml | Pre-existing tags identical through it 19 | Differing tags (max \|Δ\|) |
|---|---|---|---|---|
| a1 vs a2 (spread) | identical | identical | 112/116 | priv_reg 8.6e-7; Perf ×3 |
| a1 vs b | identical | 1 diff: `scene.robot.spawn.usd_path` (tree-local path) | 111/116 | priv_reg 1.85e-6; hist_latent 5.96e-8; Perf ×3 |
| a2 vs b | — | — | 111/116 | priv_reg 1.88e-6; hist_latent 5.96e-8; Perf ×3 |
| **a2 vs b2** | identical | 1 diff: `usd_path` | **113/116** | **Perf ×3 only** |
| a1 vs b2 | — | — | 112/116 | priv_reg 8.6e-7 (= the a1–a2 spread); Perf ×3 |
| b vs b2 (worktree spread) | — | — | 132/137 | priv_reg 1.88e-6; hist_latent 5.96e-8; Perf ×3 |

- **The `usd_path` difference** is the same asset seen from two trees: `cmp` of the main and worktree
  `assets/crab.usda` reports them byte-identical.
- **`Loss/priv_reg` and `Loss/hist_latent`** are the history encoder's outputs. They are nondeterministic run to
  run in both trees:
  - b and b2, the same worktree code, differ by exactly the amounts b and the a-runs differ by;
  - b2 reproduces a2 bit for bit on every non-timing tag.

  Neither tag can feed training here. `Loss/priv_reg_coef` is 0.0 through iteration 19, and the history encoder is
  not in the acting path. `hist_latent` is one value, from the iteration-0 dagger update, carried for 20
  iterations; its difference is 1 float32 ULP at 0.572.
- **Every reward, loss, value, entropy, std, metric and termination tag is bit-identical** a↔b through iteration 19.
- **(b) adds 21 tags:** 19 `PPO/*`, plus `Metrics/base_parkour/track_ratio_mean_norsi` and
  `track_ratio_move_mean_norsi`.

**(c) hand diagnostic ON vs (b)** (`cmp_b_c.txt`):
- `agent.yaml` differs only in `algorithm.hand_diag_vy_obs_col: 9` and `max_iterations` 30; `env.yaml` is
  identical.
- 132/137 pre-existing tags are identical through iteration 19. The 5 that differ:
  - `Loss/priv_reg`, 1.6e-6, inside the spread above;
  - the 3 `Perf/*` tags;
  - `Train/mean_episode_length/time`, which is indexed by wall-clock seconds, not iteration: its values agree
    wherever the step indices coincide.
- `hist_latent` is identical b↔c.
- New tags appear at iterations 0 and 25 only (the 25-iteration cadence):
  - `PPO/hand_grad_cos`: −0.178 at 0, +0.061 at 25;
  - `PPO/hand_adv_mean_{l,r}`;
  - `PPO/hand_frac_{l,r}`: 0.33/0.47 at 0.
- Log: `[KNOB] KRABBY_PPO_HAND_DIAG=1 consumed` and `[PPO] construction: hand_diag_vy_obs_col=9, …`.

**Verdict: PASS.** The worktree, knob-off and with the hand diagnostic on, matches the pre-change tree on every
pre-existing tag through iteration 19, within the pre-change tree's own run-to-run spread. Bit-identically so on
every tag but the two history-encoder tags and the wall-clock tags.

## 2. Knob-on smokes (20 its, 64 envs, seed 11, worktree) — all PASS

| Chain (run dir basename) | Consumed lines | agent.yaml / env.yaml dose | `PPO/effective_std` it 0 → 19 | verdict wiring read at it 19 | Driver audit |
|---|---|---|---|---|---|
| `C_lr__A` (`09-45-10_ppoc_smk_on_C_lr__A_s11`) | 6/6 (+ `KRABBY_ENTROPY_COEF`) | entropy 0.0025; `noise_std_type: soft`, init 0.8; bound 1.1, coef 0.001, ramp [1e-4, 1e-3, 500]; hand col 9 | 0.80000 → 0.7989 | PASS | 21/21 |
| `C__A` (`09-46-00_ppoc_smk_on_C__A_s11`) | 5/5 (no hand diagnostic: one direction) | as A, `hand_diag_vy_obs_col` absent | 0.80000 → 0.7976 | PASS | 17/17 |
| `C_lr__B` (`09-46-48_ppoc_smk_on_C_lr__B_s11`) | 6/6 | as A, plus `rewards.reward_action_rate.weight: 0.0` | 0.80000 → 0.7978 | PASS (dump check action_rate_w 0.0 ok) | 22/22 |
| `C_lr_rsi_st__A` + `KRABBY_RSI_CAM_STAGE_DEBUG=1` (`09-47-39_…`) | 7/7 incl. `KRABBY_RSI_CAM_STAGE=1` | as A; `events.rsi_reference_reset.params`: `stage_cam: true`, `stage_cam_debug: true`, fraction 0.2, source all | 0.80000 → 0.7987 | PASS | 25/25 |
| `C_lr_rsi__C0` + `KRABBY_RSI_CAM_STAGE_DEBUG=1` (`09-48-52_…`) | hand diagnostic; no `RSI_CAM_STAGE` line (`"0"` is off) | no `stage_cam` param in env.yaml (only the debug flag) | 1.5 (C0) | n/a (C0) | 13/13 |

- **`Loss/action_bound` is logged on every bound arm** (TB tag present from iteration 0; `Mean action_bound:`
  printed, the audit's "the bound loss ran" check).
  - Its value is 0.0 through iteration 19: the rollout means are well inside ±1.1 at init (median |μ| 0.08–0.15,
    `sat15` 0). The hinge is zero inside the bound by design, so this is the expected value, not a wiring gap.
    Whether the bound binds is the iteration-1000 dose read's question.
- **`Loss/action_bound_coef` ramps by update counter:** 1.000e-4 at iteration 0, 1.036e-4 at iteration 1 (counter
  2: the iteration-0 dagger update also advances it), then +1.8e-6 per update to 1.360e-4 at iteration 19. That is
  the CHARTER §4.3 behaviour.
- **B's executed action rate stays readable:** `Episode_Reward/reward_action_rate` is 0.0 at every iteration,
  while `PPO/exec_action_rate` is still logged (3.87 → 3.83).

**Staging smoke (`C_lr_rsi_st__A`, staging ON).**
- **Four tagged lines** were printed, one per reset that contained RSI envs: the initial reset (`n_rsi=12` of 64),
  then three single-env RSI resets.
- **Every line read `stage=on target_err=0 hist_err=0`:**
  - `target_absmax` 3.14159 (clamped at π while `bank_absmax` is 3.17), with `hist_absmax` 1;
  - in one line the frame was below π: `target_absmax` 2.58967, `hist_absmax` 0.824319 = 2.58967/π.
- **Non-RSI resets print nothing.** Episodes ended in at least nine distinct iterations (`Train/mean_episode_length`
  changes at 1, 4, 5, 11, 14–18), but tagged lines appear in only four. This also follows from the code: a reset
  with `n_rsi = 0` prints nothing.

**Staging OFF (`C_lr_rsi__C0`, `KRABBY_RSI_CAM_STAGE: "0"`).**
- Five tagged lines, all `stage=off target_absmax=0 hist_absmax=0`, with `bank_absmax` 2.26–3.18; the
  `target_err`/`hist_err` values equal the frame speeds.
- Same seed, same draws: the first three lines' `n_rsi` and `bank_absmax` equal the staging-ON run's.

## 3. Step-4 smokes (CHARTER §6; plan step 5) — all PASS

**`C_lr__A_fixed`** (`09-49-43_ppoc_smk_s4_C_lr__A_fixed_s11`; consumed 10/10; audit 29/29; the verdict wiring
read, including `step4a_wiring_fails`, PASSes at it 19).
- **Policy LR:** `PPO/lr_applied` = 1e-4 at all 20 iterations, and `Loss/learning_rate` = 1e-4 at all 20.
- **In the optimizer itself:** the checkpoint's policy optimizer has one param group, lr 1e-4.
- **History encoder:** `PPO/hist_lr` = 3e-4 at all 20 iterations, and the checkpoint `alg_state`'s
  `hist_encoder_optimizer` has lr 3e-4.
- **`PPO/kl_stop`** is present at every iteration. It fired at iterations 15, 16, 18 and 19
  (`PPO/epochs_done` 4, 2, 4, 4; 5 elsewhere).
- **Log:** `[PPO] construction: schedule=fixed, …, kl_stop=0.02, hist_encoder_learning_rate=0.0003, …, learning_rate=0.0001`.

**`C_lr__A_epochkl`** (`09-50-34_ppoc_smk_s4_C_lr__A_epochkl_s11`; consumed 9/9; audit 27/27; wiring PASS;
`check_lr_epochkl.txt`).
- **Test:** for each iteration, every LR path that changes at most once per epoch was enumerated: 3^5 epoch-end
  choices of /1.5 (floor 1e-5), ×1.5 (cap 1e-2) or no change, starting from the previous iteration's end LR. The
  check asks whether some path reproduces both `PPO/lr_applied` (the mean of the 5 epoch LRs) and the end LR
  (`Loss/learning_rate`), to a relative 1e-6.
- **Result:** 20/20 iterations are consistent.
  - Iteration 0 needs 5 changes: 3e-4 → 3.95e-5, one /1.5 per epoch, applied mean 1.563e-4.
- **The test has power:** run on the adaptive `C_lr__A` smoke, whose LR changes per minibatch, 20/20 iterations are
  inconsistent.
- **Descriptive:** under desired KL 0.003 the LR reached the 1e-5 floor by iteration 2 and then oscillated between
  the floor and 1.5× the floor. This is relevant to the §4.8 LR-floor read if 4a runs.

**1,024 envs through the driver's command builder** (`driver_train_smoke.py C_lr__C0_1024_s11 20`; run
`09-51-53_ppoc_smk_drv_C_lr__C0_1024_s11_s1`; train log `drvcamp/logs/…_train.log`; audit 11/11).
- **Command:** `_train_num_envs` → 1024, so the driver's command carried `--num_envs 1024`. The train log opens
  with the driver's `[KNOB] KRABBY_NUM_ENVS=1024 consumed`, and Isaac reports `Number of environments: 1024`.
- **Dumped config:**
  - `env.yaml`: `scene.num_envs: 1024`;
  - `agent.yaml`: `num_mini_batches: 16`, `num_steps_per_env: 24`, `num_learning_epochs: 5`;
  - total timesteps 491,520 = 20 × 1024 × 24.
- **Throughput:** 1.54 s/iter (collection 1.23 s, learning 0.31 s; 15,916 steps/s).

**One A-head stage fork as in P1** (reference `P1_C_lr__A_s16`, both stages through the driver path):
- **t3:** stage 0, 20 its from scratch, run `09-52-59_…_t3`.
- **c4:** stage 1, 20 its resumed from t3's `model_19.pt`, run `09-53-58_…_c4`.
- **Audits:** 21/21 on each.

| Read | t3 head (`model_19`) | c4 (resumed) | Expected if it restarted |
|---|---|---|---|
| σ parameter | `std_raw` only (no `std`), soft σ mean 0.7960088 | loaded with `strict=True` (no error); `PPO/effective_std` at its first iteration 0.7960086 | 0.8000 (re-initialised) |
| bound ramp | `Loss/action_bound_coef` 1.360e-4 at it 19 (counter 21) | 1.378e-4 at its first iteration (counter 21) → 1.738e-4 at it 38 | 1.000e-4 |
| alg_state | counter 21, LR 1e-5 | `model_38` counter 42 (21 + 20 updates + 1 dagger), LR resumed from 1e-5 (lr_applied 5.5e-5 in its first iteration) | counter 21, LR 3e-4 |

Note: the resumed window re-runs iteration index 19 (`Learning iteration 19/39`; final `model_38`). This is
existing rsl_rl resume numbering and was not changed.

**Verdict: PASS.** σ continues, the bound ramp continues rather than restarting, and `alg_state` is carried.

## 4. Rung 0b (CHARTER §5) — decision: STAGING OFF

**Command:** `CODE_ROOT=<worktree> scripts/rung0b.sh`.
- **Eval script:** the worktree's own `eval_crab_hex_gait.py`, with
  `PYTHONPATH=<wt>/parkour/parkour_tasks:<wt>/parkour`.
- **Head:** X02, `2026-09-29_01-52-56_seed3_X02_s1_A15pB/model_2999.pt`, named in the side bisect's
  `heads.json` and `state.json` (chain C s3).
- **Environment:** X02's driver-stamp env (`side22_25__omni/seed001/2026-09-29_07-00-57`).
- **Runs:** one process per row, 400 episodes per row.
- **Output:** `parkour/outputs/omni_plan/ppo_construction/rung0b/`, with `rung0b_read.json` and the queue lines in
  `queue.log`.

**Fix to `rung0b.sh` / `rung0b_read.py` (a clear bug, fixed before any decision).**
- **The bug:** both scripts accepted a stamp only if it held `complete.json`. `report.write_run` writes that marker
  for category-suite rows only. The rung-0b rows are plain `schedule` rows (as the plan requires), so the first row
  (`rung0b_rsi_on_e1__omni`, 09:55) produced a full stamp (400 raw npz) and was refused as "no complete stamp".
- **The fix:**
  - `rung0b_read.stamp_complete()` accepts `complete.json`, or a `run_meta.json` whose `n_steps_per_env` count
    equals the raw `episode_*.npz` count;
  - `rung0b.sh` requires exit code 0 and that same check.
- **Tests:** added to `tests/unit/test_ppo_construction_tooling.py`:
  - `test_rung0b_finds_plain_schedule_stamps_without_complete_json`;
  - `test_rung0b_rejects_a_partial_schedule_stamp`;
  - `test_rung0b_sh_uses_the_reader_completeness_check`.

  The file now runs 25 passed.
- **The probe row's stamp is used as is:** same eval code; the fix touched only the bookkeeping scripts. The other
  7 rows were then run by the fixed script.

**Staging verification** (the env's tagged lines; `rung0b_read.py` status OK on all 8 rows):
- **RSI staging-on rows:** first reset `n_rsi=400 stage=on target_err=0 hist_err=0 target_absmax=3.14159
  hist_absmax=1`, with `bank_absmax` 3.273 (seed 1) and 3.271 (seed 2). Every later line also has errors of 0.
- **RSI staging-off rows:** `n_rsi=400 stage=off target_absmax=0 hist_absmax=0`, with `bank_absmax` 3.273 / 3.271.
- **FRAC-0 rows:** no tagged line; only the `[KNOB] … NOT consumed: RSI is off` lines.
- **Distinct pairs:** the RSI off/on raw digests differ in both seeds.

| Row | n | falls in [0, 2) s | fall share | frame speed (median) | along-command speed [0, 2) s (median) |
|---|---|---|---|---|---|
| frac0_off e1 (normal) | 400 | 0 | 0.000 | 0.000 | +0.202 |
| frac0_on e1 | 400 | 0 | 0.000 | 0.000 | +0.202 |
| rsi_off e1 | 400 | 1 | 0.0025 | 0.282 | +0.186 |
| rsi_on e1 | 400 | 1 | 0.0025 | 0.282 | +0.182 |
| frac0_off e2 (normal) | 400 | 0 | 0.000 | 0.000 | +0.195 |
| frac0_on e2 | 400 | 0 | 0.000 | 0.000 | +0.195 |
| rsi_off e2 | 400 | 2 | 0.005 | 0.273 | +0.182 |
| rsi_on e2 | 400 | 0 | 0.000 | 0.273 | +0.181 |

Every RSI frame is +y (source 1, along the lat_l command).

**Rule** (seed-pair spread 0.0025):
- **(a) a deficit exists:** fails in both seeds. The deficit (RSI off − normal) is 0.0025 and 0.005, below 0.04.
- **(b) staging removes at least half of it:** fails in seed 1 (on − normal = 0.0025 > ½ × 0.0025) and holds in
  seed 2.

**Decision: STAGING OFF.** C_lr_rsi runs as the `C_lr_rsi__C0/__A/__B` chains, identical to X16's definition.
Descriptive: an RSI start from a fast-gait frame costs X02 essentially nothing in [0, 2) s, in falls or in speed.
The along-command speed is 0.18 against 0.20 from a normal spawn.

## 5. Rung 1 throughput probe (CHARTER §6; plan step 2 item 6) — 4b NOT triggered

**Runs:** `rung1.sh N 200`, direct `train.py` launches (not the driver).
- **Config:** the worktree, the `C_lr__C0` overlay (hand diagnostic included, as the arm runs), base `1b`, seed 11,
  200 iterations.
- **Order:** ascending, one Isaac process at a time, each under a named `systemd-run` scope (MemoryMax=48G) with
  `timeout -s TERM 3600`.
- **Window:** all in one window, 10:03–10:27; the 256-env run is the same-window baseline.
- **Sampling:**
  - nvidia-smi every 2 s: the train process's own GPU memory, and the device total, which includes about 0.4 GB
    of desktop;
  - the scope cgroup's `memory.peak` for peak RAM.
- **s/iter** = `Perf/collection time` + `Perf/learning_time`, averaged over iterations 10–199.
- **Read:** `rung1_read.py`; the table is also in `rung1_table.md`.

| Envs | s/iter mean (median) | Collection s | Learning s | Steps/s | × 256-env s/iter | Peak VRAM, process / device (MiB) | Peak RAM (GiB) | NaN | Episode fall share at it 199 (`crab_failure_flat`) | Mean reward at it 199 |
|---|---|---|---|---|---|---|---|---|---|---|
| 256 | 1.312 (1.309) | 1.012 | 0.300 | 4,682 | 1.00 | 4,746 / 5,183 | 6.1 | no | 0.586 | 9.1 |
| 512 | 1.419 (1.411) | 1.110 | 0.308 | 8,662 | 1.08 | 4,970 / 5,405 | 6.2 | no | 0.628 | 20.6 |
| 1,024 | 1.583 (1.575) | 1.267 | 0.316 | 15,529 | **1.21** | 5,378 / 5,813 | 6.4 | no | 0.289 | 28.1 |
| 2,048 | 2.143 (2.140) | 1.818 | 0.325 | 22,936 | **1.63** (> 1.5: stop) | 6,226 / 6,660 | 6.8 | no | 0.070 | 38.9 |
| 4,096 | not run (stop rule at 2,048) | | | | | | | | | |

- **Stop rule:** 2,048 envs is the first size above 1.5× the 256-env s/iter, so 4,096 was not launched.
- **No failures:** no size hit 10 GB of VRAM, and none had a NaN or non-finite tag, an OOM or a non-zero exit.
- **Run dirs** (worktree log root): `2026-10-01_10-03-20_ppoc_rung1_256`, `…_10-08-14_ppoc_rung1_512`,
  `…_10-13-38_ppoc_rung1_1024` and `…_10-19-32_ppoc_rung1_2048`.
- **Falls and reward are descriptive only.** At an equal iteration count a bigger batch has seen more data.

**4b trigger (CHARTER §6: 1,024 envs ≤ 1.15× the 256-env s/iter and ≤ 10 GB): NOT met.**
- The VRAM half passes: 5.4 GB.
- The time half fails: 1,024 envs take 1.583 s/iter, 1.21× the 256-env 1.312 s/iter.
- The 20-iteration driver-path smoke at 1,024 envs (§3) measured 1.54 s/iter, which agrees.
- Collection time, not learning time, grows with the env count (learning 0.30 → 0.33 s).

## 6. CPU tests after the fix

From the worktree, with `PYTHONPATH=$WT/parkour:$WT/parkour/parkour_tasks`:
- **These five files pass: 213 tests.**
  - `test_ppo_construction_tooling.py` (25, including the 3 new rung-0b tests);
  - `test_ppo_construction_knobs.py`;
  - `test_rsi_cam_stage.py`;
  - `test_crab_hex_omni_motion_knobs.py`;
  - `test_crab_hex_omni_exposure.py`.
- **`test_crab_hex_omni_gate.py`: 163 pass, 1 fails.** Neither problem is a worktree code defect.
  - **Six fixture tests need logs the worktree lacks.** The `track_ratio` fixture tests need
    `tests/unit/fixtures/omni_track_ratio/seed3_1{a,b}_track_ratio.log`. Those files are git-ignored (`*.log`), so
    they exist in the main checkout only. With copies placed temporarily (and removed afterwards), all six pass.
  - **One test fails on main too.**
    `test_train_without_the_new_kwargs_names_no_run_and_carries_no_overlay` also fails on the main checkout at
    `fd255444`. `tests/unit/conftest.py` exports `PYTHONPATH`, and the test's expected key set of the child
    environment does not allow for it.
