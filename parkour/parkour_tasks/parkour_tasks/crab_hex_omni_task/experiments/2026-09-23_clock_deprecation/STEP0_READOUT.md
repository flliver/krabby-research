# Step 0 readout: diagnostics and replay gate

*2026-09-23. Plan step 9 of `/home/nickmagus/.claude/plans/lets-explore-options-for-groovy-spindle.md`. CPU only (no
Isaac, no GPU). Replay report: `STEP0_REPLAY.json` (same directory). Reading 3 uses the campaign's standalone D00s,
`resolution.36e5a8af7000.json` (inputs E1/E2, acceptance ok, null all-TIE 0.729 inside [0.616, 0.98]). That file was
written by another session while this readout was in progress.*

> **2026-09-25 correction.** The replay gate used the wrong action clip and overstated `action_rate`. The corrected
> RG1/RG2/RG3 are in the addendum at the end. The original numbers below are left exactly as recorded.

## Verdicts (pre-registered rules)

| Item | Result |
|---|---|
| Reading 1 (gating) | **PASS** |
| Reading 2: do the heads follow the observed clock? | **Yes.** `offset_R` is 0.957–0.995 on every moving category of E1 and E3 (p < 1e-60). |
| Reading 3: dependence | **Both heads are clock-dependent**, by all three routes: survival drop, F/F_mt/T WORSE, and tripod down ≥ 50% |
| E4 and E5 both collapse (below the c6 floors) | **Yes.** E4: diag 0.28 < 0.59. E5: fwd 0.41 < 0.72 and diag 0.41 < 0.59. |
| Reading 4: does cadence follow the command channel? | **No.** Hidden `cadence_ratio` is 1.31–2.11 and never inside [0.9, 1.1]. |
| RG1 | **FAIL.** 3 of 18 judged triples fail; 6 more triples were already reversed clock-held. |
| RG2 | **FAIL.** Rise +12.3 pts (limit 10) and 35.4% free (limit 25%). |
| RG3 | **Does not fire.** Stand share is 15.9%, below the 30% threshold. |

## Stamps

Prefix: `parkour/logs/rsl_rl/gait_eval/omni_phases/2026-09-23_clock_deprecation/`. Every stamp is `suite__omni`,
seed001, with 1000 episodes.

| # | Head | Clock | Stamp | Completion |
|---|---|---|---|---|
| E1 | X08:c6 model_7996 | visible | `step0_E1_X08-c6_visible/.../2026-09-23_21-23-41` | 0.897 |
| E2 | X08:c4 model_4498 | visible | `step0_E2_X08-c4_visible/.../2026-09-23_21-31-11` | 0.822 |
| E3 | X08:t3 model_2999 | visible | `step0_E3_X08-t3_visible/.../2026-09-23_21-38-11` | 0.826 |
| E4 | X08:t3 model_2999 | hidden | `step0_E4_X08-t3_hidden/.../2026-09-23_21-45-11` | 0.825 |
| E5 | X08:c6 model_7996 | hidden | `step0_E5_X08-c6_hidden/.../2026-09-23_21-52-41` | 0.788 |
| E6 | statue (`--zero-actions`) | visible | `step0_E6_statue/.../2026-09-23_21-59-41` | 1.000 |
| E7 | 1c creeper `2026-09-17_00-13-40/model_6997` | visible | `step0_E7_creeper-1c/.../2026-09-23_22-07-41` | 0.864 |

**Match with MQ's stamps.** E1 and E2 are identical to MQ's `reeval_X08-c6` and `reeval_X08-c4` stamps (the bb788393
D00s inputs) on all 154 shared category medians. The plan's verification line expected agreement only within D00s R, so
the match is closer than required.

## Reading 1: knob sanity (gating): PASS

| | `env_vars` HIDE | `terrain_source` | max \|clock_obs\| | Rows with clock_obs ≠ 0 (pre-termination) | max \|clock_obs − (sin, cos)(clock_phase)\| | Episodes whose phase advances |
|---|---|---|---|---|---|---|
| E1 | 0 | raycast_mesh | 1.0 | 746807 / 746807 | 8.8e-8 | 984 |
| E2 | 0 | raycast_mesh | 1.0 | 729445 / 729445 | 8.8e-8 | 991 |
| E3 | 0 | raycast_mesh | 1.0 | 728265 / 728265 | 8.8e-8 | 988 |
| **E4** | **1** | raycast_mesh | **0** (all rows, including post-termination) | **0 / 736598** | — | 992 |
| **E5** | **1** | raycast_mesh | **0** (all rows, including post-termination) | **0 / 732479** | — | 983 |
| E6 | 0 | raycast_mesh | 1.0 | 775000 / 775000 | 8.8e-8 | 900 |
| E7 | 0 | raycast_mesh | 1.0 | 746505 / 746505 | 8.8e-8 | 900 |

Under HIDE the env keeps integrating `clock_phase` for telemetry, so offset_R and cadence_ratio are still defined for
E4 and E5.

## Reading 2: phase follower

The table gives `offset_R` with its Rayleigh p in brackets. n = 100 (fwd, back, diag) or 200 (side, turn, move_turn).
The 99% null is about 0.21 at n = 100 and about 0.15 at n = 200.

| Category | E1 c6 | E3 t3 | E4 t3 hidden | E5 c6 hidden | E2 c4 |
|---|---|---|---|---|---|
| fwd | **0.987** (2e-71) | **0.978** (8e-68) | 0.130 (0.19) | 0.047 (0.80) | 0.986 |
| side | **0.974** (8e-134) | **0.958** (6e-123) | 0.053 (0.58) | 0.034 (0.79) | 0.968 |
| back | **0.976** (3e-67) | **0.957** (3e-61) | 0.091 (0.44) | 0.060 (0.70) | 0.961 |
| diag | **0.967** (4e-64) | **0.968** (6e-64) | 0.064 (0.66) | 0.051 (0.77) | 0.979 |
| turn | **0.995** (1e-153) | **0.995** (9e-154) | 0.061 (0.47) | 0.070 (0.38) | 0.995 |
| move_turn | **0.989** (1e-146) | **0.987** (2e-144) | 0.099 (0.14) | 0.013 (0.97) | 0.986 |

With the clock visible, both heads are phase-locked to the observed clock. With it hidden, the offsets are at the null.

## Reading 3: dependence (E4 vs E3, E5 vs E1) at D00s R_eval

**Survival.** A head is flagged when the drop exceeds max(0.10, R_eval), where R_eval is the D00s one-sided 99%
survival guard. ✗ marks a flag.

| Category | E3 | E4 | Δ | E1 | E5 | Δ | Threshold | c6 floor |
|---|---|---|---|---|---|---|---|---|
| stand | 0.990 | 1.000 | +0.010 | 0.990 | 1.000 | +0.010 | 0.100 | 0.79 |
| fwd | 0.730 | 0.800 | +0.070 | 0.920 | **0.410** | **−0.510 ✗** | 0.110 | 0.72 |
| side | 0.920 | 0.990 | +0.070 | 0.975 | 0.995 | +0.020 | 0.100 | 0.77 |
| back | 0.780 | 0.910 | +0.130 | 0.920 | 0.810 | −0.110 ✗ (marginal) | 0.105 | 0.72 |
| diag | 0.660 | **0.280** | **−0.380 ✗** | 0.790 | **0.410** | **−0.380 ✗** | 0.140 | 0.59 |
| turn | 0.965 | 1.000 | +0.035 | 0.950 | 0.985 | +0.035 | 0.100 | 0.75 |
| move_turn | 0.665 | 0.640 | −0.025 | 0.750 | 0.645 | −0.105 ✗ (marginal; the binomial floor 0.108 would TIE) | 0.100 | 0.55 |

**Blocks** (z = mean δᵢ/R_eval,ᵢ; WORSE if z < −R):

| Block | R (D00s) | E4 − E3 | E5 − E1 |
|---|---|---|---|
| F (turn facing_closed, yaw_progress) | 0.966 | **−7.31 WORSE** (0.536→0.174, 0.072→0.023) | **−8.76 WORSE** (0.618→0.185, 0.083→0.025) |
| F_mt (move_turn, same keys) | 0.912 | **−4.50 WORSE** (0.174→−0.007, 0.026→−0.001) | **−4.55 WORSE** (0.235→0.055, 0.035→0.007) |
| T (ratio: fwd / side / back / diag) | 0.487 | **−3.70 WORSE** (0.558→0.316 / 0.478→0.390 / 0.524→0.401 / 0.521→0.478) | **−3.17 WORSE** (0.748→0.475 / 0.430→0.407 / 0.736→0.509 / 0.527→0.516) |
| RIG (context, not part of reading 3) | 0.330 | −3.15: steps part −13.79, support part +7.48 | −3.49: steps part −12.12, support part +5.15 |

The move_turn `ratio` guard also fell: 0.430→0.202 for E3→E4 and 0.424→0.170 for E1→E5.

**Per-category metrics.** Each cell is control → hidden. The per-m/rad column is `steps_per_m` on the translation
categories, `steps_per_rad` on turn and move_turn, and `steps_per_s` on stand.

| Category | Tripod E3→E4 | Tripod E1→E5 | support_feet E3→E4 | support_feet E1→E5 | Steps per m/rad E3→E4 | Steps per m/rad E1→E5 | swing_apex E3→E4 | swing_apex E1→E5 |
|---|---|---|---|---|---|---|---|---|
| stand | 0.542→0.474 (−13%) | 0.587→0.565 (−4%) | 3.05→3.31 | 3.14→3.45 | 3.17→4.17 /s | 3.08→3.75 /s | — | — |
| fwd | 0.501→0.246 (**−51%**) | 0.576→0.334 (−42%) | 2.95→3.23 | 3.02→3.21 | 24.3→46.8 | 16.3→35.4 | 0.109→0.059 | 0.116→0.058 |
| side | 0.435→0.375 (−14%) | 0.476→0.437 (−8%) | 3.46→3.64 | 3.56→3.62 | 31.7→47.0 | 32.9→48.2 | 0.059→0.047 | 0.064→0.047 |
| back | 0.514→0.342 (−34%) | 0.623→0.389 (−38%) | 2.98→3.25 | 3.04→3.29 | 27.2→48.9 | 17.0→36.4 | 0.108→0.062 | 0.129→0.058 |
| diag | 0.407→0.248 (−39%) | 0.396→0.367 (−7%) | 3.24→3.38 | 3.36→3.45 | 26.3→33.0 | 26.0→32.4 | 0.077→0.048 | 0.064→0.048 |
| turn | 0.590→0.238 (**−60%**) | 0.537→0.214 (**−60%**) | 3.01→3.24 | 3.12→3.28 | 48.1→162.3 | 44.2→139.1 | 0.117→0.056 | 0.094→0.050 |
| move_turn | 0.490→0.139 (**−72%**) | 0.504→0.238 (**−53%**) | 3.05→3.14 | 3.07→3.23 | 109.2→194.1 | 89.4→187.4 | 0.100→0.051 | 0.090→0.055 |

**Verdict: both heads are clock-dependent.**

- **E4 (t3).** diag survival −0.38; F, F_mt and T WORSE; tripod down ≥ 50% on fwd, turn and move_turn.
- **E5 (c6).** Survival falls on fwd (−0.51) and diag (−0.38), with marginal drops on back and move_turn; F, F_mt and
  T WORSE; tripod down ≥ 50% on turn and move_turn.

**Context (zero-shot only, not a trained arm).** With the clock hidden, both heads take 1.25–3.4× more steps per m or
rad, keep 0.06–0.30 more feet loaded, and lose 20–55% of `swing_apex`. These are the directions of RIG's steps-WORSE /
support-BETTER split and of the lift-loss flag.

## Reading 4: cadence source

| Category | E4 cadence_ratio | E4 gait_hz / clock_hz | E5 cadence_ratio | E5 gait_hz / clock_hz |
|---|---|---|---|---|
| fwd | 1.756 | 0.671 / 0.376 | 1.778 | 0.662 / 0.367 |
| side | 1.886 | 0.659 / 0.340 | 2.108 | 0.690 / 0.326 |
| back | 1.991 | 0.665 / 0.334 | 1.899 | 0.653 / 0.336 |
| diag | 1.652 | 0.653 / 0.386 | 1.747 | 0.659 / 0.376 |
| turn | 1.343 | 0.671 / 0.500 | 1.306 | 0.653 / 0.500 |
| move_turn | 1.343 | 0.671 / 0.500 | 1.306 | 0.653 / 0.500 |

For contrast, the visible heads read E1 0.985–1.016 and E3 0.989–1.007. None of the hidden values is in [0.9, 1.1], so
cadence does **not** follow the command channel.

The hidden heads settle into a single rhythm of about 0.65–0.69 Hz whatever the command. That is close to E1's stand
`gait_hz` of 0.60, the chatter rate. `step_rate_hz` (steps per second, summed over the six feet) rises by 0.4–1.5 Hz;
for example, fwd goes from 3.00 to 4.24 on E5. `gait_hz` comes from the tripod-difference FFT, so it is less reliable where tripod coherence has collapsed.

## Reading 5: saturation (MQ's `ANALYSIS_M1_saturation.md` method)

`actions` is the pre-clip deterministic mean over the pre-termination steps. Zero command means the settle and stand
holds; moving means every other hold. The median and p90 are taken on a 1-in-7 element subsample.

| Head | Holds | frac \|μ\| > 1 | frac \|μ\| > 2 | median \|μ\| | p90 \|μ\| | Sign flips per dim per s |
|---|---|---|---|---|---|---|
| E1 c6 | zero command | 0.956 | 0.903 | 7.5 | 15.1 | 2.27 |
| E1 c6 | moving | 0.962 | 0.916 | 8.5 | 17.0 | 1.96 |
| E3 t3 | zero command | 0.938 | 0.863 | 5.5 | 9.8 | 3.06 |
| E3 t3 | moving | 0.944 | 0.876 | 6.1 | 11.1 | 2.41 |
| E4 t3 hidden (context) | moving | 0.938 | 0.863 | 5.5 | 10.2 | 2.93 |
| E5 c6 hidden (context) | moving | 0.956 | 0.905 | 7.5 | 16.3 | 2.35 |
| E7 creeper (context) | moving | 0.977 | 0.941 | 8.4 | 13.3 | 0.79 |

**Method check.** On E1's first 400 episodes the numbers are zero command 0.957 and 2.24 flips/s, moving 0.964 and
1.87. MQ's X08:c6 re-eval row is 0.956 / 2.23 and 0.965 / 1.83.

Both healthy heads are bang-bang saturated. t3 is less saturated than c6, but it chatters more.

## Replay gate (`replay_clock_economy.py`, output `STEP0_REPLAY.json`)

**Setup.**

- Healthy heads E1 and E3; degenerates E6 and E7.
- Every rollout is priced under X08:c6's `params/env.yaml`. Clock-held is (1.0, 1.0) and clock-free is (0.001, 0.001).
- The clock phase is the recorded one in all 4000 episodes.
- Not replayed: `reward_stride_length` (0.5) and `penalty_tibia_deviation_in_stance` (−0.28).
- Known simplifications: `action_rate` has no action delay, and air time uses exact step timing.

### RG1: FAIL (15 of 18 judged triples pass)

| Triple | Held H / D | Held ratio | Free H / D | Free ratio | Floor | Result |
|---|---|---|---|---|---|---|
| E1 > E7 side | 0.881 / 0.806 | 1.093 | 0.308 / 0.326 | 0.945 | 1.093 | **FAIL**: order flips |
| E1 > E7 back | 1.158 / 1.113 | 1.040 | 0.541 / 0.686 | 0.788 | 1.040 | **FAIL**: order flips |
| E3 > E7 move_turn | 0.380 / 0.321 | — | −0.170 / −0.160 | undefined (D ≤ 0) | — | **FAIL**: order flips |
| E1 > E7 turn | 0.904 / 0.683 | 1.324 | 0.391 / 0.294 | 1.329 | 1.100 | pass |
| E3 > E7 turn | 0.924 / 0.683 | 1.353 | 0.338 / 0.294 | 1.148 | 1.100 | pass |
| E1 > E7 move_turn | 0.426 / 0.321 | — | −0.106 / −0.160 | undefined (D ≤ 0) | — | pass (on ordering only) |
| E1, E3 > E6, all 6 moving categories (12 triples) | 0.38–1.16 / −34.97 to −36.08 | undefined | 0.09–0.54 (move_turn −0.11 / −0.17) / about −35.2 to −36.3 | undefined | — | pass (on ordering only, audit item 11) |

**Reversed even with the clock held** (reported separately; not RG1 criteria):

- E1 > E7 on fwd (1.043 / 1.266) and diag (0.658 / 1.091);
- E3 > E7 on fwd (0.716 / 1.266), side (0.800 / 0.806), back (0.694 / 1.113) and diag (0.633 / 1.091).

All six stay reversed with the clock free.

**Why.** Per-term income under the clock-held economy:

- **action_rate.** This is what lets E7 out-earn the walkers even with the clock held. The walkers pay −0.82 to
  −1.05/s on moving holds; E7 pays −0.17 to −0.30/s. The walkers' cost is the bang-bang chatter of reading 5
  (1.96–2.41 flips/s against E7's 0.79).
- **Clock income.** E7 keeps clock_sched income of 0.22–0.38/s, close to the walkers' 0.32–0.41.
- **E6.** Its totals are dominated by `foot_idle` at −36/s.

### RG2: FAIL

Across 915,817 steps of healthy moving holds, the fraction with pre-clip total ≤ 0 goes from **0.231** clock-held to
**0.354** clock-free. That is a rise of **+12.3 pts** against a limit of 10, and 0.354 against a cap of 0.25, so both
conditions fail.

Held → free per category:

| Head | fwd | side | back | diag | turn | move_turn | stand (not in RG2) |
|---|---|---|---|---|---|---|---|
| E1 | 0.165→0.271 | 0.199→0.317 | 0.154→0.255 | 0.271→0.421 | 0.191→0.288 | 0.315→0.457 | 0.133→0.206 |
| E3 | 0.259→0.381 | 0.207→0.328 | 0.251→0.372 | 0.270→0.412 | 0.188→0.309 | 0.320→0.463 | 0.207→0.304 |

### RG3 (report only): does not fire

Share of positive income removed, pooled over E1 and E3:

| | stand | fwd | side | back | diag | turn | move_turn |
|---|---|---|---|---|---|---|---|
| Positive income held → free (/s) | 2.393→2.011 | 2.452→1.862 | 2.324→1.768 | 2.497→1.899 | 2.233→1.681 | 2.349→1.799 | 2.088→1.547 |
| Share removed | **15.9%** | 24.1% | 23.9% | 24.0% | 24.7% | 23.4% | 25.9% |

The stand share is below 30%, so the `stand.steps_per_s` BREAK-route flag does not fire. Audit item 9, whether the
stand guard may become a flag, does not arise.

## S1: what the plan offers

- **RG1 and RG2 both failed**, so the plan's failure menu applies:
  1. **Proceed** with wave 1 (the plan's default).
  2. **Go straight to wave 2.** This is also offered because E4 and E5 both collapse. Run X0 (step 11), then skip the
     A arms: record `ledger_cli decide {A6000,A4500,A3000} --choose skip` together before releasing the A6000 park.
     Wave 2 forks X0 s3's c6 head and reuses X0's stamps.
  3. **Add replacement pressure.** This is a re-budget option. New reward code moves code_sha and forces an X0 rerun,
     costing at least 5.2 GPU-h beyond the 47-h cap. It runs only on an explicit yes, with the cost restated at S1.
- **RG3 did not fire**, so it adds nothing to the menu.
- **Carried to S4 (step 14).** The pre-registered 2000-it mask-ramp variant (+1.6 GPU-h) applies "if wave-1 BREAKs
  coincide with a step-0 hidden collapse". The step-0 half of that condition is now on record.
- **Also owed at S1, not covered here.**
  - The `--q-spread-check` demotions. The D00s' frozen rule shows `per_category.demoted: {}`.
  - The D00s acceptance: ok, null all-TIE 0.729.

## Addendum 2026-09-25: replay gate re-run with the runner's action clip

**What was wrong.** `replay_clock_economy.py` took its action clip from the OF tool: `ACTION_CLIP = [1.0]*6 + [4.8]*12`
(`2026-09-16_1805_omni_formation/replay_reward_economy.py:100`). That is only the action term's own per-joint clip. The
runner clamps first:

- every step-0 head's `params/agent.yaml` has `clip_actions: 1.0`;
- `train.py:182`, `evaluation.py:121` and `eval_crab_hex_gait.py` pass it to `ParkourRslRlVecEnvWrapper`, which clamps
  before `env.step` (`vecenv_wrapper.py:135-136`);
- `reward_action_rate` reads the clamped `raw_actions` (`rewards.py:335`).

Reading 5 shows the heads are bang-bang saturated, so the extra ±4.8 headroom inflated every hip and knee flip. The
tool's docstring said it "clips exactly as the runner does". It did not.

**The fix.** The tool now reads the clip from each stamp's own head: `params/agent.yaml` beside `run_meta.checkpoint`.
`--action-clip` can pass it explicitly instead, and a missing agent.yaml is refused rather than defaulted.
`action_rate` applies the runner's clamp first and then the action term's own clip.
`tests/unit/test_crab_hex_clock_economy_replay.py` pins the clip to agent.yaml.

**Re-run.** The inputs are unchanged: E1 and E3 healthy, E6 and E7 degenerate, the X08:c6 economy, and the recorded clock
phase in all 4000 episodes. The clip resolves to 1.0 for all four stamps. E6's `--zero-actions` stamp uses the X08:c6
checkpoint, and its actions are all zero anyway. Output: `STEP0_REPLAY_2026-09-25_runner_clip.json`.
`STEP0_REPLAY.json` is left as it was.

**Control.** The pre-fix code re-run on the same inputs reproduces `STEP0_REPLAY.json` byte for byte, so the clip is the
only change.

### `action_rate` income (/s, both economies; original → corrected)

| Head | Moving holds | Overstatement | stand |
|---|---|---|---|
| E1 c6 | −0.84 to −1.05 → **−0.25 to −0.31** | 3.3–3.4× | −1.085 → −0.325 |
| E3 t3 | −0.82 to −1.04 → **−0.27 to −0.36** | 2.9–3.0× | −1.181 → −0.403 |
| E7 creeper | −0.17 to −0.30 → **−0.12 to −0.15** | 1.4–2.1× | −0.080 → −0.080 |
| E6 statue | 0 → 0 | — | 0 → 0 |

The walkers' extra `action_rate` cost over E7 falls from 0.65–0.88/s to **0.13–0.23/s**.

### Corrected verdicts

| Gate | Original (2026-09-23) | Corrected (2026-09-25) |
|---|---|---|
| RG1 | **FAIL.** 3 of 18 judged triples fail; 6 more were reversed with the clock held. | **FAIL.** 2 of 24 judged triples fail; none is reversed with the clock held. |
| RG2 | **FAIL.** 0.231 → 0.354 (+12.3 pts; 35.4% free) | **PASS.** 0.085 → 0.139 (+5.4 pts, limit 10; 13.9% free, cap 25%) |
| RG3 | Does not fire (stand 15.9%) | **Unchanged**, identical in every category |

RG3 cannot move: `action_rate` is a negative term, and RG3 prices positive income only.

### RG1 corrected: FAIL (22 of 24 judged triples pass)

| Triple | Held H / D | Held ratio | Free H / D | Free ratio | Floor | Result |
|---|---|---|---|---|---|---|
| E3 > E7 fwd | 1.363 / 1.338 | 1.019 | 0.783 / 0.788 | 0.993 | 1.019 | **FAIL**: order flips |
| E3 > E7 back | 1.369 / 1.163 | 1.176 | 0.790 / 0.736 | 1.073 | 1.100 | **FAIL**: ratio below the floor |
| E1 > E7 fwd | 1.781 / 1.338 | 1.331 | 1.181 / 0.788 | 1.498 | 1.100 | pass |
| E1 > E7 side | 1.465 / 0.857 | 1.709 | 0.892 / 0.377 | 2.364 | 1.100 | pass |
| E1 > E7 back | 1.902 / 1.163 | 1.635 | 1.285 / 0.736 | 1.745 | 1.100 | pass |
| E1 > E7 diag | 1.344 / 1.165 | 1.153 | 0.790 / 0.570 | 1.385 | 1.100 | pass |
| E1 > E7 turn | 1.621 / 0.797 | 2.032 | 1.108 / 0.409 | 2.708 | 1.100 | pass |
| E1 > E7 move_turn | 1.157 / 0.478 | — | 0.625 / −0.003 | undefined (D ≤ 0) | — | pass (on ordering only) |
| E3 > E7 side | 1.351 / 0.857 | 1.576 | 0.811 / 0.377 | 2.148 | 1.100 | pass |
| E3 > E7 diag | 1.235 / 1.165 | 1.060 | 0.687 / 0.570 | 1.206 | 1.060 | pass |
| E3 > E7 turn | 1.605 / 0.797 | 2.013 | 1.019 / 0.409 | 2.491 | 1.100 | pass |
| E3 > E7 move_turn | 1.034 / 0.478 | — | 0.485 / −0.003 | undefined (D ≤ 0) | — | pass (on ordering only) |
| E1, E3 > E6, all 6 moving categories (12 triples) | 1.03–1.90 / −34.97 to −36.08 | undefined | 0.49–1.29 / −35.15 to −36.26 | undefined | — | pass (on ordering only, audit item 11) |

Both failures are t3 against the creeper. On fwd, t3 barely out-earns E7 even with the clock held (ratio 1.019).
Removing the clock then costs t3 more than E7: 0.580/s against 0.550/s on fwd, and 0.579/s against 0.427/s on back.

### RG2 corrected: PASS

The step count is unchanged at 915,817. The fraction of steps with pre-clip total ≤ 0 goes from **0.085** clock-held to
**0.139** clock-free. That is a rise of **+5.4 pts** against a limit of 10, and 0.139 against a cap of 0.25.

Held → free per category, corrected:

| Head | fwd | side | back | diag | turn | move_turn | stand (not in RG2) |
|---|---|---|---|---|---|---|---|
| E1 | 0.050→0.077 | 0.067→0.124 | 0.044→0.065 | 0.091→0.163 | 0.052→0.074 | 0.126→0.201 | 0.033→0.041 |
| E3 | 0.119→0.182 | 0.079→0.144 | 0.109→0.163 | 0.110→0.191 | 0.053→0.086 | 0.146→0.225 | 0.050→0.065 |

**Method check.** Run with the old clip, the same breakdown script reproduces the original per-category table exactly.

### What this changes in the readout above

- **"Why" (RG1).** The `action_rate` bullet is withdrawn. It was the clip, not the chatter, that let E7 out-earn the
  walkers with the clock held. Corrected, E7 out-earns neither walker on any moving category.
- **Unaffected.** The clock-income bullet (E7 keeps clock_sched at 0.22–0.38/s) and the E6 bullet (`foot_idle` about
  −36/s) stand: neither term reads actions. Readings 1–5 do not use the replay.
- **S1.** The plan offers the same menu on an RG1 or an RG2 failure. RG1 still fails, so the menu still applied. Its
  basis is now narrower: 2 t3-vs-creeper triples, and no RG2 failure.
- **The OF tool has the same clip bug.** Its three heads (1a `2026-09-16_23-11-00`, 1b `2026-09-16_23-44-00`, 1c
  `2026-09-17_00-13-40`) all have `clip_actions: 1.0`. On its canary raws' moving holds it overstates `action_rate`
  2.4–2.5× for the walker, 2.8–2.9× for the mid head and 1.5–2.0× for the creeper. `run_candidates.py` also runs it, in advisory mode.
  Neither the tool nor its records (`replay_reward_economy.md`) were changed here.
