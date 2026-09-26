# Stroke formation: arm the stroke or centring term from iteration 0 of omni formation

A two-stage campaign. Stage 1 is a t3 gate from scratch. Stage 2 runs only for arms that pass it.

- **Design:** the user's, 2026-09-26. It follows the stroke/posture screen
  ([`../2026-09-26_stroke_posture/CLOSE.md`](../2026-09-26_stroke_posture/CLOSE.md)): neither term moved the side
  posture in a 4000-it fine-tune of W. The user chose to apply both terms during early training instead.
- **Who runs it:** the orchestrator, without asking, under the standing instruction. It push-notifies the user
  at each gate: the dose smoke, the t3 gate and the Stage-2 decision.
- **The machine-readable spec:** [`candidates.yaml`](candidates.yaml), Stage 1 only. Stage 2 gets its own yaml
  after the gate.
- **The readout:** `../tools/replay_stroke_posture.py pooled <arm_s3> <B_f_s3> <arm_s2> <B_f_s2> --json ...` on
  the six t3 `suite__omni` evals decides the gate (per-seed `readout --vs` for the unison flag).
  `income` on a t3 (clock-held) stamp needs `--econ-env-yaml <that run's own params/env.yaml>`: the
  default is W's clock-free economy, and `income` refuses the mismatch.

## The question

1. **Posture.** If the centring term (P) or the stroke term (S) is armed from iteration 0 of formation, does the
   lateral posture fail to form? On side holds, the body leads its six-foot centroid by about 0.23 m on W.
2. **Stride.** Does either arm lengthen side strides?

## Evidence: when the posture forms

The orchestrator's scoping pass (2026-09-26) measured side `body_offset` (m) across the omni lineages:

| Lineage | Point | Side `body_offset` |
|---|---|---|
| 1b | 2000 its | 0.133 |
| X08 (C1d) | end of t3 | 0.210 |
| Z (never clocked) | end of t3, s3 / s2 | 0.237 / 0.251 |
| B0 | s3 / s2 | 0.193 / 0.146 |
| W (head of record) | end of c6x | 0.231; the c6y fine-tune arms did not move it |
| sideways head (same plant) | end | 0.014 |

**Reading.**
- The posture forms within the first ~1000 iterations of side commands in every omni lineage.
- It is complete by the end of C1d's t3. That stage trains from scratch, with side commands from iteration 0 and
  the clock held.
- The sideways task never forms it.

So a fine-tune from W starts too late. The first 3000 iterations of t3 are where a term can act on the posture.

**Formation seed band ≈ 0.05 m** (B0 s3 vs s2). That equals P's bar. A single-seed arm cannot be read, so every
arm runs at both seeds.

## Design

### Stage 1 (this yaml)

**Chain `t3f`: one stage, `t3`.** It is C1d's `curriculum_turnfirst` t3, verbatim:
- preset 1a, from scratch, 3000 its;
- pool `stand:0.15, fwd:0.25, side_l:0.16, side_r:0.16, back:0.08, turn:0.2`, with its economy overlay;
- floor `category_survival`, `min_new 0.10`, `min_prev 0.30`.

**The clock is HELD.** No K: no `KRABBY_CLOCK_W`, `KRABBY_APEX_W` or `KRABBY_OMNI_CLOCK_HIDE`, as in X08:t3.

**The chain overlay also carries:**
- `KRABBY_OMNI_STROKE_W 0.001` and `KRABBY_OMNI_CENTRING_W 0.001`, so B_f inherits both knobs at eps;
- `KRABBY_OMNI_REWARD_SPLIT 1`, pinned by `require_env`.

The two eps knobs cannot be pinned, because each arm re-sets one of them.

**Arms.** Each differs from B_f in one weight (`overrides_chain`), with `control: baseline`.

| Arm | Knob | Dose |
|---|---|---|
| B_f | both at eps | — |
| S_f | `KRABBY_OMNI_STROKE_W` | 0.119 (final) |
| P_f | `KRABBY_OMNI_CENTRING_W` | 0.158 (final) |

**Doses are provisional until the t3 dose smoke.**
- The smoke runs the stroke_posture window-0 method from the baked C1d t3 head,
  `policy/1_t3/model_2999.pt`.
- It computes r = delivered ÷ replay-predicted: r_S on the fwd row, r_P on the side row.
- If r is in [0.5, 2], it rescales the dose by 1/r. Otherwise it STOPS.
- The final values are edited into `candidates.yaml` before the first launch. `candidates_sha256` is recorded at
  launch.

**Seeds: 3 and 2 for every arm, whatever the s3 verdict.** The driver option `pair_on: [always]` does this.
- The order is B_f s3, B_f s2, S_f s3, S_f s2, P_f s3, P_f s2.
- **Exception:** a floor or dead_walker REJECT at s3 is not paired. That arm has already failed the health
  criterion below, so its s2 run could not change the gate.

**The driver decides nothing.** Its rule has no block RIG, so there are no flags and no parks. Its lines are
records. Every run records its t3 suite eval.

### Stage 2 (after the gate)

Stage 2 runs for each passing arm AND for B_f:
- Fork each arm's t3 head, both seeds, through c4 → c5 → c6 with the clock held. The C1d overlays are verbatim.
- Then run c6x: K, clock-free.
- S steps down 0.15 / 0.14 / 0.13 / 0.117 across c4 / c5 / c6 / c6x. P holds its dose.

**Stage 2's decision** uses the stroke_posture c6y rule
([`../2026-09-26_stroke_posture/candidates.yaml`](../2026-09-26_stroke_posture/candidates.yaml)):
- blocks F, F_mt, T, RIG and LEN, with wave 2's guards;
- floors at W − 0.20, where a move_turn or diag floor parks and flags;
- LEN is decisive only at c6x.

### Budget

| Stage | GPU-h |
|---|---|
| Stage 1: 6 × t3 from scratch + suite eval | ≈ 8.3 |
| Dose smoke | ≈ 0.3 |
| Stage 2, per continuing arm (2 seeds × 9000 its), B_f included | ≈ +17 |

## The pre-registered t3 gate

The **orchestrator** decides the gate from the pooled s3 + s2 readout. The driver's lines do not count.

**How it is measured.**
- Each arm is compared with B_f.
- Deltas are pooled means over the two seeds of arm − B_f. Each seed's delta is the tool's point
  delta: a difference of row medians (of means for the tool's `MEAN_KEYS`).
- Band = max(pooled bootstrap MDE, |B_f s3 − B_f s2|) per key.

**P passes** if pooled Δ side `body_offset` ≤ −0.05 m.

**S passes** if both of these hold:
1. **Its dose landed:** `Episode_RewardCat` fwd delivered ≥ 0.5 × replay-predicted, at both seeds.
2. **It lengthened the stroke:** side `stance_stroke`, side `stroke_share` or fwd `stance_stroke` rose by more
   than its band (Δ > +band). A drop beyond the band is a finding, not a pass (decision 3 below).

**Outcomes reported for every arm (they do not gate):**
- side `stride_along` (reference effect +0.02 m);
- side `stance_stroke` (+0.010 m);
- side lead/trail asymmetry.

**LEN is reported only.** At t3 the cadence is clock-locked.

**Health: an arm must pass all of these to pass the gate.**
- Moving ratio ≥ 0.30 on fwd, side and back, at both seeds.
- The t3 floor at both seeds, with no floor or dead_walker REJECT.
- The unison flag is clear against B_f. It fires if tripod coherence falls below its seed band, or roll_rms
  collapses.

**STOP RULE.**
- If neither arm passes, stop and report to the user. Stage 2 does not run.
- Otherwise run Stage 2 for the passing arm(s) and B_f.

## Caveats

- **Resolution inputs.** D00s is computed from the clock campaign's E1/E2 suite evals, with X08-c6 as control.
  They exist only under the main checkout's `parkour/logs`. **Launch from the main checkout**, as the stroke_posture
  screen was. A launch from this worktree would halt `resolution D00s`. D00s never decides the gate.
- **Dead-walker abort.** A candidate s3 run carries the driver's dead_walker abort at its 1000:
  track_ratio_mean ≥ max(0.35, 0.75 × B_f s3's its-800–1000 mean). A dose that slows formation can trip it. That is
  a health failure with no s2 run.
- **B_f is not X08:t3.** It adds the eps knobs and the telemetry flag, and runs under a later `code_sha`. X08:t3's
  +0.210 m is context only.

## Pre-launch decisions (orchestrator, 2026-09-26, under the standing run-without-asking instruction)

The build verifier left four items open. Each is settled here with the conservative default.

1. **Dead-walker abort: kept, as documented under Caveats.** It is the driver's standing from-scratch safeguard,
   and the arms dose about 10 % of tracking income, so a healthy arm should clear 0.75 × B_f by its 1000. No
   driver change. An arm it kills fails the gate's health criterion and is reported.
2. **R4 carries over to t3: w_S 0.17 → 0.135.** The user's plan decision 6 binds the single stroke weight to fwd
   delivered ≤ 10 % of tracking income. The builder's 0.17 gave 12.6 % on X08:t3's E3. At 0.135:

   | Category | S delivered share | P delivered share (w_P 0.16) |
   |---|---|---|
   | fwd | 9.9 % | — |
   | side | 2.3 % | 9.4 % |
   | back | 7.8 % | — |
   | diag | 7.5 % | 8.9 % |
   | turn | 5.1 % | — |
   | move_turn | 1.4 % | — |

   Command: `replay_stroke_posture.py income <E3 suite stamp> --w-stroke 0.135 --w-centring 0.16
   --econ-env-yaml <X08:t3 params/env.yaml>`. The dose smoke's 1/r rescale still applies to both doses.
3. **S's stroke test is directional: lengthening only.** Stage 2 costs about 17 GPU-h per arm, and its purpose is
   longer side steps. A term that shortens the stroke gives no reason to continue. The screen's decision 9 was a
   direction-free manipulation check on a fine-tune; this gate also decides whether to continue.
4. **Launch from the main checkout** after fast-forwarding `feat/crab-hex-omni-task` to this branch, exactly as
   the stroke_posture screen did. The D00s inputs live only under the main checkout's `parkour/logs`.

## t3 dose smoke — PASS, doses rescaled (2026-09-26 17:32–17:45)

- **What ran.** `formation_dose_smoke.py`: 300 its of arm C's composed `t3f` env (both doses at 0.135 / 0.16,
  clock held, no K, the split flag, play keys ring / 0.6), forked from `policy/1_t3/model_2999.pt` (sha 03b87cd3…,
  checked against the manifest). Run dir: worktree
  `parkour/logs/rsl_rl/crab_hex_omni_flat_walk/2026-09-26_17-32-42_t3_dose_smokeC` (model_3298.pt).
- **Wrapper → replay parity: PASS** (300 steps × 16 envs). Max |diff|: pay_sum 1.2e-7, b 3.0e-7, pay 1.3e-7,
  o 6.0e-8.
- **r = delivered (`Episode_RewardCat`, post-clip, last 200 its) ÷ replay-predicted on E3 (X08:t3's suite eval,
  priced in X08:t3's own clock-held economy):**

  | Category | Stroke log/s | Stroke replay/s | r_S | Centring log/s | Centring replay/s | r_P |
  |---|---|---|---|---|---|---|
  | fwd | 0.0803 | 0.0708 | **1.13** | 0.0015 | 0 | — |
  | side | 0.0183 | 0.0160 | 1.14 | 0.0678 | 0.0668 | **1.01** |
  | back | 0.0648 | 0.0587 | 1.10 | 0 | 0 | — |
  | turn | 0.0184 | 0.0438 | 0.42 | 0 | 0 | — |

  diag and move_turn are not in t3's pool (replay only). Turn's 0.42 is the command-distribution effect the
  screen's window 0 recorded (training turn holds reach their target and the relay drops to 0); it sets no dose.
- **Dose rule** (r in [0.5, 2] → rescale by 1/r): w_S → 0.135 / 1.134 = **0.119**; w_P → 0.16 / 1.015 =
  **0.158**. Side's r_S (1.14) is in band. At 0.158, diag's replayed centring share is ≈ 8.8 % (≤ 10 %).
  w_S lands within 2 % of the screen's R4 dose on W (0.117).
- JSON: `parkour/outputs/omni_plan/stroke_formation/t3_dose_smoke/t3_dose_smokeC_r.json` (worktree).

