# POWER_PREVIEW: stroke/posture plan, step 8 (premise and power, 0 GPU)

Date 2026-09-26, worktree `feat/omni-stroke-posture` at `d716483b`. CPU only; no Kit, no GPU. The
machine-readable record is `power_preview.json`, next to this file.

## Verdicts

| Check | Value | Threshold | Verdict |
|---|---|---|---|
| Premise: W side `body_offset` | +0.231 m | ≥ 0.15 m | **PASS** |
| Premise: W side `stride_along` ÷ fwd | 0.0835 ÷ 0.2400 = 0.35 | ≤ 0.6 | **PASS** |
| (c) LEN power: S's +0.02 m side stride as Δ side `steps_per_m` | −6.20 steps/m | \|Δ\| > LEN per-member MDE 1.644 | **PASS** (caveat 1) |
| (d) side `body_offset` MDE | 0.017 m (seed term; bootstrap 0.006) | ≤ 0.05 m | **PASS** |
| (d) side `stride_along` MDE | 0.0034 m = 4.0 % of W | ≤ 25 % | **PASS** |
| (d) side `stance_stroke` MDE | 0.0030 m | ≤ 0.010 m | **PASS** |
| (d) S manipulation keys | side `stroke_share` ±0.033; fwd `stance_stroke` ±0.027 m | reported only | REPORT |

**No STOP fired.** The two caveats below (the LEN side-only case and the seed-dominated posture
keys) go to SA0.

## Inputs (resolved stamps)

All paths are under `parkour/logs/rsl_rl/gait_eval/omni_phases/`, each at `suite__omni/seed001/<stamp>`,
with 1000 episodes and `complete.json` present.

| Role | Run | Stamp |
|---|---|---|
| W (primary) | `2026-09-23_clock_deprecation_w2/seed3_X03_c6x_A15pB` | `2026-09-25_20-31-29` |
| Eval pair, control | E1 = `2026-09-23_clock_deprecation/step0_E1_X08-c6_visible` | `2026-09-23_21-23-41` |
| Eval pair, other | E2 = `2026-09-23_clock_deprecation/step0_E2_X08-c4_visible` | `2026-09-23_21-31-11` |
| Seed pair, s3 | X0x s3 = `2026-09-23_clock_deprecation_w2/seed3_X01_c6x_A15pB` (w2 ladder X01) | `2026-09-25_17-01-55` |
| Seed pair, s2 | X0x s2 = `2026-09-23_clock_deprecation_w2/seed2_X02_c6x_A15pB` (w2 ladder X02) | `2026-09-25_18-45-27` |

E1/E2 are exactly the w2 D00s `resolution_inputs`, and X01/X02 are exactly w2's recorded D00bs pair
(ledger `D00bs.evidence.pair = {s3: X01, s2: X02}`). The provisional LEN numbers therefore sit on the
same pairs as every other w2 block.

## Premise (W's eval, `replay_stroke_posture.py readout`)

| Row | `body_offset` | lead \|y\| | trail \|y\| | `stride_along` | `stance_stroke` | `stroke_share` | `lead_scuff_frac` |
|---|---|---|---|---|---|---|---|
| fwd | +0.047 | 1.114 | 1.155 | 0.240 | 0.171 | 0.870 | 0.265 |
| side | **+0.231** | 0.724 | 1.214 | **0.083** | 0.040 | 0.562 | 0.419 |
| side/l | +0.231 | 0.723 | 1.213 | 0.082 | 0.038 | 0.570 | 0.411 |
| side/r | +0.231 | 0.727 | 1.214 | 0.085 | 0.042 | 0.559 | 0.420 |
| diag | +0.173 | 0.796 | 1.183 | 0.115 | 0.063 | 0.675 | 0.412 |
| back | +0.049 | 1.111 | 1.144 | 0.214 | 0.150 | 0.870 | 0.245 |

This matches the pre-measured premise (+0.231, 0.083 against 0.240) to the printed digit. The heading
guard held (max \|wrap(ψ − (face − heading_err))\| = 3.0e-7). Both hands carry the posture equally
(+0.231 / +0.231).

## Power

### (a) Guard-level D00bs MDEs (w2 `notify.log` line 3, 2026-09-25 14:53 RESOLUTION D00bs)

The guards are one-sided at 99 %, with MDE = max(R_eval, R_seed) in steps/m.

| Guard | R_eval (E1 vs E2) | R_seed (\|X01 − X02\|) | MDE | Seed-dominated |
|---|---|---|---|---|
| `cat.side.steps_per_m` | 1.569 | 1.850 | **±1.850** | yes |
| `cat.diag.steps_per_m` | 2.024 | 3.205 | **±3.205** | yes |

These match the plan's step-8 text.

### (b) Provisional LEN block

LEN = {`cat.side.steps_per_m`, `cat.diag.steps_per_m`}, lower is better (`metric_sign` −1), 95 %
two-sided. R = max(R_eval, R_seed).

**How R_eval was computed.** The driver's own `resolution.py` ran on a scratch copy of w2's
`candidates.yaml`. The copy's only changes were:
- a `LEN` block added under `decision_rule.per_category.blocks`:
  `{level: 0.95, units: ratio, members: {side: [steps_per_m], diag: [steps_per_m]}}`;
- absolute `run_dir`s.

It used the same rule otherwise, B = 2000, rng 20260923, decisive_at = all seven categories, and E1
(control) vs E2. Acceptance was ok.

The run is a cross-check on itself: F 0.966, F_mt 0.912, T 0.487 and RIG 0.330 all reproduce w2's
recorded D00s exactly.

- R_eval,i (95 % half-width per member): side 1.3767 and diag 1.6227 steps/m.
- `blocks.LEN.R_eval` (the 95 % half-width of the bootstrapped z_LEN = mean_i(δ_i / R_eval,i)) =
  **0.6945**.

**How R_seed was computed.** `resolution.py` is eval-only, so the seed term comes from the driver's
D00bs path: `run_candidates.d00bs_evidence(s3_screen = X01, s2_screen = X02, resolution = the D00s
above, rule = the LEN-augmented rule, decisive_at = all seven)`. It uses w2 `state.json`'s recorded
screens, which are what `_write_d00bs` passes. The call reproduces w2's recorded D00bs block terms
exactly (F 1.897, F_mt 0.114, T 1.025, RIG 1.132); acceptance is ok and the construction check is
all-TIE.

The arithmetic (`resolution.block_seed_term`), with the signed delta = sign × (s2 − s3) and sign = −1:

| Member | s3 (X01) | s2 (X02) | Signed δ | R_eval,i | z_i = δ / R_eval,i |
|---|---|---|---|---|---|
| side `steps_per_m` | 33.605 | 35.454 | −1.8495 | 1.3767 | −1.3435 |
| diag `steps_per_m` | 26.890 | 23.685 | +3.2052 | 1.6227 | +1.9753 |

- z_pair = mean(z_i) = 0.3159. The members cancel, so the second term governs.
- rms(z_i) / √k = √((1.3435² + 1.9753²) / 2) / √2 = **1.1944**.
- **R_seed = max(0.3159, 1.1944) = 1.1944.**

**Result.**

| | Value |
|---|---|
| R_eval | 0.6945 |
| R_seed | 1.1944 (seed-dominated) |
| **R** | **1.1944** |
| Per-member MDE = R × R_eval,i (driver convention, `d00bs_evidence.member_mde`) | **side 1.644, diag 1.938 steps/m** |
| Eval-only per-member MDE (R_eval × R_eval,i) | side 0.956, diag 1.127 |
| Side-only MDE (diag unmoved, so z_LEN = z_side / 2 and δ_side ≥ 2·R·R_eval,i) | side 3.289 steps/m |

This is PROVISIONAL. Swap the seed term to B s3/s2 once those screens exist.

### (c) S's practical minimum effect as a Δ in side `steps_per_m` on W

W's own values:
- side: `steps_per_m` = 32.088 (the suite screen via `resolution.summarize_run_dir`), `stride_along` =
  0.08350 (the mean of per-episode means);
- fwd: 14.704 / 0.2400.

**Primary conversion (inverse law).** `steps_per_m` is steps ÷ net body distance, so at a fixed gait
it scales as 1/stride. Holding k = SPM × stride fixed at W's side value gives k = 32.088 × 0.08350 =
2.6792. Then

  SPM' = 2.6792 / (0.08350 + 0.02) = 2.6792 / 0.10350 = 25.887, so **Δ = −6.20 steps/m**

(equivalently Δ = −SPM × 0.02 / (s + 0.02) = −32.088 × 0.02 / 0.1035).

**Sensitivity**, all on W:

| Conversion | Δ side `steps_per_m` | > per-member MDE 1.644? | > side-only MDE 3.289? |
|---|---|---|---|
| Inverse law, k fixed (primary) | −6.20 | yes | yes |
| Within-W per-episode linear slope (−191.4 per m, n = 200) | −3.83 | yes | yes |
| Within-W log-log elasticity (b = −0.476), 32.088 × (0.1035 / 0.0835)^b | −3.12 | yes | **no** |
| fwd↔side chord (14.704 − 32.088) / (0.2400 − 0.0835) = −111.0 per m | −2.22 | yes | **no** |

The two per-episode fits are attenuated by per-episode stride noise (regression dilution), so they
lower-bound the true response.

Inverted, the LEN per-member MDE equals a side stride gain of **+0.0045 m**, and the side-only MDE a
gain of +0.0095 m, both under S's +0.02 m.

**Verdict: PASS.** Every conversion exceeds the LEN per-member MDE.

- **Caveat 1.** If S moved side only and left diag unchanged, the block would need 3.29 steps/m, which
  the two conservative conversions miss. It is still met by the primary, and the side guard's ±1.850
  is exceeded by every conversion. Diag is half-lateral and carries the same posture (+0.173), so
  S should move both. Record LEN as mechanism evidence only (decision 5).

### (d) Readout-key MDEs (`readout --vs`, env-clustered paired bootstrap, B = 2000, 95 % centred half-width)

The columns are:
- **Eval MDE:** the bootstrap half-width of E1 vs E2. E1 and E2 are different heads, so their Δ is
  real and only the half-width is noise, as in D00s.
- **Seed MDE:** the bootstrap half-width of X0x s3 vs s2.
- **\|Δ seed\|:** the seed pair's own delta, the readout analogue of R_seed.
- **Effective:** max(eval MDE, seed MDE, \|Δ seed\|), the single-seed resolution under the driver's
  max(R_eval, R_seed) rule. The STOP checks are read on it; each also passes on the bootstrap MDE
  alone.

| Key | W | Eval MDE (Δ E1 − E2) | Seed MDE (Δ s3 − s2) | Effective | Rel. to W | Threshold | Verdict |
|---|---|---|---|---|---|---|---|
| side `body_offset` (m) | +0.2309 | 0.0057 (+0.0065) | 0.0053 (**+0.0171**) | **0.0171** | 7.4 % | ≤ 0.05 m | **PASS** |
| side `stride_along` (m) | 0.0835 | 0.0030 (−0.0019) | 0.0034 (+0.0008) | **0.0034** | **4.0 %** | ≤ 25 % | **PASS** |
| side `stance_stroke` (m) | 0.0402 | 0.0030 (−0.0017) | 0.0021 (−0.0027) | **0.0030** | 7.5 % | ≤ 0.010 m | **PASS** |
| side `stroke_share` (S manipulation) | 0.562 | 0.0152 (+0.0229) | 0.0208 (**−0.0329**) | **0.033** | 5.9 % | report | REPORT |
| fwd `stance_stroke` (m) (S manipulation) | 0.1707 | 0.0177 (+0.0371) | 0.0272 (+0.0117) | **0.027** | 16 % | report | REPORT |
| side `lead_scuff_frac` | 0.419 | 0.0087 (−0.0184) | 0.0093 (+0.0093) | **0.0093** | 2.2 % | report | REPORT |

- **Caveat 2.** Two keys are seed-dominated:
  - side `body_offset`: the seed pair differs by +0.017 m, 3× its bootstrap half-width (per hand
    +0.015 / +0.021);
  - side `stroke_share`: −0.033 against ±0.021.

  So a single-seed posture or share change must clear the seed term, not the bootstrap MDE. This
  agrees with the plan's premise note that no 4000-it stage has yet moved the posture beyond seed
  noise (0.017).
- **Manipulation-check scale.** For S's dose check (decision 9), the smallest stroke change a single
  seed resolves is:
  - side `stance_stroke`: 0.0030 m (7.5 % of W);
  - side `stroke_share`: 0.033;
  - fwd `stance_stroke`: 0.027 m (16 % of W).

  fwd `stance_stroke` is the coarsest of the three.
- **Stance definition.** `stroke_share` is the tool's scored-stance B / Ba, not the investigation's
  0.81 / 0.40 stance definition (see the tool docstring). W reads fwd 0.870 and side 0.562.

## Reproduce

Run from the worktree root, with the PYTHONPATH as in the plan:

```
PY=/home/nickmagus/krabby/isaac_venv/bin/python
T=parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task/experiments/tools
$PY $T/replay_stroke_posture.py readout <W>
$PY $T/replay_stroke_posture.py readout <E1> --vs <E2>
$PY $T/replay_stroke_posture.py readout <X0x s3> --vs <X0x s2>
# LEN R_eval: w2 candidates.yaml + the LEN block above (absolute run_dirs)
$PY $T/resolution.py --campaign-dir <scratch> --candidates <that yaml> --out <scratch>/resolution_len.json \
    --eval-sha 36e5a8af7000preview --decisive-at back,diag,fwd,move_turn,side,stand,turn
# LEN R_seed: run_candidates.d00bs_evidence(state.X01.screen, state.X02.screen, resolution_len, rule, all seven)
```
