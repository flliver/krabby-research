# W gate — per-category stroke weights (S_cat), plan step 8 (2026-09-28)

**Verdict: FAIL.** The dose landed. The stride test fails, and health fails separately.
- **Dose landed: PASS.** fwd r is 0.67 / 0.77 and side r is 1.34 / 1.06 at s3 / s2 (bar ≥ 0.5).
- **Stride: FAIL.** Pooled side stride_along Δ is **−0.0042 m**, against a band of 0.0049. It needed Δ > +0.0049.
- **Health: FAIL.** At s3, the fwd tripod-coherence unison flag fires (−0.133 against a 0.100 band). Separately,
  the driver's rule marks 4 guards pooled WORSE. The pooled F and F_mt blocks are TIE, so they pass.
- **Mechanism: neither** faster walking nor longer steps at the same speed. Side speed is flat.
  stride_per_speed fell beyond its band, and taps rose beyond theirs, at both seeds.
- **Nothing is adopted.** Per the plan's Phase C header, Phase C (t3) still runs after this gate.

**What ran.** Two `--reference-chain S_cat` runs (chain `c6y_scat`), 4000 iterations each, forked from W's baked
copy (its 11996 → 15994).
- Final table: `STROKE_W` 0.126240, `STROKE_CAT_W` `fwd:1,back:1,side:2.928327,diag:1.314741,turn:1.214263,move_turn:1`.
- s3 = X01, head `2026-09-28_08-11-28_seed3_X01_c6y_A15pB/model_15994.pt`, suite stamp `…/2026-09-28_13-53-35`.
- s2 = X02, head `2026-09-28_14-37-59_seed2_X02_c6y_A15pB/model_15994.pt`, suite stamp `…/2026-09-28_20-18-06`.
- s2 ran on the user's override of the s3 pilot STOP ([pilot/PILOT.md](pilot/PILOT.md)).
- Floors held on both runs (notify.log `REFERENCE` lines; `floor_watch` none in both headers).

**How the gate is read.**
- **Rule:** the pre-registered step-8 gate in [CHARTER.md](CHARTER.md), which is verbatim from the plan.
- **Controls:** the reused stroke_posture W-screen B pair: B s3 = X01 `2026-09-26_08-39-30`, B s2 = X02
  `2026-09-26_10-25-08`.
- **Deltas:** pooled means over seeds 3 and 2 of S_cat − B at the same seed (`replay_stroke_posture.py pooled`).
  The tool's DEFAULT_BARS `verdict` field is not used.
- **Band:** max(pooled bootstrap MDE, |B s3 − B s2|) per key.
- **Evidence:** all JSON is in [`gate/`](gate/).

**Confound.** The controls ran a day earlier, under code_sha `709161927627`. The S_cat runs ran under
`fb17d2ea389e`. Every header's eval_sha is `36e5a8af7000`. Window 0's identity checks (0/20 dumps differ; the knob-on
dump differs only in the stroke weight and `cat_scale`) are the control for this confound.

## 1. Dose landed — PASS

r = logged income (`Episode_RewardCat`, last 200 iterations) ÷ the replay prediction at the final table on B sN's
raws, in W's clock-free economy (`2026-09-25_14-53-27_seed3_X03_c6x_A15pB/params/env.yaml`). The train-log weight
guard passed on both runs: the dumped weight and `cat_scale` match the table.

| Seed | fwd | side | back | diag | turn | move_turn | Landed (fwd, side ≥ 0.5) |
|---|---|---|---|---|---|---|---|
| s3 | **0.67** | **1.34** | 0.74 | 0.69 | 0.17 | 0.73 | PASS |
| s2 | **0.77** | **1.06** | 0.81 | 0.87 | 0.19 | 0.93 | PASS |

- Logged side income was 0.076 / 0.060 per s. The replay prediction was 0.057 per s.
- turn's low r is expected. turn is replay-set, and training turn holds ungate the term (CHARTER).
- JSON: `gate/S_cat_s3_dose.json`, `gate/S_cat_s2_dose.json` (tables in the matching `.txt`).

## 2. Stride — FAIL

The gate key is side `stride_along` (m, the episode MEAN), shown per seed beside the s3 pilot.

| | S_cat | B (same seed) | Δ ± MDE |
|---|---|---|---|
| s3 (the pilot) | 0.0865 | 0.0964 | **−0.0099 ± 0.0028** |
| s2 | 0.0930 | 0.0915 | +0.0015 ± 0.0027 |
| **Pooled** | | | **−0.0042**, band **0.0049** |

- The band is B's seed gap (|0.0964 − 0.0915| = 0.0049). The pooled MDE is 0.0020.
- **FAIL.** The pooled Δ is below zero, not above the band.
- **s2 did not agree with the s3 pilot's shortening.** Its Δ is inside its own MDE. It also shows no lengthening.
  Neither seed's side stride rose beyond noise.
- The hand rows agree: side/l pooled −0.0027 (band 0.0028), side/r −0.0057 (band 0.0075).

## 3. Mechanism (reported, not gating) — neither

All side rows. cadence_ratio comes from each stamp's `summary.md` rigidity line (the `scenario_metrics.json` median).
Its band is B's seed gap, since cadence_ratio has no bootstrap.

| Side key | Δ s3 | Δ s2 | Pooled Δ | Band | Reading |
|---|---|---|---|---|---|
| achieved speed (m/s) | −0.0098 | +0.0053 | −0.0023 | 0.0039 | inside |
| stride_per_speed (s) | −0.0158 | −0.0244 | **−0.0201** | 0.0140 | **SHORTER per unit speed**, both seeds |
| cadence_ratio | +0.078 (1.418 vs 1.340) | −0.052 (1.361 vs 1.413) | +0.013 | 0.073 | inside |
| touchdown Hz/foot | +0.0111 | −0.0042 | +0.0035 | 0.0185 | inside |
| taps/s | +0.181 | +0.296 | **+0.239** | 0.092 | **MORE taps**, both seeds |

- **Not faster walking.** Side speed is flat when pooled. s3 slowed, and s2 sped up slightly.
- **Not longer steps at the same speed.** stride_per_speed moved the other way, and the pooled Δ clears the band.
  - s3's Δ (−0.0158) is inside its own MDE of 0.017.
  - s2's is not: −0.0244 against an MDE of 0.018, 95 % CI [−0.043, −0.006].
  - This was corrected by the fable check; it strengthens the "shorter per unit speed" reading.
- **More taps.** The side stroke income bought extra short unloaded contacts, not longer loaded steps.

## 4. Health — FAIL

| Criterion | s3 | s2 | Result |
|---|---|---|---|
| Moving ratios ≥ 0.30 (fwd / side / back) | 0.711 / 0.490 / 0.752 | 0.773 / 0.545 / 0.767 | PASS |
| No floor WATCH trip | floors held; `floor_watch` none | floors held; `floor_watch` none | PASS |
| Unison clear under B's seed band | **fwd tripod coherence −0.133 vs band 0.100 → FLAG** | all clear | **FAIL** |
| Pooled F block not WORSE | | | TIE (z −0.78) → PASS |
| Pooled F_mt block not WORSE | | | TIE (z −0.99) → PASS |

**Unison detail.** At s3, fwd tripod coherence fell from 0.584 to 0.451. B's fwd seed Δ is +0.007, so the 0.10 floor
sets the band. Every other s3 row is clear: back −0.080, diag −0.020, side −0.015. No roll collapse (ratios
0.99–1.24×). At s2, every row is clear: fwd +0.031, side −0.024, back +0.049, diag +0.086. The unison readouts use
`--seed-a B s3 --seed-b B s2`. The s3 readout is byte-identical to the pilot's.

**`score_reference.py`** (the CHARTER's exact command). It calls the driver's `decide_pooled` against the old
campaign's D00s (`resolution.36e5a8af7000.json`), with seed terms D00bs / D00bs.

| Block | Pooled z | R | Pooled class | s3 class | s2 class |
|---|---|---|---|---|---|
| F | −0.78 | 1.81 | TIE | TIE | TIE |
| F_mt | −0.99 | 2.45 | TIE | **WORSE** (z −4.82) | BETTER (z +2.85) |
| T | −1.32 | 1.67 | TIE | WORSE | TIE |
| RIG | +0.75 | 0.42 | BETTER | BETTER | TIE |
| LEN | −0.82 | 1.53 | TIE | WORSE | TIE |

- **Guards: 4 of 23 are pooled WORSE at level 0.99.**
  - fwd ratio: −0.085 (s3 0.861 → 0.711, s2 0.793 → 0.773).
  - fwd foot_slip: 0.122 → 0.146 at s3, 0.116 → 0.127 at s2.
  - back foot_slip.
  - turn foot_slip.

  No guard is MISSING.
- **Driver pick: REJECT (clear).** Anchored drift FAILs on the same four guards.
- The step-8 health line names F and F_mt as binding. The CHARTER adds that the other blocks and guards "are read
  the same way". Under that reading, the 4 WORSE guards are a second health failure. Health fails on unison either
  way.
- JSON: `gate/S_cat_score_reference.json` (printout: `gate/S_cat_score_reference.txt`), `gate/S_cat_s3_vs_B_s3.json`,
  `gate/S_cat_s2_vs_B_s2.json`, `gate/S_cat_summary_health.json`.

## 5. Reported, not gating

**Side stroke and posture** (pooled, `gate/S_cat_pooled.json`):

| Side key | Δ s3 | Δ s2 | Pooled Δ | Band | Reading |
|---|---|---|---|---|---|
| stance_stroke (m) | −0.0037 | +0.0075 | +0.0019 | 0.0025 | inside |
| stroke_share | −0.0074 | +0.0311 | +0.0119 | 0.0104 | just beyond the band; carried by s2 alone |
| body_offset (m) | +0.0139 | +0.0067 | **+0.0103** | 0.0090 | MORE lean, both seeds |

**LEN:** TIE pooled (z −0.82). Side steps_per_m moved −0.89 and diag −1.61. By seed, LEN is WORSE at s3 and TIE at s2.

**Elsewhere** (pooled against band):
- **fwd:**
  - stride_along −0.037 (band 0.028), carried by s3 alone (−0.073; s2 +0.000);
  - stroke_share −0.028 (band 0.014), both seeds;
  - body_offset +0.020 (band 0.007), both seeds.
- **back:**
  - stride_along −0.028, inside its band of 0.035 (s3 −0.058);
  - stroke_share −0.040 (band 0.014), both seeds.
- **diag:** stride_along −0.006, inside its band of 0.018.
- **The one gain was survival on diag and move_turn.**
  - diag: 0.56 → 0.70 at s3, 0.64 → 0.84 at s2.
  - move_turn: 0.585 → 0.735 at s3, 0.525 → 0.660 at s2.

  Neither guard clears 0.99 (z +1.21 / +1.43). Completion is 0.865 → 0.896 at s3 and 0.861 → 0.907 at s2.

**Delivered share in training.** This is each category's logged stroke income ÷ its logged tracking income, both
from `Episode_RewardCat`, as a ratio of MEANS.
- Tracking income is `track_lin_vel_xy_exp` on translation rows, the replay's definition.
- For turn it is `track_ang_vel_z_exp + reward_yaw_progress`. For move_turn it is all three terms.
- The stroke numerators match the dose JSONs' `train_log` rows (for example, side s3 0.075 against 0.076). The dose
  JSONs carry no per-category training tracking, so the tracking income was read from the same tensorboard run.
- JSON: `gate/S_cat_train_share.json`.

| Category | s3 last 200 its | s3 whole run | s3 max 500-it window | s2 last 200 its | s2 whole run | s2 max 500-it window |
|---|---|---|---|---|---|---|
| fwd | 13.4 % | 13.5 % | 14.0 % | 13.8 % | 13.6 % | 13.9 % |
| side | 11.2 % | 9.6 % | 11.1 % | 8.7 % | 11.2 % | 14.4 % |
| back | 11.6 % | 10.8 % | 12.9 % | 10.9 % | 10.9 % | 11.6 % |
| diag | 9.8 % | 11.5 % | **16.5 %** | 10.3 % | 10.7 % | 13.8 % |
| turn | 1.6 % | 2.5 % | 6.1 % | 1.6 % | 2.7 % | 4.6 % |
| move_turn | 2.0 % | 2.3 % | 3.1 % | 2.2 % | 2.3 % | 2.7 % |

- **No category exceeds 15 % over the last 200 iterations or over the whole run.** One 500-it window does:
  - **Flag: s3 diag, its 15001–15500, 16.5 %.** It comes from spikes. The window holds 4 iterations above 10 × the
    median, including 5.79/s at 15337–15338. Excluding iterations above 10 × the run median, s3 diag reads 9.0–10.4 %
    in every 500-it window (9.8 % over the whole run).
- **fwd sits at 13–14 % at both seeds.** That is 1.35 × the 10 % target, and fwd r (0.67 / 0.77) was below the
  replay prediction. The 10 %-of-tracking design point was not the delivered share for fwd.
- **The s2 side spike** (PILOT.md: it 12458, 6.59/s, 120 × the window median) was not isolated.
  - s2 logged 11 side iterations above 10 × the median of 0.059/s. Four were larger than the pilot's: 7.41/s at
    12753, 7.24/s at 13948, 7.06/s at 14085, and 8.60/s at 15557.
  - s3 logged 9, the largest 3.19/s.
  - These spikes lift s2's side windows to 13.8–14.4 %. Excluding iterations above 10 × the run median, s2's side
    share is 8.6–9.6 % in every 500-it window (9.0 % over the whole run; s3 9.1 %). The window medians stay flat at
    0.056–0.063/s.
  - Heavy-tailed spikes at side × 2.93, with no trend, are the pattern PILOT.md flagged. No learned exploit is
    visible.

## Reading

At the per-category dose, side stroke income did land. It sat at 9–11 % of side tracking (3 × the single-weight dose)
over 4000 iterations from W, at two seeds. Its effects:
- **Side strides:** no longer. The pooled Δ is −0.004 m. s3 got shorter by slowing down, and s2 was flat.
- **Side gait:** the stroke income bought more taps and a slightly larger lean, not longer loaded steps.
- **fwd at s3:** a large loss. Stride fell −0.073, the ratio fell 0.86 → 0.71, and tripod coherence fell −0.13.
  That did not replicate at s2.
- **Foot slip:** worse on fwd, back and turn when pooled.
- **Survival:** diag and move_turn survival rose at both seeds. That is the only improvement, and it is inside the
  guard level.

The pilot's s3 FAIL stands, and s2 did not reverse it. Per-category weighting of the stroke term does not lengthen
omni side strides on W.

## Independent check (fable·high, 2026-09-28): the verdict holds

The check re-ran on CPU: the pooled readout, both seeds' dose-landed income, `score_reference.py` (JSON byte-equal),
the seed-banded unison readouts and the training-share table. Every number reproduces.

**Provenance:** the S_cat `params/env.yaml` differs from B's in exactly the 7 stroke-weight / `cat_scale` leaves at
BOTH seeds.

**One correction,** applied above in section 3: s2's stride_per_speed Δ is beyond its own MDE.

**Two reading notes, neither changing the verdict:**
- Counting the 4 WORSE guards as a health failure extends the plan's step 8, which names F and F_mt as the binding
  blocks. The unison flag alone fails health.
- The unison tool calls its flag "never gates", but this plan's step 8 lists it under Health, so it gates here.

