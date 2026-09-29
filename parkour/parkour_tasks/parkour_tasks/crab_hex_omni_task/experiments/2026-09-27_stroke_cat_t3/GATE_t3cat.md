# t3 gate — per-category stroke weights at formation (S_cat_f), plan step 10 (2026-09-28)

**Verdict: FAIL.** The dose landed. The stride test fails, and health fails separately.
- **Dose landed: PASS.** fwd r is 1.47 / 0.98 and side r is 1.27 / 1.14 at s3 / s2 (bar ≥ 0.5). The
  dwell-weighted side r is 1.30 / 1.06. No small-dwell spike carries either seed's side r.
- **Stride: FAIL.** Pooled side stride_along Δ is **−0.0015 m**. The floored band is 0.0049, so it needed Δ > +0.0049.
  The seeds disagree in sign: s3 is +0.0047 and s2 is −0.0078, each beyond its own MDE.
- **Health: FAIL.** At s2, the diag tripod-coherence unison flag fires: −0.1008.
  - That is against the tool's 0.100 floor. B_f's own diag seed band is 0.077, so the drop is 0.024 beyond it.
  - diag is untrained at t3, but the pre-registered rule names no category exclusion.
  - The pooled F block is BETTER (z +2.42), so it passes.
  - Moving ratios and floors pass at both seeds.
- **Mechanism: neither.** There is no stride gain to attribute.
  - Side speed rose beyond its band when pooled (+0.005 m/s). s3 carries all of it, against the weak B_f s3 baseline.
  - stride_per_speed fell beyond its band at both seeds (pooled −0.070 s).
- **Nothing is adopted.** Step 11 (close) is next.

**What ran.** Two `--reference-chain S_cat_f` runs (chain `t3f_scat`), each 3000 iterations from scratch with the clock
held.
- Final table: `STROKE_W` 0.112339, `STROKE_CAT_W` `fwd:1,back:1,side:3.460881,diag:1.599056,turn:2.115884,move_turn:1`.
  The effective side weight is 0.389, about 3.3× the formation screen's single weight of 0.119.
- s3 = X01, head `2026-09-28_16-40-58_seed3_X01_t3_A15pB/model_2999.pt`, suite stamp `…/2026-09-28_21-57-04`.
- s2 = X02, head `2026-09-28_18-05-55_seed2_X02_t3_A15pB/model_2999.pt`, suite stamp `…/2026-09-28_23-32-01`.
- Floors held on both runs: the notify.log `REFERENCE` lines, the header `floor_watch` is none, and the t3 floor PASSES
  in REPORT.md.

**How the gate is read.**
- **Rule:** the pre-registered step-10 gate in [CHARTER.md](CHARTER.md). It is the plan's step-8 rule with the side
  stride_along band floor.
- **Controls:** the stroke_formation B_f pair, reused.
  - B_f s3 = X01, `2026-09-26_22-56-51` (run `2026-09-26_17-42-44_seed3_X01_t3_A15pB`).
  - B_f s2 = X02, `2026-09-27_04-08-09` (run `2026-09-26_22-54-02_seed2_X02_t3_A15pB`).
- **Deltas:** pooled means over seeds 3 and 2 of S_cat_f − B_f at the same seed (`replay_stroke_posture.py pooled`).
  The tool's DEFAULT_BARS `verdict` field is not used.
- **Band:** max(pooled bootstrap MDE, |B_f s3 − B_f s2|) per key. The one exception is side stride_along, whose band
  is max(pooled MDE, 0.0049).
- **Evidence:** all JSON is in [`gate/`](gate/).

**Confound 1: the controls ran earlier under different code.** The controls ran on 2026-09-26/27 under code_sha
`709161927627`, and the S_cat_f runs under `fb17d2ea389e`. Every header's eval_sha is `36e5a8af7000`. The reuse control
passes at BOTH seeds:
- each run's `params/env.yaml` differs from its B_f's only in `reward_stride_length`'s `weight` (0.001 → 0.112339) and
  its added `cat_scale`;
- `agent.yaml` is identical.

WINDOW0_t3.md checked s3. This gate re-checked both seeds.

**Confound 2: B_f s3 is the weak baseline seed** (stroke_formation `GATE_t3.md`).
- fwd / back / side survival is 0.49 / 0.51 / 0.76, and completion is 0.643.
- fwd / back tripod coherence is 0.35 / 0.40.

So every s3 delta and every seed band built from the B_f pair overstates the arm on fwd and back. On the decisive
key, B_f's two seeds agree (side stride_along 0.0800 / 0.0810). That agreement is why the band is floored there.

## 1. Dose landed — PASS

r = logged income (`Episode_RewardCat`, last 200 iterations) ÷ the replay prediction at the final table on B_f sN's
raws.
- The prediction is priced in B_f sN's own clock-held economy (`params/env.yaml`; clock weights 1.0 / 1.0, asserted).
- The train-log weight guard passed on both runs: the dumped weight and `cat_scale` match the table.

| Seed | fwd | side | back | turn (replay-set) | diag | move_turn | Landed (fwd, side ≥ 0.5) |
|---|---|---|---|---|---|---|---|
| s3 | **1.47** | **1.27** | 1.19 | 0.77 | — | — | PASS |
| s2 | **0.98** | **1.14** | 0.85 | 0.39 | — | — | PASS |

- diag and move_turn are not in t3's training pool. They log no income, so there is no r.
- turn's low r is expected: turn is replay-set, and training turn holds ungate the term (CHARTER). At s3, turn's 0.77
  is lifted by one spike, it 2943 (5.63/s on a 0.85-s dwell). Without it, turn r is 0.31.
- Logged side income was 0.063 / 0.060 per s. The replay predicted 0.050 / 0.053 per s.

**Dwell-weighted r and spikes** (the t3 smoke's side r of 4.61 was a 0.008-s-dwell division artifact, WINDOW0_t3.md).
The dwell-weighted r is sum(rate × dwell_s) ÷ sum(dwell_s) over the same last-200 window, ÷ the same replay
prediction. A spike is an iteration above 10 × the window median.

| Seed | Row | r (tool: mean of rates) | r, dwell-weighted | r, spikes excluded | Spikes in the window | Min side dwell |
|---|---|---|---|---|---|---|
| s3 | side | 1.27 | **1.30** | 1.27 | none | 0.002 s |
| s2 | side | 1.14 | **1.06** | 0.99 | it 2959: 1.63/s on a 4.17-s dwell (32 × the 0.050 median) | 0.005 s |
| s3 | fwd | 1.47 | 1.54 | 1.47 | none | — |
| s2 | fwd | 0.98 | 1.03 | 0.98 | none | — |

Every reading of side r lies between 0.99 and 1.30. The side dose that trained is the one the dwell-weighted smoke set.
- JSON: `gate/S_cat_f_s3_dose.json`, `gate/S_cat_f_s2_dose.json` (tables in the matching `.txt`).
- `gate/S_cat_f_dose_dwell.json` holds the dwell-weighted r, spikes, window medians and dwells.

## 2. Stride — FAIL

The gate key is side `stride_along` (m, the episode MEAN).

| | S_cat_f | B_f (same seed) | Δ ± MDE (95 % CI) |
|---|---|---|---|
| s3 | 0.0846 | 0.0800 | +0.0047 ± 0.0029 ([+0.0018, +0.0076]) |
| s2 | 0.0733 | 0.0810 | **−0.0078 ± 0.0031** ([−0.0108, −0.0046]) |
| **Pooled** | | | **−0.0015**; band **0.0049** (floored) |

- **The band is the floor.** The pooled MDE is 0.0019 and B_f's seed gap is 0.0010, so the step-10 floor of 0.0049
  sets it. Without the floor the result is the same: −0.0015 is below zero, not above 0.0019.
- **The seeds disagree.** s3 lengthened slightly against the weak baseline seed. s2 shortened, by more than s3
  lengthened.
- The hand rows agree: side/l pooled −0.0018 (MDE band 0.0031), side/r −0.0013 (0.0025).
- **For scale:** the single-weight S_f arm read 0.079 / 0.075 against the same controls (pooled −0.0034). Tripling the
  side weight did not move side stride out of the 0.073–0.085 range of all eight t3 runs.

## 3. Mechanism (reported, not gating) — neither

All side rows. cadence_ratio comes from each stamp's `summary.md` rigidity line. Its band is B_f's seed gap, since
cadence_ratio has no bootstrap.

| Side key | Δ s3 (± MDE) | Δ s2 (± MDE) | Pooled Δ | Band | Reading |
|---|---|---|---|---|---|
| achieved speed (m/s) | +0.0123 ± 0.0050 | −0.0017 ± 0.0031 | **+0.0053** | 0.0028 | FASTER, s3 only |
| stride_per_speed (s) | −0.0789 ± 0.0277 | −0.0613 ± 0.0227 | **−0.0701** | 0.0186 | **SHORTER per unit speed**, both seeds |
| cadence_ratio | +0.016 (1.004 vs 0.988) | −0.009 (0.996 vs 1.005) | +0.004 | 0.017 | inside; clock-locked at ≈ 1 |
| touchdown Hz/foot | −0.0144 ± 0.0167 | +0.0373 ± 0.0133 | +0.0114 | 0.0248 | inside |
| taps/s | +1.676 ± 0.158 | +0.247 ± 0.183 | +0.961 | 1.047 | inside the band; up at both seeds, each beyond its own MDE |

- **Not "faster walking" as the source of a stride gain,** because there is no stride gain.
  - Side speed rose beyond its band when pooled, but only at s3, where B_f s3 is the weak seed (side survival 0.76).
  - At s2 speed is flat.
- **Not "longer steps at the same speed".** stride_per_speed fell at both seeds, each beyond its own MDE.
- **Taps rose at both seeds** (+1.68 at s3, +0.25 at s2). The pooled band is B_f's seed gap of 1.05, which the weak
  seed sets, so the pooled Δ reads inside.
- **The clock locks the cadence** (side cadence_ratio 0.988–1.005 in all four stamps). A longer stride could only have come
  from a longer stance sweep or a faster body. The sweep did not lengthen (stance_stroke, section 5).

## 4. Health — FAIL

| Criterion (binding per plan step 8 / CHARTER) | s3 | s2 | Result |
|---|---|---|---|
| Moving ratios ≥ 0.30 (fwd / side / back) | 0.662 / 0.539 / 0.664 | 0.565 / 0.468 / 0.580 | PASS |
| t3 floor (`category_survival`, min_new 0.10, min_prev 0.30); no floor or dead_walker REJECT | floor PASS; `floor_watch` none | floor PASS; `floor_watch` none | PASS |
| Unison clear under B_f's seed band | all clear | **diag tripod coherence −0.1008 vs band 0.100 → FLAG** | **FAIL** |
| Pooled F block not WORSE | | | BETTER (z +2.42) → PASS |

- There is no F_mt criterion: t3 trains no move_turn, and the t3 rule drops F_mt (CHARTER).
- S_cat_f is a reference run, so it is structurally exempt from the dead_walker abort and the dead_plant check. Only
  the end-of-run stage-eval floor applies (CHARTER).

**Unison detail.** The readouts use `--seed-a B_f s3 --seed-b B_f s2`.
- **s2, diag:** tripod coherence fell from 0.4955 to 0.3947. B_f's diag seed Δ is −0.077, so the 0.10 floor sets the
  band. −0.1008 clears that floor by 0.0008, and B_f's own diag seed band (0.077) by 0.024. diag is an untrained
  (WATCH) row at t3.
- **Precedent:** the formation gate counted P_f s2's diag flag (−0.104) toward a health FAIL. That FAIL also rested on
  s2's survival losses and on an F WORSE reject. This gate applies the pre-registered rule as written.
- **s2, other rows:** fwd −0.143 (band 0.260) and back −0.110 (band 0.214) are inside their bands only because B_f s3's
  weak fwd/back coherence widens them. side is +0.004. No roll collapse (ratios 1.05–1.28×).
- **s3:** every row is clear: fwd +0.211, back +0.184, diag +0.045, side −0.031. Roll ratios are 0.76–1.34×.

**`score_reference.py`** (the CHARTER's exact command). It calls the driver's `decide_pooled` against the old
campaign's D00s (`resolution.36e5a8af7000.json`), with seed terms D00bs / D00bs and cand spec `S_f`.

| Block | Pooled z | R | Pooled class | s3 class (z) | s2 class (z) |
|---|---|---|---|---|---|
| F | +2.42 | 1.26 | **BETTER** | BETTER (+4.85) | TIE (−0.01) |
| T | +2.06 | 0.99 | BETTER | BETTER (+3.86) | TIE (+0.25) |
| LEN | +1.24 | 1.55 | TIE | BETTER (+3.90) | TIE (−1.42) |

- **F rows:**
  - turn facing_closed: 0.467 → 0.716 at s3, and 0.412 → 0.394 at s2.
  - turn yaw_progress: 0.063 → 0.094 at s3, and 0.053 → 0.055 at s2.

  The pooled BETTER comes from s3, against the weak baseline seed.
- **Guards: 0 of 17 active guards are pooled WORSE at level 0.99.** The rule has 23; the 6 diag / move_turn guards read
  N/A because those categories are untrained. No guard is MISSING.
- **Driver pick: ADOPT (clear).** Anchored drift is OK. The per-seed score is s3 +10.30 and s2 −0.67 (every s2 block
  TIE). This is a record only; the gate above decides.
- **Not binding, but s2 lost ground against its own baseline:**
  - survival: fwd 0.91 → 0.84, back 0.97 → 0.86, turn 0.98 → 0.875; untrained diag 0.83 → 0.63 and move_turn
    0.77 → 0.615;
  - completion: 0.910 → 0.822;
  - gate_1c WATCH: PASS → FAIL (walking-hold tripod 0.385, training track_ratio 0.594).

  None of these reaches WORSE at 0.99, because the B_f seed gap sets R. s3 gained against its weak baseline on the
  same rows: fwd 0.49 → 0.96, back 0.51 → 0.98, side 0.76 → 0.99, and completion 0.643 → 0.922.
- JSON: `gate/S_cat_f_score_reference.json` (printout: `gate/S_cat_f_score_reference.txt`).
- Unison readouts: `gate/S_cat_f_s3_vs_B_f_s3.json` and `gate/S_cat_f_s2_vs_B_f_s2.json`. The s3 readout re-runs the
  earlier per-seed preview with the same command, and the output size is unchanged.
- `gate/S_cat_f_summary_health.json` holds the per-category survival, ratio, tripod, cadence_ratio and foot_slip from
  the four `summary.md` files.

## 5. Reported, not gating

**Side stroke and posture** (pooled, `gate/S_cat_f_pooled.json`):

| Side key | Δ s3 (± MDE) | Δ s2 (± MDE) | Pooled Δ | Band | Reading |
|---|---|---|---|---|---|
| stance_stroke (m) | +0.0020 ± 0.0025 | −0.0070 ± 0.0029 | −0.0025 | 0.0034 | inside |
| stroke_share | −0.0728 ± 0.0249 | −0.0750 ± 0.0230 | **−0.0739** | 0.0153 | **LOWER**, both seeds |
| body_offset (m) | +0.0120 ± 0.0062 | +0.0594 ± 0.0080 | **+0.0357** | 0.0255 | **MORE lean**, both seeds |

- The side slip ratio (rho) rose by +0.50 at s3 and +0.15 at s2. The side foot_slip guard went 0.179 → 0.232 at s3
  (pooled −0.024 in the guard's better-is-positive sign). It stays inside the guard's R of 0.039.
- The side stroke income paid for more slipping, lower-share stances and a larger lean, not for a longer loaded sweep.

**LEN:** TIE pooled (z +1.24). Side steps_per_m went 32.4 → 27.0 at s3 and 30.3 → 32.2 at s2. By seed, LEN is BETTER
at s3 and TIE at s2. At t3 the cadence is clock-locked, so LEN is reported only.

**Elsewhere** (pooled against band). Both big fwd/back gains are s3 against the weak seed, and s2 went the other way:
- **fwd:** stride_along +0.037 against a band of 0.068.
  - s3 +0.094.
  - s2 −0.019, beyond its MDE of 0.014.
- **back:** stride_along +0.032 against a band of 0.048.
  - s3 +0.077.
  - s2 −0.012, beyond its MDE of 0.012.
- **diag:** stride_along +0.003 against a band of 0.023.

**Delivered share in training.** This is each category's logged stroke income ÷ its logged tracking income, both
from `Episode_RewardCat`, as a ratio of MEANS.
- Tracking income is `track_lin_vel_xy_exp` on translation rows, the replay's definition.
- For turn it is `track_ang_vel_z_exp + reward_yaw_progress`.
- diag and move_turn are not in t3's pool, so they log nothing.
- The 500-iteration windows start at 0, 500, …, 2500. A spike is an iteration above 10 × the run median.
- JSON: `gate/S_cat_f_train_share.json`.

| Category | s3 last 200 its | s3 whole run | s3 max 500-it window | s2 last 200 its | s2 whole run | s2 max 500-it window |
|---|---|---|---|---|---|---|
| fwd | 9.6 % | 8.8 % | 10.5 % | 9.1 % | 8.4 % | 9.7 % |
| side | 9.4 % | 10.4 % | **16.7 %** | 9.0 % | 8.3 % | 9.6 % |
| back | 7.5 % | 6.3 % | 7.1 % | 6.1 % | 6.3 % | 7.3 % |
| turn | 5.8 % | 3.3 % | 5.9 % | 2.2 % | 2.4 % | 2.8 % |

- **No category exceeds 15 % over the last 200 iterations or over the whole run.** One 500-it window does:
  - **Flag: s3 side, its 0–499, 16.7 %.** It comes from spikes. The window holds 2 of them, the larger at it 441
    (16.6/s on a 1.67-s dwell, about 300 × the 0.055 run median). Without spikes that window reads 9.4 %. Without
    spikes, s3 side reads 7.2–9.8 % in every 500-it window (9.2 % over the whole run).
- **Other spikes:**
  - s3 side: 4 in all (the rest at its 2306, 931 and 359, the largest after 441 being 4.24/s).
  - s2 side: 3 (5.67/s at 1732, 3.55/s at 1193, 1.63/s at 2959). s2 side reads 7.1–8.2 % per window without them
    (7.7 % whole run).
  - turn: 13 at s3 and 20 at s2, several on sub-0.02-s dwells (8.37/s at s3 it 1492 on a 0.018-s dwell). Without them,
    s3 turn reads 2.3 % over the whole run.
  - No trend accompanies any of them, and no learned exploit is visible.
- **The side dose hit its design point.** Side delivered about 9 % of side tracking, against the single-weight S_f's
  2–3 %. So this is a fair test of an equal-share side stroke income at formation. fwd sat at 8–10 %, also on target
  (W's fwd ran at 13–14 %).

## Reading

At formation, the per-category dose delivered side stroke income at about 9 % of side tracking, over 3000 iterations
from scratch at two seeds. That is about 3–4× the share the single-weight S_f arm gave side (2–3 %). Its effects:
- **Side strides:** no longer. Pooled Δ is −0.0015 m. s3 +0.005 against the weak baseline seed, and s2 −0.008; both
  seeds are beyond their own MDEs, in opposite directions.
- **Side gait:** the income bought slippier, lower-share stances, more taps and more lean, at both seeds. It did not
  buy a longer loaded sweep.
- **By seed:**
  - s3 looks better on almost everything: survival, tracking and turning. But every s3 comparison is against B_f s3,
    the baseline seed that failed its own gate_1c.
  - s2, against a healthy baseline, lost survival on fwd, back and turn (side −0.01), lost completion, lost fwd/back/diag
    tripod coherence, and flips gate_1c to FAIL.
  - The two seeds do not agree on the arm's effect on health either.

## Comparison with the W gate

Both settings fail the same way, and the verdict at t3 does not depend on the unison flag: stride fails on its
own. In each setting, with the clock free (W, fine-tune from the baked head) and with the clock held (t3, from
scratch):
- the per-category dose landed, with side r of 1.34 / 1.06 on W and 1.27 / 1.14 here;
- side stroke income sat near its 10 %-of-tracking design point;
- side stride_along did not rise: pooled −0.0042 on W and −0.0015 here, against the same 0.0049 band;
- the mechanism read "neither", and stride_per_speed fell beyond its band (W −0.020, here −0.070);
- body_offset rose beyond its band (W +0.010, here +0.036);
- taps rose at both seeds.

The seeds split differently:
- **W:** s3 was the damaged seed (fwd stride −0.073, fwd unison flag).
- **t3:** s3 is the flattered one (weak baseline), and s2 is damaged (survival losses, the diag unison flag).

The driver's blocks differ too:
- **W:** F and F_mt TIE, with 4 guards WORSE.
- **t3:** F BETTER, with no guard WORSE. That result is carried by s3's comparison against B_f s3.

With the single-weight S and S_f screens and the centring arms already on record, the plan has now tested per-category
weighting of the stroke term at the designed equal-share dose in both settings. It does not lengthen omni side strides
in either.

## Independent check (fable·high, 2026-09-28): the verdict holds

The check re-ran on CPU and every number reproduces:
- the pooled readout (0 diffs);
- dose-landed income at both seeds, with the weight guard passing;
- the s2 seed-banded readout (the diag flag at −0.1008);
- `score_reference.py` (byte-identical printout);
- a tensorboard recompute of the dwell-weighted r, the spikes and the share windows.

The side stride band floor was applied as the plan prescribes; pooled −0.0015 fails with or without it.

**Provenance.** env.yaml differs from B_f's only in the stroke weight and `cat_scale` at both seeds, and agent.yaml is
identical.

**Two corrections, both applied above:** the diag flag's strength against B_f's seed band, and the P_f precedent's
other causes. Neither changes the verdict.
