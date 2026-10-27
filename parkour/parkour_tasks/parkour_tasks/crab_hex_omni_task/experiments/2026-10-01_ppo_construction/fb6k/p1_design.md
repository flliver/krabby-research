# P1 design after FB6k (opus·high, workflow wf_a570110e-2aa)

## What FB6k licenses and does not

Paths: PC = crab_hex_omni_task/experiments/2026-10-01_ppo_construction. MF = crab_hex_omni_task/policy/manifest.yaml.

WHAT FB6k LICENSES
1. The B package forms both side hands from scratch, at fresh seeds, when fwd and back share the pool.
   - B was FAST/FAST in 3/3 runs at model_5999, with feet 0.203/0.202, 0.203/0.198 and 0.226/0.187. C0 was 0/3, at 0.085-0.111 (PC/fb6k/fb6k_read.md:11-16, 24).
   - It used no fork, so the premise that a fork equals a continuous run, carried over from x6k, now holds from scratch (CHARTER.md:548).
   - Per-hand pool weight was 0.2 (CHARTER.md:483).
2. Formation timing:
   - At 4500, the second hand was FAST in 2/3 runs. s18's R hand was INT at 0.172 at 4500 and FAST at 5999 (fb6k_read.md:26-28).
   - Per-hand exposure at completion is ≈900 weighted iterations for 2/3 of seeds and ≤1200 for 3/3. On C_lr it was ≤1600 (0.4 × ≤4000; x6k/next_options.md:13).
   - So halving the weight cost about 1.4× the iterations, not 2×. Formation is sub-linear in weight.
3. fwd and back are not hurt; they improve.
   - B's seed-means are fwd 0.966 and back 0.935, against C0's 0.638 and 0.669 (fb6k_read.md:21-22).
   - B's suite survival is 1.00 against C0's 0.80-0.98 (fb6k_read.json:55-67, 292-462).
4. B's policy mean stays unsaturated and C0's stays saturated.
   - B's median |μ| is 0.51-0.54 (cams) / 0.84-0.88 (rods).
   - C0's is 6.84-8.34 / 9.15-9.42 (fb6k_read.md:11-16).
5. diag (descriptive only, not trained) reads B 0.82-0.96 against C0 0.48-0.58. That is a first sign that cam and rod strokes blend.
6. The label stays "B package (construction + action_rate 0) vs C0" (plan foamy-wondering-ullman.md:389).
   - INC0 (C0 plus action_rate 0 alone) was 0/3 FAST/FAST, so action_rate 0 alone does not flip the result.
   - But INC0 did form one FAST hand in 3/3 runs, against C0 0/16 at 3000 (CHARTER.md:959-960). So action_rate 0 contributes.
7. The pre-registered precondition for P1 is met (CHARTER.md:536-537). Running P1 is the user's call.

WHAT FB6k DOES NOT LICENSE
1. Clock-free walking.
   - Every FB6k run had the clock visible and the clock reward on (base 1b; x6k/next_options.md:25).
   - The side stroke is believed to be tripod-phase-locked (side_bisect/HANDOFF_PPO.md:142), so survival through the c6x cut is untested.
2. Turn, move_turn or a trained diag. The slots were 0.2:0.8:0:0 with no turn slot (candidates_v5.yaml:720).
3. Any claim below a per-hand weight of 0.2. P1's t3 uses 0.16 and c6 uses 0.09 (candidates_v5.yaml:794, 847).
4. Any comparison to W. W used a different recipe (4-image mirror, RSI 0.2, seed-3 selection; plan:396-401). FB-bed ratios are not omni-suite ratios.
5. The construction without action_rate 0. A (the same construction with action_rate −0.3) stalls its second hand: σ decays and the rods re-saturate (RESULTS.md:110-113).
6. Smoothness.
   - B's jitter is 0.152-0.157, against C0's 0.096-0.106 (CHARTER.md:1001, 1042, 1062; fb6k_read.json:264, 343, 422).
7. Adoption or a bake.
8. Strong statistics. With n = 3 per arm, 3/3 vs 0/3 gives Fisher one-sided p = 0.05 and two-sided p = 0.10.
9. A formal 4c trigger. The trigger was amended after the result was seen (x6k/next_options.md:18), so P1 runs on the user's approval.

## P1-B6 design

P1-B6 is the P1_C_lr chain with t3 re-sized to 6000. It compares the B lineage against the C0 lineage, at seeds 16-18. Same seed with a different config is a new sample, not a replay (memory same-seed-training-deterministic).

STAGES. Each stage is identical in both arms except for the B knobs; overlays are from PC/scripts/candidates_v5.yaml:786-1124.

Every stage keeps:
- fa mirror, RSI 0, ellipse 0.35:0.35:0.25;
- FACING 0.2:0.6 and YAW_W 0.2;
- KRABBY_PPO_HAND_DIAG=1 (RNG-free, as in FB6k);
- 256 envs, base 1a.

| Stage | Iterations | Pool | Slots | Clock | Per-hand side exposure |
|---|---|---|---|---|---|
| t3 | 6000 from scratch (was 3000) | stand .15 / fwd .25 / side_l .16 / side_r .16 / back .08 / turn .2 | .15:.65:.2:0 | visible, CLOCK_W 1.0 | 960 |
| c4 | 1500 | adds fwd_turn .05 and side_l/side_r_turn .025 each | .15:.6:.15:.1 | held | pure side 210 (248 with side_turn) |
| c5 | 1500 | adds omni (diag) .15 | .15:.6:.1:.15 | held | 150 |
| c6 | 2000 | adds omni_turn .07 | .15:.55:.1:.2 | held | 180 |
| c6x | 4000 | the c6 pool | the c6 slots | hard cut from iteration 0: CLOCK_W 0.001, APEX_W 0.001, KRABBY_OMNI_CLOCK_HIDE=1 (W's recipe, MF:229-233) | — |

- The chain totals 15,000 iterations (W's chain is 12,000).
- **Why t3 is 6000:**
  - It puts 960 per-hand exposure inside t3. FB6k's 2/3 completion point was ≈900.
  - The pure-side exposure before the clock cut then totals 1500, which is above FB6k's 3/3 point of ≤1200. So late seeds can still finish in c4-c6, while the clock is still visible.
  - Variant: t3 at 7500 puts 1200 inside t3 alone (+1500 iterations per run, +3.9 GPU-h).
- **Base 1a vs FB's 1b:** 1a and 1b differ only in DIR_BAND, SLOTS and RSI_SOURCE (crab_hex_omni_phases.py:266-271). The overlays override the first two, and RSI_SOURCE is inert at RSI_FRAC 0. Confirm by diffing the dumped params/env.yaml on CPU before launch.

ARMS
- **B:** the B knobs on every stage:
  - KRABBY_ACTION_BOUND_LOSS 0.001, ACTION_BOUND_RAMP 0.0001:0.001:500, ACTION_BOUND 1.1;
  - STD_PARAM soft, INIT_STD 0.8;
  - ENTROPY_COEF 0.0025;
  - ACTION_RATE_W 0.
- **C0 lineage:** P1_C_lr__C0 verbatim, with t3 at 6000 (equal budget) and no canonical training.
- **W** (policy/5_c6x_clockfree/model_11995.pt) is a head-vs-head reference only. Re-score it at the campaign code_sha with suite__omni and side22_25lr__omni, clock hidden, because cross-manifest resolution is ±0.04 (memory eval-process-state-sensitivity).
- **Seed 19-20:** spares, not auto-run.

RUN ORDER (stage-major, with one stop point)
1. **t3 block:** C0_s16, B_s16, C0_s17, B_s17, C0_s18, B_s18.
2. **t3 read** (CPU).
3. **Tail block:** c4 → c5 → c6 → c6x per run, forked from each t3 head, in the same order.
4. **W re-score**, then the reader, then an independent check.

One Isaac process at a time. The runs are detached through the queue runner and log-sentinel watcher (memory gpu-queue-chaining).

EVALS. Every eval set is suite__omni (stand / fwd / back / side / diag / turn / move_turn) plus the both-hands side22_25lr__omni, read with side22_classify and along_cmd (§4.1 FAST rule: feet AND along, 0.155 / 0.175).

| Stage | Read at | Contents |
|---|---|---|
| t3 | model_4500 | lr only |
| t3 | head | suite and lr |
| c4 | head | suite |
| c6 | head (pre-cut, clock visible) | suite and lr |
| c6x | +2000 | lr (cut transient) |
| c6x | head (clock hidden) | suite and lr |

Every eval set also carries:
- rod and cam odd RMS (sym/odd_rms.py);
- the Q block (action_jitter and action_fft_freq), both per suite category and on the lr side;
- the B dose: median |μ| over each stage's last 100 iterations, which must be ≤ 1.5;
- the turn metrics facing_closed and yaw_progress;
- move_turn survival and ratio.

Clock hidden at eval: lr_eval inherits the driver stamp's env_vars (run_arm.sh:270-276). Assert KRABBY_OMNI_CLOCK_HIDE=1 appears in every c6x stamp's run_meta.json, and stop on its absence (MF:42-47 warns that eval builds the clock VISIBLE by default).

KILLS
- **t3:**
  - the iteration-200 wiring kill and the iteration-1000 dose kill, B only (§4.2);
  - live divergence;
  - DEAD-at-1000 is advisory, since it is uncalibrated on this pool (as CHARTER.md:504).
- **Every stage boundary:**
  - a B dose failure flags the run; at the c6x head it fails the run's FAST/FAST count;
  - a missing read is PENDING, never a decision.
- **Tripwire:** a fall share above max(0.15, ref + 0.10), where ref is the C0-lineage median at the same read (§4.4).

PRE-REGISTERED STOP AFTER t3 (saves ≈27 GPU-h)
- **STOP and ask** if B has ≥ 1 FAST hand in ≤ 1/3 seeds. Formation fails once turn shares the pool.
- **STOP and ask** if C0 is FAST/FAST in ≥ 2/3 seeds. The construction effect is unresolved.
- **Otherwise:** run the tails. The t3 FAST/FAST count is descriptive.

FINAL GATE (c6x heads, clock hidden; the first rule that matches decides)
1. **PENDING:** any read is missing.
2. **CLOCK-CUT LOSS:** B is FAST/FAST at c6 in ≥ 2/3 seeds, but at c6x in ≤ 1/3. Stop and ask (the next lever would be a clock anneal, not the construction).
3. **FAIL:** either of these:
   - B counted FAST/FAST (both hands, untripped, dose OK) in ≤ 1/3 seeds;
   - any category clearly inferior to the C0 lineage. "Clearly inferior" means the seed-mean is more than 0.03 below C0's AND B's best seed is below C0's worst seed. The categories are fwd, back and diag ratio; turn facing_closed and yaw_progress; move_turn survival and ratio.
4. **PASS:** B is FAST/FAST in ≥ 2/3 seeds, C0 is ≤ 1/3, no B run trips, and every category holds (seed-mean ≥ C0's − 0.03).
5. **PARTIAL:** anything else, reported with the numbers.

REPORTED NEXT TO THE GATE (not part of it)
- **vs W:** per-category deltas against the re-scored W. Flag move_turn survival (W 0.365, MF:250) and turn facing_closed (W 0.787).
- **Adoption-candidate flag:** set when, on the same seed, a B head is FAST/FAST, its side ratio beats W, and no category is more than 0.05 below W. Any adoption or bake is stop-and-ask.
- **Smoothness flag:** set when any suite category's B jitter exceeds 1.5× the C0 lineage's, or B's fft exceeds C0's + 0.3. The flag makes an S-stage mandatory before any adoption.

GPU-h (≈1.3 per 3000 iterations; ≈0.1 per eval)

| Item | Calculation | GPU-h |
|---|---|---|
| Training | 6 × 15,000 iterations = 30 × 1.3 | 39.0 |
| Evals | 6 runs × 9 evals × 0.1 | 5.4 |
| W re-score | 2 evals | 0.2 |
| **Total** | | **≈ 45** (≈ 45 h wall, serial) |
| If the t3 stop fires | 6 × 2.6 + 1.8 | ≈ 17.4 |
| t3 at 7500 instead | | +3.9 |

PRE-LAUNCH (CPU, free)
1. A candidates_v6 with t3 at 6000 for both arms, plus the HAND_DIAG overlay, validated by a dry run.
2. run_arm/queue support for per-stage lr evals (t3 model_4500, the c6 head, c6x +2000). Today's EXTRA_LR_EVALS is single-run-dir (run_arm.sh:390).
3. p1_read.py, with a self-test of the gate.
4. Confirm across a stage fork that:
   - the bound ramp and the priv_reg counter continue (x6k/CHECK.md:25; ppo_with_extractor.py:963);
   - the soft σ loads without INIT_STD re-initialising it (candidates_v5.yaml:783 claims both).
5. The 1a/1b env-dump diff.

WHAT COULD MAKE IT FAIL
(a) **The clock cut.**
   - The stroke may be phase-locked to the clock (HANDOFF_PPO.md:142).
   - W's cut cost move_turn survival: 0.865 in the clock-held twin against 0.365 after the cut (MF:250, 262).
   - The c6 pre-cut read separates "never formed" from "lost at the cut".
(b) **Turn and move_turn sharing.**
   - Turn left/right is itself a direction-flipped stroke, and it now competes from t3 at 0.2.
   - Per-hand side weight falls from 0.16 to 0.09.
   - move_turn's nose-dive failure (MF followups: pitch penalty 0, yaw 0.02-0.07 of 0.3 rad/s) is not addressed by B, so move_turn can fail in both lineages.
(c) **Smoothness.**
   - action_rate 0 now also covers turn and move_turn.
   - INC0 shows what action_rate 0 does without the bound: chatter, fft 2.89-4.12 (CHARTER.md:933, 945, 958).
(d) **A weak control.**
   - The C0 lineage is saturated (|μ| 7-9), so non-inferiority against it is an easy bar.
   - W is the real bar, but it is a different recipe and a single seed.
(e) **Noise.** A 0.03 margin is inside the ±0.04 eval resolution, so only "clearly inferior" can FAIL.
(f) **Sample size.** n = 3, so a 2/3 result is fragile.

## Smoothness

Recommendation: do not change action_rate inside P1, and do not spend GPU on it before P1. Run a free CPU read now, and pre-register a post-P1 S-stage.

Why not inside P1:
- FB6k licenses only the B package verbatim.
- A shows that action_rate −0.3 with the same construction stalls the second hand: σ decays and the rods re-saturate (PC/RESULTS.md:110-113).
- Re-introducing it in c6x would confound the clock-cut read and the construction comparison.

Why not before P1: B's jitter looks like stroke amplitude rather than chatter.
- B's fft is 1.42-1.49, the same as C0's 1.40-1.57 (fb6k_read.json:28, 107, 186 vs 265, 344, 423), and below W's suite fft of 1.72-1.83.
- Real chatter looks like INC0: jitter 0.36-0.59 and fft 2.89-4.12 (CHARTER.md:933, 945, 958).
- B's rod odd RMS is 1.60-1.85, against C0's ≈0.5, so B makes bigger strokes.
- B's side jitter (0.152-0.157) is about 1.4× the shipped W's suite side jitter (0.109; W stamp 2026-09-25_20-31-29/scenario_metrics.json, by_category.side). The scenarios and speeds differ, though; B walks about 2× faster.

Free CPU read now (scripts only):
- Compute jitter normalised by stroke amplitude, or the high-band (>3 Hz) action-energy fraction, from the saved raws (`--save-raw`) of the FB6k B and C0 lr stamps and of W's stamp.
- If B's high-band fraction is at or below C0's, the extra jitter is stroke, and the S-stage is only a precaution.

Post-P1 S-stage (pre-registered, B-lineage PASS heads only):
- **Run:** fork each c6x head for +2000 iterations on the c6x overlay, clock hidden, with KRABBY_ACTION_RATE_W restored to −0.3.
- **Option:** −0.1 for 1000 iterations, then −0.3 for 1000. No ramp knob exists (motion_knobs.py:124, a step weight only), and writing one is a code change.
- **Gate:**
  - jitter at or below the C0-lineage median, by suite category;
  - side FAST/FAST kept on both hands;
  - every category within 0.03 of the pre-fork head;
  - B dose still ≤ 1.5. σ decay and re-saturation is A's failure mode.
- **Cost:** 3 × (0.87 + 0.2) ≈ 3.2 GPU-h.

If the user wants smoothness settled before the ≈45 GPU-h, the same probe on the three FB6k B heads (FB pool, +3000 iterations) costs ≈4.5 GPU-h. That is ranked alternative 2.

## Alternatives

### 1. Head-fork c6x probe: fork the three FB6k B model_5999 heads (optionally the C0 heads too) into the c6x pool with the clock hidden for 4000 iterations, W as the reference

It is the cheapest read of P1's two largest risks, applied to strokes that have already formed.
- Does a two-hand side stroke survive the hard clock cut (feet/along class, rod odd RMS)?
- Does it survive sharing the network with turn, move_turn and omni (diag)?
- It also gives a first trained diag/turn read from a stroke-bearing net, against W.

It is a diagnostic, not a pipeline candidate:
- The lineage starts on the FB bed (1b, clock visible, no turn curriculum).
- Turn and move_turn are learned cold in c6x, which confounds the side read.
- The C0 heads have no stroke to lose, so they are a weak control.

Run it before P1 only if the user wants the clock-cut answer for about 1/8 of P1's cost; a FAIL here would save ≈45 GPU-h.

GPU-h: ≈5.8 B-only (3 × (1.73 + 0.2)); ≈11.6 with the C0 heads

### 2. Smoothness-first probe: fork the three FB6k B heads on the FB pool for 3000 iterations with KRABBY_ACTION_RATE_W −0.3

It settles whether a formed two-hand stroke survives restored action-smoothing.
- What is known: A's σ decay and rod re-saturation stalled formation (RESULTS.md:110-113), and A's jitter was 0.043-0.066 (CHARTER.md:908).
- What is unknown: whether action_rate applied after formation erases the stroke.
- Gate: jitter at or below C0's 0.096-0.106, both hands still FAST, and dose ≤ 1.5.

This is a precondition for any eventual bake, and it is independent of P1. It ranks below option 1 because B's fft already equals C0's (no chatter signature), so it de-risks less of P1. Do the free CPU high-band read first.

GPU-h: ≈4.5 (3 × (1.3 + 0.2))

### 3. P1 at the default chain length (t3 3000; 12,000 iterations total), B vs C0 lineage, seeds 16-18

This is the cheapest full-pipeline answer, and it keeps W's exact stage lengths. But FB6k's timing predicts a budget false negative:
- t3 per-hand exposure is 0.16 × 3000 = 480, about half of FB6k's 2/3 completion point (≈900).
- Pure-side exposure before the clock cut totals ≈1020, below FB6k's 3/3 point (≤1200). So the second hand would likely form only after the cut, or not at all.

It is dominated by the recommended re-sized design: it saves ≈8 GPU-h but risks a misleading FAIL.

GPU-h: ≈36 (6 × 12,000 iterations ≈ 31.2 + evals ≈ 5)
