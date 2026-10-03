# x6k: implications, limits and the confirm design (opus·high, workflow wf_698962eb-d6d)

## What it means

Paths: PC = crab_hex_omni_task/experiments/2026-10-01_ppo_construction; SB = crab_hex_omni_task/experiments/2026-09-28_side_bisect.

1. **A two-direction net formed both side strokes without any symmetry machinery.** All five unselected C_lr B seeds are FAST/FAST at model_5998 (PC/x6k/x6k_read.md:7-11).
   - The second hands crossed between model_3100 (all INT) and model_4000 (all FAST: 0.222 / 0.196 / 0.250 / 0.174). No first hand dropped, no fork tripped, and the largest fall share is 0.003 (x6k_read.md:7-11).
   - Hold speeds are 0.205–0.277 m/s, with rod odd RMS 1.81–2.28 on every hand (x6k/gate/*_odd.json). The historical FAST band is 1.50–2.16 (SB/HANDOFF_PPO.md:143).
   - The C0 control stayed frozen through the same 3000 iterations: 0.086 / 0.082, +0.001, |μ| 6.71 / 9.77, odd RMS 0.45–0.57 (x6k_read.md:12; C_lr__C0_x6k_s11_odd.json).
   - B used no canonical training and no M_y mirror; the `fa` mirror is M_x only (PC/RESULTS.md:130). So this is the user's preferred lever: a stroke learned per direction from data. That is the precondition for generalising to diagonals, though it is not evidence that it does.
2. **Step 3's PARTIAL was late formation, not a construction ceiling.** RESULTS.md:103-113 read the remaining gap as "rate and lag, not capacity", with B's second hand still rising. The late-rise rule just missed it: s14 and s15 rose +0.027 and +0.028 against a 0.03 threshold (RESULTS.md:108-109).
3. **For the omni pipeline, the formation budget is the binding constraint.** On C_lr each side hand has a pool weight of 0.4, and the second hand needed ≤4000 iterations: ≤1600 iterations × weight.
   - FB_C_lr gives each hand 0.2 (SB/candidates.yaml:402). P1's t3 gives 0.16 (:443).
   - The cumulative side + side_turn exposure over the full P1 chain is ≈1,690 (t3 480, c4 248, c5 210, c6 250, c6x 500; :443-510), which only just reaches C_lr's completion point. Most of it arrives after the clock is cut.
   - So the pre-registered 3000-iteration FB and t3 templates (:397, :438) are probably too short. They would likely read one-FAST and give a false FAIL.
4. **What this implies for the pre-registered 4c path:**
   - **4c is still formally not triggered.** It needed an UNLOCK at 3000 iterations, and step 3 gave B 1/5 (RESULTS.md:30; plan step 12 trigger, foamy-wondering-ullman.md:378). x6k's 5/5 against C0 0/1 at 6000 would match the UNLOCK corner (5/5 vs ≤1/5) only with C0 at 0/5. Even then it is an amended trigger, because the budget was changed after the result was seen. 4c therefore runs only "or the user approves", and the C-bed guard reading (RESULTS.md:167-172) is still pending.
   - **INC0's stop rule is vacuous at 3000.** The rule is INC0 FAST ≥ B's − 1 (plan:385), and with B at 1/5 the threshold is 0, so any INC0 result stops. INC0 only means something at 6000 against B's 5/5.
   - **B carries by default.** A never completed its second hand: its σ decays to about 0.50 (RESULTS.md:110-113), and A at 6000 was never run.
   - **The 4c budget roughly doubles,** from the plan's ≈48 GPU-h (plan:416) to ≈60–75 GPU-h at re-sized iterations.

## Limits

- **The bed is side-only and has a clock.** The C_lr pool is stand/side_l/side_r. The heads use base 1b with the clock visible and the clock reward on (KRABBY_OMNI_CLOCK_HIDE 0, CLOCK_W 1.0 in PC/x6k/gate/C_lr__B_x6k_s12.json), the fa mirror, RSI 0 and ellipse 0.35. No head has trained on fwd, back, turn, diag or move_turn, so none can serve the omni pipeline as is. W is clock-free with a different recipe: 4-image mirror and RSI 0.2 (plan:396-401). The handoff says the side stroke is locked to the tripod phase (SB/HANDOFF_PPO.md:142), so whether it survives c6x's clock cut is untested.
- **Diagonals are not measured at all.** The odd-stroke reading is defined for pure ±y under M_y. A diagonal must blend the cam (forward) and rod (side) strokes. The handoff lists 'diagonals are untested' as a P1 risk (HANDOFF_PPO.md:812).
- **B is a package.** It is the construction plus action_rate weight 0 (CHARTER.md §2 arms table). INC0 (attribution) has not been run, and B's executed jitter/fft (Q block) has not been read. RESULTS.md:173-178 reports only A's jitter. action_rate 0 across every omni category is a real-robot smoothness risk.
- **The C0 control at 6000 is a single seed** (s11; x6k_read.md:12). Its freeze is well characterised: exec change 0.003, σ at the 2.0 clamp in 11/11 runs (RESULTS.md:85-90). But formally the comparison is B 5/5 against C0 0/1 (Fisher not significant), not against 0/5.
- **The window is confounded.** Completion happened between model_3100 and model_4000, inside the 0.038 → 0.1 priv_reg ramp (iterations ≈2999–4760). C0 did not move under the same ramp (x6k_read.md:12), so the ramp alone is not sufficient, but it may interact with B. A from-scratch run has the same ramp timing, so a confirm inherits this confound.
- **Forks are not continuous runs.** They restart the env episodes (task facts), and no from-scratch 6000-iteration B run exists. Seeds 11–15 are spent: same-seed training is bit-identical, so a rerun is a replay (memory same-seed-training-deterministic). Confirms must use 16–18.
- **The eval coverage is narrow.** It holds only 0.22 and 0.25 m/s lateral (FAST ≥ 0.155 / 0.175). No reversal or pause eval was run on these heads, although omni switches hands mid-episode. s15's former second hand (L) is the weakest, at 0.219 mean (x6k_read.md:11). n = 5 per arm is descriptive (CHARTER §10.5).

## Confirm design

This is the smallest confirm that makes the B construction a pipeline candidate. It is 4c step (2) re-budgeted, plus INC0 at a budget where its stop rule means something. Nothing launches without the user.

**0. CPU-only first (free).** Read Q-block action_jitter and action_fft_freq for the five x6k B heads and the C0 head from the existing side22_25lr__omni stamps (record lr_stamp → scenario_metrics.json). This shows what action_rate 0 costs before any GPU spend.

**1. FB6k: from scratch, fresh seeds 16 / 17 / 18. No forks:** seeds 16–18 have no 2999 heads, and same-seed reruns of 11–15 are replays.
- **Bed:** the FB_C_lr template verbatim (SB/candidates.yaml:393-410):
  - base 1b;
  - pool stand / fwd / side_l / side_r / back at 0.2 each;
  - slots 0.2:0.8:0:0, fa, RSI 0, ellipse 0.35, clock visible as on C_lr.
- **Arms:** B (the CHARTER §2 B knobs) and a C0 lineage (incumbent plus hand diag) at the same seeds.
- **Length:** 6000 iterations. That is C_lr's ≤4000-iteration completion point scaled by the halved per-hand weight, ≈8000 at full scaling, read in two steps.
- **Evals** at model_3000, model_4500 and model_5999:
  - both-hands side22_25lr__omni and the driver side22_25;
  - along_cmd and the class (feet AND along, §4.1);
  - the §4.4 tripwire;
  - dose (median |μ| ≤ 1.5 over the last 100 iterations) and odd RMS;
  - the suite (fwd, back, stand ratios);
  - the Q block.
- **Pre-registered extension:** any B arm that is one-FAST at 5999, with its second hand's feet_ss up ≥ 0.03 from 4500, is forked for +3000 iterations. Use the CHARTER §10.3 resume read; the x6k forks validated it. Read it at 8998. C0 is extended only if a hand moved ≥ 0.03.
- **Gate, PASS:** all of the following.
  - B is FAST on both hands, untripped and dosed, in ≥ 2/3 seeds at its final read.
  - C0 is 0/3.
  - B's seed-mean fwd and back suite ratios are no more than 0.03 below the C0 lineage's.
  - The t3 record (fwd 0.558 / back 0.524 / side 0.478, policy/manifest.yaml:113-115) is a reference, not a control.
  - No B run trips.
- **Gate, FAIL:** B ≤ 1/3, or fwd/back more than 0.03 below C0. Stop: the construction does not survive a multi-command pool at this budget.
- **Otherwise:** PARTIAL, reported. No P1 runs on a PARTIAL.

**2. INC0 at 6000, in parallel, for attribution.**
- **Arm:** C0 plus KRABBY_ACTION_RATE_W=0 on the C_lr bed, from scratch, seeds 11–13, 6000 iterations.
- **Comparison:** against B's 3/3 FAST/FAST at seeds 11–13 in x6k.
- **Rule** (plan:384-385, applied at 6000): INC0 FAST/FAST ≥ 2/3 → stop and ask (the flip belongs to action_rate). Otherwise B is labelled "the B package vs C0".

**GPU-h (≈1.3 per 3000 iterations, ≈0.1 per eval set):**

| Item | Calculation | GPU-h |
|---|---|---|
| FB6k | 6 × 2.6 + evals ≈ 1.8 | ≈ 17.4 |
| Extension | ≤ 3 × 1.45 | ≤ 4.4 |
| INC0 | 3 × 2.6 + 0.8 | ≈ 8.6 |
| **Total** | | **≈ 26–30** (≈ 17–22 without INC0) |

**3. Only on FB PASS: P1_C_lr B vs a C0 P1 lineage, seeds 16–18** (plan:396-401).
- **Chain:** t3 → c4 → c5 → c6 → c6x with the clock hidden. Carry every B knob through all stages.
- **Re-size:** extend t3 by +3000 iterations, because t3's per-hand weight is 0.16 against C_lr's 0.4. That makes 15,000 iterations, ≈6.5 GPU-h per arm.
- **Read:** side (both hands), fwd, back, diag, turn and move_turn. Eval with KRABBY_OMNI_CLOCK_HIDE=1.
- **Construction effect:** the difference between the B and C0 lineages.
- **W** (policy/5_c6x_clockfree/model_11995.pt; side ratio 0.473, HANDOFF_PPO.md:808) is a head-vs-head reference only.
- **Gate:** B lineage side FAST on both hands in ≥ 2/3, and no other category more than 0.03 below the C0 lineage.
- **Cost:** ≈ 39 + 3 evals ≈ 42 GPU-h. Stop and ask before any adoption or bake.

## Options (ranked)

### 1. FB6k confirm, B vs C0 at seeds 16-18, from scratch, 6000 iterations with a pre-registered fork to 9000, plus INC0 on C_lr at 6000 (seeds 11-13)

- settles: Whether the B construction forms both side hands at fresh seeds when each hand's pool weight halves (0.4 to 0.2) and fwd/back compete for samples. Whether fwd/back hold against a matched C0 lineage. Whether the flip belongs to the construction or to action_rate 0 (INC0 ≥ 2/3 means stop). This is the plan's 4c step (2) gate, re-sized so it is not a budget false negative.
- GPU-h: ≈26-30 (FB 17.4 + extension ≤4.4 + INC0 8.6); ≈17-22 without INC0
- why this rank: Smallest run that makes B a pipeline candidate. It uses fresh seeds and a multi-command pool, controls against a C0 lineage, and resolves the action_rate attribution, which is meaningless at 3000 iterations (B 1/5 makes the INC0 stop rule fire at a threshold of 0). Its PASS is the required precondition for spending ≈40 GPU-h on P1.

### 2. P1_C_lr, B lineage vs C0 lineage, seeds 16-18, t3 extended to 6000 (15,000 iterations total; only after option 1 passes)

- settles: The user's real question: whether the construction gives fast two-hand side walking inside the full phase-1 chain with the clock hidden. Reads diag, turn, move_turn, fwd and back against both the C0 lineage (the construction effect) and W (reference). It is the only option that measures diagonals.
- GPU-h: ≈42 (6 × 6.5 + evals); ≈34 at the default 12,000-iteration chain, but default t3 per-hand exposure (0.16 × 3000) is ≈0.4× C_lr's, so it risks a late second hand
- why this rank: Highest value, but highest cost and highest risk. Cumulative per-hand side exposure across the default chain (≈1,690) barely reaches C_lr's completion point, and most of it comes after the clock cut. Run it only behind the FB gate.

### 3. Head-fork probe: fork x6k B heads s12 and s14 (model_5998) plus the C0 x6k head into the c6x pool with the clock hidden for 4000 iterations

- settles: The cheapest read on whether an already-formed two-hand stroke survives the clock cut, and survives sharing the network with fwd/back/turn/omni (diag) commands: rod odd RMS and side speed kept or lost. It also gives a first diag and turn read from a stroke-bearing net, against W as a reference.
- GPU-h: ≈6 (3 × (1.73 + 0.3 evals))
- why this rank: A diagnostic, not a candidate route. The lineage (side-only 1b start, clock visible) is not the pipeline recipe. Learning fwd/back/turn from scratch confounds the side read, and C0 is a weak control because it has no stroke to lose. Useful only if the user wants the clock-cut and diag risk read before paying for P1.

### 4. Complete the x6k C0 control: fork C0 heads s12-s15 (model_2999) for +3000 iterations

- settles: Moves the 6000-iteration comparison from B 5/5 vs C0 0/1 to 5/5 vs 0/5 on the same seeds (two-sided Fisher p ≈ 0.008), closing the n=1 control limit. That makes x6k a 6000-iteration UNLOCK-shaped result for the user's 4c approval.
- GPU-h: ≈5.8 (4 × 1.45)
- why this rank: Cheap, but mostly a formality. The C0 freeze (σ at the 2.0 clamp in 11/11 runs, exec change 0.003, +0.001 over 3000 forked iterations) already predicts SLOW. It is still an amended, post-hoc-budget trigger, so 4c needs the user either way. And it adds nothing on fwd/back/diag, so it does not move B closer to the pipeline.

### 5. C_lr fresh-seed replication: B vs C0 from scratch at seeds 16-18, 6000 iterations

- settles: That 'late formation' replicates at fresh seeds, on from-scratch continuous runs with no env restart at 2999, on the bed where it was found.
- GPU-h: ≈16.5 (6 × 2.6 + evals)
- why this rank: Dominated by option 1. It spends the reserved confirm seeds on a bed the pipeline never uses, and the 11-15 result was already unselected (5/5 of all B seeds). Prefer it only if the user wants a clean pre-registered UNLOCK at 6000 before any multi-command bed.
