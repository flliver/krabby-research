# x6k: B to 6000 iterations on C_lr (results)

2026-10-03. Pre-registered in [CHARTER.md §10](../CHARTER.md) before any fork ran, and reviewed before launch. Each run forks a
step-3 C_lr head at `model_2999` and trains it for 3000 more iterations. The reading comes from `scripts/x6k_read.py`
(`x6k_read.{json,md}`), committed before any head was read. The records are in `gate/`. An independent fable·high
check (workflow `wf_698962eb-d6d`) re-derived every number from the raw files and **PASSES**.

## The answer: LATE FORMATION

Step 3 left two-direction B one hand short at 3000 iterations. Given 3000 more, **every one of those second hands
completed**. The incumbent stayed frozen.

| fork | role | model_2999 | model_3100 | model_4000 | model_5000 | head (5998) | rise L / R | fall | dose cam/rod (last 100) | priv_reg | result |
|---|---|---|---|---|---|---|---|---|---|---|---|
| C_lr__B_x6k_s11 | one-FAST | FAST 0.240 / INTE 0.149 | FAST 0.245 / INTE 0.144 | FAST 0.247 / FAST 0.222 | FAST 0.250 / FAST 0.244 | FAST 0.237 / FAST 0.224 | -0.003 / +0.075 | 0.000 | 0.27/1.00 | 0.1000 | completes; **candidate** |
| C_lr__B_x6k_s12 | FAST/FAST | FAST 0.259 / FAST 0.193 | FAST 0.250 / FAST 0.224 | FAST 0.238 / FAST 0.210 | FAST 0.258 / FAST 0.250 | FAST 0.260 / FAST 0.262 | +0.002 / +0.069 | 0.000 | 0.28/1.00 | 0.1000 | STABLE; **candidate** |
| C_lr__B_x6k_s13 | one-FAST | FAST 0.228 / INTE 0.137 | FAST 0.233 / INTE 0.148 | FAST 0.232 / FAST 0.196 | FAST 0.231 / FAST 0.236 | FAST 0.240 / FAST 0.239 | +0.012 / +0.101 | 0.003 | 0.30/1.00 | 0.1000 | completes; **candidate** |
| C_lr__B_x6k_s14 | one-FAST | FAST 0.231 / INTE 0.144 | FAST 0.249 / INTE 0.161 | FAST 0.248 / FAST 0.250 | FAST 0.261 / FAST 0.255 | FAST 0.266 / FAST 0.251 | +0.035 / +0.107 | 0.000 | 0.28/1.03 | 0.1000 | completes; **candidate** |
| C_lr__B_x6k_s15 | one-FAST | INTE 0.165 / FAST 0.248 | INTE 0.163 / FAST 0.242 | FAST 0.174 / FAST 0.262 | FAST 0.188 / FAST 0.235 | FAST 0.219 / FAST 0.255 | +0.055 / +0.007 | 0.000 | 0.31/1.07 | 0.1000 | completes; **candidate** |
| C_lr__C0_x6k_s11 | C0 control | SLOW 0.084 / SLOW 0.081 | SLOW 0.085 / SLOW 0.080 | SLOW 0.086 / SLOW 0.082 | SLOW 0.086 / SLOW 0.080 | SLOW 0.086 / SLOW 0.082 | +0.001 / +0.001 | 0.025 | 6.71/9.77 | 0.1000 | FROZEN |

Hands are L / R: class and feet speed in m/s (the mean of the 0.22 and 0.25 m/s holds, both-hands eval).

- **One-FAST forks:** 4/4 complete. The second hands rose by +0.055 to +0.107 and crossed into FAST between model_3100
  (all still INT) and model_4000 (all FAST). The first hands stayed FAST throughout.
- **No drops and no resume artifact:** every fork passed the resume read (it continues its head), and every hand that
  was FAST at 2999 was still FAST at 3100.
- **s12 (already FAST/FAST): STABLE.** It reads 0.260 / 0.262.
- **C0 control: FROZEN.** It reads 0.086 / 0.082 (+0.001 / +0.001) over 3000 iterations, with median |μ| rising to
  6.7 / 9.8. Its two-direction SLOW is a freeze, not slowness.
- **Two-direction candidates:** all five B forks. Each is FAST/FAST with falls ≤ 0.003 and median |μ| 0.27–0.31 (cam) /
  0.99–1.07 (rod). Before this campaign no network had ever been fast in both side directions.
- **No symmetry machinery:** no canonical training and no M_y mirror (the `fa` mirror is M_x only). Each direction's
  stroke was learned from data.

## Independent check (fable·high)

It reproduces every class, feet mean, rise, fall share, resume read and the reading itself. All 13 rule applications
agree, and no head is marginal: the smallest distance to a FAST line is +0.050, and the eval CIs are ±0.003. How it
rates the alternative explanations:

| Alternative explanation | Strength | Why |
|---|---|---|
| The priv_reg ramp (0.038 → 0.1, iterations 2999–4760) did it | **open** | The second hands were already rising before the fork, and s15 completed late at a flat coefficient. The term is small (~8e-5 against a surrogate of ~1e-2), and C0 did not move under the same ramp. No B fork ran with the ramp frozen, so a contribution cannot be excluded. A from-scratch run has the same ramp timing. |
| The env restart at the fork did it | weakens | No jump at model_3100. The rise comes in 3100–4000, after the episodes desynchronise. A restart-triggered trajectory change cannot be excluded without a continuous run. |
| Eval noise at the class line | rules it out | The smallest margin is +0.050, the smallest completion is +0.055, and feet and along-command classes agree at all 36 reads. |
| Selection | weakens | All five C_lr B seeds were forked, so the count is unselected. It is conditional on heads whose formation was already under way, and B was chosen over A on the step-3 count. |
| A from-scratch 6000-iteration B run would reproduce it | **open** | Likely: no schedule keys on max_iterations, and the ramps are keyed on the update counter, which carries. But no such run exists, and a fork is not a replay of a continuous run. |
| It holds beyond the C_lr bed | **serious (scope)** | The pool was stand / side_l / side_r only. With fwd, back and turn competing, each side hand's samples roughly halve. Nothing here speaks to the full omni pool or to diagonals. |

Corrections it raised:
- **The C0 sentence in §10.5:** the step-3 freeze *onset* (iterations 500–1000) was at priv_reg 0, but the heads at 2999
  were at 0.0383. The FROZEN outcome is unaffected.
- **The heads are single checkpoints:** s11 fell −0.013 / −0.020 between model_5000 and the head (still FAST by ≥ +0.05).
  Any bake decision should look at more than one late checkpoint.

## Motion quality: what action_rate 0 costs

Executed action jitter and fft come from the Q block of the both-hands stamps:

| Arm | Iterations | Jitter | FFT |
|---|---|---|---|
| B | 6000 | 0.117–0.155 | 1.07–1.37 |
| B | 3000 | 0.102–0.135 | 1.01–1.30 |
| C0 | 3000 | 0.065–0.104 | 1.29–1.48 |
| C0 | 6000 (fork) | 0.079 | 1.41 |
| A | 3000 | 0.043–0.066 | 0.95–1.07 |

- **B is about 1.6–2× C0's jitter,** and it rose from 3000 to 6000 in 4/5 seeds. That is the price of B's
  `action_rate` weight 0. It matters for the real robot, so any pipeline version of B should re-introduce the
  action-rate term once the stroke has formed. That is a new arm, not tested here.
- **A is the smoothest arm.** It has the same construction with action_rate −0.3, but its second hand stalled at 3000
  (σ decayed to about 0.5), and A was never run at 6000.

## What it means

- **The step-3 PARTIAL was late formation, not a construction ceiling.** The construction unfreezes learning, and B
  then forms each direction's stroke in turn. The second takes 1000–2000 more iterations on this bed, at a per-hand
  pool weight of 0.4.
- **For the pipeline the binding constraint is the formation budget.** FB's pool gives each side hand 0.2 and P1's t3
  gives 0.16. Under the pre-registered 3000-iteration FB / t3 templates, B would probably read one-FAST and give a false
  FAIL. Any confirm needs about 6000 iterations at the start of the chain.
- **The pre-registered 4c path is still formally untriggered.** It needed an UNLOCK at 3000 iterations. x6k's 5/5
  (against C0 0/1 at 6000) is a result with an amended budget, so 4c and anything beyond it is your call.

## Decisions for you

1. **What next (ranked).** The independent assessment is in `x6k/next_options.md`. Nothing launches without you.

   | # | Option | What it settles | GPU-h |
   |---|---|---|---|
   | 1 | **FB6k confirm:** B against a C0 lineage, from scratch, seeds 16–18, 6000 iterations, on the FB pool (stand / fwd / side_l / side_r / back), with a pre-registered fork to 9000 for any B run still one-FAST. Plus **INC0 at 6000** (C0 + action_rate 0, C_lr, seeds 11–13), to attribute the flip to the construction or to action_rate. | Whether B forms both side hands at fresh seeds when each hand's samples halve and fwd/back compete; whether fwd/back hold up; whether the flip is the construction's. The precondition for P1. | ≈ 26–30 (≈ 17–22 without INC0) |
   | 2 | **P1_C_lr:** the B lineage against a C0 lineage, seeds 16–18, with t3 extended to 6000, clock hidden; read side, fwd, back, **diag**, turn and move_turn against W. Only if option 1 passes. | Your real question, including the diagonals. | ≈ 42 |
   | 3 | **Head-fork probe:** fork x6k heads s12 and s14 into the c6x pool, clock hidden, 4000 iterations. | Whether an already-formed two-hand stroke survives the clock cut and sharing the network with fwd/back/turn; a first diagonal read. A diagnostic, not a pipeline route. | ≈ 6 |
   | 4 | **Complete the C0 control:** fork C0 s12–s15 for +3000 iterations. | Moves x6k from 5/5 against 0/1 to 5/5 against 0/5 (p ≈ 0.008). Mostly a formality. | ≈ 6 |

2. **Still open from step 3:** the C-bed guard reading and the motion-quality proposal for C_lr_rsi A. See
   [RESULTS.md](../RESULTS.md).
3. **Smoothness:** whether a smoother B (action-rate re-introduced after formation) should be part of option 1.
