# Why omni side walking is slow (2026-09-28, read-only investigation)

**Status: DIAGNOSIS, not causal proof. Nothing was trained or evaluated** (CPU replay of existing evals only).

**Method.** A zero-GPU workflow: four analyst lenses (reward economy; commands, curriculum and representation;
kinematics, slip and actuation; history and cross-task), each checked by an adversarial verifier. A synthesis ranked the
causes and an independent fable·high check verified the ranking; its corrections are folded in here. Draft:
[REPORT_draft.md](REPORT_draft.md). Probe scripts: [scripts/](scripts/) (copied from `/tmp/claude-1000/sidespeed/`).

## Verdict

- **Omni side speed is not capped by the plant.**
  - The same plant walks sideways at 0.18–0.225 m/s (hold mean) in 12/12 sideways-task heads.
  - The sideways head reaches **0.249 m/s** at a 0.25 m/s command by 8–10 s into a hold.
- **Omni side speed is not capped by the reward pricing slow side walking either.**
  - In steady state, W's own reward weights pay the sideways head's lateral gait about **+0.40/s** more than W's side
    gait (40–50 % of side income).
  - Another +0.05 m/s of side speed pays +0.10/s locally.
- **It is capped by a gait attractor the omni policy falls into: "lean and shuffle".** In the first 2–4 s of every side
  hold:
  - the body shifts about 0.25 m over its planted feet toward the travel direction;
  - that parks the trailing row outboard and folds the leading row inboard;
  - each stance can then sweep only 0.02–0.03 m relative to the body (the sideways head sweeps 0.10–0.14);
  - the feet advance at about **0.10 m/s**, and that is the whole steady side speed.
- **The attractor is closed-loop.** It re-forms at every side onset. At left↔right switches it re-forms in the opposite
  direction from an opposite-lean start.
- **The lean is paid locally, which is a concrete trap mechanism.** Building the lean moves the body fast for about 2 s,
  and W's tracking terms pay that window **+0.23/s above steady state**. That is inside the policy's roughly 2-s
  discount horizon (γ 0.99 at 50 Hz). Every side onset therefore rewards the lean. Its long-run cost, a 0.10 m/s
  plateau, lies beyond what the gradient sees.
- **What keeps omni training in this attractor is not yet identified, but CPU-1 narrows it to the omni task's own
  configuration.** Side speed is 0.03–0.12 m/s across about 89 walking omni heads (sd about 0.015). The attractor forms
  in side-only, unidirectional, un-co-trained 1b, and in from-scratch, co-trained 1c_004, but not in fwd-only 1a. That
  leaves an omni-config constant (clock combine, tracking kernel, facing term, mirror, obs, RSI) as the parsimonious
  suspect, to be bisected in three side-only runs.

## Scale of the gap

| | Side speed at 0.22 | Steady feet-centroid speed | Lean (m) | Stance sweep (m) | Trailing / leading foot vs femur (m) |
|---|---|---|---|---|---|
| W (omni head of record) | 0.10 | 0.095–0.100 | 0.25 | 0.024–0.029 | +0.27 / −0.20 |
| t3 X08 (clock live, from scratch) | 0.10 | 0.092–0.100 | 0.24 | 0.022 | +0.26 / −0.19 |
| Sideways head (creep at 0.25) | 0.22–0.25 | 0.21–0.25 | ≈ 0 | 0.10–0.11 | ≈ 0 / ≈ 0 |
| W fwd (at 0.25) | 0.20 | 0.20–0.22 | — | — | — |

The sideways head is commanded 0.25 m/s, not 0.22, so the comparison is not speed-matched.

## The onset time course (2-s windows from the start of the hold, m/s)

| Window | W body | W feet | W lean build | t3 feet | Sideways head feet |
|---|---|---|---|---|---|
| 0–2 s | 0.139 | **0.037** | +0.099 | 0.046 | 0.100 |
| 2–4 s | 0.111 | 0.084 | +0.026 | 0.085 | 0.207 |
| 4–6 s | 0.097 | 0.095 | 0.000 | 0.092 | 0.234 |
| 6–8 s | 0.098 | 0.100 | −0.001 | 0.097 | 0.246 |
| 8–10 s | 0.093 | 0.096 | −0.003 | 0.100 | 0.247 |

"Lean build" is body speed minus feet speed.
- **W's apparent "fast start" is the body sliding over its feet.** The feet accelerate to about 0.10 and stay there.
- **The sideways head never leans.** Its feet reach 0.21 m/s within 4 s.

## CPU-1 (run 2026-09-28): the attractor across the whole lineage

Script `scripts/syn/cpu1_lineage.py`, output [CPU1_lineage.txt](CPU1_lineage.txt). Side holds are split into 2-s
windows; "lean built" is the body-over-feet shift in the first 4 s.

| Stamp | Setup | Lean built 0–4 s (m) | Steady feet speed (m/s) |
|---|---|---|---|
| **1a_001** | **fwd only** (never trained on side) | **0.04** | **0.033** (barely moves sideways) |
| **1b_002** | **side-only, unidirectional (body +y), no co-training**, forked from 1a, 1000 its | **0.20** | **0.063** |
| 1c_003 | from 1b | 0.03 | 0.006 (collapsed: barely moves) |
| **1c_004** | **from scratch, co-trained, bidirectional** | **0.19** | **0.077** |
| 14 arm stamps + W + X08:t3 (co-trained, bidirectional) | t3 / c6y lineages | 0.17–0.27 | 0.089–0.113 |

- **The attractor is universal in side-trained omni heads.** Every one of the 18 side-trained stamps (excluding the
  collapsed 1c_003) builds the lean in 0–4 s and plateaus at 0.06–0.11 m/s.
- **The lean is learned during side training.** It is not the forward gait's generic sideways response: 1a (fwd only)
  does not lean.
- **Neither co-training nor bidirectionality is necessary.** 1b is side-only and unidirectional, the sideways task's
  convention, and leans.
- **The forward warm start is not necessary either.** 1c_004 is from scratch and leans.
- **What 1b and 1c_004 share is the omni task's configuration and code path (2c).** This is now the parsimonious
  candidate. The alternative, that each run has a different sufficient cause, is not excluded.

## Ranked causes

| # | Cause | Confidence | Status |
|---|---|---|---|
| 1 | **Lean-and-shuffle attractor.** Both leg rows sit at the stance end of their stroke, sweep 0.02–0.03 m, and the feet advance at about 0.10 m/s. | medium | **Primary, proximate.** Reproduces in every omni stamp (1b, 1c, 2b, t3, W, 14 arms) and in no sideways stamp. Re-forms at every onset and l↔r switch. Not yet tested by intervention. |
| 2 | **The omni training regime forms and locks the attractor.** A formation/optimisation trap, not a slow-side incentive. The onset payoff (+0.23/s for about 2 s) is a concrete trap mechanism. | medium (trap), low (which factor) | **Primary, root class.** The responsible factor is unidentified. |
| 2a | **Forward warm start / wide-stance template.** *(CPU-1: not necessary; 1c_004 from scratch leans)* The side gait is built from the forward gait's wide, all-outboard stance (W stand and fwd u ≈ 0.51 on every leg, against the sideways head's stand corners at 0.11). | low | Untested. 1b has no co-training yet leans and is already behind at matched iterations (progress 0.049 vs the sideways 1a's 0.076 at 1000 its). |
| 2b | **Bidirectional side commands.** *(CPU-1: not necessary; unidirectional 1b leans)* Omni trains side_l and side_r; the sideways task is unidirectional (+y only, scalar vx). Every l↔r switch re-pays the onset lean. | low | Untested. Added by the fable check. |
| 2c | **An omni-config constant** *(CPU-1: the one factor 1b and 1c_004 share; now the leading candidate)*. Candidates: clock product vs sum; the xy tracking kernel (std 0.2, L1 −2) vs the travel kernel (std 0.316, L1 −1); the facing reward and relay; the 4-image mirror; obs layout; RSI speed clamp; the lateral band b = 0.25. | low | Untested. Each argued down one at a time; none tested. |
| 3 | **Traction: loaded feet skate along the travel direction.** On omni side the feet move at +0.043 m/s while loaded, 45 % of body speed. The sideways head's figure is about 5 % and W fwd's about 8 %. The leading row carries 63 % of the load. | low–medium | **Open**: a symptom of #1, or a contributor. It explains why extra taps buy no speed. |
| 4 | Minor: the facing-ladder stages (about −0.01 m/s, about 1.6 × seed noise); the reward-floor clip absorbing 11 % of side's marginal tracking income vs 7 % on fwd; taps concentrated on the loaded leading row | low | Minor, or symptoms of #1 |

**Why the cause is not incentive.**
- **Steady-state pricing (startup transient removed).** W's economy pays the sideways gait +0.396/s more:
  - track_lin +0.104, progress +0.071, L1 +0.055;
  - tibia +0.065, delta_torque +0.045, excess_feet +0.038;
  - foot_idle −0.021.
- **The local margin is positive.**
- **Three reward campaigns landed their doses and bought only taps.** At formation the centring bonus moved the lean the
  wrong way.
- **RSI already shows the policy the fast lateral gait** in about 10 % of resets, and the gait is not retained.
- **Caveat.** The +0.40/s is an endpoint comparison against a gait W cannot reach without passing through a re-centring
  transition. The onset payoff (+0.23/s for about 2 s) is what the local gradient sees.

## Ruled out, or reduced to minor

| Candidate | Killed by |
|---|---|
| Plant lateral ceiling at about 0.22 m/s | The sideways head reaches 0.249 at a 0.25 command (8–10 s). Its 0.219 figure was a hold mean including the start from stand. At a 0.35 command it runs slower (0.215): policy calibration, not the plant. |
| Joint limits, rod stroke, torque or velocity caps, A15+B mounting | Same plant at 0.18–0.25 in 12/12 sideways heads; no limit is hit differently |
| Reward prices slow side walking / flat landscape (E1) | +0.40/s steady-state preference; the +0.075 figure was the sideways head's startup transient |
| Instantaneous-velocity kernel penalising jerky lateral gaits (E2) | In steady state the fast gait is no jerkier (sd 0.133 vs 0.14–0.15) |
| `penalty_foot_idle_when_forward` as a brake (E3) | −0.02/s in steady state. Its large training telemetry is counter carry-over: `idle_steps` is reset only at episode reset, not at command resample, and it is as large on stand and turn rows. |
| Forward-era terms on side rows | `_cmd_gate` is command-general under omni. L1 pays for speed. stance_support and mechanical_power have weight 0. |
| Regulariser tax on speed | At the two endpoint gaits every term favours the fast gait except cross_vel and foot_idle (−0.03/s together) |
| Clock cadence law or clock reward | t3 (clock-locked) and W (clock-free) are equally slow; the product clock pays the fast gait more (0.603 vs 0.536) |
| Under-training side | Side has the largest per-category dwell at every stage; speed is flat from 3k to 12k iterations |
| Asymmetric command encoding, facing relay | Encoding is symmetric (raw fwd/side in slots 8/9, \|cmd\| in 10). The relay is active on 7–12 % of side steps. |
| Exploration saturation (omni-specific) | The fast sideways head is equally bang-bang saturated |
| Forking from fwd 1a as the cause of W's deficit | W's lineage starts from the from-scratch t3, and 1c_004 (from scratch) is slow too |
| Lean sets the stride | Within omni the lean varies (0.14–0.21 m) at a flat stride; both rows sit at their stroke ends regardless |
| W side "decays" within a hold | That is lean formation; the feet accelerate and plateau |

## Experiments (nothing has run)

| # | Test | Splits | Cost | Readout |
|---|---|---|---|---|
| CPU-1 | Onset time course (`scripts/syn/win_centroid.py`, `win_posture.py`) on 1a, 1b, 1c_003, 1c_004, 2b and the 14 arm stamps | #1 as attractor; 2a hint | ≈ 20 min CPU | Every side-trained stamp builds about 0.2 m of lean in 2–4 s, then plateaus → #1. 1a (fwd-only) leaning at side onset → forward-template hint. |
| GPU-0 | Eval-only side sweep at 0.15 / 0.22 / 0.30 / 0.35 on W | #1 cap vs band calibration | ≈ 0.2 GPU-h | W feet flat at about 0.10 with lean about 0.25 at every command → attractor cap. W climbing toward 0.18 at 0.30–0.35 with less lean → the band/calibration story revives. (The sideways head needs no run: it reaches 0.249 at 0.25.) |
| **GPU-1** | **Two side-only omni formations from scratch**, t3 economy, 3000 its. Arm U: **unidirectional**, DIR_BAND fixed to +y, RSI off or sideways bank only. Arm B: **bidirectional**, side_l/side_r. Controls exist: t3 X08, B_f s2/s3 (co-trained, 0.10 m/s); sideways 1a (fast). | 2a vs 2b vs 2c | ≈ 3.2 GPU-h + 0.4 eval; free read at it 1000 | U fast, B slow → **bidirectional switching / onset payoff (2b)**. Both fast → **forward warm start or co-training (2a)**. Both slow → **omni-config constant (2c)**. "Fast" means ≥ 0.17 m/s at 0.22 with lean < 0.05 m and sweep ≥ 0.08 m. |
| GPU-2 | Conditional on GPU-1 | | | Both slow → bisect 2c in two bundled arms: R, the sideways reward economy; C, command/obs/mirror/RSI (≈ 3.4 GPU-h). Both fast → a W side-only fine-tune against B s3/s2 to separate interference from formation lock-in (≈ 2 GPU-h). |
| Lever to test once the factor is known | **Onset-masked tracking** (no tracking income for the first about 2 s of a side hold, or on body-over-feet shift) or **longer side dwell**, aimed at the onset payoff | the trap mechanism | ≈ 2 GPU-h | Lean at onset falls and steady feet speed rises past 0.13 → the onset payoff is the trap |
| GPU-3 (optional) | W resume with `KRABBY_OMNI_PROGRESS_W` 2.0 | whether income can buy the transition | ≈ 2 GPU-h | The prior is a TIE plus more taps (as the three reward campaigns). Run only if GPU-1/2 leave the reward route open. |

**Recommended order, revised after CPU-1:** GPU-0, then a **2c bisection run directly in the 1b setting**, not
GPU-1's U/B pair. U/B tests bidirectionality, which CPU-1 already shows is not necessary. The bisection runs side-only
unidirectional omni from scratch, 3000 its, in three arms:
- **O**, the omni config as is: the expected-slow control, to be compared with 1b and the sideways 1a;
- **R**, the sideways formation's reward economy: sum clock, travel kernel std 0.316, L1 −1, facing reward off;
- **C**, the sideways formation's command/obs/mirror/RSI: fore-aft-only mirror, only a scalar travel speed observed,
  sideways RSI bank.

That is about 4.2 GPU-h plus 0.6 eval. An early read at iteration 1000 (progress income against the sideways 1a's 0.076)
is free. Some knobs in R and C need small code changes, first.

## Side notes

- **`penalty_foot_idle_when_forward`'s idle counter carries across command resamples.** It is reset only at episode
  reset, so a stand hold's idle legs are charged at the next hold's onset. This inflates that term's per-category
  training telemetry on every row. It is a small, separate defect worth a look in reward simplification.
- **Episode_RewardCat per-category rates spike on short dwells** (see the per-category stroke CLOSE). Use dwell-weighted
  aggregates.

## Caveats

- **The evidence is observational.** Posture and slowness co-occur across the omni/sideways split, and that split also
  differs in lineage, reward, commands, observations and directionality.
- **The sideways comparisons are not speed-matched.** Sideways heads are commanded at 0.25 or 0.35 m/s, omni at 0.22.
- **The omni stamps are not independent.** Most are re-evaluations within a few lineages.
- **Deterministic eval rollouts.** The pricing uses deterministic eval rollouts, not stochastic training rollouts. The
  stride and yaw terms are omitted (small).
- **Single seed.** GPU runs here would be single-seed. Add a seed only for a result within 0.03 m/s of a boundary.
