# PPO construction on the side beds: results (step 3, final)

2026-10-03. The pre-registered rules are in [CHARTER.md](CHARTER.md), §4, and the dated log is §9. The approved plan is
`~/.claude/plans/foamy-wondering-ullman.md` (RL review steps 2–4).

- **Machine reads:**
  - `verdict/verdict.{json,md}` (`scripts/verdict.py`);
  - `verdict/mechanism_reads.{json,md}` (`scripts/mechanism_reads.py`);
  - every per-run class, along-command, model_2500 and odd-RMS JSON in `gate/`.
- **Checks:** an independent fable·high re-derivation of every verdict and an opus·high implementation
  review (final-read workflow `wf_594e4e5d-d98`). Both reproduce every verdict line; see the checks section.
- **This file vs `REPORT.md`:** the candidate driver regenerates `REPORT.md` (its ladder table). This file is
  the campaign's write-up.

## The answer

The question was whether PPO-construction changes let a two-direction network form the direction-flipped side
stroke without breaking the one-direction gait. The changes are a bounded mean, a gradient-passing σ at 0.8 and
lower entropy (arm **A**), and the same plus action_rate weight 0 (arm **B**).

| Bed | Pre-registered verdict | Construction arms | Incumbent C0 |
|---|---|---|---|
| **C** (one direction): guard | **PASS** (A FAST 3/3)¹ | A FAST 3/3 (0.213–0.260 m/s, lean ≈ 0) | FAST 1/3 (SLOW, FAST, INT) |
| **C_lr** (both directions) | A **PARTIAL**, B **PARTIAL** (inconclusive) | B FAST 1/5; **5/5 B runs have a FAST hand**. A FAST 0/3; 1/3 has a FAST hand | 0/5, SLOW/SLOW every run |
| **C_lr_rsi** (C_lr + RSI 0.2) | A **PARTIAL**, B **PARTIAL** (inconclusive) | A 0/3 (2/3 with a FAST hand); B 0/3 (3/3 with a FAST hand) | 0/3, SLOW/SLOW every run |

¹ The pre-registered record also reads "C bed not reliable at fresh seeds (C0 1/3)". Under the literal §4.7
text the guard would be UNDEFINED; see decision 1 below.

- **No UNLOCK on any bed,** so 4c is not triggered. UNLOCK needed 4/5 against C0's 0/5; C_lr B reached 1/5.
- **No 4a trigger:** the A/B LR floor share is 0.50–0.67 (the trigger is above 0.9), and the A/B applied LR is
  0.55–0.82× C0's (the trigger is below 0.5×).
- **No 4b trigger:** 1,024 envs cost 1.21× per iteration (rung 1).
- **The approved plan ends at step 3.** Step 4 runs nothing.

**In one sentence:** the construction is a real limiter but not the only one. Under the incumbent the
direction-flipped stroke never formed on a two-direction bed (0 of 16 hands). Under A/B it formed on 12 of 28
hands, and in 1 of 14 runs on both hands: C_lr__B_s12, the first two-direction network ever FAST/FAST
([video](../../../../../logs/videos/ppoc_C_lr_B_s12_categories.mp4), not in git).

## Per-run classes (generated from `gate/`)

Feet_ss in m/s at the 0.22 / 0.25 holds of the both-hands eval (`side22_25lr__omni`). The C bed uses the driver
eval (lat_l only). A hand is FAST only if its feet and along-command classes are both FAST (§4.1). Rod odd RMS
measures the direction-flipped stroke; historical two-direction networks reach at most 0.93.

| bed | arm | seed | L class | L feet 0.22 / 0.25 | R class | R feet 0.22 / 0.25 | rod odd L / R | run | model_2500 L / R |
|---|---|---|---|---|---|---|---|---|---|
| C | C0 | 11 | SLOW | 0.083 / 0.092 | — | — | 0.65 | not FAST | — |
| C | C0 | 12 | FAST | 0.185 / 0.201 | — | — | 1.31 | **FAST** | — |
| C | C0 | 13 | INT | 0.147 / 0.154 | — | — | 1.36 | not FAST | — |
| C | A | 11 | FAST | 0.229 / 0.251 | — | — | 2.06 | **FAST** | — |
| C | A | 12 | FAST | 0.213 / 0.217 | — | — | 2.09 | **FAST** | — |
| C | A | 13 | FAST | 0.241 / 0.260 | — | — | 2.31 | **FAST** | — |
| C_lr | C0 | 11 | SLOW | 0.080 / 0.088 | SLOW | 0.077 / 0.086 | 0.60 / 0.42 | not FAST | SLOW / SLOW |
| C_lr | C0 | 12 | SLOW | 0.091 / 0.100 | SLOW | 0.078 / 0.087 | 0.69 / 0.59 | not FAST | SLOW / SLOW |
| C_lr | C0 | 13 | SLOW | 0.063 / 0.077 | SLOW | 0.074 / 0.072 | 0.38 / 0.25 | not FAST | SLOW / SLOW |
| C_lr | C0 | 14 | SLOW | 0.079 / 0.085 | SLOW | 0.075 / 0.078 | 0.40 / 0.30 | not FAST | SLOW / SLOW |
| C_lr | C0 | 15 | SLOW | 0.064 / 0.079 | SLOW | 0.090 / 0.092 | 0.50 / 0.70 | not FAST | SLOW / SLOW |
| C_lr | A | 11 | SLOW | 0.088 / 0.095 | INT | 0.168 / 0.158 | 0.77 / 1.51 | not FAST | SLOW / INT |
| C_lr | A | 12 | FAST | 0.178 / 0.187 | SLOW | 0.086 / 0.084 | 1.67 / 0.79 | not FAST | FAST / SLOW |
| C_lr | A | 13 | SLOW | 0.073 / 0.084 | SLOW | 0.093 / 0.101 | 0.65 / 1.14 | not FAST | DEAD / SLOW |
| C_lr | B | 11 | FAST | 0.233 / 0.247 | INT | 0.140 / 0.158 | 1.85 / 1.67 | not FAST | FAST / SLOW |
| C_lr | B | 12 | FAST | 0.245 / 0.273 | FAST | 0.189 / 0.197 | 2.32 / 1.96 | **FAST** | FAST / FAST |
| C_lr | B | 13 | FAST | 0.221 / 0.235 | INT | 0.132 / 0.143 | 1.95 / 1.64 | not FAST | FAST / SLOW |
| C_lr | B | 14 | FAST | 0.218 / 0.244 | INT | 0.138 / 0.151 | 1.95 / 1.12 | not FAST | FAST / SLOW |
| C_lr | B | 15 | INT | 0.156 / 0.174 | FAST | 0.237 / 0.258 | 1.74 / 1.96 | not FAST | SLOW / FAST |
| C_lr_rsi | C0 | 11 | SLOW | 0.072 / 0.078 | SLOW | 0.102 / 0.109 | 0.47 / 1.06 | not FAST | SLOW / SLOW |
| C_lr_rsi | C0 | 12 | SLOW | 0.079 / 0.087 | SLOW | 0.054 / 0.059 | 0.79 / 0.90 | not FAST | SLOW / SLOW |
| C_lr_rsi | C0 | 13 | SLOW | 0.067 / 0.070 | SLOW | 0.065 / 0.070 | 0.70 / 0.78 | not FAST | SLOW / SLOW |
| C_lr_rsi | A | 11 | SLOW | 0.090 / 0.096 | FAST | 0.223 / 0.236 | 0.61 / 2.07 | not FAST | SLOW / FAST |
| C_lr_rsi | A | 12 | SLOW | 0.090 / 0.103 | SLOW | 0.095 / 0.089 | 0.35 / 0.25 | not FAST | SLOW / SLOW |
| C_lr_rsi | A | 13 | SLOW | 0.094 / 0.107 | FAST | 0.228 / 0.226 | 0.87 / 1.68 | not FAST | SLOW / FAST |
| C_lr_rsi | B | 11 | FAST | 0.240 / 0.242 | INT | 0.141 / 0.156 | 1.88 / 1.44 | not FAST | FAST / SLOW |
| C_lr_rsi | B | 12 | FAST | 0.224 / 0.248 | INT | 0.153 / 0.156 | 1.92 / 1.88 | not FAST | FAST / INT |
| C_lr_rsi | B | 13 | SLOW | 0.127 / 0.140 | FAST | 0.238 / 0.267 | 1.40 / 1.99 | not FAST | SLOW / FAST |

C__A_s11 is excluded: it was dose-killed under the pre-amendment rule, and its rerun C__A_s11_r2 counts (§9).
No run tripped the survival tripwire (every threshold is 0.15; the largest A/B fall share is 0.040).

## What the telemetry says (descriptive, §4.9; `verdict/mechanism_reads.md`)

**Strong:** consistent over 28 runs, three beds and deterministic seeds.

- **The incumbent freezes early.** C0's executed policy stops changing by about iteration 500–1000:
  - its training track ratio is flat after that;
  - its executed-mean change per update is 0.003;
  - its median |μ| at 3000 is 4.2–5.8 (cam) and 6.9–8.6 (rod), with 91–97 % of rows beyond 1.5;
  - σ reaches the 2.0 clamp in all 11 C0 runs.

  This matches the RL review's "phantom update" picture: larger parameter steps that buy almost no executed
  change.
- **A and B keep learning.**
  - Executed change per update is 2× (A) and 4× (B) that of C0, while the applied LR is lower.
  - Median |μ| at 3000 is ≈ 0.9 / 1.4 for A and ≈ 0.25 / 0.97 for B.
  - σ holds at 0.74–0.76 in B and decays to 0.50–0.55 in A.
- **That is enough to form the stroke on one direction, and more reliably than the incumbent:**
  - C bed: A 3/3 vs C0 1/3;
  - two-direction beds: at least one FAST hand in 11/14 A/B runs vs 0/8 for C0.
- **B's non-FAST hands carry far more stroke than C0's.** Their rod odd RMS is 1.12–1.88, about 2–3× any C0
  hand (0.25–1.06).

**Moderate:** what remains after unfreezing looks like rate and lag, not capacity.

- **B's first hand forms 2.5–3× later than one-direction A.** Track ratio reaches 0.55 at 1,250–2,250
  iterations, against 500–750 for A on the C bed.
- **B's second hand is still rising at 3000.** It went SLOW → INT between model_2500 and model_2999 in 5 of 7
  one-FAST runs, and the training curves rise +0.04 to +0.13 track per 1000 iterations. The pre-registered
  late-rise rule (above 0.03) did not fire; the nearest were B s14 and s15 at +0.027 and +0.028.
- **A's second hand is flat over the same window.** A and B differ only in action_rate weight (−0.3 vs 0).
  The difference that tracks the stall is σ: A's decays and its rods re-saturate onto the hinge (three
  "dose lost late" flags), while B holds σ at 0.75. So exploration, not the bound, is what keeps the second
  hand moving.

**Weak but real:** the hands are not independent.

- **B gives exactly one FAST hand in 7/8 runs.** That is P ≈ 0.03 under per-hand independence; the test is
  post hoc and pools the two beds.
- **A first-hand lag fits better than independent hands.** A shared backbone carries one direction's stroke
  first, and the other is learned as a correction on about 40 % of the samples.
- **Which hand forms first is not predictable** from early per-hand advantage (FAST − other: −0.002 [−0.018,
  0.014]). With deterministic training, it is a function of the seed.

**Not supported:**

- **Strong gradient cancellation, hypothesis (i):** `PPO/hand_grad_cos` is ≈ 0 in every arm, the same whether
  or not the arm forms the stroke. The statistic is weak: it is a whole-backbone cosine, so it does not rule
  out a small-subspace (i).
- **A capacity limit:** the second hand is never worse than C0's hands.
- **A mirror-loss role:** the `fa` mirror is M_x only, so it cannot couple the ±y hands.

**Cannot decide from these data:** when each hand forms during training (only model_2500 and model_2999 were
evaluated), and whether B's second hand would reach FAST by 6000 or plateau at INTERMEDIATE.

## Checks, corrections and incidents

- **Independent fable·high check:** it re-derived every class, exclusion, tripwire, dose, extension, verdict, the
  guard, 4a/4b/UNLOCK and Q from `gate/` and the tfevents before reading verdict.py. It agrees on every line, and
  no finding changes a verdict. Its rulings:
  - **guard:** PASS with the record string attached (the UNDEFINED reading is textually admissible: your call);
  - **C__A_s11's superseded dose kill:** consistent with your 2026-10-01 amendment;
  - **C_lr_rsi__B_s12's `_e2` retry:** valid;
  - **"adopt for motion quality":** a proposal for C_lr_rsi A only (decision 2).
- **Opus·high implementation review:** two confirmed mismatches.
  1. The guard precedence: the same question as decision 1.
  2. "Q better" compared against all five C_lr C0 seeds; §4.6 pre-registers 3 v 3 at seeds 11–13. Fixed, and
     it changes no line.
- **Post-result verdict.py edits, none of which changes a verdict:**
  - the superseded-dose-kill rule (logged before the final read);
  - the guard's record string;
  - "Q better" restricted to the screen seeds;
  - the share of |μ| > 1.5 reported per run (the 4.3 amendment).
- **Incidents (§9):**
  - My concurrent video recording caused a GPU OOM at the launch of C_lr_rsi__B_s12. The driver retried as
    `_e2`, the runner was fixed to follow retries (a605c9e2), and the post-hoc wiring and dose reads pass.
  - Same-seed training was found to be bit-identical: C__A_s11 and its rerun match through iteration 1140.
    The `_e2` run therefore probably equals the uninterrupted run. That is an inference: the recorder shared
    the GPU during its first hour.
- **§9 corrections:**
  - the 00:55 entry's "31 runs" is 29 records (28 counted + 1 excluded);
  - C__A_s13 and C_lr__A_s13 also carry "dose lost late" (final rod median 1.625 / 1.647; the dose is read at
    iteration 1000, so their class counts);
  - a few seed-11 log entries quoted model_2500 speeds. The table above is the record.

## Decisions for you

1. **The C-bed guard reading.**
   - **PASS:** verdict.py and the independent check. It reads the guard's question as "does A still form the
     one-direction stroke?", and A 3/3 answers it.
   - **UNDEFINED:** the literal §4.7 text, because C0 is at 1/3.

   Only decision 2 depends on this; 4c is unreachable either way.
2. **"Adopt for motion quality": a proposal for C_lr_rsi A** (§4.10).
   - **The read:** A's executed action jitter is below C0's at all three seeds (0.046–0.056 vs 0.087–0.118), its
     fft is lower, and no run tripped.
   - **I recommend declining it.** The arm is 0/3 FAST on that bed. The lower jitter comes from the bounded
     mean and lower σ, which every construction arm has on every bed, not from a better gait. A's σ also decays
     enough to stall its second hand.
   - **Any adoption is a bake,** and it stays your call.
3. **What next.** This is outside the approved plan, so it needs your go-ahead. Ranked by how much each resolves,
   with cost at one process at a time:

| # | Experiment | What it settles | GPU-h |
|---|---|---|---|
| 1 | **B to 6000 iterations on C_lr** (4 one-FAST B heads plus 1 C0 control, resumed from model_2999 after a 20-iteration resume smoke; or from scratch at 2× cost) | Whether the second hand completes ("late formation") or plateaus. A B head FAST/FAST at 6000 would be the first two-direction candidate. | ≈ 6 (≈ 11 with C_lr_rsi) |
| 2 | **Checkpoint-series evals of the existing runs** (model_1000/1500/2000; B, the one-FAST A runs, 2 C0) | Per-hand formation times, which no training tag records: lag vs interference vs independent slow learners. No training. | ≈ 2–3.5 |
| 3 | **T2 interaction feature under B** (tanh(v_y)·sin/cos φ obs, vs a sign(v_y) control) | The only designed split between (i) and (ii). Needs about 60–100 lines of code. | ≈ 5 + code |
| 4 | **B + four-image mirror (M_y coupling)** | Whether an M_y-equivariance loss transfers the first hand's stroke to the second. **It relies on the robot's lateral symmetry, the objection you raised to canonical training**, so it ranks last. | ≈ 4 |

The non-triggered 4a (B at a fixed LR) is a cheaper but confounded proxy for experiment 1. 4b (batch) does not
discriminate between the hypotheses.
