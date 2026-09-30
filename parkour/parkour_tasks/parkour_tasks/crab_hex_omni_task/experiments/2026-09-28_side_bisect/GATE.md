# Side-speed bisection: gate (plan step 10), 2026-09-29

## Answer

In side-only omni (preset 1b, trained from scratch), the factor that forms the slow side gait is in arm C's bundle:
the sideways task's command, obs, mirror, RSI and band settings. Whether it carries over to full omni is untested.

- **Within the bundle, at seed 3, only the one-direction pair moves the gait on its own.** The pair is the fore-aft-only
  mirror plus the direction-blind command obs.
- **At seed 2 the pair alone matches C.**
- **The other single pieces stay slow:** the wider band alone, RSI-off alone and R's four reward knobs. Each is one
  seed.
- **C itself is FAST at both seeds.**

## Status

Stopped by the user after C_mo_s2 for the write-up.
- **C_mo's two seeds disagree** (INTERMEDIATE vs FAST).
  - The pre-registered decision default is a seed-1 tie-break.
  - It was not run because of the stop, which is a recorded deviation.
- **The level-2 arms (C_fa, C_nb, C_nr)** are designed and dry-run clean (`scripts/candidates_v4_dryrun.txt`). They
  were not applied or launched.
- **Nothing is adopted.** No default changed; the knobs stay default-off.

**Sources.**
- Plan: `~/.claude/plans/side-speed-bisection.md` (audited, cap 2).
- Charter and log: [CHARTER.md](CHARTER.md). Pre-arm checks: [CHECKS.md](CHECKS.md). Scripts: [scripts/](scripts/).
- Per-arm outputs: `parkour/outputs/omni_plan/side_bisect/arms/*.json`, including `lean_abs_cadence.json`.
- Checked independently: a fable·high recompute and an opus·high skeptic, 2026-09-29. Their corrections are folded in
  below.

## Setup

- **Arms.** Every arm is omni preset 1b from scratch: side-left only (DIR_BAND π/2), 20 % stand, facing held,
  3000 its. Each ran as a `references:` chain.
- **Eval.** Every arm was scored on the gate row `side22_25__omni`: lat_l holds at 0.22 and 0.25 m/s, 2-s settle,
  12-s holds, about 100 episodes each.
- **Statistic.** feet_ss is the feet-centroid speed along the hold displacement over hold-relative [4, 10) s. The class
  reads the median over episodes.
- **Gate (pre-registered):**

  | Class | Rule |
  |---|---|
  | FAST | ≥ 0.155 at 0.22 and ≥ 0.175 at 0.25 |
  | SLOW | ≤ 0.13 and ≤ 0.145 |
  | DEAD | < 0.05 |
  | INTERMEDIATE | anything else → seed 2 |

- **Checks, all PASS** ([CHECKS.md](CHECKS.md)):
  - knob smokes;
  - W harness feet_ss 0.096 (target 0.095 ± 0.01);
  - sideways FAST pin at model_3000: 0.196/0.193 (s3) and 0.201/0.202 (s2).
- **Overlays verified.** Each arm's `params/*.yaml` carries exactly its stated overlay, and each eval `run_meta` carried
  the right env. `KRABBY_OMNI_CMD_OBS=scalar` is on the four pair arms only, and the mirror knob is never on an eval.

## Results

| Arm | Change from O | Seed | feet_ss 0.22 [IQR] | feet_ss 0.25 [IQR] | n | Class |
|---|---|---|---|---|---|---|
| O | — (1b preset) | 3 | 0.084 [0.068, 0.101] | 0.094 [0.083, 0.101] | 98/96 | **SLOW** |
| C | mirror fa + cmd obs scalar + RSI off + band 0.35 | 3 | 0.215 [0.196, 0.230] | 0.214 [0.195, 0.231] | 90/93 | **FAST** |
| C_s2 | same as C | 2 | 0.194 [0.180, 0.212] | 0.209 [0.193, 0.217] | 90/95 | **FAST** |
| R | clock sum, track σ² 0.1, L1 −1, yaw_progress 0 | 3 | 0.075 [0.066, 0.083] | 0.078 [0.071, 0.088] | 97/95 | **SLOW** |
| C_mo | mirror fa + cmd obs scalar | 3 | 0.162 [0.140, 0.182] | 0.173 [0.158, 0.188] | 100/100 | **INTERMEDIATE** (0.002 short) |
| C_mo_s2 | same as C_mo | 2 | 0.203 [0.191, 0.214] | 0.210 [0.190, 0.222] | 95/98 | **FAST** |
| C_band | band 0.35 only | 3 | 0.093 [0.084, 0.103] | 0.100 [0.087, 0.109] | 96/95 | **SLOW** |
| C_rsi | RSI off only | 3 | 0.069 [0.015, 0.084] | 0.077 [0.017, 0.090] | 96/99 | **SLOW** |

- No arm was DEAD.
- The n below 100 are episodes that fell before 8 s into the hold. The exception is C_rsi at 0.22, where 2 of the 4
  moved under 0.05 m ("static"). Counting those, 36/98 of C_rsi's episodes stall.

## Mechanism labels (labels only, never gates)

Each cell gives the 0.22 hold / the 0.25 hold.

| Arm | Lean (m) | Episodes leaning > 0.15 m | Stance sweep (m) | Cadence (Hz) | Stall (< 0.03) | Early read (its 800–1000) | Final track_ratio |
|---|---|---|---|---|---|---|---|
| O | 0.183 / 0.197 | 94 % / 100 % | 0.042 / 0.047 | 0.83 / 0.86 | 0 | 0.564 | 0.568 |
| R | 0.157 / 0.180 | 64 % / 100 % | 0.023 / 0.023 | 1.22 / 1.25 | 0 | 0.473 | 0.462 |
| C_band | 0.197 / 0.211 | 95 % / 100 % | 0.049 / 0.052 | 0.92 / 0.97 | 0 | 0.561 | 0.579 |
| C_rsi | 0.203 / 0.211 | 69 % / 66 % | 0.034 / 0.037 | 1.03 / 0.97 | 35 % / 34 % | 0.498 | 0.536 |
| C_mo | 0.116 / 0.110 | 17 % / 13 % | 0.056 / 0.057 | 0.97 / 1.03 | 0 | 0.654 | 0.773 |
| C_mo_s2 | 0.005 / 0.031 | 2 % / 1 % | 0.086 / 0.087 | 0.86 / 0.92 | 0 | 0.700 | 0.870 |
| C | 0.030 / 0.051 | 0 % / 1 % | 0.085 / 0.083 | 0.83 / 0.89 | 0 | 0.671 | 0.798 |
| C_s2 | 0.049 / 0.056 | 0 % / 0 % | 0.082 / 0.090 | 0.86 / 0.83 | 0 | 0.702 | 0.746 |

**Definitions.**
- **Lean** is the absolute body-over-feet-centroid offset along travel, median over [4, 10) s (`lean_abs`).
- **Sweep** is the median loaded-stance sweep along travel.
- **Cadence** is stance onsets per foot per second (F > 50 N) over [4, 10) s.

**Correction to the first draft.** The draft used lean_lvl, which subtracts the offset at hold start.
- On W, where the statistic was calibrated, the hold-start offset is about 0, so the subtraction was harmless.
- On these from-scratch 1b arms the body already sits about ±0.15 m off the feet at hold start, after the stand
  settle: median +0.155 on C_mo and C_rsi, sign-random on O.
- That made O look "bimodal" and C_mo look "lean-free". Both readings were artifacts.
- The CHARTER's 10:52 note "the lean is gone" carries the same artifact.

## Reading (plan interpretation table)

**Row: O SLOW, C FAST, R SLOW → the factor is in C's bundle.** RC is not run, since it needs C and R both SLOW. C
replicates at seed 2.

**The lean-and-shuffle picture stands.**
- By absolute offset, every SLOW arm leans: 0.16–0.21 m, in 64–100 % of episodes.
- No FAST run does: ≤ 0.056 m, ≤ 2 % of episodes.
- C_mo s3 sits between (0.11 m, 13–17 %), matching its INTERMEDIATE speed.
- Within O, the more-leaning half of the episodes is slower: 0.077 vs 0.092 at 0.22, and 0.086 vs 0.098 at 0.25.
- **The stroke carries the speed at matched cadence.** O and C step at the same cadence (about 0.85 Hz), so C's
  2.3× speed is stroke length (sweep 0.04 → 0.085).
- **But a short sweep does not by itself make an arm slow.** R keeps about 90 % of O's speed at half O's sweep by
  stepping at 1.2 Hz. C_mo s3 reaches 0.16 with a 0.056 sweep at about 1.0 Hz.

**The one-level split of C** (sufficiency: each arm is O plus one piece of C; one seed each unless stated):
- **One-direction pair (C_mo):** the only piece that moves the gait on its own.
  - **Seed 3:** 0.162/0.173. The lean is halved (0.11 m), not gone. It steps faster (about 1.0 Hz) with a mid-length
    sweep (0.056): a different gait from C's.
  - **Seed 2:** 0.203/0.210, within 0.01 of C_s2's 0.194/0.209, and with C's gait: sweep 0.086, cadence about 0.9 Hz,
    no lean.
  - **Telemetry sits with C at both seeds:** early read 0.654 and 0.700; final track_ratio 0.773 and 0.870.
  - **Pre-registered class:** the seeds disagree. The tie-break was not run, so the label is undetermined. The pair is
    FAST-or-near at both seeds and SLOW at neither.
- **Wider band (C_band):** SLOW and leaning, like O. Its cadence and tracking-economy changes alone do not break the
  attractor.
- **RSI off (C_rsi):** SLOW and leaning. About a third of its episodes also stall at onset (they step in place). Alone,
  removing the fast-gait seeding hurts.
- **R's four knobs:** SLOW and leaning, with the shortest sweep (0.023) and the highest cadence (1.2 Hz). They do not
  break the attractor (one seed).

**The C_mo → C gap is unresolved at n = 2.**
- At seed 3 it is 0.05/0.04. The bootstrap CI excludes 0 within that seed, and the gait differs (sweep, cadence, lean).
- At seed 2 it is absent.
- Two seeds cannot tell a true gap of 0 from one of 0.05.
- So whether the band and/or RSI-off complete C's gait on top of the pair is open. The seed-1 tie-break decides the
  label, and C_nb / C_nr would attribute any gap.

**What the pair changes** (code read, 2026-09-29; checked by the skeptic):
- **Mirror images.** The 4-image mirror's images are `[x, M_y x, M_x x, M_x M_y x]`. Two of the three mirrored rows
  carry −y, although 1b only samples +y, so O was never a one-direction run.
- **The mirror-loss target is detached.** Gradient flows only through the policy on the *mirrored* states. In a
  +y-only run, M_y therefore shapes only the −y outputs, and reaches the +y gait through shared weights (interference).
  On walking rows, "M_y equivariance" and "phantom −y training" are the same thing. The only separate piece is on the
  stand rows (20 %): left-right symmetric standing.
- **'fa' also triples the M_x image's effective weight** (coefficient 0.5 over B rows instead of 3B). This is not
  separated by any arm.
- **The scalar obs** zeroes the body-frame command direction (slots 8–9). In 1b the direction is always +y at an
  angle of π/2 + facing error, which slots 6–7 carry within the ±0.6-rad clip. So within 1b training it removes almost
  no information.
- **Consequence:** C_fa (the fa mirror with direction visible) is predicted to match C_mo. It is a low-information
  confirmation in 1b, and it cannot test portability.

## Caveats

- **Sufficiency, not necessity.** No leave-one-out arm ran.
- **Single seeds.** O, R, C_band and C_rsi are one seed each. Their SLOW calls sit far below the FAST line and agree
  with their early reads.
- **C_band is not a clean isolation.** RSI stays on there, and the bank-staged starts are capped at 0.35 instead of
  0.25.
- **Limited eval coverage.**
  - The gate tests only lat_l at 0.22 and 0.25, the bottom 29 % of C's training band.
  - −y is never tested.
  - A C or C_mo head cannot walk right, since its obs cannot see the direction.
- **Scope.** Side-only 1b from scratch. The mechanism found (phantom −y rows) does not exist in full omni, where −y is
  really sampled.
- **The early-read trigger (0.58) was uncalibrated.** At this n it separated the arms perfectly: all four pair runs
  read 0.654–0.702, all four others 0.473–0.564.
- **Budget.**
  - About 10 GPU-h (8 arms × about 1.2 h, plus 0.4 h of checks), over the plan's 8.5 cap. The autonomy rule says
    report, not stop.
  - About 5.4 h of GPU sat idle overnight (03:03–08:28, a chaining bug; see the CHARTER log and memory
    `gpu-queue-chaining`).

## Not run (designed; dry run clean in `scripts/candidates_v4_dryrun.txt`)

| Arm | Overlay on 1b | Question | Value now |
|---|---|---|---|
| C_mo seed 1 | pair (reference not yet added) | Pre-registered tie-break: does the pair alone give C's full gait? | **high**; the recorded deviation |
| C_fa | mirror fa, direction visible | Mirror vs hidden direction | low in 1b; predicted ≈ C_mo |
| C_nb | pair + RSI off | Does RSI-off complete C's gait with the pair? | only if the tie-break reads INTERMEDIATE |
| C_nr | pair + band 0.35 | Does the band complete C's gait with the pair? | only if the tie-break reads INTERMEDIATE |

## For the user (nothing below has run)

1. **Run the C_mo seed-1 tie-break** (≈ 1.2 GPU-h). It is the pre-registered default, and it decides whether the
   one-direction pair alone gives the sideways gait.
2. **The portability question: a two-direction side test from scratch** (≈ 2.4 GPU-h). Sample both sides with direction
   visible, e.g. `KRABBY_OMNI_CMD_POOL=stand:0.2,side_l:0.4,side_r:0.4`, and compare the 4-image mirror against the fa
   mirror under the same gate.
   - With −y really sampled, the phantom-row mechanism is gone.
   - If the fa arm is still FAST and the 4-image arm SLOW, the left-right mirror tie itself is the problem.
   - If both are SLOW, the one-direction benefit does not port, and full omni needs a different lever.
3. **Only then, full omni with the fa mirror.**
   - In full omni, M_y also carries the forward gait's tripod-set handedness constraint. The mirror module notes that
     unconstrained training "breaks at random" without it.
   - So that test must gate forward and turning (tripod balance, the fwd rows), not only side speed.
4. **Housekeeping.** Commits are local on `feat/crab-hex-omni-task`; nothing is pushed. `aa2fed42` moved `code_sha`
   and `eval_sha`, so other omni campaigns' next relaunch needs `--accept-code-change` / `--accept-eval-change`.

---

# Addendum: next steps (user, 2026-09-29 evening: "run the next steps")

The rules were pre-registered in [CHARTER.md](CHARTER.md) "Next steps" before any result. Per-arm outputs:
`parkour/outputs/omni_plan/side_bisect/arms/{C_mo_s1,LRfa,LRfa_lr,LR4,LR4_lr}.json`. The both-hands eval row is
[scenarios_side_lr.yaml](scenarios_side_lr.yaml), run with each arm's own driver-eval env.

## Results

| Arm | Setup (1b from scratch, 3000 its) | Seed | Left 0.22 / 0.25 | Right 0.22 / 0.25 | Class (L / R) |
|---|---|---|---|---|---|
| C_mo_s1 | fa mirror + scalar cmd obs, +y only | 1 | 0.184 / 0.185 | — | **FAST** |
| LRfa | pool side_l + side_r, direction visible, fa mirror | 3 | 0.061 / 0.065 | 0.087 / 0.088 | **SLOW / SLOW** |
| LR4 | pool side_l + side_r, direction visible, 4-image mirror | 3 | 0.085 / 0.091 | 0.082 / 0.096 | **SLOW / SLOW** |

The driver's own lat_l eval agrees: LRfa 0.061/0.064, LR4 0.085/0.092. The n are 90–100 per hold.

| Arm | Lean (m), L / R | Episodes > 0.15 m | Sweep (m) | Cadence (Hz) | Early read | Final track_ratio |
|---|---|---|---|---|---|---|
| C_mo_s1 | 0.065 / 0.074 (lat_l at 0.22 / 0.25) | 0–1 % | 0.069–0.076 | 0.81–0.89 | 0.765 | 0.853 |
| LRfa | 0.25–0.26 / 0.16–0.18 | 100 % / 70–85 % | 0.033–0.038 | 0.93–1.11 | 0.536 | 0.523 |
| LR4 | 0.12–0.13 / 0.11–0.13 | 7–23 % / 6–12 % | 0.033–0.035 | 1.01–1.11 | 0.571 | 0.598 |

## Reading

**The tie-break is resolved: C_mo's 3-seed majority is FAST.** The seeds read s3 INTERMEDIATE by 0.002, s2 FAST and
s1 FAST. In one-direction 1b, the fore-aft mirror + scalar cmd obs pair is enough on its own to form the fast
sideways gait at 2 of 3 seeds (the third is INTERMEDIATE by 0.002). Per the rule, C_nb and C_nr were skipped.

**Two-direction test: LRfa SLOW, LR4 SLOW (both hands).** This is the pre-registered row "two-direction training forms
the attractor whatever the mirror; the one-direction benefit does not port → stop". Step 3 (full omni with the fa
mirror) is therefore not warranted, and was not run.
- **The pair works in 1b because it makes the run truly one-directional.** It removes the phantom −y rows under a +y-only
  sampler.
- **Once −y is really sampled, both mirror settings are slow.** Dropping the left-right mirror tie (LRfa) did not help;
  if anything LRfa's left hand is worse (0.06, heavy lean).
- **So the fast gait in this setup needs the policy to be asked to walk only one way.**

**Amendment: the short stroke is the invariant, not the lean.** LR4 is slow with only a moderate lean (0.11–0.13 m).
Like R, it runs a short stroke (0.033 m) at a high cadence (1.0–1.1 Hz). Across all 11 runs, the SLOW ones share a
stance sweep ≤ 0.053 m (max 0.052, C_band at 0.25). The one INTERMEDIATE run is at 0.056, and every FAST run is at
≥ 0.069. The heavy lean appears in most of them, but not all. This corrects the main gate's "every SLOW
arm leans" line.

**LRfa's left/right asymmetry.** The left hand is slower and leans more (0.06, 0.25 m) than the right (0.09, 0.17 m).
The left is also the hand the +y-only RSI bank seeds. The cause is untested.

## Hypothesis and next discriminator (drafted, NOT run)

The investigation saw the lean re-form at every left↔right switch. A two-direction run with 10-s resamples in 40-s
episodes reverses often.

**Hypothesis:** frequent reversals favour a direction-agnostic short-stroke shuffle over the handed sideways gait,
which needs a directional leading/trailing leg posture.

**Discriminator, LR4_dwell:** LR4 with the command held for the whole 40-s episode (`KRABBY_RESAMPLE_S=40:40`), so
there are no reversals inside an episode. One arm, about 1.2 GPU-h, drafted in `scripts/candidates_v8.yaml`.
- It needs one driver addition first: an `OVERLAY_DUMP_CHECKS` entry for `KRABBY_RESAMPLE_S`. The dry run refuses the
  knob without one (`scripts/candidates_v8_dryrun.txt`).
- **FAST:** reversals are the cause. Levers for full omni would be longer side dwell, fewer or gentler side reversals, or
  a reversal-onset curriculum.
- **SLOW:** training two directions at all is the cause. Levers would be direction-specific policy structure
  (e.g. separate ±y heads or a direction-conditioned mixture), not sampling.

## Cost

About 3.5 GPU-h for C_mo_s1, LRfa and LR4 (measured: 3.34 h of training plus the driver evals), plus 0.12 h of
both-hands evals. The campaign total is about 13.6 GPU-h. Independently recomputed by fable·high: nothing blocking,
and these three wording fixes are applied.
