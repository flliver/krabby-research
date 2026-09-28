# Morphology × training-side fixes campaign — CHANGELOG

Approved plan: `/home/nickmagus/.claude/plans/wiggly-gathering-sloth.md` (user-approved 2026-09-04).
Folds the paused leg-mount morphology campaign (`sim_fine_tuning/2026-09-02_1446_leg_mount_morphology/`,
rungs i–iv done, rung v partial) together with the training-side levers built and measured by the
closed obstacle-exposure campaign (`sim_fine_tuning/2026-09-03_1156_obstacle_exposure/`).

## Charter (2026-09-04)
**Question:** does a formation config that actually trains walking (KRABBY_STAND_FRAC=0.2; then
40-s episodes with 10-s holds and the curriculum equilibrium held) let the wider plants form
robust gaits from scratch, and do they survive obstacle onsets better than the golden plant in
training (exposure telemetry) and in eval?
**User decisions (2026-09-04):** staged (all 7 variants in P1, top 3 + golden in P2); every arm
from scratch (no fine-tuning of golden-plant heads); 40-s episodes with promotion fractions
scaled by 20/40 and 10-s command holds. Carried over: evaluate every sound variant, only
plant-unsound arms drop, splay-only is a tie-breaker, default plant stays golden, hardware / bake
decisions are the user's.
**Protocols:** P1 = rung-v formation config + STAND_FRAC 0.2, all 8 configs, 0→5k → STOP A.
P2 = top 3 + golden: stage 1 (0→5k) P1 + EPISODE_S 40 + RESAMPLE_S 10:10; stage 2 (5k→10k)
window-1 element stack + recal2b2w + TERRAIN_PROMOTE 0.225:0.125 → STOP B → seed-2 of the pick.
**Evals per stage:** morph manifest `slow__<cfg>`, `step__<cfg>` (rung-iv/v continuity) + PLAN H
obstacle eval (recal2b2w 0.20–0.70) on the variant plant; exposure + per-tile failure columns
from the training log. Gates are scored columns (0.85× golden reference, rung-v noise floor
0.06 tripod / 0.05 completion / 5 falls); drops only for plant-unsoundness.

## Ledger

### 2026-09-04 11:30 — code landed (CPU): orchestrator, launcher, decision-table assembler, tests
`run_morph_exposure.py` (imports the closed campaigns' helpers by path: run_exposure.py train-loop
/ eval runners / exposure parser / hazard-normalised backstop; run_formation_arms.py formation
config, plant map, morph-manifest eval summariser), `launch_morph.sh`, `assemble_final.py`,
`heartbeat_morph.sh`; `tests/unit/test_morph_exposure_stacks.py` (10 tests: P1 == rung-v config +
STAND_FRAC; P2 stage 1 adds only the horizon; stage 2 = window-1 elements + recal2b2w +
promotion 0.225:0.125 with P0-null bank and window-1 gait weights; plant paths resolve; golden has
no override; ranking rule; corrected stand fraction). Plant guard: every eval's run_meta.json
`usd_path` is checked against the requested plant. No env-code changes were needed.
Launched 13:09 as unit morph-x-exposure-1788541782 (smoke → P1 automatically).
> NOTIFY: morphology x exposure: SMOKE FAILED — see REPORT; fix before relaunch
- 13:20 smoke FAIL was a checker defect (resampling range is a multi-line YAML tuple); params confirmed 10 s holds; checker fixed, relaunched (checks re-evaluated on the kept record, no retrain).
- 2026-09-04 15:49 P1 base: ok; slow 0.460 step falls 71 obst 0.390

### 2026-09-04 15:50 — P1 golden `base` landed: the walking-slot formation config is NOT a free improvement on the golden plant
From-scratch 5k with STAND_FRAC 0.2: slow canary tripod 0.444 / completion 0.46 / tracking 0.596 /
54 falls (rung-v control on the old config: 0.595 / 0.88 / 0.397 / 12; rung-v base reproduction
0.537 / 0.83 / 0.426 / 17); step onset 71/100 (rung v: 69 base, 39 control); recal2b2w obstacle
eval 0.39. Training telemetry: reach_edge 0.88 / reach_obst 0.64 / field_frac 0.31 on the light
shallow formation tiles (frozen level ~1), flat failure 0.23, obstacle 0.35, ep len 780, corrected
standing time 0.24. Read: 80 % walking slots from iteration 0 form a faster (tracking +0.2) but far
less stable 5k walker on the golden plant — 96 % of canary falls are nose-down pitch, the plant's
known topple. Far beyond the rung-v noise floor (0.05 completion / 5 falls). The variants' P1 rows
(B training now) decide whether the wider plants convert the extra walking into survival.
- 2026-09-04 18:20 P1 B: ok; slow 0.820 step falls 46 obst 0.540
- 2026-09-04 19:32 P1 A10: aborted; slow None step falls None obst None

### 2026-09-04 19:35 — A10 aborted by a carried-over live backstop (flat failure 0.37 > golden 0.23 + 0.10 at 2.5k); backstop REMOVED, A10 rerun
The PLAN H mid-segment backstop kills an arm whose flat-tile failure share exceeds the control's
by 0.10 — appropriate for fine-tuning windows against a settled control, wrong here: the user's
rule is that only plant-unsound arms drop, formation-phase failure shares at 2.5k are noisy, and
the golden P1 arm (itself unstable: canary 0.46) is not a stability reference for a wider plant.
`run_stage` no longer arms any live check; soundness smoke (NaN / dead plant / traceback) stays.
A10 (and the just-started A15) rerun in order; the aborted A10 log is kept as
`logs/p1_004_A10_train_ABORTED_backstop.log`.
Relaunched 19:37 as unit morph-x-exposure-1788564806 (A10 rerun, then A15 …).
- 2026-09-04 22:02 P1 A10: ok; slow 0.570 step falls 84 obst 0.150
- 2026-09-05 00:31 P1 A15: ok; slow 0.900 step falls 58 obst 0.490
- 2026-09-05 03:00 P1 A20: ok; slow 0.500 step falls 67 obst 0.340
- 2026-09-05 05:27 P1 A10+B: ok; slow 0.440 step falls 79 obst 0.290

### 2026-09-05 05:30 — P1 through A10+B: the failure mode flips to BACKWARD pitch on the widest bases
Slow-canary fall classes (per 100): golden 52 fwd / 2 back; B 11 / 7; A10 40 / 3; A15 10 / 0;
A20 19 / **31**; A10+B 8 / **48**. Step onset: A20 37 / 30, A10+B 45 / 34. The nose-down basin the
campaign targets shrinks with base width, but beyond ~A15/B the 5k formation policies topple
tail-down instead (roll stays ≈0.05 rad everywhere). A15 (0.90, 10 falls, all forward) and B
(0.82, 18) sit in the window where forward falls are reduced without backward ones appearing.
Ranking after six arms: B 0.44, A15 0.38, A20 0.17, golden 0.13, A10+B 0.12, A10 0.09.
- 2026-09-05 07:56 P1 A15+B: ok; slow 1.000 step falls 5 obst 0.910

### 2026-09-05 07:57 — P1 A15+B: best row of the campaign by a wide margin
Slow canary 1.00 completion / 0 falls / tripod 0.564 / tracking 0.753; step onset 5 falls (1 fwd,
4 back), completion 0.95; recal2b2w obstacle eval 0.91 (9 falls) — above every golden-plant head
on that eval (PLAN H B10/B20/B30 = 0.80/0.60/0.34, PLAN H arms ≤ 0.54). Training: flat failure
0.03, obstacle-tile 0.06, reach_obst 0.815, cov[2] 0.35, cov[3] 0.086, mean reward 43. Smoke at 2k
already 0.046 failure. Non-monotonic vs A10+B (0.44 / 56 falls, 48 backward) and A20 (backward
falls): either the 15°+2.5-in geometry is the sweet spot or formation-run variance on new plants
is large — the seed-2 confirmation of the pick decides. Ranking after seven arms: A15+B 0.95,
B 0.44, A15 0.38, A20 0.17, golden 0.13, A10+B 0.12, A10 0.09.
- 2026-09-05 10:23 P1 A20+B: ok; slow 0.980 step falls 46 obst 0.510
> NOTIFY: P1 complete — default top 3 by slow completion x (1 - step fall share): ['A15+B', 'A20+B', 'B']; PAUSED at STOP A (relaunch with --start-p2 [--top A,B,C])

### 2026-09-05 10:23 — P1 COMPLETE (8/8 sound); PAUSED at STOP A
Ranking (slow completion × (1 − step fall share)): A15+B 0.95, A20+B 0.53, B 0.44, A15 0.38,
A20 0.16, golden 0.13, A10+B 0.09, A10 0.09. Every plant passed the soundness smoke (no NaN, no
collision income, episode length rising). Findings: (1) the walking-slot formation config is not a
free improvement on the golden plant (canary 0.46, 54 nose-down falls) but turns the wider plants
around (B 0.31→0.82, A15 0.46→0.90 vs rung v); (2) the failure mode flips to BACKWARD pitch on
A20 and A10+B; (3) A15+B is the standout (canary 1.00 / 0 falls, step 5 falls, recal2b2w
obstacle eval 0.91 — above every golden-plant head), A20+B second (0.98 / 2 falls, obstacles 0.51).
Default top 3 = [A15+B, A20+B, B]; recommendation put to the user: swap A20+B for A15 to keep the
best splay-only (hardware-preferred) candidate in P2, or run four (+8 h).

### 2026-09-05 — USER DECISION: P2 = option 2 ("let's run option 2") → top 3 = A15+B, B, A15 (+ golden control)
One candidate per hardware route (combined, re-hinge-only, splay-only). Relaunched with
--start-p2 --top A15+B,B,A15. Order: golden first (control), then A15+B, B, A15; each 0→5k
(40-s episodes, 10-s holds) then 5k→10k (window-1 elements, recal2b2w, promotion 0.225:0.125).
Launched 12:03 as unit morph-x-exposure-1788624209.
- 2026-09-05 16:51 P2 base seed 3: s1 ok / s2 ok

### 2026-09-05 16:52 — P2 golden control: both stages landed
Stage 1 (0→5k, 40-s episodes, 10-s holds): canary 0.28 / 72 falls (42 fwd, 30 back), step 91,
recal2b2w obstacle eval 0.13; training flat failure 0.81 (hazard 0.81/1k — ×3 the P1 golden
hazard 0.28): longer episodes with sustained holds make golden-plant formation WORSE per step,
not just per episode. Stage 2 (5k→10k, window-1 elements, recal2b2w, promotion 0.225:0.125):
canary 0.68 / 32 falls, step 67, obstacle eval 0.31; flat hazard 0.33/1k, obstacle 0.80/1k;
**terrain level 5.83 — inside the 3–7 band, the promotion scaling held the curriculum
equilibrium** (C2 at 70 s had collapsed to 1.6); coverage cov[3] 0.25 / cov[6] 0.027 — the
40-s horizon at level ~6 gives real mid-course coverage (PLAN H best cov[3] 0.115). Now A15+B.

### 2026-09-05 19:20 — P2 stage 1 A15+B (40-s episodes): sustained holds cost the plant some eval stability but it stays far above golden
Canary 0.86 / 14 falls (all forward), step 31 (0.69), recal2b2w obstacle eval 0.68 (32 falls) —
vs its own P1 (20-s) arm 1.00 / 0, 5, 0.91 and vs golden P2 stage 1 0.28 / 72, 91, 0.13. Training:
flat hazard 0.128/1k (golden P2 s1 0.81/1k — 6× lower), obstacle hazard 0.34/1k; reach_obst 0.86,
cov[2] 0.54, cov[3] 0.33, cov[6] 0.049 (frozen level ~1); mean reward 72. Read: the 40-s / 10-s-hold
formation is harder for every plant, but A15+B holds a sustained walk where golden cannot.
- 2026-09-05 21:37 P2 A15+B seed 3: s1 ok / s2 ok

### 2026-09-05 21:38 — P2 stage 2 A15+B (10k): the obstacle window on the wide plant delivers
Canary 0.95 / 5 falls (all forward), tripod 0.589; step 23 (0.77); recal2b2w obstacle eval 0.73
(27 falls) — vs golden stage 2 0.68 / 32, 67, 0.31. Training at terrain level 6.03 (in band):
flat hazard 0.064/1k (golden 0.33), obstacle hazard 0.32/1k (golden 0.80); exposure reach_obst
0.915, cov[2..6] 0.71 / 0.49 / 0.30 / 0.15 / 0.06, goals_passed 1.0, field_frac 0.60 — the PLAN H
mid-course exposure target is essentially met on this plant at level 6 (cov[3] 0.49 vs 0.50;
goals_passed 1.0 vs 2; cov[6] 0.06 vs 0.20), with obstacle-tile failure 0.48 where golden sits at
0.80. Two of four P2 arms done; B stage 1 running.

### 2026-09-06 00:10 — P2 stage 1 B (40-s episodes): lowest training hazards of the campaign
Canary 0.79 / 21 falls (20 fwd, 1 back), step 32 (0.68), recal2b2w obstacle eval 0.67 (33) — vs its
P1 (20-s) arm 0.82 / 18, 46, 0.54 and vs A15+B stage 1 0.86 / 14, 31, 0.68. Training: flat hazard
0.049/1k (A15+B 0.128, golden 0.81), obstacle hazard 0.166/1k, ep len 1915 of 2000 (episodes mostly
run out the clock), mean reward 91; reach_obst 0.89, cov[2] 0.63, cov[3] 0.37. Read: the re-hinge-only
plant tolerates sustained holds best in training; its evals sit with A15+B's. Stage 2 running.
- 2026-09-06 02:36 P2 B seed 3: s1 ok / s2 ok

### 2026-09-06 02:38 — P2 stage 2 B (10k): best obstacle-window row so far
Canary 0.90 / 10 falls (all forward), tripod 0.455; step 14 (0.86); recal2b2w obstacle eval 0.83
(17 falls) — vs A15+B stage 2 0.95 / 5, 23, 0.73 and golden 0.68 / 32, 67, 0.31. Training at level
5.75 (in band): flat hazard 0.039/1k, obstacle hazard 0.173/1k (both campaign lows; A15+B 0.064 /
0.32, golden 0.33 / 0.80); obstacle-tile failure 0.29; reach_obst 0.93, cov[3] 0.53, cov[6] 0.078,
goals_passed 0.97. Read: at 10k the re-hinge-only plant survives the obstacle window best; A15+B
keeps the cleaner flat gait (tripod 0.59 vs 0.46, canary 0.95 vs 0.90). A15 (splay-only) last.

### 2026-09-06 05:06 — P2 stage 1 A15 (40-s episodes): the splay-only plant picks up the backward-pitch mode under sustained holds
Canary 0.78 / 22 falls (12 fwd, **10 back**), step 55 (15 fwd, **40 back**, 0.45), recal2b2w
obstacle eval 0.48 (52) — vs its P1 (20-s) arm 0.90 / 10 (all fwd), 58, 0.49. Training: flat hazard
0.082/1k, obstacle 0.27/1k; reach_obst 0.85, cov[3] 0.32; mean reward 74. Read: in P1 A15 had no
backward falls; with 10-s holds it leans back at step onsets — the failure class A20 / A10+B showed
in P1. Weakest of the three P2 variants at stage 1 (B 0.79 / 21, A15+B 0.86 / 14). Stage 2 running.
- 2026-09-06 07:24 P2 A15 seed 3: s1 ok / s2 ok
> NOTIFY: P2 complete — PAUSED at STOP B; run assemble_final.py for the decision table; relaunch with --seed2 <cfg> for the confirmation of the pick

### 2026-09-06 07:24 — P2 COMPLETE (golden + A15+B, B, A15; all sound); PAUSED at STOP B; decision table assembled
At 10k (stage 2, terrain level 5.7–6.0, promotion scaling held everywhere): B canary 0.90 / 10, step 14,
recal2b2w obstacle eval 0.83, hazards 0.04 / 0.17 (best onset survival); A15+B 0.95 / 5, 23, 0.73,
hazards 0.06 / 0.32 (best gait: tripod 0.59); A15 1.00 / 0, 26, 0.67, hazards 0.08 / 0.39 (splay-only;
backward falls seen at stage 1 vanished by stage 2); golden 0.68 / 32, 67, 0.31, hazards 0.33 / 0.80.
Longer episodes judged: stage-1 formation with 10-s holds is harder for every plant (golden hazard ×3,
A15+B / A15 lose canary points, B unaffected), fully recovered by stage 2 on the wide plants; the
promotion scaling held the curriculum equilibrium (5.7–6.0 vs the 20-s lineage's ~4.9–5.8), so the
40-s horizon is usable. Exposure target of the closed campaign met in training on all three wide plants
(cov[3] 0.40–0.53, goals_passed ~1.0, field_frac 0.5–0.6). Options put to the user for the seed-2
confirmation: B (recommended: onset survival, re-hinge only), A15+B, A15 (splay-only), or two runs.
`final_decision_table.csv` + REPORT decision-table block written (not yet cross-linked into the
morphology RESULTS.md — done at close with --to-results).

### 2026-09-06 — USER DECISION: seed-2 confirmation on TWO plants ("do two runs. B and A15+B")
Both P2 stages (0→5k 40-s formation, 5k→10k obstacle window) on seed 2 for B, then A15+B (≈ 8 h
each). Orchestrator extended to accept a comma list (--seed2 B,A15+B). Relaunched.
Launched 11:13 as unit morph-x-exposure-1788707596 (seed2s1 B first).

### 2026-09-06 13:44 — seed-2 stage 1 B: formation on the wide plant is seed-sensitive
Seed 2: canary 0.78 / 22 falls (6 fwd, **16 back**), step 63 (30 fwd, **33 back**), recal2b2w obstacle
eval 0.29 (71) — vs seed 3 stage 1: 0.79 / 21 (20 fwd, 1 back), step 32, 0.67. Training telemetry
close on flat (hazard 0.063 vs 0.049/1k) but obstacle hazard doubled (0.35 vs 0.17) and the
backward-pitch mode appeared. The canary completion reproduces; the onset behaviour at 5k does not.
Stage 2 (obstacle window) is the decision-relevant stage — seed-3 arms recovered strongly there.
- 2026-09-06 16:03 SEED2 B seed 2: s1 ok / s2 ok

### 2026-09-06 16:04 — seed-2 stage 2 B: recovers through the obstacle window, lands below seed 3 on onsets
Seed 2 at 10k: canary 0.96 / 4 falls (1 fwd, 3 back), step 28 (17 fwd, 11 back, 0.72), recal2b2w
obstacle eval 0.69 (31); hazards flat 0.068 / obstacle 0.43 per 1k; level 5.84; reach_obst 0.82,
cov[3] 0.35 — vs seed 3: 0.90 / 10, 14 (0.86), 0.83; hazards 0.039 / 0.17. Two-seed envelope for B
at 10k: canary 0.90–0.96, step 14–28 falls, obstacle eval 0.69–0.83 (spread 0.14, beyond the
±0.10 eval band) — both seeds far above golden (0.68 / 67 / 0.31). The stage-1 backward-pitch
formation largely resolved in the obstacle window (11 backward onset falls remain). A15+B seed 2 next.

### 2026-09-06 18:34 — seed-2 stage 1 A15+B: same seed-sensitivity as B — backward-pitch formation on seed 2
Seed 2: canary 0.58 / 42 falls (19 fwd, **23 back**), step 52 (32 fwd, 20 back, 0.48), recal2b2w
obstacle eval 0.33 (67); hazards flat 0.158 / obstacle 0.27 per 1k; exposure high (reach_obst 0.90,
cov[3] 0.56, cov[6] 0.13) — vs seed 3 stage 1: 0.86 / 14 (all fwd), 31, 0.68; hazards 0.128 / 0.34.
Read: on both re-hinge plants the 40-s / 10-s-hold formation lands in the backward-pitch basin on
seed 2 and not on seed 3; the obstacle window repaired B's (0.29 → 0.69). A15+B stage 2 decides.
- 2026-09-06 20:51 SEED2 A15+B seed 2: s1 ok / s2 ok
> NOTIFY: seed-2 replay of ['B', 'A15+B'] complete — run assemble_final.py --to-results; hardware decision is the user's

### 2026-09-06 20:51 — CAMPAIGN COMPLETE: seed-2 confirmations landed; final table cross-linked into the morphology RESULTS.md
A15+B seed 2 stage 2: canary 0.96 / 4, step 8 (0.92), recal2b2w obstacle eval 0.95 (5 falls), hazards
0.06 / 0.16, cov[3] 0.64 — best 10k row of the campaign, from a seed-2 formation that had started
in the backward-pitch basin (0.33 at 5k). Two-seed 10k envelopes: B canary 0.90–0.96 / step 14–28 /
obstacles 0.69–0.83; A15+B 0.95–0.96 / 8–23 / 0.73–0.95; golden 0.68 / 67 / 0.31. Records: REPORT
(CAMPAIGN COMPLETE block), final_decision_table.csv, morphology RESULTS.md (cross-link block).
Hardware decision = user. GPU free; monitors stopped.

### 2026-09-06 — USER DECISION: A15+B ("let's move forward with a15+b")
Plant of choice = 15° outward splay shims + yaw axes re-hinged to 2.5 in from the body ends
(`assets/variants/crab_simple__splay15_axis2p5in.usda`). Two-seed 10k evidence: canary 0.95–0.96 /
4–5 falls, step-onset 8–23 falls, recal2b2w obstacle eval 0.73–0.95, training hazards 0.06 / 0.16–0.32
per 1k steps (golden 0.68 / 67 / 0.31 / 0.33 / 0.80). Rung-iv transfer (30k head, no retrain): 0 step
falls; rung iii open-loop 8/8 survivors, hip/femur contact 50 N at full throw (scored, not unsound).
