> Superseded 2026-09-09: A15+B is now the MAIN asset (assets/crab.usda); the golden geometry is assets/crab_simple.usda = legacy_golden. See docs/crab-hexapod-plant.md.

# A15+B lineage retrain — CHANGELOG

## Charter (2026-09-06)
**User decisions:** plant of choice = **A15+B** (15° outward splay shims + yaw axes re-hinged to 2.5 in;
`assets/variants/crab_simple__splay15_axis2p5in.usda`) from the morphology × exposure campaign
(`sim_fine_tuning/2026-09-04_1105_morph_x_exposure/`, two-seed 10k: canary 0.95–0.96, step-onset falls
8–23, recal2b2w obstacle eval 0.73–0.95 vs golden 0.68 / 67 / 0.31); "move forward" = **full lineage
retrain** (0→30k from scratch on A15+B). Bake decision stays the user's. Golden asset and the default
plant are untouched (A15+B opt-in via KRABBY_HEX_USD_PATH).
**Config:** six 5k windows, seed 3. w0 = formation (rung-v config + STAND_FRAC 0.2 + 40-s episodes /
10-s holds). w1–w5 = the schedule of record (elements @5k and @10k; gait-income ramps apex/airtime/
stride halves @5k → ε @10k, clock 1.0→0.5@15k→0.2@20k→ε@25k) on recal2b2w with the curriculum on
and promotion fractions 0.225:0.125 (×20/40), STAND_FRAC 0.2 and 40-s episodes throughout; RSI bank
P0-null throughout. Evals after every window on the A15+B plant: morph-manifest `slow__A15pB`,
`step__A15pB`, PLAN H obstacle eval (recal2b2w 0.20–0.70); exposure + per-tile hazards from the log.
Reference: the golden 30k and 20k heads of record evaluated on the golden plant in the same three
scenarios. No live kills (soundness smoke in w0 only). Stop-and-wait after seed 3 (push-notify);
`--seed2` runs the replay. ≈ 20 h per seed.

## Ledger
Launched 2026-09-06 21:32 as unit a15b-lineage-1788744762 (window 0 training).
- 2026-09-06 23:59 windows w0 seed 3: ok; canary 0.860 step falls 31 obst 0.680
- 2026-09-07 00:01 window 0 reproduced the morph-x-exposure P2 stage-1 A15+B seed-3 run EXACTLY (same seed/config/plant: canary 0.86/14, step 31, obst 0.68, smoke fail@2k 0.500, ep_len@2k 1484.577) — same-seed training is bit-reproducible on this stack for this config.
- 2026-09-07 02:19 orchestrator crashed writing window 1's report (fmt() arity bug) AFTER window 1's training + evals were saved; fixed, report backfilled on relaunch, resumed at window 2.
- 2026-09-07 02:19 windows w1 seed 3: ok; canary 0.920 step falls 26 obst 0.560
Relaunched 02:22 as unit a15b-lineage-1788761985 (window 2 training).
- 2026-09-07 04:38 windows w2 seed 3: ok; canary 0.850 step falls 37 obst 0.580
- 2026-09-07 06:57 windows w3 seed 3: ok; canary 0.810 step falls 32 obst 0.660
- 2026-09-07 09:16 windows w4 seed 3: ok; canary 0.690 step falls 58 obst 0.290

### 2026-09-07 09:17 — window 4 (20k→25k, clock 0.5→0.2): the late-window collapse reproduces on the wide plant
Canary 0.69 / 31 falls (25 fwd, 6 back), tripod 0.371 (w3 0.500); step 58 (39 fwd, **19 back**, 0.42);
recal2b2w obstacle eval 0.29 (71 falls) — from 0.66 (34) at 20k. Hazards flat 0.25 / obstacle 0.53
(w3 0.20 / 0.42); level 6.21; exposure still high (reach_obst 0.88, cov[3] 0.39); mean reward 33.
Same shape as the golden lineage's decay (obstacle 0.60 @20k → 0.45 @25k → 0.27 @30k; PLAN H C0 with
weights held 0.60 → 0.42) — the clock anneal 0.5→0.2 and/or the level-6 promotion equilibrium in
the 20k–25k window, now on A15+B. Backward-pitch falls returned at onsets. Window 5 (clock → ε)
runs to complete the schedule of record; the 20k head (w3: canary 0.81, obstacles 0.66) is the
strongest checkpoint of this lineage so far. Decision framing for the pause: bake head choice
(20k vs 30k) and/or a schedule variant holding the clock at 0.5 — user's call.
- 2026-09-07 11:35 windows w5 seed 3: ok; canary 0.590 step falls 40 obst 0.480
> NOTIFY: A15+B lineage seed 3 complete at 30k — canary 0.590, obstacle eval 0.480; PAUSED (relaunch with --seed2 for the replay; bake decision is the user's)

### 2026-09-07 11:45 — SEED-3 LINEAGE COMPLETE (30k); PAUSED at await_user; golden references evaluated
w5 (30k, clock → ε): canary 0.59 / 41 falls (40 fwd), step 40 (0.60), recal2b2w obstacle eval 0.48 (52);
hazards flat 0.29 / obstacle 0.45; level 5.85. Partial recovery from w4 (0.29) but well below w3 (20k:
0.81 / 19, step 32, obst 0.66). Golden heads of record on the golden plant, same scenarios: 30k canary
0.81 / 19, step 74 (0.26), obst 0.34; 20k canary 1.00 / 0, step 42 (0.58), obst 0.60. Read: at matched
iteration the wide plant beats golden on onsets and obstacles (20k: 32 vs 42 step falls, 0.66 vs 0.60;
30k: 40 vs 74, 0.48 vs 0.34) but trails on the flat canary (0.81 vs 1.00 at 20k; 0.59 vs 0.81 at 30k),
and the schedule's late windows degrade both. Strongest A15+B checkpoints measured: morph-x-exposure
P2 10k heads (seed 3: canary 0.95 / obst 0.73; seed 2: 0.96 / 0.95, step 8 — no gait-income ramps),
then this lineage's 20k head (w3). Decision framing put to the user: bake-head choice, an
anneal-hold variant 20k→30k from w3, and/or the seed-2 lineage replay.

### 2026-09-07 — USER DECISION: BAKE — this lineage's 20k head is the A15+B policy of record
"make this lineage's 20k head the policy of record and bake" → **A15+B reference of record =
`parkour/logs/rsl_rl/crab_hex_flat_walk/2026-09-07_04-38-50/model_19996.pt`** (window 3 head, seed 3).
Baked schedule for the A15+B plant = windows 0–3 of this lineage: formation (rung-v config +
STAND_FRAC 0.2 + 40-s episodes / 10-s holds, P0-null RSI 0.2) → @5k elements + apex/airtime/stride to
half, recal2b2w, curriculum with promotion 0.225:0.125 → @10k elements + satellites to ε → 15k–20k
clock 1.0→0.5. The 20k–30k clock anneal (0.5→0.2→ε) is NOT baked for this plant (late-window
collapse: obstacles 0.66 → 0.29 → 0.48). Baked-head evals on A15+B: slow canary tripod 0.500 /
completion 0.81 / 19 falls; step onset 32 falls (0.68); recal2b2w obstacle eval 0.66 (34 falls);
training hazards flat 0.20 / obstacle 0.42 per 1k steps at level 5.75. Golden 20k head of record on
the golden plant, same scenarios: 1.00 / 0; 42 (0.58); 0.60. Seed-2 lineage replay and the
anneal-hold variant NOT run (user chose to bake). Next: 60-s videos (flat / light / medium / tough).

### 2026-09-07 12:25 — videos of the baked head (user request)
`videos/a15b_20k_{flat,light,medium,tough}.mp4`: play task `Isaac-Crab-Hex-Flat-Walk-Play-v0`
(60-s episode, follow cam), A15+B plant, checkpoint `2026-09-07_04-38-50/model_19996.pt`, fixed
command 0.30 m/s straight ahead (KRABBY_LIN_VEL_X=0.30:0.30, heading 0), 3000 steps @ 50 Hz, 1 env.
Tiers: flat = 100 % flat tiles; light / medium / tough = all-obstacle tiles on recal2b2w at
difficulty 0.05–0.20 / 0.20–0.45 / 0.45–0.70 (the training range's top). Recipe in
`videos/record_*.log`.

### 2026-09-07 12:30 — USER DECISION: CAMPAIGN CLOSED ("close this campaign and update the memory")
Baked head stands (20k, windows 0–3). Seed-2 replay and anneal-hold variant waived. State phase = closed;
GPU free; no monitors. Records: REPORT (BAKE + CAMPAIGN CLOSE blocks), CHANGELOG, state.json, logs/, videos/.
