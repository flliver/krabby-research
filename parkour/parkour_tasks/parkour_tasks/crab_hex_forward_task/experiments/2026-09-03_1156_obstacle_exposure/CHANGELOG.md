# PLAN H — obstacle exposure + corridor widening — CHANGELOG

Approved plan: `/home/nickmagus/.claude/plans/wiggly-gathering-sloth.md` (user-approved 2026-09-03).
Successor to the gait-income phase-out campaign (`sim_fine_tuning/2026-08-31_1414_gait_income_phaseout/`).

## Charter (2026-09-03)

**Scope (user):** (1) raise training-time obstacle exposure to a measured "partway through the
course" target without breaking the gait; (2) widen the obstacle corridors to fit the robot
(recal fix, applied uniformly to control and arms). Out of scope: zero-clip economics, blind
yaw on obstacle tiles, promotion rule, exploration std / LR.

**Exposure target (non-RSI obstacle-tile episodes):** reach_obst ≥ 0.80, goals_passed ≥ 2,
obst_coverage[3] ≥ 0.50, obst_coverage[6] ≥ 0.20, field_frac ≥ 0.20.

**Phases:** B code (B0a corridor recal, B0 telemetry, B0c probe, B1 stand-frac, B2 spawn
offset, B3 spawn spread, B4 RSI spawn fix, orchestrator) → A measure (A0 probe, E4, E1) → C
single-lever arms from the 20k head on widened geometry vs fresh C0 → wave 2 pairings → D fold
into the baked schedule from 5k. Bake decisions are the user's.

## Ledger

### 2026-09-03 — Phase B code landed (CPU only; unarmed = bit-identical; unit-tested)
Files: `parkour_isaaclab/envs/mdp/parkours/exposure_stats.py` (pure ledger + spawn helpers),
`parkour_isaaclab/envs/mdp/parkour_commands/command_sampling.py` (pure slot sampler),
`crab_hexapod_task/mdp/exposure_knobs.py` (pure env-var parsing + corridor bounds);
edits in `parkour_event.py` (B0), `events.py` (B3 + spawn reporting), `crab_hex_rsi.py`
(B4), `uniform_parkour_command.py` + `parkour_command_cfg.py` (B1), `crab_hex_env_cfg.py`
(B0a preset `recal2b2w`, knob wiring), `parkour_terrain_generator.py` + `terrains/utils.py`
(per-tile height fields retained for the spread z lookup). Tests: `tests/unit/
test_exposure_stats.py`, `test_exposure_command_sampling.py`, `test_exposure_knobs.py`,
`test_crab_hex_rsi_spawn_fix.py` (isaac-venv pytest).

**Plan refinements recorded during implementation (no change to arms or gates):**
1. **B2 bound corrected: spawn offset ≥ 1.6, not ≥ 1.0.** Tile-local spawn x = 4.0 − offset
   (16×4 m tiles: 0.5·size_x − (size_y + offset)); offset 1.0 would put the spawn at 3.0 m,
   already past the 2.48 m platform edge. C3 (offset 2.0 → 2.0 m) is unaffected.
2. **B3 z lookup = retained height field, not goal waypoints.** The generator now keeps each
   tile's int16 height field (same pixel frame as the feet-edge mask); spread spawns read z
   there and pass a flatness clearance (max−min ≤ 0.05 m over a ±0.6 m × ±1.2 m window: above
   the ±0.02 m tile noise, below the shallowest gap/hurdle; the y half-window covers the
   1.19 m half-stance so no foot spawns in a side trench). Up to 4 re-draws, then the platform
   spawn (kind recorded as platform, so the exposure split stays honest). Stepping-stone tiles
   will mostly reject (2.8 m stones vs a 2.4 m window) — expected, reported via
   `spread_frac_actual`.
3. **Spread spawns re-target the goal index** to the first goal ahead (+0.3 m) so the parkour
   goal machinery never points backwards, and `goals_passed` is credited from that index;
   their terrain-promotion distance is measured from the actual spawn (otherwise every spread
   episode promotes for free). Platform/RSI spawns keep the historical formula bit-for-bit.
4. **Goal-k semantics pinned:** gap/hurdle/step place goal k at the midpoint *before*
   obstacle k, stones at the stone centre; goal 0 = platform-edge marker (test pins index 1 =
   first obstacle). `reach_obst` therefore reads "committed to the first obstacle's approach".
5. **`Metrics/base_velocity/stand_frac_actual` is always logged** (standing-TIME fraction of
   the episode, per-step indicator / max_episode_length) so C0 reports the control's own
   number; it is telemetry only. Expect ≈0.57 unarmed on the 0.0:0.35 band, ≈p armed.
6. **B0 telemetry keys** (`Metrics/base_parkour/*`, ring of the last 1024 finished episodes,
   NaN-empty groups emitted as 0.0): reach_edge_frac, reach_obst_frac, field_frac_mean,
   field_steps_mean, goals_passed_mean, obst_coverage_1..6 (+ `_rsi` twins),
   crab_failure_flat/obst/obst_rsi, crab_failure_hazard_flat/obst (per 1000 steps),
   ep_steps_flat/obst, spread_frac_actual, rsi_frac_actual.
7. **Not mock-tested:** `reset_root_state`'s spread branch (events.py imports omni/isaaclab;
   its math lives in the tested pure helpers). Covered by the armed smoke run instead.
8. **Orchestrator `run_exposure.py` + `launch_exposure.sh`** (fork of run_phaseout.py). Phase D
   replays with the banks the baked lineage actually trained on per window (P0-null, pg_r1..pg_r4)
   rather than PLAN G's P0-null-throughout confirmation convention — the point of D is to
   reproduce the schedule of record with the lever armed. C arms use the window-4 bank (pg_r3);
   A0 probes each head with its own window's bank (20k: pg_r2, 30k: pg_r4). Creep-speed gate
   reads `tracking_by_hold.creep.achieved_vx` (median) from the canary aggregate.
9. **B0a preset also narrows the corridor's lateral offset** (found by the armed Isaac check):
   the stock `y_range` shifts every gap/hurdle/step segment by up to ±0.4 m and the crab walks
   the centre line blind to it, so half-width alone cannot fit the 1.19 m half-stance
   (needs half ≥ 1.19 + |offset|; at ±0.4 the outer foot still trenches in ~20% of segments
   under (1.40, 1.70)). `recal2b2w` sets y_range (−0.2, 0.2) for gap/hurdle/step and
   '0.05, 0.2' (= 0.08 m at the 0.08 m grid) for stones; env-var overrides never touch offsets.
   Also fixed: the stepping-stone y-slice assumed an even pixel width (2.8 m = 35 px crashed
   terrain generation); the slice is now sized by the width — even widths are unchanged.
10. **Integration checks (16 envs × 300 steps, 20k head):** unarmed probe ran end to end
   (B0 telemetry keys, height fields, ledger, probe summary). First observation: under a
   zero command the robot creeps (eval stand-hold achieved_vx ≈ 0.08 m/s), so the probe now
   also reports a "fast" motion profile at > 0.12 m/s. Armed check re-run after fix 9.
11. **Platform edge from the config, not goal 0** (found in the unarmed check's timeline): the
   step terrain puts its marker 0 at `platform_len − 1 m` = 1.5 m tile-local, half a metre after
   the spawn, so an edge read from goal 0 would credit `reach_edge` on the platform. The
   telemetry now uses `platform_edge_rel(platform_len, size_x, h)` (2.5 m tile-local + 1 border
   px) for every tile, and `goals_passed` counts obstacle goals only (goal index − 1, marker 0
   is never an obstacle), credited relative to the spawn's count. Unarmed check also confirmed
   the RSI bug in telemetry (RSI spawns at origin-relative x = 0.0, platform spawns at −7.0)
   and a zero-command forward creep of ≈0.05 m/s (57% of zero-command steps > 0.05 m/s,
   19% > 0.12 m/s) — the 20 s episode drifts ~1 m even when never commanded.
12. **Armed check (recal2b2w + C1 + C4 + C5) ran cleanly**: standing fraction of steps 0.36
   (C1), RSI spawns at tile-local 1.0 m (C5 fix), spread spawns re-target the goal index
   (5.98 m → goal 3, 4.31 m → goal 2). Defect found: with a ±0.6 m x-window the flatness check
   rejected nearly every in-field draw on obstacle tiles (flat stretches between obstacles are
   0.5–1.35 m), so accepted spread spawns on obstacle tiles all sat on the platform. Window
   narrowed to ±0.3 m (y stays ±1.2 m), tries 4→8; the ledger now also reports
   `crab_failure_obst_spread` / `ep_steps_obst_spread` (z-lookup sanity: spread spawns must not
   fall on arrival). Verified again in the smoke phase before wave 1.

### 2026-09-03 13:26 — Phase A launched (unit obstacle-exposure-1788456336); 14:35 relaunched (unit obstacle-exposure-1788460063)
First A0 probe crashed at 13:29 in its deterministic pass: the env reset between policy modes
ran outside `torch.inference_mode()` and the parkour term's per-step-recreated metric tensors
(inference tensors after the first rollout) rejected the in-place reset write. Probe fixed
(whole mode incl. reset under inference_mode; training/eval never reset outside their rollout
context so they are unaffected); orchestrator now kills a probe whose log shows a Traceback
instead of waiting out the 90-min teardown. Stochastic 20k numbers from the crashed run
(recorded in `a001_probe_20k_probe_CRASH1.log`): reach_edge 0.42, reach_obst 0.10,
field_frac 0.055, goals_passed 0.09, coverage[2+] 0; motion profile flat across bins
(0.68/0.68/0.64/0.62, late-motion ratio 0.92 → NOT systematic late motion); cmd-on 0.385,
achieved vx 0.143 m/s; obstacle-tile failure 0.49 vs flat 0.17. Phase A restarted from scratch.
> NOTIFY: Phase A complete — A0 late-motion ratio 0.955, E4 0.400, E1 B10/B20/B30 0.800/0.600/0.340; T*=70 s; PAUSED before smoke + wave 1 (--start-smoke)

### 2026-09-03 14:58 — Phase A complete; PAUSED at await_user_a (decision: --start-smoke)
**Read of the evidence (REPORT blocks a0/e4/e1):**
1. *Exposure hypothesis CONFIRMED.* Training-time (stochastic) non-RSI obstacle-tile episodes:
   20k head reach_edge 0.31 / reach_obst 0.06 / field_frac 0.04 / goals_passed 0.05 /
   coverage[2..6] = 0; 30k head 0.50 / 0.06 / 0.03 / 0.04 / coverage[2] 0.006. The only
   in-field training data is the RSI 17–26% (bug spawn 7 m downrange: reach_edge 1.0,
   coverage[6] 0.37–0.47, failure 0.29–0.68).
2. *Motion is NOT systematically late* (late-motion ratios 0.96–1.01 on both heads, both
   policy modes); the hardware-session pattern is not a scheduler defect. Command-on 0.37–0.41
   of steps; achieved speed on commanded slots 0.145–0.17 m/s; zero-command creep puts
   |vx| > 0.05 on ~55% of uncommanded steps (> 0.12 on ~20%).
3. *E4 trench-only (30k):* 0.40 (60 falls) vs 0.27 hard / 0.26 light — the trench reproduces
   most of the collapse, not all of it.
4. *E1 widened baselines (recal2b2w 0.20–0.70):* B10 0.80 (narrow 0.73), B20 0.60 (0.60),
   B30 0.34 (0.27). **Predictions (30k ≥ 0.55, 10k ≥ 0.85) NOT met** — lifts of +0.07 sit
   inside the ±0.10 noise band. Per the plan: the widening is recorded as UNSUPPORTED by the
   eval as a stand-alone fix and stays in force (user decision); it strengthens the case that
   the late-window collapse is an exposure (never-trained-in-the-field) problem rather than a
   geometry problem.
5. C2 horizon derived: T* = 70 s (7.0 m / (0.8 × 0.145 m/s) = 61 s → 70 s).
Next on approval: `launch_exposure.sh --start-smoke` → B smoke (2 × 200 iters) → wave 1
(C0, C3, C1, C4, C2 @ 70 s, C5; ≈ 6 × 3.7 h) → await_user_c.
> NOTIFY: (user-facing) Phase A read recorded; awaiting the smoke + wave-1 go-ahead

### 2026-09-03 — USER DECISION: start smoke + wave 1 ("go ahead and start the smoke and wave 1")
Relaunched with --start-smoke: B smoke (unarmed, armed C1+C4+C5; 200 iters each from the 20k
head on recal2b2w) → wave 1 arms C0, C3, C1, C4, C2 (T*=70 s), C5 → await_user_c.
Launched 15:25 as unit obstacle-exposure-1788463517 (a first launch attempt at 15:10 stalled in the launcher's GPU-clear wait and was abandoned). Smoke unarmed run started (s007).
- 2026-09-03 18:16 arm C0 seed 3: CONTROL (reach_obst 0.054, goals_passed 0.055, cov3 0.000, cov6 0.000, obst completion 0.420)
- 2026-09-03 20:42 arm C3 seed 3: FAIL (reach_obst 0.535, goals_passed 0.416, cov3 0.000, cov6 0.000, obst completion 0.520)

### 2026-09-03 21:20 — C3 miss anatomy (user question: falls or time limit?)
Probe roll of the C3 head (`2026-09-03_18-16-56/model_24995.pt`, C3 stack, stochastic policy;
`probe_c3_termination/`, `analyze_termination.py`): 117 platform-spawned obstacle-tile
episodes — reached obstacle 1: 56%; missed and TIMED OUT: 28% (63% of misses; 97% of them
past the edge, median furthest point 0.88 m from spawn vs 1.16 m to the obstacle-1 goal);
missed and FELL: 16% (all in the field just past the edge, median 3.3 s / 0.88 m; none on the
platform). Flat-tile fall rate in the same roll 38% vs obstacle-tile 40% → falls are the gait's
base rate under this mix, not obstacle-specific. Of the reachers, 34% fell later (median 14.5 s,
0.64 m past goal 1). Read: C3's misses are a walking-time problem → C1/C2 are the relevant levers.
- 2026-09-03 23:07 arm C1 seed 3: FAIL (reach_obst 0.520, goals_passed 0.348, cov3 0.000, cov6 0.000, obst completion 0.510)

### 2026-09-03 23:15 — C1 (KRABBY_STAND_FRAC=0.2) FAIL on the exposure floor; safety held, canary IMPROVED
Standing time 0.44 → 0.185; reach_edge 0.76, reach_obst 0.52 (C0 0.054), field_frac 0.21 (passes),
goals_passed 0.35, cov[2] 0.038, cov[3+] 0. Canary tripod 0.573 / completion 0.98 / tracking 0.521
/ creep 0.144 — all better than C0 (0.536 / 0.96 / 0.43 / 0.108). Flat failure 0.30 (C0 0.43),
obstacle 0.61 (C0 0.59), terrain level 5.8 (highest). Obstacle eval 0.51 (C0 0.42).
Miss anatomy (`detail_C1.md`): 49% reach obstacle 1; of the misses 82% FELL (C3: 37%), 72% of
those with the body still before the edge, median 1.24 m from spawn (edge 1.58) — i.e. within
a foot's reach of the platform edge. With C3 the fallers went down 0.3 m PAST the edge. Both
arms' falls cluster within ±0.35 m of the platform→field transition: the onset itself is the
hazard, and falls per walking-time are the gait's base rate (flat-tile 39%).
**Wave-2 qualification concern:** by the plan's letter a single lever is PARTIAL only if it
meets the 0.80 first-obstacle floor; C3 (0.535) and C1 (0.52) both FAIL, so the pre-registered
pairings (C1+C4, C1+C2) would not run at await_user_c. Flagged to the user for decision.

### 2026-09-03 23:30 — USER DECISION: wave-2 pairing eligibility amended
"Admit a lever to the pairings when it held every safety gate and raised reach_obst at least
three-fold over C0." Implemented as `wave2_eligible()` in run_exposure.py (PASS/PARTIAL still
qualify; the 0.80 floor remains the PASS/PARTIAL criterion). Takes effect at --start-wave2 (new
process). Under the rule so far: C1 ELIGIBLE (safety held, reach_obst 0.520 ≥ 3×0.054);
C3 not eligible (creep watchdog tripped). Eligibility is written to REPORT at wave-2 start.
- 2026-09-04 01:34 arm C4 seed 3: FAIL (reach_obst 0.099, goals_passed 0.233, cov3 0.115, cov6 0.015, obst completion 0.330)

### 2026-09-04 01:45 — C4 (KRABBY_SPAWN_SPREAD=1.0:11.0:0.5) FAIL — coverage lever works as a mechanism, hurts the eval
Spread bit at 0.497 of non-RSI resets, but 56% of spread spawns landed on the platform (x < 2.5;
quartiles 1.27/1.51/2.31/5.21/7.99 m): the clearance check still rejects most in-field draws.
Only arm with coverage past obstacle 2: cov[2..6] 0.16/0.115/0.076/0.043/0.015 (gates 0.50 at
[3], 0.20 at [6]); field_frac 0.32 passes; platform-spawned reach_obst 0.099 (< 3× C0 → NOT
eligible under the amended rule). Safety: canary completion 0.90 vs floor 0.91 (VIOLATED by
0.01; tripod 0.569 / tracking 0.449 / creep 0.118 all better than C0). Spread-spawned episodes
fail 0.68 (log) / 0.75 (probe) after a median 1.26 m and 9 s — not on arrival (5% < 2 s), so the
z lookup is sound; they fall walking in level-5 fields they never trained on. Obstacle eval 0.33
(C0 0.42, worst of wave 1), mean reward 15.4 (lowest). Miss anatomy (platform spawns): 9% reach
obstacle 1; 63% of misses fell, 90% of those before the edge (1.20 m from spawn, edge 1.58) —
the edge-transition zone again; probe flat-tile fall rate 0.49 (probe terrain levels are not the
training equilibrium — caveat). Read: spread delivers far-obstacle coverage but at 50% it floods
training with in-field falls and degrades platform-start competence; the pre-registered C1+C4
pairing is now blocked by the amended eligibility rule (C4 fails on reach_obst, the wrong metric
for its mechanism) — decision for the user at await_user_c (options: run C1+C4 as pre-registered;
run it with spread restricted to the field at a lower fraction, e.g. 2.5:11.0:0.25; or drop it).

### 2026-09-04 02:46 — C2 spuriously ABORTED by the live backstop; fixed 02:55, relaunched (unit obstacle-exposure-1788505141)
The backstop compared the raw flat-tile failure SHARE (0.62 > C0 0.43 + 0.10) for C2's 70-s
episodes; the plan's rule for horizon-changing arms is hazard-normalised. C2's flat hazard at the
abort was 0.338/1k steps vs C0 0.590 (backstop allowance 0.727, final gate 0.658) — inside.
Fix: `live_backstop(..., hazard_norm)` and `_obst_ceiling()` run on the per-1k-step hazard for
arms with KRABBY_EPISODE_S (allowances mapped by the control's share→hazard ratio: flat
0.727/1k, obstacle ceiling 1.275/1k). C2 record kept as `arms.C2_aborted_spurious`
(`c013_C2_train_ABORTED_spurious.log`); C2 rerun in order, then C5 (its 18-print start was
discarded). Observed in the aborted C2 tail (2541 prints): reach_obst 0.36, cov[2] 0.14, cov[3]
0.06, field_frac 0.16, terrain level 1.65 (population demoted by the T-scaled promotion threshold,
as predicted), mean reward 28 (longer episodes). Note: `stand_frac_actual` is the per-step
indicator / max_episode_length, so it under-reads by eplen/max (C0 0.44 ≈ 0.57 × 0.70; C2 0.22 ≈
0.57 × 0.36); the comparison table now carries a corrected column.
- 2026-09-04 05:18 arm C2 seed 3: FAIL (reach_obst 0.503, goals_passed 0.391, cov3 0.056, cov6 0.001, obst completion 0.360)

### 2026-09-04 05:26 — C2 (KRABBY_EPISODE_S=70) FAIL — horizon buys obstacle 1–2 on easy tiles, breaks the flat canary
reach_edge 0.63, reach_obst 0.50, field_frac 0.215 (passes), goals_passed 0.39, cov[2] 0.17 (best of
wave 1), cov[3] 0.056, cov[4+] ≈ 0. Per-step hazards are the lowest of the wave (flat 0.295/1k,
obstacle 0.68/1k vs C0 0.59/0.94) but on EASY tiles: terrain level 1.63 (promotion threshold
∝ cmd·T demotes the population, as predicted). Episode length 1534 steps (31 s of 70): 75% of
episodes end by falling. Safety VIOLATED: flat canary completion 0.79 (floor 0.91; 21 falls),
tripod 0.524, tracking 0.436, creep 0.112. Obstacle eval 0.36 (C0 0.42). Miss anatomy
(`detail_C2.md`, 70-s probe): 50% reach obstacle 1; ALL misses fell (median 10.4 s, 1.67 m from
spawn — edge 1.58: the transition zone again); 63% of reachers fell later (median 32 s, 0.74 m past
goal 1); flat-tile fall rate 58% per 70-s episode. Not wave-2 eligible (safety). Under the amended
rule only C1 is eligible so far → wave 2 as coded would run only C1+C5 (if C5 passes); decision
framing for the user at await_user_c.
- 2026-09-04 07:39 arm C5 seed 3: FAIL (reach_obst 0.121, goals_passed 0.097, cov3 0.000, cov6 0.000, obst completion 0.270)
> NOTIFY: wave 1 complete — C0: CONTROL, C3: FAIL, C1: FAIL, C4: FAIL, C2: FAIL, C5: FAIL; PAUSED before wave 2 (--start-wave2)

### 2026-09-04 07:39 — WAVE 1 COMPLETE; PAUSED at await_user_c
Verdicts: C0 control; C3 FAIL (creep watchdog 0.387 < 0.40; reach_obst 0.535); C1 FAIL on the
0.80 floor only (reach_obst 0.520, every safety gate held, best canary of the wave); C4 FAIL
(completion 0.90 vs floor 0.91; reach_obst 0.099; only arm with cov[3..6] > 0); C2 FAIL
(completion 0.79; terrain equilibrium collapsed to level 1.6); C5 FAIL (completion 0.89; RSI-episode
failure 0.676 vs allowance 0.664; obstacle eval 0.27 — removing the buggy centre-spawn removed the
lineage's only in-field exposure). No single lever reached the mid-course target; the target needs
walking time (C1) AND far-course spawns (spread) by the timing table. Wave-2 eligibility under the
amended rule: C1 only → as coded wave 2 would run nothing (C5 failed). Options put to the user:
(A) C1+C4′ with spread restricted to the field at a lower fraction (2.5:11.0:0.25) as the
coverage pairing; (B) additionally C1+C2′ with the promotion fractions scaled by 20/70 to hold the
curriculum equilibrium (touches the promotion rule — out of scope unless the user opts in);
(C) stop here: "exposure levers raise exposure but none reaches the target alone; combinations
untested".

### 2026-09-04 08:05 — USER DECISION: wave 2 = option A ("run option a")
Wave 2 runs the user-registered pairing **C1+C4f** = KRABBY_STAND_FRAC=0.2 + KRABBY_SPAWN_SPREAD=
2.5:11.0:0.25 (spread restricted to the field at a quarter of non-RSI resets; no C5, which failed
its hygiene gates). The computed pairings (eligibility rule) would have run nothing. Winner logic
unchanged: PASS → seed-2 confirmation → await_user_d. Implemented via `wave2_plan()` reading
`state.wave2.user_plan`; relaunched with --start-wave2.
Launched 08:19 as unit obstacle-exposure-1788524336 (a first attempt at 08:06 stalled in the launcher's GPU-clear wait while the C5 detail probe was tearing down). w016_C1_C4f training.
- 2026-09-04 10:40 arm C1+C4f seed 3: FAIL (reach_obst 0.475, goals_passed 0.417, cov3 0.077, cov6 0.014, obst completion 0.460)
> NOTIFY: wave 2 complete — no combination reached PASS; best trade-off C1+C4f (FAIL); PAUSED

### 2026-09-04 10:40 — WAVE 2 COMPLETE: C1+C4f FAIL on exposure only; safety held; PAUSED at await_user_d
C1+C4f (STAND_FRAC 0.2 + SPAWN_SPREAD 2.5:11.0:0.25): reach_edge 0.72, reach_obst 0.475, field_frac
0.30 (passes), goals_passed 0.42, cov[2..6] 0.13/0.077/0.060/0.041/0.014 (gates cov[3] 0.50, cov[6]
0.20). spread_frac_actual 0.194 (of 0.25; field-only clearance rejects the rest); spread episodes
fail 0.80 at terrain level 5.9. Canary tripod 0.576 / completion 0.92 / tracking 0.515 / creep
0.135 — every safety gate held, all better than C0. Obstacle eval 0.46 (C0 0.42, B20 0.60).
Miss anatomy (`detail_C1_C4f.md`): 49% reach obstacle 1; 83% of misses fell, half before / half
just past the edge (median 1.60 m from spawn, edge 1.58); 38% of reachers fell later (0.65 m past
goal 1). No PASS winner → Phase D has nothing to replay under the plan. Campaign outcome so far:
"answered but not solved" — every lever raises exposure; the safe best trade-off is C1+C4f (and
C1 alone), neither reaches the mid-course target; the platform→field transition and the
level-5–6 field are where episodes die regardless of how they get there.
> NOTIFY: (user-facing) wave 2 read recorded; awaiting the closing / Phase D decision

### 2026-09-04 11:00 — USER DECISION: CAMPAIGN CLOSED ("let's close this campaign")
Outcome: ANSWERED BUT NOT SOLVED (REPORT close block). No Phase D, no re-bake; schedule of record
unchanged. Next step per the user: the splay plan, continued in another session. Monitors stopped;
orchestrator state phase = done.
