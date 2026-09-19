<!-- paths-note -->
> **Paths note (2026-09-09):** this campaign moved from `sim_fine_tuning/2026-08-20_2100_gait_formation/` to `parkour/parkour_tasks/parkour_tasks/crab_hex_forward_task/experiments/2026-08-20_2100_gait_formation/`. Absolute paths below (`/home/.../sim_fine_tuning/...`, `parkour/logs/rsl_rl/...`) name raw artifacts that stay on disk untracked; the tracked checkpoint of record is `head/` (see `bundle.yaml`) and the eval summaries are in place / under `evals/`.

# Gait-formation campaign (PLAN D) — CHANGELOG

Objectives (priority order): 1) stable tripod gait; 2) legs lifting well; 3) smooth,
non-jerky motion; 4) continuous cam spin. Survival (anti-wheelie) is the prerequisite
phase. Protocol: from-scratch cumulative stacks, 1k screens -> 5k winners, env-var-delta
arms, offline replay gate for new/retuned income terms, mechanism diagnosis on every
failure, second-seed reproduction before bakes. Plan: ~/.claude/plans (PLAN D) +
this dir. Plant: hardware-morphology model (see 2026-08-20_1506_hardware_morphology/).

## Phase 0 — instrumentation & π-recalibration (2026-08-20 evening)

Baked (unit corrections to the 2.0 s gait cycle at CAM_VEL_SCALE=π, not hypotheses):
- tripod band min/max_period 0.10/0.60 → 0.30/1.40 s; corr_tau 0.20 → 0.70 s
  (old band was two independent kills at π: out-of-band crossings AND var collapse)
- air-time threshold 0.20 → 0.38 s (same 0.55× selectivity vs the 0.68 s return stroke)
- spin/phase-lock speed_ref 4.0 → 2.8 (was capping both terms at 0.785 of nominal)
- foot-idle max_idle_steps 60 → 90 (1.2 s taxed legitimate sub-throttle swings)
- max_speed_scale 1.75 → 1.05 (paid the wheelie to 1.14 m/s; KRABBY_MAX_SPEED_SCALE)
- cam-ACCELERATION slew baked in the action term (CAM_ACCEL_LIMIT_RAD_S2 = 2π rad/s²:
  full speed over 0.5 s — hardware gearmotors ramp; kills instant-full-throttle)

Instrumentation:
- override block widened: KRABBY_{ANGVEL,ORIENT,ACTION_RATE,DELTA_TORQUE,EXCESS_CONTACT,
  STANCE_SUPPORT,STRIDE,AIRTIME,SWING_MIN_CLEAR}_W; KRABBY_TRIPOD_{MIN_PERIOD,MAX_PERIOD,
  CORR_TAU,MIN_AMP}; KRABBY_MAX_SPEED_SCALE; KRABBY_FOOT_CLEAR_FLAT (lifts the
  parkour_flat mask on the clearance terms); KRABBY_LIN_VEL_X ("lo:hi");
  KRABBY_CAM_CLIP_LO (unidirectional cam clamp, lower raw-action bound)
- penalty_swing_min_clearance registered in the flat-walk stack (inert)
- eval: scenarios_v2.yaml (10 s holds → ≥4 cycles/window), root fallback 1.0 → 1.05

Unit suite: 270 passed.

Replay gate (replay_tripod_band_gate.py, new-plant fixtures @ weight 0.30): PASS —
ideal-2s-tripod 6.258/min (56.7 paid crossings/min); wheelie 0.000; zero-action stand
0.000; ideal under the OLD band 0.000 (the old band cannot pay a perfect 2 s tripod —
the recalibration was necessary, not cosmetic). Fixtures:
parkour/logs/rsl_rl/gait_eval/gait_formation_fixtures/ (smoke2 model_1999 rollout +
zero-action, 4 episodes each, --save-raw). Manifest v2 accepted by the loader
(schedule.py); scenarios_v2 checkpoint placeholders are per-run CLI overrides.

Eval sample size (2026-08-21, user decision): scenarios_v2 episodes 10 -> 100
(episodes == parallel envs, single rollout of the 36 s schedule; measured cost ~70 s ->
est. 2-3 min; completion-rate CI ~±0.28 -> ~±0.09). Applies from the C9 evals onward —
earlier completion numbers (C5/C7 validations) carry the n=10 caveat.

KRABBY_RESAMPLE_S override added (2026-08-21): command resampling "lo:hi" seconds
(training resampled 6 s vs the eval's 10 s holds — the sustained-hold mismatch).

## Arm log

| phase | arm | delta | 1k screen (ep_len / fail% / reward) | eval (tripod / clearance / spin ratio) | verdict |
|---|---|---|---|---|---|
| A | B0 | baked Phase-0 stack only | 68 / 100% / 2.06 | — | baseline. Slew alone does NOT clear the wheelie (ramped full-throttle still tips); run 2026-08-20_21-12-52 |
| A | A1 | ANGVEL_W=-0.1 | 76 / 100% / 2.08 | — | null: pitch-rate tax can't outbid wheelie income at the hot envelope |
| A | A2 | ANGVEL_W=-0.3 | 72 / 100% / 1.98 | — | null (higher dose no better) |
| A | A3 | ORIENT_W=-1.5 | 66 / 100% / 2.03 | — | null: tilt² tax invisible until the tip is already unrecoverable |
| A | A4 | TRACK_L1_W=-1.0 | 72 / 100% / 2.09 | — | null: overspeed error priced but income still dominates |
| A | A5 | LIN_VEL_X=0.0:0.35 | **367 / 91% / 7.68** | — | ALIVE (5.4× B0): slow commands drain the full-throttle incentive; combo round built on A5 |
| A | A5-diag | rollout on model_999 | — | — | slow-burn wheelie: survives 10+ s, vx creeps 0.17→0.73 vs 0.35 cmd, rear-feet-only posture develops early, pitch grinds to the 0.5 limit. Overspeed persists because the exp tracking well (σ²=0.02) is BLIND past ±0.25 m/s error — only L1 prices the runaway |
| A | C1 | A5 + ANGVEL -0.1 | 436 / 90% / 8.99 | — | helps on the slow-command landscape (penalties inert alone, ~+19% here) |
| A | C2 | A5 + ORIENT -1.5 | 384 / 91% / 7.81 | — | marginal |
| A | C3 | A5 + TRACK_L1 -1.0 | **442 / 88% / 9.70** | — | best combo — overspeed pricing is the right axis, consistent with the diag |
| A | C4 | A5 + ANGVEL + ORIENT | 388 / 94% / 8.67 | — | penalty stacking without tracking pressure regresses |
| A | C5 | A5 + TRACK_L1 -1.0 + SIGMA2 0.25 | **881 / 45% / 20.0** | — | ep-len gate PASS. σ² was the missing gradient (C3 442 → 881 from the well alone) |
| A | C6 | C5 + ANGVEL -0.1 | 461 / 86% / 12.2 | — | angvel penalty actively harmful on the working stack — retired from Phase A |
| A | C7 | 0.0:0.25 + TRACK_L1 -1.0 + SIGMA2 0.25 | **904 / 18.1% / 25.2** | — | BOTH GATES PASS. Caveat: lin_vel_clip 0.2 zeroes most commands in this range — validation must show it walks, not just stands |
| A | C8 | C5 with TRACK_L1 -2.0 | 508 / 80% / 11.4 | — | L1 overdose taxes exploration income; -1.0 is the dose |
| A | C7-5k | C7 stack, 5k validation | 926 / 19.5% / 30.7 | completion **0.0** (10/10 falls on flat_walk_slow_v2) | REJECTED: 0.0:0.25 range trains a stand-specialist that falls out-of-range |
| A | C5-5k | C5 stack, 5k validation | 931 / 19.5% / 26.8 | completion **0.7** (7 complete / 3 falls); shaft ~1.28 rad/s; tripod ~0; tippy 0.36; spin ratio 0.04 | seed-1 pass; seed-2 reproduction REQUIRED |
| A | C5-5k-s2 | same stack, seed 2 | 940 / 25% / 28.4 (train REPRODUCES) | n=10: 0.1 → **n=100: 0.34** | see correction row below |
| A | n=100 re-evals | user decision: episodes 10 → 100 | — | C5 seed1 **0.23** (was 0.70 at n=10!), seed2 **0.34** (was 0.10) | **CORRECTION: the seeds REPRODUCE (0.23 vs 0.34, CIs overlap) — the "reproduction failure" was n=10 noise in both directions. True C5 completion ≈ 0.3: consistent but NOT bake-worthy.** Prior verdicts re-scored. C9 margin arm still the right next test (its evals run at n=100); if C9 also lands ~0.3, next mechanism is the 6 s-resample vs 10 s-hold training/eval mismatch |
| A | C9-s1 | C5 + range 0.0:0.40, seed 1 | 810 / 36% | completion 0.4 (n=10 — eval predates the n=100 change; re-eval queued) | — |
| A | C9-s2 | same, seed 2 | 872 / 25% | completion **0.28 (n=100)** | MARGIN HYPOTHESIS DEAD: statistically identical to C5 (~0.3). → hold-mismatch arms C10/C11 (KRABBY_RESAMPLE_S) as 1k screens |
| A | C9-s1-n100 | re-eval | — | completion **0.34 (n=100)** | plateau confirmed 4th time: all 5k runs 0.23–0.34 |
| A | C10 | C5 + RESAMPLE_S 10:10 | 777 / 54% (worse than C5-1k 881/45%) | completion **0.00 (n=100)** | hold-mismatch hypothesis DEAD |
| A | C11 | C5 + RESAMPLE_S 8:12 | 636 / 64% | completion **0.00 (n=100)** | same. Mechanism: 10 s holds → ~2 commands/episode → command-diversity starvation + standing-dominated batches |
| A | C5-1k-ctrl | control eval of the C5 1k screen | — | completion **0.12 (n=100)** | scale control: 0.12@1k → ~0.3@5k — completion still climbing with training scale (batch-continuation candidate) |
| A | C5-5k-diag | rollout + fall-time histograms | — | — | **PHASE A CLOSING DIAGNOSIS**: zero falls in 30 s under training-style commands; eval falls concentrate uniformly through the WALKING holds (~5-8%/s hazard), ~none while standing. The 0.3 completion ceiling is GAIT-LIMITED, not survival-limited: no limit cycle (tripod 0.0) → constant stumble hazard. Phase B is the direct attack on the remaining failures |
| A | C5-10k | C5 seed1 5k resumed +5k (batch continuation) | mean ep len grew through 10k | completion **0.92 (n=100)**; falls 8, all in the 0.35 hold's last 8 s; tripod 0.0 | **PLATEAU BROKEN BY SCALE ALONE**: 0.12@1k → ~0.3@5k → 0.92@10k. The 5k constant-hazard diagnosis was right about mechanism, wrong about asymptote — continued training shrinks the hazard without gait formation. Phase A survival gate smashed; seed2 reproduction launched |
| A | creep-audit | achieved-vx from eval raws (user prompted by 10k video) | — | 5k-s1 0.044/0.041, 5k-s2 0.067/0.103, 10k 0.047/0.057 m/s vs cmds 0.25/0.35 | **TRACKING-ABANDONMENT HACK across the whole C5 lineage** (~15-30% of command; also drifts ~0.05 at cmd 0.0). Scale bought fall-avoidance-while-creeping, not walking. Root cause: sigma2 0.25 flattens the speed gradient; survival + weak tracking pay beats striding. Phase A gates must add an achieved-speed floor; bake proposal reframed |
| — | tracking-instr | eval harness: tracking_ratio first-class (user directive) | — | rescored: 10k ratio-median 0.16, 5k-s1 0.07, 5k-s2 0.24 | `tracking_metrics` gains cmd-floor-gated `ratio`; computed per hold regardless of tripod validity; aggregated as `tracking_ratio` + `tracking_by_hold` in scenario_metrics.json, summary, compact rows. Offline rescore via rescore_tracking.py (originals untouched, `*_tracking.json` alongside). Phase B winner gate added to PLAN D: ratio median ≥ 0.5 (strong ≥ 0.7). 44 gait-eval unit tests green (stale v2-manifest test fixed) |
| A | C5-10k-s2 | seed2 5k resumed +5k (reproduction of C5-10k) | train exit 0 (run 2026-08-21_14-55-08) | completion **0.38 (n=100)**, tracking_ratio **0.094**, tripod 0.0; falls 62 | **10k BREAKTHROUGH DOES NOT REPRODUCE** (seed1 0.92 vs seed2 0.38). Worse: seed2's ratio fell 0.24→0.09 as completion rose 0.34→0.38 — the scale gradient descends into deeper creep (slower = safer). PHASE A CLOSED: survival infrastructure real (wheelie dead, stand stable), but the stack's attractor is creep-survival; no reproducible walking winner exists under completion + tracking gates. Bake decision → user |

## Phase B — tripod formation (opened 2026-08-21, user baked C5 stack as base)

Base (all arms): `KRABBY_LIN_VEL_X=0.0:0.35 KRABBY_TRACK_L1_W=-1.0 KRABBY_TRACK_SIGMA2=0.25` on Phase-0 defaults, from scratch, seed 1, 1k screens.
Ranking: tripod formation (train income + eval tripod) + tracking_ratio; completion is a constraint, not the headline. Winner gates (5k): tripod ≥ 0.3, tracking_ratio ≥ 0.5, survival maintained, one_direction_ratio reported.
Wave 1 arms: B_T30C (tripod 0.3 + cam clip lo 0.1 — priority, the clamp+income recipe), B_T30, B_T15, B_T50, B_CLAMP (clamp-alone control), B_T30S (tripod 0.3 + sigma2 0.1 speed-pressure arm, user-approved amendment). Eval: flat_walk_slow_v2 n=100 (in-range band; forward_v2's holds exceed the 0.35 cap).
| B | B_T30C | tripod 0.3 + clamp 0.1 | ep len 751; tripod income ~0 (15/1000 iters, max 1e-4) | completion 0.01, ratio 0.14, tripod 0.0 | income never fires |
| B | B_T30 | tripod 0.3 | ep len 543; income 4/1000 | completion 0.0, ratio 0.23, tripod p75 1e-4 | income never fires |
| B | B_T15 | tripod 0.15 | ep len 834; income 0/1000 | completion 0.12, ratio 0.10 | ≈ C5-1k baseline; weak income inert |
| B | B_T50 | tripod 0.5 | ep len 733; income 8/1000 | completion 0.0, ratio 0.22 | income never fires |
| B | B_CLAMP | clamp 0.1 alone | ep len 811 | completion 0.02, ratio 0.15, tripod p75 0.009 | faint tripod tail; clamp alone insufficient (as lit predicted) |
| B | B_T30S | tripod 0.3 + sigma2 0.1 | ep len 415 | completion 0.0, ratio **0.43** (n=8), tripod median **0.0026** (only nonzero) | speed pressure forces stepping attempts; survival collapses at 1k |
| B | wave-1 verdict | — | — | — | **TRIPOD TERM IS GRADIENT-DEAD at creep-shuffle**: crossing credit needs swap crossings that never occur (min_amp 0.15 gate); weight sweeps irrelevant. Wave 2 = dense-gradient ladder: EXCESS_CONTACT + AIRTIME to cause stepping, TRIPOD_MIN_AMP 0.05 so embryonic swaps pay, tripod 0.3 as target |
| B | replay-gate | MIN_AMP 0.15→0.05 retune (wave-2 ladder) | offline replay on fixtures | ideal 6.258/min unchanged; wheelie 0; stand 0 | **PASS** — relaxed amp gate pays no known degenerate; wave-2 screens legitimate per the replay-gate rule |
| B | B_DENSE_T30 | ladder (XC+AIR+T30+amp0.05) | income 4/1000 | tripod p75 0.004, ratio n=1, completion 0.0 | ladder did not light income |
| B | B_DENSE_T30C | ladder + clamp | income 7/1000 | completion 0.12, ratio 0.088 | clamp reverts to creep-survival |
| B | B_DENSE_T30S | ladder + sigma2 0.1 | income 10/1000 | tripod med 0.00045/p75 0.0046, ratio **0.39** (n=55), completion 0.01 | best stepping+speed combo |
| B | B_XC | excess-contact −0.4 alone | — | ratio 0.35 (n=11), tripod 0.0 | lifts without alternation |
| B | B_AIR | airtime 1.2 alone | — | tripod med **0.0037**/p75 **0.015** (campaign best), ratio 0.35 (n=48) | dense swing pay alone produces the strongest embryonic alternation |
| B | wave-2 verdict | — | — | — | MIN_AMP 0.05 does NOT bridge the bootstrap gap (income stays dead). Dense airtime is the formation driver; speed pressure keeps it honest. Promotion wave (5k): B_AIR, B_AIR_T30 (does income engage once stepping exists at scale?), B_DENSE_T30S, B_T30S. Question: does stepping survive survival-consolidation at 5k or does creep reassert? |
| B | promote-verdict | 5k frontier | — | P_AIR 0.25/0.32; P_AIR_T30 0.14/0.40 (income 354); P_DENSE_T30S 0.0/0.57; P_T30S 0.05/**0.74** (income 205) | Speed pressure produces first real walking (74% of cmd). But income flickers at 1e-4 (not compounding) and band-replay on P_T30S raws shows band-widening changes nothing (0.026/min all variants): **no anti-phase alternation exists — legs step at random phase**. New lead mechanism: cam-phase organization (spin-gated PhaseLock + clamp) on the T30S walking base. XC term implicated in P_DENSE 100% falls |
| B | wave-3 launch | — | — | — | Arms: P_T30S_cont (+5k consolidation test), P_T30S_PL (phase-lock 0.1), P_T30S_SPIN_PL (+spin 0.2), P_T30S_C_PL (clamp 0.1 + phase-lock 0.1, structural spin). All on T30S base (tripod 0.3, sigma2 0.1) |
| — | PLANT FIX | leg self-collision (user caught interpenetration in P_T30S video) | femur collisionEnabled=0 + tibia collider missing since primitive era | — | **All prior campaign numbers are old-plant.** Wave 3 killed mid-arm-1; relaunching on the fixed plant with P_T30S_cont replaced by fresh P_T30S (re-validates the walking base). Behavioral findings (creep attractor, sigma2 speed pressure, phase-organization gap) carried as hypotheses, not results. Statics gate pending |
| — | statics-gate | collision-fixed plant settle | — | pitch −1.39°, roll 0.06°, smooth trace, mass 230.06, six feet loaded 35–701 N | **PASS** — no spurious self-contacts at neutral. Wave 3 relaunched on fixed plant: P_T30S_v2 (fresh 5k base re-validation), P_T30S_PL, P_T30S_SPIN_PL, P_T30S_C_PL |
| — | CAMPAIGN HALT | user directive post collision-fix | — | — | Wave 3 stopped mid-arm-1 (P_T30S_v2 reached ~iter 3400, checkpoints retained in 2026-08-22_08-58-48). Collision fix invalidated prior quantitative results; user ordered full plan revision against the three lit reviews (stability, continuous-spin, plasticity). All rewards on the table incl. mirror symmetry; new rewards and action spaces in scope |
