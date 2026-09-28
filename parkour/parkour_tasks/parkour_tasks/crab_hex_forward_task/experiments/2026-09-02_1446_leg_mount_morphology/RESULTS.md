<!-- paths-note -->
> **Paths note (2026-09-09):** this campaign moved from `sim_fine_tuning/2026-09-02_1446_leg_mount_morphology/` to `parkour/parkour_tasks/parkour_tasks/crab_hex_forward_task/experiments/2026-09-02_1446_leg_mount_morphology/`. Absolute paths below (`/home/.../sim_fine_tuning/...`, `parkour/logs/rsl_rl/...`) name raw artifacts that stay on disk untracked; the tracked checkpoint of record is `head/` (see `bundle.yaml`) and the eval summaries are in place / under `evals/`.

# Leg-mount morphology campaign — RESULTS

**STATUS: PAUSED 2026-09-03 11:40 (user) — default plant = golden crab_simple.usda; see the "CAMPAIGN PAUSED" entry at the end.**

Ledger of rungs, gates-as-scores, and the final decision table. Reporting contract: blocks
end with `>>> ENTRY <marker>` (monitor relay). Plan: PLAN.md / approved plan file.

## CAMPAIGN OPEN — 2026-09-02 14:46
- code landed: `MorphVariant` in `assets/scripts/generate_crab_simple.py` (default byte-identical;
  variant tests), `crab_hex_foot_fk.py` (13 tests), `support_polygon_metrics` +
  `fall_direction_metrics` in `gait_eval/metrics.py` (7 tests), 7 variant USDs + MANIFEST.
- rung (i) running: `kinematic_screen.py` over 30k-flat (100 ep), 30k-light-obstacle (100 ep),
  20k-reference-obstacle (100 ep) traces × 9 rows (8 variants + the inward A−20 control);
  `clearance_sweep.py` over cam phase × tripod/in-phase offsets × stance/lifted poses.
>>> ENTRY campaign open

## RUNG (i) — kinematic screen (0 GPU) — 2026-09-02 15:40
Rigid mount transform (same joint angles as the recorded policy) on 300 stored eval episodes
(30k head flat; 30k head light obstacles; 20k reference obstacles). Identity check: the
`base` row reproduces the historical leading-contact tip 18°/13° (p50/p10) exactly.
**Primary statistic = polygon forward tip angle** (CoM projection to the loaded-foot hull's
forward boundary; the tripod's forward edge is a diagonal, so this is far smaller than the
leading-contact measure). Values below: 30k-flat trace set; the two obstacle sets track it
within ~1° (full CSV: `kinematic_screen.csv`).

| variant | walk polygon tip p50/p10 | walk lead-contact tip p50/p10 | pre-fall polygon tip p50/p10 | pre-fall frac neg margin | G1 | G1-strong (lead p50 ≥ 28.6°) | G2 (−40% neg) |
|---|---|---|---|---|---|---|---|
| base | 6.9° / 1.7° | 18.1° / 14.1° | 1.7° / −4.4° | 0.33 | — | — | — |
| A−20 (insect ctrl) | 1.0° / −3.6° | 7.1° / 2.9° | −8.4° / −14.8° | 0.97 | miss | miss | miss (worse) |
| B | 8.8° / 3.7° | 22.0° / 18.0° | 5.5° / −1.2° | 0.15 | miss (lead) | miss | pass |
| A10 | 9.5° / 4.4° | 22.8° / 19.0° | 6.6° / 0.0° | 0.10 | miss (lead) | miss | pass |
| A15 | 10.8° / 5.7° | 24.9° / 21.2° | 8.9° / 1.9° | 0.04 | marginal (lead p50 24.9) | miss | pass |
| A20 | 12.1° / 7.0° | 26.7° / 23.2° | 11.1° / 3.7° | 0.01 | pass | miss | pass |
| A10+B | 11.5° / 6.3° | 26.4° / 22.7° | 10.3° / 2.8° | 0.03 | pass | miss | pass |
| A15+B | 12.8° / 7.7° | 28.3° / 24.8° | 12.6° / 4.7° | 0.01 | pass | marginal (28.3) | pass |
| A20+B | 14.1° / 9.0° | 30.1° / 26.7° | 14.6° / 6.4° | 0.00 | pass | **pass** | pass |

Readings:
- The current plant walks with its CoM ~7° (median) from the polygon's forward edge and
  inside 2° at p10 — a dynamic topple needs almost no impulse. Every outward variant widens
  this monotonically; **the insect layout (front legs forward) is catastrophic**, settling
  the direction question with numbers.
- Leading-contact tip angles land within ~2° of the rung-(i) predictions (A20 26.7° vs 28.6°
  predicted; A20+B 30.1° vs 31.9°) — the idealized model slightly over-predicts because the
  transform rotates about the true leg axis with the true foot offsets.
- Honest scale: no variant brings the *polygon* margin near the 28.6° limit; the gains are
  1.5–2× on the walking median and, more importantly, the pre-fall negative-margin fraction
  falls from 33% to ≤ 4% for A15 and above (0% for A20+B).
- Clearance sweep (`clearance_sweep.csv`): with tripod phasing at FULL cam throw, adjacent
  same-side legs' swept planes cross on every variant (min clearance base −75 mm → A20 −51 →
  A20+B −41 mm; in-phase sweeps are clear by ≥ 117 mm). The plant relies on the policy never
  putting neighbours at opposite yaw extremes simultaneously; splay reduces the crossing depth
  but cannot remove it within the 20° cap. G3: all variants "not worse than base"; none strong.
- Sideways-walking reachability (α ≤ 25°): all variants pass by construction (cap 20°).
>>> ENTRY rung i kinematics

## RUNG (ii) — statics battery (20 jittered settles + cam sweep per configuration)
| config | sound? | upright/20 | penetr. | leg-link contact max N (settle / sweep) | A-share mean±sd | pitch_eq deg | root_z | com_z | yaw range (deg, RR) |
|---|---|---|---|---|---|---|---|---|---|
| base | yes | 20 | 0 | 0.0 / 0.0 | 55.7±2.5% | -0.41 | 1.0612 | 1.0167 | -26.1..+22.3 |
| B | yes | 20 | 0 | 0.0 / 0.0 | 63.3±10.5% | +0.89 | 1.0631 | 1.0186 | -25.8..+23.5 |
| A10 | yes | 20 | 0 | 0.0 / 0.0 | 53.5±1.5% | -0.27 | 1.0649 | 1.0203 | -25.0..+24.7 |
| A15 | yes | 20 | 0 | 0.0 / 0.0 | 40.8±13.2% | -1.40 | 1.0629 | 1.0184 | -25.4..+23.7 |
| A20 | yes | 20 | 0 | 0.0 / 0.0 | 54.5±9.2% | -0.96 | 1.0634 | 1.0189 | -24.3..+25.4 |
| A10+B | yes | 20 | 0 | 0.0 / 0.0 | 48.1±7.0% | +1.50 | 1.0630 | 1.0185 | -25.7..+24.4 |
| A15+B | yes | 20 | 0 | 0.0 / 0.0 | 59.0±7.9% | +0.42 | 1.0620 | 1.0175 | -25.1..+24.1 |
| A20+B | yes | 20 | 0 | 0.0 / 0.0 | 46.2±4.0% | +0.00 | 1.0623 | 1.0177 | -24.7..+24.1 |
- base root_z 1.0612 m: variants within ±10 mm keep KRABBY_HEX_SPAWN_Z=1.085
- sound = >=19/20 settles upright, no penetration, femur/hip contact < 15 N at neutral
>>> ENTRY rung ii statics

## RUNG (ii) — cross-check + scored columns (offline, from statics npz)
| config | sound? | FK residual max mm (Δ vs base) | standing polygon tip (feet loaded) | standing lead-contact tip (Δ vs base) | fwd margin m | K proxy ratio (d² model) | root_z Δ mm | tilt max settle deg | A-share sd | video |
|---|---|---|---|---|---|---|---|---|---|---|
| base | yes | 0.0 (+0.0) | 13.1° (6) | 13.1° (+0.0°, —) | 0.229 | 1.00× | +0.0 | 1.9 | 2.5% | rl-video-step-2850.mp4 |
| B | yes | 0.0 (+0.0) | 15.1° (5) | 15.1° (+2.0°, miss) | 0.265 | 1.28× | +1.9 | 4.6 | 10.5% | rl-video-step-2850.mp4 |
| A10 | yes | 0.0 (+0.0) | 16.2° (6) | 16.3° (+3.2°, miss) | 0.287 | 1.68× | +3.7 | 1.0 | 1.5% | rl-video-step-2850.mp4 |
| A15 | yes | 0.0 (+0.0) | 10.8° (4) | 19.1° (+5.9°, pass) | 0.188 | 1.45× | +1.8 | 3.2 | 13.2% | rl-video-step-2850.mp4 |
| A20 | yes | 0.0 (+0.0) | 20.3° (6) | 20.6° (+7.4°, pass) | 0.365 | 2.50× | +2.2 | 1.9 | 9.2% | rl-video-step-2850.mp4 |
| A10_B | yes | 0.0 (+0.0) | 18.2° (6) | 18.2° (+5.1°, pass) | 0.323 | 2.66× | +1.8 | 2.7 | 7.0% | rl-video-step-2850.mp4 |
| A15_B | yes | 0.0 (+0.0) | 21.0° (6) | 21.1° (+8.0°, pass) | 0.378 | 3.21× | +0.8 | 1.3 | 7.9% | rl-video-step-2850.mp4 |
| A20_B | yes | 0.0 (+0.0) | 23.1° (5) | 23.1° (+10.0°, strong) | 0.420 | 3.75× | +1.1 | 0.8 | 4.0% | rl-video-step-2850.mp4 |
- FK residual = |Isaac footpad (body frame) − offline FK toe| at the last settle; the base row's value is the footpad-vs-toe-point offset common to all rows, so Δ vs base is the mount-transform check (plan tolerance 5 mm)
- standing lead-contact tip gate: pass ≥ base +4°, strong ≥ base +8° (plan: baseline +8°); K proxy = Σ(x_foot − x_com)² over loaded feet, a geometric d² stand-in, relative only
- root_z Δ within ±10 mm keeps the default KRABBY_HEX_SPAWN_Z for the variant
- standing polygon tip is a single-frame value from the last settle (a 5-foot settle exposes a diagonal forward edge); the lead-contact tip is the gate statistic
>>> ENTRY rung ii crosscheck

## STOP 1 — 2026-09-02 15:33 — rung (ii) sign-off (waiting for the user)
- All 8 configurations are plant-sound: 20/20 settles upright, no penetration, zero hip/femur
  contact at neutral and through the full cam sweep, yaw ranges ±25° reached on every plant.
- FK cross-check: offline toe FK matches Isaac's settled footpads to 0.01 mm on all six legs of
  all 8 plants (right legs use the documented negated-knee convention; pinned by
  `tests/unit/test_crab_hex_foot_fk.py::test_fk_matches_isaac_settled_footpads_within_5mm`
  with the base settle as a fixture). The mount rotation + axis translation in the generated
  USDs are therefore exactly the transforms rung (i) assumed.
- Standing lead-contact tip vs base (gate: pass ≥ +4°, strong ≥ +8°): B +2.0 miss, A10 +3.2
  miss, A15 +5.9 pass, A20 +7.4 pass, A10+B +5.1 pass, A15+B +8.0 pass, A20+B +10.0 strong.
  Predicted-vs-measured (rung i leading-contact deltas at the walking pose vs neutral standing)
  agree in ordering; the neutral stance sits ~8° below the walking-pose values.
- d² stiffness proxy (Σ x² of loaded feet about the CoM): 1.3× (B) … 3.75× (A20+B); the plan's
  ≥ 2.5× consistency line is met by A20, A10+B, A15+B, A20+B. Single-settle values — A15's low
  polygon tip (10.8°, 4 feet loaded) is a lopsided settle, not a geometry effect.
- Spawn height: all variants settle within +0.8…+3.7 mm of base → KRABBY_HEX_SPAWN_Z unchanged.
- Infra notes: a configuration takes 2.2 min (the plan budgeted 25); the driver's
  wait-for-Isaac pgrep matched its own command line (and later the monitor shell's) and idled
  10 min per configuration — fixed in all four drivers by anchoring the pattern on the Isaac
  python invocation and excluding the driver pid.
- Next (after sign-off): rung (iii) `run_probe_matrix.py` (~40 min) then rung (iv)
  `run_transfer_probe.py` (48 evals, ~2.2 h) → STOP 2.
>>> ENTRY stop 1

## STOP 1 sign-off — 2026-09-02 15:40
- user: "go ahead with rungs iii and iv" → all 8 configurations proceed; rung (iii) probe matrix then
  rung (iv) transfer probe launched as one serial chain (`probe_matrix.log`, `transfer_probe.log`).
>>> ENTRY stop 1 signoff

## RUNG (iii) — open-loop scripted tripod gait (8 jittered envs per cell; in-session baseline)
| config | w | survivors/8 | ttf median s (ratio to base) | pitch-fwd share | vx first 2 s (ratio) | walk tip p50 | prefall frac neg | leg contact max N |
|---|---|---|---|---|---|---|---|---|
| base | 0.3 | 6 | 15.0 (1.00×) | 0.00 | -0.051 (1.00×) | 9.3° | 0.00 | 1186.4 |
| base | 0.5 | 4 | 10.8 (1.00×) | 0.00 | -0.085 (1.00×) | 9.9° | 0.25 | 1120.9 |
| B | 0.3 | 8 | 15.0 (1.00×) | — | -0.000 (0.00×) | 13.7° | — | 1753.4 |
| B | 0.5 | 8 | 15.0 (1.39×) | — | -0.009 (0.11×) | 12.5° | — | 2006.7 |
| A10 | 0.3 | 8 | 15.0 (1.00×) | — | -0.006 (0.12×) | 10.7° | — | 892.8 |
| A10 | 0.5 | 5 | 15.0 (1.39×) | 0.00 | 0.000 (-0.00×) | 9.4° | 0.24 | 1022.3 |
| A15 | 0.3 | 8 | 15.0 (1.00×) | — | 0.013 (-0.25×) | 11.7° | — | 1247.8 |
| A15 | 0.5 | 8 | 15.0 (1.39×) | — | 0.001 (-0.01×) | 9.9° | — | 904.4 |
| A20 | 0.3 | 6 | 15.0 (1.00×) | 0.00 | 0.003 (-0.06×) | 12.1° | 0.67 | 809.6 |
| A20 | 0.5 | 6 | 15.0 (1.39×) | 0.00 | -0.002 (0.03×) | 10.2° | 0.09 | 771.3 |
| A10+B | 0.3 | 8 | 15.0 (1.00×) | — | 0.012 (-0.24×) | 15.1° | — | 989.3 |
| A10+B | 0.5 | 8 | 15.0 (1.39×) | — | 0.001 (-0.01×) | 14.2° | — | 857.1 |
| A15+B | 0.3 | 8 | 15.0 (1.00×) | — | 0.009 (-0.17×) | 15.4° | — | 1035.7 |
| A15+B | 0.5 | 8 | 15.0 (1.39×) | — | 0.007 (-0.08×) | 14.8° | — | 803.1 |
| A20+B | 0.3 | 8 | 15.0 (1.00×) | — | 0.011 (-0.21×) | 15.6° | — | 573.8 |
| A20+B | 0.5 | 8 | 15.0 (1.39×) | — | -0.011 (0.13×) | 15.5° | — | 943.7 |
- ttf median counts survivors at the full hold length; ratios are vs the in-session base row at the same w; no kills
>>> ENTRY rung iii open-loop

## RUNG (iii) — open-loop scripted tripod gait (8 jittered envs per cell; in-session baseline) — RESCORED (pre-reset fall frame; pre-fall leg contact split hip/femur vs tibia)
| config | w | survivors/8 | ttf median s (ratio to base) | pitch-fwd share | vx first 2 s (ratio) | walk tip p50 | prefall frac neg | hip/femur | tibia contact max N (pre-fall) |
|---|---|---|---|---|---|---|---|---|---|
| base | 0.3 | 6 | 15.0 (1.00×) | 0.50 | -0.051 (1.00×) | 9.3° | 0.00 | 0.0 | 1186.4 |
| base | 0.5 | 4 | 10.8 (1.00×) | 0.25 | -0.086 (1.00×) | 9.9° | 0.25 | 0.0 | 1120.9 |
| B | 0.3 | 8 | 15.0 (1.00×) | — | -0.000 (0.00×) | 13.7° | — | 0.0 | 1753.4 |
| B | 0.5 | 8 | 15.0 (1.39×) | — | -0.009 (0.11×) | 12.5° | — | 0.0 | 2006.7 |
| A10 | 0.3 | 8 | 15.0 (1.00×) | — | -0.006 (0.12×) | 10.7° | — | 358.7 | 892.8 |
| A10 | 0.5 | 5 | 15.0 (1.39×) | 0.33 | 0.000 (-0.00×) | 9.4° | 0.24 | 191.7 | 1022.3 |
| A15 | 0.3 | 8 | 15.0 (1.00×) | — | 0.013 (-0.25×) | 11.7° | — | 0.0 | 1247.8 |
| A15 | 0.5 | 8 | 15.0 (1.39×) | — | 0.001 (-0.01×) | 9.9° | — | 0.0 | 904.4 |
| A20 | 0.3 | 6 | 15.0 (1.00×) | 1.00 | 0.003 (-0.06×) | 12.1° | 0.66 | 0.0 | 809.6 |
| A20 | 0.5 | 6 | 15.0 (1.39×) | 0.00 | -0.002 (0.03×) | 10.2° | 0.09 | 40.3 | 771.3 |
| A10+B | 0.3 | 8 | 15.0 (1.00×) | — | 0.012 (-0.24×) | 15.1° | — | 0.0 | 989.3 |
| A10+B | 0.5 | 8 | 15.0 (1.39×) | — | 0.001 (-0.01×) | 14.2° | — | 0.0 | 857.1 |
| A15+B | 0.3 | 8 | 15.0 (1.00×) | — | 0.009 (-0.17×) | 15.4° | — | 41.0 | 1035.7 |
| A15+B | 0.5 | 8 | 15.0 (1.39×) | — | 0.007 (-0.08×) | 14.8° | — | 50.3 | 803.1 |
| A20+B | 0.3 | 8 | 15.0 (1.00×) | — | 0.011 (-0.21×) | 15.6° | — | 301.7 | 573.8 |
| A20+B | 0.5 | 8 | 15.0 (1.39×) | — | -0.011 (0.13×) | 15.5° | — | 943.4 | 943.7 |
- ttf median counts survivors at the full hold length; ratios are vs the in-session base row at the same w; no kills
>>> ENTRY rung iii open-loop rescored

## RUNG (iii) notes — 2026-09-02 16:12
- The scripted tripod is nearly stationary on every plant (|vx| ≤ 0.09 m/s), so the rung-(iii)
  signal is survival, not locomotion: base keeps 6/8 (w 0.3) and 4/8 (w 0.5) envs upright over the
  15 s hold; every outward variant keeps ≥ 6/8 and most keep 8/8 at both speeds. Falls on base are
  split pitch-forward / pitch-back (the open-loop sweep rocks the platform both ways).
- Hip/femur contact (pre-fall frames) is zero on base, B, A15, A10+B and brief (1–4 frames) on A10
  (359 N, one env), A15+B (≤ 50 N), A20 (40 N) and A20+B (132–943 N, three envs): localized to
  adjacent-leg crossings when the mid leg is at full throw toward a near-perpendicular outer leg
  (cam-derived yaws −17°/+25° at contact) — the crossing rung (i) predicts for every plant at tripod
  phase. On base the same crossing engages tibias only, which the tibia channel cannot separate
  from tibia–ground load (tibias carry 0.6–1 kN of ground contact 35–80 % of frames). Scored, not a
  soundness fail: the trained policy chooses the phase relation; rung (v) prices it via
  reward_collision.
- Rung (iv) first attempt crashed at the run-metadata step (harness `os` scope bug in the new
  usd_path field); fixed 16:09 and the transfer driver restarted from scratch.
>>> ENTRY rung iii notes

## CoM proxy check — 2026-09-02 16:16 (`com_check.py` on the 30k head, slow scenario, base plant)
- Exact whole-body CoM (body positions × masses, 230.1 kg) sits 39 mm below and 4 mm behind the
  root origin while walking. Polygon tip angles with the exact CoM are +0.6° (walking) / +0.9°
  (pre-fall) above the root-proxy values; pre-fall negative-margin fraction 0.30 exact vs 0.42
  proxy. The proxy used in rungs (i)–(iv) is therefore ~1° conservative and identical across
  configurations (same rigid offset), so variant-vs-base deltas are unaffected.
- Rung (iv) base identity check: walking polygon tip 6.7° (proxy) vs rung (i) 6.9° on the same
  head — the augmented harness reproduces the kinematic screen within 0.2°.
>>> ENTRY com proxy check

## RUNG (iv) notes — 2026-09-02 16:45 — plant selection defect and restart
- The manifest `env:` block cannot select the plant: `crab_hex_scene_cfg.py` reads
  `KRABBY_HEX_USD_PATH` when the config module is imported, which happens before the harness
  applies scenario env vars (the terrain vars in the same block are read at cfg instantiation and
  do apply). The first 17 variant evals of the 30k head therefore ran on the golden plant and
  reproduced the base rows exactly (19 / 100 / 74 falls) — discarded to
  `logs/rsl_rl/gait_eval/leg_mount_morphology/invalid_no_usd_env/`. Rungs (ii) and (iii) set the
  variable in the process environment and are unaffected.
- Fix: both eval drivers export `KRABBY_HEX_USD_PATH` per configuration in the eval process
  environment; `run_meta.usd_path` now records the plant actually spawned (from the robot cfg)
  next to `usd_path_requested`. Driver restarted 16:45 (base rows kept; 45 evals to go, ~1.2 h).
- Base rows of the 30k head (valid): slow 19 falls / completion 0.81; fwd 100 / 0.00 (the 0.30–0.65
  holds are outside the 0.0–0.35 trained band — the fwd scenario saturates for these heads and its
  fall delta carries no information; step-onset 74 / 0.26 is the informative scenario).
>>> ENTRY rung iv notes

## RUNG (iv) — checkpoint transfer probe (same policy, variant plant; one-sided)
| head | config | scenario | completion | falls | pitch_fwd / back / roll | tripod | tracking | walk polygon tip p50 (deg) | prefall frac neg |
|---|---|---|---|---|---|---|---|---|---|
| 30k | base | slow | 0.81 | 19 | 19 / 0 / 0 | 0.580 | 0.420 | 6.7 | 0.42 |
| 30k | base | fwd | 0.00 | 100 | 100 / 0 / 0 | 0.430 | 0.457 | 4.6 | 0.22 |
| 30k | base | step | 0.26 | 74 | 74 / 0 / 0 | 0.606 | 0.543 | 5.8 | 0.41 |
| 30k | B | slow | 0.98 | 2 | 2 / 0 / 0 | 0.620 | 0.389 | 9.0 | 0.58 |
| 30k | B | fwd | 0.60 | 40 | 40 / 0 / 0 | 0.458 | 0.308 | 7.6 | 0.27 |
| 30k | B | step | 0.70 | 30 | 30 / 0 / 0 | 0.597 | 0.399 | 8.3 | 0.3 |
| 30k | A10 | slow | 1.00 | 0 | 0 / 0 / 0 | 0.634 | 0.383 | 10.0 | None |
| 30k | A10 | fwd | 0.67 | 33 | 33 / 0 / 0 | 0.480 | 0.281 | 8.6 | 0.34 |
| 30k | A10 | step | 0.79 | 21 | 21 / 0 / 0 | 0.605 | 0.394 | 9.5 | 0.12 |
| 30k | A15 | slow | 0.99 | 1 | 1 / 0 / 0 | 0.646 | 0.380 | 11.6 | 0.61 |
| 30k | A15 | fwd | 0.72 | 28 | 28 / 0 / 0 | 0.515 | 0.280 | 9.9 | 0.2 |
| 30k | A15 | step | 0.94 | 6 | 6 / 0 / 0 | 0.597 | 0.380 | 10.9 | 0.19 |
| 30k | A20 | slow | 0.98 | 2 | 2 / 0 / 0 | 0.653 | 0.369 | 13.4 | 0.27 |
| 30k | A20 | fwd | 0.69 | 31 | 31 / 0 / 0 | 0.524 | 0.278 | 11.4 | 0.11 |
| 30k | A20 | step | 0.93 | 7 | 7 / 0 / 0 | 0.612 | 0.368 | 12.5 | 0.06 |
| 30k | A10pB | slow | 1.00 | 0 | 0 / 0 / 0 | 0.662 | 0.374 | 12.2 | None |
| 30k | A10pB | fwd | 0.76 | 24 | 24 / 0 / 0 | 0.523 | 0.290 | 10.7 | 0.13 |
| 30k | A10pB | step | 0.98 | 2 | 2 / 0 / 0 | 0.599 | 0.361 | 11.9 | 0.42 |
| 30k | A15pB | slow | 1.00 | 0 | 0 / 0 / 0 | 0.658 | 0.367 | 14.0 | None |
| 30k | A15pB | fwd | 0.78 | 22 | 22 / 0 / 0 | 0.539 | 0.280 | 12.3 | 0.09 |
| 30k | A15pB | step | 1.00 | 0 | 0 / 0 / 0 | 0.600 | 0.365 | 13.3 | None |
| 30k | A20pB | slow | 1.00 | 0 | 0 / 0 / 0 | 0.660 | 0.374 | 15.5 | None |
| 30k | A20pB | fwd | 0.81 | 19 | 19 / 0 / 0 | 0.549 | 0.269 | 13.7 | 0.02 |
| 30k | A20pB | step | 0.99 | 1 | 1 / 0 / 0 | 0.623 | 0.355 | 14.9 | 0.08 |
| 20k | base | slow | 0.95 | 5 | 5 / 0 / 0 | 0.601 | 0.439 | 7.1 | 0.65 |
| 20k | base | fwd | 0.05 | 95 | 95 / 0 / 0 | 0.390 | 0.362 | 5.9 | 0.39 |
| 20k | base | step | 0.64 | 36 | 35 / 1 / 0 | 0.577 | 0.462 | 6.1 | 0.37 |
| 20k | B | slow | 1.00 | 0 | 0 / 0 / 0 | 0.645 | 0.418 | 9.3 | None |
| 20k | B | fwd | 0.94 | 6 | 6 / 0 / 0 | 0.484 | 0.311 | 8.5 | 0.35 |
| 20k | B | step | 0.83 | 17 | 17 / 0 / 0 | 0.583 | 0.438 | 8.1 | 0.16 |
| 20k | A10 | slow | 1.00 | 0 | 0 / 0 / 0 | 0.656 | 0.416 | 10.0 | None |
| 20k | A10 | fwd | 1.00 | 0 | 0 / 0 / 0 | 0.520 | 0.309 | 9.1 | None |
| 20k | A10 | step | 0.94 | 6 | 6 / 0 / 0 | 0.614 | 0.427 | 9.4 | 0.15 |
| 20k | A15 | slow | 1.00 | 0 | 0 / 0 / 0 | 0.661 | 0.399 | 11.6 | None |
| 20k | A15 | fwd | 0.99 | 1 | 0 / 1 / 0 | 0.557 | 0.309 | 10.5 | 0.0 |
| 20k | A15 | step | 0.97 | 3 | 3 / 0 / 0 | 0.620 | 0.395 | 10.8 | 0.0 |
| 20k | A20 | slow | 1.00 | 0 | 0 / 0 / 0 | 0.664 | 0.409 | 13.2 | None |
| 20k | A20 | fwd | 0.99 | 1 | 1 / 0 / 0 | 0.574 | 0.318 | 11.9 | 0.03 |
| 20k | A20 | step | 1.00 | 0 | 0 / 0 / 0 | 0.616 | 0.391 | 12.1 | None |
| 20k | A10pB | slow | 1.00 | 0 | 0 / 0 / 0 | 0.655 | 0.395 | 12.4 | None |
| 20k | A10pB | fwd | 1.00 | 0 | 0 / 0 / 0 | 0.568 | 0.312 | 11.1 | None |
| 20k | A10pB | step | 0.99 | 1 | 1 / 0 / 0 | 0.609 | 0.392 | 11.4 | 0.17 |
| 20k | A15pB | slow | 1.00 | 0 | 0 / 0 / 0 | 0.675 | 0.393 | 13.6 | None |
| 20k | A15pB | fwd | 1.00 | 0 | 0 / 0 / 0 | 0.587 | 0.311 | 12.5 | None |
| 20k | A15pB | step | 1.00 | 0 | 0 / 0 / 0 | 0.633 | 0.383 | 12.8 | None |
| 20k | A20pB | slow | 1.00 | 0 | 0 / 0 / 0 | 0.666 | 0.397 | 15.4 | None |
| 20k | A20pB | fwd | 1.00 | 0 | 0 / 0 / 0 | 0.599 | 0.320 | 14.0 | None |
| 20k | A20pB | step | 1.00 | 0 | 0 / 0 / 0 | 0.629 | 0.383 | 14.4 | None |
- fall-rate deltas vs the base row of the same head/scenario are the scored column (−30% = strong; unchanged = neutral: policy mismatch)
>>> ENTRY rung iv transfer

## STOP 2 — rungs i–iv combined (interim decision table; rung v pending the user's budget confirmation)
| config | hardware | splay | axis in | rung i walk tip p50/p10 | rung i lead tip p50 | rung i prefall frac neg | rung i min clearance mm (tripod) | rung i lateral reach loss mm | rung ii sound | rung ii standing lead tip Δ | rung ii K proxy | rung iii survivors/8 @0.3 / @0.5 | rung iii ttf ratio @0.5 | rung iii pitch-fwd share @0.5 | rung iii hip/femur N | rung iv 30k falls fwd / step (Δ vs base) | rung iv 20k falls fwd / step (Δ vs base) | rung iv 30k step pitch-fwd share (of falls) | rung iv 30k step prefall tip p25 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| base | none | 0° | 5.5 | 6.9 / 1.7 | 18.1 | 0.33 | -75 | 0 | yes | 0.0° | 1.00× | 6 / 4 | 1.00× | 0.25 | 0 | 100 / 74 | 95 / 36 | 1.00 | -1.0° |
| B | re-hinge 3 in | 0° | 2.5 | 8.8 / 3.7 | 22.0 | 0.15 | -75 | 0 | yes | 2.0° | 1.28× | 8 / 8 | 1.39× | — | 0 | 40 (-60) / 30 (-44) | 6 (-89) / 17 (-19) | 1.00 | -0.6° |
| A10 | 10° shims | 10° | 5.5 | 9.5 / 4.4 | 22.8 | 0.10 | -62 | 10 | yes | 3.2° | 1.68× | 8 / 5 | 1.39× | 0.33 | 192 | 33 (-67) / 21 (-53) | 0 (-95) / 6 (-30) | 1.00 | 1.6° |
| A15 | 15° shims | 15° | 5.5 | 10.8 / 5.7 | 24.9 | 0.04 | -56 | 22 | yes | 5.9° | 1.45× | 8 / 8 | 1.39× | — | 0 | 28 (-72) / 6 (-68) | 1 (-94) / 3 (-33) | 1.00 | 2.0° |
| A20 | 20° shims | 20° | 5.5 | 12.1 / 7.0 | 26.7 | 0.01 | -51 | 39 | yes | 7.4° | 2.50× | 6 / 6 | 1.39× | 0.00 | 40 | 31 (-69) / 7 (-67) | 1 (-94) / 0 (-36) | 1.00 | 4.2° |
| A10+B | 10° shims + re-hinge | 10° | 2.5 | 11.5 / 6.3 | 26.4 | 0.03 | -62 | 10 | yes | 5.1° | 2.66× | 8 / 8 | 1.39× | — | 0 | 24 (-76) / 2 (-72) | 0 (-95) / 1 (-35) | 1.00 | -4.9° |
| A15+B | 15° shims + re-hinge | 15° | 2.5 | 12.8 / 7.7 | 28.3 | 0.01 | -55 | 22 | yes | 8.0° | 3.21× | 8 / 8 | 1.39× | — | 50 | 22 (-78) / 0 (-74) | 0 (-95) / 0 (-36) | — | — |
| A20+B | 20° shims + re-hinge | 20° | 2.5 | 14.1 / 9.0 | 30.1 | 0.00 | -41 | 39 | yes | 10.0° | 3.75× | 8 / 8 | 1.39× | — | 943 | 19 (-81) / 1 (-73) | 0 (-95) / 0 (-36) | 1.00 | 8.1° |
- rung i = same joint angles as the recorded 30k policy, mounts moved (lower bound); rung ii = 20 settles + sweep; rung iii = open-loop scripted tripod (8 envs, in-session base); rung iv = 30k/20k heads replayed on the variant plant (one-sided; unchanged falls = policy mismatch, not a null result)
- the tibia contact channel reads tibia–ground load in this plant (colliders reach the floor), so hip/femur is the interference signal; tibia–tibia contact is covered kinematically by the rung i clearance sweep only
>>> ENTRY stop 2

## STOP 2 — 2026-09-02 18:20 — readings (waiting for the user's rung-(v) budget confirmation)
- **Same policy, no retraining, the variants remove most falls.** 30k head, step-onset scenario:
  base 74/100 falls → B 30, A10 21, A15 6, A20 7, A10+B 2, A15+B 0, A20+B 1. Slow-walk canary:
  base 19 → 0–2 on every variant. 20k head, step-onset: base 36 → B 17, A10 6, A15 3, A20 0,
  A10+B 1, A15+B 0, A20+B 0. Every remaining fall is still nose-down pitch (no roll/back falls
  appear), i.e. the variants shrink the basin without changing the failure mode.
- The forward scenario (0.30/0.475/0.65 holds, outside the 0–0.35 trained band) saturates on base
  (100/100 and 95/100 falls) but not on the variants (30k: 19–40; 20k: 0–6): the wider base lets
  the same gait carry speeds the current plant cannot survive.
- Measured walking polygon tip (30k slow, root proxy): base 6.7°, B 9.0, A10 10.0, A15 11.6, A20
  13.4, A10+B 12.2, A15+B 14.0, A20+B 15.5 — within 1.4° of the rung-(i) predictions, in the same
  order. The ordering of fall reduction follows the tip angle, with diminishing returns above ~12°
  (A15/A20/A10+B/A15+B/A20+B are within a few falls of each other at step onset).
- Costs visible so far: tracking ratio on the slow canary drops 0.42 → 0.37–0.39 (30k) and
  0.44 → 0.39–0.42 (20k) — the policy's stride was tuned for the old geometry; tripod score rises
  (0.58 → 0.62–0.66). Rung (i) hardware costs: lateral reach −10/−22/−39 mm per side at 10/15/20°;
  cos α thrust factor 0.98/0.97/0.94. Rung (iii) hip/femur contact in the open-loop sweep: A20+B
  943 N (adjacent-leg crossing at full throw), A10 192 N, A15+B 50 N, A20 40 N, others 0.
- Rung (v) budget: 8 arms (7 variants + a base reproduction on the same seed) × 5k iterations
  ≈ 2.2 h each + 3 evals ≈ 19–20 GPU-h serial (drop the base arm → ~17 h, the plan's figure).
  Control = the existing seed-3 anchor evaluated in this harness (3 evals, ~5 min).
>>> ENTRY stop 2 readings

## STOP 2 sign-off — 2026-09-02 18:2x
- user: "go ahead with rung v on all 8 arms" → `run_formation_arms.py` launched (control evals first, then
  per arm: 5k seed-3 formation → slow/fwd/step evals; state in `formation_state.json`, logs in
  `formation_logs/`).
>>> ENTRY stop 2 signoff

## RUNG (v) note — 2026-09-02 23:42 — reproduction noise floor
- The `base` arm (seed 3, identical 0–5k config and plant to the anchor 2026-08-31_03-42-16) came
  out at slow canary tripod 0.537 / completion 0.83 / tracking 0.426 / 12→17 falls vs the anchor's
  0.595 / 0.88 / 0.397 / 12 (run `2026-09-02_21-12-20/model_4999.pt`). Same-seed training is not
  bit-reproducible on this GPU stack, so **the run-to-run spread is ≈ 0.06 tripod, 0.05
  completion, 0.03 tracking, ~5 falls/100 on the slow canary**. Rung (v) variant deltas smaller
  than that are noise; the control row and the base row together bracket it.
>>> ENTRY rung v noise floor

## RUNG (v) interim — 2026-09-03 09:50 — variant 5k arms collapse in the eval; diagnostics
- Four arms trained (base, B, A10, A15; all plants verified in params/env.yaml; evals spawned the
  right plants per run_meta). Slow canary: control 0.595/0.88/12 falls, base 0.537/0.83/17, but
  B 0.256/0.31/69, A10 0.145/0.23/77, A15 0.243/0.46/54; step-onset 39 → 69/90/68 falls.
  Training told the opposite story: failure 0.29 (base) vs 0.13/0.13/0.29, tracking income 0.84
  vs 0.94/0.97/0.87, clock income 0.37 vs 0.54/0.59/0.41.
- Where they fall: A10/A15 within 2–6 s of the creep hold starting (after the 10 s stand); B
  spread over creep and low; base only late in the 0.35 hold. Fall class: nose-down.
- Not gait initiation from parked cams: cams keep cycling at 1.3–1.5 rad/s through the stand hold
  on every arm (A10 0.8), and the tripod phase relation at creep onset does not predict early
  falls (B 0.02 either way; A10 0.48 in-phase vs 0.33 anti-phase; A15 0.29 vs 0.25).
- Not the stand prologue alone: `creep20` (18 s of 0.25 m/s from reset, no stand) — B 43/100
  falls (median 8.8 s into the bout), base 12/100. The variant policies' sustained walking is
  weak; their training metrics hide it because (a) 57 % of training commands sit below the
  0.2 m/s clip (uniform 0–0.35), where the clock term pays pure stance quality — the wide plants
  stand quietly and earn it, base steps in place and does not; (b) walking bouts last ≤ 6 s
  (resampling 6 s), so a gait that degrades after ~8 s never fails in training. Same signature as
  the gait-formation-v2 note in crab_hex_env_cfg.py (C5/C9: training gates passed, ~0.3 eval
  completion, "never trained against a sustained hold").
- RSI: possible contributor, not indicated. The P0-null bank is golden-plant geometry (foot
  placement rotates about the mount on the variants, toe height invariant) and seeds 20 % of
  resets; the initiation and phase tests above do not show an RSI-shaped signature. Clean tests,
  2.4 h each, for the user to choose after the matrix: (1) RSI-off arm on B; (2) B arm with a
  bank re-harvested on the B plant from the transferred 30k head (rung iv: 2 slow falls).
- Practical alternative for a deployable policy: fine-tune the 30k head on the chosen plant
  (rung iv shows it already walks every variant better than any 5k formation arm).
>>> ENTRY rung v interim diagnostics

## RUNG (v) PARTIAL TABLE at PAUSE — 5 of 8 arms trained (base, B, A10, A15, A20); A10+B stopped at iteration 3219; A15+B, A20+B not started
| config | hardware | rung i walk tip p50/p10 | rung i lead tip p50 | rung ii standing lead tip Δ | rung iii survivors @0.3/@0.5 | rung iii hip/femur N | rung iv 30k falls fwd/step (Δ) | rung iv 20k falls fwd/step (Δ) | rung v status | smoke fail@2k / coll@2k | rung v slow tripod (ratio) | rung v slow completion (ratio) | rung v slow tracking (ratio) | rung v slow falls Δ | rung v fwd falls (Δ) | rung v fwd pitch_max p50 (ratio) | rung v step falls (ratio) | rung v step pitch-fwd share (ratio) | rung v step prefall tip p25 Δ | rung v walk tip p50 | user preference |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| control (anchor, base plant) | none | — | — | — | — | — | — | — | ok | — | 0.595 | 0.88 | 0.397 | 12 | 100 | 0.49 | 39/100 | 0.92 | -0.7° | 9.0° | — |
| base | none | 6.9 / 1.7 | 18.1 | 0.0° | 6 / 4 | 0 | 100 / 74 | 95 / 36 | ok | 0.40 / 0.000 | 0.537 (0.90×) | 0.83 (0.94×) | 0.426 (1.07×) | 17 (+5) | 99 (-1) | 0.49 (1.00×) | 69/100 (1.77×) | 0.87 (0.94×) | -0.4° (+0.4) | 8.5° | — |
| B | re-hinge 3 in | 8.8 / 3.7 | 22.0 | 2.0° | 8 / 8 | 0 | 40 (-60) / 30 (-44) | 6 (-89) / 17 (-19) | ok | 0.08 / 0.000 | 0.256 (0.43×) | 0.31 (0.35×) | 0.534 (1.35×) | 69 (+57) | 100 (+0) | 0.49 (1.00×) | 69/100 (1.77×) | 0.96 (1.04×) | 0.2° (+0.9) | 7.6° | re-hinge |
| A10 | 10° shims | 9.5 / 4.4 | 22.8 | 3.2° | 8 / 5 | 192 | 33 (-67) / 21 (-53) | 0 (-95) / 6 (-30) | ok | 0.13 / 0.000 | 0.145 (0.24×) | 0.23 (0.26×) | 0.685 (1.73×) | 77 (+65) | 100 (+0) | 0.49 (1.01×) | 90/100 (2.31×) | 1.00 (1.08×) | -3.6° (-2.8) | 5.9° | splay only ✓ |
| A15 | 15° shims | 10.8 / 5.7 | 24.9 | 5.9° | 8 / 8 | 0 | 28 (-72) / 6 (-68) | 1 (-94) / 3 (-33) | ok | 0.15 / 0.000 | 0.243 (0.41×) | 0.46 (0.52×) | 0.525 (1.32×) | 54 (+42) | 100 (+0) | 0.49 (1.00×) | 68/100 (1.74×) | 0.90 (0.97×) | -1.5° (-0.8) | 6.8° | splay only ✓ |
| A20 | 20° shims | 12.1 / 7.0 | 26.7 | 7.4° | 6 / 6 | 40 | 31 (-69) / 7 (-67) | 1 (-94) / 0 (-36) | ok | 0.17 / 0.000 | 0.319 (0.54×) | 0.76 (0.86×) | 0.707 (1.78×) | 24 (+12) | 100 (+0) | 0.49 (1.01×) | 66/100 (1.69×) | 1.00 (1.08×) | -0.2° (+0.6) | 6.9° | splay only ✓ |
| A10+B | 10° shims + re-hinge | 11.5 / 6.3 | 26.4 | 5.1° | 8 / 8 | 0 | 24 (-76) / 2 (-72) | 0 (-95) / 1 (-35) | pending | — / — | — (—) | — (—) | — (—) | — | — | — (—) | — | — (—) | — | — | shims + re-hinge |
| A15+B | 15° shims + re-hinge | 12.8 / 7.7 | 28.3 | 8.0° | 8 / 8 | 50 | 22 (-78) / 0 (-74) | 0 (-95) / 0 (-36) | pending | — / — | — (—) | — (—) | — (—) | — | — | — (—) | — | — (—) | — | — | shims + re-hinge |
| A20+B | 20° shims + re-hinge | 14.1 / 9.0 | 30.1 | 10.0° | 8 / 8 | 943 | 19 (-81) / 1 (-73) | 0 (-95) / 0 (-36) | pending | — / — | — (—) | — (—) | — (—) | — | — | — (—) | — | — (—) | — | — | shims + re-hinge |
- control = lineage anchor 2026-08-31_03-42-16/model_4999 (seed 3, 5k) on the base plant; variant arms = fresh seed-3 5k formation on their plant with the identical 0–5k config; 'base' arm = same on the golden plant (reproduction sample)
- ratios are variant/control; 0.85× is the plan's reference line on the slow canary; falls are per 100 episodes; 5k is a formation snapshot, not a lineage result
- rung i–iv columns carried over from the STOP 2 table (stop2_table.csv); hardware cost columns there: lateral reach −10/−22/−39 mm per side at 10/15/20°, cos α 0.98/0.97/0.94, sweep overlap unchanged
>>> ENTRY rung v partial table (paused)

## CAMPAIGN PAUSED — 2026-09-03 11:40 (user decision)
- user: "pause this plan for now and keep the unmodified robot as the default. I'd like to first
  explore alterations to the training before further testing on changes to the geometry."
- Actions: formation driver + the A10+B arm stopped (iteration 3219, no 5k checkpoint); GPU freed;
  heartbeat stopped. Default plant unchanged throughout: the golden `assets/crab_simple.usda` is
  byte-pinned to `generate()` and is what every task loads unless `KRABBY_HEX_USD_PATH` is set;
  the 7 variant USDs live only under `assets/variants/` and are opt-in.
- State preserved for resume: `formation_state.json` (5 arms with checkpoints + evals; control
  evals), `formation_logs/`, eval runs under `logs/rsl_rl/gait_eval/leg_mount_morphology/`,
  `stop2_table.csv`, the partial rung (v) table above. Resume = re-run `run_formation_arms.py`
  (skips arms with checkpoints; A10+B retrains from scratch) then `assemble_stop3.py`.
- Standing conclusions at pause: geometry gains are real for the *existing* policies (rung iv:
  same 30k head, step-onset falls 74 → 0–30); the 0–5k formation config does not form a robust
  gait on the wider plants (rung v interim: standing is stable and lucrative, walking bouts ≤ 6 s,
  57 % stop commands) — the training-side questions the user now wants to explore first.
>>> ENTRY campaign paused

## DECISION TABLE — morphology x training-side fixes (P1: formation + walking slots; P2: + 40-s episodes, 10-s holds; stage 2 on recal2b2w with the promotion equilibrium held)
| config | hardware | rung iv 30k falls fwd/step (Δ) | rung v slow tripod / compl / track (old config) | rung v step falls (ratio) | P1 status | P1 slow tripod / compl / track (ratio to golden) | P1 slow falls | P1 step falls (ratio) | P1 obst recal2b2w compl (falls) | P1 reach_obst | P1 field_frac | P1 cov[3] | P1 fail hazard flat / obst (/1k) | P1 terrain level | P1 step prefall tip p25 | P2s1 status | P2s1 slow tripod / compl / track (ratio to golden) | P2s1 slow falls | P2s1 step falls (ratio) | P2s1 obst recal2b2w compl (falls) | P2s1 reach_obst | P2s1 field_frac | P2s1 cov[3] | P2s1 fail hazard flat / obst (/1k) | P2s1 terrain level | P2s1 step prefall tip p25 | P2s2 status | P2s2 slow tripod / compl / track (ratio to golden) | P2s2 slow falls | P2s2 step falls (ratio) | P2s2 obst recal2b2w compl (falls) | P2s2 reach_obst | P2s2 field_frac | P2s2 cov[3] | P2s2 fail hazard flat / obst (/1k) | P2s2 terrain level | P2s2 step prefall tip p25 | seed-2 P2s2 slow / step / obst | user preference |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| base | none | 100 / 74 | 0.537 (0.90×) / 0.83 (0.94×) / 0.426 (1.07×) | 69/100 (1.77×) | ok | 0.444 / 0.46 / 0.596 (—, —, —) | 54/100 | 71/100 (—) | 0.39 (61/100) | 0.635 | 0.306 | 0.008 | 0.278 / 0.423 | 0.97 | -2.8° | ok | 0.418 / 0.28 / 0.696 (—, —, —) | 72/100 | 91/100 (—) | 0.13 (87/100) | 0.532 | 0.217 | 0.073 | 0.812 / 1.406 | 0.97 | -3.1° | ok | 0.520 / 0.68 / 0.625 (—, —, —) | 32/100 | 67/100 (—) | 0.31 (69/100) | 0.741 | 0.376 | 0.251 | 0.330 / 0.795 | 5.83 | -2.3° | — | — |
| B | re-hinge 3 in | 40 (-60) / 30 (-44) | 0.256 (0.43×) / 0.31 (0.35×) / 0.534 (1.35×) | 69/100 (1.77×) | ok | 0.377 / 0.82 / 0.685 (0.85×, 1.78×, 1.15×) | 18/100 | 46/100 (0.65×) | 0.54 (46/100) | 0.684 | 0.305 | 0.021 | 0.128 / 0.560 (golden 0.278 / 0.423) | 0.97 | -1.7° | ok | 0.382 / 0.79 / 0.642 (0.91×, 2.82×, 0.92×) | 21/100 | 32/100 (0.35×) | 0.67 (33/100) | 0.893 | 0.563 | 0.369 | 0.049 / 0.166 (golden 0.812 / 1.406) | 0.97 | -2.1° | ok | 0.455 / 0.90 / 0.622 (0.87×, 1.32×, 0.99×) | 10/100 | 14/100 (0.21×) | 0.83 (17/100) | 0.927 | 0.612 | 0.532 | 0.039 / 0.173 (golden 0.330 / 0.795) | 5.75 | 1.2° | 0.96 / 28 / 0.69 | re-hinge |
| A10 | 10° shims | 33 (-67) / 21 (-53) | 0.145 (0.24×) / 0.23 (0.26×) / 0.685 (1.73×) | 90/100 (2.31×) | ok | 0.370 / 0.57 / 0.685 (0.83×, 1.24×, 1.15×) | 43/100 | 84/100 (1.18×) | 0.15 (85/100) | 0.582 | 0.279 | 0.009 | 0.414 / 0.995 (golden 0.278 / 0.423) | 0.97 | -1.9° | pending | — | — | — | — | — | — | — | — | — | — | pending | — | — | — | — | — | — | — | — | — | — | — | splay only ✓ |
| A15 | 15° shims | 28 (-72) / 6 (-68) | 0.243 (0.41×) / 0.46 (0.52×) / 0.525 (1.32×) | 68/100 (1.74×) | ok | 0.494 / 0.90 / 0.648 (1.11×, 1.96×, 1.09×) | 10/100 | 58/100 (0.82×) | 0.49 (51/100) | 0.744 | 0.343 | 0.023 | 0.071 / 0.508 (golden 0.278 / 0.423) | 0.97 | 1.3° | ok | 0.494 / 0.78 / 0.595 (1.18×, 2.79×, 0.86×) | 22/100 | 55/100 (0.60×) | 0.48 (52/100) | 0.849 | 0.534 | 0.320 | 0.082 / 0.271 (golden 0.812 / 1.406) | 0.97 | 4.4° | ok | 0.540 / 1.00 / 0.584 (1.04×, 1.47×, 0.93×) | 0/100 | 26/100 (0.39×) | 0.67 (33/100) | 0.858 | 0.525 | 0.401 | 0.079 / 0.388 (golden 0.330 / 0.795) | 6.03 | -0.9° | — | splay only ✓ |
| A20 | 20° shims | 31 (-69) / 7 (-67) | 0.319 (0.54×) / 0.76 (0.86×) / 0.707 (1.78×) | 66/100 (1.69×) | ok | 0.466 / 0.50 / 0.617 (1.05×, 1.09×, 1.03×) | 50/100 | 67/100 (0.94×) | 0.34 (66/100) | 0.732 | 0.354 | 0.023 | 0.141 / 0.493 (golden 0.278 / 0.423) | 0.97 | 1.7° | pending | — | — | — | — | — | — | — | — | — | — | pending | — | — | — | — | — | — | — | — | — | — | — | splay only ✓ |
| A10+B | 10° shims + re-hinge | 24 (-76) / 2 (-72) | — (—) / — (—) / — (—) | — | ok | 0.399 / 0.44 / 0.615 (0.90×, 0.96×, 1.03×) | 56/100 | 79/100 (1.11×) | 0.29 (71/100) | 0.624 | 0.298 | 0.027 | 0.202 / 0.877 (golden 0.278 / 0.423) | 0.97 | 0.9° | pending | — | — | — | — | — | — | — | — | — | — | pending | — | — | — | — | — | — | — | — | — | — | — | shims + re-hinge |
| A15+B | 15° shims + re-hinge | 22 (-78) / 0 (-74) | — (—) / — (—) / — (—) | — | ok | 0.564 / 1.00 / 0.753 (1.27×, 2.17×, 1.26×) | 0/100 | 5/100 (0.07×) | 0.91 (9/100) | 0.815 | 0.423 | 0.086 | 0.031 / 0.061 (golden 0.278 / 0.423) | 0.97 | 6.0° | ok | 0.539 / 0.86 / 0.676 (1.29×, 3.07×, 0.97×) | 14/100 | 31/100 (0.34×) | 0.68 (32/100) | 0.860 | 0.522 | 0.326 | 0.128 / 0.338 (golden 0.812 / 1.406) | 0.97 | -1.4° | ok | 0.589 / 0.95 / 0.622 (1.13×, 1.40×, 0.99×) | 5/100 | 23/100 (0.34×) | 0.73 (27/100) | 0.915 | 0.599 | 0.487 | 0.064 / 0.320 (golden 0.330 / 0.795) | 6.03 | 1.1° | 0.96 / 8 / 0.95 | shims + re-hinge |
| A20+B | 20° shims + re-hinge | 19 (-81) / 1 (-73) | — (—) / — (—) / — (—) | — | ok | 0.562 / 0.98 / 0.686 (1.27×, 2.13×, 1.15×) | 2/100 | 46/100 (0.65×) | 0.51 (49/100) | 0.773 | 0.428 | 0.064 | 0.145 / 0.505 (golden 0.278 / 0.423) | 0.97 | -0.4° | pending | — | — | — | — | — | — | — | — | — | — | pending | — | — | — | — | — | — | — | — | — | — | — | shims + re-hinge |
- every arm from scratch, seed 3; ratios are variant / the same protocol's golden arm; 0.85× is the canary reference line; rung-v noise floor ≈ 0.06 tripod / 0.05 completion / 5 falls per 100
- rung iv / rung v columns carried from the morphology campaign (stop2_table.csv, stop3_decision_table.csv; rung v = old formation config)
- fail hazards are per 1000 env steps (episode-length neutral); P2 stage-2 terrain level should sit near the lineage's 4–6 or the promotion scaling is re-derived
- source: /home/nickmagus/krabby/krabby-research/sim_fine_tuning/2026-09-04_1105_morph_x_exposure
>>> ENTRY morph x exposure decision table
