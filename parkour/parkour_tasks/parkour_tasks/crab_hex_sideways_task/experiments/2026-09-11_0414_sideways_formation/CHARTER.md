# Sideways-formation campaign — CHARTER and verification record

Approved plan: `/home/nickmagus/.claude/plans/shimmering-beaming-ullman.md` (user, 2026-09-11),
"`crab_hex_sideways_task` — the sideways-walk sibling of the forward-walk task (first plan)".
Successor to nothing directly — first training campaign for the new task package. The sideways
task travels along the robot's body **+y** axis ("Shape B"); the forward task's baked lineage
(`crab_hex_forward_task/policy/manifest.yaml`, head `3a_student/model_24995.pt`) is the reference
this campaign is measured against, not a checkpoint it resumes from (phase 1a trains from scratch).

## Question

Does the sideways-walk task, built as a thin subclass of the forward task's cfgs with the travel
axis rotated to body +y and the mirror-symmetry loss re-derived for the fore-aft reflection plane,
form a usable tripod gait through the forward schedule of record (1a formation -> 2a/2b/2c
elements), with RSI off?

## Hypotheses

- **H1 — formation without RSI.** The clock-driven tripod reward alone (no reference-state
  initialization; the RSI bank of record is a forward gait and would teleport a sideways walker
  back to yaw 0) forms a tripod gait on phase 1a within the flat canary's written gate.
- **H2 — fore-aft mirror gives balanced A/B duty.** The fore-aft mirror-symmetry loss (front<->rear
  legs, ML/MR fixed, coefficient 0.5, the same recipe as the forward lineage's left-right mirror)
  keeps the two tripod sets' duty factors close (|A - B| <= 0.10, both in [0.30, 0.55]) even though
  it maps each tripod set to itself rather than swapping them.
- **H3 — tracking at the 0.25 hold.** Achieved-vs-commanded velocity tracking (travel-frame) at the
  vx 0.25 hold reaches ratio >= 0.5; the vx 0.35 hold is expected to saturate (estimated sideways
  speed ceiling 0.21-0.31 m/s against a 0.21-0.38 m/s radial stroke speed at the 0.5 rev/s clock).

## Held fixed

- Plant: A15+B (`assets/crab.usda`), single plant for the whole campaign.
- Presets: the forward schedule of record's windows (`crab_hex_phases.py` via
  `crab_hex_side_phases.py`), minus RSI (`KRABBY_RSI_FRAC` / `KRABBY_RSI_BANK` dropped from every
  window) and plus `KRABBY_STUDENT_MDP=1` on the phase-3 windows.
- PPO and distillation hyper-parameters, seed 3, 256 envs (rl phases) / 192 envs (distill phases),
  5000 iterations per phase.
- `KRABBY_SYM_LOSS_COEF` 0.5 (mirror-symmetry loss weight, unchanged from the forward recipe).
- Terrain: recal2b2w geometry, promotion 0.225:0.125, domain randomization -- the forward regime,
  unchanged (Caveat 3: the corridor geometry does not suit the sideways footprint as well as it
  suits the forward one; obstacle numbers compare to forward only as a caveat).
- Harness: `crab_hex_forward_task/scripts/eval_crab_hex_gait.py` (one harness serves both tasks,
  `travel_yaw` plumbed through); manifest shape frozen once evaluated (`experiments/eval/
  scenarios_side.yaml`).
- Acceptance: the Isaac config-delta test
  (`tests/integration/test_crab_hex_sideways_configs.py::test_env_delta_is_exactly_the_travel_axis`)
  passing is the acceptance criterion that nothing besides the travel axis changed; see Appendix A
  of the plan for the exact expected diff.

## Delta list (Parts A/B of the plan — everything the sideways task changes vs. forward)

| Site | Forward | Sideways | Notes |
|---|---|---|---|
| Heading offset | none | `commands.base_velocity.heading_offset = -pi/2` | `CrabHexSideVelocityCommandCfg`, the only place the field exists |
| Telemetry frame | body frame | rotated by `travel_yaw` | `metrics.py` fall-direction / orientation / support-polygon, `report.py` tracking / stride |
| `delta_yaw` / `delta_next_yaw` sites | `target - yaw` (no wrap) | `wrap_to_pi(target - yaw - travel_yaw)` | `mdp/observations.py`, 3 call sites (test-pinned) |
| Height-scan offset | `pos (0.6, 0.0, 20.0)` | `pos (0.0, 0.6, 20.0)` | size `[2.4, 3.4]` / resolution 0.3 / ordering KEPT (108 rays, obs 1149 unchanged) |
| Spawn rotation | identity `(1,0,0,0)` | `(0.7071, 0, 0, -0.7071)` (yaw -pi/2) | `scene.robot.init_state.rot` |
| Reward velocity sites | body-x | travel-frame | `penalty_lin_vel_y_l2`, `reward_obstacle_clearance`, `reward_forward_progress_along_command`, `penalty_tracking_error_l1`, `RewardStrideLength` |
| `track_lin_vel_travel_exp` | `track_lin_vel_xy_exp` (func name in the reward table) | `track_lin_vel_travel_exp` (new func, same term name/params) | `CrabHexSideFlatWalkRewardsCfg` re-declares the term |
| Fore-aft mirror | left-right (`crab_hex_mirror.py`) | front-rear, ML/MR fixed | `crab_hex_side_mirror.py`, `LEG_SWAP`, `HEAD_SIGNS`, `scan_axis="x"` |
| Runner experiment names | `crab_hex_{flat_walk,teacher,student}` | `crab_hex_side_{flat_walk,teacher,student}` | `agents/rsl_rl_ppo_cfg.py`, log dirs |
| `KRABBY_STUDENT_MDP` | n/a for 1a/2a-2c | exported on the sideways 3a/3b presets | forward student cfg gates on `is_student_phase()` |
| Harness `travel_yaw` | 0.0 (identity path) | `pi/2` | `eval_crab_hex_gait.py`, `run_meta["travel_yaw"]`, all three metric-call sites |

## Driver invocation

```bash
SIDE=$R/parkour/parkour_tasks/parkour_tasks/crab_hex_sideways_task
CAMP=$SIDE/experiments/2026-09-11_0414_sideways_formation
"$SIDE/experiments/tools/launch_phases.sh" --campaign-dir "$CAMP" --plant A15+B --phases 1a,2a,2b,2c --seed 3
"$SIDE/experiments/tools/heartbeat_phases.sh" <unit> "$CAMP" 1800     # run_in_background; the launcher prints the unit name
```

`--phases` is always passed explicitly (the driver's default is `1a,2a,2b,2c,3a,3b`; this campaign
stops after 2c — phase 3 needs the camera-placement decision first). The driver ends in
`state.phase = "done"` after the 2c evals: that is the decision point, recorded in `state.json`,
`notify.log` and `CHANGELOG.md`. Halts: `dead_plant` (from-scratch smoke gate at iteration 2000),
`gate_1a` (the written formation gate below), `nan`, `infra` — each notifies; relaunch with
`--continue` (resumes an interrupted run, or overrides a `gate_1a` halt using the already-trained
checkpoint without retraining) or `--redo` after a fix. For a hard stop after formation regardless
of the gate, add `--pause-after 1a`.

## The written 1a gate (also enforced by the driver after the 1a evals)

On `slow__side`: **walking-hold tripod >= 0.40 AND completion >= 0.70** (the enforced pair — a
failed formation must not burn the 2a-2c hours unattended). The walking-hold tripod is the
minimum, over the holds the harness scores for tracking (|cmd| > 0.2 m/s: creep 0.25 and low
0.35; the cmd-0 stand hold is excluded), of the per-hold tripod median
(`tripod_score_by_hold`; driver `walking_tripod()` / `gate_1a_verdict()`). Watched but not enforced: tracking ratio
median >= 0.50 at the 0.25 hold (no forward 1a tracking number is on record for comparison; the
0.35 hold is expected to saturate), |duty A - duty B| <= 0.10 with both in [0.30, 0.55], cam
`shaft_mean_abs_vel` median <= 0.3 rad/s (parked, per Caveat 2). Reference: forward era-B 1a
0.86 / 14 falls / episode tripod 0.539 (`policy/manifest.yaml`), i.e. **walking-hold tripod
0.595** (creep 0.633 / low 0.595; `gait_eval/a15b_lineage/windows_w0_001_A15pB/slow__A15pB`) in
the gate's own units. 0.40 is the era-A pre-mirror-loss tripod level (0.401 -> 0.517 on the
walking-only `flat_walk_forward` schedule, 10 episodes:
`crab_hex_forward_task/experiments/2026-08-13_0035_mirror_symmetry/RESULTS.md`) — a floor, not a
like-for-like bar. The driver halts with status `gate_1a` both when the pair fails AND when the
`slow__side` canary eval produced no usable number (three failed attempts, a schedule without a
walking hold, or a walking hold on which no episode produced a scorable gait window): an ungated
head never starts 2a unattended; carrying on is an explicit `--continue`, which rewrites the 1a
record to `status ok` + `gate_1a_override` so a later relaunch never retrains 1a by accident.

**Why the walking holds (revised 2026-09-11, after the first 1a run).** The gate was first written
as the harness's episode `tripod_score` median, which is a steady-step-weighted mean over all
three holds of the schedule with no command gate (`gait_eval/report.py`), stand hold included.
The first sideways 1a head truly stands at cmd 0 (achieved 0.006 m/s, stand-hold tripod 0.001)
and scored **0.355** on the episode number while its walking holds scored **0.537 / 0.533**; the
forward 1a reference's 0.539 contains **0.453 earned on its stand hold by stepping through the
stop** (0.066 m/s), its walking holds being 0.633 / 0.595. Under the episode definition a head
that stands still needs walking tripod >= 0.60 to pass — above the forward reference itself —
and the 0.40 floor was calibrated on a walking-only schedule (era-A 0.401 -> 0.517 on
`flat_walk_forward`, `2026-08-13_0035_mirror_symmetry/RESULTS.md`). The driver does not re-score
the first run's record; the user overrode it with `--continue` (see the Log).

## Decision table (filled by `decision_table.py`; forward cells from `crab_hex_forward_task/policy/manifest.yaml`)

Cell format: `completion / falls / episode tripod (walk <min over walking holds — the gated number>)`
for the sideways rows; the forward cells are episode tripods from the manifest, except the 1a flat
cell which also carries the forward walking-hold value (0.595) so the gated statistic has a
like-for-like reference.

Tracking has no forward 1a-2c reference on record (the 0.65 figure on record is the 3a student).

| phase | flat canary side | fwd | step onset side | fwd | obstacles side | fwd | tracking | duty A / B | falls (travel frame) | cam | slip |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1a | | 0.86 / 14 / 0.539 | | 0.69 / 31 | | 0.68 / 32 | | | | | |
| 2a | | 0.92 / 8 / 0.513 | | 0.74 / 26 | | 0.56 / 44 | | | | | |
| 2b | | 0.85 / 15 / 0.507 | | 0.63 / 37 | | 0.58 / 42 | | | | | |
| 2c | | 0.81 / 19 / 0.500 | | 0.68 / 32 | | 0.66 / 34 | | | | | |

Decisions the user takes at the 2c row: continue to 3a (camera placement first), iterate on the
policy (a follow-on plan), or bake.

## Fallback — sideways RSI bank

If H1 fails (1a does not clear the written gate): `harvest_side_rsi_bank.py` in this directory
(copied from `crab_hex_forward_task/experiments/2026-08-22_1200_gait_formation_v2/
harvest_rsi_bank.py`, `moving` filter re-pointed to `root_lin_vel_b[:, 1]` — travel speed — instead
of `[:, 0]`) harvests a sideways-native RSI bank from the 1a checkpoint's own rollouts (the bank
records `root_quat_w` verbatim, so the -pi/2 spawn yaw is preserved — no teleport-to-yaw-0 failure
mode). Run:

```bash
KRABBY_SIDE_PHASE=1a $PY harvest_side_rsi_bank.py --headless \
  --task Isaac-Crab-Hex-Side-Walk-Play-v0 --checkpoint <1a head> \
  --num_envs 16 --steps 800 --out rsi_bank_side_1a.npz
```

then add a `1a_rsi` preset (= 1a + `KRABBY_RSI_FRAC=0.2`, `KRABBY_RSI_BANK=<bank>`) to
`crab_hex_side_phases.py` and relaunch `--phases 1a_rsi,2a,2b,2c --redo`. Not run by default; this
is the recipe, held in reserve.

## Caveats and risks

1. Formation without RSI: per the gait-formation-v2 record the clock alone formed the forward
   tripod on 1 of 4 seeds and RSI flipped the failing seed — the 1a gate + fallback above exist
   because of this precedent.
2. Cam parking: outer legs hold perpendicular at shaft ~52.8 deg (residual sweep +10/-40 deg)
   through a velocity channel (stiffness 0, damping 1); drift or free spinning is possible — watch
   `shaft_spin` / `action_metrics.camshaft` (the decision table's "cam" column).
3. Terrain regime for 2a-2c: cross-corridor half-extent drops to 0.39-0.53 m (corridors over-wide
   for the sideways footprint), along-course extent rises to 1.97-2.38 m (front/rear rows cannot
   fit one 0.6-1.0 m stepping stone; gaps are crossed by the wide side). Kept unchanged by decision
   (Held fixed, above); obstacle numbers compare to forward only as a caveat.
4. Speed band top 0.35 m/s is above the estimated sideways ceiling (0.21-0.31 m/s at the 0.5 rev/s
   clock; the radial stroke speed is 0.21-0.38 m/s) — the 0.35 hold saturates; read tracking at the
   0.25 hold (H3).
5. Knee drive torque is optimistic in sim (~36 N.m rated vs ~206 N.m static moment; sim effort
   400 N.m); a radial stride drives the knee under load — sim-to-real, not a training blocker.
6. Camera placeholder on the +y face is where the hip plates hinge; hardware decision deferred to
   the 2c decision point (phase 3 is not in this campaign's `--phases`).
7. Manifest shape: the forward 1a-2c reference numbers above were measured with the
   pre-2026-09-09 manifest (an `env:` block present on the equivalent row); `slow__side` carries
   none — +-0.04 completion / +-4 falls is the cross-shape resolution (see the header note in
   `scenarios_side.yaml`).
8. The scan grid's coverage swaps roles (longer look-ahead, narrower cross-width) with the size
   kept unchanged — accepted (Part B3 of the plan).
9. Forward guards: the recorded-window identity test needs the lineage run dirs (present);
   `mdp_pins.yaml` does not cover spawn rotation / scanner offset / symmetry string — hence the
   literal-guard unit test and the pre/post `cmp` in Verification steps 0 and 3 of the plan.
10. **Phase-3 blocker found in verification (2026-09-11, plan step 9, camera check 4b):** the
    student scene's depth camera is placed correctly in the config (offset `pos (0, 0.625, 0.10)`,
    `rot = q_z(+90 deg) * forward rot`) but the shared prestartup event `random_camera_position`
    (`parkour_isaaclab/envs/mdp/events.py`) passes the parent-relative `offset.rot` to
    `RayCasterCamera.set_world_poses(..., convention="ros")` as a WORLD orientation. With the
    forward identity spawn the two frames coincide (no-op, forward unaffected); under the
    sideways spawn (yaw -pi/2) the camera ends up rotated by an extra +90 deg about z — measured
    optical axis in the body frame `(-0.94, -0.01, -0.34)` instead of the intended
    `(0, +0.94, -0.34)`, position correct (`outputs/sideways_plan/check_side_camera.log`).
    Zero impact on this campaign (phases 1a-2c never build the camera). Before any 3a/3b run
    the event must compose the offset with the body's world orientation (a forward-neutral fix,
    since the forward body is at identity when the event fires) and the 4b check re-run; it is
    NOT worked around in the sideways package.

## Log

- **2026-09-11 10:57** launched `--plant A15+B --phases 1a,2a,2b,2c --seed 3` (unit
  `phases-2026-09-11_0414_sideways_formation-1789138664`).
- **2026-09-11 13:37 halt `gate_1a`** on `seed3_1a_001_A15pB` (`model_4999.pt`, 2.57 h): episode
  tripod 0.355 < 0.40 with completion 0.93, 7 falls, tracking 0.634. Per hold: stand 0.001 /
  creep 0.537 / low 0.533. Step 0.77 / 23 falls; obstacles 0.78 / 22 (forward 1a: 0.69 / 31,
  0.68 / 32). Analysis (workflow `wf_d7b3a538-afb`: 5 readers, 6 key claims x 2 adversarial
  lenses; session brief `side_1a_brief.md`):
  the gate compared a true stand against the forward head's stepping-through-the-stop; on the
  walking holds the gait is an alternating tripod (cross-set contact correlation -0.83), duty A/B
  inside the watched band, front/rear duty within 0.04, speed ceiling ~0.19 m/s at both walking
  commands. Real deficits vs forward, carried as **watched items through 2a-2c**: walking-hold
  tripod -0.06 to -0.10; tippy-tap 0.37 vs 0.23 on walking steps; outer-foot slip path 2-4x
  forward (propulsion on ML/MR); cams not parked in EITHER task (1.79 vs 1.54 rad/s, FR/RR
  shaft drift); the 7 falls are cross-course body-pitch collapses (4 nose-down / 3 tail-down,
  3 of them during the stand hold while drifting); 12/100 stand holds drift > 0.5 m.
  Clock income still rising ~+0.04/1k at 5k, above forward's, with no known conversion to
  harness tripod.
- **2026-09-11 (user decision, after a GUI look at the gait): fix the gate to the walking-hold
  tripod, then `--continue`** — 2a-2c resume from `model_4999.pt` unretrained; the override
  rewrites the 1a record to `status ok` + `gate_1a_override` (CHANGELOG row + REPORT block).

- **2026-09-11 22:08 campaign complete (seed 3)** — all four phases `ok`, 11.6 h wall; heads in `heads.json`
  (1a `crab_hex_side_flat_walk/2026-09-11_10-57-55/model_4999.pt`, 2a `crab_hex_side_teacher/2026-09-11_14-50-01/model_9998.pt`,
  2b `.../2026-09-11_17-19-40/model_14997.pt`, 2c `.../2026-09-11_19-43-19/model_19996.pt`). 60-s videos of the 2c head
  at four terrain tiers in `videos/` (gitignored): fixed command 0.30 m/s down the travel axis, `Side-Walk-Play`,
  flat / recal2b2w 0.05-0.20 / 0.20-0.45 / 0.45-0.70.
- **2026-09-12 decision-point analysis** (5 readers, 8 key claims x 2 adversarial lenses; session brief
  `side_2c_decision_brief.md`). Per-hold walking statistics, paired episode bootstraps (n=100): resolution is
  +-0.06..0.10 completion, +-6..10 falls, +-0.016..0.056 walking tripod, so every flat-canary difference among
  2a/2b/2c is inside noise. **2a** is strongest on measured robustness (obst 0.94/6, resolved vs 1a p=0.002 and vs 2c
  p=0.023; gated flat tripod 0.548; last head inside the H2 duty band; lowest cam 1.618) but leads only two of three
  evals -- 1a leads step onset (gated 0.416). **2c is the campaign's only resolved regression** (step 0.63/37 vs
  0.82/18 for 2a and 2b; paired -0.19, McNemar p=0.004; obst -0.12 vs 2a). The drop is NOT a clock-tracking collapse
  (per-step clock income flat across the anneal), NOT a harder curriculum (every exposure channel flat-to-easier while
  flat-tile failures rose 32%), NOT a timing loss (tripod / cadence / alternation unchanged): it is a survival loss,
  158/160 falls cross-course roll, support margin worse. Confounded by window position, and forward keeps RSI 0.2 in
  every window where sideways has none. Largest single ambiguity: the 2c HEAD may be a low draw (its last-100-iteration
  `crab_failure` 0.302 vs the window mean 0.246, ~1.7 block-sd) -- a mid-2c checkpoint re-eval would settle it in ~10 min.
  **H1 CONFIRMED** (formation without RSI; the stand is controlled -- drift p50 ~0.10 m vs forward 0.6-1.2 m).
  **H2 partial**: |A-B| <= 0.10 at every phase and hold, but duty B leaves [0.30, 0.55] at the 0.35 hold from 2b
  (0.566, 0.595) -- the fore-aft mirror constrains front<->rear (|FL-RL| <= 0.019 vs forward 0.12-0.24), not A<->B.
  **H3 CONFIRMED, improving** (creep tracking 0.736 -> 0.787); the speed ceiling is 0.19-0.20 m/s, below caveat 4's
  0.21-0.31 estimate, and no window moved it. Watched items: cam FAILS in both tasks (dither, one-direction ~0.05;
  FR/RR free-spin 0.45/0.41 = +-3.3-3.6 rev/episode, mirror-forced by the CamShaft -1 sign); 0.35-hold slip +0.031 with
  outer-foot skate 2.5-4x the middle legs and 2-3x forward; tippy-tap ~1.5x forward, unchanged; knee torque at the
  0.35 hold worsening 37 -> 76 N.m median with 31% of samples at the 400 N.m sim ceiling (caveat 5).
  Phase-3 note: the caveat-10 camera bug is ALSO live in the shipped forward 3a student (`rot_noise_range` on the same
  startup event), so the fix must cover the noise branch and be proven forward-neutral; and the +y placeholder sits
  inside the middle hip plate's swept volume, which sim cannot see.
- **2026-09-12 USER DECISION: a second seed, 1a -> 2c, progressing automatically.** Seed 2 from scratch into its own
  `state.json` `keys.seed2` bucket; seed-3 records untouched. The revised 1a gate stays armed: a formation failure on a
  fresh seed is the evidence this run exists to produce (CHARTER caveat 1: forward clock-only formation formed on 1 of
  4 seeds).
