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

On `slow__side`: **tripod median >= 0.40 AND completion >= 0.70** (the enforced pair — a failed
formation must not burn the 2a-2c hours unattended). Watched but not enforced: tracking ratio
median >= 0.50 at the 0.25 hold (no forward 1a tracking number is on record for comparison; the
0.35 hold is expected to saturate), |duty A - duty B| <= 0.10 with both in [0.30, 0.55], cam
`shaft_mean_abs_vel` median <= 0.3 rad/s (parked, per Caveat 2). Reference: forward era-B 1a
0.86 / 14 falls / tripod 0.539 (`policy/manifest.yaml`). 0.40 is the era-A pre-mirror-loss tripod
level (0.401 -> 0.517 after the mirror fix in the gait-formation-v2 record) — a floor, not a
like-for-like bar. The driver halts with status `gate_1a` both when the pair fails AND when the
`slow__side` canary eval produced no metrics at all (three failed attempts): an ungated head never
starts 2a unattended; carrying on is an explicit `--continue`.

## Decision table (filled by `decision_table.py`; forward cells from `crab_hex_forward_task/policy/manifest.yaml`)

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
