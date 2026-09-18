# Omni-formation campaign — CHARTER and verification record

Approved plan: `/home/nickmagus/.claude/plans/lets-move-on-to-composed-moth.md` (user, 2026-09-16),
"`crab_hex_omni_task` — omnidirectional walking (travel direction + facing, first plan)".
Successor to nothing directly — first training campaign for the new task package. The omni task
travels in ANY world direction while facing ANY world heading, independently: command 1 = a world
travel direction + speed, command 2 = an absolute world facing target. The forward task's baked
lineage (`crab_hex_forward_task/policy/manifest.yaml`) and the sideways task's head of record
(`crab_hex_sideways_task/experiments/2026-09-11_0414_sideways_formation`) are the references this
campaign is measured against, not checkpoints it resumes from (phase 1a trains from scratch); its
own RSI bank is harvested from BOTH of their heads.

## Question

Does the omnidirectional task, built as a thin subclass of the forward cfgs with world-frame
travel-direction and facing commands in four regime slots (stand / move / turn / move+turn), a ring
of obstacles around the spawn instead of a corridor, and an RSI bank harvested from both the
forward and sideways heads of record, form a usable tripod gait through a three-stage formation
curriculum (1a body +x only → 1b body +y only → 1c any direction), and does the redesigned facing
mechanism (yaw-rate cap, tighter dead band, a facing reward from the omni stage 1c) fix turning — which no
head of record, forward or sideways, has ever done (measured |wz| ≈ 0.003 rad/s against 0.6–1.0
rad/s commanded, REPORT.md:524's "yaw-rate aggregate not recorded" gate having passed on a vx
proxy)?

## Hypotheses

- **H1 — formation curriculum.** The three-stage curriculum (1a forward-only formation, 1b
  sideways-only formation, 1c the full omni regime with the merged RSI bank) yields walking-hold
  tripod ≥ 0.40 on EVERY moving hold (`fwd, lat_l, diag, back, move_turn`) by the end of 1c — the
  written formation gate.
- **H2 — the facing mechanism fixes turning.** The yaw-rate cap (`ang_vel_max` 0.3), the tighter
  dead band (`ang_vel_clip` 0.2), `track_ang_vel_z_exp` std 0.15, the clock driven by turning
  (`r_turn` 1.2 m), `reward_facing_exp` from 1c and (added 2026-09-17) `reward_yaw_progress` together
  produce a median yaw-progress rate (`wz · sign(wz_cmd)`) ≥ 0.05 rad/s on the 90° turn holds — ≥ 10×
  the record 0.003 rad/s — where the stock P-control never has (the
  Verification-9 yaw-capability smoke gates the whole campaign on this before any 5000-iteration
  window runs).
- **H3 — ring obstacles are survivable.** Completion on `ring__omni` (the ring geometry at 0.20–0.70
  difficulty) falls within the sideways campaign's step-onset completion range, despite the
  obstacle numbers not being directly comparable (Caveat 6): a different schedule (8 holds — 6 × 8 s
  plus the two turn holds at 12 s each, 72 s of holds — vs. 3 × 10 s, two of the eight holds turns)
  and a different geometry (a ring around the spawn vs. a
  corridor).
- **H4 — the 2c clock anneal reproduces the sideways regression, or does not.** The sideways
  campaign found a replicated step-onset regression at its 2c window (pooled ≈ −0.17 completion vs.
  2b, [−0.24, −0.10], with no clock-held control run to separate the anneal from the extra
  iterations). The `2c_clockheld` twin (2c's env minus `KRABBY_PHASEOUT`, `KRABBY_CLOCK_W=1.0`,
  same iteration budget, both resuming the 2b head) answers whether the omni chain shows the same
  regression and, if so, whether the clock anneal or the extra iterations is the cause.

## Held fixed (What does NOT change, plan)

Plant A15+B; the 18-channel action layer and actuator model; PPO / distillation hyper-parameters;
seed 3 (+2), 256 envs, 5000 iterations per full window (1c/2a/2b/2c/2c_clockheld; 1a/1b are 1000
each); the scan grid size/resolution/ordering; the 35 reward NAMES and every weight/ramp/phase-out;
the clock constants; terminations (roll/pitch/height/timeout, `crab_failure`); the forward and
sideways packages' own behaviour (config dumps and 20-iteration smokes bitwise); `travel_frame.py`;
the eval harness's body-frame rows; the facing-error observation clip (`KRABBY_OMNI_FACING_OBS_CLIP`,
fixed at 0.6 rad in EVERY phase, Revision 3, 2026-09-17 — unlike the facing band itself, which ramps
0.6 → 1.2 → 2.4, the clip does not widen with it). Teacher-only: no depth student in this plan (the
camera-event bug and the +y mount decision stay open, Caveat 11).

## Delta list (Parts A–D of the plan; one line per design section)

| # | Section | What changes vs. the forward cfgs |
|---|---|---|
| D1 | Command | `CrabHexOmniCommand`: world travel direction + speed (elliptic sampling cap) and an absolute world facing target, four regime slots (stand/move/turn/move+turn), a yaw-rate cap (`ang_vel_max`) with a dead band, per-step re-capped world→body conversion. |
| D2 | Observation | `CrabHexOmniParkourObservations`: steering channels = wrapped facing error on every tile, **clipped to ±0.6 rad by default** (`facing_obs_clip`, knob `KRABBY_OMNI_FACING_OBS_CLIP`, Revision 3, 2026-09-17 — see the Log); command channels = body-frame `(vx, vy, ‖cmd_xy‖)`; the spare yaw-command slot carries the commanded wz; facing error kept in history (`ZERO_STEERING_HISTORY = False`); width 1149 unchanged. |
| D3 | Gait clock / gates | `clock_command_speed(cmd, planar, turn_radius)`: the forward `cmd[:,0].abs()` literal kept verbatim for `planar=False`; the omni branch is `max(‖cmd_xy‖, turn_radius·|cmd[:,2]|)` so an active turn never stalls the clock. Five `cmd[:,0]`-only gate sites and two clock-reward sites dispatch through shared helpers with the forward branch = each site's old literal. **Revision 2 (2026-09-17):** the five command gates read the same clock speed max(‖cmd_xy‖, r_turn·|wz|) rather than ‖cmd_xy‖ alone, so a pure turn switches the gait-shaping terms (foot/swing clearance, excess-contact, stance-support, idle-foot) on like translation does — both yaw smokes had stepped in place with them off. |
| D4 | Rewards | Four terms re-declared under their forward NAMES with omni funcs: `penalty_lin_vel_y`→cross-command-velocity penalty, `reward_tracking_goal_vel`→command-vel-on-obstacle, `reward_tracking_yaw`→`reward_facing_exp` (weight 0.2 from 1c; 1a/1b keep the forward 0.0 — no turning slots there), `reward_obstacle_clearance`→its omni twin; `track_ang_vel_z_exp` std tightened 0.5→0.15 (gradient at 0.15–0.3 rad/s instead of none). **Plus one added term (Revision 1, 2026-09-17):** `reward_yaw_progress` = `clip(wz_b·sign(cmd_wz), ±ang_vel_max)`, gated `cmd_wz ≠ 0`, weight 1.0 via `KRABBY_YAW_PROGRESS_W` in every window — the rotational twin of `reward_forward_progress_along_command`, added because the yaw-capability smoke stood and stepped in place but rotated the wrong way (median −0.025 rad/s against a saturated +0.30 rad/s command inside 0.28 rad/s of gait wobble) and nothing in the 35 rewarded rotating; the omni stack is therefore **36 names**, the forward 35 ⊂ omni's 36. |
| D5 | Mirror | 4-image composition (L/R × fore-aft = R_z(π)): two sign tables derived from the frame rules for the omni head's live slots 5/8/9/10 (yaw pseudo-scalar, true-vector command, norm). As measured (plan A8, step-6 finding): `HEAD_SIGNS_LR` is numerically identical to the forward L/R table and `HEAD_SIGNS_FA` differs from the sideways fore-aft table at slot 8 alone (the forward head's slots 5/8/9 were structurally zero, so their signs were free) — `differ_from_forward_LR == set()`, `differ_from_side_FA == {8}`, pinned by `tests/unit/test_crab_hex_omni_mirror.py`; the scanner offset (0,0,20) lies on both mirror planes. |
| D6 | RSI | `reset_from_reference_states_omni`: a provenance-tagged bank (forward frames `source=0`, sideways frames `source=1`, 1c draws from `all`), re-yawed to a sampled ψ, world velocity rotated into the new frame, and the FIRST command staged so the frame starts on-command with zero facing error. |
| D7 | Scene | `CrabHexOmniTeacherSceneCfg`: the ring terrain generator installed; height-scanner offset re-centred to body (0,0,20) (both mirror planes and lateral/backward visibility need it). |
| D8 | Terrain | New `$LIB/terrains/omni/` package: a square tile, a clear central disc, an annulus of randomly placed bars/slabs (curriculum by difficulty), 8 sector goal anchors that never advance; both-axis edge-detection mask (`parkour_field_to_mesh_xy`). |
| D9 | Parkour/events | `OmniParkourEvent`: tile-centre spawn reference, a goal index that never advances, radial/sector exposure ledgers, promotion by TRACKING RATIO (not downrange distance — fixes the forward term's dead threshold for lateral/backward commands); `reset_root_state_ring`: tile-centre spawn with a uniform random world yaw. |
| D10 | Eval harness | World-frame holds (`speed, dir, face`) compiled and driven through the SAME `schedule_hook`/`_update_command` P-control training uses; per-step travel-frame rotation in `metrics.py`/`report.py`; `facing_metrics` (heading error, settle time, the gated signed yaw-progress rate and the net-rotation cross-check; the magnitude ratio kept for continuity only); `scenarios_omni.yaml` (`slow__omni`, `light__omni`, `ring__omni`). All three rows run the checkpoint on the `Walk-Play` task, never `Teacher-Play` (whose env cfg requires `KRABBY_HEX_TEACHER_MODE`, unset by every eval here); `ring__omni` carries no `env:` block — the driver's `eval_obst` injects `KRABBY_FLAT_TERRAIN_MODE=light`, `_FLAT_FRAC=0.0`, `_GEOM=ring`, `_DIFF=0.20:0.70` at run time (MODE/FLAT_FRAC are load-bearing: without them the shared knob loop's unconditional sub-terrain-proportion reset leaves the ring at proportion 0, i.e. an all-flat build). |
| D11 | Presets | `KRABBY_OMNI_PHASE` (`1a,1b,1c,2a,2b,2c,2c_clockheld`); phase 1 is three stages sharing one MDP, differing only in the command sampler's direction band, slot mix and RSI frame source (1a body +x / 1b body +y / 1c any direction, merged bank); `KRABBY_HEADING` dropped everywhere (the facing sampler replaces it) and is refused if set (`_check_omni`). `KRABBY_OMNI_FACING_OBS_CLIP=0.6` (Revision 3, 2026-09-17) is armed in `OMNI_COMMON` — every window, unlike the facing band it does not ramp with phase. |
| D4b | **Revision 4 — the reward economy (2026-09-18, user decision D)** | The verified 1c diagnosis (`ANALYSIS_1c_failure_2026-09-18.md`): under 1c's terms standing still was the optimum — a Monte-Carlo of the 1c sampler predicts a zero-velocity policy earns `track_lin_vel_xy_exp` 0.835 / L1 −0.182 against the logged head's 0.826 / −0.177, motion-requiring terms were 1.2 % of income, and training speed tracking fell 0.747 → 0.433 → 0.245 while mean reward rose 61 → 93. Four values, applied in `OMNI_COMMON` so the economy is UNIFORM in every window: `KRABBY_TRACK_SIGMA2` 0.1 → **0.04**, `KRABBY_TRACK_L1_W` −1.0 → **−2.0** (both are forward key names, so they are differing VALUES in the delta, not additions), `KRABBY_OMNI_ANG_VEL_STD` (new knob) 0.15 → **0.5** — back to the forward kernel, since the tight one taxed gait wobble on the 88 % of steps that command no turn — and `KRABBY_OMNI_CLOCK_COMBINE` (new omni knob) sum → **product**, so `reward_clock_schedule` pays `swing_quality × stance_quality` instead of handing a creeper the whole stance half free. The clock `combine` MECHANISM is the forward task's own (`clock_schedule_income(..., combine=)`, 2026-08-22); only the omni spelling is new and no forward preset moves. Run shape: a FROM-SCRATCH 1c window (5000–8000 its, `--from-scratch`), then 2a/2b from its head. **The three tuned values are PROVISIONAL until the offline replay confirms the walker out-earns the creeper.** |
| D4c | **A fifth knob, STAGED and DEFAULT OFF (2026-09-18 user decision)** | `KRABBY_OMNI_ANG_TURN_ONLY` (`0` \| `1`, nothing else parses) swaps `rewards.track_ang_vel_z_exp`'s **func** — not a param — for `crab_hex_omni_rewards.track_ang_vel_z_exp_turn_gated`, the stock term multiplied by `(cmd[:, 2] != 0)`, keeping the weight and every param (`std` included, so `KRABBY_OMNI_ANG_VEL_STD` still owns the kernel). **No preset sets it and the rewards class body keeps the stock func**, so every shipped omni window is unchanged; it is armed by hand for one run. WHY IT EXISTS: the offline replay (`replay_reward_economy.md`, PROBE table) found the decided four do NOT clear the `>= 25 %` walker-vs-creeper margin on their own (`fwd` hold COMMAND margin **+16.6 %**) and named the leak — `track_ang_vel_z_exp` is an even function of the yaw-rate ERROR, so a motionless robot with zero error earns ~0.98/s on the ~88 % of rows that command no turn, ~45 % of a stander's whole positive income at `std` 0.5. Gating it takes that to ~0 and moves the margin to **+37.5 %** (FAIL → PASS), creeper motion-requiring share 2.5 % → 4.5 %. WHY IT IS OFF: the gate also removes the term's yaw-rate DAMPING on non-turn rows (today `cmd_wz == 0` pays the policy to hold `wz_b` near zero, i.e. walk straight and not wobble); what remains pricing heading there is `reward_facing_exp` (a position term) and the command's own P-control. That is a behavioural change, not a bookkeeping fix. Gate on the knob is the command's own dead band (`cmd_wz != 0` ⟺ \|facing error\| > 0.133 rad), identical to `reward_yaw_progress`, so the two switch on together. |
| D12b | **Gates read the training log (2026-09-18)** | Every gate (soft 1a/1b and `gate_1c`) additionally requires `Metrics/base_parkour/track_ratio_mean ≥ GATE_TRACK_RATIO_MIN 0.6` over the stage's last 500 iterations — tripod and survival are not translation metrics, and the campaign's own canaries were defective, so the one translation number that needs no GPU and no eval env belongs in every verdict. Seed-3: 1a 0.693 PASS, 1b 0.406 FAIL, 1c 0.246 FAIL — this floor alone would have halted the chain after 1b. Plus a from-scratch early abort: at iteration 1000, its-800-1000 mean < 0.5 → `dead_walker` (checkpoint kept, `--continue`-overridable like a gate). Fixtures: the real seed-3 prints in `tests/unit/fixtures/omni_track_ratio/`. |
| D12 | Package / driver / campaign | Registration (4 `Isaac-Crab-Hex-Omni-*` ids — Walk / Walk-Play / Teacher / Teacher-Play, teacher-only; 16 `Isaac-Crab-Hex-*` ids in total with the 6 forward + 6 sideways), `apply_omni_knobs`/`_check_omni`; this experiments layer: `run_phases.py`/`launch_phases.sh`/`heartbeat_phases.sh`/`bundle_experiment.py`, this CHARTER + `bundle.yaml` + `decision_table.py`. |

## Driver invocation

```bash
OMNI=$R/parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task
CAMP=$OMNI/experiments/2026-09-16_1805_omni_formation
"$OMNI/experiments/tools/launch_phases.sh" --campaign-dir "$CAMP" --plant A15+B --phases 1a,1b,1c,2a,2b --seed 3
"$OMNI/experiments/tools/heartbeat_phases.sh" <unit> "$CAMP" 1800     # run_in_background; the launcher prints the unit name
# then --seed 2 the same way; then, from EACH seed's 2b head (two SEPARATE launches -- both resume
# 2b, never each other; the driver refuses --phases 2c,2c_clockheld together):
"$OMNI/experiments/tools/launch_phases.sh" --campaign-dir "$CAMP" --plant A15+B --phases 2c --seed 3 --from-checkpoint <2b head>
"$OMNI/experiments/tools/launch_phases.sh" --campaign-dir "$CAMP" --plant A15+B --phases 2c_clockheld --seed 3 --from-checkpoint <2b head>
```

The driver's default `--phases` is `1a,1b,1c,2a,2b` (the campaign of record's main chain, Decision 6
of the plan) and its default `--pause-after` is `2b` — the decision point (execution step 19): name
the head, then launch the 2c / `2c_clockheld` pair. Halts: `dead_plant` (step-16 review decision 2,
2026-09-17: applies only to from-scratch windows ≥ 2000 it — none in the omni chain, 1a is the only
from-scratch window and it is 1000 it; collapse is covered by the soft gates and gate_1c),
`gate_1a`/`gate_1b` (soft, short-stage),
`gate_1c` (the written formation gate), `gate_facing` (the facing-precision gate, enforced from 2a),
`nan`, `infra` — each notifies; relaunch `--continue` (finishes an interrupted run, or overrides ANY
of the four gate halts using the already-trained checkpoint without retraining, rewriting the
phase's record to `status ok` + `gate_<name>_override`) or `--redo` after a fix.

## Short-stage handling (1a/1b)

After 1a and 1b the driver runs `slow__omni` ONLY — no `light__omni`/`ring__omni` row (~2 min
instead of ~40): the eval schedule always issues all eight holds regardless of which slots the
stage actually trained (1a samples body +x only, no turning; 1b body +y only, no turning), so a
soft gate reads only the stage's OWN hold rather than the full walking-hold minimum: **1a** `fwd`
walking-hold tripod ≥ 0.30 AND completion ≥ 0.70; **1b** `lat_l` ≥ 0.30 AND completion ≥ 0.70. These
are floors, not like-for-like bars (0.30 is well under the forward/sideways creep references below)
— they exist so a from-scratch formation failure on the very first stage does not burn 1b/1c/2a-2b
hours unattended. `--continue` overrides either halt (`gate_1a`/`gate_1b`).

**Third criterion (2026-09-18): the stage's own training log.** Both soft gates additionally require
`Metrics/base_parkour/track_ratio_mean >= GATE_TRACK_RATIO_MIN` **0.6**, averaged over the stage's last
`GATE_TRACK_RATIO_N` **500** iterations; a missing number is a halt, not a skip. Tripod and survival are gait and
robustness metrics — the seed-3 1b head cleared this gate at 0.360 tripod / 0.96 survival through `lat_l` while its
speed tracking had already fallen to 0.406, and 1c then drifted to 0.245 with nothing watching. Seed-3 numbers on the
real logs (extracted to `tests/unit/fixtures/omni_track_ratio/`): **1a 0.693 PASS, 1b 0.406 FAIL, 1c 0.246 FAIL** —
this floor alone would have halted the chain after 1b.

## The written formation gate (`gate_1c`, enforced by the driver after 1c)

On `slow__omni`: **walking-hold tripod ≥ 0.40 on EACH moving hold** (`fwd, lat_l, diag, back,
move_turn` — enumerated by `tracking_by_hold[h].ratio.n > 0`, i.e. `|cmd| > lin_vel_clip` 0.2, the
same rule as the forward/sideways 1a gates) **AND completion ≥ 0.70 AND a turning-rate criterion**
on `turn_l90`/`turn_r90`: **median `yaw_progress_rate` ≥ 0.05 rad/s on EACH turn hold, AND a
per-direction tracking floor**: **median tracking ratio (`speed_ratio`) ≥ `GATE_1C_DIR_RATIO` 0.3 on
EACH of `lat_l`, `diag`, `back`, `move_turn`.**

**Per-direction floor (step-16 review decision 3, 2026-09-17).** `tracking_by_hold[h]["ratio"]["median"]`
— the harness's per-hold `speed_ratio`, achieved speed along the commanded direction ÷ commanded
speed — must clear 0.3 on every non-`fwd` moving hold. Coverage hole found in review: the
walking-hold tripod floor alone does not require the robot to move in any particular direction, so
a forward-only walker (near-zero net velocity, or velocity aimed the wrong way, on every hold but
`fwd`) could still clear tripod ≥ 0.40 by stepping through the stop on `lat_l`/`diag`/`back`/
`move_turn` without actually tracking them — a forward-only walker must not pass an omnidirectional
formation gate. `fwd` is excluded from this floor because the tripod/completion criteria above
already constrain it. `gate_1c_verdict` (`run_phases.py`) and `gate_1c` (`decision_table.py`) both
read `dir_ratio_by_hold`, name the failing hold(s) in the message, and are cross-checked by
`tests/unit/test_crab_hex_omni_gate.py`.

**Fifth criterion (2026-09-18): the training log.** As for the soft gates — `track_ratio_mean >= 0.6` over 1c's last
500 iterations. This one is independent of the eval env entirely, which matters because the whole seed-3 campaign's
canaries were themselves defective (the missing facing-obs clip): the training number would have been right even when
every eval number was wrong.

**`dead_walker` — the from-scratch early abort (2026-09-18, decision D).** A from-scratch window longer than 1000
iterations is killed at iteration 1000 when the mean of its its-800-1000 `track_ratio_mean` is below
`ABORT_TRACK_RATIO_MIN` **0.5**: no walker is forming, and the remaining 4000-7000 iterations (2-3.5 GPU-h) buy
nothing. Unlike `dead_plant`, the checkpoint IS kept and the verdict is `--continue`-overridable, because it is a
floor on a head that exists rather than a run that produced nothing. Calibration: 1a formed a walker from scratch at
0.747 over exactly this window, so the floor clears any run that is forming.

`yaw_progress_rate` (`gait_eval/metrics.py::facing_metrics`, per hold, aggregated by
`report.py::aggregate` into `facing_by_hold[h]`) is the mean of `wz_t · sign(wz_cmd_t)` over the
hold's steps that carry a yaw command: the achieved rate **toward** the commanded turn, in rad/s,
signed. Gait wobble averages to ≈ 0 in it, a body turning the wrong way is negative, and a body
turning with the command is positive — so the floor (0.05 rad/s, ≥10× the 0.003 rad/s record) means
what Part H's wording intends. Its cross-check on the same line is `net_yaw_rate_toward`, the hold's
signed net rotation toward its (constant) facing target:
`−(unwrap(heading_err)[−1] − unwrap(heading_err)[0]) / (n·dt) · sign(heading_err[0])`, positive when the
body ended the hold rotated toward the target.

**The retired proxy (2026-09-17).** The gate previously read `yaw_rate_ratio × ang_vel_max` (0.3) as
a stand-in for "median |wz| over the first 5 s of the turn hold", the harness exposing no first-5-s
key. `yaw_rate_ratio` divides *mean achieved |wz|* by *mean commanded |wz|* — two magnitudes — so the
gait's own per-step yaw oscillation scores as turning. The yaw-capability smoke measured exactly
that: `yaw_rate_ratio` 0.946 on `turn_l90` (proxy → 0.284 rad/s, 5.7× the floor) while the body's net
rotation over the hold was −0.025 rad/s against a command held at +0.30 rad/s for all 8 s — the proxy
would have PASSED a head that does not turn. Proxy retired; `ANG_VEL_MAX_DEFAULT` is no longer a gate
multiplier. `yaw_rate_ratio` is still printed (continuity) and still reported per hold, but it is
documented as wobble-sensitive and is gated nowhere. `gate_1c_verdict` (`run_phases.py`) and
`gate_1c` (`decision_table.py`) both read `yaw_progress_rate`'s median and are cross-checked by
`tests/unit/test_crab_hex_omni_gate.py`.

No usable number on any of the four criteria (three failed eval attempts, or a hold never scored)
halts, same as a numeric failure — an unmeasured formation must not start 2a unattended.

**The environment every gate is read in** (corrected 2026-09-17; see the Log). Each gate reads
`slow__omni` built by `run_phases.py::eval_env_for` — the canary's fixed reward frame
(`CANARY_KEYS` off `FORMATION`, so a 2a verdict stays comparable with a 1c one) **plus the phase
preset's observation/command knobs** (`EVAL_CARRY`: `KRABBY_OMNI_FACING_OBS_CLIP`,
`_SPEED_ELLIPSE`, `_ANG_VEL_MAX`, `_CLOCK_TURN_R`, `_YAW_RANGE`), `KRABBY_RSI_FRAC=0`, the required
ring geometry, and `KRABBY_PLANT`. The preset variable itself is never exported to an eval. The
sampling, reward and terrain knobs stay out by design: the canary is a minimal flat eval driven by
the manifest's schedule override, and a gate verdict must not move when a stage re-weights a
reward. A gate number produced under any other environment is not a measurement of the policy.

## The facing-precision gate (WATCHED at 1c, ENFORCED from 2a)

Median `heading_error_final_abs` ≤ 0.30 rad on EACH turn hold. Watched (not enforced) at 1c because
the facing band there is only ±0.6 rad — the gate is calibrated for the wider ±1.2 rad band that
starts at 2a, where precision at the eval's ±90° turns is a fairer test. `gate_facing_verdict`
(`run_phases.py`) / `gate_facing` (`decision_table.py`).

## Watched, not enforced (any phase)

Per-direction `speed_ratio` ≥ 0.5; |duty A − duty B| ≤ 0.10; cam `shaft_mean_abs_vel`; slip; knee
saturation; `recap_frac`. No hold-specific reference exists for these on the omni schedule (Caveat
6); they are tracked in the decision table for trend, not gated.

## Decision table (`decision_table.py`)

Columns: flat omni (completion / falls / walking-hold tripod, by hold) · light ring · ring ·
tracking fwd/lat/diag/back (`speed_ratio`) · facing final/settle/progress (per turn hold; `progress` =
median `yaw_progress_rate`, the gated number) ·
duty A/B · falls (travel frame) · cam · slip. Rows: `1a, 1b, 1c, 2a, 2b, 2c, 2c_clockheld`, plus two
hard-coded reference rows:

- **fwd ref** — forward `slow__A15pB` creep: completion 0.86, falls 14/100, walking-hold tripod
  0.633 (`crab_hex_forward_task/policy/manifest.yaml`, era-B 1a).
- **side ref** — sideways `slow__side`: seed 2 completion 0.97, falls 3/100, walking-hold tripod
  0.644 (the named head of record, 2a); seed 3 completion 0.88, falls 12/100, walking-hold tripod
  0.548 (`crab_hex_sideways_task/experiments/2026-09-11_0414_sideways_formation/CHARTER.md`).

**Verification-11 caveat:** these reference cells are recomputed under the OMNI report.py's
definitions (Verification 11 of the plan — no GPU, from each task's own raw eval dirs) so the
`walking_tripod`/`cell` functions agree on both sides; they are NOT phase-matched (forward/sideways
1a used a different command regime, geometry and schedule length entirely) and are printed once,
not per phase. **The facing columns have no reference cell for either row: neither the forward nor
the sideways head has ever commanded a world facing target, so there is nothing to recompute** —
those cells are blank by construction, not a missing-data bug.

## Budget (measured sideways windows 2.3–2.6 h; omni evals 3 rows × 80 s ≈ +8 min/phase; mesh ≈ 78% of forward)

≈2.6–2.9 h per 5000-iteration window (1c, 2a, 2b, 2c, 2c_clockheld), ≈0.6 h per short stage (1a,
1b) incl. its canary-only eval → **1a–2b ≈ 9.7 h per seed, ≈19.5 h for both seeds**; the 2c /
`2c_clockheld` pair ≈ 11 h for both seeds; harvests ≈ 20 min; identity dumps ≈ 25 min; yaw smoke ≈ 30
min (Verification 9, run once before any window). Two seeds (3 then 2), trained serially — one
256-env run saturates the GPU (memory: parallel-gpu-training-runs).

## Caveats and risks

1. The plant has never turned under any head of record; the facing half rests on D1/D3/D4's fixes
   and the yaw-capability smoke (Verification 9) passing BEFORE any 5000-iteration window runs.
2. Turning in place at `ang_vel_max` 0.3 rad/s: a 2.4 rad target takes 8 s of a 10 s resample slot;
   the facing band therefore ramps per phase (1c ±0.6, 2a ±1.2, 2b+ ±2.4) rather than starting wide.
2b. The short stages are seeded, not proven: 1000 iterations with 20% RSI resets should re-form the
    forward/sideways gait fragments (the forward campaigns formed from scratch in 5000 without any
    RSI seed at all), but if the 1a soft gate trips, lengthen 1a before changing anything else; 1b's
    right-leading travel is never sampled and relies on the L/R mirror image of left-leading —
    watched on the 1c evals (there is no `lat_r` hold in the manifest; add one if the 1c gate shows
    a left/right asymmetry).
3. The speed ellipse's 0.35 m/s fore-aft top is above the measured ~0.19–0.22 m/s achieved speeds on
   the forward/sideways plant (policy ceilings at the 0.5 rev/s clock); it is a SAMPLING cap, not a
   target — the eval holds sit at 0.22–0.25 m/s and per-direction ceilings get measured during 1a.
4. Backward travel has no RSI seed (rotating a stored frame never yields backward momentum) — the
   cam must spin the other way to walk backward, which is allowed, just unseeded.
5. Two heads' clock↔contact conventions mix in the merged bank; every staged frame starts on-command
   via the RSI command-staging mechanism (D6), not via its SOURCE task's own command distribution.
6. Ring obstacle numbers are not comparable to the corridor rows of forward/sideways, and the omni
   canary schedule (8 holds × 8 s plus the two turn holds `turn_l90`/`turn_r90` lengthened to 12 s
   each — step-16 review decision 1, 2026-09-17 — 72 s of holds, `episode_length_s` 80) is not
   comparable to the earlier 3 × 10 s schedules either: completion and falls-per-100 are schedule-bound and compare only WITHIN the
   omni campaign; the per-hold walking tripod and the per-direction `speed_ratio` are the
   like-for-like cells against the forward/sideways reference rows (facing has no reference cell at
   all — see the decision-table section above).
7. The `KRABBY_HEADING` mechanism and `ranges.heading` are inert for omni by design; `_check_omni`
   refuses a process where `KRABBY_HEADING` is set (silently-inert config is worse than a loud
   refusal).
8. `import_packages` order is forward, omni, sideways: the omni activation must be a strict no-op
   without `KRABBY_OMNI_PHASE` set (pinned by a delta test's `_environ` check) and must raise if
   EITHER other preset variable (`KRABBY_PHASE`, `KRABBY_SIDE_PHASE`) is also set — this driver's
   `_subenv` asserts both are absent from every subprocess env it builds, belt-and-braces on top of
   the config-level guard.
9. Promotion by tracking ratio demotes a robot blocked backward into an obstacle and one standing
   still on a hard tile identically to a robot that is actually struggling — accepted (same
   trade-off the forward/sideways promotion rules make).
10. Eval cost ≈ 4× the sideways campaign's (3 rows × 100 episodes × 80 s; raw NPZ ≈ 4.5 GB per
    seed); confirm disk headroom before the first `--save-raw` eval lands.
11. The camera-event bug (the shared `random_camera_position` prestartup event misapplies a
    non-identity spawn orientation to the depth camera) and the +y mount decision remain open; there
    is no phase 3 (student distillation) in this plan — teacher-only throughout.
12. (found in step 4 of the plan) The recal2b2 height formulas (≤ 0.09 m at difficulty 1) never
    cross the mesh edge-detection trigger (`slope_threshold × horizontal_scale` = 0.12 m per pixel),
    so `x_edge_mask` is EMPTY for the forward corridor at every difficulty today and the ring's `xy`
    mask is likewise empty at this plan's heights: **`reward_feet_edge` is inert on both the
    corridor and the ring.** Pre-existing, not a regression introduced by this plan; the both-axes
    mask is kept (the mechanism is proven correct on a synthetic 0.5 m block in the terrain unit
    test) and this CHARTER records that edge avoidance is not a live reward term at these heights.
13. (found in step 6 of the plan) With `use_mirror_loss=True` the fork runs `act_inference` on the
    augmented batch: 4 images here vs 2 forward, so the symmetry-loss forward pass per PPO update
    roughly doubles — expect omni windows to run somewhat slower than the 2.6–2.9 h estimate;
    measured on the 1a/1b stages first.
14. (found in step 8 of the plan) The shipped knob loop (`_apply_crab_hex_easy_mixed_terrain`,
    shared with the forward/sideways tasks) overwrites every sub-terrain's `noise_range` to
    (0.02, 0.02) at 1a AND at every 2a+ window (config dumps confirm), so the disc roughness is
    −0.030…+0.025 m at every difficulty and `block_height_tol_m` 0.03 yields NO false-positive
    block pixels from roughness; the residual bias is under-reporting at 1a, whose lowest blocks
    (2.25–3.3 cm) sit at the tolerance. `reach_obst` / `block_steps_mean` are watched-only keys
    either way.

## Log

- **2026-09-16 plan approved; branch created; baseline captured** (`dumps_base` SHA256SUMS; smokes
  byte-identical run-to-run).
- **2026-09-17 yaw-capability smoke (500 it turn-only, run `2026-09-16_20-35-23`) FAILED**: net
  rotation −0.025 rad/s median on `turn_l90` against +0.30 commanded; `yaw_rate_ratio` 0.95 was
  wobble — proxy retired, gate now reads the signed yaw-progress rate; user chose to add
  `reward_yaw_progress` (rotational twin of the forward progress term) and re-run.
- **2026-09-17 step-16 review, applied before the first eval**: (1) `turn_l90`/`turn_r90` hold
  durations 8 → 12 s, `episode_length_s` 70 → 80 in `scenarios_omni.yaml` (a 90° turn to ≤ 0.30 rad
  in 8 s needs ≥ 0.16 rad/s sustained, above the plant's ~0.15–0.2 rad/s; manifest shape frozen at
  this value); (2) `dead_plant` soundness in `run_phases.py::run_phase` restored to the sideways
  from-scratch-only rule (`resume_ckpt is None and iters >= SMOKE_ITERS`) — no omni window
  qualifies, collapse is covered by the soft gates and `gate_1c`; (3) `gate_1c` gains the
  per-direction floor above (coverage hole: a forward-only walker must not pass an omnidirectional
  formation gate).
- 2026-09-17 second yaw-capability smoke (500 it turn-only, `reward_yaw_progress` live, run `2026-09-16_21-17-41`) FAILED:
  `turn_l90` yaw-progress −0.002 rad/s / net −0.001 over a 12-s hold at +0.30 commanded; completion 0.625; training
  `Metrics/base_velocity/yaw_progress` trended −0.001 → +0.014 rad/s (right sign, far below 0.05). Root cause candidate: the
  omni branch of `_cmd_gate` read ‖cmd_xy‖, so a pure turn left every gait-shaping term off (only the clock ran) — the head
  learned to step in place. User decision (option 1 of 4): Revision 2 — `_cmd_gate` reads the clock speed
  max(‖cmd_xy‖, r_turn·|wz|) in its omni branch (forward/sideways literals untouched), and the smoke is re-run at 1500
  iterations so the budget is not the confound.
- 2026-09-17 third yaw-capability smoke (1500 it turn-only, Revisions 1+2 live, run `2026-09-16_21-38-25`): training
  `yaw_progress` climbed to +0.086 rad/s (facing error 0.16 rad, flat failure 0.8 %), but the `slow__omni` read-out FAILED
  differently — all 16 episodes fell inside the +π/2-face holds (`turn_l90` 12/16, `move_turn` 4/16), progress −0.034. Diagnosis:
  the 1c band trains |facing err| ≤ 0.6 rad and the canary's 90° turn hands the policy 1.57 rad on obs slots 6/7 — 2.6× anything
  it has seen. In-band diagnostic (scratch manifest, 0.5-rad turns, same head): completion 1.0, settles 16/16 on every turn hold
  (final |err| ≈ 0.05 rad, yaw-progress 0.05–0.11 rad/s, settle 3–7 s), no falls — the mechanism works; the out-of-band
  observation breaks it. Candidate fix: clip the facing-error observation (knob `KRABBY_OMNI_FACING_OBS_CLIP`, obs term param
  `facing_obs_clip`, default off) so any out-of-band target reads as the band's edge — the yaw-rate command already saturates
  there, so beyond the clip the policy needs only the sign. Verdict on the canary with the clip: see the next entry.
- 2026-09-17 canary read-out of the same 1500-it head WITH the facing-error clip (`KRABBY_OMNI_FACING_OBS_CLIP=0.6`, eval-only,
  `outputs/omni_plan/yaw_smoke3/eval_clip/`): completion 0.81 (3 falls vs 16/16 without the clip); yaw-progress `turn_l90`
  +0.063 rad/s, `turn_r90` +0.094 (settles 13/16, final 0.055 rad), `move_turn` +0.071 — the gate's turning criterion (≥ 0.05 on
  both turn holds) PASSES at 20–30× the 0.003 rad/s record; `turn_l90` final error 0.78 rad after 12 s misses the smoke's watched
  0.6-rad precision line (a turn-only 1500-it head; precision is enforced from 2a). Yaw-capability gate: PASS with Revision 3.
- 2026-09-17 after Revisions 1–3: the four forward/sideways 20-iteration twin smokes (`outputs/omni_plan/smokes_new2/`) are
  byte-identical to the pre-omni baseline after normalisation (fwd1a/side1a 1740 lines, fwd2b/side2b 1800 lines, zero diffs) and
  all 12 forward/sideways config dumps `cmp`-identical — the shared seams are behaviour-preserving for both lineages of record.
- 2026-09-17 user decision: adopt KRABBY_OMNI_FACING_OBS_CLIP=0.6 in every omni preset (Revision 3); launch seed 3.
- 2026-09-17 03:40 seed-3 1a HALTED on the soft gate as first written: own-hold tripod 0.452 (pass) but whole-schedule
  completion 0.09 — 91 of 100 falls were on the five holds a forward-only stage never trains (back 24, move_turn 20,
  turn_l90 17, diag 13, turn_r90 13; only 7 inside `fwd`, 0 in `stand`); `fwd` achieved 0.19 of 0.25 m/s. The criterion was
  mis-specified for the short stages (the review had flagged it): the soft gates now read SURVIVAL THROUGH THE STAGE'S OWN
  HOLD (`survival_by_hold[h]` = share of episodes reaching the end of hold h, cut from per-episode `n_steps` at the manifest's
  hold ends; driver `survival_by_hold()`), same 0.70 floor. Recomputed on the same eval: stand 1.00, fwd **0.93**, lat_l 0.87,
  diag 0.74, back 0.50, turn_l90 0.33, turn_r90 0.20, move_turn 0.09 → 1a PASSES the corrected gate. Relaunched with
  `--continue` (recorded as `gate_1a_override` by the driver; the override is the criterion fix, not a waiver).
- 2026-09-17 (afternoon) EVAL DEFECT + arm C. The driver's `eval_morph` env carries no `KRABBY_OMNI_PHASE` (run_phases.py:548-549),
  so every campaign canary built the Play env on class defaults — facing-error obs UNCLIPPED (1.57 rad on the 90° holds vs ≤0.6 in
  training). Reproduced on the 1c head, 100 episodes each, driver's exact bare env: no clip → completion 0.00, 100/100 falls, turn_l90
  progress 0.005 (identical to the campaign's gate_1c numbers); + clip only → completion 0.39, 61 falls, turn_l90 progress 0.027,
  turn_r90 settles 48/57; with the full preset expanded (KRABBY_OMNI_PHASE=1c set) → 0.49 / 51 falls. VERDICT: the 100 % fall rate
  was the eval defect; the 1a/1b/1c gate verdicts are void until re-evaluated through the corrected env. The policy-level deficit is
  real and separate: even with the clip the head tracks ~0 speed (fwd 0.040/0.25, lat_l 0.006/0.22, tripod 0.85 on lat/back/diag =
  stepping in place) and turns at ~0.03 rad/s — a corrected gate_1c still fails on tripod/completion/speed/turn. A 5-min continuous
  recording under the training distribution (facing targets ±0.6 rad relative, clip live) shows no falls at all (`videos/omni_1c_seed3_5min.mp4`;
  the once-a-minute resets are the Play cfg's hard-coded 60-s timeout, crab_hex_omni_env_cfg.py:205).
- 2026-09-17 (afternoon) EVAL DEFECT **FIXED** in the driver (`EVAL_CARRY` / `eval_env_for`), with a config-dump audit.
  **Defect.** `eval_morph` built the eval subprocess env from five `FORMATION` reward keys + `TERRAIN_GEOM_REQUIRED`, and
  `_subenv` strips every `KRABBY_*` from the parent. `KRABBY_OMNI_PHASE` was never passed (correctly — two presets in one
  process raise), so `activate_phase` never expanded `OMNI_COMMON` in the eval process and the Play env built on **class
  defaults**: `facing_obs_clip` unset (raw 1.57-rad facing error on the 90° holds) against the 0.6 the policy trained under.
  `eval_obst` had the same hole. Separately, `with_plant()` set only `KRABBY_HEX_USD_PATH`, and `plant_usd_path("A15+B")`
  is `None` (A15+B **is** the main asset) — so **no plant variable reached any eval at all**. `_plant_guard` passed because
  the class-default USD coincides with A15+B; `run_meta.plant_requested` recorded `<default>`, i.e. the right plant by luck,
  not by request. `with_plant` now always sets `KRABBY_PLANT`.
  **Evidence** (seed-3 1c head `2026-09-17_00-13-40/model_6997.pt`, `slow__omni`, 100 episodes/arm, arm C above):
  driver's exact bare env → completion **0.00**, 100/100 falls; the same env + only `KRABBY_OMNI_FACING_OBS_CLIP=0.6` →
  **0.39**, 61 falls; full preset expanded → **0.49**, 51 falls. The clip alone moves 0.00 → 0.39, and the `fwd`/`lat_l`/
  `diag`/`back` tripod medians are bit-identical across those two arms — the clip changes behaviour only from `turn_l90`
  on, exactly where the obs channels saturate. Earlier yaw-capability smoke: 16/16 falls raw vs 3/16 clipped.
  **Fix.** `EVAL_CARRY` = the preset knobs that define what the policy OBSERVES or how its command is computed —
  `KRABBY_OMNI_FACING_OBS_CLIP` (obs channels 6/7), `_SPEED_ELLIPSE` and `_ANG_VEL_MAX` (a world-frame row is written by
  the command term's own `_write_command`, not by the harness), `_CLOCK_TURN_R` (the observed clock phase), `_YAW_RANGE`
  (spawn band) — carried from `ph.phase_env(spec.name)` into **both** `eval_morph` and `eval_obst` by the new pure
  `eval_env_for(spec, plant, ...)`, together with an explicit `KRABBY_RSI_FRAC=0`. Sampling/reward/terrain knobs stay out
  by design. Pinned by eight cases in `tests/unit/test_crab_hex_omni_gate.py`.
  **Audit** (`parkour/outputs/omni_plan/eval_env_audit/`, `crab_hex_phase_cfg_dump.py` on `Isaac-Crab-Hex-Omni-Walk-Play-v0`):
  OLD-eval vs training differed in `observations.policy.extreme_parkour_observations.params.facing_obs_clip` (absent vs 0.6)
  — the ONLY observation difference, and the whole train/eval input mismatch. NEW-eval vs training: **zero** differences
  under `observations`, `actions`, `scene.height_scanner`, `terminations` and `episode_length_s`. The 23 remaining
  differences are all deliberately eval-specific: `rsi_reference_reset` absent (10, RSI off), flat-vs-light terrain mix and
  difficulty (7), parkour curriculum unfrozen (3), `reward_tracking_yaw.weight` 0.0 vs 0.2 (reward only — the canary's fixed
  `FORMATION` frame; rewards do not steer a rollout), and `commands.base_velocity.resampling_time_range` / `stand_frac`
  (2 — sampling, inert under the schedule override, which pins `time_left` and drives the term through `schedule_hook`; the
  harness never reads the command term's own metrics). Carrying `_YAW_RANGE` also removed a float mismatch on
  `events.reset_root_state.params.yaw_range` (π vs 3.14159). `heading_control_stiffness` needs no knob: it is 1.5 in every
  arm — `apply_flat_walk_knobs` sets it unconditionally (`crab_hex_env_cfg.py:563`), overwriting the omni cfg's Go2 0.8.
  **Consequence.** Every campaign eval so far — the 1a, 1b and 1c canaries — was measured under class-default observation
  semantics and its recorded verdict must be re-derived through the corrected path. **The 1c gate verdict is VOID until
  re-evaluated.** (As arm C also shows, the corrected gate is expected to fail 1c on its own merits: ~0 tracked speed and
  ~0.03 rad/s turning. The defect explains the 100 % fall rate, not the policy deficit.)
- 2026-09-18 CORRECTED RE-EVALUATION of the seed-3 heads through the fixed eval env (`eval_env_for`, EVAL_CARRY + RSI off + plant;
  `outputs/omni_plan/reeval_corrected/RESULT.md`, 100 episodes each on `slow__omni`):
  | head | completion | walking tripod min | speed ratio fwd/lat_l/diag/back/move_turn | yaw-progress turn_l90/turn_r90 | verdict |
  |---|---|---|---|---|---|
  | 1a `model_999` | 0.15 | 0.308 | 0.76 / 0.13 / 0.32 / 0.33 / 0.18 | 0.021 / 0.037 | soft gate PASS (0.93 through fwd) |
  | 1b `model_1998` | 0.50 | 0.360 | 0.16 / 0.35 / 0.29 / 0.17 / 0.08 | 0.012 / 0.009 | soft gate PASS (0.96 through lat_l) |
  | 1c `model_6997` | 0.39 | 0.270 | 0.16 / 0.03 / 0.15 / 0.33 / −0.04 | 0.027 / 0.046 | gate_1c FAIL (tripod, completion, turn rate, lat_l/diag/move_turn floors) |
  Reading: 1a walked forward (0.76 of command); 1b traded forward for lateral; 1c traded lateral away too — it survives every
  translation hold (survival 1.0) by stepping in place with a 0.85 tripod rhythm and ~0 displacement, and its 61 falls are all on
  the turn holds (survival 0.60/0.55/0.39). The 1c formation failure is real but it is a speed-tracking collapse (the reward economy
  pays stepping-in-place ≥ 92 % of walking's income — trainlog/crosshead lenses), not the 100 % fall rate the uncorrected eval recorded.
  The soft-gate verdicts for 1a/1b stand under the corrected env; the gate_1c halt stands, for the right reasons.
- 2026-09-18 VERIFIED DIAGNOSIS (`ANALYSIS_1c_failure_2026-09-18.md`, 31-agent workflow, 4/8 merged claims survive 3-lens refutation):
  (1) ESTABLISHED — the canaries ran without the facing-obs clip (eval defect; fixed d0fdd9c); (2) the translation deficit is a real
  policy failure visible in training before any eval: track_ratio_mean 0.75 → 0.43 → 0.25 across 1a → 1b → 1c; (3) the reward economy
  makes stillness the optimum: stepping in place under the clock earns ~99 % of available income, walking pays ~25 % more before
  penalties and LESS after; (4) the resume chain lost translation at every handoff and the soft gates (tripod/survival) are blind to it.
  Refuted/not decision-bearing: the turn deficit as a band limit, the tip-over fall signature as a cause, turning under-pricing as primary.
  Options costed: A --continue (proves nothing), B --redo 1c from 1b with TRACK_SIGMA2 0.1→0.04 / TRACK_L1_W −1→−2 / ang-vel std → 0.5 /
  clock combine product (~2.5 GPU-h), C rebuild 1b with a band spanning 0 and ±π/2 + retention gates then B (~3.5 GPU-h), D single-stage
  from-scratch 1c with B's terms (2.5–4 GPU-h). Synthesis recommends: land driver/gate fixes, replay B's terms offline on the saved raws,
  then D (or B); never --continue. Decision: option D, taken 2026-09-18 (below).
- **2026-09-18 USER DECISION D — a from-scratch 1c with the reward-economy fix; values PROVISIONAL until the
  offline replay.** Not `--continue` (option A proves nothing: 2a would inherit a 0.04 m/s walker), not a `--redo`
  from the 1b head (option B preserves a head whose forward travel is already gone). Option D: a single FROM-SCRATCH
  1c window of 5000–8000 iterations on the 1c preset carrying the four Revision-4 values
  (`KRABBY_TRACK_SIGMA2` 0.04, `KRABBY_TRACK_L1_W` −2.0, `KRABBY_OMNI_ANG_VEL_STD` 0.5, `KRABBY_OMNI_CLOCK_COMBINE`
  product), with an early abort at 1000 iterations if `track_ratio_mean` < 0.5; then 2a and 2b resume from that head.
  The three tuned numbers stay PROVISIONAL until the offline replay on the saved 1a/1c raws shows the walker
  out-earning the creeper under them. Landed this day (no GPU): the two new omni knobs, the uniform `OMNI_COMMON`
  economy, `run_phases.py --from-scratch`, the `track_ratio_mean ≥ 0.6` floor on every gate, the `dead_walker` abort,
  and `run_meta["env_vars"]` recording the eval process's real `KRABBY_*` environment (it was `{}`, which is why the
  facing-clip defect was invisible in the run records).
- 2026-09-18 FROM-SCRATCH 1c (option D, Revision 4 economy; run `seed3_1c_004_A15pB`, 5000 it, 2.0 h): gate_1c HALTED on two
  marginal criteria — completion 0.60 (floor 0.70) and training track_ratio_mean 0.568 (floor 0.60) — while every gait criterion
  passes: walking-hold tripod min 0.489 (fwd 0.53, lat_l 0.54, diag 0.49, back 0.58, move_turn 0.51, turns 0.54/0.58),
  per-direction speed_ratio lat_l 0.41 / diag 0.41 / back 0.50 / move_turn 0.44 (achieved 0.15 / 0.09 / 0.10 / 0.11 / 0.10 m/s),
  yaw-progress turn_l90 0.092 / turn_r90 0.098 rad/s (turn_r90 settles 50/75, final 0.09 rad; turn_l90 final 0.49). 40/100 falls,
  concentrated at the first walking onset (fwd 16) then ~1–4 per hold. light__omni completion 0.56. Training: crab_failure_flat
  0.32–0.45 (the economy makes the robot walk — and fall — where the old chain stood still), track_ratio plateau 0.56–0.58 from
  1000 it, dead_walker never armed (0.557 at it 1000). First omni head that walks every direction AND turns; the economy fix
  (Revision 4) is validated on the mechanism. Decision on the halt: the user's.
- 2026-09-18 LEVER STAGED, DEFAULT OFF (user decision): `KRABBY_OMNI_ANG_TURN_ONLY`. The offline replay of the decided four
  (`replay_reward_economy.md`) cleared G3 but NOT G1 — `fwd` COMMAND margin +16.6 % against a +25 % floor — and its PROBE table
  named the largest remaining leak: `track_ang_vel_z_exp` is an even function of the yaw-rate ERROR, so a motionless robot with
  zero error earns ~0.98/s on the ~88 % of 1c rows that command no turn (~45 % of a stander's positive income at std 0.5).
  Landed (no preset, no forward/sideways change): `crab_hex_omni_rewards.track_ang_vel_z_exp_turn_gated` (CALLS the stock
  isaaclab term, multiplies by `(cmd[:, 2] != 0)`, stock argument order, `std` passed through) and
  `omni_knobs._apply_ang_turn_only_knob` (applied right after the std knob; refuses a missing term, a lost `std`, or a func it
  does not recognise). VERIFIED this day: replay `--new-ang-turn-only` → `fwd` COMMAND margin **+37.5 %**, G1 FAIL → **PASS**,
  overall GATE (G1+G3) FAIL → **PASS**; creeper `track_ang_exp` on the translation holds 0.977–0.982 → **0.000–0.004**; creeper
  motion-requiring income share (fwd) 2.5 % → **4.5 %**; creeper stand-hold POS 2.957 → **1.969**. Two 20-iteration 1c smokes,
  seed 3, 64 envs, A15+B (`crab_hex_omni_flat_walk/2026-09-18_08-16-22` off, `.../2026-09-18_08-17-51` on): `params/env.yaml`
  shows the stock func + std 0.5 off and the gated func + std 0.5 on; `Episode_Reward/track_ang_vel_z_exp` 20-iteration mean
  **0.0992 → 0.0338**, final iteration **0.2224 → 0.1475**, last-5 mean 0.1787 → 0.0916, all finite, at comparable episode
  lengths (298.1 vs 292.3 steps). Not adopted: the gate also removes the term's yaw-rate DAMPING on non-turn rows, so it is a
  per-run decision. TRAP recorded: the driver's `_subenv` strips every `KRABBY_*` from the parent env, so exporting this knob
  before `launch_phases.sh` is silently dropped — under the driver it must reach the window's `env` dict, and either way the
  check is `params/env.yaml`'s `rewards.track_ang_vel_z_exp.func`.
- 2026-09-18 LITERATURE REVIEW `experiments/lit-review-omnidirectional-gait-formation.md` (verified pipeline: 5 angles → 12 claims,
  12/12 held after opening every reference; 59 refs). Load-bearing findings for this campaign: (1) the field's stacks (legged_gym /
  Isaac Lab) observe only the saturated yaw-rate command, never the raw heading error — our clipped-error channels are in convention,
  a (cos, sin) facing encoding or dropping the raw channels removes the band problem instead of masking it; (2) sprawled hexapods turn
  by a left/right stride differential at 0.19–0.7 rad/s and the kinematic budget puts this plant's ceiling near 0.15–0.18 rad/s, so
  `ang_vel_max` 0.3 and the 0.35 m/s cap sit ~2× above the plant, and a move+turn hold at 0.22 m/s + 0.3 rad/s is kinematically
  infeasible at the outer foot; (3) the Rudin tracking kernel pays a standstill 78 % of full income at 0.25 m/s by construction and
  command-gated gait income is farmable by stepping in place — only clipped-progress forms are zero at standstill; `reward_yaw_progress`
  is potential-based shaping whose clip/dead band, not weight, caps its income; (4) resumed single-axis heads violate the zero-mean
  assumption behind mirror augmentation (Su et al.) and the state-subspace forgetting mechanism (Wołczyk) explains the handoff losses —
  two more arguments for the from-scratch window. Ranked implications §8: size the caps to the plant, make move+turn feasible, promote
  the zero-at-standstill progress term, gate gait/angular income on ACHIEVED motion, from-scratch windows, seed turn/backward RSI
  frames, (cos, sin) facing, a polar command-grid eval with per-seed stats.
