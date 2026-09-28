<!-- paths-note -->
> **Paths note (2026-09-09):** this campaign moved from `sim_fine_tuning/2026-08-20_1506_hardware_morphology/` to `parkour/parkour_tasks/parkour_tasks/crab_hex_forward_task/experiments/2026-08-20_1506_hardware_morphology/`. Absolute paths below (`/home/.../sim_fine_tuning/...`, `parkour/logs/rsl_rl/...`) name raw artifacts that stay on disk untracked; the tracked checkpoint of record is `head/` (see `bundle.yaml`) and the eval summaries are in place / under `evals/`.

# 2026-08-20 hardware morphology: rebuild sim model from measured physical robot

User supplied measured dimensions/weights/ROMs of the built robot (body 28×48×12.5 in,
350 lb; legs 26.2 lb each, 1-in plywood; linear actuators on hip/knee; yaw ±25°). CAD
cross-check (numeric parse of `~/krabby/joint_specs/KrabV3-Legs.svg`) resolved conflicts
with the user: femur hinge-to-hinge 23.0 in (CAD), tibia knee-to-toe 32.5 in + 3.0 in
knee lever (as-built), inner femur actuator hole all legs, ROMs superseding the
2026-08-13 set (hip 45–150° from vertical-UP → sim [−45°,+60°]; knee interior 5–140° →
sim L [−50°,+85°]; yaw ±25° → K = sin25° = 0.42262, replacing SVG-derived 0.4778).

## Model changes (this campaign, steps 1–3 of the approved plan)

- `assets/crab_simple.usda` is now GENERATED: `assets/scripts/generate_crab_simple.py`
  reads `crab_hexapod_task/mdp/crab_hex_dimensions.py` (single source of truth, stdlib
  only). `tests/unit/test_crab_hex_usd_generation.py` pins the committed asset to the
  generator byte-for-byte — hand-edits now fail CI. Joint anchors ≡ prim translates by
  construction (kills the old ~0.18 m CamShaft anchor drift).
- Geometry: body 0.7112×1.2192×0.3175 m; leg mounts x = ±0.2159/0 (5.5 in from body
  ends, to the yaw axis), y = ±0.6096 (yaw hinge flush with the side face); hip beam as
  horizontal link, yaw→femur-pivot 0.5673 m, beam center 3.25 in below body bottom;
  femur 0.5842 m; tibia knee→toe 0.8255 m; toe = tibia box bottom end, footpad proxy
  there (bare-plywood friction 0.45/0.35 — no rubber feet on hardware).
- Masses: body 158.757 kg (350 lb, all-inclusive); per leg 11.884 kg (26.2 lb) split
  hip 7.413 / femur 2.066 / tibia 2.055 / footpad 0.05 / cam rotor 0.3 (1-in-ply volume
  estimate + actuator bodies on the hip beam, normalized to the measured total; authored
  `physics:centerOfMass` on the hip). Total 230.06 kg (was 106).
- `crab_hex_linkage.py` (new): closed-form linear-actuator linkage for hip
  (anchor solved from the 450–650 mm rod window — geometrically forced to ~21 in above
  the beam near the pivot, matching assembly photos; flagged estimate pending
  measurement) and knee (anchor 1 in from beam outboard tip, 3-in tibia lever, hip→knee
  coupling). `tests/unit/test_crab_hex_linkage.py` pins inverses, derivatives-vs-FD,
  the ROM↔stroke feasibility envelope, coupling sign, and corner unreachability
  (hip full-down + knee full-fold exceeds the rod max — the box limits over-approximate
  the reachable set).
- Defaults = both actuators at exact mid-stroke: hip 0.1105 rad (6.33°), knee L
  0.2341 rad (13.41°), R mirrored. Retires hand-tuned 0.30/−0.07/+0.10.
- spawn_z default 1.05 → 1.10.
- Capability headline (vs old rotary model 1500 N·m / 6 rad/s): hip 120–241 N·m and
  0.23–0.47 rad/s across the ROM; knee 7–38 N·m, 0.43–2.3 rad/s (weak-but-fast when
  deeply folded). Actuator model wiring is the NEXT step (linkage-clamped DC motor).

## Checkpoint A results (geometry statics, this dir)

`static_posture_report.json` (300-step zero-action settle, spawn 1.10):

| Gate | Result |
|---|---|
| total mass | **230.062 kg** ✓ (= 350 lb + 6×26.2 lb exactly) |
| equilibrium pitch / roll | **−0.0025° / −0.096°** ✓ (level; old model needed knee hand-tune) |
| per-foot forces | FL 313 / FR 349 / ML 409 / MR 374 / RL 374 / RR 360 N — **all 6 loaded** ✓ (no floating-pair pattern) |
| tripod A-share | **48.7%** ✓ |
| CoM longitudinal offset | −0.09 mm ✓ |

`verify_joint_drive.log`: all 18 actuated joints OK (gravity on). `verify_contact_physics`
(audit JSON in parkour/logs/.../diagnostics/): footpad regex/bodies resolve, 31 contact
bodies, only footpads in ground contact; 4/6 pads loaded at the 80-step snapshot
(mid-settle transient; the 300-step statistics above are the gate).

`verify_cam_mechanism.log`: hip tracks the quick-return with the new K through
multi-revolution spins, **but** max |hip| = 0.4887 rad = exactly the new ±28° hard stop —
the reversal overshoots ~3° past the 25° mechanism sweep into the stop (heavier legs +
old 2000/40 tracking gains + old 6 rad/s shaft speed). The script's printed
"hard limit 0.5585" is stale (old ±32°); its PASS is misleading. To address in the
actuator step: shaft speed drops to π rad/s (30 RPM hardware motor) which slows the
reversal; re-verify at Checkpoint B, raise tracking gains only if still touching.

Unit suite: 41 crab-hex tests green (limits, generation-diff, linkage, cam, mirror).

## Checkpoint B (actuation) — the stability saga and where it landed

The linkage actuator model went through four measured iterations before the plant stood:

1. **Knee holding cap too low** (80 N·m = "2× drive force"): knees yielded under the
   ~105 N·m stance demand → −4.4° pitch, front feet overloaded. A self-locking screw
   holds until mechanical failure, not 2× drive. Raised → worse (−9.4°), which exposed:
2. **Live-state knee coupling pumped energy**: the knee target was re-solved from the
   LIVE hip angle each substep; through the knee PD's phase lag this fed the body's
   rocking mode (growing pitch oscillation, worse with stiffer knees). Fixed by
   computing the coupling from the hip's COMMANDED trajectory (rod-state feedforward
   only) — `parkour_actions.py` NOTE(coupling-stability).
3. **Quasi-static tipping**: with explicit-stable PD gains (k=2000/912) the stance's net
   pitch stiffness was NEGATIVE: gravity's destabilizer m·g·h ≈ 2100 N·m/rad vs only
   ~400 N·m/rad of leg-PD restoring through the narrow ±0.216 m front/rear leg rows —
   the robot tipped over in ~1 s at near-zero velocity (damping can't help; measured in
   `diag_settle*.log`). No explicit actuator can reach screw stiffness at dt=0.005 →
   switched hip/knee to IMPLICIT PhysX drives (`ImplicitActuatorCfg`).
4. **Fully rigid drives degenerate load sharing**: at k=40000/15000 the hyperstatic
   6-leg distribution collapsed to a diagonal (1061/979/0/0 N). Landed in the window
   satisfying BOTH constraints: **k_hip 16000 / k_knee 6000** (≈1.5× pitch-stability
   margin, ~17 kN/m vertical per foot so mm mismatches redistribute tens of newtons).

Also fixed en route: spawn must clear the KINEMATIC toe height over the terrain SURFACE
(which sits ~18 mm above z=0 on flat-walk) — penetrating spawns (1.05–1.105) caused
violent depenetration transients. `KRABBY_HEX_SPAWN_Z` default = **1.13** (~9 mm clearance).

### Final battery (spawn 1.13, implicit drives in the window)

| Gate | Result |
|---|---|
| settle stability | ✓ no rocking, no tipping, no terminations; pitch converges in ~1.5 s |
| equilibrium pitch / roll | ±1.4° / ~0.1° — residual varies per settle draw (see below) |
| per-foot forces | all 6 capable; single-settle maps are the KNOWN hyperstatic lottery (one draw 234–552 N all loaded; another FR ≈ 22 N) — per the 2026-08-12 campaign, judge only multi-settle statistics |
| total mass | 230.06 kg ✓ |
| settled root height | 1.113 above z=0 → ~1.095 above surface → `GROUND_OFFSET_FROM_ROOT_M = -1.10` |
| joint drive | 18/18 OK (min_vel 0.05 rad/s, real joint speeds 0.23–2.3 rad/s) |
| cam @ π rad/s | PASS, max hip 0.454 rad < 0.4887 hard limit (live-limit check fixed) |
| stance proxy | recall 0.98 (0.999 while driving), dead zone ~2%; precision fell to ~0.72 with the stiffer drives (drive torques fire the proxy in swing) — thresholds are sensing-model params to iterate during retraining |

## Smoke train (fresh Phase-A flat walk, 2000 iters, 256 envs, seed 1)

Run: `parkour/logs/rsl_rl/crab_hex_flat_walk/2026-08-20_16-23-35/` (final `model_1999.pt`);
log `smoke_train.log`; rollout diagnosis `diag_policy_rollout.py` / `diag_rollout.log`.

- Trained cleanly start to finish (no NaN/instability/crash) — the new plant is
  numerically sound under full 256-env training.
- Reward rose 1.55 → 3.13 but episode length plateaued at ~104 steps (~2.1 s), 100%
  failure terminations from iteration ~500 onward.
- **Failure mode (from the checkpoint rollout): the acceleration wheelie.** The policy
  discovered cam-sweep propulsion immediately (all six cams at full ±π rad/s, coordinated
  L/R alternation) and accelerates to vx ≈ 1.2 m/s — 2× the command envelope — while the
  front/middle feet unload and the robot pitches back onto the rear leg row
  (pitch 0.1 → 0.48 rad in ~0.6 s), then terminates. The pitch joints stay AT DEFAULT
  throughout: the policy never moves them — and physically cannot catch a ~0.5 s pitch
  event with 0.23–0.47 rad/s joints. Pitch on this machine must be managed by THROTTLE
  (cam speed modulation), and the velocity-tracking-dominant reward currently pays full
  throttle.
- **Verdict: the plant passes the smoke test** — it stands statically, generates strong
  locomotion thrust, and survives training. The 2-s wall is a reward/curriculum problem
  (retrain-campaign scope): candidate levers are pitch/pitch-rate penalties, command
  envelope enforcement, cam-accel limits, or a stand-first curriculum stage.
- **Hardware flag**: the same physics applies to the real robot — ±0.22 m fore-aft
  support under a ~0.96 m CoM means hard cam acceleration can tip it backward, and the
  linear actuators are too slow to catch it. Throttle ramping should be a firmware/policy
  constraint on hardware too. (Second flag: adjacent hip beams share a plane and their
  ±25° sweeps overlap — no collisions observed in training, but the geometric
  interference exists on hardware as well.)

## Vertical-plate hip correction (PLAN A, 2026-08-20 evening)

The user's video review caught a geometry misreading no numeric gate could: the hip is a
VERTICAL 25.7×5×1 in plate door-hinged to the body side face (yaw axis = its inboard
vertical edge), extending 5 in outboard and ~6.6 in past the body top and bottom; the
femur pivot is mid-width — only 2.5 in outboard of the wall — 22.334 in below the plate's
top tip. The first model had read "22 in tip to hinge" as a horizontal outboard beam:
every leg sat ~0.5 m too far out (toe-to-toe 4.1 m instead of ~3.1 m).

Two structural confirmations fell out of the correction:
- the plate is centered on the body mid-height EXACTLY (asserted invariant in
  `crab_hex_dimensions.py`);
- the hip actuator's CAD anchor hole (2.08 in from the plate top = 20.25 in above the
  pivot) is precisely the near-vertical anchor the linkage rod-window had already forced
  the solver to invent — `crab_hex_linkage` anchors updated to u=0, d=−21 in (clevis at
  the plate top), rod window re-verified [0.456, 0.641] ⊂ [0.450, 0.650] over the ROM.

Changes: dims module (plate constants + CoM in the plate plane, with a z component),
generator (plate box, yaw anchors at mid-plate on the wall edge, Hip_Femur anchor as a
pre-scale Z offset, femur/tibia/footpad shifted inboard to pivot y = ±0.6731), scanner
grid [2.4, 3.4], mid-stroke defaults recomputed (hip 0.0272 rad / knee ±0.3102 —
standing pose sits ~5 cm lower), spawn 1.13 → **1.085** and `GROUND_OFFSET_FROM_ROOT_M`
−1.10 → **−1.05** (both re-measured, not assumed).

Verification (platefix* logs/reports, this dir): 269 unit tests green (one flaky linkage
round-trip test fixed: now seeded + ROM-bounded — the closed-form inverse is only defined
on the operating branch); statics stable and level (−1.3° at spawn 1.13; +0.45° with a
≤0.6° transient at spawn 1.085); mass 230.06; joint drive 18/18; cam max 0.468 rad under
the 0.4887 hard stop. Per-settle foot forces remain a hyperstatic lottery draw-to-draw
(one draw all-six 209–507 N, others drop 1–2 feet) — multi-settle statistics stay the
standard. Geometry review video: `videos/rl-video-step-0.mp4` — awaiting user sign-off
(STOP point; smoke re-run happens after).

## CAD-profile leg meshes (PLAN C, 2026-08-20 evening)

User video review round 2: silhouette right, but the legs are tapered on the real robot
and boxes in sim. The three leg links are now extruded CAD-outline meshes:

- `assets/scripts/extract_leg_profiles.py` (new) samples the KrabV3-Legs.svg outlines
  (beziers subsampled; elliptical arcs sampled BY ANGLE via the W3C endpoint→center
  conversion — control-point bboxes had inflated part lengths: the femur is truly
  28.0 in = 23 in hinge-to-hinge + 2.5 in semicircular caps, not 29.667), maps them into
  link frames anchored on the physically-confirmed datums (femur hinges ±11.5 in; tibia
  knee at +12.835 in with the below-knee span stretched to the as-built 32.5 in; hip
  femur-pivot at −9.5 in), normalizes the asymmetric sides (tibia actuator arm → inboard,
  hip straight hinge edge → wall), decimates (Douglas-Peucker 0.05 in), and writes the
  checked-in `crab_hex_leg_profiles.py`. Profiles: tibia tapers 5.0 → ~1.6 in; femur is a
  waisted dog-bone 5.0/3.0/5.0; femur symmetric (0.001 in), hip/tibia strongly asymmetric
  (3.3 / 4.4 in) → per-side mirrored meshes with handedness-corrected winding.
- Generator emits `def Mesh` for hip/femur/tibia: ear-clipped caps + side quads,
  `subdivisionScheme = "none"`, points in METRIC link-local units (joint anchors on these
  links switched from pre-scale fractions to metric offsets), and EXPLICIT mass
  properties — polygon shoelace area/centroid/second moments × thickness, so inertia
  follows the taper (tibia CoM sits 6 cm above its box center, toward the knee). Hip is a
  composite: plate mesh + both actuator point masses (parallel-axis; products of inertia
  dropped, documented). Hip collision = convexHull of its mesh; femur/tibia collision
  unchanged (disabled/none); footpad box remains the toe contact proxy at the mesh toe.
- Tests: mass parser extended to Mesh prims (tempered regex — the massless ground
  CollisionMesh had swallowed a leg mass), CoM/inertia authored-count checks, and an
  outline-area↔ply-mass tripwire. 270 unit tests green.

Battery (mesh_* logs/reports): statics stable (+1.2° pitch, flat trace, all six feet
loaded in this draw, 230.06 kg), joint drive 18/18, cam max 0.4707 < 0.4887 hard stop.
Geometry review video: `videos/rl-video-step-0.mp4` — awaiting user sign-off before the
smoke re-run.

## Smoke re-train on the corrected geometry (2026-08-20 late evening)

Same protocol as the first smoke (2k iters, 256 envs, seed 1, from scratch), on the
vertical-plate + CAD-mesh robot. Run: latest `crab_hex_flat_walk` dir; log
`smoke_train2.log`; rollout `diag_rollout2.log`.

- Trains cleanly; reward 1.90 → 2.10; episode length WALLS at ~70 steps (1.4 s), 100%
  crab_failure — shorter than the old geometry's 104-step wall.
- Rollout: **the same acceleration wheelie, faster and harder** — all six cams at full
  ±π, front feet airborne from the start, robot riding the middle+rear rows to
  vx = 1.46 m/s (2.2× the envelope), pitch ramping to exactly the 0.5 rad
  `crab_failure` tilt limit at ~1.3 s. Pitch joints parked at defaults throughout.
- **Conclusion: the wheelie is structural** (reproduced across two leg geometries, worse
  with the legs closer in). Pitch on this machine is throttle-managed; the reward stack
  pays full throttle (`reward_forward_progress_along_command.max_speed_scale = 1.75`
  pays up to 1.14 m/s); nothing penalizes pitch rate; termination fires deterministically.
  This is the entry evidence for the PLAN B lever decision.

## Open items
- **2026-08-22 leg-collision fix (user caught legs phasing through each other in the P_T30S video)**: femur colliders existed but were emitted with `physics:collisionEnabled = 0`, tibias had no CollisionAPI at all — both inherited from the primitive-box era; the only leg colliders were hip plates + footpad tip cubes. Fix: generator now emits femur+tibia with explicit `PhysicsMeshCollisionAPI` + `approximation = "convexHull"` + collision enabled (also silences the parse-time fallback errors, closing that open item). Pad still protrudes 2 cm below the tibia hull, so flat-ground contact stays on the footpads. Mass unchanged (230.06 kg); 42 USD/linkage tests green. Physical significance: with ±25° yaw and 8.5" hip spacing, adjacent legs genuinely interfere on the real machine — leg-leg collision is a real constraint (and a natural driver of gait phasing). ALL training before this fix (smoke, Phase A, Phase B waves 1-2, promote wave) ran with intangible leg shafts; quantitative results do not transfer to the corrected plant.

- Cosmetic: femur/tibia meshes lack explicit `physics:approximation = "convexHull"` — PhysX logs a fallback Error per env at parse but uses convexHull anyway (identical to the hip's explicit setting). Silence by adding the attribute in `mesh_geom_text` next time the USDA is regenerated for a real change (byte-equality test + replay fixtures make a standalone regen not worth it).
 carried forward

- Measure on hardware: hip-actuator anchor coordinates (est. u=+2.33 in, d=−21 in from
  femur pivot), actuator retracted pin-to-pin length (450 mm assumed from label).
- Current-sense proxy: retune `_CURRENT_SENSE_*` thresholds (precision ~0.72 vs 0.95 with
  the soft drives) and rerun the stance study on a real trained gait, not the scripted sine.
- Multi-settle statics statistics (20+ jittered settles) before reading anything into a
  single settle's A/B split, per the 2026-08-12 campaign method.
- Retrain fork: fresh Phase-A-style flat run as smoke test; all pre-2026-08-20 checkpoints
  are invalidated (mass, geometry, action semantics, obs semantics all changed).
