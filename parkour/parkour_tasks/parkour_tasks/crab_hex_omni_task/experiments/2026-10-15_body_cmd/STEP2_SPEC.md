# Step 2 spec — eval harness v3 (body rows routed through the term) + adapters + body scenario file

Source of truth: `PLAN.md` D2, D5, D6 and the gates; `CHARTER.md` §3; the step-1 API as built (read
`mdp/crab_hex_omni_command.py` / `mdp/crab_hex_omni_command_math.py` / `config/crab_hex/body_knobs.py` for the exact
names: `CrabHexOmniBodyCommand.schedule_hook_body`, `body_cmd_violations`, `shape_body_command`, `body_hold_step`,
`body_hold_world_row`, `BodyEnvelope` / `parse_body_envelope`, `body_cap_wz`, `command_signature`,
`intended_bearing_b`). Paths relative to `parkour/parkour_tasks/parkour_tasks/`.

This step moves **eval_sha** on purpose (user decision: first-class body evaluation). Everything the legacy rows do
stays bitwise: v1 / v2 manifests parse, validate, compile and score exactly as at HEAD (compile snapshots and golden
tests unchanged), and a world head on `suite__omni` / `field_hard__omni` reproduces its recorded stamps bit for bit
(the orchestrator proves this on GPU in step 3).

## 0. Rules

Same as step 1: CPU only (no Isaac / Kit), never commit / stash / reset / revert, never touch other sessions' files
(`crab_hex_omni_task/experiments/2026-09-*`, `2026-10-07_omni_phase2/`, `lit-review-*.md`, `term`), never
`pgrep -f` / `pkill -f` a string in your own command line. Do not change training-side code (`mdp/`, `config/`)
except to fix a defect you find in the step-1 API — report any such change separately.

## 1. Manifest v3 (`crab_hex_forward_task/scripts/gait_eval/schedule.py`)

- `manifest_version: 3` with a top-level `command_route: term` (the only v3 route) and a top-level
  `envelope: {w_min, ceiling, w_max, cap: [[v, wz], ...]}` block parsed with the step-1 `parse_body_envelope`
  semantics (one envelope definition). Plus `speed_ellipse` and `defaults` as in v2.
- **Body rows:** `BodyScheduleRow(vx, vy, wz, label, step)` (floats, or per-env tensors in the compiled form). Hold
  fields `vx`, `vy`, `wz` (body frame, m/s and rad/s). `wz` may be a number or a SYMBOL resolved from the manifest
  envelope at parse time: `w_min`, `ceiling`, `w_max`, `-w_min`, `-ceiling`, `-w_max`, and `cap80` / `-cap80`
  (= ±0.8 × `body_cap_wz(|v|)`). The resolved numbers are what compile and run_meta record (plus the symbol).
- **Validator:** every hold passes `body_cmd_violations(vx, vy, wz, envelope, ellipse, walk_floor)` = (), except
  rows of `kind: envelope_grid` (the premise grid, which deliberately probes beyond the envelope; it is run only by
  world heads through `body_hold`, and a body head on it is refused). Hold-duration rule: `hold_s` ≥ settle + 2 s.
- **Shapes supported for v3:** the single-pass timeline (`schedule:` list, like `field_hard__omni`) and
  `kind: category_suite` (waves, like `suite__omni`), plus `kind: envelope_grid` (a category_suite whose categories
  are grid cells). Labels: legacy labels where the meaning carries over (`stand`, `fwd`, `lat_l`, `lat_r`, `back`,
  `diag`), new labels for turns (`turn_l_<level>`, `turn_r_<level>`, `move_turn_l`, `move_turn_r`, grid
  `g_v<speed>_<fwd|lat>_w<wz>`). Category assignment for v3 suites: `turn` / `move_turn` with hands as today.
- v1 / v2 parsing, validation, compile and every snapshot test unchanged.

## 2. Harness routing (`crab_hex_forward_task/scripts/eval_crab_hex_gait.py`)

- `--command-adapter {none,body_hold,world_rate}` (default `none`), recorded in run_meta.
- **Route check (gate 0 INVALID otherwise):** head frame = the env's command class (body iff the term has
  `schedule_hook_body`). v3 rows on a body env → `none` (drive `schedule_hook_body` per step through the same
  `compute` override slot `_install_schedule_override_world` uses). v3 rows on a world env → only `body_hold`.
  v1 / v2 world rows on a world env → `none` (bitwise legacy path). v2 world rows on a body env → only `world_rate`.
  Any other combination → SystemExit naming both.
- **`body_hold` adapter** (world head on body rows): per env, per step, `face = body_hold_step(face, yaw, wz, dt,
  windup=0.6)` (face starts at the spawn yaw and is carried across holds; on a reset it restarts at the spawn yaw),
  then the world `schedule_hook(ScheduleRow(speed=|v|, dir=yaw + atan2(vy, vx), face))`. Record the adapter's
  per-step face in the raw npz (`adapter_face`).
- **`world_rate` adapter** (body head on world rows): per step, the world row → the same body command the world term
  would produce (`body_command_from_world` + `heading_rate_command` with the row manifest's `ang_vel_max`, the
  term's stiffness and clip — reuse the step-1 math, do not re-derive) → `shape_body_command(..., turns=T)` with the
  head's envelope (from its env's command cfg) → `schedule_hook_body` with per-env tensors. `T` = `--adapter-turns
  {on,off}` (default on); off for heads of stages without turn slots. Log the clipped-step share per hold
  (`adapter_clipped_frac` by hold in the metrics JSON).
- **Raw additions for every v3 run and every adapter run:** `obs_cmd` (T, 6) = the policy observation's command
  channels 5–10 as the policy received them (read from the obs buffer the policy consumed, not recomputed);
  `cmd_applied` stays. `spawn_yaw: first_face` pin applies to v3 rows too (face 0 = yaw 0).
- run_meta: `command_route`, `command_adapter`, `adapter_turns`, the resolved envelope (manifest and, for a body
  head, the env's own), `command_frame` of the env, and the existing env_vars record.

## 3. Metrics (`gait_eval/metrics.py`) and report (`gait_eval/report.py`)

For every hold of a v3 run (and every hold of an adapter run), over the hold-relative scored window `[4, hold_s)` s
(12-s holds → [4, 12)):
- `net_yaw_rate` = net heading change (unwrapped yaw from `root_quat_w`) / window length; `net_yaw_ratio` =
  `net_yaw_rate / wz_cmd` (SIGNED; None where wz_cmd = 0); `wz_ratio` (mean `root_ang_vel_b[:, 2]` / wz_cmd, signed;
  reported, never gated); `yaw_ripple` = std of the non-overlapping 1-s block means of `root_ang_vel_b[:, 2]` within
  the window; `heading_drift` (wz_cmd = 0 holds) = |mean `root_ang_vel_b[:, 2]`| over the window; `planar_drift`
  (pure turns) = |xy displacement| / window length;
- body-frame translation: `body_ratio` = mean achieved body velocity projected on the commanded body bearing / |v_cmd|
  (moving holds), and `cross_body` = mean |perpendicular component|;
- survival per hold (no termination in the hold) as today; per-episode values kept in the metrics JSON so readers
  can take medians / p90 / per-hand splits.
`summarize()['cat']` gains these keys for v3 categories; extend `eval_categories.CAT_KEYS` and its cross-pin tests
accordingly (the "rule v3": `rule_version: 3` = v2 + the body keys; v2 expansion bitwise). `trained_categories(env)`
must work for a body env (same pool names → same categories; `turn_annotation` says `body` instead of a facing
band). summary.md gets a "body yaw" block per hold. v1 / v2 reports byte-identical.

## 4. Driver (`crab_hex_omni_task/experiments/tools/run_phases.py`, eval side) and hashes

- `EVAL_CARRY` += `KRABBY_OMNI_COMMAND_FRAME`, `KRABBY_OMNI_BODY_ENVELOPE`, `KRABBY_OMNI_BODY_CAP`;
  `EVAL_CARRY_DEFAULTS` += `KRABBY_OMNI_COMMAND_FRAME: "world"` only (the envelope keys have no default: a world
  eval must not carry them, the knob refuses them in world mode). The training seam stays bitwise (its existing
  test). A body head's eval carries `KRABBY_OMNI_FACING_OBS_ENCODING=none` from its preset — check
  `EVAL_CARRY_DEFAULTS`' `raw` cannot override it.
- Wherever run_phases / run_candidates build an eval command for a row, pass `--command-adapter` from the pairing
  (head frame × manifest version) and `--adapter-turns` from the stage's trained categories (`turn` or `move_turn`
  present → on), so the driver never launches an INVALID combination.
- `experiments/tools/run_candidates.py` `EVAL_PATHSPECS` += `crab_hex_omni_task/experiments/eval/scenarios_omni_body.yaml`.

## 5. `crab_hex_omni_task/experiments/eval/scenarios_omni_body.yaml` (new; v3)

Header comment in the house style (why each row exists, the frozen-shape rule, the envelope note: the `envelope:`
block is PROVISIONAL until step 4 freezes it from P1; the orchestrator rewrites only that block then). Provisional
envelope: `w_min 0.05, ceiling 0.167, w_max 0.21, cap [[0, 0.21], [0.22, 0.13], [0.30, 0.10]]`. `defaults` as
`scenarios_omni.yaml`; `speed_ellipse {fwd 0.35, lat 0.35, back 0.25}` (the trained ellipse). Rows (all Walk-Play,
`spawn_yaw: first_face`, 100 episodes / envs per wave, env_seed 1):

- `suite__omni_body` (category_suite, 10 waves, settle 2 s, hold 12 s): stand (0,0,0); fwd (0.25,0,0); side lat_l
  (0,0.22,0) / lat_r (0,−0.22,0); back (−0.22,0,0); diag (0.25 cos45°, 0.25 sin45°, 0); turn turn_l_ceiling
  (0,0,ceiling) / turn_r_ceiling (0,0,−ceiling); move_turn move_turn_l (0.22,0,cap80) / move_turn_r (0.22,0,−cap80).
- `turnlevels__omni_body` (category_suite, 10 waves, settle 2 s, hold 12 s): turn l/r at w_min, 0.08, 0.15,
  ceiling, w_max (one category per level, hands l/r).
- `grid__omni_body` (`kind: envelope_grid`, settle 2 s, hold 12 s): |v| ∈ {0, 0.22, 0.30} × {fwd (+x), lat (+y)} ×
  wz ∈ {0, 0.08, 0.15, 0.30} (|v| = 0 once per wz) = 20 waves, labels `g_v<…>_<…>_w<…>`.
- `field_hard__omni_body`, `field_light__omni_body`, `ring_obst__omni_body`: the 8-hold timelines of
  `experiments/2026-10-07_omni_phase2/scenarios_p2.yaml`'s `field_hard__omni` / `field_light__omni` /
  `ring_obst__omni` with the SAME env blocks and episode lengths, holds translated to body rows: stand 8 s, fwd
  (0.25,0,0) 8 s, lat_l (0,0.22,0) 8 s, diag 8 s, back (−0.22,0,0) 8 s, turn_l_ceiling 12 s, turn_r_ceiling 12 s,
  move_turn_l (0.22,0,cap80) 8 s.

## 6. Tests (CPU)

v1 / v2 golden + compile snapshots unchanged; v3 parse / validate (each refusal, symbol resolution, grid exemption,
hold-duration rule); route-check table (every combination: allowed / SystemExit); adapters on synthetic yaw traces
(`body_hold` anti-windup, face carried across holds and reset at reset; `world_rate` equals the world term's body
command before shaping, shaped output valid, turns-off → wz 0, clipped share counted); metrics on synthetic traces
(net ratio sign per hand, ripple of a constant-rate trace = 0, heading drift, planar drift, body ratio); CAT_KEYS
cross-pins; EVAL_CARRY frame defaults and the encoding-carry check; EVAL_PATHSPECS contains the new file. Run the full
`tests/unit/` suite at the end and report counts.

## 7. Report (final message)

Files changed (one line each), pins you made beyond this spec, test counts, anything unsure. Terse.
