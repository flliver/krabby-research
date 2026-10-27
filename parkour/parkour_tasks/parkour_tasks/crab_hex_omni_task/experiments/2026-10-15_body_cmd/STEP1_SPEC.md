# Step 1 spec — body-frame command mode (code, default-off) + CPU tests

Source of truth: `PLAN.md` (D1, D2, D4, D6) as amended by `CHARTER.md` §3 (2026-10-15: 2-s yaw window, σ 0.10 rad,
tent weight 2.0). This file turns them into an implementation contract. Where this file pins a detail PLAN leaves
open, the pin is recorded in the CHARTER by the orchestrator.

Paths below are relative to `parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task/` unless absolute.

## 0. Invariants (non-negotiable)

1. **Default-off, bitwise.** With `KRABBY_OMNI_COMMAND_FRAME` unset (or `world`) every world preset's config dump,
   the sampler's RNG stream, every reward / observation value and every metric are bitwise what HEAD `60d7f45d`
   produces. No new cfg FIELD may appear in a knob-off dump (use the established `getattr`-on-instance or
   subclass-only patterns; read how `relay_hyst` / `stand_hold` / motion knobs avoid widening the dump). The
   orchestrator will compare Kit config dumps before/after (13 targets: presets 1a 1b 1c 2a 2b 2c 2c_clockheld + the
   shipped stage overlays fb6k c4 c5 c6 c6x p2a) — any diff in a knob-off dump is a defect.
2. **CPU only.** Do not launch Isaac / Kit / any GPU process (one-Isaac-process rule; the orchestrator runs Kit).
3. **Git.** Do not commit, stash, reset or revert anything. Other sessions have uncommitted edits in this checkout
   (`experiments/2026-09-*`, `2026-10-07_omni_phase2/`, `lit-review-*.md`, `term`): never touch them.
4. Match the surrounding code's style: long docstrings explaining WHY, loud named errors, pure math in
   `mdp/crab_hex_omni_command_math.py` with CPU tests, relative imports inside `mdp/`.
5. Not in this step: the eval harness (`gait_eval/`, `eval_crab_hex_gait.py`, `run_phases.py`, scenarios files,
   `eval_categories.py`) — step 2. Do not edit them.

## 1. Knobs (body mode)

Put the body knobs in a new `config/crab_hex/body_knobs.py`, path-loaded and re-exported by `omni_knobs.py` exactly
the way `motion_knobs.py` / `clock_knobs.py` are (so the candidates driver's whitelist reads the `*_VAR` names).

| knob | meaning |
|---|---|
| `KRABBY_OMNI_COMMAND_FRAME` | `world` (= unset / blank, the default) or `body`; anything else refused |
| `KRABBY_OMNI_BODY_ENVELOPE` | `w_min:ceiling:w_max` (rad/s), `0 < w_min < ceiling <= w_max`; REQUIRED in body mode, REFUSED in world mode |
| `KRABBY_OMNI_BODY_CAP` | `v0@wz0,v1@wz1,...` coupled move+turn cap: v strictly increasing, `v0 == 0`, wz non-increasing, every wz in `(0, w_max]`; REQUIRED in body mode, REFUSED in world mode |

Body mode, applied at one documented point in `apply_omni_knobs` AFTER every existing command / obs / reward knob
has been applied (so all of them carry over), does:

- swap `commands.base_velocity` for a `CrabHexOmniBodyCommandCfg` instance carrying EVERY field of the existing cfg
  instance unchanged, plus the parsed envelope (`body_w_min`, `body_ceiling`, `body_w_max`, `body_cap` as a tuple of
  `(v, wz)` float pairs);
- set the omni obs term's `facing_obs_encoding` param to `"none"` (D2);
- apply the body reward set (§4) through ONE function `_apply_body_reward_set(cfg)`;
- refuse (RuntimeError naming the knob and value) when any of these is set: `KRABBY_OMNI_FACING` (any non-blank
  value), `KRABBY_OMNI_FACING_OBS_CLIP`, `KRABBY_OMNI_FACING_OBS_ENCODING` other than blank / `none`,
  `KRABBY_OMNI_FACING_TURN_W` non-zero, `KRABBY_OMNI_TRACK_FACING_SIGMA`, `KRABBY_OMNI_RELAY_HYST`,
  `KRABBY_OMNI_ANG_TURN_ONLY=1`, `KRABBY_RSI_FRAC` > 0, a `cmd_pool` type with a non-unit `ang_scale`;
- skip `_check_clock_radius_rule` (§5 keeps the clock running on every turning row in body mode) — say so in its
  docstring and in a body-mode log line.

World mode: `facing_obs_encoding="none"` refused (a world head would be facing-blind); the two envelope knobs refused.

## 2. Command term (D1)

`class CrabHexOmniBodyCommand(CrabHexOmniCommand)` and `@configclass class CrabHexOmniBodyCommandCfg(CrabHexOmniCommandCfg)`
(`class_type` = the body class) in `mdp/crab_hex_omni_command.py`; the body fields exist only on the subclass.

**Pure math** (in `crab_hex_omni_command_math.py`, CPU-tested):
- `BodyEnvelope(NamedTuple)`: `w_min, ceiling, w_max, cap` + `parse_body_envelope(env_text, cap_text)` (the knob
  grammar; validation as §1).
- `body_cap_wz(v, env)`: piecewise-linear interpolation of the cap points in |v|; **constant beyond the last point**;
  `v = 0` → `cap[0].wz`. Tensor and float inputs.
- `body_v_turn_max(env)`: the largest |v| with `body_cap_wz(v) >= w_min` (`inf` if the cap never drops below w_min).
- `sample_yaw_rate(u_mag, u_sign, lo, hi)`: `|wz| = lo + u_mag (hi - lo)`, sign from `u_sign` with EXACTLY
  `sample_facing`'s sign convention (read it). Per-row `lo` / `hi` tensors.
- `slot_from_body_cmd(vx, vy, wz)`: `moving = (vx, vy) != 0`, `turning = wz != 0`, `slot = moving + 2 turning`.
- `body_cmd_violations(vx, vy, wz, env, ellipse, walk_floor, tol=1e-6)`: the ONE envelope definition (sampler tests,
  the step-2 validator, the eval hook, `shape_body_command`). Violations: speed not in `{0} ∪ [floor, ellipse_cap(atan2(vy, vx))]`;
  |wz| not in `{0} ∪ [w_min, w_max]`; moving AND turning with `|wz| > body_cap_wz(|v|)`. Return a tuple of short
  strings (float path) and a batch bool-mask helper (tensor path).
- `shape_body_command(vx, vy, wz, env, ellipse, walk_floor, turns=True)` → `(vx, vy, wz, clipped)`: the deployment /
  `world_rate` mapper. `turns=False` → `wz = 0`. `0 < |wz| < w_min` → 0. Speed: `0 < |v| < floor` → floor; above the
  ellipse cap → the cap (bearing kept). Turning and moving: if `body_cap_wz(|v|) < w_min`, reduce |v| to
  `body_v_turn_max`, then clamp |wz| to `[w_min, body_cap_wz(|v|)]`; pure turn: clamp |wz| to `[w_min, w_max]`.
  `clipped` = any component changed by > 1e-9. The result always satisfies `body_cmd_violations == ()`.
- `body_hold_step(face, yaw, wz, dt, windup=0.6)` → `yaw + clamp(wrap(face + wz·dt − yaw), ±windup)` and
  `body_hold_world_row(vx, vy, wz, face, yaw)` → `(speed=|v|, dir=yaw + atan2(vy, vx), face)` — the `body_hold`
  adapter (a world-trained head driven on body rows); pure, used by the harness in step 2.

**Sampler (`_resample_command`, body mode).** The SAME uniforms in the SAME order as the world sampler of the same
branch (pool / pool-less): categorical, direction, speed, then the two "facing" uniforms — the latter now feed
`sample_yaw_rate`. Assert in a test that the torch RNG state after a body resample equals the world one's for the
same seed and pool. Per row:
- `phi = wrap(theta − yaw)` (theta from the unchanged direction draw) — the body bearing, held;
- speed exactly as the world sampler at `phi` (ellipse cap × the type's `speed_scale`, walk floor);
- turning rows: pure turn `hi = w_max`; moving `hi = min(w_max, body_cap_wz(v))`, and if `hi < w_min` the speed is
  first reduced to `body_v_turn_max` and `hi = body_cap_wz(v)`; `lo = w_min`; `wz = sample_yaw_rate(...)`;
  non-turning rows `wz = 0`; non-moving rows `v = 0`;
- stand rows are exactly `(0, 0, 0)`; `slot_id` comes from the type table as today and MUST equal
  `slot_from_body_cmd` of the drawn command (assert in tests, never per step on CUDA);
- RSI staging is refused in body mode (§1); if a staged row ever arrives, raise.
Held state: `body_phi`, `body_speed`, `body_wz` (N,). `vel_command_b = (v cos phi, v sin phi, wz)`, constant until
the next resample.

**Every step (`_update_command`, body mode):** no re-projection, no relay. `vel_command_b` stays held;
`heading_target[:] = yaw`, `travel_dir_w[:] = yaw + body_phi`, `speed_cmd[:] = body_speed`, `phi_b = body_phi`,
`recap_mask = False`, so legacy readers see defined values. Metrics keep their names and meanings
(`facing_err_abs` is then 0; `yaw_progress` keeps its `cmd_wz != 0` gate).

**Eval hooks.** `schedule_hook_body(vx, vy, wz)` (also accepts a row mapping / object with `vx, vy, wz`): validates
finiteness AND `body_cmd_violations == ()` (raise otherwise), pins every env to the held body command,
`slot_id = slot_from_body_cmd`, `type_id = -1`, writes the command. The world `schedule_hook(row)` raises in body
mode (route mismatch); `schedule_hook_body` raises on a world term.

**Command identity (both classes).** `command_signature()` → `(N, 3)` tensor that changes exactly when a NEW command
is drawn or hooked (world: whatever tuple of held world fields the existing consumers compare today, so world
consumers stay bitwise; body: `(body_phi, body_speed, body_wz)`), and `intended_bearing_b()` → `(N,)` the commanded
body bearing (world: `travel_dir_w − heading_target` exactly as `telemetry_category` uses it; body: `body_phi`).
Find EVERY reader of `heading_target` / `travel_dir_w` / `speed_cmd` / `phi_b` under `mdp/`,
`../crab_hex_forward_task/mdp/`, `../../../parkour_isaaclab/` that runs in an omni env and classify it:
(a) facing semantics (heading error) — fine in body mode because `heading_target == yaw` (but list them);
(b) change detection or intended bearing — switch to `command_signature()` / `intended_bearing_b()` with a
world path that is bitwise the old expression; (c) commanded-path integration (exposure / promotion) — check it
stays correct when the body command rotates with the robot. Write the classification table into your report.

## 3. Observation (D2)

`facing_obs_encoding` gains `"none"`: slots 6–7 (the facing error) are exactly 0.0, in the observation AND in every
history buffer built from it. Keep `FACING_OBS_ENCODINGS` and `omni_knobs.FACING_OBS_ENCODINGS` equal (they pin each
other). Slot 5 (wz command) and 8–10 (body vx, vy, |v|) keep their meanings — confirm they read `vel_command_b`.
Width unchanged (1149); mirror tables unchanged.

## 4. Rewards (D4 as amended 2026-10-15)

`_apply_body_reward_set(cfg)` swaps FUNCS, keeps term NAMES and weights (weights come from the presets / knobs):
- `reward_tracking_yaw` → class term `RewardYawWindowExp(window_s=2.0, sigma=0.10)`: per-env ring buffers (length
  `N = round(window_s / env.step_dt)`, = 100) of achieved `root_ang_vel_b[:, 2]·dt` and commanded
  `vel_command_b[:, 2]·dt`; zero-filled on the env's episode reset (`reset(env_ids)`); NOT reset on a command change
  (the commanded-history integral handles it — the window compares achieved and commanded yaw over the same trailing
  steps). Value `exp(−|Σach − Σcmd| / sigma)`. On wz = 0 rows this is a 2-s heading hold.
- `reward_yaw_progress` → class term `RewardYawProgressTent(window_s=2.0, w_max from the command cfg)`: same window;
  `cbar = Σcmd / T`, `p = sign(cbar)·Σach / T`, value `min(p, 2|cbar| − p)` clamped below at `−w_max`, exactly 0.0 where
  `cbar == 0`. Weight from `KRABBY_YAW_PROGRESS_W` (2.0 in the `_b` presets).
- `track_ang_vel_z_exp` → `track_ang_vel_z_exp_straight_gated`: the stock term (CALLED, not re-derived) × `(cmd_wz == 0)`;
  std stays 0.5, weight 1.0.
- `reward_facing_turn` must have weight 0 (refused otherwise, §1).
The two window terms may share one buffer helper; buffers must be on the env device, preallocated, no host syncs
per step. Pure income functions (`yaw_window_income`, `yaw_tent_income` on window sums) live in a pure module and
are what the CPU tests pin; the class terms call them.

## 5. Slow turns stay "moving" (D4)

In body mode every place that decides "a motion command is active" or drives the gait clock from the command must
treat a turning row (`wz != 0`) as moving at least at the walk floor (0.21 m/s): the gait clock speed is
`max(|v|, r|wz|, floor·[wz != 0])`. Find the seam (`crab_hex_forward_task/mdp/crab_hex_clock_reward.py::clock_command_speed`
and its omni callers; the gait-reward gates `min_cmd_norm` / `min_forward_speed_cmd` 0.12 — find what norm they read
and whether it includes `r|wz|`). Add a default-0 parameter (e.g. `turn_floor`) so the world path is bitwise; set it
only in body mode. Report every consumer you changed and every one you checked and left.

## 6. Presets (`config/crab_hex/crab_hex_omni_phases.py`)

`BODY_DELTA` = one dict pinned key by key: sets `KRABBY_OMNI_COMMAND_FRAME=body`,
`KRABBY_OMNI_FACING_OBS_ENCODING=none`, `KRABBY_YAW_PROGRESS_W=2.0`, `KRABBY_RSI_FRAC=0`; drops
`KRABBY_OMNI_FACING`, `KRABBY_OMNI_FACING_OBS_CLIP`. Presets `1a_b`, `1b_b`, `2a_b` = the base preset's env + the
delta (same task / experiment / iterations as the base). They are reachable through `phase_env` / `activate_phase`;
do not change the world presets or `PHASE_NAMES`' world entries (check what the existing tests pin and extend them
rather than break them). The envelope knobs are NOT in the presets (the campaign overlay sets them after step 4), so a
`_b` preset alone must refuse to build without them — that is intended.

## 7. Driver whitelist (tool, not hashed)

`experiments/tools/run_candidates.py`: add the three knobs to its knob whitelist / parser table and add
`OVERLAY_DUMP_CHECKS` for body mode (dump shows the body command cfg class + the envelope fields; obs encoding
`none`; the three reward funcs swapped; no `facing_delta` change knob / facing obs clip present in the realised env;
`KRABBY_RSI_FRAC` 0), each tested in both directions on synthetic dumps (the `test_omni_p2b_dumpchecks.py` pattern).
Also accept the `_b` base names wherever the driver validates a stage's `base`.

## 8. CPU tests (new `tests/unit/test_crab_hex_omni_body_cmd.py` + extensions of existing files)

- world sampler bitwise: for several seeds × {no pool, the c4 / c6 / FB pools}, the HEAD-`60d7f45d` sampler (load
  `git show 60d7f45d:<path>` sources into a private package, or an equivalent frozen copy) and the new code produce
  identical tensors and identical RNG state;
- body sampler: every draw passes `body_cmd_violations`; stand exactly `(0, 0, 0)`; `slot_id == slot_from_body_cmd`;
  RNG consumption equals world mode; `vel_command_b` constant across `_update_command` while yaw moves;
  `heading_target == yaw`, `travel_dir_w == yaw + phi`; the move-turn coupled cap and the `v_turn_max` reduction;
- `command_signature` changes exactly at resample / hook; world values bitwise the old expression;
- obs slots 6–7 = 0 under `none` in obs and history; encoding / frame / envelope refusals (each refusal of §1);
- reward pure functions: perfect tracker 1.0 window income; still stander on c = 0.05 → `exp(−1)`; tent values,
  sign, lower clamp, exact zero on straight rows; zero-fill at reset; straight-gated ang-vel term;
- clock speed floor in body mode, bitwise world path;
- presets: `1a_b` / `1b_b` / `2a_b` = base + `BODY_DELTA` exactly; world presets unchanged;
- `shape_body_command` / `body_hold_step` / `body_cmd_violations` properties (shaped output always valid;
  anti-windup bound; dead zone);
- run the whole `tests/unit/` suite at the end (`/home/nickmagus/krabby/isaac_venv/bin/python -m pytest -q tests/unit`)
  and report pass / fail counts; pre-existing failures unrelated to this work must be reported as such (check them
  against `git stash`-free means: e.g. run the failing test file at HEAD in a temporary `git worktree` — or just
  report and say you could not attribute).

## 9. Report (your final message)

Files changed (one line each); every pin you made that this spec did not (list); the §2 consumer classification
table; the §5 consumer list; test counts; anything you could not do or are unsure about. No summary prose beyond
that.
