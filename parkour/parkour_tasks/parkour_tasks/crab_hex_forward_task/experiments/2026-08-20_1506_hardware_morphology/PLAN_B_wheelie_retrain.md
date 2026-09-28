# PLAN B — anti-wheelie retrain campaign (SAVED, NOT STARTED)

Prerequisite: Plan A complete, including the corrected-geometry smoke results. First
action is the user's lever decision (presented with those results):
(a) **structural + reward** — bake a cam-ACCELERATION slew in
    `CrabHexDelayedJointPositionAction` (mirrors the rod slew; the real gearmotor can't
    step 0→30 RPM instantly; removes instant-full-throttle by construction) plus reward
    arms; or
(b) **reward-only** — action term frozen; escalate to (a) only if a full reward round
    fails its floor.

## Campaign (fully autonomous once levers are chosen)

Campaign dir: `sim_fine_tuning/<date>_wheelie_retrain/` with CHANGELOG.md in the
phased-flat format (arm rows, gates, mechanism diagnoses). Serial GPU;
`systemd-run --scope -p MemoryMax=45G` + timeout on every launch. Stops only at bake
proposals / fork decisions (push-notify).

### Arms are config deltas on existing machinery (no new reward code unless noted)

From the reward-stack map (`CrabHexFlatWalkRewardsCfg`, parkour_mdp_cfg.py L764-1147):

| lever | mechanism | current | arm values |
|---|---|---|---|
| progress overspeed | `reward_forward_progress_along_command.max_speed_scale` | **1.75** (pays up to 1.14 m/s — prime wheelie suspect) | 1.0–1.1 |
| pitch rate | `reward_ang_vel_xy` (exists, impl rewards.py:106) | **0.0** | −0.05 … −0.3 (add `KRABBY_ANGVEL_W` to the L1118 override block — small edit, precedented) |
| tilt penalty | `reward_orientation` | −0.7 | −1.5 … −3.0 |
| tracking error | `penalty_tracking_error_l1` (`KRABBY_TRACK_L1_W`) | −0.5 | −1.0 … −2.0 |
| stand-first curriculum | `commands.ranges.lin_vel_x` | (0.30, 0.65) | stage A (0.0, 0.35) → widen after stand gate |
| cam accel (if chosen) | new slew in action term, `CAM_ACCEL_LIMIT` in crab_hex_dimensions | none | ~π rad/s per 0.5 s |
| action rate | `reward_action_rate` (covers cam cols) | −0.3 | −0.6 … −1.0 |

New-term rule (offline replay gate, per project memory): any NEW reward function is
replayed before its first screen on fixtures FROM THE CORRECTED PLANT — the smoke
wheelie rollout (degenerate), a zero-action stand (healthy-static), and a synthetic
ideal, recorded via `eval_crab_hex_gait.py --save-raw` npz. Old-model fixtures are
invalid. Weight-only changes to existing terms don't need the gate.

### Batch protocol (2-stage, user-approved)

- **B0 baseline** = current config from scratch, 1k iters — the number arms must beat.
- **Screens**: 1k iterations (~25 min), from scratch, `--num_envs 256 --seed 1`,
  env-var deltas only (`run_combo_round.sh` pattern from the phased-flat campaign).
- **Screen gates** (train log): mean episode length ≥ 400 (vs the ~104 wheelie wall);
  crab_failure fraction < 0.5 (vs 1.0); reward ≥ B0.
- **Promotion**: survivors → 5k iterations. Winner gates via
  `eval_crab_hex_gait.py --scenario flat_walk_forward` (10 episodes, holds
  0.30/0.475/0.65): completion ≥ 0.9 (timeout-dominated), pitch_max_abs < 0.3,
  vx deficits < 0.1 at low/mid holds, stride/slip sane; tripod ≥ 0.3 is a stretch goal,
  not a gate (gait quality is a later campaign).
- **Escalation ladder**: singles → combos → structural fork (if reward-only chosen) →
  user fork. Floor rule: an arm must double B0's episode length to stay alive; a fully
  null singles round is a hard stop.
- Multi-settle statistics for any statics claims (2026-08-12 campaign method).

### Eval tooling fixes (small, campaign setup)

- `eval_crab_hex_gait.py:491`: hard-coded 1.0 m root-to-ground fallback → 1.10 (and
  check `run_meta["terrain_source"]` before trusting clearance numbers).
- Scenario manifest: sha-pinned old checkpoints → `--allow-checkpoint-sha-mismatch`;
  add `scenarios_v2.yaml` only if holds change (v1 header forbids in-place edits).

### Verification

Every arm gets a CHANGELOG row (config delta, train metrics, eval metrics, mechanism
diagnosis for failures); the winner is reproduced once with a different env seed before
any bake proposal; bake proposals go to the user with the eval table.

---
