# Omni phase 2 (obstacles) on the side-first B recipe — CHARTER

- **Pre-registered:** 2026-10-06. Every rule here was written before any phase-2 run.
- **Plan:** approved and audited (3 passes; cap reached; the last fix was not re-audited), at
  `/home/nickmagus/.claude/plans/foamy-wondering-ullman.md`.
- **Precedence:** the CHARTER is the record. Where the two differ, the CHARTER governs, and any change is logged in §9
  before the result it affects is read.

## 1. Question and outcome

**Question.** Does phase 2 on a direction-agnostic obstacle field, carrying the baked side-first B recipe, improve each
seed's phase-1 head on obstacles without losing the flat omni task?

**Secondary questions:**
- What does field+pits buy over the existing ring course (s18)?
- Do three one-knob siblings help?
  - S1: action_rate −0.05.
  - E1: an actuated-joint energy penalty.
  - F1: a stand-slot stance income.

**Outcome.**
- Per-seed verdicts (§6), a recipe read (§6.2), sibling recommendations (§6.4), and a head-of-record proposal (§7).
- **Adoption, bake and push are the user's.**

## 1a. Ground rules

- **Code:**
  - Land one commit before any arm, staging only this campaign's paths. Record code_sha and eval_sha here at landing:
    code_sha `TBD`, eval_sha `TBD`.
  - eval_sha must not move. Never edit `run_phases.py`, the eval harness, `gait_eval/`, `scenarios_omni.yaml` or
    `eval_categories.py`.
- **Running:**
  - One Isaac process at a time, through a detached queue runner and a log-sentinel watcher.
  - Stop runs with `systemctl --user stop`, never SIGKILL. No GPU rendering while the queue can launch.
  - Never `pgrep -f` a string from the watcher's own command line.
- **Authority:** push-notify at every gate and stop. Never push.

## 2. Starting heads (the anchors)

| Seed | Head (c6x, model_14995) | Role |
|---|---|---|
| 18 | `parkour/logs/rsl_rl/crab_hex_omni_flat_walk/2026-10-06_02-24-21_seed18_X60_c6x_A15pB/model_14995.pt` (sha b40f9a19…, = `policy/5_c6x_clockfree`) | head of record, the gate seed |
| 16 | `…/2026-10-05_19-54-57_seed16_X58_c6x_A15pB/model_14995.pt` | replicate |
| 17 | `…/2026-10-05_23-10-39_seed17_X59_c6x_A15pB/model_14995.pt` | replicate |

- Each sha256 is verified against the ppo_construction records before launch.
- Forks use `{checkpoint: <repo-relative .pt>}`, at the head's own seed.
- Alg state carries over: counter 15750, priv_reg 0.1, the bound ramp is finished. The env side restarts (episodes,
  terrain levels).

## 3. Recipe (every arm)

**Base preset:** `2a`, with stage name `p2a` (`compose_recipe` blanks stages named `2a`/`2b`, at run_candidates.py:1064).

**B overlay, restated in full:**
- `KRABBY_ACTION_BOUND_LOSS 0.001`, `_RAMP 0.0001:0.001:500`, `KRABBY_ACTION_BOUND 1.1`;
- `KRABBY_STD_PARAM soft`, `KRABBY_INIT_STD 0.8`, `KRABBY_ENTROPY_COEF 0.0025`, `KRABBY_ACTION_RATE_W 0`;
- `KRABBY_OMNI_MIRROR_IMAGES fa`, `KRABBY_RSI_FRAC 0`, `KRABBY_OMNI_SPEED_ELLIPSE 0.35:0.35:0.25` (absolute_units);
- the c6x `KRABBY_OMNI_CMD_POOL`, `KRABBY_OMNI_SLOTS 0.15:0.55:0.1:0.2`, `KRABBY_OMNI_DIR_BAND ""`;
- `KRABBY_YAW_W 0.2`, `KRABBY_CLOCK_W 0.001`, `KRABBY_APEX_W 0.001`, `KRABBY_OMNI_CLOCK_HIDE 1`, `KRABBY_PPO_HAND_DIAG 1`.

**Phase-2 additions (overlay):**
- `KRABBY_PHASEOUT ""`. This removes the preset's clock/satellite ramps.
- `KRABBY_OMNI_FACING 0.2:1.2`.
- `KRABBY_FLAT_TERRAIN_GEOM field` with the field knobs:
  - pad radius 1.7 m (S3 confirms);
  - 14/28 blocks per tile;
  - bar:slab:pit 0.4:0.4:0.2;
  - ramp edges.
- `KRABBY_TERRAIN_PROMOTE` from §5.

**From the 2a preset:**
- curriculum on, DIFF 0.20:0.70, FLAT_FRAC 0.5;
- DR push 0.5 / mass −0.5:1.5 / CoM 0.01;
- collision −2, edge −0.3 (inert on ramps), stumble −1;
- airtime 0.8 / stride 0.5, verified equal to what B trained with (X60 c6x `env.yaml` lines 1282, 1342);
- 40-s episodes, STAND_FRAC 0.2.

**Length:** 5000 iterations per arm. Extra lr, suite and `field_hard` reads at model_16000, 17500 and 19000.

**Premises recorded:**
- E1 "actuated" prices the rods in practice. The cams carry about 0 of τ·q̇, because their load sits on the Body_Hip
  emulator (REPLAY_GATE.md:254).
- The 2026-09-22 all-joint power term failed Gate 1 on 7/7 categories, so E1 may well fail §4. That stops and asks.

## 4. Offline replay gate (CPU; before E1/F1 train)

The tool is `experiments/tools/replay_motion_terms.py`, run on saved suite npz: the E6 statue, B s18, W, C0 side-first,
and their stillest and jerkiest windows.

**E1 (actuated power):**
- Passes Gate 1 (REPLAY_GATE.md: |ρ| ≥ 0.5, with the right sign, between the term's cost and jitter or
  contact_switch_hz) on stand and on ≥ 4/6 moving categories, on the B head.
- Weight: 1–5 % of locomotion income on the moving categories.

**F1 (stand stance income):** stand rows only (cmd 0 and wz 0).
- The statue beats every stepping head.
- Within each head, the income correlates with stand steps/s at ρ ≤ −0.5. This is decisive.
- Its ρ with stand lift clearance and tripod score is reported only, since the goal is a low tap.
- It is zero on every moving row (asserted).
- Weight: the smallest value in 0.5–1.0 at which a still stance beats the head's stepping by ≥ 0.5/s on stand rows.

**A failure of either stops and asks the user.** No arm is dropped automatically.

## 5. Smokes, calibration and anchors (on landed, reviewed code)

**Smokes:**
- **S1 cfg dump:** field classes, proportions, noise, difficulty, curriculum and radii.
- **S2 live terrain:** pad |h| ≤ 0.035 m, coverage in band, corners populated, heights ≤ formula + 0.02 m.
- **S3, 200 resets:** no foot over occupancy at t = 0, and zero failures in the first 1 s. The measured maximum toe
  radius must be ≤ 1.6 m, or stop and ask.
- **S4, the head walking 2 episodes,** with thresholds from B's ring telemetry:
  - reach_edge ≥ 0.9;
  - field_frac ≥ 0.51;
  - block steps per obstacle-tile episode ≥ 230;
  - off-tile block steps > 0.
- **S5:** `field_light` at 7 episodes, with zero stand falls.

**Promotion calibration, in this order:**
1. A ≥ 300-iteration fork smoke from s18 at the preset 0.45:0.25 on the field. It must pass the resume read, the
   wiring read (§8) and the dump audit.
2. r̄ = the median per-episode `ep_track_dist/ep_cmd_dist` on field-tile envs. The smoke dumps per-episode values;
   `track_ratio_mean` is an all-tile running mean and is not r̄.
3. up = 0.9·r̄, down = 0.5·r̄, rounded to 0.05.
4. A second fork smoke, ≥ 300 its, at the final values. It must pass the dump audit, with the PROMOTE values
   realised, and its terrain levels must move.

**Dry run:** every validator OK and the clock rules pass.

**Anchors:** the three c6x heads.
- **Rows:**
  - suite and side22_25lr;
  - `slow__omni`, the flat 8-hold row, giving the flat completion ceiling;
  - the §6.0 rows: field_light, field_hard, ring_obst, field_turn and field_suite.
- **Second env_seed:** field_hard, field_light and ring_obst also run at env_seed 2; s18 also runs its suite there.
- **Code-state sidecar:** each stamp gets one holding `run_candidates.code_sha()`, `eval_sha()` and a dirty flag on
  the code and eval pathspecs. Record commits and tool-file edits do not change it.
- **Inertness:** the s18 suite reproduces `policy/manifest.yaml` exactly (same manifest).
- **Discrimination:**
  - anchor field_hard completion is in [0.15, 0.85];
  - field_light ≥ field_hard + 0.10.

  Otherwise stop and ask before any re-tune. The flat completion (`slow__omni`) is reported beside it, as the gain
  headroom.
- **Noise band, per seed and metric:** the |difference| between that anchor's env_seed 1 and 2 evals. It is floored at:
  - the pooled binomial SE and ±0.04, for completion and survival;
  - 0.02 for ratios and facing_closed;
  - 0.005 for yaw_progress.

  Suite metrics use s18's band.
- **Re-anchoring:** any code or field-knob change after the anchors re-runs them.

## 6. Pre-registered reads

**6.0 Rows** (campaign-local `scenarios_p2.yaml`, 100 episodes each; every field/ring row carries `MODE light` +
`FLAT_FRAC 0.0`):
- `field_light__omni`: field, DIFF 0.05:0.2, the `light__omni` 8-hold schedule.
- `field_hard__omni`: field, DIFF 0.20:0.70; the main read.
- `ring_obst__omni`: ring, DIFF 0.20:0.70, the `ring__omni` schedule with an env block.
- `field_turn__omni`: walk out 10 s, stand 8 s, turn_l90 / turn_r90 12 s each.
- `field_suite__omni`: `suite__omni` with the field env.

**"Completion"** is the 8-hold rows' `schedule_completion_rate`, pooled over env_seeds 1 and 2 (n = 200) for anchors and
arm heads alike. Every arm head therefore runs field_hard, field_light and ring_obst at both env_seeds.

**6.1 Per seed**, against its own anchor. First match decides:
1. **INVALID EVAL:**
   - a stamp lacks CLOCK_HIDE=1, STD_PARAM soft, ellipse 0.35:0.35:0.25 or FACING_OBS_CLIP 0.6;
   - or its row env block, or the code_sha / eval_sha / dirty flag in its code-state sidecar, differs from its
     anchor's;
   - a missing sidecar is INVALID, never equal.
2. **PENDING:** a read is missing or NA.
3. **HAND LOSS:** a side hand is not FAST (feet AND along-command), or hold-fall share > 0.15.
4. **DOSE:** median |μ| > 1.5 over the last 100 iterations.
5. **FLAT FAIL:** a listed metric worse than the anchor by more than max(floor, 2·band), or translate survival < 0.95.
   - Floor: 0.10, except 0.03 for yaw_progress.
   - The list:
     - side, fwd, back and diag tracking error |ratio − 1|, where worse = larger;
     - turn facing_closed and yaw_progress;
     - move_turn survival.
6. **NO GAIN:** this fails: field_hard completion ≥ anchor + max(0.10, 2·band). Gain also needs:
   - field_light ≥ anchor − max(0.03, band);
   - ring_obst ≥ anchor − max(0.05, band).
7. **COST:** a gain, but a listed metric worse than the anchor by more than max(margin, band). Margins: |ratio − 1| and
   fc 0.03, yp 0.01, move_turn survival 0.05.
8. **PASS:** otherwise.

**6.2 Gates and recipe read:**
- A1 s18 PASS continues. **Any other outcome stops and asks.**
- At 3 seeds:
  - ADOPTABLE: ≥ 2/3 PASS and no HAND LOSS;
  - TRADE-OFF: ≥ 2/3 gain with the same COST in ≥ 2/3;
  - NOT ADOPTABLE: ≤ 1/3 gain, or ≥ 2/3 HAND LOSS;
  - MIXED: otherwise.

**6.3 Field+pits vs ring (A1 s18 − A0 s18; paired, descriptive):**
- field_hard and ring_obst completion;
- falls per obstacle-tile episode;
- exposure from eval raws with one estimator for both arms;
- terrain levels side by side.

**6.4 Siblings vs A1 s18 (paired, n = 1).** Every verdict stops and asks with the numbers; the rule gives the
recommendation.
- **S1/E1, recommend keeping** if:
  - both hands are FAST;
  - side feet_ss drops by ≤ max(0.02, band);
  - jitter is down ≥ 25 %;
  - nothing loses beyond max(margin, band).

  E1 also needs mean |τ·q̇| down, and is reported on stand stepping too.
- **F1, recommend keeping (low tap)** if:
  - A1 s18 still marches at stand (clearance > 0.05 m, tripod > 0.4);
  - F1 has stand lift clearance ≤ 0.05 m and tripod ≤ 0.4;
  - nothing loses beyond max(margin, band).

  Steps/s and six-down share are reported only. If A1 s18 already taps, F1 is not recommended (no effect to
  attribute) and is reported descriptively.

**6.5 p2b** (base 2b, clearance terms, `FACING 0.2:1.6`, satellites held at 0.8/0.5, blank PHASEOUT):
- Triggered by gain in ≥ 2/3 and a remaining deficit (field_hard completion < 0.85, or walking-hold tracking < 0.7) in
  ≥ 2/3.
- **The trigger stops and asks.**

## 7. Head-of-record rule (stated before judging)

- **Candidates:** the A1 heads, plus a sibling only if the user kept it.
- **Intermediate checkpoints:** screened at n = 100. A candidate that beats its head on field_hard completion by more
  than max(0.10, 2·band) gets an env_seed-2 read. It replaces the head only if the margin holds at n = 200 with floors
  held.
- **Ranking:**
  1. floors (PASS or COST);
  2. obstacle rank-sum: field_hard, field_light and ring_obst completion, and the field rows' move_turn-hold survival;
  3. the phase-1 whole-task rank-sum, with |ratio − 1|, if obstacle rank-sums are within 2;
  4. jitter and stand stepping;
  5. turn band.
- **Three judges.**
- **Bake (user's call):** records + policy folder, adding `6_p2a_field/`. If the winner is not s18, stages 1–5 are
  re-baked from its lineage.

## 8. Arms, order, kills

| Arm | Seeds | Differs from A1 | GPU-h |
|---|---|---|---|
| A1 field (main) | 18 (gate), 16, 17 | — | ≈ 10.5 |
| A0 ring control | 18 | GEOM ring (existing course, no pits) | ≈ 3.3 |
| S1 smoothness | 18 | ACTION_RATE_W −0.05 | ≈ 3.3 |
| E1 energy | 18 | POWER_W at the §4 weight, POWER_JOINTS actuated | ≈ 3.3 |
| F1 stand fix | 18 | STAND_STANCE_W at the §4 weight | ≈ 3.3 |

**Order:** A1 s18 (gate), then A0 s18, S1, E1, F1, then A1 s16 and A1 s17.

**Live kills** (iterations counted from the fork start; each stops the queue and asks):
- **Resume read:** rc 3.
- **Wiring at +200:**
  - clock_hide true; top-level `curriculum: null`; clock/apex 0.001; action_rate per arm;
  - DR and element weights present; the arm's geometry sub-terrains present; reach_obst > 0;
  - no dropped PPO keys;
  - **overlay audit:** every §3 key is realised in that arm's `params/agent.yaml` / `env.yaml`, including the final
    PROMOTE values;
  - **term liveness:** each arm-specific term's `Episode_Reward` is non-zero and within 3× of its §4 replay magnitude,
    normalised to the same per-max-episode-second basis. F1's is non-zero only in stand-slot envs.
- **Terrain at +1000:**
  - the mean level on obstacle tiles has risen above its start;
  - promotion counts are non-zero;
  - the mean level is not pinned at 0 or at the top level.
- **Collapse:** rolling-200 episode length < 0.5× the source tail, or crab_failure > 0.6, after +300.

**Budget:** ≈ 29 GPU-h, including ring_obst at env_seed 2 on every arm head (+≈ 10 if p2b runs).

## 9. Log

- 2026-10-06, **pre-registered.**
  - The plan was approved after a 3-pass audit.
  - User decisions:
    - a 1.7 m pad, kept with the corrected exposure numbers;
    - ramps now, walls built for later;
    - arms S1, E1, F1, pits and an A0 ring control at s18 only;
    - F1 success = a low tap.
  - The audit's implementation checks are folded into §3–§8.
- 2026-10-06, **charter corrected before any run.** An independent fable check of CHARTER vs plan found 6 material and
  3 minor transcription issues, all fixed:
  - **Code-state sidecar:** code_sha / eval_sha / dirty in place of the git HEAD, so record commits do not invalidate
    stamps.
  - **Overlay audit:** added per arm at +200.
  - **Discrimination window:** reverted to the plan's [0.15, 0.85], with flat completion reported.
  - **F1 replay:** ρ ≤ −0.5 on steps/s is decisive; clearance and tripod are reported.
  - **Second calibration smoke:** ≥ 300 its, with dump audit.
  - **Terrain kill:** the "mean level has risen above its start" clause is restored.
  - **F1 keep rule:** requires A1 s18 to still march, so that an effect can be attributed. If A1 already taps, F1 is
    not recommended.
  - **Ground rules block:** added.
  - **ring_obst:** runs at env_seed 2 on every arm head.
  - **Intermediate replacement:** confirmed at n = 200.
- 2026-10-06, **tooling review applied; readings logged before any run.** An independent review of the campaign
  tooling found one tooling defect and two launch dependencies. No run had been launched and no result read.
  - **F1 term name, fixed:** the F1 term is `reward_stand_stance_quality` (`motion_knobs.STAND_STANCE_TERM`). The
    live reader had a wrong name, which would have killed F1 at +200. A test now pins the S1, E1 and F1 term names to
    `motion_knobs`.
  - **Launch dependency, obstacle-tile telemetry:** §8's +1000 terrain read and the CAL2 "levels must move" audit
    accept only these exact tags:
    - `Metrics/base_parkour/terrain_levels_obst`, the mean of `terrain_levels` over `is_obst_tile` envs;
    - `Metrics/base_parkour/promote_count`, which is `move_up.sum()` per reset.

    Neither tag is emitted in the tree yet. The all-tile `terrain_levels` is never accepted in their place. Adding
    them is a code change outside this campaign's paths, so it lands in the §1a commit and the anchors follow it.
    Until then, `run_arm.sh` refuses every arm and CAL2 before launch (exit 7, using `p2_live.py --preflight`) rather
    than killing them at +1250. CAL1 needs neither tag.
  - **Reading, F1 "non-zero only in stand-slot envs" (§8):** no per-slot reward telemetry exists. When per-slot
    telemetry is logged, it must read 0. When none is logged, the clause holds by construction if both of these hold:
    - the dumped term's `func` is `parkour_isaaclab.envs.mdp.rewards:reward_stand_stance_quality`;
    - every `return` of that function in the code root is a product that includes the name bound to
      `command_term.stand_mask()` (an AST check).

    This is checked at preflight and again at +200. If either check fails, F1 fails.
  - **Launch dependency, replay magnitudes:** the §4 read writes `replay_gate.json` as
    `{"E1": {"episode_reward": x}, "F1": {"episode_reward": y}}`, on rsl_rl's `Episode_Reward` basis (per
    max-episode-second). Preflight refuses E1 and F1 without it, and §8 term liveness compares against it.
  - **Readings the CHARTER left open (the builder's, logged per the precedence rule):**
    - §6.4 "side feet_ss drops by ≤ max(0.02, band)": §5 defines no feet band, so the 0.02 m/s floor alone applies.
    - "Jitter" is the mean over fwd, back, side, diag, turn and move_turn of `by_category.<c>.action_jitter.median`.
    - §6.4 "nothing loses" also covers the obstacle completions, at field_hard 0.05, field_light 0.03 and ring_obst
      0.05 (each taken as max(tolerance, band)).
    - The move_turn-survival band floor uses the binomial SE at the mean of the two env_seed reads, n = 200.
    - The completion band floor uses the binomial SE at the pooled p, n = 200.
  - **Reading, §7 item 5 "turn band":** `turn_band__omni` is not evaluated. The `field_turn__omni` turn holds stand
    in: the mean of turn_l90 and turn_r90 for facing_closed and yaw_progress_rate, ranked as a turn rank-sum. Its
    completion and fall share are reported with them.
  - **Tightened checks:**
    - INVALID EVAL also compares the field_suite and field_turn stamps (env, row env block and sidecar).
    - `--anchors-check` also requires every anchor stamp's `run_meta.checkpoint_sha256` to equal `anchors.yaml`.
    - §6.3 reports terrain levels side by side, for all tiles and for obstacle tiles when logged. Each shows the mean
      of the fork's first 20 logged iterations and of its last 100.
    - Calibration also accepts a per-episode `track_ratio` dump (= `ep_track_dist/ep_cmd_dist`).
  - **Kept as built:** the §7 intermediate reads, including the seed-2 screen, also run on A0 and the siblings. This
    is faithful to §3 and costs only eval time.
- 2026-10-06, **obstacle-tile training telemetry landed (with the field code, §1a commit); no run launched, nothing
  read.** `OmniParkourEvent` now logs, on ring AND field runs (telemetry only: no cfg field, so the ring config dump
  is unchanged key for key):
  - `Metrics/base_parkour/terrain_levels_obst`: the mean of `terrain_levels` over the envs on obstacle tiles
    (`name != parkour_flat`), read at each reset call before any tile moves. It is left out of a reset's log when no
    env is on an obstacle tile.
  - `Metrics/base_parkour/promote_count` and `demote_count`: the up / down decisions applied in that reset call to
    episodes that ended on an obstacle tile (0 under `freeze_terrain_levels`). rsl_rl averages the reset calls of an
    iteration, so a logged value is promotions per reset step, and it is > 0 iff any promotion happened.
  - `Metrics/base_parkour/track_ratio_obst_p50`: the median per-episode `ep_track_dist/ep_cmd_dist` (the ratio
    `promotion_mask` gates on, with its `ep_cmd_dist < min_cmd_dist_m` exclusion) over the ledger window's last
    `exposure_window` (1024) obstacle-tile episodes. Non-finite ratios are dropped, and the key is absent until the
    window holds one episode.

    The window is the ledger's rolling one, not a single reset call: one step ends only a few obstacle-tile episodes
    (estimated, not measured), and the iteration mean of such tiny medians would be a mean in all but name.
  - The §8 tags `p2_live.py` expects are unchanged, and its preflight now finds both keys. The exit-7 refusal for
    missing obstacle telemetry no longer fires; the E1/F1 replay-magnitude requirement still applies.
  - **Operational definition of §5 step 2, "the median per-episode ratio on field tiles":** r̄ = the mean of
    `track_ratio_obst_p50` over CAL1's last 100 logged iterations, read from its tfevents by
    `p2_read.py --calibrate <CAL1 run_dir>`. On a field run the obstacle tiles are the field tiles. Fewer than 100
    finite logged iterations fail closed. A per-episode dump is still accepted.
