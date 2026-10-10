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
    code_sha `dfcc53107d66797d…` (163ce038; first landing 345884d6 at 7b39a86d, moved once more by the pit gap before any anchor), eval_sha `1d021625844bb372…` (unmoved).
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
- 2026-10-06, **landed** in 345884d6: code_sha 7b39a86d… (moved once, as planned), eval_sha 1d021625… (unmoved).
  - **Tests:** 431 passed in the phase-2 and regression suites.
  - **Replay gate (§4): E1 FAIL and F1 FAIL,** independently rechecked; numbers in REPLAY_GATE.md.
    - E1 best ρ: stand +0.21, 0/6 moving categories.
    - F1: the statue beats every head (0.78 vs 0.41/s), but within-head ρ with steps/s is +0.25 for B.
    - **Stop and ask the user;** neither arm is dropped automatically.
  - **Preflight now refuses E1/F1 unless `replay_gate.json` shows PASS or a recorded user overrule.**
- 2026-10-06, **user decisions on the §4 replay-gate FAILs:**
  - **F1 overruled:** it runs at weight 0.90. The within-head FAIL is read as range restriction, since every B stand
    episode marches alike; the statue earns 0.78/s against the head's 0.41/s. Recorded as `overrule` in
    `replay_gate.json`; F1 is set to 0.9 in `candidates.yaml` (copied to `scripts/candidates_v2.yaml`).
  - **E1 dropped.** Its chain keeps its placeholder, so it can never launch. The queue's sibling loop is S1 → F1.
  - **Budget:** ≈ 25.5 GPU-h.
- 2026-10-06, **pre-launch audit applied to the tooling (B1, M1–M5); no run launched, nothing read.** Only
  `scripts/` and `tests/unit/test_omni_phase2_prelaunch.py` changed, so code_sha and eval_sha did not move.
  - **B1, smokes before anchors.** `scripts/p2_smoke.py` runs S1–S4 in one headless process. It builds the env the
    p2a arm trains on, composed by run_candidates' own `reference_recipe` → `stage_env` → `rp._subenv`.
    - A1's PROMOTE placeholder takes CAL1's launch value 0.45:0.25 for the smoke, and the substitution is recorded.
    - S5 is `run_arm.sh --row-eval SMOKE_S5_s18 … field_light__omni 1 --episodes 7`, read by
      `p2_read.py --smoke-s5`.
    - `p2_read.py --write-smokes-ok` writes `$OUT/SMOKES_OK` only after S1–S5 and the CAL1 fork smoke's dump audit
      pass on the live code state. It appends the S3 toe radius and the S4 numbers to this log.
    - `p2_queue.sh` refuses to start, from any step, without a valid SMOKES_OK.
  - **M1, code-state guard.** `p2_live.py --code-guard` compares the live code_sha, eval_sha and dirty flag with the
    sidecar of the s18 anchor's suite stamp. Before any anchor stamp exists it requires dirty = false. `run_arm.sh`
    runs it after preflight in run_ref and at the top of every row eval and lr eval. A refusal exits 7 with STOP +
    NOTIFY. `P2_REANCHOR=1` skips only the comparison, for the first row of a deliberate re-anchor.
  - **M2, CAL2 gate.** Before P2__A1_s18, `p2_read.py --cal2-gate` requires P2__CAL2_s18's record to be a smoke that
    was not killed, with a PASS dump audit and PROMOTE equal to the A1 chain's value.
  - **M3:** a DR event dumped as `null` now reads as absent.
  - **M4:** REPLAY_GATE.md's preflight sentence is corrected, and the queue header no longer lists E1.
  - **M5:** after a driver infra retry, the watcher follows the retry's `<tag>_r<k>_train.log`.
  - **Readings the plan left open (the builder's):**
    - **S2:**
      - "corners populated" means all four corners beyond r 4.5 m populated on ≥ 80 % of field tiles, per tile.
        The per-corner share is reported beside it.
      - Coverage is the occupied share of the whole tile.
      - d is the tile's curriculum-row difficulty.
      - "pits ≤ 0" means every tile with pits has sunken occupied pixels, and none is deeper than
        pit_depth(d) + 0.02 m.
    - **S3:** terrain levels are frozen through the resets (the launch levels). The toe radius is the footpad
      origin's horizontal distance from the root, over t = 0 and the 1-s zero-action window. The walking maximum from
      S4 is reported only.
    - **S4:** block steps are the ledger's `block_steps_mean` (per obstacle-tile episode). The head runs in the A1
      env. Precondition: STD_PARAM, INIT_STD, CLOCK_HIDE, SPEED_ELLIPSE and FACING_OBS_CLIP equal the anchor stamp's.
      These were checked on CPU and are equal.
    - **S5:** a stand fall is a fall whose last segment is the prologue or the stand hold.
  - **CPU replica of S2, not a smoke result.** It ran the real decorated `omni_field_terrain` over 400 tiles at the
    generator's 10 difficulty rows (0.20–0.70) with roughness. It predicts that S2, as written, FAILS three
    sub-checks:
    - Heights reach formula + 0.031 m. The spline roughness overshoots to ±0.035 m.
    - Edge-mask pixels appear on 23–37 of 40 tiles per row from d ≈ 0.42. These are risers over 0.12 m where a raised
      block meets a pit.
    - All four corners are populated on 34 % of sparse and 81 % of dense tiles. The per-corner share is 78 % and 95 %.

    The pad check is borderline: off-block roughness reaches |h| 0.035 m. No threshold was changed. If the smoke
    confirms, it stops and asks.
- 2026-10-06, **pit/raised gap and S2 recalibration (CPU only, before any GPU smoke; nothing launched, nothing
  read).**
  - **Pit/raised 2-px gap** (`omni_terrains.field_layout`). Every pit keeps ≥ 2 clear pixels (0.16 m, Chebyshev, so
    diagonals count) from every raised pixel, and every raised block the same from every pit. A centre that breaks
    it is redrawn (the centre only), as for the pad.
    - **Why:** ramp mode must be ramp-only. Where a raised block met a pit, the step was block + pit + roughness.
      That crossed the 0.12 m wall trigger from d ≈ 0.31 and made 0.13 m walls at d ≥ 0.42 (the CPU replica above:
      edge pixels on 23–37 of 40 tiles per row).
    - **After:** 0 edge pixels on 4000 CPU field tiles over the 10 rows. A unit test pins the real decorated
      function's ramp-mode edge mask EMPTY at all 10 rows × 40 seeds × both densities, roughness on.
    - **RNG stream:** bars and slabs are drawn before any pit, so their draws are the pre-gap stream exactly (pinned
      by a test). A pit that lands inside the gap draws more centres. From there, the tile's later pits shift, and
      under the global `numpy.random` so do its roughness and every later tile.
    - **Pit retries:** a pit gets 4 × `max_tries` centre draws. At 1×, 1 % of dense tiles lost a pit; at 4×, 0 of
      1000. The 0.4:0.4:0.2 mix is fully placed up to d = 1.0 (test).
    - **Beyond the shipped range:** from d ≈ 0.8, one block's own step plus roughness can still cross 0.12 m, with or
      without pits. Roughness reaches 0.035 m, and neighbour differences reach 0.06 m. This is out of reach at
      difficulty_range (0.20, 0.70). The tests pin emptiness through d 0.75, and the gap itself up to d 1.0.
    - The ring and forward goldens are byte-identical (tests pass).
  - **S2 recalibrated from the real generator on CPU** (`p2_smoke.py --calibrate-s2 40`). That is 40 seeded replicas
    of the generator × 100 field tiles = 4000 tiles: the real decorated `omni_field_terrain`, roughness on, rows d
    0.20–0.70. Flat tiles are generated in order, so the global stream advances as in the build. Old → new, with the
    CPU value:
    - **Pad:** |h| ≤ 0.035 m → no block occupancy within the pad (exact, from block_masks; CPU 0 px) and |h| ≤
      0.045 m (CPU max 0.035).
    - **Heights:** ≤ formula + 0.02 m → formula + 0.050 m (CPU max excess 0.033).
    - **Pits:** ≥ −(pit_depth + 0.02 m) → −(pit_depth + 0.050 m) (CPU max over-depth 0.0363).
    - **Corners:** all four on ≥ 80 % of tiles → a per-corner rate, pooled per density, of ≥ 0.70 sparse and ≥ 0.90
      dense. CPU: 0.818 and 0.984. Per 100-tile replica (one smoke's worth): 0.770–0.885 and 0.965–1.000. All four
      corners: 0.415 and 0.936, reported only.
    - **Unchanged:** the coverage band (CPU sparse 0.060–0.116, mean 0.084; dense 0.131–0.207, mean 0.164) and no
      edge pixels in ramp mode (CPU 0).
    - **Derivation:** each bound = source + roughness A + rounding steps + 0.005 m. The pad is A + 1 step (the rint).
      Heights and pits are A + 2 steps (block rounding ≤ ½ step, plus the rint).
    - **One deviation from the requested numbers:** with the nominal A = 0.02 m, the bounds are 0.030 m and formula
      ± 0.035 m. But `random_uniform_terrain`'s cubic spline overshoots the ±0.02 m grid to 7 steps: the off-block
      maximum is 0.035 m on 20 % of tiles and 0.040 m on 1 of 4000. At A = 0.02, the CPU replica fails. The pad hit
      0.035 m on 80 of 4000 tiles, so ≈ 87 % of 100-tile smokes would fail. Pits went 0.0363 m deep on 6 of 4000,
      so ≈ 14 % would fail. A is therefore the realised 0.035 m: the pad bound is 0.045 m, and heights and pits are
      ± 0.050 m.
    - `--print-plan` prints each threshold next to its CPU value. The thresholds are also in plan.json as
      `s2_thresholds`.
  - **code_sha moves:** `omni_terrains.py` is in the code pathspec. code_sha moves once more before the anchors. None
    exist yet, so no stamp is invalidated.
- 2026-10-06, **S2 edge check refined** (fable review of the pit-gap change, minor finding 1).
  - Single-block risers (block step + roughness) still reach the 0.12 m wall threshold on ~0.2 % of d-0.70 dense
    tiles. The earlier §9 line calling them "out of reach" in the shipped band was too strong.
  - **Hard check:** no raised-to-pit edge pixel, which is what the 2-px gap exists for.
  - **Other edge pixels:** the check fails only if more than 2 % of field tiles have any. A hard "== 0" would have
    failed about 1.3 % of smokes spuriously.
  - **Pre-existing, recorded:** diagonal-only risers are not in the edge mask.
- 2026-10-06, **code state for the anchors and arms:** code_sha dfcc5310… (163ce038), eval_sha 1d021625… (unmoved).
- 2026-10-06, **smokes S1–S5 PASS** (stamp `smokes/2026-10-06T22-01-35Z`; S5 = `SMOKE_S5_s18`).
  - **S3:** max toe radius 1.07 m, under the 1.6 m limit; 256 resets.
  - **S4:** reach_edge 0.959, field_frac 0.581, 584 block steps per obstacle-tile episode, off-tile block steps 9.6.
    The head's ring numbers are 0.963 / 0.514 / ≈ 230, so the field gives ≈ 2.5× the block contact.
  - **S5:** 7 episodes, 0 stand falls (1 fall in another hold).
  - **Tooling defect found by S5 and fixed:** the harness writes `complete.json` only for category_suite rows, so
    every 8-hold row eval would have been rejected. `run_arm.sh` now writes `p2_complete.json` after a clean exit,
    and the readers accept either marker (tested).
- 2026-10-06 18:18, **CAL1 false wiring kill at +200, fixed and relaunched.**
  - **Cause:** the runner's standard warning that it ignores `share_cnn_encoders` (the documented allow-listed key;
    the baked B c6x log carries the same line) matched the "dropped PPO keys" check.
  - **Fix:** `p2_live.dropped_ppo_keys` parses the key list and allow-lists only `share_cnn_encoders`; any other key
    still kills (self-test).
  - **Every other +200 item passed:**
    - resume read: σ Δ 4e-5, priv_reg 0.1, bound 0.001;
    - clock hidden 1.0;
    - reach_obst 0.77.
- 2026-10-06 18:22, **CAL1 relaunch killed by the resume read** (correctly).
  - **Cause:** the driver keeps an in-flight record per chain, so relaunching the killed `P2__CAL1_s18` continued that
    run from its model_15200 (tag `_c`) instead of forking the head's model_14995.
  - **Fix:** a fresh chain `P2__CAL1b_s18`, identical stage, `candidates_v3`, is now the calibration smoke
    (`p2_read` CAL1 constant).
  - **Rule from here:** any relaunch after a kill uses a fresh chain name (`<chain>b`, `<chain>c`, …), recorded in §9.
    The driver's state is never edited by hand.
- 2026-10-06, **CHARTER 5 smokes passed; `SMOKES_OK` written before any anchor** (`scripts/p2_read.py --write-smokes-ok`).
  - S1-S4: `parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task/experiments/2026-10-07_omni_phase2/smokes/2026-10-06T22-01-35Z`.
  - S3: max toe radius 1.070 m (limit 1.6 m), over 256 resets; walking max (S4) 1.556 m.
  - S4: reach_edge 0.959, field_frac 0.581, block steps per obstacle-tile episode 584.0, off-tile block steps 9.64.
  - S5: `/home/nickmagus/krabby/krabby-research/parkour/outputs/omni_plan/phase2/lr/SMOKE_S5_s18/field_light__omni/seed001/2026-10-06_22-09-32`, 7 episodes, 0 stand-hold falls.
  - CAL1 dump audit PASS. code_sha dfcc5310…, eval_sha 1d021625….
- 2026-10-06 18:36, **promotion calibrated** from `P2__CAL1b_s18`: r̄ = mean `track_ratio_obst_p50` over its last 100
  iterations = 0.824, so PROMOTE = up 0.75 / down 0.40 (`candidates_v4`).
  - CAL1b at the preset 0.45:0.25 confirmed the premise: obstacle-tile terrain level rose 1.1 → 5.5 in 400 its with
    0 demotions. crab_failure rose 0.02 → 0.25 and reach_obst to 0.80.
  - SMOKES_OK written: S1–S5 + CAL1b audit PASS, code_sha dfcc5310. Walking toe radius maximum 1.56 m, inside the
    1.7 m pad.
  - Next: CAL2 (≥ 300 its at 0.75:0.4, levels must move), then the queue.
- 2026-10-06 18:48, **CAL2 PASS** (PROMOTE 0.75:0.4 realised; obstacle terrain level 1.1 → 4.7 in 400 its, promotions 0.18 → 0.36/window, 0 demotions, track ratio 0.81). **Queue launched** (p2_queue.sh from the anchors).
- 2026-10-06 21:08, **STOP: the §5 anchor discrimination check fails** (queue exit 4, as pre-registered: "stop and ask
  before any re-tune"). Inertness PASS (the s18 suite reproduces the manifest).

  | Anchor | flat `slow__omni` | field_light (pooled) | field_hard (pooled; es1 / es2) | ring_obst (pooled) |
  |---|---|---|---|---|
  | s16 | 0.89 | 0.815 | **0.705** (0.78 / 0.63) | 0.695 |
  | s17 | 1.00 | 0.995 | **0.990** (0.99 / 0.99) | 0.985 |
  | s18 | 1.00 | 0.905 | **0.850** (0.86 / 0.84) | 0.805 |

  - **Fails:**
    - s17 field_hard is above 0.85;
    - s17 field_light is not ≥ hard + 0.10;
    - s18 field_light 0.905 is not ≥ 0.85 + 0.10.
  - **Where the falls are** (fall hold = the last hold with steps):
    - Of 91 field_hard falls across the three anchors, 86 are in **move_turn**, the last hold, and all are `pitch_fwd`.
      Only 5 are in the walking holds (s17: 2 diag; s18: 2 back, 1 diag).
    - Per anchor, move_turn falls are 59/200 (s16), 0/200 (s17) and 27/200 (s18). Flat move_turn falls are 10/100
      for s16 and 0 for s17 and s18.
    - ring_obst shows the same pattern: 96 of 103 falls are in move_turn.
    - So on 3–7 cm ramps the walking holds are already solved. "Completion" here measures move_turn survival on bumps.
  - **Gain-test reachability (§6 NO GAIN rule):**
    - s18 needs field_hard ≥ 0.95 (band 0.04): reachable.
    - s16 needs ≥ 1.005 (band 0.15, from its es1/es2 split of 0.78 vs 0.63): unreachable.
    - s17 needs ≥ 1.09: unreachable.
    - So the ≥ 2/3 PASS recipe read cannot be met as built.
  - **Training headroom:** the height formulas top out at 0.09 m bars / 0.08 m slabs at d = 1.0, all below the 0.12 m
    ramp threshold, so no training difficulty reaches a vertical step.
  - **Probe launched while the user decides** (eval-only, not a re-tune, no gate reads it):
    - `probes/scenarios_probe_hard.yaml` + `probes/run_probe_hard.sh`;
    - 3 anchors × {walls at DIFF 0.20:0.70, ramps at 0.70:1.0, walls at 0.70:1.0}, 50 episodes, env_seed 1;
    - stamps under `lr/PROBE_s{16,17,18}/`.
    - It also serves as the first live smoke of `KRABBY_FIELD_EDGES=wall`, which so far was CPU-tested only.
- 2026-10-06 21:42, **probe results** (50 episodes, env_seed 1; completion, with falls by hold):

  | Setting | s16 | s17 | s18 |
  |---|---|---|---|
  | ramps 0.20:0.70 (the anchors; pooled, n = 200) | 0.705 | 0.990 | 0.850 |
  | walls 0.20:0.70 | 0.76 | 0.84 | 0.74 |
  | ramps 0.70:1.0 | 0.68 | 0.94 | 0.76 |
  | walls 0.70:1.0 | 0.42 | 0.56 | 0.40 |

  - **Ramps at maximum difficulty** stay move_turn-dominated: falls are 15/16 (s16), 2/3 (s17) and 8/12 (s18) in
    move_turn. s17 stays above 0.85.
  - **Walls move the falls into the walking holds:**
    - at 0.20:0.70, s17's falls are lat_l 6, back 1, diag 1;
    - at 0.70:1.0, falls are mostly diag and back: 13/29 (s16), 17/22 (s17), 22/30 (s18).
  - **The wall mesh builds and evals cleanly.** This is the first live use of `KRABBY_FIELD_EDGES=wall`.
  - **Edge penalty on walls:** `reward_feet_edge` (−0.3) indexes the global edge mask. On walls it becomes live for
    terrain levels > 3, as in the forward task. Diagonal-only risers stay outside the mask (pre-existing).
  - The decision is the user's (it reverses "ramps now, walls later"); asked with a recommendation.
- 2026-10-06 ~22:00, **USER DECISION: walls, hardest at the top.** This reverses "ramps now, walls later" in light of
  the anchor stop and the probe. **Amendments, all made before any arm has run:**
  - **§3 recipe:** every un-run arm adds `KRABBY_FIELD_EDGES wall` and `KRABBY_FLAT_TERRAIN_DIFF 0.20:1.0` (was the 2a
    preset's 0.20:0.70).
    - A0 (ring) takes the same DIFF. A1 and A0 now differ in two things: the course (field vs ring) and walls vs
      ramps. §6.3 reads A0 as "continue on the phase-1 course" — still descriptive, never a gate.
    - `reward_feet_edge` (−0.3, terrain levels > 3) is live on walls, as in the forward task.
  - **§5 calibration:** redone on walls with fresh chains, `P2__CALW1_s18` (preset 0.45:0.25) and then `P2__CALW2_s18`
    (final values). Every arm's PROMOTE is `TBD_PENDING_CALIBRATION_WALLS` until CALW1 fixes it (`candidates_v5`).
    The ramp-era CAL1 / CAL1b / CAL2 stay as run.
  - **§5 smokes:** S1–S4 are re-run in wall mode at 0.20:1.0. S5 is re-run as `SMOKE_S5W_s18` on the wall
    field_light.
    - S1 expects the composed env's edges and DIFF.
    - S2's edge checks invert in wall mode: every field tile has walls, and no wall pixel sits more than 2 px from a
      block. CPU replica, 10 generators × 100 tiles at 0.20:1.0: PASS, 0 off-block pixels.
    - A new `SMOKES_OK` guard refuses any sentinel whose smoked terrain (S1 summary), S5 row edges or calibration chain
      differs from A1's terrain. The ramp-era sentinel therefore no longer opens the queue.
  - **§6.0 rows** (`scenarios_p2.yaml`):
    - field_light, field_hard, field_turn and field_suite use walls;
    - field_hard moves to DIFF 0.70:1.0, the top of the training range (probe: 0.42 / 0.56 / 0.40);
    - ring_obst stays on ring ramps at 0.20:0.70, as the ramp hold-the-line row;
    - the env-block shape is unchanged.
  - **§5 re-anchoring:** anchors re-run on the wall manifest under fresh names `ANCW_s16/17/18` (`anchors.yaml`). The
    ramp-era `lr/ANC_s*` stamps are kept and no longer read. Discrimination and inertness checks are as pre-registered.
  - **Tooling:**
    - `run_candidates.py` gains a parser and a dump check for `KRABBY_FLAT_TERRAIN_DIFF` (the generator's
      `difficulty_range`);
    - `p2_smoke.py`, `p2_read.py`, `run_arm.sh` and the phase-2 unit tests are updated;
    - 203 tooling tests pass. Five failures in other campaigns' tests (facing ladder, motion ladder, resolution,
      ppo_construction real-dump) touch none of these files.
    - code_sha dfcc5310 and eval_sha 1d021625 are unmoved; tool and campaign files only.
  - **Order:** CALW1 → calibrate → v6 (PROMOTE) → smokes S1–S4 (fill from CALW1) → S5W → SMOKES_OK → CALW2 → queue from
    the anchors.
- 2026-10-06 22:15, **CALW1 (walls) PASS → promotion recalibrated to 0.7:0.4** (`candidates_v6`).
  - CALW1 ran the wall field at DIFF 0.20:1.0 with the preset 0.45:0.25 (run `2026-10-06_21-59-10_seed18_X04_p2a_cal_A15pB`).
  - Reads: resume OK (σ Δ 4e-5, priv_reg 0.1); +200 wiring PASS (reach_obst 0.78, clock hidden); dump audit PASS.
  - r̄ = 0.798 (mean `track_ratio_obst_p50`, last 100 its), so up = 0.9·r̄ → 0.70 and down = 0.5·r̄ → 0.40.
  - At 0.45:0.25 the obstacle level rose 1.0 → 6.5 with 0 demotions; crab_failure reached 0.29; the edge penalty is
    live (−0.005 to −0.019).
  - **Review of 4d0c954b** (fable, read-only): no blocking finding.
    - Fixed: the code-guard anchor-name lookup now fails closed instead of falling back to `ANC_s*`.
    - Fixed: the queue's CAL2 messages now name CALW2 (f006ff4d).
- 2026-10-06 22:18, **wall smokes: S1–S3 PASS, S4 FAIL on one line** (`smokes/2026-10-07T02-14-30Z`).
  - S4: reach_edge 0.943, **field_frac 0.507 < 0.51**, block steps 502 per obstacle-tile episode, off-tile 7.0.
  - S3: toe radius 1.071 m over 256 resets.
  - **Decision (decide-record-notify; a sanity check, not an outcome gate):** in wall mode the field_frac floor is
    0.50; every other S4 floor is unchanged.
    - The 0.51 floor was copied from B's ring telemetry (0.514) as a "no less exposure than the ring" check.
    - On walls the time beyond the pad is 0.007 below the ring's, while the exposure the check exists for, block
      contact, is 2.2× the ring's (502 vs ≈ 230). Walls are meant to slow the head.
  - The smoke is re-run in full; it is deterministic, so S4 reproduces.
- 2026-10-06, **CHARTER 5 smokes passed; `SMOKES_OK` written before any anchor** (`scripts/p2_read.py --write-smokes-ok`).
  - S1-S4: `smokes/2026-10-07T02-18-59Z`.
  - S3: max toe radius 1.071 m (limit 1.6 m), over 256 resets; walking max (S4) 1.733 m.
  - S4: reach_edge 0.943, field_frac 0.507, block steps per obstacle-tile episode 502.3, off-tile block steps 6.96.
  - S5: `/home/nickmagus/krabby/krabby-research/parkour/outputs/omni_plan/phase2/lr/SMOKE_S5W_s18/field_light__omni/seed001/2026-10-07_02-23-19`, 7 episodes, 0 stand-hold falls.
  - CAL1 dump audit PASS. code_sha dfcc5310…, eval_sha 1d021625….
- 2026-10-06 22:38, **pre-launch gates passed on walls; queue relaunched from the anchors** (`wall_chain`).
  - S5W (`SMOKE_S5W_s18`): 7 episodes, 0 stand falls, 1 other.
  - SMOKES_OK was rewritten with `terrain: {difficulty_range [0.2, 1.0], wall: true}`.
  - CALW2 (`2026-10-06_22-25-50_seed18_X05_p2a_cal_A15pB`):
    - resume OK; +200 PASS (reach_obst 0.75);
    - dump audit PASS with PROMOTE 0.7:0.4 realised;
    - obstacle level 1.0 → 5.9 in 400 its, 0 demotions; crab_failure 0.30.
  - The CAL2 gate passed (CALW2).
  - p2_queue started the ANCW anchors at 22:38.
- 2026-10-07 00:59, **wall anchors (ANCW) PASS the §5 checks** (discrimination + inertness).

  | Anchor | flat | field_light (es1 / es2) | field_hard (es1 / es2) | ring_obst | gain needs field_hard ≥ |
  |---|---|---|---|---|---|
  | s16 | 0.89 | 0.765 (0.81 / 0.72) | **0.435** (0.43 / 0.44) | 0.695 | 0.535 |
  | s17 | 1.00 | 0.975 (0.96 / 0.99) | **0.560** (0.59 / 0.53) | 0.985 | 0.680 |
  | s18 | 1.00 | 0.830 (0.79 / 0.87) | **0.500** (0.45 / 0.55) | 0.805 | 0.700 |

  - The gain target is anchor + max(0.10, 2·band). The field_hard band is 0.04 / 0.06 / 0.10.
  - The queue went on to the CALW2 gate (PASS), then A1 s18.
- 2026-10-07 03:10, **A1 s18 trained 5000 its** (`2026-10-07_00-59-10_seed18_X06_p2a_A15pB`, head model_19994).
  - Reads: resume OK; +200 PASS (reach_obst 0.75); +1000 PASS (obstacle level 1.08 → 5.79, promotions logged); no
    collapse kill.
  - Head reads and intermediate reads are running, then the gate read.
- 2026-10-07 04:27, **GATE A1 s18: PASS** (gain, no flat loss, no cost). Pooled completion over es1 + es2, n = 200.

  | Metric | ANCW s18 | A1 s18 |
  |---|---|---|
  | field_hard completion (walls, 0.70–1.0) | 0.500 | **0.745** (needs ≥ 0.70) |
  | field_light completion | 0.830 | 0.995 |
  | ring_obst completion (ring ramps) | 0.805 | 0.995 |
  | fwd / back / side / diag \|ratio − 1\| | 0.041 / 0.030 / 0.045 / 0.044 | 0.007 / 0.026 / 0.045 / 0.031 |
  | turn facing_closed / yaw_progress | 0.797 / 0.104 | 0.928 / 0.128 |
  | move_turn survival | 0.945 | 1.000 |

  - The queue continued to A0 (ring control) as pre-registered; push-notified.
- 2026-10-07 07:50, **FIELD-VS-RING (§6.3, descriptive): A0 (ring ramps, DIFF 0.20:1.0) ≈ A1 (field walls)** on every
  read. Pooled n = 200.

  | Metric | ANCW s18 | A1 (field + walls) | A0 (ring ramps) |
  |---|---|---|---|
  | field_hard / field_light / ring_obst | 0.500 / 0.830 / 0.805 | 0.745 / 0.995 / 0.995 | 0.715 / 1.000 / 1.000 |
  | falls per field_hard episode | — | 0.255 | 0.285 |
  | fwd / back / side / diag \|ratio − 1\| | 0.041 / 0.030 / 0.045 / 0.044 | 0.007 / 0.026 / 0.045 / 0.031 | 0.005 / 0.032 / 0.062 / 0.015 |
  | turn fc / yp; move_turn survival | 0.797 / 0.104; 0.945 | 0.928 / 0.128; 1.000 | 0.896 / 0.118; 1.000 |
  | obstacle level start → tail | — | 1.08 → 6.19 | 1.09 → 5.88 |

  - **Exposure in the field_hard eval** (one estimator for both arms):
    - beyond-pad 0.707 for both;
    - block steps per episode 1285 (A1) vs 1346 (A0).
  - **Reading:** the field+walls course buys at most ≈ +0.03 on walls (inside the noise, SE ≈ 0.03 per arm). Most of
    the phase-2 gain over the anchor (+0.22 to +0.25 on walls, +0.19 on the ring) comes from phase-2 training itself:
    the obstacle curriculum, the 2a DR and 5000 more iterations. A0 never trained on a vertical riser: ring blocks are
    at most 0.09 m, below the 0.12 m wall threshold, so they render as ramps.
  - Not a gate. A0 is not a head-of-record candidate (§7: A1 heads plus kept siblings).
  - The queue continued to S1.
- 2026-10-07 11:19, **SIBLING S1 (action_rate −0.05) vs A1 s18: rule recommends DO NOT KEEP.** Queue stopped
  (§6.4); resume with FROM_STEP=f1.
  - **Conditions:** both hands FAST ✓; side feet_ss drop −0.006 ✓; jitter 0.150 ≤ 0.75 × 0.203 ✓ (−26 %);
    **nothing loses ✗**.
  - **The losses are overshoot.** \|ratio − 1\| worse by 0.043 (fwd) and 0.061 (back). Median tracking ratio is fwd
    1.050 / back 1.087 for S1 vs 0.993 / 1.026 for A1; side 1.051 vs 1.045; diag 1.015 vs 0.969.
  - **Otherwise S1 ≥ A1:**
    - field_hard 0.800 vs 0.745;
    - turn facing_closed 0.958 vs 0.928, yaw_progress 0.158 vs 0.128;
    - diag \|ratio − 1\| 0.015 vs 0.031;
    - move_turn survival 1.0 for both.
  - **Stand:** steps/s 2.83 vs 3.67; lift clearance 0.135 vs 0.090 m; tripod 0.759 vs 0.765.
  - Against the anchor, S1 reads COST (the same fwd / back overshoot). n = 1, paired with A1 at the same seed.
- 2026-10-07, **USER: hold the queue** after the S1 verdict, to look at S1 first. The S1 keep decision is open; F1,
  A1 s16/s17 and the recipe read wait. Resume with `FROM_STEP=f1 scripts/p2_queue.sh`.
- 2026-10-07 ~11:50, **USER DECISION: KEEP S1** (action_rate −0.05), overruling the rule's DO NOT KEEP. The user judged
  the stride metrics worth the 5–9 % fwd / back speed overshoot.
  - **Suite medians, A1 → S1:**

    | Hold | tripod | steps/m | taps/m | swing apex (m) | work/m |
    |---|---|---|---|---|---|
    | fwd | 0.53 → 0.74 | 12.4 → 9.6 | 17.5 → 12.3 | 0.091 → 0.139 | 1232 → 1066 |
    | back | 0.62 → 0.73 | 14.5 → 11.1 | 21.9 → 14.4 | 0.089 → 0.135 | 1287 → 1143 |
    | side | 0.69 → 0.75 | 15.0 → 12.3 | 12.6 → 7.6 | 0.069 → 0.085 | — |
    | diag | 0.65 → 0.76 | 13.6 → 11.6 | 17.2 → 11.2 | — | — |

    - move_turn tripod 0.60 → 0.71.
    - Contact-switch rate is down in every category (e.g. fwd 2.47 → 1.94 Hz). Action jitter is down 23–28 % per
      category.
  - **Status:** per §6.4 a kept sibling is a proposal: S1 s18 joins the §7 head-of-record candidates. The p2a recipe is
    not changed by this alone.
  - **Videos** (requested, then stopped by the user once the stride metrics decided it): partial A1 suite-mode clips in
    `logs/videos/p2_A1_s18_reset_*` (stand / fwd / lat_l / lat_r / back complete). The wall-field recordings failed on
    a manifest path in the runner script and were not re-run.
  - **Queue resumed:** `FROM_STEP=f1`.
- 2026-10-07 ~12:05, **USER DECISION: the replicates run the S1 recipe at s16 / s17** (instead of A1). Amends §6.2,
  §6.5, §7 and §8 before any replicate has run.
  - **Chains** (`candidates_v7`): `P2__S1_s16` / `P2__S1_s17` = A1 s16 / s17 + `KRABBY_ACTION_RATE_W -0.05`, the one
    knob S1 s18 changes (tested). They fork the same heads. `P2__A1_s16` / `_s17` stay defined and are not queued.
  - **Queue:** the `a1rep` step (name kept for FROM_STEP) runs P2__S1_s16 then P2__S1_s17, with the same head reads.
    `p2_queue.sh` was replaced atomically while F1 ran; the running instance exits at F1's sibling stop.
  - **§6.2 recipe read and §6.5 p2b trigger:** on S1 at s16 / s17 / s18 (`p2_read.RECIPE_ARM`), with the same
    per-seed ladder against each seed's ANCW anchor.
    - Expected reading: S1 s18 is COST against its anchor (the fwd / back overshoot the user accepted). If the
      replicates repeat that cost, the pre-registered reading is TRADE-OFF (≥ 2/3 gain with the same COST), not
      ADOPTABLE. The user has accepted that trade.
  - **§7 candidates:** the S1 heads at every seed, plus A1 s18, plus any `--kept` sibling.
  - F1 stays a one-knob contrast against A1 s18, as pre-registered.
- 2026-10-07 ~12:40, **Review of cb53eb37** (workflow wf_b1fbf4b2: 3 lenses, fable adversarial verify; 4 minor confirmed,
  5 refuted, nothing blocking). Fixed:
  1. **Verdict table:** `REPLICATED_ARMS = (S1,)`, so the never-queued A1 s16 / s17 no longer show as PENDING. E1 now
     reads NOT RUN (dropped), not PENDING.
  2. **CHARTER 7 intermediate replacement:** the rule (n = 200 margin + floors held) was computed but never applied
     in the judging path, and the queue discarded the m-screen output. This predates cb53eb37.
     - `p2_read.charter7_candidates` now substitutes a REPLACES HEAD checkpoint (tested).
     - `--json` / `--md` carry the m-screen read and the candidate list.
     - The queue logs the n = 100 m-screen and, after any env_seed-2 row, the n = 200 verdict (NOTIFY).
  3. **Single-instance lock:** `p2_queue.sh` takes a flock on `$OUT/p2_queue.lock` and refuses a second instance
     (exit 3, tested).
     - The queue file was replaced atomically again. The running F1 instance holds the old inode and exits at F1's
       sibling stop.
     - FROM_STEP=a1rep is launched only after that stop.
  4. **Correction to the ~12:05 entry:** S1 s18 is COST against its anchor on **err_back only** (+0.057). The fwd
     overshoot is a loss against A1, not a COST against the anchor (+0.009).
     - TRADE-OFF therefore needs err_back COST at ≥ 2 of the 3 seeds.
     - If the replicates show gain without that same cost, the reading is ADOPTABLE (≥ 2/3 PASS) or MIXED.
- 2026-10-07 15:24, **SIBLING F1 (stand-stance income 0.9) vs A1 s18: rule recommends DO NOT KEEP.** Queue stopped
  (§6.4); resume with FROM_STEP=a1rep.
  - **Low-tap conditions fail:**
    - stand clearance 0.093 m (A1 0.090; needs ≤ 0.05);
    - stand tripod 0.789 (A1 0.765; needs ≤ 0.4).
    - The march persists: steps/s 3.67 → 2.75 and contact switches 2.19 → 1.81 Hz, but lifts stay full height.
  - **Losses vs A1:**
    - err_back +0.051, err_side +0.041; ratios back 1.077, side 1.086;
    - field_hard 0.695 vs 0.745, field_light 0.985 vs 0.995;
    - turning and move_turn unchanged (fc 0.927, mt_surv 0.995); jitter unchanged (0.203).
  - **Against the anchor:** NO GAIN on the queue's read; field_hard 0.695 vs the 0.70 target.
  - Push-notified; asked the user.
- 2026-10-07, **USER: hold the queue** after the F1 verdict. The F1 keep decision is open; the S1 s16 / s17 replicates
  and the recipe read wait. Resume with `FROM_STEP=a1rep scripts/p2_queue.sh` (single-instance lock).
- 2026-10-07, **F1 vs S1 gait comparison** (user request; descriptive). Suite medians at n = 100 per category, from the
  driver suite stamps; A1 s18 is the shared parent recipe.
  - **Walking: S1 has the cleaner gait in every direction.**
    - Tripod: fwd 0.74 vs 0.60, back 0.73 vs 0.68, diag 0.76 vs 0.71, move_turn 0.71 vs 0.69; side equal (0.75).
    - Taps/m: fwd 12.3 vs 15.5, diag 11.2 vs 15.9.
    - Swing apex: fwd 0.139 vs 0.114 m.
    - Work/m about 12 % lower: fwd 1066 vs 1235, back 1143 vs 1289, diag 1005 vs 1136.
    - Slip/m lower; jitter about 25 % lower in every category.
    - F1 also lengthened fwd strides vs A1 (steps/m 10.5; S1 9.6; A1 12.4). F1's side cadence is fast (step rate
      3.4 vs 2.8 Hz), with overshoot 1.09.
  - **Turning and move_turn: F1 is worse.**
    - Turn: fc 0.927 vs 0.958; work/rad 2851 vs 2263; slip 0.121 vs 0.098.
    - move_turn: ratio 0.725 vs 0.807; facing_closed 0.61 vs 0.75; work/rad 4197 vs 3022.
  - **Stand: neither stands still.**
    - Both step about 25 % less often than A1 (2.75 / 2.83 vs 3.67 steps/s), at tripod 0.79 / 0.76 and 0.46–0.47 Hz.
    - F1 lifts like A1 (0.093 m); S1 lifts higher (0.135 m). S1 is smoother at stand (jitter 0.215 vs 0.309).
    - Both spend more work standing than A1 (286 / 309 vs 251).
  - **Caveat:** one run each, forked from the same head; the knob effect and run-to-run divergence are not separable.
- 2026-10-07, **USER DECISION: F1 dropped** (not kept; not a §7 candidate). **Queue resumed:** `FROM_STEP=a1rep`, so
  the S1 s16 / s17 replicates run, then the recipe read.
- 2026-10-07 ~19:40, **S1 s16 (replicate) reads complete: PASS against ANCW_s16** (gain, no flat loss, no cost; hands
  FAST, no hold falls).

  | Metric | ANCW s16 | S1 s16 |
  |---|---|---|
  | field_hard / field_light / ring_obst | 0.435 / 0.765 / 0.695 | **0.830** / 0.980 / 1.000 (gain needs ≥ 0.535) |
  | fwd / back / side / diag \|ratio − 1\| | 0.011 / 0.025 / 0.024 / 0.034 | 0.004 / 0.041 / 0.039 / 0.053 (all within 0.03) |
  | turn facing_closed / yaw_progress | 0.763 / 0.100 | 0.963 / 0.161 |
  | move_turn survival | 0.740 | 0.995 |
  | jitter | 0.195 | 0.145 |

  - **m-screen:** no intermediate beats the head (field_hard at n = 100: 0.77 / 0.85 / 0.82 vs head 0.86).
  - **Gait vs ANCW s16:**
    - tripod fwd 0.53 → 0.70, back 0.62 → 0.72, diag 0.69 → 0.77, but side 0.79 → 0.71;
    - fwd steps/m 11.7 → 10.5, taps/m 16.0 → 13.7, swing apex 0.100 → 0.130 m;
    - fwd work/m 1142 → 1242, unlike s18 where it fell;
    - speed ratio fwd 0.996 / back 1.041, so the s18 overshoot did not reappear;
    - stand steps/s 3.50 → 2.50.
  - S1 s17 is training; the recipe read follows it.
- 2026-10-07 22:37, **RECIPE READ (§6.2, on S1): ADOPTABLE.** 2/3 PASS (s16, s17); s18 COST (err_back only); gain at
  3/3; no HAND LOSS; no shared cost.

  | | ANCW → S1, s16 | ANCW → S1, s17 | ANCW → S1, s18 |
  |---|---|---|---|
  | field_hard | 0.435 → 0.830 | 0.560 → 0.730 | 0.500 → 0.800 |
  | field_light | 0.765 → 0.980 | 0.975 → 0.985 | 0.830 → 0.995 |
  | ring_obst | 0.695 → 1.000 | 0.985 → 1.000 | 0.805 → 1.000 |
  | turn fc / yp | 0.763 / 0.100 → 0.963 / 0.161 | 0.714 / 0.095 → 0.948 / 0.145 | 0.797 / 0.104 → 0.958 / 0.158 |
  | move_turn survival | 0.740 → 0.995 | 0.985 → 0.990 | 0.945 → 1.000 |
  | jitter | 0.195 → 0.145 | 0.199 → 0.157 | 0.198 → 0.150 |

  - S1 s17's |ratio − 1| is within margin: side 0.061 vs anchor 0.033 (+0.028 < 0.03).
  - m-screen: no intermediate replaces a head at any seed.
  - **p2b trigger (§6.5) FIRED:** gain 3/3 and a deficit at 3/3 (field_hard < 0.85 everywhere). The queue stopped
    to ask, and the user was notified.
  - **§7 judging started** (3 judges + fable check) on candidates S1 s16 / s17 / s18 and A1 s18.
    - Inputs: `judging_inputs.json` / `.md`.
    - Mechanical obstacle rank-sum: S1 s18 6.5, S1 s16 10.0, A1 s18 11.5, S1 s17 12.0.
- 2026-10-07 ~23:00, **§7 JUDGING: S1 s18, unanimous** (3 opus judges and a fable check; workflow wf_7cc1da08;
  `judging_result.json`). The check recomputed every input from the stamps: no arithmetic or rule errors.
  - **Ranking:** S1 s18 > S1 s16 > A1 s18 > S1 s17.
    - Step 1: all four hold the floor (PASS or COST, applied as a gate).
    - Step 2: obstacle rank-sum 6.5 / 10.0 / 11.5 / 12.0. The leader is more than 2 ahead, so step 3 does not
      decide first place.
  - **Robustness: fragile.** The 3.5-point lead comes from near-ceiling rows: field_light 0.995 vs 0.980 and
    field_suite move_turn 1.000 vs 0.995, 1–3 episodes in 200.
    - field_hard, the one obstacle read with real spread, favours S1 s16 (0.83 vs 0.80). That gap is inside the bands
      and smaller than each head's own es1 / es2 split.
    - S1 s16 also leads phase-1 tracking (14 vs 18.5), jitter, stand stepping and turning.
    - A tie-aware reading, or reading PASS as ranking above COST, would pick S1 s16. That needs a stages 1–5
      re-bake on the s16 lineage.
  - **Noted for the record:** the §7 rule has no tie tolerance. "Floors (PASS or COST)" was applied as a gate. The
    field_suite move_turn reading was the pre-registered one (scenarios_p2.yaml notes, 4d0c954b).
  - Head of record, bake and p2b were put to the user.
- 2026-10-07, **USER: don't bake yet; plan p2b first.** No head of record is named for phase 2 yet. The p2b arms
  (fork heads, overlay, gates) are drafted for review before anything trains.

## 10. Stage p2b (pre-registered 2026-10-08, before any p2b run)

The approved, audited plan is copied verbatim to `P2B_PLAN.md`: 2 audit passes; the last change (terrain-relative lift)
was not re-audited. That file is the pre-registration. Its decisive rules, restated:

- **Arms** (stage `p2b`, base 2a, 5000 its, each forked from its seed's S1 p2a head model_19994, S1 overlay verbatim):
  - **C0** (s18 only): no change, a p2a continuation.
  - **CL** (s16 / 17 / 18): `KRABBY_FOOT_CLEAR_W 1.0`, `_MIN 0.03`, `_FLAT 1`, `KRABBY_SWING_MIN_CLEAR_W −0.4`, plus the new
    `KRABBY_FOOT_CLEAR_TERRAIN 1` (terrain-relative foot height). The replay gate may re-dose; one weight per term
    across seeds.
  - **TG** (s16 / 17 / 18): new `KRABBY_ORIENT_ALL_TILES 1` (`reward_orientation` → `reward_orientation_upright`, all
    tiles), with `KRABBY_ORIENT_W` from the replay band.
- **Replay gate (CPU, before training):**
  - CL: income rises with terrain-relative apex, pit swings included; counterfactual crouch payoff ≤ 10 % of the lift
    payoff per cm; foot-clearance income 1–10 % of tracking; swing-min penalty ≤ 5 %.
  - TG: pre-fall cost ≥ 5× walking; mean cost 1–5 % of tracking.
  - A failure stops and asks.
- **References `REF_S1_s{16,17,18}`:** the S1 p2a heads re-evaluated on the new code, with field_hard at env_seeds 1–4
  (n = 400). They must reproduce their p2a reads exactly; otherwise stop and ask.
- **Ladder per arm and seed vs REF:** INVALID / PENDING / HAND LOSS / DOSE / FLAT FAIL (vs REF, and vs ANCW through
  chained suite deltas) / NO GAIN / COST / PASS.
  - **NO GAIN** unless all three hold:
    - field_hard (n = 400) ≥ REF + max(0.05, 2·SE), with SE = √(p_REF·q_REF/400 + p_arm·q_arm/400);
    - field_light ≥ REF − max(0.03, band);
    - ring_obst ≥ REF − max(0.05, band).
- **Attribution (s18):** a lever beats continuation only if its REF verdict is a gain, it beats C0 by the same SE
  margin, and it has no FLAT FAIL against C0. s16 / s17 reads are "lever + continuation".
- **Stops:** after the s18 screen (C0, CL, TG), stop and ask (the user may prune); after the replicates, stop and ask
  with the recipe reads (§6.2 rules).
- **Head of record:** §7 unchanged (no tie tolerance). Candidates: the REF heads plus every p2b head reading PASS or
  COST, C0 included. Bake is the user's call.
- 2026-10-08, **p2b step 1 landed.** Two code knobs, both default off and config-identical when unset:
  - `KRABBY_ORIENT_ALL_TILES` swaps `reward_orientation` to `reward_orientation_upright`.
  - `KRABBY_FOOT_CLEAR_TERRAIN` swaps both clearance terms to terrain-relative twins in `mdp/omni_clearance.py`, using
    the same tile lookup as smoke S3.
  - Both refuse to arm when every swapped term has weight 0.
  - **Tooling:**
    - dump checks for all seven p2b knobs;
    - run_arm fork verification for p2a heads, plus the `P2_GUARD_REF` code-guard reference;
    - p2_live p2b liveness/preflight and checkpoints 21000 / 22500 / 24000;
    - p2_read p2b reads;
    - p2_mech (the diagnosis method as a tool: pit 35 / leg strike 35 / trailing 16 / block 10 / none 4 % on the S1
      falls, median onset 1.72 s);
    - p2b_queue.sh;
    - candidates_v8 (p2b chains; CL / TG weights TBD until the replay gate).
  - Implemented via workflow wf_0a765082 and reviewed by fable. Two pre-run blockers were fixed: guard re-pointing and
    the chains. Minor items were fixed or recorded here. 696 tests pass; self-tests pass.
  - **Recorded clarifications of §10:**
    - **Chained ANCW floor:** link 2 is (S1 p2a driver suite − ANCW), the same checkpoint and code as the p2a ladder,
      because the inertness check compares row evals only.
    - **TG liveness:** requires `Episode_Reward/reward_orientation` ≥ 2× the seed's p2a tail. The planned 1–5 % band
      sits ≈ 3–14× above that tail.
    - **Replay magnitudes for liveness:** signed, in TensorBoard units (`Episode_Reward`: weighted episodic sum /
      max episode seconds).
    - **Inertness:** any REF difference stops and asks; the eval process-state flip hazard is noted.
- 2026-10-08, **p2b REPLAY GATE: CL PASS, TG PASS** (`REPLAY_GATE_P2B.md`, `replay_gate.json` p2b; workflow
  wf_59a05294; a fable check agreed on every check and number).
  - **Data:** 4200 S1 episodes (field_hard / field_light at env_seeds 1–2, plus the suite), 139 falls.
  - **Tracking income** (named before reading): 1.25 · track_lin_vel_xy_exp, std 0.2.
  - **CL (terrain-relative lift):**
    - income vs terrain-relative apex: ρ +0.97 on pit and ordinary swings, at every seed;
    - crouch payoff: exactly 0 (the root-relative form paid 100 % of a lift);
    - foot-clearance share 3.4 %; swing-min share 1.7 %;
    - weights unchanged: FOOT_CLEAR_W 1.0, SWING_MIN_CLEAR_W −0.4. The clamp is MAX_CLEARANCE_M 0.26.
  - **TG (tilt guard):**
    - pre-fall vs walking tilt cost 12–15×;
    - share at −0.7 was 0.38 %, so it was re-dosed to **ORIENT_W −2.3** (1.0–1.3 %; feasible |w| 2.25–8.57).
    - It concentrates on move_turn (4–5 % there).
  - **Liveness expectations** (TensorBoard units): CL +0.0267 / −0.0130.
    - TG is read at its calibrated value of −0.0241: the replay's −0.0112 × 2.15, the training/replay factor measured
      on flat-tile tilt, 1.7–2.9× per head.
    - This was set before any TG launch because the fable check flagged a likely false +200 kill at s18.
  - `candidates_v9` fills the weights. The p2b queue had started at 01:08 with the REF re-evals, on code e3fe70fb.
- 2026-10-08 03:40, **REF re-evals done; INERTNESS PASS.** All 9 read rows of REF_S1_s16 / s17 / s18 equal their p2a
  reads exactly; field_hard env_seeds 3–4 are new.
  - The screen's first launch (C0) was refused by the driver's code_sha check (dfcc5310 → e3fe70fb, the intended p2b
    step-1 move). No driver state changed.
  - Fix: run_arm.sh passes `--accept-code-change "$ACCEPT_CODE_WHY"` when set. The queue is resumed `FROM_STEP=screen`
    with it, under the same chain names, since the driver never started.
- 2026-10-08 14:30, **p2b s18 SCREEN: all three NO GAIN; no lever beats continuation.** Queue stopped as pre-registered
  (`p2b_screen.json`, `p2b_mech_screen.json`). The read was only seen at 15:20, because the watcher was not re-armed
  after TG's milestone (operator slip).

  | s18 head | field_hard (n = 400) | falls / 400 (pit, leg strike, move_turn) | stand steps/s, clearance (suite) | fwd tripod, apex |
  |---|---|---|---|---|
  | REF_S1_s18 | 0.8125 | 75 (29, 23, 5) | 2.83, 0.135 m | 0.74, 0.139 |
  | C0 continuation | 0.785 | 86 (43, 33, 28) | 2.67, 0.113 | 0.70, 0.132 |
  | CL terrain-relative lift | **0.695** | 122 (46, 49, 31) | 1.42, 0.077 | 0.64, 0.112 |
  | TG tilt guard | 0.8275 | 69 (19, 27, 14) | **1.08**, 0.104 | 0.74, 0.159 |

  - **Gain gate** (REF + max(0.05, 2·SE) ≈ 0.867–0.873): none passes.
  - **Attribution vs C0:** CL −0.09; TG +0.0425, needing +0.056.
  - **No FLAT FAIL** anywhere, including the chained ANCW floor.
  - **CL:**
    - It raised neither walking apex on walls (diag 0.052 vs 0.058 m) nor flat apex (0.112 vs 0.139).
    - It added leg strikes and pit falls; flat tripod fell from 0.74 to 0.64.
    - Training lift income was 2.8× the replay's prediction, so the term was live but did not produce lift where it
      counts.
  - **TG:**
    - pit falls 29 → 19; total falls lowest of the four;
    - flat stand stepping 2.83 → 1.08 steps/s (−62 %), an unplanned effect on the open stand-march problem;
    - walls +0.015 vs REF, within noise.
  - **C0:** continuation alone slightly hurts walls (move_turn falls 5 → 28).
- 2026-10-08, **USER: STOP p2b here.** No replicates; p2b is closed with no lever adopted.
  - CL hurt; TG gave no wall gain at s18. TG's stand-stepping cut (−62 %) is recorded as a lead for the open stand-march
    problem.
  - **§7 candidates:** the REF heads plus p2b heads reading PASS or COST, of which there are none. So the candidates are
    the S1 p2a heads, as already judged (unanimous S1 s18; S1 s16 tie-aware).
  - The bake decision returns to the user.
- 2026-10-08, **S1 s16 vs S1 s18 breakdown** (user request before the bake decision). Obstacle rows from the REF
  re-evals on the current code; flat rows from the p2a suite.
  - **Walls tie at n = 400:** 0.815 vs 0.8125, with per-seed values 0.86 / 0.80 / 0.80 / 0.80 vs 0.79 / 0.81 / 0.84 /
    0.81. The earlier 0.83 vs 0.80 was n = 200 only. field_light is 0.98 vs 0.995 and ring 1.0 for both.
  - **Falls on walls** (74 vs 75):
    - s16: move_turn 27, back 18, diag 16;
    - s18: diag 40, back 13, move_turn 5.
  - **Flat speed tracking: s16 is better.** Ratios fwd / side / back / diag 0.996 / 1.039 / 1.041 / 0.947 vs 1.050 /
    1.051 / 1.087 / 1.015.
  - **Gait: s18 is better.**
    - tripod fwd 0.74 vs 0.70, side 0.75 vs 0.71;
    - fwd steps/m 9.6 vs 10.5;
    - fwd work/m 1066 vs 1242 (−14 %).
  - **move_turn: s18 is better.** Ratio 0.81 vs 0.72, facing_closed 0.75 vs 0.64; s16 has 27 move_turn falls on walls.
  - **Turning ties** (fc 0.96 / 0.96).
  - **Stand:** 2.5 vs 2.8 steps/s.
  - **Lineage:** s18 adds `6_p2a_field/` only. s16 re-bakes stages 1–5 from the B s16 line, whose phase-1 head was the
    weakest: flat completion 0.89, move_turn survival 0.74.
- 2026-10-08, **USER DECISION: bake S1 s18** as the omni phase-2 head of record (`policy/6_p2a_field/`, model_19994),
  after the s16-vs-s18 breakdown. Stages 1–5 are unchanged (same s18 lineage). Bake in progress (workflow
  wf_c70ed026, then a fable review).
- 2026-10-08, **BAKE APPLIED:** omni phase-2 head of record = S1 s18, `policy/6_p2a_field/model_19994.pt` (sha256
  968cc550…c23c; byte-identical to the X08 run's model_19994).
  - `manifest.yaml`: `current: 6_p2a_field`; `heads: {phase1: 5_c6x_clockfree, phase2: 6_p2a_field}`; the stage-6
    recipe by `overlay_ref`, evals of record, selection and replicates.
  - Follow-ups recorded: move_turn, stand_march (TG lead), p2b_negative, back_overshoot.
  - README regenerated; `bundle_policy --check` OK. Tests: 390 pass with the .pt staged.
  - code_sha e3fe70fb and eval_sha 1d021625 unchanged.
  - Done via workflow wf_c70ed026; fable review found nothing blocking, and its minor notes were applied.
- 2026-10-08, **campaign write-up:** `PHASE2_REPORT.md` (workflow wf_376b2642). The fable fact-check found no numeric
  errors. Its 9 citation, attribution and overstatement notes were applied: §10 citations for the 2026-10-08
  entries; the E1 and TD drops attributed to the user; the back-overshoot cause hedged (n = 1); C0's drop inside
  noise. **Phase 2 is closed.**

## 11. Stage p2s: stop the head stepping at rest (pre-registered 2026-10-08, before any p2s run)

The plan of record is `P2S_PLAN.md`: audited in 2 passes (the cap) and approved by the user on 2026-10-08. Its
verdict rules, replay gate, guards and budget are the pre-registration. This section logs the run.

**User decisions (2026-10-08):**
- **Target "mostly still"** on the suite stand at both env_seeds: median ≤ 0.5 steps/s and lifts ≤ 5 cm
  (`stand_clearance`, median over feet).
- **Lever:** a stand-slot-only term pair:
  - T1 `reward_stand_support` = g·(Σ clamp(F/50 N) − 3);
  - T2 `penalty_stand_slide` = g·Σ s_f·|v_f|.
  - Two doses at s18 (ST2 first), then replicates. TG replicates are not run.
- **Cost: none beyond noise.** Flat ladder against REFS and chained against ANCW; walls non-inferior.
- **Walkstop** (stands after walking) is a COST guard only, not part of STILL.
- **Home:** this campaign, §11.

### 11.1 Log

- 2026-10-08, **TG stand "lead" reclassified as a hover artefact** (verified on the saved suite raws, X14 TG
  `suite__omni/seed001`).
  - All 9 of TG's zero-step stand episodes hold 3 feet unloaded for > 95 % of the 12-s hold. It is the same tripod
    (FR/ML/RR), lifted during the settle. The harness counts a step only on a lift-off inside the window, so these
    episodes read 0 steps/s.
  - TG's unloaded-foot share (< 50 N) is 0.487, against REF 0.455 and C0 0.468.
  - TG's median T1 is ≈ +0.06 (about 3 loaded feet), against REF ≈ +0.28.
  - Under the p2s definition (unloaded AND > 3 cm above the pooled loaded-foot stance offset, > 95 % of the hold),
    TG has 18 hover episodes, CL 4, and REF / C0 / A1 / F1 0.
  - **Consequence:** stand steps/s alone can be gamed by hovering. The p2s read needs off-ground share, a hover count
    and taps. The manifest `stand_march` lead and the PHASE2_REPORT lead line are corrected in a separate record
    commit, with the user's nod.
- 2026-10-08, **implementation note:** the p2s rows (`stand_only`, `walkstop`) go in a new campaign manifest,
  `scenarios_p2s.yaml`, so that `scenarios_p2.yaml` stays byte-identical for the REFS / C0R inertness check.
  - Step 2's noise probe uses the shared `suite__omni` row instead of a new stand-only row. Its stand wave is the
    same 100 × 12-s hold, and REF's deterministic suite es1 stamp (`lr/REF_S1_s18/suite__omni/seed001/
    2026-10-08_05-11-48`) is then the same-row, same-window f denominator.
- 2026-10-08 21:41–22:04, **step 2 noise premise probe** (≈ 0.4 GPU-h, code e3fe70fb, `suite__omni` es1;
  `p2s_noise_probe.json`). Runs:
  - fresh zero-action statue (`DIAG_STATUE_s18`): STILL on every clause;
  - REF with `--sample-actions` (`DIAG_REFSAMP_s18`);
  - the noisy statue, a model_19994 copy with the actor output zeroed, sampled (`DIAG_NOISYSTATUE_s18`).

  Sampled σ: cams 1.0–1.3, rods 0.6–0.75.

  | stand (median) | steps/s | taps/s | clearance | off-ground | loaded feet | T1 | T2 | falls |
  |---|---|---|---|---|---|---|---|---|
  | statue (det) | 0 | 0 | 0.000 | 0.00 | 5.0 | 2.00 | 0.095 | 0 |
  | REF (det) | 2.83 | 2.83 | 0.135 | 0.34 | 3.26 | 0.28 | 0.258 | 0 |
  | noisy statue (sampled) | 4.42 | 19.1 | 0.026 | 0.03 | 3.82 | 0.87 | 0.320 | 2 |
  | REF (sampled) | 2.58 | 3.25 | 0.103 | 0.34 | 3.12 | 0.13 | 0.223 | 0 |

  - **f at ST2's planning dose (1.12, −1.25) = 0.707 / 2.131 = 0.332.** The rule stops only below 0.33, so this is a
    **marginal PASS**. ST1's dose gives 0.316.
  - f is a lower bound: the open-loop statue cannot correct its own noise.
  - **The slide term T2 reverses under noise.** The noisy still stance moves its loaded feet more than the noisy
    march, so at ST2 T2 favours the march by ≈ 0.12/s. T1 alone keeps f = 0.43.
  - Proceeding per the rule. The T2 reversal is flagged for the step-4 replay and the step-6 read.
- 2026-10-08, **step 1 landed:** `scripts/p2s_stand.py`, the one stand-estimator set, with 20 unit tests, and
  `p2s_baseline.json`.
  - Self-check PASS: the E6 statue reads STILL on every clause; TG has 18 hover episodes; steps/s reproduces the
    harness exactly.
  - Suite stand baselines (steps/s, clearance, off-ground, hover):

    | head | steps/s | clearance | off-ground | hover |
    |---|---|---|---|---|
    | REF es1 | 2.833 | 0.135 | 0.338 | 0 |
    | REF es2 | 2.833 | 0.133 | 0.334 | 0 |
    | C0 | 2.667 | 0.113 | 0.341 | 0 |
    | CL | 1.417 | 0.077 | 0.352 | 4 |
    | TG | 1.083 | 0.104 | 0.401 | 18 |
    | A1 | 3.667 | 0.090 | 0.270 | 0 |
    | F1 | 2.750 | 0.093 | 0.347 | 0 |

  - Field-row opening stands: REF 2.75 steps/s, 0.13 m; C0 2.4, 0.12 m; TG 0.4–0.7, 0.04 m.
  - Onset medians: 0.15–0.16 s for REF, C0 and TG.
- 2026-10-08, **step 3 code landed (e1a08baf, code_sha moves):** `reward_stand_support` and `penalty_stand_slide` in
  `mdp/crab_hex_omni_motion_rewards.py`; knobs `KRABBY_STAND_SUPPORT_W` (≥ 0) and `KRABBY_STAND_SLIDE_W` (≤ 0) in
  `motion_knobs.py`; dump checks; 45 unit tests.
  - Unset, the config is byte-identical.
  - 269 related tests pass.
  - 13 failures elsewhere (`test_crab_hex_omni_gate`, the dose-smoke tests, `test_crab_hex_omni_motion_ladder`, one
    ppo_construction dump test) reproduce on the unchanged HEAD. They predate this change.
- 2026-10-08, **step 4 replay gate: PASS** (`REPLAY_GATE_P2S.md`, `replay_gate.json` "p2s"; workflow wf_a076c5b0;
  fable check CONFIRMs it, re-derived by hand).
  - **Fidelity:** the landed kernels equal `p2s_stand`'s T1/T2 to 2e-7.
  - **B** (the existing-term still-minus-march gap, S1 weights, named before the solve) = **+0.304/s**.
  - **G1:** AUC 1.000 for T1; ≥ 0.967 for T2 (worst TG).
  - **G2:** TG −0.909 / CL −0.532 (CL at the line).
  - **G3:** passes at both doses. ST2: hover 2.31 (needs ≥ 1.25), shuffle 0.654 (needs ≥ 0.625), crouch 0.
  - **G4:** 0 non-zero values on 2.6 M non-stand steps.
  - **G5:** finite on the wall stands; 3–4 % of feet over pits in `field_turn` stands.
  - **Anti-termination** (net stand-slot reward):
    - ST2: REF det +2.13 / +2.15, REF sampled +1.87;
    - field_turn walls: REF +2.07, C0 +2.01, TG +1.95.
  - **Final doses** (pooled means): **ST2 (1.1, −1.25)**, **ST1 (0.33, −0.5)**, giving realised margins 2.56 / 1.00.
    These are carried in candidates v11.
  - **Reported finding: T2 reverses under sampling** (+0.123/s for the march at ST2). T1 carries the sampled margin.
    - **f at the final weights is 0.298 (ST2) / 0.272 (ST1) on pooled means, under the 0.33 line.** On step 2's median
      convention it is 0.330 / 0.301. f with T1 alone is 0.387.
    - Step 2's rule passed at 0.332 on the median convention. The drop is the convention switch plus rounding, not new
      evidence. Because it sits under the line on one convention, **the ST2 launch waits for the user** (the plan's
      "stop and ask" applies). The references run meanwhile (`UNTIL_STEP=inertness`).
- 2026-10-08, **step 3 tooling landed** (workflow wf_a076c5b0, fable code review: nothing blocking):
  - `scenarios_p2s.yaml` `walkstop__omni`: flat, fwd → stand → lat_l → stand → back → stand, giving 3 stands and
    2 onset pairs (the last stand ends the episode).
  - candidates v10 → **v11** (replay doses).
  - `p2_live.py` p2s:
    - liveness against the sampled replay magnitudes;
    - the stand-only AST check;
    - σ guard (kill at 0.8×);
    - termination: episode length kills at 0.9× against C0, and 0.85× against the s16/s17 constant source tail
      (review fix: a pure continuation dips to 0.925 there); crab_failure +0.08 is notify-only;
    - preflight: replay PASS plus the exact solved weights.
  - `p2_read.py` `--p2s-*`: the ladder, attribution, inertness and recipe.
  - `p2s_queue.sh`: `UNTIL_STEP` added. The ST1 branch now STOPs on a non-final STILL read; previously a reader error
    silently skipped ST1 (review fix).
  - `run_arm.sh` hooks, with its own CHECKP2S error flag.
  - The implied Σs_f liveness reads the realised `stand_frac_actual` when logged (review fix: the pinned 0.15
    compressed the band).
  - Tests: 478 pass (p2s, p2b and phase-2 tooling).
  - Sibling-pair replay on the real logs: 0 kills on 12 pairs (min length ratio 0.920); the notify flag fires on 5 of
    12.
- 2026-10-08 ~23:20, **USER DECISION: train ST2 as solved** (1.1, −1.25) at s18, after the f question: 0.298 on means
  and 0.330 on medians against the 0.33 line, with the T2 reversal under sampling. ST2 launches automatically when the
  references and their inertness read finish clean (the chained launcher, `FROM_STEP=st2`). Any other stop blocks
  it.
- 2026-10-09 01:17, **step 5 references on the new code (ae892cec): inertness PASS.** REFS_S1_s18 and C0R_s18
  reproduce REF_S1_s18 and P2__C0_s18 exactly on every shared row: field_hard at es1–4, field_light, ring_obst,
  field_turn, field_suite, lr, and suite at es1 and es2.
  - Took ≈ 2.0 GPU-h (23:16 → 01:17).
  - New walkstop baselines: REFS 2.83 steps/s, 0.133 m after walking. The march persists after a stop too.
  - **ST2 (P2__ST2_s18) launched 01:18** through the chain (FROM_STEP=st2). Fork sha 968cc550 verified.
- 2026-10-09 02:43, **ST2 notify-only termination flag at +3264** (the run continues, as pre-registered).
  - ST2's rolling-500 crab_failure is 0.277 against C0's 0.191 in the same window: +0.086, over the +0.08 flag line.
    Sibling noise reaches +0.116.
  - Episode length is 1685 against 1802 (0.935×, above the 0.9 kill).
  - σ is 0.837 against 0.828.
  - Earlier windows (+0 to +1166) tracked C0 within noise.
  - Training checks: RESUME, +200 (implied Σs_f 3.51) and +1000 (curriculum 1.1 → 5.7) all PASS. The support income
    has risen steadily, to an implied Σs_f ≈ 4.9 at +2300.
  - The flag is for the user at the s18 screen stop.
- 2026-10-09 04:36, **s18 screen read** (`p2s_screen.json/.md`, `p2s_mech_screen.json`). ST2 (P2__ST2_s18,
  model_24993) = **OBSTACLE LOSS**.

  **Stand outcome: PARTIAL.**
  - Suite es1 / es2: steps/s 0.17 / 0.17 (REFS 2.83); clearance 0.005 m (0.135); off-ground 0 (0.34); hover 0;
    KS 0.40 (1.72); loaded feet 5.5 (3.3); six-down 0.59 (0.008); body height 1.060 (1.068); falls 0. The user's
    target (≤ 0.5 steps/s, ≤ 5 cm) is met.
  - The taps clause fails: 16.1 / 15.3 taps/s against ≤ 1.0. This is **force chatter, not lifting**. Foot z stays
    within ±2 mm while one foot's force swings 0 ↔ 500–900 N between 20-ms steps.
  - Action jitter at stand is 0.355 against REFS 0.215 (+65 %); contact switches 5.4 Hz against 1.9.
  - **The march became a planted buzz.**

  **Walls: field_hard (n = 400) 0.6175 against REFS 0.8125** (and C0R 0.785). The margin was max(0.05, 2·SE 0.062).
  - Falls 153 against 75 (C0R 86): diag 53, move_turn 45, back 29, turn_l90 18. Only 20 fall in a hold's first 1.5 s,
    so this is not onset.
  - **Mechanism: the stand term bled into the walking gait.**
    - Swing apex fell in every hold: ordinary 0.039 m against C0R 0.077, pit 0.044 against 0.084, diag 0.047 against
      0.086, lat 0.048 against 0.093, turns 0.039 against 0.10–0.115.
    - Swings rose to 208 k against 139 k: shorter, more frequent steps.
    - Pit falls 61 against 43, leg strikes 53 against 33.
  - Training shows it from +500: `penalty_excess_feet_contact_forward` (5–6 feet down while moving or turning) is 2×
    C0's (−0.0073 vs −0.0042), as the support income rose. The curriculum level matched C0 throughout.

  **Other reads:**
  - **Onset** (walkstop median): 0.72 s against REFS 0.14 (+0.58 s, over the 0.3-s COST floor).
  - **Flat tracking improved:** err_fwd −0.042, err_back −0.066, err_side −0.041. No FLAT FAIL.
  - field_light 0.96 (REFS 0.995), ring 0.995 (1.0). lr hands FAST / FAST.
  - Notify-only fall flag at +3264, as logged above.
  - **Attribution: NOT ATTRIBUTED** (the walls are below C0R − margin).

  **Consequences:**
  - ST1 is not run: ST2 does not read STILL, per the pre-registered rule.
  - The queue stopped. **STOP and ask** (P2S_PLAN step 6).
- 2026-10-09 ~04:45, **USER DECISION: read ST2's early checkpoints.** model_20500 (+506) and model_21000 (+1006) get
  suite es1, field_hard at es1–4 (n = 400) and walkstop es1, about 1.5 GPU-h, evals only.
  - The question: does an early point keep most of the stillness without the wall loss?
  - Names: `P2__ST2_s18_m20500` and `_m21000`, with the arm's driver-stamp env.
- 2026-10-09 08:37, **ST2 early-checkpoint reads**: suite es1, field_hard at es1–4 (n = 400), walkstop es1; code
  ae892cec; ≈ 0.8 GPU-h.

  | head | its | stand steps/s | taps/s | clearance | loaded feet | body height | field_hard n=400 | back onset |
  |---|---|---|---|---|---|---|---|---|
  | REFS_S1_s18 | 0 | 2.83 | 2.8 | 0.135 | 3.26 | 1.068 | 0.8125 | 0.14 |
  | C0R_s18 | +5000, no lever | 2.67 | 2.1 | 0.113 | 3.18 | 1.073 | 0.785 | 0.16 |
  | ST2 m20500 | +506 | 1.50 | 11.2 | 0.010 | 4.65 | 1.006 | **0.725** | 0.14 |
  | ST2 m21000 | +1006 | 0.83 | 11.1 | 0.009 | 4.76 | 1.033 | **0.700** | 0.36 |
  | ST2 model_24993 | +5000 | 0.17 | 16.1 | 0.005 | 5.51 | 1.060 | **0.6175** | 0.99 |

  - **No sweet spot.** Stand stillness and the wall loss grow together from the start.
  - At +506, with the stand only half fixed, walls had already lost 0.09, past the ≈ 0.06 margin. That point also
    crouches 6 cm.
  - Tap chatter (11–16/s) is present at every point.
  - **Reading:** the stand-only reward reshapes the shared gait (low lift, many loaded feet) at any exposure. A
    lower dose (ST1) is expected to sit on the same curve.
  - STOP and ask.
- 2026-10-09, **USER DECISIONS:**
  - **(1) p2s ends without adoption.** The next step is to **plan a stand-regime route**: a stand mode the policy can
    separate from walking. It gets a new plan, in plan mode with an audit. The head of record stays S1 s18; the p2s
    knobs stay default-off.
  - **(2) The TG correction is committed** to `policy/manifest.yaml` (`stand_march`: the hover artefact, the p2s
    result, the new lever) and the PHASE2_REPORT lead line. The policy README was regenerated (`bundle_policy --sync`);
    `test_policy_of_record_omni` passes.

## 12. Stage p2r: a stand regime (pre-registered 2026-10-09, before any p2r run)

The plan of record is `P2R_PLAN.md`: audited in 2 passes (the cap) and approved by the user on 2026-10-09. Its steps,
stop rules, verdicts and budget are the pre-registration. This section logs the run.

**User decisions (2026-10-09):**
- **A stand regime** instead of more stand-only reward on the shared gait.
- **Probe both, cheap first:**
  - **step 1** (eval only): a controller-level stand mode, (a) a zero-action hold and (b) ST2 as a stand specialist,
    keyed on the latched stand slot, plus a drift probe;
  - **STOP and ask**;
  - **step 2**: graft a stand bit into S1's hidden clock column 15 (`KRABBY_OMNI_STAND_OBS=clock`) + the p2s stand
    terms at the ST2 dose + a walking anchor loss to frozen S1 on non-stand rows. G2 (bit + anchor) first, then G1
    (bit only), with the early anchor-strength check (CPU drift at +500 / +1000 against ST2).
- **Unchanged:** target "mostly still"; no walking or wall cost beyond noise; the p2s ladder and estimators.

### 12.1 Log
- 2026-10-09, **step 1 code landed** (campaign tool files only; code_sha ae892cec and eval_sha 1d021625 unchanged;
  workflow wf_e005a4f5).
  - `scripts/p2r_mode_eval_crab_hex_gait.py`: the stand-mode launcher.
    - It patches `g["main"].__globals__` of the locked harness and closes the app itself.
    - Extra runners are built on a stand-in env, with all RNG state saved and restored.
    - The latched stand slot is armed at runtime and re-derived on reset (the suite-wave carry-over).
    - True priv columns are captured before the harness overwrites them in place.
    - It writes the `run_meta.p2r_mode` block, `p2r_stand_mode` / `p2r_dmu_*` raws and `p2r_bank.npz`.
  - `run_arm.sh`: `P2_EVAL` and `P2_ENV_EXTRA`.
  - `p2_read.py`: `--p2r-probe` / `--p2r-step1`.
  - `scripts/p2r_drift.py`: CPU drift scorer.
  - Tests: 46 + 26 unit tests; drift self-test PASS.
  - Fable review: nothing blocking in the launcher. Two read fixes were applied: a probe-name typo, and mode heads
    are now read by stamp name since they have no record. The hold swap-live proof also runs on walkstop, where it
    is non-vacuous.
- 2026-10-09 13:15–14:50, **step 1 runs** (≈ 1.6 GPU-h, code ae892cec; `p2r_step1.md/.json`, `p2r_drift/`).
  - **Launcher inertness: PASS.** P2R_PROBE_s18 (suite, field_hard, walkstop at es1) is bitwise equal to REFS_S1_s18
    on every raw key, actions included, over 1000 episodes on the suite.
  - Stand mode is exactly the latched slot: share 1.0 on settle and stand holds in every wave, 0 elsewhere.
  - **HOLD swap-live: PASS.** Actions are exactly 0 on every stand-mode step, and pre-entry actions equal REFS on
    suite and walkstop.

  | head | stand (suite es1 / es2) | field_hard n=400 (REFS 0.8125) | walkstop onset | field_turn stand on walls | verdict |
  |---|---|---|---|---|---|
  | **P2R_HOLD_s18** (zero-action hold) | 0 steps, 0 taps, 0 clearance, 5.0 loaded / same | 0.810 (76 falls vs 75, none in stand) | 0.12 s (REFS 0.14) | 0.5 steps/s, 0.4 taps/s, 4.1 loaded, completion 0.99 | **STILL**, no tier-1 COST |
  | P2R_ST2M_s18 (ST2 specialist) | 0.17 steps, **16.1 / 15.3 taps**, 5 mm | 0.835 | 0.14 s | 0.75 steps/s, 12 taps/s, 5.2 loaded, 0.98 | **PARTIAL** (taps) |

  - **HOLD:**
    - flat deltas against REFS are all ≤ 0.009; turn facing_closed 0.956;
    - field_light 0.99 (= REFS); 0 falls in stand mode, no handover falls;
    - the stand does not correct heading drift: stand-hold facing_closed −0.002 against REFS 0.377.
  - **Drift premise (CPU, three 20k banks of non-stand states, anchor path):**
    - D(ST2) at +506 / +1006 / +5000 is 2.28–2.29 / 3.64–3.82 / 4.52–5.05.
    - D(C0) at +506 / +1006 is 1.35–1.42 / 1.63–2.02; D(C0R +5000) is 2.74–3.25.
    - **STOP FIRES:** D(C0 +506) 1.35–1.42 > 0.5·D(ST2 +506) = 1.14. Continuation alone moves the walking means
      about 60 % as far as the stand lever, so an action-mean anchor cannot single out the harmful drift.
  - **STOP and ask** (P2R_PLAN step 3). Step 2 is not started.
- 2026-10-09 ~15:00, **USER DECISION at the step-3 stop: learn the statue in-network.**
  - A new step-2 variant replaces G2/G1: the grafted stand bit plus a stand-side prior toward zero action (the
    statue), instead of ST2's stand terms. The goal is no runtime mode switch.
  - It gets its own plan and audit, as an amendment to P2R_PLAN.
  - The hold mode stays on record as the working controller-level fallback.
- 2026-10-09, **step-2 amendment approved: learn the statue in-network** (`P2R_STATUE_PLAN.md`; 1-pass audit, opus +
  fable, blocking fixes applied; user-approved).
  - **Design:** distil the step-1 hold controller into one network, using two masked μ losses in PPO, each a sum over
    its masked rows ÷ B, in KL units at weight 1.
    - On stand rows (bit = 1): a statue prior ½Σ(μ/σ_A)².
    - On walking rows (bit = 0): a walking anchor ½Σ((μ − μ_S1)/σ_A)².
    - The grafted stand bit (`KRABBY_OMNI_STAND_OBS=clock`, col 15) separates the two.
  - **Arm:** one, `P2__Z1_s18` (S1 overlay verbatim, no ST2 terms, 5000 its).
  - **Early checks** at +500 / +1000 (notify-only): walking drift ≤ ½ C0's; statue formation against its fork value;
    residual plateau.
  - **Budget:** ≈ 4.4 GPU-h.
- 2026-10-09, **statue step 1 code landed** (workflow wf_f53039e0; code_sha moves once).
  - `KRABBY_OMNI_STAND_OBS=clock` writes the latched stand bit into the current obs column 15, never history.
    - It is refused without clock_hide, with a ramp, or with STAND_OBS=1.
    - Mirror +1. The latch re-derives on reset.
  - Two masked μ losses in `ppo_with_extractor.py` (sum ÷ B): anchor (bit 0, frozen graft actor, `hist_encoding=False`)
    and statue prior (bit 1). σ_A is fixed. Grad-norm split every 50 iterations.
  - Knobs `KRABBY_ANCHOR_W` / `KRABBY_ANCHOR_CKPT` / `KRABBY_STAND_PRIOR_W`, with run_candidates dump checks.
  - The post-run clock check is narrowed (col 16 ≡ 0, col 15 = the stand label, with the one-step lag).
  - **Graft built:** `parkour/outputs/omni_plan/phase2/graft/S1_s18_c15/model_19994.pt`. Exactly 9 leaves differ
    (3 weights + 6 Adam moments, col 15 only). The CPU identity check passes.
  - **Tooling:** `run_arm` graft branch; Z-arm `p2_live` preflight and +200 wiring; `p2r_early.py`; `p2_read --p2r-z`;
    candidates v12 chain `P2__Z1_s18`.
  - **Fable review:** one blocker, fixed. The repo-relative `KRABBY_ANCHOR_CKPT` failed from the training cwd
    (`parkour/`); it now resolves against the repo root, and the dump check matches.
  - Also fixed:
    - the Z read's raw-bit INVALID rule now uses the driver's label check, so plain Z evals are checked;
    - the +500 early check fires from the orchestrator's sentinel on model_20500.
  - **Note:** the latch re-derive on reset also changes eval wave boundaries for runs using STAND_HOLD / PARK /
    OBS=1. No head of record uses them, and the inertness bridge runs with them off.
  - **Tests:** 396 related tests pass and the 4 tool self-tests pass. The remaining unrelated failures predate this
    work, or sit in files other sessions are editing.
- 2026-10-09 17:31–18:10, **statue step 2: identity + inertness PASS** (≈ 0.65 GPU-h; new code_sha d9f0022f).
  - **Inertness bridge:** REFB_S1_s18 (new code, knobs off) is bitwise equal to REFS_S1_s18 (ae892cec) on every raw
    key of suite es1, field_hard es1–4, walkstop es1 and field_turn es1.
    - Recorded in `inertness_bridge.json`. `p2_read`'s INVALID rule now accepts exactly that code_sha pair (same
      eval_sha, clean trees).
  - **Graft GPU identity:** GRAFT_S1_s18 with `STAND_OBS=clock` is bitwise equal to REFS on every raw key except
    `clock_obs`, on suite, field_turn and walkstop.
    - `clock_obs[:,0]` equals the stand label on 1,179,914 labelled steps (0 off); col 16 is 0.
    - The GPU stamps are written to graft.json.
  - **Step 3:** doses 1.0 / 1.0 (KL units, sum ÷ B). Expected first-update `Loss/stand_prior` ≈ 2.3; anchor 0.
  - Validator dry run: `P2__Z1_s18` ok. `p2_live --preflight` PASS. **Z1 launches.**
- 2026-10-09 18:11–21:26, **Z1 (P2__Z1_s18, X16 model_24993) trained and read** (≈ 3.3 GPU-h; `p2r_z.md/.json`,
  `p2r_z1_early.json`).

  **Training checks:**
  - +200 wiring PASS: anchor rows 0.84, prior rows 0.16, anchor 0 at the fork; first-update statue loss 1.99
    (expected ≈ 2.3, band [1.2, 4.6]).
  - +1000 curriculum 1.1 → 5.6.

  **Early checks** (both PASS):

  | | walking drift D(Z1) | D(C0) | statue formation (rolling-20 at +200) | stand residual per row |
  |---|---|---|---|---|
  | +506 | 0.023 | 1.35 | 0.0093 (line 0.20) | 0.015 |
  | +1006 | 0.013 | 1.93 | — | 0.010 |

  The cosine of Z1's Δμ against ST2's is −0.25 / −0.19.

  **Read: STILL, no COST, attribution LEVER against C0R.**

  | | REFS | Z1 |
  |---|---|---|
  | stand steps/s (es1 / es2) | 2.83 / 2.83 | 0.00 / 0.08 |
  | taps/s | 2.8 / 2.7 | 0.08 / 0.08 |
  | clearance | 0.135 | 0.0013 |
  | off-ground | 0.34 | 0 |
  | hover | 0 | 0 |
  | KS | 1.72 | 0.18 |
  | action jitter at stand | 0.215 | 0.0 |
  | work at stand | 309 W/s | 43 W/s |
  | field_hard (n = 400) | 0.8125, 75 falls | 0.83, 68 falls |
  | field_light / ring | 0.995 / 1.0 | 0.995 / 1.0 |
  | onset (walkstop median) | 0.14 s | 0.12 s |
  | stand falls | — | 0 / 1500 |
  | body height | 1.067 | 1.064 |
  | field_turn | 0.99 | 1.0 |

  - Flat deltas against REFS are all ≤ 0.017; no FLAT FAIL against REFS or chained against ANCW.
  - Walkstop stands after walking: 0.17 / 0.25 steps/s and 0.33 / 0.42 taps/s (touchdowns on entry).
  - The raw stand bit matches the labels on all 15 Z stamps.
  - The fable check of the read is running. **STOP and ask** (P2R_STATUE_PLAN step 5): replicates and the bake are
    the user's.
- 2026-10-09 ~21:50, **fable check of the Z1 read: CONFIRM** (STILL, no COST, LEVER against C0R). Every gated number
  re-derives from the stamps.
  - The strict stand-bit re-check finds 0 mismatches over 5.96 M labelled steps.
  - The INVALID rule is clean; the only differing sidecar pair is the bridged one.
  - **Minor notes, recorded:**
    - The HOLD fidelity column read None (a `p2r_z_fidelity` read bug: it gates HOLD on a record HOLD never had).
      HOLD's real values are suite 0 / 0 steps, 0 taps, KS 0.10. Z1 is marginally chattier (about one tap per 12-s
      hold, KS 0.18) and comparable after walking.
    - **Early-check timing:** the sentinel's +500 run failed twice (`--bank` missing). The +500 check was run by hand
      during training (≈ 18:37) and the +1000 by the relaunched sentinel. Both PASS. Their JSON was only saved after
      the reads; the per-checkpoint sidecars hold a truncated line.
    - The early-check "reported" tags were null (a `read_tb` extra-tags omission). The real values at +506 / +1006:
      KL 0.012; LR ≈ 5e-5; grad_norm 15; grad split statue / anchor / rest 0.29 / 5.5 / 18.9, then 0.24 / 2.1 / 12.9;
      σ 0.82 → 0.86 at the end. Nothing unstable.
    - **DOSE rung:** skipped by the Z read, but it would pass (median |μ| cam 0.73 / rod 0.87; stand clip share 0 vs
      REFS 0.325).
    - The bridge accept rule is per code_sha pair. It was verified on 7 rows and extrapolated to the others.
- 2026-10-09 22:00–23:11, **Z1 videos** (flat, field_light, field_medium, field_heavy, speed_oob, walkstop; then a
  10-min randomised heavy-walls run). The 10-min run had 49 commands of 5–20 s each (13 stands; speeds 0.04–0.32
  inside the ellipse; any direction or facing).
  - Falls in the 10-min run: 3, all while walking (35 s at move 0.14, 111 s at move 0.22, 265 s at move 0.28). There
    were 0 falls in stands or turns.
  - Videos are in `parkour/logs/videos/p2z1_*` (not committed).
  - **Findings that motivated Z2** (measured on Z1's walkstop raws, 300 stands):
    - **Five feet.** In 55 % of stands a foot carries < 10 N; the gap is a median 6.8 mm. This is the
      static-indeterminacy lottery, not the policy.
    - **Cams not homed.** Cams are velocity-commanded, so μ = 0 freezes them wherever walking left them. Hip yaw is
      4.9–8.7° on average, against a default fan of 0.
- 2026-10-10, **user decision: plan Z2, "all six feet planted and cams homed"**. The plan is `P2R_Z2_PLAN.md`, audited
  2 passes (the cap) and approved; it amends `P2R_STATUE_PLAN.md`.
  - **Design.**
    - **Controller R.** A memoryless scripted stand controller R re-steps one tripod at a time (lift, home the cams in
      the air, seat). It seats each leg on the policy's current-sense contact proxy.
    - **Held proxy.** That proxy becomes a sample-and-hold observation on stand rows (`KRABBY_OMNI_STAND_PROXY=hold`):
      a parked rod reads exactly −0.5, so a still stand is otherwise blind.
    - **Probe, then distil.** R is proven at controller level on S1 (`P2R_RST_s18`), then distilled into one network
      by setting Z1's statue-prior target to R(obs) (`KRABBY_STAND_PRIOR_TARGET=restance`).
  - **User decisions:**
    - re-step by tripod, not homing in place;
    - current-sense load sensing, which carries over to hardware (not the sim-only joint sag, not cams-only);
    - fork from the graft like Z1;
    - a loaded cam inside the home band gets a ±0.03 restoring trim, cancelling Z1's |μ_cam| ≈ 0.019 residual,
      which otherwise drifts out of the band in about 1.6 s;
    - a residual-injection probe pass, plus a CPU stand-fidelity check during training;
    - every R lift and press moves the knee too, so all proxy readings sit on the one knee-inclusive scale.
  - **Pre-registered Z2 gates** (step 5):
    - STILL on the suite;
    - **six down:** ≥ 90 % of stands six-down ≥ 0.90 of the window. Baselines: Z1 walkstop 22–24 %, HOLD 34–35 %,
      REFS 0 %;
    - **homed:** ≥ 90 % of walkstop stands. Z1, HOLD and REFS: 0 %;
    - **settle:** ≥ 95 % of stands settle; median within the probe's + 1 s;
    - post-settle stillness;
    - stand falls;
    - the full p2s ladder.
  - **Budget:** ≈ 5.6–6.1 GPU-h.
- 2026-10-10 ~01:50, **Z2 step 1: code landed** (ad89ba23; code_sha 4d4d833d; eval_sha 1d021625 unchanged; workflow
  wf_f1b12efc). Three opus implementers plus a fable review.
  - **What landed:**
    - `mdp/crab_hex_omni_restance.py`: controller R, mirror-equivariant to 0 error in both planes on random states;
    - `KRABBY_OMNI_STAND_PROXY=hold`: the held proxy, updated only on a full **knee** gate (the one-scale decision);
    - `KRABBY_STAND_PRIOR_TARGET=restance` + `KRABBY_RESTANCE`;
    - `--p2r-mode restance` (+ `--p2r-residual`, `--p2r-stand-bank`);
    - `p2r_z2_dose.py` and `p2r_z2_fidelity.py`, and `p2r_early --mode z2`;
    - the reads: `p2s_stand` per-stand Z2 estimators over all three walkstop stands, `p2_read --p2r-rst` / `--p2r-z2`,
      and `p2_live` Z2 +200 wiring;
    - candidates v13 (`P2__Z2_s18`, with `KRABBY_RESTANCE: default` until the probe fixes the constants).
  - **Tests:** 151 new tests pass. On the related suites, 3419 pass and 5 fail, all environmental: one fails on a clean
    HEAD worktree too; three read files another session is editing; one reads Z1's on-disk agent.yaml.
  - **Real baselines** on the new per-stand statistics (es1 / es2):

    | | six-down ≥ 0.9 share | homed |
    |---|---|---|
    | Z1 walkstop | 21.7 / 24.3 % | 0 % |
    | HOLD walkstop | 34.7 / 33.7 % | 0 % |
    | REFS walkstop | 0 % | 0 % |
    | Z1 suite | 9 / 11 % | 0 % |
    | HOLD suite | 29 / 20 % | 100 % (spawn) |

  - **Decide-record (fable review; two pre-registered numbers the measured data contradicts):**
    1. **The +200 held = live line.**
       - The plan's ≥ 0.99 predates the knee-only validity rule. On the training-noise bank (DIAG_REFSAMP_s18,
         60,000 stand rows) the knee gate is full on 97.6 % of steps, and held = live on 97.95 %.
       - The line is now the dose file's `proxy_live_min` = bank share − 0.02. As written, 0.99 would have killed Z2
         at +200.
    2. **The dose band.**
       - It cannot exclude target-0 at any small R, because S1's marching μ dominates both priors at the fork: 16.57
         (R) vs 16.09 (0) per stand row, a paired Δ of 0.48.
       - Exclusion is reported, not required. The band stays as the +200 range check, and the wiring proof is
         `Loss/restance_nonzero` ≥ 0.5 plus the branch shares.
       - On the bank, R's branch shares are hold 0.14, wait 0.36, lift 0.03, home 0.41, seat 0.06.
    3. **Z2's episode-length guard is notify-only**, per the plan's "Left for implementation": R re-steps through most
       noisy training stands.
  - **Dose-script cross-check:** the proxy rebuild matches recorded obs within 4.8e-7 and the cams bitwise. DIAG
    stamps record SAMPLED actions; on REFSAMP the noise term is subtracted (σ_sample = σ_A).
- 2026-10-10 01:57–02:24, **Z2 step 2: inertness bridge PASS** (≈ 0.45 GPU-h). REFC_S1_s18 runs the new code
  (ad89ba23, code_sha 4d4d833d, every Z2 knob off) and is **bitwise equal to REFS_S1_s18 (ae892cec) on every raw key**
  of suite es1 (1000 episodes), field_hard es1–4, walkstop es1 and field_turn es1.
  - Appended to `inertness_bridge.json`. The read's accept rule chains pairs, so the Z1 code (d9f0022f) and the Z2
    code (4d4d833d) both link to REFS.
  - The guard was re-anchored on REFC for the first row (`P2_REANCHOR=1`, deliberate; the code moved). Later rows are
    guarded by REFC.
- 2026-10-10 02:25–02:40, **Z2 step 3: calibration pass + tuning pass 1** (S1 + R at stand rows; walkstop es1, then
  + suite es1). Reads from `scratchpad z2_calib.py` (live proxy rebuilt from raws; branch occupancy) and the new
  per-stand estimators.

  | | homed (walkstop) | six-down ≥ 0.9 | stand falls | stuck in last 5 s | notes |
  |---|---|---|---|---|---|
  | c0 (seeds: lift −0.5, ρ 0.5, θ_L 0.35, θ_S 0.75, κ 0.5) | 2.9 % | 0.4 % | 14 / 276 | WAIT 38 %, SEAT 30 % | body +1.5 cm |
  | t1 (lift −0.4, **ρ 2.0**, θ_S 0.9) walkstop | 27 % | 9 % | **50 / 231** | LIFT 11 % | — |
  | t1 suite (from spawn) | 100 % | 0 % (median share 0.24) | 0 | SEAT 21 % | **40 taps/s**, KS 0.34 |

  - **Findings:**
    1. **Knee gate closes before the hip finishes.** With the knee-only validity rule, at ρ 0.5 the knee rod ends
       its travel well before the hip, so the held reading froze mid-motion:
       - a lifted leg kept 0.35 "loaded" in the air and sat in LIFT forever;
       - pressing legs kept 0.07 "unloaded" while carrying 240–760 N, and pressed to full extension.
       By the linkage, the knee rod is about 18 mm per action unit at 33 mm/s against the hip's 29 mm at 28 mm/s,
       so ρ ≥ 1.9 keeps the knee moving through the hip's travel. ρ 2.0 it is, which forces lift −0.4.
    2. **R's support rule was missing** (a logic flaw in the plan's rule as written). Any unloaded, un-homed leg
       lifted, so a planted-tripod leg that lost load lifted while the stepping tripod was in the air (traced: FL
       and FR up, pitch to −28°).
       - **Fixed in d5aa8c2e** (decide-record): HOME needs the other tripod fully loaded; otherwise the leg seats.
       - This is a code move, so a short bridge (REFD: suite, field_hard es1, walkstop) runs before tuning pass 2.
    3. **θ_L too low.** In-air readings during a ρ-2 lift reach 0.39–0.49, while loaded legs read 1.0. θ_L → 0.6.
    4. **Chatter at the lottery leg.** The proportional seat law, closed through the gated current reading, chatters:
       suite 40 taps/s and 21 % SEAT. A leg that unloads at zero extension has no stable point.
       - Tuning pass 2 tests a halved gain (κ 0.25).
       - **If chatter persists:** the premise "a memoryless seat law on the held proxy plants six feet stably"
         fails, and step 3's STOP-and-ask comes due (fallbacks: sag signal, cams-only, or a bounded last-action
         press).
- 2026-10-10 02:42–03:05, **short bridge PASS** for the R fix (REFD_S1_s18 at fa4ce645, bitwise = REFS on suite,
  field_hard es1 and walkstop), then **tuning pass 2** (θ_L 0.6, κ 0.25, support rule on):

  | | homed | six-down ≥ 0.9 | stand falls | other |
  |---|---|---|---|---|
  | walkstop es1 | 50 % | 22 % (Z1 22–24 %, HOLD 34–35 %) | 4 / 298 | 116 / 294 stands end in a WAIT + SEAT deadlock |
  | suite es1 (from spawn) | 100 % | 0 % | 0 | **37 taps/s**, KS 0.31 (STILL fails) |

- **STEP-3 PREMISE STOP** (P2R_Z2_PLAN step 3, after 2 of 3 tuning passes): the current-sense seat law cannot plant six
  feet stably. The failure is structural, not a matter of constants:
  1. **Stale holds.** A parked leg cannot refresh its reading.
     - 71 % of the legs stuck in SEAT at stand end are actually loaded (median 494 N) while reading a held 0.10.
     - The last full-gate reading is often knee-only, taken after the hip rod (which carries the load torque) has
       stopped.
     - The stale "unloaded" blocks the other tripod (the support rule) → a deadlock in 116 / 294 walkstop stands.
  2. **Chatter.** When the reading does stay live, the press never settles. SEAT legs' knee rods move 99 % of the
     time, with 23 crossings of 50 N per leg-second, and they shake the HOLD legs (8 / s).
  3. **Overlap.** In-air readings during a ρ-2 lift (p90 0.39–0.49) overlap light-load readings (40–120 N median
     0.57, p10 −0.09).
  - **What works:** the cam homing itself (suite 100 %, walkstop 50 % and limited only by the deadlock) and the
    support rule (falls 50 → 4).
  - **NOTIFY + ask the user** (the fallbacks are theirs).
- 2026-10-10 ~03:15, **user decision: Z2 continues CAMS-ONLY with a KINEMATIC re-step** (no load sensing).
  - **The new R:**
    - **Phase from the hip position:** a leg is "down" when its hip-pitch is near default and "up" near the lift
      target.
    - **Stepping:** a tripod steps only while the other tripod is entirely down.
    - **Cams:** a cam homes only while its leg is up; a down leg in band gets the ±0.03 trim.
    - **Rods:** every target is absolute (lift or 0).
  - **Six-foot loading** stays the static-indeterminacy lottery (Z1 22–24 %, HOLD 34–35 %).
  - **Amendment to P2R_Z2_PLAN** (recorded here and in the plan file's amendment note):
    - The held proxy (`KRABBY_OMNI_STAND_PROXY`) is no longer used by R or Z2. It stays in the code, default off.
    - The six-down gate becomes REPORTED. Settle = homed, held 0.5 s.
    - All other gates stand: STILL on the suite; homed ≥ 90 % of walkstop stands at both seeds; settle;
      post-settle stillness; falls; the p2s ladder; the field rows and body height in the probe's premise gate.
    - A code change, a short bridge, and a fresh probe (calibration + up to 3 tuning passes) come before training.
- 2026-10-10, **Z2 cams-only code landed** (416f6658; code_sha 7ad542de).
  - **The kinematic R:**
    - phase from the hip-pitch position: down ≥ 0.4 × 0.25 × lift, up ≤ 0.6 × 0.25 × lift;
    - a tripod steps only while the other is all down;
    - a cam homes only while up; the ±0.03 trim applies when down and in band;
    - all targets are absolute;
    - mirror-equivariant to 0 error in both planes.
  - **Plumbing:**
    - the held proxy is refused in the probe and the reads;
    - the dose obs come from cams + hip positions, bitwise against recorded obs on 5000 rows;
    - the dose file must carry the chain's own R constants;
    - settle = homed held 0.5 s; six-down is reported.
  - **Fable review:** nothing blocking. 636 targeted tests pass.
  - **Short bridge REFE PASS:** bitwise = REFS on suite, field_hard es1 and walkstop.
- 2026-10-10 10:26–10:36, **kinematic calibration pass P2R_RSTk0_s18** (seed constants: lift −0.5, ρ 1, up 0.6,
  down 0.4). **Every premise clause on es1 passes, so no tuning is needed.**

  | | walkstop es1 (300 stands after walking) | suite es1 (from spawn) |
  |---|---|---|
  | homed | **97.3 %** (Z1 / HOLD 0 %) | 100 % |
  | settled | 97.3 %, median 3.06 s | 100 % |
  | post-settle steps / taps / KS | 0.11 / 0 / 0.14 | 0 / 0 / 0.10 (STILL, every clause) |
  | stand falls | 1 / 300 | 0 |
  | body height | 1.062 (REFS 1.068) | 1.065 |
  | six-down ≥ 0.9 (reported) | 35 % (Z1 22 %) | 29 % (= HOLD) |
  | mean stopped hip yaw | 0.28° (Z1 ~6°) | 0.09° |

  - Of 300 stands, 292 end with every leg in HOLD and 7 with a waiting leg. The re-step takes a median 7 foot lifts.
  - **Constants fixed = the seeds.** Next: the full set as `P2R_RST_s18` (suite and walkstop at es1 / es2,
    field_hard es1, field_turn, stand bank on), then the residual pass `P2R_RSTres_s18`. The residual is Z1's walkstop
    mean stand action, `p2r_z1_residual.json`: cam ≤ 0.019, rod ≤ 0.038.
- 2026-10-10 10:37–11:03, **probe full set P2R_RST_s18** (seed constants, stand bank on). `p2_read --p2r-rst`:
  **FAIL**.

  | | es1 (tuning seed) | es2 (held out) |
  |---|---|---|
  | suite STILL | PASS (0 steps, 0 taps, KS 0.10) | PASS (KS 0.11) |
  | homed | 97.3 % | 95.6 % |
  | settled | 97.3 %, 3.06 s | 95.6 %, 3.10 s |
  | post-settle steps / taps / KS | 0.11 / 0 / 0.14 | 0.13 / 0 / 0.14 |
  | stand falls | 1 / 300 | **4 / 296** (line 1) |
  | body height vs REFS | within 0.007 | within 0.006 |
  | six-down (reported) | 35 % suite 29 % | 30 % suite 20 % |

  - **Field rows (es1):**
    - field_hard: 0 stand falls, completion 0.76 (HOLD line 0.71), PASS;
    - **field_turn: 8 / 100 stand falls, completion 0.90** (HOLD 0 falls, 0.97+), FAIL.
  - **Fall pattern (12 falls traced):** while one tripod is in the air homing its cams (HOME holds for 0.8–2 s), the
    body tips steadily to ±25° on the other tripod. That tripod's feet sit where walking left them, and one of them is
    often not yet planted: "down" allowed the hip within 40 % of the lift, about 3 cm above the ground.
  - **Residual pass:** aborted at launch. The residual JSON carried an extra `what` key, which the loader refuses by
    design. The file is fixed (`what` folded into `source`); the pass reruns on the final constants.
  - **Tuning pass 1 (k1):** down_frac 0.4 → 0.25 (support fully planted), up_frac 0.6 → 0.4 and lift −0.5 → −0.4
    (shorter air time). Rows: walkstop es1 + field_turn es1. The suite is skipped because R from spawn is the statue,
    unaffected by these constants; es2 stays held out.
- 2026-10-10 11:04–11:31, **kinematic tuning passes 1–2** (walkstop es1 + field_turn es1; es2 held out).

  | pass | constants | walkstop homed | walkstop stand falls | field_turn stand falls / completion |
  |---|---|---|---|---|
  | full set (seeds) | lift −0.5, up 0.6, down 0.4 | 97.3 % | 1 / 300 | 8 / 100, 0.90 |
  | k1 | lift −0.4, up 0.4, **down 0.25** | **23 %** (deadlock) | 0 | 1 / 100, 0.98 |
  | k2 | lift −0.4, up 0.5, down 0.4, **tilt guard 0.10 rad** (code e6c82ab2, REFF bridge PASS) | 85 % | 0 | **0 / 100, 1.00** |

  - **k1:** a tighter DOWN band stops the falls, but the hip sag of loaded legs reads as "not down" (21 % of planted
    legs), so the homing deadlocks. Position-based "down" cannot go below the load sag.
  - **k2:** the **tilt guard** (no lifting while |roll| or |pitch| > 0.10 rad; lifted legs come down) removes the
    terrain falls. Flat onset tilt has a median of 3° and a p90 of 5°, against a settled 0.5°, so the guard does not
    block flat homing.
  - The k2 homing loss is the smaller lift: the absolute DOWN band shrinks to −0.04 rad, and 3.4 % of planted legs
    read "not down".
  - **k3 (the last tuning pass):** lift −0.5 (DOWN −0.05, as in the seeds), up 0.5, down 0.4, tilt guard 0.10.
- **Residual pass P2R_RSTres_s18:** aborted at launch twice. The JSON's extra `what` key was refused; the file was
  fixed after the first abort. It reruns on the final constants.
- 2026-10-10 11:31–12:13, **k3 + final probe set + residual pass.**
  - **k3** (lift −0.5, up 0.5, down 0.4, tilt 0.10):
    - walkstop es1: homed 95.7 %, 0 falls;
    - field_turn: 1 / 100 falls, completion 0.98.
  - **Decide-record (b dbb00b15):** the field clause now uses 2 SE of the DIFFERENCE (pooled). HOLD's own SE is
    exactly 0 at its 0 field_turn falls, so 1 / 100 against 0 / 100 "failed"; the pooled 2 SE is 0.020. A real
    regression still fails: 8 / 100 → line 0.055.
  - **Final probe P2R_RSTf_s18 (k3 constants, stand bank on).** `p2_read --p2r-rst`: **FAIL on one clause only.**

    | | es1 | es2 (held out) |
    |---|---|---|
    | suite STILL | PASS | PASS |
    | homed (line 90 %) | 95.7 % | 94.7 % |
    | **settled (line 95 %)** | 95.7 % | **94.67 %** (one stand short) |
    | stand falls | 0 / 300 | 0 / 300 |
    | post-settle steps / taps / KS | 0.12 / 0 / 0.15 | 0.12 / 0 / 0.15 |
    | body height vs REFS | within 0.007 | within 0.007 |
    | six-down (reported) | 35 % | 33 % |

    - **Field rows:** field_turn 1 / 100 falls (line 0.020), completion 0.98 (line 0.956); field_hard 0 falls,
      completion 0.76. PASS.
    - **The unhomed 4–5 % of walkstop stands are level, not tilt-blocked.** Most have 1–2 legs stuck just above the
      DOWN band in SEAT, which holds the other tripod in WAIT; a few are still mid-re-step at the stand's end.
  - **Residual pass P2R_RSTresf_s18** (R + Z1's measured stand residual: cam ≤ 0.019, rod ≤ 0.038):
    - walkstop es1: homed 97.7 %, settled 98.3 %, 0 falls, post-settle 0.12 / 0 / 0.15;
    - suite es1: STILL every clause;
    - the cam trim holds the cams at a median 0.063 rad, the predicted 0.019 / 0.3 equilibrium, inside the 0.10
      band. **PASS** on the rows run.
  - **Under the amendment, settled = homed held 0.5 s**, so the 95 % settle clause tests the same event as the 90 %
    homed gate with a stricter line. The es2 miss is 0.26 SE (n = 300). Tuning passes are exhausted, so per step 3:
    **STOP and ask.**
- 2026-10-10 ~12:20, **user decision: train Z2; the settled line = the homed line (90 %)** for the probe and for
  Z2's step-5 read. Under the cams-only amendment settled = homed held 0.5 s, so the two clauses test the same event.
  The median-settle clause (≤ probe + 1 s) stays.
  - Re-read: `p2_read --p2r-rst P2R_RSTf_s18` → **PASS** ("every evaluated clause holds").
  - **candidates v15:** `P2__Z2_s18` carries `KRABBY_RESTANCE=lift=-0.5,rho=1,up_frac=0.5,down_frac=0.4,k_c=2,`
    `k_h=0.3,trim=0.03,eps_c=0.1,eps_air=0.05,tilt_max=0.1`.
  - **Dose** (`p2r_z2_dose.json`, those constants): rc 0, band [1.047, 5.369], R ≠ 0 on 99.9 % of training stand
    rows, home branch 0.17.
    - On the noise banks, stand-row tilt exceeds 0.10 rad on 0–0.9 % of rows, so the dose rebuild leaving out roll and
      pitch is negligible.
