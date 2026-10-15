# Robot-frame velocity commands for the omni walker — CHARTER

- **Pre-registered:** 2026-10-15, before any run of this campaign.
- **Plan of record:** [`PLAN.md`](PLAN.md) — approved by the user 2026-10-15 after a 3-pass audit (cap reached;
  7 → 6 → 2 blocking; the pass-3 fixes were applied but not re-audited; trail at the top of PLAN.md). Its design
  decisions D1–D7, steps, stop points, pre-registered gates 0–7, budget and seeds ARE the pre-registration; they are
  not restated here.
- **Precedence:** this CHARTER is the record. Any change to a PLAN.md rule, and every number a PLAN.md rule leaves to
  be pinned, is logged in §3 before the result it affects is read.

## 1. Question and outcome

**Question.** Can the omni walker, retrained from scratch under pure robot-frame velocity commands (vx, vy, wz) —
the facing target and its observation removed — turn at a commanded rate across the robot's measured envelope
(smoothly, including slow turns and turning while moving), hold heading on straight commands, and keep the facing
lineage's translation and obstacle performance, at ≥ 2 of 3 training seeds (PLAN gates 0–7)?

**Outcome.** Per-seed gate reads against the anchor (the facing three-seed set S1 s16 / s17 / s18 through the
`body_hold` adapter on the body rows), a stand-network re-check with the selected body walker, and an
adoption-grade report. **Adoption, bake and push are the user's.**

## 2. Ground rules

- **Code:** shared code changes are part of this plan (D1, D2, D4–D6): default-off behind
  `KRABBY_OMNI_COMMAND_FRAME=body`; code_sha and eval_sha each move once, recorded with reasons (eval_sha moves by the
  user's decision: first-class body evaluation). Campaign scripts live in `scripts/` here. Never edit another
  campaign's scripts (copy or wrap).
- **Running:** one Isaac process at a time (detached queue runner + log-sentinel watcher; never `pgrep -f` a string
  in the watcher's own command line); stop with `systemctl --user stop`, never SIGKILL Kit; check the Kit
  DerivedDataCache before the first launch; memory-capped scopes; videos only through the streaming recorder.
- **Git:** commit only this campaign's paths and the shared files the plan names (with their tests); never
  `git add -A`; never push. Other sessions share the checkout: never revert changes this campaign did not make.
- **Authority:** push-notify at every stop point (PLAN "Stop points"); decide-record-notify elsewhere; bake /
  adoption / push are the user's.

## 3. Log

- 2026-10-15, **campaign opened.** PLAN.md approved (audit trail in its header). User decisions recorded in PLAN:
  retrain from scratch; measured envelope; first-class body eval (eval_sha moves); stand net re-check, retrain only if
  needed; fresh stage anchors for the old stages; replicates auto-launch on a pilot pass; video tool support later;
  `track_ang_vel_z_exp` gated to straight rows; "turns better" = smoothness + move-turn; same 1.2× headroom on the
  move-turn wz axis; legacy world rows reported, not gated.
- 2026-10-15, **step 0.** HEAD `bbb4634f`, code_sha `14301cd2f2ba6080…`, eval_sha `1d021625844bb372…` (the
  stand-height campaign's values; unchanged). No Isaac / driver process running (`ps`, no user units).
  `2026-09-28_side_bisect/state.json` and `2026-10-07_omni_phase2/state.json` read `phase: run` but are stale
  (mtimes 2026-09-30 and 2026-10-11, no driver process) — neither campaign is live; their files are another session's
  uncommitted edits and are not touched.
- 2026-10-15, **step-1a rule pinned before the read (PLAN D4 "turn-pricing check").**
  - **Yaw-term set priced:** `reward_tracking_yaw` 0.2 × `exp(−|Δyaw_1s − ∫wz_cmd| / σ)`, σ = 0.05 rad, 1-s window
    = 50 steps at dt 0.02; `reward_yaw_progress` 1.0 × tent `min(p, 2|c| − p)`, p = sign(c)·Δyaw_1s / 1 s;
    `track_ang_vel_z_exp` pays 0 on turning rows (body mode). Weights are S1 p2a's realised ones
    (`params/env.yaml` of `2026-10-07_07-50-10_seed18_X08_p2a_A15pB`: 0.2 / 1.0 / 1.0).
  - **Traces:** REFS_S1_s18 `suite__omni` es1 (`phase2/lr/REFS_S1_s18/suite__omni/seed001/2026-10-09_03-20-24`),
    per-step `root_ang_vel_b[:, 2]` over each hold segment after its first 2 s; translation holds (fwd, lat_l,
    lat_r, back, diag) and move_turn holds (both hands); each trace mean-removed (= wobble). Healthy tracker =
    wobble + c; step-in-place = wobble; still stander = 0.
  - **Levels:** w_min 0.05, 0.08, ceiling 0.167 (S1's measured sustained pure-turn rate from the world suite; step 4
    re-measures it — a ceiling that moves by more than 0.01 re-runs this check), w_max 0.21.
  - **Pass rule:** (i) at the ceiling and at w_max, each non-rotating trace's mean yaw-term income ≤ 10 % of the
    healthy tracker's; (ii) at w_min, each non-rotating trace's share ≤ **B = 0.40**.
    **Why B = 0.40 and not the idealised value:** the idealised-tracker share at w_min is
    0.2·e^(−1) / (0.2 + 0.05) = 0.294 for a wobble-free tracker, and the still stander's income is exactly the
    idealised stander's while any real tracker earns less than a perfect one — so "measured share ≤ the idealised
    share" fails by construction and cannot be the intended check. B is pinned at the top of the range PLAN states
    (≈ 30–40 %) before any measured share is computed; the idealised value is reported beside it.
  - **Recorded, not gated:** the tracker-minus-stander premium (per step and per second) at w_min and 0.08; the
    straight-row (c = 0) yaw income of the wobble trace (heading-hold term + ang-vel term); the 1-s window spread.
  - **A fail → STOP and ask** (PLAN step 1a).
- 2026-10-15, **step 1a read (CPU; `scripts/bc_turn_pricing.py` → `step1a_turn_pricing.{json,md}`): FAIL at w_min →
  STOP and ask.** S1 traces: 500 translation holds, 200 move_turn holds (none terminated); 1-s Δyaw wobble std 0.024 /
  0.034 rad (|·| p50 0.015 / 0.023). Non-rotating share of the yaw-term income (still / step-in-place):
  - w_min 0.05: translation **0.418 / 0.452**, move_turn **0.494 / 0.523** — all > B 0.40 (idealised 0.294);
  - 0.08: 0.196 / 0.219 and 0.226 / 0.255 (recorded); ceiling 0.167: ≤ 0.027 (PASS, ≤ 0.10); w_max 0.21: ≤ 0.010 (PASS).
  - Premium tracker − still: w_min 5.1 / 3.8 per s, 0.08 8.3 / 6.9 per s (translation / move_turn traces).
  - Straight rows (c = 0): S1's wobble earns 1.112 / 1.082 per step of the 1.200 a perfectly still heading-hold earns.
  - Cause: the gait wobble integrates to ≈ 0.015–0.023 rad per 1-s window, a third to half of σ, so a real tracker
    loses ≈ 30 % of the window term and the tent's |W|; the stander's exp(−c/σ) is untouched.
  - **Sensitivity (`scripts/bc_turn_pricing_sweep.py` → `step1a_sweep.md`; descriptive, read after the verdict):**
    w_min 0.06 → move_turn SIP 0.418 (fail); w_min 0.07 → max share 0.328; 2-s window σ 0.10 → max 0.399 (borderline);
    σ 0.03 → 0.403; tent weight 2.0 → 0.431; 2-s window σ 0.10 + tent 2.0 → max **0.333** (premium 8.4 / s at w_min,
    ceiling shares ≤ 0.03 unchanged in kind).
- 2026-10-15, **USER DECISION at the step-1a stop: "2-s window + tent 2.0".** PLAN D4 amended (before any code):
  `reward_tracking_yaw` → `0.2 × exp(−|Δyaw_2s − ∫wz_cmd| / σ)` with a **2-s window** (100 steps) and **σ = 0.10 rad**
  (still σ = w_min × window); `reward_yaw_progress` → tent `min(p, 2|c| − p)` with **p = the 2-s mean yaw rate**
  toward the command, at **weight 2.0** (was 1.0; its top income at the ceiling ≈ 0.33 / step, about the facing
  lineage's yaw_progress cap of 0.3); w_min stays 0.05; window reset on a command change / reset as planned (the
  window integral is taken over the steps since the reset when fewer than 100 have elapsed — implementation pins it).
  Every PLAN text that says "1-s" for these two terms now reads "2-s"; the eval's ripple statistic (std of the 1-s
  mean wz, D5) is an eval statistic and is unchanged.
- 2026-10-15, **step 1a re-read under the amended D4 (`step1a_turn_pricing_v2.{json,md}`): PASS.** Non-rotating
  share (still / step-in-place): w_min translation 0.304 / 0.316, move_turn 0.333 / 0.327 (≤ B 0.40; idealised
  0.245); 0.08: ≤ 0.144; ceiling ≤ 0.016; w_max ≤ 0.006. Premium vs still at w_min 8.4 / 7.4 per s, at 0.08 13.1 /
  12.0 per s. Straight-row wobble income 1.131 / 1.108 of 1.200. (The v1 read's idealised line is regenerated with
  the window length in the formula; its verdict and shares are unchanged.) Next: step 1 code.
- 2026-10-15, **step 1 dispatched (opus·xhigh) on `STEP1_SPEC.md`; pins it makes beyond PLAN, recorded before any
  result is read:** (1) envelope knobs `KRABBY_OMNI_BODY_ENVELOPE=w_min:ceiling:w_max` and
  `KRABBY_OMNI_BODY_CAP=v@wz,...` (required in body mode, refused in world mode; not in the presets — the campaign
  overlay sets them after step 4); (2) the coupled cap is a function of |v| only (piecewise-linear through the P1
  points, constant beyond the last point) — this is the deferred "off-grid bearings" rule: at each |v| the P1 point
  used is the worse of the fwd / lat reads (step 4 pins it when it fits the cap); (3) a move+turn draw whose cap
  falls below w_min has its speed reduced to the largest |v| with cap ≥ w_min; (4) the 2-s window is NOT reset on a
  command change (it compares achieved and commanded yaw over the same trailing steps) and is zero-filled at an
  episode reset — this replaces PLAN's "window reset on a command change / reset" (a reset on change would pay or
  charge the first seconds of every command on a partial window); (5) the tent is clamped below at −w_max and pays
  exactly 0 where the window's mean command is 0; (6) a pool type with a non-unit `ang_scale` is refused in body mode
  (no shipped pool uses one); (7) the clock-radius rule is skipped in body mode (the walk-floor rule keeps the clock
  running on every turning row) — this closes the deferred "clock-radius rule" item; (8) world mode refuses the
  `none` obs encoding.
- 2026-10-15, **inertness baseline:** Kit config dumps at HEAD of the 13 targets in `dump_targets.json` (presets
  1a 1b 1c 2a 2b 2c 2c_clockheld + overlays fb6k c4 c5 c6 c6x p2a), `scripts/bc_dump_cfgs.py` →
  `parkour/outputs/omni_plan/body_cmd/dumps_head/` (13/13 ok). The after-code dumps must match them byte for byte.
- 2026-10-15, **step-2 procedure pinned (`STEP2_SPEC.md`), deviation from PLAN "eval_sha moves once":** the body
  scenario file `eval/scenarios_omni_body.yaml` carries the envelope as a manifest block (turn levels and move_turn
  rows are SYMBOLS resolved from it: `w_min`, `ceiling`, `w_max`, `cap80` = 0.8 × cap(|v|)). It is written with a
  PROVISIONAL envelope at step 2 (0.05 / 0.167 / 0.21, cap (0, 0.21) (0.22, 0.13) (0.30, 0.10)) and rewritten — that
  block only — when step 4 freezes the envelope from P1, so **eval_sha moves twice** (harness at step 2; envelope at
  step 4). Every anchor and every gate read runs at the post-freeze eval_sha; the step-3 harness proof (0
  route / slot / obs mismatches) and the P1 grid row do not depend on the envelope numbers. Body field rows translate
  the world 8-hold timeline: turn holds at ±ceiling for 12 s, move_turn at (0.22, 0, cap80) for 8 s.
- 2026-10-15, **step 1 built (opus·xhigh; report in the session record): 183 new CPU tests; full `tests/unit/` 13
  failed / 4847 passed — the same 13 fail at HEAD (pre-existing, unrelated); no Kit run by the implementer.** Pins it
  made beyond the spec are in its report (window not reset on a command change, zero-filled at reset; tent clamp
  −w_max; body signature (φ, v, wz); hook writes the row verbatim and validates only; envelope with no moving turn
  refused; RSI refused also by the built cfg; `omni_clock_turn_floor` = walk floor 0.21; `_b` presets resume
  `1b_b → 1a_b`, `2a_b` from nothing). Gates that read only ‖cmd_xy‖ (air time, stride, progress, obstacle
  clearance) get no turn floor — the same as world mode, where they already ignore turns.
- 2026-10-15, **inertness check (Kit dumps, `scripts/bc_dump_compare.py`): the 13 knob-off targets after the step-1
  code are IDENTICAL to the HEAD dumps** (`parkour/outputs/omni_plan/body_cmd/dump_compare_head_vs_after.json`).
  Body-preset dumps (`dumps_body/`, provisional envelope) show the body command class + envelope fields, obs encoding
  `none`, the three swapped reward funcs (2-s window, σ 0.10; tent weight 2.0; straight-gated ang-vel), facing-turn
  weight 0, turn floor 0.21, no RSI event.
- 2026-10-15, **DECISION (decide-record-notify): the 2-s windowed yaw term is weighted 0.2 in EVERY `_b` preset**
  (`KRABBY_YAW_W=0.2` joins `BODY_DELTA`). Found in the body dumps: `1a_b` / `1b_b` carry weight 0 because the world
  1a / 1b presets never set `KRABBY_YAW_W` — the facing lineage had no facing reward before c4 because its relay held
  heading (any drift past 0.133 rad commanded a corrective turn). Body mode has no relay, so at weight 0 nothing would
  price slow heading drift in fb6k_b (the straight-gated ang-vel kernel, std 0.5, is flat at 0.01 rad/s). 0.2 is the
  lineage's own weight from c4 on.
- 2026-10-15, **step-1 review (fable·high): no blocking defect.** World path bitwise by inspection on every hunk;
  sampler RNG order, window-term timing vs Isaac Lab's step order (reward before reset before command compute; the
  window pushes the command the policy acted on), history zeros, clock floor seam, knob order and the real
  `configclass` / hydra round trip all checked. Applied: (MINOR 2) non-zero `KRABBY_OMNI_STROKE_W` /
  `KRABBY_OMNI_CENTRING_W` refused in body mode (their own clock speed has no turn floor); the `KRABBY_YAW_W=0.2`
  decision above. Recorded, not changed: (MINOR 1) `2a_b` has no preset `resume_from` (there is no `1c_b`) — the
  campaign's chain stages name every fork explicitly. 3 of the 13 pre-existing test failures are caused by other
  sessions' uncommitted ledger edits (they pass at HEAD + only this change). Committed as `3a4b7bda` (step-1 code +
  tests; code_sha moves here, reason: the body command mode).
- 2026-10-15, **step 2 built (opus·xhigh; report in the session record): 124 new CPU tests; full suite 14 failed /
  4974 passed — the 13 known plus `test_sh_video_tour::test_recorder_pre_kit_path_with_a_stub_kit`, which fails BY
  DESIGN: the closed stand-height campaign's video recorder pins the eval harness's sha256 against SH_T17's reference
  stamp, and the harness changed. Consequence: recording stand-height videos now needs a fresh reference stamp or
  `--allow-reference-mismatch` (left as is; the guard is doing its job).** Pins beyond the spec (recorded before any
  read): body-metric window = hold-relative [4 s, hold_s) ∩ alive / fall-trimmed steps, rates over (n−1)·dt, ripple =
  population std of whole 1-s blocks (None under 2 blocks), `cross_body` in m/s; scored holds need hold_s ≥ 6 s;
  hands must match the sign (l = +wz on turns, +vy on translation); grid cells exempt from the wz / cap checks only,
  body heads refused on the grid; `body_hold` face starts / restarts at the spawn pin; `world_rate` takes ellipse,
  stiffness and dead band from the head's cfg and `ang_vel_max` from the row manifest, `heading_err` recorded against
  the world row; legacy runs gain no run_meta / raw key; adapter flags appended to driver command lines only when
  not `none`; `<x>_body` rows map to `<x>__omni_body`; `_b` presets evaluate `suite_body` (+ `ring_obst_body` for
  `2a_b`). The two adapters use harness-side per-env twins of the term's hooks (tested equal to the real hooks).
  Carry-over noted for step 9: the stand-network launchers' frozen carry lists do not know the frame keys.
- 2026-10-15, **step 3 launched on the uncommitted step-2 tree** (`scripts/step3_queue.sh`, `scripts/bc_eval_row.sh`
  = phase 2's row_eval with the env replayed exactly from the REFS stamps): G0 suite__omni es1 and field_hard__omni es1
  (S1 at code b9d20a93 + the step-2 harness), G0w = suite es1 with `KRABBY_OMNI_COMMAND_FRAME=world` added (the
  driver now injects it into every eval: it must not move a legacy eval), and the harness proof (S1 + `body_hold` on
  `suite__omni_body` es1, provisional envelope). Run before the step-2 review returns so the GPU is not idle; any row
  whose path a review fix touches is re-run before the step-3 verdict.
- 2026-10-15, **step-4 envelope rule pinned before P1 is read (PLAN D3 made operational):**
  - **ceiling** = the mean, over both hands × es1 + es2, of S1 s18's NET heading change over the first 6 s of the
    world suite's pure-turn holds (turn_l90 / turn_r90, relay saturated) / 6 s. PLAN's wording "mean |wz|" is loose:
    its quoted 0.167–0.170 are this net statistic (REFS: 0.1664 / 0.1685 es1, 0.1676 / 0.1693 es2 → **0.168**);
    mean |wz| would read 0.177–0.179 (gait wobble). The G0 rerun must reproduce these REFS stamps bitwise, so the value
    is the new code's. Cross-check (reported, not used): P1's saturated spot cell (|v| 0, wz cmd 0.30, `body_hold`)
    net rate over [4, 12) s; if it differs from the ceiling by > 0.01, STOP and ask.
  - **w_max** = 1.2 × ceiling rounded UP to 0.01 → 0.2016 → **0.21** (the step-1a value; no re-run of 1a needed).
  - **coupled cap** points: (0, w_max) and, for each commanded |v| ∈ {0.22, 0.30}, (|v|, 1.2 × the smaller of the
    fwd / lat cells' NET yaw rate over [4, 12) s at wz cmd 0.30 through `body_hold`), each wz rounded DOWN to 0.005;
    the frontier is non-increasing by construction (a larger |v| point is clamped to the smaller |v| point's wz).
    The cap is in COMMAND space (commanded |v|, achieved wz), a function of |v| only (step-1 pin 2).
  - STOP and ask if cap(0.22) < w_min (no move+turn pair) — PLAN D3.
- 2026-10-15, **step-2 review (fable·high): no blocking defect.** Legacy path bitwise by inspection of every hunk
  (goldens, compile snapshots and the training-seam test pass); metrics probed on synthetic traces (wrap through π,
  wobble, bias); adapters and route table correct. Applied: `world_rate` saturates at the HEAD's own
  `cfg.ang_vel_max` and refuses a manifest that disagrees (`_world_rate_cap`; equal today, 0.3); the resolution test's
  provenance strip also drops the new `KRABBY_OMNI_COMMAND_FRAME` default (it passes in a clean worktree at HEAD + this
  campaign's files; in the shared tree it still fails from other sessions' ledger edits). **Rule for every gate reader
  (from the review): per-hold body metrics of an episode that falls inside the hold are computed on a truncated
  window — gate 2 / 3 / 4 statistics are taken over SURVIVING episodes only, survival is gated separately (gate 1).**
  Noted: the driver passes `--trained-categories` to every category_suite row, which the harness refuses on
  `turnlevels__omni_body` / `grid__omni_body` (loud, not silent) — this campaign runs those rows with its own launcher.
- 2026-10-15, **step 3 result: G0 BITWISE, harness proof PASS — no stop.** S1 s18 at code b9d20a93 + the step-2
  harness (uncommitted tree; the only later edit is the `world_rate` cap helper, not on these paths):
  `suite__omni` es1 1000 / 1000 episodes and `field_hard__omni` es1 100 / 100 identical to REFS_S1_s18 on every raw
  key and on the scored metrics; `G0w` (suite es1 with `KRABBY_OMNI_COMMAND_FRAME=world` added to the process env) also
  bitwise — the driver's new default is inert. Proof (`scripts/bc_proof_read.py`): S1 + `body_hold` on
  `suite__omni_body` es1 — 700 000 live steps, 0 mismatches (obs cols 5 / 8–10 = applied command, cols 6–7 = clipped
  heading error, applied xy = the row, applied wz = the world relay of the adapter's face error, face inside the
  0.6-rad windup and advancing by wz·dt when unclamped); route metadata body_hold / world / term. Descriptive (not a
  gate, provisional envelope): turns at the ceiling 0.167 net ratio 0.944 (l) / 0.926 (r), ripple 0.023 / 0.021;
  move_turn (0.22, cap80 = ±0.104) net ratio 0.782 / 0.724; straight-hold drift 0.003–0.008 rad/s.
  **Accepted eval change (the driver's `--accept-eval-change` reason):** "body-command campaign step 2: manifest v3 +
  body_hold / world_rate adapters + body yaw metrics + scenarios_omni_body.yaml (user decision 2026-10-15: first-class
  body evaluation); legacy path proven bitwise on S1 (G0)". G0 is re-run at the post-freeze eval_sha at the start of
  step 4 (≈ 15 min) so the record carries it at the final hash.
- 2026-10-15, **P1 read statistic pinned before P1 is read:** a grid cell's achieved yaw rate = the MEDIAN over its
  surviving episodes of the harness's per-episode `net_yaw_rate` (window [4, 12) s), averaged over es1 and es2;
  the cap uses the wz-cmd 0.30 cells (`scripts/bc_freeze_envelope.py`). Gate-reader conventions for the whole
  campaign (`scripts/bc_read.py`): body statistics over surviving episodes only (per-episode records under
  `metrics/episode_*.json` → `holds[label].body`); suite / grid survival = schedule completed; field-row completion =
  `schedule_completion_rate`, field_hard pooled over es1–es4 (n = 400, as REFS), field_light / ring_obst over es1 +
  es2. code_sha moved a second time at the step-2 commit (`fe714b41…`): the `_b` presets' `eval_scenarios` (driver
  metadata, no training value) sit in a hashed config file. eval_sha = `cb813876…` (step 2).
- 2026-10-15, **step 4 P1 read + ENVELOPE FROZEN (`envelope_freeze.{json,md}`; S1 s18 + `body_hold` on
  `grid__omni_body` es1 + es2): no STOP.** ceiling **0.168** (world suite, 4 reads 0.1664–0.1693); spot-cell
  cross-check 0.1676 (Δ 0.0004 ≤ 0.01); w_max **0.21**; coupled cap **(0, 0.21) (0.22, 0.105) (0.30, 0.065)** —
  set by the FORWARD cells (fwd + turn at wz cmd 0.30: 0.088 at |v| 0.22, 0.056 at 0.30; lat + turn: 0.144 /
  0.128): S1 turns far less while walking forward than while walking sideways; the |v|-only cap takes the worse.
  cap80 at |v| 0.22 = 0.084 rad/s (the body move_turn rows). Knobs: `KRABBY_OMNI_BODY_ENVELOPE=0.05:0.168:0.21`,
  `KRABBY_OMNI_BODY_CAP=0@0.21,0.22@0.105,0.3@0.065`. P1 map (descriptive): spot turns at 0.08 / 0.15 net ratio
  0.991 / 0.978; fwd 0.30 m/s cells lose 2–3.5 % survival; translation body ratio 0.88–1.12 at wz 0. The scenario
  file's envelope block is rewritten (comment + block only; eval_sha moves the second time); the shipped-manifest
  test pins the frozen numbers.
