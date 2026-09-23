# Clock deprecation -- omni phase 1 without the gait clock (omni task)

**Status: specified, NOT launched (2026-09-23).** This directory is step 10 of the approved, audited plan
`/home/nickmagus/.claude/plans/lets-explore-options-for-groovy-spindle.md`. Nothing has trained under this charter.
Launch waits on the pre-launch checklist below. The two D00s inputs in `candidates.yaml` are placeholders
(`<E1_STAMP>`, `<E2_STAMP>`) until the step-0 evals E1 and E2 are recorded.

## The question

The gait clock is a per-environment phase in the action term, and its speed follows the command. It formed the gait,
but it also makes the policy rigid:
- the schedule pays for exactly 3 unloaded feet at every instant (duty fixed at 50/50, fixed tripod offsets);
- cadence follows the command, not the achieved speed;
- turning is expensive (0.5 Hz stepping for 0.083 rad/s);
- the policy reads the clock directly (sin/cos in dims 15-16 and all 10 history frames, 22 columns; the critic too).

Removing the clock observation, or removing reward and observation together with a matched control, has never been
tried on any plant.

**The ladder asks one thing per arm: does removing both clock rewards and the clock observation at this C1d
boundary make the gait less rigid (block RIG BETTER, both halves) without losing facing, turning, speed, slip or
survival, against the clock-held twin X0?**

The goal: by the end of omni phase 1 (the C1d chain), the policy is neither paid by the clock nor able to see it,
does measurably less stepping without travel, and depends less on the fixed tripod.

## Settled decisions (user, 2026-09-23; verbatim from the plan, not reopened)

1. **Scope of removal.** "Remove the clock" means both clock rewards (`reward_clock_schedule`,
   `reward_clock_swing_apex`) **and** the clock observation (dims 15–16 and every history copy). By the end of
   phase 1 the robot needs no runtime clock. The env may keep integrating `clock_phase` for telemetry.
2. **Task.** Omni only. Incumbent recipe: C1d `curriculum_turnfirst`, which is t3 for 3000 its, then c4 1500, c5 1500
   and c6 2000, 8000 its in total. Boundaries fall at 3000, 4500 and 6000. Head: X08:c6.
3. **Order of work.** Land the PPO resume fix first. Then test removal **at the stage boundaries**. Only if the
   boundaries don't work, test **extending the final stage and ramping the clock down** within it.
4. **Win rule.** A new decisive RIGIDITY block built on the two side effects the user chose:
   - **stepping without travel**;
   - **fixed 50/50 tripod**, scored as the loaded-foot count `support_feet`, higher is better.

   F, F_mt, T and category survival are non-inferiority guards. Tripod is WATCH. Cadence-lock and terrain
   adaptation are WATCH.
5. **Budget and GPU.** About 47 GPU-h (40, plus the budgeted confirm), strictly serial. The clock campaign runs
   before MQ's next screens. **The MQ session stays paused through the whole clock campaign.**
6. **From-scratch arm.** One never-clocked arm (Z), about 4 GPU-h, runs unconditionally.
7. **Margin flags.** Accepted as proposed: turn `facing_closed` −0.05, move_turn `facing_closed` −0.10, any category
   survival −0.05, any walking speed ratio −0.05, each against the matched clock-held twin.
8. **Confirm.** A formation-seed-2 confirm of the winner is budgeted now and runs automatically when a winner exists.
9. **MQ infrastructure.** Build on what MQ committed today: `checkpoint:` forks, the block and guard machinery,
   `adopt_requires`, `await_if`, and `min_by_category`. MQ's knobs (STAND_PARK, STAND_HOLD, STAND_OBS, RELAY_HYST,
   the std cap) stay off.
10. **Re-budget options** (user, during the audit). Replacement pressure at S1 and the lift rescue at the lift-loss
    flag stay in the plan as options beyond the cap. Each runs only on the user's explicit yes, with its cost stated at
    that stop.
11. **Balanced RIG** (user, during the audit). A WIN needs both halves of RIG's z (steps and support) ≥ 0 as well as
    RIG BETTER (see "Decision rule").
12. **Audit-added mechanisms kept** (user, during the audit): `control: baseline` with the drift anchor pinned, a
    driver park on every flag with an `ack-flag` release, the turn `foot_slip` guard, the t3_s2 formation check, the
    terrain WATCH evals, and the Z-exclusive GPU window. The user accepts that most DECISIONs will park.
13. **Standing rules.**
    - Offline replay gate before any screen.
    - Never push.
    - One Isaac process at a time, under `systemd-run --user --scope -p MemoryMax=…`, launched with `setsid nohup`.
    - Weight ε = 0.001, never 0.0 (0.0 kills term telemetry).
    - Push-notify at every stop point.
    - Stop the driver only during a stage eval.
    - The user confirms or overrules every DECISION.

## The design

The removal set is **K** = `{KRABBY_CLOCK_W: "0.001", KRABBY_APEX_W: "0.001", KRABBY_OMNI_CLOCK_HIDE: "1"}`: both clock
rewards to ε and the clock observation zeroed at full width (1149), so a resumed head still loads.

- **Chain `tail`** (every screen, 5000 its): C1d's c4 → c5 → c6, overlays verbatim from
  `../2026-09-19_facing_tracking_ladder/candidates.yaml` (`curriculum_turnfirst`), preset `1a`. Stage c4 forks
  **X08:t3** (`.../2026-09-22_09-10-29_seed3_X08_t3_A15pB/model_2999.pt`) by `resume_from: {checkpoint: ...}`. The PPO
  resume fix (eabe7324) carries the priv_reg counter, LR and aux optimisers across the fork. Every stage gets a suite
  eval (c5's is new).
- **Baseline X0**: the tail with the clock held, at s3 and s2. It is the exact clock-held twin of every arm (arms re-run
  their clock-held prefix), and its pair gives D00bs, the seed term.
- **Wave 1**: three alternatives, run latest-first, each `kind: clock`, `control: baseline` (judged against X0, never
  composed, drift anchor pinned to X0), `eval_carry: [KRABBY_OMNI_CLOCK_HIDE]`. No critic reset in arm or twin.

  | Rank | Arm | Overlay | Clock-free its | Gate |
  |---|---|---|---|---|
  | 1 | A6000 | `{c6: K}` | 2000 | `user_decision` (S2); the pilot (S3p) |
  | 2 | A4500 | `{c5: K, c6: K}` | 3500 | -- |
  | 3 | A3000 | `{c4: K, c5: K, c6: K}` | 5000 | `depends_on [A6000, A4500]`; `await_if` both `arm_class == BREAK` |

- **Z** (`references: scratch_free`, seed 3): C1d from scratch with K from iteration 0, t3 → c4 → c5 → c6 (8000 its),
  C1d's floors (WATCH on a reference), 4 suite evals, run by `--reference-chain` right after the last wave-1
  DECISION. **Nothing else touches the GPU until Z's chain and evals finish** (terrain rows, wave 2 and the confirm
  queue behind it). Judged by hand at S4b: the t3 formation readout (moving ratio ≥ 0.30 on fwd, side, back; survival
  ≥ the floors), then Z:c6 vs X0 s3:c6 under the full decision rule. c4–c6 run even after a failed readout. Z is
  never driver-adopted.
- **Wave 2** (only on FAIL-SAFETY, or FAIL-BENEFIT plus the user's choice): a separate directory
  `..._clock_deprecation_w2/`. W extends c6 by 4000 its from X0 s3's c6 head (reward cosine to ε over its 0–1000,
  per-episode hide 0 → 1 over 1000–2000, clock-free 2000–4000), judged against the clock-held extension X0x.
- **Confirm** (runs automatically once the user names a winner): `references: t3_s2`, C1d t3 from scratch at seed 2,
  clock held, with the step-13 formation check before any tail. Both the clock-held tail and the winner's tail then
  fork it (`resume_from: {reference: t3_s2}`), declared at step 15 under `--accept-candidates-change`. Judged by hand
  under the full rule. **Pass**: RIG z has the same sign as the s3 decision, both RIG parts ≥ 0, nothing WORSE at R,
  no flag of any kind; otherwise the user decides.

**What an ADOPT means:** removal at that boundary, **including** that boundary's element arrival.

## The decision rule (frozen into D00s)

Rule v2 per category, as in MQ, with Q replaced by RIG and `adopt_requires: RIG`. Every metric is the suite row's
per-category number over the 12-s command window. Each delta is sign × (candidate − X0); z_b = mean_i(δ_i / R_i);
R_i = max(R_eval,i from D00s, R_seed,i from D00bs), plus R_binom for survival guards.

- **Blocks (95 %, two-sided).** F = turn `facing_closed`, `yaw_progress`; F_mt = the same on move_turn; T = `ratio`
  on fwd, side, back, diag; **RIG** (10 members) = `steps_per_m` on fwd, side, back, diag and `steps_per_rad` on turn
  (lower is better; turn uses its whole hold) plus `support_feet` on fwd, side, back, diag, turn (higher is better).
- **Guards (99 %, one-sided; 23).** Survival on all 7 categories; `move_turn.ratio`; `ratio@guard` on the 4 T
  categories; per-foot `foot_slip` on the 4 T categories **and turn** (it does not scale with `support_feet`, so the
  hoped-for longer stance is not a BREAK); `steps_per_m@guard` on the 4 T categories and `steps_per_rad@guard` on turn;
  `stand.steps_per_s`.
- **ADOPT** = RIG BETTER, nothing WORSE, no floor, anchored drift OK. RIG TIE while another block is BETTER is a clear
  REJECT, as is any WORSE, unless a `pair_on` case applies.
- **`pair_on: [adopt, neutral, lone_guard]`.** An s3 ADOPT, a NEUTRAL line (RIG TIE, nothing WORSE) and a lone-guard
  line (RIG BETTER, exactly one guard WORSE, no block WORSE) each queue a seed-2 pair. The s3 entry stays
  `speculative`; the **pooled decision is final** and again requires pooled RIG BETTER.
- **Balanced-RIG rule (settled decision 11).** A WIN, and any ranking of WINs, also needs the steps part and the
  support part of RIG's z to be ≥ 0 each. `support_feet` resolves far better than steps (eval R 0.7–2 % vs 4–13 %), so
  without this rule RIG could be won on loaded-foot count alone. A pooled ADOPT with a negative part is NEUTRAL.
- **Classes** (`arm_class`, pooled line, all vs X0; precedence BREAK > RIGID > NEUTRAL): **WIN** = pooled ADOPT, both
  RIG parts ≥ 0, removal verified; **BREAK** = F, F_mt or T WORSE, any guard WORSE, or a floor; **RIGID** = RIG WORSE,
  nothing else WORSE; **NEUTRAL** = nothing WORSE and RIG TIE, or a pooled ADOPT with a negative part; **UNCLASSED** =
  RIG N/A or MISSING, a guard MISSING, SKIPped, or user-overruled (ask at S4). **Boundaries work** ⟺ at least one WIN
  with no unresolved flag.
- **Flags** (stop and ask; never a verdict; the driver parks on any flagged DECISION until
  `ledger_cli ack-flag <Did>`; read on the pooled pair when one exists, with the member's MDE): the margins of settled
  decision 7; taps up > 25 %; `swing_apex` down ≥ 30 % (lift loss; the lift rescue is a re-budget option);
  `support_feet` down > 10 % in any category; with `steps_guard_flag`, steps adverse > 15 % of X0's; speed-driven
  (steps members' mean δ/R BETTER, mean Δlog `step_rate_hz` ≥ 0, T z > 0); removal verification failed. Removal
  failures and the pilot stop park at the **seed-3** line, never deferred. A baseline floor also parks. Every
  DECISION line prints RIG's z split (steps part / support part). The thresholds are driver constants; the yaml's
  `decision_rule.flags` block is their frozen record.
- **Tripod is not a flag**: `tripod = coh × (1−d)/d` falls by construction as duty rises, the success direction.
- **Removal verification (hard).** Every HIDE eval shows `clock_obs ≡ 0` and the carry in `run_meta`; every clock-free
  stage ends with effective clock weights ≤ 0.001; `clock_hidden_frac` is 1.0 over the tail.
- **WATCH.** Cadence set (`gait_hz`, `clock_hz`, `cadence_ratio`, `hz_speed_slope`), `offset_R`, taps, `swing_apex`,
  the turn pre/post-closure split, move_turn `steps_per_rad` and `support_feet`, tripod, `slip_per_m`, MQ's Q members,
  `six_down_frac`, saturation, training clock adherence, value loss and episode length over the 300 its after a
  removal, and `R_twin(c4)`.

**The S3 rule.** The user confirms or overrules **only decided or pooled lines**. A speculative line (pair queued) is
answered by its pooled line. Confirming or overruling a speculative line cancels its seed-2 pair (squashing it if in
flight), which makes WIN unreachable: `ledger_cli confirm|overrule` refuses it unless `--cancel-pair` says the cancel
is intended.

**Floors** (`category_survival`, band-tested one-sided at 99 % on n_env 100; collapse detectors):
c4 `min_by_category` = X08:c4 survival − 0.30 (stand 0.69, fwd 0.52, side 0.65, back 0.55, turn 0.62, move_turn
0.315); c5 `min_new` 0.10 / `min_prev` 0.30; c6 X08:c6 − 0.20 (0.79 / 0.72 / 0.77 / 0.72 / 0.59 / 0.75 / 0.55 for
stand / fwd / side / back / diag / turn / move_turn).

## Stop points (each one push-notified)

| Point | Trigger |
|---|---|
| S1 | step 0 plus replay gate (RG1–RG3), spread check and standalone D00s |
| S2 | X0 pair done; the driver parks at `await_user:A6000` after D00bs (X0 sanity vs X08:c6 at full R, RIG power, D00bs acceptance) |
| S3 | every wave-1 DECISION |
| S3p | the pilot stop: A6000's seed-3 line with RIG TIE (before its pair and A4500; released by `ack-flag`); the user chooses continue wave 1, wave 2, or pause for MQ's action-bound fix |
| S4 | end of wave 1: classes, `R_twin(c4)`, the winner (highest pooled RIG z; within each other's R, the latest boundary); decisions that need Z wait for S4b |
| S4b | Z's chain and evals finished (Z readout); only then any further GPU work (terrain rows, confirm, wave 2) |
| S6 | wave-2 DECISIONs |
| S7a | confirm |
| S7 | close: recipe of record, ε vs literal 0.0 in deployment, MQ's next fork target, clock-free 2a/2b/2c follow-on |

Any flag, a floor on X0, or a removal-verification failure also stops the run.

## Pre-launch checklist (in order)

1. Substitute `<E1_STAMP>` (X08:c6) and `<E2_STAMP>` (X08:c4) with the step-0 stamp directories (repo-relative).
2. `run_candidates.py --campaign-dir <this dir> --q-spread-check <E1> <E2> --q-spread-heads X08-c6,X08-c4`. Demote any
   zero-spread member under `decision_rule.per_category.demoted`, never force-accepted.
3. Compute **D00s standalone** (`resolution.py`). Null all-TIE band [0.616, 0.98] (nominal 0.99²³ × 0.95⁴ ≈ 0.646).
   **Pre-registered on a null-band failure:** stop and ask; read the per-guard false-WORSE rates; remove the five
   steps `@guard` entries from `guards.members` **only** (`demoted` would strip them from RIG too); set
   `steps_guard_flag: true`; re-run; if still outside, `--accept-resolution D00s` with the reason recorded.
4. `--dry-run` clean, then S1.
5. Launch through `launch_candidates.sh` under a systemd unit with MemoryMax and `setsid nohup`, with `notify.log`
   relayed as push notifications.

## Budget (≈2440 it/h, 0.05 h per process start, 0.133 h per suite eval; strictly serial)

| Item | GPU-h |
|---|---|
| Smokes (step 8) | 0.60 |
| Step-0 evals E1–E7 | 0.93 |
| X0 s3 + s2 | 5.20 |
| Step 0b (hidden zero-shot evals of X0 s3's c4/c5 heads) | 0.27 |
| Wave 1 at s3 | 7.80 |
| Terrain WATCH (`light__omni` on X0 s3:c6 and the winner) | 0.75 |
| Wave-1 pairs, worst case (3) | 7.80 |
| Z reference | 4.01 |
| Wave 2, conditional | 7.55 |
| Confirm, worst case | 8.00 |
| Contingency: one tail rerun | 2.60 |
| **Worst case** | **45.5** (cap 47) |

Typical paths: an A6000 win ≈ 31 GPU-h; FAIL-SAFETY then a wave-2 win ≈ 37 GPU-h. The re-budget options (replacement
pressure at S1, ≥ 5.2 GPU-h; the lift rescue, 2.6–5.2 GPU-h) are beyond the cap and run only on an explicit yes.

## Caveats

- **Element arrival at every boundary.** Every C1d boundary adds an element: c4 move_turn, c5 omni, c6 omni_turn. An
  arm removes the clock exactly where a new element arrives, and the scout's clock notches failed exactly there
  (completion 0.99 → 0.48). A BREAK may be the arrival, not the removal; wave 2 forks a mature c6 head to separate them.
- **Bang-bang saturation.** MQ found the policy mean saturated (|μ| > 1 on 96 % of steps). Stand stepping (0.58 Hz
  with the clock frozen on 85 % of stand steps) is not clock-driven. The confound is matched across every arm and its
  twin and read as WATCH; a NEUTRAL arm at X0-level saturation reads "inconclusive: re-test after MQ's action-bound fix".
- **The n = 0 premise and the pilot stop.** Nothing yet shows that removing the clock moves `support_feet` or steps
  beyond R: there are no clock-free runs, and five X08-lineage heads keep 3.14–3.19 feet loaded at stand even with the
  clock frozen. A6000's seed-3 RIG TIE therefore parks before anything else is spent (S3p).
- **RSI on a clocked bank for Z.** Z keeps C1d's RSI 0.2 on the clocked bank (its formation aid), so its initial
  states come from clocked gaits. From-scratch seed spread also exceeds the tail-only D00bs (LAD's full-chain seed R
  on F/F_mt/T was 3.19/4.12/1.13): read only large effects.
- **The fork-level null.** Arms re-run their clock-held prefix, so X0's s3/s2 spread (D00bs) is the exact fork-level
  null, but it is one seed pair. `R_twin(c4)` (the identical clock-held c4 stages of X0 s3, A6000 s3, A4500 s3) is
  reported at S4 as a check. The settled margins sit at or below single-screen noise (turn `facing_closed` MDE ≈
  0.15), so expect most DECISIONs to park.
- **The recipe moved.** X0 is retrained under the PPO resume fix. If X0:c6 is WORSE than X08:c6 at S2, results may
  not transfer to the recipe of record.

## Records

The driver writes `ledger.json`, `ledger_overrides.jsonl`, `state.json`, `heads.json`, `REPORT.md`, `CHANGELOG.md`,
`notify.log` and `orchestrator.log` beside this file. `replay_clock_economy.py` (plan step 5) is the clock-economy
replay tool for the step-9 gate.

## Log

- **2026-09-23 — plan approved; campaign specified, not launched.** Plan `lets-explore-options-for-groovy-spindle.md`
  (audit cap 6; settled decisions 1–13 above). Code steps 1–5 committed locally, nothing pushed: eabe7324 (PPO resume
  fix), c9444d24 (clock-hide knob), 60c7e009 (rigidity metrics), eca92003 (ladder support and the replay tool),
  c04251d0 (phase-config test). Step 10: `candidates.yaml` (from the step-4 fixture) and this charter written;
  `--dry-run` validator clean except the two placeholder D00s stamps. Next: step-0 stamps, spread check, standalone
  D00s, S1.
