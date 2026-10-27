# Plan — robot-frame velocity commands for the omni walker (campaign `2026-10-15_body_cmd`)

> **Audit trail (plan-audit, cap 3 reached).** Pass 1: 7 blocking (opus) + 6 overlapping (fable) — envelope rule,
> anchor set, reward-pricing premise, inertness proof, signed turn statistics, adapter shaping. Pass 2: 6 / 5 —
> straight-gated ang-vel term, gate-2 semantics, move-turn cap headroom, stage anchors, world rows reported. Pass 3:
> 1 / 2, applied as minimal edits but not re-audited — tent `p` = 1-s mean rate + step 1a on S1's measured wz traces;
> gate 2 adds a w_max level (top-rate compared at saturation, ripple only below the ceiling); step-6 stage band =
> max(floor, three-anchor range), STOP only on collapse. Trend 7 → 6 → 2, all in gate semantics / premises.
> **Left to implementation (pass-3 deferred, non-blocking):** P2 may pass gate 4 → add a drift-injected known-bad if
> it does; gate 3's 0.8 near w_min read against the anchor; a monotone tolerance (= band); step-in-place total income
> reported in step 1a; the clock-radius rule (a w_max below 0.208 is refused today); a coupled-cap rule for bearings
> off the P1 grid; a cheap p2a_b smoke; which env seed the step-6 noise comparisons use.
> **Size:** 11 steps / ~2,660 words vs the reviewed 11 / ~1,926 (+38 %); the growth is blocking fixes and your
> decisions (no new steps) — kept as is. Reject with "one more pass" to run a fourth audit pass first.

## Context

The omni walker is commanded with a WORLD travel direction + speed and an ABSOLUTE world facing target; the command
term turns that into a body (vx, vy, wz) each step through a P-relay (wz = clip(1.5·heading error, ±0.3), zeroed
below 0.2), so the policy has only ever seen wz ∈ {0} ∪ (0.2, 0.3] and also observes the facing error itself
(obs cols 6–7). The user wants the facing command replaced by a **commanded angular velocity**: movement commands
become **pure robot-frame velocity (vx, vy, wz)**. Deployment already speaks this language (HAL teleop sends
body-frame `vx, vy, yaw_rate`); the facing target exists only in training, evaluation and the campaign tools. The
facing campaign had proposed a body frame (C12) as "the only change that makes move+turn kinematically
honourable" and deferred it; lit reviews recommend observing only a yaw-rate command; move_turn barely turns today
(yaw_progress 0.097).

**User decisions (2026-10-15):** retrain the walker chain FROM SCRATCH under body-frame commands; command range =
the robot's MEASURED envelope; make body-frame evaluation FIRST-CLASS in the locked harness now (eval_sha moves);
re-check the stand network with the new walker and retrain it only if needed. Adoption / bake / push stay the
user's.

## Design (decisions D1–D7)

**D1 — Body command mode, default-off.** `KRABBY_OMNI_COMMAND_FRAME=body` swaps `commands.base_velocity` for a
`CrabHexOmniBodyCommandCfg` subclass (class `CrabHexOmniBodyCommand(CrabHexOmniCommand)`,
`mdp/crab_hex_omni_command.py`); unset ⇒ every world preset's config dump and sampler RNG stream bitwise unchanged.
Per resample the same pool/slot machinery draws (same uniforms, same order) a body bearing, speed (ellipse
0.35:0.35:0.25, walk floor 0.21) and wz (continuous, D3), held constant in the body frame for the slot
(`vel_command_b = (v cos φ, v sin φ, wz)`). `slot_id = moving + 2·turning` is a pure function of the held command;
**stand ≡ (0, 0, 0)** (no latch). `heading_target = yaw`, `travel_dir_w = yaw + φ` refreshed every step so legacy
consumers read defined values. New `schedule_hook_body(vx, vy, wz)`; pure math in `crab_hex_omni_command_math.py`
(`sample_yaw_rate`, `slot_from_body_cmd`, `body_cmd_violations` = the ONE envelope definition used by sampler,
hook, validator and a deployment `shape_body_command`).

**D2 — Observation, same 1149 width.** Col 5 = wz, cols 8–10 = body vx, vy, |v| (meanings kept); cols 6–7 (facing
error) = 0 under a new encoding `none` (raw / sincos refused in body mode). Mirror tables unchanged.
`KRABBY_OMNI_COMMAND_FRAME` joins `run_phases.EVAL_CARRY` (default `world`); an eval whose head frame and row route
disagree without a NAMED adapter is INVALID.

**D3 — Command range = measured envelope.** Premise P1 (step 4) maps the achieved (|v|, wz) of S1 driven through the
`body_hold` adapter on a grid (|v| ∈ {0, 0.22, 0.30}, fwd / lat, wz ∈ {0, 0.08, 0.15, 0.30}). Pre-registered rule:
w_min = 0.05 rad/s; ceiling = S1's sustained pure-turn rate (mean |wz| over the first 6 s of the saturated pure-turn
holds; S1 world suite today: 0.167–0.170); w_max = 1.2 × ceiling rounded up to 0.01 (expected 0.21); move+turn types
draw (v, wz) inside a coupled cap = the piecewise frontier through the P1 (|v|, wz) points with the same 1.2×
headroom on its wz axis at each fixed |v| ≥ 0.21 (user decision); body move_turn EVAL rows sit inside the cap (80 % of
its wz at fixed v ≥ 0.21); the envelope is a knob / manifest parameter, not a code constant.
Frozen in the CHARTER before any training. **STOP and ask** if the cap admits no pair with v ≥ 0.21 and wz ≥ w_min.

**D4 — Rewards (body mode only, one applier `_apply_body_reward_set`, term names kept).**
- `reward_tracking_yaw` (facing, 0.2) → windowed yaw tracking `exp(−|Δyaw_1s − ∫wz_cmd| / σ)`, σ = 0.05 rad
  (= w_min × the 1-s window) (heading hold on wz = 0, rate tracking otherwise; window reset on a command change /
  reset);
- `reward_yaw_progress` → tent peaking at the commanded rate, `min(p, 2|c| − p)` with p = the 1-s mean yaw rate
  toward the command (the windowed term's Δyaw_1s / 1 s; a per-step rate makes the tent concave under gait wobble);
- `track_ang_vel_z_exp` (std 0.5) pays ONLY on wz = 0 rows in body mode (heading hold; a straight-gated twin of the
  existing default-off turn-gated variant) — turning rows are paid by the tent + the windowed term (user decision);
- **turn-pricing check, CPU, before any code (step 1a)**, on S1's MEASURED per-step wz traces from translation and
  move_turn holds (healthy tracker = the trace with its mean removed + c; step-in-place = mean removed, not shifted;
  still stander = zeros): the summed yaw-term income of each non-rotating trace ≤ 10 % of the healthy tracker's at the
  ceiling, and the tracker-minus-stander premium at w_min and at 0.08 rad/s recorded in the CHARTER, with the w_min
  non-rotating share ≤ the idealised-tracker bound (≈ 30–40 %) pinned there; a fail → STOP;
- **slow turns stay "moving"**: body-mode gait gates / gait clock use clock speed ≥ the walk floor whenever wz ≠ 0
  (today `max(|v|, 1.2|wz|)` shuts gait terms off below 0.10 rad/s and stops the clock below 0.167);
- refused in body mode: facing-turn weight, facing-sigma twin, relay hysteresis, `KRABBY_OMNI_FACING`, RSI frames.
- Stroke / event consumers switch to `command_signature()` / `intended_bearing_b()` (world mode bitwise).

**D5 — First-class evaluation (moves eval_sha).** Manifest v3 with `command_route: term`: `BodyScheduleRow(vx, vy,
wz)` routed through `schedule_hook_body` (slot / stand mask / telemetry correct), a validator (ellipse at
atan2(vy, vx), walk floor, wz ∈ {0} ∪ [w_min, w_max], coupled cap), yaw-rate metrics (`wz_ratio`, `net_yaw_rate`,
`net_yaw_ratio`, yaw ripple = std of the 1-s mean wz, `heading_drift` on wz = 0 holds, planar drift on pure turns),
report blocks, `eval_categories` rule v3, new `experiments/eval/scenarios_omni_body.yaml` (`suite__omni_body`: stand,
fwd, side l/r, back, diag, turn ±, move_turn ±; the envelope-grid row; body field rows). `scenarios_omni.yaml` and
v1/v2 rows are not edited (compile snapshots unchanged). Turn
statistics are SIGNED per hand: net ratio = net rotation / (wz_cmd · T) over the scored window [4, 12) s of a
12-s hold (2-s settle); a magnitude ratio (mean |wz| / |cmd|, wobble-sensitive) is never gated.

**D6 — Adapters, so every head runs both row families.** `body_hold` (a world-trained head on body rows: integrate
wz into the facing target with ±0.6-rad anti-windup, dir = yaw + atan2(vy, vx)) and `world_rate` (a body head on the
legacy world rows: the relay output passed through `shape_body_command` for the head's trained envelope — wz ≡ 0 for
stages without turn slots — then col 5, cols 6–7 = 0; the clipped-step share is logged per hold). The ANCHOR every
gate compares against = the facing three-seed set S1 s16 / s17 / s18 (REFS heads at the current code) driven through
`body_hold` on the body rows.

**D7 — Stand network.** Re-check SH_T17 beside the selected body walker (K4 restore + hand-back, K5 G1–G5 at
1.064, K6, a stand-after-turn row; switch keyed on cmd (0, 0, 0)) in a campaign launcher that pairs the new walker;
re-DAgger SH (warm start SH_T17, new walker frozen in the loop) only if a gate fails.

## Steps

0. Campaign dir `experiments/2026-10-15_body_cmd/` (CHARTER = pre-registration + log incl. the gate set below, PLAN =
   this plan); check no omni driver is running (`ps`; side_bisect / phase2 `state.json` read `phase: run` — stale?)
   and record. → sonnet · medium — transcription
1. **(1a) D4 turn-pricing arithmetic** (CPU, before any code; **STOP and ask** on a fail). **Command / obs / reward
   code (D1, D2, D4, D6 adapters), default-off**, + CPU tests: knob-off dumps of every world
   preset + the S1 overlay bitwise; world RNG stream + `command_signature` bitwise; body sampler stays inside
   `body_cmd_violations`; stand exactly (0, 0, 0); slot from command; cols 6–7 = 0 in obs AND history; encoding /
   frame refusals; the `_b` preset family (`1b_b … 2a_b`, one delta dict pinned key by key).
   → opus · xhigh — new command semantics; a stale facing value trains quietly against turning
   ✓ fable · high — independent review
2. **Eval harness v3 (D5) + adapters in the harness**, CPU tests (v1/v2 golden + compile snapshots unchanged; v3
   parse / validate; rule v3; CAT_KEYS cross-pins; EVAL_CARRY frame refusals).
   → opus · xhigh — locked harness; a wrong route scores the wrong thing
   ✓ fable · high — independent review
3. **Inertness + harness proof (≈ 1.5 GPU-h).** G0: S1 on `suite__omni` + `field_hard` es1 at the new code / eval sha
   reproduces REFS_S1_s18 BITWISE (raw keys); S1 + `body_hold` on `suite__omni_body`: 0 slot / route / obs mismatches.
   Record the accepted eval change (`--accept-eval-change`, reason). **STOP and ask** if G0 is not bitwise.
   → main (opus) — runs + the stop decision
4. **Premises + anchors (≈ 3.5 GPU-h + CPU).** P1 envelope grid → freeze w_min / w_max / coupled cap (D3 rule) in
   the CHARTER; then the anchor (S1 s16 / s17 / s18 + `body_hold`) on the frozen `suite__omni_body`, grid and body
   field rows, es1 + es2, and every anchor-relative gate clause written into the CHARTER as a NUMBER. P2 S1 zero-shot in
   body mode (drift + turn = the known-bad controls for STRAIGHT and TURN WORKS). Offline replay of the D4 reward set
   at w_min and at the ceiling on saved healthy (S1 turn / move npz) and degenerate traces (over-spin on a low command,
   stand-on-turn, **step-in-place without rotating on a w_min command**): each degenerate's yaw-term income within the
   step-1a margins (a 0.01 rad/s drift on straight is reported only — no 1-s term can separate it from healthy wobble;
   drift is policed live by the stage-end drift stop in step 6). There is no healthy
   slow-turn trace (S1 never saw wz < 0.2), so the low end is checked live at the c6_b stage end (step 6).
   **STOP and ask** if a degenerate fails the margin, a named known-bad passes the gate it is known-bad for, or an
   anchor-relative clause would need a net ratio > 0.95 or a ripple < 0.03.
   → opus · high — the numbers every gate rests on
   ✓ fable · high — independent check of the envelope fit
5. **Training smoke (≈ 0.5 GPU-h)**, 100 its each on `1b_b`, `1a_b` + the c4 overlay (clock visible, turning) and
   `2a_b` + the c6x overlay: dump audit (frame, band, rewards swapped, no facing knob in any realised overlay),
   cols 6–7 = 0, col 5 / 8–9 = command, slot = f(command), facing term absent, every turning row down to w_min has its
   gait gate open AND the clock phase advancing, sampled (v, wz) inside the envelope; one smoke head through the shaped
   `world_rate` and `suite__omni_body` with the gate-0 column checks. → sonnet · high — assertions fail loudly
6. **Pilot chain s18 from scratch (≈ 7.6 h train + ≈ 2 h evals)**, the shipped lineage's stages and iterations under
   the `_b` presets + the same pools / slots (turn types now draw wz): fb6k_b 6000 → c4_b 1500 → c5_b 1500 → c6_b 2000
   → c6x_b 4000 (clock-free) → p2a_b 5000 (field fork). Before each stage trains, its realised dump is asserted
   (frame, envelope, ellipse 0.35:0.35:0.25, rewards swapped, no facing knob). Stage-end read on `suite__omni_body`
   vs FRESH anchors of the facing same-stage heads of ALL THREE seeds (FB6k__B / SF__B_c4 / SF__B_rest stage
   checkpoints s16 / s17 / s18 + `body_hold`, run at step 4 on the translation and wz = 0 holds at the new eval sha,
   ≈ 4.5 GPU-h): on the stage's trained categories, a shortfall against the worst same-stage anchor beyond max(floor,
   the three anchors' range) is decide-record-notify; **STOP and ask** only on a collapse (survival or speed ratio
   < 0.5× the worst same-stage anchor's) or when heading drift fails gate 4's clauses against the same-stage anchors
   (from fb6k_b on); turn rows are read descriptively, except the low-end check at the c6_b stage end (where
   the facing lineage first turned): **STOP and ask** if the pure-turn net ratio at w_min is < 0.5.
   → main (opus) — chained queue + stage reads
7. **Pilot gates** (below) on `suite__omni_body`, the envelope grid and body field rows (legacy world rows via the
   shaped `world_rate`: reported); **STOP and ask** before replicates if any gate fails. → main (opus)
8. **Replicates s16 / s17**, same config and evals (≈ 19 GPU-h), launched automatically when the pilot passes the
   pilot gate set = gates 0–5 + gate 7's per-head clauses (user decision; gate 6 stays at step 9), with a notification.
   → main (opus) — chained queue
9. **Stand-network re-check (D7, ≈ 2–3 GPU-h; + ≈ 3.5 if a re-DAgger is needed)** with the selected body walker.
   → main (opus)
10. **Report + STOP for the user's adoption decision** (bake = new body lineage in `policy/manifest.yaml`, presets,
    play carry, stand pairing; nothing baked or pushed without the user). → opus · xhigh ✓ fable · high

**Stop points (push-notify each):** step 3 inertness, step 4 premises, step 6 stage stop, step 7 gate failure,
step 10 adoption.

## Pre-registered gates (each body seed vs the anchor = the facing three-seed set S1 s16 / s17 / s18 + `body_hold` on
the body rows; "worst anchor" = the weakest of the three on that statistic, "best anchor" = the strongest; band = the
anchor's |es1 − es2| floored at ratios 0.02, yaw rates 0.005 rad/s, completion ±0.04 / binomial SE; speed ratios read
as |ratio − 1|, larger = worse; turn statistics signed per hand (D5); stamps compared only within one manifest shape;
every anchor-relative threshold is written into the CHARTER as a number at step 4)

0. **INVALID EVAL** — frame / CLOCK_HIDE / STD / ellipse missing from run_meta; code / eval sha mismatch; cols 5, 8, 9
   ≠ the row (adapter rows: ≠ the adapter's output) or cols 6–7 ≠ 0 in body mode.
1. **FLOOR** — survival ≥ 0.95 on every straight and turn hold; move_turn survival ≥ worst anchor − 0.05 per hand.
2. **TURN WORKS** — on the levels {w_min, 0.08, 0.15, the ceiling, w_max} (listed in the CHARTER at step 4): pure-turn
   net ratio ≥ 0.8 at every level above w_min up to the ceiling and ≥ 0.5 at w_min; monotone response; yaw ripple at
   every level BELOW the ceiling ≤ 0.5× the worst anchor's (the adapter's stop-go relay is its weakness there; at and
   above the ceiling the relay saturates); top turn rate: net rate at w_max ≥ the best anchor's saturated net rate
   (its read at w_max) − band. Known-bad: S1 zero-shot body.
3. **MOVE+TURN** — body move_turn rows: net ratio ≥ 0.8; body-frame translation |ratio − 1| ≤ the best anchor's
   (better than the facing set).
4. **STRAIGHT** — median |mean wz| on wz = 0 holds ≤ max(0.01, 2× the worst anchor's median); p90 ≤ max(0.02, 2× the
   worst anchor's p90). Known-bad: P2.
5. **RETENTION** — on the body rows: category |ratio − 1| ≤ worst anchor's + max(0.03, band), both side hands;
   field_hard (body rows, n = 400) ≥ worst anchor − max(0.05, 2·band); field_light, ring_obst likewise. World rows via
   the shaped `world_rate` and gait shape (tripod, steps per m, apex) are reported, not gated.
6. **STAND PAIR** (before the body walker may become current) — K4, K5 G1–G5, K6 pass; stand-after-turn 0 falls.
7. **ADOPTION** — ≥ 2 of 3 seeds pass 1–5 incl. TURN WORKS with no hand loss (a side hand slower than the worst
   anchor's by > band); adoption is the user's.

## Budget

≈ 43–48 GPU-h: inertness proof 1.5, premises + anchors 3.5, stage anchors 4.5, smoke 0.5, pilot ≈ 9.5, replicates
≈ 19, stand 2–3 (+ 3.5), evals within.
Seeds 18 (pilot), 16, 17; env seeds es1 tune / es2 held out.

## Verification

- CPU: knob-off bitwise dumps + RNG; v1/v2 eval goldens unchanged; v3 / adapter / validator tests; reward replay.
- GPU: G0 bitwise S1 reproduction at the new hashes; harness proof (0 mismatches); smoke assertions; every gate
  read with its known-good / known-bad controls.
- code_sha and eval_sha each move once, recorded with reasons; no other campaign's files touched; nothing pushed.

## Later phases (not run)

Bake + preset switch + play / export carry (after adoption); deployment mapper using `shape_body_command`; a
from-scratch re-run of the stand-net campaign if the re-DAgger is insufficient; facing-only cleanup; body-row
support in the shared video tool `record_categories.py` (user decision: later).

Routing: strong · roster ok (opus = Opus 5.5) · planned on opus · ceiling: opus · 1× sonnet·medium, 1× sonnet·high, 1× opus·high, 3× opus·xhigh, 5× main, 4× fable·high check, fable retry on 2 fails
Effort enforcement: prompt-level
