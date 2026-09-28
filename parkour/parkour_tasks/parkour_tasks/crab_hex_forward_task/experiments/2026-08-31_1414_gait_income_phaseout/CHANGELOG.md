# Gait-income phase-out campaign (PLAN G) — CHANGELOG

Approved plan: `/home/nickmagus/.claude/plans/there-are-a-variety-federated-frog.md` (user-approved 2026-08-31).
Successor to the graduated round-search campaign (`sim_fine_tuning/2026-08-26_2200_gated_lineage/`).

## Charter (2026-08-31)

**Hypothesis:** after gait formation (0–5k), gait-formation reward income is farmable
forever and caps real-objective progress; it can be phased out per-term without
losing the gait. Suggestive evidence: the two graduated 20k seeds anti-correlate
tripod vs tracking at full gait income (primary 0.436/0.495 vs seed-3 0.606/0.433).

**Design:** reverse round search on the **seed-3 lineage**. Per-term anneal notches
(clock 1.0→0.5→0.2→ε; apex 1.0→0.5→ε; airtime 0.8→0.4→ε; stride 0.5→0.25→ε;
mirror-loss 0.5→0.25→0 late-eligible after clock_eps settles) tested one at a time
per round (r1 5k–10k, r2 10k–15k, r3 15k–20k, r4 optional to 25k), co-scheduled with
the graduated curriculum element arrivals (7@5k, 3@10k). A passing term stays in the
catalogue for lower notches; a failing notch retries in a later round; terms leave
only at ε. Cosine ramp ~1k iters then hold; all gates on settled tails,
unit-weight normalized. ε floor = 1e-3 (weight==0.0 kills term telemetry,
`parkour_reward_manager.py:27`); total-reward ≥0 clip floor
(`parkour_reward_manager.py:38`) guarded by a per-round income-budget table.

**Kept fixed:** gait penalties (foot_idle / excess_contact / tibia_deviation), RSI
0.2 (bank re-harvest only from backstop-passing bakes; **RSI-off terminal
verification mandatory** before any bake proposal), clock observation (structural).

**Anchors (seed-3 confirmation replay):**
- 5k anchor / campaign start: `parkour/logs/rsl_rl/crab_hex_flat_walk/2026-08-31_03-42-16/model_4999.pt`
- r1 control: segment 1 `2026-08-31_06-04-16` (10k head `model_9998.pt`; TB on disk; retroactive canary pending)
- Full-income reference & terminal control: `2026-08-31_10-46-19/model_19996.pt`
  (flat 0.606/0.990/0.433; all-obstacle-light 0.64, tripod 0.577) — the reference of record.

**Entry caveats inherited from graduation:** recal-terrain 0.480 waived; absolute
failure-tail <0.25 waived (terrain-promotion equilibrium); goal_idx void; turn
probe by vx-proxy only. Terminal failure gate is control-relative (≤ control +0.05).

**Terminal WIN (vs reference of record, matched iterations):** obstacle completion
≥ control +0.05 OR canary tracking ratio ≥ control +0.10; AND tripod not worse than
control by >0.05; failure ≤ control +0.05; completion within 0.05; RSI-off
verification passes. Confirmation replay on the primary seed before any bake
proposal. Bake decisions are the user's.

**Pre-approved GPU:** retroactive canaries (model_4999, model_9998) + the scout
segment only (5k–10k from the anchor with all four gait income terms at ε=0.001,
env-var only). Then PAUSE, push-notify, report collapse/middle/jackpot, wait for
user decision on the round schedule.

## Ledger

### 2026-08-31 14:20 — pre-GPU: reward-manager ε unit tests PASS
`tests/unit/test_parkour_reward_manager_epsilon.py` (5 tests, isaac venv pytest):
weight-0.0 terms skipped with flat telemetry (func never called); ε=1e-3 terms
compute/contribute/log; total-reward ≥0 clip pinned; episodic sums linear in weight
(validates unit-weight normalization); telemetry NOT clipped (signed penalty sums
survive).

### 2026-08-31 14:25 — pre-GPU: income-budget table (r1 window, seed-3 seg1 TB)
Source: `parkour/logs/rsl_rl/crab_hex_flat_walk/2026-08-31_06-04-16` (5k–10k, full
weights), tail mean n=20, RSL-RL Episode_Reward units.

| bucket | income |
|---|---|
| gait income (clock 0.389 + apex 0.071 + airtime 0.0006 + stride 0.00004) | **+0.460** |
| other positive (track_lin 0.798, track_ang 0.775, yaw 0.022, progress 0.027) | +1.623 |
| penalties (action_rate −0.178, track_L1 −0.119, delta_torques −0.043, rest ≤0.02 each) | −0.415 |
| net at full weights | +1.668 |
| **projected scout net (gait income → 0, pessimistic)** | **+1.208** |

**VERDICT: PASS** — the scout keeps ~72% of current net income; the ≥0 clip floor is
not in play (teacher-hand-off economics do NOT recur here: tracking terms dominate).
Context: crab_failure tail 0.194, mean ep len 897, value loss 0.0076 (pre-ramp
baseline for the 3× watchdog).

**Finding for the notch order:** `feet_air_time_positive` (+0.0006) and
`stride_length` (+0.00004) are already economically dead in this window — 3 orders
below clock. Their notches are near-no-ops economically; any effect they retain is
gradient-shaping, not income. Clock (0.389) is the only economically live gait term;
apex (0.071) is second.

### 2026-08-31 14:35 — pre-GPU: floor-weight offline replay GATE PASS
`floor_weight_replay.py` (campaign dir): behavioral economy of the remaining stack
(gait income at ε) replayed over the tripod-era fixture set (2 healthy + 4 degenerate
families). Behavioral net /min: healthy 121–122 > tiprock 99 > lunge 95 ≫ skate −78 ≫
drag −1746. **Healthy out-earns every degenerate family.** Two structural findings:
1. **`foot_idle` (kept at −0.12) is the load-bearing anti-skate/drag defense** at
   floor weights (−177/min on skate, −1799/min on drag). Without it, skate would net
   within ~20% of healthy — do not weaken the gait penalties during this campaign.
2. The healthy-vs-tiprock margin is only ~19%, carried by track_lin — the creep
   watchdog (completion ≥0.90 with tracking <0.40 → abort) stays essential.
Caveats recorded in the script header: fixtures are old-plant/position-era traces —
ordering is behavior-level and transferable, absolute incomes are not;
action-stream regularizers excluded from the gate (era-nontransferable and
phase-out-invariant); tibia_deviation not replayable (no joint_pos in traces).
> NOTIFY: scout done — MIDDLE (partial degradation — notch search discovers per-window income needs); campaign PAUSED for user decision

### 2026-08-31 17:35 — USER DECISION: round search approved ("go ahead")
Scout world MIDDLE (tripod 1.00×/tracking +10% at ε income; completion 0.48 vs 0.99,
failure +0.174 — income buys survival persistence, not gait shape). Orchestrator
relaunched with --start-rounds: r1 candidates in order clock_half, apex_half,
airtime_half, stride_half; r1 control = seg1 records; watchdogs armed.

### 2026-09-01 — USER DECISION: r4/r5 opt-in ("opt into the r4 window to test clock_low/clock_eps")
Extend the search past the r3 cap: r4 (20k–25k) tests clock_low (0.5→0.2), r5
(25k–30k) tests clock_eps (0.2→ε) — sequential unlock requires one window each.
max_round=5. Terminal iteration-matched comparison will extend the full-income
reference by two matched segments. Mirror-loss elements remain out of scope
(follow-up campaign per plan). To apply on orchestrator exit at
await_user_terminal: state.json phase→"round", max_round→5, relaunch.
> NOTIFY: phase-out rounds complete — settled weights [reward_clock_schedule=0.5, reward_clock_swing_apex=0.001, reward_feet_air_time_positive=0.001, reward_stride_length=0.001]; PAUSED before terminal phase
> NOTIFY: phase-out rounds complete — settled weights [reward_clock_schedule=0.001, reward_clock_swing_apex=0.001, reward_feet_air_time_positive=0.001, reward_stride_length=0.001]; PAUSED before terminal phase

### 2026-09-02 — USER DECISION: terminal phase, all 3 steps ("run all 3")
Extended full-income control (REF_HEAD 20k→30k, 2 segments) → RSI-off hold (+2k,
RSI_FRAC=0, gate ratio ≥0.85 vs final bake + no creep) → seed-2 replay of the full
combined schedule (0→30k, 6 segments, P0-null bank per confirm-replay precedent)
→ terminal verdict table vs the matched 30k control. Orchestrator extended with
phase_terminal (resumable), launched with --start-terminal.

### 2026-09-02 — USER DECISION: BAKE ("go ahead and stop the RSI and seed 2 verification. Bake this version of the curriculum")
RSI-off verification + seed-2 replay STOPPED/WAIVED. Baked: the combined
phase-out schedule (see REPORT BAKE block) with reference of record
`2026-09-02_00-42-53/model_29994.pt` (30k). Campaign CLOSED.

### 2026-09-02 — post-bake obstacle check (user-requested after videos)
30k head: light-obstacle completion 0.26 ≈ hard 0.27 — fails at obstacle onset regardless of
difficulty (ref-of-record scored 0.64 on light). 20k head videos recorded for side-by-side.
Bake caveat strengthened: 30k head is a poor obstacle deployment candidate; 20k round-3 bake
head (obstacle 0.60) is the stronger checkpoint on this lineage. Decision left to the user.
