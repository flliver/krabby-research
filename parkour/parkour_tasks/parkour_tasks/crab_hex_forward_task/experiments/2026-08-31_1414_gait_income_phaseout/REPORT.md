<!-- paths-note -->
> **Paths note (2026-09-09):** this campaign moved from `sim_fine_tuning/2026-08-31_1414_gait_income_phaseout/` to `parkour/parkour_tasks/parkour_tasks/crab_hex_forward_task/experiments/2026-08-31_1414_gait_income_phaseout/`. Absolute paths below (`/home/.../sim_fine_tuning/...`, `parkour/logs/rsl_rl/...`) name raw artifacts that stay on disk untracked; the tracked checkpoint of record is `head/` (see `bundle.yaml`) and the eval summaries are in place / under `evals/`.

# Gait-income phase-out — REPORT

Reporting contract: blocks terminated by `>>> ENTRY <marker>` markers, relayed to chat
by the monitor. Campaign charter in CHANGELOG.md; approved plan at
`/home/nickmagus/.claude/plans/there-are-a-variety-federated-frog.md`.

## CAMPAIGN OPEN — 2026-08-31 14:14
- lineage: seed-3 (reference-of-record seed); 5k anchor `2026-08-31_03-42-16/model_4999.pt`
- pre-GPU queue: reward-manager ε unit test | income-budget table (seg-1 TB) | offline degenerate replay at floor weights | ramp curriculum term + tests | orchestrator fork
- GPU queue (pre-approved): retroactive canaries (model_4999, model_9998) → scout (5k–10k, gait income at ε) → PAUSE for user
>>> ENTRY campaign open
## ENTRY GATE — retroactive canaries + r1 control record (seed-3 lineage)
- 5k anchor canary (2026-08-31_03-42-16/model_4999): tripod 0.595 | completion 0.880 | tracking 0.397 | slip 0.321
- r1 control canary (2026-08-31_06-04-16/model_9998): tripod 0.599 | completion 0.990 | tracking 0.413 | slip 0.199
- r1 control probe (seg1 TB tail): unit-clock 0.389 | track income 0.798 | failure 0.194 | ep len 896.876 | vloss 0.0078 | mean reward 36.820
- formation band (unit-clock >= 0.38, failure <= 0.30): WITHIN band
>>> ENTRY entry gate canaries

## SCOUT — all gait income at eps for the full 5k-10k window
| metric | scout | seg1 control | margin |
|---|---|---|---|
| unit-weight clock income | 0.325 | 0.389 | 0.84x |
| tracking income | 0.817 | 0.798 | 1.02x |
| failure tail | 0.368 | 0.194 | +0.174 |
| value loss | 0.0067 | 0.0078 | 0.9x |
| mean reward | 22.607 | 36.820 | 0.61x |
| canary t/c/tr/slip | 0.599/0.480/0.455/0.298 | 0.599/0.990/0.413/0.199 | tripod 1.00x |
- **WORLD: MIDDLE (partial degradation — notch search discovers per-window income needs)**
- next: PAUSED for user decision on the round schedule (plan Phase 0.2)
>>> ENTRY scout verdict

## NOTCH decision — reward_clock_schedule 1.0 -> 0.5 @ round 1
| metric | candidate | reference | margin |
|---|---|---|---|
| unit-clock (settled) | 0.364 | 0.389 | 0.94x |
| track income | 0.795 | 0.798 | 1.00x |
| failure tail | 0.250 | 0.194 | +0.055 |
| canary t/c/tr | 0.574/0.990/0.439 | 0.599/0.990/0.413 | |
- escalation history: ambiguous at 2k (unit-clock 0.94x, track 1.00x, fail +0.055); continued to 5k
- **DECISION: PASS** — escalated canary: tripod 0.96x >= 0.85 and ratio 0.96 >= 0.85
>>> ENTRY decision reward_clock_schedule@0.5 round1 PASS

## NOTCH decision — reward_clock_swing_apex 1.0 -> 0.5 @ round 1
| metric | candidate | reference | margin |
|---|---|---|---|
| unit-clock (settled) | 0.399 | 0.364 | 1.10x |
| track income | 0.882 | 0.795 | 1.11x |
| failure tail | 0.233 | 0.250 | -0.017 |
- **DECISION: PASS** — fail -0.017 <= +0.05, unit-clock 1.10x >= 0.85, track 1.11x >= 0.85 (settled tail)
>>> ENTRY decision reward_clock_swing_apex@0.5 round1 PASS

## NOTCH decision — reward_feet_air_time_positive 0.8 -> 0.4 @ round 1
| metric | candidate | reference | margin |
|---|---|---|---|
| unit-clock (settled) | 0.385 | 0.399 | 0.96x |
| track income | 0.819 | 0.882 | 0.93x |
| failure tail | 0.293 | 0.233 | +0.060 |
| canary t/c/tr | 0.595/0.970/0.421 | 0.574/0.990/0.439 | |
- escalation history: ambiguous at 2k (unit-clock 0.96x, track 0.93x, fail +0.060); continued to 5k
- **DECISION: PASS** — escalated canary: tripod 1.04x >= 0.85 and ratio 0.96 >= 0.85
>>> ENTRY decision reward_feet_air_time_positive@0.4 round1 PASS

## NOTCH decision — reward_stride_length 0.5 -> 0.25 @ round 1
| metric | candidate | reference | margin |
|---|---|---|---|
| unit-clock (settled) | 0.446 | 0.385 | 1.16x |
| track income | 0.920 | 0.819 | 1.12x |
| failure tail | 0.153 | 0.293 | -0.140 |
- **DECISION: PASS** — fail -0.140 <= +0.05, unit-clock 1.16x >= 0.85, track 1.12x >= 0.85 (settled tail)
>>> ENTRY decision reward_stride_length@0.25 round1 PASS

## BAKE — round 1 boundary (10k)
- notches: [reward_clock_schedule:1.0->0.5, reward_clock_swing_apex:1.0->0.5, reward_feet_air_time_positive:0.8->0.4, reward_stride_length:0.5->0.25]
- backstop vs round control: unit-clock 0.94x (>=0.85) | failure +0.170 (<=+0.10) | canary ratio 0.94 (>=0.85)
- checkpoint: 2026-08-31_23-49-32/model_9998.pt
- flat canary: tripod 0.614 | completion 0.930 | tracking 0.480 | slip 0.238
- all-obstacle eval: completion 0.450 | tripod 0.573
- **FAIL**
- ejecting most costly notch: **reward_clock_schedule -> 0.5** — retries a later round
>>> ENTRY bake round1 EJECT reward_clock_schedule

## BAKE — round 1 boundary (10k)
- notches: [reward_clock_swing_apex:1.0->0.5, reward_feet_air_time_positive:0.8->0.4, reward_stride_length:0.5->0.25]
- backstop vs round control: unit-clock 1.18x (>=0.85) | failure -0.070 (<=+0.10) | canary ratio 0.97 (>=0.85)
- checkpoint: 2026-09-01_02-22-26/model_9998.pt
- flat canary: tripod 0.582 | completion 1.000 | tracking 0.443 | slip 0.218
- all-obstacle eval: completion 0.730 | tripod 0.539
- **PASS**
>>> ENTRY bake round1 PASS

## RSI refresh — bank rsi_bank_pg_r1.npz adopted (backstop passed)
>>> ENTRY rsi r1

## BASELINE control — round 2 (r2_008_control)
- probe: unit-clock 0.422 | track 0.896 | failure 0.231 | vloss 0.0241 | mean reward 25.895
- flat canary: tripod 0.607 | completion 0.990 | tracking 0.467 | slip 0.204
>>> ENTRY baseline round2

## NOTCH decision — reward_clock_schedule 1.0 -> 0.5 @ round 2
| metric | candidate | reference | margin |
|---|---|---|---|
| unit-clock (settled) | 0.349 | 0.422 | 0.83x |
| track income | 0.755 | 0.896 | 0.84x |
| failure tail | 0.372 | 0.231 | +0.141 |
- **DECISION: FAIL** — fail +0.141 > +0.10 or unit-clock 0.83x < 0.70
>>> ENTRY decision reward_clock_schedule@0.5 round2 FAIL

## NOTCH decision — reward_clock_swing_apex 0.5 -> 0.001 @ round 2
| metric | candidate | reference | margin |
|---|---|---|---|
| unit-clock (settled) | 0.415 | 0.422 | 0.98x |
| track income | 0.864 | 0.896 | 0.96x |
| failure tail | 0.244 | 0.231 | +0.013 |
- **DECISION: PASS** — fail +0.013 <= +0.05, unit-clock 0.98x >= 0.85, track 0.96x >= 0.85 (settled tail)
>>> ENTRY decision reward_clock_swing_apex@0.001 round2 PASS

## NOTCH decision — reward_feet_air_time_positive 0.4 -> 0.001 @ round 2
| metric | candidate | reference | margin |
|---|---|---|---|
| unit-clock (settled) | 0.411 | 0.415 | 0.99x |
| track income | 0.870 | 0.864 | 1.01x |
| failure tail | 0.183 | 0.244 | -0.061 |
- **DECISION: PASS** — fail -0.061 <= +0.05, unit-clock 0.99x >= 0.85, track 1.01x >= 0.85 (settled tail)
>>> ENTRY decision reward_feet_air_time_positive@0.001 round2 PASS

## NOTCH decision — reward_stride_length 0.25 -> 0.001 @ round 2
| metric | candidate | reference | margin |
|---|---|---|---|
| unit-clock (settled) | 0.397 | 0.411 | 0.97x |
| track income | 0.842 | 0.870 | 0.97x |
| failure tail | 0.238 | 0.183 | +0.055 |
| canary t/c/tr | 0.597/0.990/0.455 | 0.607/0.990/0.467 | |
- escalation history: ambiguous at 2k (unit-clock 0.97x, track 0.97x, fail +0.055); continued to 5k
- **DECISION: PASS** — escalated canary: tripod 0.98x >= 0.85 and ratio 0.97 >= 0.85
>>> ENTRY decision reward_stride_length@0.001 round2 PASS

## BAKE — round 2 boundary (15k)
- notches: [reward_clock_swing_apex:0.5->0.001, reward_feet_air_time_positive:0.4->0.001, reward_stride_length:0.25->0.001]
- backstop vs round control: unit-clock 0.92x (>=0.85) | failure +0.051 (<=+0.10) | canary ratio 0.97 (>=0.85)
- checkpoint: 2026-09-01_10-45-00/model_14997.pt
- flat canary: tripod 0.593 | completion 0.960 | tracking 0.487 | slip 0.203
- all-obstacle eval: completion 0.550 | tripod 0.578
- **PASS**
>>> ENTRY bake round2 PASS

## RSI refresh — bank rsi_bank_pg_r2.npz adopted (backstop passed)
>>> ENTRY rsi r2

## BASELINE control — round 3 (r3_014_control)
- probe: unit-clock 0.381 | track 0.822 | failure 0.304 | vloss 0.0201 | mean reward 23.655
- flat canary: tripod 0.556 | completion 0.980 | tracking 0.443 | slip 0.203
>>> ENTRY baseline round3

## NOTCH decision — reward_clock_schedule 1.0 -> 0.5 @ round 3
| metric | candidate | reference | margin |
|---|---|---|---|
| unit-clock (settled) | 0.385 | 0.381 | 1.01x |
| track income | 0.866 | 0.822 | 1.05x |
| failure tail | 0.316 | 0.304 | +0.012 |
- **DECISION: PASS** — fail +0.012 <= +0.05, unit-clock 1.01x >= 0.85, track 1.05x >= 0.85 (settled tail)
>>> ENTRY decision reward_clock_schedule@0.5 round3 PASS

## BAKE — round 3 boundary (20k)
- notches: [reward_clock_schedule:1.0->0.5]
- backstop vs round control: unit-clock 0.88x (>=0.85) | failure +0.072 (<=+0.10) | canary ratio 1.01 (>=0.85)
- checkpoint: 2026-09-01_15-10-31/model_19996.pt
- flat canary: tripod 0.561 | completion 0.990 | tracking 0.426 | slip 0.227
- all-obstacle eval: completion 0.600 | tripod 0.543
- **PASS**
>>> ENTRY bake round3 PASS

## RSI refresh — bank rsi_bank_pg_r3.npz adopted (backstop passed)
>>> ENTRY rsi r3

## ROUNDS COMPLETE — settled weights: reward_clock_schedule=0.5, reward_clock_swing_apex=0.001, reward_feet_air_time_positive=0.001, reward_stride_length=0.001 | head /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_flat_walk/2026-09-01_15-10-31/model_19996.pt — terminal phase (RSI-off verification + primary-seed replay) awaits user go-ahead
>>> ENTRY rounds complete

## BASELINE control — round 4 (r4_017_control)
- probe: unit-clock 0.300 | track 0.684 | failure 0.576 | vloss 0.0179 | mean reward 16.620
- flat canary: tripod 0.576 | completion 0.810 | tracking 0.453 | slip 0.201
>>> ENTRY baseline round4

## NOTCH decision — reward_clock_schedule 0.5 -> 0.2 @ round 4
| metric | candidate | reference | margin |
|---|---|---|---|
| unit-clock (settled) | 0.326 | 0.300 | 1.09x |
| track income | 0.752 | 0.684 | 1.10x |
| failure tail | 0.441 | 0.576 | -0.135 |
- **DECISION: PASS** — fail -0.135 <= +0.05, unit-clock 1.09x >= 0.85, track 1.10x >= 0.85 (settled tail)
>>> ENTRY decision reward_clock_schedule@0.2 round4 PASS

## BAKE — round 4 boundary (25k)
- notches: [reward_clock_schedule:0.5->0.2]
- backstop vs round control: unit-clock 1.01x (>=0.85) | failure -0.139 (<=+0.10) | canary ratio 1.01 (>=0.85)
- checkpoint: 2026-09-01_19-40-14/model_24995.pt
- flat canary: tripod 0.584 | completion 0.890 | tracking 0.415 | slip 0.214
- all-obstacle eval: completion 0.410 | tripod 0.606
- **PASS**
>>> ENTRY bake round4 PASS

## RSI refresh — bank rsi_bank_pg_r4.npz adopted (backstop passed)
>>> ENTRY rsi r4

## BASELINE control — round 5 (r5_020_control)
- probe: unit-clock 0.296 | track 0.692 | failure 0.442 | vloss 0.0176 | mean reward 16.024
- flat canary: tripod 0.615 | completion 0.910 | tracking 0.437 | slip 0.224
>>> ENTRY baseline round5

## NOTCH decision — reward_clock_schedule 0.2 -> 0.001 @ round 5
| metric | candidate | reference | margin |
|---|---|---|---|
| unit-clock (settled) | 0.285 | 0.296 | 0.96x |
| track income | 0.678 | 0.692 | 0.98x |
| failure tail | 0.481 | 0.442 | +0.038 |
- **DECISION: PASS** — fail +0.038 <= +0.05, unit-clock 0.96x >= 0.85, track 0.98x >= 0.85 (settled tail)
>>> ENTRY decision reward_clock_schedule@0.001 round5 PASS

## BAKE — round 5 boundary (30k)
- notches: [reward_clock_schedule:0.2->0.001]
- backstop vs round control: unit-clock 0.91x (>=0.85) | failure +0.042 (<=+0.10) | canary ratio 0.87 (>=0.85)
- checkpoint: 2026-09-02_00-42-53/model_29994.pt
- flat canary: tripod 0.563 | completion 0.790 | tracking 0.407 | slip 0.234
- all-obstacle eval: completion 0.270 | tripod 0.604
- **PASS**
>>> ENTRY bake round5 PASS

## RSI refresh — bank rsi_bank_pg_r5.npz adopted (backstop passed)
>>> ENTRY rsi r5

## ROUNDS COMPLETE — settled weights: reward_clock_schedule=0.001, reward_clock_swing_apex=0.001, reward_feet_air_time_positive=0.001, reward_stride_length=0.001 | head /home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_flat_walk/2026-09-02_00-42-53/model_29994.pt — terminal phase (RSI-off verification + primary-seed replay) awaits user go-ahead
>>> ENTRY rounds complete

## TERMINAL — extended full-income control at 30k (matched iterations)
- checkpoint: 2026-09-02_09-51-23/model_29994.pt
- probe (25k-30k tail): unit-clock 0.321 | track 0.722 | failure 0.516
- flat canary: tripod 0.615 | completion 0.730 | tracking 0.460 | slip 0.205
- all-obstacle eval: completion 0.190 | tripod 0.596
>>> ENTRY terminal extctrl

## BAKE — GAIT-INCOME PHASE-OUT SCHEDULE (user decision, 2026-09-02)
**REFERENCE OF RECORD: `parkour/logs/rsl_rl/crab_hex_flat_walk/2026-09-02_00-42-53/model_29994.pt`** (30k, seed-3 lineage)

Schedule of record (combined curriculum, single flat-task lineage):
- 0–5k: formation — baseline core, full gait income (clock 1.0, apex 1.0, airtime 0.8, stride 0.5)
- @5k: +7 curriculum elements (yaw, terrain50, recal, terrain-curriculum, DR push, DR mass/CoM, safety pack); apex→0.5, airtime→0.4, stride→0.25 (cosine, 1k iters)
- @10k: +3 elements (clearance pack, turning, goal-vel); apex/airtime/stride→ε
- @15k: clock→0.5 | @20k: clock→0.2 | @25k: clock→ε
- ε = 0.001 floor throughout (weight-0.0 kills term telemetry); RSI 0.2, mirror loss 0.5, gait penalties unchanged.

Terminal evidence at matched 30k (vs extended full-income control `2026-09-02_09-51-23/model_29994.pt`):
- obstacle completion 0.270 vs 0.190 (**obstacle WIN criterion met**, ≥ ctrl+0.05)
- flat completion 0.790 vs 0.730 | failure ~0.48 vs 0.516 | tripod 0.563 vs 0.615 (−0.052; hold band missed by 0.002, within single-canary noise) | tracking 0.407 vs 0.460
- Full-income control degraded MORE late — gait income at full weight bought no late-stage protection.

Verification state at bake: extended full-income control DONE; **RSI-off verification WAIVED** and **seed-2 schedule replay WAIVED** (user decision — orchestrator stopped mid-RSI-off-hold at ~iter 266). The self-sustaining-without-RSI claim and cross-seed reproducibility are therefore UNVERIFIED for this bake; the clock OBSERVATION remains in the policy input (claims are about income, not the clock signal).
>>> ENTRY BAKE phase-out schedule (user)
## POST-BAKE CHECK — obstacle competence of the 30k head (2026-09-02, user-requested)
Prompted by videos falling at the first obstacles on every tier.
- 30k head on LIGHT obstacles (shallow geometry, diff 0.05–0.20, all-obstacle): **completion 0.26**
  (fall 74 / complete 26); tripod 0.606; stand hold survived by 100/100, creep entered by 97,
  low reached by 40. Falls cluster at steps 750–1000 (median 864 ≈ 5 s after walking starts).
- Same head on HARD obstacles (recal, 0.20–0.70): 0.27 with the identical failure-timing signature.
  => obstacle completion is INVARIANT to difficulty: the 30k head fails at obstacle ONSET, not on
  tall hurdles. Reference of record (graduation, 20k, full income) scored 0.64 on this light config.
- Lineage context (hard config): 10k bake 0.73 → 15k 0.55 → 20k 0.60 → 25k 0.41 → 30k 0.27;
  extended full-income control at 30k: 0.19. The 25k–30k windows destroyed obstacle initiation in
  BOTH arms — late-training/terrain-promotion degradation, not a phase-out cost, but the baked 30k
  head is a poor deployment candidate on any obstacle terrain.
- Videos: 20k head (`2026-09-01_15-10-31/model_19996.pt`, clock 0.5 / satellites ε, obstacle 0.60)
  recorded on the same medium/heavy tiles for side-by-side (`videos/play/obstacles_{medium,heavy}_20k.mp4`).
- Bake status unchanged (user decision pending): candidate alternative reference for deployment =
  the 20k round-3 bake head (flat 0.561/0.990/0.426, obstacle 0.60), or cap the schedule at 20k.
>>> ENTRY post-bake obstacle check
