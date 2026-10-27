# REPLAY_GATE_CAT: per-category stroke plan, step 4 (offline replay gate)

Plan: `~/.claude/plans/per-category-stroke-weights.md`, Phase A step 4 (decisions 3 and 7; implementation items 10, 11
and 15). The run used CPU only, in worktree `feat/omni-stroke-posture` at `6d660ce2`. It did not start Isaac and did not
use the GPU.

## Verdict: STOP-and-ask. Four restated-criterion FAILs, one of them new at the table.

| # | Economy | Criterion | What failed | Depends on the table? |
|---|---|---|---|---|
| 1 | t3 | **R5 in-plant, no-flip** | Bf_s3 vs E7 creeper on move_turn. The ratio fell from 1.120 without the stroke term to 1.047 with it, under the 1.10 floor. The ordering still holds (0.528 > 0.504). | **Yes (new).** It comes from the creeper's move_turn ω×r credit at w_mt = 0.1362. |
| 2 | t3 | **R8 hand gap** | E3 turn gap −0.214, 95 % CI [−0.368, −0.072], which excludes 0. | No (scale-invariant), but the heads are new. p15 never priced a t3 head. |
| 3 | t3 | **R3 tap test** (side) | Spearman(pay per stance, switch rate) is below 0 on all three t3 heads: E3 −0.058 [−0.199, +0.087], Bf_s3 −0.177 [−0.315, −0.019], Bf_s2 −0.100 [−0.229, +0.035]. | No (scale-invariant). The heads are new. |
| 4 | W | **R3 tap test** (side), if every head counts | E1 (X08:c6) side −0.153 [−0.283, −0.008]. W itself reads **+0.191 (PASS)**; X0x s3 +0.009 and s2 +0.032. | No. This is identical to p15, so the value was already there when p15 was accepted. |

Notes on the table:
- **FAIL 4 depends on the clause's scope.**
  - SA0 decision 6 quotes only "W side". The p15 check (REPLAY_GATE_CHECK.md) looked only at W.
  - Read as "the head being fine-tuned", the clause passes. Read as "every healthy head", it fails.
  - I count it as a FAIL, per the instruction not to rationalise one away. It is for the user to rule.
- **The stride_per_speed band passes:** W 0.0140 and t3 0.0213, both against the 0.08 STOP.
- **Both dose tables reproduce the plan's rows to within 0.25 %.**
- **R7 passes** in both economies.
- **R5 passes on W.**

`any_fail = true`.

## 1. Solved dose tables (replay scale, delivered share 10 % per category, hand-min, fwd = back tie)

The configs are `config_W.json` and `config_t3.json`. Run each with
`replay_stroke_posture.py gate --config <cfg> --out <W|t3>`.

**W economy (clock-free).** Economy: W's `params/env.yaml`. Dose run: W.

| | fwd | back | side | diag | turn | move_turn |
|---|---|---|---|---|---|---|
| solved w_c (per category) | 0.08647 | 0.1120 (then tied) | 0.6147 (bound by hand r) | 0.1593 | 0.1533 (bound by hand r) | not solved |
| **effective weight** | **0.08647** | **0.08647** | **0.6147** | **0.1593** | **0.1533** | **0.08647** (m = 1) |
| plan row | 0.0865 | 0.0865 | 0.615 | 0.159 | 0.153 | (0.567 was the equal-share value; decision 7 holds m = 1) |
| difference | 0.04 % | 0.04 % | 0.05 % | 0.2 % | 0.2 % | -- |
| delivered share, W | 0.100 | 0.077 | 0.100 | 0.100 | 0.099 | 0.014 |
| X0x s3 / X0x s2 / E1 | .086 / .096 / .089 | .071 / .075 / .073 | .103 / .108 / .099 | .101 / .109 / .094 | .082 / .062 / .078 | .012 / .013 / .010 |

Knob values at replay scale:
- `KRABBY_OMNI_STROKE_W=0.08646779396708197`
- `KRABBY_OMNI_STROKE_CAT_W=fwd:1.0,back:1.0,side:7.109225977086123,diag:1.8426117426622708,turn:1.7727771075114038,move_turn:1.0`

**t3 economy (clock-held).**
- Economy: X08:t3's `2026-09-22_09-10-29_seed3_X08_t3_A15pB/params/env.yaml`. Its clock weights are 1.0 / 1.0, and the
  tool asserted them.
- Dose run: E3.

| | fwd | back | side | diag | turn | move_turn |
|---|---|---|---|---|---|---|
| solved w_c | 0.1362 | 0.1727 (then tied) | 0.5653 (bound by hand r) | 0.1796 | 0.2377 (bound by hand r) | not solved |
| **effective weight** | **0.1362** | **0.1362** | **0.5653** | **0.1796** | **0.2377** | **0.1362** (m = 1) |
| plan row | 0.136 | 0.136 | 0.565 | 0.180 | 0.238 | (0.859 was the equal-share value; m = 1) |
| difference | 0.16 % | 0.16 % | 0.06 % | 0.2 % | 0.1 % | -- |
| delivered share, E3 | 0.100 | 0.078 | 0.099 | 0.100 | 0.091 | 0.015 |
| Bf_s3 / Bf_s2 | .079 / .105 | .068 / .079 | .104 / .123 | .090 / .098 | .076 / .055 | .010 / .011 |

Knob values at replay scale:
- `KRABBY_OMNI_STROKE_W=0.1362210975498265`
- `KRABBY_OMNI_STROKE_CAT_W=fwd:1.0,back:1.0,side:4.150041141570944,diag:1.318716993366581,turn:1.7449369222400448,move_turn:1.0`

Notes:
- **These are replay-scale values.** The training values come from step 6/9: fwd, back, side and diag are divided by
  each category's smoke r_c; turn and move_turn stay as they are (plan Design).
- **No category is above 15 % on any head.** The largest is Bf_s2 side at 0.123.
- **The tables mix hand conventions (item 15).** Side and turn are bound by their lower hand; fwd, back and diag are
  unpaired. The pooled alternatives:
  - W: side 0.6148 (+0.02 %), turn 0.1551 (+1.2 %).
  - t3: side 0.5684 (+0.5 %), turn 0.2603 (**+9.5 %**, because E3's turn hand gap is large; see FAIL 2).
  - The t3 turn difference is larger than the plan's "< 3 %".

## 2. Stamps, and the clock check for the t3 economy (item 10)

**W economy.** It uses the p15 audited set as-is: W, X0x s3/s2 and E1 (healthy); the sideways head (flat and step);
Z_t3, Z_c6 and A4500 (posture); E6 statue and E7 creeper (degenerate).

**t3 economy.**
- **Healthy:** E3 (X08:t3, clock visible), Bf_s3 and Bf_s2. `stamp_clock` gives (1.0, 1.0) for each.
- **Degenerate, kept:** E6 statue (checkpoint X08:c6) and E7 creeper (`2026-09-17_00-13-40/model_6997`). Both read
  (1.0, 1.0), so both are valid in a clock-held economy.
- **Sideways head, kept:** `crab_hex_side_teacher/2026-09-12_03-04-37/model_9998`, clock-held (1.0, 1.0). It is used
  for the R1–R3/R6/P1 rows.
- **Dropped:**
  - Z_t3, Z_c6 and A4500 (posture, the R5 no-flip set). `stamp_clock` gives (0.001, 0.001). Pricing these clock-free
    heads in a clock-held economy would misstate their clock income.
  - W, X0x and E1. They are not t3-setting heads; the plan names E3 plus B_f.
  - The forward-era degenerates, as in p15. The tool cannot load them.
- Full listing: `clock_check.txt` (from `scripts/clocks.py`).

## 3. Restated criteria per economy (SA0 decisions 6–7)

Source: `scripts/restated_cat.py` → `<W|t3>/restated_cat.json`. The tool's literal R-lines in `REPLAY_GATE.md` are
not the gate.

| Criterion | W economy | t3 economy | Scale-invariant? |
|---|---|---|---|
| **R3a** short-stance share ≤ 2 % | PASS (max E1.side 0.96 %) | PASS (max Bf_s3.side 1.06 %) | Yes. Identical to p15 on W. |
| **R3b** tap test: Spearman(pay per stance, switch rate) ≥ 0 on side | W +0.191 PASS; X0x s3 +0.009, s2 +0.032; **E1 −0.153 FAIL** (CI excludes 0) | **FAIL**: E3 −0.058, Bf_s3 −0.177 (CI excludes 0), Bf_s2 −0.100 | Yes. Identical to p15 on W. |
| **R5 no-flip** (healthy + posture vs degenerate totals, at the table) | PASS (84 rows, 0 fails) | **FAIL**: 1 of 36 rows, Bf_s3 > E7 creeper on move_turn, ratio 1.120 → 1.047 < 1.10 floor | **No. New evidence.** |
| **R5** creeper term-alone ≤ 0.5 × healthy (translation) | PASS (max 0.143, creeper/X0x_s3 fwd) | PASS (max 0.315, creeper/Bf_s3 fwd) | Yes (ratio within a category). t3 is new heads. |
| R5 cross-plant (forward-era lunge) | known FAIL, not counted (unison flag) | same | -- |
| **R7** clip shift ≤ 5 pp | PASS (max \|shift\| 0.62 pp, E1.diag; p15 0.51) | PASS (max 0.41 pp, Bf_s3.diag) | **No. New evidence.** |
| **R7** absorbed ≤ 15 % | PASS (max 10.9 %, E1.move_turn; W.side 1.4 %) | PASS (max 10.7 %, Bf_s3.move_turn; side ≤ 4.8 %) | **No. New evidence.** |
| **R8** hand gap: FAIL only if \|gap\| > 10 % and its 95 % CI excludes 0 | PASS (max E1.move_turn +0.120, CI [−0.051, +0.292]) | **FAIL**: E3.turn −0.214, CI [−0.368, −0.072]. Other rows pass: E3.move_turn −0.150 [−0.307, +0.023], Bf_s2.move_turn +0.211 [−0.025, +0.466]. | Yes. Identical to p15 on W. |

**Scale-invariant criteria (item 11).**
- R3a/b, R8 and R5 term-alone are ratios or rank statistics within one category, and within a category the multiplier
  is a constant. The table cannot move them.
- On W they reproduce p15 exactly: short share 0.96 % (E1.side), hand gap 0.120 (E1.move_turn), creeper term-alone
  0.143. They are **not fresh evidence** there.
- At t3 they are new only because the heads are new.

**What carries information at the table.**
- R5 no-flip.
- R7.
- The creeper's move_turn ω×r credit:

| Economy | w_move_turn | Creeper S/s ÷ healthy S/s (move_turn) | Creeper credit as a share of the healthy head's move_turn tracking | Healthy credit, own share | Creeper excess |
|---|---|---|---|---|---|
| W | 0.0865 (= p15's w_S) | 1.35–1.91× | 1.9–2.1 % | 1.1–1.5 % | 0.5–1.0 % |
| t3 | 0.1362 | **2.17–3.19×** | 3.3–3.6 % | 1.1–1.5 % | **1.8–2.5 %** |

**Reading FAIL 1.**
- **Why the margin is thin.** On Bf_s3's move_turn, the healthy-over-creeper total is only 1.120 without the stroke
  term. Under the t3 economy the creeper earns 3.2× Bf_s3's per-second move_turn stroke income, through ω×r (SA0
  decision 2 kept that credit).
- **What would pass.** Linear in w_mt, the row holds 1.10 only for w_mt ≲ 0.034 (m_move_turn ≈ 0.25 at t3's w_fwd).
- **Why no decision 7 table passes it.** Decision 7's m = 1 gives w_mt = w_fwd, which is 0.136 at replay scale.
- **The ordering never flips.** The clause fails on the 1.10 floor.
- **Decision 7 was made without this number.** The t3 formation screen had no clock-held replay gate, so no earlier
  record carries it.

## 4. stride_per_speed seed band (MEAN key, side)

Sources:
- `readout --json`: `band/B_s3_vs_B_s2.json`, `band/B_s2.json`, `band/Bf_s3_vs_Bf_s2.json` and `band/Bf_s2.json`.
  Bf_s2 is priced in X08:t3's economy.
- `scripts/band.py` → `band/stride_per_speed_band.json`.

| Setting | s3 | s2 | Seed Δ | Paired-bootstrap MDE (one pair) | **Pooled MDE** | **Band = max(\|Δ\|, pooled)** | vs 0.08 |
|---|---|---|---|---|---|---|---|
| W (B s3 / B s2) | 0.8138 | 0.7998 | +0.0140 | 0.0176 | 0.0125 | **0.0140** | PASS |
| t3 (B_f s3 / B_f s2) | 0.7954 | 0.7812 (2 episodes ≤ 0.03 m/s excluded) | +0.0141 | 0.0307 | 0.0213 | **0.0213** | PASS |

**How the pooled MDE was built.** No arm exists yet, so it is a stand-in for step 8's pooled MDE:
- Each seed's paired bootstrap was replaced by the B pair's own env-clustered paired bootstrap (s3 − s2), drawn twice
  with independent bootstrap seeds (the tool's `paired_bootstrap` with `return_raw`).
- The two draws were averaged element by element, and the centred 95 % half-width taken (`resolution.half_width`).

**Why not the tool's own `pooled`.** Run as `pooled` with arm = the other seed's base, it gives mean Δ 0 and pooled MDE
0, because the two draws cancel. It cannot be used as the proxy.

Even the one-pair MDE (0.0176 / 0.0307) is under a quarter of the 0.08 stop.

**Other keys, for reference (seed Δ; pooled MDE).**

| Setting | stride_along | achieved_speed |
|---|---|---|
| W | +0.0049; 0.0020 | +0.0039; 0.0017 |
| t3 | −0.0010; 0.0024 | +0.0020; 0.0038 |

The t3 stride_along gap is the degenerate one that plan step 10 floors at 0.0049.

## 5. FAIL list (STOP-and-ask per the plan)

1. **t3 R5 in-plant no-flip:** Bf_s3 vs E7 creeper, move_turn, ratio 1.047 < 1.10 at w_mt = 0.1362 (decision 7's
   m = 1). This depends on the table.
2. **t3 R8:** E3 turn hand gap −0.214, CI [−0.368, −0.072]. It is scale-invariant. It also means turn's hand-min dose
   pays E3's l hand less than 10 %.
3. **t3 R3 tap test:** side Spearman(pay per stance, switch rate) is negative on E3, Bf_s3 (CI excludes 0) and Bf_s2.
   It is scale-invariant.
4. **W R3 tap test on E1:** −0.153 (CI excludes 0). W itself is +0.191. It is scale-invariant and identical to p15.
   Whether it counts depends on the clause's scope (W only, or every healthy head).

The stride_per_speed band does not trip the stop in either setting.

## Decision (user, 2026-09-28 04:5x)

**STOP-and-ask resolved: "Proceed as planned".** Four criteria FAIL; each was judged as follows.

1. **t3 R5 no-flip** (B_f s3 vs creeper, move_turn, 1.120 → 1.047 < 1.10): no training effect. move_turn is not in t3's
   training pool, so the move_turn weight never pays in Phase C.
2. **t3 R8 turn hand gap, and the t3 / E1 side tap tests:** weight-table-invariant properties of the heads. The
   single-weight arms already trained at these heads and settings.

The fable check confirmed every number. It found nothing that puts Phase B's smoke or gate on a wrong number. One
provenance correction: the W tap-test values at p = 1.5 are recorded here for the first time.
