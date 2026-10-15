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
