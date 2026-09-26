# REPLAY_GATE_CHECK: independent check of step 7 (fable, 2026-09-26)

Read-only re-computation of the step-7 doses and verdicts at `d716483b`. CPU only; nothing committed.
Scratch outputs: `/tmp/claude-1000/-home-nickmagus-krabby-krabby-research/3dbdec2f-0380-4df7-b155-e9e45cbd8414/scratchpad/check/`.

## Verdict

**The doses and every gate verdict reproduce exactly. No computational error found.** One labelling
problem must be fixed before SA0 (R5 is a FAIL as written, not a PASS), and the p = 2 recommendation is a
judgment call that the gate evidence does not decide. Details below.

## What was re-run

| Check | Method | Result |
|---|---|---|
| Full gate from `replay_gate_config.json` | `replay_stroke_posture.py gate` into a scratch dir, numeric diff of `criteria` | 0 differences; w_S 0.10335718, w_P 0.16121227 identical; all 13 pass/fail flags identical |
| R4 dose and binding row | tool re-run + `income` at w 0.103/0.161 | fwd delivered 0.0865 / tracking 0.868 = 0.100, binding fwd; back 0.071, turn 0.048, move_turn 0.012, diag 0.045, side 0.009 |
| P2 w_P | by hand from `cat_income` at w = 1 | dS = mean(0.970, 0.639) − 0.065 = 0.739; dP = mean(0.942, 0.752) − 0.373 = 0.474; w_P = 0.10336 × 0.739/0.474 = 0.16121; flat-only 0.1644 |
| R1 literal and restated, W + X0x s2 + sideways flat/step | own script, per-stance `ungated` / `after_anchor` / `after_slip` / `pay` | W turn 0.761 / 0.926 (anchor 17.8 %, slip 4.0 %, guard 2.1 %); X0x s2 turn 0.644 / 0.837 (anchor 23.1 %, slip 3.0 %, guard 9.5 %); sideways flat 0.872 / 0.936, step 0.788 / 0.912; all as recorded |
| R2 b + c | own script | slide (ρ > 0.8) kept 0.044 on the 4 runs loaded (tool pools 6 → 0.039); omni W side / sideways 0.067 flat, 0.102 step |
| R5 degenerate ratios | tool re-run | creeper term-alone translation max 0.113 (X0x s3 fwd); turn 0.29–0.51; move_turn 1.78–2.81; no-flip 0 fails on the 79 judged rows (5 Z-vs-creeper rows already reversed before the term, `pass=None`, narrow with it) |
| R7 absorbed | tool re-run | max 0.115 (E1 move_turn); W move_turn 0.104 with both terms |
| Forward-era term-alone | `replay_gate_calib/fwd_era.py` | lunge 5.124×, tip-rock 1.528×, skate 0.998×, drag 0.069×, v4 1.353 / 1.310× H2000 |
| R8 mirror parity on W | `replay_gate_calib/mirror.py W` | stroke max per-episode diff 0.0; centring 6.1e-5 |
| Turn-guard premise | own script: signed `wz / cmd_wz` on own-gated turn steps by hand | W +0.337 (l) / +0.328 (r), |wz|/|cmd| 0.57; X0x s2 +0.253 / +0.265, 0.53; sign(wz) = sign(cmd) on only 67–73 % of steps. Both hands agree, so the claim "heads turn at 0.24–0.43 of command, signed" is real, not a convention bug |
| Survival floors | per-category count from `metrics/episode_*.json` | stand 1.000, fwd 0.95, side 0.995, back 0.98, diag 0.87, turn 0.99 match; move_turn is 0.355 by schedule completion, 0.365 by the driver's `survival_by_hold` (the w2 statistic the report uses) |

## Blocking (labelling; numbers are right and already escalated)

1. **R5 is a FAIL as written, not "PASS (in-plant)".** Plan step 7 lists the forward-era degenerates as R5
   inputs "if their npz can be located; otherwise dropped". They were located and replayed, and the term-alone
   clause (≤ 0.5 × healthy on translation rows) fails on three of them: lunge 5.12×, tip-rock 1.53×, skate
   1.00×. The bottom line's "6 pass / 7 fail" and `replay_gate.json → step7.verdicts.literal.R5 = true` count
   it as a pass, and `replay_gate_config.json → dropped` still says "NOT loadable ... dropped" although the
   rollouts were priced. **Smallest fix:** in REPLAY_GATE.md's bottom line and §2 table mark R5 **FAIL
   (term-alone, cross-plant; in-plant clauses pass)** and restate the count as 5 pass / 8 fail; set
   `verdicts.literal.R5` false with the two clauses split; rename the config key to
   "replayed outside the tool (fwd_era.py)". The SA0 escalation (unison/phase flag) already carries the
   substance.

## Non-blocking notes (in order of weight for SA0)

2. **p = 2 is not decided by the evidence; present it as the judgment call the plan makes it (SA0 #2).**
   Every cost of p = 1.5 sits inside the plan's own limits: creeper term-alone 0.143 (limit 0.5), R2c 0.150
   (limit 0.2), short-stance share 0.96 % (limit 2 %), and restated R1 at **0.7996** against 0.80, a 4e-4
   miss on X0x s2 turn, where the seed spread of the same statistic is 0.09 (W 0.926 vs X0x s2 0.837). Its
   rejection rests on the step-7 agent's stricter rule ("never pay a degenerate more"), not on a gate. The
   benefit that matters for the plan's target is the side pull: at W's side median stance stroke 0.040 m
   (C = B/L = 0.115) the per-stance marginal d(pay)/dC is 2C = 0.23 at p = 2 against 1.5·C^0.5 = 0.51 at
   p = 1.5; at the respective doses (w_S 0.1034 vs 0.0865) that is 1.85× more delivered pull on side at
   p = 1.5 with the fwd share held at 10 % by construction. The rotational creeper (2.81 → 1.91) and the
   lunge (5.1 → 3.0) also move the right way. The plan's manipulation check is what decides "did S bite";
   p = 2 makes that bite smaller. Recommend showing both p values' side pull and creeper numbers at SA0
   rather than "p = 1.5 rejected".
3. **R1 restated undersells what the anchor removes.** On healthy turn, 41–46 % of scored stances have
   B > Ba (foot pushed backward in the world while loaded), and the anchor removes 18–23 % of credit. That is
   ordinary push-slip of a healthy turning hexapod, not only the paddling exploit. The restatement is still
   legitimate (the anchor is a credit definition added after R1 was written), but SA0 should read it as: the
   term's total tax on healthy turn is 24–36 % (fwd 14–16 %), and turn's dose share is 0.048.
4. **R3b restated (CI lower bound > +0.2) is weak** at n = 200 (half-width 0.14, so a true ρ of +0.3 usually
   passes). The property the report actually relies on is Spearman(pay per scored stance, switch rate) ≥ 0
   (W side +0.136; fwd/back/turn are negative because those heads step less when they stroke long). Suggest
   adding that sign as the restated clause; it is the tap test.
5. **R3a's "< 0.1 s" half is vacuous by construction:** stances with n < 5 credited steps are unpaid
   (`MIN_LOADED_STEPS`), so the short-stance share measures only paid stances with B < 0.02 m. Fine, but say so.
6. **Survival floors:** 0.365 is the driver's `survival_by_hold` (a fall after the hold's last step
   survives); per-category schedule completion gives move_turn 0.355. The floors must be applied with the
   driver's statistic, as w2 did; the report should name it.
7. **Forward-era rollouts are n = 10 episodes each.** The lunge (5.1×) is unambiguous; tip-rock (1.53×) and
   skate (1.00×) are near the 0.5 line only in the sense that they fail it with little data behind them.
   The unison/phase flag recommendation stands on the lunge alone.
8. **No combined S + P no-flip was run** (R5 prices S alone, P3 prices P alone; arm C carries both). Moot on
   the margins: the creeper's centring pay is 0.023/s × 0.161 = 0.004/s, and its move_turn total with S is
   +0.011 against healthy ≥ 0.117. Worth one line in the record.
9. **The creeper becomes a net earner on move_turn with S** (−0.023 → +0.011 in W's economy). The tool's
   rg1 passes it on ordering because the base total is ≤ 0. Correctly reported under SA0 #3; keep the
   move_turn stroke-share WATCH.
10. **Calibration screen** (`step7.calibration_screen`, 28 settings): the report's table transcribes it
    correctly (n_off 3 income bias 4.45 %; prog_lo 0.10 creeper 0.127; prog_lo 0.20 R1r 0.722; sigma 0.25
    W out of band 0.509; phi_sat 0.15 creeper 0.101). None of the screened settings pays the creeper less
    on translation than the defaults except prog_lo 0.20 / prog_width 0.20 / n_off 3 / p 1.5+n_off 3, each
    of which fails R1r. "Nothing moved" is supported.
11. **Restated R8** (exact mirror parity) is a tautology of the equivariance unit test; its real content is
    that the replay tool's conventions (hand, lat_share, slot) are symmetric, which it shows. The CI rule
    is what carries the criterion; X0x s2 move_turn's CI [−0.006, +0.255] is the closest call.

## What I did not re-run

The restated R9 grid (21 neighbours × full replays) and the 28-setting calibration screen were read from
`replay_gate.json`, not recomputed; their base-row entries match the gate I did re-run.
