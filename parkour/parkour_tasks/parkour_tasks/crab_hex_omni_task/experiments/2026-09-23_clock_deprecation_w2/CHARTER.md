# Clock deprecation, wave 2: a hard clock cut in an extra phase-1 stage after c6

Wave 2 of the omni phase-1 clock-deprecation campaign (wave 1: `../2026-09-23_clock_deprecation/`).

- **Plan of record:** `/home/nickmagus/.claude/plans/lets-explore-options-for-groovy-spindle.md` step 14.
- **Amendment** (user, 2026-09-25): `parkour/outputs/omni_plan/clock_deprecation/c7_design/STAGE_PLAN.md`.
- **The evidence behind it:** `c7_design/RECOMMENDATION.md`, with an independent check at its end.

## The question

Does cutting the gait clock on a mature c6 policy make the gait less rigid? Here "cutting" means both clock rewards
at ε 0.001 and the clock observation zeroed at full width. The comparison is 4000 further iterations, with no
command category arriving, against the same 4000 iterations with the clock held. "Less rigid" means RIG BETTER with
its steps part and its support part each ≥ 0. It must not lose facing, turning, speed, slip or survival.

Wave 1 always removed the clock at a C1d boundary where a new category arrived:

| Boundary | Category arriving |
|---|---|
| c4 | move_turn |
| c5 | omni |
| c6 | omni_turn |

Every wave-1 arm was a BREAK:

| Arm | Why it broke |
|---|---|
| A6000, A4500 | side/turn foot_slip guards |
| A3000 | move_turn survival floor at c4 |

Wave 2 separates "clock removed" from "clock removed while a category arrives".

## Shape

| | |
|---|---|
| Chain `c6x` | one stage, preset 1a, C1d's c6 overlay verbatim, **4000 its** |
| Fork | X0 s3's c6 head `2026-09-23_20-09-04_seed3_X01_c6_A15pB/model_7996.pt`, clock held through c6 |
| Phase 1 total | 12000 its |
| Baseline **X0x** | c6x with the clock held, s3 and s2 (both fork the s3 head, as wave 1's X02 forked the s3 t3); its pair is D00bs |
| Candidate **W** | c6x with K = `{KRABBY_CLOCK_W: 0.001, KRABBY_APEX_W: 0.001, KRABBY_OMNI_CLOCK_HIDE: 1}` from iteration 0 (a hard cut; no PHASEOUT, no HIDE_RAMP); `control: baseline`; HIDE carried into its evals |
| Seeds | W s3; W s2 only when `pair_on` fires |
| Floor | X0 s3 c6 survival − 0.20 per category |
| Decision rule | wave 1's, verbatim: blocks F / F_mt / T / RIG, balanced RIG, 23 guards, margin flags, classes, R = max(D00s, D00bs). The only omission is the pilot key, which named A6000. |
| D00s | reuses wave 1's E1/E2 stamps if `eval_sha` has not moved |

## Settled decisions (user, 2026-09-25)

1. **Shut off, not ramp.** This replaces step 14's pre-registered ramp as the first arm.
2. **4000 iterations.**
3. **Pre-launch check E10 first:** X0 s3 c6, clock hidden zero-shot. If hidden stand, side or turn survival is
   < 0.90, or fwd and diag are both < 0.10, W becomes the plan's ramp instead: PHASEOUT 0:24000, then HIDE_RAMP
   24000:48000.
4. **The decision rule stays frozen.** The slip-guard R question is raised again at the readout: R is 0.0053 side /
   0.0039 turn, against one same-seed replicate spread of 0.014 / 0.010.
5. **The amendment is audited once before launch.**

## Expected result and what each outcome means

The likely outcome is REJECT on side/turn foot_slip. This is inferred: clock-free slip ran 0.014–0.030 above its
clock-held twin in all four wave-1 measurements. The readout (S6) is the user's.

The menu is keyed on the driver's `arm_class` and its WORSE list. Any guard WORSE is class BREAK, so a slip-only
REJECT is class BREAK.

| Outcome | What it means | Next step |
|---|---|---|
| WIN | Wave 1's slip breaks came from the collision with arriving categories | the seed-2 confirm (plan step 15) |
| BREAK, only the side/turn `foot_slip` guards WORSE | slip is the price of the clock's removal itself | the user trades turning/speed against slip; W s2 (+1.69 h) is offered for the slip-guard R question |
| BREAK by a floor or a survival guard | the cut's shock did not heal on a c6 head | a ramp arm next; the user picks W at 4000 its with a 1000-it mask (reusing X0x), or the 2000-it-mask variant at 5000 its (X0x rerun at 5000) |
| Any other BREAK (F, F_mt, T or another guard), or RIGID | not a scheduling problem | plan step 14: stop and report the out-of-scope levers |
| NEUTRAL | a simplicity call | the user decides |

## Rules carried from wave 1

- **Speculative lines.** Confirming or overruling a speculative seed-3 line cancels its seed-2 pair. WIN then
  becomes unreachable.
- **Isaac.** One Isaac process at a time, under MemoryMax. Stop the driver only during a stage eval.
- **Records.** Never pushed.
- **MQ.** The MQ session stays paused.
- **Diagnostics, never gated:**
  - +2000 suite evals on `model_10000.pt` of X0x s3, X0x s2 and W s3;
  - a CPU check that W's crab_failure and episode-length excess over X0x is back inside the X0x s3/s2 band by
    it 1500.

## Log

- 2026-09-25: directory created. The design workflow (9 agents plus a fable check) recommended a cut at 4000 its. The
  user chose a cut, 4000 its, E10 first, the rule frozen, and a 1-pass audit.
- 2026-09-25: E10 run on X0 s3 c6 with the clock hidden, zero-shot (`step0_E10_X0s3-c6_hidden/…/2026-09-25_15-07-17`).
  - `clock_obs ≡ 0` on all 400 npz.
  - Hidden survival: stand 1.000, fwd 0.680, side 0.995, back 0.910, diag 0.590, turn 1.000, move_turn 0.770.
  - The switch did not fire; the cut goes ahead.
  - The c6 head is far less clock-dependent than X0's c5 head was (E9: fwd 0.21, diag 0.24).
- 2026-09-25: the 1-pass audit found 2 blocking items.
  - B1, fixed: the outcome menu is now keyed on class plus the WORSE list, because a slip-only REJECT is class BREAK.
  - B2: moot, because E10 did not fire. The ramp-variant choice after a shock BREAK is now asked at the readout.
  - The dry run is clean. eval_sha 36e5a8af7000 and code_sha 8ce6da6156b3 are both unchanged from wave 1, so D00s
    reuses the wave-1 stamps.
