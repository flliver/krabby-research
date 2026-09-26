# SA0 — replay gate decisions (2026-09-26)

**Who decided.** The orchestrator, under the user's standing instruction of 2026-09-25: "proceed with the whole plan
without asking". Each decision takes the plan's or the evidence's recommended default. The user was push-notified.

**Evidence.**
- `REPLAY_GATE.md` / `replay_gate.json`: step 7 at p = 2 (the pre-SA0 defaults).
- `gate_p15/`: the same gate at the adopted p = 1.5.
- `REPLAY_GATE_CHECK.md`: the fable check, which reproduced every number.
- `POWER_PREVIEW.md`: step 8.

## Adopted parameters (the math's defaults, `crab_hex_omni_stroke_math.py`)

| Parameter | Value |
|---|---|
| Stroke | RHO0 0.5, DRHO 0.3, **POWER 1.5**, C_MAX 1.5 |
| Progress guard | PROG_TAU_S 4.0, PROG_LO 0.15, PROG_WIDTH 0.15 |
| Contact | FORCE_OPEN_N 50, FORCE_HOLD_N 20, N_OFF 2, MIN_LOADED_STEPS 5 |
| Centring | SIGMA 0.20, TAU 2.0, MIN_CMD 0.12, PHI_LO 0.10, PHI_SAT 0.25 |
| **Stroke dose** | `KRABBY_OMNI_STROKE_W` = **0.0865**. R4 binds on W fwd at exactly 10 % of tracking income, delivered (post-clip). |
| **Centring dose** | `KRABBY_OMNI_CENTRING_W` = **0.155**. P2 prize parity with S; band [0.124, 0.194]. |

Both doses are subject to the window-0 dose smoke, as the plan's rule states:
- r_S is taken on the fwd row, r_P on the side row;
- rescale by 1/r only when r is in [0.5, 2];
- otherwise stop.

At these doses:
- the sideways-head side share is 0.146 flat and 0.124 on steps (reported; not capped, per decision 6);
- diag's centring share is at most 0.084.

## Decisions

1. **p = 1.5, not 2.** The step-7 agent recommended 2; the fable check found the evidence does not decide it. The
   screen's main risk is a stroke dose that does not bite on side (pass-1 audit, probe B). p = 1.5 gives about
   1.85× the delivered marginal pull on side at the same 10 % fwd cap. It also pays the older-plant lunge 3.0×
   instead of 5.1×, and the creeper's move_turn 1.91× instead of 2.81×.

   Costs, all inside the plan's limits:
   - short-stance share 0.96 % (limit 2 %);
   - creeper translation 0.143× (limit 0.5×);
   - R2c 0.150 (limit 0.2).

   Restated R1 reads 0.7996 against 0.80, a 4e-4 miss on a statistic with 0.09 seed spread; accepted as noise.
2. **Keep the rotational credit (ω×r)** on turn and move_turn. Turn earns only through it. The creeper's extra
   move_turn income is about 1 % of move_turn tracking, and no-flip holds. Watch the move_turn stroke share at SA4.
3. **Eval keys: offline readout** (`replay_stroke_posture.py readout` at each park); `eval_sha` stays unchanged.
4. **Stand excluded, command-only normaliser:** yes, both.
5. **Floors:** W's suite survival (the driver's `survival_by_hold`) − 0.20:

   | stand | fwd | side | back | diag | turn | move_turn |
   |---|---|---|---|---|---|---|
   | 0.80 | 0.75 | 0.795 | 0.78 | 0.67 | 0.79 | 0.165 |

   **move_turn and diag park and flag instead of rejecting** (user decision, audit pass 3).
6. **Restated criteria**, adopted with step 7's evidence. Each literal form measures something other than what it
   protects:
   - **R1:** planted-credit retention, because the anchor cap removes backward slip by design.
   - **R2:** clauses b + c. Clause a's 0.40 was calibrated on the old term's credit.
   - **R3:** short-stance share ≤ 2 %, plus a tap test: Spearman(pay per stance, switch rate) ≥ 0 on side; W side
     +0.136. This replaces the income-vs-switch Spearman ≤ +0.2, whose confidence interval is as wide as its
     threshold.
   - **R6:** the income clause only; the liftoff-count clause conflicts with it.
   - **R8:** a hand gap whose 95 % CI includes 0. The term is exactly mirror-symmetric on real rollouts.
   - **P1:** judged on the flat sideways scenario. On the step scenario the statue guard fires while the head
     stalls, and the arms train flat.
7. **R5 as written: FAIL** (labelling fix from the fable check). The in-plant clauses pass (0 flips in 79 rows;
   creeper term-alone 0.143× on translation). The cross-plant term-alone clause fails on the forward-era lunge:
   3.0× at p = 1.5, 5.1× at p = 2. A per-foot stroke income cannot see unison or phase. **Mitigation, adopted:** a
   unison/phase flag on arms S and C at every readout park — tripod coherence falls beyond its seed band, or
   roll_rms collapses, relative to B.
8. **Turn progress guard:** accepted as is. Every omni head turns at 0.24–0.43 of the commanded yaw rate, which is
   inside the guard's 0.15–0.30 band, so turn credit carries a tracking tax. That tax rewards tracking the
   commanded yaw; a separate band per slot would be a math change. Watch turn stroke income at SA4.
9. **Reward-simplification candidate #2:** the stride half is superseded by the per-foot stroke replacement
   (`KRABBY_OMNI_STROKE_W`, a func swap under `reward_stride_length`). The air-time half is untouched: its own task,
   #28. `loaded_stance_edges()` is offered to it.
   - v4 (per-foot touchdown stride, 2026-08) was also dead in training (the same float32 bug).
   - The note lives here rather than in `CANDIDATES*.md`, because those files carry another session's uncommitted
     edits in the main checkout.

## Step 8 (no STOP)

- **Premise:** side body_offset +0.231 m (≥ 0.15); side/fwd stride 0.35 (≤ 0.6).
- **Provisional LEN block:** R = 1.194, seed-dominated. Per-member MDE: side 1.644, diag 1.938 steps/m. S's
  minimum effect (+0.02 m side stride) ≈ −6.2 steps/m, above the MDE.
- **Readout-key effective MDEs** (max of eval, seed and seed Δ):

  | Key | MDE |
  |---|---|
  | side body_offset | 0.017 m |
  | side stride_along | 4 % |
  | side stance_stroke | 0.003 m |
  | side stroke_share | 0.033 |
  | fwd stance_stroke | 0.027 m |

  A single-seed change must clear the seed term.
