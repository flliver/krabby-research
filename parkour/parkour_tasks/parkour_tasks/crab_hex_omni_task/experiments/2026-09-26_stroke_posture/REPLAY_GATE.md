# REPLAY_GATE: stroke/posture plan, step 7 (replay gate and calibration, 0 GPU)

Date 2026-09-26, worktree `feat/omni-stroke-posture` at `d716483b`. CPU only; no Kit, no GPU, nothing
committed. Machine-readable record: `replay_gate.json` (the tool's own gate output at the final
parameters, plus a `step7` block with the calibration, the restated criteria and the cross-plant
degenerates). Gate config of record: `replay_gate_config.json`. The helper scripts behind the
calibration and the restated criteria are in `replay_gate_calib/`.

## Bottom line

- **Parameters: keep the math's current defaults.** None of the 30 alternative settings screened
  passes more criteria without making a failing one worse, or without paying a degenerate more.
- **Doses: w_S = 0.103, w_P = 0.161.**
  - R4 binds on W fwd (10.0 % of fwd tracking, delivered).
  - The sideways head's side share at w_S is 14.7 % flat and 12.3 % on steps. This is reported, not
    capped (decision 6).
  - P2 sets w_P at prize parity with S. W's diag share at w_P is 7.7 % (≤ 10 %).
- **Literal verdicts at the final parameters:**
  - PASS: R4, R5 (in-plant), R7, P2, P3, P4.
  - FAIL: R1, R2, R3, R6, R8, R9, P1.
- **Restated verdicts** (below): R1, R2, R3, R6, R8 and P1 pass. Each failure is a criterion that
  measures something other than what it protects, shown with evidence.
- **Two items are not fixable by parameters and go to SA0:**
  1. **Turn knife-edge (a real property, class c).** The progress guard has one band [0.15, 0.30]
     for every slot. Every omni head turns at 0.24–0.43 of the commanded yaw rate, per step and
     signed, which is inside that band. So turn credit is knife-edge in `prog_lo` / `prog_width`:
     restated R9 fails at 3 of 21 neighbours on X0x s2 turn.
  2. **Cross-plant degenerates (design limit, class c).** The 2026-08 forward-era degenerates, which
     were located and replayed through the same math, earn more than their own healthy reference
     from the term alone:

     | Degenerate | × healthy |
     |---|---|
     | lunge | 5.1 |
     | tip-rock | 1.5 |
     | skate | 1.0 |

     A per-foot stroke income cannot see phase or unison. No parameter in the neighbourhood changes
     this.

## 1. Final parameter set and doses (adopt as the defaults)

| Term | Parameter | Value | Note |
|---|---|---|---|
| stroke | `rho0` (`RHO0`) | 0.5 | unchanged |
| stroke | `drho` (`DRHO`) | 0.3 | unchanged |
| stroke | `power` (`POWER`) | **2.0** | p = 1.5 rejected (SA0 item 2) |
| stroke | `c_max` (`C_MAX`) | 1.5 | unchanged |
| stroke | `prog_tau` (`PROG_TAU_S`) | 4.0 s | unchanged |
| stroke | `prog_lo` (`PROG_LO`) | 0.15 | unchanged (lowering it pays the creeper more; see §5) |
| stroke | `prog_width` (`PROG_WIDTH`) | 0.15 | unchanged |
| stroke | `force_open` (`FORCE_OPEN_N`) | 50 N | unchanged |
| stroke | `force_hold` (`FORCE_HOLD_N`) | 20 N | unchanged |
| stroke | `n_off` (`N_OFF`) | 2 | unchanged (n_off 3 rejected, §3) |
| stroke | `min_loaded` (`MIN_LOADED_STEPS`) | 5 | unchanged |
| centring | `sigma` (`CENTRING_SIGMA_M`) | 0.20 m | unchanged; near the steepest slope at W's offset (σ\* = o = 0.23) |
| centring | `tau` (`CENTRING_TAU_S`) | 2.0 s | unchanged |
| centring | `min_cmd` (`CENTRING_MIN_CMD_M_S`) | 0.12 m/s | unchanged |
| centring | `phi_lo` / `phi_sat` | 0.10 / 0.25 | unchanged (lowering them pays the creeper 0.101 × sideways; P3 FAIL) |
| dose | **w_S** (`KRABBY_OMNI_STROKE_W`) | **0.103** | exact 0.10336; binding row W fwd |
| dose | **w_P** (`KRABBY_OMNI_CENTRING_W`) | **0.161** | exact 0.16121; band [0.129, 0.202]; with the flat target only it would be 0.164 |

**R4 dose rows** (delivered, post-clip ÷ kernel tracking, over gated moving own-hold steps, at w_S):

| Head | fwd | back | turn | move_turn | diag (rep.) | side (rep.) |
|---|---|---|---|---|---|---|
| **W (dose run)** | **0.100** | 0.071 | 0.048 | 0.012 | 0.045 | 0.009 |
| X0x s3 | 0.087 | 0.070 | 0.040 | 0.009 | 0.046 | 0.011 |
| X0x s2 | 0.099 | 0.074 | 0.028 | 0.010 | 0.050 | 0.011 |
| E1 (X08:c6) | 0.089 | 0.072 | 0.037 | 0.007 | 0.041 | 0.010 |

- Stand pays exactly 0 on every head. Turn and move_turn both pay more than 0.
- The sideways head's side share is 0.147 on the flat stamp and 0.123 on steps. This is nominal (it
  is not priced in W's economy) and not capped.

**P2 dose.** The prize per unit weight is the sideways head's side minus W's side:

| Term | Prize per unit weight | Prize at its dose |
|---|---|---|
| stroke | 0.739 | 0.076/s |
| centring | 0.474 | 0.076/s |

The resulting shares at w_P:

| Row | Share at w_P |
|---|---|
| side | 0.076–0.088 |
| diag | 0.077–0.085 (≤ 0.10 on every head) |
| move_turn | 0 (λ = 0 on non-MOVE slots) |

## 2. Per-criterion table at the final parameters

The literal column is the tool's verdict. The restated column is judged by the rule in §4.

| Id | Literal | Restated | Deciding numbers |
|---|---|---|---|
| R1 | **FAIL** | **PASS** | Kept share of credit: literal (all credit) vs restated (planted, anchor-capped credit) — see the table below. |
| R2 | **FAIL** (a) | **PASS** (b+c) | (a) omni side kept 0.403–0.465 against ≤ 0.40. (b) slide stances (ρ > 0.8) kept **0.039** (≤ 0.25). (c) omni side ÷ sideways head **0.067–0.112** (≤ 0.2). |
| R3 | **FAIL** (b) | **PASS** | (a) short stances carry 0.23–0.32 % of income (≤ 2 %). (b) Spearman(income/s, switch rate): W side +0.226, CI [+0.08, +0.36]; every other row ≤ +0.146. (c) v4 is cross-plant: 1.35× H2000 per s, but its short-stance share is 0.01 %. |
| R4 | **PASS** | – | w_S 0.1034; binding W fwd 0.100; omni fwd ≥ 0.087 on every head; stand 0; turn and move_turn > 0 |
| R5 | **PASS** (in-plant) | – | In-plant rows: see the R5 list below. Cross-plant (not in the tool): lunge 5.12×, tip-rock 1.53×, skate 1.00× → **FAIL** (§5). |
| R6 | **FAIL** (liftoff) | **PASS** | Income shift −0.15 … −2.7 % (≤ 5 %; every omni row negative). Liftoff shift up to 4.1 % (side and diag; ≤ 2 % was required). Sideways f32 replica −1.5 / −1.6 %. |
| R7 | **PASS** | – | Max \|clip-bind shift\| 0.43 pp (E1 diag). Max absorbed 11.5 % (E1 move_turn; ≤ 15 %). |
| R8 | **FAIL** | **PASS** | Hand gaps E1 move_turn 0.161, X0x s2 move_turn 0.127; side and turn ≤ 0.099. The term's mirror parity on real rollouts is **exact** (W, E1 and X0x s2: stroke Δ = 0.0 per episode). Every gap's 95 % CI includes 0. |
| R9 | **FAIL** (all 21; base fails R1–R3) | **FAIL (4/21)** | Restated grid: fails at `prog_lo` 0.20 (X0x s2 turn 0.722), `prog_width` 0.20 (0.777), power 1.5 (0.800) and `n_off` 1 (W side CI low +0.221). Passes the other 17. |
| P1 | **FAIL** | **PASS** | Sideways flat **0.942** (≥ 0.85); sideways step 0.752. W side 0.373, in [0.15, 0.45]. fwd, back, stand and turn = 0 (max 4.2e-8). |
| P2 | **PASS** | – | w_P 0.1612; diag ≤ 0.085. Healthy side ≤ 0.088 (the design panel's 15 % line is not reached). |
| P3 | **PASS** | – | No-flip holds on all 79 judged rows (0 fail). X08 side total 0.878 against the statue's −35.1. Creeper ÷ sideways 0.027 (≤ 0.10). |
| P4 | **PASS** | – | Max \|clip shift\| 1.79 pp (E1 diag). Hands ≤ 0.025. Centring absorbed 8.5 % on W side, at most 15.5 % (X0x s3 diag; reported). |

**R1 kept shares:**

| Row | Literal kept (all credit) | Restated kept (planted credit) |
|---|---|---|
| turn | 0.644–0.761 | 0.837–0.926 |
| fwd | 0.844–0.871 | 0.939–0.954 |
| back | 0.825–0.845 | 0.950–0.957 |
| sideways side, flat / step | 0.872 / 0.788 | 0.936 / 0.912 |

**R5 in-plant rows:**
- No-flip holds on all 79 judged rows (0 fail). The tightest is Z c6 vs creeper on diag: 1.011 →
  1.047, so the term widens the gap.
- There are 5 reversals already present before the term (Z vs creeper on fwd, back and diag). All
  narrow with it, by 5–19 %.
- Term alone on translation rows: at most 0.113× (creeper ÷ X0x s3 on fwd).
- Rotational rows (reported): turn 0.29–0.51×, move_turn 1.78–2.81×.

**Step-6 validation** (re-run at `d716483b`):
- E1: side `body_offset` +0.233; lead / trail |y| 0.727 / 1.230; side / fwd `stride_along` 0.075 /
  0.196; `stroke_share` fwd 0.829 > 1.5 × side 0.518.
- Sideways head: lead / trail |y| 0.933 / 0.960.
- All hard targets reproduce.

## 3. What was calibrated, and why nothing moved

Every stroke parameter in the R9 grid was screened on its own, as a full gate (all 11 runs, every
criterion), plus four combinations and three centring settings. The full table is
`replay_gate.json → step7.calibration_screen`.

| Setting | Literal fails | Restated R1 min | R2a max | R2c max | R3b W side | Creeper term-alone (transl. / rot.) | R6 income / liftoff | Verdict |
|---|---|---|---|---|---|---|---|---|
| **defaults** | R1 R2 R3 R6 R8 P1 | **0.837** | 0.465 | 0.112 | 0.226 | **0.113 / 2.81** | **2.7 % / 4.1 %** | **adopt** |
| power 1.5 | R1 R2 R3 R6 R8 P1 | 0.800 | 0.434 | 0.150 | 0.293 | 0.143 / 1.91 | 2.2 / 4.1 | ✗ pays the creeper more; R1r at the line |
| n_off 3 | R1 R2 R8 P1 | 0.825 | 0.401 | 0.100 | 0.148 | 0.104 / 2.72 | **4.4** / 1.9 | ✗ side income shift 5× worse; R8 side gap 0.132 |
| p 1.5 + n_off 3 | R1 R3 R8 P1 | 0.787 | 0.377 | 0.131 | 0.201 | 0.134 / 1.84 | 4.5 / 1.9 | ✗ fails R1r |
| rho0 0.4 + drho 0.2 | R1 R6 R8 P1 | 0.806 | **0.386** | 0.100 | 0.188 | 0.113 / 2.77 | 2.9 / 4.1 | ✗ R2a only by taxing healthy turn/fwd (R1 min 0.620) |
| rho0 0.3 | R1 R6 R8 P1 | 0.797 | 0.375 | 0.099 | 0.187 | 0.114 / 2.77 | 2.9 / 4.1 | ✗ fails R1r |
| rho0 0.4 / 0.6 | as defaults | 0.821 / 0.846 | 0.423 / 0.503 | ~0.11 | ~0.22 | 0.113 / 2.8 | ~2.7 / 4.1 | no gain |
| drho 0.2 / 0.4 | as defaults | 0.827 / 0.843 | 0.430 / 0.497 | ~0.11 | ~0.22 | 0.113 / 2.8 | ~2.7 / 4.1 | no gain |
| c_max 1.0 / 1.25 / 1.75 / 2.0 | as defaults | 0.836–0.837 | 0.465 | 0.112–0.115 | 0.226 | 0.116 / 2.23 … 0.113 / 3.02 | 2.7–3.1 / 4.1 | c_max 1.0 cuts the rotational creeper ratio but raises the translation one |
| prog_lo 0.10 | as defaults | **0.906** | 0.465 | 0.112 | 0.226 | **0.127** / 2.85 | 2.8 / 4.1 | ✗ pays the creeper +12 % |
| prog_lo 0.20 | as defaults | 0.722 | 0.463 | 0.113 | 0.225 | 0.093 / 2.80 | 2.5 / 4.1 | ✗ fails R1r |
| prog_width 0.10 / 0.20 | as defaults | 0.883 / 0.777 | 0.465 | 0.112 | 0.226 | 0.120 / 0.104 | ~2.6 / 4.1 | ✗ (pays the creeper / fails R1r) |
| prog_tau 3 / 5 | as defaults | 0.837 / 0.839 | 0.465 | 0.112 | 0.226 | 0.110 / 0.115 | 2.8 / 4.1 | no gain |
| force_hold 15 / 25; force_open 40 / 60; min_loaded 4 / 6 | as defaults | 0.836–0.837 | 0.459–0.472 | 0.110–0.114 | 0.224–0.234 | 0.112–0.113 | 2.5–2.7 / 4.1–4.3 | no gain |
| sigma 0.15 / 0.25 | as defaults (+P2 at 0.25) | – | – | – | – | – | – | 0.25 puts W out of the P1 band (0.509); 0.15 keeps W in band (0.220) but gives a weaker gradient at W's offset (1.9 vs 3.1 per m) |
| phi_sat 0.20; phi_lo 0.05 / phi_sat 0.15 | as defaults (+P3 at 0.05 / 0.15) | – | – | – | – | P3 creeper 0.035 / **0.101** | – | ✗ pays the creeper more |

**Rule applied.** "Passes the most criteria without making a failing one worse, and never by paying
a degenerate more." The settings that turn literal criteria to PASS each pay for it:

- **n_off 3** passes R3 and R6. It raises the side stale-data income bias from −0.8 % to −4.5 %,
  close to the 5 % bound, and makes R1r and R8 worse.
- **rho0 0.4 + drho 0.2** passes R2 and R3 by gating healthy stances harder:
  - R1 literal min 0.620;
  - R1r min 0.806;
  - its R3 "pass" (0.188) is within the ±0.14 noise band.
- **prog_lo 0.10 and prog_width 0.10** are the only settings that make turn robust, and both pay
  the in-plant creeper more on translation (0.127 and 0.120 against 0.113).

So nothing moved. The forward-era degenerates barely move anywhere in the neighbourhood: lunge
4.8–5.2× and tip-rock 1.53–1.55× at p = 2. Only p changes them: 3.0× / 1.35× at p = 1.5 and 1.9× /
1.20× at p = 1.

## 4. What was restated, and why (class b: the criterion mis-specified for what it protects)

1. **R1 → kept ≥ 0.80 of the *planted* (anchor-capped) credit.**
   - **What the anchor cap removes.** It removes credit only when a loaded foot's world position
     moves backward along its commanded direction. By construction, B − Ba equals that backward
     slide, so it is slip.
   - **How much slip there is.** On turn the measured backward slide is 12–19 % of the own sweep,
     against 5–7 % on fwd. Literal turn kept falls in the same order:

     | Head | Backward slide / own sweep | Literal turn kept |
     |---|---|---|
     | W | 0.123 | 0.761 |
     | E1 | 0.150 | 0.716 |
     | X0x s3 | 0.156 | 0.707 |
     | X0x s2 | 0.185 | 0.644 |

   - **Why no parameter passes literal R1.** On X0x s2 turn the anchor alone removes 0.231 of the
     credit. The anchor has no parameter, and it closes the backward-paddling exploit.
   - **The restated form.** The slip gate plus the progress guard must keep ≥ 0.80 of what remains.
     They do, with a minimum of 0.837. The anchor-removed share is reported per row.
2. **R2a (omni side kept ≤ 0.40) is dropped as a gate and reported.**
   - The 0.40 came from the investigation's "40 % of the OLD term's credit (body progress per planted
     leg) is real stroke".
   - The new term's ungated credit is already the foot's own sweep, which excludes the forward scuff
     that made up most of the old non-stroke part. Under the new stance definition, omni side's own
     sweep is 52–56 % of body progress (`stroke_share`).
   - The property R2 protects, that omni side's scuffing gait is not paid, is measured directly by
     R2b (slide stances kept 0.039) and R2c (omni side 0.067–0.112 × the sideways head). Both pass
     with at least 2× margin.
   - Reaching 0.40 needs rho0 ≤ 0.4 and drho 0.2, which taxes healthy fwd and turn (above).
3. **R3b judged against its sampling noise: FAIL only if the 95 % CI lower bound exceeds +0.2.**
   - At n = 100–200 the bootstrap CI half-width is ±0.13–0.20. That is as wide as the threshold.
   - The same lineage moves with no change to the term: E1 vs E2 side reads −0.099 vs +0.017, and
     the X0x seeds read +0.074 / +0.096.
   - **Mechanism.** On W side, pay per scored stance *rises* with switch rate (+0.136). A tap trade
     would lower it: taps earn nothing (short stances carry 0.23 % of income, and splitting a stance
     halves its pay at p = 2). The correlation reflects episodes that step more actively.
   - The restated rule still catches a real tap regime. At `n_off` 1, where every load dip closes a
     stance, the CI lower bound is +0.221, a FAIL. That is the evidence for keeping N_OFF ≥ 2.
4. **R3c (v4 ≤ healthy) → v4's short-stance income share ≤ 2 %, which is 0.01 %.**
   - Forward-era v4 earns 1.35× H2000 because its stances are longer: median sweep 0.100 m against
     0.084 m, and per-stance pay 0.052 against 0.034. Its extra tapping earns nothing.
   - This is also cross-plant (§5).
5. **R6 → the income clause only (≤ 5 %). The liftoff count is reported.**
   - Under 1-in-5 stale readings the extra closes are split stances. Scored stances rise 1.6–3.1 %.
   - Income falls in every omni row (−0.15 … −2.7 %). Under a convex per-stance pay, extra edges
     cannot raise pay.
   - The healthy sideways gait's liftoff shift is only 0.1–0.7 %. Omni side and diag reach +3–4 %
     from their one-step load dips (scuffing leading feet).
   - The setting that passes the liftoff clause (`n_off` 3, 1.9 %) makes the income clause 5× worse
     on side.
   - For the air-time task (#28): if `loaded_stance_edges()` is reused for a per-edge pay, its edge
     count moves +3–4 % under stale data on omni side and diag.
6. **R8 → the term's own hand symmetry, checked on real rollouts.**
   - **The test.** Every episode was mirrored through M_y (y → −y, FL↔FR, ML↔MR, RL↔RR, heading and
     w_z negated) and replayed.
   - **The result.** Stroke income is identical per episode on W, E1 and X0x s2 (max Δ 0.0).
     Centring differs by at most 6e-5 on totals of about 6e4.
   - **What the l/r gaps are.** They are the heads' gait asymmetry plus eval noise:
     - bootstrap 95 % half-width ±0.12–0.23;
     - E1 vs E2 on move_turn: +0.161 vs −0.160, a sign flip.
   - The design panel's R8 named only side and turn, where the maximum gap is 0.099.
   - **The restated form.** Mirror parity is exact (PASS), and a gap above 10 % fails only if its CI
     excludes 0 (none do).
   - X0x s2 move_turn is the closest case. Its CI is [−0.006, +0.255], and it crosses 0 under
     setting changes that are pure noise for this key.
7. **P1 → the flat sideways stamp is the target (0.942). `step__side` is reported (0.752).**
   - The step scenario's loss comes from the statue guard while the head stalls at the steps (g_phi
     mean 0.78) and from the offset (0.93).
   - The omni arms train on flat ring terrain.
   - Loosening the guard to pass it pays the creeper 0.101 × sideways (P3 FAIL).

## 5. What could not be fixed (for SA0)

- **(c) The turn progress-guard band sits on the omni heads' turn operating point.**
  - The guard applies one band [`prog_lo`, `prog_lo` + `prog_width`] = [0.15, 0.30] to every slot.
  - On turn holds the signed per-step ratio of body yaw rate to commanded w_z is:
    - X0x s2: median 0.24–0.28 by time in hold;
    - W: 0.28–0.43.
  - The eval's |w_z|-based `yaw_rate_ratio` reads 0.52–0.56. That is why the design check, which
    used side ratios of 0.39–0.52, missed this.
  - **Effect.** The guard removes 2 % (W) to 9.5 % (X0x s2) of turn credit at the defaults. At the
    +0.05 neighbours, X0x s2 turn's planted-credit kept drops from 0.837 to 0.722 (`prog_lo`) and
    0.777 (`prog_width`), and restated R9 fails there.
  - **Why no parameter fixes it.** Any lower band makes turn robust (`prog_lo` 0.10: X0x s2 turn
    0.906), and every lower band also pays the in-plant creeper more on translation, which the
    guard exists to catch (0.113 → 0.127).
  - **What would fix it.** Only a math change can: a separate, lower band on turning slots, for
    example half-width. It would need its own creeper check on turn, where the creeper's
    term-alone ratio is 0.29–0.51× and its guard loss is 25 % against healthy's 2–10 %.
  - Accepting it as-is is also defensible: the tax is an incentive to track the commanded yaw.
- **(c) Cross-plant long-stroke degenerates.**
  - The 2026-08 tripod-campaign rollouts were located (tripod_stability campaign directory). The
    tool refuses them: they are forward-task stamps on the pre-2026-08-20 plant, with no suite and
    no `joint_pos`.
  - They were replayed through the same math by `replay_gate_calib/fwd_era.py`, term alone, against
    the same era's healthy references H2000 / H3000 / H5000.
  - Term alone, per second, against H2000:

    | Rollout | × H2000 |
    |---|---|
    | v1 lunge | 5.12 |
    | v2 tip-rock | 1.53 |
    | v3 skate | 1.00 |
    | v3b drag | 0.07 |
    | v4 @3000 | 1.35 |
    | v4 @5000 | 1.31 |

  - The lunge's stances are genuine long planted strokes: median 0.176 m against 0.084 m, ρ 0.08,
    and 6.6× the pay per stance.
  - A per-foot stroke income cannot see unison or phase, and W's economy cannot price these
    rollouts. The in-plant degenerates pass (statue 0, creeper ≤ 0.113 on translation).
  - **Why this is a risk.** At w_S, stroke is 10 % of fwd tracking income, so a stable
    longer-stroking gait can gain several percent of tracking income from the term. **LEN and RIG
    would read fewer, longer steps as BETTER**, and tripod is only a WATCH.
  - **Recommendation.** Add a unison/phase FLAG to arms S and C: tripod `anti_phase` / coherence
    down beyond its MDE, or `roll_rms` collapse. The forward-era lunge signature was tripod 0 with
    roll_rms 0.011 against 0.038.
- **R9 at the defaults (4 of 21 neighbours).**
  - 3 of the 4 are the turn band above: `prog_lo` 0.20, `prog_width` 0.20, and p = 1.5 at exactly
    0.800.
  - The 4th, `n_off` 1, is the dip-close regime that N_OFF = 2 exists to exclude.

## 6. SA0 recommendations (adopted as defaults unless clearly unsafe)

| # | Decision | Recommendation | Deciding numbers |
|---|---|---|---|
| 1 | Parameters and doses | **The defaults in §1; w_S 0.103, w_P 0.161.** Dose-smoke rule as planned: rescale by 1/r for r in [0.5, 2], otherwise stop; r_S from the fwd row (side's r reported, also in [0.5, 2]); r_P from the side row (diag's r reported, and w_P/r_P must keep diag ≤ 10 %). | W fwd 0.100 binding. At w_P: side 0.080, diag 0.077 (headroom to 10 % allows r_P ≥ 0.77). Absorbed at C doses: stroke fwd 0.8 %, side 5.4 %; centring side 8.4 %, diag 10.7 %. |
| 2 | Convexity | **p = 2.** | p = 1.5 fails restated R1 (0.800) and pays the in-plant creeper more on translation (0.113 → 0.143). It triples the short-stance share (0.32 → 0.96 %), moves omni side toward the sideways cap (0.112 → 0.150) and raises W side Spearman (0.226 → 0.293). Its gains (forward-era lunge 5.1 → 3.0×, rotational creeper 2.81 → 1.91×, about 1.6× gradient at side's current stroke) are cross-plant or on reported rows. At p = 2, side's delivered share at w_S is 0.9 % (weak pull); S's manipulation check decides whether it bit. |
| 3 | Rotational credit | **Keep w × r on turn and move_turn.** | Turn pays only through w × r (R4 needs turn > 0). Creeper term alone: turn 0.29–0.51×, move_turn 1.78–2.81×. In W's economy, however, the creeper's move_turn total is −0.023 → +0.011 against healthy ≥ 0.117 (W 0.312 → 0.331), and its extra is about 0.015/s ≈ 1 % of move_turn tracking (1.41/s). W does not turn on move_turn; w × r is the only stroke credit aligned with fixing that. WATCH the move_turn stroke share and facing_closed at SA4. If less rotational exposure is wanted, c_max 1.0 cuts the creeper's move_turn ratio to 2.23 (translation 0.113 → 0.116). |
| 4 | Eval keys | **Default: offline readout at each DECISION; `eval_sha` unchanged.** | Readout is 20–40 s per stamp at 0 GPU and holds every SA4 key, flag and manipulation check. The alternative costs +0.55 GPU-h and forces MQ to re-evaluate. |
| 5 | Stand exclusion; clock-law normaliser | **Yes; yes.** | Stand pays exactly 0 on every head (W: 3610 stand closings, all rejected). The normaliser must be command-only: with W's unlocked cadence (ratio 1.04–1.43), a cadence-based L would make C ≈ constant in stroke length and remove the stride incentive. Command-only L = 0.35 m pays absolute stroke (W fwd C ≈ 0.49, side 0.11, sideways head ≈ 0.48). |
| 6 | Floors re-based on W | **W suite survival − 0.20; move_turn and diag park and flag** (plan default). | W: stand 1.000, fwd 0.950, side 0.995, back 0.980, diag 0.870, turn 0.990, move_turn 0.365. Floors: 0.80 / 0.75 / 0.795 / 0.78 / **0.67** / 0.79 / **0.165**. X0x's diag seed spread (0.74 / 0.58) is comparable to the 0.20 margin, and W's move_turn over its clock-free stage ran 0.62 / 0.255 / 0.28 / 0.365. |
| + | Restatements in §4 | **Adopt R1r, R2(b+c), R3b-CI, R3c-tap, R6-income, R8-mirror, P1-flat.** | Evidence as in §4. |
| + | Turn guard band (§5) | **Decide: accept, or ask for a turning-slot band (math change).** | X0x s2 turn planted-kept 0.837 at the defaults; 0.722 / 0.777 at the +0.05 neighbours. |
| + | Unison/phase flag on S and C (§5) | **Add** (tripod coherence or anti_phase drop beyond its MDE, or roll_rms collapse). | Forward-era lunge 5.1×; LEN and RIG would score a lunge BETTER. |

"Left for implementation" items this step closes:
- **P1 fwd/back leak:** gone. λ now comes from the hold's slot and bearing (max zero row 4.2e-8).
- **P has no healthy cap:** the shares are above (side ≤ 0.088, diag ≤ 0.085, move_turn 0).
- **R4 omits diag:** deliberate (audit pass 2). Diag at w_S is 0.041–0.050 on every head, so a 10 %
  diag cap would not bind.

## 7. W's economy at the doses (per second over gated moving own-hold steps)

Both terms on (arm C): w_S 0.103, w_P 0.161. With S alone, side stroke absorbed is 6.1 %; with P
alone, centring side absorbed is 8.5 %.

| Row | Tracking | Stroke @ w = 1 | Stroke delivered | Stroke share | Stroke absorbed | Centring @ w = 1 | Centring delivered | Centring share | Centring absorbed | Clip-bind, base → with |
|---|---|---|---|---|---|---|---|---|---|---|
| stand | 0 | 0 | 0 | – | – | 0 | 0 | – | – | 0.7 → 0.7 % |
| fwd | 0.868 | 0.846 | 0.0865 | **0.100** | 0.8 % | 0 | 0 | 0 | – | 4.6 → 4.5 % |
| side | 0.684 | 0.065 | 0.0064 | 0.009 | 5.4 % | 0.373 | 0.0550 | **0.080** | 8.4 % | 10.3 → 9.2 % |
| back | 0.908 | 0.628 | 0.0643 | 0.071 | 0.6 % | 0 | 0 | 0 | – | 3.2 → 3.1 % |
| diag | 0.693 | 0.315 | 0.0311 | 0.045 | 4.3 % | 0.370 | 0.0532 | **0.077** | 10.7 % | 14.5 → 12.9 % |
| turn | 0.869 | 0.417 | 0.0420 | 0.048 | 2.3 % | 0 | 0 | 0 | – | 5.5 → 5.3 % |
| move_turn | 1.408 | 0.182 | 0.0168 | 0.012 | 10.4 % | 0 | 0 | 0 | – | 24.7 → 24.5 % |

Stroke gate losses on W, as a share of all credit (`step7.gate_loss_breakdown_default`):

| Row | Anchor | Slip | Progress guard | Kept |
|---|---|---|---|---|
| fwd | 9.4 % | 4.9 % | 0 | 0.857 |
| side | 10.9 % | 48.7 % | 0 | 0.403 |
| back | 13.7 % | 3.7 % | 0 | 0.825 |
| diag | 13.3 % | 18.7 % | 0 | 0.679 |
| turn | 17.8 % | 4.0 % | 2.0 % | 0.761 |
| move_turn | 17.2 % | 11.7 % | 13.7 % | 0.572 |

The move_turn progress-guard loss is W's known failure there: it translates but barely turns.

## 8. R9 under the restated R1–R3 (21 neighbours of the final set)

| Neighbour | R1r worst | R3b max CI low | Verdict |
|---|---|---|---|
| force_hold 15 / 25 | 0.836 / 0.837 | +0.071 / +0.082 | PASS |
| n_off 1 | 0.848 | **+0.221** (W side) | **FAIL** |
| n_off 3 | 0.825 | −0.013 | PASS |
| rho0 0.4 / 0.6 | 0.821 / 0.846 | +0.055 / +0.057 | PASS |
| drho 0.2 / 0.4 | 0.827 / 0.843 | +0.062 / +0.056 | PASS |
| power 1.5 | **0.800** (X0x s2 turn) | +0.136 | **FAIL** (at the line) |
| c_max 1.25 / 1.75 | 0.837 | +0.066 | PASS |
| prog_lo 0.10 / 0.20 | 0.906 / **0.722** | +0.066 / +0.067 | PASS / **FAIL** |
| prog_width 0.10 / 0.20 | 0.883 / **0.777** | +0.066 | PASS / **FAIL** |
| prog_tau 3 / 5 | 0.837 / 0.839 | +0.066 | PASS |
| force_open 40 / 60 | 0.837 | +0.066 | PASS |
| min_loaded 4 / 6 | 0.837 / 0.836 | +0.067 / +0.064 | PASS |

R2r (slide ≤ 0.25 and side ≤ 0.2 × sideways) passes at every neighbour. The literal R9 fails at all
21, trivially, because the base fails literal R1–R3.

## Runs

All paths are under `parkour/logs/rsl_rl/gait_eval/`, at `suite__omni/seed001/<stamp>`: 1000
episodes each, with `complete.json` present and the heading guard ≤ 3e-7.

| Name | Role | Run / stamp |
|---|---|---|
| W (primary, dose) | healthy | `omni_phases/2026-09-23_clock_deprecation_w2/seed3_X03_c6x_A15pB` / `2026-09-25_20-31-29` |
| X0x s3 | healthy | `…_w2/seed3_X01_c6x_A15pB` / `2026-09-25_17-01-55` |
| X0x s2 | healthy | `…_w2/seed2_X02_c6x_A15pB` / `2026-09-25_18-45-27` |
| E1 (X08:c6) | healthy | `omni_phases/2026-09-23_clock_deprecation/step0_E1_X08-c6_visible` / `2026-09-23_21-23-41` |
| sideways head, flat and step | sideways | `side_phases/2026-09-11_0414_sideways_formation/seed2_2a_006_A15pB/{slow__side,step__side}` / `2026-09-12_09-18-39`, `2026-09-12_09-20-17` (100 episodes each) |
| Z t3 / Z c6 / A4500 | posture (R5/P3 no-flip only) | `…/seed3_X05_t3_A15pB` `2026-09-25_04-33-44`; `…/seed3_X05_c6_A15pB` `2026-09-25_06-52-47`; `…/seed3_X04_c6_A15pB` `2026-09-24_16-39-45` |
| E6 statue / E7 creeper | degenerate | `…/step0_E6_statue` `2026-09-23_21-59-41`; `…/step0_E7_creeper-1c` `2026-09-23_22-07-41` |
| E2 (X08:c4) | eval-noise pair (R8 CI, R3 noise) | `…/step0_E2_X08-c4_visible` `2026-09-23_21-31-11` |
| Forward-era (cross-plant, term alone) | degenerate / v4 | `crab_hex_forward_task/experiments/2026-08-10_0058_tripod_stability/`: `healthy_refs/gait_eval_h2000,h3000`, `fromscratch_tripod_v4_swap_short/gait_eval_h5000` (healthy); `fromscratch_tripod_reward_w0.15/gait_eval` (lunge); `fromscratch_tripod_v2_w0.15/gait_eval_midtrain` (tip-rock); `fromscratch_tripod_v3_w0.15_short/gait_eval_3000` (skate); `fromscratch_tripod_v3b_slide_short/gait_eval_3000` (drag); `fromscratch_tripod_v4_swap_short/gait_eval_3000,_5000` (v4) |

Economy: W's `params/env.yaml`, clock-free (asserted clock weights 0.001 / 0.001), with
`penalty_tibia_deviation_in_stance` (−0.28) priced from `joint_pos`.

## Reproduce

Run from the worktree root with the plan's PYTHONPATH:

```
PY=/home/nickmagus/krabby/isaac_venv/bin/python
T=parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task/experiments/tools/replay_stroke_posture.py
C=parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task/experiments/2026-09-26_stroke_posture
$PY $T gate --config $C/replay_gate_config.json --out <dir>              # literal R1-R9, P1-P4 (about 6 min)
$PY $C/replay_gate_calib/calib.py $C/replay_gate_config.json <dir> final= --r9r   # restated + R9r
$PY $C/replay_gate_calib/calib.py $C/replay_gate_config.json <dir> p15=power=1.5 noff3=n_off=3 ...
$PY $C/replay_gate_calib/fwd_era.py [k=v ...]                           # forward-era, term alone
$PY $C/replay_gate_calib/mirror.py W                                    # R8 mirror parity
$PY $T income <W stamp> --w-stroke 0.103 --w-centring 0.161             # §7
```
