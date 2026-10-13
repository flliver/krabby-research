# Report — a separate stand network for the omni crab (campaign `2026-10-11_stand_network`, 2026-10-11 → 2026-10-12)

**Status: COMPLETE 2026-10-12. STOP for the user's adoption / bake decision. Nothing is baked.** Arm `A-mask` meets the
pre-registered adoption-grade rule (PLAN L117-118: every hard gate at both env seeds, on the chosen arm, at all three
training seeds). s18, s16 and s17 each pass G1–G5 at es1 and es2 (C:L257-259). The walking head of record is unchanged:
S1 s18, `policy/6_p2a_field/model_19994.pt`, sha256 `968cc550…`. code_sha 14301cd2 and eval_sha 1d021625 did not move.
All code is campaign-local (`scripts/`). Adoption, bake and push are the user's.

Sources live in this directory unless a path says otherwise. `CHARTER.md` §3 is the record and is cited as C:L<line>.
PLAN.md is cited as P:L<line>.

## 1. The question and the answer

**Question** (CHARTER §1). Train a stand network separately from omni locomotion, by DAgger imitation of the scripted
re-stance controller R with S1 frozen in the loop. Route it at the latched stand slot next to S1 s18. Does it pass the
pre-registered closed-loop gates G1–G5 (STILL, HOMED, FALLS incl. exit handovers, WALLS, HANDOVER) at three training
seeds?

**Answer: yes, for `A-mask`.** In the suite stand, the routed pair takes 0 steps/s and 0 taps/s (REFS: 2.83 steps/s).
It homes 99.7–100 % of walkstop stands (R 97.0 / 97.7 %) with a median settle of 2.66–2.83 s (R 2.92 / 2.96 s). It has
0 stand falls and 0 exit-handover falls. field_hard is 0.7825 / 0.810 / 0.805 (REFS 0.8125, lines 0.756–0.757), and
the walking onset after a stand is 0.12 s (REFS 0.14).

There are two weaknesses. The HOLD cam trim is under-fit, so homing comes from re-steps. Post-settle steps / taps
read 0.38–0.47 / 0.74–0.88 per s against lines of 0.5 / 1.0. `A-full` failed G2 on exactly that taps clause.

The verdict is a gate verdict in simulation. It is not a deployment test (§10).

## 2. Design as run (PLAN D1–D6, Amendment 1)

- **Network (D1).** A copy of S1's `CrabHexActorCriticRMA` (obs 1149, action 18), warm-started from S1. Only the actor
  backbone trains; the scan encoder, history encoder and estimator stay frozen at S1's (OPEN-1, C:L86).
  - The executed action and the fitted output are one call: `act_inference(mask(obs with own estimate),
    hist_encoding=True)`.
  - **A-mask** zeroes cols 65–88 and their history frames (own last action + current-sense proxy) on the stand net's
    input only, after the estimator. **A-full** is unmasked.
- **Env (D2).** Teacher-v0 with preset 2a and the `P2__S1_s18` overlay: wall field, terrain curriculum, pushes
  ±0.5 m/s every 8 s, mass / CoM DR, 40-s episodes.
  - The command pool is set to stand ≈ 0.5, with resample draws of 3–12 s. The smoke realised a stand-row share of
    0.492.
  - Frozen S1 walks and the student (or R) stands, so every stand starts from a real handover.
  - DART σ is drawn per episode: 0 / 0.1 / 0.3 at ½ / ¼ / ¼.
- **Objective (D3).** Online DAgger of R with the validated constants. β goes 1 → 0 by it 1333 of 4000, and a fixed
  10 % of episodes stay R-executed as the concurrent control.
  - The buffer is a 4 GB reservoir (1.01 M rows). The loss is MSE on the actor mean, with per-leg inverse-branch
    weights and LIFT-onset oversampling (OPEN-2…5, C:L87-94).
  - Training settings: Adam 3e-4, 1024 envs, 1.54 s/it. Each arm is one job, and the judged checkpoint is
    `model_4000` (C:L175).
- **Switch and gates (D5, D6).**
  - The switch is hard, on the latched stand slot: no debounce, exit on the step the slot changes. `STAND_OBS=0` on
    both sides, and S1's EVAL_CARRY block applies to both nets.
  - The gates are closed-loop only, at es1 and es2. field_hard is pooled over es1–4 (n = 400).
- **Amendment 1** (user, after the lit review; C:L78-84, P:L188-241).
  - Adopted: the A1 recoverability pre-check before the pilot, and A2, a branch-gated third arm that runs only if no
    arm passes all of G1–G5 and at least one arm's only failing gate is G2 (P:L226-228).
  - Declined: fast β, gated takeover, onset weighting, and σ from the residual.
  - The lit review's exit-from-SEAT recommendation was conditional on a G3 exit failure, which never occurred.
- **User decisions logged before the reads they govern.**
  - A1 control-homed floor 0.75 (C:L81-82).
  - Proceed to the pilot after the A1 STOP (C:L168-170).
  - G3 walkstop stand falls are a raw count ≤ max(1, Z1's) per env seed (300 scheduled stands), not p2_read's rate per
    reached stand. This was ruled before any trained head's gate was read (C:L189-193).
- **Orchestrator changes, also logged before their reads.**
  - First-draw scoring for A1 and step 7 (OPEN-7 amendment, C:L118-121).
  - In the step-6 obs-parity check, the literal cols-0–1148 capture was replaced by three checks: env parity on every
    obs-affecting knob, column checks, and offline trainer→launcher action parity (C:L122-126).

## 3. Calibration (step 2) and the bridges (steps 3, 6)

**Calibration: 15 / 15 expected cells land as listed, 0 mismatches, 0 pending** (`calib.md`; C:L47-55). The read was
re-run after the G3 count ruling and still gave 15 / 15 (C:L198-200).

| Gate | Known-good (pass) | Known-bad (fail) |
|---|---|---|
| G1 STILL | R, HOLD | REFS (steps 2.83 / 2.83), ST2M (taps 16.08 / 15.33) |
| G2 HOMED | R (homed 0.970 / 0.977, settle 2.92 / 2.96 s) | HOLD, Z1, REFS (homed 0); Z2d (0.017 / 0.023) |
| G3 FALLS | R, HOLD (0 falls; 0 / 200 exit falls per seed) | — |
| G4 WALLS | HOLD (0.810), R (**0.825**, n = 400) | — |
| G5 HANDOVER | HOLD (0.12 s), R (0.12 s, first read) | — |

- **R's field_hard at n = 400 = 0.825.** `SN_RSTg_s18` re-ran R on field_hard es1–4. Its es1 is bitwise equal to the
  phase-2 probe `P2R_RSTg_s18` on every raw key (100 / 100 episodes), even though the launcher, the restance module and
  code_sha have all moved since then (C:L39-46; `calib.md` "Identity check").
- **Identity bridge: BITWISE.** `SN_ID_s18` runs S1 as the stand checkpoint with mask 0. It equals REFS_S1_s18 on
  every raw key in suite, walkstop and field_hard es1, with max |Δa| 0 over 235,000 / 180,000 / 47,500 stand steps
  (`bridge_identity.json`).
- **Swap-live: PASS.** `SN_IDM_s18` runs S1 with mask 1. Its stand action differs from S1's on all 180,000 stand
  steps, and its pre-entry rows equal REFS (`bridge_swaplive.json`). The smoke checkpoints load and act on every
  stand step (`smoke_ckpt_{mask,full}.json`).
- **Smokes: PASS** (`smoke_{mask,full,A1}.json`). Each check came in as planned (C:L135-141):
  - stand-row share 0.492 / 0.503;
  - DART σ 0.100 / 0.300;
  - control share 0.104;
  - forward parity bitwise;
  - reservoir at its 4.0 GB cap;
  - A1 executed rows were R-or-spoiler only, with 0 mismatches.

**Caveat (C:L56-57).** G3, G4 and G5 have no failing known-bad reference: every reference passes them. The calibration
shows that they compute and that R and HOLD pass. It does not show that their lines can bind. The tightest reference
margins were field_hard C0R 0.785 vs 0.7558 and onset Z2d 0.18 vs 0.44 s (`calib.md` "Negative-control coverage").
The heads' G3–G5 passes below should be read as "no harm seen at these resolutions", not as discriminating tests.

## 4. A1 recoverability pre-check (step 6b) and the user's decision to proceed

The run (C:L145-150) used the step-4 trainer env with σ = 0 and β = 1, so R executes every stand and there are no
student updates. It covered 700 its in ≈ 0.29 GPU-h. Half of the stands were spoiled for U[1, 4] s before R took
over. Source: `a1.json`.

The table counts **5154 scored stands** (first draw, scheduled ≥ 8 s, not timeout-cut). That is 14,703 records minus
1246 timeout-cut, 168 run-start and 8135 short stands. CHARTER L150 was corrected from a typo "6154" to 5154 at this report.

| Group | n | Homed | Line | Stand-fall share | Line | Falls before takeover | Exit falls |
|---|---|---|---|---|---|---|---|
| control (R from clean handovers) | 2586 | **0.969** | ≥ 0.75 | 0.0039 | — | 0 | 0 / 1995 |
| statue (zero action) | 906 | 0.975 | ≥ 0.775 | 0.0044 | ≤ 0.0139 | 4 | 0 / 700 |
| chatter (R + σ 0.3) | 838 | 0.994 | ≥ 0.775 | 0.0048 | ≤ 0.0139 | 3 | 0 / 663 |
| stuck lift (one leg held up, cam ±0.4) | 824 | 0.933 | ≥ 0.775 | **0.0534 FAIL** | ≤ 0.0139 | 28 | 0 / 614 |

- **R homes from every student-like state.** With R-time ≥ 4 s, homed is 0.961–1.000 in every spoiler bin, and the
  control reads 0.972 at ≥ 8 s. Legs left in the trim-only band at stand end are ≈ 0 (2 of 15,344 in the control).
- **The one failing clause is falls under the stuck-lift spoiler: 44 / 824.**
  - 28 of them fell inside the spoiler window itself.
  - The other 16 fell after takeover, all within 0.82 s of it. The post-takeover share is 0.020, against the control's
    0.004 (C:L162-166).
  - The descriptive reading: R adds no late falls, but it cannot catch a robot that is already tipping when it takes
    over.
- **Verdict STOP by rule. The user decided to proceed to the pilot** (C:L168-170). The grounds: R homes from every
  student-like state, the excess falls come from the spoiler window and the first 0.8 s after it, and G3 still gates
  falls at eval.

## 5. Gates G1–G5 per head (steps 9–10)

Each head is a stand net at `model_4000`, routed next to S1 s18. Cells read es1 / es2 unless pooled; **bold** marks a
failing value. Sources: head columns from `head_SN_Amask_s18.md`, `head_SN_Amask_s16.md`, `head_SN_Amask_s17.md` and
`head_SN_Afull_s18.md` (clause values and lines in the matching `.json`; per-head G4 SEs are (REFS − line) / 2). Reference columns are from `calib.md`.
R means `P2R_RSTg_s18`, except on field_hard, where it means `SN_RSTg_s18` es1–4. Z1 is `P2__Z1_s18` and Z2d is
`P2__Z2dR1_s18`. Every head stamp passed the D5 validity rule: `stand_net` variant, one checkpoint sha, one mask, and
S1's sha.

| Gate · statistic | Line | REFS | HOLD | R | Z1 | Z2d | A-mask s18 | A-mask s16 | A-mask s17 | A-full s18 |
|---|---|---|---|---|---|---|---|---|---|---|
| G1 suite steps/s | ≤ 0.5 | **2.83 / 2.83** | 0 / 0 | 0 / 0 | 0 / 0.083 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| G1 suite taps/s | ≤ 1 | **2.83 / 2.67** | 0 / 0 | 0 / 0 | 0.083 / 0.083 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| G1 suite clearance (m) | ≤ 0.05 | **0.135 / 0.133** | 1e-5 / 1e-5 | 0 / 0 | 0.0013 / 0.0012 | 8e-5 / 9e-5 | 2e-5 / 2e-5 | 1e-5 / 1e-5 | 1e-5 / 1e-5 | 3e-5 / 2e-5 |
| G1 off-ground share | ≤ 0.10 | **0.338 / 0.334** | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| G1 suite KS (m/s) | ≤ 0.5 | **1.72 / 1.72** | 0.098 / 0.104 | 0.098 / 0.105 | 0.180 / 0.178 | 0.135 / 0.143 | 0.097 / 0.108 | 0.104 / 0.107 | 0.108 / 0.105 | 0.096 / 0.103 |
| G2 walkstop homed | ≥ 0.90 | **0 / 0** | **0 / 0** | 0.970 / 0.977 | **0 / 0** | **0.017 / 0.023** | 0.997 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 | 0.997 / 1.000 |
| G2 settled | ≥ 0.90 | **0 / 0** | **0 / 0** | 0.970 / 0.977 | **0 / 0** | **0.173 / 0.187** | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 |
| G2 median settle (s) | ≤ 3.92 / 3.96 | **—** | **—** | 2.92 / 2.96 | **—** | 2.78 / 2.20 | 2.83 / 2.78 | 2.82 / 2.66 | 2.68 / 2.71 | 2.75 / 2.76 |
| G2 post-settle steps/s | ≤ 0.5 | **—** | **—** | 0.113 / 0.112 | **—** | **—** | 0.444 / 0.458 | 0.466 / 0.432 | 0.407 / 0.380 | 0.353 / 0.432 |
| G2 post-settle taps/s | ≤ 1 | **—** | **—** | 0 / 0 | **—** | **—** | 0.764 / 0.855 | 0.882 / 0.816 | 0.743 / 0.745 | **1.139 / 1.143** |
| G2 post-settle KS (m/s) | ≤ 0.5 | **—** | **—** | 0.149 / 0.147 | **—** | **—** | 0.152 / 0.159 | 0.162 / 0.162 | 0.155 / 0.155 | 0.154 / 0.154 |
| G3 walkstop stand falls (count of 300) | ≤ max(1, Z1's 0) = 1 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| G3 suite stand falls | = 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| G3 exit-handover falls ≤ 2 s (of 200 exits) | ≤ max(1, R's 0) = 1 | n/a | 0 / 0 | 0 / 0 | n/a | n/a | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| G4 field_hard es1–4 pooled, n = 400 | ≥ REFS − max(0.05, 2 SE) ≈ 0.756–0.757 | 0.8125 | 0.810 | 0.825 | 0.830 | 0.8225 | 0.7825 | 0.810 | 0.805 | 0.815 |
| G4 field_turn es1 | ≥ 0.94 | 0.99 | 0.99 | 1.00 | 1.00 | 0.99 | 0.99 | 0.97 | 1.00 | 0.99 |
| G4 stand-fall rate, field_turn es1 / field_hard | ≤ HOLD + 2 SE(diff) | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0.01 (line 0.020) / 0 | 0 / 0 | 0 / 0 |
| G5 walkstop onset median (s) | ≤ REFS + max(0.3, band) = 0.44 | 0.14 / 0.14 | 0.12 / 0.12 | 0.12 / 0.12 | 0.12 / 0.12 | 0.18 / 0.18 | 0.12 / 0.12 | 0.12 / 0.12 | 0.12 / 0.12 | 0.12 / 0.12 |
| G5 onset stalls (of 200) | ≤ REFS + 1 = 1 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| G5 max \|height − REFS\| (m), suite + walkstop | ≤ 0.03 | 0 | 0.0046 | 0.0067 | 0.0053 | 0.0105 | 0.0054 | 0.0067 | 0.0068 | 0.0056 |
| **Verdict** | | FAIL G1, G2 | FAIL G2 | PASS | FAIL G2 | FAIL G2 | **PASS all** | **PASS all** | **PASS all** | FAIL G2 |

Notes on the table:
- "—" means there is no settled stand to read. The clause fails closed (`calib.md` deviation 15).
- Hover was 0 per 100 everywhere (line ≤ 5).
- The pooled es1+es2 onset is 0.12 s for every head.
- The G4 field_hard lines per head are s18 0.7557, s16 0.7572, s17 0.7569 and A-full 0.7574 (SE 0.0284 / 0.0277 /
  0.0278 / 0.0275).
- The s16 field_turn stand fall is one fall in 100 episodes, inside HOLD + 2 SE (C:L245-246).
- The evals for each head are 9 rows: suite es1–2, walkstop es1–2, field_hard es1–4 and field_turn es1. The pilot's
  set cost ≈ 0.6 GPU-h (C:L202).

**The binding margins are all in G2 post-settle.**
- Smallest steps margin: s16 es1, 0.466 vs 0.5 (0.034). s18 es2 has 0.042; s17's smallest is 0.093.
- Smallest taps margin: s16 es1, 0.882 vs 1.0 (0.118). s18 es2 has 0.145; s17 has 0.255.
- Outside G2, the tightest clauses are field_hard at s18, 0.027 above its line (less than one SE), and field_turn at
  s16 (0.97 vs 0.94).

## 6. Arm rule (step 9)

**Chosen arm: `A-mask`** (C:L231-235). This is decide-record-notify under the pre-registered rule (P:L171-174): when
only one arm passes, that arm is chosen. Only `A-mask` passed all of G1–G5 at es1 and es2.

- **`A-full` fails G2 on one clause: post-settle taps 1.139 / 1.143 per s** against a line of 1. Its other G2 clauses
  pass: homed 0.997 / 1.000, settled 1.0 / 1.0, settle 2.75 / 2.76 s, post-settle steps 0.353 / 0.432. G1, G3, G4
  (field_hard 0.815) and G5 pass (C:L222-230).
- **A2 was not triggered.** Its trigger needs no arm to pass all gates (P:L226-228), and one arm passed.
- **Replicates.** On the user's go (C:L237-239), A-mask was replicated at s16 and s17. Both pass at es1 and es2
  (C:L242-256).
- **The two arms share the binding statistic.** A-mask's post-settle taps are 0.74–0.88 and A-full's are 1.14, against
  a line of 1. Both arms show the same trim under-fit (§7). What the mask bought, at s18 (one seed per arm), is the
  taps margin. It did not change the mechanism.

## 7. Observe pass (closed loop, reported, not gated)

The observe pass reads every latched stand row of walkstop and suite at es1 and es2. Applied = clip(stand-net action,
±1), and R is evaluated on the same row. Sources: `observe_SN_Amask_s18.md`, `observe_SN_Amask_s16.md`,
`observe_SN_Amask_s17.md` and `observe_SN_Afull_s18.md` (JSON beside each). The Z2d values are the p2r figures quoted
in those files' headers. LIFT numbers come from walkstop only, because suite stands start homed (C:L195).

| Statistic (pooled) | A-mask s18 | A-mask s16 | A-mask s17 | A-full s18 | R | Z2d |
|---|---|---|---|---|---|---|
| walkstop LIFT onsets | 2064 | 2017 | 1943 | 2184 | — | — |
| μ_hip on LIFT rows | −0.479 | −0.472 | −0.491 | −0.479 | −0.5 | −0.054 |
| LIFT rows with hip < −0.25 | 99.5 % | 99.4 % | 99.5 % | 98.6 % | — | 1.9 % |
| onset launched within 0.3 s | 99.6 % | 99.7 % | 99.3 % | 98.9 % | — | — |
| first onset per stand-leg launched | 99.7 % | 99.5 % | 99.4 % | 99.1 % | — | — |
| HOLD \|applied − R\| mean, cam / hip | 0.015 / 0.015 | 0.009 / 0.012 | 0.011 / 0.016 | 0.015 / 0.014 | 0 | — |
| HOLD cam-trim slope, within-leg, \|c\| ∈ [0.03, 0.3] | −0.032 | −0.029 | −0.028 | −0.028 | −0.52 / −0.50 / −0.52 / −0.53 (same rows) | per leg −0.04…−0.12 |
| same, unsaturated band \|c\| ∈ [0.03, 0.1] | −0.077 | −0.116 | −0.083 | −0.052 | −1 | — |
| HOLD rows in [0.03, 0.3]: pooled (worst leg) | 13.3 % (MR 60.1 %) | 3.1 % (RL 4.6 %) | 6.4 % (MR 19.5 %) | 12.1 % (ML 41.8 %) | — | — |
| SEAT \|applied − R\| hip mean | 0.063 | 0.061 | 0.061 | 0.057 | 0 | — |

- **The LIFT onset is learned, which is the p2r failure fixed.**
  - μ_hip on LIFT rows is −0.472 to −0.491, against R's −0.5 and Z2d's −0.054.
  - The median launch latency is 0.00 s. The hip is already below −0.25 on the onset row in 2036 / 2064 (s18),
    1996 / 2017 (s16), 1907 / 1943 (s17) and 2103 / 2184 (A-full) onsets.
  - The D3 recipe (labels on student-visited states, per-leg branch weights, onset oversampling) fixed this without
    the A2 branch head. Which component did the work is not isolated.
- **The HOLD cam trim is still under-fit, as in Z2d.**
  - The within-leg slope is −0.028 to −0.032, against R's −0.50 to −0.53 on the same rows. On the unsaturated band it
    is −0.052 to −0.116, against −1.
  - The per-leg spread varies by seed. At s18, leg MR sits in the band on 60 % of its HOLD rows; at s16 no leg exceeds
    4.6 %.
  - **The net homes by re-stepping, not by trimming.** Post-settle steps are 0.38–0.47 per s, against R's 0.11, and
    post-settle taps are 0.74–0.88 per s, against R's 0 (§5). This is why G2's post-settle clauses bind.
- **Post-settle margins are thin.** The worst are 0.034 steps/s and 0.118 taps/s, both at s16 es1. A-full shows what a
  small shift does: the same homed and settle numbers, with taps at 1.14.
- **Walls.** A-mask field_hard reads 0.7825 / 0.810 / 0.805, against REFS 0.8125.
  - The gaps are −0.030 / −0.0025 / −0.0075. SE is ≈ 0.028 per head, and all three gaps sit inside the ±0.04
    process-state band, which the G4 floor of 0.05 was set above (P:L13).
  - The pilot's 0.7825 did not replicate (s16 0.810, s17 0.805), so it reads as noise rather than an arm effect.
  - For comparison: A-full 0.815, R 0.825, HOLD 0.810.
- **Reported only** (`head_*.md` "Reported only"; `calib.md`):
  - Walkstop hip-yaw max is 2.1–3.6° for the heads, against R's 10.3 / 17.2° and HOLD's 22.5 / 23.9°. The yaw means
    are 0.37–0.46°, against R's 0.36–0.40°.
  - Six-down (suite / walkstop) is s18 0.33 / 0.42 and 0.25 / 0.39, s16 0.24 / 0.39 and 0.13 / 0.32, s17 0.16 / 0.24
    and 0.08 / 0.23, and A-full 0.34 / 0.48 and 0.30 / 0.46. R reads 0.29 / 0.37 and 0.20 / 0.32. Six-down is the
    static-indeterminacy lottery, which is why it is not gated.

## 8. Train-time reads against the closed-loop gates

The train-time statistic is the step-7 premise read. It counts σ = 0 stands longer than 4 s, scored on their first
draw, over its 1334–4000. The student is compared with the concurrent 10 % R-executed slice on the same iterations and
terrain levels. Sources: `pilot_premise_final.json`, `afull_premise_final.json`, `s16_premise_final.json` and
`s17_premise_final.json`; the mid-run reads are in the `*_premise_mid.json` files.

| Head | Train homed, student / R slice (ratio) | Train stand falls, student / R | Train exit falls, student / R | Mid-run ratio (line 0.5) | Eval walkstop homed es1 / es2 (R 0.970 / 0.977) | Eval falls (stand; exit) |
|---|---|---|---|---|---|---|
| A-mask s18 | 0.982 (n 19255) / 0.945 (n 4443) (1.04) | 0.80 % / 0.61 % | 8 / 15141 vs 2 / 3489 | 1.03 | 0.997 / 1.000 | 0; 0 / 200 per seed |
| A-mask s16 | 0.980 (19288) / 0.941 (4203) (1.04) | 0.85 % / 0.79 % | 3 / 15101 vs 0 / 3230 | 1.04 | 1.000 / 1.000 | 0; 0 / 200 |
| A-mask s17 | 0.979 (19228) / 0.952 (4266) (1.03) | 0.89 % / 0.66 % | 7 / 15000 vs 2 / 3331 | 1.03 | 1.000 / 1.000 | 0; 0 / 200 |
| A-full s18 | 0.984 (18967) / 0.950 (4224) (1.04) | 0.66 % / 0.69 % | 2 / 14867 vs 0 / 3301 | 1.04 | 0.997 / 1.000 | 0; 0 / 200 |

- **Where they agree.**
  - Both settings put the student above R on homed: ×1.03–1.04 in training, and 0.997–1.000 against 0.970 / 0.977 in
    the eval.
  - The student's falls are comparable to R's: per head 0.80 vs 0.61 % (s18), 0.85 vs 0.79 % (s16), 0.89 vs 0.66 %
    (s17), 0.66 vs 0.69 % (A-full) in the pushed, terrain-level training env (s17 ≈ 1.3× the R slice, ≈ 1.4 SE), and
    0 against 0 in the eval. The pilot's mid-run watch item (student falls ≈ 3× a small control count, C:L181-182)
    closed as counts grew (C:L186).
  - The premise never came close to its stop. The lowest recorded ratio was 0.96, in the pilot's β → 0 window
    (C:L177-178).
- **Where the train-time read is blind.**
  - It cannot see the statistic that binds. Train-time homed ranks A-full at or above A-mask (0.984 vs 0.982), but the
    arms separated only on G2 post-settle taps. The premise read checks recoverability; it is not a proxy for G2.
  - It hints only faintly at the trim under-fit. At stand end, 0.08–0.13 % of loaded legs are in R's trim-only band
    (|c| ∈ (0.1, 0.3]), against 0–0.03 % for the R slice.
- **Resolution (arithmetic on the rows above).** At the train-time exit-fall rates (≈ 1–5 per 10,000 exits), 200 exits
  per seed would expect ≈ 0.1 falls. At the train-time stand-fall rates (0.66–0.89 %), 300 stands would expect ≈ 2–3.
  - The eval's 0 stand falls are therefore a statement about the flat, push-free walkstop, not about the pushed regime.
    R shows the same split, at 0.61–0.79 % in training and 0 in the eval.
  - The pushed-stand evidence is train-time only (§10).

## 9. Cost

| Stage | GPU-h | Source |
|---|---|---|
| Step 2: R field_hard es1–4 rows (`SN_RSTg_s18`) | ≈ 0.25 | C:L39 |
| Steps 3 + 6: identity bridge, swap-live, three smokes, smoke-checkpoint rows | ≈ 0.5 | C:L130 |
| Step 6b: A1 (700 its) | ≈ 0.29 | C:L149-150 |
| Step 7: pilot `SN_Amask_s18` training (4000 its) | ≈ 1.65 | C:L183 |
| Step 9: pilot evals (9 rows) | ≈ 0.6 | C:L202 |
| Step 8 + 9: `SN_Afull_s18` training + evals | ≈ 2.7 (by difference from the ≈ 6.0 running total, C:L236) | derived |
| Step 10: `SN_Amask_s16` + `SN_Amask_s17` training + evals | ≈ 4.6 (by difference, ≈ 10.6 − 6.0) | derived |
| **Total** | **≈ 10.6** (plan ≈ 12.3; A2's conditional ≈ 3.1 not spent) | C:L259, P:L240 |

Step 0 (the lit review) and every read (calibration, heads, observe, premise) ran on the CPU.

## 10. What is not shown, open items, and the user's options

**Not shown / open.**
- **No hardware.** Everything above is Isaac, on one walking head (S1 s18) and one plant. The stand nets were never
  run next to S1 s16 or s17.
- **No pushes during a stand in eval.** The pushed regime has only train-time evidence: stand falls of 0.66–0.89 %,
  matching the R slice (§8). Lit review §12 rec 6 asks for a push-perturbed stand read.
- **The HOLD cam trim is under-fit** (§7), and homing depends on re-steps.
- **The G2 post-settle margins are thin** (0.034 steps/s, 0.118 taps/s at worst). The records carry no SE for these
  rates, so the per-seed ordering may not survive a re-eval.
- **G3–G5 have no failing known-bad reference** (§3). field_turn is es1 only, because REFS and R have no es2.
- **Stand-hold facing drift** is listed as "reported" in D6 (P:L116), but no head read carries it.
- **Two-network deployment / export is not built.** There is no `policy/` bundle and no manifest entry. There is no
  play or video path for the routed pair: `sn_stand_net_eval_crab_hex_gait.py` is an eval launcher run through
  run_arm `--row-eval`. The switch and the A-mask input function (`scripts/sn_common.py`) exist only inside that
  launcher and the trainer.
- **Tooling defect (phase-2, not edited).** `run_arm.sh` `drivers_idle` uses plain `ps -eo args`, whose output is
  truncated under an exported `COLUMNS`. `sn_eval_rows.sh` repeats the wait at `ps -ww` as a workaround (C:L128-129).
  A minor defect of our own: an interrupted `recover_probe` writes an A1 verdict on partial stands, so read A1 only from
  a run whose status is complete (C:L127-128).

**Options for the user** (none is taken by this report).
1. **Adopt `A-mask` as the stand network of record and pick one checkpoint.** All three are adoption-grade.
   - **Recommendation: s17**, `runs/SN_Amask_s17/model_4000.pt`, sha256 `769cf253…`.
   - The tie-break is proposed here, after the reads, so it is not pre-registered:
     1. the largest worst-case margin on the clauses that bind (G2 post-settle steps and taps, where A-full failed);
     2. then field_hard closest to REFS.
   - On (1), s17 leads outright. Its worst steps margin is 0.093 (s18 0.042, s16 0.034), and its worst taps margin is
     0.255 (s18 0.145, s16 0.118). On (2) it is second, at 0.805 (s16 0.810, s18 0.7825). It also reads lowest on
     six-down and hip-yaw max, though both are reported only.
   - Caveat: the margins come from one 300-stand eval per env seed. All three stand nets ride the same S1 s18, so the
     stand-net seed carries no lineage meaning.
   - A walls-first tie-break would pick s16 instead, at the smallest steps margin.
2. **Bundle / export (bake work, PLAN Later phases P:L264).** This would cover:
   - a two-network entry in `policy/` (manifest + `bundle_policy.py`);
   - the stand-slot switch and the A-mask input function outside the eval launcher;
   - a play / video path for the routed pair, under the video-memory caps.
3. **A trim fix, as a later phase.** The target statistics are the within-leg HOLD slope and the post-settle
   steps / taps. Candidates already on record:
   - the A2 branch head (designed, ≈ 3.1 GPU-h, P:L238);
   - lit-review rec 3 (onset / trim weighting smoothed over |c|, so that HOLD-slope rows count);
   - rec 4 (σ from the measured residual);
   - a PPO fine-tune past R (P:L263).
4. **More evidence, before or after adoption.** Candidates: a push-perturbed stand eval, a facing-drift read, and
   repeat evals of the chosen checkpoint, to put an SE on the post-settle rates.
5. **Keep HOLD as the fallback stand.** It is STILL, with 0 falls, but its cams are unhomed (C:L52-53).

## 11. Pointers

- **Record:** `CHARTER.md` §3 (L34-259); plan `PLAN.md` (D1–D6 L42-118, Amendment 1 L188-241); predecessor
  `../2026-10-07_omni_phase2/P2R_CLOSE.md`; lit review `../lit-review-stand-network.md` §12.
- **Reads:**
  - `calib.{md,json}`, `a1.json`, `bridge_identity.json`, `bridge_swaplive.json`;
  - `smoke_{mask,full,A1}.json`, `smoke_ckpt_{mask,full}.json`;
  - `{pilot,afull,s16,s17}_premise_{mid,final}.json`;
  - `head_SN_{Amask_s18,Amask_s16,Amask_s17,Afull_s18}.{md,json}` and the matching `observe_*.{md,json}`.
- **Code (campaign-local):** `scripts/sn_common.py`, `sn_stand_net_eval_crab_hex_gait.py` (launcher variant
  `stand_net`, sha256 ccefcb1a…, unchanged since b6578792, before every eval; the stamps' `launcher_sha256` 59539301 is
  the imported phase-2 launcher, not this file), `sn_eval_rows.sh`, `sn_bridge_read.py`, `sn_calib.py`,
  `sn_observe_read.py`, `sn_dagger{,_core,_spec,_parity}.py`, `sn_dagger.sh`, the queues `sn_queue_{pre7,step89,step10}.sh`,
  and `IMPLEMENTATION_MAP.md`.
- **Checkpoints** (under `parkour/outputs/omni_plan/stand_network/runs/`, untracked):

  | Head | Path | sha256 | Mask | Gates |
  |---|---|---|---|---|
  | A-mask s18 (pilot) | `SN_Amask_s18/model_4000.pt` | `f7228a99…` | 1 | PASS all |
  | A-mask s16 | `SN_Amask_s16/model_4000.pt` | `b17c5b5a…` | 1 | PASS all |
  | A-mask s17 | `SN_Amask_s17/model_4000.pt` | `769cf253…` | 1 | PASS all |
  | A-full s18 | `SN_Afull_s18/model_4000.pt` | `1671fd55…` | 0 | FAIL G2 |
  | S1 s18 (walking, unchanged) | `parkour/logs/rsl_rl/crab_hex_omni_teacher/2026-10-07_07-50-10_seed18_X08_p2a_A15pB/model_19994.pt` = `policy/6_p2a_field/model_19994.pt` | `968cc550…` | — | — |

- **Eval stamps:** `parkour/outputs/omni_plan/phase2/lr/<NAME>/<row>/seed00<k>/`, for `SN_Amask_s{18,16,17}`,
  `SN_Afull_s18`, `SN_ID_s18` (identity), `SN_IDM_s18` (swap-live), `SN_SMK_{mask,full}_s18` (smoke checkpoints) and
  `SN_RSTg_s18` (R field_hard). Training logs are in `runs/<NAME>.log`.
- **Evaluate the two-network setup** (one Isaac process at a time; `sn_eval_rows.sh` refuses while the trainer runs).
  The script takes the REFS stamp's env (STAND_OBS=0, CLOCK_HIDE=1, STD_PARAM soft, INIT_STD 0.8, ellipse
  0.35:0.35:0.25). `P2_ENV_EXTRA` must be empty and no `P2R_*` variables may be set. `MASK` must match the checkpoint
  (1 for A-mask); the launcher refuses a mismatch. Use a fresh NAME: the readers take the newest complete stamp per
  name and row, so re-running under an existing name would supersede the recorded stamps.

  ```bash
  S=/home/nickmagus/krabby/krabby-research/parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task/experiments/2026-10-11_stand_network/scripts
  CK=/home/nickmagus/krabby/krabby-research/parkour/outputs/omni_plan/stand_network/runs/SN_Amask_s17/model_4000.pt
  bash $S/sn_eval_rows.sh --dry-run SN_Amask_s17_re $CK 1 walkstop__omni@1          # print the run_arm commands, no Isaac
  bash $S/sn_eval_rows.sh SN_Amask_s17_re $CK 1 suite__omni@1 suite__omni@2 walkstop__omni@1 walkstop__omni@2 \
       field_hard__omni@1 field_hard__omni@2 field_hard__omni@3 field_hard__omni@4 field_turn__omni@1
  PY=/home/nickmagus/krabby/isaac_venv/bin/python                                   # CPU reads (numpy)
  $PY $S/sn_bridge_read.py --head SN_Amask_s17_re --live                            # stand-net action reaches stand rows
  $PY $S/sn_calib.py --head SN_Amask_s17_re --json <dir>/head_SN_Amask_s17_re.json --md <dir>/head_SN_Amask_s17_re.md
  $PY $S/sn_observe_read.py --head SN_Amask_s17_re --json <dir>/observe_SN_Amask_s17_re.json --md <dir>/observe_SN_Amask_s17_re.md
  ```

  With no arguments, `sn_eval_rows.sh` runs the identity bridge (S1 as the stand checkpoint, mask 0, `SN_ID_s18`).
  **Play / video:** there is no two-network viewer yet; see option 2.