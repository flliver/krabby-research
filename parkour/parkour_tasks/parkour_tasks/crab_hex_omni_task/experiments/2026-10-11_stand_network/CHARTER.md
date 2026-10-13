# Stand network for the omni crab (separate from locomotion) — CHARTER

- **Pre-registered:** 2026-10-11, before any run of this campaign.
- **Plan of record:** [`PLAN.md`](PLAN.md) — approved by the user 2026-10-11 after a 3-pass audit (converged: opus·xhigh
  + fable·high on pass 3; trail at the top of PLAN.md). Its design decisions D1–D6 (architecture, training env,
  objective, code placement, switch contract, gates), steps, stop points, budget and seeds ARE the pre-registration;
  they are not restated here so the two cannot drift.
- **Precedence:** this CHARTER is the record. Any change to a PLAN.md rule is logged in §3 before the result it affects
  is read.
- **Predecessor:** stage p2r of `../2026-10-07_omni_phase2/` (closed 2026-10-11; handoff in `P2R_CLOSE.md`).

## 1. Question and outcome

**Question.** Can a stand network trained separately from omni locomotion (DAgger imitation of the scripted re-stance
controller R, S1 frozen in the loop) and routed at the latched stand slot next to the head of record S1 s18
(`policy/6_p2a_field/model_19994.pt`) pass the pre-registered closed-loop gates G1–G5 (PLAN.md D6) — STILL, homed /
settled, falls incl. exit handovers, walls unchanged, handover — at three training seeds?

**Outcome.** Per-arm and per-seed G1–G5 reads against REFS_S1_s18 / P2R_HOLD_s18 / P2R_RSTg_s18, an arm choice by the
pre-registered rule, and an adoption-grade verdict. **Adoption, bake and push are the user's.**

## 2. Ground rules

- **Code:** campaign-local only (`scripts/` here; excluded from code_sha and eval_sha). At pre-registration:
  HEAD `2ce304a4`, code_sha `14301cd2f2ba6080…`, eval_sha `1d021625844bb372…`. eval_sha must not move: never edit the
  eval harness, `gait_eval/`, scenarios files, `eval_categories.py`, `run_phases.py`. The phase-2 launcher
  `p2r_mode_eval_crab_hex_gait.py` is imported and extended, never edited. A shared-code change is a stop-and-ask.
- **Running:** one Isaac process at a time (detached queue runner + log-sentinel watcher); stop with
  `systemctl --user stop`, never SIGKILL Kit; check the Kit DerivedDataCache before the first launch; memory-capped
  scopes; no GPU rendering while the queue can launch.
- **Git:** commit only this campaign's paths; never `git add -A`; never push. Other sessions share the checkout.
- **Authority:** push-notify at every stop point (PLAN.md "Stop points"); the step-9 arm choice is decide-record-notify.

## 3. Log

- 2026-10-11, **campaign opened.** PLAN.md approved (audit trail in its header). Step 1 done (this dir, CHARTER,
  PLAN). Nothing run. Next: step 0 (lit review) and step 2 (gate calibration), on the user's go.
- 2026-10-11 ~23:00, **user go for steps 0 and 2.** Nothing else authorised (steps 3+ wait for the user).
- 2026-10-11 23:19–23:34, **step 2 GPU rows** (`scripts/sn_calib_rows.sh`, ≈ 0.25 GPU-h, one Isaac process, via the
  phase-2 `run_arm.sh --row-eval` path, guard anchored on REFH_S1_s18 = HEAD code_sha 14301cd2, eval_sha 1d021625):
  R (validated constants, `--p2r-mode restance`, STAND_OBS=clock) on S1 s18 as `SN_RSTg_s18`, field_hard es1–4.
  - **Identity: PASS.** SN_RSTg_s18 field_hard es1 is bitwise equal to P2R_RSTg_s18 es1 on every raw key, 100 / 100
    episodes, although the launcher (b5ba0c63 → 59539301), restance module (7006dff1 → 7a16b8d1) and code_sha
    (07e02578 → 14301cd2) moved since the probe. The stand banks differ only by the reservoir cap (100000 then, 20000
    now); no step of this campaign reads them.
  - R field_hard pooled n = 400: **0.825** (REFS 0.8125, HOLD 0.810, C0R 0.785).
- 2026-10-12 00:01, **step 2 calibration read: 15 / 15 expected cells land as listed, 0 mismatches, 0 pending — the
  step-2 STOP does not fire.** `scripts/sn_calib.py` (campaign-local, CPU, reuses the phase-2 readers read-only) →
  `calib.json` / `calib.md`. Built by an opus·xhigh agent, verified from three lenses (numbers recomputed from raw
  npz without the tool's code; gate logic vs D5/D6; expectations on the merits), 6 minor defects fixed, re-verified
  by fable·high (no remaining blocking defect).
  - G1: R, HOLD PASS; REFS FAIL (steps 2.83 / 2.83), ST2M FAIL (taps 16.08 / 15.33). G2: R PASS (homed 0.970 / 0.977,
    settle 2.92 / 2.96 s vs lines 3.92 / 3.96); HOLD, Z1, REFS homed 0, Z2d 0.017 / 0.023 FAIL. G3: 0 falls on every
    reference; HOLD, R, ST2M 0 / 200 exit-handover falls per seed. G4: every reference PASS (line 0.758). G5: R's
    walkstop onset (in no earlier read) **0.12 s** at es1, es2 and pooled (REFS 0.14, HOLD 0.12; line 0.44 s), 0 stalls.
  - **Caveat (recorded, not a mismatch):** G3, G4 and G5 have no known-bad reference in D6 and every reference passes
    them, so the calibration shows they compute and that R / HOLD pass, not that their lines can bind.
  - **Deviations from p2_read's rules** (each listed in calib.md): the D5 STAND_OBS=0 validity rule for REFS / HOLD /
    ST2M / C0R and for `stand_net` heads (which must also carry one stand-ckpt sha and one `stand_mask` per head — a
    field contract for the step-3 launcher: `p2r_mode.specialist.sha256`, `p2r_mode.stand_mask`); G5 adds a per-seed
    stall clause (censored onsets ≤ REFS + 1) because the onset median drops stalls; G3's exit clause audits the
    denominator (200 scheduled exits per seed, R's episode count); a head's complete stamp with no readable statistic
    FAILS (p2_read reads PENDING); Z1's suite es1 resolves to its driver stamp (lr row missing).
- 2026-10-12 00:03, **step 0 lit review done** (`../lit-review-stand-network.md`; 6 angles, 64 raw findings, 12 / 12
  claims held after one-verifier-per-claim, 20 gap findings, 84 references). **Its verdict: nothing contradicts D1–D3,
  so the step-0 STOP does not fire.** It recommends amendments that would change the pre-registered design; they are
  the user's call, not applied:
  0. a recoverability pre-check before step 7 (R taking over from student-like failure states, ≈ 0.3 GPU-h);
  2. β decayed within the first iterations (indicator or 0.5^i) and the R budget spent on discrepancy-gated takeovers
     with hysteresis, keeping the 10 % control slice;
  3. onset weighting by a last-action-only predictor's error, smoothed over (branch, |c| − eps_step);
  4. DART σ per channel from the measured closed-loop |μ − R| instead of {0, 0.1, 0.3};
  5. a branch-gated head (classifier on R's branch label + branch-masked regression) as a pre-registered third arm if
     an arm is STILL but misses G2 — the review's main disagreement with D1's flat MSE head (needs a loader adapter);
  6. exit only from R's SEAT state / post-exit crediting if G3's exit clause fails.
- **STOP (decide-record-notify): steps 0 and 2 complete; step 3 (launcher variant) waits for the user's go and their
  call on the lit-review amendments.**
- 2026-10-12, **USER DECISION at the step-0/2 stop:** adopt two lit-review amendments — the recoverability pre-check
  (A1, step 6b) and the branch-gated conditional third arm (A2) — and decline fast β, gated takeover, onset weighting
  and σ-from-residual. Go through step 6 (+ 6b); stop before step 7.
  - PLAN.md "Amendment 1" written, audited 1 pass (opus·xhigh): 6 blocking, all fixed (trail in PLAN.md); one needed
    the user's number: **control homed floor 0.75** for A1 (user decision 2026-10-12).
  - Step 9's arm rule and the stop points now route to A2 when an arm's only failing gate is G2; A1 PASS records and
    notifies, it does not launch step 7.
- 2026-10-12, **implementation decisions** (orchestrator; from `scripts/IMPLEMENTATION_MAP.md` §8, before any code):
  - OPEN-1: the student trains the actor backbone only; scan encoder, history encoder and estimator stay frozen at S1's.
  - OPEN-2: Adam lr 3e-4; 24 env steps per iteration; fixed gradient steps per iteration (builder sets ≤ 16
    minibatches of ≤ 4096); reservoir buffer ≤ 4 GB GPU (smoke asserts the projection); num_envs 1024 (memory note:
    more envs per process nearly free) unless the smoke shows otherwise.
  - OPEN-3: branch balancing per leg — each leg's 3 action columns weighted by the inverse frequency of that leg's R
    branch in the buffer; LIFT-onset rows (any leg entering LIFT this step) oversampled in row sampling.
  - OPEN-4: the 10 % R-executed control is a per-episode draw (envs stay exchangeable; same terrain levels).
  - OPEN-5: DART σ per episode ∈ {0: ½, 0.1: ¼, 0.3: ¼} on executed stand rows (student and β-R episodes); σ = 0 on
    the control slice and throughout A1; labels always clean R.
  - OPEN-6: `init_at_random_ep_len` on, from a private seeded generator.
  - OPEN-7: a stand = a maximal run of stand-slot rows in an episode; scheduled end = entry + the current draw's
    time_left, extended on each stand→stand resample merge; A1 scores stands with scheduled length ≥ 8 s and scheduled
    end before the episode timeout; a fall before the scheduled end is scored (fall = 1, homed = 0).
  - OPEN-8: A2 heads write `variant: stand_net` + `branch_head: true` (sn_calib's rule unchanged).
  - OPEN-9: `stand_net` records R's target / branch on stand rows now (observe-ready; R is deterministic, no RNG).
  - Identity-bridge miss rule (PLAN "left for implementation", stated before the run): the bridge must be bitwise =
    REFS on every raw key; on a miss, PASS-with-note only if actions differ ≤ 1e-5 and every G1–G5 statistic is
    identical, else STOP and ask. Trainer↔launcher action parity: same batch shape in one process, CPU `torch.equal`;
    GPU atol 1e-5.
- 2026-10-12, **steps 3–5 code landed (campaign-local + tests/unit; CPU only, nothing run on Kit).** Workflow
  wf_456cfa24-68c: opus·xhigh builders (launcher, then trainer), opus·xhigh + fable·high reviews (2 blocking + 10
  minor), opus·xhigh fix, fable·high re-review (both blocking fixed, nothing regressed). 195 CPU unit tests pass;
  code_sha 14301cd2 / eval_sha 1d021625 unchanged, no dirty code paths.
  - Files: `scripts/sn_common.py` (the one stand-net input function: estimate on the unmasked prop → new tensor →
    `stand_mask_obs` iff mask), `scripts/sn_stand_net_eval_crab_hex_gait.py` (launcher variant `stand_net`; private
    by-path P2L instance, StandNetMode with a full `__call__`), `scripts/sn_eval_rows.sh`, `scripts/sn_bridge_read.py`,
    `scripts/sn_dagger.py` / `sn_dagger_core.py` / `sn_dagger_spec.py` / `sn_dagger.sh` / `sn_dagger_parity.py`
    (DAgger trainer, modes `dagger` and `recover_probe`), `tests/unit/test_sn_*.py`, `scripts/IMPLEMENTATION_MAP.md`.
  - Raw keys added by stand_net: `sn_stand_action`, `sn_s1_action`, `sn_r_target`, `sn_r_branch`; pre-Kit refusals:
    STAND_OBS ≠ 0, ENV4 ≠ REFS, env-level STAND_MASK / PROXY, KRABBY_RESTANCE, STAND_HOLD / PARK set; post-Kit,
    pre-episode refusals: stand-ckpt estimator ≠ S1's; a trainer checkpoint whose recorded `stand_mask` ≠
    `--stand-mask` (review blocking 1); a non-trainer checkpoint that is not S1.
  - **OPEN-7 amendment (review blocking 2, logged before any A1 / step-7 read):** A1 and step-7 scoring use each
    stand's FIRST draw — its scheduled length, timeout cut, fall before its scheduled end, and homed / |c| / trim on its
    last row. Merging stand→stand resamples extends only stands that survived their first draw, so a floor on the
    merged length made inclusion outcome-dependent. Stands already in progress at run start are excluded.
  - **Step-6 deviation:** the literal "cols 0–1148 of one stand row captured in the trainer and in stand_net from the
    same state" check is replaced by env parity on every obs-affecting EVAL_CARRY knob vs the REFS stamp plus layout /
    clock / stand-bit column checks on captured rows (eval stamps carry no obs cfg; Teacher vs Walk-Play share the obs
    cfg class, map C5); the offline trainer→launcher action-parity check (same batch, CPU bitwise, GPU atol 1e-5)
    pins the forward path.
  - Known minor (watched by hand): a recover_probe run that ends interrupted still writes an A1 verdict on partial
    stands — read A1 only from a run whose status is complete. The phase-2 run_arm.sh `drivers_idle` uses plain
    `ps -eo args` (truncated under an exported COLUMNS); sn_eval_rows.sh works around it; not edited (phase-2 file).
- 2026-10-12 03:24–03:54, **steps 3 + 6 GPU checks: all PASS** (`scripts/sn_queue_pre7.sh`, ≈ 0.5 GPU-h, serial).
  - **Identity bridge (step 3): BITWISE.** S1 as the stand ckpt, mask 0 (`SN_ID_s18`) = REFS_S1_s18 on every raw key,
    suite / walkstop / field_hard es1 (max |Δa| 0; 235k / 180k / 47.5k stand steps) — `bridge_identity.json`.
  - **GPU swap-live:** S1 as the stand ckpt, mask 1 (`SN_IDM_s18`, walkstop es1): the stand-net action reaches all
    180,000 stand steps, pre-entry rows = REFS — `bridge_swaplive.json`.
  - **Smokes (step 6), 100 its each: PASS** (`smoke_{mask,full,A1}.json`). DAgger: stand-row share 0.492; 1351
    handovers / 1401 exits; DART σ realised 0.1000 / 0.3 within 10 %; β schedule, control share 0.104; env + column
    obs parity; in-process forward parity bitwise; S1 deterministic; reservoir 1.01 M rows = 4.0 GB cap; ckpt strict
    load; terrain levels recorded. 1.54 s / it at 1024 envs (→ ≤ 4500 its fit the 2 GPU-h cap). A1 mode: shares
    0.48 / 0.17 / 0.18 / 0.17; executed rows R-or-spoiler only (0 mismatches); no optimizer; σ 0; statue executes 0;
    chatter σ 0.295–0.300 per channel; stuck-lift reached in 99 % of takeovers; last-row read exact; 485 scored stands /
    100 its.
  - **Smoke checkpoints in stand_net (walkstop es1):** load, estimator = S1's, action on every stand step
    (`smoke_ckpt_{mask,full}.json`); offline trainer→StandNetMode parity CPU bitwise, CPU vs GPU capture 2.1e-6
    (≤ atol 1e-5).
- 2026-10-12 ~03:56, **step 6b (A1) launched:** `SN_A1_s18`, `--mode recover_probe --seed 18 --iterations 700`
  (≈ 0.3 GPU-h; ≈ 3.4k scored stands expected). Rule as PLAN Amendment 1 (control homed ≥ 0.75; each spoiler ≥ 0.8 ×
  control, fall share ≤ control + 0.01); read only from a run whose status is complete. PASS records and notifies;
  step 7 waits for the user's go.
- 2026-10-12 ~04:15, **step 6b (A1) read: verdict STOP — on one clause** (`a1.json`; run status done, 700 its, 1047 s,
  ≈ 0.29 GPU-h; 5154 scored stands [corrected 2026-10-12 from a typo "6154": 2586 + 906 + 838 + 824]; scored on the first draw, scheduled ≥ 8 s, not timeout-cut).

  | group | n | homed | line | stand-fall share | line | falls before takeover |
  |---|---|---|---|---|---|---|
  | control (R from clean handovers) | 2586 | **0.969** | ≥ 0.75 | 0.0039 | — | 0 |
  | statue | 906 | 0.975 | ≥ 0.775 | 0.0044 | ≤ 0.0139 | 4 |
  | chatter | 838 | 0.994 | ≥ 0.775 | 0.0048 | ≤ 0.0139 | 3 |
  | stuck lift | 824 | 0.933 | ≥ 0.775 | **0.0534 FAIL** | ≤ 0.0139 | 28 |

  - **R homes from every student-like state** (homed 0.93–0.99 against the control's 0.97; homed by R-time ≥ 4 s ≥ 0.96
    in every group; trim-band legs at stand end ≈ 0), and the control clears the 0.75 floor in the training env
    (pushes, terrain levels 0–9).
  - **The failing clause is falls under the stuck-lift spoiler:** 44 / 824 — 28 during the spoiler window itself (one
    leg held up with its cam driven to ±0.4 rad for up to 4 s), 16 after takeover, all within 0.82 s of it (none
    later), on terrain levels 1–9 and mostly corner legs. Post-takeover share 16 / 796 = 0.020 vs control 0.004.
    Descriptive reading (not a rule change): R does not add late falls; it cannot catch a robot already tipping when it
    takes over. Statue and chatter fall shares are at the control's level; exit falls are 0 in every group.
  - Per the pre-registered rule this is a **STOP and ask**; nothing further launched.
- 2026-10-12, **USER DECISION at the A1 stop: proceed to the pilot** (accepting that R homes from every student-like
  state and that the stuck-lift fall excess comes from the spoiler window and the first 0.8 s after takeover; G3 still
  gates falls at eval). Go for step 7 only; step 8 (A-full) waits for the user after the pilot.
- 2026-10-12, **step 7 launched: pilot `SN_Amask_s18`** — `--mode dagger --arm mask --seed 18 --iterations 4000`
  (1.54 s / it → ≈ 1.7 GPU-h ≤ the 2 GPU-h cap; β → 0 at it 1333; ckpts every 500; one job, no resume). Early checks,
  notify-only, at it ≈ 1333 (β → 0) and it 2000 (mid-run): student vs concurrent R-slice train-time homed share
  (σ = 0 stands > 4 s, first-draw scoring), window its 1334–2000 at mid-run; student < 0.5 × R → STOP and ask.
  **Judged checkpoint: model_4000.pt.**
- 2026-10-12, **step 7 early checks (notify-only; train-time, σ = 0 stands > 4 s, first-draw scoring):**
  - β → 0 (its 1000–1340): student homed 0.918 (n 2043) vs concurrent R control 0.952 (n 542) — ratio 0.96; stand
    falls student 1.4 % (28), β-R 1.4 % (6 / 425), control 0.4 % (2); exit falls 0.
  - **Mid-run (its 1334–2000), the pre-registered premise check: NO STOP.** Student homed **0.982** (n 4693) vs control
    0.950 (n 1150) — ratio 1.03 (line 0.5). Stand falls student 1.0 % (48) vs control 0.35 % (4); exit falls 2 / 3669
    vs 0 / 899 (`pilot_premise_mid.json`). Watch item for step 9: the student's train-time stand-fall share is ~3× the
    control's (small control count); G3 gates falls at eval.
- 2026-10-12, **step 7 pilot `SN_Amask_s18` DONE** (status done, 4000 its, ≈ 1.65 GPU-h, MSE 0.039 → 0.0038; judged
  checkpoint `runs/SN_Amask_s18/model_4000.pt`, sha256 f7228a99…). End-of-run train-time read (`pilot_premise_final.json`,
  its 1334–4000, σ = 0 stands > 4 s): student homed **0.982** (n 19255) vs concurrent R control 0.945 (n 4443), ratio
  1.04; stand falls 0.80 % vs 0.61 % (the mid-run fall gap closed as control counts grew); exit falls 8 / 15141 vs
  2 / 3489. Last 1000 its: 0.981 vs 0.943. Train-time only — the decisive read is closed-loop G1–G5 at step 9.
  Step 8 (A-full) and step-9 evals wait for the user.
- 2026-10-12 ~12:40, **USER DECISION (before any trained head's gates were read): G3 walkstop stand falls = raw
  count ≤ max(1, Z1's count) per env seed** (300 scheduled stands), not p2_read's rate per reached stands (300 ×
  falls / reached ≤ 1.0, which one fall cutting an episode's later stands fails: 300/298 = 1.0067). Found by the
  observe-reader build on the smoke ckpt SN_SMK_mask_s18; the pilot's walkstop stamps existed but no gate of a trained
  head had been read. References unchanged (R, HOLD, Z1: 0 falls).
- 2026-10-12, **observe reader built** (`scripts/sn_observe_read.py`, 21 CPU tests; verified by independent numpy
  recomputation). Notes for step 9: suite stands start homed, so LIFT / launch numbers come from walkstop only; HOLD
  cam-trim slope is compared with R's slope on the same rows or on the unsaturated band |c| ∈ [0.03, 0.1] (R = −1);
  launch headline = first onset per stand-leg (R's tripod selection flickers LIFT/WAIT).
  - G3 ruling implemented in `scripts/sn_calib.py` (clause `walkstop stand falls (count) es<k> <= max(1, Z1's count)`;
    z2_stand_gates' rate clause kept as a reported note; selftest + a 300/298 case PASS); calibration re-run: still
    **15 / 15 cells land**, 0 mismatches (references all 0 falls).
- 2026-10-12 12:40, **step 9 (first arm): pilot `SN_Amask_s18` (model_4000, mask 1) passes G1–G5 at es1 and es2**
  (`head_SN_Amask_s18.{json,md}`; evals ≈ 0.6 GPU-h, suite / walkstop es1+es2, field_hard es1–4, field_turn es1;
  head-validity rule PASS on every stamp).
  - G1 STILL: steps 0 / 0, taps 0 / 0, clearance 2e-5 m, KS 0.097 / 0.108.
  - G2 HOMED: homed **0.997 / 1.000**, settled 1.000 / 1.000, settle median **2.83 / 2.78 s** (R 2.92 / 2.96; lines
    3.92 / 3.96); post-settle steps 0.44 / 0.46 per s (line 0.5; R 0.11), taps 0.76 / 0.85 per s (line 1; R 0),
    KS 0.15 / 0.16 — **thin margins on post-settle steps / taps**.
  - G3: walkstop stand falls 0 / 0, suite stand falls 0 / 0, exit-handover falls 0 / 0 (of 200 per seed).
  - G4: field_hard **0.7825** (n 400; line 0.7557 = REFS 0.8125 − 0.057; REFS 0.8125, HOLD 0.810, R 0.825);
    field_turn 0.99; stand falls 0 on both fields. Margin 0.027 — watch (within the ±0.04 process-state band).
  - G5: onset 0.12 / 0.12 s (REFS 0.14), 0 stalls; body height within 0.0054 m of REFS.
  - **Observe pass** (`observe_SN_Amask_s18.{json,md}`; walkstop 2064 LIFT onsets): μ_hip on LIFT rows **−0.479**
    (R −0.5; Z2d −0.054), 99.5 % below −0.25; onset launch 99.6 % (first per stand-leg 99.7 %) — the p2r LIFT-onset
    failure is gone. HOLD fit |μ − R| 0.015 cam / 0.015 hip. **The HOLD cam trim is still under-fit like Z2d:**
    within-leg slope −0.032 vs R's −0.52 on the same rows (−0.077 vs −1 on the unsaturated band); leg MR sits in the
    trim band on 60 % of its HOLD rows. Homing comes from re-steps, not the trim — consistent with the post-settle
    steps / taps near their lines.
  - The arm rule waits for `SN_Afull_s18` (training now). Not adoption-grade by itself (one arm, one seed).
- 2026-10-12, **step 8 `SN_Afull_s18` early checks (notify-only):** β → 0 (its 1000–1340): student homed 0.954 vs
  control 0.948 (ratio 1.01), falls 1.2 % vs 0.7 %. **Mid-run (its 1334–2000): NO STOP** — student 0.985 (n 4771) vs
  control 0.950 (n 1095), ratio 1.04; falls 0.96 % vs 0.91 %; exit falls 0 / 3762 (`afull_premise_mid.json`).
- 2026-10-12 ~15:05, **step 8 done + step 9 (second arm): `SN_Afull_s18` (model_4000 sha 1671fd55…, mask 0) FAILS G2
  on one clause** (`head_SN_Afull_s18.{json,md}`, `observe_SN_Afull_s18.{json,md}`, `afull_premise_final.json`).
  - Train-time (its 1334–4000): student homed 0.984 (n 18967) vs control 0.950, falls 0.66 % vs 0.69 %.
  - G1 PASS (steps 0, taps 0, KS 0.096 / 0.103). **G2 FAIL: post-settle taps 1.139 / 1.143 per s (line 1)**; homed
    0.997 / 1.000, settled 1.0 / 1.0, settle median 2.75 / 2.76 s, post-settle steps 0.35 / 0.43. G3 PASS (0 falls,
    0 exit falls). G4 PASS: field_hard **0.815** (line 0.756), field_turn 0.99. G5 PASS (onset 0.12 s, 0 stalls,
    height ≤ 0.0056 m).
  - Observe: μ_hip on LIFT −0.479, launch 98.9 %; HOLD within-leg trim slope −0.029 (R −0.52) — the same trim
    under-fit as A-mask.
- **STEP 9 ARM RULE (pre-registered; decide-record-notify): chosen arm = `A-mask`** — the only arm passing all of
  G1–G5 at es1 and es2 ("one passes → that one"). **A2 not triggered** (an arm passes all gates). Both arms share the
  binding statistic: post-settle taps (A-mask 0.76 / 0.85, A-full 1.14 / 1.14 vs line 1.0), consistent with the cam
  trim under-fit (homing by re-steps rather than trim). Walls: A-mask 0.7825 vs A-full 0.815 / REFS 0.8125 (inside
  the ±0.04 process-state band; replicate reads will show whether it is real).
- **STOP (user go): step 10 (A-mask replicates at s16 / s17) waits for the user.** GPU used so far ≈ 6.0 h.
- 2026-10-12, **USER GO: step 10 (A-mask replicates s16 / s17) + step 11 report, then the adoption stop.** Launched
  `scripts/sn_queue_step10.sh`: `SN_Amask_s16` then `SN_Amask_s17` (same config as the pilot, 4000 its, judged
  model_4000; S1 s18 walks), each with the step-9 eval rows; mid-run premise checks (its 1334–2000) as for the pilot.
- 2026-10-12, **step 10 `SN_Amask_s16` mid-run premise (its 1334–2000): NO STOP** — student homed 0.983 (n 4778) vs
  control 0.942 (n 1107), ratio 1.04; falls 0.98 % vs 1.26 %; exit falls 1 / 3721 (`s16_premise_mid.json`).
- 2026-10-12 ~18:00, **step 10 replicate `SN_Amask_s16` (model_4000) passes G1–G5 at es1 and es2**
  (`head_SN_Amask_s16.{json,md}`, `observe_SN_Amask_s16.{json,md}`): G1 steps / taps 0, KS 0.104 / 0.107; G2 homed
  **1.000 / 1.000**, settled 1.0 / 1.0, settle median 2.82 / 2.66 s, post-settle steps **0.47 / 0.43** (line 0.5),
  taps 0.88 / 0.82 (line 1); G3 0 falls, 0 exit falls; G4 field_hard **0.810**, field_turn 0.97 (one field_turn stand
  fall, inside HOLD + 2 SE); G5 onset 0.12 s, 0 stalls, height ≤ 0.0067 m. Observe: μ_hip on LIFT −0.472, launch
  99.7 %, HOLD within-leg trim slope −0.029 (R −0.50) — the same trim under-fit.
- 2026-10-12, **step 10 `SN_Amask_s17` mid-run premise (its 1334–2000): NO STOP** — student homed 0.981 (n 4873) vs
  control 0.954 (n 1074), ratio 1.03; falls 1.0 % vs 0.56 %; exit falls 2 / 3785 vs 1 / 851 (`s17_premise_mid.json`).
- 2026-10-12 ~20:20, **step 10 replicate `SN_Amask_s17` (model_4000 sha 769cf253…) passes G1–G5 at es1 and es2**
  (`head_SN_Amask_s17.{json,md}`, `observe_SN_Amask_s17.{json,md}`): G1 steps / taps 0, KS 0.108 / 0.105; G2 homed
  **1.000 / 1.000**, settled 1.0 / 1.0, settle median 2.68 / 2.71 s, post-settle steps 0.41 / 0.38, taps 0.74 / 0.75;
  G3 0 falls, 0 exit falls; G4 field_hard **0.805**, field_turn 1.00; G5 onset 0.12 s, 0 stalls, height ≤ 0.0068 m.
  Observe: μ_hip on LIFT −0.491, launch 99.3 %, trim slope −0.028 (R −0.52).
  - End-of-run train-time reads (its 1334–4000; `s16_premise_final.json`, `s17_premise_final.json`): s16 homed 0.980
    vs control 0.941, falls 0.85 % vs 0.79 %; s17 0.979 vs 0.952, falls 0.89 % vs 0.66 %. s16 ckpt sha b17c5b5a….
- **ADOPTION-GRADE (pre-registered: all hard gates at both env seeds on the chosen arm at all three training seeds):
  A-mask MEETS IT — s18, s16, s17 each pass G1–G5 at es1 and es2.** Step 11 (report) next; adoption / bake are the
  user's. GPU used ≈ 10.6 h (plan ≈ 12.3).
- 2026-10-12, **step 11 report written** (`REPORT.md`; opus·xhigh draft, fable·high independent check re-derived
  every gate verdict and the adoption-grade rule — 5 minor wording / attribution fixes applied, no number changed).
  **STOP for the user's adoption / bake decision.**
- 2026-10-13, **user request: a 3-min video, random commands with a random-length stand between each.** Recorder
  built campaign-locally (`scripts/sn_video_schedule.py`, `sn_video_core.py`, `sn_record_video.{py,sh}` — a copy of
  `tools/record_categories.py` sequence mode with stand-net routing through `sn_common`; opus·xhigh build, fable·high
  review, 3 minor fixes; 94 CPU tests). Smoke (20 s) then the full run: seed 20261013, 180 s, flat, 9 commands
  (6 move 0.12–0.27 m/s, 2 turn, 1 move_turn) and 10 stands (2.6–11 s); S1 s18 walks, `SN_Amask_s17` (mask 1) on every
  stand hold (schedule rule). Result (`video_sn_random_s20261013_sanity.json`; mp4 in
  `parkour/logs/videos/sn_stand_network/`, untracked): 0 falls, 0 route mismatches, stand-net share 1.0 on every stand,
  9 / 10 stands homed at their end (the 3.9-s stand ended before homing; settles 1.6–4.1 s), peak RSS 10.1 GB.
  - **Deployment finding (not a gate, recorded for the adoption decision):** the eval's latched stand slot keeps
    "turning" after any facing change until a new facing target or a reset, so on this schedule it reads TURN on 7 of
    the 10 stands (every stand after the first facing change) — under the evaluated switch contract S1 would have
    marched through those stands. The campaign's eval rows never put a stand after a facing change (suite / walkstop /
    field rows reset or walk straight first), so the gates did not see this. A deployed switch should key on an
    explicit stand command; the video routes by the schedule's stand holds (`--stand-rule slot` reproduces the eval's
    latch).
- 2026-10-13, **what the post-settle "taps" / "steps" are physically** (user question after the video; CPU read of the
  walkstop es1+es2 raws with p2s_stand's own settle and step_tap_runs definitions — `scripts/sn_tap_anatomy.py`,
  `tap_anatomy.json`; 600 stands each). A "tap" = a foot's load < 50 N for 1–4 control steps; a "step" = ≥ 5 steps.
  - A-mask s17 taps: 1.2 per s pooled; median duration one control step (0.02 s); median foot lift **0.0 mm**, 83 % lift
    < 1 mm, 97 % < 5 mm, none > 20 mm; the hip command never lifts (0 % below −0.25); R's branch is HOLD on 99.8 %.
    They are load flickers across the 50 N line (six feet are statically indeterminate; the net's small continuous
    corrections shift load), not motion. s18: the same (0.0 mm median, 81 % < 1 mm).
  - A-mask s17 "steps": 0.55 per s; median lift 1.4 mm, median 0.28 s; 68 % fall within 2 s of the settle point (the
    tail of settling: "settled" = cams homed for 0.5 s, the last seat-downs come after it). Visible ones (> 5 mm) after
    2 s past settle: 15 in 600 stands (≈ 1 per 40 stands).
  - R for comparison: taps 0.13 per s (median force at the flicker 48 N — barely crossing the line), steps 0.19 per s.
  - **Implication for the gates:** G2's post-settle taps clause mostly counts sub-millimetre load flicker; A-full's
    G2 failure (1.14 vs 1.0) is therefore a load-flicker difference, not visible tapping. Recorded, no rule changed.
- 2026-10-13, **USER DECISION: "Adopt A-mask s17 as the stand network of record."** Implemented in the omni policy
  of record (workflow: opus·xhigh build, fable·high review, 1 minor fix): `policy/stand_net/model_4000.pt` (sha256
  769cf253…, = `runs/SN_Amask_s17/model_4000.pt`), a new top-level `stand_network:` block in `policy/manifest.yaml`
  (provenance, deployment contract as evaluated incl. the switch-keying caveat, per-seed G1–G5, selection, caveats),
  `heads.stand: stand_net`, README section "Stand network of record", follow-ups `stand_switch_keying`,
  `stand_trim_underfit`, `two_network_export` and a rewritten `stand_march`. `experiments/tools/bundle_policy.py`
  gained optional `stand_network` support (validate / sync / check / render; unchanged output for manifests without
  it); `tests/unit/test_policy_of_record_omni.py` pins it (37 policy tests pass; `--check` OK). The walking head of
  record is unchanged (`current: 6_p2a_field`, S1 s18). code_sha 14301cd2 / eval_sha 1d021625 unchanged.
  REPORT.md §7's trim-slope ranges corrected (A-full's values had been mixed into the A-mask range).
- **Campaign CLOSED — ADOPTED.** Open items carried as policy follow-ups: explicit stand-command keying for any
  deployment, the HOLD cam-trim under-fit, a two-network export / play path.
