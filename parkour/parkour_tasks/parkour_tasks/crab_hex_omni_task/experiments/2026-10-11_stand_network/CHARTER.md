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
  ≈ 0.29 GPU-h; 6154 scored stands; scored on the first draw, scheduled ≥ 8 s, not timeout-cut).

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
