# Stand re-pairing to TDR — pre-registration

**Status: PRE-REGISTERED 2026-11-09 10:28, before any GPU row.** The user adopted TDR and c31000 at 09:55 and asked to
"start the SN/SH re-pairing. Bake only after I approve." (CHARTER §6 09:55). This file fixes the method, the gates and the
stops. Its log (§9) records what ran.

## 1. Goal and scope

- Re-run Task 4's two stand stages with the walker switched from T\* (`policy/f19_8_p3b_payload180/model_25992.pt`, sha
  `03b6009a`) to TDR (`2026-11-07_19-49-46_RSD_TDR_s19/model_28991.pt`, sha `03c3e472`):
  - `sn`, the stand network: DAgger of the scripted re-stance controller R, beside the walker;
  - `sh`, the stand-height network: DAgger of R_h, warm-started from `sn`.
- The recipe is Task 4's (`../2026-10-16_task4_pipeline/pipeline.yaml` 766–827; `tools/pipeline/pipeline.py`
  `ex_dagger_sn` / `ex_dagger_sh` 2438–2581). Only the walker changes (§3).
- The judge is Task 4's own gate code. `pipeline.stand_gate` (D1 for SN, D2 for SH) runs as a library on
  `thresholds_v1_2.json`, with its relaxed G4 and K4 ≤ 3.
- `pipeline.py` itself cannot run here. Its lock pins another code sha, and its 150 GB disk floor fails. So
  `scripts/rsd_stand.py` reproduces the two executors step by step, as `rsd_lib` did for selection.
- **Not in scope:**
  - training the stand nets under rod-speed DR. The trainers refuse the knob, and the F19 recipe has none.
  - standing under payload. The stand nets are trained and judged at 0 kg only (DESIGN 3.6).
  - the bake, which is the user's call after this.

## 2. The recipe (Task 4's, unchanged)

| Step | Command (worktree copies of the scripts) | Rows / output |
|---|---|---|
| REFS | `run_arm.sh --row-eval REFS_RSD_TDR <TDR> <env stamp> <manifest> <scenario> <es>` | walker alone: suite@1,2; walkstop@1,2; field_hard@1–4; field_turn@1 |
| SN train | `sn_dagger.sh --wait-idle SN_RSD_TDR_sn_s19 -- --mode dagger --arm mask --seed 19 --iterations 4000` | judged `model_4000.pt` |
| SN rows | `sn_eval_rows.sh SN_RSD_TDR_sn_s19 <ck> 1 <the nine rows>` | stand net beside TDR, mask 1 |
| SN read | `sn_bridge_read.py --head <SN> --refs REFS_RSD_TDR --live`; `sn_calib.py --head <SN>` | gate D1 |
| SH train | `sh_dagger.sh --wait-idle SH_RSD_TDR_sh_s19 -- --mode dagger --arm mask --seed 19 --iterations 5000 --deadband 0.25 --trim-branch --trim-min 0.0999 --warm-start <SN model_4000.pt> --warm-sha <its sha256>` | judged `model_5000.pt` |
| SH table | `sh_table_test.sh --wait-idle --foreground --controllers stand_net --videos stand_net --stand-ckpt <SH ck> --label <SH> --deadband 0.25` | table stats s1, s2 |
| SH rows | `sh_eval_rows.sh --deadband 0.25 <SH> stand_net manifest armed <ck> 1 hwalkstop@1,2 hraise@1,2`; the same with `<SH>z … zero …` on the nine rows | height rows; K5 zero-command rows |
| SH read | `sh_read.py --spec <spec>`, the spec built by `pipeline.sh_read_spec` from `head_SH_T17_spec.json` with refs REFS_RSD_TDR; `sn_calib.py --head <SH>z` (K5) | gate D2 |

- **Walker overlay:**
  - `CANDIDATES` = Task 4's `F19/candidates.yaml`, `OVERLAY_CHAIN` = `F19_p2a`, and `S1_OVERLAY` = `pipeline.yaml`'s
    p2a overlay, verbatim (27 keys).
  - 0 kg, with no rod-speed DR key.
  - The trainer's composed `KRABBY_*` env (`params/krabby_env.json`) must equal F19's key for key. The one exception is
    the root of `KRABBY_RSI_BANK`'s path: the preset derives it from the checkout, the files are byte-identical, and
    the RSI fraction is 0. (Amended 10:52, §9; F19's dump hashes `9f868e2c…`, the worktree's `38631fe9…`.)
- **Live rules (Task 4's):**
  - The progress watchdog stops a trainer after 30 min without progress.
  - The trainer's own premise read runs once, at iteration ≥ N/2, over [N/3, N/2] (SN 1333:2000; SH 1666:2500). A STOP
    there stops the run.
- **Caps (the launchers' defaults):**
  - trainers 48G + 14400 s;
  - rows 48G + 3600 s each;
  - table jobs 24G + 3600 s.

## 3. The walker swap (the only change)

- **`SN_WALKER_SPEC`.** Every step after REFS gets the walker env: `SN_WALKER_SPEC`, `SN_WALKER_CK`,
  `SN_WALKER_ST` and `P2_REANCHOR=1`, as `pipeline.Queue.walker_spec_env` builds it, with TDR in T\*'s place. The spec:
  - `S1_CKPT` / `S1_SHA256` / `S1_AGENT_YAML`: TDR's checkpoint, its sha and its `params/agent.yaml`.
  - `REFS_ENV_STAMP` = `SN_WALKER_ST`: REFS(TDR)'s own suite@1 stamp.
    - TDR has no Task 4 driver stamp. T\*'s was a 0-kg suite es1 Walk-Play eval.
    - REFS(TDR) suite@1 stands in for it: the same task, scenario, env seed and env, with TDR's checkpoint. The SH table
      test requires exactly that (task Walk-Play, checkpoint = the walker).
  - `CANDIDATES`, `OVERLAY_CHAIN`, `S1_OVERLAY`: as F19's (§2).
  - `REFS_NAME` = `REFS_RSD_TDR`.
- **The REFS rows' own env input** is T\*'s driver stamp (`gait_eval/omni_phases/F19/seed19_X08_p3b_A15pB/suite__omni/seed001/2026-10-22_16-32-02`), exactly as F19's REFS rows used it.
  - `run_arm` reads only its `env_vars`.
  - Those are TDR's own 0-kg suite env key for key: 27 keys, no DR, stress or payload key.
  - Every REFS stamp is checked: Walk-Play, TDR's sha, payload none. Its env must equal the driver's plus the row's
    own manifest env (the field rows add their terrain keys), and also REFS_T4F19's same row (amended 10:52, §9).
- **Names** are new, and each is checked absent before launch:
  - `REFS_RSD_TDR`, `SN_RSD_TDR_sn_s19`, `SH_RSD_TDR_sh_s19` and `SH_RSD_TDR_sh_s19z`;
  - the lazy control `SH_RH_RSD_TDR`.
  - A DAgger run never resumes. A retry trains afresh under `<name>_a2`, on the user's yes.
- **Roots, as F19's: the main checkout.** That is where the readers find the S1-era controls, and the outputs outlive
  the worktree.
  - `P2_OUT` = `parkour/outputs/omni_plan/phase2`: rows and reads.
  - `SN_OUT` = `…/stand_network` and `SH_OUT` = `…/stand_height`: the run dirs.
  - `SH_TABLE_OUT` = `parkour/logs/videos/sh_table`.
  - The `SH_*` variables are set on their own commands only, because `sh_eval_rows.sh` refuses any `SH_*` in its env.
- **Code.** The worktree's copies of the stand, phase-2 and pipeline scripts run at the campaign pin:
  - code `de3772e2`, eval `9fe964d7`, a clean tree, checked before every launch;
  - `PYTHONPATH` = the worktree;
  - every child env starts from `pipeline.clean_env()`.
- **Records:**
  - `results/stand/`: the walker spec, heads, premise reads, CPU reads, gates and lazy reads.
  - logs: `parkour/outputs/omni_plan/rod_speed_dr/stand/` (main checkout).

## 4. Decision rules (Task 4's)

- **SN gate (D1, G1–G5, 35 decisive + 8 WATCH lines).** PASS or WATCH → SH. HALT → stop: lazy controls, or the user's
  call.
- **SH gate (D2, K1–K6, 101 decisive lines).** PASS or WATCH → done. HALT → lazy controls, else stop.
- **Lazy controls (DESIGN 3.6).**
  - The R_h-, R- and HOLD-derived lines sit at their S1-era values. If one fails, its control runs beside the walker.
  - If the control fails the same line, the line re-anchors on the control. That failure is walker-caused, and it is
    decide-record-notify.
  - Otherwise the stage HALTs, and the user decides.
  - **R_h beside TDR:**
    - The rows: `sh_eval_rows.sh SH_RH_RSD_TDR r_h manifest armed - - <the four height rows>` with the walker env, ≈ 0.4 h.
    - When it runs: right after the SH rows, whatever the read says, as reversible work. Both payload walkers needed it
      (F19's p3b and R18's p3a).
    - When it is read: only when a failed line is R_h-derived. The read uses the SH read spec with `roles.R_h.name` set
      to `SH_RH_RSD_TDR`, F19's `sh_lazy_rh_read_spec.json` pattern.
    - R_h-derived means the clause names R_h's value and has no thresholds override. K4 miss ≤ 3 is fixed (D9).
  - **R and HOLD** (corrected 10:52, §9):
    - R covers G2 settle (vs the R probe) ×2 and G3 exit-handover falls ×2. HOLD covers G4 stand-fall rate ×2.
    - They run only if such a line fails. The command is worked out then, from the phase-2 launchers, and the same rule
      applies.
    - G3 walkstop falls ×2 is Z1-derived. DESIGN 3.6 freezes it, so a failure is a HALT.
    - The same eight lines inside D2's K5 are HALTs for the user.
  - **K3** (the table lines) has no lazy path: the lazy spec keeps the S1-era R_h table stats. A K3 failure is a HALT.
- **The body-height lines re-anchor on the R and HOLD controls** (the user's decision, 2026-11-09 11:53, made before
  any stand row was read; §9). These are the four D1 G5 lines `|body height − REFS| ≤ 0.03` and their four K5/G5
  copies in D2. A failed line re-anchors when the R and the HOLD control (`sn_calib`'s own rows) each fail the same
  line against REFS(TDR). The head then passes if it stands within 0.03 m of each control. Otherwise it HALTs.
- **Other fixed lines are never re-anchored.** (As registered at 10:28, the body-height lines were the known risk
  here, a HALT for the user. TDR stands at 1.120 m against the stand controllers' 1.065 m, with F19's gap at
  0.019–0.022. The user's 11:53 decision above replaced that.)
- **A premise STOP** is a HALT for the user. Nothing retries automatically.
- **Reader exit codes are not the gate**, as in Task 4. Beside a non-S1 walker, sh_read exits 4 and its own K5 roll-up
  reads INVALID. The gate reads each clause.

## 5. Pre-flight (CPU) and in-run checks

- **P1, the gate replay.**
  - `stand_gate` on F19's recorded inputs (`F19/stand/sn_calib.json`, `sh_read.json`, `sh_k5_calib.json`) must
    reproduce `F19/gates/sn.json` and `sh.json`.
  - Both verdicts must match: PASS, and HALT on K6 bin 4 es1.
  - So must every decisive and WATCH line: its id, value, line and pass.
- **P2, the reader replay.**
  - The worktree's `sn_calib.py` and `sh_read.py` run on F19's stamps, with `P2_OUT` = main and F19's walker spec. They
    must reproduce every clause (value, line, ok) of `sn_calib.json`, `sh_read.json`, `sh_k5_calib.json` and
    `sh_lazy_rh_read.json`.
  - P1 on the replayed reads must give F19's verdicts.
  - This shows that the worktree readers find the S1-era controls. It also shows that `sn_calib.py`'s root-resolution
    change (64f397e5) moved no value.
- **P3, the lazy-decision replay.** The re-anchor code on F19's HALT and F19's lazy read must re-anchor: the control
  read 0.89 against 0.945, and the head 0.935 passes the re-anchored line.
- **P4, static checks:**
  - TDR's and T\*'s shas;
  - the driver stamp's task, sha and 0 kg;
  - its env equal to TDR's own 0-kg suite env;
  - no DR, stress or payload key;
  - the names absent;
  - `candidates.yaml` and every stand, phase-2 and pipeline script byte-identical to the main checkout's;
  - the thresholds file;
  - a clean tree at the pin;
  - ≥ 30 GB free and the DerivedDataCache ≤ 78 GB.
- **In-run checks:**
  - After REFS: every REFS stamp passes the walker checks (§3).
  - At each trainer's start: `params/krabby_env.json` equals F19's, except the RSI bank path's root (§2). It is
    checked once the train log exists, and a mismatch stops the run.
  - The table dir must name the head; `LATEST` is shared.
  - Every reader must have written fresh JSON.
- **When the pre-flight runs.** It runs beside the REFS rows. Those rows decide nothing, so the GPU is not left idle.
  SN training starts only after P1–P4 pass.

## 6. Stops and guards

- **Before every Isaac launch:**
  - a clean tree at the pin;
  - ≥ 30 GB free and the DerivedDataCache ≤ 78 GB;
  - no other Isaac process running (gate_row.sh's patterns), otherwise it waits.
- **rc ≠ 0, a missing judged checkpoint, or a missing read** → an INFRA stop. The session retries at most once per step.
- **Stops are SIGTERM only.** The trainer pid gets SIGTERM, 120 s, SIGTERM, 120 s. It is never SIGKILLed: Task 4's
  escalation to SIGKILL is not reproduced.
- **One serial queue runner** (`scripts/rsd_stand_queue.sh`), detached, with log sentinels and a watcher. A
  `STOP_QUEUE` file stops it between steps.

## 7. Budget (wall-clock; F19's measured times)

| Part | F19 | Here |
|---|---|---|
| REFS rows | 0.62 h | ≈ 0.6 h |
| SN train + rows + read | 2.43 h | ≈ 2.4 h |
| SH train + table + rows + read | 3.56 h | ≈ 3.6 h |
| Lazy R_h control | 0.42 h | ≈ 0.4 h |
| **Total** | **7.03 h** | **≈ 7.0 h** |

The campaign's GPU use goes from ≈ 27.6 h to ≈ 34.6 h of the 35 h approved, and the user approved the re-pairing at
09:55. Extra lazy controls (R, HOLD) or a retry that would pass 35 h: stop and ask.

## 8. After

- Record the verdicts in CHARTER §6 and in a REPORT addendum, update memory, and notify.
- The bake is the user's call. It rewrites `policy/manifest.yaml` and `tests/unit/test_policy_of_record_omni.py` for
  TDR, c31000 and the new SN / SH, as a separate step.

## 9. Log

- **2026-11-09 10:28:** pre-registered. The REFS(TDR) rows start next, with the pre-flight (CPU) beside them.
- **10:28, REFS(TDR) started** (queue `refs`). The suite@1 row finished at 10:35 and passes every walker check:
  Walk-Play, TDR's sha, 0 kg, the driver's 27 env vars, sidecar code `de3772e2` / eval `9fe964d7`, clean, 1000
  episodes.
- **10:42, pre-flight PASS** (`results/stand/preflight.json`, `ok: true`).
  - P1, the gate replay: the SN and SH gates equal F19's line for line. SN is PASS (35 decisive, 8 WATCH). SH is HALT
    on K6 bin 4 es1 (101 decisive).
  - P2, the reader replay: every clause of `sn_calib`, `sh_read` and `sh_k5_calib` equals F19's. P1 on these re-reads
    is equal too.
  - P3, the lazy-decision replay: re-anchored, as F19's was. The control read 0.89 against the 0.945 line, and the head
    reads 0.935 against the re-anchored line of 0.834.
  - P4, the static checks: all pass. Every stand, phase-2 and pipeline script is byte-identical to the main checkout's.
  - The lazy-read comparison has one recorded exclusion, made after the first run (10:35) and before any gate read.
    - The first run found 24 differences, all in `sh_read`'s own K5 roll-up (`gates.K5`).
    - In F19's hand-run lazy read, the S1-era controls read PASS and the head INVALID there. Here, and in F19's own
      pipeline read, the controls read INVALID and the head PASS.
    - So F19's hand-run read scored K5 against S1's REFS. Neither the gate (it takes K5 from `sn_calib` on the
      head's z rows) nor the re-anchor (K6) reads that roll-up.
    - The comparison now leaves `gates.K5` out of the lazy read. It also checks that the replay's K5 equals F19's
      pipeline read: equal.
  - The pre-flight's name check now also accepts `REFS_RSD_TDR` when every stamp under it is TDR's, because it runs
    beside the `refs` step.
- **10:52, adversarial review** (workflow `rsd-stand-repair-verify`: a fable fidelity checker, an opus runtime
  checker at xhigh and an opus logic checker; read-only). It confirmed the launcher reproduces Task 4's executors:
  commands, rows, seeds, iterations, warm start, read specs and the gate call. It found two blocking bugs in this
  launcher's own added checks, both fixed before any trainer started:
  - **The REFS walker check.** It required every REFS stamp's env to equal the driver's. But the five field rows record
    their manifest's terrain keys on top, as F19's REFS_T4F19 field rows do. It now requires the driver env plus the
    row's own manifest env, equal to REFS_T4F19's same row. The running `refs` process had loaded the old check, so
    it will stop at its end; a re-run of `refs` (CPU only, its rows skipped) writes the walker spec.
  - **The trainer-env check.** It required `params/krabby_env.json` to hash like F19's. But the omni preset sets
    `KRABBY_RSI_BANK` to an absolute path inside the checkout, so the worktree's dump hashes `38631fe9…`, not
    `9f868e2c…`. The check is now key for key against F19's dump, with only the bank path's root allowed to differ.
    The bank files are byte-identical, and `KRABBY_RSI_FRAC` is 0. The check also waits for the trainer's train log,
    so the trainer's own SIGTERM handler is installed before any stop.
  - Minor fixes made:
    - Before the trainer exists, the watchdog clock does not run. A launcher stuck waiting with no trainer is stopped
      through its process group (bash / sleep, no Kit).
    - An unreadable premise read retries. A post-run read is labelled as such, and a STOP from it also stops the step.
    - `sn_train` refuses unless the pre-flight is ok.
    - A partial row step re-runs only its missing rows.
    - K3 is no longer labelled R_h-derived. The lazy spec keeps the S1-era R_h table stats, so a K3 failure has no
      lazy path and is a HALT.
  - Corrections to §4's list of control-derived lines:
    - D1: G2 settle (vs the R probe) ×2 and G3 exit-handover ×2 are R-derived. G4 stand-fall ×2 is HOLD-derived.
      G3 walkstop falls ×2 is Z1-derived, frozen by DESIGN 3.6, so it can only HALT.
    - The same eight lines inside D2's K5 are HALTs for the user.
  - Recorded facts:
    - The row launchers check their STOP_QUEUE files under the worktree's `parkour/outputs/omni_plan/{phase2,
      stand_network,stand_height}`, not the main checkout's.
    - TDR's checkpoint lives only in the worktree; stamps and run records cite it there, and a bake must copy it out.
  - **The body-height line (logic checker, blocking for the plan, not the code).** The four G5 lines
    `|body height − REFS| ≤ 0.03` in D1, and their four K5 copies in D2, will almost surely fail beside TDR.
    - The stand controllers hold about 1.065 m whatever walker stands beside them.
    - Measured 10:50 with `p2s_stand.head_summary` on suite rows, es1 / es2:

      | Rows | Body height (m) |
      |---|---|
      | REFS(TDR) | 1.1201 / 1.1205 |
      | REFS_T4F19 (T\*) | 1.0866 / 1.0856 |
      | F19's SN beside T\* | 1.0655 / 1.0664 |
      | The S1-era R (`P2R_RSTg_s18`) | 1.0648 / 1.0651 |
      | The S1-era HOLD (`P2R_HOLD_s18`) | 1.0648 / 1.0651 |

    - The expected gap is about 0.055 against the 0.03 line. R18's p3a walker passed it by 0.0003.
    - The R and HOLD controls fail the same line against REFS(TDR), so the failure is walker-caused. But DESIGN 3.6's
      re-anchor covers only R_h-, R- and HOLD-derived lines, not this REFS-relative fixed line.
    - The judgement is the user's, asked before any stand row is read.
    - SN training runs meanwhile. It is the same under every option except a redesign of the stand posture.
- **≈ 10:55, the body-height question was put to the user** (push notification and the session's question): re-anchor the
  eight body-height lines on the R / HOLD controls as walker-caused, keep them strict, or redesign the stand posture.
  The current rule stays strict until the user answers.
- **11:04, REFS(TDR) done.** All nine rows ran with rc 0, about 4 min each (10:28–11:04). The `refs` process held
  the old walker check, so it stopped at its end, as the review predicted (rc 4, the five field rows).
- **11:05, `refs` re-run with the fixed check.** The rows were skipped, and all nine stamps pass. `walker_spec.json`
  is written with REFS_ENV_STAMP = `phase2/lr/REFS_RSD_TDR/suite__omni/seed001/2026-11-09_15-28-33`. The queue went
  straight on: **SN training `SN_RSD_TDR_sn_s19` launched at 11:05:06**. The queue runs
  `sn_train sn_rows sn_read sh_train sh_table sh_rows rh_rows sh_read` and stops at the first stop.
- **11:07, SN start-up check passed.** The trainer's `params/krabby_env.json` equals F19's key for key, except the
  worktree's RSI bank path. Its sha is `38631fe9e608d6b1`, as the runtime checker predicted. Training is logging.
- **11:53, THE USER'S DECISION: re-anchor the body-height lines on the controls.** The user answered "Re-anchor on
  controls", whose text reads: "Re-anchor the eight body-height lines (SN G5 and SH K5-G5) on the R and HOLD controls
  as walker-caused, fixed before any stand row is read." At that point no SN or SH stand row existed (checked 11:53);
  SN training was at iteration 1871.
  - The rule, now part of §4, in DESIGN 3.6's form:
    - It covers the four D1 lines G5 `(suite|walkstop) es(1|2) |body height − REFS| ≤ 0.03` and their four K5/G5
      copies in D2.
    - For a failed line, the R and the HOLD control must each fail the same line against REFS(TDR), so the failure is
      walker-caused.
    - The head then passes the re-anchored line if it stands within 0.03 m of each control. A control that passes the
      line leaves the head's failure a HALT.
  - The numbers are the reader's own. `sn_calib` records each row's G5 `heights`, `[row, REFS]` per suite / walkstop
    and seed: the head's in `sn_calib.json` (D1) and `sh_k5_calib.json` (D2), and the R and HOLD rows' in the same
    files. Nothing new runs on the GPU.
  - Every other line keeps its rule. D2's R_h-derived lines still go through the lazy R_h control. The SH stage passes
    only if both rules cover every failed line.
- **12:02, the body-height re-anchor is implemented in `scripts/rsd_stand.py` and tested on CPU.** The SN read and
  the SH read call it after the gate.
  - **The function.** `body_height_reanchor` reads `sn_calib`'s G5 `heights` for the head and for the R and HOLD rows.
    For each failed line it requires:
    - each control fails the same line;
    - each control was read against the head's REFS height;
    - the head stands within the tolerance of each control.
  - **The decision record.** The stage's decision goes to `results/stand/gates/<sid>_reanchor.json`, and SH training
    accepts a re-anchored SN.
  - **The SH order.** The body-height copies are tried first. The R_h-derived lines that remain go through the lazy
    R_h control, via `lazy_decide(…, ids=remaining)`. Every failed line must be covered.
  - **Tests**, each on all four D1 lines, built from F19's recorded `sn_calib.json`:
    - REFS set to TDR's heights (1.1201 / 1.1205 suite) gives 4 / 4 re-anchored. The controls fail, and the head sits
      0.0008 m from them.
    - The head 0.04 m below the controls gives 0 / 4.
    - REFS at 1.08, where the controls pass the line, gives 0 / 4.
    - The D2 K5/G5 copies on F19's `sh_k5_calib.json`, with TDR-like REFS, give 4 / 4.
    - The lazy R_h decision on F19's HALT is unchanged: re-anchored.
- **11:57, SN premise read (Task 4's live rule) over iterations 1333–2000: no STOP.**
  - The ratio is 1.031.
  - Student vs control (`results/stand/sn_premise.json`):

    | | Student | Control |
    |---|---|---|
    | Falls | 53 / 4775 | 10 / 1109 |
    | Exit falls | 3 / 3718 | 0 / 857 |

  - Training continues.
- **13:24, SN gate (D1) read: HALT on the four G5 body-height lines only. All four re-anchor under the user's 11:53
  rule, so SN passes. SH training started at 13:24 from SN's head.**
  - **The head.** `SN_RSD_TDR_sn_s19/model_4000.pt`, sha 6cd82ff5. It trained 11:05–12:47, and its nine stand rows ran
    12:47–13:23.
  - **31 of 35 decisive lines pass**, each at or near F19's SN value (`results/stand/gates/sn.json`; F19's values from
    the pre-flight replay):

    | Line | SN beside TDR | F19's SN | Line |
    |---|---|---|---|
    | G1 suite ks (m/s), es1 / es2 | 0.108 / 0.103 | 0.109 / 0.106 | ≤ 0.5 |
    | G2 walkstop homed, es1 / es2 | 0.997 / 1.000 | 1.000 / 1.000 | ≥ 0.9 |
    | G2 median settle (s), es1 / es2 | 2.70 / 2.80 | 3.05 / 2.92 | ≤ 3.92 / 3.96 |
    | G3 stand and exit-handover falls | 0 on every row | 0 | per line |
    | G4 field_hard completion, es1–4 | 0.925 | 0.910 | ≥ 0.825 |
    | G4 field_turn completion | 1.000 | 1.000 | ≥ 0.900 |
    | G5 walkstop onset median (s) | 0.12 | 0.12 | ≤ 0.44 |

  - **The four body-height lines and their re-anchor** (`gates/sn_reanchor.json`, 4 of 4 covered; heights in m):

    | Line | Head | R | HOLD | REFS(TDR) | REFS − head | Largest head–control gap |
    |---|---|---|---|---|---|---|
    | suite es1 | 1.0656 | 1.0648 | 1.0648 | 1.1201 | 0.0545 | 0.0009 |
    | walkstop es1 | 1.0618 | 1.0619 | 1.0640 | 1.1191 | 0.0573 | 0.0022 |
    | suite es2 | 1.0661 | 1.0651 | 1.0651 | 1.1205 | 0.0544 | 0.0009 |
    | walkstop es2 | 1.0626 | 1.0625 | 1.0641 | 1.1192 | 0.0565 | 0.0015 |

    - Both controls fail each line against REFS(TDR) by about the head's margin, so the failure is walker-caused.
    - The head stands within 0.0022 m of each control, against the rule's 0.03.
    - F19's SN read 0.019–0.021 on the same lines beside T*.
  - **WATCH: 8 lines, no flags.** Post-settle stepping at walkstop rose beside TDR, but stays under its lines:

    | WATCH line, es1 / es2 | SN beside TDR | F19's SN | Line |
    |---|---|---|---|
    | steps/s | 0.41 / 0.45 | 0.35 / 0.35 | 0.5 |
    | taps/s | 0.81 / 0.70 | 0.67 / 0.58 | 1.0 |

    Recorded for the REPORT.
  - **The reader's exit code 4 is its standing output, not a gate.**
    - `sn_calib` exits 4 on 14 calibration-matrix mismatches. Each is "a listed cell cannot be read", because the S1-era
      control stamps carry other code / eval pins than REFS.
    - F19's own read, replayed in the pre-flight, lists the same 14 mismatches and the same 71 INVALID cells, with the
      run names swapped.
    - The R and HOLD heights above are the records the 11:53 rule names. They equal the 10:52 measurement (1.0648 /
      1.0651 suite).
  - **Time.** SN took 2.32 h (11:05–13:24) against F19's 2.43 h.
    - SH passed its start-up env check at 13:26: krabby_env sha 38631fe9, the same as SN's.
    - SH, with its table and rows (≈ 3.6 h), and the lazy R_h control rows (≈ 0.4 h) follow. The SH read is due near
      17:25.
- **17:17, SH gate (D2) read: HALT. Ten of the twelve failed lines re-anchor. K4's two restore-miss lines remain, and
  they are the user's call (§4).** The queue stopped there as designed (rc 4).
  - **The head.** `SH_RSD_TDR_sh_s19/model_5000.pt`, sha 89f4428b, warm-started from SN (sha 6cd82ff5).
    - Trained 13:24–15:44. The premise read at 14:34 (iterations 1666–2500) found no STOP: on random targets the
      student homed 0.98 against the control's 0.93 (`results/stand/sh_premise.json`).
    - The table test ran 15:44–15:50, the height and K5 rows 15:50–16:49, and the lazy R_h control beside TDR
      (`SH_RH_RSD_TDR`, 4 rows) 16:49–17:11. Read at 17:14, decided at 17:17.
  - **The gate.** HALT on 12 of 101 decisive lines (`results/stand/gates/sh.json`); the other 89 pass.
  - **Four K5 body-height lines re-anchor under the user's 11:53 rule** (`gates/sh_reanchor.json`). The head stands at
    1.0621–1.0651 m and REFS(TDR) at 1.119–1.120 m. Both controls fail each line, and the head is within 0.002 m of each.
  - **Six K6 homing lines re-anchor on the lazy R_h control beside TDR** (DESIGN 3.6). The control fails each S1-era line,
    and the head beats the control on all six:

    | K6 line | S1-era line | R_h beside TDR | Head | Re-anchored line |
    |---|---|---|---|---|
    | homed share bin 1, es1 | 0.829 | 0.67 | 0.74 | 0.541 |
    | settled share bin 1, es1 | 0.831 | 0.67 | 0.76 | 0.543 |
    | homed share bin 4, es1 | 0.945 | 0.825 | 0.885 | 0.755 |
    | settled share bin 4, es1 | 0.945 | 0.83 | 0.93 | 0.766 |
    | homed share bin 1, es2 | 0.884 | 0.66 | 0.81 | 0.537 |
    | settled share bin 1, es2 | 0.886 | 0.66 | 0.82 | 0.538 |

    F19's head read 0.98 in bin 1 and 0.935 / 0.955 in bin 4 es1 beside T*; its bin-4 homed es1 also re-anchored on
    its control.
  - **K4 stays a HALT.** K4 counts restore misses among the 800 restore-required hand-backs per env seed. Its line is
    D9's fixed ≤ 3, so it has no lazy path (§4). The head reads 5 (es1) and 6 (es2), against these references:

    | Restore misses per 800, es1 / es2 | Count |
    |---|---|
    | R_h beside S1 (the S1-era control) | 1 / 1 |
    | R_h beside T* (F19's lazy control) | 5 / 5 |
    | R_h beside T* p3a (R18's lazy control) | 7 / 3 |
    | R_h beside TDR (this campaign's control) | 4 / 5 |
    | F19's SH beside T* | 2 / 3 |
    | R18's SH beside T* p3a | 13 / 10 |
    | **This SH beside TDR** | **5 / 6** |

    - The head has 0 exit falls on both seeds, and its onset excess p90 is −1.305 s against a line of ≤ 0.
    - Re-anchored on the control's own count, max(1, R_h's) = 4 / 5, the head would still fail by one miss per seed.
  - **The readers' exit code 4 is their standing output.** `sh_read` and `sh_lazy_rh_read` list 2 mismatches and the K5
    `sn_calib` 14. These are the same as F19's replayed reads, with the run names swapped.
  - **The precedent.** Task 4's CHARTER, 2026-10-21 15:20: R18's SH failed K4 at 13 / 10 against its control's 7 / 3.
    The user chose "accept, disclose", leaving the lines unchanged (DESIGN 3.7 (a)).
  - The CPU check of this read (a five-agent workflow) runs before anything is acted on; its result is the 17:34 entry.
- **17:34, the CPU check of the SH read (workflow `sh-k4-halt-verify`, 5 agents, read-only) confirms every number and
  the rule application, and adds four disclosures.**
  - **Confirmed.**
    - Three agents independently reproduced:
      - the 12 failed lines and the 89 passes;
      - the body-height and K6 re-anchors, to 4 dp;
      - the identity of every head and control stamp: walker TDR 03c3e472, head 89f4428b, warm start 6cd82ff5,
        eval_sha 9fe964d7, clean tree.
    - Two of them recounted K4 from the raw npz timeout flags. Every count in the 17:17 table reproduces.
    - The rules check (fable) found that `rsd_stand.py` covered exactly what §4 and DESIGN 3.6 allow. K4 is
      correctly left to the user, because D9's override fixes its line at 3 whichever R_h the reader uses.
  - **What a miss is.**
    - A K4 miss is a restore that timed out at 5.31 s with the body already at height. In 19 of 20 TDR-era misses
      (head and control), one loaded middle hip rod sat 0.0–0.9 mm over the 8 mm band for the whole post-ramp window.
      The 20th is a lifted leg 7 of 10 rows from completing.
    - It is a static sag at a tolerance edge, not a slow restore, so a longer timeout would not change the count.
    - Head and control share the signature and the place: 11 of the 20 TDR-era misses follow the 1.346 m stand,
      against 1 of 15 beside T* (Fisher p = 0.004).
    - No head miss shares an episode with a control miss.
    - The eras also differ in code / eval pins, so "walker-caused" is the most direct reading, not an isolated one.
  - **Statistics.**
    - Head against its control beside TDR: 11 vs 9 of 1600, two-sided p = 0.82.
    - New head against F19's head: 11 vs 5, p = 0.21.
    - The R18 precedent: 23 vs 10, p = 0.035.
    - Against D9's own calibration (≈ 0.74 misses per seed) 5 / 6 is far outside chance (p 0.012 / 0.0025). So a
      chance-level acceptance can only be argued relative to the control beside TDR, and the reason must say so.
    - At the control's rate (≈ 4.5 per seed), a head as good as R_h passes ≤ 3 with probability 0.34 per seed, 0.12 on
      both seeds.
  - **Disclosures the HALT does not show:**
    - **K5.G2 walkstop homed reads 0.92 / 0.91 against a 0.90 line**, passing by 6 and 3 stands of 300.
      - SN beside the same TDR homes 0.997 / 1.000, F19's SH 0.993 / 0.997, and the S1-era R 0.97 / 0.977.
      - Median settle is 2.86 / 3.00 s, against F19's 2.58 / 2.66.
      - No control covers this line, so it is a weakness of this SH head.
    - **Bin 1 (0.844–0.970 m) homes 0.74 / 0.81.** F19's SH homed 0.98 / 0.98 there, and the TDR control 0.67 / 0.66.
      The line re-anchors, but 19–26 % of low stands do not home: a capability loss to state for hardware.
    - **0 exit falls and onset −1.305 s are common to every controller** (R_h beside S1, T* and TDR; F19's SH). They say
      nothing about this head.
    - **K3 table falls es1 are 0.67 per 100**, against F19's 0.17. This passes, and is under R_h's 0.83.
  - **Context for the decision:**
    - **R18 never shipped.** It was Task 4's fallback, and its SH also carried an open K6 miss. This SH would be the first
      shipped above D9's line; F19's SH passed it at 2 / 3.
    - **Task 4's two texts disagree on eligibility.** DESIGN 3.7 (a) says an accepted run "forfeits 4c", Task 4's
      run-of-record rule. The CHARTER authority rule says it "stays eligible, disclosed", and the user followed the
      CHARTER for R18 and for F19's p3a. Here there is one SH candidate and the bake is the user's call, so the
      difference decides nothing.
    - **A second SH seed would be a new sample**, since a same-seed retrain is a replay.
      - It costs ≈ 3.8 h and passes the 35 GPU-h line, so it needs the user's yes (§7).
      - Its decision rule must be fixed before launch. Adopting it only if it passes is a best-of-two on the gated
        statistic.
    - **An amendment (DESIGN 3.7 (c)), K4 ≤ R_h(TDR) + 2, keeps D9's slack over the teacher.** It reads 6 / 7, which
      the head passes. It is post hoc and would be disclosed as such.
  - **Two record notes:**
    - `sh_reanchor.json`'s K6 `control_note` quotes the control-only SE; the re-anchored lines use the head + control SE.
      The numbers are unaffected.
    - The suite body-height lines cannot tell heads apart: head, R and HOLD read 1.06478 / 1.06514 to 5 dp, as F19's
      did. Only the walkstop lines discriminate.
- **18:10, the user's decision on K4: "Accept and disclose."** Task 4 DESIGN 3.7 (a), as for R18 on 2026-10-21.
  - The SH stage is accepted with its HALT disclosed. The lines stay unchanged: thresholds v1.2, no re-pin. The record
    is `results/stand/gates/sh_decision.json`.
  - **Recorded reason.** The misses are chance-level relative to the scripted teacher beside TDR (11 vs 9 of 1600,
    p = 0.82), not relative to D9's S1-era calibration. They are a static sag at the 8 mm restore band, shared with the
    teacher at the same place.
  - **Disclosed with it, in the REPORT addendum:**
    - K4 5 / 6 against the control's 4 / 5 and D9's 3;
    - the head-specific K5 walkstop homing (0.92 / 0.91 against 0.90);
    - bin-1 homing at 0.74 / 0.81, against F19's 0.98;
    - the ten re-anchored lines;
    - the uninformative exit-fall reading.
  - **The TDR pairing's stand networks:**
    - SH: `SH_RSD_TDR_sh_s19/model_5000.pt` (sha 89f4428b).
    - SN: `SN_RSD_TDR_sn_s19/model_4000.pt` (sha 6cd82ff5), SH's warm-start ancestor.
  - **Next (§8):** the REPORT addendum, then a bake proposal for the user's approval. F19 stays the policy of record in
    `policy/` until the bake.
- **18:41, the REPORT addendum is written (REPORT §13), and the bake plan is proposed (`BAKE_PLAN.md`).**
  - The addendum was drafted by an opus writer and claim-checked by fable against this log, `results/stand/` and the
    17:34 verification. Every number reproduced.
  - The checker's six fixes went in before it was written, including:
    - the "no fall" reading is now labelled uninformative;
    - the ten re-anchored lines are listed among the disclosures;
    - two stale REPORT lines (§1 item 8, §9.2) are updated.
  - The bake plan maps the F19 bake (de81f98a) and lists the defaults the user's approval would cover. Its independent
    check found no blocking problem; its two important fixes are in the plan:
    - the stand block's walker spec keeps `REFS_ENV_STAMP`;
    - the gate-table test asserts the new reads' two K5-control mismatches.
  - Nothing is baked or committed.
