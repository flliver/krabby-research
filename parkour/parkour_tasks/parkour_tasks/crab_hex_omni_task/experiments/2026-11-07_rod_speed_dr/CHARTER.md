# Rod-speed DR in teacher training (rung 3, T01 fix) — CHARTER

- **Pre-registered:** 2026-11-07, before any GPU row of this campaign ran.
- **Plan of record:** the approved plan (`~/.claude/plans/rosy-toasting-dongarra.md`, approved 2026-11-07 after a
  3-pass audit that hit its cap; pass 3's fixes were not re-audited). This CHARTER restates its design where it
  decides a verdict and pins what the plan left to implementation (§4). Where they differ, this CHARTER wins, and
  every change is logged in §6 before the result it affects is read.
- **Source:** `../2026-11-06_dr_stress_sweep/REPORT.md` §3 and §9 decisions 2–3; `../lit-review-domain-randomization.md`
  §13 rung 3.
- **Branch:** `feat/rod-speed-dr` (cut from `feat/dr-stress-sweep` at 44c8165a), worktree
  `.claude/worktrees/dr-stress-sweep`. Nothing is pushed.

## 1. Question and outcome

**Question.** Does per-env rod-speed DR in teacher training remove the sweep's one FRAGILE term? At 90 kg, rod slew
×0.7 fails both seeds for the F19 student (completion 0.58 / 0.66) and for the teacher T* (0.59 / 0.61); ×0.8 splits.
The campaign fine-tunes T* with the DR (TDR), tests it against a no-DR control (TCTL) and against old T* at the
teacher gate G1, and only on a G1 PASS re-distils the depth student and tests it at G2.

**Outcome.** The G1 verdict (with attribution) and, on a PASS, the G2 verdict for a new student candidate, each with
the numbers behind it. Adoption, bake (including SN/SH re-pairing) and push are the user's calls.

## 2. Ground rules

- One Isaac process at a time, under `systemd-run --user --scope` memory caps; Kit is stopped with SIGTERM only.
- Run unattended once approved; every stop point is decide-record-notify. Pause only for push, delete or bake.
- eval_sha stays 9fe964d7: no edits to `experiments/tools/run_phases.py` (the whole file), `eval_crab_hex_gait.py`,
  `gait_eval/*`, `eval_categories.py` or the two omni scenario manifests, and no per-env draw is recorded in
  `run_meta`. code_sha moves once, at the knob commit (§4); everything from the reference arm to G2 runs on that pin.
- The sweep's scripts and stamps are read-only. Never re-point `cells.CKPT` or run `run_sweep.py`; no row of this
  campaign uses a `DRS_` or `DRST_` name.
- Training launches never carry a `KRABBY_STRESS_*` key, and eval rows never carry a `KRABBY_DR_*` key.
- A crashed training run is never auto-resumed (the env step counter, and so the ramp, would restart).

## 3. Design (pre-registered)

### 3.1 The knob
- **`KRABBY_DR_ROD_SPEED=lo:hi[:ramp_steps]`**, default off. Parsed in a new
  `crab_hex_omni_task/config/crab_hex/dr_knobs.py`, called from `omni_knobs.apply_omni_knobs` right after the stress
  knobs; it writes `actions.joint_pos.rod_speed_dr = {lo, hi, ramp_steps}`.
- **Refusals:** refused together with `KRABBY_STRESS_ROD_SPEED` (in the knob and again in the action term).
  0.3 ≤ lo ≤ hi ≤ 2.0 (lo is floored at 0.3, where the current-proxy gate saturates); with a ramp, lo ≤ 1 ≤ hi.
- **The draw:** a new pure-torch `crab_hex_forward_task/mdp/rod_speed_dr.py`, modelled on `stress_push.py`.
  - When armed it builds an (N, 1) `scale` tensor, never rebound, and the action term aliases it as
    `_rod_speed_scale`, so the slew lines (`parkour_actions.py` 368-371 / 400-405) change by zero lines.
  - At each env reset it redraws `s_e ~ U(lo_eff, hi)` for the reset envs only, from a private CPU
    `torch.Generator` seeded from the env seed plus a salt, in env-id order. Hip and knee share `s_e`, which is
    constant within an episode.
  - **Ramp:** lo_eff goes linearly from 1.0 to lo over `ramp_steps` env steps, keyed on `common_step_counter`.
  - **Off means nothing:** no object is built, no draw is made, no attribute is written.
  - It logs `DR/rod_speed_{lo,mean,min,max}` and prints one `[DR]` line on its first reset (sha16 of the draws, the
    module path). The current-proxy normaliser stays the rated constant.
- **Settings:** teacher fine-tune `0.6:1.35:12000` (12000 env steps = 500 iterations at 24 steps per iteration);
  the extension and the student distill `0.6:1.35`, no ramp.

### 3.2 Arms (teacher)
- **TDR:** resume T* `policy/f19_8_p3b_payload180/model_25992.pt` (sha 03b6009a…, `alg_state` counter 27300,
  lr 1.5e-5), seed 19, 256 envs, `Isaac-Crab-Hex-Omni-Teacher-v0`, 3000 iterations → `model_28991`. The env is
  `../2026-10-16_task4_pipeline/F19/envs/p3b.distill.env` plus the knob; everything else reproduces p3b byte for
  byte (env and argv checked against the main checkout's `F19/orchestrator.log:30`).
- **TCTL (lazy control):** identical without the knob. It runs only if TDR's ×0.7 L090 pooled completion beats old
  T* (in-pin) by ≥ m.
- **Margin:** m = max(0.05, 2·SE_diff), Task 4's pooled paired completion margin (pooled over es1 + es2, paired
  episodes).

### 3.3 G1 — the teacher gate
Teacher role, `Isaac-Crab-Hex-Omni-Student-Play-v0`, per-seed Task 4 `p3a.stable90` lines (thresholds v1.2), es1 and
es2. One code pin from the pre-flight to G2.

- **Old-T* reference arm and bridge, before any training (new pin):**
  - Old T* at L000 / L090 / L135 nominal and T01 ×0.7 / ×0.8 L090, es1 + es2 (10 cell-seeds). Plus old T*'s 7
    non-suite L000 selection rows: side22@1, field_hard@1/2, ring_obst@1/2, field_turn@1/2. These in-pin rows are
    G1's only references.
  - **Bridge (a code-path check only):** the six cell-seeds the sweep also ran (L090 nominal, ×0.7, ×0.8 L090, es1 +
    es2) must reproduce its DRST stamps bitwise — all 18 line values equal, as the sweep's CHARTER defines it — on
    the first launch or on one relaunch (the process-state class rule). Otherwise STOP and diagnose on CPU: cfg
    dumps, per-stamp dose, first divergent step. A code-path difference stops the campaign. A diagnosed class flip
    (dumps identical, divergence outside the action path) is recorded, and G1 proceeds on the in-pin references.
- **TDR rows:** the same 10 cell-seeds; band edges ×0.7 / ×1.35 L000 and ×1.35 L090 (es1 + es2); Task 4's 11
  teacher-role selection rows, reused in selection (TDR's L000 nominal cell-seeds are 4 of them — suite@1/2,
  field_light@1/2 — run once and used for both).
- **TCTL rows:** L090 nominal and ×0.7 / ×0.8 L090, es1 + es2. **Closing:** one re-run of TDR L090 nominal es1.
- **Launch checks on every stamp:** the pin and a clean tree; `checkpoint_sha256`, role and task equal to the row's
  policy; no `KRABBY_DR_*` key in the eval env; dose: max |rod_speed| from the raw npz equals s × (0.028, 0.033) m/s
  ±1e-6. Gate names embed the checkpoint's sha8 (`RSDT_<sha8>_<cell>` teacher, `RSDS_<sha8>_<cell>` student), with
  an own output tree (`parkour/outputs/omni_plan/rod_speed_dr`), results dir and pin.
- **Reads:**
  - **S, stress:** every line on both seeds at ×0.7 and ×0.8 L090 and at the three band edges. Carve-out at ×0.7 /
    ×0.8 L090 only: a tracking line also passes if TDR's track_err is no worse than in-pin old T*'s by more than the
    v1.2 per-category, per-seed margin max(R, 3·SE_diff, |drift|·its_past_p2a/5000), with the same its_past_p2a as R
    and SE_diff the env-clustered paired bootstrap of `thresholds.paired_suite_se` on |r − 1| (labelled
    CAPABILITY-LIMITED); R and drift are reused from Task 4's 0-kg Walk-Play calibration, recorded as such. No carve-out at the
    band edges (no reference). Roll and pitch stay absolute.
  - **N, nominal:** per-seed lines at L000 / L090 / L135, except L135 completion. Pooled completion is non-inferior to
    in-pin old T* by m at all three. At L135, a per-seed line (other than completion) that in-pin old T* also fails on
    the same seed is a shared capability limit: recorded in the REPORT, not an N failure.
  - **R, regression (stated once):** the v1.2 paired 0-kg statistic (`thresholds_v1_2.json` zero_kg_statistic, on its
    own row list including the side22 FAST-lost check) at L000, against in-pin old T*, with its_past_p2a = 2999 (5998
    after EXTEND), Task 4's convention (iterations trained past the reference). Its R and drift margins are reused from Task 4's Walk-Play calibration, recorded as such. N covers
    L090.
  - **A, attribution:** TDR − TCTL at ×0.7 pooled ≥ m, both at +3000.
  - **WATCH, not gating:** L135 per-seed completion, survival.stand, per-hand side ratios, and TCTL's L090 nominal
    (shows the no-DR continuation did not collapse).
  - **L135 completion, stated once for both gates:** WATCH per seed and inside es3. At L135, es3 drops its "pooled
    completion ≥ 0.85" clause and "every line" excludes completion. Only the pooled L135 non-inferiority by m is
    decisive.
- **es3:** a cell is near-line when its only failing or knife-edge line is completion at 84–86/100 (one episode either
  side of the 85 line), a survival count within one episode of its line, or a roll_rms / pitch_rms value within 0.003
  of its line on one seed, with every other line passing on both seeds. A near-line cell gets es3 once; it passes iff
  ≥ 2 of 3 seeds pass every line and pooled completion ≥ 0.85. A clear failure of any other line on any seed stands.
  At most 4 es3 cell-seeds per gate; beyond that the verdict is UNRESOLVED.
- **Verdicts:**

  | Verdict | Condition | Action |
  |---|---|---|
  | PASS | S, N, R and A all pass | continue to the distill |
  | FAIL-REGRESSION | N or R fails | stop |
  | UNATTRIBUTED | S, N, R pass; A fails | stop + dose audit |
  | EXTEND (once) | only stress cells fail, on completion or survival (a roll/pitch-only miss is FAIL-STRESS); A ≥ m; TCTL − old < m at ×0.7 (if this alone fails: FAIL-STRESS, noted "no-DR continuation moved ×0.7 by ≥ m"); ×0.7 pooled ≥ old + ½(0.85 − old); TDR − old ≥ m | +3000 iterations without ramp, re-read TDR's rows; final |
  | FAIL-STRESS | anything else | stop; the escalation (p2a → student ≈ 30 GPU-h, or the priv-slot fallback) is the user's |

  After EXTEND, A compares TDR(+6000) with TCTL(+3000) and is reported as length-unmatched; PASS then also needs
  TCTL − old < m at ×0.7.

### 3.4 Student stage (G1 PASS only)
- **Distill:** `experiments/tools/pipeline/distill.sh` with `--checkpoint` = TDR (absolute), an env file =
  `p3b.distill.env` plus `KRABBY_DR_ROD_SPEED=0.6:1.35`, seed 19, 5000 iterations, 192 envs,
  `Isaac-Crab-Hex-Omni-Student-v0`; a wrapper exports the worktree PYTHONPATH. At +10 iterations: the `[KNOB]` and
  `[DR]` lines, teacher sha = TDR, and the live-watch tail read from TDR's run dir; a miss means kill.
- **Selection:** Task 4 `sel` rules unchanged, reproduced from `pipeline.py:2244-2410` (5 candidates prescreened,
  full blocks, `select_student.py`, at most 2 failed blocks). The reproduction sets the level rows to 90 kg
  explicitly (`pipeline.py:2362` would otherwise read 0 kg). Teacher reference rows are G1's TDR stamps. No stress
  cell is used in selection. A HALT stops the campaign.
- **G2:**
  - **c28000 reference arm (in-pin, before the pick's rows):** L000 / L090 / L135 nominal and ×0.7 / ×0.8 L090, es1
    + es2. Its L090 nominal es1 and ×0.7 L090 es1 bridge to the DRS stamps under §3.3's bridge rule.
  - **Pick rows:** ×0.7 / ×0.8 L090, the three band edges and L135 nominal (es1 + es2); L000 / L090 nominal come from
    its selection full block where the rows are identical.
  - **WATCH cells:** T02 ×0.85 L090 and T13 150 kg, against the sweep's stamps (cross-pin); L135 per-seed completion.
  - Closing.
  - **Reads:** S as in G1, with the carve-out against in-pin c28000 (its_past_p2a = 0). N: per-seed lines (L135
    completion excepted; at L135 a line in-pin c28000 also fails on the same seed is a shared capability limit, recorded
    in the REPORT) plus pooled completion non-inferiority to in-pin c28000 by m; a pooled shortfall in
    (m, m + 0.04] is class-ambiguous — relaunch that cell once on both sides (≤ 4 cell-seeds, from the es3 allowance)
    and decide on the pooled four-launch read; a shortfall beyond m + 0.04, or one that survives the relaunch, is a
    FAIL. R (WATCH): the pick's full-block field_hard, ring_obst and field_turn completion and its 0-kg track_err
    against c28000's Task 4 full block (cross-pin); the REPORT flags any row below c28000 − m. Diagnostic: the pick
    vs TDR at ×0.7.
- **Outcome:** PASS goes to the REPORT; FAIL stops with the diagnostic recorded. ONNX export plus CPU parity for the
  pick.

### 3.5 Disk (user decision 2026-11-07)
- Done 2026-11-07 15:55 (§6): the R18 and stepC `raw/` dirs were listed and deleted; 175 GB free; the Kit
  DerivedDataCache is 72 GB against `distill.sh`'s 80 GB.
- Free-space guard in `rsd_gate`, `rsd_train` and `rsd_distill`: before every row or run, stop and notify below 30 GB
  free, or if the Kit DerivedDataCache exceeds 78 GB (a prune is the user's call).

### 3.6 Budget and stop rules
Approved ≈ 35 GPU-h including EXTEND (2026-11-07); ≈ 0.18 GPU-h per cell-seed.

| Phase | GPU-h | Main path | EXTEND path | Stops here when |
|---|---|---|---|---|
| Pre-flight Kit checks (K1, K2) | 0.3 | 0.3 | 0.3 | any mismatch (fix, re-run) |
| Old-T* reference arm + bridge | 2.4 | 2.7 | 2.7 | code-path bridge failure |
| TDR fine-tune | 1.3 | 4.0 | 4.0 | pre-flight check fails in-run |
| TDR G1 rows | 3.5 | 7.5 | 7.5 | DR null: TDR ×0.7 not ≥ old + m |
| TCTL fine-tune + rows (lazy) | 2.4 | 9.9 | 9.9 | FAIL-REGRESSION / UNATTRIBUTED |
| Closing + es3 allowance | 0.9 | 10.8 | 10.8 | UNRESOLVED |
| Extension, once | 4.8 | — | 15.6 | final G1 read |
| Distill | 10.2 | 21.0 | 25.8 | distill check fails |
| Selection: prescreen 25 rows + first full block 10 rows (TDR teacher rows reused) | 2.9 | 23.9 | 28.7 | HALT |
| G2 | 5.6 | 29.5 | 34.3 | G2 FAIL |

Each further selection full block (at most 2, `pipeline.yaml` max_failed_blocks) costs ≈ 0.8 GPU-h and is not in
the totals. If the projected total passes 35 GPU-h (for example the second extra block on the EXTEND path), stop and
ask before launching it.

## 4. Pinned at implementation
(filled in 2026-11-07, before the reference arm runs)
- **Code pin:** commit `1682e2e1` on `feat/rod-speed-dr` (local, not pushed); code_sha `de3772e26a3e…`, eval_sha
  `9fe964d7f46d…` (unchanged); `results/launch.json` written by `rsd_gate.py pin` on the clean tree, 2026-11-07
  17:38. The replay re-ran exact and checks-clean at this commit (17:38).
- **K1** (`results/k1.out`, `results/k1.reread.out`; the dumps stay untracked in `results/k1/`): 9 / 9 PASS.
  - Knob off, five cases (Teacher-v0 × `p3b.distill.env` and × `p3b.train.env`; Student-v0 × `p3b.distill.env`
    minus the 22 student-refused keys; Student-Play × `p3b.env` and × `p3b.field_light.env`): each dump is
    byte-identical to main ab390e6f's after root normalisation.
  - Knob on (`0.6:1.35:12000`, Teacher-v0 and Student-v0): the structural diff touches exactly
    `actions.joint_pos.rod_speed_dr` and `_environ/KRABBY_DR_ROD_SPEED`.
  - DR plus `KRABBY_STRESS_ROD_SPEED=0.7`: refused at cfg build (the exclusivity `ValueError` from
    `dr_knobs.apply_dr_knobs`; no dump written). The run's own check read FAIL on exit code 0, because the dump
    script's `simulation_app.close()` exits 0 after a caught exception. The checker now reads the error and the
    missing dump; the same log re-reads OK (`k1.reread.out`).
- **K2** (`results/k2.out`; the JSON and logs stay untracked in `results/k2/`): 3 / 3 groups, 85 / 85 checks.
  Teacher-v0, 256 envs, seed 7, 600 steps, forced resets at steps 100 / 200 / 400.
  - D0 (off): no DR object and no `[DR]` line; every env's peak rod speed equals rated (0.028 / 0.033 m/s, worst
    relative error 3e-8).
  - D1 (`0.5:1.4`): in all 707 episode segments of ≥ 10 steps the peak equals s × rated (worst relative error
    1.2e-7); s is constant within a segment and changes only at resets (459 changes); every draw equals a CPU replica
    bit for bit (214 calls); KS passes (construction D 0.084, pooled redraws D 0.030).
  - D2 (`0.5:1.0:300`): the construction draws are exactly 1.0; the forced-reset draws lie inside [lo_eff(t), 1] at
    lo_eff 0.832 / 0.665 / 0.5; the logged `DR/rod_speed_lo` equals lo_eff.
- **Replay** (`results/replay.{json,md}`): exact, and every recorded stamp passes the launch checks, dose included.
  - The sweep's `table.json` and `teacher.json` from its 91 DRS / DRST cell-seeds: all 18 line values, flags,
    deltas, level verdicts and the closing match.
  - Task 4's F19 selection: the prescreen of c27000–c30991, the rank c28000 > c29000 > c30991 > c27000 > c30000,
    the c28000 block's margins, gate PASS and the pick (c28000, sha 4fa81f0f).
  - The R wrapper on F19: p3a FAIL (max margin used 2.7098692720929014, the same four fails) at its_past_p2a 2999;
    p3b PASS (0.6619605845918585) at 5998.
- **Predicted in-run lines for TDR** (PF3, PF7, PF9, PF13):
  - `Restored algorithm state: counter 27300, learning_rate 1.5e-05`, `Learning iteration 25992/28992`, and a final
    `model_28991.pt`.
  - `[DR] rod_speed n=256 step=0 bounds=[1, 1.35] (lo=0.6 hi=1.35 ramp_steps=12000) seed=125822752223351
    min/mean/max 1.0051/1.1776/1.3496 sha16=fbb1c962f71c03f3`. The distill (192 envs, no ramp) predicts sha16
    `13857bf91a92b80d`.
- **Launcher tolerances** (implementation choices; the CHARTER said only "follows the ramp" and "about (lo+hi)/2"):
  - The logged `DR/rod_speed_lo` may lag the ramp by ≤ 50 iterations, because the env rewrites its log extras only
    at resets. The low edge is never relaxed.
  - Late window = the last 1000 iterations: |mean − (lo+hi)/2| ≤ 0.03, and min / max within 0.05 of the band edges.
  - `DR/` lines in ≥ 90 % of the logged blocks.
- **Implementation choices in the gate code** (`rsd_lib.IMPLEMENTATION_NOTES`, copied into every gate json):
  - es3 stamps are read with the es1 line ids, after asserting the es1 and es2 line values are identical.
  - At es3 no reference stamp exists, so an es3 tracking line is read absolute, and the L135 shared-limit rule does
    not apply.
  - Near-line means exactly one failing line key over es1 + es2, every failing value inside its knife-edge band, and
    a roll / pitch miss on one seed only.
  - EXTEND's "only stress cells fail, on completion or survival" means every failing S line is
    `field_light_completion` or `survival.<c>`. An h_meas, turn_fc, tracking, roll or pitch miss gives FAIL-STRESS.
  - DR null (TDR ×0.7 − old < m, so TCTL never runs): FAIL-REGRESSION if N or R fails; otherwise UNATTRIBUTED if S
    passes; otherwise FAIL-STRESS noted "DR null", after the dose audit (PF14's `DR/` trace read).
  - A pending es3 or relaunch is resolved by enumeration. The allowance is per gate: the post-EXTEND read also
    counts the pre-EXTEND TDR's es3, and a G2 relaunch is charged 4 cell-seeds per ambiguous cell from the moment it
    is due.
  - Relaunches use the gate name plus `_RL`; the first launch stays the reference.
  - A row whose stamp fails the checks on two launches stops the queue.
  - The shared L000 rows run with the cell composition. The strip is a no-op, which is checked.
  - G2's cross-pin track_err WATCH reuses `select_student.tracking_watch` (+0.05).
  - G2 closing = the pick's L090 nominal es1, launch Z, against its selection L090 es1 row.
  - Stage guards:
    - `g1` / `tctl` / `extend` need `bridge_TSTAR.json` PASS.
    - `g2` needs `sel.json` SELECTED for the pick's sha and `bridge_C28000.json` PASS.
    - `sel` needs `g1.json` PASS for the same TDR sha.
    - `rsd_distill.sh --teacher-policy` needs `g1.json` PASS and the checkpoint's sha = the registered TDR's.
    - A STOP that was diagnosed as a class flip and recorded proceeds only with `--bridge-class-flip`.
- **Carried-over checks from the plan's "left for implementation" list:**
  - K2's ramp case asserts the draws in [lo_eff(t), hi], and runs with hi = 1.0 (D2).
  - The R replay uses Task 4's its_past_p2a 2999 / 5998 (above).
  - After EXTEND, one registry name (TDR6, `model_31990`) feeds `extend`, `read g1 --extended --prior-tdr TDR`,
    `sel --teacher` and `rsd_distill.sh --teacher-policy`. Its launch needs `--expect-counter` and a ramp-free `--dr`
    (PF3 refuses otherwise). PF7 predicts its `[DR]` sha16 at launch.
  - The DR-null dose audit is PF14's read of TDR's `DR/rod_speed_{lo,min,max}` trace.
  - TDR's ×0.7 L090 cell-seeds run first; `tctl_due` requires R PASS, no N failure, the pooled reads OK and the
    trigger.
  - The pooled paired reads use es1 + es2 on both sides only.
  - The distill knob lives in `envs/p3b_dr.distill.env`, because `distill.sh` unsets the parent's KRABBY_*.
  - G1 N at L135: a per-seed line that in-pin old T* also fails on the same seed is a shared capability limit.
  - Unbudgeted costs (an extra selection block, a bridge relaunch) are tracked in §6 against the 35 GPU-h stop.
- **ONNX for the pick (step 12):** `scripts/rsd_onnx.py --checkpoint PICK --out-dir DIR --out JSON`.
  - It exports with the Task 4 exporter (`omni_student_export.export_omni_student`) and runs the sweep's
    `onnx_parity.py` unchanged.
  - Input: the sweep's recorded G0 rollout, copied to `<TP_OUT>/onnx/G0_record.npz` (sha256 `0012cd155099…`).
  - Tolerance 1e-4 on actions and latents, the sweep's rule.
  - Tested on c28000: action error 1.07e-6 and latent error 4.0e-7 over 120 steps / 25 encodes. The generated
    `omni_student_io.json` is identical to the bundled `policy/f19_9_student/`.
  - Negative control (the c28000 graphs against c27000's `.pt`): action error 0.343, FAIL.
- **Student class flip** (c28000 bridge, G2 relaunch, closing; amended 2026-11-08 20:12 by the user's decision, the
  2026-11-07 text is quoted in §6):
  - A mismatch between two stamps of the same row is a class flip when all of these hold:
    - The cfg dumps and the per-stamp dose are identical.
    - The step-0 state and observations are identical in every episode.
    - The earliest difference anywhere in the stamp is in the action alone and at a depth-frame step (every 5th
      step, step 0 included). Its size is not bounded: in the known same-code split S81V0_base / S81V0_base2 it
      reaches 1.45e-5 at step 0.
    - At the c28000 bridge, which compares two code pins, the new stamp also matches another recorded stamp from an
      earlier launch of the same policy and row (same cfg dump, dose and eval seed), bit for bit in every raw-npz
      array of every episode. This shows the new code reproducing a known class.
  - Later differences may take any form. They spread through the policy's depth steps and through the shared
    physics scene, which can move many envs' states at one step (50 envs at step 48 at the c28000 bridge, 11 envs at
    step 165 in the known split).
  - Multi-wave rows (clarified 2026-11-08 23:06 by the user's decision, made after the G2 closing data were read;
    see §6):
    - A suite row runs its 10 waves of 100 envs one after another in one process. The conditions above apply to its
      first wave (wave 0), which starts the process.
    - Each later wave starts from the previous wave's end state. `env.reset()` re-spawns the robots and zeroes the
      action-delay and observation-history buffers. Before the wave's first physics step it does not refresh three
      things: the articulation's applied torque, the contact sensor's force history and the actuator's rod speed.
    - The policy's current-sense observation reads that torque and rod speed at the wave's first step, and the
      harness records torque and foot force before that step (`crab_hex_forward_task/mdp/observations.py` 118–142;
      `eval_crab_hex_gait.py` 1540–1600, 1757–1781). The mechanism was corrected at 23:26; see §6.
    - So any difference in a later wave is a later difference, including one at that wave's own step 0.
  - Anything else is a code-path difference and stops the campaign: a difference in a cfg dump, the dose, or the
    step-0 state or observations; an earliest difference outside the action or at a step that isn't a depth-frame
    step; or, at the bridge, no bitwise match to a recorded stamp.
  - A class flip is recorded in §6. At the bridge, the stage it blocked then proceeds with `--bridge-class-flip`.

## 5. Files
- `scripts/`:
  - `rsd_lib.py`, `rsd_gate.py`: gates.
  - `rsd_train.py`, `rsd_launch.py`, `rsd_distill.sh`, `rsd_distill_check.py`: launchers.
  - `rsd_queue.sh`: the serial queue runner.
  - `rsd_onnx.py`: the ONNX export and parity.
  - `rsd_dump_check.sh`, `rsd_dump_compare.py`, `rsd_smoke.{py,sh}`: K1 / K2.
  - `classflip_check.py`, `wave_diff.py`, `diag_closing_rows.py`: the §4 class tools (copied in 2026-11-08 23:29).
  - `clip_pick.sh`: the pick's headless veto clip (2026-11-09; Task 4's clip recipe).
- `envs/`: composed env files. `results/`: pins, reads, verdicts, logs; `results/closing/`: the closing and diagnostic
  comparison outputs.
- `clip/`: the veto clip mp4s (local only; `*.mp4` is gitignored).
- Stamps (main checkout, untracked): `parkour/outputs/omni_plan/rod_speed_dr/lr/<GATE>/<scenario>/seed00K/<ts>/`.
- Training runs: the worktree's `parkour/logs/rsl_rl/` (gitignored; removing the worktree deletes them).

## 6. Log
- **2026-11-07 15:40, pre-registered** from the approved plan (3-pass audit, cap reached, accepted as is).
- **2026-11-07 15:55, disk freed (§3.5, user decision):** deleted the 283 `raw/` dirs under
  `parkour/outputs/omni_plan/task4/R18/` (136 dirs, 61,200 `.npz`, 63 GB) and `…/task4/stepC/` (147 dirs, 81,124
  `.npz`, 69 GB). The listing held only `.npz` files and no nested dirs, matching the approved 59 + 66 GiB; nothing
  else was touched (summaries, metrics, run_meta stay: R18 8.1 GB, stepC 11 GB left). Free space 51 → 175 GB. Kit
  DerivedDataCache 72 GB, under `distill.sh`'s 80 GB limit.
- **2026-11-07 16:25, amendments after the independent fable check of this CHARTER (before any GPU row):**
  - §3.3 S: the carve-out margin is written out in full: max(R, 3·SE_diff, |drift|·its_past_p2a/5000), the same
    its_past_p2a as R, SE_diff defined, and the Walk-Play reuse noted.
  - §3.3 R: its_past_p2a = 2999 (5998 after EXTEND), Task 4's convention, instead of 3000 / 6000.
  - §3.3 N and §3.4 G2 N: at L135, a per-seed line the in-pin reference also fails on the same seed is a shared
    capability limit (recorded, not an N failure). This restores an item of the approved plan that the first draft
    dropped.
  - §3.5: marked done; the guard also stops if the Kit DerivedDataCache exceeds 78 GB.
  - §3.6: the selection row's 2.9 covers the prescreen and the first full block; each further block (≤ 2) is ≈ 0.8
    GPU-h outside the totals and triggers the stop-and-ask above 35 GPU-h.
  - Pinned for §4: after EXTEND the TDR checkpoint is `model_31990` (28991 + 2999), not the plan's model_31991.
- **2026-11-07 16:25, side effect of the §3.5 deletion:** two local tests in `tests/unit/test_task4_pipeline.py`
  (`test_student_full_block_records_the_tracking_watch`, `test_retarget_tstar_on_a_copy_of_the_recorded_f19_and_r18_runs`)
  replay R18 stamps and now fail on the missing `raw/episode_*.npz`. They read the main checkout's untracked outputs;
  the deletion was not checked for test dependencies. Raised with the user; not fixed here. Also removed two `.pyc`
  files that audit agents wrote into `experiments/tools/pipeline/__pycache__` (a bytecode-guard test failed on them).
- **2026-11-07 17:25, campaign scripts written** (§5), and an independent fable check of them against §3.3–3.6.
  - It traced every rule to code that computes the stated statistic on the stated rows, seeds, references and
    margins.
  - One blocking defect: a half-finished G2 class-ambiguity relaunch read UNRESOLVED, because its own stamps were
    charged against the es3 allowance.
  - Six minor gaps:
    - the es3 allowance restarted after EXTEND;
    - an es3 could launch after the read was already UNRESOLVED;
    - the c28000 bridge was read only after the pick's rows;
    - no resume or ramp guard on the EXTEND launch;
    - the PF13 / PF14 min bound was looser than lo_eff;
    - no G1-PASS guard on the distill wrapper.
  - The test writer found two more: the es3 and g2relaunch guards trusted a stale or foreign JSON, and `sel_row`
    accepted 0 kg.
  - All fixed and tested. Rejected: a file gate tying the row phases to the replay; the replay is step 8's first
    command instead.
  - Added afterwards: `g1` / `tctl` / `extend` refuse to run until `bridge_TSTAR.json` reads PASS (or a recorded
    class flip).
  - Tests: `test_rod_speed_dr_campaign.py` has 74 tests. With the knob's suite and the 12 related suites: 1098
    passed, 10 skipped, 2 failed (the two R18 tests above).
- **2026-11-07 17:26–17:32, K1 and K2** (results in §4): K1 9 / 9 after the refusal checker fix, K2 85 / 85.
- **2026-11-07 17:35, PF12 premise checked:** PF12 compares the new run's params with p3b's
  (`2026-10-22_11-14-00_seed19_X08_p3b_A15pB/params`).
  - Between p3b's training tree (code_sha cbaf6797, last held by commit 149d5cd4) and main ab390e6f, the code
    pathspecs differ in two files only.
    - `observations.py`: the `crab_hex_ref` joint lookup. The A15+B plant takes the same path.
    - `play_stream_video.py`.
  - Neither changes a cfg value, so p3b's yaml stays a valid reference.
  - p3b's agent.yaml also has `max_iterations: 3000`, the same as TDR's.
- **2026-11-07 17:38, code pin** (§4): commit 1682e2e1, code_sha de3772e2, eval_sha 9fe964d7. The replay re-ran
  exact. Both training arms pass PF1–PF8 (`--preflight-only`).
- **2026-11-07 17:40, steps 8–9 queued** as one detached runner (`scripts/rsd_queue.sh refarm bridge train_tdr
  policy_tdr g1_tdr read_g1`). It adds no decision of its own: exit codes, plus the bridge verdict (PASS continues,
  RELAUNCH_DUE relaunches once, anything else stops). Sentinels go to `<TP_OUT>/queue.log`.
- **2026-11-07 17:47, first reference-arm row** (old T* L000 nominal suite es1): 7 min, stamp OK.
  - code_sha = the pin, eval_sha 9fe964d7, clean tree.
  - T*'s sha 03b6009a on the teacher role, no `KRABBY_DR_*` / `KRABBY_STRESS_*` key.
  - Max |rod speed| = rated (0.028 / 0.033 m/s) on all six legs.
- **Disk projection** from the measured stamp sizes (per seed: suite 805 MB, field_light 355, field_hard 336,
  ring_obst 355, field_turn 209, side22 400):
  - Main path ≈ 95 GB; EXTEND path ≈ 116 GB; each extra selection block ≈ +5 GB.
  - 175 GB free against the 30 GB guard.
  - The Kit DerivedDataCache is at 73 GB (guard 78).
- **2026-11-07 18:10, ONNX tool for step 12** written and tested on c28000 (§4).
- **2026-11-07 19:49, step 8 done: old-T* reference arm and bridge.**
  - Reference arm: 27 / 27 rows, every stamp clean on the first try (2 h 09 min).
  - Bridge (`results/bridge_TSTAR.json`): PASS. All six cell-seeds are `bitwise_first`: all 18 line values equal the
    sweep's DRST stamps on the first launch, so no relaunch was needed.
  - The in-pin references therefore equal the sweep's numbers. Old T* L090 completion: nominal 0.98 / 0.96, ×0.8
    0.87 / 0.84, ×0.7 0.59 / 0.61. The ×0.7 pooled value is 0.60, so EXTEND's "old + ½(0.85 − old)" bar is 0.725.
- **2026-11-07 19:50, TDR fine-tune launched** (`RSD_TDR_s19`, unit `rsd-train-RSD_TDR_s19-1794098975.scope`).
  PF1–PF13 PASS:
  - Pin and clean tree; every module resolves under the worktree.
  - T* sha 03b6009a, iteration 25992, counter 27300, lr 1.5e-5.
  - 34 KRABBY_* keys (p3b's 33 plus the DR key). argv = the run_phases template.
  - `[DR]` line: seed 125822752223351, step-0 bounds [1, 1.35], sha16 fbb1c962f71c03f3, as predicted.
  - `Restored algorithm state: counter 27300, learning_rate 1.5e-05`. `params/*.yaml` = p3b's.
  - `Learning iteration 25992/28992`. Free disk 161 GB, DDC 73 GB.
- **2026-11-07 21:02, TDR trained** (1 h 13 min): `2026-11-07_19-49-46_RSD_TDR_s19/model_28991.pt`, sha 03c3e472,
  registered as `TDR` (teacher).
  - PF14 PASS: the final checkpoint exists, and the `DR/rod_speed_lo` trace follows the ramp (1.0 → 0.6 by
    iteration 26492, then flat). The late-window mean is 0.9796 against a midpoint of 0.975; min / max reach
    0.60 / 1.35.
  - Training curve against p3b over the same iteration offsets, from tensorboard (observation, not a gate):

    | Offset (its) | p3b reward / ep. length / crab_failure | TDR reward / ep. length / crab_failure |
    |---|---|---|
    | 100–500 | 58.4 / 1673 / 0.325 | 58.0 / 1638 / 0.308 |
    | 1000–2000 | 58.6 / 1658 / 0.328 | 57.3 / 1665 / 0.310 |
    | 2500–3000 | 58.3 / 1661 / 0.331 | 60.6 / 1771 / 0.223 |

    p3b was flat; TDR improved over its last 1000 iterations while training on the harder plant (mean scale 0.98,
    down to 0.6). Whether the same continuation without DR would also improve is what TCTL measures. LR stayed
    near 1.2e-5 and the action-noise std at 0.80 in both runs.
  - GPU used so far ≈ 3.7 h (K1 / K2 0.3, reference arm 2.15, TDR 1.22), against the main path's 4.0 at this
    point.
- **2026-11-07 21:02, TDR's G1 rows started** (39 rows, ×0.7 L090 first).
- **2026-11-08 00:17, TDR's G1 rows done:** 39 / 39 stamps clean on the first try (3 h 15 min).
- **2026-11-08 00:18, G1 read without TCTL** (`results/g1.json`): **TCTL-DUE.** No line has failed, so the
  lazy control is due.
  - **S:** all five stress cells pass every line on both seeds. The CAPABILITY-LIMITED carve-out was not used.
  - Field_light completion, TDR es1 / es2 against in-pin old T*:

    | Cell | TDR | old T* |
    |---|---|---|
    | ×0.7 L090 | 0.94 / 0.92 | 0.59 / 0.61 |
    | ×0.8 L090 | 0.95 / 0.96 | 0.87 / 0.84 |
    | ×0.7 L000 | 0.99 / 1.00 | — |
    | ×1.35 L000 | 1.00 / 1.00 | — |
    | ×1.35 L090 | 1.00 / 1.00 | — |
    | nominal L000 | 1.00 / 0.99 | 1.00 pooled |
    | nominal L090 | 1.00 / 0.99 | 0.98 / 0.96 |
    | nominal L135 | 0.98 / 0.94 | 0.89 / 0.86 |

  - At ×0.7 L090, TDR's suite tracking error is larger than at nominal (side 0.138 / 0.131, diag 0.078 / 0.080),
    but inside its lines.
  - **N:** the per-seed lines pass at all three loads. Pooled completion against old T*:

    | Load | TDR | old T* | Diff | m |
    |---|---|---|---|---|
    | L000 | 0.995 | 1.000 | −0.005 | 0.05 |
    | L090 | 0.995 | 0.970 | +0.025 | 0.05 |
    | L135 | 0.960 | 0.875 | +0.085 | 0.05 |

  - **R:** PASS on its 26 decisive rows (36 with the 10 WATCH rows), its_past_p2a 2999. The closest row is forward
    track_err on suite es2: TDR's |r − 1| is
    0.039 against old T*'s 0.007, using 63 % of R = 0.051. TDR's 0-kg forward speed ratio has moved, but inside
    the calibrated resolution.
  - **Trigger:** TDR ×0.7 L090 pooled 0.93 against old 0.60. The diff is +0.33; SE_diff is 0.038, so
    m = 0.075 (SE-bound).
  - **Half-gap clause:** holds (0.93 ≥ 0.725). Only the attribution A is still open.
  - es3: 0 / 4 used, none due. The closing row has not been read yet (it runs after TCTL).
  - An adversarial check of this read (independent recompute, rule application, stamp provenance, a refuter) is
    running in parallel with TCTL.
- **2026-11-08 00:20, TCTL queued:** `scripts/rsd_queue.sh train_tctl policy_tctl tctl closing read_g1c`, unit
  `rsd-train-RSD_TCTL_s19-1794115208.scope`. PF1–PF13 PASS:
  - Pin and clean tree; T* sha 03b6009a, counter 27300, lr 1.5e-5.
  - 33 KRABBY_* keys = p3b's, and no DR knob armed.
  - No `[DR]` line; `params/*.yaml` = p3b's; `Learning iteration 25992/28992`.
  - GPU used so far ≈ 7.0 h, against the main path's 7.5 at this point.
- **2026-11-08 00:30, adversarial check of the G1 read: 4 / 4 confirm TCTL-DUE**, read-only workflow
  `g1-read-verify`. The four checkers:
  - Independent recompute from the stamps (opus): every per-seed line, pooled read and R row matches `g1.json` to
    1e-12. The ×0.7 pairs are 114 both / 72 TDR-only / 6 old-only / 8 neither, and m stays in 0.074–0.077 across 20
    bootstrap seeds.
  - Rule application against §3.3 (fable).
  - Provenance of all 66 stamps (sonnet): sha, role, task, pin, env seed, stress and payload keys, no DR key, dose
    over every episode, one stamp per row.
  - Refuter (opus): TDR and old T* differ only in the checkpoint within each cell-seed. No episode was dropped or
    truncated. At ×0.7, TDR's falls drop in exactly old T*'s failing holds (diag 34 → 3, move_turn 19 → 3, back
    17 → 7). It tracks commanded speed better than old T* except sideways, where it is marginally worse (below).
  - **Record-only notes (none changes a verdict):**
    - R's decisive ring_obst completion row has zero margin under v1.2 (1.0 vs 1.0 gives R, 3·SE and drift all 0),
      so a single lost ring_obst episode by a later TDR head would fail R. That is the pre-registered statistic.
    - The closing row has no verdict effect. `closing_read` records match or mismatch with the per-line diff; §3.3
      gives it no rule.
    - TCTL's ×0.8 L090 rows enter no statistic. The REPORT shows them descriptively.
    - The field_light completion line is defined pooled in v1.2. Like the sweep, the read applies it per seed,
      which is stricter. TDR is unaffected (0.94 / 0.92). Old T* ×0.8 es2 (0.84) fails per seed but would pass
      pooled (0.855).
    - TDR also improved at nominal (L135 pooled 0.96 vs 0.875, field_hard 0.96 vs 0.87). Part of the ×0.7 gain may
      be continued training, which is what A tests.
    - Side walking at ×0.7 is the one place TDR is not better than old T*: |r − 1| is 0.138 / 0.131 against
      0.132 / 0.128, inside the 0.25 line.
    - TDR stands about 0.03 m taller in every cell (h_meas.suite L000 1.117 vs 1.089). This is ungated.
    - The L000 and selection stamps carry no runtime payload record. Zero payload is confirmed by body mass 158.76 kg
      (L090 mass − 90).
    - One selection row log (TDR field_turn es1) has two benign Kit plugin-load `[Error]` lines at startup; its stamp
      is complete and clean.
  - For the read with TCTL, the rule checker derived the outcomes: with TDR at 186 / 200, PASS needs TCTL ×0.7
    pooled ≤ 0.88 when the floor binds, about ≤ 0.865 when 2·SE binds. TCTL ≥ 0.885 is UNATTRIBUTED. EXTEND is
    unreachable because S passed.
- **2026-11-08 02:56, G1 PASS** (`rsd_gate.py read g1 --tdr TDR --tctl TCTL`, `results/g1.json`):
  - TCTL trained 00:20–01:37. PF1–PF14 PASS: `model_28991.pt`, sha 72d63d51, no `[DR]` line and no `DR/` keys.
    Its 12 rows ran 01:37–02:44 and TDR's 2 closing rows 02:44–02:55, every stamp ok on the first launch.
  - field_light completion at L090 (es1 / es2):

    | Cell | TDR | TCTL | old T* |
    |---|---|---|---|
    | ×0.7 | 0.94 / 0.92 | 0.71 / 0.58 | 0.59 / 0.61 |
    | ×0.8 | 0.95 / 0.96 | 0.94 / 0.92 | 0.87 / 0.84 |
    | nominal | 1.00 / 0.99 | 1.00 / 1.00 | 0.98 / 0.96 |

  - **A:** TDR − TCTL at ×0.7 pooled is 0.930 − 0.645 = +0.285. SE_diff is 0.038, so m = 0.076 (SE-bound): PASS.
  - TCTL − old T* at ×0.7 is +0.045 against m = 0.088, so the no-DR continuation did not move the cell by m. That
    clause binds only after an EXTEND, and none ran.
  - TCTL's L090 nominal WATCH line: every per-seed line passes on es1 and es2, so the no-DR continuation did not
    collapse.
  - **Closing:** TDR's re-run of L090 nominal es1 (cell `T00_nominal_L090_Z`) reproduces the first launch's 18 line
    values exactly (match).
  - es3 0 / 4 used; not extended; S, N and R as in the 00:18 read.
  - **Descriptive, no statistic:** at ×0.8 the extra 3000 iterations alone took old T*'s 0.87 / 0.84 to
    0.94 / 0.92, about as far as TDR's 0.95 / 0.96. DR's margin over the control is specific to ×0.7, the cell it
    was aimed at.
  - GPU used ≈ 9.6 h (TCTL training 1.28, its rows 1.12, closing 0.19), against the main path's 10.8 at this point
    (the es3 allowance is unused).
- **2026-11-08 02:57, student distill started** (§3.4): `scripts/rsd_distill.sh --name RSD_ST_s19 --checkpoint
  <TDR model_28991.pt> --teacher-policy TDR --wait-idle`, unit `tp-distill-RSD_ST_s19-1794124633.scope`.
  - Env `envs/p3b_dr.distill.env` (sha256 8ba683b4dad7…) = F19 `p3b.distill.env` + `KRABBY_DR_ROD_SPEED=0.6:1.35`.
    Seed 19, iterations 28991 → 33991, 192 envs, Student-v0. Guard: 133 GB free, DerivedDataCache 73 GB.
  - +10-iteration checks D1–D6 PASS at iteration 29007: the knob lines = T4F19_st_a2's + the DR line; the `[DR]`
    line = the predicted first draw; teacher sha 03c3e472 = TDR; first iteration 28991; the live rules are clean.
  - Pace 6.9 s per iteration, ETA ≈ 12:25.
  - A 4-agent adversarial check of the PASS read is running in parallel (read-only).
- **2026-11-08 03:03, step 12 armed:** `scripts/rsd_queue.sh wait_distill sel g2ref bridge_c28k g2 read_g2`,
  detached. These stages are new in the runner:
  - `wait_distill` waits for `TPDISTILL END RSD_ST_s19` and stops unless rc is 0.
  - `sel` runs `rsd_gate.py sel --student-run <the single *_RSD_ST_s19 run dir> --teacher TDR`.
  - `bridge_c28k` applies the old-T* bridge's logic to C28000: one relaunch on RELAUNCH_DUE; anything but PASS
    stops.
  - Then `g2 --policy PICK` and `read g2 --pick PICK --tdr TDR`.
  - Dry runs: `sel` gives 15 rows for the one candidate saved so far, `g2ref` 20 rows, and `g2` 34 pick rows after
    g2ref's 20 (stamped reference rows are skipped).
  - A sel HALT or BUDGET-ASK, a failed bridge, or any non-zero exit stops the queue.
- **2026-11-08 03:10, adversarial check of the PASS read: 4 / 4 confirm**, read-only workflow `g1-pass-verify`:
  - Recompute (opus): every number matches `g1.json` bit for bit.
    - The A pairs are 121 both / 65 TDR-only / 8 TCTL-only / 6 neither. m stays ≤ 0.0763 over 20 bootstrap seeds,
      so the diff is ≈ 7.5 SE.
    - Paired episodes start from identical root positions and terrain.
    - The closing `_Z` stamps are a fresh launch (own dirs and start times), and all 1,100 of their raw npz are
      byte-identical to the first launch's.
  - Rules (fable): PASS is the table's verdict, and A is the CHARTER's statistic. The distill launch matches §3.4,
    and its `[DR]` sha16 13857bf91a92b80d equals §4's prediction.
  - Provenance (sonnet): the 14 new stamps and TCTL's training check out. The dose matches in all 7,700 episode
    files, and TCTL's params differ from TDR's only in the DR block and the run name.
  - Refuter (opus): could not refute.
    - TCTL differs from TDR only in the knob, did not collapse, and is paired episode for episode.
    - TDR's ×0.7 completions do not come from walking slower: it is faster than TCTL on diag, fwd, back and
      move_turn.
  - **Record-only notes (none changes a verdict):**
    - At ×0.7 the extra training alone fixed old T*'s move_turn falls (19 → 1) but not diag / back (34 / 17 →
      30 / 21). TDR fixes those (3 / 7), so A's +0.285 is carried by diag and back.
    - At ×0.8 the no-DR control beats old T* by m on its own (0.930 vs 0.855, m 0.061), and TDR does not beat TCTL
      by m (+0.025). "Attributable to DR" holds at ×0.7 only, the cell A tests.
    - TDR's taller posture (≈ +0.03 m) and halved pitch_rms at ×0.7 (0.028 vs 0.053) come with the DR: TCTL
      stands like old T*. This is ungated; carry it to the student and hardware discussion.
    - In training, TCTL's failure rate plateaued at ≈ 0.28–0.30 (std and entropy creeping up), while TDR's fell to
      ≈ 0.18 under DR.
    - A rests on one training seed per arm (as pre-registered); the gap is ≈ 7.5 eval SEs.
    - The checkpoints' final adaptive LR differs (TDR 1.5e-5, TCTL 1e-5), from the KL schedule. Both restored
      1.5e-5 and averaged the same applied LR.
    - The +10-iteration check printed to stdout only. Its output is now saved in
      `results/runs/RSD_ST_s19.check.out`, with a PASS re-run at iteration 29070.
    - A closing mismatch would have had no verdict effect at G1 (§3.3 gives it no rule). `rsd_distill.sh`'s guard
      does not check that the read included TCTL; that is immaterial, because PASS needs TCTL.
- **2026-11-08 12:33, student distill ended rc 0; selection started.**
  - `TPDISTILL END RSD_ST_s19 rc=0` at 12:32:33, last iteration 33990 (5000 iterations from TDR's 28991),
    wall ≈ 9.6 h (§3.6 budgeted 10.2).
  - The five selection candidates are saved: model_30000 / 31000 / 32000 / 33000 / 33990 in
    `parkour/logs/rsl_rl/crab_hex_omni_student/2026-11-08_02-57-24_RSD_ST_s19`.
  - Training trace, every 500 iterations (depth_actor_loss / crab_failure): 29000 1.41 / 0.10, 30000 0.66 / 0.24,
    31000 0.59 / 0.21, 32000 1.13 / 0.40 (one noisy iteration), 33000 0.60 / 0.18, 33990 0.61 / 0.20.
    The loss flattened by ≈ 30500. This is a record, not a read; selection decides.
  - The queue's `wait_distill` stage passed at 12:33:08 and `sel` started on its first prescreen row.
  - GPU used ≈ 19.2 h of 35. The main path projects ≈ 27.7 (selection 2.9 + G2 5.6 still to run).
- **2026-11-08 15:23, selection: SELECTED c31000; ONNX parity ok.**
  - `rsd_gate.py sel --student-run …/2026-11-08_02-57-24_RSD_ST_s19 --teacher TDR` (35 rows, 12:33–15:23, ≈ 2.83 GPU-h
    against 2.9 budgeted). Results are in `results/sel.json` and `results/sel.rows.json`.
  - The pick is `model_31000.pt`, sha256 155b5981a718ec2c…, registered as policy PICK.
  - Prescreen, all five candidates with 0 misses:

    | Candidate | field_hard | mean_abs_dev |
    |---|---|---|
    | c30000 | 0.905 | 0.022 |
    | c31000 | 0.94 | 0.022 |
    | c32000 | 0.93 | 0.044 |
    | c33000 | 0.92 | 0.029 |
    | c33990 | 0.915 | 0.022 |

    The rank is c31000 > c32000 > c33000 > c33990 > c30000.
  - c31000's full block passed on the first try:
    - All 21 decisive lines pass. Completion paired against TDR: field_hard 0.94 ≥ 0.91, field_light 0.99 ≥ 0.945,
      ring_obst 1.00 ≥ 0.95, field_turn 0.995 ≥ 0.94. The floors and side22 FAST / FAST pass too.
    - WATCH: the level is stable at 90 kg on both seeds (field_light completion 0.995), and the tracking watch
      raised no flags.
    - No further block ran and there was no BUDGET-ASK.
  - ONNX (15:24, CPU): `rsd_onnx.py` wrote `parkour/outputs/omni_plan/rod_speed_dr/onnx/PICK_c31000/`.
    - Files: policy.onnx ac0ea3e2…, depth_latest.onnx 0a04e065…, their .pt twins and omni_student_io.json.
    - Parity on the G0 record: 120 steps and 25 depth encodes. Max |action| error 1.2e-6 and latent error 1.3e-7,
      against a tolerance of 1e-4. Record: `results/onnx_PICK.json`.
  - The queue moved to `g2ref` (c28000's reference arm) at 15:23. Four read-only checkers verify the selection read
    meanwhile (workflow `sel-verify`).
  - GPU used ≈ 22.0 h of 35. The main path projects ≈ 27.6 (G2 5.6 to run).
- **2026-11-08 15:35, adversarial check of the selection: 4 / 4 confirm**, read-only workflow `sel-verify`:
  - Recompute (opus): every sel.json value matches to 1e-9 from the stamps, and every pass flag matches.
    - Covers the prescreen, the rank, all 21 decisive lines, the 35 level lines at 90 kg and the 20 tracking-watch
      rows.
    - The candidate mapping (nearest save to 28991 + offset) and the five checkpoint hashes are right.
  - Rules (fable): ph_sel reproduces pipeline.py ex_select / full_block line for line. The only behavioural
    differences are the two §3.4 sanctions (90 kg explicit; TDR's G1 stamps as the teacher rows) plus the §3.6
    budget check and the g1-PASS guard.
  - Provenance (sonnet): all 35 selection stamps and the 11 TDR reference stamps are clean.
    - Each has one stamp, the right checkpoint, role and task, the pin, a clean tree, and no DR or stress key.
    - Payload 90 kg on the L090 rows and none at L000. Episode counts are right, and the dose is rated.
  - Refuter (opus): could not refute.
    - Student and TDR rows pair episode for episode: the same env vars, env index map, step-0 roots and terrain, and
      segment ids.
    - The student was distilled from TDR with the knob on. Nothing is out of family against F19's c28000 block.
  - **Record-only notes (none changes the pick):**
    - First place rests on 2 of 200 field_hard episodes: c31000 0.94 vs c32000 0.93. Task 4's rank is strictly
      ordered (misses, −field_hard, mean_abs_dev, iteration) with no tolerance band. F19's own first place came down to
      1 episode. The runner-up c32000 has the worst tracking deviation (0.044).
    - All four completion margins bind at the 0.05 floor. field_hard is the tightest: 0.94 against a line of 0.91,
      with TDR at 0.96 and the discordant pairs 8 student-only and 12 TDR-only.
    - At 90 kg the prescreen shows no later candidate as a better 90-kg student. c31000 has 0 / 1000 suite falls and
      100 / 100 field_light; c33990 tracks marginally closer (mean |r − 1| 0.016 vs 0.020).
    - Not reproduced: Task 4's post-selection courtesy, "one headless clip goes to the user for a veto (never a gate)"
      (pipeline.py 2301–2305). It needs the GPU, so it can only run after G2; it is offered in the REPORT.
    - Record keeping:
      - The `rows=40` in the log's `RSD END sel` counts row keys, not stamps (35 distinct).
      - The `at` times in sel.rows.json are when the rows were recorded, not launched.
      - The distill record's launcher_argv lists only `--name` and `--checkpoint`; `--teacher-policy` is mandatory,
        so it was passed.
      - Field stamps carry no complete.json, as in Task 4.
      - §3.6's BUDGET-ASK example (the "second" extra block on the EXTEND path) does not match the code, which would
        ask on the first. It is moot, since there was no EXTEND and no extra block.
- **2026-11-08 17:24, c28000 bridge STOP; the queue stopped before the pick's G2 rows.**
  - `g2ref` finished its 20 rows at 17:13, ≈ 1.85 GPU-h.
  - `bridge_c28k`:
    - T01 ×0.7 L090 es1 matches the sweep bitwise on the first launch (suite and field_light).
    - T00 nominal L090 es1 matches on its suite row (all line values) but not on its field_light row: completion
      0.95 vs the sweep's 0.97, and h_meas, pitch_rms and roll_rms of field_light differ. That holds on the first
      launch and on the one relaunch (`_RL`, 17:24).
    - Verdict STOP (`results/bridge_C28000.json`). §3.3 then asks for a CPU diagnosis.
  - First CPU facts (main thread):
    - The launch and the relaunch are bitwise identical in all 100 episodes.
    - The first launch is bitwise identical, in all 100 episodes, to Task 4's own stamp of the same row
      (`task4/F19/lr/T4F19_sel_c28000_L090/field_light__omni/seed001/2026-10-23_14-41-05`, Task 4's code; completion
      0.95, the Task 4 record).
    - The sweep's stamp (`DRS_T00_nominal_L090_A/field_light__omni/seed001/2026-11-06_22-44-29`) differs from both,
      in all 100 episodes.
    - Metadata, new vs sweep: run_meta differs only in outcomes and start time, tp_row in the env-file path, name and
      write time, and code_state in code_sha and write time.
    - First divergence, new vs sweep: an action-only delta at a multiple of 5 steps, or at step 48 together with the
      physics state.
    - Against §4's student class-flip text taken literally: 3 episodes first differ at step 0 (deltas 1.2e-6,
      1.0e-5, 1.5e-5), and 4 first deltas exceed 1e-5 (max 3.3e-5).
  - A read-only diagnosis workflow (`c28k-bridge-diagnose`) is checking this. It covers the signature against a known
    same-code class split, the cfg and dose, a refuter, and the rule reading. No GPU row runs until it decides.
- **2026-11-08 18:00, c28000 bridge diagnosis: a class flip on the evidence, but §4's pinned rule read literally says
  code-path difference. The decision goes to the user; nothing runs meanwhile.** Read-only workflow
  `c28k-bridge-diagnose`, 4 checkers (opus signature, sonnet cfg + dose, opus refuter, fable rules). All 4
  conclude class flip.
  - Evidence:
    - The new pin reproduces Task 4's 2026-10-23 stamp of the row bit for bit: launch = relaunch = Task 4 in every
      array of all 100 episodes. The suite rows are bitwise identical across all four stamps (1000 / 1000).
    - Cfg: identical. That covers run_meta's env_vars, env_overrides, payload, body masses, checkpoint and student
      block, and the env-file contents.
      - K1 C5 (Student-Play × p3b.field_light.env, knob off) is byte-identical to main at 0 kg.
      - No dump exists at L090 for either pin.
    - Dose: rated in every episode of every stamp.
    - Step 0: the state and observations are identical everywhere.
      - In envs 26, 43, 54 and 81 the step-0 action takes one of exactly two bitwise values. Both come from one
        clean code (S81V0_base vs S81V0_base2, code 6a4e4c6e, 2026-10-27), in the same 4 envs with the same deltas
        (1.0e-6 to 1.45e-5).
      - The new pin and Task 4 hold one value; the sweep's stamp and Task 4's own L000 row hold the other.
    - After step 0:
      - Divergences are action-only at depth-encode steps (period 5) and reach the state about 2 steps later (the
        1-step action delay).
      - At step 48, 50 envs' states diverge at once (cross-env coupling). The known split at 0 kg has its analogue at
        step 165 (11 envs).
    - Code between the pins: 44c8165a → 1682e2e1 is the DR commit only, additions only. Its knob-off path builds
      nothing and runs no draw.
    - Across pins every c28000 stamp falls into one of two classes, and same-class stamps of a cell-seed match bitwise
      (9 groups, no exception).
  - §4 read literally:
    - "A divergence at the first step … is a code-path difference", and the first divergence here is at step 0.
    - "Below 1e-5" fails in 2 of the 4 step-0 envs and in 5 of 100 episodes (max 3.3e-5).
    - The same clauses would call the known same-code split a code-path difference (15 episodes above 1e-5, max
      2.3e-4). §4 mis-transcribed its own reference signature: the memory's "~1e-6 at steps 0/5/10" understates it.
    - Read literally, the campaign stops before G2.
  - Not an executor call: amending a clause that decides a verdict after its data is read is the user's decision (the
    rules checker concurs). Options put to the user:
    - **(a) Amend §4 and run G2 with `--bridge-class-flip`.** Proposed text: "a mismatch is a class flip when the cfg
      and the per-stamp dose are identical, the step-0 state and observations are identical, and the stamp's first
      divergences are action-only deltas at depth-frame steps (step 0 included), with no magnitude bound, as in the
      known same-code split S81V0_base / S81V0_base2. A divergence in the step-0 state or observations, in the dose
      or in the cfg is a code-path difference, and it stops the campaign." G2 then costs ≈ 3.4 GPU-h (34 rows),
      projecting ≈ 27.5 of 35.
    - **(b) The literal rule.** End before G2. G1 PASS, the selection and ONNX stand on in-pin references whose own
      bridge was bitwise clean. The REPORT has no G2 verdict.
  - Record-only for G2 and the REPORT:
    - The in-pin c28000 reference arm is itself a per-launch mix of classes: L090 nominal es2 field_light and the
      ×0.8 L090 rows sit in the other class from the sweep's or Task 4's.
    - Class swings at stressed 90-kg cells reach 0.05–0.07 per seed (×0.8 L090 field_light: es1 0.76 vs 0.83, es2
      0.85 vs 0.80). That exceeds the ±0.04 the G2 class-ambiguity window assumes.
    - The sweep's L090 nominal es1 stamp was already off Task 4's class on 2026-11-06 (0.97 vs the record 0.95,
      inside its 0.04 tolerance). So the cross-pin WATCH reads against the sweep are cross-class too.
    - Cosmetic:
      - bridge_C28000.json's why text says "G1 proceeds".
      - `--bridge-class-flip` leaves no mark in the JSON, so "recorded" means this entry.
    - Unbudgeted: the bridge relaunch, ≈ 0.2 GPU-h. g2ref took ≈ 1.85. GPU used ≈ 24.1 h of 35.
- **2026-11-08 19:49, G2's rows started while the §4 decision is pending.** The user, 2026-11-08: this runs on a
  local GPU, so only wall-clock time matters, and stopping to save GPU time helps only if it also saves wall-clock.
  The GPU had been idle since the 17:24 STOP. The 34 rows do not depend on the decision; only their read does.
  - New queue stage `g2cf` = `rsd_gate.py g2 --policy PICK --bridge-class-flip`, the pre-registered g2 command,
    launched as `scripts/rsd_queue.sh g2cf`, detached. c28000's 20 reference rows skip as valid stamps.
  - Here the flag means "rows started pending the decision". It neither amends §4 nor records a verdict.
  - Nothing is read until the user decides: no `read g2`, no row values. Under (a) the amendment is logged here
    before the read; under (b) the rows stay unread and the REPORT says they ran.
  - Disk: 100 GB free; the rows add ≈ 20 GB (≈ 2.3 GB per cell).
- **2026-11-08 20:12, §4's student class-flip rule amended: the user's decision, made after the c28000 bridge data
  were read.** The user chose option (a) of the 18:00 entry and approved the text shown to them, which added a
  bridge bitwise-match condition to the 18:00 proposal. Writing it in, the executor found one more mis-stated
  condition and corrected it (below); the user is told with this amendment.
  - The 2026-11-07 text, verbatim: "a mismatch is a class flip when the cfg dumps and the per-stamp dose are
    identical and the first divergence is an action delta below 1e-5 at a depth-frame step (the known process-state
    signature). A divergence at the first step, in the dose or in a cfg dump is a code-path difference, and it stops
    the campaign."
  - What changed, and why:
    - Step 0 counts as a depth-frame step. The old "first step" clause contradicted the recorded signature, which
      lists step 0.
    - No size bound. At step 0 this split and the known same-code split both reach 1.45e-5, above the old 1e-5.
    - The action-only condition applies to the stamp's earliest difference, not to each episode's first difference.
      The per-episode version shown to the user (and the 18:00 proposal, read per episode) fails this case: some
      episodes first differ at step 48, in the physics state, when 50 envs' states move at once through the shared
      scene (17:24 and 18:00 entries). The known split does the same at step 165.
    - New: identical step-0 state and observations, stated outright now that step 0 is allowed.
    - New, at the bridge only: a bitwise match to another recorded stamp of the same row. A code change that only
      nudged the depth network's arithmetic would show the same signature; reproducing an older stamp exactly is
      something it cannot do. G2's relaunch and closing compare launches under one pin, whose launch checks already
      exclude a code change.
  - Applied to the c28000 bridge STOP (`results/bridge_C28000.json`; T00 nominal L090 es1, field_light):
    - Cfg dumps and dose: identical in every episode of the four stamps (launch, relaunch, sweep, Task 4).
    - Step-0 state and observations: identical.
    - Earliest difference: step 0, action only, in envs 26, 43, 54 and 81 (1.0e-6 to 1.45e-5), the known split's
      values.
    - The launch and the relaunch equal Task 4's 2026-10-23 stamp in every raw-npz array of all 100 episodes.
    - **Class flip**, recorded here. T01 ×0.7 L090 es1 matched the sweep bitwise on its first launch and needs no
      rule.
  - Correction to the 17:24 entry: 4 episodes first differ at step 0, not 3, and 5 first deltas exceed 1e-5, not 4
    (the 18:00 count).
  - Consequence: G2 proceeds under `--bridge-class-flip`. Its rows have run since 19:49 (stage `g2cf`), and none has
    been read. `read g2 --pick PICK --tdr TDR` follows when they finish. The G2 closing and any relaunch are judged
    by the amended rule.
- **2026-11-08 20:18, the user confirmed the corrected wording** of the 20:12 amendment ("Keep it"): the action-only,
  depth-frame condition applies to the stamp's earliest difference. G2 is read under it when its rows finish.
- **2026-11-08 22:55, G2 read: PASS on every gated line. The closing's suite row waits on the user's reading of §4
  for multi-wave rows.**
  - Stage `g2cf` ended rc 0 at 22:53: 34 new rows, each ok on its first launch; c28000's 20 reference rows skipped
    as valid stamps; no rerun, no stop. Stage `read_g2` (`read g2 --pick PICK --tdr TDR`) wrote
    `results/g2.{json,md}`: verdict PASS, no notes.
  - S: all five cells pass on es1 and es2, no carve-out used. Field_light completion: ×0.7 L090 0.92 / 0.95, ×0.8
    L090 0.97 / 0.96, ×0.7 L000 1.00 / 0.99, ×1.35 L000 1.00 / 0.99, ×1.35 L090 1.00 / 1.00. Every survival line 1.0.
  - N: every per-seed line passes (L135 completion is WATCH: 0.96 / 0.99). Pooled paired completion against in-pin
    c28000: L000 0.990 vs 0.995 (−0.005, m 0.050), L090 0.995 vs 0.950 (+0.045, m 0.050), L135 0.975 vs 0.875
    (+0.100, m 0.051). No relaunch due; es3 0 / 4.
  - WATCH cells against the sweep's c28000 (cross-pin): T02 ×0.85 effort 0.99 / 0.96 vs 0.87 / 0.85; T13 150 kg
    0.93 / 0.96 vs 0.86 / 0.73 (c28000's es2 fails completion; the pick passes every line).
  - R WATCH against c28000's Task 4 full block: field_hard 0.94 vs 0.89, ring_obst 1.00 vs 1.00, field_turn 0.995 vs
    1.00; no completion or tracking flag. Diagnostic: the pick vs TDR at ×0.7 L090, pooled 0.935 vs 0.930.
  - Closing (`RSDS_155b5981_T00_nominal_L090_Z` vs `RSDSEL_155b5981_L090`, es1): mismatch on both rows. Same pin,
    identical run_meta (env vars included) and dose in every episode.
    - Field_light: a class flip under §4. Step-0 state identical; earliest difference at step 0, action only (envs
      5, 7 and 30, ≤ 6.1e-6).
    - Suite: §4 as written gives a code-path difference. The row runs 10 waves of 100 envs in one process. Wave 0
      meets every condition: earliest difference at step 0, action only; step-0 state bit-equal. In waves 1–9 all
      900 episodes differ at their own step 0 (actions, applied torque, foot force), with every pose array
      bit-equal.
    - Why: before each later wave the harness re-seeds, clears the policy's GRU state and latent, and calls
      `env.reset()`. It keeps the step counter that drives the action delay and the 5-step observation refresh
      (`eval_crab_hex_gait.py` 1540–1600, 1815–1825). Wave 0's divergence therefore reaches each later wave's step
      0. No suite row whose wave 0 flips can meet the "every episode" clause.
    - Not decided here. How §4 applies to multi-wave rows is the user's call, as the 20:12 amendment was; asked
      23:05. Until then G2 stands as PASS on its rows, with the closing pending.
  - 23:02, diagnostic relaunch, read by no gate: `RSDS_155b5981_T00_nominal_L090_Z_D1` and
    `RSDSEL_155b5981_L090_D1`, es1, suite + field_light, through rsd_gate's own Runner (driver
    `diag_closing_rows.py`). It shows whether each row shape reproduces a recorded stamp bit for bit (≈ 0.4 GPU-h).
    The class tools `classflip_check.py` and `wave_diff.py` and this driver go into `scripts/` with the record
    commit.
  - The adversarial check of the read (four checkers, workflow `g2-verify`) started 23:04.
  - GPU used ≈ 27.2 h of 35 (24.1 before G2, its rows 3.1), plus ≈ 0.4 for the diagnostic.
- **2026-11-08 23:06, §4 clarified for multi-wave rows: the user's decision, made after the G2 closing data were
  read.** The user answered "1" to the 23:05 question.
  - The two options offered:
    - (1) Judge wave 0: test §4's conditions on wave 0, the process start. Later waves start from the previous
      wave's end state and count as later differences. The suite closing is recorded as a class flip, G2 PASS
      stands, and the REPORT follows.
    - (2) Read §4 literally: the suite closing is a code-path difference, and the campaign stops.
  - Text added to §4 (Student class flip): the multi-wave bullet. Nothing else in §4 changed.
  - Applied to the G2 closing (22:55 entry):
    - Field_light: class flip, as before.
    - Suite: wave 0 meets every condition. Cfg, dose and step-0 state are identical. The earliest difference is at
      step 0, in the action alone, in envs 0, 4, 15, 35, 36, 58 and 75 (≤ 1.8e-5). Waves 1–9 are later
      differences.
    - **Class flip**, recorded here on the executor's check.
  - Still to come: the four-checker verify (started 23:04) and the diagnostic relaunch (started 23:02) are logged
    when they finish. If either contradicts these facts, the user hears before the REPORT.
  - Consequence: G2 PASS stands. Step 13 (REPORT) follows the verify.
- **2026-11-08 23:26: G2 read verified (4 / 4 confirm); the diagnostic relaunch reproduces the closing's stamp; the
  §4 multi-wave mechanism corrected.**
  - `g2-verify` (run `wf_752c320a-14a`, 23:04–23:23) had four checkers: recompute (opus), rules (fable),
    provenance (sonnet) and refuter (opus). All four confirm.
    - Every G2 number recomputes from the stamps within 1e-9. No carve-out, es3 or relaunch was due. No pick line
      sits within one episode (or 0.003 on roll / pitch) of its threshold. Every read stamp passes the launch checks.
    - Record-only:
      - `g2.json` carries no closing verdict, since the closing has no effect on `g2_decide`. Its class-flip
        verdict is in this log (23:06).
      - The pooled SE moves in the 4th decimal with the bootstrap seed. This has no effect: m binds at the 0.05
        floor at L000 / L090, and L135 clears it by +0.049.
      - The row logs capture no Kit stderr.
  - Two corrections to the 22:55 and 23:06 entries:
    - In waves 1–9 of the suite closing, applied torque and foot force differ at step 0 in all 900 episodes. The
      action differs in only 271 of them (31 / 21 / 23 / 40 / 26 / 21 / 41 / 37 / 31 per wave, up to 1.04). The
      tools print the union of keys per wave.
    - The carrier is not the step counter:
      - `env.reset()` zeroes the action delay and the observation history. The 775-step wave is a multiple of 5, so
        every wave's step 0 is a depth-frame step in both launches.
      - What carries over is the articulation's applied torque, the contact sensor's force history and the
        actuator's rod speed, which `env.reset()` leaves at the previous wave's end values.
      - The policy's current-sense observation reads that torque and rod speed (`observations.py` 118–142). The
        harness records torque and foot force before the step.
      - §4's multi-wave bullet now says so. The rule itself is unchanged.
  - More evidence for the class reading (refuter):
    - At one eval seed, the pick's step-0 state is identical across its cells. Its step-0 actions in the class envs
      take exactly two bitwise values over 11 launches: field_light es1 (envs 5, 7, 30) 8 / 3, suite es1 9 / 2.
      c28000's 16 stamps show the same two-value structure in the known split's envs 26, 43, 54 and 81.
    - c28000's own cross-class suite pairs show the same multi-wave signature: in-pin against the sweep at ×0.8
      L090 es1 and L000 es2, and against Task 4 at L135 es2. Pairs whose wave 0 is bit-equal are bit-equal in all
      10 waves (c28000's L090 es1 launch and its `_RL`). Read literally, the 20:12 clause would have made every
      cross-class suite pair a code-path stop.
  - Diagnostic relaunch (23:02–23:24, 4 rows, each ok on its first try):
    - `RSDS_155b5981_T00_nominal_L090_Z_D1` (the cell row) and `RSDSEL_155b5981_L090_D1` (the selection row) both
      equal Z bit for bit: suite in all 1000 episodes, field_light in all 100.
    - So the row repeats exactly, both row shapes land in the same class, and the closing's reference
      (`RSDSEL_155b5981_L090` es1, launched 13:14 during the selection) is the other of the two classes. The class
      does not follow the row shape.
  - Effect on G2: the verdict is unchanged.
    - The pick's L090 nominal es1 rows in N are the minority class. In Z's class its field_light completion is
      0.99, not 1.00, and its move_turn survival 0.995, not 1.0.
    - The pooled L090 read would then be +0.040 rather than +0.045 (m 0.050). Across every class observed for
      either policy it lies between +0.03 and +0.045 (refuter).
  - GPU used ≈ 27.6 h of 35; the diagnostic took 0.36 h.
- **2026-11-08 23:47, step 13: REPORT written** (`REPORT.md`, ≈ 14,900 words).
  - How it was made:
    - The pre-draft was written without G2's data (workflow `rsd-report-draft`, 20:13) and critiqued (8 issues).
    - A final writer filled in G2 and applied the 8 fixes (opus xhigh, `rsd-report-final-write`, 23:10–23:27).
    - The executor filled in the diagnostic and verify outcomes and corrected the closing's mechanism and action
      count to match the 23:26 entry.
    - Then a claim-by-claim check ran (`rsd-report-final-check`, 23:29–23:45): two fable claim checkers by section
      range, an opus completeness critic, and an opus fixer that returned exact edits.
  - The check raised 21 issues, all minor. None changed a verdict or a G2 number. 25 edits were applied, with three
    small adjustments by the executor.
    - The commonest issues were stale "running" text about the diagnostic relaunch and row counts that left out its
      4 rows: 175 rows in all, 171 campaign rows plus 4 diagnostic.
    - The commit list was corrected: `1682e2e1` already holds the cell env files, `p3b_dr.distill.env`, `k*.out`
      and the first versions of `policies.json` and `replay.*`.
    - Leftover `[missing:]` brackets were removed.
  - New files (untracked): `results/closing/` holds the closing and diagnostic comparison outputs
    (`closing_g2_{field_light,suite,suite_waves}.out`, `diag_compare.out`). `scripts/` holds the three class tools
    (`classflip_check.py`, `wave_diff.py`, `diag_closing_rows.py`, copied 23:29).
  - Correction to the 15:35 entry: Task 4's veto-clip text ("never a gate") is at `pipeline.py` 2315–2317, not
    2301–2305.
  - Nothing is committed or pushed. The REPORT's §11 lists the user's decisions: accept G2, the headless veto clip,
    adoption and bake, the local commit, the R18 test fix, protecting the run dirs, a replication seed, the class
    window for the next gate, the next rungs, and push.
- **2026-11-09 09:51, the user's decisions 4 and 6: veto clip recorded, record committed locally.** The user answered
  "Yes, both" (received ≈ 09:43) to the 23:47 closing question. That means: run the headless veto clip now, and commit
  the record locally with REPORT §11 decision 6's file list, no push. Adoption waits for the user's look at the clip.
  - Clip (decision 4; a look, not a gate): `scripts/clip_pick.sh` runs Task 4's recipe
    (`../2026-10-16_task4_pipeline/step11/clip_F19.sh`) with this worktree's code on PYTHONPATH.
    - Settings: `play_stream_video.py`, Student-Play, the pick c31000 (sha `155b5981`), 1 env, 1500 steps.
    - Env: the wall field env `F19/envs/p3b.field_hard.env` (0 kg; no DR, stress or payload key), with every parent
      KRABBY_* unset.
    - Cap: `systemd-run --user --scope -p MemoryMax=40G -p MemorySwapMax=0` and a 2400-s timeout.
    - Ran 09:46:47–09:48:30 (103 s, rc 0), the only Isaac process, on an idle GPU.
    - The log (`parkour/outputs/omni_plan/rod_speed_dr/clip_pick.log`, main checkout) shows the worktree's code paths,
      the checkpoint load and `[VIDEO] 1500/1500 frames`. At exit the viewport camera controller's `__del__` raised a
      harmless AttributeError.
    - Output: `clip/student_c31000_field_hard.mp4` (1280 × 720, 50 fps, 30.0 s) and `_small.mp4` (960 × 540). Both
      are local only (`*.mp4` is gitignored).
    - Both were sent to the user, with F19's c28000 clip for comparison. The terrain draw differs between the two
      clips. In frames sampled every 2 s, the robot stays upright throughout.
    - Side effect, as play.py: `exported_deploy/` (JIT + ONNX) was written beside the checkpoint, in the worktree's
      gitignored run dir. The campaign's ONNX of record stays `parkour/outputs/omni_plan/rod_speed_dr/onnx/PICK_c31000/`.
  - Record commit (decision 6): the REPORT §11 list plus `scripts/clip_pick.sh` and this entry, 32 files. Left out:
    `results/rsd_gate.lock`, the sweep's `run_sweep.lock`, and the `k1/` and `k2/` dumps and logs. Not pushed.
  - Edits:
    - REPORT updated for both: the status line, the header's Commits bullet, §1 item 8, §5.5, §10, §11 decisions
      4–6 and §12.
    - §5 lists the class tools, `clip_pick.sh`, `results/closing/` and `clip/`.
  - GPU used ≈ 27.6 h of 35 (the clip ≈ 0.03 h).
  - Still the user's calls: accepting G2 (decision 3), adoption (5), the R18 test fix (7), protecting the runs (8), a
    replication seed (9), the next gate's class window (10), the next rungs (11) and push (12).
- **2026-11-09 09:55, decision 5: the user ADOPTED TDR and c31000.** The user answered "Yes, adopt" to "Adopt the new
  teacher and student, then re-pair the stand network?". The button's text: "Adopt TDR and c31000, then start the
  SN/SH re-pairing. Bake only after I approve."
  - TDR (`model_28991.pt`, sha `03c3e472`) and c31000 (`model_31000.pt`, sha `155b5981`) are the next omni teacher
    and student. Adopting the pick takes G2 PASS as accepted (decision 3).
  - F19 stays the policy of record in `crab_hex_omni_task/policy/` (`manifest.yaml`) until a bake, which only the
    user can approve.
  - Next: re-pair the stand networks (Task 4's stages `sn` and `sh`, walker T* → TDR). The re-pairing will be
    pre-registered here, as a new section, before its first GPU row. Started 09:57 by mapping Task 4's stand stages,
    their gates and the F19 record (workflow `rsd-stand-repair-map`, read-only).
- **2026-11-09 10:28, the stand re-pairing is pre-registered in `STAND_REPAIR.md`, before any GPU row.** It is a
  separate file rather than a CHARTER section, so this log stays the CHARTER's last section; it has its own log (§9).
  - Method: Task 4's `sn` and `sh` recipe with the walker T* → TDR.
    - Seed 19; 4000 / 5000 iterations; the F19_p2a overlay; 0 kg; no DR key.
    - The rows, the table test and the readers are Task 4's.
    - Judged by Task 4's own `pipeline.stand_gate` (D1 / D2, thresholds v1.2), loaded as a library by
      `scripts/rsd_stand.py`; `pipeline.py` itself cannot run here (its lock pin, its 150 GB disk floor).
  - The walker swap:
    - REFS(TDR) rows under a new name; REFS(TDR) suite@1 stands in for the driver stamp TDR does not have.
    - New names for every head.
    - Outputs in the main checkout beside F19's, where the readers find the S1-era controls.
  - Decisions as DESIGN 3.6:
    - The lazy R_h control beside TDR runs up front and is read only if an R_h-derived line fails.
    - A re-anchor is decide-record-notify; any other HALT is the user's call.
    - Known risk: the fixed G5 body-height line (0.03; F19 read 0.019–0.022; TDR walks ≈ 0.03 m taller).
  - Pre-flight (CPU) runs beside the REFS rows: F19's gates, reads and re-anchor replayed through this code. SN training
    starts only after it passes.
  - Budget ≈ 7.0 h wall-clock; the campaign's GPU use goes to ≈ 34.6 of 35 h.
- **2026-11-09 11:53, the user's decision on the stand body-height lines: re-anchor on the controls.**
  - The question (asked ≈ 10:55): the stand gates' fixed body-height lines will fail beside TDR. Measured on suite rows,
    the stand controllers hold ≈ 1.065 m whatever the walker, and TDR stands at 1.120 m. The line is 0.03.
  - The user answered "Re-anchor on controls": "Re-anchor the eight body-height lines (SN G5 and SH K5-G5) on the R and
    HOLD controls as walker-caused, fixed before any stand row is read."
  - No stand row existed when the decision was made; SN training was at iteration 1871.
  - The rule is recorded in `STAND_REPAIR.md` §4 and §9, in DESIGN 3.6's form:
    - both controls fail the same line, so the failure is walker-caused;
    - the head then passes if it stands within 0.03 m of each control.
  - The heights come from `sn_calib`'s own G5 records, so no extra GPU runs.
  - Earlier on the stand re-pairing, all in `STAND_REPAIR.md` §9:
    - the pre-flight passed (10:42);
    - the review fixed two launcher bugs (10:52);
    - REFS(TDR) ran 10:28–11:04;
    - SN training started at 11:05, and its start-up env check passed at 11:07.
- **2026-11-09 13:24, the SN stand gate passes under the user's body-height rule. SH training started.**
  - The D1 read is HALT on exactly the four G5 body-height lines: the head stands 0.054–0.057 m below REFS(TDR),
    against a 0.03 line.
    - The R and HOLD controls fail each line by about the same amount.
    - The head stands within 0.0022 m of each control, so all four lines re-anchor
      (`results/stand/gates/sn_reanchor.json`).
  - The other 31 decisive lines pass at or near F19's SN values: field_hard completion 0.925 against 0.910, and no stand
    falls.
  - Post-settle stepping at walkstop rose beside TDR. It is a WATCH line and stays under its line.
  - SN head: `SN_RSD_TDR_sn_s19/model_4000.pt` (sha 6cd82ff5). SH warm-starts from it, and its env check passed at
    13:26.
  - The SH read is due near 17:25. Details are in `STAND_REPAIR.md` §9.
- **2026-11-09 17:17, SH stand gate HALT on K4 restore misses only. The user's call.**
  - Twelve of 101 decisive lines failed. Ten re-anchor by the pre-registered rules:
    - the four body-height copies, on the R / HOLD controls (the user's 11:53 rule);
    - six K6 homing shares in bins 1 and 4, on the lazy R_h control beside TDR. The control fails them, and the head
      beats it.
  - K4 remains: restore misses 5 / 6 per 800, against D9's fixed ≤ 3.
    - The scripted R_h control beside TDR misses 4 / 5; it missed 1 / 1 beside S1 and 5 / 5 beside T*.
    - F19's SH missed 2 / 3.
    - There are 0 exit falls.
  - Precedent: on 2026-10-21 the user accepted and disclosed R18's 13 / 10 against its control's 7 / 3.
  - Details are in `STAND_REPAIR.md` §9. A CPU verification workflow is running before the question is closed.
- **2026-11-09 17:34, the CPU check of the SH read confirms it.**
  - The five-agent workflow reproduced every number, including K4 recounted from the raw npz, and every stamp's
    identity. The rule application matches §4 and DESIGN 3.6.
  - **K4.**
    - The misses are a static sag 0.0–0.9 mm over the 8 mm restore band.
    - Head and control share the same place, after the 1.346 m stand, a place the T*-era misses did not use.
    - The head against its control is 11 vs 9 of 1600 (p = 0.82). Against D9's own calibration, 5 / 6 is not chance.
  - **New disclosures:**
    - K5 walkstop homed 0.92 / 0.91, against a 0.90 line. SN beside the same TDR reads 0.997 / 1.000 and F19's SH
      0.993 / 0.997, so this is a weakness of the head.
    - Bin-1 homing 0.74 / 0.81, against F19's 0.98.
    - The 0-exit-fall reading is common to every controller.
  - The decision on K4 stays with the user. Details are in `STAND_REPAIR.md` §9, 17:34.
- **2026-11-09 18:10, the user's decision on SH's K4: "Accept and disclose."**
  - Task 4 DESIGN 3.7 (a), as for R18. The lines are unchanged (thresholds v1.2).
  - Recorded reason: chance-level relative to R_h beside TDR, not to D9's calibration
    (`results/stand/gates/sh_decision.json`).
  - The stand re-pairing to TDR is complete:
    - SN `SN_RSD_TDR_sn_s19/model_4000.pt` (6cd82ff5) passed under the body-height re-anchor;
    - SH `SH_RSD_TDR_sh_s19/model_5000.pt` (89f4428b) is accepted with its K4 HALT disclosed.
  - Next: the REPORT addendum, then the bake proposal. The bake, the commit and the push are the user's.
- **2026-11-09 18:41, the REPORT stand addendum is written (§13), and the bake is proposed (`BAKE_PLAN.md`).**
  - The addendum (≈ 2,100 words) is claim-checked, and its fixes are applied. It updates the REPORT status line, §1
    item 8, §9.2, §10, decision 5 of §11, and §12.
  - The bake plan:
    - bundles TDR, c31000's exports and SN/SH under new `rsd_*` dirs;
    - cites c31000's `.pt` by sha;
    - keeps F19's files as superseded, with nothing deleted;
    - touches no pinned file, so there is no relock.
  - It runs as two local commits (A: the record; B: the bake). It awaits the user's approval.
- **2026-11-09 23:08, the user approved the bake: "Bake as proposed."** It approves `BAKE_PLAN.md` with its §2 defaults:
  - the names `rsd_9_p3b_rodspeed_dr` (TDR), `rsd_10_student` (c31000) and `rsd_stand_net` (SH + SN); `current` and
    `heads.payload` point at TDR;
  - two local commits, A (this record) and then B (the bake), every file staged by name; nothing is pushed;
  - the hardlink copy of the TDR, TCTL and student run dirs into the main checkout's gitignored `parkour/logs/rsl_rl/`
    (REPORT §11 decision 8, taken as approved with it);
  - the `rsd_lib.manifest_student_sha` fallback, the living docs in commit B (pinned prose deferred) and the
    disclosures in the manifest.
  - Not included (§2 item 7): the mix video, Walk-Play rows for TDR, a local tag, a campaign deliverable / head
    keep-set.
  - Done 23:17: the three run dirs are hardlinked into the main checkout's `parkour/logs/rsl_rl/` (TDR 34 files, TCTL
    34, student 47; every file shares its inode, link count 2, so no extra disk). Re-hashed there: TDR
    `model_28991.pt` 03c3e472, TCTL `model_28991.pt` 72d63d51, c31000 `model_31000.pt` 155b5981.
- **2026-11-09 23:17, the user asked for sample videos: "once its done baking record sample videos with the baked
  model."** This adds the mix video of `BAKE_PLAN.md` §2 item 7. VIDEO ONLY, read by no gate.
  - `scripts/bake_videos.sh`, one Isaac process at a time on this worktree's code, from the baked files (TDR
    `policy/rsd_9_p3b_rodspeed_dr/model_28991.pt`, SH `policy/rsd_stand_net/model_5000.pt`; each sha checked first):
    - the walker + stand mix of F19's two bake videos, with the same plan (`--plan mix --seed 1514`) on
      `field_medium` and `field_max`;
    - the depth student c31000 alone, 60 s on the play env of record (`F19/envs/p3b.env`, 0 kg).
  - It runs while the CPU-only build does, since the files are byte-identical to the bake's. The videos are cited in
    the stand block if they are done before commit B.
- **2026-11-10 00:56, BAKE DONE (local commit B; nothing is pushed).** The rod-speed DR pairing is the omni policy of
  record in `crab_hex_omni_task/policy/` (`manifest.yaml`; the README regenerated by `bundle_policy.py --sync`):
  - `current` = `heads.payload` = `rsd_9_p3b_rodspeed_dr` (TDR `model_28991.pt`, 03c3e472). `rsd_10_student`
    (c31000): `policy.onnx` ac0ea3e2, `depth_latest.onnx` 0a04e065 and `omni_student_io.json` 4ee03d02 bundled;
    `model_31000.pt` 155b5981 cited, not bundled. `rsd_stand_net`: SH_RSD_TDR `model_5000.pt` 89f4428b and its
    warm-start ancestor SN_RSD_TDR `model_4000.pt` 6cd82ff5.
  - Superseded with their files kept: F19 (record_commit 64f397e5), F19_student, SH_T4F19, SN_T4F19. F19's s1 .. p3b
    stay live stages (T* is TDR's resume point). `--check`: 10 stages (1 cited, not bundled) + the stand network +
    13 kept superseded files, OK. Nothing is deleted.
  - Built by workflow `rsd-bake-build`: an opus manifest builder (every number read from its record by script), an
    opus test rewriter, a sonnet docs editor. Checked by `rsd-bake-verify`: fable on the records, fable on tests and
    mechanics, opus on docs and deploy, an opus adversarial pass, then an opus fixer.
    - Nothing blocking. 14 of 15 findings were confirmed and fixed: the SN height range (1.0661 m), the 23:06 rule's
      timing, TDR's `overlay_delta` note, the recipe count, the F19 marker on `camera_use_unverified`, the DR glossary
      entry, a CI-reachable test of the `rsd_lib` fallback, and these records.
    - Rejected: teaching `bundle_policy.py` to render `overlay_delta` (a logic change); TDR's stage note covers it.
  - Tests:
    - `test_policy_of_record_omni.py` grew from 78 to 96 tests, and all pass.
    - The unit suite's failure set equals the pre-bake baseline.
    - A CI-shaped run of the bake merged into main adds no failure: depth 1, uid 1001, `env -i`, no main checkout;
      the Isaac venv stands in for CI's, so 45 collection errors are environment gaps.
    - The branch's two pre-existing CI failures remain, both campaign tests and outside the bake:
      `test_ct_yaml_root_normaliser_replaces_the_worktree_before_main` and
      `test_fx_distill_wrapper_reports_the_g1_pass_guard`.
    - CI is not called green before a real run.
  - Pins: no bake file is in Task 4's `pipeline.lock.json` or under the code / eval pathspecs. code_sha de3772e2 and
    eval_sha 9fe964d7 are unchanged, and there is no relock.
  - Also in commit B:
    - the living docs of `BAKE_PLAN.md` §2 item 5 (the omni README, `docs/` FOLDER_LAYOUT, GLOSSARY,
      JETSON_DEPLOYMENT and RUNTIME_ARCHITECTURE, and the root README);
    - the `bundle_policy.py` docstring;
    - the `rsd_lib.manifest_student_sha` fallback;
    - `scripts/bake_videos.sh`;
    - the REPORT (status, §1, §9.2, §11 decisions 5 and 8, §12, §13.9, §13.10) and `BAKE_PLAN.md` (DONE).
    Pinned prose is deferred to a milestone pass.
  - Sample videos (the 23:17 request), recorded 23:18–23:38 from the bundled files:
    - TDR + SH on `field_medium` and `field_max`. All 4 stands arrived and homed in each, 0 falls.
    - c31000 alone, 60 s on `F19/envs/p3b.env`.
    They were sent to the user, and are cited in the stand block and the student's `play_notes`.
