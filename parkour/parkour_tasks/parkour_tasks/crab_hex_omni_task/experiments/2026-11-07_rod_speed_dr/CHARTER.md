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
- **Code pin:** CODE_PIN_PENDING
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
- **Student class flip** (c28000 bridge, G2 relaunch, closing): a mismatch is a class flip when the cfg dumps and
  the per-stamp dose are identical and the first divergence is an action delta below 1e-5 at a depth-frame step
  (the known process-state signature). A divergence at the first step, in the dose or in a cfg dump is a code-path
  difference, and it stops the campaign.

## 5. Files
- `scripts/`: `rsd_lib.py`, `rsd_gate.py`, `rsd_train.py`, `rsd_distill.sh`, dump and smoke scripts.
- `envs/`: composed env files. `results/`: pins, reads, verdicts, logs.
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
