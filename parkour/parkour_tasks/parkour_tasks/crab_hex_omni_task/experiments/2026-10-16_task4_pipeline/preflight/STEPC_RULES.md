# Step 8 (Phase C preflight): pass rules and STOP-C. DRAFT, to be pinned before the first GPU read

- **Status:** draft by the step-8 builder, 2026-10-16. It is pinned by a CHARTER §3 entry *before* any step-8 output
  is read. Until then every number here can still change. PLAN step 8 is authoritative; where this draft fills in a
  number PLAN leaves open, it says so (**[pin]**).
- **Code:**
  - queue: `scripts/t4_stepC_queue.sh`, running the job table `scripts/t4_stepC_jobs.py`;
  - checks: `scripts/t4_stepC_read.py --check NAME`, one per rule below;
  - summary: `t4_stepC_read.py` → `preflight/stepC_read.{json,md}`;
  - Kit scripts: `t4_stepC_camera_play.py`, `t4_stepC_readback_play.py`;
  - dry-pass driver: `t4_stepC_dry.py`;
  - static inputs: `t4_make_stepC.py` → `preflight/stepC/`.
- **Outputs:**
  - rows under `parkour/outputs/omni_plan/task4/stepC/lr/<NAME>/…`;
  - driver runs in the campaign dir `preflight/stepC/` (its own `state.json`; the Phase-A records stay untouched);
  - check records in `preflight/stepC/checks/`.
- **Failure classes:**
  - **STOP-C**: the queue exits. The user picks one of: fix and re-prove (`--retry`), amend the rule in the CHARTER,
    or accept with a disclosure (`--accept NAME --why`).
  - **RE-ANCHOR** (C2 only): decide-record-notify, carried out before the v1 freeze.
  - **REPORT**: the queue continues, and the result feeds v1 or the report.

## 0. Common definitions

| Term | Definition |
|---|---|
| **S1** | X08 `model_19994.pt`, sha256 `968cc550…`. Rows use the run-dir copy, which carries `params/`; drivers resume the bundled copy, which is byte-identical. |
| **bitwise** (eval) | Two conditions, both required (`t4_stepC_lib.bitwise`): (1) every `raw/episode_*.npz` key is `np.array_equal`, with NaN equal to NaN, the same episode set and no extra key; (2) `scenario_metrics.json` `aggregate` and `episodes` are equal once the metadata key `trained_categories` is dropped from both. That key is not read by scoring (PLAN item 3). `run_meta` is never compared. **Stand rows compare aggregates only** (PLAN item 10). |
| **P-early rule** (training) | `replay_read.pearly_series` over steps 0–19, as at STOP-0. Timing tags are ignored. `Loss/priv_reg` may differ by \|Δ\| ≤ 2e-6, and `*hist_latent*` by ≤ 4 ULP when every other non-timing tag is bitwise. Everything else must be bitwise. Strict bitwise equality is also reported. |
| **pipeline row path** | `TP/pipeline.py row_job` on the R18 plan: the generated `R18/envs/<stage>[.<family>].env`, the trained categories and adapter the pipeline passes, levels from `R18/envs/levels/L###.env`, Student-Play rows with `TP_STRIP_KEYS` set to the refused runner knobs. Each row is run by `TP/gate_row.sh`. A stamp is complete when it has `tp_row.json` and the checkpoint sha matches. |
| **REFS path** | `run_arm.sh --row-eval` with `P2_REANCHOR=1`, the launcher of the pipeline's REFS(T\*) rows. |
| **stand path** | `sh_eval_rows.sh --deadband 0.25 … stand_net <manifest\|zero> armed <ckpt> 1`, with `P2_REANCHOR=1`. |
| **R, G** | R is `thresholds_v0.json` `R_used` (`cat.<c>.<stat>`). G is the binomial guard `thresholds.binomial_guard_floor(p̄, n)`. |
| Env seed | 1 unless a row says otherwise. |

## C1. Inertness bridges (code moved at M1a / M1b / E1). Any failure is **STOP-C**

PLAN: from-scratch smokes only. A resumed stage carries the nondeterministic `priv_reg` term once the counter
passes 2000.

| # | What | Rows / runs | Rule (**[pin]** where marked) |
|---|---|---|---|
| C1.1 | 13 cfg dumps | `t4_dump_cfgs.py` at the build tree, with the targets in `body_cmd/dump_targets.json` | Each target's JSON equals the M1a knob-off reference `preflight/stepC/dump_ref/<t>.json` (copied from the step-3 scratchpad) once the reference's worktree root is replaced by the repo root. No target carries `payload_reset`. With no reference the check is REPORT. |
| C1.2 | P-config, repeated after C1 | Six 20-it smokes `PC_<stage>_s18`, copied verbatim from Phase A's `PF_*`: s1 from scratch; c4…p2a resume the bundled lineage heads | `replay_read.pconfig` against X43 / X57 / X60 ×3 / X08 under the CHARTER step-2 allow-list. Any key outside the allow-list fails. |
| C1.3 | P-early | `PC_s1` against X43 | The P-early rule (it would be STOP-R at R18). |
| C1.4 | Post-build smoke against A2 | `PC_s1` against Phase A's `PF_s1` run (from `preflight/orchestrator.log`) | The P-early rule. **[pin]** Strict bitwise is reported but does not decide: Phase A showed 1.4e-6 `priv_reg` / 2-ULP `hist_latent` noise between same-code runs. |
| C1.5 | Payload armed at 0 kg against off | `PC_s1pay00_s18` = `PC_s1` + `KRABBY_PAYLOAD_KG=0:0`, `ZERO_FRAC=1`, `COM_XY=0`, `COM_Z=0.309`, against `PC_s1` | The P-early rule on **shared tags only**: `Payload/*` exists on the armed side only (CHARTER step 3). Also required: the armed run's `params/env.yaml` holds `payload_reset`, and `Payload/kg_mean` ≡ 0. |
| C1.6 | Eval inertness | Each row runs with the recorded stamp's `run_meta.env_vars` exactly, and with the recorded `--trained-categories` / `--command-adapter`. Rows: S1 `suite__omni` es1 against REFS_S1_s18 es1; S1 `field_hard__omni` es1 against REF_S1_s18 es1; S1 `suite__omni_body` es1 against ANC_S1_s18 es1; X43 s1, X60 c6x and X10 p2a `suite__omni` es1 against their lineage stamps (the PPO driver stamp, ANC_s18, the P2 driver stamp); S1 suite es1 **armed at 0:0** (level file L000) against REFS es1 | Bitwise. The armed row's `run_meta.payload` must also read 0 kg on every env. |
| C1.7 | Forward 3a slow row, same manifest shape | `slow__A15pB` es1, `--task Isaac-Crab-Hex-Student-v0`, in the **recorded** shape: an env block holding the recorded `KRABBY_HEX_USD_PATH`, written to `rows/scenarios_fwd3a_oldshape.yaml`. The process env is the FWD policy README canary env. The 2026-09-08 harness recorded only the block, so this process env is a reconstruction. | Bitwise against the 2026-09-08 stamp (0.79 / 21), on the **shared record** **[pin]**. The stamp predates three raw keys today's harness writes (`clock_cmd_speed`, `clock_running`, `rod_speed`); those, and any aggregate or episode key that exists only on the new side, are allowed and listed. Every shared raw key, and the reference's aggregate and episode keys, must be bitwise. The row in today's shape is REPORT only: 0.75 / 25 is expected, and the difference is the forward manifest-shape split. |

## C2. Eval bridge (the pipeline's row path on recorded heads). A failing family is **RE-ANCHORED**, not STOP-C

| # | Rows (es1) | Reference | Compare |
|---|---|---|---|
| C2.1 | S1 through the pipeline row path, R18 p2a composed env. The composed env adds `COMMAND_FRAME=world`, shown inert by G0w. Families: suite, side22, field_hard, field_light, ring_obst, field_turn, field_suite, suite_body, turnlevels_body, grid_body. | REFS / REF_S1_s18, P2__S1_s18 side22, ANC_S1_s18 | bitwise |
| C2.1b | turn_band on X60 c6x, with the c6x env. S1 has no turn_band stamp. | SF__B_rest_s18_tb | bitwise |
| C2.1c | REFS path: suite@1, walkstop@1 | REFS_S1_s18 | bitwise |
| C2.2 | Stand path: SH_T17 `hwalkstop__omni`@1 and `hraise__omni`@1 (manifest command); SH_T17z `walkstop__omni`@1 (zero command) | SH_T17 / SH_T17z | aggregates equal |

Also run: S1's es2 rows suite, field_hard, field_light, ring_obst and field_turn. They are not bridges: they are the
p2a side of the C3.8 paired check and of the C4.3 parity. dirsweep_body has no recorded stamp; it is anchored in C5.

**C2.3 Re-anchor procedure** (before the v1 freeze, at the pin), for each differing family F:

1. Re-run F on every recorded stamp of F in `threshold_inputs.yaml`, at the same checkpoint and env seed, through
   the pipeline row path with that stage's R18 composed env. This covers the s16/17/18 heads of every stage that
   reads F and the known-bads read on F.
2. Stand families go through the SN/SH launchers instead, using the head's recorded controller, command and
   deadband, with `P2_REANCHOR=1`. This covers SN_Amask_s16/17/18 and SH_T16/17/18 at es1 and es2.
3. Write `threshold_inputs_v1.yaml`: v0 with F's stamps replaced.
4. Run `thresholds.py --self-check` → `thresholds_v1.json`.
5. If the self-check fails (a recorded seed fails a decisive line, or a known-bad passes): **STOP-C**. The line set
   then needs the user.
6. Make a CHARTER entry listing every moved line (old → new value and rate).
7. Re-anchor WATCH-only families too (body, turn_band, grid, field_suite), so the WATCH comparisons stay
   like-for-like. They cannot fail the self-check, because they have no decisive line.

## C3. Payload

| # | What | Rule | On fail |
|---|---|---|---|
| C3.1 | **PhysX readback after partial resets.** Three envs: Teacher-v0 with `R18/envs/p3a.train.env` (the D5 training values, startup DR live), 16 envs, 24 rounds; Walk-Play with p2a.env + L090, 16 envs; and in C4, Student-v0 with the stripped p3a distill env, 8 envs. Each round resets a random subset of 2 to N/4 envs through `env._reset_idx`. | Read `root_physx_view.get_masses/get_coms/get_inertias` and compare with an **independent** float64 composite of the term's own record. Tolerances **[pin]**: \|mass\| ≤ 1e-3 kg; CoM ≤ 1e-4 m (inf-norm); **principal moments** (sorted eigenvalues) relative error ≤ 1e-3. Eigenvalues are used because they are frame-invariant and PhysX re-diagonalises (CHARTER step 3); the link-frame tensor error is reported. Also required: every 0-kg env is bitwise its captured base; envs outside the reset subset are bitwise unchanged; every draw lies in `kg_range`; both loaded→0 and 0→loaded transitions are seen; at a level, every env is at exactly L with CoM (0, 0, 0.309). | STOP-C |
| C3.2 | **100-it timing** (`PT_p3a_s18`, from S1, p3a overlay) | Median wall s/it, from the tfevents wall times, against p2a's 1.49 s/it | REPORT (budget) |
| C3.3 | **Deck-CoM zero-action settle** at 0, 45, …, 270 kg. A3 protocol, run by `t4_eval_row.sh` in ad-hoc mode: Walk-Play, R18 p2a.env + `KG=L:L ZERO_FRAC=0 COM_XY=0 COM_Z=0.309`, `--holds 0:10:stand --episodes 32 --zero-actions`. | **Tips** (A3 rule) means any episode whose `termination_reason` is not `schedule_complete`. **STOP-C if it tips at any L ≤ 90**; tipping above 90 is REPORT. Σ foot force (`payload_read.actuator_load`: the median over episodes of the per-episode median Σ\|F_foot\| / ((230.06 + L)·g)) must lie within **1 ± 0.03 at every level that did not tip** (PLAN: hard only here). For reference, A3 measured 0.976 → 0.973 at 0–180 kg, a thin margin. `run_meta.payload` must read exactly L on every env. | STOP-C |
| C3.4 | **S1's 0–270 curve**, the payload "before". Level rows suite@1, suite@2, field_light@1 and field_light@2, armed L:L through the pipeline row path, ascending, stopping at the first unstable level. | Stable at L means `payload_read.level_reads` on the `p3a.stable90.*` lines, over the trained translate categories fwd, side, back and diag. **S1 at level 0 (armed 0:0) must be stable** (PLAN self-check; a CPU pre-read of the unarmed REFS rows is stable, 35 lines). Every level row's realised payload must equal L. max_stable(S1) is reported. | STOP-C (level 0 or payload); otherwise REPORT |
| C3.5 | **Pilot.** `PP_p3a_s18`: 3000 its from S1, seed 18, the spec's p3a overlay (p2a verbatim + 0:90, 0.25, 0.10, 0.309). Saves every 100 its. | Read points: the saved checkpoint nearest 19994 + {500, 1000, …, 3000}, i.e. model_20500, …, model_22500 and model_22993 (PLAN item 8). | — |
| C3.6 | **N_p** | At each read point: suite es1 + es2 at L090. For every trained translate category, pool survival (n-weighted) and ratio (the mean of the two env-seed medians). **N_p is the first read k ≥ 1000 at which every category's \|Δ survival\| < R(`cat.c.survival`) and \|Δ ratio\| < R(`cat.c.ratio`) against the read at k − 500**, clamped to [1500, 3000]. If no read settles, N_p = 3000, flagged `not_converged` with the budget-bound tail read. **[pin]** The "pilot checkpoint at N_p" is the saved one nearest 19994 + N_p, at most 6 its after the p3a stage head; this is disclosed. | STOP-C only if N_p is undetermined, or a read row's payload ≠ 90 |
| C3.7 | **Live rules under load** | `live_watch` driver rules (resume, collapse, NaN), replayed every 50 its on the pilot's tfevents with source X08. **STOP-C if any rule would kill the pilot.** Reported for calibration: min over polls of rolling-200 episode length / (0.5 × source tail), and max rolling-200 crab_failure. For reference, p2a replayed on X08 from X60 gives 1.94 and 0.29. | STOP-C |
| C3.8 | **The full p3a gate on the pilot at N_p** | Rows: the 0-kg row list (suite@1 run through gate_row, suite@2, side22@1, field_hard@1-2, field_light@1-2, ring_obst@1-2, field_turn@1-2; unarmed), plus the curve from 0 upward. The paired side is S1's same rows (C2 es1 + the C2 es2 rows). **PASS iff** (a) the `p3a.floor.*` lines pass; (b) max_stable ≥ 90; (c) `gate_read.zero_kg_check` passes, with its_past_p2a = pilot iteration − 19994; (d) the 0-kg rows carry no payload and every level row is exactly L. Reported: the first unstable level and its limiting factor, and each 0-kg statistic's margin use. | STOP-C |

## C4. Cameras and the student

| # | What | Rule | On fail |
|---|---|---|---|
| C4.1 | **Camera check per camera**, on the built Student-Play env: flat (R18 p2a env) and field (p2a field_hard env); 64 envs, 4 resets, 50 zero-action steps each | (a) **Axis error ≤ 1.0°**, by two measures: the sensor's world quaternion rotated into the root frame, and the mean cast-ray direction, against `omni_student_spec.expected_axes`. Max over envs and resets. (b) Flat: the ground band (in-range hits within 0.02 m of the env's ground level, on the look axis in the body frame) **covers `cameras.yaml` target_4b in ≥ 95 % of env reads**. The targets are front +x [0.5, 1.2], rear −x [0.5, 1.2], left +y [0.75, 1.45] and right −y [0.75, 1.45]. The depth image is non-empty (some pixel < 2 m) in ≥ 95 %. (c) Field: **≥ 1 env read has ≥ 5 in-range rays more than 0.02 m off the ground**, from any camera. The ray-cast mesh set is shared, so one camera seeing a block proves the blocks are in it. Per-camera counts are reported. | STOP-C |
| C4.2 | **Student-env payload readback** (Student-v0 + the p3a distill env) | The C3.1 rules | STOP-C |
| C4.3 | **Teacher-on-Student-Play parity.** S1 with `--policy-role teacher` on Student-Play (refused knobs stripped) against S1 on Walk-Play (C2 rows), on suite@1, side22@1, field_hard@1, field_hard@2 and field_light@1 | Each row passes if it is **bitwise**, or **within bands** **[pin]**: suite, per trained translate category, \|Δ survival\| ≤ max(R(`cat.c.survival`), G(p̄, n = 100)) and \|Δ ratio\| ≤ R(`cat.c.ratio`); turn and move_turn survival take the same band; turn facing_closed \|Δ\| ≤ R(`cat.turn.facing_closed`); side22 classes must be identical; field rows \|Δ completion\| ≤ max(0.05, 2·SE_paired). Also required: `run_meta.task` is Student-Play, and no refused knob appears in the env. | STOP-C |
| C4.4 | **4-camera timing.** A 200-it distill from S1, 192 envs | Median wall s/it over its 20–200 × 1000 / 3600 **> 4.0 h per 1k its → STOP-C**. Reported: the projected 5000-it hours. Forward K = 1 is 1.94 h/1k (CPU-measured). | STOP-C |
| C4.5 | **2 × 50-it determinism.** Same seed, checkpoint and env | Deterministic means every non-timing tag is bitwise at every step **and** the final checkpoint sha256 is equal. This tells whether a student rerun is a replay. | REPORT |
| C4.6 | **The manifest-shape split.** On the depth pilot's head, two pairs: field_hard es1 with its env block against **without** it (`scenarios_shape_field.yaml`; the keys come pre-Kit from the row env file), and suite es1 without a block against **with a no-op block** (`scenarios_shape_suite.yaml`, `KRABBY_RSI_FRAC: '0'`, the pre-Kit value) | **split = max \|Δ completion\|** over the two pairs, rounded **up** to 0.01, which goes to `sel.completion_margin.manifest_shape_split`. It is flagged low-power if the pilot's field_hard completion is < 0.2. Since the margin is max(0.05, 2·SE_diff, split), a split ≤ 0.05 changes nothing; the forward 3a recorded 0.04. | REPORT (a v1 value) |
| C4.7 | **Depth helps.** Depth pilot against zero-depth pilot, 1000 its each from S1, seed 18. Zero depth is Hydra `agent.depth_encoder.zero_depth=true`; it lands in the run's `agent.yaml`, so the eval of that run also zeroes depth. | Over field_hard es1-2 (Student-Play, student role), completion paired on (env seed, env_index): **Δ = c_depth − c_zero ≥ max(0.05, 2·SE_paired)**. DESIGN wording. | STOP-C |
| C4.8 | **live_watch on student logs**, replayed every 50 its | Depth pilot (T\* tail from X08): **the episode-length / crab_failure rules must not kill** it, else STOP-C. The loss rule (rolling-50 at +250 ≤ 0.5 × first): if it kills on the depth pilot **or** on the forward 3a (T\* = forward 2c), v1 demotes it to **WATCH**, as PLAN says; this is not a stop. The zero-depth pilot is reported, as a known-bad candidate for these rules. A CPU pre-read of the forward 3a found that neither the loss nor the ep_len/crab rules kill it. | STOP-C (ep_len/crab); otherwise a v1 value |
| C4.9 | **Distill from the payload-pilot head** (at N_p) under the p3a distill env, 200 its | The training logs carry `Payload/kg_mean`, last-50 mean within ±25 % of 0.75 × 90 / 2 = 33.75 kg, and `Payload/loaded_frac` within 0.75 ± 0.10. Per-env mass and CoM come from C4.2. **One 90-kg row** (suite es1, Student-Play, student role, L090): its `run_meta.payload` reads 90 kg on every env, and its task is Student-Play. Its performance is REPORT. | STOP-C |

## C5. Sweep anchors (report only)

- Rows: `dirsweep__omni_body` es1-2 on S1 s16 (X10), s17 (X11) and s18 (X08), through the pipeline row path (R18
  p2a env, body_hold).
- The stamps become threshold_inputs v1 `sweep_anchors` for the WATCH sweep lines.

## C6. Stand

| # | What | Rule | On fail |
|---|---|---|---|
| C6.1 | **Smokes beside a non-S1 walker.** **[pin]** The walker is the payload pilot at N_p. Its overlay carries `KRABBY_PAYLOAD_*`, so it is the strict test; DESIGN named X10. `SN_WALKER_SPEC` is built exactly as `pipeline.walker_spec_env`: S1_OVERLAY = the p2a overlay, OVERLAY_CHAIN = `PC_p2a_s18`, REFS_ENV_STAMP = the pilot's 0-kg suite es1 stamp, plus `P2_REANCHOR=1`. | (a) `sn_dagger.sh` / `sh_dagger.sh --dry-run` each compose an env, with **no `export KRABBY_PAYLOAD_*`**. (b) SN 100-it smoke (`--smoke --smoke-iters 100`, horizon 4000, DS 17), then SH 100-it smoke (horizon 5000, the recipe options, warm start = the SN smoke's head). Each needs `smoke.json` verdict PASS, `params/krabby_env.json` with no `KRABBY_PAYLOAD_*`, and `params/env.yaml` with no `payload_reset`. | STOP-C |
| C6.2 | **Determinism, which sets `x_rule`.** Two SN smokes (walker unset = the bundled S1, DS 17, horizon 4000) and two SH smokes (unset, default warm start, horizon 5000, the recipe options) | **x_rule = `bitwise`** iff both pairs have equal final checkpoint sha256 **and** equal `train_log.jsonl` loss curves (`replay_read.loss_curve`); otherwise `report_only`. Reported: whether a's first 100 rows equal the recorded SN_Amask_s17 / SH_T17 prefix, which predicts X. | REPORT (a v1 value) |

## C7. Full-graph dry pass. FAIL is **STOP-C**

`t4_stepC_dry.py` drives `pipeline.py run --run dry` end to end.

- **Spec.**
  - Default: the **reduced** spec `preflight/stepC/dryspec/pipeline.yaml`. It keeps the official stage graph,
    overlays, levels and stand rows. Shorter gate row lists still exercise every row kind: suite, lr, field, body,
    level, Student-Play and stand. It has its own lock and row root.
  - `T4C_DRY=official` runs the official spec instead, ≈ 12 GPU-h. Re-lock PL first: `PL/scripts/*` are lock inputs.
- **Injections:** c4 FAIL, c5 INVALID, c6 GUARD, st LIVE, sn LIVE.
- **Natural HALTs:** every walker and payload gate halts on its 20-it head; this is expected. sel ends in a STUDENT
  HALT, because a 50-it student has one candidate and fails its block.
- **Pre-registered actions:**

  | HALT | Action |
  |---|---|
  | FAIL / LIVE | `continue --accept-chance` |
  | The injected c6 GUARD | `continue --retry` |
  | STUDENT | Left open |
  | Anything else | Fails the dry pass |

- **Resume test:** once, while a driver stage after s1 is training, send SIGTERM to the **queue** process (never Kit)
  and relaunch `run`.

**PASS iff all of these hold:**

- (a) Terminal statuses:

  | Stages | Allowed status |
  |---|---|
  | s1 … p2a, p3a | accepted / pass |
  | p3b | ceiling / accepted / pass / skipped |
  | p3c | skipped / ceiling / accepted / pass |
  | tstar, st | pass |
  | sel | halt (STUDENT) / pass |
  | sn, sh | accepted / pass |
  | sn_ref, sh_ref | pass |
  | bundle | blocked / pass |

- (b) Every injection fired once, with its effect:
  - c4: a FAIL record whose line is `c4.injected`;
  - c5: the "INVALID once" notify, after which its rows rerun;
  - c6: a GUARD record;
  - st: a LIVE record, then `distill_attempt` ≥ 2;
  - sn: a LIVE record, then `dagger_attempt` ≥ 2.
- (c) No CRASH, INFRA, DRIVER or STOP-R HALT, no INVALID HALT (Gate 0 refusing real stamps twice), and no GUARD
  other than the injected one.
- (d) After the SIGTERM, the relaunched queue logged `ATTACH stage=<s>`.
- (e) Never more than one Isaac process.
- (f) Every driver stage has `gates/<s>.json`.
- (g) Each generated chain resumes `{reference: <previous chain>}`.

## STOP-C triggers (consolidated)

The queue stops on any of:

1. A C1 bridge fails: cfg dumps, P-config, P-early, post-build against A2, armed 0:0, or an eval row not bitwise
   (including the forward 3a old-shape row).
2. The payload readback is wrong (C3.1, C4.2).
3. The settle tips at ≤ 90 kg, Σ foot force is outside 1 ± 0.03, or the realised mass ≠ L (C3.3).
4. S1 at 0 kg fails the a-priori stable lines (C3.4).
5. The pilot is killed by the live rules (C3.7), or fails the full p3a gate at N_p (C3.8).
6. A camera check fails (C4.1).
7. Parity falls outside the bands (C4.3).
8. Timing is above 4 h per 1k its (C4.4).
9. Depth fails to help (C4.7).
10. The ep_len/crab live rules kill the depth pilot (C4.8).
11. The payload-pilot distill shows the payload missing or wrong (C4.9).
12. A stand smoke fails, or a payload key appears in the DAgger env (C6.1).
13. The dry pass fails (C7).
14. A C2 re-anchor makes the v1 self-check fail (C2.3).

## V. What step 8 feeds into v1 / `pipeline.yaml`, and how

Every value goes in with a CHARTER entry. `stepC_read.json` carries all of them under `v1` / `patch`.

1. **`pipeline.yaml`:**
   - `N_p` (C3.6);
   - `x_rule` (C6.2);
   - `stages[sel].completion_margin.manifest_shape_split` (C4.6).

   Then run `pipeline.py gen` (R18, F19, dry), `pipeline.py lock`, and `check` (which refuses placeholders).
2. **thresholds v1.** `threshold_inputs_v1.yaml` is v0 plus:
   - (a) the C2.3 re-anchored stamps;
   - (b) a `p3a_pilot` block: the pilot's L090 rows at N_p, its 0-kg rows and S1's;
   - (c) `sweep_anchors` (C5).

   Then run `thresholds.py --self-check`. The p3a terms of the run sum are recomputed from the pilot:
   - each `p3a.stable90.*` rate = `payload_apriori_rate(line, μ_pilot@90, σ̂_p2a, SE)`;
   - the paired 0-kg rate from the pilot pair's `paired_rate(margin, SE_diff)`.

   `t4_stepC_read` computes both as a cross-check (`v1.p3a_rates`). **This needs a TP change (below).**
3. **Live rules:**
   - `live.student.loss_rule`: kill | watch (C4.8; **needs a TP change**);
   - the collapse rule under load is reported (C3.7), and stays `unvalidated`.
4. **Budget:** the measured payload s/it (C3.2) and student h/1k (C4.4, C4.9) update the CHARTER budget.

## TP changes this step needs (not made here)

1. **`pipeline.Queue.walker_spec_env`** returns `{}` for the X stages (the bundled S1). The X stand rows then run
   `sn_eval_rows.sh` / `sh_eval_rows.sh` → `run_arm.sh` `code_guard` against REFH_S1_s18's sidecar without
   `P2_REANCHOR`. That sidecar predates M1, so at the pin code_sha they are **refused** (rc 7), and sn_ref / sh_ref
   would HALT as INFRA. The fix: return `{"P2_REANCHOR": "1"}` there, since the pin guard holds the code state. The C7
   dry pass would expose this defect as a FAIL.
2. **`thresholds.py`:**
   - consume a `p3a_pilot` input, filling the `p3a.stable90.*` rates and the zero_kg_paired rate in the run sum;
   - consume `sweep_anchors` for the WATCH sweep lines.
3. **`live_watch.judge_student` + `pipeline.yaml live.student`:** a `loss_rule: kill|watch` switch. With watch, the
   rule is computed and reported but never kills.
4. **Process, not code:** `pipeline.py lock` must be rerun after this step, because the new `PL/scripts/t4_stepC_*`
   files are lock inputs. Without that, the official `run` / `check` report "files … not in the lock".
