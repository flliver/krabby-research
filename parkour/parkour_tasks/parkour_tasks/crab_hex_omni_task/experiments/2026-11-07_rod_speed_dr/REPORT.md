# Rod-speed DR in teacher training (rung 3, T01 fix) — REPORT

**Status: FINAL 2026-11-08 23:46. G2 PASS, read 2026-11-08 22:55 and verified (§8.4). Teacher gate G1 PASS; selection SELECTED c31000; ONNX parity ok; student gate G2 PASS. Nothing is adopted, baked or pushed. Updated 2026-11-09: the pick's veto clip is recorded and the record is committed locally (§11 decisions 4 and 6). Adoption, bake and push are the user's calls.**

- **Campaign:** `crab_hex_omni_task/experiments/2026-11-07_rod_speed_dr/`. It follows the DR stress sweep (`../2026-11-06_dr_stress_sweep/REPORT.md` §3, §9 decisions 2–3) and runs rung 3 of `../lit-review-domain-randomization.md` §13, shrunk to rod speed. Pre-registered 2026-11-07 15:40, before any GPU row [7 15:40]. Plan of record: `~/.claude/plans/rosy-toasting-dongarra.md` (3-pass audit, cap reached, accepted as is).
- **Code and pins:** branch `feat/rod-speed-dr`, commit `1682e2e1` (local). code_sha `de3772e2…`; eval_sha `9fe964d7…`, the sweep's, unchanged. Pinned 2026-11-07 17:38 on a clean tree (`launch.json`). Every row from the reference arm to G2 ran on this pin (`g2.json` pin). Lines: Task 4 `p3a.stable90`, thresholds v1.2, applied per seed.
- **Policies** (`policies.json`):
  - **T\*** (old T\*): F19's teacher, `policy/f19_8_p3b_payload180/model_25992.pt`, sha `03b6009a`.
  - **TDR**: T\* plus 3000 iterations with rod-speed DR, `…/crab_hex_omni_teacher/2026-11-07_19-49-46_RSD_TDR_s19/model_28991.pt`, sha `03c3e472`.
  - **TCTL**: T\* plus 3000 iterations without DR (the control), `…/2026-11-08_00-20-19_RSD_TCTL_s19/model_28991.pt`, sha `72d63d51`.
  - **PICK** (c31000): `model_31000.pt` of the student run `…/crab_hex_omni_student/2026-11-08_02-57-24_RSD_ST_s19`, sha `155b5981`.
  - **c28000**: F19's depth student of record (`f19_9_student`), sha `4fa81f0f`.
- **Rows:** 175 rows in all (171 campaign rows and 4 diagnostic rows), each with clean launch checks on its first try.
  - 137 ran before G2. `rsd_gate.log` has 137 `RSD ROW … try1 … ok=True` lines before `RSD START g2`, the pick's 15 selection rows and c28000's 2 `_RL` bridge relaunch rows included.
  - G2 added 34 rows, all `try1 … ok=True` (`rsd_gate.log`, [8 22:55]).
  - Four diagnostic rows ran 23:02–23:24, read by no gate (§8.4, [8 23:26]).
- **Budget (stated once):** approved ≈ 35 GPU-h (2026-11-07). Used ≈ 24.1 before G2's rows [8 18:00]. G2's 34 rows ran 19:49–22:53: 3.1 h of wall-clock and of GPU, since the queue is serial [8 22:55]. The read took ≈ 30 s on CPU (22:54:34–22:55:06, `queue.log`). The diagnostic relaunch took 0.36 h (23:02–23:24, four rows) [8 23:26]. Total ≈ 27.6 of the approved ≈ 35. On 2026-11-08 the user said that on this local GPU only wall-clock time matters, and that stopping to save GPU time helps only if it also saves wall-clock [8 19:49]. So every duration below is in wall-clock hours.
- **Checks so far:** four read-only adversarial workflows, four agents each, all confirming:
  - the G1 read before TCTL (`g1-read-verify`, 4 / 4, [8 00:30]);
  - the G1 PASS read (`g1-pass-verify`, 4 / 4, [8 03:10]);
  - the selection (`sel-verify`, 4 / 4, [8 15:35]);
  - the c28000 bridge diagnosis (`c28k-bridge-diagnose`, 4 / 4 class flip, [8 18:00]).

  For G2 two more checks ran after the read, and both agree with §8.4 [8 23:26]:
  - a diagnostic relaunch of the closing row (23:02–23:24). Both relaunches, one in the cell row's shape and one in the selection row's, equal the closing's Z stamp bit for bit on both rows; the closing's reference is the other class;
  - `g2-verify` (23:04–23:23), 4 / 4 confirm. Every G2 number recomputes within 1e-9. It made two corrections to the record's account of the closing (the state that carries across waves; 271 of 900 actions), with no effect on any verdict.

  This REPORT was then checked claim by claim before it was saved (`rsd-report-final-check`: two fable claim checkers and an opus completeness critic). They raised 21 issues, all minor. None changed a verdict or a G2 number, and all were fixed in 25 edits [8 23:46].
- **Commits:** `1682e2e1` holds the knob, its tests, CHARTER §1–3 and the pre-pin §4, the cell env files and `p3b_dr.distill.env`, `k1.out` / `k1.reread.out` / `k2.out`, and the first versions of `policies.json` and `replay.*`. The record commit of 2026-11-09 (decision 6, local, not pushed) holds the rest:
  - in the CHARTER: the §4 code-pin line, the ONNX block (18:10), the amended class-flip rule (20:12) and its multi-wave clarification (23:06), the §5 files list, and every §6 entry after `1682e2e1`;
  - this REPORT;
  - `scripts/rsd_queue.sh`, `scripts/rsd_onnx.py` and `scripts/clip_pick.sh` (the veto clip, decision 4);
  - `scripts/classflip_check.py`, `scripts/wave_diff.py` and `scripts/diag_closing_rows.py` (copied from the scratchpad at 23:29);
  - `results/`, including the modified `policies.json` and `replay.*` and `results/closing/`;
  - the two training env files.
  - Left untracked: the empty `results/rsd_gate.lock`, the sweep's `run_sweep.lock`, and the `k1/` and `k2/` dumps and logs. The clip mp4s and `*.log` files are gitignored.

**Notation.**
- **Cell:** one stress level on one payload base. "×0.7 L090" means rod slew scaled to 0.7 × rated (hip 0.028, knee 0.033 m/s) with a 90 kg payload.
- **L000 / L090 / L135:** payload base 0 / 90 / 135 kg.
- **es1 / es2 / es3:** eval env seeds 1 / 2 / 3. es3 is run only for a near-line cell.
- **Row:** one scenario on one env seed, giving one stamp.
- **Cell-seed:** one cell on one env seed, i.e. two rows: the flat category suite (10 waves × 100 envs, 12-s holds) and field_light (100 episodes of an 8-hold schedule on the walled light field).
- **Wave:** one of the suite row's 10 batches of 100 envs. All 10 run one after another in one process.
- **Closing:** one relaunch of a row already stamped, to check that it reproduces. Its cell name ends in `_Z`.
- **In-pin / cross-pin:** run on this campaign's code pin / compared with a stamp from another pin (the sweep's, 2026-11-06).
- **Lines:** the 18 per-seed Task 4 `p3a.stable90` lines (thresholds v1.2):
  - field_light completion ≥ 0.85;
  - survival ≥ 0.9 (fwd, side, back, diag) and ≥ 0.885 (turn, move_turn);
  - tracking |r − 1| ≤ 0.25 (fwd, side, back, diag), where r is the median achieved / commanded speed;
  - turn_fc ≥ 0.8;
  - on suite and on field_light: h_meas ≥ 0.85 m, roll_rms ≤ 0.1, pitch_rms ≤ 0.1.

  Line values are in `sel.json`'s level lines.
- **Pooled:** es1 + es2 together (200 field_light episodes), paired episode for episode.
- **m:** max(0.05, 2·SE_diff), with SE_diff the paired SE of the pooled difference (Task 4's margin).
- **Class flip:** defined in §7.1.
- **Citations:** [7 hh:mm] and [8 hh:mm] are CHARTER §6 entries of 2026-11-07 and 2026-11-08. File names are under `results/` unless a path is given.
- **†:** recomputed for this REPORT from the stamps. Line values use the sweep's `stress_read.read_stamps` (CPU, in memory). Episode-pair counts use each field_light stamp's `scenario_metrics.json` under Task 4's `field_episodes` rule (complete = `schedule_complete`). These values are in no results file.

## 1. Bottom line

1. **G1 PASS: per-env rod-speed DR in teacher training removes the sweep's FRAGILE cell for the teacher.**
   - At ×0.7 L090, TDR completes 0.94 / 0.92 of field_light episodes. Old T\* completes **0.59 / 0.61** (`g1.json`, `bridge_TSTAR.json`).
   - TDR passes every line on both seeds in all five stress cells, the band edges ×0.7 L000, ×1.35 L000 and ×1.35 L090 included. The tracking carve-out was not used.
2. **At ×0.7 the gain comes from the DR, not from the extra training.**
   - TCTL had the same 3000 iterations without the knob. It reaches **0.71 / 0.58** and fails completion and survival.diag on both seeds†.
   - A = TDR − TCTL, pooled, is +0.285 against m = 0.076, about 7.5 SE (`g1.json`, [8 03:10]).
   - At ×0.8 the extra training alone does most of the work: TCTL 0.94 / 0.92, TDR 0.95 / 0.96.
3. **No nominal regression, and some ungated shifts.**
   - Pooled completion against old T\*: L000 −0.005, L090 +0.025, L135 +0.085, each with m = 0.05 (`g1.json`).
   - R passes with max margin used 0.632. The closest row is TDR's 0-kg forward speed ratio: |r − 1| 0.039 against 0.007.
   - TDR stands ≈ 0.03 m taller in every cell, and at ×0.7 its suite pitch_rms is about half of old T\*'s (0.028 against 0.053). TCTL stands like old T\*. Neither shift is gated.
4. **The student is distilled, selected and exported.**
   - RSD_ST_s19 was distilled from TDR with the DR on: 5000 iterations in 9.6 h [8 12:33].
   - Selection picked c31000. It had 0 prescreen misses and ranked first on field_hard by 2 of 200 episodes. Its full block passed all 21 decisive lines against TDR on the first try (`sel.json`).
   - Its ONNX export matches the `.pt` to 1.2e-6 on CPU (`onnx_PICK.json`).
5. **The c28000 bridge stopped, and a rule change by the user resolved it.**
   - One field_light stamp of F19's student (L090 nominal es1) differed from the sweep's on launch and relaunch: completion 0.95 against 0.97 (`bridge_C28000.json`).
   - It reproduces Task 4's own 2026-10-23 stamp bit for bit. That makes it a class flip: the eval landed in the other of its two repeatable process-state outcome classes, with no code change.
   - §4's pinned rule, read literally, called it a code-path difference. The user amended §4 [8 20:12].
   - While writing the amendment in, the executor corrected one clause of the text the user had approved. The user kept the corrected wording at 20:18 [8 20:18].
6. **G2 PASS (read 22:55): the re-distilled student keeps the teacher's fix.**
   - At ×0.7 L090 the pick completes 0.92 / 0.95. In-pin c28000 completes **0.58 / 0.66**†, and TDR 0.94 / 0.92. Paired against TDR, the pick is +0.005 pooled (`g2.json` diagnostic).
   - S: all five stress cells pass every line on both seeds. Every survival line is 1.00, and the carve-out was not used (`g2.json`).
   - N: pooled completion against in-pin c28000 is −0.005 at L000, +0.045 at L090 and +0.100 at L135 (m 0.050 / 0.050 / 0.051). No cell was class-ambiguous, so no relaunch ran; es3 0 / 4.
   - WATCH, not deciding: R against c28000's Task 4 block raises no flag (field_hard 0.94 against 0.89). The pick passes every line at T02 ×0.85 L090 (0.99 / 0.96) and at T13 150 kg (0.93 / 0.96). There the sweep's c28000 had 0.87 / 0.85 and 0.86 / **0.73** (cross-pin).
   - Closing: both rows mismatched, and both are recorded as class flips. field_light is one under §4 as amended at 20:12. The suite row is one under the user's 23:06 decision to judge multi-wave rows on wave 0. Both rule decisions came after their data were read [8 22:55], [8 23:06].
7. **Wall-clock time:**
   - ≈ 29.5 h from the first Kit check (2026-11-07 17:26) to the G2 read (2026-11-08 22:55). ≈ 26.4 h of it passed before G2's rows started (19:49).
   - 2.4 h of that was the GPU sitting idle while the bridge decision waited (17:24–19:49).
   - The distill was the longest single step (9.6 h). G2's rows took 3.1 h.
8. **Nothing is pushed.** On 2026-11-09 the user approved the veto clip and the local record commit (§11 decisions 4 and 6), and both are done. Adopting TDR or the pick, the bake (SN/SH re-pairing) and the push are the user's calls (§11).

## 2. Verdict table

| Step | Verdict | Deciding numbers | Source |
|---|---|---|---|
| Kit checks K1 / K2, CPU replay | PASS | K1 9 / 9 (knob-off cfg dumps byte-identical to main in 5 cases); K2 85 / 85; replay exact | `k1.out`, `k1.reread.out`, `k2.out`, `replay.md` |
| Old-T\* bridge | PASS | 6 / 6 cell-seeds bitwise equal to the sweep on the first launch | `bridge_TSTAR.json` |
| TDR / TCTL training | PF1–PF14 PASS | TDR `[DR]` sha16 as predicted; TCTL has no `[DR]` line | `runs/RSD_TDR_s19.json`, `runs/RSD_TCTL_s19.json` |
| G1 (teacher) | **PASS** | S 5 / 5 cells; N 3 / 3; R PASS (0.632); A +0.285 vs m 0.076; closing match; es3 0 / 4 | `g1.json`, `g1.md` |
| Student distill | rc 0; checks D1–D6 PASS | 28991 → 33990, 192 envs | `runs/RSD_ST_s19.check.out`, [8 12:33] |
| Selection | **SELECTED c31000** | rank c31000 > c32000 > c33000 > c33990 > c30000; 21 / 21 decisive lines; no WATCH flags | `sel.json` |
| ONNX | parity ok | max action error 1.16e-6, latent 1.34e-7 (tol 1e-4) | `onnx_PICK.json` |
| c28000 bridge | **STOP**, then a class flip under the amended rule | ×0.7 L090 es1 bitwise; L090 nominal es1 field_light 0.95 vs 0.97; corrected wording kept by the user | `bridge_C28000.json`, [8 20:12], [8 20:18] |
| G2 (student) | **PASS** | S 5 / 5 cells, no carve-out (×0.7 L090 0.92 / 0.95); N 3 / 3, pooled vs in-pin c28000 −0.005 / +0.045 / +0.100 (m 0.050 / 0.050 / 0.051); no class-ambiguous cell; es3 0 / 4; R WATCH no flag | `g2.json`, `g2.md`, [8 22:55] |
| G2 closing | mismatch on both rows; class flip on both | field_light by §4 (earliest difference at step 0, action only); suite by wave 0 (§4 multi-wave bullet) | `g2.json` closing, [8 22:55], [8 23:06] |

## 3. What ran

### 3.1 The knob (C §3.1, §4)
- **Setting:** `KRABBY_DR_ROD_SPEED=lo:hi[:ramp_steps]`, default off. It is parsed in `config/crab_hex/dr_knobs.py` and writes `actions.joint_pos.rod_speed_dr`.
- **The draw (`crab_hex_forward_task/mdp/rod_speed_dr.py`):**
  - At each env reset, the reset envs draw a scale s_e ~ U(lo_eff, hi).
  - The draw uses a private CPU generator seeded from the env seed, in env-id order.
  - Hip and knee share s_e, and s_e is constant within an episode.
  - The slew lines read an aliased (N, 1) tensor, so they change by zero lines.
- **Ramp:** lo_eff goes linearly from 1.0 to lo over `ramp_steps` env steps.
- **Off and refusals:**
  - When off, nothing is built, drawn or written.
  - The knob is refused together with `KRABBY_STRESS_ROD_SPEED`.
  - lo is floored at 0.3.
- **What the policy sees:** no observation carries the scale. The current-proxy normaliser stays at the rated constant.
- **Settings used:**
  - TDR: `0.6:1.35:12000`. 12000 env steps = 500 iterations of ramp.
  - The distill: `0.6:1.35`, with no ramp.

### 3.2 Arms

| Arm | Start | Knob | Iterations | Seed / envs | Result (sha8) | Wall-clock |
|---|---|---|---|---|---|---|
| TDR | T\* `model_25992` (alg_state counter 27300, lr 1.5e-5) | `0.6:1.35:12000` | 3000 (25992 → 28991) | 19 / 256 | `model_28991` (`03c3e472`) | 1.22 h |
| TCTL | the same | none | 3000 | 19 / 256 | `model_28991` (`72d63d51`) | 1.28 h |
| RSD_ST_s19 (student) | distilled from TDR | `0.6:1.35` | 5000 (28991 → 33990) | 19 / 192 | saves c30000 … c33990 | 9.6 h |

Both teachers used F19's `p3b.distill.env`. TDR added only the DR key: 34 KRABBY keys, p3b's 33 plus the DR key. TCTL's 33 keys equal p3b's. The argv equals the run_phases template [7 19:50], [8 00:20]. Both resumes printed `Restored algorithm state: counter 27300, learning_rate 1.5e-05` (`runs/*.json`).

### 3.3 Rows

| Block | Policy (role) | Rows | Local time | Source |
|---|---|---|---|---|
| Reference arm | old T\* (teacher) | 27 = 10 cell-seeds + 7 non-suite L000 selection rows | 11-07 17:40–19:49 | `queue.log`, [7 19:49] |
| G1 rows | TDR | 39 = 16 cell-seeds + 7 selection rows (the 4 L000 nominal rows double as selection rows) | 21:02–00:17 | [8 00:17] |
| TCTL rows | TCTL | 12 (L090 nominal, ×0.7, ×0.8 L090; es1 + es2) | 11-08 01:37–02:44 | [8 02:56] |
| Closing (cell `_Z`) | TDR | 2 (L090 nominal es1, relaunched) | 02:44–02:55 | [8 02:56] |
| Selection | 5 student candidates | 35 = 25 prescreen + 10 full block | 12:33–15:23 | [8 15:23] |
| G2 reference arm | c28000 (student) | 20 (L000 / L090 / L135 nominal, ×0.7 / ×0.8 L090; es1 + es2) | 15:23–17:13 | [8 17:24] |
| c28000 bridge relaunch | c28000 | 2 (`_RL`) | 17:13–17:24 | `rsd_gate.log` |
| G2 rows | PICK | 34 = 24 S / N + 8 WATCH + 2 closing | 19:49–22:53 | `queue.log`, `rsd_gate.log`, [8 22:55] |
| Diagnostic relaunch (`_D1`) | PICK | 4 (the closing row and the selection L090 row, es1, suite + field_light) | 23:02–23:24 | `rsd_gate.log`, [8 23:26] |

Gate names embed the checkpoint's sha8 (`RSDT_<sha8>_<cell>` for teachers, `RSDS_<sha8>_<cell>` for students), so a stamp cannot silently report another policy.

### 3.4 Pre-flight checks (C §4)
- **K1** (cfg dumps, `k1.out`, `k1.reread.out`): 9 / 9.
  - Knob off: five cases byte-identical to main `ab390e6f`. They are Teacher-v0 × `p3b.distill.env` and × `p3b.train.env`, Student-v0 × the distill env, and Student-Play × `p3b.env` and × `p3b.field_light.env`.
  - Knob on: the diff touches exactly `rod_speed_dr` and the environ key.
  - DR + stress knob: refused at cfg build. The first checker read this refusal as a FAIL because Kit exits 0 after a caught exception. The checker was fixed and the same log re-read OK.
- **K2** (dose smoke, Teacher-v0, 256 envs, 600 steps, `k2.out`): 85 / 85.
  - Off: rated peaks, worst relative error 3e-8.
  - `0.5:1.4`: the peak equals s × rated in all 707 segments (worst 1.2e-7); 459 changes, at resets only; every draw equals a CPU replica bit for bit (214 draw calls); KS D 0.084 / 0.030.
  - Ramp case: the draws lie inside [lo_eff(t), 1].
- **Replay** (`replay.md`): the gate code reproduced, exactly:
  - the sweep's reads over 91 cell-seeds;
  - Task 4's F19 selection (pick c28000);
  - F19's p3a FAIL (2.7099) and p3b PASS (0.6620) under the R wrapper.
- **CHARTER changes before any GPU row** [7 16:25]. An independent check of the CHARTER led to six amendments:
  - the carve-out margin written out in full, max(R, 3·SE_diff, |drift|·its_past_p2a/5000);
  - its_past_p2a 2999 / 5998 instead of 3000 / 6000;
  - the L135 shared-capability-limit rule for G1 and G2;
  - the 78 GB DerivedDataCache guard;
  - the post-EXTEND checkpoint `model_31990`;
  - the §3.6 note that each selection block beyond the first (≤ 2) costs ≈ 0.8 GPU-h outside the totals and triggers the stop-and-ask above 35 GPU-h.

  The campaign scripts then had an independent check against §3.3–3.6. It found one blocking defect and six minor gaps, all fixed and tested [7 17:25]. These are the only rule changes made before data. §7 covers the two made after data were read.
- **Tests:** `test_rod_speed_dr_campaign.py` has 74 tests. With the related suites: 1098 passed, 10 skipped, and 2 failed. The 2 failures are the R18 tests broken by the disk deletion (see below) [7 17:25].
- **PF12 premise:** between p3b's training tree and main, only `observations.py` (the `crab_hex_ref` lookup) and `play_stream_video.py` differ. Neither changes a cfg value [7 17:35].
- **Disk:** on the user's decision, the R18 and stepC `raw/` npz dirs were deleted: 63 + 69 GB, free space 51 → 175 GB [7 15:55].
  - Side effect: two tests in `tests/unit/test_task4_pipeline.py` replay R18 stamps and now fail [7 16:25].
  - A test-only skip-guard fix is commit `96028b98` on branch `claude/nostalgic-bouman-546381`. It is not merged and not pushed (decision 7).

### 3.5 Launch and dose checks
Every stamp is checked for:
- the pin and a clean tree;
- `checkpoint_sha256`, role and task equal to the row's policy;
- no `KRABBY_DR_*` key in the eval env;
- dose: max |rod_speed| from the raw npz equals s × (0.028, 0.033) m/s, ± 1e-6.

All 137 rows before G2 and all 34 G2 rows passed on the first launch (`rsd_gate.log`, [8 22:55]). The provenance checkers re-checked:
- all 66 G1 stamps [8 00:30];
- the 14 newer stamps, with the dose in all 7,700 episode files [8 03:10];
- the 35 selection stamps and 11 TDR reference stamps [8 15:35].

The G2 stamps are covered by `g2-verify` (§8.4).

### 3.6 Bridges
A bridge is a code-path check only. A cell-seed that the sweep also ran must reproduce the sweep's stamp bitwise, i.e. all 18 line values equal, on the first launch or on one relaunch (C §3.3).
- **Old T\*** (`bridge_TSTAR.json`): PASS. All six cell-seeds (L090 nominal, ×0.7 and ×0.8 L090, es1 + es2) are `bitwise_first`. The in-pin references therefore equal the sweep's numbers [7 19:49].
- **c28000** (`bridge_C28000.json`): ×0.7 L090 es1 is `bitwise_first`. L090 nominal es1 matched on its suite row but not on its field_light row, on launch and on relaunch: STOP. §7 covers what followed.

### 3.7 Durations (wall-clock)

| Phase | Local time | Hours | Source |
|---|---|---|---|
| K1 + K2 | 11-07 17:26–17:32 | 0.1 | [7 17:26–17:32] |
| Old-T\* reference arm | 17:40–19:49 | 2.15 | `queue.log` |
| TDR fine-tune | 19:49–21:02 | 1.22 | `runs/RSD_TDR_s19.json` (wall_s 4395) |
| TDR G1 rows | 21:02–00:17 | 3.25 | `queue.log` |
| TCTL fine-tune | 11-08 00:20–01:37 | 1.28 | `runs/RSD_TCTL_s19.json` (wall_s 4620) |
| TCTL rows + TDR closing | 01:37–02:55 | 1.31 | `queue.log` |
| Student distill | 02:57–12:32 | 9.6 | [8 12:33] |
| Selection | 12:33–15:23 | 2.83 | `queue.log` |
| ONNX export + parity (CPU) | 15:24 | < 0.01 (5.6 s) | `onnx_PICK.json` |
| c28000 reference arm | 15:23–17:13 | 1.85 | `queue.log` |
| c28000 bridge + relaunch | 17:13–17:24 | 0.18 | `queue.log` |
| GPU idle, decision pending | 17:24–19:49 | 2.42 | `queue.log`, [8 19:49] |
| G2 rows (stage `g2cf`) | 19:49:41–22:53:34 | 3.07 | `queue.log` |
| G2 read (stage `read_g2`, CPU) | 22:54:34–22:55:06 | < 0.01 (32 s) | `queue.log` |
| Diagnostic relaunch | 23:02–23:24 | 0.36 | `rsd_gate.log`, [8 23:26] |

One row takes ≈ 0.09 h (20 reference rows in 1.85 h; G2's 34 rows in 3.07 h), so one cell-seed takes ≈ 0.18 h.

## 4. G1 — the teacher gate

**Rules (C §3.3).** G1 ran on the teacher role on `Isaac-Crab-Hex-Omni-Student-Play-v0`. PASS needs all four reads:
- **S, stress:** every line on both seeds at ×0.7 / ×0.8 L090 and at the three band edges.
  - At ×0.7 / ×0.8 only, a tracking line may also pass by the CAPABILITY-LIMITED carve-out, against in-pin old T\*.
- **N, nominal:**
  - Per-seed lines at L000 / L090 / L135. L135 completion is WATCH only.
  - Pooled completion non-inferior to in-pin old T\* by m.
  - At L135, a per-seed line other than completion that old T\* also fails on the same seed is a shared capability limit.
- **R, regression:** the v1.2 paired 0-kg statistic against in-pin old T\*, with its_past_p2a 2999. Its R and drift are reused from Task 4's Walk-Play calibration.
- **A, attribution:** TDR − TCTL at ×0.7, pooled, ≥ m.

TCTL ran only because TDR beat old T\* at ×0.7 by ≥ m (the lazy-control trigger).

### 4.1 field_light completion per arm (es1 / es2; pooled)

| Cell | TDR | TCTL | old T\* | Source |
|---|---|---|---|---|
| ×0.7 L090 | 0.94 / 0.92 (0.930) | **0.71 / 0.58** (0.645)† | **0.59 / 0.61** (0.600) | `g1.json`, `bridge_TSTAR.json` |
| ×0.8 L090 | 0.95 / 0.96 (0.955) | 0.94 / 0.92 (0.930)† | 0.87 / **0.84** (0.855) | same |
| nominal L000 | 1.00 / 0.99 (0.995) | — | 1.00 / 1.00 (1.000)† | `g1.json` N_pooled |
| nominal L090 | 1.00 / 0.99 (0.995) | 1.00 / 1.00 (1.000)† | 0.98 / 0.96 (0.970) | same |
| nominal L135 | 0.98 / 0.94 (0.960) | — | 0.89 / 0.86 (0.875) | `g1.json` watch |
| ×0.7 L000 (edge) | 0.99 / 1.00 | — | — | `g1.json` |
| ×1.35 L000 (edge) | 1.00 / 1.00 | — | — | `g1.json` |
| ×1.35 L090 (edge) | 1.00 / 1.00 | — | — | `g1.json` |

Old T\*'s ×0.8 es2 value of 0.84 fails per seed but would pass pooled (0.855). The read applies the completion line per seed, as the sweep did, which is stricter than v1.2's pooled definition [8 00:30].

### 4.2 Stress cells at L090, per line (es1 / es2)

| Cell · arm | survival.diag | survival.move_turn | tracking.diag | tracking.side | pitch_rms suite / field | h_meas.suite (m) | turn_fc | Failing lines |
|---|---|---|---|---|---|---|---|---|
| ×0.7 · TDR | 1.00 / 1.00 | 1.00 / 1.00 | 0.078 / 0.080 | 0.138 / 0.131 | 0.028 / 0.054 (both seeds) | 1.100 / 1.100 | 0.944 / 0.947 | none |
| ×0.7 · TCTL† | **0.75 / 0.78** | 0.995 / 0.985 | 0.217 / 0.224 | 0.096 / 0.097 | 0.053 / 0.083 (es1), 0.059 / 0.088 (es2) | 1.073 / 1.072 | 0.914 / 0.907 | **completion, survival.diag (both seeds)** |
| ×0.7 · old T\* | **0.79 / 0.79** | 0.895 / 0.930 | 0.226 / 0.249 | 0.131 / 0.128 | 0.053 / 0.095 (es1), 0.051 / 0.096 (es2) | 1.072 / 1.073 | 0.936 / 0.933 | **completion, survival.diag (both seeds)** |
| ×0.8 · TDR | 1.00 / 1.00 | 1.00 / 1.00 | 0.049 / 0.048 | 0.078 / 0.083 | 0.024 / 0.052 (es1), 0.023 / 0.053 (es2) | 1.106 / 1.107 | 0.954 / 0.954 | none |
| ×0.8 · TCTL† | 0.97 / 0.99 | 1.00 / 1.00 | 0.104 / 0.110 | 0.022 / 0.025 | 0.042 / 0.070 (es1), 0.044 / 0.071 (es2) | 1.080 / 1.080 | 0.920 / 0.922 | none |
| ×0.8 · old T\* | 0.98 / 0.99 | 0.990 / 1.000 | 0.063 / 0.062 | 0.049 / 0.047 | 0.045 / 0.078 (es1), 0.046 / 0.080 (es2) | 1.078 / 1.078 | 0.967 / 0.965 | **completion es2 (0.84)** |

Sources: TDR from `g1.json`, old T\* from `bridge_TSTAR.json`, TCTL †. TCTL's ×0.7 survival.back es1 is 0.97†. Every other survival line not shown is ≥ 0.99 for all three arms.

**Where the falls go (×0.7)** [8 00:30], [8 03:10]:
- TDR's falls drop in exactly the holds where old T\* fails: diag 34 → 3, move_turn 19 → 3, back 17 → 7.
- The extra training alone (TCTL) fixed move_turn (19 → 1) but not diag or back (30 / 21). So A's +0.285 comes from diag and back.
- TDR is faster than TCTL on diag, fwd, back and move_turn. Its completions do not come from walking slower.
- Side walking at ×0.7 is the one place TDR is not better than old T\*: |r − 1| 0.138 / 0.131 against 0.131 / 0.128, inside the 0.25 line.
- TDR's per-hand side ratios at ×0.7 are lat_l 0.826 / 0.825 and lat_r 0.907 / 0.904 (`g1.json` watch). The worst hand is at |r − 1| 0.175.

### 4.3 Nominal cells, per line (es1 / es2)

| Cell · arm | tracking fwd / side / back / diag | pitch_rms.suite | h_meas.suite (m) | turn_fc |
|---|---|---|---|---|
| L000 · TDR | 0.039 / 0.047 / 0.018 / 0.006 (es1); 0.039 / 0.048 / 0.019 / 0.007 (es2) | 0.026 / 0.026 | 1.117 / 1.117 | 0.976 / 0.975 |
| L000 · old T\*† | 0.012 / 0.061 / 0.012 / 0.014; 0.007 / 0.060 / 0.008 / 0.010 | 0.030 / 0.031 | 1.089 / 1.089 | 0.986 / 0.985 |
| L090 · TDR | 0.048 / 0.021 / 0.013 / 0.001; 0.048 / 0.026 / 0.014 / 0.001 | 0.026 / 0.025 | 1.114 / 1.114 | 0.972 / 0.972 |
| L090 · TCTL† | 0.042 / 0.061 / 0.086 / 0.003; 0.045 / 0.060 / 0.081 / 0.001 | 0.035 / 0.035 | 1.089 / 1.089 | 0.942 / 0.941 |
| L090 · old T\* | 0.031 / 0.066 / 0.020 / 0.033; 0.030 / 0.064 / 0.021 / 0.031 | 0.035 / 0.032 | 1.085 / 1.084 | 0.987 / 0.989 |
| L135 · TDR | 0.048 / 0.005 / 0.014 / 0.011; 0.043 / 0.002 / 0.017 / 0.007 | 0.025 / 0.025 | 1.111 / 1.112 | 0.971 / 0.969 |
| L135 · old T\*† | 0.038 / 0.056 / 0.027 / 0.044; 0.039 / 0.056 / 0.027 / 0.045 | 0.035 / 0.034 | 1.082 / 1.083 | 0.988 / 0.986 |

Every nominal line passes for all three arms. TCTL's L090 back tracking (0.086 / 0.081) and turn_fc (0.942) moved away from old T\*'s, inside the lines. So the no-DR continuation is not a neutral copy of T\* at nominal either (descriptive, no statistic).

### 4.4 Band edges (TDR only; no reference, no carve-out)

| Cell | Completion | tracking.side | tracking.diag | pitch_rms.field_light | turn_fc |
|---|---|---|---|---|---|
| ×0.7 L000 | 0.99 / 1.00 | 0.117 / 0.118 | 0.090 / 0.088 | 0.052 / 0.052 | 0.952 / 0.955 |
| ×1.35 L000 | 1.00 / 1.00 | 0.107 / 0.106 | 0.057 / 0.052 | 0.058 / 0.057 | 0.985 / 0.982 |
| ×1.35 L090 | 1.00 / 1.00 | 0.094 / 0.098 | 0.057 / 0.056 | 0.057 / 0.057 | 0.976 / 0.977 |

All 18 lines pass on both seeds, and survival is 1.00 throughout (`g1.json`). The fast side shows lateral overshoot only (side ratios 1.07–1.12, `g1.json` watch), as in the sweep.

### 4.5 The reads (`g1.json`, `g1.md`)
- **S: PASS.** All five cells pass every line on both seeds, and the carve-out was not used. At ×0.7 L090, TDR's suite tracking error is larger than at nominal (side 0.138 / 0.131, diag 0.078 / 0.080), but inside the lines.
- **N: PASS.**
  - The per-seed lines pass at all three loads. No shared capability limit was needed. L135 completion (WATCH): TDR 0.98 / 0.94, old T\* 0.89 / 0.86.
  - Pooled completion against old T\*:

    | Load | TDR | old T\* | Diff | m | Binds |
    |---|---|---|---|---|---|
    | L000 | 0.995 | 1.000 | −0.005 | 0.05 | floor |
    | L090 | 0.995 | 0.970 | +0.025 | 0.05 | floor |
    | L135 | 0.960 | 0.875 | +0.085 | 0.05 | floor |

- **R: PASS.** 26 decisive rows plus 10 WATCH rows, its_past_p2a 2999. The closest decisive rows by margin used:
  - fwd track_err, suite es2: TDR |r − 1| 0.039 against 0.007; 0.632 of R = 0.051.
  - fwd track_err, suite es1: 0.528.
  - field_turn completion: 0.99 against 1.00; 0.478.
  - field_light completion: 0.995 against 1.000; 0.332.

  side22 is FAST / FAST on both seeds, against FAST / FAST, so no FAST hand was lost. field_hard completion rose from 0.87 to 0.96. ring_obst completion is 1.0 against 1.0 with a zero margin, so one lost ring_obst episode by a later TDR head would fail R. That is the pre-registered statistic [8 00:30].
- **A: PASS.** TDR − TCTL at ×0.7, pooled: 0.930 − 0.645 = +0.285; SE_diff 0.038, m 0.076 (SE-bound).
  - Episode pairs: 121 both complete, 65 TDR only, 8 TCTL only, 6 neither.
  - m stays ≤ 0.0763 across 20 bootstrap seeds [8 03:10].
- **Trigger and half-gap clause:**
  - TDR − old T\* at ×0.7 = 0.930 − 0.600 = +0.330 against m 0.075. Pairs: 114 both, 72 TDR only, 6 old only, 8 neither [8 00:30].
  - The half-gap clause also holds: 0.93 ≥ 0.725, where 0.725 = old + ½(0.85 − old).
  - EXTEND was unreachable because S passed.
- **Closing:** TDR's relaunch of L090 nominal es1 (cell `T00_nominal_L090_Z`) reproduces all 18 line values. All 1,100 of its raw npz are byte-identical to the first launch's [8 03:10].
- **es3:** 0 of 4 used.

### 4.6 TCTL minus old T\*
- **×0.7:** 0.645 − 0.600 = +0.045 against m = 0.088. The no-DR continuation did not move the cell by m (`g1.json` `tctl_vs_old_x07`). This clause would have bound only after an EXTEND.
- **×0.8:** 0.930 − 0.855, which beats old T\* by m (0.061) on its own. TDR − TCTL there is +0.025, so "attributable to the DR" holds at ×0.7 only, the cell A tests [8 03:10].
- **L090 nominal (WATCH):** TCTL passes every per-seed line, so the continuation did not collapse.
- **Training** [7 21:02], [8 03:10]:
  - TDR's failure rate fell to ≈ 0.18 under the DR. TCTL's plateaued at ≈ 0.28–0.30, with std and entropy creeping up.
  - Final adaptive LR: TDR 1.5e-5, TCTL 1e-5. Both restored 1.5e-5 and averaged the same applied LR.

TDR training against p3b over the same iteration offsets (tensorboard; an observation, not a gate) [7 21:02]:

| Offset (its) | p3b reward / ep. length / crab_failure | TDR |
|---|---|---|
| 100–500 | 58.4 / 1673 / 0.325 | 58.0 / 1638 / 0.308 |
| 1000–2000 | 58.6 / 1658 / 0.328 | 57.3 / 1665 / 0.310 |
| 2500–3000 | 58.3 / 1661 / 0.331 | 60.6 / 1771 / 0.223 |

The `DR/rod_speed_lo` trace followed the ramp: 1.0 → 0.6 by iteration 26492, then flat. The late-window mean was 0.9796 against a midpoint of 0.975, and min / max reached 0.60 / 1.35 (PF14, [7 21:02]).

### 4.7 Adversarial checks
- **TCTL-DUE read** [8 00:30]:
  - The independent recompute matched `g1.json` to 1e-12.
  - The rule check and the provenance check of 66 stamps were clean.
  - The refuter found that TDR and old T\* differ only in the checkpoint within each cell-seed, and that no episode was dropped.
- **PASS read** [8 03:10]:
  - The recompute matched bit for bit, and paired episodes start from identical roots and terrain.
  - PASS is the table's verdict.
  - Provenance was clean. TCTL's params differ from TDR's only in the DR block and the run name.
  - The refuter could not refute.
- **Record-only notes** (none changes a verdict):
  - ring_obst R has zero margin.
  - The closing row has no verdict effect at G1.
  - TCTL's ×0.8 rows enter no statistic.
  - The per-seed completion line is stricter than pooled.
  - Part of TDR's nominal gain may be continued training.
  - TDR stands ≈ 0.03 m taller and halves pitch_rms at ×0.7. These come with the DR, since TCTL stands like old T\*.
  - A rests on one training seed per arm.
  - The L000 stamps carry no payload record. Zero payload was confirmed from body mass (158.76 kg).
  - One selection row log has two benign Kit plugin-load `[Error]` lines.

## 5. Student distill and selection

### 5.1 Distill and the +10-iteration checks
- **Launch:** `scripts/rsd_distill.sh --name RSD_ST_s19 --checkpoint <TDR model_28991.pt> --teacher-policy TDR`.
  - Env `envs/p3b_dr.distill.env` (sha256 `8ba683b4dad7…`) = F19 `p3b.distill.env` + `KRABBY_DR_ROD_SPEED=0.6:1.35`.
  - Seed 19, 192 envs, `Isaac-Crab-Hex-Omni-Student-v0`, iterations 28991 → 33990 (the distill log `RSD_ST_s19.log` named in `runs/RSD_ST_s19.distill.json`, and `scripts/rsd_distill.sh` for the task).
  - No other rider: the student stage reads T01 alone.
- **Checks D1–D6 at iteration 29007, all PASS; re-run PASS at 29070** (`runs/RSD_ST_s19.check.out`):
  - D1: the knob lines equal `T4F19_st_a2`'s plus the DR line.
  - D2: the `[DR]` line equals the predicted draw, sha16 `13857bf91a92b80d` [8 03:10].
  - D3: teacher sha `03c3e472` = TDR.
  - D4: first iteration 28991.
  - D5: TDR's tail read: ep_len 1765.8, crab_failure 0.221.
  - D6: the live rules are clean.
- **End:** `TPDISTILL END RSD_ST_s19 rc=0` at 12:32:33, pace 6.9 s per iteration [8 02:57], [8 12:33].
- **Training trace** (depth_actor_loss / crab_failure, a record and not a read): 29000 1.41 / 0.10; 30000 0.66 / 0.24; 31000 0.59 / 0.21; 32000 1.13 / 0.40 (one noisy iteration); 33000 0.60 / 0.18; 33990 0.61 / 0.20. The loss flattened by ≈ 30500 [8 12:33].

### 5.2 Candidates and prescreen (`sel.json`)
Task 4's `sel` rules are reproduced line for line from `pipeline.py:2244-2410`, with two §3.4 sanctions: the level rows are set to 90 kg explicitly, and TDR's G1 stamps serve as the teacher reference rows.
- **Candidates:** the save nearest to 28991 + 1000 … 5000.
- **Prescreen:** 5 rows per candidate: suite@1, field_hard@1/2, and suite@1 + field_light@1 at 90 kg.
- **Rank:** misses, then −field_hard, then mean_abs_dev, then iteration, with no tolerance band.

| Candidate | Target | Misses | field_hard | mean_abs_dev | Rank |
|---|---|---|---|---|---|
| c30000 | 29991 | 0 | 0.905 | 0.022 | 5 |
| c31000 | 30991 | 0 | 0.940 | 0.022 | 1 |
| c32000 | 31991 | 0 | 0.930 | 0.044 | 2 |
| c33000 | 32991 | 0 | 0.920 | 0.029 | 3 |
| c33990 | 33991 | 0 | 0.915 | 0.022 | 4 |

### 5.3 c31000's full block (10 more rows; PASS on the first try)
- **Floor lines (16):** survival is 1.00 on both seeds in fwd, side, back and diag. Speed ratio, es1 / es2, against a line of ≥ 0.7: fwd 1.033 / 1.037, side 1.041 / 1.041, back 1.013 / 1.015, diag 1.002 / 0.998. side22 is FAST / FAST.
- **Completion against TDR**, paired, with margin max(0.05, 2·SE_diff). All four bind at the 0.05 floor:

  | Row | c31000 | Line | TDR |
  |---|---|---|---|
  | field_hard | 0.94 | 0.91 | 0.96 |
  | field_light | 0.99 | 0.945 | 0.995 |
  | ring_obst | 1.00 | 0.95 | 1.00 |
  | field_turn | 0.995 | 0.94 | 0.99 |

- **WATCH:**
  - Stable at 90 kg on both seeds: 35 level lines pass, with pooled field_light completion 0.995.
  - The tracking watch (20 rows, +0.05 against TDR) raised no flags.
  - No further block ran, and there was no BUDGET-ASK.

### 5.4 Why c31000
- Every candidate had 0 misses, so field_hard decided the rank. c31000 led c32000 by 0.94 against 0.93, i.e. 2 of 200 episodes.
- Task 4 selects the first full-block PASS in rank order, and c31000's block passed.
- F19's own first place also came down to one episode [8 15:35].
- The runner-up c32000 has the worst tracking deviation of the five (0.044).
- At 90 kg no later candidate looks like a better student. c31000 has 0 / 1000 suite falls and 100 / 100 field_light completions; c33990 tracks marginally closer (mean |r − 1| 0.016 against 0.020) [8 15:35].

### 5.5 Selection check (`sel-verify`, 4 / 4 confirm, [8 15:35])
- **Recompute:** every `sel.json` value matches to 1e-9 from the stamps. That covers the prescreen, rank, 21 decisive lines, 35 level lines and 20 tracking-watch rows. The five checkpoint hashes are right.
- **Rules:** `ph_sel` reproduces `pipeline.py` ex_select / full_block line for line, apart from the two sanctions, the budget check and the g1-PASS guard.
- **Provenance:** all 35 selection stamps and 11 TDR reference stamps are clean.
- **Refuter:** could not refute. Student and TDR rows pair episode for episode.
- **Record-only notes:**
  - The tightest margin is field_hard: 0.94 against 0.91. Its discordant pairs are 8 student only and 12 TDR only.
  - `rows=40` in `RSD END sel` counts row keys; there are 35 distinct stamps.
  - The `at` times in `sel.rows.json` are record times.
  - The distill record's `launcher_argv` omits `--teacher-policy`. The flag is mandatory, so it was passed.
  - §3.6's BUDGET-ASK example does not match the code. This is moot.
- **Veto clip, run after G2:** Task 4's post-selection headless veto clip, "never a gate" (`pipeline.py` 2315–2317). For c28000 it produced `F19/clip/student_c28000_field_hard.mp4` (`REPORT_baseline_vs_final.md`, Student section). It needs the GPU, so it could only run after G2. It ran on 2026-11-09 at the user's yes (decision 4): `clip/student_c31000_field_hard.mp4`, made by `scripts/clip_pick.sh`.

## 6. ONNX (`onnx_PICK.json`, 15:24, CPU)
- **Export:** Task 4's exporter (`omni_student_export.export_omni_student`) via `scripts/rsd_onnx.py`, into `parkour/outputs/omni_plan/rod_speed_dr/onnx/PICK_c31000/` (main checkout, untracked).
- **Files (sha256 prefix):** `policy.onnx` `ac0ea3e2`, `depth_latest.onnx` `0a04e065`, `policy.pt` `94aa2833`, `depth_latest.pt` `c5d6e8c7`, `omni_student_io.json` `4ee03d02`. The io file's sha256 prefix equals Task 4's c28000 export (`REPORT_baseline_vs_final.md`, Student section), so the deploy interface is unchanged.
- **Parity:** the sweep's `onnx_parity.py`, unchanged, on the sweep's recorded G0 rollout (`G0_record.npz`, sha256 `0012cd155099…`), runtime `onnx.reference`.
  - 120 steps and 25 depth encodes.
  - Max action error 1.16e-6, raw action error 1.43e-6, latent error 1.34e-7, against a tolerance of 1e-4: **parity ok**.
- **Tool check** [7 18:10]:
  - On c28000: 1.07e-6 / 4.0e-7, with the io file identical to `policy/f19_9_student/`.
  - Negative control (c28000's graphs against c27000's `.pt`): action error 0.343, FAIL as it should.
- **Limit:** this checks the export on one CPU record. It does not test the deploy runtime or the robot's timing.

## 7. The c28000 bridge, the class flip and the §4 changes

### 7.1 What a class flip is
The Isaac eval has two bitwise-repeatable outcome classes that depend on process state. Two launches of the same row with the same code, cfg and seed can land in different classes, worth ≈ ±0.04 completion (sweep REPORT §8). A **class flip** is a mismatch between two stamps of the same row caused by the eval landing in the other class, with no code change behind it. The bridge exists to tell a class flip apart from a code-path difference, which would invalidate every in-pin reference.

### 7.2 What the bridge found [8 17:24]
- **×0.7 L090 es1:** bitwise equal to the sweep on the first launch.
- **L090 nominal es1:** the suite row matched on every line value. The field_light row differed on the first launch and identically on the one relaunch (`bridge_C28000.json`):

  | field_light line (L090 nominal es1) | New pin: launch = relaunch | Sweep stamp (2026-11-06) |
  |---|---|---|
  | completion | 0.95 | 0.97 |
  | h_meas | 1.0876 | 1.0881 |
  | pitch_rms | 0.0685 | 0.0677 |
  | roll_rms | 0.0307 | 0.0305 |

- **Stamp comparisons:**
  - The launch and the relaunch are bitwise identical in all 100 episodes.
  - Both are bitwise identical to Task 4's own stamp of the row (`task4/F19/lr/T4F19_sel_c28000_L090/field_light__omni/seed001/2026-10-23_14-41-05`, completion 0.95, the Task 4 record).
  - The sweep's stamp differs from all three.
- **Verdict:** STOP. The queue stopped before the pick's rows.

### 7.3 Evidence (`c28k-bridge-diagnose`, 4 / 4 class flip, [8 18:00])
- **Cfg:** identical. That covers run_meta's env_vars, env_overrides, payload, body masses, checkpoint and student block, and the env-file contents. K1's C5 (Student-Play × field_light env, knob off) is byte-identical to main at 0 kg. No L090 dump exists for either pin.
- **Dose:** rated in every episode of every stamp.
- **Step 0:** state and observations are identical everywhere.
  - In envs 26, 43, 54 and 81, the step-0 action takes one of exactly two bitwise values (deltas 1.0e-6 to 1.45e-5).
  - Both values occur under one clean code: the known same-code split S81V0_base / S81V0_base2 (code `6a4e4c6e`, 2026-10-27), in the same four envs with the same deltas.
  - The new pin and Task 4 hold one value. The sweep's stamp and Task 4's own L000 row hold the other.
- **After step 0:**
  - Differences are action-only at depth-encode steps (every 5th step). They reach the state about 2 steps later, through the 1-step action delay.
  - At step 48, 50 envs' states move at once through the shared physics scene. The known split does the same at step 165 (11 envs).
- **Code between the pins:** `44c8165a → 1682e2e1` is the DR commit only, additions only. Its knob-off path builds nothing and draws nothing.
- **All c28000 stamps:** across pins, every c28000 stamp falls into one of two classes, and same-class stamps of a cell-seed match bitwise (9 groups, no exception).

### 7.4 Why the pinned rule said stop
The 2026-11-07 text, quoted in [8 20:12]: "a mismatch is a class flip when the cfg dumps and the per-stamp dose are identical and the first divergence is an action delta below 1e-5 at a depth-frame step (the known process-state signature). A divergence at the first step, in the dose or in a cfg dump is a code-path difference, and it stops the campaign."

Read literally, this case fails it in two ways:
- The first divergence is at step 0.
- Five first deltas exceed 1e-5 (max 3.3e-5). The 17:24 entry's count of "4" was corrected at [8 20:12], as was its "3 episodes first differ at step 0", now 4.

The same clauses would also call the known same-code split a code-path difference (15 episodes above 1e-5, max 2.3e-4). §4 had mis-transcribed its own reference signature [8 18:00]. The rules checker agreed that amending a verdict clause after its data were read is the user's decision. The options were (a) amend and run G2, or (b) end before G2. Nothing ran for 2.4 h [8 18:00], [8 19:49].

### 7.5 The amendment [8 20:12]
- **Run before the decision:** at 19:49 the 34 G2 rows were started before the decision, because the rows do not depend on it and only their read does. The `--bridge-class-flip` flag then meant "rows started pending the decision". Nothing was read before the amendment was logged [8 19:49], [8 20:12].
- **The user's choice:** option (a). The approved text added a bridge bitwise-match condition to the 18:00 proposal.
- **The rule now in §4:** a mismatch between two stamps of the same row is a class flip when all of these hold:
  - the cfg dumps and the per-stamp dose are identical;
  - the step-0 state and observations are identical in every episode;
  - the earliest difference anywhere in the stamp is in the action alone and at a depth-frame step (step 0 included), with no size bound;
  - at the c28000 bridge only, which compares two code pins: the new stamp also matches another recorded stamp from an earlier launch of the same policy and row (same cfg dump, dose and eval seed), bit for bit in every raw-npz array of every episode.

  Later differences may take any form. Anything else is a code-path difference and stops the campaign.
- **What changed, and why:**
  - Step 0 now counts as a depth-frame step, as the recorded signature lists it.
  - The 1e-5 bound is gone: both splits reach 1.45e-5 at step 0.
  - Identical step-0 state and observations are now stated outright.
  - The bitwise-match condition is new. A code change that only nudged the depth network's arithmetic would show the same signature, but it could not reproduce an older stamp exactly. G2's relaunch and closing compare launches under one pin, whose launch checks already exclude a code change.
- **The executor's correction:** while writing the rule in, the executor changed one clause of the approved text. The action-only condition now applies to the **stamp's earliest difference**, not to each episode's first difference.
  - The per-episode version shown to the user fails this case: some episodes first differ at step 48, in the physics state, when 50 envs move at once. The 18:00 proposal, read per episode, fails it too.
  - The user was told with the amendment and kept the corrected wording at 20:18 ("Keep it") [8 20:18].
- **Applied to the c28000 bridge:**
  - Cfg and dose are identical across the four stamps (launch, relaunch, sweep, Task 4).
  - Step 0 is identical.
  - The earliest difference is at step 0, action only, in envs 26, 43, 54 and 81, with the known split's values.
  - The launch and the relaunch equal Task 4's 2026-10-23 stamp in every array of all 100 episodes.
  - Verdict: **class flip**, recorded in §6. G2 proceeded under `--bridge-class-flip`. G2's closing was judged by the amended rule (§7.6, §8.4); no relaunch was due.

### 7.6 The multi-wave clarification [8 22:55], [8 23:06]
- **What the literal text gave:** G2's closing suite row (`RSDS_155b5981_T00_nominal_L090_Z` against the pick's selection row `RSDSEL_155b5981_L090`, es1) mismatched. Tested as one stamp, it fails two of the 20:12 conditions:
  - the step-0 state is not identical in every episode;
  - the earliest difference includes `applied_torque` and `foot_force_norm`, not the action alone.

  Under §4 as amended at 20:12 that is a code-path difference, which stops the campaign (`classflip_check.py` verdict CODE-PATH).
- **Why no flipped suite row could meet it:** a suite row runs 10 waves of 100 envs one after another in one process.
  - Before each later wave the harness re-seeds, clears the policy's GRU state and latent, and calls `env.reset()`.
  - `env.reset()` zeroes the action delay and the observation history. It does not refresh the articulation's applied torque, the contact sensor's force history or the actuator's rod speed before the wave's first physics step. The policy's current-sense observation reads that torque and rod speed at the wave's first step, and the harness records torque and foot force before that step (`crab_hex_forward_task/mdp/observations.py` 118–142; `eval_crab_hex_gait.py` 1540–1600, 1757–1781). The 22:55 and 23:06 entries first named the step counter instead; `g2-verify` corrected that, and the rule did not change [8 23:26].
  - So wave 0's divergence reaches every later wave's step 0. In waves 1–9 all 900 episodes differ at their own step 0 in `applied_torque` and `foot_force_norm`, and 271 of them in the action too. Every kinematic state array is bit-equal there (`wave_diff.py`) [8 23:26].
  - c28000's own cross-class suite pairs show the same signature: in-pin against the sweep at ×0.8 L090 es1 and L000 es2, and against Task 4 at L135 es2. Read literally, the 20:12 clause would have made every cross-class suite pair a code-path stop [8 23:26].
  - Any suite row whose wave 0 flips class therefore fails the "every episode" clause.
- **The user's choice (23:06):** option 1, judge wave 0 (the user answered "1"). Option 2 was the literal reading: a code-path difference, and the campaign stops.
- **Text added to §4:** the multi-wave bullet. The conditions apply to the row's first wave (wave 0), which starts the process. Each later wave starts from the previous wave's end state, so any difference in a later wave is a later difference, one at that wave's own step 0 included. Nothing else in §4 changed. The bullet's account of why later waves inherit the state was corrected at 23:26 (above); its rule was not.
- **When:** like the 20:12 amendment, this was the user's decision after the data it governs (the G2 closing) were read [8 23:06].

### 7.7 Class-noise caveats G2 inherits [8 18:00]
- **The reference arm mixes classes.** The in-pin c28000 arm differs per launch: L090 nominal es2 field_light and the ×0.8 L090 rows sit in the other class from the sweep's or Task 4's.
- **The swings exceed the ambiguity window.** At stressed 90-kg cells they reach 0.05–0.07 per seed: ×0.8 L090 field_light was 0.76 in the sweep against 0.83† in-pin on es1, and 0.85 against 0.80† on es2. G2's class-ambiguity window (m, m + 0.04] assumes ±0.04. At G2 no N cell came near it (§8.4), so the window was not exercised.
- **The WATCH reads are cross-class.** The sweep's L090 nominal es1 stamp was already off Task 4's class on 2026-11-06 (0.97 against the record's 0.95). G2's WATCH reads against the sweep are cross-pin and cross-class.
- **Cosmetic:** `bridge_C28000.json`'s `why` text says "G1 proceeds". `--bridge-class-flip` leaves no mark in any JSON, so "recorded" means the [8 20:12] entry.

## 8. G2 — the student gate

### 8.1 What ran
`rsd_gate.py g2 --policy PICK --bridge-class-flip` (queue stage `g2cf`, 19:49:41–22:53:34, rc 0) ran 34 rows on the student role, each ok on its first launch (`queue.log`, `rsd_gate.log`, [8 22:55]):
- **6 S / N cells for the pick, es1 + es2 (24 rows):** ×0.7 L090, ×0.8 L090, ×0.7 L000, ×1.35 L000, ×1.35 L090, and L135 nominal.
- **2 WATCH cells (8 rows):** T02 rod effort ×0.85 L090 and T13 150 kg, compared against the sweep's c28000 stamps.
- **Closing (2 rows):** the pick's L090 nominal es1, launch Z, compared against its selection row.

c28000's 20 reference rows were skipped as valid stamps (`rsd_gate.log`, 19:49; `RSD END g2 rows=54`). The pick's L000 and L090 nominal cell-seeds are its selection full-block rows (`RSDSEL_155b5981_L000` / `_L090`). `read g2 --pick PICK --tdr TDR` ran as queue stage `read_g2` at 22:54:34–22:55:06 (`queue.log`).

### 8.2 What decides it (C §3.4; `rsd_lib.g2_decide`)
- **PASS iff S and N pass.**
- **S:** every line on both seeds at the five stress cells. At ×0.7 / ×0.8 L090 a tracking line may pass by the carve-out against in-pin c28000 (its_past_p2a 0). Roll and pitch stay absolute.
- **N, per seed:** the lines at L000 / L090 / L135. L135 completion is WATCH only. At L135, a line that in-pin c28000 also fails on the same seed is a shared capability limit.
- **N, pooled:** completion must be non-inferior to in-pin c28000 by m.
  - A shortfall in (m, m + 0.04] is class-ambiguous. That cell is relaunched once on both sides (4 cell-seeds from the es3 allowance) and decided on the pooled four-launch read.
  - A shortfall beyond m + 0.04, or one that survives the relaunch, is a FAIL.
- **es3:** near-line cells as at G1, with at most 4 cell-seeds per gate, the relaunch charge included.
- **Not deciding:**
  - R (WATCH, cross-pin): the pick's full-block field_hard, ring_obst and field_turn completion and its 0-kg tracking, against c28000's Task 4 full block. Rows below c28000 − m are flagged.
  - The WATCH cells.
  - The diagnostic: the pick against TDR at ×0.7, where TDR scored 0.94 / 0.92.
- **Closing:** a mismatch is judged by the amended §4 rule. A code-path difference stops the campaign.
- **Outcome:** PASS goes to this REPORT. FAIL stops the campaign with the diagnostic recorded.

### 8.3 The reference arm (in-pin c28000, field_light completion es1 / es2)

| Cell | In-pin c28000† (pooled) | Sweep, same policy | Per-seed failing lines in-pin† |
|---|---|---|---|
| nominal L000 | 1.00 / 0.99 (0.995) | 1.00 / 0.99 | none |
| nominal L090 | 0.95 / 0.95 (0.950) | 0.97 / 0.95 | none |
| nominal L135 | 0.89 / 0.86 (0.875) | not run | none (completion is WATCH) |
| ×0.7 L090 | **0.58 / 0.66** | 0.58 / 0.66 | **completion, survival.diag (0.83 / 0.87)** |
| ×0.8 L090 | **0.83 / 0.80** | 0.76 / 0.85 | **completion (both seeds)** |

Sweep values are from the sweep REPORT §3.1. The ×0.8 cell, which split in the sweep, now fails both seeds in-pin. The ×0.7 cell is unchanged. Sweep WATCH baselines (sweep REPORT §4, §6): T02 ×0.85 L090 completion 0.87 / 0.85; T13 150 kg 0.86 / 0.73. §8.4 sets the pick beside them, descriptively and cross-pin. The pooled values in the first column are also `g2.json`'s N references.

### 8.4 Results (`g2.json`, `g2.md`)

**Verdict: PASS**, read 2026-11-08 22:55:06 (`g2.json` `read_at`; stage `read_g2`, 22:54:34–22:55:06 on CPU, `queue.log`). S passes 5 / 5 cells and N 3 / 3 loads. The read has no notes, no missing stamp and no pending branch (`notes`, `missing` and `possible` are empty).

**S, per seed (es1 / es2).** The carve-out was available at ×0.7 / ×0.8 L090 and used nowhere (`carved` is empty on every seed).

| Cell · policy | Completion | survival.diag | survival.move_turn | tracking.diag | tracking.side | pitch_rms suite / field | h_meas.suite (m) | turn_fc | Failing lines |
|---|---|---|---|---|---|---|---|---|---|
| ×0.7 L090 · PICK | 0.92 / 0.95 | 1.00 / 1.00 | 1.00 / 1.00 | 0.090 / 0.093 | 0.151 / 0.137 | 0.026 / 0.053 (es1), 0.027 / 0.054 (es2) | 1.102 / 1.103 | 0.945 / 0.944 | none |
| ×0.7 L090 · c28000 in-pin† | **0.58 / 0.66** | **0.83 / 0.87** | 0.885 / 0.900 | 0.241 / 0.244 | 0.130 / 0.136 | 0.043 / 0.095 (es1), 0.049 / 0.094 (es2) | 1.082 / 1.081 | 0.928 / 0.929 | **completion, survival.diag (both seeds)** |
| ×0.7 L090 · TDR | 0.94 / 0.92 | 1.00 / 1.00 | 1.00 / 1.00 | 0.078 / 0.080 | 0.138 / 0.131 | 0.028 / 0.054 (both seeds) | 1.100 / 1.100 | 0.944 / 0.947 | none |
| ×0.8 L090 · PICK | 0.97 / 0.96 | 1.00 / 1.00 | 1.00 / 1.00 | 0.058 / 0.060 | 0.085 / 0.085 | 0.022 / 0.050 (es1), 0.023 / 0.052 (es2) | 1.110 / 1.109 | 0.955 / 0.955 | none |
| ×0.8 L090 · c28000 in-pin† | **0.83 / 0.80** | 0.99 / 0.99 | 0.980 / 0.995 | 0.064 / 0.067 | 0.065 / 0.061 | 0.037 / 0.077 (es1), 0.036 / 0.078 (es2) | 1.084 / 1.083 | 0.946 / 0.948 | **completion (both seeds)** |
| ×0.8 L090 · TDR | 0.95 / 0.96 | 1.00 / 1.00 | 1.00 / 1.00 | 0.049 / 0.048 | 0.078 / 0.083 | 0.024 / 0.052 (es1), 0.023 / 0.053 (es2) | 1.106 / 1.107 | 0.954 / 0.954 | none |
| ×0.7 L000 · PICK | 1.00 / 0.99 | 1.00 / 1.00 | 1.00 / 1.00 | 0.099 / 0.099 | 0.127 / 0.129 | 0.024 / 0.050 (es1), 0.024 / 0.051 (es2) | 1.109 / 1.109 | 0.950 / 0.957 | none |
| ×0.7 L000 · TDR | 0.99 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 | 0.090 / 0.088 | 0.117 / 0.118 | 0.024 / 0.052 (both seeds) | 1.106 / 1.106 | 0.952 / 0.955 | none |
| ×1.35 L000 · PICK | 1.00 / 0.99 | 1.00 / 1.00 | 1.00 / 1.00 | 0.043 / 0.047 | 0.093 / 0.094 | 0.032 / 0.058 (both seeds) | 1.123 / 1.123 | 0.980 / 0.980 | none |
| ×1.35 L000 · TDR | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 | 0.057 / 0.052 | 0.107 / 0.106 | 0.031 / 0.058 (es1), 0.031 / 0.057 (es2) | 1.118 / 1.118 | 0.985 / 0.982 | none |
| ×1.35 L090 · PICK | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 | 0.043 / 0.045 | 0.090 / 0.088 | 0.031 / 0.058 (es1), 0.031 / 0.059 (es2) | 1.121 / 1.121 | 0.979 / 0.981 | none |
| ×1.35 L090 · TDR | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 | 0.057 / 0.056 | 0.094 / 0.098 | 0.032 / 0.057 (es1), 0.031 / 0.057 (es2) | 1.117 / 1.117 | 0.976 / 0.977 | none |

Sources: the pick from `g2.json` S; TDR from `g1.json` S (and `g2.json` diagnostic at ×0.7 L090); c28000 †.
- survival.side is 1.00 for all three policies in these cells (c28000†). Every survival line of the pick is 1.00 in all five cells.
- c28000 has no in-pin row at the band edges. The sweep's cross-pin completion there was 0.98 / 0.95 at ×0.7 L000, 1.00 / 1.00 at ×1.35 L000 and 0.99 / 1.00 at ×1.35 L090 (sweep REPORT §3.1).
- **What the pick changes at ×0.7 L090** (descriptive; no gate statistic compares the pick with c28000 here):
  - pooled completion 0.935 against 0.620; pairs 117 both, 70 pick only, 7 c28000 only, 6 neither†;
  - survival.diag 1.00 against 0.83 / 0.87†, and tracking.diag 0.090 / 0.093 against 0.241 / 0.244†;
  - field_light pitch_rms 0.053 / 0.054 against 0.095 / 0.094†.

  At ×0.8 L090: 0.965 against 0.815 pooled; pairs 159 / 34 / 4 / 3†.
- **Per-hand side ratios at ×0.7 L090** (WATCH, `g2.json` watch.pick): lat_l 0.812 / 0.811 and lat_r 0.901 / 0.899. The worst hand is at |r − 1| 0.189. TDR's are 0.826 / 0.825 and 0.907 / 0.904. The sweep's c28000 had 0.89 / 0.84 on es1 (sweep REPORT §3.1). So the pick's left-hand side walking is the slowest of the three at ×0.7, inside a 0.25 per-hand line.

**N, per seed (es1 / es2).** No per-seed line fails at any load, so no shared capability limit was used (`shared_limit` is empty). Every survival line of the pick is 1.00 at all three loads.

| Load · policy | Completion | tracking fwd / side / back / diag | pitch_rms.suite | h_meas.suite (m) | turn_fc | Failing |
|---|---|---|---|---|---|---|
| L000 · PICK | 0.98 / 1.00 | 0.033 / 0.041 / 0.013 / 0.002 (es1); 0.037 / 0.041 / 0.015 / 0.002 (es2) | 0.027 / 0.026 | 1.120 / 1.120 | 0.977 / 0.978 | none |
| L000 · c28000 in-pin† | 1.00 / 0.99 | 0.008 / 0.046 / 0.022 / 0.009; 0.003 / 0.047 / 0.019 / 0.012 | 0.030 / 0.029 | 1.089 / 1.089 | 0.986 / 0.986 | none |
| L090 · PICK | 1.00 / 0.99 | 0.038 / 0.017 / 0.014 / 0.009; 0.039 / 0.030 / 0.011 / 0.010 | 0.026 / 0.026 | 1.118 / 1.117 | 0.974 / 0.973 | none |
| L090 · c28000 in-pin† | 0.95 / 0.95 | 0.010 / 0.048 / 0.004 / 0.026; 0.013 / 0.050 / 0.006 / 0.028 | 0.030 / 0.031 | 1.086 / 1.086 | 0.987 / 0.988 | none |
| L135 · PICK | 0.96 / 0.99 (WATCH) | 0.036 / 0.002 / 0.011 / 0.018; 0.036 / 0.008 / 0.013 / 0.018 | 0.025 / 0.025 | 1.115 / 1.115 | 0.970 / 0.972 | none |
| L135 · c28000 in-pin† | 0.89 / 0.86 (WATCH) | 0.016 / 0.042 / 0.015 / 0.033; 0.016 / 0.044 / 0.015 / 0.031 | 0.031 / 0.032 | 1.084 / 1.084 | 0.987 / 0.985 | none |

The pick's L000 / L090 rows are its selection rows (`RSDSEL_155b5981_*`), read in `g2.json` N.

**N, pooled completion against in-pin c28000** (`g2.json` N_pooled):

| Load | PICK | c28000 | Diff | SE_diff | m (binds) | Shortfall | In (m, m + 0.04]? | Result |
|---|---|---|---|---|---|---|---|---|
| L000 | 0.990 | 0.995 | −0.005 | 0.009 | 0.050 (floor) | 0.005 | no (window 0.050–0.090) | pass |
| L090 | 0.995 | 0.950 | +0.045 | 0.016 | 0.050 (floor) | none | no | pass |
| L135 | 0.975 | 0.875 | +0.100 | 0.025 | 0.051 (SE) | none | no | pass; beats c28000 by m |

No cell fell in the class-ambiguity window, so no class-ambiguity relaunch was due (`relaunch_due` is empty).

**es3 and relaunch allowance:** 0 of 4 cell-seeds used (es3 0, relaunch charge 0; nothing due) (`g2.json`).

**R WATCH against c28000's Task 4 full block** (cross-pin; `g2.json` R_watch). The JSON's key `tstar` holds c28000's Task 4 block values; the name comes from the reused Task 4 function. The values match Task 4's record (field_hard 0.890, ring_obst 1.000, field_turn 1.000; `REPORT_baseline_vs_final.md` line 9).

| Row | PICK | c28000 (Task 4) | SE_diff | m (binds) | Flag below | Flag |
|---|---|---|---|---|---|---|
| field_hard | 0.94 | 0.89 | 0.027 | 0.054 (SE) | 0.836 | none |
| ring_obst | 1.00 | 1.00 | 0 | 0.05 (floor) | 0.95 | none |
| field_turn | 0.995 | 1.00 | 0.005 | 0.05 (floor) | 0.95 | none |

- The tracking watch has 20 rows (ratio and |r − 1| for fwd, side, back, diag and move_turn, es1 and es2) with a +0.05 margin. It raised no flags.
- The closest row is forward |r − 1|: the pick 0.033 / 0.037 against 0.008 / 0.013, worse by 0.025 / 0.024 of the 0.05 margin. The pick's forward ratio is 1.033 / 1.037 against 0.992 / 0.987. It walks forward 3–4 % above command at 0 kg, a shift of the size TDR showed at G1 (|r − 1| 0.039 against 0.007, §4.5).

**WATCH cells against the sweep's c28000 stamps.** These are cross-pin (2026-11-06 pin) and so possibly cross-class: c28000's own stamps swing 0.05–0.07 per seed between classes at stressed 90-kg cells (§7.7). No statistic is computed.

| Cell | PICK es1 / es2 | Sweep c28000 es1 / es2 | Lines |
|---|---|---|---|
| T02 rod effort ×0.85 L090: completion | 0.99 / 0.96 | 0.87 / 0.85 | the pick passes all 18 on both seeds; the sweep's c28000 passed, es2 at 85 / 100 (sweep REPORT §4) |
| T02: h_meas.suite / pitch_rms.field_light | 1.116 / 1.116; 0.053 / 0.053 | 1.085 / 1.085; 0.072 / 0.073 | — |
| T13 150 kg: completion | 0.93 / 0.96 | 0.86 / **0.73** | the pick passes all 18 on both seeds; the sweep's c28000 es2 fails completion |
| T13: h_meas.suite / pitch_rms.field_light | 1.114 / 1.113; 0.054 / 0.054 | 1.083 / 1.083; 0.073 / 0.075 | — |

L135 per-seed completion (WATCH): the pick 0.96 / 0.99 (`g2.json` watch); in-pin c28000 0.89 / 0.86†.

**Diagnostic: the pick against TDR at ×0.7 L090, paired** (`g2.json` diagnostic, pairs †):

| | PICK | TDR | Diff | SE_diff | m (binds) | Pairs: both / pick only / TDR only / neither |
|---|---|---|---|---|---|---|
| field_light completion, es1 / es2 (pooled) | 0.92 / 0.95 (0.935) | 0.94 / 0.92 (0.930) | +0.005 | 0.024 | 0.05 (floor) | 175 / 12 / 11 / 2† |

The pick is non-inferior to its teacher at ×0.7 and does not beat it by m. Its suite tracking at ×0.7 is a little worse than TDR's (diag 0.090 / 0.093 against 0.078 / 0.080; side 0.151 / 0.137 against 0.138 / 0.131), inside the lines. Distillation lost nothing measurable at the cell the DR targets.

**Closing: the pick's L090 nominal es1 at launch Z against its selection row** (`RSDS_155b5981_T00_nominal_L090_Z` vs `RSDSEL_155b5981_L090`; `g2.json` closing; [8 22:55], [8 23:06]):
- **The mismatch:** both rows mismatch. 13 of the 18 line values differ: field_light completion is 0.99 against 1.00, suite survival.move_turn 0.995 against 1.000, and the rest differ in the third or fourth decimal.
- **What is identical:** same pin, same code state, run_meta identical apart from time and outcome fields (env vars included), and the dose rated and identical in every episode (`classflip_check.py` output).
- **field_light (100 episodes):** all 100 differ.
  - The step-0 state is identical. The earliest difference is at step 0, in the action alone, in envs 5, 7 and 30 (≤ 6.1e-6).
  - 71 episodes first differ action-only at a depth-frame step; 29 first differ later, otherwise.
  - Verdict: **class flip** under §4 as amended at 20:12.
- **suite (1000 episodes, 10 waves × 100):** all 1000 differ. Tested as one stamp, the 20:12 text gives CODE-PATH: the step-0 state is not identical, and the earliest keys include `applied_torque` and `foot_force_norm` (max first action delta 1.04). Per wave (`wave_diff.py`):
  - **Wave 0** meets every condition. The step-0 state is bit-equal (no pose array differs). The earliest difference is at step 0, in the action alone, in 7 episodes: envs 0, 4, 15, 35, 36, 58 and 75 (≤ 1.8e-5).
  - **Waves 1–9:** all 900 episodes differ at their own step 0 in `applied_torque` and `foot_force_norm`. 271 of them differ in the action too (31 / 21 / 23 / 40 / 26 / 21 / 41 / 37 / 31 per wave, up to 1.04). Every kinematic state array is bit-equal at step 0 [8 23:26].
  - The cause: `env.reset()` leaves the applied torque, the contact force history and the actuator's rod speed at the previous wave's end values. The current-sense observation reads them at the wave's first step. Wave 0's divergence therefore reaches each later wave's step 0 (§7.6) [8 23:26].
- **The rule:** §4 as amended at 20:12, read literally, called the suite mismatch a code-path difference, which stops the campaign. At 23:06 the user decided to judge multi-wave rows on wave 0 (option 1; the §4 multi-wave bullet). The suite closing is recorded as a **class flip**, and G2 PASS stands [8 23:06].
- **Diagnostic relaunch** (23:02, read by no gate): `RSDS_155b5981_T00_nominal_L090_Z_D1` and `RSDSEL_155b5981_L090_D1`, es1, suite + field_light (4 rows), through rsd_gate's own Runner (driver `diag_closing_rows.py`). It shows whether each row shape reproduces a recorded stamp bit for bit [8 22:55]. Its outcome [8 23:26]:
  - The four rows ran 23:02–23:24 (0.36 h), each ok on its first try.
  - Both relaunches equal Z bit for bit, in every raw-npz array of all 1000 suite episodes and all 100 field_light episodes (`classflip_check.py`). That holds for the cell row (`_Z_D1`) and the selection-shaped row (`RSDSEL_155b5981_L090_D1`).
  - So the row's outcome repeats exactly, and both row shapes land in the same class. The closing's reference, the selection launch at 13:14, is the other of the two classes. The class does not follow the row shape.
  - Effect on N: the pick's L090 nominal es1 rows in the N read are the minority class. In Z's class its field_light completion is 0.99, not 1.00. That would make the pooled L090 read +0.040 rather than +0.045 (m 0.050). Across every class observed for either policy, the read lies between +0.03 and +0.045 [8 23:26].

**Adversarial check of the G2 read** (`g2-verify`, four read-only checkers, started 23:04 [8 22:55]). The checkers were recompute (opus), rules (fable), provenance (sonnet) and a refuter (opus); they ran 23:04–23:23. **All four confirm** [8 23:26]:
  - Every G2 number recomputes from the stamps within 1e-9. No carve-out, es3 or relaunch was due. No pick line sits within one episode (or 0.003 on roll / pitch) of its threshold, and every read stamp passes the launch checks.
  - **Class structure, measured directly (refuter):**
    - At one eval seed the pick's step-0 state is identical across its cells, and its step-0 actions in the class envs take exactly two bitwise values over 11 launches: field_light es1 (envs 5, 7, 30) 8 / 3, suite es1 9 / 2.
    - c28000's 16 stamps show the same two values in the known split's envs 26, 43, 54 and 81.
    - Every pick S row at L090 es1 landed in Z's class, so the pick's other-class values at the stress cells are unmeasured.
    - The S cell with the least headroom is ×0.7 L090 es1, at 0.92, 7 episodes above the line. c28000's observed class swings at ×0.8 L090 are 5–7 episodes.
  - **Corrections to the record**, now in CHARTER §4 / §6 and above: the state that carries across waves (§7.6), and the 271-of-900 action count. Neither changes a verdict.
  - **Record-only:**
    - `g2.json` holds no closing verdict: the closing does not enter `g2_decide`, and its class-flip verdict is in CHARTER §6 (23:06).
    - The pooled SE moves in the 4th decimal with the bootstrap seed. This has no effect, since m binds at the 0.05 floor at L000 / L090.
    - The row logs capture no Kit stderr.
  - **WATCH-only traits, both shared with TDR and far from any line:**
    - The pick walks sideways slower on the left hand than on the right: at ×0.7 L090 the l / r side ratio is 0.81 / 0.90 (TDR 0.83 / 0.91).
    - It overshoots forward speed slightly: tracking.fwd 0.033–0.039 against c28000's 0.003–0.016.

## 9. What this changes

### 9.1 The review's rungs 2–4 (baseline: sweep REPORT §7 and §9)

| Item | Sweep call / decision | Now | Evidence |
|---|---|---|---|
| Rung 3: rod speed U(0.6, 1.35) in teacher training (decision 2) | UNDER-CALLED → widen, teacher-side | **Done for both roles.** Teacher: G1 PASS, attributable at ×0.7. Student: G2 PASS. The pick completes 0.92 / 0.95 at ×0.7 L090 against c28000's 0.58 / 0.66† and matches TDR there (paired +0.005). | §4; §8.4 |
| Rung 3 entry: p3b-entry fine-tune first, p2a → student only on a G1 fail (decision 3) | cheap entry first | The p3b entry was enough: PASS at +3000 iterations with no EXTEND, and the student distilled from it passed G2. The p2a escalation is not needed. | `g1.json` (`extended: false`); `g2.json` |
| Rung 3: rod-speed priv-latent slot (decision 3, open) | open | Not added, and G2 did not need it. The DR buys robustness, not adaptation: no input carries the scale, and the teacher's 41 priv inputs are constant (plan, Context). | §3.1 |
| Rung 3: T02 rod effort (decision 4) | no DR now; watch ×0.85 L090 | Not trained. G2 WATCH: the pick passes every line on both seeds (0.99 / 0.96), where the sweep's c28000 sat at the line (0.87 / 0.85; cross-pin). For the pick, the flagged pass is no longer near the line. | §8.4 |
| Rung 3: force-dependent speed line, other actuator terms | DEFER until the bench | Unchanged | — |
| Rung 2 (decision 8) | fold into the rebuild's student stage; distil rider-free first | This distill was rider-free, and G2 read T01 alone and passed. The riders (depth dropout / blackout, obs noise, current noise, the vxy decision) can now go into a second distill: ≈ 9.6 h at this distill's pace, plus ≈ 5.9 h of selection and G2 at this campaign's pace. | `runs/RSD_ST_s19.distill.json`; §3.7 |
| Rung 4: payload (decision 10) | keep U(0, 180); set an operating bound ≤ 135 kg | U(0, 180) was kept in TDR and in the distill. L135 pooled: TDR 0.960 against old T\*'s 0.875; the pick 0.975 against c28000's 0.875. At 150 kg the pick completes 0.93 / 0.96 and passes every line; the sweep's c28000 had 0.86 / 0.73 (cross-pin, WATCH). The teacher at 150 kg is untested. The ≤ 135 kg operating bound stays until a same-pin read and the bench exist. | `g1.json`; `g2.json` |
| Rung 1b (decision 7) | ≈ 1.8 h eval batch incl. T01 ×0.7 / ×0.8 at L135 | Not run. Only L135 nominal ran here, for TDR and the pick. | — |
| Gate hygiene (decision 9) | per-hand side lines; es3 at 85–86 / 100; re-read T01 on both roles | es3 rule adopted; not needed at G1 or G2. Per-hand ratios are WATCH only: the worst hand at ×0.7 L090 is TDR's lat_l 0.825 and the pick's lat_l 0.811 (|r − 1| 0.189), inside a 0.25 per-hand line. T01 was re-read on both roles: G1 and G2 PASS. The class-ambiguity window is narrower than the observed class swings (§7.7). | `g1.json`, `g2.json` watch |
| Bench (decision 5); velocity slot (decision 6) | before first stand; lead rung 2 | Unchanged by this campaign | — |

### 9.2 Hardware
- **Deploy interface:** unchanged. The pick's `omni_student_io.json` has the same sha256 prefix as F19's export, and the ONNX matches the `.pt` (§6).
- **What the pick changes:** in sim, the deployable student no longer fails slow rods at 90 kg. At ×0.7 L090 its survival.diag is 1.00 against c28000's 0.83 / 0.87†, and its completion 0.92 / 0.95 against 0.58 / 0.66†. It is not a deploy candidate yet: the veto clip, SN/SH re-pairing and the bake remain (decision 5), and the bench comes before first stand (sweep decision 5).
- **Posture:**
  - TDR stands ≈ 0.03 m taller in every cell: suite h_meas 1.117 against 1.089 at L000, 1.114 against 1.085 at L090.
  - TDR's suite pitch_rms at ×0.7 is about half of old T\*'s (0.028 against 0.053).
  - The pick inherits the stance, now measured against in-pin c28000. Its suite h_meas at L090 nominal is 1.118 / 1.117 against 1.086 / 1.086†. At ×0.7 L090 its field_light pitch_rms is 0.053 / 0.054 against 0.095 / 0.094†.
  - Whether the taller stance raises or lowers stance rod force was not measured for TDR or the pick. The sweep's knee figure (p50 2.7–3.3 kN against a 500 N rating, sweep REPORT §3.5) is F19's.
- **Forward speed at 0 kg:** the pick walks forward 3–4 % above command (ratio 1.033 / 1.037 against c28000's Task 4 0.992 / 0.987; `g2.json` R_watch). It uses about half of the +0.05 tracking WATCH margin.
- **Robust, not adaptive:** the policy cannot sense slow rods. It was trained to tolerate a per-episode scale U(0.6, 1.35) drawn independently of payload.
  - In sim the slew is load-blind. On hardware the screws slow with load, so what ×0.7 means at 90 kg depends on the bench's speed-against-force curve.
  - The bench (sweep decision 5) still comes first.
- **Coverage of the band:** only ×0.7 and ×0.8 at L090, plus ×0.7 and ×1.35 at L000 and ×1.35 at L090, were tested, for both TDR and the pick. Untested: the trained low edge ×0.6, slow rods at L135, and slow rods combined with delay or estimator bias.
- **Payload:** L135 completion improved for the teacher (0.960 against 0.875 pooled) and for the student (0.975 against 0.875). At 150 kg the pick passes every line (0.93 / 0.96; cross-pin WATCH). The operating bound (≤ 135 kg, sweep decision 10) stays until a same-pin 150-kg read and the bench's holding figures exist.
- **Unchanged risks:** the velocity-slot dependence (sweep T10b), the knee holding cap and the camera-use question (Task 4: UNVERIFIED) are not touched. The new student was distilled the same way as F19's, so its camera use is untested too.

## 10. Caveats and limits
- **One training seed per arm** (seed 19), **and one distill** (seed 19). A's gap is ≈ 7.5 eval SEs, but training-seed variance is untested. G2 rests on one student.
- **The DR's specific gain is shown for the teacher at ×0.7.** At ×0.8 most of the teacher's gain is extra training (TCTL 0.930 against TDR 0.955 pooled). For the student there is no control: whether the pick's gain over c28000 comes from TDR, from the DR in the distill, or from both is not separated (a control student is on the plan's left-out list).
- **Ungated shifts:**
  - TDR's 0-kg forward speed ratio moved (|r − 1| 0.039 against 0.007; 63 % of R). The pick shows the same size of shift (|r − 1| 0.033 / 0.037 against c28000's 0.008 / 0.013; half the WATCH margin).
  - TDR and the pick stand ≈ 0.03 m taller, and pitch_rms at ×0.7 is roughly halved.
  - The pick's left-hand side ratio at ×0.7 L090 (0.81) is the slowest of the three policies.
  - TCTL's back tracking and turn_fc drifted at nominal. Both TCTL drifts stay inside the lines.
- **R's ring_obst row has zero margin.** One lost ring_obst episode would fail R for a later head.
- **Per-seed completion line:** S applies it per seed, which is stricter than v1.2's pooled definition. It made old T\*'s ×0.8 es2 a fail.
- **Class noise:**
  - c28000's in-pin rows are a class mix.
  - Cross-launch swings reach 0.05–0.07 per seed at stressed 90-kg cells, more than G2's ±0.04 ambiguity window (§7.7).
  - At G2 no N cell came near the window. The closest is L000, with a shortfall of 0.005 against m 0.050.
  - The tightest S cell, ×0.7 L090 es1 (0.92, 7 episodes above the line), has about as much headroom as c28000's observed 5–7-episode class swing at ×0.8 L090. The pick's other-class S values were not measured.
- **G2's L000 / L090 N inputs were partly visible before the read.** The pick's selection rows were read at selection (`sel.json`), and c28000's in-pin arm was read at the bridge diagnosis [8 18:00]. For those two loads the read was not blind. c28000's whole in-pin arm, its S references and L135 rows included, was read at the bridge diagnosis [8 18:00]. Its S references enter only the carve-out, which G2 did not use. The pick's S cells, its L135 rows and the WATCH cells were first read at 22:55.
- **Two verdict rules changed after their data were read.** Both were the user's decisions:
  - the §4 class-flip amendment (20:12, with the executor's correction kept at 20:18), made after the bridge data;
  - the multi-wave clarification (23:06), made after the G2 closing data.

  G2's closing used both. Before any GPU row the CHARTER was amended once more, at [7 16:25] (§3.4).
- **The closing:** both closing rows mismatched. They count as class flips only under the two post-data rule decisions. The diagnostic relaunch reproduced Z bit for bit in both row shapes, so the closing's reference is the other class. The pick's L090 nominal es1 N rows are then the minority class. In Z's class the pooled L090 read is +0.040, not +0.045 (§8.4).
- **The WATCH cells are cross-pin**, and possibly cross-class. The R WATCH is cross-pin too (Task 4's stamps).
- **Selection resolution:**
  - First place rests on 2 of 200 field_hard episodes.
  - All four completion margins bind at the 0.05 floor, with field_hard tightest (0.94 against 0.91).
  - Task 4's veto clip, which follows its selection, ran here only after G2 (2026-11-09, decision 4). It is a look, not a gate.
- **Scope:**
  - Flat suite and field_light for the gate cells. Selection adds field_hard, ring_obst, field_turn and side22 at 0 kg, plus the 90-kg level.
  - One term at a time, with no interactions.
  - Rod speed tested at 0 and 90 kg; 135 kg nominal only; 150 kg as a WATCH cell for the student only.
- **Sim plant:** load-blind slew; drive force not enforced; `applied_torque` is a PD estimate; rated forces are never enforced (sweep REPORT §8).
- **ONNX parity covers one 120-step CPU record.**
- **Storage:** TDR, TCTL and RSD_ST_s19 exist only under the worktree's gitignored `parkour/logs/` [C §5]. Removing the worktree deletes them. The stamps are untracked in the main checkout.
- **Tests:** two `test_task4_pipeline.py` tests fail on main-checkout data since the disk deletion. The fix is unmerged (§3.4). The plan also lists three dose-smoke unit tests that fail in this worktree since `parkour/logs/rsl_rl/metrics_history` appeared there. They are on its left-out list and unfixed.
- **The 2.4-h idle GPU** at the bridge STOP is the campaign's one avoidable wall-clock loss. From 19:49, rows ran pending the decision.

## 11. Decisions for you

1. **Done 2026-11-08 20:18: the amended §4 class-flip rule, with the executor's correction.** You kept the corrected wording ("Keep it"): the action-only, depth-frame condition applies to the stamp's earliest difference, not to each episode's first difference [8 20:18]. G2 was read under it.
2. **Done 2026-11-08 23:06: multi-wave rows are judged on wave 0.** You answered "1" to the 23:05 question (§7.6). The suite closing is a class flip, and G2 PASS stands [8 23:06].
3. **Accept G2 PASS?**
   - Evidence: §8.4. Every S and N line passes on both seeds. No relaunch or es3 ran. R WATCH and the WATCH cells raise no flag.
   - Its closing rests on decisions 1 and 2, both made after the data were read.
   - Both checks agree with §8.4: the diagnostic relaunch reproduces Z bit for bit in both row shapes, and `g2-verify` confirms 4 / 4 [8 23:26].
   - **Recommend yes.**
4. **Done 2026-11-09 09:46–09:49: the pick's headless veto clip is recorded** (your "Yes, both"). Task 4 sent one clip of its selected student to you for a veto, never as a gate (`pipeline.py` 2315–2317).
   - Recipe: Task 4's (`../2026-10-16_task4_pipeline/step11/clip_F19.sh`, which made F19's c28000 clip), run by `scripts/clip_pick.sh` on this worktree's code.
     - `play_stream_video.py`, Student-Play, 1 env, 1500 steps (30 s at 50 Hz).
     - The wall field env `F19/envs/p3b.field_hard.env`: 0 kg, with the DR and stress knobs off.
     - Run under `systemd-run --user --scope -p MemoryMax=40G -p MemorySwapMax=0`. Never `play.py --video` / `RecordVideo`: its frame buffering has crashed this machine twice.
   - Clip: `clip/student_c31000_field_hard.mp4` (1280 × 720, 50 fps) and `clip/student_c31000_field_hard_small.mp4` (960 × 540), sent to you. Both are local only (`*.mp4` is gitignored).
   - F19's c28000 clip is in the main checkout at `../2026-10-16_task4_pipeline/F19/clip/`. The terrain draw differs between the two clips, so they are not a paired comparison.
   - What it shows: in frames sampled every 2 s, the robot stays upright for the whole 30 s. It is a look before adoption (decision 5), not a gate.
   - Wall-clock: 103 s in Kit, about 2 min in all.
5. **Adopt TDR and the pick as the omni teacher and student of record?**
   - Evidence: §4, §5, §8. For the student, the pick removes F19's FRAGILE cell (×0.7 L090 0.92 / 0.95 against c28000's 0.58 / 0.66†) and matches TDR there.
   - What still stands between the pick and a deploy candidate:
     - your look at the veto clip (decision 4, recorded 2026-11-09);
     - SN/SH re-pairing to TDR and the bake into `crab_hex_omni_task/policy/` (the plan's left-out list). For F19 these took sn 3.05 h + sh 3.56 h of wall-clock (`REPORT_baseline_vs_final.md`, F19 stage table), then a new bundle;
     - copies of the run dirs that outlive the worktree (decision 8);
     - for hardware, the sweep's open items that this campaign did not touch: the bench before first stand (sweep decision 5) and the velocity slot (sweep decision 6).
   - **Recommend adopting both together after the clip, then SN/SH re-pairing (≈ 6.6 h), with the bake only on your approval.** F19 stays of record until the bake. TDR alone changes no deployed policy.
6. **Done 2026-11-09: the campaign record is committed locally** (your "Yes, both" to decisions 4 and 6). Not pushed.
   - Contents:
     - this REPORT;
     - the CHARTER: the §4 code-pin line, the ONNX block (18:10), the 20:12 amendment and the 23:06 clarification, the §5 files list, and every §6 entry after `1682e2e1`;
     - `scripts/rsd_queue.sh` and `scripts/rsd_onnx.py`;
     - the three class tools, copied into `scripts/` (identical to the scratchpad copies): `classflip_check.py`, `wave_diff.py`, `diag_closing_rows.py`;
     - `scripts/clip_pick.sh`, the veto clip's recipe (decision 4), added at the commit;
     - the closing and diagnostic comparison outputs, copied into `results/closing/`: `closing_g2_field_light.out`, `closing_g2_suite.out`, `closing_g2_suite_waves.out` and `diag_compare.out`;
     - `envs/RSD_TDR_s19.train.env` and `envs/RSD_TCTL_s19.train.env`;
     - `results/`: `g1.*`, `g2.*`, `sel*.json`, `bridge_*.json`, `onnx_PICK.json`, the modified `policies.json` and `replay.*`, `launch.json`, `notify.jsonl` and `runs/`. `k1.out`, `k1.reread.out` and `k2.out` are already in `1682e2e1`, unchanged.
   - Left out: the empty `results/rsd_gate.lock`, the sweep's `run_sweep.lock`, and the `k1/` and `k2/` dumps and logs, which §4 keeps untracked. `*.log` and `*.mp4` files are gitignored.
   - The two checks behind it are logged in §6 [8 23:26].
7. **Merge the test fix `96028b98`?** It sits on branch `claude/nostalgic-bouman-546381`, not merged and not pushed. It adds test-only skip guards to the two `test_task4_pipeline.py` R18 tests broken by the disk deletion. **Recommend yes, via your PR.**
8. **Protect the trained runs?**
   - TDR, TCTL and the student run live only in the worktree's gitignored `parkour/logs/`.
   - **Recommend hardlink-copying them into the main checkout's `parkour/logs/`** (set aside in the plan) **before any worktree cleanup**, or bundling them under `policy/` at adoption. No GPU.
9. **Replicate on a second training seed?**
   - Costs, from §3.7:
     - the teacher pair (1.22 + 1.28 h) plus the ×0.7 L090 A rows (2 cell-seeds per arm) ≈ 3.2 h;
     - with TDR's full G1 rows (3.25 h) and TCTL's 12 rows (1.12 h) ≈ 6.9 h;
     - carrying the seed through a student (distill 9.6 h, selection 2.83 h, G2 3.07 h) adds ≈ 15.5 h.
   - A control student (no DR in the distill) would separate the teacher's share of the student's gain from the distill DR's. The plan left it out at ≈ 14 h.
   - **Recommend not before adoption, unless adoption should rest on more than one seed.** A's gap is ≈ 7.5 eval SE, the falls moved in the holds the mechanism predicts, and the pick reproduces TDR at ×0.7 (paired +0.005).
10. **The class-ambiguity window for the next student gate?**
    - Observed class swings at stressed 90-kg cells are 0.05–0.07 per seed (§7.7), wider than the ±0.04 window. G2 did not need the window: the closest shortfall was 0.005 at L000, against a window of 0.050–0.090.
    - **Recommend widening it, or relaunching both classes, in the next student gate's CHARTER, before any of its data are read.**
11. **Next rungs?**
    - **Recommend, if the pair is adopted:**
      - the rung-1b batch re-based on TDR / the pick: T01 ×0.7 / ×0.8 at L135, T07 d = 2 and T10a σ 0.1 at L090 (≈ 1.8 h, sweep decision 7's estimate on the same serial queue);
      - then the velocity-slot cells (sweep decision 6: ≈ 0.73 h plus knob code);
      - then a rider distill (rung 2): ≈ 9.6 h, plus ≈ 5.9 h of selection and G2;
      - bench work in parallel.
    - The DR review has no rung-1 outcome note yet (sweep decision 11). Neither the worktree's nor the main checkout's `lit-review-domain-randomization.md` has one. **Recommend appending one dated note covering the sweep and this campaign** (no GPU).
12. **Push?** Nothing is pushed. Pushing happens only through your PR.

## 12. Files and how to re-read them
- **Record** (`crab_hex_omni_task/experiments/2026-11-07_rod_speed_dr/`):
  - `CHARTER.md`: design §1–3; pins §4, with the amended class-flip rule (20:12) and its multi-wave bullet (23:06); files §5; log §6.
  - `results/`:
    - G1: `g1.{json,md}`.
    - G2: `g2.{json,md}` (read 2026-11-08 22:55:06).
    - Selection: `sel.json` and `sel.rows.json` (rows by key).
    - Bridges and policies: `bridge_TSTAR.json`, `bridge_C28000.json`, `policies.json`, `launch.json`.
    - ONNX and replay: `onnx_PICK.json`, `replay.{json,md}`.
    - Kit checks: `k1.out`, `k1.reread.out`, `k2.out`, and `k1/` and `k2/` (the dumps and logs, untracked).
    - Runs: `runs/RSD_{TDR,TCTL}_s19.json` (PF1–PF14), `runs/RSD_ST_s19.{distill.json,check.out}`.
    - Logs: `rsd_gate.log` (the `RSD ROW / SKIP / READ / STOP` sentinels; gitignored) and `notify.jsonl`.
  - `envs/`: the cell env files, `p3b_dr.distill.env`, `RSD_T{DR,CTL}_s19.train.env`.
  - `scripts/`: `rsd_lib.py`, `rsd_gate.py`, `rsd_train.py`, `rsd_launch.py`, `rsd_distill.sh`, `rsd_distill_check.py`, `rsd_queue.sh`, `rsd_onnx.py`, `rsd_dump_check.sh`, `rsd_dump_compare.py`, `rsd_smoke.{py,sh}`, `clip_pick.sh` (the veto clip, decision 4).
  - `clip/`: the pick's veto clip, `student_c31000_field_hard.mp4` and `student_c31000_field_hard_small.mp4`. They are local only, since `*.mp4` is gitignored. Its Kit log is `parkour/outputs/omni_plan/rod_speed_dr/clip_pick.log` in the main checkout.
  - The class tools: `classflip_check.py` (the §4 test for two stamps of one row; `--known` for the cross-pin bridge), `wave_diff.py` (per-wave first differences of a suite row) and `diag_closing_rows.py` (the diagnostic relaunch driver). They were copied into `scripts/` at 23:29 and committed in the record commit (decision 6).
  - The closing and diagnostic comparisons, as `classflip_check.py` / `wave_diff.py` printed them: `results/closing/closing_g2_{field_light,suite,suite_waves}.out` and `results/closing/diag_compare.out` (in the record commit).
- **Stamps** (main checkout, untracked): `parkour/outputs/omni_plan/rod_speed_dr/lr/<GATE>/<scenario>/seed00K/<ts>/`.
  - GATE names:
    - `RSDT_<sha8>_<cell>`: teacher rows.
    - `RSDS_<sha8>_<cell>`: student rows (`RSDS_155b5981_*` = the pick's G2 cells; `RSDS_4fa81f0f_*` = c28000's in-pin reference arm).
    - `RSDSELT_<sha8>_L000`: teacher selection rows.
    - `RSDSEL_<sha8>_{L000,L090}`: student selection rows.
    - `_RL`: relaunch. Cell `T00_nominal_L090_Z`: closing. `_D1`: the diagnostic relaunch.
  - The `<ts>` directory name is the row's launch time in UTC (local + 5 h). `rsd_gate.log` instead records when the row was logged, after it finished. Example: c28000's first reference row was launched at 15:23 local, is stamped `2026-11-08_20-23-05`, and was logged at 15:30.
- **Queue and training logs** (main checkout): `parkour/outputs/omni_plan/rod_speed_dr/{queue.log,<stage>.driver.out,train/RSD_T*_s19.log,distill/RSD_ST_s19.log}`. ONNX: `…/onnx/PICK_c31000/`.
- **Trained runs** (worktree only, gitignored):
  - `parkour/logs/rsl_rl/crab_hex_omni_teacher/2026-11-07_19-49-46_RSD_TDR_s19/`
  - `…/2026-11-08_00-20-19_RSD_TCTL_s19/`
  - `parkour/logs/rsl_rl/crab_hex_omni_student/2026-11-08_02-57-24_RSD_ST_s19/`
- **References:**
  - `../2026-11-06_dr_stress_sweep/REPORT.md` (the baseline).
  - `../2026-10-16_task4_pipeline/REPORT_baseline_vs_final.md` (F19, T\*, c28000, stage wall-clock).
  - `../2026-10-16_task4_pipeline/thresholds_v1_2.json`.
- **Re-read** (CPU only, from `scripts/`):

```bash
PY="/home/nickmagus/krabby/isaac_venv/bin/python -B -I"; G="$PY rsd_gate.py"
$G policy list                                                   # the registry (prints only)
$G read g1 --tdr TDR --tctl TCTL --out /tmp/rsd_g1.json          # rebuild G1 from the stamps into /tmp (also writes /tmp/rsd_g1.md and appends a READ line to results/rsd_gate.log)
$G read g2 --pick PICK --tdr TDR --out /tmp/rsd_g2.json          # rebuild G2 the same way (same side effects)
$G replay --out /tmp/rsd_replay.json                             # the CPU replay (same side effects)
```

- **The † values:** `stress_read.read_stamps(<suite stamp>, <field_light stamp>, es)` from `../2026-11-06_dr_stress_sweep/scripts/`, loaded with its `cells.py` under `python -B -I`. It writes nothing. Pair counts: each field_light stamp's `scenario_metrics.json`, an episode counting as complete when its `termination_reason` is `schedule_complete` (Task 4's `thresholds.field_episodes`).
