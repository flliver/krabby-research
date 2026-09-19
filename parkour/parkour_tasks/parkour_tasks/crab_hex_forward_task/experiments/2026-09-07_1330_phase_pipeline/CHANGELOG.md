# CHANGELOG — 2026-09-07_1330_phase_pipeline

### 2026-09-07 13:59 — campaign start: plant A15+B, seed 3, phases 3a, 3b

### 2026-09-07 18:17 — USER: pause for a restart
Phase 3a stopped cleanly at iteration 22000 (checkpoint `crab_hex_student/2026-09-07_13-59-12/model_22000.pt`, verified loadable; 2996 iterations left to the phase's final 24995). Resume with `--continue` (finishes 3a from that checkpoint, records the phase whole, pauses after 3a as planned, then 3b):

    sim_fine_tuning/tools/launch_phases.sh --campaign-dir sim_fine_tuning/2026-09-07_1330_phase_pipeline --plant A15+B --phases 3a,3b --seed 3 --continue

### 2026-09-07 23:13 — USER: resume
Relaunched with `--continue` (unit phases-2026-09-07_1330_phase_pipeline-1788837176): phase 3a continues from `model_22000.pt` for 2996 iterations (target 24995); depth encoder restored from the checkpoint; 7.4 s/it, GPU 13.4 GB.

### 2026-09-08 05:34 — phase 3a ok (continued from model_22000.pt): model_24995.pt (seed3_3a_002_A15pB_it2996)
> NOTIFY: phase 3a done (model_24995.pt); PAUSED before 3b — relaunch with the same --phases to continue

### 2026-09-08 05:52 — phase 3a INVALID: root cause found and fixed
The 3a student (model_24995) fell in 100/100 episodes on all three evals; the training log shows mean episode length ~165 steps and crab_failure 1.0 from the first distillation iteration (the August 2b2 student reached ~540 steps by iteration 50). Diagnostic: the 2c TEACHER itself (privileged path, new harness flag `--policy-role teacher`) also fell 100/100 on `Isaac-Crab-Hex-Student-v0` + `KRABBY_STUDENT_MDP=1`, while scoring 0.81 on the flat-walk task. Cause: `CrabHexStudentPPORunnerCfg.clip_actions` was unset (None) — the vec-env wrapper clips RAW policy actions to +-clip_actions before the action term scales them; the flat-walk/phase-2 runners use 1.0. Harmless on the legacy 2b2 student MDP (its action term clipped raw actions to +-1 itself) but on the phase-3 MDP (teacher's full action space, raw clip +-4.8) the joints were driven at ~5x the trained authority. Fix: `clip_actions = 1.0` on the student runner. Confirmation: teacher on the student env with the fix — completion 0.70, falls 30/100, tripod 0.48, tracking 0.66 (flat canary). Coverage added: driver survival checks + early warning for distillation phases; identity test asserts student/teacher runner clip parity. 3a record moved to `3a_invalid_unclipped`; 3a relaunched from the 20k head.

### 2026-09-08 06:01 — 3a rerun healthy (unit phases-2026-09-07_1330_phase_pipeline-1788861231)
From the 20k head with the clip fix: mean episode length 72 -> 545 (it 9) -> 1730 (it 19) -> ~1350-1560 (it 29-60); crab_failure 0.72 (it 9) -> 0.36 (it 60), i.e. the teacher's own ~0.34 on this MDP; depth_actor_loss 19.7 -> 6.6. Open item: with the fix the teacher scores 0.70 completion / 30 falls on the student env vs 0.81 / 19 on the flat-walk task (same head, same knobs, env_seed 1) — a residual difference to quantify once 3a's evals land.

### 2026-09-08 15:51 — phase 3a ok: model_24995.pt (seed3_3a_003_A15pB)
> NOTIFY: phase 3a done (model_24995.pt); PAUSED before 3b — relaunch with the same --phases to continue

### 2026-09-08 16:02 — phase 3a (rerun) COMPLETE — PAUSED before 3b (user go/no-go)
Student model_24995 matches the 2c teacher: flat 0.79/21 (teacher 0.81/19), step 0.71/29 (0.68/32), obstacles 0.64/36 (0.66/34). Resume: `sim_fine_tuning/tools/launch_phases.sh --campaign-dir sim_fine_tuning/2026-09-07_1330_phase_pipeline --plant A15+B --phases 3a,3b --seed 3` (3a skipped as recorded ok).
> NOTIFY: pushed to the user

### 2026-09-09 02:12 — USER: "go ahead with 3b"
Launched unit phases-2026-09-07_1330_phase_pipeline-1788933999: phase 3b (5k distillation its, band 0.70–0.90) resuming the 3a head model_24995; 3a skipped as recorded ok.

### 2026-09-09 12:04 — phase 3b ok: model_29994.pt (seed3_3b_004_A15pB)
> NOTIFY: phases 3a, 3b complete for seed3 — see REPORT.md

### 2026-09-09 12:13 — phase 3b COMPLETE; seed-3 pipeline 3a→3b done
3b head `crab_hex_student/2026-09-09_02-06-51/model_29994.pt` (5k its on 0.70–0.90): flat 0.81/19 (teacher 0.81/19), step 0.68/32 (0.68/32), obstacles 0.20–0.70 0.62/38 (teacher 0.66/34; 3a 0.64/36). Training ep len 1367, crab_failure 0.44 on the hard band; depth_actor_loss 5.75→3.20. All checks PASS. Hard-band (0.70–0.90) evals of teacher / 3a / 3b running for the bake decision. Next per plan: seed-2 replay 1a→3b (user go), then bake decision (user).
> NOTIFY: pushed to the user

### 2026-09-09 12:19 — hard-band (0.70–0.90) evals
teacher 0.45/55, 3a 0.51/49, 3b 0.52/48 (completion/falls). Summary table in REPORT. Awaiting user: seed-2 replay go; bake choice 3a vs 3b.

### 2026-09-09 12:26 — USER DECISION: BAKE 3a as the phase-3 head; campaign CLOSED
Head of record `parkour/logs/rsl_rl/crab_hex_student/2026-09-08_05-54-01/model_24995.pt`. 3b run but not baked (equivalent). Seed-2 replay waived.
