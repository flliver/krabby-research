# DESIGN — detail of record for the Task 4 pipeline (annotated copy)

> **Read this first.** Below this header is the verified design synthesis written during planning (2026-10-16),
> copied unchanged. [`PLAN.md`](PLAN.md) was approved after it and **wins wherever the two differ**. The
> differences known at the copy:
>
> - **Cameras (D1):** 4 cameras, not 3 — ZED 2i front + DFRobot SEN0583 ToF (320×240, 0.2–2 m) left, right and
>   rear (user, 2026-10-16). Back travel is covered, so back field travel counts for the student; K = 4.
> - **Wrong paths in the body:** the sideways scene cfg lives under `crab_hex_sideways_task`; the forward
>   phase-pipeline REPORT is `FWD/experiments/2026-09-07_1330_phase_pipeline/REPORT.md`; the USD is repo-root
>   `assets/crab.usda`.
> - **Gates (§3.1, §3.2, §3.5, D12, D13):** the HALT set is PLAN's "decisive lines" — the catastrophe floors (lower
>   at s1–c6x, values fixed at STOP-0) plus p2a field_hard, p3a stable-at-90 + the paired 0-kg check, student
>   completion, and the stand lines. Everything else is WATCH. Retired as HALT lines: tripod, sweep, grid, turn
>   levels, the 50 N gait set, heading drift, the student grid/sweep lines, the "stable at L" support_feet /
>   swing_apex lines, and the §3.1 demotion order.
> - **False-HALT bound (D12, "≲ 0.04"):** superseded. Each read's chance rate is reported with a stated model
>   (PLAN Gates → Rule); the run sum is reported, not capped. **HALTs are prompts:** a HALT accepted as chance lets
>   the run continue and keeps it eligible as the deliverable, disclosed (D13 superseded; PLAN step 12).
> - **0-kg check (§3.4):** paired against the run's own p2a head on one row list (suite es1–2, side22, field_hard,
>   field_light, ring_obst, field_turn), margin max(R, 3·SE_diff, the X12 − X08 continuation drift scaled by
>   iterations past p2a / 5000). Not the absolute p2a lines.
> - **Payload:** p3b/p3c floor misses and live kills are measured ceilings, never run HALTs; budget-bound comes from
>   the training tail, not an N_p/2 eval; the Σ-foot-force check is hard only at the step-8 settle.
> - **K4 (D9):** ≤ 3/800 per env seed, approved at STOP-0.
> - **R18 verdict (§5):** two verdicts — "process reproduced" (P-config, P-early, P-live; STOP-R only on this) and
>   the R18 gates read like F19's. The P-gates are not part of a gate pass.
> - **Confirmed as the design recommends:** D2 (velocity cols privileged), D8 (force side report-only, sim-plant
>   label), D10 (full R18 + X), D16 (sweep, report-only), stand nets teacher-side. ONNX export kept in the student
>   step (user).

---

# M18 Task 4 on the omni task: synthesized implementation plan

Read-only synthesis. I read the six mapping reports (t4map/agent0-5) and the three plans (reuse, risk, repro). Where the plans disagree on a code fact, I re-opened the code. Nothing was edited, launched or trained.

## Path prefixes

| Prefix | Path |
|---|---|
| R | `/home/nickmagus/krabby/krabby-research` |
| OMNI | `R/parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task` |
| FWD | `R/parkour/parkour_tasks/parkour_tasks/crab_hex_forward_task` |
| T | `OMNI/experiments/tools` |
| RC / RP | `T/run_candidates.py` / `T/run_phases.py` |
| H | `FWD/scripts/eval_crab_hex_gait.py` |
| GE | `FWD/scripts/gait_eval` |
| RSL | `R/parkour/scripts/rsl_rl` |
| PIL | `R/parkour/parkour_isaaclab` |
| PPO | `OMNI/experiments/2026-10-01_ppo_construction` |
| P2 | `OMNI/experiments/2026-10-07_omni_phase2` |
| BC | `OMNI/experiments/2026-10-15_body_cmd` |
| SN | `OMNI/experiments/2026-10-11_stand_network/scripts` |
| SH | `OMNI/experiments/2026-10-13_stand_height/scripts` |
| MAN | `OMNI/policy/manifest.yaml` |
| TP (new) | `T/pipeline/`: undated, reusable tools; no dated path in source; no `__init__.py` |
| PL (new) | `OMNI/experiments/<first-commit-date>_task4_pipeline/`: the M18 instance, with spec, thresholds, records, and the run dirs `preflight/`, `R18/`, `F19/` |

---

## 0. How the three plans compare

### 0.1 Code facts I re-checked where the plans disagree

| # | Claim | Verdict (source) |
|---|---|---|
| 1 | The direction sweep needs no harness change (risk, repro) | **Wrong.** Rigidity and motion-quality keys are looked up by category **name** (`GE/metrics.py:1295-1305`, `:1613-1622`). `own_category` is the category name (`GE/schedule.py:375-376`, `GE/report.py:225-231`). A category named `d022` therefore gets `None` for `steps_per_m`, `taps_per_m`, `swing_apex` and `support_feet`. Reuse's opt-in `metric_class` field, part of the eval change, is needed. |
| 2 | Put the actuator metrics, h_meas or `swing_s` into the harness (risk, repro) | **This breaks the inertness bridges as built.** `bc_bitwise.py` compares `scenario_metrics.json` `aggregate` and `episodes` as whole JSON (`BC/scripts/bc_bitwise.py:44-50`), so new keys fail the bridge. Every input is already in the raw npz (`--save-raw` defaults on, H:90; arrays at H:1289-1324). Reuse's reader over the raw npz keeps the bridges bitwise. The spec permits "a documented scoring step". |
| 3 | Mask velocity cols 13-14 in the student (reuse, repro) | **Unsafe as the default.** The frozen estimator's target is `root_lin_vel_b × 2` (PIL `envs/mdp/observations.py:140-146`). Its input `obs[:, :75]` contains cols 13-14, which are the same velocity × 2 (FWD `mdp/observations.py:206-216`). Masking them removes the estimator's x/y signal, and `learn_vision` never trains the estimator. Risk's "keep privileged" is the safer default; this is decision D2. |
| 4 | Mask cols 11-12 (reuse) | Cols 11-12 are `env_idx` and `invert_env_idx` (FWD `mdp/observations.py:211-212`). Zeroing both gives an impossible one-hot. Use the constant "field" pair (risk, repro). |
| 5 | Set `STUDENT_PHASES` (reuse, repro) | `tests/unit/test_crab_hex_omni_phases.py:157-158` asserts `== ()` and no distill phase. No preset is needed because the student env is composed from T\*'s overlay. Leave it at `()` (risk). |
| 6 | Build and arm a composed camera event (reuse), or build it and leave it off (repro) | The teacher sets `random_camera_position = None` (`extreme_parkour_task/config/go2/parkour_teacher_cfg.py:61`), and the student inherits from the teacher. With no event there is no frame bug and no code. Camera-pose randomisation is M15 scope (risk). |
| 7 | Seed 19 is "used" (risk; it proposes 21) | `FB_C_lr` and `P1_C_lr` s19/s20 are declared references (`PPO/candidates.yaml:2926-2952`) that never ran: no run dir, and not in `state.json` references. Seed 19 is clean. |
| 8 | Repro's line numbers | Not consistent with its own rule. Its field_hard line of .675 implies m = .05, but 2·SD(.815, .725, .8125) = .103. Its "downgrade to WATCH if margin < 1σ" can never fire, because m ≥ 2σ. Reuse's and risk's walker lines agree with each other and with min − range. |
| 9 | `2026-09-19_facing_tracking_ladder/resolution.74f93b91dc8b.json` is a committed resolution input (all three) | **Untracked** (`git status ??`; another session's file). The tracked input is `2026-09-23_clock_deprecation/resolution.36e5a8af7000.json`. |
| 10 | Student code paths: trainer, harness, play.py and exporter (all) | **All three missed** `RSL/play_stream_video.py:207-213`. It has its own student branch with the cols 6:8 overwrite and is in code_sha (`parkour/scripts`). The shared student step must cover it too. |
| 11 | Stand launchers: point `P2_GUARD_REF` at a new sidecar (risk, repro) or set `P2_REANCHOR=1` (reuse) | The launchers hard-code `P2_GUARD_REF=REFH_S1_s18` inside `cmd=(env …)` (`SN/sn_eval_rows.sh:93`, `SH/sh_eval_rows.sh:167`). They inherit `P2_REANCHOR` from the parent env (`P2/scripts/run_arm.sh:251`). Reuse's approach needs no launcher edit; the pipeline guard enforces the pin. |
| 12 | `launch_candidates.sh` guarantees one Isaac process | **Partly.** It waits at most 10 min, and only for eval/train/probe processes, not for DAgger trainers or play (`T/launch_candidates.sh:29-38`). The queue must gate every launch with its own anchored `ps -eo args` check (the BC `ISAAC_PAT` pattern, `BC/scripts/bc_eval_row.sh:19`). |
| 13 | Where the code lives | Every candidate location sits outside CODE_PATHSPECS and EVAL_PATHSPECS (RC:164-187). `T/pipeline/` is not even in tools_sha, whose TOOL_FILES are 3 files (RC:204). Dated `PL` paths are dropped by `*/experiments/20*` (RC:200). So the lock file must pin them by sha256. |

### 0.2 Plan-level comparison

| Criterion | Reuse | Risk | Repro |
|---|---|---|---|
| Correctness | Best. Right on the sweep metrics, the bridges, the guard and the bundle tool. Wrong on 11-14 masking, STUDENT_PHASES and the camera event. | Right on the camera event, velocity and STUDENT_PHASES. Wrong on the sweep and the harness metrics. Seed reasoning is off. | Wrong on the sweep and the harness metrics. Thresholds are inconsistent and the downgrade rule is vacuous. |
| Completeness (4a-4e and the user's decisions) | Complete, plus the stand-tooling replay beside the bundled S1. Lacks a parity check at HEAD before building and a payload feasibility probe. Its per-stage budget of 0.10 sums to about 30-50 % per run. | Complete. Best risk register: Phase A at HEAD, R1-R10. Lacks a stand replay against SH_T17. | Complete. Per-run self-contained records. Live bitwise check at +200. 4c falls back to R18. Its stand add-on runs beside R18's walker, which can only give a statistical comparison. |
| Risk ordering and STOPs | Good. The pilots come after the builds. | Best: cheapest disqualifiers first, and a depth vs zeroed-depth pilot. "PARTIAL → STOP" would fire by chance at about 30 %. | Clean split between the low-side gate and a high-side WATCH. |
| GPU budget | Most itemised (125-138 GPU-h). | Optimistic: 42-49 per run, with a thin student screen and thin payload. | 110-140 GPU-h. None of the three measured the cost of K = 3 cameras. |
| Minimal new code | Best: reader metrics, `P2_REANCHOR`, opt-in `metric_class`. Builds an unnecessary camera event. | Medium: harness metrics, linkage promotion, `bundle_policy` changes before any bake. | Most: unarmed event, `swing_s`, an actuator module, an 8-subcommand driver, `bundle_policy` extensions. |
| Reproducibility | Config parity uses trainer-written `params/*.yaml`. Stand replay beside the bundled S1. | Parity at HEAD before building; external stack pinned. | Fresh campaign dir per run; live +200 bitwise check; tiered verdict. |

**Spine:** reuse, for its code correctness, its reuse of the BC tooling, and its itemised budget.

**Grafted from risk:**
- Phase A at HEAD, run before any build.
- The payload feasibility probe.
- The depth vs zeroed-depth pilot.
- No camera event.
- Keep velocity as the default.
- Controls run only when needed ("lazy").
- Student before the stand nets.

**Grafted from repro:**
- One campaign dir per run.
- A single entry point.
- The live +200 bitwise check.
- HALT only on capability lines; catastrophe lines everywhere.
- The payload staging stop rule.
- The 4c fallback to R18.
- Known-bad tagging of every line.
- `num_prop` 89.
- The 9-cell in-envelope grid.

---

## 1. Summary

- **One committed spec.** `PL/pipeline.yaml` is run by a single entry point, `TP/pipeline.py`, with subcommands `gen`, `check`, `run`, `continue`, `status`, `gate` and `report`. Each official run is one command: `pipeline.py run --run R18` (then `F19`).
- **Stages.** Six walker stages and up to three payload stages run through the unchanged `RC --reference-chain`, each stage resuming `{reference: <previous>}`. After them come a CPU step that picks the final teacher T\*, the depth student, then A-mask → stand-height DAgger. Stage gates are read between stages by a CPU gate reader against frozen absolute lines, and the run halts on FAIL. There is no manual hand-off anywhere: every input is read from the record the previous stage wrote.
- **Threshold rule.** One rule (§3.1) derives every line from the recorded s16/s17/s18 stamps of that stage. Every recorded seed passes; each HALT line names a known-bad that fails it, or is tagged `unvalidated`. A run-level false-HALT bound of ≤ 0.20 is met by moving statistics that are not part of the stage's capability set to WATCH. Gait gates use the 50 N rigidity metrics.
- **One move of each hash, then a freeze.** New code lands as one code_sha move (M1) and one eval_sha move (E1), both proven inert bitwise. Then comes a local annotated pin tag. The guard halts on any drift until both runs end.
- **Two official runs, serial.**
  - **R18** is the seed-18 replay, with DAgger seed 17. Parity is judged on: config identity, bitwise training up to the priv_reg onset (about iteration 1905), every gate passed, and a stand-tooling replay beside the bundled S1 compared with SH_T17.
  - **F19** is fresh seed 19 throughout.
  - The 4c run of record is F19 if every gate passed, otherwise R18 if every gate passed.
- **Budget.** About 125-155 GPU-h (5.2-6.5 days serial on the RTX 5080). The 200-iteration student timing smoke narrows the range. Dev effort is about 8-11 days against the grant's 3.

---

## 2. Pipeline design

### 2.1 Mechanics

```
pipeline.py gen  --run R18|F19   → PL/<run>/candidates.yaml   (generated, committed; one reference per stage)
pipeline.py run  --run <run>     → detached queue (setsid nohup; resumable; PushNotification at every halt)
  per stage:
    guard     live code_sha/eval_sha (RC's own code_sha()/eval_sha()/code_drop) == pin; CODE+EVAL clean;
              sha256 of every file in PL/pipeline.lock.json; assets/crab.usda sha; Isaac/IsaacLab/rsl-rl versions;
              ≥150 GB free; DerivedDataCache size guard (sh_dagger.sh pattern); Isaac idle (anchored ps -eo args)
    driver    T/launch_candidates.sh --campaign-dir PL/<run> --reference-chain <stage>_<run> --seed S
    dagger    SN/sn_dagger.sh | SH/sh_dagger.sh   (walker / warm start from the previous stage record)
    distill   TP/distill.sh → RSL/train.py --task Isaac-Crab-Hex-Omni-Student-v0 --resume --checkpoint <T*>
    live      TP/live_watch.py: resume read at +0; dead-walker (s1, its 800-1000 track < 0.35);
              collapse after +300 (episode length < 0.5× source tail, crab_failure > 0.6, NaN) → driver STOP file
    after     TP/after_stage.py: head from state.json ladder.references[chain][-1]; params/env.yaml audit;
              gate rows via BC/scripts/bc_eval_row.sh (BC_OUT=…/task4_pipeline; env = the stage's driver
              suite stamp, or a generated KEY=VALUE file for payload levels / student --task)
    gate      TP/gate_read.py → PL/<run>/gates/<stage>.json; exit 0 PASS, 2 PASS+WATCH (notify),
              3 HALT, 4 INVALID (rerun once, then halt), 5 GUARD (halt; resumable once the tree is restored)
```

- **Per-run campaign dirs.** `PL/R18/` and `PL/F19/` are created at the pin, so neither launch needs `--accept-code-change`. The pre-pin smokes live in `PL/preflight/`.
- **Records are tracked.** Each run's `candidates.yaml`, `state.json`, `gates/*.json`, `stage_reads/*.json` and `halt.json` are committed. A `PL/.gitignore` keeps `*.log`, `logs/`, `raw/`, `metrics/` and `STOP*` out of git.
- **Stage hand-off.**
  - Driver to driver: `{reference:}` resolves only runs at the same code_sha (RC:13267-13283), so a code move halts instead of forking a stale head.
  - Driver to DAgger, and teacher to student: `stage_reads/<stage>.json` carries `{ckpt, sha256, run_dir, agent.yaml, overlay ref, REFS stamps}`.
- **Never killed mid-training**, except by the pre-registered live watcher.

### 2.2 Stage table (per run)

Seeds by run:

| Run | Walker, payload and student seed S | DAgger seed DS |
|---|---|---|
| R18 | 18 | 17 (recipe of record) |
| F19 | 19 | 19 |

Common settings: 256 envs for driver stages, plant A15+B, `clip_actions` 1.0, `absolute_units: true`, `floor: {kind: none}`. The generator copies every overlay verbatim from the shipped files and checks it by sha256. No stage is named "2a" or "2b", because `compose_recipe` blanks those (RC:1067).

| # | Stage | Kind | Base / task (envs) | Resume | Iters | Overlay (source) | Categories |
|---|---|---|---|---|---|---|---|
| W1 | `s1` | driver | 1b / Omni-Walk-v0 | `scratch` | 6000 | FB6k__B (PPO/candidates.yaml:1288-1313), 14 keys | back, fwd, side, stand |
| W2 | `c4` | driver | 1a | `{reference: s1}` | 1500 | SF__B_c4_s18 (:2443-2469), 16 keys | + move_turn, turn |
| W3 | `c5` | driver | 1a | `{reference: c4}` | 1500 | SF__B_rest_s18 c5, 16 keys; **adds `eval: [suite]`** (eval only, precedent `BC/scripts/bc_make_candidates.py:13-14`) | all 7 |
| W4 | `c6` | driver | 1a | `{reference: c5}` | 2000 | c6, 16 keys | all 7 |
| W5 | `c6x` | driver | 1a | `{reference: c6}` | 4000 | c6x, 19 keys | all 7 |
| W6 | `p2a` | driver | 2a / Omni-Teacher-v0 | `{reference: c6x}` (replaces the literal path at P2/candidates.yaml:548) | 5000 | P2__S1_s18 (:540-579), 27 keys | all 7 |
| P1 | `p3a` | driver | 2a | `{reference: p2a}` | N_p | p2a overlay + `PAYLOAD_KG 0:90`, `_ZERO_FRAC 0.25`, `_COM_XY 0.10`, `_COM_Z 0.309` | all 7 |
| P2 | `p3b` | driver | 2a | `{reference: p3a}` | N_p | `KG 0:180`; trained only if p3a passes | all 7 |
| P3 | `p3c` | driver | 2a | `{reference: p3b}` | N_p | `KG 0:270`; trained only if max_stable(p3b) ≥ 180 | all 7 |
| T | `tstar` | CPU | — | — | — | Rule §3.4 picks T\* | — |
| S1 | `st` | distill | Omni-Student-v0 (192) | `--resume --checkpoint T*` (counter continues) | 5000 | `KRABBY_OMNI_PHASE=2a` + T\*'s overlay, minus the runner-only keys (`ACTION_BOUND_LOSS/_RAMP/ACTION_BOUND`, `ENTROPY_COEF`, `PPO_HAND_DIAG`, `OMNI_MIRROR_IMAGES`) | — |
| S2 | `sel` | screen | Omni-Student-Play-v0 | — | — | Rule §3.5 over +1000 … +5000 | — |
| D1 | `sn` | DAgger | Teacher-v0, 2a + P2__S1_s18 overlay (0 kg) + the SN changes (1024) | walker = T\* frozen; warm start from T\* | 4000 | `sn_dagger.sh -- --mode dagger --arm mask --seed DS --iterations 4000` | — |
| D2 | `sh` | DAgger | same env | warm start = SN head with W[:,15] zeroed | 5000 | `sh_dagger.sh --wait-idle … --deadband 0.25 --trim-branch --trim-min 0.0999 --warm-start <D1> --warm-sha <sha>` | — |
| X | `sn_ref`, `sh_ref` (R18 only) | DAgger | as D1/D2 but beside the **bundled S1**, DS 17 | the recipe of record | 4000 / 5000 | `SN_WALKER_SPEC` unset | — |
| B | `bundle` | CPU | — | — | — | Keep-set + README (§2.4) | — |

- **Order after T\*:** student first (S1, S2), then D1, D2, then X. A HALT in the student branch stops that branch only; the stand branch still runs, and the user is notified.
- **N_p, the payload iterations**, is set before the pin by the payload pilot (§5, C3): the first read at which the 90-kg level's translate ratio and survival changed by less than R from the previous read, rounded up to 500, clamped to [1500, 3000]. The same N_p is used for all three payload stages.

### 2.3 Pin

- **Pin.** Builds land as **M1**, one code commit series (code_sha a8519343 → pin), then **E1**, one eval commit (eval_sha 567d1704 → pin), then the CPU tests and the GPU inertness proofs. After that come the pin commit and a local annotated tag `omni-pipeline-pin-<date>`.
- **Freeze.** No code or eval move is allowed until F19 ends (about 6 days). Other sessions must stay off CODE_PATHSPECS, EVAL_PATHSPECS, TP and PL (decision D14).
- **`PL/pipeline.lock.json`** is written by `gen` and verified by the guard. It holds:
  - the commit, code_sha, eval_sha and tools_sha;
  - the sha256 of every TP file;
  - the reused out-of-hash files: BC scripts, `PPO/scripts/side22_classify.py`, `along_cmd.py`, `P2/scripts/p2_read.py`, `p2_live.py`, `run_arm.sh`, the SN/SH scripts and `SH/sh_kin.py`;
  - the row manifests: `2026-09-28_side_bisect/scenarios_side_lr.yaml`, `PPO/scenarios_turn_band.yaml`, `P2/scenarios_p2.yaml`, `P2/scenarios_p2s.yaml`, the SH scenarios and `PL/rows/*`;
  - the resolution inputs and `thresholds.json`;
  - `assets/crab.usda`;
  - the external stack: Isaac Sim 5.1, IsaacLab b0542fe2d, rsl-rl-lib 5.0.1, isaaclab_rl 0.5.2.

### 2.4 Deliverable (4e)

- **The 4c run's keep-set** goes in `PL/head/` via `T/bundle_experiment.py`, which supports `head` + `extra_heads` and is checked by `test_bundled_heads_match_their_manifests`:
  - head = the selected student;
  - `extra_heads` = T\*, the SN head and the SH stand net.
  - Size is about 75 MB. Teacher checkpoints are 12.6 MB, stand nets 6.1 MB, and the forward student was 49 MB.
- **Stage heads** (W1-W6, the P stages) are recorded by sha256 only, unless the user wants them shipped (D18).
- **`PL/README.md`** is generated. It holds the per-stage re-run commands, the gate commands, the play command with the carry env and `--task`, and the selection record.
- **Tag** `omni-sim2real-baseline-<date>`: annotated and local.
- **The user's alone:** pushing the tag, and baking into `OMNI/policy/`. A bake would add `kind: distill` to `T/bundle_policy.py:121-125` and rewrite `tests/unit/test_policy_of_record_omni.py` (:47, :132-174, :422-521).

---

## 3. Gates

### 3.1 Threshold rule

Committed in `TP/thresholds.py`; output `PL/thresholds.json`.

**Inputs** for statistic x at stage k:
- The three recorded seed values from that stage's stamps:
  - PPO X40/X42/X43 (s1), X55/X56/X57 (c4), X58/X59/X60 (c6 and c6x);
  - P2 X10/X11/X08 (p2a), plus REF_S1_s{16,17,18} (field);
  - ANCW (c6x field);
  - `ANC_S1_s{16,17,18}` (body block);
  - the new sweep anchors (S1 s16/s17/s18, read in preflight).
  - The C0 arms (X39, X41, X44, X61-X64) are excluded; they serve as known-bads.
- **R_x**, the larger of the paired resolutions in `2026-09-23_clock_deprecation/resolution.36e5a8af7000.json` (tracked) and `2026-09-19_facing_tracking_ladder/resolution.74f93b91dc8b.json` (untracked; see D19).
  - Proportions also get the binomial guard G(n) (`T/resolution.py:350-366`; 0.0717 at n=100, 0.0507 at n=200).
  - Statistics with no resolution entry (rigidity, field) use 2·SE from the episode bootstrap of the head's own stamp (`T/resolution.py`).
  - Training-log tails use a floor of 0.03.

**Lines:**
- Higher-is-better: `min(x) − max(R, range(x), G)`.
- Lower-is-better (`taps_per_m`, `taps_per_rad`, `steps_per_m`): `max(x) + max(R, range(x))`.
- Tracking: `|r − 1| ≤ max|x − 1| + max(R, range|x − 1|)`. A seed that tracks better is never failed.
- Class: the worst class recorded across the three seeds at that stage (side22 is FAST/FAST everywhere on record; `side22_classify.py:106-118`).
- **Catastrophe lines at every stage**, including c5, which has no anchors: translate survival ≥ 0.90; every translate ratio ≥ 0.70 (recorded minimum .798); live collapse.

**HALT vs WATCH.**
- HALT = the catastrophe lines plus the stage's **capability set** (§3.2). Everything else is WATCH and notifies.
- **Self-checks** (`thresholds.py` exits non-zero if any fails):
  1. every recorded seed passes every line;
  2. each HALT line names a known-bad that fails it, or is tagged `unvalidated` in the README. Known-bads: the C0 arms for side22 at s1; ANCW c6x .435/.56/.50 for the p2a field lines; P2 zero-shot body mode for the turn levels; Z1/Z2d/ST2M/C0R for the stand gates. The drift line has no known-bad, because P2 passes it (`BC/anchor_numbers.md`);
  3. **run-level false-HALT bound.** Σ over HALT lines of Φ((line − x̄)/σ̂), with σ̂² = (range/1.693)² + (R/1.96)², must be ≤ 0.20. If not, statistics move from HALT to WATCH in a pre-registered order (rigidity first, then body field rows, then move_turn), never touching the capability set.
  - With two official runs, the chance that both fail on chance alone is ≲ 0.04.
- **Validity at the pin.** The recorded anchors were read at eval_sha 1d021625. The E1 bridges (§9) re-run X43 s1, X60 c6x, X10 p2a and S1 rows bitwise, so the recorded numbers stand.

**Why min − range.** For three normal draws, E[min] ≈ μ − 0.85σ and E[range] ≈ 1.69σ. The line therefore sits at about μ − 2.5σ, about 0.6 % per statistic.

### 3.2 Walker gates

Provisional numbers below are worked by the rule from the recorded values; reuse and risk agree on all of them. `thresholds.py` is the record. Rows run at es1 unless marked.

| Gate on head | Rows | HALT: capability set | WATCH |
|---|---|---|---|
| s1 | driver `suite__omni`; `side22_25lr__omni` | side22 FAST/FAST (known-bad: C0 0/3); \|r−1\| fwd ≤ .130, side ≤ .162, back ≤ .136; translate survival; track tail ≥ .708 | diag (untrained), turn, rigidity, 1 N metrics |
| c4 | suite; side22 | FAST/FAST; \|r−1\| fwd ≤ .199, side ≤ .171, back ≤ .282; translate survival; track tail ≥ .698 | turn fc and move_turn survival (seed range 0-.945) |
| c5 | suite (new) | catastrophe lines + track tail ≥ .744 | everything (no anchors) |
| c6 | suite; side22; `turn_band__omni` | FAST/FAST; \|r−1\| ≤ .179 / .068 / .276 / .170; translate survival; track tail ≥ .756 | turn_band, rigidity |
| c6x | suite; side22; turn_band; field_hard / light / ring es1-2 | \|r−1\| ≤ .082 / .066 / .084 / .087; turn fc ≥ .631; yaw_progress ≥ .086; move_turn survival ≥ .495; FAST/FAST; track tail ≥ .797. INVALID unless the stamp has `CLOCK_HIDE=1`. | field rows (before field training) |
| **p2a** (head-of-record capability) | suite es1-2; side22; field_hard es1-4 (n=400); field_light, ring_obst es1-2; field_turn; field_suite; body block `suite / turnlevels / grid__omni_body` es1-2 + body field rows (via `--command-adapter body_hold`); `dirsweep__omni_body` es1-2 | **Tracking:** \|r−1\| ≤ .096 / .083 / .157 / .098; move_turn ratio ≥ .639; turn fc ≥ .898; yaw_progress ≥ .129; move_turn survival ≥ .885; translate survival ≥ .928. **Field:** field_hard ≥ .635; field_light ≥ .930; ring ≥ .949; field_turn ≥ .888. **50 N gait:** tripod per translate category; steps_per_m fwd/side; swing_apex fwd/side. **Grid:** turnlevels net ratio w_min ≥ .858, ceiling ≥ .835, w_max ≥ .724; the 9 in-envelope cells (\|v\| 0 × wz {0, .08, .15}; 0.22 × {0, .08} fwd/lat; 0.30 × 0 fwd/lat) with \|achieved − cmd\| inside the seed band; drift median ≤ .0100 and p90 ≤ .0207 (`unvalidated`). **Sweep:** per-bin \|r−1\| line; max adjacent-bin jump in ratio, tripod and steps_per_m ≤ the anchors' max + range. **Plus** FAST/FAST and track tail ≥ .782. | taps_per_m/rad, support_feet, the other rigidity values, tripod on turn/move_turn, body field rows, stand steps_per_s (S1 marches in place at 2.5-2.8; that is the stand net's job), out-of-envelope cells, BC g2 ripple (recorded STOP), 1 N metrics |

**Invariant checks (gate 0)** make a stamp INVALID unless all of these hold:
- the stamp is complete;
- `code_state.json` shas equal the pin, and the tree is not dirty;
- the checkpoint sha equals the stage record;
- the carry env holds: `CLOCK_HIDE` 0 through c6 and 1 from c6x; `STD_PARAM` soft; `INIT_STD` 0.8; `SPEED_ELLIPSE 0.35:0.35:0.25`; `FACING_OBS_CLIP 0.6`; `GEOM` ring (field rows: their own block); `COMMAND_FRAME=world` allowed (G0w inert);
- walker rows have no `STAND_OBS`, `P2R_*`, `SH_*` or `SN_STAND_*`;
- body rows record `body_hold`.

The benign phase_check FAIL "omni RSI bank path" (RSI_FRAC=0) is whitelisted by exact message.

**What passing means.** `CHARTER.md` states it per stage, and the README carries it:
- s1: a side-first walker formed, with fast lateral strides on both hands;
- c4-c6: the command widening cost no translation;
- c6x: the clock-free walker tracks and turns;
- p2a: head-of-record capability, at the standard spanned by the three recorded seeds.

### 3.3 Payload levels (a priori lines, user-approved)

**Self-check:** S1 at 0 kg passes every line.

**"Stable at L"** means all of the following on `suite__omni` es1-2 and `field_light__omni` es1-2:
- translate survival ≥ 0.90;
- turn and move_turn survival ≥ .885;
- translate \|r−1\| ≤ 0.25 (a slower loaded gait is allowed; TASK-3 "lower, wider, slower");
- turn fc ≥ 0.80;
- field_light completion ≥ 0.85;
- median h_meas ≥ 0.85 m (above the 0.719 m stand floor, so a deliberate crouch passes; belly collapse already terminates at > 500 N hip contact);
- roll and pitch RMS ≤ 0.10 rad;
- support_feet ≥ 2.5 and swing_apex ≥ 0.05 m (still stepping).

**How levels are set.** Levels go through pre-Kit env only, as a KEY=VALUE env file through `bc_eval_row.sh`: `KRABBY_PAYLOAD_KG=L:L`, `_ZERO_FRAC=0`, `_COM_XY=0`. Levels never go in a manifest `env:` block, which the harness writes after Kit starts (H:1059-1070). The manifest shape is identical at every level.

**Report-only actuator metrics** (`TP/payload_read.py`, from the raw npz):
- knee/hip effort-cap share, |τ| ≥ 0.98 cap (`applied_torque` is the implicit-actuator estimate; caveat stated);
- rod force p50/p90 against rated (2000 / 500 N) and self-lock (2800 / 4500 N), via `SH/sh_kin.py` rod math;
- h_meas sag against 0 kg;
- Σ foot force against weight.

### 3.4 Payload stages and the final teacher

- **Per p3x stage:**
  - a reduced 0-kg check: suite es1-2, field_hard es1-2 and field_light es1-2, against the p2a HALT lines re-derived at matching n;
  - levels {45, 90, …, 270} in ascending order, stopping at the first unstable level.
- **max_stable(head)** = the largest L such that every level ≤ L is stable.
- **p3a: HALT unless stable at 90** (criterion 3d). A p3a failing its 0-kg check also halts.
- **Staging (repro, matching the grant's "as long as training stays stable"):**
  - train p3b only if p3a passed;
  - train p3c only if max_stable(p3b) ≥ 180.
  - A p3b or p3c that misses its own top level is a measured ceiling, decide-record-notify, not a gate failure (D7).
- **T\*** = the argmax of max_stable over trained heads that pass their 0-kg check; ties go to the earlier stage.
- **T\* full block** at 0 kg is the 3f "no unloaded regression" check and the Task-2 capability check on the final teacher:
  - the p2a world, body and sweep rows. HALT on the p2a capability set; the paired Δ against its own p2a head is WATCH and reported;
  - then the full 0-270 curve, every 45 kg.
- **Limiting factor** at the first failing level, first match wins:
  1. training instability: a live kill or NaN, or the training tail below its line;
  2. actuator: cap share ≥ 2× its 0-kg share **and** h_meas sag ≥ 0.05 m;
  3. stability: survival, tilt or height;
  4. tracking.

### 3.5 Student (4d)

**Live checks during training:**
- `depth_actor_loss` finite, and down ≥ 50 % by +250;
- last-50 mean episode length ≥ 0.8× T\*'s training tail;
- crab_failure ≤ T\*'s tail + 0.10.

The forward check (≥ 600 steps, < 0.9) passed a student that fell in 36 % of episodes (FWD phase_pipeline REPORT:34), so it is too weak.

**Candidates:** +1000, +2000, +3000, +4000 and +5000.

**Prescreen** (Student-Play): suite es1 and field_hard es1-2. Order the candidates by field_hard, then by mean \|r−1\|.

**Full block, in prescreen order** until one passes: the p2a world, body and sweep rows, plus the payload levels {0, 90}.

**Student lines (HALT):**
- **Paired against T\*** on the same rows and env seeds: field_hard, field_light, ring and field_turn completion ≥ T\* − max(0.05, 2·SE_diff), on the camera-covered categories. Back travel is WATCH (rear blind, D1).
- **Absolute:** p2a tracking lines + 0.05; turn fc ≥ T\* − 0.05; translate survival ≥ the p2a line − 0.05; FAST/FAST; grid and sweep core lines + 0.05.
- **Payload:** stable at min(90, max_stable(T\*)).
- Rigidity is WATCH.

**Selected** = the first candidate that passes; ties within resolution go to the earlier checkpoint. Then:
- the full 0-270 payload curve on the selected student;
- one headless clip of ≤ 60 s (`play_stream_video.py` under `systemd-run MemoryMax`), shown in the notification. The user can veto it at STOP-END. It is never a gate.

If no candidate passes → HALT, student branch only.

### 3.6 Stand (method of record, re-anchored on T\*)

- **Always run beside T\*:**
  - REFS(T\*): walker-alone rows (suite@1,2, walkstop@1,2, field_hard@1-4, field_turn@1);
  - the SN identity bridge (`SN/sn_bridge_read.py`).
- **SN → SH gate** (`SN/sn_calib.py` G1-G5, with NAMES taken from a spec):
  - G3 stand falls ≤ 1, frozen absolute;
  - G4 field_hard ≥ REFS(T\*) − max(0.05, 2SE);
  - G5 onset from REFS(T\*);
  - G1/G2 absolute lines as coded.
  - The S1-era known-bad calibration (Z1/Z2d/ST2M/C0R) stands once and is not re-run.
- **SH → done** (`SH/sh_read.py --spec`, es1 + es2):
  - **K1:** worst-bin median ≤ 20 mm, p90 ≤ 40 mm.
  - **K2:** p90 arrival ≤ Δh/0.08 + 1 s; falls ≤ max(1, R_h).
  - **K3:** STOOD ≥ 0.933 / 0.928 (S1-era R_h line); falls ≤ 1/100.
  - **K4:** misses ≤ **3/800 per env seed**. The recorded DAgger seeds were 1/2, 0/1 and 1/3 (SH/REPORT.md:131); D9. Exit falls ≤ 1; onset ≤ REFS(T\*) + Δh/0.08 + 1.5 s.
  - **K5:** G1-G5 at 1.064 against REFS(T\*).
  - **K6:** homed ≥ S1-era R_h − max(5 pp, 2SE); ≤ 0.5 steps/s; ≤ 1 tap/s.
- **Lazy controls (risk).** R_h-, R- and HOLD-derived lines are frozen at their S1-era values. If any of them fails, R_h/R/HOLD run beside T\* (about 3 GPU-h).
  - If the control fails the same line, the line re-anchors on the control. This is walker-caused, and is decide-record-notify.
  - Otherwise → HALT (stand branch).
- The stand net is trained and evaluated at 0 kg only; standing under payload is a recorded follow-up.

### 3.7 Failure handling

- **FAIL** halts the dependent subtree, records and notifies. There is no automatic retry, because a same-seed retrain is a replay. The user chooses one of:
  - (a) accept as chance-level with a recorded reason (precedent SH_T18 K4). That run then forfeits 4c;
  - (b) abort;
  - (c) amend a line in a CHARTER amendment that discloses the failing number, which makes a new spec version, before any further run is read.
- **INVALID:** rerun once, then halt.
- **GUARD:** never overridable inside a run.
- Every verdict can be re-read on the CPU from saved stamps: `pipeline.py gate --stage X --thresholds vN`.

---

## 4. Builds

### M1: code (one commit series; all default-off, or forward bit-identical)

| Item | Files | Notes |
|---|---|---|
| B1 Payload knobs | new `OMNI/config/crab_hex/payload_knobs.py` (`clock_knobs.py` pattern) | `KRABBY_PAYLOAD_KG lo:hi`, `_ZERO_FRAC`, `_COM_XY r`, `_COM_Z z`. `apply_payload_knobs` is called from `apply_omni_knobs` (`omni_knobs.py:1653`) before `_apply_command_frame` (:1717). Unset → writes nothing. Box inertia is a code constant: the deck footprint 0.711 × 1.219 × 0.30 m (`FWD/mdp/crab_hex_dimensions.py:46-53`). |
| B1 Payload event | `payload_reset(ManagerTermBase)` in `OMNI/mdp/omni_events.py`; created dynamically when armed (`rsi_reference_reset` pattern, `FWD/config/crab_hex/crab_hex_env_cfg.py:551`) | **Captures each env's mass, CoM and inertia lazily at the first reset**, which is after the startup DR events have run, so p2a's startup DR (mass −0.5:1.5, CoM ±0.01) is preserved exactly. On reset of `env_ids` it sets an absolute composite: mass, CoM, and parallel-axis inertia. It uses a private `torch.Generator` seeded from the env seed, and logs extras. Written fresh: the IsaacLab and parkour CoM functions are cumulative and break on partial resets (`PIL/envs/mdp/events.py:261`). |
| B3 Student env | `CrabHexOmniStudentSceneCfg(CrabHexOmniTeacherSceneCfg)` in `OMNI/config/crab_hex/crab_hex_omni_scene_cfg.py` | Three RayCasterCameras: front (0.37, 0, 0.10) and left/right (0, ±0.625, 0.10), intrinsics from `parkour_tasks/default_cfg.py:88-106`. Rotations by quaternion composition with the tripwire (`SIDE/config/crab_hex/crab_hex_side_scene_cfg.py:58-94`). Pitch per D1. Raycast `/World/ground` only (field blocks are in it; tables are not). **No camera event.** |
| B3 | `CrabHexOmniStudentObservationsCfg` + `CrabHexOmniStudentEnvCfg(CrabHexOmniTeacherEnvCfg)` (192 envs) + `…EnvCfgPLAY(CrabHexOmniFlatWalkEnvCfgPLAY)` in `crab_hex_omni_env_cfg.py`; ids `Isaac-Crab-Hex-Omni-Student-v0` / `-Student-Play-v0` in `config/crab_hex/__init__.py` | Observations = the omni policy term (:90-121) + one depth group per camera; no `delta_yaw_ok`. Post-init: `install_omni_terrain_generator` → super (mode 2a) → camera `update_period`. Do not derive from the forward student cfg (Go2 scene, `is_student_phase`). `STUDENT_PHASES` stays `()`. |
| B3 Batched image term | new `image_features_batched` in `OMNI/mdp/crab_hex_omni_observations.py` | One `F.interpolate` for all envs, the forward crop and normalisation. Built up front: K=3 multiplies the per-env Python loop (`PIL/envs/mdp/observations.py:233-238`). The forward term is untouched. Reset refill as in the forward term (stale ≤ 5 steps, recorded). |
| B3 Runner | `CrabHexOmniStudentPPORunnerCfg(CrabHexStudentPPORunnerCfg)` in `OMNI/config/crab_hex/agents/rsl_rl_ppo_cfg.py` | `crab_hex_omni_student`, `clip_actions` 1.0 inherited. Applies only `STD_PARAM=soft` and `INIT_STD`; refuses any other agent knob. Omni depth-encoder subclass, so `FWD/policy/mdp_pins.yaml` does not move. |
| B4 Shared student step | new `RSL/modules/omni_student_step.py` | **One function used by the trainer, harness, play.py and play_stream_video.py:** (a) every 5th step: K images → shared conv → `combination_mlp(32K + num_prop)` → GRU → 32-d latent; `num_prop` = 89 (D4). (b) **No write to cols 6:8** (the facing command). (c) Constant "field" pair in cols 11-12, and in the history copies at 259+89j+{11,12}. (d) Cols 13-14 untouched by default (D2). (e) Estimator into cols 207-215 from the student copy; the teacher label is `act_inference` on the true obs with its own estimator write. (f) GRU reset on done rows. |
| B4 | `RSL/modules/on_policy_runner_with_extractor.py` `learn_vision` (:419-551); `distillation_with_extractor.py:58-62`; `feature_extractors/depth_backbone.py` | `omni_student=True` routes to (B4); `delta_yaw_ok` optional (:448, :499); zero-weight yaw loss; K-image path + `reset(dones)`. **Forward defaults stay bit-identical.** |
| B5 Play / export | `RSL/play.py:197-208`; **`RSL/play_stream_video.py:159-213`**; `RSL/exporter.py` | Shared step. Depth dummy (1, K, 58, 87); `num_prop` dummy. Jit `depth_latest.pt`; ONNX depth recorded as stateless (an M15 item). |

### E1: eval (one commit)

1. **H student branch (:1515-1527):** when the checkpoint's `params/agent.yaml` marks an omni student, call `omni_student_step`. The forward path is unchanged. Move the command-reaches-obs check (:1500-1503) after the step.
2. **run_meta:** env-0 CoM and inertia next to the masses (H:1706), plus per-env payload mass and CoM when the payload event is armed. run_meta is not compared by the bridges.
3. **Opt-in `metric_class: translate|turn`** on a category: parsed in the v3 parser in `GE/schedule.py` and applied to the rigidity and motion-quality lookups in `GE/report.py`. Existing rows carry no such field, so they stay byte-identical.

### Outside both hashes (pinned in the lock)

| Item | Files |
|---|---|
| B2 Payload reader | `TP/payload_read.py`: level stability, max_stable, limiting factor, actuator metrics; pure numpy on the raw npz; loads `SH/sh_kin.py` by path with `sys.dont_write_bytecode`. |
| B6 Stand on a new walker | `SN/sn_dagger_spec.py`: one override `SN_WALKER_SPEC=<json>` for `S1_CKPT`, `S1_SHA256`, `S1_AGENT_YAML`, `REFS_ENV_STAMP`, `CANDIDATES`, `OVERLAY_CHAIN`, `S1_OVERLAY` (:57-104); unset → today. `SH/sh_dagger_spec.py` inherits it (:87-95). `SH/sh_dagger.py` CLI gains `--warm-start/--warm-sha` (already accepted by `sh_dagger_core.py:180-181`; the estimator-equality guard :205-209 is kept). Launchers take `CK`/`ST` from `SN_WALKER_CK`/`SN_WALKER_ST`, plus `P2_REANCHOR=1` (the guard enforces the pin). Readers: `sn_calib.py:104` NAMES and `sh_read.py:242`, :1420-1443 via the spec; `--s1-sha` on the table tests. Unit tests `test_sn_dagger.py:116`, `test_sh_dagger_loop.py:589`, `test_sh_core.py:59`: defaults kept, override cases added. |
| B7 Pipeline | `TP/pipeline.py` (gen from `BC/scripts/bc_make_candidates.py:37-124`; queue from `bc_train_queue.sh`; after-stage from `bc_after_stage.py:45-147`; rows = `bc_eval_row.sh` unchanged), `thresholds.py`, `gate_read.py` (world stats via RP `cat_summary`, body via `bc_read.py`, FAST via `side22_classify.py`, field via `p2_read.py`), `live_watch.py` (`p2_live.py:137-148` rules + `PPO/scripts/resume_read.py:39-41`), `distill.sh` (from `SH/sh_dagger.sh`: busy check, DDC guard, `systemd-run --scope MemoryMax=48G`, `timeout --signal=TERM`, `env -u` every `KRABBY_*`), `select_student.py`, `replay_read.py`, `report.py`. |
| B1 Driver side (tools_sha only) | RC `OVERLAY_DUMP_CHECKS` entries + parsers for the payload knobs (RC:2771, ~:3203; precedent e47d9bef); knob-whitelist re-export (RC:3223-3256). No step-keyed ramp. |
| B8 Rows | `PL/rows/scenarios_pl_body.yaml` (v3, `command_route: term`, envelope block copied verbatim): `dirsweep__omni_body`, 16 translate categories `d000 … d338` at 22.5°, \|v\| 0.22, wz 0, `metric_class: translate`, 12-s holds, 100 episodes per wave, `first_face`, Walk-Play, through `body_hold`. Payload level env files are generated. |
| B9 Yaw-rate interface | No new training. `body_hold` (H:518, :685-720; windup 0.6) is the documented robot-frame (vx, vy, ωz) interface of record for the grid and the sweep, for teacher and student. Recorded limit: it refuses `eval_slot_latch` / `stand_hold` (H:699-705), so walker + body_hold + stand net is never evaluated together. The deploy adapter is M15. |

---

## 5. Runs (order)

| Phase | What | Gate |
|---|---|---|
| **0** (CPU) | thresholds v0 + self-checks on recorded stamps; camera geometry per mount (rows that hit ground; ground band in body-x) | STOP-0: user approves the plan, decisions, thresholds v0 and budget |
| **A** (HEAD, no new code; ~1.3 GPU-h) | **A2:** `PL/preflight` 20-it smokes: s1 from scratch; c4/c5/c6/c6x/p2a resumed from the bundled lineage heads (`OMNI/policy/*/model_*.pt`). **P-config:** trainer-written `params/{env,agent}.yaml` against the X43/X57/X60/X08 run dirs, allow-list = new default-off keys, paths, timestamps. **P-early:** s1 tfevents against X43's first 20 its. **A3:** S1 payload feasibility via `KRABBY_DR_MASS=m:m` (no code): zero-action settle + suite es1 at 0/45/90/135/180 kg | STOP-A |
| **B** | M1, E1, TP, rows, B6; CPU tests (§9) | — |
| **C** (build tree; ~18-20 GPU-h) | Ordered cheapest-disqualifier first. **C1** inertness bridges. **C2** launcher bridges per row family on S1 s18. **C3** payload: partial-reset readback (≥ 2 envs), 100-it timing, deck-CoM settle per level, the **pilot** (3000 its from S1, reads at +1000/2000/3000 → N_p), and B0b, the S1 0-270 curve (the payload "before"). **C4** student: check 4b per camera (axis error ≤ 1°, ground band covers body-x 0.5-1.2 m); S1 `--policy-role teacher` on Student-Play against Walk-Play within bands; 200-it timing; two 50-it same-seed runs; **depth vs zeroed-depth pilot**, 1000 its each from S1. **C5** sweep anchors on S1 s16/s17/s18 es1-2. **C6** SN/SH 100-it smokes beside X10 (s16 p2a) + identity bridge. **C7** an end-to-end dry pass of `pipeline.py` on smoke chains, with a forced FAIL and a resume | STOP-C; then freeze `thresholds.json` v1 + `pipeline.yaml`, commit, tag the pin |
| **D** | R18, the full pipeline, then X (the stand-tooling replay) | STOP-R, HALTs |
| **E** | F19 (decide-record-notify hand-on after R18 is clean) | HALTs |
| **F** | `pipeline.py report`, keep-set, local tag | STOP-END |

### R18 parity verdict (pre-registered)

| Check | Pass |
|---|---|
| P-config (A2, repeated after C1) | Identical up to the allow-list |
| P-early (A2; C1 against A2) | ≥ 112/116 tags bitwise; only `Loss/priv_reg` (≤ 2e-6), `hist_latent` (≤ 1 ULP) and `Perf/*` may differ (SMOKES a1/a2) |
| P-live (R18 s1 at +200) | Same tag rule against X43, else HALT: about 5 min in, a cheap drift catch |
| P-onset | Reported: first divergent iteration per tag, expected ≈ 1905 (priv_reg enters at counter 2000; `ppo_with_extractor.py:158-167`); per-module tensor max\|Δ\| every 100 its |
| P-gates | R18 passes every gate. The low side is the gate; a high-side excursion beyond max + m is WATCH |
| P-head | R18 p2a against S1, per statistic: Δ and R reported |
| X (stand replay) | SN/SH beside the **bundled S1**, DS 17, against SN_Amask_s17 (769cf253…) and SH_T17 (99c377a1…): file sha (bonus; DAgger determinism unverified), loss curves, K1-K6 within the recorded value ± 2SE |

**Verdict:**
- **REPRODUCED-BITWISE:** the p2a sha is equal (not expected).
- **REPRODUCED:** P-config + P-early + P-live + P-gates.
- **NOT REPRODUCED:** STOP-R.

**F19.** Same spec and pin; only the seeds change. PROMOTE 0.7:0.4 stays pinned. No best-of-N selection.

**Deliverable rule (D13):** F19 if every gate passed, else R18 if every gate passed, else neither. There is no best-of-2 by metrics.

**Wall-clock and hardware** come from `state.json` `wall_h` and the logs: RTX 5080 16 GB, Ryzen 9 9950X3D, 59 GB RAM, host fractal-ubuntu, plus the external stack from the lock.

---

## 6. Baseline-vs-final report

Output: `PL/REPORT_baseline_vs_final.md` + `.json`, generated by `TP/report.py` from records only.

| Column | Content |
|---|---|
| H (history) | v1 numbers as committed (`FWD/experiments/eval/baselines/v1/`, stand-ins 21100/29098/19999; tripod 0.052-0.080, 1 N tippy 0.28-0.39). Flagged as not re-runnable. 6300/9800 do not load: critic 512×1007 vs 1149, `std` vs `std_raw`, pre-cam plant, 2026-05-23 USD, and the a0adfb21 code has no harness. |
| F (previous-generation student, the measured "before") | Forward 3a `FWD/policy/3a_student/model_24995.pt` at the pin, on its own task (`Isaac-Crab-Hex-Student-v0` + `KRABBY_STUDENT_MDP=1`), forward rows. Lateral, yaw, grid and sweep cells are N/A (no command channel), not 0. Mass-only payload curve via `KRABBY_DR_MASS=m:m`, labelled as such. ±0.04 process-state noted. |
| T0 (teacher before) | S1 + SH_T17: existing stamps + sweep anchors + the B0b curve |
| T1 / St (final) | T\* and the selected student of the 4c run; the other run in an appendix with the R18 parity verdict |

**Metrics:**
- **Task 0:**
  - tripod and duty;
  - 50 N rigidity: `support_feet`, `steps_per_m` (stride ≈ 6/steps_per_m), `taps_per_m`, `swing_apex`, `step_rate_hz`;
  - 1 N air time and tippy-tap, reported with the misread caveat (M:1271-1275);
  - tracking per axis: ratio, `direction_error`, `cross_track`, vy/wz RMS;
  - roll/pitch RMS;
  - action rate and reversals;
  - torque and speed, plus the reader's cap share.
- **Task 2:** grid (9 gated + 11 reported cells), turnlevels, body suite, the 16-bin sweep with discontinuity, the field rows, side22.
- **Task 3:** 0-270 curves for S1, T\* and the student; max stable; limiting factor; rod-force exceedance (report-only).
- **Reconciliation prose:**
  - robot 230.06 kg, not 83 kg, so 90/180/270 kg = 0.39 / 0.78 / 1.17× robot mass;
  - SLA08 2000 N @ 28 mm/s and YH8-523D 500 N @ 33 mm/s, not 500 N / 80 mm/s and 400 N / 60 mm/s;
  - the ceiling is a sim-plant number: no rod-force limit, and the hip rod is about 3.4 kN static at 0 kg.
- **Stand:** K1-K6 per run beside SH_T17.
- **Provenance:** wall-clock and GPU-h per stage, the pins, every gate verdict.

---

## 7. STOP points

All other pauses are decide-record-notify.

| ID | Trigger | User decides |
|---|---|---|
| STOP-0 | Before any GPU work | Approve the plan, decisions §8, thresholds v0, budget |
| STOP-A | P-config diff outside the allow-list; P-early < 112/116; A3: zero-action settle tips at ≤ 90 kg **and** S1 90-kg suite survival < 0.5 | Find the code cause among the 14 commits since f7166d55, or replay at the old tree in a worktree; a plant change (PD/caps) or a lower level grid |
| STOP-C | A bridge not bitwise; payload readback wrong; 4b fails; student > 3 h/1k; teacher-on-student-env outside bands; depth − zeroed field_hard (es1-2) < max(0.05, 2·SE_diff); stand smoke fails; dry pass fails | Fix and re-prove; camera layout (D1); re-anchor a row family |
| STOP-R | NOT REPRODUCED (incl. P-live) | Diagnose, re-pin and re-run, or accept statistical-only parity |
| HALT | Any HALT line or a p3a 3d fail, either run; INVALID twice; no student candidate; a stand K/G fail not explained by its control | §3.7 options |
| STOP-END | After F19 | Deliverable (D13), clip veto, bake into `policy/`, tag push |

---

## 8. Open decisions (only the user)

| # | Decision | Recommendation |
|---|---|---|
| D1 | Camera layout | **Front + left + right** (matches the HAL; no rear). Front about 55° down, 2 m range; sides about 55-60°, 1.5 m. Final pitch from the Phase-0 geometry and 4b. Back field travel is WATCH for the student. |
| D2 | Velocity cols 13-14 in the student | **Keep privileged** (forward precedent; the frozen estimator's x/y signal depends on them, §0.1 #3). Deploy needs a velocity estimate (ZED VIO). A masked variant is an M15 diagnostic. |
| D3 | Tile flag cols 11-12 | Constant "field" pair in training, eval and deploy |
| D4 | Depth `num_prop` | **89**: restores the last-action tail and the current-sense proxy (the only deployable load channel) that the teacher's scan encoder saw. It is one parameter of the shared step. |
| D5 | Payload geometry and distribution | Reset U[0, L] with 25 % zero; CoM z +0.309 m (a 0.30 m deck load), xy ±0.10 m; deck-footprint box inertia; startup DR preserved (lazy capture) |
| D6 | Level stability lines (§3.3) | As listed (a priori) |
| D7 | Semantics above 90 kg | 90 is HARD (3d); 180/270 are ceiling stages; stop raising when a stage misses its own top; T\* = argmax max_stable |
| D8 | Force-side realism (grant 3a/3b) | **Report-only.** Keep the 400/1000 N·m holding-level caps (lead screws cannot be back-driven; self-lock 2800/4500 N). Record it as an open grant gap. |
| D9 | Stand lines | Method of record re-anchored on REFS(T\*); **K4 ≤ 3/800 per env seed**; lazy R_h/R/HOLD controls; 0 kg only |
| D10 | R18 scope | **Full pipeline + X** (+5.2 GPU-h). A walker-only R18 saves about 40 GPU-h but leaves the new stages with one sample. |
| D11 | Fresh seed | **19** (declared but never run) |
| D12 | Margin rule and bound | min − max(R, range, G); run-level false-HALT ≤ 0.20 |
| D13 | Deliverable | F19 if clean, else R18 if clean |
| D14 | Isolation | Code freeze + guard + resumable queue for about 6 days; other sessions stay off CODE/EVAL/TP/PL. A worktree would need the hard-coded R paths changed (`launch_candidates.sh:46`, every launcher). |
| D15 | Yaw-rate interface | `body_hold` of record; deploy adapter is M15 |
| D16 | Sweep | 16 × 22.5° body rows at 0.22 m/s + opt-in `metric_class`; a continuous hold is deferred |
| D17 | PROMOTE | Pinned 0.7:0.4 |
| D18 | Keep-set contents | The 4e set (student, T\*, SN, SH); stage heads by sha only |
| D19 | Untracked `resolution.74f93b91dc8b.json` (another session's) | User commits it before the pin; otherwise `thresholds.py` recomputes it with `T/resolution.py` into PL |
| D20 | Tag names, bake, push | The user's |

---

## 9. Verification

### CPU (pytest; new `tests/unit/test_task4_pipeline.py`)

- **`gen --check`:**
  - byte equality;
  - every overlay's sha equals its PPO/P2 source block;
  - key counts 14/16/16/16/19/27;
  - no stage named "2a"/"2b";
  - categories and `absolute_units`;
  - `validate_candidates(strict=True)` + the clock-chain rules (RC:3979-4082).
- **Thresholds:** `thresholds.py --self-check` (seeds pass, known-bads fail, false-HALT bound, provenance shas).
- **`gate_read` golden tests:**
  - PASS on S1 s16/s17/s18 at p2a, and on the lineage heads of all seeds at each stage;
  - HALT on C0 X39/X41/X44 for s1 side22;
  - the field lines fail on ANCW;
  - INVALID on a stamp with `CLOCK_HIDE` flipped.
- **Payload:**
  - parsers + `OVERLAY_DUMP_CHECKS` in both directions;
  - composite mass, CoM and parallel-axis math against hand values;
  - lazy capture after startup DR;
  - partial-reset indexing;
  - the private generator leaves the global stream untouched;
  - knobs unset → the 13 `BC/dump_targets.json` cfg dumps are byte-identical.
- **Student:**
  - the shared step never writes cols 6:8;
  - cols 11-12 + their history copies get the constant pair; 13-14 untouched;
  - the estimator writes into 207-215 from the student copy;
  - the GRU reset touches only done rows;
  - K-camera shapes; the exporter dummy;
  - forward `learn_vision` / harness / play / play_stream_video with default flags are call-for-call identical on a fake env;
  - an integration cfg-parity test: student env == T\* env except cameras and depth groups (`tests/integration/test_crab_hex_phase_configs.py:72-76` model);
  - `body_hold` composes with the student step.
- **E1:** `metric_class` parse/apply; rows without it are byte-identical.
- **Stand:** `SN_WALKER_SPEC` unset → constants identical and existing tests pass; set → replaced; SH inherits.
- **Layout and existing suites:**
  - `test_experiment_layout.py`, `bundle_experiment.py --check`;
  - `test_crab_hex_omni_phases.py` (STUDENT_PHASES still `()`; update the "no student" comment only);
  - `test_policy_of_record_omni.py` and the SN/SH/forward suites pass at the pin.

### GPU inertness (C1, bitwise via `BC/scripts/bc_bitwise.py`)

- S1 s18 `suite__omni` es1 + `field_hard__omni` es1 against G0/REF;
- `suite__omni_body` es1 against ANC_S1_s18;
- X43 s1, X60 c6x and X10 p2a suites against their lineage stamps;
- forward 3a slow row, same manifest shape;
- payload armed at `0:0` / ZERO_FRAC 1 against no event: suite es1, plus a 20-it p2a-resume training smoke;
- the post-build stage smokes against the A2 smokes;
- the 13 cfg dumps identical;
- C2: every row family on S1 s18 through `bc_eval_row.sh`, bitwise against its recorded stamp. A family that fails is re-anchored on S1 s16/s17/s18 before v1.

### Hygiene (report to the user; change nothing)

- the untracked resolution file (D19);
- `SH/scripts/__pycache__/sh_kin.cpython-311.pyc` from another session (gitignored);
- `P2/state.json` still shows X01 as running. It is irrelevant, since the runs use fresh dirs.

---

## 10. GPU budget

M = measured, E = estimate.

| Block | GPU-h | Source |
|---|---|---|
| **Preflight A** | 1.3 | 6 smokes about 0.4; A3 5 × (settle + suite 7 min) about 0.9 (E) |
| **Preflight C** | 17-19 | C1 1.7; C2 1.2; C3 6.1 (pilot 3000 its × 1.49 s/it M + 3 reads; B0b 7 × 0.35); C4 6-8 (2 × 1000-it pilots at 2-3 h/1k E; forward K=1 measured 2.0 h/1k); C5 1.2; C6 0.5; C7 0.5 |
| *Per run:* walker training | 7.7 | M (`state.json` `wall_h`) |
| Walker gate rows | 4.2 | Suite 417 s, field rows 206-219 s, side22 216 s, grid 829 s, turnlevels 419 s (M, stamp logs); body block 1.4 (M); sweep about 0.4 (E) |
| Payload training | 2.1-3.1 | N_p ≈ 2500 × 1.49 s/it (M), 2-3 stages |
| Payload gates + T\* block + curve | 6.6-8.1 | Reduced 0-kg 0.45 + about 3 levels × 0.35 per stage; T\* block 2.6; curve top-up about 1.0 (E) |
| Student training 5000 its | 10-15 | Forward 9.87 h at K=1 (M); K=3 unmeasured; capped by STOP-C at 3 h/1k |
| Student screen | 8.2-11.6 | Prescreen 5 × 0.3; full block 3.4-6.8 (×1.3 camera factor, E); curve 3.2; clip 0.1 |
| Stand | 6.9 (+3 if controls trigger) | SN 1.71 (M) + REFS 0.8 + SN rows 0.8 + bridge 0.1 + SH 2.35 (M) + K rows 1.1 (M) |
| **Per run** | **≈ 46-60** | |
| X stand replay (R18) | 5.2 | SN 1.7 + SH 2.35 + K reads 1.1 |
| Report extras (forward 3a rows + mass curve) | 2 | E |
| Contingency (INVALID reruns, lazy controls, one partial re-run) | 8 | E |
| **Total** | **≈ 125-155 GPU-h ≈ 5.2-6.5 days serial** | Calendar time is longer, by the dev work (8-11 days) and the decision waits |

---

## 11. Disagreements resolved

1. **Sweep metrics:** the opt-in `metric_class` in E1 (reuse). Risk and repro were wrong that no harness change is needed (§0.1 #1).
2. **Actuator, h_meas and `swing_s` metrics:** a reader over the raw npz (reuse), not new harness keys (risk, repro), which would break the bitwise bridges. `swing_s` is dropped; the user's decision makes the 50 N rigidity set the gait gate.
3. **Velocity cols 13-14:** kept privileged (risk) against masking (reuse, repro), because of the estimator-copy fact. This is D2 for the user.
4. **Cols 11-12:** the constant "field" pair (risk, repro), not zero (reuse).
5. **STUDENT_PHASES:** stays `()` (risk); there is no preset and no test change.
6. **Camera event:** none (risk). Reuse's composed event and repro's unarmed one are both unneeded.
7. **Depth `num_prop`:** 89 (repro) over 75 (risk), as one parameter of the shared step.
8. **Image term:** the batched omni term built up front (risk), not built only after the timing smoke (reuse). That avoids a second code_sha move.
9. **Fresh seed:** 19 (reuse, repro). Risk's 21 rested on references that never ran.
10. **Threshold rule:** reuse's min − max(R, range, G) with a run-level false-HALT bound of ≤ 0.20, replacing reuse's per-stage 0.10, which compounds to 30-50 % per run. Repro's inconsistent numbers and its vacuous downgrade are dropped. From repro: its capability-only HALT and its known-bad tagging.
11. **Replay criterion:** P-config uses trainer-written params (reuse), run at HEAD before any build (risk). Repro's live +200 check is added. The low side of the band is the gate and the high side is WATCH (repro). Risk's "PARTIAL → STOP" would fire by chance at about 30 %.
12. **Stand replay:** beside the bundled S1, DS 17 (reuse). It isolates the stand tooling from walker drift. Repro's add-on beside R18's walker can only compare statistically.
13. **Stand controls:** lazy (risk). REFS(T\*) is always run; R_h/R/HOLD run only on a failure, then re-anchor or HALT. The stand lines follow the recorded formulas otherwise.
14. **Stand launcher guard:** `P2_REANCHOR=1` + the pipeline guard (reuse), not a new `P2_GUARD_REF` sidecar.
15. **Payload staging:** train the next stage only if the previous reached its own top (repro). T\* = argmax max_stable, ties to the earlier stage (risk, repro), not the latest passing (reuse). Startup DR is preserved by lazy capture rather than folded into the payload draw (a new design).
16. **Payload xy:** ±0.10 m (risk, repro) over ±0.05 m (reuse). The box is the measured deck footprint (risk).
17. **Branch order after T\*:** student first (risk); a HALT in one branch does not block the other (repro).
18. **Student full block:** run in prescreen order until one candidate passes, not on the top two (reuse, repro). This halves the cost.
19. **Deliverable location:** a campaign keep-set via the existing `bundle_experiment.py` (reuse). `bundle_policy.py` changes wait for the user's bake.
20. **Per-run campaign dirs and one entry point** (repro), on reuse's components; `play_stream_video.py` is added to the shared-step users.
21. **Budget:** 125-155 GPU-h, above all three plans' midpoints. The K=3 student cost is unmeasured and dominates; STOP-C caps it.
