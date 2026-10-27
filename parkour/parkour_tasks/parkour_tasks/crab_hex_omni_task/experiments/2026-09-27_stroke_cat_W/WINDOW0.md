# Window 0 (plan step 6), W setting

Outputs:
- dumps: `parkour/outputs/omni_plan/stroke_cat/window0/` (main checkout);
- smoke: `parkour/outputs/omni_plan/stroke_cat/W_dose_smoke/` (worktree).

Script: `~/.claude/plans/per-category-stroke-weights.evidence/window0_dumps_W.sh`.

## (i) Config identity — PASS (2026-09-28 07:51–07:58)

- **Trees compared.** The merge base `a2f4f0eb` (a detached worktree, `.claude/worktrees/stroke-cat-mb`) against the
  branch worktree (`f5c91f36`, which carries the new knob).
- **What was dumped.**
  - the three config-identity suites: forward 1a–3a, sideways 1a–3a, omni 1a/1b/1c/2a/2b/2c;
  - the `c6y` chain knob-off;
  - the `c6y` chain with B's eps knobs (`KRABBY_OMNI_STROKE_W` / `CENTRING_W` 0.001 plus the split flag). That is the
    exact config the reused B pair trained under.

  That is 20 files per tree.
- **Result.** With both roots normalised to `<ROOT>` and `_environ` dropped, **0 of 20 files differ**.
- **Suites.** 210 passed, 5 failed in BOTH trees. The same 5 legacy-window tests
  (`test_phase_preset_reproduces_recorded_window[1a/2a/2b/2c]`, `test_phase_preset_equals_legacy_env_stack`) already
  fail at the old merge base.
- **Knob-on dump** (S_cat at the replay-scale table), against the B config, on both `Isaac-Crab-Hex-Omni-Walk-v0` and
  `-Play-v0`:
  - `rewards.reward_stride_length.weight` 0.001 → 0.08646779396708197;
  - `rewards.reward_stride_length.params.cat_scale` added:
    `{back: 1.0, diag: 1.8426, fwd: 1.0, move_turn: 1.0, side: 7.1092, turn: 1.7728}`.

  Nothing else moved, and the agent (runner) cfg did not move.
- **Reuse control, env half.** S_cat's composed stage env (driver dry run / `--print-env`) differs from old B s3's
  logged stage env (stroke_posture `orchestrator.log`) only in `KRABBY_OMNI_STROKE_W` (0.001 → w_fwd) and the added
  `KRABBY_OMNI_STROKE_CAT_W`.
  - The smoke additionally carries the play keys (`KRABBY_FLAT_TERRAIN_GEOM=ring`, `KRABBY_OMNI_FACING_OBS_CLIP=0.6`)
    and the parity vars, as the 2026-09-26 window-0 smoke did.
  - **The decisive dump half is checked on the S_cat s3 reference run's own `params/env.yaml`**, against B s3's, as
    soon as it is written (step 7).

## (ii) Dose smoke — PASS, doses rescaled (2026-09-28 07:59–08:12)

- **What ran.** `stroke_cat_dose_smoke.py`: 300 its of S_cat's composed `c6y_scat` env at the replay-scale table, plus
  the play keys and the parity vars. It forked from W's baked copy (sha checked against the manifest).
  - Run dir: worktree `parkour/logs/rsl_rl/crab_hex_omni_flat_walk/2026-09-28_07-59-09_W_dose_smokeS_cat`
    (model_12294.pt).
- **Wrapper → replay parity: PASS** (300 steps × 16 envs). Max |diff|: pay_sum 6.1e-7, b 3.0e-7, pay 1.2e-7, o 4.5e-8.
  - The reward was checked as `cat_scale[cat] · Σpay / dt` per category.
  - **0 category mismatches** against the dump's own stand / turning / lat_share columns (4,800 steps).
- **The train-log weight guard passed:** the dumped `params/env.yaml` weight and `cat_scale` equal the table the r was
  priced at.
- **r_c** = logged nominal ÷ replay nominal on W's suite stamp, in W's clock-free economy, last 200 its:

  | Category | Log/s | Replay/s | r_c | Role |
  |---|---|---|---|---|
  | fwd | 0.0599 | 0.0875 | **0.685** | sets `STROKE_W` |
  | side | 0.1153 | 0.0694 | **1.663** | rescaled |
  | back | 0.0665 | 0.0705 | 0.944 | tied to fwd |
  | diag | 0.0684 | 0.0713 | **0.960** | rescaled |
  | turn | 0.0178 | 0.0872 | 0.205 | replay-set (training turn holds ungate) |
  | move_turn | 0.0197 | 0.0216 | 0.909 | m = 1 (decision 7) |

  Every dose-setting r is in [0.5, 2], so there is no STOP.
- **Final arithmetic** (CHARTER):
  - `STROKE_W` = 0.08646779 / 0.6849 = **0.126240**.
  - m_side = (0.61472 / 1.6629) / `STROKE_W` = **2.928327**.
  - m_diag = (0.15932 / 0.9600) / `STROKE_W` = **1.314741**.
  - m_back = m_move_turn = 1.
  - m_turn = 0.15329 / `STROKE_W` = **1.214263**.
  - **Effective:** fwd = back = move_turn 0.1262, side 0.3697 (2.93 × fwd), diag 0.1660, turn 0.1533.
  - Every effective weight asserts within 1 % of its target. Written into `candidates.yaml` (`c6y_scat`); `--dry-run`
    is clean.
- **Note.** The side multiplier fell from the replay's 7.1 to 2.9, because side delivered 1.66 × its prediction in
  training. That is the same side r (1.39–1.50) the single-weight screens showed.
- **JSON:** `parkour/outputs/omni_plan/stroke_cat/W_dose_smoke/{W_dose_smoke_r.json, final_table.json}` (worktree).

## (iii) Knob smoke of the driver's overlay audit — PASS (2026-09-28 08:08–08:10)

- **What ran.** `run_candidates.py --smoke-stage 1a:20` with both final `--smoke-overlay` keys, on a scratch copy of this
  campaign's yaml (`parkour/outputs/omni_plan/stroke_cat/knob_smoke_campaign/`, worktree), so that its drill records
  stay out of this campaign.
- **Result.** The recorded `overlay_checks` against the real `params/env.yaml`:
  - `KRABBY_OMNI_STROKE_CAT_W=fwd:1.000000,back:1.000000,side:2.928327,diag:1.314741,turn:1.214263,move_turn:1.000000`:
    **True**;
  - `KRABBY_OMNI_STROKE_W=0.126240`: **True**.

  There was no `config_mismatch` halt. The reference run's post-training overlay audit is exercised.

## Launch checks (plan step 7), S_cat s3

- **Launch.** Landed as `feat/crab-hex-omni-task` → `84ff2588` (fast-forward, local). Launched 08:11 from the main
  checkout as unit `cands-2026-09-27_stroke_cat_W-1790597477` (`--reference-chain S_cat --seed 3`).
  - Run dir: `parkour/logs/rsl_rl/crab_hex_omni_flat_walk/2026-09-28_08-11-28_seed3_X01_c6y_A15pB`.
- **Reuse control, dump half: PASS.** The run's own `params/env.yaml` against old B s3's
  (`2026-09-26_03-01-22_seed3_X01_c6y_A15pB`) differs in 7 leaves, all in `reward_stride_length`:
  - `weight` 0.001 → 0.12624;
  - `params.cat_scale` added: back 1, diag 1.314741, fwd 1, move_turn 1, side 2.928327, turn 1.214263.

  `agent.yaml` is identical.
- **Hazard.** The reference run's tag is `seed3_X01_c6y_A15pB`, the same as old B s3's run in the stroke_posture
  campaign. `prepare_resume` finds run dirs by tag across the experiment dir, so a stop before this run's first
  checkpoint could resume from B's checkpoint. Do not stop it mid-training.
- **Iteration-~500 overdose stop: ok.** Ratio of means over its 12446–12546 of logged stroke ÷ logged
  track_lin_vel_xy_exp:

  | fwd | side | back | diag |
  |---|---|---|---|
  | 12.4 % | 8.6 % | 10.1 % | 9.4 % |

  All are ≤ 15 %. Side's delivered share is about 3 × the single-weight screens' 2.2–3.5 %.
