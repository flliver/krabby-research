# Window 0 for t3 (plan step 9)

Outputs:
- smoke: `parkour/outputs/omni_plan/stroke_cat/t3_dose_smoke/` (worktree);
- dumps: `parkour/outputs/omni_plan/stroke_cat/window0_t3/` (main checkout).

## Dose smoke — STOP on the raw statistic, resolved by the user (2026-09-28 16:26–16:40)

**What ran.** `stroke_cat_t3_dose_smoke.py`: 300 its of S_cat_f's composed `t3f_scat` env at the replay-scale t3 table.
- The replay-scale table: `STROKE_W` 0.13622, side ×4.150, diag ×1.319, turn ×1.745, move_turn ×1.
- Clock held, with play keys ring and FACING_OBS_CLIP 0.6, plus the parity vars.
- It forked `policy/1_t3/model_2999.pt` (sha checked).
- Run dir: worktree `parkour/logs/rsl_rl/crab_hex_omni_flat_walk/2026-09-28_16-26-51_t3_dose_smokeS_cat_f`.

**Parity: PASS.** Max |diff|: pay_sum 4.2e-7, b 2.4e-7, pay 1.2e-7, o 4.5e-8. 0 category mismatches.

**r_c** = logged ÷ replay nominal on E3 (X08:t3's suite) in X08:t3's clock-held economy, last 200 its:

| Category | Log/s | Replay/s | r (tool: mean of per-iteration rates) | r, dwell-weighted | r, spikes (> 10 × median) excluded |
|---|---|---|---|---|---|
| fwd | 0.0866 | 0.0714 | **1.21** | 1.25 | 1.21 |
| side | 0.3091 | 0.0670 | **4.61** → STOP | **1.45** | 1.12 |
| back | 0.0647 | 0.0593 | 1.09 | 1.22 | 1.09 |
| turn | 0.0192 | 0.0771 | 0.25 (replay-set) | — | — |

**Why side's 4.61 is not the dose.** Two iterations carry it:
- **3147:** side 32.5/s on a side dwell of **0.008 s**. The per-category rate divides by the category's dwell, so this is a
  division artifact.
- **3146:** side 13.0/s on a 3.75-s dwell.

Every 50-it window's median side rate is 0.067–0.088.

The same check on the W smoke: side mean-of-rates r 1.66, dwell-weighted 1.76, spike-excluded 1.40. Its spikes had
long dwells (11.7 s and 5.8 s), so the W dose stands.

**User decision (2026-09-28): use the dwell-weighted side r 1.45** (income per total side dwell second; the
conservative dose). fwd uses the plan's statistic, 1.21, which is in band with no spikes.

**Final arithmetic:**
- `STROKE_W` = 0.13622 / 1.2126 = **0.112339**.
- m_side = (0.56532 / 1.4540) / `STROKE_W` = **3.460881**.
- m_diag = 0.17964 / `STROKE_W` = **1.599056** (replay-set; not in t3's pool).
- m_turn = 0.23770 / `STROKE_W` = **2.115884** (replay-set).
- m_back = m_move_turn = 1 (move_turn is not in t3's pool).
- **Effective:** fwd = back = move_turn 0.1123, side 0.3888, diag 0.1796, turn 0.2377.
- Every effective weight asserts within 1 %. Written into `candidates.yaml` (`t3f_scat`); `--dry-run` is clean.

JSON: `parkour/outputs/omni_plan/stroke_cat/t3_dose_smoke/{t3_dose_smoke_r.json, final_table.json}` (worktree).

## Identity, knob-on dump and knob smoke — PASS (2026-09-28 16:38–16:40)

Script: `~/.claude/plans/per-category-stroke-weights.evidence/window0_t3.sh`.

- **B_f-config identity.** `t3f` plus B_f's eps knobs dumped from the merge base `a2f4f0eb` and from the branch:
  **IDENTICAL** once the tree roots are normalised. The knob-off omni/forward/sideways suites were already proven
  identical in the W window 0.
- **Knob-on (`t3f_scat` final table) against the B_f config.** On both tasks only `reward_stride_length.weight`
  (0.001 → 0.112339) and the added `params.cat_scale` differ. The agent cfg is unchanged.
- **Driver knob smoke** (`--smoke-stage 1a:20`, scratch copy of the yaml): both `overlay_checks` entries are **True**
  against the real `params/env.yaml`.
- **The decisive reuse control** runs at launch: the reference run's own `params/env.yaml` against B_f's.
- **Hazard.** The reference run's tag will be `seed3_X01_t3_A15pB`, the same as B_f s3's in stroke_formation. Never stop
  it mid-training.
