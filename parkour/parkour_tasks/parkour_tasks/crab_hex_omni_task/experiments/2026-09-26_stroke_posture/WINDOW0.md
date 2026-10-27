# Window 0 (plan step 12)

**Order changed from the charter.** Window 0 ran BEFORE SA2 (landing), from the branch worktree, so nothing reaches
the shared `feat/crab-hex-omni-task` unless the knob-off configs are identical and the dose smoke passes. Outputs:
`parkour/outputs/omni_plan/stroke_posture/window0/` in the main checkout (the dumps) and in the worktree (the smoke).

## (i) Config identity — PASS (2026-09-26 02:41–02:47)

- **What ran.** For both the merge base `fb203f7d` (a detached worktree) and the branch worktree:
  - the three config-identity suites, which dump forward 1a/2a/2b/2c/2c_noplant/3a, sideways 1a–3a and omni
    1a/1b/1c/2a/2b/2c;
  - the screen's chain `c6y` knob-off dump.

  That is 19 fixtures per tree.
- **Byte comparison.** Every file differs, but only by each tree's absolute root (the dumps embed the checkout path
  in the USD / RSI-bank / evidence paths). With both roots normalised to `<ROOT>` and `_environ` dropped, **0 of 19
  files differ**.
- **Suites.** 210 passed and 5 failed in BOTH trees, the same 5:
  - `test_phase_preset_equals_legacy_env_stack`;
  - `test_phase_preset_reproduces_recorded_window[1a/2a/2b/2c]`.

  They were already failing at the merge base, and the branch does not cause them.
- **Knob-on dump** (arm C's doses plus the split flag, branch tree), on both `Isaac-Crab-Hex-Omni-Walk-v0` and
  `-Play-v0`, differs from knob-off in exactly the expected places:

  | Key | Knob-off → knob-on |
  |---|---|
  | `rewards.reward_stride_length.func` | `RewardStrideLength` → `RewardFootStroke` |
  | its `params` | six Footpads, `preserve_order` True; the old term's `min_cmd_norm` / `min_phase_duration` / `power` gone |
  | its `weight` | 0.5 → 0.0865 |
  | `rewards.reward_body_centring` | added, weight 0.155 |
  | `reward_category_split` | → True |

  The runner (agent) cfg did not move.

## (ii) Dose smoke — PASS, doses rescaled (2026-09-26 02:48–02:56)

- **What ran.** 300 iterations of arm C (both doses at their SA0 values, 0.0865 / 0.155), outside the ladder via
  window0_dose_smoke.py, from W's baked copy. The env was:
  - arm C's composed env (c6 overlay + K + doses + split);
  - the play block's policy-input keys;
  - the parity recorder.

  Run dir: worktree parkour/logs/rsl_rl/crab_hex_omni_flat_walk/2026-09-26_02-48-59_window0_doseC
  (model_12294.pt).
- **The term is alive past 2 s in real Isaac.** Training telemetry (Episode_RewardCat) logs stroke income on
  every moving category.
- **Wrapper → replay parity: PASS** (300 steps × 16 envs). Max |diff|: pay_sum 2.2e-7, b 3.0e-7, pay 2.2e-7,
  o 4.5e-8, all under 1e-5. R = pay_sum / dt reaches 1.1e-5, which is float noise amplified by 1/dt.
- **Dose ratios.** r = delivered income (Episode_RewardCat, post-clip, last 200 its) ÷ replay-predicted
  income at the same doses on W's suite raws.

  | Category | Stroke log/s | Stroke replay/s | r_S | Centring log/s | Centring replay/s | r_P |
  |---|---|---|---|---|---|---|
  | fwd | 0.0651 | 0.0875 | **0.74** | 0.0047 | 0 | — |
  | side | 0.0136 | 0.0098 | 1.39 | 0.0560 | 0.0578 | **0.97** |
  | back | 0.0561 | 0.0705 | 0.80 | 0.0048 | 0 | — |
  | diag | 0.0311 | 0.0387 | 0.80 | 0.0493 | 0.0574 | 0.86 |
  | turn | 0.0085 | 0.0492 | 0.17 | 0 | 0 | — |
  | move_turn | 0.0210 | 0.0217 | 0.97 | 0 | 0 | — |

  Absorbed share in training: fwd 4.3 %, side 6.2 %, move_turn 8.6 % (≤ 15 %).
- **SA0 dose rule** (the pre-approved rescale):
  - r_S = 0.74 is on the fwd row, the one that sets the dose, and in [0.5, 2]; side's r_S = 1.39 is in band too.
    **w_S → 0.0865 / 0.74 = 0.117** (1.35×, under the 2× limit).
  - r_P = 0.97 on side: **w_P → 0.155 / 0.97 = 0.160**. Diag's centring share becomes about 7.3 % (≤ 10 %).
  - Not an SA3 stop.
- **Notes.**
  - Turn's r of 0.17 is a command-distribution effect, not a defect. Training turn holds reach their facing target
    and the relay's yaw rate then dead-bands to 0, which ungates the term. The eval's turn holds saturate at
    0.3 rad/s throughout. Turn is reported only; it does not set a dose.
  - Centring pays about 0.005/s on the fwd/back training rows by design: the lateral share is |sin(bearing)| of
    the intended bearing, up to 0.38 inside the fwd band.
- **Updated.** candidates.yaml doses: S KRABBY_OMNI_STROKE_W 0.117; P KRABBY_OMNI_CENTRING_W 0.160.
