# Step 2 (Phase A) read

Verdict: P-config PASS; P-early FAIL; A3 PASS; STOP-A YES

## P-config

- **s1** PASS: env.yaml 0 allowed / 0 not; agent.yaml 2 allowed / 0 not
- **c4** PASS: env.yaml 0 allowed / 0 not; agent.yaml 3 allowed / 0 not
- **c5** PASS: env.yaml 0 allowed / 0 not; agent.yaml 3 allowed / 0 not
- **c6** PASS: env.yaml 0 allowed / 0 not; agent.yaml 3 allowed / 0 not
- **c6x** PASS: env.yaml 0 allowed / 0 not; agent.yaml 3 allowed / 0 not
- **p2a** PASS: env.yaml 0 allowed / 0 not; agent.yaml 3 allowed / 0 not

## P-early (s1, steps 0–19)

- shared tags 148 (X43 148, new 152); bitwise 142; differ 6 → FAIL
  - `Loss/hist_latent` max|Δ| 5.96e-08 (2 ulp) from step 0 — NOT tolerated
  - `Loss/priv_reg` max|Δ| 1.43e-06 (48 ulp) from step 1 — tolerated
  - `Perf/collection time` max|Δ| 0.0806 (1352368 ulp) from step 0 — tolerated
  - `Perf/learning_time` max|Δ| 0.0297 (996416 ulp) from step 0 — tolerated
  - `Perf/total_fps` max|Δ| 396 (851968 ulp) from step 0 — tolerated
  - `Train/mean_episode_length/time` max|Δ| 20.3 (1331500 ulp) from step 9 — tolerated
  - only X43: []; only new: ['Metrics/base_parkour/demote_count', 'Metrics/base_parkour/promote_count', 'Metrics/base_parkour/terrain_levels_obst', 'Metrics/base_parkour/track_ratio_obst_p50']

## A3 (S1 + fixed added mass)

| mass | added (run_meta) | settle: terminated / n | settle max tilt ° | settle sag m | suite survival | completion |
|---|---|---|---|---|---|---|
| 0 kg | 0.00 / 0.00 | 0 / 32 | 1.76 | 0.0009 | 1.000 | 1.0 |
| 45 kg | 45.00 / 45.00 | 0 / 32 | 1.81 | 0.0010 | 1.000 | 1.0 |
| 90 kg | 90.00 / 90.00 | 0 / 32 | 1.83 | 0.0010 | 0.999 | 0.999 |
| 135 kg | 135.00 / 135.00 | 0 / 32 | 1.87 | 0.0011 | 0.988 | 0.988 |
| 180 kg | 180.00 / 180.00 | 0 / 32 | 1.90 | 0.0012 | 0.922 | 0.92 |

A3 verdict: {'knob_inert': False, 'settle_tips_le_90': False, 'suite_survival_90': 0.999, 'stop_a': False, 'status': 'PASS'}
Bonus (0-kg suite vs REFS_S1_s18 es1): {'metrics_equal': True, 'n_value_mismatch': 0, 'n_key_mismatch': 0, 'missing_episodes': 0}
