# Step 4 — anchor-relative gate numbers (frozen into the CHARTER)

Anchor = S1 s16 / s17 / s18 + `body_hold`, es1 + es2 (field_hard es1–es4). Rule: PLAN gates, CHARTER.

| clause | number |
|---|---|
| g1.move_turn_l.survival_min | 0.9500 |
| g1.move_turn_r.survival_min | 0.9350 |
| g2.w_min.ripple_max | 0.0186 |
| g2.0p08.ripple_max | 0.0191 |
| g2.0p15.ripple_max | 0.0139 |
| g2.w_max.net_rate_min | 0.1668 |
| g3.move_turn_l.transl_dev_max | 0.1183 |
| g3.move_turn_r.transl_dev_max | 0.0110 |
| g4.drift_median_max | 0.0100 |
| g4.drift_p90_max | 0.0207 |
| g5.fwd.ratio_dev_max | 0.0823 |
| g5.lat_l.ratio_dev_max | 0.1486 |
| g5.lat_r.ratio_dev_max | 0.1048 |
| g5.back.ratio_dev_max | 0.1209 |
| g5.diag.ratio_dev_max | 0.0880 |
| g5.field_hard__omni_body.completion_min | 0.5200 |
| g5.field_light__omni_body.completion_min | 0.8950 |
| g5.ring_obst__omni_body.completion_min | 0.9200 |

P2 (S1 zero-shot, body mode): drift median 0.00659238748386997, p90 0.015663990943888165; level net ratios {'w_min': 0.022598670513318493, '0p08': -0.0013893493427727965, '0p15': 0.0005405445541110468, 'ceiling': 0.005484091522302495, 'w_max': 0.008402068529114157}; passes gate 2: False; passes gate 4: True

**STOPS:** ['gate 2 ripple clause at w_min would need ripple < 0.03 (0.0186)', 'gate 2 ripple clause at 0p08 would need ripple < 0.03 (0.0191)', 'gate 2 ripple clause at 0p15 would need ripple < 0.03 (0.0139)', 'known-bad P2 PASSES gate 4 (drift median 0.0066, p90 0.0157)']

| anchor | level | net ratio (es1, es2) | ripple | net rate | survival |
|---|---|---|---|---|---|
| s16 | w_min | 0.972, 0.979 | 0.034, 0.032 | 0.049, 0.049 | 1.000, 1.000 |
| s16 | 0p08 | 1.029, 1.024 | 0.038, 0.038 | 0.082, 0.082 | 1.000, 1.000 |
| s16 | 0p15 | 1.028, 1.016 | 0.028, 0.027 | 0.154, 0.152 | 1.000, 1.000 |
| s16 | ceiling | 0.991, 0.979 | 0.023, 0.024 | 0.166, 0.164 | 1.000, 1.000 |
| s16 | w_max | 0.811, 0.815 | 0.027, 0.027 | 0.170, 0.171 | 1.000, 1.000 |
| s17 | w_min | 0.914, 0.920 | 0.033, 0.034 | 0.046, 0.046 | 1.000, 1.000 |
| s17 | 0p08 | 0.997, 0.986 | 0.033, 0.033 | 0.080, 0.079 | 1.000, 1.000 |
| s17 | 0p15 | 0.959, 0.956 | 0.022, 0.021 | 0.144, 0.143 | 1.000, 1.000 |
| s17 | ceiling | 0.906, 0.913 | 0.019, 0.020 | 0.152, 0.153 | 1.000, 1.000 |
| s17 | w_max | 0.770, 0.772 | 0.023, 0.023 | 0.162, 0.162 | 1.000, 1.000 |
| s18 | w_min | 0.961, 0.950 | 0.036, 0.038 | 0.048, 0.047 | 1.000, 1.000 |
| s18 | 0p08 | 1.018, 1.015 | 0.037, 0.037 | 0.081, 0.081 | 1.000, 1.000 |
| s18 | 0p15 | 0.978, 0.973 | 0.022, 0.021 | 0.147, 0.146 | 1.000, 1.000 |
| s18 | ceiling | 0.931, 0.934 | 0.024, 0.023 | 0.156, 0.157 | 1.000, 1.000 |
| s18 | w_max | 0.818, 0.818 | 0.028, 0.027 | 0.172, 0.172 | 1.000, 1.000 |
