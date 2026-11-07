| term | base | band | verdict | status by level | tolerated low / high | lenient low / high | drop flags |
|---|---|---|---|---|---|---|---|
| T01 rod_speed | L000 | (0.7, 1.35) | ROBUST | 0p5: fail, 0p7: pass, 0p8: pass, 1p15: pass, 1p35: pass | 0.7 / 1.35 | 0.5 / 1.35 | 1 |
| T01 rod_speed | L090 | (0.7, 1.35) | FRAGILE | 0p5: fail, 0p7: fail, 0p8: split, 1p15: pass, 1p35: pass | None / 1.35 | 0.7 / 1.35 | 13 |
| T02 rod_effort | L000 | (0.85, 1.0) | ROBUST | 0p5: fail, 0p7: pass, 0p85: pass | 0.7 / None | 0.5 / None | 3 |
| T02 rod_effort | L090 | (0.85, 1.0) | ROBUST | 0p5: fail, 0p7: fail, 0p85: pass | 0.85 / None | 0.7 / None | 15 |
| T03 cam | L000 | (0.8, 1.2) | ROBUST | 0p5: fail, 0p8: pass, 1p2: pass | 0.8 / 1.2 | 0.5 / 1.2 | 0 |
| T04 kpkd | L000 | (0.75, 1.25) | ROBUST | 0p5: pass, 0p75: pass, 1p25: pass | 0.5 / 1.25 | 0.5 / 1.25 | 0 |
| T05 leg_mass | L000 | (0.9, 1.15) | ROBUST | 0p9: pass, 1p15: pass, 1p5: pass | 0.9 / 1.5 | 0.9 / 1.5 | 0 |
| T06 mu | L000 | (0.4, 1.2) | ROBUST | 0p3: pass, 0p4: pass, 0p8: pass, 1p2: pass | 0.3 / 1.2 | 0.3 / 1.2 | 0 |
| T07 act_delay | L000 | (1, 2) | ROBUST | 2: pass, 3: fail, 4: fail | None / 2 | None / 4 | 2 |
| T08 obs_noise | L000 | (0.0, 1.0) | ROBUST | 0p5: pass, 1: pass, 2: pass, 4: pass | None / 4.0 | None / 4.0 | 0 |
| T09a cur_scale | L000 | (0.85, 1.15) | ROBUST | 0p85: pass, 1p15: pass | 0.85 / 1.15 | 0.85 / 1.15 | 0 |
| T09b cur_noise | L000 | (0.0, 0.1) | ROBUST | 0p1: pass, 0p25: pass | None / 0.25 | None / 0.25 | 0 |
| T10a vxy_noise | L000 | (0.0, 0.1) | ROBUST | 0p05: pass, 0p1: pass, 0p2: fail | None / 0.1 | None / 0.2 | 0 |
| T10b vxy_zero | L000 | None | None | 1: fail | None / None | None / None | 7 |
| T11 depth_lag | L000 | (1, 2) | ROBUST | 2: pass, 3: pass | None / 3 | None / 3 | 0 |
| T12 zero_depth | L000 | None | None | 1: pass | None / 1 | None / 1 | 0 |
| T13 payload_kg | payload | None | None | 150: split, 165: fail | None / None | None / 165.0 | 0 |
| T14 payload_com | L090 | None | None | 0p2: pass, 0p3: split | None / 0.2 | None / 0.3 | 4 |
| T15 push_dv | L000 | None | None | 0p1: pass, 0p2: pass, 0p35: pass, 0p5: pass | None / 0.5 | None / 0.5 | 0 |

**Combined verdicts (worst over bases):** T01 FRAGILE, T02 ROBUST, T03 ROBUST, T04 ROBUST, T05 ROBUST, T06 ROBUST, T07 ROBUST, T08 ROBUST, T09a ROBUST, T09b ROBUST, T10a ROBUST, T11 ROBUST

| report-only term | base | last pass | first fail | split | one-seed levels | search stopped |
|---|---|---|---|---|---|---|
| T13 payload_kg | payload | None | 165.0 | 150 | - | - |
| T14 payload_com | L090 | 0.2 | None | 0p3 | 0p2 | - |
| T15 push_dv | L000 | 0.5 | None | - | 0p1, 0p2, 0p35 | - |

| cell-seed | ref (n) | worst survival delta | completion delta | flags |
|---|---|---|---|---|
| T01 rod_speed 0p5 L000 es1 | L000 (2) | -0.010 | -0.150 | 1 |
| T01 rod_speed 0p7 L000 es1 | L000 (2) | +0.000 | -0.020 | 0 |
| T01 rod_speed 0p7 L000 es2 | L000 (2) | +0.000 | -0.040 | 0 |
| T01 rod_speed 0p8 L000 es1 | L000 (2) | +0.000 | +0.000 | 0 |
| T01 rod_speed 1p15 L000 es1 | L000 (2) | +0.000 | -0.020 | 0 |
| T01 rod_speed 1p35 L000 es1 | L000 (2) | +0.000 | +0.000 | 0 |
| T01 rod_speed 1p35 L000 es2 | L000 (2) | +0.000 | +0.010 | 0 |
| T01 rod_speed 0p5 L090 es1 | L090 (1) | -0.790 | -0.940 | 5 |
| T01 rod_speed 0p7 L090 es1 | L090 (1) | -0.170 | -0.390 | 3 |
| T01 rod_speed 0p7 L090 es2 | L090 (1) | -0.130 | -0.290 | 3 |
| T01 rod_speed 0p8 L090 es1 | L090 (1) | -0.020 | -0.210 | 1 |
| T01 rod_speed 0p8 L090 es2 | L090 (1) | -0.010 | -0.100 | 1 |
| T01 rod_speed 1p15 L090 es1 | L090 (1) | +0.000 | +0.020 | 0 |
| T01 rod_speed 1p35 L090 es1 | L090 (1) | +0.000 | +0.020 | 0 |
| T01 rod_speed 1p35 L090 es2 | L090 (1) | +0.000 | +0.050 | 0 |
| T02 rod_effort 0p5 L000 es1 | L000 (2) | -0.105 | -0.760 | 3 |
| T02 rod_effort 0p7 L000 es1 | L000 (2) | +0.000 | +0.000 | 0 |
| T02 rod_effort 0p85 L000 es1 | L000 (2) | +0.000 | +0.000 | 0 |
| T02 rod_effort 0p85 L000 es2 | L000 (2) | +0.000 | +0.010 | 0 |
| T02 rod_effort 0p5 L090 es1 | L090 (1) | -1.000 | -0.970 | 7 |
| T02 rod_effort 0p7 L090 es1 | L090 (1) | -0.170 | -0.850 | 6 |
| T02 rod_effort 0p85 L090 es1 | L090 (1) | +0.000 | -0.100 | 1 |
| T02 rod_effort 0p85 L090 es2 | L090 (1) | -0.010 | -0.100 | 1 |
| T03 cam 0p5 L000 es1 | L000 (2) | -0.005 | +0.000 | 0 |
| T03 cam 0p8 L000 es1 | L000 (2) | +0.000 | +0.000 | 0 |
| T03 cam 0p8 L000 es2 | L000 (2) | +0.000 | +0.010 | 0 |
| T03 cam 1p2 L000 es1 | L000 (2) | +0.000 | +0.000 | 0 |
| T03 cam 1p2 L000 es2 | L000 (2) | +0.000 | +0.010 | 0 |
| T04 kpkd 0p5 L000 es1 | L000 (2) | +0.000 | -0.010 | 0 |
| T04 kpkd 0p75 L000 es1 | L000 (2) | +0.000 | +0.000 | 0 |
| T04 kpkd 0p75 L000 es2 | L000 (2) | +0.000 | +0.000 | 0 |
| T04 kpkd 1p25 L000 es1 | L000 (2) | +0.000 | -0.010 | 0 |
| T04 kpkd 1p25 L000 es2 | L000 (2) | +0.000 | +0.010 | 0 |
| T05 leg_mass 0p9 L000 es1 | L000 (2) | +0.000 | +0.000 | 0 |
| T05 leg_mass 0p9 L000 es2 | L000 (2) | +0.000 | +0.010 | 0 |
| T05 leg_mass 1p15 L000 es1 | L000 (2) | +0.000 | +0.000 | 0 |
| T05 leg_mass 1p15 L000 es2 | L000 (2) | +0.000 | +0.000 | 0 |
| T05 leg_mass 1p5 L000 es1 | L000 (2) | +0.000 | +0.000 | 0 |
| T06 mu 0p3 L000 es1 | L000 (2) | +0.000 | +0.000 | 0 |
| T06 mu 0p4 L000 es1 | L000 (2) | +0.000 | +0.000 | 0 |
| T06 mu 0p4 L000 es2 | L000 (2) | +0.000 | +0.010 | 0 |
| T06 mu 0p8 L000 es1 | L000 (2) | +0.000 | +0.000 | 0 |
| T06 mu 1p2 L000 es1 | L000 (2) | +0.000 | -0.010 | 0 |
| T06 mu 1p2 L000 es2 | L000 (2) | +0.000 | +0.010 | 0 |
| T07 act_delay 2 L000 es1 | L000 (2) | +0.000 | -0.010 | 0 |
| T07 act_delay 2 L000 es2 | L000 (2) | +0.000 | +0.010 | 0 |
| T07 act_delay 3 L000 es1 | L000 (2) | +0.000 | -0.020 | 0 |
| T07 act_delay 4 L000 es1 | L000 (2) | -0.110 | -0.230 | 2 |
| T08 obs_noise 0p5 L000 es1 | L000 (2) | +0.000 | -0.020 | 0 |
| T08 obs_noise 1 L000 es1 | L000 (2) | +0.000 | +0.000 | 0 |
| T08 obs_noise 1 L000 es2 | L000 (2) | +0.000 | +0.010 | 0 |
| T08 obs_noise 2 L000 es1 | L000 (2) | +0.000 | -0.010 | 0 |
| T08 obs_noise 4 L000 es1 | L000 (2) | +0.000 | +0.000 | 0 |
| T09a cur_scale 0p85 L000 es1 | L000 (2) | +0.000 | +0.000 | 0 |
| T09a cur_scale 0p85 L000 es2 | L000 (2) | +0.000 | +0.000 | 0 |
| T09a cur_scale 1p15 L000 es1 | L000 (2) | +0.000 | +0.000 | 0 |
| T09a cur_scale 1p15 L000 es2 | L000 (2) | +0.000 | +0.010 | 0 |
| T09b cur_noise 0p1 L000 es1 | L000 (2) | +0.000 | +0.000 | 0 |
| T09b cur_noise 0p1 L000 es2 | L000 (2) | +0.000 | +0.010 | 0 |
| T09b cur_noise 0p25 L000 es1 | L000 (2) | +0.000 | -0.010 | 0 |
| T10a vxy_noise 0p05 L000 es1 | L000 (2) | +0.000 | +0.000 | 0 |
| T10a vxy_noise 0p1 L000 es1 | L000 (2) | +0.000 | +0.000 | 0 |
| T10a vxy_noise 0p1 L000 es2 | L000 (2) | +0.000 | +0.010 | 0 |
| T10a vxy_noise 0p2 L000 es1 | L000 (2) | +0.000 | +0.000 | 0 |
| T10b vxy_zero 1 L000 es1 | L000 (2) | -1.000 | -1.000 | 7 |
| T11 depth_lag 2 L000 es1 | L000 (2) | +0.000 | -0.010 | 0 |
| T11 depth_lag 2 L000 es2 | L000 (2) | +0.000 | +0.000 | 0 |
| T11 depth_lag 3 L000 es1 | L000 (2) | +0.000 | +0.000 | 0 |
| T12 zero_depth 1 L000 es1 | L000 (2) | -0.040 | -0.030 | 0 |
| T13 payload_kg 150 payload es1 | L090 (1) | -0.010 | -0.110 | 0 |
| T13 payload_kg 150 payload es2 | L090 (1) | -0.015 | -0.220 | 0 |
| T13 payload_kg 165 payload es1 | L090 (1) | -0.050 | -0.290 | 0 |
| T13 payload_kg 165 payload es2 | L090 (1) | -0.040 | -0.240 | 0 |
| T14 payload_com 0p2 L090 es1 | L090 (1) | -0.025 | -0.020 | 0 |
| T14 payload_com 0p3 L090 es1 | L090 (1) | -0.060 | -0.120 | 2 |
| T14 payload_com 0p3 L090 es2 | L090 (1) | -0.085 | -0.190 | 2 |
| T15 push_dv 0p1 L000 es1 | L000 (2) | +0.000 | -0.010 | 0 |
| T15 push_dv 0p2 L000 es1 | L000 (2) | +0.000 | +0.000 | 0 |
| T15 push_dv 0p35 L000 es1 | L000 (2) | +0.000 | -0.010 | 0 |
| T15 push_dv 0p5 L000 es1 | L000 (2) | +0.000 | +0.000 | 0 |
| T15 push_dv 0p5 L000 es2 | L000 (2) | -0.010 | +0.000 | 0 |
