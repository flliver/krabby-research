# Fault-injection sweep — verdicts (fault_read.py)

Root `/home/nickmagus/krabby/krabby-research/parkour/outputs/fault_sweep/lr`; 304 launches; schema fault_read/v1. Every A read: sim plant, force side not enforced. c1, D1 and kind rows: report only.

## Cells

| cell | policy | payload | es1 | es2 | verdict | split | marginal | report only | launch errors |
|---|---|---|---|---|---|---|---|---|---|
| A1_front | C1S | L000 | FALLS | FALLS | FALLS |  | 0.1 |  |  |
| A1_front | c1 | L000 | FALLS | – | FALLS (pending) |  |  | yes |  |
| A1_mid | C1S | L000 | DEGRADES | FALLS | FALLS | split | 0.1 |  |  |
| A1_mid | c1 | L000 | FALLS | – | FALLS (pending) |  |  | yes |  |
| A1_rear | C1S | L000 | DEGRADES | DEGRADES | DEGRADES |  | 0.03 |  |  |
| A1_rear | c1 | L000 | DEGRADES | – | DEGRADES (pending) |  | 0.03 | yes |  |
| A2_front | C1S | L000 | FALLS | – | FALLS |  |  |  |  |
| A2_front | c1 | L000 | FALLS | – | FALLS |  |  | yes |  |
| A2_mid | C1S | L000 | FALLS | FALLS | FALLS |  |  |  |  |
| A2_mid | c1 | L000 | FALLS | – | FALLS (pending) |  |  | yes |  |
| A2_rear | C1S | L000 | FALLS | – | FALLS |  |  |  |  |
| A2_rear | c1 | L000 | FALLS | – | FALLS |  |  | yes |  |
| A3_mid | C1S | L000 | DEGRADES | DEGRADES | DEGRADES |  | 0.03 |  |  |
| A3_mid | c1 | L000 | DEGRADES | – | DEGRADES (pending) |  | 0.03 | yes |  |
| A4_mid | C1S | L000 | DEGRADES | DEGRADES | DEGRADES |  | 0.03 |  |  |
| A4_mid | c1 | L000 | DEGRADES | – | DEGRADES (pending) |  | 0.03 | yes |  |
| A5_mid | C1S | L090 | FALLS | FALLS | FALLS |  |  |  |  |
| A5_mid | c1 | L090 | FALLS | – | FALLS (pending) |  |  | yes |  |
| B1_10 | C1S | L000 | TOLERATED | TOLERATED | TOLERATED |  | 0.03 |  |  |
| B1_10 | c1 | L000 | TOLERATED | – | TOLERATED (pending) |  | 0.03 | yes |  |
| B1_2 | C1S | L000 | TOLERATED | TOLERATED | TOLERATED |  |  |  |  |
| B1_2 | c1 | L000 | TOLERATED | – | TOLERATED (pending) |  |  | yes |  |
| B1_25 | C1S | L000 | FALLS | FALLS | FALLS |  |  |  |  |
| B1_25 | c1 | L000 | FALLS | – | FALLS (pending) |  |  | yes |  |
| B1_5 | C1S | L000 | TOLERATED | TOLERATED | TOLERATED |  |  |  |  |
| B1_5 | c1 | L000 | TOLERATED | – | TOLERATED (pending) |  |  | yes |  |
| B2_0p05 | C1S | L000 | TOLERATED | TOLERATED | TOLERATED |  |  |  |  |
| B2_0p2 | C1S | L000 | TOLERATED | TOLERATED | TOLERATED |  |  |  |  |
| B3_perm | C1S | L000 | DEGRADES | DEGRADES | DEGRADES |  | 0.03 |  |  |
| B4_0p1 | C1S | L000 | TOLERATED | TOLERATED | TOLERATED |  |  |  |  |
| B4_0p1 | c1 | L000 | TOLERATED | – | TOLERATED (pending) |  |  | yes |  |
| B4_0p5 | C1S | L000 | FALLS | FALLS | FALLS |  |  |  |  |
| B4_0p5 | c1 | L000 | FALLS | – | FALLS (pending) |  |  | yes |  |
| B5_perm | C1S | L000 | DEGRADES | DEGRADES | DEGRADES |  |  |  |  |
| B6_0p1 | C1S | L000 | TOLERATED | TOLERATED | TOLERATED |  |  |  |  |
| B6_0p5 | C1S | L000 | TOLERATED | TOLERATED | TOLERATED |  | 0.03 |  |  |
| D1_1 | C1S | L000 | FALLS | – | FALLS |  |  | yes |  |
| S10_0p5 | C1S | L000 | TOLERATED | TOLERATED | TOLERATED |  |  |  |  |
| S10_2 | C1S | L000 | TOLERATED | TOLERATED | TOLERATED |  |  |  |  |
| S10_perm | C1S | L000 | DEGRADES | DEGRADES | DEGRADES |  | 0.03 |  |  |
| S11_2 | C1S | L000 | TOLERATED | TOLERATED | TOLERATED |  |  |  |  |
| S11_perm | C1S | L000 | TOLERATED | TOLERATED | TOLERATED |  |  |  |  |
| S12_30 | C1S | L000 | TOLERATED | TOLERATED | TOLERATED |  |  |  |  |
| S1_0p1 | C1S | L000 | TOLERATED | TOLERATED | TOLERATED |  |  |  |  |
| S1_0p5 | C1S | L000 | FALLS | – | FALLS |  |  |  |  |
| S1_2 | C1S | L000 | FALLS | – | FALLS |  |  |  |  |
| S1_perm | C1S | L000 | FALLS | – | FALLS |  |  |  |  |
| S2_0p5 | C1S | L000 | DEGRADES | DEGRADES | DEGRADES |  | 0.05, 0.1 |  |  |
| S2_2 | C1S | L000 | FALLS | FALLS | FALLS |  |  |  |  |
| S2_perm | C1S | L000 | FALLS | – | FALLS |  |  |  |  |
| S3_0p1 | C1S | L000 | DEGRADES | DEGRADES | DEGRADES |  | 0.03 |  |  |
| S3_0p2 | C1S | L000 | DEGRADES | DEGRADES | DEGRADES |  | 0.1 |  |  |
| S4_perm | C1S | L000 | FALLS | FALLS | FALLS |  |  |  |  |
| S5_0p1 | C1S | L000 | TOLERATED | TOLERATED | TOLERATED |  |  |  |  |
| S5_0p5 | C1S | L000 | TOLERATED | TOLERATED | TOLERATED |  |  |  |  |
| S5_2 | C1S | L000 | TOLERATED | TOLERATED | TOLERATED |  | 0.03 |  |  |
| S6_0p05 | C1S | L000 | TOLERATED | TOLERATED | TOLERATED |  |  |  |  |
| S6_0p1 | C1S | L000 | TOLERATED | TOLERATED | TOLERATED |  |  |  |  |
| S8_held | C1S | L000 | DEGRADES | DEGRADES | DEGRADES |  | 0.03 |  |  |
| S8_zero | C1S | L000 | TOLERATED | TOLERATED | TOLERATED |  |  |  |  |
| T10b | C1S | L000 | FALLS | – | FALLS |  | 0.03, 0.1 |  |  |
| T12 | C1S | L000 | DEGRADES | DEGRADES | DEGRADES |  | 0.1 |  |  |
| Z1_0p5 | C1S | L000 | RECOVERS | DEGRADES | DEGRADES | split | 0.03, 0.05, 0.1 |  |  |
| Z1_2 | C1S | L000 | FALLS | – | FALLS |  |  |  |  |
| Z1_perm | C1S | L000 | FALLS | – | FALLS |  |  |  |  |

## Cell-seeds

| cell | es | policy | payload | ΔF worst (where, n) | F worst | F2 | pre-onset falls | hazard /min | R worst (where) | lines failed | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A1_front | 1 | C1S | L000 | 0.120 (wave8:move_turn, 100) | 0.120 | 0 | 0 | 0.161 | 0.250 (wave3:lat_r) | tracking.side, tracking.back, tracking.lat_r | FALLS |
| A1_front | 2 | C1S | L000 | 0.190 (wave8:move_turn, 100) | 0.190 | 0 | 0 | 0.253 | 0.200 (wave3:lat_r) | tracking.side, tracking.back, tracking.lat_r | FALLS |
| A1_front | 1 | c1 | L000 | 0.150 (wave8:move_turn, 100) | 0.150 | 0 | 0 | 0.282 | 0.110 (wave3:lat_r) | tracking.side, tracking.back, tracking.lat_r | FALLS |
| A1_mid | 1 | C1S | L000 | 0.100 (wave8:move_turn, 100) | 0.100 | 0 | 0 | 0.163 | 0.170 (wave3:lat_r) | tracking.side, tracking.diag, tracking.lat_l, tracking.lat_r, turn_fc.turn, turn_fc.turn_l90, turn_fc.turn_r90 | DEGRADES |
| A1_mid | 2 | C1S | L000 | 0.150 (field_light, 100) | 0.150 | 0 | 0 | 0.243 | 0.220 (wave3:lat_r) | tracking.side, tracking.diag, tracking.lat_l, tracking.lat_r, turn_fc.turn, turn_fc.turn_l90, turn_fc.turn_r90 | FALLS |
| A1_mid | 1 | c1 | L000 | 0.170 (field_light, 100) | 0.170 | 0 | 0 | 0.294 | 0.230 (wave3:lat_r) | tracking.side, tracking.diag, tracking.lat_l, tracking.lat_r, turn_fc.turn, turn_fc.turn_l90, turn_fc.turn_r90, field_light_completion | FALLS |
| A1_rear | 1 | C1S | L000 | 0.040 (field_light, 100) | 0.040 | 0 | 0 | 0.063 | 0.340 (wave1:fwd) | tracking.fwd, tracking.side, tracking.lat_r | DEGRADES |
| A1_rear | 2 | C1S | L000 | 0.060 (wave3:lat_r, 100) | 0.060 | 0 | 0 | 0.062 | 0.270 (wave3:lat_r) | tracking.fwd, tracking.side, tracking.lat_r | DEGRADES |
| A1_rear | 1 | c1 | L000 | 0.020 (wave3:lat_r, 100) | 0.020 | 0 | 0 | 0.031 | 0.200 (wave3:lat_r) | tracking.fwd, tracking.side, tracking.lat_l, tracking.lat_r | DEGRADES |
| A2_front | 1 | C1S | L000 | 1.000 (wave4:back, 100) | 1.000 | 294 | 0 | 15.130 | 0.000 (field_light) | survival.fwd, survival.side, survival.back, survival.diag, survival.turn, survival.move_turn, tracking.fwd, tracking.side, tracking.diag, tracking.lat_l, turn_fc.turn, turn_fc.turn_l90, turn_fc.turn_r90, pitch_rms.suite, pitch_rms.field_light, field_light_completion | FALLS |
| A2_front | 1 | c1 | L000 | 1.000 (wave4:back, 100) | 1.000 | 271 | 0 | 12.246 | 0.000 (field_light) | survival.fwd, survival.side, survival.back, survival.diag, survival.turn, survival.move_turn, tracking.fwd, tracking.side, tracking.diag, tracking.lat_l, turn_fc.turn, turn_fc.turn_l90, turn_fc.turn_r90, pitch_rms.field_light, field_light_completion | FALLS |
| A2_mid | 1 | C1S | L000 | 0.320 (wave8:move_turn, 100) | 0.320 | 0 | 0 | 0.393 | 0.010 (wave3:lat_r) | survival.move_turn, tracking.fwd, tracking.side, tracking.back, tracking.diag, tracking.lat_l, tracking.lat_r, turn_fc.turn, turn_fc.turn_l90, turn_fc.turn_r90, pitch_rms.field_light | FALLS |
| A2_mid | 2 | C1S | L000 | 0.250 (wave8:move_turn, 100) | 0.250 | 0 | 0 | 0.293 | 0.000 (wave1:fwd) | survival.move_turn, tracking.fwd, tracking.side, tracking.back, tracking.diag, tracking.lat_l, tracking.lat_r, turn_fc.turn, turn_fc.turn_l90, turn_fc.turn_r90 | FALLS |
| A2_mid | 1 | c1 | L000 | 0.180 (wave8:move_turn, 100) | 0.180 | 0 | 0 | 0.243 | 0.000 (wave3:lat_r) | tracking.fwd, tracking.side, tracking.back, tracking.diag, tracking.lat_l, tracking.lat_r, turn_fc.turn, turn_fc.turn_l90, turn_fc.turn_r90, pitch_rms.field_light | FALLS |
| A2_rear | 1 | C1S | L000 | 1.000 (wave1:fwd, 100) | 1.000 | 598 | 0 | 20.185 | 0.000 (field_light) | survival.fwd, survival.side, survival.back, survival.diag, survival.turn, survival.move_turn, tracking.side, tracking.back, tracking.lat_l, turn_fc.turn, turn_fc.turn_l90, turn_fc.turn_r90, pitch_rms.suite, pitch_rms.field_light, field_light_completion | FALLS |
| A2_rear | 1 | c1 | L000 | 1.000 (wave1:fwd, 100) | 1.000 | 569 | 0 | 17.392 | 0.000 (field_light) | survival.fwd, survival.side, survival.back, survival.diag, survival.turn, survival.move_turn, tracking.side, tracking.back, tracking.lat_l, turn_fc.turn, turn_fc.turn_l90, turn_fc.turn_r90, pitch_rms.suite, pitch_rms.field_light, field_light_completion | FALLS |
| A3_mid | 1 | C1S | L000 | 0.010 (field_light, 100) | 0.010 | 0 | 0 | 0.016 | 0.740 (wave3:lat_r) | – | DEGRADES |
| A3_mid | 2 | C1S | L000 | 0.000 (wave0:stand, 100) | 0.000 | 0 | 0 | 0.000 | 0.660 (wave3:lat_r) | turn_fc.turn_l90 | DEGRADES |
| A3_mid | 1 | c1 | L000 | 0.010 (field_light, 100) | 0.010 | 0 | 0 | 0.015 | 0.620 (wave3:lat_r) | turn_fc.turn_l90 | DEGRADES |
| A4_mid | 1 | C1S | L000 | 0.030 (field_light, 100) | 0.030 | 0 | 0 | 0.047 | 0.860 (wave1:fwd) | – | DEGRADES |
| A4_mid | 2 | C1S | L000 | 0.050 (field_light, 100) | 0.050 | 0 | 0 | 0.077 | 0.895 (field_light) | – | DEGRADES |
| A4_mid | 1 | c1 | L000 | 0.040 (field_light, 100) | 0.040 | 0 | 0 | 0.063 | 0.957 (field_light) | – | DEGRADES |
| A5_mid | 1 | C1S | L090 | 0.414 (field_light, 99) | 0.414 | 0 | 1 | 0.823 | 0.140 (wave3:lat_r) | tracking.side, tracking.diag, tracking.lat_l, tracking.lat_r, turn_fc.turn, turn_fc.turn_l90, turn_fc.turn_r90, field_light_completion | FALLS |
| A5_mid | 2 | C1S | L090 | 0.370 (field_light, 100) | 0.370 | 0 | 0 | 0.751 | 0.210 (wave3:lat_r) | tracking.fwd, tracking.side, tracking.diag, tracking.lat_l, tracking.lat_r, turn_fc.turn, turn_fc.turn_l90, turn_fc.turn_r90, field_light_completion | FALLS |
| A5_mid | 1 | c1 | L090 | 0.400 (field_light, 100) | 0.400 | 0 | 0 | 0.797 | 0.180 (wave3:lat_r) | tracking.side, tracking.diag, tracking.lat_l, tracking.lat_r, turn_fc.turn, turn_fc.turn_l90, turn_fc.turn_r90, field_light_completion | FALLS |
| B1_10 | 1 | C1S | L000 | 0.020 (wave1:fwd, 100) | 0.020 | 1 | 0 | 0.020 | 0.979 (field_light) | – | TOLERATED |
| B1_10 | 2 | C1S | L000 | 0.000 (wave0:stand, 100) | 0.000 | 0 | 0 | 0.000 | 0.981 (field_light) | – | TOLERATED |
| B1_10 | 1 | c1 | L000 | 0.020 (wave8:move_turn, 100) | 0.020 | 2 | 0 | 0.020 | 1.000 (field_light) | – | TOLERATED |
| B1_2 | 1 | C1S | L000 | 0.000 (wave0:stand, 100) | 0.000 | 0 | 0 | 0.000 | 1.000 (field_light) | – | TOLERATED |
| B1_2 | 2 | C1S | L000 | 0.000 (wave0:stand, 100) | 0.000 | 0 | 0 | 0.000 | 0.965 (field_light) | – | TOLERATED |
| B1_2 | 1 | c1 | L000 | 0.000 (wave0:stand, 100) | 0.000 | 0 | 0 | 0.000 | 1.000 (field_light) | – | TOLERATED |
| B1_25 | 1 | C1S | L000 | 0.220 (wave8:move_turn, 100) | 0.220 | 138 | 1 | 1.588 | 0.962 (wave1:fwd) | survival.fwd, survival.back, survival.diag, survival.move_turn | FALLS |
| B1_25 | 2 | C1S | L000 | 0.300 (wave0:stand, 100) | 0.300 | 168 | 0 | 1.921 | 0.949 (wave1:fwd) | survival.fwd, survival.back, survival.diag, survival.turn, survival.move_turn | FALLS |
| B1_25 | 1 | c1 | L000 | 0.280 (wave0:stand, 100) | 0.280 | 170 | 0 | 1.840 | 0.974 (wave4:back) | survival.fwd, survival.side, survival.turn, survival.move_turn, field_light_completion | FALLS |
| B1_5 | 1 | C1S | L000 | 0.000 (wave0:stand, 100) | 0.000 | 0 | 0 | 0.000 | 1.000 (field_light) | – | TOLERATED |
| B1_5 | 2 | C1S | L000 | 0.000 (wave0:stand, 100) | 0.000 | 0 | 0 | 0.000 | 0.964 (field_light) | – | TOLERATED |
| B1_5 | 1 | c1 | L000 | 0.000 (wave0:stand, 100) | 0.000 | 0 | 0 | 0.000 | 1.000 (field_light) | – | TOLERATED |
| B2_0p05 | 1 | C1S | L000 | 0.000 (wave0:stand, 100) | 0.000 | 0 | 0 | 0.000 | 1.000 (field_light) | – | TOLERATED |
| B2_0p05 | 2 | C1S | L000 | 0.000 (wave0:stand, 100) | 0.000 | 0 | 0 | 0.000 | 0.982 (field_light) | – | TOLERATED |
| B2_0p2 | 1 | C1S | L000 | 0.000 (wave0:stand, 100) | 0.000 | 0 | 0 | 0.000 | 1.000 (field_light) | – | TOLERATED |
| B2_0p2 | 2 | C1S | L000 | 0.000 (wave0:stand, 100) | 0.000 | 0 | 0 | 0.000 | 0.982 (field_light) | – | TOLERATED |
| B3_perm | 1 | C1S | L000 | 0.010 (field_light, 100) | 0.010 | 0 | 0 | 0.016 | 0.830 (wave2:lat_l) | – | DEGRADES |
| B3_perm | 2 | C1S | L000 | 0.000 (wave0:stand, 100) | 0.000 | 0 | 0 | 0.000 | 0.770 (wave2:lat_l) | – | DEGRADES |
| B4_0p1 | 1 | C1S | L000 | 0.000 (wave0:stand, 100) | 0.000 | 0 | 0 | 0.000 | 1.000 (field_light) | – | TOLERATED |
| B4_0p1 | 2 | C1S | L000 | 0.000 (wave0:stand, 100) | 0.000 | 0 | 0 | 0.000 | 0.982 (field_light) | – | TOLERATED |
| B4_0p1 | 1 | c1 | L000 | 0.000 (wave0:stand, 100) | 0.000 | 0 | 0 | 0.000 | 1.000 (field_light) | – | TOLERATED |
| B4_0p5 | 1 | C1S | L000 | 0.430 (wave0:stand, 100) | 0.430 | 287 | 0 | 3.454 | 0.896 (wave4:back) | survival.fwd, survival.side, survival.back, survival.diag, survival.turn, survival.move_turn, field_light_completion | FALLS |
| B4_0p5 | 2 | C1S | L000 | 0.330 (wave0:stand, 100) | 0.330 | 252 | 0 | 3.090 | 0.928 (wave1:fwd) | survival.fwd, survival.side, survival.back, survival.diag, survival.turn, survival.move_turn, field_light_completion | FALLS |
| B4_0p5 | 1 | c1 | L000 | 0.410 (wave0:stand, 100) | 0.410 | 300 | 1 | 3.609 | 0.908 (wave4:back) | survival.fwd, survival.side, survival.back, survival.diag, survival.turn, survival.move_turn, field_light_completion | FALLS |
| B5_perm | 1 | C1S | L000 | 0.000 (wave0:stand, 100) | 0.000 | 0 | 0 | 0.000 | 0.800 (wave2:lat_l) | – | DEGRADES |
| B5_perm | 2 | C1S | L000 | 0.000 (wave0:stand, 100) | 0.000 | 0 | 0 | 0.000 | 0.800 (wave2:lat_l) | – | DEGRADES |
| B6_0p1 | 1 | C1S | L000 | 0.000 (wave0:stand, 100) | 0.000 | 0 | 1 | 0.000 | 0.978 (field_light) | – | TOLERATED |
| B6_0p1 | 2 | C1S | L000 | 0.000 (wave0:stand, 100) | 0.000 | 0 | 0 | 0.000 | 0.946 (field_light) | – | TOLERATED |
| B6_0p5 | 1 | C1S | L000 | 0.010 (wave8:move_turn, 100) | 0.010 | 1 | 0 | 0.010 | 1.000 (field_light) | – | TOLERATED |
| B6_0p5 | 2 | C1S | L000 | 0.010 (wave8:move_turn, 100) | 0.010 | 0 | 0 | 0.010 | 0.981 (field_light) | – | TOLERATED |
| D1_1 | 1 | C1S | L000 | 0.908 (wave8:move_turn, 65) | 0.908 | 119 | 165 | 3.997 | 0.000 (field_light) | survival.fwd, survival.back, survival.diag, survival.move_turn, tracking.fwd, tracking.side, tracking.back, tracking.diag, tracking.lat_l, tracking.lat_r, turn_fc.turn, turn_fc.turn_l90, turn_fc.turn_r90, pitch_rms.field_light, field_light_completion | FALLS |
| S10_0p5 | 1 | C1S | L000 | 0.000 (wave0:stand, 100) | 0.000 | 0 | 0 | 0.000 | 1.000 (field_light) | – | TOLERATED |
| S10_0p5 | 2 | C1S | L000 | 0.000 (wave0:stand, 100) | 0.000 | 0 | 0 | 0.000 | 0.981 (field_light) | – | TOLERATED |
| S10_2 | 1 | C1S | L000 | 0.000 (wave0:stand, 100) | 0.000 | 0 | 0 | 0.000 | 1.000 (field_light) | – | TOLERATED |
| S10_2 | 2 | C1S | L000 | 0.000 (wave0:stand, 100) | 0.000 | 0 | 0 | 0.000 | 0.978 (field_light) | – | TOLERATED |
| S10_perm | 1 | C1S | L000 | 0.020 (field_light, 100) | 0.020 | 0 | 0 | 0.031 | 0.550 (wave5:diag) | turn_fc.turn, turn_fc.turn_l90, turn_fc.turn_r90 | DEGRADES |
| S10_perm | 2 | C1S | L000 | 0.050 (field_light, 100) | 0.050 | 0 | 0 | 0.077 | 0.710 (wave5:diag) | turn_fc.turn, turn_fc.turn_l90, turn_fc.turn_r90 | DEGRADES |
| S11_2 | 1 | C1S | L000 | 0.000 (wave0:stand, 100) | 0.000 | 0 | 0 | 0.000 | 1.000 (field_light) | – | TOLERATED |
| S11_2 | 2 | C1S | L000 | 0.000 (wave0:stand, 100) | 0.000 | 0 | 0 | 0.000 | 1.000 (field_light) | – | TOLERATED |
| S11_perm | 1 | C1S | L000 | 0.000 (wave0:stand, 100) | 0.000 | 0 | 0 | 0.000 | 1.000 (field_light) | – | TOLERATED |
| S11_perm | 2 | C1S | L000 | 0.000 (wave0:stand, 100) | 0.000 | 0 | 0 | 0.000 | 0.982 (field_light) | – | TOLERATED |
| S12_30 | 1 | C1S | L000 | 0.000 (wave0:stand, 100) | 0.000 | 0 | 0 | 0.000 | 0.978 (field_light) | – | TOLERATED |
| S12_30 | 2 | C1S | L000 | 0.000 (wave0:stand, 100) | 0.000 | 0 | 0 | 0.000 | 0.982 (field_light) | – | TOLERATED |
| S1_0p1 | 1 | C1S | L000 | 0.000 (wave0:stand, 100) | 0.000 | 0 | 0 | 0.000 | 1.000 (field_light) | – | TOLERATED |
| S1_0p1 | 2 | C1S | L000 | 0.000 (wave0:stand, 100) | 0.000 | 0 | 0 | 0.000 | 0.982 (field_light) | – | TOLERATED |
| S1_0p5 | 1 | C1S | L000 | 0.880 (wave1:fwd, 100) | 0.880 | 242 | 1 | 2.838 | 0.900 (wave1:fwd) | survival.fwd, survival.back, survival.move_turn, field_light_completion | FALLS |
| S1_2 | 1 | C1S | L000 | 0.990 (wave5:diag, 100) | 0.990 | 457 | 0 | 7.237 | 0.000 (wave5:diag) | survival.fwd, survival.back, survival.diag, survival.move_turn, tracking.fwd, tracking.back, tracking.diag, field_light_completion | FALLS |
| S1_perm | 1 | C1S | L000 | 1.000 (wave1:fwd, 100) | 1.000 | 449 | 0 | 8.455 | 0.000 (wave1:fwd) | survival.fwd, survival.side, survival.back, survival.diag, survival.move_turn, tracking.fwd, tracking.side, tracking.back, tracking.diag, tracking.lat_l, tracking.lat_r, pitch_rms.suite, pitch_rms.field_light, field_light_completion | FALLS |
| S1_perm | 2 | C1S | L000 | – (, ) | – | – | – | – | – () | – | – |
| S2_0p5 | 1 | C1S | L000 | 0.080 (wave8:move_turn, 100) | 0.080 | 14 | 0 | 0.121 | 0.989 (wave1:fwd) | – | DEGRADES |
| S2_0p5 | 2 | C1S | L000 | 0.070 (wave8:move_turn, 100) | 0.070 | 11 | 0 | 0.112 | 0.980 (field_light) | – | DEGRADES |
| S2_2 | 1 | C1S | L000 | 0.220 (wave8:move_turn, 100) | 0.220 | 46 | 0 | 0.770 | 1.000 (field_light) | survival.move_turn | FALLS |
| S2_2 | 2 | C1S | L000 | 0.200 (field_light, 100) | 0.200 | 43 | 0 | 0.639 | 0.962 (field_light) | survival.move_turn, field_light_completion | FALLS |
| S2_perm | 1 | C1S | L000 | 0.980 (field_light, 100) | 0.980 | 36 | 0 | 7.352 | 0.293 (field_light) | survival.fwd, survival.back, survival.diag, survival.move_turn, pitch_rms.field_light, field_light_completion | FALLS |
| S2_perm | 2 | C1S | L000 | – (, ) | – | – | – | – | – () | – | – |
| S3_0p1 | 1 | C1S | L000 | 0.000 (wave0:stand, 100) | 0.000 | 0 | 0 | 0.000 | 0.580 (wave3:lat_r) | – | DEGRADES |
| S3_0p1 | 2 | C1S | L000 | 0.010 (field_light, 100) | 0.010 | 0 | 0 | 0.015 | 0.610 (wave3:lat_r) | – | DEGRADES |
| S3_0p2 | 1 | C1S | L000 | 0.070 (field_light, 100) | 0.070 | 0 | 0 | 0.117 | 0.350 (wave3:lat_r) | pitch_rms.field_light | DEGRADES |
| S3_0p2 | 2 | C1S | L000 | 0.080 (field_light, 100) | 0.080 | 0 | 0 | 0.133 | 0.310 (wave3:lat_r) | pitch_rms.field_light | DEGRADES |
| S4_perm | 1 | C1S | L000 | 0.380 (field_light, 100) | 0.380 | 0 | 0 | 0.734 | 0.696 (field_light) | pitch_rms.field_light, field_light_completion | FALLS |
| S4_perm | 2 | C1S | L000 | 0.510 (field_light, 100) | 0.510 | 0 | 0 | 1.043 | 0.740 (wave3:lat_r) | pitch_rms.field_light, field_light_completion | FALLS |
| S5_0p1 | 1 | C1S | L000 | 0.000 (wave0:stand, 100) | 0.000 | 0 | 0 | 0.000 | 0.957 (field_light) | – | TOLERATED |
| S5_0p1 | 2 | C1S | L000 | 0.000 (wave0:stand, 100) | 0.000 | 0 | 0 | 0.000 | 0.982 (field_light) | – | TOLERATED |
| S5_0p5 | 1 | C1S | L000 | 0.000 (wave0:stand, 100) | 0.000 | 0 | 0 | 0.000 | 1.000 (field_light) | – | TOLERATED |
| S5_0p5 | 2 | C1S | L000 | 0.000 (wave0:stand, 100) | 0.000 | 0 | 0 | 0.000 | 0.981 (field_light) | – | TOLERATED |
| S5_2 | 1 | C1S | L000 | 0.000 (wave0:stand, 100) | 0.000 | 0 | 0 | 0.000 | 1.000 (field_light) | – | TOLERATED |
| S5_2 | 2 | C1S | L000 | 0.010 (field_light, 100) | 0.010 | 0 | 0 | 0.015 | 1.000 (field_light) | – | TOLERATED |
| S6_0p05 | 1 | C1S | L000 | 0.000 (wave0:stand, 100) | 0.000 | 0 | 0 | 0.000 | 1.000 (field_light) | – | TOLERATED |
| S6_0p05 | 2 | C1S | L000 | 0.000 (wave0:stand, 100) | 0.000 | 0 | 0 | 0.000 | 0.982 (field_light) | – | TOLERATED |
| S6_0p1 | 1 | C1S | L000 | 0.000 (wave0:stand, 100) | 0.000 | 0 | 0 | 0.000 | 1.000 (field_light) | – | TOLERATED |
| S6_0p1 | 2 | C1S | L000 | 0.000 (wave0:stand, 100) | 0.000 | 0 | 0 | 0.000 | 0.965 (field_light) | – | TOLERATED |
| S8_held | 1 | C1S | L000 | 0.050 (field_light, 100) | 0.050 | 0 | 0 | 0.081 | 0.060 (wave5:diag) | tracking.side, tracking.back, tracking.diag, tracking.lat_l, tracking.lat_r | DEGRADES |
| S8_held | 2 | C1S | L000 | 0.020 (wave4:back, 100) | 0.020 | 0 | 0 | 0.020 | 0.120 (wave5:diag) | tracking.side, tracking.diag, tracking.lat_l, tracking.lat_r | DEGRADES |
| S8_zero | 1 | C1S | L000 | 0.000 (wave0:stand, 100) | 0.000 | 0 | 0 | 0.000 | 0.940 (wave4:back) | – | TOLERATED |
| S8_zero | 2 | C1S | L000 | 0.000 (wave0:stand, 100) | 0.000 | 0 | 0 | 0.000 | 0.960 (wave4:back) | – | TOLERATED |
| T10b | 1 | C1S | L000 | 1.000 (wave9:move_turn_r, 2) | 1.000 | 33 | 632 | 130.435 | 0.000 (field_light) | survival.fwd, survival.side, survival.back, survival.diag, survival.move_turn, pitch_rms.field_light, field_light_completion | FALLS |
| T12 | 1 | C1S | L000 | 0.061 (wave8:move_turn, 99) | 0.061 | 3 | 1 | 0.093 | 0.470 (wave5:diag) | tracking.diag, turn_fc.turn, turn_fc.turn_l90 | DEGRADES |
| T12 | 2 | C1S | L000 | 0.090 (field_light, 100) | 0.090 | 2 | 6 | 0.141 | 0.370 (wave5:diag) | tracking.diag, turn_fc.turn, turn_fc.turn_l90 | DEGRADES |
| Z1_0p5 | 1 | C1S | L000 | 0.040 (wave8:move_turn, 100) | 0.040 | 17 | 0 | 0.131 | 0.977 (field_light) | – | RECOVERS |
| Z1_0p5 | 2 | C1S | L000 | 0.090 (wave9:move_turn_r, 100) | 0.090 | 22 | 0 | 0.246 | 0.944 (field_light) | – | DEGRADES |
| Z1_2 | 1 | C1S | L000 | 0.570 (wave6:turn_l90, 100) | 0.570 | 244 | 0 | 4.459 | 1.000 (field_light) | survival.fwd, survival.side, survival.back, survival.diag, survival.turn, survival.move_turn, turn_fc.turn, turn_fc.turn_l90, turn_fc.turn_r90, field_light_completion | FALLS |
| Z1_perm | 1 | C1S | L000 | 1.000 (field_light, 100) | 1.000 | 282 | 0 | 19.837 | 0.143 (field_light) | survival.fwd, survival.side, survival.back, survival.diag, survival.turn, survival.move_turn, tracking.fwd, tracking.back, turn_fc.turn, turn_fc.turn_l90, turn_fc.turn_r90, pitch_rms.suite, pitch_rms.field_light, field_light_completion | FALLS |
| Z1_perm | 2 | C1S | L000 | – (, ) | – | – | – | – | – () | – | – |

## Tolerated duration (transient families, C1S)

| family | tolerated d | stopped by |
|---|---|---|
| B1/C1S | 0.20 s (10 steps) | FALLS |
| B4/C1S | 0.10 s (5 steps) | FALLS |
| B6/C1S | 0.50 s (25 steps) | – |
| S1/C1S | 0.10 s (5 steps) | FALLS |
| S10/C1S | 2.00 s (100 steps) | – |
| S11/C1S | 2.00 s (100 steps) | – |
| S2/C1S | – s (– steps) | DEGRADES |
| S5/C1S | 2.00 s (100 steps) | – |
| Z1/C1S | – s (– steps) | DEGRADES |

## Kind rows (report only)

| cell | es | row | success | nominal | Δsuccess | F | F ref |
|---|---|---|---|---|---|---|---|
| S10_perm | 1 | ledge_down | 0.470 | 0.520 | -0.050 | 0.520 | 0.480 |
| S10_perm | 1 | pits | 0.250 | 0.570 | -0.320 | 0.680 | 0.400 |
| S10_perm | 2 | ledge_down | 0.380 | 0.600 | -0.220 | 0.610 | 0.400 |
| S10_perm | 2 | pits | 0.230 | 0.450 | -0.220 | 0.720 | 0.520 |
| S11_perm | 1 | ledge_down | 0.540 | 0.520 | 0.020 | 0.460 | 0.480 |
| S11_perm | 1 | pits | 0.520 | 0.550 | -0.030 | 0.450 | 0.420 |
| S11_perm | 2 | ledge_down | 0.510 | 0.600 | -0.090 | 0.490 | 0.400 |
| S11_perm | 2 | pits | 0.440 | 0.450 | -0.010 | 0.530 | 0.520 |
| S1_perm | 1 | ledge_down | 0.000 | 0.520 | -0.520 | 1.000 | 0.480 |
| S1_perm | 1 | pits | 0.000 | 0.550 | -0.550 | 1.000 | 0.420 |
| S1_perm | 2 | ledge_down | 0.000 | 0.600 | -0.600 | 1.000 | 0.400 |
| S1_perm | 2 | pits | 0.000 | 0.450 | -0.450 | 1.000 | 0.520 |
| S2_perm | 1 | ledge_down | 0.000 | 0.520 | -0.520 | 1.000 | 0.480 |
| S2_perm | 1 | pits | 0.000 | 0.550 | -0.550 | 1.000 | 0.420 |
| S2_perm | 2 | ledge_down | 0.000 | 0.600 | -0.600 | 1.000 | 0.400 |
| S2_perm | 2 | pits | 0.000 | 0.450 | -0.450 | 1.000 | 0.520 |
| S3_0p2 | 1 | ledge_down | 0.300 | 0.520 | -0.220 | 0.670 | 0.480 |
| S3_0p2 | 1 | pits | 0.180 | 0.550 | -0.370 | 0.810 | 0.420 |
| S3_0p2 | 2 | ledge_down | 0.270 | 0.600 | -0.330 | 0.710 | 0.400 |
| S3_0p2 | 2 | pits | 0.240 | 0.450 | -0.210 | 0.730 | 0.520 |
| T12 | 1 | ledge_down | 0.230 | 0.520 | -0.290 | 0.770 | 0.480 |
| T12 | 1 | pits | 0.130 | 0.570 | -0.440 | 0.850 | 0.400 |
| T12 | 2 | ledge_down | 0.290 | 0.600 | -0.310 | 0.710 | 0.400 |
| T12 | 2 | pits | 0.160 | 0.450 | -0.290 | 0.840 | 0.520 |
| Z1_perm | 1 | ledge_down | 0.000 | 0.520 | -0.520 | 1.000 | 0.480 |
| Z1_perm | 1 | pits | 0.000 | 0.570 | -0.570 | 1.000 | 0.400 |
| Z1_perm | 2 | ledge_down | 0.000 | 0.600 | -0.600 | 1.000 | 0.400 |
| Z1_perm | 2 | pits | 0.000 | 0.450 | -0.450 | 1.000 | 0.520 |

## Launch errors

- none
