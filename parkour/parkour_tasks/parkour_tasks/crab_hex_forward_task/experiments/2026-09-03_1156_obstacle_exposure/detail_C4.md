# Detail — arm C4 (2026-09-04 01:43)

| metric | C0 | C3 | C1 | C4 |
|---|---|---|---|---|
| verdict | CONTROL | FAIL | FAIL | FAIL |
| lever | none | SPAWN_OFFSET=2.0 | STAND_FRAC=0.2 | SPAWN_SPREAD=1.0:11.0:0.5 |
| reach_edge | 0.327 | 0.942 | 0.759 | 0.402 |
| reach_obst (gate ≥0.80) | 0.054 | 0.535 | 0.520 | 0.099 |
| field_frac (gate ≥0.20) | 0.035 | 0.396 | 0.206 | 0.316 |
| goals_passed (gate ≥2) | 0.055 | 0.416 | 0.348 | 0.233 |
| cov[1] | 0.054 | 0.535 | 0.520 | 0.388 |
| cov[2] | 0.002 | 0.012 | 0.038 | 0.160 |
| cov[3] (gate ≥0.50) | 0.000 | 0.000 | 0.000 | 0.115 |
| cov[4] | 0.000 | 0.000 | 0.000 | 0.076 |
| cov[5] | 0.000 | 0.000 | 0.000 | 0.043 |
| cov[6] (gate ≥0.20) | 0.000 | 0.000 | 0.000 | 0.015 |
| fail flat | 0.429 | 0.301 | 0.296 | 0.277 |
| fail flat /1k steps | 0.590 | 0.373 | 0.374 | 0.340 |
| fail obst | 0.591 | 0.457 | 0.612 | 0.642 |
| fail obst-RSI | 0.614 | 0.499 | 0.705 | 0.620 |
| fail obst-spread | 0.000 | 0.000 | 0.000 | 0.677 |
| ep len (steps) | 700 | 780 | 671 | 622 |
| stand time frac | 0.444 | 0.452 | 0.185 | 0.370 |
| spread frac | 0.000 | 0.000 | 0.000 | 0.497 |
| RSI frac | 0.200 | 0.214 | 0.228 | 0.186 |
| terrain level | 4.86 | 5.11 | 5.77 | 5.28 |
| goal_idx (legacy) | 0.164 | 0.688 | 0.436 | 1.281 |
| mean reward | 16.8 | 17.0 | 17.3 | 15.4 |
| value loss | 0.0173 | 0.0151 | 0.0255 | 0.0194 |
| canary tripod | 0.536 | 0.567 | 0.573 | 0.569 |
| canary completion | 0.960 | 0.930 | 0.980 | 0.900 |
| canary tracking | 0.430 | 0.387 | 0.521 | 0.449 |
| canary creep vx | 0.108 | 0.091 | 0.144 | 0.118 |
| canary slip | 0.226 | 0.215 | 0.244 | 0.218 |
| obstacle eval completion | 0.42 | 0.52 | 0.51 | 0.33 |
| obstacle eval falls/100 | 58 | 48 | 49 | 67 |

GATES for C4: verdict FAIL (kind exposure, exposure status FAIL)
- exposure: reach_obst_frac 0.099 < 0.8 (C0 0.054)
- exposure: goals_passed_mean 0.233 < 2.0 (C0 0.055)
- exposure: obst_coverage_3 0.115 < 0.5 (C0 0.000)
- exposure: obst_coverage_6 0.015 < 0.2 (C0 0.000)
- exposure: field_frac_mean 0.316 >= 0.2 (C0 0.035)
- safety: VIOLATED — canary completion 0.900 < C0 0.960 - 0.05
- obstacle-tile ceiling: under ceiling

## Miss anatomy (probe roll of the arm's head, its own training stack, stochastic policy; episode 20 s)
episodes complete: 351 | platform-spawned obstacle-tile: 104
reached obstacle 1: 9 (0.09) | missed: 95
MISSED obstacle 1 -> fell: 60/95 (0.63) | timed out: 35/95 (0.37) | other/early non-failure: 0
  fell ON the platform (before the edge): 54/60 (0.90); in the field before obstacle 1: 6/60 (0.10)
  fall time: median 6.1 s | distance from spawn at fall: median 1.20 m | edge is 1.58 m from spawn, obstacle-1 goal 2.16 m
  timed-out non-reachers: max x from spawn median 1.60 m (edge 1.58, obstacle-1 goal 2.16); past the edge: 18/35 (0.51)
REACHED obstacle 1 -> fell later: 6/9 (0.67) | timed out: 3/9 (0.33)
  fall time after reaching: median 16.3 s | x past obstacle-1 goal at fall: median 0.21 m
flat-tile platform episodes: 65 | fell: 32/65 (0.49) (gait-only base rate)
