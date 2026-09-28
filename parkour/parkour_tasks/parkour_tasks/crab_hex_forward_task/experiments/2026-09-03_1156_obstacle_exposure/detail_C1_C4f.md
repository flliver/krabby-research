# Detail — arm C1+C4f (2026-09-04 10:45)

| metric | C0 | C3 | C1 | C4 | C2 | C5 | C2_aborted_spurious | C1+C4f |
|---|---|---|---|---|---|---|---|---|
| verdict | CONTROL | FAIL | FAIL | FAIL | FAIL | FAIL | ABORTED | FAIL |
| lever | none | SPAWN_OFFSET=2.0 | STAND_FRAC=0.2 | SPAWN_SPREAD=1.0:11.0:0.5 | EPISODE_S=70 | RSI_SPAWN_FIX=1 | EPISODE_S=70 | STAND_FRAC=0.2, SPAWN_SPREAD=2.5:11.0:0.25 |
| reach_edge | 0.327 | 0.942 | 0.759 | 0.402 | 0.632 | 0.492 | 0.519 | 0.716 |
| reach_obst (gate ≥0.80) | 0.054 | 0.535 | 0.520 | 0.099 | 0.503 | 0.121 | 0.371 | 0.475 |
| field_frac (gate ≥0.20) | 0.035 | 0.396 | 0.206 | 0.316 | 0.215 | 0.059 | 0.162 | 0.296 |
| goals_passed (gate ≥2) | 0.055 | 0.416 | 0.348 | 0.233 | 0.391 | 0.097 | 0.279 | 0.417 |
| cov[1] | 0.054 | 0.535 | 0.520 | 0.388 | 0.503 | 0.121 | 0.371 | 0.552 |
| cov[2] | 0.002 | 0.012 | 0.038 | 0.160 | 0.173 | 0.003 | 0.145 | 0.131 |
| cov[3] (gate ≥0.50) | 0.000 | 0.000 | 0.000 | 0.115 | 0.056 | 0.000 | 0.063 | 0.077 |
| cov[4] | 0.000 | 0.000 | 0.000 | 0.076 | 0.019 | 0.000 | 0.021 | 0.060 |
| cov[5] | 0.000 | 0.000 | 0.000 | 0.043 | 0.008 | 0.000 | 0.011 | 0.041 |
| cov[6] (gate ≥0.20) | 0.000 | 0.000 | 0.000 | 0.015 | 0.001 | 0.000 | 0.003 | 0.014 |
| fail flat | 0.429 | 0.301 | 0.296 | 0.277 | 0.580 | 0.391 | 0.637 | 0.367 |
| fail flat /1k steps | 0.590 | 0.373 | 0.374 | 0.340 | 0.295 | 0.508 | 0.356 | 0.479 |
| fail obst | 0.591 | 0.457 | 0.612 | 0.642 | 0.867 | 0.694 | 0.938 | 0.682 |
| fail obst-RSI | 0.614 | 0.499 | 0.705 | 0.620 | 0.863 | 0.676 | 0.853 | 0.722 |
| fail obst-spread | 0.000 | 0.000 | 0.000 | 0.677 | 0.000 | 0.000 | 0.000 | 0.801 |
| ep len (steps) | 700 | 780 | 671 | 622 | 1534 | 690 | 1218 | 678 |
| stand time frac (logged) | 0.444 | 0.452 | 0.185 | 0.370 | 0.257 | 0.416 | 0.201 | 0.138 |
| stand time frac (corrected) | 0.634 | 0.580 | 0.275 | 0.595 | 0.586 | 0.602 | 0.576 | 0.204 |
| spread frac | 0.000 | 0.000 | 0.000 | 0.497 | 0.000 | 0.000 | 0.000 | 0.194 |
| RSI frac | 0.200 | 0.214 | 0.228 | 0.186 | 0.220 | 0.206 | 0.215 | 0.189 |
| terrain level | 4.86 | 5.11 | 5.77 | 5.28 | 1.63 | 4.23 | 1.59 | 5.89 |
| goal_idx (legacy) | 0.164 | 0.688 | 0.436 | 1.281 | 0.560 | 0.245 | 0.640 | 1.088 |
| mean reward | 16.8 | 17.0 | 17.3 | 15.4 | 34.5 | 18.1 | 27.2 | 18.5 |
| value loss | 0.0173 | 0.0151 | 0.0255 | 0.0194 | 0.0153 | 0.0192 | 0.0159 | 0.0261 |
| canary tripod | 0.536 | 0.567 | 0.573 | 0.569 | 0.524 | 0.552 | n/a | 0.576 |
| canary completion | 0.960 | 0.930 | 0.980 | 0.900 | 0.790 | 0.890 | n/a | 0.920 |
| canary tracking | 0.430 | 0.387 | 0.521 | 0.449 | 0.436 | 0.447 | n/a | 0.515 |
| canary creep vx | 0.108 | 0.091 | 0.144 | 0.118 | 0.112 | 0.119 | n/a | 0.135 |
| canary slip | 0.226 | 0.215 | 0.244 | 0.218 | 0.235 | 0.238 | n/a | 0.215 |
| obstacle eval completion | 0.42 | 0.52 | 0.51 | 0.33 | 0.36 | 0.27 | n/a | 0.46 |
| obstacle eval falls/100 | 58 | 48 | 49 | 67 | 64 | 73 | n/a | 54 |

GATES for C1+C4f: verdict FAIL (kind exposure, exposure status FAIL)
- exposure: reach_obst_frac 0.475 < 0.8 (C0 0.054)
- exposure: goals_passed_mean 0.417 < 2.0 (C0 0.055)
- exposure: obst_coverage_3 0.077 < 0.5 (C0 0.000)
- exposure: obst_coverage_6 0.014 < 0.2 (C0 0.000)
- exposure: field_frac_mean 0.296 >= 0.2 (C0 0.035)
- safety: all gates held
- obstacle-tile ceiling: under ceiling

## Miss anatomy (probe roll of the arm's head, its own training stack, stochastic policy; episode 20 s)
episodes complete: 317 | platform-spawned obstacle-tile: 125
reached obstacle 1: 61 (0.49) | missed: 64
MISSED obstacle 1 -> fell: 53/64 (0.83) | timed out: 11/64 (0.17) | other/early non-failure: 0
  fell ON the platform (before the edge): 26/53 (0.49); in the field before obstacle 1: 27/53 (0.51)
  fall time: median 7.7 s | distance from spawn at fall: median 1.60 m | edge is 1.58 m from spawn, obstacle-1 goal 2.16 m
  timed-out non-reachers: max x from spawn median 2.17 m (edge 1.58, obstacle-1 goal 2.32); past the edge: 11/11 (1.00)
REACHED obstacle 1 -> fell later: 23/61 (0.38) | timed out: 38/61 (0.62)
  fall time after reaching: median 17.1 s | x past obstacle-1 goal at fall: median 0.65 m
flat-tile platform episodes: 87 | fell: 31/87 (0.36) (gait-only base rate)
