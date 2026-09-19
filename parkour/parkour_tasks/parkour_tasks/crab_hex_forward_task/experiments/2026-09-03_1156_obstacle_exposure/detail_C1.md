# Detail — arm C1 (2026-09-03 23:15)

| metric | C0 | C3 | C1 |
|---|---|---|---|
| verdict | CONTROL | FAIL | FAIL |
| lever | none | SPAWN_OFFSET=2.0 | STAND_FRAC=0.2 |
| reach_edge | 0.327 | 0.942 | 0.759 |
| reach_obst (gate ≥0.80) | 0.054 | 0.535 | 0.520 |
| field_frac (gate ≥0.20) | 0.035 | 0.396 | 0.206 |
| goals_passed (gate ≥2) | 0.055 | 0.416 | 0.348 |
| cov[1] | 0.054 | 0.535 | 0.520 |
| cov[2] | 0.002 | 0.012 | 0.038 |
| cov[3] (gate ≥0.50) | 0.000 | 0.000 | 0.000 |
| cov[4] | 0.000 | 0.000 | 0.000 |
| cov[5] | 0.000 | 0.000 | 0.000 |
| cov[6] (gate ≥0.20) | 0.000 | 0.000 | 0.000 |
| fail flat | 0.429 | 0.301 | 0.296 |
| fail flat /1k steps | 0.590 | 0.373 | 0.374 |
| fail obst | 0.591 | 0.457 | 0.612 |
| fail obst-RSI | 0.614 | 0.499 | 0.705 |
| fail obst-spread | 0.000 | 0.000 | 0.000 |
| ep len (steps) | 700 | 780 | 671 |
| stand time frac | 0.444 | 0.452 | 0.185 |
| spread frac | 0.000 | 0.000 | 0.000 |
| RSI frac | 0.200 | 0.214 | 0.228 |
| terrain level | 4.86 | 5.11 | 5.77 |
| goal_idx (legacy) | 0.164 | 0.688 | 0.436 |
| mean reward | 16.8 | 17.0 | 17.3 |
| value loss | 0.0173 | 0.0151 | 0.0255 |
| canary tripod | 0.536 | 0.567 | 0.573 |
| canary completion | 0.960 | 0.930 | 0.980 |
| canary tracking | 0.430 | 0.387 | 0.521 |
| canary creep vx | 0.108 | 0.091 | 0.144 |
| canary slip | 0.226 | 0.215 | 0.244 |
| obstacle eval completion | 0.42 | 0.52 | 0.51 |
| obstacle eval falls/100 | 58 | 48 | 49 |

GATES for C1: verdict FAIL (kind exposure, exposure status FAIL)
- exposure: reach_obst_frac 0.520 < 0.8 (C0 0.054)
- exposure: goals_passed_mean 0.348 < 2.0 (C0 0.055)
- exposure: obst_coverage_3 0.000 < 0.5 (C0 0.000)
- exposure: obst_coverage_6 0.000 < 0.2 (C0 0.000)
- exposure: field_frac_mean 0.206 >= 0.2 (C0 0.035)
- safety: all gates held
- obstacle-tile ceiling: under ceiling

## Miss anatomy (probe roll of the arm's head, its own training stack, stochastic policy; episode 20 s)
episodes complete: 334 | platform-spawned obstacle-tile: 145
reached obstacle 1: 71 (0.49) | missed: 74
MISSED obstacle 1 -> fell: 61/74 (0.82) | timed out: 13/74 (0.18) | other/early non-failure: 0
  fell ON the platform (before the edge): 44/61 (0.72); in the field before obstacle 1: 17/61 (0.28)
  fall time: median 5.7 s | distance from spawn at fall: median 1.24 m | edge is 1.58 m from spawn, obstacle-1 goal 2.08 m
  timed-out non-reachers: max x from spawn median 1.94 m (edge 1.58, obstacle-1 goal 2.08); past the edge: 10/13 (0.77)
REACHED obstacle 1 -> fell later: 24/71 (0.34) | timed out: 47/71 (0.66)
  fall time after reaching: median 17.0 s | x past obstacle-1 goal at fall: median 0.64 m
flat-tile platform episodes: 110 | fell: 43/110 (0.39) (gait-only base rate)
