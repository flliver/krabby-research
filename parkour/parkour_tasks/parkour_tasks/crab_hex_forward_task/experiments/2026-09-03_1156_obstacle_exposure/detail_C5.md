# Detail — arm C5 (2026-09-04 07:43)

| metric | C0 | C3 | C1 | C4 | C2 | C5 | C2_aborted_spurious |
|---|---|---|---|---|---|---|---|
| verdict | CONTROL | FAIL | FAIL | FAIL | FAIL | FAIL | ABORTED |
| lever | none | SPAWN_OFFSET=2.0 | STAND_FRAC=0.2 | SPAWN_SPREAD=1.0:11.0:0.5 | EPISODE_S=70 | RSI_SPAWN_FIX=1 | EPISODE_S=70 |
| reach_edge | 0.327 | 0.942 | 0.759 | 0.402 | 0.632 | 0.492 | 0.519 |
| reach_obst (gate ≥0.80) | 0.054 | 0.535 | 0.520 | 0.099 | 0.503 | 0.121 | 0.371 |
| field_frac (gate ≥0.20) | 0.035 | 0.396 | 0.206 | 0.316 | 0.215 | 0.059 | 0.162 |
| goals_passed (gate ≥2) | 0.055 | 0.416 | 0.348 | 0.233 | 0.391 | 0.097 | 0.279 |
| cov[1] | 0.054 | 0.535 | 0.520 | 0.388 | 0.503 | 0.121 | 0.371 |
| cov[2] | 0.002 | 0.012 | 0.038 | 0.160 | 0.173 | 0.003 | 0.145 |
| cov[3] (gate ≥0.50) | 0.000 | 0.000 | 0.000 | 0.115 | 0.056 | 0.000 | 0.063 |
| cov[4] | 0.000 | 0.000 | 0.000 | 0.076 | 0.019 | 0.000 | 0.021 |
| cov[5] | 0.000 | 0.000 | 0.000 | 0.043 | 0.008 | 0.000 | 0.011 |
| cov[6] (gate ≥0.20) | 0.000 | 0.000 | 0.000 | 0.015 | 0.001 | 0.000 | 0.003 |
| fail flat | 0.429 | 0.301 | 0.296 | 0.277 | 0.580 | 0.391 | 0.637 |
| fail flat /1k steps | 0.590 | 0.373 | 0.374 | 0.340 | 0.295 | 0.508 | 0.356 |
| fail obst | 0.591 | 0.457 | 0.612 | 0.642 | 0.867 | 0.694 | 0.938 |
| fail obst-RSI | 0.614 | 0.499 | 0.705 | 0.620 | 0.863 | 0.676 | 0.853 |
| fail obst-spread | 0.000 | 0.000 | 0.000 | 0.677 | 0.000 | 0.000 | 0.000 |
| ep len (steps) | 700 | 780 | 671 | 622 | 1534 | 690 | 1218 |
| stand time frac (logged) | 0.444 | 0.452 | 0.185 | 0.370 | 0.257 | 0.416 | 0.201 |
| stand time frac (corrected) | 0.634 | 0.580 | 0.275 | 0.595 | 0.586 | 0.602 | 0.576 |
| spread frac | 0.000 | 0.000 | 0.000 | 0.497 | 0.000 | 0.000 | 0.000 |
| RSI frac | 0.200 | 0.214 | 0.228 | 0.186 | 0.220 | 0.206 | 0.215 |
| terrain level | 4.86 | 5.11 | 5.77 | 5.28 | 1.63 | 4.23 | 1.59 |
| goal_idx (legacy) | 0.164 | 0.688 | 0.436 | 1.281 | 0.560 | 0.245 | 0.640 |
| mean reward | 16.8 | 17.0 | 17.3 | 15.4 | 34.5 | 18.1 | 27.2 |
| value loss | 0.0173 | 0.0151 | 0.0255 | 0.0194 | 0.0153 | 0.0192 | 0.0159 |
| canary tripod | 0.536 | 0.567 | 0.573 | 0.569 | 0.524 | 0.552 | n/a |
| canary completion | 0.960 | 0.930 | 0.980 | 0.900 | 0.790 | 0.890 | n/a |
| canary tracking | 0.430 | 0.387 | 0.521 | 0.449 | 0.436 | 0.447 | n/a |
| canary creep vx | 0.108 | 0.091 | 0.144 | 0.118 | 0.112 | 0.119 | n/a |
| canary slip | 0.226 | 0.215 | 0.244 | 0.218 | 0.235 | 0.238 | n/a |
| obstacle eval completion | 0.42 | 0.52 | 0.51 | 0.33 | 0.36 | 0.27 | n/a |
| obstacle eval falls/100 | 58 | 48 | 49 | 67 | 64 | 73 | n/a |

GATES for C5: verdict FAIL (kind hygiene, exposure status n/a)
- safety: VIOLATED — canary completion 0.890 < C0 0.960 - 0.05
- obstacle-tile ceiling: under ceiling
- RSI-episode failure share ok: False

## Miss anatomy (probe roll of the arm's head, its own training stack, stochastic policy; episode 20 s)
episodes complete: 318 | platform-spawned obstacle-tile: 136
reached obstacle 1: 17 (0.12) | missed: 119
MISSED obstacle 1 -> fell: 76/119 (0.64) | timed out: 43/119 (0.36) | other/early non-failure: 0
  fell ON the platform (before the edge): 55/76 (0.72); in the field before obstacle 1: 21/76 (0.28)
  fall time: median 7.9 s | distance from spawn at fall: median 1.31 m | edge is 1.58 m from spawn, obstacle-1 goal 2.08 m
  timed-out non-reachers: max x from spawn median 1.64 m (edge 1.58, obstacle-1 goal 2.08); past the edge: 27/43 (0.63)
REACHED obstacle 1 -> fell later: 6/17 (0.35) | timed out: 11/17 (0.65)
  fall time after reaching: median 17.3 s | x past obstacle-1 goal at fall: median 0.36 m
flat-tile platform episodes: 118 | fell: 42/118 (0.36) (gait-only base rate)
