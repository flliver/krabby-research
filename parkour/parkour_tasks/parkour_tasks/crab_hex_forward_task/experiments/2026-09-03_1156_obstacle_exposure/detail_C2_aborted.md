# Detail — arm C2 (2026-09-04 02:48)

| metric | C0 | C3 | C1 | C4 | C2 |
|---|---|---|---|---|---|
| verdict | CONTROL | FAIL | FAIL | FAIL | ABORTED |
| lever | none | SPAWN_OFFSET=2.0 | STAND_FRAC=0.2 | SPAWN_SPREAD=1.0:11.0:0.5 | EPISODE_S=70 |
| reach_edge | 0.327 | 0.942 | 0.759 | 0.402 | 0.519 |
| reach_obst (gate ≥0.80) | 0.054 | 0.535 | 0.520 | 0.099 | 0.371 |
| field_frac (gate ≥0.20) | 0.035 | 0.396 | 0.206 | 0.316 | 0.162 |
| goals_passed (gate ≥2) | 0.055 | 0.416 | 0.348 | 0.233 | 0.279 |
| cov[1] | 0.054 | 0.535 | 0.520 | 0.388 | 0.371 |
| cov[2] | 0.002 | 0.012 | 0.038 | 0.160 | 0.145 |
| cov[3] (gate ≥0.50) | 0.000 | 0.000 | 0.000 | 0.115 | 0.063 |
| cov[4] | 0.000 | 0.000 | 0.000 | 0.076 | 0.021 |
| cov[5] | 0.000 | 0.000 | 0.000 | 0.043 | 0.011 |
| cov[6] (gate ≥0.20) | 0.000 | 0.000 | 0.000 | 0.015 | 0.003 |
| fail flat | 0.429 | 0.301 | 0.296 | 0.277 | 0.637 |
| fail flat /1k steps | 0.590 | 0.373 | 0.374 | 0.340 | 0.356 |
| fail obst | 0.591 | 0.457 | 0.612 | 0.642 | 0.938 |
| fail obst-RSI | 0.614 | 0.499 | 0.705 | 0.620 | 0.853 |
| fail obst-spread | 0.000 | 0.000 | 0.000 | 0.677 | 0.000 |
| ep len (steps) | 700 | 780 | 671 | 622 | 1218 |
| stand time frac | 0.444 | 0.452 | 0.185 | 0.370 | 0.201 |
| spread frac | 0.000 | 0.000 | 0.000 | 0.497 | 0.000 |
| RSI frac | 0.200 | 0.214 | 0.228 | 0.186 | 0.215 |
| terrain level | 4.86 | 5.11 | 5.77 | 5.28 | 1.59 |
| goal_idx (legacy) | 0.164 | 0.688 | 0.436 | 1.281 | 0.640 |
| mean reward | 16.8 | 17.0 | 17.3 | 15.4 | 27.2 |
| value loss | 0.0173 | 0.0151 | 0.0255 | 0.0194 | 0.0159 |
| canary tripod | 0.536 | 0.567 | 0.573 | 0.569 | n/a |
| canary completion | 0.960 | 0.930 | 0.980 | 0.900 | n/a |
| canary tracking | 0.430 | 0.387 | 0.521 | 0.449 | n/a |
| canary creep vx | 0.108 | 0.091 | 0.144 | 0.118 | n/a |
| canary slip | 0.226 | 0.215 | 0.244 | 0.218 | n/a |
| obstacle eval completion | 0.42 | 0.52 | 0.51 | 0.33 | n/a |
| obstacle eval falls/100 | 58 | 48 | 49 | 67 | n/a |

## Miss anatomy (probe roll of the arm's head, its own training stack, stochastic policy; episode 70 s)
probe failed — see /home/nickmagus/krabby/krabby-research/sim_fine_tuning/2026-09-03_1156_obstacle_exposure/C2_termination_probe.log
