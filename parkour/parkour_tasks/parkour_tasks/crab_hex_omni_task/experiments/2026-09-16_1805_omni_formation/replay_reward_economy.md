# Offline reward-economy replay -- omni 1c fix (2026-09-18)

episodes: walker(1a) n=100, mid(1b) n=100, creeper(1c) n=100
clock-phase fit: 36 offsets/episode, argmax of OLD clock income (schedule[sum]+apex)
step mask: steady_mask & segment & pre-termination
weights: 1c training-log reward table; income/s = mean_t(weight * value) (Isaac Lab accumulates weight*value*dt)
not replayed: reward_stride_length (w 0.5; logged income 0.000/s in every 1c window -- the term is identically 0 for these heads), penalty_tibia_deviation_in_stance (w -0.28; logged -0.042/s -- needs default_joint_pos, not in the raw), and the 18 terms at weight 0.

## clock-phase fit sensitivity (income/s of schedule[sum]+apex over the offset grid)

| head | best | mean | worst | spread |
|---|---|---|---|---|
| walker(1a) | 0.679 | 0.526 | 0.329 | 0.350 |
| mid(1b) | 0.743 | 0.549 | 0.315 | 0.428 |
| creeper(1c) | 0.845 | 0.585 | 0.350 | 0.495 |

## Reconstruction fidelity -- 1c head, OLD terms, every steady step vs the 1c training log

Training column = ``Episode_Reward/<name>`` at iteration 6997 / 40 s (the log's per-second income). The two populations differ (100 canary episodes on a fixed 8-hold schedule vs 256 training envs on the 1c sampler), so agreement to ~15 % is the expected ceiling; it is the check that the formulas, weights, gates and the fitted clock phase are right.

| term | replay (canary) | training log 1c @6997 |
|---|---|---|
| track_lin_exp | 0.841 | 0.826 |
| track_ang_exp | 0.683 | 0.759 |
| clock_sched | 0.641 | 0.708 |
| clock_apex | 0.197 | 0.224 |
| facing | 0.159 | 0.180 |
| progress | 0.022 | 0.027 |
| yaw_progress | 0.007 | 0.007 |
| air_time | 0.013 | 0.001 |
| L1 | -0.199 | -0.177 |
| action_rate | -0.211 | -0.220 |
| foot_idle | -0.119 | -0.060 |
| excess_feet | -0.075 | -0.015 |
| cross_vel | -0.015 | -0.017 |

## Per-hold income (per second) on the MOVING holds

### OLD  (s2=0.1 L1=-1 astd=0.15 sum)

| hold | head | v_along | track_lin_exp | L1 | progress | clock_sched | clock_apex | track_ang_exp | facing | yaw_progress | air_time | cross_vel | action_rate | foot_idle | excess_feet | orientation | lin_vel_z | delta_torque | TRACK | COMMAND | REG | POS | NEG | NET |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fwd | walker(1a) | 0.184 | 0.894 | -0.183 | 0.124 | 0.622 | 0.212 | 0.553 | 0.184 | -0.002 | 0.041 | -0.021 | -0.516 | -1.747 | -0.006 | -0.006 | -0.002 | -0.053 | 0.835 | 2.425 | -2.330 | 2.630 | -2.536 | 0.095 |
| fwd | mid(1b) | 0.039 | 0.698 | -0.251 | 0.051 | 0.685 | 0.246 | 0.595 | 0.184 | -0.002 | 0.033 | -0.031 | -0.619 | -0.427 | -0.010 | -0.003 | -0.002 | -0.044 | 0.498 | 2.210 | -1.106 | 2.493 | -1.389 | 1.104 |
| fwd | creeper(1c) | 0.044 | 0.759 | -0.223 | 0.039 | 0.557 | 0.180 | 0.831 | 0.193 | 0.000 | 0.014 | -0.014 | -0.198 | -0.441 | -0.179 | -0.000 | -0.001 | -0.016 | 0.576 | 2.338 | -0.835 | 2.574 | -1.071 | 1.503 |
| lat_l | walker(1a) | 0.027 | 0.653 | -0.265 | 0.032 | 0.590 | 0.208 | 0.436 | 0.170 | -0.007 | 0.029 | -0.108 | -0.568 | -0.217 | -0.013 | -0.006 | -0.002 | -0.066 | 0.421 | 1.738 | -0.873 | 2.117 | -1.252 | 0.865 |
| lat_l | mid(1b) | 0.074 | 0.786 | -0.215 | 0.060 | 0.622 | 0.177 | 0.659 | 0.185 | 0.000 | 0.027 | -0.058 | -0.589 | -0.031 | -0.023 | -0.003 | -0.002 | -0.085 | 0.631 | 2.244 | -0.732 | 2.517 | -1.005 | 1.512 |
| lat_l | creeper(1c) | 0.005 | 0.698 | -0.242 | 0.014 | 0.686 | 0.272 | 0.861 | 0.195 | 0.000 | 0.023 | -0.039 | -0.211 | -0.046 | -0.107 | -0.000 | -0.002 | -0.012 | 0.471 | 2.470 | -0.378 | 2.750 | -0.659 | 2.092 |
| diag | walker(1a) | 0.074 | 0.696 | -0.255 | 0.064 | 0.564 | 0.217 | 0.345 | 0.154 | -0.005 | 0.027 | -0.081 | -0.569 | -0.368 | -0.014 | -0.008 | -0.002 | -0.066 | 0.505 | 1.726 | -1.027 | 2.068 | -1.369 | 0.699 |
| diag | mid(1b) | 0.077 | 0.733 | -0.236 | 0.065 | 0.586 | 0.209 | 0.573 | 0.183 | -0.001 | 0.022 | -0.051 | -0.608 | -0.076 | -0.026 | -0.007 | -0.002 | -0.099 | 0.562 | 2.083 | -0.819 | 2.372 | -1.108 | 1.264 |
| diag | creeper(1c) | 0.036 | 0.708 | -0.240 | 0.033 | 0.726 | 0.322 | 0.846 | 0.194 | -0.000 | 0.027 | -0.030 | -0.236 | -0.000 | -0.068 | -0.000 | -0.002 | -0.016 | 0.501 | 2.586 | -0.323 | 2.856 | -0.593 | 2.263 |
| back | walker(1a) | 0.086 | 0.739 | -0.245 | 0.082 | 0.565 | 0.233 | 0.274 | 0.138 | -0.010 | 0.027 | -0.059 | -0.574 | -0.259 | -0.012 | -0.010 | -0.002 | -0.067 | 0.576 | 1.744 | -0.923 | 2.057 | -1.237 | 0.820 |
| back | mid(1b) | 0.038 | 0.719 | -0.244 | 0.052 | 0.656 | 0.200 | 0.511 | 0.179 | -0.002 | 0.030 | -0.047 | -0.641 | -0.102 | -0.011 | -0.003 | -0.002 | -0.056 | 0.527 | 2.054 | -0.815 | 2.347 | -1.107 | 1.239 |
| back | creeper(1c) | 0.069 | 0.889 | -0.178 | 0.051 | 0.726 | 0.319 | 0.843 | 0.194 | 0.000 | 0.028 | -0.013 | -0.228 | -0.001 | -0.060 | -0.000 | -0.002 | -0.019 | 0.762 | 2.860 | -0.311 | 3.051 | -0.502 | 2.549 |
| move_turn | walker(1a) | 0.045 | 0.675 | -0.267 | 0.058 | 0.555 | 0.219 | 0.263 | 0.096 | 0.011 | 0.028 | -0.081 | -0.584 | -0.375 | -0.010 | -0.011 | -0.002 | -0.070 | 0.466 | 1.558 | -1.051 | 1.905 | -1.399 | 0.507 |
| move_turn | mid(1b) | 0.021 | 0.615 | -0.300 | 0.064 | 0.533 | 0.231 | 0.220 | 0.054 | 0.005 | 0.025 | -0.076 | -0.720 | -0.243 | -0.016 | -0.010 | -0.003 | -0.082 | 0.379 | 1.371 | -1.074 | 1.747 | -1.450 | 0.297 |
| move_turn | creeper(1c) | -0.005 | 0.604 | -0.315 | 0.060 | 0.518 | 0.194 | 0.186 | 0.055 | 0.016 | 0.019 | -0.046 | -0.289 | -0.330 | -0.087 | -0.005 | -0.002 | -0.034 | 0.349 | 1.290 | -0.747 | 1.651 | -1.108 | 0.543 |

### NEW  (s2=0.04 L1=-2 astd=0.5 product)

| hold | head | v_along | track_lin_exp | L1 | progress | clock_sched | clock_apex | track_ang_exp | facing | yaw_progress | air_time | cross_vel | action_rate | foot_idle | excess_feet | orientation | lin_vel_z | delta_torque | TRACK | COMMAND | REG | POS | NEG | NET |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fwd | walker(1a) | 0.184 | 0.669 | -0.365 | 0.124 | 0.377 | 0.212 | 0.882 | 0.184 | -0.002 | 0.041 | -0.021 | -0.516 | -1.747 | -0.006 | -0.006 | -0.002 | -0.053 | 0.428 | 2.101 | -2.330 | 2.489 | -2.718 | -0.229 |
| fwd | mid(1b) | 0.039 | 0.420 | -0.501 | 0.051 | 0.481 | 0.246 | 0.896 | 0.184 | -0.002 | 0.033 | -0.031 | -0.619 | -0.427 | -0.010 | -0.003 | -0.002 | -0.044 | -0.030 | 1.778 | -1.106 | 2.312 | -1.640 | 0.672 |
| fwd | creeper(1c) | 0.044 | 0.441 | -0.446 | 0.039 | 0.302 | 0.180 | 0.977 | 0.193 | 0.000 | 0.014 | -0.014 | -0.198 | -0.441 | -0.179 | -0.000 | -0.001 | -0.016 | 0.035 | 1.688 | -0.835 | 2.147 | -1.294 | 0.853 |
| lat_l | walker(1a) | 0.027 | 0.338 | -0.530 | 0.032 | 0.350 | 0.208 | 0.805 | 0.170 | -0.007 | 0.029 | -0.108 | -0.568 | -0.217 | -0.013 | -0.006 | -0.002 | -0.066 | -0.159 | 1.287 | -0.873 | 1.932 | -1.517 | 0.415 |
| lat_l | mid(1b) | 0.074 | 0.507 | -0.430 | 0.060 | 0.401 | 0.177 | 0.924 | 0.185 | 0.000 | 0.027 | -0.058 | -0.589 | -0.031 | -0.023 | -0.003 | -0.002 | -0.085 | 0.138 | 1.794 | -0.732 | 2.283 | -1.221 | 1.062 |
| lat_l | creeper(1c) | 0.005 | 0.329 | -0.484 | 0.014 | 0.484 | 0.272 | 0.982 | 0.195 | 0.000 | 0.023 | -0.039 | -0.211 | -0.046 | -0.107 | -0.000 | -0.002 | -0.012 | -0.141 | 1.777 | -0.378 | 2.300 | -0.901 | 1.399 |
| diag | walker(1a) | 0.074 | 0.407 | -0.510 | 0.064 | 0.319 | 0.217 | 0.749 | 0.154 | -0.005 | 0.027 | -0.081 | -0.569 | -0.368 | -0.014 | -0.008 | -0.002 | -0.066 | -0.039 | 1.341 | -1.027 | 1.938 | -1.624 | 0.314 |
| diag | mid(1b) | 0.077 | 0.452 | -0.473 | 0.065 | 0.357 | 0.209 | 0.892 | 0.183 | -0.001 | 0.022 | -0.051 | -0.608 | -0.076 | -0.026 | -0.007 | -0.002 | -0.099 | 0.045 | 1.656 | -0.819 | 2.181 | -1.344 | 0.837 |
| diag | creeper(1c) | 0.036 | 0.362 | -0.480 | 0.033 | 0.534 | 0.322 | 0.981 | 0.194 | -0.000 | 0.027 | -0.030 | -0.236 | -0.000 | -0.068 | -0.000 | -0.002 | -0.016 | -0.085 | 1.942 | -0.323 | 2.452 | -0.833 | 1.619 |
| back | walker(1a) | 0.086 | 0.491 | -0.489 | 0.082 | 0.315 | 0.233 | 0.687 | 0.138 | -0.010 | 0.027 | -0.059 | -0.574 | -0.259 | -0.012 | -0.010 | -0.002 | -0.067 | 0.084 | 1.416 | -0.923 | 1.974 | -1.482 | 0.492 |
| back | mid(1b) | 0.038 | 0.439 | -0.487 | 0.052 | 0.442 | 0.200 | 0.854 | 0.179 | -0.002 | 0.030 | -0.047 | -0.641 | -0.102 | -0.011 | -0.003 | -0.002 | -0.056 | 0.003 | 1.659 | -0.815 | 2.195 | -1.351 | 0.844 |
| back | creeper(1c) | 0.069 | 0.611 | -0.357 | 0.051 | 0.533 | 0.319 | 0.979 | 0.194 | 0.000 | 0.028 | -0.013 | -0.228 | -0.001 | -0.060 | -0.000 | -0.002 | -0.019 | 0.305 | 2.347 | -0.311 | 2.717 | -0.680 | 2.036 |
| move_turn | walker(1a) | 0.045 | 0.396 | -0.534 | 0.058 | 0.311 | 0.219 | 0.686 | 0.096 | 0.011 | 0.028 | -0.081 | -0.584 | -0.375 | -0.010 | -0.011 | -0.002 | -0.070 | -0.080 | 1.191 | -1.051 | 1.805 | -1.666 | 0.139 |
| move_turn | mid(1b) | 0.021 | 0.370 | -0.600 | 0.064 | 0.287 | 0.231 | 0.638 | 0.054 | 0.005 | 0.025 | -0.076 | -0.720 | -0.243 | -0.016 | -0.010 | -0.003 | -0.082 | -0.166 | 0.998 | -1.074 | 1.674 | -1.750 | -0.076 |
| move_turn | creeper(1c) | -0.005 | 0.367 | -0.630 | 0.060 | 0.278 | 0.194 | 0.679 | 0.055 | 0.016 | 0.019 | -0.046 | -0.289 | -0.330 | -0.087 | -0.005 | -0.002 | -0.034 | -0.203 | 0.991 | -0.747 | 1.667 | -1.423 | 0.244 |

## GATE -- walker vs creeper margin per moving hold

margin = (walker - creeper) / walker POSITIVE income, on three nested subtotals:
  NET     = every replayed term
  COMMAND = every term that prices the command (tracking + clock + ang + facing + yaw + air-time + cross-vel); excludes the gait/action regularizers, which the fix does not touch
  TRACK   = track_lin_exp + L1 + progress, the sub-economy the fix moves directly
target: >= +25 % on every moving hold

| hold | economy | walker v_along | creeper v_along | TRACK margin | COMMAND margin | NET margin | creeper/walker NET |
|---|---|---|---|---|---|---|---|
| fwd | OLD | 0.184 | 0.044 | +9.9 % | +3.3 % | -53.5 % | 1587 % |
| fwd | NEW | 0.184 | 0.044 | +15.8 % | +16.6 % | -43.5 % | -373 % |
| lat_l | OLD | 0.027 | 0.005 | -2.4 % | -34.6 % | -57.9 % | 242 % |
| lat_l | NEW | 0.027 | 0.005 | -1.0 % | -25.3 % | -51.0 % | 337 % |
| diag | OLD | 0.074 | 0.036 | +0.2 % | -41.6 % | -75.6 % | 324 % |
| diag | NEW | 0.074 | 0.036 | +2.4 % | -31.0 % | -67.3 % | 515 % |
| back | OLD | 0.086 | 0.069 | -9.0 % | -54.3 % | -84.0 % | 311 % |
| back | NEW | 0.086 | 0.069 | -11.2 % | -47.2 % | -78.2 % | 414 % |
| move_turn | OLD | 0.045 | -0.005 | +6.2 % | +14.0 % | -1.9 % | 107 % |
| move_turn | NEW | 0.045 | -0.005 | +6.8 % | +11.0 % | -5.8 % | 175 % |

## Sweep -- COMMAND margin (% of walker positive income) per moving hold

| sigma^2 | L1 | combine | ang_std | fwd | lat_l | diag | back | move_turn | min | fwd>=25% |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.04 | -1.5 | sum | 0.5 | +14.0 | -20.3 | -24.8 | -37.9 | +8.7 | -37.9 | no |
| 0.04 | -1.5 | product | 0.5 | +15.8 | -24.7 | -30.6 | -45.5 | +9.7 | -45.5 | no |
| 0.04 | -2 | sum | 0.5 | +14.8 | -20.8 | -25.1 | -39.3 | +9.9 | -39.3 | no |
| 0.04 | -2 | product | 0.5 | +16.6 | -25.3 | -31.0 | -47.2 | +11.0 | -47.2 | no |
| 0.04 | -3 | sum | 0.5 | +16.2 | -21.8 | -25.8 | -42.3 | +12.3 | -42.3 | no |
| 0.04 | -3 | product | 0.5 | +18.2 | -26.5 | -31.8 | -50.6 | +13.7 | -50.6 | no |
| 0.06 | -1.5 | sum | 0.5 | +12.2 | -20.3 | -24.5 | -37.0 | +9.2 | -37.0 | no |
| 0.06 | -1.5 | product | 0.5 | +13.7 | -24.5 | -30.0 | -44.1 | +10.2 | -44.1 | no |
| 0.06 | -2 | sum | 0.5 | +12.9 | -20.8 | -24.9 | -38.4 | +10.3 | -38.4 | no |
| 0.06 | -2 | product | 0.5 | +14.5 | -25.0 | -30.4 | -45.7 | +11.4 | -45.7 | no |
| 0.06 | -3 | sum | 0.5 | +14.4 | -21.8 | -25.5 | -41.2 | +12.6 | -41.2 | no |
| 0.06 | -3 | product | 0.5 | +16.1 | -26.1 | -31.1 | -48.9 | +13.9 | -48.9 | no |

best swept point by worst-hold COMMAND margin: s2=0.06 L1=-1.5 astd=0.5 sum -> -37.0 %

## Sweep -- NET margin (% of walker positive income) per moving hold

| sigma^2 | L1 | combine | ang_std | fwd | lat_l | diag | back | move_turn | min | fwd>=25% |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.04 | -1.5 | sum | 0.5 | -40.7 | -43.1 | -57.0 | -65.4 | -6.1 | -65.4 | no |
| 0.04 | -1.5 | product | 0.5 | -44.3 | -50.4 | -66.9 | -76.6 | -7.2 | -76.6 | no |
| 0.04 | -2 | sum | 0.5 | -39.9 | -43.6 | -57.3 | -66.9 | -4.9 | -66.9 | no |
| 0.04 | -2 | product | 0.5 | -43.5 | -51.0 | -67.3 | -78.2 | -5.8 | -78.2 | no |
| 0.04 | -3 | sum | 0.5 | -38.4 | -44.6 | -58.0 | -69.9 | -2.6 | -69.9 | no |
| 0.04 | -3 | product | 0.5 | -41.8 | -52.1 | -68.1 | -81.6 | -3.1 | -81.6 | no |
| 0.06 | -1.5 | sum | 0.5 | -40.4 | -41.8 | -55.0 | -63.2 | -4.8 | -63.2 | no |
| 0.06 | -1.5 | product | 0.5 | -43.9 | -48.4 | -64.1 | -73.6 | -5.6 | -73.6 | no |
| 0.06 | -2 | sum | 0.5 | -39.7 | -42.2 | -55.3 | -64.7 | -3.7 | -64.7 | no |
| 0.06 | -2 | product | 0.5 | -43.1 | -49.0 | -64.5 | -75.1 | -4.4 | -75.1 | no |
| 0.06 | -3 | sum | 0.5 | -38.3 | -43.2 | -56.0 | -67.5 | -1.5 | -67.5 | no |
| 0.06 | -3 | product | 0.5 | -41.6 | -50.1 | -65.2 | -78.3 | -1.9 | -78.3 | no |

best swept point by worst-hold NET margin: s2=0.06 L1=-1.5 astd=0.5 sum -> -63.2 %

## Motion-requiring income (progress + yaw_progress + air_time) as a share of POSITIVE income

The diagnosis's headline number: under 1c's terms only 1.2 % of the creeper's income required moving. If the fix does not move this, standing stays affordable.

| hold | head | OLD | NEW |
|---|---|---|---|
| fwd | walker(1a) | 6.2 % | 6.6 % |
| fwd | mid(1b) | 3.3 % | 3.6 % |
| fwd | creeper(1c) | 2.1 % | 2.5 % |
| lat_l | walker(1a) | 2.6 % | 2.8 % |
| lat_l | mid(1b) | 3.5 % | 3.8 % |
| lat_l | creeper(1c) | 1.4 % | 1.6 % |
| diag | walker(1a) | 4.2 % | 4.4 % |
| diag | mid(1b) | 3.6 % | 3.9 % |
| diag | creeper(1c) | 2.1 % | 2.4 % |
| back | walker(1a) | 4.8 % | 5.0 % |
| back | mid(1b) | 3.4 % | 3.6 % |
| back | creeper(1c) | 2.6 % | 2.9 % |
| move_turn | walker(1a) | 5.1 % | 5.4 % |
| move_turn | mid(1b) | 5.4 % | 5.6 % |
| move_turn | creeper(1c) | 5.7 % | 5.7 % |

## PROBE -- two levers beyond the decided four (fwd hold, COMMAND margin)

Not part of the 2026-09-18 decision; both are fix candidates the diagnosis already listed. `ang@turn` zeroes track_ang_vel_z_exp on rows with cmd_wz == 0 (it pays a motionless robot ~1.0/s on the 88 % of rows that command no turn); `prog=` raises reward_forward_progress_along_command's weight from 0.6.

| terms | walker COMMAND | creeper COMMAND | margin | creeper motion-req. share |
|---|---|---|---|---|
| NEW (decided four) | 2.101 | 1.688 | +16.6 % | 2.5 % |
| NEW + ang@turn | 1.375 | 0.715 | +37.5 % | 4.5 % |
| NEW + prog 1.5 | 2.287 | 1.747 | +20.2 % | 5.1 % |
| NEW + prog 2.5 | 2.494 | 1.812 | +23.7 % | 7.8 % |
| NEW + ang@turn + prog 1.5 | 1.561 | 0.773 | +40.4 % | 9.1 % |
| NEW + ang@turn + prog 2.5 | 1.767 | 0.839 | +43.1 % | 13.7 % |

## Isolating the four knobs (fwd hold, COMMAND margin walker - creeper, % of walker POS)

| step | terms | walker COMMAND | creeper COMMAND | margin |
|---|---|---|---|---|
| OLD | s2=0.1 L1=-1 astd=0.15 sum | 2.425 | 2.338 | +3.3 % |
| + sigma^2 0.04 | s2=0.04 L1=-1 astd=0.15 sum | 2.200 | 2.020 | +7.5 % |
| + L1 -2.0 | s2=0.04 L1=-2 astd=0.15 sum | 2.017 | 1.797 | +9.2 % |
| + ang_std 0.5 | s2=0.04 L1=-2 astd=0.5 sum | 2.347 | 1.943 | +14.8 % |
| + combine product (= NEW) | s2=0.04 L1=-2 astd=0.5 product | 2.101 | 1.688 | +16.6 % |

## Tracking sub-economy vs achieved along-command speed (analytic, cmd 0.25 m/s)

income/s of track_lin_exp + L1 + progress at perfect direction; the last column is the slope at v = 0, i.e. the gradient PPO sees from a standstill.

| economy | v=0 | v=0.05 | v=0.1 | v=0.15 | v=0.2 | v=0.25 | d/dv at 0 |
|---|---|---|---|---|---|---|---|
| OLD (s2=0.1 L1=-1 astd=0.15 sum) | +0.419 | +0.668 | +0.908 | +1.121 | +1.289 | +1.400 | +4.96 |
| NEW (s2=0.04 L1=-2 astd=0.5 product) | -0.238 | +0.090 | +0.472 | +0.864 | +1.194 | +1.400 | +6.01 |

best swept point by worst-hold margin: s2=0.06 L1=-1.5 astd=0.5 sum -> -63.2 %

## Zero-velocity check -- the command-tracking sub-economy under a move command

terms: track_lin_vel_xy_exp + penalty_tracking_error_l1 + forward_progress (the three terms whose value depends on the achieved planar velocity; on a pure-translation row cmd_wz = 0 so track_ang / yaw_progress are constants).

| cmd | economy | v=0 net | v=cmd net | v=0 sign |
|---|---|---|---|---|
| 0.22 m/s | OLD | +0.550 | +1.382 | POSITIVE |
| 0.22 m/s | NEW | -0.067 | +1.382 | NEGATIVE |
| 0.25 m/s | OLD | +0.419 | +1.400 | POSITIVE |
| 0.25 m/s | NEW | -0.238 | +1.400 | NEGATIVE |

sweep of the same check (v=0 net; must be negative):

| sigma^2 | L1 | cmd 0.22 | cmd 0.25 |
|---|---|---|---|
| 0.04 | -1.5 | +0.043 | -0.113 |
| 0.04 | -2 | -0.067 | -0.238 |
| 0.04 | -3 | -0.287 | -0.488 |
| 0.04 | -1 | +0.153 | +0.012 |
| 0.06 | -1.5 | +0.228 | +0.066 |
| 0.06 | -2 | +0.118 | -0.059 |
| 0.06 | -3 | -0.102 | -0.309 |
| 0.06 | -1 | +0.338 | +0.191 |
| 0.1 | -1.5 | +0.440 | +0.294 |
| 0.1 | -2 | +0.330 | +0.169 |
| 0.1 | -3 | +0.110 | -0.081 |
| 0.1 | -1 | +0.550 | +0.419 |

## Reference: the stand hold (commanded stop) under both economies

| head | economy | POS | NEG | NET |
|---|---|---|---|---|
| walker(1a) | OLD | 2.105 | -1.021 | 1.084 |
| mid(1b) | OLD | 2.277 | -0.873 | 1.404 |
| creeper(1c) | OLD | 2.975 | -0.168 | 2.807 |
| walker(1a) | NEW | 2.091 | -1.215 | 0.876 |
| mid(1b) | NEW | 2.282 | -1.022 | 1.260 |
| creeper(1c) | NEW | 2.957 | -0.240 | 2.717 |

## VERDICT

holds that actually test walking-vs-creeping (1a speed ratio >= 0.3 AND 1a out-travels 1c by >= 2x): ['fwd'] -- on every other moving hold the '1a walker' creeps too, so the margin there compares two creepers and the sign carries no information about whether walking pays.

- G1 COMMAND margin >= 25 % on every hold with a genuine walker: FAIL (fwd +16.6 %)
- G2 COMMAND margin >= 25 % on EVERY moving hold: FAIL (fwd +16.6 %, lat_l -25.3 %, diag -31.0 %, back -47.2 %, move_turn +11.0 %)
- G3 zero-velocity tracking income negative at both commanded speeds: PASS (0.22 -> -0.067, 0.25 -> -0.238)

GATE: FAIL (on the G1 + G3 reading; G2 is not decidable from these raws -- see the walkable-hold line above)

### Candidate term sets on the decidable hold (fwd)

| set | terms | COMMAND margin | G1 | v=0 @0.22 | v=0 @0.25 | G3 |
|---|---|---|---|---|---|---|
| OLD (1c as run) | s2=0.1 L1=-1 astd=0.15 sum | +3.3 % | FAIL | +0.550 | +0.419 | FAIL |
| D: decided four | s2=0.04 L1=-2 astd=0.5 product | +16.6 % | FAIL | -0.067 | -0.238 | PASS |
| D + ang@turn | s2=0.04 L1=-2 astd=0.5 product ang@turn | +37.5 % | PASS | -0.067 | -0.238 | PASS |
| D + ang@turn + prog 1.5 | s2=0.04 L1=-2 astd=0.5 product ang@turn prog=1.5 | +40.4 % | PASS | -0.067 | -0.238 | PASS |
| D + ang@turn + prog 2.5 | s2=0.04 L1=-2 astd=0.5 product ang@turn prog=2.5 | +43.1 % | PASS | -0.067 | -0.238 | PASS |
| D with L1 -1.5 | s2=0.04 L1=-1.5 astd=0.5 product | +15.8 % | FAIL | +0.043 | -0.113 | FAIL |
| D with L1 -3.0 | s2=0.04 L1=-3 astd=0.5 product | +18.2 % | FAIL | -0.287 | -0.488 | PASS |
| D with sigma^2 0.06 | s2=0.06 L1=-2 astd=0.5 product | +14.5 % | FAIL | +0.118 | -0.059 | FAIL |
| D with combine sum | s2=0.04 L1=-2 astd=0.5 sum | +14.8 % | FAIL | -0.067 | -0.238 | PASS |

