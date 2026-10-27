# Offline reward-economy replay -- omni 1c fix (2026-09-18)

episodes: walker(1a) n=100, mid(1b) n=100, creeper(1c) n=100
clock-phase fit: 36 offsets/episode, argmax of OLD clock income (schedule[sum]+apex)
step mask: steady_mask & segment & pre-termination
weights: 1c training-log reward table; income/s = mean_t(weight * value) (Isaac Lab accumulates weight*value*dt)
not replayed: reward_stride_length (w 0.5; logged income 0.000/s in every 1c window -- the term is identically 0 for these heads), penalty_tibia_deviation_in_stance (w -0.28; logged -0.042/s -- needs default_joint_pos, not in the raw), and the 18 terms at weight 0.

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
| fwd | walker(1a) | 0.152 | 0.954 | -0.157 | 0.100 | 0.626 | 0.200 | 0.424 | 0.178 | 0.003 | 0.030 | -0.028 | -1.137 | -0.099 | -0.007 | -0.009 | -0.002 | -0.080 | 0.896 | 2.328 | -1.335 | 2.514 | -1.521 | 0.993 |
| fwd | mid(1b) | 0.039 | 0.698 | -0.251 | 0.051 | 0.685 | 0.246 | 0.595 | 0.184 | -0.002 | 0.033 | -0.031 | -0.619 | -0.427 | -0.010 | -0.003 | -0.002 | -0.044 | 0.498 | 2.210 | -1.106 | 2.493 | -1.389 | 1.104 |
| fwd | creeper(1c) | 0.044 | 0.759 | -0.223 | 0.039 | 0.557 | 0.180 | 0.831 | 0.193 | 0.000 | 0.014 | -0.014 | -0.198 | -0.441 | -0.179 | -0.000 | -0.001 | -0.016 | 0.576 | 2.338 | -0.835 | 2.574 | -1.071 | 1.503 |
| lat_l | walker(1a) | 0.117 | 0.938 | -0.160 | 0.079 | 0.533 | 0.185 | 0.456 | 0.182 | -0.001 | 0.024 | -0.030 | -1.063 | -0.049 | -0.029 | -0.004 | -0.002 | -0.121 | 0.856 | 2.205 | -1.268 | 2.397 | -1.460 | 0.937 |
| lat_l | mid(1b) | 0.074 | 0.786 | -0.215 | 0.060 | 0.622 | 0.177 | 0.659 | 0.185 | 0.000 | 0.027 | -0.058 | -0.589 | -0.031 | -0.023 | -0.003 | -0.002 | -0.085 | 0.631 | 2.244 | -0.732 | 2.517 | -1.005 | 1.512 |
| lat_l | creeper(1c) | 0.005 | 0.698 | -0.242 | 0.014 | 0.686 | 0.272 | 0.861 | 0.195 | 0.000 | 0.023 | -0.039 | -0.211 | -0.046 | -0.107 | -0.000 | -0.002 | -0.012 | 0.471 | 2.470 | -0.378 | 2.750 | -0.659 | 2.092 |
| diag | walker(1a) | 0.141 | 0.887 | -0.182 | 0.096 | 0.553 | 0.211 | 0.343 | 0.172 | -0.007 | 0.021 | -0.037 | -1.091 | -0.046 | -0.026 | -0.011 | -0.002 | -0.116 | 0.800 | 2.056 | -1.291 | 2.282 | -1.517 | 0.765 |
| diag | mid(1b) | 0.077 | 0.733 | -0.236 | 0.065 | 0.586 | 0.209 | 0.573 | 0.183 | -0.001 | 0.022 | -0.051 | -0.608 | -0.076 | -0.026 | -0.007 | -0.002 | -0.099 | 0.562 | 2.083 | -0.819 | 2.372 | -1.108 | 1.264 |
| diag | creeper(1c) | 0.036 | 0.708 | -0.240 | 0.033 | 0.726 | 0.322 | 0.846 | 0.194 | -0.000 | 0.027 | -0.030 | -0.236 | -0.000 | -0.068 | -0.000 | -0.002 | -0.016 | 0.501 | 2.586 | -0.323 | 2.856 | -0.593 | 2.263 |
| back | walker(1a) | 0.133 | 0.999 | -0.140 | 0.085 | 0.601 | 0.198 | 0.424 | 0.178 | 0.006 | 0.028 | -0.024 | -1.171 | -0.115 | -0.010 | -0.006 | -0.002 | -0.083 | 0.944 | 2.355 | -1.388 | 2.519 | -1.551 | 0.968 |
| back | mid(1b) | 0.038 | 0.719 | -0.244 | 0.052 | 0.656 | 0.200 | 0.511 | 0.179 | -0.002 | 0.030 | -0.047 | -0.641 | -0.102 | -0.011 | -0.003 | -0.002 | -0.056 | 0.527 | 2.054 | -0.815 | 2.347 | -1.107 | 1.239 |
| back | creeper(1c) | 0.069 | 0.889 | -0.178 | 0.051 | 0.726 | 0.319 | 0.843 | 0.194 | 0.000 | 0.028 | -0.013 | -0.228 | -0.001 | -0.060 | -0.000 | -0.002 | -0.019 | 0.762 | 2.860 | -0.311 | 3.051 | -0.502 | 2.549 |
| move_turn | walker(1a) | 0.098 | 0.930 | -0.167 | 0.071 | 0.613 | 0.205 | 0.194 | 0.055 | 0.020 | 0.026 | -0.026 | -1.216 | -0.180 | -0.010 | -0.012 | -0.002 | -0.079 | 0.834 | 1.920 | -1.499 | 2.114 | -1.693 | 0.421 |
| move_turn | mid(1b) | 0.021 | 0.615 | -0.300 | 0.064 | 0.533 | 0.231 | 0.220 | 0.054 | 0.005 | 0.025 | -0.076 | -0.720 | -0.243 | -0.016 | -0.010 | -0.003 | -0.082 | 0.379 | 1.371 | -1.074 | 1.747 | -1.450 | 0.297 |
| move_turn | creeper(1c) | -0.005 | 0.604 | -0.315 | 0.060 | 0.518 | 0.194 | 0.186 | 0.055 | 0.016 | 0.019 | -0.046 | -0.289 | -0.330 | -0.087 | -0.005 | -0.002 | -0.034 | 0.349 | 1.290 | -0.747 | 1.651 | -1.108 | 0.543 |

### NEW  (s2=0.04 L1=-2 astd=0.5 product yaw_prog_w=3)

| hold | head | v_along | track_lin_exp | L1 | progress | clock_sched | clock_apex | track_ang_exp | facing | yaw_progress | air_time | cross_vel | action_rate | foot_idle | excess_feet | orientation | lin_vel_z | delta_torque | TRACK | COMMAND | REG | POS | NEG | NET |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fwd | walker(1a) | 0.152 | 0.739 | -0.315 | 0.100 | 0.375 | 0.200 | 0.824 | 0.178 | 0.010 | 0.030 | -0.028 | -1.137 | -0.099 | -0.007 | -0.009 | -0.002 | -0.080 | 0.524 | 2.113 | -1.335 | 2.456 | -1.679 | 0.778 |
| fwd | mid(1b) | 0.039 | 0.420 | -0.501 | 0.051 | 0.481 | 0.246 | 0.896 | 0.184 | -0.005 | 0.033 | -0.031 | -0.619 | -0.427 | -0.010 | -0.003 | -0.002 | -0.044 | -0.030 | 1.775 | -1.106 | 2.312 | -1.643 | 0.669 |
| fwd | creeper(1c) | 0.044 | 0.441 | -0.446 | 0.039 | 0.302 | 0.180 | 0.977 | 0.193 | 0.000 | 0.014 | -0.014 | -0.198 | -0.441 | -0.179 | -0.000 | -0.001 | -0.016 | 0.035 | 1.688 | -0.835 | 2.147 | -1.294 | 0.853 |
| lat_l | walker(1a) | 0.117 | 0.708 | -0.321 | 0.079 | 0.282 | 0.185 | 0.851 | 0.182 | -0.003 | 0.024 | -0.030 | -1.063 | -0.049 | -0.029 | -0.004 | -0.002 | -0.121 | 0.465 | 1.957 | -1.268 | 2.311 | -1.622 | 0.689 |
| lat_l | mid(1b) | 0.074 | 0.507 | -0.430 | 0.060 | 0.401 | 0.177 | 0.924 | 0.185 | 0.000 | 0.027 | -0.058 | -0.589 | -0.031 | -0.023 | -0.003 | -0.002 | -0.085 | 0.138 | 1.794 | -0.732 | 2.283 | -1.221 | 1.062 |
| lat_l | creeper(1c) | 0.005 | 0.329 | -0.484 | 0.014 | 0.484 | 0.272 | 0.982 | 0.195 | 0.000 | 0.023 | -0.039 | -0.211 | -0.046 | -0.107 | -0.000 | -0.002 | -0.012 | -0.141 | 1.777 | -0.378 | 2.300 | -0.901 | 1.399 |
| diag | walker(1a) | 0.141 | 0.649 | -0.364 | 0.096 | 0.304 | 0.211 | 0.779 | 0.172 | -0.022 | 0.021 | -0.037 | -1.091 | -0.046 | -0.026 | -0.011 | -0.002 | -0.116 | 0.380 | 1.809 | -1.291 | 2.232 | -1.714 | 0.518 |
| diag | mid(1b) | 0.077 | 0.452 | -0.473 | 0.065 | 0.357 | 0.209 | 0.892 | 0.183 | -0.004 | 0.022 | -0.051 | -0.608 | -0.076 | -0.026 | -0.007 | -0.002 | -0.099 | 0.045 | 1.653 | -0.819 | 2.181 | -1.347 | 0.834 |
| diag | creeper(1c) | 0.036 | 0.362 | -0.480 | 0.033 | 0.534 | 0.322 | 0.981 | 0.194 | -0.000 | 0.027 | -0.030 | -0.236 | -0.000 | -0.068 | -0.000 | -0.002 | -0.016 | -0.085 | 1.942 | -0.323 | 2.452 | -0.833 | 1.619 |
| back | walker(1a) | 0.133 | 0.792 | -0.279 | 0.085 | 0.364 | 0.198 | 0.826 | 0.178 | 0.019 | 0.028 | -0.024 | -1.171 | -0.115 | -0.010 | -0.006 | -0.002 | -0.083 | 0.597 | 2.186 | -1.388 | 2.489 | -1.691 | 0.798 |
| back | mid(1b) | 0.038 | 0.439 | -0.487 | 0.052 | 0.442 | 0.200 | 0.854 | 0.179 | -0.006 | 0.030 | -0.047 | -0.641 | -0.102 | -0.011 | -0.003 | -0.002 | -0.056 | 0.003 | 1.655 | -0.815 | 2.195 | -1.355 | 0.840 |
| back | creeper(1c) | 0.069 | 0.611 | -0.357 | 0.051 | 0.533 | 0.319 | 0.979 | 0.194 | 0.000 | 0.028 | -0.013 | -0.228 | -0.001 | -0.060 | -0.000 | -0.002 | -0.019 | 0.305 | 2.347 | -0.311 | 2.717 | -0.680 | 2.036 |
| move_turn | walker(1a) | 0.098 | 0.701 | -0.335 | 0.071 | 0.376 | 0.205 | 0.697 | 0.055 | 0.059 | 0.026 | -0.026 | -1.216 | -0.180 | -0.010 | -0.012 | -0.002 | -0.079 | 0.438 | 1.830 | -1.499 | 2.191 | -1.860 | 0.331 |
| move_turn | mid(1b) | 0.021 | 0.370 | -0.600 | 0.064 | 0.287 | 0.231 | 0.638 | 0.054 | 0.016 | 0.025 | -0.076 | -0.720 | -0.243 | -0.016 | -0.010 | -0.003 | -0.082 | -0.166 | 1.009 | -1.074 | 1.685 | -1.750 | -0.065 |
| move_turn | creeper(1c) | -0.005 | 0.367 | -0.630 | 0.060 | 0.278 | 0.194 | 0.679 | 0.055 | 0.047 | 0.019 | -0.046 | -0.289 | -0.330 | -0.087 | -0.005 | -0.002 | -0.034 | -0.203 | 1.022 | -0.747 | 1.699 | -1.423 | 0.275 |

## Turn-hold income ordering -- the terms the A5 levers move, on turn_l90/turn_r90/move_turn

Candidates C2 (--yaw-progress-w), C4 (--facing-turn-w), C6 (--coupled-sigma) and C9 (--progress-w / --new-progress-w, one flag under two spellings) are priced here on the holds where a turn is commanded, OLD vs the armed NEW variant, per head. Any one of them SUPPLIED arms this table (the flag, not the value), and C9's weight is what pays the `progress` column. YAW = the four facing/turn-rate terms (what C2 and C4 move); TRACK = the translation sub-economy (what C6 rescales through track_lin_exp and what the NEW variant's progress weight pays through progress); the two clock columns are context, in neither subtotal. COMMAND is the row's full command subtotal -- the same one the GATE scores -- so it also contains air_time and cross_vel, which are NOT columns here: a row's COMMAND is the sum of its printed columns only on the two in-place turn holds, where both of those terms are identically 0; on move_turn they are not. 'order' ranks the four YAW terms by income, descending -- which term dominates the hold's turning economy.

| hold | head | variant | track_ang_exp | yaw_progress | facing | facing_turn | YAW | track_lin_exp | L1 | progress | TRACK | clock_sched | clock_apex | COMMAND | order (desc, YAW) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| turn_l90 | walker(1a) | OLD | 0.270 | 0.067 | 0.072 | 0.000 | 0.409 | 1.056 | -0.118 | 0.000 | 0.939 | 0.647 | 0.225 | 2.219 | track_ang_exp=+0.270, facing=+0.072, yaw_progress=+0.067, facing_turn=+0.000 |
| turn_l90 | walker(1a) | NEW | 0.760 | 0.200 | 0.072 | 0.000 | 1.032 | 0.877 | -0.235 | 0.000 | 0.642 | 0.427 | 0.225 | 2.326 | track_ang_exp=+0.760, yaw_progress=+0.200, facing=+0.072, facing_turn=+0.000 |
| turn_l90 | mid(1b) | OLD | 0.227 | 0.011 | 0.054 | 0.000 | 0.293 | 0.708 | -0.257 | 0.000 | 0.451 | 0.557 | 0.251 | 1.552 | track_ang_exp=+0.227, facing=+0.054, yaw_progress=+0.011, facing_turn=+0.000 |
| turn_l90 | mid(1b) | NEW | 0.635 | 0.034 | 0.054 | 0.000 | 0.723 | 0.458 | -0.515 | 0.000 | -0.057 | 0.311 | 0.251 | 1.229 | track_ang_exp=+0.635, facing=+0.054, yaw_progress=+0.034, facing_turn=+0.000 |
| turn_l90 | creeper(1c) | OLD | 0.207 | 0.024 | 0.055 | 0.000 | 0.286 | 0.755 | -0.256 | 0.000 | 0.498 | 0.542 | 0.210 | 1.536 | track_ang_exp=+0.207, facing=+0.055, yaw_progress=+0.024, facing_turn=+0.000 |
| turn_l90 | creeper(1c) | NEW | 0.689 | 0.071 | 0.055 | 0.000 | 0.815 | 0.543 | -0.513 | 0.000 | 0.030 | 0.310 | 0.210 | 1.366 | track_ang_exp=+0.689, yaw_progress=+0.071, facing=+0.055, facing_turn=+0.000 |
| turn_r90 | walker(1a) | OLD | 0.355 | 0.052 | 0.143 | 0.000 | 0.549 | 1.083 | -0.107 | 0.000 | 0.976 | 0.554 | 0.172 | 2.251 | track_ang_exp=+0.355, facing=+0.143, yaw_progress=+0.052, facing_turn=+0.000 |
| turn_r90 | walker(1a) | NEW | 0.804 | 0.155 | 0.143 | 0.000 | 1.102 | 0.923 | -0.214 | 0.000 | 0.709 | 0.390 | 0.172 | 2.372 | track_ang_exp=+0.804, yaw_progress=+0.155, facing=+0.143, facing_turn=+0.000 |
| turn_r90 | mid(1b) | OLD | 0.414 | 0.008 | 0.149 | 0.000 | 0.571 | 0.848 | -0.201 | 0.000 | 0.647 | 0.497 | 0.156 | 1.871 | track_ang_exp=+0.414, facing=+0.149, yaw_progress=+0.008, facing_turn=+0.000 |
| turn_r90 | mid(1b) | NEW | 0.775 | 0.024 | 0.149 | 0.000 | 0.948 | 0.605 | -0.401 | 0.000 | 0.204 | 0.360 | 0.156 | 1.668 | track_ang_exp=+0.775, facing=+0.149, yaw_progress=+0.024, facing_turn=+0.000 |
| turn_r90 | creeper(1c) | OLD | 0.674 | 0.015 | 0.175 | 0.000 | 0.864 | 1.103 | -0.094 | 0.000 | 1.009 | 0.624 | 0.055 | 2.552 | track_ang_exp=+0.674, facing=+0.175, yaw_progress=+0.015, facing_turn=+0.000 |
| turn_r90 | creeper(1c) | NEW | 0.918 | 0.044 | 0.175 | 0.000 | 1.137 | 0.986 | -0.188 | 0.000 | 0.798 | 0.541 | 0.055 | 2.531 | track_ang_exp=+0.918, facing=+0.175, yaw_progress=+0.044, facing_turn=+0.000 |
| move_turn | walker(1a) | OLD | 0.194 | 0.020 | 0.055 | 0.000 | 0.269 | 0.930 | -0.167 | 0.071 | 0.834 | 0.613 | 0.205 | 1.920 | track_ang_exp=+0.194, facing=+0.055, yaw_progress=+0.020, facing_turn=+0.000 |
| move_turn | walker(1a) | NEW | 0.697 | 0.059 | 0.055 | 0.000 | 0.812 | 0.701 | -0.335 | 0.071 | 0.438 | 0.376 | 0.205 | 1.830 | track_ang_exp=+0.697, yaw_progress=+0.059, facing=+0.055, facing_turn=+0.000 |
| move_turn | mid(1b) | OLD | 0.220 | 0.005 | 0.054 | 0.000 | 0.279 | 0.615 | -0.300 | 0.064 | 0.379 | 0.533 | 0.231 | 1.371 | track_ang_exp=+0.220, facing=+0.054, yaw_progress=+0.005, facing_turn=+0.000 |
| move_turn | mid(1b) | NEW | 0.638 | 0.016 | 0.054 | 0.000 | 0.708 | 0.370 | -0.600 | 0.064 | -0.166 | 0.287 | 0.231 | 1.009 | track_ang_exp=+0.638, facing=+0.054, yaw_progress=+0.016, facing_turn=+0.000 |
| move_turn | creeper(1c) | OLD | 0.186 | 0.016 | 0.055 | 0.000 | 0.257 | 0.604 | -0.315 | 0.060 | 0.349 | 0.518 | 0.194 | 1.290 | track_ang_exp=+0.186, facing=+0.055, yaw_progress=+0.016, facing_turn=+0.000 |
| move_turn | creeper(1c) | NEW | 0.679 | 0.047 | 0.055 | 0.000 | 0.781 | 0.367 | -0.630 | 0.060 | -0.203 | 0.278 | 0.194 | 1.022 | track_ang_exp=+0.679, facing=+0.055, yaw_progress=+0.047, facing_turn=+0.000 |

## GATE -- walker vs creeper margin per moving hold

margin = (walker - creeper) / walker POSITIVE income, on three nested subtotals:
  NET     = every replayed term
  COMMAND = every term that prices the command (tracking + clock + ang + facing + yaw + air-time + cross-vel); excludes the gait/action regularizers, which the fix does not touch
  TRACK   = track_lin_exp + L1 + progress, the sub-economy the fix moves directly
target: >= +25 % on every moving hold

| hold | economy | walker v_along | creeper v_along | TRACK margin | COMMAND margin | NET margin | creeper/walker NET |
|---|---|---|---|---|---|---|---|
| fwd | OLD | 0.152 | 0.044 | +12.7 % | -0.4 % | -20.3 % | 151 % |
| fwd | NEW | 0.152 | 0.044 | +19.9 % | +17.3 % | -3.1 % | 110 % |
| lat_l | OLD | 0.117 | 0.005 | +16.1 % | -11.0 % | -48.2 % | 223 % |
| lat_l | NEW | 0.117 | 0.005 | +26.2 % | +7.8 % | -30.7 % | 203 % |
| diag | OLD | 0.141 | 0.036 | +13.1 % | -23.2 % | -65.6 % | 296 % |
| diag | NEW | 0.141 | 0.036 | +20.9 % | -5.9 % | -49.3 % | 313 % |
| back | OLD | 0.133 | 0.069 | +7.2 % | -20.0 % | -62.8 % | 264 % |
| back | NEW | 0.133 | 0.069 | +11.7 % | -6.5 % | -49.7 % | 255 % |
| move_turn | OLD | 0.098 | -0.005 | +22.9 % | +29.8 % | -5.8 % | 129 % |
| move_turn | NEW | 0.098 | -0.005 | +29.2 % | +36.9 % | +2.5 % | 83 % |

## Sweep -- COMMAND margin (% of walker positive income) per moving hold

| sigma^2 | L1 | combine | ang_std | fwd | lat_l | diag | back | move_turn | min | fwd>=25% |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.04 | -1.5 | sum | 0.5 | +14.1 | +7.4 | -3.7 | -5.5 | +30.2 | -5.5 | no |
| 0.04 | -1.5 | product | 0.5 | +15.7 | +6.1 | -6.6 | -7.8 | +33.7 | -7.8 | no |
| 0.04 | -2 | sum | 0.5 | +15.3 | +9.0 | -2.5 | -4.8 | +33.3 | -4.8 | no |
| 0.04 | -2 | product | 0.5 | +17.1 | +7.9 | -5.3 | -7.0 | +37.1 | -7.0 | no |
| 0.04 | -3 | sum | 0.5 | +17.7 | +12.2 | -0.2 | -3.3 | +39.5 | -3.3 | no |
| 0.04 | -3 | product | 0.5 | +19.8 | +11.4 | -2.7 | -5.4 | +44.0 | -5.4 | no |
| 0.06 | -1.5 | sum | 0.5 | +12.2 | +5.1 | -5.1 | -6.4 | +29.3 | -6.4 | no |
| 0.06 | -1.5 | product | 0.5 | +13.6 | +3.6 | -8.1 | -8.6 | +32.5 | -8.6 | no |
| 0.06 | -2 | sum | 0.5 | +13.4 | +6.6 | -4.0 | -5.7 | +32.2 | -5.7 | no |
| 0.06 | -2 | product | 0.5 | +14.8 | +5.3 | -6.8 | -7.9 | +35.7 | -7.9 | no |
| 0.06 | -3 | sum | 0.5 | +15.7 | +9.7 | -1.8 | -4.3 | +38.1 | -4.3 | no |
| 0.06 | -3 | product | 0.5 | +17.4 | +8.6 | -4.3 | -6.4 | +42.3 | -6.4 | no |

best swept point by worst-hold COMMAND margin: s2=0.04 L1=-3 astd=0.5 sum -> -3.3 %

## Sweep -- NET margin (% of walker positive income) per moving hold

| sigma^2 | L1 | combine | ang_std | fwd | lat_l | diag | back | move_turn | min | fwd>=25% |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.04 | -1.5 | sum | 0.5 | -4.4 | -27.3 | -42.7 | -45.2 | -1.2 | -45.2 | no |
| 0.04 | -1.5 | product | 0.5 | -4.7 | -32.4 | -50.0 | -51.3 | -1.2 | -51.3 | no |
| 0.04 | -2 | sum | 0.5 | -3.2 | -25.7 | -41.5 | -44.5 | +1.9 | -44.5 | no |
| 0.04 | -2 | product | 0.5 | -3.4 | -30.6 | -48.7 | -50.5 | +2.2 | -50.5 | no |
| 0.04 | -3 | sum | 0.5 | -0.8 | -22.5 | -39.2 | -43.0 | +8.0 | -43.0 | no |
| 0.04 | -3 | product | 0.5 | -0.7 | -27.1 | -46.1 | -48.9 | +9.1 | -48.9 | no |
| 0.06 | -1.5 | sum | 0.5 | -5.7 | -28.3 | -42.5 | -44.6 | -0.8 | -44.6 | no |
| 0.06 | -1.5 | product | 0.5 | -6.1 | -33.2 | -49.4 | -50.4 | -0.8 | -50.4 | no |
| 0.06 | -2 | sum | 0.5 | -4.5 | -26.7 | -41.3 | -43.9 | +2.1 | -43.9 | no |
| 0.06 | -2 | product | 0.5 | -4.8 | -31.5 | -48.2 | -49.7 | +2.5 | -49.7 | no |
| 0.06 | -3 | sum | 0.5 | -2.2 | -23.7 | -39.1 | -42.6 | +8.0 | -42.6 | no |
| 0.06 | -3 | product | 0.5 | -2.2 | -28.2 | -45.7 | -48.2 | +9.0 | -48.2 | no |

best swept point by worst-hold NET margin: s2=0.06 L1=-3 astd=0.5 sum -> -42.6 %

## Motion-requiring income (progress + yaw_progress + air_time) as a share of POSITIVE income

The diagnosis's headline number: under 1c's terms only 1.2 % of the creeper's income required moving. If the fix does not move this, standing stays affordable.

| hold | head | OLD | NEW |
|---|---|---|---|
| fwd | walker(1a) | 5.3 % | 5.7 % |
| fwd | mid(1b) | 3.3 % | 3.5 % |
| fwd | creeper(1c) | 2.1 % | 2.5 % |
| lat_l | walker(1a) | 4.3 % | 4.3 % |
| lat_l | mid(1b) | 3.5 % | 3.8 % |
| lat_l | creeper(1c) | 1.4 % | 1.6 % |
| diag | walker(1a) | 4.8 % | 4.3 % |
| diag | mid(1b) | 3.6 % | 3.8 % |
| diag | creeper(1c) | 2.1 % | 2.4 % |
| back | walker(1a) | 4.7 % | 5.3 % |
| back | mid(1b) | 3.4 % | 3.4 % |
| back | creeper(1c) | 2.6 % | 2.9 % |
| move_turn | walker(1a) | 5.5 % | 7.1 % |
| move_turn | mid(1b) | 5.4 % | 6.3 % |
| move_turn | creeper(1c) | 5.7 % | 7.4 % |

## PROBE -- two levers beyond the decided four (fwd hold, COMMAND margin)

Not part of the 2026-09-18 decision; both are fix candidates the diagnosis already listed. `ang@turn` zeroes track_ang_vel_z_exp on rows with cmd_wz == 0 (it pays a motionless robot ~1.0/s on the 88 % of rows that command no turn); `prog=` raises reward_forward_progress_along_command's weight from 0.6.

| terms | walker COMMAND | creeper COMMAND | margin | creeper motion-req. share |
|---|---|---|---|---|
| NEW (decided four) | 2.106 | 1.688 | +17.1 % | 2.5 % |
| NEW + ang@turn | 1.552 | 0.715 | +44.2 % | 4.5 % |
| NEW + prog 1.5 | 2.256 | 1.747 | +19.6 % | 5.1 % |
| NEW + prog 2.5 | 2.422 | 1.812 | +22.0 % | 7.8 % |
| NEW + ang@turn + prog 1.5 | 1.701 | 0.773 | +45.4 % | 9.1 % |
| NEW + ang@turn + prog 2.5 | 1.867 | 0.839 | +46.5 % | 13.7 % |

## Isolating the four knobs (fwd hold, COMMAND margin walker - creeper, % of walker POS)

| step | terms | walker COMMAND | creeper COMMAND | margin |
|---|---|---|---|---|
| OLD | s2=0.1 L1=-1 astd=0.15 sum | 2.328 | 2.338 | -0.4 % |
| + sigma^2 0.04 | s2=0.04 L1=-1 astd=0.15 sum | 2.114 | 2.020 | +4.1 % |
| + L1 -2.0 | s2=0.04 L1=-2 astd=0.15 sum | 1.956 | 1.797 | +6.9 % |
| + ang_std 0.5 | s2=0.04 L1=-2 astd=0.5 sum | 2.357 | 1.943 | +15.3 % |
| + combine product (= NEW) | s2=0.04 L1=-2 astd=0.5 product | 2.106 | 1.688 | +17.1 % |

## Tracking sub-economy vs achieved along-command speed (analytic, cmd 0.25 m/s)

income/s of track_lin_exp + L1 + progress at perfect direction; the last column is the slope at v = 0, i.e. the gradient PPO sees from a standstill.

| economy | v=0 | v=0.05 | v=0.1 | v=0.15 | v=0.2 | v=0.25 | d/dv at 0 |
|---|---|---|---|---|---|---|---|
| OLD (s2=0.1 L1=-1 astd=0.15 sum) | +0.419 | +0.668 | +0.908 | +1.121 | +1.289 | +1.400 | +4.96 |
| NEW (s2=0.04 L1=-2 astd=0.5 product yaw_prog_w=3) | -0.238 | +0.090 | +0.472 | +0.864 | +1.194 | +1.400 | +6.01 |

best swept point by worst-hold margin: s2=0.06 L1=-3 astd=0.5 sum -> -42.6 %

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
| walker(1a) | OLD | 2.206 | -1.587 | 0.619 |
| mid(1b) | OLD | 2.277 | -0.873 | 1.404 |
| creeper(1c) | OLD | 2.975 | -0.168 | 2.807 |
| walker(1a) | NEW | 2.406 | -1.679 | 0.727 |
| mid(1b) | NEW | 2.282 | -1.022 | 1.260 |
| creeper(1c) | NEW | 2.957 | -0.240 | 2.717 |

## VERDICT

holds that actually test walking-vs-creeping (1a speed ratio >= 0.3 AND 1a out-travels 1c by >= 2x): ['fwd', 'lat_l', 'diag', 'move_turn'] -- on every other moving hold the '1a walker' creeps too, so the margin there compares two creepers and the sign carries no information about whether walking pays.

- G1 COMMAND margin >= 25 % on every hold with a genuine walker: FAIL (fwd +17.3 %, lat_l +7.8 %, diag -5.9 %, move_turn +36.9 %)
- G2 COMMAND margin >= 25 % on EVERY moving hold: FAIL (fwd +17.3 %, lat_l +7.8 %, diag -5.9 %, back -6.5 %, move_turn +36.9 %)
- G3 zero-velocity tracking income negative at both commanded speeds: PASS (0.22 -> -0.067, 0.25 -> -0.238)

GATE: FAIL (on the G1 + G3 reading; G2 is not decidable from these raws -- see the walkable-hold line above)

### Candidate term sets on the decidable hold (fwd)

| set | terms | COMMAND margin | G1 | v=0 @0.22 | v=0 @0.25 | G3 |
|---|---|---|---|---|---|---|
| OLD (1c as run) | s2=0.1 L1=-1 astd=0.15 sum | -0.4 % | FAIL | +0.550 | +0.419 | FAIL |
| D: decided four | s2=0.04 L1=-2 astd=0.5 product | +17.1 % | FAIL | -0.067 | -0.238 | PASS |
| D + ang@turn | s2=0.04 L1=-2 astd=0.5 product ang@turn | +44.2 % | PASS | -0.067 | -0.238 | PASS |
| D + ang@turn + prog 1.5 | s2=0.04 L1=-2 astd=0.5 product ang@turn prog=1.5 | +45.4 % | PASS | -0.067 | -0.238 | PASS |
| D + ang@turn + prog 2.5 | s2=0.04 L1=-2 astd=0.5 product ang@turn prog=2.5 | +46.5 % | PASS | -0.067 | -0.238 | PASS |
| D with L1 -1.5 | s2=0.04 L1=-1.5 astd=0.5 product | +15.7 % | FAIL | +0.043 | -0.113 | FAIL |
| D with L1 -3.0 | s2=0.04 L1=-3 astd=0.5 product | +19.8 % | FAIL | -0.287 | -0.488 | PASS |
| D with sigma^2 0.06 | s2=0.06 L1=-2 astd=0.5 product | +14.8 % | FAIL | +0.118 | -0.059 | FAIL |
| D with combine sum | s2=0.04 L1=-2 astd=0.5 sum | +15.3 % | FAIL | -0.067 | -0.238 | PASS |

