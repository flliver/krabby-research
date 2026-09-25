# Wave-2 metrics: every per-category metric (2026-09-25)

Stage-end suite evals (100 episodes per category; paired categories pooled over both hands). `Δ` = W − X0x s3. `MDE` = the D00bs guard band (99 % one-sided, guards) or the block per-member MDE (95 %, block members), from the w2 notify.log. **beyond** = |Δ| past the MDE in the adverse direction for a guard, either direction for a block member — indicative only; the driver's verdicts pool members per block with a paired bootstrap. W was decided on its move_turn FLOOR (0.365 < 0.545), so the driver computed no blocks or guards for it.

Stamps: X0 s3 c6 (start) `parkour/logs/rsl_rl/gait_eval/omni_phases/2026-09-23_clock_deprecation/seed3_X01_c6_A15pB/suite__omni/seed001/2026-09-24_00-55-05`; E10 start, hidden `parkour/logs/rsl_rl/gait_eval/omni_phases/2026-09-23_clock_deprecation/step0_E10_X0s3-c6_hidden/suite__omni/seed001/2026-09-25_15-07-17`; X0x s3 `parkour/logs/rsl_rl/gait_eval/omni_phases/2026-09-23_clock_deprecation_w2/seed3_X01_c6x_A15pB/suite__omni/seed001/2026-09-25_17-01-55`; X0x s2 `parkour/logs/rsl_rl/gait_eval/omni_phases/2026-09-23_clock_deprecation_w2/seed2_X02_c6x_A15pB/suite__omni/seed001/2026-09-25_18-45-27`; W s3 `parkour/logs/rsl_rl/gait_eval/omni_phases/2026-09-23_clock_deprecation_w2/seed3_X03_c6x_A15pB/suite__omni/seed001/2026-09-25_20-31-29`

## Survival and speed

| metric · category | X0 s3 c6 (start) | E10 start, hidden | X0x s3 | X0x s2 | W s3 | Δ W−X0x s3 | MDE | |
|---|---|---|---|---|---|---|---|---|
| survival · stand | 0.990 | 1.000 | 0.990 | 0.980 | 1.000 | 0.010 | 0.030 |  |
| survival · fwd | 0.850 | 0.680 | 0.840 | 0.800 | 0.950 | 0.110 | 0.110 |  |
| survival · side | 0.970 | 0.995 | 0.965 | 0.950 | 0.995 | 0.030 | 0.045 |  |
| survival · back | 0.880 | 0.910 | 0.870 | 0.850 | 0.980 | 0.110 | 0.105 |  |
| survival · diag | 0.720 | 0.590 | 0.740 | 0.580 | 0.870 | 0.130 | 0.140 |  |
| survival · turn | 0.985 | 1.000 | 0.970 | 0.995 | 0.990 | 0.020 | 0.058 |  |
| survival · move_turn | 0.745 | 0.770 | 0.865 | 0.785 | 0.365 | -0.500 | 0.100 | **beyond (guard)** |
| ratio · fwd | 0.733 | 0.457 | 0.719 | 0.768 | 0.785 | 0.066 | 0.051 | beyond (block, better) |
| ratio · side | 0.396 | 0.348 | 0.418 | 0.415 | 0.473 | 0.055 | 0.021 | beyond (block, better) |
| ratio · back | 0.746 | 0.470 | 0.712 | 0.723 | 0.784 | 0.073 | 0.062 | beyond (block, better) |
| ratio · diag | 0.539 | 0.396 | 0.523 | 0.595 | 0.581 | 0.059 | 0.029 | beyond (block, better) |
| ratio · move_turn | 0.498 | 0.168 | 0.521 | 0.571 | 0.442 | -0.079 | 0.037 | **beyond (guard)** |
| achieved_speed · stand | 0.000 | 0.004 | 0.001 | -0.000 | -0.001 | -0.003 |  |  |
| achieved_speed · fwd | 0.183 | 0.114 | 0.180 | 0.192 | 0.196 | 0.016 |  |  |
| achieved_speed · side | 0.087 | 0.077 | 0.092 | 0.091 | 0.104 | 0.012 |  |  |
| achieved_speed · back | 0.164 | 0.103 | 0.157 | 0.159 | 0.173 | 0.016 |  |  |
| achieved_speed · diag | 0.135 | 0.099 | 0.131 | 0.149 | 0.145 | 0.015 |  |  |
| achieved_speed · turn | 0.003 | -0.001 | -0.000 | 0.001 | 0.001 | 0.001 |  |  |
| achieved_speed · move_turn | 0.110 | 0.037 | 0.115 | 0.126 | 0.097 | -0.017 |  |  |

## Turning / facing

| metric · category | X0 s3 c6 (start) | E10 start, hidden | X0x s3 | X0x s2 | W s3 | Δ W−X0x s3 | MDE | |
|---|---|---|---|---|---|---|---|---|
| facing_closed · stand | 0.294 | 0.273 | 0.867 | -0.037 | 0.437 | -0.430 |  |  |
| facing_closed · fwd | 0.750 | 0.546 | 0.351 | 0.821 | -0.242 | -0.592 |  |  |
| facing_closed · side | 0.819 | 0.558 | 0.650 | -0.170 | 0.114 | -0.536 |  |  |
| facing_closed · back | — | 0.566 | -0.457 | 0.946 | 0.336 | 0.793 |  |  |
| facing_closed · diag | -0.487 | 0.280 | -0.692 | — | -0.402 | 0.290 |  |  |
| facing_closed · turn | 0.629 | 0.181 | 0.687 | 0.584 | 0.787 | 0.100 | 0.094 | beyond (block, better) |
| facing_closed · move_turn | 0.330 | 0.033 | 0.298 | 0.303 | 0.287 | -0.011 | 0.037 |  |
| yaw_progress · stand | -0.002 | -0.001 | -0.001 | -0.002 | 0.001 | 0.003 |  |  |
| yaw_progress · fwd | -0.001 | -0.002 | -0.001 | -0.001 | -0.002 | -0.000 |  |  |
| yaw_progress · side | -0.002 | -0.001 | -0.002 | -0.002 | -0.001 | 0.001 |  |  |
| yaw_progress · back | -0.001 | -0.001 | -0.002 | -0.002 | -0.000 | 0.002 |  |  |
| yaw_progress · diag | -0.021 | -0.010 | -0.030 | -0.030 | -0.021 | 0.008 |  |  |
| yaw_progress · turn | 0.082 | 0.024 | 0.089 | 0.078 | 0.102 | 0.013 | 0.013 | beyond (block, better) |
| yaw_progress · move_turn | 0.048 | 0.005 | 0.042 | 0.042 | 0.052 | 0.010 | 0.005 | beyond (block, better) |
| net_yaw · stand | 0.003 | 0.004 | 0.002 | 0.001 | 0.004 | 0.002 |  |  |
| net_yaw · fwd | 0.005 | 0.004 | 0.002 | 0.002 | 0.003 | 0.001 |  |  |
| net_yaw · side | 0.005 | 0.006 | 0.004 | 0.003 | 0.003 | -0.001 |  |  |
| net_yaw · back | 0.002 | 0.004 | 0.002 | 0.004 | 0.004 | 0.002 |  |  |
| net_yaw · diag | 0.013 | 0.004 | -0.006 | 0.019 | 0.018 | 0.024 |  |  |
| net_yaw · turn | 0.082 | 0.024 | 0.089 | 0.077 | 0.102 | 0.013 |  |  |
| net_yaw · move_turn | 0.048 | 0.005 | 0.042 | 0.043 | 0.054 | 0.012 |  |  |
| heading_final · stand | 0.083 | 0.058 | 0.069 | 0.089 | 0.038 | -0.032 |  |  |
| heading_final · fwd | 0.074 | 0.082 | 0.089 | 0.077 | 0.073 | -0.016 |  |  |
| heading_final · side | 0.079 | 0.071 | 0.078 | 0.111 | 0.068 | -0.010 |  |  |
| heading_final · back | 0.079 | 0.085 | 0.090 | 0.116 | 0.065 | -0.025 |  |  |
| heading_final · diag | 0.249 | 0.156 | 0.302 | 0.299 | 0.262 | -0.040 |  |  |
| heading_final · turn | 0.582 | 1.282 | 0.498 | 0.664 | 0.329 | -0.169 |  |  |
| heading_final · move_turn | 1.065 | 1.512 | 1.095 | 1.083 | 1.134 | 0.039 |  |  |
| settle_s · stand | 5.560 | 5.590 | 0.000 | 3.940 | 0.000 | 0.000 |  |  |
| settle_s · fwd | 7.500 | 4.220 | 6.390 | 7.350 | 6.820 | 0.430 |  |  |
| settle_s · side | 6.620 | 4.980 | 6.090 | 7.640 | 5.850 | -0.240 |  |  |
| settle_s · back | 8.580 | 7.270 | 7.240 | 7.700 | 6.660 | -0.580 |  |  |
| settle_s · diag | 3.410 | 6.270 | 0.000 | 0.000 | 11.740 | 11.740 |  |  |
| settle_s · turn | 11.430 | — | 11.340 | 11.800 | 11.540 | 0.200 |  |  |
| wz_active_frac · stand | 0.083 | 0.114 | 0.048 | 0.086 | 0.000 | -0.048 |  |  |
| wz_active_frac · fwd | 0.168 | 0.132 | 0.193 | 0.177 | 0.095 | -0.097 |  |  |
| wz_active_frac · side | 0.142 | 0.111 | 0.143 | 0.239 | 0.073 | -0.070 |  |  |
| wz_active_frac · back | 0.183 | 0.164 | 0.185 | 0.323 | 0.071 | -0.114 |  |  |
| wz_active_frac · diag | 0.479 | 0.372 | 0.571 | 0.511 | 0.573 | 0.003 |  |  |
| wz_active_frac · turn | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 |  |  |
| wz_active_frac · move_turn | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 |  |  |

## RIG and stepping

| metric · category | X0 s3 c6 (start) | E10 start, hidden | X0x s3 | X0x s2 | W s3 | Δ W−X0x s3 | MDE | |
|---|---|---|---|---|---|---|---|---|
| steps_per_m · fwd | 16.339 | 35.515 | 16.094 | 14.671 | 14.704 | -1.389 | 1.933 |  |
| steps_per_m · side | 35.112 | 53.563 | 33.605 | 35.454 | 32.088 | -1.517 | 1.569 |  |
| steps_per_m · back | 17.352 | 39.822 | 17.818 | 17.373 | 16.941 | -0.877 | 2.395 |  |
| steps_per_m · diag | 25.117 | 37.912 | 26.890 | 23.685 | 24.114 | -2.776 | 2.024 | beyond (block, better) |
| steps_per_rad · turn | 39.863 | 147.896 | 36.715 | 43.452 | 30.390 | -6.325 | 4.402 | beyond (block, better) |
| steps_per_rad · move_turn | 68.863 | 194.664 | 75.038 | 71.856 | 53.509 | -21.529 |  |  |
| steps_per_rad_pre · turn | 39.863 | 147.896 | 36.715 | 43.452 | 30.390 | -6.325 |  |  |
| steps_per_rad_post · turn | 1.716 | — | 1.880 | 0.000 | 2.038 | 0.158 |  |  |
| support_feet · stand | 3.262 | 3.600 | 3.219 | 3.089 | 3.156 | -0.063 |  |  |
| support_feet · fwd | 3.061 | 3.308 | 3.027 | 3.020 | 3.110 | 0.083 | 0.034 | beyond (block, better) |
| support_feet · side | 3.528 | 3.691 | 3.573 | 3.497 | 3.546 | -0.027 | 0.031 |  |
| support_feet · back | 3.050 | 3.397 | 3.056 | 2.998 | 3.108 | 0.053 | 0.033 | beyond (block, better) |
| support_feet · diag | 3.340 | 3.507 | 3.424 | 3.292 | 3.383 | -0.042 | 0.084 |  |
| support_feet · turn | 3.187 | 3.369 | 3.144 | 3.045 | 3.196 | 0.052 | 0.025 | beyond (block, better) |
| support_feet · move_turn | 3.127 | 3.297 | 3.105 | 3.042 | 3.030 | -0.075 |  |  |
| steps_per_s · stand | 3.500 | 3.917 | 3.292 | 3.250 | 2.750 | -0.542 | 0.312 |  |
| step_rate_hz · fwd | 2.917 | 4.083 | 2.833 | 2.833 | 2.917 | 0.083 |  |  |
| step_rate_hz · side | 3.250 | 4.417 | 3.250 | 3.417 | 3.417 | 0.167 |  |  |
| step_rate_hz · back | 2.833 | 4.167 | 2.833 | 2.833 | 2.917 | 0.083 |  |  |
| step_rate_hz · diag | 3.500 | 4.000 | 3.583 | 3.582 | 3.500 | -0.083 |  |  |
| step_rate_hz · turn | 3.250 | 4.083 | 3.250 | 3.333 | 3.083 | -0.167 |  |  |
| step_rate_hz · move_turn | 3.417 | 3.917 | 3.417 | 3.340 | 3.164 | -0.253 |  |  |
| taps_per_m · fwd | 13.652 | 24.924 | 12.164 | 10.533 | 10.166 | -1.998 |  |  |
| taps_per_m · side | 36.290 | 42.261 | 36.245 | 35.650 | 34.378 | -1.867 |  |  |
| taps_per_m · back | 14.865 | 27.124 | 12.990 | 12.585 | 11.280 | -1.709 |  |  |
| taps_per_m · diag | 24.903 | 32.253 | 26.121 | 20.524 | 25.094 | -1.027 |  |  |
| taps_per_rad · turn | 22.526 | 114.212 | 18.834 | 21.097 | 21.156 | 2.322 |  |  |
| taps_per_rad · move_turn | 47.224 | 160.638 | 47.527 | 41.187 | 40.635 | -6.892 |  |  |
| swings_per_m · fwd | 30.097 | 58.541 | 28.399 | 25.643 | 25.096 | -3.303 |  |  |
| swings_per_m · side | 71.530 | 96.657 | 70.010 | 71.613 | 66.694 | -3.317 |  |  |
| swings_per_m · back | 33.019 | 66.114 | 31.508 | 30.310 | 28.968 | -2.540 |  |  |
| swings_per_m · diag | 50.221 | 70.238 | 53.260 | 45.807 | 49.209 | -4.051 |  |  |
| swings_per_rad · turn | 63.651 | 269.061 | 56.033 | 65.479 | 52.124 | -3.909 |  |  |
| swings_per_rad · move_turn | 115.589 | 361.107 | 122.516 | 114.359 | 94.345 | -28.172 |  |  |

## Slip and tracking quality

| metric · category | X0 s3 c6 (start) | E10 start, hidden | X0x s3 | X0x s2 | W s3 | Δ W−X0x s3 | MDE | |
|---|---|---|---|---|---|---|---|---|
| foot_slip · stand | 0.108 | 0.110 | 0.098 | 0.093 | 0.128 | 0.031 |  |  |
| foot_slip · fwd | 0.147 | 0.130 | 0.138 | 0.134 | 0.133 | -0.005 | 0.007 |  |
| foot_slip · side | 0.173 | 0.134 | 0.173 | 0.166 | 0.185 | 0.012 | 0.005 | **beyond (guard)** |
| foot_slip · back | 0.143 | 0.126 | 0.136 | 0.134 | 0.129 | -0.007 | 0.007 |  |
| foot_slip · diag | 0.186 | 0.147 | 0.178 | 0.170 | 0.187 | 0.009 | 0.010 |  |
| foot_slip · turn | 0.115 | 0.131 | 0.106 | 0.103 | 0.109 | 0.003 | 0.004 |  |
| foot_slip · move_turn | 0.134 | 0.147 | 0.130 | 0.125 | 0.139 | 0.009 |  |  |
| slip_per_m · fwd | 2.589 | 3.575 | 2.276 | 2.165 | 2.119 | -0.157 |  |  |
| slip_per_m · side | 6.835 | 6.045 | 6.385 | 6.171 | 6.164 | -0.221 |  |  |
| slip_per_m · back | 2.779 | 4.193 | 2.666 | 2.481 | 2.374 | -0.292 |  |  |
| slip_per_m · diag | 4.539 | 4.903 | 4.637 | 3.698 | 4.394 | -0.243 |  |  |
| slip_per_rad · turn | 4.587 | 16.894 | 3.752 | 4.288 | 3.343 | -0.409 |  |  |
| slip_per_rad · move_turn | 8.773 | 23.382 | 9.116 | 8.408 | 7.575 | -1.541 |  |  |
| direction_error · fwd | 0.044 | 0.063 | 0.068 | 0.062 | 0.038 | -0.029 |  |  |
| direction_error · side | 0.101 | 0.115 | 0.083 | 0.100 | 0.076 | -0.007 |  |  |
| direction_error · back | 0.062 | 0.073 | 0.070 | 0.074 | 0.044 | -0.025 |  |  |
| direction_error · diag | 0.077 | 0.177 | 0.101 | 0.069 | 0.084 | -0.017 |  |  |
| direction_error · move_turn | 0.114 | 0.312 | 0.131 | 0.060 | 0.198 | 0.066 |  |  |
| cross_track_ratio · fwd | 0.034 | 0.029 | 0.045 | 0.042 | 0.031 | -0.014 |  |  |
| cross_track_ratio · side | 0.040 | 0.039 | 0.036 | 0.042 | 0.036 | -0.000 |  |  |
| cross_track_ratio · back | 0.047 | 0.032 | 0.044 | 0.056 | 0.037 | -0.007 |  |  |
| cross_track_ratio · diag | 0.044 | 0.077 | 0.054 | 0.039 | 0.049 | -0.005 |  |  |
| cross_track_ratio · move_turn | 0.056 | 0.052 | 0.074 | 0.034 | 0.077 | 0.004 |  |  |
| drift_speed · stand | 0.082 | 0.083 | 0.079 | 0.085 | 0.080 | 0.001 |  |  |
| drift_speed · fwd | 0.212 | 0.155 | 0.206 | 0.223 | 0.213 | 0.007 |  |  |
| drift_speed · side | 0.168 | 0.147 | 0.172 | 0.167 | 0.178 | 0.006 |  |  |
| drift_speed · back | 0.191 | 0.146 | 0.190 | 0.198 | 0.189 | -0.001 |  |  |
| drift_speed · diag | 0.205 | 0.172 | 0.209 | 0.210 | 0.202 | -0.007 |  |  |
| drift_speed · turn | 0.093 | 0.125 | 0.090 | 0.098 | 0.098 | 0.007 |  |  |
| drift_speed · move_turn | 0.159 | 0.151 | 0.160 | 0.170 | 0.161 | 0.001 |  |  |
| foot_disp_end · stand | 0.244 | 0.236 | 0.233 | 0.228 | 0.219 | -0.014 |  |  |

## Gait shape / cadence (WATCH)

| metric · category | X0 s3 c6 (start) | E10 start, hidden | X0x s3 | X0x s2 | W s3 | Δ W−X0x s3 | MDE | |
|---|---|---|---|---|---|---|---|---|
| tripod · stand | 0.604 | 0.413 | 0.614 | 0.532 | 0.699 | 0.085 |  |  |
| tripod · fwd | 0.590 | 0.306 | 0.581 | 0.568 | 0.613 | 0.032 |  |  |
| tripod · side | 0.496 | 0.424 | 0.494 | 0.479 | 0.477 | -0.018 |  |  |
| tripod · back | 0.586 | 0.329 | 0.616 | 0.554 | 0.645 | 0.029 |  |  |
| tripod · diag | 0.385 | 0.258 | 0.375 | 0.359 | 0.360 | -0.015 |  |  |
| tripod · turn | 0.643 | 0.222 | 0.634 | 0.559 | 0.487 | -0.147 |  |  |
| tripod · move_turn | 0.516 | 0.167 | 0.541 | 0.513 | 0.401 | -0.140 |  |  |
| gait_hz · stand | 0.665 | 0.696 | 0.635 | 0.662 | 0.452 | -0.183 |  |  |
| gait_hz · fwd | 0.366 | 0.677 | 0.366 | 0.372 | 0.427 | 0.061 |  |  |
| gait_hz · side | 0.330 | 0.739 | 0.330 | 0.339 | 0.464 | 0.134 |  |  |
| gait_hz · back | 0.330 | 0.677 | 0.336 | 0.354 | 0.439 | 0.104 |  |  |
| gait_hz · diag | 0.403 | 0.671 | 0.421 | 0.412 | 0.439 | 0.018 |  |  |
| gait_hz · turn | 0.494 | 0.696 | 0.494 | 0.494 | 0.415 | -0.079 |  |  |
| gait_hz · move_turn | 0.494 | 0.671 | 0.494 | 0.494 | 0.439 | -0.055 |  |  |
| clock_hz · stand | 0.034 | 0.047 | 0.018 | 0.034 | 0.000 | -0.018 |  |  |
| clock_hz · fwd | 0.367 | 0.367 | 0.371 | 0.366 | 0.361 | -0.010 |  |  |
| clock_hz · side | 0.327 | 0.322 | 0.326 | 0.338 | 0.320 | -0.007 |  |  |
| clock_hz · back | 0.333 | 0.331 | 0.334 | 0.351 | 0.320 | -0.014 |  |  |
| clock_hz · diag | 0.413 | 0.383 | 0.426 | 0.412 | 0.422 | -0.005 |  |  |
| clock_hz · turn | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 | 0.000 |  |  |
| clock_hz · move_turn | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 | 0.000 |  |  |
| cadence_ratio · stand | 2.082 | 6.984 | 1.776 | 2.874 | 6.398 | 4.622 |  |  |
| cadence_ratio · fwd | 0.991 | 1.849 | 0.990 | 0.994 | 1.164 | 0.174 |  |  |
| cadence_ratio · side | 1.006 | 2.225 | 1.011 | 1.006 | 1.428 | 0.417 |  |  |
| cadence_ratio · back | 0.993 | 2.028 | 1.002 | 1.010 | 1.345 | 0.343 |  |  |
| cadence_ratio · diag | 0.987 | 1.720 | 0.988 | 0.994 | 1.041 | 0.052 |  |  |
| cadence_ratio · turn | 0.989 | 1.392 | 0.989 | 0.989 | 0.842 | -0.146 |  |  |
| cadence_ratio · move_turn | 0.989 | 1.343 | 0.989 | 0.989 | 0.879 | -0.110 |  |  |
| hz_speed_slope · fwd | -0.504 | 0.247 | -0.443 | 0.537 | -0.048 | 0.395 |  |  |
| hz_speed_slope · side | 0.647 | 1.382 | 0.916 | 1.132 | -0.367 | -1.283 |  |  |
| hz_speed_slope · back | -0.524 | 0.015 | -0.116 | 0.930 | -0.182 | -0.066 |  |  |
| hz_speed_slope · diag | 0.064 | 0.829 | -0.134 | -0.119 | 0.032 | 0.166 |  |  |
| hz_speed_slope · move_turn | 0.756 | 0.559 | 0.108 | 0.079 | 0.231 | 0.123 |  |  |
| offset_R · fwd | 0.981 | 0.073 | 0.984 | 0.965 | 0.118 | -0.865 |  |  |
| offset_R · side | 0.978 | 0.110 | 0.970 | 0.969 | 0.023 | -0.947 |  |  |
| offset_R · back | 0.964 | 0.055 | 0.966 | 0.957 | 0.049 | -0.917 |  |  |
| offset_R · diag | 0.967 | 0.069 | 0.956 | 0.968 | 0.087 | -0.869 |  |  |
| offset_R · turn | 0.997 | 0.040 | 0.995 | 0.997 | 0.101 | -0.894 |  |  |
| offset_R · move_turn | 0.981 | 0.063 | 0.991 | 0.989 | 0.051 | -0.939 |  |  |
| swing_apex · fwd | 0.119 | 0.056 | 0.118 | 0.120 | 0.110 | -0.009 |  |  |
| swing_apex · side | 0.068 | 0.047 | 0.062 | 0.064 | 0.068 | 0.006 |  |  |
| swing_apex · back | 0.129 | 0.056 | 0.122 | 0.116 | 0.109 | -0.012 |  |  |
| swing_apex · diag | 0.067 | 0.046 | 0.055 | 0.063 | 0.061 | 0.006 |  |  |
| swing_apex · turn | 0.107 | 0.051 | 0.103 | 0.107 | 0.097 | -0.006 |  |  |
| swing_apex · move_turn | 0.098 | 0.052 | 0.093 | 0.099 | 0.090 | -0.003 |  |  |
| six_down_frac · stand | 0.011 | 0.002 | 0.006 | 0.000 | 0.002 | -0.004 |  |  |
| settle_six_down_s · stand | 12.000 | 12.000 | 12.000 | 12.000 | 12.000 | 0.000 |  |  |
| contact_switch_hz · stand | 1.785 | 2.139 | 1.674 | 1.674 | 1.500 | -0.174 |  |  |
| contact_switch_hz · fwd | 1.806 | 2.333 | 1.668 | 1.616 | 1.632 | -0.036 |  |  |
| contact_switch_hz · side | 2.194 | 2.569 | 2.250 | 2.278 | 2.375 | 0.125 |  |  |
| contact_switch_hz · back | 1.778 | 2.340 | 1.611 | 1.611 | 1.611 | 0.000 |  |  |
| contact_switch_hz · diag | 2.306 | 2.423 | 2.312 | 2.236 | 2.417 | 0.104 |  |  |
| contact_switch_hz · turn | 1.688 | 2.389 | 1.681 | 1.639 | 1.750 | 0.069 |  |  |
| contact_switch_hz · move_turn | 1.897 | 2.444 | 1.833 | 1.778 | 1.817 | -0.016 |  |  |
| clock_restarts · stand | 2.000 | 2.000 | 1.000 | 2.500 | 0.000 | -1.000 |  |  |
| clock_restarts · fwd | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |  |  |
| clock_restarts · side | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |  |  |
| clock_restarts · back | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |  |  |
| clock_restarts · diag | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |  |  |
| clock_restarts · turn | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |  |  |
| clock_restarts · move_turn | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |  |  |

## Actions and effort (WATCH)

| metric · category | X0 s3 c6 (start) | E10 start, hidden | X0x s3 | X0x s2 | W s3 | Δ W−X0x s3 | MDE | |
|---|---|---|---|---|---|---|---|---|
| action_jitter · stand | 0.139 | 0.152 | 0.131 | 0.128 | 0.122 | -0.008 |  |  |
| action_jitter · fwd | 0.124 | 0.155 | 0.119 | 0.114 | 0.122 | 0.003 |  |  |
| action_jitter · side | 0.107 | 0.116 | 0.109 | 0.107 | 0.109 | -0.000 |  |  |
| action_jitter · back | 0.126 | 0.159 | 0.121 | 0.117 | 0.123 | 0.001 |  |  |
| action_jitter · diag | 0.118 | 0.130 | 0.118 | 0.115 | 0.120 | 0.002 |  |  |
| action_jitter · turn | 0.123 | 0.165 | 0.121 | 0.113 | 0.127 | 0.006 |  |  |
| action_jitter · move_turn | 0.128 | 0.156 | 0.126 | 0.117 | 0.122 | -0.003 |  |  |
| action_fft_freq · stand | 1.959 | 2.049 | 1.925 | 1.917 | 1.734 | -0.191 |  |  |
| action_fft_freq · fwd | 1.693 | 2.103 | 1.720 | 1.692 | 1.745 | 0.025 |  |  |
| action_fft_freq · side | 1.681 | 1.944 | 1.761 | 1.736 | 1.722 | -0.039 |  |  |
| action_fft_freq · back | 1.714 | 2.112 | 1.748 | 1.698 | 1.743 | -0.005 |  |  |
| action_fft_freq · diag | 1.695 | 1.988 | 1.758 | 1.761 | 1.771 | 0.014 |  |  |
| action_fft_freq · turn | 1.722 | 2.175 | 1.760 | 1.681 | 1.827 | 0.067 |  |  |
| action_fft_freq · move_turn | 1.790 | 2.123 | 1.819 | 1.734 | 1.793 | -0.026 |  |  |
| joint_speed_mean · stand | 0.592 | 0.549 | 0.582 | 0.601 | 0.656 | 0.075 |  |  |
| joint_speed_mean · fwd | 0.828 | 0.666 | 0.820 | 0.814 | 0.818 | -0.002 |  |  |
| joint_speed_mean · side | 0.730 | 0.601 | 0.716 | 0.709 | 0.702 | -0.014 |  |  |
| joint_speed_mean · back | 0.811 | 0.639 | 0.791 | 0.798 | 0.779 | -0.012 |  |  |
| joint_speed_mean · diag | 0.799 | 0.675 | 0.783 | 0.793 | 0.791 | 0.008 |  |  |
| joint_speed_mean · turn | 0.722 | 0.705 | 0.697 | 0.693 | 0.771 | 0.074 |  |  |
| joint_speed_mean · move_turn | 0.759 | 0.739 | 0.747 | 0.751 | 0.789 | 0.042 |  |  |
| rod_speed_mean · stand | 0.021 | 0.022 | 0.020 | 0.020 | 0.021 | 0.002 |  |  |
| rod_speed_mean · fwd | 0.024 | 0.023 | 0.022 | 0.021 | 0.022 | -0.000 |  |  |
| rod_speed_mean · side | 0.019 | 0.017 | 0.018 | 0.018 | 0.019 | 0.001 |  |  |
| rod_speed_mean · back | 0.024 | 0.023 | 0.022 | 0.021 | 0.022 | -0.000 |  |  |
| rod_speed_mean · diag | 0.023 | 0.020 | 0.021 | 0.022 | 0.022 | 0.000 |  |  |
| rod_speed_mean · turn | 0.022 | 0.024 | 0.020 | 0.021 | 0.021 | 0.001 |  |  |
| rod_speed_mean · move_turn | 0.024 | 0.024 | 0.022 | 0.022 | 0.022 | 0.000 |  |  |
| work_per_m · fwd | 1591.108 | 2020.255 | 1443.838 | 1357.018 | 1372.830 | -71.008 |  |  |
| work_per_m · side | 3532.676 | 3316.450 | 3358.815 | 3321.460 | 3179.898 | -178.916 |  |  |
| work_per_m · back | 1760.184 | 2313.589 | 1650.070 | 1648.415 | 1569.328 | -80.742 |  |  |
| work_per_m · diag | 2501.523 | 2507.961 | 2433.355 | 2178.013 | 2341.911 | -91.444 |  |  |
| work_per_rad · turn | 3122.834 | 9023.734 | 2589.136 | 3158.113 | 2365.236 | -223.900 |  |  |
| work_per_rad · move_turn | 5420.571 | 11791.605 | 5541.730 | 5472.448 | 4617.117 | -924.613 |  |  |
| work_per_s · stand | 370.450 | 376.233 | 338.386 | 335.454 | 449.714 | 111.327 |  |  |
| cam_work_per_m · fwd | 1121.015 | 1641.773 | 1117.470 | 971.638 | 973.731 | -143.739 |  |  |
| cam_work_per_m · side | 1797.686 | 1469.899 | 1641.863 | 1550.755 | 1456.302 | -185.561 |  |  |
| cam_work_per_m · back | 1285.981 | 1777.654 | 1292.561 | 1241.585 | 973.635 | -318.926 |  |  |
| cam_work_per_m · diag | 1284.992 | 1498.766 | 1337.673 | 1105.882 | 1126.733 | -210.940 |  |  |
| cam_work_per_rad · turn | 2018.156 | 9190.585 | 1787.367 | 1972.137 | 1708.044 | -79.323 |  |  |
| cam_work_per_rad · move_turn | 3800.731 | 12174.176 | 4039.481 | 3765.972 | 3818.640 | -220.840 |  |  |

## Diagnostics (2026-09-25, never gated): mid-stage checkpoints and terrain

Suite evals of the c6x checkpoints, each with the driver's carry shape: the whole stage overlay (W plus K) and
`trained_categories` = the 7 categories. Script: `parkour/outputs/omni_plan/clock_deprecation/run_w2_diag.py`;
stamps are under `gait_eval/omni_phases/2026-09-23_clock_deprecation_w2/diag_*`.

| metric · category | W +1000 | W +2000 | W +3000 | W +4000 | X0x s3 +2000 | X0x s3 +4000 | X0x s2 +2000 | X0x s2 +4000 |
|---|---|---|---|---|---|---|---|---|
| survival · move_turn | 0.620 | 0.255 | 0.280 | 0.365 | 0.710 | 0.865 | 0.835 | 0.785 |
| ratio · move_turn | 0.440 | 0.447 | 0.550 | 0.442 | 0.468 | 0.521 | 0.497 | 0.571 |
| yaw_progress · move_turn | 0.046 | 0.043 | 0.063 | 0.052 | 0.036 | 0.042 | 0.046 | 0.042 |
| tripod · move_turn | 0.480 | 0.412 | 0.453 | 0.401 | 0.465 | 0.541 | 0.530 | 0.513 |
| survival · fwd | 0.970 | 0.940 | 0.960 | 0.950 | 0.630 | 0.840 | 0.820 | 0.800 |
| survival · back | 1.000 | 0.960 | 0.960 | 0.980 | 0.700 | 0.870 | 0.860 | 0.850 |
| survival · diag | 0.780 | 0.740 | 0.780 | 0.870 | 0.530 | 0.740 | 0.620 | 0.580 |
| ratio · side | 0.402 | 0.463 | 0.485 | 0.473 | 0.417 | 0.418 | 0.389 | 0.415 |
| facing_closed · turn | 0.475 | 0.547 | 0.780 | 0.787 | 0.505 | 0.687 | 0.541 | 0.584 |
| steps_per_rad · turn | 49.549 | 40.082 | 27.294 | 30.390 | 50.758 | 36.715 | 46.886 | 43.452 |
| steps_per_m · side | 40.433 | 33.566 | 32.437 | 32.088 | 32.516 | 33.605 | 38.050 | 35.454 |
| foot_slip · side | 0.169 | 0.178 | 0.183 | 0.185 | 0.173 | 0.173 | 0.159 | 0.166 |
| foot_slip · turn | 0.126 | 0.125 | 0.123 | 0.109 | 0.110 | 0.106 | 0.110 | 0.103 |
| steps_per_s · stand | 2.750 | 2.750 | 2.750 | 2.750 | 3.000 | 3.292 | 3.417 | 3.250 |

Terrain WATCH (`light__omni`, 100 episodes each, stage-end heads, same carry):

| Row | Completion | Falls / n | Slip ratio (median) |
|---|---|---|---|
| X0x s3 +4000 | 0.380 | 62 / 100 | 0.369 |
| W +4000 | 0.340 | 66 / 100 | 0.413 |

