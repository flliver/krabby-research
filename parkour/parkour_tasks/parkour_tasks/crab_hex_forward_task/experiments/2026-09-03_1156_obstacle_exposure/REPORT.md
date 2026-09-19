<!-- paths-note -->
> **Paths note (2026-09-09):** this campaign moved from `sim_fine_tuning/2026-09-03_1156_obstacle_exposure/` to `parkour/parkour_tasks/parkour_tasks/crab_hex_forward_task/experiments/2026-09-03_1156_obstacle_exposure/`. Absolute paths below (`/home/.../sim_fine_tuning/...`, `parkour/logs/rsl_rl/...`) name raw artifacts that stay on disk untracked; the tracked checkpoint of record is `head/` (see `bundle.yaml`) and the eval summaries are in place / under `evals/`.

# PLAN H — obstacle exposure + corridor widening — REPORT

Reporting contract: blocks terminated by `>>> ENTRY <marker>` markers.

## CAMPAIGN OPEN — 2026-09-03
- base head: `2026-09-01_15-10-31/model_19996.pt` (20k, clock 0.5 / satellites ε)
- geometry standard for C0 and arms: recal2b2w (widened corridors)
>>> ENTRY campaign open
## A0 — training-timeline probe 20k head
- checkpoint: 2026-09-01_15-10-31/model_19996.pt | episode 20.0 s | resample [6.0, 6.0] | band [0.0, 0.35] | policy std 2.000
### stochastic (300 episodes, 4000 steps x 64 envs, obstacle-tile share 0.500, terrain level 3.609)
- exposure (non-RSI obstacle tiles): reach_edge 0.308 | reach_obst 0.060 | field_frac 0.041 (steps 39.564) | goals_passed 0.051
- obst_coverage[1..6]: 1:0.060 2:0.000 3:0.000 4:0.000 5:0.000 6:0.000
- RSI episodes: reach_edge 1.000 | reach_obst 1.000 | goals_passed 0.000 | coverage 1:1.000 2:1.000 3:1.000 4:0.978 5:0.711 6:0.467
- failure share flat 0.210 (hazard 0.250/1k) | obst 0.488 (hazard 0.701/1k) | obst-RSI 0.378 | obst-spread n/a (ep len n/a) | campaign-wide None
- covariates: stand_frac_actual None | spread_frac_actual 0.000 | rsi_frac_actual 0.257 | terrain_levels None | how_far None | goal_idx None | ep len None | prints None
- motion profile (walking fraction per 6-s bin, non-RSI): [0.698, 0.660, 0.642, 0.636] | command-on per bin: [0.361, 0.382, 0.367, 0.377] | late-motion ratio 0.955
- fast (> 0.12 m/s, above zero-command creep) per bin: [0.403, 0.339, 0.323, 0.294] | fast|cmd-on 0.618 | fast|cmd-off 0.184 | late-motion ratio (fast) 0.829
- walking fraction all 0.661 / non-RSI 0.668 | cmd-on fraction 0.379 | walking|cmd-on 0.854 | walking|cmd-off 0.544 | achieved vx (cmd-on) 0.145 | vx while walking 0.155
### deterministic (362 episodes, 4000 steps x 64 envs, obstacle-tile share 0.500, terrain level 4.469)
- exposure (non-RSI obstacle tiles): reach_edge 0.394 | reach_obst 0.069 | field_frac 0.045 (steps 43.037) | goals_passed 0.044
- obst_coverage[1..6]: 1:0.069 2:0.000 3:0.000 4:0.000 5:0.000 6:0.000
- RSI episodes: reach_edge 1.000 | reach_obst 1.000 | goals_passed 0.000 | coverage 1:1.000 2:1.000 3:1.000 4:0.927 5:0.610 6:0.366
- failure share flat 0.255 (hazard 0.325/1k) | obst 0.448 (hazard 0.687/1k) | obst-RSI 0.293 | obst-spread n/a (ep len n/a) | campaign-wide None
- covariates: stand_frac_actual None | spread_frac_actual 0.000 | rsi_frac_actual 0.174 | terrain_levels None | how_far None | goal_idx None | ep len None | prints None
- motion profile (walking fraction per 6-s bin, non-RSI): [0.726, 0.669, 0.674, 0.687] | command-on per bin: [0.396, 0.358, 0.406, 0.428] | late-motion ratio 0.996
- fast (> 0.12 m/s, above zero-command creep) per bin: [0.438, 0.348, 0.362, 0.366] | fast|cmd-on 0.634 | fast|cmd-off 0.219 | late-motion ratio (fast) 0.956
- walking fraction all 0.692 / non-RSI 0.694 | cmd-on fraction 0.407 | walking|cmd-on 0.859 | walking|cmd-off 0.578 | achieved vx (cmd-on) 0.151 | vx while walking 0.163
>>> ENTRY a0 probe 20k

## A0 — training-timeline probe 30k head
- checkpoint: 2026-09-02_00-42-53/model_29994.pt | episode 20.0 s | resample [6.0, 6.0] | band [0.0, 0.35] | policy std 2.000
### stochastic (345 episodes, 4000 steps x 64 envs, obstacle-tile share 0.500, terrain level 3.125)
- exposure (non-RSI obstacle tiles): reach_edge 0.500 | reach_obst 0.062 | field_frac 0.033 (steps 28.900) | goals_passed 0.038
- obst_coverage[1..6]: 1:0.062 2:0.006 3:0.000 4:0.000 5:0.000 6:0.000
- RSI episodes: reach_edge 1.000 | reach_obst 1.000 | goals_passed 0.000 | coverage 1:1.000 2:1.000 3:1.000 4:0.968 5:0.645 6:0.419
- failure share flat 0.422 (hazard 0.566/1k) | obst 0.733 (hazard 1.221/1k) | obst-RSI 0.677 | obst-spread n/a (ep len n/a) | campaign-wide None
- covariates: stand_frac_actual None | spread_frac_actual 0.000 | rsi_frac_actual 0.171 | terrain_levels None | how_far None | goal_idx None | ep len None | prints None
- motion profile (walking fraction per 6-s bin, non-RSI): [0.728, 0.706, 0.710, 0.719] | command-on per bin: [0.360, 0.332, 0.372, 0.412] | late-motion ratio 1.006
- fast (> 0.12 m/s, above zero-command creep) per bin: [0.421, 0.391, 0.396, 0.399] | fast|cmd-on 0.653 | fast|cmd-off 0.253 | late-motion ratio (fast) 0.990
- walking fraction all 0.713 / non-RSI 0.717 | cmd-on fraction 0.367 | walking|cmd-on 0.874 | walking|cmd-off 0.620 | achieved vx (cmd-on) 0.169 | vx while walking 0.165
### deterministic (383 episodes, 4000 steps x 64 envs, obstacle-tile share 0.500, terrain level 4.938)
- exposure (non-RSI obstacle tiles): reach_edge 0.529 | reach_obst 0.102 | field_frac 0.050 (steps 45.484) | goals_passed 0.083
- obst_coverage[1..6]: 1:0.102 2:0.000 3:0.000 4:0.000 5:0.000 6:0.000
- RSI episodes: reach_edge 1.000 | reach_obst 1.000 | goals_passed 0.000 | coverage 1:1.000 2:1.000 3:1.000 4:0.957 5:0.681 6:0.468
- failure share flat 0.318 (hazard 0.445/1k) | obst 0.525 (hazard 0.833/1k) | obst-RSI 0.383 | obst-spread n/a (ep len n/a) | campaign-wide None
- covariates: stand_frac_actual None | spread_frac_actual 0.000 | rsi_frac_actual 0.214 | terrain_levels None | how_far None | goal_idx None | ep len None | prints None
- motion profile (walking fraction per 6-s bin, non-RSI): [0.745, 0.732, 0.703, 0.730] | command-on per bin: [0.403, 0.402, 0.345, 0.405] | late-motion ratio 1.004
- fast (> 0.12 m/s, above zero-command creep) per bin: [0.441, 0.418, 0.374, 0.397] | fast|cmd-on 0.636 | fast|cmd-off 0.267 | late-motion ratio (fast) 0.965
- walking fraction all 0.726 / non-RSI 0.731 | cmd-on fraction 0.394 | walking|cmd-on 0.864 | walking|cmd-off 0.636 | achieved vx (cmd-on) 0.160 | vx while walking 0.165
>>> ENTRY a0 probe 30k

## A0 — verdicts
- motion timing: late-motion ratio 0.955 -> motion is spread over random slots (no episode clock); exposure levers apply
- C2 horizon T* = 70 s (7.0 m / (0.8 x 0.145 m/s) = 61 s -> 70 s)
>>> ENTRY a0 verdicts

## E4 — trench-only eval (30k head, recal2b2 @ difficulty 0.00-0.05: heights ~0, corridor unchanged)
- completion 0.400 | tripod 0.602 | falls 60/100
- prediction was ~0.27 (trench alone reproduces the collapse)
>>> ENTRY e4

## E1 — widened-geometry baselines (recal2b2w @ 0.20-0.70; the only valid comparators downstream)
- B10 0.800 (narrow-geometry record 0.73) | B20 0.600 (0.60) | B30 0.340 (0.27)
- falls/100: 20 / 40 / 66
- prediction: 30k >= 0.55, 10k >= 0.85; if the heads are not lifted the widening is recorded as unsupported by the eval (stays in force per the user decision)
>>> ENTRY e1 baselines

## B smoke — unarmed vs armed (200 iters from the 20k head, recal2b2w)
- PASS: unarmed: telemetry keys present
- PASS: unarmed: stand_frac_actual ~ 0.57 (0.40-0.75)
- PASS: unarmed: spread_frac_actual == 0
- PASS: armed: stand_frac_actual ~ 0.2 (0.05-0.40)
- PASS: armed: spread_frac_actual > 0.25
- PASS: armed: rsi_frac_actual 0.10-0.30
- PASS: armed: goals_passed_mean parses
### unarmed
- exposure (non-RSI obstacle tiles): reach_edge 0.225 | reach_obst 0.045 | field_frac 0.027 (steps 26.127) | goals_passed 0.040
- obst_coverage[1..6]: 1:0.045 2:0.000 3:0.000 4:0.000 5:0.000 6:0.000
- RSI episodes: reach_edge 1.000 | reach_obst 1.000 | goals_passed 0.000 | coverage 1:1.000 2:1.000 3:1.000 4:0.919 5:0.666 6:0.435
- failure share flat 0.281 (hazard 0.388/1k) | obst 0.526 (hazard 0.899/1k) | obst-RSI 0.601 | obst-spread 0.000 (ep len 0.000) | campaign-wide 0.360
- covariates: stand_frac_actual 0.421 | spread_frac_actual 0.000 | rsi_frac_actual 0.201 | terrain_levels 3.508 | how_far 1.874 | goal_idx 0.243 | ep len 753.948 | prints 200
### armed (C1+C4+C5)
- exposure (non-RSI obstacle tiles): reach_edge 0.405 | reach_obst 0.212 | field_frac 0.311 (steps 154.240) | goals_passed 0.273
- obst_coverage[1..6]: 1:0.431 2:0.168 3:0.115 4:0.072 5:0.031 6:0.006
- RSI episodes: reach_edge 0.348 | reach_obst 0.183 | goals_passed 0.074 | coverage 1:0.183 2:0.000 3:0.000 4:0.000 5:0.000 6:0.000
- failure share flat 0.460 (hazard 0.761/1k) | obst 0.722 (hazard 1.470/1k) | obst-RSI 0.653 | obst-spread 0.775 (ep len 390.768) | campaign-wide 0.547
- covariates: stand_frac_actual 0.096 | spread_frac_actual 0.435 | rsi_frac_actual 0.199 | terrain_levels 1.087 | how_far 2.446 | goal_idx 1.606 | ep len 608.008 | prints 200
- **SMOKE PASS**
>>> ENTRY smoke PASS

## ARM C0 (seed 3) — **CONTROL**
- extra: {} | checkpoint 2026-09-03_15-41-21/model_24995.pt
- exposure (non-RSI obstacle tiles): reach_edge 0.327 | reach_obst 0.054 | field_frac 0.035 (steps 33.254) | goals_passed 0.055
- obst_coverage[1..6]: 1:0.054 2:0.002 3:0.000 4:0.000 5:0.000 6:0.000
- RSI episodes: reach_edge 1.000 | reach_obst 1.000 | goals_passed 0.000 | coverage 1:1.000 2:1.000 3:1.000 4:0.939 5:0.681 6:0.475
- failure share flat 0.429 (hazard 0.590/1k) | obst 0.591 (hazard 0.942/1k) | obst-RSI 0.614 | obst-spread 0.000 (ep len 0.000) | campaign-wide 0.489
- covariates: stand_frac_actual 0.444 | spread_frac_actual 0.000 | rsi_frac_actual 0.200 | terrain_levels 4.856 | how_far 1.826 | goal_idx 0.164 | ep len 700.476 | prints 5000
- flat canary: tripod 0.536 | completion 0.960 | tracking 0.430 | creep vx 0.108 | slip 0.226
- obstacle eval (recal2b2w 0.20-0.70): completion 0.420 | tripod 0.521 | falls 58/100 | vs B20 0.600
>>> ENTRY arm C0 CONTROL

## ARM C3 (seed 3) — **FAIL**
- extra: {'KRABBY_SPAWN_OFFSET': '2.0'} | checkpoint 2026-09-03_18-16-56/model_24995.pt
- exposure (non-RSI obstacle tiles): reach_edge 0.942 | reach_obst 0.535 | field_frac 0.396 (steps 323.446) | goals_passed 0.416
- obst_coverage[1..6]: 1:0.535 2:0.012 3:0.000 4:0.000 5:0.000 6:0.000
- RSI episodes: reach_edge 1.000 | reach_obst 1.000 | goals_passed 0.000 | coverage 1:1.000 2:1.000 3:1.000 4:0.905 5:0.639 6:0.433
- failure share flat 0.301 (hazard 0.373/1k) | obst 0.457 (hazard 0.631/1k) | obst-RSI 0.499 | obst-spread 0.000 (ep len 0.000) | campaign-wide 0.388
- covariates: stand_frac_actual 0.452 | spread_frac_actual 0.000 | rsi_frac_actual 0.214 | terrain_levels 5.114 | how_far 1.731 | goal_idx 0.688 | ep len 779.766 | prints 5000
- flat canary: tripod 0.567 | completion 0.930 | tracking 0.387 | creep vx 0.091 | slip 0.215
- obstacle eval (recal2b2w 0.20-0.70): completion 0.520 | tripod 0.516 | falls 48/100 | vs B20 0.600 | vs C0 0.420
- exposure status: FAIL — reach_obst_frac 0.535 < 0.8 (C0 0.054); goals_passed_mean 0.416 < 2.0 (C0 0.055); obst_coverage_3 0.000 < 0.5 (C0 0.000); obst_coverage_6 0.000 < 0.2 (C0 0.000); field_frac_mean 0.396 >= 0.2 (C0 0.035)
- safety: VIOLATED — CREEP signature: completion 0.930 with tracking 0.387
- obstacle-tile ceiling: under ceiling
>>> ENTRY arm C3 FAIL

## ARM C1 (seed 3) — **FAIL**
- extra: {'KRABBY_STAND_FRAC': '0.2'} | checkpoint 2026-09-03_20-42-17/model_24995.pt
- exposure (non-RSI obstacle tiles): reach_edge 0.759 | reach_obst 0.520 | field_frac 0.206 (steps 192.345) | goals_passed 0.348
- obst_coverage[1..6]: 1:0.520 2:0.038 3:0.000 4:0.000 5:0.000 6:0.000
- RSI episodes: reach_edge 1.000 | reach_obst 1.000 | goals_passed 0.000 | coverage 1:1.000 2:1.000 3:1.000 4:0.947 5:0.726 6:0.483
- failure share flat 0.296 (hazard 0.374/1k) | obst 0.612 (hazard 0.940/1k) | obst-RSI 0.705 | obst-spread 0.000 (ep len 0.000) | campaign-wide 0.490
- covariates: stand_frac_actual 0.185 | spread_frac_actual 0.000 | rsi_frac_actual 0.228 | terrain_levels 5.772 | how_far 2.412 | goal_idx 0.436 | ep len 671.139 | prints 5000
- flat canary: tripod 0.573 | completion 0.980 | tracking 0.521 | creep vx 0.144 | slip 0.244
- obstacle eval (recal2b2w 0.20-0.70): completion 0.510 | tripod 0.554 | falls 49/100 | vs B20 0.600 | vs C0 0.420
- exposure status: FAIL — reach_obst_frac 0.520 < 0.8 (C0 0.054); goals_passed_mean 0.348 < 2.0 (C0 0.055); obst_coverage_3 0.000 < 0.5 (C0 0.000); obst_coverage_6 0.000 < 0.2 (C0 0.000); field_frac_mean 0.206 >= 0.2 (C0 0.035)
- safety: all gates held
- obstacle-tile ceiling: under ceiling
>>> ENTRY arm C1 FAIL

## ARM C4 (seed 3) — **FAIL**
- extra: {'KRABBY_SPAWN_SPREAD': '1.0:11.0:0.5'} | checkpoint 2026-09-03_23-07-36/model_24995.pt
- exposure (non-RSI obstacle tiles): reach_edge 0.402 | reach_obst 0.099 | field_frac 0.316 (steps 181.643) | goals_passed 0.233
- obst_coverage[1..6]: 1:0.388 2:0.160 3:0.115 4:0.076 5:0.043 6:0.015
- RSI episodes: reach_edge 1.000 | reach_obst 1.000 | goals_passed 0.020 | coverage 1:1.000 2:1.000 3:1.000 4:0.924 5:0.670 6:0.455
- failure share flat 0.277 (hazard 0.340/1k) | obst 0.642 (hazard 1.074/1k) | obst-RSI 0.620 | obst-spread 0.677 (ep len 564.108) | campaign-wide 0.496
- covariates: stand_frac_actual 0.370 | spread_frac_actual 0.497 | rsi_frac_actual 0.186 | terrain_levels 5.283 | how_far 3.114 | goal_idx 1.281 | ep len 621.975 | prints 5000
- flat canary: tripod 0.569 | completion 0.900 | tracking 0.449 | creep vx 0.118 | slip 0.218
- obstacle eval (recal2b2w 0.20-0.70): completion 0.330 | tripod 0.582 | falls 67/100 | vs B20 0.600 | vs C0 0.420
- exposure status: FAIL — reach_obst_frac 0.099 < 0.8 (C0 0.054); goals_passed_mean 0.233 < 2.0 (C0 0.055); obst_coverage_3 0.115 < 0.5 (C0 0.000); obst_coverage_6 0.015 < 0.2 (C0 0.000); field_frac_mean 0.316 >= 0.2 (C0 0.035)
- safety: VIOLATED — canary completion 0.900 < C0 0.960 - 0.05
- obstacle-tile ceiling: under ceiling
>>> ENTRY arm C4 FAIL

## ARM C2 (seed 3) — ABORTED
- extra: {'KRABBY_EPISODE_S': '70'}
- exposure (non-RSI obstacle tiles): reach_edge 0.519 | reach_obst 0.371 | field_frac 0.162 (steps 360.346) | goals_passed 0.279
- obst_coverage[1..6]: 1:0.371 2:0.145 3:0.063 4:0.021 5:0.011 6:0.003
- RSI episodes: reach_edge 1.000 | reach_obst 1.000 | goals_passed 0.000 | coverage 1:1.000 2:1.000 3:1.000 4:0.964 5:0.848 6:0.496
- failure share flat 0.637 (hazard 0.356/1k) | obst 0.938 (hazard 1.010/1k) | obst-RSI 0.853 | obst-spread 0.000 (ep len 0.000) | campaign-wide 0.814
- covariates: stand_frac_actual 0.201 | spread_frac_actual 0.000 | rsi_frac_actual 0.215 | terrain_levels 1.587 | how_far 2.791 | goal_idx 0.640 | ep len 1218.282 | prints 2541
>>> ENTRY arm C2 ABORTED

## NOTE — the C2 ABORT above was SPURIOUS (orchestrator bug, fixed 2026-09-04 02:55)
- The live backstop compared the raw flat-tile failure SHARE (0.62 > C0 0.43 + 0.10) although C2 runs 70-s episodes; the plan's rule for horizon-changing arms is hazard-normalised. C2's per-1000-step flat hazard at the abort was 0.338 vs the control's 0.590 (backstop allowance 0.727, final-gate allowance 0.658) — well inside. At the control's hazard a 70-s episode would show a share of 0.87.
- Backstop and obstacle-tile ceiling now run on the hazard for arms that change the episode length; C2 is rerun in order (then C5). The aborted log is kept as c013_C2_train_ABORTED_spurious.log; its 2541-print tail: reach_obst 0.36, cov[2] 0.14, cov[3] 0.06, terrain level 1.65 (population demoted by the T-scaled promotion threshold, as the plan predicted).
>>> ENTRY note c2 abort spurious

## ARM C2 (seed 3) — **FAIL**
- extra: {'KRABBY_EPISODE_S': '70'} | checkpoint 2026-09-04_02-59-06/model_24995.pt
- exposure (non-RSI obstacle tiles): reach_edge 0.632 | reach_obst 0.503 | field_frac 0.215 (steps 559.831) | goals_passed 0.391
- obst_coverage[1..6]: 1:0.503 2:0.173 3:0.056 4:0.019 5:0.008 6:0.001
- RSI episodes: reach_edge 1.000 | reach_obst 1.000 | goals_passed 0.000 | coverage 1:1.000 2:1.000 3:1.000 4:0.972 5:0.836 6:0.521
- failure share flat 0.580 (hazard 0.295/1k) | obst 0.867 (hazard 0.676/1k) | obst-RSI 0.863 | obst-spread 0.000 (ep len 0.000) | campaign-wide 0.753
- covariates: stand_frac_actual 0.257 | spread_frac_actual 0.000 | rsi_frac_actual 0.220 | terrain_levels 1.626 | how_far 2.956 | goal_idx 0.560 | ep len 1534.115 | prints 5000
- flat canary: tripod 0.524 | completion 0.790 | tracking 0.436 | creep vx 0.112 | slip 0.235
- obstacle eval (recal2b2w 0.20-0.70): completion 0.360 | tripod 0.495 | falls 64/100 | vs B20 0.600 | vs C0 0.420
- exposure status: FAIL — reach_obst_frac 0.503 < 0.8 (C0 0.054); goals_passed_mean 0.391 < 2.0 (C0 0.055); obst_coverage_3 0.056 < 0.5 (C0 0.000); obst_coverage_6 0.001 < 0.2 (C0 0.000); field_frac_mean 0.215 >= 0.2 (C0 0.035)
- safety: VIOLATED — canary completion 0.790 < C0 0.960 - 0.05
- obstacle-tile ceiling: under ceiling
>>> ENTRY arm C2 FAIL

## ARM C5 (seed 3) — **FAIL**
- extra: {'KRABBY_RSI_SPAWN_FIX': '1'} | checkpoint 2026-09-04_05-18-20/model_24995.pt
- exposure (non-RSI obstacle tiles): reach_edge 0.492 | reach_obst 0.121 | field_frac 0.059 (steps 55.949) | goals_passed 0.097
- obst_coverage[1..6]: 1:0.121 2:0.003 3:0.000 4:0.000 5:0.000 6:0.000
- RSI episodes: reach_edge 0.249 | reach_obst 0.060 | goals_passed 0.036 | coverage 1:0.060 2:0.001 3:0.000 4:0.000 5:0.000 6:0.000
- failure share flat 0.391 (hazard 0.508/1k) | obst 0.694 (hazard 1.184/1k) | obst-RSI 0.676 | obst-spread 0.000 (ep len 0.000) | campaign-wide 0.482
- covariates: stand_frac_actual 0.416 | spread_frac_actual 0.000 | rsi_frac_actual 0.206 | terrain_levels 4.233 | how_far 0.775 | goal_idx 0.245 | ep len 689.898 | prints 5000
- flat canary: tripod 0.552 | completion 0.890 | tracking 0.447 | creep vx 0.119 | slip 0.238
- obstacle eval (recal2b2w 0.20-0.70): completion 0.270 | tripod 0.562 | falls 73/100 | vs B20 0.600 | vs C0 0.420
- exposure status: n/a — 
- safety: VIOLATED — canary completion 0.890 < C0 0.960 - 0.05
- obstacle-tile ceiling: under ceiling
- RSI-episode failure share ok: False
>>> ENTRY arm C5 FAIL

## Wave 2 — pairing eligibility (user rule 2026-09-03: PASS/PARTIAL, or safety held and reach_obst >= 3x C0)
- C1: ELIGIBLE — safety held and reach_obst 0.520 >= 3x C0 0.054
- C2: not eligible — safety gate violated: canary completion 0.790 < C0 0.960 - 0.05
- C3: not eligible — safety gate violated: CREEP signature: completion 0.930 with tracking 0.387
- C4: not eligible — safety gate violated: canary completion 0.900 < C0 0.960 - 0.05
- C5 hygiene arm not passed (omitted from pairings)
- plan (user-registered): C1+C4f = {'KRABBY_STAND_FRAC': '0.2', 'KRABBY_SPAWN_SPREAD': '2.5:11.0:0.25'}
>>> ENTRY wave2 eligibility

## ARM C1+C4f (seed 3) — **FAIL**
- extra: {'KRABBY_STAND_FRAC': '0.2', 'KRABBY_SPAWN_SPREAD': '2.5:11.0:0.25'} | checkpoint 2026-09-04_08-18-59/model_24995.pt
- exposure (non-RSI obstacle tiles): reach_edge 0.716 | reach_obst 0.475 | field_frac 0.296 (steps 207.751) | goals_passed 0.417
- obst_coverage[1..6]: 1:0.552 2:0.131 3:0.077 4:0.060 5:0.041 6:0.014
- RSI episodes: reach_edge 1.000 | reach_obst 1.000 | goals_passed 0.018 | coverage 1:1.000 2:1.000 3:1.000 4:0.937 5:0.754 6:0.483
- failure share flat 0.367 (hazard 0.479/1k) | obst 0.682 (hazard 1.130/1k) | obst-RSI 0.722 | obst-spread 0.801 (ep len 437.294) | campaign-wide 0.481
- covariates: stand_frac_actual 0.138 | spread_frac_actual 0.194 | rsi_frac_actual 0.189 | terrain_levels 5.887 | how_far 3.020 | goal_idx 1.088 | ep len 677.723 | prints 5000
- flat canary: tripod 0.576 | completion 0.920 | tracking 0.515 | creep vx 0.135 | slip 0.215
- obstacle eval (recal2b2w 0.20-0.70): completion 0.460 | tripod 0.542 | falls 54/100 | vs B20 0.600 | vs C0 0.420
- exposure status: FAIL — reach_obst_frac 0.475 < 0.8 (C0 0.054); goals_passed_mean 0.417 < 2.0 (C0 0.055); obst_coverage_3 0.077 < 0.5 (C0 0.000); obst_coverage_6 0.014 < 0.2 (C0 0.000); field_frac_mean 0.296 >= 0.2 (C0 0.035)
- safety: all gates held
- obstacle-tile ceiling: under ceiling
>>> ENTRY arm C1+C4f FAIL

## CAMPAIGN CLOSE — user decision 2026-09-04 ("let's close this campaign")
**Outcome under the plan's success criteria: ANSWERED BUT NOT SOLVED.** No Phase D replay, no re-bake;
the 30k schedule of record (`2026-09-02_00-42-53/model_29994.pt`) stands unchanged.

### Verdicts
- **Exposure hypothesis: CONFIRMED.** In the lineage's actual training configuration, 5–6% of
  platform-spawned obstacle-tile episodes reach the first obstacle and none pass the second
  (A0 20k/30k heads; C0 control reproduces it inside a full segment). The only in-field training
  data was the RSI placement bug (tile centre, 7 m downrange), worth ~0.15 obstacle completion
  (C5: 0.42 → 0.27 when fixed).
- **Motion timing: NOT systematically late** (late-motion ratios 0.96–1.01 on both heads). The
  hardware-session pattern is not a scheduler defect.
- **Corridor widening (recal2b2w): validated as a hazard, UNSUPPORTED as a stand-alone fix.**
  E4 trench-only reproduced most of the collapse (0.40 vs 0.27); E1 lifted the heads by +0.07
  (B10/B20/B30 = 0.80/0.60/0.34 vs narrow 0.73/0.60/0.27), inside the ±0.10 band. It stays in
  force for the campaign's arms and is recommended as the obstacle geometry going forward
  (with the lateral offsets narrowed, see CHANGELOG 9).
- **Exposure target (reach_obst ≥ 0.80, goals_passed ≥ 2, cov[3] ≥ 0.50, cov[6] ≥ 0.20,
  field_frac ≥ 0.20): NOT reached by any lever or pairing.** Best safe trade-offs: C1
  (STAND_FRAC 0.2: safety held, best canary, reach_obst 0.52, obstacle eval 0.51) and C1+C4f
  (+ field-only spread 25%: safety held, reach_obst 0.475, cov[3] 0.077, obstacle eval 0.46).
- **Why the target is out of reach at this plant:** 0.15 m/s on commanded slots → ~2 m per 20-s
  episode → platform starts reach obstacles 1–2 only; far-course coverage needs in-field spawns,
  and 80% of those fall within ~10 s at terrain levels 5–6, degrading the platform-start eval
  (C4). Longer horizons (C2) collapse the promotion equilibrium (level 1.6) and cost 17 points
  of flat canary. In every arm's miss anatomy the falls cluster within ±0.35 m of the platform→
  field transition at the gait's base fall rate per unit of walking time — **the limiter is
  survival at the onset and in the level-5–6 field, not exposure per se** (out of PLAN H scope;
  next step per the user: the splay plan, other session).

### All arms (20k head → 25k on recal2b2w; seed 3)
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

### Deliverables kept
- Always-on exposure telemetry (`Metrics/base_parkour/*`, `Metrics/base_velocity/stand_frac_actual`),
  env-var knobs (STAND_FRAC, SPAWN_OFFSET, SPAWN_SPREAD, RSI_SPAWN_FIX, CORRIDOR_HALF_WIDTH,
  STONE_WIDTH), geometry preset `recal2b2w`, per-tile height fields in the terrain generator,
  the training-timeline probe + miss-anatomy analysis, and the orchestrator — all unarmed =
  bit-identical, unit-tested (59 tests).
- Records: this REPORT, CHANGELOG (decisions + 12 implementation refinements), `detail_*.md`
  per arm, `probe_*` timelines, state.json (phase done).
>>> ENTRY campaign close

