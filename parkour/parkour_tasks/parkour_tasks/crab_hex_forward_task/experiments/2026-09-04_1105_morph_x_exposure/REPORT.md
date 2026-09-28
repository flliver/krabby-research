<!-- paths-note -->
> **Paths note (2026-09-09):** this campaign moved from `sim_fine_tuning/2026-09-04_1105_morph_x_exposure/` to `parkour/parkour_tasks/parkour_tasks/crab_hex_forward_task/experiments/2026-09-04_1105_morph_x_exposure/`. Absolute paths below (`/home/.../sim_fine_tuning/...`, `parkour/logs/rsl_rl/...`) name raw artifacts that stay on disk untracked; the tracked checkpoint of record is `head/` (see `bundle.yaml`) and the eval summaries are in place / under `evals/`.

# Morphology × training-side fixes campaign — REPORT

Reporting contract: blocks terminated by `>>> ENTRY <marker>` markers.

## CAMPAIGN OPEN — 2026-09-04 11:05
- P1: 8 from-scratch 5k formation arms (base, B, A10, A15, A20, A10+B, A15+B, A20+B) with STAND_FRAC 0.2.
- P2: top 3 + golden, 0→10k with 40-s episodes / 10-s holds; stage 2 on recal2b2w with the promotion equilibrium held.
- Controls: the golden `base` arm of each protocol; rung-v control/base rows as the old-config reference.
>>> ENTRY campaign open

## SMOKE — A15, P2 stage-1 stack, 200 iterations
- PASS: plant is A15 (params/env.yaml usd path)
- PASS: episode_length_s 40 in params
- FAIL: resampling 10 s in params
- PASS: stand_frac_actual (corrected) in 0.05-0.45
- PASS: exposure keys present
- PASS: mean episode length <= 2000 steps
- PASS: morph eval ran on the A15 plant (run_meta guard passed)
- **SMOKE FAIL — P1 not started**
>>> ENTRY smoke FAIL

## SMOKE — A15, P2 stage-1 stack, 200 iterations
- PASS: plant is A15 (params/env.yaml usd path)
- PASS: episode_length_s 40 in params
- PASS: resampling 10 s in params
- PASS: stand_frac_actual (corrected) in 0.05-0.45
- PASS: exposure keys present
- PASS: mean episode length <= 2000 steps
- PASS: morph eval ran on the A15 plant (run_meta guard passed)
- **SMOKE PASS**
>>> ENTRY smoke PASS

## P1 — base (from scratch 0->5k, formation + STAND_FRAC 0.2) — ok
- status ok | checkpoint 2026-09-04_13-18-07/model_4999.pt | smoke fail@2k 0.270 coll@2k 0.000 ep_len@2k 848.713
- exposure (non-RSI obstacle tiles): reach_edge 0.881 | reach_obst 0.635 | field_frac 0.306 (steps 294.018) | goals_passed 0.377
- obst_coverage[1..6]: 1:0.635 2:0.115 3:0.008 4:0.002 5:0.000 6:0.000
- RSI episodes: reach_edge 1.000 | reach_obst 1.000 | goals_passed 0.000 | coverage 1:1.000 2:1.000 3:1.000 4:0.966 5:0.836 6:0.651
- failure share flat 0.234 (hazard 0.278/1k) | obst 0.346 (hazard 0.423/1k) | obst-RSI 0.451 | obst-spread 0.000 (ep len 0.000) | campaign-wide 0.312
- covariates: stand_frac_actual 0.189 | spread_frac_actual 0.000 | rsi_frac_actual 0.203 | terrain_levels 0.973 | how_far 2.583 | goal_idx 0.445 | ep len 779.536 | prints 5000
- stand time frac corrected 0.243 (logged 0.189) | mean reward 33.379 | vloss 0.012
- slow canary (morph manifest): tripod 0.444 | completion 0.460 | tracking 0.596 | falls 54/100 | pitch-fwd share 0.963 | prefall tip p25 -3.318 | walk tip p50 6.395
- step onset (morph manifest, shallow 0.05-0.2): completion 0.290 | falls 71/100 | pitch-fwd share 0.930 | prefall tip p25 -2.787
- obstacle eval (recal2b2w 0.20-0.70): completion 0.390 | tripod 0.438 | falls 61/100
>>> ENTRY p1 base ok

## P1 — B (from scratch 0->5k, formation + STAND_FRAC 0.2) — ok
- status ok | checkpoint 2026-09-04_15-49-15/model_4999.pt | smoke fail@2k 0.138 coll@2k 0.000 ep_len@2k 919.974
- exposure (non-RSI obstacle tiles): reach_edge 0.900 | reach_obst 0.684 | field_frac 0.305 (steps 276.621) | goals_passed 0.390
- obst_coverage[1..6]: 1:0.684 2:0.146 3:0.021 4:0.001 5:0.000 6:0.000
- RSI episodes: reach_edge 1.000 | reach_obst 1.000 | goals_passed 0.000 | coverage 1:1.000 2:1.000 3:1.000 4:0.981 5:0.889 6:0.698
- failure share flat 0.119 (hazard 0.128/1k) | obst 0.445 (hazard 0.560/1k) | obst-RSI 0.369 | obst-spread 0.000 (ep len 0.000) | campaign-wide 0.181
- covariates: stand_frac_actual 0.175 | spread_frac_actual 0.000 | rsi_frac_actual 0.206 | terrain_levels 0.973 | how_far 2.910 | goal_idx 0.509 | ep len 889.635 | prints 5000
- stand time frac corrected 0.196 (logged 0.175) | mean reward 39.932 | vloss 0.011
- slow canary (morph manifest): tripod 0.377 | completion 0.820 | tracking 0.685 | falls 18/100 | pitch-fwd share 0.611 | prefall tip p25 2.431 | walk tip p50 5.636
- step onset (morph manifest, shallow 0.05-0.2): completion 0.540 | falls 46/100 | pitch-fwd share 0.826 | prefall tip p25 -1.670
- obstacle eval (recal2b2w 0.20-0.70): completion 0.540 | tripod 0.301 | falls 46/100
>>> ENTRY p1 B ok

## P1 — A10 (from scratch 0->5k, formation + STAND_FRAC 0.2) — aborted
- status aborted | checkpoint — | smoke fail@2k 0.437 coll@2k 0.000 ep_len@2k 777.642
>>> ENTRY p1 A10 aborted

## P1 — A10 (from scratch 0->5k, formation + STAND_FRAC 0.2) — ok
- status ok | checkpoint 2026-09-04_19-33-30/model_4999.pt | smoke fail@2k 0.437 coll@2k 0.000 ep_len@2k 777.642
- exposure (non-RSI obstacle tiles): reach_edge 0.861 | reach_obst 0.582 | field_frac 0.279 (steps 242.862) | goals_passed 0.294
- obst_coverage[1..6]: 1:0.582 2:0.107 3:0.009 4:0.002 5:0.000 6:0.000
- RSI episodes: reach_edge 1.000 | reach_obst 1.000 | goals_passed 0.000 | coverage 1:1.000 2:1.000 3:1.000 4:0.973 5:0.856 6:0.643
- failure share flat 0.337 (hazard 0.414/1k) | obst 0.659 (hazard 0.995/1k) | obst-RSI 0.657 | obst-spread 0.000 (ep len 0.000) | campaign-wide 0.357
- covariates: stand_frac_actual 0.144 | spread_frac_actual 0.000 | rsi_frac_actual 0.190 | terrain_levels 0.973 | how_far 2.743 | goal_idx 0.600 | ep len 772.002 | prints 5000
- stand time frac corrected 0.187 (logged 0.144) | mean reward 31.677 | vloss 0.020
- slow canary (morph manifest): tripod 0.370 | completion 0.570 | tracking 0.685 | falls 43/100 | pitch-fwd share 0.930 | prefall tip p25 -2.504 | walk tip p50 7.060
- step onset (morph manifest, shallow 0.05-0.2): completion 0.160 | falls 84/100 | pitch-fwd share 0.917 | prefall tip p25 -1.918
- obstacle eval (recal2b2w 0.20-0.70): completion 0.150 | tripod 0.388 | falls 85/100
>>> ENTRY p1 A10 ok

## P1 — A15 (from scratch 0->5k, formation + STAND_FRAC 0.2) — ok
- status ok | checkpoint 2026-09-04_22-02-27/model_4999.pt | smoke fail@2k 0.186 coll@2k 0.000 ep_len@2k 902.360
- exposure (non-RSI obstacle tiles): reach_edge 0.956 | reach_obst 0.744 | field_frac 0.343 (steps 318.484) | goals_passed 0.449
- obst_coverage[1..6]: 1:0.744 2:0.177 3:0.023 4:0.000 5:0.000 6:0.000
- RSI episodes: reach_edge 1.000 | reach_obst 1.000 | goals_passed 0.000 | coverage 1:1.000 2:1.000 3:1.000 4:0.981 5:0.894 6:0.722
- failure share flat 0.068 (hazard 0.071/1k) | obst 0.407 (hazard 0.508/1k) | obst-RSI 0.402 | obst-spread 0.000 (ep len 0.000) | campaign-wide 0.119
- covariates: stand_frac_actual 0.186 | spread_frac_actual 0.000 | rsi_frac_actual 0.212 | terrain_levels 0.973 | how_far 2.819 | goal_idx 0.529 | ep len 915.271 | prints 5000
- stand time frac corrected 0.203 (logged 0.186) | mean reward 40.976 | vloss 0.009
- slow canary (morph manifest): tripod 0.494 | completion 0.900 | tracking 0.648 | falls 10/100 | pitch-fwd share 1.000 | prefall tip p25 -0.663 | walk tip p50 7.340
- step onset (morph manifest, shallow 0.05-0.2): completion 0.420 | falls 58/100 | pitch-fwd share 0.879 | prefall tip p25 1.327
- obstacle eval (recal2b2w 0.20-0.70): completion 0.490 | tripod 0.456 | falls 51/100
>>> ENTRY p1 A15 ok

## P1 — A20 (from scratch 0->5k, formation + STAND_FRAC 0.2) — ok
- status ok | checkpoint 2026-09-05_00-31-35/model_4999.pt | smoke fail@2k 0.143 coll@2k 0.000 ep_len@2k 918.188
- exposure (non-RSI obstacle tiles): reach_edge 0.900 | reach_obst 0.732 | field_frac 0.354 (steps 327.918) | goals_passed 0.450
- obst_coverage[1..6]: 1:0.732 2:0.184 3:0.023 4:0.000 5:0.000 6:0.000
- RSI episodes: reach_edge 1.000 | reach_obst 1.000 | goals_passed 0.000 | coverage 1:1.000 2:1.000 3:1.000 4:0.973 5:0.882 6:0.739
- failure share flat 0.131 (hazard 0.141/1k) | obst 0.407 (hazard 0.493/1k) | obst-RSI 0.380 | obst-spread 0.000 (ep len 0.000) | campaign-wide 0.136
- covariates: stand_frac_actual 0.183 | spread_frac_actual 0.000 | rsi_frac_actual 0.194 | terrain_levels 0.973 | how_far 2.877 | goal_idx 0.592 | ep len 933.115 | prints 5000
- stand time frac corrected 0.196 (logged 0.183) | mean reward 41.130 | vloss 0.012
- slow canary (morph manifest): tripod 0.466 | completion 0.500 | tracking 0.617 | falls 50/100 | pitch-fwd share 0.380 | prefall tip p25 3.087 | walk tip p50 6.320
- step onset (morph manifest, shallow 0.05-0.2): completion 0.330 | falls 67/100 | pitch-fwd share 0.552 | prefall tip p25 1.722
- obstacle eval (recal2b2w 0.20-0.70): completion 0.340 | tripod 0.470 | falls 66/100
>>> ENTRY p1 A20 ok

## P1 — A10+B (from scratch 0->5k, formation + STAND_FRAC 0.2) — ok
- status ok | checkpoint 2026-09-05_03-00-33/model_4999.pt | smoke fail@2k 0.237 coll@2k 0.000 ep_len@2k 867.389
- exposure (non-RSI obstacle tiles): reach_edge 0.903 | reach_obst 0.624 | field_frac 0.298 (steps 258.722) | goals_passed 0.366
- obst_coverage[1..6]: 1:0.624 2:0.127 3:0.027 4:0.005 5:0.000 6:0.000
- RSI episodes: reach_edge 1.000 | reach_obst 1.000 | goals_passed 0.000 | coverage 1:1.000 2:1.000 3:1.000 4:0.968 5:0.872 6:0.682
- failure share flat 0.183 (hazard 0.202/1k) | obst 0.607 (hazard 0.877/1k) | obst-RSI 0.505 | obst-spread 0.000 (ep len 0.000) | campaign-wide 0.192
- covariates: stand_frac_actual 0.217 | spread_frac_actual 0.000 | rsi_frac_actual 0.193 | terrain_levels 0.973 | how_far 2.897 | goal_idx 0.549 | ep len 872.569 | prints 5000
- stand time frac corrected 0.249 (logged 0.217) | mean reward 35.942 | vloss 0.013
- slow canary (morph manifest): tripod 0.399 | completion 0.440 | tracking 0.615 | falls 56/100 | pitch-fwd share 0.143 | prefall tip p25 6.811 | walk tip p50 9.046
- step onset (morph manifest, shallow 0.05-0.2): completion 0.210 | falls 79/100 | pitch-fwd share 0.570 | prefall tip p25 0.922
- obstacle eval (recal2b2w 0.20-0.70): completion 0.290 | tripod 0.376 | falls 71/100
>>> ENTRY p1 A10+B ok

## P1 — A15+B (from scratch 0->5k, formation + STAND_FRAC 0.2) — ok
- status ok | checkpoint 2026-09-05_05-27-19/model_4999.pt | smoke fail@2k 0.046 coll@2k 0.000 ep_len@2k 975.659
- exposure (non-RSI obstacle tiles): reach_edge 0.921 | reach_obst 0.815 | field_frac 0.423 (steps 421.244) | goals_passed 0.503
- obst_coverage[1..6]: 1:0.815 2:0.349 3:0.086 4:0.012 5:0.000 6:0.000
- RSI episodes: reach_edge 1.000 | reach_obst 1.000 | goals_passed 0.000 | coverage 1:1.000 2:1.000 3:1.000 4:0.980 5:0.892 6:0.761
- failure share flat 0.030 (hazard 0.031/1k) | obst 0.059 (hazard 0.061/1k) | obst-RSI 0.170 | obst-spread 0.000 (ep len 0.000) | campaign-wide 0.031
- covariates: stand_frac_actual 0.211 | spread_frac_actual 0.000 | rsi_frac_actual 0.214 | terrain_levels 0.973 | how_far 2.881 | goal_idx 0.626 | ep len 968.307 | prints 5000
- stand time frac corrected 0.218 (logged 0.211) | mean reward 43.256 | vloss 0.011
- slow canary (morph manifest): tripod 0.564 | completion 1.000 | tracking 0.753 | falls 0/100 | pitch-fwd share None | prefall tip p25 None | walk tip p50 10.852
- step onset (morph manifest, shallow 0.05-0.2): completion 0.950 | falls 5/100 | pitch-fwd share 0.200 | prefall tip p25 6.002
- obstacle eval (recal2b2w 0.20-0.70): completion 0.910 | tripod 0.503 | falls 9/100
>>> ENTRY p1 A15+B ok

## P1 — A20+B (from scratch 0->5k, formation + STAND_FRAC 0.2) — ok
- status ok | checkpoint 2026-09-05_07-56-31/model_4999.pt | smoke fail@2k 0.193 coll@2k 0.000 ep_len@2k 908.573
- exposure (non-RSI obstacle tiles): reach_edge 0.959 | reach_obst 0.773 | field_frac 0.428 (steps 395.406) | goals_passed 0.471
- obst_coverage[1..6]: 1:0.773 2:0.278 3:0.064 4:0.009 5:0.000 6:0.000
- RSI episodes: reach_edge 1.000 | reach_obst 1.000 | goals_passed 0.000 | coverage 1:1.000 2:1.000 3:1.000 4:0.980 5:0.875 6:0.710
- failure share flat 0.134 (hazard 0.145/1k) | obst 0.412 (hazard 0.505/1k) | obst-RSI 0.382 | obst-spread 0.000 (ep len 0.000) | campaign-wide 0.173
- covariates: stand_frac_actual 0.183 | spread_frac_actual 0.000 | rsi_frac_actual 0.193 | terrain_levels 0.973 | how_far 2.908 | goal_idx 0.686 | ep len 894.919 | prints 5000
- stand time frac corrected 0.204 (logged 0.183) | mean reward 37.663 | vloss 0.016
- slow canary (morph manifest): tripod 0.562 | completion 0.980 | tracking 0.686 | falls 2/100 | pitch-fwd share 1.000 | prefall tip p25 1.197 | walk tip p50 6.427
- step onset (morph manifest, shallow 0.05-0.2): completion 0.540 | falls 46/100 | pitch-fwd share 0.935 | prefall tip p25 -0.449
- obstacle eval (recal2b2w 0.20-0.70): completion 0.510 | tripod 0.459 | falls 49/100
>>> ENTRY p1 A20+B ok

## P2 stage 1 — base (0->5k, formation + STAND_FRAC 0.2 + 40-s episodes / 10-s holds, seed 3) — ok
- status ok | checkpoint 2026-09-05_12-03-33/model_4999.pt | smoke fail@2k 0.550 coll@2k 0.000 ep_len@2k 1302.512
- exposure (non-RSI obstacle tiles): reach_edge 0.765 | reach_obst 0.532 | field_frac 0.217 (steps 254.904) | goals_passed 0.257
- obst_coverage[1..6]: 1:0.532 2:0.160 3:0.073 4:0.040 5:0.023 6:0.009
- RSI episodes: reach_edge 1.000 | reach_obst 1.000 | goals_passed 0.000 | coverage 1:1.000 2:1.000 3:1.000 4:0.967 5:0.841 6:0.642
- failure share flat 0.814 (hazard 0.812/1k) | obst 0.942 (hazard 1.406/1k) | obst-RSI 0.902 | obst-spread 0.000 (ep len 0.000) | campaign-wide 0.773
- covariates: stand_frac_actual 0.135 | spread_frac_actual 0.000 | rsi_frac_actual 0.198 | terrain_levels 0.973 | how_far 3.419 | goal_idx 0.624 | ep len 1049.303 | prints 5000
- stand time frac corrected 0.258 (logged 0.135) | mean reward 39.660 | vloss 0.020
- slow canary (morph manifest): tripod 0.418 | completion 0.280 | tracking 0.696 | falls 72/100 | pitch-fwd share 0.583 | prefall tip p25 -1.707 | walk tip p50 4.673
- step onset (morph manifest, shallow 0.05-0.2): completion 0.090 | falls 91/100 | pitch-fwd share 0.681 | prefall tip p25 -3.054
- obstacle eval (recal2b2w 0.20-0.70): completion 0.130 | tripod 0.395 | falls 87/100
>>> ENTRY p2 s1 base ok

## P2 stage 2 — base (5k->10k, window-1 elements + recal2b2w + promotion x 20/40, seed 3) — ok
- status ok | checkpoint 2026-09-05_14-32-24/model_9998.pt | smoke fail@2k 0.735 coll@2k -0.043 ep_len@2k 996.390
- exposure (non-RSI obstacle tiles): reach_edge 0.831 | reach_obst 0.741 | field_frac 0.376 (steps 544.873) | goals_passed 0.612
- obst_coverage[1..6]: 1:0.741 2:0.431 3:0.251 4:0.124 5:0.059 6:0.027
- RSI episodes: reach_edge 1.000 | reach_obst 1.000 | goals_passed 0.000 | coverage 1:1.000 2:1.000 3:1.000 4:0.958 5:0.835 6:0.620
- failure share flat 0.466 (hazard 0.330/1k) | obst 0.797 (hazard 0.795/1k) | obst-RSI 0.818 | obst-spread 0.000 (ep len 0.000) | campaign-wide 0.567
- covariates: stand_frac_actual 0.120 | spread_frac_actual 0.000 | rsi_frac_actual 0.211 | terrain_levels 5.827 | how_far 3.548 | goal_idx 0.647 | ep len 1174.396 | prints 5000
- stand time frac corrected 0.204 (logged 0.120) | mean reward 42.055 | vloss 0.021
- slow canary (morph manifest): tripod 0.520 | completion 0.680 | tracking 0.625 | falls 32/100 | pitch-fwd share 0.969 | prefall tip p25 -3.651 | walk tip p50 5.264
- step onset (morph manifest, shallow 0.05-0.2): completion 0.330 | falls 67/100 | pitch-fwd share 0.985 | prefall tip p25 -2.291
- obstacle eval (recal2b2w 0.20-0.70): completion 0.310 | tripod 0.484 | falls 69/100
- terrain level 5.827 (lineage 20-s window-1 ~4-6; outside 3-7 = re-derive the promotion scaling)
>>> ENTRY p2 s2 base ok

## P2 stage 1 — A15+B (0->5k, formation + STAND_FRAC 0.2 + 40-s episodes / 10-s holds, seed 3) — ok
- status ok | checkpoint 2026-09-05_16-51-20/model_4999.pt | smoke fail@2k 0.500 coll@2k 0.000 ep_len@2k 1484.577
- exposure (non-RSI obstacle tiles): reach_edge 0.980 | reach_obst 0.860 | field_frac 0.522 (steps 882.675) | goals_passed 0.613
- obst_coverage[1..6]: 1:0.860 2:0.540 3:0.326 4:0.212 5:0.123 6:0.049
- RSI episodes: reach_edge 1.000 | reach_obst 1.000 | goals_passed 0.000 | coverage 1:1.000 2:1.000 3:1.000 4:0.981 5:0.921 6:0.818
- failure share flat 0.227 (hazard 0.128/1k) | obst 0.479 (hazard 0.338/1k) | obst-RSI 0.426 | obst-spread 0.000 (ep len 0.000) | campaign-wide 0.263
- covariates: stand_frac_actual 0.152 | spread_frac_actual 0.000 | rsi_frac_actual 0.201 | terrain_levels 0.973 | how_far 4.218 | goal_idx 0.909 | ep len 1652.749 | prints 5000
- stand time frac corrected 0.184 (logged 0.152) | mean reward 72.094 | vloss 0.011
- slow canary (morph manifest): tripod 0.539 | completion 0.860 | tracking 0.676 | falls 14/100 | pitch-fwd share 1.000 | prefall tip p25 1.201 | walk tip p50 6.715
- step onset (morph manifest, shallow 0.05-0.2): completion 0.690 | falls 31/100 | pitch-fwd share 0.935 | prefall tip p25 -1.391
- obstacle eval (recal2b2w 0.20-0.70): completion 0.680 | tripod 0.523 | falls 32/100
>>> ENTRY p2 s1 A15+B ok

## P2 stage 2 — A15+B (5k->10k, window-1 elements + recal2b2w + promotion x 20/40, seed 3) — ok
- status ok | checkpoint 2026-09-05_19-18-29/model_9998.pt | smoke fail@2k 0.279 coll@2k -0.051 ep_len@2k 1678.041
- exposure (non-RSI obstacle tiles): reach_edge 0.964 | reach_obst 0.915 | field_frac 0.599 (steps 1030.212) | goals_passed 1.001
- obst_coverage[1..6]: 1:0.915 2:0.709 3:0.487 4:0.297 5:0.146 6:0.058
- RSI episodes: reach_edge 1.000 | reach_obst 1.000 | goals_passed 0.000 | coverage 1:1.000 2:1.000 3:1.000 4:0.973 5:0.883 6:0.755
- failure share flat 0.119 (hazard 0.064/1k) | obst 0.478 (hazard 0.320/1k) | obst-RSI 0.461 | obst-spread 0.000 (ep len 0.000) | campaign-wide 0.276
- covariates: stand_frac_actual 0.158 | spread_frac_actual 0.000 | rsi_frac_actual 0.200 | terrain_levels 6.033 | how_far 4.022 | goal_idx 0.992 | ep len 1616.303 | prints 5000
- stand time frac corrected 0.196 (logged 0.158) | mean reward 68.100 | vloss 0.013
- slow canary (morph manifest): tripod 0.589 | completion 0.950 | tracking 0.622 | falls 5/100 | pitch-fwd share 1.000 | prefall tip p25 2.288 | walk tip p50 9.526
- step onset (morph manifest, shallow 0.05-0.2): completion 0.770 | falls 23/100 | pitch-fwd share 0.739 | prefall tip p25 1.064
- obstacle eval (recal2b2w 0.20-0.70): completion 0.730 | tripod 0.505 | falls 27/100
- terrain level 6.033 (lineage 20-s window-1 ~4-6; outside 3-7 = re-derive the promotion scaling)
>>> ENTRY p2 s2 A15+B ok

## P2 stage 1 — B (0->5k, formation + STAND_FRAC 0.2 + 40-s episodes / 10-s holds, seed 3) — ok
- status ok | checkpoint 2026-09-05_21-37-35/model_4999.pt | smoke fail@2k 0.168 coll@2k 0.000 ep_len@2k 1816.745
- exposure (non-RSI obstacle tiles): reach_edge 0.970 | reach_obst 0.893 | field_frac 0.563 (steps 1069.104) | goals_passed 0.518
- obst_coverage[1..6]: 1:0.893 2:0.625 3:0.369 4:0.206 5:0.108 6:0.041
- RSI episodes: reach_edge 1.000 | reach_obst 1.000 | goals_passed 0.000 | coverage 1:1.000 2:1.000 3:1.000 4:0.978 5:0.907 6:0.773
- failure share flat 0.092 (hazard 0.049/1k) | obst 0.279 (hazard 0.166/1k) | obst-RSI 0.490 | obst-spread 0.000 (ep len 0.000) | campaign-wide 0.109
- covariates: stand_frac_actual 0.195 | spread_frac_actual 0.000 | rsi_frac_actual 0.189 | terrain_levels 0.973 | how_far 4.059 | goal_idx 0.719 | ep len 1914.899 | prints 5000
- stand time frac corrected 0.203 (logged 0.195) | mean reward 91.115 | vloss 0.009
- slow canary (morph manifest): tripod 0.382 | completion 0.790 | tracking 0.642 | falls 21/100 | pitch-fwd share 0.952 | prefall tip p25 -2.808 | walk tip p50 7.902
- step onset (morph manifest, shallow 0.05-0.2): completion 0.680 | falls 32/100 | pitch-fwd share 0.906 | prefall tip p25 -2.115
- obstacle eval (recal2b2w 0.20-0.70): completion 0.670 | tripod 0.340 | falls 33/100
>>> ENTRY p2 s1 B ok

## P2 stage 2 — B (5k->10k, window-1 elements + recal2b2w + promotion x 20/40, seed 3) — ok
- status ok | checkpoint 2026-09-06_00-09-00/model_9998.pt | smoke fail@2k 0.233 coll@2k -0.007 ep_len@2k 1684.758
- exposure (non-RSI obstacle tiles): reach_edge 0.963 | reach_obst 0.927 | field_frac 0.612 (steps 1143.790) | goals_passed 0.967
- obst_coverage[1..6]: 1:0.927 2:0.756 3:0.532 4:0.307 5:0.165 6:0.078
- RSI episodes: reach_edge 1.000 | reach_obst 1.000 | goals_passed 0.000 | coverage 1:1.000 2:1.000 3:1.000 4:0.967 5:0.880 6:0.738
- failure share flat 0.074 (hazard 0.039/1k) | obst 0.291 (hazard 0.173/1k) | obst-RSI 0.452 | obst-spread 0.000 (ep len 0.000) | campaign-wide 0.177
- covariates: stand_frac_actual 0.178 | spread_frac_actual 0.000 | rsi_frac_actual 0.175 | terrain_levels 5.749 | how_far 4.111 | goal_idx 0.751 | ep len 1730.694 | prints 5000
- stand time frac corrected 0.206 (logged 0.178) | mean reward 78.744 | vloss 0.015
- slow canary (morph manifest): tripod 0.455 | completion 0.900 | tracking 0.622 | falls 10/100 | pitch-fwd share 1.000 | prefall tip p25 -3.760 | walk tip p50 9.801
- step onset (morph manifest, shallow 0.05-0.2): completion 0.860 | falls 14/100 | pitch-fwd share 1.000 | prefall tip p25 1.242
- obstacle eval (recal2b2w 0.20-0.70): completion 0.830 | tripod 0.389 | falls 17/100
- terrain level 5.749 (lineage 20-s window-1 ~4-6; outside 3-7 = re-derive the promotion scaling)
>>> ENTRY p2 s2 B ok

## P2 stage 1 — A15 (0->5k, formation + STAND_FRAC 0.2 + 40-s episodes / 10-s holds, seed 3) — ok
- status ok | checkpoint 2026-09-06_02-36-19/model_4999.pt | smoke fail@2k 0.178 coll@2k 0.000 ep_len@2k 1808.592
- exposure (non-RSI obstacle tiles): reach_edge 0.958 | reach_obst 0.849 | field_frac 0.534 (steps 956.737) | goals_passed 0.579
- obst_coverage[1..6]: 1:0.849 2:0.552 3:0.320 4:0.174 5:0.088 6:0.028
- RSI episodes: reach_edge 1.000 | reach_obst 1.000 | goals_passed 0.000 | coverage 1:1.000 2:1.000 3:1.000 4:0.978 5:0.919 6:0.811
- failure share flat 0.149 (hazard 0.082/1k) | obst 0.415 (hazard 0.271/1k) | obst-RSI 0.528 | obst-spread 0.000 (ep len 0.000) | campaign-wide 0.219
- covariates: stand_frac_actual 0.144 | spread_frac_actual 0.000 | rsi_frac_actual 0.216 | terrain_levels 0.973 | how_far 4.174 | goal_idx 0.773 | ep len 1679.673 | prints 5000
- stand time frac corrected 0.171 (logged 0.144) | mean reward 73.756 | vloss 0.008
- slow canary (morph manifest): tripod 0.494 | completion 0.780 | tracking 0.595 | falls 22/100 | pitch-fwd share 0.545 | prefall tip p25 -0.008 | walk tip p50 7.284
- step onset (morph manifest, shallow 0.05-0.2): completion 0.450 | falls 55/100 | pitch-fwd share 0.273 | prefall tip p25 4.356
- obstacle eval (recal2b2w 0.20-0.70): completion 0.480 | tripod 0.446 | falls 52/100
>>> ENTRY p2 s1 A15 ok

## P2 stage 2 — A15 (5k->10k, window-1 elements + recal2b2w + promotion x 20/40, seed 3) — ok
- status ok | checkpoint 2026-09-06_05-05-22/model_9998.pt | smoke fail@2k 0.320 coll@2k -0.032 ep_len@2k 1610.232
- exposure (non-RSI obstacle tiles): reach_edge 0.966 | reach_obst 0.858 | field_frac 0.525 (steps 882.690) | goals_passed 0.984
- obst_coverage[1..6]: 1:0.858 2:0.629 3:0.401 4:0.195 5:0.077 6:0.036
- RSI episodes: reach_edge 1.000 | reach_obst 1.000 | goals_passed 0.000 | coverage 1:1.000 2:1.000 3:1.000 4:0.972 5:0.875 6:0.716
- failure share flat 0.143 (hazard 0.079/1k) | obst 0.541 (hazard 0.388/1k) | obst-RSI 0.544 | obst-spread 0.000 (ep len 0.000) | campaign-wide 0.316
- covariates: stand_frac_actual 0.164 | spread_frac_actual 0.000 | rsi_frac_actual 0.191 | terrain_levels 6.032 | how_far 3.866 | goal_idx 0.890 | ep len 1617.874 | prints 5000
- stand time frac corrected 0.202 (logged 0.164) | mean reward 67.798 | vloss 0.015
- slow canary (morph manifest): tripod 0.540 | completion 1.000 | tracking 0.584 | falls 0/100 | pitch-fwd share None | prefall tip p25 None | walk tip p50 8.013
- step onset (morph manifest, shallow 0.05-0.2): completion 0.740 | falls 26/100 | pitch-fwd share 0.962 | prefall tip p25 -0.876
- obstacle eval (recal2b2w 0.20-0.70): completion 0.670 | tripod 0.501 | falls 33/100
- terrain level 6.032 (lineage 20-s window-1 ~4-6; outside 3-7 = re-derive the promotion scaling)
>>> ENTRY p2 s2 A15 ok

## DECISION TABLE — morphology x training-side fixes (P1: formation + walking slots; P2: + 40-s episodes, 10-s holds; stage 2 on recal2b2w with the promotion equilibrium held)
| config | hardware | rung iv 30k falls fwd/step (Δ) | rung v slow tripod / compl / track (old config) | rung v step falls (ratio) | P1 status | P1 slow tripod / compl / track (ratio to golden) | P1 slow falls | P1 step falls (ratio) | P1 obst recal2b2w compl (falls) | P1 reach_obst | P1 field_frac | P1 cov[3] | P1 fail hazard flat / obst (/1k) | P1 terrain level | P1 step prefall tip p25 | P2s1 status | P2s1 slow tripod / compl / track (ratio to golden) | P2s1 slow falls | P2s1 step falls (ratio) | P2s1 obst recal2b2w compl (falls) | P2s1 reach_obst | P2s1 field_frac | P2s1 cov[3] | P2s1 fail hazard flat / obst (/1k) | P2s1 terrain level | P2s1 step prefall tip p25 | P2s2 status | P2s2 slow tripod / compl / track (ratio to golden) | P2s2 slow falls | P2s2 step falls (ratio) | P2s2 obst recal2b2w compl (falls) | P2s2 reach_obst | P2s2 field_frac | P2s2 cov[3] | P2s2 fail hazard flat / obst (/1k) | P2s2 terrain level | P2s2 step prefall tip p25 | seed-2 P2s2 slow / step / obst | user preference |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| base | none | 100 / 74 | 0.537 (0.90×) / 0.83 (0.94×) / 0.426 (1.07×) | 69/100 (1.77×) | ok | 0.444 / 0.46 / 0.596 (—, —, —) | 54/100 | 71/100 (—) | 0.39 (61/100) | 0.635 | 0.306 | 0.008 | 0.278 / 0.423 | 0.97 | -2.8° | ok | 0.418 / 0.28 / 0.696 (—, —, —) | 72/100 | 91/100 (—) | 0.13 (87/100) | 0.532 | 0.217 | 0.073 | 0.812 / 1.406 | 0.97 | -3.1° | ok | 0.520 / 0.68 / 0.625 (—, —, —) | 32/100 | 67/100 (—) | 0.31 (69/100) | 0.741 | 0.376 | 0.251 | 0.330 / 0.795 | 5.83 | -2.3° | — | — |
| B | re-hinge 3 in | 40 (-60) / 30 (-44) | 0.256 (0.43×) / 0.31 (0.35×) / 0.534 (1.35×) | 69/100 (1.77×) | ok | 0.377 / 0.82 / 0.685 (0.85×, 1.78×, 1.15×) | 18/100 | 46/100 (0.65×) | 0.54 (46/100) | 0.684 | 0.305 | 0.021 | 0.128 / 0.560 (golden 0.278 / 0.423) | 0.97 | -1.7° | ok | 0.382 / 0.79 / 0.642 (0.91×, 2.82×, 0.92×) | 21/100 | 32/100 (0.35×) | 0.67 (33/100) | 0.893 | 0.563 | 0.369 | 0.049 / 0.166 (golden 0.812 / 1.406) | 0.97 | -2.1° | ok | 0.455 / 0.90 / 0.622 (0.87×, 1.32×, 0.99×) | 10/100 | 14/100 (0.21×) | 0.83 (17/100) | 0.927 | 0.612 | 0.532 | 0.039 / 0.173 (golden 0.330 / 0.795) | 5.75 | 1.2° | — | re-hinge |
| A10 | 10° shims | 33 (-67) / 21 (-53) | 0.145 (0.24×) / 0.23 (0.26×) / 0.685 (1.73×) | 90/100 (2.31×) | ok | 0.370 / 0.57 / 0.685 (0.83×, 1.24×, 1.15×) | 43/100 | 84/100 (1.18×) | 0.15 (85/100) | 0.582 | 0.279 | 0.009 | 0.414 / 0.995 (golden 0.278 / 0.423) | 0.97 | -1.9° | pending | — | — | — | — | — | — | — | — | — | — | pending | — | — | — | — | — | — | — | — | — | — | — | splay only ✓ |
| A15 | 15° shims | 28 (-72) / 6 (-68) | 0.243 (0.41×) / 0.46 (0.52×) / 0.525 (1.32×) | 68/100 (1.74×) | ok | 0.494 / 0.90 / 0.648 (1.11×, 1.96×, 1.09×) | 10/100 | 58/100 (0.82×) | 0.49 (51/100) | 0.744 | 0.343 | 0.023 | 0.071 / 0.508 (golden 0.278 / 0.423) | 0.97 | 1.3° | ok | 0.494 / 0.78 / 0.595 (1.18×, 2.79×, 0.86×) | 22/100 | 55/100 (0.60×) | 0.48 (52/100) | 0.849 | 0.534 | 0.320 | 0.082 / 0.271 (golden 0.812 / 1.406) | 0.97 | 4.4° | ok | 0.540 / 1.00 / 0.584 (1.04×, 1.47×, 0.93×) | 0/100 | 26/100 (0.39×) | 0.67 (33/100) | 0.858 | 0.525 | 0.401 | 0.079 / 0.388 (golden 0.330 / 0.795) | 6.03 | -0.9° | — | splay only ✓ |
| A20 | 20° shims | 31 (-69) / 7 (-67) | 0.319 (0.54×) / 0.76 (0.86×) / 0.707 (1.78×) | 66/100 (1.69×) | ok | 0.466 / 0.50 / 0.617 (1.05×, 1.09×, 1.03×) | 50/100 | 67/100 (0.94×) | 0.34 (66/100) | 0.732 | 0.354 | 0.023 | 0.141 / 0.493 (golden 0.278 / 0.423) | 0.97 | 1.7° | pending | — | — | — | — | — | — | — | — | — | — | pending | — | — | — | — | — | — | — | — | — | — | — | splay only ✓ |
| A10+B | 10° shims + re-hinge | 24 (-76) / 2 (-72) | — (—) / — (—) / — (—) | — | ok | 0.399 / 0.44 / 0.615 (0.90×, 0.96×, 1.03×) | 56/100 | 79/100 (1.11×) | 0.29 (71/100) | 0.624 | 0.298 | 0.027 | 0.202 / 0.877 (golden 0.278 / 0.423) | 0.97 | 0.9° | pending | — | — | — | — | — | — | — | — | — | — | pending | — | — | — | — | — | — | — | — | — | — | — | shims + re-hinge |
| A15+B | 15° shims + re-hinge | 22 (-78) / 0 (-74) | — (—) / — (—) / — (—) | — | ok | 0.564 / 1.00 / 0.753 (1.27×, 2.17×, 1.26×) | 0/100 | 5/100 (0.07×) | 0.91 (9/100) | 0.815 | 0.423 | 0.086 | 0.031 / 0.061 (golden 0.278 / 0.423) | 0.97 | 6.0° | ok | 0.539 / 0.86 / 0.676 (1.29×, 3.07×, 0.97×) | 14/100 | 31/100 (0.34×) | 0.68 (32/100) | 0.860 | 0.522 | 0.326 | 0.128 / 0.338 (golden 0.812 / 1.406) | 0.97 | -1.4° | ok | 0.589 / 0.95 / 0.622 (1.13×, 1.40×, 0.99×) | 5/100 | 23/100 (0.34×) | 0.73 (27/100) | 0.915 | 0.599 | 0.487 | 0.064 / 0.320 (golden 0.330 / 0.795) | 6.03 | 1.1° | — | shims + re-hinge |
| A20+B | 20° shims + re-hinge | 19 (-81) / 1 (-73) | — (—) / — (—) / — (—) | — | ok | 0.562 / 0.98 / 0.686 (1.27×, 2.13×, 1.15×) | 2/100 | 46/100 (0.65×) | 0.51 (49/100) | 0.773 | 0.428 | 0.064 | 0.145 / 0.505 (golden 0.278 / 0.423) | 0.97 | -0.4° | pending | — | — | — | — | — | — | — | — | — | — | pending | — | — | — | — | — | — | — | — | — | — | — | shims + re-hinge |
- every arm from scratch, seed 3; ratios are variant / the same protocol's golden arm; 0.85× is the canary reference line; rung-v noise floor ≈ 0.06 tripod / 0.05 completion / 5 falls per 100
- rung iv / rung v columns carried from the morphology campaign (stop2_table.csv, stop3_decision_table.csv; rung v = old formation config)
- fail hazards are per 1000 env steps (episode-length neutral); P2 stage-2 terrain level should sit near the lineage's 4–6 or the promotion scaling is re-derived
>>> ENTRY decision table

## SEED2 stage 1 — B (0->5k, formation + STAND_FRAC 0.2 + 40-s episodes / 10-s holds, seed 2) — ok
- status ok | checkpoint 2026-09-06_11-13-20/model_4999.pt | smoke fail@2k 0.300 coll@2k 0.000 ep_len@2k 1555.417
- exposure (non-RSI obstacle tiles): reach_edge 0.940 | reach_obst 0.823 | field_frac 0.477 (steps 843.606) | goals_passed 0.604
- obst_coverage[1..6]: 1:0.823 2:0.513 3:0.369 4:0.237 5:0.127 6:0.048
- RSI episodes: reach_edge 1.000 | reach_obst 1.000 | goals_passed 0.000 | coverage 1:1.000 2:1.000 3:1.000 4:0.986 5:0.943 6:0.785
- failure share flat 0.115 (hazard 0.063/1k) | obst 0.485 (hazard 0.351/1k) | obst-RSI 0.485 | obst-spread 0.000 (ep len 0.000) | campaign-wide 0.206
- covariates: stand_frac_actual 0.192 | spread_frac_actual 0.000 | rsi_frac_actual 0.193 | terrain_levels 1.027 | how_far 4.296 | goal_idx 0.824 | ep len 1531.920 | prints 5000
- stand time frac corrected 0.251 (logged 0.192) | mean reward 70.368 | vloss 0.009
- slow canary (morph manifest): tripod 0.492 | completion 0.780 | tracking 0.664 | falls 22/100 | pitch-fwd share 0.273 | prefall tip p25 3.481 | walk tip p50 6.727
- step onset (morph manifest, shallow 0.05-0.2): completion 0.370 | falls 63/100 | pitch-fwd share 0.476 | prefall tip p25 2.467
- obstacle eval (recal2b2w 0.20-0.70): completion 0.290 | tripod 0.393 | falls 71/100
>>> ENTRY seed2 s1 B ok

## SEED2 stage 2 — B (5k->10k, window-1 elements + recal2b2w + promotion x 20/40, seed 2) — ok
- status ok | checkpoint 2026-09-06_13-42-22/model_9998.pt | smoke fail@2k 0.346 coll@2k -0.026 ep_len@2k 1577.109
- exposure (non-RSI obstacle tiles): reach_edge 0.916 | reach_obst 0.821 | field_frac 0.486 (steps 819.509) | goals_passed 0.883
- obst_coverage[1..6]: 1:0.821 2:0.576 3:0.351 4:0.176 5:0.093 6:0.045
- RSI episodes: reach_edge 1.000 | reach_obst 1.000 | goals_passed 0.000 | coverage 1:1.000 2:1.000 3:1.000 4:0.985 5:0.891 6:0.725
- failure share flat 0.123 (hazard 0.068/1k) | obst 0.569 (hazard 0.427/1k) | obst-RSI 0.508 | obst-spread 0.000 (ep len 0.000) | campaign-wide 0.331
- covariates: stand_frac_actual 0.133 | spread_frac_actual 0.000 | rsi_frac_actual 0.189 | terrain_levels 5.844 | how_far 4.011 | goal_idx 0.894 | ep len 1455.494 | prints 5000
- stand time frac corrected 0.182 (logged 0.133) | mean reward 62.564 | vloss 0.017
- slow canary (morph manifest): tripod 0.578 | completion 0.960 | tracking 0.650 | falls 4/100 | pitch-fwd share 0.250 | prefall tip p25 6.628 | walk tip p50 8.287
- step onset (morph manifest, shallow 0.05-0.2): completion 0.720 | falls 28/100 | pitch-fwd share 0.607 | prefall tip p25 -0.513
- obstacle eval (recal2b2w 0.20-0.70): completion 0.690 | tripod 0.484 | falls 31/100
- terrain level 5.844 (lineage 20-s window-1 ~4-6; outside 3-7 = re-derive the promotion scaling)
>>> ENTRY seed2 s2 B ok

## SEED2 stage 1 — A15+B (0->5k, formation + STAND_FRAC 0.2 + 40-s episodes / 10-s holds, seed 2) — ok
- status ok | checkpoint 2026-09-06_16-03-30/model_4999.pt | smoke fail@2k 0.239 coll@2k 0.000 ep_len@2k 1757.623
- exposure (non-RSI obstacle tiles): reach_edge 0.941 | reach_obst 0.898 | field_frac 0.620 (steps 1121.437) | goals_passed 0.599
- obst_coverage[1..6]: 1:0.898 2:0.719 3:0.563 4:0.402 5:0.262 6:0.133
- RSI episodes: reach_edge 1.000 | reach_obst 1.000 | goals_passed 0.000 | coverage 1:1.000 2:1.000 3:1.000 4:0.976 5:0.908 6:0.807
- failure share flat 0.272 (hazard 0.158/1k) | obst 0.417 (hazard 0.268/1k) | obst-RSI 0.371 | obst-spread 0.000 (ep len 0.000) | campaign-wide 0.329
- covariates: stand_frac_actual 0.170 | spread_frac_actual 0.000 | rsi_frac_actual 0.195 | terrain_levels 1.027 | how_far 4.232 | goal_idx 0.736 | ep len 1672.106 | prints 5000
- stand time frac corrected 0.203 (logged 0.170) | mean reward 65.426 | vloss 0.016
- slow canary (morph manifest): tripod 0.476 | completion 0.580 | tracking 0.713 | falls 42/100 | pitch-fwd share 0.452 | prefall tip p25 0.288 | walk tip p50 8.668
- step onset (morph manifest, shallow 0.05-0.2): completion 0.480 | falls 52/100 | pitch-fwd share 0.615 | prefall tip p25 -1.328
- obstacle eval (recal2b2w 0.20-0.70): completion 0.330 | tripod 0.405 | falls 67/100
>>> ENTRY seed2 s1 A15+B ok

## SEED2 stage 2 — A15+B (5k->10k, window-1 elements + recal2b2w + promotion x 20/40, seed 2) — ok
- status ok | checkpoint 2026-09-06_18-32-32/model_9998.pt | smoke fail@2k 0.163 coll@2k -0.020 ep_len@2k 1808.404
- exposure (non-RSI obstacle tiles): reach_edge 0.982 | reach_obst 0.942 | field_frac 0.658 (steps 1222.386) | goals_passed 0.979
- obst_coverage[1..6]: 1:0.942 2:0.814 3:0.641 4:0.432 5:0.249 6:0.123
- RSI episodes: reach_edge 1.000 | reach_obst 1.000 | goals_passed 0.000 | coverage 1:1.000 2:1.000 3:1.000 4:0.972 5:0.881 6:0.776
- failure share flat 0.111 (hazard 0.060/1k) | obst 0.269 (hazard 0.160/1k) | obst-RSI 0.388 | obst-spread 0.000 (ep len 0.000) | campaign-wide 0.158
- covariates: stand_frac_actual 0.242 | spread_frac_actual 0.000 | rsi_frac_actual 0.186 | terrain_levels 5.919 | how_far 4.184 | goal_idx 0.969 | ep len 1785.134 | prints 5000
- stand time frac corrected 0.271 (logged 0.242) | mean reward 76.612 | vloss 0.014
- slow canary (morph manifest): tripod 0.660 | completion 0.960 | tracking 0.726 | falls 4/100 | pitch-fwd share 1.000 | prefall tip p25 -2.849 | walk tip p50 11.066
- step onset (morph manifest, shallow 0.05-0.2): completion 0.920 | falls 8/100 | pitch-fwd share 1.000 | prefall tip p25 -0.829
- obstacle eval (recal2b2w 0.20-0.70): completion 0.950 | tripod 0.578 | falls 5/100
- terrain level 5.919 (lineage 20-s window-1 ~4-6; outside 3-7 = re-derive the promotion scaling)
>>> ENTRY seed2 s2 A15+B ok

## DECISION TABLE — morphology x training-side fixes (P1: formation + walking slots; P2: + 40-s episodes, 10-s holds; stage 2 on recal2b2w with the promotion equilibrium held)
| config | hardware | rung iv 30k falls fwd/step (Δ) | rung v slow tripod / compl / track (old config) | rung v step falls (ratio) | P1 status | P1 slow tripod / compl / track (ratio to golden) | P1 slow falls | P1 step falls (ratio) | P1 obst recal2b2w compl (falls) | P1 reach_obst | P1 field_frac | P1 cov[3] | P1 fail hazard flat / obst (/1k) | P1 terrain level | P1 step prefall tip p25 | P2s1 status | P2s1 slow tripod / compl / track (ratio to golden) | P2s1 slow falls | P2s1 step falls (ratio) | P2s1 obst recal2b2w compl (falls) | P2s1 reach_obst | P2s1 field_frac | P2s1 cov[3] | P2s1 fail hazard flat / obst (/1k) | P2s1 terrain level | P2s1 step prefall tip p25 | P2s2 status | P2s2 slow tripod / compl / track (ratio to golden) | P2s2 slow falls | P2s2 step falls (ratio) | P2s2 obst recal2b2w compl (falls) | P2s2 reach_obst | P2s2 field_frac | P2s2 cov[3] | P2s2 fail hazard flat / obst (/1k) | P2s2 terrain level | P2s2 step prefall tip p25 | seed-2 P2s2 slow / step / obst | user preference |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| base | none | 100 / 74 | 0.537 (0.90×) / 0.83 (0.94×) / 0.426 (1.07×) | 69/100 (1.77×) | ok | 0.444 / 0.46 / 0.596 (—, —, —) | 54/100 | 71/100 (—) | 0.39 (61/100) | 0.635 | 0.306 | 0.008 | 0.278 / 0.423 | 0.97 | -2.8° | ok | 0.418 / 0.28 / 0.696 (—, —, —) | 72/100 | 91/100 (—) | 0.13 (87/100) | 0.532 | 0.217 | 0.073 | 0.812 / 1.406 | 0.97 | -3.1° | ok | 0.520 / 0.68 / 0.625 (—, —, —) | 32/100 | 67/100 (—) | 0.31 (69/100) | 0.741 | 0.376 | 0.251 | 0.330 / 0.795 | 5.83 | -2.3° | — | — |
| B | re-hinge 3 in | 40 (-60) / 30 (-44) | 0.256 (0.43×) / 0.31 (0.35×) / 0.534 (1.35×) | 69/100 (1.77×) | ok | 0.377 / 0.82 / 0.685 (0.85×, 1.78×, 1.15×) | 18/100 | 46/100 (0.65×) | 0.54 (46/100) | 0.684 | 0.305 | 0.021 | 0.128 / 0.560 (golden 0.278 / 0.423) | 0.97 | -1.7° | ok | 0.382 / 0.79 / 0.642 (0.91×, 2.82×, 0.92×) | 21/100 | 32/100 (0.35×) | 0.67 (33/100) | 0.893 | 0.563 | 0.369 | 0.049 / 0.166 (golden 0.812 / 1.406) | 0.97 | -2.1° | ok | 0.455 / 0.90 / 0.622 (0.87×, 1.32×, 0.99×) | 10/100 | 14/100 (0.21×) | 0.83 (17/100) | 0.927 | 0.612 | 0.532 | 0.039 / 0.173 (golden 0.330 / 0.795) | 5.75 | 1.2° | 0.96 / 28 / 0.69 | re-hinge |
| A10 | 10° shims | 33 (-67) / 21 (-53) | 0.145 (0.24×) / 0.23 (0.26×) / 0.685 (1.73×) | 90/100 (2.31×) | ok | 0.370 / 0.57 / 0.685 (0.83×, 1.24×, 1.15×) | 43/100 | 84/100 (1.18×) | 0.15 (85/100) | 0.582 | 0.279 | 0.009 | 0.414 / 0.995 (golden 0.278 / 0.423) | 0.97 | -1.9° | pending | — | — | — | — | — | — | — | — | — | — | pending | — | — | — | — | — | — | — | — | — | — | — | splay only ✓ |
| A15 | 15° shims | 28 (-72) / 6 (-68) | 0.243 (0.41×) / 0.46 (0.52×) / 0.525 (1.32×) | 68/100 (1.74×) | ok | 0.494 / 0.90 / 0.648 (1.11×, 1.96×, 1.09×) | 10/100 | 58/100 (0.82×) | 0.49 (51/100) | 0.744 | 0.343 | 0.023 | 0.071 / 0.508 (golden 0.278 / 0.423) | 0.97 | 1.3° | ok | 0.494 / 0.78 / 0.595 (1.18×, 2.79×, 0.86×) | 22/100 | 55/100 (0.60×) | 0.48 (52/100) | 0.849 | 0.534 | 0.320 | 0.082 / 0.271 (golden 0.812 / 1.406) | 0.97 | 4.4° | ok | 0.540 / 1.00 / 0.584 (1.04×, 1.47×, 0.93×) | 0/100 | 26/100 (0.39×) | 0.67 (33/100) | 0.858 | 0.525 | 0.401 | 0.079 / 0.388 (golden 0.330 / 0.795) | 6.03 | -0.9° | — | splay only ✓ |
| A20 | 20° shims | 31 (-69) / 7 (-67) | 0.319 (0.54×) / 0.76 (0.86×) / 0.707 (1.78×) | 66/100 (1.69×) | ok | 0.466 / 0.50 / 0.617 (1.05×, 1.09×, 1.03×) | 50/100 | 67/100 (0.94×) | 0.34 (66/100) | 0.732 | 0.354 | 0.023 | 0.141 / 0.493 (golden 0.278 / 0.423) | 0.97 | 1.7° | pending | — | — | — | — | — | — | — | — | — | — | pending | — | — | — | — | — | — | — | — | — | — | — | splay only ✓ |
| A10+B | 10° shims + re-hinge | 24 (-76) / 2 (-72) | — (—) / — (—) / — (—) | — | ok | 0.399 / 0.44 / 0.615 (0.90×, 0.96×, 1.03×) | 56/100 | 79/100 (1.11×) | 0.29 (71/100) | 0.624 | 0.298 | 0.027 | 0.202 / 0.877 (golden 0.278 / 0.423) | 0.97 | 0.9° | pending | — | — | — | — | — | — | — | — | — | — | pending | — | — | — | — | — | — | — | — | — | — | — | shims + re-hinge |
| A15+B | 15° shims + re-hinge | 22 (-78) / 0 (-74) | — (—) / — (—) / — (—) | — | ok | 0.564 / 1.00 / 0.753 (1.27×, 2.17×, 1.26×) | 0/100 | 5/100 (0.07×) | 0.91 (9/100) | 0.815 | 0.423 | 0.086 | 0.031 / 0.061 (golden 0.278 / 0.423) | 0.97 | 6.0° | ok | 0.539 / 0.86 / 0.676 (1.29×, 3.07×, 0.97×) | 14/100 | 31/100 (0.34×) | 0.68 (32/100) | 0.860 | 0.522 | 0.326 | 0.128 / 0.338 (golden 0.812 / 1.406) | 0.97 | -1.4° | ok | 0.589 / 0.95 / 0.622 (1.13×, 1.40×, 0.99×) | 5/100 | 23/100 (0.34×) | 0.73 (27/100) | 0.915 | 0.599 | 0.487 | 0.064 / 0.320 (golden 0.330 / 0.795) | 6.03 | 1.1° | 0.96 / 8 / 0.95 | shims + re-hinge |
| A20+B | 20° shims + re-hinge | 19 (-81) / 1 (-73) | — (—) / — (—) / — (—) | — | ok | 0.562 / 0.98 / 0.686 (1.27×, 2.13×, 1.15×) | 2/100 | 46/100 (0.65×) | 0.51 (49/100) | 0.773 | 0.428 | 0.064 | 0.145 / 0.505 (golden 0.278 / 0.423) | 0.97 | -0.4° | pending | — | — | — | — | — | — | — | — | — | — | pending | — | — | — | — | — | — | — | — | — | — | — | shims + re-hinge |
- every arm from scratch, seed 3; ratios are variant / the same protocol's golden arm; 0.85× is the canary reference line; rung-v noise floor ≈ 0.06 tripod / 0.05 completion / 5 falls per 100
- rung iv / rung v columns carried from the morphology campaign (stop2_table.csv, stop3_decision_table.csv; rung v = old formation config)
- fail hazards are per 1000 env steps (episode-length neutral); P2 stage-2 terrain level should sit near the lineage's 4–6 or the promotion scaling is re-derived
>>> ENTRY decision table

## CAMPAIGN COMPLETE — 2026-09-06 20:51 (all GPU work done; the hardware decision is the user's)
### Two-seed envelope at 10k (P2 stage 2: 40-s episodes, obstacle window on recal2b2w at terrain level 5.7–6.0)
| plant | hardware | canary completion / falls | step-onset falls | obstacle eval recal2b2w | training hazard flat / obstacle (/1k steps) | cov[3] |
|---|---|---|---|---|---|---|
| golden (1 seed) | none | 0.68 / 32 | 67 | 0.31 | 0.33 / 0.80 | 0.25 |
| B (2 seeds) | re-hinge 2.5 in | 0.90–0.96 / 4–10 | 14–28 | 0.69–0.83 | 0.04–0.07 / 0.17–0.43 | 0.35–0.53 |
| A15+B (2 seeds) | 15° shims + re-hinge | 0.95–0.96 / 4–5 | 8–23 | 0.73–0.95 | 0.06 / 0.16–0.32 | 0.49–0.64 |
| A15 (1 seed) | 15° shims | 1.00 / 0 | 26 | 0.67 | 0.08 / 0.39 | 0.40 |
### Verdicts against the plan's success criteria
- **Training-side fix confirmed**: with walking slots from iteration 0 the wide plants form robust gaits from scratch (P1: B 0.82, A15 0.90, A15+B 1.00 canary completion vs rung v 0.31 / 0.46 / —); the golden plant does not (0.46).
- **Morphology advantage in training**: at matched curriculum equilibrium every wide plant beats golden by far more than the noise floor on flat-tile and obstacle-tile hazards, onset falls, and both obstacle evals, on both seeds where run.
- **Longer episodes judged**: 40-s episodes with 10-s holds are usable with the promotion fractions scaled by 20/40 (terrain level 5.7–6.0 on every stage-2 arm). Formation with sustained holds is harder for every plant and seed-sensitive on the wide ones (backward-pitch basin on seed 2 for both B and A15+B at 5k), and the obstacle window repairs it (B 0.29→0.69, A15+B 0.33→0.95).
- **Failure-mode map**: nose-down falls shrink with base width; beyond ~A15/B the 5k policies fall backward (A20, A10+B; seed-2 stage 1 of B and A15+B); no roll falls anywhere.
- Exposure target of the closed obstacle-exposure campaign met in training on all wide plants at level ~6 (cov[3] 0.35–0.64, ~1 obstacle goal per episode, field time 0.5–0.66).
### Standing recommendation (for the user's decision)
A15+B has the stronger two-seed envelope on every obstacle column and the best gait scores; B is the simpler hardware change (re-hinge only) and still 2–3× better than golden. A15 (shims only, one seed) is competitive on the flat canary but 0.16–0.28 behind on obstacles.
>>> ENTRY campaign complete

