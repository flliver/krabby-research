<!-- paths-note -->
> **Paths note (2026-09-09):** this campaign moved from `sim_fine_tuning/2026-09-06_2130_a15b_lineage/` to `parkour/parkour_tasks/parkour_tasks/crab_hex_forward_task/experiments/2026-09-06_2130_a15b_lineage/`. Absolute paths below (`/home/.../sim_fine_tuning/...`, `parkour/logs/rsl_rl/...`) name raw artifacts that stay on disk untracked; the tracked checkpoint of record is `head/` (see `bundle.yaml`) and the eval summaries are in place / under `evals/`.

# A15+B lineage retrain — REPORT

Reporting contract: blocks terminated by `>>> ENTRY <marker>` markers.

## CAMPAIGN OPEN — 2026-09-06 21:35
- 0→30k from scratch on A15+B (seed 3), schedule of record + validated levers; per-window evals; golden references; seed-2 replay on approval.
>>> ENTRY campaign open

## WINDOWS — window 0 (0k -> 5k, seed 3) — ok
- stack: formation (P2 stage-1) | ramps none
- status ok | checkpoint 2026-09-06_21-32-46/model_4999.pt | smoke fail@2k 0.500 coll@2k 0.000 ep_len@2k 1484.577
- exposure (non-RSI obstacle tiles): reach_edge 0.980 | reach_obst 0.860 | field_frac 0.522 (steps 882.675) | goals_passed 0.613
- obst_coverage[1..6]: 1:0.860 2:0.540 3:0.326 4:0.212 5:0.123 6:0.049
- RSI episodes: reach_edge 1.000 | reach_obst 1.000 | goals_passed 0.000 | coverage 1:1.000 2:1.000 3:1.000 4:0.981 5:0.921 6:0.818
- failure share flat 0.227 (hazard 0.128/1k) | obst 0.479 (hazard 0.338/1k) | obst-RSI 0.426 | obst-spread 0.000 (ep len 0.000) | campaign-wide 0.263
- covariates: stand_frac_actual 0.152 | spread_frac_actual 0.000 | rsi_frac_actual 0.201 | terrain_levels 0.973 | how_far 4.218 | goal_idx 0.909 | ep len 1652.749 | prints 5000
- stand time frac corrected 0.184 (logged 0.152) | mean reward 72.094 | vloss 0.011
- slow canary (morph manifest): tripod 0.539 | completion 0.860 | tracking 0.676 | falls 14/100 | pitch-fwd share 1.000 | prefall tip p25 1.201 | walk tip p50 6.715
- step onset (morph manifest, shallow 0.05-0.2): completion 0.690 | falls 31/100 | pitch-fwd share 0.935 | prefall tip p25 -1.391
- obstacle eval (recal2b2w 0.20-0.70): completion 0.680 | tripod 0.523 | falls 32/100
>>> ENTRY windows w0 ok

## WINDOWS — window 1 (5k -> 10k, seed 3) — ok
- stack: elements <= 1 + recal2b2w + promotion x 20/40 | ramps [('reward_clock_swing_apex', 1.0, 0.5), ('reward_feet_air_time_positive', 0.8, 0.4), ('reward_stride_length', 0.5, 0.25)]
- status ok | checkpoint 2026-09-06_23-59-54/model_9998.pt | smoke fail@2k 0.346 coll@2k -0.048 ep_len@2k 1592.279
- exposure (non-RSI obstacle tiles): reach_edge 0.971 | reach_obst 0.904 | field_frac 0.527 (steps 780.137) | goals_passed 1.154
- obst_coverage[1..6]: 1:0.904 2:0.612 3:0.400 4:0.244 5:0.111 6:0.048
- RSI episodes: reach_edge 1.000 | reach_obst 1.000 | goals_passed 0.000 | coverage 1:1.000 2:1.000 3:1.000 4:0.972 5:0.889 6:0.741
- failure share flat 0.326 (hazard 0.202/1k) | obst 0.684 (hazard 0.562/1k) | obst-RSI 0.549 | obst-spread 0.000 (ep len 0.000) | campaign-wide 0.395
- covariates: stand_frac_actual 0.166 | spread_frac_actual 0.000 | rsi_frac_actual 0.202 | terrain_levels 6.052 | how_far 4.258 | goal_idx 1.009 | ep len 1430.212 | prints 5000
- stand time frac corrected 0.232 (logged 0.166) | mean reward 56.330 | vloss 0.016
- slow canary (morph manifest): tripod 0.513 | completion 0.920 | tracking 0.641 | falls 8/100 | pitch-fwd share 0.625 | prefall tip p25 2.210 | walk tip p50 8.326
- step onset (morph manifest, shallow 0.05-0.2): completion 0.740 | falls 26/100 | pitch-fwd share 0.962 | prefall tip p25 0.589
- obstacle eval (recal2b2w 0.20-0.70): completion 0.560 | tripod 0.429 | falls 44/100
- terrain level 6.05 (curriculum on; lineage band 3-7)
>>> ENTRY windows w1 ok

## WINDOWS — window 2 (10k -> 15k, seed 3) — ok
- stack: elements <= 2 + recal2b2w + promotion x 20/40 | ramps [('reward_clock_swing_apex', 0.5, 0.001), ('reward_feet_air_time_positive', 0.4, 0.001), ('reward_stride_length', 0.25, 0.001)]
- status ok | checkpoint 2026-09-07_02-19-49/model_14997.pt | smoke fail@2k 0.468 coll@2k -0.031 ep_len@2k 1439.559
- exposure (non-RSI obstacle tiles): reach_edge 0.976 | reach_obst 0.918 | field_frac 0.587 (steps 958.180) | goals_passed 1.244
- obst_coverage[1..6]: 1:0.918 2:0.689 3:0.457 4:0.272 5:0.139 6:0.078
- RSI episodes: reach_edge 1.000 | reach_obst 1.000 | goals_passed 0.000 | coverage 1:1.000 2:1.000 3:1.000 4:0.977 5:0.915 6:0.767
- failure share flat 0.269 (hazard 0.158/1k) | obst 0.529 (hazard 0.373/1k) | obst-RSI 0.501 | obst-spread 0.000 (ep len 0.000) | campaign-wide 0.394
- covariates: stand_frac_actual 0.167 | spread_frac_actual 0.000 | rsi_frac_actual 0.193 | terrain_levels 6.141 | how_far 4.202 | goal_idx 0.952 | ep len 1518.114 | prints 5000
- stand time frac corrected 0.220 (logged 0.167) | mean reward 48.255 | vloss 0.021
- slow canary (morph manifest): tripod 0.507 | completion 0.850 | tracking 0.589 | falls 15/100 | pitch-fwd share 1.000 | prefall tip p25 1.038 | walk tip p50 9.871
- step onset (morph manifest, shallow 0.05-0.2): completion 0.630 | falls 37/100 | pitch-fwd share 0.730 | prefall tip p25 0.235
- obstacle eval (recal2b2w 0.20-0.70): completion 0.580 | tripod 0.388 | falls 42/100
- terrain level 6.14 (curriculum on; lineage band 3-7)
>>> ENTRY windows w2 ok

## WINDOWS — window 3 (15k -> 20k, seed 3) — ok
- stack: elements <= 2 + recal2b2w + promotion x 20/40 | ramps [('reward_clock_schedule', 1.0, 0.5)]
- status ok | checkpoint 2026-09-07_04-38-50/model_19996.pt | smoke fail@2k 0.455 coll@2k -0.019 ep_len@2k 1392.633
- exposure (non-RSI obstacle tiles): reach_edge 0.967 | reach_obst 0.889 | field_frac 0.556 (steps 882.637) | goals_passed 1.248
- obst_coverage[1..6]: 1:0.889 2:0.630 3:0.425 4:0.272 5:0.155 6:0.085
- RSI episodes: reach_edge 1.000 | reach_obst 1.000 | goals_passed 0.000 | coverage 1:1.000 2:1.000 3:1.000 4:0.976 5:0.906 6:0.765
- failure share flat 0.320 (hazard 0.195/1k) | obst 0.557 (hazard 0.418/1k) | obst-RSI 0.485 | obst-spread 0.000 (ep len 0.000) | campaign-wide 0.477
- covariates: stand_frac_actual 0.195 | spread_frac_actual 0.000 | rsi_frac_actual 0.200 | terrain_levels 5.754 | how_far 4.383 | goal_idx 1.018 | ep len 1472.548 | prints 5000
- stand time frac corrected 0.265 (logged 0.195) | mean reward 38.114 | vloss 0.019
- slow canary (morph manifest): tripod 0.500 | completion 0.810 | tracking 0.649 | falls 19/100 | pitch-fwd share 0.947 | prefall tip p25 -0.086 | walk tip p50 7.554
- step onset (morph manifest, shallow 0.05-0.2): completion 0.680 | falls 32/100 | pitch-fwd share 0.906 | prefall tip p25 2.843
- obstacle eval (recal2b2w 0.20-0.70): completion 0.660 | tripod 0.490 | falls 34/100
- terrain level 5.75 (curriculum on; lineage band 3-7)
>>> ENTRY windows w3 ok

## WINDOWS — window 4 (20k -> 25k, seed 3) — ok
- stack: elements <= 2 + recal2b2w + promotion x 20/40 | ramps [('reward_clock_schedule', 0.5, 0.2)]
- status ok | checkpoint 2026-09-07_06-57-52/model_24995.pt | smoke fail@2k 0.452 coll@2k -0.020 ep_len@2k 1419.218
- exposure (non-RSI obstacle tiles): reach_edge 0.954 | reach_obst 0.880 | field_frac 0.539 (steps 804.555) | goals_passed 1.169
- obst_coverage[1..6]: 1:0.880 2:0.588 3:0.391 4:0.251 5:0.131 6:0.072
- RSI episodes: reach_edge 1.000 | reach_obst 1.000 | goals_passed 0.000 | coverage 1:1.000 2:1.000 3:1.000 4:0.971 5:0.889 6:0.730
- failure share flat 0.390 (hazard 0.252/1k) | obst 0.663 (hazard 0.532/1k) | obst-RSI 0.565 | obst-spread 0.000 (ep len 0.000) | campaign-wide 0.532
- covariates: stand_frac_actual 0.120 | spread_frac_actual 0.000 | rsi_frac_actual 0.198 | terrain_levels 6.206 | how_far 4.115 | goal_idx 1.032 | ep len 1350.029 | prints 5000
- stand time frac corrected 0.178 (logged 0.120) | mean reward 33.088 | vloss 0.018
- slow canary (morph manifest): tripod 0.371 | completion 0.690 | tracking 0.633 | falls 31/100 | pitch-fwd share 0.806 | prefall tip p25 -0.237 | walk tip p50 7.947
- step onset (morph manifest, shallow 0.05-0.2): completion 0.420 | falls 58/100 | pitch-fwd share 0.672 | prefall tip p25 0.831
- obstacle eval (recal2b2w 0.20-0.70): completion 0.290 | tripod 0.334 | falls 71/100
- terrain level 6.21 (curriculum on; lineage band 3-7)
>>> ENTRY windows w4 ok

## WINDOWS — window 5 (25k -> 30k, seed 3) — ok
- stack: elements <= 2 + recal2b2w + promotion x 20/40 | ramps [('reward_clock_schedule', 0.2, 0.001)]
- status ok | checkpoint 2026-09-07_09-16-39/model_29994.pt | smoke fail@2k 0.483 coll@2k -0.013 ep_len@2k 1431.110
- exposure (non-RSI obstacle tiles): reach_edge 0.979 | reach_obst 0.899 | field_frac 0.564 (steps 869.172) | goals_passed 1.331
- obst_coverage[1..6]: 1:0.899 2:0.651 3:0.474 4:0.327 5:0.180 6:0.093
- RSI episodes: reach_edge 1.000 | reach_obst 1.000 | goals_passed 0.000 | coverage 1:1.000 2:1.000 3:1.000 4:0.981 5:0.911 6:0.772
- failure share flat 0.429 (hazard 0.294/1k) | obst 0.585 (hazard 0.452/1k) | obst-RSI 0.510 | obst-spread 0.000 (ep len 0.000) | campaign-wide 0.543
- covariates: stand_frac_actual 0.156 | spread_frac_actual 0.000 | rsi_frac_actual 0.202 | terrain_levels 5.848 | how_far 4.441 | goal_idx 1.058 | ep len 1344.389 | prints 5000
- stand time frac corrected 0.233 (logged 0.156) | mean reward 29.037 | vloss 0.017
- slow canary (morph manifest): tripod 0.370 | completion 0.590 | tracking 0.695 | falls 41/100 | pitch-fwd share 0.976 | prefall tip p25 1.234 | walk tip p50 7.993
- step onset (morph manifest, shallow 0.05-0.2): completion 0.600 | falls 40/100 | pitch-fwd share 0.950 | prefall tip p25 -0.946
- obstacle eval (recal2b2w 0.20-0.70): completion 0.480 | tripod 0.357 | falls 52/100
- terrain level 5.85 (curriculum on; lineage band 3-7)
>>> ENTRY windows w5 ok

## A15+B LINEAGE (seed 3) — per-window table with golden references
| window | iters | canary tripod / compl / falls | step falls / compl | obst recal2b2w (falls) | hazard flat / obst (/1k) | reach_obst / cov3 / cov6 | level |
|---|---|---|---|---|---|---|---|
| w0 | 5k | 0.539 / 0.860 / 14 | 31 / 0.690 | 0.680 (32) | 0.128 / 0.338 | 0.860 / 0.326 / 0.049 | 0.97 |
| w1 | 10k | 0.513 / 0.920 / 8 | 26 / 0.740 | 0.560 (44) | 0.202 / 0.562 | 0.904 / 0.400 / 0.048 | 6.05 |
| w2 | 15k | 0.507 / 0.850 / 15 | 37 / 0.630 | 0.580 (42) | 0.158 / 0.373 | 0.918 / 0.457 / 0.078 | 6.14 |
| w3 | 20k | 0.500 / 0.810 / 19 | 32 / 0.680 | 0.660 (34) | 0.195 / 0.418 | 0.889 / 0.425 / 0.085 | 5.75 |
| w4 | 25k | 0.371 / 0.690 / 31 | 58 / 0.420 | 0.290 (71) | 0.252 / 0.532 | 0.880 / 0.391 / 0.072 | 6.21 |
| w5 | 30k | 0.370 / 0.590 / 41 | 40 / 0.600 | 0.480 (52) | 0.294 / 0.452 | 0.899 / 0.474 / 0.093 | 5.85 |
| golden_30k (golden plant) | — | 0.580 / 0.810 / 19 | 74 / 0.260 | 0.340 (66) | — | — | — |
| golden_20k (golden plant) | — | 0.556 / 1.000 / 0 | 42 / 0.580 | 0.600 (40) | — | — | — |
>>> ENTRY lineage table seed3

## BAKE — 2026-09-07 (user decision): A15+B policy of record = window-3 head (20k)
- checkpoint: `2026-09-07_04-38-50/model_19996.pt` (seed 3; plant `assets/variants/crab_simple__splay15_axis2p5in.usda`)
- schedule baked for A15+B: windows 0–3 (formation w/ walking slots + 40-s episodes; elements @5k/@10k; apex/airtime/stride → half @5k → ε @10k; clock 1.0→0.5 @15k–20k; recal2b2w; promotion 0.225:0.125; P0-null RSI 0.2). Clock anneal beyond 0.5 not baked.
- evals (A15+B plant): slow canary 0.500 / 0.81 / 19 falls | step onset 32 falls, completion 0.68 | recal2b2w obstacle eval 0.66 (34 falls)
- golden 20k head of record (golden plant, same scenarios): 0.556 / 1.00 / 0 | 42 (0.58) | 0.60 (40)
- waived (user): seed-2 lineage replay; anneal-hold variant 20k→30k.
>>> ENTRY bake a15b 20k

## CAMPAIGN CLOSE — 2026-09-07 12:30 (user decision)
- **Outcome:** A15+B lineage retrained 0→30k from scratch (seed 3). **Baked:** the 20k head `2026-09-07_04-38-50/model_19996.pt` as the A15+B policy of record (schedule = windows 0–3). The 20k–30k clock anneal is not baked: obstacle eval 0.66 @20k → 0.29 @25k → 0.48 @30k, the same late-window collapse the golden lineage showed.
- **Versus golden at matched iteration (same three scenarios, each on its own plant):** 20k — onset falls 32 vs 42, obstacle eval 0.66 vs 0.60, canary 0.81 vs 1.00; 30k — 40 vs 74, 0.48 vs 0.34, 0.59 vs 0.81. The wide plant wins on obstacles and onsets and trails on the flat canary.
- **Strongest A15+B checkpoints measured anywhere:** the morphology campaign's P2 10k heads trained without the gait-income ramps (seed 3: canary 0.95 / obstacles 0.73; seed 2: 0.96 / 0.95, 8 onset falls) — recorded for future reference; the user baked this lineage's 20k head.
- **Waived (user):** seed-2 lineage replay; the anneal-hold variant 20k→30k.
- **Deliverables:** per-window REPORT blocks with exposure/hazard telemetry and the three evals; golden reference evals; `videos/a15b_20k_{flat,light,medium,tough}.mp4`; orchestrator `run_lineage.py` (window stacks reusable for any plant).
>>> ENTRY campaign close

