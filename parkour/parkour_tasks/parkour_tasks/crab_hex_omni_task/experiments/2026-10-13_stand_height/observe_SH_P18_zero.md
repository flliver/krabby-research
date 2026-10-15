# Stand-height observe pass — `SH_P18z`

Stand-height observe pass (sh_observe_read.py, a wrapper of sn_observe_read.py): stand rows = p2r_stand_mode (who acted: slot | restoring); RESIDUAL FRAME -- applied = clip(sh_a_net, +-1) (the executed action minus the posture q(h_lim)), R = sh_rh_target - sh_q18 (R_h minus the same posture); at 1.064 (q = 0) this is the predecessor's read exactly. Families: zero = walkstop + suite (zero command), height = hwalkstop + hraise (scenarios_sh.yaml).

Generated 2026-10-14T21:22:21-04:00 by `sh_observe_read.py` (sha 7f1419535d40); status **complete**. Reported, not gated (PLAN D6). Applied = clip(stand-net action, ±1); R = D3 constants on the same row.
Arm: stand ckpt `50b7c60a7e2b` (sh_dagger), stand_mask 1, S1 `968cc5509899`.

| row | status | stamp | episodes | stands | stand rows | LIFT onsets |
|---|---|---|---|---|---|---|
| walkstop@1 | READ | 2026-10-14_05-31-35 | 100 | 300 | 180000 | 934 |
| walkstop@2 | READ | 2026-10-14_05-34-19 | 100 | 300 | 180000 | 1022 |
| suite@1 | READ | 2026-10-14_05-16-50 | 1000 | 1000 | 235000 | 0 |
| suite@2 | READ | 2026-10-14_05-24-12 | 1000 | 1000 | 235000 | 0 |

## Headline (per row; pooled = every READ row)

| block | μ_hip LIFT (R -0.5; Z2d -0.054) | LIFT hip < -0.25 (Z2d 1.9 %) | onset launch ≤ 0.3 s | first onset per stand-leg | HOLD cam slope per leg, min … max (R -1; Z2d −0.04…−0.12) | within-leg slope | R's within-leg slope, same rows | within-leg slope, unsat band |
|---|---|---|---|---|---|---|---|---|
| walkstop@1 | -0.480 (n 6710) | 98.6 % | 98.3 % (918/934) | 98.7 % (622/630) | -0.087 … -0.008 | -0.015 (n 116497) | -0.514 | -0.044 |
| walkstop@2 | -0.474 (n 7339) | 98.8 % | 99.0 % (1012/1022) | 100.0 % (665/665) | -0.022 … -0.009 | -0.012 (n 112012) | -0.514 | -0.037 |
| suite@1 | — (n 0) | — | — (0/0) | — (0/0) | -0.032 … -0.014 | -0.015 (n 14892) | -1.000 | -0.015 |
| suite@2 | — (n 0) | — | — (0/0) | — (0/0) | -0.015 … 0.098 | 0.010 (n 14517) | -1.000 | 0.010 |
| pooled zero | -0.477 (n 14049) | 98.7 % | 98.7 % (1930/1956) | 99.4 % (1287/1295) | -0.060 … -0.008 | -0.014 (n 257918) | -0.515 | -0.040 |
| pooled all | -0.477 (n 14049) | 98.7 % | 98.7 % (1930/1956) | 99.4 % (1287/1295) | -0.060 … -0.008 | -0.014 (n 257918) | -0.515 | -0.040 |

## |applied − R| by R branch (pooled: walkstop@1, walkstop@2, suite@1, suite@2)

| branch | leg-rows | cam mean / p90 | hip mean / p90 | knee mean / p90 | hip signed |
|---|---|---|---|---|---|
| HOLD | 4720942 | 0.011 / 0.022 | 0.015 / 0.029 | 0.015 / 0.029 | -0.005 |
| WAIT | 70275 | 0.004 / 0.010 | 0.037 / 0.079 | 0.037 / 0.079 | -0.024 |
| LIFT | 14049 | 0.004 / 0.008 | 0.042 / 0.093 | 0.042 / 0.093 | 0.023 |
| HOME | 83394 | 0.028 / 0.050 | 0.025 / 0.055 | 0.025 / 0.055 | 0.008 |
| SEAT | 91340 | 0.027 / 0.053 | 0.061 / 0.155 | 0.061 / 0.155 | -0.027 |

LIFT rows: μ_hip -0.477 (pre-clip -0.477, median -0.488), median |c| 0.503 rad. HOME rows (R −0.5 too): μ_hip -0.492, < −0.25 99.8 % (n 83394).

### LIFT-onset launch per leg (hip < −0.25 within 0.3 s; censored = stand ended first)

| leg | onsets | launched | censored | share | already low at onset | latency median (s) | entry LIFTs | entry launched |
|---|---|---|---|---|---|---|---|---|
| FL | 265 | 264 | 0 | 99.6 % | 255 | 0.00 | 24 | 100.0 % |
| FR | 498 | 491 | 0 | 98.6 % | 477 | 0.00 | 31 | 100.0 % |
| ML | 344 | 336 | 0 | 97.7 % | 322 | 0.00 | 0 | — |
| MR | 223 | 221 | 0 | 99.1 % | 219 | 0.00 | 1 | 100.0 % |
| RL | 317 | 315 | 0 | 99.4 % | 306 | 0.00 | 33 | 100.0 % |
| RR | 309 | 303 | 0 | 98.1 % | 295 | 0.00 | 45 | 100.0 % |
| pooled | 1956 | 1930 | 0 | 98.7 % | 1874 | 0.00 | 134 | 100.0 % |
| pooled, first per stand-leg | 1295 | 1287 | 0 | 99.4 % | 1246 | 0.00 | — | — |

### HOLD cam-trim slope per leg (|c| in [0.03, 0.3] rad; applied cam on c)

| leg | n | slope | intercept | fixed point (rad) | R² | R's slope same rows | slope on [0.03, 0.1] | band share of HOLD |
|---|---|---|---|---|---|---|---|---|
| FL | 30458 | -0.014 | 0.001 | 0.038 | 0.207 | -0.524 | -0.045 | 3.9 % |
| FR | 9978 | -0.060 | -0.006 | -0.102 | 0.120 | -0.435 | -0.452 | 1.3 % |
| ML | 57300 | -0.009 | 0.000 | 0.033 | 0.042 | -0.667 | -0.015 | 7.2 % |
| MR | 47013 | -0.008 | -0.000 | -0.003 | 0.189 | -0.463 | -0.026 | 6.0 % |
| RL | 85041 | -0.021 | 0.001 | 0.027 | 0.320 | -0.482 | -0.037 | 10.8 % |
| RR | 28128 | -0.010 | 0.001 | 0.064 | 0.089 | -0.510 | -0.049 | 3.5 % |
| pooled (within-leg) | 257918 | -0.014 | — | — | 0.080 | -0.515 | -0.040 | 5.5 % |

Self-checks (pooled): routed action = sn_stand_action (max |Δ| 0.0); R's HOLD cam target vs −clip(k_h c, ±trim) max |Δ| 1.2e-07 over 4720942 leg-rows; branch rows ok True. Pre-clip |a| > 1 share: cam 0.3 %, hip 0.0 %, knee 0.0 %.
