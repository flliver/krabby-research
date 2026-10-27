# Stand-height observe pass — `SH_T18`

Stand-height observe pass (sh_observe_read.py, a wrapper of sn_observe_read.py): stand rows = p2r_stand_mode (who acted: slot | restoring); RESIDUAL FRAME -- applied = clip(sh_a_net, +-1) (the executed action minus the posture q(h_lim)), R = sh_rh_target - sh_q18 (R_h minus the same posture); at 1.064 (q = 0) this is the predecessor's read exactly. Families: zero = walkstop + suite (zero command), height = hwalkstop + hraise (scenarios_sh.yaml).

Generated 2026-10-14T21:21:18-04:00 by `sh_observe_read.py` (sha 7f1419535d40); status **complete**. Reported, not gated (PLAN D6). Applied = clip(stand-net action, ±1); R = D3 constants on the same row.
Arm: stand ckpt `6935037a6fa9` (sh_dagger), stand_mask 1, S1 `968cc5509899`.

| row | status | stamp | episodes | stands | stand rows | LIFT onsets |
|---|---|---|---|---|---|---|
| hwalkstop@1 | READ | 2026-10-14_17-07-39 | 100 | 700 | 506289 | 2167 |
| hwalkstop@2 | READ | 2026-10-14_17-14-41 | 100 | 700 | 506325 | 2029 |
| hraise@1 | READ | 2026-10-14_17-21-41 | 100 | 200 | 389900 | 548 |
| hraise@2 | READ | 2026-10-14_17-26-15 | 100 | 200 | 389992 | 590 |

## Headline (per row; pooled = every READ row)

| block | μ_hip LIFT (R -0.5; Z2d -0.054) | LIFT hip < -0.25 (Z2d 1.9 %) | onset launch ≤ 0.3 s | first onset per stand-leg | HOLD cam slope per leg, min … max (R -1; Z2d −0.04…−0.12) | within-leg slope | R's within-leg slope, same rows | within-leg slope, unsat band |
|---|---|---|---|---|---|---|---|---|
| hwalkstop@1 | -0.475 (n 36187) | 96.6 % | 97.3 % (2108/2167) | 97.7 % (1495/1530) | -0.321 … -0.138 | -0.249 (n 145554) | -0.532 | -0.441 |
| hwalkstop@2 | -0.486 (n 36381) | 98.2 % | 97.4 % (1976/2029) | 97.5 % (1500/1538) | -0.303 … -0.140 | -0.249 (n 146023) | -0.529 | -0.441 |
| hraise@1 | -0.492 (n 4509) | 99.1 % | 98.9 % (542/548) | 99.0 % (405/409) | -0.423 … -0.190 | -0.297 (n 42780) | -0.554 | -0.558 |
| hraise@2 | -0.482 (n 4963) | 97.9 % | 97.3 % (574/590) | 98.2 % (429/437) | -0.383 … -0.197 | -0.278 (n 46922) | -0.545 | -0.514 |
| pooled height | -0.481 (n 82040) | 97.5 % | 97.5 % (5200/5334) | 97.8 % (3829/3914) | -0.305 … -0.144 | -0.258 (n 381279) | -0.535 | -0.461 |
| pooled all | -0.481 (n 82040) | 97.5 % | 97.5 % (5200/5334) | 97.8 % (3829/3914) | -0.305 … -0.144 | -0.258 (n 381279) | -0.535 | -0.461 |

## |applied − R| by R branch (pooled: hwalkstop@1, hwalkstop@2, hraise@1, hraise@2)

| branch | leg-rows | cam mean / p90 | hip mean / p90 | knee mean / p90 | hip signed |
|---|---|---|---|---|---|
| HOLD | 9872141 | 0.009 / 0.016 | 0.000 / 0.000 | 0.000 / 0.000 | -0.000 |
| WAIT | 257805 | 0.013 / 0.026 | 0.015 / 0.000 | 0.015 / 0.000 | -0.015 |
| LIFT | 82040 | 0.010 / 0.022 | 0.051 / 0.105 | 0.051 / 0.105 | 0.019 |
| HOME | 241297 | 0.037 / 0.077 | 0.033 / 0.072 | 0.033 / 0.072 | 0.010 |
| SEAT | 301753 | 0.036 / 0.072 | 0.019 / 0.000 | 0.019 / 0.000 | -0.019 |

LIFT rows: μ_hip -0.481 (pre-clip -0.481, median -0.497), median |c| 0.807 rad. HOME rows (R −0.5 too): μ_hip -0.490, < −0.25 99.5 % (n 241297).

### LIFT-onset launch per leg (hip < −0.25 within 0.3 s; censored = stand ended first)

| leg | onsets | launched | censored | share | already low at onset | latency median (s) | entry LIFTs | entry launched |
|---|---|---|---|---|---|---|---|---|
| FL | 937 | 907 | 0 | 96.8 % | 879 | 0.00 | 75 | 100.0 % |
| FR | 1061 | 1043 | 0 | 98.3 % | 1015 | 0.00 | 40 | 97.5 % |
| ML | 935 | 910 | 0 | 97.3 % | 885 | 0.00 | 0 | — |
| MR | 835 | 805 | 0 | 96.4 % | 784 | 0.00 | 8 | 100.0 % |
| RL | 940 | 914 | 0 | 97.2 % | 877 | 0.00 | 86 | 97.7 % |
| RR | 626 | 621 | 0 | 99.2 % | 610 | 0.00 | 36 | 97.2 % |
| pooled | 5334 | 5200 | 0 | 97.5 % | 5050 | 0.00 | 245 | 98.4 % |
| pooled, first per stand-leg | 3914 | 3829 | 0 | 97.8 % | 3712 | 0.00 | — | — |

### HOLD cam-trim slope per leg (|c| in [0.03, 0.3] rad; applied cam on c)

| leg | n | slope | intercept | fixed point (rad) | R² | R's slope same rows | slope on [0.03, 0.1] | band share of HOLD |
|---|---|---|---|---|---|---|---|---|
| FL | 80562 | -0.144 | 0.001 | 0.003 | 0.381 | -0.531 | -0.231 | 4.9 % |
| FR | 41724 | -0.285 | -0.008 | -0.030 | 0.487 | -0.492 | -0.815 | 2.5 % |
| ML | 67674 | -0.305 | -0.009 | -0.028 | 0.518 | -0.544 | -0.603 | 4.1 % |
| MR | 71878 | -0.299 | 0.009 | 0.031 | 0.420 | -0.552 | -0.480 | 4.4 % |
| RL | 44731 | -0.255 | 0.003 | 0.011 | 0.476 | -0.565 | -0.393 | 2.7 % |
| RR | 74710 | -0.302 | 0.004 | 0.012 | 0.626 | -0.542 | -0.484 | 4.4 % |
| pooled (within-leg) | 381279 | -0.258 | — | — | 0.475 | -0.535 | -0.461 | 3.9 % |

Self-checks (pooled): routed action = sn_stand_action (max |Δ| 0.0); R's HOLD cam target vs −clip(k_h c, ±trim) max |Δ| 1.2e-07 over 9872141 leg-rows; branch rows ok True. Pre-clip |a| > 1 share: cam 0.0 %, hip 0.0 %, knee 0.0 %.
