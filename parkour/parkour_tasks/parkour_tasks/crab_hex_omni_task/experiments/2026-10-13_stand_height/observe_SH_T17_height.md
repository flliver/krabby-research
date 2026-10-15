# Stand-height observe pass — `SH_T17`

Stand-height observe pass (sh_observe_read.py, a wrapper of sn_observe_read.py): stand rows = p2r_stand_mode (who acted: slot | restoring); RESIDUAL FRAME -- applied = clip(sh_a_net, +-1) (the executed action minus the posture q(h_lim)), R = sh_rh_target - sh_q18 (R_h minus the same posture); at 1.064 (q = 0) this is the predecessor's read exactly. Families: zero = walkstop + suite (zero command), height = hwalkstop + hraise (scenarios_sh.yaml).

Generated 2026-10-14T21:21:53-04:00 by `sh_observe_read.py` (sha 7f1419535d40); status **complete**. Reported, not gated (PLAN D6). Applied = clip(stand-net action, ±1); R = D3 constants on the same row.
Arm: stand ckpt `99c377a17b1b` (sh_dagger), stand_mask 1, S1 `968cc5509899`.

| row | status | stamp | episodes | stands | stand rows | LIFT onsets |
|---|---|---|---|---|---|---|
| hwalkstop@1 | READ | 2026-10-15_00-12-36 | 100 | 700 | 506214 | 3374 |
| hwalkstop@2 | READ | 2026-10-15_00-19-27 | 100 | 700 | 506185 | 2248 |
| hraise@1 | READ | 2026-10-15_00-26-16 | 100 | 200 | 389900 | 664 |
| hraise@2 | READ | 2026-10-15_00-30-44 | 100 | 200 | 389900 | 648 |

## Headline (per row; pooled = every READ row)

| block | μ_hip LIFT (R -0.5; Z2d -0.054) | LIFT hip < -0.25 (Z2d 1.9 %) | onset launch ≤ 0.3 s | first onset per stand-leg | HOLD cam slope per leg, min … max (R -1; Z2d −0.04…−0.12) | within-leg slope | R's within-leg slope, same rows | within-leg slope, unsat band |
|---|---|---|---|---|---|---|---|---|
| hwalkstop@1 | -0.470 (n 38668) | 96.8 % | 98.4 % (3321/3374) | 97.8 % (1541/1575) | -0.466 … -0.203 | -0.343 (n 95605) | -0.511 | -0.724 |
| hwalkstop@2 | -0.474 (n 39160) | 97.5 % | 98.2 % (2208/2248) | 98.3 % (1527/1554) | -0.468 … -0.195 | -0.342 (n 90064) | -0.514 | -0.710 |
| hraise@1 | -0.496 (n 4936) | 99.3 % | 99.5 % (661/664) | 99.5 % (430/432) | -0.550 … -0.222 | -0.360 (n 28309) | -0.530 | -0.752 |
| hraise@2 | -0.474 (n 5668) | 94.7 % | 98.8 % (640/648) | 99.3 % (436/439) | -0.613 … -0.300 | -0.388 (n 27903) | -0.526 | -0.765 |
| pooled height | -0.473 (n 88432) | 97.2 % | 98.5 % (6830/6934) | 98.4 % (3934/4000) | -0.434 … -0.212 | -0.351 (n 241881) | -0.517 | -0.726 |
| pooled all | -0.473 (n 88432) | 97.2 % | 98.5 % (6830/6934) | 98.4 % (3934/4000) | -0.434 … -0.212 | -0.351 (n 241881) | -0.517 | -0.726 |

## |applied − R| by R branch (pooled: hwalkstop@1, hwalkstop@2, hraise@1, hraise@2)

| branch | leg-rows | cam mean / p90 | hip mean / p90 | knee mean / p90 | hip signed |
|---|---|---|---|---|---|
| HOLD | 9875160 | 0.007 / 0.014 | 0.000 / 0.000 | 0.000 / 0.000 | -0.000 |
| WAIT | 259204 | 0.012 / 0.026 | 0.013 / 0.000 | 0.013 / 0.000 | -0.013 |
| LIFT | 88432 | 0.009 / 0.023 | 0.054 / 0.121 | 0.054 / 0.121 | 0.027 |
| HOME | 243515 | 0.035 / 0.063 | 0.032 / 0.067 | 0.032 / 0.067 | 0.014 |
| SEAT | 286883 | 0.033 / 0.064 | 0.017 / 0.000 | 0.017 / 0.000 | -0.016 |

LIFT rows: μ_hip -0.473 (pre-clip -0.473, median -0.496), median |c| 0.820 rad. HOME rows (R −0.5 too): μ_hip -0.486, < −0.25 99.2 % (n 243515).

### LIFT-onset launch per leg (hip < −0.25 within 0.3 s; censored = stand ended first)

| leg | onsets | launched | censored | share | already low at onset | latency median (s) | entry LIFTs | entry launched |
|---|---|---|---|---|---|---|---|---|
| FL | 1035 | 1018 | 0 | 98.4 % | 961 | 0.00 | 97 | 99.0 % |
| FR | 1383 | 1361 | 0 | 98.4 % | 1304 | 0.00 | 45 | 88.9 % |
| ML | 1322 | 1300 | 0 | 98.3 % | 1263 | 0.00 | 1 | 100.0 % |
| MR | 910 | 897 | 0 | 98.6 % | 850 | 0.00 | 8 | 100.0 % |
| RL | 1346 | 1325 | 0 | 98.4 % | 1257 | 0.00 | 106 | 99.1 % |
| RR | 938 | 929 | 0 | 99.0 % | 905 | 0.00 | 32 | 96.9 % |
| pooled | 6934 | 6830 | 0 | 98.5 % | 6540 | 0.00 | 289 | 97.2 % |
| pooled, first per stand-leg | 4000 | 3934 | 0 | 98.4 % | 3738 | 0.00 | — | — |

### HOLD cam-trim slope per leg (|c| in [0.03, 0.3] rad; applied cam on c)

| leg | n | slope | intercept | fixed point (rad) | R² | R's slope same rows | slope on [0.03, 0.1] | band share of HOLD |
|---|---|---|---|---|---|---|---|---|
| FL | 28626 | -0.331 | 0.003 | 0.009 | 0.584 | -0.480 | -0.873 | 1.8 % |
| FR | 63691 | -0.339 | -0.010 | -0.030 | 0.582 | -0.516 | -0.650 | 3.9 % |
| ML | 41234 | -0.351 | -0.005 | -0.016 | 0.574 | -0.535 | -0.733 | 2.5 % |
| MR | 26200 | -0.409 | 0.002 | 0.004 | 0.595 | -0.534 | -0.932 | 1.6 % |
| RL | 43217 | -0.212 | -0.002 | -0.009 | 0.469 | -0.542 | -0.385 | 2.7 % |
| RR | 38913 | -0.434 | 0.003 | 0.006 | 0.728 | -0.504 | -0.986 | 2.3 % |
| pooled (within-leg) | 241881 | -0.351 | — | — | 0.594 | -0.517 | -0.726 | 2.4 % |

Self-checks (pooled): routed action = sn_stand_action (max |Δ| 0.0); R's HOLD cam target vs −clip(k_h c, ±trim) max |Δ| 1.2e-07 over 9875160 leg-rows; branch rows ok True. Pre-clip |a| > 1 share: cam 0.0 %, hip 0.0 %, knee 0.0 %.
