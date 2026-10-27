# Stand-height observe pass — `SH_T17z`

Stand-height observe pass (sh_observe_read.py, a wrapper of sn_observe_read.py): stand rows = p2r_stand_mode (who acted: slot | restoring); RESIDUAL FRAME -- applied = clip(sh_a_net, +-1) (the executed action minus the posture q(h_lim)), R = sh_rh_target - sh_q18 (R_h minus the same posture); at 1.064 (q = 0) this is the predecessor's read exactly. Families: zero = walkstop + suite (zero command), height = hwalkstop + hraise (scenarios_sh.yaml).

Generated 2026-10-14T21:22:01-04:00 by `sh_observe_read.py` (sha 7f1419535d40); status **complete**. Reported, not gated (PLAN D6). Applied = clip(stand-net action, ±1); R = D3 constants on the same row.
Arm: stand ckpt `99c377a17b1b` (sh_dagger), stand_mask 1, S1 `968cc5509899`.

| row | status | stamp | episodes | stands | stand rows | LIFT onsets |
|---|---|---|---|---|---|---|
| walkstop@1 | READ | 2026-10-15_00-49-40 | 100 | 300 | 180000 | 800 |
| walkstop@2 | READ | 2026-10-15_00-52-20 | 100 | 300 | 180000 | 782 |
| suite@1 | READ | 2026-10-15_00-35-11 | 1000 | 1000 | 235000 | 0 |
| suite@2 | READ | 2026-10-15_00-42-25 | 1000 | 1000 | 235000 | 0 |

## Headline (per row; pooled = every READ row)

| block | μ_hip LIFT (R -0.5; Z2d -0.054) | LIFT hip < -0.25 (Z2d 1.9 %) | onset launch ≤ 0.3 s | first onset per stand-leg | HOLD cam slope per leg, min … max (R -1; Z2d −0.04…−0.12) | within-leg slope | R's within-leg slope, same rows | within-leg slope, unsat band |
|---|---|---|---|---|---|---|---|---|
| walkstop@1 | -0.477 (n 7571) | 98.2 % | 98.2 % (786/800) | 98.5 % (607/616) | -0.446 … -0.177 | -0.331 (n 43878) | -0.508 | -0.751 |
| walkstop@2 | -0.473 (n 7077) | 97.8 % | 98.1 % (767/782) | 98.5 % (607/616) | -0.474 … -0.170 | -0.328 (n 44869) | -0.506 | -0.731 |
| suite@1 | — (n 0) | — | — (0/0) | — (0/0) | — … — | — (n 0) | — | — |
| suite@2 | — (n 0) | — | — (0/0) | — (0/0) | — … — | — (n 0) | — | — |
| pooled zero | -0.475 (n 14648) | 98.0 % | 98.2 % (1553/1582) | 98.5 % (1214/1232) | -0.462 … -0.173 | -0.330 (n 88747) | -0.508 | -0.742 |
| pooled all | -0.475 (n 14648) | 98.0 % | 98.2 % (1553/1582) | 98.5 % (1214/1232) | -0.462 … -0.173 | -0.330 (n 88747) | -0.508 | -0.742 |

## |applied − R| by R branch (pooled: walkstop@1, walkstop@2, suite@1, suite@2)

| branch | leg-rows | cam mean / p90 | hip mean / p90 | knee mean / p90 | hip signed |
|---|---|---|---|---|---|
| HOLD | 4757079 | 0.004 / 0.008 | 0.000 / 0.000 | 0.000 / 0.000 | -0.000 |
| WAIT | 64948 | 0.013 / 0.029 | 0.011 / 0.000 | 0.011 / 0.000 | -0.011 |
| LIFT | 14648 | 0.008 / 0.017 | 0.053 / 0.116 | 0.053 / 0.116 | 0.025 |
| HOME | 62876 | 0.032 / 0.056 | 0.035 / 0.069 | 0.035 / 0.069 | 0.014 |
| SEAT | 80449 | 0.031 / 0.065 | 0.012 / 0.000 | 0.012 / 0.000 | -0.011 |

LIFT rows: μ_hip -0.475 (pre-clip -0.475, median -0.494), median |c| 0.531 rad. HOME rows (R −0.5 too): μ_hip -0.486, < −0.25 98.8 % (n 62876).

### LIFT-onset launch per leg (hip < −0.25 within 0.3 s; censored = stand ended first)

| leg | onsets | launched | censored | share | already low at onset | latency median (s) | entry LIFTs | entry launched |
|---|---|---|---|---|---|---|---|---|
| FL | 213 | 209 | 0 | 98.1 % | 200 | 0.00 | 24 | 83.3 % |
| FR | 381 | 374 | 0 | 98.2 % | 354 | 0.00 | 24 | 91.7 % |
| ML | 289 | 281 | 0 | 97.2 % | 269 | 0.00 | 1 | 100.0 % |
| MR | 204 | 202 | 0 | 99.0 % | 199 | 0.00 | 2 | 100.0 % |
| RL | 215 | 211 | 0 | 98.1 % | 205 | 0.00 | 31 | 96.8 % |
| RR | 280 | 276 | 0 | 98.6 % | 264 | 0.00 | 28 | 100.0 % |
| pooled | 1582 | 1553 | 0 | 98.2 % | 1491 | 0.00 | 110 | 93.6 % |
| pooled, first per stand-leg | 1232 | 1214 | 0 | 98.5 % | 1166 | 0.00 | — | — |

### HOLD cam-trim slope per leg (|c| in [0.03, 0.3] rad; applied cam on c)

| leg | n | slope | intercept | fixed point (rad) | R² | R's slope same rows | slope on [0.03, 0.1] | band share of HOLD |
|---|---|---|---|---|---|---|---|---|
| FL | 15456 | -0.406 | 0.005 | 0.013 | 0.751 | -0.527 | -0.895 | 1.9 % |
| FR | 8811 | -0.342 | -0.018 | -0.052 | 0.598 | -0.481 | -0.872 | 1.1 % |
| ML | 11980 | -0.302 | -0.000 | -0.001 | 0.535 | -0.512 | -0.787 | 1.5 % |
| MR | 15501 | -0.355 | 0.008 | 0.023 | 0.621 | -0.479 | -1.026 | 2.0 % |
| RL | 28059 | -0.173 | -0.001 | -0.005 | 0.396 | -0.527 | -0.350 | 3.5 % |
| RR | 8940 | -0.462 | 0.002 | 0.005 | 0.746 | -0.520 | -0.978 | 1.1 % |
| pooled (within-leg) | 88747 | -0.330 | — | — | 0.593 | -0.508 | -0.742 | 1.9 % |

Self-checks (pooled): routed action = sn_stand_action (max |Δ| 0.0); R's HOLD cam target vs −clip(k_h c, ±trim) max |Δ| 1.2e-07 over 4757079 leg-rows; branch rows ok True. Pre-clip |a| > 1 share: cam 0.0 %, hip 0.0 %, knee 0.0 %.
