# Stand-height observe pass — `SH_T18z`

Stand-height observe pass (sh_observe_read.py, a wrapper of sn_observe_read.py): stand rows = p2r_stand_mode (who acted: slot | restoring); RESIDUAL FRAME -- applied = clip(sh_a_net, +-1) (the executed action minus the posture q(h_lim)), R = sh_rh_target - sh_q18 (R_h minus the same posture); at 1.064 (q = 0) this is the predecessor's read exactly. Families: zero = walkstop + suite (zero command), height = hwalkstop + hraise (scenarios_sh.yaml).

Generated 2026-10-14T21:21:25-04:00 by `sh_observe_read.py` (sha 7f1419535d40); status **complete**. Reported, not gated (PLAN D6). Applied = clip(stand-net action, ±1); R = D3 constants on the same row.
Arm: stand ckpt `6935037a6fa9` (sh_dagger), stand_mask 1, S1 `968cc5509899`.

| row | status | stamp | episodes | stands | stand rows | LIFT onsets |
|---|---|---|---|---|---|---|
| walkstop@1 | READ | 2026-10-14_17-45-37 | 100 | 300 | 180000 | 963 |
| walkstop@2 | READ | 2026-10-14_17-48-21 | 100 | 300 | 180000 | 916 |
| suite@1 | READ | 2026-10-14_17-30-47 | 1000 | 1000 | 235000 | 0 |
| suite@2 | READ | 2026-10-14_17-38-12 | 1000 | 1000 | 235000 | 0 |

## Headline (per row; pooled = every READ row)

| block | μ_hip LIFT (R -0.5; Z2d -0.054) | LIFT hip < -0.25 (Z2d 1.9 %) | onset launch ≤ 0.3 s | first onset per stand-leg | HOLD cam slope per leg, min … max (R -1; Z2d −0.04…−0.12) | within-leg slope | R's within-leg slope, same rows | within-leg slope, unsat band |
|---|---|---|---|---|---|---|---|---|
| walkstop@1 | -0.471 (n 7799) | 97.1 % | 98.4 % (948/963) | 98.4 % (628/638) | -0.374 … -0.143 | -0.224 (n 67528) | -0.522 | -0.479 |
| walkstop@2 | -0.481 (n 7597) | 98.2 % | 98.8 % (905/916) | 98.7 % (602/610) | -0.343 … -0.151 | -0.232 (n 65679) | -0.531 | -0.476 |
| suite@1 | — (n 0) | — | — (0/0) | — (0/0) | — … — | — (n 0) | — | — |
| suite@2 | — (n 0) | — | — (0/0) | — (0/0) | — … — | — (n 0) | — | — |
| pooled zero | -0.476 (n 15396) | 97.7 % | 98.6 % (1853/1879) | 98.6 % (1230/1248) | -0.360 … -0.147 | -0.228 (n 133207) | -0.527 | -0.477 |
| pooled all | -0.476 (n 15396) | 97.7 % | 98.6 % (1853/1879) | 98.6 % (1230/1248) | -0.360 … -0.147 | -0.228 (n 133207) | -0.527 | -0.477 |

## |applied − R| by R branch (pooled: walkstop@1, walkstop@2, suite@1, suite@2)

| branch | leg-rows | cam mean / p90 | hip mean / p90 | knee mean / p90 | hip signed |
|---|---|---|---|---|---|
| HOLD | 4746428 | 0.007 / 0.011 | 0.000 / 0.000 | 0.000 / 0.000 | -0.000 |
| WAIT | 70239 | 0.012 / 0.025 | 0.016 / 0.000 | 0.016 / 0.000 | -0.016 |
| LIFT | 15396 | 0.009 / 0.019 | 0.057 / 0.122 | 0.057 / 0.122 | 0.024 |
| HOME | 64504 | 0.034 / 0.070 | 0.030 / 0.063 | 0.030 / 0.063 | 0.008 |
| SEAT | 83433 | 0.035 / 0.074 | 0.017 / 0.000 | 0.017 / 0.000 | -0.016 |

LIFT rows: μ_hip -0.476 (pre-clip -0.476, median -0.496), median |c| 0.520 rad. HOME rows (R −0.5 too): μ_hip -0.492, < −0.25 99.4 % (n 64504).

### LIFT-onset launch per leg (hip < −0.25 within 0.3 s; censored = stand ended first)

| leg | onsets | launched | censored | share | already low at onset | latency median (s) | entry LIFTs | entry launched |
|---|---|---|---|---|---|---|---|---|
| FL | 253 | 247 | 0 | 97.6 % | 241 | 0.00 | 26 | 100.0 % |
| FR | 393 | 390 | 0 | 99.2 % | 385 | 0.00 | 28 | 100.0 % |
| ML | 313 | 311 | 0 | 99.4 % | 306 | 0.00 | 1 | 100.0 % |
| MR | 204 | 202 | 0 | 99.0 % | 195 | 0.00 | 3 | 100.0 % |
| RL | 254 | 247 | 0 | 97.2 % | 236 | 0.00 | 34 | 100.0 % |
| RR | 462 | 456 | 0 | 98.7 % | 449 | 0.00 | 28 | 96.4 % |
| pooled | 1879 | 1853 | 0 | 98.6 % | 1812 | 0.00 | 120 | 99.2 % |
| pooled, first per stand-leg | 1248 | 1230 | 0 | 98.6 % | 1205 | 0.00 | — | — |

### HOLD cam-trim slope per leg (|c| in [0.03, 0.3] rad; applied cam on c)

| leg | n | slope | intercept | fixed point (rad) | R² | R's slope same rows | slope on [0.03, 0.1] | band share of HOLD |
|---|---|---|---|---|---|---|---|---|
| FL | 35618 | -0.147 | -0.001 | -0.005 | 0.377 | -0.547 | -0.290 | 4.5 % |
| FR | 7296 | -0.287 | -0.020 | -0.069 | 0.456 | -0.458 | -1.038 | 0.9 % |
| ML | 12908 | -0.299 | -0.004 | -0.014 | 0.522 | -0.520 | -0.671 | 1.6 % |
| MR | 31852 | -0.242 | 0.008 | 0.031 | 0.536 | -0.526 | -0.550 | 4.0 % |
| RL | 28325 | -0.204 | -0.004 | -0.017 | 0.475 | -0.513 | -0.459 | 3.6 % |
| RR | 17208 | -0.360 | 0.003 | 0.007 | 0.663 | -0.553 | -0.502 | 2.2 % |
| pooled (within-leg) | 133207 | -0.228 | — | — | 0.474 | -0.527 | -0.477 | 2.8 % |

Self-checks (pooled): routed action = sn_stand_action (max |Δ| 0.0); R's HOLD cam target vs −clip(k_h c, ±trim) max |Δ| 1.2e-07 over 4746428 leg-rows; branch rows ok True. Pre-clip |a| > 1 share: cam 0.0 %, hip 0.0 %, knee 0.0 %.
