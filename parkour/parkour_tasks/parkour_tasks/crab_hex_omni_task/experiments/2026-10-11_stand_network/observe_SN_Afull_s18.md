# Observe pass — `SN_Afull_s18`

Generated 2026-10-12T14:58:37-04:00 by `sn_observe_read.py` (sha 5c2f79e70b6d); status **complete**. Reported, not gated (PLAN D6). Applied = clip(stand-net action, ±1); R = D3 constants on the same row.
Arm: stand ckpt `1671fd55d2cb` (sn_dagger), stand_mask 0, S1 `968cc5509899`.

| row | status | stamp | episodes | stands | stand rows | LIFT onsets |
|---|---|---|---|---|---|---|
| walkstop@1 | READ | 2026-10-12_18-35-23 | 100 | 300 | 180000 | 1050 |
| walkstop@2 | READ | 2026-10-12_18-38-02 | 100 | 300 | 180000 | 1134 |
| suite@1 | READ | 2026-10-12_18-21-05 | 1000 | 1000 | 235000 | 0 |
| suite@2 | READ | 2026-10-12_18-28-13 | 1000 | 1000 | 235000 | 0 |

## Headline (per row; pooled = every READ row)

| block | μ_hip LIFT (R -0.5; Z2d -0.054) | LIFT hip < -0.25 (Z2d 1.9 %) | onset launch ≤ 0.3 s | first onset per stand-leg | HOLD cam slope per leg, min … max (R -1; Z2d −0.04…−0.12) | within-leg slope | R's within-leg slope, same rows | within-leg slope, unsat band |
|---|---|---|---|---|---|---|---|---|
| walkstop@1 | -0.482 (n 6766) | 99.0 % | 99.0 % (1040/1050) | 99.4 % (651/655) | -0.056 … -0.006 | -0.028 (n 168949) | -0.517 | -0.057 |
| walkstop@2 | -0.477 (n 7841) | 98.3 % | 98.7 % (1119/1134) | 98.9 % (709/717) | -0.059 … -0.008 | -0.029 (n 171338) | -0.518 | -0.058 |
| suite@1 | — (n 0) | — | — (0/0) | — (0/0) | -0.207 … -0.022 | -0.025 (n 114105) | -0.980 | -0.028 |
| suite@2 | — (n 0) | — | — (0/0) | — (0/0) | -0.155 … 0.199 | -0.021 (n 115703) | -0.941 | -0.029 |
| pooled walkstop | -0.479 (n 14607) | 98.6 % | 98.9 % (2159/2184) | 99.1 % (1360/1372) | -0.058 … -0.008 | -0.029 (n 340287) | -0.518 | -0.058 |
| pooled suite | — (n 0) | — | — (0/0) | — (0/0) | -0.155 … -0.019 | -0.023 (n 229808) | -0.960 | -0.028 |
| pooled all | -0.479 (n 14607) | 98.6 % | 98.9 % (2159/2184) | 99.1 % (1360/1372) | -0.058 … -0.007 | -0.028 (n 570095) | -0.526 | -0.052 |

## |applied − R| by R branch (pooled: walkstop@1, walkstop@2, suite@1, suite@2)

| branch | leg-rows | cam mean / p90 | hip mean / p90 | knee mean / p90 | hip signed |
|---|---|---|---|---|---|
| HOLD | 4717432 | 0.015 / 0.034 | 0.014 / 0.029 | 0.014 / 0.029 | 0.008 |
| WAIT | 75571 | 0.007 / 0.014 | 0.039 / 0.080 | 0.040 / 0.081 | -0.022 |
| LIFT | 14607 | 0.005 / 0.011 | 0.041 / 0.094 | 0.041 / 0.094 | 0.021 |
| HOME | 82002 | 0.030 / 0.059 | 0.043 / 0.103 | 0.043 / 0.103 | 0.021 |
| SEAT | 90388 | 0.027 / 0.052 | 0.057 / 0.135 | 0.057 / 0.135 | -0.026 |

LIFT rows: μ_hip -0.479 (pre-clip -0.479, median -0.493), median |c| 0.499 rad. HOME rows (R −0.5 too): μ_hip -0.479, < −0.25 98.8 % (n 82002).

### LIFT-onset launch per leg (hip < −0.25 within 0.3 s; censored = stand ended first)

| leg | onsets | launched | censored | share | already low at onset | latency median (s) | entry LIFTs | entry launched |
|---|---|---|---|---|---|---|---|---|
| FL | 277 | 273 | 0 | 98.6 % | 268 | 0.00 | 18 | 100.0 % |
| FR | 503 | 497 | 0 | 98.8 % | 485 | 0.00 | 32 | 100.0 % |
| ML | 393 | 385 | 0 | 98.0 % | 373 | 0.00 | 0 | — |
| MR | 317 | 314 | 0 | 99.1 % | 299 | 0.00 | 0 | — |
| RL | 323 | 322 | 0 | 99.7 % | 318 | 0.00 | 26 | 100.0 % |
| RR | 371 | 368 | 0 | 99.2 % | 360 | 0.00 | 43 | 100.0 % |
| pooled | 2184 | 2159 | 0 | 98.9 % | 2103 | 0.00 | 119 | 100.0 % |
| pooled, first per stand-leg | 1372 | 1360 | 0 | 99.1 % | 1321 | 0.00 | — | — |

### HOLD cam-trim slope per leg (|c| in [0.03, 0.3] rad; applied cam on c)

| leg | n | slope | intercept | fixed point (rad) | R² | R's slope same rows | slope on [0.03, 0.1] | band share of HOLD |
|---|---|---|---|---|---|---|---|---|
| FL | 24882 | -0.058 | 0.001 | 0.016 | 0.242 | -0.510 | -0.160 | 3.2 % |
| FR | 28559 | -0.020 | 0.001 | 0.067 | 0.125 | -0.468 | -0.083 | 3.7 % |
| ML | 330813 | -0.007 | 0.001 | 0.199 | 0.004 | -0.712 | -0.016 | 41.8 % |
| MR | 22436 | -0.032 | 0.002 | 0.051 | 0.159 | -0.453 | -0.168 | 2.9 % |
| RL | 30574 | -0.036 | -0.001 | -0.026 | 0.354 | -0.486 | -0.084 | 3.9 % |
| RR | 132831 | -0.010 | -0.000 | -0.038 | 0.040 | -0.523 | -0.036 | 16.8 % |
| pooled (within-leg) | 570095 | -0.028 | — | — | 0.105 | -0.526 | -0.052 | 12.1 % |

Self-checks (pooled): routed action = sn_stand_action (max |Δ| 0.0); R's HOLD cam target vs −clip(k_h c, ±trim) max |Δ| 1.2e-07 over 4717432 leg-rows; branch rows ok True. Pre-clip |a| > 1 share: cam 0.2 %, hip 0.0 %, knee 0.0 %.
