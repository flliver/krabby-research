# Observe pass — `SN_Amask_s16`

Generated 2026-10-12T17:54:56-04:00 by `sn_observe_read.py` (sha 5c2f79e70b6d); status **complete**. Reported, not gated (PLAN D6). Applied = clip(stand-net action, ±1); R = D3 constants on the same row.
Arm: stand ckpt `b17c5b5a9257` (sn_dagger), stand_mask 1, S1 `968cc5509899`.

| row | status | stamp | episodes | stands | stand rows | LIFT onsets |
|---|---|---|---|---|---|---|
| walkstop@1 | READ | 2026-10-12_21-31-46 | 100 | 300 | 180000 | 1004 |
| walkstop@2 | READ | 2026-10-12_21-34-25 | 100 | 300 | 180000 | 1013 |
| suite@1 | READ | 2026-10-12_21-17-29 | 1000 | 1000 | 235000 | 0 |
| suite@2 | READ | 2026-10-12_21-24-37 | 1000 | 1000 | 235000 | 0 |

## Headline (per row; pooled = every READ row)

| block | μ_hip LIFT (R -0.5; Z2d -0.054) | LIFT hip < -0.25 (Z2d 1.9 %) | onset launch ≤ 0.3 s | first onset per stand-leg | HOLD cam slope per leg, min … max (R -1; Z2d −0.04…−0.12) | within-leg slope | R's within-leg slope, same rows | within-leg slope, unsat band |
|---|---|---|---|---|---|---|---|---|
| walkstop@1 | -0.473 (n 7202) | 99.6 % | 100.0 % (1004/1004) | 100.0 % (646/646) | -0.037 … -0.018 | -0.029 (n 71671) | -0.495 | -0.118 |
| walkstop@2 | -0.471 (n 7567) | 99.2 % | 99.3 % (1006/1013) | 98.9 % (651/658) | -0.041 … -0.018 | -0.029 (n 74666) | -0.503 | -0.118 |
| suite@1 | — (n 0) | — | — (0/0) | — (0/0) | -0.176 … -0.176 | -0.176 (n 809) | -1.000 | -0.176 |
| suite@2 | — (n 0) | — | — (0/0) | — (0/0) | -0.303 … -0.303 | -0.303 (n 1075) | -1.000 | -0.303 |
| pooled walkstop | -0.472 (n 14769) | 99.4 % | 99.7 % (2010/2017) | 99.5 % (1297/1304) | -0.039 … -0.018 | -0.029 (n 146337) | -0.499 | -0.117 |
| pooled suite | — (n 0) | — | — (0/0) | — (0/0) | -0.092 … -0.092 | -0.092 (n 1884) | -1.000 | -0.092 |
| pooled all | -0.472 (n 14769) | 99.4 % | 99.7 % (2010/2017) | 99.5 % (1297/1304) | -0.039 … -0.018 | -0.029 (n 148221) | -0.500 | -0.116 |

## |applied − R| by R branch (pooled: walkstop@1, walkstop@2, suite@1, suite@2)

| branch | leg-rows | cam mean / p90 | hip mean / p90 | knee mean / p90 | hip signed |
|---|---|---|---|---|---|
| HOLD | 4732231 | 0.009 / 0.019 | 0.012 / 0.020 | 0.012 / 0.020 | -0.000 |
| WAIT | 66571 | 0.008 / 0.018 | 0.047 / 0.099 | 0.047 / 0.099 | -0.015 |
| LIFT | 14769 | 0.006 / 0.012 | 0.042 / 0.091 | 0.042 / 0.091 | 0.028 |
| HOME | 78422 | 0.031 / 0.059 | 0.042 / 0.105 | 0.042 / 0.105 | 0.025 |
| SEAT | 88007 | 0.033 / 0.072 | 0.061 / 0.150 | 0.061 / 0.150 | -0.023 |

LIFT rows: μ_hip -0.472 (pre-clip -0.472, median -0.481), median |c| 0.487 rad. HOME rows (R −0.5 too): μ_hip -0.475, < −0.25 99.0 % (n 78422).

### LIFT-onset launch per leg (hip < −0.25 within 0.3 s; censored = stand ended first)

| leg | onsets | launched | censored | share | already low at onset | latency median (s) | entry LIFTs | entry launched |
|---|---|---|---|---|---|---|---|---|
| FL | 287 | 286 | 0 | 99.7 % | 285 | 0.00 | 20 | 100.0 % |
| FR | 457 | 455 | 0 | 99.6 % | 447 | 0.00 | 36 | 100.0 % |
| ML | 337 | 336 | 0 | 99.7 % | 335 | 0.00 | 2 | 100.0 % |
| MR | 193 | 191 | 0 | 99.0 % | 189 | 0.00 | 0 | — |
| RL | 302 | 302 | 0 | 100.0 % | 302 | 0.00 | 29 | 96.6 % |
| RR | 441 | 440 | 0 | 99.8 % | 438 | 0.00 | 28 | 100.0 % |
| pooled | 2017 | 2010 | 0 | 99.7 % | 1996 | 0.00 | 115 | 99.1 % |
| pooled, first per stand-leg | 1304 | 1297 | 0 | 99.5 % | 1287 | 0.00 | — | — |

### HOLD cam-trim slope per leg (|c| in [0.03, 0.3] rad; applied cam on c)

| leg | n | slope | intercept | fixed point (rad) | R² | R's slope same rows | slope on [0.03, 0.1] | band share of HOLD |
|---|---|---|---|---|---|---|---|---|
| FL | 28998 | -0.036 | -0.001 | -0.020 | 0.230 | -0.513 | -0.142 | 3.7 % |
| FR | 8049 | -0.039 | 0.004 | 0.091 | 0.216 | -0.428 | -0.226 | 1.0 % |
| ML | 27675 | -0.018 | -0.001 | -0.038 | 0.040 | -0.553 | -0.061 | 3.5 % |
| MR | 35601 | -0.020 | 0.003 | 0.136 | 0.075 | -0.478 | -0.138 | 4.5 % |
| RL | 36085 | -0.032 | -0.006 | -0.195 | 0.127 | -0.532 | -0.104 | 4.6 % |
| RR | 11813 | -0.038 | -0.001 | -0.016 | 0.258 | -0.482 | -0.147 | 1.5 % |
| pooled (within-leg) | 148221 | -0.029 | — | — | 0.130 | -0.500 | -0.116 | 3.1 % |

Self-checks (pooled): routed action = sn_stand_action (max |Δ| 0.0); R's HOLD cam target vs −clip(k_h c, ±trim) max |Δ| 1.2e-07 over 4732231 leg-rows; branch rows ok True. Pre-clip |a| > 1 share: cam 0.2 %, hip 0.0 %, knee 0.0 %.
