# Observe pass — `SN_Amask_s17`

Generated 2026-10-12T20:13:28-04:00 by `sn_observe_read.py` (sha 5c2f79e70b6d); status **complete**. Reported, not gated (PLAN D6). Applied = clip(stand-net action, ±1); R = D3 constants on the same row.
Arm: stand ckpt `769cf2533140` (sn_dagger), stand_mask 1, S1 `968cc5509899`.

| row | status | stamp | episodes | stands | stand rows | LIFT onsets |
|---|---|---|---|---|---|---|
| walkstop@1 | READ | 2026-10-12_23-50-25 | 100 | 300 | 180000 | 953 |
| walkstop@2 | READ | 2026-10-12_23-53-04 | 100 | 300 | 180000 | 990 |
| suite@1 | READ | 2026-10-12_23-36-10 | 1000 | 1000 | 235000 | 0 |
| suite@2 | READ | 2026-10-12_23-43-16 | 1000 | 1000 | 235000 | 0 |

## Headline (per row; pooled = every READ row)

| block | μ_hip LIFT (R -0.5; Z2d -0.054) | LIFT hip < -0.25 (Z2d 1.9 %) | onset launch ≤ 0.3 s | first onset per stand-leg | HOLD cam slope per leg, min … max (R -1; Z2d −0.04…−0.12) | within-leg slope | R's within-leg slope, same rows | within-leg slope, unsat band |
|---|---|---|---|---|---|---|---|---|
| walkstop@1 | -0.492 (n 6705) | 99.4 % | 99.5 % (948/953) | 99.5 % (656/659) | -0.095 … -0.017 | -0.028 (n 124295) | -0.525 | -0.088 |
| walkstop@2 | -0.490 (n 7178) | 99.5 % | 99.2 % (982/990) | 99.3 % (669/674) | -0.078 … -0.016 | -0.028 (n 127870) | -0.518 | -0.083 |
| suite@1 | — (n 0) | — | — (0/0) | — (0/0) | -0.034 … 0.026 | -0.013 (n 29722) | -1.000 | -0.013 |
| suite@2 | — (n 0) | — | — (0/0) | — (0/0) | -0.139 … -0.011 | -0.013 (n 22593) | -1.000 | -0.013 |
| pooled walkstop | -0.491 (n 13883) | 99.5 % | 99.3 % (1930/1943) | 99.4 % (1325/1333) | -0.086 … -0.017 | -0.028 (n 252165) | -0.522 | -0.085 |
| pooled suite | — (n 0) | — | — (0/0) | — (0/0) | -0.026 … 0.011 | -0.012 (n 52315) | -1.000 | -0.012 |
| pooled all | -0.491 (n 13883) | 99.5 % | 99.3 % (1930/1943) | 99.4 % (1325/1333) | -0.086 … -0.017 | -0.028 (n 304480) | -0.523 | -0.083 |

## |applied − R| by R branch (pooled: walkstop@1, walkstop@2, suite@1, suite@2)

| branch | leg-rows | cam mean / p90 | hip mean / p90 | knee mean / p90 | hip signed |
|---|---|---|---|---|---|
| HOLD | 4732202 | 0.011 / 0.022 | 0.016 / 0.027 | 0.016 / 0.027 | -0.002 |
| WAIT | 68234 | 0.011 / 0.023 | 0.045 / 0.093 | 0.045 / 0.093 | -0.008 |
| LIFT | 13883 | 0.006 / 0.013 | 0.035 / 0.072 | 0.035 / 0.072 | 0.009 |
| HOME | 79390 | 0.031 / 0.059 | 0.042 / 0.096 | 0.042 / 0.096 | 0.017 |
| SEAT | 86291 | 0.029 / 0.064 | 0.061 / 0.143 | 0.061 / 0.143 | -0.013 |

LIFT rows: μ_hip -0.491 (pre-clip -0.491, median -0.500), median |c| 0.472 rad. HOME rows (R −0.5 too): μ_hip -0.483, < −0.25 99.0 % (n 79390).

### LIFT-onset launch per leg (hip < −0.25 within 0.3 s; censored = stand ended first)

| leg | onsets | launched | censored | share | already low at onset | latency median (s) | entry LIFTs | entry launched |
|---|---|---|---|---|---|---|---|---|
| FL | 255 | 255 | 0 | 100.0 % | 251 | 0.00 | 22 | 95.5 % |
| FR | 399 | 396 | 0 | 99.2 % | 391 | 0.00 | 30 | 100.0 % |
| ML | 345 | 342 | 0 | 99.1 % | 338 | 0.00 | 0 | — |
| MR | 279 | 278 | 0 | 99.6 % | 275 | 0.00 | 0 | — |
| RL | 256 | 253 | 0 | 98.8 % | 250 | 0.00 | 35 | 97.1 % |
| RR | 409 | 406 | 0 | 99.3 % | 402 | 0.00 | 33 | 100.0 % |
| pooled | 1943 | 1930 | 0 | 99.3 % | 1907 | 0.00 | 120 | 98.3 % |
| pooled, first per stand-leg | 1333 | 1325 | 0 | 99.4 % | 1307 | 0.00 | — | — |

### HOLD cam-trim slope per leg (|c| in [0.03, 0.3] rad; applied cam on c)

| leg | n | slope | intercept | fixed point (rad) | R² | R's slope same rows | slope on [0.03, 0.1] | band share of HOLD |
|---|---|---|---|---|---|---|---|---|
| FL | 31058 | -0.017 | 0.004 | 0.252 | 0.090 | -0.513 | -0.079 | 3.9 % |
| FR | 8463 | -0.086 | -0.014 | -0.167 | 0.281 | -0.422 | -0.439 | 1.1 % |
| ML | 61857 | -0.037 | -0.003 | -0.094 | 0.025 | -0.588 | -0.078 | 7.8 % |
| MR | 153159 | -0.019 | -0.003 | -0.130 | 0.086 | -0.573 | -0.057 | 19.5 % |
| RL | 29601 | -0.031 | 0.006 | 0.202 | 0.179 | -0.476 | -0.139 | 3.7 % |
| RR | 20342 | -0.027 | 0.002 | 0.085 | 0.130 | -0.528 | -0.055 | 2.6 % |
| pooled (within-leg) | 304480 | -0.028 | — | — | 0.076 | -0.523 | -0.083 | 6.4 % |

Self-checks (pooled): routed action = sn_stand_action (max |Δ| 0.0); R's HOLD cam target vs −clip(k_h c, ±trim) max |Δ| 1.2e-07 over 4732202 leg-rows; branch rows ok True. Pre-clip |a| > 1 share: cam 0.2 %, hip 0.0 %, knee 0.0 %.
