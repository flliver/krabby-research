# Observe pass — `SN_Amask_s18`

Generated 2026-10-12T12:40:25-04:00 by `sn_observe_read.py` (sha 5c2f79e70b6d); status **complete**. Reported, not gated (PLAN D6). Applied = clip(stand-net action, ±1); R = D3 constants on the same row.
Arm: stand ckpt `f7228a998c75` (sn_dagger), stand_mask 1, S1 `968cc5509899`.

| row | status | stamp | episodes | stands | stand rows | LIFT onsets |
|---|---|---|---|---|---|---|
| walkstop@1 | READ | 2026-10-12_16-16-41 | 100 | 300 | 180000 | 1048 |
| walkstop@2 | READ | 2026-10-12_16-19-23 | 100 | 300 | 180000 | 1016 |
| suite@1 | READ | 2026-10-12_16-01-37 | 1000 | 1000 | 235000 | 0 |
| suite@2 | READ | 2026-10-12_16-09-15 | 1000 | 1000 | 235000 | 0 |

## Headline (per row; pooled = every READ row)

| block | μ_hip LIFT (R -0.5; Z2d -0.054) | LIFT hip < -0.25 (Z2d 1.9 %) | onset launch ≤ 0.3 s | first onset per stand-leg | HOLD cam slope per leg, min … max (R -1; Z2d −0.04…−0.12) | within-leg slope | R's within-leg slope, same rows | within-leg slope, unsat band |
|---|---|---|---|---|---|---|---|---|
| walkstop@1 | -0.478 (n 7761) | 99.4 % | 99.4 % (1042/1048) | 99.7 % (680/682) | -0.063 … -0.015 | -0.032 (n 183643) | -0.511 | -0.082 |
| walkstop@2 | -0.481 (n 7257) | 99.6 % | 99.8 % (1014/1016) | 99.7 % (668/670) | -0.067 … -0.014 | -0.031 (n 189469) | -0.520 | -0.083 |
| suite@1 | — (n 0) | — | — (0/0) | — (0/0) | -0.042 … 0.023 | -0.036 (n 125631) | -1.000 | -0.036 |
| suite@2 | — (n 0) | — | — (0/0) | — (0/0) | -0.050 … 0.107 | -0.047 (n 131324) | -1.000 | -0.047 |
| pooled walkstop | -0.479 (n 15018) | 99.5 % | 99.6 % (2056/2064) | 99.7 % (1348/1352) | -0.065 … -0.014 | -0.032 (n 373112) | -0.516 | -0.083 |
| pooled suite | — (n 0) | — | — (0/0) | — (0/0) | -0.046 … 0.067 | -0.041 (n 256955) | -1.000 | -0.041 |
| pooled all | -0.479 (n 15018) | 99.5 % | 99.6 % (2056/2064) | 99.7 % (1348/1352) | -0.065 … -0.015 | -0.032 (n 630067) | -0.522 | -0.077 |

## |applied − R| by R branch (pooled: walkstop@1, walkstop@2, suite@1, suite@2)

| branch | leg-rows | cam mean / p90 | hip mean / p90 | knee mean / p90 | hip signed |
|---|---|---|---|---|---|
| HOLD | 4721555 | 0.015 / 0.038 | 0.015 / 0.028 | 0.014 / 0.028 | 0.007 |
| WAIT | 74023 | 0.007 / 0.015 | 0.047 / 0.099 | 0.047 / 0.099 | -0.013 |
| LIFT | 15018 | 0.006 / 0.013 | 0.040 / 0.090 | 0.040 / 0.090 | 0.021 |
| HOME | 80826 | 0.030 / 0.056 | 0.041 / 0.099 | 0.041 / 0.099 | 0.017 |
| SEAT | 88578 | 0.031 / 0.065 | 0.063 / 0.154 | 0.063 / 0.155 | -0.023 |

LIFT rows: μ_hip -0.479 (pre-clip -0.479, median -0.489), median |c| 0.502 rad. HOME rows (R −0.5 too): μ_hip -0.483, < −0.25 99.2 % (n 80826).

### LIFT-onset launch per leg (hip < −0.25 within 0.3 s; censored = stand ended first)

| leg | onsets | launched | censored | share | already low at onset | latency median (s) | entry LIFTs | entry launched |
|---|---|---|---|---|---|---|---|---|
| FL | 308 | 307 | 0 | 99.7 % | 303 | 0.00 | 29 | 100.0 % |
| FR | 428 | 425 | 0 | 99.3 % | 420 | 0.00 | 28 | 100.0 % |
| ML | 357 | 356 | 0 | 99.7 % | 353 | 0.00 | 0 | — |
| MR | 253 | 251 | 0 | 99.2 % | 249 | 0.00 | 6 | 100.0 % |
| RL | 335 | 335 | 0 | 100.0 % | 333 | 0.00 | 34 | 100.0 % |
| RR | 383 | 382 | 0 | 99.7 % | 378 | 0.00 | 35 | 100.0 % |
| pooled | 2064 | 2056 | 0 | 99.6 % | 2036 | 0.00 | 132 | 100.0 % |
| pooled, first per stand-leg | 1352 | 1348 | 0 | 99.7 % | 1335 | 0.00 | — | — |

### HOLD cam-trim slope per leg (|c| in [0.03, 0.3] rad; applied cam on c)

| leg | n | slope | intercept | fixed point (rad) | R² | R's slope same rows | slope on [0.03, 0.1] | band share of HOLD |
|---|---|---|---|---|---|---|---|---|
| FL | 24328 | -0.051 | -0.001 | -0.017 | 0.445 | -0.509 | -0.137 | 3.1 % |
| FR | 27752 | -0.037 | -0.003 | -0.070 | 0.207 | -0.473 | -0.196 | 3.6 % |
| ML | 20018 | -0.065 | 0.006 | 0.098 | 0.088 | -0.537 | -0.185 | 2.5 % |
| MR | 473226 | -0.015 | -0.002 | -0.143 | 0.099 | -0.550 | -0.046 | 60.1 % |
| RL | 32846 | -0.037 | -0.002 | -0.056 | 0.380 | -0.506 | -0.104 | 4.2 % |
| RR | 51897 | -0.016 | -0.000 | -0.005 | 0.101 | -0.530 | -0.026 | 6.6 % |
| pooled (within-leg) | 630067 | -0.032 | — | — | 0.141 | -0.522 | -0.077 | 13.3 % |

Self-checks (pooled): routed action = sn_stand_action (max |Δ| 0.0); R's HOLD cam target vs −clip(k_h c, ±trim) max |Δ| 1.2e-07 over 4721555 leg-rows; branch rows ok True. Pre-clip |a| > 1 share: cam 0.2 %, hip 0.0 %, knee 0.0 %.
