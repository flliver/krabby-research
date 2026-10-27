# Stand-height observe pass — `SH_T16`

Stand-height observe pass (sh_observe_read.py, a wrapper of sn_observe_read.py): stand rows = p2r_stand_mode (who acted: slot | restoring); RESIDUAL FRAME -- applied = clip(sh_a_net, +-1) (the executed action minus the posture q(h_lim)), R = sh_rh_target - sh_q18 (R_h minus the same posture); at 1.064 (q = 0) this is the predecessor's read exactly. Families: zero = walkstop + suite (zero command), height = hwalkstop + hraise (scenarios_sh.yaml).

Generated 2026-10-14T21:21:39-04:00 by `sh_observe_read.py` (sha 7f1419535d40); status **complete**. Reported, not gated (PLAN D6). Applied = clip(stand-net action, ±1); R = D3 constants on the same row.
Arm: stand ckpt `230cc4973bea` (sh_dagger), stand_mask 1, S1 `968cc5509899`.

| row | status | stamp | episodes | stands | stand rows | LIFT onsets |
|---|---|---|---|---|---|---|
| hwalkstop@1 | READ | 2026-10-14_20-45-26 | 100 | 700 | 506200 | 2148 |
| hwalkstop@2 | READ | 2026-10-14_20-52-18 | 100 | 700 | 506445 | 2346 |
| hraise@1 | READ | 2026-10-14_20-59-08 | 100 | 200 | 389992 | 753 |
| hraise@2 | READ | 2026-10-14_21-03-35 | 100 | 200 | 389900 | 636 |

## Headline (per row; pooled = every READ row)

| block | μ_hip LIFT (R -0.5; Z2d -0.054) | LIFT hip < -0.25 (Z2d 1.9 %) | onset launch ≤ 0.3 s | first onset per stand-leg | HOLD cam slope per leg, min … max (R -1; Z2d −0.04…−0.12) | within-leg slope | R's within-leg slope, same rows | within-leg slope, unsat band |
|---|---|---|---|---|---|---|---|---|
| hwalkstop@1 | -0.487 (n 35282) | 98.7 % | 97.4 % (2093/2148) | 97.7 % (1505/1540) | -0.341 … -0.150 | -0.271 (n 142821) | -0.532 | -0.540 |
| hwalkstop@2 | -0.479 (n 37925) | 97.6 % | 98.3 % (2305/2346) | 98.4 % (1571/1597) | -0.338 … -0.157 | -0.278 (n 135237) | -0.528 | -0.555 |
| hraise@1 | -0.482 (n 4894) | 98.6 % | 99.5 % (749/753) | 99.8 % (404/405) | -0.568 … -0.130 | -0.252 (n 63760) | -0.550 | -0.459 |
| hraise@2 | -0.493 (n 5021) | 98.7 % | 98.6 % (627/636) | 99.5 % (433/435) | -0.576 … -0.122 | -0.246 (n 65007) | -0.540 | -0.473 |
| pooled height | -0.483 (n 83122) | 98.2 % | 98.1 % (5774/5883) | 98.4 % (3913/3977) | -0.337 … -0.143 | -0.269 (n 406825) | -0.534 | -0.524 |
| pooled all | -0.483 (n 83122) | 98.2 % | 98.1 % (5774/5883) | 98.4 % (3913/3977) | -0.337 … -0.143 | -0.269 (n 406825) | -0.534 | -0.524 |

## |applied − R| by R branch (pooled: hwalkstop@1, hwalkstop@2, hraise@1, hraise@2)

| branch | leg-rows | cam mean / p90 | hip mean / p90 | knee mean / p90 | hip signed |
|---|---|---|---|---|---|
| HOLD | 9878752 | 0.008 / 0.017 | 0.000 / 0.000 | 0.000 / 0.000 | -0.000 |
| WAIT | 261201 | 0.012 / 0.028 | 0.012 / 0.000 | 0.012 / 0.000 | -0.012 |
| LIFT | 83122 | 0.010 / 0.023 | 0.048 / 0.108 | 0.048 / 0.108 | 0.017 |
| HOME | 239933 | 0.040 / 0.077 | 0.031 / 0.065 | 0.031 / 0.065 | 0.009 |
| SEAT | 292214 | 0.037 / 0.076 | 0.017 / 0.000 | 0.017 / 0.000 | -0.016 |

LIFT rows: μ_hip -0.483 (pre-clip -0.483, median -0.500), median |c| 0.817 rad. HOME rows (R −0.5 too): μ_hip -0.491, < −0.25 99.4 % (n 239933).

### LIFT-onset launch per leg (hip < −0.25 within 0.3 s; censored = stand ended first)

| leg | onsets | launched | censored | share | already low at onset | latency median (s) | entry LIFTs | entry launched |
|---|---|---|---|---|---|---|---|---|
| FL | 1171 | 1145 | 0 | 97.8 % | 1089 | 0.00 | 98 | 100.0 % |
| FR | 1078 | 1061 | 0 | 98.4 % | 1018 | 0.00 | 45 | 91.1 % |
| ML | 1030 | 1008 | 0 | 97.9 % | 964 | 0.00 | 0 | — |
| MR | 870 | 850 | 0 | 97.7 % | 817 | 0.00 | 13 | 100.0 % |
| RL | 1112 | 1089 | 0 | 97.9 % | 1036 | 0.00 | 103 | 99.0 % |
| RR | 622 | 621 | 0 | 99.8 % | 612 | 0.00 | 36 | 100.0 % |
| pooled | 5883 | 5774 | 0 | 98.1 % | 5536 | 0.00 | 295 | 98.3 % |
| pooled, first per stand-leg | 3977 | 3913 | 0 | 98.4 % | 3780 | 0.00 | — | — |

### HOLD cam-trim slope per leg (|c| in [0.03, 0.3] rad; applied cam on c)

| leg | n | slope | intercept | fixed point (rad) | R² | R's slope same rows | slope on [0.03, 0.1] | band share of HOLD |
|---|---|---|---|---|---|---|---|---|
| FL | 34381 | -0.317 | 0.006 | 0.020 | 0.556 | -0.478 | -0.812 | 2.1 % |
| FR | 163910 | -0.292 | -0.010 | -0.035 | 0.597 | -0.592 | -0.318 | 9.9 % |
| ML | 72365 | -0.143 | 0.004 | 0.031 | 0.286 | -0.539 | -0.417 | 4.4 % |
| MR | 51001 | -0.337 | 0.012 | 0.036 | 0.538 | -0.512 | -0.791 | 3.1 % |
| RL | 30836 | -0.295 | 0.007 | 0.023 | 0.586 | -0.526 | -0.743 | 1.9 % |
| RR | 54332 | -0.331 | 0.013 | 0.039 | 0.697 | -0.510 | -0.762 | 3.2 % |
| pooled (within-leg) | 406825 | -0.269 | — | — | 0.525 | -0.534 | -0.524 | 4.1 % |

Self-checks (pooled): routed action = sn_stand_action (max |Δ| 0.0); R's HOLD cam target vs −clip(k_h c, ±trim) max |Δ| 1.2e-07 over 9878752 leg-rows; branch rows ok True. Pre-clip |a| > 1 share: cam 0.0 %, hip 0.0 %, knee 0.0 %.
