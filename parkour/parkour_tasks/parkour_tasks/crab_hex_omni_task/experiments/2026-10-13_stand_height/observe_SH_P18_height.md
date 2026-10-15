# Stand-height observe pass — `SH_P18`

Stand-height observe pass (sh_observe_read.py, a wrapper of sn_observe_read.py): stand rows = p2r_stand_mode (who acted: slot | restoring); RESIDUAL FRAME -- applied = clip(sh_a_net, +-1) (the executed action minus the posture q(h_lim)), R = sh_rh_target - sh_q18 (R_h minus the same posture); at 1.064 (q = 0) this is the predecessor's read exactly. Families: zero = walkstop + suite (zero command), height = hwalkstop + hraise (scenarios_sh.yaml).

Generated 2026-10-14T21:22:13-04:00 by `sh_observe_read.py` (sha 7f1419535d40); status **complete**. Reported, not gated (PLAN D6). Applied = clip(stand-net action, ±1); R = D3 constants on the same row.
Arm: stand ckpt `50b7c60a7e2b` (sh_dagger), stand_mask 1, S1 `968cc5509899`.

| row | status | stamp | episodes | stands | stand rows | LIFT onsets |
|---|---|---|---|---|---|---|
| hwalkstop@1 | READ | 2026-10-14_04-53-31 | 100 | 700 | 506110 | 2758 |
| hwalkstop@2 | READ | 2026-10-14_05-00-36 | 100 | 700 | 506100 | 2744 |
| hraise@1 | READ | 2026-10-14_05-07-38 | 100 | 200 | 389900 | 619 |
| hraise@2 | READ | 2026-10-14_05-12-15 | 100 | 200 | 389900 | 737 |

## Headline (per row; pooled = every READ row)

| block | μ_hip LIFT (R -0.5; Z2d -0.054) | LIFT hip < -0.25 (Z2d 1.9 %) | onset launch ≤ 0.3 s | first onset per stand-leg | HOLD cam slope per leg, min … max (R -1; Z2d −0.04…−0.12) | within-leg slope | R's within-leg slope, same rows | within-leg slope, unsat band |
|---|---|---|---|---|---|---|---|---|
| hwalkstop@1 | -0.477 (n 36192) | 97.8 % | 97.7 % (2695/2758) | 98.8 % (1623/1643) | -0.053 … -0.004 | -0.025 (n 417424) | -0.563 | -0.038 |
| hwalkstop@2 | -0.479 (n 35950) | 98.0 % | 97.5 % (2675/2744) | 98.3 % (1579/1607) | -0.062 … -0.004 | -0.025 (n 436553) | -0.574 | -0.034 |
| hraise@1 | -0.492 (n 3963) | 99.3 % | 99.4 % (615/619) | 99.5 % (433/435) | -0.047 … -0.008 | -0.014 (n 158689) | -0.571 | -0.040 |
| hraise@2 | -0.479 (n 4592) | 97.8 % | 98.0 % (722/737) | 97.7 % (473/484) | -0.041 … -0.010 | -0.015 (n 160480) | -0.556 | -0.040 |
| pooled height | -0.479 (n 80697) | 98.0 % | 97.8 % (6707/6858) | 98.5 % (4108/4169) | -0.059 … -0.005 | -0.023 (n 1173146) | -0.568 | -0.036 |
| pooled all | -0.479 (n 80697) | 98.0 % | 97.8 % (6707/6858) | 98.5 % (4108/4169) | -0.059 … -0.005 | -0.023 (n 1173146) | -0.568 | -0.036 |

## |applied − R| by R branch (pooled: hwalkstop@1, hwalkstop@2, hraise@1, hraise@2)

| branch | leg-rows | cam mean / p90 | hip mean / p90 | knee mean / p90 | hip signed |
|---|---|---|---|---|---|
| HOLD | 9791851 | 0.015 / 0.033 | 0.015 / 0.026 | 0.015 / 0.026 | -0.002 |
| WAIT | 280165 | 0.006 / 0.012 | 0.041 / 0.094 | 0.041 / 0.094 | -0.026 |
| LIFT | 80697 | 0.006 / 0.013 | 0.044 / 0.104 | 0.044 / 0.104 | 0.021 |
| HOME | 284674 | 0.034 / 0.067 | 0.029 / 0.064 | 0.029 / 0.064 | 0.010 |
| SEAT | 314673 | 0.031 / 0.062 | 0.064 / 0.172 | 0.064 / 0.172 | -0.033 |

LIFT rows: μ_hip -0.479 (pre-clip -0.479, median -0.495), median |c| 0.772 rad. HOME rows (R −0.5 too): μ_hip -0.490, < −0.25 99.6 % (n 284674).

### LIFT-onset launch per leg (hip < −0.25 within 0.3 s; censored = stand ended first)

| leg | onsets | launched | censored | share | already low at onset | latency median (s) | entry LIFTs | entry launched |
|---|---|---|---|---|---|---|---|---|
| FL | 1180 | 1160 | 0 | 98.3 % | 1118 | 0.00 | 111 | 100.0 % |
| FR | 1273 | 1233 | 0 | 96.9 % | 1178 | 0.00 | 38 | 100.0 % |
| ML | 1302 | 1263 | 0 | 97.0 % | 1200 | 0.00 | 1 | 100.0 % |
| MR | 1080 | 1061 | 0 | 98.2 % | 1022 | 0.00 | 7 | 100.0 % |
| RL | 1127 | 1114 | 0 | 98.8 % | 1079 | 0.00 | 102 | 100.0 % |
| RR | 896 | 876 | 0 | 97.8 % | 840 | 0.00 | 33 | 100.0 % |
| pooled | 6858 | 6707 | 0 | 97.8 % | 6437 | 0.00 | 292 | 100.0 % |
| pooled, first per stand-leg | 4169 | 4108 | 0 | 98.5 % | 3988 | 0.00 | — | — |

### HOLD cam-trim slope per leg (|c| in [0.03, 0.3] rad; applied cam on c)

| leg | n | slope | intercept | fixed point (rad) | R² | R's slope same rows | slope on [0.03, 0.1] | band share of HOLD |
|---|---|---|---|---|---|---|---|---|
| FL | 107473 | -0.023 | 0.001 | 0.045 | 0.178 | -0.516 | -0.060 | 6.7 % |
| FR | 78903 | -0.059 | 0.001 | 0.016 | 0.101 | -0.509 | -0.188 | 4.8 % |
| ML | 359791 | -0.012 | 0.001 | 0.060 | 0.042 | -0.665 | -0.016 | 22.1 % |
| MR | 289032 | -0.005 | -0.000 | -0.051 | 0.014 | -0.617 | -0.020 | 17.7 % |
| RL | 191097 | -0.011 | -0.000 | -0.035 | 0.072 | -0.631 | -0.011 | 11.7 % |
| RR | 146850 | -0.026 | -0.001 | -0.030 | 0.181 | -0.496 | -0.066 | 8.9 % |
| pooled (within-leg) | 1173146 | -0.023 | — | — | 0.070 | -0.568 | -0.036 | 12.0 % |

Self-checks (pooled): routed action = sn_stand_action (max |Δ| 0.0); R's HOLD cam target vs −clip(k_h c, ±trim) max |Δ| 1.2e-07 over 9791851 leg-rows; branch rows ok True. Pre-clip |a| > 1 share: cam 0.6 %, hip 0.0 %, knee 0.0 %.
