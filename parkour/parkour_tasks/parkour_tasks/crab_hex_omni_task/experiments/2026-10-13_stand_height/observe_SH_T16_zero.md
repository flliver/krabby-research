# Stand-height observe pass — `SH_T16z`

Stand-height observe pass (sh_observe_read.py, a wrapper of sn_observe_read.py): stand rows = p2r_stand_mode (who acted: slot | restoring); RESIDUAL FRAME -- applied = clip(sh_a_net, +-1) (the executed action minus the posture q(h_lim)), R = sh_rh_target - sh_q18 (R_h minus the same posture); at 1.064 (q = 0) this is the predecessor's read exactly. Families: zero = walkstop + suite (zero command), height = hwalkstop + hraise (scenarios_sh.yaml).

Generated 2026-10-14T21:21:46-04:00 by `sh_observe_read.py` (sha 7f1419535d40); status **complete**. Reported, not gated (PLAN D6). Applied = clip(stand-net action, ±1); R = D3 constants on the same row.
Arm: stand ckpt `230cc4973bea` (sh_dagger), stand_mask 1, S1 `968cc5509899`.

| row | status | stamp | episodes | stands | stand rows | LIFT onsets |
|---|---|---|---|---|---|---|
| walkstop@1 | READ | 2026-10-14_21-22-33 | 100 | 300 | 180000 | 793 |
| walkstop@2 | READ | 2026-10-14_21-25-12 | 100 | 300 | 180000 | 1011 |
| suite@1 | READ | 2026-10-14_21-08-02 | 1000 | 1000 | 235000 | 0 |
| suite@2 | READ | 2026-10-14_21-15-18 | 1000 | 1000 | 235000 | 0 |

## Headline (per row; pooled = every READ row)

| block | μ_hip LIFT (R -0.5; Z2d -0.054) | LIFT hip < -0.25 (Z2d 1.9 %) | onset launch ≤ 0.3 s | first onset per stand-leg | HOLD cam slope per leg, min … max (R -1; Z2d −0.04…−0.12) | within-leg slope | R's within-leg slope, same rows | within-leg slope, unsat band |
|---|---|---|---|---|---|---|---|---|
| walkstop@1 | -0.487 (n 7173) | 98.7 % | 98.7 % (783/793) | 98.8 % (587/594) | -0.387 … -0.159 | -0.307 (n 48567) | -0.522 | -0.716 |
| walkstop@2 | -0.476 (n 8286) | 98.2 % | 98.1 % (992/1011) | 98.1 % (621/633) | -0.373 … -0.156 | -0.298 (n 49646) | -0.512 | -0.677 |
| suite@1 | — (n 0) | — | — (0/0) | — (0/0) | — … — | — (n 0) | — | — |
| suite@2 | — (n 0) | — | — (0/0) | — (0/0) | — … — | — (n 0) | — | — |
| pooled zero | -0.481 (n 15459) | 98.5 % | 98.4 % (1775/1804) | 98.5 % (1208/1227) | -0.380 … -0.157 | -0.303 (n 98213) | -0.517 | -0.695 |
| pooled all | -0.481 (n 15459) | 98.5 % | 98.4 % (1775/1804) | 98.5 % (1208/1227) | -0.380 … -0.157 | -0.303 (n 98213) | -0.517 | -0.695 |

## |applied − R| by R branch (pooled: walkstop@1, walkstop@2, suite@1, suite@2)

| branch | leg-rows | cam mean / p90 | hip mean / p90 | knee mean / p90 | hip signed |
|---|---|---|---|---|---|
| HOLD | 4743262 | 0.005 / 0.009 | 0.000 / 0.000 | 0.000 / 0.000 | -0.000 |
| WAIT | 70375 | 0.014 / 0.028 | 0.014 / 0.000 | 0.014 / 0.000 | -0.014 |
| LIFT | 15459 | 0.008 / 0.017 | 0.048 / 0.108 | 0.048 / 0.108 | 0.019 |
| HOME | 63614 | 0.035 / 0.066 | 0.028 / 0.060 | 0.028 / 0.060 | 0.005 |
| SEAT | 87290 | 0.034 / 0.069 | 0.017 / 0.000 | 0.017 / 0.000 | -0.016 |

LIFT rows: μ_hip -0.481 (pre-clip -0.481, median -0.501), median |c| 0.573 rad. HOME rows (R −0.5 too): μ_hip -0.495, < −0.25 99.7 % (n 63614).

### LIFT-onset launch per leg (hip < −0.25 within 0.3 s; censored = stand ended first)

| leg | onsets | launched | censored | share | already low at onset | latency median (s) | entry LIFTs | entry launched |
|---|---|---|---|---|---|---|---|---|
| FL | 331 | 329 | 0 | 99.4 % | 327 | 0.00 | 24 | 100.0 % |
| FR | 385 | 376 | 0 | 97.7 % | 370 | 0.00 | 24 | 95.8 % |
| ML | 298 | 288 | 0 | 96.6 % | 281 | 0.00 | 1 | 100.0 % |
| MR | 186 | 186 | 0 | 100.0 % | 185 | 0.00 | 2 | 100.0 % |
| RL | 306 | 304 | 0 | 99.3 % | 300 | 0.00 | 34 | 100.0 % |
| RR | 298 | 292 | 0 | 98.0 % | 280 | 0.00 | 49 | 100.0 % |
| pooled | 1804 | 1775 | 0 | 98.4 % | 1743 | 0.00 | 134 | 99.3 % |
| pooled, first per stand-leg | 1227 | 1208 | 0 | 98.5 % | 1179 | 0.00 | — | — |

### HOLD cam-trim slope per leg (|c| in [0.03, 0.3] rad; applied cam on c)

| leg | n | slope | intercept | fixed point (rad) | R² | R's slope same rows | slope on [0.03, 0.1] | band share of HOLD |
|---|---|---|---|---|---|---|---|---|
| FL | 18033 | -0.380 | 0.012 | 0.031 | 0.721 | -0.540 | -0.673 | 2.3 % |
| FR | 11620 | -0.324 | -0.004 | -0.012 | 0.654 | -0.505 | -0.631 | 1.5 % |
| ML | 15155 | -0.157 | 0.007 | 0.044 | 0.286 | -0.533 | -0.520 | 1.9 % |
| MR | 20475 | -0.328 | 0.014 | 0.043 | 0.666 | -0.517 | -0.767 | 2.6 % |
| RL | 22499 | -0.271 | -0.002 | -0.008 | 0.607 | -0.495 | -0.775 | 2.8 % |
| RR | 10431 | -0.358 | 0.011 | 0.031 | 0.707 | -0.512 | -0.788 | 1.3 % |
| pooled (within-leg) | 98213 | -0.303 | — | — | 0.609 | -0.517 | -0.695 | 2.1 % |

Self-checks (pooled): routed action = sn_stand_action (max |Δ| 0.0); R's HOLD cam target vs −clip(k_h c, ±trim) max |Δ| 1.2e-07 over 4743262 leg-rows; branch rows ok True. Pre-clip |a| > 1 share: cam 0.0 %, hip 0.0 %, knee 0.0 %.
