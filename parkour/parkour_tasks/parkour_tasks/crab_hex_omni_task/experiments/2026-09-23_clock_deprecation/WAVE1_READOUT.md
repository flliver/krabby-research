# Wave-1 readout: clock deprecation, all experiments (2026-09-25)

Read-only; nothing trained or evaluated. Every per-category number is the driver's own statistic,
`run_phases.summarize(aggregate)['cat'][category][key]`: survival is the n-weighted own-hold survival, every other key is
the median over the category's episodes (both hands pooled). Block and guard verdicts are quoted from `ledger.json`
(D05, D06, D07), not recomputed. Values are rounded to 3 decimals; `—` means the key is not defined for that category.

## Summary

Wave 1 is finished and has **no WIN**, so the S4 question goes to the user. All three arms are clear REJECTs against the
clock-held twin X0 s3, and each is class BREAK.
- **A6000** (clock removed at c5→c6): F is BETTER, but RIG is a TIE (z +0.389, R 0.786). It is WORSE on the side and turn
  `foot_slip` guards.
- **A4500** (removed at c4→c5): the best arm. F, F_mt, T and RIG are all BETTER. RIG is balanced (steps +0.638 /
  support +0.335), turn `steps_per_rad` falls 39.9 → 30.1, and side ratio rises 0.396 → 0.501. It still breaks on the same
  two slip guards (side 0.173 → 0.203, turn 0.115 → 0.123) and carries a margin flag: move_turn survival 0.745 → 0.590.
- **A3000** (removed at t3→c4): hit the c4 floor on move_turn survival, 0.190 against X0 s3:c4's 0.795.

Every clock-free head shares two changes:
- `offset_R` collapses from 0.96–1.00 to 0.02–0.20, so the fixed tripod phase offsets are gone;
- `cadence_ratio` leaves 1.0 (0.78–1.39 on the arms' walking and turning categories), so cadence decouples from the clock.

On both arms, compared with X0 s3:
- stand `steps_per_s` drops from 3.500 to 2.750;
- per-foot slip rises in side, turn, stand and move_turn;
- survival rises on all four travel categories (for example fwd 0.850 → 0.990).

**Z** is the never-clocked arm, trained from scratch.
- It failed the t3 formation readout on both seeds: fwd and back survival 0.000, although moving ratios were 0.46–0.76.
- By c6 the s3 seed partly recovered (fwd 0.460, back 0.560).
- The s2 seed did not: fwd/back/diag/move_turn survival stayed at 0.000–0.010, and turn `steps_per_rad` is 193.5.

**Appendix results.**
- Hiding the clock zero-shot from a clocked head is destructive: X08:c6 fwd survival falls 0.920 → 0.410 and turn
  `facing_closed` 0.618 → 0.185.
- On light terrain, A4500:c6 tracks better than X0 s3:c6 (0.581 vs 0.531), but it completes slightly less (0.430 vs
  0.460) and slips more (0.399 vs 0.364).

## Decisions (ledger, verbatim numbers)

All three lines are seed 3 only. No seed-2 pairs were queued: a clear REJECT and a floor do not pair.

### A6000 — D05 (X03, K on c6, 2000 clock-free its)

| Item | Value |
|---|---|
| F (turn facing_closed, yaw_progress) | z +1.537, R 1.280 — **BETTER** |
| F_mt (move_turn) | z −0.060, R 0.912 — TIE |
| T (ratio fwd/side/back/diag) | z +0.326, R 0.487 — TIE |
| RIG (10 members) | z +0.389, R 0.786 — TIE |
| RIG split (steps / support, k 5 / 5) | +0.088 / +0.301 |
| Guards WORSE (99 %, one-sided) | `side.foot_slip` 0.173 → 0.190 (Δ −0.017, R 0.005); `turn.foot_slip` 0.115 → 0.121 (Δ −0.006, R 0.004) |
| Anchored drift | FAIL (same two guards) |
| Removal verification | ok (weights 0.001, clock_hidden_frac 1.0, clock_obs ≡ 0) |
| Flags | `pilot` (S3p: seed-3 RIG TIE); acked by the user 2026-09-24: continue wave 1 |
| Final | **REJECT (clear, decided)**, driver 2026-09-24 10:25; class **BREAK** |

### A4500 — D06 (X04, K on c5+c6, 3500 clock-free its)

| Item | Value |
|---|---|
| F | z +2.067, R 1.280 — **BETTER** |
| F_mt | z +3.188, R 0.912 — **BETTER** |
| T | z +1.757, R 0.487 — **BETTER** |
| RIG | z +0.973, R 0.786 — **BETTER** |
| RIG split (steps / support, k 5 / 5) | +0.638 / +0.335 (both ≥ 0: balanced) |
| Guards WORSE | `side.foot_slip` 0.173 → 0.203 (Δ −0.030, R 0.005); `turn.foot_slip` 0.115 → 0.123 (Δ −0.008, R 0.004) |
| Anchored drift | FAIL (same two guards) |
| Removal verification | ok |
| Flags | `margin`: move_turn.survival 0.745 → 0.590 (Δ −0.155 ≤ −0.05; guard MDE 0.235; the guard itself TIE); acked under the user's autonomous-wave-1 instruction |
| Final | **REJECT (clear, decided)**, driver 2026-09-24 12:46; class **BREAK** |

### A3000 — D07 (X08, K on c4+c5+c6; stopped at c4)

| Item | Value |
|---|---|
| Blocks / guards | not scored: the chain stopped at the c4 floor before c6 |
| Floor (c4 `min_by_category`, one-sided 99 % band, n_env 100) | stand 1.000 (≥ 0.69); fwd 0.740 (≥ 0.52); side 0.990 (≥ 0.65); back 0.810 (≥ 0.55); turn 0.990 (≥ 0.62); **move_turn 0.190 (bound 0.281, min ≥ 0.315) — FAIL** |
| Gate | ran under the user's autonomous-wave-1 instruction (the two-BREAK `await_if` was answered `run` by the autopilot) (A6000 and A4500 both BREAK) |
| Final | **REJECT (clear, floor at c4)**, driver 2026-09-25 09:08; no pair; class **BREAK** (floor) |

## Per-category metrics, final heads (c6)

X08:c6 is the pre-campaign recipe of record, with the clock visible. X0 is the clock-held twin under the PPO fix. A6000
and A4500 are seed 3 and are judged against X0 s3. Z is the never-clocked reference, trained from scratch.

### Survival

| Category | X08:c6 | X0 s3 | X0 s2 | A6000 | A4500 | Z s3 | Z s2 |
|---|---|---|---|---|---|---|---|
| stand | 0.990 | 0.990 | 0.990 | 1.000 | 1.000 | 0.810 | 0.990 |
| fwd | 0.920 | 0.850 | 0.950 | 0.990 | 0.990 | 0.460 | 0.000 |
| side | 0.975 | 0.970 | 0.985 | 0.995 | 1.000 | 0.865 | 0.875 |
| back | 0.920 | 0.880 | 0.920 | 1.000 | 0.990 | 0.560 | 0.010 |
| diag | 0.790 | 0.720 | 0.740 | 0.870 | 0.820 | 0.300 | 0.000 |
| turn | 0.950 | 0.985 | 0.985 | 1.000 | 1.000 | 0.805 | 0.980 |
| move_turn | 0.750 | 0.745 | 0.980 | 0.730 | 0.590 | 0.550 | 0.000 |

### Speed ratio (achieved / commanded)

| Category | X08:c6 | X0 s3 | X0 s2 | A6000 | A4500 | Z s3 | Z s2 |
|---|---|---|---|---|---|---|---|
| fwd | 0.748 | 0.733 | 0.740 | 0.694 | 0.716 | 0.466 | 0.503 |
| side | 0.430 | 0.396 | 0.389 | 0.435 | 0.501 | 0.514 | 0.419 |
| back | 0.736 | 0.746 | 0.696 | 0.725 | 0.724 | 0.478 | 0.503 |
| diag | 0.527 | 0.539 | 0.544 | 0.551 | 0.590 | 0.548 | 0.609 |
| move_turn | 0.424 | 0.498 | 0.514 | 0.510 | 0.559 | 0.483 | 0.467 |

### Turning

| Metric | X08:c6 | X0 s3 | X0 s2 | A6000 | A4500 | Z s3 | Z s2 |
|---|---|---|---|---|---|---|---|
| facing_closed · turn | 0.618 | 0.629 | 0.560 | 0.704 | 0.726 | 0.508 | 0.076 |
| yaw_progress · turn | 0.083 | 0.082 | 0.074 | 0.092 | 0.096 | 0.065 | 0.010 |
| facing_closed · move_turn | 0.235 | 0.330 | 0.365 | 0.330 | 0.450 | 0.185 | 0.059 |
| yaw_progress · move_turn | 0.035 | 0.048 | 0.048 | 0.047 | 0.068 | 0.032 | 0.022 |

### RIG members (steps lower is better; support_feet higher is better)

| Metric | X08:c6 | X0 s3 | X0 s2 | A6000 | A4500 | Z s3 | Z s2 |
|---|---|---|---|---|---|---|---|
| steps_per_m · fwd | 16.348 | 16.339 | 15.133 | 16.905 | 16.164 | 28.750 | 19.056 |
| steps_per_m · side | 32.930 | 35.112 | 34.729 | 36.221 | 31.145 | 26.037 | 27.552 |
| steps_per_m · back | 17.005 | 17.352 | 17.083 | 18.166 | 18.782 | 31.932 | 24.225 |
| steps_per_m · diag | 25.951 | 25.117 | 25.675 | 24.697 | 22.552 | 22.680 | 17.442 |
| steps_per_rad · turn | 44.205 | 39.863 | 44.703 | 31.407 | 30.057 | 47.495 | 193.544 |
| steps_per_rad · move_turn (WATCH) | 89.397 | 68.863 | 69.156 | 60.753 | 43.044 | 93.144 | 80.831 |
| support_feet · fwd | 3.023 | 3.061 | 3.008 | 3.087 | 3.102 | 2.815 | 2.517 |
| support_feet · side | 3.557 | 3.528 | 3.518 | 3.564 | 3.541 | 3.634 | 3.423 |
| support_feet · back | 3.035 | 3.050 | 3.016 | 3.091 | 3.087 | 2.887 | 2.585 |
| support_feet · diag | 3.362 | 3.340 | 3.339 | 3.315 | 3.331 | 3.268 | 2.857 |
| support_feet · turn | 3.120 | 3.187 | 3.097 | 3.182 | 3.195 | 2.978 | 2.888 |
| support_feet · stand (WATCH) | 3.144 | 3.262 | 3.159 | 3.143 | 3.155 | 3.043 | 2.918 |
| support_feet · move_turn (WATCH) | 3.072 | 3.127 | 3.109 | 3.087 | 3.087 | 2.900 | 2.543 |

### Guards (lower is better)

| Metric | X08:c6 | X0 s3 | X0 s2 | A6000 | A4500 | Z s3 | Z s2 |
|---|---|---|---|---|---|---|---|
| foot_slip · fwd | 0.133 | 0.147 | 0.139 | 0.138 | 0.145 | 0.136 | 0.092 |
| foot_slip · side | 0.180 | 0.173 | 0.178 | 0.190 | 0.203 | 0.224 | 0.173 |
| foot_slip · back | 0.130 | 0.143 | 0.140 | 0.135 | 0.141 | 0.138 | 0.095 |
| foot_slip · diag | 0.181 | 0.186 | 0.187 | 0.182 | 0.187 | 0.206 | 0.152 |
| foot_slip · turn | 0.110 | 0.115 | 0.112 | 0.121 | 0.123 | 0.142 | 0.086 |
| foot_slip · stand (WATCH) | 0.105 | 0.108 | 0.109 | 0.146 | 0.145 | 0.137 | 0.084 |
| foot_slip · move_turn (WATCH) | 0.129 | 0.134 | 0.130 | 0.152 | 0.147 | 0.147 | 0.092 |
| steps_per_s · stand | 3.083 | 3.500 | 3.167 | 2.750 | 2.750 | 2.917 | 2.417 |

### WATCH

| Metric | X08:c6 | X0 s3 | X0 s2 | A6000 | A4500 | Z s3 | Z s2 |
|---|---|---|---|---|---|---|---|
| tripod · stand | 0.587 | 0.604 | 0.586 | 0.692 | 0.711 | 0.636 | 0.694 |
| tripod · fwd | 0.576 | 0.590 | 0.618 | 0.642 | 0.650 | 0.320 | 0.282 |
| tripod · side | 0.476 | 0.496 | 0.502 | 0.461 | 0.445 | 0.413 | 0.450 |
| tripod · back | 0.623 | 0.586 | 0.622 | 0.646 | 0.636 | 0.386 | 0.266 |
| tripod · diag | 0.396 | 0.385 | 0.404 | 0.436 | 0.423 | 0.299 | 0.238 |
| tripod · turn | 0.537 | 0.643 | 0.613 | 0.549 | 0.513 | 0.596 | 0.668 |
| tripod · move_turn | 0.504 | 0.516 | 0.585 | 0.480 | 0.489 | 0.419 | 0.336 |
| gait_hz · stand | 0.598 | 0.665 | 0.568 | 0.433 | 0.446 | 0.366 | 0.232 |
| gait_hz · fwd | 0.360 | 0.366 | 0.366 | 0.433 | 0.433 | 0.360 | 0.269 |
| gait_hz · side | 0.336 | 0.330 | 0.330 | 0.470 | 0.458 | 0.372 | 0.238 |
| gait_hz · back | 0.336 | 0.330 | 0.330 | 0.439 | 0.439 | 0.366 | 0.253 |
| gait_hz · diag | 0.409 | 0.403 | 0.412 | 0.446 | 0.443 | 0.372 | 0.244 |
| gait_hz · turn | 0.494 | 0.494 | 0.500 | 0.397 | 0.391 | 0.372 | 0.232 |
| gait_hz · move_turn | 0.494 | 0.494 | 0.494 | 0.439 | 0.421 | 0.372 | 0.244 |
| cadence_ratio · stand | 1.994 | 2.082 | 1.856 | 3.311 | 3.510 | 2.841 | 1.994 |
| cadence_ratio · fwd | 0.985 | 0.991 | 0.989 | 1.135 | 1.196 | 0.976 | 0.752 |
| cadence_ratio · side | 1.016 | 1.006 | 1.006 | 1.390 | 1.378 | 1.113 | 0.738 |
| cadence_ratio · back | 0.995 | 0.993 | 0.996 | 1.244 | 1.334 | 1.108 | 0.777 |
| cadence_ratio · diag | 0.991 | 0.987 | 0.983 | 1.055 | 1.090 | 0.979 | 0.677 |
| cadence_ratio · turn | 0.989 | 0.989 | 1.001 | 0.802 | 0.781 | 0.745 | 0.464 |
| cadence_ratio · move_turn | 0.989 | 0.989 | 0.989 | 0.879 | 0.842 | 0.745 | 0.488 |
| offset_R · fwd | 0.987 | 0.981 | 0.985 | 0.150 | 0.154 | 0.146 | 0.039 |
| offset_R · side | 0.974 | 0.978 | 0.966 | 0.083 | 0.109 | 0.063 | 0.019 |
| offset_R · back | 0.976 | 0.964 | 0.965 | 0.049 | 0.125 | 0.036 | 0.106 |
| offset_R · diag | 0.967 | 0.967 | 0.979 | 0.034 | 0.058 | 0.202 | 0.115 |
| offset_R · turn | 0.995 | 0.997 | 0.997 | 0.123 | 0.089 | 0.084 | 0.039 |
| offset_R · move_turn | 0.989 | 0.981 | 0.993 | 0.123 | 0.028 | 0.034 | 0.037 |
| swing_apex · fwd | 0.116 | 0.119 | 0.127 | 0.112 | 0.112 | 0.093 | 0.037 |
| swing_apex · side | 0.064 | 0.068 | 0.070 | 0.067 | 0.068 | 0.057 | 0.034 |
| swing_apex · back | 0.129 | 0.129 | 0.134 | 0.117 | 0.112 | 0.104 | 0.032 |
| swing_apex · diag | 0.064 | 0.067 | 0.069 | 0.075 | 0.074 | 0.058 | 0.029 |
| swing_apex · turn | 0.094 | 0.107 | 0.106 | 0.106 | 0.103 | 0.130 | 0.072 |
| swing_apex · move_turn | 0.090 | 0.098 | 0.097 | 0.100 | 0.098 | 0.098 | 0.039 |
| taps_per_m · fwd | 12.721 | 13.652 | 11.499 | 12.897 | 11.921 | 24.059 | 11.026 |
| taps_per_m · side | 38.866 | 36.290 | 38.518 | 40.478 | 37.166 | 39.438 | 33.676 |
| taps_per_m · back | 13.042 | 14.865 | 14.080 | 13.767 | 13.780 | 27.884 | 15.050 |
| taps_per_m · diag | 27.159 | 24.903 | 26.024 | 25.131 | 24.442 | 28.172 | 15.295 |
| taps_per_rad · turn | 22.582 | 22.526 | 19.884 | 23.668 | 22.034 | 46.844 | 174.902 |
| taps_per_rad · move_turn | 55.776 | 47.224 | 40.171 | 54.176 | 35.839 | 87.796 | 49.560 |
| action_jitter · stand | 0.139 | 0.139 | 0.116 | 0.130 | 0.130 | 0.223 | 0.197 |
| action_jitter · fwd | 0.126 | 0.124 | 0.108 | 0.128 | 0.128 | 0.181 | 0.160 |
| action_jitter · side | 0.109 | 0.107 | 0.099 | 0.116 | 0.121 | 0.150 | 0.139 |
| action_jitter · back | 0.128 | 0.126 | 0.108 | 0.129 | 0.126 | 0.192 | 0.157 |
| action_jitter · diag | 0.120 | 0.118 | 0.108 | 0.125 | 0.126 | 0.171 | 0.156 |
| action_jitter · turn | 0.123 | 0.123 | 0.103 | 0.134 | 0.132 | 0.219 | 0.196 |
| action_jitter · move_turn | 0.128 | 0.128 | 0.111 | 0.127 | 0.124 | 0.191 | 0.159 |

### A3000 vs X0 s3 at c4 (A3000's last eval)

| Metric | A3000 c4 | X0 s3 c4 | Δ (A3000 − X0) |
|---|---|---|---|
| survival · stand | 1.000 | 1.000 | 0.000 |
| survival · fwd | 0.740 | 0.870 | -0.130 |
| survival · side | 0.990 | 0.940 | +0.050 |
| survival · back | 0.810 | 0.850 | -0.040 |
| survival · diag | 0.590 | 0.620 | -0.030 |
| survival · turn | 0.990 | 0.975 | +0.015 |
| survival · move_turn | 0.190 | 0.795 | -0.605 |
| ratio · fwd | 0.678 | 0.698 | -0.020 |
| ratio · side | 0.490 | 0.472 | +0.017 |
| ratio · back | 0.701 | 0.633 | +0.068 |
| ratio · move_turn | 0.503 | 0.473 | +0.030 |
| facing_closed · turn | 0.419 | 0.505 | -0.086 |
| yaw_progress · turn | 0.055 | 0.067 | -0.012 |
| facing_closed · move_turn | 0.120 | 0.134 | -0.014 |
| yaw_progress · move_turn | 0.030 | 0.018 | +0.013 |
| steps_per_m · fwd | 17.831 | 18.725 | -0.893 |
| steps_per_m · side | 29.466 | 31.246 | -1.780 |
| steps_per_m · back | 18.911 | 22.157 | -3.245 |
| steps_per_rad · turn | 53.070 | 51.513 | +1.557 |
| support_feet · fwd | 2.998 | 2.957 | +0.041 |
| support_feet · side | 3.577 | 3.510 | +0.067 |
| support_feet · back | 3.023 | 2.952 | +0.071 |
| support_feet · turn | 3.139 | 3.042 | +0.097 |
| foot_slip · fwd | 0.150 | 0.148 | +0.002 |
| foot_slip · side | 0.213 | 0.199 | +0.014 |
| foot_slip · back | 0.151 | 0.146 | +0.005 |
| foot_slip · turn | 0.136 | 0.123 | +0.013 |
| steps_per_s · stand | 2.417 | 3.375 | -0.958 |
| tripod · fwd | 0.538 | 0.577 | -0.039 |
| tripod · side | 0.449 | 0.463 | -0.014 |
| tripod · back | 0.586 | 0.525 | +0.061 |
| tripod · turn | 0.548 | 0.600 | -0.053 |
| tripod · move_turn | 0.458 | 0.516 | -0.058 |

## Z formation readout (never clocked)

The t3 readout needs a moving ratio ≥ 0.30 on fwd, side and back, and survival at or above the floors (t3 `min_new` 0.10).
- **Ratio:** passes on both seeds (0.456–0.755).
- **Survival:** fails on both seeds. fwd and back are 0.000; diag and move_turn are untrained at t3.

The X0 c6 columns are for reference.

| Metric | Z s3 t3 | Z s2 t3 | Z s3 c6 | Z s2 c6 | X0 s3 c6 | X0 s2 c6 |
|---|---|---|---|---|---|---|
| survival · stand | 0.690 | 0.930 | 0.810 | 0.990 | 0.990 | 0.990 |
| survival · fwd | 0.000 | 0.000 | 0.460 | 0.000 | 0.850 | 0.950 |
| survival · side | 0.695 | 0.615 | 0.865 | 0.875 | 0.970 | 0.985 |
| survival · back | 0.000 | 0.000 | 0.560 | 0.010 | 0.880 | 0.920 |
| survival · diag | 0.010 | 0.000 | 0.300 | 0.000 | 0.720 | 0.740 |
| survival · turn | 0.750 | 0.925 | 0.805 | 0.980 | 0.985 | 0.985 |
| survival · move_turn | 0.000 | 0.000 | 0.550 | 0.000 | 0.745 | 0.980 |
| ratio · fwd | 0.660 | 0.755 | 0.466 | 0.503 | 0.733 | 0.740 |
| ratio · side | 0.498 | 0.456 | 0.514 | 0.419 | 0.396 | 0.389 |
| ratio · back | 0.642 | 0.717 | 0.478 | 0.503 | 0.746 | 0.696 |
| ratio · diag | 0.688 | 0.823 | 0.548 | 0.609 | 0.539 | 0.544 |
| ratio · move_turn | 0.656 | 0.677 | 0.483 | 0.467 | 0.498 | 0.514 |
| facing_closed · turn | 0.062 | 0.067 | 0.508 | 0.076 | 0.629 | 0.560 |
| yaw_progress · turn | 0.009 | 0.009 | 0.065 | 0.010 | 0.082 | 0.074 |
| facing_closed · move_turn | 0.018 | 0.059 | 0.185 | 0.059 | 0.330 | 0.365 |
| yaw_progress · move_turn | 0.009 | 0.034 | 0.032 | 0.022 | 0.048 | 0.048 |
| steps_per_m · fwd | 21.731 | 13.720 | 28.750 | 19.056 | 16.339 | 15.133 |
| steps_per_m · side | 26.902 | 22.792 | 26.037 | 27.552 | 35.112 | 34.729 |
| steps_per_m · back | 24.425 | 14.578 | 31.932 | 24.225 | 17.352 | 17.083 |
| steps_per_rad · turn | 203.757 | 184.295 | 47.495 | 193.544 | 39.863 | 44.703 |
| support_feet · stand | 2.977 | 2.942 | 3.043 | 2.918 | 3.262 | 3.159 |
| support_feet · fwd | 2.754 | 2.531 | 2.815 | 2.517 | 3.061 | 3.008 |
| support_feet · side | 3.585 | 3.334 | 3.634 | 3.423 | 3.528 | 3.518 |
| support_feet · back | 2.749 | 2.541 | 2.887 | 2.585 | 3.050 | 3.016 |
| support_feet · turn | 2.937 | 2.890 | 2.978 | 2.888 | 3.187 | 3.097 |
| steps_per_s · stand | 3.167 | 1.833 | 2.917 | 2.417 | 3.500 | 3.167 |
| foot_slip · fwd | 0.134 | 0.084 | 0.136 | 0.092 | 0.147 | 0.139 |
| foot_slip · side | 0.227 | 0.175 | 0.224 | 0.173 | 0.173 | 0.178 |
| foot_slip · back | 0.136 | 0.092 | 0.138 | 0.095 | 0.143 | 0.140 |
| foot_slip · turn | 0.139 | 0.082 | 0.142 | 0.086 | 0.115 | 0.112 |
| tripod · fwd | 0.193 | 0.271 | 0.320 | 0.282 | 0.590 | 0.618 |
| tripod · side | 0.385 | 0.514 | 0.413 | 0.450 | 0.496 | 0.502 |
| tripod · back | 0.194 | 0.234 | 0.386 | 0.266 | 0.586 | 0.622 |
| tripod · turn | 0.539 | 0.737 | 0.596 | 0.668 | 0.643 | 0.613 |

## Appendix A: zero-shot clock hiding (same checkpoint, clock obs zeroed at eval)

| Metric | X08:t3 vis (E3) | X08:t3 hid (E4) | X08:c6 vis (E1) | X08:c6 hid (E5) | X0s3:c4 vis | X0s3:c4 hid (E8) | X0s3:c5 vis | X0s3:c5 hid (E9) |
|---|---|---|---|---|---|---|---|---|
| survival · stand | 0.990 | 1.000 | 0.990 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| survival · fwd | 0.730 | 0.800 | 0.920 | 0.410 | 0.870 | 0.460 | 0.880 | 0.210 |
| survival · side | 0.920 | 0.990 | 0.975 | 0.995 | 0.940 | 0.995 | 0.980 | 1.000 |
| survival · back | 0.780 | 0.910 | 0.920 | 0.810 | 0.850 | 0.700 | 0.870 | 0.580 |
| survival · diag | 0.660 | 0.280 | 0.790 | 0.410 | 0.620 | 0.450 | 0.720 | 0.240 |
| survival · turn | 0.965 | 1.000 | 0.950 | 0.985 | 0.975 | 0.985 | 0.990 | 0.995 |
| survival · move_turn | 0.665 | 0.640 | 0.750 | 0.645 | 0.795 | 0.570 | 0.870 | 0.715 |
| ratio · fwd | 0.558 | 0.316 | 0.748 | 0.475 | 0.698 | 0.559 | 0.718 | 0.480 |
| ratio · side | 0.478 | 0.390 | 0.430 | 0.407 | 0.472 | 0.407 | 0.404 | 0.311 |
| ratio · back | 0.524 | 0.401 | 0.736 | 0.509 | 0.633 | 0.554 | 0.706 | 0.425 |
| ratio · diag | 0.521 | 0.478 | 0.527 | 0.516 | 0.587 | 0.516 | 0.550 | 0.447 |
| ratio · move_turn | 0.430 | 0.202 | 0.424 | 0.170 | 0.473 | 0.273 | 0.497 | 0.172 |
| facing_closed · turn | 0.536 | 0.174 | 0.618 | 0.185 | 0.505 | 0.320 | 0.492 | 0.092 |
| yaw_progress · turn | 0.072 | 0.023 | 0.083 | 0.025 | 0.067 | 0.043 | 0.066 | 0.013 |
| facing_closed · move_turn | 0.174 | -0.007 | 0.235 | 0.055 | 0.134 | 0.026 | 0.290 | -0.001 |
| yaw_progress · move_turn | 0.026 | -0.001 | 0.035 | 0.007 | 0.018 | 0.005 | 0.038 | -0.001 |
| support_feet · stand | 3.050 | 3.312 | 3.144 | 3.445 | 3.049 | 3.195 | 3.227 | 3.547 |
| support_feet · fwd | 2.947 | 3.229 | 3.023 | 3.211 | 2.957 | 3.036 | 3.049 | 3.218 |
| support_feet · turn | 3.007 | 3.242 | 3.120 | 3.282 | 3.042 | 3.184 | 3.196 | 3.234 |
| steps_per_s · stand | 3.167 | 4.167 | 3.083 | 3.750 | 3.375 | 4.083 | 3.250 | 3.750 |

## Appendix B: terrain WATCH (`light__omni`, 100 episodes each)

| Row | Checkpoint | Completion | Tracking (median) | Slip ratio (median) | Tripod (median) | Falls / n |
|---|---|---|---|---|---|---|
| T1 X0 s3:c6 | X01 c6 model_7996 | 0.460 | 0.531 | 0.364 | 0.515 | 54 / 100 |
| T2 A4500:c6 | X04 c6 model_7996 | 0.430 | 0.581 | 0.399 | 0.524 | 57 / 100 |

## Sources

The rows are `summarize()` of these stamps, under `parkour/logs/rsl_rl/gait_eval/omni_phases/2026-09-23_clock_deprecation/`.
Each is the newest `seed001` stamp for its row.

| Row | Stamp |
|---|---|
| step0_E1_X08-c6_visible | `step0_E1_X08-c6_visible/suite__omni/seed001/2026-09-23_21-23-41` |
| seed3_X01_c6_A15pB | `seed3_X01_c6_A15pB/suite__omni/seed001/2026-09-24_00-55-05` |
| seed2_X02_c6_A15pB | `seed2_X02_c6_A15pB/suite__omni/seed001/2026-09-24_03-11-37` |
| seed3_X03_c6_A15pB | `seed3_X03_c6_A15pB/suite__omni/seed001/2026-09-24_14-18-43` |
| seed3_X04_c6_A15pB | `seed3_X04_c6_A15pB/suite__omni/seed001/2026-09-24_16-39-45` |
| seed3_X05_c6_A15pB | `seed3_X05_c6_A15pB/suite__omni/seed001/2026-09-25_06-52-47` |
| seed2_X07_c6_A15pB | `seed2_X07_c6_A15pB/suite__omni/seed001/2026-09-25_12-17-09` |
| seed3_X08_c4_A15pB | `seed3_X08_c4_A15pB/suite__omni/seed001/2026-09-25_13-01-06` |
| seed3_X01_c4_A15pB | `seed3_X01_c4_A15pB/suite__omni/seed001/2026-09-23_23-11-33` |
| seed3_X05_t3_A15pB | `seed3_X05_t3_A15pB/suite__omni/seed001/2026-09-25_04-33-44` |
| seed2_X07_t3_A15pB | `seed2_X07_t3_A15pB/suite__omni/seed001/2026-09-25_09-56-06` |
| step0_E4_X08-t3_hidden | `step0_E4_X08-t3_hidden/suite__omni/seed001/2026-09-23_21-45-11` |
| step0_E3_X08-t3_visible | `step0_E3_X08-t3_visible/suite__omni/seed001/2026-09-23_21-38-11` |
| step0_E5_X08-c6_hidden | `step0_E5_X08-c6_hidden/suite__omni/seed001/2026-09-23_21-52-41` |
| step0_E8_X0s3-c4_hidden | `step0_E8_X0s3-c4_hidden/suite__omni/seed001/2026-09-24_03-19-22` |
| step0_E9_X0s3-c5_hidden | `step0_E9_X0s3-c5_hidden/suite__omni/seed001/2026-09-24_03-26-22` |
| seed3_X01_c5_A15pB | `seed3_X01_c5_A15pB/suite__omni/seed001/2026-09-24_00-00-34` |
| step0_E2_X08-c4_visible | `step0_E2_X08-c4_visible/suite__omni/seed001/2026-09-23_21-31-11` |
| s4_T1_X0s3-c6_light | `s4_T1_X0s3-c6_light/light__omni/seed001/2026-09-25_13-11-35` |
| s4_T2_A4500-c6_light | `s4_T2_A4500-c6_light/light__omni/seed001/2026-09-25_13-15-05` |
