# S2 readout: X0 sanity vs X08:c6, and RIG power (2026-09-23)

Plan step 11, S2. Read-only; nothing trained or evaluated. Every number below comes from the driver's own code:
- `run_candidates.summarize_run_dir`, which calls `resolution.summarize_run_dir`, which reads `run_phases.summarize()['cat']`;
- `_screen_compare`, which uses `metric_rows`, `block_stat`, `classify`, `block_resolution`, `guard_rows` and `binomial_floor`.

Inputs:
- the rule is `candidates.yaml` `decision_rule`, byte-identical to the one frozen in D00s;
- R_eval comes from `resolution.36e5a8af7000.json` (D00s);
- R_seed comes from the ledger's D00bs `seed_terms`;
- decisive = present = D00s `decisive_at` (all 7 categories).

**Cross-check (passes).**
- `d00bs_evidence` on the stored X01/X02 screens reproduces the printed D00bs line exactly: F −1.280, F_mt +0.461, T −0.246, RIG −0.786 at eval-only R.
- `_screen_compare` on the re-summarised run dirs gives the same four numbers.
- The re-summarised X01/X02 `cat` tables match the screens the driver stored (0 differences).

| Row | Suite stamp (`parkour/logs/rsl_rl/gait_eval/omni_phases/2026-09-23_clock_deprecation/`) |
|---|---|
| X08:c6 (incumbent) | `step0_E1_X08-c6_visible/suite__omni/seed001/2026-09-23_21-23-41` |
| X0 s3:c6 (X01) | `seed3_X01_c6_A15pB/suite__omni/seed001/2026-09-24_00-55-05` |
| X0 s2:c6 (X02) | `seed2_X02_c6_A15pB/suite__omni/seed001/2026-09-24_03-11-37` |

## 1–3. Blocks at full R

Each delta is sign × (X0 − X08:c6), so a positive z means X0 is better. z = mean_i(δ_i / R_eval,i). The verdict is BETTER if z > R, WORSE if z < −R, otherwise TIE.

How R is set:
- **Single seed:** R = max(R_eval, R_seed).
- **Pooled:** R = max(R_eval/√2, R_seed), and δ is the mean of the s3 and s2 deltas. The base is X08:c6 on both seeds.

| Block | (1) X0 s3: z / R | (2) X0 s2: z / R | (3) pooled: z / R |
|---|---|---|---|
| F | +0.053 / 1.280 **TIE** | −1.227 / 1.280 **TIE** | −0.587 / 1.280 **TIE** |
| F_mt | +2.282 / 0.912 **BETTER** | +2.744 / 0.912 **BETTER** | +2.513 / 0.645 **BETTER** |
| T | −0.395 / 0.487 **TIE** | −0.641 / 0.487 **WORSE** | −0.518 / 0.344 **WORSE** |
| RIG (WATCH) | +0.338 / 0.786 TIE | −0.448 / 0.786 TIE | −0.055 / 0.786 TIE |
| RIG split (steps / support) | −0.012 / +0.350 | −0.056 / −0.392 | −0.034 / −0.021 |

Where R comes from: F's full R is its seed term (1.280). F_mt's is R_eval (0.912), T's is R_eval (0.487) and RIG's is its seed term (0.786). The seed term also sets RIG's pooled R, so it does not shrink when pooled.

What drives each block:
- **T** comes from `side.ratio`: X08 0.430 against X0 0.396 (s3) and 0.389 (s2), z_i −1.86 / −2.27 / −2.07 pooled. That is about −9 % relative on both seeds. `back.ratio` in s2 adds −0.040 (z_i −0.76).
  - Side step rate is flat: 3.17 Hz on X08, 3.25 / 3.08 Hz on X0. So X0 covers less side distance at the same cadence, which is also why `side.steps_per_m` rises.
- **F_mt:** `move_turn.facing_closed` goes from 0.235 to 0.330 / 0.365, and `yaw_progress` from 0.035 to 0.048.
- **F:** `turn.facing_closed` goes from 0.618 to 0.629 / 0.560. The s2 z of −1.227 sits just inside the seed-dominated R of 1.280; at eval-only R (0.966) it is WORSE.

## Survival guards

These are one-sided at 99 %: WORSE iff Δ < −R, with Δ = X0 − X08. R is max(R_eval, R_seed), with a binomial floor.

Pooled guards use the driver's form: R = √((R_eval/√2)² + R_seed²), floored at R_binom·√3/2 (`anchor_s2_reused`, because the X08 screen is reused). The letter after each R names what set it: b = binomial floor, e = eval, s = seed, e+s = both.

| Category | X08 | X0 s3 / s2 | (1) s3: Δ / R | (2) s2: Δ / R | (3) pooled: Δ / R |
|---|---|---|---|---|---|
| stand | 0.990 | 0.990 / 0.990 | 0.000 / 0.072 b | 0.000 / 0.072 b | 0.000 / 0.062 b |
| fwd | 0.920 | 0.850 / 0.950 | −0.070 / 0.110 e | +0.030 / 0.110 e | −0.020 / 0.127 e+s |
| side | 0.975 | 0.970 / 0.985 | −0.005 / 0.051 b | +0.010 / 0.051 b | +0.003 / 0.044 b |
| back | 0.920 | 0.880 / 0.920 | −0.040 / 0.105 e | 0.000 / 0.105 e | −0.020 / 0.084 e+s |
| diag | 0.790 | 0.720 / 0.740 | −0.070 / 0.142 b | −0.050 / 0.140 e | −0.060 / 0.122 b |
| turn | 0.950 | 0.985 / 0.985 | +0.035 / 0.058 e | +0.035 / 0.058 e | +0.035 / 0.044 b |
| move_turn | 0.750 | 0.745 / 0.980 | −0.005 / 0.235 s | +0.230 / 0.235 s | +0.113 / 0.245 e+s |

**All 21 survival readings are TIE.**

## Other guards that read WORSE (the rest of the 23)

These guards were not in the S2 list, but they fall under the plan's "anything WORSE → ask". Values are Δ / R; a negative Δ is worse for every metric.

| Guard | (1) s3 | (2) s2 | (3) pooled |
|---|---|---|---|
| `side.ratio@guard` | −0.033 / 0.021 WORSE | −0.041 / 0.021 WORSE | −0.037 / 0.017 WORSE |
| `side.steps_per_m@guard` (32.9 → 35.1 / 34.7) | −2.18 / 1.57 WORSE | −1.80 / 1.57 WORSE | −1.99 / 1.17 WORSE |
| `back.foot_slip` (0.130 → 0.143 / 0.140) | −0.013 / 0.0066 WORSE | −0.010 / 0.0066 WORSE | −0.0115 / 0.0055 WORSE |
| `fwd.foot_slip` (0.133 → 0.147 / 0.139) | −0.014 / 0.0081 WORSE | −0.006 / 0.0081 TIE | −0.0099 / 0.0096 WORSE |
| `turn.foot_slip` | −0.0057 / 0.0039 WORSE | TIE | −0.0040 / 0.0044 TIE |
| `stand.steps_per_s` (3.08 → 3.50 / 3.17) | −0.42 / 0.33 WORSE | TIE | −0.25 / 0.40 TIE |

`move_turn.ratio` sits above +R in all three readings (0.424 → 0.498 / 0.514).

**Sensitivity of the pooled reading.** The pooled line reuses the single X08:c6 screen as the base on both seeds. The eval part of its variance therefore scales by √3/2, not by the 1/√2 the driver applies to blocks; the driver uses √3/2 only for the survival floor. At √3/2:
- F_mt R becomes 0.790 and stays BETTER.
- T R becomes 0.422 and stays WORSE (z −0.518).
- F and RIG are unchanged, because their R is seed-dominated.
- `fwd.foot_slip` flips to TIE (−0.0099 / 0.0103).
- `side.ratio`, `side.steps_per_m` and `back.foot_slip` stay WORSE.
- All survival guards stay TIE.

**Eval-only reading (WATCH).**
- s3: F TIE; F_mt BETTER; T TIE; RIG BETTER (+0.338 / 0.330, support part +0.350, steps part −0.012).
- s2: F WORSE (−1.227 / 0.966); F_mt BETTER; T WORSE; RIG WORSE (−0.448 / 0.330).
- Pooled: F TIE (−0.587 / 0.683); F_mt BETTER; T WORSE (−0.518 / 0.344); RIG TIE (−0.055 / 0.233).
- Survival is all TIE here too.

## RIG power (from D00bs, ratio system)

The inputs:
- R_RIG = 0.786, seed-dominated. A pooled arm line uses the same R: max(0.330/√2, 0.786) = 0.786.
- RIG has 10 members: 5 steps members and 5 support members.

Three ways to read the MDE:
- **Uniform:** every member moves by R_b·R_i. This is the per-member MDE the driver printed.
- **Steps-only:** only the 5 steps members move, so the MDE is 2·R_b·R_i.
- **Single-member:** one member moves alone, so the MDE is 10·R_b·R_i.

Percentages are of X0's own value (mean of s3 and s2; the per-seed range is in brackets).

| Member | R_i | X0 s3 / s2 / mean | Uniform (as printed) | Steps-only | Single-member |
|---|---|---|---|---|---|
| side `steps_per_m` | 1.377 | 35.11 / 34.73 / 34.92 | 1.082 = **3.1 %** | 2.165 = **6.2 %** | 10.83 = **31.0 %** (30.8–31.2) |
| turn `steps_per_rad` | 3.898 | 39.86 / 44.70 / 42.28 | 3.065 = **7.2 %** (6.9–7.7) | 6.130 = **14.5 %** (13.7–15.4) | 30.65 = **72.5 %** (68.6–76.9) |
| (fwd / back / diag `steps_per_m`) | | | 8.2 / 9.3 / 5.0 % | 16.3 / 18.6 / 10.0 % | 81.7 / 92.8 / 50.2 % |

**The S2 stop rule** is to stop if the side or turn steps MDE exceeds 30 % of X0's value.
- Uniform reading: **PASS** (3.1 % and 7.2 %).
- Steps-only reading: **PASS** (6.2 % and 14.5 %).
- Single-member reading: **FAIL** on both. Side is 31.0 %, only just over; turn is 72.5 %.

For context, the steps guards' own R (99 %, one-sided) is 1.569 for side (4.5 % of X0) and 4.840 for turn (11.4 %).

## Summary for S2

- **WORSE is present, so the plan says ask.**
  - T is WORSE on s2 and pooled at full R.
  - Four guards are WORSE pooled at full R: `side.ratio`, `side.steps_per_m`, `back.foot_slip` and `fwd.foot_slip`. The first three are WORSE on both seeds; `fwd.foot_slip` is WORSE only on s3 and is marginal pooled.
  - Plainly: X0, the tail retrained under the PPO resume fix, walks sideways about 9 % slower than X08:c6 at the same cadence, so it takes about 6 % more steps per metre. It also slips more on back and fwd.
- F_mt is BETTER on every reading: X0's move_turn facing is better.
- F is TIE, and so is RIG (WATCH). All survival readings are TIE.
- Arms are judged against X0, not X08, so this bears on whether results transfer to the recipe of record, not on how an arm is scored.
- Caveat: X08:c6 is one seed-3 screen with no pair of its own. The seed term here is X0's pair, so the X08 recipe's own seed spread (before the resume fix) is not measured.
