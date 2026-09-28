<!-- paths-note -->
> **Paths note (2026-09-09):** this campaign moved from `sim_fine_tuning/2026-08-12_1305_stride_definancing/` to `parkour/parkour_tasks/parkour_tasks/crab_hex_forward_task/experiments/2026-08-12_1305_stride_definancing/`. Absolute paths below (`/home/.../sim_fine_tuning/...`, `parkour/logs/rsl_rl/...`) name raw artifacts that stay on disk untracked; the tracked checkpoint of record is `head/` (see `bundle.yaml`) and the eval summaries are in place / under `evals/`.

# Stride-definancing campaign (S-series): remove the income financing the duty asymmetry

Successor to `2026-08-12_1017_lean_reduction/` (closed: lean invariant to all penalties). New
mechanism, user-approved 2026-08-12: offline attribution on the baseline traces showed
**98.1% of reward_stride_length income goes to the B tripod set** — only 61/1740 A-set stances
(3.5%) clear the 0.1s `min_phase_duration` payment floor vs 84% of B-set stances, and the floor
(not the power=2 convexity) is the dominant gatekeeper (at power=1 the split stays 97.4% B).
The term that fixed tippy-tap is bankrolling the 0.146/0.556 duty asymmetry; every lean tax was
outbid by this income. The S-series defunds the asymmetry instead of taxing the posture.

## Protocol

Same as the lean campaign: each arm fine-tunes from healthy model_19999, 2000 iters, 256 envs,
seed 1; two-point deterministic gait-eval (mid ~21000 + final ~21998); one change per arm;
serial training, evals overlapped. All changes are Hydra weight/param overrides — no code.

**Success axes**: duty_A rising off 0.146 and/or signed pitch off +0.209 (first structural budge
in three campaigns = KEPT); tripod ≥0.45 at two consecutive checkpoints = bake bar.
**Guardrails** (breach = REVERT): completion=100%, slip ≤3.5%, roll ≥0.03, EMA ≤0.174,
stride ∈[0.11,0.22]m, tippy ≤8% (tippy-tap relapse watch — this campaign touches the anti-tippy
term itself; baseline 6.5%).

## Baseline

tripod 0.401 | pitch +0.209 | duty 0.146/0.556 | pearson −0.606 | stride-income split A/B
1.9%/98.1% | completion 100% | slip 2.6% | roll 0.0387 | EMA 0.134 | stride 0.163 | tippy 6.5%

## Arms

| arm | override(s) | rationale |
|---|---|---|
| S1 | reward_stride_length.params.min_phase_duration 0.1→0.05 | the dominant gatekeeper; lets A-set stances earn at all |
| S2 | params.power 2.0→1.0 + weight 0.5→0.05 (scale-matched, measured 9.5×) | removes concentration preference at constant income magnitude |
| S3 | S1+S2 combined | fully participation-neutral stride income; only if singles move |
| S4 | best mover + reward_tripod_schedule 0.3 | v5 amplifier combo, only if duty/pitch shifts |

## Runs

| S1 | min_phase_duration 0.1→0.05 | mid/final: pitch +0.211/+0.210, tripod 0.405/0.387, tippy 7.0/**6.5%** (NO relapse), duty 0.142/0.557, completion 100/100%, slip 2.3/2.3% | **INERT on structure, instrument validated**: A-set paid stances 3.5%→**43%** (748/1730) — the floor was the participation gate exactly as attributed — but income split only 1.9→4.6% A because power=2 pays an A tap (~0.03m²) 9× less than a B stance (~0.09m²). Floor gates participation; convexity gates income. |

**Sequence adjustment (analysis-driven)**: S2 alone (power=1, floor 0.1) predictably cannot move
the split either — baseline traces at power=1 measure 2.6% A (participation still floor-gated).
S3 (floor 0.05 + power 1.0 + weight 0.05) is the only rule where a marginal A stance is worth
taking (predicted split ~12% A — income can never fully balance while duty is 0.14/0.56, since
income mechanically follows duty; the test is whether marginal-A-incentive changes duty). S3
therefore runs unconditionally after S2; the original "only if singles move" condition was
mis-calibrated against this arithmetic.

| S2 | power 2.0→1.0 + weight 0.5→0.05 (scale-matched) | mid/final: pitch +0.211/+0.209, tripod 0.395/0.420, tippy 7.8/7.0%, completion 100/100%, slip 2.5/2.3%, duty 0.144/0.556 | **INERT as the S1 arithmetic predicted** — with the 0.1s floor still excluding A participation, the exponent has nothing to rebalance. Guardrails clean. Singles complete; S3 (floor+power) is the hypothesis test. |

| S3 | floor 0.05 + power 1.0 + weight 0.05 | mid/final: pitch +0.2087/+0.2082, tripod 0.406/**0.4255**, tippy 5.9/7.2%, slip 2.1/2.3%, EMA 0.136/**0.128**, completion 100/100%, duty 0.142/0.564 | **Income split moved exactly as predicted (1.9%→12.0% A, 768 A-stances paid) — duty and pitch did not.** Guardrails clean, several axes marginally better than baseline. S4 not launched: its premise (income moves duty) is refuted. |

## Campaign summary: income follows duty — CLOSED

The S-series answered its question in three steps: S1 proved the payment floor was the
participation gate (A paid stances 3.5%→43%); S2 proved the exponent alone changes nothing
while the floor gates; S3 removed both gates, delivered the predicted 6× income rebalance
(12.0% A) — and the gait did not move a millimeter of duty or a milliradian of pitch.

**The stride reward finances the asymmetry only in the accounting sense; it does not cause
it.** Three campaigns now triangulate the same diagnosis from independent directions:
1. Rewarding alternation directly (tripod campaign, v1-v5 + b6/b7): timing sharpens, duty immune.
2. Taxing the lean (L-series): pitch invariant to every penalty at up to 5× doses.
3. Defunding the asymmetry (S-series): accounting moves, behavior doesn't.

The 14%/56% duty split and +12° lean are plant-side: CoM placement, cam-mechanism foot-path
geometry, and/or default posture. Reward configuration — functions, weights, and params alike —
is exhausted as a route to tripod >0.45. Recommended next: (a) spawn/default-posture experiment
(cheapest), (b) CoM shift in the model, (c) cam-geometry workspace analysis; the eval pipeline,
deterministic checkpoint selection, and these ledgers carry over unchanged.
