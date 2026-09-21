# Stance sensing: current-sense proxy vs footpad contact, on a real gait (2026-09-21)

**Question.** How often does the motor-current stance proxy the policy observes agree with the
privileged footpad contact sensor — on the A15+B plant of record, under a gait a trained head
actually produces, rather than the scripted sine of 2026-08-20?

This closes the open item the hardware-morphology campaign left behind
([2026-08-20_1506_hardware_morphology/RESULTS.md](../2026-08-20_1506_hardware_morphology/RESULTS.md),
"Open / next": *"Current-sense proxy: retune `_CURRENT_SENSE_*` thresholds (precision ~0.72 vs 0.95
with the soft drives) and rerun the stance study on a real trained gait, not the scripted sine."*)
Nothing between 2026-08-26 and today re-measured it, and the plant changed (A15+B) in between.

**Answer.** On a real gait the proxy carries a genuine stance signal — ROC AUC 0.83–0.85, per-leg
0.92–0.97 — but **no single threshold can read it**, because the middle legs sit at a different
current offset from the corner legs. One global threshold tops out at **0.76–0.79 agreement**;
per-leg thresholds reach **0.92**. The 2026-08-20 precision collapse was not the sensing model
failing, it was one threshold serving two load regimes.

## What was run

[`stance_current_probe_v2.py`](stance_current_probe_v2.py) — successor to the 2026-08-20 probe.
Changes: a `--mode policy` that lets a trained checkpoint drive (the scripted sine held feet down
72–97 % of the time and barely exercised the false-positive side); agreement and the full confusion
split in the report, not only precision/recall/F1, plus the always-stance baseline, per-leg
breakdown and ROC AUC; and the proxy is read from the **live observation term**
(`CrabHexParkourObservations._get_contact_fill`) instead of a second copy of the sensing model.
`--mode scripted` reproduces the 2026-08-20 test exactly.

Ground truth throughout is the privileged `ContactSensor`, >5 N on a footpad = stance.

| run | mode | head | env | leg-steps | report |
|---|---|---|---|---|---|
| scripted | sine, env 0 | — | Flat-Walk-Play, A15+B | 4,800 | [report_scripted_A15pB.json](report_scripted_A15pB.json) |
| 2c | policy | `policy/2c_teacher/model_19996.pt` | Teacher-Play, `KRABBY_PHASE=2c`, 32 envs × 1200 | 227,550 | [report_policy_2c.json](report_policy_2c.json) |
| 1a | policy | `policy/1a_formation/model_4999.pt` | Flat-Walk-Play, `KRABBY_PHASE=1a`, 32 envs × 900 | 170,580 | [report_policy_1a.json](report_policy_1a.json) |

`live_vs_recomputed_max_abs_gap = 0.0` in all three: the 2026-08-20 recomputation from primitives is
still bit-identical to what the policy sees.

## The scripted sine, old plant vs new

Same motion, so these are comparable to the 2026-08-20 `final4` row:

| | final4 (2026-08-26) | A15+B (today) |
|---|---|---|
| true stance fraction | 0.722 | 0.838 |
| recall @0.1 | 0.982 | 0.963 |
| precision @0.1 | 0.723 | 0.840 |
| agreement @0.1 | 0.72 | 0.815 |
| always-stance baseline | 0.72 | 0.838 |
| dead zone (% of stance) | 1.7 | 2.9 |

Unchanged verdict on this test, and the reason it was never conclusive: agreement (0.815) sits
*below* the trivial always-stance predictor (0.838), specificity @0.1 is 0.049. The sine barely
lifts feet, so there is almost no swing to get wrong.

## The trained gait

Prevalence is balanced (0.50 / 0.49), so the coin-flip baseline is 0.50 and the false-positive side
is finally exercised. Global, best single threshold:

| | 2c teacher | 1a formation |
|---|---|---|
| best threshold | 0.3 | 0.5 (0.3 within noise) |
| agreement | 0.762 | 0.788 |
| recall / precision | 0.929 / 0.697 | 0.893 / 0.731 |
| balanced accuracy | 0.763 | 0.791 |
| ROC AUC | 0.830 | 0.845 |
| dead zone (% of stance) | 1.0 | 0.2 |
| mean vx, resets | 0.19 m/s, 16 | 0.18 m/s, 5 |

Per leg (2c / 1a), where the result actually lives:

| leg | proxy in stance | proxy in swing | own best thr | agreement | AUC |
|---|---|---|---|---|---|
| FL | 1.10 / 1.10 | 0.23 / 0.25 | 0.35 / 0.40 | 0.917 / 0.909 | 0.946 / 0.941 |
| FR | 1.10 / 1.11 | 0.24 / 0.28 | 0.40 / 0.40 | 0.912 / 0.883 | 0.942 / 0.929 |
| ML | 1.47 / 1.47 | 0.77 / 0.77 | 1.20 / 1.45 | 0.950 / 0.945 | 0.953 / 0.958 |
| MR | 1.47 / 1.47 | 0.74 / 0.75 | 1.25 / 1.40 | 0.970 / 0.952 | 0.973 / 0.966 |
| RL | 1.04 / 1.11 | 0.19 / 0.20 | 0.35 / 0.35 | 0.897 / 0.909 | 0.928 / 0.951 |
| RR | 1.02 / 1.04 | 0.20 / 0.19 | 0.35 / 0.35 | 0.888 / 0.904 | 0.920 / 0.941 |

The middle legs are not blind — they are **offset**. Their swing current (~0.75) sits where the
corner legs' *stance* current begins, so any threshold tuned for the corners calls ML/MR stance
permanently (specificity 0.06). Their separation is in fact the cleanest of the six (AUC 0.95/0.97);
they carry more load through a different moment arm, so both their stance and swing currents ride high.

**Per-leg thresholds fix it, and they transfer across heads:**

| threshold set | agreement on 2c | agreement on 1a |
|---|---|---|
| single 0.3 (global best) | 0.762 | 0.780 |
| 2c-tuned per leg | 0.922 | 0.915 |
| 1a-tuned per leg | 0.922 | 0.917 |
| **corners 0.375 / middles 1.2** | **0.922** | **0.915** |

A threshold set tuned on one head loses ~0.002 on the other — the offsets are a property of the
plant and the leg row, not of the policy.

## What this means

- **For the sim MDP: nothing is broken.** The policy sees six per-leg continuous channels in fixed
  order and can learn each leg's own offset; it never needed a shared threshold. Retuning
  `_CURRENT_SENSE_*` is not indicated by this measurement, and would fork every checkpoint.
- **For hardware: the stance bit needs per-leg calibration.** Firmware inferring stance from motor
  current must calibrate per actuator (middle row ≈ 3× the corner-row threshold), not ship one
  constant. With that, ~0.92 agreement against a contact sensor is the realistic expectation, i.e.
  roughly one leg-step in twelve disagrees, most of them late-swing/early-stance edges.
- **The worm-drive dead zone is a non-issue on a real gait** (0.2–1.0 % of stance time, vs ~2–3 %
  under the sine): a walking policy is driving something on every leg nearly all the time.

## Caveats

- Two heads, one plant, flat-ish terrain (phase 1a/2c play configs). Obstacle terrain is unmeasured.
- Ground truth is a 5 N footpad threshold; light-touch contacts count as stance, so some of the
  residual 8 % disagreement is threshold-edge, not sensing error.
- The 3a student head (the head that ships) was not measured — the probe's policy mode does not
  carry the depth branch. Its gait is distilled from 2c, so the per-leg structure should hold, but
  that is an inference, not a measurement.
