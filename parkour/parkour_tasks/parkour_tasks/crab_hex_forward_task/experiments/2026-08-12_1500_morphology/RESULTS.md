<!-- paths-note -->
> **Paths note (2026-09-09):** this campaign moved from `sim_fine_tuning/2026-08-12_1500_morphology/` to `parkour/parkour_tasks/parkour_tasks/crab_hex_forward_task/experiments/2026-08-12_1500_morphology/`. Absolute paths below (`/home/.../sim_fine_tuning/...`, `parkour/logs/rsl_rl/...`) name raw artifacts that stay on disk untracked; the tracked checkpoint of record is `head/` (see `bundle.yaml`) and the eval summaries are in place / under `evals/`.

# Morphology campaign: plant-side anchors of the lean and duty asymmetry

Successor to three reward campaigns (tripod v1-v5+b6/b7, lean L-series, stride S-series) that
proved the +12° lean and 0.146/0.556 duty split are immune to reward configuration. User
directive 2026-08-12: "move forward with those next steps" (measure, then intervene).

## M1/M2: static measurements (measure_static_posture.py, 300 zero-action steps, 1 env)

| hypothesis | measurement | verdict |
|---|---|---|
| lean anchored in default posture | equilibrium pitch **+0.0096 rad (+0.55°)**, roll −0.33° | **REFUTED** — plant stands level; the lean is a locomotion choice |
| CoM forward of support | longitudinal offset **−3.3 mm** (aft) | **REFUTED** — CoM is on the centroid |
| default pose loads tripod B | static forces: FL 163, FR **252**, ML 134, MR **64**, RL 198, RR 180 N → **A share 42.8%** | **CONFIRMED** — B-tilted 57/43 at zero action, level body; FR carries 4× MR |

The knee-default hand-tune (left −0.07 / right +0.10) balanced roll (L 495 vs R 497 N ✓) but
left the diagonal untouched — and the diagonal is exactly what the tripod sets sample. The
policy's 79/21 duty split is the trained amplification of this 57/43 static seed.

Note: sim total mass 106 kg (body 66) vs URDF reference ~23 kg — the known auto-computed
base-mass discrepancy; irrelevant to the within-sim asymmetry, relevant to sim-to-real later.

## M4: static-load symmetrization (in progress)

Search over default joint angles (per-leg knee, then hip-femur if needed) via the measurement
script's override flag — target all six feet at 165±15 N with pitch/roll ≤1°. Found values then
baked into crab_hex_scene_cfg.py (revertable commit) and validated by M5: fine-tune healthy
19999 on the symmetrized plant, standard two-point eval — watch duty_A, pitch, tripod.

## M4: static-load symmetrization — ABANDONED (objective ill-posed); M4b statistics decisive

The knee-angle Newton solver diverged (rms 44→61 N, steps flip-flopping at clamps) and exposed
why: the hyperstatic 6-foot force distribution is **path-dependent** — feet stick where friction
catches them during settle, so per-settle force maps are a landing lottery (probes swung A-share
33-65% on ±0.02 rad knee changes). Equalizing one settle's map is meaningless.

**M4b (decisive)**: A-share over 20 jittered zero-action settles = **51.0% ± 2.1%**. The plant
is statically A/B-balanced; M1b's 57/43 was one lottery draw. (Systematic structure exists but
is A/B-neutral: settles consistently rest on a FL/ML/MR/RR tetrapod with FR+RL floating at
~35 N — one floater from each tripod set; a front/mid cam-phase effect, not a set bias.)

## Campaign conclusion: the anchor is the BASIN, not the plant

| suspected anchor | measurement | verdict |
|---|---|---|
| default posture leans | equilibrium +0.55° | plant stands level |
| CoM forward | −3.3 mm | centered |
| static A/B load bias | 51.0% ± 2.1% | balanced |

Combined with the reward campaigns: the +12° lean and B-duty preference are neither
plant-forced nor reward-priced — they are **learned symmetry-breaking locked into the baseline
policy's basin** (a tripod gait must elect a lead set; the original 20k run's basin chose B and
a leaned carry). Supporting evidence: from-scratch runs in other basins show different leans
(v5's unison-glide: +0.13 rad vs 0.209), and 2000-iter fine-tunes never escape a basin.
The previous "morphology-locked" conclusion is REPLACED by direct measurement: morphology is
innocent; history is guilty. The route to tripod ≥0.45 is **basin selection at from-scratch
time** — a deliberate multi-seed from-scratch search (the basin lottery run as a search, scored
by the deterministic eval, with v5 active to amplify any alternating basin found) or
symmetry-enforced training. Proposal presented to user; campaign closed.
