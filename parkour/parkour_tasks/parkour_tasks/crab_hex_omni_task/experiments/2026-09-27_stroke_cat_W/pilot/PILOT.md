# W s3 pilot read (decision 5), 2026-09-28 10:0x — FAIL → STOP-and-ask

**Run.** S_cat s3 (reference X01, chain `c6y_scat`, final table: `STROKE_W` 0.126240, side ×2.928).
- 4000 its from W's baked copy, 08:11–10:00.
- Head: `parkour/logs/rsl_rl/crab_hex_omni_flat_walk/2026-09-28_08-11-28_seed3_X01_c6y_A15pB/model_15994.pt`.
- Suite: `parkour/logs/rsl_rl/gait_eval/omni_phases/2026-09-27_stroke_cat_W/seed3_X01_c6y_A15pB/suite__omni/seed001/2026-09-28_13-53-35`.
- Floors held.

**Pilot rule.** The episode-mean side stride_along must beat B s3's (0.0964), point Δ > 0. **It does not.**

| Side key | S_cat s3 | B s3 | B s2 | Δ vs B s3 | Δ vs B s2 |
|---|---|---|---|---|---|
| stride_along (m) | 0.0865 | 0.0964 | 0.0915 | **−0.010 ± 0.003** | **−0.005 ± 0.003** |
| achieved speed (m/s) | 0.108 | 0.118 | 0.114 | −0.010 ± 0.002 | −0.006 ± 0.002 |
| stride_per_speed (s) | 0.798 | 0.814 | 0.800 | −0.016 ± 0.017 | −0.002 ± 0.017 |
| touchdown Hz/foot | 0.560 | 0.549 | 0.568 | +0.011 ± 0.009 | −0.007 ± 0.010 |
| body_offset (m) | 0.248 | 0.234 | 0.225 | +0.014 ± 0.007 | +0.023 ± 0.007 |

**Reading.** Side strides got shorter because side walking got slower. Stride per unit speed and touchdown rate did not
move beyond noise. A side stroke income at about 8.6 % of side tracking (3 × the single weight's) bought neither speed
nor longer steps. The audit's matched-speed premise probe had predicted exactly this risk.

**Elsewhere** (vs B s3; reported, not gating):
- **fwd** stride −0.073 ± 0.015, speed −0.032, and the fwd tripod-coherence unison flag **fires** (−0.133 against a
  0.100 band, B seed Δ +0.007).
- back stride −0.058.
- diag stride −0.029.
- move_turn survival 0.735 (B s3 0.585).
- **Survival:** fwd 0.88, side 0.995, back 0.94, diag 0.70, turn 0.99.

**Decision.** Per decision 5, W s2 and Phase C are NOT launched; this is STOP-and-ask. The user had also asked to pause
for a reboot once this eval finished.

JSON: `S_cat_s3_vs_B_s3.json` (seed-banded unison) and `S_cat_s3_vs_B_s2.json` beside this file.

**User override (2026-09-28 14:40, after the reboot):** "Run W s2 and t3 as planned".
- W s2 runs next, then the pooled W gate (step 8), then Phase C (t3) regardless of the W verdict.
- The pilot's FAIL stands as a recorded s3 reading.

**W s2 iteration-~500 share report (2026-09-28 ~15:00; report only, no stop for s2 per the plan).**
- Side read 18.9 % over its 12446–12546, and fwd 13.4 %. The side figure comes from ONE iteration (12458), which logged
  a side stroke income of 6.59/s, 120 × the window median of 0.055. Without it, side is 8.5 %.
- Over 200-it windows (12046–12446) s2's side share is 9.4 % and 8.3 %, the same as s3's (8.3 % / 8.7 %).
- **Exploit check on s3's whole run:** the side stroke median held at 0.056–0.060/s in every 500-it window, with 0–3
  spikes (> 10 × median) per window and no trend. No learned exploit.
- **Flag for the gate:** at side × 2.93, the known rare heavy-tailed stroke spikes are large (up to about 6.6/s
  logged).
