# t3 gate — stroke formation, Stage 1 (2026-09-27)

**Verdict: neither arm passes. STOP RULE → Stage 2 does not run.**
- **S_f** fails its stroke-lengthening test.
- **P_f** fails its posture test, and separately fails health.

**What ran.** Stage 1 ran all six t3 screens: B_f, S_f and P_f, each at seeds 3 and 2.
- Every run trained its full 3000 iterations from scratch, with no floor or dead_walker REJECT.
- **GPU:** about 8.3 GPU-h, including the 0.2-h dose smoke. The reboot pause discarded about 10 more minutes of
  training.
- **Checked by:** an independent three-agent pass (numbers recomputed from the eval stamps, gate logic against the
  charter, adversarial claims review), before this version.

**How the gate is read.**
- **Rule:** the orchestrator's pre-registered rule in [CHARTER.md](CHARTER.md). The driver's lines (D05, D06) are
  records only.
- **Deltas:** pooled means over seeds 3 and 2 of arm − B_f at the same seed (`replay_stroke_posture.py pooled`).
- **Band:** max(pooled bootstrap MDE, |B_f s3 − B_f s2|).
- **Evidence:** all JSON is in [`gate/`](gate/). The unison readouts `*_vs_B_f_*.json` use the charter's seed band
  (`--seed-a/--seed-b` = B_f s3 / B_f s2).

**Confound: B_f s3 is the weak baseline seed on every translation row.**
- Side survival is 0.76, against 0.97–0.985 in the other five runs.
- Its fwd stride_along is 0.067, against 0.135–0.165 elsewhere.
- Its fwd/back tripod coherence is 0.35/0.40, against 0.49–0.63 elsewhere.
- It was also the only run trained before the reboot. The code_sha was the same.

Every s3 delta, and every seed band built from B_f's pair, therefore overstates arm effects.

## The six runs (side holds, eval `suite__omni`, 200 side episodes each)

| Run | body_offset (m) | lead/trail asym | stride_along (m) | stance_stroke (m) | stroke_share | side ratio | fwd stride_along | fwd stance_stroke |
|---|---|---|---|---|---|---|---|---|
| B_f s3 | 0.169 | 0.388 | 0.080 | 0.029 | 0.527 | 0.458 | 0.067 | 0.055 |
| B_f s2 | 0.143 | 0.351 | 0.081 | 0.033 | 0.530 | — | 0.135 | 0.068 |
| S_f s3 | 0.150 | 0.356 | 0.079 | 0.032 | 0.562 | 0.452 | 0.165 | 0.076 |
| S_f s2 | 0.176 | 0.405 | 0.075 | 0.032 | 0.528 | 0.457 | 0.145 | 0.066 |
| P_f s3 | 0.143 | 0.339 | 0.079 | 0.034 | 0.558 | 0.451 | 0.157 | 0.066 |
| P_f s2 | 0.214 | 0.466 | 0.081 | 0.024 | 0.432 | 0.438 | 0.136 | 0.072 |

The side ratio is the driver's suite tracking ratio; B_f s2's is not printed in the D05/D06 lines.

**What the table shows.**
- **The lateral posture forms in every run.** Side body_offset spans 0.143–0.214 m.
  - The two seeds of one arm differ by 0.026 m (B_f) to 0.071 m (P_f).
  - That spread is as large as any arm-vs-B_f delta, and comparable to P's −0.05 m bar.
- **Side stride_along does not follow the lean.** It stays at 0.075–0.081 m in all six runs.
  - The least-leaning run (B_f s2, 0.143) and the most-leaning (P_f s2, 0.214) both read 0.081.
  - W reads a 0.231 lean with a 0.083 stride (X08:c6: 0.234 lean, 0.075 stride).
  - The side tracking ratio is flat too (0.44–0.46).

## S_f (per-foot stroke income, w_S 0.119) — FAIL

**1. Dose landed: PASS.** r = delivered stroke income (`Episode_RewardCat`, last 200 its) ÷ the term's nominal
(pre-clip) replay prediction at w_S. The prediction is priced on B_f sN's eval stamp in B_f sN's clock-held economy
(`params/env.yaml` of X01 / X02), which is the dose smoke's method.

| Seed | fwd | side | back |
|---|---|---|---|
| s3 | 1.50 | 1.19 | 1.16 |
| s2 | 1.19 | 0.89 | 0.98 |

**The single fwd-bound weight gave side only about 2–3 % of side tracking income** (delivered 0.018 / 0.014 per s
against 0.63 / 0.65). S's null on side is therefore a weak test of stroke income on side.

**2. Stroke lengthened beyond its band: FAIL.**

| Key | s3 | s2 | Pooled | Band | Reading |
|---|---|---|---|---|---|
| side stance_stroke | +0.0029 | −0.0009 | +0.0010 | 0.0034 | inside |
| side stroke_share | +0.0347 | −0.0019 | +0.0164 | 0.0172 | inside (carried by s3 alone) |
| fwd stance_stroke | +0.0214 | −0.0020 | +0.0097 | 0.0126 | inside (carried by s3 alone) |

**Disclosure: hand rows.** The gate reads the category rows the charter names (side = both hands).
- The pooled tool also reports hand rows. There, side/l stroke_share alone clears its band: +0.030 against 0.021
  (s3 +0.055, s2 +0.004).
- side/r reads −0.001.
- The rise is one-handed and carried by s3, and the charter does not pre-register hand rows, so it is not a pass.

**Outcomes** (pooled; reported, not gated):

| Key | Pooled | Band | Reading |
|---|---|---|---|
| side stride_along | −0.0034 | 0.0023 | slightly SHORTER, carried by s2 alone (−0.0061) |
| side body_offset | +0.0068 | 0.0255 | TIE |
| side lead/trail asym | +0.011 | 0.037 | TIE |

**Health: PASS.**
- Moving ratios are ≥ 0.30 at both seeds (fwd/side/back: s3 0.61/0.45/0.63, s2 0.61/0.46/0.55).
- The its-800–1000 mean track_ratio is 0.548 at s3, against a dead_walker floor of 0.403.
- The floor passes at both seeds.
- Unison is clear at both seeds under the seed band.
- Stroke-income telemetry shows no learned exploit: the 250-it window mean stays at the median.

**Driver D05:** REJECT (disputed, final). Every block and guard is TIE; the s3 turn-slip WORSE (0.106 → 0.129) did not
replicate at s2.

## P_f (side body-centring bonus, w_P 0.158) — FAIL (posture, and health)

**Posture: FAIL.** Pooled Δ side body_offset is **+0.022 m**; the pass bar is ≤ −0.05 m.

| Key | s3 | s2 | Pooled | Band | Reading |
|---|---|---|---|---|---|
| side body_offset | −0.026 | **+0.070** | +0.022 | 0.026 | wrong direction |
| side lead/trail asym | −0.049 | +0.115 | +0.033 | 0.037 | TIE |
| side stride_along | −0.001 | −0.000 | −0.001 | 0.002 | TIE |
| side stance_stroke | +0.005 | −0.008 | −0.002 | 0.003 | TIE |
| side stroke_share | +0.031 | −0.098 | −0.033 | 0.015 | DROP beyond band, carried by s2 alone |

- **s3's −0.026 equals the baseline seed gap (0.0255).** It cannot be told apart from seed.
- **s2 ended with a larger lean than its own baseline** (0.214 against 0.143). That is still inside the unarmed
  recipe's historical range (X08:t3 0.210; B0 0.193 / 0.146; Z 0.237 / 0.251), and one seed cannot show that the
  bonus caused it.
- **s2's lower delivered centring income** (0.72 × nominal prediction; 0.81 delivered-vs-delivered; s3 1.05) follows
  from the larger lean.

**Health: FAIL** (independent of the posture verdict).
- **Unison.** Under the charter's seed band, the s2 flag fires on diag: tripod coherence −0.104 against a 0.100
  band (B_f's diag seed Δ 0.077 sits under the 0.10 floor). fwd (−0.123) and back (−0.107) stay inside their seed
  bands of 0.260 and 0.214. No roll collapse.
- **s2 also lost survival** against B_f s2:
  - trained rows: fwd 0.91 → 0.71, back 0.97 → 0.68, turn 0.98 → 0.90;
  - untrained rows: diag 0.83 → 0.51, move_turn 0.77 → 0.055.

  Completion fell 0.910 → 0.676, and the gate_1c WATCH flips PASS → FAIL. The t3 floor still passes. S_f s2, on the
  same seed, keeps fwd/back survival at 0.95/0.91.
- Moving ratios are ≥ 0.30 at both seeds. The its-800–1000 track_ratio at s3 is 0.560.

**Turning got worse at both seeds** (driver D06, REJECT clear final; F WORSE pooled z −3.72):
- turn facing_closed 0.467 → 0.292 at s3 (z −3.55) and 0.412 → 0.212 at s2 (z −3.89);
- turn steps/rad WORSE, 62.7 → 82.7 at s3 and 60.7 → 131.7 at s2.

**P did change the side gait in one way:** side tripod coherence rose at both seeds (+0.042, +0.081).

**Dose arithmetic.**
- The term pays exp(−(o/0.2)²), so cutting the lean from 0.17 to 0.12 m is worth about +0.034/s at w_P 0.158. That
  is about 5 % of side tracking income, near the term's steepest slope.
- The tracking cost of recentring was never measured, so whether that prize is "too small" is not known. The
  ~10 %-of-tracking dose was a design choice.
- P landed at 11–14 % of side tracking.

## Reading

Armed from iteration 0 of formation:
- the per-foot stroke income (fwd-bound, 10 % of fwd tracking, 2–3 % of side) landed but did not lengthen any stroke
  beyond seed noise;
- the centring bonus (about 11–14 % of side tracking) neither pulled the side lead toward the bar nor lengthened side
  strides, and it cost turning, plus survival at one seed.

Together with the W fine-tune screen ([`../2026-09-26_stroke_posture/CLOSE.md`](../2026-09-26_stroke_posture/CLOSE.md),
one seed per arm), neither lever lengthened side strides or moved body_offset to the bar at these doses, at formation
or after it.

**The new finding here.** Across these six formation runs, side stride did not follow the posture. body_offset varied
by 0.07 m, more than P's bar, while side stride_along and the side tracking ratio stayed flat.
- Neither campaign licenses a posture → stride mechanism.
- The original investigation's "short side steps are a posture outcome" is not supported inside the omni range.
- The lean is specific to the omni setup, not the plant: the sideways head on the same plant does not lean (0.014 m).
- The sideways head also walks about 2× faster (≈ 0.22 m/s achieved, against ≈ 0.10 m/s for omni side holds; ratio
  about 0.45 at 0.22 m/s commands), with about 3.4× longer strides (0.27 m). The short side stride may be downstream
  of the side speed deficit rather than of the posture. That is a hypothesis; nothing here tests it.

## For the user (nothing below has run)

1. **Stop the stroke/centring line here** (recommended; the stop rule's default).
   - Keep the knobs default-off.
   - Hand the dead-stride removal to reward-simplification #2.
2. **P at 2–3× at formation.** Not recommended.
   - **Cost:** about 2.7 GPU-h per dose level (5.4 for both), plus a dose smoke.
   - **Dose bound:** it breaks the ~10 %-of-tracking bound. P would pay about 22–40 % of side tracking.
   - **Harm:** at 1×, one seed already lost 20–30 pp of fwd/back survival, move_turn collapsed, and turning got worse.
   - **Power:** with seed gaps up to 0.07 m, two seeds may not resolve the −0.05 m bar.
   - **Commitment:** a pass commits Stage 2 (about 17 GPU-h per arm, B_f included).
   - **Payoff:** the six runs suggest recentring may not lengthen strides at all.
3. **Ask why omni side walking is slow and short, before any further reward lever.**
   - **Zero-GPU steps** re-read existing stamps:
     - a speed-matched readout (stride_along against achieved side speed across the omni heads and the sideways
       head);
     - 1b against the sideways head.

     1b is side-axis only, with no turn slots and no turn-first curriculum, and forks from the forward-walking 1a. It
     already leans 0.133 m, so the turn share and the curriculum are unlikely origins of the lean.
   - **What separates 1b from the sideways head:**
     - the command frame and observed channels (world-frame with facing P-control, against body-frame vx only);
     - the reward economy;
     - the achieved side speed;
     - the 1a fork.
   - **Not zero-GPU:** evaluating the sideways head on omni side commands needs an Isaac eval and a command adapter,
     because that head observes only a body-frame vx travel speed.
4. **S side-weighted or beyond R4.** This is outside plan decision 6. It would need about 4× to reach 10 % of side
   tracking. The s3-only fwd gains are weak evidence for it.
