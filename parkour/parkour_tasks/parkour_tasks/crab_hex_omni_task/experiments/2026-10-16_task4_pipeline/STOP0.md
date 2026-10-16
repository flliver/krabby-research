# STOP-0 — thresholds v0, cameras, budget, design decisions (step 1, CPU only)

Everything below was computed on CPU from recorded stamps. Nothing was trained, evaluated or launched.
- Machine record: [`thresholds_v0.json`](thresholds_v0.json), written by `../tools/pipeline/thresholds.py`.
- Its input map, with every stamp and its eval_sha: [`threshold_inputs.yaml`](threshold_inputs.yaml).
- Camera spec: [`cameras.yaml`](cameras.yaml).

To re-run:

```bash
cd /home/nickmagus/krabby/krabby-research
PL=parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task/experiments/2026-10-16_task4_pipeline
/home/nickmagus/krabby/isaac_venv/bin/python -B parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task/experiments/tools/pipeline/thresholds.py \
  --inputs $PL/threshold_inputs.yaml --out $PL/thresholds_v0.json --self-check
```

## Decisions for you (one recommendation each)

| # | Decision | Recommendation |
|---|---|---|
| 1 | **Thresholds v0**: the decisive lines, their read models and the sums (§1) | **Approve.** At the recommended values the run sum is **1.06** (P(≥ 1 chance HALT) ≈ **0.66**); at the standard values it is 2.82 (≈ 0.95). Expect about one chance-HALT prompt per run. The model choices are those PLAN states. |
| 2 | **Lowered s1–c6x floors** (§2) | **Survival ≥ 0.80, ratio ≥ 0.50, side22 not-SLOW** (FAST/FAST relaxed to "fail only if a hand reads SLOW or DEAD"). This takes the s1–c6x cost from 1.40 to 0.33. C0 still fails at every stage that has a C0 stamp, through the side ratio (0.42–0.49) and side22 (SLOW). |
| 3 | **Stand lines** (§5) | Keep everything except three lines. (a) **WATCH** the SN (D1) G2 post-settle steps / taps; they are re-gated on SH through K5 at the coded lines. (b) **Relax** G4 field_hard and field_turn to REFS(T\*) − max(0.10, 2 SE). (c) K4 ≤ 3/800 (D9). D1 goes from 0.68 to 0.09 and D2 from 0.34 to 0.24. |
| 4 | **Cameras** (§7) | **Approve the placeholders.** Front: ZED 2i, 55° down. Left, right and rear: SEN0583, 54° down. All four: 58 × 87 rays, an update every step, target `/World/ground`. Every camera meets its ground band. The forward camera in use today does not (it sees ground only from 1.40 m). |
| 5 | **Budget** ≈ 130–170 GPU-h | **Approve.** Nothing in step 1 changes it. The 4-camera cost stays unmeasured until step 8 (STOP-C at > 4 h per 1k its). |
| 6 | **D5 payload geometry** | **Approve.** Reset draw U[0, L] with 25 % zero; CoM z +0.309 m, xy ±0.10 m; deck-footprint box inertia; startup DR preserved by lazy capture. |
| 7 | **D9 K4 ≤ 3/800 per env seed** | **Approve.** SH_T16/T18 fail the coded ≤ 1. Rate per run: 0.15 (env-seed λ, conservative) or 0.09 (pooled λ = 8/6). |
| 8 | **D14 code freeze** (~6 days) | **Approve.** The freeze notice must say "commit or stash before the pin". Another session is active in PL right now: `preflight/`, `scripts/t4_*` and a CHARTER edit, none of them mine. |
| 9 | **D19** untracked `resolution.74f93b91dc8b.json` | **Drop it from the v1 inputs; no recompute is needed.** It raises only the fwd/back survival guard R (0.12 against 0.11 / 0.105 in the tracked file). No recommended decisive rate uses that R. Its only effect would be a 0-kg survival margin wider by ≤ 0.015. The alternative is that its owner commits it before the pin. |

## Things that contradict PLAN, or that I could not compute

- **side22 FAST/FAST costs more than PLAN says.** PLAN gives ".03–.06 per read" at s1. That is the cost of **one median**. A FAST/FAST read tests four medians (both hands × 0.22 / 0.25 m/s): **0.22 at s1, 0.29 at c4, 0.13 at c6, 0.06 at c6x, 0.04 at p2a**. This is why item 2 relaxes it.
- **The 0-kg drift figures differ from PLAN.** PLAN quotes move_turn −.048 and yaw_progress −.012. On the harness key (`by_category.<c>.<key>.median`) I get:
  - move_turn ratio: −0.057 (es1) and −0.062 (es2);
  - yaw_progress: −0.0099 (es1) and −0.0076 (es2).

  I do not know where PLAN's figures came from; perhaps a mean, or another row.
- **The fixed-line t (2 df) has very heavy tails.** p2a, p3a and the student keep ratio ≥ 0.70 against recorded ratios of about 0.95–1.09. Even so, each read costs about 1 %, about 0.075 per stage over 8 reads. That comes from the 3-seed model, not from data near the line.
- **Survival floors cost ≈ 0 under PLAN's binomial model,** at every candidate value. The earlier "≈ 0.75" figure is reproduced by a t-model that uses the survival guard R (0.10–0.14). That model is reported as a sensitivity only.
- **Zero-count lines** (falls == 0 and the like) cost 0 under the plain Poisson / binomial model. They cost up to 0.134 per read under the Jeffreys predictive. The sensitivity run sum is 2.14 at the recommended values; "suite stand falls == 0" is the main term.
- **c5 is read through proxies.**
  - There is no world-mode stamp: anchors are the body-mode `STG_c5` reads (a different row and adapter).
  - There is no c5 side22 row: its rate is the c4 proxy. The gate row must be added (PLAN implementation item 4).
  - There is no c5 known-bad.
- **p2a, p3a and student floors have no same-stage known-bad,** because the C0 lineage stopped at c6x. The C0 c4 / c6x stamps serve as cross-stage known-bads for the same 0.90 / 0.70 lines.
- **Payload a-priori lines (p3a stable-at-90) are pending step 8.**
  - S1 at 0 kg passes every line (§4).
  - The formula is in the JSON. The h_meas definition is a proposal for `payload_read.py`.
- **The 0-kg drift margin scales linearly with iterations, and that is untested.**
  - At N_p = 2500 the scaled margin is half of X12's 5000-iteration drift.
  - X12-versus-X08 itself would fail at that scale on side tracking, move_turn and yaw_progress.
  - It passes only at scale 1, by construction, so the margin stays `unvalidated`.
- **The student completion line is only illustrated.** No student exists, so its SE_diff is taken from the X12–X08 teacher pair.
- **Live kills** are listed as `unvalidated` and excluded from every sum, as PLAN says.
- **`PL/DESIGN.md` was not edited.** Its header already carries the step-1 annotation (4 cameras, K = 4, the three path fixes, the decisive list, the audit answers), and this step's write list excludes it.

## 1. Sums (per run)

Run sum = Σ of per-read chance-HALT rates over decisive reads; "union" = 1 − Π(1 − p). Reported, not enforced.

| Stage | Reads (rec.) | Standard: sum | Standard: union | **Recommended: sum** | Recommended: union | What dominates (recommended) |
|---|---|---|---|---|---|---|
| s1 | 7 | 0.258 | 0.249 | **0.047** | 0.047 | side22 not-SLOW 0.036 |
| c4 | 7 | 0.414 | 0.375 | **0.102** | 0.099 | side22 0.073; back ratio 0.016 |
| c5 | 9 | 0.370 | 0.346 | **0.095** | 0.093 | side22 proxy (c4) 0.073 |
| c6 | 9 | 0.261 | 0.239 | **0.071** | 0.069 | ratio 0.042 (back 0.024); side22 0.029 |
| c6x | 9 | 0.095 | 0.093 | **0.020** | 0.020 | side22 0.006 |
| p2a | 18 | 0.163 | 0.153 | **0.163** | 0.153 | ratio 0.075, FAST/FAST 0.045, field_hard 0.043 |
| p3a | 17 | 0.120 | 0.114 | **0.120** | 0.114 | p2a-proxy floors 0.120; paired 0-kg 0.004 |
| student | 18 | 0.120 | 0.115 | **0.120** | 0.115 | p2a-proxy floors; completion 0.0005 (lower bound) |
| D1 (SN) | 39 | 0.677 | 0.514 | **0.090** | 0.088 | G4 field_hard 0.051, field_turn 0.029 |
| D2 (SH) | 101 | 0.343 | 0.305 | **0.236** | 0.220 | K4 es2 0.143; K5-G4 field_hard 0.050 |
| **Run** | **234** | **2.82** | **0.95** | **1.06** | **0.66** | Jeffreys sensitivity: 3.90 / 2.14 |

- "Standard" means the floors at 0.90 / 0.70 with FAST/FAST at every stage, and the stand lines as coded, except K4 at 3 (D9). With the S1-era K4 ≤ 1, D2 alone rises by about 0.6.
- p3a's stable-at-90 lines are pending (pilot), so they are excluded from the sums.
- p3b and p3c never HALT (measured ceiling).

## 2. Lowered floors at s1–c6x: candidates

Each cell gives the stage's summed rate over its floor reads at that value. KB is the number of C0 seeds that fail.

| Stage | Survival 0.90 | 0.80 | 0.70 | 0.60 | Ratio 0.70 | 0.60 | **0.50** | 0.40 | side22 FAST/FAST | **not-SLOW** |
|---|---|---|---|---|---|---|---|---|---|---|
| s1 | ≈0 (t: .32) KB 1 | **≈0 (t: .11)** KB 0 | ≈0 KB 0 | ≈0 KB 0 | .036 KB 3 | .018 KB 3 | **.011 KB 3** | .007 KB 0 | .222 KB 3 | **.036 KB 3** |
| c4 | ≈0 (t: .44) KB 2 | **≈0 (t: .15)** KB 1 | ≈0 KB 1 | ≈0 KB 0 | .123 KB 3 | .053 KB 3 | **.029 KB 3** | .019 KB 0 | .291 KB 3 | **.073 KB 3** |
| c5 | ≈0 (t: .52) | **≈0 (t: .19)** | ≈0 | ≈0 | .080 | .038 | **.022** | .015 | no row (proxy .291) | **proxy .073** |
| c6 | ≈0 (t: .57) KB 0 | **≈0 (t: .21)** KB 0 | ≈0 | ≈0 | .133 KB 1 | .069 KB 1 | **.042 KB 1** | .028 KB 0 | .128 KB 1 | **.029 KB 1** |
| c6x | 0 (t: .50) KB 0 | **0 (t: .18)** KB 0 | 0 | 0 | .037 KB 1 | .021 KB 1 | **.014 KB 1** | .010 KB 0 | .059 KB 1 | **.006 KB 1** |

- Survival is binomial on the pooled recorded falls at n = 100 (side n = 200); "t:" is the guard-R t-model sensitivity.
- Ratio is the fixed-line t (2 df) with R from the resolution files (fwd .041, side .018, back .053, diag .026).
- side22 is the fixed-line t on each `feet_ss` median (R = 2 × the bootstrap SE). FAST is 0.155 / 0.175; SLOW is 0.130 / 0.145.

**Why ratio 0.50.** It is the lowest value at which C0 still fails: its side ratio is 0.437 / 0.443 / 0.450 at s1, 0.421 / 0.430 / 0.448 at c4, 0.419 at c6 and 0.488 at c6x. At 0.40 the line no longer catches C0.

**Why survival 0.80.** It costs ≈ 0 under the binomial model. C0 X63 (c4 s18: fwd 0.66, back 0.74) still fails it.

**Why not-SLOW.** C0 reads SLOW / SLOW on every recorded stamp. The relaxed side22 line cuts its cost 4–9×.

The fwd / back / diag ratio lines and most of the survival lines then have no failing C0 seed of their own. They are tagged `unvalidated`, while each stage keeps at least one floor line that C0 fails.

## 3. Capability lines at p2a and the student

| Line | Seed values (s16 / s17 / s18) | Line | Model / rate | Known-bad |
|---|---|---|---|---|
| p2a field_hard, pooled es1–4 (n = 400), `aggregate.schedule_completion_rate` | 0.815 / 0.725 / 0.8125 | **0.635** = min − max(R 0.045, range 0.090, G 0.068) | three-seed; r = R/σ̂ = 1.07 → **0.043** | ANCW 0.435 / 0.56 / 0.50 fail (n = 200) |
| p2a floors (es1 + es2): survival 0.90, ratio 0.70, FAST/FAST | every translate category 1.00 survival, ratio 0.95–1.09 | 0.90 / 0.70 | binomial ≈ 0; t: 0.075 / 8 reads; FAST 0.045 | C0 X63 c4 (cross-stage), C0 X64 c6x (cross-stage) |
| Student completion, paired against T\* (field_hard, field_light, ring_obst, field_turn) | — | T\* − max(0.05, 2 SE_diff, split) | paired; illustrative p_block = 0.023 → HALT (2 blocks) **0.0005**, a lower bound | `unvalidated` |

The 567d1704 stamps (G0f_S1_s18 suite and field_hard es1) are aggregate-equal to the 1d021625 REFS / REF re-reads; this was checked in `thresholds_v0.json` `equality_checks`.

## 4. p3a: stable-at-90 (a priori) and the paired 0-kg check

**Stable at 90.** S1 at 0 kg passes every line (suite and field_light, es1–2).

| Line | S1 s16 / s17 / s18 (es1) | Line |
|---|---|---|
| translate survival (fwd, side, back, diag) | 1.00 / 1.00 / 1.00 | ≥ 0.90 |
| \|ratio − 1\|, worst category (back) | 0.041 / 0.017 / 0.087 | ≤ 0.25 |
| turn / move_turn survival | 1.00 · 0.995 / 0.99 / 1.00 | ≥ 0.885 |
| turn facing_closed | 0.963 / 0.948 / 0.959 | ≥ 0.80 |
| field_light completion, es1–2 | 0.98 / 0.985 / 0.995 | ≥ 0.85 |
| median h_meas (suite; field_light) | 1.110 / 1.108 / 1.077 m; 1.130 / 1.120 / 1.091 m | ≥ 0.85 m |
| roll / pitch RMS (median), suite and field_light | ≤ 0.027 / ≤ 0.059 | ≤ 0.10 rad |

- The rate formula is Φ((line − pilot) / √(2σ̂²_p2a + SE²)). It is pending step 8.
- σ̂_p2a per line is in the JSON.
- support_feet and swing_apex are retired (PLAN).

**Paired 0-kg check.** Margin = max(R, 3 SE_diff, |drift| × its_past_p2a / 5000). The drift is X12 (P2__C0_s18, +5000 its) minus X08 (S1 s18).

| Statistic | es1 drift (worse +) | es2 drift | SE_diff | R | Margin at N_p = 2500 (binds) |
|---|---|---|---|---|---|
| \|r−1\| fwd / side / back / diag | −.008 / **+.026** / −.006 / −.011 | −.011 / **+.019** / −.023 / −.016 | .003–.005 | .041 / .018 / .053 / .026 | R |
| \|r−1\| move_turn | **+.057** | **+.062** | .0055 | .037 | R (0.0367) |
| translate survival (4 categories) | 0 | 0 | 0 | .051–.14 | R / G |
| turn facing_closed | −.005 | −.007 | .003 | .050 | R |
| turn yaw_progress | **+.0099** | **+.0076** | .0008 | .0066 | R (0.0066) |
| field_hard es1–2 / field_light / ring / field_turn completion | +.005 / +.015 / 0 / 0 | — | .038 / .011 / 0 / .014 | .057 / .020 / 0 / .020 | 3 SE |
| side22 | FAST / FAST both | — | — | — | class, no margin |

- **Rate** is Σ Φ(−margin / SE) ≈ **0.004** at any N_p. That is illustrative: the SE_diff comes from X12–X08.
- **Self-check** (X12 vs X08 at scale 1): passes, by construction.
- **Bold** marks drifts larger than R. At N_p = 2500 these exceed the half-scaled margin, so the same drift over a payload stage would HALT. The margin is `unvalidated`.

## 5. Stand lines: keep, relax or WATCH

Seeds: SN_Amask s16/17/18 (G1–G5); SH_T16/17/18 (K1–K6; K5 = G1–G5 re-read by `sn_calib` on the SH_T1xz zero-command stamps). Models: continuous = fixed t with R = 0 (no committed resolution entry; flagged), shares = binomial, counts = Poisson (per env seed).

| Line (per env seed) | Recorded seeds (es1 · es2) | Coded line | Rate per run, coded | Options → rate per run | Known-bad | **Recommendation** |
|---|---|---|---|---|---|---|
| D1 G2 post-settle steps/s | .466/.407/.444 · .432/.380/.458 | ≤ 0.5 | 0.276 | ≤ 0.6 → 0.068; WATCH → 0 | none by value (REFS, Z1, Z2d fail closed) | **WATCH** (SH's K5-G2 reads 0.10–0.11 against 0.5) |
| D1 G2 post-settle taps/s | .882/.743/.764 · .816/.745/.855 | ≤ 1.0 | 0.143 | ≤ 1.2 → 0.043 (A-full 1.14 then passes); WATCH → 0 | A-full s18 1.139 fails 1.0 | **WATCH** (K5-G2 taps 0 / 0 / 0) |
| G4 field_hard es1–4 (D1; K5) | .81/.805/.7825; .825/.825/.7875 | REFS − max(.05, 2SE) ≈ .757 | 0.149; 0.127 | REFS − max(.10, 2SE) ≈ .7125 → 0.051; 0.050 | none (`unvalidated`) | **Relax** |
| G4 field_turn es1 (D1; K5) | .97/1.0/.99; .98/1.0/1.0 | REFS − max(.05, band) = .94 | 0.099; 0.043 | REFS − .10 = .89 → 0.029; 0.013 | none (`unvalidated`) | **Relax** |
| K4 restore misses | 1/0/1 · 2/1/3 | ≤ 1 (S1-era) | 0.74 (per env seed λ) | ≤ 3 (D9) → 0.148; ≤ 4 → 0.053 | no-restore 788 / 784 | **≤ 3 (D9)** |
| K3 falls / 100 | .5/.17/.5 · .33/.33/.33 | ≤ 1 | 0.015 | — | A-mask, S1 fail | Keep |
| Every other G / K clause (132 reads) | far from their lines | as coded | 0.02 in total | — | listed per line in JSON; G3, G5, K2-falls, K4-exit/onset and K5 are `unvalidated` | Keep |

Two notes on the D1 recommendations:
- The SN head is only SH's warm start, and SH is gated on K5 (the G lines at the coded values). WATCH therefore only removes an early SN HALT that would save about 2.35 GPU-h of SH training when SN is bad.
- The relaxed G4 lines are written at the S1-era REFS values (0.8125 / 0.99). At gate time they re-anchor on REFS(T\*), as PLAN says.

## 6. Self-checks (recommended set)

- **Every recorded seed passes every decisive line.** 0 failures. The exception is c5 side22: there is no stamp, so the check cannot be made.
- **Known-bad or `unvalidated`.** Every line without a failing known-bad is tagged `unvalidated` in the JSON. Each s1–c6x stage keeps a C0-failing line, through the side ratio and side22.
- **Rates and sums** are reported (§1). Live kills and the drift margin are `unvalidated` and excluded.
- **Floors and the stand reads.** The floors do **not** apply to the D1 / D2 reads (`floors_apply_to_D1_D2: false`). T\* is frozen there, and G4 / K5 carry the walls (PLAN item 15).

## 7. Cameras (spec only; implementation in step 4)

Sources:
- ZED 2i: [Stereolabs product page](https://www.stereolabs.com/products/zed-2).
- SEN0583: [DFRobot wiki](https://wiki.dfrobot.com/sen0583/). The M7 doc lacks the FOV.

| Camera | FOV H × V | Range (sim clip) | Mount (body, m), **placeholder** | Pitch down | Ray grid / update | Ground band along the look axis | Target (C4 "4b") |
|---|---|---|---|---|---|---|---|
| Front ZED 2i (2.1 mm) | 110° × 70° | 0.3–20 m (2 m) | (0.37, 0, 0.10) | 55° | 58 × 87, 0.02 s | x 0.37 → 1.98 m | x 0.5–1.2 ✓ |
| Left SEN0583 | 55° × 72° | 0.2–2 m (2 m) | (0, +0.625, 0.10), yaw +90° | 54° | 58 × 87, 0.02 s | y 0.62 → 2.24 m | y 0.75–1.45 ✓ |
| Right SEN0583 | 55° × 72° | 0.2–2 m | (0, −0.625, 0.10), yaw −90° | 54° | 58 × 87, 0.02 s | −y 0.62 → 2.24 m | ✓ |
| Rear SEN0583 | 55° × 72° | 0.2–2 m | (−0.37, 0, 0.10), yaw 180° | 54° | 58 × 87, 0.02 s | −x 0.37 → 1.98 m | −x 0.5–1.2 ✓ |
| *Forward student today* | 87° × 58° | 2 m | (0.37, 0, 0.10) | 20° | 106 × 60 → crop → 58 × 87 | x 1.40 → 1.98 m | ✗ |

- **Camera height.** H = S1's stance h_meas (1.08–1.11 m) + 0.10 ≈ 1.19 m. The bands move by < 2 cm over that range. Each pitch puts the bottom image row at nadir.
- **SEN0583 orientation (flag).** The wiki lists 55°(H) × 72°(V) for a 320 × 240 image, which only fits a portrait mount or swapped labels. If V is 55°, the side band starts at 0.80 m instead and still meets its target. Resolve at M15.
- **Ray-cast targets.** `/World/ground` only. The field blocks are stamped into the generated terrain's height field (`omni_terrains.field_layout`), so they are in that mesh. This is from code reading; the step-8 check must also hit a block.
  - Robot links are not in the set, so there is no self-occlusion; on hardware the side cameras sit over the middle-leg hips.
  - The SH tables are not in the set either.
- **Ray cost per env per second, relative to the forward camera** (6,360 rays at 50 Hz):

  | Configuration | Cost |
  |---|---|
  | K = 4 at 58 × 87, every step | **3.17×** |
  | K = 4 at 60 × 95 with the forward crop | 3.58× |
  | K = 4 updating every 5th step | 0.63× |

  The per-env Python image loop (×4) likely costs more than the rays. That is DESIGN B3's batched term; step 8 measures it.
- **Proposal for step 4.** Cast each grid directly at 58 × 87 over the datasheet FOV, with no crop or resize. The forward crop would narrow the front HFOV (87° → 82.5° on the D435 pattern). Keep the 0.02 s update; the 0.1 s update is a fallback only if STOP-C timing fires, with an alignment check.

## 8. Inputs used (eval_sha)

Every stamp, with its eval_sha, is listed in `threshold_inputs.yaml` and under `provenance` in `thresholds_v0.json`.

| Stage | Stamps | eval_sha |
|---|---|---|
| s1 (X40/X42/X43), c4 (X55/X56/X57), c6 (X58/X59/X60), C0 arms | suite es1, side22_25lr es1 | none recorded (PPO era); inferred 1d021625 |
| c5 | `STG_c5_s16/17/18` suite__omni_body es1 | 567d1704 |
| c6x | `ANC_s16/17/18` suite and side22 | 1d021625; aggregate-equal to the PPO-era X58/59/60 stamps |
| p2a | REFS_S1 suite es1–2 (s18 es1: G0f), REF_S1 field rows (s18 field_hard es1: G0f), P2__S1 side22 | 1d021625; G0f 567d1704 |
| Drift pair | C0R_s18 / P2__C0_s18 (X12) against REFS / REF / P2__S1_s18 (X08) | 1d021625 |
| Stand | head_SN_Amask_s1x.json, head_SH_T1x.json (tracked campaign reads); K5 via `sn_calib --head SH_T1xz` (CPU re-read, values embedded) | 1d021625 |

## Independent check (fable·high, 2026-10-16)

No blocking errors. Every decisive line reads the key and row `run_phases.cat_summary` reads, the three seed
values match the stamps, and the models match PLAN.
- **Re-derived independently:** p2a field_hard; side22 at every stage; the C0 known-bads; K4; the drift; the stand
  sums; the camera ground bands and ray costs.
- **Points for STOP-0** (the full list is in the CHARTER):
  - **0-kg "tracking" statistic.** PLAN says a ratio *drop* (one-sided); thresholds.py computes |ratio − 1|.
    Under |r − 1|, an overshoot also HALTs: X12's side ratio went 1.051 → 1.077, so |r − 1| rose 0.026 > R 0.018.
  - **R level.** The 0-kg margins use the 95 % block R (fwd .041 / back .053). The one-sided 99 % guard R
    (.051 / .062) is `resolution.py`'s convention for a guard-type read and gives slightly wider margins.
  - **p2a field_hard** must run at es1–4 (its line and ANCW are pooled over n = 400), on top of the es1–2 0-kg
    row list.
  - **c6x.** C0's side ratio (0.488) sits within R of the 0.50 line, so c6x's robust known-bad is side22 (SLOW).
  - **FAST/FAST** rates (0.22 / 0.29) are upper bounds, because one policy's two speeds are correlated.
  - PLAN's drift example quoted means; it is now corrected to the medians the harness reads (.057 / .010).

## Step 2 — Phase A at HEAD (STOP-A read)

From `preflight/stepA_read.md`; the rules were pinned in the CHARTER before the runs.
- **P-config: PASS** at all six stages. `env.yaml` is identical key for key to the lineage run dirs; `agent.yaml`
  differs only in allowed keys.
- **P-early: FAIL on the letter.** 142 of 148 shared tags are bitwise at steps 0–19.
  - Every reward, metric and episode tag is bitwise.
  - `Loss/priv_reg` (1.4e-6) and the timing tags are tolerated.
  - `Loss/hist_latent` is off by **2 ULP** (5.96e-8) from step 0. The rule allows ≤ 1 ULP.
  - Every downstream tag stays bitwise through step 19, so the training trajectory did not move.
- **Eval:** the 0-kg suite row equals REFS_S1_s18 es1 bitwise. The current code reproduces S1's recorded eval.
- **A3 payload probe: PASS.** S1 with a fixed body-centred mass (`KRABBY_DR_MASS=m:m`; run_meta confirms the
  added mass exactly):

| mass | zero-action settle: terminated / 32, max tilt, sag | suite survival |
|---|---|---|
| 0 kg | 0, 1.76°, 0.9 mm | 1.000 |
| 45 kg | 0, 1.81°, 1.0 mm | 1.000 |
| 90 kg | 0, 1.83°, 1.0 mm | 0.999 |
| 135 kg | 0, 1.87°, 1.1 mm | 0.988 |
| 180 kg | 0, 1.90°, 1.2 mm | 0.922 |

This mass sits at the body centre. The deck-mounted CoM (+0.309 m, D5) is the harder case; step 8's deck-CoM
settle reads it.
