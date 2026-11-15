# Rough-terrain teacher c1 and its student and stand pairing — REPORT

**Status: BAKED — your "Yes, bake" (2026-11-14 23:25), done in local commit B on 2026-11-15. Merge prep is done on the branch (2026-11-15, "Get it merge ready"): origin/main b99b9232 is merged in (87cbbdec) and Task 4 is relocked at code 89c1916b / eval cd5ae167. Nothing is pushed, and the branch is not yet merged into main (your PR).** c1 + C1S + SH/SN_RT are the omni policy of record (`policy/manifest.yaml`). TDR + c31000 + SH/SN_RSD_TDR are superseded with their files kept, and TDR stays a live stage as c1's resume point (CHARTER "2026-11-15 01:50 — BAKE DONE"). The P2 read routed to USER_DECIDES (`results/p2/read.json` `decision`), and you decided yes. The stand re-pair is done (SN and SH re-anchored, nothing remaining), and the bake videos were recorded on 2026-11-13 at 23:19. Read this beside `BAKE_PLAN.md`.

- **Campaign:** `crab_hex_omni_task/experiments/2026-11-10_rough_terrain/`, branch `feat/rough-terrain`, cut from `origin/main` `3f944c4b` (the PR #13 merge). Every gate row ran at code_sha `89c1916b` / eval_sha `9fe964d7`. The one cross-pin comparison is the P2 step-3 bridge: c31000's RSD stamps are at code_sha `de3772e2` [12 15:08]. The decision log is `CHARTER.md`; the plans are `PLAN.md`, `short_arm/ARM.md` and `p2/P2.md`.
- **Citations:** [10 hh:mm] … [14 hh:mm] are CHARTER entries dated 2026-11-10 … 2026-11-14. A file name means `results/<file>` unless a path is given. † marks a count made for this REPORT from `p2/read.json` (`falls`, `variant_split`); it is in no gate.

**Terms.**
- **Policies.** TDR was the teacher of record until the bake (`model_28991`, `03c3e472`); c1 is now. RT_s19 is this campaign's fine-tune of TDR. **c1** is RT_s19 `model_29500` (`6d1f29be`), 509 iterations after TDR. **C1S** is the depth student distilled from c1 (RT_ST_s19 `model_34499`, `6b48ef33`). **c31000** was the student of record until the bake (`155b5981`); C1S is now.
- **Stand nets.** SN is the stand net and SH the stand-height net trained after it. Each is re-paired to the walker.
- **Seeds and statistics.** es*N* is eval env seed *N*. es1–2 is the selection seed pair, es3 was used for band choice, and es5–8 and es9–12 are fresh seeds. SE is the paired standard error. Success means the schedule completed AND the root crossed the kind's radius (3.0 m for slopes, 3.2 m for ledges, pits and rubble). b1/b2/b3 are a kind's three difficulty bands.
- **Teacher gates (PLAN).** **G** is gain per kind: at least k = 2 gated kinds gain new − TDR ≥ max(0.10, 3·SE) with CI lower bound ≥ 0.05, and no gated kind falls below TDR − max(0.05, 2·SE). **R** is retention: completion rows and solved kinds at new ≥ TDR − max(0.05, 2·SE), plus 22 v1.2 tracking and turning lines, each at its own margin R (the line's resolution).
- **Student gates (P2).** **T** is G's rule on slopes b2 and ledge_down b2: KEEPS if both gain, PARTIAL if one does, LOST if neither. **T-reg** is G's no-regress clause on ledge_up b2, pits b2 and rubble b3. **N** is retention completion. **Tr** is 18 tracking lines. **E** is an envelope flag on the two overruled lines (back track_err, turn yaw_progress).
- **Class flip:** one eval launch lands in the other of its two bitwise-repeatable process-state outcome classes. There is no code change, and the actions differ at step 0.

## 1. Bottom line

1. **What was tried.** Blind fine-tunes of TDR on a new 20-column mix: slopes, hills, steps up and down, potholes, gravel and rubble, with no walls. Flat and field retention tiles made up 30 % of the mix. Rod-speed DR stayed on [10 steps 2 and 4].
2. **The full run failed.** RT_s19 at +5999 iterations (`d58ceb34`) was NOT ACCEPTED: G FAIL (only slopes gained; ledge_down regressed) and R FAIL (4 tracking lines) (`step9/gates.json`).
3. **An early checkpoint passed G on fresh seeds.** c1 (+509) passed G on es5–6: slopes b2 +0.130 and ledge_down b2 +0.175. It failed R on 3 flat-tracking lines, so the read was HALT_R (`short_arm/gates_fresh.json`). You took c1 forward by a disclosed overrule of R tracking ("A", [11 17:25]).
4. **The student holds ledge_down and most of slopes.** C1S against c31000 on es9–12 gives ledge_down +0.171 (GAIN) and slopes +0.094, which misses the 0.10 floor by 0.006. So T is PARTIAL. N passes (`p2/read.json`).
5. **What it gains.** Slope descent improves: down-variant success is 0.8875 against 0.625, and roll tips are 8 against 72 / 65 per 400†. Drops of 0.14 m and more improve: success 0.432 against 0.231 (n 303). Completion on the field rows is level (`p2/read.json`).
6. **What it costs.**
   - ledge_up b2 regresses by 0.005 past its margin (−0.055 pooled; T-reg FAIL).
   - One diag tracking line fails by 0.00076, labelled a predicted cost of c1.
   - Back holds track worse: back tracking error (|speed ratio − 1|) 0.061 / 0.066 against c31000's 0.016 / 0.013. It stays inside R (0.0617).
   - In-place turning is slower: yaw_progress 0.165 / 0.164 against 0.172 / 0.172, past R 0.0066 on both seeds. Both lines are E-only, and neither reaches its E flag (`p2/read.json` `Tr.E`).
7. **The stand.** SN (`5bedc50e`) is HALT on 4 body-height lines only. All 4 re-anchor on the R and HOLD controls under your 2026-11-09 11:53 rule, and nothing remains (`p2/stand/gates/sn_reanchor.json`). SH (`6f8b558e`) is HALT on 7 of 101 decisive lines, and all 7 re-anchor (`gates/sh_reanchor.json`, remaining []). The 4 body-height lines re-anchor on R and HOLD. The 3 K6 share lines miss their S1-era lines by 0.0007 to 0.017, and the lazy R_h beside c1 fails the same lines, so they re-anchor on it. No K4 line fails: restore misses are 1 and 2 against the gate line 3. SH's 7 failed lines are a strict subset of the baked RSD stand's 12, and SH is closer on each of them; RSD also needed the K4 chance rule (5 and 6 misses). Some passing lines are worse, though. The walk-stop median settle is 0.6 to 1.1 s slower (SN 3.80 / 3.89 s, SH 3.74 / 3.64 s, against lines 3.92 / 3.96 and RSD's 2.70 / 2.80 and 2.86 / 3.00), so its margins are thin: 0.07 to 0.32 s (§5).
8. **What is not known.**
   - The recipe is not replicated: seed 20 never gained on both kinds.
   - The student's own share of the ledge_up loss is not measured.
   - Gravel and rubble are rigid, so loose ground is untested.
   - ledge_up, pits and rubble did not gain in any read, and ledge_up and pits at or above 0.14 m did not move. The b3 ledge and pit bands (above the blind envelope) are report-only.
9. **Nothing is pushed.** With the bake (local commit B), the branch changes 183 files against `3f944c4b` (167 before the bake). They are 135 in the campaign dir, 19 code and 14 test files, 8 policy-of-record files, 6 living docs and the lit review (git diff, §6). The code is all default-off; the bake added only the policy-test rewrite and a `bundle_policy.py` docstring.

## 2. Verdict table

| Step | Verdict | Deciding numbers | Source |
|---|---|---|---|
| Step 7 gate freeze (TDR es3) | proceed, 5 gated | gated: slopes b2 0.70, ledge_up b2 0.41, ledge_down b2 0.37, pits b2 0.55, rubble b3 0.79; solved: hills, gravel | `thresholds_rough_v1.json` (ce45e01b), [10 23:39] |
| Step 9: RT_s19 34990 vs TDR, es1–2 | **NOT ACCEPTED**: G FAIL, R FAIL | G 1 of 5 (slopes +0.150; ledge_down −0.085 REGRESS); R tracking 4 / 22 fail (back 0.092 / 0.089 vs 0.0617; yaw 0.0241 / 0.0238 vs 0.0066) | `step9/gates.json` |
| Checkpoint sweep | report only | 29500: slopes +0.190, ledge_down +0.185 (es1–2; c1 was later picked on these episodes, so they are inflated) | `sweep/sweep.json` |
| Short-arm selection, es1–2 | pick c1 (tier 3, "R tracking predicted marginal") | Q = {c1}, E empty, z −0.398 | `short_arm/selection.json` |
| Short-arm confirm: c1 vs TDR, es5–8 | **HALT_R**: G PASS, R FAIL (tracking only) | slopes +0.130, ledge_down +0.175; back 0.0702 / 0.0641 vs 0.0617; yaw es6 0.0070 vs 0.0066 | `short_arm/g1.json`, `short_arm/gates_fresh.json` |
| Your decision | **"A"**: c1 to P2 by a disclosed R overrule, staged | final accept-or-overrule at this proposal | [11 17:25] |
| P2 step 3 (smoke, d', bridge) | PASS / NONE / PASS (the ×0.7 L090 bridge row needed one relaunch, a class flip) | d' c1 − c31000: slopes +0.150, ledge_down +0.230 (es1–2, picked-on, inflated; the 14:41 projection was slopes +0.09 to +0.135) | `p2/step3.json`, [12 14:41], [12 15:08] |
| P2 selection | **PICK 6b48ef33** (model_34499) | first full block PASS; recompute agrees (233 / 309 numbers, max diff 0) | `p2/sel.json`, [13 09:36] |
| ONNX | ok | action err 1.16e-6, latent 2.24e-7; io json = c31000's | `p2/onnx_PICK.json` |
| P2 read: C1S vs c31000, es9+ | **USER_DECIDES** | T PARTIAL; T-reg FAIL (ledge_up −0.055); N PASS; Tr 1 / 18 fail; E none | `p2/read.json` |
| Stand SN | HALT on 4 G5 lines → re-anchored on R / HOLD | 31 / 35 decisive pass, 8 / 8 WATCH; remaining [] | `p2/stand/gates/sn.json`, `sn_reanchor.json` |
| Stand SH | HALT on 7 lines (4 K5.G5, 3 K6) → re-anchored | 94 / 101 decisive pass; K4 PASS (misses 1 / 2 vs 3); remaining [] | `p2/stand/gates/sh.json`, `sh_reanchor.json` |
| Videos | Recorded, rc 0 (report only) | Two walker + SH mixes (seed 1514, field_medium and field_max, 180 s each): falls [], 4 / 4 stands arrived and homed, 4 / 4 restores completed. Student C1S on p3b.env (3000 steps): no reset before step 3000. These three clips are not on the rough-terrain geometry (the mixes are on the wall field, the student on the flat ring); the terrain tours are the next row. In the roughest mix one stand (0.871 m) ended 12 mm off with 3 feet loaded; the other seven are 0.6–4.6 mm | main checkout `parkour/outputs/omni_plan/rough_terrain_p2/videos/`: `sh_mix_RTC1_6f8b558e_s1514_{rough,roughest}_sequence{,_phone}.mp4` (+ `_sanity.json`), `student_6b48ef33_p3b{,_phone}.mp4` |
| Terrain videos (2026-11-14, your request) | Recorded, 14 / 14 rc 0 (report only) | c1 and C1S on each of the seven kinds at the read's bands (slopes / ledge_up / ledge_down / pits b2, hills / gravel / rubble b3), the kind row's own eval environment, holds fwd, lat_l, diag, back (33.5 s per clip): 0 falls in 14 clips; every clip reaches 4.1–5.1 m from its spawn, past the crossing radius. One episode on one tile per clip: a sample, not a rate. Slopes and hills read faintly on the white ground; the HUD's body-height line carries them [14 10:40] | `p2/terrain_videos.json`; main checkout `parkour/outputs/omni_plan/rough_terrain_p2/videos/terrain/`: `tour_{c1_6d1f29be,C1S_6b48ef33}{,_phone}.mp4` + the 14 clips and their `_sanity.json` |

## 3. What ran

### 3.1 Main run RT_s19 (2026-11-10 to 11-11)
- **Setup.** Steps 0–4 ported a trimmed set of site-terrain files and added the slope and hills tiles, the friction contract (gravel μ U(0.2, 0.5)) and the read-out. Step 5's smokes passed [10 20:58], and the neutrality row was bitwise [10 19:40]. The step-6 TDR baseline was 40 rows [10 23:39].
- **Training.** A resume of TDR for 6000 iterations (28991 → 34990) on `RT_train.env`, seed 19, 23:43 → 03:17. It took 12 859 s, or 3.6 h, against the ≈ 2.5 h planned [11 03:17]. `runs/RT_s19.json` gives wall_s 12858.3.
- **Gate block.** 47 rows, 03:18 → 06:23, fail 0. A fable re-derivation reproduced every value to 1e-12 [11 06:50].

### 3.2 Checkpoint sweep (report only)
15 rows, 06:52 → ≈ 07:55: model_29500, 31000 and 33000 on slopes b2 and ledge_down b2 at es1–2, plus suite es1. Two readers and a reconciler agreed on every number [11 08:15].

### 3.3 Short arm (Amendment A)
- **Pool.** Four candidates, pre-registered: c1 / c2 = RT_s19 +509 / +1009, and c3 / c4 = a new seed-20 run RT_s20s at the same two lengths. RT_s20s took 2327 s, 12:08 → 12:47.
- **Rows.** TDR's fresh rows (RTF, 33 rows) ran first and stayed unread until the pick was committed. Then 19 selection rows on es1–2, confirm-1 (12 rows) and confirm-2 (21 rows) on es5–8. All rows were fail 0 [11 14:50 – 16:55].
- **Checks.** fable re-derived 382 values to 1e-12 [11 17:10].

### 3.4 P2 distill, selection and ONNX
- **Distill.** RT_ST_s19 from c1: 5000 iterations (29500 → 34499), 192 envs, seed 19, on the RT mix. It ran 2026-11-12 15:07 → 11-13 ≈ 05:43, ≈ 14.6 h, at the top of the 10.2–14.7 h estimate [13 09:23].
  - The D1–D5 checks passed at +10 iterations [12 15:10].
  - The curriculum read at +1000 fired no trigger: slopes level 6.93, ledge_down 4.70, loss ratio 0.659 [12 ≈ 17:50].
- **Selection** (05:43 → 09:22). 8 c1 teacher rows, 25 prescreen rows (0 floor misses), and one full block of 19 rows on 6b48ef33, which passed.
  - The rank leans on one episode in 200 (field_hard 0.980 against 0.975 for it31500) [13 09:36].
  - Disclosed rules: the 32500 and 33500 grid targets tie and the selector takes the earlier save; the student live
    rules were not applied after +10 (P2.md step 4).
  - WATCH, against c1 on es1–2: slopes 0.920 vs 0.935 and ledge_down 0.555 vs 0.590 (`p2/sel.json` `watch_terrain`).

### 3.5 P2 read (es9–12)
09:36 → 16:00, fail 0, 88 stamps (84 rows plus 4 ledge_up relaunch rows). sel.json was committed before the first es9 stamp (`repick_check` ok) [13 16:00].

### 3.6 Stand re-pair
- **Launch.** 16:02, after the cache move [13 16:00].
- **SN.** SN_RT_6d1f29be_sn_s19: 4000 iterations in 5973 s, 6,097,610 B (`p2/stand/sn_head.json`). Read 18:56 (`gates/sn.json`). The SN bridge passes on suite, field_hard and walkstop (`sn_bridge.json`).
- **SH.** SH_RT_6d1f29be_sh_s19: DAgger, 5000 iterations in 8651 s, deadband 0.25, trim branch, warm start SN 5bedc50e; 6,102,474 B, sha 6f8b558e… (`p2/stand/sh_head.json`). Table test, 4 height rows, 9 K5 rows and the 4 lazy R_h rows beside c1 all rc 0; read 22:55 (`gates/sh.json`). The chain ended at 22:58 (6.9 h, budget ≈ 7.0).

## 4. The P2 read in detail

| Gate | Row (n per policy) | C1S | c31000 | Diff (SE) | Line | Read |
|---|---|---|---|---|---|---|
| T | slopes b2, es9–12 ×2 launches (400) | 0.8875 | 0.79375 | +0.094 (0.021), CI lb 0.052 | gain ≥ 0.10 | no gain (short 0.006) |
| T | ledge_down b2, es9–12 ×2 (400) | 0.5625 | 0.39125 | +0.171 (0.023), CI lb 0.125 | gain ≥ 0.10 | **GAIN** |
| T-reg | ledge_up b2, es9–10 + 1 relaunch (200) | 0.335 | 0.39 | −0.055 (0.024) | regress < −0.05 | **REGRESS** by 0.005 |
| T-reg | pits b2, es9–10 (200) | 0.53 | 0.46 | +0.07 (0.045) | −0.090 | pass |
| T-reg | rubble b3, es9–10 (200) | 0.84 | 0.805 | +0.035 (0.037) | −0.075 | pass |
| N | field_hard, es9–12 (400) | 0.960 | 0.9425 | +0.0175 | line 0.8925 | pass |
| N | field_light L000, es9–10 (200) | 0.99 | 1.00 | −0.010 | line 0.95 | pass |
| N | ×0.7 L090 field_light, ×2 (200) | 0.92 | 0.9325 | −0.0125 | line 0.8825 | pass |
| Tr | track_err diag, es9 | 0.0304 | 0.0006 | worse 0.02982 | R 0.02907 | **FAIL** (over 0.00076) |
| Tr | the other 17 lines | — | — | — | own R | pass (diag es10: 0.02762) |
| E | back track_err, es9 / es10 | 0.061 / 0.066 | 0.016 / 0.013 | worse 0.044 / 0.053 | flag 0.132 | no flag |
| E | yaw_progress turn, es9 / es10 | 0.1654 / 0.1641 | 0.1722 / 0.1722 | worse 0.0069 / 0.0081 (past R 0.0066 on both) | flag 0.0136 | no flag |

**Where each verdict sits.**
- **Slopes.** C1S read 0.8875 on both launches. c31000's two launches read 0.7875 and 0.80, a class flip.
  - On c31000's first launch alone the diff is 0.100, and the gain clause passes: T would read KEEPS. On its second launch the diff is 0.0875.
  - The pre-registered pooled rule gives 0.094 (47 fractional keys) [13 16:00].
  - Per seed, C1S reads 0.91 / 0.90 / 0.93 / 0.81 against c31000's 0.85 / 0.79–0.82 / 0.78–0.80 / 0.73 (`T.kinds.slopes.launches`).
- **ledge_up.** The first launch reads 0.33 vs 0.41 (−0.08); the relaunch reads 0.34 vs 0.37 (−0.03); pooled −0.055. The near-miss read made it due (shortfall 0.017 inside the 0.07 window), and the rule allows one relaunch only (`near_miss.lines.ledge_up`).
- **Diag line.** It is diag-only, and worse_by 0.0298 is within the 0.035 bound of amendment 1, so it is labelled "predicted cost of c1".
  - The cause is c1's diag hold: ratio 0.973–0.976 against c31000's ≈ 1.0.
  - Against c1 itself, selection's tracking watch shows C1S worse by only 0.0054 / 0.0047 (student ratio 0.9705 / 0.9691 against c1's 0.9759 / 0.9739) (`Tr.diag_label`).
- **Odds and outcome.** Before any number, P2 put KEEPS ≈ 0.4, PARTIAL ≈ 0.4, LOST ≈ 0.2 and N passing ≈ 0.75 (P2.md Odds). The result was PARTIAL with N passing.

**Q3 split** (es9–10, success, pooled; identity error 0; `report.q3_split`). It splits C1S − c31000 into the distill loss (C1S − c1), the teacher gain (c1 − TDR) and the old student's offset (TDR − c31000).

| Kind | C1S − c31000 | C1S − c1 | c1 − TDR | TDR − c31000 |
|---|---|---|---|---|
| slopes | +0.0775 | **−0.055** (CI −0.098 … −0.012) | +0.160 | −0.0275 |
| ledge_down | +0.1925 | 0.000 | +0.170 | +0.0225 |

The student keeps all of c1's ledge_down gain and gives back about a third of its slopes gain. These are es9–10 only, so they differ from T's es9–12 numbers.

**Report lines** (never gated).
- **Slopes up / down** (es9–12, 200 per variant):
  - down 0.8875 vs 0.625 (+0.2625);
  - up 0.8875 vs 0.9625 (−0.075);
  - the equal C1S values are a coincidence that both re-derivations confirmed;
  - falls are 45 of 400 per launch for C1S, 8 of them roll tips, against c31000's 85 / 80, with 72 / 65 roll tips†.
- **0.14 m split:**

  | Kind | Below 0.14 m: C1S vs c31000 (n) | At or above 0.14 m: C1S vs c31000 (n) |
  |---|---|---|
  | ledge_down | 0.969 vs 0.892 (97) | 0.432 vs 0.231 (303) |
  | ledge_up | 0.607 vs 0.699 (98) | 0.074 vs 0.093 (102) |

  The ledge_up loss sits mostly on the lower steps.
- **ledge_down falls** per 400: 175 for C1S against 242 / 245 for c31000. Forward-pitch tips are 131 against 206 / 207, and roll tips 44 against 36 / 38†.
- **Ceilinged rows.** hills b3 and gravel b3 read 1.00 vs 1.00, so they carry no information.
- **Pairs.** 24 same-row launch pairs: 12 bitwise-equal and 12 class flips (step-0 actions only). None goes beyond a class signature. In 5 class-flip pairs some episodes diverge later, but the registered rule judges the earliest step (`pairs`, [13 16:00]).
- **Re-derivations** (CPU, raw stamps, own readers). fable compared 328 numbers (plus 38 booleans and 24 pair verdicts) and opus 271 (plus 24), both with max diff 0. Both confirm 88 stamps at the pin, clean, rc 0, with the right sha, role and seed [13 16:00].

## 5. Disclosures and caveats

- **Overrules.**
  - c1 failed R tracking on fresh seeds; it went forward by your "A". The frozen verdict stays HALT_R. The costs disclosed then: back 1.08× vs TDR's 1.01–1.02× (0.238 vs 0.222 m/s at 0.22), yaw −3.5 %, and the recipe not seed-replicated [11 17:25].
  - A bake now would also set aside T PARTIAL, the T-reg fail and the Tr diag fail.
- **One seed, one checkpoint.**
  - c1 was picked from 4 candidates on es1–2.
  - On the screen, RT_s20s gained one kind at each length, never both, and its turning was far worse (z −23.9 / −33.8).
  - Along RT_s19, ledge_down swings by ≈ 4 paired SE per 500 iterations [11 17:10]. No seed-2 replicate ran.
- **Student share of the ledge_up loss is not measured.** The read has no c1 or TDR ledge_up rows at es9–10. At es5–6, c1 was −0.030 vs TDR, inside the regress margin 0.070 [13 16:00], [11 17:10].
- **Rigid ground.** Gravel is a fixed rough height field with μ_eff U(0.2, 0.5) per env, and rubble is fixed cells. Nothing sinks or rolls. The gravel result means slippery uneven ground, not loose ground [12, P5].
- **Back overshoot** has appeared in five fine-tunes, and the reward barely prices it (net ≈ −1.4 per m/s).
  - At +509 the likely source is stride length and action magnitude: corr(|μ|, back ratio) 0.94, n = 5 (CPU
    mechanism read, report only; supported, not proven). A second n = 5 correlation, 0.93, between the slopes gain
    and the back ratio hints that the slopes gain and the overshoot may be coupled.
  - It does not appear to come from the forward lean. The lean that sank step 9's ledge_down grew late along RT_s19 (fwd pitch +0.061 at 29500, +0.110 at 34990, TDR +0.063) [11 08:15], [11 17:10].
  - On C1S the back line sits inside R (0.044 / 0.053 against 0.0617).
- **Yaw at its R floor.**
  - c1 reads 0.1645–0.1659 on all four seeds, against TDR's ≈ 0.171.
  - C1S is worse than c31000 by 0.0069 / 0.0081, past R 0.0066 on both seeds. It is E-only and under its flag (`Tr.E`).
  - The likely cause is the mix (CPU mechanism read; supported, not proven): flat and field fell to 0.15 each, from
    TDR's 0.5 / 0.5, and the yaw drop appears within about 400 iterations in both seeds [11 17:10].
- **Class-flip noise.** A T decision depends on which class each launch drew: slopes reads KEEPS on one c31000 launch and PARTIAL pooled. The pooled SE is conditional on the drawn classes (P2.md item 23).
- **No gain on three kinds.** ledge_up, pits and rubble never gained in any read (the b3 ledge and pit bands are
  above the blind envelope; rubble b3 is in-envelope and gated). TDR already handled hills and gravel blind
  [10 23:39].
- **Stand.**
  - SN stands 0.042–0.045 m below REFS(c1) on the 4 G5 lines. The R and HOLD controls are just as low: the head is within 0.004 m of each.
  - The R and HOLD controls are the S1-era rows (walker 968cc550, October), not rows run beside c1, as in RSD's accepted re-anchor; the 11:53 rule names those rows. REFS(c1) sits 0.014 m lower than REFS(TDR), so the controls fail the line by a thinner spare: 0.011 m against RSD's 0.025 m.
  - **Slower walk-stop settle (passing, thin margins).** Median settle after a walk-stop: SN 3.80 / 3.89 s and SH 3.74 / 3.64 s, against the lines 3.92 / 3.96 s (the S1-era R control + 1.0 s) and RSD's SN 2.70 / 2.80 and SH 2.86 / 3.00. SN's walk-stop also shows a higher KS (0.364 / 0.358 against 0.294 / 0.299) and more six-down time (0.357 / 0.343 against 0.243 / 0.207). REFS(c1) walks into the stop with more hip yaw than REFS(TDR) (3.62° / 3.61° against 2.75° / 2.73° mean).
    - *Cause, report-only CPU diagnostic* (workflow `wf_acdfce51-c27`; an opus analyst and an independent opus recompute agree on every number). The settle clock starts at the stop row, which is also the hand-over to the stand net, and ends when all six cams have held within 0.10 rad of home for 0.5 s (`2026-10-07_omni_phase2/scripts/p2s_stand.py` `stand_z2_metrics`). The gap is not uniform over an episode's three walk-stops: about +1.4 s after the back walk, +0.3 to +0.5 s after the forward walk, and none after the sideways walk (SN +0.04 s, SH −0.26 s). The pooled median of that two-mode mix shows it as about 1 s.
    - After the back walk, c1 hands over with the cams about twice as far from home as TDR (summed over legs 5.7 against 2.6 rad), and most of the extra time there is extra lifted homing. Both RT nets move the cams as fast as the RSD nets and follow the re-stance expert as closely.
    - About half the gap is not explained by the hand-over cam state: after the forward walk c1's deficit is the smaller one, and the RT nets are still slower. It may come from the RT stand nets or from hand-over features not measured, such as c1's faster-turning cams.
    - **The split is undetermined.** Settling it needs a walker × stand-net cross-pairing walk-stop eval (GPU), which was not run (§7). The fixed schedule stops every env at about the same gait phase, so the after-back deficit may partly be a phase artefact.
  - Other passing lines worse than RSD's: SN field_turn stand-fall 0.01 (line 0.020; RSD 0); SH bin-2 homed and settled share at es1 0.97 (line 0.93; RSD 1.00); SH K6 stillness taps/s 0.207 / 0.167 (line 1.0; RSD 0.139 / 0.148); SN walk-stop homed 0.98 / 0.98 (line 0.90; RSD 0.997 / 1.00); SN WATCH post-settle taps/s 0.818 / 0.815 (line 1.0; RSD 0.807 / 0.704).
  - SN's calibration table reads 1 / 15 cells "as listed" (reader rc 4, `sn.json` `head.reader_rc`), the same as RSD's SN read, which was accepted (`2026-11-07_rod_speed_dr/STAND_REPAIR.md:352`: each is a listed cell
    that cannot be read).
  - SH stands 0.041–0.044 m below REFS(c1) on the 4 K5.G5 lines; R and HOLD are within 0.0025 m of the head.
  - SH's 3 K6 lines (homed share bin 1 at es1 0.84 and es2 0.88, settled share bin 1 at es2 0.90) are under the S1-era
    lines (0.8407 / 0.8966 / 0.9010). They pass only after re-anchoring on the lazy R_h beside c1 (0.62 / 0.72 / 0.72,
    n 100 each; re-anchored lines 0.498 / 0.609 / 0.612). The rule is Task 4 DESIGN §3.6: when the control fails the
    same line, the rule treats the miss as caused by the walker, not the stand net.
  - SH's K4 passes the gate (line 3). The stand reader's own K4 column uses this read's R_h (1 miss per seed) as its
    line, so it marks es2's 2 misses as over. Beside c1 the lazy R_h misses 5 and 6, and the reader passes the head.
  - SH's read shows the same reader rc 4 as RSD's accepted SH read. Its calibration table reads 12 / 14 cells, and
    the 2 K5 non-regression cells are INVALID (a listed cell cannot be read). It also shows the same R_h stand-up
    premise MISS at es1 table_0p635 (reported, never gated).
- **Unaudited text.** PLAN's pass-4 fixes, the short arm's pass-2 fix and P2's amendment 1 were not re-audited. Amendment 1 is the diag bound used above.

## 6. What a bake changes and what it does not

| Role | Of record today | After a yes |
|---|---|---|
| Teacher | TDR, `rsd_9_p3b_rodspeed_dr/model_28991.pt` (03c3e472) | c1, `rt_11_*`, RT_s19 `model_29500.pt` (6d1f29be…, 12,596,767 B) |
| Student | c31000, `rsd_10_student` (ONNX bundled; .pt 155b5981 cited by sha, 122 MB) | C1S, `rt_12_student`: policy.onnx 86d1c801, depth_latest.onnx cca6952d, io json 4ee03d02; .pt 6b48ef33… (122,244,457 B, over the 100 MiB guard, so cited by sha) |
| Stand | `rsd_stand_net`: SH_RSD_TDR model_5000 (89f4428b), SN_RSD_TDR model_4000 (6cd82ff5) | `rt_stand_net`: SH 6f8b558e… (6,102,474 B), SN 5bedc50e… (6,097,610 B) |

- **TDR stays a live stage.** c1 is a resume of it.
- **Superseded, files kept.** c31000 and SH/SN_RSD_TDR.
- **back_overshoot** is reopened as a follow-up.
- **Deploy interface unchanged.** C1S's `omni_student_io.json` (4ee03d02) is byte-identical to c31000's. Its ONNX graphs differ from c31000's bundled ones (new weights), and parity is 1.2e-6 against its `.pt` (`p2/onnx_PICK.json`).
- **Code.** Before the bake the branch changed 31 code and test files outside the campaign dir, all default-off knobs: site/rt terrains, `rt_friction.py`, the omni knobs, `observations.py`, `tools/run_candidates.py` and tests. TDR's path is unchanged: neutrality was bitwise [10 19:40].
  - A merge moves main's code_sha to 89c1916b and touches the Task 4-locked `run_candidates.py`, so a relock and a CI-faithful simulation come first (P2.md Out of scope). Both are done on the branch (CHARTER "2026-11-15 12:30 — Merge prep"): origin/main merged in, the lock rewritten, and the simulation fails only main's own ids. Main's eval change (eval cd5ae167) moves no field any row reads, so nothing is re-read (followup `merge_relock`).
  - Push, PR and merge are your call after the bake.
- **What does not change.**
  - Payload and rod-speed DR ranges.
  - The scan layout (v1).
  - The bench-first rule for hardware.
  - Camera use is still untested: the student was distilled the same way as c31000.

## 7. Follow-ups

1. **P5 loose ground** (yours, 2026-11-12; not scoped). Candidate levers: support-failure or sinkage events per foot, footing that changes step to step and with load, ground compliance DR, an RFT force hook, and a per-foot slip-ratio read.
2. **back_overshoot.** Price it: a relative progress clamp (≤ 1.0 × |cmd|) or a one-sided overshoot term. Both are code changes and move the pin.
3. **Yaw retention.** An env-only arm: flat + field ≈ 0.4, `KRABBY_ORIENT_ALL_TILES` after a replay gate, and a turn dose, read on es13+ against fresh rows.
4. **ledge_up.** Measure the student's share with c1 and TDR ledge_up rows on fresh seeds. Then a per-kind ladder clamp or crossing-based promotion (a code change).
5. **Seed-2 replicate** of the short fine-tune, to test the recipe rather than the checkpoint.
6. **P3 perception** (your set-asides, PLAN settled 5): the scan-ablation knob port, the `_flat` twins, the site recorder port and the band-above rows, then a v2 scan for the > 0.14 m steps and potholes.
7. **P4 lost-foothold levers.** A looser training pitch termination, ≥ 4 loaded feet and stance slowing near drops. Every non-completer in these rows was a 0.5 rad tilt-limit tip.
8. **Walk-stop settle split** (§5 Stand). A walkstop cross-pairing eval, walker {TDR, c1} × stand net {RSD, RT}, for SN and SH and both env seeds, read through `stand_z2_metrics`. Plus a walkstop variant with jittered walk durations, to test whether c1's after-back cam deficit is a schedule-phase artefact. Each stand net carries its own walker's estimator and records the walker it was trained beside (`trained_beside_s1_sha256`). Whether the stand harness accepts a cross pairing has not been checked; it may need a report-only override.
9. **Housekeeping.**
   - Reconcile `readout.g_line`'s `and not le_noise` clause with the frozen rule text before reuse [11 06:50].
   - The heading-drift note at 33000 (73 fwd holds) is not investigated [11 08:15].
