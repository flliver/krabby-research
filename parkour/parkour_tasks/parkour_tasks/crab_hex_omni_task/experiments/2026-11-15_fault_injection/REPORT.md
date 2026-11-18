# Fault-injection sweep of the omni student of record: REPORT

**Status: COMPLETE 2026-11-17 (last row 01:37 local). Eval only, nothing trained. Nothing is adopted, baked, pushed or merged.** The work is on branch `feat/fault-injection-sweep` (worktree `.claude/worktrees/fault-sweep`): 9 local commits over base 78b8f9db, HEAD 65782be5. The branch has no remote and is 1 commit behind origin/main (06c8f8d1). `results/` is untracked except `results/null_fault/` and `results/smoke/`. A merge needs a Task 4 relock, because the hooks sit under Task 4's code pathspec (PLAN §8.2, CHARTER §2.3).

- **Campaign files.** All paths are under `crab_hex_omni_task/experiments/2026-11-15_fault_injection/`.
  - Plan: `PLAN.md`, approved 2026-11-15 as recommended.
  - Frozen spec: `CHARTER.md`, with the amendment log in §10.
  - Cells: `cells.yaml`.
  - Reader: `scripts/fault_read.py`, which writes `results/read.json` and `results/verdicts.md`.
  - GPU spend: `results/queue_ledger.jsonl`.
  - Pin: `results/pin.json` (HEAD 65782be5, pinned 2026-11-16 01:27).
- **Subject.** The depth student C1S, `RT_ST_s19/model_34499.pt` (6b48ef33), the omni student of record.
- **Report-only second subject.** The walker c1, `policy/rt_11_p3b_rough/model_29500.pt` (6d1f29be). It ran only on the action-path and actuator cells (B1, B4, A1–A5).
- **Rows.**
  - `read.json` reads 304 launches: 0 launch errors, 0 incomplete stamps, 0 unknown-cell stamps.
  - All 50 C1S cells ran suite + field_light at es1.
  - es2 ran on the 40 cells whose es1 worst-row F was below 0.5 (CHARTER §4).
  - 28 kind rows ran: 7 permanent perception cells × (ledge_down b2, pits b2) × es1–2.
  - c1 ran 15 cells plus its nominals at es1.
- **GPU.** 24.31 of the 27 GPU-h approved (sum of the ledger's `gpu_s`), against ≈ 24.6 planned (CHARTER §9). The ledger excludes the 2 FSD diagnosis suite launches (≈ 0.25 GPU-h by the per-row mean), so total spend is ≈ 24.56 GPU-h, still under the cap.
- **Verification.** Claim check (step 10 ✓, fable · high, 2026-11-17): every §3 row matched `read.json` and `verdicts.md`; 7 wrong, 3 unsupported and 10 wording findings were corrected in this version (mostly derived counts, "could flip" statements and unmarked inferences). Independent recompute from raw stamps (step 9 ✓, opus · medium, own code, 2026-11-17): fall counts match on 1224/1224 per-wave reads, ΔF worst (value and where) and the FALLS call on 105/105 cell-seeds (all C1S es1/es2 and all c1), and kind-row success on 32/32. R was spot-checked on 12 cells and the 5 differences come from defining when a transient fault clears: the reader uses the sidecar's `policy_visible_last + 1`, the recompute used `planned_last + 1`; with the reader's definition all match, and no verdict moves. Post-onset line verdicts were not independently recomputed.

Notation:
- **ΔF.** Post-onset fall share minus the paired nominal's share over the same per-env windows. A cell-seed takes its worst value over the 10 suite waves and field_light (n ≤ 100 each).
- **R.** The share of episodes tracking within ±0.25 of the command in a 2-s window after the fault. It is computed on translate waves and field_light translate holds, where nominal R is 0.96–1.00.
- **es1 / es2.** Paired values are es1 / es2.
- **Waves.** w0 stand, w1 fwd, w2 lat_l, w3 lat_r, w4 back, w5 diag, w6 turn_l90, w7 turn_r90, w8 move_turn, w9 move_turn_r. fl = field_light.
- **†** marks a count I made for this REPORT from `read.json`. It is in no verdict.
- **Every A-family read is "sim plant, force side not enforced."**

## 1. Bottom line

1. **What the policy is robust to (TOLERATED on both seeds):**
   - stale action bursts up to 0.2 s (B1 2/5/10 steps);
   - random tick loss up to p = 0.2 (B2);
   - a 0.1-s runaway (B4);
   - a position-hold stop up to 0.5 s (B6);
   - an IMU freeze up to 2 s (S5);
   - roll/pitch bias up to 0.1 rad (S6);
   - a 0.1-s vxy dropout (S1);
   - a front-depth blackout up to 2 s (S10);
   - a side/rear ToF freeze, 2 s and permanent (S11);
   - the ZED's 30-Hz IMU and vxy cadence (S12);
   - contact_fill set to 0, which is today's HAL behaviour (S8_zero).

   Every one of these has ΔF ≤ 0.02 and R ≥ 0.94 (`verdicts.md`, Cell-seeds).
2. **What it falls on (FALLS, ΔF > 0.10):**
   - vxy lost for 0.5 s or longer: S1 0.88 at 0.5 s, 0.99 at 2 s, 1.00 permanent. 242 of 253 post-onset falls at 0.5 s come within 2 s of onset†.
   - vxy held for 2 s or permanently: S2 0.22 / 0.20 at 2 s, 0.98 permanent.
   - vxy random-walk drift: S4 0.38 / 0.51, field_light only.
   - ZED loss for 2 s or permanently: Z1 0.57 at 2 s, 1.00 permanent. This includes falls while standing (w0 0.20 at 2 s, 0.45 permanent).
   - a 0.5-s stale action: B1 0.22 / 0.30.
   - a 0.5-s runaway: B4 0.43 / 0.33.
   - a locked leg: A1 front 0.12 / 0.19; A1 mid 0.10 / 0.15, split.
   - a limp leg: A2 front and rear 1.00 (standing fails too: stand-wave F 0.53 front, 0.56 rear); A2 mid 0.32 / 0.25.
   - a locked mid leg at 90 kg: A5 0.41 / 0.37.
   - the anchors: T10b (vxy zero from step 0) and D1 (today's HAL observation contract, report only).
3. **Where it degrades without (many) extra falls (DEGRADES):** tracking or turning is lost while ΔF stays at 0.00–0.09.
   - **S2 0.5 s:** ΔF 0.08 / 0.07 (marginal flags, but no ±2-episode change moves the cell verdict).
   - **S3 vxy bias 0.1 and 0.2 m/s:** R 0.31–0.61.
   - **S8 contact_fill held:** R 0.06 / 0.12; lateral speed ratio 0.52–0.60.
   - **S10 permanent:** turn facing 0.74 / 0.75 against nominal 0.90.
   - **T12:** see item 6.
   - **B3, latency 20–60 ms, and its 40-ms control B5:** R 0.77–0.83 on lat_l.
   - **A1 rear, A3 slow leg, A4 stuck cam.**
   - **Z1 0.5 s:** split RECOVERS / DEGRADES.
4. **Tolerated durations** (CHARTER §5.3: the longest d such that it and every shorter level are TOLERATED or RECOVERS on both seeds; `read.json` `tolerated_durations`):

   | Family | Levels tested | Tolerated d | Stopped by |
   |---|---|---|---|
   | B1 stale action (rods keep slewing to the old target) | 40, 100, 200, 500 ms | **0.2 s** (10 steps) | 0.5 s FALLS (0.22 / 0.30) |
   | B4 runaway (today: open-loop PWM, no watchdog) | 0.1, 0.5 s | **0.1 s** (5 steps) | 0.5 s FALLS (0.43 / 0.33) |
   | B6 stop (a watchdog that holds position) | 0.1, 0.5 s; follow-up 2 s, permanent | **≥ 2 s** (2 s RECOVERS; follow-up) | permanent DEGRADES (ΔF 0.07 / 0.05, field progress R 0) |
   | S1 vxy → 0 | 0.1, 0.5, 2 s | **0.1 s** | 0.5 s FALLS (0.88, es1) |
   | S2 vxy held | 0.5, 2 s; follow-up 0.1, 0.2 s | **0.2 s** (follow-up) | 0.5 s DEGRADES (marginal) |
   | S5 IMU frozen | 0.1, 0.5, 2 s | **≥ 2 s** | – |
   | S10 front depth raw 0 | 0.5, 2 s | **≥ 2 s** | – (permanent DEGRADES) |
   | S11 ToF frozen | 2 s | **≥ 2 s** | – (permanent TOLERATED) |
   | Z1 ZED loss (S10 + S5 + S2 together) | 0.5, 2 s; follow-up 0.1, 0.2 s | **0.2 s** (follow-up) | 0.5 s DEGRADES (split) |

5. **Z1 is not predictable from its parts.** At 2 s, the parts alone read 0.00 (S5_2), 0.00 (S10_2) and 0.22 / 0.20 (S2_2). Together they read 0.57, with falls on every wave including stand and the turns (`read.json` per-wave F).
6. **Tie-in to the 2026-11-07 sweep (anchors).**
   - **T10b reproduces the old finding on C1S.** Survival is 0.00 on fwd, back, diag and move_turn, and completion is 0.00. The old values are the same (old REPORT:144).
   - **T12: same pooled picture, flipped weak hand.** On the old sweep's whole-hold pooled lines, C1S under T12 passes every line, as F19's student did (old REPORT:119): diag |r−1| 0.247 / 0.246, turn facing 0.903 / 0.904. It reads DEGRADES here only on this sweep's post-onset, per-hand lines: diag tracking 0.253 / 0.254 (line 0.25), and turn facing 0.785 / 0.783. The weak hand has flipped from lat_r (old 0.644) to lat_l (0.76 against lat_r 0.86, on both reads). The body still rises (+2.6 to 2.7 cm).
7. **The teacher shares the fragility.** c1's es1 verdict matches C1S's on 14 of the 15 shared action and actuator cells†. On the 15th, A1_mid, the two agree once both of C1S's seeds are combined. So the B and A failures most likely come from the teacher, not from distillation (inference).
8. **Kind rows (report only).**
   - Corrupting depth costs success on both kinds, pits included (pits never gained in rough-terrain training). Front-only S10_perm loses 0.05–0.22 on ledge_down and 0.22–0.32 on pits; T12 (all images zeroed) loses 0.29–0.44.
   - The ToF freeze moves success by at most 0.09.
   - Permanent vxy loss or hold, or permanent ZED loss, drops success to 0.00.
   - The sweep cannot say whether this is terrain information lost or a gait shifted by an off-distribution image (§5).
9. **Hardware findings that need no GPU** (PLAN §8.1). These belong in REPORT §1 whatever the sweep shows:
   - there is no host-link watchdog, and open-loop PWM keeps rods running when the link drops;
   - no observation carries a validity or health bit;
   - there is no e-stop and no sit-down;
   - joint position, joint velocity and contact are placeholders in the HAL;
   - the omni observation contract is not wired;
   - the HAL runs at 100 Hz against the sim's 50 Hz.

   The sweep adds deadlines to the first three. These are inferences from separate cells. A runaway is tolerated at 0.1 s and falls at 0.5 s (B4), and a 0.5-s position hold is tolerated (B6), so the watchdog should trip into a hold within 0.1 s; a runaway followed by a hold was not tested. A VO-loss response must act within ≈ 0.1 s (S1). A ZED-loss response must act in under 0.5 s (Z1). D1 shows the placeholder contract fails walking (§4).

## 2. The three ranked lists (PLAN §1)

### 2.1 Riders for the second distill

These go on top of the rod-speed DR rung-2 list (depth dropout and blackout, obs noise, current noise, the vxy decision; rod-speed DR REPORT:703). Ranked by likelihood × severity.

1. **ZED grab failure as one event (Z1).**
   - Dose: front depth raw 0 m (image −0.5), IMU held at its last value and vxy held at its last value, all at the same onset.
   - Duration: 0.1 s up to 2 s.
   - Evidence: 0.5 s DEGRADES (ΔF 0.04 / 0.09); 2 s FALLS 0.57. It is the most likely real failure (PLAN §3.1b), and its effect does not follow from its parts.
   - A permanent ZED loss is a hardware item (§2.2 item 5), not a rider.
2. **vxy dropout-to-zero and hold-last (S1, S2), 0.1 s to 2 s.**
   - S1 at 0.5 s: F 0.88 on fwd and 0.73 on back.
   - S2 at 2 s: 0.22 / 0.20.
   - With vxy zeroed, the policy overshoots its command (fwd/back/diag ratio 1.32–1.99) and pitches out (pitch_rms 0.123 against 0.027 nominal; S1_perm). This is the old "lunge" (old REPORT:46, :144).
   - Whether a rider can remove this dependence at all is the vxy decision (§9 item 2).
3. **vxy bias and drift (S3, S4).**
   - S3: bias 0.1–0.2 m/s at a random heading. R 0.31–0.61; the median ratios stay near 1 because signed errors cancel. Kind rows lose 0.21–0.37.
   - S4: random walk ≈ 0.2 m/s RMS at 10 s. FALLS 0.38 / 0.51 on field_light; the realised 10-s RMS was 0.197 / 0.196 m/s.
4. **Front-depth blackout with the hardware stimulus (S10: raw 0 m, image −0.5), permanent.**
   - DEGRADES on the flat suite: turn facing 0.74–0.75, turn_l90 0.69–0.70, R on diag 0.55 / 0.71.
   - Kind rows lose 0.05–0.32.
   - Add T12's normalised-zero image only if the deployed normaliser maps a dropout there. That normaliser does not exist yet (PLAN §3.1).
5. **Current-sense / contact_fill hold-last (S8_held).** DEGRADES: the lateral gait slows to ratio 0.52–0.60 and R falls to 0.06 / 0.12. A zero-fill (S8_zero) is already tolerated. This extends the "current noise" rider.
6. **Optional: action latency of U{1, 2, 3} steps (B3).** DEGRADES on R only (0.77–0.83), with ΔF ≤ 0.01. A hardware latency budget (§2.2 item 6) covers it more cheaply.

**Not needed (TOLERATED on both seeds):** IMU hold ≤ 2 s (S5), tilt bias ≤ 0.1 rad (S6), the 30-Hz cadence (S12), the ToF freeze (S11), contact zero (S8_zero), tick loss (B2). Obs noise was not tested here; the old sweep found T08 ROBUST at k ≤ 1.

### 2.2 Hardware-side mitigations

1. **Wire the omni observation contract (D1).** It is a prerequisite.
   - Today's contract (joint position = last target, joint velocity = 0, contact = 0) is active from step 0 and fails walking: 104 of 1000 suite episodes fall before the onset table and 295 after. Survival is fwd 0.12, back 0.19, diag 0.42, move_turn 0.14, and field_light completion is 0.14.
   - The policy also stops moving sideways (lateral ratio 0.05–0.09) and stops turning (facing 0.02–0.06).
   - S8_zero is TOLERATED, so the damage comes from the joint position/velocity placeholders, not from contact (inference).
2. **Host-link watchdog: hold position within 0.1 s** (inference from B4 and B6; see §1 item 9).
   - Today's runaway (B4) is tolerated only at 0.1 s and FALLS at 0.5 s (0.43 / 0.33). Even the stand wave falls (0.43 / 0.33).
   - A position hold with cams stopped (B6) is TOLERATED at 0.5 s (ΔF ≤ 0.01).
   - 0.1 s is 5 policy ticks at 50 Hz, or 10 HAL ticks at 100 Hz (arithmetic).
   - A hold longer than 0.5 s is untested.
3. **A VO health flag and its response.**
   - A vxy dropout is tolerated for only 0.1 s, and a hold-last is not tolerated at 0.5 s. Bias at 0.1 m/s, the smallest level tested, already DEGRADES, so the VO bias spec should be below 0.1 m/s. That rests on R alone (0.58 / 0.61; ΔF ≤ 0.01, no line fails).
   - The flag needs a validity bit on vxy (PLAN §8.1). At 30 Hz, 0.1 s is about 3 VO frames (arithmetic), so the flag must trip within about 3 missed frames.
   - Under S1_perm, the stand and turn waves show 0 falls, which suggests zeroing the translation command may be safe. That is inference: the walk-to-stand transition under the fault was not tested.
4. **A ZED grab-failure flag with a response in under 0.5 s.**
   - A single `grab()` failure zero-fills depth and holds IMU and vxy (PLAN §3).
   - Z1 already DEGRADES at 0.5 s, and shorter levels are untested.
   - Commanding a stand does not save it: the stand wave falls 0.20 at 2 s and 0.45 permanent.
5. **A sit-down / safe-stop fallback.** None exists (PLAN §8.1). It is needed for a ZED loss beyond the hold window, a permanent VO loss, a B6 hold beyond 0.5 s, and every actuator fault in §2.3.
6. **A latency budget of ≤ 20 ms end-to-end, one policy step (preferred; ≤ 40 ms acceptable).** The nominal sim applies the previous step's action (the delay-1 note in `fault_read.py`). A constant 40 ms (B5) passes every post-onset line with ΔF 0 and reads DEGRADES only through R 0.80 on lat_l; a variable 20–60 ms (B3) reads R 0.83 / 0.77, ΔF ≤ 0.01. The old sweep's budget was ≤ 40 ms (old REPORT:117); this sweep tightens it only on the R read. The 100-Hz HAL loop itself is untested (S12 tests only the 30-Hz IMU/vxy cadence).

**Not needed:** tick loss up to p = 0.2 (B2); IMU mount bias up to 0.1 rad (S6), so ±0.1 rad is enough calibration tolerance; the 30-Hz cadence (S12); SEN0581 stale frames (S11, so the 3-s freshness check that only logs is enough in sim); current-sense zero-fill (S8_zero).

### 2.3 Faults that would need a trained fault regime

c1 fails in the same way as C1S on items 1, 2 and 4 (§6); item 3 never ran on c1. Any such regime belongs at teacher level, not in a distill rider.

1. **Limp leg (A2, broken linkage).** On a limp front or rear leg it falls on every walking wave (F 1.00) and about half its stand episodes fall (stand F 0.53 front, 0.56 rear); completion 0.00. A limp mid leg gives FALLS 0.32 / 0.25 and walking at ratio 0.31–0.56. This is the five-leg case.
2. **Locked leg (A1).** This also covers an unpowered rod, since the lead screws self-lock.
   - Front: FALLS 0.12 / 0.19. Mid: FALLS 0.10 / 0.15 (split). Rear: DEGRADES 0.04 / 0.06.
   - Lateral speed ratio drops to 0.61–0.80, and fwd to 0.56–0.62 with the rear leg locked.
   - At 90 kg (A5), field_light completion falls to 0.58 / 0.63 against 0.99 nominal.
3. **Permanent ZED loss** (Z1 perm 1.00; S1 perm 1.00; S2 perm 0.98). This means walking or standing on proprioception alone. It is needed only if hardware cannot guarantee the sit-down in §2.2 item 5.
4. **Slow leg or stuck cam** (A3 ×0.3 slew; A4). DEGRADES with few falls (A3 R 0.66–0.74, ΔF ≤ 0.01; A4 ΔF 0.03–0.05). Low priority; it could ride a per-leg rod-speed DR.

## 3. Verdict table (C1S)

Source: `verdicts.md` (Cells, Cell-seeds); the "why" column comes from `read.json` `verdict_reads`.
- **One-seed cells.** These ran es1 only because their es1 worst-row F was ≥ 0.5 (CHARTER §4, CHARTER.md:124–125). That is FALLS on one seed by rule, not missing data. Their es2 cell-seed rows in `verdicts.md` exist only for their kind rows (S1_perm, S2_perm, Z1_perm).
- **Marginal.** A ΔF within 2 episodes of a threshold (CHARTER §5.3). The flags column says whether a ±2-episode change could move the verdict.

| Cell | Fault, in plain words | Verdict | ΔF worst es1 / es2 | R worst es1 / es2 | Flags | Why |
|---|---|---|---|---|---|---|
| S1_0p1 | vxy slot zero for 0.1 s (VO lost) | TOLERATED | 0.00 / 0.00 | 1.00 / 0.98 | – | – |
| S1_0p5 | vxy zero for 0.5 s | FALLS | 0.88 w1 / – | 0.90 / – | one seed | fwd 0.88, back 0.73; F2 242 of 253† |
| S1_2 | vxy zero for 2 s | FALLS | 0.99 w5 / – | 0.00 / – | one seed | fwd 0.96, back 0.94, diag 0.99 |
| S1_perm | vxy zero to episode end | FALLS | 1.00 w1 / – | 0.00 / – | one seed | fwd/back/diag/move_turn 1.00; overshoot 1.32–1.99 |
| S2_0p5 | vxy held at last value for 0.5 s (ZED grab fail) | DEGRADES | 0.08 w8 / 0.07 w8 | 0.99 / 0.98 | marginal 0.05, 0.10; FALLS needs es1 +3, RECOVERS needs es1 −3 and es2 −2: stable within ±2 | ΔF > 0.05; no line fails |
| S2_2 | vxy held for 2 s | FALLS | 0.22 w8 / 0.20 fl | 1.00 / 0.96 | – | survival.move_turn 0.835 / 0.865; completion 0.85 / 0.80 |
| S2_perm | vxy held to episode end | FALLS | 0.98 fl / – | 0.29 fl / – | one seed | suite ≤ 0.36 (w8); field_light completion 0.02 |
| S3_0p1 | vxy bias 0.1 m/s, random heading | DEGRADES | 0.00 / 0.01 | 0.58 w3 / 0.61 w3 | marginal 0.03, cannot flip | R only |
| S3_0p2 | vxy bias 0.2 m/s | DEGRADES | 0.07 fl / 0.08 fl | 0.35 w3 / 0.31 w3 | marginal 0.10; FALLS needs es2 +3: stable within ±2 | R; pitch_rms fl 0.112 / 0.127 |
| S4_perm | vxy random walk, ≈ 0.2 m/s RMS at 10 s | FALLS | 0.38 fl / 0.51 fl | 0.70 fl / 0.74 w3 | – | field_light only (suite ≤ 0.01); completion 0.62 / 0.49 |
| S5_0p1 | IMU frozen 0.1 s | TOLERATED | 0.00 / 0.00 | 0.96 / 0.98 | – | – |
| S5_0p5 | IMU frozen 0.5 s | TOLERATED | 0.00 / 0.00 | 1.00 / 0.98 | – | – |
| S5_2 | IMU frozen 2 s | TOLERATED | 0.00 / 0.01 | 1.00 / 1.00 | marginal 0.03, cannot flip | – |
| S6_0p05 | roll/pitch bias 0.05 rad | TOLERATED | 0.00 / 0.00 | 1.00 / 0.98 | – | – |
| S6_0p1 | roll/pitch bias 0.1 rad | TOLERATED | 0.00 / 0.00 | 1.00 / 0.96 | – | – |
| S8_zero | contact_fill set to 0 (today's HAL) | TOLERATED | 0.00 / 0.00 | 0.94 w4 / 0.96 w4 | – | – |
| S8_held | contact_fill held (stale Mega frame) | DEGRADES | 0.05 fl / 0.02 w4 | 0.06 w5 / 0.12 w5 | marginal 0.03, cannot flip | tracking side/diag/lat_l/lat_r fail |
| S10_0p5 | front ZED depth raw 0 m (image −0.5), 0.5 s | TOLERATED | 0.00 / 0.00 | 1.00 / 0.98 | – | – |
| S10_2 | same, 2 s | TOLERATED | 0.00 / 0.00 | 1.00 / 0.98 | – | – |
| S10_perm | same, permanent | DEGRADES | 0.02 fl / 0.05 fl | 0.55 w5 / 0.71 w5 | marginal 0.03, cannot flip | turn facing 0.739 / 0.752 (nominal 0.896 / 0.899) |
| S11_2 | side + rear ToF frozen, 2 s | TOLERATED | 0.00 / 0.00 | 1.00 / 1.00 | – | – |
| S11_perm | same, permanent | TOLERATED | 0.00 / 0.00 | 1.00 / 0.98 | – | – |
| Z1_0p5 | ZED loss (S10 + S5 + S2 together), 0.5 s | DEGRADES | 0.04 w8 / 0.09 w9 | 0.98 / 0.94 | **split** (RECOVERS / DEGRADES); marginal 0.03 / 0.05 / 0.10 | ΔF only; no line fails; within ±2 only FALLS (es2 +2) is reachable |
| Z1_2 | ZED loss, 2 s | FALLS | 0.57 w6 / – | 1.00 / – | one seed | every wave falls, including stand 0.20 |
| Z1_perm | ZED loss, permanent | FALLS | 1.00 fl / – | 0.14 fl / – | one seed | stand 0.45, turn_l90 0.98 |
| B1_2 | action held (stale rod target) 40 ms | TOLERATED | 0.00 / 0.00 | 1.00 / 0.96 | – | – |
| B1_5 | same, 100 ms | TOLERATED | 0.00 / 0.00 | 1.00 / 0.96 | – | – |
| B1_10 | same, 200 ms | TOLERATED | 0.02 w1 / 0.00 | 0.98 / 0.98 | marginal 0.03: worst case RECOVERS | – |
| B1_25 | same, 500 ms | FALLS | 0.22 w8 / 0.30 w0 | 0.96 / 0.95 | – | survival lines 0.78–0.88 |
| B2_0p05 | random tick loss p 0.05 | TOLERATED | 0.00 / 0.00 | 1.00 / 0.98 | – | – |
| B2_0p2 | random tick loss p 0.2 | TOLERATED | 0.00 / 0.00 | 1.00 / 0.98 | – | – |
| B3_perm | **action latency 20–60 ms** (U{1, 2, 3} steps) | DEGRADES | 0.01 fl / 0.00 | 0.83 w2 / 0.77 w2 | marginal 0.03, cannot flip | R only; lat_l ratio 0.82 |
| B5_perm | constant 40-ms action delay (B3's control) | DEGRADES | 0.00 / 0.00 | 0.80 w2 / 0.80 w2 | – | R only; lat_l ratio 0.82 |
| B4_0p1 | runaway: rods keep last slew, cams last velocity, 0.1 s | TOLERATED | 0.00 / 0.00 | 1.00 / 0.98 | – | – |
| B4_0p5 | runaway, 0.5 s | FALLS | 0.43 w0 / 0.33 w0 | 0.90 / 0.93 | – | every wave falls; completion 0.69 / 0.77 |
| B6_0p1 | stop: rods held at current length, cams 0, 0.1 s | TOLERATED | 0.00 / 0.00 | 0.98 / 0.95 | – | – |
| B6_0p5 | stop, 0.5 s | TOLERATED | 0.01 w8 / 0.01 w8 | 1.00 / 0.98 | marginal 0.03, cannot flip | – |
| A1_front | left front hip + knee rods locked at onset | FALLS | 0.12 w8 / 0.19 w8 | 0.25 w3 / 0.20 w3 | marginal 0.10 on es1 only | tracking side/back/lat_r |
| A1_mid | left mid leg rods locked | FALLS | 0.10 w8 / 0.15 fl | 0.17 w3 / 0.22 w3 | **split** (DEGRADES / FALLS); marginal 0.10 on es1 | 7 lines fail on both seeds |
| A1_rear | left rear leg rods locked | DEGRADES | 0.04 fl / 0.06 w3 | 0.34 w1 / 0.27 w3 | marginal 0.03, cannot flip | fwd ratio 0.56 / 0.62 |
| A2_front | left front leg limp (stiffness, damping, effort 0) | FALLS | 1.00 w4 / – | 0.00 / – | one seed | stand 0.53; completion 0.00 |
| A2_mid | left mid leg limp | FALLS | 0.32 w8 / 0.25 w8 | 0.01 / 0.00 | – | ratios 0.31–0.56 |
| A2_rear | left rear leg limp | FALLS | 1.00 w1 / – | 0.00 / – | one seed | stand 0.56; completion 0.00 |
| A3_mid | left mid leg rod slew ×0.3 | DEGRADES | 0.01 fl / 0.00 | 0.74 w3 / 0.66 w3 | marginal 0.03, cannot flip | turn_l90 facing 0.783 (es2) |
| A4_mid | left mid cam stuck at its onset angle | DEGRADES | 0.03 fl / 0.05 fl | 0.86 w1 / 0.895 fl | marginal 0.03, cannot flip | R; es2 ΔF |
| A5_mid | A1 mid at 90 kg | FALLS | 0.41 fl / 0.37 fl | 0.14 w3 / 0.21 w3 | – | completion 0.58 / 0.63 (nominal 0.99) |
| T10b | anchor: vxy zero from step 0 | FALLS | 1.00 w9 (n 2) / – | 0.00 fl (no qualifying episode) / – | one seed; marginal only because n = 2 | §4 |
| T12 | anchor: all depth images zero at the encoder, from step 0 | DEGRADES | 0.06 w8 / 0.09 fl | 0.47 w5 / 0.37 w5 | marginal 0.10: es2 +2 episodes → FALLS | §4 |
| S12_30 | anchor: IMU + vxy refreshed at 30 Hz | TOLERATED | 0.00 / 0.00 | 0.98 / 0.98 | – | – |
| D1_1 | anchor, report only: today's HAL observation contract | FALLS | 0.91 w8 (n 65) / – | 0.00 / – | one seed | §4 |

**Split cells.**
- **A1_mid.** es1 ΔF is exactly 0.10, which is not > 0.10, so es1 reads DEGRADES. es2 reads 0.15 (FALLS), so the cell is FALLS. Both seeds fail the same 7 lines (tracking side/diag/lat_l/lat_r, turn facing ×3), with R 0.17 / 0.22.
- **Z1_0p5.** es1 ΔF 0.04 with R 0.98 reads RECOVERS. es2 ΔF 0.09 reads DEGRADES. No line fails on either seed. The cell takes the worse seed: es2 needs −4 episodes to reach RECOVERS, so within ±2 per seed the cell stays DEGRADES or goes to FALLS (es2 +2). Z1 has no tolerated duration at the grid's levels, and that is firm; the 2026-11-18 follow-up found 0.2 s tolerated (§9, follow-up results).

**Mechanism notes** (from `read.json` per-hand post-onset ratios; report only).
- **vxy zero overshoots the command.** Under S1_perm, the fwd, back and diag ratios are 1.32–1.99.
- **S2_perm hold-last and S4 drift fail mostly on field_light** (S2_2 does not: es1 suite w8 0.22 against field_light 0.15). S2_perm's suite F is ≤ 0.36, but its field_light F is 0.98. S4's suite F is ≤ 0.01. A suite wave holds one command for 12 s, so a held vxy stays near-correct, and only 4–8 s of drift accumulate after onset. field_light changes command every 8 s and exposes 13–63 s after onset (inference).
- **B3 and B5 read alike.** lat_l ratio 0.82 / 0.82, lat_r 0.86 / 0.86, R 0.77–0.83 against 0.80. The constant 40-ms delay carries the loss, so B3 reads as **latency 20–60 ms**. This matches the old sweep's T07 d = 2 on F19's student (lat_l 0.859 / 0.845, old REPORT:117; cross-student, cross-pin).
- **Falls under action-path faults come fast.** 287 of 301 B4_0p5 es1 falls (252 of 269 es2), and 168 of 177 B1_25 es2 falls, land within 2 s of onset†.

## 4. Anchors (T10b, T12, S12, D1)

Anchors are active from step 0 but read on the row's standard onset table (CHARTER §3.6). So their "pre-onset falls" column (`verdicts.md`) counts falls *under the fault* before the read window, not leaks. Their F counts only the episodes that survived to the onset. They are read against both nominals by design.

- **T10b (vxy zero from step 0).**
  - 632 pre-onset falls: 533 on the suite, 99 on field_light†.
  - fwd, back, diag and move_turn: 100 of 100 fell before the onset step; move_turn_r 98.
  - Own-hold survival: fwd / back / diag / move_turn 0.00, side 0.48, turn 0.995, stand 1.00. Completion 0.00.
  - Per hand†: lat_l 0.38, lat_r 0.58 (22 + 40 and 12 + 30 falls).
  - Old sweep on F19's student: 0.00 on the same four lines, side 0.47 (lat_l 0.14, lat_r 0.80), turn 0.945, completion 0.00 (old REPORT:144). **The finding stands on the rough-terrain student, and it is less hand-asymmetric.**
  - The ΔF worst of 1.00 at w9 comes from n = 2. The FALLS verdict stands on lat_l (F 0.51, n 78).
- **T12 (all images zeroed at the encoder, `omni_student_step.py:201`).**
  - DEGRADES on both seeds: diag tracking 0.253 / 0.254 (line 0.25); turn facing 0.785 / 0.783; turn_l90 0.753 / 0.728 (nominal 0.91–0.93); R on diag 0.47 / 0.37; ΔF 0.06 / 0.09.
  - Post-onset hands: lat_l 0.759 / 0.763, lat_r 0.857 / 0.855. Both pass, lat_l by 0.009 / 0.013.
  - h_meas is 1.139–1.140 m against 1.113 m: +2.6 to 2.7 cm.
  - Old sweep (F19's student): every pooled line passed, lat_l 0.899, lat_r 0.644, body +3–4 cm (old REPORT:119). On the same pooled lines C1S also passes; the weak hand has moved from lat_r to lat_l. Whether C1S leans on depth more on flat ground is not shown (the old lat_r deviation, 0.356, is larger than any C1S deviation).
  - Kind rows: §5.
- **S12 (30-Hz sample-and-hold of IMU and vxy).** TOLERATED on both seeds; every post-onset line is within 0.07 of nominal.
- **D1 (today's HAL contract; report only).**
  - 165 pre-onset falls: 104 suite, 61 field_light†.
  - Total falls per 100†: fwd 88, back 81, diag 58, move_turn 94, move_turn_r 78. Stand, lateral and turn waves: 0.
  - The policy does not translate sideways (lat ratio 0.069 left / 0.053 right) and does not turn (facing 0.015 left / 0.063 right).
  - field_light completion 0.14; pitch_rms 0.149.
  - As PLAN §3.4 expected, D1 feeds the hardware list, not a verdict.

## 5. Kind rows (report only; CHARTER §5.2)

Success means the schedule completed and the root reached r_max ≥ 3.2 m. Onset falls in the approach, before the first edge or pit contact (CHARTER §3.2). Nominal success:
- ledge_down: 0.52 (es1), 0.60 (es2); the nominal pairs are same-class and identical.
- pits: 0.57 (launch A) and 0.55 (launch B, the other class) at es1; 0.45 at es2.

| Cell | ledge_down Δsuccess es1 / es2 | pits Δsuccess es1 / es2 | F faulted / ref (worst) |
|---|---|---|---|
| S10_perm (front raw 0) | −0.05 / −0.22 | −0.32 / −0.22 | pits 0.68 / 0.40 |
| T12 (all images zero, from step 0) | −0.29 / −0.31 | −0.44 / −0.29 | pits 0.85 / 0.40 |
| S11_perm (ToF frozen) | +0.02 / −0.09 | −0.03 / −0.01 | ledge_down 0.49 / 0.40 |
| S3_0p2 (vxy bias 0.2) | −0.22 / −0.33 | −0.37 / −0.21 | pits 0.81 / 0.42 |
| S1_perm, S2_perm, Z1_perm | −0.52 / −0.60 (success 0.00) | −0.55 to −0.57 / −0.45 (success 0.00) | 1.00 |

**Against the PLAN hypothesis** (PLAN.md:248: does depth matter on these kinds at all? RT REPORT: pits never gained, RT REPORT:33, :167):
- **Depth input matters on both kinds.** Front-only corruption (S10_perm) costs 0.05–0.22 on ledge_down and 0.22–0.32 on pits; zeroing every image (T12) costs 0.29–0.31 and 0.29–0.44.
- **pits loses at least as much as ledge_down in 3 of the 4 depth pairs** (T12 es2 is the exception, −0.29 against −0.31), even though rough-terrain training never improved pits.
- **The side/rear ToF stale frame (S11) costs ≤ 0.09.**
- **The gap from flat ground.** The same faults barely add falls on flat ground (S10_perm suite ΔF ≤ 0.01) but add 0.04–0.28 on the kinds.
- **What stays open: terrain use against off-distribution sensitivity.** Both stimuli are off-distribution images that also shift the flat gait (S10_perm turn facing −0.15; T12 speed ratio 0.75–0.89, body +2.6 cm). So the sweep cannot separate lost terrain information from a gait perturbed by an unfamiliar image. The old sweep made the same caveat (old REPORT:119).
- **The test that would separate them.** A frozen front frame, which is an in-distribution stale image, would separate the two. It is not in the grid: S11 freezes only the ToF cameras.

## 6. Teacher c1 (report only; es1, one seed)

c1 is the teacher, so it ran no sensor cells (CHARTER §4). In `verdicts.md` its rows read "(pending)". That is the reader's marker for a one-seed verdict below the F ≥ 0.5 one-seed rule (`fault_read.py` `combine_seeds`). c1 runs one seed by design, so these rows are complete, not missing.

| Cell | C1S es1 verdict | ΔF | completion | c1 verdict | ΔF | completion |
|---|---|---|---|---|---|---|
| B1_2 / B1_5 | TOLERATED | 0.00 | 1.00 | TOLERATED | 0.00 | 1.00 |
| B1_10 | TOLERATED | 0.02 w1 | 1.00 | TOLERATED | 0.02 w8 | 1.00 |
| B1_25 | FALLS | 0.22 w8 | 0.88 | FALLS | 0.28 w0 | 0.78 |
| B4_0p1 | TOLERATED | 0.00 | 1.00 | TOLERATED | 0.00 | 1.00 |
| B4_0p5 | FALLS | 0.43 w0 | 0.69 | FALLS | 0.41 w0 | 0.67 |
| A1_front | FALLS | 0.12 w8 | 0.97 | FALLS | 0.15 w8 | 0.92 |
| A1_mid | DEGRADES | 0.10 w8 | 0.90 | FALLS | 0.17 fl | 0.83 |
| A1_rear | DEGRADES | 0.04 fl | 0.96 | DEGRADES | 0.02 w3 | 0.98 |
| A2_front | FALLS | 1.00 w4 | 0.00 | FALLS | 1.00 w4 | 0.03 |
| A2_mid | FALLS | 0.32 w8 | 0.86 | FALLS | 0.18 w8 | 0.85 |
| A2_rear | FALLS | 1.00 w1 | 0.00 | FALLS | 1.00 w1 | 0.01 |
| A3_mid | DEGRADES | 0.01 fl | 0.99 | DEGRADES | 0.01 fl | 0.99 |
| A4_mid | DEGRADES | 0.03 fl | 0.97 | DEGRADES | 0.04 fl | 0.96 |
| A5_mid (L090) | FALLS | 0.41 fl | 0.58 | FALLS | 0.40 fl | 0.60 |

- **Agreement.** The verdicts match on 14 of 15 cells†. A1_mid sits at the C1S threshold, and C1S's es2 reads FALLS.
- **Gaps.** The largest ΔF gap is A2_mid (0.32 against 0.18). Every other gap is ≤ 0.07.
- **Nominals.** Both nominals read suite falls 0 / 1000 and field_light completion 1.00. At L090, c1 reads 1.00 and C1S 0.99.
- **Reading.** The action-path and actuator fragility comes from the teacher (inference). A student rider cannot remove it; it needs a teacher-level regime (§2.3) or hardware (§2.2).

## 7. Execution

**Pre-flight and smokes** (2026-11-15 23:16 → 2026-11-16 01:27; 1.68 GPU-h):
- The knob-off cfg dump was byte-identical on suite, field_light, kind and L090 (`results/smoke/dump_check.txt`).
- All 44 smoke rows ended ok (CHARTER §10, 03:10). They got there after 7 smoke stops (`results/STOP.2026-11-16_*.json`, all in the smoke phase). Each stop was cleared by a reader fix or a CHARTER amendment before the grid phases started (commits b63d3243 … 37b02d77; pin 01:27). Five rows that the read counts as grid data were launched during the smoke phase (A2_mid suite es1, S1_perm field_light and ledge_down es1, A5_mid field_light es1, c1 A1_mid field_light); two of them triggered the last two amendments. The amendments change identity checks only, not thresholds. CHARTER §10's time labels are on a different clock from the ledger and commits.
- Baseline gate: nominal A field_light es1 was bitwise equal to P2SEL, and the suite lines were within 0.003 (CHARTER §10, 00:20).
- The null-fault read was TOLERATED against both a same-class and a cross-class nominal for S1_perm, S1_2, S10_2, B1_25 and B4_0p5 (`results/null_fault/`).
- The pin was written at 01:27; DDC was 14.08 GB at the start (`ddc_start.json`).

**Grid** (2026-11-16 01:27 → 2026-11-17 01:37):
- `scripts/chain.sh` ran the phases in order: nominals → es1 → read → es2 → kind → c1 → closing. No queue stop fired after the smoke phase.
- **Relaunches.** 4 grid launches landed class-unmatched: S5_0p1 field_light es1, S3_0p1 field_light es2, A5_mid suite es2 and Z1_perm pits es2. Each was relaunched once and the relaunch matched (ledger). That is CHARTER §5.1's rule.
- **GPU sharing.** Two es2 gaps came from another session's eval rows on new plant variants. First, a 15.6-min wait (15:54 → 16:10) while the queue's busy-GPU check held row 47. Second, a requested ≈ 1-h pause: a ledger gap from 2026-11-16 16:29 to 17:25, between B2_0p2's suite and field_light es2 rows. The pause was requested so another session could use the GPU for eval rows on new plant variants; running two Kit processes against one DerivedDataCache risks corrupting it. The queue was stopped between rows with `results/STOP_QUEUE` (now `STOP_QUEUE.2026-11-16_pause_done`) and resumed with `PHASES="es2 kind c1 closing"`. The queue resumed at the next row; no row was lost or repeated.
- **Closing nominal.** Z (suite + field_light es1) landed in launch A's class with no check error.

**GPU by phase** (ledger; total **24.31 GPU-h** of 27):

| smoke | nominals | es1 | es2 | kind | c1 | closing |
|---|---|---|---|---|---|---|
| 1.68 | 0.86 | 9.00 | 7.80 | 1.68 | 3.10 | 0.19 |

The ledger holds 309 entries. 302 are launches with GPU time; the other 7 are bookkeeping entries for stamps that already existed. The 2 FSD diagnosis launches (CHARTER §10, 00:20) are not in the ledger. 302 + 2 = 304, the number `read.json` reads.

## 8. Caveats

- **Pairing amendments** (CHARTER §10, labelled 01:10, 01:40 and 02:45, all committed before the pin and the grid phases; see §7 on the five grid-counted rows launched during smoke).
  - Spontaneous GPU contact-solver drift (01:10).
  - Batch coupling through physics (01:40) and through the render/obs path (02:45).
  - So **a faulted episode is not paired one-to-one with its nominal twin after its wave's first onset.**
  - Grid totals†: 6770 batch-coupled envs over 238 grid rows (field_light 5034, suite 795, kind 941), 0 physics-drift envs and 0 identity breaks.
  - Coupled envs are dropped from the per-episode dose-pairing checks. In 3 rows fewer than 5 envs remained, so their dose pairing became report-only (A2_rear es1 fl, B1_25 es2 fl, B4_0p5 es2 fl); the sidecar dose probes still applied.
  - **The verdict statistics are unaffected,** because they are per-side group counts (F per wave/row, line medians, R shares).
  - **The leak controls passed.** All 5 armed-but-never rows were bitwise equal to their same-class nominal up to the latest grid onset. The per-(wave, env) inactivity probes were clean.
  - The 5 stray pre-onset falls in faulted, non-anchor field_light rows (B6_0p1, S1_0p5, B1_25 es1, A5_mid es1, c1 B4_0p5) all sit in rows with 67–99 coupled envs, and none enters F.
- **Process-state classes.**
  - 28 grid rows had no same-class nominal†. 16 are anchor rows (by design); the other 12 are 11 field_light rows plus A5_mid's es2 suite row.
  - They were read against both nominals, and the worse delta decided. At L090 only one nominal launch exists.
  - No extra nominal pair ran (`chain.sh` passes no `--extra`). The suite showed both classes at 0 kg, but field_light at 0 kg and L090 never did.
  - **Departure from CHARTER §4 / PLAN §8.5:** the approved extra nominal launches until both classes appear were not run, so field_light at 0 kg and every L090 row have a nominal in one class only. Impact is probably low (inference): every field_light nominal that ran had F 0, and every nominal reads suite falls 0 / 1000. The A5 L090 verdict rests on field_light, which had a same-class reference.
- **The A family is a sim plant with the force side not enforced.**
  - Rod slew is load-blind, and rated forces are never enforced (old REPORT:77, 110).
  - A1 locks the rod state, but the leg's joints still move (drift from onset p50 0.08–0.15 rad, max 0.92 rad at 0 kg and 1.00–1.04 rad for A5; `dose_report`). A rigid real jam may be harsher or milder; unknown.
  - A4's cam drifts ≤ 0.05 rad.
  - Only the left side was tested; mirror equivalence is not assumed (PLAN §3.3).
  - A2 models a broken linkage, not lost power: an unpowered rod is A1.
- **c1 is one seed and report only.** Its "(pending)" label is not missing data (§6).
- **Marginal cells that a ±2-episode change could move:**
  - S2_0p5 and S3_0p2 carry marginal flags but need ≥ 3 episodes on one seed to move, so they are stable within ±2.
  - T12: could read FALLS.
  - Z1_0p5: could read FALLS (es2 +2), never better than DEGRADES.
  - B1_10: worst case RECOVERS, so the B1 tolerated duration would not change.
  - Every other marginal flag sits on a TOLERATED with room (S5_2, B6_0p5), on a DEGRADES verdict decided by R or by failed lines, or on a FALLS with margin on the other seed (A1_front, A1_mid), or comes from T10b's n = 2.
- **Split cells:** A1_mid and Z1_0p5 (§3).
- **Stimulus choice.** S10 and Z1 inject raw 0 m, which the HAL's zero-fill produces and which normalises to −0.5. T12 injects the normalised zero. The deployed normaliser does not exist yet, so which stimulus the hardware will produce is open (PLAN §3.1). A depth fault reaches the policy 0.1–0.2 s after onset (CHARTER §3.3).
- **Exposure differs between rows.** The suite exposes 4–8 s after onset; field_light exposes 13–63 s. The S2_perm and S4 failures show mainly on field_light (§3 notes).
- **Scope.**
  - One fault at a time; Z1 is the only designed compound.
  - A5 is the only loaded cell.
  - Combinations with slow rods, payload or latency are untested.
  - Rows are the flat suite, the light field and two kind rows; there is no loose ground.
  - The sim runs at 50 Hz, while the HAL runs at 100 Hz with IMU/vxy at 30 Hz. Only the 30-Hz cadence was tested (S12).
- **Thin turn-facing margin.** Nominal per-hand facing is 0.86–0.94 at 0 kg against the 0.80 line (CHARTER §5.3). The turn values in §3–§4 are given beside their nominal.
- **The CHARTER §10 log ends at 03:10 on 2026-11-16 (step 6 complete).** The grid phases, the 4 relaunches and the pause are recorded only in the ledger.
- **Independent checks.** See the header's Verification line.

## 9. Decisions for you

1. **The second distill's riders.**
   - The candidates are the §2.1 list: Z1 compound transients, vxy dropout and hold-last, vxy bias and drift, the front blackout with the raw-0 stimulus, and contact_fill hold-last. Action latency is optional.
   - The cost is ≈ 9.6 h of distill plus ≈ 5.9 h of selection and G2 (rod-speed DR REPORT:703).
   - **Recommend:** adopt items 1–5. Leave out the TOLERATED channels. As the acceptance read, rerun S1–S4, S8_held, S10_perm and Z1 on the new student: ≈ 15 cells × 2 seeds × ≈ 0.19 GPU-h ≈ 5.7 GPU-h, an estimate from the ledger's per-row times.
2. **The vxy decision** (rod-speed DR decision 6, still open, REPORT:707).
   - Today a 0.5-s VO dropout fells 88 % of forward-walking episodes, and the student has no other velocity input.
   - **Recommend:** decide before the second distill whether riders are enough or an estimator head is needed. Riders alone may not remove the dependence (inference). Either way, item 4's health flag is needed.
3. **Host-link watchdog (HAL requirement).**
   - **Recommend:** trip in ≤ 0.1 s (5 policy ticks, 10 HAL ticks; inference from separate B4 and B6 cells) into B6's behaviour: rod targets held at their current length, cams at 0.
   - Never keep extrapolating open-loop PWM: that is B4, which FALLS at 0.5 s.
   - A hold longer than 0.5 s is untested, so go to a sit-down after it.
4. **Validity bits, deadlines and a safe stop (HAL requirement).**
   - **Recommend:** add validity/age bits for ZED grab success, VO tracking state and the Mega frame.
   - Response deadlines: VO loss ≤ 0.1 s; ZED grab failure < 0.5 s.
   - Specify the VO bias below 0.1 m/s.
   - Design a sit-down or safe stop. None exists, and it is out of this sweep's scope.
5. **The HAL observation contract (D1).**
   - **Recommend:** treat it as a hard prerequisite before any walking on hardware. Wire real joint position and velocity, and wire the omni observation mapping.
   - A contact zero-fill is tolerated (S8_zero), so current sense can come later.
6. **Latency budget.**
   - **Recommend:** a hardware spec of ≤ 20 ms end-to-end action latency, the trained one-step delay, with ≤ 40 ms as the hard limit (B5 falls nothing and fails no line). Add the U{1, 2, 3} delay rider only if the budget cannot be met.
7. **A trained fault regime (five legs, locked leg).**
   - **Recommend:** not now. c1 fails these faults in the same way, so any regime is a teacher-level campaign.
   - Hardware should first detect an actuator fault and sit down.
   - If a regime goes ahead, run the mirrored A1/A2 cells first (they were not run, by your 2026-11-15 decision).
8. **An optional follow-up eval batch** (needs new approval, new knob levels in code and a CHARTER amendment).
   - Candidate cells: S2 and Z1 at 0.1 / 0.2 s; B6 at 2 s and permanent; S5 permanent. Together ≈ 7 cells × 2 seeds ≈ 2.6 GPU-h (estimate).
   - These would size the VO and ZED deadlines and the hold-to-sit-down handover; today S2 and Z1 have no tolerated level.
   - A front-camera frozen-frame cell on the kind rows would settle the depth-use question in §5.
   - **Recommend:** run the S2, Z1 and B6 cells before the HAL spec is written. Run the front-freeze cell only if the depth question changes the distill.
9. **Record and merge.**
   - **Recommend:** commit this REPORT plus the results keep-set locally on `feat/fault-injection-sweep`, with a curated file list and never `git add -A`. The keep-set is `verdicts.md`, `read.json` (4.7 MB), `queue_ledger.jsonl` and `pin.json`; `null_fault/` and `smoke/` are already tracked.
   - Append a CHARTER §10 closing entry covering the grid, the relaunches and the pause.
   - Keeping or deleting the 7 smoke STOP files is your call.
   - A merge needs main's 1 new commit (06c8f8d1) merged in, then a Task 4 relock. Push and PR are yours.

**Decided 2026-11-18 (user):**
- 9.1: riders = items 1–5, as recommended; no latency rider.
- 9.2: riders first; an estimator head only if the acceptance rerun still falls.
- 9.8: run the follow-up batch's S2, Z1 and B6 cells (not S5 permanent, not the front-freeze cell). This needs new knob levels and a CHARTER amendment.
- 9.3–9.6: all four adopted as recommended (watchdog ≤ 0.1 s into a B6 hold, then sit-down; validity/age bits, VO ≤ 0.1 s and ZED < 0.5 s deadlines, VO bias < 0.1 m/s, a sit-down design; real joint pos/vel + omni obs mapping before any hardware walking; latency ≤ 20 ms, 40 ms hard limit). The 9.3/9.4 deadlines are to be resized by the 9.8 follow-up batch.
- 9.7 (no trained fault regime now) and 9.9 (recorded locally as ee0fd065; the merge is pending): unchanged.

**Follow-up batch results (2026-11-18, §9.8; CHARTER §10 amendment of 2026-11-18):**
- Run: 6 cells × {suite, field_light} × es1 + es2 = 24 launches, 2.39 GPU-h of the 2.6 cap, own pin (`results_followup/pin.json`, code_sha 9376f6b10e22), 0 launch errors, 0 check errors. References are the grid's nominals. 21 of 24 rows matched a same-class nominal bitwise before onset (so the knob-level code change is neutral); 3 rows (S2_0p1 es2 field_light, Z1_0p1 es1 field_light, Z1_0p2 es2 field_light) drew a launch class with no same-class nominal and were read against both nominals, as in the grid.

  | Cell | es1 | es2 | Cell verdict | Worst ΔF (es1 / es2) | Lines failed |
  |---|---|---|---|---|---|
  | S2 vxy held 0.1 s | TOLERATED | TOLERATED | TOLERATED | 0.00 / 0.00 | – |
  | S2 vxy held 0.2 s | TOLERATED | TOLERATED | TOLERATED | 0.00 / 0.00 | – |
  | Z1 ZED loss 0.1 s | TOLERATED | TOLERATED | TOLERATED | 0.00 / 0.01 | – |
  | Z1 ZED loss 0.2 s | TOLERATED | TOLERATED | TOLERATED | 0.00 / 0.00 | – |
  | B6 hold 2 s | RECOVERS | RECOVERS | RECOVERS | 0.04 / 0.05 | tracking (all six directions) |
  | B6 hold permanent | DEGRADES | DEGRADES | DEGRADES | 0.07 / 0.05 | tracking, turn facing; field_light progress R 0 |

- What it changes in 9.3 and 9.4 (inference from these cells; the HAL behaviours are modelled by the knobs, not measured on hardware):
  - **VO loss:** the policy tolerates a held last value for 0.2 s (S2), but a value forced to 0 only for 0.1 s (S1). So on VO loss the HAL should hold the last velocity and set the invalid bit; the 0.1 s deadline stays the target, with 0.2 s as the hard limit while holding. Never substitute 0.
  - **ZED grab failure:** tolerated for 0.2 s, DEGRADES at 0.5 s. Tighten the deadline from < 0.5 s to **≤ 0.2 s**.
  - **Watchdog hold → sit-down:** a position hold is survivable for 2 s (RECOVERS: few extra falls, walking stops tracking), and even a permanent hold adds only 0.05–0.07 falls but makes no progress. So the hold is a safe intermediate state: trip the watchdog in ≤ 0.1 s as decided, and hand over to the sit-down within **2 s**.
- Not run (as decided): S5 permanent and the front-freeze cell.


## 10. Files

- `PLAN.md` (§1 outcome; §8.1 hardware list at PLAN.md:335–341), `CHARTER.md` (§5.3 verdict rule; §10 amendments at CHARTER.md:258–304), `cells.yaml`.
- `results/verdicts.md` (Cells, Cell-seeds, Tolerated duration, Kind rows) and `results/read.json` (`cell_seeds[*].verdict_reads`, `rows.*.reads[*].waves`, `checks.*.identity` / `dose_report`, `tolerated_durations`, `classes`, `nominal_checks`, `smoke_checks`).
- `results_followup/` (the 2026-11-18 follow-up batch: `verdicts.md`, `read.json` over all launches, `queue_ledger.jsonl`, `pin.json`, `ddc_start.json`).
- `results/queue_ledger.jsonl` (spend, relaunches, the pause), `results/pin.json`, `results/null_fault/`, `results/smoke/`, `results/STOP.2026-11-16_*.json` (smoke stops).
- Re-read: `isaac_venv/bin/python scripts/fault_read.py --root /home/nickmagus/krabby/krabby-research/parkour/outputs/fault_sweep/lr` (CPU). The raw stamps (untracked) are in the main checkout under `parkour/outputs/fault_sweep/lr/`.
- Tie-ins: `../2026-11-06_dr_stress_sweep/REPORT.md` (:117 T07, :119 T12, :144 T10b), `../2026-11-10_rough_terrain/REPORT.md` (:33, :167), `../2026-11-07_rod_speed_dr/REPORT.md` (:703, :707).