# Report: height-commanded stand network for the omni crab (campaign `2026-10-13_stand_height`, 2026-10-13 → 2026-10-14)

**Status: COMPLETE 2026-10-14. Stopped for the user's adoption decision (PLAN step 8). Nothing is baked.**

- **Recommended head:** `SH_T17` (`runs/SH_T17/model_5000.pt`, sha256 `99c377a1…`).
- **Walking head of record (unchanged):** S1 s18, `policy/6_p2a_field/model_19994.pt`, `968cc550…`.
- **Stand network of record (unchanged):** A-mask s17, `policy/stand_net/model_4000.pt`, `769cf253…`.
- **Code:** all campaign-local (`scripts/`). code_sha and eval_sha were unchanged at every recorded check (C:L78, L178-179, L264, L294).
- **Decisions:** adoption, bake and push are the user's.

Citations: `C:L<n>` is a line in `CHARTER.md` (the record) and `P:L<n>` is a line in `PLAN.md`. Files without a path are in this campaign directory.

## 1. The question and the answer

**Question** (C:L15-19). A stand network is trained by DAgger of a height-aware scripted controller R_h, acting as a residual on a posture offset. Can it pass K1–K6 at three training seeds? K1–K6 cover:
- standing still and homed at commanded heights across the reachable, hip-force-capped range;
- raising and lowering between heights;
- standing up from retracted starts on tables and from the ground;
- restoring walking height before handing back to S1.

**Answer: yes, for table starts with tops of 0.635 m or higher. Not from the ground, and not from lower tables.** The final configuration has four parts: the posture map q(h), a hard 0.25 hip/knee residual deadband, the trim-branch DAgger weights, and the H6 restore. With it, each of the three seeds does the following at both env seeds:
- **Height tracking:** it stands at every height in [0.719, 1.346] m. The worst-bin median error is ≤ 2.6 mm and the p90 ≤ 6.4 mm, against lines of 20 / 40 mm.
- **Raise / lower:** it moves at the limiter's 0.08 m/s. p90 arrival is 1.16–1.17 s inside the allowance, with 0 falls.
- **Stand-up:** from knee-folded and random-fold starts on 0.635–0.762 m tables, STOOD is 0.977–0.992 against R_h's 0.983 / 0.978. Falls are ≤ 0.5 per 100.
- **Hand-back:** it restores 1.064 m before handing back. Exit falls are 0 and walking onset sits 1.3 s inside its line.
- **No regression at 1.064 m:** G1–G5 all pass.

**Verdict per head** (head_SH_T17/T16/T18.md):

| Head | Seed | K1 | K2 | K3 | K4 restore misses (es1 / es2; line 1) | K5 | K6 | Verdict |
|---|---|---|---|---|---|---|---|---|
| **SH_T17** | 17 | PASS | PASS | PASS | 0 / 1 | PASS | PASS | **Every clause passes at es1 and es2** |
| SH_T16 | 16 | PASS | PASS | PASS | 1 / **2** | PASS | PASS | All but a K4 es2 count |
| SH_T18 | 18 | PASS | PASS | PASS | 1 / **3** | PASS | PASS | All but a K4 es2 count; the user accepted it as chance-level (C:L326-327) |

**Every K4 miss is the same kind** (C:L321-325, L342-343):
- each is a restore timeout with the robot already at walking height (h 1.062–1.074 m);
- one hip rod sits 8.1–8.5 mm off against the 8-mm band, while a leg seats;
- R_h's own two misses look the same (8.2 mm).

One SH_T16 es2 case had a leg mid re-step (8.3–11.7 mm). The CHARTER puts 3 / 800 against 1 / 800 at a two-proportion p ≈ 0.3 (C:L324). The user's acceptance names SH_T18 only. SH_T16's 2 / 800 is classed as the same kind (C:L342), but whether that acceptance covers it is the user's call. Pooled over both env seeds, the configuration misses 8 / 4800 restores (0.17 %), against R_h's 2 / 1600 (0.13 %).

**Recommendation: adopt SH_T17.** This tie-break comes after the reads, so it is not pre-registered. Reasons:
1. It is the only head that passes every pre-registered clause as coded at both env seeds.
2. It learned the most of R_h's in-band cam trim:
   - train-time ratio 0.51 over its 3334–5000, rising to 0.59;
   - closed-loop HOLD slope −0.351, against R's −0.517 on the same rows;
   - the fewest HOLD rows off home (2.4 %).
3. It has the lowest K3 fall rate (0.17 / 0.33 per 100).

Caveats:
- The K4 differences between seeds (0/1, 1/2, 1/3 per 800) are count-level.
- SH_T17's stillness reads slightly higher than SH_T18's (steps 0.16 / 0.18 against 0.14 / 0.14 per s), though all are far below the 0.5 line.
- All three stand nets ride the same S1 s18.

**Not delivered:**
- **Ground get-up.**
- **Stand-up from tables below 0.635 m** (the 18-in / 0.457 m table).
- **rod_min starts.** These are run and reported, never gated.

The limit is mechanical and R_h has it too (C:L139-148):
- A loaded, folded knee cannot extend. The PD demand is ≈ 2500 N·m against the knee's 400 N·m effort cap, so on the ground the robot stays pinned 13 cm low.
- At the −45° hip stop, unfolding drops the toe by up to 12 cm, so low tops load the toes mid-unfold.
- On hardware the knee is hold-only and rated far below this load.

The user narrowed stand-up to tall tables at the step-4 stop (C:L150-171). Ground get-up needs its own scripted motion and a hip-force check (§10).

## 2. Final configuration (as trained and evaluated)

Sources: `runs/SH_T17/run.json`, `video_sh_raise_T17_sanity.json` (`controller_record`, `clip`), `facts.md`, C:L279-331.

**Action path** (stand-net rows):
- The stand action is `q(h_lim) + deadband(clamp(a_net, ±1), 0.25 on hip / knee)`.
- The deadband is hard: `r ← r if |r| ≥ 0.25 else 0`. It applies after the ±1 clamp and before adding q, on hip / knee columns 6–17 only, never on cams.
- Hip / knee rod targets are clamped to ±4.8 and cams to ±1.
- S1 rows are `clamp(S1, ±1)`.
- The harness `clip_actions` is raised to 4.8 at runtime only; the config value stays 1.0.

**Posture map q(h):**
- Campaign-local IK at constant foot reach (U0 0.3742 m), anchored at q(1.064) = (0, 0) exactly, and monotone.
- q(h_min) = (−2.748, +0.371) action units. Grid step 0.5 mm.

**Height command:**
- Absolute root height above the mean terrain under the six feet.
- Range [h_min, h_max] = **[0.719, 1.346] m** (frozen, `facts.json`).
- Rate limiter **0.08 m/s**. Its state starts at the command for walk-handover starts and at the measured height for table / random starts.
- Encoded in stand-net input **column 15 = (h_lim − 1.064) / 0.4**, written after the estimator.

**Restore before hand-back (H6):**
- Engages when the slot leaves stand with |h_lim − 1.064| > **2 cm**. The stand net keeps acting with h_cmd → 1.064 through the limiter, and command columns 5 and 8–10 are zeroed.
- **Complete** when the limiter is at 1.064 **and** every hip rod is within **8 mm** of rod(q(1.064)) for **0.2 s** (10 steps).
- **Timeout 5.3125 s** = (1.064 − 0.719) / 0.08 + 1 s. On timeout it hands back anyway, which counts as a K4 miss.

**Network:**
- A copy of S1's actor, mask 1 (A-mask input), warm-started from A-mask s17 with **W[:, 15] zeroed** (C:L204).
- Only the actor backbone trains. Frozen: scan / history / privileged encoders, critic, std, estimator.

**Training:**
- **DAgger of R_h** (D3 constants, `lift=-0.5,rho=1,…,trim=0.1,…`), with residual labels R_h − q(h_lim).
- β goes 1 → 0 linearly over the first third, per episode. A 10 % R_h-executed control slice runs throughout.
- DART σ is 0 / 0.1 / 0.3 at ½ / ¼ / ¼. LIFT-onset share 0.1.
- 4 GB reservoir. Weighted MSE on the unclipped actor mean, with per-leg inverse-branch-frequency weights (cap 20) (`scripts/sh_dagger_core.py`).
- **Plus the trim branch:** a HOLD leg whose R_h cam label is saturated (|label| ≥ 0.0999) counts as its own branch.
- Adam (`scripts/sh_dagger_core.py`) at lr 3e-4, 1024 envs, 24 steps / it, 16 × 4096 minibatches, **5000 its**, judged `model_5000.pt`.
- The deadband 0.25 is applied to student-executed stand rows only; labels and the R_h executor are untouched.
- Flags: `--mode dagger --arm mask --iterations 5000 --deadband 0.25 --trim-branch --trim-min 0.0999`.

**Starts and commands** (env Isaac-Crab-Hex-Omni-Teacher-v0):
- Start mix: walk handover **0.50**, table **0.40**, ground **0**, random height **0.10**.
- Tables: one kinematic cuboid per env, top **U[0.635, 0.805] m**. The first stand is ≥ top + 0.264 m and long enough for ramp + settle + 3 s. The table is parked after the first exit.
- Retracted poses: **knee_folded 0.75 / random fold 0.25 / rod_min 0**.
- h_cmd ~ U[h_min, h_max] with a 35 % atom at 1.064. Pushes are suppressed until the robot is up.

## 3. K1–K6 per head against the controls

Sources:
- Head columns: `head_SH_{P18,P18db,T18,T16,T17}.md/.json`.
- R_h and the known-bad columns: `step4b_read.md` (control table **14 / 14 cells land, 0 mismatches**, identical in every head read).
- R_h's table cells come from the step-4b stamp `sh_table/2026-10-13_21-59-02_full`.

Values are es1 / es2, and **bold** marks a failing value. K1 has 5 bins over [0.719, 1.346] with 100–200 holds per bin per seed.

| Gate · statistic | Line | R_h (known-good) | Known-bad (must fail) | SH_P18 pilot | SH_P18db probe | SH_T18 | SH_T16 | SH_T17 |
|---|---|---|---|---|---|---|---|---|
| K1 worst-bin median \|h − h_cmd\| (mm) | ≤ 20 | 3.0 / 2.5 | R, old ±1 path: **295 / 296**; A-mask s17, old path: **294 / 294** | 2.5 / 2.8 | 2.7 / 2.6 | 2.4 / 2.4 | 2.5 / 2.6 | 2.5 / 2.6 |
| K1 worst-bin p90 (mm) | ≤ 40 | 6.3 / 5.9 | **347 / 346**; **347 / 346** | 6.3 / 6.4 | 6.4 / 5.9 | 5.7 / 6.2 | 5.7 / 5.7 | 5.8 / 6.4 |
| K2 p90 arrival minus allowance (s), 800 events | ≤ 0 | −1.165 / −1.17 | both **never reached** | −1.165 / −1.16 | −1.175 / −1.175 | −1.17 / −1.165 | −1.17 / −1.16 | −1.165 / −1.17 |
| K2 stand falls | ≤ 1 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| K3 tables STOOD, 6 gated cells (n 600) | ≥ 0.933 / 0.928 | 0.983 / 0.978 | S1: **0 / 0** | **0.800 / 0.815** | 0.988 / 0.983 | 0.990 / 0.978 | 0.990 / 0.980 | 0.992 / 0.977 |
| K3 falls per 100 | ≤ 1 | 0.83 / 0.33 | S1: **3.67 / 3.17** | 0 / 0.33 | 0.67 / 0 | 0.50 / 0.33 | 0.50 / 0.33 | 0.17 / 0.33 |
| K4 restore misses (of 800) | ≤ max(1, R_h's) = 1 | 1 / 1 | no restore: **788 / 784** | 0 / 0 | 1 / 0 | 1 / **3** | 1 / **2** | 0 / 1 |
| K4 exit falls (900 hand-backs) | ≤ 1 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| K4 onset excess p90 (s) | ≤ 0 | −1.305 | −1.54 | −1.305 | −1.305 | −1.305 | −1.305 | −1.305 |
| K4 hand-back \|h − 1.064\| median (max), mm, reported | — | 1.6 (11.8) / 1.6 (9.2) | — | 1.9 (8.6) / 1.9 (7.1) | 1.6 (8.2) / 1.5 (8.0) | 1.7 (7.3) / 1.6 (18.7) | 1.6 (10.5) / 1.6 (9.3) | 1.5 (10.6) / 1.6 (7.7) |
| K5 G1–G5 at 1.064 (zero-command rows) | all PASS | PASS | none (A-mask s17 is a second known-good: PASS) | PASS | **G2 FAIL** | PASS | PASS | PASS |
| K6 lowest homed share in any bin, reported | — (lines are per bin, next row) | 0.93 / 0.955 | posture only: **0 / 0** | 0.955 / 0.96 | **0.655 / 0.735** | 0.98 / 0.99 | 0.99 / 0.98 | 0.98 / 1.00 |
| K6 tightest margin to a bin's line (that bin's homed share) | ≥ 0 in every bin; line = R_h's − max(5 pp, 2 SE) | — (sets the lines) | posture only: **−0.95 / −0.95** | +0.010 (0.955) / +0.020 (0.97) | **−0.27 (0.655) / −0.15 (0.735)** | +0.04 (0.99) / +0.05 (0.99) | +0.04 (0.99) / +0.045 (0.995) | +0.04 (0.99) / +0.05 (1.00) |
| K6 stillness steps / s, pooled | ≤ 0.5 | 0.165 / 0.163 | posture: 0 / 0.20 | **0.59 / 0.61** | 0.14 / 0.13 | 0.14 / 0.14 | 0.16 / 0.14 | 0.16 / 0.18 |
| K6 stillness taps / s, pooled | ≤ 1.0 | 0.147 / 0.147 | posture: 0 / 0.30 | **2.85 / 2.78** | 0.13 / 0.12 | 0.16 / 0.16 | 0.19 / 0.17 | 0.17 / 0.16 |
| **Verdict** | | PASS | each fails its gate | FAIL K3, K6 | FAIL K5, K6 | FAIL K4 (es2 count) | FAIL K4 (es2 count) | **PASS all** |

Notes:
- **The two K6 homing rows are different statistics.** The lowest share and the tightest margin can sit in different bins, because each bin's line follows R_h's share in that bin (`head_SH_*.json` K6 clauses). Example: SH_T17 es1's lowest share is 0.98 (bins 1 and 2, margins +0.108 / +0.050), but its tightest margin is +0.04 at bin 3 (0.99).
- **Reported-only controls:**
  - posture-only passes K1, K2 and K3;
  - A-mask s17 through the new q path fails K1 (worst median 87–88 mm), K2, K3 (0.13 / 0.16 STOOD) and K6 (C:L130-131, `step4b_read.md`).
- **Every head read exits 4.** That flag is R_h's own floor at es1 on the 0.635 top, where 4 of 200 robots had hip-contact events. It was accepted under the pinned rule (C:L180-184, L190-194); it is not a head failure.

**K3 per cell, STOOD share (falls per 100), es1 / es2** (`head_SH_*.json` K3 `per_cell`). "Gated" cells decide K3; the rest are run and reported but never gated.

| Cell | R_h | SH_T17 | SH_P18 (no deadband) | SH_P18db |
|---|---|---|---|---|
| 0.635 knee_folded (gated) | 0.98 (0) / 0.95 (0) | 0.98 (0) / 0.95 (0) | 0.77 / 0.76 | 1.00 / 0.95 |
| 0.635 random (gated) | 0.95 (4) / 0.97 (2) | 0.98 (1) / 0.98 (2) | 0.76 / 0.76 | 0.97 / 0.99 |
| 0.6985 knee_folded / random (gated) | 1.00, 0.98 (1) / 1.00, 0.97 | 1.00, 0.99 / 1.00, 0.95 | 0.78, 0.84 / 0.79, 0.84 | 1.00, 0.97 / 1.00, 0.97 |
| 0.762 knee_folded / random (gated) | 1.00, 0.99 / 1.00, 0.98 | 1.00, 1.00 / 1.00, 0.98 | 0.79, 0.86 / 0.87, 0.87 | 1.00, 0.99 / 1.00, 0.99 |
| 0.457 knee_folded (reported) | 0.04 (0) / 0.09 (0) | 0.05 (1) / 0.11 (0) | **0.97 (0) / 0.98 (0)** | 0.18 (0) / 0.32 (0) |
| 0.457 random / rod_min (reported) | 0.55 (11), 0.42 (8) / 0.51 (13), 0.33 (17) | 0.63 (9), 0.57 (7) / 0.68 (8), 0.53 (8) | 0.68 (8), 0.60 (12) / 0.72 (9), 0.57 (14) | 0.74 (11), 0.51 (17) / 0.64 (13), 0.52 (13) |
| 0.635 rod_min (reported) | 0.80 (15) / 0.89 (10) | 0.83 (13) / 0.91 (5) | 0.66 (8) / 0.72 (3) | 0.79 (18) / 0.95 (2) |
| ground knee_folded (reported) | 0.00 (32) / 0.00 (21) | 0.00 (35) / 0.00 (28) | **0.49 (28) / 0.62 (15)** | 0.05 (46) / 0.11 (25) |
| ground random / rod_min (reported) | 0.61, 0.20 / 0.68, 0.18 | 0.63, 0.22 / 0.70, 0.20 | 0.69, 0.27 / 0.75, 0.29 | 0.61, 0.22 / 0.76, 0.18 |

The two bold pilot cells are an unexplained lead, discussed in §10.

## 4. The path: decisions and why

1. **Step 1: force cap from foot-load statics** (C:L46-70, `facts.md`).
   - h_min = 0.719 m. The binding term is hip room (≥ 0.125 rad to the −45° stop).
   - The static model missed the measured joint torques by −32 % (hip) and −39 % (knee). The cap then fell below walking height: scaled h_max 1.02 m < 1.10. That was a STOP.
   - Why they disagree: τ_hip − τ_knee = 246.8 N·m on every stamp, while the measured foot loads imply ≈ 216 N·m. That is an unexplained ≈ 14 % excess in the sim's PD-torque reading, and it lands on F_hip.
   - **User decision:** cap from the measured-foot-load statics, giving **h_max 1.346 m**. The PD reading is reported beside every height. The excess stays open.

2. **Restore band 4 → 8 mm** (C:L104-119). This was pinned before any K4 read.
   - At 4 mm, 119 / 600 (hwalkstop) and 36 / 200 (hraise) R_h restores timed out. Every one was settled in HOLD with one loaded leg at 4.1–4.9 mm of hip-rod PD sag, not moving.
   - R's own settled walking-height stands have p99 deviations of 7.16 / 6.75 mm.
   - Rule: δ = the smallest whole mm ≥ R's p99, so 8 mm. An offline replay then gives 600 / 600 and 200 / 200 completions.

3. **Identity bridge: bitwise** (C:L120-123). This ran on the 4.8 / S1-clamp path with the restore armed, so K5 could reuse the existing R and A-mask stamps.

4. **Step-4 STOP: R_h cannot stand up from the ground or low tables** (C:L124-149, `step4_read.md`).
   - The control table landed 14 / 14 and h_max held (1912 N).
   - **R_h's own stand-up floor missed:**
     - 18-in table: STOOD 98 / 300 and 87 / 300;
     - 25-in table: 93 / 97 % STOOD, but 9 / 8 falls per 300;
     - ground knee_folded: 0 / 100;
     - ground rod_min: 17 / 21 %.
   - Traces show the hips track their targets, but a **loaded folded knee does not extend.** On the ground it stays 0.42 rad (≈ 16 mm of knee rod) short for all 16 s, against a PD demand ≈ 2500 N·m and a **400 N·m effort cap**, with the robot pinned 13 cm low.
   - Geometry: at the −45° hip stop an unfold drops the toe by up to 12 cm, so any top below ≈ 0.53 m loads the toes mid-unfold. From the ground, no hip angle lets the legs unfold in the air.
   - **User re-scope:** tall tables only (C:L150-171).
     - Lowest top 0.635 m.
     - Gated cells: tops {0.635, 0.6985, 0.762} × poses {knee_folded, random}.
     - Reported only: the 18-in table, rod_min cells and ground cells.
     - Training starts changed to walk 0.50 / table 0.40 / ground 0 / random 0.10.
   - **R_h's floor events at 0.635 were accepted** (C:L163-168). All five were hip-contact (> 500 N) with the robot upright (tilt 2.5–3.3°) and still standing at the end. They are hip-body contacts, not tip-overs. The student is held to the same contacts through its max(1, R_h's) fall line.

5. **Step 4b re-calibration** (C:L185-196, `step4b_read.md`).
   - Controls landed 14 / 14. R_h stood 590 / 600 and 587 / 600.
   - On the 0.635 es1 top, 193 / 200 stood with 4 hip-contact-only events: tilt 2.6–4.3°, peak hip force 568–11 932 N. Read as coded that is a MISS; it was accepted under the floor-read rule pinned beforehand.

6. **Smoke** (C:L197-204). Passed every check, at 1.80 s / it.

7. **Pilot SH_P18: K3 and K6 fail on sub-mm load flicker** (C:L212-242).
   - K3 STOOD was 0.800 / 0.815. The misses come almost entirely from the four-feet clause: a median 87 % of the window had ≥ 4 feet loaded, against the 95 % line.
   - K6 stillness: 0.59 / 0.61 steps and 2.85 / 2.78 taps per s.
   - On legs R_h holds (label exactly 0), the student's hip / knee residual is p50 0.010 and p99 0.147 units, ≈ 0.3–4 mm of rod.
   - With hip PD at 16 000 N·m/rad, a 1-mm rod error on one stance leg moves enough load to cross the 50-N loaded threshold. That one mechanism explains both failures.
   - Train-time table STOOD plateaued at 0.77 from it ≈ 3300, so more iterations were unlikely to help.

8. **Deadband probe: fixes K3 / K6 stillness, exposes the missing trim** (C:L243-278).
   - The 0.25 midpoint was set from data: R_h's hip / knee labels are strictly 0 or 0.5, with none in between across 3.6 M entries.
   - With the pilot checkpoint at deadband 0.25, K3 STOOD rose to 0.988 / 0.983 and stillness fell to 0.14 / 0.13 steps per s.
   - But **K6 homing (0.655–0.97 against lines 0.83–0.95) and K5 G2 failed.**
     - 47 / 420 holds ended with a cam 0.1–0.3 rad off home (R_h: 4 / 420).
     - There, R_h trims with a median of −0.100 units; the student commands −0.002, about 2 % of that.
     - The pilot had homed through its small re-steps. The deadband removed those re-steps and exposed the trim the student never learned.

9. **Trim branch + deadband in the training loop** (C:L279-325).
   - The rule was corrected before any run (C:L293-299). "Cam label non-zero" would have moved ≈ 99 % of HOLD legs into the trim branch and inverted the weights. The user's choice was the **saturated** label (|label| ≥ 0.0999 ⇔ |c| ≥ 0.1 rad), which gives trim weights of 5.1–15.5 against HOLD's 0.19.
   - SH_T18 smoke: trim legs were 7.7 % of the reservoir at a mean weight of 2.16.
   - Mid-run, table STOOD was 0.917 against R_h's 0.931 (pilot: 0.673).
   - **The read passed every gate except K4, which missed on one count clause: es2 had 3 / 800 restore misses against a line of 1.** That was a STOP.

10. **The user accepted the count miss, and the replicates were run** (C:L326-349). SH_T17 passes all gates. SH_T16 misses K4 es2 at 2 / 800, of the same kind.

## 5. Forces per height bin

Source: `head_SH_T17.md` "Forces per height bin". It covers settled stand rows with ≥ 5 feet loaded, es1 + es2 pooled. F_hip is the six-leg mean |F_hip| from the **measured foot loads through the linkage**, which is the cap statistic. "PD" is the sim PD-torque reading converted the same way. "p90" is the p90 over rows of the six-leg mean, and the worst leg is the highest per-leg mean. Knee values are foot-load / PD.

| Bin (m) | R_h rows | R_h mean h | R_h F_hip | p90 | PD | Worst leg | Knee | SH_T17 rows | SH_T17 mean h | SH_T17 F_hip | p90 | PD | Worst leg | Knee |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.719–0.844 | 71 117 | 0.777 | 1140 | 1243 | 1515 | ML 1336 | 1336 / 1323 | 80 186 | 0.775 | 1136 | 1248 | 1463 | MR 1209 | 1364 / 1522 |
| 0.844–0.970 | 65 297 | 0.906 | 1394 | 1507 | 1783 | ML 1518 | 1482 / 1288 | 68 794 | 0.907 | 1398 | 1506 | 1780 | ML 1496 | 1480 / 1286 |
| 0.970–1.095 | 50 606 | 1.063 | 1609 | 1674 | 2082 | ML 1813 | 1372 / 1009 | 50 470 | 1.063 | 1615 | 1674 | 2066 | ML 1800 | 1388 / 1124 |
| 1.095–1.221 | 66 669 | 1.149 | 1738 | 1803 | 2233 | ML 1903 | 1281 / 1057 | 66 791 | 1.149 | 1743 | 1809 | 2228 | ML 1804 | 1291 / 1082 |
| 1.221–1.346 | 122 689 | 1.279 | **1912** | 2018 | 2457 | MR 2068 | 1107 / 1125 | 122 160 | 1.283 | 1918 | 2040 | 2467 | MR 2065 | 1102 / 1120 |
| outside [h_min, h_max] | 39 948 | 1.295 | 1907 | 2050 | 2484 | MR 2232 | 1024 / 1167 | 43 155 | 1.316 | 1933 | 2041 | 2517 | MR 2095 | 1016 / 1158 |

- **h_max re-check (pre-registered statistic): R_h's top bin is 1912 N ≤ 2000 N**, so there is no re-freeze. Reported beside it: rows within 2 cm of 1.346 m read **1994 N** (PD 2582 N, 65 948 rows).
- SH_T18 and SH_T16 top bins read 1912 / 1919 N.
- **Margins at the top of the range are thin.** The six-leg mean stays under 2000 N in every bin, but in the top bin the p90 row (2018–2040 N) and the worst leg's mean (2065–2068 N) are above it.
- **The PD reading runs 27–33 % above the foot-load statistic in every bin.** This is the ≈ 14 % τ_hip − τ_knee excess from step 1, landing entirely on the hip rod. Read by PD, the cap would bind below walking height: step 1's scaled h_max was 1.02 m.
- **Knee rod force is 1.0–1.5 kN in every bin**, against a 500 N rating (P:L53-55). The knee is treated as hold-only, so this is reported, not capped.

## 6. Train-time premise reads and the trim

Sources: `pilot_premise_{mid,final}.json`, `t18_ / t16_ / t17_premise_{mid,final}.json`. Values are student / concurrent R_h slice. No STOP line was crossed in any run (lines: < 0.5× R_h on homed / STOOD; falls or restore timeouts > 2× R_h + 1 %).

The trim ratio is the student's executed cam command toward home on loaded in-band HOLD legs (|c| ∈ (0.1, 0.3]), divided by R_h's saturated 0.100. It is given pooled over the read window, then per 500-it window.

| Run | Its | Table STOOD | Walk homed | Random homed | Falls | σ-0 restore timeouts | Trim ratio: pooled; by window |
|---|---|---|---|---|---|---|---|
| SH_P18 | 1667–2500 | **0.673** / 0.930 | 0.954 / 0.954 | 0.960 / 0.987 | 0.65 / 0.62 % | 0.03 / 0.28 % | not recorded (≈ 0.02 closed-loop, C:L272-275) |
| SH_P18 | 3334–5000 | **0.769** / 0.923 | 0.968 / 0.950 | 0.985 / 0.978 | 0.50 / 0.60 % | 0.01 / 0.13 % | — |
| SH_T18 | 1667–2500 | 0.917 / 0.931 | 0.974 / 0.962 | 0.980 / 0.985 | 0.61 / 0.58 % | 0.20 / 0.28 % | 0.24; 0.22 → 0.25 |
| SH_T18 | 3334–5000 | 0.931 / 0.930 | 0.968 / 0.955 | 0.980 / 0.972 | 0.66 / 0.70 % | 0.15 / 0.13 % | 0.34; 0.30 → 0.33 → 0.35 → **0.37** |
| SH_T16 | 1667–2500 | 0.931 / 0.932 | 0.964 / 0.951 | 0.983 / 0.987 | 0.60 / 0.33 % | 0.11 / 0.28 % | 0.28; 0.25 → 0.30 |
| SH_T16 | 3334–5000 | 0.935 / 0.937 | 0.967 / 0.942 | 0.983 / 0.970 | 0.56 / 0.75 % | 0.19 / 0.27 % | **0.43**; 0.41 → 0.41 → 0.43 → 0.45 |
| SH_T17 | 1667–2500 | 0.940 / 0.926 | 0.969 / 0.945 | 0.989 / 0.972 | 0.42 / 0.62 % | 0.21 / 0.13 % | 0.26; 0.25 → 0.27 |
| SH_T17 | 3334–5000 | 0.933 / 0.931 | 0.970 / 0.956 | 0.984 / 0.966 | 0.47 / 0.64 % | 0.19 / 0.06 % | **0.51**; 0.41 → 0.47 → 0.52 → 0.59 |

- **Two clarifications on the trim ratios quoted in C:L336-337.**
  - The CHARTER figures 0.37 / 0.43 / 0.51 (T18 / T16 / T17) mix two statistics. T18's 0.37 is its last 500-it window; T16's and T17's are pooled over 3334–5000.
  - On one statistic: pooled 0.34 / 0.43 / 0.51, last window 0.37 / 0.45 / 0.59. The order T17 > T16 > T18 holds on both, and all three were still rising at 5000.
- **The trim branch closed the pilot's table gap.** Train-time table STOOD now matches R_h, where the pilot plateaued at 0.77. The eval table test agrees (§3).

## 7. Observe pass (closed loop, reported, not gated)

Sources: `observe_SH_{P18,T18,T16,T17}_{height,zero}.md`. "Height" is hwalkstop + hraise and "zero" is walkstop + suite. Applied = the executed residual. R = R_h minus the same posture.

| Statistic | SH_P18 (no deadband) | SH_T18 | SH_T16 | SH_T17 | R / R_h |
|---|---|---|---|---|---|
| HOLD \|applied − R\| hip, mean / p90 (height) | 0.015 / 0.026 | **0.000 / 0.000** | **0.000 / 0.000** | **0.000 / 0.000** | 0 |
| HOLD \|applied − R\| cam, mean / p90 (height) | 0.015 / 0.033 | 0.009 / 0.016 | 0.008 / 0.017 | 0.007 / 0.014 | 0 |
| μ_hip on LIFT rows, height / zero | −0.479 / −0.477 | −0.481 / −0.476 | −0.483 / −0.481 | −0.473 / −0.475 | −0.5 |
| LIFT onsets launched ≤ 0.3 s, height / zero | 97.8 / 98.7 % | 97.5 / 98.6 % | 98.1 / 98.4 % | 98.5 / 98.2 % | — |
| HOLD cam-trim within-leg slope, height (R's on the same rows) | −0.023 (−0.568) | −0.258 (−0.535) | −0.269 (−0.534) | **−0.351 (−0.517)** | — |
| Same slope, zero-command rows | −0.014 (−0.515) | −0.228 (−0.527) | −0.303 (−0.517) | −0.330 (−0.508) | — |
| Slope on the unsaturated band [0.03, 0.1], height | −0.036 | −0.461 | −0.524 | −0.726 | −1 |
| HOLD rows off home (\|c\| ∈ [0.03, 0.3]), height / zero | 12.0 / 5.5 % | 3.9 / 2.8 % | 4.1 / 2.1 % | 2.4 / 1.9 % | — |

- **The deadband zeroes HOLD residuals.** On HOLD rows the executed hip / knee residual equals R_h's exactly (0.000) on every trained head, removing the flicker source the pilot had.
- **The LIFT onset is learned at every height.** μ_hip is −0.473 to −0.483 against −0.5, and 97.5–98.5 % of onsets launch within 0.3 s.
- **The trim slope moved from 4 % of R's to 48–68 %.**
  - Height rows: −0.258 / −0.269 / −0.351 against R's −0.52 to −0.54. The pilot reached −0.023, and the predecessor's A-mask s17 −0.028 at 1.064 m.
  - Per leg, SH_T17 ranges −0.21 to −0.43, against R's −0.48 to −0.54 (C:L353).
  - The trim is still under-fit, but it now carries the homing that re-steps used to do.

## 8. Videos

1. **Raise / lower, SH_T17**, recorded by the streaming recorder.
   - File: `parkour/logs/videos/sh_stand_height/sh_raise_T17_sequence.mp4`, 154.5 s. Sanity record: `video_sh_raise_T17_sanity.json`. Stills: `…/sh_stand_height/stills/`.
   - Setup: S1 s18 + SH_T17 at deadband 0.25, flat ground, env seed 1.
   - Sequence: walk forward, then stands at 1.064 → 1.30 → 0.75 → 1.15 → 0.85 → 1.064, walk left, 1.346 → 0.719 → 1.25, walk back, 0.80 → 1.20, walk right, 0.90 → 1.064, walk forward.
   - Result:
     - all **13 / 13 heights arrived**, with window median error 0.4–4.4 mm;
     - 13 / 13 were homed at the end, with 0 falls;
     - both restores completed: 1.25 → walking in 2.52 s, 1.20 → walking in 1.88 s;
     - 0 route or execution mismatches over 7725 steps, and the reference checks against the hraise es1 stamp all match.
   - The slowest homing is the 0.80 m hold (cam settle 4.64 s, `height_summary.per_stand`).
2. **Recorder smoke, SH_T18**: `…/sh_stand_height/sh_smoke_T18_sequence.mp4`, 20.5 s, sanity `video_sh_smoke_T18_sanity.json`. Stands at 1.20 and 1.00 m (errors 1.7 / 1.5 mm), and one restore completed in 0.98 s.
3. **Table get-ups**, one env each at es1, recorded by the table test.
   - SH_T17: `parkour/logs/videos/sh_table/2026-10-14_20-06-28_full/video_stand_net_{table_0p635__knee_folded,table_0p6985__random,table_0p762__knee_folded}.mp4`.
   - SH_T16: `…/2026-10-14_16-39-16_full/`. SH_T18: `…/2026-10-14_13-01-29_full/`.
   - All three stand. Commands 1.1455 / 1.177 / 1.209 m, window error 1.3–1.7 mm, lift-off 1.44 / 1.50 / 2.48 s, max tilt ≤ 0.018 rad, cams homed (`*_summary.json`).
4. **Ground-limit evidence, R_h at step 4**, in `sh_table/2026-10-13_20-49-59_full/`:
   - `video_r_h_ground__knee_folded.mp4`: stays 0.158 m from its 1.0325 m command (window median) without falling. The loaded knees never extend.
   - `video_r_h_ground__rod_min.mp4`: 0.187 m off, tilt 0.12 rad.
   - `video_r_h_table_0p635__knee_folded.mp4`: stands, 1.7 mm.
   - The same stamp's `video_stand_net_*` files show A-mask s17 through the q path on these cells: ground rod_min tips over (tilt 3.14 rad), and the 0.635 table start does not stand.

## 9. Deviations from PLAN.md (each logged in the CHARTER)

1. **h_max statistic:** measured-foot-load statics instead of the joint-torque conversion of P:L76-79 (user, C:L64-70). The step-4 re-check was restated to the same statistic.
2. **Restore band δ:** the PLAN's 2-mm example became a 4-mm pin at build (C:L80-81), then **8 mm** at the step-4 smoke (C:L104-119). The 4-mm stamps were re-run.
3. **Implementation statistics pinned before their reads** (C:L79-100):
   - the STOOD rise uses the root rise, not the lowest corner;
   - off-height estimator inputs are accepted and reported;
   - the step-5 restore assertion gates only the R_h slices (P:L176 asked for 0 timeouts outright);
   - K4 completion comes from launcher events;
   - K5's field_turn row is on the shared manifest.
4. **Stand-up scope narrowed to tall tables** (user, C:L150-171):
   - ground starts removed from training and from K3;
   - gated tops changed to {0.635, 0.6985, 0.762};
   - the 18-in table, rod_min and ground are reported only;
   - start shares changed from 0.50 / 0.25 / 0.15 / 0.10 (P:L91-93) to 0.50 / 0.40 / 0 / 0.10;
   - poses changed from 0.60 / 0.20 / 0.20 to 0.75 / 0.25 / 0 (C:L160-162);
   - training tops became U[0.635, 0.805].
5. **R_h floor read rule:** hip-contact-only events with the robot upright are accepted (C:L163-168, L180-184). This applied at es1 on the 0.635 top in step 4b (C:L190-194).
6. **Action path:** a hard 0.25 hip / knee residual deadband was added to H1 (user, after the step-6 stop). It was probed at eval first (C:L243-255), then trained in (C:L279-292).
7. **Objective:** the trim branch (`--trim-min 0.0999`) was added to H4's branch weights. Its rule was corrected to the saturated label before any run (C:L293-303).
8. **Seeds:** four 5000-it runs instead of three. The deadband-free pilot SH_P18 is not an adoption seed, and s18 was retrained as SH_T18 under the final configuration (C:L252-255, L285-292).
9. **Gate verdict:** SH_T18's K4 es2 count miss was accepted by the user, and the replicates ran anyway (C:L326-331). SH_T16's 2 / 800 is classed as the same kind (C:L342-344). So the adoption-grade rule "all gates, three seeds" (P:L111-112) is met to the letter only by SH_T17.
10. **Premise timing:** the SH_T16 / SH_T17 mid-run premise reads were made after the runs, because the usage limit paused monitoring but not the queue (C:L332-333). No line was crossed.
11. **Videos:** the recorder was deferred to step 8 (C:L101, L345-349). There is no ground get-up video of a trained head; ground is out of scope, and the R_h step-4 ground videos stand in.
12. **Budget:** ≈ 18.8 GPU-h against ≈ 15 (§11).

## 10. Open items and follow-ups

- **Ground get-up** (C:L144-148, L150-152). This needs a scripted get-up motion, for example belly down, toes slid out under light load, then a hip push-up, plus a hip-force check against the 2000 N cap.
  - **Unexplained lead:** the deadband-free pilot SH_P18 stood **97 / 98 %** on the 18-in table from knee_folded (window error 4.2 / 4.9 mm, 0 falls) and **49 / 62 %** from ground knee_folded (28 / 15 falls per 100). R_h stands 4–9 % and 0 % on those cells.
  - The same checkpoint at deadband 0.25 drops to 18 / 32 % and 5 / 11 %, so the sub-0.25 hip / knee residual does it.
  - Its traces and hip forces have not been read.
- **Deployment integration, if adopted.** The `stand_network:` manifest block, `bundle_policy.py` and `test_policy_of_record_omni.py` would need to carry:
  - the checkpoint and its sha;
  - the column-15 height channel written after the estimator, with command columns 5 and 8–10 zeroed on restore rows;
  - the q(h) map (sh_kin, 0.5 mm grid);
  - the 0.08 m/s limiter clamped to [0.719, 1.346];
  - the H6 restore (2 cm / 8 mm / 0.2 s / 5.3125 s);
  - the 0.25 deadband, ±1 residual clamp, rods ±4.8 and cams ±1;
  - the 4.8 harness clip with S1 rows clamped ±1.

  Three things are not built yet:
  - **explicit stand-command keying**, the predecessor's latched-slot finding (P:L109);
  - **a height-command source**: here commands came only from the eval schedules (`scenarios_sh.yaml`) and the trainer's draws;
  - **a play path for the routed pair outside this campaign**: the recorder is campaign-local.
- **PD-torque excess:** the unexplained ≈ 14 % excess in τ_hip − τ_knee is still open. It reads 27–33 % on F_hip per bin and needs a Kit-side check (facts.md (b)).
- **Hip contacts at 0.635:** hip-contact events with the robot upright on the 25-in table. Their source is not identified (C:L163-168, L192).
- **Reported cells are weak:**
  - 18-in table: SH_T17 STOOD 0.05–0.11 from knee_folded.
  - rod_min at 0.635: 0.83 / 0.91, with 13 / 5 falls per 100 (R_h 15 / 10).
  - ground: 0.00–0.70 by pose.
- **Hardware:**
  - The knee is hold-only. Its rod force is 1.0–1.5 kN in every bin against a 500 N rating.
  - The posture map moves the knee rod from +0.371 to −0.344 action units across [0.719, 1.346] m (`sh_kin.q`; the `facts.md` (a) table lists −0.349 at 1.350 m, just above h_max). Whether a hold-only knee can do that under load is open.
  - Hip-rod margins at the top bin are thin (§5).
- **Trim:** the trim is learned only to 0.34–0.51× R_h's on train-time in-band legs, and was still rising at 5000 its. The closed-loop slope is 0.48–0.68× R's. Options: a longer run, a smoothed |c|-band weighting (lit-review-stand-network §12 rec 3), or a PPO fine-tune (P:L209).
- **Eval coverage:**
  - Eval stands are push-free. Pushed-regime evidence is train-time only: student fall share 0.42–0.66 % against 0.33–0.75 % for the R_h slice (§6).
  - The frozen estimator sees out-of-distribution inputs away from 1.064 m. This is accepted, not gated (C:L85-88).

## 11. Budget

**Total ≈ 18.8 GPU-h against the plan's ≈ 15** (P:L196-199).

How this was tallied: wall-clock between the START / END and per-job `rc` timestamps in:
- `outputs/omni_plan/stand_height/{q4a,q4b,q6,qdb,qt,qr}.log`;
- `sh_dagger.log`, for SH_SMOKE;
- the step-4b table-test `queue.log` (`sh_table/2026-10-13_21-59-02_full`);
- the two video logs.

The campaign's queues ran one Isaac process at a time, so wall-clock equals GPU occupancy, Kit start-up included, with one exception. `sh_rows.log` L131 logs `SHROWS WAIT another Isaac process running` at 2026-10-14T21:08:29, inside the SH_T17 K5 rows (SH_T17z). The es4 field_hard row then took 6 min 28 s, against 3 min 45–52 s for the other three field_hard rows. So ≈ 0.04 h of SH_T17's 1.14 GPU-h is waiting, not eval work (≈ 1.10 GPU-h of evals). The wait was a false-positive busy check, not a second Kit process: at that moment I was running the CPU-only reader `sh_read.py` for SH_T16 by relative path from `scripts/`, and the rows script's busy pattern exempts the campaign's CPU readers only when they are launched by absolute path (the same gap the video review fixed in `sh_record_video.sh`; see §10). The total is unchanged at the stated precision. CPU reads are excluded. Training intervals agree with each run's `status.json` `wall_s` to within 1 %.

| Stage | GPU-h | Source |
|---|---|---|
| Step 4: launcher + table smoke | 0.34 | q4a.log |
| Step 4: identity bridge, control height rows, K3 calibration | 2.69 | q4b.log |
| Step 4b: K3 re-calibration | 0.33 | sh_table/…21-59-02_full/queue.log |
| Smokes: SH_SMOKE, SH_T18_SMOKE | 0.11 | sh_dagger.log, qt.log |
| Training: SH_P18 / SH_T18 / SH_T16 / SH_T17 | 2.44 / 2.46 / 2.40 / 2.36 | q6 / qt / qr logs; `status.json` 1.75 / 1.77 / 1.72 / 1.69 s / it |
| Head evals (height rows, K5 rows, table test): P18 / T18 / T16 / T17 | 1.12 / 1.12 / 1.10 / 1.14 (T17 includes the ≈ 0.04-h wait) | q6 / qt / qr logs; `sh_rows.log` |
| Deadband probe evals (SH_P18db) | 1.12 | qdb.log |
| Step-8 videos | 0.14 | sh_stand_height/*.log |
| **Total** | **≈ 18.8** | |

Against the plan's lines:
- probe + calibration: 3.36 vs 2.5;
- smoke: 0.11 vs 0.3;
- training: 9.65 vs 7.5;
- evals: 4.48 + the 1.12 probe vs 4.5;
- videos: 0.14, unbudgeted.

The ≈ 3.8 GPU-h overrun is the path through step 6: the superseded deadband-free pilot cost 3.56 GPU-h (training + evals) and the probe 1.12. Head evals came in cheaper than planned, at ≈ 1.1 GPU-h per head against ≈ 1.5.

C:L310 gives SH_T18 as 1.73 s / it, but that is the smoke's figure; its `status.json` reads 1.765 s / it.

## 12. Options for the user and pointers

**Options** (this report takes none):
1. **Adopt SH_T17 as the stand network of record**, as a replacement for A-mask s17 (K5 shows no regression at 1.064 at deadband 0.25) or beside it. The §10 deployment work comes with it.
2. **Adopt a different seed:** SH_T18 (its K4 es2 count miss is user-accepted) or SH_T16 (same kind, but not yet accepted).
3. **Keep A-mask s17** (1.064 m only) and open the follow-ups first: ground get-up, the trim, and the PD excess.

**Checkpoints** (under `parkour/outputs/omni_plan/stand_height/runs/`, untracked):

| Head | Path | sha256 | Config | Gates |
|---|---|---|---|---|
| **SH_T17** | `SH_T17/model_5000.pt` | `99c377a1…` | final | PASS all |
| SH_T16 | `SH_T16/model_5000.pt` | `230cc497…` | final | K4 es2 2 / 800 |
| SH_T18 | `SH_T18/model_5000.pt` | `6935037a…` | final | K4 es2 3 / 800 (accepted) |
| SH_P18 | `SH_P18/model_5000.pt` | `50b7c60a…` | no deadband, no trim branch | FAIL K3, K6 (at db 0.25: FAIL K5, K6) |

**Record:**
- `CHARTER.md` §3 (L40-356), `PLAN.md`, `facts.{md,json}`.
- Calibration: `step4_read.*`, `step4b_read.*`.
- Heads: `head_SH_*.{md,json}`.
- Premise reads: `*_premise_{mid,final}.json`.
- Observe passes: `observe_SH_*_{height,zero}.*`.
- Videos: `video_sh_*_sanity.json`.
- Height-row eval stamps: `parkour/outputs/omni_plan/phase2/lr/SH_T17{,z}/…`.
- Code: `scripts/sh_core.py`, `sh_kin.py`, `sh_dagger*.py`, `sh_eval_*`, `sh_table_*`, `sh_read.py`, `sh_observe_read.py`, `sh_record_video.*`.