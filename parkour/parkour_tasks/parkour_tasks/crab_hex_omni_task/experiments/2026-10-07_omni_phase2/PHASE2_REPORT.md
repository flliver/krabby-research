# Omni phase 2 (obstacles): campaign report

**Status: CLOSED, BAKED 2026-10-08.** Sources live in this directory unless a path says otherwise. `CHARTER.md` is the
record: §9 is the log for 2026-10-06 and 2026-10-07; §10 is stage p2b, and its 2026-10-08 entries (p2b, the bake) sit under it. Entries are cited below as "§9 <date time>" or "§10 <date time>".

## 1. Outcome

**Head of record (omni phase 2):** S1 s18, chain `P2__S1_s18`, `policy/6_p2a_field/model_19994.pt`.
- Its sha256 is `968cc550…c23c`, byte-identical to the X08 run's model_19994 (§10 2026-10-08 BAKE APPLIED;
  `policy/manifest.yaml`, stage `6_p2a_field`).
- Stages 1–5 are unchanged, on the same s18 lineage. `5_c6x_clockfree` stays the phase-1 head.
- Trained at code_sha dfcc5310 and eval_sha 1d021625. The REF re-evals reproduce it exactly on code e3fe70fb.

**Recipe.**
- Base preset 2a: curriculum, 2a DR, collision / edge / stumble terms, 40-s episodes.
- The full side-first B overlay and the c6x clock-removal set. `KRABBY_PHASEOUT ""` and `KRABBY_OMNI_FACING 0.2:1.2`.
- The **wall field**: GEOM field, pad 1.7 m, blocks 14:28, bar:slab:pit 0.4:0.4:0.2, `FIELD_EDGES wall`,
  **DIFF 0.20:1.0**.
- **PROMOTE 0.7:0.4**, calibrated on CALW1 (r̄ 0.798).
- **`KRABBY_ACTION_RATE_W −0.05`** (S1).
- 5000 iterations (15000 → 20000), forked from the c6x model_14995. See `candidates.yaml` `P2__S1_s18` and §9 2026-10-06
  22:15.

**Gains against each seed's phase-1 anchor (ANCW).** The field_hard / field_light / ring_obst rows are completion at n = 200 (env_seeds 1 + 2) unless marked; the turn, move_turn and jitter rows are flat-suite metrics at env_seed 1.
Sources: §9 2026-10-07 22:37 (recipe read), and `policy/manifest.yaml` (stage 6 and `replicates`) for n = 400.

| | s16 | s17 | s18 (baked) |
|---|---|---|---|
| field_hard (walls, d 0.70–1.0) | 0.435 → 0.830 (**0.815** n=400) | 0.560 → 0.730 (0.725 n=400) | 0.500 → 0.800 (**0.8125** n=400) |
| field_light | 0.765 → 0.980 | 0.975 → 0.985 | 0.830 → 0.995 |
| ring_obst (phase-1 ramps) | 0.695 → 1.000 | 0.985 → 1.000 | 0.805 → 1.000 |
| turn fc / yp | 0.763 / 0.100 → 0.963 / 0.161 | 0.714 / 0.095 → 0.948 / 0.145 | 0.797 / 0.104 → 0.958 / 0.158 |
| move_turn survival | 0.740 → 0.995 | 0.985 → 0.990 | 0.945 → 1.000 |
| jitter | 0.195 → 0.145 | 0.199 → 0.157 | 0.198 → 0.150 |
| §6.1 verdict | PASS | PASS | **COST** (err_back +0.057) |

**Cost: back overshoot at s18.** The suite speed ratios are back 1.087, side 1.051, fwd 1.050 and diag 1.015. The c6x
head had 1.030 / 1.045 / 1.041 / 0.956 (manifest follow-up `back_overshoot`).
- It is most likely the S1 knob's cost (one paired run at s18; the knob effect and run-to-run divergence are not separable). Against A1 s18, S1 is worse by 0.043 on fwd |ratio − 1| and 0.061 on back
  (§9 2026-10-07 11:19).
- It does not show at other seeds: S1 s16 tracks back at 1.041 and fwd at 0.996.

## 2. The path

1. **The ramp field was too easy** (§9 2026-10-06 21:08).
   - The anchors failed the §5 discrimination check: field_hard was 0.705 / 0.990 / 0.850, and s17 sat above the 0.85
     ceiling.
   - 86 of 91 falls were `pitch_fwd` in move_turn. Completion was measuring move_turn survival on bumps.
   - At d = 1.0 the generator tops out at 0.09 m, below the 0.12 m wall threshold.
2. **A probe** (`probes/scenarios_probe_hard.yaml`; §9 2026-10-06 21:42), 50 episodes:

   | Setting | s16 | s17 | s18 |
   |---|---|---|---|
   | walls 0.20:0.70 | 0.76 | 0.84 | 0.74 |
   | ramps 0.70:1.0 | 0.68 | 0.94 | 0.76 |
   | walls 0.70:1.0 | 0.42 | 0.56 | 0.40 |

   Walls moved the falls into the walking holds.
3. **The user chose walls, hardest at the top** (§9 ~22:00).
   - DIFF 0.20:1.0; field_hard moved to 0.70:1.0.
   - Everything was recalibrated (PROMOTE 0.7:0.4), re-smoked and re-anchored.
   - The wall anchors (ANCW) passed discrimination: field_hard 0.435 / 0.560 / 0.500, bands 0.04 / 0.06 / 0.10
     (§9 2026-10-07 00:59).
4. **A1 s18 gate: PASS** (§9 04:27).
   - field_hard 0.500 → 0.745 (needed ≥ 0.70); field_light 0.995; ring_obst 0.995.
   - move_turn survival 0.945 → 1.000.
5. **The A0 ring control is about equal to A1** (§9 07:50; descriptive).
   - A0 read 0.715 / 1.000 / 1.000, against A1's 0.745 / 0.995 / 0.995.
   - The field and walls buy at most ≈ +0.03, which is inside the noise. **Most of the gain comes from phase-2 training
     itself**: the curriculum, the 2a DR and 5000 more iterations.
6. **S1 smoothness kept by the user, over the rule's DO NOT KEEP** (§9 11:19, ~11:50).
   - Against A1, jitter fell 26 %. fwd tripod went 0.53 → 0.74, fwd steps/m 12.4 → 9.6, taps/m 17.5 → 12.3, and work/m
     1232 → 1066. field_hard was 0.800 against 0.745.
   - The price is the fwd / back overshoot.
   - The user then ran the replicates on S1 instead of A1 (§9 ~12:05).
7. **F1 stand fix dropped** (§9 15:24, then the user's decision).
   - It failed the low-tap test: clearance 0.093 m, tripod 0.789.
   - field_hard was 0.695, against a 0.70 target.
   - E1 had already been dropped by the user after its §4 replay-gate FAIL (the gate stops and asks; it drops nothing itself).
8. **S1 replicated** (§9 19:40, 22:37). PASS at s16 and s17, COST at s18, gain at 3/3, no HAND LOSS. The recipe read is
   **ADOPTABLE**. The p2b trigger fired, because field_hard was below 0.85 at every seed.
9. **§7 judging: unanimous S1 s18** (§9 ~23:00; `judging_result.json`).
   - 3 opus judges and a fable check.
   - Ranking: S1 s18 > S1 s16 > A1 s18 > S1 s17. Obstacle rank-sums were 6.5 / 10.0 / 11.5 / 12.0.
   - **The lead is fragile.** It comes from near-ceiling rows: field_light 0.995 vs 0.980, and field_suite move_turn
     1.000 vs 0.995.
   - A tie-aware reading, or one that ranks PASS above COST, picks S1 s16.
10. **s16 vs s18 breakdown** (§10 2026-10-08).

    | | S1 s16 | S1 s18 |
    |---|---|---|
    | field_hard, n = 400 | 0.815 | 0.8125 |
    | move_turn ratio / fc | 0.72 / 0.64 | 0.81 / 0.75 |
    | move_turn falls on walls | 27 | 5 |
    | fwd tripod / steps/m / work/m | 0.70 / 10.5 / 1242 | 0.74 / 9.6 / 1066 |
    | flat speed ratios (fwd / side / back / diag) | 0.996 / 1.039 / 1.041 / 0.947 | 1.050 / 1.051 / 1.087 / 1.015 |
    | lineage cost | re-bake stages 1–5 from the weakest phase-1 line (flat 0.89) | adds stage 6 only |

11. **The user baked S1 s18** (§10 2026-10-08; commits 70c0ce62 and dd972b6a).

## 3. Stage p2b (CHARTER §10, `P2B_PLAN.md`)

**Diagnosis** (`P2B_PLAN.md` Context). The S1 field_hard raws had 128 falls in 600 episodes, all slow tip-overs. Tilt
stays above 0.15 rad for a median 1.7 s while the robot keeps tracking.

| Fall category | Share |
|---|---|
| pit under the tip-side foot | 36 % |
| leg strike | 34 % |
| feature under the trailing feet | 17 % |
| block | 7 % |
| no feature | 6 % |

Other findings:
- Pit swings under 0.08 m escape 4 % of the time; swings of 0.10 m or more escape 47 %.
- `reward_orientation` is zeroed on obstacle tiles. Its training share is about 1/267 of tracking.
- Tilt² in the 2 s before a fall is 15× the walking level.
- The user dropped a top-difficulty remap once the curriculum premise was corrected (audit pass 1): about 62 % of obstacle episodes are
  already at d ≥ 0.70.

**The plan and its audit** (2 passes, cap reached).
- The audit fixed an unreachable NO GAIN bar, an under-dosed TG and the wrong crouch statistic.
- The last change, terrain-relative lift, was not re-audited.
- **Arms:** C0 (a continuation, s18 only), CL (lift terms, terrain-relative) and TG (`reward_orientation_upright` on all
  tiles).
- **Gain gate:** REF + max(0.05, 2·SE) at n = 400.

**Replay gate: PASS** (`REPLAY_GATE_P2B.md`, `replay_gate.json` `p2b`; 4200 episodes, 139 falls).
- **CL:**
  - income vs terrain-relative apex ρ +0.97, pit swings included;
  - crouch payoff exactly 0 (the root-relative form paid 100 %);
  - shares 3.4 % (foot clearance) and 1.7 % (swing-min); weights unchanged.
- **TG:**
  - pre-fall cost 12–15× walking;
  - the share at −0.7 was 0.38 %, so it was **re-dosed to −2.3** (1.0–1.3 %).

**s18 screen** (§10 2026-10-08 14:30; `p2b_screen.json`, `p2b_mech_screen.json`). All three are NO GAIN; the bar was
≈ 0.867–0.873.

| Head | field_hard (n=400) | Falls (pit / leg strike / move_turn) | Stand steps/s, clearance | fwd tripod, apex (m) |
|---|---|---|---|---|
| REF_S1_s18 | 0.8125 | 75 (29 / 23 / 5) | 2.83, 0.135 | 0.74, 0.139 |
| C0 | 0.785 | 86 (43 / 33 / 28) | 2.67, 0.113 | 0.70, 0.132 |
| CL | **0.695** | 122 (46 / 49 / 31) | 1.42, 0.077 | 0.64, 0.112 |
| TG | 0.8275 | 69 (19 / 27 / 14) | **1.08**, 0.104 | 0.74, 0.159 |

- **CL hurt.** It raised no apex where it counts (walking diag 0.052 vs 0.058 m), even though its training income was
  2.8× the replay's prediction.
- **TG** had the fewest falls and fewest pit falls, and **cut stand stepping by 62 %**. Against C0 it gained +0.0425,
  where +0.056 was needed.

**The user stopped p2b.** No replicates ran and no lever was adopted. Both knobs stay default-off.

## 4. Lessons for the next campaign

- **n = 200 is too coarse to rank heads.**
  - S1 s16 vs s18 read 0.83 vs 0.80 at n = 200 and 0.815 vs 0.8125 at n = 400.
  - Per-env_seed values span 0.79–0.86, and the anchor bands reach 0.10 (s18) and 0.15 (s16 on ramps).
  - Rank-sums built on near-ceiling rows flip on 1–3 episodes. Pre-register a tie tolerance, or use n = 400 for any
    head-of-record comparison.
- **Continuation alone does not help.**
  - No intermediate (16000 / 17500 / 19000) replaced a head at any seed.
  - C0's further 5000 iterations went 0.8125 → 0.785 (inside noise, SE ≈ 0.028), but move_turn falls on walls rose 5 → 28.
  - Plan levers, not length.
- **2b-style shaping still does not pay.** 2b was flat-to-negative against 2a in every earlier task. CL, its lift terms
  made crouch-proof, was the worst arm here.
- **A replay PASS shows that a term is wired and sized. It does not show that the term works.**
  - CL passed every replay check and was live in training, yet it hurt.
  - The one replay FAIL that was trained, F1 (after the user overruled it), failed in training too; E1 never trained.
- **Anchor discrimination checks are worth their stop.** The ramp field would have produced unreachable gain gates
  (s16 ≥ 1.005, s17 ≥ 1.09).
- **Process:**
  - **Watcher re-arming.** The screen verdict sat unread from 14:30 to 15:20, because the watcher was not re-armed after
    TG's milestone.
  - **code_sha acceptance.** The intended p2b code move was refused by the driver until run_arm passed
    `--accept-code-change` (§10 03:40).
  - **Fresh chain names after a kill.** The CAL1 relaunch continued the killed run instead of forking; the resume read caught it (§9 18:22).
  - **Inertness must be exact.** The anchor suite reproduced the manifest, and all 9 REF rows reproduced their p2a
    reads. Like-for-like row evals are the only valid comparison, given the eval process-state hazard.

## 5. Open follow-ups (`policy/manifest.yaml` `followups`)

- **move_turn: nobody actually turns.**
  - Survival is 1.000, but yaw_progress is 0.097 rad/s (about a third of the 0.3 commanded), fc 0.747 and
    heading_final 0.396 rad.
  - Phase 2 raised move_turn survival (s17's change is within noise) without closing the turn rate.
- **The robot marches at stand.**
  - It takes 2.83 steps/s at 0.135 m clearance.
  - F1 was dropped. S1 and F1 each cut stepping by about 25 % but kept full-height lifts.
  - **Lead:** TG's 2.83 → 1.08 steps/s (n = 1, unplanned). Untested levers: TG replicated and read on stand, and a
    stand-only TG, each through the replay gate first.
- **Back overshoot** of 1.087 at s18. Untested levers: a back-speed tracking re-weight, or a short fine-tune at a lower
  action_rate weight.
- **The wall deficit remains.**
  - field_hard is 0.8125, below the §6.5 0.85 bar. Diag accounts for 40 of 75 falls and back for 13; pit falls 29 and
    leg strikes 23.
  - **Tried:**
    - walls and DIFF 0.20:1.0;
    - A0 on the ring;
    - S1 (kept) and F1 (dropped);
    - continuation (C0);
    - terrain-relative lift (CL, hurt);
    - tilt guard (TG, no wall gain).
  - **Not tried:**
    - a combined-lever arm;
    - 2b's goal-velocity income and wider facing band;
    - an obstacle-crossing bonus;
    - the top-difficulty remap (dropped on premise);
    - E1 (dropped).
  - The policy still lifts no higher on hard walls than on light, so pit escape (swings ≥ 0.10 m) remains the obvious
    mechanism target.
