# Plan: omni phase 2, stage p2b (two levers on the S1 recipe)

> **Audit trail** (cap 2, reached)
> - **Pass 1** (evidence inventory, 5 premise probes, 2 lens-split opus · xhigh auditors). Blocking fixes:
>   - TG dose floor;
>   - per-term CL replay bands;
>   - flat floor also vs ANCW (chained);
>   - attribution needs its own gain;
>   - fall breakdown restated from the saved data;
>   - curriculum premise corrected.
> - **Your answers during the audit:**
>   - gain gate = binomial-SE margin at n = 400;
>   - CL = lift terms only;
>   - §7 unchanged (tie tolerance removed);
>   - TD dropped.
> - **Pass 2** (opus · xhigh + fable · high). Fable: nothing blocking. Opus: one finding, that the CL terms paid a
>   crouch → your answer: terrain-relative lift. **That last change was not re-audited.** The replay gate's
>   counterfactual check, the unit tests and the s18 stop guard it.
> - **Gate check:**
>   - the NO GAIN bar was unreachable for one-knob levers (fixed);
>   - TG was under-dosed (fixed);
>   - the crouch check measured the wrong statistic (fixed).
> - **Premises probed:** clearance wiring and dose, tilt precursor (15× before falls), curriculum saw-tooth,
>   reference inertness.
> - **Size:** 7 steps / ≈ 2,180 words (reviewed: 7 / ≈ 1,580). The growth is blocking fixes plus your answers.

> The parent plan (phase 2 / p2a) is complete. It is recorded in `2026-10-07_omni_phase2/CHARTER.md` §1–§9. This plan
> amends that campaign: it becomes CHARTER §10, written before any p2b run.

## Context

**What p2a delivered.** Recipe S1 is the B overlay on the wall field at DIFF 0.20:1.0, plus action_rate −0.05. Its
recipe read is ADOPTABLE: PASS at s16 and s17, COST at s18 (back overshoot).
- field_hard (walls, d 0.70–1.0) went 0.435 / 0.56 / 0.50 → 0.83 / 0.73 / 0.80.
- The p2b trigger fired: field_hard is below 0.85 at every seed.
- The user deferred the bake and asked for p2b to be planned first.

**What limits field_hard.** From the read-only diagnosis of the S1 field_hard raws (128 falls / 600 episodes; labels
as saved in the diagnosis `falls.pkl`):
- **The falls are slow tip-overs over the body's short axis** (body frame; the travel-frame classifier labels 27/128 as
  roll). Tilt stays above 0.15 rad for a median 1.7 s before failure, while the robot keeps tracking at the commanded
  speed.
- **Fall categories:**
  - 36 %: pit under the tip-side foot;
  - 34 %: leg strike (femur/tibia contact > 200 N);
  - 17 %: feature under the trailing feet;
  - 7 %: tip-side foot on a block;
  - 6 %: no feature.
- Pit swings under 0.08 m escape 4 % of the time; swings of 0.10 m or more escape 47 %.
- The policy lifts no higher on hard walls than on light (+0.005–0.009 m).
- **The orientation penalty is zeroed on obstacle tiles** (`reward_orientation`, `rewards.py:387-397`). In the S1 run
  its training share is about 1/267 of tracking. Tilt² in the 2 s before a fall is 15× the walking level (probe,
  S1 s18, 21 falls).
- **The curriculum:** promotion only, and envs that pass level 9 are resampled into 0–9 (`parkour_event.py:271-276`).
  The logged mean level of ≈ 6 is that saw-tooth's stationary mean, not a ceiling.
  - Training already spends about 62 % of obstacle episodes at d ≥ 0.70 and about 18 % on the d = 1.0 row.
  - A top-difficulty remap (0.40:1.0) would raise the in-band share only to about 73 %, so it was dropped.
- The intermediate checkpoints are flat over the last 4000 its, so more iterations alone are unlikely to help.

**Prior evidence on "2b".** Forward, sideways and the old omni: 2b was flat-to-negative against 2a every time, and
never had a control or an isolated term. Clearance terms have never been shown to raise lift on the A15+B plant.

**User decisions (2026-10-07), including the answers during audit pass 1:**
- Two levers, each as its own arm: **clearance (lift terms only)** and **tilt guard**.
  - Top-difficulty curriculum was dropped after its premise was corrected.
  - The obstacle-crossing bonus is out of CL: it is a tilt-gated progress income.
- The p2a-continuation **control at s18 only**; lever arms at all three seeds.
- **Gain gate:** binomial-SE margin with field_hard at n = 400.
- **Ranking:** CHARTER §7 unchanged, no tie tolerance.
- Not chosen: 2b's goal-velocity income and the wider facing band.
- Bake deferred.

## Ground rules (carried from CHARTER §1a)

- Branch `feat/crab-hex-omni-task`. Never push.
- Stage only this campaign's paths; parallel sessions share the tree.
- One Isaac process at a time; `p2_queue.sh` holds a flock.
- Stop Isaac with `systemctl --user stop`; never SIGKILL Kit.
- eval_sha does not move: run_phases.py, the harness, gait_eval/, scenarios_omni.yaml and eval_categories.py are
  untouched.
- The memory rule holds: new reward terms pass the offline replay gate before they train.
- Every stop below push-notifies the user.
- Adoption, bake and push are the user's.

## Recipe (every p2b arm)

- **Fork:** each seed's S1 p2a head (`P2__S1_s{16,17,18}` model_19994), as `{checkpoint}`.
  - The fork sha256 is verified against the arm record.
  - Stage name `p2b`, base `2a`, 5000 its (19995 → 24994).
- **Overlay:** the S1 overlay verbatim (walls, DIFF 0.20:1.0, PROMOTE 0.7:0.4, ACTION_RATE_W −0.05, B construction,
  blank PHASEOUT, FACING 0.2:1.2). Base 2a keeps every other 2b difference out.
- **Each lever arm changes exactly one knob family:**

| Arm | Seeds | Change vs the control |
|---|---|---|
| **C0** control | 18 | none (p2a continuation) |
| **CL** clearance (lift only, terrain-relative) | 16, 17, 18 | `KRABBY_FOOT_CLEAR_W 1.0`, `KRABBY_FOOT_CLEAR_MIN 0.03`, `KRABBY_FOOT_CLEAR_FLAT 1`, `KRABBY_SWING_MIN_CLEAR_W −0.4` (2b's values; the replay gate may re-dose them), plus the new `KRABBY_FOOT_CLEAR_TERRAIN 1`: both terms measure foot height above the terrain under the foot, not above root − 1.05 m, so a crouch earns nothing and a foot in a pit is measured from the pit floor |
| **TG** tilt guard | 16, 17, 18 | new `KRABBY_ORIENT_ALL_TILES 1`: `reward_orientation` uses the existing `reward_orientation_upright` (all tiles), with `KRABBY_ORIENT_W` set from the replay gate's 1–5 % band (≈ −1.3 to −2). One weight covers all tiles, so the flat-tile penalty rises by the same factor. |

## Steps

1. **Tilt-guard and terrain-relative-lift knobs + tooling, landed as one commit (code_sha moves once).**
   - **Code, both knobs default off and in the omni knob layer** (`crab_hex_omni_task/config/crab_hex/omni_knobs.py`,
     applied in the omni env cfg):
     - `KRABBY_ORIENT_ALL_TILES` swaps `reward_orientation`'s func to `reward_orientation_upright`.
     - `KRABBY_FOOT_CLEAR_TERRAIN` makes `reward_foot_clearance` and `penalty_swing_min_clearance` take foot height
       above the terrain height-field sample at the foot's xy, instead of `root_z − 1.05`.
       - Reuse the field-mode pixel lookup (`tiled_pixel_index` in `mdp/omni_exposure.py`).
       - Use a terrain-relative variant of each func; the existing funcs are untouched.
     - Unset, the config is byte-identical (golden/config identity test).
   - **Tooling (tool files, no sha move):**
     - `run_candidates.py`: parsers and `OVERLAY_DUMP_CHECKS` for the four CL knobs and `ORIENT_ALL_TILES`
       (weight/param/func checks, using the existing `_weight_ok` / `_func_swapped_ok` patterns).
     - `run_arm.sh`: verify `{checkpoint}` forks of p2a heads against the p2a eval stamps'
       `run_meta.checkpoint_sha256`; resolve `fork_source_rd` for them.
     - `p2_live.py`: per-arm liveness against step 2's replay magnitudes (CL → `reward_foot_clearance` and the
       swing-min penalty; TG → `reward_orientation` above the seed's p2a training tail, which has no obstacle-tile
       share); p2b intermediate checkpoints 21000 / 22500 / 24000.
     - `p2_read.py`: p2b references, the ladder, attribution and the mechanism reads (step 6).
     - `p2_mech.py`: the diagnosis fall/swing classifier, saved as a tool and re-run on REF so the mechanism reads use
       one method.
     - `p2_queue.sh`: p2b steps.
     - Unit tests for each.
   - Then a fable · high review of the landed diff.

   → opus · high (code) / sonnet · high (tests); ✓ fable · high

2. **Offline replay gate** (memory rule; CPU), on the S1 p2a heads' saved field_hard / field_light raws.
   - **CL passes if, per term, on field and suite raws** (FOOT_CLEAR_FLAT pays on flat tiles too), using the dumped
     params (the swing-min floor is 0.04 m):
     - foot-clearance income rises with terrain-relative swing apex, **including on pit swings**, so there is a
       pit-escape signal;
     - **no crouch payoff (counterfactual):** re-score the raws with the root lowered 1 cm at fixed foot positions.
       The change in each term's income must be ≤ 10 % of its payoff per cm of terrain-relative lift (≈ 0 by
       construction). Otherwise stop and ask.
     - foot-clearance income is 1–10 % of tracking income in non-fall walking;
     - the swing-min penalty is ≤ 5 % of tracking.
   - **TG passes if:**
     - the cost in the 2 s before falls is ≥ 5× the cost in non-fall walking windows;
     - mean cost is 1–5 % of tracking income in non-fall walking (the floor matters: at −0.7 it is ≈ 0.5 %).
   - **Re-dose rule:** if a term's share is outside its band, pick the nearest weight that lands inside it. Step 2
     writes each term's replay magnitude to `replay_gate.json` for the +200 liveness check.
   - **A failed check stops and asks.**
   - Tooling: `experiments/tools/replay_motion_terms.py` gets the two kernels (tool file).

   → opus · high; ✓ fable · high check of the read

3. **References on the new code.**
   - Re-evaluate the three S1 p2a heads on every read row: suite, lr, field_light / ring_obst at env_seeds 1 + 2,
     **field_hard at env_seeds 1–4 (n = 400)**, field_turn, field_suite. s18 also runs its suite at env_seed 2 (the
     suite band). Names `REF_S1_s{16,17,18}`.
   - This keeps the INVALID EVAL rule (sidecar equality) valid for every p2b comparison.
   - **Code-inertness check:** each reference reproduces its p2a reads exactly. Evals are deterministic and the knob is
     off, so any difference stops and asks. ≈ 5 GPU-h.

   → sonnet · high

4. **s18 screen:** C0, then CL and TG at s18 (≈ 11 GPU-h).
   - **Live kills as in CHARTER §8:** resume read; +200 wiring + liveness; +1000 terrain; collapse.
   - Any relaunch after a kill uses a fresh chain name (§9 rule).
   - Each head gets the standard reads (field_hard at env_seeds 1–4) plus intermediate reads at 21000 / 22500 / 24000.
   - **STOP and ask** with the s18 verdicts (step 6). The user may prune levers before the replicates.

   → sonnet · high to run; opus · high reads

5. **Replicates:** each kept lever at s16, then s17 (≤ 4 arms, ≈ 14 GPU-h), with the same reads. Then **STOP and
   ask** with the recipe reads.

   → sonnet · high to run

6. **Pre-registered reads (p2_read; stated before any p2b run).**
   - **Per arm and seed vs that seed's REF** (the p2a head on the new code). The CHARTER §6.1 ladder applies, with REF
     in place of the anchor:
     - INVALID / PENDING / HAND LOSS / DOSE;
     - FLAT FAIL: same list, floors and bands, with bands from REF's es1 / es2 split;
     - **NO GAIN** unless all three hold:
       - field_hard (pooled over env_seeds 1–4, n = 400) ≥ REF + max(0.05, 2·SE), where
         SE = √(p_REF(1 − p_REF)/400 + p_arm(1 − p_arm)/400) (≈ +0.06);
       - field_light ≥ REF − max(0.03, band);
       - ring_obst ≥ REF − max(0.05, band).
     - COST;
     - PASS.
   - **Flat floor against phase 1 too:** each p2b head must also clear the §6.1 FLAT FAIL list against its seed's
     ANCW anchor, using chained suite deltas (p2b − REF plus REF − ANCW), so flat losses cannot ratchet.
   - **Attribution at s18:** a lever "beats continuation" if both hold:
     - its own REF verdict is a gain (PASS or COST);
     - its field_hard beats C0 by the gain gate's margin rule (both at n = 400), and there is no FLAT FAIL against C0.
     - s16 / s17 have no control (user choice). Their reads are against REF only and are labelled "lever +
       continuation".
   - **Mechanism checks** (descriptive, field_hard raws). The diagnosis fall/swing classifier is saved as a tool in
     step 1 and re-run on REF, so both sides use one method.
     - CL: swing apex on walking holds ≥ REF + 0.02 m, reported beside root height above terrain (a crouch is not
       lift); pit-category fall share down.
     - TG: tilted-and-tracking time before falls down, and falls per episode down.
   - **Recipe read per lever at 3 seeds:** the CHARTER §6.2 rules (ADOPTABLE / TRADE-OFF / NOT ADOPTABLE / MIXED).

   → opus · high decides; ✓ fable · high

7. **Head-of-record proposal and write-up.**
   - **Candidates:** the three S1 p2a heads (REF) plus every p2b head that reads PASS or COST, the C0 continuation
     included. A C0 win means "the S1 recipe trained 5000 more its".
   - **Ranking:** the CHARTER §7 rule unchanged (no tie tolerance). Field_hard is read pooled at n = 400 where both
     heads have it. 3 opus judges + a fable check; their noise notes stay descriptive.
   - REPORT.md answers: did any lever raise field_hard beyond continuation, by which mechanism, at what flat cost.
   - **The bake is the user's call.** Records + policy folder; if the winner is not s18, stages 1–5 are re-baked from
     its lineage.

   → opus · high; ✓ fable · high

## Budget

| Item | GPU-h |
|---|---|
| References on the new code (field_hard at 4 env_seeds) | ≈ 5 |
| s18 screen (3 arms, incl. the extra field_hard evals) | ≈ 11 |
| Replicates (≤ 4 arms) | ≈ 14 |
| **Total** | **≈ 30** (≈ 1.3 days) |

Steps 1, 2, 6 and 7 are CPU only.

## Verification

- Unit tests:
  - the tilt-guard knob, including config identity when unset;
  - the terrain-relative lift kernels: a lowered root at fixed feet changes nothing; a foot over a pit is measured from
    the pit floor; a foot over a block from the block top; config identity when unset;
  - every new dump check (round-trips and failures on a real dump);
  - run_arm fork verification;
  - p2_live liveness;
  - the p2_read ladder against REF (the n = 400 SE gate), attribution and the chained ANCW flat floor (self-test);
  - the p2_queue step order and stops.
- Before the screen: replay gate PASS, the reference inertness check, and the dry run of every p2b chain through the
  driver validator.
- Each arm: resume read, the +200 wiring/liveness and dump audit, and the INVALID EVAL check on every stamp.
- Every verdict goes to CHARTER §10 / CHANGELOG and is push-notified.

## Left for implementation (checks)

- REF s18 needs its suite at env_seed 2 too (`bands()` reads `suite2`), or every p2b verdict reads PENDING.
- The re-dose rule moves only the out-of-band term's own weight.
- The replay kernels read the dumped params. The swing-min floor is 0.04 m (`MIN_SWING_CLEARANCE_M`);
  FOOT_CLEAR_MIN moves only the income floor.
- The dry run composes each lever's env and dumps it: the TG func swap, the CL params (FOOT_CLEAR_FLAT → no flat mask).
  The +200 check is the first live exercise.
- Stage only this campaign's paths.
- **Code-state pins after the step-1 commit:**
  - Re-point the code guard's reference from ANCW_s18 to REF_S1_s18's suite stamp.
  - Re-run SMOKES_OK on the new code, or `p2_queue.sh` exits 4 and every row eval exits 7.
- Add `KRABBY_ORIENT_W` to the dump checks.
- The replay kernels read the lever's composed dry-run dump, not the p2a dump (it has foot-clear min 0.07, flat-masked).
  Run that dry run before step 2.
- Re-dosing gives each term one weight across all three seeds.
- Name the "tracking income" term set used for the share bands before step 2 is read.
- p2_read pools field_hard over env_seeds 1–4.
- p2_live needs two CL liveness terms (sign-aware) and none for C0. The verdict table lists C0 only at s18.
- The step-3 inertness comparison is like-for-like: run_arm row eval against run_arm row eval, not against the driver's
  suite stamp.
- Every p2b verdict reports walking-hold tracking beside completion, since completion is survival-only.
- p2_mech reports pit swings and ordinary swings separately.

## Out of scope (not run)

- A combined-lever arm (a follow-up if more than one lever is kept).
- 2b's goal-velocity income and facing widening.
- Stand-march fixes.
- Phase 3 (student).
