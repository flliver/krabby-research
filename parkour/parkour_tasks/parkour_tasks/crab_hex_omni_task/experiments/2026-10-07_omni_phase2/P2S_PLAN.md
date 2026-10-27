# Plan: stop the head stepping at rest (omni phase 2, stage p2s, CHARTER §11)

> **Audit trail** (cap 2, reached)
> - **Pass 1** (inventory, 2 premise probes on the raws and the configs, 2 lens-split opus · xhigh auditors). Blocking
>   fixes:
>   - hover = unloaded **and** raised (force-only flagged the statue in 71/100 episodes);
>   - off-ground / G2 measured against a pooled loaded-foot stance reference (a self-reference cannot see a held-up
>     foot);
>   - STILL clearance on the median-over-feet statistic your 5-cm target is on, and off-ground ≤ 0.10;
>   - the termination guard widened.
> - **Your answer:** walkstop is a guard, not part of STILL.
> - **Pass 2** (opus · xhigh + fable · high, whole document). Both found one blocking issue: the widened termination
>   guard was still inside sibling noise. Fix: that clause is now notify-only; the episode-length clause kills.
>   **This last fix was not re-audited.**
> - **Re-derived by both pass-2 auditors on the raws:** the statue reads STILL; TG has 18 hover episodes; REF
>   clearance is 0.135; G2 is TG −0.91 / CL −0.53.
> - **Premises verified:** no termination penalty with zero bootstrap on a fall; `--sample-actions` works on this
>   teacher; walkstop needs no code; stand slot holds the full draw.
> - **Size:** 8 steps / ≈ 2,880 words (reviewed: 8 / ≈ 2,580). The growth is audit fixes plus the implementation
>   checklist.
> - **Routing:** roster ok · ceiling: opus · fable checks on steps 3, 4 and 8.
>
> The p2b plan this file used to hold is archived in `2026-10-07_omni_phase2/P2B_PLAN.md`.

## Context

**The problem.** The head of record (S1 s18, `policy/6_p2a_field/model_19994.pt`) marches in place when it is commanded
to stand.
- Suite stand hold (100 episodes × 12 s): 2.83 steps/s, 0.135 m lifts, tripod 0.76, all six feet down 0.8 % of the
  time.
- It is the walking rhythm (≈ 0.47 Hz) run at a zero command.

**Why nothing has fixed it:**
- **The reward barely prefers stillness.** At the S1 weights, standing still earns ≈ +0.2/s more than marching, out of
  ≈ 2.4/s.
  - Nothing prices foot lift or pays for planted feet. The clock stance income that used to is at 0.001.
  - Stand is 15 % of training draws (`KRABBY_OMNI_CMD_POOL stand:0.15`). `stand_mask()` (slot-based) feeds only
    default-off knobs.
- **Weak pulls failed:**
  - The clock-era stance income (+0.76/s) and F1 (stance income 0.9) both left full-height lifts.
  - STAND_POSE failed its replay gate.
  - Smoothness terms (action_rate in W, C0 and S1) slow the march; they do not stop it.
- **The TG "lead" is a measurement artefact** (verified in this session on the raws):
  - All 9 of TG's zero-step stand episodes hold one tripod in the air for > 95 % of the hold. The eval counts no steps
    for a foot that lifted before the window.
  - TG's unloaded-foot share is 0.487, against REF 0.455 and C0 0.468.
  - **So steps/s alone can be gamed by hovering**, and the manifest `stand_march` lead must be corrected.
- **Literature** (`lit-review-gait-motion-quality.md` §2/§4/§10): stacks that stand still make stand a command-keyed
  regime, with stand-gated stillness and support terms at 5–20× walking weights.

**Your decisions (2026-10-08):**
- **Target: mostly still.** Median ≤ 0.5 steps/s and lifts ≤ 5 cm on the suite stand, at both env_seeds; an
  occasional settling step is fine. Stands after walking (walkstop) are a guard, not part of the target.
- **Lever: a stand-only stillness term.** The replay gate shapes it. Two doses at s18, then replicates. TG replicates
  are not run.
- **Cost: none beyond noise.** The flat ladder and walls are non-inferior; start-from-stand and stand-on-walls are
  unchanged.
- **Home: a phase-2 amendment, stage p2s** (CHARTER §11). It reuses run_arm / p2_read / p2_live, the REF and C0 heads,
  and the S1 overlay. A winner bakes as policy stage 7.

**Ground rules (CHARTER §1a):**
- Never push.
- Stage only this campaign's paths.
- One Isaac process at a time; stop it with `systemctl --user stop`.
- eval_sha does not move: run_phases.py, eval_crab_hex_gait.py, gait_eval/, scenarios_omni.yaml and
  eval_categories.py are untouched. The new estimators are campaign-local.
- Replay gate before training.
- Fresh chain names after a kill.
- Push-notify at every stop; re-arm the watchers after every milestone.
- Adoption, bake and push are yours.

## The lever: two stand-slot terms (default off, one knob family)

Both go in `mdp/crab_hex_omni_motion_rewards.py`.

**Gate:**
- g = `stand_mask()` (the command slot only) × upright (−g_z > 0.9, as F1).
- The gate is the slot, not "slot and zero turn". The policy could hold its facing error past the dead band and escape
  a turn-keyed gate.

**Feet:** the six footpads. F_f is the contact-force norm and v_f the 3-D foot speed. s_f = clamp(F_f / 50 N, 0, 1);
50 N is the eval's "loaded" line.
- The robot is 230 kg, so stance feet carry 200–800 N.
- s_f is therefore effectively a contact bit, and pressing feet sideways cannot inflate it.

**T1 `reward_stand_support`** (`KRABBY_STAND_SUPPORT_W` ≥ 0): g · (Σ s_f − 3), in supported feet beyond a tripod.
- Values: six down +3, the E6 statue +2 (one foot unloaded), REF's march ≈ +0.3, a tripod hover 0.
- **Why an income with a −3 offset:** there is no termination penalty and γ = 0.99 gives a ≈ 2-s horizon. A pure
  penalty at the high dose would make the stand slot net-negative (≈ −1.1/s) and falling the better option. With the
  offset, the marching head's stand-slot reward stays ≈ +2.3/s.

**T2 `penalty_stand_slide`** (`KRABBY_STAND_SLIDE_W` ≤ 0): g · Σ s_f·v_f, in m/s, unsquared.
- It prices loaded feet that move: shuffle, slide, scuff. That closes the route T1 leaves open.
- It is weighted by s_f, so it does not favour hover. Plain foot speed does not price a still hover at all.

**Doses (step 4 sets the final values):**
- B = the gap between still and marching under the existing terms, measured at the S1 weights.
- Target margins: M ∈ {1.0, 2.5}/s for ST1 and ST2. That is 5–12× the current ≈ 0.2/s.
  - ST2 is well above the pulls that failed (+0.76/s clock stance income; ≈ 0.95/s for action_rate −0.3 in W and C0).
  - ST1 is about the size of the second of those, so it runs only as a lower-dose check after ST2 works.
- w_slide = −0.5·M per (m/s). w_support carries the rest.
- Planning estimates: ST1 = (+0.38, −0.50), ST2 = (+1.12, −1.25).

**Not used:**
- A stand-gated action-rate term. Under σ ≈ 0.82 it is almost pure pressure on σ.
- Plain foot speed (KS). It rewards hovering.

## Steps

1. **Stand estimators, baselines, and the TG correction (CPU).**
   - New `scripts/p2s_stand.py`, one estimator set for every head and hold:
     - steps/s, and taps/s (from `step_tap_runs`);
     - **stance reference:** terrain-relative foot height (foot_z − terrain_z) minus the episode's stance offset.
       The offset is the median over loaded samples pooled across the six feet (the gait_eval `stance_z_offset`
       convention). A per-foot self-reference cannot see a foot that never lands. Every height clause uses this
       reference, never the force share.
     - **off-ground share:** foot samples more than 3 cm above the reference, over the steady window;
     - **hover episodes:** a foot both unloaded (< 50 N) and raised (> 3 cm) for > 95 % of the hold. Force alone
       flags the still statue in 71 of 100 episodes: a grounded foot can carry 0 N;
     - **clearance:** `p2_read.stand_clearance` (per-foot p95 − p5, median over feet), the statistic the 5-cm target
       and the 0.135 m baseline are on. The maximum over feet is reported only;
     - KS (Σ foot speed), loaded-foot count, body height above terrain, drift, facing error;
     - **onset latency**: time until the along-command speed (5-step mean) reaches 0.5·|cmd| and holds it for 0.5 s.
   - Baseline table: REF (es1 + es2), C0, CL, TG, A1, F1, E6. Cover suite stands, the field rows' opening stand,
     `field_turn`'s stand on walls, and onset baselines.
   - **Before the rule is frozen:** the E6 statue reads STILL under the §4 rule, and TG reads ≥ 9 hover episodes. The
     same check runs again on step 2's fresh statue.
   - Record the TG hover finding in CHARTER §11. Correct the manifest `stand_march` lead and the PHASE2_REPORT lead
     line.

   → sonnet · high (tool) / opus · high (read and records)

2. **Noise premise probe (≈ 0.4 GPU-h, current code, no code_sha move).**
   - New campaign row `stand_only__omni` (`scenarios_p2.yaml`). Run on it:
     - a fresh zero-action statue on today's code;
     - REF with `--sample-actions`;
     - a "noisy statue": a copy of model_19994 with the actor's output layer zeroed, written to
       `outputs/…/phase2/diag/` only, run with `--sample-actions`.
   - f = (T1 + T2 margin under sampling) / (deterministic margin).
     - **If f < 0.33, stop and ask**: the term would mostly push the global σ down, not change the mean.
     - Otherwise f·M is reported as the training-time margin.
   - These sampled runs feed this stop rule only, never a verdict.

   → sonnet · high to run; opus · high read

3. **Code: one commit (code_sha moves once).**
   - The T1 and T2 kernels.
   - Knobs in `config/crab_hex/motion_knobs.py`, following F1 (`KRABBY_STAND_STANCE_W` at :151 / :240 / :366 /
     :644-651 / :696-719): sign-checked parsers, `[KNOB] … consumed` lines, no slot latch, not EVAL_SEMANTIC.
   - Unset, the config is byte-identical.
   - Tool files, outside code_sha:
     - `run_candidates.py` `OVERLAY_DUMP_CHECKS`: slide → `_added_term_ok`; support → a twin of
       `_stand_stance_w_ok`.
     - `p2_live.py`:
       - liveness: the slide term within 3× of its replay magnitude; the support term's implied mean Σs_f in
         [2.5, 6];
       - generalise F1's stand-only AST check to the motion-rewards module;
       - a σ guard and a termination guard (risks 2 and 3).
     - `p2_read.py`: `--p2s-*` modes for the ladder, attribution and self-test.
     - New `p2s_queue.sh` (copy of `p2b_queue.sh`).
     - `candidates.yaml` v10: chains `P2__ST2_s18`, `P2__ST1_s18`, then the s16 / s17 replicates. Stage `p2s`, base 2a,
       5000 its, `{checkpoint}` fork of the S1 head, S1 overlay verbatim.
     - `scenarios_p2.yaml` `walkstop__omni` (flat): fwd 6 s → stand 12 s → lat 6 s → stand 12 s → back 6 s → stand
       12 s.
   - Unit tests: kernel arithmetic, gate, signs, config identity, every dump check, the p2s read, live and queue.
   - Then a fable review of the landed diff.

   → opus · high (code) / sonnet · high (tests); ✓ fable · high

4. **Replay gate (CPU), via a new `scripts/p2s_replay.py`.**
   - Data: deterministic stands from REF (es1, es2), C0, CL, TG, A1, F1 and the fresh statue; `field_turn` stands on
     walls; moving holds.
   - It imports the landed module (pure torch), with a fidelity check as in `p2b_replay`.
   - Checks:
     - **G1, statue separation:** AUC ≥ 0.95 against every head, for both terms.
     - **G2, T1 tracks lifting:** within-head ρ(T1/s, off-ground share) ≤ −0.5 on TG and CL, the heads with spread.
       - Off-ground share uses step 1's definition.
       - Keyed on off-ground share, not steps/s, because hover makes steps/s anti-track.
       - Disclosed as not blind: the audit measured TG −0.90 and CL −0.54 to −0.55 (near the line).
     - **G3, gaming counterfactuals:**
       - a still tripod hover costs ≥ 0.5·M more than the statue;
       - a six-down shuffle at 0.1 m/s per foot costs ≥ 0.25·M more than a still six-down stance;
       - a 1-cm crouch changes nothing.
     - **G4:** exactly zero on non-stand rows.
     - **G5:** finite on wall stands; the share of feet over pits is reported.
     - **Anti-termination:** the marching head's net stand-slot reward stays ≥ +0.5/s at both doses. Checked on the
       deterministic stands and on step 2's sampled REF stands, which are the training-time statistic.
   - Doses from the M rule. Magnitudes go to `replay_gate.json` "p2s" for liveness.
   - **A failed check stops and asks.**

   → opus · high; ✓ fable · high

5. **References on the new code (≈ 3.6 GPU-h).**
   - Fresh names `REFS_S1_s18` and `C0R_s18`. Reusing the p2b names would silently re-point the newest-stamp lookup.
   - Run every p2b read row, suite at es2 and `walkstop__omni`.
   - **Inertness:** every row shared with p2b reproduces its p2b read exactly; any difference stops and asks.
   - Re-point `P2_GUARD_REF`, re-run `SMOKES_OK`, and pass `--accept-code-change` once.

   → sonnet · high

6. **s18 screen: ST2 first (≈ 3.9 GPU-h); ST1 only if ST2 reads STILL (with or without COST).**
   - Live kills as in §8, plus three p2s guards:
     - **termination:**
       - **Kill:** episode length ≥ 0.9 × C0's same window. The minimum ratio over all 12 s18 sibling pairs is
         0.936. The §8 absolute collapse guard (rolling-200 `crab_failure` 0.6) stays as the hard backstop.
       - **Notify only (no kill):** rolling-500 `crab_failure` > C0's same window + 0.08 sets a flag, push-notifies
         and asks.
         - The sibling noise reaches +0.116.
         - 3 of 6 same-seed p2a pairs, and CL (mean difference ≈ 0), touch +0.08.
         - Same-seed reruns replay a trip, so a kill here would cost the arm on noise.
       - A unit test replays both clauses on every s18 sibling pair (X06 / X08 / X09 and X12 / X13 / X14, both
         orders): zero kills.
     - **σ:** `effective_std` ≥ 0.8 × C0's at +1000 and after;
     - **liveness** (step 3).
   - Every head gets the step-5 reads plus `p2s_stand`.
   - **Stop and ask** with the s18 verdicts. ST2 reading NO EFFECT or GAMED ends the screen at ≈ 8 GPU-h.

   → sonnet · high to run; opus · high read

7. **Replicates: the kept dose at s16 and s17**, after `REFS_S1_s16` and `REFS_S1_s17` on the new code (≈ 11 GPU-h).
   - These seeds have no control, so they are labelled "lever + continuation".
   - **Stop and ask** with the recipe read: the §6.2 rules applied to the stand verdicts.

   → sonnet · high

8. **Head proposal and write-up.**
   - A p2s head is proposed only if it reads STILL with no COST and attribution holds. PARTIAL results go to you to
     decide.
   - REPORT answers: did the march stop, by which route, and at what walking or wall cost.
   - The bake is yours. A bake goes in as `policy/7_p2s_stand` plus the records.

   → opus · high; ✓ fable · high

## Pre-registered verdicts (per head, against REFS on the same seed; first match decides)

1. **INVALID EVAL / PENDING / HAND LOSS / DOSE**, as §10. The walkstop stamp and suite at es2 are required.
2. **FLAT FAIL**, as §10, against REFS and chained against ANCW.
3. **OBSTACLE LOSS** if any of:
   - field_hard (n = 400) < REFS − max(0.05, 2·SE);
   - field_light < REFS − max(0.03, band);
   - ring_obst < REFS − max(0.05, band).
4. **Stand outcome.** Bands come from REFS es1 vs es2, with floors: off-ground 0.02, steps/s 0.25, taps/s 0.25,
   KS 0.10 m/s, onset 0.1 s.
   - **STILL**, on suite es1 and es2, if all of the clauses below hold. The walkstop stands feed COST (falls, onset)
     and report their stillness; they are not part of STILL (your decision).
     - median steps/s ≤ 0.5;
     - median clearance (`stand_clearance`) ≤ 0.05 m;
     - off-ground share ≤ 0.10. 0.05 would cap REF-length steps (≈ 0.9 s airborne) at ≈ 0.33 steps/s, below the
       0.5 target;
     - hover episodes (step-1 definition) ≤ 5/100;
     - taps/s ≤ 1.0;
     - KS ≤ 0.5 m/s.
   - **PARTIAL**, on both suite seeds, if all of:
     - off-ground share and steps/s each ≤ 0.5 × C0R's (at s16 and s17, against REFS);
     - hover ≤ C0R's + 5;
     - KS ≤ C0R's + band.
   - **GAMED**: steps/s ≤ 0.5 but a PARTIAL off-ground, hover or KS clause fails. A flag only; it never adopts.
   - **NO EFFECT** otherwise.
5. **COST**, on STILL or PARTIAL, if any of:
   - a §6.1 COST margin is exceeded;
   - pooled stand-hold falls (suite, walkstop, field opening stands, field_turn stand) > REFS + max(0.03, 2·SE);
   - onset latency > REFS + max(0.3 s, band);
   - stand body height is more than 0.03 m off REFS;
   - field_turn completion < REFS − max(0.05, band).

**Attribution (s18):** the effect belongs to the lever if:
- ST reads STILL or PARTIAL;
- C0R reads NO EFFECT;
- there is no FLAT FAIL against C0R;
- field_hard ≥ C0R − max(0.05, 2·SE).

**Reported only:**
- σ at the fork, +1000 and the end (per cam and rod);
- `steps_per_rad_post`, i.e. stepping after a turn closes;
- walking tripod, steps/m and apex against C0R;
- the share of stand actions at the clip.

## Budget

| Item | GPU-h |
|---|---|
| Noise probe (statue, sampled REF, sampled noisy statue) | ≈ 0.4 |
| REFS_S1_s18 + C0R_s18 on the new code | ≈ 3.6 |
| ST2 s18 (+ ST1 if it reads STILL) | ≈ 3.9 (+ 3.9) |
| Replicates: REFS s16/s17 + 2 arms | ≈ 11.2 |
| **Total** | **≈ 23** (≈ 1 day); ≈ 8 if ST2 fails |

Steps 1, 3, 4 and 8 are CPU only.

## Quiet-failure risks and guards

1. **Hover, shuffle or a light touch passes as "still".**
   - T1 prices airborne feet; T2 prices sliding loaded feet.
   - G3 tests both routes.
   - The read needs off-ground share, hover count, taps/s and KS; GAMED is a separate class.
2. **The stand slot becomes falling.**
   - T1 is an income with an offset.
   - The anti-termination replay check and the termination live guard.
   - Stand-fall and walkstop reads, including stands on walls.
3. **Global σ collapse, or a mean saturated to kill the noise.**
   - The f stop rule and the σ guard.
   - The DOSE rule (|μ| ≤ 1.5), with the clip share reported.
4. **Collateral damage:** slower start from stand, a crouch, walking-gait drift (a settled turn also reads a zero
   command).
   - COST on onset and body height, FLAT FAIL, OBSTACLE LOSS.
   - The post-turn and walking reads.
5. **The C0 / REF comparison breaks after the code move.**
   - Config-identity test; REFS and C0R reproduce their p2b reads exactly.
   - Fresh names, the guard re-pointed, fresh chain names after any kill.

## Verification

- **Unit tests:**
  - kernels: six-down +3, tripod hover 0, statue-like +2, sliding loaded foot priced, zero off the stand slot, zero
    when tipped;
  - knob parsers and signs;
  - config identity when unset;
  - each dump check, on the real S1 dump (off passes, armed fails);
  - p2_live liveness and guards;
  - p2_read ladder self-test (STILL / PARTIAL / GAMED / COST cases);
  - queue order and stops.
- **Before the screen:** the dry run of every p2s chain through the driver validator, walkstop accepted by the
  validator, the replay gate PASS and the inertness check.
- **Each arm:** resume read, the +200 wiring / liveness / dump audit, and INVALID EVAL on every stamp.
- Every verdict goes to CHARTER §11 and the CHANGELOG, and is push-notified.

## Left for implementation (checks)

- **Liveness bands:**
  - Calibrate both bands from step 2's sampled vs deterministic REF. Training runs at σ ≈ 0.82; p2b's factors were
    2.1–2.9×.
  - Keep §8's non-zero requirement beside the Σs_f band; a dead T1 implies Σs_f = 3.
  - Pin the stand-slot share used (SLOTS 0.15).
- **Onset COST** uses the median (es1 / es2 noise ≤ 0.02 s); p90 noise reaches 0.24 s.
- **Step 2:**
  - The f denominator comes from a deterministic REF run on the same `stand_only__omni` row and window.
  - Name the T1:T2 weighting f uses (ST2's planning dose).
  - Say in the stop message that the open-loop noisy statue makes f a lower bound.
- **At the step-6 stop,** show C0R's own clause values beside each arm's, so continuation drift is not read as lever
  cost.
- **Walkstop estimators:** state the window (steady vs full hold), the episode count and the env_seed.
- **Step 7:** map STILL / PARTIAL / COST onto the §6.2 recipe rules explicitly in `p2_read`.
- **TG correction:** cite TG's median T1 ≈ +0.06 and its unloaded-and-raised hover episodes, not only the
  lifted-before-the-window mechanism.
- **Read alongside steps:** read steps / taps beside clearance and off-ground. A foot held near 50 N can log steps
  without lifting.
- **s16 / s17 guards** have no same-seed C0. Use each seed's own p2a source tail as the reference; a cross-seed C0
  would trip at once (s17 crab_failure tail 0.29 vs 0.15–0.19).
- **G2 population:** pin it to suite stands, steady window. TG (−0.91) is decisive; CL (−0.53) sits at the line, and
  a flip only stops and asks.
- **M reference:** measure M against the statue's measured T1/T2 (+2.0, static indeterminacy), not six-down +3.
  Write the solved weights to `replay_gate.json`.
- **f stop message:** add the noisy statue's own falls and Σs_f, so an open-loop artefact is not read as σ pressure.
- **Reads that always show:**
  - the stand outcome, even when an earlier rung (FLAT FAIL / OBSTACLE LOSS) matched;
  - the onset delta beside its label (the 0.3-s floor is ≈ 3× REF's median);
  - a note that PARTIAL can come from the clearance clause alone. A median-over-feet clause binds before steps/s
    when 3+ feet each take one full-height step.
- **Attribution's "C0R reads NO EFFECT"** means "C0R is not STILL".
- **Rolling-500** is undefined before +500. Wait, as the collapse guard does.
- **Anti-termination** is also reported on the deterministic `field_turn` wall stands.
- **TG correction** to the manifest and PHASE2_REPORT goes in its own record commit, with your nod at the step-1
  stop.

## If ST2 reads NO EFFECT

Stop and ask. The documented next lever is the regime route: a stand bit in the observation plus the stand-gated set,
trained from scratch. That is not in this plan.

## Out of scope (not run)

- TG replicates.
- A stand-share change (C8).
- STAND_HOLD / PARK.
- The stand bit (from scratch).
- Push-recovery evals.
- Move_turn, back overshoot and the wall deficit (separate follow-ups).
