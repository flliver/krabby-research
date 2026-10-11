# Plan: Z2d, a stand-row input mask + fine-tune (p2r amendment 4, CHARTER §12)

> **Audit trail (cap 1, reached).**
> - **Pass 1** (opus): one blocking finding. Every pre-training check was open-loop, and the model_20500 HOLD gate is
>   blind to the mask (Z2c 0.0204 / 0.0200 unmasked / masked). Fixed by step 2b: a mask-only closed-loop control of
>   Z2c before training.
> - **Other fixes applied:**
>   - LIFT / HOME reported against their physical lines (0.30 / 0.25) with Z2c's curve as the expectation;
>   - adoption criteria copied verbatim;
>   - the gate records `masked`, and an unmasked observe bank is refused.
> - **Checked and fine:** the premise-table numbers match the probe outputs. The mask is consistent across PPO
>   storage, critic, mirror, prior and cloning rows. The crouch bank does not confound the reads. Every read stamp is
>   covered by the INVALID rule.
> - **These fixes were not re-audited.**

> Routing: roster ok · ceiling: opus · fable checks on steps 1 and 5.
> It amends `P2R_Z2C_PLAN.md`. Same stand bit, graft fork, walking anchor, statue prior to R, R's constants, cloning
> term and ladder. It adds a stand-row input mask, trains under it and reads the result closed-loop.

## Context

**Z2c failed closed-loop (CHARTER §12.1, 2026-10-10).** The cloning worked where it was trained: on held-out R-driven
stands the final checkpoint matches R to hip 0.007 / cam 0.022. The real stand still fails:
- homed 0 %, 17–28 re-steps per stand;
- about 50 taps/s, body 2.6 cm low.

Walking is intact (walls 0.8425 against Z1 0.83).

**Diagnosis (observe pass + fable·high check, both confirmed):**
- **Two inputs carry a feedback loop:** the contact proxy (cols 83–88) and the network's own last action
  (cols 65–82).
- On Z2c's own stand states, R says HOLD on 96 % of leg-rows, but Z2c holds a crouch (hip −0.08 to −0.115).
- **Masking both inputs** (proxy → parked −0.5, last action → 0) brings that to +0.003; each alone only halves it.
- **But the mask alone kills stepping.** Masked, LIFT and HOME go to hip ≈ 0 where R lifts, because the network reads
  its stepping phase from its own last action.

**Your decision (2026-10-10): Z2d = mask + fine-tune**, so the phase is learned from leg state, as R reads it.

**Premise probe (offline, CPU; `scratchpad/z2d_premise.py`):** the GRAFT actor is fitted to R on the masked Z2c
banks and scored on held-out masked rows (eval path).

| held-out rows | fit at 4k steps | fit at 12k | fit at 12k, + Z2c suite crouch bank | Z2c masked, no fine-tune | GRAFT |
|---|---|---|---|---|---|
| walkstop es2, all-HOLD (hip / cam) | 0.006 / 0.012 | 0.009 / 0.011 | 0.008 / 0.017 | 0.007 / 0.022 | 0.53 / 0.54 |
| walkstop es2, LIFT hip \|μ − R\| | 0.264 | 0.204 | 0.187 | 0.467 | 1.0 |
| walkstop es2, HOME hip / cam | 0.12 / 0.25 | 0.09 / 0.15 | 0.09 / 0.13 | 0.45 / 0.40 | 1.0 / 0.40 |
| Z2c walkstop crouch, all-HOLD hip | 0.080 | 0.073 | **0.045** (held out) | 0.016 | 0.49 |

- **The phase is learnable from state:** LIFT falls below 0.25. That line is physical: a LIFT residual above 0.25
  leaves the rod short of R's "up" threshold (−0.0625 = 0.25 × −0.25), so the leg would never home.
- **Clean banks alone do not fit Z2c's crouch states** (0.07–0.08). Adding the suite crouch bank brings the held-out
  walkstop crouch to 0.045.

## Design

**The stand-row mask (env side).**
- Knob `KRABBY_OMNI_STAND_MASK=1` sets a new omni obs-term param `stand_mask`. It is refused without
  `KRABBY_OMNI_STAND_OBS=clock` and with `KRABBY_OMNI_STAND_PROXY=hold` (both write cols 83–88).
- **Which rows:** the current step's stand rows (the stand bit).
- **What it sets, in the RETURNED observation:**
  - last-action cols 65–82 → 0;
  - proxy cols 83–88 → −0.5;
  - the same two blocks of each of the 10 real history frames → 0 / −0.5. The frames are 89 wide at cols
    259 + 89j; the actor's own 75-wide history view is misaligned with them, so the mask follows the real frames.
- **The history part is defence in depth.** The history latent is constant on this fork: masking the history moves
  the eval μ by ≤ 4e-6, so the premise numbers stand.
- **What it leaves alone:** walking rows are untouched, and the stored history buffer is never written. A rule on the
  returned tensor only, like `stand_proxy_hold`, is bitwise inert on walking rows and when off.
- **One pure function** `stand_mask_obs(obs, rows, num_prop, num_hist)` in `mdp/crab_hex_omni_restance.py`
  (loaded by path, pure torch) is shared by:
  - the env;
  - the cloning bank loader;
  - the fidelity scorer and gate;
  - the observe read.
- **Mirror-invariant:** the constants are 0 and the same −0.5 on every leg.
- **Mask liveness:** the obs term logs `Metrics/stand_mask_frac`, the share of rows masked this log window.
- **Deployment contract if adopted:** `STAND_OBS=clock` + `STAND_MASK=1`.

**Training (one chain, `P2__Z2d_s18`).**
- It is the `P2__Z2c_s18` chain plus:
  - `KRABBY_OMNI_STAND_MASK=1`;
  - the cloning banks masked on load: a `bc_stand_mask` flag, set by the same knob when cloning is armed;
  - **one extra cloning bank**, Z2c's closed-loop suite observe bank (crouch states, R-labelled; premise row 3).
- It forks GRAFT, as Z2 / Z2c, for 5000 its.
- The walkstop crouch bank and the RSTg es2 banks stay held out.

**The evals must carry the mask (quiet-failure guard).**
- `run_phases.EVAL_CARRY` is eval-locked, so the knob joins `EVAL_CARRY_EXTERNAL`. It reaches evals through
  run_arm's env (stamp env vars + `P2_ENV_EXTRA`).
- **The training-end driver suite eval builds WITHOUT it.** So:
  - suite es1 is re-read via run_arm with the mask;
  - `p2_read` refuses (INVALID) any stamp of a masked head whose `run_meta.env_vars` lacks `STAND_MASK=1`.
- The observe bank proves the mask on stand rows directly: banked obs cols 65–88 equal the constants.

## Steps

1. **Code, one commit** (code_sha moves once).
   - **The mask:**
     - `stand_mask_obs` + tests (only stand rows change; inert when off; mirror invariance; the bank and env paths
       agree);
     - the sibling of `KRABBY_OMNI_STAND_PROXY` everywhere it is wired:
       - `motion_knobs.py`: parser, `MOTION_KNOB_PARSERS`, `__all__`, `EVAL_SEMANTIC_KNOBS`, `EVAL_CARRY_EXTERNAL`,
         `KNOB_OFF_VALUES`, cross rules, `_set_obs_param`;
       - the obs term: class attribute, check, `__init__`, the `__call__` keyword, `stand_mask_frac`;
       - `run_candidates.py`: `OVERLAY_DUMP_CHECKS` + log audit;
     - `bc_stand_mask` (`PPOWithExtractor._load_bc_bank`, after `restance_target`; `rc.BC_ALG_FIELDS`; set by
       `motion_knobs` when the knob and cloning are both on);
     - the test pins that list the knobs: `test_crab_hex_omni_motion_knobs` `PHASE2_KNOBS`, and `test_omni_p2r_z2_core`
       `EVAL_CARRY_EXTERNAL`.
   - **The gate:** `p2r_z2c_gate.py` masks the bank (current step + history) when the run's `params/env.yaml` carries
     `stand_mask` (or under a forcing flag); its JSON records `masked`.
     - The hip / cam all-HOLD lines (0.05) still kill. Masking does not change this clause (Z2c 0.0204 / 0.0200 at
       20500); it stays as the cloning-health check (GRAFT reads 0.53 masked).
     - **Reported, against physical lines:**
       - LIFT hip |μ − R| ≤ 0.30 (a leg must leave DOWN at −0.05);
       - HOME hip ≤ 0.25 (a leg must reach UP at −0.0625);
       - the held-out walkstop crouch bank.
     - **Expected curve:** Z2c's own held-out LIFT, with its last-action cue: 0.33 (+500), 0.27 (+1000), 0.24 (+2000),
       0.18 (end).
   - **The observe read** refuses a bank whose stand-row cols 65–88 are not the mask constants when the head is
     masked.
   - **The arm wiring.** Without it a Z2d chain silently falls through to Z1's checks and gets no gate:
     - p2_live gets an explicit `Z2D` arm (dispatch, preflight, `ep_len_notify`): Z2C's checks + `stand_mask_frac`
       in the prior-rows band + the overlay has the mask on;
     - run_arm runs the Z2c gate and its launch refusal for `P2__Z2d*` too;
     - p2_read: `z2_env_reasons(want_mask)` from the chain overlay, the widened INVALID filter, and
       `z2_suite_stamp` skipping the driver stamp for a masked head.

   → opus · high (code) / sonnet · high (tests); ✓ fable · high (diff review)

2. **Short bridge (≈ 0.2 GPU-h).** REFG / a new reference with the knob off, bitwise = REFS on suite, field_hard es1
   and walkstop; append the pair.

   → sonnet · high

2b. **Mask-only closed-loop control (≈ 0.2 GPU-h; audit fix; before candidates v17).** Every other pre-training check
   is open-loop, and Z2c failed exactly at open-loop → closed-loop.
   - **Run:** Z2c model_24993 through run_arm `--row-eval --p2r-observe` on suite es1 + walkstop es1, with
     `P2_ENV_EXTRA` carrying `KRABBY_OMNI_STAND_MASK=1` (appended, never replacing the rest).
   - **Continue only if all hold:**
     - suite es1 is STILL (p2_read's clauses: steps ≤ 0.5/s, taps ≤ 1/s, KS ≤ 0.5);
     - suite body height is within 0.01 m of the probe's 1.065;
     - the observe all-HOLD hip bias is ≤ 0.02 on both rows;
     - the banked stand-row cols 65–88 equal the mask constants (a refusal, not a report).
   - **Otherwise STOP and ask:** another input carries the loop.
   - **Expected:** walkstop does NOT home, because masked LIFT / HOME read 0.47 / 0.45. Its homed share and lifts are
     the mask-only baseline that step 5 reports Z2d against.

   → sonnet · high (run) / opus · high (call)

3. **Pre-launch (CPU), then candidates v17.**
   - **Dose:** the GRAFT `Loss/bc_mb0` on the masked seven-bank set gives the dose band (× [0.52, 2]).
   - **Gate controls through the exact gate path, masking forced:**
     - GRAFT: expected FAIL;
     - the offline masked fit (premise, 12k steps): expected PASS.
     - Both JSONs must record `masked = true`.
   - **Candidates v17:** `P2__Z2d_s18`; then preflight.

   → opus · high

4. **Train Z2d at s18 (≈ 2.2 GPU-h), fork GRAFT, 5000 its.**
   - **+200 wiring:** Z2c's checks + mask liveness.
   - **FIDELITY GATE at model_20500**, as Z2c: held-out masked walkstop es2, eval path, all-HOLD hip / cam ≤ 0.05.
     LIFT and the crouch bank are reported (+500 is early for the stepping branches; the closed loop decides).
   - The kill goes through run_arm's rc-3 path.

   → sonnet · high (run) / opus · high (gate)

5. **Reads, closed loop, then STOP.**
   - The Z2 read set with the mask, suite es1 re-read via run_arm, plus an observe pass (suite + walkstop es1).
   - **Adoption-grade if all hold** (verbatim from `P2R_Z2C_PLAN.md` step 6):
     - STILL on suite es1 + es2;
     - homed ≥ 90 % and settled ≥ 90 % of walkstop stands at both seeds;
     - median settle ≤ the probe's + 1 s;
     - post-settle steps ≤ 0.5/s, taps ≤ 1/s, KS ≤ 0.5;
     - stand falls ≤ max(1, Z1's) per 300;
     - no OBSTACLE LOSS / FLAT FAIL / COST; attribution LEVER against C0R.
   - **Reported:**
     - R-agreement on Z2d's own stand states (observe);
     - six-down;
     - walls against Z1 and Z2c;
     - walkstop homed / lifts against the step-2b mask-only baseline.
   - Fable check, then **STOP and ask.** Z1 stays the stand of record.

   → opus · high decides; ✓ fable · high

## Budget

| Item | GPU-h |
|---|---|
| Bridge | ≈ 0.2 |
| Mask-only closed-loop control (2b) | ≈ 0.2 |
| Training | ≈ 2.2 |
| Reads + suite es1 re-read | ≈ 1.8 |
| Observe pass | ≈ 0.2 |
| **Total** | **≈ 4.6** |

Steps 1 and 3 are CPU only.

## Quiet-failure risks and guards

1. **An eval without the mask** (the driver suite, a ladder row) silently reads the unmasked policy. Guard: the INVALID
   rule, the suite re-read, and the observe-bank proof.
2. **The mask is not live in training** (knob dropped, wrong rows). Guard: `stand_mask_frac` at +200, plus the
   unit tests.
3. **Under-lift:** the leg never reaches "up", so it never homes. Guard: LIFT (0.30) and HOME (0.25) are reported at
   +500 against their physical lines and Z2c's curve; homed % decides at step 5.
7. **The mask breaks the loop open-loop but not closed-loop** (Z2c's failure mode). Guard: step 2b, before training.
4. **Walking changes:** the mask is on stand rows only. Guard: the walking anchor, drift at +500, the ladder.
5. **Bank / env mask mismatch:** one shared function, tested both ways.
6. **The Z2d chain silently runs as a generic Z arm** (Z1's checks only, no fidelity gate). Guard: the explicit `Z2D`
   arm + `is_z2d` in run_arm, with tests that `P2__Z2d_s18` maps to them.

## Out of scope (not run)

- DAgger rounds beyond the one Z2c crouch bank.
- Oversampling the stepping rows (premise LIFT 0.204 → 0.178 at 12k; left out, say if you want it).
- Replicates and the bake.
