# Plan: learn the statue in-network (p2r step-2 amendment, CHARTER §12)

> **Audit trail (cap 1, reached).**
> - **Pass 1** (a premise probe, then opus · xhigh + fable · high; the opus auditor added a CPU proxy of the update
>   dynamics). Blocking fixes:
>   - the early statue check is re-based on the fork value (formation finishes before +200), with a residual-plateau
>     read;
>   - the loss normalisation is stated once (sum over masked rows ÷ B), with a realised-dose assert at +200;
>   - the loss-value dose check is dropped (it cannot fail), and dominance is restated as formation-only, with
>     gradient-norm logging.
>   **These fixes were not re-audited.**
> - **Premises verified:** statue loss 19.4 per stand row (2.3 per batch); stand share 0.12; recorded μ is pre-clip;
>   the critic has no shared layers; every update is already grad-clipped.

> It amends `P2R_PLAN.md` step 2 (G2/G1, ST2 stand terms), which step 1's stop replaced.
> Routing: roster ok · ceiling: opus · fable checks on steps 1 and 5.

## Context

**Where step 1 left us (CHARTER §12.1).** A controller-level stand mode works with no training:
- The S1 head walks, and at the commanded stand slot the controller outputs **zero actions** (the statue).
- It reads STILL on every clause at both suite seeds: 0 steps, 0 taps, 5 loaded feet.
- It has no cost:
  - field_hard 0.810 against REFS 0.8125 (n = 400);
  - flat deltas ≤ 0.009;
  - walkstop onset 0.12 s against 0.14;
  - 0 falls in stand or at handover;
  - field_turn wall stands 0.99.
- Step 1's drift premise fired: ordinary continuation (C0) moves walking about 60 % as far as ST2's stand lever. So
  an anchor dosed to single out ST2's harmful drift was off.
- ST2's stand terms chatter (16 taps/s).

**Your decision (2026-10-09): learn the statue in-network.** One network should behave like the hold controller, so
deployment needs no policy switch, only the stand bit as an input.

**The approach: distil the proven controller with two masked supervised losses on the action mean.**
- The grafted stand bit (column 15, from P2R_PLAN step 4) lets the network tell stand from walking.
- **Statue prior:** on stand rows (bit = 1), pull μ toward 0.
- **Walking anchor:** on every other row (bit = 0), hold μ at frozen S1's μ.
- Here the anchor's job is to keep walking at S1. Ordinary continuation drift had no measured gain (C0's walls 0.785
  were within noise of REFS), so step 1's premise, which concerned telling harmful drift from harmless drift, no
  longer applies.
- Both terms act on μ, not on sampled actions, so exploration noise (σ ≈ 0.8) does not dilute them. That is the
  failure mode that weakened the stand rewards (f ≈ 0.3).
- ST2's stand terms are not used.

**Scale (measured on REFS raws):**
- At the fork the statue loss ½Σ(μ/σ)² is 19.4 per stand row, so about 2.3 as a batch mean at the 0.12 stand share.
- **Dominance holds during formation only.** A CPU proxy of the update (resumed Adam, LR at its 1e-5 floor, clip
  1.0) suggests the statue forms within about 100–200 iterations.
- After that, the stand residual and walking drift settle where RL meets the quadratic restoring forces.
- At stand, RL optimises the noisy sampled statue (19 taps/s under σ, CHARTER §11.1), so it may push μ off 0. The
  residual-plateau read (step 4) watches this.

**Unchanged:**
- target "mostly still" (STILL, all six clauses including taps);
- no walking or wall cost beyond noise;
- the p2s ladder;
- the step-1 tools (launcher, banks, drift scorer);
- ground rules (never push; one Isaac process; eval_sha fixed; fresh chain names after a kill; adoption, bake and
  push are yours).

## Steps

1. **Code, one commit (code_sha moves once).** These are P2R_PLAN step-4 items, kept as audited, plus the statue
   prior.
   - **`KRABBY_OMNI_STAND_OBS=clock`:**
     - writes `stand_mask()` (the latched slot) into the current step's column 15 only, never history;
     - requires `CLOCK_HIDE=1`; arms the eval latch; mirror `lr_signs[15] = +1`;
     - re-derives the latch on episode reset (the suite-wave carry-over);
     - adds parser and dump check.
   - **Two masked μ losses in `parkour/scripts/rsl_rl/modules/ppo_with_extractor.py`.** Both go after the bound loss
     (`:781-788`) and before the symmetry block (`:793`). Both are computed on the full minibatch, then masked, so
     the fork values are exact.
     - **Anchor** (bit-0 rows): ½Σ_d((μ_θ − μ_A)/σ_A,d)², μ_A from a frozen deepcopy of the graft's actor
       (`hist_encoding=False`).
     - **Statue prior** (bit-1 rows): ½Σ_d(μ_θ/σ_A,d)².
     - σ_A is fixed (the graft's soft std), so neither term pushes σ.
     - **Normalisation, stated once:** each loss is the sum over its masked rows divided by the minibatch size B.
       Every row then carries weight 1 in KL units, the same as the surrogate's per-row weight. Fork values: statue
       ≈ 2.3, anchor exactly 0.
     - Every ~50 iterations it logs ‖grad statue‖, ‖grad anchor‖ and ‖grad rest‖.
     - Knobs `KRABBY_ANCHOR_W`, `KRABBY_ANCHOR_CKPT`, `KRABBY_STAND_PRIOR_W`, all agent-side. Refused unless
       `STAND_OBS=clock`, or with a negative weight.
     - Logs `Loss/anchor`, `Loss/stand_prior`, `anchor_rows`, `prior_rows` and the fork-exactness `anchor_mb0`.
   - **Graft tool `C/scripts/p2r_graft.py`:**
     - zeroes col 15 of `actor_backbone.0`, `estimator.0` and `critic.0`, plus their Adam moments (the estimator's
       state is in `alg_state["estimator_optimizer"]`);
     - leaves everything else bitwise identical;
     - writes `graft.json`.
   - **Tooling:**
     - `verify_head_sha` graft branch, and `fork_source_rd` resolves to the S1 run;
     - the arm pattern accepts `P2__Z[0-9]+`;
     - `p2_live` +200 wiring:
       - `anchor_rows` ≈ 0.88 in [0.8, 0.95] and `prior_rows` ≈ 0.12 in [0.05, 0.2];
       - `anchor_mb0 == 0`, read at the first logged iteration;
       - both losses finite and > 0 after the first update;
       - **realised dose:** the first-update `Loss/stand_prior` in [1.2, 4.6], and the realised weights in the dump.
     - the post-run clock check is narrowed (col 16 ≡ 0, col 15 = the stand label);
     - candidates v12 chain `P2__Z1_s18`: stage p2r, base 2a, 5000 its, fork = the graft, the **S1 overlay
       verbatim** (no ST2 terms) + `STAND_OBS: clock` + the three loss knobs.
   - **Tests:**
     - config identity with the knob unset / 0 / 1;
     - history col 15 ≡ 0;
     - each loss's gradient comes only from its own rows;
     - weights 0 ⇒ loss bitwise unchanged;
     - graft identity (`torch.equal`, actor on both paths, estimator, critic);
     - dump checks.
   - Then a fable review of the landed diff.

   → opus · high (code) / sonnet · high (tests); ✓ fable · high

2. **Graft identity and inertness (≈ 0.45 GPU-h).** As P2R_PLAN step 5.
   - `GRAFT_S1_s18` with the bit on: suite, field_turn and walkstop at es1 reproduce REFS bitwise (excluding the raw
     `clock_obs` series and env vars). `clock_obs` is exactly 1 on settle and stand steps.
   - Inertness bridge under a fresh name, knob off, on the new code: suite, field_hard es1–4, walkstop and field_turn
     bitwise equal to the ae892cec stamps. Otherwise run the full REFS + C0R re-evaluation (≈ 2 GPU-h).

   → sonnet · high

3. **Dose (CPU).**
   - `KRABBY_STAND_PRIOR_W = KRABBY_ANCHOR_W = 1.0`: KL units, per-row weight 1 under the step-1 normalisation.
   - Record the expected fork values (statue ≈ 2.3, anchor 0) for the +200 realised-dose assert.
   - Dry run of the chain through the validator.

   → opus · high

4. **Train Z1 at s18 (≈ 3.9 GPU-h with reads).**
   - Live kills as in §8, plus the p2s guards (σ, episode-length kill, notify-only crab_failure) and the +200 wiring.
   - **Early checks** (CPU, notify-only, at model_20500 and model_21000; own sentinel):
     - **walking drift** D(Z1) on the step-1 non-stand bank should be ≤ ½·D(C0) at the same iteration (C0 1.35–1.42
       at +506, 1.63–2.02 at +1006). Otherwise the anchor is not holding walking;
     - **formation:** rolling-20 `Loss/stand_prior` at +200 ≤ 0.1 × its first-update value. Otherwise the stand is not
       forming;
     - **residual:** report the +500 and +1000 plateau (rolling-50, as per-stand-row loss and |μ|). Notify if it rises
       more than 2× above its +200..+500 minimum, meaning RL is pulling the stand back toward marching.
     - Reported beside them: `PPO/kl_mean`, the applied LR, `PPO/grad_norm`, `Loss/value_function` and the
       gradient-norm split. Expect high KL and clip fraction for the first ~100 iterations; that is not
       instability.
   - On a trip, notify and ask; the run continues.

   → sonnet · high to run; opus · high on the early checks

5. **Pre-registered reads, then STOP.**
   - Z1 on the p2s ladder against REFS_S1_s18 (C0R as the PARTIAL comparator), at the full p2s read set: suite
     es1 + es2, walkstop es1 + es2, field_hard es1–4 (n = 400), field_light, ring_obst, field_turn, field_suite, lr.
   - **Adoption-grade if all hold:**
     - STILL on both suite seeds (all six clauses);
     - no OBSTACLE LOSS / FLAT FAIL (against REFS and chained against ANCW) / COST;
     - attribution LEVER against C0R.
   - **Fidelity to the hold controller (reported):** Z1's stand clauses beside P2R_HOLD's; Z1's walking drift D;
     taps and jitter.
   - **If Z1 reads PARTIAL or COST: stop and ask.** The hold controller stays the working fallback.
   - G-heads' INVALID rule: stamps must carry `STAND_OBS=clock`, and the raw bit must match the stand labels (with
     the one-step lag allowed).

   → opus · high decides; ✓ fable · high

6. **Write-up and proposal.** CHARTER §12 and REPORT.
   - Replicates (s16 / s17) and the bake are yours.
   - The deployment contract: col 15 = 1 exactly while commanded to stand.

   → opus · high

## Budget

| Item | GPU-h |
|---|---|
| Graft identity + inertness bridge | ≈ 0.45 (≈ 2.4 if the full re-eval is needed) |
| Z1 (5000 its + full reads) | ≈ 3.9 |
| **Total** | **≈ 4.4** (≈ 6.3 worst case) |

Steps 1, 3 and 6 are CPU only.

## Quiet-failure risks and guards

1. **The graft is not identical, or the bit leaks into history.**
   - Guard: CPU `torch.equal`; history ≡ 0 test; GPU bitwise rows; `anchor_mb0 == 0`.
2. **Either loss masks the wrong rows.**
   - Guard: per-loss gradient-mask tests; the `anchor_rows` / `prior_rows` bands at +200; cross-knob refusal.
3. **The statue is learned but bleeds into walking anyway.**
   - Guard: the anchor; the early walking-drift check against C0; the full ladder.
4. **The stand bit is wrong in an eval.**
   - Guard: `STAND_OBS=clock` is already carried; `P2_ENV_EXTRA`; the raw-bit INVALID rule.
5. **The supervised terms overwhelm RL and degrade recovery.**
   - Guard: the ladder's falls and COST clauses, and walls at n = 400. The hold controller (zero RL at stand) already
     showed that a pure statue keeps falls at zero on these rows.

## Left for implementation (checks)

- **`p2_live` p2s mode** maps arm names to stand-term doses and needs the replay gate's weights. Z arms have no stand
  terms: wire only the σ, episode-length and crab_failure guards plus the new wiring.
- **The +500 sentinel** must fire on G/Z's `model_20500` (+506, matching C0's and ST2's drift points).
- **Early drift check:** also report the cosine of Z1's Δμ against ST2's Δμ on the same bank (direction, not just
  magnitude).
- **Training stands are the noisy sampled statue,** so the episode-length kill or the crab_failure notify could fire
  on a working arm. If either fires, read stand-slot terminations separately before acting.
- **Carry P2R_PLAN's checklist:**
  - the critic's col-15 write is current-step only;
  - do not copy slot 5's history re-patch;
  - map estimator Adam indices before zeroing;
  - re-derive the latch on reset;
  - capture the true priv columns for any bank;
  - allow the raw-bit one-step lag.
- **Step 5's fidelity table** includes the walkstop stands' steps/s and taps beside P2R_HOLD's (residual μ after
  walking is the one row the hold never tested).
- **Keep the inertness bridge's row list in `inertness_bridge.json`,** so the ladder's accept rule names the rows it
  extrapolates.

## Verification

- Unit tests and the fable review (step 1).
- The bitwise identity rows and inertness bridge (step 2).
- The dose premise (step 3).
- The +200 wiring and the early checks (step 4).
- INVALID on every stamp.
- Each verdict goes to CHARTER §12 and is push-notified.

## Out of scope (not run)

- A no-anchor control arm (statue prior only).
- The ST2-term arms G2 / G1.
- Replicates and bake.
- Runtime/deploy code.
- Push-recovery evals.
- From-scratch STAND_OBS.
