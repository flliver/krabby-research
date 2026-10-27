# Plan: Z2c, clean-state cloning for the in-network re-stance (p2r amendment 3, CHARTER §12)

> **Audit trail (cap 1, reached).**
> - **Premise probe** (offline BC on the RSTf banks, CPU): **the premise holds.**
>   - Held-out es2 eval-path all-HOLD hip / cam bias is 0.007–0.012 against the 0.05 lines (Z2: 0.13–0.18).
>   - **Condition:** BC must not move the priv-encoder latent. Its input is constant on every row, so BC uses it as a
>     free bias that the eval-path history encoder never sees. With it trainable, eval-path hip is 0.20–0.25; frozen,
>     eval = train to ≤ 0.001.
>   - The auditor's CPU replay with priv_reg at 0.1 (as from this fork) keeps the gap closed. Detaching is adopted
>     anyway: harmless, and it removes the risk.
> - **Pass 1** (opus · xhigh). Blocking fix: the gate's held-out set was unpinned. Suite es2 is a near-duplicate of
>   the BC-trained suite es1 bank (100 % all-HOLD, cams 0; held-out loss 0.0008), so pooling diluted the gate about
>   2.4×. The gate now scores **walkstop es2 alone**; suite es2 is reported.
> - **Other fixes applied:**
>   - BC's priv latent is detached;
>   - the +200 dose check reads `Loss/bc_mb0`;
>   - a positive control runs through the exact gate path;
>   - one "HOLD row" definition;
>   - the residual pass's bias composition and rule;
>   - the per-row number corrected.
>   **These fixes were not re-audited.**

> Routing: roster ok · ceiling: opus · fable checks on steps 1 and 6.
> It amends `P2R_Z2_PLAN.md` (cams-only, kinematic R). Same stand bit, graft, walking anchor, statue prior and ladder;
> it adds a clean-state cloning term and makes R drift-proof.

## Context

**Z2 failed at step 5 (CHARTER §12.1, 2026-10-10).** The scripted controller R works at controller level: the final
probe P2R_RSTf_s18 homes 95 % of walkstop stands and is STILL on the suite, with 0 falls. The distilled network does not:
- suite not STILL: 0.92 steps/s, 5–7 taps/s, KS ≈ 1.1;
- walkstop homed 0 %, with 11–22 re-step lifts per stand;
- body 2.6 cm low.

**Diagnosis (measured on the 13,960 all-HOLD rows of the P2R_RSTf stand bank).**
- Where R commands 0, Z2 commands a partial LIFT on every leg: hip −0.13…−0.18 (about 30 % of lift), knee coupled.
- The loaded cams carry a +0.06 bias on two legs, more than the ±0.03 trim.
- Cause: exploration noise (σ ≈ 0.8, cams ≈ 1.15) keeps the training stand rows away from the clean all-homed /
  all-down state the eval visits, so the network averages LIFT / HOME / SEAT into it.

**Your decision (2026-10-10): Z2c, add clean-state cloning.**
- Each PPO update also fits R on a fixed bank of clean, R-driven stand states.
- The drift-proof R changes go in with it: a re-step dead band and more trim authority.
- A CPU fidelity gate decides early whether the full run continues.

## Design

**Clean-state cloning (BC) term in `PPOWithExtractor.update`.**
- Per minibatch, sample `B_bc` rows from the bank with a PRIVATE generator, so it never touches the global RNG.
- Compute `mu = policy.actor(bank_obs, False)`, the same training-path forward as PPO, never `act()`, **with the
  priv-encoder latent detached** (premise probe). BC's gradient then reaches the scan encoder and backbone only,
  never the latent the eval-path history encoder must match.
- `L_bc = mean_rows ½Σ_d((mu − R(bank_obs))/σ_A,d)²`.
- **R is recomputed from the bank obs** with the chain's `KRABBY_RESTANCE`, so targets always match the constants
  trained.
- Weight `KRABBY_BC_W` (KL units, per-row weight 1, like the prior).
- The bank is held on GPU as plain attributes (cols 0:257 only: all the actor's training path and R read), so it is in
  no `state_dict`. Knob off ⇒ nothing loads, nothing runs, bitwise.

**Drift-proof R** (code; defaults keep the current R):
- **`eps_step`:** a DOWN leg re-steps only when |c| > eps_step. Between eps_c and eps_step it stays HOLD (trimmed),
  so a small drift cannot start a re-step.
- **Trim authority `trim` = 0.1, `k_h` = 1.0** (saturates at |c| = 0.1). This cancels a Z2-sized cam bias (0.06),
  where 0.03 could not.

**Data.**
- Banks are recorded by the new probe with `P2R_BANK_MAX` = 100k on walkstop and suite, plus the field rows and a
  residual pass, so they cover clean trajectories and their bias-perturbed neighbours.
- **es2 banks are held out** from BC.
  - **Walkstop es2 is the fidelity gate's test set** (69 % all-HOLD, post-walk re-stance states).
  - Suite es2 is a near-duplicate of the trained suite es1 bank (statue from spawn), so it is reported only.

## Steps

1. **Code, one commit** (code_sha moves once).
   - **R** (`mdp/crab_hex_omni_restance.py`): `eps_step` (default = eps_c, so the current R is unchanged), the HOLD
     band extended to eps_step, tripod selection on eps_step; tests for mirror and branches.
   - **BC term** (`ppo_with_extractor.py`), at the insertion points the exploration found:
     - constructor args `bc_coef / bc_bank / bc_batch_size / bc_seed`;
     - load only when armed: width 1149, col 15 ≡ 1; bank cols 0:257 to device;
     - σ_A from `anchor_ckpt` (as the prior);
     - the term goes after the statue block, inside the global clip and the non-finite skip;
     - the priv latent is detached;
     - `Loss/bc`, `Loss/bc_mb0` (the first minibatch before any step, like `anchor_mb0`), `Loss/grad_norm_bc`, the
       `[PPO] construction` line; the generator state in `resume_state`.
   - **Knobs** (`motion_knobs.py`): `KRABBY_BC_W`, `KRABBY_BC_BANK` (comma list, repo-relative resolved, existence
     checked), `KRABBY_BC_BATCH`.
     - Refused unless `STAND_OBS=clock`, `STAND_PRIOR_TARGET=restance` and a parseable `KRABBY_RESTANCE` (R needs the
       constants).
     - Kept out of EVAL_CARRY.
     - `run_candidates` dump and log audits (`Mean bc:`).
   - **Fidelity scorer:** score both eval-path (history encoder + estimator) and training-path μ; report the HOLD-row
     rod bias as well as the cam residual.
   - **Tests:**
     - knob off ⇒ loss and RNG bitwise unchanged;
     - the BC gradient equals the hand formula; the private generator never touches the global RNG;
     - bank validation refusals;
     - R dead-band cases.

   → opus · high (code) / sonnet · high (tests); ✓ fable · high (diff review)

2. **Short bridge (≈ 0.2 GPU-h).** REFG_S1_s18 with knobs off, bitwise = REFS on suite, field_hard es1 and walkstop;
   append the pair.

   → sonnet · high

3. **Probe the drift-proof R and record the banks (≈ 1 GPU-h).**
   - `P2R_RSTg_s18`: the k3 constants + `eps_step 0.3, trim 0.1, k_h 1.0`, with `P2R_BANK_MAX 100000` on walkstop and
     suite.
   - Rows: walkstop es1/es2, suite es1/es2, field_turn, field_hard es1.
   - Then the residual pass `P2R_RSTresg_s18` (walkstop es1 + suite es1, bank on).
     - Its bias is the **cams of `p2r_z2_residual.json`** (≤ 0.063) + the **rods of `p2r_z1_residual.json`**. Never
       Z2's rods: they carry the −0.15 partial lift.
     - It passes iff the premise gate holds on the rows it runs; the trim must now hold a Z2-sized cam bias.
   - **Premise gate:** `p2_read --p2r-rst` unchanged (settled line 90 %), field rows on SE(diff).
   - Tuning: constants only, at most 2 passes. If it still fails, **STOP and ask**.

   → opus · high (calls) / sonnet · high (runs)

4. **Dose and pre-launch (CPU).**
   - **BC dose:** the fork's (GRAFT) BC loss per row on the training banks (≈ 8.9 on RSTf) is the expected
     `Loss/bc_mb0`. The +200 check bands that value (× [0.52, 2]), so the term is live and in band. The mean
     first-update `Loss/bc` falls fast inside the update and is reported only.
   - `KRABBY_BC_W` = 1.0, `KRABBY_BC_BATCH` = 2048.
   - Regenerate the dose file for the new constants.
   - Candidates v16: `P2__Z2c_s18` = the `P2__Z2_s18` chain + the new R constants + the three BC knobs. Training banks:
     the es1 + field + residual banks; es2 is held out.
   - Fidelity-scorer dry runs through the EXACT gate path on the held-out walkstop es2 bank, then preflight:
     - on GRAFT, expected to fail;
     - on one offline BC-fit actor (a positive control), expected to pass, so a path bug cannot false-stop a good
       run.

   → opus · high

5. **Train Z2c at s18 (≈ 2.2 GPU-h), fork GRAFT, 5000 its.**
   - **+200 wiring:** Z2's checks + `Loss/bc` finite and inside its dose band.
   - **FIDELITY GATE at model_20500**: CPU, the held-out **walkstop es2** bank alone, eval path, R recomputed with
     the run's constants.
     - Statistic: on **all-HOLD rows** (every leg in branch HOLD), the per-leg mean signed μ − R.
     - STOP the run and ask if either holds:
       - max over legs |hip| > 0.05 (Z2: 0.13–0.18);
       - max over legs |cam| > 0.05 (half the new trim authority).
     - The stop goes through run_arm's rc-3 path (`systemctl --user stop`, never SIGKILL; STOP_QUEUE; notify).
   - Drift D ≤ ½·D(C0) and formation are reported as before.

   → sonnet · high (run) / opus · high (gate)

6. **Pre-registered reads, then STOP.** The full read set as Z2. **Adoption-grade if all hold:**
   - STILL on suite es1 + es2;
   - homed ≥ 90 % and settled ≥ 90 % of walkstop stands at both seeds;
   - median settle ≤ the probe's + 1 s;
   - post-settle steps ≤ 0.5/s, taps ≤ 1/s, KS ≤ 0.5;
   - stand falls ≤ max(1, Z1's) per 300;
   - no OBSTACLE LOSS / FLAT FAIL / COST; attribution LEVER against C0R.
   - Reported: six-down, fidelity to the probe, walls against Z1, Z2 alongside.
   - PARTIAL or COST: **stop and ask.** Z1 stays the stand of record.

   → opus · high decides; ✓ fable · high

## Budget

| Item | GPU-h |
|---|---|
| Bridge | ≈ 0.2 |
| Probe + banks + residual pass | ≈ 1.0 |
| Z2c training | ≈ 2.2 |
| Reads | ≈ 1.7 |
| **Total** | **≈ 5.1** |

Steps 1 and 4 are CPU only.

## Quiet-failure risks and guards

1. **The BC term changes walking.** It acts only on bank rows (all stand rows, col 15 = 1), but it shares the
   backbone.
   - Guard: the walking anchor; drift D at +500; the ladder.
2. **Eval path ≠ training path.**
   - BC trains the priv-encoder latent, but eval uses the history encoder, which never sees the bank (on bank rows
     the priv latent is constant).
   - Guard: BC's priv latent is detached (premise probe); the fidelity gate scores the EVAL path on held-out walkstop
     rows.
3. **Bank-state coverage.** At eval Z2c may wander off the bank's R-driven states (compounding error).
   - Guard: the on-policy prior still covers noisy states; the residual-pass banks add perturbed neighbours; step 6
     reads it end to end.
4. **The private generator or bank leaks into the global RNG or checkpoints.**
   - Guard: unit tests; the bridge (knob off).

## Left for implementation (checks)

- **The actor's slice layout:** it reads `obs[:, :75]` as prop (not 0:89) and its "scan" slice holds the true lin-vel.
  R's columns (3, 4, 18–34) are inside 0:75. Trim the bank to cols 0:257 consistently.
- **σ_A:** the bank's `s1_sigma` equals σ_A to 1.2e-7; use the `anchor_ckpt` path.
- **Suite banks** include prologue and settle rows: keep them, they are stand-mode rows.
- **Grad-norm split:** subtract BC from the "rest" norm.
- **Sentinel:** attach only to a run dir newer than the launch marker; save the full JSON.
  - `p2r_early` is notify-only, its `--watch` waits for BOTH checkpoints, and it uses CAM_LINE 0.015 with no hip line.
    So the Z2c gate needs its own read of model_20500 with the 0.05 lines, wired to run_arm's kill path.
- **Fidelity scorer:** it must recompute R(obs) with the run's KRABBY_RESTANCE (or refuse a bank whose `restance`
  differs); today it uses the bank's stored target.
- **R:**
  - the trim extends to |c| ≤ eps_step (else dead-band legs never home);
  - WAIT / LIFT selection keys on eps_step;
  - validation eps_c ≤ eps_step; `format_restance` carries eps_step (dose-file and bank-string match).
- **BC bank GPU memory:** about 450k rows × 257 × 4 B ≈ 0.46 GB beside the envs.
- Z2's stand-prior +200 band also sees BC's pre-fit inside the first update: check it reads the prior's own
  `_mb0`-style value.

## Out of scope (not run)

- Six-foot loading (cams-only, as amended).
- Replicates and the bake.
- Warm-starting from Z2.
- A larger network.
