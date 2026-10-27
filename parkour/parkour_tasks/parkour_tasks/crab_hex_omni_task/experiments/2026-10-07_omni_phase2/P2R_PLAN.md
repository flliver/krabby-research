# Plan: a stand regime for the omni head (phase-2 stage p2r, CHARTER §12)

> **Audit trail (cap 2, reached).**
> - **Pass 1** (inventory, a code premise probe, 2 lens-split opus · xhigh auditors). Blocking fixes:
>   - D defined on non-stand steps, on the anchor's path; the monotone premise demoted to report-only;
>   - anchor liveness is wiring-only (the value band read backwards);
>   - the step-1 mode is keyed on the latched stand slot (a command-zero rule aliased settled turns);
>   - the helps / costs verdicts need the clause's margin;
>   - step 3 says PARTIAL-by-taps is the expected step-2 best case.
> - **Your answer:** add the early anchor-strength check (step 7).
> - **Premises verified:** the harness can be patched via `main.__globals__`; the bit never reaches history;
>   the anchor's data path and its exact zero at the fork; `STAND_OBS` is carried into evals.
> - **Pass 2** (opus · xhigh + fable · high, whole document). Opus: nothing blocking. Fable: one blocking issue.
>   The D premise compared C0R at +5000 against ST2 at +506; it now uses the matched +506 pair. **This last fix was
>   not re-audited.**
> - **Gate check:**
>   - the anchor-liveness band was backwards (fixed);
>   - the D statistic had no population or path (fixed);
>   - the D premise was exposure-mismatched (fixed);
>   - the helps / costs verdicts had no margin (fixed).
> - **Size:** 9 steps / ≈ 2,600 words (reviewed: 9 / ≈ 2,120). The growth is audit fixes plus the implementation
>   checklist.
> - **Routing:** roster ok · ceiling: opus · fable checks on steps 4 and 9.

> The p2s plan this file used to hold is archived in `2026-10-07_omni_phase2/P2S_PLAN.md`.

## Context

**The problem.** The head of record (S1 s18, `policy/6_p2a_field/model_19994.pt`) marches in place at a zero command:
2.83 steps/s, 0.135 m lifts.

**What p2s showed (CHARTER §11).** A stand-slot reward on the shared policy (support income + loaded-foot slide,
ST2) stopped the march:
- 0.17 steps/s, 5 mm lifts, 5.5 loaded feet.
- Costs:
  - force chatter: 16 taps/s, action jitter +65 %;
  - **bleed into walking:** swing apex about halved in every hold, field_hard 0.8125 → 0.6175 (n = 400), back onset
    0.99 s.
- The +506 / +1006 checkpoints lie on the same trade-off curve, so there is no sweet spot.

**Why.**
- The policy has no clean stand feature. A settled in-place turn shows the same zero command as stand.
- One shared network carries any stand change into the gait. Walking rows were already distinguishable by their
  command, and they still bled.
- Precedents that separate stand from walking: a separate standing policy (Disney), routed rewards with an observed
  mode (Kim, Peng), and classical gait engines with a STOPPED state (`lit-review-gait-motion-quality.md` §2/§7).

**Your decisions (2026-10-09):**
- Plan a stand regime (not more stand-only reward on the shared gait).
- **Probe both routes, cheap first:**
  - Step 1 (eval only, no training) tests a controller-level stand mode.
  - Step 2 is the single-network regime, informed by step 1.
- **Single-network route = graft a bit + a walking anchor.**
  - Fine-tune the S1 head with a stand bit grafted into its unused (hidden) clock column, the p2s stand terms, and a
    loss holding the policy near frozen S1 on non-stand states.
  - Two s18 arms: bit + anchor, and bit only.
  - A from-scratch STAND_OBS chain (≈ 9 GPU-h/seed) is the fallback, not in this plan.
- **Unchanged:**
  - target "mostly still" (STILL on the suite stand at both env_seeds);
  - no walking or wall cost beyond noise;
  - walkstop is a COST guard;
  - the p2s ladder and estimators (`p2s_stand.py`, `p2_read.py --p2s-*`).

**Ground rules (CHARTER §1a):**
- Never push. Stage only this campaign's paths.
- One Isaac process at a time.
- eval_sha does not move: `eval_crab_hex_gait.py`, `gait_eval/`, `scenarios_omni.yaml`, `eval_categories.py` and
  `run_phases.py` are untouched.
- Fresh chain names after a kill.
- Push-notify at every stop; re-arm watchers after every milestone.
- Adoption, bake and push are yours.

## Step 1: controller-level stand mode (eval only; code_sha does not move)

1. **Launcher and reads (CPU, campaign tool files).**
   - **Launcher:** `C/scripts/p2r_mode_eval_crab_hex_gait.py`. The name must end in `eval_crab_hex_gait.py`, or
     `run_arm`'s idle check misses it.
     - It loads the locked harness with `runpy.run_path(…, run_name="p2r_harness")` and swaps its module-level
       `make_on_policy_runner` (imported `:179`, used `:708`) before calling `main()`.
     - `get_inference_policy` then returns a mode wrapper. The wrapper's output is the action that is applied and
       recorded (`:1030 / :1048 / :1071`).
   - **Mode rule (one state per env): the same stand set as step 2's bit and the deployment contract.**
     - Stand mode = the command term's **latched** stand slot. The wrapper arms the term's eval slot latch, which with
       every stand knob off feeds only `slot_id`, and reads `stand_mask()`.
     - Settle and stand holds are stand. A turn hold stays turn for its whole length, even once it settles.
     - Entry is on the first stand step (no debounce); exit is on the step the slot changes.
     - A command-zero rule is not used: it would also fire inside every settled turn (about 13 s of statue per
       field_hard episode).
   - **Variants:**
     - **(a) hold:** zero action (statue).
     - **(b) ST2 specialist:** ST2 `model_24993` acts. It sees the same obs with its own estimator output written into
       cols 207:216. Both policies run every step.
     - **probe:** S1 acts. It logs the per-step drift ½Σ_d((μ_probe − μ_S1)/σ_S1,d)² for ST2 at +506 / +1006 / +5000
       and for C0R. C0 model_20500 / model_21000 are scored on CPU from the bank, giving matched-iteration pairs.
       - μ is computed on **the anchor's path** (`hist_encoding=False`, true priv_explicit columns), each network on
         its own clone of the obs.
       - It also saves a bank of non-stand obs for CPU re-use.
       - **D** = that per-step value averaged over **non-stand steps only** (the step-2 bit-0 domain: every hold that
         is not stand or settle, settled turns included). Stand-step D is reported separately.
   - **Inertness:**
     - Extra runners are built on a stand-in env, with the CPU and CUDA random state saved and restored.
     - `run_meta.p2r_mode` records the variant, N, channels, shas and switches per episode.
     - Raw `p2r_stand_mode` and `p2r_dmu_*` series are added to the stamp.
   - **`run_arm.sh`:** `EVAL=${P2_EVAL:-…}` and `${P2_ENV_EXTRA:-}` after the stamp env.
   - **`p2_read.py`:** `--p2r-probe` and `--p2r-step1` read mode heads with the p2s ladder against REFS_S1_s18.
     - A stamp without a matching `run_meta.p2r_mode`, or with a checkpoint sha other than REFS's, is INVALID.
   - Unit tests for the mode rule, the record and the reads.

   → opus · high (launcher) / sonnet · high (tests, reads)

2. **Runs (≈ 1.6 GPU-h; the GPU is idle).**

   | Run | Rows | GPU-h |
   |---|---|---|
   | `P2R_PROBE_s18` | suite es1, field_hard es1, walkstop es1 | ≈ 0.22 |
   | `P2R_HOLD_s18` (a) | suite es1+2, walkstop es1+2, field_hard es1–4 (n = 400), field_turn es1, field_light es1 | ≈ 0.65 |
   | `P2R_ST2M_s18` (b) | same rows | ≈ 0.72 |

   - **The probe run must reproduce REFS_S1_s18 bitwise**, raw actions included, or stop. It is the launcher-inertness
     check.
   - **Tier 2** (field_light es2, ring_obst es1–2, field_suite, lr; +0.35 GPU-h) runs only for a variant that
     survives tier 1.

   → sonnet · high

3. **Step-1 read, then STOP and ask.**
   - **Per variant: the p2s ladder against REFS.**
     - STILL clauses on both suite seeds (variant (b) is expected to fail taps).
     - Pooled stand-hold falls; the statue on walls is the main unknown.
     - Onset (walkstop median, es1 + es2).
     - OBSTACLE LOSS on field_hard (n = 400).
     - FLAT FAIL on suite es1.
     - Body height, field_turn completion.
     - The statue on walls is read from field_turn's stand hold (field_hard's opening stand is on the clear pad).
   - **Reported:**
     - stand-mode share per hold;
     - falls in stand mode and within 2 s of a handover, by entry gait phase;
     - turn final heading against REFS.
   - **Step-2 premise from the probe run (D as defined in step 1):**
     - D_ref = D(ST2 +506), the smallest drift that already lost walls.
     - **Stop if D(C0 +506) > 0.5·D(ST2 +506)** (matched iterations, the same comparison step 7 makes): the anchor
       could not tell harmless continuation from harmful drift. The +1006 pair is reported; D(C0R +5000) against
       D(ST2 +5000) is context only.
     - Reported, not a stop: D at +506 / +1006 / +5000.
   - **Said plainly at this stop:**
     - Variant (b), keyed on the latched slot, approximates the best case for G2: ST2 at commanded stands, S1
       everywhere else. If (b) reads OBSTACLE LOSS or onset COST, step 2 is unlikely to pass unless G2's own stand
       end-state differs from ST2's.
     - G2 and G1 keep ST2's stand terms, which chattered (11–16 taps/s) at every checkpoint, and nothing in step 2
       acts on stand rows. So step 2 is **expected to read PARTIAL-by-taps at best**.
     - You decide whether G2 carries a stand-side lever (for example an anchor toward zero action at stand, if (a)
       shows the statue is safe), whether PARTIAL-by-taps is acceptable, or whether the controller mode goes forward
       instead.

   → opus · high

## Step 2: single-network regime (graft + anchor), if step 1's stop says go

4. **Code, one commit (code_sha moves once).**
   - **Stand bit as the new value `KRABBY_OMNI_STAND_OBS=clock`.** It reuses the knob already in the eval-locked
     `EVAL_CARRY` and `EVAL_SEMANTIC_KNOBS`; a new name would be dropped from evals.
     - Writes `stand_mask()` (the latched slot) into **the current step's column 15 only, never the history buffer**.
       So history column 15 stays 0, as in S1.
     - Requires `CLOCK_HIDE=1` and no ramp.
     - Arms the eval slot latch: settled turns read 0, stand holds read 1.
     - Mirror: `lr_signs[15] = +1` under the param.
     - Parser `parse_stand_obs` for 0 / 1 / clock; dump check (`clock` → `stand_bit_clock: true`, no `stand_obs`).
   - **Walking anchor in `ppo_with_extractor.py`:**
     - Inserted after the bound loss (`:781-788`), before the symmetry block reassigns `obs_batch` (`:793`).
     - Loss: mean over bit-0 rows (`obs[:B, 15] == 0`) of ½Σ_d(μ_θ − μ_A)²/σ_A,d². It acts on means only, so it
       never pushes the shared σ.
     - The anchor is a frozen deepcopy of the actor, loaded from the graft, forward with `hist_encoding=False`.
     - Knobs `KRABBY_ANCHOR_W` / `KRABBY_ANCHOR_CKPT` (agent-side, not carried into evals). Refused unless
       `STAND_OBS=clock`, unpaired, or W < 0.
     - Logs `Loss/anchor`, `anchor_rows`, `anchor_stand` (no gradient) and `anchor_mb0`.
   - **Graft tool `C/scripts/p2r_graft.py` (CPU):**
     - Checks the source sha 968cc550.
     - Zeroes column 15 of `actor.actor_backbone.0`, `estimator.0` and `critic.0` (norms 1.74 / 1.38 / 1.19), and
       their Adam moments (asserted < 1e-30 first).
     - Leaves everything else bitwise identical.
     - Writes `outputs/…/phase2/graft/S1_s18_c15/model_19994.pt` and `graft.json`.
   - **Tooling:**
     - `run_arm` `verify_head_sha` gains a graft branch (graft.json shas + passed identity checks), and
       `fork_source_rd` resolves to the S1 run.
     - The arm-name pattern accepts `P2__G[0-9]+` (dose ST2) in `run_arm`, `p2_live` and `p2_read`.
     - `p2_live` +200 anchor liveness is **wiring only**:
       - `anchor_rows` in [0.8, 0.95];
       - `mb0 == 0`;
       - `Loss/anchor` finite and > 0 after the first update;
       - the realised λ in the dump.
       - There is no value band: a working anchor keeps drift small, and a lower bound would kill it.
     - Candidates v12: `P2__G2_s18` (bit + anchor) and `P2__G1_s18` (bit only); stage p2r, base 2a, 5000 its, fork =
       the graft, ST2's overlay verbatim + `STAND_OBS: clock` (+ anchor knobs for G2).
   - **Tests:**
     - config identity with the knob unset / 0 / 1;
     - history col 15 ≡ 0;
     - anchor gradient from bit-0 rows only;
     - `anchor_coef` 0 ⇒ loss bitwise unchanged;
     - graft identity (`torch.equal`, actor on both paths, estimator, critic);
     - dump checks;
     - arm patterns.
   - Then a fable review of the landed diff.

   → opus · high (code) / sonnet · high (tests); ✓ fable · high

5. **Graft identity and inertness (≈ 0.45 GPU-h).**
   - **GPU identity:** `GRAFT_S1_s18` (`P2_ENV_EXTRA="KRABBY_OMNI_STAND_OBS=clock"`) on suite es1, field_turn es1 and
     walkstop es1 must reproduce REFS bitwise.
     - The raw `clock_obs` column must be exactly 1 on settle and stand steps and 0 elsewhere, settled turns
       included.
   - **Inertness bridge:** REFS_S1_s18 on the new code, knob off, on suite es1, field_hard es1, walkstop es1 and
     field_turn es1, must be bitwise equal to the ae892cec stamps.
     - If so, `inertness_bridge.json` lets the ladder accept that code_sha pair.
     - Any difference means the full REFS + C0R re-evaluation under fresh names (≈ 2 GPU-h).

   → sonnet · high

6. **Anchor dose (CPU).**
   - λ = S / D_ref.
     - S is the median |`Loss/surrogate`| over ST2's iterations +200 to +5000.
     - D_ref is from step 1's probe run.
   - λ goes in candidates v12, with a dry run of both chains through the driver validator.
   - The **post-run** clock-removal check (`clock_obs ≡ 0`, `run_candidates.py:9945-9950`) would park the driver after
     G2 trains. Narrow it for these chains: col 16 ≡ 0, col 15 ∈ {0, 1} and equal to the stand label. Test the
     narrowing on step 5's GRAFT raws.

   → opus · high

7. **s18 screen: G2 first, then G1 (≈ 7.8 GPU-h).**
   - Live kills as in §8, plus the p2s guards (σ, episode-length kill, notify-only crab_failure) and the anchor
     liveness.
   - **Early anchor-strength check** (your decision, added in audit pass 1; CPU, no Isaac process):
     - While G2 trains, compute D at G2's model_20500 and model_21000 against ST2's at the same iterations, and C0's,
       on the step-1 bank of non-stand obs.
     - **If D(G2) ≥ ½·D(ST2) at either point: notify and ask, while the run continues.** The anchor is too weak at
       this λ. A mid-run kill would lose the arm.
       - Pre-registered re-dose, for your call: λ′ = λ·D(G2)/(¼·D(ST2)), one retry under a fresh chain name.
       - Reported beside it: G2's stand-term income against ST2's at the same iterations, so a too-strong anchor
         (no stand regime forming) is visible early.
   - **If G2 reads OBSTACLE LOSS with the anchor live, stop and ask before G1.**

   → sonnet · high to run

8. **Pre-registered reads.**
   - **Per head:** the p2s ladder against REFS_S1_s18, with C0R as the PARTIAL comparator and attribution as in p2s.
     - G-head stamps must carry `STAND_OBS=clock`, and their raw bit must match the stand labels; otherwise INVALID.
   - **Anchor effect (G2 vs G1):**
     - **helps** only if G2 beats G1 by the margin of the clause that differs:
       - field_hard (n = 400): max(0.05, 2·SE);
       - field_light: max(0.03, band);
       - ring: max(0.05, band).
     - Two heads straddling the OBSTACLE LOSS line by less than that margin read **inconclusive**.
     - **costs** if G2's stand rung is lower than G1's.
   - **Bit effect (G1 vs ST2,** same seed and dose, already on record): same rules. Expect G1 may bleed like ST2:
     stand income can be anticipated from walking states across slot boundaries.
   - **Reported:**
     - the drift D of each G head, computed on CPU from the step-1 non-stand obs bank;
     - one knob-off eval of the best head (what happens without the bit);
     - taps and jitter beside every stand verdict.
   - **PARTIAL from taps alone with no OBSTACLE LOSS: stop and ask.** Options then: a stand-side anchor toward zero
     action (if step 1(a) proved the statue safe), or the controller mode.

   → opus · high decides; ✓ fable · high

9. **Stop, write-up, proposal.**
   - **Adoption needs all of:**
     - STILL on both suite seeds (all six clauses);
     - no OBSTACLE LOSS / FLAT FAIL (against REFS and chained against ANCW) / COST;
     - attribution LEVER against C0R;
     - the deployment contract recorded: col 15 = 1 exactly while commanded to stand.
   - **Replicates at s16/s17 and the bake are yours** to call after this stop.

   → opus · high

## Budget

| Item | GPU-h |
|---|---|
| Step 1 (probe + hold + ST2 specialist, tier 1) | ≈ 1.6 (≈ 1.95 with tier 2) |
| Step 2 identity + inertness bridge | ≈ 0.45 (≈ 2.4 if the full re-eval is needed) |
| G2 + G1 (5000 its + reads each) | ≈ 7.8 |
| Knob-off eval of the best head (step 8) | ≈ 0.3 |
| **Total** | **≈ 10** (≈ 12.5 worst case) |

## Quiet-failure risks and guards

1. **The launcher is not inert.** Extra networks draw random numbers.
   - Guard: random state restored; a stand-in env; the probe run must equal REFS bitwise; reads require
     `run_meta.p2r_mode`.
2. **The graft is not identical at the fork, or the bit leaks into history.**
   - Guard: CPU `torch.equal` on every path; history ≡ 0 test; GPU bitwise rows; `anchor_mb0 == 0`.
3. **The bit is missing or wrong in an eval** (dropped knob, stamp env override).
   - Guard: `STAND_OBS=clock` (already carried); `P2_ENV_EXTRA`; a raw-bit check makes a wrong stamp INVALID.
4. **The anchor masks the wrong rows, is dead, or dominates.**
   - Guard: placement before `:793`; gradient-mask unit test; cross-knob refusal; +200 liveness; the step-3 D premise.
5. **The stand outcome is misread:** step 1 and step 2 use different stand sets, and chatter persists in step 2.
   - Guard: both steps key on the same latched stand slot; stand-mode share is reported; taps stay in STILL; the
     taps-only stop; step 3 says plainly that PARTIAL-by-taps is the expected step-2 best case.

## Left for implementation (checks)

- **Launcher:**
  - `runpy.run_path` returns a copy of the globals: patch `g["main"].__globals__["make_on_policy_runner"]`.
  - The launcher must call `simulation_app.close()` itself; `main()` doesn't.
  - Prove the swap is live:
    - (a)'s applied actions are exactly 0 on every stand-mode step;
    - stand-mode share > 0 on stand holds;
    - actions before each env's first stand entry equal REFS bitwise.
  - Give every extra network a clone of the obs carrying its own estimator output (the harness writes S1's in place).
- **Step-1 reads:**
  - `--p2r-step1` skips the rungs tier 1 cannot feed (ring, lr, DOSE) explicitly.
  - Define "survives tier 1" (no stand-hold-falls COST, no OBSTACLE LOSS on field_hard, no FLAT FAIL) before
    reading.
  - Compare field_light at matched n.
- **Anchor:**
  - Forward the frozen anchor on the full minibatch tensor that produced `mu_batch`, then mask, so `mb0` is exact.
  - The deepcopy is the whole actor module (priv and scan encoders included).
  - Edit `parkour/scripts/rsl_rl/modules/ppo_with_extractor.py`, not the `compute/` copy.
- **Graft:** the estimator's Adam state lives in `alg_state["estimator_optimizer"]`; map optimizer indices to names
  before zeroing.
- **Raw-bit checks** allow the one-step command-to-obs lag at hold boundaries.
- **Inertness:**
  - Run the inertness bridge under a fresh name (not REFS_S1_s18).
  - Consider field_hard es1–4 in it.
  - A ~50-it knob-off replay of C0R's chain against its logged scalars backs training-path inertness.
- **The col-15 write must not copy slot 5's history re-patch.**
- **Latch carry-over across suite waves:** the move_turn wave's settle inherits the turn wave's latch on about 62 % of
  envs.
  - Fix it in the step-4 commit (re-derive on episode reset); don't exempt the step-5 check.
  - Arming at runtime needs `_slot_latch = True` plus the `_hook_turning` buffer, set in the swapped factory.
  - Step 1 reports the settle-mode share per wave.
- **Step 7's +500 check** needs its own wake (a sentinel on G2's model_20500), because p2_live wakes only at +200 /
  +1000.
- **The probe bank's priv columns** are overwritten in place by S1's estimator before the wrapper sees obs: capture
  them by wrapping `env.step`, or use the same estimator-filled content for every D.
- **Step 5 bitwise identity** excludes the raw `clock_obs` series and `run_meta.env_vars`.
- **Critic obs:** verify that the col-15 write reaches the critic group (current step only).

## Verification

- Unit tests (steps 1 and 4) and the fable review of the step-4 diff.
- Bitwise checks: the probe run (launcher), the graft identity rows, the inertness bridge.
- Dry runs of the G chains; the +200 wiring, liveness and dump audit per arm; INVALID EVAL on every stamp.
- Every verdict goes to CHARTER §12 and is push-notified.

## Out of scope (not run)

- From-scratch STAND_OBS chain.
- Replicates and bake.
- A runtime/deploy mode switch (no omni deploy path exists yet).
- Push-recovery evals.
- Multi-critic.
