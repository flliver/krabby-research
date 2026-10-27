# Plan: Z2d seed replicates (s16, s17), the statue's bake evidence (p2r, CHARTER §12)

> **Audit trail (cap 1, reached).** Pass 1 (opus) found three blocking issues, all fixed:
> - **B1:** the "beats Z1 s18" lines had no margin (Z1 and HOLD nearly pass them). Replaced by a same-seed
>   `P2R_HOLD_s<seed>` reference on walkstop.
> - **B2:** `p2s_queue refsrep` would launch the abandoned ST2 replicate trainings. REFS now runs directly via run_arm.
> - **B3:** "six STILL clauses + no COST" admits OBSTACLE LOSS / FLAT FAIL heads. Now `ladder_ok`.
> - **Notes applied:** s17 runs regardless; `adoption_grade` is false by construction at s16/s17; `p2r_early` gets the
>   seed's S1.
> - **These fixes were not re-audited.**

> Routing: roster ok · ceiling: opus · fable check on step 6.
> It replicates `P2__Z2dR1_s18` (`P2R_Z2D_PLAN.md`) at seeds 16 and 17: same recipe, its own S1 fork per seed. Nothing
> new is trained or tuned.

## Context

**Z2d** (X20 `P2__Z2dR1_s18`, CHARTER §12.1, 2026-10-11) reads STILL, no COST, attribution LEVER (walls 0.8225 against
Z1 0.83, n = 400). It fails every Z2 homing gate (walkstop homed 2 %). Fable confirmed it is a better statue than Z1 on
like-for-like rows:
- six-down 0.33 / 0.55 vs 0.10 / 0.23;
- stopped hip-yaw mean 0.76° / 1.9° vs 4.2° / 5.9°;
- KS 0.135 vs 0.18.

That evidence is one seed only.

**Your decision (2026-10-11): "Z2d as a better statue"**, so run s16 / s17 replicates first; then you decide the bake.

**What the replicates need (exploration):**
- **No baselines at s16 / s17 on the current code.** Only `REF_S1_s16/s17` exist, without walkstop; `REFS_S1_s16/s17`
  never ran (`p2s_queue.sh` stopped before `refsrep`).
- **No s16 / s17 grafts.** The GPU identity stamps are a manual procedure (CHARTER.md:1266).
- **`p2_read --p2r-z2` crashes at a non-18 seed** (`KeyError: 16`, `p2_read.py:2363`).
- **The Z2D dose file is the s18 graft's.** A reused band would pass silently: the bank-set check cannot see the
  fork.

## Design

- **Same recipe per seed:** `P2__Z2dR1_s18`'s overlay verbatim; only the fork changes (its own graft) and
  `KRABBY_ANCHOR_CKPT` = that graft.
  - The seven cloning banks and the gate bank stay the s18 recordings. They are R-labelled stand states and R is
    seed-independent. The crouch bank cannot be re-recorded per seed, and a replicate must not change the recipe.
- **Per-seed baselines, all on the current code:**
  - `REFS_S1_s<seed>`: the ladder / COST reference;
  - `DIAG_REFSAMP_s<seed>`: the dose band's raws;
  - `P2R_HOLD_s<seed>`: the zero-action stand at that seed's S1, the statue reference (audit B1).
- **Pre-registered replicate rule (`replicate_grade`).** The statue replicates if, at EACH of s16 and s17:
  - **`ladder_ok`**, exactly as `p2r_z2_report` computes it: verdict STILL and stand outcome STILL against
    `REFS_S1_s<seed>`, i.e. no HAND LOSS / FLAT FAIL / OBSTACLE LOSS / COST (audit B3);
  - walkstop stand falls ≤ 1 per 300;
  - **walkstop stopped hip-yaw mean ≤ 0.5 × the same-seed HOLD's** (es1 / es2 mean);
  - **walkstop six-down > the same-seed HOLD's** (es1 / es2 mean).
- **Calibrated at s18:**
  - Z2d passes: yaw 1.94° ≤ 0.5 × 6.05° = 3.03°; six-down 0.55 > 0.34.
  - Z1 fails: 5.91°; 0.23.
  - The suite is not used for the statue clauses, because HOLD from spawn is already homed (yaw 0.09°).
- **Reported, not gating:**
  - Z1 s18 as cross-seed context, as es1 / es2 means: six-down 0.10 / 0.23, hip-yaw 4.21° / 5.91°;
  - homed / settled (Z2's gates, expected to fail as at s18);
  - attribution (C0R exists at s18 only, so PENDING at s16 / s17). `adoption_grade` is false by construction there;
    only `replicate_grade` is read;
  - walls against Z2d s18.
- **s17 runs whatever s16 reads**, for the record. A KILL at either seed stops and asks.

## Steps

1. **Code (CPU, one commit; campaign scripts only, code_sha does not move).**
   - **`p2_read.py --p2r-z2`:** seed from the head name (`:4271`). At a non-18 seed:
     - REFS / bands at that seed;
     - Z1 / HOLD / probe / C0R reported as the s18 references, labelled cross-seed;
     - the falls line from Z1 s18 (documented);
     - the replicate rule above, as `replicate_grade`.
   - **`p2_live.py`:** `z2_dose_default(arm, seed)` selects `p2r_z2d_dose_s<seed>.json` for a Z2D chain at seed ≠ 18.
     The preflight refuses a dose file whose `bc.checkpoint` is not the chain's fork (closes the silent-reuse hole).
   - **A graft-identity reader:** `p2_read --p2r-graft-identity NAME SEED` runs `p2r_bitwise_diff` of
     GRAFT_S1_s<seed> against REFS_S1_s<seed> on suite / field_turn / walkstop es1 (all raw keys but `clock_obs`, plus
     the `clock_obs[:,0]` == stand-label check), and prints the stamps to write.
   - **Tests.**

   → opus · high (code) / sonnet · high (tests)

2. **Baselines (GPU ≈ 1.5 h per seed).**
   - **`REFS_S1_s16` then `REFS_S1_s17`** (lr + 14 rows), run DIRECTLY with run_arm `--lr-eval` / `--row-eval` from
     each S1 head and its driver-stamp env, with `P2_GUARD_REF=REFH_S1_s18`. Never through `p2s_queue.sh refsrep`:
     the queue runs straight on into the abandoned `P2__ST2_s16/s17` trainings (audit B2).
   - Then `p2_read --p2s-inertness --p2s-seeds 16,17`.
   - **`P2R_HOLD_s16/s17`:** the p2r launcher's `hold` variant on suite + walkstop at es1 / es2.
   - **`DIAG_REFSAMP_s16/s17`:** suite es1, sampled actions.

   → sonnet · high

3. **Grafts.**
   - CPU: `p2r_graft.py --source <X10 / X11 model_19994> --expect-source-sha256 <s16 / s17>`, then `--identity-check`.
   - GPU ≈ 0.2 h per seed: `GRAFT_S1_s<seed>` rows (suite, field_turn, walkstop es1; `STAND_OBS=clock`; env = the REFS
     suite stamp).
   - The identity reader must PASS before the stamps are written into the sidecar.

   → sonnet · high

4. **Pre-launch (CPU).**
   - Per-seed dose files `p2r_z2d_dose_s16/s17.json` (`--ckpt <graft> --agent-yaml <X10 / X11> --raws
     <DIAG_REFSAMP_s<seed>> --bc-bank ×7 --bc-stand-mask`).
   - Candidates v19: `P2__Z2d_s16` / `P2__Z2d_s17`.
   - Preflight.

   → opus · high

5. **Train + read, s16 then s17 (GPU ≈ 3.5 h per seed).** For each seed:
   - 5000 its with Z2d's +200 checks (prior liveness, cloning band from the per-seed dose, mask live) and the
     model_20500 fidelity gate (same bank and lines; kill on fail);
   - the +500 / +1000 early report with that seed's S1 checkpoint (`p2r_early --s1-ckpt`); its drift line uses C0 s18,
     so it is reported as cross-seed, not acted on;
   - masked reads (lr + 14 rows, `P2_ENV_EXTRA` mask, suite es1 via run_arm);
   - the observe pass.

   A kill at either seed → STOP and ask.

   → sonnet · high (runs) / opus · high (gates)

6. **Replicate read, then STOP.**
   - `p2_read --p2r-z2 P2__Z2d_s16` and `_s17`: `replicate_grade` per seed, the statue table across s16 / s17 / s18
     against Z1.
   - Fable check, then **STOP and ask**. Bake, deployment contract (`STAND_OBS=clock` + `STAND_MASK=1`) and push are
     yours.

   → opus · high decides; ✓ fable · high

## Budget

| Item | GPU-h |
|---|---|
| Baselines incl. HOLD (2 seeds) | ≈ 3.0 |
| Graft identity rows | ≈ 0.4 |
| Train + reads + observe (2 seeds) | ≈ 7.0 |
| **Total** | **≈ 10.4** |

Steps 1 and 4 are CPU.

## Quiet-failure risks and guards

1. **The s18 dose file reused at s16 / s17** (bank equality passes). Guard: the per-seed file plus the preflight's
   `bc.checkpoint` == fork check.
2. **A graft that is not inert at its seed.** Guard: the GPU identity reader against REFS at that seed, before any
   training.
3. **The read silently falls back to s18 baselines.** Guard: the seed parsed from the name; cross-seed references
   labelled; a test at seed 16.
4. **Unmasked read stamps.** Guard: the masked-head INVALID rule (already live).

## Out of scope (not run)

- Z1 replicates (Z1 stays single-seed; Z1 s18 is the cross-seed reference).
- New cloning banks per seed.
- The bake itself.
