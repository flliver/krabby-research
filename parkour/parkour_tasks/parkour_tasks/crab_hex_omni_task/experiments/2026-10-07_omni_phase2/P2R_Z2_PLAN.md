# Plan: Z2, a stand with the cams homed and all six feet planted (p2r amendment 2, CHARTER §12)

> **Audit trail (cap 2, reached; you accepted with fixes).**
> - **Pass 1** (4 premise probes, then two opus · xhigh auditors split by lens). Blocking fixes:
>   - **loaded-cam residual drift** → your in-band trim, the residual pass and the CPU fidelity check;
>   - **six-down gate** was a median over bimodal stands → "≥ 90 % of stands";
>   - **4-s settle line** was uncalibrated and unreachable → relative to the probe, post-settle read from each stand's
>     own settle, and the field rows added to the premise gate;
>   - **no dose check for the new target** → a CPU dose step and +200 branch asserts;
>   - corrections: the proxy readings, a tripod-selection deadlock, the walking-rows bitwise check.
> - **Pass 2** (opus · xhigh). Blocking fixes:
>   - **proxy scale** (a hip-only press vs knee-inclusive training) → your choice: the knee moves in every lift and
>     press; the calibration stops if the scales still differ; the dose step reports branch shares;
>   - **no stand bank for the fidelity check** → banked on the final and residual passes, with a dry run before
>     launch;
>   - **stand body height** added to the premise gate.
>   **Pass 2's fixes and the knee decision were not re-audited.**

> Routing: roster ok · ceiling: opus · fable checks on steps 1 and 5.
> It amends `P2R_STATUE_PLAN.md`. Same stand bit, graft, anchor, ladder and ground rules; only the stand target changes.

## Context

**Z1 (CHARTER §12)** learned the statue (μ → 0 at the stand bit) and reads STILL with no cost (walls 0.83, fable-confirmed).
Its stopped pose has two flaws, both measured on Z1's walkstop raws (300 stands after walking):

- **Five feet, not six.** In 55 % of stands at least one foot carries < 10 N. The gap is a median 6.8 mm (IQR 1.7–11.6,
  max 44). That leg's rod actions match the loaded legs' (|μ| ≤ 0.04), so this is the static-indeterminacy lottery,
  not the policy. A uniform preload cannot fix it, because it only raises the body; each leg needs its own load
  feedback.
- **Cams not homed.** Cams are velocity-commanded, so μ = 0 freezes each cam wherever walking left it:
  - stand cams span −1.25…+1.55 rad, and hip yaw is 4.9–8.7° on average (max 22°);
  - a residual |μ| of 0.002–0.019 keeps the cams drifting a further 0.33 rad during the stand.
  Home is cam ≡ 0, which gives hip yaw 0, the default fan (end legs splayed 15° by the USD shim). Spawn is at home.

**Your decisions (2026-10-09):**
- **Homing:** home by re-stepping one tripod at a time: lift, home the cams in the air, put down.
- **Load sensing:** the policy's own current-sense contact proxy, read while a leg presses. This one carries over to
  hardware; the sim-only joint-sag signal was rejected.
- **Fork:** from the S1 graft, like Z1. 5000 iterations, the same anchor; only the stand target changes.

**The load-sensing problem, and the fix.**
- The proxy (obs cols 83–88) is `gate·clamp(0.1 + F_rod/F_rated) − 0.5`, gated by rod speed (full at ≥ 0.3 × rated).
  - the reading is max(hip, knee) channel. With the knee rod moving (training, walking), measured on
    DIAG_NOISYSTATUE_s18 / P2R_PROBE_s18: about +0.06–0.09 at < 5 N, +0.64–0.68 at 40–120 N, +0.8–0.9 at 250–450 N.
    Hip only (knee parked): −0.33 / −0.11 / +0.19…+0.60. So **every lift and press R makes moves the knee too**,
    keeping all readings on the one knee-inclusive scale (your 2026-10-10 decision);
  - parked: exactly −0.5, so the still statue is blind.
- A still stand can therefore only know its loads from the last time each leg moved. That needs memory.
- **Last-action memory is unsafe.** "Hold the last action" is an integrator. Under training noise (σ ≈ 0.65 on rods)
  the stand-row commands would random-walk to the clip.
- **So the memory goes into the observation, as a sample-and-hold proxy:**
  - on stand-bit rows, cols 83–88 report each leg's last fully gated reading;
  - live while the leg moves; reset to −0.5 ("unknown") at episode reset;
  - current step only (history stays raw, as with col 15);
  - walking rows unchanged.
- Hardware can do the same in its runtime. In training, noise keeps the rods moving, so stand rows see near-live
  values; the hold only bites at a deterministic stand.

**The approach (as in Z1): prove a scripted controller, then distil it.**
1. **Controller R** is a memoryless function of the current observation. Its targets are restoring and absolute, with
   no last-action terms. Per leg i:
   - c_i = cam angle (col 18+2i, already wrapped to [−π, π]);
   - h_i = held proxy (col 83+i);
   - loaded_i = h_i ≥ θ_L; homed_i = |c_i| ≤ ε_c (0.10 rad, about 1.7° of hip yaw).
2. **Cam command:**
   - while unloaded: −clip(k_c·c_i, ±1);
   - while loaded and homed: a restoring trim −clip(k_h·c_i, ±0.03). This cancels a Z1-sized residual (|μ_cam| ≈
     0.019, which otherwise drifts about 0.06 rad/s, leaving the band in about 1.6 s). It turns a loaded foot by at
     most about 1.7° of hip yaw (your 2026-10-10 decision);
   - while loaded and not homed: 0, waiting for its tripod's re-step.
3. **Hip-pitch command:**
   - lift to L if not homed and (unloaded, or its tripod is the one selected to step);
   - otherwise, the seat law S(h_i) = κ·clip(θ_S − h_i, 0, Δ_max) in the extend direction. That is about 0 when
     loaded and extends a light leg until it carries θ_S (θ_L < θ_S).
4. **Knee:** moves with the hip at a fixed ratio ρ in every lift and press (so its gate is open whenever R reads
   load); otherwise 0.
5. **Tripod selection** (A = FL, MR, RL; B = FR, ML, RR):
   - a tripod may step only when it has an un-homed leg and every leg of the other tripod is loaded;
   - if both may, the one with the larger Σ|c| over its un-homed legs steps.
   - This is mirror-equivariant under both planes (fore-aft maps each tripod to itself; left-right swaps A and B),
     provided the hip and cam constants are the same on every leg and ρ follows the knee's per-side sign convention.
6. **Last tripod down:** it seats to θ_S against a planted tripod that is statically determinate. That is what makes
   six feet carry load.
7. **From spawn:** every h = −0.5 (unknown), so every leg seats once.
8. **Distillation:** the statue prior's target changes from 0 to R(obs): ½Σ((μ − R)/σ_A)² on bit-1 rows. Same
   normalisation, anchor and σ_A as Z1.

## Steps

1. **Code, one commit (code_sha moves once).**
   - **Obs:** `KRABBY_OMNI_STAND_PROXY=hold`
     - per-env, per-leg held buffer in `mdp/crab_hex_omni_observations.py`, written beside the col-15 stand bit;
     - updated when `max(gate_h, gate_k) ≥ 1`, from the forward task's `_get_contact_fill` (`crab_hex_forward_task/mdp/observations.py:77-107`);
     - reset on episode reset; current step only;
     - refused without `STAND_OBS=clock`.
   - **Controller:** R as one torch function in a shared module, used by both the probe and the PPO loss.
     - Constants (L with its sign, ρ, θ_L, θ_S, κ, Δ_max, k_c, k_h, ε_c) come through one knob, `KRABBY_RESTANCE`, so tuning
       never moves code.
     - Seed values: L from S1's swing (lift is negative hip-pitch; swing median about −0.7 for about 100 mm of
       clearance). θ_L and θ_S come from step 3's calibration pass only.
   - **Probe:** `--p2r-mode restance` in `scripts/p2r_mode_eval_crab_hex_gait.py`, as the `hold` branch (`:431`) with
     `R(obs)` in place of zeros. It logs per-leg h, c and the gate at stands. On the final full-set pass and the
     residual pass it banks stand-row obs (true priv columns, held proxy, R(obs), branch label) for step 4's fidelity
     check.
   - **PPO:** `KRABBY_STAND_PRIOR_TARGET=restance` in `ppo_with_extractor._statue_losses` (`:408-430`).
     - Refused unless `STAND_PROXY=hold`.
     - Unset gives Z1's path, bitwise.
   - **Reads:**
     - `p2s_stand.py` adds per-stand estimators:
       - **homed** — max_i |c_i| over the last 5 s ≤ 0.10;
       - **six-down share** — all six feet > 50 N, last 5 s;
       - **settle time** — homed ∧ six-down held for 0.5 s;
       - **post-settle stillness** — steps, taps and KS from each stand's own settle time (an unsettled stand fails);
       - **re-stance foot lifts**.
     - `p2_read.py --p2r-z2` reads the gates below.
     - candidates v13 chain `P2__Z2_s18` = the Z1 chain + `STAND_PROXY: hold` + `STAND_PRIOR_TARGET: restance` +
       `RESTANCE` (the probe's final constants).
   - **Tests:**
     - knobs off ⇒ obs and loss bitwise identical to Z1's;
     - held-proxy hold, reset and current-step-only behaviour;
     - R unit cases (spawn, mid-air homing, tripod gating, seat law);
     - R mirror equivariance;
     - prior gradient only from bit-1 rows.

   → opus · high (code) / sonnet · high (tests); ✓ fable · high (review of the landed diff)

2. **Inertness bridge (≈ 0.45 GPU-h).** With knobs off on the new code: suite, field_hard es1–4, walkstop and field_turn
   bitwise equal to the bridged Z1-era stamps. Append the rows to `inertness_bridge.json`. If any row differs, run the
   full REFS + C0R re-eval (≈ 2 GPU-h).

   → sonnet · high

3. **Controller probe, the premise (≈ 1–1.5 GPU-h).** S1 + R at stand rows (`P2R_RST_s18`), the way the HOLD controller
   was proven.
   - **Calibration pass:** walkstop es1. Read proxy-vs-foot-force per leg in R's own lift and press (knee moving),
     beside the training-scale bank, then fix θ, L and ρ. If the two scales still differ, STOP with the per-channel
     numbers.
   - **Tuning:** at most 3 passes on walkstop es1 + suite es1, changing constants only. Pre-registered fallback if
     stale holds leave a foot unloaded: seat the two tripods in sequence from spawn.
   - **Then the full set:** suite es1/es2, walkstop es1/es2, field_hard es1 (n = 100), field_turn. Walking rows before
     the first stand must be bitwise equal to REFS (no held-buffer leak into walking rows or history).
   - **Premise gate:**
     - the Z2 gates below, excluding the ladder rows and the relative settle clause, which the probe itself sets;
     - field_turn and field_hard es1 stand falls and completion no worse than P2R_HOLD's by more than 2 SE, so tuning
       cannot buy settle time with a terrain-unsafe L;
     - stand body height within 0.03 m of REFS on suite and walkstop stands. A seat law that preloads every leg
       raises the body; this is the ladder's COST clause, read early.
   - **Residual pass (≈ 0.2 GPU-h, your 2026-10-10 decision).** R plus Z1's measured per-leg stand residual (cam
     about ±0.02, plus its rod residual), on walkstop es1 + suite es1, under the same premise gate. It tests the
     loaded-cam hold before training.
   - If the premise gate or the residual pass fails after 3 tuning passes, **STOP and ask**. The fallbacks are yours:
     the sag signal, or cams only.

   → opus · high (tuning calls) / sonnet · high (runs)

4. **Train Z2 at s18 (≈ 2.2 GPU-h).**
   - Fork from GRAFT_S1_s18 with Z1's recipe and guards, plus the two knobs.
   - **Dose, before launch (CPU).** Compute the expected first-update prior loss under R on fork stand rows from the
     noisy-statue bank (DIAG_NOISYSTATUE_s18 raws, proxy rebuilt bitwise). That sets the +200 realised-dose band, which
     must exclude the target-0 value (about 2.0, Z1's band). Report R's branch shares on that training-scale bank
     beside the probe's; STOP if the in-air homing branch is absent.
   - **+200 wiring:**
     - Z1's row and finiteness checks;
     - the dose band above;
     - the share of bit-1 rows with R ≠ 0, and the lift / home / seat branch shares;
     - held = live proxy on ≥ 99 % of training stand rows.
   - Early checks (notify-only, as for Z1):
     - walking drift D at +506 and +1006 ≤ ½·D(C0);
     - formation: rolling-20 prior loss at +500 ≤ 0.25 × its first-update value (Z1's 0.1× line assumed a constant
       target);
     - stand fidelity (CPU, on the probe's banked stand states, your 2026-10-10 decision; dry-run the scorer on
       GRAFT_S1 against the bank before launch): the checkpoint's μ against
       R by branch. Notify if the loaded-leg cam residual |μ_cam − R_cam| > 0.015, half the trim's ±0.03 authority.
       The 0.006 line applied only to a hold with no trim.

   → sonnet · high (run) / opus · high (early checks)

5. **Pre-registered reads, then STOP (≈ 1.7 GPU-h).** The full p2s read set (as Z1), plus walkstop es1 + es2 for the Z2
   gates. **Adoption-grade if all hold:**
   - **STILL** on suite es1 + es2 (all six clauses).
   - **Six down:** ≥ 90 % of stands have a six-down share ≥ 0.90 (all six feet > 50 N), on suite stands (steady
     window) and walkstop stands (last 5 s), at both seeds. Baselines on this statistic: Z1 walkstop 22–24 %, HOLD
     34–35 %, REFS 0 %.
   - **Homed:** ≥ 90 % of walkstop stands homed, at both seeds. Z1, HOLD and REFS: 0 %.
   - **Re-stance cost (walkstop):**
     - stand falls ≤ max(1, Z1's) per 300, counted over all three stands;
     - ≥ 95 % of stands settle within the stand, and the median settle ≤ the probe's median + 1 s;
     - post-settle steps ≤ 0.5/s, taps ≤ 1/s, KS ≤ 0.5.
   - **Ladder:** no OBSTACLE LOSS / FLAT FAIL / COST; attribution LEVER against C0R.
   - **Reported:**
     - Z1's baselines for each gate;
     - fidelity to the probe (settle time, lifts, six-down);
     - walls against Z1 (n = 400);
     - stopped hip yaw.
   - **If Z2 reads PARTIAL or COST: stop and ask.** Z1 stays the working stand.

   → opus · high decides; ✓ fable · high

6. **Write-up.** CHARTER §12 and REPORT; the deployment contract gains the sample-and-hold proxy. Replicates and the
   bake are yours.

   → opus · high

## Budget

| Item | GPU-h |
|---|---|
| Inertness bridge | ≈ 0.45 (≈ 2.4 worst case) |
| Probe, with tuning and the residual pass | ≈ 1.2–1.7 |
| Z2 training | ≈ 2.2 |
| Reads | ≈ 1.7 |
| **Total** | **≈ 5.6–6.1** (≈ 8.1 worst case) |

Steps 1 and 6 are CPU only.

## Quiet-failure risks and guards

1. **The held proxy goes stale.** A parked leg reads "loaded" after another leg's press has taken its load.
   - Guard: the last tripod seats against a determinate tripod; the six-down gate is read on the last 5 s; the
     sequential-seat fallback.
2. **R has a discontinuous target** (lift / seat / tripod switches), learned only from noisy rows.
   - Guard: the probe proves R closed-loop; the step-4 dose band and branch shares; step 5's fidelity table compares
     Z2 to R.
3. **Homing drags a loaded foot.**
   - Guard: a loaded cam gets 0 outside the home band and a ±0.03 trim inside it (at most about 1.7° of hip yaw);
     the post-settle KS clause.
4. **Train/eval mismatch.** Training stands see live proxies (the noise keeps the gate open); eval stands see held ones.
   - Guard: by construction the held value equals the last live value; the probe runs the eval regime directly.
5. **The re-step costs stability on terrain.**
   - Guard: field_turn and field_hard stand falls; the ladder's COST clause.

## Verification

- Unit tests and the fable review (step 1).
- Bitwise bridge rows (step 2).
- The closed-loop premise probe (step 3).
- +200 wiring and the early checks (step 4).
- INVALID on every stamp: needs `STAND_OBS=clock` + `STAND_PROXY=hold`; the raw bit must match the labels.
- Each verdict goes to CHARTER §12 and is push-notified.

## Left for implementation (checks, from the audit)

- **All three walkstop stands.** `p2s_stand.stand_hold_index` finds only the first "stand"; every new estimator must
  enumerate all three. Re-derived over 300 stands, six-down share is Z1 21.7 / 24.3 % and HOLD 34.7 / 33.7 %.
- **Z1's suite es1 stamp** is at `logs/rsl_rl/gait_eval/omni_phases/2026-10-07_omni_phase2/seed18_X16_p2r_A15pB/suite__omni`,
  not under `lr/`.
- **Calibration thresholds.**
  - The calibration pass needs provisional θ: take them from the knee-inclusive readings.
  - Home in the air to a band tighter than ε_c, so a coast after landing stays inside 0.10.
- **R's unit tests.**
  - The seat law is proportional, so its equilibrium sits below θ_S with a standing extension. Pin that in the unit
    cases.
  - Re-run R's mirror test on the final constants. Only the fore-aft mirror is trained (`MIRROR_IMAGES=fa`).
- **Dose bank.** The fork's first rollout is S1 marching under noise (DIAG_REFSAMP_s18), not the zero-mean statue.
  The raws hold joint states, not full obs, so rebuild obs carefully. A false fail at +200 is loud.
- **Gate rows.**
  - The residual pass runs es1 only; apply the gate clauses to the rows it runs.
  - Only field_turn (n = 100) tests the re-step on terrain. 2 SE against HOLD's 0 falls is 0, so one fall lands in
    the stop-and-ask.
- **Training guards.** Under noise the training stands re-step constantly. Read stand-slot terminations separately
  before acting on an episode-length kill or a crab_failure notify.
- **Bookkeeping.** After the bridge, add Z2's code_sha to `p2_read`'s INVALID accept rule.
- **Fidelity scorer.** Use the eval path's `hist_encoding` setting (the drift scorer and the anchor use False).
- **One job.** Run Z2 start to finish as one job; a resume resets the priv_reg ramp and the LR.
- **Seeds.** Calibration and tuning use es1; report es2 separately as the held-out seed.

## Out of scope (not run)

- The sag-signal and cams-only variants (fallbacks on your call only).
- Replicates and the bake.
- Hardware runtime code for the sample-and-hold.
- Push-recovery evals.

## Amendment 2026-10-10 (user decision after the step-3 premise STOP; CHARTER 12.1)

The current-sense seat law failed the premise:
- stale holds deadlocked 116 / 294 walkstop stands;
- the press chattered at 37 taps/s on the suite.

**Z2 is now CAMS-ONLY with a kinematic re-step.**
- **R drops load sensing.**
  - The leg phase comes from the hip-pitch position: down near default, up near the lift target.
  - A tripod steps only while the other tripod is all down.
  - Cams home only while their leg is up; the in-band trim applies when down.
  - All targets are absolute.
- **`KRABBY_OMNI_STAND_PROXY` is not used** (it stays off).
- **Gates:**
  - six-down is REPORTED, not gated;
  - settle = homed held 0.5 s;
  - every other gate, the premise gate and the ladder are unchanged.
- **Before training:** a fresh probe (calibration + up to 3 tuning passes) and a short bridge for the code move.
