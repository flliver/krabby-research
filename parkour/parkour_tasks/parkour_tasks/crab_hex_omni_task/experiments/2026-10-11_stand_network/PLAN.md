# Plan — a dedicated STAND network for the omni crab (campaign `2026-10-11_stand_network`)

> **Audit trail (cap 3):**
> - Pass 1 (opus·xhigh; 3 blocking): col-15 / obs parity between trainer and eval unpinned → D5 "Obs parity" rule
>   (STAND_OBS=0 both sides, adapted read-validity rule) + smoke obs-parity asserts; pilot trigger uncalibrated →
>   relative to R on the same statistic; judged checkpoint unstated → final checkpoint of each run.
> - Pass 2 (opus·xhigh + fable·high; 4 blocking): student forward path unpinned (hist vs priv latent) → D1 one rule
>   + offline action-parity smoke; pilot R reference from a different curriculum window → concurrent 10 % R-executed
>   episodes; walkstop exit-handover falls ungated → G3 exit clause (`p2r_hold_reads`); unbounded DAgger buffer →
>   capped reservoir + smoke memory projection.
> - Pass 3: no blocking findings — converged (opus·xhigh + fable·high).
> - Gate check: 5 gates; all calibrated on existing stamps except R's field_hard es2–4 and walkstop onset (run/read
>   in step 2); G4 margin 0.05 > the ±0.04 process-state flip.
> - Premises: R's inputs inside the actor view (holds); R works from S1 handovers (holds, RSTg); DAgger fixes the
>   Z2c/Z2d coverage gaps (untested → step-7 pilot stop vs concurrent R).
> - Left for implementation (check, not edited in): overlay SLOTS must change with the stand-0.5 pool (record as a
>   deviation); `resampling_time_range` set in the trainer; pre-state the identity-bridge rule if bitwise misses
>   (process-state flip); G4 field_turn is es1-only, define "band"; row-level key for per-leg R branches and
>   LIFT onsets; fixed gradient steps per iteration; read wrapper keeps the `clock` validity rule for R stamps (R's
>   es2–4 field_hard runs under `clock`) and `0` for heads; history masking is partial (misaligned tail); smoke
>   branch shares on executed-policy rows; calibrate REFS/HOLD G2 via `z2_stand_gates` on summaries; exit clause is
>   a raw count per env seed (200 exits), None → INVALID; offline action parity: same batch shape or stated atol on
>   GPU; launcher must write `p2r_stand_mode`; review the mask with the actor's own 75-wide slicing; state whether
>   the scan encoder trains.
> - Size: 12 steps / ~2,400 words (reviewed: 12 / ~2,130); scope: no flags.
> - Routing: 3× opus·xhigh auditor, 2× fable·high auditor, 2× sonnet inventory/probes, main classification.

## Context

S1 s18 (`policy/6_p2a_field/model_19994.pt`) walks well (field_hard 0.8125) but marches in place at stand
(2.83 steps/s, 0.135 m lifts). Two in-network routes failed: p2s bled into walking (0.81 → 0.62), p2r gave at
best a still statue (Z1) and its distillations of the scripted re-stance controller R never homed closed-loop
(Z2c/Z2d: open-loop gates passed, walkstop homed 0–2 %). What works: a separate controller routed at the latched
stand slot costs walking nothing, and R routed that way (probe `P2R_RSTg_s18`) homes 97.0 / 97.7 % of walkstop
stands, settle median 2.92 / 2.96 s, STILL, 0 falls. This campaign trains a **separate stand network**, deploys it
next to the frozen S1 through the same stand-mode switch, and judges it closed-loop against S1-alone (REFS), HOLD
and R. Nothing is baked by this campaign; bake and adoption are the user's.

Campaign dir: `crab_hex_omni_task/experiments/2026-10-11_stand_network/` (CHARTER.md = pre-registration + log,
PLAN.md = this plan, `scripts/`, read JSONs; runs and banks stay untracked per `experiments/.gitignore`).

## Design decisions (recommendation each)

**D1 — Architecture and inputs: S1-compatible network, actor-side mask.**
- The stand net is a `CrabHexActorCriticRMA` checkpoint with S1's obs width 1149, layout and action dim 18,
  warm-started from S1's weights. Its estimator is S1's, frozen. **Forward path (one rule):** the action the
  student executes in the DAgger loop and the output its MSE fits are both the launcher's exact call,
  `act_inference(mask(obs_with_own_estimate), hist_encoding=True)`; the priv-latent path is unused and the history
  encoder stays frozen at S1's. Reason: R is a memoryless function of obs cols
  3-4 (roll/pitch), 18-28 (cams) and 29-34 (hip pitch), all inside the actor's 75-wide prop view, so the input set
  already suffices; the existing `_load_extra` / runner factory / `p2_read` path takes it unchanged; a new obs
  set buys nothing R needs and costs a new loader plus a new training env.
- The p2r copycat channel (own last action + current-sense proxy, lesson 4) is handled by a **clone-side mask**
  on the stand net's actor input only (cols 65-88 and their history frames, via the existing
  `crab_hex_omni_restance.stand_mask_obs`), applied *after* the estimator reads the unmasked prop. One function,
  called identically by the trainer and the launcher. The env-level `KRABBY_OMNI_STAND_MASK` is **not** used, so S1
  never sees masked frames after exit. Masked vs unmasked is the one screened arm pair (D3).

**D2 — Training environment and initial states: S1-in-the-loop handovers.**
- Task `Isaac-Crab-Hex-Omni-Teacher-v0`, preset 2a + the `P2__S1_s18` overlay verbatim (wall field, terrain
  curriculum, push DR ±0.5 m/s every 8 s, mass/CoM DR, 40-s episodes), with the command pool set to stand ≈ 0.5
  and the S1 walking/turn categories for the rest, resample interval drawn over a range (≈ 3–12 s).
- Frozen S1 acts on non-stand rows; the student (or R, per β) acts on stand rows. So every stand starts from a
  real walkstop handover (all seven categories, on flat pad and on blocks), from spawn when the first draw is
  stand, and repeats walk→stand→walk inside an episode — exactly the deployment contract (D5). No state bank or
  RSI change is needed (RSI can only stage `SLOT_MOVE`).
- Off-home coverage (lesson 6): DART-style action noise on a share of stand rows (per-episode σ ∈ {0, 0.1, 0.3}),
  plus the pushes. Labels are always R on the state actually visited.

**D3 — Objective: DAgger imitation of R (online, closed-loop), not RL, not offline BC.**
- R labels every visited stand row on GPU (`restance_target`, validated constants as the full string:
  `lift=-0.5,rho=1,up_frac=0.5,down_frac=0.4,k_c=2,k_h=1,trim=0.1,eps_c=0.1,eps_air=0.05,tilt_max=0.1,eps_step=0.3`).
  β-mixture, drawn per episode: R executes stand rows at the start, β → 0 over the first third, then the
  student's deterministic mean executes; a fixed 10 % of episodes stay R-executed for the whole run (labelled data
  too, and the concurrent control for step 7). Aggregated buffer, capped (reservoir); MSE on the actor mean; minibatches **balanced by R branch** (HOLD / WAIT /
  LIFT / HOME / SEAT) with LIFT-onset rows (branch changes to LIFT) oversampled.
- How this addresses each p2r failure: own-action/proxy feedback → masked (arm `A-mask`) and, in `A-full`,
  learned on the student's own action history; onset discontinuity → onsets are where a failing student keeps
  its cams > eps_step, so DAgger keeps generating LIFT labels there, plus oversampling; clean-state-only data →
  student-visited + DART states; σ ≈ 0.8 noise → no PPO σ, deterministic mean + scheduled DART; open-loop gates →
  no open-loop gate at all, only closed-loop gates (D6) and an observe pass.
- RL (stand rewards on a stand-only env, or a PPO fine-tune past R) is not run here; it is the fallback in
  "Later phases" if both arms fail.

**D4 — Code placement: campaign-local, eval_sha and code_sha unchanged.**
- Trainer and launcher variant live under the new campaign's `scripts/` (excluded from both hashes). The
  phase-2 launcher `p2r_mode_eval_crab_hex_gait.py` is imported and extended, not edited; the locked harness is
  patched at runtime as p2r did. No change to `mdp/`, `config/`, `gait_eval/`, scenarios files,
  `eval_categories.py`, `run_phases.py`. If implementation finds a shared-code change unavoidable: stop and ask,
  and it then needs a knobs-off inertness pair against REFS (`inertness_bridge.json` rule).

**D5 — Switch contract and deployment (inherited from p2r, lesson 11).**
- Entry on the first latched-stand step, no debounce; exit on the step the slot changes; turn holds stay "turn";
  latch re-derived on reset. Hard switch, no blending (p2r handover falls 0 with a hard switch).
- In: the student sees S1's last 10 frames in history (masked away in `A-mask`). Out: S1 sees the student's
  actions in its history — measured by walkstop onset and exit falls.
- Deployment knobs: launcher variant `stand_net` with `--stand-ckpt <path>` and `--stand-mask {0,1}`; the S1
  EVAL_CARRY block (CLOCK_HIDE=1, STD_PARAM soft, INIT_STD 0.8, ellipse 0.35:0.35:0.25, …) for both nets. A
  shipped two-network bundle is a bake question, out of scope.
- **Obs parity (one rule):** trainer and launcher both run `KRABBY_OMNI_STAND_OBS=0` (col 15 = 0, as S1 / REFS /
  HOLD), the stand latch derived by the launcher (as HOLD) and in training by the pool's stand slot. The campaign
  read wrapper reuses `still_clauses` / `z2_stand_gates` unchanged but replaces p2_read's Z2 "`STAND_OBS=clock`
  or INVALID" check with: `STAND_OBS=0` + `run_meta.p2r_mode.variant == stand_net`, fail-closed otherwise.

**D6 — Pre-registered closed-loop gates (all existing statistics, calibrated in step 2 before any training).**
Hard gates, each at es1 (tune) and es2 (held out); field_hard pooled es1–4 (n = 400):

| gate | statistic (existing code) | line | known-good must pass | known-bad must fail |
|---|---|---|---|---|
| G1 STILL | `p2s_stand.still_clauses` on suite stand (steps ≤ 0.5/s, taps ≤ 1/s, clearance ≤ 0.05 m, off-ground ≤ 0.10, hover ≤ 5/100, foot-speed sum ≤ 0.5 m/s) | absolute (P2S) | R, HOLD | REFS (steps 2.83), ST2M (taps 16) |
| G2 HOMED | `z2_stand_gates`: walkstop homed ≥ 90 %, settled ≥ 90 %, median settle ≤ R + 1 s (3.92 / 3.96 s), post-settle stillness | Z2 gates, `--probe P2R_RSTg_s18` | R | HOLD, Z1, Z2d, REFS |
| G3 FALLS | walkstop stand falls ≤ max(1, Z1's) per 300 (`z2_stand_gates`); suite stand falls 0; walkstop exit-handover falls within 2 s ≤ max(1, R's) per 300 per env seed (`p2_read.p2r_hold_reads` `handover_falls_by_entry_phase`) | Z2 rule; exit clause new | R, HOLD | — (report) |
| G4 WALLS | field_hard ≥ REFS − max(0.05, 2·SE); field_turn completion ≥ REFS − max(0.05, band); field stand-fall rate ≤ HOLD + 2·SE(pooled diff) | P2S OBSTACLE LOSS + RSTg field clause | HOLD (0.810), R | — (C0R as attribution floor) |
| G5 HANDOVER | walkstop onset ≤ REFS + max(0.3 s, band); body height within 0.03 m of REFS | P2S COST | HOLD (0.12 s), R (read in step 2) | — |

Reported, not gated: six-down (static-indeterminacy lottery), hip-yaw mean/max, stand-hold facing drift, the
observe-pass per-branch |μ − R|. **Adoption-grade** = all hard gates at both env seeds on the chosen arm at all
three training seeds; 2/3 is reported as such. The user decides adoption.

## Steps

0. Short verified lit review: separate skill / stand controllers, DAgger with scripted experts, mode-switch
   handover in legged robots → `experiments/lit-review-stand-network.md`. Read before step 4; if it contradicts
   D1–D3, STOP and ask.
   → per the `lit-review` skill's own routing (its saved workflow) — the skill owns its stages and checks
1. Create the campaign dir; write CHARTER.md (D1–D6 verbatim, budget, stop points) and PLAN.md; commit those
   paths only.
   → sonnet · medium — transcription of this approved plan
2. **Calibrate the gates on existing stamps, no training spend:** run `p2_read` (`--p2r-z2 … --probe
   P2R_RSTg_s18`, STILL, COST, walls) on REFS_S1_s18, P2R_HOLD_s18, P2R_RSTg_s18, Z1, Z2dR1, ST2M; record R's
   walkstop onset and G4 rows (R's field_hard is es1 n = 100 only — run R field_hard es2–4 to n = 400, ≈ 0.3 GPU-h).
   Every row of the D6 table must land as listed; any mismatch → STOP, notify, ask.
   → opus · high — decides whether the gates mean what they claim; quiet if wrong
3. Launcher variant `stand_net` (campaign-local; imports the p2r launcher): load via `_load_extra`, actor-side
   mask flag, `run_meta` block with ckpt sha + mask, swap-live proof. **Identity bridge:** S1 itself as the stand
   ckpt, mask 0 → bitwise = REFS_S1_s18 on suite / walkstop / field_hard es1 (≈ 0.3 GPU-h). Unit test: launcher
   mask == trainer mask on the same obs.
   → sonnet · high — exact spec on an existing pattern; the identity bridge catches errors loudly
4. DAgger trainer (campaign-local): env per D2, frozen S1 + student routed by `stand_mask()`, R labels, β and DART
   schedules, branch-balanced buffer, saves rsl_rl-format checkpoints loadable by step 3. Unit tests: R-label
   parity with `restance_target`, mask parity, checkpoint round-trip through `_load_extra`.
   → opus · high — new training loop; a wrong routing/label/mask would train quietly on the wrong thing
5. Code review of steps 3–4 (including the A1 mode) against this plan (routing on the latched slot, labels on visited states, mask after
   estimator, no shared-code edits, hashes unchanged).
   → opus · xhigh — fresh context; the training spend rests on it
   ✓ fable · high — independent check; every arm rests on the trainer
6. **Smoke** (≈ 0.3 GPU-h, Kit cache checked first, `systemd-run` memory cap, `systemctl --user stop` only):
   ~100 iterations of each arm's exact config, asserting the realized dose — stand-row share ≈ 0.5, handover count,
   branch shares on student-visited rows with LIFT-onset ≥ 1 % after balancing, the β schedule function (asserted,
   not reached), DART σ realized; obs parity per D5 — the trainer env carries every obs-affecting var of the eval
   stamp (CLOCK_HIDE, STAND_OBS, SPEED_ELLIPSE, …), and cols 0–1148 of one stand row captured in the trainer and in
   `stand_net` from the same state agree; a trainer-captured obs row fed offline to the `stand_net` launcher's
   policy object returns the trainer's action bitwise (pins the D1 forward path); S1 runs deterministic
   `act_inference`; buffer memory projected to the final iteration count fits the cap; saved ckpt loads in
   `stand_net` and runs one walkstop episode. **A1-mode smoke** (Amendment 1): assignment ≈ ½ control and ⅙ per
   spoiler; executed stand rows are R or spoiler only and no optimizer step runs; DART σ = 0; the spoiler state is
   reached at takeover (statue executes exactly 0; chatter's realised σ ≈ 0.3 per channel; on most stuck-lift stands
   the held leg is up, hip ≤ R's up threshold, with |c| > 0.3); the stand-end value is read on the last stand row
   before the exit; scored-stand filter per A1.
   → sonnet · high — assertions fail loudly
7. **Pilot `A-mask` s18** (one job, no resume; cap 2 GPU-h; ckpts every 500). Early check at the β→0 point and
   midway, notify-only: train-time closed-loop homed share at stand end (|c| ≤ 0.1 all legs), restricted to σ = 0
   stands longer than 4 s, and exit falls. R's reference is the same statistic on the concurrent R-executed 10 %
   episodes over the same iterations (same terrain levels), excluded from the student's value; if the student's is
   < 0.5 × R's at mid-run, STOP and ask (DAgger premise failing).
   **Judged checkpoint:** the final checkpoint of each one-job run is the one evaluated in steps 9–10;
   intermediate checkpoints serve only these notify reads.
   → main (opus) — monitoring and the stop decision
8. `A-full` s18 (unmasked), same config otherwise, serial after 7.
   → main (opus) — same driver as 7
9. Evaluate both arms (suite, walkstop es1+es2, field_hard es1–4, field_turn, observe pass), read G1–G5.
   **Arm rule:** both pass → higher pooled walkstop homed; within 3 pp → `A-mask`. One passes → that one. None →
   if Amendment 1 A2's trigger holds, run `A-branch` (A2; decide-record-notify); otherwise STOP and ask (Later
   phases). Decide, record in CHARTER, notify.
   → opus · high — applies a pre-registered rule; reads are the numbers the user acts on
10. Replicates of the chosen arm at s16 and s17 (stand-net training seed only; the walking head stays S1 s18), same
    evals and reads.
    → main (opus) — chained queue per the GPU-chaining memory
11. Report (REPORT.md: per-seed G1–G5 table vs REFS / HOLD / R, observe-pass findings, adoption-grade verdict),
    commit campaign paths only, **STOP for the user's adoption/bake decision**, notify.
    → opus · xhigh — the synthesis the user decides on
    ✓ fable · high — independent check of the verdict against the reads

**Stop points (push-notify each):** step 2 calibration mismatch; step 0 contradicting the design; step 6b
recoverability stop (A1); step 7 premise stop; step 9 no arm passes and A2 not triggered; A-branch fails (A2); step 11
adoption. Step-9 arm choice and the A2 trigger are decide-record-notify (rules pre-registered).

## Amendment 1 (2026-10-12, user decision after step 0)

> **Amendment 1 audit (1 pass, opus·xhigh, 2026-10-12): 6 blocking, all fixed.** A1 mode unsmoked/unreviewed → built
> in step 4, reviewed in step 5, A1-mode smoke in step 6; scored-stand set undefined (falls / timeouts drift the
> ratio to PASS) → scheduled ≥ 8 s, not timeout-cut, a fall scores homed 0, last stand row; DART on in the control →
> σ = 0 in A1; no absolute floor → control homed ≥ 0.75 (user's number); A2 trigger too broad / mask from the wrong
> arm → "only failing gate is G2", that arm's mask; A1 PASS no longer launches step 7. Left for implementation: fall
> margin max(0.01, 2·SE(diff)) vs +0.01; stuck-lift knee held at −ρ·side·hip and cam by saturated P-law; A2 launcher
> must satisfy sn_calib's `variant == stand_net` rule (write `stand_net` + `branch_head`, or extend the rule); per-stand
> assignment carry-over to the next stand (report it).

The step-0 lit review (`../lit-review-stand-network.md` §12) found nothing contradicting D1–D3 and proposed
amendments. The user adopted two (recoverability pre-check, branch-gated third arm) and declined the rest (fast β,
gated takeover, onset weighting, σ from residual); D1–D6 and steps 1–11 stand otherwise.

**A1 — step 6b, recoverability pre-check (≈ 0.3 GPU-h; after the step-6 smoke, before step 7).**
- *Premise tested:* DAgger beats offline BC only where the expert recovers from the states the student creates
  (review §4); R was measured only from clean S1 handovers (P2R_RSTg_s18, 97 %).
- *Run:* the step-4 trainer env, D2 config except DART σ = 0 throughout (pushes and DR stay), β = 1 on every stand
  (R executes; no student, no update), one job. The A1 mode is built in step 4, reviewed in step 5 and smoked in
  step 6 (A1 smoke below). **Scored stands:** scheduled length ≥ 8 s and not cut by the episode timeout; a stand
  that falls scores fall = 1 and homed = 0 (falls before takeover reported separately); |c| is read on the last stand
  row before the exit, never on a row after a reset. Half of them (random per stand) are **spoiled**: a spoiler acts for the first
  U[1, 4] s after entry, then R takes over for the rest; the other half are R from entry (the concurrent control).
  Spoilers, one third each: (a) **statue** — zero action (Z1 / HOLD-like: unhomed cams, five-foot stands);
  (b) **chatter** — R's action + Gaussian noise σ = 0.3 on all 18 channels (ST2M-like taps); (c) **stuck lift** — one
  random leg's hip held at R's lift (−0.5 rod) with its cam driven to ±0.4 rad, the other legs per R (hover, |c| > 0.3).
- *Statistic:* the step-7 train-time homed share at stand end (|c| ≤ 0.1 rad, all legs), stand falls and exit falls,
  per spoiler and for the control, same iterations and terrain levels. Also reported: homed by R-time bin (time R
  had before stand end), and the share of loaded legs with |c| in (0.1, 0.3] rad at stand end (R's trim-only band).
- *Rule (fixed before the read):* PASS if the control's homed share ≥ 0.75 (absolute floor, user decision
  2026-10-12: R must label well in the env DAgger trains in; its 97 % is flat walkstop) and, for every spoiler,
  homed ≥ 0.8 × the control's homed and stand-fall share
  ≤ control's + 0.01; then record and notify — step 7 waits for the user's go. Otherwise **STOP and ask** (R strands a share of student-like states:
  the DAgger premise is weak there; options are A-branch first, the RL fallback, or an R fix).
- → main (opus) — a short run on the step-4 trainer; the read is a premise decision

**A2 — `A-branch`, a pre-registered conditional third arm.**
- *Trigger (read at step 9):* no arm passes all of G1–G5, and at least one arm's **only** failing gate is G2 (HOMED)
  — G1, G3, G4 and G5 all pass at es1 and es2 — the Z2d pattern the review attributes to the flat MSE head (§7).
  Otherwise A-branch is not run.
- *Design:* D2 env and D3 DAgger unchanged; the mask setting of the qualifying arm (both qualify → the higher pooled
  walkstop homed; tie → `A-mask`). Network: S1's trunk (warm start, D1 forward path), plus a per-leg branch classifier (6 legs × 5
  branches HOLD / WAIT / LIFT / HOME / SEAT; class-balanced focal CE on R's branch label) and per-branch regression
  heads (MSE only on rows of that branch). Deployed action per leg = the head of the argmax branch. A loader adapter is
  needed: launcher variant `stand_net_branch` writing the same `p2r_mode` fields (`specialist.sha256`, `stand_mask`)
  plus `branch_head: true`, under the D5 obs-parity rule and sn_calib's head-validity rule.
- *Steps when triggered:* implement + review (opus·xhigh, ✓ fable·high) → smoke as step 6 → train s18 (one job,
  ≤ 2 GPU-h, the step-7 premise stop against its concurrent R slice) → evaluate as step 9 → G1–G5. Passes all
  gates → it is the chosen arm and step 10 replicates it; fails → **STOP and ask** (Later phases).
- *Budget when triggered:* ≈ 3.1 GPU-h (smoke 0.3, train ≤ 2, eval 0.8); replicates come from step 10's budget.

**Budget after Amendment 1:** ≈ 12.3 GPU-h planned (the original ≈ 12 + 0.3 for A1), + ≈ 3.1 conditional (A2).
**User go (2026-10-12):** through step 6 (and 6b); stop before step 7.

## Budget, seeds

- GPU: calibration ≈ 0.3, identity bridge ≈ 0.3, smoke ≈ 0.3, 4 training runs × ≤ 2 = ≤ 8, evals 4 heads ×
  ≈ 0.8 = 3.2 → **≈ 12 GPU-h**, one Isaac process at a time. Training cost per iteration is measured in the smoke
  and the per-arm iteration count is set to fit the 2-GPU-h cap before launch.
- Seeds: screen at s18; replicates s16 / s17 (same-seed reruns replay bit-identically, so replicates are new
  seeds). Env seeds es1 tune / es2 held out; field_hard es1–4.

## Verification

- Gates calibrated on the six existing references before any training (step 2).
- Identity bridge: S1-as-stand-net reproduces REFS bitwise; `code_sha` / `eval_sha` (`1d021625`) checked
  unchanged by `p2_live.py --code-guard` before every launch.
- Smoke asserts realized doses; observe pass on every evaluated stand net; closed-loop gates only.

## Later phases (not run; planned only on your say)

- RL fallback: PPO stand network on the D2 env with stand rewards (support / slide / cam-home / still), R-BC as a
  regulariser.
- Smoothed-onset R (ramped LIFT at eps_step) re-validated as a probe, then DAgger on it.
- PPO fine-tune past R (heading drift, six-down) once a stand net is adoption-grade.
- Two-network bundle/export in `policy/` (bake decision).

Routing: strong · roster ok (opus = Opus 5.5, Claude Code 2.1.280) · planned on opus · ceiling: opus · 2× sonnet·high, 1× sonnet·medium, 3× opus·high, 2× opus·xhigh, 4× main, 2× fable·high check, lit-review per its skill, fable retry on 2 fails
Effort enforcement: prompt-level (tier agents not set up — say the word and I'll create them)
