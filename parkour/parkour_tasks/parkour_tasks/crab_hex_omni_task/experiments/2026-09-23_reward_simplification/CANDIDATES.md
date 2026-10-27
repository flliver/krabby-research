# Omni reward simplification: candidate catalogue

*2026-09-23. **Plan name:** omni reward simplification. It lists 38 candidates for simplifying the `crab_hex_omni_task`
reward. Each one can be planned independently: say "Plan the `<candidate name>` candidate from the omni reward
simplification plan" (protocol in §3).*

- **Long form:** evidence with file:line pointers, falsifiers, and the adversarial-review notes per candidate are in
  [CANDIDATES_DETAIL.md](CANDIDATES_DETAIL.md).
- **Literature:** [../lit-review-omni-reward-simplification.md](../lit-review-omni-reward-simplification.md).
- **Provenance:** a read-only research workflow, `wf_15be1644-c4a` (92 agents). It had:
  - 8 reward clusters, each with a repo dossier, a literature sweep and per-reference verification (96/97 findings kept
    at least one verified reference);
  - candidate design and a cross-cluster merge;
  - two-lens adversarial refutation;
  - a completeness critic;
  - an independent final check.

**Standing coordination**
- **Clock.** A separate session owns clock deprecation: `reward_clock_schedule`, `reward_clock_swing_apex`, the clock
  observation and the clock-speed gate. Clock-touching candidates are coordination points with it.
- **Motion quality.** The action-bound-loss fine-tune (M1b) is planned separately.
- **GPU.** Training is serial; the GPU is used only when the user releases it.

## 2. What the review found

1. **The biggest incomes pay for holding still.**
   - The two ungated exp kernels are track_ang_vel_z_exp at 24.0 % of |income| and track_lin_vel_xy_exp at 22.5 %.
     Together they are 68 % of positive income.
   - The terms that require motion (progress, yaw_progress) are 2.9 %.
   - The per-step zero clip binds. The pre-clip Σ income against the return is −27.0 vs +6.4 early in t3, and 43.7 vs
     52.0 late in c6, so about 16 % of the late return is forgiven.
   - Every income table and replay in the repo is pre-clip.
2. **13 names do nothing.**
   - 10 sit at weight 0 in every phase.
   - `reward_feet_edge` can never fire on the ring terrain: the riser threshold is 0.12 m and blocks are at most 0.072 m.
   - Air time and stride pay about 0 because of the float32 first-contact bug. The upstream IsaacLab fix (PR #7574)
     would switch them back on, silently, at weights 0.8 and 0.5 that the incumbent never trained with.
3. **Most weights are old and untested here.**
   - 12 of the 18 active flat weights were set on the May or era-A plants.
   - No omni ablation exists for any penalty.
   - The literature's stacks run about 9–12 terms with one tracking kernel per axis. The hexapod papers found omit the
     posture terms.
4. **Facing is priced three ways:** the facing exp, a yaw-rate kernel on a P-controller command, and signed yaw
   progress. yaw_progress earns only 0.1 %, but it carries about 40 % of the binding turn-row slope, so removing it is
   a dose cut, not a free deletion.
5. **The ring bonuses duplicate tracking.** `reward_obstacle_clearance_omni` has no lift or crossing factor; it is an
   upright-progress indicator. Together with the masked goal velocity it adds a 34 % income top-up, and 2b outcomes
   did not improve.
6. **The decision machinery can't adopt a simplification cheaply.** A clear ADOPT needs at least one BETTER, so an
   all-TIE removal buys a seed-2 pair, and single-seed noise is large (turn facing_closed MDE ≈ 0.15). A
   non-inferiority rule (#10) is the enabling step.
7. **End state if the main candidates adopt:**
   - flat active terms go from 18 to about 7–10;
   - teacher active terms go from 25 to about 9–12;
   - registered names go from 37 to 26 immediately (#1).

## 3. How a candidate gets planned

When the user says "Plan the `<name>` candidate from the omni reward simplification plan":

1. Enter plan mode.
2. Read this plan's section for `<name>` and the long-form section in
   `crab_hex_omni_task/experiments/2026-09-23_reward_simplification/CANDIDATES_DETAIL.md`. The detail file carries
   file:line evidence, falsifiers and the adversarial review notes.
3. Check the status of everything the candidate depends on: other candidates, the clock session, M1b, ladder state.
4. Re-verify the cited evidence at the current HEAD.
5. Plan the candidate with the repo's standing discipline:
   - the offline replay gate first;
   - serial GPU use;
   - same-length fork controls, because a resume resets the PPO counters;
   - never stacked with clock removal;
   - decided under #10 once it is ratified.
6. Route the plan's steps and run plan-audit as usual.

## 4. Candidates

Tags: **kind** · cost · confidence.

Waves:
- **W0**: now, no GPU.
- **W0b**: offline evidence and the decision rule.
- **W1**: clock-independent screens.
- **T**: teacher stage.
- **C**: with or after the clock decision.
- **X**: conditional or parked.

### W0 — cleanup, no behaviour change, no GPU

**1. zero-weight-purge** — cleanup · 0 GPU-h, 1–2 h · high
- **What:** unregister the 10 names at weight 0 in every omni phase and drop feet_edge from 2a–2c. The 10 are:
  feet_slide, base_pitch_forward_linear, cam_contact_schedule, motor_direction_reversal, ang_vel_xy, cam_phase_lock,
  dof_error, one_direction_spin, stance_support_feet, tripod_schedule. A fail-loud guard rejects any knob that targets
  a removed term. Omni only.
- **Why:**
  - All of them log exactly 0 in X08 t3–c6 and teacher 2a/2b.
  - Each has a null record, e.g. tripod was gradient-dead in 3 screens, and spin/phase-lock were null in era A.
  - feet_edge's mask can't fire on ring risers; its pixel index also drifts up to 0.8 m.
  - legged_gym drops zero scales and Isaac Lab skips None terms.
- **Decided by:**
  - Regenerated dumps differ only in the removed names.
  - Forward and sideways dumps are byte-identical.
  - Guard unit tests. No training.
- **Needs:**
  - Land it on a branch, because the clock session and M1b edit the same files.
  - `penalty_mechanical_power` stays, on purpose.

**2. dead-swing-income** — remove · 0 GPU-h · high
- **What:** remove `reward_feet_air_time_positive` and `reward_stride_length`, together with their phase-out ramp rows.
- **Why:**
  - They pay +0.0006/s and +0.00003/s. REPLAY_GATE shows air time identically 0 over 999 episodes (the float32 bug).
  - Even with the bug fixed, air time would be about 1.5 % and fails Gate 1. Stride duplicates progress and excludes
    turns.
  - An IsaacLab bump would silently revive both.
  - Standard stacks do keep a working air-time term, so this removal rests on repo evidence. The successor, if the
    clock goes, is #28.
- **Decided by:** a dump diff, plus a unit test that any knob or ramp naming these terms raises. No training arm,
  because a single-seed TIE is uninformative.
- **Needs:** ships with #1.

**3. config-truth-align** — cleanup · 0 GPU-h, 2–3 h · medium
- **What:**
  - **(c)** An omni sign guard. It refuses positive weights on the power, action_rate and delta_torques penalties. The
    forward L3 arm ran power at +0.001, which is a reward, and its record gets a note saying so.
  - **(a)** A user decision.
    - **a1**, recommended: keep the knob-timing pillar, and refuse to build an omni env without `KRABBY_OMNI_PHASE`.
    - **a2**: set the class literals to the shipped values. Six of them disagree in every phase: track std
      0.141 vs 0.2, ang std 0.15 vs 0.5, L1 −0.5 vs −2.0, clock weight 0 vs 1, apex weight 0 vs 1, combine sum vs
      product. The seventh, tracking_yaw, differs only by phase.
  - **(b)** After the clock goes, collapse the clock literals and combine knobs.
- **Why:**
  - About 52 `KRABBY_*` variables shape the reward and terminations.
  - An env built without a preset silently trains at ang std 0.15.
  - The env scrub has already voided the 1c canary once.
- **Decided by:** dump diffs and unit tests.
- **Needs:** the user's choice on (a); (b) waits for the clock.

**4. dormant-lever-retire** — cleanup · 0 GPU-h, 2–4 h · medium
- **What:**
  - **Tier A:** delete three never-trained levers and their knobs. They are `track_lin_vel_xy_exp_facing_coupled`,
    `track_ang_vel_z_exp_turn_gated` and `penalty_stand_pose_l1`. The reward table is unchanged.
  - **Tier B:** unregister `reward_facing_turn` if #16 does not adopt it.
- **Why:**
  - C4, C6 and C11 were removed unrun.
  - The turn-gated kernel rewards not settling, by construction.
  - Stand pose failed Gate 1 (rho −0.61).
- **Decided by:**
  - Tier A: the dumps are unchanged.
  - Tier B: the dumps differ only by the removed entry.
  - The frozen ladder fixtures stay unedited: the validator accepts retired knobs in historical mode, or the fixtures
    are checked against a pinned commit.
- **Needs:** the ladders that own these knobs have closed; Tier B comes after #16.

**5. height-backstop-tag** — fix-then-decide (terminations) · ~0 GPU-h · high/low
- **What:**
  - total_terminates' roll/pitch > 1.5 rad branches are strictly subsumed by crab_failure at 0.5 rad. Keep or remove
    them behind an assertion that limit_angle < 1.5.
  - The root-z < −0.25 branch is a real failure tagged time_out=True, so PPO bootstraps it as a timeout. Count how often
    it fires, and retag it only if the count is material.
- **Decided by:** an offline count and a unit test.

**6. removal-knob-batch** — enabling · 0 GPU-h, ~0.5 day · high
- **What:** two landings.
  - **(A)** Driver only: dump checks and parsers in run_candidates.py for knobs that already exist. code_sha does not
    move.
  - **(B)** Weight knobs for the live terms that have none: foot_idle, tibia, lin_vel_z and penalty_lin_vel_y.
    Preferably one generic `KRABBY_REWARD_W__<term>` mechanism.
  - Screens set weights to ε = 1e-3, which keeps the telemetry.
- **Why:** editing the code pathspecs in the middle of a ladder invalidates recorded screens and re-runs the anchors.
- **Decided by:** all 16 preset dumps byte-identical at defaults, plus one unit test per knob.
- **Needs:** land (B) at a campaign boundary. Coordinate the date with the clock and M1b landings, and put it *before*
  the clock landing if #20 is to run.

**7. gait-gate-decouple** — coordination seam, offered to the clock session · 0 GPU-h · high
- **What:**
  - **Variant A**, a proven no-op: the omni `_cmd_gate` used by foot_idle, excess_contact, stance_support, foot_clearance
    and swing_min_clearance becomes "‖cmd_xy‖ > 0 OR cmd_wz ≠ 0". That lets the clock session change the clock coupling
    (r_turn) without moving these gates.
  - **Variant B**, a behaviour change needing a replay and a screen: gate on the command slot instead, which closes the
    stand-slot facing-drift leak.
- **Why:** today's gate already equals variant A for any r_turn > 0.6, and the knob rule forbids r_turn = 0.
- **Decided by:** an analytic-invariant unit test.
- **Needs:** nothing. Close it if the clock session absorbs it.

### W0b — offline evidence and the decision rule

**8. survival-floor-audit** — fix-then-decide · A: 0 GPU-h; B: ≤7.2 GPU-h · medium
- **What:** `clip(total, min=0)` (`parkour_reward_manager.py:38`, no flag) forgives about 16 % of the late return.
  - **Stage A** measures the forgiven mass per phase and slot, from TB logs and logging-only telemetry. From then on,
    every removal candidate reports its post-clip effect.
  - **Stage B** runs only if some removal is projected to push the forgiven mass above about 1.5× X08:c6's. The default
    outcome keeps the clip and declares it. The b0 arm (plain sum, the Isaac Lab form) is the only one that tests
    whether the clip is load-bearing.
- **Lit:** legged_gym clips (`only_positive_rewards`); Isaac Lab does not. Reda et al. describe the survival-bonus
  minima.
- **Needs:** nothing. It precedes every removal replay.

**9. term-discrimination-replay** — evidence · 0–0.2 GPU-h, 2–4 h · high
- **What:** one offline table. For each live term, its share of the walker-minus-degenerate NET margin per category.
  - Degenerate populations: creep, the bang-bang M1 heads, falls, stand fidgeting.
  - Computed with and without clock income, and post-clip.
- **Fix fidelity first.** Today's replay misses training income by 2–13× on exactly the contested terms. The preferred
  fix is a per-term reward-manager dump in the `--save-raw` eval (about 0.2 GPU-h, queued behind the GPU campaign).
- **Scope:** it ranks terms for screening only. A removal still needs a training screen.
- **Needs:** #8 Stage A.

**10. ni-adoption-rule** — decision protocol · 0 GPU-h, ~1 day · medium
- **What:** a non-inferiority path for candidates declared `kind: simplification`:
  - margins in metric units per protected member, fixed in advance, not in multiples of R;
  - protected rows chosen where the penalised behaviour would appear;
  - a cumulative budget against one fixed anchor;
  - move_turn and diag survival as hard one-sided guards;
  - at most 3 deferred adoptions before a pooled checkpoint.
  - The motion ladder's frozen rule is untouched.
- **Lit:** TOST equivalence testing (Lakens 2017); bio-creep (Everson-Stewart & Emerson 2010).
- **Decided by:** driver unit tests, plus retro-scoring D11 and the X0/B0 seed pairs as synthetic nulls. Then the user
  ratifies it as a charter amendment.
- **Needs:** a seed-2 X08 run (about 3.6 GPU-h) when a scratch arm first needs one; none exists.

**11. contact-telemetry-audit** — fix-then-decide (teacher) · 0–1 GPU-h; screen 5–10 only if triggered · medium
- **What:**
  - **Step 0, CPU on existing npz:** classify femur and hip contacts as ground or leg-leg, and stumble flags by facet.
  - **Step 1:** read-only force telemetry, plus a tilt/hip split of crab_failure in the logs.
  - **Step 2, decisions:**
    - stumble (−1.0): remove it, make it geometry-aware, or accept it. Either way it is priced once.
    - collision (−2.0): keep by default. The repo reads femur contact as leg-leg interference, which is a hardware
      constraint.
- **Why:**
  - At friction ≤ 1.5, stumble's |F_xy| > 4|F_z| test can't trip on flat ground.
  - The ring risers are 32–51° ramps, not walls.
  - Femur contact happens on 1.4–2.1 % of steps and is not tied to obstacles.
- **Needs:** any screen comes after #22.

### W1 — clock-independent screens (fork X08:c6 or the M1b head; formation where noted)

**12. cross-penalty-remove** — remove · 0.8–1.6 GPU-h · medium
- **What:** remove `penalty_lin_vel_y` (the omni `penalty_cross_command_vel_l2`, −3.0) in 1a–1c. 2a–2c follow only after
  a teacher check.
- **Why:**
  - It is 0.5 % of income.
  - Full-vector tracking stacks (legged_gym, Isaac Lab, CaT) price the cross error inside the kernel.
  - Against: omni also has projection-only terms (progress, ring goal_vel), which is the regime where papers keep an
    orthogonal penalty, and this term carries about 34 % of the large-error tail slope.
- **Decided by:** a 2000-it fork against the X01/X02 fork controls.
  - Falsified if diag or move_turn cross_track_ratio rises ≥ 0.025, or direction_error rises ≥ 0.05 rad.
  - Formation rides along in the next scratch chain.
- **Needs:** #6 and #10; softly after #25.

**13. posture-penalty-drop** — remove by zeroing · ~2–2.5 GPU-h · medium
- **What:** set `reward_orientation` (−0.7) and `reward_lin_vel_z` (−0.15) to 0 and keep both registered.
- **Why:**
  - They are 0.11 % and 0.05 % of income.
  - The replay cost is the same for walker, creeper and faller, and there is no leading signal (pre-onset pitch is about
    0.012 rad).
  - Doses were null in era A and era B. The hexapod papers omit both terms.
  - Against: every stock quadruped flat config keeps orientation on and v_z at −2.0.
- **Decided by:** a CPU leading-window test, then forks on s3 and s2. A fork shows only "harmless at convergence"; a
  claim about formation needs a scratch arm, or folds into #20.
- **Fallbacks:** on REJECT, drop lin_vel_z alone; then #37 and #38 apply.
- **Needs:** #6 (the lin_vel_z knob) and #10. The no-clock recipe gets its own posture-off check before the terms are
  unregistered.

**14. tibia-pose-retire** — remove · ~1.8 GPU-h · low–medium
- **What:** remove `penalty_tibia_deviation_in_stance` (−0.28, 1.4 %).
- **Why:**
  - It was added on 2026-05-23 on crab_simple, with position actions.
  - The 2026-08-20 rebuild moved q_default to mid-stroke, and the term was never re-tested afterwards.
  - Under the saturated mean it acts as a tax on the number of loaded knees, working against six-down stance. That is
    the same family as the dropped M6 stand pose.
  - legged_gym and Isaac Lab have no contact-gated precedent.
- **Decided by:** an offline R² of the cost against loaded-feet count vs knee angle, then forks on s3 and s2.
- **Fallback:** only if M1b is adopted: a command-gated stand_still form.
- **Needs:** #6 and #10.

**15. foot-idle-audit** — fix-then-decide · ~0 / 1.8 / 7.2 GPU-h · medium
- **What it prices:** `penalty_foot_idle_when_forward` (−0.12) is 7.9 % of income, the 4th largest term. It prices a
  foot that stays *airborne* for more than 90 steps.
- **Defects:**
  - The cost is unbounded and quadratic.
  - The counter keeps running while the gate is closed and pays out when it opens (carry-over across slots).
  - Stand-slot facing corrections open the gate.
- **Steps:**
  1. The existing replay with `--idle reset` vs carry. It splits carry-over from stand accrual and from real parking.
  2. Only if carry-over is at least about 1/5 of the cost: test a reset and a per-foot cap separately, on forks.
  3. Removal only in a scratch chain after the clock decision. Removing both the clock and this guard is the
     documented failure regime.
- **Lit:** no verified bounded anti-parking precedent.
- **Needs:** #8. Conflicts with #20, which drops the term.

**16. turn-slot-facing** — replace, env only · ~1.8 GPU-h · medium
- **What:** `KRABBY_YAW_W=0` and `KRABBY_OMNI_FACING_TURN_W=0.2`. The facing price stays exactly as today on the turn and
  move_turn slots, and becomes 0 on stand and move rows, where it pays 0.17–0.19/s whatever the robot does.
- **Decided by:** a replay (under 0.5 h), then forks on s3 and s2. heading_final and stand drift are promoted to guards.
- **Needs:** #10. It is the alternative to #17 part A.

**17. one-yaw-kernel** — merge · ~9–13 GPU-h per part · medium
- **What:** move to one facing representation, the reference pattern of a P-controller plus one rate kernel
  (track_ang_vel_z_exp).
  - **Part A:** remove `reward_tracking_yaw` (the facing exp): 5.0 % of income, near-constant, never ablated.
  - **Part B**, only after A: a dose cut of yaw_progress, or a dose-preserving merge into the kernel.
- **Caution:** yaw_progress carries about 40 % of the binding turn-row slope. D10 took its weight from 1 to 3 and moved
  facing_closed 0.419 → 0.748. Removing it predicts about −0.16, which is at the MDE.
- **Decided by:** an offline gradient gate, then a fork kill-switch, then a from-scratch C1d chain per part against X08,
  with a registered margin (e.g. pooled Δ turn facing_closed ≥ −0.10).
- **Needs:** #8 and #10. Never stacked with clock removal.

**18. delta-torque-drop** — remove · ~1.8 + 4–6 GPU-h · medium
- **What:** remove `reward_delta_torques` (−1e-6, 2.3 %).
- **Why:**
  - It has never been tested alone; it only ever moved jointly with action_rate, in the position era.
  - legged_gym and Isaac Lab don't have it.
  - The walker pays 3–4× what the creeper pays.
- **Decided by:** a fork on the post-M1b head (s3 and s2), plus a short formation chain judged on a tippy-tap proxy and
  swings_per_m, which a fork cannot see.
- **Contingency:** #31.
- **Needs:** #10 and #3's sign guard; runs after M1b.

**19. rate-follows-bounds** — M1b follow-on · 1.7–3.5 GPU-h · medium
- **What:** one extra arm added to M1b's plan: M1b plus a re-derived rate term. Squared first (plus second) difference,
  weighted from M1b's own rollouts to at most 10 % of task income (prior −0.01…−0.03). It replaces the unsquared −0.3
  (11.4 %) and retires the smooth2 knob.
- **Why:**
  - Once M1b bounds the mean, the stock term would charge a floor of about 0.63–0.77/s against 1.5–2.1/s of task
    income.
  - The −0.3 bake is position-era.
  - legged_gym and Isaac Lab use −0.01 squared. Seyde et al. found rate costs curb bang-bang control but cost
    performance.
- **Decided by:** after M1b's first result, 2000-it forks on 2 seeds, judged on non-circular Q, T, F and the saturation
  diagnostics.
- **Needs:** M1b's first result. It is the alternative to #32.

**20. minimal-reward-arm** — structural diagnostic · ~12 GPU-h · medium
- **What:** 9 terms off (foot_idle, tibia, excess_contact, progress, the cross penalty, orientation, lin_vel_z, air time,
  stride), so 18 active flat terms become 9. legged_gym ships 9 nonzero scales.
  - The term set is frozen at pre-registration.
  - It runs on the clock-bearing recipe, before the clock goes. This is a hard gate: it needs a working swing term.
- **Decided by:** a post-clip subtraction on C1d raws, then a kill-only fork, then from-scratch C1d on s3 and s2 plus a
  mandatory seed-2 X08 replicate.
- **Add-back order** if it fails: excess_contact first, then foot_idle.
- **Relations:**
  - It pre-empts #12–#15: cheaper, but it loses attribution.
  - It conflicts with #17 and #25.
  - Run it or #21 first, not both blind.
- **Needs:** #8 and #10, and #6's knobs landed at the boundary before the clock landing, or a pinned commit with its own
  overlay. That clears the dependency loop the final check found. The M1b base is optional.

**21. adopted-set-formation-check** — structural · ~7.6 (+3.9 anchor) GPU-h · medium
- **What:** one from-scratch C1d lineage carrying the union of at least 2 fork-adopted removals, judged against one fixed
  anchor under #10's cumulative budget.
- **Why:** fork verdicts are relative to a continued-training head. A 2000-it fork alone moved move_turn facing
  0.235 → 0.411.
- **Needs:** at least 2 adoptions, and a named post-clock anchor.

### T — teacher stage (needs a 2a head)

**22. ring-bonus-drop** — remove · 7.5–10 GPU-h · medium
- **What:** in 2b/2c, zero `reward_obstacle_clearance` (0.9) and `reward_tracking_goal_vel` (0.75).
- **Why:**
  - The omni "clearance" is ring tile × no-stumble × progressing × moving × upright. It has no lift factor and pays
    regardless of obstacles.
  - It is on for 51 % of move steps even for a head never trained with it.
  - The two terms add +0.216/s (a 34 % top-up), and ring completion went 0.56 → 0.54.
  - Removing them also ends stumble's double pricing.
  - The parkour papers use one progress reward plus a curriculum (Zhuang 2023, Cheng 2023).
- **Decided by:** paired 2b forks from a 2a head on s3 and s2, scored on obstacle-conditioned crossing metrics.
- **Fallback:** drop only the clearance term.
- **Needs:** #10 and a 2a substrate. No C1d-lineage 2a exists; growing one costs about +2.25 GPU-h per seed.

### C — with or after the clock decision (coordination with the clock session)

**23. apex-term-retire** — remove after formation · ~4.1 GPU-h · medium
- **What:** zero `reward_clock_swing_apex` from the first fine-tune stage on, as a pre-registered step of the clock work.
  It is moot if the clock goes.
- **Why:**
  - It never delivered lift. "Reward routes for lift exhausted" is the forward record.
  - Forward already phases it out after formation.
  - Its root-relative reference lets a crouch earn it with the feet down.
- **Decided by:** an offline stance/swing decomposition (needs clock_phase logged in the eval raws), then paired forks.

**24. clock-hold-twin** — phase-out handover · 3.6–10.8 GPU-h · low–medium
- **What:** offer the clock session an income-only anneal of `reward_clock_schedule`, keeping the clock observation,
  against a clock-held control.
- **Why:** omni has never ablated the clock or held a control. When the clock stops, it is the only income for keeping
  six feet still at stand.
- **Close it if** the clock session's design already includes a held control.

**25. progress-relative-cap** — replace · parity 0; promoted 3.6–7.2 GPU-h · medium
- **What:** switch the progress term to the existing omni function min(v·d, |cmd|)/|cmd|: capped at the command, 0 at
  zero command, no gate.
- **Why:** today's cap is an absolute 1.05 m/s, which never binds on 0.21–0.35 m/s commands, and retreating is free.
  Extreme Parkour, Lee 2020 and RMA all cap at the command.
- **Plan:**
  - The parity dose is settled offline now.
  - The promoted dose (about 0.6, which lifts side/back tracking) trains only after the clock decision and after #22,
    or progress is paid twice in 2b/2c.

**26. track-single-kernel** — merge · 7.2 GPU-h · low
- **What:** replace exp (22.5 %) + L1 (7.7 %) with one bounded kernel: Laplace, or a re-dosed Gaussian, optionally with a
  small linear tail.
- **Why:** it is the same error priced twice. One kernel is standard (legged_gym, Isaac Lab, Spot's Laplace).
- **Caution:** Revision 4 co-tuned the pair (marked PROVISIONAL). The real cost is a lower slope for escaping
  standstill.
- **Needs:** the clock decision, #25, M1b and #8.

**27. swing-height-rebase** — fix-then-decide (teacher) · ~10+ GPU-h · medium/low
- **What:** after the clock decision, either one terrain-relative, gated clearance kernel that absorbs
  `penalty_swing_min_clearance`, or drop the pair.
- **Why:**
  - The root-relative reference is about 10 cm noisy, larger than its 3–4 cm thresholds, and it pays a crouch.
  - Apex income collapsed inside 2b under a constant weight, and foot_clearance is the prime suspect.
- **Timing:** the offline steps can run now.

**28. mode-time-after-clock** — replace · 7.2–14.4 GPU-h · low
- **What:** only if the clock is removed. One state-based swing-timing term replaces foot_idle and excess_contact:
  Spot-style mode-time on the accumulated air/contact timers, which the float32 bug does not affect.
- **Alternative (arm A):** a timing-fixed air time.
- **Caution:** the literature supports an air-time floor more than mode-time.

### X — conditional or parked

**29. yaw-progress-phaseout**
- Only if #17 part B shows yaw_progress is needed during formation: zero it from the first move_turn stage, a
  performance-gated removal (Rudin 2022).
- About 10 GPU-h.

**30. continuous-heading-servo**
- Command side: drop the 0.133 rad yaw dead band (the Isaac Lab and legged_gym servos have none), re-gate yaw_progress
  on the slot, and arm STAND_HOLD.
- Runs after M1b; conflicts with M5 hysteresis.
- About 1.8 + 7–11 GPU-h.

**31. rod-power-after-clock**
- #18's contingency: rod positive work, used only if Q regresses once delta_torques is gone.
- CPU step 0 now; kill it if it ranks the defect in 2 of 7 categories or fewer.
- Energy terms have a bad record on this plant.

**32. caps-smoothness-loss**
- The alternative to #19: CAPS temporal smoothness in the PPO loss (Mysore 2021).
- It is a shared PPO change and comes after M1b.
- 5.4–12.6 GPU-h.

**33. clockless-tripod-prior**
- Only if tripod loss is observed after the clock goes: one dense pair-sync term, adapted from Spot's GaitReward, or a
  CaT-style "≈3 feet down" prior.
- The risk precedent is the tripod_schedule v6 result, which was gamed.

**34. own-tripod-discriminator**
- Parked: an AMP-style style term trained on a clean tripod source.
- Entry only if both the clockless head and #20 lose the tripod.
- X08:c6 fails the source gate (|μ| ≈ 8).
- A lot of machinery to replace 3 live terms.

**35. stock-reference-arm**
- A scale-matched Isaac Lab velocity reward as a reference arm.
- Stage 0 already shows the raw stock torque penalty would cost about 4× all task income.
- Only after #20.

**36. cat-style-constraints**
- CaT stochastic terminations for tibia and excess_contact (plus the teacher contact terms).
- Only if #20 and #14 both REJECT.

**37. soft-pitch-constraint**
- A CaT soft tilt band below 0.5 rad.
- Only if #13 REJECTs. Re-derive it on in-distribution falls: the motivating falls came from an eval defect.

**38. orientation-unmask**
- Only if #13 REJECTs and orientation stays: remove the Extreme Parkour tile mask.
- The mask zeroes the term on 20 % / 50 % of envs, and 2–9 cm blocks give it no omni rationale.

**Ordering, in short.**
1. W0 and W0b run now, CPU-only.
2. #10 is ratified and #6(B) lands at a campaign boundary; then the W1 forks.
3. #20 runs before the clock goes, or not at all.
4. Wave C waits for the clock session.
5. X candidates run only on their triggers.

**Alternatives:** adopt at most one of #16/#17A, #19/#32, #18/#31.

**Declined during review** (reasons in the detail file):
- slot-gating the exp kernels;
- halving the yaw kernel;
- a one-step exp + L1 + progress merge;
- multi-critic heads;
- re-deriving the crab_failure thresholds;
- a world-pose sparse task reward.

## 5. Per-term review (every name)

Columns: weight (phases) | X08:c6 share | literature standing | verdict → candidate.

- **track_lin_vel_xy_exp** — 1.25 all | 22.5 % | standard single kernel (narrow σ) | keep; later merge → #26, #3
- **penalty_tracking_error_l1** — −2.0 all | 7.7 % | a separate unbounded term is non-standard | keep until a single
  kernel passes → #26
- **reward_forward_progress_along_command** — 0.6 all | 1.8 % | standard only when capped at the command | replace → #25
- **penalty_lin_vel_y** (cross-command) — −3.0 all | 0.5 % | neutral here (projection terms present) | remove in 1a–1c
  → #12
- **track_ang_vel_z_exp** — 1.0 all | 24.0 % | standard (refs ~0.5× lin, continuous servo) | keep as the one yaw term →
  #17, #3, #30
- **reward_yaw_progress** — 1.0 all | 0.15 % (≈40 % of turn slope) | approximate potential-based shaping; Rudin removes
  bias terms | registered-margin decision → #17B, #29
- **reward_tracking_yaw** (facing exp) — 0.2 from 1c | 5.0 % | Extreme Parkour only; ref stacks use the rate kernel |
  remove or slot-gate → #17A / #16
- **reward_facing_turn** — 0 | 0 | no precedent | adopt at 0.2, else retire → #16 / #4
- **reward_clock_schedule** — 1.0 (2c ramp never run) | 10.3 % | periodic composition standard; product/dose unprecedented
  | clock session → #24, #33, #28, #34
- **reward_clock_swing_apex** — 1.0 → ε by 2b | 4.4 % | no removal precedent | retire after formation → #23
- **reward_feet_air_time_positive** — 0.8 → ε | 0.02 % | standard family (working) | remove (dead, revival hazard) → #2
- **reward_stride_length** — 0.5 → ε | 0.00 % | no precedent | remove → #2
- **penalty_foot_idle_when_forward** — −0.12 all | 7.9 % | non-standard | fix-then-decide → #15, #9
- **penalty_excess_feet_contact_forward** — −0.2 all | 0.6 % | CaT uses a contact-count constraint | keep for now (the
  only term that charges the creeper) → #9, #20, #28
- **penalty_tibia_deviation_in_stance** — −0.28 all | 1.4 % | standard forms are command-gated | remove → #14
- **reward_orientation** — −0.7 all (tile-masked) | 0.11 % | hexapods omit; quadruped flat configs keep | zero → #13,
  #38
- **reward_lin_vel_z** — −0.15 all | 0.05 % | quadrupeds −2.0; hexapods drop | zero → #13
- **reward_action_rate** — −0.3 all (unsquared, on clipped actions) | 11.4 % | standard is −0.01 squared | re-derive with
  M1b → #19 / #32
- **reward_action_rate squared variant** (knob `KRABBY_ACTION_RATE_SQ`) — off | Gate 1 FAIL 4/7 | — | folds into #19
- **reward_delta_torques** — −1e-6 all | 2.3 % | absent from legged_gym and Isaac Lab | remove → #18 (contingency #31)
- **penalty_mechanical_power** — 0 (L3 ran +0.001 by sign error) | 0 | energy terms standard | keep at 0, sign guard →
  #3, #31
- **Zero-weight names** (feet_slide, base_pitch_forward_linear, ang_vel_xy, dof_error, stance_support_feet,
  motor_direction_reversal, tripod_schedule, one_direction_spin, cam_contact_schedule, cam_phase_lock) — 0 | 0 | mixed;
  all superseded or null in the repo | unregister → #1
- **reward_collision** — −2.0 from 2a | ~0.7 % (teacher) | standard everywhere | keep; test leg-leg reading → #11
- **reward_feet_edge** — −0.3 from 2a | 0 over 10 k its | Extreme Parkour only | unregister → #1
- **reward_feet_stumble** — −1.0 from 2a | ~0.5 % (teacher) | kernel standard; double pricing is not | fix-then-decide →
  #11, #22
- **reward_foot_clearance** — 1.0 in 2b/2c | 2.2 % (2b) | terrain-relative, phase-gated forms are standard | rebase or
  retire → #27
- **penalty_swing_min_clearance** — −0.4 in 2b/2c | 0.5 % (2b) | no precedent as a separate term | merge or retire → #27
- **reward_obstacle_clearance** (omni) — 0.9 in 2b/2c | 4.2 % (2b) | no precedent (no lift factor) | remove → #22
- **reward_tracking_goal_vel** (omni, obstacle-masked) — 0.75 in 2b/2c | 3.5 % (2b) | a tile-masked duplicate is
  unprecedented | remove → #22, #25
- **Knob-gated functions**:
  - facing_coupled, turn_gated, stand_pose → retire (#4).
  - action_second_diff → merge (#19 / #32).
  - joint_vel_l2 → merge or retire (#31 / #32).
  - loaded_foot_slip → keep (motion ladder; the single slip definition).
- **Structural**:
  - zero clip → measure and declare (#8);
  - crab_failure → keep, and split its log (#11);
  - total_terminates branches → #5.
