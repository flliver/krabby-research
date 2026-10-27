# Omni reward simplification: candidate catalogue (long form)

*2026-09-23. Long-form companion to [CANDIDATES.md](CANDIDATES.md), which holds the condensed catalogue, the waves and the per-term verdicts. This is the output of the read-only research workflow `wf_15be1644-c4a` (92 agents). The independent final check's catalogue corrections are applied, and each is marked "(final check)". To start a candidate, say "Plan the `<name>` candidate from the omni reward simplification plan" (see README.md).*

**Baseline and conventions.**
- The incumbent head is X08:c6, the C1d turn-first curriculum, model_7996, which is a 1c recipe on the flat stages.
- The reward registers 37 names. There are also 6 knob-gated functions, the per-step zero clip, and the terminations.
- "Income" is Isaac Lab's Episode_Reward: the per-episode term sum divided by the maximum episode seconds, averaged over the last 10% of iterations. It is measured before the zero clip. "Share" is |term| / Σ|terms| on X08:c6 unless stated otherwise.
- Noise, single seed: R_seed on block F is about 3.2 (scratch), and the minimum detectable effect on turn facing_closed is about 0.15. Same-seed runs are not bit-reproducible.
- GPU cost is serial, at about 0.45 GPU-h per 1000 iterations.
- Every candidate below went through adversarial verification. None was killed. Each section is the revised form, and the last line of each section records what the verification changed.
- Scope of that verification (final check): the 29 merged candidates each had a repo-evidence lens and a literature/RL lens (priority ≤ 2) or one combined lens. The 6 gap candidates (term-discrimination-replay, ni-adoption-rule, removal-knob-batch, adopted-set-formation-check, stock-reference-arm, own-tripod-discriminator) had one combined-lens refutation. The split-offs (height-backstop-tag, soft-pitch-constraint, orientation-unmask) inherit their parent's review.
- Candidates that touch the gait clock are coordination points with the separate clock-deprecation session. They do not duplicate its work.

## Candidate index

Wave key:
- **W0**: now, no GPU.
- **W0b**: offline or CPU evidence.
- **W1**: fork screens that do not depend on the clock.
- **M1b**: planned with or after the M1b action-bound fine-tune.
- **T**: teacher stages; needs a 2a head to fork from.
- **C**: after the clock decision.
- **X**: conditional or parked.

| # | name | kind | terms | priority | depends on | rough cost | confidence |
|---|---|---|---|---|---|---|---|
| 1 | zero-weight-purge | cleanup | 10 always-zero names + reward_feet_edge | W0 | none (branch coordination) | 0 GPU-h, 1-2 h | high |
| 2 | dead-swing-income | remove | reward_feet_air_time_positive, reward_stride_length | W0 | none (ships with #1) | 0 GPU-h | high |
| 3 | survival-floor-audit | fix-then-decide | zero clip, implicit survival premium | W0b (Stage A); Stage B on trigger | none | A: 0 GPU-h; B: ≤7.2 GPU-h | medium |
| 4 | term-discrimination-replay | fix-then-decide (evidence) | 15 live terms | W0b | #3 | 0-0.2 GPU-h, 2-4 h | high |
| 5 | ni-adoption-rule | structural (protocol) | ladder decision rule | W0 | user ratification | 0 GPU-h (+3.6 for a seed-2 anchor) | medium |
| 6 | removal-knob-batch | cleanup (enabling) | knobs for foot_idle, tibia, lin_vel_z, lin_vel_y; driver checks | W0: (A) now, (B) at a campaign boundary | boundary timing | 0 GPU-h, ~0.5 day | high |
| 7 | config-truth-align | cleanup | sign guard, class literals, knob surface | W0 (sign guard + L3 note); (a) needs user decision; (b) after clock | user; clock session | 0 training GPU-h | medium |
| 8 | dormant-lever-retire | cleanup | facing-coupled, turn-gated, stand-pose levers; reward_facing_turn | W0 (Tier A once owning ladders close); Tier B after #17 | #17; ladder closure | 0 GPU-h, 2-4 h | medium |
| 9 | gait-gate-decouple | coordination refactor | _cmd_gate for 5 terms | W0, offered to clock session | clock session | 0 GPU-h | high |
| 10 | cross-penalty-remove | remove | penalty_lin_vel_y | W1 | #5, #6 | 0.77-1.6 GPU-h + evals | medium |
| 11 | posture-penalty-drop | remove (zero, keep registered) | reward_orientation, reward_lin_vel_z | W1 | #5, #6 | ~2-2.5 GPU-h; formation +3.6/seed | medium |
| 12 | foot-idle-audit | fix-then-decide | penalty_foot_idle_when_forward | W0b offline; W1 fork | #3; clock (for removal) | ~0 / 1.8 / 7.2 GPU-h | medium |
| 13 | rate-follows-bounds | fix-then-decide (M1b follow-on) | reward_action_rate, second-difference knob | M1b | M1b first result; #7 | 1.7-3.5 GPU-h | medium |
| 14 | delta-torque-drop | remove | reward_delta_torques | W1 (after M1b) | #5, #7 | 1.8 + 4-6 GPU-h | medium |
| 15 | contact-telemetry-audit | fix-then-decide | reward_feet_stumble, reward_collision, hip branch | W0b Step 0; T | #18 before any screen | 0-1 GPU-h; screen 5-10 | medium |
| 16 | one-yaw-kernel | merge (A: remove; B: dose cut) | reward_tracking_yaw, reward_yaw_progress | W1 (A); B after A | #3, #5 | ~9-13 GPU-h per part | medium |
| 17 | turn-slot-facing | replace (env overlay) | reward_tracking_yaw → reward_facing_turn | W1 | #5 | ~1.8 GPU-h | medium |
| 18 | ring-bonus-drop | remove | reward_obstacle_clearance, reward_tracking_goal_vel | T | 2a head to fork from; #5 | 7.5-10 GPU-h | medium |
| 19 | apex-term-retire | remove (after formation) | reward_clock_swing_apex | C (a step inside the clock work) | clock session | ~4.1 GPU-h | medium |
| 20 | progress-relative-cap | replace | reward_forward_progress_along_command | parity: W0b CPU; promoted dose: C | #18; clock | 0 / 3.6-7.2 GPU-h | medium |
| 21 | tibia-pose-retire | remove | penalty_tibia_deviation_in_stance | W1 | #5, #6 | ~1.8 GPU-h + evals | low-medium |
| 22 | minimal-reward-arm | structural (diagnostic) | 9 terms off | formation (before the clock goes) | #3, #5, #6; clock (hard) | ~12 GPU-h | medium |
| 23 | adopted-set-formation-check | structural | union of fork-adopted removals | after W1 | #5; ≥2 adoptions; clock anchor | ~7.6 (+3.9) GPU-h | medium |
| 24 | stock-reference-arm | replace (reference) | scale-matched Isaac Lab stock set | X | #22; stage-0 gate | 2-4 h; 3.9 GPU-h/seed | medium |
| 25 | clock-hold-twin | phase-out (handover) | reward_clock_schedule | C | clock session (hard) | 3.6-10.8 GPU-h | low-medium |
| 26 | swing-height-rebase | fix-then-decide | reward_foot_clearance, penalty_swing_min_clearance | W0b offline; C training | clock | ~10+ GPU-h | medium (defect) / low (lift) |
| 27 | track-single-kernel | merge | track_lin_vel_xy_exp + penalty_tracking_error_l1 | C | clock; #20; M1b; #3 | 7.2 GPU-h | low |
| 28 | yaw-progress-phaseout | phase-out (conditional) | reward_yaw_progress | X | #16 part B | ~10 GPU-h | low-medium |
| 29 | mode-time-after-clock | replace | foot_idle, excess_contact, air-time successor | C | clock removed | 7.2-14.4 GPU-h | low |
| 30 | rod-power-after-clock | contingency for #14 | power, delta_torques, joint_vel | W0b step 0; C | #14; M1b; #7 | 1 CPU-h; 2-8 GPU-h | low |
| 31 | caps-smoothness-loss | structural | action_rate, second-difference, joint_vel | after M1b | M1b; alternative to #13 | 5.4-12.6 GPU-h | low |
| 32 | clockless-tripod-prior | replace (conditional) | clock phasing role | X (on observed tripod loss) | clock; #9 | ~3.6 GPU-h | low |
| 33 | own-tripod-discriminator | fix-then-decide (parked) | gait family + new style term | X | clockless head and #22 both lose tripod | 1-2 days code; ≥7-8 GPU-h | low |
| 34 | continuous-heading-servo | simplify (command side) | yaw target, yaw gates | C (soft) | M1b; STAND_HOLD | 1.8 + 7-11 GPU-h | low-medium |
| 35 | cat-style-constraints | structural (conditional) | tibia, excess_contact (+ teacher contact terms) | X | #21/#22 REJECT; #3 | ~2-2.5 GPU-h + code | low |
| 36 | soft-pitch-constraint | addition (conditional) | new soft tilt band below 0.5 rad | X | #11 REJECT | ~2-2.5 GPU-h | low |
| 37 | orientation-unmask | fix-then-decide (conditional) | reward_orientation tile mask | X | #11 REJECT | replay + one teacher arm | low |
| 38 | height-backstop-tag | fix-then-decide | total_terminates tilt and height branches | W0b | #1 | ~0 GPU-h | high (tilt) / low (height) |

## Candidates

### zero-weight-purge — Unregister the omni reward's dead names: 10 always-zero terms and the never-paying feet_edge

**Kind & scope.**
- Kind: cleanup. It changes only CrabHexOmniFlatWalkRewardsCfg (crab_hex_omni_rewards.py:652) and the omni presets.
- Terms at weight 0 in every phase: feet_slide, penalty_base_pitch_forward_linear, penalty_cam_contact_schedule, penalty_motor_direction_reversal, reward_ang_vel_xy, reward_cam_phase_lock, reward_dof_error, reward_one_direction_spin, reward_stance_support_feet_when_forward, reward_tripod_schedule.
- reward_feet_edge: weight -0.3 in 2a-2c, but pays 0.
- The forward and sideways classes, mdp_pins.yaml and rewards.py are untouched.

**What changes.**
1. The 10 always-zero names are set to None. The IsaacLab RewardManager skips None terms (reward_manager.py:226).
2. feet_edge leaves 2a-2c, 2c_clockheld included, by adding KRABBY_EDGE_W to DROPPED_KEYS (crab_hex_omni_phases.py:142-149; _omni_env drops these keys at :251-256). This route is omni-only. Forward ELEMENTS_5K (crab_hex_phases.py:112) is untouched, so no forward/sideways sign-off is needed.
3. One fail-loud guard rejects, at cfg build and with a clear ValueError, any env var whose target term is None. It runs after DROPPED_KEYS filtering and before both the forward `_overrides` weight loop and the `_tripod_params` loop (parkour_mdp_cfg.py:1277-1323). Guarded knobs: KRABBY_TRIPOD_W plus the tripod band knobs, KRABBY_SPIN_REWARD_W, KRABBY_CAM_SCHED_W, KRABBY_PHASE_LOCK_W, KRABBY_REVERSAL_W, KRABBY_ANGVEL_W, KRABBY_EDGE_W and KRABBY_STANCE_SUPPORT_W.
4. The name pins are rewritten on purpose (tests/unit/test_crab_hex_omni_rewards.py:1185-1204; tests/integration/test_crab_hex_omni_configs.py:887-896). Their intent is kept: surviving forward names stay a subset of omni names, the removed set is pinned explicitly, and nothing is renamed.
5. The plan must say whether a dumped cfg shows removed names as null values or as absent keys. A configclass dict dump probably keeps the key with a null value. The honest count is the active manager term count, 37 → 26.

Out of scope:
- penalty_mechanical_power stays registered because motion M2 and rod-power-after-clock may use it.
- reward_facing_turn is handled by turn-slot-facing or dormant-lever-retire.
- The clock and apex belong to the clock session.
- The termination branches go to height-backstop-tag.
- The forward teacher reward class only appears on the omni path as a class-field default, which _phase_rewards_cfg replaces (crab_hex_omni_env_cfg.py:268-300). Changing it is optional typing cleanup whose only test is that the dumps do not change.

**Why — repo evidence.**
- All 10 names are 0.0 in every omni dump; every train-id entry has 37 names. They log exactly 0.0 in X08 t3/c4/c5/c6 and teacher 2a/2b.
- Removing them saves no compute: ParkourRewardManager skips weight-0 terms (parkour_reward_manager.py:24-35). The cost is surface area: log rows that look live, reset calls on four stateful terms, and the knobs.
- feet_edge at -0.3 is evaluated every step in 2a-2c (rewards.py:143-165), so removing it does save a little compute.

Per-name records:
- **tripod_schedule**: no gradient in three screens (2026-08-10_0058_tripod_stability RESULTS.md:67-74, 110-145; task1_velocity CHANGELOG.md:63-70; gait_formation CHANGELOG.md:98). Superseded by the clock. Not omni-aware: its ||cmd_xy|| gate cannot see turn slots, and its [0.30, 1.40] s period band excludes slow cadences.
- **spin and phase-lock**: null in era A (phased_flat CHANGELOG.md:64-74, 95-104); de-scoped 2026-08-18.
- **reversal**: lost to action_rate/delta_torques (short_runs RESULTS.md:74-76). Its telemetry is covered by an eval metric (gait_eval/metrics.py:912-995).
- **cam_contact_schedule**: screened only in the invalidated onedir campaign.
- **pitch_forward_linear**: gates on body-frame cmd_x > 0.12 and relu(pitch) (rewards.py:426-437). It misses back, side, stand and turn slots and about 2/3 of fall directions (1c canary pitch_end +0.48 in 36/100, -0.48 in 64/100). Era-A doses moved pitch by 0.000 rad.
- **ang_vel_xy**: null in era A (a7) and era B (A1/A2). C6 cut episode length 881→461, but that run was invalidated by the leg-collision fix.
- **dof_error**: 0 since 8c0c1f54 (2026-05-17); never screened.
- **stance_support**: its only test was era A, flat, and reverted.
- **feet_slide**: its 1 N max-over-history gate trips on brushing swing feet. Superseded by penalty_loaded_foot_slip (50 N gate).
- **feet_edge**: Episode_Reward 0.0000 in the first and last 10% of both 5000-it teacher windows. The terrain generator overrides the sub-terrain horizontal scale 0.05 with 0.08 (IsaacLab terrain_generator.py:128-130), so the riser threshold is 1.5 × 0.08 = 0.12 m and the dilation is int(0.05/0.08) = 0. The tallest block in the [0.2, 0.7] difficulty band is 0.072 m (0.09 m at d = 1.0). terrain_levels ≈ 6-7, so the >3 gate is not the cause. The world-to-pixel index also drifts up to 0.8 m (omni_exposure.py:250-254). Known inert since 2026-09-16 (omni_formation CHARTER.md:288-294).

Consumers:
- No omni file or omni experiments/tools script names any of the 11.
- The forward policy_summary.py maps KRABBY_TRIPOD_* to tripod params (:68-71). It is forward-only and harmless.
- parkour/scripts/rsl_rl/evaluation.py:156/188 hard-reads reward_feet_edge and its feet_at_edge buffer. It must be handled or exempted as non-omni.

**Why — literature.**
- legged_gym drops zero scales in _prepare_reward_function [1]. Isaac Lab skips None terms [3], which is the direct precedent for unregistering. (Final check: a further Isaac Lab H1-config precedent was dropped because its reference was not among the verified inputs.)
- LiBR keeps ang_vel_xy registered at 0.0 [15]. Registering at zero weight is a common alternative idiom, so this is neither support nor objection.
- Eureka's "discard" option applies to active components that stay near-constant (Sec. 3.3; Appendix A Prompt 2) [16]. It does not bear on zero-weight names.
- Extreme Parkour's feet_edge assumes 0.05 m pixels and tall boxes [12,13,14].

**Hypothesis / falsifier.**
- Hypothesis (config equivalence): every surviving reward and termination term keeps the same func, weight, params and order. Removed terms had weight 0, or (feet_edge) pay exactly 0 on the ring band. The per-step reward is therefore mathematically unchanged. Training identity cannot be tested because runs are not bit-reproducible.
- Falsified by any of:
  - a regenerated dump differs in a surviving term;
  - a consumer reads a removed name (run_candidates overlay checks, EVAL_CARRY, run_phases telemetry, evaluation.py);
  - an omni preset env still carries a purged knob after the overlay;
  - an omni 2a-2c log shows nonzero feet_edge;
  - a CPU edge-mask regeneration at the generator scale 0.08 finds edge pixels in the training band.

**How it would be decided.** No training.
1. Regenerate the dumps for 1a-2c, 2c_clockheld and the C1d and fork_c6 overlays. The existing dumps already contain the train ids, so there is no "PLAY-id" defect to fix.
2. Diff against dumps_motion_knobs_off. Only the removed names may differ, in the declared representation.
3. Forward and sideways dumps must be byte-identical (test_mdp_pins_match_dumps).
4. Unit tests: every omni preset raises when a purged knob is set, and the 2a-2c env has no KRABBY_EDGE_W.
5. The edge-mask check must use scale 0.08. With the raw 0.05, d = 1.0 hurdles would show pixels.
6. Optional: a smoke run of at most 100 iterations per preset.

**Dependencies & conflicts.**
- No scientific dependency.
- Land it on a branch: the clock session and the M1b planning edit the same cfg files.
- Ships with dead-swing-income, in the landing slot described in removal-knob-batch but as a separate commit from its landing (B).
- penalty_mechanical_power is deliberately excluded.

**Risks.**
- Loses one-knob sweepability of ideas that all have null records.
- Guard ordering: an override applied after a term is set to None becomes a silent no-op, and the tripod params loop dereferences the term too.
- Ends the documented "forward 35 + 2" omni contract.
- feet_edge would matter on risers of 0.12 m or more, but its pixel drift makes this implementation wrong there anyway.
- Dropping KRABBY_EDGE_W changes the 2a-2c preset env and the resume-chain env. The ladder recipe_id does not change, because it hashes only {chain, overlay_by_stage} plus fork checkpoints (run_candidates.py:1045-1058).

**Rough cost.** 0 GPU-h. Dump regeneration needs one Kit launch, run while the GPU is idle and under a memory cap. Optional smoke under 0.1 GPU-h. About 1-2 h of engineering.

**Adversarial review.** Two verifiers kept it at high confidence. Changes:
- The shared-ELEMENTS_5K route was replaced by DROPPED_KEYS.
- KRABBY_STANCE_SUPPORT_W and the tripod-params loop were added to the guard, and evaluation.py was added as a consumer.
- The false PLAY-id fix and the recipe-id churn risk were removed.
- The compute and Eureka claims were corrected.
- The termination branches were moved out to height-backstop-tag.

### dead-swing-income — Remove the two swing incomes that the float32 first-contact bug keeps at about 0

**Kind & scope.** Remove.
- reward_feet_air_time_positive: 0.8 in 1a-1c; ramped 0.8→0.4 in 2a and 0.4→0.001 in 2b; 0.001 in 2c.
- reward_stride_length: 0.5; ramped 0.5→0.25, then 0.25→0.001.
- Also their KRABBY_PHASEOUT rows and their GAIT_WEIGHTS values.
- Phases 1a-2c.

**What changes.**
- Both terms leave the omni reward. The apex rows in the same KRABBY_PHASEOUT strings stay, because the clock session owns them.
- Mechanism, if shipped with zero-weight-purge (which already rewrites the name pins): set both to None. Any leftover ramp row then fails loudly at env construction through ramp_reward_weight's get_term_cfg (curriculums.py:95-116).
- Mechanism, if shipped alone: keep both registered at weight 0 and add an omni-side guard that raises when KRABBY_AIRTIME_W, KRABBY_STRIDE_W or a KRABBY_PHASEOUT entry naming them is set. Without the guard, a leftover ramp would silently revive a weight-0 term at 0.4/0.25.
- In both cases, override both GAIT_WEIGHTS and GAIT_RAMPS inside _omni_env. Never mutate the forward dicts, which omni aliases (crab_hex_omni_phases.py:123-124).
- Consumers to update:
  - run_candidates.py:1482 (AIRTIME verifier) and :1574 (type table);
  - run_phases.py:76-77 (CANARY_KEEP);
  - omni_knobs.py:130,178;
  - motion_knobs.py:89-94,131;
  - the 2c_clockheld derivation (crab_hex_omni_phases.py:310) and the phase notes (:291);
  - the replay tools.
- Omni has no bundle manifest.
- The float32 bug is not fixed here.

**Why — repo evidence.**

Incomes (last 10%):

| run | airtime /s | stride /s |
|---|---|---|
| X08:c6 | +0.00056 (0.02%) | +0.000035 |
| t3, first 10% | +0.00050 | — |
| teacher 2a | 0.00031 | 1.9e-5 |
| teacher 2b | 7.5e-7 | 7.9e-8 |

Why air time pays 0:
- REPLAY_GATE.md:226-238 shows airtime paying identically 0 over 999 X08:c6 eval episodes. The replay reproduces the float32 sensor clock.
- compute_first_contact tests elapsed < dt + 1e-8 (contact_sensor.py:182, 215, 218, 250). After about 2 s the elapsed time reads 0.0200001-0.0200005 s.
- Past about 16 s, refresh skips (elapsed 0, then 0.025) keep the test failing through a 40-s episode (replay_motion_terms.py:332-365).

Stride:
- Stride uses compute_first_contact and compute_first_air (rewards.py:1390-1391).
- Its zero shows in the logs (3.3e-5 to 5.7e-5 /s over t3-c6) but has not been replay-verified.

Even with the bug fixed, both stay weak:
- Airtime would pay about a 1.2-1.6% share at 0.8 (exact-timing replay). It fails Gate 1 because it pays long swings (rho vs contact_switch_hz -0.28 to -0.40) and does not track the chatter defect.
- Its small share comes from sparsity, one event per swing with relu(air − 0.38), not from dt scaling.
- Stride would be about 1e-3/s at 0.2-0.35 m/s. It duplicates the along-command speed that progress already pays densely, and its gate excludes turns.

History:
- The documented reasons for both terms come from era-A or old-plant runs where they paid at most 0.002/s (parkour_mdp_cfg.py NOTEs at :966-983, :988-1010, :1135-1144).
- No NOTE records an exploit that removal would reopen. The stride "v2 snap" exploit belonged to a removed variant.

**Why — literature.**
- Standard practice keeps a working air-time term: Isaac Lab RewardsCfg has feet_air_time at 0.125 with threshold 0.5 and no stride term [4], and legged_gym has _reward_feet_air_time [1]. Removal therefore rests on repo evidence: the term is dead now, would be about 1.5% if fixed, and fails Gate 1. Convention supports only the stride removal.
- The upstream bug is fixed in IsaacLab PR #7574, merged 2026-09-07 [11]. Its reproducer missed 352/500 transitions. The new default abs_tol = 0.5·max(update_period, physics_dt) is resolved at call time.
- The repo calls compute_first_contact(env.step_dt) with no abs_tol (rewards.py:868, 1390-1391). An IsaacLab bump would therefore silently switch on +0.8 air time and stride at weights the incumbent never trained with. This hazard alone justifies acting.
- Rejected cheaper mitigation: pin abs_tol = 1e-8 at both call sites. That freezes dead code in place.
- Eureka (Sec. 3.3) [16] reads a near-constant component as unoptimisable "as written" and offers three remedies: rescale, rewrite or discard. Discard is chosen here because rewriting belongs to the clock-successor decision.
- WTW shapes step length through foot swing and a Raibert foothold [36].

**Hypothesis / falsifier.**
- Hypothesis: removal changes the per-step reward by at most about 0.0006/s (under 0.03% of |income|), and only in the first ~2 s of each episode. That includes from-scratch 1a, where short episodes make this window a larger fraction. No eval block or guard moves.
- Falsified if a float32-faithful replay shows either term paying more than 1% of income in some category or state (episode start, reset, RSI start).
- Also falsified if the first-10% from-scratch share per unit of actual episode time (Episode_Reward plus mean episode length) exceeds 1%.

**How it would be decided.**
- (a) CPU replay as confirmation only: extend the float32-faithful path to stride, with per-category income on the X08:c6 suite and the 1c creeper raws.
- (b) Acceptance gate: a dump diff in which only these weights and ramp rows differ, plus a unit test that the ramp or weight knobs naming these terms raise on an omni task.
- No dedicated fork: a single-seed all-TIE is uninformative at R_seed ≈ 3.2. At most, fold the change into the next scheduled from-scratch control arm and label it a no-regression note.

**Dependencies & conflicts.**
- Independent of M1b and of the clock. Ships with zero-weight-purge.
- Coordination point for the clock session: air time does not pay today, so the clock is the only foot-timing prior in the omni reward. If the clock goes, that session chooses the successor: either a float32-safe event air-time term (legged_gym / Isaac Lab style, with its own edge detection) or mode-time-after-clock. Neither depends on reviving these exact terms.

**Risks.**
- Silent revival under a weight-0 implementation (hence the guard).
- Aliasing of the forward GAIT dicts.
- Forward and sideways keep their dead copies and the IsaacLab-bump hazard; that is their decision.
- Could be misread as "air time was tried and dropped".
- If the clock session picks event air time, this is partly undone. That is cheap, because the successor is a new function anyway.

**Rough cost.** 0 GPU-h: a CPU replay and config work.

**Adversarial review.** Both verifiers confirmed the numbers and the float32 mechanism. Changes:
- The uninformative 0.9 GPU-h fork was deleted.
- An explicit mechanism choice was added, along with the missed consumers.
- The literature was corrected: canonical configs keep a working air-time term, so this removal rests on repo evidence.

### survival-floor-audit — Measure what the per-step zero clip forgives, declare it, and choose a survival floor only when a removal would change it

**Kind & scope.** Fix-then-decide, all omni phases.
- The ParkourRewardManager per-step clip (parkour_reward_manager.py:38).
- The implicit survival premium: there is no alive or termination term.
- crab_failure as the implicit price of a fall.
- The effective weight of every penalty.
- reward_clock_schedule, as a coordination point.

**What changes.**

*Stage A: no behaviour change.*
1. The primary measure is the free TB gap on the training distribution: Train/mean_reward − Σ Episode_Reward/* × max_episode_length_s.
   - Take the first and last 10% of every archived run: X08 t3-c6, X01/X05, 1a/1b/1c, omni teacher 2a/2b, and the phase-out and forward lineage runs.
   - Caveat on lag: Episode_Reward averages the envs that reset in that iteration, while the rewbuffer holds the last 100 completed post-clip episodes. The gap is exact only in steady state.
2. Second source: logging-only telemetry of the clipped-step fraction and the forgiven mass per command slot. It goes in the omni env extras, not in the shared manager.
3. Attribution: replay_reward_economy.py run post-clip, after adding tibia_deviation. Report the sensitivity to its clock-phase fit (argmax vs median over 36 offsets), or use the recorded clock phase. replay_motion_terms.py prices only part of the economy and is out of scope.
4. Output: a clip budget per phase and per slot. Every removal candidate reports its projected change to that budget, extending the phase-out's income-budget pattern.

*Stage B: conditional.*
- Trigger: a removal candidate replayed post-clip (clock deprecation, one-yaw-kernel) is projected to raise the forgiven mass materially, e.g. above 1.5× the X08:c6 late gap.
- Default outcome **b1**: keep the clip, add a config flag defaulting to True, and document it in the omni README as the survival floor of record.
- If Stage B runs, the arms are:
  - **b0**: plain sum (the Isaac Lab form). This is the only arm that tests whether the clip is load-bearing.
  - **b1**: keep the clip.
  - **b1'**: the legged_gym form, a clip plus an explicit termination penalty added after the clip.
  - **b2**: an alive constant. Flag the hexapod standing-minimum risk.
  - **b3**: r_pos·exp(c·r_neg). This is a re-weighting, not a floor test.

**Why — repo evidence.**

The clip:
- Line 38 has no flag. The only commit touching the file is 905b044c.
- It is known and has been designed around twice:
  - the tripod v2 support bonus was built as a reward dual because a penalty would be "silently clipped away" (crab_hex_tripod_reward.py:55-59);
  - the phase-out had a clip-floor budget table and watchdog (2026-08-31_1414_gait_income_phaseout CHANGELOG.md:21-22; run_phaseout.py:15, 369-371).
- The phase-out's "clip not in play" verdict (CHANGELOG:70) rested on a mean net of +1.668. A positive mean cannot rule out per-step clipping.
- _episode_sums accumulate pre-clip values. Every share, replay gate and income table in the repo is therefore pre-clip.

TB gap, reproduced read-only:

| run window | Σ Episode_Reward × 40 | Train/mean_reward |
|---|---|---|
| X08:t3, first 10% (mean length 686) | -27.0 | +6.4 |
| t3, last 10% | 38.5 | 52.0 |
| c6, first 10% | 32.1 | 38.3 |
| c6, last 10% | 43.7 | 52.0 |

The clip binds, and it forgives about 16% of the late return.

The implicit premium on X08:c6:
- Pre-clip income is +2.051/s positive and -0.958/s negative, net about +1.09/s.
- With γ = 0.99 at dt = 0.02 s (a horizon of about 2 s), a fall forfeits about 2 discounted units.
- 68% of positive income comes from the two ungated exp kernels. The motion-requiring terms (progress, yaw_progress) are 2.9%.
- crab_failure ends 36% of X08:c6 episodes, and 47% / 53% of teacher 2a / 2b episodes.
- The hand-off "suicide" NOTE (parkour_mdp_cfg.py:383-392) reasons pre-clip. With r ≥ 0 and no termination reward, early death cannot be strictly optimal. Confirm the env used ParkourRewardManager; every parkour env does (parkour_manager_based_rl_env.py:59).
- replay_reward_economy.md has the creeper out-earning the walker on fwd NET (-0.229 vs +0.853/s), yet X08:c6 walks (fwd ratio 0.748). The clip plausibly explains part of this.

Couplings:
- Clock plus apex pay 0.444/s: 14.8% of |income| and 21.7% of positive income.
- The yaw cluster estimates that removing the near-constant yaw income (about 0.7-0.9/s) would cut net income to about 0.2-0.4/s.

**Why — literature.**
- legged_gym sets only_positive_rewards=True ("avoids early termination problems"). compute_reward clips, then adds a termination reward that ships at 0 [1,2].
- Isaac Lab has no clip [3] and no alive or termination reward in its velocity task [4].
- Reda et al. [20], Survival Bonus: a bonus too small gives the falling-forward minimum; too large gives standing still. Their evidence is PyBullet Hopper/Walker. For a statically stable hexapod, standing is the more likely minimum, which argues against b2.
- Kim, Lee & Park [19]: R = r_pos·exp(0.2·r_neg). Cited for the equation only; the paper credits multi-critic training for its early-termination behaviour.
- CaT [17] assumes r ≥ 0.
- Andrychowicz et al. [21]: changes in return scale act through value fitting.

**Hypothesis / falsifier.**
- That the clip binds is already shown.
- Hypothesis: the forgiven mass concentrates in formation and in specific slots (falls, stand-to-move transitions, turn rows), and removal candidates change it enough to matter.
- Stage A result against it: no removal candidate moves the forgiven mass beyond the trigger. Then b1, and close.
- Stage B falsifier: b0 is TIE on every block and guard at two seeds, so the clip is not load-bearing.

**How it would be decided.**
- Stage A: 0 GPU-h for the TB gap. Telemetry piggybacks on the next scheduled run.
- Stage B, only on trigger: a from-scratch C1d chain of b0 (optionally also b1') against the incumbent, seed 3 then seed 2, on the per-category suite.
  - Decisive: survival guards in all 7 categories, move_turn ratio, cat_tripod_min.
  - T, F, F_mt and Q must not be WORSE.
  - Training log: crab_failure share, the dead_walker abort, track_ratio_mean ≥ 0.6, value loss and explained variance.
- A 2000-it fork probes only the late effect.

**Dependencies & conflicts.**
- Stage A goes before every removal candidate's replay and before minimal-reward-arm.
- Coordinate with the clock session so it reports its own post-clip budget delta.
- Independent of M1b.
- The free standstill income question (exp kernels on stand rows) is re-measured here before anyone gates kernels.

**Risks.**
- b0 reopens legged_gym's early-termination problem and the Reda minima.
- An alive constant pays for standing.
- b3 re-weights every penalty at once, which confounds the comparison.
- A return-scale shift moves the value-clip dynamics (value clip 0.2 absolute on V ≈ 2).
- Two seeds detect only large effects.
- Eval rollouts under-sample training states, which is why the TB gap is the primary measure.

**Rough cost.** Stage A: 0 GPU-h plus a small code task. Stage B: 0 by default, up to 7.2 GPU-h plus evals on trigger.

**Adversarial review.** The verifier confirmed the TB gap and that the clip binds. Changes:
- Noted the repo has designed around the clip twice.
- Fixed the Stage-B logic: only a plain-sum b0 arm tests whether the clip is load-bearing.
- Made Stage B conditional.
- Lowered confidence to medium.

### term-discrimination-replay — Measure each live term's share of the walker-vs-degenerate margin on current A15+B raws

**Kind & scope.** Fix-then-decide (evidence only), offline, covering all phases.
- Terms: foot_idle, excess_contact, tibia, orientation, lin_vel_z, action_rate, delta_torques, penalty_lin_vel_y, progress, collision, stumble, foot_clearance, swing_min_clearance, obstacle_clearance, tracking_goal_vel.
- Tracking, clock and yaw terms appear in the margins.

**What changes.** No reward change. The output is one offline table: for every term, its share of the NET walker-minus-degenerate margin per category, and whether dropping it flips or strengthens the ranking. Per-term sign labels are not used. action_rate, delta_torques and similar terms charge for motion by design, so a sign label would push the catalogue toward removing every regulariser.

Populations:
- Healthy: heads that actually move on each hold (check v_along): X08 t3/c4/c5/c6 per category, and the 2b head on ring.
- Degenerate: standing/creep (1c), bang-bang saturated (the M1 heads), falls, and stand-hold fidgeting.
- Bootstrap over episodes and heads.

Reporting and fidelity:
- Report every margin with and without clock income, so the table survives the clock work.
- Fix fidelity first. Either use population-matched rollouts, or (preferred) add a per-term reward-manager step dump to the --save-raw eval (about 0.2 GPU-h, queued serially behind the GPU campaign), so contact and pose terms are measured exactly.

Special cases:
- feet_edge: settle by reading the code (terrain_levels > 3 gate; the raws carry no edge mask), giving "inert by construction".
- collision, obstacle_clearance and goal_vel: first confirm the raw fields suffice, or use the dump.
- stride and air time: mark as "blocked on the float32 fix", not "INERT".

**Why — repo evidence.**
- 12 of the 18 active weights date from the May pre-cam or era-A position plants (git weight trace 3c29d32d, afbc0a46, 06697969, 3e0af07a). Teacher-term evidence is non-harm only (gated_lineage safety_pack/clearance_pack).
- The existing A15+B replay (replay_reward_economy.md) does not match the training population. Its fidelity against the log (@6997) is about 15% only on tracking and clock terms. Examples:

| term | replay | log | ratio |
|---|---|---|---|
| foot_idle | -0.119 | -0.060 | ~2× |
| excess_feet | -0.075 | -0.015 | ~5× |
| air_time | 0.013 | 0.001 | ~13× |

- The 1a "walker" barely walks off the fwd hold: lat v_along 0.027 vs 0.074 for the mid head. On back, the creeper's 0.069 is close to the walker's 0.086.
- The foot_idle 4× walker tax holds only on fwd (-1.747 vs -0.441) and lat (4.7×). It is about 1.1× on move_turn (-0.375 vs -0.330), and the diag/back creeper pays about 0.
- The fwd NET gap (walker 0.095 vs creeper 1.503) is mostly foot_idle, which is why the margin framing is the one that matters for decisions.
- excess_contact is the only regulariser that charges the creeper more (fwd -0.179 vs -0.006), but it carries the 5× fidelity error.
- The raw npz fields are: actions, applied_torque, body_pos_w, cmd_applied, foot_force_norm, foot_pos_w, joint_pos, leg_contact_force, terrain_z. No edge mask, terrain level or obstacle map.
- Raws exist for seed3_X08_c6 suite, reeval_X08-c6, 2b ring, 1c and reeval 2a.

**Why — literature.** None needed. This is the repo's standing offline replay-gate practice.

**Hypothesis / falsifier.**
- Hypothesis (falsifiable): at matched fidelity, foot_idle's walker tax exceeds a pre-set X% of the fwd/lat NET margin on X08:c6 vs 1c, and excess_contact's charge on the creeper survives exact measurement (correct sign and a ratio above 2× on at least 3 moving categories).
- Falsified if a term's fidelity is still worse than about 25% after the chosen fix, or if healthy and degenerate populations are too alike to rank (the rel-IQR noise seen in Gate 1).

**How it would be decided.** The offline table ranks terms for screening only. Any "remove" verdict still goes to a training screen, because exploration shaping during formation cannot be seen offline.

**Dependencies & conflicts.**
- Needs survival-floor-audit Stage A for post-clip margins.
- Should precede posture-penalty-drop, foot-idle-audit, cross-penalty-remove, delta-torque-drop, tibia-pose-retire, ring-bonus-drop and swing-height-rebase.

**Risks.**
- Replay evidence is counterfactual.
- A new eval must not perturb existing metrics: any env write after Kit startup can flip same-seed evals.

**Rough cost.** 0-0.2 GPU-h, plus 2-4 h of CPU-side work and the step-dump code.

**Adversarial review.** The verifier found that the replay's own fidelity misses by 2-13× on exactly the terms expected to flip, and that the off-fwd "walker" does not walk. Changes:
- Sign labels were replaced with margin shares.
- Multiple degenerate classes were added.
- The exact reward-manager dump was added as the preferred fidelity fix.

### ni-adoption-rule — Non-inferiority adoption rule for declared simplification candidates

**Kind & scope.** Structural (decision protocol) for the ladder driver: decision_rule, speculative_default, anchored drift, batched_s2.

**What changes.** A thin layer on the existing machinery. A candidate declared `kind: simplification` uses speculative_default: candidate, batched_s2 deferral, and a cumulative budget against one fixed anchor.

1. Any WORSE is REJECT, as now.
2. Margins are stated in metric units per protected member, e.g. the maximum acceptable loss in turn facing_closed or in tripod_min. They are fixed in advance and justified by what the task needs, with 1.5R only as an upper cap. Members are tested individually, not only block means.
3. Protected rows are declared by where the penalised behaviour would appear in the degenerate-npz replay, not by where the term pays income. A working penalty pays about 0 exactly where it suppresses the behaviour.
4. Cumulative budget: the current recipe, with every adopted removal, is compared against one fixed anchor on every block.
5. An all-TIE result in [-R, -0.5R) goes to the batched checkpoint, not to a terminal REJECT. This keeps the existing Occam path.
6. move_turn and diag survival are hard, one-sided guards at the batched checkpoint on pooled R. The recorded harms happened there: C2 move_turn survival 0.58→0.12, and C1e diag survival WORSE.
7. At most 3 adoptions may be deferred before a mandatory batched pooled checkpoint.

Scope limits:
- The motion ladder's frozen `adopt_requires: Q` rule is not amended.
- Fork removal arms go in a new ladder that reuses X0 D02/D04 under the same code_sha, with a stated rule for when a code landing forces re-anchoring.

**Why — repo evidence.**
- CHARTER decision 5: a clear ADOPT needs at least one BETTER, and all-TIE is UNCERTAIN, which buys a pair. decide() uses anchored drift z ≥ -R. _speculative_pick honours speculative_default (C1c precedent, facing CHARTER.md:351).
- Pooled R = max(R_eval/√2, R_seed): the seed term does not shrink with a pair.
- Single-seed R values:

| block | scratch | fork (D05) |
|---|---|---|
| F | 3.194 | 2.073 |
| F_mt | 4.118 | 6.049 |
| T | 1.131 | 0.517 |
| Q | — | 0.319 |
| tripod_min | 0.028 | — |
| survival | 0.051-0.23 | — |
| move_turn.ratio | — | 0.036 |

- X0 seed 3 vs seed 2 fork deltas: turn closed 0.610/0.508, move_turn closed 0.411/0.142, diag survival 0.86/0.63.
- The current drift check at -R already lets a removal lose about 0.14 of turn facing_closed (0.618 → ~0.47). A 1.5R margin would allow about 0.21, more than C2's gain. Hence margins in metric units.
- The forward phase-out used a matched-control non-inferiority rule successfully: unit clock ≥ 0.85×, failure ≤ +0.10, canary ≥ 0.85×.
- Motion candidates.yaml l.8 freezes its decision rule into D00s, and l.65 sets adopt_requires: Q.
- No X08 seed-2 run exists. heads.json seed2 holds only X02:1c and X07:1c, and D12 (C1d) was a user-overruled REJECT with no pair.

**Why — literature.**
- Lakens 2017 [63]: TOST equivalence testing; bounds set from available resources need an explicit justification.
- Everson-Stewart & Emerson 2010 [64]: bio-creep, and why the comparator must stay fixed.
- Patterson et al. 2024, secs 2.6, 4.4, 7 [65], and Colas et al. 2018 on power [66]: evidence from few seeds is weak. The rule does not certify non-inferiority at one seed.

**Hypothesis / falsifier.**
- Hypothesis: with metric-unit margins and a fixed-anchor budget, true-null removals are adopted without buying a pair per candidate, and total drift from X0/X08 stays bounded.
- The earlier 55-65% pass-rate figure is withdrawn: -0.49σ gives about 0.69, not 0.84.
- Falsified if retro-scoring D11 C3a (the only recorded all-TIE) and the X0 s2-vs-s3 and B0 s2-vs-s3 pairs, treated as synthetic null simplifications, shows the rule rejecting nulls or accepting a known regression.
- Report eval-only and seed-level pass rates separately. Seed-level type-I/II rates cannot be estimated from about 4 same-recipe seed pairs (B0, X0, X01/X02 1c, X05/X07 1c); say so.

**How it would be decided.** Offline:
- Driver unit tests.
- The retro-scoring above. D08-D13 and M1 were all WORSE, so they test nothing.
- A null calibration with the eval-only part reported separately.
- A check that X0 s3/s2 give a finite R for every protected member.
- Then user ratification as a charter amendment.

**Dependencies & conflicts.** Needs only ratification. Every removal screen depends on it. Scratch arms first need an X08 C1d seed-2 run (~3.6 GPU-h).

**Risks.**
- The single-seed point threshold is still noisy.
- Deferred pairs could build on a lucky seed. This is bounded by the batched checkpoint plus a bisect.
- Code landings re-anchor the baseline, which is why the re-anchoring rule is needed.

**Rough cost.** 0 GPU-h and about a day of driver and test work. Add 3.6 GPU-h for the scratch seed-2 anchor when it is first needed.

**Adversarial review.** The verifier showed the proposed retro test could not fail and the null calibration used eval-only noise. Changes:
- The power arithmetic was corrected.
- Margins moved to metric units.
- The missing seed-2 anchor and the frozen motion-ladder rule were named.

### removal-knob-batch — Weight knobs for every live term that lacks one, landed at a campaign boundary

**Kind & scope.** Cleanup (enabling).
- New knobs: penalty_foot_idle_when_forward, penalty_tibia_deviation_in_stance, reward_lin_vel_z, penalty_lin_vel_y.
- Driver checks for knobs that already exist: KRABBY_ORIENT_W, DELTA_TORQUE_W, STRIDE_W, EXCESS_CONTACT_W, TRACK_L1_W, APEX_W/CLOCK_W (with the clock session), COLLISION_W, STUMBLE_W, EDGE_W, CLEARANCE_W, FOOT_CLEAR_W, SWING_MIN_CLEAR_W, GOAL_VEL_W.

**What changes.** Two landings.

- **(A) Driver only.** Add OVERLAY_DUMP_CHECKS entries (_weight_ok) and parsers in run_candidates.py for knobs already present in parkour_mdp_cfg's `_overrides` table (e.g. KRABBY_EXCESS_CONTACT_W already maps at :1302). run_candidates.py is outside CODE_PATHSPECS (:145-152), so code_sha does not move.
- **(B) Cfg landing.** Presence-based overrides for the four terms that have no knob. Their names are inherited, so a forward `_overrides` entry reaches the omni twin. Prefer one generic mechanism (e.g. KRABBY_REWARD_W__<term>, with an auto-derived dump check) so later arms never need another code landing. Leave out the tracking-kernel weights unless a rebalance candidate needs them.

Rules:
- Do not put zero-weight-purge or dormant-lever-retire in the same commit.
- Screens use ε = 1e-3, which keeps telemetry (forward phase-out precedent).
- A bake at 0 should be a preset env value where possible. Each bake is itself a code_sha move unless it is batched at a boundary.

**Why — repo evidence.**
- _check_knob (:2160) refuses any overlay key that has no whitelist entry or no dump check. OVERLAY_DUMP_CHECKS (:1451-1500) covers only YAW, YAW_PROGRESS, OMNI_FACING_TURN, OMNI_PROGRESS, POWER, ACTION_RATE and AIRTIME.
- There is no knob for lin_vel_z (parkour_mdp_cfg.py:946-953), penalty_lin_vel_y (crab_hex_omni_rewards.py:675-679) or tibia (:1011-1027).
- CODE_PATHSPECS covers omni mdp/config, forward config/mdp, parkour/scripts and parkour_isaaclab. Editing them in a continuing ladder:
  - invalidates recorded screens;
  - makes the launch refuse without --accept-code-change;
  - queues a B<k> rerun;
  - re-runs the base twins of uncertain candidates (run_candidates.py:10377-10389; anchors resolve under the current code_sha, :7263-7275).
- prove_code_alias aliases only pathspec narrowing.
- The base class is shared, so all 16 preset dumps are affected (dumps_motion_knobs_off + SHA256SUMS).

**Why — literature.** None needed.

**Hypothesis / falsifier.** At default env values, all 16 effective dumps are byte-identical before and after (B), and (A) touches no dump. Falsified if any dump differs at defaults, or if a knob at 0.0 or 1e-3 does not reach the weight.

**How it would be decided.**
- (A): a validator `--dry-run` with a multi-key overlay.
- (B): regenerate all 16 dumps and check byte equality against SHA256SUMS; one unit test per knob.
- No GPU re-anchor to "prove" a no-op: under the charter's R it cannot falsify anything, and runs are not bit-reproducible.
- Timing rule: land (B) at a campaign boundary. Then the next removal campaign's B0/X0 is its normal control cost.
- If a live ladder must continue across the landing, either count its full re-screen cost (every B<k>; scratch anchors about 3.6 GPU-h per seed) or ask the user to approve a dump-equivalence alias alongside prove_code_alias.

**Dependencies & conflicts.**
- Set one boundary date, coordinated with the clock-deprecation and M1b landings. If minimal-reward-arm is to run, land (B) at a boundary *before* the clock landing (final check: this resolves a dependency loop).
- Must land before the fork arms of cross-penalty-remove, posture-penalty-drop, tibia-pose-retire and minimal-reward-arm, and before the foot-idle-audit knobs.
- Lever retirement waits until the ladders that name those keys close.

**Risks.** Landing in the middle of a ladder invalidates its screens. Knobs set to 0.0 silence telemetry.

**Rough cost.** (A) about 0.5 h, 0 GPU. (B) about 0.5 day, 0 GPU when landed at a boundary.

**Adversarial review.** The verifier found:
- excess_contact already has a knob;
- driver-only edits cost no re-anchor;
- the cost of landing mid-ladder was undercounted;
- bundling the purge contradicts the identical-dump claim.

Split into (A)/(B), and the GPU re-anchor was replaced with a timing rule.

### config-truth-align — Guard the penalty-knob signs first; make the omni cfg say what trains only by user choice

**Kind & scope.** Cleanup across all omni phases; no effective reward change.
- Class literals: track std, ang std, L1 weight, tracking_yaw weight.
- Clock, apex and combine literals, with clock-session consent.
- The KRABBY_* override surface.
- The sign of KRABBY_POWER_W, ACTION_RATE_W and DELTA_TORQUE_W.

**What changes.** Independent parts.

- **(c-omni), first: an omni-side sign guard.**
  - run_candidates.py:1574 parses POWER_W, ACTION_RATE_W and DELTA_TORQUE_W with motion_knobs.parse_penalty_w, which refuses values above 0 (motion_knobs.py:196-208).
  - apply_omni_knobs raises if the penalty_mechanical_power, reward_action_rate or reward_delta_torques weight is above 0 after the forward __post_init__.
  - Optional separate sub-candidate: guard the shared forward table (parkour_mdp_cfg.py:1283, 1300-1301). Its test must include forward and sideways dump diffs, and it makes the historical L3 recipe unrunnable.
- **(d) A note in the gait_formation_v2 record.** Wave L3 ran penalty_mechanical_power at +0.001. The function returns +Σ|τ·q̇| (rewards.py:1132-1143), and L3 logged +0.4988 → +0.3937/s (L3_full_train.log). Only the L3 arm (tripod 0.604, completion 0.64) is confounded; L1 and L2 were correctly signed.
- **(a) User decision.**
  - (a1, recommended): keep the knob-timing pillar (crab_hex_omni_env_cfg.py:25-38; rewards docstring :652-667). Make an omni env refuse to build, or warn loudly, when KRABBY_OMNI_PHASE is unset outside a test allow-list. This closes the no-preset trap for all ~52 knobs at once.
  - (a2): align the class literals with what trains:
    - track std 0.2 (class body 0.141);
    - L1 -2.0 (class body -0.5);
    - ang std 0.5, by dropping the omni re-declaration and the OMNI_ANG_VEL_STD = 0.15 constant (crab_hex_omni_rewards.py:150-154, 670-674) so the term inherits the forward sqrt(0.25) (parkour_mdp_cfg.py:831-835, same stock func imported at :127);
    - update the references (omni_knobs.py:188; docstrings :38-46, 87, 150-154);
    - the tracking_yaw class body stays 0.0: 1a/1b train at 0, and 1c's +0.2 is a real preset delta.
- **(b) Collapse KRABBY_CLOCK_COMBINE and KRABBY_OMNI_CLOCK_COMBINE, and the clock literals.** Only after clock removal, sequenced by the clock session. Once reward_clock_schedule is gone, omni_knobs.py:1016-1022 raises if no term takes "combine".

**Why — repo evidence.**
- 7 values disagree between the class body and every shipped run, uniformly across all six phases:

| value | class body | shipped |
|---|---|---|
| track std | 0.141 | 0.2 |
| ang std | 0.15 | 0.5 |
| L1 weight | -0.5 | -2.0 |
| clock weight | 0 | 1 |
| apex weight | 0 | 1 |
| combine | sum | product |

  (That is 6 rows; tracking_yaw differs by phase, 0 in 1a/1b and 0.2 in 1c-2c, so it is not a uniform disagreement.)
- About 52 KRABBY_* variables shape reward or terminations. The run_phases _subenv scrub has already voided the 1c canary through this surface (ANALYSIS_1c_failure §2.1). README.md:257-266 documents levers that were silently dropped.
- An env built without a preset silently trains at ang std 0.15 (crab_hex_omni_phases.py:221). This is a training-only hazard: a reward std does not change play or eval behaviour. Every training path goes through run_phases/run_candidates presets.
- motion_knobs REUSED_WEIGHT_KNOBS routes POWER_W and ACTION_RATE_W through the unguarded forward table. A repo-wide grep finds no positive use of these knobs except L3.

**Why — literature.**
- Isaac Lab track_ang_vel_z_exp std sqrt(0.25) [4] and legged_gym tracking_sigma 0.25 [2]: the ang std being kept is the reference value.
- The omni lin std 0.2 is a local tune; Isaac Lab uses sqrt(0.25).
- Booth et al. [60] is general caution only: 68% of 30 experts overfit the reward to their chosen algorithm, in a gridworld.
- The sign guard is a safety fix and needs no literature.

**Hypothesis / falsifier.**
- Hypothesis: behaviour-neutral for every shipped preset and overlay; closes the no-preset trap (under a1) and the mis-signed penalty path.
- Falsified by any rewards or terminations (func, weight, params) difference in the regenerated dumps of the six phases or of the C1d and fork_c6 overlays (plus forward and sideways if the forward table is touched), or by a consumer that relies on a positive guarded value.
- Recipe_ids cannot fail this test: they hash only {chain, overlay_by_stage} plus fork checkpoints (run_candidates.py:1052-1058), so they are not a behaviour test. (Correction to the table above: six values disagree uniformly across phases; the seventh, tracking_yaw, differs by phase, 0 in 1a/1b and 0.2 in 1c-2c.)

**How it would be decided.** No training. Before/after dump diffs. Unit tests for the sign guard (refuses values above 0, accepts 0 and negatives) and, under a1, for the unset-phase guard.

**Dependencies & conflicts.**
- c-omni and d land first, before any M2, rod-power or action-rate screen.
- Part a needs the user's decision on the knob-timing pillar.
- Part b waits for clock removal.
- The base class is shared, so every change lives in the omni subclass, except the optional forward-table guard.
- c-omni can ride with removal-knob-batch (B), because its expected dump diff is empty.

**Risks.** a2 reverses a documented design pillar. Historical recipes need a pinned commit to re-run. Ad-hoc runs without a preset change behaviour (this is intended).

**Rough cost.** 0 training GPU-h. Dump regeneration needs Kit launches while the GPU is idle, under a memory cap (DerivedDataCache hazard). About 2-3 h of code.

**Adversarial review.** The verifier:
- corrected the value count and removed tracking_yaw from the uniform list;
- dropped the analogy to the obs-clip eval defect and the recipe-id test, which cannot fail;
- showed that the sign guard as written would also change forward and sideways;
- added the cheaper a1 option that keeps the pillar.

### dormant-lever-retire — Retire the never-trained knob levers (Tier A); retire reward_facing_turn only if turn-slot-facing does not adopt it (Tier B)

**Kind & scope.** Cleanup. No preset arms any of these.
- **Tier A** (no change to behaviour or to the reward table):
  - track_lin_vel_xy_exp_facing_coupled (KRABBY_OMNI_TRACK_FACING_SIGMA);
  - track_ang_vel_z_exp_turn_gated (KRABBY_OMNI_ANG_TURN_ONLY);
  - penalty_stand_pose_l1 (KRABBY_STAND_POSE_W).
- **Tier B**: reward_facing_turn, the 37th registered name at weight 0, together with KRABBY_OMNI_FACING_TURN_W.

**What changes.**
- Tier A removes:
  - the functions (crab_hex_omni_rewards.py:495-557, 561-620; crab_hex_omni_motion_rewards.py:73-84);
  - their parsers and appliers (omni_knobs.py:259-269, 456-465, 840-910; motion_knobs.py:318-321, 342-354);
  - the run_candidates entries (OVERLAY_DUMP_CHECKS :1470-1487, parser map :1568, facing_coupled needle :1382);
  - replay_reward_economy's --coupled-sigma option (:617);
  - the tests in test_crab_hex_omni_{rewards,phases,candidates,gate,replay_motion,motion_knobs}.py and test_crab_hex_omni_configs.py.
- Fixture policy:
  - The frozen snapshots candidates_v1_2026-09-19.yaml (:406, :428, :490) and candidates_v2_2026-09-22.yaml (:528, :550, :613) arm C4, C6 and C11.
  - validate_candidates rejects unknown knobs (:2156-2159).
  - Either (a) add a RETIRED_KNOBS set that is accepted in historical or non-strict mode and refused in strict mode, or (b) validate the fixtures against a pinned commit. Never edit the snapshots.
- Keep the replay-side stand_pose TermSpec (replay_motion_terms.py:50, 320, 474, 581, 665) so that REPLAY_GATE.md stays reproducible, or record the pinned commit in REPLAY_GATE.md.
- Tier B takes the registered name count from 37 to 36:
  - the dumps change;
  - the 37-count asserts become 36;
  - Episode_Reward/reward_facing_turn disappears from future logs.

**Why — repo evidence.**
- **Facing-coupled lever.**
  - Added 2026-09-18. C6 was only smoke-tested (PLAN_AMENDMENTS.md:119) and was removed unrun on 2026-09-22 (item 38). It was never priced in a replay.
  - At sigma 1.0, with facing targets up to 2.4 rad, early move_turn translation income is scaled by exp(-5.76), about 0.3% of its maximum.
  - It couples only one of four additive velocity terms.
  - It adds facing pressure on infeasible move_turn rows, the pattern that cost survival in D10.
- **Turn-gated lever.**
  - Never trained: only two 20-iteration smokes (2026-09-18_08-16-22, 08-17-51).
  - Its cmd_wz ≠ 0 gate works against settling by construction: a dawdler earns 0.698/s, a body parked just outside the dead band about 0.85/s, and a finished turn 0.
  - The eval's settle tolerance of 0.15 rad (metrics.py:701) would barely see this.
  - C11 was removed unrun.
- **Stand pose.**
  - Failed Gate 1 on one policy's 99 stand episodes (rho -0.61 vs contact_switch_hz, -0.12 vs jitter). Gate 2 passed, and the rel-IQR caveat applies (REPLAY_GATE.md:83, 161, 242).
  - Dropped from M6 (candidates.yaml:22, 291-297).
  - default_joint_pos has not been verified as an A15+B stand pose.
- **facing_turn.** Logs 0 in every run. C4 (weight 0.8) was removed unrun. Its gate, slot_id ≥ SLOT_TURN, includes move_turn.

**Why — literature.**
- Published multiplicative composition couples the whole task reward (WTW Sec. 3.1 [36]; dm_control quadruped Move [53]).
- A heading-coupled translation gate, or an angular kernel gated on the command, was not found in legged_gym or Isaac Lab [1,4,5].
- Stand-still pose terms are off by default and use yaw-blind command gates (legged_gym _reward_stand_still [1]; Isaac Lab stand_still_joint_deviation_l1 [5]).
- If stand pose ever becomes a target, CaT's tolerance band [17] is the better-behaved form.

**Hypothesis / falsifier.**
- Hypothesis: Tier A changes neither trained behaviour nor the reward table. Tier B removes one zero-weight name, which changes log keys and dumps but no gradient.
- Falsifiers:
  - the facing ladder commits to screening facing-coupled within the horizon (then park it instead);
  - the motion ladder plans to arm stand pose;
  - a knob is armed in any omni candidates.yaml, preset, knob manifest, tests/unit/fixtures/*.yaml, or a live ladder's state or ledger;
  - the knobs-off dumps change under Tier A.

**How it would be decided.**
- Owner sign-offs from the facing ladder, the motion ladder, and the turn-slot-facing outcome.
- Tier A: the knobs-off dumps and the X08:c6 env.yaml reward table stay byte-identical, and the fixture-driven tests pass against unedited fixtures.
- Tier B: the dumps differ only by the removed entry, and the income and replay tools tolerate the missing key when comparing old and new runs.

**Dependencies & conflicts.**
- Tier B waits on turn-slot-facing.
- Stand pose waits on the motion ladder, the clock session and M1b, which own the stand regime.
- Lever removal waits until the ladders that name these keys close (see removal-knob-batch).

**Risks.**
- Loses ready levers. A slot-gated rebuild would be about 60 lines plus tests.
- The fixture/validator coupling.
- Loss of REPLAY_GATE reproducibility.
- The 37-name invariant cited in PLAN_AMENDMENTS.md:119.
- run_candidates and the replay references must be removed together, or the ladder tooling breaks.

**Rough cost.** 0 GPU-h; 2-4 h.

**Adversarial review.** The verifier found that the frozen fixtures arm three of these levers and that facing_turn is a registered name. The candidate was split into a table-neutral Tier A and a table-changing Tier B, with a fixture policy.

### gait-gate-decouple — Replace the clock-cadence gate on the five non-clock gated terms with a command-owned predicate

**Kind & scope.** Coordination refactor, all omni phases.
- Terms: _cmd_gate and _clock_cmd_speed (r_turn 1.2) as used by penalty_foot_idle_when_forward, penalty_excess_feet_contact_forward, reward_stance_support_feet_when_forward, reward_foot_clearance and penalty_swing_min_clearance (rewards.py:570, 622, 898, 946, 986).
- The clearance pair is affected only from 2b.

**What changes.**
- **Variant A (a no-op).** The omni _cmd_gate becomes (||cmd_xy|| > 0) OR (cmd_wz ≠ 0). This is the OR of the existing travel gate and yaw_progress's turn gate; r_turn leaves the non-clock sites.
  - The clock keeps _clock_running, the cadence, omni_clock_turn_radius and the CLOCK_TURN_MIN_SPEED rule; those belong to the clock session.
  - Forward and sideways branches are unchanged.
  - Fix the stale comments (rewards.py:569, 621) and the duplicated logic in replay_reward_economy.py:309-313 and replay_motion_terms.py:421.
  - Scope: the xy-only gates on air time (:872, > 0.1) and stride (stride_reward.py:87-89) leave turn-only slots ungated. That is moot if dead-swing-income lands; otherwise it is a separate, behaviour-changing follow-on.
- **Variant B (a behaviour change; needs a replay and a ladder screen).** Gate on the command slot itself (slot_id ≠ SLOT_STAND via stand_mask, command.py:646). This closes the stand-slot facing-drift leak without STAND_HOLD, and a turn slot stays gated after its facing settles.

**Why — repo evidence.**
- Every omni gated site uses min_forward_speed_cmd 0.12 (cfg_omni_2b.json; relay_hyst None, cmd_pool None).
- Why the two gates agree today:
  - the P-controller gives wz ∈ {0} ∪ (0.2, 0.3] (command_math.py:253-258);
  - moving slots have ||cmd_xy|| ≥ 0.21 (command.py:227-237, 583-586), and stand is exactly 0;
  - so _cmd_gate(0.12) already means "moving OR relay engaged" for any r_turn > 0.6, and r_turn is inert for these gates.
- What rides on these gates: foot_idle (-0.236/s, 7.9%) and excess_contact (-0.017/s), plus the clearance pair from 2b.
- The knob rule forbids r_turn = 0 (omni_knobs.py:943-977). The clock session therefore cannot try "turning does not drive the clock" without moving these gates too.
- Revision 2 (omni formation CHARTER.md:322-327): xy-only gating left turn slots ungated and the policy stepped in place. The turn-inclusive meaning is kept.
- RELAY_HYST (off by default; motion M5): the current gate closes at |err| < 0.067 rad and the clock stops at 0.111 rad, while the latch releases at 0.05 rad. Under hysteresis wz has no dead band, so wz ≠ 0 tracks the latch exactly (command_math.py:275-315).
- The CMD_POOL ang_scale path (omni_knobs.py:219-229; command.py:488-490, 697-705) is unused by every ladder and motion pool string. Where s ≤ 1/3, today's gate never opens on that turn row, so variant A would fix it.
- Stand behaviour is preserved, not fixed: stand.wz_active_frac is 0.114 at X0 (motion REPORT.md:204). M5 STAND_HOLD would close that leak.

**Why — literature.**
- Isaac Lab feet_air_time gates on norm(command[:, :2]) > 0.1, which is exactly the turn-in-place gap [5].
- Spot air_time_reward and GaitReward gate on the full command norm > 0 OR body speed above a threshold [6]. That is a precedent for variant A's form. Spot's command is sampled directly, with no relay, so the precedent says nothing about the stand leak.
- The r_turn-scaled max has no precedent.

**Hypothesis / falsifier.**
- Hypothesis: variant A equals the current gate on every step of the shipped command path while RELAY_HYST is off.
- Falsified if this invariant fails for any shipped knob combination:
  - wz ∈ {0} ∪ (clip·s, max·s] with r_turn·clip·s > 0.12, and
  - ||xy|| ∈ {0} ∪ [walk_floor, cap] with walk_floor > 0.12.
- RSI speeds are clamped into [floor, cap] (command.py:590-596), and eval holds use the same _write_command (:870-931).

**How it would be decided.**
- A unit test pinning the invariant across ANG_VEL_CLIP, ANG_VEL_MAX, CLOCK_TURN_R, pool ang_scale and walk floor, plus an assertion at construction time.
- The RELAY_HYST difference is recorded as intended.
- No smoke run: runs are not bit-reproducible, so a smoke cannot test a no-op.
- Variant B needs a replay and a ladder screen.

**Dependencies & conflicts.**
- Offer it to the clock session first as a ready-made seam; close it if they absorb it.
- Coordinate with M5.
- clockless-tripod-prior, mode-time-after-clock and swing-height-rebase all use it.

**Risks.** Stand-slot behaviour is preserved, not fixed. Under RELAY_HYST, or a future small ang_scale, the gate becomes more permissive; this is intended, and it is a fix for s ≤ 1/3.

**Rough cost.** 0 GPU-h.

**Adversarial review.** The verifier:
- confirmed the equivalence from code;
- replaced the sampled sweep and the smoke run with an analytic invariant test;
- narrowed the scope to the five callers;
- added variant B;
- reclassified the candidate as a seam for the clock session.

### cross-penalty-remove — Remove the omni cross-command velocity penalty in 1a-1c; decide 2a-2c after a teacher check

**Kind & scope.** Remove. Term: penalty_lin_vel_y, implemented by the omni function penalty_cross_command_vel_l2 at weight -3.0 in every phase (crab_hex_omni_rewards.py:675-679). Adoption is scoped to 1a-1c.

**What changes.**
- A new knob (e.g. KRABBY_OMNI_CROSS_VEL_W, default -3.0, byte-identical when unset), added through removal-knob-batch.
- On adoption, weight 0 in 1a-1c.
- 2a-2c stay at -3.0 until a teacher check passes. Unregister the term only after both are decided.
- Forward and sideways keep their penalty_lin_vel_y_l2.

**Why — repo evidence.**
- **Income.** -0.014/s (0.5%) at X08:c6; -0.014 to -0.017 per stage; teacher 2a -0.015, 2b -0.017.
- **Slope-share heuristic.** This compares reward slopes in velocity space; it is not the PPO gradient. Setup: std 0.2, weights 1.25 / -2.0 / -3.0, command 0.25.

  | (along, cross) error, m/s | exp slope | L1 slope | cross-L2 slope | cross-L2 share |
  |---|---|---|---|---|
  | (0.03, 0.03) | 1.79 | 1.41 | 0.18 | 5% |
  | (0.08, 0.05) | 2.50 | 1.06 | 0.30 | 8% |
  | (0.12, 0.08) | 2.97 | 1.11 | 0.48 | 11% |
  | (0.1, 0.3), tail | ~1.54 | ~1.90 | 1.80 | ~34% |

  The tail is where drift episodes, move_turn transients and obstacle deflections sit.
- The walker-vs-creeper effect is negligible: about +3.6 pp on the lat_l COMMAND margin, against a -25.3% gap.
- **History.** Added 2026-05-19 (afbc0a46) with position actions, on the old plant, forward-only, before L1 existed. Its rationale, that "larger [progress] values ... reintroduce north/south drift", is a coupling to the progress term.
- **Current metrics** (motion ledger D02, X0 control):
  - cross_track_ratio: fwd 0.046, side 0.036, back 0.056, diag 0.064, move_turn 0.095;
  - direction_error: 0.067 / 0.091 / 0.084 / 0.123 / 0.178 rad.
- **Noise** (resolution.74f93b91dc8b.json):
  - R_eval for cross_track_ratio: diag 0.0159, move_turn 0.0160;
  - R_eval for direction_error: diag 0.0344, move_turn 0.0383;
  - R_seed is null for all of these.

**Why — literature.**
- Full-vector tracking prices the cross error inside the tracking term, with no separate penalty: legged_gym [2], Isaac Lab (only track_lin_vel_xy_exp for linear xy) [4], CaT [17], Kim et al. Table I [18], and the hexapods HeLoM [31] and Liu 2025 [30].
- Orthogonal penalties appear alongside projection-only task terms: Lee 2020 S4 [25], RMA [27], and Qu 2024 Table I (v_y² at -1 with a clipped v_x task term) [28].
- The omni reward has projection-only terms of its own: progress 0.6 in every phase, and the obstacle projection tracking 0.75 in 2b/2c. It therefore sits in the "penalty present" regime, and the literature is at best neutral here.
- The case is strongest if progress-relative-cap, or progress removal, is decided first or at the same time.

**Hypothesis / falsifier.**
- Hypothesis: block T and all survival guards TIE, and tail cross-track does not worsen.
- Falsified if any of:
  - cross_track_ratio on diag or move_turn rises by +0.025 absolute or more (at least 1.5 × R_eval);
  - direction_error rises by +0.05 rad or more;
  - a tail metric (p90 cross-track or cross-speed per category, WATCH) worsens beyond an R_seed estimated from X01 vs X02 before the margins are declared;
  - T or any guard is WORSE.

**How it would be decided.**
- Prerequisite: ni-adoption-rule is registered. Under today's rules, an all-TIE is a speculative pick in the facing ladder and a Clear REJECT under the motion charter's `adopt_requires: Q`.
1. Offline: COMMAND margins without the cross column, post-clip once survival-floor-audit Stage A exists, plus the term's per-category income share on the X08:c6 raws (stamp 2026-09-23_05-43-42).
2. Fork X08:c6 model_7996 for 2000 its with the term off, against X01:c6ft (s3) and X02:c6ft (s2). These controls are valid only for model_7996 forks; a fork of the M1b head needs a fresh control. Run seed 3 first, and add seed 2 only if a WATCH metric crosses its margin.
3. Formation: ride along in the next from-scratch C1d chain.
4. A 2b teacher fork, or the next teacher rebuild, before extending to 2a-2c.

**Dependencies & conflicts.**
- Needs the knob and ni-adoption-rule.
- Soft ordering: after, or together with, progress-relative-cap.
- Independent of the clock.
- minimal-reward-arm also drops this term.

**Risks.**
- The progress-term coupling: projection-only income whose only cross price is exp and L1, and both decay in the tail.
- The large tail share.
- move_turn cross-track may creep up.
- Lateral drift near ring obstacles in 2b/2c.

**Rough cost.** Small code and a rule amendment. Fork: 0.77 GPU-h with X0 reused, or about 1.6 GPU-h with a fresh control, plus evals. The teacher check is extra.

**Adversarial review.** The verifier found:
- there is no knob for this term;
- no registered rule adopts an all-TIE result;
- the +0.02 margins sat below eval noise;
- the tail share is about 34%;
- the omni reward's own projection-only terms put it in the regime where the literature keeps an orthogonal penalty.

Adoption was narrowed to 1a-1c.

### posture-penalty-drop — Zero reward_orientation and reward_lin_vel_z on the clock-bearing recipe; keep them registered

**Kind & scope.** Remove by zeroing; the terms stay registered.
- reward_orientation: -0.7, tilt², masked to flat tiles.
- reward_lin_vel_z: -0.15, body-frame v_z², halved on ring tiles.
- All phases; screened on the c6 overlay.

**What changes.**
- Both weights go to 0.0 but both terms stay registered, following the repo's sweepable-inert convention (parkour_mdp_cfg.py NOTE ~:863).
- No unregistration until the no-clock recipe has its own paired posture-off verdict.
- Prerequisite: a KRABBY_LIN_VEL_Z_W knob (removal-knob-batch). KRABBY_ORIENT_W already exists (:1299).
- Fallback on REJECT: drop lin_vel_z alone, since it has the weaker case for keeping.
- Unmasking orientation is a separate candidate (orientation-unmask).
- crab_failure (tilt > 0.5 rad OR hip contact > 500 N) remains.

**Why — repo evidence.**
- **Income.** Orientation -0.0033/s (0.11%) and lin_vel_z -0.0015/s (0.05%). The same holds for t3-c6 and for the teacher.
- **The replay cannot tell basins apart with these terms.** Per hold, orientation costs -0.000 to -0.011/s and lin_vel_z -0.001 to -0.003/s, the same for walker, mid and creeper, against NET basin differences of 1-2/s (replay_reward_economy.md).
- **No leading signal.** Before onset, faller pitch_pk is about 0.012-0.013 rad (sin² ≈ 2e-4), the same order as survivors (≤ 0.089) (ANALYSIS_1c_failure:87-88). Onset to termination takes 0.54 s.
- **The 0.5 rad cutoff does not prevent tip-overs either.** crab_failure ends 36% of X08:c6 episodes (Episode_Termination/crab_failure 0.361; total_terminates 0.640). It also includes the hip-contact branch, so not all of those are tilt falls.
- **Dose history.** Era A: L1 at -2.0 was inert (lean_reduction RESULTS.md:37,46,58). Era B: A3 at -1.5 was null (gait_formation CHANGELOG.md:59). lin_vel_z has never been screened (3c29d32d, on the ~104 kg plant).
- The 2026-08-20 wheelie was fixed by max_speed_scale 1.05 (parkour_mdp_cfg.py:848-851), not by orientation.
- **The tile mask.** It comes from the Extreme Parkour port (^7283a7bb). It zeroes 20% of envs in 1a-1c and 50% in 2a-2c. Ring blocks are 2-9 cm under a ~1.0-1.1 m body, and the policy observes the tile flag (observations.py:180-184, 212-213).
- There is no A15+B or omni ablation of either term.

**Why — literature.**
- Hexapod designs carry no attitude or v_z shaping:
  - Qu 2024 Table I [28];
  - LiBR: lin_vel_z -1e-8, ang_vel_xy 0.0 [15];
  - Yang, Gao, Li [29] dropped z-velocity and roll/pitch-rate terms and the curves coincide, though they warn that pure RL stumbles.
- Rudin 2021 has no orientation row, and crashes end by reset [22]. Kim et al. cover posture with one COM-frame constraint [18].
- A termination acts as a price only while reward is net positive: CaT's r ≥ 0 [17], legged_gym's only_positive_rewards [2].
- The mask is Extreme Parkour's rule [12].
- **Counter-evidence:**
  - Isaac Lab's base/rough flat_orientation_l2 is 0.0 [4], but every stock flat config turns it on: ANYmal-D at -5.0 [8]; local copies show anymal_b/c/d at -5.0 and go1, go2, a1 and cassie at -2.5.
  - Quadrupeds keep v_z at -2.0 [2,4].
  - Lee et al. 2023 keep vertical velocity and roll/pitch rates as style rewards [26].
  - Hwangbo 2019 phase regularisers in during formation [24].

**Hypothesis / falsifier.**
- Hypothesis (a): at convergence on the clock-bearing c6 recipe, zeroing both terms is a TIE.
- Hypothesis (b), separate: formation is unaffected. Only a from-scratch arm can test this.
- Falsified (two seeds vs X0) by any of:
  - a survival-guard breach, especially on diag, back or move_turn;
  - a rise beyond seed noise in slow-canary pitch-forward share, pitch_max_abs_p50, or the training crab_failure hazard by tile type;
  - Q WORSE (contact_switch_hz or slip_per_m, through bobbing).
- Offline leading-window test: orientation and v_z cost from -3 s to -1 s before tilt onset (> 0.1 rad) separates fallers from survivors per category.

**How it would be decided.**
0. CPU: run the orient and vz2 kernels of replay_reward_economy.py on the X08:c6 suite npz (2026-09-23_05-43-42, 999 episodes) and the slow canary. Report per-category cost, the leading-window test, and per-slot post-clip net reward (stand especially). Proceed unless some slot's net is non-positive.
1. A 2000-it fork_c6, seeds 3 and 2, against X01/X02 under D00bs. Judge F, F_mt and Q at R_seed. An ADOPT under ni-adoption-rule zeroes the weights on the c6 recipe only. The fork certifies only "harmless at convergence", and a TIE is expected on a saturated-mean policy (median |mu| ≈ 9).
2. Stronger claims need a paired from-scratch arm, preferably inside minimal-reward-arm or the no-clock pairing, and never bundled with clock removal.

If M1b adopts first, fork from its head.

**Dependencies & conflicts.**
- Needs the lin_vel_z knob and ni-adoption-rule.
- Clock: removing the clock also removes its min_upright_gz 0.9 gates (tilt < 0.451 rad) and 0.444/s of income, and Yang warns that base-motion penalties matter more without a gait prior. The no-clock recipe gets its own posture-off arm before any unregistration.
- M1b; minimal-reward-arm also drops both terms; a REJECT triggers soft-pitch-constraint and orientation-unmask.

**Risks.**
- The contribution during formation is unknown.
- The v_z literature disagrees. The re-entry form, if needed, is one merged base-motion term (Isaac Lab Spot [6]; Kim's orthogonal-velocity constraint [18]), not re-enabling -0.15.
- Do not change max_speed_scale at the same time.
- Removes the only continuous tilt price below 0.5 rad on flat tiles.
- A per-tile income shift may move terrain promotion.
- The fork TIE is close to tautological.

**Rough cost.** Offline under 1 CPU-h; about 0.5 h for the knob; fork about 2-2.5 GPU-h including evals. A decisive formation answer needs at least 3.6 GPU-h per seed per arm unless it is folded into minimal-reward-arm.

**Adversarial review.** The verifier:
- moved Isaac Lab to counter-evidence;
- replaced a circular "last second before the fall" test with a leading-window test;
- found the missing lin_vel_z knob;
- kept both terms registered until the no-clock check;
- moved the unmasking fallback out.

### foot-idle-audit — Find what foot_idle's 7.9% pays for; test a carry-over reset separately from a cap; remove only after the clock decision

**Kind & scope.** Fix-then-decide. Term: penalty_foot_idle_when_forward, -0.12 with a 90-step threshold, all phases; it has no knob.

**What changes.**
1. An offline decomposition.
2. Two edits, tested separately:
   - **(2a) Carry-over reset:** the counter restarts when the gate goes from closed to open, or at slot boundaries. Replay both variants.
   - **(2b) Per-foot cap:** an optional knob (KRABBY_FOOT_IDLE_CAP, default inf), swept and never bundled with 2a.
   - Also add KRABBY_FOOT_IDLE_W and KRABBY_FOOT_IDLE_MAX_STEPS knobs, defaulting to the stock values.
   - Keep the registered name so log keys stay comparable; fix only the docstring and cfg NOTE.
3. Choose between the corrected form and removal. Removal is decided only after the clock decision.

**Why — repo evidence.**
- **What it prices.** A foot kept AIRBORNE for more than 90 steps; the counter resets at more than 0.1 N (rewards.py:951-987).
- **Size.** -0.236/s at X08:c6 (7.9%, the 4th largest term); -0.744/s in the first 10% of t3; it grew from -0.22 to -0.46 during c4.
- **Defects.**
  - The counter runs while the gate is closed and pays out when it opens.
  - The cost is unbounded and quadratic in hang time (:975-987).
- **Stand slot.** The stand slot opens the gate when the facing error exceeds 0.133 rad (wz ≥ 0.2 gives 0.24 > 0.12). The clock runs then too. Stand-slot charges are therefore largely charges on a leg parked through a real turn command. Stand six-down is 0.4%.
- **Replay gates.**
  - Gate 2 medians are 0-0.046 per category and pass everywhere (REPLAY_GATE.md:174-180), so the formed deterministic head rarely pays.
  - The mean is -0.21/s in replay and -0.28/s in training (:268, REPLAY_GATE's training-log window; the X08:c6 last-10% value is -0.236/s; final check); nonzero_frac is 0.45-0.81.
  - Gate 1 tests chatter, the opposite defect, so its FAIL is not evidence here.
- **The walker penalty.** The walker's -1.747/s against the creeper's -0.441 on the 1c fwd hold is the --idle carry variant (replay_reward_economy.py:374-387, 629, default carry). It comes under a stand→fwd order that maximises carry-over. On other holds the walker pays -0.217 (lat_l) and -0.368 (diag). The reset bracket was never reported.
- In training, 1a ended at -0.55/s and 1c at -0.06 (ANALYSIS_1c:78, 105).
- **The case for keeping it.** The phase-out floor replay called it "the load-bearing anti-skate/drag defense", at -177/min for skate and -1799/min for drag (phaseout CHANGELOG:84-90). That was old plant, position era, 1.0 N threshold. The margin comes from the unbounded tail, which a cap removes.
- **Margin to the clock.** The slowest running clock (about 0.29 rev/s at 50/50 duty) gives a scheduled swing of about 1.75 s against the 1.8 s threshold. Check this against the actual walk floor and the measured airborne time.

**Why — literature.**
- Yu, Turk & Liu Sec. 6.1 [32]: a hexapod trained without gait terms lifts its front legs at high speed; the paper presents this as the learned high-speed gait. It is only weak motivation for some anti-parking pressure.
- Isaac Lab Spot [6]:
  - air_time_reward withdraws income above mode_time and pays stance time under a zero command;
  - air_time_variance_penalty covers only completed swings (last_air_time) and is not command-gated;
  - neither guards a permanently parked leg.
- No verified reference supports a bounded anti-parking penalty. CaT [17] and Kim [18] express similar roles as constraints.

**Hypothesis / falsifier.**
- Hypothesis: most of the -0.24/s is carry-over across slot boundaries plus a small episode tail. Stand-slot accrual during facing corrections is a separate, partly legitimate component.
- Live alternative (dormant guard): the formed head rarely pays; the training cost is an exploration-noise tax that no reform removes without removing the guard.
- Falsified if any of:
  - offline carry-over is under about 1/5 of the cost and the tail is genuine in-motion leg parking (then only the knobs and docstring change);
  - the corrected form no longer ranks the forward skate/drag fixtures and the 1c creeper below the healthy heads;
  - in training, the corrected fork is WORSE on T, F or survival, or the leg-parking guards worsen.

**How it would be decided.**
1. Offline, first action: rerun replay_reward_economy.py with --idle reset vs carry on the 1a/1b/1c npz.
   - Split stand-slot accrual from cross-slot carry-over.
   - Estimate the TRAINING carry-over share from the training slot sampler's statistics (stand_frac, slot durations, 40-s episodes), not from eval hold order.
   - Report the post-clip share.
   - Ranking check on the forward skate/drag/tiprock/lunge fixtures with the cap swept: healthy must out-earn every degenerate family, with a drag margin at least the stock one.
2. A fork, only if carry-over is at least about 1/5 of the cost: 2000 its from X08:c6 against X0, seeds 3 and 2.
   - Decided by non-inferiority with margins tied to the documented MDEs.
   - Decisive guards: stand-slot six-down, per-foot minimum duty, and foot_never_lands, both overall and on stand.
3. Removal is decided only in a from-scratch C1d chain, 2 seeds, after the clock decision.

**Dependencies & conflicts.**
- The offline audit can start now; it uses survival-floor-audit Stage A for the post-clip share.
- M5 (stand wz held at 0) and KRABBY_OMNI_STAND_PARK already remove the stand-slot component, so evaluate 2a on top of them or instead of them.
- M1b is preferable before training arms.
- The removal arm waits on the clock: removing both the clock and this guard is the documented failure regime.
- minimal-reward-arm drops this term outright, which conflicts with this candidate.

**Risks.**
- Reopens skate, drag or parked legs.
- The carry-over reset removes the only penalty on stand-slot leg parking unless M5/M6 land.
- The cap is the edit most likely to reopen drag.
- The logged income is pre-clip.

**Rough cost.** Offline about 0 GPU-h, using an existing script. Fork 2 × about 0.9 GPU-h plus evals, and it may be skipped. Removal arm 7.2 GPU-h plus evals.

**Adversarial review.** The verifier:
- corrected the reading of Spot's terms as "bounded guards";
- dropped Gate 1 as evidence and dropped the rename;
- split the carry-over reset from the cap;
- pointed to the existing --idle reset replay as the first zero-GPU test.

### rate-follows-bounds — One re-derived action-rate arm on top of the already planned M1b

**Kind & scope.** Fix-then-decide, as a follow-on arm to M1b. Terms: reward_action_rate (-0.3, unsquared, all phases) and penalty_action_second_diff (knob).

**What changes.**
- M1b is already planned as a single-factor overlay: KRABBY_ACTION_BOUND_LOSS 0.001, ACTION_STD_RESET 0.4, ACTION_STD_MAX 0.5 (plan there-is-a-lot-compressed-unicorn.md:138-146). That design is not touched.
- This candidate adds one arm, "M1b + re-derived rate", which differs from M1b only in the rate kernel and weight:
  - one squared smoothness term, the legged_gym first difference, optionally merged with the second difference in Kim's style, which retires the smooth2 knob;
  - its weight derived from M1b's own rollouts at σ 0.4-0.5 (prior range -0.01 to -0.03), chosen so it is at most 10% of task income in the binding category.
- An optional "M1b + rate 0" arm. M1b itself is the stock control.
- The staged M3 overlay is marked stale.
- clip_actions stays at 1.0.

**Why — repo evidence.**
- **What the term reads.** raw_actions clipped to ±1 (vecenv_wrapper.py:135-136), delayed one step (parkour_actions.py:348-357), priced unsquared (rewards.py:333-343).
- **Monte Carlo of E|Δa| per dimension:**

  | σ | at \|mu\| = 0 | 1 | 2 | 3 | 5 | 9 |
  |---|---|---|---|---|---|---|
  | 2 | 0.95 | 0.82 | 0.54 | 0.27 | 0.03 | 0.00 |
  | 1 | 0.82 | 0.55 | 0.15 | | | |
  | 0.5 | 0.55 | 0.28 | 0.01 | | | |

- **The post-M1b noise floor** at σ 0.4-0.5 and |mu| ≈ 0.5 over 18 dimensions is about 2.1-2.6 per step, i.e. about 0.63-0.77/s at -0.3, against task income of 1.49-2.10/s.
  - REPLAY_GATE's σ-0.5 share for the stock term is 15.1-16.1% (:184-191), but it was computed on the saturated mean. The relevant number is the unsaturated share, measured on M1b's rollouts.
  - A squared -0.01 term would cost about 0.07-0.16/s (3-11%); the top of the prior may exceed 10%.
- **Current state.**
  - X08:c6 median |mu| is 8.1, with 96% of channel-steps above 1.
  - The term costs -0.411/s in training (the REPLAY_GATE.md:266 training-log figure, a different window from the X08:c6 last-10% value of -0.342/s; final check), -0.297/s on the replayed mean and -0.478/s under the σ-2 Monte Carlo.
  - On X08 t3, σ goes 1.50 (it 0) → 1.64 (300) → 1.82 (900) → 1.93 (1500) → 2.00 (2700). The rate term's Episode_Reward goes -0.005 (it 0, a normalisation artefact) → -0.518 (300) → -0.578 (2700). σ rising to its clamp while the term is active weakly argues against the term dominating σ.
- **Replay gates.**
  - Gate 1: rho vs contact_switch_hz is 0.13-0.38 in fwd/side/diag/turn/move_turn. The per-second cost does track switching in turn (rho ≥ 0.5), so the FAIL is partly a per-metre/per-radian normalisation artefact (:106).
  - Gate 2: 14.5-18.3% share, failing in 5 of 7 categories (23.5-28.3% under σ 2).
- **History.** The -0.3 bake rests on position-era, pre-rebuild evidence (short_runs r14). The one post-rebuild test (wave L) found no arm that cut the action rate by 20% or more, and completion fell 0.90 → 0.81 at -1.0. L3 is confounded by the +0.001 power weight.
- 33 t3 checkpoints exist (model_0 to model_2900).

**Why — literature.**
- legged_gym uses -0.01 on the sum of squares with clip 100 [2]; Isaac Lab action_rate_l2 is -0.01 [4]; Extreme Parkour uses an unsquared norm at -0.1 with clip 1.2 [13].
- Seyde et al. Sec. 5.4.2 / Fig. 9 [41]: difference costs mitigate bang-bang behaviour but lower nominal performance. A rate cost prices sign flips, not a constant saturated sign, so this fits with the saturation argument. Fig. 7: large action penalties can suppress exploration, which is the relevant warning after M1b.
- Andrychowicz Sec. 3.2 [21]: an initial std of 0.5 did best except on Hopper; tanh is slightly better, possibly through smaller initial actions.
- Bound losses: rl_games soft bound 1.1 [55]; IsaacGymEnvs AnymalPPO bounds_loss_coef 0.001 [56].
- Kim Sec. V-B [18]: one smoothness term of first plus second difference. This is a structural precedent only; its k_s 0.06 does not transfer.
- The SpiderPi hexapod uses a squared term at -0.5 [28].

**Hypothesis / falsifier.**
- Hypothesis: once M1b bounds the mean, the stock term charges a floor of about 0.63-0.77/s that pushes the policy back toward saturation or shorter episodes. A squared term at no more than 10% of task income keeps the mean bounded, with the non-circular Q metrics no worse and T/F held.
- Falsified if:
  - (a) M1b plus the stock term matches the re-derived arm on frac |mu|>1, median |mu|, T, F and Q;
  - (b) the re-derived or zero arm is WORSE than R_seed on non-circular Q or stand six-down;
  - (d) under M1b, the rate term's income does not rise toward the predicted floor (stays below about 0.45/s).

**How it would be decided.**
- **Step 0 (CPU, descriptive only).** Saturation forensics on the t3 checkpoints. This cannot attribute cause, because σ is 1.5 from iteration 0 and the rate price and the clip effect start together. Optional attribution ablation: from-scratch t3 for 300-500 its, rate -0.3 vs 0, two seeds, comparing when frac |mu|>1 sets in (about 0.15-0.25 GPU-h per arm-seed).
- **Step 1.** Offline replay gate at σ 0.4-0.5 on M1b smoke rollouts (proxy: X08:c6 npz with mu clipped to ±1.1). Gate 1 is judged against contact_switch_hz only.
- **Step 2, conditional.** Run M1b as planned, logging Loss/action_bound, frac |mu|>1 and the rate term's income.
  - If M1b re-saturates or shortens episodes while the rate cost rises, run the re-derived arm.
  - If M1b stays unsaturated and passes Q, the arm becomes a simplification screen.
  - Arms: 2000-it forks, two seeds each, with X01/D00s/D00bs as noise references. Decide on non-circular Q, T, F, F_mt, survival and the saturation diagnostics.
  - Adoption into the shipped recipe needs a later from-scratch C1d chain.

**Dependencies & conflicts.**
- Soft: comes after M1b's first result; M1b's plan is fixed.
- Clock: the clock pays for the rod reversals this term taxes, so redo Step 1 on clock-free rollouts if the clock goes first.
- Supersedes the M3 calibration. Alternative to caps-smoothness-loss. Needs config-truth-align's sign guard.

**Risks.**
- Reopens the only positive evidence for the bake, which is position-era: tippy-tap 19.1% → 13.4%, stride 0.178 → 0.198 m.
- Jitter may rise (M1 saw +10-15%).
- Forks start from a saturated head.
- Seyde's exploration-suppression effect.
- If M1b is REJECTED for unrelated reasons, this reverts to a stand-alone from-scratch simplification screen.
- Wave L's finding that "churn is load-bearing" may still hold.

**Rough cost.** Steps 0-1: CPU hours. Marginal cost: 1-2 arms × 2 seeds ≈ 1.7-3.5 GPU-h. Optional ablation about 1 GPU-h. From-scratch confirmation 3.6 GPU-h per seed × 2.

**Adversarial review.** The verifier found M1b already fixed as a single-factor overlay, so this became a follow-on arm. It also:
- recomputed the noise floor at M1b's σ 0.4-0.5;
- made the Step-0 forensics descriptive only;
- corrected the Seyde locator and reconciled its bang-bang finding.

### delta-torque-drop — Remove reward_delta_torques; test on a converged head and, separately, during formation

**Kind & scope.** Remove. Term: reward_delta_torques, -1e-6 in all phases.

**What changes.** Weight 0, then unregister.

Fallbacks:
- (i) a ramped weight, RMA-style, from small up to the current value, instead of a fixed -1e-6. This addresses the load-transition concern.
- (ii) optional hygiene: make the class honour asset_cfg.joint_ids. Today's config deliberately passes all joints (SceneEntityCfg('robot'), parkour_mdp_cfg.py:1150-1154), and a rod-only filter changes only about 5.5% of the cost.

**Why — repo evidence.**
- **Income.** -0.069/s (2.3%) at X08:c6; teacher 2a -0.071, 2b -0.072. It grows within stages (t3 -0.078 → -0.088), which is what a walking-proportional tax should do; that growth is not evidence of harm.
- **Healthy vs degenerate.** The creeper pays about 3-4× less than the walker on every hold: fwd -0.016 vs -0.053, diag -0.016 vs -0.066, back -0.019 vs -0.067 (replay_reward_economy.md:41-75). That ~0.04-0.05/s differential is small next to the foot_idle (0.2-1.7/s) and action_rate differentials.
- **Never tested alone.** Its only evidence varied jointly with action_rate in the position-era short_runs Arm B: r12/r13/r14 cut tippy-tap from 19.09% to 13.39% at 2000 its, always with action_rate at -0.3 (RESULTS.md:82-103). task1_velocity kept it as a "position-era value" (CHANGELOG.md:89). L3's -3e-6 is confounded.
- **Code.** The class ignores the joint filter (rewards.py:824-842). The rod torque it prices is an implicit-PD estimate at k = 16000/6000 (crab_hex_scene_cfg.py:147-150, 270-285).
- **Unreproduced analysis** (no script or output saved; treat as unverified):
  - joint split: Hip_Femur 62.4%, Femur_Tibia 32.0%, Body_Hip 5.5%, cams about 0;
  - 51% of Σdτ² falls on the 22% of steps where a foot crosses 50 N;
  - correlation with ||Δa||: 0.03 at lag 0, 0.21 at lag 2.
- The 1c creeper's income moved from -0.064 to -0.017 (ANALYSIS_1c:78).

**Why — literature.**
- legged_gym [2] and Isaac Lab [4] use torque² and dof_acc penalties and no delta-torque term.
- Extreme Parkour uses delta_torques at -1e-7 [13]. Weights are not comparable across torque scales (Krabby's Σdτ² is about 8.2e4 per step), so compare income shares instead.
- RMA [27] uses a Smoothness term at 0.001, inside a curriculum of small initial coefficients, because naive penalties made the agent stay in place.
- Fu et al. Fig. 7 [39] report delta torque minimised as a byproduct of an energy term. That energy route has a negative record on this plant: power fails Gate 1 in 7/7, and forward C3/C4 found it suppressed lift.

**Hypothesis / falsifier.**
- Hypothesis: on a converged head, removal is non-inferior on Q, T, F, F_mt, the survival guards and stand six-down. Neutrality during formation is a separate, weaker claim.
- Falsifier (a): Gate 1, judged against a metric that is non-circular for a contact-transition term, shows the term ranks the defect (rho ≥ 0.5 in most categories). Suitable metrics are swings_per_m, swings_per_rad or a short-swing rate, per unit of progress. Raw contact_switch_hz is circular here.
- Falsifier (b): the removal fork is WORSE than R_seed on stand six-down, slip_per_m, contact_switch_hz, survival, or the tippy-tap proxy (swing-duration distribution, short-swing fraction, swings_per_m). The eval suite has no tippy_tap metric (eval_categories.py:154).

**How it would be decided.**
1. Reuse the existing omni reward-economy replay; use replay_motion_terms.py only for the Gate 1/2 tables.
2. A 2000-it fork with KRABBY_DELTA_TORQUE_W=0 against X01, two seeds, reusing X01/D00s/D00bs as controls and noise references. This is a weak test: it will predictably TIE and shows only "no harm to a converged head".
3. Pair it with a formation-time short chain (1a→1c, or X08 t3 alone) at weight 0 vs -1e-6, same seeds, judged on the tippy-tap proxy, swings_per_m and survival. Otherwise, state explicitly that formation risk is untested.

Never attach this to the clock-deprecation chain without a matched control arm. If M1b lands first, run on the post-M1b head.

**Dependencies & conflicts.**
- Ordered after M1b, or re-baselined if M1b lands mid-test: the measured behaviour comes from a bang-bang head.
- Needs ni-adoption-rule and config-truth-align's sign guard.
- rod-power-after-clock is the contingency if Q regresses. It does not supersede this candidate.

**Risks.**
- Tippy-tapping could return, and the fine-tune test cannot see it.
- Touchdown damping on the lead-screw hardware is unknown.
- Impact transients on the ring are unmeasured.
- All evidence comes from a saturated-mean head.
- A TIE is the expected outcome; the gain is simplicity.

**Rough cost.** Replay: CPU hours. Forks: about 0.9 GPU-h per arm-seed (about 1.8 GPU-h marginal). Formation short chain: about 0.9-1.4 GPU-h per arm-seed (about 4-6 GPU-h total), or waived.

**Adversarial review.** The verifier:
- replaced a circular Gate-1 falsifier;
- surfaced an existing replay showing the creeper pays 3-4× less;
- showed the fine-tune cannot detect the named tippy-tap risk and added a formation short chain;
- downgraded the joint-filter fallback to hygiene.

### contact-telemetry-audit — Settle what the teacher contact terms fire on, CPU-first, then keep, fix or drop each

**Kind & scope.** Fix-then-decide. Teacher phases; telemetry in all phases.
- reward_feet_stumble: -1.0 in 2a-2c.
- reward_collision: -2.0 in 2a-2c.
- crab_failure's hip-contact branch (500 N).

**What changes.**
- **Step 0, CPU only.** From the existing seed-3 npz (foot_pos_w, terrain_z/mesh, body_pos_w, and leg_contact_force for Hip/Femur/Tibia):
  - classify each femur and hip contact step as close to the ground or close to another leg;
  - compute the fraction of footpad-loaded steps on steep facets (ramp risers or roughness);
  - compute riser geometry per difficulty.
- **Step 1.** Add only what cannot be derived: the per-foot net force vector (xy/z) and the chassis body force norm, both read-only sensor reads.
  - No filtered partner sensor unless Step 0 is ambiguous. If one is added, re-baseline the same heads in the same eval build.
  - The crab_failure tilt vs hip split becomes a one-line addition to the training log.
- **Step 2, decisions.**
  - Stumble: (a) remove; (b) a geometry-aware or partner-aware form, e.g. one that excludes self-contact; (c) accept that the sim ring has no vertical stubs and drop it. In every outcome it is priced once; ring-bonus-drop removes the embedded no_stumble factor.
  - Collision: keep at -2.0 by default. Alternatives: trim the body list (the hip entries only on A15+B; A20+B shows 943 N), or filter self-contact if it is an asset artefact. (d) If femur contact turns out to be leg-leg interference, keep the term and consider it for earlier phases; that would be a separate from-scratch candidate.

**Why — repo evidence.**
- **Stumble income is confounded.** It went 2a -0.0151 → -0.0150 and 2b -0.0140 → -0.0150, but over the same windows terrain_levels rose 4.24 → 5.96 and 4.56 → 5.99, and mean episode length rose 1426 → 1541 and 1366 → 1482 under max-length normalisation. The rate per unit exposure therefore fell. Use per-step flag rates at a fixed difficulty instead.
- **Stumble kernel.** |F_xy| > 4|F_z| (rewards.py:486-498), with no force floor and no tile mask. It is also a factor inside obstacle_clearance_omni (crab_hex_omni_rewards.py:308).
- **Friction.** Terrain friction is 1.0 ('average' combine) and robot friction 0.6-2.0, so the effective coefficient is at most about 1.5. A ratio of 4 can trip only through solver transients, surfaces steeper than about 30-45°, or non-ground partners. Self-collisions are enabled, and the footpad net force includes leg-leg contact. parkour_flat tiles are roughened (noise 0.02).
- **Riser geometry** (terrains/utils.py:106-118): a vertical wall needs a rise of 1.5 × 0.05 = 7.5 cm per cell (12 cm with the generator's 0.08). step_height = 0.02 + 0.06d over [0.2, 0.7] is 3.2-6.2 cm, so no riser is vertical; each is a one-cell ramp of about 32-51°.
- **Collision replay** (seed-3 eval npz; eval_crab_hex_gait.py:563, 862-879):
  - hip contact above 0.1 N on fewer than 0.005% of link-steps, and 0 steps above 500 N;
  - femur contact on 1.4-2.1% of steps, not tied to obstacles: slow 1.75% → 1.50%; ring 1.57% → 1.39%; 1.46% with a foot over a block vs 1.37% without;
  - concentrated on the fwd (3.5-4.5%) and back (2.5-3.9%) holds, on the FL/FR/RL/RR femurs;
  - median force 90-120 N (p90 360-1000 N); 4-7% of contacts within 2 s before a failure.
- **The repo's own reading** is that hip and femur contact is the leg-leg interference signal (leg_mount_morphology RESULTS.md:274; hardware_morphology RESULTS.md:226, "leg-leg collision is a real constraint").
- Collision training income: 2a -0.0235 → -0.0184, flat in 2b. The chassis body is excluded from the termination (cfg :1422), so only this term prices body contact. Collision has never been ablated.
- **Omni teacher termination** is 0.5 rad / 500 N / .*_Hip (cfg_omni_2a/2b; crab_hex_omni_env_cfg.py:186-200). The TB tags crab_failure_{flat,obst,obst_rsi,hazard_*} already exist.

**Why — literature.**
- Stumble:
  - legged_gym _reward_stumble uses k=5, default weight 0 [1,2];
  - Extreme Parkour uses k=4 at -1, stated as "feet hitting vertical surfaces" [12]; the sim ring has none at training difficulties;
  - ANYmal Parkour uses k=2 at -1 [45];
  - SoloParkour treats it as a soft constraint (App. A.1, Table 2) [46];
  - none of them charges stumble twice.
- Collision is kept in every reference: legged_gym -1 [2]; Isaac Lab undesired_contacts on the thighs [4]; ANYmal Parkour [45]; the hexapod CPG-RL [29]; CaT as a hard constraint [17].

**Hypothesis / falsifier.**
- Ranked hypotheses for stumble: flags come from (i) steep contact on the ramp risers, (ii) rough-noise facets, or (iii) footpad self-contact. Near-zero flags on horizontal ground are expected.
- Ranked hypotheses for femur contact: (i) adjacent-leg interference, (ii) femur-on-ground posture, or (iii) obstacle contact.
- Falsifiers:
  - Stumble flags concentrate on riser or facet contacts near blocks and rise with difficulty at fixed exposure: then stumble is obstacle-relevant; keep it, priced once.
  - Femur contact is leg-leg interference: keep collision.
  - Femur contacts are riser or obstacle contacts, or cluster before falls: keep collision as it is.

**How it would be decided.**
- Step 0: 0 GPU.
- Step 1: re-evaluate 2-3 heads (seed-3 1c, 2a, 2b, plus the X08:c6 suite) on the ring, light and slow rows.
- Training screen only if Steps 0-1 show the flags are self-contact or facet noise: a 2a fork, paired control vs arm, two seeds.
  - Pre-registered non-inferiority margin: ring completion and falls within the pooled two-seed CI, plus the per-step stumble-flag rate at a fixed level.
  - At a 0.5% income share the screen can only exclude large harm.
  - It can ride along in ring-bonus-drop's screen, at the cost of attribution.

**Dependencies & conflicts.**
- ring-bonus-drop goes first.
- The recorder change must not perturb existing metrics.
- Shares the ramped-riser geometry with feet_edge in zero-weight-purge.
- Independent of the clock; re-measure femur posture on a clock-free baseline.
- Informs cat-style-constraints.

**Risks.**
- Dropping collision removes the only price on leg-leg interference, which is a hardware constraint.
- A filtered partner sensor changes eval process state.
- A ratio test that includes self-contact mislabels leg-leg footpad hits as stumbles.
- Toe-drag on riser faces is sim-to-real sensitive.

**Rough cost.** Step 0: 0 GPU-h. Step 1: about 0.5-1 GPU-h. Screen: 5-10 GPU-h, only if triggered.

**Adversarial review.** The verifier:
- showed the flat-income argument was confounded;
- showed ring risers are 32-51° ramps, so a force floor targets the wrong mechanism;
- cited the repo's own reading of femur contact as leg interference;
- reordered the work into CPU Step 0 first.

### one-yaw-kernel — One yaw representation, as two ordered parts: A facing-exp-retire, then B yaw-progress-retire

**Kind & scope.** Merge.
- Part A removes reward_tracking_yaw (the omni reward_facing_exp, 0.2 from 1c on).
- Part B is a dose cut, or a dose-preserving merge, of reward_yaw_progress (1.0 in every phase).
- Kept unchanged: track_ang_vel_z_exp (weight 1.0, std 0.5, ungated).
- Applies to every omni phase and every C1d stage.

**What changes.**
- **(A) facing-exp-retire:** set reward_tracking_yaw to 0 and change nothing else.
- **(B) yaw-progress-retire**, run only after A's outcome. Two options:
  - a dose cut with a registered acceptable loss; or
  - a dose-preserving merge: a small track_ang weight bump on gate-open rows only, sized so the total d(income)/d(wz) at wz = 0.083 is unchanged. This option adds knob surface.
- On adoption, delete the names and knobs: KRABBY_YAW_PROGRESS_W, the omni use of KRABBY_YAW_W, and _yaw_progress_clip.

**Why — repo evidence.**
- **The two terms are redundant in form** (crab_hex_omni_rewards.py:352-426; command_math.py:213-258).
  - On gate-open rows yaw_progress = clip(wz·sign(cmd_wz), ±0.3), a linear slope of 1. The kernel has a slope of about 1.44-1.68 per rad/s there.
  - yaw_progress is also the time derivative of −|e|, the level that facing_exp prices.
  - Through the relay, the kernel pays 0.698 on a saturated turn row and 1.0 once the turn settles. That +0.30/s settle bonus is much larger than facing_exp's turn-row span of about 0.07/s.
- **They are not redundant in dose.**
  - Removing yaw_progress cuts about 40% of the total turn-row slope (from ~2.4-2.7 to ~1.44-1.68).
  - D10 shows the total slope binds. Raising yaw_progress from 1 to 3 moved turn facing_closed 0.419 → 0.748 and the yaw_progress rate 0.054 → 0.097 rad/s, while the plant turns at 0.083 rad/s against a 0.3 cap.
  - Linear extrapolation predicts a loss of about 0.16 if yaw_progress is removed. That is at the single-seed MDE (~0.15).
- **facing_exp** earns +0.150/s (5.0%). In replay it pays 0.172-0.195/s on every translation hold for every head (replay_C2.md). It is near-constant, carries little gradient, and has never been ablated.
- **yaw_progress** earns +0.0044/s (0.15%). It has never been run at 0 with std 0.5. The "wobble-blind" argument for it was made at std 0.15.
- **move_turn.**
  - The ungated kernel still chases cmd_wz = 0.3 there, earning 0.19-0.22/s against yaw_progress's 0.005-0.02/s. Part B therefore barely lowers move_turn rotation pressure.
  - C3a's survival gain (0.58 → 0.98/0.995) came from lowering the cap, which moves both terms.
  - Logged c6 slot fractions: turn 0.150, move_turn 0.076.
- **X08:c6 reference values:** turn facing_closed 0.618, move_turn facing_closed 0.235, move_turn survival 0.75, cat_tripod_min 0.396.

**Why — literature.**
- The reference stacks represent a heading target once: a P-controller in the command plus one rate kernel in the reward.
  - legged_gym: commands[:,2] = clip(0.5·wrap(·)), heading_command True, tracking_ang_vel 0.5 vs lin 1.0 [1,2].
  - Isaac Lab: heading stiffness 0.5, track_ang weight 0.5 [4,10].
  - Also CaT [17], Kim Table I [18], Liu 2025 [30], Yang & Gao [29].
- Qualification: those servos are continuous, so their rate kernel prices facing error all the way to zero. The omni relay has a dead band (0.133 rad) and saturates above 0.2 rad, so the precedent transfers only partly.
- Extreme Parkour uses the facing kernel only where no rate command exists [12].
- van Marum Sec. V-C [34] compares controllers, not reward terms.
- Yang & Gao found a reward with fewer terms held up only with a rhythm prior [29].
- A hexapod paper cited by the yaw cluster (Jiang et al. 2026; its source was opened in the literature sweep) keeps a heading-hold term but does not ablate it. It is therefore not evidence either way (final check: corrected from "not re-verified").

**Hypothesis / falsifier.**
- Hypothesis A: removing facing_exp keeps F within a registered margin and keeps heading_final and direction_error on fwd/side/back/diag within R.
- Hypothesis B: removal is expected to cost about 0.1-0.16 of turn facing_closed. The question is whether that loss falls inside a registered margin, or whether a dose-preserving merge avoids it. move_turn survival is a guard, not a predicted gain.
- Falsified if the registered margin is violated (for example, pooled two-seed Δ turn facing_closed below −0.10, or any single-seed F WORSE), heading metrics fall outside R, or F_mt or move_turn survival is WORSE.
- A "WORSE on both seeds" falsifier would have only about 25% power at the expected effect size, so it is not used.

**How it would be decided.**
- **Offline gradient gate.**
  - Data: saved X08:c6 raws (seed3_X08_c6_A15pB/suite__omni/seed001/raw) and the X01:1c / X05:1c raws; all exist.
  - Compute the per-step marginal income d/dwz on gate-open turn rows, with and without yaw_progress.
  - Compute the price of the in-band heading hold (|e| < 0.133) on translation holds with facing_w set to 0.
  - Change weights through a modified --train-env-yaml; replay_motion_terms.py has no weight flags (:215-235, :1513-1527).
  - Also report the post-clip binding rate.
- **Maintenance kill switch.** fork_c6, 2000 its at 0.77 h each, against X01:c6ft and X02:c6ft. Both fork the seed-3 head, so this tests maintenance only.
- **Decisive test.** A from-scratch C1d chain (t3 3000 → c4 1500 → c5 1500 → c6 2000) with the part's weight at 0 on every stage.
  - Seed 3 is compared against X08.
  - Seed 2 needs a new seed-2 C1d control.
  - Register the margin and the tie-break (ni-adoption-rule) first.
- The F-block member "yaw_progress" is the eval metric yaw_progress_rate and is not affected by removing the reward.

**Dependencies & conflicts.**
- Needs survival-floor-audit Stage A and ni-adoption-rule.
- Clock: A can run as a paired arm on the clock-free base. B should not, because the gait gates read r_turn·|cmd_wz| and the clock session changes that coupling. Never stack either part with clock removal.
- M1b: fork from its head if it becomes the incumbent. C3a/C12 change how F_mt reads.
- turn-slot-facing is the alternative to A; yaw-progress-phaseout is the follow-up to B.
- minimal-reward-arm keeps both terms.

**Risks.**
- B reopens the reason yaw_progress was added.
- A leaves the in-band hold unpriced.
- A relay ceiling of about 0.915 on facing_closed, which does not bind at 0.62.
- Removing positive income makes the clip bind more often.
- Name-keyed telemetry must be updated.

**Rough cost.** Replay under 1 h. Forks 2 × 0.77 GPU-h plus evals. From scratch, 3.6 + 3.6 GPU-h per part, plus one shared seed-2 C1d control (3.6 GPU-h). About 9-13 GPU-h per part.

**Adversarial review.** The verifier showed that yaw_progress carries about 40% of the binding turn-row slope, so removing it is a dose cut. The bundle was split into A and B, the ordering replay was replaced with a gradient gate, and a registered margin replaced the low-power both-seeds veto.

### turn-slot-facing — Move the facing price off stand and move rows onto the turn and move_turn slots, using a pure env overlay

**Kind & scope.** Replace, with no code change. Swaps reward_tracking_yaw (every row) for reward_facing_turn with its shipped gate (slot_id ≥ SLOT_TURN). Applies from 1c onward, in C1d stages that set KRABBY_YAW_W 0.2, and in the fork_c6 overlay.

**What changes.**
- Set KRABBY_YAW_W=0 and KRABBY_OMNI_FACING_TURN_W=0.2 (crab_hex_omni_rewards.py:430-458).
- Turn and move_turn rows keep exactly today's facing price. Stand and move rows go to 0.
- A turn-only gate is a separate variant, needed only if a later move_turn-feasibility decision (C3a/C12) requires it.
- Raising the dose (C4's 0.8) is out of scope.

**Why — repo evidence.**
- Facing income on translation holds is 0.172-0.195/s across all heads trained with the term (1a/1b/1c; replay_reward_economy.py in 2026-09-16_1805_omni_formation). Equal income shows that every head holds heading. It does not show the term is unneeded.
- On turn holds the term separates heads (0.054-0.175/s). move_turn holds pay 0.055/s (replay_C2.md:89-94).
- The non-turn share of X08:c6's 0.150/s must be recomputed from the c6 pool mix. The ~0.13/s figure is an estimate.
- The heading hold on non-turn rows resets at every 10-s resample (psi = yaw, command.py:561-562). Outside 0.133 rad, the relay and the kernel hold heading. Inside it, track_ang_vel_z_exp prices only wz ≈ 0, so slow drift up to ±0.133 rad goes unpriced.
- A slot gate keeps paying after arrival, so it creates no overshoot or stop-short incentive.
- The replay tool already gates facing_turn on TURN_HOLDS = (turn_l90, turn_r90, move_turn) (:138, 428, 456), which matches this candidate.
- D10 (the C2 dose-up of yaw_progress) and D11 (C3a caps) argue for caution when changing move_turn pricing in either direction. That is why the move_turn price is kept.

**Why — literature.**
- ANYmal Parkour gates its heading reward on the final second (1{t*<1}, Table S2); that is a weak analogy. Its Table S6 "No H" ablation removes the navigation heading output, not a reward [45].
- Counter-evidence: Extreme Parkour's _reward_tracking_yaw is an ungated, every-step exp(−|e|) [12], which is the status quo.
- van Marum Sec. V-C [34]: a heading reward gave accurate rotation (a controller comparison).

**Hypothesis / falsifier.**
- Hypothesis (counterfactual): removing the facing price on stand and move rows does not increase translation or stand heading drift beyond a declared margin, and F, F_mt and T TIE.
- Falsified if heading_final (fwd/side/back/diag) or stand drift_speed exceeds its margin, or T is WORSE.
- F or F_mt WORSE would point to a slot-semantics bug, since the turn-row price is identical.

**How it would be decided.**
- Replay with the existing tool (under 0.5 h). PASS if turn and move_turn incomes are unchanged, translation and stand facing income goes to 0, and a within-band drift statistic is reported.
- Promote heading_final and stand drift_speed from WATCH (CHARTER.md:1113-1115) to non-inferiority guards sized against R_seed.
- fork_c6 on seeds 3 and 2 against X0.
- From scratch only if the forks TIE and the change is to be baked into new lineages.

**Dependencies & conflicts.**
- Needs no code, so it can run before one-yaw-kernel. It is the alternative to part A.
- Needs ni-adoption-rule.
- If adopted, dormant-lever-retire retires reward_tracking_yaw instead of reward_facing_turn.

**Risks.**
- The eval latch and schedule_hook re-derive slot_id semantics (command.py:876-925); a mismatch would pay the wrong rows.
- Drift inside the dead band goes unpriced.
- Stand rows lose about 0.19/s of free income. That is the intent, but it shifts the stand/move economy; check it against the C8 stand-share history.
- A later dose-up reopens C4.

**Rough cost.** Replay under 0.5 h. Forks 2 × 0.9 GPU-h. Optional from-scratch run 7-11 GPU-h.

**Adversarial review.** The verifier found that narrowing to turn-only would cut the move_turn price the candidate meant to protect. It was revised to the shipped gate as a pure overlay, Extreme Parkour was moved to counter-evidence, and the heading-drift metrics were promoted to guards.

### ring-bonus-drop — Drop the two ring-tile incomes; let the flat tracking family and the ring curriculum price obstacle travel

**Kind & scope.** Remove, in 2b and 2c:
- reward_obstacle_clearance (reward_obstacle_clearance_omni, weight 0.9)
- reward_tracking_goal_vel (reward_tracking_command_vel_on_obstacle, weight 0.75)
- stumble's second price

**What changes.**
- Zero KRABBY_CLEARANCE_W and KRABBY_GOAL_VEL_W in the omni-only 2b/2c window deltas (crab_hex_omni_phases.py). The module imports ELEMENTS_10K from the forward task at :119 and applies its own deltas, so ELEMENTS_10K is not edited and the forward task needs no coordination.
- Travel is then priced the same on flat and ring tiles, by track_lin, L1 and progress.
- Crossing difficulty comes from tracking-ratio promotion (ep_track_dist/ep_cmd_dist, KRABBY_TERRAIN_PROMOTE 0.45:0.25).
- Pre-declared fallback: remove only obstacle_clearance and keep goal_vel.

**Why — repo evidence.**
- **Code fact.** reward_obstacle_clearance_omni (crab_hex_omni_rewards.py:283-323) has no lift, height or crossing factor, despite its docstring. It is the product: ring tile × no_stumble × progressing × moving × landed. It is a gated upright-progress indicator that duplicates the tracking family by construction.
- **Effective params.** min_goal_progress 0.05 and min_forward_speed 0.1 (cfg_omni_2b.json). The function defaults are 0.15/0.25, so replays must use the effective values.
- **Training 2b (2026-09-18_12-58-58).**
  - Clearance went +0.111 → +0.118 and goal_vel +0.092 → +0.098 between the first and last 10%. Both were at full level from the start.
  - Together they pay 0.216/s, against 0.585 (track_lin) + 0.047 (progress): about a 34% top-up in all-env means. The per-ring-tile top-up depends on the flat-tile share, which has not been reported.
- **Replay (seed-3 ring rows, 100 episodes each).**
  - The indicator is on for 51% of move-slot steps for the 1c head, which never trained with it, and 56% for 2b.
  - Its pay share with a foot over a block is 0.154/0.198, against a block-step share of 0.125/0.206. It is indifferent to obstacles.
  - 69-82% of the pay falls within 2 m of spawn. Training field_frac is about 0.10-0.11.
- **Outcome did not improve.** Ring completion went 0.56 (2a) → 0.54 (2b); light 0.59 → 0.54.
- **Real 2a→2b confounders.** KRABBY_HEADING is already in DROPPED_KEYS (:33-35, 137-150), so it is not one of them. The actual confounders:
  - the facing band 1.2 → 2.4 (heading_err_rms 0.308 → 0.604)
  - foot_clearance +1.0, also armed on flat tiles
  - swing_min -0.4
  - apex/airtime/stride halved
  - +5000 its
- **Counter-hint.** diag speed_ratio 0.46 → 0.51 and back 0.54 → 0.60, which the fallback keeps covered.
- **Benefit.** Removing a full-dose income step at 2a→2b shrinks the critic value-scale jump (forward teacher_handoff chains #3-#5).
- **Phase C supports removal.** Clearance at 1.8 produced no lift; a dense swing-height shaper did (phased_flat CHANGELOG.md:160-205; pre-rebuild).
- **The 2b1 collapse is not reopened.** That came from unmasked goal_vel; this removes the masked copy.
- Neither function uses the clock.

**Why — literature.**
- Zhuang et al. Sec 3.1 [44]: one r_skill = r_forward + r_energy + r_alive for every skill.
- Cheng et al. Sec 4.2 / Eq. 2 [14]: "simple reward functions impose no priors". Its min(<v, d_w>, v_cmd) is the primary, unmasked tracking term on all terrain. That equally supports the opposite de-duplication: one unmasked progress family.
- SoloParkour's progress kernel [46].
- This is a de-duplication, not a rejection of the kernel.

**Hypothesis / falsifier.**
- Hypothesis: the two terms pay for walking along the command on ring tiles, which the 2a head already does. Removal is non-inferior on obstacle-conditioned crossing, on ring/light completion and falls, on per-category speed_ratio, and on flat survival and tripod.
- Falsified by (pooled over two seeds, against a pre-registered margin):
  - worse obstacle-conditioned metrics. Training: reach_obst_frac, goals_passed, field_frac, block_steps_mean. Eval: falls or stalls conditioned on block contact or field entry, crossings per episode.
  - a diag/back speed_ratio drop larger than the block-T half-width.
- Completion and falls are guards only, because ring rows track flat rows almost exactly (2a: 0.57/0.59/0.56; 2b: 0.50/0.54/0.54). With single-seed noise of ±0.10, they can only exclude large harm.
- terrain_levels is a watch; it is driven by tracking.

**How it would be decided.**
- A paired 2b fork from a 2a head, control vs arm, full 2b length, EVAL_FULL at the end.
- Substrate: the seed-3 2a head 2026-09-18_09-41-10, whose recorded 2b head 2026-09-18_12-58-58 serves as a cheap, non-reproducible first control; or a 2a grown from X08:c6.
- Seeds 3 and 2. Run the fallback arm only if the main arm fails. Adopt under ni-adoption-rule.

**Dependencies & conflicts.**
- Needs a 2a substrate; no C1d-lineage 2a exists.
- Independent of the clock. If M1b is adopted, fork from the M1b lineage.
- Makes stumble's double pricing moot.
- Hard prerequisite for progress-relative-cap in 2b/2c.
- Can share an arm with swing-height-rebase's removal branch, at the cost of attribution.

**Risks.**
- The diag/back gain may come from goal_vel; the fallback keeps it.
- Ring tiles become about 25% less lucrative per step (0.216/0.848, all-env), which shifts the relative prices of action_rate and collision there.
- A terrain_levels shift is confounded with tracking.

**Rough cost.**
- About 2.25 GPU-h per 2b arm plus about 0.3 GPU-h of eval.
- About 10 GPU-h for 2 seeds × (arm + control); about 7.5 reusing the seed-3 control.
- +2.25 GPU-h per seed if a 2a must be grown. The fallback adds about 5 GPU-h.

**Adversarial review.** The verifier:
- found KRABBY_HEADING already dropped and listed the real confounders
- corrected the curriculum description and the income arithmetic
- re-read Phase C as support for removal
- replaced flat-dominated completion with obstacle-conditioned deciding metrics

### apex-term-retire — Retire the clock-gated swing-apex term after formation, as a pre-registered step of the clock work

**Kind & scope.** Remove after formation only. reward_clock_swing_apex today:
- 1.0 in 1a-1c / X08
- ramped 1.0→0.5 in 2a and 0.5→0.001 in 2b
- 0.001 in 2c

**What changes.**
- Zero apex from the first fine-tune stage onward (X08 c4+ / 1c+ / 2a), which moves the existing ε step earlier.
- Keep it in 1a / X08 t3 formation, unless a from-scratch t3 arm is added (apex 0 vs 1.0, 2 seeds, about +2.7 GPU-h).
- Remove only the apex tuples from KRABBY_PHASEOUT; that string also carries the airtime, stride and schedule ramps.
- Unregister in the omni subclass (set to None), after checking reuse of _fwd.phaseout_spec (crab_hex_omni_phases.py:125) and 2c_clockheld (:310).
- reward_clock_schedule is unchanged.
- Fallback, only if clearance regresses: swing-height-rebase's terrain-relative kernel.

**Why — repo evidence.**
- **It never delivered lift.** Forward J1 clearance 0.045 vs 0.044, J2 0.047. Clearance stayed pinned at 0.042-0.051 ("REWARD ROUTES FOR LIFT EXHAUSTED", gait_formation_v2 CHANGELOG.md:42-44).
- **Formation.** E1 (legacy_golden) formed without apex (:21). But every formation of record since M1C starts with apex at 1.0: crab_hex_phases.py:89-91 and GAIT_WEIGHTS[0], gated_lineage REPORT.md:196/204/212, and omni 1a / X08 t3.
- **Already phased out after formation in forward** (phaseout REPORT.md:46-53, 111-118). The omni teacher is already down to 0.0001 at 2b.
- **Measured defect route.** The reference is root_z − 1.05 (parkour_mdp_cfg.py:913-922). On the 1c fwd hold root z is 0.995, so a grounded foot reads h ≈ 0.055 plus the foot radius, close to the 0.08 target. A crouch can earn apex with the feet down.
- **The creeper earns the most apex**, but the effect is small and mixed:

  | hold | creeper | walker |
  |---|---|---|
  | lat_l | 0.272 | 0.208 |
  | diag | 0.322 | 0.217 |
  | back | 0.319 | 0.233 |
  | fwd | 0.180 | 0.212 |
  | move_turn | 0.194 | 0.219 |

  On lat_l/diag/back this is 6-8% of the creeper's NET advantage (+0.98 / +1.31 / +1.54). It reverses on fwd and move_turn.
- The eval's per-swing apex in world z and the term's root-relative in-window height are not comparable.
- The settled root sits 1.071-1.080 m above the stance terrain.
- Income is +0.133/s: 4.4% of |income|, about 6.5% of positive income.
- Apex income falls as turning is learned (0.215 → 0.069 in the yaw smoke).

**Why — literature.**
- Plain removal has no direct literature precedent; the case rests on the repo's null-lift evidence (forward, legacy_golden, single seed).
- Li et al. [35] replace their phase-gated gait rewards (Sec. III-B, Table I) with a discriminator toward stage-I tripod data (Sec. III-C). That is replace-then-drop.
- WTW's footswing term [36] is a commanded-height penalty gated by C_foot (Table 1, −0.6).
- Clock-free height pricing (Unitree G1 [54]; Spot foot_clearance, target 0.1 [7]) supports the fallback only.

**Hypothesis / falsifier.**
- Hypothesis: removal after formation does not reduce terrain-relative swing clearance, stride or tripod. A decomposition shows whether apex pays grounded feet under crouch.
- Falsified if, in both seeds, the apex-0 fork loses terrain-relative clearance or stride beyond pre-registered margins, fails cat_tripod_min or a survival guard, or regresses the turn block.
- Also falsified if the decomposition shows apex pays swing feet at genuine lift heights.

**How it would be decided.**
1. Offline decomposition on the X08:c6 eval raws: stance-foot vs swing-foot contributions, and the crouch share (h measured from root_z vs from the terrain). This needs clock_phase logged in the eval raws first (currently missing, eval_crab_hex_gait.py:689-698).
2. Paired forks from one base head: apex 0 vs 1.0, 2000 its, 2 seeds.
   - Pre-registered Q margins on terrain-relative clearance and stride.
   - 2a-onset stumble/collision as a guard.
   - An explicit turn-block readout.
   - Two seeds only establish "no large regression".

**Dependencies & conflicts.**
- Make this a pre-registered sub-arm, or the first step, of the clock-deprecation work. Close it automatically if the clock goes.
- If clock-hold-twin also runs, run this first.
- Shares a base head with M1b.
- The fallback depends on swing-height-rebase.

**Risks.**
- The "income-priced clearance lost to survival economics" note (:909-911) could reopen.
- Removes 4.4% of positive income.
- Nothing prices swing height on flat tiles before 2b.
- The forward evidence is single-seed.

**Rough cost.** Replay 0 GPU-h (plus logging). Forks about 3.6 GPU-h plus 0.5 GPU-h of evals. Optional t3 arm +2.7 GPU-h.

**Adversarial review.** The verifier:
- showed every formation of record since M1C uses apex from iteration 0, so the scope is now after formation only
- showed the creeper effect is small and reverses on fwd/move_turn
- showed the replay-margin falsifier could not fail
- noted the hexapod paper replaces gait rewards rather than removing them

### progress-relative-cap — Replace the absolute-cap progress term with the command-normalised, command-capped form

**Kind & scope.** Replace reward_forward_progress_along_command (weight 0.6, all phases; the omni twin). reward_tracking_goal_vel is involved as a coordination point only.

**What changes.**
- The omni progress term switches to the existing omni function reward_tracking_command_vel: min(v·d_cmd, |cmd|)/|cmd|.
  - It is signed, capped at the command, and exactly 0 at zero command, so it needs no gate (crab_hex_omni_rewards.py:157-166, 227-243).
- This is not a parameter reduction. The 0.12 gate and the absolute 1.05 clamp are traded for a v_min 0.05 guard (:148), a ±10 clamp, a NaN scrub and a new signed retreat penalty.
- **Parity dose** is derived, not asserted: w = 0.055 / E[min(v,c)/c over moving rows] on the X08:c6 raws.
- **Promoted dose** is about 0.6.
- The stall indicator from lit-review recommendation 3 is deliberately deferred.

**Why — repo evidence.**
- **The current cap never binds.** rewards.py:902-925 clamps to [0, 1.05 m/s absolute]. That value was set against the forward wheelie at 0.6-1 m/s (parkour_mdp_cfg.py:847-852). The omni caps are 0.35 fwd and 0.25 lat/back, with a 0.21 floor.
- **Consequences of the current form.**
  - Overspeed earns +0.6 per m/s; only L1 and the exp kernel hold it back.
  - The term is one-sided, so retreating is free.
  - Income at perfect tracking is 0.126/s at the lateral floor vs 0.21/s forward (1.67×).
- **But progress is not the main direction bias.** The exp kernel's standstill floor, 1.25·exp(−c²/0.04), pays about 0.26-0.42/s on side/back commands (c = 0.21-0.25) against about 0.06/s forward (c = 0.35). That is 3-6× progress's fwd/side difference.
- The promoted dose raises the zero-speed restoring slope on low-speed commands to 1.7-2.9 per m/s, which partly offsets that floor.
- **Size.** +0.055/s (1.8%); 9-16% of the along-command gradient below the command.
- **Tracking gap.** C1d T ratios are fwd 0.748 vs side 0.430. The gap partly predates C1d (B0 side 0.521 → 0.430-0.437 under the curriculum; REPORT.md:382).
- **The replay probe measured fwd only.** prog 1.5 gave +20.2% and prog 2.5 gave +23.7%, vs a +16.6% baseline. Under the NEW economy the margins are lat_l −25.3%, diag −31.0%, back −47.2% (:238). The 1c-era lateral "walker" moves only 0.027 m/s.
- **History.** Lit-review recommendation 3 (lines 213-216) was never acted on. C9 was removed unrun.
- The target function already runs obstacle-masked in 2b (+0.098/s). The mask exists because of the forward 2b1 collapse (crab_hex_omni_rewards.py:252-257).

**Why — literature.**
- Extreme Parkour: min(<v, d̂>, v_cmd)/(v_cmd + eps) [12]. The function body is verified; the claims about its weight and "only term" need their own config locator.
- Lee 2020 caps at 0.6 m/s [25]; RMA uses min(v_x, 0.35) [27].
- No verified source caps progress far above the command.
- Humanoid-gym XBotL keeps a banded ratio-to-command low_speed term [52], an argument for keeping a normalised progress signal.
- Rudin 2022 removes r_bias after formation [23]; Hwangbo keeps objective terms un-annealed [24].

**Hypothesis / falsifier.**
- Hypothesis: at weight 0.6 the normalised, capped form raises the zero-speed restoring slope on side/back/diag commands, partly offsets the exp standstill floor, and lifts lateral/back/diag tracking without a survival cost. Parity is a config simplification.
- Promoted arm is falsified if:
  - per-category COMMAND margins on lat_l, back and diag, computed on X08:c6 eval raws (side achieved ~0.11 m/s), do not rise by a stated amount (e.g. ≥ +10 points each);
  - side ratio does not rise beyond R_seed (0.058), meaning "dose-driven lift not reproduced";
  - the move_turn survival or ratio guard is WORSE.
- Parity is falsified if replay margins or G3 fall outside noise, or if the signed branch charges ≥ 1% of |income| on back/move_turn rows.

**How it would be decided.**
- **Offline gate, decisive for parity.** Add a normalised-progress column to replay_reward_economy.py; the existing --progress-w flag only rescales the current form. Report, post-clip:
  - per-category COMMAND margins;
  - the slope at v = 0;
  - the signed-branch charge on reverse drift and at move_turn starts;
  - all of the above with and without clock columns.
- **Train only the promoted arm.** From-scratch C1d, seed 3, then seed 2 if BETTER. Only after the clock decision and after ring-bonus-drop.

**Dependencies & conflicts.**
- ring-bonus-drop is a hard prerequisite for 2b/2c. Otherwise the same signal is paid twice there (0.6 + 0.75).
- The clock decision: progress is the natural carrier of motion income once the clock goes.
- Should precede track-single-kernel, which is the competing fix for the exp floor.
- Soft ordering: before cross-penalty-remove.
- M1b: saturation may mask tracking effects until it lands.
- Conflicts with minimal-reward-arm.

**Risks.**
- The signed branch stacks with L1 on reverse drift.
- About 1.7× higher side/back gain may worsen the move_turn start transient.
- Normalisation raises the gain on low-speed denominators.
- The promoted dose is closer to economy tuning than to simplification.
- The wheelie is not reopened: the relative cap is stricter.

**Rough cost.** Parity: CPU only. Promoted: 3.6 GPU-h plus evals, and +3.6 GPU-h for a seed-2 confirmation.

**Adversarial review.** The verifier showed that:
- the exp kernel's standstill floor, not progress, is the main fwd/side income bias;
- the fwd-margin falsifier was nearly tautological;
- the flat term's mask exists because of the 2b1 collapse.

As a result, parity became CPU-only and ring-bonus-drop became a hard prerequisite.

### tibia-pose-retire — Remove the May-era contact-gated knee-deviation penalty

**Kind & scope.** Remove penalty_tibia_deviation_in_stance (penalty_joint_deviation_when_in_contact, weight −0.28, all phases). It has no knob.

**What changes.**
- Plain removal in the omni subclass.
- A new knob (e.g. KRABBY_TIBIA_DEV_W) with a dump check.
- Fallback, only if M1b is adopted (bounded mean) and removal regresses stand: a command-gated form (Isaac Lab stand_still_joint_deviation_l1, cmd < 0.06). The M6 rho −0.61 precedent is a known risk for that form.
- If M1b is rejected, removal is final and there is no fallback.

**Why — repo evidence.**
- **Origin.** Added 2026-05-23 (06697969) on crab_simple with position actions, to hold "stance tibiae closer to perpendicular".
- **Plant change.** Since the 2026-08-20 rebuild, q_default is the knee at mid-stroke (crab_hex_linkage.py:214-222, ±0.310). It was never tuned, ablated or replayed after that.
- **Income.** −0.042/s (1.4%). The rise across stages (t3 −0.014 → c6 −0.031 in each first 10%) is confounded by survival and stance count.
- **Mechanism.**
  - About 0.15 rad² summed fits knee targets pinned at the ±0.25 rad action bound: 0.0625 per loaded knee × about 2.5 loaded knees × the survival fraction.
  - It does not fit the actuator stroke ends, which are 1.4-1.8 rad away and unreachable under clip 1.0.
  - Under bang-bang control both signs cost the same. So the term acts as a command-independent tax of about 0.0175/s per loaded leg, pushing against six-down stance (stand six_down 0.004).
  - This is the same failure family as the dropped M6 stand pose (rho −0.61; candidates.yaml:22-23, 288-296).
- **Replayable.** replay_motion_terms.py reads joint_pos (:474), the Femur_Tibia defaults (:155-162) and per-foot force (:314).
- It also taxes loaded-knee flexion in 2b/2c.

**Why — literature.** No contact-gated joint-deviation precedent exists in legged_gym or Isaac Lab velocity mdp. Those use command-gated or always-on forms: legged_gym _reward_stand_still [1], Isaac Lab stand_still_joint_deviation_l1 [5], Spot joint_pos −0.7 with stand_still_scale 5.0 [7]. This supports the fallback form only; it is not evidence of harm.

**Hypothesis / falsifier.**
- Hypothesis: while the mean is saturated, the cost is explained by n_loaded_feet × about 0.0625 × 0.28. Removal does not hurt T/F/F_mt/survival/slip, and does not worsen (may improve) stand six_down and settle_six_down.
- Falsified if:
  - the cost is explained by knee angle rather than contact count;
  - the cost correlates positively with stable six-down stance;
  - the fork is WORSE on survival, slip or the stand metrics.

**How it would be decided.**
- **Offline, on the X08:c6 motion-quality npz:**
  - per-step R² of the cost against n_loaded_feet vs against mean |q_knee − q0|;
  - the fraction of loaded-knee steps within 0.02 rad of ±0.25;
  - the cost by slot;
  - Spearman correlation with contact_switch_hz and six_down.
- **Training.** A 2000-it fork of the then-incumbent, weight 0 vs X0, seed 3 plus a seed-2 confirmation. Decided on T, F, F_mt, survival and slip, with the stand Q members as guards. Accept on no-WORSE, stated up front.

**Dependencies & conflicts.**
- Removal proceeds whatever M1b's outcome; the fallback needs M1b.
- Needs the knob and ni-adoption-rule. Independent of the clock.
- minimal-reward-arm also drops this term.

**Risks.**
- After M1b, knee angles become commandable, so stance knee posture could drift. Check it on an M1b fork.
- Interaction with 2b/2c clearance.
- A TIE is likely.

**Rough cost.** Offline about 0 GPU-h. About 0.5 h for the knob. Fork 2 × about 0.9 GPU-h plus evals. Fallback +1.8 GPU-h.

**Adversarial review.** The verifier found the "stroke end" claim confused the action bound with the actuator stroke. It identified the stronger mechanism, a stance-count tax under bang-bang in the same family as the dropped M6 term, and restricted the fallback to after M1b.

### minimal-reward-arm — Minimal-reward diagnostic arm: task kernels, smoothness, the clock as the one swing term, terminations; 9 terms off

**Kind & scope.** Structural, diagnostic only; adopting it is not the default outcome. Flat C1d stages.
- **Off:** foot_idle, tibia, excess_contact, progress, penalty_lin_vel_y, orientation, lin_vel_z, air time, stride.
- **Kept:** track_lin, L1, track_ang, yaw_progress, tracking_yaw (1c+), action_rate, delta_torques, crab_failure, and one swing term.
- 18 active terms become 9.

**What changes.**
- Kept terms stay at their effective Revision-4 weights.
- The single swing term is the clock (plus apex), frozen as it is. The arm runs strictly before the clock is removed.
- **Hard gate:** if the clock is already gone and no working swing term exists, the arm cannot run. The air-time timing fix is unimplemented, and the term fails Gate 1 off-axis.
- Freeze the term set and weights by recipe hash at pre-registration. Do not pick up other candidates' decisions mid-flight. Arm B, which also applies one-yaw-kernel's removals, is pre-registered separately.
- Re-balancing: either specify it exactly (e.g. one scalar on the negative regulariser set) or do none. The dropped set is net negative (about −0.30/s). "Restoring net income" would mean scaling positive terms down or adding a constant, and either changes the survival incentive under the zero clip.

**Why — repo evidence.**
- **X08:c6 income of the dropped set:** foot_idle −0.236 (7.9%), progress +0.055 (1.8%), tibia −0.042 (1.4%), excess −0.017 (0.6%), cross −0.014 (0.5%), orientation −0.003, lin_vel_z −0.0015, air time about 0, stride 0.
- **Subtracting the dropped columns** from the existing NEW-economy tables (no code) moves the walker-minus-creeper NET margin:

  | hold | before | after |
  |---|---|---|
  | fwd | −1.08 | ~−0.02 |
  | lat_l | −0.98 | −0.85 |
  | diag | −1.30 | −0.96 |
  | back | −1.54 | −1.31 |
  | move_turn | −0.105 | −0.094 |

  The creeper still wins every hold. A pre-clip "no collapse" gate therefore passes trivially and is informative only post-clip and on C1d raws.
- **foot_idle:** −1.747 is fwd-only and partly stand-hold carry-over; other holds are −0.22 to −0.38. These heads are 1a/1b/1c.
- **excess_contact** is the only dropped term that charges the creeper (−0.179 fwd, −0.107 lat_l, −0.087 move_turn).
- **tibia** is not subsumed by the termination, which bounds body tilt, not knee pose. The case for dropping it is its era-A rationale and the changed default. It is not replayable from these raws (:7).
- **L1** is the largest standstill-escaping slope, not the only one. The exp slope at v = 0 is about 2.6 at c = 0.25 and about 0.8 at c = 0.35; progress adds +0.6.
- Every dropped term was justified on era-A or era-B forward evidence. There is no omni A/B test on a dropped term; the only omni reward A/B, C2, was on a kept term.
- The Revision-4 isolation shows the terms interact (σ² +4.2, L1 +1.7, ang std +5.6, product +1.8 points).
- heads.json has no seed-2 X08.

**Why — literature.**
- legged_gym ships 9 nonzero scales [2]; this is the fair precedent.
- van Marum Sec. IV-B [34]: command-only terms produced hopping, so keep a regulariser floor.
- Fu [39]: energy alone lets gaits emerge.
- Knox et al. Table 1, check 6 [61]: eliminate redundant attributes.
- Booth [60]: a drop after ablating an overfit reward may reflect co-tuning.
- Kim et al.'s "only three reward terms are sufficient" [18] and CaT [17] work because style is moved into constraints. They support a different arm, not plain deletion.
- Disagreement: the literature's commands are large relative to kernel width. At omni's 0.21-0.3 m/s, standing competes (ANALYSIS_1c_failure). That is why L1 and yaw_progress are kept.

**Hypothesis / falsifier.**
- Hypothesis: the dropped terms buy little. TIE on T/F/F_mt and every survival guard; Q moves less than its resolution. A two-seed TIE at R_seed F ≈ 3.2 is compatible with effects up to the MDE and does not show zero effect.
- Falsified if:
  - a survival guard or cat_tripod_min is WORSE;
  - the dead_walker abort fires at t3 its 800-1000;
  - Q is WORSE (six_down, swings_per_m, foot_disp_end);
  - stand or move watch metrics regress.
- Pre-registered add-back order: excess_contact first, then foot_idle.

**How it would be decided.**
- **Step 0:** the subtraction above, repeated on C1d/X08 raws with the stand-hold reset variant, post-clip.
- **Step 1:** a kill-only safety fork. The M1b base is optional (final check); if M1b is not ready, use X08:c6. A TIE here is not evidence.
- **Step 2:** from-scratch C1d, seed 3 then seed 2, against the incumbent, plus a mandatory seed-2 X08 replicate.
  - Decisive: survival in all 7 categories, move_turn ratio, cat_tripod_min, and T/F/F_mt/Q.
  - Watch: stand.drift_speed, stand.wz_active_frac, direction_error, cross_track_ratio.
  - Training log: crab_failure share, track_ratio_mean ≥ 0.6, value loss, explained variance.

**Dependencies & conflicts.**
- Needs survival-floor-audit Stage A, ni-adoption-rule and removal-knob-batch. The knobs must be landed at a boundary before the clock landing, or the arm runs on a pinned pre-clock commit with its own overlay (final check: this resolves the loop in which the knobs land together with the clock landing while the arm must run before it). The clock condition is a hard gate. The M1b base is optional.
- Pre-empts cross-penalty-remove, posture-penalty-drop, tibia-pose-retire and the foot-idle removal arm.
- Conflicts with progress-relative-cap and one-yaw-kernel.
- Run either this or adopted-set-formation-check first, not both blind.

**Risks.**
- The standing basin, hedged by keeping L1 and yaw_progress.
- Idle or tippy-tap legs.
- Cross-track drift.
- C1d was tuned with the full stack.
- A TIE is not proof of no effect.
- Return-scale confound.
- Dropping both contact-count shapers at once means that if a shuffle basin appears, the arm cannot tell which one was load-bearing.

**Rough cost.** About 12 GPU-h plus suite evals: 0.9 for step 1, plus 3 × 3.6 for s3, s2 and the seed-2 incumbent.

**Adversarial review.** The verifier:
- corrected the claim that L1 is the only standstill gradient;
- showed step 0 passes trivially pre-clip;
- made the seed-2 incumbent replicate mandatory;
- reframed Kim and CaT as constraint-based support;
- made the swing term a hard gate.

### adopted-set-formation-check — From-scratch confirmation that the fork-adopted removals also hold during formation

**Kind & scope.** Structural. Covers the union of removals adopted on forks, for example penalty_lin_vel_y, orientation, lin_vel_z, tibia, delta_torques and yaw_progress.

**What changes.**
- Runs only once the fork arms have settled, and only if at least 2 removals were adopted. With a single adoption, fold its formation check into that removal's own confirmation step.
- One from-scratch C1d lineage (t3 3000 → c4 1500 → c5 1500 → c6 2000) carrying the union of removals, judged against one fixed anchor under ni-adoption-rule's cumulative budget. One pass/fail rule for both the hypothesis and the falsifier.
- The post-clock anchor must be named. Either run this after the clock-deprecation scratch arm is adopted as the new C1d anchor (fresh budget), or fold clock removal into one joint union arm scored against X08. Coordinate with the clock session.

**Why — repo evidence.**
- A fork is not the same head as X08:c6. The 2000-it fork alone moved move_turn facing 0.235 → 0.411 (s3) and move_turn survival 0.75 → 0.935, so fork verdicts are relative to a continued-training head.
- The forward phase-out scout was a fork on an earlier plant. It detected completion 0.99 → 0.48 while the tripod held at 1.00× with gait income at ε (phaseout REPORT.md:30). That supports only the claim that a held tripod can mask gait loss.
- lit-review-plasticity-cliff.md:109-113: critic reset was falsified; the rigidity sits in the actor's representation. A formed actor can keep a gait whose incentive is gone.
- During formation, foot_idle pays −0.74 at t3 start and tibia grows −0.014 → −0.030.
- Noise is large: X0 s3 vs s2 differ by 0.27 on move_turn facing; C1-family turn facing ranges 0.17-0.62; R_F 3.19 and R_T 1.131.

**Why — literature.** Patterson et al. sec 7, untuned agents in ablations [65]. Ash & Adams [67], warm start vs fresh init, is a supervised-learning analogy only.

**Hypothesis / falsifier.**
- Hypothesis: removals that pass as "maintain" on forks also pass as "form" from scratch, under the ni-adoption-rule margins and budget. cat_tripod_min is judged with its R of 0.028. gate_1c stays WATCH, since X08:c6 itself fails it (0.396).
- Falsified if the pooled pair fails the rule on any block or guard, or the cumulative budget is breached.
- A single seed can only rule out gross failure (z < −R on F, T or a survival guard). Bisect by fork-stage groups only if a gross failure appears in both seeds.
- F_mt and diag survival stay drift-only unless 3 or more seeds are run.

**How it would be decided.**
- Pooled s3 + s2 as the first decision.
- Either buy a C1d s2 anchor (~3.9 GPU-h), or explicitly reuse the s3 anchor with the √3/2 floor (run_candidates.py:3131-3154, 3440-3444).

**Dependencies & conflicts.** ni-adoption-rule; removal-knob-batch; at least 2 fork adoptions; the post-clock anchor decision. Overlaps minimal-reward-arm.

**Risks.** Scratch runs are not bit-reproducible, and C1d formation variance is large.

**Rough cost.** About 7.6 GPU-h for the pair, plus about 3.9 GPU-h for the anchor if bought.

**Adversarial review.** The verifier:
- removed the WATCH-gate falsifier;
- unified the pass/fail rule with ni-adoption-rule;
- made the first decision a pooled pair;
- required a named anchor for after the clock change.

### stock-reference-arm — Scale-matched literature-stock velocity reward on the omni command, as a reference arm

**Kind & scope.** Replace, as a low-priority reference arm. Flat C1d stages 1a-1c.

**What changes.**
- Apply the Isaac Lab velocity RewardsCfg set on the omni command. Every term is scale-matched to the plant under one declared rule:
  - the torque weight is set by (stock torque scale / A15+B torque scale)², or replaced by the existing penalty_mechanical_power knob;
  - torque and acceleration terms cover only the 18 actuated joints, not the passive linkage DOFs (the raws carry 24 columns);
  - the tracking variance is scaled to the omni envelope (σ² ≈ 0.03-0.04).
- Terms and weights:
  - yaw tracking at 0.5× lin, driven by the facing P-controller;
  - lin_vel_z −2.0, ang_vel_xy −0.05;
  - squared action_rate −0.01;
  - feet_air_time 0.125 with a 0.5 s threshold (inert until the float32 fix; say so);
  - undesired_contacts −1.0 on body/hip/femur.
- Clock terms are included only if the clock session still ships them.
- Pin the C1d stage env blocks. The stages set KRABBY_YAW_W 0.2 (candidates.yaml ~:365-377), and that must be overridden.
- Keep the 15% stand pool. It is part of the task, and departs from stock's rel_standing_envs 0.02.
- Declare the negative-total policy. Isaac Lab has none; legged_gym clips.
- Code needed: register joint_torques_l2, a flat-class dof_acc (reward_dof_acc exists only in the teacher classes, :295/:589) and stock undesired_contacts, plus knobs and replay-tool columns.
- Unlike minimal-reward-arm, this arm contains no repo-evolved form.

**Why — repo evidence.**
- **Headline stage-0 finding, already decided without GPU.** Raw stock dof_torques (−1e-5·Στ²) would cost:
  - −9.07 / −9.40 / −9.70 per s on the 1a walker;
  - −6.93 / −7.66 / −7.19 per s on the 1c creeper;
  - (reeval_corrected/{1a,1c}/slow__omni, episodes 00-02; p95 |τ| 670-1000 N·m).

  That is about 4× all positive task income (~2.1-3.0/s), and about 2/s heavier on the walker. The raw stock arm would collapse.
- The other stock terms are harmless on this plant: dof_acc ≈ −0.001, lin_vel_z ≈ −0.015, ang_vel_xy ≈ −0.002, squared action_rate ≈ −0.068.
- No post-fix from-scratch run has used a near-stock reward. The May first walk was pre-cam. Pre-fix PLAN D C5 (σ² 0.25 plus L1) crept, with tracking_ratio 0.07-0.24.
- 12 of the 18 current weights were set on May or era-A plants.

**Why — literature.**
- Isaac Lab RewardsCfg (std sqrt(0.25), no negative tracking term), CommandsCfg (rel_standing_envs 0.02, heading stiffness 0.5), and its base-contact termination [4].
- legged_gym scales, tracking_sigma 0.25, only_positive_rewards [2].
- The unsquared action_rate lineage is Extreme Parkour [12].

**Hypothesis / falsifier.**
- Hypothesis: the scale-matched stock form walks under C1d, T ties X08:c6, and Q is no worse. The repo's extra terms mainly buy F and F_mt.
- Caveat: scale-matching makes this a principled hand-tuned minimal reward that overlaps minimal-reward-arm, and a loss cannot be attributed between "stock" and a particular adaptation.
- Stage 0 fails on comparative gates:
  - walker-minus-stander COMMAND margin (synthetic v = 0 stander) below X% on fwd;
  - walker-minus-creeper margin;
  - any regulariser above ~15% of the walker's positive income.
- The G3 "negative at v = 0" gate is not used. It cannot pass for positive-only kernels.
- Stage 1 fails on:
  - creep (speed_ratio < 0.3);
  - wheelie or topple (pitch-forward fall share > 0.8, with rising falls);
  - no tripod (< 0.2);
  - breached survival floors;
  - a T or F loss beyond resolution.

**How it would be decided.**
- Stage 0 offline; the raws carry joint_vel and applied_torque.
- Stage 1 runs only if stage 0 passes and minimal-reward-arm has run. It is paired against the same X08:c6 control: from-scratch C1d on seed 3, with seed 2 only if the result lands inside single-seed resolution.

**Dependencies & conflicts.** Clock session; air-time status; minimal-reward-arm; the new-term code.

**Risks.**
- Creep or standing remain likely without L1 or progress, since nothing else gives a constant gradient at v = 0.
- lin_vel_z at −2.0 is 13× the repo weight, on a 230 kg plant that bobs.
- The yaw kernel pays non-turners.
- Stand-heavy economy under positive-only kernels.
- The termination set differs from stock.

**Rough cost.** Stage 0: about 2-4 h of agent time including term code. Stage 1: about 3.9 GPU-h per seed.

**Adversarial review.** The verifier:
- computed that raw stock torque costs about 4× the task income here;
- showed that G3 cannot pass for positive-only kernels;
- revised the arm to a declared scale-matched set with comparative gates, and made training conditional.

### clock-hold-twin — Income-only clock anneal vs a clock-held control, offered to the clock session

**Kind & scope.** Phase-out, as an evidence handover to the clock session. Terms: reward_clock_schedule, and apex only if apex-term-retire has not landed. The clock observation is kept.

**What changes.**
- **Coordination first.** Ask the clock session whether it wants an income-vs-signal decomposition. If its design already includes a held control, close this candidate and pass on only the gate definition.
- **If it runs:** paired forks from the incumbent lineage.
  - ANNEAL arm: reward_clock_schedule 1.0 → ε, following the forward step pattern.
  - HOLD twin: stays at 1.0.
  - Both arms share the same base and the same iteration count.
  - Optional third arm: ANNEAL + KRABBY_OMNI_STAND_PARK.
- **Why ε, not 0:** telemetry (run_phases.py:1729) and the combine applier keep working.
- **Plumbing.** The flat arm needs a KRABBY_PHASEOUT-on-X08 fork path, because the 2c / 2c_clockheld presets (crab_hex_omni_phases.py:309-318) resume from the old omni 2b head, not from C1d.
- The obstacle-window twin waits until a C1d-lineage 2b head exists.

**Why — repo evidence.**
- **No held control ever.** The clock has never been ablated, annealed or compared against a held control in omni (omni formation CHARTER.md:508-520).
- **Forward precedent** (legacy_golden, seed 3, sum mode, RSI):
  - The anneal was BAKED at 30k: obstacle completion 0.27 vs 0.19 and flat 0.79 vs 0.73, but tripod 0.563 vs 0.615 and tracking 0.407 vs 0.460 (phaseout REPORT.md:236-257).
  - The 1.0 → 0.5 step was ejected at 10k (+0.170) and failed round 2 (+0.141) before passing at 15k.
  - Those steps were bundled with curriculum additions (:244-248), so their timing does not transfer quantitatively.
- **Other unheld comparisons.** The A15+B anneal coincided with a collapse (a15b REPORT.md:57-126). Sideways 2c regressed (step −0.19, p = 0.004).
- **Mechanism.**
  - When the clock is stopped, product mode returns stance_quality alone (crab_hex_clock_reward.py:146-153; rewards.py:108-123). On stand and dead-band steps, the clock is therefore the only income for "six feet loaded and still".
  - It pays for stepping on drift-restart steps and on moving and turning slots.
  - The creeper matches or beats the walker on clock income on lat_l (0.484 vs 0.350), diag and back. The 1c stand hold scored tripod 0.576.
  - foot_idle and excess_contact share the clock-speed gate (rewards.py:60-90), so they also force cycling. That confounds any stepping readout.
- **Incumbent state.** cat_tripod_min is 0.396, below the 0.4 formation gate (CHANGELOG.md:89).
- **Stake.** 0.311/s: 10.3% of |income| and 15.2% of positive income.

**Why — literature.**
- Siekmann Sec. IV [37]: a periodic reward needs the clock observed. The converse does not hold, which is what allows an anneal that keeps the observation.
- Li et al. [35]: replace-then-drop. Here there is no replacement prior beyond the mirror loss. Their Sec. V-A found that hand-set gait rewards hindered terrain learning.
- HRP-5P [38], Sec. V / Table V: its remedy lets the policy modulate the clock. That supports cadence decoupling more than income removal.
- WTW [36], limitations / Table 5: this finding is from a sprint regime, extrapolated here.

**Hypothesis / falsifier.** Predictions, split by regime:
- (a) tripod and duty stay within pre-registered margins relative to HOLD;
- (b) F, F_mt and T do not regress relative to HOLD;
- (c1) on moving steps and on turning steps outside the dead band, stepping without translation or rotation falls;
- (c2) on stand and dead-band steps, six_down falls (more fidgeting) unless M5/M6 stand park/hold or a stand-pose term is on.

Falsified if, in both seeds, the anneal arm drops tripod or duty, or regresses F/F_mt/T, beyond the pre-registered margins relative to HOLD. A null result on stepping is uninformative, because of the shared gate.

**How it would be decided.**
- **Phase-free re-pricing of stand and dead-band steps** from saved rollouts. Valid now: only stance_quality pays there.
- **Moving-slot re-pricing** first needs clock_phase recorded (eval_crab_hex_gait.py:689-698). Otherwise the argmax fit is biased upward (walker best 0.679 vs mean 0.526). Use X08:c6's own rollouts.
- **Forks:** at least 2 seeds, guards relative to HOLD, numeric margins fixed in advance. Two seeds detect only large regressions.

**Dependencies & conflicts.**
- The clock session is a hard gate.
- If apex-term-retire also runs, it goes first.
- M1b shares the base head.
- survival-floor-audit sizes what removing 0.311/s costs.
- Couples with any candidate that touches the idle/excess penalties.
- The GPU is currently held by the clock campaign.

**Risks.**
- Qualitative timing risk from the forward steps.
- Stand regression, per prediction (c2).
- Tripod erosion: the mirror loss is necessary under the clock, as E5 showed, era B.
- A product-mode anneal has no precedent.
- The PPO counter resets on both arms.

**Rough cost.**
- Replay: 0 GPU-h, plus logging.
- One step 1.0 → 0.5 (2 arms × 2 seeds × 2000 its): about 3.6 GPU-h.
- Full anneal: about 10.8 GPU-h.
- Evals: about 1 GPU-h.
- Obstacle twin: about +4.5 +3.6 GPU-h.

**Adversarial review.** The verifier:
- showed that the stopped clock is the only stand-still income, which flips the stand prediction;
- found that the idle/excess penalties confound the stepping readout;
- found that the incumbent already fails the absolute tripod gate, so all guards must be relative to HOLD;
- noted the hexapod paper is replace-then-drop.

### swing-height-rebase — After the clock decision: one terrain-relative, gated clearance kernel, or none

**Kind & scope.** Fix-then-decide. Terms: reward_foot_clearance (1.0 in 2b-2c) and penalty_swing_min_clearance (−0.4 in 2b-2c).

**What changes.** After the clock decision, one of:
- (a) a single bounded terrain-relative clearance kernel that absorbs the micro-swing penalty;
- (b) drop both terms from 2b/2c.

Details:
- Branch (a) states its gate: the clock-speed gate or the gait-gate-decouple replacement. Never ungated, and off in the stand slot.
- Terrain lookup, preferred: a per-foot downward mesh raycast, as in eval _terrain_z_under_feet (~:444-455; 6 rays per env per step).
- Fallback lookup: the omni_exposure pixel index (:236). Never the drifting feet_edge convention.
- Both branches drop the root-relative GROUND_OFFSET −1.05 constant (cfg :76-86) and MIN_SWING_CLEARANCE_M.

**Why — repo evidence.** From a read-only replay of seed-3 A15+B rows:
- **The root-relative reference is noisy.** Terrain-relative stance pad height is 0.033 m (p10 0.026, p90 0.041). Root-relative it is 0.012 / 0.003 (median), p10 −0.032 / −0.045, p90 0.049 / 0.058. That is about 10 cm wide, larger than the 3-4 cm thresholds it gates.
- **A crouch pays.** The root-relative kernel correlates with base height at rho −0.30 to −0.42; the terrain-relative one at +0.46 to +0.48.
- **The floor is lower than it looks.** The settled root is 1.071-1.080 m, so the 0.03 floor means about 2-3 cm of real clearance.
- **Posture, not feet.** swing_min's root-relative flag rate went 0.105 → 0.148 per foot-step (1c → 2b), while its terrain-relative rate stayed at 0.050 → 0.052.
- **Apex collapsed inside 2b.**
  - Apex weight is a constant 0.5 through 2b (2c: 0.001). Yet income per unit weight fell from about 0.11 to 0.0003 (0.0564 → 0.00014).
  - Candidate causes: overshoot of the 0.08 m root-relative Gaussian (σ 0.03), swings leaving the mid-swing window, or reference drift.
  - foot_clearance pays linearly up to 0.26 m at any time in the swing, which makes it a prime suspect.
- **The earlier comparison was wrong.** 1c 0.119 vs 2b 0.116 m used the wrong control and a different metric. Lift fell at 2a (1c 0.057 / 0.084 vs 2a 0.017 / 0.060 m).
- **To reconcile:** the 2a apex weight is 1.0 in the dump vs 0.5 in the dossier. Likely static weight vs in-run ramp; confirm.
- **History.** On the golden plant, clearance terms added from scratch destroyed formation and were inert on a formed gait (gait_formation_v2 CHANGELOG.md:35-50). The clearance pack passed PLAN F only on its third round.
- **Current role.** 2b foot_clearance moved +0.058 → +0.063 (2.2%). In 2c it is the only lift shaper.

**Why — literature.**
- Terrain-relative references gate clearance by gait phase:
  - Lee 2020, Suppl. S4 Eq. 15: swing feet above the maximum of the terrain scan [25];
  - Kim, Lee & Park Sec. III-B: gated by the gait cycle, as a barrier constraint [19].
- Neither shows an ungated kernel sustaining lift without a gait prior. That weakens branch (a) after the clock goes.
- Spot foot_clearance_reward [6,7] uses world-frame z (flat terrain only), and a stationary foot gets full reward. Cited for the form only.
- Removal branch:
  - Zhuang [44] has no swing-height shaper;
  - Yang, Gao & Li [29]: without a gait prior, a simplified reward gives stumbling gaits. That is why this waits for the clock decision.

**Hypotheses.**
- H1 (offline): terrain-relative ranks episodes by lift, not by pose.
- H1b: the 2b apex collapse is overshoot. If so, dropping the pair restores apex income.
- H2: with the clock kept, dropping the pair leaves terrain-relative apex and ring outcomes unchanged. Without the clock, lift either persists (drop) or decays (then the kernel restores it).

**Falsifiers.**
- H1 fails if the terrain-relative kernel does not beat root-relative on Spearman vs apex or swings per metre, or still correlates negatively with base height.
- H2 (drop branch) fails, pooled over two seeds, if:
  - the terrain-relative apex median falls by more than a threshold set from the measured seed spread (not a fixed 2 cm);
  - ring outcomes breach their guard;
  - stumbles rise.

**How it would be decided.**
1. Step 1, now, offline: ring/light/slow npz and degenerate rollouts, in the Gate 1/2 pattern.
2. Step 1b: a like-for-like 1c → 2a → 2b comparison of pad-corrected terrain-relative apex, plus an overshoot-vs-desync test (under 1 h CPU).
3. Step 2, after the clock decision: a paired 2b fork from a 2a head, control vs drop. Add the rebased arm only if the drop fails. Extend through 2c, or state that 2c is left undecided.

**Dependencies & conflicts.**
- Training is blocked on the clock session. Tell that session now that apex income dies inside 2b and the pair is a suspect.
- gait-gate-decouple supplies the gate.
- Can share a screen arm with ring-bonus-drop.
- This kernel is also apex-term-retire's fallback.

**Risks.**
- Slow lift decay over 2c.
- Arming early harms formation (wave F).
- A wrong pixel convention misregisters by up to 0.8 m.
- Toe-drag on flat terrain.
- Evidence so far is seed 3 only.

**Rough cost.**
- Offline: 1-2 h, plus under 1 h for Step 1b.
- Screen: about 10 GPU-h, plus 2c length if included.
- +5 GPU-h for the rebased arm; +4.5 GPU-h if a clock-free 2a is needed.

**Adversarial review.** The verifier:
- found that apex collapsed under a constant 0.5 weight inside 2b;
- found that 1c was the wrong control;
- noted Spot's term is world-frame;
- noted the terrain-relative references are all phase-gated.

### track-single-kernel — One bounded linear-tracking kernel in place of exp + L1 (arm 1 only)

**Kind & scope.** Merge of track_lin_vel_xy_exp (weight 1.25, σ² 0.04) and penalty_tracking_error_l1 (weight −2.0).

**What changes.**
- One bounded kernel replaces the pair for omni: a re-dosed Gaussian or a Laplace exp(−‖e‖/s), optionally with a small linear tail as in humanoid-gym.
- The kernel is fitted offline.
- L1 and KRABBY_TRACK_L1_W are retired for omni.
- The exact fold (arm 2) was split off; see Dropped candidates.

**Why — repo evidence.**
- **Same error, two prices.** Both terms price e = cmd_xy − v_b,xy, ungated. Income: +0.676 (22.5%) and −0.233 (7.7%).
- **Revision 4 set them jointly.** σ² 0.04 and L1 −2.0 were chosen together so that income at v = 0 is negative (G3). Either change alone breaks that: +0.043 with L1 −1.5; +0.118 with σ² 0.06. The pairing was adopted over a failed walker-vs-creeper gate (+16.6% vs a +25% target) and is PROVISIONAL (crab_hex_omni_phases.py:196-197).
- **Pair numbers.**
  - Range (walk minus stand): 1.255 / 1.488 / 1.892 at cmd 0.21 / 0.25 / 0.35.
  - Slope: 6.87 / 7.36 / 5.28 / 3.02 / 2.06 at e = 0.1 / 0.14 / 0.25 / 0.35 / 0.5.
- **Candidate kernels.**
  - Laplace, w 2.2, s 0.2: range 1.43 / 1.57 / 1.82; standstill 0.77 / 0.63 / 0.38; slope 3.15 at e = 0.25, against the pair's 5.3.
  - Gaussian, w 1.8: range 1.20 / 1.42 / 1.72; slope 0.087 at e = 0.5.
  - The range can be matched analytically. The real cost is the lower escape slope.
- **Standstill level is not the problem.** A constant offset cannot change which velocity the policy prefers on a given row. A positive level only raises the value of surviving, which is neutral to helpful (Hwangbo S3).
- **Error tail unmeasured.** Mean ‖e‖ is about 0.12 m/s from L1 income; the tail is not known.
- **Evidence age.** All L1 evidence is old-plant. There is no omni ablation.

**Why — literature.**
- One kernel is standard: legged_gym and Isaac Lab [1,2,4], Rudin 2021 [22], CaT [17], HeLoM [31], Liu 2025 [30].
- Spot's base_linear_velocity_reward is a single Laplace kernel [6].
- humanoid-gym's track_vel_hard is exp(−10‖e‖) − 0.2‖e‖, placed beside a separate squared tracking term [51]. That is a precedent for stacking, and supports the "Laplace + linear tail" variant.
- Hwangbo S3 [24] and Ha et al. survey Sec. 3.3.1 [62] support bounded tracking.
- Counterpoint: literature kernels are tuned for ±1 m/s commands. This plant runs at 0.21-0.35 m/s.

**Hypothesis / falsifier.**
- Hypothesis: a kernel meeting all three criteria preserves the walk-vs-creep margins and formation:
  - range within about 10% of the pair's;
  - escape slope at e = cmd at least a stated fraction of the pair's;
  - a nonzero restoring slope out to the incumbent's p95 error.
- Falsified by any of:
  - a heavy error tail: then prefer Laplace + tail, or drop the arm;
  - no (w, s) meets the three criteria;
  - track_ratio_move_tail < 0.45;
  - overspeed (> 1.3× cmd) or reverse travel;
  - block T worse than about 1 × R_seed on the two-seed mean.

**How it would be decided.**
1. Zero-GPU pre-check: histogram ‖cmd_xy − v‖ per slot on the X08:c6 eval rollouts.
2. Replay: add a single-kernel column. The 1a/1b/1c raws come from a failed gate, so judge the kernel against the pair's own economy. The clock-zeroed column decides. Add the X08:c6 rollouts.
3. From-scratch C1d on seeds 3 and 2. Decisive: T, T_abs, move_turn.ratio, survival. Watch: stand drift.

**Dependencies & conflicts.** After the clock decision (the clock plus apex were 34% of the 1c creeper's positive income). After progress-relative-cap, M1b and survival-floor-audit Stage A.

**Risks.** Reopens L1's jobs: escaping standstill, pricing overspeed, pricing reverse travel. The range saturates at 0.35 m/s. Side-direction single-seed noise is high (R_seed 0.058).

**Rough cost.** Replay on CPU. 7.2 GPU-h for two seeds, +3.6 if a seed-2 B0 control is needed.

**Adversarial review.** The verifier showed that a cap on standstill income tests the wrong quantity, and that the real cost is the escape slope. It split off the no-op fold (arm 2) and added the error-tail pre-check.

### yaw-progress-phaseout — Zero yaw_progress after turning forms (conditional follow-up of one-yaw-kernel part B)

**Kind & scope.** Phase-out of reward_yaw_progress; a conditional follow-up.

**What changes.**
- Dispatch only if one-yaw-kernel part B shows that yaw_progress is needed during formation.
- Weight 0 (not ε) from the first stage that draws move_turn: c4 in C1d, or the first move_turn stage of the preset ladder.
- t3 keeps 1.0, so the name, config entry and knob remain.
- Preferred trigger, faithful to Rudin 2022: zero the weight once the training yaw_progress metric on turn rows (or turn facing_closed) reaches a set fraction of its t3-end value. A trigger at a fixed stage boundary is the fallback.

**Why — repo evidence.**
- **Only positive dose-response.** D10 at weight 3: turn facing_closed 0.419 → 0.748 and yaw_progress 0.054 → 0.097; move_turn facing 0.050 → 0.147, but move_turn survival 0.58 → 0.12.
- **The command, not this term, drives move_turn deaths.**
  - C3a (cap 0.21 + CLOCK_TURN_R 1.72) raised move_turn survival to 0.98 / 0.995.
  - X08 raised it 0.58 → 0.75 with the term at 1.0 (D12).
  - X09 went 0.75 → 0.90 and was still a TIE (R ≈ 0.105-0.11).
- **Income at weight 1.0:** 0.0039 / 0.0052 / 0.0041 / 0.0044 per second (t3 / c4 / c5 / c6).
- **It also holds heading on move rows.** On MOVE rows (0.60 / 0.60 / 0.55 of slots in c4-c6), the gate opens once facing drifts beyond 0.133 rad, and the term pays for turning back (crab_hex_omni_rewards.py:393-401). This is the only signed heading-hold gradient in the reward.
- **Branch points exist:** X08 t3 model_2999 and c4 model_4498.

**Why — literature.**
- Rudin 2022, Sec. II-B3 [23]: r_bias is removed once r_task reaches 50%, i.e. a performance-gated removal.
- Ng et al. [47] and Grzes [48]: the implemented term is clipped, dead-band gated and in γ = 1 rate form, so it is only approximately potential-based.
- Jeon, Heim, Khazoom & Kim [49]: potential-based shaping gives marginal convergence benefit but is much more robust to scaling. The D10 sensitivity shows this term departs from potential-based behaviour on move_turn rows, where the gate never closes.
- IsaacGymEnvs Ant [50] uses the exact potential-difference form.

**Hypothesis / falsifier.**
- Hypothesis (non-inferiority): turn facing_closed, net_yaw, F_mt and move-row heading hold all TIE X08. Survival is a guard only; the single-seed survival MDE is about 0.11, so no positive survival effect can be shown.
- Falsified if turn facing_closed is WORSE (C1 and C1b lost turning, 0.42 → 0.19 / 0.17), heading or stand drift worsens, or F / F_mt fall.

**How it would be decided.**
- Replay first: break down the term's income by slot at weight 1 on the X08:c6 suite. Proceed only if a material share is paid on move_turn rows.
- Training: branch at X08:t3 and rerun c4 → c6 (5000 its) with the term at 0, against a weight-1.0 re-branch from t3 as the control.
- Seed 2 needs a seed-2 t3.

**Dependencies & conflicts.** one-yaw-kernel part B; C3a / C12; clock; M1b.

**Risks.** Turning may be forgotten. Removes the only signed heading-hold gradient. Adds a per-stage schedule.

**Rough cost.** About 10 GPU-h including the re-branch control.

**Adversarial review.** The verifier found that the name cannot be deleted (t3 still uses it), that the survival effect was only ever seen at weight 3.0, and that the term carries heading hold on move rows. The candidate was recast as a conditional non-inferiority follow-up.

### mode-time-after-clock — After the clock goes: one state-based swing-timing term

**Kind & scope.** Replace penalty_foot_idle_when_forward and penalty_excess_feet_contact_forward, and act as the air-time successor. Applies only if the clock session removes the clock.

**What changes.**

**Arm B (preferred; not affected by the float32 bug).** A per-foot mode-time reward, optionally with a cross-foot variance penalty, reading the accumulated air/contact times. It departs from Spot's version as follows:
- Ceiling taken from the measured mode lengths of the clock-removed control, and speed-scaled (e.g. k·stroke/|cam speed cmd|), or given a soft roll-off instead of a hard cliff.
- t_max = max(air, contact) also bounds stance.
- Spot's 0.5 s variance clip is re-scaled or removed.
- The variance term is limited to straight-line categories, or run as its own sub-arm.
- The contact threshold is stated explicitly (1.0 N sensor vs 0.1 N).
- foot_idle stays as a backstop unless the replay shows arm B's drag/skate margin under the zero clip is at least foot_idle's.

**Arm A.** A timing-fixed feet_air_time at a replay-gated weight. Needs the float32 fix. It gives no prior in turn or stand (gate ||cmd_xy|| > 0.1), so a turn-gate variant is needed if it is kept.

excess_feet is folded in only if the new term also prices shuffling.

**Why — repo evidence.**
- Today the clock carries the timing prior (schedule 10.3%, apex 4.4%). After it goes, the remaining step pressure is thin.
- Why arm B avoids the bug: is_first_contact = (current_air_time > 0) · is_contact, and elapsed time accumulates regardless (contact_sensor.py:420-443). current_* and last_* never pass through compute_first_contact's dt + 1e-8 test.
- The measured gait is sub-0.4 s, not the 0.6-1.8 s cam-cycle regime:
  - fwd ~28 swings/m, side ~69, turn ~63-79 swings/rad (motion REPORT.md:210-235);
  - exact-timing air time pays only 1.2-1.6% (REPLAY_GATE.md:169-173).
- Arm A fails Gate 1 on side/back/diag (:91-95). Its per-second rho of −0.09 to −0.40 is mostly a shared-denominator effect (:104).
- foot_idle's floor is an unbounded penalty (−1799/min on drag). Arm B only withholds bounded income.
- Era-A stride_definancing ("income follows duty"): test ranking, not share.

**Why — literature.**
- Spot [6,7]:
  - air_time_reward and air_time_variance_penalty: t_max vs mode_time, variance clip at max 0.5;
  - weights: air_time 5.0 (mode_time 0.3), variance −1.0;
  - Spot also runs a GaitReward with synced pairs, so its timing prior is never mode-time alone.
- A minimum-air-time floor (arm A's form) is supported by:
  - Chane-Sane et al. 2024 [33]: removing the air-time constraint "severely impairs" walking (Table 3);
  - van Marum Sec. IV-B [34]: steps too frequent without positive air time.
- Hexapod priors: Liu 2025 [30], Yu 2018 [32].
- Li et al. [35] swap reward terms between training stages.
- The standard stacks keep feet_air_time (legged_gym 1.0 [2]; Isaac Lab 0.125 [4]).
- There is no hexapod or cam-driven precedent for mode-time.

**Hypothesis / falsifier.**
- Hypothesis: one bounded timing term keeps all six legs in use and holds timing at least as well as foot_idle + excess_feet, improves the stand regime, and (arm B) is not exposed to the timing bug.
- Falsified offline if the term:
  - fails to rank healthy above parked, skate, drag or creeper rollouts, or pays the creeper;
  - lets the variance term rank C1d turning below the creeper;
  - shows an income cliff at low commanded speed.
- Falsified in training if the term is WORSE than the stock foot_idle/excess arm on T, survival, move_turn.ratio, tippy-tap, contact_switch_hz or leg parking, beyond pre-registered margins (e.g. 1 × R_seed).

**How it would be decided.**
- Offline:
  - price both arms on X08:c6, 1a/1b/1c and the forward degenerate fixtures, post-clip;
  - measure the clip-binding frequency with and without the new term;
  - compare arm B's drag/skate/parked margin with foot_idle's;
  - sweep the ceiling from 0.2 to 2 s, plus speed-scaled variants.
- Training: from-scratch C1d, clock removed + stock terms vs clock removed + the new term. Three seeds, or two plus a stopping rule. Reuse the clock session's run as the control only if commit, knobs, schedule and eval manifest are identical.

**Dependencies & conflicts.**
- Hard: the clock is removed and a clock-free recipe exists; arm A needs the float32 fix.
- Soft: dead-swing-income, foot-idle-audit, M1b, gait-gate-decouple.
- Do not stack with clockless-tripod-prior unless tripod loss is observed.

**Risks.**
- The cam largely dictates swing duration.
- A speed-independent ceiling fights low-speed and turn tracking.
- The variance term fights differential turning (the C1d turning gain 0.42 → 0.62).
- A new positive income shifts the zero clip for every penalty.
- The contact definition changes (0.1 N → 1.0 N).
- A planted-feet branch interacts with stand facing corrections.
- Void if the clock stays.

**Rough cost.** Offline about 0. Training 7.2 GPU-h with a reused control, 14.4 GPU-h otherwise.

**Adversarial review.** The verifier confirmed arm B's bug immunity. It showed a fixed ceiling and the variance term would fight low-speed tracking and differential turning, that the literature supports an air-time floor rather than mode-time, and that arm B only withholds income where foot_idle penalises.

### rod-power-after-clock — Rod positive work as the contingency for delta-torque-drop

**Kind & scope.** A merge used only as a contingency. Terms: penalty_mechanical_power (currently 0), reward_delta_torques, penalty_joint_vel_l2 (knob).

**What changes.**
- One concrete term: Σ max(τ·q̇, 0) over the 12 rod joints, the same grouping metrics.py:1507-1512 already uses.
- Lead-screw F·v is parked until a linkage force model exists.
- Enabled only if a Q regression (tapping or chatter) appears in the clock-free chain after delta_torques is dropped.
- Dropping the weight-0 power and joint_vel terms is zero-weight-purge's job, not this candidate's.
- Separate small fix: penalty_mechanical_power is scoped to rods and positive work (today it sums all 24 joints; rewards.py:1132-1144).
- Keeps config-truth-align's sign guard.
- M2 is re-specified as rod positive work.

**Why — repo evidence.**
- Power split: rods 69-76%, cams 0%, Body_Hip emulator 24-31% (REPLAY_GATE.md:244-254).
- The all-joint form fails Gate 1 in 7/7 categories (rho 0.11-0.37). Rod-only joint_vel still fails 5/7 (:278), so scoping to rods alone is not expected to fix the ranking.
- Rods are PD springs whose limits act through slew rate (crab_hex_scene_cfg.py:120-129). τ·q̇ is therefore an estimated load times slew-capped motion, not controlled effort.
- Earlier record (pre-rebuild):
  - C5a from scratch failed (reward 11.4 vs 21.7). The policy shortened its steps to save power (−13%, 1096 W) and tippy-tap rose to 41.5% (task1_velocity CHANGELOG.md:84-86).
  - B2 on an already-formed walker was the best walker (tripod 0.621).
  - C3/C4 suppressed lift.
  - L3 had the wrong sign.
- Block Q's work_per_m/rad/s is exactly this quantity (metrics.py:1462-1472, 1542-1561). Q results would be a manipulation check, not evidence.
- The stand chatter defect (1.653 Hz) comes from the bang-bang mean, which an energy term does not address.

**Why — literature.**
- Fu et al. [39]: energy as the single regulariser (α1 0.04). But they train one policy per fixed speed with a minimal reward, and multi-velocity training collapsed.
- Tan et al. [40], Eq. 2: the form only.
- Kim et al. [18]: their effort term is torque, and their small reward set relies on 11 constraints.

**Hypothesis / falsifier.**
- (a) Offline: rod positive work ranks contact_switch_hz at rho ≥ 0.5 in every category. Killed before any training if it passes in ≤ 2/7.
- (b) After the clock goes: added in the last stage of a formed walker, it improves the non-work Q metrics without shortening steps or reducing lift.
- Kill guards for (b):
  - swings_per_m or contact_switch_hz rises;
  - the robot stands still at low speed;
  - 2b clearance income falls;
  - diag/side survival or T falls beyond R_seed.

**How it would be decided.**
- Step 0: now, on CPU, without code, using rod_work_pos_j on the X08:c6 npz.
- Step 1: only if step 0 passes and residual tapping appears. Late-stage arms (optionally from scratch), 2 seeds each. Decide on F/F_mt/T/survival/stand and the non-work Q metrics.
- Step 2: a 2b lift check.

**Dependencies & conflicts.** Step 0 is independent. Step 1 comes after the clock and after delta-torque-drop. Needs M1b and the sign guard.

**Risks.** Hurt formation and lift in earlier runs. Multi-command collapse. Invisible cam motor draw. Needs a linkage model for F·v.

**Rough cost.** Step 0: about 1 CPU-h. Step 1: about 2 GPU-h per late-stage arm-seed (7-8 GPU-h from scratch). Step 2: about 2 GPU-h.

**Adversarial review.** The verifier found that the block Q work metrics are the term itself, that the only from-scratch energy screen produced short-step tapping, and that step 0 is free. The candidate was reframed as delta-torque-drop's contingency.

### caps-smoothness-loss — Move action smoothness into the PPO loss (a CAPS temporal term), as an alternative to rate-follows-bounds

**Kind & scope.** Structural. Terms: reward_action_rate, penalty_action_second_diff, penalty_joint_vel_l2.

**What changes.**
- A CAPS temporal term L_T = ‖mu(s_t) − mu(s_{t+1})‖₂. It is unsquared, as in CAPS; a squared form would be a declared deviation. It sits behind a knob.
- Next-step pairing needs new storage code (rollout_storage.py:222-257; ppo_with_extractor.py:290-341, 428-452):
  - pair index idx with idx+N in the flattened [T,N] layout;
  - mask dones, timeouts and the last step;
  - use the non-mirrored half only;
  - reuse the rollout's hist_encoding.
- Run after M1b. M1b's bound loss handles |mu|; CAPS then only handles smoothness.
- Retire second_diff and joint_vel only after the arm B vs arm A result.

**Why — repo evidence.**
- The reward form fails the Q-normalised Gate 1 in 5/7 categories and contact_switch_hz in 7/7 (REPLAY_GATE.md:84-90, 104). It costs 14.5-18.3% of task income.
- The combined actor loss holds only priv_reg and mirror. hist_latent is DAgger-only, and the estimator is back-propagated separately. Mirror is an MSE on the same act_inference mean, so it is a direct neighbour of this term.
- Scale changes a lot as |mu| desaturates: the raw-mean L_T is about 80× larger on the saturated head than near |mu| ≈ 1.
  - 42% of Σda² comes from full ±1 flips.
  - Median |mu| is 7.4-9.5; flip rates are 2.23 (stand) and 1.83 (moving) per dimension per second.
  - A mean held constant at ±9 costs 0, so the term does not pull |mu| back inside the clip.
- The 1c creeper was partly paid for by action_rate income improving while the robot stopped (−0.407 → −0.220). Wave L found that "churn is load-bearing".

**Why — literature.**
- CAPS [42], Sec. III, Eq. 1-3: the binary-response evidence is for TD3, and PPO is fairly smooth on its own. In live flights (Table II) current dropped from 22.87 to 4.86 A, but tracking MAE was 9.28 vs 5.19.
- LCP [43], Table I: return with no smoothing 28.87; smoothness reward 26.56; LCP 26.03; LPF 24.98. The λ window is narrow: 0.005 → 23.92, 0.01 → 16.11.
- Seyde [41], Sec. 5.4.2: smoothness penalties cost performance.
- None of these is a hexapod.

**Hypothesis / falsifier.**
- Hypothesis: CAPS matches the reward arm on non-circular Q and matches the rate-0 control on T/F.
- Falsified if:
  - CAPS is no better than rate-0 on Q, meaning the loss is inert and plain removal is the answer;
  - it is worse than the reward arm beyond the MDE;
  - stepping collapses during formation.

**How it would be decided.**
- Offline: size λ by the gradient-norm ratio ‖∇θ λL_T‖ / ‖∇θ L_surr‖ on X08:c6 and M1b batches.
- Training: 3 arms × 2 seeds × 2000 its from one fork head:
  - (A) rate 0, no CAPS;
  - (B) rate 0 + CAPS;
  - (C) the rate-follows-bounds winner.
- Confirm from scratch only if B beats A on Q and ties C on T/F.

**Dependencies & conflicts.** Comes after M1b, one loss change at a time. It is an alternative to rate-follows-bounds. Interacts with the mirror loss and with the priv_reg reset on resume.

**Risks.**
- It changes a PPO loss shared by all Krabby tasks.
- New hyperparameters.
- Scale change as |mu| desaturates, and a narrow λ window.
- Taxes rod reversals the gait needs, risking the creeper basin.

**Rough cost.** About 5.4 GPU-h, plus 7.2 GPU-h conditional; about 12.6 GPU-h total, plus code.

**Adversarial review.** The verifier found that no source shows a task gain from moving smoothness into the loss. It added the missing rate-0 control, fixed the λ sizing yardstick and the CAPS distance, and flagged the scale change.

### clockless-tripod-prior — After the clock goes, and only on observed tripod loss: a dense, hexapod-adapted pair-sync term

**Kind & scope.** Replace, conditionally. It takes over the phasing role of reward_clock_schedule and succeeds the dead reward_tripod_schedule. It reads the same air/contact timers as feet_air_time.

**What changes.**

Trigger: only if a no-clock arm, or clock-hold-twin's anneal, loses per-category tripod.

One gait-pattern term, adapted from Spot GaitReward:
- **(a)** Combine factors by mean or geometric mean. A product of 15 factors (6 sync + 9 async) has a floor of about 0.45^15 ≈ 6e-6, against about 0.008 for Spot.
- **(b)** Set max_err relative to the ~2 s cycle (e.g. 0.3-0.5 s), or normalise errors by the commanded period.
- **(c)** Either state the duty-0.5 assumption, or use a duty-agnostic form (touchdown phase lag, or "~3 feet in contact").
- **(d)** Couple to progress, or state that stepping in place is priced elsewhere.

Other points:
- Gate: gait-gate-decouple's predicate. Spot's 0.5 m/s threshold is above the omni cap and misses turns.
- Sibling arm to replay alongside it: a CaT-style |n_contact − 3| prior.
- No float32 dependency: last_air_time and last_contact_time latch on every transition (contact_sensor.py:420-443).

**Why — repo evidence.**
- Without the clock, nothing specifies leg alternation.
- tripod_schedule (era A):
  - Dead from scratch.
  - Live when fine-tuned from a tripod, but gamed: crossing income tripled through timing sharpening while tripod went 0.401 → 0.400 and duty/coherence did not change (tripod_stability RESULTS.md ~386-420). This is the main risk precedent for a timing-only term.
- E5 (era B, with the clock): turning the mirror loss off halves the tripod (0.257 vs 0.541). That shows the mirror loss is necessary under the clock, not that it is sufficient without one.
- E1 seeds 2-4: a 1-in-4 basin rate even with the clock and RSI.
- The E2 clamp arm "danced in place".
- cat_tripod_min is 0.396, a guard the user overruled.
- Era-A duty asymmetry (0.146 / 0.556) shows that anti-sync at duty 0.5 prescribes a gait.

**Why — literature.**
- Spot GaitReward [6,7]: a product of clipped exp factors; weight 10.0, std 0.1, max_err 0.2, velocity threshold 0.5.
- CaT Table I [17]: at most 2 contacts on Solo-12. The tripod equivalent, "3 contacts", is a cheap alternative.
- Liu 2025 [30]: an AMP-style adversarial prior that needs a dataset.
- Fu [39]: gaits emerge from energy alone.
- No hexapod demonstration of this form was found.

**Hypothesis / falsifier.**
- Hypothesis: the adapted score ranks tripod > creep/shuffle > six-down, gives gradient from creep, and holds tripod and duty with a smaller shaping share than the clock takes today.
- Falsified if any of these holds:
  - The ranking fails, or income sits near the floor for creep, which would refute the "dense" premise offline.
  - Stepping in place pays as much as walking.
  - Income rises through timing sharpening with no tripod or duty change (the v6 pattern).
  - No gain on cat_tripod_min over 2 seeds.

**How it would be decided.**
- Replay, 0 GPU-h. Rollouts: healthy, creep, stand-stepping, shuffle, dance-in-place, turn-in-place, and the M1 heads. Price both forms.
- On trigger: fine-tunes from a tripod head, which tests retention; 2 seeds.
- Formation claims need at least 4 seeds, or RSI.

**Dependencies & conflicts.** Triggered by the clock session or clock-hold-twin. Uses gait-gate-decouple. The stand branch goes to the stand owner. Do not change it in the same screen as a timing fix.

**Risks.**
- Prescribes duty 0.5.
- Sparsity from the product floor.
- Gate mismatch.
- Pays stepping in place.
- The basin lottery.
- Adds a term; it only simplifies if it replaces the clock plumbing.

**Rough cost.** Replay 0 GPU-h. Forks about 3.6 GPU-h. Formation runs need at least 4 seeds.

**Adversarial review.** The verifier removed a false float32 dependency, showed that a 15-factor product is as sparse as the dead event credit, and cited the repo's own v6 result as the main risk.

### own-tripod-discriminator — Parked: one discriminator style term, only if both the clockless head and the minimal arm lose the tripod

**Kind & scope.** Fix-then-decide, parked.
- Live gait terms it would replace: foot_idle, excess_contact, tibia.
- The clock and apex terms go through the clock session.
- New term: reward_style_amp.

**What changes.**

Entry condition: enter only if BOTH hold:
- the clock session's clockless head loses per-category tripod beyond R_seed, and
- minimal-reward-arm also loses it (cat_tripod_min, diag/turn especially).

Style term:
- r_style = clamp(1 − 0.25(D(s,s') − 1)², 0), with an LSGAN discriminator and a gradient penalty.
- D inputs: actuated joint pos/vel, base lin/ang velocity, height, and foot heights in the base frame.
- Excluded from D inputs: passive linkage DOFs (the npz joint_pos is (T,24), while the action is 18-dim) and clock/rod signatures.

Scope and safeguards:
- Walking categories only; stand is left to the task and regulariser terms.
- Log style income per category.
- Pre-register a rule for switching to per-category discriminators.

Sources:
- the raw npz of a clean head (see Gate 0), or
- the repo's synthetic IDEAL tripod trace (lit-review-hexapod-reward-stability.md §4, rec. 3). This removes the M1b dependency; note its gap in turn coverage.
- Ask the clock session to archive the suite npz of its last clean clocked head.

**Why — repo evidence.**
- Payoff: only 3 live terms go (foot_idle −0.236, tibia −0.042, excess −0.017). Air time and stride are free removals elsewhere.
- Cost: system complexity rises: a network, optimizer, loader, buffer, gradient penalty, normaliser, hyperparameters, (s,s') observations, and a dataset to re-cut on every plant or command change.
- What exists:
  - The raw npz carries every needed field, steady_mask included (eval_crab_hex_gait.py:666-686).
  - Each hold has about 1,050-1,190 s of steady frames per 100-episode suite.
- What is missing: rsl_rl_lib 5.0.1 has no AMP. The only auxiliary reward hook is RND (ppo_with_extractor.py:75-81, 204-221).
- The incumbent is unusable as a source (ANALYSIS_M1_saturation.md:36-38):
  - median |mu| 8.1, with 96% of |mu| > 1;
  - stand six-down 0.004;
  - diag tripod 0.396;
  - move_turn survival 0.75.
- Prior proposal: the forward lit review already proposed AMP at rank 3 (lit-review-hexapod-reward-stability.md:146-175, 325-328).
- foot_idle's 90-step window is a stall guard (parkour_mdp_cfg.py:1028-1044).

**Why — literature.**
- Li et al. [35], Table I and Sec. III: a style reward of 1.0, trained on 9.6 s of the robot's own stage-I tripod. Stage II ran 15,000 its against 5,000 for stage I. Sim only. Fig. 3: a high style weight slows learning.
- Liu et al. [30], Table I: task + style + regularisers, trained on 8.6 s of synthetic trajectory-optimisation data (the same group as Li et al.).
- Escontrela et al. [57]: 13 hand terms replaced by 4.5 s of mocap. AMP_for_hardware config: coef 2.0, lerp 0.3, grad penalty 10.
- Vollenweider et al. [58], Sec. II: a single discriminator struggles with hybrid gaits.
- Huang, Xie & Li [59], Sec. III-C: mode collapse.

**Hypothesis / falsifier.**
- Hypothesis: from a clean source, one style term keeps the tripod in the walking categories without the clock or the hand-built gait family, with T/F within noise.
- Offline falsifiers:
  - D trained on head A fails to score a held-out clean head B high in every category;
  - D keys on clock or passive-joint features.
- Online falsifiers:
  - tripod WORSE beyond R_seed;
  - mode collapse (turn becomes a shuffle, diag alternates, stand steps);
  - style income falls to 0 while D accuracy goes to 1;
  - T/F or survival WORSE.

**How it would be decided.**
- **Gate 0 (CPU):** source eligibility.
  - median pre-clip |mu| ≤ ~1.1 per category;
  - tripod ≥ 0.45 including diag;
  - a numeric stand threshold, to be set.
  - X08:c6 fails this gate, so the candidate stays parked.
- **Gate 1 (CPU):** the generalisation checks above.
- **Gate 2:** a from-scratch flat 1a→1c chain, style on vs no gait terms, two seeds. A fork is only a smoke check: it tests retention, and a tie there must not decline the candidate.
- **Gate 3:** a full from-scratch chain.

**Dependencies & conflicts.** Strictly downstream of the clockless head and minimal-reward-arm. Needs M1b unless the synthetic trace is used. Needs the clock-session archive. Needs about 1-2 days of code.

**Risks.**
- Bakes in the source head's defects.
- Mode collapse.
- The discriminator overpowers the task reward.
- Runs against the simplification goal.
- Thin hexapod evidence.
- Shallow turn coverage (steady |wz| ~0.165 vs a 0.3 cap).

**Rough cost.**
- Code: 1-2 days.
- Gates 0-1: 1-2 h of CPU.
- Gate 2: not yet estimated; it replaces the ~4-4.5 GPU-h fork estimate.
- Gate 3: about 7-8 GPU-h.

**Adversarial review.** The verifier found four problems:
- the fork gate tests retention, not formation;
- the degenerate-separation check is nearly tautological;
- the repo's synthetic-trace proposal removes the M1b dependency;
- the benefit is 3 live terms for a large increase in machinery.

The candidate is ranked below every simplification that only changes the reward table.

### continuous-heading-servo — Drop the yaw dead band so the rate kernel prices small heading errors

**Kind & scope.** Simplify, on the command side. It changes the target of track_ang_vel_z_exp and the gates of reward_yaw_progress and track_ang_vel_z_exp_turn_gated.

**What changes.**
- Set ang_vel_clip to 0 on the per-step path, so cmd_wz = clip(1.5·wrap(e), ±0.3) on every row.
- For the stand slot, either:
  - (a) arm KRABBY_OMNI_STAND_HOLD, which zeroes wz on STAND in the way Isaac Lab zeroes standing envs, or
  - (b) state that stand rows receive a drift-correcting wz.
- Re-gate yaw_progress and the turn-gated lever on the slot (_slot_is_turning; crab_hex_omni_rewards.py:430-459). Otherwise yaw_progress becomes a dense 1.0-weight heading hold on every row.
- Relax the parser of KRABBY_OMNI_ANG_VEL_CLIP so it accepts 0.
- M5 hysteresis becomes moot.

**Why — repo evidence.**
- The dead band is 0.133 rad. At cap 0.3, the linear band is only 0.133-0.2 rad. (D11's "near bang-bang" referred to C3a's 0.21 cap.)
- Inside the dead band, only facing_exp prices the error (~0.025/s). The ceiling on facing_closed is about 0.915, and translation drift below 7.6° is unpriced.
- Implicit thresholds already exist, so the clock coupling is small:
  - the gait gates open at |e| > 0.067 rad (0.12/1.2/1.5);
  - the clock stops below 0.2 m/s, i.e. |e| < 0.111 rad (crab_hex_dimensions.py:197).
- The approach phase is unchanged, because it is saturated. The flat std-0.5 kernel (docstring :38-53) is why yaw_progress exists.
- The stand slot has psi = yaw (command.py:561). STAND_HOLD is keyed on slot_id (:457-468, 712-716).
- The dead band is applied per step (uniform_parkour_command.py:100-106).
- The plant turns at about 0.08 rad/s, and a bang-bang mean may not produce small rates.

**Why — literature.**
- Isaac Lab applies clip(stiffness·error) with no threshold and zeroes standing envs [10].
- legged_gym uses clip(0.5·wrap(...), −1, 1) [1].
- Provenance: Extreme Parkour zeroes commands[:,2] below ang_vel_clip at resample only [12]. The repo's port applies it every step, which is what creates the relay.

**Hypothesis / falsifier.**
- Hypothesis: the kernel prices hold and settle below 0.133 rad, and facing_closed and move-row heading_final improve or hold.
- Falsified if stand drift, foot_idle activity or stand stepping rises; if F or T is WORSE; if the observation shift destabilises a fork; or if turn holds show limit cycling.

**How it would be decided.**
1. Replay the kernel and yaw_progress under both gate choices.
2. Fork runs gated on M1b.
3. Primary metrics: settle_s, heading_final and stand drift, with 3 or more seeds, or pooled. With 2 seeds, the ≤0.085 ceiling gain cannot be resolved against an MDE of ~0.15.
4. Then a from-scratch run.

**Dependencies & conflicts.** The clock is a soft dependency. STAND_HOLD is enabling, not conflicting. It conflicts with M5 hysteresis. It is best placed after yaw-progress-phaseout. M1b is a confound.

**Risks.** Loses the exact-zero stand command (mitigated by STAND_HOLD). Shifts the cmd_wz distribution. Interacts with the obs clip. Affects every cmd_wz consumer.

**Rough cost.** Replay under 1 h, plus about 1 h for the variant. Forks 2 × 0.9 GPU-h. From scratch 7-11 GPU-h.

**Adversarial review.** The verifier corrected the D11 quote and showed the clock already acts as a ~0.111 rad band, which makes the clock dependency soft. It recast STAND_HOLD as enabling and required yaw_progress to be re-gated on the slot.

### cat-style-constraints — CaT stochastic terminations for style penalties, only if the plain removals fail

**Kind & scope.** Structural and conditional. Set S covers tibia and excess_contact, plus collision/stumble in the teacher phases. action_rate is excluded.

**What changes.**
- Runs only if minimal-reward-arm and tibia-pose-retire REJECT, i.e. the terms turn out to be load-bearing. Otherwise the candidate is moot.
- δ is applied in the PPO rollout storage and in GAE:
  - rewards *= (1 − δ);
  - δ is merged into the float done used by GAE only;
  - the real dones still drive policy.reset and env resets.
- Ordering of the timeout bootstrap (ppo_with_extractor.py:226-228, :234, :251) must be decided explicitly.
- Evals never see δ.
- Each member of Set S needs a physical threshold on the linkage plant (e.g. a knee band around mid-stroke, n_loaded > 4). A member without one stays a penalty or is dropped.
- action_rate is excluded because it fails replay separation (REPLAY_GATE.md:104).
- Mechanics: EMA max normalisation, p_max ramped 0.05 → 0.25.

**Why — repo evidence.**
- The zero clip applies to the step total (parkour_reward_manager.py:38). A penalty is forgiven only on steps whose net would be negative; the per-slot clip-hit rate comes from survival-floor-audit.
- On a step already clipped to 0, r(1 − δ) changes nothing now. The lever is δ in the done signal.
- crab_failure is already a CaT-style hard constraint.
- Income of the style terms is small (1.4%, 0.6%). Swapping them for new machinery is not a simplification unless they are load-bearing.
- Tibia has no principled limit, because the default knee sits at mid-stroke.
- Survival already binds: C2 move_turn fell 0.58 → 0.12, and the failure fraction is 0.364.

**Why — literature.**
- CaT [17], Algorithm 1: "not environment resets, but merely future reward terminations". Hard constraints use p = 1; soft ones ramp 0.05 → 0.25. Its limits are all physical, and it assumes r ≥ 0.
- N-P3O [26], Table II: 0.49 violations vs 533.
- Kim et al. [18]: 3 rewards plus 11 constraints.
- Hwangbo [24]: early constraint pressure produces a stand-still tendency.

**Hypothesis / falsifier.**
- Hypothesis: at matched T/F/F_mt, violations of Set S drop with no survival loss, and one p_max schedule replaces 2-3 weights.
- Falsified if a survival guard or a pre-registered stand-still guard is WORSE; if violation counts are not lower; or if offline violation rates do not separate the REPLAY_GATE healthy and degenerate populations on stochastic rollouts.

**How it would be decided.**
1. Offline: implied δ rates at p_max 0.05 and 0.25.
2. A 2000-it fork against a same-length X0 fork, seeds 3 and 2, with the MDE stated.
3. From scratch only if the fork wins.

**Dependencies & conflicts.** Needs REJECTs from the drop candidates. Needs survival-floor-audit, M1b and the clock outcome. contact-telemetry-audit informs the teacher set. Plumbing must reach both the omni flat and teacher cfgs (crab_hex_omni_env_cfg.py:185-200).

**Risks.** Value pessimism leading to stand-still. Done/timeout ordering bugs. Coupling with recurrent resets. Drift in EMA strength. Resume resets the ramp.

**Rough cost.** Offline about 1 CPU-h, plus about half a day for the algorithm change. Forks about 1.8-2.5 GPU-h. From scratch 7.2 GPU-h.

**Adversarial review.** The verifier corrected the mechanism: CaT scales rewards and dones, and does not reset envs or cut samples. It excluded action_rate and made the candidate conditional on the plain drops failing.

### soft-pitch-constraint — A soft tilt band below the hard 0.5 rad, only if posture-penalty-drop REJECTs

**Kind & scope.** Conditional addition. It introduces a new training-only soft pitch/tilt constraint.

**What changes.**
- A tilt band implemented as a CaT constraint (same mechanism as cat-style-constraints).
- It ramps from well above healthy walking tilt up to the existing hard limit of 0.5 rad.
- Its evidence must be re-derived on in-distribution X08:c6 falls.

**Why — repo evidence.**
- ANALYSIS_1c_failure describes a lunge phase-locked to the 0.5 Hz tripod, with this timeline:
  - pre-onset pitch_pk grows 0.021 → 0.047 → 0.132 over the last ~1.5 s;
  - tilt above 0.1 rad reaches termination in 0.54 s;
  - 0.25 → 0.5 rad takes 0.30 s;
  - |roll| > |pitch| in 0 of 100 falls.
- Those falls come from a canary that the analysis itself classifies as an eval defect: an unclipped 1.57-rad facing error, against ±0.6 in training (:8). It had 11 survivors, so it is motivation only.
- In-distribution ladder canaries attribute 0.78-0.88 of falls to pitching forward (REPORT.md:42,85).
- The tilt² term cannot act in time (~0.0001 per step).
- The hard termination already gives the critic a gradient before 0.5 rad. A soft band adds lead time only if it starts before states the critic already rates as doomed.

**Why — literature.** CaT treats base orientation as a flat-only constraint at 0.1 rad [17]. Kim et al. use a COM-frame posture constraint [18]. Hwangbo et al. ramp constraints in [24].

**Hypothesis / falsifier.**
- Hypothesis: pricing from about 0.15 rad gives 0.3-0.4 s of extra lead and lowers the diag/back/move_turn fall hazard.
- Falsified if survivors often cross the threshold; if fallers cross less than ~0.2 s before the limit; if V(s) at the threshold already predicts the fall; or if a survival guard is WORSE.

**How it would be decided.**
1. Threshold scan on in-distribution falls only: X08:c6 suite 2026-09-23_05-43-42 plus the slow canary with facing_obs_clip 0.6.
2. Critic check.
3. Fork as in cat-style-constraints.

**Dependencies & conflicts.** Requires a posture-penalty-drop REJECT. Also depends on survival-floor-audit, M1b (bang-bang may drive the lunge) and the clock (the fall mode is phase-locked to the clock-driven tripod).

**Risks.** Survival collapse on a plant that is already limited by falls. Value pessimism.

**Rough cost.** Offline about 1 CPU-h. Forks about 1.8-2.5 GPU-h.

**Adversarial review.** Split out of cat-soft-constraints because it adds shaping. Its motivating falls were an eval defect.

### orientation-unmask — Whether to remove orientation's Extreme Parkour tile mask, only if orientation is kept

**Kind & scope.** Conditional fix-then-decide on the tile mask of reward_orientation.

**What changes.** Only if posture-penalty-drop REJECTs. Decide whether to remove the mask (rewards.py:386-396). Today the mask hard-zeroes the term on 20% of envs in 1a-1c and on 50% in 2a-2c (FLAT_FRAC 0.8/0.5; omni_ring.py:17-19).

**Why — repo evidence.**
- The mask came from the upstream port (^7283a7bb), which targeted tall obstacles.
- Ring blocks are 2-9 cm under a body about 1.0-1.1 m tall, so the mask has no omni rationale.
- Unmasking raises penalty mass and changes behaviour on ring tiles. The policy observes the tile flag (observations.py:180-184, 212-213).

**Why — literature.** Extreme Parkour's class-17 rule [12].

**Hypothesis / falsifier.**
- Hypothesis: unmasking is no worse on ring rows and removes a tile-dependent discontinuity.
- Falsified if ring completion or falls, or obstacle-conditioned crossing, get worse, or promotion shifts.

**How it would be decided.** A ring-row replay of the masked vs unmasked term, then a teacher 2b fork on the ring-bonus-drop substrate.

**Dependencies & conflicts.** A posture-penalty-drop REJECT, and a 2a substrate.

**Risks.** Adds penalty mass on 20-50% of envs. Interacts with the tile-flag observation.

**Rough cost.** CPU replay, plus about 2.25 GPU-h per 2b arm-seed (as in ring-bonus-drop).

**Adversarial review.** Moved out of posture-penalty-drop's fallbacks, because it adds penalty mass and needs its own screen.

### height-backstop-tag — Guard the subsumed tilt branches; decide whether the height branch should be tagged as a failure

**Kind & scope.** Fix-then-decide on terminate_episode_no_goals (omni total_terminates, time_out=True; crab_hex_omni_rewards.py:624-647, 745-753). Two parts:
- the roll/pitch > 1.5 rad branches;
- the root z < −0.25 branch.

**What changes.**
- **(i) Tilt branches.** Keep them and add a _check_omni assertion that crab_failure.limit_angle < 1.5, or remove them guarded by that assertion. KRABBY_TERM_ANGLE can raise the limit, and the legacy CrabHexTerminationsCfg uses 1.5.
- **(ii) Height branch.** It is a genuine failure, registered with time_out=True. rsl_rl therefore bootstraps it as a timeout (rewards += γ·V·time_outs). Either keep it as is or move it to a term tagged as a failure.

**Why — repo evidence.**
- crab_failure uses bad_orientation with cos(tilt) = cos(roll)·cos(pitch) (crab_hex_mdp_terminations.py:15-49).
- With pitch in [−π/2, π/2], |roll| or |pitch| > 1.5 implies tilt > 1.5 > 0.5 in the same step.
- So at the current limit_angle of 0.5 (dumps cfg_omni_1c/2a; parkour_mdp_cfg.py:1372-1381), the tilt branches are strictly subsumed. The only possible effect is a spurious timeout flag.
- In the 1c canary, only crab_failure's tilt branch fired.
- Moving the height branch changes bootstrapping. That is a learning-dynamics change, which a config dump diff cannot decide.

**Why — literature.** None needed; this is repo mechanics.

**Hypothesis / falsifier.**
- Hypothesis: the height branch fires rarely, so retagging it is neutral. If it does fire, failure-tagging removes a spurious bootstrap.
- Falsified if an offline count shows it firing materially, and a retag screen is WORSE on survival or T.

**How it would be decided.**
1. Offline count, from a root-z replay on saved raws or a logging-only split of the total_terminates tag.
2. The assertion as a unit test.
3. A screen only if the count is material.

**Dependencies & conflicts.** Spun out of zero-weight-purge. Shares bootstrap economics with survival-floor-audit.

**Risks.** Without the assertion, removing the tilt branches silently loses a backstop if the angle limit is ever raised to 1.5 or more.

**Rough cost.** About 0 GPU-h.

**Adversarial review.** Spun out because a termination change is not covered by the purge's reward-equivalence test, and moving the height branch changes bootstrapping.

## Per-term review

| term | function (omni twin) | phases: weight | X08:c6 income share | literature standing | verdict | candidate |
|---|---|---|---|---|---|---|
| track_lin_vel_xy_exp | exp(−‖cmd_xy−v‖²/std²), ungated; std 0.2 via KRABBY_TRACK_SIGMA2 (class body 0.141) | 1.25 in 1a-2c | +0.676/s, 22.5% | standard single xy kernel [1,2,4,17,22]; width narrow vs ~0.5 (Liu 0.15 [30]); its standstill floor pays 0.26-0.42/s on side/back vs 0.06 fwd | keep; possible later merge | track-single-kernel; config-truth-align (a2) |
| penalty_tracking_error_l1 | Euclidean ‖cmd_xy−v‖ (misnamed "l1"), ungated | −2.0 in 1a-2c (class body −0.5) | −0.233/s, 7.7% | a separate unbounded term is non-standard; humanoid-gym stacks a Laplace fold [51]; bounded kernels favoured [24,62] | keep until a single kernel passes; largest standstill-escape slope | track-single-kernel |
| reward_forward_progress_along_command | clamp(v·d_cmd, 0, 1.05 m/s absolute), gate 0.12 | 0.6 in 1a-2c | +0.055/s, 1.8% | standard only when capped at the command [12,25,27] | replace with the normalised form (parity offline; promoted dose after the clock) | progress-relative-cap |
| penalty_lin_vel_y | penalty_cross_command_vel_l2, move-slot gated | −3.0 in 1a-2c | −0.014/s, 0.5% | kept alongside projection-only task terms [25,27,28]; absent in full-vector stacks [2,4,17,18]; omni has projection terms, so evidence is neutral | remove in 1a-1c after screen; 2a-2c after teacher check | cross-penalty-remove |
| track_ang_vel_z_exp | exp kernel vs relay cmd_wz; std 0.5 via KRABBY_OMNI_ANG_VEL_STD (class body 0.15) | 1.0 in 1a-2c | +0.721/s, 24.0% | standard [1,2,4,17,18,29,30]; references weight it ~0.5× lin and use continuous servos | keep as the single yaw term; fix the no-preset std trap | one-yaw-kernel (survivor); config-truth-align; continuous-heading-servo |
| reward_yaw_progress | clip(wz·sign(cmd_wz), ±0.3) on cmd_wz≠0 rows; also move-row heading hold | 1.0 in 1a-2c | +0.0044/s, 0.15% | approximate PBRS [47,48,49,50]; Rudin 2022 removes r_bias after formation [23] | carries ~40% of turn-row slope; decide by registered margin | one-yaw-kernel B; yaw-progress-phaseout |
| reward_tracking_yaw | reward_facing_exp, exp(−\|e\|) on every row | 0 in 1a/1b; 0.2 in 1c-2c | +0.150/s, 5.0% | Extreme Parkour ungated [12]; reference stacks use the rate kernel only [1,4] | remove, or move to turn slots | one-yaw-kernel A; turn-slot-facing |
| reward_facing_turn | reward_facing_exp_turn_slot, slot ≥ SLOT_TURN (turn + move_turn) | 0 in all phases | 0 | no direct precedent; ANYmal Parkour terminal-time gate is weak analogy [45] | adopt at 0.2 as the replacement, else retire | turn-slot-facing; dormant-lever-retire Tier B |
| reward_clock_schedule | RewardClockContactSchedule, product combine; stance_quality only when the clock is stopped | 1.0 in 1a-2b; 2c ramp 1.0→0.5 (never run) | +0.311/s, 10.3% (15.2% of positive) | periodic composition standard [36,37]; this dose and product combine are unprecedented | coordinate with the clock session | clock-hold-twin; successors clockless-tripod-prior, mode-time-after-clock, own-tripod-discriminator |
| reward_clock_swing_apex | clock-window Gaussian on root-relative swing height (root_z − 1.05) | 1.0 in 1a-1c; 2a 1.0→0.5; 2b 0.5→0.001; 2c 0.001 | +0.133/s, 4.4% | clock-gated precedent is a penalty (WTW [36]); measured-swing forms [7,54]; no removal precedent | retire after formation, as a step of the clock work | apex-term-retire |
| reward_feet_air_time_positive | compute_first_contact (float32 edge bug); gate ‖cmd_xy‖ > 0.1; threshold 0.38 s | 0.8 in 1a-1c; 2a 0.8→0.4; 2b 0.4→0.001; 2c 0.001 | +0.0006/s, 0.02% | standard family [1,2,4]; upstream fix PR #7574 not in local tree [11] | remove (dead; latent switch-on hazard) | dead-swing-income |
| reward_stride_length | first_contact/first_air (same bug); m²; gate > 0.12 excludes turns | 0.5 in 1a-1c; 2a 0.5→0.25; 2b →0.001; 2c 0.001 | +0.00003/s, 0.00% | no precedent [4] | remove | dead-swing-income |
| penalty_foot_idle_when_forward | foot AIRBORNE > 90 steps (reset > 0.1 N); quadratic, unbounded; accrues while gate is closed | −0.12 in 1a-2c (no knob) | −0.236/s, 7.9% | non-standard; no verified bounded anti-parking form [6] | fix-then-decide; removal only after the clock | foot-idle-audit; term-discrimination-replay |
| penalty_excess_feet_contact_forward | more than 4 feet down (0.1 N), _cmd_gate | −0.2 in 1a-2c (knob exists) | −0.017/s, 0.6% | non-standard as a dense penalty; CaT uses a contact-count constraint [17] | keep (only correct-direction contact evidence; confirm under exact measurement) | term-discrimination-replay; minimal-reward-arm (first add-back) |
| reward_stance_support_feet_when_forward | bonus for ≥3 loaded feet | 0 | 0 | non-standard | unregister | zero-weight-purge |
| penalty_tibia_deviation_in_stance | knee deviation from default on loaded legs | −0.28 in 1a-2c (no knob) | −0.042/s, 1.4% | no contact-gated precedent; standard forms are command-gated [1,5,7] | remove (a stance-count tax under bang-bang) | tibia-pose-retire |
| feet_slide | isaaclab feet_slide, 1 N max-over-history gate | 0 | 0 | standard form; superseded by penalty_loaded_foot_slip | unregister | zero-weight-purge |
| reward_orientation | tilt², zeroed off parkour_flat tiles (20% of envs in 1a-1c, 50% in 2a-2c) | −0.7 in 1a-2c | −0.0033/s, 0.11% | hexapods omit it [15,28,29]; every stock flat quadruped config keeps it on [8] | set to 0, keep registered | posture-penalty-drop; orientation-unmask |
| reward_lin_vel_z | body-frame v_z², halved on ring tiles | −0.15 in 1a-2c | −0.0015/s, 0.05% | quadrupeds use −2.0 [2,4]; hexapods drop it [15,28,29] | set to 0, keep registered (needs a knob) | posture-penalty-drop |
| reward_ang_vel_xy | \|w_xy\|² | 0 | 0 | −0.05 in quadruped defaults; LiBR 0.0 [15] | unregister | zero-weight-purge |
| penalty_base_pitch_forward_linear | relu(pitch), gated on body cmd_x > 0.12 | 0 | 0 | no precedent; WTW commands pitch instead [36] | unregister | zero-weight-purge |
| reward_action_rate | unsquared ‖Δa‖ on clipped, delayed actions | −0.3 in 1a-2c | −0.342/s, 11.4% | squared −0.01 standard [2,4]; Extreme Parkour unsquared −0.1 [13]; SpiderPi −0.5 [28] | re-derive together with M1b | rate-follows-bounds; caps-smoothness-loss |
| reward_action_rate squared variant (knob KRABBY_ACTION_RATE_SQ) | squared ‖Δa‖ form of the same term (rewards.py:337-343) | off | REPLAY_GATE action_rate_sq row: Gate 1 FAIL 4/7 | squared −0.01 is the standard form [2,4] | fold into the re-derived term | rate-follows-bounds (row added by the final check) |
| reward_delta_torques | Σ(Δτ)² over all 24 joints, PD-estimate torques | −1e-6 in 1a-2c | −0.069/s, 2.3% | Extreme Parkour −1e-7 [13]; RMA ramped [27]; absent in legged_gym and Isaac Lab [2,4] | remove (converged-head screen plus a formation short chain) | delta-torque-drop; contingency rod-power-after-clock |
| penalty_mechanical_power | \|τ·q̇\| over 24 joints (24-31% is the Body_Hip emulator) | 0 | 0 (L3 ran +0.001 by sign error) | energy terms standard [39,40]; the multi-command setting differs | keep registered at 0; add sign guard; scope to rods | config-truth-align; rod-power-after-clock |
| reward_dof_error | ‖q − q_default‖², ungated | 0 (no knob) | 0 | Extreme Parkour −0.04 [13]; Spot stand-amplified [7] | unregister | zero-weight-purge |
| penalty_motor_direction_reversal | cam direction flips | 0 | 0 | non-standard; action_rate covers it | unregister | zero-weight-purge |
| reward_tripod_schedule | event crossing credit, ‖cmd_xy‖ gate, [0.30, 1.40] s band | 0 | 0 | no precedent; dense successor would be Spot GaitReward [6] | unregister | zero-weight-purge (successor: clockless-tripod-prior) |
| reward_one_direction_spin | RewardOneDirectionSpin | 0 | 0 | no precedent | unregister | zero-weight-purge |
| penalty_cam_contact_schedule | schedule referenced to hardware cam angle | 0 | 0 | phase schedules standard [37]; a cam-angle reference duplicates the clock | unregister | zero-weight-purge |
| reward_cam_phase_lock | RewardCamPhaseLock | 0 | 0 | no precedent | unregister | zero-weight-purge |
| reward_collision | contact > 0.1 N on hip/femur/body | 0 in 1a-1c; −2.0 in 2a-2c | 0 at c6 (2a −0.018, 2b −0.020, ~0.7%) | standard [2,4,17,29,45] | keep by default; test the leg-leg-interference reading | contact-telemetry-audit |
| reward_feet_edge | Extreme Parkour edge mask, terrain_levels > 3 gate; needs a 0.12 m riser; pixel index drifts up to 0.8 m | 0 in 1a-1c; −0.3 in 2a-2c | 0 (0.0000 over 10,000 teacher iterations) | Extreme Parkour only [12,13,14] | unregister via DROPPED_KEYS | zero-weight-purge |
| reward_feet_stumble | \|F_xy\| > 4\|F_z\|, no floor, no mask; also a factor in obstacle_clearance | 0 in 1a-1c; −1.0 in 2a-2c | 0 at c6 (2a/2b −0.015, ~0.5%) | kernel standard, weights vary [1,2,12,45,46]; pricing it twice is non-standard; ring risers at training difficulty are ramps, not walls | fix-then-decide | contact-telemetry-audit; ring-bonus-drop |
| reward_foot_clearance | root-relative swing-height relu (root_z − 1.05), clock gate, also active on flat tiles | 0 in 1a-2a; 1.0 in 2b-2c | 0 at c6 (2b +0.063, 2.2%) | terrain-relative, phase-gated forms are standard [19,25]; Spot uses world frame [6,7] | rebase or retire after the clock; suspected cause of the 2b apex collapse | swing-height-rebase |
| penalty_swing_min_clearance | root-relative micro-swing below 0.04 m | 0 in 1a-2a; −0.4 in 2b-2c | 0 (2b −0.014, 0.49%) | no precedent as a separate term | merge into one kernel, or retire | swing-height-rebase |
| reward_obstacle_clearance | reward_obstacle_clearance_omni: ring tile × no-stumble × progress ≥ 0.05 × speed ≥ 0.10 × upright; no lift factor | 0 in 1a-2a; 0.9 in 2b-2c | 0 (2b +0.118, 4.2%) | no precedent; parkour papers use task progress plus curriculum [14,44] | remove | ring-bonus-drop |
| reward_tracking_goal_vel | reward_tracking_command_vel_on_obstacle, min(v·d, \|cmd\|)/\|cmd\| on ring tiles | 0 in 1a-2a; 0.75 in 2b-2c | 0 (2b +0.098, 3.5%) | kernel standard [12,46]; a tile-masked duplicate is unprecedented | remove (fallback arm keeps it) | ring-bonus-drop; progress-relative-cap |
| track_lin_vel_xy_exp_facing_coupled (knob) | KRABBY_OMNI_TRACK_FACING_SIGMA swaps in a facing-coupled kernel | off | never trained or priced | no precedent; published coupling applies to the whole task reward [36,53] | retire | dormant-lever-retire Tier A |
| track_ang_vel_z_exp_turn_gated (knob) | KRABBY_OMNI_ANG_TURN_ONLY: yaw kernel × (cmd_wz ≠ 0) | off | never trained; analytic attractor | not found in [1,4] | retire | dormant-lever-retire Tier A |
| penalty_action_second_diff (knob) | motion M3 second difference | off (M3 −0.024, stale) | noise floor 22-25% at σ 2; Gate 1 FAIL 7/7 | Kim first + second difference [18] | merge into one smoothness term | rate-follows-bounds; caps-smoothness-loss |
| penalty_joint_vel_l2 (knob) | KRABBY_JOINT_VEL_W, rod q̇² | off (M3 −0.049, stale) | Gate 1 FAIL 5/7 | standard; redundant on a slew-limited plant | merge or retire | rod-power-after-clock; caps-smoothness-loss |
| penalty_loaded_foot_slip (knob) | motion M4 stance slip, 50 N gate | off (M4 −0.27 staged) | Gate 2 PASS 4-10%; Gate 1 FAIL (overruled) | standard [7,24,27] | keep (motion ladder); becomes the single slip definition | keep |
| penalty_stand_pose_l1 (knob) | KRABBY_STAND_POSE_W, stand-gated L1 | off | Gate 1 FAIL on one policy (rho −0.61); Gate 2 PASS | off by default in standard stacks [1,5]; CaT band is the better form [17] | retire (keep replay spec) | dormant-lever-retire Tier A |
| (structural) zero clip | clip(total, min=0) every step | all phases | pre-clip sum vs return: t3 first 10% −27.0 vs +6.4; c6 last 43.7 vs 52.0 | legged_gym clips [2]; Isaac Lab does not [3] | measure, declare, decide only on a trigger | survival-floor-audit |
| (termination) crab_failure | tilt > 0.5 rad OR hip contact > 500 N (time_out False) | all phases | ends 36% of c6 episodes (2a 47%, 2b 53%) | standard [4,17,18] | keep; add one-line tilt/hip log split | contact-telemetry-audit; soft-pitch-constraint |
| (termination) total_terminates | roll/pitch > 1.5 rad, z < −0.25, timeout; time_out=True | all phases | tilt branches subsumed at limit 0.5 | — | keep tilt branches with an assertion; decide the height branch | height-backstop-tag |

## Ordering and dependencies

The order is:
1. Remove dead weight first, with no GPU.
2. Measure the post-clip economy and each term's discrimination offline.
3. Register a non-inferiority rule and land the knobs at a campaign boundary.
4. Run the clock-independent fork screens.

Anything that touches clock income waits for the clock-deprecation session, which owns reward_clock_schedule, reward_clock_swing_apex, the clock observation, _clock_running and the clock-speed gate. Any candidate re-checked on the clock-free base runs as a paired arm and is never stacked with clock removal. Each resume resets PPO counters, so fork arms need same-length control forks. GPU work is serial (about 0.45 GPU-h per 1000 its), and the clock campaign holds the GPU now.

**Independent (can start any time):**
- zero-weight-purge and dead-swing-income, as one landing
- removal-knob-batch (A)
- config-truth-align (sign guard + L3 note)
- survival-floor-audit Stage A
- term-discrimination-replay (offline)
- contact-telemetry-audit Step 0
- rod-power-after-clock step 0
- height-backstop-tag count
- gait-gate-decouple, as an offer to the clock session
- the offline steps of foot-idle-audit, posture-penalty-drop, tibia-pose-retire, delta-torque-drop, swing-height-rebase (1/1b), track-single-kernel pre-check and stock-reference-arm stage 0

**Must precede:**
- ni-adoption-rule (ratified) → every removal screen: cross-penalty-remove, posture-penalty-drop, tibia-pose-retire, delta-torque-drop, one-yaw-kernel, turn-slot-facing, ring-bonus-drop, minimal-reward-arm, adopted-set-formation-check.
- removal-knob-batch (B), landed at a campaign boundary coordinated with the clock and M1b landings, and before the clock landing if minimal-reward-arm is to run (final check) → cross-penalty-remove, posture-penalty-drop (lin_vel_z), tibia-pose-retire, foot-idle-audit knobs, minimal-reward-arm.
- survival-floor-audit Stage A → term-discrimination-replay margins, minimal-reward-arm, one-yaw-kernel, cat-style-constraints.
- term-discrimination-replay → the replay steps of posture, foot-idle, cross, delta-torque, tibia, ring-bonus and swing-height.
- config-truth-align sign guard → rate-follows-bounds, delta-torque-drop, rod-power-after-clock, motion M2.
- M1b's first result → rate-follows-bounds, then caps-smoothness-loss. Also the tibia fallback, and preferably delta-torque-drop, continuous-heading-servo forks, and an own-tripod source.
- ring-bonus-drop → progress-relative-cap in 2b/2c, and the contact-telemetry-audit screen.
- progress-relative-cap → track-single-kernel. It is also a soft predecessor of cross-penalty-remove.
- turn-slot-facing or one-yaw-kernel A → one-yaw-kernel B → yaw-progress-phaseout. turn-slot-facing → dormant-lever-retire Tier B.
- Clock decision →
  - apex-term-retire (a step inside the clock work; moot if the clock goes)
  - clock-hold-twin (handover)
  - progress-relative-cap promoted arm
  - track-single-kernel
  - swing-height-rebase step 2
  - mode-time-after-clock
  - clockless-tripod-prior (only on observed tripod loss)
  - rod-power-after-clock step 1
  - continuous-heading-servo (soft)
  - survival-floor-audit Stage B trigger
  - the anchor for adopted-set-formation-check

  minimal-reward-arm must run before the clock goes, or not at all unless a working swing term exists.
- minimal-reward-arm → stock-reference-arm stage 1.
- minimal-reward-arm and tibia-pose-retire both REJECT → cat-style-constraints.
- posture-penalty-drop REJECT → soft-pitch-constraint and orientation-unmask.
- delta-torque-drop Q regression → rod-power-after-clock contingency.
- ≥2 fork adoptions → adopted-set-formation-check.
- Clockless head and minimal-reward-arm both lose tripod → own-tripod-discriminator.

**Conflicts.**
- minimal-reward-arm bundles the removals that cross-penalty-remove, posture-penalty-drop, tibia-pose-retire and the foot-idle removal arm test one at a time. It is cheaper but loses attribution; individual results already decided pre-decide its parts.
- minimal-reward-arm keeps yaw_progress and tracking_yaw, while one-yaw-kernel removes them (arm B covers that). It also drops progress, which progress-relative-cap replaces instead. Freeze its term set at pre-registration.
- rate-follows-bounds and caps-smoothness-loss are alternatives; adopt at most one.
- rod-power-after-clock is delta-torque-drop's contingency, not its replacement.
- apex-term-retire's fallback is swing-height-rebase's terrain-relative kernel, not today's root-relative reward_foot_clearance.
- Free standstill income from the exp kernels on stand/turn rows is unresolved. The tracking review calls it correct tracking; the structure review calls it a stand-attractor leak. survival-floor-audit re-measures it post-clip before anyone gates kernels. Never gate kernels on cmd_wz.
- config-truth-align (a2) reverses the documented knob-timing pillar, so it is a user decision; (a1) keeps the pillar.
- continuous-heading-servo conflicts with M5 relay hysteresis. STAND_HOLD enables it.
- gait-gate-decouple, clock-hold-twin and apex-term-retire may be absorbed by the clock session.
- mode-time-after-clock and clockless-tripod-prior are not stacked in one arm unless tripod loss is observed.
- minimal-reward-arm and adopted-set-formation-check overlap; run one first.

**Corrections to earlier notes.**
- Clock plus apex = 0.444/s, which is 14.8% of |income| and 21.7% of positive income.
- The progress cap is an absolute 1.05 m/s, not 1.05 × cmd.
- foot_idle penalizes an airborne foot, not a foot in contact.
- penalty_tibia_deviation_in_stance is replayable (replay_motion_terms.py).
- The orientation mask zeroes 20% of envs in 1a-1c and 50% in 2a-2c.
- Phase-out ramps:
  - 2a: apex 1.0→0.5, air time 0.8→0.4, stride 0.5→0.25
  - 2b: each →0.001
  - 2c: reward_clock_schedule 1.0→0.5 (never run)
- 2b eval raws exist under logs/rsl_rl/gait_eval/omni_phases/2026-09-16_1805_omni_formation/seed3_{1c_004,2b_006}_A15pB{,_obst}/.
- arXiv 2307.10142 is by Jeon, Heim, Khazoom and Kim.
- The ladder recipe_id hashes only {chain, overlay_by_stage} plus fork checkpoints. It contains no reward names and no preset env.
- The existing omni dumps already contain the train task ids.
- Six class-body values disagree uniformly with every shipped phase; the seventh, tracking_yaw, differs by phase.
- KRABBY_HEADING is already dropped for omni.
- excess_contact already has a knob.
- D11's "near bang-bang" referred to the 0.21 cap.
- No seed-2 X08 run exists.

## Dropped candidates

No candidate got a kill verdict. All 29 catalogue candidates and all 6 gap candidates were revised, and they appear above in revised form. Splits made during revision:
- cat-soft-constraints became cat-style-constraints and soft-pitch-constraint.
- one-yaw-kernel became parts A and B.
- posture-penalty-drop's unmasking fallback became orientation-unmask.
- zero-weight-purge's termination part became height-backstop-tag.

Merged or declined:
- **omni-reward-purge**: split. The dead names went to zero-weight-purge; the config surface went to config-truth-align.
- **dead-gait-term-purge, posture-dead-names, feet-edge-dead**: merged into zero-weight-purge.
- **regularizer-dead-cleanup**: split. dof_error went to zero-weight-purge; the sign guard and the L3 note went to config-truth-align.
- **dead-foot-terms**: split. stance_support and feet_slide went to zero-weight-purge; air time and stride went to dead-swing-income.
- **yaw-lever-cleanup**: split. The std trap went to config-truth-align; the three levers went to dormant-lever-retire.
- **facing-coupled-retire, stand-pose-knob-retire**: merged into dormant-lever-retire.
- **stumble-noise-audit, femur-contact-audit**: merged into contact-telemetry-audit, together with the crab_failure tilt/hip split.
- **soft-pitch-termination**: merged into cat-soft-constraints as Set P, then split back out as soft-pitch-constraint.
- **dead-swing-income part b** (timing-fixed air time after the clock): became mode-time-after-clock arm A.
- **track-single-kernel arm 2** (the exact Gaussian-minus-linear fold): split off and dropped. It is a no-op refactor of low value. If revived, it needs dormant-lever-retire as a hard prerequisite (omni_knobs.py:877-903 requires the stock func), component logging (run_phases.py:420 reads the name), and an allclose test rather than a bitwise one.
- **World-pose target with a sparse terminal reward** (Rudin 2022, Sec. V): declined. It redefines the task.
- **Slot-gating the ungated exp kernels**: declined. It drops yaw damping, the stand leak belongs to M5/M6/C8, and the evidence is pre-clip. survival-floor-audit re-measures it first.
- **Halving track_ang_vel_z_exp toward the 0.5× reference ratio**: declined. Not a simplification, and it cuts the strongest turning price.
- **Merging exp + L1 + progress in one step**: declined. It is covered stepwise by progress-relative-cap and track-single-kernel.
- **WTW-style multiplicative gating of clock income**: handed to the clock session as a reserve idea.
- **Hwangbo single-scalar penalty curriculum**: declined. It adds a mechanism.
- **Grouped or multi-critic value heads**: declined. It is a PPO change, not a reward change.
- **Value normalisation**: declined as a reward candidate. It remains a control option.
- **Re-deriving the crab_failure thresholds or splitting roll/pitch limits**: declined. It changes the eval fall definition.
- **Separate "unify gates and contact thresholds" candidate**: declined. It is covered by gait-gate-decouple and M5.
- **Merged base-motion term**: not proposed. It is the named re-entry form if posture-penalty-drop regresses.
- **RMA-style ramp-in of action_rate**: folded into rate-follows-bounds as the from-scratch escalation.
- **Running the staged M3 overlay as staged**: dropped. It was calibrated on the saturated mean.
- **Moving reward_collision into 1a-1c**: not proposed. It is informed by contact-telemetry-audit and would be its own from-scratch candidate.
- **Fixing the float32 contact-timing bug**: out of scope; it is a separate task. It gates mode-time-after-clock arm A.
- **Removing the dead copies from the forward and sideways tasks**: out of scope, because those lineages are closed.
- **Fork-based equivalence arms** (dead-swing-income (c); the gait-gate-decouple smoke; the removal-knob-batch X0 re-anchor): removed. They cannot inform when single-seed runs are not bit-reproducible.
- **Jiang et al. 2026 hexapod heading-hold term**: its source was verified, but it keeps the term without ablating it, so it is non-evidential. Not cited as a counterexample (final check).

## References

1. legged_gym, legged_robot.py — https://raw.githubusercontent.com/leggedrobotics/legged_gym/master/legged_gym/envs/base/legged_robot.py (_prepare_reward_function drops zero scales; _post_physics_step_callback heading command; _reward_feet_air_time; _reward_stand_still; _reward_stumble; compute_reward clips, then adds the termination reward)
2. legged_gym, legged_robot_config.py — https://raw.githubusercontent.com/leggedrobotics/legged_gym/master/legged_gym/envs/base/legged_robot_config.py (rewards.scales; tracking_sigma 0.25; only_positive_rewards; heading_command; action_rate −0.01)
3. Isaac Lab, RewardManager — https://raw.githubusercontent.com/isaac-sim/IsaacLab/main/source/isaaclab/isaaclab/managers/reward_manager.py (None terms skipped in _prepare_terms; no clip in compute)
4. Isaac Lab, velocity_env_cfg.py — https://raw.githubusercontent.com/isaac-sim/IsaacLab/main/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/velocity_env_cfg.py (RewardsCfg; CommandsCfg rel_standing_envs 0.02, heading stiffness 0.5; base-contact termination)
5. Isaac Lab, velocity mdp rewards.py — https://raw.githubusercontent.com/isaac-sim/IsaacLab/main/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/mdp/rewards.py (feet_air_time gate norm(cmd[:, :2]) > 0.1; stand_still_joint_deviation_l1)
6. Isaac Lab, Spot mdp rewards.py — https://raw.githubusercontent.com/isaac-sim/IsaacLab/main/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/spot/mdp/rewards.py (air_time_reward, air_time_variance_penalty, GaitReward, foot_clearance_reward, base_linear_velocity_reward)
7. Isaac Lab, Spot flat_env_cfg.py — https://raw.githubusercontent.com/isaac-sim/IsaacLab/main/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/spot/flat_env_cfg.py (gait weight 10.0, std 0.1, max_err 0.2; air_time 5.0, mode_time 0.3; foot_clearance target 0.1; joint_pos −0.7, stand_still_scale 5.0)
8. Isaac Lab, ANYmal-D flat_env_cfg.py — https://raw.githubusercontent.com/isaac-sim/IsaacLab/main/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/anymal_d/flat_env_cfg.py (flat_orientation_l2 −5.0)
9. *(removed by the final check: unverified and unused)*
10. Isaac Lab, velocity_command.py — https://raw.githubusercontent.com/isaac-sim/IsaacLab/main/source/isaaclab/isaaclab/envs/mdp/commands/velocity_command.py (_update_command: continuous heading P-control, standing envs zeroed)
11. IsaacLab PR #7574, "Fix compute_first_contact/air missing transitions as the sensor clock ages" — https://github.com/isaac-sim/IsaacLab/pull/7574 (merged 2026-09-07; 352/500 missed; new default abs_tol)
12. Extreme Parkour, legged_robot.py — https://raw.githubusercontent.com/chengxuxin/extreme-parkour/main/legged_gym/legged_gym/envs/base/legged_robot.py (_reward_tracking_goal_vel, _reward_tracking_yaw, _reward_feet_stumble "vertical surfaces", ang_vel_clip at resample, tile rule)
13. Extreme Parkour, legged_robot_config.py — https://raw.githubusercontent.com/chengxuxin/extreme-parkour/main/legged_gym/legged_gym/envs/base/legged_robot_config.py (action_rate −0.1 unsquared, clip 1.2, delta_torques −1e-7, dof_error −0.04)
14. Cheng et al., Extreme Parkour with Legged Robots — https://arxiv.org/html/2309.14341 (Sec. 4.2, Eq. 2)
15. LiBR hexapod flat_env_cfg.py — https://raw.githubusercontent.com/psu-LiBR/IsaacLab-LiBRHexapod/main/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/hexapod/flat_env_cfg.py (lin_vel_z −1e-8; ang_vel_xy 0.0)
16. Ma et al., Eureka, ICLR 2024 — https://arxiv.org/html/2310.12931 (Sec. 3.3 reward reflection; Appendix A Prompt 2)
17. Chane-Sane et al., CaT: Constraints as Terminations, IROS 2024 — https://arxiv.org/html/2403.18765 (Sec. III, Method, Algorithm 1, Table I)
18. Kim, Oh, Lee et al., T-RO 2024 — https://arxiv.org/html/2308.12517 (Table I; Sec. V-B, Table III)
19. Kim, Lee, Park, ICRA 2025 — https://arxiv.org/html/2409.15780v4 (Sec. III-B clearance; Sec. III-D r_pos·exp(0.2 r_neg))
20. Reda, Tao, van de Panne, MIG 2020 — https://arxiv.org/abs/2010.04304 (Survival Bonus section)
21. Andrychowicz et al., ICLR 2021 — https://arxiv.org/abs/2006.05990 (Sec. 3.2; Sec. 3.3)
22. Rudin et al., 2021 — https://arxiv.org/abs/2109.11978 (reward table)
23. Rudin et al., 2022 — https://arxiv.org/html/2209.12827v1 (Sec. II-B3, r_bias removed at 50% r_task)
24. Hwangbo et al., 2019 — https://arxiv.org/abs/1901.08652 (Suppl. S3; curriculum)
25. Lee et al., Science Robotics 2020 — https://arxiv.org/html/2010.11251v1 (Suppl. S4, Eq. 15)
26. Lee et al., 2023 (N-P3O) — https://arxiv.org/html/2309.15430 (Table II)
27. RMA: Rapid Motor Adaptation — https://arxiv.org/html/2107.04034 (reward list; small-coefficient curriculum)
28. Qu et al., 2024 (SpiderPi hexapod) — https://arxiv.org/html/2412.10628 (Table I)
29. Yang, Gao, Li, 2023 (CPG hexapod) — https://arxiv.org/abs/2310.07744 (reward simplification; collision terms)
30. Liu et al., 2025 (hexapod motion prior) — https://arxiv.org/html/2511.03167 (Table I; Sec. V-A)
31. HeLoM — https://www.arxiv.org/pdf/2509.23651v1 (Table III)
32. Yu, Turk, Liu, 2018 — https://arxiv.org/html/1801.08093v3 (Sec. 6.1, Hexapod)
33. Chane-Sane et al., 2024 (air-time constraint) — https://arxiv.org/html/2412.04273v1 (ablation; Table 3)
34. van Marum et al., IROS 2024 — https://arxiv.org/html/2404.19173 (Sec. IV-B; Sec. V-C)
35. Li, Wu, Liu, Guo, Xue, two-step hexapod reward, 2024 — https://arxiv.org/html/2401.12389 (Table I; Secs. III-B, III-C, V-A, V-B)
36. Margolis & Agrawal, Walk These Ways — https://arxiv.org/abs/2212.03238 (Table 1; Sec. 3.1; limitations, Table 5)
37. Siekmann et al., ICRA 2021 — https://arxiv.org/abs/2011.01387 (Sec. IV, State Space)
38. HRP-5P aperiodic walking — https://arxiv.org/html/2504.13619 (Sec. V; Table V)
39. Fu et al., CoRL 2021 — https://arxiv.org/abs/2111.01674 (Sec. 2.4; Fig. 7 caption; Secs. 4.2-4.4)
40. Tan et al., RSS 2018 — https://arxiv.org/abs/1804.10332 (Sec. IV-C, Eq. 2)
41. Seyde et al., NeurIPS 2021 — https://arxiv.org/abs/2111.02552 (Sec. 5.4.2, Fig. 9; Fig. 7)
42. Mysore et al., CAPS, ICRA 2021 — https://arxiv.org/abs/2012.06644 (Sec. III, Eq. 1-3; Table II; Fig. 2)
43. Chen et al., LCP, 2024 — https://arxiv.org/abs/2410.11825 (Table I)
44. Zhuang et al., Robot Parkour Learning, CoRL 2023 — https://arxiv.org/html/2309.05665 (Sec. 3.1)
45. Hoeller et al., ANYmal Parkour, 2023 — https://arxiv.org/html/2306.14874 (Tables S2, S6; reward table)
46. SoloParkour — https://arxiv.org/html/2409.13678 (App. A.1, Table 2)
47. Ng, Harada, Russell, ICML 1999 — https://people.eecs.berkeley.edu/~pabbeel/cs287-fa09/readings/NgHaradaRussell-shaping-ICML1999.pdf (potential-based shaping)
48. Grześ, AAMAS 2017 — https://www.ifaamas.org/Proceedings/aamas2017/pdfs/p565.pdf (Eq. 3 discussion)
49. Jeon, Heim, Khazoom, Kim, 2023 — https://arxiv.org/abs/2307.10142 (abstract: marginal convergence benefit, robust to scaling)
50. IsaacGymEnvs ant.py — https://raw.githubusercontent.com/isaac-sim/IsaacGymEnvs/main/isaacgymenvs/tasks/ant.py (potential-difference progress)
51. humanoid-gym, humanoid_env.py — https://raw.githubusercontent.com/roboterax/humanoid-gym/main/humanoid/envs/custom/humanoid_env.py (track_vel_hard; tracking_lin_vel)
52. humanoid-gym, humanoid_config.py — https://raw.githubusercontent.com/roboterax/humanoid-gym/main/humanoid/envs/custom/humanoid_config.py (XBotL low_speed)
53. dm_control, quadruped.py — https://raw.githubusercontent.com/google-deepmind/dm_control/main/dm_control/suite/quadruped.py (Move task multiplicative reward)
54. Unitree rl_gym, g1_env.py — https://github.com/unitreerobotics/unitree_rl_gym/blob/main/legged_gym/envs/g1/g1_env.py (_reward_feet_swing_height)
55. rl_games, a2c_continuous.py — https://raw.githubusercontent.com/Denys88/rl_games/master/rl_games/algos_torch/a2c_continuous.py (bound_loss, soft bound 1.1)
56. IsaacGymEnvs, AnymalPPO.yaml — https://raw.githubusercontent.com/isaac-sim/IsaacGymEnvs/main/isaacgymenvs/cfg/train/AnymalPPO.yaml (bounds_loss_coef 0.001)
57. Escontrela et al., 2022 (AMP for hardware) — https://arxiv.org/html/2203.15103; code https://github.com/escontra/AMP_for_hardware (amp_ppo.py, amp_discriminator.py, a1_amp_config.py)
58. Vollenweider et al., Multi-AMP, 2023 — https://arxiv.org/pdf/2203.14912 (Sec. II)
59. Huang, Xie, Li, 2025 — https://arxiv.org/html/2509.21810 (Sec. III-C)
60. Booth et al., AAAI 2023 — https://www.cs.utexas.edu/~pstone/Papers/bib2html-links/booth2023perils.pdf (p. 2, 68% of 30 experts)
61. Knox et al., 2023 — https://arxiv.org/abs/2104.13906 (Table 1, check 6)
62. Ha et al., survey — https://arxiv.org/html/2406.01152v2 (Sec. 3.3.1)
63. Lakens, 2017, equivalence testing — https://pmc.ncbi.nlm.nih.gov/articles/PMC5502906/ ("Testing for Equivalence", "Setting Equivalence Bounds")
64. Everson-Stewart & Emerson, Stat Med 2010 — doi:10.1002/sim.4053 (bio-creep; abstract)
65. Patterson et al., JMLR 2024 — https://jmlr.org/papers/v25/23-0183.html (secs 2.6, 4.4, 7)
66. Colas et al., 2018 — https://arxiv.org/abs/1806.08295 (power analysis)
67. Ash & Adams, NeurIPS 2020 — https://arxiv.org/abs/1910.08475 (warm start vs fresh init; supervised-learning analogy)