<!-- paths-note -->
> **Paths note (2026-09-09):** this campaign moved from `sim_fine_tuning/2026-08-12_1017_lean_reduction/` to `parkour/parkour_tasks/parkour_tasks/crab_hex_forward_task/experiments/2026-08-12_1017_lean_reduction/`. Absolute paths below (`/home/.../sim_fine_tuning/...`, `parkour/logs/rsl_rl/...`) name raw artifacts that stay on disk untracked; the tracked checkpoint of record is `head/` (see `bundle.yaml`) and the eval summaries are in place / under `evals/`.

# Lean-reduction campaign: weights-only sweep on existing terms

Goal (user, 2026-08-12): lessen the +12° forward lean and increase tripod score by tuning
weights of EXISTING registered terms only — no new reward functions. Successor to the tripod
campaign (`2026-08-10_0058_tripod_stability/`), which closed with the diagnosis that the duty
asymmetry capping tripod at ~0.40 is postural (anchored by the lean). This campaign tests the
postural hypothesis from the config side. Fully autonomous per user; v5 crossing-credit term
(registered, dormant) allowed in combo arms.

## Protocol

Every arm: fine-tune from healthy baseline model_19999
(`2026-08-09_1526_gait_tuned/logs/rsl_rl/crab_hex_flat_walk/2026-08-09_15-27-14/model_19999.pt`),
2000 iters, 256 envs, seed 1. Gait-eval BOTH the ~1000 and final ~1998 checkpoints (two-point
verdicts; the b7d lesson — single-point verdicts get fooled by checkpoint oscillation). Score:
standard metrics + signed mean pitch + duty_A/duty_B/pearson decomposition. Eval is
deterministic per checkpoint (established 2026-08-12); the yardstick is checkpoint-to-checkpoint
spread (b6 family: ±0.003 tripod).

**Primary target**: signed mean pitch ≤ +0.18 rad at both eval points (baseline +0.209;
improvement ≥0.03 rad). **Secondary**: tripod (win declared only at ≥0.45 on two consecutive
checkpoints). **Guardrails** (any breach at final ckpt = REVERT): completion=100%, slip ≤3.5%,
roll_rms ≥0.03, EMA(v_z) ≤0.174, stride ∈[0.11,0.22]m.

## Baseline reference (model_19999, deterministic protocol)

tripod 0.401 | signed pitch +0.209 | duty_A 0.146 / duty_B 0.556 | pearson −0.606 |
completion 100% | slip 2.6% | roll 0.0387 | EMA 0.134 | stride 0.163 m | tippy 6.5%

## Arms

| arm | override(s) | rationale |
|---|---|---|
| L1 | reward_orientation −0.7→−2.0 | never swept in any campaign; lean contributes ~95% of the term's signal (sin²(12°)=0.043 vs ~0.002 roll) |
| L2 | reward_orientation →−3.5 | escalation, only if L1 partial |
| L3 | penalty_base_pitch_forward_linear 0.0→−0.5 | 2× past Phase A max dose (−0.25 moved pitch ≤1%) |
| L4 | penalty_base_pitch_forward_linear →−1.0 | escalation, only if L3 partial |
| L5 | reward_forward_progress_along_command 0.6→0.3 | Task-1 §2.6 speed-pressure lever, never reached; lean-as-momentum-posture hypothesis |
| L6 | best lean-mover + v5 reward_tripod_schedule @0.3 | only if a single knob moves pitch; b7 evidence: v5 amplifies structure when active (peak tripod 0.425) |

## Runs

| L1 | reward_orientation −0.7→−2.0 | mid/final: pitch **+0.2089/+0.2091** (baseline +0.209 — zero movement), tripod 0.389/0.397, completion 100/100%, slip 2.5/2.2%, roll 0.037, EMA 0.140/0.130, stride 0.160/0.166 | **INERT on pitch** — the policy pays the tripled tilt cost (0.086/step at 12°) rather than adjust posture. L2 escalation skipped per protocol (0.000 movement at 3× dose). All guardrails clean. |

| L3 | penalty_base_pitch_forward_linear 0.0→−0.5 | mid/final: pitch **+0.2098/+0.2095** (baseline +0.209 — zero movement at 2× Phase A max dose), tripod 0.404/0.378, completion 100/100%, slip 2.5/2.2%, roll 0.037, EMA 0.135/0.144, stride 0.164/0.164 | **INERT on pitch** — dose-response flat across −0.1→−0.5 (5× range, Phase A + this arm). L4 (−1.0) skipped per protocol. Guardrails clean. |

| L5 | reward_forward_progress_along_command 0.6→0.3 | mid/final: pitch **+0.2098/+0.2094** (zero movement), tripod 0.397/0.418, completion 100/100%, slip 2.4/2.3%, roll 0.037, EMA 0.142/0.135, stride 0.164/0.164 (gait didn't even slow) | **INERT on pitch** — halving speed pressure changes neither posture nor stride; the momentum-posture hypothesis is refuted. Guardrails clean. |

## Campaign summary: the lean is invariant to every existing config lever — CLOSED

Signed mean pitch measured +0.209 rad (±0.001) at every checkpoint of every arm:

| lever | dose vs prior max | pitch response |
|---|---|---|
| L1 reward_orientation (quadratic, direction-blind) | 3× baked weight | 0.000 |
| L3 penalty_base_pitch_forward_linear (signed, targeted) | 5× total range incl. Phase A | 0.000 |
| L5 forward-progress (speed pressure) | halved | 0.000 (stride also unchanged) |

Combined with the tripod campaign (v5 shaping at 2 weights, pitch −0.1 in combo, 6k-iter
extensions) and Phase A (pitch −0.1/−0.25, ang-vel damping), the +12° forward lean has now
survived every reward-side perturbation available in the config at doses up to 5× beyond
previous maxima — while the policy demonstrably responds to these same weights on other axes
(slip, swaps, timing all move). L6 not launched (precondition failed: no pitch-mover to
combine with v5).

**Conclusion**: the lean — and the duty asymmetry it anchors, which caps tripod at ~0.40 — is
structurally locked: CoM placement relative to the foot workspace, the cam mechanism's
sinusoidal foot paths, and/or the default/spawn posture. Reward weights price behavior; they
cannot buy a posture the plant makes expensive. The route to tripod >0.45 and load-ready
stability is morphology/CoM/cam-geometry work (or spawn-posture and curriculum experiments),
reusing this campaign's eval pipeline unchanged.
