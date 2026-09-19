<!-- paths-note -->
> **Paths note (2026-09-09):** this campaign moved from `sim_fine_tuning/2026-08-13_0035_mirror_symmetry/` to `parkour/parkour_tasks/parkour_tasks/crab_hex_forward_task/experiments/2026-08-13_0035_mirror_symmetry/`. Absolute paths below (`/home/.../sim_fine_tuning/...`, `parkour/logs/rsl_rl/...`) name raw artifacts that stay on disk untracked; the tracked checkpoint of record is `head/` (see `bundle.yaml`) and the eval summaries are in place / under `evals/`.

# Mirror-symmetry training campaign

Pre-authorized by user 2026-08-13 after the seed search (0/7 alternating, handedness broke both
directions). Goal: make permanent lead-set preference impossible to encode by enforcing the
robot's L/R mirror symmetry (which maps tripod set A={FL,MR,RL} exactly onto B={FR,ML,RR}).

## Implementation route (confirmed by code exploration)

`PPOWithExtractor` (parkour/scripts/rsl_rl/modules/ppo_with_extractor.py) ALREADY carries
rsl-rl's symmetry machinery: `symmetry_cfg` dict {use_data_augmentation, use_mirror_loss,
data_augmentation_func (resolvable path), mirror_loss_coeff}; mirror-loss path at lines
428-452: L_sym = MSE(policy_mean(mirror(obs)), mirror(policy_mean(obs))), added at coef.
We only supply the crab-specific `data_augmentation_func` + cfg wiring. Function contract:
`f(obs, actions, env, obs_type)` → (cat([obs; mirror(obs)]), cat([act; mirror(act)])), each
arg optionally None.

## Mirror-map specification (policy obs = 87 proprio + 132 scan + 9 priv_e + 53 priv_l + 870 history = 1151)

**Proprio (15 head dims)**: [0]wx→−, [1]wy→+, [2]wz→−, [3]roll→−, [4]pitch→+, [5]0·dy→−,
[6]delta_yaw→−, [7]delta_next_yaw→−, [8]0·cmd_vx→+, [9]0·cmd_vy→−, [10]cmd_vx→+, [11]env_idx→+,
[12]inv_idx→+, [13]lin_vx→+, [14]lin_vy→−. Then joint_pos−default (24), joint_vel×0.05 (24) —
joint map below; last action (18) — action map below; contact fill (6): swap FL↔FR, ML↔MR,
RL↔RR, sign +.

**Joint map (24 dims, articulation order resolved at runtime)**: L↔R pair swap per joint type
with signs: Body_CamShaft − (Whitworth map atan2(K sinθ, 1+K cosθ) verified ODD → defaults
negate L/R); Body_Hip (passive) −; Hip_Femur + (defaults equal L/R, 0.30); Femur_Tibia −
(180° Z USD flip convention). KNOWN APPROXIMATION: knee defaults are −0.07(L)/+0.10(R), not an
exact negation — the 0.03 rad deliberate roll-balance asymmetry makes the delta-mirror
approximate on knees (≤0.03 rad model error; acceptable for a soft regularizer; documented).

**Action map (18 dims, action-term order resolved at runtime)**: same L↔R swap; signs
CamShaft −, Hip_Femur +, Femur_Tibia −; same knee approximation (≈0.125 action-units).

**Scan (132)**: GridPatternCfg(resolution 0.15, size [1.65,1.5]) → 12×11; mirror = y-axis flip
of the ray grid; exact flatten ordering to be read from installed isaaclab
patterns.grid_pattern (indexing mode determines permutation). On flat terrain this is ~identity
(constant heights), so the validation run is insensitive to it; still built correctly.

**priv_explicit (9)**: lin_vel_b×2 (+,−,+) then two zeroed 3-blocks with same pattern.
**priv_latent (53)**: mass(1)+, com_b(3)=(+,−,+), friction(1)+, stiffness ratio(24) joint-perm
sign+, damping ratio(24) joint-perm sign+.
**history (870)**: the 87-dim per-step map applied to each of 10 slots.
**critic obs**: dispatch on last-dim; if equal to policy dim apply same map, else raise.

## Plan

1. crab_hex_mirror.py pure module: build_permutation_and_signs(joint_names, action_joint_names,
   scan_order) → index+sign tensors; mirror_obs/mirror_actions/data-augmentation entry point;
   maps built from RESOLVED NAME LISTS, never hand-typed indices.
2. tests: involution mirror(mirror(x))==x (obs+act), permutation validity (bijection),
   per-block sign counts, zeroed-slot consistency.
3. Wiring: symmetry_cfg into the agent cfg used by crab_on_policy_runner (default OFF), enable
   per-run; verify `_env` injection point in on_policy_runner_with_extractor.
4. 100-iter smoke (coef 0.5): symmetry loss logged, decreasing, no NaN.
5. 20k from-scratch validation, PLAIN config (no v5 — seed search: 0/8 v5-active vs 1/1 plain),
   coef 0.5. Mid-checks at 3000/5000 (alternating? duty balanced?), full eval at end vs
   baseline (0.401/+0.209/0.146-0.556). STOP for user with results.

## Literature adjustments (user directive 2026-08-13, from docs/lit-review-hexapod-reward-stability.md)

Applied to this campaign once the current 20k validation completes:

1. **Enforcement method** (Abdolhosseini et al., MIG 2019): auxiliary mirror loss is the most
   consistent method (our current setup — loss primary ✓); data duplication weakest. Next arm
   after validation: **S+A** = mirror loss + use_data_augmentation=True as support.
2. **Pre-registered failure mode for the validation eval**: a symmetric POLICY can still
   produce HANDED ROLLOUTS (spontaneous symmetry breaking picks one mirrored trajectory per
   episode). The eval must therefore analyze PER-EPISODE handedness: duty_A per episode across
   the 10 eval episodes.
   - All episodes same-handed → policy-level handedness (loss too weak) → S+A arm / coef retune.
   - Episodes split both-handed (~mixed signs) → policy symmetric, rollouts break symmetry →
     **the indicated fix is symmetry + CLOCK REWARD combined** (lit review §2: Siekmann ICRA
     2021 periodic reward composition; schedule as reference, not price), NOT more symmetry
     weight. Clock term must pass the offline replay gate per §2's recipe (healthy npz scores
     well at its best-aligned phase; all degenerate traces score poorly at EVERY phase).
   - Balanced within episodes + alternating → symmetry sufficient; adoption path.
3. **Escalation** if loss+augmentation underdeliver: hard-equivariant architecture (Su et al.,
   IROS 2024 — strictly better than augmentation).

## 20k validation: SYMMETRY SUFFICIENT — campaign target broken

model_19999 (plain config + mirror loss 0.5, 6h14m, deterministic eval):

| | baseline 19999 | b7c peak (best prior) | **sym 19999** |
|---|---|---|---|
| tripod | 0.401 | 0.4248 | **0.5170** (all 10 eps 0.50-0.53) |
| duty_A / duty_B | 0.146 / 0.556 | 0.14 / 0.56 | **0.356 / 0.339** |
| per-episode handedness | all B-handed | all B-handed | **10/10 BALANCED** (pre-registered diagnostic: no policy handedness, no rollout symmetry breaking) |
| pearson(a,b) | −0.606 | −0.61 | **−0.730** |
| dominant-set swaps / 10 eps | ~0-6 | 262 | **812** |
| completion / slip / tippy | 100% / 2.6% / 6.5% | 100% / 2.3% / 7.2% | 100% / **2.02%** / 6.6% |
| roll_rms / EMA(v_z) / stride | 0.0387 / 0.134 / 0.163 | 0.038 / 0.125 / 0.164 | 0.0498 / 0.140 / 0.148 |
| signed pitch | +0.209 | +0.209 | +0.202 (lean persists — L/R mirror does not constrain pitch, as pre-registered) |
| v5 replay income | 1.4-1.9/min | ~5/min | **25.7/min** (70% of ideal synthetic; term not used in training) |

**Verdict: branch (a) of the pre-registered tree.** The mirror loss alone (coef 0.5) killed
handedness at both policy and rollout level, unlocked genuine deep tripod alternation
(0.517 ≥ the 0.45 bar every reward campaign chased), improved slip, and cost nothing
anywhere else. The one remaining item from the original goals is the ~11.6° lean — now a
clean, isolated target (candidate: clock reward with commanded body pitch per Walk These
Ways, or a dedicated pitch curriculum — both easier against a duty-balanced gait).

Adoption proposal presented to user (see chat); campaign STOPPED for review per protocol.

## Teacher-stack carry-up (symmetric lineage): COMPLETE — symmetry survives the curriculum

bridge (100) → 2b1 (100) → 2b2 (4×500, sweet spot at iter ~21994 → model_22000). Per-batch
2b2 gates: b1 2/5 → b2 3/5 (clearance passes) → b3 3/5 (ep_len passes, clearance dips) → b4
best window: clearance 0.183 ✓, crab_failure **12.0%** (best of any lineage; old lineage
oscillated 15.6-21.5%) ✓, ep_len 806 ✓, goal_idx 1.37 ✓. fwd_progress 0.089 — gate >0.15 never
reached by ANY lineage (old best 0.106); recorded as aspirational, plateau called per skill.

**Duty-balance gates (flat-walk eval, no mirror loss active in any teacher stage):**
| stage | duty_A/B | handed eps | tripod | completion | slip |
|---|---|---|---|---|---|
| 2b1 model_20197 | 0.346/0.359 | 0/9 | 0.519 | 90% (1 fall — transitional ckpt) | 2.4% |
| **2b2 model_22000** | **0.357/0.359** | **0/10** | **0.5255** | 100% | 1.8% |

The balanced tripod gait not only survived ~2200 iterations of obstacle-curriculum fine-tuning
without symmetry enforcement — it *improved* (tripod 0.517 → 0.5255 on the flat eval). The
symmetric basin is stable under downstream training; no teacher-side mirror loss needed.
2b2 teacher checkpoint for downstream stages (student/full1): model_22000.

## Student distillation (symmetric lineage) — batch 1 running

From 2b2 model_22000. Launch note: 256 envs OOM'd twice at the depth-encoder (14.0-14.1 GB vs
15.5 GB total with ~570 MB held by desktop apps — the successful Aug 6 student ran without
that load). Running at **192 envs** + expandable_segments (13.25 GB, stable); caveat: ~25%
smaller distillation batch than the old lineage's 256-env student. Rate 6.0 s/iter → batch 1
(4000 iters) ≈ 6.7 h. Primary metric: Mean depth_actor_loss (min), skill batch rules, ceiling
12000.
