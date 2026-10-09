# SPDX-License-Identifier: BSD-3-Clause
"""Motion-quality knobs (gait-stabilisation plan, step 2, 2026-09-22) -> the omni env / runner cfgs.

EVERY KNOB HERE IS DEFAULT OFF AND SET BY NO PRESET. Unset, blank or at its off value
(``"0"`` for flags, weights and the filter corner; ``"0"``/``"off"`` for the relay hysteresis)
each one writes NOTHING into the cfg -- no field, no term, no param -- so a knob-off config
dump is byte-identical to one built before this module existed (the ``dumps_ladder_base``
contract), and an eval that carries a knob at its off value builds the same cfg as one that
does not carry it at all (``run_phases.EVAL_CARRY_DEFAULTS``).

==============================  ===========================================================
variable                        what it writes when armed
==============================  ===========================================================
**exploration (runner cfg; ``apply_exploration_knobs``)**
``KRABBY_ACTION_STD_MAX``       ``policy.action_std_max`` -- the upper clamp on the Gaussian
                                std (``CrabHexActorCriticRMA``; stock 2.0)
``KRABBY_ACTION_STD_RESET``     ``policy.action_std_reset`` -- on EVERY checkpoint load
                                (resume / fork) the std parameter is set to this value
``KRABBY_ENTROPY_COEF``         ``algorithm.entropy_coef`` (stock 0.01)
**regularisers (env cfg rewards; weights are the RewTerm weight, sign included, <= 0)**
``KRABBY_ACTION_RATE_SQ=1``     ``rewards.reward_action_rate.params["squared"] = True``
                                (``||da||^2`` instead of ``||da||``; weight stays
                                ``KRABBY_ACTION_RATE_W``'s, stock -0.3)
``KRABBY_ACTION_SMOOTH2_W``     ADDS ``rewards.penalty_action_smooth2`` -- squared second
                                difference of the policy's action history
``KRABBY_JOINT_VEL_W``          ADDS ``rewards.penalty_joint_vel`` -- squared joint speed of
                                the 12 position-actuated (rod) joints
``KRABBY_FOOT_SLIP_W``          ADDS ``rewards.penalty_foot_slip`` -- UNSQUARED planar speed
                                of every foot loaded above 50 N, phase-independent
``KRABBY_STAND_POSE_W``         ADDS ``rewards.penalty_stand_pose`` -- STAND-slot-gated L1
                                deviation of the rod joints from the default pose
``KRABBY_POWER_JOINTS=actuated``  ``rewards.penalty_mechanical_power.params["asset_cfg"]`` ->
                                the 18 actuated joints only (cam shafts + rods; drops the six
                                passive Body_Hip emulator joints). Needs ``KRABBY_POWER_W`` != 0
**incomes (env cfg rewards; weights are the RewTerm weight, >= 0)**
``KRABBY_STAND_STANCE_W``       ADDS ``rewards.reward_stand_stance_quality`` -- the F1 stand
                                term: the stopped-clock stance-quality income, STAND slot only
``KRABBY_STAND_SUPPORT_W``      ADDS ``rewards.reward_stand_support`` -- p2s T1: supported feet
                                beyond a tripod (sum of clamp(|F|/50 N) - 3), STAND slot, upright
**stand-slot penalty (env cfg rewards; weight <= 0)**
``KRABBY_STAND_SLIDE_W``        ADDS ``rewards.penalty_stand_slide`` -- p2s T2: load-weighted 3-D
                                footpad speed (unsquared), STAND slot, upright
**func swaps on EXISTING forward terms (stage p2b; weight and params untouched)**
``KRABBY_ORIENT_ALL_TILES=1``   ``rewards.reward_orientation.func`` -> ``reward_orientation_upright``
                                (tilt^2 on EVERY tile; the stock func zeroes it off parkour_flat)
``KRABBY_FOOT_CLEAR_TERRAIN=1`` the ``func`` of ``rewards.reward_foot_clearance`` and of
                                ``rewards.penalty_swing_min_clearance`` -> their terrain-relative
                                twins in ``mdp/omni_clearance.py`` (foot height above the height
                                field under the foot, not above ``root_z + ground_offset``)
**action path**
``KRABBY_ACTION_LPF_HZ``        ``actions.joint_pos.lpf_hz`` -- first-order low-pass corner
                                (Hz) on the rod-length command, before the rod slew limiter
``KRABBY_ACTION_LPF_OBS=1``     omni obs term ``params["lpf_obs"] = True`` -- the filtered
                                rod command as 12 extra head channels (needs LPF_HZ)
**command / clock**
``KRABBY_OMNI_RELAY_HYST``      ``commands.base_velocity.relay_hyst = (engage, release)``
                                rad; ``1`` = :data:`RELAY_HYST_DEFAULT` (0.25, 0.05)
``KRABBY_OMNI_STAND_HOLD=1``    ``commands.base_velocity.stand_hold = True`` -- applied wz 0
                                (hence clock speed 0) on the STAND slot
**stand regime**
``KRABBY_OMNI_STAND_PARK=1``    ``omni_stand_park = True`` (env cfg) -- STAND slot: clock to
                                the next touchdown, all six legs stance, no swing/apex income
``KRABBY_OMNI_STAND_OBS=1``     omni obs term ``params["stand_obs"] = True`` -- the STAND bit
                                as an extra head channel, clock sin/cos zeroed while it is set
``KRABBY_OMNI_STAND_OBS=clock`` omni obs term ``params["stand_bit_clock"] = True`` (no
                                ``stand_obs``) -- the STAND bit written into the CURRENT step's
                                hidden clock column 15 only (never the history); needs
                                ``KRABBY_OMNI_CLOCK_HIDE=1`` with no ramp; width unchanged (p2r)
**PPO construction (runner cfg; ``apply_exploration_knobs``; PPO-construction plan, 2026-10-01)**
``KRABBY_ACTION_BOUND_LOSS``    ``algorithm.action_bound_loss_coef`` -- the FINAL coefficient c1
                                of the rl_games bound loss on the raw mean (blank / ``0`` = off)
``KRABBY_ACTION_BOUND_RAMP``    ``algorithm.action_bound_loss_ramp = [c0, c1, n]`` -- linear
                                c0 -> c1 over the first n PPO updates (``alg.counter``); needs
                                ``KRABBY_ACTION_BOUND_LOSS`` = c1
``KRABBY_ACTION_BOUND``         ``algorithm.action_bound`` (PPO default 1.1); needs the loss
``KRABBY_STD_PARAM=soft``       ``policy.noise_std_type = "soft"`` -- tanh-bounded log-std in
                                [ln 0.05, ln 2], no hard clamp (``CrabHexActorCriticRMA``)
``KRABBY_INIT_STD``             ``policy.init_noise_std`` (blank / ``0`` = the runner's own)
``KRABBY_PPO_SCHEDULE``         ``algorithm.schedule``: adaptive (stock) | fixed | epoch_kl
``KRABBY_PPO_LR``               ``algorithm.learning_rate``
``KRABBY_PPO_DESIRED_KL``       ``algorithm.desired_kl`` (adaptive / epoch_kl only)
``KRABBY_PPO_KL_STOP``          ``algorithm.kl_stop`` -- the ``fixed`` schedule's early stop
``KRABBY_PPO_HIST_LR``          ``algorithm.hist_encoder_learning_rate`` -- the history
                                encoder's own Adam LR (stock: the configured learning_rate)
``KRABBY_PPO_HAND_DIAG=1``      ``algorithm.hand_diag_vy_obs_col = 9`` -- per-hand surrogate-
                                gradient diagnostics every 25 updates (``PPO/hand_*``)
**statue distillation (runner cfg; ``apply_exploration_knobs``; p2r statue plan, 2026-10-09)**
``KRABBY_ANCHOR_W``             ``algorithm.anchor_coef`` -- the walking anchor's weight (>= 0;
                                blank / ``0`` = off): 1/2 sum_d ((mu - mu_A) / sigma_A,d)^2 on
                                the bit-0 rows, summed / B
``KRABBY_ANCHOR_CKPT``          ``algorithm.anchor_ckpt`` -- the frozen anchor checkpoint (its
                                ``actor.*`` weights give mu_A, its soft std sigma_A)
``KRABBY_STAND_PRIOR_W``        ``algorithm.stand_prior_coef`` -- the statue prior's weight
                                (>= 0): 1/2 sum_d (mu / sigma_A,d)^2 on the bit-1 rows, summed / B
                                (any of the three armed also writes ``algorithm.stand_mask_col =
                                15``; all need ``KRABBY_OMNI_STAND_OBS=clock`` and the checkpoint)
**RSI cam staging (env cfg; ``events.rsi_reference_reset`` params, applied by ``omni_knobs._repoint_rsi``)**
``KRABBY_RSI_CAM_STAGE=1``      ``params["stage_cam"] = True`` -- an RSI-seeded env starts with the
                                bank frame's cam shaft velocity as the applied cam target
                                (``clamp(jv, +-pi)`` rad/s) and in the cam columns of every
                                action-history slot (``clip(jv / pi, +-1)``); rods unchanged
``KRABBY_RSI_CAM_STAGE_DEBUG``  ``=1``: ``params["stage_cam_debug"] = True`` -- one stdout line per reset
                                that seeded RSI envs: count, mean |applied cam target|, mean
                                |history cam columns|, mean |bank cam jv| (the rung-0b step-0 read;
                                with staging off it reads the unstaged zeros)
**driver** (``run_phases.train``; never reaches a cfg)
``KRABBY_NUM_ENVS``             the training command's ``--num_envs`` (stock "256"); audited
                                against ``scene.num_envs`` in ``params/env.yaml``
==============================  ===========================================================

The PPO-construction knobs print ``[KNOB] <NAME>=<value> consumed`` when applied, so a training
log proves its wiring; the algorithm itself prints a ``[PPO] construction:`` line when one of its
knobs arrives. ``KRABBY_STD_PARAM`` and ``KRABBY_INIT_STD`` are the only runner-side knobs an eval
CARRIES (``run_phases.EVAL_CARRY``): ``soft`` changes the policy's PARAMETER STRUCTURE
(``std_raw`` instead of ``std``), so an eval must rebuild the same policy to load the checkpoint.

The omni phase-2 plan step 3 (2026-10-06) added the last two env-side knobs above:
``KRABBY_STAND_STANCE_W`` (F1, an INCOME -- weight >= 0, refused negative) and
``KRABBY_POWER_JOINTS`` (E1, a joint subset for the reused ``KRABBY_POWER_W`` term -- it rewrites
that term's ``asset_cfg`` and is refused when the term's weight is 0, since it would then price
nothing). Both print a ``[KNOB] <NAME>=<value> consumed`` line when armed, like the runner knobs;
the older env-side penalty knobs (``KRABBY_ACTION_SMOOTH2_W`` and siblings) do not.

Omni phase-2 stage p2b (``P2B_PLAN.md`` step 1, 2026-10-08) added the two func swaps above, one per
lever arm: ``KRABBY_ORIENT_ALL_TILES`` (TG, dosed by the forward ``KRABBY_ORIENT_W``) and
``KRABBY_FOOT_CLEAR_TERRAIN`` (CL, dosed by ``KRABBY_FOOT_CLEAR_W`` / ``KRABBY_SWING_MIN_CLEAR_W``).
Each swaps the FUNC of a term the forward rewards cfg already declares and passes its params
through (the terrain-relative twins accept and ignore ``ground_offset_from_root_m``), so the
weight, ``min_clearance_m`` and the ``KRABBY_FOOT_CLEAR_FLAT`` mask stay the forward knobs' own.
Both refuse a cfg that lacks the term and print a ``[KNOB] <NAME>=1 consumed`` line when armed.

Omni phase-2 stage p2s (``P2S_PLAN.md`` step 3, 2026-10-08) added the stand-slot pair, wired exactly
like F1: ``KRABBY_STAND_SUPPORT_W`` (T1, an INCOME -- weight >= 0, refused negative) and
``KRABBY_STAND_SLIDE_W`` (T2, a PENALTY -- weight <= 0, refused positive) each ADD one term from
``mdp/crab_hex_omni_motion_rewards.py`` (gate: the STAND command slot x upright), appended after F1,
and print a ``[KNOB] <NAME>=<value> consumed`` line when armed. They are reward weights, not eval
semantics (an eval never carries them), and they arm no stand slot latch.

Three knobs the plan names ALREADY EXIST in the forward rewards cfg's ``_overrides`` table and
are reused, not duplicated (they reach the omni flat-walk / teacher rewards because
``CrabHexOmniFlatWalkRewardsCfg`` inherits that ``__post_init__``): ``KRABBY_POWER_W``
(``penalty_mechanical_power``, sum |tau * qdot| over all joints), ``KRABBY_ACTION_RATE_W``
(``reward_action_rate``) and ``KRABBY_AIRTIME_W`` (``reward_feet_air_time_positive`` -- the
plan's "AIR_TIME_W"). :data:`REUSED_WEIGHT_KNOBS` names them for the ladder's dump audit.

Both obs-changing knobs (``KRABBY_ACTION_LPF_OBS``, ``KRABBY_OMNI_STAND_OBS=1``) change the
observation WIDTH: a checkpoint trained without them cannot load with them (from-scratch runs
only), and ``crab_hex_omni_mirror.build_mirrors`` grows its head tables from the same params.

Omni phase-2 stage p2r (``P2R_STATUE_PLAN.md`` step 1, 2026-10-09) added the value ``clock`` to
``KRABBY_OMNI_STAND_OBS`` (:func:`parse_stand_obs`: ``0`` / ``1`` / ``clock``): the stand bit goes into
the hidden clock column 15 of the current step (``params["stand_bit_clock"]``), WIDTH-PRESERVING, so
a clock-free head (S1) loads with it -- the grafted head's col-15 weights start at zero. It reuses
the knob already in the eval-locked ``EVAL_CARRY`` (a new name would be dropped from evals), arms the
eval slot latch like ``1``, and is refused without ``KRABBY_OMNI_CLOCK_HIDE=1`` (no ramp) on the obs
term. The same stage added three RUNNER-side knobs (:data:`STATUE_KNOBS`; not carried into evals):
``KRABBY_ANCHOR_W`` / ``KRABBY_ANCHOR_CKPT`` / ``KRABBY_STAND_PRIOR_W`` -> ``PPOWithExtractor``'s
two masked action-mean losses. They are refused unless ``KRABBY_OMNI_STAND_OBS=clock`` (the mask
column would carry no stand bit), with a negative weight, with a weight but no checkpoint (sigma_A
and mu_A come from it), or with a checkpoint but no weight (a recipe naming a loss it does not
train). Each prints ``[KNOB] <NAME>=<value> consumed`` when applied.

Pure at module scope (no Isaac, no torch): the appliers import ``RewTerm`` / ``SceneEntityCfg`` /
the reward functions lazily, and every parser runs in a bare interpreter -- see
``tests/unit/test_crab_hex_omni_motion_knobs.py``.
"""

from __future__ import annotations

import math
import os
from pathlib import Path

# ------------------------------------------------------------------------------ variables
ACTION_STD_MAX_VAR = "KRABBY_ACTION_STD_MAX"
ACTION_STD_RESET_VAR = "KRABBY_ACTION_STD_RESET"
ENTROPY_COEF_VAR = "KRABBY_ENTROPY_COEF"
ACTION_RATE_SQ_VAR = "KRABBY_ACTION_RATE_SQ"
ACTION_SMOOTH2_W_VAR = "KRABBY_ACTION_SMOOTH2_W"
JOINT_VEL_W_VAR = "KRABBY_JOINT_VEL_W"
FOOT_SLIP_W_VAR = "KRABBY_FOOT_SLIP_W"
STAND_POSE_W_VAR = "KRABBY_STAND_POSE_W"
ACTION_LPF_HZ_VAR = "KRABBY_ACTION_LPF_HZ"
ACTION_LPF_OBS_VAR = "KRABBY_ACTION_LPF_OBS"
RELAY_HYST_VAR = "KRABBY_OMNI_RELAY_HYST"
STAND_HOLD_VAR = "KRABBY_OMNI_STAND_HOLD"
STAND_PARK_VAR = "KRABBY_OMNI_STAND_PARK"
STAND_OBS_VAR = "KRABBY_OMNI_STAND_OBS"
# -- omni phase-2 plan step 3 (2026-10-06; both DEFAULT OFF, set by no preset)
#: F1: the STAND-slot stance-quality income (``rewards.reward_stand_stance_quality``).
STAND_STANCE_W_VAR = "KRABBY_STAND_STANCE_W"
#: E1: restrict the reused ``penalty_mechanical_power`` (``KRABBY_POWER_W``) to a joint subset.
POWER_JOINTS_VAR = "KRABBY_POWER_JOINTS"
# -- omni phase-2 stage p2b (P2B_PLAN step 1, 2026-10-08; both DEFAULT OFF, set by no preset)
#: TG: ``reward_orientation`` prices tilt on every tile (``reward_orientation_upright``).
ORIENT_ALL_TILES_VAR = "KRABBY_ORIENT_ALL_TILES"
#: CL: both lift terms measure foot height above the terrain under the foot (``mdp/omni_clearance``).
FOOT_CLEAR_TERRAIN_VAR = "KRABBY_FOOT_CLEAR_TERRAIN"
# -- omni phase-2 stage p2s (P2S_PLAN step 3, 2026-10-08; both DEFAULT OFF, set by no preset)
#: T1: the STAND-slot support income (``rewards.reward_stand_support``, weight >= 0).
STAND_SUPPORT_W_VAR = "KRABBY_STAND_SUPPORT_W"
#: T2: the STAND-slot loaded-foot slide penalty (``rewards.penalty_stand_slide``, weight <= 0).
STAND_SLIDE_W_VAR = "KRABBY_STAND_SLIDE_W"
#: The three EXISTING forward weight knobs the plan reuses (applied by the forward
#: ``CrabHexFlatWalkRewardsCfg.__post_init__`` ``_overrides`` table) -> the term they dose.
POWER_W_VAR = "KRABBY_POWER_W"
ACTION_RATE_W_VAR = "KRABBY_ACTION_RATE_W"
AIRTIME_W_VAR = "KRABBY_AIRTIME_W"
REUSED_WEIGHT_KNOBS = {
    POWER_W_VAR: "penalty_mechanical_power",
    ACTION_RATE_W_VAR: "reward_action_rate",
    AIRTIME_W_VAR: "reward_feet_air_time_positive",
}
# -- PPO construction (PPO-construction plan step 2, 2026-10-01; all DEFAULT OFF, set by no preset)
ACTION_BOUND_LOSS_VAR = "KRABBY_ACTION_BOUND_LOSS"
ACTION_BOUND_RAMP_VAR = "KRABBY_ACTION_BOUND_RAMP"
ACTION_BOUND_VAR = "KRABBY_ACTION_BOUND"
STD_PARAM_VAR = "KRABBY_STD_PARAM"
INIT_STD_VAR = "KRABBY_INIT_STD"
PPO_SCHEDULE_VAR = "KRABBY_PPO_SCHEDULE"
PPO_LR_VAR = "KRABBY_PPO_LR"
PPO_DESIRED_KL_VAR = "KRABBY_PPO_DESIRED_KL"
PPO_KL_STOP_VAR = "KRABBY_PPO_KL_STOP"
PPO_HIST_LR_VAR = "KRABBY_PPO_HIST_LR"
PPO_HAND_DIAG_VAR = "KRABBY_PPO_HAND_DIAG"
#: Driver-side: the omni ``run_phases.train`` command's ``--num_envs`` (applied there, not here).
NUM_ENVS_VAR = "KRABBY_NUM_ENVS"
# -- RSI cam staging (PPO-construction plan step 4, 2026-10-01; DEFAULT OFF, set by no preset). Env
# side: written into the omni RSI reset event's params by ``omni_knobs._repoint_rsi`` (only when the
# event exists, i.e. KRABBY_RSI_FRAC > 0), consumed by the action term's reset.
RSI_CAM_STAGE_VAR = "KRABBY_RSI_CAM_STAGE"
RSI_CAM_STAGE_DEBUG_VAR = "KRABBY_RSI_CAM_STAGE_DEBUG"
#: ``{variable: event param}`` -- each a flag, written as ``True`` when armed and ABSENT when off.
RSI_CAM_STAGE_PARAMS = {RSI_CAM_STAGE_VAR: "stage_cam", RSI_CAM_STAGE_DEBUG_VAR: "stage_cam_debug"}
#: The PPO-construction knobs in ONE tuple (the ladder tests and the dump audit iterate it).
# -- p2r statue distillation (P2R_STATUE_PLAN step 1, 2026-10-09; all DEFAULT OFF, set by no preset)
ANCHOR_W_VAR = "KRABBY_ANCHOR_W"
ANCHOR_CKPT_VAR = "KRABBY_ANCHOR_CKPT"
STAND_PRIOR_W_VAR = "KRABBY_STAND_PRIOR_W"
#: The three runner-side statue knobs (``algorithm.anchor_coef`` / ``anchor_ckpt`` / ``stand_prior_coef``).
STATUE_KNOBS = (ANCHOR_W_VAR, ANCHOR_CKPT_VAR, STAND_PRIOR_W_VAR)
#: ``KRABBY_OMNI_STAND_OBS`` values (blank / ``0`` = off): ``1`` the appended head bit, ``clock`` the bit in
#: the hidden clock column (p2r).
STAND_OBS_MODES = ("1", "clock")
#: The policy observation column ``clock`` mode writes the stand bit into, and ``PPOWithExtractor`` masks on
#: (``CrabHexOmniParkourObservations.STAND_BIT_CLOCK_COL``; pinned by the unit tests).
STAND_BIT_OBS_COL = 15
PPO_CONSTRUCTION_KNOBS = (ACTION_BOUND_LOSS_VAR, ACTION_BOUND_RAMP_VAR, ACTION_BOUND_VAR, STD_PARAM_VAR,
                          INIT_STD_VAR, PPO_SCHEDULE_VAR, PPO_LR_VAR, PPO_DESIRED_KL_VAR, PPO_KL_STOP_VAR,
                          PPO_HIST_LR_VAR, PPO_HAND_DIAG_VAR, NUM_ENVS_VAR)

#: Runner-side knobs (``params/agent.yaml``), everything else is env-side (``params/env.yaml``).
AGENT_KNOBS = (ACTION_STD_MAX_VAR, ACTION_STD_RESET_VAR, ENTROPY_COEF_VAR,
               ACTION_BOUND_LOSS_VAR, ACTION_BOUND_RAMP_VAR, ACTION_BOUND_VAR, STD_PARAM_VAR, INIT_STD_VAR,
               PPO_SCHEDULE_VAR, PPO_LR_VAR, PPO_DESIRED_KL_VAR, PPO_KL_STOP_VAR, PPO_HIST_LR_VAR,
               PPO_HAND_DIAG_VAR, *STATUE_KNOBS)

#: The env-side knobs that change command, clock, action-path or observation SEMANTICS: an eval
#: of a policy trained with one of them must be built with it too (``run_phases.EVAL_CARRY``).
EVAL_SEMANTIC_KNOBS = (RELAY_HYST_VAR, STAND_HOLD_VAR, STAND_PARK_VAR, ACTION_LPF_HZ_VAR,
                       ACTION_LPF_OBS_VAR, STAND_OBS_VAR)
#: The RUNNER-side knobs an eval must carry too: they change the policy's PARAMETER STRUCTURE
#: (``soft`` std: ``std_raw`` instead of ``std``), and an eval rebuilds the policy from the env
#: knobs before it loads the checkpoint (strictly). ``KRABBY_INIT_STD`` is carried with it (the
#: soft parameter is initialised through it; the load then overwrites it).
EVAL_STRUCTURE_KNOBS = (STD_PARAM_VAR, INIT_STD_VAR)
#: ... and the value each one takes when OFF (what ``EVAL_CARRY_DEFAULTS`` restates).
KNOB_OFF_VALUES = {RELAY_HYST_VAR: "0", STAND_HOLD_VAR: "0", STAND_PARK_VAR: "0",
                   ACTION_LPF_HZ_VAR: "0", ACTION_LPF_OBS_VAR: "0", STAND_OBS_VAR: "0",
                   STD_PARAM_VAR: "0", INIT_STD_VAR: "0"}
#: The omni POLICY observation column of the body-frame lateral (vy) command:
#: ``CrabHexOmniParkourObservations.OBSERVED_COMMAND_SLOTS`` maps slot 9 -> command index 1, and the
#: current step's proprio head opens the policy observation (pinned by the unit tests).
HAND_DIAG_VY_OBS_COL = 9
PPO_SCHEDULES = ("adaptive", "fixed", "epoch_kl")
STD_PARAM_MODES = ("soft",)
STD_PARAM_OFF_SPELLINGS = ("0", "off")

#: ``(engage, release)`` rad of facing error for ``KRABBY_OMNI_RELAY_HYST=1``.
RELAY_HYST_DEFAULT = (0.25, 0.05)
#: ``CrabHexActorCriticRMA``'s stock clamp floor: a std cap / reset below it is meaningless.
ACTION_STD_MIN = 0.05
#: ... and its stock ceiling (``ACTION_STD_MAX`` there), in force while the knob is unset.
STOCK_ACTION_STD_MAX = 2.0
#: Upper bound on the filter corner: the rod command is filtered at the PHYSICS rate (200 Hz on
#: the shipped sim.dt 0.005), so anything at or past its Nyquist frequency is not a low-pass.
LPF_HZ_MAX = 100.0
#: The "loaded foot" threshold of ``KRABBY_FOOT_SLIP_W`` -- the clock reward's ``TOUCH_REF_N``.
FOOT_SLIP_FORCE_N = 50.0

#: Names of the reward terms the weight knobs ADD (``rewards.<name>``).
SMOOTH2_TERM = "penalty_action_smooth2"
JOINT_VEL_TERM = "penalty_joint_vel"
FOOT_SLIP_TERM = "penalty_foot_slip"
STAND_POSE_TERM = "penalty_stand_pose"
#: ... and the INCOME term ``KRABBY_STAND_STANCE_W`` adds (phase-2 plan step 3, F1).
STAND_STANCE_TERM = "reward_stand_stance_quality"
#: ... and the p2s stand-slot pair (P2S_PLAN step 3, 2026-10-08): T1 an INCOME, T2 a penalty.
STAND_SUPPORT_TERM = "reward_stand_support"
STAND_SLIDE_TERM = "penalty_stand_slide"
#: Their shared params: the per-foot support saturates at the eval's 50 N "loaded" line, and the
#: upright gate is F1's (``-g_z > 0.9``).
STAND_LOAD_REF_N = 50.0
STAND_UPRIGHT_MIN = 0.9
#: The EXISTING forward term ``KRABBY_POWER_JOINTS`` re-scopes (``KRABBY_POWER_W`` doses it).
POWER_TERM = "penalty_mechanical_power"
#: ``KRABBY_POWER_JOINTS`` spellings: the subsets it can select (``"all"`` / blank = OFF, the stock
#: every-joint sum). ``actuated`` = the 18 joints of the forward cfg's
#: ``_CRAB_ACTUATED_JOINT_NAMES`` (6 ``*_Body_CamShaft_`` + 6 ``*_Hip_Femur_`` + 6
#: ``*_Femur_Tibia_RevoluteJoint``); the 6 ``*_Body_Hip_RevoluteJoint`` are the passive joints the
#: cam linkage emulator drives (``crab_hex_cam_mapping``), 24-31 % of the all-joint power.
POWER_JOINT_SETS = ("actuated",)
POWER_JOINTS_OFF_SPELLINGS = ("all",)
#: The EXISTING forward terms the p2b knobs re-point (``rewards.<name>``) ...
ORIENT_TERM = "reward_orientation"
FOOT_CLEAR_TERM = "reward_foot_clearance"
SWING_MIN_CLEAR_TERM = "penalty_swing_min_clearance"
#: ... and ``{knob: {term: replacement func name}}``: ``reward_orientation_upright`` is the library's
#: (``_LIB_REWARDS_MOD``), the two lift twins are ``_CLEARANCE_MOD``'s.
FUNC_SWAPS = {
    ORIENT_ALL_TILES_VAR: {ORIENT_TERM: "reward_orientation_upright"},
    FOOT_CLEAR_TERRAIN_VAR: {FOOT_CLEAR_TERM: "reward_foot_clearance_terrain",
                             SWING_MIN_CLEAR_TERM: "penalty_swing_min_clearance_terrain"},
}
#: Where the stand term's function lives (the Isaac-side rewards module, next to the clock term
#: whose stopped branch it copies).
_LIB_REWARDS_MOD = "parkour_isaaclab.envs.mdp.rewards"

_REWARDS_MOD = "parkour_tasks.crab_hex_omni_task.mdp.crab_hex_omni_motion_rewards"
_CLEARANCE_MOD = "parkour_tasks.crab_hex_omni_task.mdp.omni_clearance"

__all__ = [
    "ACTION_STD_MAX_VAR", "ACTION_STD_RESET_VAR", "ENTROPY_COEF_VAR", "ACTION_RATE_SQ_VAR",
    "ACTION_SMOOTH2_W_VAR", "JOINT_VEL_W_VAR", "FOOT_SLIP_W_VAR", "STAND_POSE_W_VAR",
    "ACTION_LPF_HZ_VAR", "ACTION_LPF_OBS_VAR", "RELAY_HYST_VAR", "STAND_HOLD_VAR",
    "STAND_PARK_VAR", "STAND_OBS_VAR", "POWER_W_VAR", "ACTION_RATE_W_VAR", "AIRTIME_W_VAR",
    "REUSED_WEIGHT_KNOBS", "AGENT_KNOBS", "EVAL_SEMANTIC_KNOBS", "KNOB_OFF_VALUES",
    "RELAY_HYST_DEFAULT", "parse_flag", "parse_std_max", "parse_std_reset",
    "parse_entropy_coef", "parse_penalty_w", "parse_lpf_hz", "parse_relay_hyst",
    "MOTION_KNOB_PARSERS", "apply_motion_knobs", "apply_exploration_knobs",
    # PPO construction (2026-10-01)
    "ACTION_BOUND_LOSS_VAR", "ACTION_BOUND_RAMP_VAR", "ACTION_BOUND_VAR", "STD_PARAM_VAR", "INIT_STD_VAR",
    "PPO_SCHEDULE_VAR", "PPO_LR_VAR", "PPO_DESIRED_KL_VAR", "PPO_KL_STOP_VAR", "PPO_HIST_LR_VAR",
    "PPO_HAND_DIAG_VAR", "NUM_ENVS_VAR", "PPO_CONSTRUCTION_KNOBS", "EVAL_STRUCTURE_KNOBS",
    "HAND_DIAG_VY_OBS_COL", "PPO_SCHEDULES", "STD_PARAM_MODES", "STD_PARAM_OFF_SPELLINGS",
    "parse_bound_loss", "parse_bound_ramp", "parse_action_bound", "parse_std_param", "parse_init_std",
    "parse_ppo_schedule", "parse_positive", "parse_num_envs",
    # RSI cam staging (2026-10-01)
    "RSI_CAM_STAGE_VAR", "RSI_CAM_STAGE_DEBUG_VAR", "RSI_CAM_STAGE_PARAMS", "rsi_cam_stage_params",
    # omni phase-2 plan step 3 (2026-10-06)
    "STAND_STANCE_W_VAR", "POWER_JOINTS_VAR", "STAND_STANCE_TERM", "POWER_TERM", "POWER_JOINT_SETS",
    "POWER_JOINTS_OFF_SPELLINGS", "parse_income_w", "parse_power_joints",
    # omni phase-2 stage p2b (2026-10-08)
    "ORIENT_ALL_TILES_VAR", "FOOT_CLEAR_TERRAIN_VAR", "ORIENT_TERM", "FOOT_CLEAR_TERM",
    "SWING_MIN_CLEAR_TERM", "FUNC_SWAPS",
    # omni phase-2 stage p2s (2026-10-08)
    "STAND_SUPPORT_W_VAR", "STAND_SLIDE_W_VAR", "STAND_SUPPORT_TERM", "STAND_SLIDE_TERM",
    "STAND_LOAD_REF_N", "STAND_UPRIGHT_MIN",
    # omni phase-2 stage p2r (2026-10-09)
    "ANCHOR_W_VAR", "ANCHOR_CKPT_VAR", "STAND_PRIOR_W_VAR", "STATUE_KNOBS", "STAND_OBS_MODES",
    "STAND_BIT_OBS_COL", "parse_stand_obs", "parse_statue_w", "parse_anchor_ckpt",
]


# ------------------------------------------------------------------------------ parsers
def _blank(text) -> bool:
    return text is None or not str(text).strip()


def _float(var: str, text) -> float:
    try:
        value = float(str(text).strip())
    except ValueError as exc:
        raise ValueError(f"{var} must be a number, got {text!r}") from exc
    if not math.isfinite(value):
        raise ValueError(f"{var} must be finite, got {text!r}")
    return value


def parse_flag(text, var: str = "flag") -> bool:
    """``"1"`` -> True, ``"0"`` / blank -> False; anything else raises (a typo must not pick)."""
    if _blank(text):
        return False
    token = str(text).strip()
    if token == "1":
        return True
    if token == "0":
        return False
    raise ValueError(f"{var} must be '0' or '1', got {text!r}")


def parse_stand_obs(text) -> str | None:
    """``KRABBY_OMNI_STAND_OBS`` -> ``"1"`` / ``"clock"``, or None for OFF (blank / ``0``); anything else raises."""
    if _blank(text):
        return None
    token = str(text).strip().lower()
    if token == "0":
        return None
    if token in STAND_OBS_MODES:
        return token
    raise ValueError(f"{STAND_OBS_VAR} must be '0', '1' or 'clock', got {text!r}")


def parse_statue_w(text, var: str = "statue weight") -> float | None:
    """``KRABBY_ANCHOR_W`` / ``KRABBY_STAND_PRIOR_W`` -> a weight > 0, or None for OFF (blank / ``0``).

    Negative values are refused (a negative quadratic would PAY the mean for leaving the anchor / the statue).
    """
    if _blank(text):
        return None
    value = _float(var, text)
    if value < 0.0:
        raise ValueError(f"{var} must be >= 0 (0 = off), got {text!r}")
    return None if value == 0.0 else value


def parse_anchor_ckpt(text) -> str | None:
    """``KRABBY_ANCHOR_CKPT`` -> the checkpoint path (stripped), or None (blank). Existence is checked at
    application (:func:`apply_exploration_knobs`), not here: the ladder round-trips values at dry run."""
    if _blank(text):
        return None
    return str(text).strip()


def resolve_anchor_ckpt(path: str) -> str:
    """A repo-relative ``KRABBY_ANCHOR_CKPT`` (as candidates.yaml carries it) resolved against the repo root, so the
    training child (launched with cwd=parkour/) finds it (fable review 2026-10-09); absolute paths pass through."""
    if os.path.isabs(path):
        return path
    return str(Path(__file__).resolve().parents[6] / path)


def parse_std_max(text) -> float | None:
    """``KRABBY_ACTION_STD_MAX`` -> the std clamp ceiling, or None (blank = stock 2.0)."""
    if _blank(text):
        return None
    value = _float(ACTION_STD_MAX_VAR, text)
    if not value > ACTION_STD_MIN:
        raise ValueError(f"{ACTION_STD_MAX_VAR} must exceed the clamp floor {ACTION_STD_MIN}, got {text!r}")
    return value


def parse_std_reset(text) -> float | None:
    """``KRABBY_ACTION_STD_RESET`` -> the std written on checkpoint load, or None (blank = keep)."""
    if _blank(text):
        return None
    value = _float(ACTION_STD_RESET_VAR, text)
    if not value >= ACTION_STD_MIN:
        raise ValueError(f"{ACTION_STD_RESET_VAR} must be >= {ACTION_STD_MIN}, got {text!r}")
    return value


def parse_entropy_coef(text) -> float | None:
    """``KRABBY_ENTROPY_COEF`` -> PPO's entropy bonus coefficient (>= 0), or None (blank)."""
    if _blank(text):
        return None
    value = _float(ENTROPY_COEF_VAR, text)
    if value < 0.0:
        raise ValueError(f"{ENTROPY_COEF_VAR} must be >= 0, got {text!r}")
    return value


def parse_penalty_w(text, var: str = "penalty weight") -> float:
    """A PENALTY weight: the RewTerm weight, sign included, so it must be <= 0 (blank -> 0).

    A positive value would PAY for the very motion the term exists to suppress; refused rather
    than sign-flipped so the recipe says exactly what trains. ``0`` means OFF (no term added).
    """
    if _blank(text):
        return 0.0
    value = _float(var, text)
    if value > 0.0:
        raise ValueError(f"{var} is a penalty weight and must be <= 0, got {text!r}")
    return value


def parse_income_w(text, var: str = "income weight") -> float:
    """An INCOME weight: the RewTerm weight, so it must be >= 0 (blank -> 0 = OFF, no term added).

    The mirror of :func:`parse_penalty_w`: a negative value would PENALISE the stillness the term
    exists to pay for; refused rather than sign-flipped.
    """
    if _blank(text):
        return 0.0
    value = _float(var, text)
    if value < 0.0:
        raise ValueError(f"{var} is an income weight and must be >= 0, got {text!r}")
    return value


def parse_power_joints(text) -> str | None:
    """``KRABBY_POWER_JOINTS`` -> ``"actuated"``, or None for OFF (blank / ``all`` = every joint)."""
    if _blank(text):
        return None
    token = str(text).strip().lower()
    if token in POWER_JOINTS_OFF_SPELLINGS:
        return None
    if token in POWER_JOINT_SETS:
        return token
    raise ValueError(f"{POWER_JOINTS_VAR} must be one of {POWER_JOINT_SETS} or off "
                     f"({POWER_JOINTS_OFF_SPELLINGS} / blank), got {text!r}")


def parse_lpf_hz(text) -> float | None:
    """``KRABBY_ACTION_LPF_HZ`` -> the corner frequency (Hz), or None for OFF (blank / ``0``)."""
    if _blank(text):
        return None
    value = _float(ACTION_LPF_HZ_VAR, text)
    if value == 0.0:
        return None
    if not 0.0 < value < LPF_HZ_MAX:
        raise ValueError(f"{ACTION_LPF_HZ_VAR} must be in (0, {LPF_HZ_MAX}) Hz or 0 (off), got {text!r}")
    return value


def parse_relay_hyst(text) -> tuple[float, float] | None:
    """``KRABBY_OMNI_RELAY_HYST`` -> ``(engage, release)`` rad, or None for OFF.

    ``""`` / ``"0"`` / ``"off"`` -> None; ``"1"`` -> :data:`RELAY_HYST_DEFAULT`; ``"e:r"`` -> an
    explicit pair with ``0 <= r < e`` (release >= engage would chatter exactly like the dead band
    it replaces).
    """
    if _blank(text):
        return None
    token = str(text).strip().lower()
    if token in ("0", "off", "none"):
        return None
    if token == "1":
        return RELAY_HYST_DEFAULT
    parts = token.split(":")
    if len(parts) != 2:
        raise ValueError(f"{RELAY_HYST_VAR} must be 0, 1 or 'engage:release' (rad), got {text!r}")
    engage, release = (_float(RELAY_HYST_VAR, p) for p in parts)
    if not 0.0 <= release < engage:
        raise ValueError(f"{RELAY_HYST_VAR} needs 0 <= release < engage, got {text!r}")
    return (engage, release)


# -- PPO construction (2026-10-01)
def parse_bound_loss(text) -> float | None:
    """``KRABBY_ACTION_BOUND_LOSS`` -> the bound loss's (final) coefficient, or None for OFF.

    Blank / ``0`` = off (nothing written, the loss is never built); negative values are refused.
    """
    if _blank(text):
        return None
    value = _float(ACTION_BOUND_LOSS_VAR, text)
    if value < 0.0:
        raise ValueError(f"{ACTION_BOUND_LOSS_VAR} must be >= 0 (0 = off), got {text!r}")
    return None if value == 0.0 else value


def parse_bound_ramp(text) -> tuple[float, float, int] | None:
    """``KRABBY_ACTION_BOUND_RAMP`` -> ``(c0, c1, n)``, or None (blank = constant coefficient).

    ``"c0:c1:n"``: linear from ``c0`` to ``c1`` over the first ``n`` PPO updates, ``c0 >= 0``,
    ``c1 > 0``, ``n`` a positive integer.
    """
    if _blank(text):
        return None
    parts = str(text).strip().split(":")
    if len(parts) != 3:
        raise ValueError(f"{ACTION_BOUND_RAMP_VAR} must be 'c0:c1:n', got {text!r}")
    c0, c1 = _float(ACTION_BOUND_RAMP_VAR, parts[0]), _float(ACTION_BOUND_RAMP_VAR, parts[1])
    try:
        n = int(parts[2].strip())
    except ValueError as exc:
        raise ValueError(f"{ACTION_BOUND_RAMP_VAR}: n must be a positive integer, got {text!r}") from exc
    if not (c0 >= 0.0 and c1 > 0.0 and n >= 1):
        raise ValueError(f"{ACTION_BOUND_RAMP_VAR} needs c0 >= 0, c1 > 0 and n >= 1, got {text!r}")
    return (c0, c1, n)


def parse_action_bound(text) -> float | None:
    """``KRABBY_ACTION_BOUND`` -> the hinge's bound on the raw mean (> 0), or None (blank = 1.1)."""
    if _blank(text):
        return None
    value = _float(ACTION_BOUND_VAR, text)
    if not value > 0.0:
        raise ValueError(f"{ACTION_BOUND_VAR} must be > 0, got {text!r}")
    return value


def parse_std_param(text) -> str | None:
    """``KRABBY_STD_PARAM`` -> ``"soft"``, or None for OFF (blank / ``0`` / ``off``)."""
    if _blank(text):
        return None
    token = str(text).strip().lower()
    if token in STD_PARAM_OFF_SPELLINGS:
        return None
    if token in STD_PARAM_MODES:
        return token
    raise ValueError(f"{STD_PARAM_VAR} must be one of {STD_PARAM_MODES} or off ({STD_PARAM_OFF_SPELLINGS}), "
                     f"got {text!r}")


def parse_init_std(text) -> float | None:
    """``KRABBY_INIT_STD`` -> ``policy.init_noise_std`` (> the 0.05 floor), or None (blank / ``0``)."""
    if _blank(text):
        return None
    value = _float(INIT_STD_VAR, text)
    if value == 0.0:
        return None
    if not value > ACTION_STD_MIN:
        raise ValueError(f"{INIT_STD_VAR} must exceed the std floor {ACTION_STD_MIN} (0 = off), got {text!r}")
    return value


def parse_ppo_schedule(text) -> str | None:
    """``KRABBY_PPO_SCHEDULE`` -> one of :data:`PPO_SCHEDULES`, or None (blank = the cfg's own)."""
    if _blank(text):
        return None
    token = str(text).strip()
    if token not in PPO_SCHEDULES:
        raise ValueError(f"{PPO_SCHEDULE_VAR} must be one of {PPO_SCHEDULES}, got {text!r}")
    return token


def parse_positive(text, var: str = "value") -> float | None:
    """A strictly positive, finite number, or None (blank) -- the LR / KL knobs."""
    if _blank(text):
        return None
    value = _float(var, text)
    if not value > 0.0:
        raise ValueError(f"{var} must be > 0, got {text!r}")
    return value


def parse_num_envs(text) -> int | None:
    """``KRABBY_NUM_ENVS`` -> a positive int, or None (blank = the driver's stock 256)."""
    if _blank(text):
        return None
    try:
        value = int(str(text).strip())
    except ValueError as exc:
        raise ValueError(f"{NUM_ENVS_VAR} must be a positive integer, got {text!r}") from exc
    if value < 1:
        raise ValueError(f"{NUM_ENVS_VAR} must be a positive integer, got {text!r}")
    return value


#: ``{variable: parser}`` for every knob this module owns (the ladder's round-trip check).
MOTION_KNOB_PARSERS = {
    ACTION_STD_MAX_VAR: parse_std_max,
    ACTION_STD_RESET_VAR: parse_std_reset,
    ENTROPY_COEF_VAR: parse_entropy_coef,
    ACTION_RATE_SQ_VAR: lambda t: parse_flag(t, ACTION_RATE_SQ_VAR),
    ACTION_SMOOTH2_W_VAR: lambda t: parse_penalty_w(t, ACTION_SMOOTH2_W_VAR),
    JOINT_VEL_W_VAR: lambda t: parse_penalty_w(t, JOINT_VEL_W_VAR),
    FOOT_SLIP_W_VAR: lambda t: parse_penalty_w(t, FOOT_SLIP_W_VAR),
    STAND_POSE_W_VAR: lambda t: parse_penalty_w(t, STAND_POSE_W_VAR),
    ACTION_LPF_HZ_VAR: parse_lpf_hz,
    ACTION_LPF_OBS_VAR: lambda t: parse_flag(t, ACTION_LPF_OBS_VAR),
    RELAY_HYST_VAR: parse_relay_hyst,
    STAND_HOLD_VAR: lambda t: parse_flag(t, STAND_HOLD_VAR),
    STAND_PARK_VAR: lambda t: parse_flag(t, STAND_PARK_VAR),
    STAND_OBS_VAR: parse_stand_obs,
    # PPO construction (2026-10-01)
    ACTION_BOUND_LOSS_VAR: parse_bound_loss,
    ACTION_BOUND_RAMP_VAR: parse_bound_ramp,
    ACTION_BOUND_VAR: parse_action_bound,
    STD_PARAM_VAR: parse_std_param,
    INIT_STD_VAR: parse_init_std,
    PPO_SCHEDULE_VAR: parse_ppo_schedule,
    PPO_LR_VAR: lambda t: parse_positive(t, PPO_LR_VAR),
    PPO_DESIRED_KL_VAR: lambda t: parse_positive(t, PPO_DESIRED_KL_VAR),
    PPO_KL_STOP_VAR: lambda t: parse_positive(t, PPO_KL_STOP_VAR),
    PPO_HIST_LR_VAR: lambda t: parse_positive(t, PPO_HIST_LR_VAR),
    PPO_HAND_DIAG_VAR: lambda t: parse_flag(t, PPO_HAND_DIAG_VAR),
    NUM_ENVS_VAR: parse_num_envs,
    # RSI cam staging (2026-10-01)
    RSI_CAM_STAGE_VAR: lambda t: parse_flag(t, RSI_CAM_STAGE_VAR),
    RSI_CAM_STAGE_DEBUG_VAR: lambda t: parse_flag(t, RSI_CAM_STAGE_DEBUG_VAR),
    # omni phase-2 plan step 3 (2026-10-06)
    STAND_STANCE_W_VAR: lambda t: parse_income_w(t, STAND_STANCE_W_VAR),
    POWER_JOINTS_VAR: parse_power_joints,
    # omni phase-2 stage p2b (2026-10-08)
    ORIENT_ALL_TILES_VAR: lambda t: parse_flag(t, ORIENT_ALL_TILES_VAR),
    FOOT_CLEAR_TERRAIN_VAR: lambda t: parse_flag(t, FOOT_CLEAR_TERRAIN_VAR),
    # omni phase-2 stage p2s (2026-10-08)
    STAND_SUPPORT_W_VAR: lambda t: parse_income_w(t, STAND_SUPPORT_W_VAR),
    STAND_SLIDE_W_VAR: lambda t: parse_penalty_w(t, STAND_SLIDE_W_VAR),
    # omni phase-2 stage p2r (2026-10-09): runner side
    ANCHOR_W_VAR: lambda t: parse_statue_w(t, ANCHOR_W_VAR),
    ANCHOR_CKPT_VAR: parse_anchor_ckpt,
    STAND_PRIOR_W_VAR: lambda t: parse_statue_w(t, STAND_PRIOR_W_VAR),
}


def rsi_cam_stage_params(environ) -> dict:
    """The RSI reset event's cam-staging params: ``{param: True}`` for each ARMED flag, else ``{}``.

    Off (unset, blank or ``"0"``) adds nothing, so a knob-off ``params/env.yaml`` is byte-identical
    to one built before the knob existed. A bad spelling raises (``parse_flag``), at config build.
    """
    return {param: True for var, param in RSI_CAM_STAGE_PARAMS.items() if parse_flag(environ.get(var), var)}


# ------------------------------------------------------------------------------ application
def _omni_obs_terms(cfg):
    """Every observation term whose func is the omni observation class (by attribute)."""
    found = []
    groups = getattr(cfg, "observations", None)
    for gname in ("policy", "critic"):
        group = getattr(groups, gname, None)
        if group is None:
            continue
        for tname, term in vars(group).items():
            if hasattr(getattr(term, "func", None), "OBSERVED_COMMAND_SLOTS"):
                found.append((gname, tname, term))
    return found


def _set_obs_param(cfg, var: str, key: str) -> None:
    terms = _omni_obs_terms(cfg)
    if not terms:
        raise RuntimeError(f"{var} is set but no omni observation term is installed")
    for _g, _t, term in terms:
        if term.params is None:
            term.params = {}
        term.params[key] = True


def _rewards(cfg, var: str):
    rewards = getattr(cfg, "rewards", None)
    if rewards is None:
        raise RuntimeError(f"{var} is set but this env cfg has no rewards block")
    return rewards


def _default_term_builders():
    """``{name: (func, params)}`` for the ADDED terms -- the four penalties, (phase-2 plan step 3)
    the F1 stand income and (stage p2s) the stand-slot support / slide pair -- with lazy Isaac
    imports."""
    import importlib  # noqa: PLC0415

    from isaaclab.managers import SceneEntityCfg  # noqa: PLC0415

    from parkour_tasks.crab_hex_forward_task.config.crab_hex.agents.parkour_mdp_cfg import (  # noqa: PLC0415
        _CRAB_FOOT_BODY_NAMES,
        _CRAB_POSITION_ACTUATED_JOINT_NAMES,
    )

    rew = importlib.import_module(_REWARDS_MOD)
    lib_rew = importlib.import_module(_LIB_REWARDS_MOD)
    rods = SceneEntityCfg("robot", joint_names=_CRAB_POSITION_ACTUATED_JOINT_NAMES, preserve_order=True)
    return {
        SMOOTH2_TERM: (rew.penalty_action_second_diff, {"action_name": "joint_pos"}),
        JOINT_VEL_TERM: (rew.penalty_joint_vel_l2, {"asset_cfg": rods}),
        FOOT_SLIP_TERM: (rew.penalty_loaded_foot_slip, {
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=_CRAB_FOOT_BODY_NAMES,
                                         preserve_order=True),
            "asset_cfg": SceneEntityCfg("robot", body_names=_CRAB_FOOT_BODY_NAMES, preserve_order=True),
            "force_threshold": FOOT_SLIP_FORCE_N,
        }),
        STAND_POSE_TERM: (rew.penalty_stand_pose_l1, {
            "asset_cfg": SceneEntityCfg("robot", joint_names=_CRAB_POSITION_ACTUATED_JOINT_NAMES,
                                        preserve_order=True),
            "command_name": "base_velocity",
        }),
        # F1 (phase-2 plan step 3, 2026-10-06): the clock term's stopped-branch inputs (footpad
        # contact force from the same sensor, planar footpad speed), its vel_ref and upright gate.
        STAND_STANCE_TERM: (lib_rew.reward_stand_stance_quality, {
            "asset_cfg": SceneEntityCfg("robot", body_names=_CRAB_FOOT_BODY_NAMES, preserve_order=True),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=_CRAB_FOOT_BODY_NAMES,
                                         preserve_order=True),
            "command_name": "base_velocity",
            "vel_ref": lib_rew.CLOCK_VEL_REF_M_S,
            "min_upright_gz": 0.9,
        }),
        # p2s (P2S_PLAN step 3, 2026-10-08): both terms take the same params -- the six footpads'
        # current contact force (sensor) and 3-D speed (robot), the robot's projected gravity for the
        # upright gate, the 50 N support saturation and F1's 0.9 upright line.
        STAND_SUPPORT_TERM: (rew.reward_stand_support, _stand_slot_params(SceneEntityCfg, _CRAB_FOOT_BODY_NAMES)),
        STAND_SLIDE_TERM: (rew.penalty_stand_slide, _stand_slot_params(SceneEntityCfg, _CRAB_FOOT_BODY_NAMES)),
    }


def _stand_slot_params(entity_cls, foot_names) -> dict:
    """The p2s pair's params (``entity_cls`` is ``SceneEntityCfg``, ``foot_names`` the six footpads)."""
    return {
        "sensor_cfg": entity_cls("contact_forces", body_names=foot_names, preserve_order=True),
        "asset_cfg": entity_cls("robot", body_names=foot_names, preserve_order=True),
        "command_name": "base_velocity",
        "load_ref_n": STAND_LOAD_REF_N,
        "upright_min": STAND_UPRIGHT_MIN,
    }


def _default_power_joint_cfgs():
    """``{subset: asset_cfg}`` for ``KRABBY_POWER_JOINTS`` (lazy Isaac imports).

    ``actuated`` names the forward cfg's ``_CRAB_ACTUATED_JOINT_NAMES`` regexes (cam shafts +
    Hip_Femur + Femur_Tibia: 18 of the 24 joints); the order is irrelevant to a sum.
    """
    from isaaclab.managers import SceneEntityCfg  # noqa: PLC0415

    from parkour_tasks.crab_hex_forward_task.config.crab_hex.agents.parkour_mdp_cfg import (  # noqa: PLC0415
        _CRAB_ACTUATED_JOINT_NAMES,
    )

    return {"actuated": SceneEntityCfg("robot", joint_names=list(_CRAB_ACTUATED_JOINT_NAMES))}


def _default_swap_funcs(var: str) -> dict:
    """``{term: func}`` for one armed p2b swap knob (:data:`FUNC_SWAPS`), with lazy Isaac imports."""
    import importlib  # noqa: PLC0415

    mod = importlib.import_module(_LIB_REWARDS_MOD if var == ORIENT_ALL_TILES_VAR else _CLEARANCE_MOD)
    return {term: getattr(mod, name) for term, name in FUNC_SWAPS[var].items()}


def apply_motion_knobs(cfg, environ=None, *, term_cls=None, builders=None, power_joint_cfgs=None,
                       swap_funcs=None) -> None:
    """Apply every ENV-side motion knob to a built omni env cfg (no-op for each one that is off).

    Called by ``omni_knobs.apply_omni_knobs`` after everything else, so a term it adds sees the
    final cfg. ``term_cls`` / ``builders`` / ``power_joint_cfgs`` / ``swap_funcs`` (``{term: func}``)
    let a unit test pass stand-ins for ``RewTerm``, the reward functions and the ``SceneEntityCfg``
    subsets (the real ones need Isaac).
    """
    env = os.environ if environ is None else environ
    get = env.get

    # -- regularisers
    if parse_flag(get(ACTION_RATE_SQ_VAR), ACTION_RATE_SQ_VAR):
        term = getattr(_rewards(cfg, ACTION_RATE_SQ_VAR), "reward_action_rate", None)
        if term is None or not isinstance(getattr(term, "params", None), dict):
            raise RuntimeError(f"{ACTION_RATE_SQ_VAR}=1 but the rewards cfg has no reward_action_rate term")
        term.params["squared"] = True
    added = [(var, name, parse_penalty_w(get(var), var)) for var, name in (
        (ACTION_SMOOTH2_W_VAR, SMOOTH2_TERM), (JOINT_VEL_W_VAR, JOINT_VEL_TERM),
        (FOOT_SLIP_W_VAR, FOOT_SLIP_TERM), (STAND_POSE_W_VAR, STAND_POSE_TERM))]
    # F1 (phase-2 plan step 3, 2026-10-06): an INCOME, so its own sign rule; appended last so a
    # knob-on dump only grows by this one term at the end of the rewards block.
    added.append((STAND_STANCE_W_VAR, STAND_STANCE_TERM, parse_income_w(get(STAND_STANCE_W_VAR),
                                                                        STAND_STANCE_W_VAR)))
    # p2s (P2S_PLAN step 3, 2026-10-08): the stand-slot pair, each with its own sign rule, appended
    # after F1 for the same reason. Rewards only: neither arms the eval slot latch below.
    added.append((STAND_SUPPORT_W_VAR, STAND_SUPPORT_TERM, parse_income_w(get(STAND_SUPPORT_W_VAR),
                                                                          STAND_SUPPORT_W_VAR)))
    added.append((STAND_SLIDE_W_VAR, STAND_SLIDE_TERM, parse_penalty_w(get(STAND_SLIDE_W_VAR),
                                                                       STAND_SLIDE_W_VAR)))
    added = [a for a in added if a[2] != 0.0]
    if added:
        if term_cls is None:
            from isaaclab.managers import RewardTermCfg as term_cls  # noqa: PLC0415,N813
        table = builders if builders is not None else _default_term_builders()
        for var, name, weight in added:
            rewards = _rewards(cfg, var)
            if getattr(rewards, name, None) is not None:
                raise RuntimeError(f"{var}: the rewards cfg already carries a {name!r} term")
            func, params = table[name]
            setattr(rewards, name, term_cls(func=func, weight=weight, params=dict(params)))
            # Only F1 and the p2s pair print the wiring line: the four penalty knobs above predate
            # it (2026-09-22) and keep their old, silent stdout -- a log grep must not expect
            # ``[KNOB] ... consumed`` for KRABBY_ACTION_SMOOTH2_W / JOINT_VEL_W / FOOT_SLIP_W /
            # STAND_POSE_W (review 2026-10-06); their wiring is proven by the env.yaml dump instead.
            if name in (STAND_STANCE_TERM, STAND_SUPPORT_TERM, STAND_SLIDE_TERM):
                _consumed(var, get(var))

    # E1 (phase-2 plan step 3, 2026-10-06): re-scope the REUSED power term (its weight is
    # KRABBY_POWER_W's, applied by the forward rewards cfg's __post_init__ before this runs).
    power_joints = parse_power_joints(get(POWER_JOINTS_VAR))
    if power_joints is not None:
        term = getattr(_rewards(cfg, POWER_JOINTS_VAR), POWER_TERM, None)
        if term is None or not isinstance(getattr(term, "params", None), dict):
            raise RuntimeError(f"{POWER_JOINTS_VAR}={power_joints} but the rewards cfg has no {POWER_TERM} term")
        if not term.weight:
            raise RuntimeError(f"{POWER_JOINTS_VAR}={power_joints} re-scopes {POWER_TERM}, whose weight is "
                               f"{term.weight!r}: set {POWER_W_VAR} (a recipe must not name a subset it "
                               "does not price)")
        subsets = power_joint_cfgs if power_joint_cfgs is not None else _default_power_joint_cfgs()
        term.params["asset_cfg"] = subsets[power_joints]
        _consumed(POWER_JOINTS_VAR, get(POWER_JOINTS_VAR))

    # p2b (P2B_PLAN step 1, 2026-10-08): func swaps on EXISTING forward terms. This runs after the
    # forward rewards cfg's __post_init__, so the weight / min_clearance_m / FOOT_CLEAR_FLAT mask
    # the forward knobs wrote are already final and are passed through untouched.
    for var in (ORIENT_ALL_TILES_VAR, FOOT_CLEAR_TERRAIN_VAR):
        if not parse_flag(get(var), var):
            continue
        rewards = _rewards(cfg, var)
        terms = {}
        for name in FUNC_SWAPS[var]:
            term = getattr(rewards, name, None)
            if term is None or not hasattr(term, "func"):
                raise RuntimeError(f"{var}=1 swaps the func of {name!r}, but the rewards cfg has no such term")
            terms[name] = term
        if all(not float(getattr(t, "weight", 0.0) or 0.0) for t in terms.values()):
            # a swap whose every term weighs 0 trains nothing (like KRABBY_POWER_JOINTS at weight 0): refuse at
            # cfg build instead of a launch the +200 liveness read would kill
            raise RuntimeError(f"{var}=1 swaps {sorted(terms)}, but every one of them has weight 0 -- set the "
                               "term weights in the same overlay")
        funcs = swap_funcs if swap_funcs is not None else _default_swap_funcs(var)
        for name, term in terms.items():
            term.func = funcs[name]
        _consumed(var, get(var))

    # -- action path
    hz = parse_lpf_hz(get(ACTION_LPF_HZ_VAR))
    if hz is not None:
        action = getattr(getattr(cfg, "actions", None), "joint_pos", None)
        if action is None:
            raise RuntimeError(f"{ACTION_LPF_HZ_VAR} is set but the env cfg has no actions.joint_pos")
        action.lpf_hz = hz
    if parse_flag(get(ACTION_LPF_OBS_VAR), ACTION_LPF_OBS_VAR):
        if hz is None:
            raise RuntimeError(f"{ACTION_LPF_OBS_VAR}=1 observes the filtered rod command, which "
                               f"only exists with {ACTION_LPF_HZ_VAR} > 0")
        _set_obs_param(cfg, ACTION_LPF_OBS_VAR, "lpf_obs")

    # -- command / clock
    hyst = parse_relay_hyst(get(RELAY_HYST_VAR))
    if hyst is not None:
        cfg.commands.base_velocity.relay_hyst = hyst
    stand_hold = parse_flag(get(STAND_HOLD_VAR), STAND_HOLD_VAR)
    if stand_hold:
        cfg.commands.base_velocity.stand_hold = True

    # -- stand regime
    stand_park = parse_flag(get(STAND_PARK_VAR), STAND_PARK_VAR)
    if stand_park:
        cfg.omni_stand_park = True
    stand_obs = parse_stand_obs(get(STAND_OBS_VAR))
    if stand_obs == "1":
        _set_obs_param(cfg, STAND_OBS_VAR, "stand_obs")
    elif stand_obs == "clock":
        # p2r (2026-10-09): the bit in the hidden clock column -- free only while the clock is hidden
        # on EVERY row. The clock knobs are applied before this block (omni_knobs.apply_omni_knobs).
        for _g, _t, term in _omni_obs_terms(cfg):
            params = term.params or {}
            if params.get("clock_hide") is not True or params.get("clock_hide_ramp") is not None:
                raise RuntimeError(f"{STAND_OBS_VAR}=clock writes the stand bit into the hidden clock column, "
                                   "which needs KRABBY_OMNI_CLOCK_HIDE=1 and no KRABBY_OMNI_CLOCK_HIDE_RAMP "
                                   f"(the obs term carries clock_hide={params.get('clock_hide')!r}, "
                                   f"clock_hide_ramp={params.get('clock_hide_ramp')!r})")
        _set_obs_param(cfg, STAND_OBS_VAR, "stand_bit_clock")
    # Every STAND-keyed knob the eval carries also latches the eval hook's slot per hold
    # (``CrabHexOmniCommand.schedule_hook``): otherwise an eval stand hold that drifts past the
    # facing dead band re-derives as "turn" and escapes the very knob it is scoring.
    if stand_hold or stand_park or stand_obs is not None:
        cfg.commands.base_velocity.eval_slot_latch = True


def _consumed(var: str, raw) -> None:
    """The wiring line a training log greps for: the knob reached the runner cfg."""
    print(f"[KNOB] {var}={str(raw).strip()} consumed", flush=True)


def apply_exploration_knobs(agent_cfg, environ=None) -> None:
    """The RUNNER-side knobs onto an omni PPO runner cfg (no-op for each one unset).

    The values land on the cfg that ``train.py`` dumps to ``params/agent.yaml`` (the ladder's
    agent-side audit reads them back from there): ``policy.action_std_max``,
    ``policy.action_std_reset``, ``policy.noise_std_type`` / ``init_noise_std`` are consumed by
    ``CrabHexActorCriticRMA``, every ``algorithm.*`` field by ``PPOWithExtractor`` (whose runner now
    REFUSES an algorithm key the class does not accept, so a field written here can never be
    dropped silently). Cross-knob rules are enforced HERE, before anything is written: a knob that
    would have no effect (a ramp without its loss, a KL stop without the fixed schedule, ...) is a
    recipe that says something it does not train, so it is refused rather than ignored.
    """
    env = os.environ if environ is None else environ
    std_max = parse_std_max(env.get(ACTION_STD_MAX_VAR))
    std_reset = parse_std_reset(env.get(ACTION_STD_RESET_VAR))
    entropy = parse_entropy_coef(env.get(ENTROPY_COEF_VAR))
    ceiling = STOCK_ACTION_STD_MAX if std_max is None else std_max
    if std_reset is not None and std_reset > ceiling:
        raise ValueError(f"{ACTION_STD_RESET_VAR}={std_reset} exceeds the std clamp ceiling {ceiling} "
                         f"({ACTION_STD_MAX_VAR}): the reset value would be clamped away on the "
                         "first rollout")
    # -- PPO construction (2026-10-01): parse and cross-check everything before writing anything
    bound_c1 = parse_bound_loss(env.get(ACTION_BOUND_LOSS_VAR))
    ramp = parse_bound_ramp(env.get(ACTION_BOUND_RAMP_VAR))
    bound = parse_action_bound(env.get(ACTION_BOUND_VAR))
    std_param = parse_std_param(env.get(STD_PARAM_VAR))
    init_std = parse_init_std(env.get(INIT_STD_VAR))
    schedule = parse_ppo_schedule(env.get(PPO_SCHEDULE_VAR))
    lr = parse_positive(env.get(PPO_LR_VAR), PPO_LR_VAR)
    desired_kl = parse_positive(env.get(PPO_DESIRED_KL_VAR), PPO_DESIRED_KL_VAR)
    kl_stop = parse_positive(env.get(PPO_KL_STOP_VAR), PPO_KL_STOP_VAR)
    hist_lr = parse_positive(env.get(PPO_HIST_LR_VAR), PPO_HIST_LR_VAR)
    hand_diag = parse_flag(env.get(PPO_HAND_DIAG_VAR), PPO_HAND_DIAG_VAR)
    # -- p2r statue distillation (2026-10-09)
    anchor_w = parse_statue_w(env.get(ANCHOR_W_VAR), ANCHOR_W_VAR)
    anchor_ckpt = parse_anchor_ckpt(env.get(ANCHOR_CKPT_VAR))
    prior_w = parse_statue_w(env.get(STAND_PRIOR_W_VAR), STAND_PRIOR_W_VAR)
    _check_statue_knobs(env, anchor_w, anchor_ckpt, prior_w)
    if ramp is not None and (bound_c1 is None or ramp[1] != bound_c1):
        raise ValueError(f"{ACTION_BOUND_RAMP_VAR}={env.get(ACTION_BOUND_RAMP_VAR)!r} ramps TO its c1, which must "
                         f"equal {ACTION_BOUND_LOSS_VAR} (got {env.get(ACTION_BOUND_LOSS_VAR)!r}): the loss "
                         "knob is the final coefficient")
    if bound is not None and bound_c1 is None:
        raise ValueError(f"{ACTION_BOUND_VAR} is set but {ACTION_BOUND_LOSS_VAR} is off: the bound has no "
                         "effect without the loss")
    if std_param == "soft":
        if std_max is not None or std_reset is not None:
            raise ValueError(f"{STD_PARAM_VAR}=soft has a fixed (0.05, 2.0) band and no clamp: "
                             f"{ACTION_STD_MAX_VAR} / {ACTION_STD_RESET_VAR} do not apply to it")
        if init_std is not None and not init_std < STOCK_ACTION_STD_MAX:
            raise ValueError(f"{INIT_STD_VAR}={init_std} must lie strictly inside the soft band "
                             f"({ACTION_STD_MIN}, {STOCK_ACTION_STD_MAX})")
    elif init_std is not None and init_std > ceiling:
        raise ValueError(f"{INIT_STD_VAR}={init_std} exceeds the std clamp ceiling {ceiling}: the initial "
                         "std would be clamped away on the first rollout")
    if kl_stop is not None or desired_kl is not None:
        effective = schedule if schedule is not None else getattr(agent_cfg.algorithm, "schedule", None)
        if kl_stop is not None and effective != "fixed":
            raise ValueError(f"{PPO_KL_STOP_VAR} is the 'fixed' schedule's early stop, but the schedule is "
                             f"{effective!r} (set {PPO_SCHEDULE_VAR}=fixed)")
        if desired_kl is not None and effective == "fixed":
            raise ValueError(f"{PPO_DESIRED_KL_VAR} has no effect under the 'fixed' schedule "
                             f"(use {PPO_KL_STOP_VAR} for its early stop)")
    if std_max is not None:
        agent_cfg.policy.action_std_max = std_max
        _consumed(ACTION_STD_MAX_VAR, env.get(ACTION_STD_MAX_VAR))
    if std_reset is not None:
        agent_cfg.policy.action_std_reset = std_reset
        _consumed(ACTION_STD_RESET_VAR, env.get(ACTION_STD_RESET_VAR))
    if entropy is not None:
        agent_cfg.algorithm.entropy_coef = entropy
        _consumed(ENTROPY_COEF_VAR, env.get(ENTROPY_COEF_VAR))
    if bound_c1 is not None:
        agent_cfg.algorithm.action_bound_loss_coef = bound_c1
        _consumed(ACTION_BOUND_LOSS_VAR, env.get(ACTION_BOUND_LOSS_VAR))
    if ramp is not None:
        agent_cfg.algorithm.action_bound_loss_ramp = [ramp[0], ramp[1], ramp[2]]
        _consumed(ACTION_BOUND_RAMP_VAR, env.get(ACTION_BOUND_RAMP_VAR))
    if bound is not None:
        agent_cfg.algorithm.action_bound = bound
        _consumed(ACTION_BOUND_VAR, env.get(ACTION_BOUND_VAR))
    if std_param is not None:
        agent_cfg.policy.noise_std_type = std_param
        _consumed(STD_PARAM_VAR, env.get(STD_PARAM_VAR))
    if init_std is not None:
        agent_cfg.policy.init_noise_std = init_std
        _consumed(INIT_STD_VAR, env.get(INIT_STD_VAR))
    if schedule is not None:
        agent_cfg.algorithm.schedule = schedule
        _consumed(PPO_SCHEDULE_VAR, env.get(PPO_SCHEDULE_VAR))
    if lr is not None:
        agent_cfg.algorithm.learning_rate = lr
        _consumed(PPO_LR_VAR, env.get(PPO_LR_VAR))
    if desired_kl is not None:
        agent_cfg.algorithm.desired_kl = desired_kl
        _consumed(PPO_DESIRED_KL_VAR, env.get(PPO_DESIRED_KL_VAR))
    if kl_stop is not None:
        agent_cfg.algorithm.kl_stop = kl_stop
        _consumed(PPO_KL_STOP_VAR, env.get(PPO_KL_STOP_VAR))
    if hist_lr is not None:
        agent_cfg.algorithm.hist_encoder_learning_rate = hist_lr
        _consumed(PPO_HIST_LR_VAR, env.get(PPO_HIST_LR_VAR))
    if hand_diag:
        agent_cfg.algorithm.hand_diag_vy_obs_col = HAND_DIAG_VY_OBS_COL
        _consumed(PPO_HAND_DIAG_VAR, env.get(PPO_HAND_DIAG_VAR))
    if anchor_w is not None:
        agent_cfg.algorithm.anchor_coef = anchor_w
        _consumed(ANCHOR_W_VAR, env.get(ANCHOR_W_VAR))
    if anchor_ckpt is not None:
        agent_cfg.algorithm.anchor_ckpt = resolve_anchor_ckpt(anchor_ckpt)
        _consumed(ANCHOR_CKPT_VAR, env.get(ANCHOR_CKPT_VAR))
    if prior_w is not None:
        agent_cfg.algorithm.stand_prior_coef = prior_w
        _consumed(STAND_PRIOR_W_VAR, env.get(STAND_PRIOR_W_VAR))
    if anchor_w is not None or prior_w is not None:
        agent_cfg.algorithm.stand_mask_col = STAND_BIT_OBS_COL


def _check_statue_knobs(env, anchor_w, anchor_ckpt, prior_w) -> None:
    """The p2r statue knobs' cross rules (before anything is written); all off -> nothing to check."""
    if anchor_w is None and anchor_ckpt is None and prior_w is None:
        return
    armed = ", ".join(f"{v}={env.get(v)!r}" for v in STATUE_KNOBS if not _blank(env.get(v)))
    if parse_stand_obs(env.get(STAND_OBS_VAR)) != "clock":
        raise ValueError(f"{armed}: the statue losses mask on the stand bit in obs column {STAND_BIT_OBS_COL}, "
                         f"which only {STAND_OBS_VAR}=clock writes (got {env.get(STAND_OBS_VAR)!r})")
    if anchor_ckpt is None:
        raise ValueError(f"{armed}: {ANCHOR_W_VAR} / {STAND_PRIOR_W_VAR} need {ANCHOR_CKPT_VAR} (the frozen "
                         "anchor's mean and the fixed sigma_A both come from that checkpoint)")
    if anchor_w is None and prior_w is None:
        raise ValueError(f"{ANCHOR_CKPT_VAR} is set but {ANCHOR_W_VAR} and {STAND_PRIOR_W_VAR} are both off: "
                         "the checkpoint would train nothing")
    if not os.path.isfile(resolve_anchor_ckpt(anchor_ckpt)):
        raise ValueError(f"{ANCHOR_CKPT_VAR}={anchor_ckpt!r} is not a file (resolved {resolve_anchor_ckpt(anchor_ckpt)!r})")
