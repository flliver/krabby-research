#!/usr/bin/env python3
"""Omni phase 2 -- the live kill reads (CHARTER 8) and the calibration-smoke audit (CHARTER 5). CPU only, read-only.

Every read counts iterations FROM THE FORK START (the run's first logged iteration; a fork of model_14995 logs from
14996): "+200" is the window [first+190, first+210], "+1000" is [first+950, first+1050]. Each read that fails STOPS
THE QUEUE and asks (scripts/run_arm.sh: kill with `systemctl --user stop`, touch STOP_QUEUE, NOTIFY line).

  wiring  (+200)  * top-level `curriculum: null` in params/env.yaml (KRABBY_PHASEOUT "" removed every ramp);
                  * clock hidden (Metrics/clock_hidden_frac >= 0.999 over the window, when logged), clock / apex
                    reward weights 0.001, the arm's action_rate weight (dump);
                  * DR present (randomize_rigid_body_mass / _com events, push_by_setting_velocity at +-0.5 m/s) and the
                    element weights of CHARTER 3 (collision -2, edge -0.3, stumble -1, airtime 0.8, stride 0.5);
                  * the arm's geometry sub-terrains present (field: omni_field_terrain and no omni_ring_terrain;
                    A0: the reverse);
                  * Metrics/base_parkour/reach_obst_frac > 0 over the window;
                  * no dropped PPO keys (no "Ignoring unsupported PPO config keys" line in the training log);
                  * the OVERLAY AUDIT: run_candidates.overlay_checks over the chain's WHOLE stage overlay (every CHARTER 3
                    key, PROMOTE included) against params/env.yaml / agent.yaml and the training log -- a key with no
                    registered dump check FAILS (it cannot be shown realised);
                  * TERM LIVENESS: the arm-specific term's Episode_Reward over the window is non-zero (S1, E1, F1) and,
                    for E1 / F1, within 3x of its CHARTER 4 replay magnitude on the same per-max-episode-second basis
                    (replay_gate.json, written when the CHARTER 4 gate is read: {"E1": {"episode_reward": x}, ...});
                    F1's stand-only clause: per-slot telemetry (F1_NONSTAND_TAGS) reading 0 when logged; when none
                    is logged, the CHARTER 9 reading -- stand-only BY CONSTRUCTION: the dumped term's func is
                    parkour_isaaclab.envs.mdp.rewards:reward_stand_stance_quality AND that function's return is
                    multiplied by `stand_mask()` (an AST check of the code root's rewards.py); either failing = FAIL.
  terrain (+1000) on OBSTACLE tiles (OBST_LEVEL_TAGS): the window mean level has risen above its start (mean of the
                  fork's first 20 logged iterations), promotions are non-zero (PROMO_TAGS if logged; otherwise implied
                  by the rise, which no run can show without a promotion), and the mean level is not pinned
                  (<= PIN_EPS above 0 or >= top - PIN_EPS, top = num_rows - 1 from params/env.yaml). Missing
                  obstacle-tile telemetry is WAIT until fork + 1250 (the window end +1050 plus GRACE 200), then FAIL
                  (fail closed: never read off all tiles).
  preflight (before launch; run_arm.sh refuses with exit 7) what the live reads need from the CODE ROOT, so a run is
                  never launched only to be killed by missing telemetry: every arm and CAL2 needs the obstacle-tile
                  level and promotion metrics emitted (the metric keys of OBST_LEVEL_TAGS / PROMO_TAGS as string
                  literals in the non-experiment parkour sources); E1 / F1 need a non-zero replay_gate.json entry;
                  F1 needs the stand-only construction (or a per-slot metric) above.
  collapse (after +300, every poll) rolling-200 Train/mean_episode_length < 0.5 x the SOURCE run's last-200 mean, or
                  rolling-200 Episode_Termination/crab_failure > 0.6.

STAGE p2s (CHARTER 11 / P2S_PLAN.md steps 3 and 6; the dose arms ST1 / ST2, chains P2__ST<k>_s<seed> or a fresh
relaunch name P2__ST<k>R<j>_s<seed>):
  wiring (+200)   on top of every read above: TERM LIVENESS of both stand-slot terms -- Episode_Reward/
                  reward_stand_support and Episode_Reward/penalty_stand_slide each non-zero (CHARTER 8's requirement,
                  kept beside the band because a dead T1 implies sum s_f = 3, inside it); the slide term <= 0 and
                  within 3x of its replay magnitude CALIBRATED ON THE SAMPLED probe (replay_gate.json "p2s".<ST1|ST2>
                  .terms.<term>: episode_reward_sampled, or episode_reward when "p2s".episode_reward_units names the
                  SAMPLED REF -- scripts/p2s_replay.py's record; absent -> FAIL, fail closed: never the deterministic
                  episode_reward_det_ref); the support term's IMPLIED mean
                  sum_f s_f = 3 + Episode_Reward / (w_support x share) inside [2.5, 6], with share = the stand-slot
                  share PINNED at 0.15 and read from the chain env (KRABBY_OMNI_SLOTS' first field; any other value
                  FAILS); the support term's ratio to its sampled replay magnitude is reported, not gated (P2S_PLAN
                  step 3: the slide term is held at 3x, the support term at the sum-s_f band). STAND-ONLY BY
                  CONSTRUCTION, F1's AST read generalised to the motion-rewards module: both dumped funcs are
                  parkour_tasks.crab_hex_omni_task.mdp.crab_hex_omni_motion_rewards:<term> and every return of both
                  functions is multiplied by a name bound to a stand_mask() call or to a module helper that gates on
                  stand_mask() (_stand_gate).
  p2s guards (every poll, run_arm.sh --at p2s), iterations counted from the fork start, against the REFERENCE run's
                  SAME fork-relative window -- at s18 the C0 continuation (arms/P2__C0_s18.record.json run_dir, the
                  X12 run); at s16 / s17 (no same-seed C0) a CONSTANT, the seed's own S1 p2a source run's mean over its
                  last 500 its (arms/P2__S1_s<seed>.record.json run_dir; a cross-seed C0 would trip at once):
                    termination KILL  rolling-200 Train/mean_episode_length < 0.9 x the reference's (from +300);
                    sigma KILL        rolling-200 PPO/effective_std < 0.8 x the reference's (from +1000);
                    termination FLAG  rolling-500 Episode_Termination/crab_failure > the reference's + 0.08 (from
                                      +500): NOTIFY-ONLY -- exit 5, never a kill (run_arm.sh writes
                                      $ARMS/<ref>.p2s_flag, push-notifies, and the run continues; scripts/
                                      p2s_queue.sh stops and asks after that arm). Sibling noise reaches +0.116.
                  The CHARTER 8 collapse guard (rolling-200 crab_failure > 0.6) stays the hard backstop.
  preflight       a p2s arm refuses to launch unless replay_gate.json "p2s" has verdict PASS (or a recorded user
                  overrule {"overrule": {"by": "user", ...}}), both terms carry a non-zero sampled magnitude for the
                  arm's dose, "p2s".<dose>.knobs equals the chain's KRABBY_STAND_SUPPORT_W / KRABBY_STAND_SLIDE_W
                  (the replay finalises the doses), the stand share is the pinned 0.15, the stand-only construction
                  reads true in the code root, and the guard's reference run is readable with all three tags.

STAGE p2r's Z arms (CHARTER 12 / P2R_STATUE_PLAN.md step 1, "learn the statue in-network": the grafted stand bit +
two masked PPO losses on the action mean -- a walking anchor on non-stand rows, a statue prior on stand rows -- NO
stand-slot REWARD terms at all, unlike the p2s dose arms; chains P2__Z<k>_s<seed> or a fresh relaunch name
P2__Z<k>R<j>_s<seed>):
  wiring (+200)   on top of every CHARTER 8 read above: the three agent knobs (KRABBY_ANCHOR_W, KRABBY_ANCHOR_CKPT,
                  KRABBY_STAND_PRIOR_W) present with the first two weights positive; Loss/anchor_rows in [0.8, 0.95]
                  and Loss/prior_rows in [0.05, 0.2] (the window mean); Loss/anchor_mb0 reads EXACTLY 0 at the fork's
                  first logged iteration (fork exactness: both terms are computed on the identical minibatch before
                  masking, so the very first logged value can never be anything else); Loss/anchor and
                  Loss/stand_prior both finite and > 0 (the window mean); the REALISED DOSE -- Loss/stand_prior at
                  the first logged iteration -- in [1.2, 4.6] (P2R_STATUE_PLAN.md step 1's "fork values: statue ~
                  2.3, anchor exactly 0").
  p2s guards      Z arms keep the SAME three guards as a p2s dose arm (termination KILL, sigma KILL, crab_failure
                  NOTIFY-ONLY flag; run_arm.sh --at p2s), read against the SAME reference (p2s_reference() already
                  resolves P2__C0_s18 at seed 18 with no Z-specific change) -- only the dose / term-liveness half of
                  a p2s arm's --at p2s read (judge_p2s_liveness / judge_p2s_construction) does not apply, since a Z
                  arm has no stand-slot reward term to read.
  preflight       a Z arm refuses to launch unless its fork checkpoint's graft.json (scripts/p2r_graft.py) reads
                  identity_cpu true with a non-empty identity_gpu_stamps, and the three agent knobs are present
                  (the first two weights positive, KRABBY_ANCHOR_CKPT equal to the chain's own fork checkpoint) --
                  it does NOT need a p2s replay-gate verdict, a sampled magnitude or a stand share (no stand terms).
  Z2 (NOTE(p2r Z2, 2026-10-10); P2R_Z2_PLAN.md step 4; chains P2__Z2_s<seed> / P2__Z2R<j>_s<seed>): the statue prior's
                  target is the re-stance controller R(obs) (KRABBY_STAND_PRIOR_TARGET=restance, KRABBY_RESTANCE).
                  NOTE(p2r Z2 cams-only, 2026-10-10): R is kinematic (cams + hip-pitch positions, no load signal), so
                  the chain runs with KRABBY_OMNI_STAND_PROXY OFF and the obs stays Z1's. Z1's wiring is unchanged; a
                  Z2 arm runs Z1's row / mb0 / finiteness checks with the realised-dose band read from
                  p2r_z2_dose.json (keys band_lo, band_hi, written at rc 0; step 4's CPU dose -- the check REFUSES,
                  i.e. fails, without the file, never falls back to Z1's [1.2, 4.6]), plus: the Z2 knobs in the overlay
                  (STAND_PROXY absent / 0, TARGET restance, a parseable RESTANCE); Loss/restance_nonzero >= 0.5 (window
                  mean); every Loss/restance_<branch> (hold / wait / lift / home / seat) logged in the window and the
                  'home' share > 0 (the in-air homing branch is live). (The held == live proxy clause is gone with the
                  proxy.) The Z2 preflight adds the same knobs and a readable dose file to Z1's.
  Z2c (NOTE(p2r Z2c, 2026-10-10); P2R_Z2C_PLAN.md step 5; chains P2__Z2c_s<seed> / P2__Z2cR<j>_s<seed>, arm 'Z2C'):
                  Z2's wiring + the clean-state cloning (BC) term: the BC knobs (KRABBY_BC_W > 0, every
                  KRABBY_BC_BANK path existing -- repo-relative resolved like KRABBY_ANCHOR_CKPT --, KRABBY_BC_BATCH a
                  positive int); Loss/bc and Loss/bc_mb0 logged, finite and > 0 (window means); the first-update
                  Loss/bc_mb0 inside [0.52, 2] x the dose file's bc_mb0_expected (p2r_z2_dose.py --bc-bank; REFUSED
                  without it, and refused when the file's bc.banks are not the chain's KRABBY_BC_BANK). Z2's
                  stand-prior dose clause reads Loss/stand_prior_mb0 at the first iteration when the trainer logs it
                  (BC's pre-fit inside the first update lowers the update's MEAN prior), else the first-update
                  Loss/stand_prior with a note. Z2's episode-length notify-only guard is inherited. The Z2c preflight
                  adds the BC knobs and the dose file's BC block to Z2's.
  Z2d (NOTE(p2r Z2d, 2026-10-11); P2R_Z2D_PLAN.md step 1 "The arm wiring"; chains P2__Z2d_s<seed> /
                  P2__Z2dR<j>_s<seed>, arm 'Z2D'): every Z2c check, read against ITS OWN dose file p2r_z2d_dose.json
                  (p2r_z2_dose.py --bc-bank ... --bc-stand-mask; refused when absent, when its bc.banks are not the
                  chain's KRABBY_BC_BANK, or when its bc.stand_mask is not true), plus: the overlay carries
                  KRABBY_OMNI_STAND_MASK=1; Metrics/stand_mask_frac logged, finite and inside the prior-rows band
                  [0.05, 0.2] (window mean: the mask is live on the stand rows). Z2's episode-length notify-only guard
                  is inherited. The Z2d preflight = Z2c's + the mask knob + the Z2d dose file's stand_mask. A Z arm
                  with a variant letter this file does not know (e.g. 'Z2E') is REFUSED -- at +200 and in the
                  preflight -- rather than falling through to Z1's checks.

Exit codes (run_arm.sh's contract): 0 pass / not applicable, 2 not enough telemetry yet (retry next poll), 3 FAIL
(kill + stop the queue), 5 NOTIFY-ONLY (p2s termination flag: notify, never kill). The last stdout line is one JSON
object.

    p2_live.py --iters RUN_DIR                                   -> "<first> <last>" logged iterations
    p2_live.py --check RUN_DIR --chain CHAIN --at 200|1000|collapse|p2s [--train-log F] [--source RUN_DIR]
               [--candidates F] [--replay F] [--arms DIR] [--z2-dose F]
               (--z2-dose: default the arm's own file -- p2r_z2d_dose.json for Z2D, p2r_z2_dose.json otherwise)
    p2_live.py --smoke-audit RUN_DIR --chain CHAIN [--train-log F] [--require-terrain-move]
    p2_live.py --preflight --chain CHAIN [--code-root DIR] [--replay F] [--arms DIR] [--z2-dose F]
    p2_live.py --code-guard [--code-root DIR] [--lrout DIR] [--anchor-name ANC_s18] [--reanchor]
    p2_live.py --self-test

  code-guard (pre-launch audit M1, 2026-10-06: parallel sessions share the checkout) the LIVE code state of the code
                  root -- run_candidates.code_sha() / eval_sha() / a dirty flag on the code + eval pathspecs, the
                  three fields of the CHARTER 5 code-state sidecar -- against the s18 ANCHOR's sidecar (the newest
                  complete <lrout>/<anchor>/suite__omni/seed001/<ts>/code_state.json, the first anchor row the queue
                  runs). Before any anchor stamp exists it requires dirty = false only. A dirty tree, a difference in
                  any of the three fields, or an anchor stamp without a sidecar REFUSES (exit 7): scripts/run_arm.sh
                  calls it in run_ref (after the preflight) and at the top of every --row-eval / --lr-eval, and
                  stops + NOTIFIES. --reanchor (P2_REANCHOR=1 in run_arm.sh, CHARTER 5 "re-anchoring") skips the
                  comparison and still requires dirty = false, for the first row of a deliberate re-anchor only.
"""
from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import math
import os
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CAMPAIGN = HERE.parent
R = Path("/home/nickmagus/krabby/krabby-research")
RC_PATH = R / "parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task/experiments/tools/run_candidates.py"

RC_PASS, RC_WAIT, RC_FAIL = 0, 2, 3
RC_NOTIFY = 5                                     # p2s termination flag: notify-only, run_arm.sh never kills on it
RC_REFUSE = 7                                     # --code-guard refusal (run_arm.sh's "refused before launch" code)
#: the code-guard reference: the s18 anchor's first queue row (scripts/p2_queue.sh runs suite__omni env_seed 1 first)
ANCHOR_REF_ROW, ANCHOR_REF_SEED = "suite__omni", 1
OUT_DEFAULT = R / "parkour/outputs/omni_plan/phase2"
W200 = (190, 210)
W1000 = (950, 1050)
AT_200, AT_1000, AT_COLLAPSE = 210, 1060, 300     # fork-relative iteration a read first runs at
GRACE = 200                                       # telemetry still missing this far past the window -> FAIL
START_N = 20                                      # "its start": the fork's first 20 logged iterations
PIN_EPS = 0.05
ROLL = 200
COLLAPSE_LEN_FRAC = 0.5
COLLAPSE_FAIL = 0.6
LIVE_RATIO = 3.0
CLOCK_W = 0.001
#: CHARTER 3 "from the 2a preset" element weights (params/env.yaml rewards.<term>.weight)
ELEMENT_WEIGHTS = {"reward_collision": -2.0, "reward_feet_edge": -0.3, "reward_feet_stumble": -1.0,
                   "reward_feet_air_time_positive": 0.8, "reward_stride_length": 0.5}
DR_EVENTS = ("randomize_rigid_body_mass", "randomize_rigid_body_com", "push_by_setting_velocity")
PUSH = 0.5
FIELD_FN, RING_FN = "omni_field_terrain", "omni_ring_terrain"
#: the arm-specific terms (CHARTER 8 term liveness), = motion_knobs.REUSED_WEIGHT_KNOBS["KRABBY_ACTION_RATE_W"],
#: motion_knobs.POWER_TERM (E1 re-scopes the existing power term to the actuated joints) and
#: motion_knobs.STAND_STANCE_TERM (pinned by tests/unit/test_omni_phase2_tooling.py).
ARM_TERMS = {"S1": "reward_action_rate", "E1": "penalty_mechanical_power", "F1": "reward_stand_stance_quality"}
#: F1's term function (params/env.yaml rewards.<term>.func) and the source the stand-only construction is read from.
F1_FUNC = "parkour_isaaclab.envs.mdp.rewards:reward_stand_stance_quality"
REWARDS_PY = "parkour/parkour_isaaclab/envs/mdp/rewards.py"
REPLAY_TERMS = ("E1", "F1")
#: Stage p2b (P2B_PLAN.md step 1 / CHARTER 10): each lever's new term(s), sign-aware -- the sign an income must
#: carry to count as live (a reward term pays >= 0, a penalty term <= 0). C0 has none (the p2a continuation).
P2B_ARM_TERMS = {"CL": ("reward_foot_clearance", "penalty_swing_min_clearance"), "TG": ("reward_orientation",)}
P2B_TERM_SIGN = {"reward_foot_clearance": 1, "penalty_swing_min_clearance": -1, "reward_orientation": -1}
#: TG's extra floor (P2B_PLAN.md step 1): |Episode_Reward/reward_orientation| at +200 must be >= this multiple of
#: the seed's S1 p2a training tail (that run's term income, which has no obstacle-tile share -- CHARTER 10).
P2B_TG_FLOOR_RATIO = 2.0
#: the p2b intermediate checkpoints' offsets from the fork's first logged iteration (P2B_PLAN.md step 1/4). Round,
#: unlike the phase-2 MCK offsets (1004/2504/4004), because every p2b fork is the same model_19994 head.
P2B_MCK_OFFSETS = (1005, 2505, 4005)
#: TG's floor tail length: the seed's S1 p2a run's last 500 its (P2B_PLAN.md step 1), not the live-read ROLL (200).
P2B_TAIL_N = 500
#: Stage p2s (P2S_PLAN.md step 3 / CHARTER 11): the dose arms and their two stand-slot terms (= motion_knobs.
#: STAND_SUPPORT_TERM / STAND_SLIDE_TERM, pinned by tests/unit/test_omni_p2s_live.py), the term -> weight-knob map, and
#: the dumped func's module (= motion_knobs._REWARDS_MOD).
P2S_ARMS = ("ST1", "ST2")
P2S_SUPPORT, P2S_SLIDE = "reward_stand_support", "penalty_stand_slide"
P2S_TERMS = (P2S_SUPPORT, P2S_SLIDE)
P2S_WEIGHT_KNOBS = {P2S_SUPPORT: "KRABBY_STAND_SUPPORT_W", P2S_SLIDE: "KRABBY_STAND_SLIDE_W"}
P2S_FUNC_MOD = "parkour_tasks.crab_hex_omni_task.mdp.crab_hex_omni_motion_rewards"
MOTION_REWARDS_PY = "parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task/mdp/crab_hex_omni_motion_rewards.py"
#: the stand-slot share the implied-sum-s_f read divides by, PINNED (P2S_PLAN "left for implementation"): the chain
#: env's KRABBY_OMNI_SLOTS first field must equal it (the logged Metrics/base_velocity/stand_frac_actual runs ~0.10 on
#: the S1 recipe; the band [2.5, 6] absorbs that, and a share mismatch in the env itself FAILS)
P2S_STAND_SHARE = 0.15
P2S_SUPPORT_SUM_BAND = (2.5, 6.0)
#: the realised stand-slot share (fable review 2026-10-08): the implied sum s_f is read against it when logged (the
#: pinned 0.15 compresses implied - 3 by ~0.67 at the S1 recipe's ~0.10 realised share); the pinned value is reported
P2S_STAND_FRAC_TAG = "Metrics/base_velocity/stand_frac_actual"
P2S_TRIPOD = 3.0                                  # reward_stand_support's offset (STAND_SUPPORT_TRIPOD)
#: the p2s guards (P2S_PLAN step 6): termination KILL (episode length), sigma KILL, termination NOTIFY-ONLY flag
P2S_LEN_FRAC, P2S_LEN_ROLL, P2S_LEN_AT = 0.9, 200, AT_COLLAPSE
#: the s16 / s17 episode-length kill line against the CONSTANT source tail (fable review 2026-10-08): a pure
#: continuation's rolling-200 vs its source's last-500 dips to 0.925 at s18 (CL 0.919), 0.02 above 0.9 -- so the
#: tail reference kills only below 0.85; the run-vs-run s18 line stays 0.9 (sibling minimum 0.920-0.936)
P2S_LEN_FRAC_TAIL = 0.85
P2S_STD_FRAC, P2S_STD_ROLL, P2S_STD_AT = 0.8, 200, 1000
P2S_CF_MARGIN, P2S_CF_ROLL, P2S_CF_AT = 0.08, 500, 500
#: the s16 / s17 constant reference: the seed's S1 p2a source run's last 500 its (P2S_PLAN "left for implementation")
P2S_REF_TAIL_N = 500
#: the s18 reference run's arms record (the C0 continuation, X12)
P2S_REF_S18 = "P2__C0_s18"
P2S_GATE_SEED = 18
#: Stage p2r's Z arms (CHARTER 12 / P2R_STATUE_PLAN.md step 1 "tooling" / "+200 wiring"): the grafted stand bit +
#: two masked PPO losses, NO stand-slot reward terms -- a Z arm is never a P2S_ARMS member and never reads
#: judge_p2s_liveness / judge_p2s_construction, but DOES share the p2s GUARDS (judge_p2s_guards) via run_arm.sh.
Z_ARM_RE = re.compile(r"^Z[0-9]+[A-Z]?$")         # NOTE(p2r Z2c, 2026-10-10): Z2c reads as Z2C
P2R_ANCHOR_W, P2R_ANCHOR_CKPT, P2R_STAND_PRIOR_W = "KRABBY_ANCHOR_W", "KRABBY_ANCHOR_CKPT", "KRABBY_STAND_PRIOR_W"
P2R_LOSS_ANCHOR, P2R_LOSS_PRIOR = "Loss/anchor", "Loss/stand_prior"
P2R_ANCHOR_ROWS_TAG, P2R_PRIOR_ROWS_TAG = "Loss/anchor_rows", "Loss/prior_rows"
P2R_ANCHOR_MB0_TAG = "Loss/anchor_mb0"
#: the +200 wiring bands (P2R_STATUE_PLAN.md step 1/4): anchor_rows ~ 0.88, prior_rows ~ 0.12 (the stand share)
P2R_ANCHOR_ROWS_BAND, P2R_PRIOR_ROWS_BAND = (0.8, 0.95), (0.05, 0.2)
#: the REALISED DOSE band: the first-update Loss/stand_prior (fork value ~ 2.3 at weight 1.0, KL units)
P2R_STAND_PRIOR_DOSE_BAND = (1.2, 4.6)
#: NOTE(p2r Z2, 2026-10-10): the Z2 arm (P2R_Z2_PLAN.md step 4 "+200 wiring"). The dose band comes from step 4's CPU
#: dose (p2r_z2_dose.json: band_lo / band_hi), never from Z1's constant; the branch names are pinned against
#: mdp/crab_hex_omni_restance.BRANCH_NAMES by the tests (kept literal here so p2_live never imports torch to log).
P2R_Z2_ARM = "Z2"
P2R_Z2_DOSE_JSON = CAMPAIGN / "p2r_z2_dose.json"
P2R_STAND_PROXY, P2R_STAND_PRIOR_TARGET, P2R_RESTANCE = (
    "KRABBY_OMNI_STAND_PROXY", "KRABBY_STAND_PRIOR_TARGET", "KRABBY_RESTANCE")
P2R_RESTANCE_NONZERO_TAG = "Loss/restance_nonzero"
P2R_RESTANCE_BRANCHES = ("hold", "wait", "lift", "home", "seat")
P2R_RESTANCE_BRANCH_TAGS = tuple(f"Loss/restance_{b}" for b in P2R_RESTANCE_BRANCHES)
#: NOTE(p2r Z2 cams-only, 2026-10-10): the held == live proxy clause (Metrics/stand_proxy_live_frac vs the dose
#: file's proxy_live_min) is DROPPED with the proxy; a Z2 chain must carry KRABBY_OMNI_STAND_PROXY off (these values).
P2R_RESTANCE_NONZERO_MIN = 0.5
P2R_STAND_PROXY_OFF = ("", "0")
RESTANCE_PY = R / "parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task/mdp/crab_hex_omni_restance.py"
#: NOTE(p2r Z2c, 2026-10-10): the Z2c arm (P2R_Z2C_PLAN.md step 5 "+200 wiring": Z2's checks + the BC term). Chains
#: P2__Z2c_s<seed> (z_arm -> "Z2C"); the dose file is Z2's p2r_z2_dose.json regenerated with --bc-bank.
P2R_Z2C_ARM = "Z2C"
P2R_BC_W, P2R_BC_BANK, P2R_BC_BATCH = "KRABBY_BC_W", "KRABBY_BC_BANK", "KRABBY_BC_BATCH"
P2R_LOSS_BC, P2R_BC_MB0_TAG, P2R_GRAD_NORM_BC_TAG = "Loss/bc", "Loss/bc_mb0", "Loss/grad_norm_bc"
#: read for Z2's stand-prior dose clause when logged (the prior's own minibatch-0 value, before BC's first step)
P2R_PRIOR_MB0_TAG = "Loss/stand_prior_mb0"
#: the realised BC dose band around the dose file's bc_mb0_expected (Z1's / Z2's [0.52x, 2x])
P2R_BC_DOSE_BAND = (0.52, 2.0)
#: NOTE(p2r Z2d, 2026-10-11): the Z2d arm (P2R_Z2D_PLAN.md step 1 "The arm wiring": Z2c's checks + the stand-row
#: input mask). Chains P2__Z2d_s<seed> (z_arm -> "Z2D"); ITS dose file is p2r_z2d_dose.json (p2r_z2_dose.py
#: --bc-bank ... --bc-stand-mask), never Z2 / Z2c's. The mask's liveness (the obs term's share of rows masked) is
#: banded like Loss/prior_rows (P2R_PRIOR_ROWS_BAND: the same stand rows).
P2R_Z2D_ARM = "Z2D"
P2R_Z2D_DOSE_JSON = CAMPAIGN / "p2r_z2d_dose.json"
P2R_STAND_MASK, P2R_STAND_MASK_FRAC_TAG = "KRABBY_OMNI_STAND_MASK", "Metrics/stand_mask_frac"
#: the Z arms with a variant letter this file judges explicitly; any other lettered Z arm is refused
P2R_Z_VARIANT_ARMS = (P2R_Z2C_ARM, P2R_Z2D_ARM)
#: the Z2-family arms (the dose file, the Z2 knobs, the episode-length notify-only guard)
P2R_Z2_FAMILY = (P2R_Z2_ARM, P2R_Z2C_ARM, P2R_Z2D_ARM)
#: F1's "non-zero only in stand-slot envs" from per-slot telemetry, when logged: any of these tags, reading 0 over the
#: window. None is emitted in the tree (2026-10-06); the stand-only construction check stands in (CHARTER 9).
F1_NONSTAND_TAGS = ("Metrics/stand_stance/nonstand_income", "Metrics/base_velocity/stand_stance_nonstand")
#: obstacle-tile terrain level and promotion telemetry -- these EXACT names (the parkour term's reset log reaches
#: tensorboard as Metrics/base_parkour/k; omni_exposure.OMNI_TERRAIN_WINDOW_KEYS, emitted by
#: OmniParkourEvent.reset on ring AND field runs since 2026-10-06): terrain_levels_obst = mean of terrain_levels over
#: is_obst_tile envs at the reset call (left out when there is none); promote_count = the applied move_up decisions on
#: episodes that ended on an obstacle tile in that reset call (rsl_rl averages the reset calls of an iteration, so a
#: logged value is promotions per reset step, > 0 iff any happened). demote_count is logged too and not read here.
#: The preflight still refuses every arm and CAL2 when a code root lacks them (an all-tile terrain_levels read is
#: never accepted).
METRIC_PREFIX = "Metrics/base_parkour/"
OBST_LEVEL_TAGS = (METRIC_PREFIX + "terrain_levels_obst",)
PROMO_TAGS = (METRIC_PREFIX + "promote_count",)
#: where the preflight looks for the emitted metric keys (relative to the code root; experiments / tests skipped)
SOURCE_DIRS = ("parkour/parkour_isaaclab", "parkour/parkour_tasks/parkour_tasks")
TAGS = {"clock_hidden": "Metrics/clock_hidden_frac", "reach_obst": "Metrics/base_parkour/reach_obst_frac",
        "ep_len": "Train/mean_episode_length", "crab_failure": "Episode_Termination/crab_failure",
        "levels_all": "Metrics/base_parkour/terrain_levels", "std": "PPO/effective_std"}


# ------------------------------------------------------------------------------------------------ small helpers
def arm_of(chain: str) -> str:
    """P2__A1_s18 -> A1; P2__CAL2_s18 -> CAL2; P2__CL_s16 -> CL; P2__TG_s17 -> TG; P2__C0_s18 -> C0."""
    m = re.match(r"^P2__([A-Z]+[0-9]*)_s\d+$", chain)
    return m.group(1) if m else ""


def seed_of(chain: str) -> int | None:
    """P2__CL_s16 -> 16 (the trailing _s<seed>); None if the chain name carries no seed."""
    m = re.search(r"_s(\d+)$", chain)
    return int(m.group(1)) if m else None


def p2b_checkpoints(fork_model_n: int) -> list[int]:
    """The p2b intermediate checkpoint numbers for a fork of ``model_<fork_model_n>.pt`` (P2B_PLAN.md step 1):
    the fork's first logged iteration (``fork_model_n + 1``) plus each of :data:`P2B_MCK_OFFSETS`."""
    first = fork_model_n + 1
    return [first + o for o in P2B_MCK_OFFSETS]


def p2b_replay_magnitude(replay: dict | None, arm: str, term: str):
    """``replay_gate.json``'s ``{"p2b": {arm: {"terms": {term: {"episode_reward": x}}}}}`` entry, or None. A term that
    carries ``episode_reward_live_expect`` (the replay value x its measured training/replay calibration, CHARTER 10
    2026-10-08: TG's tilt runs 1.7-2.9x higher in training than in the eval replay) is read at that value."""
    ent = (((replay or {}).get("p2b") or {}).get(arm) or {}).get("terms", {}).get(term, {})
    return ent.get("episode_reward_live_expect", ent.get("episode_reward"))


def p2b_replay_verdict_ok(replay: dict | None, arm: str) -> bool:
    """CHARTER 10 / P2B_PLAN.md step 2: the arm's p2b replay verdict is PASS, or a recorded user overrule."""
    ent = ((replay or {}).get("p2b") or {}).get(arm) or {}
    return ent.get("verdict") == "PASS" or (ent.get("overrule") or {}).get("by") == "user"


def p2s_arm(chain: str) -> str:
    """P2__ST2_s18 -> ST2; a fresh relaunch name after a kill, P2__ST2R1_s18 -> ST2 (the dose its replay magnitudes
    and weights are keyed by); anything else -> ''."""
    m = re.match(r"^P2__(ST[0-9]+)(?:R[0-9]+)?_s\d+$", chain)
    return m.group(1) if m and m.group(1) in P2S_ARMS else ""


def z_arm(chain: str) -> str:
    """P2__Z1_s18 -> Z1; a fresh relaunch name after a kill, P2__Z1R2_s16 -> Z1; anything else -> '' (the p2r
    equivalent of :func:`p2s_arm`, used by the p2s GUARDS -- a Z arm shares them -- never by the dose/term-liveness
    half of --at p2s, which Z arms skip: they have no stand-slot reward term). NOTE(p2r Z2c, 2026-10-10): a one-letter
    lower-case variant suffix reads upper-cased -- P2__Z2c_s18 / P2__Z2cR1_s18 -> Z2C."""
    m = re.match(r"^P2__(Z[0-9]+)([a-z]?)(?:R[0-9]+)?_s\d+$", chain)
    return (m.group(1) + m.group(2).upper()) if m else ""


def point_at(series, step):
    """The value logged at EXACTLY iteration ``step`` (never a window mean) -- Loss/anchor_mb0's "read at the
    first logged iteration" and the realised-dose "first-update Loss/stand_prior" (P2R_STATUE_PLAN.md step 1/4)."""
    for s, v in series or []:
        if s == step:
            return v
    return None


def judge_p2r_liveness(overlay: dict, ser: dict, lo: int, hi: int, first: int,
                       dose_band: tuple | None = P2R_STAND_PRIOR_DOSE_BAND,
                       dose_tag: str = P2R_LOSS_PRIOR) -> tuple[list, list, dict]:
    """Stage p2r's Z-arm +200 wiring (pure; P2R_STATUE_PLAN.md step 1/4): the graft + two masked PPO losses, no
    stand-slot reward terms at all. ``(fails, notes, live)``. ``dose_band`` (NOTE(p2r Z2, 2026-10-10)): Z1's constant
    by default; Z2 passes its p2r_z2_dose.json band, or None to skip the dose clause (the caller then refuses).
    ``dose_tag`` (NOTE(p2r Z2c, 2026-10-10)): the series the realised dose is read from at ``first`` (Z2c reads the
    prior's own minibatch-0 value when logged)."""
    fails: list = []
    notes: list = []
    live: dict = {}
    for knob in (P2R_ANCHOR_W, P2R_STAND_PRIOR_W):
        try:
            v = float(str((overlay or {}).get(knob, "")).strip())
        except ValueError:
            v = None
        live[knob] = v
        if v is None or not v > 0:
            fails.append(f"term liveness: {knob}={(overlay or {}).get(knob)!r} is not a positive weight")
    ckpt = str((overlay or {}).get(P2R_ANCHOR_CKPT, "")).strip()
    live[P2R_ANCHOR_CKPT] = ckpt or None
    if not ckpt:
        fails.append(f"term liveness: {P2R_ANCHOR_CKPT} is not set")
    ar, pr = window(ser.get(P2R_ANCHOR_ROWS_TAG), lo, hi), window(ser.get(P2R_PRIOR_ROWS_TAG), lo, hi)
    live["anchor_rows"], live["prior_rows"] = ar, pr
    lo_a, hi_a = P2R_ANCHOR_ROWS_BAND
    lo_p, hi_p = P2R_PRIOR_ROWS_BAND
    if ar is None or not (lo_a - 1e-9 <= ar <= hi_a + 1e-9):
        fails.append(f"term liveness: Loss/anchor_rows {ar} outside [{lo_a}, {hi_a}]")
    if pr is None or not (lo_p - 1e-9 <= pr <= hi_p + 1e-9):
        fails.append(f"term liveness: Loss/prior_rows {pr} outside [{lo_p}, {hi_p}]")
    mb0 = point_at(ser.get(P2R_ANCHOR_MB0_TAG), first)
    live["anchor_mb0_at_first"] = mb0
    if mb0 is None or mb0 != 0.0:
        fails.append(f"term liveness: Loss/anchor_mb0 at the first logged iteration ({first}) is {mb0}, "
                     "want exactly 0 (fork exactness: both losses share the same minibatch before masking)")
    la, lp = window(ser.get(P2R_LOSS_ANCHOR), lo, hi), window(ser.get(P2R_LOSS_PRIOR), lo, hi)
    live["loss_anchor"], live["loss_stand_prior"] = la, lp
    if la is None or not math.isfinite(la) or not la > 0:
        fails.append(f"term liveness: Loss/anchor {la} is not finite and > 0")
    if lp is None or not math.isfinite(lp) or not lp > 0:
        fails.append(f"term liveness: Loss/stand_prior {lp} is not finite and > 0")
    dose = point_at(ser.get(dose_tag), first)
    live["stand_prior_first_update"] = dose
    if dose_tag != P2R_LOSS_PRIOR:
        live["stand_prior_dose_tag"] = dose_tag
    if dose_band is not None:
        lo_d, hi_d = dose_band
        live["dose_band"] = [lo_d, hi_d]
        if dose is None or not (lo_d - 1e-9 <= dose <= hi_d + 1e-9):
            fails.append(f"term liveness: realised dose (first-update {dose_tag}) {dose} outside "
                         f"[{lo_d}, {hi_d}]")
    return fails, notes, live


_RESTANCE_MOD = None


def restance_mod(path: Path = RESTANCE_PY):
    """mdp/crab_hex_omni_restance.py loaded BY PATH (pure torch, no Isaac import), once -- only the Z2 judges need
    it (to parse KRABBY_RESTANCE exactly as the trainer does)."""
    global _RESTANCE_MOD
    if _RESTANCE_MOD is None:
        spec = importlib.util.spec_from_file_location("_p2_live_restance", path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules["_p2_live_restance"] = mod
        spec.loader.exec_module(mod)
        _RESTANCE_MOD = mod
    return _RESTANCE_MOD


def z2_dose_band(doc: dict | None, dose_name: str | None = None) -> tuple[tuple[float, float] | None, str | None]:
    """``(band, why)`` from p2r_z2_dose.json (keys band_lo, band_hi): both finite and lo < hi, else ``(None, why)``.
    ``dose_name`` (NOTE(p2r Z2d, 2026-10-11)): the file named in the reasons (default p2r_z2_dose.json)."""
    name = dose_name or P2R_Z2_DOSE_JSON.name
    if not isinstance(doc, dict):
        return None, f"no readable {name} (step 4's CPU dose sets the +200 band)"
    try:
        lo, hi = float(doc["band_lo"]), float(doc["band_hi"])
    except (KeyError, TypeError, ValueError):
        return None, (f"{name} lacks numeric band_lo / band_hi (band_lo={doc.get('band_lo')!r}, "
                      f"band_hi={doc.get('band_hi')!r})")
    if not (math.isfinite(lo) and math.isfinite(hi) and lo < hi):
        return None, f"{name} band [{lo}, {hi}] is not a finite lo < hi"
    # NOTE(p2r Z2, 2026-10-10): a dose file written at a non-OK rc (STOP: no home branch; error) is refused.
    # (cams-only, 2026-10-10: the proxy_live_min requirement is gone with the proxy.)
    if doc.get("rc") != 0:
        return None, f"{name} was written at rc {doc.get('rc')!r} ({doc.get('verdict')!r}), not 0"
    return (lo, hi), None


def judge_z2_knobs(overlay: dict) -> list:
    """The Z2 knobs a Z2 chain must carry (pure but for parsing KRABBY_RESTANCE through the trainer's own parser):
    KRABBY_STAND_PRIOR_TARGET=restance and a KRABBY_RESTANCE that parses ON, with KRABBY_OMNI_STAND_PROXY OFF
    (absent / blank / 0). NOTE(p2r Z2 cams-only, 2026-10-10): R reads no proxy; the cams-only chain keeps Z1's obs."""
    ov = overlay or {}
    fails = []
    if str(ov.get(P2R_STAND_PROXY, "")).strip().lower() not in P2R_STAND_PROXY_OFF:
        fails.append(f"{P2R_STAND_PROXY}={ov.get(P2R_STAND_PROXY)!r}, want it off (absent / 0: the cams-only chain)")
    if str(ov.get(P2R_STAND_PRIOR_TARGET, "")).strip().lower() != "restance":
        fails.append(f"{P2R_STAND_PRIOR_TARGET}={ov.get(P2R_STAND_PRIOR_TARGET)!r}, want 'restance'")
    try:
        params = restance_mod().parse_restance(ov.get(P2R_RESTANCE))
    except ValueError as exc:
        params, why = None, str(exc)
    else:
        why = "blank / 0 reads OFF"
    if params is None:
        fails.append(f"{P2R_RESTANCE}={ov.get(P2R_RESTANCE)!r} does not parse to R's constants ({why})")
    return fails


def judge_z2_dose_matches(overlay: dict, z2_dose: dict | None, dose_name: str | None = None) -> list:
    """NOTE(p2r Z2 cams-only, 2026-10-10): the dose file must have been computed for THIS chain's R -- its
    ``restance_knob`` equals the canonical spelling of the overlay's KRABBY_RESTANCE (the file is regenerated after the
    probe fixes the constants; a stale one would set the +200 band for another R)."""
    if not isinstance(z2_dose, dict):
        return []                      # the missing file is z2_dose_band's refusal
    mod = restance_mod()
    try:
        params = mod.parse_restance((overlay or {}).get(P2R_RESTANCE))
    except ValueError:
        return []                      # judge_z2_knobs reports the unparseable knob
    if params is None:
        return []
    want, got = mod.format_restance(params), z2_dose.get("restance_knob")
    # NOTE(p2r Z2c, 2026-10-10): compared as PARSED constants -- format_restance grew eps_step (default = eps_c's
    # 0.10), so a dose file written before it ("...,tilt_max=0.1") names the same R as the chain's new spelling
    try:
        same = got is not None and mod.parse_restance(str(got)) == params
    except ValueError:
        same = False
    return [] if same else [f"{dose_name or P2R_Z2_DOSE_JSON.name} was computed for R = {got!r}, the chain runs "
                            f"{want!r}"]


def judge_p2r_z2_liveness(overlay: dict, ser: dict, lo: int, hi: int, first: int,
                          z2_dose: dict | None, dose_tag: str = P2R_LOSS_PRIOR,
                          dose_name: str | None = None) -> tuple[list, list, dict]:
    """Stage p2r's Z2 +200 wiring (pure but for the RESTANCE parse; P2R_Z2_PLAN.md step 4): Z1's row / mb0 /
    finiteness checks with the realised-dose band from ``z2_dose`` (p2r_z2_dose.json; REFUSED -- a fail -- without
    it), the Z2 knobs, Loss/restance_nonzero >= 0.5 and every Loss/restance_<branch> logged with 'home' > 0 (window
    means; the held == live proxy clause is dropped -- cams-only, 2026-10-10). ``(fails, notes, live)``.
    ``dose_name``: NOTE(p2r Z2d, 2026-10-11), the dose file named in the reasons."""
    band, why = z2_dose_band(z2_dose, dose_name)
    fails, notes, live = judge_p2r_liveness(overlay, ser, lo, hi, first, dose_band=band, dose_tag=dose_tag)
    if band is None:
        fails.append(f"term liveness: the Z2 realised-dose check refuses to run: {why}")
    fails += [f"term liveness: {f}" for f in judge_z2_knobs(overlay)]
    fails += [f"term liveness: {f}" for f in judge_z2_dose_matches(overlay, z2_dose, dose_name)]
    nz = window(ser.get(P2R_RESTANCE_NONZERO_TAG), lo, hi)
    live["restance_nonzero"] = nz
    if nz is None or not nz >= P2R_RESTANCE_NONZERO_MIN - 1e-9:
        fails.append(f"term liveness: {P2R_RESTANCE_NONZERO_TAG} {nz} < {P2R_RESTANCE_NONZERO_MIN} (R's target is 0 on "
                     "most bit-1 rows: the prior is still Z1's statue)")
    shares = {b: window(ser.get(t), lo, hi) for b, t in zip(P2R_RESTANCE_BRANCHES, P2R_RESTANCE_BRANCH_TAGS)}
    live["restance_branch_shares"] = shares
    missing = [b for b, v in shares.items() if v is None]
    if missing:
        fails.append(f"term liveness: Loss/restance_<branch> not logged in the window for {missing}")
    if shares.get("home") is None or not shares["home"] > 0:
        fails.append(f"term liveness: Loss/restance_home {shares.get('home')} is not > 0 (the in-air homing branch "
                     "never fires on training stand rows)")
    return fails, notes, live


def resolve_repo_path(path: str) -> Path:
    """NOTE(p2r Z2c, 2026-10-10): a repo-relative knob path resolved against the repo root (as
    motion_knobs.resolve_anchor_ckpt); absolute paths pass through."""
    p = Path(str(path).strip())
    return p if p.is_absolute() else R / p


def bc_bank_paths(overlay: dict) -> list[Path]:
    """The chain's KRABBY_BC_BANK (comma list), each resolved; [] when absent / blank."""
    raw = str((overlay or {}).get(P2R_BC_BANK, "") or "")
    return [resolve_repo_path(x) for x in raw.split(",") if x.strip()]


def judge_z2c_bc_knobs(overlay: dict) -> list:
    """NOTE(p2r Z2c, 2026-10-10): the BC knobs a Z2c chain must carry (IO: bank existence): KRABBY_BC_W > 0, a
    non-empty KRABBY_BC_BANK whose every path exists, KRABBY_BC_BATCH a positive integer."""
    ov = overlay or {}
    fails = []
    try:
        wv = float(str(ov.get(P2R_BC_W, "")).strip())
    except ValueError:
        wv = None
    if wv is None or not math.isfinite(wv) or not wv > 0:
        fails.append(f"{P2R_BC_W}={ov.get(P2R_BC_W)!r} is not a positive weight")
    banks = bc_bank_paths(ov)
    if not banks:
        fails.append(f"{P2R_BC_BANK}={ov.get(P2R_BC_BANK)!r} names no bank")
    missing = [str(b) for b in banks if not b.is_file()]
    if missing:
        fails.append(f"{P2R_BC_BANK}: missing bank file(s) {missing}")
    bs = str(ov.get(P2R_BC_BATCH, "")).strip()
    if not (bs.isdigit() and int(bs) > 0):
        fails.append(f"{P2R_BC_BATCH}={ov.get(P2R_BC_BATCH)!r} is not a positive integer")
    return fails


def z2c_bc_band(overlay: dict, doc: dict | None, dose_name: str | None = None) -> tuple[tuple[float, float] | None,
                                                                                       list]:
    """``(band, fails)``: [0.52, 2] x the dose file's ``bc_mb0_expected`` (finite, > 0), refused when the file's
    ``bc.banks`` are not the chain's KRABBY_BC_BANK (resolved paths, as a set). ``dose_name``: NOTE(p2r Z2d,
    2026-10-11), the dose file named in the reasons (default p2r_z2_dose.json)."""
    name = dose_name or P2R_Z2_DOSE_JSON.name
    if not isinstance(doc, dict):
        return None, [f"no readable {name} (p2r_z2_dose.py --bc-bank sets the BC +200 band)"]
    try:
        exp = float(doc["bc_mb0_expected"])
    except (KeyError, TypeError, ValueError):
        return None, [f"{name} has no numeric bc_mb0_expected ({doc.get('bc_mb0_expected')!r}): rerun "
                      "p2r_z2_dose.py with --bc-bank"]
    if not (math.isfinite(exp) and exp > 0):
        return None, [f"{name} bc_mb0_expected {exp} is not a finite positive number"]
    fails = []
    got = {str(Path(b.get("path", "")).resolve()) for b in ((doc.get("bc") or {}).get("banks") or [])
           if isinstance(b, dict)}
    want = {str(p.resolve()) for p in bc_bank_paths(overlay)}
    if got != want:
        fails.append(f"{name}'s BC dose was computed on banks {sorted(got)}, the chain's {P2R_BC_BANK} is "
                     f"{sorted(want)}")
    return (P2R_BC_DOSE_BAND[0] * exp, P2R_BC_DOSE_BAND[1] * exp), fails


def judge_p2r_z2c_liveness(overlay: dict, ser: dict, lo: int, hi: int, first: int,
                           z2_dose: dict | None, dose_name: str | None = None) -> tuple[list, list, dict]:
    """NOTE(p2r Z2c, 2026-10-10): the Z2c +200 wiring (P2R_Z2C_PLAN.md step 5) -- Z2's (stand-prior dose read off
    Loss/stand_prior_mb0 when logged), the BC knobs, Loss/bc and Loss/bc_mb0 logged, finite and > 0 (window means),
    and the first-update Loss/bc_mb0 inside [0.52, 2] x the dose file's bc_mb0_expected. ``(fails, notes, live)``.
    ``dose_name``: NOTE(p2r Z2d, 2026-10-11), the dose file named in the reasons."""
    dose_tag = P2R_PRIOR_MB0_TAG if point_at(ser.get(P2R_PRIOR_MB0_TAG), first) is not None else P2R_LOSS_PRIOR
    fails, notes, live = judge_p2r_z2_liveness(overlay, ser, lo, hi, first, z2_dose, dose_tag=dose_tag,
                                               dose_name=dose_name)
    if dose_tag == P2R_LOSS_PRIOR:
        notes.append(f"{P2R_PRIOR_MB0_TAG} not logged: the stand-prior dose clause reads the first-update MEAN "
                     "Loss/stand_prior, which BC's in-update pre-fit can lower")
    fails += [f"term liveness: {f}" for f in judge_z2c_bc_knobs(overlay)]
    for tag, key in ((P2R_LOSS_BC, "loss_bc"), (P2R_BC_MB0_TAG, "bc_mb0")):
        v = window(ser.get(tag), lo, hi)
        live[key] = v
        if v is None or not math.isfinite(v) or not v > 0:
            fails.append(f"term liveness: {tag} {v} is not logged, finite and > 0 (window mean)")
    gn = window(ser.get(P2R_GRAD_NORM_BC_TAG), first, hi)
    live["grad_norm_bc"] = gn                          # reported: logged every 50 its
    band, bfails = z2c_bc_band(overlay, z2_dose, dose_name)
    fails += [f"term liveness: {f}" for f in bfails]
    dose = point_at(ser.get(P2R_BC_MB0_TAG), first)
    live["bc_mb0_first_update"] = dose
    if band is None:
        fails.append("term liveness: the BC realised-dose check refuses to run (no bc_mb0_expected)")
    else:
        live["bc_dose_band"] = list(band)
        if dose is None or not math.isfinite(dose) or not (band[0] - 1e-9 <= dose <= band[1] + 1e-9):
            fails.append(f"term liveness: BC realised dose (first-update {P2R_BC_MB0_TAG}) {dose} outside "
                         f"[{band[0]:.4g}, {band[1]:.4g}]")
    return fails, notes, live


def judge_z2d_mask_knobs(overlay: dict, z2_dose: dict | None) -> list:
    """NOTE(p2r Z2d, 2026-10-11): what a Z2d chain must carry beyond Z2c's (pure): KRABBY_OMNI_STAND_MASK=1 in the
    overlay, and a dose file (p2r_z2d_dose.json) whose BC dose was computed on MASKED bank rows (``bc.stand_mask``
    true: p2r_z2_dose.py --bc-stand-mask). The missing file is z2_dose_band's / z2c_bc_band's refusal."""
    ov = overlay or {}
    fails = []
    if str(ov.get(P2R_STAND_MASK, "")).strip() != "1":
        fails.append(f"{P2R_STAND_MASK}={ov.get(P2R_STAND_MASK)!r}, want '1' (the stand-row input mask)")
    if isinstance(z2_dose, dict) and ((z2_dose.get("bc") or {}).get("stand_mask") is not True):
        fails.append(f"{P2R_Z2D_DOSE_JSON.name}'s bc.stand_mask is {(z2_dose.get('bc') or {}).get('stand_mask')!r}, "
                     "not true (rerun p2r_z2_dose.py with --bc-stand-mask: the dose must be read on the masked rows)")
    return fails


def judge_p2r_z2d_liveness(overlay: dict, ser: dict, lo: int, hi: int, first: int,
                           z2_dose: dict | None) -> tuple[list, list, dict]:
    """NOTE(p2r Z2d, 2026-10-11): the Z2d +200 wiring (P2R_Z2D_PLAN.md step 1 "The arm wiring" / step 4 "+200
    wiring") -- every Z2c check (:func:`judge_p2r_z2c_liveness`, against p2r_z2d_dose.json), the mask knobs
    (:func:`judge_z2d_mask_knobs`), and Metrics/stand_mask_frac logged, finite and inside
    :data:`P2R_PRIOR_ROWS_BAND` (window mean; the share of rows the obs term masked -- the stand rows the prior
    reads). ``(fails, notes, live)``."""
    fails, notes, live = judge_p2r_z2c_liveness(overlay, ser, lo, hi, first, z2_dose,
                                                dose_name=P2R_Z2D_DOSE_JSON.name)
    fails += [f"term liveness: {f}" for f in judge_z2d_mask_knobs(overlay, z2_dose)]
    fr = window(ser.get(P2R_STAND_MASK_FRAC_TAG), lo, hi)
    live["stand_mask_frac"] = fr
    lo_p, hi_p = P2R_PRIOR_ROWS_BAND
    if fr is None or not math.isfinite(fr) or not (lo_p - 1e-9 <= fr <= hi_p + 1e-9):
        fails.append(f"term liveness: {P2R_STAND_MASK_FRAC_TAG} {fr} is not logged, finite and inside "
                     f"[{lo_p}, {hi_p}] (the stand-row mask is not live)")
    return fails, notes, live


def z_variant_unknown(arm: str) -> bool:
    """NOTE(p2r Z2d, 2026-10-11): a Z arm with a variant letter (``Z2E``) none of the explicit branches judges --
    refused, never read as Z1."""
    return bool(Z_ARM_RE.match(arm or "")) and arm[-1].isalpha() and arm not in P2R_Z_VARIANT_ARMS


def z2_dose_default(arm: str) -> Path:
    """NOTE(p2r Z2d, 2026-10-11): the arm's own dose file -- p2r_z2d_dose.json for Z2D, else p2r_z2_dose.json."""
    return P2R_Z2D_DOSE_JSON if arm == P2R_Z2D_ARM else P2R_Z2_DOSE_JSON


def judge_p2r_preflight(overlay: dict, graft: dict | None, fork_ckpt: str | None) -> list:
    """Stage p2r's Z-arm preflight (pure; P2R_STATUE_PLAN.md step 1 "Left for implementation": "preflight must NOT
    require the p2s replay gate's solved stand weights"): the graft's identity (CPU AND at least one GPU stamp) and
    the three agent knobs, nothing about a stand-slot dose. Returns a ``fails`` list."""
    fails = []
    if not graft:
        fails.append("no readable graft.json for this chain's fork checkpoint")
    else:
        if graft.get("identity_cpu") is not True:
            fails.append(f"graft.json identity_cpu is {graft.get('identity_cpu')!r}, not true")
        if not graft.get("identity_gpu_stamps"):
            fails.append("graft.json identity_gpu_stamps is empty (no GPU identity/inertness stamp yet)")
    ov = overlay or {}
    for knob in (P2R_ANCHOR_W, P2R_STAND_PRIOR_W):
        try:
            v = float(str(ov.get(knob, "")).strip())
        except ValueError:
            v = None
        if v is None or not v > 0:
            fails.append(f"{knob}={ov.get(knob)!r} is not a positive weight")
    ck = str(ov.get(P2R_ANCHOR_CKPT, "")).strip()
    if not ck:
        fails.append(f"{P2R_ANCHOR_CKPT} is not set")
    elif fork_ckpt and ck != str(fork_ckpt):
        fails.append(f"{P2R_ANCHOR_CKPT}={ck!r} != the chain's own fork checkpoint {fork_ckpt!r}")
    return fails


def p2s_replay_magnitude(replay: dict | None, arm: str, term: str):
    """The p2s replay magnitude CALIBRATED ON THE SAMPLED probe (P2S_PLAN step 2 / 4; TensorBoard units), or None:
    ``replay_gate.json`` ``"p2s".<arm>.terms.<term>.episode_reward_sampled`` when written, else that term's
    ``episode_reward`` ONLY when the record declares it sampled-calibrated (``"p2s".episode_reward_units`` names the
    SAMPLED REF, as scripts/p2s_replay.py writes it: weight x stand share 0.15 x the sampled REF per-stand-row mean).
    Never a deterministic value (``episode_reward_det_ref``; fail closed: the deterministic margin is ~3x the
    training-time one)."""
    p2s = (replay or {}).get("p2s") or {}
    ent = (((p2s.get(arm) or {}).get("terms") or {}).get(term)) or {}
    v = ent.get("episode_reward_sampled")
    if v is None and "SAMPLED" in str(p2s.get("episode_reward_units", "")).upper():
        v = ent.get("episode_reward")
    return v if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def p2s_replay_weights(replay: dict | None, arm: str) -> dict:
    """The replay gate's solved dose for ``arm``: ``"p2s".<arm>.knobs`` (scripts/p2s_replay.py) or ``.weights``."""
    ent = ((replay or {}).get("p2s") or {}).get(arm) or {}
    return dict(ent.get("knobs") or ent.get("weights") or {})


def p2s_replay_verdict_ok(replay: dict | None) -> bool:
    """CHARTER 11 / P2S_PLAN step 4: the p2s replay gate's verdict is PASS, or a recorded user overrule."""
    ent = (replay or {}).get("p2s") or {}
    return ent.get("verdict") == "PASS" or (ent.get("overrule") or {}).get("by") == "user"


def p2s_stand_share(overlay: dict) -> tuple[float | None, str | None]:
    """``(share, why)``: the chain env's stand-slot share (KRABBY_OMNI_SLOTS' first field), which must equal the
    pinned :data:`P2S_STAND_SHARE`; ``why`` is a failure reason or None."""
    raw = str((overlay or {}).get("KRABBY_OMNI_SLOTS", "")).strip()
    try:
        share = float(raw.split(":")[0])
    except ValueError:
        return None, f"KRABBY_OMNI_SLOTS {raw!r} unreadable: no stand-slot share"
    if abs(share - P2S_STAND_SHARE) > 1e-9:
        return share, f"stand-slot share {share} (KRABBY_OMNI_SLOTS first field) != the pinned {P2S_STAND_SHARE}"
    return share, None


def p2s_weights(overlay: dict) -> dict:
    """``{term: float weight or None}`` from the chain env's two lever knobs."""
    out = {}
    for t, k in P2S_WEIGHT_KNOBS.items():
        try:
            out[t] = float(str((overlay or {}).get(k, "")).strip())
        except ValueError:
            out[t] = None
    return out


def window(series, lo, hi):
    """Mean of [(step, value)] over lo..hi inclusive, or None."""
    vals = [v for s, v in (series or []) if lo <= s <= hi]
    return sum(vals) / len(vals) if vals else None


def tail_mean(series, n=ROLL):
    vals = [v for _, v in (series or [])][-n:]
    return sum(vals) / len(vals) if vals else None


def first_last(series_by_tag: dict) -> tuple[int | None, int | None]:
    steps = [s for ser in series_by_tag.values() for s, _ in (ser or [])]
    return (min(steps), max(steps)) if steps else (None, None)


def _yaml_tolerant(path: Path):
    import yaml  # noqa: PLC0415

    class _L(yaml.SafeLoader):
        pass

    _L.add_constructor("tag:yaml.org,2002:python/tuple", lambda ld, node: tuple(ld.construct_sequence(node)))
    _L.add_multi_constructor("", lambda ld, suffix, node: None)
    with open(path) as fh:
        return yaml.load(fh, Loader=_L)  # noqa: S506


def _find_key(d, key):
    """First value of ``key`` anywhere in a nested dict (depth-first), else None."""
    if isinstance(d, dict):
        if key in d:
            return d[key]
        for v in d.values():
            r = _find_key(v, key)
            if r is not None:
                return r
    return None


def read_tb(run_dir: Path, extra_tags=()) -> dict:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator  # noqa: PLC0415

    ea = EventAccumulator(str(run_dir), size_guidance={"scalars": 0})
    ea.Reload()
    have = set(ea.Tags().get("scalars", []))
    want = set(TAGS.values()) | set(OBST_LEVEL_TAGS) | set(PROMO_TAGS) | set(F1_NONSTAND_TAGS) | set(extra_tags) | \
        {f"Episode_Reward/{t}" for t in ARM_TERMS.values()} | \
        {f"Episode_Reward/{t}" for terms in P2B_ARM_TERMS.values() for t in terms} | \
        {f"Episode_Reward/{t}" for t in P2S_TERMS} | {P2S_STAND_FRAC_TAG} | \
        {P2R_LOSS_ANCHOR, P2R_LOSS_PRIOR, P2R_ANCHOR_ROWS_TAG, P2R_PRIOR_ROWS_TAG, P2R_ANCHOR_MB0_TAG} | \
        {P2R_RESTANCE_NONZERO_TAG} | set(P2R_RESTANCE_BRANCH_TAGS) | \
        {P2R_LOSS_BC, P2R_BC_MB0_TAG, P2R_GRAD_NORM_BC_TAG, P2R_PRIOR_MB0_TAG} | \
        {P2R_STAND_MASK_FRAC_TAG}
    return {t: [(int(e.step), float(e.value)) for e in ea.Scalars(t)] for t in want if t in have}


def load_rc(path: Path | None = None):
    spec = importlib.util.spec_from_file_location("_p2_run_candidates", path or RC_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_p2_run_candidates"] = mod
    spec.loader.exec_module(mod)
    return mod


def chain_overlay(cands: dict, chain: str) -> dict:
    ov: dict = {}
    for st in ((cands.get("chains") or {}).get(chain) or {}).get("stages") or []:
        ov.update({str(k): str(v) for k, v in (st.get("overlay") or {}).items()})
    return ov


def chain_source_ckpt(cands: dict, chain: str) -> str | None:
    st = (((cands.get("chains") or {}).get(chain) or {}).get("stages") or [{}])[0]
    rf = st.get("resume_from")
    return str(rf["checkpoint"]) if isinstance(rf, dict) and rf.get("checkpoint") else None


def s1_p2a_tail(arms_dir: Path, seed: int, n: int = P2B_TAIL_N) -> dict:
    """TG's floor (P2B_PLAN.md step 1): the seed's S1 p2a training run's last ``n`` its of
    Episode_Reward/reward_orientation (IO; the record's ``run_dir`` comes from scripts/run_arm.sh's write_record).
    ``{}`` when the record or its run dir is unreadable (judge_wiring then fails closed on the missing floor)."""
    try:
        rec = json.load(open(Path(arms_dir) / f"P2__S1_s{seed}.record.json"))
    except (OSError, ValueError):
        return {}
    rd = rec.get("run_dir")
    if not rd or not Path(rd).is_dir():
        return {}
    ser = read_tb(Path(rd), extra_tags=("Episode_Reward/reward_orientation",))
    return {"reward_orientation": tail_mean(ser.get("Episode_Reward/reward_orientation"), n)}


P2S_GUARD_TAGS = (TAGS["ep_len"], TAGS["crab_failure"], TAGS["std"])


def p2s_reference(arms_dir: Path, seed: int | None) -> dict:
    """The p2s guards' reference (P2S_PLAN step 6 / "left for implementation"; IO, records only): at s18 the C0
    continuation's run (``P2__C0_s18.record.json``, the X12 run) read window for window; at s16 / s17 the seed's own S1
    p2a source run (``P2__S1_s<seed>.record.json``) as a constant tail. ``{"error": why}`` when unresolvable."""
    if seed is None:
        return {"error": "the chain name carries no seed"}
    name = P2S_REF_S18 if seed == P2S_GATE_SEED else f"P2__S1_s{seed}"
    try:
        rec = json.load(open(Path(arms_dir) / f"{name}.record.json"))
    except (OSError, ValueError):
        return {"error": f"no readable {name}.record.json in {arms_dir}", "name": name}
    if rec.get("killed"):
        return {"error": f"{name} was killed ({rec.get('killed')})", "name": name}
    rd = rec.get("run_dir")
    if not rd or not Path(rd).is_dir():
        return {"error": f"{name}'s run_dir {rd!r} is not a directory", "name": name}
    return {"kind": "run" if seed == P2S_GATE_SEED else "tail", "name": name, "run_dir": str(rd)}


def load_p2s_reference(ref: dict, reader=None) -> dict:
    """:func:`p2s_reference` + its telemetry (IO): ``"run"`` -> ``ser`` (the three guard tags) and ``first`` (its first
    logged iteration); ``"tail"`` -> ``tail`` = the mean of each tag over the run's last :data:`P2S_REF_TAIL_N` its.
    A missing tag is an ``error`` (the guards would otherwise run against nothing)."""
    if not ref or ref.get("error"):
        return ref or {"error": "no reference"}
    ser = (reader or read_tb)(Path(ref["run_dir"]))
    missing = [t for t in P2S_GUARD_TAGS if not ser.get(t)]
    if missing:
        return dict(ref, error=f"{ref['name']}'s run lacks {missing}")
    if ref["kind"] == "tail":
        return dict(ref, tail={t: tail_mean(ser[t], P2S_REF_TAIL_N) for t in P2S_GUARD_TAGS})
    first, _ = first_last({t: ser[t] for t in P2S_GUARD_TAGS})
    return dict(ref, ser={t: ser[t] for t in P2S_GUARD_TAGS}, first=first)


def p2s_reference_ok(ref: dict) -> tuple[bool, str]:
    """``(readable, why)`` of a :func:`load_p2s_reference` result, for the preflight."""
    if not ref or ref.get("error"):
        return False, (ref or {}).get("error", "no reference")
    if ref["kind"] == "tail":
        return True, (f"{ref['name']} constant tail (last {P2S_REF_TAIL_N} its): "
                      + ", ".join(f"{t.rsplit('/', 1)[-1]} {v:.4g}" for t, v in ref["tail"].items()))
    return True, f"{ref['name']} run {Path(ref['run_dir']).name} window for window (first it {ref['first']})"


# ------------------------------------------------------------------------------------------------ pure judges
def dump_facts(env: dict | None, agent: dict | None, log_text: str | None) -> dict:
    """The dump-side facts of the wiring read, from the parsed params/env.yaml (+ the training log)."""
    if not env:
        return {"have_env": False}
    rew = env.get("rewards") or {}
    ev = env.get("events") or {}

    def w(term):
        t = rew.get(term)
        return (t or {}).get("weight") if isinstance(t, dict) else None

    push = ((ev.get("push_by_setting_velocity") or {}).get("params") or {}).get("velocity_range") or {}
    subs = _find_key(env.get("scene") or env, "sub_terrains") or {}
    fns = sorted({str((v or {}).get("function", "")).rsplit(":", 1)[-1] for v in subs.values() if isinstance(v, dict)})
    rows = _find_key(env.get("scene") or env, "num_rows")
    funcs = {t: str((rew.get(t) or {}).get("func")) if isinstance(rew.get(t), dict) else None
             for t in tuple(ARM_TERMS.values()) + P2S_TERMS}
    return {"have_env": True, "term_funcs": funcs, "curriculum_null": "curriculum" in env and env.get("curriculum") is None,
            "weights": {t: w(t) for t in list(ELEMENT_WEIGHTS) + ["reward_clock_schedule", "reward_clock_swing_apex",
                                                                    "reward_action_rate"] + list(ARM_TERMS.values())},
            # present = a cfg block, not merely the key: a disabled DR event dumps as `name: null` (pre-launch
            # audit M3, 2026-10-06), and `e in ev` read that as present
            "dr_events": {e: isinstance(ev.get(e), dict) for e in DR_EVENTS},
            "push_x": list(push.get("x") or []) if isinstance(push, dict) else [],
            "terrain_fns": fns, "num_rows": rows,
            "dropped_ppo_keys": dropped_ppo_keys(log_text)}


#: The runner's documented, deliberately ignored PPO key (the PPO-construction landing allow-list; every B run logs
#: it). Any OTHER key in an "Ignoring unsupported PPO config keys" line is a dropped knob.
PPO_KEY_ALLOW = frozenset({"share_cnn_encoders"})


def dropped_ppo_keys(log_text: str | None) -> list[str]:
    """Keys the runner ignored, minus :data:`PPO_KEY_ALLOW` (pure). A line whose key list cannot be parsed counts as
    dropped (fail closed)."""
    out = []
    for ln in (log_text or "").splitlines():
        if "Ignoring unsupported PPO config keys" not in ln:
            continue
        clean = re.sub(r"\x1b\[[0-9;]*m", "", ln)
        tail = clean.rsplit(":", 1)[-1] if ":" in clean else ""
        keys = [k.strip() for k in tail.split(",") if k.strip()]
        bad = [k for k in keys if k not in PPO_KEY_ALLOW]
        if bad or not keys:
            out.append(clean.strip())
    return out


def _close(a, b, tol=1e-9):
    return a is not None and b is not None and abs(float(a) - float(b)) <= tol


def judge_wiring(arm: str, overlay: dict, facts: dict, ser: dict, first: int, audit: list, replay: dict | None,
                 stand_gate: tuple[bool, str] | None = None, ref_tail: dict | None = None,
                 p2s_gate: dict | None = None, z2_dose: dict | None = None) -> dict:
    """CHARTER 8 wiring at +200 (pure). ``audit`` = run_candidates.overlay_checks output [(name, ok)];
    ``stand_gate`` = :func:`stand_gate_structural` of the code root (F1 only; None = not read = FAIL for F1);
    ``ref_tail`` = the seed's S1 p2a training tail (TG only; P2B_PLAN.md step 1, e.g. {"reward_orientation": x});
    ``p2s_gate`` = :func:`p2s_gate_structural` of the code root (ST1 / ST2 only; None = not read = FAIL);
    ``z2_dose`` = p2r_z2_dose.json (the Z2 / Z2C arms; p2r_z2d_dose.json for Z2D -- NOTE(p2r Z2d, 2026-10-11);
    None = not read = the dose check refuses)."""
    last = max((s for v in ser.values() for s, _ in v), default=None)
    lo, hi = first + W200[0], first + W200[1]
    if last is None or last < hi:
        return {"rc": RC_WAIT, "why": f"telemetry only to it {last} (window {lo}-{hi})"}
    fails, notes = [], []
    if not facts.get("have_env"):
        return {"rc": RC_FAIL, "result": "FAIL", "fails": ["params/env.yaml missing or unreadable"]}
    if not facts["curriculum_null"]:
        fails.append("top-level curriculum is not null (a PHASEOUT ramp survived)")
    ch = window(ser.get(TAGS["clock_hidden"]), lo, hi)
    if ch is not None and ch < 0.999:
        fails.append(f"Metrics/clock_hidden_frac {ch:.3f} < 0.999")
    wts = facts["weights"]
    for t in ("reward_clock_schedule", "reward_clock_swing_apex"):
        if not _close(wts.get(t), CLOCK_W):
            fails.append(f"{t} weight {wts.get(t)} != {CLOCK_W}")
    want_ar = float(overlay.get("KRABBY_ACTION_RATE_W", "0") or 0)
    if not _close(wts.get("reward_action_rate"), want_ar):
        fails.append(f"reward_action_rate weight {wts.get('reward_action_rate')} != {want_ar} (arm {arm})")
    for t, want in ELEMENT_WEIGHTS.items():
        if not _close(wts.get(t), want):
            fails.append(f"{t} weight {wts.get(t)} != {want}")
    fails += [f"DR event {e} absent" for e, ok in facts["dr_events"].items() if not ok]
    px = facts.get("push_x") or []
    if len(px) != 2 or not (_close(px[0], -PUSH) and _close(px[1], PUSH)):
        fails.append(f"push velocity x range {px} != [-{PUSH}, {PUSH}]")
    geom = str(overlay.get("KRABBY_FLAT_TERRAIN_GEOM", "")).strip().lower()
    fns = set(facts.get("terrain_fns") or [])
    want_fn, other = (FIELD_FN, RING_FN) if geom == "field" else (RING_FN, FIELD_FN)
    if want_fn not in fns or other in fns:
        fails.append(f"geometry {geom!r}: sub-terrain functions {sorted(fns)} (want {want_fn}, not {other})")
    ro = window(ser.get(TAGS["reach_obst"]), lo, hi)
    if ro is None:
        fails.append("Metrics/base_parkour/reach_obst_frac not logged")
    elif not ro > 0:
        fails.append(f"reach_obst_frac {ro} not > 0")
    if facts.get("dropped_ppo_keys"):
        fails.append(f"the runner dropped PPO keys: {facts['dropped_ppo_keys'][0][:160]}")
    if not audit:
        fails.append("overlay audit produced no checks")
    fails += [f"overlay audit: {name}" for name, ok in audit if not ok]
    term = ARM_TERMS.get(arm)
    live = {}
    if term:
        er = window(ser.get(f"Episode_Reward/{term}"), lo, hi)
        live = {"term": term, "episode_reward": er}
        if er is None or er == 0.0:
            fails.append(f"term liveness: Episode_Reward/{term} is {er} (must be non-zero)")
        elif arm in REPLAY_TERMS:
            exp = ((replay or {}).get(arm) or {}).get("episode_reward")
            live["replay_expected"] = exp
            if exp in (None, 0, 0.0):
                fails.append(f"term liveness: no CHARTER 4 replay magnitude for {arm} (replay_gate.json)")
            elif er / exp <= 0 or not (1.0 / LIVE_RATIO - 1e-9 <= er / exp <= LIVE_RATIO + 1e-9):
                fails.append(f"term liveness: Episode_Reward/{term} {er:.4g} vs replay {exp:.4g} (outside 3x)")
        if arm == "F1":
            ns = [window(ser.get(t), lo, hi) for t in F1_NONSTAND_TAGS if ser.get(t)]
            if ns:
                if any(v is None or v != 0.0 for v in ns):
                    fails.append(f"term liveness: F1 pays outside the stand slot ({ns})")
            else:                                    # CHARTER 9 reading: stand-only by construction
                func = (facts.get("term_funcs") or {}).get(term)
                if func != F1_FUNC:
                    fails.append(f"term liveness: F1 term func {func!r} != {F1_FUNC} (stand-only construction unshown)")
                if not stand_gate or not stand_gate[0]:
                    fails.append("term liveness: F1 stand-only construction not shown "
                                 f"({stand_gate[1] if stand_gate else 'not read'})")
                if func == F1_FUNC and stand_gate and stand_gate[0]:
                    notes.append(f"F1 stand-only by construction (CHARTER 9): {stand_gate[1]}")
    elif arm in P2B_ARM_TERMS:
        terms = P2B_ARM_TERMS[arm]
        live = {"terms": {}}
        for t in terms:
            er = window(ser.get(f"Episode_Reward/{t}"), lo, hi)
            sign = P2B_TERM_SIGN[t]
            live["terms"][t] = {"episode_reward": er}
            if er is None or er == 0.0 or (sign > 0 and er < 0) or (sign < 0 and er > 0):
                fails.append(f"term liveness: Episode_Reward/{t} is {er} (must be "
                             f"{'non-negative' if sign > 0 else 'non-positive'} and non-zero, arm {arm})")
                continue
            exp = p2b_replay_magnitude(replay, arm, t)
            live["terms"][t]["replay_expected"] = exp
            if exp in (None, 0, 0.0):
                fails.append(f"term liveness: no p2b replay magnitude for {arm}/{t} (replay_gate.json)")
            elif er / exp <= 0 or not (1.0 / LIVE_RATIO - 1e-9 <= er / exp <= LIVE_RATIO + 1e-9):
                fails.append(f"term liveness: Episode_Reward/{t} {er:.4g} vs replay {exp:.4g} (outside 3x)")
        if arm == "TG":
            er = live["terms"].get("reward_orientation", {}).get("episode_reward")
            tail = (ref_tail or {}).get("reward_orientation")
            live["ref_tail"] = tail
            if tail in (None, 0, 0.0):
                fails.append("term liveness: no seed S1 p2a run tail for TG's floor (reward_orientation)")
            elif er is not None and abs(er) < P2B_TG_FLOOR_RATIO * abs(tail):
                fails.append(f"term liveness: TG |Episode_Reward/reward_orientation| {er:.4g} < "
                             f"{P2B_TG_FLOOR_RATIO}x the seed's S1 p2a tail {tail:.4g}")
    elif arm in P2S_ARMS:
        f2, n2, live = judge_p2s_liveness(arm, overlay, ser, lo, hi, replay)
        fails += f2
        notes += n2
        f3, n3 = judge_p2s_construction(facts, p2s_gate)
        fails += f3
        notes += n3
    elif arm == P2R_Z2_ARM:
        f2, n2, live = judge_p2r_z2_liveness(overlay, ser, lo, hi, first, z2_dose)
        fails += f2
        notes += n2
    elif arm == P2R_Z2C_ARM:                         # NOTE(p2r Z2c, 2026-10-10): Z2's + the BC term
        f2, n2, live = judge_p2r_z2c_liveness(overlay, ser, lo, hi, first, z2_dose)
        fails += f2
        notes += n2
    elif arm == P2R_Z2D_ARM:                         # NOTE(p2r Z2d, 2026-10-11): Z2c's + the stand-row mask
        f2, n2, live = judge_p2r_z2d_liveness(overlay, ser, lo, hi, first, z2_dose)
        fails += f2
        notes += n2
    elif z_variant_unknown(arm):                     # NOTE(p2r Z2d, 2026-10-11): never Z1's checks by default
        fails.append(f"term liveness: Z arm {arm!r} has a variant letter no explicit branch judges "
                     f"(known: {list(P2R_Z_VARIANT_ARMS)}): refused rather than read as Z1")
    elif Z_ARM_RE.match(arm):
        f2, n2, live = judge_p2r_liveness(overlay, ser, lo, hi, first)
        fails += f2
        notes += n2
    else:
        notes.append(f"arm {arm}: no arm-specific term (liveness n/a)")
    return {"rc": RC_FAIL if fails else RC_PASS, "result": "FAIL" if fails else "PASS", "fails": fails,
            "notes": notes, "reach_obst": ro, "clock_hidden": ch, "liveness": live, "window": [lo, hi]}


def _calls_stand_mask(node) -> bool:
    return any(isinstance(c, ast.Call) and isinstance(c.func, ast.Attribute) and c.func.attr == "stand_mask"
               for c in ast.walk(node))


def _gating_helpers(tree) -> set[str]:
    """Module-level functions that GATE on the stand mask: a name in them is bound to a ``.stand_mask()`` call and every
    ``return`` keeps that name as a conjunct -- a factor of ``*`` / ``&``, through ``.to(...)`` / ``.float()`` /
    ``.type(...)`` / ``.bool()`` casts (``_stand_gate``: ``return (stand & upright).to(dtype)``)."""
    out = set()
    for fn in tree.body:
        if not isinstance(fn, ast.FunctionDef):
            continue
        gates = {t.id for n in ast.walk(fn) if isinstance(n, ast.Assign) and _calls_stand_mask(n.value)
                 for t in n.targets if isinstance(t, ast.Name)}
        rets = [n for n in ast.walk(fn) if isinstance(n, ast.Return)]
        if not gates or not rets:
            continue

        def conj(e):
            if isinstance(e, ast.BinOp) and isinstance(e.op, (ast.Mult, ast.BitAnd)):
                return conj(e.left) + conj(e.right)
            if isinstance(e, ast.Call) and isinstance(e.func, ast.Attribute) and e.func.attr in ("to", "float", "type",
                                                                                                "bool"):
                return conj(e.func.value)
            return [e]
        if all(r.value is not None and {f.id for f in conj(r.value) if isinstance(f, ast.Name)} & gates for r in rets):
            out.add(fn.name)
    return out


def stand_gate_structural(rewards_py: Path, func_name: str | None = None) -> tuple[bool, str]:
    """A stand-only construction, read from source (CPU, no import): in ``func_name`` (default F1's
    ``reward_stand_stance_quality``) a name is bound to an expression calling ``.stand_mask()`` -- or calling a
    module-level helper that itself gates on it (:func:`_gating_helpers`; the p2s pair's ``_stand_gate``, P2S_PLAN step
    3: "generalise F1's stand-only AST check to the motion-rewards module") -- and every ``return`` is a product with
    that name as a factor."""
    func_name = func_name or ARM_TERMS["F1"]
    try:
        tree = ast.parse(Path(rewards_py).read_text())
    except (OSError, SyntaxError) as e:
        return False, f"{rewards_py} unreadable ({e.__class__.__name__})"
    fn = next((n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == func_name), None)
    if fn is None:
        return False, f"no def {func_name} in {rewards_py}"
    helpers = _gating_helpers(tree) - {func_name}

    def gate_value(v) -> bool:
        return _calls_stand_mask(v) or any(isinstance(c, ast.Call) and isinstance(c.func, ast.Name)
                                           and c.func.id in helpers for c in ast.walk(v))
    gates = {t.id for n in ast.walk(fn) if isinstance(n, ast.Assign) and gate_value(n.value)
             for t in n.targets if isinstance(t, ast.Name)}
    if not gates:
        return False, "no name bound to a stand_mask() call (or to a stand-mask gating helper)"

    def factors(e):
        if isinstance(e, ast.BinOp) and isinstance(e.op, ast.Mult):
            return factors(e.left) + factors(e.right)
        return [e]
    rets = [n for n in ast.walk(fn) if isinstance(n, ast.Return)]
    if not rets:
        return False, "no return"
    for r in rets:
        names = {f.id for f in factors(r.value) if isinstance(f, ast.Name)} if r.value is not None else set()
        if not names & gates:
            return False, f"a return (line {r.lineno}) is not multiplied by the stand mask {sorted(gates)}"
    via = sorted(helpers & {c.func.id for c in ast.walk(fn) if isinstance(c, ast.Call) and isinstance(c.func, ast.Name)})
    return True, (f"every return of {fn.name} is multiplied by {sorted(gates)} (= stand_mask()"
                  + (f" via {via}" if via else "") + ")")


def p2s_gate_structural(motion_rewards_py: Path) -> dict:
    """``{term: (ok, why)}`` -- :func:`stand_gate_structural` of both p2s terms in the motion-rewards module."""
    return {t: stand_gate_structural(motion_rewards_py, t) for t in P2S_TERMS}


def judge_p2s_construction(facts: dict, p2s_gate: dict | None) -> tuple[list, list]:
    """Stand-only by construction (P2S_PLAN step 3): both dumped funcs are the motion-rewards module's and both
    functions gate every return on the stand mask. ``(fails, notes)``."""
    fails, notes = [], []
    funcs = facts.get("term_funcs") or {}
    for t in P2S_TERMS:
        want = f"{P2S_FUNC_MOD}:{t}"
        if funcs.get(t) != want:
            fails.append(f"term liveness: {t} dumped func {funcs.get(t)!r} != {want} (stand-only construction unshown)")
        g = (p2s_gate or {}).get(t)
        if not g or not g[0]:
            fails.append(f"term liveness: {t} stand-only construction not shown ({g[1] if g else 'not read'})")
        elif funcs.get(t) == want:
            notes.append(f"{t} stand-only by construction: {g[1]}")
    return fails, notes


def judge_p2s_liveness(arm: str, overlay: dict, ser: dict, lo: int, hi: int, replay: dict | None) -> tuple[list, list,
                                                                                                         dict]:
    """P2S_PLAN step 3 term liveness over the +200 window (pure). ``(fails, notes, live)``."""
    fails, notes = [], []
    live: dict = {"terms": {}}
    share, why = p2s_stand_share(overlay)
    live["stand_share"] = share
    if why:
        fails.append(f"term liveness: {why}")
    w = p2s_weights(overlay)
    for t in P2S_TERMS:
        er = window(ser.get(f"Episode_Reward/{t}"), lo, hi)
        exp = p2s_replay_magnitude(replay, arm, t)
        live["terms"][t] = {"episode_reward": er, "replay_sampled": exp, "weight": w.get(t)}
        if er is None or er == 0.0:
            fails.append(f"term liveness: Episode_Reward/{t} is {er} (must be non-zero, arm {arm})")
        if exp in (None, 0, 0.0):
            fails.append(f"term liveness: no sampled-probe replay magnitude for p2s {arm}/{t} (replay_gate.json "
                         "\"p2s\": none recorded or not declared sampled-calibrated; fail closed)")
    er, exp = live["terms"][P2S_SLIDE]["episode_reward"], live["terms"][P2S_SLIDE]["replay_sampled"]
    if er is not None and er > 0:
        fails.append(f"term liveness: Episode_Reward/{P2S_SLIDE} {er:.4g} > 0 (a penalty must be non-positive)")
    elif er not in (None, 0.0) and exp not in (None, 0, 0.0):
        r = er / exp
        live["terms"][P2S_SLIDE]["ratio"] = r
        if r <= 0 or not (1.0 / LIVE_RATIO - 1e-9 <= r <= LIVE_RATIO + 1e-9):
            fails.append(f"term liveness: Episode_Reward/{P2S_SLIDE} {er:.4g} vs sampled replay {exp:.4g} "
                         f"(outside {LIVE_RATIO:g}x)")
    er, exp = live["terms"][P2S_SUPPORT]["episode_reward"], live["terms"][P2S_SUPPORT]["replay_sampled"]
    ws = w.get(P2S_SUPPORT)
    if er not in (None, 0.0) and exp not in (None, 0, 0.0):
        live["terms"][P2S_SUPPORT]["ratio_reported"] = er / exp
    if ws is None or not ws > 0:
        fails.append(f"term liveness: KRABBY_STAND_SUPPORT_W {overlay.get('KRABBY_STAND_SUPPORT_W')!r} is not a "
                     "positive weight (the implied sum s_f needs it)")
    elif er is not None and share is not None and share > 0:
        realised = window(ser.get(P2S_STAND_FRAC_TAG), lo, hi)
        live["implied_sum_s_f_pinned"] = P2S_TRIPOD + er / (ws * share)
        live["stand_share_realised"] = realised
        use = realised if realised is not None and realised > 0 else share
        implied = P2S_TRIPOD + er / (ws * use)
        live["implied_sum_s_f"] = implied
        lo_b, hi_b = P2S_SUPPORT_SUM_BAND
        src = "realised" if use is realised else "pinned"
        if not (lo_b - 1e-9 <= implied <= hi_b + 1e-9):
            fails.append(f"term liveness: implied mean sum s_f {implied:.3f} = 3 + {er:.4g} / ({ws:g} x {use:g} {src}) "
                         f"outside [{lo_b}, {hi_b}]")
        else:
            notes.append(f"p2s implied mean sum s_f {implied:.3f} in [{lo_b}, {hi_b}] (share {use:g} {src}; pinned "
                         f"{share:g} gives {live['implied_sum_s_f_pinned']:.3f})")
    return fails, notes, live


def emitted_metric_keys(code_root: Path, keys) -> dict:
    """{key: [files]} -- the metric keys found as string literals in the non-experiment parkour sources."""
    import ast  # noqa: PLC0415

    want, found = set(keys), {k: [] for k in keys}
    for d in SOURCE_DIRS:
        for f in sorted((Path(code_root) / d).rglob("*.py")):
            if "experiments" in f.parts or "tests" in f.parts:
                continue
            try:
                txt = f.read_text(errors="replace")
            except OSError:
                continue
            if not any(k in txt for k in want):
                continue
            try:
                lits = {n.value for n in ast.walk(ast.parse(txt)) if isinstance(n, ast.Constant) and isinstance(n.value, str)}
            except SyntaxError:
                continue
            for k in want & lits:
                found[k].append(str(f.relative_to(code_root)))
    return found


def judge_preflight(arm: str, emitted: dict, replay: dict | None, stand_gate: tuple[bool, str] | None,
                    overlay: dict | None = None, p2s_gate: dict | None = None,
                    p2s_ref: tuple[bool, str] | None = None, p2r_graft: dict | None = None,
                    fork_ckpt: str | None = None, z2_dose: dict | None = None) -> dict:
    """What the live reads need before a launch (pure). ``emitted`` = :func:`emitted_metric_keys` output; for a p2s
    arm also the chain ``overlay``, ``p2s_gate`` (:func:`p2s_gate_structural`) and ``p2s_ref`` = ``(readable, why)``
    of the guards' reference run (:func:`p2s_reference_ok`); for a Z arm, ``overlay``, ``p2r_graft`` (the fork
    checkpoint's graft.json, or None) and ``fork_ckpt`` (the chain's own fork checkpoint path, for the
    KRABBY_ANCHOR_CKPT == fork check) -- :func:`judge_p2r_preflight`, NOT a p2s replay-gate verdict (no stand
    terms, P2R_STATUE_PLAN.md step 1 "Left for implementation")."""
    fails, notes = [], []
    if arm != "CAL1":                                 # every arm (+1000 terrain read) and CAL2 (levels must move)
        for tag in OBST_LEVEL_TAGS + PROMO_TAGS:
            key = tag[len(METRIC_PREFIX):]
            if not emitted.get(key):
                fails.append(f"metric {tag} is not emitted by the code root (no '{key}' literal in {list(SOURCE_DIRS)}): "
                             f"the +1000 terrain read would fail closed")
    if arm in REPLAY_TERMS:
        ent = (replay or {}).get(arm) or {}
        exp = ent.get("episode_reward")
        if not isinstance(exp, (int, float)) or exp == 0:
            fails.append(f"no CHARTER 4 replay magnitude for {arm} in replay_gate.json ({exp!r})")
        # CHARTER 4: a FAIL stops and asks -- the arm may launch only on a PASS or a recorded user overrule
        # ({"overrule": {"by": "user", "date": ..., "note": ...}} written by hand after the user's decision).
        ov = ent.get("overrule") or {}
        if ent.get("verdict") != "PASS" and ov.get("by") != "user":
            fails.append(f"CHARTER 4 replay verdict for {arm} is {ent.get('verdict')!r} with no recorded user overrule")
    if arm in P2B_ARM_TERMS:
        # CHARTER 10 / P2B_PLAN.md step 2: CL/TG may launch only on a p2b replay-gate PASS or a recorded user
        # overrule, and only once every one of the arm's terms has its own non-zero magnitude recorded.
        for t in P2B_ARM_TERMS[arm]:
            exp = p2b_replay_magnitude(replay, arm, t)
            if not isinstance(exp, (int, float)) or exp == 0:
                fails.append(f"no p2b replay magnitude for {arm}/{t} in replay_gate.json ({exp!r})")
        if not p2b_replay_verdict_ok(replay, arm):
            ent = ((replay or {}).get("p2b") or {}).get(arm) or {}
            fails.append(f"p2b replay verdict for {arm} is {ent.get('verdict')!r} with no recorded user overrule")
    if arm in P2S_ARMS:
        # CHARTER 11 / P2S_PLAN.md step 4: a p2s arm launches only on a p2s replay-gate PASS (or a recorded user
        # overrule), with both terms' sampled-probe magnitudes recorded for its dose and the replay's solved weights
        # in the chain (the planning doses of candidates v10 are not the final ones).
        if not p2s_replay_verdict_ok(replay):
            ent = (replay or {}).get("p2s") or {}
            fails.append(f"p2s replay verdict is {ent.get('verdict')!r} with no recorded user overrule "
                         "(replay_gate.json \"p2s\")")
        for t in P2S_TERMS:
            if p2s_replay_magnitude(replay, arm, t) in (None, 0, 0.0):
                fails.append(f"no sampled-probe replay magnitude for p2s {arm}/{t} in replay_gate.json "
                             "(\"p2s\".<dose>.terms.<term>: episode_reward_sampled, or episode_reward under "
                             "episode_reward_units naming the SAMPLED REF)")
        rw = p2s_replay_weights(replay, arm)
        cw = p2s_weights(overlay or {})
        for t, k in P2S_WEIGHT_KNOBS.items():
            try:
                got = float(rw.get(k))
            except (TypeError, ValueError):
                got = None
            if got is None or cw.get(t) is None or abs(got - cw[t]) > 1e-9:
                fails.append(f"chain {k}={(overlay or {}).get(k)!r} != replay_gate.json \"p2s\".{arm}.knobs.{k}="
                             f"{rw.get(k)!r} (the replay gate's solved dose)")
        _share, why = p2s_stand_share(overlay or {})
        if why:
            fails.append(why)
        for t in P2S_TERMS:
            g = (p2s_gate or {}).get(t)
            if not g or not g[0]:
                fails.append(f"{t} stand-only construction not shown ({g[1] if g else 'not read'})")
        if not p2s_ref or not p2s_ref[0]:
            fails.append(f"p2s guard reference unreadable ({p2s_ref[1] if p2s_ref else 'not read'})")
        else:
            notes.append(f"p2s guard reference: {p2s_ref[1]}")
    if Z_ARM_RE.match(arm):
        fails += judge_p2r_preflight(overlay or {}, p2r_graft, fork_ckpt)
    if z_variant_unknown(arm):
        # NOTE(p2r Z2d, 2026-10-11): a lettered Z arm with no explicit branch would read as Z1 at +200: refused
        fails.append(f"Z arm {arm!r} has a variant letter no explicit branch judges (known: "
                     f"{list(P2R_Z_VARIANT_ARMS)})")
    # NOTE(p2r Z2d, 2026-10-11): Z2D's dose file is its own (named in the reasons)
    dose_name = z2_dose_default(arm).name
    if arm in P2R_Z2_FAMILY:
        # NOTE(p2r Z2, 2026-10-10): the Z2 knobs and step 4's dose file, so the +200 dose check cannot refuse later
        fails += judge_z2_knobs(overlay or {})
        _band, why = z2_dose_band(z2_dose, dose_name)
        if why:
            fails.append(why)
        fails += judge_z2_dose_matches(overlay or {}, z2_dose, dose_name)
    if arm in (P2R_Z2C_ARM, P2R_Z2D_ARM):
        # NOTE(p2r Z2c, 2026-10-10): the BC knobs and the dose file's BC block (bc_mb0_expected on the chain's banks)
        fails += judge_z2c_bc_knobs(overlay or {})
        _band, bfails = z2c_bc_band(overlay or {}, z2_dose, dose_name)
        fails += bfails
    if arm == P2R_Z2D_ARM:
        # NOTE(p2r Z2d, 2026-10-11): the mask knob and the dose file's masked BC block
        fails += judge_z2d_mask_knobs(overlay or {}, z2_dose)
    if arm == "F1":
        tags = [t for t in F1_NONSTAND_TAGS if (emitted.get(t.rsplit("/", 1)[-1]))]
        if tags:
            notes.append(f"F1 per-slot telemetry emitted: {tags}")
        elif not stand_gate or not stand_gate[0]:
            fails.append(f"F1 stand-only construction not shown ({stand_gate[1] if stand_gate else 'not read'})")
    return {"rc": RC_FAIL if fails else RC_PASS, "result": "FAIL" if fails else "PASS", "fails": fails, "notes": notes,
            "arm": arm}


def judge_terrain(ser: dict, first: int, num_rows) -> dict:
    """CHARTER 8 terrain at +1000 (pure), on obstacle-tile telemetry only."""
    last = max((s for v in ser.values() for s, _ in v), default=None)
    lo, hi = first + W1000[0], first + W1000[1]
    if last is None or last < hi:
        return {"rc": RC_WAIT, "why": f"telemetry only to it {last}"}
    tag = next((t for t in OBST_LEVEL_TAGS if ser.get(t)), None)
    if tag is None:
        if last >= hi + GRACE:
            return {"rc": RC_FAIL, "result": "FAIL",
                    "fails": [f"no obstacle-tile terrain-level telemetry ({list(OBST_LEVEL_TAGS)}) by it {last}"]}
        return {"rc": RC_WAIT, "why": "no obstacle-tile terrain-level tag yet"}
    lv = ser[tag]
    start_vals = [v for _, v in sorted(lv)[:START_N]]
    start = sum(start_vals) / len(start_vals)
    at = window(lv, lo, hi)
    fails = []
    if at is None:
        return {"rc": RC_WAIT, "why": f"{tag} has no point in {lo}-{hi}"}
    risen = at > start
    if not risen:
        fails.append(f"obstacle-tile mean level {at:.3f} has not risen above its start {start:.3f}")
    ptag = next((t for t in PROMO_TAGS if ser.get(t)), None)
    if ptag:
        pv = [v for s, v in ser[ptag] if first <= s <= hi]
        promos = sum(pv) > 0
        promo_src = ptag
    else:
        promos = risen
        promo_src = "implied by the rise (no promotion tag logged)"
    if not promos:
        fails.append(f"no promotions ({promo_src})")
    try:
        top = int(num_rows) - 1
    except (TypeError, ValueError):
        top = None
        fails.append(f"num_rows unreadable from params/env.yaml ({num_rows!r}): cannot test the top pin")
    if at <= PIN_EPS:
        fails.append(f"mean level {at:.3f} pinned at 0")
    if top is not None and at >= top - PIN_EPS:
        fails.append(f"mean level {at:.3f} pinned at the top level {top}")
    return {"rc": RC_FAIL if fails else RC_PASS, "result": "FAIL" if fails else "PASS", "fails": fails,
            "tag": tag, "start": start, "at_1000": at, "promotions": promo_src, "top": top}


def judge_collapse(ser: dict, first: int, source_tail_len) -> dict:
    """CHARTER 8 collapse (pure): after +300 only."""
    last = max((s for v in ser.values() for s, _ in v), default=None)
    if last is None or last < first + AT_COLLAPSE:
        return {"rc": RC_PASS, "result": "n/a (before +300)"}
    el = [(s, v) for s, v in ser.get(TAGS["ep_len"]) or [] if s >= first]
    cf = [(s, v) for s, v in ser.get(TAGS["crab_failure"]) or [] if s >= first]
    if not el or not cf or source_tail_len in (None, 0):
        return {"rc": RC_WAIT, "why": "episode length / crab_failure / source tail missing"}
    r_len, r_cf = tail_mean(el), tail_mean(cf)
    fails = []
    if r_len < COLLAPSE_LEN_FRAC * source_tail_len:
        fails.append(f"rolling-200 episode length {r_len:.1f} < 0.5 x source tail {source_tail_len:.1f}")
    if r_cf > COLLAPSE_FAIL:
        fails.append(f"rolling-200 crab_failure {r_cf:.3f} > {COLLAPSE_FAIL}")
    return {"rc": RC_FAIL if fails else RC_PASS, "result": "FAIL" if fails else "PASS", "fails": fails,
            "ep_len_roll": r_len, "crab_failure_roll": r_cf, "source_tail": source_tail_len}


def judge_p2s_guards(ser: dict, first: int, ref: dict | None, *, ep_len_notify: bool = False) -> dict:
    """P2S_PLAN step 6, the three p2s guards (pure), at the run's newest logged iteration ``last`` (fork-relative
    offset ``d = last - first``), each a rolling window ending at ``last`` against the reference's SAME fork-relative
    window (``ref["kind"] == "run"``: its own series from its own first logged iteration) or its constant source tail
    (``"tail"``):
      * KILL (rc 3) if rolling-200 Train/mean_episode_length < 0.9 x the reference's (from +300);
      * KILL (rc 3) if rolling-200 PPO/effective_std < 0.8 x the reference's (from +1000);
      * NOTIFY-ONLY (rc 5, never a kill) if rolling-500 Episode_Termination/crab_failure > the reference's + 0.08
        (from +500 -- the rolling-500 window is undefined before it).
    A kill outranks the flag. A reference window that is not there yet is WAIT (rc 2) for that clause only.
    ``ep_len_notify`` (the Z2 arm; P2R_Z2_PLAN "Left for implementation"): the episode-length clause is a NOTIFY-ONLY
    flag instead of a kill -- R re-steps through most noisy training stands, so read the stand-slot terminations
    before acting on it."""
    last = max((s for v in ser.values() for s, _ in v), default=None)
    if last is None:
        return {"rc": RC_WAIT, "why": "no telemetry"}
    d = last - first
    if d < P2S_LEN_AT:
        return {"rc": RC_PASS, "result": f"n/a (before +{P2S_LEN_AT})", "d": d}
    if not ref or ref.get("error"):
        return {"rc": RC_WAIT, "why": f"p2s guard reference unreadable: {(ref or {}).get('error', 'none')}", "d": d}

    def arm_roll(tag, n):
        return window([(s, v) for s, v in ser.get(tag) or [] if s >= first], last - n + 1, last)

    def ref_roll(tag, n):
        if ref.get("kind") == "tail":
            return (ref.get("tail") or {}).get(tag)
        rf = ref.get("first")
        return None if rf is None else window((ref.get("ser") or {}).get(tag), rf + d - n + 1, rf + d)
    kills, flags, waits, read = [], [], [], {"d": d, "reference": ref.get("name"), "reference_kind": ref.get("kind")}
    el, rel = arm_roll(TAGS["ep_len"], P2S_LEN_ROLL), ref_roll(TAGS["ep_len"], P2S_LEN_ROLL)
    read.update(ep_len=el, ref_ep_len=rel)
    if el is None or rel is None:
        waits.append("episode length (arm or reference window missing)")
    elif el < (frac := P2S_LEN_FRAC_TAIL if ref.get("kind") == "tail" else P2S_LEN_FRAC) * rel:
        msg = (f"termination: rolling-{P2S_LEN_ROLL} episode length {el:.1f} < {frac} x the reference's "
               f"{rel:.1f} (+{d})")
        if ep_len_notify:
            flags.append(f"{msg} (notify-only for Z2: read the stand-slot terminations before acting)")
        else:
            kills.append(msg)
    if d >= P2S_STD_AT:
        sd, rsd = arm_roll(TAGS["std"], P2S_STD_ROLL), ref_roll(TAGS["std"], P2S_STD_ROLL)
        read.update(std=sd, ref_std=rsd)
        if sd is None or rsd is None:
            waits.append("effective_std (arm or reference window missing)")
        elif sd < P2S_STD_FRAC * rsd:
            kills.append(f"sigma: rolling-{P2S_STD_ROLL} PPO/effective_std {sd:.4f} < {P2S_STD_FRAC} x the reference's "
                         f"{rsd:.4f} (+{d})")
    if d >= P2S_CF_AT:
        cf, rcf = arm_roll(TAGS["crab_failure"], P2S_CF_ROLL), ref_roll(TAGS["crab_failure"], P2S_CF_ROLL)
        read.update(crab_failure=cf, ref_crab_failure=rcf)
        if cf is None or rcf is None:
            waits.append("crab_failure (arm or reference window missing)")
        elif cf > rcf + P2S_CF_MARGIN:
            flags.append(f"termination flag (notify-only): rolling-{P2S_CF_ROLL} crab_failure {cf:.3f} > the "
                         f"reference's {rcf:.3f} + {P2S_CF_MARGIN} (+{d})")
    if kills:
        return dict(read, rc=RC_FAIL, result="FAIL", fails=kills, flags=flags, waits=waits)
    if flags:
        return dict(read, rc=RC_NOTIFY, result="NOTIFY", fails=[], flags=flags, waits=waits)
    if waits:
        return dict(read, rc=RC_WAIT, why="; ".join(waits), fails=[], flags=[])
    return dict(read, rc=RC_PASS, result="PASS", fails=[], flags=[])


def judge_smoke(audit: list, facts: dict, ser: dict, overlay: dict, require_move: bool) -> dict:
    """CHARTER 5 calibration-smoke dump audit (pure): every overlay key realised (PROMOTE included), no dropped PPO
    key; with ``require_move`` the obstacle-tile terrain levels must move (max != min over the run)."""
    fails = [f"overlay audit: {n}" for n, ok in audit if not ok]
    if not audit:
        fails.append("overlay audit produced no checks")
    if not any("KRABBY_TERRAIN_PROMOTE" in n for n, _ in audit):
        fails.append("KRABBY_TERRAIN_PROMOTE not audited")
    if facts.get("dropped_ppo_keys"):
        fails.append("the runner dropped PPO keys")
    moved = None
    if require_move:
        tag = next((t for t in OBST_LEVEL_TAGS if ser.get(t)), None)
        if tag is None:
            fails.append("no obstacle-tile terrain-level telemetry: cannot show the levels move")
        else:
            vals = [v for _, v in ser[tag]]
            moved = max(vals) - min(vals) > 1e-6
            if not moved:
                fails.append(f"{tag} never moved ({vals[0]:.3f})")
    return {"rc": RC_FAIL if fails else RC_PASS, "result": "FAIL" if fails else "PASS", "fails": fails,
            "promote": overlay.get("KRABBY_TERRAIN_PROMOTE"), "levels_moved": moved}


# ------------------------------------------------------------------------------------------------ IO drivers
def _json_file(path) -> dict | None:
    """A JSON file's object, or None when missing / unreadable (the Z2 dose file: None makes the check refuse)."""
    try:
        doc = json.load(open(path))
    except (OSError, ValueError, TypeError):
        return None
    return doc if isinstance(doc, dict) else None


def _gather(run_dir: Path, train_log):
    envp, agp = run_dir / "params" / "env.yaml", run_dir / "params" / "agent.yaml"
    env = _yaml_tolerant(envp) if envp.is_file() else None
    agent = _yaml_tolerant(agp) if agp.is_file() else None
    log = Path(train_log).read_text(errors="replace") if train_log and Path(train_log).is_file() else None
    return env, agent, log


def check(a) -> tuple[int, dict]:
    import yaml  # noqa: PLC0415

    cands = yaml.safe_load(open(a.candidates))
    if a.chain not in (cands.get("chains") or {}):
        return RC_WAIT, {"why": f"chain {a.chain} not in {a.candidates}"}
    # NOTE(p2r Z2, 2026-10-10): z_arm reads a fresh relaunch name (P2__Z2R1_s18 -> Z2); arm_of reads it as ''
    arm = p2s_arm(a.chain) or z_arm(a.chain) or arm_of(a.chain)
    ov = chain_overlay(cands, a.chain)
    ser = read_tb(a.check)
    first, last = first_last(ser)
    if first is None:
        return RC_WAIT, {"why": "no telemetry yet"}
    if a.at == "200":
        env, agent, log = _gather(a.check, a.train_log)
        if last < first + W200[1]:
            return RC_WAIT, {"why": f"telemetry only to it {last}"}
        rc_mod = load_rc()
        audit = rc_mod.overlay_checks({"ckpt": str(a.check / "model_audit.pt"), "log": a.train_log}, ov)
        replay = json.load(open(a.replay)) if a.replay and Path(a.replay).is_file() else None
        sg = stand_gate_structural(Path(a.code_root) / REWARDS_PY) if arm == "F1" else None
        seed = seed_of(a.chain)
        ref_tail = s1_p2a_tail(a.arms, seed) if arm == "TG" and seed is not None else None
        pg = p2s_gate_structural(Path(a.code_root) / MOTION_REWARDS_PY) if arm in P2S_ARMS else None
        z2d = (_json_file(getattr(a, "z2_dose", None) or z2_dose_default(arm)) if arm in P2R_Z2_FAMILY
               else None)
        res = judge_wiring(arm, ov, dump_facts(env, agent, log), ser, first, audit, replay, sg, ref_tail, pg, z2d)
    elif a.at == "p2s":
        # a p2s dose arm OR a stage-p2r Z arm (P2R_STATUE_PLAN.md step 1 "Left for implementation": Z arms share
        # the three guards, read against the same reference, but have no dose/term-liveness to read here).
        if arm in P2S_ARMS or z_arm(a.chain):
            res = judge_p2s_guards(ser, first, load_p2s_reference(p2s_reference(a.arms, seed_of(a.chain))),
                                   ep_len_notify=(arm in P2R_Z2_FAMILY))
        else:
            res = {"rc": RC_PASS, "result": f"n/a (chain {a.chain} is not a p2s or p2r Z arm)"}
    elif a.at == "1000":
        env, _, _ = _gather(a.check, None)
        res = judge_terrain(ser, first, _find_key((env or {}).get("scene") or env or {}, "num_rows"))
    else:
        src = a.source or (R / chain_source_ckpt(cands, a.chain)).parent
        src_ser = read_tb(Path(src))
        res = judge_collapse(ser, first, tail_mean(src_ser.get(TAGS["ep_len"])))
    res.update(chain=a.chain, at=a.at, first=first, last=last)
    return int(res["rc"]), res


def smoke_audit(a) -> tuple[int, dict]:
    import yaml  # noqa: PLC0415

    cands = yaml.safe_load(open(a.candidates))
    ov = chain_overlay(cands, a.chain)
    env, agent, log = _gather(a.smoke_audit, a.train_log)
    audit = load_rc().overlay_checks({"ckpt": str(a.smoke_audit / "model_audit.pt"), "log": a.train_log}, ov)
    res = judge_smoke(audit, dump_facts(env, agent, log), read_tb(a.smoke_audit), ov, a.require_terrain_move)
    res.update(chain=a.chain)
    return int(res["rc"]), res


def preflight(a) -> tuple[int, dict]:
    import yaml  # noqa: PLC0415

    cands = yaml.safe_load(open(a.candidates))
    if a.chain not in (cands.get("chains") or {}):
        return RC_FAIL, {"result": "FAIL", "fails": [f"chain {a.chain} not in {a.candidates}"]}
    arm = p2s_arm(a.chain) or z_arm(a.chain) or arm_of(a.chain)
    keys = [t[len(METRIC_PREFIX):] for t in OBST_LEVEL_TAGS + PROMO_TAGS] + [t.rsplit("/", 1)[-1] for t in F1_NONSTAND_TAGS]
    emitted = emitted_metric_keys(Path(a.code_root), keys)
    replay = json.load(open(a.replay)) if a.replay and Path(a.replay).is_file() else None
    sg = stand_gate_structural(Path(a.code_root) / REWARDS_PY) if arm == "F1" else None
    pg = pref = p2r_graft = fork_ckpt = None
    if arm in P2S_ARMS:
        pg = p2s_gate_structural(Path(a.code_root) / MOTION_REWARDS_PY)
        pref = p2s_reference_ok(load_p2s_reference(p2s_reference(a.arms, seed_of(a.chain))))
    if Z_ARM_RE.match(arm):
        fork_ckpt = chain_source_ckpt(cands, a.chain)
        if fork_ckpt:
            gj_path = (R / fork_ckpt).with_suffix(".json")
            if gj_path.is_file():
                try:
                    p2r_graft = json.load(open(gj_path))
                except (OSError, ValueError):
                    p2r_graft = None
    res = judge_preflight(arm, emitted, replay, sg, chain_overlay(cands, a.chain), pg, pref,
                          p2r_graft=p2r_graft, fork_ckpt=fork_ckpt,
                          z2_dose=(_json_file(getattr(a, "z2_dose", None) or z2_dose_default(arm))
                                   if arm in P2R_Z2_FAMILY else None))
    res.update(chain=a.chain, emitted={k: v for k, v in emitted.items() if v})
    return int(res["rc"]), res


# ------------------------------------------------------------------------------------------------ code guard (M1)
CODE_STATE_KEYS = ("code_sha", "eval_sha", "dirty")


def anchor_reference_sidecar(lrout: Path, name: str) -> tuple[str, dict | None, str | None]:
    """The code-guard reference: ``(state, sidecar, stamp)`` of the newest COMPLETE ``<lrout>/<name>/suite__omni/
    seed001/<ts>`` stamp. ``state`` is ``none`` (no anchor stamp yet), ``missing`` (a stamp with no readable
    code_state.json -- fails closed) or ``ok``."""
    d = Path(lrout) / name / ANCHOR_REF_ROW / f"seed{ANCHOR_REF_SEED:03d}"
    hits = sorted(p for p in d.glob("*/") if (p / "complete.json").is_file()) if d.is_dir() else []
    if not hits:
        return "none", None, None
    st = hits[-1]
    try:
        data = json.load(open(st / "code_state.json"))
    except (OSError, ValueError):
        return "missing", None, str(st)
    return ("ok", data, str(st)) if isinstance(data, dict) else ("missing", None, str(st))


def judge_code_guard(live: dict, ref_state: str, ref: dict | None, reanchor: bool = False) -> dict:
    """Pre-launch audit M1 (pure). ``live`` = {code_sha, eval_sha, dirty, dirty_paths} of the code root now; ``ref`` =
    the s18 anchor's sidecar. Refuses a dirty tree always; with an anchor, any difference in code_sha / eval_sha /
    dirty; an anchor stamp without a sidecar (never "equal")."""
    fails, notes = [], []
    if live.get("dirty") or live.get("dirty") is None:
        fails.append(f"code/eval pathspecs dirty (or unreadable): {live.get('dirty_paths')}")
    if reanchor:
        notes.append("re-anchor: the anchor comparison is skipped (dirty = false still required)")
    elif ref_state == "none":
        notes.append("no anchor stamp yet: dirty = false required, nothing to compare")
    elif ref_state == "missing" or not ref:
        fails.append("the s18 anchor stamp has no code_state.json sidecar (fail closed)")
    else:
        for k in CODE_STATE_KEYS:
            a, b = live.get(k), ref.get(k)
            if k == "dirty":
                a, b = bool(a), bool(b)
            if a != b:
                fails.append(f"{k}: live {a!r} != anchor {b!r}")
    return {"rc": RC_REFUSE if fails else RC_PASS, "result": "REFUSE" if fails else "PASS", "fails": fails,
            "notes": notes, "ref_state": ref_state, "live": {k: live.get(k) for k in CODE_STATE_KEYS},
            "anchor": {k: (ref or {}).get(k) for k in CODE_STATE_KEYS} if ref else None}


def live_code_state(code_root: Path) -> dict:
    rc = load_rc(Path(code_root) / RC_PATH.relative_to(R))
    dirty = rc.dirty_paths(Path(code_root))
    return {"code_sha": rc.code_sha(Path(code_root)), "eval_sha": rc.eval_sha(Path(code_root)), "dirty": bool(dirty),
            "dirty_paths": dirty}


def _anchor_name(seed: int = 18) -> str:
    import yaml  # noqa: PLC0415

    try:
        return str(yaml.safe_load(open(CAMPAIGN / "anchors.yaml"))["anchors"][seed]["name"])
    except (OSError, KeyError, TypeError, ValueError) as e:
        # fail closed (review of the CHARTER 9 walls re-tune): a guessed name could resolve to the ramp-era ANC_s*
        # stamps, which carry today's code state and would PASS against a stale reference
        raise RuntimeError(f"anchors.yaml unreadable, no code-guard reference: {e.__class__.__name__}: {e}") from e


def code_guard(a) -> tuple[int, dict]:
    lrout = Path(a.lrout) if a.lrout else Path(os.environ.get("P2_OUT") or OUT_DEFAULT) / "lr"
    name = a.anchor_name or _anchor_name()
    try:
        live = live_code_state(Path(a.code_root))
    except Exception as e:  # noqa: BLE001 -- an unreadable code state refuses, never passes
        live = {"dirty": None, "dirty_paths": f"code state unreadable: {e.__class__.__name__}: {e}"}
    state, ref, stamp = anchor_reference_sidecar(lrout, name)
    res = judge_code_guard(live, state, ref, a.reanchor)
    res.update(anchor_name=name, anchor_stamp=stamp, code_root=str(a.code_root))
    return int(res["rc"]), res


# ------------------------------------------------------------------------------------------------ self-test
def _ser(first=14996, n=1100, **const):
    return {tag: [(first + i, v) for i in range(n)] for tag, v in const.items()}


def _facts(**over):
    f = {"have_env": True, "curriculum_null": True,
         "weights": dict(ELEMENT_WEIGHTS, reward_clock_schedule=0.001, reward_clock_swing_apex=0.001,
                         reward_action_rate=0.0, reward_stand_stance_quality=None, penalty_mechanical_power=0.0),
         "term_funcs": {"reward_stand_stance_quality": F1_FUNC},
         "dr_events": {e: True for e in DR_EVENTS}, "push_x": [-0.5, 0.5], "terrain_fns": ["omni_field_terrain",
                                                                                          "omni_flat_terrain"],
         "num_rows": 10, "dropped_ppo_keys": []}
    f.update(over)
    return f


def _self_test() -> int:
    ok = True

    def check_(name, got, want):
        nonlocal ok
        good = got == want
        ok &= good
        print(f"  [{'ok' if good else 'FAIL'}] {name}: {got}")

    F0 = 14996
    ov_a1 = {"KRABBY_FLAT_TERRAIN_GEOM": "field", "KRABBY_ACTION_RATE_W": "0"}
    base = {TAGS["reach_obst"]: 0.3, TAGS["clock_hidden"]: 1.0}
    audit_ok = [("overlay KRABBY_TERRAIN_PROMOTE=0.75:0.4 in params/env.yaml", True)]
    s = _ser(n=215, **base)
    check_("wiring: a clean A1 passes", judge_wiring("A1", ov_a1, _facts(), s, F0, audit_ok, None)["rc"], RC_PASS)
    check_("wiring: telemetry short of +210 waits", judge_wiring("A1", ov_a1, _facts(), _ser(n=150, **base), F0,
                                                                 audit_ok, None)["rc"], RC_WAIT)
    check_("wiring: counted from the fork start (14996+210), not from 0",
           judge_wiring("A1", ov_a1, _facts(), _ser(first=F0, n=205, **base), F0, audit_ok, None)["rc"], RC_WAIT)
    check_("wiring: curriculum not null fails", judge_wiring("A1", ov_a1, _facts(curriculum_null=False), s, F0,
                                                             audit_ok, None)["rc"], RC_FAIL)
    check_("wiring: a failed overlay-audit key fails", judge_wiring("A1", ov_a1, _facts(), s, F0,
                                                                    audit_ok + [("overlay X", False)], None)["rc"], RC_FAIL)
    check_("wiring: an empty audit fails closed", judge_wiring("A1", ov_a1, _facts(), s, F0, [], None)["rc"], RC_FAIL)
    check_("wiring: reach_obst 0 fails", judge_wiring("A1", ov_a1, _facts(), _ser(n=215, **dict(base, **{
        TAGS["reach_obst"]: 0.0})), F0, audit_ok, None)["rc"], RC_FAIL)
    check_("wiring: ring sub-terrains under a field arm fail",
           judge_wiring("A1", ov_a1, _facts(terrain_fns=["omni_ring_terrain"]), s, F0, audit_ok, None)["rc"], RC_FAIL)
    check_("wiring: A0 ring passes with ring sub-terrains",
           judge_wiring("A0", {"KRABBY_FLAT_TERRAIN_GEOM": "ring", "KRABBY_ACTION_RATE_W": "0"},
                        _facts(terrain_fns=["omni_ring_terrain"]), s, F0, audit_ok, None)["rc"], RC_PASS)
    check_("wiring: a dropped PPO key fails", judge_wiring("A1", ov_a1, _facts(dropped_ppo_keys=["Ignoring unsupported "
                                                           "PPO config keys: kl_stop"]), s, F0, audit_ok, None)["rc"], RC_FAIL)
    _w = "\x1b[33m18:12:10 [on_policy_runner_with_extractor.py] WARNING: Ignoring unsupported PPO config keys for PPOWithExtractor: "
    check_("dropped keys: the allow-listed share_cnn_encoders alone is not a drop",
           len(dropped_ppo_keys(_w + "share_cnn_encoders\x1b[0m")), 0)
    check_("dropped keys: any other key is a drop", len(dropped_ppo_keys(_w + "share_cnn_encoders, kl_stop\x1b[0m")), 1)
    check_("wiring: clock 1.0 (an un-blanked preset) fails", judge_wiring("A1", ov_a1, _facts(weights=dict(
        _facts()["weights"], reward_clock_schedule=1.0)), s, F0, audit_ok, None)["rc"], RC_FAIL)
    check_("wiring: element weight off (airtime 0.4) fails", judge_wiring("A1", ov_a1, _facts(weights=dict(
        _facts()["weights"], reward_feet_air_time_positive=0.4)), s, F0, audit_ok, None)["rc"], RC_FAIL)
    check_("wiring: a missing DR event fails", judge_wiring("A1", ov_a1, _facts(dr_events={
        "randomize_rigid_body_mass": True, "randomize_rigid_body_com": False, "push_by_setting_velocity": True}),
        s, F0, audit_ok, None)["rc"], RC_FAIL)
    ov_s1 = dict(ov_a1, KRABBY_ACTION_RATE_W="-0.05")
    w_s1 = dict(_facts()["weights"], reward_action_rate=-0.05)
    s_s1 = _ser(n=215, **dict(base, **{"Episode_Reward/reward_action_rate": -0.004}))
    check_("wiring: S1 live action_rate term passes", judge_wiring("S1", ov_s1, _facts(weights=w_s1), s_s1, F0,
                                                                   audit_ok, None)["rc"], RC_PASS)
    check_("wiring: S1 with a dead action_rate term fails", judge_wiring("S1", ov_s1, _facts(weights=w_s1),
                                                                         _ser(n=215, **dict(base, **{"Episode_Reward/reward_action_rate": 0.0})),
                                                                         F0, audit_ok, None)["rc"], RC_FAIL)
    s_e1 = _ser(n=215, **dict(base, **{"Episode_Reward/penalty_mechanical_power": -0.03}))
    check_("wiring: E1 within 3x of the replay passes", judge_wiring("E1", ov_a1, _facts(), s_e1, F0, audit_ok,
                                                                     {"E1": {"episode_reward": -0.02}})["rc"], RC_PASS)
    check_("wiring: E1 at 3.5x the replay fails", judge_wiring("E1", ov_a1, _facts(), s_e1, F0, audit_ok,
                                                               {"E1": {"episode_reward": -0.03 / 3.5}})["rc"], RC_FAIL)
    check_("wiring: E1 at exactly 3x passes (boundary)", judge_wiring("E1", ov_a1, _facts(), s_e1, F0, audit_ok,
                                                                      {"E1": {"episode_reward": -0.01}})["rc"], RC_PASS)
    check_("wiring: E1 of the wrong sign fails", judge_wiring("E1", ov_a1, _facts(), s_e1, F0, audit_ok,
                                                              {"E1": {"episode_reward": 0.03}})["rc"], RC_FAIL)
    check_("wiring: E1 with no replay record fails closed", judge_wiring("E1", ov_a1, _facts(), s_e1, F0, audit_ok,
                                                                         None)["rc"], RC_FAIL)
    s_f1 = _ser(n=215, **dict(base, **{"Episode_Reward/reward_stand_stance_quality": 0.05}))
    gate_ok, gate_bad = (True, "stand_mask() factor"), (False, "no stand_mask() factor")
    check_("wiring: F1 without per-slot telemetry or a construction read fails closed",
           judge_wiring("F1", ov_a1, _facts(), s_f1, F0, audit_ok, {"F1": {"episode_reward": 0.04}})["rc"], RC_FAIL)
    check_("wiring: F1 stand-only by construction (no per-slot telemetry) passes",
           judge_wiring("F1", ov_a1, _facts(), s_f1, F0, audit_ok, {"F1": {"episode_reward": 0.04}}, gate_ok)["rc"],
           RC_PASS)
    check_("wiring: F1 with an ungated term source fails",
           judge_wiring("F1", ov_a1, _facts(), s_f1, F0, audit_ok, {"F1": {"episode_reward": 0.04}}, gate_bad)["rc"],
           RC_FAIL)
    check_("wiring: F1 whose dumped func is another function fails",
           judge_wiring("F1", ov_a1, _facts(term_funcs={"reward_stand_stance_quality": "x:other"}), s_f1, F0, audit_ok,
                        {"F1": {"episode_reward": 0.04}}, gate_ok)["rc"], RC_FAIL)
    check_("wiring: F1 under the old term name reads a dead term and fails",
           judge_wiring("F1", ov_a1, _facts(), _ser(n=215, **dict(base, **{"Episode_Reward/reward_stand_stance": 0.05})),
                        F0, audit_ok, {"F1": {"episode_reward": 0.04}}, gate_ok)["rc"], RC_FAIL)
    s_f1b = _ser(n=215, **dict(base, **{"Episode_Reward/reward_stand_stance_quality": 0.05, F1_NONSTAND_TAGS[0]: 0.0}))
    check_("wiring: F1 paying only at stand passes", judge_wiring("F1", ov_a1, _facts(), s_f1b, F0, audit_ok,
                                                                  {"F1": {"episode_reward": 0.04}})["rc"], RC_PASS)
    s_f1c = _ser(n=215, **dict(base, **{"Episode_Reward/reward_stand_stance_quality": 0.05, F1_NONSTAND_TAGS[0]: 0.01}))
    check_("wiring: F1 paying outside the stand slot fails", judge_wiring("F1", ov_a1, _facts(), s_f1c, F0, audit_ok,
                                                                          {"F1": {"episode_reward": 0.04}})["rc"], RC_FAIL)

    # ------------------------------------------------------------------------------------------- stage p2b (CHARTER 10)
    cl_replay = {"p2b": {"CL": {"terms": {"reward_foot_clearance": {"episode_reward": 0.02},
                                          "penalty_swing_min_clearance": {"episode_reward": -0.01}}}}}
    s_cl = _ser(n=215, **dict(base, **{"Episode_Reward/reward_foot_clearance": 0.025,
                                       "Episode_Reward/penalty_swing_min_clearance": -0.012}))
    check_("wiring: CL both terms live and within 3x passes",
           judge_wiring("CL", ov_a1, _facts(), s_cl, F0, audit_ok, cl_replay)["rc"], RC_PASS)
    check_("wiring: CL with a dead clearance term fails",
           judge_wiring("CL", ov_a1, _facts(), _ser(n=215, **dict(base, **{"Episode_Reward/reward_foot_clearance": 0.0,
                        "Episode_Reward/penalty_swing_min_clearance": -0.012})), F0, audit_ok, cl_replay)["rc"], RC_FAIL)
    check_("wiring: CL with the clearance reward negative (wrong sign) fails",
           judge_wiring("CL", ov_a1, _facts(), _ser(n=215, **dict(base, **{"Episode_Reward/reward_foot_clearance": -0.02,
                        "Episode_Reward/penalty_swing_min_clearance": -0.012})), F0, audit_ok, cl_replay)["rc"], RC_FAIL)
    check_("wiring: CL with the swing penalty positive (wrong sign) fails",
           judge_wiring("CL", ov_a1, _facts(), _ser(n=215, **dict(base, **{"Episode_Reward/reward_foot_clearance": 0.02,
                        "Episode_Reward/penalty_swing_min_clearance": 0.012})), F0, audit_ok, cl_replay)["rc"], RC_FAIL)
    check_("wiring: CL at 3.5x the replay fails",
           judge_wiring("CL", ov_a1, _facts(), s_cl, F0, audit_ok,
                        {"p2b": {"CL": {"terms": {"reward_foot_clearance": {"episode_reward": 0.025 / 3.5},
                                                  "penalty_swing_min_clearance": {"episode_reward": -0.012}}}}})["rc"],
           RC_FAIL)
    check_("wiring: CL with no p2b replay record fails closed",
           judge_wiring("CL", ov_a1, _facts(), s_cl, F0, audit_ok, None)["rc"], RC_FAIL)
    tg_replay = {"p2b": {"TG": {"terms": {"reward_orientation": {"episode_reward": -0.05}}}}}
    s_tg = _ser(n=215, **dict(base, **{"Episode_Reward/reward_orientation": -0.045}))
    check_("wiring: TG live, >= 2x the seed tail and within 3x of the replay passes",
           judge_wiring("TG", ov_a1, _facts(), s_tg, F0, audit_ok, tg_replay, ref_tail={"reward_orientation": -0.02})["rc"],
           RC_PASS)
    check_("wiring: TG under 2x the seed's S1 p2a tail fails",
           judge_wiring("TG", ov_a1, _facts(), s_tg, F0, audit_ok, tg_replay, ref_tail={"reward_orientation": -0.03})["rc"],
           RC_FAIL)
    check_("wiring: TG at exactly 2x the tail passes (boundary)",
           judge_wiring("TG", ov_a1, _facts(), s_tg, F0, audit_ok, tg_replay, ref_tail={"reward_orientation": -0.0225})["rc"],
           RC_PASS)
    check_("wiring: TG with no seed tail fails closed",
           judge_wiring("TG", ov_a1, _facts(), s_tg, F0, audit_ok, tg_replay, ref_tail=None)["rc"], RC_FAIL)
    check_("wiring: TG dead term fails", judge_wiring("TG", ov_a1, _facts(), _ser(n=215, **dict(
        base, **{"Episode_Reward/reward_orientation": 0.0})), F0, audit_ok, tg_replay,
        ref_tail={"reward_orientation": -0.02})["rc"], RC_FAIL)
    check_("wiring: C0 has no arm-specific term (no fails from it)",
           judge_wiring("C0", ov_a1, _facts(), s, F0, audit_ok, None)["rc"], RC_PASS)
    check_("p2b_checkpoints: a model_19994 fork", p2b_checkpoints(19994), [21000, 22500, 24000])
    check_("seed_of", [seed_of("P2__CL_s16"), seed_of("P2__TG_s17"), seed_of("ANCW_s18")], [16, 17, 18])

    def lv(start, end, n=1100):
        return [(F0 + i, start + (end - start) * i / (n - 1)) for i in range(n)]
    T = OBST_LEVEL_TAGS[0]
    check_("terrain: a rising, unpinned obstacle level passes", judge_terrain({T: lv(1.0, 2.5)}, F0, 10)["rc"], RC_PASS)
    check_("terrain: before +1050 waits", judge_terrain({T: lv(1.0, 2.5, n=900)}, F0, 10)["rc"], RC_WAIT)
    check_("terrain: a flat level fails (not risen, no promotions)", judge_terrain({T: lv(1.0, 1.0)}, F0, 10)["rc"], RC_FAIL)
    check_("terrain: pinned at 0 fails", judge_terrain({T: lv(0.0, 0.04)}, F0, 10)["rc"], RC_FAIL)
    check_("terrain: pinned at the top (9) fails", judge_terrain({T: lv(8.9, 9.0)}, F0, 10)["rc"], RC_FAIL)
    check_("terrain: a promotion tag of zeros fails", judge_terrain({T: lv(1.0, 2.0), PROMO_TAGS[0]: [(F0 + i, 0.0)
                                                                                                    for i in range(1100)]}, F0, 10)["rc"], RC_FAIL)
    allt = {TAGS["levels_all"]: lv(1.0, 2.0)}
    check_("terrain: all-tile levels only (no obstacle tag) waits inside the grace", judge_terrain(allt, F0, 10)["rc"], RC_WAIT)
    check_("terrain: ... and fails closed past it", judge_terrain({TAGS["levels_all"]: lv(1.0, 2.0, n=1300)}, F0, 10)["rc"],
           RC_FAIL)
    EL, CF = TAGS["ep_len"], TAGS["crab_failure"]
    check_("collapse: before +300 is n/a", judge_collapse(_ser(n=250, **{EL: 100.0, CF: 0.9}), F0, 1800.0)["rc"], RC_PASS)
    check_("collapse: a healthy run passes", judge_collapse(_ser(n=400, **{EL: 1500.0, CF: 0.2}), F0, 1800.0)["rc"], RC_PASS)
    check_("collapse: episode length < 0.5 x source fails", judge_collapse(_ser(n=400, **{EL: 880.0, CF: 0.2}), F0,
                                                                           1800.0)["rc"], RC_FAIL)
    check_("collapse: episode length = 0.5 x source passes (boundary)",
           judge_collapse(_ser(n=400, **{EL: 900.0, CF: 0.2}), F0, 1800.0)["rc"], RC_PASS)
    check_("collapse: crab_failure 0.61 fails", judge_collapse(_ser(n=400, **{EL: 1500.0, CF: 0.61}), F0, 1800.0)["rc"],
           RC_FAIL)
    check_("collapse: no source tail waits", judge_collapse(_ser(n=400, **{EL: 1500.0, CF: 0.2}), F0, None)["rc"], RC_WAIT)
    check_("smoke: a full audit with PROMOTE realised and moving levels passes",
           judge_smoke(audit_ok, _facts(), {T: lv(1.0, 1.5)}, {"KRABBY_TERRAIN_PROMOTE": "0.75:0.4"}, True)["rc"], RC_PASS)
    check_("smoke: levels that never move fail", judge_smoke(audit_ok, _facts(), {T: lv(1.0, 1.0)}, {}, True)["rc"], RC_FAIL)
    check_("smoke: PROMOTE not audited fails", judge_smoke([("overlay KRABBY_OMNI_FACING=0.2:1.2", True)], _facts(), {}, {},
                                                           False)["rc"], RC_FAIL)
    check_("arm_of", [arm_of("P2__A1_s18"), arm_of("P2__CAL2_s18"), arm_of("P2__E1_s18"), arm_of("P2__CL_s16"),
                      arm_of("P2__TG_s17"), arm_of("P2__C0_s18")], ["A1", "CAL2", "E1", "CL", "TG", "C0"])
    em_ok = {"terrain_levels_obst": ["x.py"], "promote_count": ["x.py"]}
    check_("preflight: A1 with both obstacle metrics emitted passes", judge_preflight("A1", em_ok, None, None)["rc"],
           RC_PASS)
    check_("preflight: A1 without the obstacle-level metric is refused",
           judge_preflight("A1", {"promote_count": ["x.py"]}, None, None)["rc"], RC_FAIL)
    check_("preflight: CAL2 without the promotion metric is refused",
           judge_preflight("CAL2", {"terrain_levels_obst": ["x.py"]}, None, None)["rc"], RC_FAIL)
    check_("preflight: CAL1 needs no obstacle telemetry", judge_preflight("CAL1", {}, None, None)["rc"], RC_PASS)
    check_("preflight: E1 without a replay magnitude is refused", judge_preflight("E1", em_ok, None, None)["rc"], RC_FAIL)
    check_("preflight: E1 with a replay magnitude passes",
           judge_preflight("E1", em_ok, {"E1": {"episode_reward": -0.02, "verdict": "PASS"}}, None)["rc"], RC_PASS)
    check_("preflight E1: a FAIL verdict without a user overrule refuses",
           judge_preflight("E1", em_ok, {"E1": {"episode_reward": -0.02, "verdict": "FAIL"}}, None)["rc"], RC_FAIL)
    check_("preflight E1: a FAIL verdict with a recorded user overrule passes",
           judge_preflight("E1", em_ok, {"E1": {"episode_reward": -0.02, "verdict": "FAIL",
                                                "overrule": {"by": "user", "date": "2026-10-06"}}}, None)["rc"], RC_PASS)
    check_("preflight: F1 with replay + construction passes",
           judge_preflight("F1", em_ok, {"F1": {"episode_reward": 0.04, "verdict": "PASS"}}, gate_ok)["rc"], RC_PASS)
    check_("preflight: F1 with an ungated term is refused",
           judge_preflight("F1", em_ok, {"F1": {"episode_reward": 0.04}}, gate_bad)["rc"], RC_FAIL)
    check_("preflight: CL without a p2b replay magnitude is refused", judge_preflight("CL", em_ok, None, None)["rc"],
           RC_FAIL)
    check_("preflight: CL with both p2b terms + a PASS verdict passes",
           judge_preflight("CL", em_ok, {"p2b": {"CL": {"verdict": "PASS", "terms": {
               "reward_foot_clearance": {"episode_reward": 0.02},
               "penalty_swing_min_clearance": {"episode_reward": -0.01}}}}}, None)["rc"], RC_PASS)
    check_("preflight: CL missing one p2b term is refused",
           judge_preflight("CL", em_ok, {"p2b": {"CL": {"verdict": "PASS", "terms": {
               "reward_foot_clearance": {"episode_reward": 0.02}}}}}, None)["rc"], RC_FAIL)
    check_("preflight: CL verdict FAIL without a user overrule is refused",
           judge_preflight("CL", em_ok, {"p2b": {"CL": {"verdict": "FAIL", "terms": {
               "reward_foot_clearance": {"episode_reward": 0.02},
               "penalty_swing_min_clearance": {"episode_reward": -0.01}}}}}, None)["rc"], RC_FAIL)
    check_("preflight: CL verdict FAIL with a recorded user overrule passes",
           judge_preflight("CL", em_ok, {"p2b": {"CL": {"verdict": "FAIL", "overrule": {"by": "user"}, "terms": {
               "reward_foot_clearance": {"episode_reward": 0.02},
               "penalty_swing_min_clearance": {"episode_reward": -0.01}}}}}, None)["rc"], RC_PASS)
    check_("preflight: TG with its p2b term + a PASS verdict passes",
           judge_preflight("TG", em_ok, {"p2b": {"TG": {"verdict": "PASS", "terms": {
               "reward_orientation": {"episode_reward": -0.05}}}}}, None)["rc"], RC_PASS)
    check_("preflight: TG without a p2b replay magnitude is refused", judge_preflight("TG", em_ok, None, None)["rc"],
           RC_FAIL)
    check_("preflight: C0 needs no p2b replay magnitude", judge_preflight("C0", em_ok, None, None)["rc"], RC_PASS)

    # ------------------------------------------------------------------------------------------- stage p2s (CHARTER 11)
    ov_st2 = dict(ov_s1, KRABBY_OMNI_SLOTS="0.15:0.55:0.1:0.2", KRABBY_STAND_SUPPORT_W="1.12",
                  KRABBY_STAND_SLIDE_W="-1.25")
    rp_st2 = {"p2s": {"verdict": "PASS", "ST2": {
        "knobs": {"KRABBY_STAND_SUPPORT_W": 1.12, "KRABBY_STAND_SLIDE_W": -1.25},
        "terms": {P2S_SUPPORT: {"episode_reward": 0.05, "episode_reward_sampled": 0.02},
                  P2S_SLIDE: {"episode_reward": -0.05, "episode_reward_sampled": -0.04}}}}}
    f_p2s = _facts(weights=w_s1, term_funcs={P2S_SUPPORT: f"{P2S_FUNC_MOD}:{P2S_SUPPORT}",
                                             P2S_SLIDE: f"{P2S_FUNC_MOD}:{P2S_SLIDE}"})
    pg_ok = {t: (True, "gated") for t in P2S_TERMS}

    def s_p2s(sup, sli):
        return _ser(n=215, **dict(base, **{"Episode_Reward/reward_action_rate": -0.004,
                                           f"Episode_Reward/{P2S_SUPPORT}": sup, f"Episode_Reward/{P2S_SLIDE}": sli}))
    # implied sum s_f = 3 + ER / (1.12 x 0.15): ER 0.168 -> 4.0; ER -0.084 -> 2.5 (boundary); ER -0.1 -> 2.405
    check_("p2s wiring: both terms live, slide within 3x, implied sum s_f 4.0 passes",
           judge_wiring("ST2", ov_st2, f_p2s, s_p2s(0.168, -0.05), F0, audit_ok, rp_st2, p2s_gate=pg_ok)["rc"], RC_PASS)
    check_("p2s wiring: implied sum s_f exactly 2.5 passes (boundary)",
           judge_wiring("ST2", ov_st2, f_p2s, s_p2s(-0.084, -0.05), F0, audit_ok, rp_st2, p2s_gate=pg_ok)["rc"], RC_PASS)
    check_("p2s wiring: implied sum s_f 2.405 (< 2.5: feet lifted) fails",
           judge_wiring("ST2", ov_st2, f_p2s, s_p2s(-0.1, -0.05), F0, audit_ok, rp_st2, p2s_gate=pg_ok)["rc"], RC_FAIL)
    check_("p2s wiring: a dead support term fails (sum s_f would read 3, inside the band)",
           judge_wiring("ST2", ov_st2, f_p2s, s_p2s(0.0, -0.05), F0, audit_ok, rp_st2, p2s_gate=pg_ok)["rc"], RC_FAIL)
    check_("p2s wiring: a dead slide term fails",
           judge_wiring("ST2", ov_st2, f_p2s, s_p2s(0.168, 0.0), F0, audit_ok, rp_st2, p2s_gate=pg_ok)["rc"], RC_FAIL)
    check_("p2s wiring: a positive slide penalty fails",
           judge_wiring("ST2", ov_st2, f_p2s, s_p2s(0.168, 0.05), F0, audit_ok, rp_st2, p2s_gate=pg_ok)["rc"], RC_FAIL)
    check_("p2s wiring: slide at 3.5x the sampled replay fails",
           judge_wiring("ST2", ov_st2, f_p2s, s_p2s(0.168, -0.14), F0, audit_ok, rp_st2, p2s_gate=pg_ok)["rc"], RC_FAIL)
    check_("p2s wiring: no sampled-probe magnitude (a deterministic one only) fails closed",
           judge_wiring("ST2", ov_st2, f_p2s, s_p2s(0.168, -0.05), F0, audit_ok,
                        {"p2s": {"verdict": "PASS", "ST2": {"terms": {P2S_SUPPORT: {"episode_reward": 0.05},
                                                                      P2S_SLIDE: {"episode_reward": -0.05}}}}},
                        p2s_gate=pg_ok)["rc"], RC_FAIL)
    rp_units = {"p2s": {"verdict": "PASS", "episode_reward_units": "weight x stand_share 0.15 x the SAMPLED REF mean",
                        "ST2": {"terms": {P2S_SUPPORT: {"episode_reward": 0.02, "episode_reward_det_ref": 0.05},
                                          P2S_SLIDE: {"episode_reward": -0.04, "episode_reward_det_ref": -0.05}}}}}
    check_("p2s wiring: episode_reward declared sampled-calibrated (scripts/p2s_replay.py's record) is read",
           judge_wiring("ST2", ov_st2, f_p2s, s_p2s(0.168, -0.05), F0, audit_ok, rp_units, p2s_gate=pg_ok)["rc"], RC_PASS)
    check_("p2s wiring: ... and without that declaration fails closed",
           judge_wiring("ST2", ov_st2, f_p2s, s_p2s(0.168, -0.05), F0, audit_ok,
                        {"p2s": {k: v for k, v in rp_units["p2s"].items() if k != "episode_reward_units"}},
                        p2s_gate=pg_ok)["rc"], RC_FAIL)
    check_("p2s wiring: a stand share other than the pinned 0.15 fails",
           judge_wiring("ST2", dict(ov_st2, KRABBY_OMNI_SLOTS="0.2:0.5:0.1:0.2"), f_p2s, s_p2s(0.168, -0.05), F0,
                        audit_ok, rp_st2, p2s_gate=pg_ok)["rc"], RC_FAIL)
    check_("p2s wiring: an ungated term source fails",
           judge_wiring("ST2", ov_st2, f_p2s, s_p2s(0.168, -0.05), F0, audit_ok, rp_st2,
                        p2s_gate=dict(pg_ok, **{P2S_SLIDE: (False, "ungated")}))["rc"], RC_FAIL)
    check_("p2s wiring: a dumped func from another module fails",
           judge_wiring("ST2", ov_st2, _facts(weights=w_s1, term_funcs={P2S_SUPPORT: "x:reward_stand_support",
                                                                       P2S_SLIDE: f"{P2S_FUNC_MOD}:{P2S_SLIDE}"}),
                        s_p2s(0.168, -0.05), F0, audit_ok, rp_st2, p2s_gate=pg_ok)["rc"], RC_FAIL)
    check_("p2s_arm: doses and fresh relaunch names", [p2s_arm("P2__ST2_s18"), p2s_arm("P2__ST1R2_s16"),
                                                       p2s_arm("P2__ST3_s18"), p2s_arm("P2__CL_s18")],
           ["ST2", "ST1", "", ""])

    # ------------------------------------------------------------------------------------------- stage p2r Z arms (CHARTER 12)
    check_("z_arm: doses and fresh relaunch names", [z_arm("P2__Z1_s18"), z_arm("P2__Z1R2_s16"),
                                                     z_arm("P2__ST2_s18"), z_arm("P2__CL_s18")],
           ["Z1", "Z1", "", ""])
    check_("arm_of reads a Z chain as Z1", arm_of("P2__Z1_s18"), "Z1")
    ov_z1 = dict(ov_s1, KRABBY_OMNI_STAND_OBS="clock", KRABBY_ANCHOR_W="1.0", KRABBY_STAND_PRIOR_W="1.0",
                KRABBY_ANCHOR_CKPT="parkour/outputs/omni_plan/phase2/graft/S1_s18_c15/model_19994.pt")

    def s_z1(anchor_rows=0.88, prior_rows=0.12, mb0_first=0.0, loss_anchor=0.01, loss_prior=2.3, n=215):
        extra = {P2R_ANCHOR_ROWS_TAG: anchor_rows, P2R_PRIOR_ROWS_TAG: prior_rows, P2R_LOSS_ANCHOR: loss_anchor,
                P2R_LOSS_PRIOR: loss_prior}
        s = _ser(n=n, **dict(base, **{"Episode_Reward/reward_action_rate": -0.004}, **extra))
        s[P2R_ANCHOR_MB0_TAG] = [(F0 + i, mb0_first if i == 0 else 0.05) for i in range(n)]
        return s
    check_("p2r wiring: a clean Z1 (fork values) passes",
           judge_wiring("Z1", ov_z1, _facts(weights=w_s1), s_z1(), F0, audit_ok, None)["rc"], RC_PASS)
    check_("p2r wiring: anchor_rows outside [0.8, 0.95] fails",
           judge_wiring("Z1", ov_z1, _facts(weights=w_s1), s_z1(anchor_rows=0.5), F0, audit_ok, None)["rc"], RC_FAIL)
    check_("p2r wiring: prior_rows outside [0.05, 0.2] fails",
           judge_wiring("Z1", ov_z1, _facts(weights=w_s1), s_z1(prior_rows=0.5), F0, audit_ok, None)["rc"], RC_FAIL)
    check_("p2r wiring: anchor_mb0 != 0 at the first logged iteration fails",
           judge_wiring("Z1", ov_z1, _facts(weights=w_s1), s_z1(mb0_first=0.3), F0, audit_ok, None)["rc"], RC_FAIL)
    check_("p2r wiring: a dead anchor loss fails",
           judge_wiring("Z1", ov_z1, _facts(weights=w_s1), s_z1(loss_anchor=0.0), F0, audit_ok, None)["rc"], RC_FAIL)
    check_("p2r wiring: realised dose (first-update stand_prior) 0.5 (< 1.2) fails",
           judge_wiring("Z1", ov_z1, _facts(weights=w_s1), s_z1(loss_prior=0.5), F0, audit_ok, None)["rc"], RC_FAIL)
    check_("p2r wiring: realised dose at exactly 1.2 passes (boundary)",
           judge_wiring("Z1", ov_z1, _facts(weights=w_s1), s_z1(loss_prior=1.2), F0, audit_ok, None)["rc"], RC_PASS)
    check_("p2r wiring: KRABBY_ANCHOR_W=0 fails",
           judge_wiring("Z1", dict(ov_z1, KRABBY_ANCHOR_W="0"), _facts(weights=w_s1), s_z1(), F0, audit_ok,
                        None)["rc"], RC_FAIL)
    check_("p2r wiring: no KRABBY_ANCHOR_CKPT fails",
           judge_wiring("Z1", {k: v for k, v in ov_z1.items() if k != "KRABBY_ANCHOR_CKPT"}, _facts(weights=w_s1),
                        s_z1(), F0, audit_ok, None)["rc"], RC_FAIL)
    graft_ok = {"identity_cpu": True, "identity_gpu_stamps": ["GRAFT_S1_s18/suite__omni/seed001/20260101"]}
    check_("p2r preflight: a readable graft + matching knobs passes",
           judge_preflight("Z1", em_ok, None, None, ov_z1, p2r_graft=graft_ok,
                           fork_ckpt=ov_z1["KRABBY_ANCHOR_CKPT"])["rc"], RC_PASS)
    check_("p2r preflight: no graft.json is refused",
           judge_preflight("Z1", em_ok, None, None, ov_z1, p2r_graft=None,
                           fork_ckpt=ov_z1["KRABBY_ANCHOR_CKPT"])["rc"], RC_FAIL)
    check_("p2r preflight: identity_cpu false is refused",
           judge_preflight("Z1", em_ok, None, None, ov_z1, p2r_graft=dict(graft_ok, identity_cpu=False),
                           fork_ckpt=ov_z1["KRABBY_ANCHOR_CKPT"])["rc"], RC_FAIL)
    check_("p2r preflight: empty identity_gpu_stamps is refused",
           judge_preflight("Z1", em_ok, None, None, ov_z1, p2r_graft=dict(graft_ok, identity_gpu_stamps=[]),
                           fork_ckpt=ov_z1["KRABBY_ANCHOR_CKPT"])["rc"], RC_FAIL)
    check_("p2r preflight: KRABBY_ANCHOR_CKPT not matching the chain's fork is refused",
           judge_preflight("Z1", em_ok, None, None, ov_z1, p2r_graft=graft_ok, fork_ckpt="some/other/head.pt")["rc"],
           RC_FAIL)
    check_("p2r preflight: no p2s replay-gate verdict needed (never checked for a Z arm)",
           judge_preflight("Z1", em_ok, None, None, ov_z1, p2r_graft=graft_ok,
                           fork_ckpt=ov_z1["KRABBY_ANCHOR_CKPT"])["rc"], RC_PASS)
    # NOTE(p2r Z2, 2026-10-10): the Z2 arm -- Z1's checks, the dose band from p2r_z2_dose.json, the R reads
    # (cams-only, 2026-10-10: no proxy knob, no held == live clause)
    ov_z2 = dict(ov_z1, KRABBY_STAND_PRIOR_TARGET="restance", KRABBY_RESTANCE="default")
    dose_z2 = {"band_lo": 3.0, "band_hi": 9.0, "rc": 0,
               "restance_knob": restance_mod().format_restance(restance_mod().RestanceParams())}

    def s_z2(loss_prior=5.0, nonzero=0.9, home=0.05, drop=None):
        s = s_z1(loss_prior=loss_prior)
        for t, v in ((P2R_RESTANCE_NONZERO_TAG, nonzero),
                     *((t, v) for t, v in zip(P2R_RESTANCE_BRANCH_TAGS, (0.6, 0.1, 0.1, home, 0.15)))):
            if t != drop:
                s[t] = [(F0 + i, v) for i in range(215)]
        return s
    check_("p2r Z2 wiring: a dose file computed for another R is refused",
           judge_wiring("Z2", ov_z2, _facts(weights=w_s1), s_z2(), F0, audit_ok, None,
                        z2_dose=dict(dose_z2, restance_knob="lift=-0.3"))["rc"], RC_FAIL)
    check_("p2r Z2 wiring: a clean Z2 passes",
           judge_wiring("Z2", ov_z2, _facts(weights=w_s1), s_z2(), F0, audit_ok, None, z2_dose=dose_z2)["rc"], RC_PASS)
    check_("p2r Z2 wiring: no dose file refuses (fails), never falls back to Z1's band",
           judge_wiring("Z2", ov_z2, _facts(weights=w_s1), s_z2(loss_prior=2.3), F0, audit_ok, None)["rc"], RC_FAIL)
    check_("p2r Z2 wiring: Z1's band value 2.3 outside the Z2 band fails",
           judge_wiring("Z2", ov_z2, _facts(weights=w_s1), s_z2(loss_prior=2.3), F0, audit_ok, None,
                        z2_dose=dose_z2)["rc"], RC_FAIL)
    check_("p2r Z2 wiring: restance_nonzero 0.4 fails",
           judge_wiring("Z2", ov_z2, _facts(weights=w_s1), s_z2(nonzero=0.4), F0, audit_ok, None,
                        z2_dose=dose_z2)["rc"], RC_FAIL)
    check_("p2r Z2 wiring: home share 0 fails",
           judge_wiring("Z2", ov_z2, _facts(weights=w_s1), s_z2(home=0.0), F0, audit_ok, None,
                        z2_dose=dose_z2)["rc"], RC_FAIL)
    check_("p2r Z2 wiring: a missing branch tag fails",
           judge_wiring("Z2", ov_z2, _facts(weights=w_s1), s_z2(drop="Loss/restance_wait"), F0, audit_ok, None,
                        z2_dose=dose_z2)["rc"], RC_FAIL)
    check_("p2r Z2 wiring: KRABBY_OMNI_STAND_PROXY=hold in the overlay fails (the cams-only chain runs it off)",
           judge_wiring("Z2", dict(ov_z2, KRABBY_OMNI_STAND_PROXY="hold"), _facts(weights=w_s1), s_z2(), F0,
                        audit_ok, None, z2_dose=dose_z2)["rc"], RC_FAIL)
    check_("p2r Z2 wiring: KRABBY_OMNI_STAND_PROXY=0 in the overlay passes (off)",
           judge_wiring("Z2", dict(ov_z2, KRABBY_OMNI_STAND_PROXY="0"), _facts(weights=w_s1), s_z2(), F0,
                        audit_ok, None, z2_dose=dose_z2)["rc"], RC_PASS)
    check_("p2r Z2 wiring: no Metrics/stand_proxy_live_frac logged still passes (the clause is dropped)",
           "Metrics/stand_proxy_live_frac" in s_z2(), False)
    check_("p2r Z2 wiring: a missing / unparseable KRABBY_RESTANCE fails",
           [judge_wiring("Z2", {k: v for k, v in ov_z2.items() if k != "KRABBY_RESTANCE"}, _facts(weights=w_s1),
                         s_z2(), F0, audit_ok, None, z2_dose=dose_z2)["rc"],
            judge_wiring("Z2", dict(ov_z2, KRABBY_RESTANCE="theta_L=0.6"), _facts(weights=w_s1), s_z2(), F0,
                         audit_ok, None, z2_dose=dose_z2)["rc"]], [RC_FAIL, RC_FAIL])
    check_("p2r Z2 wiring: a dose file written at rc 2 (no home branch) is refused",
           judge_wiring("Z2", ov_z2, _facts(weights=w_s1), s_z2(), F0, audit_ok, None,
                        z2_dose=dict(dose_z2, rc=2))["rc"], RC_FAIL)
    check_("p2r Z2 wiring: a dose file without proxy_live_min passes (the line is gone with the proxy)",
           judge_wiring("Z2", ov_z2, _facts(weights=w_s1), s_z2(), F0, audit_ok, None,
                        z2_dose=dose_z2)["rc"], RC_PASS)
    check_("p2r Z2 wiring: Z1 itself is unchanged by a Z2 dose file (Z1 band)",
           judge_wiring("Z1", ov_z1, _facts(weights=w_s1), s_z1(loss_prior=5.0), F0, audit_ok, None,
                        z2_dose=dose_z2)["rc"], RC_FAIL)
    check_("p2r Z2 preflight: knobs + dose file pass; no dose file refused",
           [judge_preflight("Z2", em_ok, None, None, ov_z2, p2r_graft=graft_ok, fork_ckpt=ov_z2["KRABBY_ANCHOR_CKPT"],
                            z2_dose=dose_z2)["rc"],
            judge_preflight("Z2", em_ok, None, None, ov_z2, p2r_graft=graft_ok, fork_ckpt=ov_z2["KRABBY_ANCHOR_CKPT"],
                            z2_dose=None)["rc"]], [RC_PASS, RC_FAIL])
    pref_ok = (True, "ref ok")
    check_("p2s preflight: PASS verdict + magnitudes + matching weights + gate + reference passes",
           judge_preflight("ST2", em_ok, rp_st2, None, ov_st2, pg_ok, pref_ok)["rc"], RC_PASS)
    check_("p2s preflight: no p2s replay record is refused",
           judge_preflight("ST2", em_ok, None, None, ov_st2, pg_ok, pref_ok)["rc"], RC_FAIL)
    check_("p2s preflight: a FAIL verdict without a user overrule is refused",
           judge_preflight("ST2", em_ok, {"p2s": dict(rp_st2["p2s"], verdict="FAIL")}, None, ov_st2, pg_ok,
                           pref_ok)["rc"], RC_FAIL)
    check_("p2s preflight: a FAIL verdict with a recorded user overrule passes",
           judge_preflight("ST2", em_ok, {"p2s": dict(rp_st2["p2s"], verdict="FAIL", overrule={"by": "user"})}, None,
                           ov_st2, pg_ok, pref_ok)["rc"], RC_PASS)
    check_("p2s preflight: chain weights other than the replay's solved dose are refused",
           judge_preflight("ST2", em_ok, rp_st2, None, dict(ov_st2, KRABBY_STAND_SUPPORT_W="1.0"), pg_ok,
                           pref_ok)["rc"], RC_FAIL)
    check_("p2s preflight: an unreadable guard reference is refused",
           judge_preflight("ST2", em_ok, rp_st2, None, ov_st2, pg_ok, (False, "no record"))["rc"], RC_FAIL)
    EL2, CF2, SD2 = TAGS["ep_len"], TAGS["crab_failure"], TAGS["std"]

    def run_ref(el=1800.0, cf=0.2, sd=0.82, n=5000, first=19994):
        return {"kind": "run", "name": "P2__C0_s18", "first": first,
                "ser": {t: [(first + i, v) for i in range(n)] for t, v in ((EL2, el), (CF2, cf), (SD2, sd))}}

    def arm_ser(n, el=1800.0, cf=0.2, sd=0.82, first=19994):
        return {t: [(first + i, v) for i in range(n)] for t, v in ((EL2, el), (CF2, cf), (SD2, sd))}
    check_("p2s guards: before +300 is n/a", judge_p2s_guards(arm_ser(250, el=100.0), 19994, run_ref())["rc"], RC_PASS)
    check_("p2s guards: a healthy arm passes", judge_p2s_guards(arm_ser(1200), 19994, run_ref())["rc"], RC_PASS)
    check_("p2s guards: episode length 0.89 x C0's kills",
           judge_p2s_guards(arm_ser(400, el=1602.0), 19994, run_ref())["rc"], RC_FAIL)
    check_("p2s guards: episode length exactly 0.9 x C0's passes (boundary)",
           judge_p2s_guards(arm_ser(400, el=1620.0), 19994, run_ref())["rc"], RC_PASS)
    check_("p2s guards: sigma 0.79 x C0's at +1000 kills",
           judge_p2s_guards(arm_ser(1100, sd=0.6478), 19994, run_ref())["rc"], RC_FAIL)
    check_("p2s guards: sigma 0.79 x before +1000 is not read",
           judge_p2s_guards(arm_ser(900, sd=0.6478), 19994, run_ref())["rc"], RC_PASS)
    check_("p2s guards: crab_failure +0.09 over C0's from +500 is NOTIFY-ONLY (rc 5, never a kill)",
           judge_p2s_guards(arm_ser(600, cf=0.29), 19994, run_ref())["rc"], RC_NOTIFY)
    check_("p2s guards: crab_failure +0.09 before +500 is not read",
           judge_p2s_guards(arm_ser(450, cf=0.29), 19994, run_ref())["rc"], RC_PASS)
    check_("p2s guards: a kill outranks the flag",
           judge_p2s_guards(arm_ser(600, el=1500.0, cf=0.29), 19994, run_ref())["rc"], RC_FAIL)
    check_("p2s guards: the SAME fork-relative window (the reference drops late: no kill early)",
           judge_p2s_guards(arm_ser(400, el=1700.0), 19994,
                            dict(run_ref(), ser={EL2: [(19994 + i, 1800.0 if i < 1000 else 1000.0) for i in range(5000)],
                                                 CF2: [(19994 + i, 0.2) for i in range(5000)],
                                                 SD2: [(19994 + i, 0.82) for i in range(5000)]}))["rc"], RC_PASS)
    tail_ref = {"kind": "tail", "name": "P2__S1_s17", "tail": {EL2: 1726.0, CF2: 0.29, SD2: 0.81}}
    check_("p2s guards: s16 / s17 against the seed's constant source tail (kill line 0.85 there, not 0.9)",
           [judge_p2s_guards(arm_ser(600, el=1700.0, cf=0.30, sd=0.81), 19994, tail_ref)["rc"],
            judge_p2s_guards(arm_ser(600, el=1500.0, cf=0.30, sd=0.81), 19994, tail_ref)["rc"],
            judge_p2s_guards(arm_ser(600, el=1400.0, cf=0.30, sd=0.81), 19994, tail_ref)["rc"]],
           [RC_PASS, RC_PASS, RC_FAIL])
    check_("p2s guards: an unreadable reference waits (the preflight refuses it before launch)",
           judge_p2s_guards(arm_ser(600), 19994, {"error": "no record"})["rc"], RC_WAIT)
    print("self-test:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--iters", type=Path)
    ap.add_argument("--check", type=Path)
    ap.add_argument("--smoke-audit", type=Path)
    ap.add_argument("--chain")
    ap.add_argument("--at", choices=("200", "1000", "collapse", "p2s"))
    ap.add_argument("--train-log", default=None)
    ap.add_argument("--source", type=Path, default=None)
    ap.add_argument("--candidates", type=Path, default=CAMPAIGN / "candidates.yaml")
    ap.add_argument("--replay", type=Path, default=CAMPAIGN / "replay_gate.json")
    ap.add_argument("--require-terrain-move", action="store_true")
    ap.add_argument("--preflight", action="store_true")
    ap.add_argument("--code-root", type=Path, default=R)
    ap.add_argument("--code-guard", action="store_true")
    ap.add_argument("--lrout", type=Path, default=None)
    ap.add_argument("--anchor-name", default=None)
    ap.add_argument("--reanchor", action="store_true")
    ap.add_argument("--arms", type=Path, default=None,
                     help="p2b (TG) / p2s (the guards' reference): the arms dir to read the records from "
                          "(default $P2_OUT/arms)")
    ap.add_argument("--p2b-checkpoints", type=int, default=None, metavar="FORK_MODEL_N",
                     help="print the p2b intermediate checkpoint numbers for a fork of model_<FORK_MODEL_N>.pt")
    ap.add_argument("--z2-dose", type=Path, default=None,
                     help="the Z2 arm's +200 realised-dose band (P2R_Z2_PLAN step 4; default the arm's own file: "
                          "the campaign's p2r_z2d_dose.json for Z2D -- NOTE(p2r Z2d, 2026-10-11) --, else "
                          "p2r_z2_dose.json)")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.arms is None:
        a.arms = Path(os.environ.get("P2_OUT") or OUT_DEFAULT) / "arms"
    if a.p2b_checkpoints is not None:
        print(" ".join(str(n) for n in p2b_checkpoints(a.p2b_checkpoints)))
        return 0
    if a.self_test:
        return _self_test()
    if a.code_guard:
        rc, res = code_guard(a)
        print(json.dumps(res, default=str))
        return rc
    if a.iters:
        f, l = first_last(read_tb(a.iters))
        print(f"{f if f is not None else 0} {l if l is not None else 0}")
        return 0
    if a.check:
        if not a.chain or not a.at:
            ap.error("--check needs --chain and --at")
        rc, res = check(a)
    elif a.smoke_audit:
        if not a.chain:
            ap.error("--smoke-audit needs --chain")
        rc, res = smoke_audit(a)
    elif a.preflight:
        if not a.chain:
            ap.error("--preflight needs --chain")
        rc, res = preflight(a)
    else:
        ap.error("one of --iters / --check / --smoke-audit / --preflight / --self-test")
    print(json.dumps(res, default=str))
    return rc


if __name__ == "__main__":
    sys.exit(main())
