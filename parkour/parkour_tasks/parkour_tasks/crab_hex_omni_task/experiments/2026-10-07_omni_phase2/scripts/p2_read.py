#!/usr/bin/env python3
"""Omni phase 2 -- the pre-registered reads (CHARTER 5 noise bands and anchor checks, 6.1 per-seed ladder, 6.2 recipe
read, 6.3 field-vs-ring, 6.4 sibling recommendations, 6.5 p2b trigger, 7 ranking inputs). CPU only, read-only.

INPUTS. Records $OUT/arms/<ref>.record.json (scripts/run_arm.sh: run_dir, checkpoint, driver_stamp). Eval stamps
$OUT/lr/<name>/<row>/seed00<k>/<ts>/ (run_arm.sh --row-eval / --lr-eval), each with its code_state.json sidecar.
Names: anchors ANCW_s16 / ANCW_s17 / ANCW_s18 (anchors.yaml; ANC_s* = the ramp-era set, CHARTER 9); arm heads = the reference (P2__A1_s18 ...); intermediate
checkpoints <ref>_m16000 / _m17500 / _m19000. Hand classes: <ARMS>/<name>_lr.json + _lr_along.json read through the
precedent's x6k_read.read_point (feet AND along, fail closed: a missing read is NA, never FAST).

THE READS (one point per evaluated checkpoint):
  completion  the 8-hold rows' schedule_completion_rate (field_hard / field_light / ring_obst), POOLED over env_seeds
              1 and 2: (c1*n1 + c2*n2) / (n1 + n2) -- n = 200 for anchors and arm heads alike (CHARTER 6.0).
  flat list   from suite__omni at env_seed 1 (an arm head: its driver suite stamp; an anchor: its row-eval):
              tracking error |ratio - 1| of fwd / back / side / diag (by_category.<c>.ratio.median; larger = worse),
              turn facing_closed and yaw_progress (medians; smaller = worse), move_turn survival (mean of the
              move_turn / move_turn_r survival_by_hold; smaller = worse), translate survival = every fwd / lat_l / lat_r
              / back / diag hold's survival_by_hold.
  hands       side22_25lr__omni (env_seed 1): FAST only if feet AND along FAST; fall share (verdict.fall_share).
  dose        median |mu| over the run's last 100 logged iterations, cams and rods (x6k_read.tail_reads; <= 1.5 ok).
  jitter      mean over fwd / back / side / diag / turn / move_turn of by_category.<c>.action_jitter.median.
  stand       by_category.stand tripod / steps_per_s / six_down_frac medians and the lift clearance from the raws:
              per stand episode, per foot, p95 - p5 of (foot z - terrain z) over the steady hold window, median over
              the six feet, then median over episodes. (Measured against sf/STAND_STEPPING.md T1: s18 / s16 / s17 c6x
              0.0825 / 0.1232 / 0.0763 vs T1's 0.080 / 0.122 / 0.074; the s18 c6 tap reads 0.024 vs T1 0.035 -- both
              under the 0.05 m line.) The same estimator is applied to every head compared.
  power       mean over the moving-category episodes' steady windows of sum over the ACTUATED joints (run_meta
              action_joint_names) of |tau * qdot| (applied_torque x joint_vel), W.
  exposure    (6.3, one estimator for both arms and both rows) from the env_seed-1 raws: share of steps with the base
              beyond 1.7 m of its spawn, share of steps with any foot over a block or pit (|terrain z - the spawn's
              median terrain z| > 0.02 m) and block steps per episode.
  levels      (6.3, side by side) the run's Metrics/base_parkour/terrain_levels (all tiles) and, when logged,
              terrain_levels_obst (obstacle tiles): mean of the fork's first 20 logged iterations ("start") and of its
              last 100 (the x6k tail window), from the arm's run_dir tfevents.
  field_turn  (7, the "turn band" stand-in -- CHARTER 9 reading: turn_band__omni is not evaluated) field_turn__omni at
              env_seed 1: completion, fall share and the turn_l90 / turn_r90 holds' facing_closed and yaw_progress_rate
              medians (mean of the two holds).

CHARTER 5 NOISE BANDS, per anchor seed and metric: |env_seed 1 - env_seed 2| of that anchor, floored at
  completion / survival: max(|d|, the pooled binomial SE sqrt(p(1-p)/n) at the pooled p and n, 0.04);
  |ratio - 1| and facing_closed: max(|d|, 0.02);  yaw_progress: max(|d|, 0.005).
  The suite metrics (flat list) use s18's band for every seed (s18 also runs its suite at env_seed 2).

CHARTER 6.1 PER SEED vs its own anchor, first match decides:
  INVALID EVAL  a needed stamp lacks KRABBY_OMNI_CLOCK_HIDE=1, KRABBY_STD_PARAM soft, KRABBY_OMNI_SPEED_ELLIPSE
                0.35:0.35:0.25 or KRABBY_OMNI_FACING_OBS_CLIP 0.6; or its row env block (run_meta env_vars_manifest),
                or its sidecar's code_sha / eval_sha / dirty, differs from the anchor's same row; a missing sidecar is
                INVALID, never equal.
  PENDING       a needed read missing or NA (a NA hand class included).
  HAND LOSS     a side hand not FAST, or hold-fall share > 0.15.
  DOSE          median |mu| > 1.5 (cams or rods) over the last 100 iterations.
  FLAT FAIL     a listed metric worse than the anchor by MORE than max(floor, 2 band) (floor 0.10; 0.03 for
                yaw_progress), or a translate hold's survival < 0.95.
  NO GAIN       NOT [field_hard >= anchor + max(0.10, 2 band) AND field_light >= anchor - max(0.03, band) AND
                ring_obst >= anchor - max(0.05, band)].
  COST          a gain, but a listed metric worse by MORE than max(margin, band); margins |ratio - 1| 0.03, fc 0.03,
                yp 0.01, move_turn survival 0.05.
  PASS          otherwise.
CHARTER 6.2: A1 s18 PASS continues, anything else stops (--gate). At 3 seeds: ADOPTABLE (>= 2/3 PASS and no HAND
LOSS) > TRADE-OFF (>= 2/3 gain and one metric named by COST in >= 2/3) > NOT ADOPTABLE (<= 1/3 gain or >= 2/3 HAND
LOSS) > MIXED; any INVALID / PENDING seed -> PENDING.
CHARTER 6.4 siblings vs A1 s18 (paired, n = 1; every verdict stops and asks): S1 / E1 KEEP if both hands FAST, side
feet_ss (hands-averaged lr feet) drops by <= max(0.02, band) [no feet band is defined in CHARTER 5: the 0.02 m/s
floor alone], jitter down >= 25 %, and nothing loses beyond max(margin, band) [the listed metrics at the CHARTER 6.1
margins, and -- this reader's reading of "nothing" -- the obstacle completions at the gain test's hold-the-line
tolerances, field_hard at 0.05]; E1 also needs mean |tau*qdot| down (stand stepping reported). F1 KEEP (low tap) if
A1 s18 still marches (clearance > 0.05 m and stand tripod > 0.4), F1's stand clearance <= 0.05 m and tripod <= 0.4,
and nothing loses; if A1 s18 already taps F1 is NOT RECOMMENDED (no effect to attribute) and reported.
CHARTER 6.5 p2b trigger: gain in >= 2/3 AND a remaining deficit (pooled field_hard < 0.85, or field_hard's walking-hold
tracking_ratio median, seed mean, < 0.7) in >= 2/3 -> stop and ask.
CHARTER 7: intermediate checkpoints (--m-screen): field_hard at env_seed 1 (n = 100) beats the head's by more than
max(0.10, 2 band) -> an env_seed-2 read; it REPLACES the head only if the pooled (n = 200) margin holds with floors
held (no HAND LOSS, no FLAT FAIL vs the anchor). Ranking inputs (floors, obstacle rank-sum, phase-1 rank-sum, jitter
and stand stepping, turn band) are reported for the judges; the ranking is theirs.

    p2_read.py [--json OUT] [--md OUT] [--no-tb] [--kept S1,E1]
    p2_read.py --anchors-check           CHARTER 5 discrimination + inertness + every anchor stamp's
                                         run_meta.checkpoint_sha256 == anchors.yaml: exit 0 ok / 4 stop and ask
    p2_read.py --gate                    A1 s18 (CHARTER 6.2): exit 0 PASS / 4 anything else
    p2_read.py --field-vs-ring           CHARTER 6.3 (descriptive), exit 0
    p2_read.py --sibling S1|E1|F1        CHARTER 6.4: prints the recommendation; exit 4 (always stop and ask)
    p2_read.py --recipe                  CHARTER 6.2 / 6.5: exit 4 when the p2b trigger fires or a seed is PENDING, else 0
    p2_read.py --m-screen REF            names of REF's intermediate checkpoints needing an env_seed-2 field_hard read
    p2_read.py --calibrate RUN_DIR       CHARTER 5 r-bar -> KRABBY_TERRAIN_PROMOTE "<up>:<down>": r-bar = the mean of
                                         Metrics/base_parkour/track_ratio_obst_p50 over the smoke's last 100 logged
                                         iterations (its tfevents; CHARTER 9 operational definition). A per-episode
                                         DUMP file (.json / .npz) is still accepted in place of RUN_DIR.
    p2_read.py --smoke-s5 STAMP|NAME     CHARTER 5 S5 (pre-launch audit B1): the field_light__omni stamp of
                                         run_arm.sh --row-eval <NAME> <s18 head> <s18 env stamp> scenarios_p2.yaml
                                         field_light__omni 1 --episodes 7 (a NAME resolves to its newest seed001
                                         stamp): 7 episodes, the s18 head's sha256, ZERO stand-hold falls (a fall
                                         whose last segment is the prologue or the stand hold; read from the raws,
                                         fail closed). Exit 0 PASS / 4.
    p2_read.py --write-smokes-ok --smokes DIR --s5 STAMP|NAME [--no-charter-log]
                                         writes $OUT/SMOKES_OK only when smokes/<UTC>/smokes.json (scripts/p2_smoke.py)
                                         has S1-S4 PASS on clean code, S3's toe radius <= 1.6 m and S4's numbers,
                                         S5 passes, the CAL1 fork smoke's record shows its dump audit PASS, and the
                                         smokes' and S5's code state equal the LIVE code state. It logs the S3 toe
                                         radius and the S4 numbers in CHARTER 9 (one appended line). Exit 0 / 4.
    p2_read.py --smokes-ok-check         the queue's gate before anything runs: SMOKES_OK present, PASS, complete, and
                                         its code state == the live one. Exit 0 / 4.
    p2_read.py --cal2-gate               the queue's gate before P2__A1_s18 (pre-launch audit M2): P2__CAL2_s18's
                                         record is a smoke, not killed, its dump audit PASS, and its PROMOTE == the
                                         A1 chain's KRABBY_TERRAIN_PROMOTE in candidates.yaml (== CAL2's own). 0 / 4.
    (--out DIR points those four at another $OUT; P2_OUT in the environment does the same for every read.)
    p2_read.py --p2b-inertness           CHARTER 10 / P2B_PLAN step 3: every REF_S1_s<seed> reproduces its S1 p2a head's
                                         own run_arm row-eval reads EXACTLY (field_hard es1 / es2, field_light and
                                         ring_obst es1 / es2, field_turn, field_suite, lr; like-for-like, never the
                                         driver suite stamp), same checkpoint sha256 and row env block. Exit 0 / 4.
    p2_read.py --p2b-screen              the s18 p2b verdicts (C0 and each lever vs REF_S1_s18) + the attribution
                                         JSON; exit 0 (the queue stops and asks regardless).
    p2_read.py --p2b-recipe              CHARTER 6.2 rules per lever over s16 / s17 / s18 (vs REF); exit 4 if any
                                         lever reads PENDING, else 0.
    p2_read.py --p2b-md PATH             the p2b verdict table + attribution + recipe reads as markdown (combinable).
    (P2B_LEVERS in the environment, default "CL TG", names the lever arms read.)
    p2_read.py --p2s-inertness [--p2s-seeds 18|16,17]
                                         CHARTER 11 / P2S_PLAN step 5: REFS_S1_s<seed> (and, at 18, C0R_s18) reproduce
                                         the p2b stamps' reads (REF_S1_s<seed> / P2__C0_s18) EXACTLY on every row they
                                         share (suite es1 / es2 included; never a driver suite stamp). Exit 0 / 4.
    p2_read.py --p2s-is-still NAME       exit 0 iff NAME reads STILL (with or without COST), else 4 (the queue's ST1
                                         condition, P2S_PLAN step 6).
    p2_read.py --p2s-screen              the s18 p2s verdicts (C0R, ST2, ST1 vs REFS_S1_s18), attribution vs C0R, C0R's
                                         own clause values and the reported-only items as JSON; exit 0.
    p2_read.py --p2s-recipe              the step-7 recipe read of the kept dose (P2S_DOSE) over s16 / s17 / s18, the
                                         STILL / PARTIAL / COST -> CHARTER 6.2 mapping spelt out; exit 4 if PENDING.
    p2_read.py --p2s-md PATH             the p2s verdict table + attribution + recipe as markdown (combinable).
    p2_read.py --p2s-self-test           synthetic STILL / PARTIAL / GAMED / COST / NO EFFECT (and ladder) cases.
    (P2S_ST2_S18 / P2S_ST1_S18 / P2S_DOSE / P2S_REP_S16 / P2S_REP_S17 in the environment name the p2s heads -- fresh
    names after a kill; the defaults are P2__ST2_s18, P2__ST1_s18, ST2, P2__<dose>_s16, P2__<dose>_s17.)
    p2_read.py --p2r-probe               CHARTER 12 / P2R_PLAN.md step 1's launcher-inertness check:
                                         P2R_PROBE_s18's three stamps (suite es1, field_hard es1, walkstop es1)
                                         must reproduce REFS_S1_s18 bitwise, raw actions included; plus, when
                                         P2R_HOLD_s18 exists, its swap-live proofs (zero actions on every
                                         p2r_stand_mode step, stand-mode share > 0, pre-entry actions == REFS).
                                         Exit 0 / 4.
    p2_read.py --p2r-step1 [--p2r-drift-json DRIFT.json] [--p2r-md PATH]
                                         CHARTER 12 / P2R_PLAN.md step 1's read: P2R_HOLD_s18 and P2R_ST2M_s18
                                         through their own tier-1 ladder vs REFS_S1_s18 (HAND LOSS / DOSE / ring_obst
                                         skipped outright -- tier 1 never reads them; field_light compared at
                                         matched n, es1 only), "survives tier 1" per head, the stand-mode /
                                         handover-fall / turn-heading reads, and the step-3 drift premise (STOP if
                                         D(C0 +506) > 0.5 x D(ST2 +506)). --p2r-drift-json takes scripts/p2r_drift.py's
                                         own JSON output (the CPU-scored C0 model_20500 / model_21000). Exit 0
                                         always (CHARTER 12: every verdict stops and asks regardless).
    (P2R_PROBE_S18 / P2R_HOLD_S18 / P2R_ST2M_S18 in the environment name the p2r heads -- fresh names after a kill;
    defaults P2R_PROBE_s18 / P2R_HOLD_s18 / P2R_ST2M_s18.)
    p2_read.py --p2r-rst [NAME] [--p2r-md PATH]
                                         NOTE(p2r Z2, 2026-10-10) P2R_Z2_PLAN.md step 3's PREMISE GATE on a
                                         controller-probe stamp set (default P2R_RST_s18): the Z2 stand gates
                                         without the ladder and the relative settle clause, field_turn / field_hard
                                         es1 vs P2R_HOLD_s18 (2 SE), stand body height vs REFS_S1_s18 (0.03 m); es1
                                         and es2 separately, unrun rows NOT RUN. Exit 0 PASS / PASS ON ROWS RUN, 4.
                                         (Cams-only amendment 2026-10-10: six-down REPORTED, not gated; settle =
                                         homed held 0.5 s; STAND_PROXY must be off.)
    p2_read.py --p2r-z2 [NAME] [--probe NAME] [--p2r-md PATH]
                                         P2R_Z2_PLAN.md step 5's read of NAME (default P2__Z2_s18): the --p2r-z
                                         ladder + attribution verbatim, INVALID widened (STAND_PROXY off -- the
                                         cams-only chain, 2026-10-10 -- and the raw bit),
                                         the Z2 stand gates (relative settle vs --probe, default P2R_RST_s18),
                                         adoption_grade, baselines / fidelity / walls vs Z1 / hip yaw. Exit 0 unless
                                         INVALID (4). (P2R_Z2_S18 / P2R_RST_S18 rename the heads.)
    p2_read.py --self-test

CHARTER 11 (stage p2s, P2S_PLAN.md "Pre-registered verdicts"; pre-registered before any p2s run). Points:
REFS_S1_s<seed> (the S1 p2a head model_19994 re-evaluated on the p2s code: every p2b row, suite at env_seeds 1 and 2,
walkstop__omni at 1 and 2), C0R_s18 (the p2b C0 head re-evaluated the same way), the p2s heads (suite@1 = the arm's
driver stamp; every other row by run_arm.sh). The stand reads come from scripts/p2s_stand.py (the ONE estimator set):
the suite's stand wave (head_summary), the field rows' opening stands, field_turn's stand and the walkstop stands
(every hold labelled "stand", each with episode_stand_metrics; onset stand -> moving hold with episode_onset_latency).
Per head vs REFS_S1 of its seed, first match decides:
  INVALID EVAL / PENDING / HAND LOSS / DOSE  as CHARTER 10 (stamps incl. suite@2 and walkstop@1/@2; a walkstop stamp
                with a row env block or KRABBY_FLAT_TERRAIN_MODE in its env is INVALID -- not the flat row).
  FLAT FAIL     as CHARTER 10: vs REFS, and vs ANCW through chained deltas.
  OBSTACLE LOSS field_hard (n = 400) < REFS - max(0.05, 2 SE); field_light < REFS - max(0.03, band); ring_obst < REFS -
                max(0.05, band).
  STAND OUTCOME on suite es1 AND es2: STILL (p2s_stand.still_clauses on both: steps/s <= 0.5, clearance <= 0.05 m,
                off-ground <= 0.10, hover <= 5/100, taps/s <= 1.0, KS <= 0.5 m/s) > PARTIAL (both seeds: off-ground and
                steps/s each <= 0.5 x the comparator's, hover <= comparator + 5, KS <= comparator + band; the comparator
                is C0R at s18, REFS at s16 / s17) > GAMED (steps/s <= 0.5 on both, a PARTIAL off-ground / hover / KS
                clause fails; a flag, never adopts) > NO EFFECT. Stand bands: REFS es1 vs es2, floors off-ground 0.02,
                steps/s 0.25, taps/s 0.25, KS 0.10 m/s, onset 0.1 s.
  COST          on STILL / PARTIAL: a CHARTER 6.1 COST margin vs REFS; pooled stand-hold falls (suite es1+es2,
                walkstop es1+es2, the field rows' opening stands, field_turn's stand) > REFS + max(0.03, 2 SE); walkstop
                MEDIAN onset (es1+es2 pooled) > REFS + max(0.3 s, band); suite stand body height > 0.03 m off REFS;
                field_turn completion < REFS - max(0.05, band) (band = max(binomial SE, 0.04): field_turn has one seed).
  Always shown: the stand outcome (even when an earlier rung matched), the onset delta, C0R's own clause values, and a
  note when PARTIAL comes from the clearance clause alone. Reported only: sigma (cam / rod) at the fork, +1000 and the
  end; steps_per_rad_post; walking tripod / steps-per-m / apex vs C0R; the share of stand actions at the clip.
ATTRIBUTION (s18): ST reads STILL or PARTIAL; C0R reads NO EFFECT, implemented as "C0R is not STILL"; no FLAT FAIL vs
  C0R; field_hard >= C0R - max(0.05, 2 SE).
RECIPE (step 7, CHARTER 6.2 rules on the kept dose at 3 seeds): STILL -> PASS, COST -> COST (same metric names), PARTIAL
  -> a gain that is not a PASS; gain = the stand outcome is STILL or PARTIAL (read before the HAND LOSS rung, as CHARTER
  6.1's gain is); INVALID / PENDING -> PENDING.

CHARTER 10 (stage p2b, P2B_PLAN.md step 6; pre-registered before any p2b run). Points: REF_S1_s<seed> (the S1 p2a head
model_19994 re-evaluated on the p2b code: every row by run_arm.sh, field_hard at env_seeds 1-4, s18's suite also at
env_seed 2) and the heads P2__C0_s18 (control, s18 only), P2__<lever>_s<seed> (suite = the arm's driver_stamp,
field_hard at env_seeds 1-4). Per head vs its seed's REF, first match decides:
  INVALID EVAL  CHARTER 6.1's rule against REF's stamps (env4 keys, row env block, code-state sidecar), field_hard es3 /
                es4 included.
  PENDING       a needed read missing (field_hard pooled over env_seeds 1-4 on both sides; the ANCW chain's reads).
  HAND LOSS / DOSE  as CHARTER 6.1.
  FLAT FAIL     the CHARTER 6.1 list vs REF (REF's bands; REF_S1_s18's suite@2 gives the suite band), OR vs the seed's
                ANCW anchor through chained deltas computed from values: (p2b - REF) + (the S1 p2a head's own p2a suite
                read - ANCW), at max(floor, 2 x ANCW's band) -- never through the sidecar rule (ANCW's stamps carry the
                p2a code_sha); or a translate hold < 0.95.
  NO GAIN       NOT [field_hard (n = 400) >= REF + max(0.05, 2 SE), SE = sqrt(pR qR / nR + pA qA / nA), AND
                field_light >= REF - max(0.03, band) AND ring_obst >= REF - max(0.05, band)]. At REF 0.73 2 SE = 0.0602,
                so +0.06 (316/400) is NO GAIN there and 317/400 the smallest pass; at 0.83 / 0.80 +0.06 clears.
  COST / PASS   as CHARTER 6.1 vs REF.
  Every verdict reports field_hard completion (n = 400) and walking-hold tracking (the stamps' tracking_ratio
  medians, n-weighted) beside REF's. s16 / s17 lever reads are labelled "lever + continuation" (no control there).
ATTRIBUTION (s18): a lever beats continuation only if its REF verdict is PASS or COST, its field_hard beats C0's by the
same SE margin rule (both n = 400) and it has no FLAT FAIL vs C0 (the 6.1 list at max(floor, 2 band), REF s18 bands).
RECIPE per lever at 3 seeds: :func:`recipe_read` (CHARTER 6.2) on the p2b verdicts.
"""
from __future__ import annotations

import argparse
import datetime
import functools
import importlib.util
import json
import math
import os
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CAMPAIGN = HERE.parent
R = Path("/home/nickmagus/krabby/krabby-research")
OUT = Path(os.environ.get("P2_OUT") or R / "parkour/outputs/omni_plan/phase2")
ARMS = OUT / "arms"
LROUT = OUT / "lr"
PREC = R / "parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task/experiments/2026-10-01_ppo_construction/scripts"
POLICY_MANIFEST = R / "parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task/policy/manifest.yaml"

FH, FL, RO, FT, FS = "field_hard__omni", "field_light__omni", "ring_obst__omni", "field_turn__omni", "field_suite__omni"
SUITE, SLOW, LR = "suite__omni", "slow__omni", "side22_25lr__omni"
COMP_ROWS = {"field_hard": FH, "field_light": FL, "ring_obst": RO}
SEEDS = (16, 17, 18)
GATE_SEED = 18
HANDS = ("l", "r")
FAST = "FAST"
EPS = 1e-12
#: the CHARTER 6.1 list: key -> (direction, floor, margin); direction +1 = larger is worse, -1 = smaller is worse
LISTED = {"err_fwd": (+1, 0.10, 0.03), "err_back": (+1, 0.10, 0.03), "err_side": (+1, 0.10, 0.03),
          "err_diag": (+1, 0.10, 0.03), "turn_fc": (-1, 0.10, 0.03), "turn_yp": (-1, 0.03, 0.01),
          "mt_surv": (-1, 0.10, 0.05)}
BAND_FLOOR = {"err_fwd": 0.02, "err_back": 0.02, "err_side": 0.02, "err_diag": 0.02, "turn_fc": 0.02,
              "turn_yp": 0.005, "mt_surv": 0.04}
COMP_BAND_FLOOR = 0.04
TRANSLATE_HOLDS = ("fwd", "lat_l", "lat_r", "back", "diag")
TRANSLATE_MIN = 0.95
TRIP = 0.15
GAIN_FH, HOLD_FL, HOLD_RO, HOLD_FH = 0.10, 0.03, 0.05, 0.05
NEEDED_STAMPS = ("suite@1", "lr@1", f"{FH}@1", f"{FH}@2", f"{FL}@1", f"{FL}@2", f"{RO}@1", f"{RO}@2")
#: every stamp whose env / env block / sidecar is compared with the anchor's (CHARTER 6.1 INVALID EVAL): the needed
#: stamps plus field_suite (feeds the CHARTER 7 obstacle rank-sum) and field_turn (the turn-band stand-in)
VALIDITY_STAMPS = NEEDED_STAMPS + (f"{FS}@1", f"{FT}@1")
ENV4 = {"KRABBY_OMNI_CLOCK_HIDE": "1", "KRABBY_STD_PARAM": "soft", "KRABBY_OMNI_SPEED_ELLIPSE": "0.35:0.35:0.25",
        "KRABBY_OMNI_FACING_OBS_CLIP": "0.6"}
Q_CATS = ("fwd", "back", "side", "diag", "turn", "move_turn")
DISCRIM = (0.15, 0.85)
DISCRIM_GAP = 0.10
P2B_FH, P2B_TRACK = 0.85, 0.7
FEET_DROP, JITTER_CUT = 0.02, 0.75
TAP_CLEAR, TAP_TRIPOD = 0.05, 0.4
MCKPTS = (16000, 17500, 19000)
PAD_R = 1.7
BLOCK_DZ = 0.02
SIBLINGS = ("S1", "E1", "F1")
#: The arm the CHARTER 6.2 recipe read (3 seeds), the 6.5 p2b trigger and the CHARTER 7 candidate list replicate.
#: A1 as pre-registered; S1 since the user's decisions of 2026-10-07 (CHARTER 9: S1 kept on its stride metrics, then
#: "S1 recipe at s16/s17" -- the replicates P2__S1_s16 / _s17 run instead of A1's).
RECIPE_ARM = "S1"
#: arms read at all three seeds (load_all); any other arm is read at the gate seed only. A1 is read at s18 only since
#: its s16 / s17 replicates are not queued (CHARTER 9, 2026-10-07; review wf_b1fbf4b2: no never-run PENDING rows).
REPLICATED_ARMS = (RECIPE_ARM,)
#: heads that will never run, reported as NOT RUN (never PENDING) in the verdict table
NOT_QUEUED = {"P2__E1_s18": "dropped by the user 2026-10-06 (CHARTER 4 replay gate FAIL)"}
TURN_HOLDS = ("turn_l90", "turn_r90")
LEVEL_TAGS = {"all": "Metrics/base_parkour/terrain_levels", "obst": "Metrics/base_parkour/terrain_levels_obst"}
LEVEL_START_N, LEVEL_TAIL = 20, 100
#: CHARTER 5 r-bar, operationally (CHARTER 9 log, 2026-10-06): the mean of this tag over the calibration smoke's last
#: CAL_TAIL logged iterations. Each logged value is the median per-episode ep_track_dist / ep_cmd_dist (the ratio
#: promotion_mask gates on; ep_cmd_dist < min_cmd_dist_m excluded) over the ledger window's last exposure_window
#: (1024) obstacle-tile episodes -- on a field run, the field tiles (omni_exposure.OMNI_OBST_RATIO_KEY).
CAL_TAG = "Metrics/base_parkour/track_ratio_obst_p50"
CAL_TAIL = 100


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@functools.lru_cache(maxsize=1)
def X():
    """The precedent's x6k_read (read_point, tail_reads; it loads verdict.py)."""
    return _load(PREC / "x6k_read.py", "_p2_x6k")


# ================================================================================================ stamp reads
def newest_stamp(name: str, row: str, seed: int = 1, lrout: Path = LROUT) -> str | None:
    d = lrout / name / row / f"seed{seed:03d}"
    hits = [p for p in sorted(d.glob("*/")) if stamp_complete(p)]
    return str(hits[-1]) if hits else None


def stamp_complete(p) -> bool:
    """A stamp is complete when the harness wrote complete.json (category_suite rows) or run_arm.sh wrote
    p2_complete.json after a clean exit of a non-suite (8-hold schedule) row."""
    p = Path(p)
    return (p / "complete.json").is_file() or (p / "p2_complete.json").is_file()


def _json(p):
    try:
        return json.load(open(p))
    except (OSError, ValueError, TypeError):
        return None


def stamp_validity(stamp: str | None) -> dict | None:
    if not stamp:
        return None
    meta = _json(Path(stamp) / "run_meta.json") or {}
    env = meta.get("env_vars") or {}

    def same(k, want):
        got = env.get(k)
        if got is None:
            return False
        if k == "KRABBY_OMNI_FACING_OBS_CLIP":
            try:
                return abs(float(got) - float(want)) < 1e-9
            except ValueError:
                return False
        return str(got).strip() == want
    sc = _json(Path(stamp) / "code_state.json")
    return {"env4": all(same(k, v) for k, v in ENV4.items()),
            "env4_missing": [k for k, v in ENV4.items() if not same(k, v)],
            "manifest_env": dict(meta.get("env_vars_manifest") or {}),
            "sidecar": (sc.get("code_sha"), sc.get("eval_sha"), bool(sc.get("dirty"))) if sc else None,
            "ckpt_sha": meta.get("checkpoint_sha256")}


def comp_read(stamp: str | None) -> dict | None:
    if not stamp:
        return None
    a = (_json(Path(stamp) / "scenario_metrics.json") or {}).get("aggregate") or {}
    if a.get("schedule_completion_rate") is None:
        return None
    n = int(a.get("n_episodes") or 0)
    return {"rate": float(a["schedule_completion_rate"]), "n": n,
            "falls": int((a.get("termination_reasons") or {}).get("fall", 0)),
            "track": (a.get("tracking_ratio") or {}).get("median")}


def _med(d, *keys):
    for k in keys:
        d = (d or {}).get(k)
    return d.get("median") if isinstance(d, dict) else d


def suite_reads(stamp: str | None) -> dict | None:
    if not stamp:
        return None
    a = ((_json(Path(stamp) / "scenario_metrics.json") or {}).get("aggregate") or {}).get("suite")
    if not a:
        return None
    by, surv = a.get("by_category") or {}, a.get("survival_by_hold") or {}

    def err(c):
        r = _med(by, c, "ratio")
        return None if r is None else abs(r - 1.0)
    mt = [surv.get(h) for h in ("move_turn", "move_turn_r")]
    jit = [_med(by, c, "action_jitter") for c in Q_CATS]
    cats = {}
    for c, holds in ((c, [h for h, cc in (a.get("category_of_hold") or {}).items() if cc == c]) for c in by):
        sv = [surv.get(h) for h in holds if surv.get(h) is not None]
        cats[c] = {"survival": round(sum(sv) / len(sv), 3) if sv else None,
                   "ratio": _med(by, c, "ratio"), "facing_closed": _med(by, c, "facing_closed"),
                   "yaw_progress": _med(by, c, "yaw_progress")}
    return {"err_fwd": err("fwd"), "err_back": err("back"), "err_side": err("side"), "err_diag": err("diag"),
            "turn_fc": _med(by, "turn", "facing_closed"), "turn_yp": _med(by, "turn", "yaw_progress"),
            "mt_surv": (sum(mt) / 2) if None not in mt else None,
            "trans_surv": {h: surv.get(h) for h in TRANSLATE_HOLDS},
            "jitter": (sum(jit) / len(jit)) if None not in jit else None,
            "stand": {k: _med(by, "stand", k) for k in ("tripod", "steps_per_s", "six_down_frac")},
            "categories": cats}


def turn_row_reads(stamp: str | None) -> dict | None:
    """field_turn__omni: completion, fall share and the turn holds' facing_closed / yaw_progress_rate medians."""
    if not stamp:
        return None
    a = (_json(Path(stamp) / "scenario_metrics.json") or {}).get("aggregate") or {}
    n = int(a.get("n_episodes") or 0)
    if not n:
        return None
    fac = a.get("facing_by_hold") or {}
    fc = [_med(fac, h, "facing_closed") for h in TURN_HOLDS]
    yp = [_med(fac, h, "yaw_progress_rate") for h in TURN_HOLDS]
    return {"completion": a.get("schedule_completion_rate"), "n": n,
            "fall_share": int((a.get("termination_reasons") or {}).get("fall", 0)) / n,
            "turn_fc": sum(fc) / 2 if None not in fc else None, "turn_yp": sum(yp) / 2 if None not in yp else None}


def levels_summary(series, start_n: int = LEVEL_START_N, tail: int = LEVEL_TAIL) -> dict | None:
    """(steps, values) -> mean of the first ``start_n`` logged points ("start") and of the last ``tail`` iterations."""
    if series is None or not len(series[0]):
        return None
    pts = sorted(zip((int(x) for x in series[0]), (float(v) for v in series[1])))
    last = pts[-1][0]
    head = [v for _, v in pts[:start_n]]
    tl = [v for st, v in pts if st >= last - tail + 1]
    return {"start": sum(head) / len(head), "tail": sum(tl) / len(tl), "first_step": pts[0][0], "last_step": last}


def terrain_levels(run_dir: str | None) -> dict | None:
    """CHARTER 6.3 terrain levels: all-tile and (when logged) obstacle-tile level start / tail of one run."""
    if not run_dir:
        return None
    ser = X().V.tb_series(Path(run_dir), LEVEL_TAGS)
    return {k: levels_summary(ser.get(k)) for k in LEVEL_TAGS}


# ------------------------------------------------------------------------------------------------ raw estimators
def _raw_episodes(stamp: str, keep):
    import numpy as np  # noqa: PLC0415

    meta = _json(Path(stamp) / "run_meta.json") or {}
    files = {int(f.stem.split("_")[1]): f for f in (Path(stamp) / "raw").glob("episode_*.npz")}
    labels = meta.get("hold_labels") or []
    for e in meta.get("env_index_map") or [{"env_index": i, "label": None} for i in sorted(files)]:
        if keep(e) and e["env_index"] in files:
            yield meta, labels, e, np.load(files[e["env_index"]])


@functools.lru_cache(maxsize=64)
def stand_clearance(stamp: str | None) -> float | None:
    """p95 - p5 of foot height above terrain over the steady stand window; median over feet, then over episodes."""
    if not stamp:
        return None
    import numpy as np  # noqa: PLC0415

    vals = []
    for _meta, labels, _e, d in _raw_episodes(stamp, lambda e: e.get("label") == "stand"):
        mask = (d["segment_id"] == labels.index("stand")) & d["steady_mask"]
        if mask.sum() < 25:
            continue
        z = d["foot_pos_w"][mask, :, 2] - d["terrain_z"][mask]
        vals.append(float(np.median(np.percentile(z, 95, axis=0) - np.percentile(z, 5, axis=0))))
    return float(np.median(vals)) if vals else None


@functools.lru_cache(maxsize=64)
def actuated_power(stamp: str | None) -> float | None:
    """Mean over moving-category steady steps of sum_j |tau_j * qdot_j| on the actuated joints, W."""
    if not stamp:
        return None
    import numpy as np  # noqa: PLC0415

    tot, n = 0.0, 0
    for meta, labels, e, d in _raw_episodes(stamp, lambda e: e.get("label") not in (None, "stand", "settle")):
        names, act = meta.get("joint_names") or [], meta.get("action_joint_names") or []
        idx = [names.index(j) for j in act if j in names]
        if not idx:
            return None
        mask = (d["segment_id"] == labels.index(e["label"])) & d["steady_mask"]
        p = np.abs(d["applied_torque"][mask][:, idx] * d["joint_vel"][mask][:, idx]).sum(axis=1)
        tot += float(p.sum()); n += int(mask.sum())
    return tot / n if n else None


@functools.lru_cache(maxsize=64)
def exposure(stamp: str | None) -> dict | None:
    """6.3's one estimator: beyond-pad step share, block/pit step share and block steps per episode."""
    if not stamp:
        return None
    import numpy as np  # noqa: PLC0415

    beyond, block, bsteps = [], [], []
    for _m, _l, _e, d in _raw_episodes(stamp, lambda e: True):
        xy = d["root_pos_w"][:, :2]
        if len(xy) < 2:
            continue
        beyond.append(float(np.mean(np.linalg.norm(xy - xy[0], axis=1) > PAD_R)))
        ground = float(np.median(d["terrain_z"][0]))
        over = np.any(np.abs(d["terrain_z"] - ground) > BLOCK_DZ, axis=1)
        block.append(float(np.mean(over)))
        bsteps.append(int(over.sum()))
    if not beyond:
        return None
    return {"beyond_pad_frac": float(np.mean(beyond)), "block_step_frac": float(np.mean(block)),
            "block_steps_per_ep": float(np.mean(bsteps)), "n": len(beyond)}


# ================================================================================================ points
def load_point(name: str, *, suite_stamp: str | None = None, run_dir: str | None = None, arms: Path = ARMS,
               lrout: Path = LROUT, with_tb: bool = True, with_raw: bool = False, fh_seeds=(1, 2)) -> dict:
    """Every read of one evaluated checkpoint (an anchor, an arm head or an intermediate). ``fh_seeds``: the
    field_hard env_seeds read (p2b: 1-4, CHARTER 10; every other row stays at 1 and 2)."""
    st = {"suite@1": suite_stamp or newest_stamp(name, SUITE, 1, lrout), "suite@2": newest_stamp(name, SUITE, 2, lrout),
          "lr@1": newest_stamp(name, LR, 1, lrout), f"{SLOW}@1": newest_stamp(name, SLOW, 1, lrout),
          f"{FT}@1": newest_stamp(name, FT, 1, lrout), f"{FS}@1": newest_stamp(name, FS, 1, lrout)}
    seeds_of = {k: (tuple(fh_seeds) if k == "field_hard" else (1, 2)) for k in COMP_ROWS}
    for k, row in COMP_ROWS.items():
        for s in seeds_of[k]:
            st[f"{row}@{s}"] = newest_stamp(name, row, s, lrout)
    pt = {"name": name, "stamps": st, "valid": {k: stamp_validity(v) for k, v in st.items()},
          "comp": {k: {s: comp_read(st[f"{row}@{s}"]) for s in seeds_of[k]} for k, row in COMP_ROWS.items()},
          "flat_completion": (comp_read(st[f"{SLOW}@1"]) or {}).get("rate"),
          "suite": suite_reads(st["suite@1"]), "suite2": suite_reads(st["suite@2"]),
          "field_suite": suite_reads(st[f"{FS}@1"]), "field_turn": turn_row_reads(st[f"{FT}@1"]),
          "hands": None, "fall": None, "feet_side": None, "dose_ok": None, "dose": None, "run_dir": run_dir}
    p = X().read_point(str(arms / f"{name}_lr.json"), str(arms / f"{name}_lr_along.json"), name)
    if p:
        pt["hands"], pt["fall"], pt["feet"] = p["hands"], p["fall"], p["feet"]
        f = [p["feet"][h] for h in HANDS]
        pt["feet_side"] = sum(f) / 2 if None not in f else None
    if run_dir and with_tb:
        t = X().tail_reads(run_dir)
        pt["dose_ok"], pt["dose"] = t.get("dose_ok"), {k: t.get(k) for k in ("mu_med_cam", "mu_med_rod")}
    if with_raw:
        pt["stand_clearance"] = stand_clearance(st["suite@1"])
        pt["power"] = actuated_power(st["suite@1"])
        pt["exposure"] = {k: exposure(st[f"{row}@1"]) for k, row in (("field_hard", FH), ("ring_obst", RO))}
    return pt


def pooled(pt: dict, key: str) -> tuple[float, int] | None:
    c = (pt.get("comp") or {}).get(key) or {}
    a, b = c.get(1), c.get(2)
    if not a or not b or not (a["n"] and b["n"]):
        return None
    n = a["n"] + b["n"]
    return (a["rate"] * a["n"] + b["rate"] * b["n"]) / n, n


def binom_se(p: float, n: int) -> float:
    return math.sqrt(max(p * (1 - p), 0.0) / n) if n else float("inf")


def bands(anchor: dict, s18: dict) -> dict:
    """CHARTER 5: completion bands from ``anchor`` (seed 1 vs 2), suite bands from ``s18`` (suite@1 vs suite@2).
    A band that cannot be computed is None (-> PENDING)."""
    out = {}
    for k in COMP_ROWS:
        c = (anchor.get("comp") or {}).get(k) or {}
        pl = pooled(anchor, k)
        out[k] = (max(abs(c[1]["rate"] - c[2]["rate"]), binom_se(*pl), COMP_BAND_FLOOR)
                  if pl and c.get(1) and c.get(2) else None)
    s1, s2 = s18.get("suite") or {}, s18.get("suite2") or {}
    for m, floor in BAND_FLOOR.items():
        a, b = s1.get(m), s2.get(m)
        if a is None or b is None:
            out[m] = None
        elif m == "mt_surv":
            out[m] = max(abs(a - b), binom_se((a + b) / 2, 200), floor)
        else:
            out[m] = max(abs(a - b), floor)
    return out


# ================================================================================================ 6.1 ladder (pure)
def invalid_reasons(pt: dict, ref: dict, keys=VALIDITY_STAMPS) -> list[str]:
    out = []
    for k in keys:
        va, vr = (pt.get("valid") or {}).get(k), (ref.get("valid") or {}).get(k)
        if va is None or vr is None:
            continue                                  # a missing stamp is PENDING's, not INVALID's
        if not va["env4"]:
            out.append(f"{pt['name']} {k}: stamp env lacks {va.get('env4_missing')}")
        if not vr["env4"]:
            out.append(f"{ref['name']} {k}: stamp env lacks {vr.get('env4_missing')}")
        if va["manifest_env"] != vr["manifest_env"]:
            out.append(f"{k}: row env block differs from {ref['name']}'s")
        if va["sidecar"] is None or vr["sidecar"] is None:
            out.append(f"{k}: code-state sidecar missing ({pt['name'] if va['sidecar'] is None else ref['name']})")
        elif va["sidecar"] != vr["sidecar"] and not _bridged(va["sidecar"], vr["sidecar"]):
            out.append(f"{k}: code_sha / eval_sha / dirty differ from {ref['name']}'s ({va['sidecar']} vs {vr['sidecar']})")
    return out


def _bridged(sa, sb) -> bool:
    """True iff two clean sidecars with the same eval_sha differ only by code_shas the campaign's
    ``inertness_bridge.json`` records as verified bitwise-inert (P2R_STATUE_PLAN step 2: REFB on the new code ==
    REFS on ae892cec on every raw key of the bridge rows). Any other difference stays INVALID.

    NOTE(p2r Z2, 2026-10-10): every ``ok`` pair at that eval_sha is read (never a hard-coded pair), and the accept
    is the CHAIN of recorded pairs (connected code_shas), so the Z2 bridge appended later (P2R_Z2_PLAN step 2:
    new code == the bridged Z1-era stamps) also links the new code to ae892cec's REFS stamps. Like the single-pair
    rule, it extrapolates each pair's verified rows to the others."""
    if not sa or not sb or sa[1] != sb[1] or sa[2] or sb[2]:
        return False
    br = _json(HERE.parent / "inertness_bridge.json") or {}
    adj: dict = {}
    for pr in br.get("pairs") or []:
        a, b = pr.get("from_code_sha"), pr.get("to_code_sha")
        if pr.get("ok") and pr.get("eval_sha") == sa[1] and a and b:
            adj.setdefault(a, set()).add(b)
            adj.setdefault(b, set()).add(a)
    seen, todo = {sa[0]}, [sa[0]]
    while todo:
        for nxt in adj.get(todo.pop(), ()):
            if nxt not in seen:
                seen.add(nxt)
                todo.append(nxt)
    return sb[0] in seen and sb[0] != sa[0]


def pending_reasons(pt: dict, anc: dict, bd: dict, need_dose: bool = True) -> list[str]:
    miss = []
    for k in NEEDED_STAMPS:
        if not (pt.get("stamps") or {}).get(k):
            miss.append(f"{pt['name']} {k}")
        if not (anc.get("stamps") or {}).get(k) and k != "lr@1":
            miss.append(f"{anc['name']} {k}")
    for k in COMP_ROWS:
        if pooled(pt, k) is None:
            miss.append(f"{pt['name']} pooled {k}")
        if pooled(anc, k) is None:
            miss.append(f"{anc['name']} pooled {k}")
    for who in (pt, anc):
        s = who.get("suite") or {}
        miss += [f"{who['name']} suite {m}" for m in LISTED if s.get(m) is None]
    ts = (pt.get("suite") or {}).get("trans_surv") or {}
    miss += [f"{pt['name']} survival {h}" for h in TRANSLATE_HOLDS if ts.get(h) is None]
    h = pt.get("hands")
    if not h or any(h.get(x) in (None, "NA") for x in HANDS):
        miss.append(f"{pt['name']} hand classes (NA)")
    if pt.get("fall") is None:
        miss.append(f"{pt['name']} hold-fall share")
    if need_dose and pt.get("dose_ok") is None:
        miss.append(f"{pt['name']} dose read")
    miss += [f"band {k}" for k, v in bd.items() if v is None]
    return miss


def worse_by(pt: dict, ref: dict, m: str) -> float:
    d = LISTED[m][0]
    return d * ((pt["suite"][m]) - (ref["suite"][m]))


def gain_read(pt: dict, anc: dict, bd: dict) -> dict:
    fh, fl, ro = (pooled(pt, k)[0] for k in COMP_ROWS)
    afh, afl, aro = (pooled(anc, k)[0] for k in COMP_ROWS)
    need = afh + max(GAIN_FH, 2 * bd["field_hard"])
    ok_fh = fh >= need - EPS
    ok_fl = fl >= afl - max(HOLD_FL, bd["field_light"]) - EPS
    ok_ro = ro >= aro - max(HOLD_RO, bd["ring_obst"]) - EPS
    return {"gain": ok_fh and ok_fl and ok_ro, "field_hard": [fh, afh, need], "field_light_ok": ok_fl,
            "ring_obst_ok": ok_ro, "field_hard_ok": ok_fh}


def seed_verdict(pt: dict, anc: dict, bd: dict, need_dose: bool = True) -> dict:
    """CHARTER 6.1 for one seed (pure; first match decides). ``bd`` = bands(anchor, s18 anchor)."""
    out = {"name": pt.get("name"), "gain": None, "cost": []}
    inv = invalid_reasons(pt, anc)
    if inv:
        return dict(out, verdict="INVALID EVAL", why=inv)
    pend = pending_reasons(pt, anc, bd, need_dose)
    if pend:
        return dict(out, verdict="PENDING", why=pend)
    g = gain_read(pt, anc, bd)
    out["gain"], out["gain_read"] = g["gain"], g
    out["deltas"] = {m: worse_by(pt, anc, m) for m in LISTED}
    lost = [h for h in HANDS if pt["hands"][h] != FAST]
    if lost or pt["fall"] > TRIP + EPS:
        return dict(out, verdict="HAND LOSS", why=[f"hands {pt['hands']}", f"hold-fall share {pt['fall']:.3f}"])
    if need_dose and pt["dose_ok"] is False:
        return dict(out, verdict="DOSE", why=[f"median |mu| {pt.get('dose')} > 1.5"])
    ff = [f"{m} worse by {out['deltas'][m]:.4f} > max({LISTED[m][1]}, 2x{bd[m]:.4f})" for m in LISTED
          if out["deltas"][m] > max(LISTED[m][1], 2 * bd[m]) + EPS]
    ff += [f"translate survival {h} {v:.3f} < {TRANSLATE_MIN}" for h, v in pt["suite"]["trans_surv"].items()
           if v < TRANSLATE_MIN - EPS]
    if ff:
        return dict(out, verdict="FLAT FAIL", why=ff)
    if not g["gain"]:
        return dict(out, verdict="NO GAIN", why=[f"field_hard {g['field_hard'][0]:.3f} vs need {g['field_hard'][2]:.3f}; "
                                                  f"field_light ok {g['field_light_ok']}; ring_obst ok {g['ring_obst_ok']}"])
    cost = [m for m in LISTED if out["deltas"][m] > max(LISTED[m][2], bd[m]) + EPS]
    if cost:
        return dict(out, verdict="COST", cost=cost, why=[f"{m} worse by {out['deltas'][m]:.4f}" for m in cost])
    return dict(out, verdict="PASS", why=["gain, no flat loss"])


# ================================================================================================ 6.2 / 6.5 (pure)
def recipe_read(vs: dict) -> dict:
    """vs: {seed: seed_verdict} for 16, 17, 18."""
    if any(not vs.get(s) or vs[s]["verdict"] in ("INVALID EVAL", "PENDING") for s in SEEDS):
        return {"reading": "PENDING", "why": "a seed is missing, INVALID EVAL or PENDING"}
    n_pass = sum(vs[s]["verdict"] == "PASS" for s in SEEDS)
    n_hl = sum(vs[s]["verdict"] == "HAND LOSS" for s in SEEDS)
    n_gain = sum(bool(vs[s]["gain"]) for s in SEEDS)
    costs = {}
    for s in SEEDS:
        for m in vs[s].get("cost") or []:
            costs.setdefault(m, []).append(s)
    shared = sorted(m for m, ss in costs.items() if len(ss) >= 2)
    base = {"pass": n_pass, "hand_loss": n_hl, "gain": n_gain, "shared_cost": shared}
    if n_pass >= 2 and n_hl == 0:
        return dict(base, reading="ADOPTABLE")
    if n_gain >= 2 and shared:
        return dict(base, reading="TRADE-OFF")
    if n_gain <= 1 or n_hl >= 2:
        return dict(base, reading="NOT ADOPTABLE")
    return dict(base, reading="MIXED")


def p2b_trigger(vs: dict, pts: dict) -> dict:
    """CHARTER 6.5: gain in >= 2/3 and a remaining deficit in >= 2/3 (stop and ask)."""
    n_gain = sum(bool((vs.get(s) or {}).get("gain")) for s in SEEDS)
    deficit = {}
    for s in SEEDS:
        pt = pts.get(s) or {}
        fh = pooled(pt, "field_hard") if pt else None
        tr = [((pt.get("comp") or {}).get("field_hard") or {}).get(k) for k in (1, 2)] if pt else []
        tr = [x["track"] for x in tr if x and x.get("track") is not None]
        track = sum(tr) / len(tr) if tr else None
        deficit[s] = None if fh is None or track is None else (fh[0] < P2B_FH or track < P2B_TRACK)
    n_def = sum(bool(v) for v in deficit.values())
    return {"trigger": n_gain >= 2 and n_def >= 2, "gain": n_gain, "deficit": deficit,
            "pending": any(v is None for v in deficit.values())}


# ================================================================================================ 6.4 siblings (pure)
def _loses(sib: dict, a1: dict, bd: dict) -> list[str]:
    out = [f"{m} worse by {worse_by(sib, a1, m):.4f}" for m in LISTED
           if worse_by(sib, a1, m) > max(LISTED[m][2], bd[m]) + EPS]
    for k, tol in (("field_hard", HOLD_FH), ("field_light", HOLD_FL), ("ring_obst", HOLD_RO)):
        a, b = pooled(sib, k), pooled(a1, k)
        if a is None or b is None:
            out.append(f"{k} completion unread")
        elif a[0] < b[0] - max(tol, bd[k]) - EPS:
            out.append(f"{k} completion {a[0]:.3f} vs A1 {b[0]:.3f}")
    return out


def sibling_rec(arm: str, sib: dict, a1: dict, bd: dict) -> dict:
    """CHARTER 6.4 recommendation for S1 / E1 / F1 vs A1 s18 (``bd`` = s18's bands). Always stop and ask."""
    out = {"arm": arm, "stop_and_ask": True}
    inv = invalid_reasons(sib, a1)
    if inv:
        return dict(out, rec="INVALID EVAL", why=inv)
    need = [("suite", m) for m in LISTED] + [("hands", None), ("feet_side", None)]
    if arm in ("S1", "E1"):
        need.append(("suite", "jitter"))
    if arm == "E1":
        need.append(("power", None))
    if arm == "F1":
        need += [("stand_clearance", None), ("stand_tripod", None)]
    miss = []
    for who in (sib, a1):
        for k, m in need:
            v = (who.get("suite") or {}).get(m) if k == "suite" else (
                ((who.get("suite") or {}).get("stand") or {}).get("tripod") if k == "stand_tripod" else who.get(k))
            if v is None or (k == "hands" and any(v.get(h) in (None, "NA") for h in HANDS)):
                miss.append(f"{who.get('name')} {k}{'.' + m if m else ''}")
    miss += [f"band {k}" for k, v in bd.items() if v is None]
    if miss:
        return dict(out, rec="PENDING", why=miss)
    loses = _loses(sib, a1, bd)
    report = {"stand": {"sib": dict((sib["suite"].get("stand") or {}), clearance=sib.get("stand_clearance")),
                        "A1": dict((a1["suite"].get("stand") or {}), clearance=a1.get("stand_clearance"))},
              "loses": loses}
    if arm in ("S1", "E1"):
        both = all(sib["hands"][h] == FAST for h in HANDS)
        drop = a1["feet_side"] - sib["feet_side"]
        jit = sib["suite"]["jitter"] <= JITTER_CUT * a1["suite"]["jitter"] + EPS
        conds = {"both hands FAST": both, f"side feet_ss drop {drop:.4f} <= {FEET_DROP}": drop <= FEET_DROP + EPS,
                 f"jitter {sib['suite']['jitter']:.4f} <= 0.75 x {a1['suite']['jitter']:.4f}": jit,
                 "nothing loses": not loses}
        if arm == "E1":
            conds[f"mean |tau qdot| {sib['power']:.2f} < A1 {a1['power']:.2f}"] = sib["power"] < a1["power"]
        keep = all(conds.values())
        return dict(out, rec="KEEP" if keep else "DO NOT KEEP", conditions=conds, report=report)
    a1_tripod, sib_tripod = a1["suite"]["stand"]["tripod"], sib["suite"]["stand"]["tripod"]
    marches = a1["stand_clearance"] > TAP_CLEAR and a1_tripod > TAP_TRIPOD
    if not marches:
        return dict(out, rec="NOT RECOMMENDED", why=["A1 s18 already taps at stand (no effect to attribute)"],
                    report=report)
    conds = {f"F1 stand clearance {sib['stand_clearance']:.3f} <= {TAP_CLEAR}": sib["stand_clearance"] <= TAP_CLEAR + EPS,
             f"F1 stand tripod {sib_tripod:.3f} <= {TAP_TRIPOD}": sib_tripod <= TAP_TRIPOD + EPS,
             "nothing loses": not loses}
    return dict(out, rec="KEEP" if all(conds.values()) else "DO NOT KEEP", conditions=conds, report=report)


# ================================================================================================ 5 anchors (pure)
def anchors_check(anc: dict, s18_manifest: dict | None, shas: dict | None = None) -> dict:
    """CHARTER 5: discrimination per anchor seed, inertness of the s18 suite vs policy/manifest.yaml and, with
    ``shas`` ({seed: anchors.yaml sha256}), every evaluated anchor stamp's run_meta.checkpoint_sha256 (fail closed:
    a stamp without one fails)."""
    fails, rows = [], {}
    for s, want in (shas or {}).items():
        for k, v in sorted(((anc.get(s) or {}).get("valid") or {}).items()):
            if v is not None and v.get("ckpt_sha") != want:
                fails.append(f"s{s} {k}: stamp checkpoint_sha256 {v.get('ckpt_sha')} != anchors.yaml {want[:12]}...")
    for s in SEEDS:
        pt = anc.get(s)
        fh = pooled(pt, "field_hard") if pt else None
        fl = pooled(pt, "field_light") if pt else None
        rows[s] = {"field_hard": fh and fh[0], "field_light": fl and fl[0],
                   "flat_completion": (pt or {}).get("flat_completion")}
        if fh is None or fl is None:
            fails.append(f"s{s}: field_hard / field_light pooled read missing")
            continue
        if not DISCRIM[0] - EPS <= fh[0] <= DISCRIM[1] + EPS:
            fails.append(f"s{s}: field_hard completion {fh[0]:.3f} outside [{DISCRIM[0]}, {DISCRIM[1]}]")
        if fl[0] < fh[0] + DISCRIM_GAP - EPS:
            fails.append(f"s{s}: field_light {fl[0]:.3f} < field_hard {fh[0]:.3f} + {DISCRIM_GAP}")
    inert = None
    cats = (((anc.get(GATE_SEED) or {}).get("suite") or {}).get("categories")) or {}
    if not s18_manifest or not cats:
        fails.append("inertness unread (s18 suite or policy/manifest.yaml categories missing)")
    else:
        diffs = []
        for c, want in s18_manifest.items():
            got = cats.get(c) or {}
            for k, v in want.items():
                if k == "n":
                    continue
                g = got.get(k)
                if g is None or round(float(g), 3) != round(float(v), 3):
                    diffs.append(f"{c}.{k} {g} != {v}")
        inert = not diffs
        if diffs:
            fails.append("s18 suite does not reproduce policy/manifest.yaml: " + "; ".join(diffs))
    return {"ok": not fails, "fails": fails, "rows": rows, "inert": inert}


# ================================================================================================ 7 (pure)
def charter7_candidates(res: dict, vs: dict, kept=(), load=None) -> tuple[dict, dict, dict]:
    """CHARTER 7 candidates ``(cands, cand_vs, ms_reads)``: the recipe arm's heads at every seed it ran, the A1 gate
    head and any ``kept`` sibling at s18 (CHARTER 9, 2026-10-07: the recipe arm is S1, kept and replicated). An
    intermediate checkpoint REPLACES its head when :func:`m_screen` says so (n = 200 margin + floors held; review
    wf_b1fbf4b2: the judging path applies the rule rather than only printing it). ``load`` = :func:`load_point`."""
    load = load or (lambda name: load_point(name, with_tb=False))
    cand_refs = [f"{RECIPE_ARM}_s{s}" for s in SEEDS] + [f"A1_s{GATE_SEED}"] + [f"{k}_s18" for k in kept]
    cands, cand_vs = {}, {}
    for lab in cand_refs:
        pt = res["heads"].get(f"P2__{lab}")
        if pt and not pt.get("missing") and lab not in cands:
            cands[lab] = pt
            cand_vs[lab] = vs.get(f"P2__{lab}")
    ms_reads = {}
    for lab in list(cands):
        s = int(lab.rsplit("_s", 1)[1])
        anc_s = res["anchors"][s]
        bd_s = bands(anc_s, res["anchors"][GATE_SEED])
        ms = {it: load(f"P2__{lab}_m{it}") for it in MCKPTS}
        r = m_screen(cands[lab], ms, anc_s, bd_s)
        ms_reads[lab] = r
        repl = [(v["n200"][0], it) for it, v in r.items() if v.get("status") == "REPLACES HEAD"]
        if repl:
            _fh, it = max(repl)
            new = f"{lab}_m{it}"
            cands[new], cand_vs[new] = ms[it], seed_verdict(ms[it], anc_s, bd_s)
            del cands[lab], cand_vs[lab]
    return cands, cand_vs, ms_reads


def m_screen(head: dict, ms: dict, anc: dict, bd: dict) -> dict:
    """CHARTER 7 intermediate screen. ms: {iteration: point}. Returns per checkpoint 'needs_seed2' / 'replaces' / ..."""
    out = {}
    h1 = ((head.get("comp") or {}).get("field_hard") or {}).get(1)
    for it, m in ms.items():
        c1 = ((m.get("comp") or {}).get("field_hard") or {}).get(1)
        if not c1 or not h1 or bd.get("field_hard") is None:
            out[it] = {"status": "PENDING"}
            continue
        margin = max(GAIN_FH, 2 * bd["field_hard"])
        if not c1["rate"] > h1["rate"] + margin + EPS:
            out[it] = {"status": "NO", "n100": [c1["rate"], h1["rate"], margin]}
            continue
        pm, ph = pooled(m, "field_hard"), pooled(head, "field_hard")
        if pm is None or ph is None:
            out[it] = {"status": "NEEDS_SEED2", "n100": [c1["rate"], h1["rate"], margin]}
            continue
        floors = []
        if not m.get("hands") or any(m["hands"].get(x) != FAST for x in HANDS) or m.get("fall") is None \
                or m["fall"] > TRIP + EPS:
            floors.append("HAND LOSS / NA")
        s = m.get("suite") or {}
        if any(s.get(k) is None for k in LISTED) or not anc.get("suite"):
            floors.append("flat reads missing")
        else:
            floors += [k for k in LISTED if worse_by(m, anc, k) > max(LISTED[k][1], 2 * bd[k]) + EPS]
            floors += [h for h, v in (s.get("trans_surv") or {}).items() if v is None or v < TRANSLATE_MIN - EPS]
        holds = pm[0] > ph[0] + margin + EPS
        out[it] = {"status": "REPLACES HEAD" if holds and not floors else "NO (n=200)", "n200": [pm[0], ph[0], margin],
                   "floors_broken": floors}
    return out


def rank_inputs(cands: dict, vs: dict) -> dict:
    """CHARTER 7 ranking inputs (the judges rank). cands: {label: point}; vs: {label: seed_verdict}."""
    rows = {}
    for lab, pt in cands.items():
        s = pt.get("suite") or {}
        rows[lab] = {"floors": (vs.get(lab) or {}).get("verdict"),
                     "field_hard": (pooled(pt, "field_hard") or (None,))[0],
                     "field_light": (pooled(pt, "field_light") or (None,))[0],
                     "ring_obst": (pooled(pt, "ring_obst") or (None,))[0],
                     "field_mt_surv": (pt.get("field_suite") or {}).get("mt_surv"),
                     **{m: s.get(m) for m in LISTED}, "jitter": s.get("jitter"),
                     "stand_steps_per_s": (s.get("stand") or {}).get("steps_per_s"),
                     "stand_clearance": pt.get("stand_clearance"),
                     **{f"field_turn_{k}": (pt.get("field_turn") or {}).get(k)
                        for k in ("completion", "fall_share", "turn_fc", "turn_yp")}}

    def ranks(key, higher_better=True):
        vals = {k: v[key] for k, v in rows.items() if v[key] is not None}
        order = sorted(vals, key=lambda k: -vals[k] if higher_better else vals[k])
        out = {}
        for k in order:
            tied = [j for j in order if vals[j] == vals[k]]
            out[k] = sum(order.index(j) + 1 for j in tied) / len(tied)
        return out
    obst = [ranks(k) for k in ("field_hard", "field_light", "ring_obst", "field_mt_surv")]
    ph1 = [ranks(m, higher_better=LISTED[m][0] < 0) for m in LISTED]
    turn = [ranks(k) for k in ("field_turn_turn_fc", "field_turn_turn_yp")]
    for lab in rows:
        o = [r.get(lab) for r in obst]
        p = [r.get(lab) for r in ph1]
        t = [r.get(lab) for r in turn]
        rows[lab]["obstacle_rank_sum"] = sum(o) if None not in o else None
        rows[lab]["phase1_rank_sum"] = sum(p) if None not in p else None
        rows[lab]["turn_rank_sum"] = sum(t) if None not in t else None     # item 5, the field_turn stand-in
    return rows


# ================================================================================================ calibration
def promote_from_rbar(r: float) -> dict:
    """CHARTER 5 step 3: up = 0.9 r-bar, down = 0.5 r-bar, each rounded to 0.05."""
    rnd = lambda x: round(round(x / 0.05) * 0.05, 2)  # noqa: E731
    up, down = rnd(0.9 * r), rnd(0.5 * r)
    return {"r_bar": r, "up": up, "down": down, "promote": f"{up:g}:{down:g}"}


def calibrate(track, cmd, field_tile, valid=None) -> dict:
    """CHARTER 5: r-bar = median per-episode ep_track_dist / ep_cmd_dist on field-tile (valid) episodes;
    up = 0.9 r-bar, down = 0.5 r-bar, each rounded to 0.05."""
    vals = [t / c for t, c, f, v in zip(track, cmd, field_tile, valid if valid is not None else [True] * len(track))
            if f and v and c and c > 0]
    if not vals:
        return {"ok": False, "why": "no valid field-tile episode in the dump"}
    r = statistics.median(vals)
    return dict(ok=True, n=len(vals), **promote_from_rbar(r))


def calibrate_series(series, tail: int = CAL_TAIL) -> dict:
    """CHARTER 5 r-bar from training telemetry (CHARTER 9 operational definition): the mean of
    ``track_ratio_obst_p50`` over the last ``tail`` logged iterations. ``series`` = (steps, values).
    Non-finite points are dropped BEFORE the tail is taken; fewer than ``tail`` finite points fails closed
    (the smoke is >= 300 iterations, so a short series means a missing or late-starting tag)."""
    if series is None or not len(series[0]):
        return {"ok": False, "why": f"no {CAL_TAG} in the run's tfevents"}
    pts = sorted((int(st), float(v)) for st, v in zip(series[0], series[1]) if math.isfinite(float(v)))
    if len(pts) < tail:
        return {"ok": False, "why": f"{CAL_TAG}: {len(pts)} finite logged iterations < {tail}"}
    win = pts[-tail:]
    r = sum(v for _, v in win) / tail
    return dict(ok=True, n=tail, first_step=win[0][0], last_step=win[-1][0], source=CAL_TAG, **promote_from_rbar(r))


def calibrate_run(run_dir: Path) -> dict:
    """:func:`calibrate_series` on a run directory's tfevents (or the directory of a tfevents file)."""
    d = Path(run_dir)
    if d.is_file():
        d = d.parent
    ser = X().V.tb_series(d, {"r": CAL_TAG})
    return dict(calibrate_series(ser.get("r")), run_dir=str(d))


def calibrate_file(path: Path) -> dict:
    """A per-episode dump (.json mapping or .npz) with ep_track_dist + ep_cmd_dist, OR the per-episode
    ``track_ratio`` the parkour term hands its exposure ledger (= tracking_ratio(ep_track_dist, ep_cmd_dist),
    omni_parkour_event.py), and a field-tile flag (field_tile | is_field_tile | is_obst_tile), optionally a validity
    mask (valid | track_ratio_valid)."""
    if path.suffix == ".npz":
        import numpy as np  # noqa: PLC0415
        d = {k: np.asarray(v).tolist() for k, v in np.load(path).items()}
    else:
        d = json.load(open(path))
    tile = next((d[k] for k in ("field_tile", "is_field_tile", "is_obst_tile") if k in d), None)
    valid = next((d[k] for k in ("valid", "track_ratio_valid") if k in d), None)
    if tile is not None and "ep_track_dist" in d and "ep_cmd_dist" in d:
        return calibrate(d["ep_track_dist"], d["ep_cmd_dist"], tile, valid)
    if tile is not None and "track_ratio" in d:
        r = calibrate(d["track_ratio"], [1.0] * len(d["track_ratio"]), tile, valid)
        return dict(r, source="track_ratio")
    return {"ok": False, "why": f"dump lacks ep_track_dist + ep_cmd_dist (or track_ratio) / a field-tile flag: "
                                f"keys {sorted(d)}"}


# ================================================================================================ pre-launch gates
# Pre-launch audit 2026-10-06: B1 (no anchor eval before the CHARTER 5 smokes S1-S5 and the fork smoke pass: the
# queue requires $OUT/SMOKES_OK) and M2 (no A1 s18 launch before a passed CAL2 at A1's PROMOTE values).
SMOKES_DIR = CAMPAIGN / "smokes"
SMOKES_OK = "SMOKES_OK"
SMOKE_CHECKS = ("S1", "S2", "S3", "S4")
S4_KEYS = ("reach_edge_frac", "field_frac_mean", "block_steps_mean", "off_tile_block_steps_mean")
S5_EPISODES = 7
#: the wall-mode S5 (CHARTER 9, 2026-10-06 walls re-tune; the ramp-mode S5 was SMOKE_S5_s18)
S5_NAME = "SMOKE_S5W_s18"
TOE_MAX = 1.6
#: CAL1 / CAL2 = the wall-mode calibration smokes (CHARTER 9, 2026-10-06 walls re-tune). The ramp-mode ones were
#: P2__CAL1b_s18 (a fresh re-run: P2__CAL1_s18 was killed twice) and P2__CAL2_s18.
CAL1, CAL2, A1_GATE = "P2__CALW1_s18", "P2__CALW2_s18", "P2__A1_s18"
DIFF_KEY, EDGES_KEY = "KRABBY_FLAT_TERRAIN_DIFF", "KRABBY_FIELD_EDGES"
#: the 2a preset's values when an overlay does not set them
DIFF_DEFAULT, EDGES_DEFAULT = "0.20:0.70", "ramp"
PROMOTE_KEY = "KRABBY_TERRAIN_PROMOTE"
CODE_KEYS = ("code_sha", "eval_sha", "dirty")


def _placeholder(v) -> bool:
    import re  # noqa: PLC0415
    return bool(re.match(r"(?i)^\s*TBD", str(v))) or "PENDING" in str(v).upper()


def promote_pair(text) -> tuple[float, float] | None:
    try:
        up, down = (float(x) for x in str(text).split(":"))
    except (TypeError, ValueError):
        return None
    return (up, down) if all(math.isfinite(v) for v in (up, down)) else None


def chain_overlay_value(cands: dict, chain: str, key: str):
    for st in reversed(((cands.get("chains") or {}).get(chain) or {}).get("stages") or []):
        ov = st.get("overlay") or {}
        if key in ov:
            return ov[key]
    return None


def a1_terrain(cands: dict | None) -> dict | None:
    """The terrain the A1 gate arm trains on: ``{"difficulty_range": [lo, hi], "wall": bool}`` from its overlay (the
    2a preset's 0.20:0.70 / ramp when unset). None without a candidates file."""
    if not cands:
        return None
    diff = chain_overlay_value(cands, A1_GATE, DIFF_KEY) or DIFF_DEFAULT
    edges = chain_overlay_value(cands, A1_GATE, EDGES_KEY) or EDGES_DEFAULT
    return {"difficulty_range": [float(x) for x in str(diff).split(":")], "wall": str(edges).strip().lower() == "wall"}


def smoked_terrain(doc: dict | None) -> dict | None:
    """The terrain a p2_smoke.py run actually built (its S1 summary), in :func:`a1_terrain`'s shape; None if unread or
    the field sub-terrains disagree on their edges."""
    s1 = (doc or {}).get("s1_summary") or {}
    edges = {v.get("vertical_edges") for k, v in (s1.get("sub_terrains") or {}).items() if str(k).startswith("omni_field")}
    dr = s1.get("difficulty_range")
    if len(edges) != 1 or not isinstance(dr, (list, tuple)) or len(dr) != 2:
        return None
    wall = edges.pop()
    return {"difficulty_range": [float(x) for x in dr], "wall": wall} if isinstance(wall, bool) else None


def terrain_fails(have: dict | None, want: dict | None, what: str) -> list[str]:
    """CHARTER 9 (2026-10-06 walls re-tune): smokes count only for the terrain the arms train on."""
    if want is None:
        return [f"{what}: the A1 terrain is unread (no candidates)"]
    if not have:
        return [f"{what}: smoked terrain unknown (a smoke from before the walls re-tune?)"]
    out = []
    if have.get("wall") is not want["wall"]:
        out.append(f"{what}: smoked edges {'wall' if have.get('wall') else 'ramp'} != A1's "
                   f"{'wall' if want['wall'] else 'ramp'}")
    hd, wd = have.get("difficulty_range") or [], want["difficulty_range"]
    if len(hd) != 2 or any(abs(float(a) - float(b)) > 1e-9 for a, b in zip(hd, wd)):
        out.append(f"{what}: smoked difficulty_range {hd} != A1's {wd}")
    return out


def audit_of(rec: dict | None) -> dict | None:
    """The CHARTER 5 dump audit a run_arm.sh smoke record carries (its p2_live.py --smoke-audit JSON line)."""
    raw = (rec or {}).get("smoke_audit")
    if isinstance(raw, dict):
        return raw
    try:
        d = json.loads(raw) if raw else None
    except (TypeError, ValueError):
        return None
    return d if isinstance(d, dict) else None


def smoke_record_fails(rec: dict | None, chain: str) -> list[str]:
    if not rec:
        return [f"no {chain} record"]
    fails = []
    if rec.get("chain") != chain:
        fails.append(f"record chain {rec.get('chain')!r} != {chain}")
    if not rec.get("smoke"):
        fails.append("record is not a calibration smoke")
    if rec.get("killed"):
        fails.append(f"killed: {rec.get('killed')}")
    au = audit_of(rec)
    if not au or au.get("result") != "PASS":
        fails.append(f"dump audit {(au or {}).get('result')!r} (fails {(au or {}).get('fails')})")
    return fails


def cal2_gate(rec: dict | None, cands: dict) -> dict:
    """Pre-launch audit M2 (pure): CAL2 passed its dump audit AT the A1 chain's PROMOTE values."""
    fails = smoke_record_fails(rec, CAL2)
    au = audit_of(rec) or {}
    a1 = chain_overlay_value(cands, A1_GATE, PROMOTE_KEY)
    c2 = chain_overlay_value(cands, CAL2, PROMOTE_KEY)
    pa1, pc2, pau = promote_pair(a1), promote_pair(c2), promote_pair(au.get("promote"))
    if a1 is None or _placeholder(a1) or pa1 is None:
        fails.append(f"A1 {PROMOTE_KEY} not fixed: {a1!r}")
    if pau is None:
        fails.append(f"CAL2 audit PROMOTE unreadable: {au.get('promote')!r}")
    elif pa1 is not None and any(abs(x - y) > 1e-9 for x, y in zip(pau, pa1)):
        fails.append(f"CAL2 audited PROMOTE {au.get('promote')} != A1 {a1}")
    if pc2 is None or (pa1 is not None and any(abs(x - y) > 1e-9 for x, y in zip(pc2, pa1))):
        fails.append(f"CAL2 chain PROMOTE {c2!r} != A1 {a1!r} in candidates.yaml")
    return {"ok": not fails, "fails": fails, "a1_promote": a1, "cal2_promote": c2, "cal2_audit_promote": au.get("promote"),
            "cal2_levels_moved": au.get("levels_moved")}


def resolve_stamp(arg: str, row: str, seed: int = 1, lrout: Path | None = None) -> str | None:
    p = Path(arg)
    if stamp_complete(p):
        return str(p)
    return newest_stamp(arg, row, seed, lrout=lrout or LROUT) if "/" not in arg else None


def smoke_s5_read(stamp: str | None, want_sha: str | None) -> dict:
    """CHARTER 5 S5 (pure IO on one stamp): field_light__omni, 7 episodes, the s18 head, zero stand-hold falls."""
    if not stamp:
        return {"ok": False, "fails": ["no S5 stamp"], "stamp": None}
    import numpy as np  # noqa: PLC0415

    st = Path(stamp)
    meta = _json(st / "run_meta.json") or {}
    sm = _json(st / "scenario_metrics.json") or {}
    fails = []
    if meta.get("scenario_id") != FL:
        fails.append(f"scenario {meta.get('scenario_id')!r} != {FL}")
    eps = sm.get("episodes") or []
    n = int((sm.get("aggregate") or {}).get("n_episodes") or len(eps))
    if n != S5_EPISODES or len(eps) != S5_EPISODES:
        fails.append(f"{n} episodes ({len(eps)} listed), want {S5_EPISODES}")
    if want_sha and meta.get("checkpoint_sha256") != want_sha:
        fails.append(f"checkpoint sha256 {str(meta.get('checkpoint_sha256'))[:12]} != the s18 head {want_sha[:12]}")
    labels = list(meta.get("hold_labels") or [])
    if "stand" not in labels:
        fails.append("no stand hold in the row")
    stand_idx = labels.index("stand") if "stand" in labels else None
    stand_falls, other_falls, unread = [], [], []
    for e in eps:
        if str(e.get("termination_reason")) != "fall":
            continue
        idx = int(e.get("env_index"))
        f = st / "raw" / f"episode_{idx:02d}.npz"
        try:
            seg = int(np.load(f)["segment_id"][-1])
        except (OSError, KeyError, ValueError, IndexError):
            unread.append(idx)
            continue
        (stand_falls if seg in (-1, stand_idx) else other_falls).append(idx)
    if unread:
        fails.append(f"fall episodes without a readable raw segment (fail closed): {unread}")
    if stand_falls:
        fails.append(f"stand-hold falls (prologue or stand): {stand_falls}")
    val = stamp_validity(stamp) or {}
    if not val.get("sidecar"):
        fails.append("no code-state sidecar")
    return {"ok": not fails, "fails": fails, "stamp": str(st), "n_episodes": n, "stand_falls": len(stand_falls),
            "other_falls": len(other_falls), "sidecar": val.get("sidecar"), "env4": val.get("env4")}


def smokes_doc_fails(doc: dict | None) -> list[str]:
    """What keeps a p2_smoke.py smokes.json from counting: S1-S4 PASS, finished, no errors, clean code, the S3 toe
    radius and the four S4 numbers present."""
    if not doc:
        return ["no smokes.json"]
    fails = []
    res = doc.get("results") or {}
    for c in SMOKE_CHECKS:
        if (res.get(c) or {}).get("result") != "PASS":
            fails.append(f"{c} {(res.get(c) or {}).get('result', 'not run')}")
    if not doc.get("finished") or doc.get("errors"):
        fails.append(f"smoke run unfinished or errored: {doc.get('errors')}")
    plan = doc.get("plan") or {}
    if plan.get("reference") != A1_GATE:
        fails.append(f"smoked reference {plan.get('reference')!r} != {A1_GATE}")
    cs = plan.get("code_state") or {}
    if cs.get("dirty") is not False:
        fails.append(f"smoked on a dirty or unknown tree: {cs.get('dirty_paths')}")
    toe = (res.get("S3") or {}).get("toe_radius_max_m")
    if not isinstance(toe, (int, float)) or not toe <= TOE_MAX + 1e-9:
        fails.append(f"S3 toe radius {toe!r} (want <= {TOE_MAX} m)")
    for k in S4_KEYS:
        if not isinstance((res.get("S4") or {}).get(k), (int, float)):
            fails.append(f"S4 {k} missing")
    return fails


def code_state_fails(have: dict | None, live: dict | None, what: str) -> list[str]:
    if not have:
        return [f"{what}: no code state"]
    if not live or live.get("dirty") is not False:
        return [f"live code state dirty or unreadable: {(live or {}).get('dirty_paths')}"]
    return [f"{what} {k} {have.get(k)!r} != live {live.get(k)!r}" for k in CODE_KEYS
            if (bool(have.get(k)) if k == "dirty" else have.get(k)) != (bool(live.get(k)) if k == "dirty" else live.get(k))]


def build_smokes_ok(smokes: Path, s5: dict, cal1: dict | None, live: dict | None,
                    cands: dict | None = None) -> tuple[list[str], dict]:
    doc = _json(Path(smokes) / "smokes.json")
    fails = smokes_doc_fails(doc)
    want = a1_terrain(cands)
    terrain = smoked_terrain(doc)
    fails += terrain_fails(terrain, want, "smokes")
    s5_env = (stamp_validity(s5.get("stamp")) or {}).get("manifest_env") or {}
    if want is not None and (str(s5_env.get(EDGES_KEY, EDGES_DEFAULT)).strip().lower() == "wall") is not want["wall"]:
        fails.append(f"S5: row edges {s5_env.get(EDGES_KEY)!r} != A1's {'wall' if want['wall'] else 'ramp'}")
    fails += [f"S5: {f}" for f in s5.get("fails") or []] if not s5.get("ok") else []
    fails += [f"CAL1 fork smoke: {f}" for f in smoke_record_fails(cal1, CAL1)]
    cs = ((doc or {}).get("plan") or {}).get("code_state")
    fails += code_state_fails(cs, live, "smokes")
    sc = s5.get("sidecar")
    fails += code_state_fails(dict(zip(CODE_KEYS, sc)) if sc else None, live, "S5 sidecar")
    res = (doc or {}).get("results") or {}
    payload = {"result": "FAIL" if fails else "PASS", "fails": fails,
               "written_at": datetime.datetime.now().isoformat(timespec="seconds"), "smokes": str(smokes),
               "s3_toe_radius_max_m": (res.get("S3") or {}).get("toe_radius_max_m"),
               "s3_resets": (res.get("S3") or {}).get("resets"),
               "s4": {k: (res.get("S4") or {}).get(k) for k in S4_KEYS},
               "s4_toe_radius_max_walking_m": (res.get("S4") or {}).get("toe_radius_max_walking_m"),
               "s5": {k: s5.get(k) for k in ("stamp", "n_episodes", "stand_falls", "other_falls")},
               "cal1": {"run_dir": (cal1 or {}).get("run_dir"), "audit": (audit_of(cal1) or {}).get("result")},
               "code_state": {k: (live or {}).get(k) for k in CODE_KEYS},
               "smoked_substitutions": ((doc or {}).get("plan") or {}).get("substituted"),
               "terrain": terrain}
    return fails, payload


def smokes_ok_fails(payload: dict | None, live: dict | None, cands: dict | None = None) -> list[str]:
    if not payload:
        return ["no SMOKES_OK"]
    fails = [] if payload.get("result") == "PASS" and not payload.get("fails") else [f"SMOKES_OK says {payload.get('result')}"]
    toe = payload.get("s3_toe_radius_max_m")
    if not isinstance(toe, (int, float)) or toe > TOE_MAX + 1e-9:
        fails.append(f"S3 toe radius {toe!r}")
    if any(not isinstance((payload.get("s4") or {}).get(k), (int, float)) for k in S4_KEYS):
        fails.append("S4 numbers missing")
    if (payload.get("s5") or {}).get("stand_falls") != 0:
        fails.append("S5 stand falls not 0")
    fails += code_state_fails(payload.get("code_state"), live, "SMOKES_OK")
    fails += terrain_fails(payload.get("terrain"), a1_terrain(cands), "SMOKES_OK")
    return fails


def charter_line(p: dict) -> str:
    s4 = p.get("s4") or {}
    f = lambda v, d=3: "NA" if not isinstance(v, (int, float)) else f"{v:.{d}f}"  # noqa: E731
    cs = p.get("code_state") or {}
    return (f"- {datetime.date.today().isoformat()}, **CHARTER 5 smokes passed; `SMOKES_OK` written before any anchor** "
            f"(`scripts/p2_read.py --write-smokes-ok`).\n"
            f"  - S1-S4: `{Path(p['smokes']).relative_to(CAMPAIGN) if str(p['smokes']).startswith(str(CAMPAIGN)) else p['smokes']}`.\n"
            f"  - S3: max toe radius {f(p.get('s3_toe_radius_max_m'))} m (limit {TOE_MAX} m), over {p.get('s3_resets')} "
            f"resets; walking max (S4) {f(p.get('s4_toe_radius_max_walking_m'))} m.\n"
            f"  - S4: reach_edge {f(s4.get('reach_edge_frac'))}, field_frac {f(s4.get('field_frac_mean'))}, block steps per "
            f"obstacle-tile episode {f(s4.get('block_steps_mean'), 1)}, off-tile block steps "
            f"{f(s4.get('off_tile_block_steps_mean'), 2)}.\n"
            f"  - S5: `{(p.get('s5') or {}).get('stamp')}`, {(p.get('s5') or {}).get('n_episodes')} episodes, "
            f"{(p.get('s5') or {}).get('stand_falls')} stand-hold falls.\n"
            f"  - CAL1 dump audit {(p.get('cal1') or {}).get('audit')}. code_sha {str(cs.get('code_sha'))[:8]}…, "
            f"eval_sha {str(cs.get('eval_sha'))[:8]}….\n")


def _live_code_state(code_root: Path = R) -> dict | None:
    try:
        return _load(HERE / "p2_live.py", "_p2_live_for_read").live_code_state(code_root)
    except Exception as e:  # noqa: BLE001 -- unreadable is a FAIL, never a pass
        return {"dirty": None, "dirty_paths": f"unreadable: {e.__class__.__name__}: {e}"}


# ================================================================================================ loading
def _anchors_yaml() -> dict:
    import yaml  # noqa: PLC0415
    return yaml.safe_load(open(CAMPAIGN / "anchors.yaml"))["anchors"]


def _manifest_s18() -> dict | None:
    try:
        import yaml  # noqa: PLC0415
        m = yaml.safe_load(open(POLICY_MANIFEST))
    except (OSError, ValueError):
        return None
    for st in (m or {}).get("stages") or (m or {}).get("phases") or []:
        if isinstance(st, dict) and str(st.get("sha256", "")).startswith("b40f9a19"):
            return ((st.get("evals") or {}).get("categories")) or None
    return _find_s18(m)


def _find_s18(node):
    if isinstance(node, dict):
        if str(node.get("sha256", "")).startswith("b40f9a19") and isinstance(node.get("evals"), dict):
            return node["evals"].get("categories")
        for v in node.values():
            r = _find_s18(v)
            if r:
                return r
    elif isinstance(node, list):
        for v in node:
            r = _find_s18(v)
            if r:
                return r
    return None


def _rec(ref: str) -> dict | None:
    return _json(ARMS / f"{ref}.record.json")


def load_all(with_tb: bool = True, with_raw: bool = False, arms=("A1", "A0", "S1", "E1", "F1")) -> dict:
    an = _anchors_yaml()
    anc = {int(s): load_point(a["name"], with_tb=False, with_raw=with_raw) for s, a in an.items()}
    heads = {}
    for arm in arms:
        for s in (SEEDS if arm in REPLICATED_ARMS else (GATE_SEED,)):
            ref = f"P2__{arm}_s{s}"
            rec = _rec(ref)
            if rec and not rec.get("killed"):
                heads[ref] = load_point(ref, suite_stamp=rec.get("driver_stamp"), run_dir=rec.get("run_dir"),
                                        with_tb=with_tb, with_raw=with_raw)
            else:
                heads[ref] = {"name": ref, "missing": True, "killed": (rec or {}).get("killed")}
    return {"anchors": anc, "heads": heads}


def verdicts(res: dict) -> dict:
    anc = res["anchors"]
    out = {}
    for ref, pt in res["heads"].items():
        s = int(ref.rsplit("_s", 1)[1])
        if pt.get("missing"):
            if ref in NOT_QUEUED and not pt.get("killed"):
                out[ref] = {"name": ref, "verdict": "NOT RUN", "why": [NOT_QUEUED[ref]], "gain": None, "cost": []}
                continue
            out[ref] = {"name": ref, "verdict": "PENDING", "why": [f"no record ({pt.get('killed') or 'not run'})"],
                        "gain": None, "cost": []}
            continue
        out[ref] = seed_verdict(pt, anc[s], bands(anc[s], anc[GATE_SEED]))
    return out


# ================================================================================================ 10 p2b (CHARTER 10)
# P2B_PLAN.md step 6, pre-registered before any p2b run. Every p2b head is read against its seed's REF (the S1 p2a head
# re-evaluated on the p2b code, so the INVALID EVAL sidecar rule holds); the flat floor is ALSO held against the seed's
# ANCW anchor through chained suite deltas computed from values (ANCW's stamps carry the p2a code_sha, so the sidecar
# rule cannot be applied to them).
P2B_CONTROL = "C0"
#: lever arms read at every seed; P2B_LEVERS in the environment narrows the list (the user may prune after the screen)
P2B_LEVERS_DEFAULT = "CL TG"
REF_ARM = "S1"
P2B_FH_SEEDS = (1, 2, 3, 4)
#: CHARTER 10 gain gate: field_hard (n = 400) >= REF + max(P2B_GAIN_FLOOR, P2B_SE_K * SE) (field_light / ring_obst keep
#: CHARTER 6.1's hold-the-line tolerances HOLD_FL / HOLD_RO with REF's bands)
P2B_GAIN_FLOOR, P2B_SE_K = 0.05, 2.0
P2B_NEEDED = NEEDED_STAMPS + tuple(f"{FH}@{s}" for s in P2B_FH_SEEDS[2:])
P2B_VALIDITY = VALIDITY_STAMPS + tuple(f"{FH}@{s}" for s in P2B_FH_SEEDS[2:])
#: the step-3 inertness rows: the S1 p2a head's own run_arm row-eval reads (like-for-like; never the driver suite stamp)
INERT_ROWS = (f"{FH}@1", f"{FH}@2", f"{FL}@1", f"{FL}@2", f"{RO}@1", f"{RO}@2", f"{FT}@1", f"{FS}@1", "lr@1")
INERT_TOL = 1e-12
GAIN_VERDICTS = ("PASS", "COST")


def p2b_levers(env=None) -> tuple:
    return tuple((env if env is not None else os.environ).get("P2B_LEVERS", P2B_LEVERS_DEFAULT).split())


def ref_name(seed: int) -> str:
    return f"REF_{REF_ARM}_s{seed}"


def p2b_head_names(levers=None) -> list[str]:
    """Every p2b head that is (or will be) queued: C0 at the gate seed only, each lever at every seed."""
    levers = p2b_levers() if levers is None else tuple(levers)
    out = []
    for s in SEEDS:
        if s == GATE_SEED:
            out.append(f"P2__{P2B_CONTROL}_s{s}")
        out += [f"P2__{lv}_s{s}" for lv in levers]
    return out


def pooled_seeds(pt: dict, key: str, seeds) -> tuple[float, int] | None:
    """:func:`pooled` over ``seeds`` (all must be read)."""
    c = (pt.get("comp") or {}).get(key) or {}
    got = [c.get(s) for s in seeds]
    if any(not x or not x["n"] for x in got):
        return None
    n = sum(x["n"] for x in got)
    return sum(x["rate"] * x["n"] for x in got) / n, n


def fh400(pt: dict) -> tuple[float, int] | None:
    return pooled_seeds(pt, "field_hard", P2B_FH_SEEDS)


def fh_track(pt: dict, seeds=P2B_FH_SEEDS) -> float | None:
    """field_hard walking-hold tracking: the stamps' tracking_ratio medians, n-weighted over the env_seeds read."""
    c = ((pt or {}).get("comp") or {}).get("field_hard") or {}
    got = [c.get(s) for s in seeds]
    got = [x for x in got if x and x.get("track") is not None and x.get("n")]
    if not got:
        return None
    return sum(x["track"] * x["n"] for x in got) / sum(x["n"] for x in got)


def se_need(p_ref: float, n_ref: int, p_new: float, n_new: int) -> dict:
    """The CHARTER 10 margin: REF + max(0.05, 2 SE), SE = sqrt(pR qR / nR + pA qA / nA)."""
    se = math.sqrt(max(p_ref * (1 - p_ref), 0.0) / n_ref + max(p_new * (1 - p_new), 0.0) / n_new)
    margin = max(P2B_GAIN_FLOOR, P2B_SE_K * se)
    return {"se": se, "margin": margin, "need": p_ref + margin, "ok": p_new >= p_ref + margin - EPS}


def p2b_gain_read(pt: dict, ref: dict, bd: dict) -> dict:
    (fh, nf), (rfh, nr) = fh400(pt), fh400(ref)
    g = se_need(rfh, nr, fh, nf)
    fl, ro = pooled(pt, "field_light")[0], pooled(pt, "ring_obst")[0]
    rfl, rro = pooled(ref, "field_light")[0], pooled(ref, "ring_obst")[0]
    ok_fl = fl >= rfl - max(HOLD_FL, bd["field_light"]) - EPS
    ok_ro = ro >= rro - max(HOLD_RO, bd["ring_obst"]) - EPS
    return {"gain": g["ok"] and ok_fl and ok_ro, "field_hard": [fh, rfh, g["need"]], "n": [nf, nr],
            "se": g["se"], "margin": g["margin"], "field_hard_ok": g["ok"], "field_light_ok": ok_fl,
            "ring_obst_ok": ok_ro}


def _flat_fails(deltas: dict, bd: dict, what: str) -> list[str]:
    return [f"{m} worse than {what} by {deltas[m]:.4f} > max({LISTED[m][1]}, 2x{bd[m]:.4f})" for m in LISTED
            if deltas[m] > max(LISTED[m][1], 2 * bd[m]) + EPS]


def chained_ancw(pt: dict, ref: dict, p2a: dict, ancw: dict) -> dict:
    """CHARTER 10 flat floor vs phase 1, from values: per listed metric (p2b - REF) + (the S1 p2a head's own p2a suite
    read - ANCW), both links in CHARTER 6.1's worse-is-positive direction. Link 2 is the delta the p2a ladder read."""
    return {m: {"p2b_vs_ref": worse_by(pt, ref, m), "p2a_vs_ancw": worse_by(p2a, ancw, m),
                "total": worse_by(pt, ref, m) + worse_by(p2a, ancw, m)} for m in LISTED}


def p2b_report(pt: dict, ref: dict) -> dict:
    """The field_hard completion (n = 400) and walking-hold tracking every p2b verdict carries."""
    a, r = fh400(pt or {}), fh400(ref or {})
    return {"field_hard": a and a[0], "field_hard_n": a and a[1], "field_hard_track": fh_track(pt),
            "ref_field_hard": r and r[0], "ref_field_hard_n": r and r[1], "ref_field_hard_track": fh_track(ref)}


def p2b_pending(pt: dict, ref: dict, bd: dict, p2a: dict, ancw: dict, bd_ancw: dict, need_dose: bool) -> list[str]:
    miss = pending_reasons(pt, ref, bd, need_dose)
    for k in P2B_NEEDED[len(NEEDED_STAMPS):]:
        miss += [f"{who['name']} {k}" for who in (pt, ref) if not (who.get("stamps") or {}).get(k)]
    miss += [f"{who['name']} field_hard pooled over env_seeds {P2B_FH_SEEDS}" for who in (pt, ref) if fh400(who) is None]
    for who in (p2a, ancw):
        s = (who or {}).get("suite") or {}
        miss += [f"{(who or {}).get('name')} suite {m} (ANCW chain)" for m in LISTED if s.get(m) is None]
    miss += [f"ANCW band {m}" for m in LISTED if (bd_ancw or {}).get(m) is None]
    return miss


def p2b_verdict(pt: dict, ref: dict, bd: dict, p2a: dict, ancw: dict, bd_ancw: dict, need_dose: bool = True) -> dict:
    """CHARTER 10 ladder for one p2b head vs its seed's REF (pure; first match decides). ``bd`` = bands(REF, REF s18);
    ``p2a`` = the S1 p2a head's point (its p2a suite read), ``ancw`` = the seed's ANCW anchor, ``bd_ancw`` =
    bands(ANCW, ANCW s18) -- the CHARTER 6.1 FLAT FAIL list is also held against ANCW through chained deltas."""
    out = {"name": pt.get("name"), "gain": None, "cost": [], "report": p2b_report(pt, ref)}
    inv = invalid_reasons(pt, ref, P2B_VALIDITY)
    if inv:
        return dict(out, verdict="INVALID EVAL", why=inv)
    pend = p2b_pending(pt, ref, bd, p2a, ancw, bd_ancw, need_dose)
    if pend:
        return dict(out, verdict="PENDING", why=pend)
    g = p2b_gain_read(pt, ref, bd)
    out["gain"], out["gain_read"] = g["gain"], g
    out["deltas"] = {m: worse_by(pt, ref, m) for m in LISTED}
    out["ancw_chain"] = chained_ancw(pt, ref, p2a, ancw)
    lost = [h for h in HANDS if pt["hands"][h] != FAST]
    if lost or pt["fall"] > TRIP + EPS:
        return dict(out, verdict="HAND LOSS", why=[f"hands {pt['hands']}", f"hold-fall share {pt['fall']:.3f}"])
    if need_dose and pt["dose_ok"] is False:
        return dict(out, verdict="DOSE", why=[f"median |mu| {pt.get('dose')} > 1.5"])
    ff = _flat_fails(out["deltas"], bd, ref["name"])
    ff += _flat_fails({m: v["total"] for m, v in out["ancw_chain"].items()}, bd_ancw, f"{ancw['name']} (chained)")
    ff += [f"translate survival {h} {v:.3f} < {TRANSLATE_MIN}" for h, v in pt["suite"]["trans_surv"].items()
           if v < TRANSLATE_MIN - EPS]
    if ff:
        return dict(out, verdict="FLAT FAIL", why=ff)
    if not g["gain"]:
        return dict(out, verdict="NO GAIN", why=[
            f"field_hard {g['field_hard'][0]:.4f} (n={g['n'][0]}) vs need {g['field_hard'][2]:.4f} = REF "
            f"{g['field_hard'][1]:.4f} + max({P2B_GAIN_FLOOR}, 2 SE {g['se']:.4f}); "
            f"field_light ok {g['field_light_ok']}; ring_obst ok {g['ring_obst_ok']}"])
    cost = [m for m in LISTED if out["deltas"][m] > max(LISTED[m][2], bd[m]) + EPS]
    if cost:
        return dict(out, verdict="COST", cost=cost, why=[f"{m} worse by {out['deltas'][m]:.4f}" for m in cost])
    return dict(out, verdict="PASS", why=["gain over REF (SE gate), no flat loss vs REF or ANCW (chained)"])


def p2b_attribution(lever: dict | None, c0: dict | None, lever_v: dict | None, bd: dict) -> dict:
    """CHARTER 10 attribution at s18 (pure): the lever beats continuation only if its REF verdict is a gain (PASS or
    COST), its field_hard (n = 400) beats C0's by the gain gate's margin rule, and it has no FLAT FAIL vs C0 (the
    CHARTER 6.1 list at max(floor, 2 band), REF s18's bands). ``bd`` = bands(REF s18, REF s18)."""
    out = {"lever": (lever or {}).get("name"), "control": (c0 or {}).get("name"), "beats_continuation": None,
           "lever_verdict": (lever_v or {}).get("verdict")}
    if not lever or lever.get("missing") or not lever_v or lever_v.get("verdict") in ("INVALID EVAL", "PENDING"):
        return dict(out, status="PENDING", why=["the lever's REF verdict is not read"])
    if not c0 or c0.get("missing"):
        return dict(out, status="PENDING", why=["C0 not recorded"])
    inv = invalid_reasons(lever, c0, P2B_VALIDITY)
    if inv:
        return dict(out, status="INVALID EVAL", why=inv)
    miss = [f"{w['name']} field_hard pooled over {P2B_FH_SEEDS}" for w in (lever, c0) if fh400(w) is None]
    miss += [f"{w['name']} suite {m}" for w in (lever, c0) for m in LISTED if (w.get("suite") or {}).get(m) is None]
    miss += [f"band {m}" for m in LISTED if bd.get(m) is None]
    if miss:
        return dict(out, status="PENDING", why=miss)
    (fl, nl), (fc, nc) = fh400(lever), fh400(c0)
    g = se_need(fc, nc, fl, nl)
    ff = _flat_fails({m: worse_by(lever, c0, m) for m in LISTED}, bd, c0["name"])
    conds = {"REF verdict is a gain (PASS / COST)": lever_v["verdict"] in GAIN_VERDICTS and bool(lever_v.get("gain")),
             f"field_hard {fl:.4f} >= C0 {fc:.4f} + max({P2B_GAIN_FLOOR}, 2 SE {g['se']:.4f})": g["ok"],
             "no FLAT FAIL vs C0": not ff}
    beats = all(conds.values())
    return dict(out, beats_continuation=beats, status="BEATS CONTINUATION" if beats else "DOES NOT BEAT CONTINUATION",
                conditions=conds, field_hard=[fl, fc, g["need"]], flat_fails_vs_c0=ff)


def inertness_check(refs: dict, p2a: dict, rows=INERT_ROWS) -> dict:
    """P2B_PLAN step 3 (pure): REF_S1_s<seed> reproduces the S1 p2a head's own run_arm row-eval reads EXACTLY on every
    row the head has (like-for-like: never the driver suite stamp), on the same checkpoint (stamp sha256) and row env
    block. A row the head has and REF lacks fails (fail closed); a REF without any S1 head point fails."""
    fails, out = [], {}
    for s in SEEDS:
        r, h = refs.get(s) or {}, p2a.get(s) or {}
        if not h or not r:
            fails.append(f"s{s}: {'REF' if not r else 'S1 p2a head'} point missing")
            continue
        out[s] = {}
        shas = {((h.get("valid") or {}).get(k) or {}).get("ckpt_sha") for k in rows if (h.get("stamps") or {}).get(k)}
        shas.discard(None)
        if len(shas) != 1:
            fails.append(f"s{s}: the S1 head's row stamps name {len(shas)} checkpoint sha256s")
        for k in rows:
            hs, rs = (h.get("stamps") or {}).get(k), (r.get("stamps") or {}).get(k)
            if not hs:
                out[s][k] = "head lacks the row (not compared)"
                continue
            if not rs:
                out[s][k] = "REF row unread"
                fails.append(f"s{s} {k}: REF row unread")
                continue
            vh, vr = (h.get("valid") or {}).get(k) or {}, (r.get("valid") or {}).get(k) or {}
            diffs = []
            if vr.get("ckpt_sha") is None or vr.get("ckpt_sha") != vh.get("ckpt_sha"):
                diffs.append(f"checkpoint {str(vr.get('ckpt_sha'))[:12]} != {str(vh.get('ckpt_sha'))[:12]}")
            if vr.get("manifest_env") != vh.get("manifest_env"):
                diffs.append("row env block differs")
            a, b = _inert_read(h, k), _inert_read(r, k)
            if a is None or b is None:
                diffs.append(f"read unreadable (head {a is not None}, REF {b is not None})")
            else:
                diffs += _diff(a, b, k)
            out[s][k] = "equal" if not diffs else diffs
            fails += [f"s{s} {k}: {d}" for d in diffs]
    return {"ok": not fails, "fails": fails, "rows": out}


def _inert_read(pt: dict, k: str):
    row, _, seed = k.partition("@")
    if row == "lr":
        if pt.get("hands") is None:
            return None
        return {"hands": pt.get("hands"), "fall": pt.get("fall"), "feet": pt.get("feet"), "feet_side": pt.get("feet_side")}
    if row == FT:
        return pt.get("field_turn")
    if row == FS:
        return pt.get("field_suite")
    key = next(c for c, r in COMP_ROWS.items() if r == row)
    return ((pt.get("comp") or {}).get(key) or {}).get(int(seed))


def _diff(a, b, path: str) -> list[str]:
    if isinstance(a, dict) and isinstance(b, dict):
        out = []
        for k in sorted(set(a) | set(b), key=str):
            out += _diff(a.get(k), b.get(k), f"{path}.{k}")
        return out
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            return [f"{path}: length {len(a)} != {len(b)}"]
        return [d for i, (x, y) in enumerate(zip(a, b)) for d in _diff(x, y, f"{path}[{i}]")]
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        same = (a == b) or (math.isfinite(a) and math.isfinite(b) and abs(a - b) <= INERT_TOL)
        return [] if same else [f"{path}: head {a} != REF {b}"]
    return [] if a == b else [f"{path}: head {a!r} != REF {b!r}"]


def load_p2b(with_tb: bool = True, levers=None) -> dict:
    """REF points (row evals; field_hard at env_seeds 1-4), the S1 p2a heads (their p2a reads), the ANCW anchors and
    every queued p2b head (suite = the arm's driver_stamp; field_hard pooled over env_seeds 1-4)."""
    an = _anchors_yaml()
    res = {"refs": {s: load_point(ref_name(s), with_tb=False, fh_seeds=P2B_FH_SEEDS) for s in SEEDS},
           "ancw": {int(s): load_point(a["name"], with_tb=False) for s, a in an.items()}, "p2a": {}, "heads": {}}
    for s in SEEDS:
        ref = f"P2__{REF_ARM}_s{s}"
        rec = _rec(ref) or {}
        res["p2a"][s] = (load_point(ref, suite_stamp=rec.get("driver_stamp"), run_dir=rec.get("run_dir"), with_tb=False)
                         if rec and not rec.get("killed") else {"name": ref, "missing": True})
    for ref in p2b_head_names(levers):
        rec = _rec(ref)
        if rec and not rec.get("killed"):
            res["heads"][ref] = load_point(ref, suite_stamp=rec.get("driver_stamp"), run_dir=rec.get("run_dir"),
                                           with_tb=with_tb, fh_seeds=P2B_FH_SEEDS)
        else:
            res["heads"][ref] = {"name": ref, "missing": True, "killed": (rec or {}).get("killed")}
    return res


def p2b_verdicts(res: dict, need_dose: bool = True) -> dict:
    refs, ancw, p2a = res["refs"], res["ancw"], res["p2a"]
    out = {}
    for ref, pt in res["heads"].items():
        s = int(ref.rsplit("_s", 1)[1])
        label = "lever + continuation" if s != GATE_SEED and f"__{P2B_CONTROL}_" not in ref else None
        if pt.get("missing"):
            out[ref] = {"name": ref, "verdict": "PENDING", "why": [f"no record ({pt.get('killed') or 'not run'})"],
                        "gain": None, "cost": [], "report": p2b_report({}, refs.get(s) or {}), "label": label}
            continue
        if (p2a.get(s) or {}).get("missing"):
            out[ref] = {"name": ref, "verdict": "PENDING", "why": [f"P2__{REF_ARM}_s{s} (the p2a head) not recorded"],
                        "gain": None, "cost": [], "report": p2b_report(pt, refs.get(s) or {}), "label": label}
            continue
        v = p2b_verdict(pt, refs[s], bands(refs[s], refs[GATE_SEED]), p2a[s], ancw[s], bands(ancw[s], ancw[GATE_SEED]),
                        need_dose)
        out[ref] = dict(v, label=label)
    return out


def p2b_attributions(res: dict, vs: dict, levers=None) -> dict:
    levers = p2b_levers() if levers is None else tuple(levers)
    r18 = res["refs"][GATE_SEED]
    bd = bands(r18, r18)
    c0 = res["heads"].get(f"P2__{P2B_CONTROL}_s{GATE_SEED}")
    return {lv: p2b_attribution(res["heads"].get(f"P2__{lv}_s{GATE_SEED}"), c0, vs.get(f"P2__{lv}_s{GATE_SEED}"), bd)
            for lv in levers}


def p2b_recipes(vs: dict, levers=None) -> dict:
    """CHARTER 6.2 rules per lever over s16 / s17 / s18 (vs REF)."""
    levers = p2b_levers() if levers is None else tuple(levers)
    return {lv: recipe_read({s: vs.get(f"P2__{lv}_s{s}") for s in SEEDS}) for lv in levers}


def p2b_markdown(vs: dict, attr: dict, recipes: dict) -> str:
    L = ["# Omni phase 2, stage p2b -- the pre-registered reading (CHARTER 10 / P2B_PLAN step 6)", "",
         "| head | verdict | gain | field_hard n=400 (REF) | need | fh tracking (REF) | ANCW chain worst | label | why |",
         "|---|---|---|---|---|---|---|---|---|"]
    for ref, v in vs.items():
        rp, g = v.get("report") or {}, v.get("gain_read") or {}
        ch = v.get("ancw_chain") or {}
        worst = max(ch, key=lambda m: ch[m]["total"], default=None)
        chain = "--" if worst is None else f"{worst} {ch[worst]['total']:+.3f}"
        L.append(f"| {ref} | **{v['verdict']}** | {v.get('gain')} | {_f(rp.get('field_hard'))} "
                 f"({_f(rp.get('ref_field_hard'))}) | {_f((g.get('field_hard') or [None] * 3)[2])} | "
                 f"{_f(rp.get('field_hard_track'))} ({_f(rp.get('ref_field_hard_track'))}) | "
                 f"{chain} | "
                 f"{v.get('label') or ''} | {'; '.join(str(x) for x in (v.get('why') or []))[:300]} |")
    L += ["", "**Attribution at s18 (vs C0):**"]
    L += [f"- {lv}: **{a.get('status')}** {json.dumps(a.get('conditions') or a.get('why'), default=str)}"
          for lv, a in attr.items()]
    L += ["", "**Recipe read per lever (CHARTER 6.2 rules, vs REF):**"]
    L += [f"- {lv}: **{r['reading']}** {json.dumps({k: x for k, x in r.items() if k != 'reading'})}"
          for lv, r in recipes.items()]
    return "\n".join(L) + "\n"


# ================================================================================================ 11 p2s (CHARTER 11)
# P2S_PLAN.md "Pre-registered verdicts", pre-registered before any p2s run. Every p2s head is read against its seed's
# REFS_S1 (the S1 p2a head re-evaluated on the p2s code; the stand bands come from its suite es1 vs es2); the PARTIAL
# rung compares with C0R_s18 (the p2b C0 continuation re-evaluated on the p2s code) at s18 and with REFS at s16 / s17
# (no same-seed control there: "lever + continuation"). The stand reads are scripts/p2s_stand.py's, never redefined here.
WS = "walkstop__omni"
P2S_MANIFEST = CAMPAIGN / "scenarios_p2s.yaml"
P2S_WS_SEEDS = (1, 2)
C0R = "C0R_s18"
P2S_C0_SOURCE = "P2__C0_s18"                    # the p2b head C0R re-evaluates (its record: checkpoint, run_dir, stamp)
STILL, PARTIAL, GAMED, NO_EFFECT = "STILL", "PARTIAL", "GAMED", "NO EFFECT"
P2S_STAND_GAIN = (STILL, PARTIAL)
P2S_NEEDED_EXTRA = ("suite@2", f"{FT}@1", f"{WS}@1", f"{WS}@2")
P2S_VALIDITY = P2B_VALIDITY + ("suite@2", f"{WS}@1", f"{WS}@2")
#: the stand bands' floors (P2S_PLAN rung 4: REFS es1 vs es2)
P2S_BAND_FLOOR = {"off_ground_share": 0.02, "steps_per_s": 0.25, "taps_per_s": 0.25, "ks_m_s": 0.10, "onset_s": 0.1}
P2S_PARTIAL_FRAC, P2S_HOVER_PLUS = 0.5, 5.0
P2S_STEPS_LOW = 0.5                              # GAMED's "steps/s <= 0.5" (the STILL steps clause)
P2S_OBST_FH, P2S_OBST_FL, P2S_OBST_RO = 0.05, 0.03, 0.05
P2S_FALL_FLOOR, P2S_ONSET_FLOOR, P2S_HEIGHT_TOL, P2S_FT_FLOOR = 0.03, 0.3, 0.03, 0.05
P2S_CLIP = 1.0                                   # agent.yaml clip_actions (the teacher runner's)
P2S_SIGMA_AT = 1000                              # the "+1000" sigma read: the saved checkpoint nearest fork + 1000
P2S_SIGMA_NEAR = 100
WS_WINDOW_NOTE = ("steps/s and taps/s on the FULL stand hold (fall-trimmed); clearance, off-ground, KS, loaded count and "
                  "body height on the STEADY window (the hold minus its first settle_s 1.0 s); falls on the untrimmed "
                  "hold; onset = stand -> moving hold, 5-step trailing mean reaching 0.5 x |cmd| and holding 0.5 s "
                  "(p2s_stand.episode_onset_latency), censored episodes excluded")
P2S_RECIPE_MAP = {"STILL": "PASS", "COST": "COST", "PARTIAL": "PARTIAL (a gain, not a PASS)",
                  "gain": "the stand outcome is STILL or PARTIAL (read before the HAND LOSS rung, as CHARTER 6.1's gain)",
                  "INVALID EVAL / PENDING": "PENDING"}


@functools.lru_cache(maxsize=1)
def PS():
    """scripts/p2s_stand.py -- the ONE stand-estimator set (P2S_PLAN step 1)."""
    return _load(HERE / "p2s_stand.py", "_p2_p2s_stand")


def refs_name(seed: int) -> str:
    return f"REFS_{REF_ARM}_s{seed}"


def p2s_names(env=None) -> dict:
    """The p2s head names (fresh names after a kill come in through the environment, as scripts/p2s_queue.sh's)."""
    e = os.environ if env is None else env
    dose = e.get("P2S_DOSE", "ST2")
    return {"ST2_s18": e.get("P2S_ST2_S18", "P2__ST2_s18"), "ST1_s18": e.get("P2S_ST1_S18", "P2__ST1_s18"),
            "dose": dose, "rep_s16": e.get("P2S_REP_S16", f"P2__{dose}_s16"),
            "rep_s17": e.get("P2S_REP_S17", f"P2__{dose}_s17")}


def p2s_head_names(env=None) -> list[str]:
    n = p2s_names(env)
    return [n["ST2_s18"], n["ST1_s18"], n["rep_s16"], n["rep_s17"]]


def p2s_dose_of(name: str) -> str | None:
    """P2__ST2_s18 / P2__ST2R1_s18 -> ST2 (the p2_live.p2s_arm convention); else None."""
    import re  # noqa: PLC0415
    m = re.match(r"^P2__(ST[0-9]+)(?:R[0-9]+)?_s\d+$", str(name))
    return m.group(1) if m else None


@functools.lru_cache(maxsize=256)
def p2s_summary(stamp: str | None) -> dict | None:
    """p2s_stand.head_summary (the stand hold of a suite / field / field_turn stamp), cached per stamp."""
    return PS().head_summary(stamp) if stamp else None


def _seed_of_stamp(stamp: str) -> int | None:
    try:
        return int(Path(stamp).parent.name.replace("seed", ""))
    except ValueError:
        return None


@functools.lru_cache(maxsize=64)
def walkstop_read(stamp: str | None) -> dict | None:
    """Every "stand" hold of a walkstop__omni stamp (three per episode) through p2s_stand's per-episode estimators, and
    every stand -> moving-hold onset (p2s_stand.stand_onset_pairs: stand -> lat_l, stand -> back). Pooled medians over
    the reached stand holds, per-stand medians, stand-hold falls / reached holds, onset medians (per pair and pooled;
    the per-episode values are kept so env_seeds can be pooled). The window, episode count and env_seed are stated."""
    if not stamp:
        return None
    S = PS()
    meta = _json(Path(stamp) / "run_meta.json") or {}
    labels, speeds, dt = meta.get("hold_labels") or [], meta.get("hold_speeds") or [], float(meta.get("dt", 0.02))
    stands = [i for i, lab in enumerate(labels) if lab == "stand"]
    if not stands:
        return None
    pairs = S.stand_onset_pairs(labels, speeds) if speeds else []
    per, falls, reached, n_ep, censored = [], 0, 0, 0, 0
    onset = {lab: [] for _s, _m, lab in pairs}
    for _m, _l, _e, d in S._raw_episodes(stamp, lambda e: True):
        n_ep += 1
        for i in stands:
            m = S.episode_stand_metrics(d, i, dt)
            if m["n_full"] or m["fall"]:
                reached += 1
                falls += int(bool(m["fall"]))
                per.append(dict(m, hold_idx=i))
        for _s, mi, lab in pairs:
            v = S.episode_onset_latency(d, dt, mi)
            if v is None:
                censored += 1
            else:
                onset[lab].append(v)
    if not n_ep:
        return None
    keys = ("steps_per_s", "taps_per_s", "clearance_m", "off_ground_share", "ks_m_s", "loaded_count", "body_height_m",
            "t1_per_s", "t2_per_s")

    def med(rows, k):
        vals = [r[k] for r in rows if r.get(k) is not None]
        return round(statistics.median(vals), 5) if vals else None
    vals = [v for lab in onset for v in onset[lab]]
    return {"stamp": stamp, "env_seed": _seed_of_stamp(stamp), "n_episodes": n_ep, "window": WS_WINDOW_NOTE,
            "stand_holds": stands, "onset_pairs": [[s, m, lab] for s, m, lab in pairs],
            "stand_reached": reached, "falls": falls, "hover_holds": sum(1 for r in per if r.get("hover")),
            "pooled": {k: med(per, k) for k in keys},
            "by_stand": {str(i): {k: med([r for r in per if r["hold_idx"] == i], k) for k in keys} for i in stands},
            "onset": {lab: {"median_s": round(statistics.median(v), 4) if v else None, "n": len(v)}
                      for lab, v in onset.items()},
            "onset_values": vals, "onset_median_s": round(statistics.median(vals), 4) if vals else None,
            "onset_censored": censored}


@functools.lru_cache(maxsize=64)
def stand_clip_share(stamp: str | None) -> float | None:
    """Reported only (P2S_PLAN risk 3): the share of stand-hold action samples at the clip (|a| >= clip_actions 1.0)
    over the steady window of the suite's stand episodes."""
    if not stamp:
        return None
    import numpy as np  # noqa: PLC0415

    S = PS()
    meta = _json(Path(stamp) / "run_meta.json") or {}
    dt = float(meta.get("dt", 0.02))
    hit = tot = 0
    for _labels, hold_idx, d in S.iter_stand_episodes(stamp):
        _full, steady = S.hold_masks(d, hold_idx, dt)
        a = np.abs(np.asarray(d["actions"])[steady])
        hit += int((a >= P2S_CLIP - 1e-6).sum())
        tot += int(a.size)
    return hit / tot if tot else None


def p2s_walking_reads(stamp: str | None) -> dict | None:
    """Reported only: walking tripod / steps-per-m / swing apex per translate category, and the turn category's
    steps_per_rad_post (stepping after a turn closes) -- the suite stamp's by_category medians."""
    if not stamp:
        return None
    by = (((_json(Path(stamp) / "scenario_metrics.json") or {}).get("aggregate") or {}).get("suite") or {}).get(
        "by_category") or {}
    if not by:
        return None
    return {"walking": {c: {k: _med(by, c, k) for k in ("tripod", "steps_per_m", "swing_apex")}
                        for c in ("fwd", "side", "back", "diag")},
            "steps_per_rad_post": _med(by, "turn", "steps_per_rad_post")}


def p2s_raw(stamps: dict) -> dict:
    """Every p2s stand read of one point (IO, raws)."""
    st = stamps or {}
    return {"suite": {s: p2s_summary(st.get(f"suite@{s}")) for s in (1, 2)},
            "walkstop": {s: walkstop_read(st.get(f"{WS}@{s}")) for s in P2S_WS_SEEDS},
            "field": {k: p2s_summary(st.get(k)) for k in [f"{FH}@{s}" for s in P2B_FH_SEEDS] + [f"{FL}@1", f"{FL}@2"]},
            "turn": p2s_summary(st.get(f"{FT}@1")),
            "clip_share": {s: stand_clip_share(st.get(f"suite@{s}")) for s in (1, 2)},
            "field_onset": {k: (PS().onset_summary(st[k]) if st.get(k) else None) for k in (f"{FH}@1", f"{FL}@1")}}


def load_p2s_point(name: str, *, suite_stamp: str | None = None, run_dir: str | None = None, with_tb: bool = True,
                   with_raw: bool = True, lrout: Path | None = None) -> dict:
    """:func:`load_point` with field_hard at env_seeds 1-4, plus the walkstop stamps (validity + their env), the
    reported walking reads and (``with_raw``) every p2s stand read."""
    lrout = lrout or LROUT
    pt = load_point(name, suite_stamp=suite_stamp, run_dir=run_dir, lrout=lrout, with_tb=with_tb,
                    fh_seeds=P2B_FH_SEEDS)
    pt["ws_env"] = {}
    for s in P2S_WS_SEEDS:
        stamp = newest_stamp(name, WS, s, lrout)
        pt["stamps"][f"{WS}@{s}"] = stamp
        pt["valid"][f"{WS}@{s}"] = stamp_validity(stamp)
        pt["ws_env"][s] = dict(((_json(Path(stamp) / "run_meta.json") or {}).get("env_vars") or {})) if stamp else None
    pt["walking"] = p2s_walking_reads(pt["stamps"].get("suite@1"))
    if with_raw:
        pt["p2s"] = p2s_raw(pt["stamps"])
    return pt


# ------------------------------------------------------------------------------------------------ p2s pure reads
def walkstop_flat_fails(pt: dict) -> list[str]:
    """INVALID EVAL: a walkstop stamp must be the flat row scenarios_p2s.yaml defines (no row env block, no
    KRABBY_FLAT_TERRAIN_MODE in its env -- MODE=light is what puts obstacles on the ground)."""
    out = []
    for s in P2S_WS_SEEDS:
        k = f"{WS}@{s}"
        v = (pt.get("valid") or {}).get(k)
        if v is None:
            continue
        if v.get("manifest_env"):
            out.append(f"{pt.get('name')} {k}: walkstop carries a row env block {v['manifest_env']} (not the flat row)")
        if "KRABBY_FLAT_TERRAIN_MODE" in ((pt.get("ws_env") or {}).get(s) or {}):
            out.append(f"{pt.get('name')} {k}: walkstop env sets KRABBY_FLAT_TERRAIN_MODE (not flat ground)")
    return out


def p2s_stand_bands(refs: dict) -> dict:
    """P2S_PLAN rung 4: |REFS es1 - es2| per stand metric (the suite stand) and for the walkstop MEDIAN onset, floored
    at :data:`P2S_BAND_FLOOR`. A band that cannot be computed is None (-> PENDING)."""
    p = (refs or {}).get("p2s") or {}
    a, b = (p.get("suite") or {}).get(1), (p.get("suite") or {}).get(2)
    out = {}
    for k, floor in P2S_BAND_FLOOR.items():
        if k == "onset_s":
            w1, w2 = (p.get("walkstop") or {}).get(1), (p.get("walkstop") or {}).get(2)
            x, y = (w1 or {}).get("onset_median_s"), (w2 or {}).get("onset_median_s")
        else:
            x, y = (a or {}).get(k), (b or {}).get(k)
        out[k] = None if x is None or y is None else max(abs(x - y), floor)
    return out


def _clause(v, lim) -> tuple:
    return (v, lim, v is not None and lim is not None and v <= lim + EPS)


def p2s_stand_outcome(arm: dict, cmp: dict, sbd: dict) -> dict:
    """P2S_PLAN rung 4 (pure): ``arm`` / ``cmp`` = {1: head_summary(suite es1), 2: ...(es2)} of the head and of its
    comparator (C0R at s18, REFS at s16 / s17); ``sbd`` = :func:`p2s_stand_bands` of REFS. STILL > PARTIAL > GAMED >
    NO EFFECT; ``outcome`` None when a summary or the KS band is missing (PENDING)."""
    S = PS()
    out = {"clauses": {s: S.still_clauses((arm or {}).get(s)) for s in (1, 2)}, "partial": {}, "note": None}
    miss = [f"{who} suite es{s} stand summary" for who, d in (("head", arm), ("comparator", cmp)) for s in (1, 2)
            if not (d or {}).get(s)]
    if sbd.get("ks_m_s") is None:
        miss.append("REFS KS band (suite es1 vs es2)")
    if miss:
        return dict(out, outcome=None, why=miss)
    still = all(ok for s in (1, 2) for _v, _t, ok in out["clauses"][s].values())
    for s in (1, 2):
        a, c = arm[s], cmp[s]
        out["partial"][s] = {
            "off_ground_share": _clause(a.get("off_ground_share"), None if c.get("off_ground_share") is None
                                        else P2S_PARTIAL_FRAC * c["off_ground_share"]),
            "steps_per_s": _clause(a.get("steps_per_s"), None if c.get("steps_per_s") is None
                                   else P2S_PARTIAL_FRAC * c["steps_per_s"]),
            "hover_episodes_per_100": _clause(a.get("hover_episodes_per_100"),
                                              None if c.get("hover_episodes_per_100") is None
                                              else c["hover_episodes_per_100"] + P2S_HOVER_PLUS),
            "ks_m_s": _clause(a.get("ks_m_s"), None if c.get("ks_m_s") is None else c["ks_m_s"] + sbd["ks_m_s"])}
    partial = all(ok for s in (1, 2) for _v, _t, ok in out["partial"][s].values())
    steps_low = all(arm[s].get("steps_per_s") is not None and arm[s]["steps_per_s"] <= P2S_STEPS_LOW + EPS
                    for s in (1, 2))
    gamed = steps_low and any(not out["partial"][s][k][2] for s in (1, 2)
                              for k in ("off_ground_share", "hover_episodes_per_100", "ks_m_s"))
    outcome = STILL if still else PARTIAL if partial else GAMED if gamed else NO_EFFECT
    if outcome == PARTIAL and all(ok for s in (1, 2) for k, (_v, _t, ok) in out["clauses"][s].items()
                                  if k != "clearance_m"):
        out["note"] = ("PARTIAL from the clearance clause alone: a median-over-feet clause binds before steps/s when "
                       "3+ feet each take one full-height step (P2S_PLAN 'left for implementation')")
    return dict(out, outcome=outcome)


def is_still(pt: dict) -> bool | None:
    """Every STILL clause on both suite seeds (absolute; no comparator) -- None when a summary is missing."""
    su = ((pt or {}).get("p2s") or {}).get("suite") or {}
    if not su.get(1) or not su.get(2):
        return None
    S = PS()
    return all(ok for s in (1, 2) for _v, _t, ok in S.still_clauses(su[s]).values())


def p2s_stand_falls(pt: dict) -> tuple[int, int] | None:
    """Pooled stand-hold falls / stand holds (suite es1 + es2, walkstop es1 + es2, the field rows' opening stands at
    every env_seed read, field_turn's stand). None when any part is missing (PENDING)."""
    p = (pt or {}).get("p2s") or {}
    parts = [(p.get("suite") or {}).get(s) for s in (1, 2)] + list((p.get("field") or {}).values()) + [p.get("turn")]
    if any(x is None for x in parts) or not p.get("field"):
        return None
    falls, n = sum(int(x.get("falls") or 0) for x in parts), sum(int(x.get("n") or 0) for x in parts)
    for s in P2S_WS_SEEDS:
        w = (p.get("walkstop") or {}).get(s)
        if w is None:
            return None
        falls, n = falls + int(w["falls"]), n + int(w["stand_reached"])
    return (falls, n) if n else None


def p2s_onset(pt: dict) -> float | None:
    """The walkstop MEDIAN onset latency, episodes pooled over env_seeds 1 and 2 (P2S_PLAN: the median; p90 is
    noisy)."""
    w = ((pt or {}).get("p2s") or {}).get("walkstop") or {}
    if any(w.get(s) is None for s in P2S_WS_SEEDS):
        return None
    vals = [v for s in P2S_WS_SEEDS for v in (w[s].get("onset_values") or [])]
    return statistics.median(vals) if vals else None


def p2s_body_height(pt: dict) -> float | None:
    su = ((pt or {}).get("p2s") or {}).get("suite") or {}
    v = [(su.get(s) or {}).get("body_height_m") for s in (1, 2)]
    return None if None in v else sum(v) / 2


def p2s_pending(pt: dict, refs: dict, cmp: dict, sbd: dict) -> list[str]:
    miss = []
    for who in (pt, refs):
        miss += [f"{who.get('name')} {k}" for k in P2S_NEEDED_EXTRA if not (who.get("stamps") or {}).get(k)]
        if "p2s" not in who:
            miss.append(f"{who.get('name')} p2s stand reads (raws not read)")
            continue
        miss += [f"{who.get('name')} suite es{s} stand summary" for s in (1, 2)
                 if not (who["p2s"].get("suite") or {}).get(s)]
        miss += [f"{who.get('name')} walkstop es{s} read" for s in P2S_WS_SEEDS
                 if not (who["p2s"].get("walkstop") or {}).get(s)]
        if p2s_stand_falls(who) is None:
            miss.append(f"{who.get('name')} pooled stand-hold falls")
        if p2s_onset(who) is None:
            miss.append(f"{who.get('name')} walkstop onset")
        if (who.get("field_turn") or {}).get("completion") is None:
            miss.append(f"{who.get('name')} field_turn completion")
    if cmp is not refs:
        if "p2s" not in (cmp or {}):
            miss.append(f"{(cmp or {}).get('name')} (comparator) p2s stand reads")
        else:
            miss += [f"{cmp.get('name')} (comparator) suite es{s} stand summary" for s in (1, 2)
                     if not (cmp["p2s"].get("suite") or {}).get(s)]
    miss += [f"stand band {k}" for k, v in sbd.items() if v is None]
    return miss


def p2s_cost(pt: dict, refs: dict, bd: dict, sbd: dict) -> tuple[list[str], list[str], dict]:
    """P2S_PLAN rung 5 (pure): ``(cost metric names, why lines, the reads)``."""
    cost, why = [], []
    deltas = {m: worse_by(pt, refs, m) for m in LISTED}
    for m in LISTED:
        if deltas[m] > max(LISTED[m][2], bd[m]) + EPS:
            cost.append(m)
            why.append(f"{m} worse than REFS by {deltas[m]:.4f} > max({LISTED[m][2]}, {bd[m]:.4f})")
    (fa, na), (fr, nr) = p2s_stand_falls(pt), p2s_stand_falls(refs)
    pa, pr = fa / na, fr / nr
    se = math.sqrt(max(pa * (1 - pa), 0.0) / na + max(pr * (1 - pr), 0.0) / nr)
    need = pr + max(P2S_FALL_FLOOR, 2 * se)
    if pa > need + EPS:
        cost.append("stand_falls")
        why.append(f"pooled stand-hold falls {fa}/{na} = {pa:.4f} > REFS {fr}/{nr} = {pr:.4f} + max({P2S_FALL_FLOOR}, "
                   f"2 SE {2 * se:.4f})")
    oa, orf = p2s_onset(pt), p2s_onset(refs)
    if oa > orf + max(P2S_ONSET_FLOOR, sbd["onset_s"]) + EPS:
        cost.append("onset")
        why.append(f"walkstop median onset {oa:.3f} s > REFS {orf:.3f} + max({P2S_ONSET_FLOOR}, band {sbd['onset_s']:.3f})")
    ha, hr = p2s_body_height(pt), p2s_body_height(refs)
    if ha is not None and hr is not None and abs(ha - hr) > P2S_HEIGHT_TOL + EPS:
        cost.append("stand_body_height")
        why.append(f"suite stand body height {ha:.4f} m is {ha - hr:+.4f} off REFS {hr:.4f} (> {P2S_HEIGHT_TOL})")
    fta, ftr = pt["field_turn"]["completion"], refs["field_turn"]["completion"]
    band_ft = max(binom_se(ftr, int(refs["field_turn"].get("n") or 100)), COMP_BAND_FLOOR)
    if fta < ftr - max(P2S_FT_FLOOR, band_ft) - EPS:
        cost.append("field_turn_completion")
        why.append(f"field_turn completion {fta:.3f} < REFS {ftr:.3f} - max({P2S_FT_FLOOR}, band {band_ft:.3f})")
    reads = {"deltas": deltas, "stand_falls": [fa, na, fr, nr], "onset": [oa, orf], "body_height": [ha, hr],
             "field_turn": [fta, ftr, band_ft]}
    return cost, why, reads


def p2s_report(pt: dict, refs: dict) -> dict:
    """What every p2s verdict shows, whatever rung matched: field_hard (n = 400) beside REFS, the onset delta."""
    r = p2b_report(pt, refs)
    oa, orf = p2s_onset(pt), p2s_onset(refs)
    r.update(onset_s=oa, ref_onset_s=orf, onset_delta_s=(oa - orf) if oa is not None and orf is not None else None)
    return r


def p2s_verdict(pt: dict, refs: dict, cmp: dict, bd: dict, p2a: dict, ancw: dict, bd_ancw: dict,
                need_dose: bool = True) -> dict:
    """P2S_PLAN "Pre-registered verdicts" for one head vs its seed's REFS (pure; first match decides). ``cmp`` = the
    PARTIAL comparator (C0R at s18, ``refs`` itself at s16 / s17); ``bd`` = bands(REFS, REFS s18); ``p2a`` / ``ancw`` /
    ``bd_ancw`` = the CHARTER 10 chained flat floor's links. The stand outcome is ALWAYS computed and shown."""
    sbd = p2s_stand_bands(refs)
    arm_su = ((pt.get("p2s") or {}).get("suite")) or {}
    cmp_su = (((cmp or {}).get("p2s") or {}).get("suite")) or {}
    stand = p2s_stand_outcome(arm_su, cmp_su, sbd)
    out = {"name": pt.get("name"), "dose": p2s_dose_of(pt.get("name") or ""), "comparator": (cmp or {}).get("name"),
           "gain": (stand.get("outcome") in P2S_STAND_GAIN) if stand.get("outcome") else None, "cost": [],
           "stand": stand, "stand_bands": sbd, "report": p2s_report(pt, refs)}
    inv = invalid_reasons(pt, refs, P2S_VALIDITY) + walkstop_flat_fails(pt) + walkstop_flat_fails(refs)
    if cmp is not refs and cmp is not None:
        inv += walkstop_flat_fails(cmp)
    if inv:
        return dict(out, verdict="INVALID EVAL", why=inv)
    pend = p2b_pending(pt, refs, bd, p2a, ancw, bd_ancw, need_dose) + p2s_pending(pt, refs, cmp or {}, sbd)
    if stand.get("outcome") is None:
        pend += [f"stand outcome: {w}" for w in stand.get("why") or []]
    if pend:
        return dict(out, verdict="PENDING", why=pend)
    out["deltas"] = {m: worse_by(pt, refs, m) for m in LISTED}
    out["ancw_chain"] = chained_ancw(pt, refs, p2a, ancw)
    if [h for h in HANDS if pt["hands"][h] != FAST] or pt["fall"] > TRIP + EPS:
        return dict(out, verdict="HAND LOSS", why=[f"hands {pt['hands']}", f"hold-fall share {pt['fall']:.3f}"])
    if need_dose and pt["dose_ok"] is False:
        return dict(out, verdict="DOSE", why=[f"median |mu| {pt.get('dose')} > 1.5"])
    ff = _flat_fails(out["deltas"], bd, refs["name"])
    ff += _flat_fails({m: v["total"] for m, v in out["ancw_chain"].items()}, bd_ancw, f"{ancw['name']} (chained)")
    ff += [f"translate survival {h} {v:.3f} < {TRANSLATE_MIN}" for h, v in pt["suite"]["trans_surv"].items()
           if v < TRANSLATE_MIN - EPS]
    if ff:
        return dict(out, verdict="FLAT FAIL", why=ff)
    (fh, nf), (rfh, nr) = fh400(pt), fh400(refs)
    g = se_need(rfh, nr, fh, nf)
    fl, ro = pooled(pt, "field_light")[0], pooled(pt, "ring_obst")[0]
    rfl, rro = pooled(refs, "field_light")[0], pooled(refs, "ring_obst")[0]
    obst = []
    if fh < rfh - max(P2S_OBST_FH, P2B_SE_K * g["se"]) - EPS:
        obst.append(f"field_hard {fh:.4f} (n={nf}) < REFS {rfh:.4f} - max({P2S_OBST_FH}, 2 SE {2 * g['se']:.4f})")
    if fl < rfl - max(P2S_OBST_FL, bd["field_light"]) - EPS:
        obst.append(f"field_light {fl:.4f} < REFS {rfl:.4f} - max({P2S_OBST_FL}, band {bd['field_light']:.4f})")
    if ro < rro - max(P2S_OBST_RO, bd["ring_obst"]) - EPS:
        obst.append(f"ring_obst {ro:.4f} < REFS {rro:.4f} - max({P2S_OBST_RO}, band {bd['ring_obst']:.4f})")
    out["obstacle"] = {"field_hard": [fh, rfh, g["se"]], "field_light": [fl, rfl], "ring_obst": [ro, rro]}
    if obst:
        return dict(out, verdict="OBSTACLE LOSS", why=obst)
    oc = stand["outcome"]
    if oc in P2S_STAND_GAIN:
        cost, why, reads = p2s_cost(pt, refs, bd, sbd)
        out["cost_reads"] = reads
        if cost:
            return dict(out, verdict="COST", cost=cost, why=[f"stand {oc}, with COST"] + why)
        return dict(out, verdict=oc, why=[f"stand {oc}, no COST"] + ([stand["note"]] if stand.get("note") else []))
    return dict(out, verdict=oc, why=[f"stand {oc}" + (" (a flag: never adopts)" if oc == GAMED else "")])


def p2s_reads_still(v: dict | None) -> bool:
    """The queue's ST1 condition (P2S_PLAN step 6): the head reads STILL, with or without COST."""
    return bool(v) and v.get("verdict") in (STILL, "COST") and (v.get("stand") or {}).get("outcome") == STILL


def p2s_attribution(st: dict | None, c0r: dict | None, st_v: dict | None, bd: dict) -> dict:
    """P2S_PLAN attribution at s18 (pure): the effect belongs to the lever if ST reads STILL or PARTIAL (COST
    included), C0R reads NO EFFECT -- implemented as "C0R is not STILL" (P2S_PLAN 'left for implementation') --, there
    is no FLAT FAIL vs C0R (the CHARTER 6.1 list at max(floor, 2 band), REFS s18 bands), and field_hard (n = 400) >=
    C0R - max(0.05, 2 SE)."""
    out = {"head": (st or {}).get("name"), "control": (c0r or {}).get("name"), "lever": None,
           "head_verdict": (st_v or {}).get("verdict")}
    if not st or st.get("missing") or not st_v or st_v.get("verdict") in ("INVALID EVAL", "PENDING"):
        return dict(out, status="PENDING", why=["the head's verdict is not read"])
    if not c0r or c0r.get("missing"):
        return dict(out, status="PENDING", why=["C0R not read"])
    inv = invalid_reasons(st, c0r, P2S_VALIDITY)
    if inv:
        return dict(out, status="INVALID EVAL", why=inv)
    c0r_still = is_still(c0r)
    miss = [f"{w['name']} field_hard pooled over {P2B_FH_SEEDS}" for w in (st, c0r) if fh400(w) is None]
    miss += [f"{w['name']} suite {m}" for w in (st, c0r) for m in LISTED if (w.get("suite") or {}).get(m) is None]
    miss += [f"band {m}" for m in LISTED if bd.get(m) is None]
    if c0r_still is None:
        miss.append("C0R suite stand summaries")
    if miss:
        return dict(out, status="PENDING", why=miss)
    (fl, nl), (fc, nc) = fh400(st), fh400(c0r)
    g = se_need(fc, nc, fl, nl)
    ff = _flat_fails({m: worse_by(st, c0r, m) for m in LISTED}, bd, c0r["name"])
    conds = {"head reads STILL or PARTIAL (COST included)": st_v.get("verdict") in (STILL, PARTIAL, "COST")
             and (st_v.get("stand") or {}).get("outcome") in P2S_STAND_GAIN,
             "C0R reads NO EFFECT (implemented: C0R is not STILL)": not c0r_still,
             "no FLAT FAIL vs C0R": not ff,
             f"field_hard {fl:.4f} >= C0R {fc:.4f} - max({P2S_OBST_FH}, 2 SE {2 * g['se']:.4f})":
                 fl >= fc - max(P2S_OBST_FH, P2B_SE_K * g["se"]) - EPS}
    lever = all(conds.values())
    return dict(out, lever=lever, status="LEVER" if lever else "NOT ATTRIBUTED", conditions=conds,
                field_hard=[fl, fc], flat_fails_vs_c0r=ff)


def p2s_recipe(vs: dict, names: dict | None = None) -> dict:
    """P2S_PLAN step 7: the CHARTER 6.2 rules (:func:`recipe_read`) on the kept dose at s16 / s17 / s18, with STILL /
    PARTIAL / COST mapped explicitly (:data:`P2S_RECIPE_MAP`)."""
    n = names or p2s_names()
    dose = n["dose"]
    per = {16: n["rep_s16"], 17: n["rep_s17"], 18: n[f"{dose}_s18"] if f"{dose}_s18" in n else f"P2__{dose}_s18"}
    mapped = {}
    for s, ref in per.items():
        v = vs.get(ref)
        if not v:
            mapped[s] = None
            continue
        lab = {STILL: "PASS", "COST": "COST"}.get(v["verdict"], v["verdict"])
        mapped[s] = {"verdict": lab, "gain": v.get("gain"), "cost": list(v.get("cost") or []), "p2s_verdict": v["verdict"],
                     "stand": (v.get("stand") or {}).get("outcome")}
    r = recipe_read(mapped)
    return dict(r, dose=dose, heads=per, mapped={s: m and {k: m[k] for k in ("p2s_verdict", "verdict", "gain", "cost")}
                                                 for s, m in mapped.items()}, mapping=P2S_RECIPE_MAP)


# ------------------------------------------------------------------------------------------------ p2s inertness / IO
P2S_INERT_ROWS = INERT_ROWS + ("suite@1", "suite@2")


def _p2s_inert_read(pt: dict, k: str):
    if k == "suite@1":
        return pt.get("suite")
    if k == "suite@2":
        return pt.get("suite2")
    return _inert_read(pt, k)


def p2s_inertness_check(pairs: dict, rows=P2S_INERT_ROWS) -> dict:
    """P2S_PLAN step 5 (pure): each ``pairs[label] = (new, old)`` -- the p2s-code point (REFS_S1_s<seed> / C0R_s18)
    and the p2b point of the same checkpoint (REF_S1_s<seed> / P2__C0_s18) -- reads EXACTLY the same on every row the
    old point has (its run_arm row evals only; a driver suite stamp is never compared), on the same checkpoint sha256
    and row env block. A row the old point has and the new lacks fails (fail closed); any difference = STOP."""
    fails, out = [], {}
    for lab, (new, old) in pairs.items():
        if not new or not old or old.get("missing"):
            fails.append(f"{lab}: {'p2s point' if not new else 'p2b point'} missing")
            continue
        out[lab] = {}
        for k in rows:
            hs, rs = (old.get("stamps") or {}).get(k), (new.get("stamps") or {}).get(k)
            if not hs:
                out[lab][k] = "p2b point lacks the row (not compared)"
                continue
            if not rs:
                out[lab][k] = "p2s row unread"
                fails.append(f"{lab} {k}: p2s row unread")
                continue
            vo, vn = (old.get("valid") or {}).get(k) or {}, (new.get("valid") or {}).get(k) or {}
            diffs = []
            if vn.get("ckpt_sha") is None or vn.get("ckpt_sha") != vo.get("ckpt_sha"):
                diffs.append(f"checkpoint {str(vn.get('ckpt_sha'))[:12]} != {str(vo.get('ckpt_sha'))[:12]}")
            if vn.get("manifest_env") != vo.get("manifest_env"):
                diffs.append("row env block differs")
            a, b = _p2s_inert_read(old, k), _p2s_inert_read(new, k)
            if a is None or b is None:
                diffs.append(f"read unreadable (p2b {a is not None}, p2s {b is not None})")
            else:
                diffs += _diff(a, b, k)
            out[lab][k] = "equal" if not diffs else diffs
            fails += [f"{lab} {k}: {d}" for d in diffs]
    return {"ok": not fails, "fails": fails, "rows": out}


def load_p2s_inertness(seeds) -> dict:
    """The step-5 pairs: REFS_S1_s<seed> vs REF_S1_s<seed> per seed, and C0R_s18 vs P2__C0_s18 when 18 is read."""
    pairs = {}
    for s in seeds:
        pairs[f"s{s} {refs_name(s)} vs {ref_name(s)}"] = (
            load_point(refs_name(s), with_tb=False, fh_seeds=P2B_FH_SEEDS),
            load_point(ref_name(s), with_tb=False, fh_seeds=P2B_FH_SEEDS))
    if GATE_SEED in seeds:
        old = load_point(P2S_C0_SOURCE, with_tb=False, fh_seeds=P2B_FH_SEEDS) if _rec(P2S_C0_SOURCE) else \
            {"name": P2S_C0_SOURCE, "missing": True}
        pairs[f"s{GATE_SEED} {C0R} vs {P2S_C0_SOURCE}"] = (load_point(C0R, with_tb=False, fh_seeds=P2B_FH_SEEDS), old)
    return pairs


def _sigma_of(ckpt: Path | None, cams: list[bool]) -> dict | None:
    """Reported only: mean sigma over the cam / rod action dims of one checkpoint (the soft / log / scalar std param)."""
    if not ckpt or not Path(ckpt).is_file():
        return None
    try:
        import numpy as np  # noqa: PLC0415
        import torch  # noqa: PLC0415
        sd = torch.load(str(ckpt), map_location="cpu", weights_only=False).get("model_state_dict") or {}
    except Exception:  # noqa: BLE001 -- reported only; an unreadable checkpoint reads None
        return None
    raw = {str(k).rsplit(".", 1)[-1]: v.detach().cpu().numpy().astype("float64").reshape(-1) for k, v in sd.items()
           if str(k).rsplit(".", 1)[-1] in ("std_raw", "log_std", "std")}
    if "std_raw" in raw:
        lo, hi = math.log(0.05), math.log(2.0)
        sig = np.exp(lo + 0.5 * (hi - lo) * (np.tanh(raw["std_raw"]) + 1.0))
    elif "log_std" in raw:
        sig = np.exp(raw["log_std"])
    elif "std" in raw:
        sig = raw["std"]
    else:
        return None
    if len(cams) != len(sig):
        return {"all": float(np.mean(sig))}
    cam = [x for x, c in zip(sig, cams) if c]
    rod = [x for x, c in zip(sig, cams) if not c]
    return {"cam": float(np.mean(cam)) if cam else None, "rod": float(np.mean(rod)) if rod else None}


def p2s_sigma_report(fork: Path | None, run_dir: str | None, head: str | None, stamp: str | None) -> dict:
    """Reported only (P2S_PLAN): sigma per cam and rod at the fork, at the saved checkpoint nearest fork + 1000 (within
    100 its) and at the end (the head)."""
    names = (_json(Path(stamp) / "run_meta.json") or {}).get("action_joint_names") or [] if stamp else []
    cams = ["CamShaft" in n for n in names]
    out = {"fork": _sigma_of(fork, cams), "+1000": None, "end": _sigma_of(Path(head) if head else None, cams)}
    try:
        fork_n = int(Path(fork).stem.split("_")[1]) if fork else None
    except (IndexError, ValueError):
        fork_n = None
    if run_dir and fork_n is not None and Path(run_dir).is_dir():
        best = None
        for p in Path(run_dir).glob("model_*.pt"):
            try:
                it = int(p.stem.split("_")[1])
            except (IndexError, ValueError):
                continue
            dd = abs(it - (fork_n + P2S_SIGMA_AT))
            if dd <= P2S_SIGMA_NEAR and (best is None or dd < best[0]):
                best = (dd, p)
        if best:
            out["+1000"] = dict(_sigma_of(best[1], cams) or {}, checkpoint=best[1].name)
    return out


def _chain_fork(ref: str) -> Path | None:
    try:
        import yaml  # noqa: PLC0415
        d = yaml.safe_load(open(CAMPAIGN / "candidates.yaml"))
        chain = ((d.get("references") or {}).get(ref) or {}).get("chain") or ref
        rf = d["chains"][chain]["stages"][0]["resume_from"]
        return R / rf["checkpoint"]
    except (OSError, KeyError, TypeError, ValueError, IndexError):
        return None


def load_p2s(with_tb: bool = True, seeds=SEEDS, names: dict | None = None) -> dict:
    """REFS (row evals; field_hard 1-4; suite and walkstop at 1 and 2), C0R_s18, the S1 p2a heads (their p2a reads),
    the ANCW anchors and every p2s head of ``seeds`` (suite@1 = the arm's driver_stamp)."""
    n = names or p2s_names()
    an = _anchors_yaml()
    # REFS s18 always: its suite es1 / es2 give every seed's flat (suite) bands, as in CHARTER 10
    res = {"names": n, "refs": {s: load_p2s_point(refs_name(s), with_tb=False) for s in sorted(set(seeds) | {GATE_SEED})},
           "ancw": {int(s): load_point(a["name"], with_tb=False) for s, a in an.items()},
           "p2a": {}, "heads": {}, "c0r": None}
    for s in seeds:
        ref = f"P2__{REF_ARM}_s{s}"
        rec = _rec(ref) or {}
        res["p2a"][s] = (load_point(ref, suite_stamp=rec.get("driver_stamp"), run_dir=rec.get("run_dir"), with_tb=False)
                         if rec and not rec.get("killed") else {"name": ref, "missing": True})
    if GATE_SEED in seeds:
        c0 = _rec(P2S_C0_SOURCE) or {}
        res["c0r"] = load_p2s_point(C0R, run_dir=c0.get("run_dir"), with_tb=with_tb)
    for ref in (n["ST2_s18"], n["ST1_s18"], n["rep_s16"], n["rep_s17"]):
        s = int(ref.rsplit("_s", 1)[1])
        if s not in seeds:
            continue
        rec = _rec(ref)
        if rec and not rec.get("killed"):
            pt = load_p2s_point(ref, suite_stamp=rec.get("driver_stamp"), run_dir=rec.get("run_dir"), with_tb=with_tb)
            if with_tb:
                pt["sigma"] = p2s_sigma_report(_chain_fork(ref), rec.get("run_dir"), rec.get("checkpoint"),
                                               rec.get("driver_stamp"))
            res["heads"][ref] = pt
        else:
            res["heads"][ref] = {"name": ref, "missing": True, "killed": (rec or {}).get("killed")}
    return res


def p2s_verdicts(res: dict, need_dose: bool = True) -> dict:
    """Every p2s head (and C0R at s18) vs REFS of its seed; ST heads at s18 take C0R as the PARTIAL comparator."""
    refs, ancw, p2a, c0r = res["refs"], res["ancw"], res["p2a"], res.get("c0r")
    out = {}
    items = list(res["heads"].items())
    if c0r is not None:
        items = [(C0R, c0r)] + items
    for ref, pt in items:
        s = int(ref.rsplit("_s", 1)[1])
        label = "lever + continuation" if s != GATE_SEED else None
        rs = refs.get(s) or {"name": refs_name(s)}
        if pt.get("missing"):
            out[ref] = {"name": ref, "verdict": "PENDING", "why": [f"no record ({pt.get('killed') or 'not run'})"],
                        "gain": None, "cost": [], "stand": {}, "report": {}, "label": label}
            continue
        if (p2a.get(s) or {}).get("missing"):
            out[ref] = {"name": ref, "verdict": "PENDING", "why": [f"P2__{REF_ARM}_s{s} (the p2a head) not recorded"],
                        "gain": None, "cost": [], "stand": {}, "report": {}, "label": label}
            continue
        cmp = c0r if (s == GATE_SEED and ref != C0R and c0r is not None) else rs
        v = p2s_verdict(pt, rs, cmp, bands(rs, refs.get(GATE_SEED) or rs), p2a[s], ancw[s],
                        bands(ancw[s], ancw[GATE_SEED]) if GATE_SEED in ancw else {}, need_dose)
        out[ref] = dict(v, label=label)
    return out


def p2s_c0r_clauses(c0r: dict | None) -> dict:
    """C0R's own STILL-clause values on both suite seeds (shown beside every arm's, P2S_PLAN step-6 note)."""
    su = ((c0r or {}).get("p2s") or {}).get("suite") or {}
    return {s: (PS().still_clauses(su.get(s)) if su.get(s) else None) for s in (1, 2)}


#: Stage p2r's Z arms (CHARTER 12 / P2R_STATUE_PLAN.md step 5): one network learns the statue in place of ST2's
#: stand-slot reward terms, so Z is read with the SAME full p2s read set and the SAME verdict rules (STILL / no
#: OBSTACLE LOSS / no FLAT FAIL / no COST / attribution LEVER against C0R -- CHARTER 12's "Adoption-grade" list
#: restates P2S_PLAN's exactly), plus one EXTRA invalid rule the grafted bit needs: the stand label every stamp
#: was evaluated under must actually have been the latched clock bit, not a silently-dropped knob.
P2R_HOLD_NAME = "P2R_HOLD_s18"


def z_name(seed: int = 18, env=None) -> str:
    """The Z head's name (a fresh relaunch name after a kill comes in through the environment, the p2s_queue.sh
    convention: P2R_Z_S<seed>)."""
    e = os.environ if env is None else env
    return e.get(f"P2R_Z_S{seed}", f"P2__Z1_s{seed}")


def z_clock_invalid_reasons(pt: dict) -> list[str]:
    """The Z-arm INVALID rule (P2R_STATUE_PLAN.md step 5): every stamp's ``run_meta.env_vars`` must carry
    ``KRABBY_OMNI_STAND_OBS=clock`` -- fail closed: a stamp missing the key, or carrying any other value
    (``0`` / ``1`` / unset), is INVALID. Reading a Z head without the grafted bit armed would silently score the
    frozen-S1 behaviour the walking anchor targets, not the statue the arm is meant to test."""
    out = []
    for k, stamp in sorted((pt.get("stamps") or {}).items()):
        if not stamp:
            continue
        env = (_json(Path(stamp) / "run_meta.json") or {}).get("env_vars") or {}
        v = str(env.get("KRABBY_OMNI_STAND_OBS", "")).strip().lower()
        if v != "clock":
            out.append(f"{pt.get('name')} {k}: KRABBY_OMNI_STAND_OBS={env.get('KRABBY_OMNI_STAND_OBS')!r}, "
                       "want 'clock'")
    return out


def _run_candidates_mod():
    """The driver module (experiments/tools/run_candidates.py), loaded by path once: its narrowed stand-bit clock
    check is the single implementation of the raw-bit rule (fable review 2026-10-09)."""
    global _RC_MOD
    try:
        return _RC_MOD
    except NameError:
        pass
    path = Path(__file__).resolve().parents[2] / "tools" / "run_candidates.py"
    spec = importlib.util.spec_from_file_location("_p2r_run_candidates", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_p2r_run_candidates"] = mod
    spec.loader.exec_module(mod)
    _RC_MOD = mod
    return mod


def z_raw_bit_reasons(pt: dict, lag: int = 1) -> tuple[list[str], int]:
    """P2R_STATUE_PLAN.md step 5's INVALID clause: every Z stamp's RAW stand bit (``clock_obs[:, 0]`` = obs column
    15) must equal the stand label of every recorded step (settle/stand -> 1, every other hold incl. settled turns
    -> 0; col 16 identically 0; the one-step command-to-obs lag at hold boundaries allowed; the prologue
    unchecked). Ground truth = ``segment_id`` + ``run_meta.hold_labels`` (+ the suite's settle label), checked by
    the driver's ``_npz_clock_obs_stand_bit`` -- so a plain Z policy eval IS checked (the first version only ran on
    step-1 launcher stamps). Returns ``(fails, n_stamps_checked)``; ``lag`` is fixed at 1 by the driver's rule."""
    rc = _run_candidates_mod()
    fails, checked = [], 0
    for k, stamp in sorted((pt.get("stamps") or {}).items()):
        if not stamp:
            continue
        ok, why = rc._npz_clock_obs_stand_bit(stamp)
        checked += 1
        if not ok:
            fails.append(f"{pt.get('name')} {k}: raw stand bit vs labels -- {why}")
    return fails, checked


def load_p2r_z(seed: int = 18, with_tb: bool = True, env=None, name: str | None = None,
               suite_stamp: str | None = None) -> dict:
    """Z1_s<seed> loaded with the FULL p2s read set against REFS_S1_s<seed> (C0R the PARTIAL comparator) --
    P2R_STATUE_PLAN.md step 5. Reuses :func:`load_p2s` verbatim for refs / ancw / p2a / c0r (identical machinery:
    the Z verdict rules are the SAME as a p2s dose arm's), with the Z head inserted into ``heads`` the same way a
    p2s head would be. ``name`` / ``suite_stamp`` (NOTE(p2r Z2, 2026-10-10)) read another Z head the same way (Z2)
    and override its suite es1 stamp; defaults Z1's name and the record's driver stamp."""
    res = load_p2s(with_tb=with_tb, seeds=[seed])
    name = name or z_name(seed, env)
    rec = _rec(name) or {}
    if rec and not rec.get("killed"):
        pt = load_p2s_point(name, suite_stamp=suite_stamp or rec.get("driver_stamp"), run_dir=rec.get("run_dir"),
                            with_tb=with_tb)
    else:
        pt = {"name": name, "missing": True, "killed": (rec or {}).get("killed")}
    res["heads"] = {name: pt}
    return res


def p2r_z_verdicts(res: dict) -> dict:
    """:func:`p2s_verdicts` (``need_dose=False``: a Z arm has no stand-slot dose to validate) with the p2r
    clock-bit INVALID rule (:func:`z_clock_invalid_reasons`) folded in on top of whatever verdict
    :func:`p2s_verdict` already read -- an INVALID clock bit overrides any other verdict, including a STILL /
    PASS read on numbers that were never actually scored under the grafted bit."""
    out = p2s_verdicts(res, need_dose=False)
    for name, pt in (res.get("heads") or {}).items():
        v = out.get(name)
        if not v or pt.get("missing") or v.get("verdict") == "PENDING":
            continue
        clock_fails = z_clock_invalid_reasons(pt)
        if clock_fails:
            out[name] = dict(v, verdict="INVALID EVAL", why=clock_fails + list(v.get("why") or []))
    return out


def p2r_z_fidelity(z_pt: dict | None, hold_name: str = P2R_HOLD_NAME) -> dict:
    """Z's stand clauses beside P2R_HOLD_s18's (suite and walkstop: steps/s, taps/s -- P2R_STATUE_PLAN.md step 5's
    "fidelity to the hold controller (reported)"). ``P2R_HOLD_s18`` reads p2s-style (``p2s_stand.head_summary`` /
    ``walkstop_read`` act on raws alone, independent of whether the stamp came from the p2r-mode launcher), so it
    is loaded through the SAME :func:`load_p2s_point` as any other head."""
    hold_rec = _rec(hold_name) or {}
    hold_pt = (load_p2s_point(hold_name, suite_stamp=hold_rec.get("driver_stamp"), run_dir=hold_rec.get("run_dir"),
                              with_tb=False) if hold_rec and not hold_rec.get("killed") else None)

    def clauses(pt):
        p = (pt or {}).get("p2s") or {}
        su, ws = p.get("suite") or {}, p.get("walkstop") or {}
        return {"suite": {s: {"steps_per_s": (su.get(s) or {}).get("steps_per_s"),
                              "taps_per_s": (su.get(s) or {}).get("taps_per_s")} for s in (1, 2)},
               "walkstop": {s: {"steps_per_s": ((ws.get(s) or {}).get("pooled") or {}).get("steps_per_s"),
                                "taps_per_s": ((ws.get(s) or {}).get("pooled") or {}).get("taps_per_s")}
                           for s in P2S_WS_SEEDS}}
    return {"z": clauses(z_pt), "hold": clauses(hold_pt), "hold_name": hold_name, "hold_missing": hold_pt is None}


def p2r_z_report(seed: int = 18, with_tb: bool = True, env=None, early_json: dict | None = None,
                 lag: int = 1) -> dict:
    """The whole ``--p2r-z`` read (P2R_STATUE_PLAN.md step 5): verdict + attribution + fidelity + the walking-drift
    D (from ``p2r_early.py``'s own output, when given) + the best-effort raw-bit cross-check."""
    res = load_p2r_z(seed, with_tb=with_tb, env=env)
    name = z_name(seed, env)
    verdicts = p2r_z_verdicts(res)
    v = verdicts.get(name) or {"name": name, "verdict": "PENDING", "why": ["no record"]}
    pt = (res.get("heads") or {}).get(name) or {}
    attr = p2s_attribution(pt if not pt.get("missing") else None, res.get("c0r"), v, bands(res["refs"][seed],
                                                                                          res["refs"][seed]))
    raw_fails, raw_checked = z_raw_bit_reasons(pt) if not pt.get("missing") else ([], 0)
    return {"name": name, "seed": seed, "verdict": v, "attribution": attr,
           "fidelity": p2r_z_fidelity(pt if not pt.get("missing") else None),
           "walking_drift": early_json or {"note": "no --p2r-early-json given"},
           "raw_bit_check": {"fails": raw_fails, "n_checked": raw_checked,
                             "note": "0 checked is expected for a plain policy eval (module docstring)"}}


def p2r_z_markdown(out: dict) -> str:
    v, fid = out["verdict"], out["fidelity"]
    L = [f"# Omni phase 2, stage p2r -- {out['name']} (CHARTER 12 / P2R_STATUE_PLAN.md step 5)", "",
        f"**Verdict: {v.get('verdict')}**", "", "Why:"] + [f"- {w}" for w in v.get("why") or ["(none)"]] + ["",
        f"**Attribution (vs {out['attribution'].get('control')}): {out['attribution'].get('status')}**", "",
        "## Fidelity to the hold controller", "", f"| row | seed | Z steps/s | Z taps/s | {fid['hold_name']} "
        f"steps/s | {fid['hold_name']} taps/s |", "|---|---|---|---|---|---|"]
    for s in (1, 2):
        zz, hh = fid["z"]["suite"][s], fid["hold"]["suite"][s]
        L.append(f"| suite | es{s} | {zz.get('steps_per_s')} | {zz.get('taps_per_s')} | {hh.get('steps_per_s')} | "
                f"{hh.get('taps_per_s')} |")
    for s in P2S_WS_SEEDS:
        zz, hh = fid["z"]["walkstop"][s], fid["hold"]["walkstop"][s]
        L.append(f"| walkstop | es{s} | {zz.get('steps_per_s')} | {zz.get('taps_per_s')} | {hh.get('steps_per_s')} "
                f"| {hh.get('taps_per_s')} |")
    L += ["", "## Walking drift (p2r_early.py)", "", f"```\n{json.dumps(out['walking_drift'], indent=1)}\n```", "",
         "## Raw-bit cross-check (best-effort)", "", f"```\n{json.dumps(out['raw_bit_check'], indent=1)}\n```"]
    return "\n".join(L)


def p2s_reported(pt: dict, c0r: dict | None) -> dict:
    """The reported-only items (P2S_PLAN): sigma, steps_per_rad_post, walking vs C0R, the stand clip share."""
    p = pt.get("p2s") or {}
    return {"sigma": pt.get("sigma"), "steps_per_rad_post": (pt.get("walking") or {}).get("steps_per_rad_post"),
            "c0r_steps_per_rad_post": ((c0r or {}).get("walking") or {}).get("steps_per_rad_post"),
            "walking": (pt.get("walking") or {}).get("walking"),
            "c0r_walking": ((c0r or {}).get("walking") or {}).get("walking"),
            "stand_clip_share": p.get("clip_share"), "field_onset": p.get("field_onset"),
            "walkstop": {s: (w and {k: w.get(k) for k in ("env_seed", "n_episodes", "window", "stand_reached", "falls",
                                                         "hover_holds", "pooled", "onset", "onset_median_s",
                                                         "onset_censored")})
                         for s, w in (p.get("walkstop") or {}).items()}}


def p2s_markdown(vs: dict, attr: dict, recipe: dict | None, c0r_cl: dict) -> str:
    def cl(c):
        if not c:
            return "--"
        return ", ".join(f"{k.split('_')[0]} {_f(v, '{:.3g}')}{'' if ok else '!'}" for k, (v, _t, ok) in c.items())
    L = ["# Omni phase 2, stage p2s -- the pre-registered reading (CHARTER 11 / P2S_PLAN.md)", "",
         "| head | verdict | stand | gain | onset (delta vs REFS) | field_hard n=400 (REFS) | cost | label | why |",
         "|---|---|---|---|---|---|---|---|---|"]
    for ref, v in vs.items():
        rp, st = v.get("report") or {}, v.get("stand") or {}
        on = rp.get("onset_s")
        L.append(f"| {ref} | **{v['verdict']}** | {st.get('outcome')} | {v.get('gain')} | "
                 f"{_f(on)} ({_f(rp.get('onset_delta_s'), '{:+.3f}')}) | {_f(rp.get('field_hard'))} "
                 f"({_f(rp.get('ref_field_hard'))}) | {', '.join(v.get('cost') or []) or '--'} | {v.get('label') or ''} | "
                 f"{'; '.join(str(x) for x in (v.get('why') or []))[:300]} |")
    L += ["", "**STILL clauses (suite stand, value; '!' = fails), each head beside C0R's own:**",
          f"- C0R_s18: es1 {cl(c0r_cl.get(1))} | es2 {cl(c0r_cl.get(2))}"]
    for ref, v in vs.items():
        if ref == C0R:
            continue
        c = (v.get("stand") or {}).get("clauses") or {}
        note = (v.get("stand") or {}).get("note")
        L.append(f"- {ref}: es1 {cl(c.get(1))} | es2 {cl(c.get(2))}" + (f" -- {note}" if note else ""))
    L += ["", "**Attribution at s18 (vs C0R):**"]
    L += [f"- {h}: **{a.get('status')}** {json.dumps(a.get('conditions') or a.get('why'), default=str)}"
          for h, a in attr.items()]
    if recipe:
        L += ["", f"**Recipe read ({recipe.get('dose')}, CHARTER 6.2 rules, STILL -> PASS / PARTIAL -> gain / COST -> "
                  f"COST):** **{recipe['reading']}** "
                  f"{json.dumps({k: x for k, x in recipe.items() if k not in ('reading', 'mapping')}, default=str)}"]
    return "\n".join(L) + "\n"


def p2s_screen(res: dict, vs: dict) -> dict:
    n = res["names"]
    c0r = res.get("c0r")
    heads18 = [h for h in (n["ST2_s18"], n["ST1_s18"]) if h in res["heads"]]
    r18 = res["refs"].get(GATE_SEED) or {}
    bd = bands(r18, r18)
    attr = {h: p2s_attribution(res["heads"].get(h), c0r, vs.get(h), bd) for h in heads18}
    return {"verdicts": {k: v for k, v in vs.items() if k.endswith(f"_s{GATE_SEED}")}, "attribution": attr,
            "c0r_clauses": p2s_c0r_clauses(c0r),
            "reported": {h: p2s_reported(res["heads"][h], c0r) for h in heads18 if not res["heads"][h].get("missing")},
            "st2_reads_still": p2s_reads_still(vs.get(n["ST2_s18"]))}


# ================================================================================================ p2r (CHARTER 12 /
# P2R_PLAN.md step 1 -- a controller-level stand mode; eval only, code_sha does not move). Points: REFS_S1_s18
# (unchanged from p2s), P2R_PROBE_s18 / P2R_HOLD_s18 / P2R_ST2M_s18 (the three step-1 runs). "Tier 1" is what those
# three runs actually feed: suite es1 (+es2 for HOLD / ST2M), walkstop es1+es2, field_hard es1-4 (n=400), field_turn
# es1, field_light es1 ONLY (no es2) -- never ring_obst, lr or a training log (no run_dir exists for an eval-only
# launch). :func:`p2r_tier1_verdict` is therefore its own ladder (not :func:`p2s_verdict` with extra flags): it skips
# HAND LOSS and DOSE outright (there is no lr row and no training log to read them from -- a tier-1 point that lacks
# them is not PENDING, it was never going to have them) and drops ring_obst from OBSTACLE LOSS, comparing field_light
# at MATCHED n (es1 vs REFS es1, never REFS's es1+es2 pooled n=200 -- :func:`pooled` would return None for a tier-1
# point's field_light and crash the comparison, since tier 1 never reads es2 at all).
import re as _p2r_re  # noqa: PLC0415  (kept local, as this file's own numpy imports are, to match its style)

P2R_VARIANT_OF_TAG = {"HOLD": "hold", "ST2M": "st2", "PROBE": "probe"}
P2R_NAME_RE = _p2r_re.compile(r"^P2R_(HOLD|ST2M|PROBE)_s(\d+)$")
#: every stamp a tier-1 run (HOLD / ST2M) can actually produce (P2R_PLAN.md step 1's run table)
P2R_TIER1_STAMPS = ("suite@1", "suite@2", f"{WS}@1", f"{WS}@2") + tuple(f"{FH}@{s}" for s in P2B_FH_SEEDS) + \
    (f"{FT}@1", f"{FL}@1")
#: the three stamps P2R_PROBE_s18 reads (step 1's run table); the launcher-inertness check (--p2r-probe)
P2R_PROBE_STAMPS = ("suite@1", f"{FH}@1", f"{WS}@1")
P2R_NEEDED_TIER1 = ("suite@1", "suite@2", f"{WS}@1", f"{WS}@2", f"{FH}@1", f"{FT}@1", f"{FL}@1")
#: rungs tier 1 cannot feed, reported for the record (not silently dropped) beside every --p2r-step1 verdict
P2R_TIER1_SKIPPED_RUNGS = ("HAND LOSS (no lr row in tier 1)", "DOSE (no training log in tier 1)",
                           "ring_obst (not read in tier 1)", "field_light es2 (not read in tier 1)")
#: the names the GPU harness's +500 sentinel writes drift for (P2R_PLAN.md step 1(probe) / step 3)
P2R_ST2_ITS = {506: "ST2_m20500", 1006: "ST2_m21000", 5000: "ST2_m24993"}
P2R_C0R_NAME = "C0R_m24993"


def p2r_names(env=None) -> dict:
    e = os.environ if env is None else env
    return {"probe": e.get("P2R_PROBE_S18", "P2R_PROBE_s18"), "hold": e.get("P2R_HOLD_S18", "P2R_HOLD_s18"),
            "st2m": e.get("P2R_ST2M_S18", "P2R_ST2M_s18")}


def p2r_name_tag(name: str) -> str | None:
    m = P2R_NAME_RE.match(name or "")
    return m.group(1) if m else None


def p2r_mode_of(stamp: str | None) -> dict | None:
    """One stamp's ``run_meta.p2r_mode`` (the shared-contract record: variant, checkpoint sha, channels, N,
    switches per episode, stand_mode_share, settle_mode_share_by_wave) -- None if the stamp lacks it entirely
    (a pre-p2r stamp, or one the launcher never wrote the key into)."""
    return (_json(Path(stamp) / "run_meta.json") or {}).get("p2r_mode") if stamp else None


def p2r_mode_fails(pt: dict, tag: str | None, ref_sha: str | None, stamps=P2R_TIER1_STAMPS) -> list[str]:
    """P2R_PLAN.md step 1 (i): every stamp this point has (of ``stamps``) must carry ``run_meta.p2r_mode`` with
    ``variant`` matching the head's own tag (HOLD -> hold, ST2M -> st2, PROBE -> probe) and
    ``s1_checkpoint_sha256`` equal to REFS's own checkpoint -- the mode wrapper must be driving the SAME S1
    checkpoint the ladder compares against. This is deliberately NOT the stamp's own top-level
    ``checkpoint_sha256``: for ST2M that field is legitimately ST2's sha (the acting policy), never S1's. Fail
    closed: a stamp with no ``p2r_mode`` at all is INVALID, never silently equal."""
    want = P2R_VARIANT_OF_TAG.get(tag)
    out = []
    for k in stamps:
        stamp = (pt.get("stamps") or {}).get(k)
        if not stamp:
            continue                                      # a missing stamp is PENDING's business, not this rung's
        mode = p2r_mode_of(stamp)
        if not mode:
            out.append(f"{pt.get('name')} {k}: no run_meta.p2r_mode")
            continue
        if mode.get("variant") != want:
            out.append(f"{pt.get('name')} {k}: p2r_mode.variant {mode.get('variant')!r} != {want!r}")
        ck = mode.get("s1_checkpoint_sha256")
        if ref_sha and ck != ref_sha:
            out.append(f"{pt.get('name')} {k}: p2r_mode.s1_checkpoint_sha256 {str(ck)[:12]} != REFS "
                       f"{str(ref_sha)[:12]}")
    return out


def p2r_tier1_pending(pt: dict, refs: dict, sbd: dict) -> list[str]:
    """Only the reads tier 1 actually feeds (:data:`P2R_NEEDED_TIER1` / the suite+walkstop stand raws); never lr,
    ring_obst, field_suite or a dose read -- those are skipped outright (:data:`P2R_TIER1_SKIPPED_RUNGS`), not
    PENDING."""
    miss = []
    for who in (pt, refs):
        miss += [f"{who.get('name')} {k}" for k in P2R_NEEDED_TIER1 if not (who.get("stamps") or {}).get(k)]
        if "p2s" not in who:
            miss.append(f"{who.get('name')} p2s stand reads (raws not read)")
            continue
        miss += [f"{who.get('name')} suite es{s} stand summary" for s in (1, 2)
                 if not (who["p2s"].get("suite") or {}).get(s)]
        miss += [f"{who.get('name')} walkstop es{s} read" for s in P2S_WS_SEEDS
                 if not (who["p2s"].get("walkstop") or {}).get(s)]
        if (who.get("field_turn") or {}).get("completion") is None:
            miss.append(f"{who.get('name')} field_turn completion")
    for who in (pt, refs):
        s = who.get("suite") or {}
        miss += [f"{who.get('name')} suite {m}" for m in LISTED if s.get(m) is None]
    ts = (pt.get("suite") or {}).get("trans_surv") or {}
    miss += [f"{pt.get('name')} survival {h}" for h in TRANSLATE_HOLDS if ts.get(h) is None]
    miss += [f"stand band {k}" for k, v in sbd.items() if v is None]
    return miss


def p2r_tier1_stand_falls(pt: dict) -> tuple[int, int] | None:
    """Pooled stand-hold falls over tier 1's OWN rows only: suite es1+es2, walkstop es1+es2, field_hard's opening
    stand at es1-4, field_light's opening stand at es1 (no es2 in tier 1), field_turn's stand. None (PENDING) when
    a needed part is missing -- :func:`p2s_stand_falls` is not reused here because it requires field_light es2,
    which tier 1 never reads."""
    p = (pt or {}).get("p2s") or {}
    parts = [(p.get("suite") or {}).get(s) for s in (1, 2)]
    parts += [(p.get("field") or {}).get(f"{FH}@{s}") for s in P2B_FH_SEEDS]
    parts += [(p.get("field") or {}).get(f"{FL}@1"), p.get("turn")]
    if any(x is None for x in parts):
        return None
    falls, n = sum(int(x.get("falls") or 0) for x in parts), sum(int(x.get("n") or 0) for x in parts)
    for s in P2S_WS_SEEDS:
        w = (p.get("walkstop") or {}).get(s)
        if w is None:
            return None
        falls, n = falls + int(w["falls"]), n + int(w["stand_reached"])
    return (falls, n) if n else None


def p2r_tier1_verdict(pt: dict, refs: dict, cmp: dict, bd: dict, sbd: dict) -> dict:
    """P2R_PLAN.md step 1's tier-1 ladder for one head (HOLD or ST2M) vs REFS_S1_s18 (first match decides). The
    stand outcome and the (field_hard / field_light) obstacle reads are ALWAYS attached, whatever rung matched
    first, so :func:`p2r_survives_tier1` never has to re-parse ``why`` strings."""
    tag = p2r_name_tag(pt.get("name") or "")
    out = {"name": pt.get("name"), "tag": tag, "comparator": (cmp or {}).get("name"),
           "skipped_rungs": list(P2R_TIER1_SKIPPED_RUNGS)}
    ref_sha = ((refs.get("valid") or {}).get("suite@1") or {}).get("ckpt_sha")
    inv = (invalid_reasons(pt, refs, P2R_TIER1_STAMPS) + p2r_mode_fails(pt, tag, ref_sha) +
           walkstop_flat_fails(pt) + walkstop_flat_fails(refs))
    if inv:
        return dict(out, verdict="INVALID EVAL", why=inv)
    pend = p2r_tier1_pending(pt, refs, sbd)
    if pend:
        return dict(out, verdict="PENDING", why=pend)
    deltas = {m: worse_by(pt, refs, m) for m in LISTED}
    out["deltas"] = deltas
    ff = [f"{m} worse by {deltas[m]:.4f} > max({LISTED[m][1]}, 2x{bd[m]:.4f})" for m in LISTED
          if deltas[m] > max(LISTED[m][1], 2 * bd[m]) + EPS]
    ff += [f"translate survival {h} {v:.3f} < {TRANSLATE_MIN}" for h, v in pt["suite"]["trans_surv"].items()
           if v < TRANSLATE_MIN - EPS]
    (fh, nf), (rfh, nr) = fh400(pt), fh400(refs)
    g = se_need(rfh, nr, fh, nf)
    ok_fh = fh >= rfh - max(P2S_OBST_FH, P2B_SE_K * g["se"]) - EPS
    (fl, nfl), (rfl, nrfl) = pooled_seeds(pt, "field_light", (1,)), pooled_seeds(refs, "field_light", (1,))
    ok_fl = fl >= rfl - max(P2S_OBST_FL, bd["field_light"]) - EPS
    out["obstacle"] = {"field_hard": [fh, rfh, g["se"], ok_fh, nf, nr],
                       "field_light": [fl, rfl, bd["field_light"], ok_fl, nfl, nrfl],
                       "ring_obst": "not read in tier 1"}
    arm_su = ((pt.get("p2s") or {}).get("suite")) or {}
    cmp_su = (((cmp or {}).get("p2s") or {}).get("suite")) or {}
    stand = p2s_stand_outcome(arm_su, cmp_su, sbd)
    out["stand"], out["stand_bands"] = stand, sbd
    if ff:
        return dict(out, verdict="FLAT FAIL", why=ff)
    obst_why = []
    if not ok_fh:
        obst_why.append(f"field_hard {fh:.4f} (n={nf}) < REFS {rfh:.4f} - max({P2S_OBST_FH}, 2 SE {2 * g['se']:.4f})")
    if not ok_fl:
        obst_why.append(f"field_light {fl:.4f} (n={nfl}, matched) < REFS {rfl:.4f} - "
                        f"max({P2S_OBST_FL}, band {bd['field_light']:.4f})")
    if obst_why:
        return dict(out, verdict="OBSTACLE LOSS", why=obst_why)
    oc = stand.get("outcome")
    out["gain"] = (oc in P2S_STAND_GAIN) if oc else None
    if oc is None:
        return dict(out, verdict="PENDING", why=[f"stand outcome: {w}" for w in stand.get("why") or []])
    if oc not in P2S_STAND_GAIN:
        return dict(out, verdict=oc, why=[f"stand {oc}" + (" (a flag: never adopts)" if oc == GAMED else "")])
    falls, rfalls = p2r_tier1_stand_falls(pt), p2r_tier1_stand_falls(refs)
    if falls is None or rfalls is None:
        return dict(out, verdict="PENDING", why=["tier-1 pooled stand-hold falls"])
    (fa, na), (fr, nrr) = falls, rfalls
    pa, pr = fa / na, fr / nrr
    se = math.sqrt(max(pa * (1 - pa), 0.0) / na + max(pr * (1 - pr), 0.0) / nrr)
    need = pr + max(P2S_FALL_FLOOR, 2 * se)
    cost_falls = pa > need + EPS
    out["stand_falls"] = {"arm": [fa, na], "ref": [fr, nrr], "need": need, "cost": cost_falls}
    if cost_falls:
        return dict(out, verdict="COST", cost=["stand_falls"],
                    why=[f"pooled tier-1 stand-hold falls {fa}/{na}={pa:.4f} > REFS {fr}/{nrr}={pr:.4f} + "
                         f"max({P2S_FALL_FLOOR}, 2 SE {2 * se:.4f})"])
    return dict(out, verdict=oc, why=[f"stand {oc}, no tier-1 COST"])


def p2r_survives_tier1(v: dict) -> bool | None:
    """P2R_PLAN.md step 1 (iii): no stand-hold-falls COST, no OBSTACLE LOSS on field_hard (n=400), no FLAT FAIL.
    Read off the verdict's own attached ``obstacle`` / ``cost`` data (never off ``why`` strings) so a verdict that
    matched OBSTACLE LOSS on field_light ALONE (field_hard itself still ok) still reads as surviving tier 1, per
    the plan's literal field_hard-only wording. None (unknown) on INVALID EVAL / PENDING -- fail closed, never
    silently True."""
    verdict = v.get("verdict")
    if verdict in ("INVALID EVAL", "PENDING"):
        return None
    if verdict == "FLAT FAIL":
        return False
    fh_ok = ((v.get("obstacle") or {}).get("field_hard") or [None, None, None, None])[3]
    if fh_ok is False:
        return False
    if verdict == "COST" and "stand_falls" in (v.get("cost") or []):
        return False
    return True


def _p2r_dir_episodes(stamp: str) -> dict:
    return {int(f.stem.split("_")[1]): f for f in (Path(stamp) / "raw").glob("episode_*.npz")}


def p2r_bitwise_diff(refs_stamp: str, probe_stamp: str) -> list[str]:
    """Every raw key the REFS npz has, bitwise-compared episode by episode against the probe's own
    (``np.array_equal`` -- a probe run must be bit-identical, not merely numerically close). The first mismatch
    per episode is kept (P2R_PLAN.md: "report the first mismatch"); empty = bitwise equal on every shared key and
    episode."""
    import numpy as np  # noqa: PLC0415

    fa, fb = _p2r_dir_episodes(refs_stamp), _p2r_dir_episodes(probe_stamp)
    out = []
    for ep in sorted(set(fa) & set(fb)):
        da, db = np.load(fa[ep]), np.load(fb[ep])
        for k in da.files:
            if k not in db.files:
                out.append(f"episode_{ep}: key {k!r} (in REFS) missing from the probe stamp")
                continue
            va, vb = da[k], db[k]
            if va.shape != vb.shape or not np.array_equal(va, vb):
                out.append(f"episode_{ep} {k}: not bitwise equal to REFS (shapes {va.shape} vs {vb.shape})")
    missing = sorted(set(fa) - set(fb))
    if missing:
        out.append(f"episodes in REFS but not the probe stamp: {missing}")
    return out


def p2r_launcher_inertness(refs_stamps: dict, probe_stamps: dict, rows=P2R_PROBE_STAMPS) -> dict:
    """P2R_PLAN.md step 1's launcher-inertness check (--p2r-probe): P2R_PROBE_s18's three stamps must reproduce
    REFS_S1_s18 bitwise, raw actions included."""
    fails = {}
    for k in rows:
        a, b = refs_stamps.get(k), probe_stamps.get(k)
        if not a or not b:
            fails[k] = f"stamp missing (REFS {bool(a)}, probe {bool(b)})"
            continue
        d = p2r_bitwise_diff(a, b)
        if d:
            fails[k] = d[0]
    return {"ok": not fails, "fails": fails}


def p2r_hold_swap_live(refs_suite_stamp: str | None, hold_suite_stamp: str | None) -> dict:
    """P2R_PLAN.md "Left for implementation" -- the HOLD variant's swap-live proofs, read from one stamp's raws
    (the suite stamp, which carries every stand hold the plant visits): (a) applied actions are exactly 0 on
    every ``p2r_stand_mode`` step; (b) stand-mode share is > 0 (the stamp has a stand hold at all); (c) actions
    BEFORE each episode's first stand entry equal REFS bitwise on the same row/seed (same episode index)."""
    import numpy as np  # noqa: PLC0415

    out = {"zero_actions_on_stand": None, "stand_mode_share": None, "pre_entry_matches_refs": None, "fails": []}
    if not hold_suite_stamp:
        return dict(out, fails=["no HOLD suite stamp"])
    files = _p2r_dir_episodes(hold_suite_stamp)
    ref_files = _p2r_dir_episodes(refs_suite_stamp) if refs_suite_stamp else {}
    zero_ok, shares, pre_ok = True, [], True
    for ep, f in sorted(files.items()):
        d = np.load(f)
        if "p2r_stand_mode" not in d.files:
            out["fails"].append(f"episode_{ep}: no p2r_stand_mode in the raws")
            continue
        mode = np.asarray(d["p2r_stand_mode"]).astype(bool)
        shares.append(float(mode.mean()))
        acts = np.asarray(d["actions"])
        if mode.any() and not np.array_equal(acts[mode], np.zeros_like(acts[mode])):
            zero_ok = False
            out["fails"].append(f"episode_{ep}: a nonzero action on a p2r_stand_mode step")
        first_on = int(np.argmax(mode)) if mode.any() else 0
        if first_on and ep in ref_files:
            ra = np.asarray(np.load(ref_files[ep])["actions"])
            n = min(first_on, acts.shape[0], ra.shape[0])
            if n and not np.array_equal(acts[:n], ra[:n]):
                pre_ok = False
                out["fails"].append(f"episode_{ep}: pre-entry actions (steps 0:{n}) differ from REFS")
    out["zero_actions_on_stand"], out["pre_entry_matches_refs"] = zero_ok, pre_ok
    out["stand_mode_share"] = float(np.mean(shares)) if shares else None
    if out["stand_mode_share"] is not None and out["stand_mode_share"] <= 0:
        out["fails"].append("stand-mode share is not > 0 on a stamp with a stand hold")
    return out


def p2r_settle_share(stamp: str | None) -> list | None:
    """``run_meta.p2r_mode.settle_mode_share_by_wave`` (the shared contract: suite stamps only, else None) --
    surfaced verbatim, never recomputed (the GPU harness is the one that knows the suite's category waves)."""
    mode = p2r_mode_of(stamp)
    return (mode or {}).get("settle_mode_share_by_wave") if mode else None


def p2r_hold_reads(stamp: str | None) -> dict | None:
    """P2R_PLAN.md step 1's "Reported per head" (the raw side): stand-mode share per hold label, and -- at every
    handover OUT of stand mode (the riskier direction: a fall right after the controller hands back to the shared
    gait is exactly the "bleed into walking" risk this step probes) -- whether a fall landed within 2 s (100 steps
    at dt = 0.02 s) of it, bucketed by the entry gait phase (how many of the six feet were unloaded, < 50 N, at the
    exit step). Also: falls whose step is itself inside a stand-mode segment. None if the stamp carries no
    ``p2r_stand_mode`` raw at all (a pre-p2r / REFS stamp)."""
    import numpy as np  # noqa: PLC0415

    if not stamp:
        return None
    meta = _json(Path(stamp) / "run_meta.json") or {}
    labels, dt = meta.get("hold_labels") or [], float(meta.get("dt", 0.02))
    handover_steps = max(1, int(round(2.0 / dt)))
    share_by_label: dict = {}
    falls_in_stand = 0
    handover_falls_by_phase: dict = {}
    n_eps = 0
    for _meta, _labels, _e, d in _raw_episodes(stamp, lambda e: True):
        if "p2r_stand_mode" not in d.files:
            continue
        n_eps += 1
        mode = np.asarray(d["p2r_stand_mode"]).astype(bool)
        seg = np.asarray(d["segment_id"])
        cf = np.asarray(d["crab_failure"]).astype(bool)
        force = np.asarray(d["foot_force_norm"]) if "foot_force_norm" in d.files else None
        for i, lab in enumerate(labels):
            m = seg == i
            if m.any():
                share_by_label.setdefault(lab, []).append(float(mode[m].mean()))
        if mode.any() and cf[mode].any():
            falls_in_stand += 1
        exits = np.flatnonzero((~mode[1:]) & mode[:-1]) + 1 if mode.size > 1 else np.array([], dtype=int)
        for t0 in exits:
            phase = int((force[t0] < 50.0).sum()) if force is not None else None
            window = cf[t0:t0 + handover_steps]
            if window.any():
                handover_falls_by_phase[phase] = handover_falls_by_phase.get(phase, 0) + 1
    return {"n_episodes": n_eps,
           "stand_mode_share_by_label": {lab: (sum(v) / len(v) if v else None) for lab, v in share_by_label.items()},
           "falls_in_stand_mode": falls_in_stand,
           "handover_falls_by_entry_phase": handover_falls_by_phase}


def p2r_turn_final_heading(stamp: str | None) -> dict | None:
    """``|heading_err|`` at the LAST step of each turn hold (turn_l90 / turn_r90) of a field_turn__omni stamp --
    median over episodes, per hold label. None if the stamp has neither turn hold (not a field_turn stamp, or not
    read yet)."""
    import numpy as np  # noqa: PLC0415

    if not stamp:
        return None
    meta = _json(Path(stamp) / "run_meta.json") or {}
    labels = meta.get("hold_labels") or []
    out = {}
    for lab in TURN_HOLDS:
        if lab not in labels:
            continue
        idx = labels.index(lab)
        vals = []
        for _m, _l, _e, d in _raw_episodes(stamp, lambda e: True):
            rows = np.flatnonzero(np.asarray(d["segment_id"]) == idx)
            if rows.size:
                vals.append(abs(float(np.asarray(d["heading_err"])[rows[-1]])))
        out[lab] = round(statistics.median(vals), 5) if vals else None
    return out if out else None


def p2r_dmu_series(stamp: str | None, name: str) -> dict | None:
    """One probe stamp's ``p2r_dmu_<name>`` raw, split by :data:`p2r_stand_mode` into non-stand (the step-2
    bit-0 domain: every hold that is not stand or settle, settled turns included) and stand steps. None if the
    stamp lacks either raw key everywhere."""
    import numpy as np  # noqa: PLC0415

    if not stamp:
        return None
    key = f"p2r_dmu_{name}"
    non_stand, stand = [], []
    for _m, _l, _e, d in _raw_episodes(stamp, lambda e: True):
        if key not in d.files or "p2r_stand_mode" not in d.files:
            continue
        dmu = np.asarray(d[key])
        mode = np.asarray(d["p2r_stand_mode"]).astype(bool)
        non_stand.append(dmu[~mode])
        stand.append(dmu[mode])
    if not non_stand and not stand:
        return None
    ns = np.concatenate(non_stand) if non_stand else np.array([])
    st = np.concatenate(stand) if stand else np.array([])
    return {"non_stand_mean": float(ns.mean()) if ns.size else None, "non_stand_n": int(ns.size),
           "stand_mean": float(st.mean()) if st.size else None, "stand_n": int(st.size)}


def _p2r_pool_dmu(stamps: list, name: str) -> dict | None:
    parts = [p for p in (p2r_dmu_series(s, name) for s in stamps if s) if p]
    if not parts:
        return None
    nsn = sum(p["non_stand_n"] for p in parts)
    stn = sum(p["stand_n"] for p in parts)
    ns_vals = [p["non_stand_mean"] * p["non_stand_n"] for p in parts if p["non_stand_n"]]
    st_vals = [p["stand_mean"] * p["stand_n"] for p in parts if p["stand_n"]]
    return {"non_stand": (sum(ns_vals) / nsn) if nsn else None, "n_non_stand": nsn,
           "stand": (sum(st_vals) / stn) if stn else None, "n_stand": stn}


def p2r_drift_premise(probe_stamps: list, drift_json: dict | None = None,
                      its: dict = P2R_ST2_ITS) -> dict:
    """P2R_PLAN.md step 3's drift premise: D(ST2 +506 / +1006 / +5000) and D(C0R +5000), pooled over the probe
    stamps' own raw ``p2r_dmu_*`` series (live-eval-computed, during P2R_PROBE_s18 itself); plus, when
    ``drift_json`` (``p2r_drift.py``'s own JSON output, keyed ``C0_m20500`` / ``C0_m21000``) is given, the
    CPU-scored C0 checkpoints at the matched iterations. ``D_ref`` = D(ST2 +506); **stop** if D(C0 +506) >
    0.5 x D(ST2 +506) (matched iterations) -- the anchor could not tell harmless continuation from harmful drift.
    The +1006 pair is reported, never gates; D(C0R +5000) vs D(ST2 +5000) is context only."""
    st2 = {it: _p2r_pool_dmu(probe_stamps, name) for it, name in its.items()}
    c0r = {5000: _p2r_pool_dmu(probe_stamps, P2R_C0R_NAME)}
    c0_bank = {}
    if drift_json:
        for it, key in ((506, "C0_m20500"), (1006, "C0_m21000")):
            v = (drift_json.get("ckpts") or drift_json).get(key)
            if isinstance(v, dict):
                c0_bank[it] = v.get("D")
            elif v is not None:
                c0_bank[it] = v
    d_ref = (st2.get(506) or {}).get("non_stand")
    d_c0_506 = c0_bank.get(506)
    stop = bool(d_ref is not None and d_c0_506 is not None and d_c0_506 > 0.5 * d_ref + EPS)
    return {"st2": st2, "c0r": c0r, "c0_bank": c0_bank, "d_ref_506": d_ref, "d_c0_506": d_c0_506, "stop": stop,
           "note": "+1006 pair reported only; D(C0R +5000) vs D(ST2 +5000) is context only"}


def load_p2r_step1(names: dict | None = None, lrout: Path | None = None) -> dict:
    """REFS_S1_s18 (p2s reads: suite es1+es2, walkstop es1+es2, field_hard es1-4, field_turn, field_light es1+es2)
    and the tier-1 heads (HOLD, ST2M -- each read the same way, their unused rows simply absent)."""
    lrout = lrout or LROUT
    n = names or p2r_names()
    out = {"names": n, "refs": load_p2s_point(refs_name(GATE_SEED), with_tb=False, lrout=lrout), "heads": {}}
    for key in ("hold", "st2m"):
        nm = n[key]
        rec = _rec(nm)
        if rec and rec.get("killed"):
            out["heads"][nm] = {"name": nm, "missing": True, "killed": rec.get("killed")}
            continue
        # P2R mode heads are --row-eval only (no training, no record.json): read their row stamps by name
        # (fable review 2026-10-09); a head with no stamps at all is missing
        pt = load_p2s_point(nm, suite_stamp=(rec or {}).get("driver_stamp"), run_dir=(rec or {}).get("run_dir"),
                            with_tb=False, lrout=lrout)
        out["heads"][nm] = pt if (pt.get("stamps") or {}) else {"name": nm, "missing": True, "killed": None}
    return out


def p2r_step1_markdown(out: dict) -> str:
    L = ["# Omni phase 2, stage p2r -- step 1 (controller-level stand mode; CHARTER 12 / P2R_PLAN.md)", "",
        "| head | verdict | stand | survives tier 1 | field_hard n (REFS) | why |", "|---|---|---|---|---|---|"]
    for nm, v in out.get("heads", {}).items():
        st = (v.get("stand") or {}).get("outcome")
        obst = v.get("obstacle") or {}
        fh = obst.get("field_hard") or [None, None]
        L.append(f"| {nm} | **{v.get('verdict')}** | {st} | {v.get('survives_tier1')} | "
                f"{_f(fh[0])} ({_f(fh[1])}) | {'; '.join(str(x) for x in (v.get('why') or []))[:300]} |")
    dp = out.get("drift_premise") or {}
    L += ["", f"**Drift premise:** D_ref (ST2 +506) = {_f(dp.get('d_ref_506'))}, D(C0 +506) = "
             f"{_f(dp.get('d_c0_506'))}, **stop = {dp.get('stop')}**"]
    return "\n".join(L) + "\n"


# ================================================================================================ p2r Z2 (CHARTER 12 /
# P2R_Z2_PLAN.md steps 3 and 5). NOTE(p2r Z2, 2026-10-10): Z2 changes only the stand TARGET (the statue prior pulls
# toward the re-stance controller R instead of 0), so every Z1 read stays and the new per-stand estimators
# (p2s_stand.z2_head_summary: six-down / homed / settle / post-settle / lifts, over EVERY walkstop stand) carry the
# new gates. NOTE(p2r Z2 cams-only, 2026-10-10; P2R_Z2_PLAN "Amendment 2026-10-10"): R is now KINEMATIC (no load
# sensing) and Z2 runs with KRABBY_OMNI_STAND_PROXY OFF, so the obs stays Z1's; the six-down share is REPORTED, not
# gated (``reported`` in z2_stand_gates); settle = homed held 0.5 s (p2s_stand). Two reads:
#   --p2r-rst NAME  step 3's PREMISE GATE on a controller-probe stamp set (P2R_RST_s18; S1 + R at stand rows, no
#                   training): the Z2 stand gates except the ladder and the relative settle clause (the probe SETS
#                   that line), plus field_turn / field_hard es1 stand falls and completion no worse than
#                   P2R_HOLD_s18's by more than 2 SE (HOLD's own binomial SE: against HOLD's 0 falls one fall already
#                   stops and asks, the plan's reading), plus stand body height within 0.03 m of REFS_S1_s18 on the
#                   suite and walkstop stands. es1 (the tuning seed) and es2 are reported separately; a row the
#                   stamp set never ran (the residual pass runs es1 only) is listed NOT RUN, never silently passed.
#   --p2r-z2 [NAME] step 5's read: the --p2r-z ladder verbatim (OBSTACLE LOSS / FLAT FAIL / COST, attribution vs
#                   C0R) with the INVALID rule widened (KRABBY_OMNI_STAND_OBS=clock AND KRABBY_OMNI_STAND_PROXY off --
#                   absent or 0 -- in every stamp env, and the raw stand bit folded in), plus the Z2 stand gates incl.
#                   the relative settle line against the probe, and the reported baselines / fidelity / walls / hip
#                   yaw.
STAND_PROXY_VAR, STAND_PROXY_ON = "KRABBY_OMNI_STAND_PROXY", "hold"
#: NOTE(p2r Z2 cams-only, 2026-10-10): the STAND_PROXY values a Z2-family stamp may carry (absent / blank / 0 = off)
STAND_PROXY_OFF = ("", "0")
P2R_RST_VARIANT = "restance"
#: the Z2 stand gates (P2R_Z2_PLAN.md step 5): shares of STANDS, per seed. NOTE(p2r Z2 cams-only, 2026-10-10): the
#: six-down line is REPORTED against Z2_SIXDOWN_STANDS (``reported``), no longer a gate clause.
Z2_SIXDOWN_STANDS, Z2_HOMED_STANDS, Z2_SETTLED_STANDS = 0.90, 0.90, 0.95
Z2_SETTLE_PLUS_S = 1.0                           # median settle <= the probe's median + 1 s (same seed)
Z2_POST = {"post_settle_steps_per_s": 0.5, "post_settle_taps_per_s": 1.0, "post_settle_ks_m_s": 0.5}
Z2_FALLS_FLOOR = 1.0                             # walkstop stand falls <= max(1, Z1's) per 300 stands
Z2_HEIGHT_TOL = 0.03                             # premise: stand body height within 0.03 m of REFS
Z2_FIELD_SE_K = 2.0                              # premise: field rows no worse than HOLD by more than 2 SE
Z2_SEEDS = (1, 2)
Z2_FIELD_ROWS = (f"{FT}@1", f"{FH}@1")
#: the stamps a Z2-family light point resolves (suite es1 via the record's driver stamp when there is one)
Z2_LIGHT_STAMPS = ("suite@1", "suite@2", f"{WS}@1", f"{WS}@2", f"{FT}@1") + tuple(f"{FH}@{s}" for s in P2B_FH_SEEDS)
Z2_REPORT_KEYS = ("n", "six_down_stand_share", "homed_share", "settled_share", "settle_median_s", "lifts_median",
                  "post_settle_steps_per_s", "post_settle_taps_per_s", "post_settle_ks_m_s", "falls", "falls_per_300",
                  "body_height_m", "hip_yaw_mean_deg", "hip_yaw_max_deg")


def z2_names(env=None) -> dict:
    """The Z2-stage names (fresh names after a kill through the environment, the p2s_queue.sh convention)."""
    e = os.environ if env is None else env
    return {"z2": e.get("P2R_Z2_S18", "P2__Z2_s18"), "probe": e.get("P2R_RST_S18", "P2R_RST_s18"),
            "z1": z_name(18, e), "hold": e.get("P2R_HOLD_S18", P2R_HOLD_NAME), "refs": refs_name(GATE_SEED)}


@functools.lru_cache(maxsize=256)
def z2_summary(stamp: str | None) -> dict | None:
    """p2s_stand.z2_head_summary, cached per stamp."""
    return PS().z2_head_summary(stamp) if stamp else None


def z2_reads(stamps: dict) -> dict:
    """The Z2 per-stand summaries of one point's suite and walkstop stamps (IO, raws)."""
    st = stamps or {}
    return {"suite": {s: z2_summary(st.get(f"suite@{s}")) for s in Z2_SEEDS},
            "walkstop": {s: z2_summary(st.get(f"{WS}@{s}")) for s in Z2_SEEDS}}


def _stamp_env(stamp: str | None) -> dict | None:
    return dict((_json(Path(stamp) / "run_meta.json") or {}).get("env_vars") or {}) if stamp else None


def _proxy_off(stamp: str | None) -> bool:
    """NOTE(p2r Z2 cams-only, 2026-10-10): the stamp's env carries KRABBY_OMNI_STAND_PROXY absent / blank / 0."""
    return str((_stamp_env(stamp) or {}).get(STAND_PROXY_VAR, "")).strip().lower() in STAND_PROXY_OFF


def z2_suite_stamp(name: str, lrout: Path | None = None) -> str | None:
    """Suite es1 for a Z2-family head: the record's driver stamp (as Z1's), else the newest complete
    lr/<name>/suite__omni/seed001 row-eval. NOTE(p2r Z2 cams-only, 2026-10-10): the cams-only chain runs with
    KRABBY_OMNI_STAND_PROXY OFF, so the chain's own ladder suite row (which never carries the proxy:
    run_phases.EVAL_CARRY) is the right stamp -- the earlier preference for a proxy-carrying lr row-eval is gone."""
    rec = _rec(name) or {}
    drv = rec.get("driver_stamp") if rec and not rec.get("killed") else None
    return drv or newest_stamp(name, SUITE, 1, lrout or LROUT)


def z2_light_point(name: str, lrout: Path | None = None) -> dict:
    """The stamps a Z2 gate / baseline reads, located the way the p2r reads locate them (suite es1 = the arm
    record's ``driver_stamp`` when there is one -- Z1's lives under gait_eval/omni_phases/.../seed18_X16_p2r_A15pB,
    not lr/ -- every other row the newest complete lr/<name>/<row>/seed00k stamp), with validity, env, the Z2 reads,
    the suite stand summaries (STILL, body height), and the field rows' stand summaries and completions (suite es1:
    :func:`z2_suite_stamp`: the driver stamp, else the lr row-eval). No
    training log, no lr row: a light point never reads HAND LOSS / DOSE."""
    lrout = lrout or LROUT
    st = {}
    for k in Z2_LIGHT_STAMPS:
        row, s = k.rsplit("@", 1)
        st[k] = (z2_suite_stamp(name, lrout) if k == "suite@1"
                 else newest_stamp(name, SUITE if row == "suite" else row, int(s), lrout))
    return {"name": name, "stamps": st, "valid": {k: stamp_validity(v) for k, v in st.items()},
            "env": {k: _stamp_env(v) for k, v in st.items()},
            "ws_env": {s: _stamp_env(st.get(f"{WS}@{s}")) for s in P2S_WS_SEEDS},
            "z2": z2_reads(st), "still": {s: p2s_summary(st.get(f"suite@{s}")) for s in Z2_SEEDS},
            "field": {k: {"stand": p2s_summary(st.get(k)) if k in Z2_FIELD_ROWS else None,
                          "comp": comp_read(st.get(k))}
                      for k in Z2_FIELD_ROWS + tuple(f"{FH}@{s}" for s in P2B_FH_SEEDS[1:])},
            "missing": not any(st.values())}


def _ge(v, lim) -> tuple:
    return (v, lim, v is not None and lim is not None and v >= lim - EPS)


def _le(v, lim) -> tuple:
    return (v, lim, v is not None and lim is not None and v <= lim + EPS)


def z2_stand_gates(z2: dict, still: dict, z1_z2: dict | None, probe_z2: dict | None = None, seeds=Z2_SEEDS,
                   relative: bool = True) -> dict:
    """The Z2 stand gates (P2R_Z2_PLAN.md step 5 + its 2026-10-10 amendment, pure), per seed in ``seeds``: STILL on
    the suite (all six clauses); homed >= 90 % of walkstop stands;
    walkstop stand falls <= max(1, Z1's) per 300 (over all three stands); >= 95 % of walkstop stands settled;
    (``relative``) the walkstop median settle <= the probe's same-seed median + 1 s; post-settle steps <= 0.5/s,
    taps <= 1/s, KS <= 0.5 (fail-closed medians: an unsettled stand fails them). ``z2`` = :func:`z2_reads` of the
    head, ``still`` = {seed: p2s head_summary of suite es<seed>}, ``z1_z2`` / ``probe_z2`` = the same reads of Z1
    and of the probe. A missing summary is PENDING (named in ``pending``), a present summary with a None statistic
    FAILS. NOTE(p2r Z2 cams-only, 2026-10-10): the six-down share (>= 90 % of stands six-down >= 0.90; suite steady
    window, walkstop last 5 s) is REPORTED in ``reported`` (same ``(value, line, ok)`` shape), never a clause.
    ``{"clauses": {name: (value, line, ok)}, "reported": {...}, "pending": [...], "verdict": PASS / FAIL /
    PENDING}``."""
    S = PS()
    cl, rep, pend = {}, {}, []
    for s in seeds:
        su, ws = (z2.get("suite") or {}).get(s), (z2.get("walkstop") or {}).get(s)
        if not still.get(s):
            pend.append(f"suite es{s} stand summary (STILL)")
        for k, (v, t, ok) in S.still_clauses(still.get(s)).items():
            cl[f"STILL {k} suite es{s}"] = (v, t, ok)
        if not su:
            pend.append(f"suite es{s} Z2 read")
        if not ws:
            pend.append(f"walkstop es{s} Z2 read")
        su, ws = su or {}, ws or {}
        rep[f"six-down stands suite es{s} (steady)"] = _ge(su.get("six_down_stand_share"), Z2_SIXDOWN_STANDS)
        rep[f"six-down stands walkstop es{s} (last 5 s)"] = _ge(ws.get("six_down_stand_share"), Z2_SIXDOWN_STANDS)
        cl[f"homed walkstop es{s}"] = _ge(ws.get("homed_share"), Z2_HOMED_STANDS)
        z1w = ((z1_z2 or {}).get("walkstop") or {}).get(s)
        if not z1w:
            pend.append(f"Z1 walkstop es{s} Z2 read (the stand-falls line)")
        line = max(Z2_FALLS_FLOOR, z1w["falls_per_300"]) if z1w else None
        cl[f"walkstop stand falls per 300 es{s} <= max(1, Z1)"] = _le(ws.get("falls_per_300"), line)
        cl[f"settled walkstop es{s}"] = _ge(ws.get("settled_share"), Z2_SETTLED_STANDS)
        if relative:
            pw = ((probe_z2 or {}).get("walkstop") or {}).get(s)
            if not pw or pw.get("settle_median_s") is None:
                pend.append(f"probe walkstop es{s} median settle (the relative settle line)")
            pm = (pw or {}).get("settle_median_s")
            cl[f"median settle walkstop es{s} <= probe + {Z2_SETTLE_PLUS_S} s"] = _le(
                ws.get("settle_median_s"), None if pm is None else pm + Z2_SETTLE_PLUS_S)
        for k, lim in Z2_POST.items():
            cl[f"{k} walkstop es{s}"] = _le(ws.get(k), lim)
    verdict = "PENDING" if pend else ("PASS" if all(ok for _v, _t, ok in cl.values()) else "FAIL")
    return {"clauses": cl, "reported": rep, "pending": pend, "verdict": verdict,
            "failed": [k for k, (_v, _t, ok) in cl.items() if not ok]}


def z2_env_reasons(pt: dict, stamps=None) -> list[str]:
    """The Z2 env INVALID rule (P2R_Z2_PLAN.md "Verification" + the 2026-10-10 amendment): every stamp's
    ``run_meta.env_vars`` must carry ``KRABBY_OMNI_STAND_OBS=clock`` (fail closed: missing / any other value is
    INVALID) and ``KRABBY_OMNI_STAND_PROXY`` OFF (absent / blank / 0). NOTE(p2r Z2 cams-only, 2026-10-10): R reads
    no load signal and Z2's chain runs the proxy off, so the obs stays Z1's; a stamp with the held proxy armed ran
    a different observation (the load-sensing R's regime) and is INVALID for the cams-only reads."""
    out = []
    for k, stamp in sorted((pt.get("stamps") or {}).items()):
        if not stamp or (stamps is not None and k not in stamps):
            continue
        env = (_json(Path(stamp) / "run_meta.json") or {}).get("env_vars") or {}
        if str(env.get("KRABBY_OMNI_STAND_OBS", "")).strip().lower() != "clock":
            out.append(f"{pt.get('name')} {k}: KRABBY_OMNI_STAND_OBS={env.get('KRABBY_OMNI_STAND_OBS')!r}, want 'clock'")
        if str(env.get(STAND_PROXY_VAR, "")).strip().lower() not in STAND_PROXY_OFF:
            out.append(f"{pt.get('name')} {k}: {STAND_PROXY_VAR}={env.get(STAND_PROXY_VAR)!r}, want it off "
                       "(absent / 0: the cams-only chain)")
    return out


def rst_invalid_reasons(pt: dict, refs: dict, raw_bit: bool = True) -> list[str]:
    """INVALID for a controller-probe stamp set (P2R_RST_s18): CHARTER 6.1's stamp rule against REFS (env4, row env
    block, code-state sidecar with the bridged code_sha chain) on every stamp it has, the walkstop flat-row rule,
    :func:`z2_env_reasons`, ``run_meta.p2r_mode.variant == 'restance'`` driving REFS's own S1 checkpoint, and (with
    ``raw_bit``) the raw stand bit against the labels (:func:`z_raw_bit_reasons`)."""
    have = tuple(k for k, v in (pt.get("stamps") or {}).items() if v)
    out = invalid_reasons(pt, refs, have) + walkstop_flat_fails(pt) + z2_env_reasons(pt)
    ref_sha = ((refs.get("valid") or {}).get("suite@1") or {}).get("ckpt_sha")
    for k in have:
        mode = p2r_mode_of(pt["stamps"][k])
        if not mode:
            out.append(f"{pt.get('name')} {k}: no run_meta.p2r_mode")
            continue
        if mode.get("variant") != P2R_RST_VARIANT:
            out.append(f"{pt.get('name')} {k}: p2r_mode.variant {mode.get('variant')!r} != {P2R_RST_VARIANT!r}")
        if ref_sha and mode.get("s1_checkpoint_sha256") != ref_sha:
            out.append(f"{pt.get('name')} {k}: p2r_mode.s1_checkpoint_sha256 "
                       f"{str(mode.get('s1_checkpoint_sha256'))[:12]} != REFS {str(ref_sha)[:12]}")
    if raw_bit:
        out += z_raw_bit_reasons(pt)[0]
    return out


def pooled_diff_se(xa: float, na: int, xh: float, nh: int) -> float:
    """The SE of the DIFFERENCE of two binomial rates under the pooled rate: sqrt(p (1 - p) (1/na + 1/nh)),
    p = (xa + xh) / (na + nh)."""
    if not na or not nh:
        return float("nan")
    p = (xa + xh) / (na + nh)
    return math.sqrt(max(p * (1.0 - p), 0.0) * (1.0 / na + 1.0 / nh))


def z2_field_vs_hold(arm: dict, hold: dict, rows=Z2_FIELD_ROWS) -> dict:
    """Premise clause (pure): on each of ``rows`` the arm's stand-hold fall rate <= HOLD's + 2 SE and its schedule
    completion >= HOLD's - 2 SE, SE = the pooled two-sample SE of the DIFFERENCE (:func:`pooled_diff_se`).

    NOTE(p2r Z2 probe, 2026-10-10, decide-record CHARTER 12.1): the first rule used HOLD's own binomial SE, which is
    exactly 0 at HOLD's 0 field_turn falls -- one fall in 100 then "failed" against 0 in 100, a difference no test
    resolves (pooled 2 SE = 0.020). The SE of the difference is the statistic the 2-SE rule needs. A row the arm never
    ran is ``not_run``; a row HOLD lacks is PENDING."""
    cl, pend, not_run = {}, [], []
    for k in rows:
        if not (arm.get("stamps") or {}).get(k):
            not_run.append(k)
            continue
        a, h = (arm.get("field") or {}).get(k) or {}, (hold.get("field") or {}).get(k) or {}
        if not h.get("stand") or not h.get("comp"):
            pend.append(f"{hold.get('name')} {k} (stand summary / completion)")
            continue
        hs, hc = h["stand"], h["comp"]
        hf = hs["falls"] / hs["n"] if hs.get("n") else None
        as_, ac = a.get("stand"), a.get("comp")
        af = (as_["falls"] / as_["n"]) if as_ and as_.get("n") else None
        fall_line = None
        if hf is not None and af is not None:
            fall_line = hf + Z2_FIELD_SE_K * pooled_diff_se(as_["falls"], as_["n"], hs["falls"], hs["n"])
        cl[f"{k} stand-fall rate <= HOLD + {Z2_FIELD_SE_K:g} SE(diff)"] = _le(af, fall_line)
        comp_line = None
        if (ac or {}).get("rate") is not None and ac.get("n"):
            comp_line = hc["rate"] - Z2_FIELD_SE_K * pooled_diff_se(ac["rate"] * ac["n"], ac["n"],
                                                                    hc["rate"] * hc["n"], hc["n"])
        cl[f"{k} completion >= HOLD - {Z2_FIELD_SE_K:g} SE(diff)"] = _ge((ac or {}).get("rate"), comp_line)
    return {"clauses": cl, "pending": pend, "not_run": not_run}


def z2_height_vs_refs(arm: dict, refs: dict, seeds=Z2_SEEDS) -> dict:
    """Premise clause (pure): stand body height within 0.03 m of REFS_S1_s18's, on the suite stand (p2s
    head_summary) and the walkstop stands (the Z2 per-stand median), same seed."""
    cl, pend, heights = {}, [], {}
    for s in seeds:
        for lab, a, r in (("suite", (arm.get("still") or {}).get(s), (refs.get("still") or {}).get(s)),
                          ("walkstop", ((arm.get("z2") or {}).get("walkstop") or {}).get(s),
                           ((refs.get("z2") or {}).get("walkstop") or {}).get(s))):
            if not r:
                pend.append(f"{refs.get('name')} {lab} es{s} body height")
                continue
            av, rv = (a or {}).get("body_height_m"), r.get("body_height_m")
            d = None if av is None or rv is None else abs(av - rv)
            cl[f"{lab} es{s} |body height - REFS| <= {Z2_HEIGHT_TOL}"] = _le(d, Z2_HEIGHT_TOL)
            heights[f"{lab} es{s}"] = [av, rv]
    return {"clauses": cl, "pending": pend, "heights": heights}


def _seed_ran(pt: dict, s: int) -> str:
    """'all' / 'none' / 'partial' -- whether a stamp set ran suite AND walkstop at env_seed ``s``."""
    have = [bool((pt.get("stamps") or {}).get(k)) for k in (f"suite@{s}", f"{WS}@{s}")]
    return "all" if all(have) else "none" if not any(have) else "partial"


def p2r_rst_gate(probe: dict, hold: dict, refs: dict, z1: dict, raw_bit: bool = True) -> dict:
    """P2R_Z2_PLAN.md step 3's PREMISE GATE on one controller-probe stamp set (pure but for the raw-bit read):
    INVALID EVAL > per seed (the Z2 stand gates without the ladder and the relative settle clause, plus body height
    vs REFS) > the field rows vs HOLD. ``verdict``: INVALID EVAL / FAIL (any evaluated clause) / PENDING (a
    baseline or a half-run seed) / PASS (every row of the full set ran) / PASS ON ROWS RUN (the listed rows were
    not run: the residual pass is es1 only)."""
    out = {"name": probe.get("name"), "hold": hold.get("name"), "refs": refs.get("name"), "z1": z1.get("name")}
    if probe.get("missing"):
        return dict(out, verdict="PENDING", why=[f"{probe.get('name')}: no stamps"])
    inv = rst_invalid_reasons(probe, refs, raw_bit=raw_bit)
    if inv:
        return dict(out, verdict="INVALID EVAL", why=inv)
    seeds, pend, failed, not_run = {}, [], [], []
    for s in Z2_SEEDS:
        ran = _seed_ran(probe, s)
        if ran == "none":
            seeds[s] = {"verdict": "NOT RUN"}
            not_run += [f"suite@{s}", f"{WS}@{s}"]
            continue
        if ran == "partial":
            pend.append(f"es{s}: suite and walkstop must both run")
        g = z2_stand_gates(probe.get("z2") or {}, probe.get("still") or {}, z1.get("z2"), None, seeds=(s,),
                           relative=False)
        h = z2_height_vs_refs(probe, refs, seeds=(s,))
        cl = dict(g["clauses"], **h["clauses"])
        p = g["pending"] + h["pending"]
        bad = [k for k, (_v, _t, ok) in cl.items() if not ok]
        seeds[s] = {"clauses": cl, "reported": g.get("reported") or {}, "pending": p, "failed": bad,
                    "body_height_m": h["heights"],
                    "verdict": "FAIL" if bad and not p else "PENDING" if p else "PASS",
                    "role": "tuning seed" if s == 1 else "held-out seed"}
        pend += [f"es{s}: {x}" for x in p]
        failed += [f"es{s}: {x}" for x in bad]
    fv = z2_field_vs_hold(probe, hold)
    not_run += fv["not_run"]
    pend += fv["pending"]
    failed += [k for k, (_v, _t, ok) in fv["clauses"].items() if not ok]
    out.update(seeds=seeds, field=fv, not_run=not_run, pending=pend, failed=failed)
    if seeds.get(1, {}).get("verdict") == "NOT RUN":
        pend.append("es1 (the tuning seed) not run")
    if failed and not pend:
        return dict(out, verdict="FAIL", why=failed)
    if pend:
        return dict(out, verdict="PENDING", why=pend + ([f"(also failing: {failed})"] if failed else []))
    return dict(out, verdict="PASS" if not not_run else "PASS ON ROWS RUN",
                why=["every evaluated clause holds"] + ([f"NOT RUN: {not_run}"] if not_run else []))


def p2r_rst_report(name: str | None = None, env=None, raw_bit: bool = True) -> dict:
    """``--p2r-rst NAME``: :func:`p2r_rst_gate` on the probe stamp set, plus the Z1 / HOLD / REFS baselines on
    every new statistic."""
    n = z2_names(env)
    probe = z2_light_point(name or n["probe"])
    refs, hold, z1 = z2_light_point(n["refs"]), z2_light_point(n["hold"]), z2_light_point(n["z1"])
    gate = p2r_rst_gate(probe, hold, refs, z1, raw_bit=raw_bit)
    return {"gate": gate, "probe": _z2_brief(probe),
            "baselines": {lab: _z2_brief(p) for lab, p in (("Z1", z1), ("HOLD", hold), ("REFS", refs))}}


def _z2_brief(pt: dict | None) -> dict:
    """The new statistics of one point, per row and seed (the reported baselines)."""
    z = (pt or {}).get("z2") or {}
    return {"name": (pt or {}).get("name"),
            **{row: {s: ({k: (z.get(row) or {}).get(s, {}).get(k) for k in Z2_REPORT_KEYS}
                         if (z.get(row) or {}).get(s) else None) for s in Z2_SEEDS} for row in ("suite", "walkstop")}}


def z2_invalid_reasons(pt: dict) -> list[str]:
    """The Z2 head's INVALID additions over :func:`p2r_z_verdicts` (which already requires STAND_OBS=clock):
    STAND_PROXY off on every stamp (the cams-only chain, 2026-10-10), and the raw stand bit folded into the verdict
    (P2R_Z2_PLAN "Verification":
    the raw bit must match the labels)."""
    return [r for r in z2_env_reasons(pt) if STAND_PROXY_VAR in r] + z_raw_bit_reasons(pt)[0]


def p2r_z2_verdicts(res: dict) -> dict:
    """:func:`p2r_z_verdicts` (the Z1 ladder verbatim) + :func:`z2_invalid_reasons`, which overrides any verdict but
    PENDING (as the clock rule does)."""
    out = p2r_z_verdicts(res)
    for name, pt in (res.get("heads") or {}).items():
        v = out.get(name)
        if not v or pt.get("missing") or v.get("verdict") == "PENDING":
            continue
        inv = z2_invalid_reasons(pt)
        if inv:
            out[name] = dict(v, verdict="INVALID EVAL", why=inv + list(v.get("why") or []))
    return out


def fh400_light(pt: dict) -> tuple[float, int] | None:
    """field_hard pooled over env_seeds 1-4 from a light point's completions (n = 400)."""
    got = [((pt.get("field") or {}).get(f"{FH}@{s}") or {}).get("comp") for s in P2B_FH_SEEDS]
    if any(not g or not g.get("n") for g in got):
        return None
    n = sum(g["n"] for g in got)
    return sum(g["rate"] * g["n"] for g in got) / n, n


def p2r_z2_report(seed: int = 18, with_tb: bool = True, env=None, name: str | None = None,
                  probe_name: str | None = None) -> dict:
    """The whole ``--p2r-z2`` read (P2R_Z2_PLAN.md step 5): the Z1 ladder + attribution vs C0R (verbatim), the
    widened INVALID rule, the Z2 stand gates (relative settle vs the probe), ``adoption_grade`` (ladder STILL, no
    COST, LEVER, every Z2 gate), and the reported baselines (Z1 / HOLD / REFS on every new statistic), fidelity to
    the probe, walls vs Z1 (n = 400) and stopped hip yaw."""
    n = z2_names(env)
    name = name or n["z2"]
    res = load_p2r_z(seed, with_tb=with_tb, env=env, name=name, suite_stamp=z2_suite_stamp(name))
    pt = (res.get("heads") or {}).get(name) or {}
    if not pt.get("missing"):
        pt["z2"] = z2_reads(pt.get("stamps") or {})
    verdicts = p2r_z2_verdicts(res)
    v = verdicts.get(name) or {"name": name, "verdict": "PENDING", "why": ["no record"]}
    attr = p2s_attribution(pt if not pt.get("missing") else None, res.get("c0r"), v,
                           bands(res["refs"][seed], res["refs"][seed]))
    probe = z2_light_point(probe_name or n["probe"])
    z1, hold, refs = z2_light_point(n["z1"]), z2_light_point(n["hold"]), z2_light_point(n["refs"])
    still = (((pt.get("p2s") or {}).get("suite")) or {}) if not pt.get("missing") else {}
    gates = (z2_stand_gates(pt.get("z2") or {}, still, z1.get("z2"), probe.get("z2"), relative=True)
             if not pt.get("missing") else {"verdict": "PENDING", "clauses": {}, "pending": ["no Z2 head"]})
    ladder_ok = v.get("verdict") == STILL and (v.get("stand") or {}).get("outcome") == STILL
    adoption = bool(ladder_ok and attr.get("status") == "LEVER" and gates.get("verdict") == "PASS")
    fid = {}
    for row in ("suite", "walkstop"):
        for s in Z2_SEEDS:
            a = ((pt.get("z2") or {}).get(row) or {}).get(s) or {}
            b = ((probe.get("z2") or {}).get(row) or {}).get(s) or {}
            fid[f"{row} es{s}"] = {k: [a.get(k), b.get(k)] for k in ("settle_median_s", "settled_share",
                                                                     "lifts_median", "six_down_stand_share")}
    w_z2 = fh400(pt) if not pt.get("missing") else None
    w_z1 = fh400_light(z1)
    walls = {"z2": w_z2, "z1": w_z1}
    if w_z2 and w_z1:
        g = se_need(w_z1[0], w_z1[1], w_z2[0], w_z2[1])
        walls.update(delta=w_z2[0] - w_z1[0], se=g["se"])
    yaw = {lab: {f"walkstop es{s}": {k: (((p.get("z2") or {}).get("walkstop") or {}).get(s) or {}).get(k)
                                     for k in ("hip_yaw_mean_deg", "hip_yaw_max_deg")} for s in Z2_SEEDS}
           for lab, p in (("Z2", pt), ("probe", probe), ("Z1", z1), ("HOLD", hold), ("REFS", refs))}
    raw_fails, raw_checked = z_raw_bit_reasons(pt) if not pt.get("missing") else ([], 0)
    return {"name": name, "seed": seed, "probe": probe.get("name"), "verdict": v, "attribution": attr,
            "z2_gates": gates, "adoption_grade": adoption,
            "adoption_conditions": {"ladder STILL, no COST": ladder_ok, "attribution LEVER vs C0R":
                                    attr.get("status") == "LEVER", "every Z2 gate": gates.get("verdict") == "PASS"},
            "baselines": {lab: _z2_brief(p) for lab, p in (("Z2", pt), ("probe", probe), ("Z1", z1), ("HOLD", hold),
                                                          ("REFS", refs))},
            "fidelity_to_probe": {"note": "[Z2, probe] per statistic", **fid}, "walls_vs_z1": walls,
            "stopped_hip_yaw_deg": yaw,
            "raw_bit_check": {"fails": raw_fails, "n_checked": raw_checked, "note": "folded into INVALID for Z2"}}


def _clause_lines(clauses: dict) -> list[str]:
    return [f"| {k} | {_f(v, '{:.4g}')} | {_f(t, '{:.4g}')} | {'ok' if ok else '**FAIL**'} |"
            for k, (v, t, ok) in clauses.items()]


def _reported_lines(reported: dict) -> list[str]:
    """NOTE(p2r Z2 cams-only, 2026-10-10): the reported (ungated) statistics, marked at / below their line."""
    return [f"| {k} | {_f(v, '{:.4g}')} | {_f(t, '{:.4g}')} | {'at line' if ok else 'below line'} |"
            for k, (v, t, ok) in reported.items()]


def p2r_rst_markdown(out: dict) -> str:
    g = out["gate"]
    L = [f"# Omni phase 2, stage p2r -- Z2 premise gate on {g.get('name')} (P2R_Z2_PLAN.md step 3)", "",
         f"**Verdict: {g.get('verdict')}**", "", "Why:"] + [f"- {w}" for w in g.get("why") or ["(none)"]]
    for s, sv in (g.get("seeds") or {}).items():
        L += ["", f"## es{s} ({sv.get('role', '')}): {sv.get('verdict')}", ""]
        if sv.get("clauses"):
            L += ["| clause | value | line | |", "|---|---|---|---|"] + _clause_lines(sv["clauses"])
        if sv.get("reported"):
            L += ["", "Reported, not gated (cams-only amendment 2026-10-10):", "",
                  "| statistic | value | line | |", "|---|---|---|---|"] + _reported_lines(sv["reported"])
    fv = g.get("field") or {}
    if fv.get("clauses"):
        L += ["", "## Field rows vs HOLD", "", "| clause | value | line | |", "|---|---|---|---|"]
        L += _clause_lines(fv["clauses"])
    L += ["", "## Baselines (new statistics)", "", f"```\n{json.dumps(out.get('baselines'), indent=1)}\n```"]
    return "\n".join(L) + "\n"


def p2r_z2_markdown(out: dict) -> str:
    v, g = out["verdict"], out["z2_gates"]
    L = [f"# Omni phase 2, stage p2r -- {out['name']} (CHARTER 12 / P2R_Z2_PLAN.md step 5)", "",
         f"**Adoption-grade: {out['adoption_grade']}** {json.dumps(out['adoption_conditions'])}", "",
         f"**Ladder verdict: {v.get('verdict')}**", "", "Why:"] + [f"- {w}" for w in v.get("why") or ["(none)"]]
    L += ["", f"**Attribution (vs {out['attribution'].get('control')}): {out['attribution'].get('status')}**", "",
          f"## Z2 stand gates: {g.get('verdict')}", ""]
    if g.get("pending"):
        L += [f"- PENDING: {p}" for p in g["pending"]] + [""]
    L += ["| clause | value | line | |", "|---|---|---|---|"] + _clause_lines(g.get("clauses") or {})
    if g.get("reported"):
        L += ["", "Reported, not gated (cams-only amendment 2026-10-10):", "",
              "| statistic | value | line | |", "|---|---|---|---|"] + _reported_lines(g["reported"])
    rep = {k: out[k] for k in ("baselines", "fidelity_to_probe", "walls_vs_z1", "stopped_hip_yaw_deg", "raw_bit_check")}
    L += ["", "## Baselines, fidelity, walls, hip yaw", "", f"```\n{json.dumps(rep, indent=1, default=str)}\n```"]
    return "\n".join(L) + "\n"


# ================================================================================================ self-test
def _sc():
    return ("c" * 64, "e" * 64, False)


def _valid(manifest_env=None, sidecar="default", env4=True):
    return {"env4": env4, "env4_missing": [] if env4 else ["KRABBY_OMNI_CLOCK_HIDE"],
            "manifest_env": manifest_env or {}, "sidecar": _sc() if sidecar == "default" else sidecar}


def _pt(name, fh=(0.50, 0.50), fl=(0.80, 0.80), ro=(0.60, 0.60), err=0.03, fc=0.80, yp=0.10, mt=0.90, jit=0.20,
        hands=("FAST", "FAST"), fall=0.0, dose=True, trans=1.0, feet=0.23, track=0.80, s2=None, **kw):
    def c(r):
        return {"rate": r, "n": 100, "falls": int(round((1 - r) * 100)), "track": track}
    suite = {"err_fwd": err, "err_back": err, "err_side": err, "err_diag": err, "turn_fc": fc, "turn_yp": yp,
             "mt_surv": mt, "trans_surv": {h: trans for h in TRANSLATE_HOLDS}, "jitter": jit,
             "stand": {"tripod": kw.pop("tripod", 0.79), "steps_per_s": 3.2, "six_down_frac": 0.002}}
    pt = {"name": name, "stamps": {k: "/x" for k in NEEDED_STAMPS},
          "valid": {k: _valid() for k in NEEDED_STAMPS},
          "comp": {"field_hard": {1: c(fh[0]), 2: c(fh[1])}, "field_light": {1: c(fl[0]), 2: c(fl[1])},
                   "ring_obst": {1: c(ro[0]), 2: c(ro[1])}},
          "suite": suite, "suite2": s2, "hands": {"l": hands[0], "r": hands[1]}, "fall": fall, "feet_side": feet,
          "dose_ok": dose, "field_suite": {"mt_surv": mt}}
    pt.update(kw)
    return pt


def _self_test() -> int:
    ok = True

    def check(name, got, want):
        nonlocal ok
        good = got == want
        ok &= good
        print(f"  [{'ok' if good else 'FAIL'}] {name}: {got}")

    s2 = {"err_fwd": 0.03, "err_back": 0.03, "err_side": 0.03, "err_diag": 0.03, "turn_fc": 0.80, "turn_yp": 0.10,
          "mt_surv": 0.90}
    anc = _pt("ANC_s18", s2=s2)
    bd = bands(anc, anc)
    check("bands: identical seeds take the floors (completion = max(SE, 0.04))",
          (round(bd["field_hard"], 4), bd["err_fwd"], bd["turn_yp"], round(bd["mt_surv"], 4)),
          (0.04, 0.02, 0.005, 0.04))
    wide = _pt("ANC_s16", fh=(0.40, 0.60))
    check("bands: |seed1 - seed2| 0.20 sets the field_hard band", round(bands(wide, anc)["field_hard"], 3), 0.2)
    check("bands: the binomial SE floors a near-0.5 pooled p at n=200 (0.0354 < 0.04 -> 0.04)",
          round(max(binom_se(0.5, 200), 0.04), 4), 0.04)
    check("bands: a missing seed-2 suite -> None (PENDING)", bands(anc, _pt("x"))["err_fwd"], None)

    good = _pt("P2__A1_s18", fh=(0.62, 0.62))
    check("ladder: +0.12 field_hard, no loss -> PASS", seed_verdict(good, anc, bd)["verdict"], "PASS")
    check("ladder: gain exactly anchor + 0.10 -> PASS (>= boundary)",
          seed_verdict(_pt("a", fh=(0.60, 0.60)), anc, bd)["verdict"], "PASS")
    check("ladder: gain +0.09 -> NO GAIN", seed_verdict(_pt("a", fh=(0.59, 0.59)), anc, bd)["verdict"], "NO GAIN")
    check("ladder: gain needs 2 x band when wider (band 0.2 -> +0.40)",
          seed_verdict(_pt("a", fh=(0.85, 0.85)), wide, bands(wide, anc))["verdict"], "NO GAIN")
    check("ladder: field_light 0.04 below -> NO GAIN (hold-the-line 0.04 band... floor max(0.03, 0.04))",
          seed_verdict(_pt("a", fh=(0.62, 0.62), fl=(0.755, 0.755)), anc, bd)["verdict"], "NO GAIN")
    check("ladder: field_light exactly anchor - 0.04 -> PASS (boundary)",
          seed_verdict(_pt("a", fh=(0.62, 0.62), fl=(0.76, 0.76)), anc, bd)["verdict"], "PASS")
    check("ladder: ring_obst 0.06 below -> NO GAIN", seed_verdict(_pt("a", fh=(0.62, 0.62), ro=(0.54, 0.54)), anc,
                                                                  bd)["verdict"], "NO GAIN")
    check("ladder: a missing sidecar -> INVALID EVAL (never equal)",
          seed_verdict(dict(good, valid=dict(good["valid"], **{f"{FH}@2": _valid(sidecar=None)})), anc, bd)["verdict"],
          "INVALID EVAL")
    check("ladder: a different code_sha sidecar -> INVALID EVAL",
          seed_verdict(dict(good, valid=dict(good["valid"], **{"suite@1": _valid(sidecar=("d" * 64, "e" * 64, False))})),
                       anc, bd)["verdict"], "INVALID EVAL")
    check("ladder: a dirty sidecar vs a clean anchor -> INVALID EVAL",
          seed_verdict(dict(good, valid=dict(good["valid"], **{"lr@1": _valid(sidecar=("c" * 64, "e" * 64, True))})),
                       anc, bd)["verdict"], "INVALID EVAL")
    check("ladder: a stamp without CLOCK_HIDE=1 -> INVALID EVAL",
          seed_verdict(dict(good, valid=dict(good["valid"], **{"suite@1": _valid(env4=False)})), anc, bd)["verdict"],
          "INVALID EVAL")
    check("ladder: a row env block differing from the anchor's -> INVALID EVAL",
          seed_verdict(dict(good, valid=dict(good["valid"], **{f"{FH}@1": _valid({"KRABBY_FIELD_PAD_R": "1.8"})})),
                       anc, bd)["verdict"], "INVALID EVAL")
    check("ladder: a missing env_seed-2 field_hard -> PENDING",
          seed_verdict(dict(good, comp=dict(good["comp"], field_hard={1: good["comp"]["field_hard"][1], 2: None})),
                       anc, bd)["verdict"], "PENDING")
    check("ladder: a NA hand -> PENDING, never FAST", seed_verdict(_pt("a", fh=(0.62, 0.62), hands=("FAST", "NA")),
                                                                  anc, bd)["verdict"], "PENDING")
    check("ladder: a NA metric -> PENDING", seed_verdict(dict(good, suite=dict(good["suite"], turn_yp=None)), anc,
                                                         bd)["verdict"], "PENDING")
    check("ladder: a missing dose read -> PENDING", seed_verdict(_pt("a", fh=(0.62, 0.62), dose=None), anc, bd)["verdict"],
          "PENDING")
    check("ladder: a missing band -> PENDING", seed_verdict(good, anc, dict(bd, ring_obst=None))["verdict"], "PENDING")
    check("ladder: INVALID outranks PENDING", seed_verdict(dict(_pt("a", dose=None), valid=dict(good["valid"], **{
        "lr@1": _valid(sidecar=None)})), anc, bd)["verdict"], "INVALID EVAL")
    check("ladder: a SLOW side hand -> HAND LOSS", seed_verdict(_pt("a", fh=(0.62, 0.62), hands=("FAST", "SLOW")), anc,
                                                               bd)["verdict"], "HAND LOSS")
    check("ladder: hold-fall 0.16 -> HAND LOSS", seed_verdict(_pt("a", fh=(0.62, 0.62), fall=0.16), anc, bd)["verdict"],
          "HAND LOSS")
    check("ladder: hold-fall exactly 0.15 is not HAND LOSS", seed_verdict(_pt("a", fh=(0.62, 0.62), fall=0.15), anc,
                                                                          bd)["verdict"], "PASS")
    check("ladder: dose lost -> DOSE", seed_verdict(_pt("a", fh=(0.62, 0.62), dose=False), anc, bd)["verdict"], "DOSE")
    check("ladder: HAND LOSS outranks DOSE", seed_verdict(_pt("a", hands=("SLOW", "FAST"), dose=False), anc, bd)["verdict"],
          "HAND LOSS")
    check("ladder: |ratio-1| 0.11 worse (> 0.10 floor) -> FLAT FAIL",
          seed_verdict(_pt("a", fh=(0.62, 0.62), err=0.14 + 1e-6), anc, bd)["verdict"], "FLAT FAIL")
    check("ladder: |ratio-1| exactly 0.10 worse -> not FLAT FAIL (COST: > max(0.03, band))",
          seed_verdict(_pt("a", fh=(0.62, 0.62), err=0.13), anc, bd)["verdict"], "COST")
    check("ladder: yaw_progress 0.031 lower (floor 0.03) -> FLAT FAIL",
          seed_verdict(_pt("a", fh=(0.62, 0.62), yp=0.069), anc, bd)["verdict"], "FLAT FAIL")
    check("ladder: yaw_progress 0.02 lower -> COST (margin 0.01)", seed_verdict(_pt("a", fh=(0.62, 0.62), yp=0.08), anc,
                                                                                bd)["verdict"], "COST")
    check("ladder: overshoot (ratio 1.12 vs 1.03) counts as worse tracking error",
          seed_verdict(_pt("a", fh=(0.62, 0.62), err=0.12), anc, bd)["verdict"], "COST")
    check("ladder: a translate hold at 0.94 -> FLAT FAIL", seed_verdict(_pt("a", fh=(0.62, 0.62), trans=0.94), anc,
                                                                        bd)["verdict"], "FLAT FAIL")
    check("ladder: a translate hold at exactly 0.95 passes", seed_verdict(_pt("a", fh=(0.62, 0.62), trans=0.95), anc,
                                                                          bd)["verdict"], "PASS")
    check("ladder: FLAT FAIL outranks NO GAIN", seed_verdict(_pt("a", trans=0.90), anc, bd)["verdict"], "FLAT FAIL")
    check("ladder: move_turn survival 0.06 lower -> COST (margin 0.05)",
          seed_verdict(_pt("a", fh=(0.62, 0.62), mt=0.84), anc, bd)["verdict"], "COST")
    check("ladder: move_turn survival 0.05 lower -> PASS (boundary)",
          seed_verdict(_pt("a", fh=(0.62, 0.62), mt=0.85), anc, bd)["verdict"], "PASS")

    def V(v, gain, cost=()):
        return {"verdict": v, "gain": gain, "cost": list(cost)}
    check("recipe: 3 PASS -> ADOPTABLE", recipe_read({s: V("PASS", True) for s in SEEDS})["reading"], "ADOPTABLE")
    check("recipe: 2 PASS + 1 HAND LOSS -> not ADOPTABLE (MIXED)",
          recipe_read({16: V("PASS", True), 17: V("PASS", True), 18: V("HAND LOSS", True)})["reading"], "MIXED")
    check("recipe: 2 COST on the same metric -> TRADE-OFF",
          recipe_read({16: V("COST", True, ["turn_yp"]), 17: V("COST", True, ["turn_yp", "err_side"]),
                       18: V("PASS", True)})["reading"], "TRADE-OFF")
    check("recipe: 2 COST on different metrics -> MIXED",
          recipe_read({16: V("COST", True, ["turn_yp"]), 17: V("COST", True, ["err_side"]), 18: V("PASS", True)})["reading"],
          "MIXED")
    check("recipe: 1/3 gain -> NOT ADOPTABLE",
          recipe_read({16: V("PASS", True), 17: V("NO GAIN", False), 18: V("NO GAIN", False)})["reading"], "NOT ADOPTABLE")
    check("recipe: 2/3 HAND LOSS -> NOT ADOPTABLE",
          recipe_read({16: V("HAND LOSS", True), 17: V("HAND LOSS", True), 18: V("PASS", True)})["reading"], "NOT ADOPTABLE")
    check("recipe: a PENDING seed -> PENDING", recipe_read({16: V("PASS", True), 17: V("PENDING", None),
                                                            18: V("PASS", True)})["reading"], "PENDING")
    pts = {s: _pt(f"a{s}", fh=(0.70, 0.70)) for s in SEEDS}
    check("p2b: gain 3/3 and field_hard < 0.85 3/3 -> trigger", p2b_trigger({s: V("PASS", True) for s in SEEDS}, pts)["trigger"],
          True)
    pts_hi = {s: _pt(f"a{s}", fh=(0.90, 0.90), track=0.9) for s in SEEDS}
    check("p2b: no deficit -> no trigger", p2b_trigger({s: V("PASS", True) for s in SEEDS}, pts_hi)["trigger"], False)
    pts_tr = {s: _pt(f"a{s}", fh=(0.90, 0.90), track=0.65) for s in SEEDS}
    check("p2b: walking-hold tracking 0.65 < 0.7 is a deficit", p2b_trigger({s: V("PASS", True) for s in SEEDS},
                                                                            pts_tr)["trigger"], True)
    check("p2b: gain 1/3 -> no trigger", p2b_trigger({16: V("PASS", True), 17: V("NO GAIN", False),
                                                      18: V("NO GAIN", False)}, pts)["trigger"], False)

    a1 = _pt("P2__A1_s18", fh=(0.62, 0.62), jit=0.20, feet=0.23, power=100.0, stand_clearance=0.08, tripod=0.79)
    s1 = _pt("P2__S1_s18", fh=(0.62, 0.62), jit=0.14, feet=0.22, power=90.0, stand_clearance=0.08)
    check("sibling S1: jitter -30 %, feet -0.01, nothing lost -> KEEP", sibling_rec("S1", s1, a1, bd)["rec"], "KEEP")
    check("sibling S1: jitter -20 % -> DO NOT KEEP", sibling_rec("S1", dict(s1, suite=dict(s1["suite"], jitter=0.16)),
                                                                 a1, bd)["rec"], "DO NOT KEEP")
    check("sibling S1: jitter exactly -25 % -> KEEP (boundary)",
          sibling_rec("S1", dict(s1, suite=dict(s1["suite"], jitter=0.15)), a1, bd)["rec"], "KEEP")
    check("sibling S1: side feet -0.03 -> DO NOT KEEP", sibling_rec("S1", dict(s1, feet_side=0.20), a1, bd)["rec"],
          "DO NOT KEEP")
    check("sibling S1: a SLOW hand -> DO NOT KEEP", sibling_rec("S1", dict(s1, hands={"l": "FAST", "r": "SLOW"}), a1,
                                                                bd)["rec"], "DO NOT KEEP")
    check("sibling S1: field_hard 0.10 lower -> DO NOT KEEP (nothing loses)",
          sibling_rec("S1", _pt("P2__S1_s18", fh=(0.52, 0.52), jit=0.14), a1, bd)["rec"], "DO NOT KEEP")
    check("sibling S1: a NA hand -> PENDING", sibling_rec("S1", dict(s1, hands={"l": "NA", "r": "FAST"}), a1, bd)["rec"],
          "PENDING")
    check("sibling S1: a missing sidecar -> INVALID EVAL",
          sibling_rec("S1", dict(s1, valid=dict(s1["valid"], **{"suite@1": _valid(sidecar=None)})), a1, bd)["rec"],
          "INVALID EVAL")
    check("sibling E1: power down + S1's conditions -> KEEP", sibling_rec("E1", s1, a1, bd)["rec"], "KEEP")
    check("sibling E1: power not down -> DO NOT KEEP", sibling_rec("E1", dict(s1, power=101.0), a1, bd)["rec"],
          "DO NOT KEEP")
    check("sibling E1: power unread -> PENDING", sibling_rec("E1", dict(s1, power=None), a1, bd)["rec"], "PENDING")
    f1 = _pt("P2__F1_s18", fh=(0.62, 0.62), stand_clearance=0.035, tripod=0.04)
    check("sibling F1: A1 marches, F1 taps low -> KEEP", sibling_rec("F1", f1, a1, bd)["rec"], "KEEP")
    check("sibling F1: clearance exactly 0.05 / tripod 0.4 -> KEEP (boundary)",
          sibling_rec("F1", _pt("P2__F1_s18", fh=(0.62, 0.62), stand_clearance=0.05, tripod=0.4), a1, bd)["rec"], "KEEP")
    check("sibling F1: F1 still marches -> DO NOT KEEP",
          sibling_rec("F1", _pt("P2__F1_s18", fh=(0.62, 0.62), stand_clearance=0.08, tripod=0.7), a1, bd)["rec"],
          "DO NOT KEEP")
    check("sibling F1: A1 already taps -> NOT RECOMMENDED",
          sibling_rec("F1", f1, dict(a1, stand_clearance=0.04), bd)["rec"], "NOT RECOMMENDED")
    check("sibling F1: no clearance read -> PENDING", sibling_rec("F1", dict(f1, stand_clearance=None), a1, bd)["rec"],
          "PENDING")

    man = {"stand": {"survival": 1.0, "n": 100}, "fwd": {"survival": 1.0, "ratio": 1.041, "n": 100}}
    cats = {"stand": {"survival": 1.0, "ratio": None}, "fwd": {"survival": 1.0, "ratio": 1.0413}}
    A = {s: _pt(f"ANC_s{s}", fh=(0.45, 0.45), fl=(0.70, 0.70), suite=dict(_pt("x")["suite"], categories=cats))
         for s in SEEDS}
    check("anchors: in-window and discriminating, inert -> ok", anchors_check(A, man)["ok"], True)
    check("anchors: field_hard 0.90 -> stop", anchors_check({**A, **{16: _pt("x", fh=(0.9, 0.9), fl=(1.0, 1.0))}},
                                                            man)["ok"], False)
    check("anchors: field_hard exactly 0.85 is in the window",
          anchors_check({**A, **{16: _pt("x", fh=(0.85, 0.85), fl=(0.95, 0.95))}}, man)["ok"], True)
    check("anchors: field_light < field_hard + 0.10 -> stop",
          anchors_check({**A, **{17: _pt("x", fh=(0.45, 0.45), fl=(0.54, 0.54))}}, man)["ok"], False)
    check("anchors: the s18 suite off the manifest -> stop (not inert)",
          anchors_check({**A, **{18: dict(A[18], suite=dict(A[18]["suite"], categories=dict(
              cats, fwd={"survival": 1.0, "ratio": 1.043})))}}, man)["ok"], False)
    check("anchors: no manifest -> stop (fail closed)", anchors_check(A, None)["ok"], False)

    head = _pt("P2__A1_s18", fh=(0.50, 0.50))
    m_hi = _pt("m", fh=(0.65, 0.65))
    m_1 = dict(_pt("m1", fh=(0.65, 0.65)), comp=dict(_pt("m1")["comp"], field_hard={1: {"rate": 0.65, "n": 100,
                                                                                          "falls": 35, "track": 0.8}, 2: None}))
    ms = m_screen(head, {16000: _pt("m0", fh=(0.55, 0.55)), 17500: m_1, 19000: m_hi}, anc, bd)
    check("m-screen: +0.05 -> NO; +0.15 seed-1 only -> NEEDS_SEED2; +0.15 at n=200 -> REPLACES HEAD",
          [ms[16000]["status"], ms[17500]["status"], ms[19000]["status"]], ["NO", "NEEDS_SEED2", "REPLACES HEAD"])
    ms2 = m_screen(head, {19000: _pt("m", fh=(0.65, 0.65), hands=("FAST", "SLOW"))}, anc, bd)
    check("m-screen: a broken floor blocks the replacement", ms2[19000]["status"], "NO (n=200)")
    ri = rank_inputs({"s16": _pt("a", fh=(0.7, 0.7)), "s17": _pt("b", fh=(0.6, 0.6)), "s18": _pt("c", fh=(0.6, 0.6))},
                     {})
    check("rank inputs: obstacle rank-sum, ties averaged (fh 1/2.5/2.5 + three 3-way ties at 2)", [ri[k]["obstacle_rank_sum"] for k in ("s16", "s17", "s18")],
          [7.0, 8.5, 8.5])
    cal = calibrate([0.8, 0.9, 0.5, 0.7], [1.0, 1.0, 1.0, 1.0], [1, 1, 0, 1])
    check("calibrate: r-bar = median on field tiles (0.8) -> 0.7:0.4", (round(cal["r_bar"], 3), cal["promote"]),
          (0.8, "0.7:0.4"))
    check("calibrate: no field-tile episode -> not ok", calibrate([0.8], [1.0], [0])["ok"], False)
    check("calibrate: invalid episodes are excluded", calibrate([0.1, 0.8], [1.0, 1.0], [1, 1], [0, 1])["promote"],
          "0.7:0.4")
    import tempfile  # noqa: PLC0415
    with tempfile.TemporaryDirectory() as td:
        f = Path(td) / "d.json"
        f.write_text(json.dumps({"track_ratio": [0.8, 0.9, 0.5, 0.7], "is_obst_tile": [1, 1, 0, 1],
                                 "track_ratio_valid": [1, 1, 1, 1]}))
        check("calibrate: a per-episode track_ratio dump is accepted", calibrate_file(f).get("promote"), "0.7:0.4")
        f.write_text(json.dumps({"is_obst_tile": [1]}))
        check("calibrate: a dump with no tracking key is refused", calibrate_file(f)["ok"], False)
    ser = ([14996 + i for i in range(300)], [0.6] * 200 + [0.8] * 100)
    cs = calibrate_series(ser)
    check("calibrate (tfevents): r-bar = mean of the last 100 logged its (0.8) -> 0.7:0.4",
          (round(cs["r_bar"], 6), cs["promote"], cs["first_step"]), (0.8, "0.7:0.4", 15196))
    check("calibrate (tfevents): fewer than 100 logged its fails closed",
          calibrate_series(([1, 2], [0.8, 0.8]))["ok"], False)
    check("calibrate (tfevents): no tag fails closed", calibrate_series(None)["ok"], False)
    nan_ser = ([i for i in range(150)], [0.6] * 40 + [float("nan")] * 10 + [0.9] * 100)
    check("calibrate (tfevents): NaN points are dropped, not averaged (0.9 -> 0.8:0.45)",
          calibrate_series(nan_ser)["promote"], "0.8:0.45")
    sha = "b" * 64
    A_sha = {s: dict(A[s], valid={k: dict(_valid(), ckpt_sha=sha) for k in NEEDED_STAMPS}) for s in SEEDS}
    check("anchors: every stamp's checkpoint sha matches anchors.yaml -> ok",
          anchors_check(A_sha, man, {s: sha for s in SEEDS})["ok"], True)
    check("anchors: one stamp of another checkpoint -> stop",
          anchors_check({**A_sha, 17: dict(A_sha[17], valid=dict(A_sha[17]["valid"], **{f"{FH}@2": dict(
              _valid(), ckpt_sha="c" * 64)}))}, man, {s: sha for s in SEEDS})["ok"], False)
    check("anchors: a stamp without a checkpoint sha -> stop (fail closed)",
          anchors_check(A, man, {s: sha for s in SEEDS})["ok"], False)
    check("validity: field_suite / field_turn stamps are compared (a dirty field_suite sidecar -> INVALID EVAL)",
          seed_verdict(dict(good, valid=dict(good["valid"], **{f"{FS}@1": _valid(sidecar=("c" * 64, "e" * 64, True))})),
                       dict(anc, valid=dict(anc["valid"], **{f"{FS}@1": _valid()})), bd)["verdict"], "INVALID EVAL")
    check("validity: a field_turn row env block differing from the anchor's -> INVALID EVAL",
          seed_verdict(dict(good, valid=dict(good["valid"], **{f"{FT}@1": _valid({"KRABBY_FIELD_PAD_R": "1.8"})})),
                       dict(anc, valid=dict(anc["valid"], **{f"{FT}@1": _valid()})), bd)["verdict"], "INVALID EVAL")
    lv = levels_summary(([14996 + i for i in range(1000)], [1.0 + i / 999 for i in range(1000)]))
    check("levels: start = first 20 points, tail = last 100 iterations",
          (round(lv["start"], 4), round(lv["tail"], 4), lv["last_step"]),
          (round(1.0 + 9.5 / 999, 4), round(1.0 + 949.5 / 999, 4), 15995))
    check("levels: no series -> None", levels_summary(None), None)
    ft = {"turn_fc": 0.8, "turn_yp": 0.2, "completion": 0.9, "fall_share": 0.05}
    ri2 = rank_inputs({"a": _pt("a", field_turn=ft), "b": _pt("b", field_turn=dict(ft, turn_fc=0.7, turn_yp=0.1))}, {})
    check("rank inputs: field_turn turn holds give the turn rank-sum (item 5 stand-in)",
          [ri2["a"]["turn_rank_sum"], ri2["b"]["turn_rank_sum"], ri2["a"]["field_turn_completion"]], [2.0, 4.0, 0.9])

    # ---- CHARTER 10 (p2b): the ladder vs REF (n = 400 SE gate), the chained ANCW floor, attribution, the head list
    def P4(name, fh=0.83, **kw):
        """A p2b-shaped point: field_hard read at env_seeds 1-4 (n = 400)."""
        fh4 = fh if isinstance(fh, tuple) else (fh,) * 4
        pt = _pt(name, fh=fh4[:2], **kw)
        for i, sd in enumerate(P2B_FH_SEEDS[2:], 2):
            pt["comp"]["field_hard"][sd] = dict(pt["comp"]["field_hard"][1], rate=fh4[i])
            pt["stamps"][f"{FH}@{sd}"], pt["valid"][f"{FH}@{sd}"] = "/x", _valid()
        return pt
    ANCW = _pt("ANCW_s18", s2=s2)
    bd_a = bands(ANCW, ANCW)
    P2A = _pt("P2__S1_s18")

    def pv(arm_fh, ref_fh, **kw):
        ref = P4("REF_S1_s18", ref_fh, s2=s2)
        return p2b_verdict(P4("P2__CL_s18", arm_fh, **kw), ref, bands(ref, ref), P2A, ANCW, bd_a)
    check("p2b SE gate: REF 0.83 + 0.06 -> PASS (the 0.05 floor binds: 2 SE = 0.049)", pv(0.89, 0.83)["verdict"], "PASS")
    check("p2b SE gate: REF 0.80 + 0.06 -> PASS (2 SE = 0.053)", pv(0.86, 0.80)["verdict"], "PASS")
    check("p2b SE gate: REF 0.73 + 0.06 -> NO GAIN (2 SE = 0.0602: the bar is just above +0.06 here)",
          (pv(0.79, 0.73)["verdict"], round(pv(0.79, 0.73)["gain_read"]["margin"], 4)), ("NO GAIN", 0.0602))
    check("p2b SE gate: REF 0.73 + 0.0625 (317/400, the smallest passing count) -> PASS", pv(0.7925, 0.73)["verdict"],
          "PASS")
    check("p2b SE gate: exactly REF + 0.05 where 2 SE < 0.05 -> PASS (>= boundary)", pv(0.88, 0.83)["verdict"], "PASS")
    check("p2b SE gate: REF 0.83 + 0.04 -> NO GAIN", pv(0.87, 0.83)["verdict"], "NO GAIN")
    check("p2b SE gate: field_hard pools all four env_seeds (es1/es2 0.95, es3/es4 0.83 -> 0.89 vs REF 0.83)",
          pv((0.95, 0.95, 0.83, 0.83), 0.83)["gain_read"]["field_hard"][:2], [0.89, 0.83])
    check("p2b SE gate: field_light 0.04 below REF (band floor 0.04) holds; 0.05 below -> NO GAIN",
          [pv(0.89, 0.83, fl=(0.76, 0.76))["verdict"], pv(0.89, 0.83, fl=(0.75, 0.75))["verdict"]], ["PASS", "NO GAIN"])
    no34 = P4("P2__CL_s18", 0.89)
    del no34["stamps"][f"{FH}@4"], no34["comp"]["field_hard"][4]
    r83 = P4("REF_S1_s18", 0.83, s2=s2)
    check("p2b: field_hard read at env_seeds 1-3 only (n = 300) -> PENDING",
          p2b_verdict(no34, r83, bands(r83, r83), P2A, ANCW, bd_a)["verdict"], "PENDING")
    check("p2b: an env_seed-3 stamp on other code than REF's -> INVALID EVAL",
          p2b_verdict(dict(P4("a", 0.89), valid=dict(P4("a")["valid"], **{f"{FH}@3": _valid(sidecar=("d" * 64, "e" * 64,
                                                                                                     False))})),
                      r83, bands(r83, r83), P2A, ANCW, bd_a)["verdict"], "INVALID EVAL")
    rv = pv(0.89, 0.83, track=0.71)
    check("p2b: every verdict reports field_hard walking-hold tracking beside completion",
          (round(rv["report"]["field_hard"], 4), rv["report"]["field_hard_n"], rv["report"]["field_hard_track"],
           rv["report"]["ref_field_hard_n"]), (0.89, 400, 0.71, 400))
    # chained ANCW floor: the S1 p2a head was 0.08 worse than ANCW on every |ratio - 1| (its REF reproduces it)
    p2a_bad, ref_bad = _pt("P2__S1_s18", err=0.11), P4("REF_S1_s18", 0.83, err=0.11, s2=s2)

    def pc(err, ancw=ANCW, bda=bd_a):
        return p2b_verdict(P4("P2__CL_s18", 0.89, err=err), ref_bad, bands(ref_bad, ref_bad), p2a_bad, ancw, bda)
    check("p2b ANCW chain: +0.04 vs REF (COST range) + 0.08 p2a -> 0.12 > 0.10 vs ANCW -> FLAT FAIL",
          (pc(0.15)["verdict"], round(pc(0.15)["ancw_chain"]["err_side"]["total"], 4)), ("FLAT FAIL", 0.12))
    check("p2b ANCW chain: +0.01 vs REF + 0.08 -> 0.09 <= 0.10 -> PASS", pc(0.12)["verdict"], "PASS")
    check("p2b ANCW chain: exactly 0.10 chained -> not FLAT FAIL; +0.02 vs REF is under the 0.03 COST margin -> PASS",
          pc(0.13)["verdict"], "PASS")
    check("p2b ANCW chain: computed from values -- ANCW stamps on the p2a code_sha are not INVALID",
          pc(0.12, dict(ANCW, valid={k: _valid(sidecar=("a" * 64, "e" * 64, False)) for k in NEEDED_STAMPS}))["verdict"],
          "PASS")
    check("p2b ANCW chain: no ANCW s18 suite band -> PENDING", pc(0.12, ANCW, bands(ANCW, _pt("x")))["verdict"],
          "PENDING")
    check("p2b ANCW chain: no p2a suite read -> PENDING",
          p2b_verdict(P4("a", 0.89), r83, bands(r83, r83), dict(P2A, suite=None), ANCW, bd_a)["verdict"], "PENDING")
    # attribution at s18
    bd18 = bands(r83, r83)
    pass_v = {"verdict": "PASS", "gain": True}
    check("attribution: CL 0.89 PASS vs REF, C0 0.83 -> BEATS CONTINUATION",
          p2b_attribution(P4("P2__CL_s18", 0.89), P4("P2__C0_s18", 0.83), pass_v, bd18)["status"], "BEATS CONTINUATION")
    check("attribution: C0 0.86 (CL needs >= 0.91) -> DOES NOT BEAT",
          p2b_attribution(P4("P2__CL_s18", 0.89), P4("P2__C0_s18", 0.86), pass_v, bd18)["beats_continuation"], False)
    check("attribution: the lever's REF verdict NO GAIN -> DOES NOT BEAT (needs its own gain)",
          p2b_attribution(P4("P2__CL_s18", 0.89), P4("P2__C0_s18", 0.70), {"verdict": "NO GAIN", "gain": False},
                          bd18)["beats_continuation"], False)
    check("attribution: COST vs REF still counts as a gain",
          p2b_attribution(P4("P2__CL_s18", 0.89), P4("P2__C0_s18", 0.83), {"verdict": "COST", "gain": True},
                          bd18)["beats_continuation"], True)
    check("attribution: |ratio - 1| 0.11 worse than C0 -> FLAT FAIL vs C0 -> DOES NOT BEAT",
          p2b_attribution(P4("P2__CL_s18", 0.89, err=0.13 + 1e-6), P4("P2__C0_s18", 0.83, err=0.02), pass_v,
                          bd18)["beats_continuation"], False)
    check("attribution: C0 not recorded -> PENDING",
          p2b_attribution(P4("P2__CL_s18", 0.89), {"name": "P2__C0_s18", "missing": True}, pass_v, bd18)["status"],
          "PENDING")
    # the head list: C0 at s18 only; P2B_LEVERS narrows the levers
    check("p2b heads: C0 at s18 only, levers at every seed",
          p2b_head_names(("CL", "TG")), ["P2__CL_s16", "P2__TG_s16", "P2__CL_s17", "P2__TG_s17", "P2__C0_s18",
                                         "P2__CL_s18", "P2__TG_s18"])
    check("p2b heads: P2B_LEVERS (default 'CL TG')", (p2b_levers({}), p2b_levers({"P2B_LEVERS": "TG"})),
          (("CL", "TG"), ("TG",)))
    res_m = {"refs": {s: P4(ref_name(s)) for s in SEEDS}, "ancw": {s: _pt(f"ANCW_s{s}") for s in SEEDS},
             "p2a": {s: _pt(f"P2__S1_s{s}") for s in SEEDS},
             "heads": {n: {"name": n, "missing": True} for n in p2b_head_names(("CL", "TG"))}}
    vs_m = p2b_verdicts(res_m)
    md_m = p2b_markdown(vs_m, p2b_attributions(res_m, vs_m, ("CL", "TG")), p2b_recipes(vs_m, ("CL", "TG")))
    check("p2b table: C0 listed only at s18, s16/s17 levers labelled 'lever + continuation'",
          ([ln.split(" | ")[0] for ln in md_m.splitlines() if "P2__C0" in ln], vs_m["P2__CL_s16"]["label"],
           vs_m["P2__CL_s18"]["label"]), (["| P2__C0_s18"], "lever + continuation", None))
    check("p2b recipe: every lever PENDING before any run", sorted({r["reading"] for r in p2b_recipes(vs_m).values()}),
          ["PENDING"])
    # step-3 inertness: REF reproduces the S1 p2a head's own row-eval reads exactly (like-for-like)
    ft = {"completion": 0.9, "n": 100, "fall_share": 0.05, "turn_fc": 0.8, "turn_yp": 0.2}

    def ip(name, sha="h" * 64, **kw):
        pt = (P4 if name.startswith("REF") else _pt)(name, fh=(0.83,) * 4 if name.startswith("REF") else (0.83, 0.83),
                                                     field_turn=dict(ft), field_suite=dict(_pt("x")["suite"]),
                                                     feet={"l": 0.23, "r": 0.23}, **kw)
        pt["stamps"].update({f"{FT}@1": "/x", f"{FS}@1": "/x"})
        pt["valid"] = {k: dict(_valid(), ckpt_sha=sha) for k in pt["stamps"]}
        return pt
    p2a_i = {s: ip(f"P2__S1_s{s}") for s in SEEDS}
    refs_i = {s: ip(ref_name(s)) for s in SEEDS}
    check("inertness: identical reads on every row -> ok", inertness_check(refs_i, p2a_i)["ok"], True)
    fh17 = dict(refs_i[17]["comp"]["field_hard"])
    fh17[2] = dict(fh17[2], rate=0.84)
    r_fh = dict(refs_i[17], comp=dict(refs_i[17]["comp"], field_hard=fh17))
    check("inertness: a field_hard es2 completion 0.84 vs 0.83 -> stop",
          inertness_check({**refs_i, 17: r_fh}, p2a_i)["ok"], False)
    check("inertness: a hand class differing -> stop",
          inertness_check({**refs_i, 16: dict(refs_i[16], hands={"l": "FAST", "r": "SLOW"})}, p2a_i)["ok"], False)
    no_ft = dict(refs_i[18], stamps={k: v for k, v in refs_i[18]["stamps"].items() if k != f"{FT}@1"})
    check("inertness: REF missing a row the head has -> stop (fail closed)",
          inertness_check({**refs_i, 18: no_ft}, p2a_i)["ok"], False)
    check("inertness: a REF stamp of another checkpoint -> stop",
          inertness_check({**refs_i, 18: ip(ref_name(18), sha="z" * 64)}, p2a_i)["ok"], False)
    check("inertness: the driver suite is not compared (a differing suite@1 read is ignored)",
          inertness_check({**refs_i, 18: dict(refs_i[18], suite=dict(refs_i[18]["suite"], err_back=0.5))}, p2a_i)["ok"],
          True)
    check("inertness: no REF point -> stop", inertness_check({16: refs_i[16], 17: refs_i[17]}, p2a_i)["ok"], False)
    print("self-test:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


# ------------------------------------------------------------------------------------------------ p2s self-test
def _p2s_summ(steps=2.83, taps=2.83, clear=0.135, og=0.34, hover=0.0, ks=1.72, height=1.068, falls=0, n=100) -> dict:
    """A p2s_stand.head_summary-shaped stand summary (REF's suite stand by default, P2S_PLAN step 1 baselines)."""
    return {"n": n, "steps_per_s": steps, "taps_per_s": taps, "clearance_m": clear, "off_ground_share": og,
            "hover_episodes": int(round(hover * n / 100)), "hover_episodes_per_100": hover, "ks_m_s": ks,
            "body_height_m": height, "falls": falls}


def _p2s_ws(onset=0.16, falls=0, reached=300, n=100) -> dict:
    return {"n_episodes": n, "stand_reached": reached, "falls": falls, "onset_values": [onset] * n,
            "onset_median_s": onset, "env_seed": 1}


def _p2s_pt(name, fh=0.81, s1=None, s2=None, ws=(0.16, 0.16), ws_falls=0, field_falls=0, ft=0.9, flat_s2=None,
            **kw) -> dict:
    """A p2s-shaped point: :func:`_pt` + field_hard at env_seeds 1-4, suite@2, field_turn and walkstop at 1 / 2, and
    the ``p2s`` stand reads (summaries, walkstop reads) set directly."""
    pt = _pt(name, fh=(fh, fh), field_turn={"completion": ft, "n": 100, "fall_share": 0.0, "turn_fc": 0.8,
                                            "turn_yp": 0.2}, s2=flat_s2, **kw)
    for sd in P2B_FH_SEEDS[2:]:
        pt["comp"]["field_hard"][sd] = dict(pt["comp"]["field_hard"][1])
    for k in [f"{FH}@{sd}" for sd in P2B_FH_SEEDS[2:]] + list(P2S_NEEDED_EXTRA):
        pt["stamps"][k], pt["valid"][k] = "/x", _valid()
    pt["ws_env"] = {1: {}, 2: {}}
    s1 = s1 or _p2s_summ()
    pt["p2s"] = {"suite": {1: s1, 2: s2 or s1},
                 "walkstop": {1: _p2s_ws(ws[0], ws_falls), 2: _p2s_ws(ws[1], ws_falls)},
                 "field": {k: _p2s_summ(falls=field_falls) for k in [f"{FH}@{sd}" for sd in P2B_FH_SEEDS]
                           + [f"{FL}@1", f"{FL}@2"]},
                 "turn": _p2s_summ()}
    return pt


def _p2s_self_test() -> int:
    ok = True

    def check(name, got, want):
        nonlocal ok
        good = got == want
        ok &= good
        print(f"  [{'ok' if good else 'FAIL'}] {name}: {got}")

    s2 = {"err_fwd": 0.03, "err_back": 0.03, "err_side": 0.03, "err_diag": 0.03, "turn_fc": 0.80, "turn_yp": 0.10,
          "mt_surv": 0.90}
    refs = _p2s_pt("REFS_S1_s18", s2=_p2s_summ(steps=2.75), flat_s2=s2)
    c0r = _p2s_pt("C0R_s18", s1=_p2s_summ(steps=2.67, clear=0.113, og=0.341), flat_s2=s2)
    p2a, ancw = _pt("P2__S1_s18"), _pt("ANCW_s18", s2=s2)
    bd, bd_a = bands(refs, refs), bands(ancw, ancw)
    sbd = p2s_stand_bands(refs)
    check("stand bands: |es1 - es2| over the floor (steps 0.08 -> 0.25), KS / onset at their floors",
          (sbd["steps_per_s"], sbd["ks_m_s"], sbd["onset_s"], sbd["off_ground_share"]), (0.25, 0.10, 0.1, 0.02))

    def V(pt, cmp=c0r, need_dose=True):
        return p2s_verdict(pt, refs, cmp, bd, p2a, ancw, bd_a, need_dose)
    still = _p2s_summ(steps=0.2, taps=0.3, clear=0.02, og=0.03, hover=0.0, ks=0.3)
    v = V(_p2s_pt("P2__ST2_s18", s1=still))
    check("STILL: every clause on both suite seeds, no COST", (v["verdict"], v["stand"]["outcome"], v["gain"]),
          ("STILL", "STILL", True))
    check("STILL: the queue's ST1 condition reads true", p2s_reads_still(v), True)
    part = _p2s_summ(steps=1.0, taps=1.2, clear=0.08, og=0.15, hover=2.0, ks=1.5)
    v = V(_p2s_pt("P2__ST2_s18", s1=part))
    check("PARTIAL: off-ground / steps <= 0.5 x C0R's, hover <= C0R + 5, KS <= C0R + band",
          (v["verdict"], v["stand"]["note"]), ("PARTIAL", None))
    check("PARTIAL is not the ST1 condition", p2s_reads_still(v), False)
    clr = _p2s_summ(steps=0.3, taps=0.5, clear=0.07, og=0.08, hover=0.0, ks=0.4)
    v = V(_p2s_pt("P2__ST2_s18", s1=clr))
    check("PARTIAL from the clearance clause alone carries the note",
          (v["verdict"], bool(v["stand"]["note"] and "clearance clause alone" in v["stand"]["note"])), ("PARTIAL", True))
    gam = _p2s_summ(steps=0.3, taps=0.4, clear=0.10, og=0.40, hover=18.0, ks=0.9)
    v = V(_p2s_pt("P2__ST2_s18", s1=gam))
    check("GAMED: steps/s <= 0.5 but off-ground / hover over the PARTIAL clauses (a flag)", (v["verdict"], v["gain"]),
          ("GAMED", False))
    v = V(_p2s_pt("P2__ST2_s18", s1=_p2s_summ(steps=2.5, og=0.33)))
    check("NO EFFECT: stepping on", (v["verdict"], v["gain"]), ("NO EFFECT", False))
    v = V(_p2s_pt("P2__ST2_s18", s1=still, s2=_p2s_summ(steps=0.6, taps=0.6, clear=0.03, og=0.05, ks=0.4)))
    check("STILL needs BOTH suite seeds (es2 at 0.6 steps/s -> PARTIAL)", v["stand"]["outcome"], "PARTIAL")
    v = V(_p2s_pt("P2__ST2_s18", s1=still, ws=(0.60, 0.60)))
    check("COST: walkstop median onset 0.60 s > REFS 0.16 + max(0.3, band 0.1)", (v["verdict"], v["cost"],
                                                                                 v["stand"]["outcome"]),
          ("COST", ["onset"], "STILL"))
    check("COST with STILL is still the ST1 condition", p2s_reads_still(v), True)
    check("COST: the onset delta is shown", round(v["report"]["onset_delta_s"], 3), 0.44)
    v = V(_p2s_pt("P2__ST2_s18", s1=still, ws=(0.45, 0.45)))
    check("onset +0.29 s (< 0.3 floor) is no COST", v["verdict"], "STILL")
    v = V(_p2s_pt("P2__ST2_s18", s1=still, field_falls=10))
    check("COST: pooled stand-hold falls 60/1500 = 0.04 vs REFS 0/1500 (> 0 + max(0.03, 2 SE))", v["cost"], ["stand_falls"])
    v = V(_p2s_pt("P2__ST2_s18", s1=dict(still, body_height_m=1.03)))
    check("COST: suite stand body height 0.038 m below REFS (> 0.03)", v["cost"], ["stand_body_height"])
    v = V(_p2s_pt("P2__ST2_s18", s1=dict(still, body_height_m=1.04)))
    check("body height 0.028 m below REFS is no COST", v["verdict"], "STILL")
    v = V(_p2s_pt("P2__ST2_s18", s1=still, ft=0.84))
    check("COST: field_turn completion 0.84 < REFS 0.90 - max(0.05, band 0.04)", v["cost"], ["field_turn_completion"])
    v = V(_p2s_pt("P2__ST2_s18", s1=still, err=0.07))
    check("COST: a CHARTER 6.1 margin (|ratio - 1| +0.04 > max(0.03, band))", v["cost"], ["err_fwd", "err_back",
                                                                                        "err_side", "err_diag"])
    v = V(_p2s_pt("P2__ST2_s18", s1=_p2s_summ(steps=2.5, og=0.33), ws=(0.9, 0.9)))
    check("COST applies on STILL / PARTIAL only (NO EFFECT with a slow onset stays NO EFFECT)", v["verdict"],
          "NO EFFECT")
    v = V(_p2s_pt("P2__ST2_s18", s1=still, fh=0.70))
    check("OBSTACLE LOSS: field_hard 0.70 < REFS 0.81 - max(0.05, 2 SE); the stand outcome still shows",
          (v["verdict"], v["stand"]["outcome"]), ("OBSTACLE LOSS", "STILL"))
    v = V(_p2s_pt("P2__ST2_s18", s1=still, fh=0.77))
    check("field_hard 0.04 below REFS (inside max(0.05, 2 SE)) is no OBSTACLE LOSS", v["verdict"], "STILL")
    v = V(_p2s_pt("P2__ST2_s18", s1=still, fl=(0.75, 0.75)))
    check("OBSTACLE LOSS: field_light 0.05 below REFS (> max(0.03, band 0.04))", v["verdict"], "OBSTACLE LOSS")
    v = V(_p2s_pt("P2__ST2_s18", s1=still, trans=0.90))
    check("FLAT FAIL outranks OBSTACLE LOSS / the stand rung; the stand outcome still shows",
          (v["verdict"], v["stand"]["outcome"]), ("FLAT FAIL", "STILL"))
    bad_ws = _p2s_pt("P2__ST2_s18", s1=still)
    bad_ws["valid"][f"{WS}@1"] = _valid({"KRABBY_FLAT_TERRAIN_MODE": "light"})
    check("INVALID EVAL: a walkstop stamp with a row env block (not the flat row)", V(bad_ws)["verdict"], "INVALID EVAL")
    mode_ws = _p2s_pt("P2__ST2_s18", s1=still)
    mode_ws["ws_env"][2] = {"KRABBY_FLAT_TERRAIN_MODE": "light"}
    check("INVALID EVAL: a walkstop env carrying KRABBY_FLAT_TERRAIN_MODE", V(mode_ws)["verdict"], "INVALID EVAL")
    no_ws2 = _p2s_pt("P2__ST2_s18", s1=still)
    del no_ws2["stamps"][f"{WS}@2"]
    no_ws2["p2s"]["walkstop"][2] = None
    check("PENDING: the walkstop es2 stamp is required", V(no_ws2)["verdict"], "PENDING")
    no_s2 = _p2s_pt("P2__ST2_s18", s1=still)
    no_s2["p2s"]["suite"][2] = None
    check("PENDING: the suite es2 stand read is required (and the stand outcome reads None)",
          (V(no_s2)["verdict"], V(no_s2)["stand"]["outcome"]), ("PENDING", None))
    check("PENDING: --no-tb (no dose read)", V(_p2s_pt("P2__ST2_s18", s1=still, dose=None))["verdict"], "PENDING")
    check("HAND LOSS outranks the stand rung", V(_p2s_pt("P2__ST2_s18", s1=still, hands=("FAST", "SLOW")))["verdict"],
          "HAND LOSS")
    # s16 / s17: PARTIAL against REFS (no same-seed C0R)
    v = p2s_verdict(_p2s_pt("P2__ST2_s16", s1=part), refs, refs, bd, p2a, ancw, bd_a)
    check("s16 / s17: PARTIAL reads against REFS", (v["verdict"], v["comparator"]), ("PARTIAL", "REFS_S1_s18"))
    # attribution at s18
    st_pt = _p2s_pt("P2__ST2_s18", s1=still)
    st_v = V(st_pt)
    check("attribution: STILL, C0R not STILL, no FLAT FAIL, field_hard held -> LEVER",
          p2s_attribution(st_pt, c0r, st_v, bd)["status"], "LEVER")
    check("attribution: C0R itself STILL -> NOT ATTRIBUTED (C0R does not read NO EFFECT)",
          p2s_attribution(st_pt, _p2s_pt("C0R_s18", s1=still), st_v, bd)["lever"], False)
    check("attribution: field_hard 0.10 under C0R -> NOT ATTRIBUTED",
          p2s_attribution(st_pt, _p2s_pt("C0R_s18", fh=0.91, s1=_p2s_summ(steps=2.67)), st_v, bd)["lever"], False)
    check("attribution: NO EFFECT head -> NOT ATTRIBUTED",
          p2s_attribution(st_pt, c0r, dict(st_v, verdict="NO EFFECT", stand={"outcome": "NO EFFECT"}), bd)["lever"],
          False)
    check("attribution: C0R not read -> PENDING", p2s_attribution(st_pt, None, st_v, bd)["status"], "PENDING")
    # the step-7 recipe read (CHARTER 6.2 on the kept dose)
    names = p2s_names({})

    def rv(*vv):
        return p2s_recipe({"P2__ST2_s16": vv[0], "P2__ST2_s17": vv[1], "P2__ST2_s18": vv[2]}, names)["reading"]
    S_, P_, N_ = ({"verdict": "STILL", "gain": True, "cost": []}, {"verdict": "PARTIAL", "gain": True, "cost": []},
                  {"verdict": "NO EFFECT", "gain": False, "cost": []})
    C_ = {"verdict": "COST", "gain": True, "cost": ["onset"]}
    check("recipe: STILL at 3 seeds -> ADOPTABLE (STILL = PASS)", rv(S_, S_, S_), "ADOPTABLE")
    check("recipe: COST on the same metric at 2 seeds -> TRADE-OFF", rv(C_, C_, S_), "TRADE-OFF")
    check("recipe: PARTIAL at 3 seeds -> MIXED (a gain, never a PASS)", rv(P_, P_, P_), "MIXED")
    check("recipe: NO EFFECT at 2 seeds -> NOT ADOPTABLE", rv(N_, N_, S_), "NOT ADOPTABLE")
    check("recipe: a missing replicate -> PENDING", p2s_recipe({"P2__ST2_s18": S_}, names)["reading"], "PENDING")
    check("recipe: the kept dose follows P2S_DOSE",
          p2s_recipe({}, p2s_names({"P2S_DOSE": "ST1"}))["heads"], {16: "P2__ST1_s16", 17: "P2__ST1_s17",
                                                                   18: "P2__ST1_s18"})
    check("p2s heads: fresh names through the environment", p2s_head_names({"P2S_ST2_S18": "P2__ST2R1_s18"}),
          ["P2__ST2R1_s18", "P2__ST1_s18", "P2__ST2_s16", "P2__ST2_s17"])
    check("p2s dose of a relaunch name", [p2s_dose_of("P2__ST2R1_s18"), p2s_dose_of("P2__ST1_s16"),
                                          p2s_dose_of("REFS_S1_s18")], ["ST2", "ST1", None])
    # the step-5 inertness check (like-for-like row evals, suite es1 / es2 included)
    old, new = _p2s_pt("REF_S1_s18", flat_s2=s2), _p2s_pt("REFS_S1_s18", flat_s2=s2)
    for p in (old, new):
        p["valid"] = {k: dict(_valid(), ckpt_sha="h" * 64) for k in p["stamps"]}
        p["stamps"].update({f"{FT}@1": "/x", f"{FS}@1": "/x"})
        p["valid"].update({f"{FT}@1": dict(_valid(), ckpt_sha="h" * 64), f"{FS}@1": dict(_valid(), ckpt_sha="h" * 64)})
        p["feet"] = {"l": 0.23, "r": 0.23}
    check("inertness: identical reads -> ok", p2s_inertness_check({"s18": (new, old)})["ok"], True)
    check("inertness: a suite es2 read differing -> stop",
          p2s_inertness_check({"s18": (dict(new, suite2=dict(s2, err_fwd=0.031)), dict(old, suite2=s2))})["ok"], False)
    check("inertness: a new row (walkstop) is not compared",
          p2s_inertness_check({"s18": (new, dict(old, stamps={k: v for k, v in old["stamps"].items()
                                                                if not k.startswith(WS)}))})["ok"], True)
    check("inertness: the p2s point lacking a p2b row -> stop",
          p2s_inertness_check({"s18": (dict(new, stamps={k: v for k, v in new["stamps"].items() if k != f"{FL}@2"}),
                                       old)})["ok"], False)
    check("inertness: no p2b point (C0 not recorded) -> stop",
          p2s_inertness_check({"c0r": (new, {"name": "P2__C0_s18", "missing": True})})["ok"], False)
    print("p2s self-test:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


# ================================================================================================ CLI
def _f(x, fmt="{:.3f}"):
    return "--" if x is None else fmt.format(x)


def markdown(res: dict, vs: dict) -> str:
    L = ["# Omni phase 2 -- the pre-registered reading (CHARTER 6)", "",
         "| head | verdict | gain | field_hard (pooled) | field_light | ring_obst | why |", "|---|---|---|---|---|---|---|"]
    for ref, v in vs.items():
        pt = res["heads"].get(ref) or {}
        L.append(f"| {ref} | **{v['verdict']}** | {v.get('gain')} | {_f((pooled(pt, 'field_hard') or (None,))[0])} | "
                 f"{_f((pooled(pt, 'field_light') or (None,))[0])} | {_f((pooled(pt, 'ring_obst') or (None,))[0])} | "
                 f"{'; '.join(str(x) for x in (v.get('why') or []))[:300]} |")
    rr = recipe_read({s: vs.get(f"P2__{RECIPE_ARM}_s{s}") for s in SEEDS})
    L += ["", f"**Recipe read (6.2) on {RECIPE_ARM}: {rr['reading']}** "
              f"{json.dumps({k: v for k, v in rr.items() if k != 'reading'})}"]
    pts = {s: res["heads"].get(f"P2__{RECIPE_ARM}_s{s}") for s in SEEDS}
    L += [f"**p2b trigger (6.5) on {RECIPE_ARM}:** "
          f"{json.dumps(p2b_trigger({s: vs.get(f'P2__{RECIPE_ARM}_s{s}') or {} for s in SEEDS}, pts))}"]
    return "\n".join(L) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--json", type=Path)
    ap.add_argument("--md", type=Path)
    ap.add_argument("--no-tb", action="store_true")
    ap.add_argument("--kept", default="")
    ap.add_argument("--anchors-check", action="store_true")
    ap.add_argument("--gate", action="store_true")
    ap.add_argument("--field-vs-ring", action="store_true")
    ap.add_argument("--sibling", choices=SIBLINGS)
    ap.add_argument("--recipe", action="store_true")
    ap.add_argument("--m-screen", metavar="REF")
    ap.add_argument("--calibrate", type=Path)
    ap.add_argument("--smoke-s5", metavar="STAMP|NAME")
    ap.add_argument("--write-smokes-ok", action="store_true")
    ap.add_argument("--smokes", type=Path)
    ap.add_argument("--s5", metavar="STAMP|NAME")
    ap.add_argument("--no-charter-log", action="store_true")
    ap.add_argument("--smokes-ok-check", action="store_true")
    ap.add_argument("--cal2-gate", action="store_true")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--code-root", type=Path, default=R)
    ap.add_argument("--p2b-inertness", action="store_true")
    ap.add_argument("--p2b-screen", action="store_true")
    ap.add_argument("--p2b-recipe", action="store_true")
    ap.add_argument("--p2b-md", type=Path, metavar="PATH")
    ap.add_argument("--p2s-inertness", action="store_true")
    ap.add_argument("--p2s-seeds", default=str(GATE_SEED), metavar="18|16,17")
    ap.add_argument("--p2s-is-still", metavar="NAME")
    ap.add_argument("--p2s-screen", action="store_true")
    ap.add_argument("--p2s-recipe", action="store_true")
    ap.add_argument("--p2s-md", type=Path, metavar="PATH")
    ap.add_argument("--p2s-self-test", action="store_true")
    ap.add_argument("--p2r-probe", action="store_true")
    ap.add_argument("--p2r-step1", action="store_true")
    ap.add_argument("--p2r-drift-json", type=Path, metavar="PATH")
    ap.add_argument("--p2r-md", type=Path, metavar="PATH")
    ap.add_argument("--p2r-z", action="store_true")
    ap.add_argument("--p2r-early-json", type=Path, metavar="PATH")
    ap.add_argument("--p2r-rst", nargs="?", const="", default=None, metavar="NAME")
    ap.add_argument("--p2r-z2", nargs="?", const="", default=None, metavar="NAME")
    ap.add_argument("--probe", default=None, metavar="NAME")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        return _self_test()
    if a.p2s_self_test:
        return _p2s_self_test()
    if a.p2r_probe:
        n = p2r_names()
        refs = load_p2s_point(refs_name(GATE_SEED), with_tb=False)
        # P2R heads are --row-eval only (no record.json): read stamps by name (fable review 2026-10-09)
        probe = load_p2s_point(n["probe"], with_tb=False)
        if not (probe.get("stamps") or {}):
            probe = {"name": n["probe"], "missing": True}
        inert = p2r_launcher_inertness(refs.get("stamps") or {}, probe.get("stamps") or {})
        hold_suite = newest_stamp(n["hold"], SUITE, 1, LROUT)
        swap = p2r_hold_swap_live(refs.get("stamps", {}).get("suite@1"), hold_suite) if hold_suite else None
        # the suite opens every wave with settle (= stand), so its pre-entry proof is vacuous; walkstop (walk, then
        # stand) carries the real "actions before the first stand entry equal REFS" proof
        hold_ws = newest_stamp(n["hold"], WS, 1, LROUT)
        swap_ws = (p2r_hold_swap_live(refs.get("stamps", {}).get(f"{WS}@1"), hold_ws) if hold_ws else None)
        out = {"inertness": inert, "hold_swap_live": swap, "hold_swap_live_walkstop": swap_ws}
        print(json.dumps(out, default=str))
        def _ok(sw):
            return sw is None or (sw["zero_actions_on_stand"] and sw["pre_entry_matches_refs"] and
                                  (sw["stand_mode_share"] or 0) > 0)
        return 0 if inert["ok"] and _ok(swap) and _ok(swap_ws) else 4
    if a.p2r_step1:
        res = load_p2r_step1()
        refs = res["refs"]
        bd = bands(refs, refs)
        sbd = p2s_stand_bands(refs)
        c0r_rec = _rec(P2S_C0_SOURCE)
        c0r = load_p2s_point(C0R, with_tb=False) if c0r_rec else None
        drift_extra = json.loads(a.p2r_drift_json.read_text()) if a.p2r_drift_json else None
        out = {"refs": refs.get("name"), "heads": {}}
        for key in ("hold", "st2m"):
            nm = res["names"][key]
            pt = res["heads"].get(nm) or {"name": nm, "missing": True}
            if pt.get("missing"):
                out["heads"][nm] = {"name": nm, "verdict": "PENDING",
                                    "why": [f"no record ({pt.get('killed') or 'not run'})"], "survives_tier1": None}
                continue
            cmp = c0r if c0r else refs
            v = p2r_tier1_verdict(pt, refs, cmp, bd, sbd)
            v["survives_tier1"] = p2r_survives_tier1(v)
            v["hold_reads"] = p2r_hold_reads(pt.get("stamps", {}).get("suite@1"))
            v["settle_mode_share_by_wave"] = p2r_settle_share(pt.get("stamps", {}).get("suite@1"))
            v["turn_final_heading"] = p2r_turn_final_heading(pt.get("stamps", {}).get(f"{FT}@1"))
            v["turn_final_heading_ref"] = p2r_turn_final_heading(refs.get("stamps", {}).get(f"{FT}@1"))
            out["heads"][nm] = v
        probe_rec = _rec(res["names"]["probe"])
        probe_stamps = []
        if probe_rec:
            probe_pt = load_point(res["names"]["probe"], suite_stamp=probe_rec.get("driver_stamp"), with_tb=False)
            probe_stamps = [probe_pt.get("stamps", {}).get(k) for k in P2R_PROBE_STAMPS]
        out["drift_premise"] = (p2r_drift_premise(probe_stamps, drift_json=drift_extra) if probe_stamps
                                else {"missing": f"no {res['names']['probe']} record"})
        if a.p2r_md:
            a.p2r_md.write_text(p2r_step1_markdown(out))
        print(json.dumps(out, default=str))
        return 0
    if a.p2r_rst is not None:
        # NOTE(p2r Z2, 2026-10-10): the step-3 premise gate -- exit 0 on PASS / PASS ON ROWS RUN, else 4
        out = p2r_rst_report(a.p2r_rst or None)
        if a.p2r_md:
            a.p2r_md.write_text(p2r_rst_markdown(out))
        print(json.dumps(out, default=str))
        return 0 if out["gate"].get("verdict") in ("PASS", "PASS ON ROWS RUN") else 4
    if a.p2r_z2 is not None:
        # NOTE(p2r Z2, 2026-10-10): the step-5 read -- exit 0 unless INVALID (every verdict stops and asks), as --p2r-z
        out = p2r_z2_report(name=a.p2r_z2 or None, probe_name=a.probe)
        if a.p2r_md:
            a.p2r_md.write_text(p2r_z2_markdown(out))
        print(json.dumps(out, default=str))
        return 0 if out["verdict"].get("verdict") not in ("INVALID EVAL",) else 4
    if a.p2r_z:
        early = json.loads(a.p2r_early_json.read_text()) if a.p2r_early_json else None
        out = p2r_z_report(early_json=early)
        if a.p2r_md:
            a.p2r_md.write_text(p2r_z_markdown(out))
        print(json.dumps(out, default=str))
        return 0 if out["verdict"].get("verdict") not in ("INVALID EVAL",) else 4
    if a.p2s_inertness:
        seeds = tuple(int(x) for x in a.p2s_seeds.split(",") if x.strip())
        r = p2s_inertness_check(load_p2s_inertness(seeds))
        print(json.dumps(r, default=str))
        return 0 if r["ok"] else 4
    if a.p2s_is_still or a.p2s_screen or a.p2s_recipe or a.p2s_md:
        names = p2s_names()
        seeds = SEEDS if (a.p2s_recipe or a.p2s_md) else (GATE_SEED,)
        if a.p2s_is_still and not (a.p2s_screen or a.p2s_recipe or a.p2s_md):
            try:
                seeds = (int(a.p2s_is_still.rsplit("_s", 1)[1]),)
            except (IndexError, ValueError):
                print(json.dumps({"name": a.p2s_is_still, "still": False, "why": "no _s<seed> in the name"}))
                return 4
        res = load_p2s(with_tb=not a.no_tb, seeds=seeds, names=names)
        vs = p2s_verdicts(res)
        if a.p2s_is_still and not (a.p2s_screen or a.p2s_recipe or a.p2s_md):
            v = vs.get(a.p2s_is_still)
            still = p2s_reads_still(v)
            print(json.dumps({"name": a.p2s_is_still, "still": still, "verdict": (v or {}).get("verdict"),
                              "stand": ((v or {}).get("stand") or {}).get("outcome")}, default=str))
            return 0 if still else 4
        scr = p2s_screen(res, vs) if GATE_SEED in seeds else {"attribution": {}, "c0r_clauses": {}}
        recipe = p2s_recipe(vs, names) if set(SEEDS) <= set(seeds) else None
        if a.p2s_md:
            a.p2s_md.write_text(p2s_markdown(vs, scr["attribution"], recipe, scr["c0r_clauses"]))
        if a.p2s_screen:
            print(json.dumps(scr, default=str))
            return 0
        if a.p2s_recipe:
            print(json.dumps({"dose": names["dose"], "recipe": recipe,
                              "verdicts": {k: v["verdict"] for k, v in vs.items()}}, default=str))
            return 4 if recipe["reading"] == "PENDING" else 0
        print(p2s_markdown(vs, scr["attribution"], recipe, scr["c0r_clauses"]))
        return 0
    out = a.out or OUT
    arms_dir, lr_dir = out / "arms", out / "lr"
    if a.smoke_s5:
        r = smoke_s5_read(resolve_stamp(a.smoke_s5, FL, 1, lr_dir), _anchors_yaml()[GATE_SEED]["sha256"])
        print(json.dumps(r, default=str))
        return 0 if r["ok"] else 4
    if a.write_smokes_ok:
        if not a.smokes or not a.s5:
            ap.error("--write-smokes-ok needs --smokes DIR and --s5 STAMP|NAME")
        s5 = smoke_s5_read(resolve_stamp(a.s5, FL, 1, lr_dir), _anchors_yaml()[GATE_SEED]["sha256"])
        import yaml  # noqa: PLC0415
        fails, payload = build_smokes_ok(a.smokes, s5, _json(arms_dir / f"{CAL1}.record.json"),
                                         _live_code_state(a.code_root),
                                         yaml.safe_load(open(CAMPAIGN / "candidates.yaml")))
        if fails:
            print(json.dumps({"ok": False, "fails": fails}, default=str))
            return 4
        out.mkdir(parents=True, exist_ok=True)
        (out / SMOKES_OK).write_text(json.dumps(payload, indent=1, default=str))
        if not a.no_charter_log:
            with open(CAMPAIGN / "CHARTER.md", "a") as fh:
                fh.write(charter_line(payload))
        print(json.dumps({"ok": True, "written": str(out / SMOKES_OK), **payload}, default=str))
        return 0
    if a.smokes_ok_check:
        import yaml  # noqa: PLC0415
        payload = _json(out / SMOKES_OK)
        fails = smokes_ok_fails(payload, _live_code_state(a.code_root), yaml.safe_load(open(CAMPAIGN / "candidates.yaml")))
        print(json.dumps({"ok": not fails, "fails": fails, "smokes_ok": str(out / SMOKES_OK)}, default=str))
        return 0 if not fails else 4
    if a.cal2_gate:
        import yaml  # noqa: PLC0415
        r = cal2_gate(_json(arms_dir / f"{CAL2}.record.json"), yaml.safe_load(open(CAMPAIGN / "candidates.yaml")))
        print(json.dumps(r, default=str))
        return 0 if r["ok"] else 4
    if a.calibrate:
        tb = a.calibrate.is_dir() or a.calibrate.name.startswith("events.out.tfevents")
        r = calibrate_run(a.calibrate) if tb else calibrate_file(a.calibrate)
        print(json.dumps(r))
        return 0 if r.get("ok") else 2
    if a.anchors_check:
        an = _anchors_yaml()
        anc = {int(s): load_point(x["name"], with_tb=False) for s, x in an.items()}
        r = anchors_check(anc, _manifest_s18(), {int(s): x["sha256"] for s, x in an.items()})
        print(json.dumps(r, default=str))
        return 0 if r["ok"] else 4
    if a.m_screen:
        an = _anchors_yaml()
        s = int(a.m_screen.rsplit("_s", 1)[1])
        anc = load_point(an[s]["name"], with_tb=False)
        s18 = anc if s == GATE_SEED else load_point(an[GATE_SEED]["name"], with_tb=False)
        rec = _rec(a.m_screen) or {}
        head = load_point(a.m_screen, suite_stamp=rec.get("driver_stamp"), with_tb=False)
        ms = {it: load_point(f"{a.m_screen}_m{it}", with_tb=False) for it in MCKPTS}
        r = m_screen(head, ms, anc, bands(anc, s18))
        for it, v in r.items():
            if v["status"] == "NEEDS_SEED2":
                print(f"{a.m_screen}_m{it}")
        print(json.dumps(r, default=str), file=sys.stderr)
        return 0
    if a.p2b_inertness:
        refs = {s: load_point(ref_name(s), with_tb=False, fh_seeds=P2B_FH_SEEDS) for s in SEEDS}
        p2a = {s: load_point(f"P2__{REF_ARM}_s{s}", with_tb=False) if _rec(f"P2__{REF_ARM}_s{s}") else {} for s in SEEDS}
        r = inertness_check(refs, p2a)
        print(json.dumps(r, default=str))
        return 0 if r["ok"] else 4
    if a.p2b_screen or a.p2b_recipe or a.p2b_md:
        res = load_p2b(with_tb=not a.no_tb)
        vs = p2b_verdicts(res)
        attr, recipes = p2b_attributions(res, vs), p2b_recipes(vs)
        md = p2b_markdown(vs, attr, recipes)
        if a.p2b_md:
            a.p2b_md.write_text(md)
        if a.p2b_screen:
            print(json.dumps({"verdicts": {k: v for k, v in vs.items() if k.endswith(f"_s{GATE_SEED}")},
                              "attribution": attr}, default=str))
            return 0
        if a.p2b_recipe:
            print(json.dumps({"levers": list(p2b_levers()), "recipe": recipes,
                              "verdicts": {k: v["verdict"] for k, v in vs.items()}}, default=str))
            return 4 if any(r["reading"] == "PENDING" for r in recipes.values()) else 0
        print(md)
        return 0
    with_raw = bool(a.sibling or a.field_vs_ring or a.json or a.md)
    res = load_all(with_tb=not a.no_tb, with_raw=with_raw)
    vs = verdicts(res)
    if a.gate:
        v = vs.get(f"P2__A1_s{GATE_SEED}")
        print(json.dumps(v, default=str))
        return 0 if v and v["verdict"] == "PASS" else 4
    if a.sibling:
        sib, a1 = res["heads"].get(f"P2__{a.sibling}_s18"), res["heads"].get("P2__A1_s18")
        anc = res["anchors"][GATE_SEED]
        if not sib or sib.get("missing") or not a1 or a1.get("missing"):
            print(json.dumps({"arm": a.sibling, "rec": "PENDING", "why": "sibling or A1 s18 not recorded"}))
            return 4
        r = sibling_rec(a.sibling, sib, a1, bands(anc, anc))
        r["seed_verdict_vs_anchor"] = vs.get(f"P2__{a.sibling}_s18", {}).get("verdict")
        print(json.dumps(r, default=str))
        return 4
    if a.field_vs_ring:
        a1, a0 = res["heads"].get("P2__A1_s18") or {}, res["heads"].get("P2__A0_s18") or {}
        out = {}
        for lab, pt in (("A1", a1), ("A0", a0)):
            out[lab] = {k: pooled(pt, k) for k in ("field_hard", "ring_obst")} if not pt.get("missing") else None
            if out[lab] is not None:
                out[lab]["falls_per_ep"] = {k: (sum((pt["comp"][k][s] or {}).get("falls", 0) for s in (1, 2)) /
                                                max(1, sum((pt["comp"][k][s] or {}).get("n", 0) for s in (1, 2))))
                                            for k in ("field_hard", "ring_obst")}
                out[lab]["exposure"] = pt.get("exposure")
                out[lab]["run_dir"] = pt.get("run_dir")
                out[lab]["terrain_levels"] = terrain_levels(pt.get("run_dir")) if not a.no_tb else None
        print(json.dumps(out, default=str))
        return 0
    if a.recipe:
        r = recipe_read({s: vs.get(f"P2__{RECIPE_ARM}_s{s}") for s in SEEDS})
        t = p2b_trigger({s: vs.get(f"P2__{RECIPE_ARM}_s{s}") or {} for s in SEEDS},
                        {s: res["heads"].get(f"P2__{RECIPE_ARM}_s{s}") for s in SEEDS})
        print(json.dumps({"recipe_arm": RECIPE_ARM, "recipe": r, "p2b": t}, default=str))
        return 4 if (t["trigger"] or r["reading"] == "PENDING") else 0
    kept = [k for k in a.kept.split(",") if k]
    cands, cand_vs, ms_reads = charter7_candidates(res, vs, kept)
    rk = rank_inputs(cands, cand_vs)
    md = markdown(res, vs) + "\n**CHARTER 7 candidates:** " + ", ".join(cands) + "\n\n**m-screen (CHARTER 7):** " + \
        json.dumps({lab: {it: v.get("status") for it, v in r.items()} for lab, r in ms_reads.items()}) + "\n"
    if a.json:
        a.json.write_text(json.dumps({"verdicts": vs, "rank_inputs": rk, "m_screen": ms_reads}, indent=1, default=str))
    if a.md:
        a.md.write_text(md)
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
