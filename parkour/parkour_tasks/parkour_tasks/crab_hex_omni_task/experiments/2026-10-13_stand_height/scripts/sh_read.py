#!/usr/bin/env python3
"""Stand-height campaign (``2026-10-13_stand_height``) -- the K1-K6 READERS (PLAN.md H7 + the control table, step 2;
CPU only, read-only). Reads the ``stand_height`` launcher's height-row stamps (raws), the table / ground test's per-cell
JSON and -- for K5 -- the predecessor's ``sn_calib`` reads UNCHANGED; writes one JSON and one markdown file to the paths
given by ``--json`` / ``--md`` (no defaults; refused inside the predecessor or phase-2 campaign directories).

IMPORTS. Top level: the stdlib and the campaign's eval contract ``sh_eval_common`` (stdlib + ``sh_core``, torch-free).
numpy, torch, ``sn_calib`` (-> ``p2_read`` / ``p2s_stand`` / ``gait_eval.metrics``), ``sh_table_core``, ``sh_facts``
and ``sh_kin`` load lazily, by path, without bytecode.

ONE DEFINITION EACH (never restated here): every raw name, the run_meta block, the controller / command / restore
vocabularies and the height-row manifest = ``sh_eval_common`` (the launcher's contract); h_meas = ``sh_core.h_meas``
(root z minus the mean terrain z under the six feet, on the harness's own buffers); hip rods = ``sh_core.hip_rod_dev``
on the raw hip-pitch angles; the H6 completion = ``sh_core``'s rule (the limited command at 1.064 on the previous row
AND every hip rod within ``RESTORE_ROD_TOL_M`` on ``settle_steps`` consecutive rows); homed / settled / six-down / hip
yaw = ``p2s_stand.stand_z2_metrics``; steps / taps = ``gait_eval.metrics.step_tap_runs`` (loaded = > 50 N); onset =
``p2s_stand.episode_onset_latency``; REFS's onset = ``p2_read.walkstop_read``; K3 STOOD, the K3 cells, pooling, line
and the R_h floor = ``sh_table_core`` (``classify`` per env in the test; ``pooled_views`` / ``k3_line`` / ``rh_floor``
here); the settled force rows = ``sh_facts._settled_rows`` (the step-1 rule); the force conversion =
``sh_kin.stance_joint_torques`` + ``sh_kin.rod_forces``; the clause / verdict combiner = ``sn_calib._cl`` /
``sn_calib.verdict_of``; K5 = ``sn_calib.read_reference`` / ``merge_r`` / ``calibrate`` / ``stand_net_consistency``.

INPUTS (a spec JSON, :func:`load_spec`): per role a stamp NAME (resolved as the phase-2 readers do: the newest complete
``<lr>/<NAME>/<row>/seed00<k>/*``) or explicit stamp dirs per row and seed, the table test's cell JSON (files or dirs)
per seed, and for a head its zero-command K5 name. Height rows and the gates each feeds come from the schedule manifest
``scenarios_sh.yaml`` (``stand_height.rows.<row>.gates``; the stamp's own recorded ``heights.gates`` must agree):
``hwalkstop__omni`` -> K1 / K4 / K6, ``hraise__omni`` -> K2 / K4.

STAMP VALIDITY (fail closed: any reason -> the stamp is INVALID for every gate it feeds; :func:`meta_reasons`,
:func:`episode_reasons`): ``p2r_mode.variant`` = ``stand_height``; its ``stand_height`` block (the launcher's
``block_record``) with ``command`` ``manifest``; ``hold_heights`` one entry per hold (a height in [H_MIN, H_MAX] on
every hold labelled "stand" -- speed 0 --, null on every other hold); a ``height_row`` record of THIS row (scenario,
labels == run_meta.hold_labels, the same hold_heights, ``gates`` == the manifest's); the ``controller`` record
(``HeightController.record()``) equal to sh_core's constants (rate, range, rod tolerance, settle steps, dt; engage
0.02 m when armed, null when off); ``restore`` / ``control`` / checkpoint kind matching the ROLE (only ``no_restore``
runs with the restore off); ``h_meas.valid`` (the raycast mesh, not the harness's ``root_z - 1.05`` fallback); a flat
row (no row env block, no KRABBY_FLAT_TERRAIN_MODE); every episode carries :data:`HEIGHT_RAW_REQUIRED` and its raw
command on every stand-slot row of a stand hold equals that hold's height. A head's stamps (height rows, K5 rows, table
cells) must all name ONE stand checkpoint AND ONE residual deadband (CHARTER 2026-10-14 probe; ``sh_core.
recorded_deadband``: a stamp / cell written without the key ran at 0; a malformed record makes its stamp INVALID; a mix
across a head's ok height stamps, gated table cells and K5 stamps makes every gate of that head INVALID, like the
one-checkpoint rule; the value is recorded in the head's output -- controls are not held to it). The deadband the
stand checkpoint was TRAINED with (``trained_residual_deadband``, recorded by the launcher; absent / null = not
recorded; malformed = the stamp INVALID) is compared per stamp (``sh_core.deadband_parity``): a difference is a NOTE in
the head's output (``deadband_notes``, with both values), never a refusal (the step-6 probe's mismatch is intended;
outside a probe it is a config-parity break). The launcher's full
audit (clip, swaps, R_h branches, restore lifecycle, manifest drift, ``sh_stand_eff == p2r_stand_mode |
sh_restoring``, the executed action at the recorded deadband) is ``sh_bridge_read.py``'s, not repeated here.
Raws: the latched stand SLOT and the effective stand vector (who acted) under the contract in force
(:func:`raw_slot_eff_keys`); the reader uses stand_eff = slot | ``sh_restoring`` and refuses a disagreeing record.

THE GATES (PLAN H7; statistics pinned here per the audit's "left for implementation" list; env seeds es1 AND es2):
  bins     ``K_BINS`` = 5 equal bins over the frozen [H_MIN, H_MAX] by the hold's COMMANDED height.
  K1       per bin per seed, every stand hold of the K1 rows: command arrival = the first hold row whose LIMITED command
           equals the hold's; window = the hold's rows from max(hold end - 2 s, arrival + 1 s); the |h_meas - h_cmd|
           samples of every hold of the bin POOLED: median <= 0.02 m AND p90 <= 0.04 m. A reached hold without a window
           (never arrived, or arrival + 1 s after the hold's end) or with a fall in it adds a full 2-s window of +inf
           (fail closed). A bin with no hold: PENDING.
  K2       events = every stand-slot row of the K2 rows where the raw command changes (a stand entry from walking
           included) by dh = |h_cmd - h_lim(previous row)| > 0.02 m; time to 2 cm = the first row from which
           |h_meas - h_cmd| <= 0.02 holds 0.5 s running inside that command segment; excess = time - (dh / 0.08 + 1 s);
           a fall in the segment or a never-reached event = +inf; a segment that ends before dh / 0.08 + 1.5 s without
           reaching = "short" (excluded, reported). Per seed: p90(excess) <= 0 (the pinned p90 aggregate). Falls:
           episodes with crab_failure on a stand-slot row, a raw count per seed <= max(1, R_h's) on R_h's episode count
           and schedule (INVALID otherwise).
  K3       ``sh_table_core`` on the test's cell JSON (re-scoped at the step-4 stop, CHARTER section 3 "narrow stand-up
           to tall tables"): the 6 GATED cells ``K3_GATED_CELLS`` (tops 0.635 / 0.6985 / 0.762 x knee_folded /
           random), every one present with n >= ``ENVS_PER_CELL`` per seed (else PENDING); per seed the gated start
           type ``tables`` (the 6 cells pooled): ``k3_line`` = STOOD rate >= R_h's - max(5 pp, 2 SE of the
           difference) AND falls per 100 <= max(1, R_h's). A gated cell's stored aggregate must agree with its
           per-env records, table commands >= top + 0.31 m (else INVALID). REPORTED cells (the 18-in table, every
           rod_min cell, every ground cell) are read when present, never required, never a clause (their contract
           problems are notes): per cell / top / ground pose rate, falls, 5- / 6-down, homed, and the ground line.
           R_h premise (step-4 STOP): ``rh_floor`` on the gated cells only (STOOD >= 0.90, falls <= 1 / 100 on each
           gated top pooled over the gated poses).
  K4       over the K4 rows: hand-back = the first S1 row (stand_eff = slot | restoring false) at or after a
           slot exit and before the next stand entry (none: a fall, a stand -> stand MERGE, or a cut); RESTORE-REQUIRED
           when the last stand row's limited command is > 0.02 m from 1.064. MISS = a restore-required hand-back whose
           H6 completion did not hold at the hand-back (rows t - settle + 1 .. t: h_lim(k - 1) == 1.064 and every hip
           rod within delta) -- a timeout, or no restore at all -- or one cut by the episode's end. Per seed: misses <=
           max(1, R_h's); exit falls (crab_failure from the slot exit to hand-back + 2 s) <= max(1, R_h's) (raw counts
           on R_h's episodes and schedule); onset excess = onset (from the moving hold's first row) - (REFS's walkstop
           onset median, same seed, + dh / 0.08 + 1.5 s), censored = +inf, p90 <= 0. Reported: hand-back
           |h_meas - 1.064| (flat ground), restore time, timeouts, merges, cuts.
  K5       sn_calib (unchanged reads and lines) G1-G5 on the head's zero-command stamps, read with its ``obs0`` validity
           kind (D5 obs parity + REFS's ENV4 / row env / code-state rules; its ``stand_net`` kind hard-codes the
           predecessor's variant) plus this campaign's rule: variant ``stand_height``, command ``zero``, restore armed,
           controller ``stand_net``, REFS's S1, one stand checkpoint and mask across the head's stamps
           (``sn_calib.stand_net_consistency``), and every raw row commanding 1.064. Controls: R_h = sn_calib's R row (R
           at 1.064 is R_h bitwise), A-mask s17 = its SN_Amask_s17 head (``stand_net`` kind, its own stamps).
  K6       walk-handover entries (stand holds right after a walk hold) of the K6 rows, per bin per seed, homed and
           settled share (p2s_stand) >= R_h's - max(5 pp, 2 SE); stillness: per command segment (a stand-slot run of one
           command; the window RE-OPENS after an in-hold command change) the window starts once |h_meas - h_cmd| <= 0.02
           has held 0.5 s AND, from there, the cams have settled (homed held 0.5 s); steps / taps POOLED over all bins
           per seed as sum(events) / sum(window seconds) <= p2_read.Z2_POST (0.5 / 1.0 per s); every bin reported;
           no open window at all = fail closed.
  forces   (reported; H2) settled stand rows (sh_facts._settled_rows) binned by h_meas: six-leg mean |F_hip| from the
           MEASURED FOOT LOADS through the linkage (the user's cap statistic: foot loads vertical at the toe + leg
           weight) beside the sim PD-torque reading, p90, worst leg, knee (foot-load and PD). h_max re-check: R_h's top
           bin (the one holding H_MAX), es1 + es2 pooled, foot-load statistic <= 2000 N, else a re-freeze STOP;
           unreadable force raws -> INVALID (a STOP).

CONTROL TABLE (step 4: known-good must PASS, known-bad must FAIL -- on its named clause where the plan names one, at
every seed read -- INVALID on a listed cell is a mismatch): :data:`EXPECT`, :data:`NAMED_CLAUSE`,
:data:`REPORTED_ROLES`.

    sh_read.py --spec SPEC.json --json OUT.json --md OUT.md [--no-k5] [--no-forces]
exit 0 clean / 2 pending only / 3 a head fails or is INVALID / 4 a control mismatch, an R_h premise miss, or an h_max
re-freeze / unreadable re-check (step 4: STOP, notify, ask).
"""
from __future__ import annotations

import argparse
import datetime
import glob
import hashlib
import importlib.util
import json
import math
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CAMPAIGN = HERE.parent
TASK = HERE.parents[2]
EXPERIMENTS = TASK / "experiments"
PRED_DIR = EXPERIMENTS / "2026-10-11_stand_network"
P2_DIR = EXPERIMENTS / "2026-10-07_omni_phase2"
SN_CALIB_PATH = PRED_DIR / "scripts" / "sn_calib.py"
EC_PATH = HERE / "sh_eval_common.py"
TABLE_CORE_PATH = HERE / "sh_table_core.py"
SH_FACTS_PATH = HERE / "sh_facts.py"
#: outputs are refused under these (the closed predecessor; phase 2)
FORBIDDEN_OUT_DIRS = (PRED_DIR, P2_DIR)


def _load_by_path(path: Path, name: str):
    """A private module instance loaded by path (no ``__pycache__`` written)."""
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:  # pragma: no cover
        raise ImportError(f"sh_read: cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    old = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.dont_write_bytecode = old
    return mod


#: the launcher's eval contract (names, vocabularies, the manifest loader) and, through it, the ONE core
EC = _load_by_path(EC_PATH, "_sh_read_eval_common")
CORE = EC.CORE
H_SETTLED, H_MIN, H_MAX = CORE.H_SETTLED, CORE.H_MIN, CORE.H_MAX

# ===================================================================================== constants
PASS, FAIL, PENDING, INVALID, PARTIAL = "PASS", "FAIL", "PENDING", "INVALID", "PARTIAL"
GATES = ("K1", "K2", "K3", "K4", "K5", "K6")
RAW_GATES = ("K1", "K2", "K4", "K6")
GATE_TITLE = {"K1": "TRACKING", "K2": "RAISE / LOWER", "K3": "STAND-UP", "K4": "RESTORE + HAND-BACK",
              "K5": "NON-REGRESSION @1.064", "K6": "HOMING + STILLNESS"}
SEEDS = (1, 2)
EPS = 1e-9
REFS_NAME = "REFS_S1_s18"              #: K4's onset line source (its walkstop__omni stamps) and K5's S1
K_BINS = 5
BIN_EDGES = tuple(H_MIN + (H_MAX - H_MIN) * i / K_BINS for i in range(K_BINS + 1))
CMD_TOL = 1.0e-6                       #: a limited command "equals" a target (float64 limiter)
HOLD_HEIGHT_TOL = 1.0e-6               #: sh_h_cmd vs the stamp's per-hold heights
TRACK_TOL_M = 0.02                     #: K1 / K2 / K6 "within 2 cm" and K4 restore-required
REACH_HOLD_S = 0.5                     #: K2 / K6: within 2 cm must hold this long (the campaign's settle convention)
K1_MED_M, K1_P90_M = 0.02, 0.04
K1_WINDOW_S, K1_ARRIVAL_DELAY_S = EC.K1_WINDOW_S, EC.ARRIVAL_SETTLE_S
K2_EXTRA_S = 1.0
K2_Q = K4_Q = 90.0                     #: the pinned aggregates (audit "left for implementation": p90)
FALL_FLOOR = 1                         #: "<= max(1, R_h's)"
K4_EXIT_FALL_S = 2.0
K4_ONSET_EXTRA_S = EC.K4_ONSET_SLACK_S
SHARE_PP, SHARE_SE_K = 0.05, 2.0       #: K6 "R_h's - max(5 pp, 2 SE of the difference)"
FORCE_CAP_N = 2000.0
#: the h_max re-check's REPORTED companion statistic: settled R_h rows within this of h_max (the 1.346 holds); the
#: pre-registered decision statistic stays the top-bin mean (CHARTER 2026-10-13 rule restated before its read)
AT_HMAX_TOL_M = 0.02
LEGS = ("FL", "FR", "ML", "MR", "RL", "RR")

# raw / meta contract (sh_eval_common's names)


def raw_slot_eff_keys() -> tuple[str, str]:
    """``(latched-slot key, recorded effective-stand key)`` under the launcher contract in force (``sh_eval_common``):
    a dedicated ``RAW_STAND_SLOT`` with P2L's ``p2r_stand_mode`` = slot | restoring, or ``p2r_stand_mode`` = the
    slot with a dedicated ``RAW_STAND_EFF``. Either way the reader derives stand_eff = slot | restoring (sh_core's
    definition) and refuses an episode whose recorded effective vector disagrees."""
    slot = getattr(EC, "RAW_STAND_SLOT", None)
    if slot:
        return slot, EC.RAW_STAND_MODE
    return EC.RAW_STAND_MODE, EC.RAW_STAND_EFF


RAW_SLOT, RAW_EFF = raw_slot_eff_keys()
STAND_LABEL = getattr(EC, "STAND_LABEL", "stand")
RAW_H_CMD, RAW_H_LIM, RAW_RESTORING = EC.RAW_H_CMD, EC.RAW_H_LIM, EC.RAW_RESTORING
RAW_Q, RAW_A_NET, RAW_TIMED_OUT, RAW_MERGED = EC.RAW_Q, EC.RAW_A_NET, EC.RAW_TIMED_OUT, EC.RAW_MERGED
RAW_COMPLETED = EC.RAW_COMPLETED
HARNESS_KEYS = ("root_pos_w", "terrain_z", "foot_force_norm", "foot_pos_w", "foot_lin_vel_w", "joint_pos",
                "crab_failure", "done", "segment_id", "steady_mask", "cmd_applied", "root_lin_vel_b", "heading_err")
HEIGHT_RAW_REQUIRED = HARNESS_KEYS + (RAW_SLOT, RAW_EFF, RAW_H_CMD, RAW_H_LIM, RAW_RESTORING)
RESTORE_WORDS = {"armed": True, "off": False}
HEIGHT_RAW_REPORTED = (RAW_Q, RAW_A_NET, RAW_TIMED_OUT, RAW_MERGED, RAW_COMPLETED)
#: K4: the launcher's recorded H6 events (sh_completed / sh_timed_out at the hand-back row) DECIDE complete / miss;
#: the reader's own joint-space re-derivation is the cross-check -- a disagreement beyond this rod-length band (float32
#: joint angles vs the obs column) marks the episode INVALID (fail closed), never a silent "miss"
K4_XCHECK_TOL_M = 1.0e-6
FORCE_RAW_REQUIRED = ("applied_torque", "joint_vel", "rod_speed")

# the control table (PLAN H7)
CONTROL_ROLES = ("R_h", "R_old", "Amask_old", "posture", "Amask_q", "S1", "no_restore", "Amask")
ROLE_TITLE = {"R_h": "R_h (height-aware R)", "R_old": "R on the old +-1 path (no q)",
              "Amask_old": "A-mask s17 (W15 zeroed) on the old +-1 path (no q)",
              "posture": "posture map only (zero residual)", "Amask_q": "A-mask s17 + q", "S1": "S1 alone",
              "no_restore": "hand back at height, no restore", "Amask": "A-mask s17"}
EXPECT = {("K1", "R_h"): PASS, ("K1", "R_old"): FAIL, ("K1", "Amask_old"): FAIL,
          ("K2", "R_h"): PASS, ("K2", "R_old"): FAIL, ("K2", "Amask_old"): FAIL,
          ("K3", "R_h"): PASS, ("K3", "S1"): FAIL,
          ("K4", "R_h"): PASS, ("K4", "no_restore"): FAIL,
          ("K5", "R_h"): PASS, ("K5", "Amask"): PASS,
          ("K6", "R_h"): PASS, ("K6", "posture"): FAIL}
#: a known-bad's parenthetical: the gate must fail ON this clause (short name) at EVERY seed read
NAMED_CLAUSE = {("K4", "no_restore"): "miss", ("K6", "posture"): "homed"}
REPORTED_ROLES = {"K1": ("posture", "Amask_q"), "K2": ("posture", "Amask_q"), "K3": ("posture", "Amask"),
                  "K4": ("Amask",), "K5": (), "K6": ("Amask",)}
#: the height launcher's controller per role (heads: stand_net); no_restore: any controller, restore off
ROLE_CONTROLLER = {"R_h": ("r_h",), "R_old": ("r_old",), "Amask_old": ("amask_old",), "posture": ("posture_only",),
                   "Amask_q": ("stand_net",), "Amask": ("stand_net",), "no_restore": tuple(EC.CONTROLLERS)}
ROLE_CKPT_KIND = {"Amask_q": EC.CKPT_KIND_AMASK, "Amask": EC.CKPT_KIND_AMASK, "Amask_old": EC.CKPT_KIND_AMASK}
HEAD_CONTROLLER, HEAD_CKPT_KIND = "stand_net", EC.CKPT_KIND_TRAINED
#: the table test's controller per role (heads: stand_net)
ROLE_TABLE_CONTROLLER = {"R_h": "r_h", "S1": "s1", "posture": "posture_only", "Amask": "stand_net",
                         "Amask_q": "stand_net", "Amask_old": "amask_old"}
SN_CALIB_R_ROW = "R"                   #: K5: R_h = sn_calib's R composite (P2R_RSTg_s18 + SN_RSTg_s18 field_hard)
SN_AMASK_HEAD = "SN_Amask_s17"
K5_HEAD_KIND = "obs0"                  #: sn_calib validity kind for the zero-command stand_height stamps


# ===================================================================================== lazy loaders
_CACHE: dict = {}


def _np():
    import numpy as np  # noqa: PLC0415
    return np


def _torch():
    import torch  # noqa: PLC0415
    return torch


def _snc():
    """The predecessor's ``sn_calib`` (by path; it imports ``p2_read`` read-only and sets dont_write_bytecode)."""
    if "snc" not in _CACHE:
        _CACHE["snc"] = _load_by_path(SN_CALIB_PATH, "_sh_read_sn_calib")
    return _CACHE["snc"]


def _P():
    return _snc().P


def _S():
    return _snc().S


def _tc():
    """The table / ground test's pure pieces (``sh_table_core``: the K3 definition)."""
    if "tc" not in _CACHE:
        _CACHE["tc"] = _load_by_path(TABLE_CORE_PATH, "_sh_read_table_core")
    return _CACHE["tc"]


def _facts():
    """``sh_facts`` (the step-1 settled-row rule); it also imports ``sh_kin`` and ``sn_table_core``."""
    if "facts" not in _CACHE:
        _CACHE["facts"] = _load_by_path(SH_FACTS_PATH, "_sh_read_facts")
    return _CACHE["facts"]


def _cl(value, line, ok, state="eval", seed=None, partial=False, short=None, note=None) -> dict:
    """One clause (``sn_calib._cl``)."""
    return _snc()._cl(value, line, ok, state, seed, partial, short, note)


def verdict_of(clauses: dict, invalid: list) -> dict:
    """INVALID > FAIL > PENDING > PARTIAL > PASS (``sn_calib.verdict_of``)."""
    return _snc().verdict_of(clauses, invalid)


# ===================================================================================== small pure helpers
def height_bin(h) -> int | None:
    """Bin index of a commanded height (5 equal bins over [H_MIN, H_MAX], the top edge closed); None outside."""
    h = float(h)
    if not (H_MIN - EPS <= h <= H_MAX + EPS):
        return None
    i = int(math.floor((h - H_MIN) / (H_MAX - H_MIN) * K_BINS))
    return min(max(i, 0), K_BINS - 1)


def bin_label(b: int) -> str:
    return f"{BIN_EDGES[b]:.3f}-{BIN_EDGES[b + 1]:.3f}"


def pctl(vals, q: float):
    """numpy's linear-interpolation percentile where +inf is a legal (fail-closed) sample: an interpolation touching
    +inf returns +inf. None for no samples."""
    np = _np()
    v = np.sort(np.asarray(list(vals) if not hasattr(vals, "shape") else vals, dtype=np.float64).reshape(-1))
    if v.size == 0:
        return None
    if np.isnan(v).any():
        raise ValueError("pctl: NaN sample")
    pos = (v.size - 1) * float(q) / 100.0
    lo, hi = int(math.floor(pos)), int(math.ceil(pos))
    frac = pos - lo
    if hi == lo or frac == 0.0:
        return float(v[lo])
    if math.isinf(v[hi]):
        return math.inf
    return float(v[lo] + (v[hi] - v[lo]) * frac)


def se_diff(p1: float, n1: int, p2: float, n2: int) -> float:
    """SE of a difference of two independent proportions."""
    if not n1 or not n2:
        return math.inf
    return math.sqrt(max(p1 * (1 - p1), 0.0) / n1 + max(p2 * (1 - p2), 0.0) / n2)


def share_line(p_ref: float, n_ref: int, p: float, n: int) -> tuple[float, float]:
    """``(line, se)``: R_h's share - max(5 pp, 2 SE of the difference)."""
    se = se_diff(p_ref, n_ref, p, n)
    return p_ref - max(SHARE_PP, SHARE_SE_K * se), se


def _first_run_start(cond, need: int):
    """``p2s_stand._first_run_start``: the first index from which ``cond`` holds ``need`` rows running, or None."""
    return _S()._first_run_start(cond, need)


def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, set)):
        return [_jsonable(v) for v in x]
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return str(x)
    if hasattr(x, "item") and not isinstance(x, (str, bytes)):
        try:
            return _jsonable(x.item())
        except (ValueError, TypeError):
            return str(x)
    return x


def _sha(p) -> str:
    try:
        return hashlib.sha256(Path(p).read_bytes()).hexdigest()
    except OSError:
        return "unreadable"


def _json(p) -> dict | None:
    try:
        return json.loads(Path(p).read_text())
    except (OSError, ValueError):
        return None


# ===================================================================================== the shared definitions (numpy)
def h_meas_np(root_z, terrain_z6):
    """``sh_core.h_meas`` on numpy arrays -> float64 numpy (T,)."""
    np, torch = _np(), _torch()
    return CORE.h_meas(torch.as_tensor(np.asarray(root_z)), torch.as_tensor(np.asarray(terrain_z6))).numpy()


def hip_rod_dev_np(hip_angles):
    """``sh_core.hip_rod_dev`` on the raw hip-pitch joint angles (T, 6) (absolute, rad; the obs column is angle -
    default, which hip_rod_dev adds back) -> (T, 6) float64 m."""
    np, torch = _np(), _torch()
    a = np.asarray(hip_angles, dtype=np.float64)
    obs = torch.zeros(a.shape[0], max(CORE.HIP_POS_OBS_COLS) + 1, dtype=torch.float64)
    obs[:, list(CORE.HIP_POS_OBS_COLS)] = torch.as_tensor(a - float(CORE._kin().TH0))
    return CORE.hip_rod_dev(obs).numpy()


def joint_cols(meta: dict) -> dict | None:
    """Raw joint_pos columns (articulation order -> leg order FL..RR) of the hip-pitch, cam and hip-yaw joints."""
    names = meta.get("joint_names")
    if not isinstance(names, (list, tuple)):
        return None
    try:
        return {"hip": [list(names).index(f"{leg}_Hip_Femur_RevoluteJoint") for leg in LEGS],
                "cam": [list(names).index(f"{leg}_Body_CamShaft_RevoluteJoint") for leg in LEGS],
                "hip_yaw": [list(names).index(f"{leg}_Body_Hip_RevoluteJoint") for leg in LEGS]}
    except ValueError:
        return None


# ===================================================================================== the height rows
def manifest_row_gates(manifest=None) -> dict:
    """``{row: tuple(gates)}`` of the schedule manifest's ``stand_height.rows`` (default ``scenarios_sh.yaml``)."""
    data = EC.read_yaml(manifest or EC.SCENARIOS_SH)
    rows = ((data.get(EC.MANIFEST_BLOCK) or {}).get("rows")) or {}
    return {str(r): tuple(str(g) for g in ((v or {}).get("gates") or ())) for r, v in rows.items()}


def row_gates(spec: dict) -> dict:
    """The spec's ``height_rows`` ({row: [gates]}) or the manifest's."""
    hr = spec.get("height_rows")
    if hr:
        return {str(r): tuple(str(g) for g in gs) for r, gs in hr.items()}
    return manifest_row_gates(spec.get("manifest"))


def rows_for(rg: dict, gate: str) -> list[str]:
    return [r for r, gs in rg.items() if gate in gs]


# ===================================================================================== stamp contract
def sh_block(meta: dict | None) -> dict | None:
    m = (meta or {}).get("p2r_mode")
    b = m.get(EC.BLOCK_KEY) if isinstance(m, dict) else None
    return b if isinstance(b, dict) else None


def episode_files(stamp) -> dict:
    return {int(f.stem.split("_")[1]): f for f in (Path(stamp) / "raw").glob("episode_*.npz")}


def role_rule(role: str, is_head: bool) -> dict:
    """What the role's height stamps must record: controllers, restore, checkpoint kind (None = any)."""
    if is_head:
        return {"controllers": (HEAD_CONTROLLER,), "restore": True, "kind": HEAD_CKPT_KIND}
    return {"controllers": ROLE_CONTROLLER.get(role, tuple(EC.CONTROLLERS)),
            "restore": role != "no_restore", "kind": ROLE_CKPT_KIND.get(role)}


def restore_armed(blk: dict | None):
    """The block's restore as a bool (True armed / False off), None when malformed."""
    r = (blk or {}).get("restore")
    if isinstance(r, bool):
        return r
    return RESTORE_WORDS.get(r) if isinstance(r, str) else None


def meta_reasons(meta: dict | None, *, row: str | None = None, gates=None, rule: dict | None = None) -> list[str]:
    """run_meta contract of one height-row stamp (pure; module docstring "STAMP VALIDITY")."""
    if not meta:
        return ["run_meta.json missing or unreadable"]
    out = []
    labels, speeds = meta.get("hold_labels"), meta.get("hold_speeds")
    if not isinstance(labels, list) or not labels:
        out.append("run_meta.hold_labels missing")
        labels = []
    if not isinstance(speeds, list) or len(speeds) != len(labels):
        out.append("run_meta.hold_speeds missing or not one per hold")
        speeds = [None] * len(labels)
    try:
        dt = float(meta.get("dt"))
        if not dt > 0:
            raise ValueError
    except (TypeError, ValueError):
        out.append(f"run_meta.dt {meta.get('dt')!r} unreadable")
        dt = None
    if joint_cols(meta) is None:
        out.append("run_meta.joint_names lack the hip-pitch / cam / hip-yaw joints")
    mode = meta.get("p2r_mode") if isinstance(meta.get("p2r_mode"), dict) else {}
    if mode.get("variant") != EC.VARIANT:
        out.append(f"p2r_mode.variant {mode.get('variant')!r} != {EC.VARIANT!r}")
    blk = sh_block(meta)
    if blk is None:
        return out + [f"run_meta.p2r_mode.{EC.BLOCK_KEY} missing (the height launcher's record)"]
    if blk.get("command") != "manifest":
        out.append(f"command {blk.get('command')!r} != 'manifest' (a height row)")
    hh = blk.get("hold_heights")
    if not isinstance(hh, list) or len(hh) != len(labels):
        out.append(f"hold_heights not one entry per hold ({hh!r})")
    else:
        for i, (lab, sp, h) in enumerate(zip(labels, speeds, hh)):
            if lab == STAND_LABEL:
                if sp is not None and float(sp) != 0.0:
                    out.append(f"hold {i}: labelled {STAND_LABEL!r} with speed {sp}")
                if isinstance(h, bool) or not isinstance(h, (int, float)):
                    out.append(f"hold {i} (a stand): height {h!r} not a number")
                elif not (H_MIN - EPS <= float(h) <= H_MAX + EPS):
                    out.append(f"hold {i} (a stand): height {h} outside the frozen [{H_MIN}, {H_MAX}]")
            elif h is not None:
                out.append(f"hold {i} ({lab}, not a stand): height {h!r} (want null)")
    hr = blk.get("height_row")
    if not isinstance(hr, dict):
        out.append("height_row record missing (the manifest row this stamp ran)")
    else:
        if row is not None and hr.get("scenario") != row:
            out.append(f"height_row.scenario {hr.get('scenario')!r} != row {row!r}")
        if list(hr.get("labels") or []) != list(labels):
            out.append("height_row.labels != run_meta.hold_labels")
        if list(hr.get("hold_heights") or []) != list(hh or []):
            out.append("height_row.hold_heights != the block's hold_heights")
        if gates is not None and tuple(hr.get("gates") or ()) != tuple(gates):
            out.append(f"height_row.gates {hr.get('gates')!r} != the manifest's {list(gates)}")
    rec = blk.get("controller")
    restore = restore_armed(blk)
    if restore is None:
        out.append(f"restore {blk.get('restore')!r} is neither a bool nor armed / off")
    if not isinstance(rec, dict):
        out.append("controller record (HeightController.record()) missing")
    else:
        want = {"rate_m_s": CORE.HEIGHT_RATE_M_S, "h_min": H_MIN, "h_max": H_MAX, "rod_tol_m": CORE.RESTORE_ROD_TOL_M}
        if dt is not None:
            want.update(dt=dt, settle_steps=CORE.steps_for(CORE.RESTORE_SETTLE_S, dt))
        if restore is True:
            want["engage_m"] = CORE.RESTORE_ENGAGE_M
        for k, v in want.items():
            got = rec.get(k)
            if got is None or isinstance(got, bool) or abs(float(got) - float(v)) > 1e-9:
                out.append(f"controller.{k} {got!r} != the campaign's {v} (sh_core)")
        if restore is False and rec.get("engage_m") is not None:
            out.append(f"restore off but controller.engage_m {rec.get('engage_m')!r} (want null)")
    try:
        CORE.recorded_deadband(blk)
    except ValueError as exc:
        out.append(f"{EC.DEADBAND_KEY}: {exc} (a malformed residual-deadband record)")
    try:
        CORE.recorded_trained_deadband(blk)
    except ValueError as exc:
        out.append(f"{CORE.TRAINED_DEADBAND_KEY}: {exc} (a malformed trained-deadband record)")
    if not (blk.get("h_meas") or {}).get("valid") or meta.get("terrain_source") != "raycast_mesh":
        out.append(f"h_meas not valid (terrain_source {meta.get('terrain_source')!r}, record "
                   f"{(blk.get('h_meas') or {}).get('reason')!r}): the harness's root_z - 1.05 fallback")
    if rule is not None:
        if blk.get("control") not in rule["controllers"]:
            out.append(f"control {blk.get('control')!r} not this role's {list(rule['controllers'])}")
        if restore is not rule["restore"]:
            out.append(f"restore {restore!r}, this role needs {rule['restore']!r} (armed everywhere but the "
                       "no-restore control)")
        if rule["kind"] is not None and blk.get("stand_ckpt_kind") != rule["kind"]:
            out.append(f"stand_ckpt_kind {blk.get('stand_ckpt_kind')!r} != {rule['kind']!r}")
    env = meta.get("env_vars") or {}
    if meta.get("env_vars_manifest"):
        out.append(f"the row carries an env block {meta.get('env_vars_manifest')} (not the flat row)")
    if "KRABBY_FLAT_TERRAIN_MODE" in env:
        out.append("env sets KRABBY_FLAT_TERRAIN_MODE (not flat ground)")
    return out


def episode_reasons(d: dict, meta: dict, ep: int, *, required=HEIGHT_RAW_REQUIRED) -> list[str]:
    """One episode's raw contract (pure on a dict of arrays)."""
    np = _np()
    miss = [k for k in required if k not in d]
    if miss:
        return [f"episode {ep}: raw keys missing {miss} (fail closed)"]
    T = int(np.asarray(d["segment_id"]).shape[0])
    bad = [k for k in required if int(np.asarray(d[k]).shape[0]) != T]
    if bad:
        return [f"episode {ep}: keys {bad} not {T} rows"]
    out = []
    bh = (sh_block(meta) or {}).get("hold_heights") or []
    seg = np.asarray(d["segment_id"]).astype(np.int64)
    slot = np.asarray(d[RAW_SLOT]).astype(bool)
    eff = slot | np.asarray(d[RAW_RESTORING]).astype(bool)
    if not np.array_equal(eff, np.asarray(d[RAW_EFF]).astype(bool)):
        out.append(f"episode {ep}: {RAW_EFF} != {RAW_SLOT} | {RAW_RESTORING} on "
                   f"{int((eff != np.asarray(d[RAW_EFF]).astype(bool)).sum())} rows (sh_core: stand_eff = slot | "
                   "restoring)")
    hcmd = np.asarray(d[RAW_H_CMD], dtype=np.float64)
    if not np.isfinite(hcmd).all() or not np.isfinite(np.asarray(d[RAW_H_LIM], dtype=np.float64)).all():
        out.append(f"episode {ep}: non-finite {RAW_H_CMD} / {RAW_H_LIM}")
    for i, h in enumerate(bh):
        if h is None:
            continue
        m = slot & (seg == i)
        if m.any() and float(np.abs(hcmd[m] - float(h)).max()) > HOLD_HEIGHT_TOL:
            out.append(f"episode {ep}: {RAW_H_CMD} on stand hold {i} reads {float(np.median(hcmd[m])):.4f}, the "
                       f"stamp's hold height is {h} (the per-hold height did not reach the controller)")
    return out


def _load_episode(f: Path, keys) -> dict:
    np = _np()
    with np.load(f) as z:
        return {k: np.asarray(z[k]) for k in keys if k in z.files}


def stamp_contract_reasons(stamp, *, row=None, gates=None, rule=None) -> list[str]:
    """The whole raw / meta contract of one height-row stamp (IO; for smokes and this reader)."""
    meta = _json(Path(stamp) / "run_meta.json")
    out = meta_reasons(meta, row=row, gates=gates, rule=rule)
    files = episode_files(stamp)
    if not files:
        return out + ["no raw episodes"]
    for ep, f in sorted(files.items()):
        out += episode_reasons(_load_episode(f, HEIGHT_RAW_REQUIRED), meta or {}, ep)
    return out


# ===================================================================================== per-episode reads
class Ctx:
    """The per-stamp constants every episode read needs."""

    def __init__(self, meta: dict):
        self.meta = meta
        self.labels = list(meta.get("hold_labels") or [])
        self.speeds = [float(x) for x in (meta.get("hold_speeds") or [])]
        self.heights = list((sh_block(meta) or {}).get("hold_heights") or [])
        self.dt = float(meta.get("dt", 0.02))
        self.cols = joint_cols(meta)
        sf = CORE.steps_for
        self.n_win1 = sf(K1_WINDOW_S, self.dt)
        self.n_arr1 = sf(K1_ARRIVAL_DELAY_S, self.dt)
        self.n_reach = sf(REACH_HOLD_S, self.dt)
        self.n_settle = sf(CORE.RESTORE_SETTLE_S, self.dt)
        self.n_exit = sf(K4_EXIT_FALL_S, self.dt)
        self.n_camset = max(1, int(round(_S().Z2_SETTLE_HOLD_S / self.dt)))

    def is_stand(self, i: int) -> bool:
        """A stand hold: labelled "stand" (sh_eval_common.STAND_LABEL) with a height."""
        return (0 <= i < len(self.heights) and self.heights[i] is not None and i < len(self.labels)
                and self.labels[i] == STAND_LABEL)


def read_episode(d: dict, ctx: Ctx, ep: int) -> dict:
    """Every per-episode K1 / K2 / K4 / K6 record of one height-row episode (pure on a dict of arrays)."""
    np, S = _np(), _S()
    dt = ctx.dt
    seg = np.asarray(d["segment_id"]).astype(np.int64)
    T = int(seg.shape[0])
    slot = np.asarray(d[RAW_SLOT]).astype(bool)
    eff = slot | np.asarray(d[RAW_RESTORING]).astype(bool)          # == the recorded RAW_EFF (episode_reasons)
    hcmd = np.asarray(d[RAW_H_CMD], dtype=np.float64)
    hlim = np.asarray(d[RAW_H_LIM], dtype=np.float64)
    cf = np.asarray(d["crab_failure"]).astype(bool)
    hm = h_meas_np(d["root_pos_w"][:, 2], d["terrain_z"])
    jp = np.asarray(d["joint_pos"], dtype=np.float64)
    hip_dev = np.abs(hip_rod_dev_np(jp[:, ctx.cols["hip"]])).max(axis=1)
    hip_ok = hip_dev <= CORE.RESTORE_ROD_TOL_M
    hip_ok_strict = hip_dev <= CORE.RESTORE_ROD_TOL_M - K4_XCHECK_TOL_M
    hip_ok_loose = hip_dev <= CORE.RESTORE_ROD_TOL_M + K4_XCHECK_TOL_M
    cam_abs = np.abs(S.wrap_to_pi(jp[:, ctx.cols["cam"]])).max(axis=-1)
    loaded = np.asarray(d["foot_force_norm"]) > S.TOUCH_REF_N
    upper = S.fall_trim_upper(d, dt)
    out = {"ep": ep, "T": T, "k1": [], "k2": [], "k4": [], "k6_stands": [], "k6_windows": [],
           "k2_fall": bool((cf & slot).any()), "fell": bool(cf.any()), "invalid": []}

    # ---- K1: stand holds
    for i in range(len(ctx.labels)):
        if not ctx.is_stand(i):
            continue
        rows = np.flatnonzero(seg == i)
        if not rows.size:
            continue
        h = float(ctx.heights[i])
        rec = {"ep": ep, "hold": i, "h_cmd": h, "bin": height_bin(h), "err": None, "window_rows": 0,
               "arrival_s": None, "fell": bool(cf[rows].any()), "why": None}
        arr = rows[np.abs(hlim[rows] - h) <= CMD_TOL]
        if arr.size:
            rec["arrival_s"] = float((arr[0] - rows[0]) * dt)
        end = int(rows[-1]) + 1
        if rec["fell"]:
            rec["why"] = "fall in the hold"
        elif not arr.size:
            rec["why"] = "the limited command never arrived inside the hold"
        else:
            lo = max(end - ctx.n_win1, int(arr[0]) + ctx.n_arr1)
            win = rows[(rows >= lo) & (rows < upper)]
            if not win.size:
                rec["why"] = "no window: arrival + 1 s falls after the hold's end"
            else:
                rec["err"] = np.abs(hm[win] - h)
                rec["window_rows"] = int(win.size)
        out["k1"].append(rec)

    # ---- K2 events / K6 command segments: stand-slot runs split at raw-command changes
    chg = np.zeros(T, dtype=bool)
    if T > 1:
        chg[1:] = slot[1:] & (~slot[:-1] | (np.abs(hcmd[1:] - hcmd[:-1]) > CMD_TOL))
    starts = np.flatnonzero(chg | (slot & (np.arange(T) == 0)))
    for a in starts.tolist():
        e = a + 1
        while e < T and slot[e] and not chg[e]:
            e += 1
        target = float(hcmd[a])
        ok = np.abs(hm[a:e] - target) <= TRACK_TOL_M + EPS
        r0 = _first_run_start(ok, ctx.n_reach)
        fell = bool(cf[a:e].any())
        if a >= 1:
            dh = abs(target - float(hlim[a - 1]))
            if dh > TRACK_TOL_M + EPS:
                line = dh / CORE.HEIGHT_RATE_M_S + K2_EXTRA_S
                short = False
                if r0 is not None and not fell:
                    lat = r0 * dt
                elif fell:
                    lat = math.inf
                elif (e - a) * dt < line + REACH_HOLD_S - EPS:
                    lat, short = None, True
                else:
                    lat = math.inf
                out["k2"].append({"ep": ep, "row": a, "entry": not bool(slot[a - 1]), "h_from": float(hlim[a - 1]),
                                  "h_to": target, "dh": dh, "dir": "raise" if target > hlim[a - 1] else "lower",
                                  "bin": height_bin(target), "latency_s": lat, "line_s": line,
                                  "excess_s": None if lat is None else lat - line, "fell": fell, "short": short,
                                  "segment_s": (e - a) * dt})
        e2 = min(e, upper)
        win = {"ep": ep, "row": a, "h_cmd": target, "bin": height_bin(target), "opened": False, "window_s": 0.0,
               "steps": 0, "taps": 0, "arrival_s": None, "open_s": None}
        if e2 > a:
            r_arr = _first_run_start(np.abs(hm[a:e2] - target) <= TRACK_TOL_M + EPS, ctx.n_reach)
            if r_arr is not None:
                win["arrival_s"] = r_arr * dt
                r_set = _first_run_start(cam_abs[a + r_arr:e2] <= S.Z2_HOMED_RAD, ctx.n_camset)
                if r_set is not None:
                    w0 = a + r_arr + r_set
                    steps, taps, _c = S.M.step_tap_runs(loaded[w0:e2], min_step_steps=S.MIN_STEP_STEPS)
                    win.update(opened=True, window_s=(e2 - w0) * dt, steps=len(steps), taps=len(taps),
                               open_s=(w0 - a) * dt)
        out["k6_windows"].append(win)

    # ---- K6 walk-handover stands (p2s_stand's per-stand estimators, unchanged)
    for i in range(1, len(ctx.labels)):
        if not ctx.is_stand(i) or ctx.is_stand(i - 1):
            continue
        m = S.stand_z2_metrics(d, i, dt, ctx.cols["cam"], ctx.cols["hip_yaw"], S.WINDOW_LAST)
        if not m["reached"]:
            continue
        h = float(ctx.heights[i])
        out["k6_stands"].append({"ep": ep, "hold": i, "h_cmd": h, "bin": height_bin(h), "homed": m["homed"] is True,
                                 "settled": bool(m["settled"]), "settle_s": m["settle_s"], "fall": bool(m["fall"]),
                                 "six_down_share": m["six_down_share"], "hip_yaw_mean_deg": m["hip_yaw_mean_deg"],
                                 "cam_max_last_rad": m["cam_max_last_rad"]})

    # ---- K4 hand-backs
    exits = (np.flatnonzero(slot[:-1] & ~slot[1:]) + 1).tolist() if T > 1 else []
    tmo = np.asarray(d[RAW_TIMED_OUT]).astype(bool) if RAW_TIMED_OUT in d else None
    comp = np.asarray(d[RAW_COMPLETED]).astype(bool) if RAW_COMPLETED in d else None
    armed = restore_armed(sh_block(ctx.meta))

    def rederived(t, ok):
        return all(k >= 1 and abs(float(hlim[k - 1]) - H_SETTLED) <= CMD_TOL and bool(ok[k])
                   for k in range(t - ctx.n_settle + 1, t + 1))

    for s in exits:
        h_last = float(hlim[s - 1])
        dh = abs(h_last - H_SETTLED)
        required = dh > CORE.RESTORE_ENGAGE_M + EPS
        nxt = np.flatnonzero(slot[s:])
        horizon = int(s + nxt[0]) if nxt.size else T        # the next stand entry: a hand-back must come before it
        off = np.flatnonzero(~eff[s:horizon])
        t = int(s + off[0]) if off.size else None
        stop = t if t is not None else horizon
        fell_restore = bool(cf[s:stop].any())
        complete = rec_complete = rec_timeout = None
        source = None
        if t is not None and required:
            complete = rederived(t, hip_ok)
            rec_complete = None if comp is None else bool(comp[t])
            rec_timeout = None if tmo is None else bool(tmo[t])
        if t is None:
            # no S1 row before the episode's end / the next stand: a fall, a stand->stand merge (the restore ran into
            # the next stand slot: no hand-back happened, nothing to judge) or a cut (fail closed: a miss)
            status = "fell" if fell_restore else ("merged" if horizon < T else "cut")
        elif not required:
            status = "plain"
        elif rec_complete is not None:
            # the controller's own H6 event decides (sh_core.HeightController: completion = 10 rows at h_prev 1.064
            # with every hip rod within delta, on the OBS it acted on); "miss" = a recorded timeout or no completion
            # (restore off: the hand-back at height). The re-derivation from raw joint_pos is the cross-check.
            source = "recorded"
            status = "complete" if rec_complete else "miss"
            if rec_complete and rec_timeout:
                out["invalid"].append(f"episode {ep}: hand-back row {t} records BOTH {RAW_COMPLETED} and "
                                      f"{RAW_TIMED_OUT}")
            if armed is True and not rec_complete and rec_timeout is False:
                out["invalid"].append(f"episode {ep}: an armed restore handed back at row {t} with neither "
                                      f"{RAW_COMPLETED} nor {RAW_TIMED_OUT} recorded")
            strict, loose = rederived(t, hip_ok_strict), rederived(t, hip_ok_loose)
            if (rec_complete and not loose) or (not rec_complete and strict):
                out["invalid"].append(
                    f"episode {ep}: K4 cross-check -- the launcher records {RAW_COMPLETED}={rec_complete} at hand-back "
                    f"row {t}, the joint-space re-derivation (raw joint_pos - TH0, delta "
                    f"{CORE.RESTORE_ROD_TOL_M} m +- {K4_XCHECK_TOL_M}) says {not rec_complete} (fail closed: the "
                    "reader and the controller read the restore differently)")
        else:
            source = "rederived"
            status = "complete" if complete else "miss"
        miss = required and status in ("miss", "cut")
        exit_fall = fell_restore or (t is not None and bool(cf[t:min(T, t + ctx.n_exit)].any()))
        move = int(seg[s])
        onset, has_pair = None, False
        if 1 <= move < len(ctx.labels) and not ctx.is_stand(move) and move < len(ctx.speeds) \
                and ctx.speeds[move] > 0 and ctx.is_stand(move - 1):
            has_pair = True
            onset = S.episode_onset_latency(d, dt, move)
        out["k4"].append({"ep": ep, "row": s, "handback_row": t, "h_last": h_last, "dh": dh, "required": required,
                          "status": status, "miss": miss, "exit_fall": exit_fall,
                          "restore_s": None if t is None else (t - s) * dt,
                          "flat_err_m": None if t is None else abs(float(hm[t]) - H_SETTLED),
                          "timed_out": None if tmo is None else bool(tmo[s:stop + 1].any()),
                          "completion_source": source, "complete_rederived": complete,
                          "move_hold": move, "onset_pair": has_pair, "onset_s": onset})

    # ---- dose (reported)
    if RAW_A_NET in d and RAW_Q in d and eff.any():
        q = np.asarray(d[RAW_Q], dtype=np.float64)[eff]
        tgt = q[:, 6:12] + np.clip(np.asarray(d[RAW_A_NET], dtype=np.float64)[eff][:, 6:12], -1.0, 1.0)
        out["dose"] = {"rows": int(eff.sum()), "rows_hip_below_m1": int((tgt < -1.0 - 1e-6).any(axis=1).sum()),
                       "rows_hip_above_p1": int((tgt > 1.0 + 1e-6).any(axis=1).sum()),
                       "a_net_abs_max": float(np.abs(np.asarray(d[RAW_A_NET])[eff]).max()),
                       "q_abs_max": float(np.abs(q).max())}
    return out


def read_height_stamp(stamp: str | None, *, row: str | None = None, gates=None, rule=None) -> dict:
    """``status`` missing / invalid / ok plus every episode's records (IO)."""
    if not stamp:
        return {"stamp": None, "status": "missing", "reasons": [], "row": row}
    meta = _json(Path(stamp) / "run_meta.json")
    reasons = meta_reasons(meta, row=row, gates=gates, rule=rule)
    files = episode_files(stamp)
    if not files:
        reasons.append("no raw episodes (fail closed)")
    base = {"stamp": str(stamp), "reasons": reasons, "n_episodes": len(files), "row": row}
    if reasons:
        return dict(base, status="invalid")
    ctx = Ctx(meta)
    eps, keys = [], HEIGHT_RAW_REQUIRED + HEIGHT_RAW_REPORTED
    for ep, f in sorted(files.items()):
        d = _load_episode(f, keys)
        r = episode_reasons(d, meta, ep)
        if r:
            reasons += r
            continue
        e = read_episode(d, ctx, ep)
        reasons += e["invalid"]
        eps.append(e)
    if reasons:
        return dict(base, status="invalid")
    blk = sh_block(meta) or {}
    mode = meta.get("p2r_mode") or {}
    agg = {k: [x for e in eps for x in e[k]] for k in ("k1", "k2", "k4", "k6_stands", "k6_windows")}
    doses = [e["dose"] for e in eps if e.get("dose")]
    dose = ({"rows": sum(x["rows"] for x in doses), "rows_hip_below_m1": sum(x["rows_hip_below_m1"] for x in doses),
             "rows_hip_above_p1": sum(x["rows_hip_above_p1"] for x in doses),
             "a_net_abs_max": max(x["a_net_abs_max"] for x in doses),
             "q_abs_max": max(x["q_abs_max"] for x in doses)} if doses else None)
    return dict(base, status="ok", schedule={"hold_labels": ctx.labels, "hold_heights": ctx.heights,
                                             "hold_speeds": ctx.speeds}, dt=ctx.dt,
                controller=blk.get("control"), restore=restore_armed(blk),
                stand_ckpt_kind=blk.get("stand_ckpt_kind"), residual_deadband=CORE.recorded_deadband(blk),
                trained_residual_deadband=CORE.recorded_trained_deadband(blk),
                stand_sha256=((mode.get("specialist") or {}).get("sha256")
                              if isinstance(mode.get("specialist"), dict) else None),
                k2_falls=sum(1 for e in eps if e["k2_fall"]), episodes_fell=sum(1 for e in eps if e["fell"]),
                dose=dose, **agg)


def gate_view(by_row: dict, rg: dict, gate: str) -> dict:
    """The reads one gate uses: every row whose manifest gates include ``gate`` (pure). status: invalid (any row) >
    missing (any row) > ok (records concatenated; per-row episode counts and schedules kept for denominators)."""
    rows = rows_for(rg, gate)
    parts = {r: (by_row or {}).get(r) or {"status": "missing", "stamp": None, "reasons": []} for r in rows}
    if not rows:
        return {"status": "missing", "rows": {}, "reasons": [f"no height row feeds {gate}"]}
    inv = [f"{r}: {x}" for r, p in parts.items() if p.get("status") == "invalid" for x in p.get("reasons") or []]
    if inv:
        return {"status": "invalid", "rows": parts, "reasons": inv,
                "stamp": "; ".join(str(p.get("stamp")) for p in parts.values())}
    if any(p.get("status") != "ok" for p in parts.values()):
        return {"status": "missing", "rows": parts, "reasons": [],
                "missing_rows": [r for r, p in parts.items() if p.get("status") != "ok"]}
    out = {"status": "ok", "rows": parts, "reasons": [], "dt": next(iter(parts.values())).get("dt"),
           "stamp": "; ".join(str(p.get("stamp")) for p in parts.values())}
    for k in ("k1", "k2", "k4", "k6_stands", "k6_windows"):
        out[k] = [x for p in parts.values() for x in p[k]]
    out["k2_falls"] = sum(p["k2_falls"] for p in parts.values())
    return out


# ===================================================================================== gate helpers
def _st(v: dict | None) -> str:
    return (v or {}).get("status", "missing")


def _inv(role: str, s: int, v: dict | None, what: str = "") -> list[str]:
    if _st(v) != "invalid":
        return []
    return [f"[{role}{what}] es{s}: {x}" for x in v.get("reasons") or []]


def _denominator_reasons(role: str, s: int, v: dict, ref: dict) -> list[str]:
    """A raw count is only comparable on R_h's set: per row, the same episode count and the same schedule."""
    out = []
    for r, p in (v.get("rows") or {}).items():
        q = (ref.get("rows") or {}).get(r) or {}
        if p.get("n_episodes") != q.get("n_episodes"):
            out.append(f"[{role}] es{s} {r}: {p.get('n_episodes')} episodes vs R_h's {q.get('n_episodes')} -- the "
                       "raw-count line needs R_h's denominator")
        if p.get("schedule") != q.get("schedule"):
            out.append(f"[{role}] es{s} {r}: a different schedule (hold labels / heights / speeds) from R_h's")
    return out


def _count_clause(cl: dict, inv: list, role: str, s: int, v: dict, ref: dict, value, ref_value, short: str,
                  what: str) -> None:
    """``value <= max(FALL_FLOOR, R_h's)`` per seed, on R_h's denominator."""
    name = f"{short} es{s} <= max({FALL_FLOOR}, R_h's)"
    if _st(ref) == "missing":
        cl[name] = _cl(value, None, None, "pending", s, short=short, note=f"R_h es{s} missing ({what})")
        return
    if _st(ref) == "invalid":
        inv.extend(_inv("R_h", s, ref, " line"))
        cl[name] = _cl(value, None, None, "pending", s, short=short, note="R_h line unreadable")
        return
    if role != "R_h":
        inv.extend(_denominator_reasons(role, s, v, ref))
    line = max(FALL_FLOOR, int(ref_value))
    cl[name] = _cl(int(value), line, int(value) <= line, "eval", s, short=short, note=what)


def _view(views: dict, role: str, s: int, gate: str) -> dict:
    return (((views or {}).get(role) or {}).get(s) or {}).get(gate) or {"status": "missing", "reasons": []}


# ===================================================================================== K1
def gate_k1(role: str, views: dict, seeds=SEEDS) -> dict:
    np = _np()
    cl, inv, per = {}, [], {}
    for s in seeds:
        v = _view(views, role, s, "K1")
        st = _st(v)
        inv += _inv(role, s, v)
        for b in range(K_BINS):
            nm = (f"median bin{b} es{s} <= {K1_MED_M}", f"p90 bin{b} es{s} <= {K1_P90_M}")
            if st != "ok":
                for n, line, sh in zip(nm, (K1_MED_M, K1_P90_M), ("median", "p90")):
                    cl[n] = _cl(None, line, None, "pending", s, short=sh, note=f"stamp {st}")
                continue
            holds = [h for h in v["k1"] if h["bin"] == b]
            if not holds:
                for n, line, sh in zip(nm, (K1_MED_M, K1_P90_M), ("median", "p90")):
                    cl[n] = _cl(None, line, None, "pending", s, short=sh,
                                note=f"no stand hold commands bin {bin_label(b)}: the schedule does not cover it")
                continue
            n_win = CORE.steps_for(K1_WINDOW_S, float(v.get("dt") or 0.02))
            parts, n_fc = [], 0
            for h in holds:
                if h["err"] is None:
                    parts.append(np.full(n_win, math.inf))
                    n_fc += 1
                else:
                    parts.append(np.asarray(h["err"], dtype=np.float64))
            x = np.concatenate(parts)
            med, p90 = pctl(x, 50), pctl(x, 90)
            note = f"{len(holds)} holds, {x.size} samples, {n_fc} fail-closed holds"
            cl[nm[0]] = _cl(med, K1_MED_M, med <= K1_MED_M + EPS, "eval", s, short="median", note=note)
            cl[nm[1]] = _cl(p90, K1_P90_M, p90 <= K1_P90_M + EPS, "eval", s, short="p90", note=note)
            per[f"es{s} bin{b}"] = {"bin": bin_label(b), "holds": len(holds), "fail_closed_holds": n_fc,
                                    "median_m": med, "p90_m": p90,
                                    "why_fail_closed": sorted({h["why"] for h in holds if h["why"]})}
    return dict(verdict_of(cl, inv), clauses=cl, per_bin=per,
                statistic="pooled window samples of |h_meas - h_cmd| per bin per seed: median <= 0.02, p90 <= 0.04; "
                          "window = max(hold end - 2 s, arrival + 1 s) .. hold end; no window / fall = 2 s of +inf")


# ===================================================================================== K2
def gate_k2(role: str, views: dict, seeds=SEEDS) -> dict:
    cl, inv, rep = {}, [], {}
    for s in seeds:
        v, ref = _view(views, role, s, "K2"), _view(views, "R_h", s, "K2")
        st = _st(v)
        inv += _inv(role, s, v)
        name = f"time-to-2cm excess p90 es{s} <= 0 s"
        if st != "ok":
            cl[name] = _cl(None, 0.0, None, "pending", s, short="reach", note=f"stamp {st}")
            cl[f"stand falls es{s} <= max({FALL_FLOOR}, R_h's)"] = _cl(None, None, None, "pending", s,
                                                                       short="falls", note=f"stamp {st}")
            continue
        ev = [e for e in v["k2"] if not e["short"]]
        short = [e for e in v["k2"] if e["short"]]
        if not ev:
            cl[name] = _cl(None, 0.0, None, "pending", s, short="reach",
                           note="no raise / lower event > 2 cm in the schedule")
        else:
            p = pctl([e["excess_s"] for e in ev], K2_Q)
            cl[name] = _cl(p, 0.0, p <= EPS, "eval", s, short="reach",
                           note=f"{len(ev)} events ({sum(1 for e in ev if math.isinf(e['excess_s']))} never / fell); "
                                f"{len(short)} short segments excluded")
        _count_clause(cl, inv, role, s, v, ref, v["k2_falls"], ref.get("k2_falls", 0), "stand falls",
                      "episodes with crab_failure on a stand-slot row")
        rep[f"es{s}"] = {"events": len(ev), "short": len(short),
                         "median_excess_s": pctl([e["excess_s"] for e in ev], 50) if ev else None,
                         "by_dir": {d_: {"n": len(x), "p90_excess_s": pctl([e["excess_s"] for e in x], K2_Q)}
                                    for d_ in ("raise", "lower") for x in [[e for e in ev if e["dir"] == d_]] if x},
                         "by_bin": {bin_label(b): {"n": len(x), "p90_excess_s": pctl([e["excess_s"] for e in x], K2_Q)}
                                    for b in range(K_BINS) for x in [[e for e in ev if e["bin"] == b]] if x}}
    return dict(verdict_of(cl, inv), clauses=cl, reported=rep,
                statistic="per seed p90 over raise / lower events of (time to |h_meas - h_cmd| <= 2 cm held 0.5 s) - "
                          "(dh / 0.08 + 1 s), never / fall = +inf; stand-slot falls <= max(1, R_h's) (raw count)")


# ===================================================================================== K4
def refs_onsets(spec: dict, seeds=SEEDS) -> dict:
    """{seed: {"onset_median_s", "stamp", "source", "status"}}: REFS's walkstop onset median (p2_read.walkstop_read)
    or the spec's ``refs_onset_s`` override (recorded as such)."""
    out = {}
    over = spec.get("refs_onset_s") or {}
    P = None
    for s in seeds:
        if str(s) in over or s in over:
            v = over.get(str(s), over.get(s))
            out[s] = {"onset_median_s": None if v is None else float(v), "stamp": None, "source": "spec override",
                      "status": "ok" if v is not None else "missing"}
            continue
        P = P or _P()
        name = spec.get("refs") or REFS_NAME
        lr = Path(spec["lr_root"]) if spec.get("lr_root") else P.LROUT
        stamp = P.newest_stamp(name, P.WS, s, lr)
        w = P.walkstop_read(stamp) if stamp else None
        out[s] = {"onset_median_s": (w or {}).get("onset_median_s"), "stamp": stamp,
                  "source": f"p2_read.walkstop_read({name} {P.WS} es{s})",
                  "status": "missing" if not stamp else ("ok" if (w or {}).get("onset_median_s") is not None
                                                         else "unreadable")}
    return out


def gate_k4(role: str, views: dict, refs_on: dict, seeds=SEEDS) -> dict:
    np = _np()
    cl, inv, rep = {}, [], {}
    for s in seeds:
        v, ref = _view(views, role, s, "K4"), _view(views, "R_h", s, "K4")
        st = _st(v)
        inv += _inv(role, s, v)
        names = (f"miss es{s} <= max({FALL_FLOOR}, R_h's)", f"exit falls es{s} <= max({FALL_FLOOR}, R_h's)",
                 f"onset excess p90 es{s} <= 0 s")
        if st != "ok":
            for n, sh in zip(names, ("miss", "exit falls", "onset")):
                cl[n] = _cl(None, None, None, "pending", s, short=sh, note=f"stamp {st}")
            continue
        hb = v["k4"]
        req = [h for h in hb if h["required"]]
        if not req:
            cl[names[0]] = _cl(None, None, None, "pending", s, short="miss",
                               note="no restore-required hand-back: the schedule does not exercise H6")
        else:
            rmiss = sum(1 for h in (ref.get("k4") or []) if h["miss"])
            _count_clause(cl, inv, role, s, v, ref, sum(1 for h in req if h["miss"]), rmiss, "miss",
                          f"{len(req)} restore-required exits: "
                          f"{sum(1 for h in req if h['status'] == 'complete')} complete, "
                          f"{sum(1 for h in req if h['status'] == 'miss')} missed the completion, "
                          f"{sum(1 for h in req if h['status'] == 'cut')} cut, "
                          f"{sum(1 for h in req if h['status'] == 'fell')} fell in the restore, "
                          f"{sum(1 for h in req if h['status'] == 'merged')} merged into the next stand")
        _count_clause(cl, inv, role, s, v, ref, sum(1 for h in hb if h["exit_fall"]),
                      sum(1 for h in (ref.get("k4") or []) if h["exit_fall"]), "exit falls",
                      "crab_failure from the slot exit to hand-back + 2 s")
        pairs = [h for h in hb if h["onset_pair"]]
        ro = (refs_on or {}).get(s) or {}
        if ro.get("onset_median_s") is None:
            if ro.get("status") == "unreadable":
                inv.append(f"[REFS] es{s} {ro.get('stamp')}: walkstop onset median unreadable -- the onset line "
                           "cannot be drawn")
            cl[names[2]] = _cl(None, 0.0, None, "pending", s, short="onset", note=f"REFS es{s} onset {ro}")
        elif not pairs:
            cl[names[2]] = _cl(None, 0.0, None, "pending", s, short="onset",
                               note="no stand -> moving-hold hand-back in the schedule")
        else:
            ex = [math.inf if h["onset_s"] is None else
                  h["onset_s"] - (ro["onset_median_s"] + h["dh"] / CORE.HEIGHT_RATE_M_S + K4_ONSET_EXTRA_S)
                  for h in pairs]
            p = pctl(ex, K4_Q)
            cl[names[2]] = _cl(p, 0.0, p <= EPS, "eval", s, short="onset",
                               note=f"{len(pairs)} hand-backs into a moving hold, "
                                    f"{sum(1 for h in pairs if h['onset_s'] is None)} censored; line REFS "
                                    f"{ro['onset_median_s']} s + dh / 0.08 + {K4_ONSET_EXTRA_S} s")
        fl = [h["flat_err_m"] for h in hb if h["flat_err_m"] is not None]
        rs = [h["restore_s"] for h in req if h["restore_s"] is not None]
        rep[f"es{s}"] = {"handbacks": len(hb), "restore_required": len(req),
                         "status": {k: sum(1 for h in hb if h["status"] == k)
                                    for k in ("plain", "complete", "miss", "cut", "fell", "merged")},
                         "timeouts_raw": (sum(1 for h in hb if h["timed_out"]) if all(h["timed_out"] is not None
                                                                                       for h in hb) else None),
                         "handback_abs_h_minus_1064_m": ({"median": pctl(fl, 50), "p90": pctl(fl, 90),
                                                          "max": float(np.max(fl))} if fl else None),
                         "restore_s": {"median": pctl(rs, 50), "max": float(np.max(rs))} if rs else None}
    return dict(verdict_of(cl, inv), clauses=cl, reported=rep,
                statistic="H6 joint-space completion at every restore-required hand-back (misses <= max(1, R_h's)); "
                          "exit falls slot exit .. hand-back + 2 s <= max(1, R_h's); p90 onset excess over "
                          "REFS + dh / 0.08 + 1.5 s <= 0")


# ===================================================================================== K6
def gate_k6(role: str, views: dict, seeds=SEEDS) -> dict:
    P = _P()
    lim_steps = float(P.Z2_POST["post_settle_steps_per_s"])
    lim_taps = float(P.Z2_POST["post_settle_taps_per_s"])
    cl, inv, rep = {}, [], {}
    for s in seeds:
        v, ref = _view(views, role, s, "K6"), _view(views, "R_h", s, "K6")
        st, rst = _st(v), _st(ref)
        inv += _inv(role, s, v)
        if role != "R_h" and rst == "invalid":
            inv += _inv("R_h", s, ref, " line")
        for b in range(K_BINS):
            for key, short in (("homed", "homed"), ("settled", "settled")):
                name = f"{short} share bin{b} es{s} >= R_h's - max({SHARE_PP}, {SHARE_SE_K:g} SE)"
                if st != "ok" or rst != "ok":
                    cl[name] = _cl(None, None, None, "pending", s, short=short, note=f"stamp {st}, R_h {rst}")
                    continue
                a = [x for x in v["k6_stands"] if x["bin"] == b]
                r = [x for x in ref["k6_stands"] if x["bin"] == b]
                if not a or not r:
                    cl[name] = _cl(None, None, None, "pending", s, short=short,
                                   note=f"no walk-handover stand in bin {bin_label(b)} ({len(a)} / R_h {len(r)})")
                    continue
                p = sum(1 for x in a if x[key]) / len(a)
                pr = sum(1 for x in r if x[key]) / len(r)
                line, se = share_line(pr, len(r), p, len(a))
                cl[name] = _cl(p, line, p >= line - EPS, "eval", s, short=short,
                               note=f"n {len(a)} vs R_h {pr:.3f} n {len(r)}, SE {se:.4f}")
        names = (f"stillness steps/s pooled es{s} <= {lim_steps:g}", f"stillness taps/s pooled es{s} <= {lim_taps:g}")
        if st != "ok":
            for n, sh, lim in zip(names, ("still steps", "still taps"), (lim_steps, lim_taps)):
                cl[n] = _cl(None, lim, None, "pending", s, short=sh, note=f"stamp {st}")
            continue
        w = v["k6_windows"]
        opened = [x for x in w if x["opened"]]
        tot = sum(x["window_s"] for x in opened)
        stp = sum(x["steps"] for x in opened)
        tap = sum(x["taps"] for x in opened)
        for n, sh, lim, cnt in zip(names, ("still steps", "still taps"), (lim_steps, lim_taps), (stp, tap)):
            if tot <= 0:
                cl[n] = _cl(None, lim, False, "eval", s, short=sh,
                            note=f"no stillness window opened in {len(w)} command segments: fails closed")
            else:
                x = cnt / tot
                cl[n] = _cl(x, lim, x <= lim + EPS, "eval", s, short=sh,
                            note=f"{cnt} in {tot:.1f} s over {len(opened)} of {len(w)} segments")
        by = {}
        for b in range(K_BINS):
            ob = [x for x in opened if x["bin"] == b]
            tb = sum(x["window_s"] for x in ob)
            sb = [x for x in v["k6_stands"] if x["bin"] == b]
            by[bin_label(b)] = {"segments": sum(1 for x in w if x["bin"] == b), "opened": len(ob), "window_s": tb,
                                "steps_per_s": (sum(x["steps"] for x in ob) / tb) if tb > 0 else None,
                                "taps_per_s": (sum(x["taps"] for x in ob) / tb) if tb > 0 else None,
                                "stands": len(sb),
                                "homed_share": (sum(1 for x in sb if x["homed"]) / len(sb)) if sb else None,
                                "settled_share": (sum(1 for x in sb if x["settled"]) / len(sb)) if sb else None,
                                "six_down_share_median": pctl([x["six_down_share"] for x in sb
                                                               if x["six_down_share"] is not None], 50),
                                "hip_yaw_mean_deg_median": pctl([x["hip_yaw_mean_deg"] for x in sb
                                                                 if x["hip_yaw_mean_deg"] is not None], 50)}
        rep[f"es{s}"] = by
    return dict(verdict_of(cl, inv), clauses=cl, reported=rep,
                statistic="walk-handover stands per bin: homed / settled share (p2s_stand) >= R_h's - max(5 pp, 2 SE); "
                          "stillness pooled over bins per seed (window after arrival + cam settle, re-opened per "
                          "command): steps / taps per s <= p2_read.Z2_POST")


# ===================================================================================== K3 (sh_table_core)
def _cell_paths(items) -> list[Path]:
    out = []
    for it in items or []:
        p = Path(it)
        if p.is_dir():
            out += sorted(Path(x) for x in glob.glob(str(p / "*.json")))
        elif any(c in str(it) for c in "*?["):
            out += sorted(Path(x) for x in glob.glob(str(it)))
        else:
            out.append(p)
    return out


def read_table_cells(items, seed: int, *, controller: str | None = None, amask: bool = False) -> dict:
    """``{"cells": {cell: aggregate}, "per_cell": {cell: report}, "stand_sha256": [...], "reasons", "notes"}`` from
    the table test's per-cell JSON (files / dirs / globs) of one role and env seed (IO; fail closed on the GATED cells:
    a REPORTED cell's contract problem is a note -- it never decides, so it never invalidates)."""
    TC = _tc()
    cells, rep, reasons, notes, shas, rep_shas = {}, {}, [], [], set(), set()
    dbs, rep_dbs = set(), set()
    for p in _cell_paths(items):
        d = _json(p)
        if d is None:
            reasons.append(f"{p}: unreadable JSON")
            continue
        if d.get("kind") != TC.CELL_KIND:
            notes.append(f"{p}: kind {d.get('kind')!r}, not a {TC.CELL_KIND} file (skipped)")
            continue
        m = d.get("meta") or {}
        if m.get("smoke") or m.get("video"):
            notes.append(f"{p}: smoke / video run, not counted")
            continue
        gated = TC.cell_gated(m.get("cell"))

        def flag(msg, gated=gated):
            (reasons if gated else notes).append(msg if gated else f"[reported cell, not gated] {msg}")

        try:
            surface, _pose = TC.parse_cell(m.get("cell"))
        except (TypeError, ValueError) as exc:
            reasons.append(f"{p}: {exc}")
            continue
        try:
            seed_ok = int(m.get("seed")) == int(seed)
        except (TypeError, ValueError):
            seed_ok = False
        if not seed_ok:
            flag(f"{p}: meta.seed {m.get('seed')!r} != env seed {seed}")
            continue
        cell = m["cell"]
        if controller is not None and m.get("controller") != controller:
            flag(f"{p}: controller {m.get('controller')!r}, this role is {controller!r}")
        if m.get("stand_sha256"):                  # the head's one-checkpoint rule reads the GATED cells' shas
            (shas if gated else rep_shas).add(m["stand_sha256"])
        try:                                       # ... and its one-deadband rule too (absent = 0)
            (dbs if gated else rep_dbs).add(CORE.recorded_deadband(m))
        except ValueError as exc:
            flag(f"{p}: {EC.DEADBAND_KEY}: {exc}")
        if amask and m.get("stand_sha256") != CORE.STAND_NET_SHA256:
            flag(f"{p}: stand checkpoint {str(m.get('stand_sha256'))[:12]} is not the A-mask s17 file")
        if cell in cells:
            flag(f"{p}: duplicate cell {cell} ({rep[cell]['run']})")
            continue
        a, pe = d.get("aggregate") or {}, d.get("per_env") or []
        if (a.get("n_envs") != len(pe) or a.get("n_stood") != sum(1 for r in pe if r.get("stood"))
                or a.get("falls") != sum(1 for r in pe if r.get("fall"))):
            flag(f"{p}: aggregate (n {a.get('n_envs')}, stood {a.get('n_stood')}, falls {a.get('falls')}) "
                 "disagrees with its per-env records")
        top = TC.SURFACE_TOP.get(surface)
        if top is not None:
            low = [r.get("h_cmd_m") for r in pe if r.get("h_cmd_m") is None
                   or float(r["h_cmd_m"]) < top + TC.CMD_MARGIN_EVAL_M - 1e-6]
            if low:
                flag(f"{p}: {len(low)} envs command < top {top} + {TC.CMD_MARGIN_EVAL_M} m")
        cells[cell] = a
        rep[cell] = {"run": str(p), "gated": gated, "n": a.get("n_envs"), "stood_share": a.get("stood_share"),
                     "falls_per100": a.get("falls_per100"), "n_stood_5down": a.get("n_stood_5down"),
                     "n_stood_6down": a.get("n_stood_6down"), "homed": a.get("homed"),
                     "homed_share": a.get("homed_share"), "episode_s": m.get("episode_s"),
                     "k3_episode_s": m.get("k3_episode_s")}
    if rep_shas - shas and shas:
        notes.append(f"[reported cells, not gated] stand checkpoints {[x[:12] for x in sorted(rep_shas - shas)]} "
                     "differ from the gated cells'")
    if rep_dbs - dbs and dbs:
        notes.append(f"[reported cells, not gated] residual deadbands {sorted(rep_dbs - dbs)} differ from the gated "
                     "cells'")
    return {"cells": cells, "per_cell": rep, "stand_sha256": sorted(shas), "stand_sha256_reported": sorted(rep_shas),
            "residual_deadband": sorted(dbs), "residual_deadband_reported": sorted(rep_dbs),
            "reasons": reasons, "notes": notes}


def _k3_complete(t: dict | None) -> list[str]:
    """Missing / short GATED cells (each of ``K3_GATED_CELLS`` present, n >= ENVS_PER_CELL); reported cells are never
    required."""
    TC = _tc()
    if t is None:
        return ["no table / ground JSON"]
    out = [f"{c} missing" for c in TC.K3_GATED_CELLS if c not in t["cells"]]
    out += [f"{c} n {t['cells'][c].get('n_envs')} < {TC.ENVS_PER_CELL}" for c in TC.K3_GATED_CELLS
            if c in t["cells"] and int(t["cells"][c].get("n_envs") or 0) < TC.ENVS_PER_CELL]
    return out


def _k3_reported_line(pa, pr, stt: str):
    """The K3 line on a REPORTED start type (never a clause): None when a side lacks it or the cell sets differ."""
    TC = _tc()
    s, r = (pa or {}).get(stt), (pr or {}).get(stt)
    if s is None or r is None or s.get("cell_ids", s["cells"]) != r.get("cell_ids", r["cells"]):
        return None
    return dict(TC.k3_line(s, r), gated=False)


def gate_k3(role: str, tables: dict, seeds=SEEDS) -> dict:
    TC = _tc()
    cl, inv, rep = {}, [], {}
    for s in seeds:
        tr, rr = (tables.get(role) or {}).get(s), (tables.get("R_h") or {}).get(s)
        if tr is not None:
            inv += [f"[{role}] es{s}: {x}" for x in tr["reasons"]]
        if role != "R_h" and rr is not None:
            inv += [f"[R_h line] es{s}: {x}" for x in rr["reasons"]]
        miss_a, miss_r = _k3_complete(tr), _k3_complete(rr)
        pa = TC.pooled_views(tr["cells"]) if tr is not None and tr["cells"] else None
        pr = TC.pooled_views(rr["cells"]) if rr is not None and rr["cells"] else None
        for stt in TC.START_TYPES:                          # the GATED start type only (the 6 gated cells pooled)
            nr, nf = (f"STOOD rate {stt} es{s} >= R_h's - max({TC.K3_RATE_PP}, {TC.K3_SE_K:g} SE)",
                      f"falls/100 {stt} es{s} <= max({TC.K3_FALLS_FLOOR_PER100:g}, R_h's)")
            if miss_a or miss_r:
                why = f"cells: {miss_a[:6]} / R_h {miss_r[:6]}"
                for n, sh in ((nr, f"stood {stt}"), (nf, f"falls {stt}")):
                    cl[n] = _cl(None, None, None, "pending", s, short=sh, note=why)
                continue
            ln = TC.k3_line(pa[stt], pr[stt])
            cl[nr] = _cl(ln["rate"], ln["rate_line"], ln["rate_pass"], "eval", s, short=f"stood {stt}",
                         note=f"n {ln['n']} vs R_h {ln['ref_rate']:.3f} n {ln['ref_n']}, SE {ln['se_diff']:.4f}")
            cl[nf] = _cl(ln["falls_per100"], ln["falls_line_per100"], ln["falls_pass"], "eval", s,
                         short=f"falls {stt}", note=f"R_h {ln['ref_falls_per100']:.2f} per 100")
        if tr is not None:
            rep[f"es{s}"] = {"gated_cells": list(TC.K3_GATED_CELLS),
                             "gated_cells_present": (pa or {}).get("gated_cells", []),
                             "reported_cells_present": (pa or {}).get("reported_cells", []),
                             "per_cell": tr["per_cell"], "notes": tr["notes"], "stand_sha256": tr["stand_sha256"],
                             "stand_sha256_reported": tr.get("stand_sha256_reported", []),
                             "per_top": (pa or {}).get("per_top"),
                             "per_top_all_poses_reported": (pa or {}).get("per_top_all_poses"),
                             "ground_per_pose_reported": (pa or {}).get("ground_per_pose"),
                             "reported_lines": {st: _k3_reported_line(pa, pr, st) for st in TC.REPORTED_START_TYPES}}
    return dict(verdict_of(cl, inv), clauses=cl, reported=rep, gated_cells=list(TC.K3_GATED_CELLS),
                reported_cells=list(TC.K3_REPORTED_CELLS),
                statistic=f"sh_table_core: the {len(TC.K3_GATED_CELLS)} GATED cells {list(TC.K3_GATED_CELLS)} (n >= "
                          f"{TC.ENVS_PER_CELL} each, per seed) pooled as start type 'tables': k3_line STOOD rate >= "
                          "R_h's - max(5 pp, 2 SE), falls / 100 <= max(1, R_h's); the reported cells (18-in table, "
                          "every rod_min cell, every ground cell) and the ground line are reported, never gated "
                          "(CHARTER section 3, step-4 decision)")


def k3_premise(tables: dict, seeds=SEEDS) -> dict:
    """R_h's own stand-up floor (PLAN step 4 STOP, re-scoped at the step-4 stop): ``sh_table_core.rh_floor`` per seed
    on the gated cells only (per gated top pooled over the gated poses; pending when a gated cell is missing / short);
    reported cells never make a miss."""
    TC = _tc()
    out, misses = {}, []
    for s in seeds:
        rr = (tables.get("R_h") or {}).get(s)
        miss = _k3_complete(rr)
        if miss:
            out[f"es{s}"] = {"state": "pending", "why": miss[:6]}
            continue
        fl = TC.rh_floor(TC.pooled_views(rr["cells"]))
        out[f"es{s}"] = fl
        misses += [f"es{s} {k}" for k, v in fl.items() if not k.startswith("_") and v["gated"] and not v["floor_pass"]]
    return {"seeds": out, "misses": misses, "gated_cells": list(TC.K3_GATED_CELLS),
            "rule": f"sh_table_core.rh_floor: STOOD >= {TC.RH_FLOOR_STOOD} and falls <= {TC.RH_FLOOR_FALLS_PER100:g} "
                    f"per 100 on each gated top {list(TC.K3_GATED_SURFACES)} pooled over the gated poses "
                    f"{list(TC.K3_GATED_POSES)}; the reported cells are listed, never a miss"}


# ===================================================================================== K5
def zero_command_reasons(stamp: str | None) -> list[str]:
    """This campaign's check on a K5 stamp: the launcher's zero command, restore armed, stand_net, and every row
    commanding and limiting 1.064 (fail closed on missing raws)."""
    if not stamp:
        return []
    np = _np()
    meta = _json(Path(stamp) / "run_meta.json") or {}
    out = []
    mode = meta.get("p2r_mode") if isinstance(meta.get("p2r_mode"), dict) else {}
    if mode.get("variant") != EC.VARIANT:
        out.append(f"{stamp}: p2r_mode.variant {mode.get('variant')!r} != {EC.VARIANT!r}")
    blk = sh_block(meta)
    if blk is None:
        return out + [f"{stamp}: run_meta.p2r_mode.{EC.BLOCK_KEY} missing -- the zero command is not recorded"]
    for k, want in (("command", "zero"), ("control", HEAD_CONTROLLER)):
        if blk.get(k) != want:
            out.append(f"{stamp}: {k} {blk.get(k)!r} != {want!r}")
    if restore_armed(blk) is not True:
        out.append(f"{stamp}: restore {blk.get('restore')!r} is not armed")
    if blk.get("height_row") is not None:
        out.append(f"{stamp}: a zero-command stamp carries a height_row record")
    if any(h is not None and abs(float(h) - H_SETTLED) > CMD_TOL for h in (blk.get("hold_heights") or [])):
        out.append(f"{stamp}: hold_heights {blk.get('hold_heights')!r} not all 1.064 / null")
    files = episode_files(stamp)
    if not files:
        return out + [f"{stamp}: no raw episodes"]
    for ep, f in sorted(files.items()):
        d = _load_episode(f, (RAW_H_CMD, RAW_H_LIM))
        if RAW_H_CMD not in d or RAW_H_LIM not in d:
            out.append(f"{stamp}: episode {ep} lacks {RAW_H_CMD} / {RAW_H_LIM} (fail closed)")
            break
        dev = max(float(np.abs(np.asarray(d[k], dtype=np.float64) - H_SETTLED).max(initial=0.0))
                  for k in (RAW_H_CMD, RAW_H_LIM))
        if dev > CMD_TOL:
            out.append(f"{stamp}: episode {ep} commands a height {dev:.4f} m off 1.064 -- not a zero-command pass")
            break
    return out


def k5_from_calib(out: dict, rows: dict, extra: dict | None = None) -> dict:
    """K5 per role from a ``sn_calib.calibrate`` result (pure). ``rows``: {role: sn_calib row name};
    ``extra``: {role: [this campaign's validity reasons]}. Verdict: INVALID if a gate (or an extra reason) is INVALID,
    FAIL if any of G1-G5 fails, PENDING if any is PENDING / PARTIAL, else PASS."""
    snc = _snc()
    res = {}
    for role, row in rows.items():
        r = (out.get("rows") or {}).get(row)
        cl, inv = {}, list((extra or {}).get(role) or [])
        if not r:
            for g in snc.GATES:
                cl[f"{g} {snc.GATE_TITLE[g]}"] = _cl(None, PASS, None, "pending", None, short=g,
                                                     note=f"sn_calib row {row} not read")
        else:
            for g in snc.GATES:
                v = r["gates"][g]["verdict"]
                if v == INVALID:
                    inv += [f"[{role} -> {row}] {g}: {x}" for x in r["gates"][g].get("invalid") or []] or \
                           [f"[{role} -> {row}] {g}: INVALID"]
                state = "pending" if v in (PENDING, PARTIAL) else "eval"
                cl[f"{g} {snc.GATE_TITLE[g]}"] = _cl(v, PASS, v == PASS, state, None, short=g,
                                                     note=f"sn_calib {row}: failed {r['gates'][g].get('failed')}")
        res[role] = dict(verdict_of(cl, inv), clauses=cl, sn_calib_row=row,
                         statistic="sn_calib (unchanged reads and lines) G1-G5 on the zero-command stamps; all five "
                                   "PASS")
    return res


def _stamp_modes(stamps: dict) -> dict:
    return {k: (_json(Path(st) / "run_meta.json") or {}).get("p2r_mode") for k, st in stamps.items() if st}


def k5_read(spec: dict, heads: dict, *, snc=None) -> dict:
    """K5 (IO): sn_calib's reads for its references + each head's zero-command name (validity kind ``obs0``) +
    A-mask s17 (``stand_net``), merged and calibrated exactly as ``sn_calib.run`` does (no identity check, nothing
    written); then this campaign's rule on the heads' stamps."""
    snc = snc or _snc()
    amask = (spec.get("k5") or {}).get("amask") or SN_AMASK_HEAD
    tasks = {role: (role, snc.NAMES[role], snc.KIND[role]) for role in tuple(snc.ROLES) + ("R_SN",)}
    for name in dict.fromkeys(heads.values()):
        tasks[name] = (name, name, K5_HEAD_KIND)
    tasks[amask] = (amask, amask, "stand_net")
    reads = {role: snc.read_reference(*a) for role, a in tasks.items()}
    reads["R"] = snc.merge_r(reads["R"], reads.pop("R_SN"))
    names = tuple(dict.fromkeys(list(heads.values()) + [amask]))
    out = snc.calibrate(reads, None, heads=names)
    refs_sha = (((snc.valid_only_point(snc.NAMES["REFS"]) or {}).get("valid") or {}).get("suite@1") or {}).get(
        "ckpt_sha")
    extra, head_shas, head_dbs, head_par = {}, {}, {}, {}
    for role, name in heads.items():
        stamps = {k: v for k, v in (((out.get("rows") or {}).get(name) or {}).get("stamps") or {}).items() if v}
        rs = [x for st in stamps.values() for x in zero_command_reasons(st)]
        modes = _stamp_modes(stamps)
        for k, m in modes.items():
            got = (m or {}).get("s1_checkpoint_sha256")
            if not refs_sha or got != refs_sha:
                rs.append(f"{name} {k}: p2r_mode.s1_checkpoint_sha256 {str(got)[:12]} != REFS {str(refs_sha)[:12]}")
        cons = _snc().stand_net_consistency(name, {k: m for k, m in modes.items() if m})
        rs += sorted({x for v in cons.values() for x in v})[:1]
        extra[role] = rs
        head_shas[role] = sorted({(m.get("specialist") or {}).get("sha256") for m in modes.values()
                                  if isinstance(m, dict) and isinstance(m.get("specialist"), dict)} - {None})
        dbs, par = set(), []
        for k, m in modes.items():
            blk = (m or {}).get(EC.BLOCK_KEY)
            try:
                ev = CORE.recorded_deadband(blk)
            except ValueError as exc:
                rs.append(f"{name} {k}: {EC.DEADBAND_KEY}: {exc}")
                continue
            dbs.add(ev)
            try:                                   # evaluated vs trained: a note in run(), never a reason
                par.append({"stamp": f"{name} {k}", "evaluated": ev, "trained": CORE.recorded_trained_deadband(blk)})
            except ValueError as exc:
                rs.append(f"{name} {k}: {CORE.TRAINED_DEADBAND_KEY}: {exc}")
        head_dbs[role] = sorted(dbs)
        head_par[role] = par
    rows = {"R_h": SN_CALIB_R_ROW, "Amask": amask, **heads}
    return {"gates": k5_from_calib(out, rows, extra), "sn_calib_mismatches": out.get("mismatches"),
            "sn_calib_summary": out.get("summary"), "rows": rows, "campaign_reasons": extra,
            "head_stand_sha256": head_shas, "head_residual_deadband": head_dbs, "head_deadband_parity": head_par,
            "refs_s1_sha256": refs_sha}


# ===================================================================================== forces per height (reported)
def force_rows(stamp: str) -> dict | None:
    """sh_facts' settled stand rows of one stamp -> per-row foot-load and PD-torque rod forces (IO). None if no row;
    a stamp lacking a force raw (:data:`FORCE_RAW_REQUIRED`) raises KeyError (the caller fails closed)."""
    np, F = _np(), _facts()
    K = F.K
    for ep, f in sorted(episode_files(stamp).items()):
        with np.load(f) as z:
            miss = [k for k in FORCE_RAW_REQUIRED if k not in z.files]
        if miss:
            raise KeyError(f"episode {ep}: force raws missing {miss}")
    try:
        rows = F._settled_rows(Path(stamp))
    except ValueError:            # np.concatenate of nothing: no settled row
        return None
    th, tk = rows["th"], rows["tk"]
    fl_h, fl_k = K.rod_forces(th, tk, *K.stance_joint_torques(th, tk, rows["ff"], leg_weight=True))
    pd_h, pd_k = K.rod_forces(th, tk, rows["tau_h"], rows["tau_k"])
    return {"h": np.asarray(rows["h"], dtype=np.float64), "fl_h": np.abs(fl_h), "fl_k": np.abs(fl_k),
            "pd_h": np.abs(pd_h), "pd_k": np.abs(pd_k), "nload": rows["nload"]}


def force_table(fr: list[dict]) -> dict:
    """Per h_meas bin over the given row sets (pure): six-leg mean |F_hip| foot-load (the cap statistic) and PD,
    p90, worst leg, knee."""
    np = _np()
    if not fr:
        return {}
    h = np.concatenate([x["h"] for x in fr])
    cat = {k: np.concatenate([x[k] for x in fr]) for k in ("fl_h", "fl_k", "pd_h", "pd_k", "nload")}
    bins = np.array([-1 if height_bin(v) is None else height_bin(v) for v in h])
    out = {}
    for b in list(range(K_BINS)) + [-1]:
        m = bins == b
        if not m.any():
            continue
        lab = bin_label(b) if b >= 0 else "outside [h_min, h_max]"
        fl6, pd6 = cat["fl_h"][m].mean(axis=1), cat["pd_h"][m].mean(axis=1)
        leg_fl, leg_pd = cat["fl_h"][m].mean(axis=0), cat["pd_h"][m].mean(axis=0)
        kn_fl, kn_pd = cat["fl_k"][m].mean(axis=0), cat["pd_k"][m].mean(axis=0)
        out[lab] = {"n_rows": int(m.sum()), "h_meas_mean_m": float(h[m].mean()),
                    "share_6_loaded": float((cat["nload"][m] == 6).mean()),
                    "F_hip_six_leg_mean_foot_load_N": float(fl6.mean()),
                    "F_hip_six_leg_mean_foot_load_p90_N": float(np.percentile(fl6, 90)),
                    "F_hip_six_leg_mean_pd_torque_N": float(pd6.mean()),
                    "worst_hip_leg_foot_load": {"leg": LEGS[int(leg_fl.argmax())], "mean_abs_N": float(leg_fl.max())},
                    "worst_hip_leg_pd_torque": {"leg": LEGS[int(leg_pd.argmax())], "mean_abs_N": float(leg_pd.max())},
                    "F_knee_six_leg_mean_foot_load_N": float(kn_fl.mean()),
                    "F_knee_six_leg_mean_pd_torque_N": float(kn_pd.mean()),
                    "worst_knee_leg_foot_load": {"leg": LEGS[int(kn_fl.argmax())], "mean_abs_N": float(kn_fl.max())},
                    "over_cap_foot_load": bool(fl6.mean() > FORCE_CAP_N)}
    return out


def at_h_max(rh_rows: list[dict], tol: float = AT_HMAX_TOL_M) -> dict:
    """The SAME statistic (six-leg mean |F_hip| from the measured foot loads, the PD reading beside it) on the settled
    R_h rows with |h_meas - H_MAX| <= ``tol`` only (pure; reported). The top bin [1.221, 1.346] mixes the 1.250 /
    1.300 / 1.346 holds, so its mean sits near h ~ 1.29, and the foot-load force rises ~1.3 N per mm of height
    (facts.md: 1870 N at 1.25 -> 2004 N at 1.35): the bin mean can pass while F(h_max) > 2000 N."""
    np = _np()
    lo, hi = H_MAX - float(tol), H_MAX + float(tol)
    out = {"window_m": [lo, hi], "n_rows": 0, "value_N": None, "pd_torque_N": None, "h_meas_mean_m": None,
           "over_cap": None}
    if not rh_rows:
        return out
    h = np.concatenate([np.asarray(x["h"], dtype=np.float64) for x in rh_rows])
    fl = np.concatenate([np.asarray(x["fl_h"], dtype=np.float64) for x in rh_rows])
    pd = np.concatenate([np.asarray(x["pd_h"], dtype=np.float64) for x in rh_rows])
    m = np.abs(h - H_MAX) <= float(tol) + EPS
    if not m.any():
        return out
    v = float(fl[m].mean(axis=1).mean())
    return dict(out, n_rows=int(m.sum()), value_N=v, pd_torque_N=float(pd[m].mean(axis=1).mean()),
                h_meas_mean_m=float(h[m].mean()), over_cap=bool(v > FORCE_CAP_N + EPS))


def h_max_recheck(rh_rows: list[dict], errors=()) -> dict:
    """R_h's top bin (the one holding H_MAX), es1+es2 pooled: foot-load six-leg mean |F_hip| <= 2000 N -- the
    PRE-REGISTERED decision statistic (``state``). ``errors``: R_h stamps whose force raws could not be read -> INVALID
    (the cap cannot be re-checked: a step-4 STOP). Also computed, reported and FLAGGED (``flags``; not the decision
    unless CHARTER section 3 logs the change before the read): the same statistic on the rows within 2 cm of h_max
    (:func:`at_h_max`)."""
    if errors:
        return {"state": "INVALID", "bin": bin_label(K_BINS - 1), "errors": list(errors), "flags": []}
    t = force_table(rh_rows)
    lab = bin_label(K_BINS - 1)
    b = t.get(lab)
    near = at_h_max(rh_rows)
    flags = []
    if near["over_cap"]:
        flags.append(f"AT_H_MAX_OVER_CAP: six-leg mean |F_hip| (foot loads) {near['value_N']:.1f} N > "
                     f"{FORCE_CAP_N:g} N on the {near['n_rows']} settled R_h rows within {AT_HMAX_TOL_M} m of h_max "
                     f"{H_MAX} (the top-bin mean is the pre-registered decision statistic; read both before deciding)")
    elif near["n_rows"] == 0:
        flags.append(f"AT_H_MAX_NO_ROWS: no settled R_h row within {AT_HMAX_TOL_M} m of h_max {H_MAX}")
    if not b:
        return {"state": "pending", "bin": lab, "note": "no settled R_h row in the top bin", "at_h_max": near,
                "flags": flags}
    v = b["F_hip_six_leg_mean_foot_load_N"]
    return {"state": "ok" if v <= FORCE_CAP_N + EPS else "REFREEZE", "bin": lab, "value_N": v, "cap_N": FORCE_CAP_N,
            "pd_torque_N": b["F_hip_six_leg_mean_pd_torque_N"], "n_rows": b["n_rows"],
            "h_meas_mean_m": b["h_meas_mean_m"], "at_h_max": near, "flags": flags,
            "rule": "PLAN step 4 / CHARTER 2026-10-13: six-leg mean |F_hip| from R_h's measured foot loads through the "
                    "linkage in its top bin; over 2000 N -> lower h_max and re-freeze before any training. Reported "
                    f"and flagged beside it: the same statistic within {AT_HMAX_TOL_M} m of h_max (at_h_max)"}


# ===================================================================================== control table + heads
def lands(gate: str, role: str, res: dict) -> tuple[str, str | None]:
    """("yes" / "NO" / "pending" / "reported", why) for one control cell (sn_calib.lands semantics; a named clause
    must fail at EVERY seed read)."""
    exp = EXPECT.get((gate, role))
    v = res.get("verdict")
    if exp is None:
        return "reported", None
    if v == INVALID:
        return "NO", "INVALID (a listed cell cannot be read)"
    if v in (PENDING, PARTIAL):
        return "pending", v
    if v != exp:
        return "NO", f"expected {exp}, observed {v}"
    named = NAMED_CLAUSE.get((gate, role))
    if named and exp == FAIL:
        cl = res.get("clauses") or {}
        seeds = sorted({c["seed"] for c in cl.values() if c["short"] == named and c["state"] == "eval"})
        bad_seeds = sorted({c["seed"] for c in cl.values() if c["short"] == named and c["state"] == "eval"
                            and not c["ok"]})
        if not seeds or bad_seeds != seeds:
            return "NO", f"fails, but not on its named clause '{named}' at every seed read ({bad_seeds} of {seeds})"
    return "yes", None


def _force_invalid(res: dict, reasons: list[str]) -> dict:
    """Re-combine a gate result with extra INVALID reasons (a head's mixed checkpoints)."""
    return dict(res, **verdict_of(res.get("clauses") or {}, list(res.get("invalid") or []) + list(reasons)))


# ===================================================================================== spec + IO
def load_spec(path) -> dict:
    spec = json.loads(Path(path).read_text())
    roles = spec.get("roles") or {}
    bad = [r for r, v in roles.items() if r not in CONTROL_ROLES and not (v or {}).get("head")]
    if bad:
        raise SystemExit(f"sh_read: spec roles {bad} are neither a control {CONTROL_ROLES} nor marked head: true")
    clash = [r for r, v in roles.items() if (v or {}).get("head") and r in CONTROL_ROLES]
    if clash:
        raise SystemExit(f"sh_read: a head may not take a control's role name: {clash}")
    return spec


def resolve_stamp(spec: dict, role: str, row: str, s: int) -> str | None:
    r = (spec.get("roles") or {}).get(role) or {}
    st = ((r.get("stamps") or {}).get(row) or {})
    if str(s) in st or s in st:
        v = st.get(str(s), st.get(s))
        return str(v) if v else None
    if not r.get("name"):
        return None
    P = _P()
    lr = Path(spec["lr_root"]) if spec.get("lr_root") else P.LROUT
    return P.newest_stamp(r["name"], row, s, lr)


def _forbidden(p: Path) -> bool:
    rp = Path(p).resolve()
    return any(rp == d.resolve() or d.resolve() in rp.parents for d in FORBIDDEN_OUT_DIRS)


def run(spec: dict, *, k5: bool = True, forces: bool = True, snc=None) -> dict:
    """The whole read (IO): every role's height-row stamps and table JSON, every gate, the control table, the heads."""
    seeds = tuple(int(x) for x in (spec.get("seeds") or SEEDS))
    roles = spec.get("roles") or {}
    heads = [r for r, v in roles.items() if (v or {}).get("head")]
    rg = row_gates(spec)
    reads, views, tables = {}, {}, {}
    for role, rv in roles.items():
        rule = role_rule(role, role in heads)
        reads[role] = {s: {row: read_height_stamp(resolve_stamp(spec, role, row, s), row=row, gates=gates, rule=rule)
                           for row, gates in rg.items()} for s in seeds}
        views[role] = {s: {g: gate_view(reads[role][s], rg, g) for g in RAW_GATES} for s in seeds}
        tb = (rv or {}).get("table") or {}
        ctl = HEAD_CONTROLLER if role in heads else ROLE_TABLE_CONTROLLER.get(role)
        tables[role] = {s: read_table_cells(tb.get(str(s), tb.get(s)), s, controller=ctl,
                                            amask=role in ("Amask", "Amask_q", "Amask_old"))
                        for s in seeds if tb.get(str(s), tb.get(s))}
    refs_on = refs_onsets(spec, seeds)
    gate_roles = {g: [r for r in roles if (g, r) in EXPECT or r in REPORTED_ROLES[g] or r in heads] for g in GATES}
    res = {g: {} for g in GATES}
    for r in gate_roles["K1"]:
        res["K1"][r] = gate_k1(r, views, seeds)
    for r in gate_roles["K2"]:
        res["K2"][r] = gate_k2(r, views, seeds)
    for r in gate_roles["K3"]:
        res["K3"][r] = gate_k3(r, tables, seeds)
    for r in gate_roles["K4"]:
        res["K4"][r] = gate_k4(r, views, refs_on, seeds)
    for r in gate_roles["K6"]:
        res["K6"][r] = gate_k6(r, views, seeds)
    k5_heads = {r: roles[r]["k5_name"] for r in heads if roles[r].get("k5_name")}
    k5_out = None
    if k5 and not (spec.get("k5") or {}).get("skip"):
        k5_out = k5_read(spec, k5_heads, snc=snc)
        res["K5"] = dict(k5_out["gates"])
    for r in heads:
        if r not in res["K5"]:
            c = {"K5": _cl(None, PASS, None, "pending", None, short="K5", note="no k5_name / K5 not read")}
            res["K5"][r] = dict(verdict_of(c, []), clauses=c, statistic="not read")
    # one stand checkpoint per head (height rows, K5 rows, table cells)
    head_ckpt = {}
    for h in heads:
        shas = {p.get("stand_sha256") for s in seeds for p in reads[h][s].values() if p.get("status") == "ok"}
        shas |= {x for t in tables[h].values() for x in t["stand_sha256"]}
        shas |= set(((k5_out or {}).get("head_stand_sha256") or {}).get(h) or [])
        shas.discard(None)
        head_ckpt[h] = sorted(shas)
        if len(shas) > 1:
            why = [f"[{h}] its stamps / cells name {len(shas)} stand checkpoints {[x[:12] for x in sorted(shas)]} "
                   "(one checkpoint per head)"]
            for g in GATES:
                if h in res[g]:
                    res[g][h] = _force_invalid(res[g][h], why)
    # one residual deadband per head (CHARTER 2026-10-14; absent = 0): ok height stamps, gated table cells, K5 stamps
    head_db = {}
    for h in heads:
        dbs = {p.get("residual_deadband", 0.0) for s in seeds for p in reads[h][s].values() if p.get("status") == "ok"}
        dbs |= {x for t in tables[h].values() for x in t.get("residual_deadband", [])}
        dbs |= set(((k5_out or {}).get("head_residual_deadband") or {}).get(h) or [])
        head_db[h] = sorted(dbs)
        if len(dbs) > 1:
            why = [f"[{h}] its stamps / cells mix residual deadbands {sorted(dbs)} (one deadband per head; a stamp or "
                   "cell without the record ran at 0)"]
            for g in GATES:
                if h in res[g]:
                    res[g][h] = _force_invalid(res[g][h], why)
    # the deadband each head's checkpoint was TRAINED with vs the one it was evaluated at (per stamp; height + K5
    # stamps carry both): a difference is a NOTE, never a refusal (the step-6 probe evaluates a 0-trained head at 0.25)
    head_trained, head_db_notes = {}, {}
    for h in heads:
        par = [{"stamp": f"es{s} {row}", "evaluated": p.get("residual_deadband", 0.0),
                "trained": p.get("trained_residual_deadband")}
               for s in seeds for row, p in reads[h][s].items() if p.get("status") == "ok"]
        par += list(((k5_out or {}).get("head_deadband_parity") or {}).get(h) or [])
        head_trained[h] = sorted({x["trained"] for x in par if x["trained"] is not None})
        off = [x for x in par if x["trained"] is not None and x["trained"] != x["evaluated"]]
        pairs = sorted({(x["evaluated"], x["trained"]) for x in off})
        head_db_notes[h] = ([f"[{h}] {len(off)} of {len(par)} stamps were evaluated at a residual deadband other than "
                             f"the one their stand checkpoint was trained at ((evaluated, trained) {pairs}; e.g. "
                             f"{off[0]['stamp']}) -- a note, not a refusal: intended only for a probe"] if off else [])
    for (g, r) in EXPECT:
        if r not in res[g]:
            res[g][r] = {"verdict": PENDING, "clauses": {}, "failed": [], "pending": ["role not in the spec"],
                         "invalid": []}
    mism, pend, table = [], [], []
    for (g, r), exp in EXPECT.items():
        ld, why = lands(g, r, res[g][r])
        res[g][r]["expected"], res[g][r]["lands"], res[g][r]["lands_why"] = exp, ld, why
        table.append({"gate": g, "role": r, "expected": exp, "observed": res[g][r]["verdict"], "lands": ld,
                      "why": why})
        if ld == "NO":
            mism.append(f"{g} {GATE_TITLE[g]} x {r} ({ROLE_TITLE.get(r, r)}): {why}")
        elif ld == "pending":
            pend.append(f"{g} {GATE_TITLE[g]} x {r}: {why} -- {'; '.join(res[g][r].get('pending') or [])[:300]}")
    head_out = {}
    for h in heads:
        v = {g: (res[g].get(h) or {}).get("verdict", PENDING) for g in GATES}
        head_out[h] = {"verdicts": v, "all_pass": all(x == PASS for x in v.values()),
                       "failed": [g for g, x in v.items() if x in (FAIL, INVALID)],
                       "pending": [g for g, x in v.items() if x in (PENDING, PARTIAL)],
                       "stand_sha256": head_ckpt.get(h),
                       "residual_deadband": (head_db[h][0] if len(head_db.get(h) or []) == 1 else
                                             (head_db.get(h) or None)),
                       "trained_residual_deadband": (head_trained[h][0] if len(head_trained[h]) == 1 else
                                                     (head_trained[h] or None)),
                       "deadband_notes": head_db_notes[h]}
    premise = k3_premise(tables, seeds)
    frc, hmax = {}, {"state": "not read"}
    if forces:
        rows_by_role, errs = {}, {}
        for role in roles:
            rr = []
            for s in seeds:
                for row, p in reads[role][s].items():
                    if p.get("status") != "ok":
                        continue
                    try:
                        fr = force_rows(p["stamp"])
                    except (KeyError, OSError) as exc:
                        errs.setdefault(role, []).append(f"es{s} {row} {p['stamp']}: {exc}")
                        frc.setdefault(role, {})[f"es{s} {row}"] = {"error": str(exc)}
                        continue
                    if fr is not None:
                        rr.append(fr)
                        frc.setdefault(role, {})[f"es{s} {row}"] = force_table([fr])
            rows_by_role[role] = rr
            if rr:
                frc.setdefault(role, {})["pooled"] = force_table(rr)
        hmax = (h_max_recheck(rows_by_role.get("R_h") or [], errs.get("R_h") or ()) if "R_h" in roles
                else {"state": "pending", "note": "no R_h role"})
    keep = ("stamp", "status", "reasons", "n_episodes", "controller", "restore", "stand_ckpt_kind", "stand_sha256",
            "k2_falls", "episodes_fell", "dose", "residual_deadband", "trained_residual_deadband")
    return {"gates": res, "control_table": table, "mismatches": mism, "pending": pend, "heads": head_out,
            "premise_rh_standup": premise, "forces": frc, "h_max_recheck": hmax, "refs_onset": refs_on,
            "row_gates": {r: list(g) for r, g in rg.items()},
            "k5": None if k5_out is None else {k: v for k, v in k5_out.items() if k != "gates"},
            "reads": {r: {f"es{s}": {row: {k: v for k, v in p.items() if k in keep} for row, p in by.items()}
                          for s, by in reads[r].items()} for r in roles},
            "tables": {r: {f"es{s}": {"reasons": t["reasons"], "notes": t["notes"], "cells": sorted(t["cells"]),
                                      "stand_sha256": t.get("stand_sha256", []),
                                      "stand_sha256_reported": t.get("stand_sha256_reported", []),
                                      "residual_deadband": t.get("residual_deadband", []),
                                      "residual_deadband_reported": t.get("residual_deadband_reported", [])}
                           for s, t in v.items()} for r, v in tables.items()},
            "seeds": list(seeds)}


def exit_code(out: dict) -> int:
    if out["mismatches"] or out["premise_rh_standup"]["misses"] or (out.get("h_max_recheck") or {}).get(
            "state") in ("REFREEZE", "INVALID"):
        return 4
    if any(h["failed"] for h in out["heads"].values()):
        return 3
    if out["pending"] or any(h["pending"] for h in out["heads"].values()):
        return 2
    return 0


# ===================================================================================== markdown
def _fmt(v, nd=4):
    if v is None:
        return "--"
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, (int, float)):
        if isinstance(v, float) and math.isinf(v):
            return "inf"
        return f"{v:.{nd}g}"
    return str(v)


def _cells(res: dict) -> str:
    parts = []
    for k, c in (res.get("clauses") or {}).items():
        tag = (c.get("short") or k) + (f" es{c['seed']}" if c.get("seed") is not None else "")
        if c["state"] == "pending":
            parts.append(f"{tag} ?")
        elif c["state"] == "na":
            parts.append(f"{tag} n/a")
        elif not c["ok"]:
            parts.append(f"**{tag} {_fmt(c['value'])} (line {_fmt(c['line'])})**")
        else:
            parts.append(f"{tag} {_fmt(c['value'])}")
    s = "; ".join(parts)
    return s if len(s) < 1500 else s[:1500] + " ..."


def markdown(out: dict) -> str:
    m = out["meta"]
    L = ["# Stand height -- K1-K6 read (PLAN.md H7)", "",
         f"Generated {m['generated']} by `scripts/sh_read.py` (sha256 {m['tool_sha256'][:12]}; CPU, read-only). "
         f"Env seeds {', '.join('es' + str(s) for s in out['seeds'])}; height rows "
         f"{', '.join(f'{r} -> {g}' for r, g in out['row_gates'].items())}. Statistics as pinned in the module "
         "docstring.", "",
         f"**Control table: {sum(1 for x in out['control_table'] if x['lands'] == 'yes')} / "
         f"{len(out['control_table'])} cells land, {len(out['mismatches'])} mismatch(es), "
         f"{sum(1 for x in out['control_table'] if x['lands'] == 'pending')} pending. Exit {m['exit_code']}.**", ""]
    if out["mismatches"]:
        L += ["**Mismatches (step 4: STOP, notify, ask):**", ""] + [f"- {x}" for x in out["mismatches"]] + [""]
    pr = out["premise_rh_standup"]
    pend = [f"{es}: {'; '.join(str(x) for x in (v.get('why') or []))}" for es, v in (pr.get("seeds") or {}).items()
            if isinstance(v, dict) and v.get("state") == "pending"]
    L += [f"**R_h stand-up premise ({pr['rule']}):** "
          + ("MISS in " + ", ".join(pr["misses"]) if pr["misses"] else "")
          + (("; " if pr["misses"] else "") + "pending (" + " | ".join(pend) + ")" if pend else "")
          + ("" if pr["misses"] or pend else "no miss read"), ""]
    TC = _tc()
    L += [f"**K3 cells:** GATED ({len(TC.K3_GATED_CELLS)}; the K3 line and the R_h floor read these only): "
          + ", ".join(TC.K3_GATED_CELLS) + f". REPORTED ({len(TC.K3_REPORTED_CELLS)}; run and read, never gated): "
          + ", ".join(TC.K3_REPORTED_CELLS) + ".", ""]
    sha_rows = []
    for role, by in (out.get("tables") or {}).items():
        for es, t in by.items():
            g, rp = list(t.get("stand_sha256") or []), list(t.get("stand_sha256_reported") or [])
            if g or rp:
                sha_rows.append(f"- {role} {es}: gated cells' stand checkpoint {[x[:12] for x in g]}; reported cells' "
                                f"{[x[:12] for x in rp]}" + (" -- **reported cells ran another checkpoint (their "
                                                              "numbers are not this head's)**" if set(rp) - set(g)
                                                              else ""))
    if sha_rows:
        L += ["**K3 stand checkpoints (the one-checkpoint rule reads the gated cells only):**", ""] + sha_rows + [""]
    hm = out.get("h_max_recheck") or {}
    near = hm.get("at_h_max") or {}
    L += [f"**h_max re-check (R_h top bin {hm.get('bin')}):** {hm.get('state')}"
          + (f" -- foot-load six-leg mean |F_hip| {_fmt(hm.get('value_N'))} N vs cap {_fmt(hm.get('cap_N'))} N (PD "
             f"reading {_fmt(hm.get('pd_torque_N'))} N, {hm.get('n_rows')} rows, mean h {_fmt(hm.get('h_meas_mean_m'))}"
             " m)" if hm.get("value_N") is not None else "") + (f" -- {hm['errors'][0][:300]}" if hm.get("errors")
                                                                 else ""),
          f"  Reported beside it (not the decision statistic): within {AT_HMAX_TOL_M} m of h_max, "
          f"{_fmt(near.get('value_N'))} N (PD {_fmt(near.get('pd_torque_N'))} N, {near.get('n_rows')} rows)."]
    L += [f"  **FLAG** {x}" for x in (hm.get("flags") or [])] + [""]
    if out["heads"]:
        L += ["## Heads", "", "| head | " + " | ".join(GATES) + " | all PASS |", "|---|" + "---|" * (len(GATES) + 1)]
        for h, v in out["heads"].items():
            L.append(f"| {h} | " + " | ".join(v["verdicts"][g] for g in GATES) + f" | {v['all_pass']} |")
        L.append("")
        for h, v in out["heads"].items():
            L.append(f"- {h}: residual deadband {v.get('residual_deadband')!r} (one per head; absent record = 0); "
                     f"trained at {v.get('trained_residual_deadband')!r} (null = not recorded)")
            L += [f"  - **NOTE** {x}" for x in v.get("deadband_notes") or []]
        L.append("")
    L += ["## Gates", "", "| gate | role | expected | observed | lands | values |", "|---|---|---|---|---|---|"]
    for g in GATES:
        for r, res in out["gates"][g].items():
            ld = res.get("lands") or ("head" if r in out["heads"] else "reported")
            L.append(f"| {g} {GATE_TITLE[g]} | {r} | {res.get('expected') or '--'} | {res.get('verdict')} | "
                     f"{ld}{': ' + res['lands_why'] if res.get('lands_why') and ld != 'yes' else ''} | {_cells(res)} |")
    L += ["", "## Pending", ""] + ([f"- {p}" for p in out["pending"]] or ["- none"])
    inv = [f"{g} x {r}: {'; '.join(res.get('invalid') or [])[:400]}" for g in GATES
           for r, res in out["gates"][g].items() if res.get("verdict") == INVALID]
    L += ["", "## INVALID cells", ""] + ([f"- {x}" for x in inv] or ["- none"])
    L += ["", "## Forces per height bin (reported; foot-load statics = the cap statistic, PD torque alongside)", "",
          "| role | bin | rows | h_meas | F_hip 6-leg mean foot-load (N) | p90 | PD reading (N) | worst hip leg "
          "(foot-load) | F_knee foot-load / PD (N) | over cap |", "|---|---|---|---|---|---|---|---|---|---|"]
    for role, by in (out.get("forces") or {}).items():
        for k, v in by.items():
            if isinstance(v, dict) and v.get("error"):
                L.append(f"| {role} | {k}: force raws unreadable -- {v['error'][:200]} | | | | | | | | |")
        for lab, b in (by.get("pooled") or {}).items():
            L.append(f"| {role} | {lab} | {b['n_rows']} | {_fmt(b['h_meas_mean_m'])} | "
                     f"{_fmt(b['F_hip_six_leg_mean_foot_load_N'])} | {_fmt(b['F_hip_six_leg_mean_foot_load_p90_N'])} | "
                     f"{_fmt(b['F_hip_six_leg_mean_pd_torque_N'])} | {b['worst_hip_leg_foot_load']['leg']} "
                     f"{_fmt(b['worst_hip_leg_foot_load']['mean_abs_N'])} | "
                     f"{_fmt(b['F_knee_six_leg_mean_foot_load_N'])} / {_fmt(b['F_knee_six_leg_mean_pd_torque_N'])} | "
                     f"{b['over_cap_foot_load']} |")
    L += ["", "## Reads", ""]
    for r, by in out["reads"].items():
        for s, rows in by.items():
            for row, x in rows.items():
                L.append(f"- {r} {s} {row}: {x.get('status')} `{x.get('stamp')}`"
                         + (f" -- {'; '.join(x.get('reasons') or [])[:300]}" if x.get("reasons") else ""))
    L += ["", "Full per-clause values, per-bin reads and stamp paths: the JSON.", ""]
    return "\n".join(L)


# ===================================================================================== main
def constants() -> dict:
    """Every pinned constant (run records)."""
    return {"h_settled": H_SETTLED, "h_min": H_MIN, "h_max": H_MAX, "bins": [bin_label(b) for b in range(K_BINS)],
            "k1": {"median_m": K1_MED_M, "p90_m": K1_P90_M, "window_s": K1_WINDOW_S,
                   "arrival_delay_s": K1_ARRIVAL_DELAY_S},
            "k2": {"tol_m": TRACK_TOL_M, "reach_hold_s": REACH_HOLD_S, "extra_s": K2_EXTRA_S, "q": K2_Q,
                   "fall_floor": FALL_FLOOR},
            "k3": "sh_table_core (K3_GATED_CELLS decide, K3_REPORTED_CELLS reported; ENVS_PER_CELL, classify, "
                  "pooled_views, k3_line, rh_floor)",
            "k3_gated_cells": list(_tc().K3_GATED_CELLS), "k3_reported_cells": list(_tc().K3_REPORTED_CELLS),
            "k4": {"engage_m": CORE.RESTORE_ENGAGE_M, "rod_tol_m": CORE.RESTORE_ROD_TOL_M,
                   "settle_s": CORE.RESTORE_SETTLE_S, "exit_fall_s": K4_EXIT_FALL_S,
                   "onset_extra_s": K4_ONSET_EXTRA_S, "q": K4_Q},
            "k5": {"head_validity_kind": K5_HEAD_KIND, "r_row": SN_CALIB_R_ROW, "amask_head": SN_AMASK_HEAD},
            "k6": {"share_pp": SHARE_PP, "se_k": SHARE_SE_K}, "force_cap_n": FORCE_CAP_N,
            "at_h_max_tol_m": AT_HMAX_TOL_M,
            "residual_deadband": "one value per head over its ok height stamps, gated table cells and K5 stamps "
                                 "(sh_core.recorded_deadband: absent = 0); a mix = every gate of the head INVALID; "
                                 "controls not held to it",
            "trained_residual_deadband": "per stamp (height + K5), sh_core.deadband_parity: a trained value recorded "
                                         "and different from the evaluated one = a head NOTE (deadband_notes), never "
                                         "a refusal; malformed = the stamp INVALID",
            "raw_keys": {"slot": RAW_SLOT, "stand_eff": RAW_EFF, "h_cmd": RAW_H_CMD, "h_lim": RAW_H_LIM,
                         "restoring": RAW_RESTORING, "q": RAW_Q, "a_net": RAW_A_NET},
            "expect": {f"{g} {r}": v for (g, r), v in EXPECT.items()},
            "named_clause": {f"{g} {r}": v for (g, r), v in NAMED_CLAUSE.items()}}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Stand-height K1-K6 readers (PLAN.md H7); CPU only, read-only.")
    ap.add_argument("--spec", required=True, type=Path, help="the read spec (roles -> stamps / names / table JSON)")
    ap.add_argument("--json", required=True, type=Path)
    ap.add_argument("--md", required=True, type=Path)
    ap.add_argument("--no-k5", action="store_true", help="skip the sn_calib K5 read (K5 then reads PENDING)")
    ap.add_argument("--no-forces", action="store_true")
    a = ap.parse_args(argv)
    for p in (a.json, a.md):
        if _forbidden(p):
            raise SystemExit(f"sh_read: refusing to write {p} inside {[str(d) for d in FORBIDDEN_OUT_DIRS]}")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    sys.dont_write_bytecode = True
    spec = load_spec(a.spec)
    out = run(spec, k5=not a.no_k5, forces=not a.no_forces)
    out["meta"] = {"generated": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
                   "tool": str(Path(__file__).resolve()), "tool_sha256": _sha(Path(__file__).resolve()),
                   "sh_core_sha256": _sha(CORE.__file__), "sh_eval_common_sha256": _sha(EC_PATH),
                   "sh_table_core_sha256": _sha(TABLE_CORE_PATH), "sn_calib_sha256": _sha(SN_CALIB_PATH),
                   "sh_facts_sha256": _sha(SH_FACTS_PATH), "sh_kin_sha256": _sha(CORE.SH_KIN_PATH),
                   "p2_read_sha256": _sha(P2_DIR / "scripts" / "p2_read.py"),
                   "p2s_stand_sha256": _sha(P2_DIR / "scripts" / "p2s_stand.py"), "spec": str(a.spec),
                   "spec_body": spec, "constants": constants()}
    out["meta"]["exit_code"] = exit_code(out)
    a.json.parent.mkdir(parents=True, exist_ok=True)
    a.md.parent.mkdir(parents=True, exist_ok=True)
    a.json.write_text(json.dumps(_jsonable(out), indent=1, default=str))
    a.md.write_text(markdown(out))
    print(f"sh_read: {sum(1 for x in out['control_table'] if x['lands'] == 'yes')}/{len(out['control_table'])} "
          f"control cells land, {len(out['mismatches'])} mismatch(es), {len(out['pending'])} pending; heads "
          f"{ {h: v['all_pass'] for h, v in out['heads'].items()} }; wrote {a.json} and {a.md}")
    for x in out["mismatches"]:
        print(f"MISMATCH: {x}")
    for x in (out.get("h_max_recheck") or {}).get("flags") or []:
        print(f"FLAG: {x}")
    return out["meta"]["exit_code"]


if __name__ == "__main__":
    sys.exit(main())
