#!/usr/bin/env python3
"""Stand-height campaign (``2026-10-13_stand_height``) -- the EVAL CONTRACT shared by the ``stand_height`` launcher
(``sh_eval_crab_hex_gait.py``), its rows script's tests and every reader (``sh_bridge_read.py`` now, the K1-K6 readers
later): run_meta names, raw-series keys, controller / command / restore vocabularies, checkpoint kinds, and the
campaign-local SCHEDULE MANIFEST (``scenarios_sh.yaml``) loader that maps every hold to its height command.

LAZY IMPORTS ONLY: the launcher loads this module before Kit starts. At import it reads the stdlib and ``sh_core``
(itself torch-free at import); PyYAML, numpy and torch load inside the functions that need them.

THE SCHEDULE MANIFEST (``scenarios_sh.yaml``; PLAN step 2 "launcher flag deriving per-hold heights from a
campaign-local schedule manifest (harness and scenario files untouched)"). It is a valid harness manifest (the harness
loader ignores unknown top-level keys) plus one campaign block:

    stand_height:
      version: 1
      rows:                     # rows run with --sh-command manifest
        <scenario id>: {gates: [...], hold_heights: [null, 0.719, null, ...]}   # ONE entry per hold, in order
      identity_rows:            # rows run with --sh-command zero, on the SHARED manifests (read-only)
        <scenario id>: <manifest path relative to crab_hex_omni_task/experiments>

The mapping is per hold and deterministic: the command of a step is its hold's ``hold_heights`` entry (the hold =
the harness's ``segment_id``, i.e. the hold's index), 1.064 (the walking height, the zero command) on a walk hold
(``null``), and the prologue carries the env's first hold's command (the harness's own prologue rule). Every stand hold
is LABELLED ``"stand"`` (the predecessor's ``p2s_stand`` helpers and the K-readers identify stands by that label, never
by a zero speed), so a label cannot carry a height: the map is positional, and run_meta records the same list
(``p2r_mode.stand_height.hold_heights``). :func:`height_row` validates a row before Kit (:func:`height_row_reasons`):
one entry per hold; every speed-0 hold is labelled "stand" and carries a height, every walk hold is not and does not;
every height lies in the frozen [h_min, h_max]; one facing for the whole row (so every speed-0 hold is a stand slot,
never a turn); the row starts and ends on a walk (walk-handover starts; every stand is handed back); and every hold is
long enough for its gate windows:

* a stand hold entered at level ``l`` (the previous stand's height, or 1.064 after a walk): ``hold_s >= |h - l| / 0.08
  + 1 s + 2 s`` (K1: the last 2 s of the hold, from command arrival + 1 s; K2 times arrival inside the hold);
* a walk hold after a stand at ``h``: ``hold_s >= |h - 1.064| / 0.08 + 1.5 s + 2 s`` (the restore ramp, K4's 1.5-s
  walking-onset slack, then 2 s of walking).
"""
from __future__ import annotations

import hashlib
import importlib.util
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
#: scripts -> 2026-10-13_stand_height
CAMPAIGN = HERE.parent
#: scripts -> 2026-10-13_stand_height -> experiments
EXPERIMENTS = HERE.parents[1]
TASK = HERE.parents[2]
CORE_PATH = HERE / "sh_core.py"
SCENARIOS_SH = CAMPAIGN / "scenarios_sh.yaml"


def load_by_path(path, name: str):
    """A private module instance loaded by path (not registered in ``sys.modules``; no ``__pycache__`` written)."""
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:  # pragma: no cover
        raise ImportError(f"sh_eval_common: cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    old = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.dont_write_bytecode = old
    return mod


#: the ONE core (posture map, R_h, limiter, restore, router, input); torch-free at import
CORE = load_by_path(CORE_PATH, "_sh_eval_common_core")


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ===================================================================================== run_meta contract
VARIANT = "stand_height"                 #: run_meta.p2r_mode.variant
BLOCK_KEY = "stand_height"               #: run_meta.p2r_mode.<BLOCK_KEY>: the campaign block
MASK_KEY = "stand_mask"                  #: run_meta.p2r_mode.<MASK_KEY>: an INT 0 / 1 (network controllers only)

#: the stand controllers (PLAN step-4 control table)
CONTROLLERS = ("stand_net", "amask_old", "r_h", "r_old", "posture_only")
#: controllers that run a stand network (P2L's st2 extra runner; --stand-ckpt / --stand-mask)
NET_CONTROLLERS = ("stand_net", "amask_old")
#: the OLD +-1 path: executed at h = 1.064 (no q, input column 15 = 0, R's own +-1); the limiter / restore still run
OLD_PATH_CONTROLLERS = ("amask_old", "r_old")
EXECUTORS = {
    "stand_net": "q18(h_lim) + clamp(a_net, +-1), rods +-4.8, cams +-1; input column 15 = (h_lim - 1.064) / 0.4",
    "amask_old": "clamp(a_net, +-1) (old +-1 path: q = 0, input column 15 = 0); A-mask s17 with W[:, 15] zeroed",
    "r_h": "R_h(obs, h_lim) (sh_core.rh_target, D3 constants, rods +-4.8)",
    "r_old": "R(obs) (sn_common.r_label, D3 constants, +-1; no q)",
    "posture_only": "q18(h_lim) (zero residual)",
}
#: --sh-command: ``zero`` = every hold at 1.064 (identity bridge, K5); ``manifest`` = per-hold heights from the
#: harness manifest's stand_height block
COMMANDS = ("zero", "manifest")
#: --sh-restore: ``armed`` = the H6 restore; ``off`` = hand back at height (the K4 known-bad), engage band infinite
RESTORES = ("armed", "off")
#: the inner P2L mode a controller runs on (st2: S1 + the stand network as P2L's extra runner; hold: S1 only)
INNER_MODE = {c: ("st2" if c in NET_CONTROLLERS else "hold") for c in CONTROLLERS}

#: ``stand_height.stand_ckpt_kind``: a stand-height trainer checkpoint (``infos.<CKPT_INFOS_KEY>.stand_mask`` an int
#: equal to --stand-mask), the A-mask s17 file (sha pinned; W[:, 15] zeroed in memory with sh_core's warm start), or S1
#: itself (the identity bridge: stand_net, zero command)
CKPT_KIND_TRAINED, CKPT_KIND_AMASK, CKPT_KIND_S1 = "sh_dagger", "amask_s17_w15zeroed", "s1_identity"
CKPT_KINDS = (CKPT_KIND_TRAINED, CKPT_KIND_AMASK, CKPT_KIND_S1)
CKPT_INFOS_KEY = "sh_dagger"
#: S1's training clip: the wrapper's clip_actions before the stand_height launcher raises it (refused otherwise)
S1_TRAIN_CLIP = 1.0


def ckpt_kind(stand_sha: str | None, s1_sha: str | None) -> str | None:
    """The kind fixed by the file hash alone: the A-mask s17 file, S1 itself, or None (a trainer checkpoint to check
    by its ``infos``)."""
    if stand_sha and stand_sha == CORE.STAND_NET_SHA256:
        return CKPT_KIND_AMASK
    if stand_sha and s1_sha and stand_sha == s1_sha:
        return CKPT_KIND_S1
    return None


def ckpt_kind_reasons(kind: str | None, *, controller: str, command: str, stand_mask: int | None) -> list[str]:
    """The pairing rules of a checkpoint kind (``None`` = a trainer checkpoint, its infos checked post-Kit)."""
    out = []
    if controller not in NET_CONTROLLERS:
        return out
    if kind == CKPT_KIND_AMASK:
        if stand_mask != 1:
            out.append(f"the A-mask s17 checkpoint is the A-mask arm: --stand-mask must be 1, got {stand_mask!r}")
    elif kind == CKPT_KIND_S1:
        if controller != "stand_net" or command != "zero":
            out.append(f"S1 as the stand checkpoint is the identity bridge only (--sh-controller stand_net "
                       f"--sh-command zero), got {controller} / {command}")
    if controller == "amask_old" and kind != CKPT_KIND_AMASK:
        out.append("--sh-controller amask_old evaluates the A-mask s17 checkpoint (policy/stand_net/model_4000.pt) only")
    return out


# ===================================================================================== raw series (per episode)
RAW_STAND_SLOT = "sh_stand_slot"         # (T,) bool: the command term's latched stand slot
RAW_RESTORING = "sh_restoring"           # (T,) bool: an H6 restore row (the stand controller acts, h -> 1.064)
RAW_ENGAGED = "sh_engaged"               # (T,) bool: a restore started this step
RAW_COMPLETED = "sh_completed"           # (T,) bool: a restore completed this step (row handed to S1)
RAW_TIMED_OUT = "sh_timed_out"           # (T,) bool: a restore timed out this step (row handed to S1)
RAW_MERGED = "sh_merged"                 # (T,) bool: a stand arrived mid-restore (stand -> stand)
RAW_RESET_CUT = "sh_reset_cut"           # (T,) bool: an episode reset cut a running restore
RAW_LEAVE_OFF = "sh_leave_off_height"    # (T,) bool: the slot left stand with the limited command > 2 cm off 1.064
RAW_H_CMD = "sh_h_cmd"                   # (T,) float64: the scheduled raw height command (1.064 on walk holds)
RAW_H_LIM = "sh_h_lim"                   # (T,) float64: the limited command this step used
RAW_H_MEAS = "sh_h_meas"                 # (T,) float64: root z - mean terrain z under the six feet (harness buffers)
RAW_Q = "sh_q18"                         # (T, 18) float32: the posture actually added (q18), stand rows, 0 elsewhere
RAW_A_NET = "sh_a_net"                   # (T, 18) float32: the stand network's raw output, stand rows, 0 elsewhere
RAW_STAND_ACTION = "sh_stand_action"     # (T, 18) float32: the EXECUTED stand action, stand rows, 0 elsewhere
RAW_S1_ACTION = "sh_s1_action"           # (T, 18) float32: clamp(S1, +-1) (counterfactual), stand rows, 0 elsewhere
RAW_RH_TARGET = "sh_rh_target"           # (T, 18) float32: R_h at the limited command, stand rows, 0 elsewhere
RAW_RH_BRANCH = "sh_rh_branch"           # (T, 6) int8: R_h's per-leg branch, stand rows, -1 elsewhere
RAW_BOOL_KEYS = (RAW_STAND_SLOT, RAW_RESTORING, RAW_ENGAGED, RAW_COMPLETED, RAW_TIMED_OUT, RAW_MERGED, RAW_RESET_CUT,
                 RAW_LEAVE_OFF)
RAW_H_KEYS = (RAW_H_CMD, RAW_H_LIM)
RAW_ACT_KEYS = (RAW_Q, RAW_A_NET, RAW_STAND_ACTION, RAW_S1_ACTION, RAW_RH_TARGET)
#: the series the mode records (P2L's _close_pass writes every key of c["rst"]); RAW_H_MEAS is added at write time
MODE_RAW_KEYS = RAW_BOOL_KEYS + RAW_H_KEYS + RAW_ACT_KEYS + (RAW_RH_BRANCH,)
RAW_KEYS = MODE_RAW_KEYS + (RAW_H_MEAS,)
#: P2L's per-step stand series: here the EFFECTIVE stand vector (slot | restoring) -- the rows the stand controller acts
#: on (P2L's "stand mode"); the latched slot alone is RAW_STAND_SLOT
RAW_STAND_MODE = "p2r_stand_mode"
#: what the harness's own ``actions`` key holds (its pre-clip convention, kept so a zero-command bridge is bitwise REFS)
RECORDED_ACTIONS = ("the harness's raw 'actions' keep its pre-clip convention: S1 rows = S1's raw output; stand rows = "
                    "the stand controller's value before the router's clamps (network: where(q != 0, a_net + q, a_net); "
                    "scripted: its executed target). The EXECUTED action is sh_core.route(stand_eff, sh_stand_action, "
                    "S1) = where(p2r_stand_mode, sh_stand_action, clamp(actions, +-1)), swapped in at env.step.")


def h_meas_series(raw: dict, run_meta: dict | None):
    """``(h_meas (T, E) float64, record)``: :func:`sh_core.h_meas` on the harness's OWN buffers ``root_pos_w`` and
    ``terrain_z`` (its ``_terrain_z_under_feet`` raycast, both recorded in the same loop iteration as the policy call,
    i.e. at the instant the policy acted). NaN with ``valid`` False when the harness used its ``root_z - 1.05`` fallback
    (``run_meta.terrain_source`` not ``raycast_mesh``) or a buffer is missing -- h_meas would then be a constant."""
    import numpy as np  # noqa: PLC0415
    import torch  # noqa: PLC0415

    acts = np.asarray(raw["actions"])
    T, E = int(acts.shape[0]), int(acts.shape[1])
    src = (run_meta or {}).get("terrain_source")
    rec = {"definition": "sh_core.h_meas(root_pos_w[..., 2], terrain_z) on the harness's raw buffers",
           "terrain_source": src, "valid": False, "reason": None}
    rz, tz = raw.get("root_pos_w"), raw.get("terrain_z")
    if rz is None or tz is None:
        rec["reason"] = "the harness raws carry no root_pos_w / terrain_z"
        return np.full((T, E), np.nan), rec
    rz = np.asarray(rz)
    tz = np.asarray(tz)
    hm = CORE.h_meas(torch.from_numpy(np.ascontiguousarray(rz[..., 2]).reshape(-1)),
                     torch.from_numpy(np.ascontiguousarray(tz).reshape(-1, 6))).numpy().reshape(T, E)
    if src != "raycast_mesh":
        rec["reason"] = f"terrain_source {src!r} (not the raycast mesh): h_meas would be constant"
        return np.full((T, E), np.nan), rec
    rec["valid"] = True
    return hm, rec


# ===================================================================================== the schedule manifest
MANIFEST_BLOCK = "stand_height"
MANIFEST_BLOCK_VERSION = 1
ROW_KEYS = ("hold_heights", "gates", "notes")
#: the label every stand hold carries (p2s_stand / the K-readers key stands on it)
STAND_LABEL = "stand"
#: gate-window rules (module docstring)
ARRIVAL_SETTLE_S = 1.0          # K1 window opens at command arrival + 1 s
K1_WINDOW_S = 2.0               # K1: the last 2 s of the hold
K4_ONSET_SLACK_S = 1.5          # K4: walking onset <= REFS + dh / 0.08 + 1.5 s
WALK_AFTER_RESTORE_S = 2.0      # walking observed after the onset bound
_TOL = 1e-9
#: the identity-bridge rows (PLAN step 4: bitwise = REFS on suite / walkstop / field_hard es1)
IDENTITY_ROW_IDS = ("suite__omni", "walkstop__omni", "field_hard__omni")
#: the zero-command rows K5 needs BEYOND the bridge's (sn_calib G4 reads field_turn es1), on their SHARED manifests
#: (path relative to crab_hex_omni_task/experiments; kept here, not in scenarios_sh.yaml, whose identity_rows block lists
#: the bridge rows)
K5_EXTRA_ZERO_ROWS = {"field_turn__omni": "2026-10-07_omni_phase2/scenarios_p2.yaml"}
#: every zero-command row on a SHARED manifest: the bridge's rows + the K5-only ones
ZERO_COMMAND_ROW_IDS = IDENTITY_ROW_IDS + tuple(K5_EXTRA_ZERO_ROWS)
#: K5 (sn_calib G1-G5, unchanged reads) needs a head's zero-command stamps on EXACTLY these row@es (queue them in one
#: pass: ``sh_eval_rows.sh NAME stand_net zero armed CKPT 1 <K5_ROWS...>``)
K5_ROWS = ("suite__omni@1", "suite__omni@2", "walkstop__omni@1", "walkstop__omni@2", "field_turn__omni@1",
           "field_hard__omni@1", "field_hard__omni@2", "field_hard__omni@3", "field_hard__omni@4")


def read_yaml(path) -> dict:
    import yaml  # noqa: PLC0415

    data = yaml.safe_load(Path(path).read_text())
    if not isinstance(data, dict):
        raise ValueError(f"{path}: top level must be a mapping")
    return data


def _block(data: dict, path) -> dict:
    blk = data.get(MANIFEST_BLOCK)
    if not isinstance(blk, dict):
        raise ValueError(f"{path}: no {MANIFEST_BLOCK!r} block (the schedule manifest's height map)")
    if blk.get("version") != MANIFEST_BLOCK_VERSION:
        raise ValueError(f"{path}: {MANIFEST_BLOCK}.version {blk.get('version')!r} != {MANIFEST_BLOCK_VERSION}")
    return blk


def has_height_row(manifest, scenario: str | None) -> bool:
    """True when ``manifest`` names ``scenario`` under ``stand_height.rows`` (such a row runs with its heights)."""
    if not manifest or not scenario or not Path(manifest).is_file():
        return False
    blk = read_yaml(manifest).get(MANIFEST_BLOCK)
    rows = blk.get("rows") if isinstance(blk, dict) else None
    return isinstance(rows, dict) and scenario in rows


def _num(v) -> float | None:
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(float(v)):
        return None
    return float(v)


def height_row_reasons(holds: list[dict], hold_heights: list, *, h_min: float | None = None,
                       h_max: float | None = None) -> list[str]:
    """Every rule of the module docstring on one row. ``holds``: ``[{label, speed, face, hold_s}]`` in schedule
    order; ``hold_heights``: one entry per hold (a float on stand holds, None on walk holds). Empty = valid."""
    h_min = CORE.H_MIN if h_min is None else float(h_min)
    h_max = CORE.H_MAX if h_max is None else float(h_max)
    rate = CORE.HEIGHT_RATE_M_S
    if not holds:
        return ["empty schedule"]
    if len(hold_heights) != len(holds):
        return [f"hold_heights has {len(hold_heights)} entries for {len(holds)} holds (one per hold, in order)"]
    out = []
    if holds[0]["speed"] <= 0:
        out.append(f"the first hold {holds[0]['label']!r} must be a walk (a walk-handover start; the prologue holds it)")
    if holds[-1]["speed"] <= 0:
        out.append(f"the last hold {holds[-1]['label']!r} must be a walk (every stand is handed back inside the row)")
    face0 = holds[0]["face"]
    level, prev_stand = CORE.H_SETTLED, False
    for i, (h, hh) in enumerate(zip(holds, hold_heights)):
        lab, sp, hs = h["label"], h["speed"], h["hold_s"]
        if h["face"] != face0:
            out.append(f"hold {i} {lab!r}: face {h['face']} != {face0} (one facing per row: a speed-0 hold must be a "
                       "stand slot, never a turn)")
        if sp < 0:
            out.append(f"hold {i} {lab!r}: speed {sp} < 0")
            continue
        if sp == 0:
            if lab != STAND_LABEL:
                out.append(f"hold {i} {lab!r}: a stand hold (speed 0) must be labelled {STAND_LABEL!r} (p2s_stand and "
                           "the K-readers identify stands by that label)")
            if hh is None:
                out.append(f"hold {i} {lab!r}: a stand hold (speed 0) without a height")
                continue
            if not h_min - _TOL <= hh <= h_max + _TOL:
                out.append(f"hold {i} {lab!r}: height {hh} m is outside the frozen range [{h_min}, {h_max}]")
            need = abs(hh - level) / rate + ARRIVAL_SETTLE_S + K1_WINDOW_S
            if hs + _TOL < need:
                out.append(f"hold {i} {lab!r}: hold_s {hs} < |{hh} - {level:.3f}| / {rate} + {ARRIVAL_SETTLE_S} + "
                           f"{K1_WINDOW_S} = {need:.2f} s (ramp, arrival + 1 s, the 2-s window)")
            level, prev_stand = hh, True
        else:
            if lab == STAND_LABEL:
                out.append(f"hold {i}: a walk hold (speed {sp}) labelled {STAND_LABEL!r}")
            if hh is not None:
                out.append(f"hold {i} {lab!r}: a walk hold (speed {sp}) carries a height {hh}")
            if prev_stand:
                need = abs(level - CORE.H_SETTLED) / rate + K4_ONSET_SLACK_S + WALK_AFTER_RESTORE_S
                if hs + _TOL < need:
                    out.append(f"hold {i} {lab!r}: hold_s {hs} < |{level} - 1.064| / {rate} + {K4_ONSET_SLACK_S} + "
                               f"{WALK_AFTER_RESTORE_S} = {need:.2f} s (restore ramp, K4 onset slack, 2 s of walking)")
            level, prev_stand = CORE.H_SETTLED, False
    return out


def height_row(manifest, scenario: str) -> dict:
    """The validated height row ``scenario`` of ``manifest`` (raises ValueError with every reason):
    ``{manifest, manifest_sha256, scenario, labels, hold_heights, gates}`` (``hold_heights``: one entry per hold, a
    height on stand holds, None on walk holds; run_meta records this dict as ``p2r_mode.stand_height.height_row`` and
    the list as ``p2r_mode.stand_height.hold_heights``)."""
    path = Path(manifest)
    data = read_yaml(path)
    blk = _block(data, path)
    rows = blk.get("rows")
    if not isinstance(rows, dict) or scenario not in rows:
        raise ValueError(f"{path}: no height row {scenario!r} under {MANIFEST_BLOCK}.rows "
                         f"({sorted(rows) if isinstance(rows, dict) else rows!r})")
    row = rows[scenario]
    if not isinstance(row, dict):
        raise ValueError(f"{path}: {MANIFEST_BLOCK}.rows.{scenario} must be a mapping")
    extra = sorted(set(row) - set(ROW_KEYS))
    if extra:
        raise ValueError(f"{path}: {MANIFEST_BLOCK}.rows.{scenario} carries unknown keys {extra} (only {ROW_KEYS})")
    hraw = row.get("hold_heights")
    if not isinstance(hraw, list) or not hraw:
        raise ValueError(f"{path}: {MANIFEST_BLOCK}.rows.{scenario}.hold_heights must be a non-empty list (one entry "
                         "per hold)")
    hold_heights = []
    for i, v in enumerate(hraw):
        f = None if v is None else _num(v)
        if v is not None and f is None:
            raise ValueError(f"{path}: hold_heights[{i}] = {v!r} is neither null nor a finite number")
        hold_heights.append(f)
    scen = [s for s in (data.get("scenarios") or []) if isinstance(s, dict) and s.get("id") == scenario]
    if len(scen) != 1:
        raise ValueError(f"{path}: {len(scen)} scenarios with id {scenario!r} (want exactly 1)")
    s = scen[0]
    if str(s.get("kind", "schedule")) != "schedule":
        raise ValueError(f"{path}: {scenario} is a {s.get('kind')!r} row; a height row is a plain schedule row")
    if s.get("rotate_holds_by_env"):
        raise ValueError(f"{path}: {scenario} rotates holds by env; a height row runs one order (walk-handover starts)")
    holds = []
    for i, hd in enumerate(s.get("schedule") or []):
        if not isinstance(hd, dict) or any(_num(hd.get(k)) is None for k in ("speed", "dir", "face", "hold_s")):
            raise ValueError(f"{path}: {scenario} hold {i} must be a world-frame hold with speed / dir / face / hold_s")
        holds.append({"label": str(hd.get("label") or f"hold{i}"), "speed": float(hd["speed"]),
                      "face": float(hd["face"]), "hold_s": float(hd["hold_s"])})
    reasons = height_row_reasons(holds, hold_heights)
    if reasons:
        raise ValueError(f"{path}: height row {scenario} refused:\n  " + "\n  ".join(reasons))
    return {"manifest": str(path.resolve()), "manifest_sha256": sha256_file(path), "scenario": scenario,
            "labels": [h["label"] for h in holds], "hold_heights": list(hold_heights),
            "gates": list(row.get("gates") or [])}


def zero_hold_heights(hold_labels) -> list:
    """The ``hold_heights`` record of a zero-command run: 1.064 on every ``"stand"``-labelled hold, None elsewhere (the
    convention of a height row's list: a number on stand holds, null on the others)."""
    return [CORE.H_SETTLED if lab == STAND_LABEL else None for lab in hold_labels]


def identity_rows(manifest=SCENARIOS_SH) -> dict:
    """``{scenario id: absolute shared-manifest path}`` of the ``stand_height.identity_rows`` block (zero-command rows
    on the SHARED manifests: the identity bridge's rows)."""
    path = Path(manifest)
    blk = _block(read_yaml(path), path)
    rows = blk.get("identity_rows")
    if not isinstance(rows, dict) or not rows:
        raise ValueError(f"{path}: {MANIFEST_BLOCK}.identity_rows must be a non-empty mapping")
    return {str(k): (EXPERIMENTS / str(v)).resolve() for k, v in rows.items()}


def zero_command_rows(manifest=SCENARIOS_SH) -> dict:
    """Every zero-command row (the bridge's :func:`identity_rows` + :data:`K5_EXTRA_ZERO_ROWS`, which K5 needs) ->
    absolute shared-manifest path, in :data:`ZERO_COMMAND_ROW_IDS` order. Raises if the two sources overlap."""
    rows = identity_rows(manifest)
    clash = sorted(set(rows) & set(K5_EXTRA_ZERO_ROWS))
    if clash:
        raise ValueError(f"zero_command_rows: {clash} both in {MANIFEST_BLOCK}.identity_rows and K5_EXTRA_ZERO_ROWS")
    rows.update({k: (EXPERIMENTS / v).resolve() for k, v in K5_EXTRA_ZERO_ROWS.items()})
    return rows


def height_table(segment_id, hold_heights):
    """``(T, E)`` float64 per-step raw height command: ``hold_heights[segment]`` (the hold's index), 1.064 on a None
    entry (walk holds); prologue steps (segment -1) carry the env's FIRST hold's command (the harness's prologue holds
    that hold's command). Pure numpy, deterministic."""
    import numpy as np  # noqa: PLC0415

    seg = np.asarray(segment_id).astype(np.int64)
    if seg.ndim != 2:
        raise ValueError(f"height_table: segment_id must be (T, E), got {seg.shape}")
    H = np.array([CORE.H_SETTLED if h is None else float(h) for h in hold_heights] or [CORE.H_SETTLED],
                 dtype=np.float64)
    pos = seg >= 0
    first_t = np.argmax(pos, axis=0)
    first = np.where(pos.any(axis=0), seg[first_t, np.arange(seg.shape[1])], -1)
    segf = np.where(pos, seg, first[None, :])
    if (segf >= len(hold_heights)).any():
        raise ValueError("height_table: a segment id past the hold list")
    return np.where(segf >= 0, H[np.clip(segf, 0, None)], CORE.H_SETTLED)
