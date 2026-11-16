#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""Fault-injection sweep reader (CHARTER §5, §6.2): classes, launch checks, post-onset statistics and verdicts.

CPU only: numpy on the raw per-episode npz + the ``fault_onsets/v1`` sidecar; torch is used only to CALL
``crab_hex_forward_task/mdp/fault_inject.onset_table`` (imported by path, never re-implemented).

Layout read (``gate_row.sh``): ``<root>/<NAME>/<SCENARIO>/seed%03d/<stamp>/`` with ``raw/episode_NN.npz``,
``metrics/``, ``scenario_metrics.json``, ``run_meta.json``, ``tp_row.json`` (the completion marker) and, on an armed
cell, ``fault_onsets.json``. A launch's cell and row come from its ``tp_row.json`` env file
(``envs/<cell>__<row>[_L090].env``); its policy from the checkpoint sha (6b48ef33 = C1S, 6d1f29be = c1); its payload
from ``run_meta.payload`` (None = L000). A SLOT is (policy, row, env seed, payload).

    fault_read.py [--root R] [--out-dir D/results]                    full read -> read.json + verdicts.md
    fault_read.py [--root R] --check-only STAMP [STAMP ...]           launch checks only; rc 0 ok, 2 launch error,
                                                                      3 class unmatched (relaunch candidate), 4 usage
    fault_read.py --null-fault A B --sidecar S [--null-fault A B --sidecar S ...] [--cell CELL]
                                                                      read nominal A as if faulted against nominal B
                                                                      on sidecar S's onsets (CHARTER §6.1.5); S may be
                                                                      ``synth:<cell>`` (onset_table + the cell's d)

Definitions (CHARTER §5.2, the harness's own where it has one -- ``gait_eval/report.py`` / ``metrics.py``):

* step t of an npz = the state the policy step t's observation was built from (recorded BEFORE ``env.step``), the
  policy's raw action of step t, and ``rod_speed`` / ``done`` AFTER step t. A fault active on [onset, onset + d)
  changes the applied action / plant of step onset, so every npz row t < onset is pre-onset.
* fall = termination_reason != schedule_complete; the fall step = n_steps - 1. Alive at onset = no fall before onset.
* F (per suite wave, per field_light / kind row) = post-onset falls / episodes alive at onset; F2 = post-onset falls
  with fall step < onset + 100; hazard = post-onset falls per post-onset minute (exposure = n_steps - onset steps).
* post-onset lines: own-hold windows exactly as ``report._window_mask`` (segment == own hold, steady for tracking /
  whole hold for facing, fall-trimmed 25 steps on a fall) intersected with t >= onset; survival = own-hold survival
  (n_steps >= the hold's end, unchanged); ratio = mean travel-frame vx / mean |cmd xy| (None at < 50 steps or a
  command under lin_vel_clip 0.2); facing_closed re-anchored at onset = clip(1 - |e_last| / max(|e_onset|, 0.15),
  -1, 1), |e_onset| < 0.15 excluded and counted; h_meas = per-episode median of root z - mean terrain z under the
  feet over segment > 0, t >= onset, pooled median; roll / pitch RMS in the travel frame over [onset, n_steps),
  pooled median; field_light completion = schedule_complete share (unchanged).
* R = share of episodes alive at the window start (window inside one translate hold's planned extent) whose
  2-s window ratio is within 0.25 of 1 (a fall inside the window counts as not tracking); window [clear + 2 s,
  clear + 4 s) for a transient (clear = planned_last + 1; depth cells: the sidecar's policy-visible clear,
  policy_visible_last + 1), [onset + 2 s, onset + 4 s) for a permanent fault; the < 25 floor counted on the reference.
"""
from __future__ import annotations

import argparse
import collections
import glob
import hashlib
import importlib.util
import json
import math
import re
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True

HERE = Path(__file__).resolve().parent
D = HERE.parent
EXPERIMENTS = D.parent
TASKS = D.parents[2]                                   # parkour/parkour_tasks/parkour_tasks
FAULT_INJECT_PY = TASKS / "crab_hex_forward_task" / "mdp" / "fault_inject.py"
THRESHOLDS_JSON = EXPERIMENTS / "2026-10-16_task4_pipeline" / "thresholds_v1_2.json"
CELLS_YAML = D / "cells.yaml"
DEFAULT_ROOT = Path("/home/nickmagus/krabby/krabby-research/parkour/outputs/fault_sweep/lr")
DEFAULT_OUT = D / "results"
SCHEMA = "fault_read/v1"

DT = 0.02
SCENARIO_ROW = {"suite__omni": "suite", "field_light__omni": "field_light", "rt_ledge_down_b2__omni": "ledge_down",
                "rt_pits_b2__omni": "pits",
                # smoke-only short rows (manifests/scenarios_smoke.yaml); launch checks only, never verdicts
                "smk_suite__omni": "smk_suite", "smk_field_light__omni": "smk_field_light"}
KIND_ROWS = ("ledge_down", "pits")
POLICY_OF_SHA = {"6b48ef33": "C1S", "6d1f29be": "c1"}
NOMINAL_CELLS = ("NOM",)
COMPLETE = "schedule_complete"
FALL_REASONS = ("fall", "hard_fall")                   # report._window_mask's fall trim
# harness defaults (gait_eval/schedule.py DEFAULTS, scenarios_omni.yaml lin_vel_clip 0.2)
LIN_VEL_CLIP = 0.2
MIN_WINDOW_STEPS = 50                                  # min_window_s 1.0
FALL_TRIM_STEPS = 25                                   # fall_exclusion_s 0.5
FC_FLOOR = 0.15                                        # metrics.FACING_CLOSED_MIN_INITIAL_RAD
F2_STEPS = 100
MAX_PHYSICS_DRIFT = 3                                  # CHARTER 5.1 (2026-11-16 amendment): > 3 drift envs / row = launch error
MIN_PAIRED_ENVS = 5                                    # CHARTER 5.1 (2026-11-16 01:40): < 5 non-excluded envs = dose pairing report-only
TRANSLATE_LABELS = ("fwd", "lat_l", "lat_r", "back", "diag")
R_LAG_STEPS, R_LEN_STEPS = 100, 100                    # [clear + 2 s, clear + 4 s)
R_TOL, R_MIN, R_FLOOR_N = 0.25, 0.90, 25
DF_FALLS, DF_TOL, DF_REC = 0.10, 0.03, 0.05
MARGINAL_EPISODES = 2
CROSS_R_M = 3.2                                        # rt_spec.CROSS_R ledges / pits
ROD_CAP_M_S = {"hip": 0.028, "knee": 0.033}            # crab_hex_dimensions speed_loaded_m_s (rod_speed's unit)
ZERO_TOL = 1e-6                                        # self-check probes (6 significant digits in the sidecar)
SEVERITY = {"TOLERATED": 0, "RECOVERS": 1, "DEGRADES": 2, "FALLS": 3}
SUITE_LINE_CATS = ("fwd", "side", "back", "diag", "turn", "move_turn")


# ====================================================================================== imports by path
def _load_by_path(path: Path, name: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def fault_inject():
    """The injector module (torch; its lazy linkage import is never reached here)."""
    return _load_by_path(FAULT_INJECT_PY, "fault_inject_for_reader")


def onset_table_np(lo_s: float, hi_s: float, num_envs: int, env_seed, waves: int, dt: float = DT) -> np.ndarray:
    """``fault_inject.onset_table`` as int64 numpy ``(waves, N)``."""
    return fault_inject().onset_table(lo_s, hi_s, dt, num_envs, env_seed, waves=max(int(waves), 1)).numpy()


def load_cells(path: Path = CELLS_YAML) -> dict:
    import yaml  # noqa: PLC0415

    doc = yaml.safe_load(Path(path).read_text())
    cells = {c["id"]: dict(c) for c in doc["cells"]}
    for s in doc.get("smokes") or []:
        cells[s["id"]] = dict(s, family="smoke", smoke=True)
    cells["NOM"] = {"id": "NOM", "fault": None, "family": "nominal"}
    rows = dict(doc["rows"])
    # Smoke rows: the parent row's onset window and class horizon (smk_field_light's depth smokes use 10.0:14.0,
    # same lower bound, so the same horizon; fault_queue.py make-smoke-envs).
    rows.setdefault("smk_suite", dict(rows["suite"], scenario="smk_suite__omni", smoke=True))
    rows.setdefault("smk_field_light", dict(rows["field_light"], scenario="smk_field_light__omni", onset="10.0:14.0",
                                            smoke=True))
    return {"rows": rows, "cells": cells, "doc": doc}


def cell_fault_id(cell: dict) -> str | None:
    f = cell.get("fault")
    return None if not f else str(f).split(":")[0]


def cell_duration_steps(cell: dict, dt: float = DT) -> int | None:
    """A transient cell's d in steps (None = permanent / anchor / nominal)."""
    if cell.get("transient_steps") is not None:
        return int(cell["transient_steps"])
    if cell.get("transient_s") is not None:
        return int(round(float(cell["transient_s"]) / dt))
    return None


def is_anchor(cell: dict) -> bool:
    return cell.get("family") == "anchor"


def is_armed(cell: dict) -> bool:
    return bool(cell.get("fault"))


# ====================================================================================== thresholds
def load_lines(path: Path = THRESHOLDS_JSON) -> dict:
    """{(stat, scope, es): (line, direction)} of the p3a.stable90.* lines (payload_read.stable_lines' ids)."""
    doc = json.loads(Path(path).read_text())
    out = {}
    for x in doc["lines"]:
        lid = x["id"]
        if not lid.startswith("p3a.stable90."):
            continue
        if lid == "p3a.stable90.field_light_completion":
            out[("field_light_completion", None, None)] = (float(x["line"]), x["direction"])
            continue
        m = re.match(r"^p3a\.stable90\.turn_fc\.es(\d+)$", lid)
        if m:
            out[("turn_fc", "turn", int(m.group(1)))] = (float(x["line"]), x["direction"])
            continue
        m = re.match(r"^p3a\.stable90\.(?P<stat>[a-z_]+)\.(?P<scope>[a-z_]+)\.es(?P<es>\d+)$", lid)
        if m is None:
            raise ValueError(f"unknown stable90 line {lid!r}")
        out[(m.group("stat"), m.group("scope"), int(m.group("es")))] = (float(x["line"]), x["direction"])
    return out


def line_for(lines: dict, stat: str, scope, es: int):
    for key in ((stat, scope, es), (stat, scope, 1), (stat, scope, None)):
        if key in lines:
            return lines[key]
    return None


def passes(v, line, direction) -> bool:
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return False
    return v >= line - 1e-12 if direction == ">=" else v <= line + 1e-12


# ====================================================================================== launches
def _json(p: Path, default=None):
    p = Path(p)
    if not p.exists():
        return default
    return json.loads(p.read_text())


_ENV_RE = re.compile(r"^(?P<cell>[A-Za-z0-9_]+?)__(?P<row>suite|field_light|ledge_down|pits|smk_field_light)(?P<lvl>_L\d{3})?\.env$")


def payload_label(run_meta: dict) -> str:
    p = run_meta.get("payload")
    if not p:
        return "L000"
    kg = p.get("kg_range") or [max(p.get("payload_kg") or [0.0])]
    return f"L{int(round(float(max(kg)))):03d}"


class Launch:
    """One stamp dir (lazy JSON reads)."""

    def __init__(self, stamp: Path, cell: str | None = None):
        self.stamp = Path(stamp)
        self.tp_row = _json(self.stamp / "tp_row.json", {}) or {}
        self.run_meta = _json(self.stamp / "run_meta.json", {}) or {}
        self.scenario = self.run_meta.get("scenario_id") or self.tp_row.get("scenario") or self.stamp.parents[1].name
        self.row = SCENARIO_ROW.get(self.scenario, self.scenario)
        seed = self.tp_row.get("env_seed")
        if seed is None:
            m = re.match(r"seed(\d+)$", self.stamp.parent.name)
            seed = int(m.group(1)) if m else None
        self.seed = int(seed) if seed is not None else None
        self.payload = payload_label(self.run_meta)
        sha = str(self.run_meta.get("checkpoint_sha256") or "")
        self.policy = POLICY_OF_SHA.get(sha[:8], sha[:8] or "?")
        self.name = self.stamp.parents[2].name if len(self.stamp.parents) > 2 else "?"
        env_file = Path(str(self.tp_row.get("env_file") or "")).name
        m = _ENV_RE.match(env_file)
        self.env_file = env_file
        self.cell = cell or (m.group("cell") if m else None)
        self.sidecar_path = self.stamp / "fault_onsets.json"
        self._sidecar = None
        self._episodes = None

    @property
    def slot(self) -> tuple:
        return (self.policy, self.row, self.seed, self.payload)

    @property
    def complete(self) -> bool:
        return (self.stamp / "tp_row.json").exists() and (self.stamp / "scenario_metrics.json").exists()

    @property
    def sidecar(self) -> dict | None:
        if self._sidecar is None and self.sidecar_path.exists():
            self._sidecar = json.loads(self.sidecar_path.read_text())
        return self._sidecar

    @property
    def num_envs(self) -> int:
        return int(self.run_meta.get("num_envs") or 0)

    @property
    def waves(self) -> int:
        return int(self.run_meta.get("waves") or 1) if self.run_meta.get("suite") else 1

    def episodes(self) -> list:
        """Per episode: idx, wave, env, label, category, hand, n_steps, reason (scenario_metrics + run_meta)."""
        if self._episodes is None:
            sm = _json(self.stamp / "scenario_metrics.json")
            by = {int(e["env_index"]): e for e in sm["episodes"]}
            n = self.num_envs
            out = []
            if self.run_meta.get("suite"):
                for m in self.run_meta["env_index_map"]:
                    i = int(m["env_index"])
                    out.append({"idx": i, "wave": int(m["wave"]), "env": int(m["env"]), "label": m["label"],
                                "cat": m["category"], "hand": m.get("hand"), "n": int(by[i]["n_steps"]),
                                "reason": by[i]["termination_reason"]})
            else:
                for i in sorted(by):
                    out.append({"idx": i, "wave": 0, "env": i % max(n, 1), "label": None, "cat": None, "hand": None,
                                "n": int(by[i]["n_steps"]), "reason": by[i]["termination_reason"]})
            self._episodes = out
        return self._episodes

    def npz(self, idx: int) -> Path:
        return self.stamp / "raw" / f"episode_{int(idx):02d}.npz"

    def suite_hold_extent(self) -> tuple[int, int] | None:
        """The own hold's planned [start, end) steps on a suite row (prologue + settle, + hold_s)."""
        sm = _json(self.stamp / "scenario_metrics.json")
        s = (sm.get("aggregate") or {}).get("suite")
        if not s:
            return None
        dt = float(s["dt"])
        start = int(round((float(s["prologue_s"]) + float(s["settle_hold_s"])) / dt))
        hs = {float(v) for v in s["hold_s"].values()}
        if len(hs) != 1:
            raise ValueError(f"{self.stamp}: suite holds of unequal length {sorted(hs)}")
        return start, start + int(round(hs.pop() / dt))

    def __repr__(self):
        return f"Launch({self.cell}@{self.row}/es{self.seed}/{self.payload}/{self.policy}:{self.stamp.name})"


def scan(root: Path) -> tuple[list, list]:
    """(complete launches, incomplete stamp dirs) under ``root``."""
    ok, bad = [], []
    for st in sorted(glob.glob(str(Path(root) / "*" / "*" / "seed*" / "*"))):
        p = Path(st)
        if not p.is_dir():
            continue
        if (p / "tp_row.json").exists():
            ok.append(Launch(p))
        else:
            bad.append(str(p))
    return ok, bad


# ====================================================================================== onset tables
def sidecar_onsets(sc: dict) -> np.ndarray:
    return np.asarray([w["onset"] for w in sorted(sc["waves"], key=lambda w: w["wave"])], dtype=np.int64)


def sidecar_clear(sc: dict, depth: bool) -> np.ndarray | None:
    """(waves, N) clear step of a transient (first step the policy sees no fault), None for a permanent fault."""
    if sc.get("duration_steps") is None or (sc.get("fault") or {}).get("from_start"):
        return None
    rows = []
    for w in sorted(sc["waves"], key=lambda w: w["wave"]):
        planned = np.asarray(w["planned_last"], dtype=np.int64) + 1
        if depth and w.get("policy_visible_last") is not None:
            vis = np.asarray(w["policy_visible_last"], dtype=np.int64)
            bound = np.asarray(w.get("policy_visible_bound") or (planned - 1), dtype=np.int64)
            planned = np.where(vis >= 0, vis + 1, bound + 1)
        rows.append(planned)
    return np.asarray(rows, dtype=np.int64)


def synth_sidecar(cells: dict, cell_id: str, row: str, num_envs: int, env_seed: int, waves: int) -> dict:
    """A minimal fault_onsets/v1 stand-in for a null-fault read: the row's onset_table and the cell's d."""
    cell = cells["cells"][cell_id]
    lo, hi = (float(x) for x in str(cells["rows"][row]["onset"]).split(":"))
    tab = onset_table_np(lo, hi, num_envs, env_seed, waves)
    d = cell_duration_steps(cell)
    fid = cell_fault_id(cell)
    recs = []
    for w in range(waves):
        on = tab[w]
        recs.append({"wave": w, "onset": on.tolist(), "start": on.tolist(),
                     "planned_last": None if d is None else (on + d - 1).tolist(),
                     "first_active": on.tolist(), "last_active": (on + (d or 0) - 1).tolist() if d else on.tolist(),
                     "fell_at": [-1] * num_envs, "probes": {}, "state_cleared": True})
    return {"schema": "fault_onsets/v1", "synthetic": True, "fault": {"id": fid, "knob": cell.get("fault"),
            "from_start": bool(cell.get("from_start"))}, "kind": fid, "duration_steps": d, "dt": DT,
            "num_envs": num_envs, "env_seed": env_seed,
            "onset": {"window_s": [lo, hi], "window_steps": [int(round(lo / DT)), int(round(hi / DT))]},
            "waves": recs}


# ====================================================================================== pure metric functions
def fall_info(n_steps: int, reason: str, onset: int) -> dict:
    """Fall bookkeeping of one episode against its onset (fall step = n_steps - 1)."""
    fell = reason != COMPLETE
    fs = int(n_steps) - 1 if fell else None
    # alive at onset = the episode actually REACHED its onset step (onset < n_steps; a fall at step fs >= onset
    # implies it). A completed episode shorter than its onset (armed-but-never smoke: onsets ~150000 steps) never
    # reached it and is not in the F denominator.
    alive = int(onset) < int(n_steps)
    post = fell and fs >= onset
    return {"fell": fell, "fall_step": fs, "alive_at_onset": bool(alive), "post_fall": bool(post),
            "pre_fall": bool(fell and fs < onset), "f2": bool(post and fs < onset + F2_STEPS),
            "exposure_steps": int(max(0, int(n_steps) - onset)) if alive else 0}


def window_idx(seg: np.ndarray, steady: np.ndarray | None, hold_idx: int, n_steps: int, ended_in_fall: bool,
               start: int = 0) -> np.ndarray:
    """``report._window_mask`` (the harness's hold window) intersected with t >= ``start``."""
    total = seg.shape[0]
    upper = min(int(n_steps), total)
    if ended_in_fall:
        upper = max(0, upper - FALL_TRIM_STEPS)
    m = np.zeros(total, dtype=bool)
    m[max(0, int(start)):upper] = True
    m &= seg == hold_idx
    if steady is not None:
        m &= steady.astype(bool)
    return np.flatnonzero(m)


def travel_vx(lin_vel_b: np.ndarray, travel_yaw: np.ndarray) -> np.ndarray:
    """Body-frame xy velocity in the per-step travel frame, x component (metrics.lin_vel_to_travel_frame)."""
    v = np.asarray(lin_vel_b, np.float64)
    a = -np.asarray(travel_yaw, np.float64)
    return np.cos(a) * v[:, 0] - np.sin(a) * v[:, 1]


def cmd_speed(cmd_applied: np.ndarray) -> np.ndarray:
    return np.linalg.norm(np.asarray(cmd_applied, np.float64)[:, :2], axis=-1)


def ratio_of(vx: np.ndarray, sp: np.ndarray, idx: np.ndarray, min_steps: int = MIN_WINDOW_STEPS):
    """metrics.tracking_metrics' vx ratio on ``idx`` (None under ``min_steps`` or a sub-clip command)."""
    if idx.size < min_steps:
        return None
    c = float(sp[idx].mean())
    if abs(c) <= LIN_VEL_CLIP:
        return None
    return float(vx[idx].mean()) / c


def facing_reanchored(err: np.ndarray, idx: np.ndarray, floor: float = FC_FLOOR):
    """(facing_closed re-anchored at the window's first step, excluded) -- metrics.facing_metrics' rule:
    clip(1 - |e_last| / max(|e_first|, floor), -1, 1); excluded (None, True) when |e_first| < floor."""
    if idx.size == 0:
        return None, False
    e0, e1 = abs(float(err[idx[0]])), abs(float(err[idx[-1]]))
    if e0 < floor:
        return None, True
    return float(min(1.0, max(-1.0, 1.0 - e1 / max(e0, floor)))), False


def roll_pitch_rms(quat: np.ndarray, travel_yaw: np.ndarray, idx: np.ndarray):
    """metrics.orientation_metrics (travel-frame quaternion) on ``idx``."""
    if idx.size == 0:
        return None, None
    q = np.asarray(quat[idx], np.float64)
    h = np.asarray(travel_yaw[idx], np.float64) / 2.0
    cw, sw = np.cos(h), np.sin(h)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    w, x, y, z = w * cw - z * sw, x * cw + y * sw, y * cw - x * sw, z * cw + w * sw
    roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))
    roll = (roll + np.pi) % (2.0 * np.pi) - np.pi
    pitch = (pitch + np.pi) % (2.0 * np.pi) - np.pi
    return float(np.sqrt(np.mean(roll ** 2))), float(np.sqrt(np.mean(pitch ** 2)))


def r_window(onset: int, clear, permanent: bool) -> tuple[int, int]:
    """[start, end) of R's 2-s window: [onset + 2 s, onset + 4 s) permanent, [clear + 2 s, clear + 4 s) transient."""
    base = int(onset) if permanent else int(clear)
    return base + R_LAG_STEPS, base + R_LAG_STEPS + R_LEN_STEPS


def r_episode(vx, sp, n_steps: int, reason: str, ws: int, we: int, extent: tuple[int, int]) -> dict:
    """One episode's R entry: inside (window within the translate hold's planned extent), alive at ws, ok."""
    inside = extent is not None and extent[0] <= ws and we <= extent[1]
    fell = reason != COMPLETE
    alive = ws < int(n_steps)                    # the episode reached the window start (fell or not)
    ok = None
    if inside and alive:
        if int(n_steps) < we:                    # fell inside the window: not tracking
            ok = False
        else:
            c = float(np.mean(sp[ws:we]))
            ok = bool(c > LIN_VEL_CLIP and abs(float(np.mean(vx[ws:we])) / c - 1.0) <= R_TOL)
    return {"inside": bool(inside), "alive": bool(alive), "ok": ok}


def r_share(entries: list):
    """(R, n qualifying) over r_episode entries (inside and alive)."""
    q = [e for e in entries if e["inside"] and e["alive"]]
    if not q:
        return None, 0
    return float(sum(1 for e in q if e["ok"]) / len(q)), len(q)


def median_or_none(v):
    v = [x for x in v if x is not None]
    return float(np.median(v)) if v else None


def verdict(dF: float, lines_ok: bool, R, R_required: bool, transient: bool) -> str:
    """CHARTER §5.3 rules 1-4, in order (first match decides)."""
    r_ok = (not R_required) or (R is not None and R >= R_MIN - 1e-12)
    if dF > DF_FALLS + 1e-12:
        return "FALLS"
    if lines_ok and dF <= DF_TOL + 1e-12 and r_ok:
        return "TOLERATED"
    if transient and dF <= DF_REC + 1e-12 and r_ok:
        return "RECOVERS"
    return "DEGRADES"


def marginal(dF: float, n: int, transient: bool) -> list:
    """The ΔF thresholds the worst ΔF lies within ``MARGINAL_EPISODES`` episodes of (n = its row's denominator)."""
    if n <= 0:
        return []
    ths = [DF_TOL, DF_FALLS] + ([DF_REC] if transient else [])
    return sorted(t for t in ths if abs(dF - t) * n <= MARGINAL_EPISODES + 1e-9)


def worse(*vs: str) -> str:
    return max(vs, key=lambda v: SEVERITY[v])


def combine_seeds(per_seed: dict, es1_F_worst=None) -> dict:
    """Cell verdict = the worse of the seed verdicts, ``split`` when they differ; es2 absent: FALLS stands on es1
    when es1's worst-row F >= 0.5 (CHARTER §4), otherwise pending."""
    vs = {s: v for s, v in per_seed.items() if v is not None}
    if not vs:
        return {"verdict": None, "split": False, "pending": True}
    if len(vs) == 1:
        (s, v), = vs.items()
        if v == "FALLS" and es1_F_worst is not None and es1_F_worst >= 0.5:
            return {"verdict": v, "split": False, "pending": False, "one_seed_rule": True}
        return {"verdict": v, "split": False, "pending": True}
    w = worse(*vs.values())
    return {"verdict": w, "split": len(set(vs.values())) > 1, "pending": False}


def tolerated_duration(levels: list) -> dict:
    """``levels``: [(d_steps, cell verdict or None, pending)] of one transient family. The longest d such that it and
    every shorter level are TOLERATED or RECOVERS (on both seeds, i.e. the combined verdict, not pending)."""
    best, stopped = None, None
    for d, v, pend in sorted(levels, key=lambda x: x[0]):
        if pend or v is None:
            stopped = "pending"
            break
        if v not in ("TOLERATED", "RECOVERS"):
            stopped = v
            break
        best = d
    return {"tolerated_steps": best, "tolerated_s": None if best is None else best * DT, "stopped_by": stopped}


# ====================================================================================== per-episode workers
_SLIM_KEYS = ("segment_id", "steady_mask", "cmd_applied", "root_lin_vel_b", "travel_yaw", "heading_err",
              "root_quat_w", "root_pos_w", "terrain_z")


def episode_record(job: dict) -> dict:
    """One episode's onset-dependent record + its R series (vx_travel, |cmd|). Pure on the npz."""
    ep, onset, path = job["ep"], int(job["onset"]), job["path"]
    with np.load(path, allow_pickle=False) as z:
        a = {k: z[k] for k in _SLIM_KEYS if k in z.files}
    n = int(ep["n"])
    if a["segment_id"].shape[0] != n:
        raise ValueError(f"{path}: raw has {a['segment_id'].shape[0]} samples, n_steps {n}")
    seg, steady = a["segment_id"].astype(np.int64), a["steady_mask"].astype(bool)
    rec = dict(ep)
    rec["onset"] = onset
    rec.update(fall_info(n, ep["reason"], onset))
    ended_in_fall = ep["reason"] in FALL_REASONS
    vx = travel_vx(a["root_lin_vel_b"], a["travel_yaw"])
    sp = cmd_speed(a["cmd_applied"])
    hold_labels = job.get("hold_labels")
    rec["surv_own"] = None
    if job.get("suite") and ep["label"] in (hold_labels or []):
        h = hold_labels.index(ep["label"])
        rec["surv_own"] = bool(n >= int(job["hold_end"]))
        for tag, st in (("post", onset), ("whole", 0)):
            idx = window_idx(seg, steady, h, n, ended_in_fall, st)
            rec[f"ratio_{tag}"] = ratio_of(vx, sp, idx) if rec["alive_at_onset"] or tag == "whole" else None
            widx = window_idx(seg, None, h, n, ended_in_fall, st)
            if "heading_err" in a and ep["cat"] == "turn":
                fc, exc = facing_reanchored(a["heading_err"], widx)
                if tag == "post" and not rec["alive_at_onset"]:
                    fc, exc = None, False
                rec[f"fc_{tag}"], rec[f"fc_{tag}_excluded"] = fc, exc
    # orientation / h_meas: [onset, n) post; [0, n) whole (h: segment > 0)
    hz = a["root_pos_w"][:, 2].astype(np.float64) - a["terrain_z"].astype(np.float64).mean(axis=1)
    for tag, st in (("post", onset), ("whole", 0)):
        if tag == "post" and not rec["alive_at_onset"]:
            rec.update({f"roll_{tag}": None, f"pitch_{tag}": None, f"h_{tag}": None})
            continue
        idx = np.arange(min(st, n), n)
        r, p = roll_pitch_rms(a["root_quat_w"], a["travel_yaw"], idx)
        rec[f"roll_{tag}"], rec[f"pitch_{tag}"] = r, p
        hidx = idx[seg[idx] > 0]
        rec[f"h_{tag}"] = float(np.median(hz[hidx])) if hidx.size else None
    xy = a["root_pos_w"][:, :2].astype(np.float64)
    rec["r_max"] = float(np.linalg.norm(xy - xy[0:1], axis=1).max()) if n else 0.0
    rec["complete"] = ep["reason"] == COMPLETE
    return {"rec": rec, "vx": vx.astype(np.float32), "sp": sp.astype(np.float32), "seg": seg.astype(np.int16)}


def first_diff(job: dict) -> dict:
    """First npz row (any key) where two episodes differ bitwise (classflip_check.first_diff), over the full length;
    a length mismatch with equal common rows -> the shorter length. ``act_step`` = the first row where the
    ``actions`` key alone differs (None = equal over the common rows): a first divergence at a step with
    ``act_step`` None or > step has equal policy outputs through that step ("physics drift", ``identity_read``)."""
    x, y = job["x"], job["y"]
    with np.load(x, allow_pickle=False) as a, np.load(y, allow_pickle=False) as b:
        if set(a.files) != set(b.files):
            return {"idx": job["idx"], "step": 0, "keys": sorted(set(a.files) ^ set(b.files)), "struct": True}
        best, which, act_step = None, [], None
        for k in sorted(a.files):
            u, v = a[k], b[k]
            if u.ndim == 0:
                if u.tobytes() != v.tobytes():
                    return {"idx": job["idx"], "step": 0, "keys": [k], "struct": True}
                continue
            if u.dtype != v.dtype or u.shape[1:] != v.shape[1:]:
                return {"idx": job["idx"], "step": 0, "keys": [k], "struct": True}
            n = min(len(u), len(v))
            if n:
                ub = np.ascontiguousarray(u[:n]).view(np.uint8).reshape(n, -1)
                vb = np.ascontiguousarray(v[:n]).view(np.uint8).reshape(n, -1)
                neq = np.any(ub != vb, axis=1)
                t = int(np.argmax(neq)) if neq.any() else (n if len(u) != len(v) else None)
            else:
                t = 0 if len(u) != len(v) else None
            if t is None:
                continue
            if k == "actions":
                act_step = t
            if best is None or t < best:
                best, which = t, [k]
            elif t == best:
                which.append(k)
    return {"idx": job["idx"], "step": best, "keys": which, "struct": False, "act_step": act_step}


def class_hash(job: dict) -> tuple:
    """sha256 of every npz key's rows [0, horizon] (bitwise) of one episode."""
    h = hashlib.sha256()
    with np.load(job["path"], allow_pickle=False) as z:
        for k in sorted(z.files):
            u = z[k]
            h.update(k.encode())
            h.update(str(u.dtype).encode())
            if u.ndim == 0:
                h.update(u.tobytes())
            else:
                h.update(np.ascontiguousarray(u[: job["horizon"] + 1]).tobytes())
    return job["idx"], h.hexdigest()


def dose_series(job: dict) -> dict:
    """Sim-side dose recompute of one faulted episode (CHARTER §6.2) from its npz."""
    kind, on, last = job["kind"], int(job["onset"]), job.get("last")
    with np.load(job["path"], allow_pickle=False) as z:
        rs = z["rod_speed"].astype(np.float64) if "rod_speed" in z.files else None
        q = z["joint_pos"].astype(np.float64)
        qd = z["joint_vel"].astype(np.float64)
        act = z["actions"].astype(np.float64)
    n = q.shape[0]
    end = n if last is None else min(n, int(last) + 1)
    out = {"idx": job["idx"], "n": n, "active_steps": max(0, end - on)}
    if end <= on or rs is None:
        return out
    c = job.get("leg_col")
    if kind in ("A1", "A3") and c is not None:
        leg = rs[on:end][:, [c, 6 + c]]
        out["leg_rod_speed_absmax"] = float(np.abs(leg).max())
        out["leg_hip_ratio_max"] = float(np.abs(leg[:, 0]).max() / ROD_CAP_M_S["hip"])
        out["leg_knee_ratio_max"] = float(np.abs(leg[:, 1]).max() / ROD_CAP_M_S["knee"])
        jids = job.get("joint_ids") or []
        if jids and end > on + 1:
            out["leg_joint_pos_drift"] = float(np.abs(q[on + 1:end][:, jids] - q[on + 1][jids]).max())
    if kind == "B6":
        out["rod_speed_absmax"] = float(np.abs(rs[on:end]).max())
    if kind == "B4" and on >= 1:
        last_slew = rs[on - 1]
        cur = rs[on:end]
        same = np.abs(cur - last_slew[None, :]) <= 1e-6 + 1e-4 * np.abs(last_slew[None, :])
        clipped = (np.abs(cur) < np.abs(last_slew[None, :]) + 1e-9) & (np.sign(cur) * np.sign(last_slew)[None, :] >= 0)
        out["b4_slew_kept_frac"] = float(same.mean())
        out["b4_bad_steps"] = int((~(same | clipped)).any(axis=1).sum())
    if kind == "A2":
        jids = job.get("joint_ids") or []
        if jids:
            out["a2_joint_leave_max"] = float(np.abs(q[on:end][:, jids] - q[on][jids]).max())
    if kind == "A4" and job.get("cam_joint_id") is not None:
        cj = int(job["cam_joint_id"])
        out["a4_cam_joint_vel_absmax"] = float(np.abs(qd[on + 1:end, cj]).max()) if end > on + 1 else 0.0
        out["a4_cam_angle_drift"] = float(np.abs(q[on:end, cj] - q[on, cj]).max())
    if kind in ("B1", "B5") and on >= 2:
        # the applied action is not recorded (npz 'actions' = the policy output, nominal delay 1 -> applied[t] =
        # actions[t - 1]); the held / delayed action's gap to that nominal applied stream, on the run's own outputs
        held = act[on - 2] if kind == "B1" else None
        gaps = []
        for t in range(on, end):
            app = held if kind == "B1" else act[t - 2]
            gaps.append(float(np.abs(app - act[t - 1]).max()))
        out["applied_gap_max"] = max(gaps) if gaps else 0.0
    return out


def _pool(jobs: int):
    return Pool(max(1, int(jobs)))


# ====================================================================================== reading a launch
class Reader:
    def __init__(self, cells: dict, lines: dict, jobs: int = 8):
        self.cells = cells
        self.lines = lines
        self.jobs = jobs
        self._rec_cache = {}
        self._keep = set()
        self._hash_cache = {}
        self._pool = None

    def pool(self):
        if self._pool is None:
            self._pool = _pool(self.jobs)
        return self._pool

    def close(self):
        if self._pool is not None:
            self._pool.close()
            self._pool.join()
            self._pool = None

    # ---------------------------------------------------------------- per-launch records
    def records(self, L: Launch, onsets: np.ndarray) -> list:
        """Per-episode records of a launch on an onset table (nominals kept; other launches in a 4-deep LRU)."""
        key = (str(L.stamp), hashlib.sha1(onsets.tobytes()).hexdigest())
        if key in self._rec_cache:
            self._rec_cache[key] = self._rec_cache.pop(key)
            return self._rec_cache[key]
        hold_labels = L.run_meta.get("hold_labels")
        ext = L.suite_hold_extent() if L.run_meta.get("suite") else None
        jobs = []
        for ep in L.episodes():
            jobs.append({"ep": ep, "onset": int(onsets[ep["wave"], ep["env"]]), "path": str(L.npz(ep["idx"])),
                         "suite": bool(L.run_meta.get("suite")), "hold_labels": hold_labels,
                         "hold_end": None if ext is None else ext[1]})
        out = self.pool().map(episode_record, jobs, chunksize=8)
        transient = [k for k in self._rec_cache if k not in self._keep]
        if len(transient) >= 4:
            self._rec_cache.pop(transient[0])
        self._rec_cache[key] = out
        if L.cell in NOMINAL_CELLS:
            self._keep.add(key)
        return out

    def class_fingerprint(self, L: Launch) -> str:
        if str(L.stamp) in self._hash_cache:
            return self._hash_cache[str(L.stamp)]
        horizon = int(self.cells["rows"][L.row]["pre_onset_last_step"])
        eps = [e for e in L.episodes() if e["wave"] == 0]
        res = self.pool().map(class_hash, [{"path": str(L.npz(e["idx"])), "horizon": horizon, "idx": e["idx"]}
                                           for e in eps], chunksize=8)
        h = hashlib.sha256()
        for idx, d in sorted(res):
            h.update(f"{idx}:{d};".encode())
        self._hash_cache[str(L.stamp)] = h.hexdigest()
        return self._hash_cache[str(L.stamp)]

    def diffs(self, X: Launch, Y: Launch) -> dict:
        """{episode idx: first_diff} of two launches of a slot (every episode)."""
        ex, ey = {e["idx"] for e in X.episodes()}, {e["idx"] for e in Y.episodes()}
        if ex != ey:
            raise ValueError(f"episode sets differ: {X} vs {Y}")
        if X.stamp == Y.stamp:
            return {i: {"idx": i, "step": None, "keys": [], "struct": False} for i in ex}
        res = self.pool().map(first_diff, [{"x": str(X.npz(i)), "y": str(Y.npz(i)), "idx": i} for i in sorted(ex)],
                              chunksize=8)
        return {r["idx"]: r for r in res}


def grid_onset_horizon(row: dict, dt: float = DT) -> int:
    """Latest step a grid onset of this row can take = round(hi / dt) - 1 (suite 574, field_light 2999, ledge_down
    549, pits 669): the identity horizon of an armed-but-never row, whose own onsets lie beyond the episode."""
    hi = float(str(row["onset"]).split(":")[1])
    return int(round(hi / dt)) - 1


def is_physics_drift(d: dict) -> bool:
    """A divergence whose first differing step still has bitwise-equal ``actions`` through that step (CHARTER 5.1,
    2026-11-16 amendment): the policy output had not diverged, so the injector (which writes from the onset only)
    is not the cause -- spontaneous contact-solver drift. Actions differing at or before the step stay a break."""
    st = d.get("step")
    if st is None or d.get("struct"):
        return False
    a = d.get("act_step")
    return a is None or a > st


def wave_first_onsets(X: Launch, onsets: np.ndarray, horizon: int | None = None) -> dict:
    """Per wave, the first onset the wave actually reached: min onset over the envs alive at their onset
    (episode length > onset). A wave with no env reaching an onset (an armed-but-never row) has ``horizon + 1``
    (the row's latest grid onset + 1), or None without a horizon (no batch coupling is then possible)."""
    first: dict = {}
    for ep in X.episodes():
        on = int(onsets[ep["wave"], ep["env"]])
        if ep["n"] > on:
            first[ep["wave"]] = min(first.get(ep["wave"], on), on)
        else:
            first.setdefault(ep["wave"], None)
    out = {}
    for w, f in first.items():
        if horizon is not None:
            f = (int(horizon) + 1) if f is None else min(f, int(horizon) + 1)
        out[w] = f
    return out


def identity_read(X: Launch, diffs: dict, onsets: np.ndarray, gated_waves=(0,), horizon: int | None = None) -> dict:
    """Per-env pre-onset identity on the gated waves, and the divergence counts of the other waves (step-0
    divergences; divergences before the env's onset).

    Identity is strict per env up to ``min(own onset, wave first onset) - 1`` (CHARTER 5.1, 2026-11-16 01:40
    amendment: once some envs of the batch are faulted the batched GPU simulator perturbs the rest). A divergence
    there with equal actions through its first differing step is physics drift (``is_physics_drift``): counted,
    listed in ``drift`` / ``drift_idx``, not a break; more than ``MAX_PHYSICS_DRIFT`` of them make ``ok`` False.
    Any other divergence there (actions first or at the same step, structural) is a break. A divergence at a step
    in ``[wave first onset, own onset)`` that is physics-first (actions still equal) is batch coupling: counted
    per wave (``n_coupled``, ``first_coupled_step``) and in ``coupled`` / ``coupled_idx``, never capped, never an
    error; an actions-first one is a break (a policy-input change before the onset with no physics cause).
    Drift and coupled envs are excluded from per-episode pairing checks by the callers. ``horizon`` caps the own
    onset at that last step + 1 (an armed-but-never row: ``grid_onset_horizon``); a divergence after it is not
    pre-onset."""
    first_onsets = wave_first_onsets(X, onsets, horizon)
    breaks, drift, coupled, per_wave = [], [], [], {}
    for ep in X.episodes():
        d = diffs[ep["idx"]]
        on = int(onsets[ep["wave"], ep["env"]])
        if horizon is not None:
            on = min(on, int(horizon) + 1)
        fo = first_onsets.get(ep["wave"])
        strict_end = on if fo is None else min(on, int(fo))
        st = d["step"]
        pre = st is not None and st < on
        w = per_wave.setdefault(ep["wave"], {"n": 0, "step0": 0, "pre_onset": 0, "any": 0, "first_onset": fo,
                                             "n_coupled": 0, "first_coupled_step": None})
        w["n"] += 1
        w["any"] += int(st is not None)
        w["step0"] += int(st == 0)
        w["pre_onset"] += int(pre)
        if ep["wave"] in gated_waves and (pre or d.get("struct")):
            rec = {"idx": ep["idx"], "wave": ep["wave"], "env": ep["env"], "onset": on, "step": st,
                   "keys": d["keys"][:6], "act_step": d.get("act_step"), "first_onset": fo}
            if not pre:
                breaks.append(rec)
            elif st < strict_end:
                (drift if is_physics_drift(d) else breaks).append(rec)
            elif is_physics_drift(d):                       # fo <= st < own onset, actions still equal
                coupled.append(rec)
                w["n_coupled"] += 1
                w["first_coupled_step"] = st if w["first_coupled_step"] is None else min(w["first_coupled_step"], st)
            else:
                breaks.append(rec)
    return {"ok": not breaks and len(drift) <= MAX_PHYSICS_DRIFT, "breaks": breaks[:20], "n_breaks": len(breaks),
            "drift": drift[:20], "n_physics_drift": len(drift), "drift_idx": sorted(r["idx"] for r in drift),
            "coupled": coupled[:20], "n_batch_coupled": len(coupled), "coupled_idx": sorted(r["idx"] for r in coupled),
            "first_onsets": first_onsets, "horizon": horizon, "per_wave": per_wave}


# ====================================================================================== launch checks (§6.2)
def _probe(w: dict, name: str):
    """A sidecar probe as float64 (N,) or (N, width); None -> NaN; absent -> None."""
    v = (w.get("probes") or {}).get(name)
    if v is None:
        return None
    if v and isinstance(v[0], list):
        return np.asarray([[np.nan if y is None else y for y in row] for row in v], dtype=np.float64)
    return np.asarray([np.nan if x is None else x for x in v], dtype=np.float64)


def sidecar_checks(L: Launch, cell: dict, sc: dict, waves_expected: int, num_envs: int, env_seed: int,
                   window_s: tuple) -> dict:
    """The sidecar's realised dose and inactivity probes (CHARTER §6.2). Returns {errors: [...], report: {...}}."""
    errs, rep = [], {}
    fi = fault_inject()
    fid = cell_fault_id(cell)
    if sc.get("schema") != fi.SIDECAR_SCHEMA:
        errs.append(f"sidecar schema {sc.get('schema')!r}")
    if str((sc.get("fault") or {}).get("id")) != fid:
        errs.append(f"sidecar fault id {(sc.get('fault') or {}).get('id')!r} != cell {fid!r}")
    kind = str(sc.get("kind"))
    d_want = cell_duration_steps(cell)                 # None: permanent levels and the S12 / D1 anchors
    if sc.get("duration_steps") != d_want:
        errs.append(f"duration_steps {sc.get('duration_steps')} != cell {d_want}")
    waves = sorted(sc.get("waves") or [], key=lambda w: w["wave"])
    if len(waves) != waves_expected or [w["wave"] for w in waves] != list(range(waves_expected)):
        errs.append(f"sidecar waves {[w['wave'] for w in waves]} (want {waves_expected})")
    if int(sc.get("num_envs") or -1) != num_envs:
        errs.append(f"sidecar num_envs {sc.get('num_envs')} != {num_envs}")
    if sc.get("env_seed") is not None and int(sc["env_seed"]) != int(env_seed):
        errs.append(f"sidecar env_seed {sc['env_seed']} != row env seed {env_seed}")
    if tuple(float(x) for x in (sc.get("onset") or {}).get("window_s", [])) != tuple(window_s):
        errs.append(f"sidecar onset window {(sc.get('onset') or {}).get('window_s')} != row {list(window_s)}")
    if waves:
        tab = onset_table_np(window_s[0], window_s[1], num_envs, env_seed, len(waves))
        if not np.array_equal(tab, sidecar_onsets(sc)):
            errs.append("sidecar onsets != fault_inject.onset_table(row window, dt, N, env seed)")
    depth = kind in fi.DEPTH_KINDS
    from_start = bool((sc.get("fault") or {}).get("from_start"))
    n_alive = n_active = 0
    for w in waves:
        wv = w["wave"]
        on = np.asarray(w["onset"], np.int64)
        start = np.asarray(w["start"], np.int64)
        fell = np.asarray(w["fell_at"], np.int64)
        fa = np.asarray(w["first_active"], np.int64)
        la = np.asarray(w["last_active"], np.int64)
        if not w.get("state_cleared", False):
            errs.append(f"wave {wv}: state_cleared false")
        if not np.array_equal(start, np.zeros_like(on) if from_start else on):
            errs.append(f"wave {wv}: start != {'0' if from_start else 'onset'}")
        n_ep = np.full(on.shape, -1, np.int64)          # episode length of (wave, env), from scenario_metrics
        for ep in L.episodes():
            if ep["wave"] == wv and 0 <= ep["env"] < n_ep.size:
                n_ep[ep["env"]] = int(ep["n"])
        if (n_ep < 0).any():
            errs.append(f"wave {wv}: no episode length for envs {np.flatnonzero(n_ep < 0)[:8].tolist()}")
        survived = (fell < 0) | (fell > start)          # did not fall before the start step
        reached = start < n_ep                          # the start step exists in the episode (onset actually reached)
        alive = survived & reached
        unreached = survived & ~reached                 # onset beyond the episode: must stay fully inactive
        n_alive += int(alive.sum())
        n_active += int((fa >= 0).sum())
        # inactive before the start (every stage), off again after a transient
        for name in (w.get("probes") or {}):
            if name.endswith("_pre_dev") or name.endswith("_post_dev"):
                arr = _probe(w, name)
                bad = np.flatnonzero(np.nan_to_num(arr, nan=0.0) > ZERO_TOL)
                if bad.size:
                    errs.append(f"wave {wv}: {name} > 0 on envs {bad[:8].tolist()}")
        if (fa[~survived] >= 0).any():
            errs.append(f"wave {wv}: active on envs that fell before the start {np.flatnonzero(~survived & (fa >= 0))[:8].tolist()}")
        if (unreached & ((fa >= 0) | (la >= 0))).any():
            errs.append(f"wave {wv}: active on envs whose start is beyond the episode (armed-but-never dose) "
                        f"{np.flatnonzero(unreached & ((fa >= 0) | (la >= 0)))[:8].tolist()}")
        lag = (fa - start)[alive]
        if depth:
            okfa = (lag >= 0) & (lag < 5)
        else:
            okfa = lag == 0
        if (~okfa).any():
            errs.append(f"wave {wv}: first_active != start on {int((~okfa).sum())} alive envs (lags {sorted(set(lag[~okfa].tolist()))[:6]})")
        if w.get("planned_last") is not None:
            pl = np.asarray(w["planned_last"], np.int64)
            if (la > pl).any():
                errs.append(f"wave {wv}: active after planned_last on envs {np.flatnonzero(la > pl)[:8].tolist()}")
            through = ((fell < 0) | (fell > pl)) & reached      # unreached envs never run the dose
            if depth:
                vis = np.asarray(w.get("policy_visible_last") or [-1] * len(on), np.int64)
                bnd = np.asarray(w.get("policy_visible_bound") or pl.tolist(), np.int64)
                if (vis > bnd).any():
                    errs.append(f"wave {wv}: policy_visible_last > policy_visible_bound")
                low = through & (la < pl - 4)
            else:
                low = through & (la != pl)
            if low.any():
                errs.append(f"wave {wv}: last_active short of planned_last on envs {np.flatnonzero(low)[:8].tolist()}")
        errs.extend(f"wave {wv}: {e}" for e in _dose_probe_errors(kind, cell, w, alive))
    rep.update(n_alive_at_start=n_alive, n_active=n_active)
    rep.update(_dose_probe_report(kind, waves))
    return {"errors": errs, "report": rep}


def _zero(w, name, alive, errs, tol=ZERO_TOL, required=True):
    arr = _probe(w, name)
    if arr is None:
        if required and alive.any():
            errs.append(f"probe {name} missing")
        return
    a = np.nan_to_num(arr if arr.ndim == 1 else arr.max(axis=1), nan=0.0)
    bad = alive & (np.abs(a) > tol)
    if bad.any():
        errs.append(f"{name} > {tol:g} on envs {np.flatnonzero(bad)[:8].tolist()} (max {float(np.abs(a[bad]).max()):.3g})")


def _dose_probe_errors(kind: str, cell: dict, w: dict, alive: np.ndarray) -> list:
    """Per-kind realised-dose probes (self_check + sim) of one wave, on envs alive at the start."""
    e = []
    if kind == "S1":
        _zero(w, "vxy_absmax", alive, e)
    if kind in ("S2", "Z1"):
        _zero(w, "vxy_held_dev", alive, e)
    if kind in ("S5", "Z1"):
        _zero(w, "ang_vel_held_dev", alive, e)
        _zero(w, "roll_pitch_held_dev", alive, e)
    if kind in ("S3", "S6"):
        _zero(w, "bias_dev", alive, e, tol=1e-5)
    if kind == "S8":
        if str(cell.get("fault", "")).endswith(":zero"):
            _zero(w, "cf_absmax", alive, e)
        else:
            _zero(w, "contact_fill_held_dev", alive, e)
    if kind in ("S10", "Z1"):
        _zero(w, "front_raw_absmax", alive, e)
        _zero(w, "front_img_dev", alive, e, tol=1e-5)
        fin = _probe(w, "front_raw_in_max")
        if fin is not None and alive.any() and not (np.nan_to_num(fin[alive]) > 0).any():
            e.append("front_raw_in_max is 0 everywhere (the real front frame never had depth)")
        for cam in ("left", "right", "rear"):
            _zero(w, f"depth_{cam}_change_max", alive, e, required=False)
    if kind == "S11":
        for cam in ("left", "right", "rear"):
            _zero(w, f"{cam}_frozen_dev", alive, e)
        _zero(w, "depth_front_change_max", alive, e, required=False)
    if kind == "S12":
        _zero(w, "s12_hold_dev", alive, e)
        hn, rn, an = (_probe(w, k) for k in ("s12_hold_n", "s12_refresh_n", "obs_active_n"))
        if hn is not None and rn is not None and an is not None:
            if (np.abs(hn + rn - an) > 0)[alive].any():
                e.append("s12 hold + refresh != active obs steps")
            frac = hn[alive].sum() / max(an[alive].sum(), 1)
            if abs(frac - 0.4) > 0.01:
                e.append(f"s12 hold share {frac:.4f} != 2/5")
    if kind == "D1":
        _zero(w, "d1_jpd_dev", alive, e, tol=1e-5)
        _zero(w, "d1_jv_cf_absmax", alive, e)
    if kind == "B1":
        rn, an = _probe(w, "repeat_n"), _probe(w, "act_active_n")
        if rn is not None and an is not None and (rn != an)[alive].any():
            e.append("B1 repeat_n != active steps")
    if kind == "B3":
        ds = [_probe(w, f"delay{d}_n") for d in (1, 2, 3)]
        an = _probe(w, "act_active_n")
        if all(x is not None for x in ds) and an is not None and (sum(ds) != an)[alive].any():
            e.append("B3 delay counts do not sum to the active steps")
    if kind == "B4":
        for fam in ("hip", "knee"):
            _zero(w, f"b4_{fam}_slew_dev", alive, e, tol=1e-6)
        _zero(w, "b4_cam_cmd_dev", alive, e)
    if kind == "B6":
        for fam in ("hip", "knee"):
            _zero(w, f"b6_{fam}_dlam_absmax", alive, e)
        _zero(w, "b6_cam_cmd_absmax", alive, e)
    if kind == "A1":
        for fam in ("hip", "knee"):
            _zero(w, f"a1_{fam}_dlam_absmax", alive, e)
            _zero(w, f"a1_{fam}_len_drift", alive, e, tol=1e-6)
    if kind == "A3":
        for fam in ("hip", "knee"):
            arr = _probe(w, f"a3_{fam}_slew_ratio_max")
            if arr is not None and (np.nan_to_num(arr) > 0.3 * (1 + 1e-5))[alive].any():
                e.append(f"a3_{fam}_slew_ratio_max > 0.3")
    if kind == "A4":
        _zero(w, "a4_cam_target_absmax", alive, e)
    if kind == "A2":
        post = _probe(w, "a2_gain_post")
        pre = _probe(w, "a2_gain_pre")
        wrote = _probe(w, "a2_write_step")
        if post is None or pre is None:
            if alive.any():
                e.append("A2 gain read-backs missing")
        else:
            got = ~np.isnan(post).all(axis=1)
            if (alive & ~got).any():
                e.append(f"A2 gain never written on alive envs {np.flatnonzero(alive & ~got)[:8].tolist()}")
            if (np.abs(np.nan_to_num(post[got])) > ZERO_TOL).any():
                e.append("A2 PhysX read-back after the write != 0 (stiffness / damping / max effort)")
            if not (np.nan_to_num(pre[got][:, :2]) > 0).all():
                e.append("A2 PhysX stiffness before the write is not > 0")
            if wrote is not None:
                on = np.asarray(w["onset"], np.float64)
                if (np.abs(np.nan_to_num(wrote, nan=-1)[got] - on[got]) > 0).any():
                    e.append("A2 written at a step != onset")
        _zero(w, "a2_gain_pre_onset_dev", alive, e, tol=1e-4)
        _zero(w, "a2_gain_wave_start_dev", np.ones_like(alive), e, tol=1e-4)
    return e


def _dose_probe_report(kind: str, waves: list) -> dict:
    """Report-only realised dose numbers pooled over the waves (S4 RMS, B2 p, B3 shares, A2 leave)."""
    rep = {}
    if kind == "S4":
        v = [np.asarray(x, np.float64) for w in waves for x in ((w.get("probes") or {}).get("walk_10s") or [])
             if x is not None and not any(y is None for y in x)]
        if v:
            v = np.stack(v)
            rep["s4_walk_10s_rms_per_axis"] = float(np.sqrt(np.mean(v ** 2)))
            rep["s4_walk_10s_norm_rms"] = float(np.sqrt(np.mean((v ** 2).sum(axis=1))))
            rep["s4_walk_10s_n"] = int(v.shape[0])
    if kind == "B2":
        rn = sum(float(np.nansum(_probe(w, "repeat_n"))) for w in waves if _probe(w, "repeat_n") is not None)
        an = sum(float(np.nansum(_probe(w, "act_active_n"))) for w in waves if _probe(w, "act_active_n") is not None)
        rep["b2_realised_p"] = rn / an if an else None
        rep["b2_active_steps"] = an
    if kind == "B3":
        cs = [sum(float(np.nansum(_probe(w, f"delay{d}_n"))) for w in waves if _probe(w, f"delay{d}_n") is not None)
              for d in (1, 2, 3)]
        tot = sum(cs)
        rep["b3_delay_shares"] = [c / tot for c in cs] if tot else None
    if kind == "A2":
        v = [np.nan_to_num(_probe(w, "a2_target_leave_max")) for w in waves if _probe(w, "a2_target_leave_max") is not None]
        if v:
            v = np.concatenate(v)
            rep["a2_target_leave_max_p50"] = float(np.median(v))
            rep["a2_target_leave_gt_0p2_share"] = float((v > 0.2).mean())
        tq = [np.nan_to_num(_probe(w, "a2_applied_torque_absmax")) for w in waves
              if _probe(w, "a2_applied_torque_absmax") is not None]
        if tq:
            rep["a2_applied_torque_absmax_model_estimate"] = float(np.concatenate(tq).max())
    if kind == "A4":
        v = [np.nan_to_num(_probe(w, "a4_cam_joint_vel_absmax")) for w in waves
             if _probe(w, "a4_cam_joint_vel_absmax") is not None]
        if v:
            rep["a4_cam_joint_vel_absmax_sim"] = float(np.concatenate(v).max())
    return rep


def npz_dose_checks(reader: Reader, L: Launch, cell: dict, sc: dict, onsets: np.ndarray, diffs: dict | None,
                    paired_waves=(0,), exclude=frozenset()) -> dict:
    """Sim-side doses recomputed from the npz (CHARTER §6.2): A1 locked leg rod_speed 0 (+ joint_pos drift
    reported), A3 / B4 / B6 rod_speed, B1 / B5 applied-action gap, A2 joint leave, A4 cam; on paired waves the
    first sim divergence of an A / B fault must lie at or after the onset (and inside the active window + 1 for
    B1 / B4 / B6). ``exclude`` = episode idx dropped from that pairing check (physics-drift and batch-coupled envs);
    with fewer than ``MIN_PAIRED_ENVS`` non-excluded envs left the pairing check is report-only (a warning)."""
    kind = str(sc.get("kind"))
    errs, rep, warns = [], {}, []
    if kind not in ("A1", "A2", "A3", "A4", "B1", "B2", "B3", "B4", "B5", "B6"):
        return {"errors": errs, "report": rep, "warnings": warns}
    b = sc.get("binding") or {}
    waves = {w["wave"]: w for w in sc["waves"]}
    jobs = []
    for ep in L.episodes():
        w = waves.get(ep["wave"])
        if w is None:
            continue
        on = int(onsets[ep["wave"], ep["env"]])
        if ep["n"] <= on:
            continue
        last = None if w.get("planned_last") is None else int(w["planned_last"][ep["env"]])
        jobs.append({"path": str(L.npz(ep["idx"])), "idx": ep["idx"], "kind": kind, "onset": on, "last": last,
                     "leg_col": b.get("leg_col"), "joint_ids": b.get("joint_ids"),
                     "cam_joint_id": (L.run_meta.get("joint_names") or []).index(b["cam_joint"])
                     if b.get("cam_joint") in (L.run_meta.get("joint_names") or []) else None})
    res = reader.pool().map(dose_series, jobs, chunksize=8)
    if kind in ("A1",):
        m = max((r.get("leg_rod_speed_absmax", 0.0) for r in res), default=0.0)
        rep["a1_leg_rod_speed_absmax"] = m
        if m > 1e-7:
            errs.append(f"A1 locked-leg rod_speed max {m:.3g} m/s after onset (want 0)")
        dr = [r["leg_joint_pos_drift"] for r in res if "leg_joint_pos_drift" in r]
        if dr:
            rep["a1_leg_joint_pos_drift_p50"] = float(np.median(dr))
            rep["a1_leg_joint_pos_drift_max"] = float(max(dr))
    if kind == "A3":
        hr = max((r.get("leg_hip_ratio_max", 0.0) for r in res), default=0.0)
        kr = max((r.get("leg_knee_ratio_max", 0.0) for r in res), default=0.0)
        rep["a3_leg_rod_speed_over_cap_max"] = {"hip": hr, "knee": kr}
        if hr > 0.3 * (1 + 1e-4) or kr > 0.3 * (1 + 1e-4):
            errs.append(f"A3 leg rod_speed / cap max (hip {hr:.4f}, knee {kr:.4f}) > 0.3")
        # report only: whether the reduced cap binds depends on what the policy commands (the smoke asserts it)
        rep["a3_reduced_cap_reached"] = bool(hr >= 0.3 * (1 - 1e-3) or kr >= 0.3 * (1 - 1e-3))
        rep["a3_cap_note"] = "cap = crab_hex_dimensions speed_loaded_m_s (no rod-speed DR on the sweep's env files)"
    if kind == "B6":
        m = max((r.get("rod_speed_absmax", 0.0) for r in res), default=0.0)
        rep["b6_rod_speed_absmax"] = m
        if m > 1e-7:
            errs.append(f"B6 rod_speed max {m:.3g} m/s while active (want 0)")
    if kind == "B4":
        bad = sum(r.get("b4_bad_steps", 0) for r in res)
        rep["b4_slew_kept_frac_p50"] = median_or_none([r.get("b4_slew_kept_frac") for r in res])
        rep["b4_bad_steps"] = bad
        if bad:
            errs.append(f"B4 rod_speed left the last pre-onset slew (unclipped) on {bad} steps")
    if kind in ("B1", "B5"):
        g = [r["applied_gap_max"] for r in res if "applied_gap_max" in r]
        rep["applied_gap_max_p50"] = median_or_none(g)
        rep["applied_gap_note"] = "policy-output proxy: nominal applied[t] = actions[t-1] (delay 1), pre-clip"
    if kind == "A2":
        v = [r["a2_joint_leave_max"] for r in res if "a2_joint_leave_max" in r]
        if v:
            rep["a2_joint_leave_from_onset_p50"] = float(np.median(v))
            rep["a2_joint_leave_gt_0p2_share"] = float(np.mean(np.asarray(v) > 0.2))
    if kind == "A4":
        v = [r.get("a4_cam_joint_vel_absmax") for r in res if r.get("a4_cam_joint_vel_absmax") is not None]
        rep["a4_cam_joint_vel_absmax_npz"] = max(v) if v else None
        a = [r.get("a4_cam_angle_drift") for r in res if r.get("a4_cam_angle_drift") is not None]
        rep["a4_cam_angle_drift_max_npz"] = max(a) if a else None
    if diffs is not None:
        early, never, late, n, excl = 0, 0, 0, 0, 0
        for ep in L.episodes():
            if ep["wave"] not in paired_waves:
                continue
            if ep["idx"] in exclude:
                excl += 1
                continue
            on = int(onsets[ep["wave"], ep["env"]])
            if ep["n"] <= on:
                continue
            n += 1
            st = diffs[ep["idx"]]["step"]
            w = waves[ep["wave"]]
            last = None if w.get("planned_last") is None else int(w["planned_last"][ep["env"]])
            if st is None:
                never += 1
            elif st < on:
                early += 1
            elif kind in ("B1", "B4", "B6") and last is not None and st > last + 1:
                late += 1
        rep["paired_divergence"] = {"n": n, "never": never, "before_onset": early, "after_active_window": late,
                                   "physics_drift_excluded": excl}
        report_only = n < MIN_PAIRED_ENVS
        rep["paired_divergence"]["report_only"] = report_only
        sink = warns if report_only else errs
        if report_only and (early or never or late):
            warns.append(f"pairing check report-only: {n} non-excluded paired env(s) < {MIN_PAIRED_ENVS}")
        if early:
            sink.append(f"{early} paired episodes diverge before the onset")
        if n and (never + late) > 0.10 * n:
            sink.append(f"{never + late}/{n} paired episodes show no sim divergence inside the active window")
    return {"errors": errs, "report": rep, "warnings": warns}


# ====================================================================================== statistics of one row read
def _agg_suite(recs: list) -> dict:
    by_cat = collections.defaultdict(list)
    by_label = collections.defaultdict(list)
    for r in recs:
        by_cat[r["cat"]].append(r)
        by_label[r["label"]].append(r)
    out = {"categories": {}, "hands": {}}
    for c, rs in by_cat.items():
        surv = [r["surv_own"] for r in rs if r["surv_own"] is not None]
        e = {"n": len(rs), "survival": float(np.mean(surv)) if surv else None,
             "ratio_post": median_or_none([r.get("ratio_post") for r in rs]),
             "ratio_whole": median_or_none([r.get("ratio_whole") for r in rs])}
        if c == "turn":
            e["fc_post"] = median_or_none([r.get("fc_post") for r in rs])
            e["fc_post_n"] = sum(1 for r in rs if r.get("fc_post") is not None)
            e["fc_post_excluded"] = sum(1 for r in rs if r.get("fc_post_excluded"))
            e["fc_whole"] = median_or_none([r.get("fc_whole") for r in rs])
        out["categories"][c] = e
    for lab, rs in by_label.items():
        e = {"n": len(rs), "ratio_post": median_or_none([r.get("ratio_post") for r in rs]),
             "ratio_whole": median_or_none([r.get("ratio_whole") for r in rs])}
        if rs and rs[0]["cat"] == "turn":
            e["fc_post"] = median_or_none([r.get("fc_post") for r in rs])
            e["fc_post_n"] = sum(1 for r in rs if r.get("fc_post") is not None)
            e["fc_post_excluded"] = sum(1 for r in rs if r.get("fc_post_excluded"))
            e["fc_whole"] = median_or_none([r.get("fc_whole") for r in rs])
        out["hands"][lab] = e
    for tag in ("post", "whole"):
        out[f"h_meas_{tag}"] = median_or_none([r.get(f"h_{tag}") for r in recs])
        out[f"roll_rms_{tag}"] = median_or_none([r.get(f"roll_{tag}") for r in recs])
        out[f"pitch_rms_{tag}"] = median_or_none([r.get(f"pitch_{tag}") for r in recs])
    return out


def _agg_field(recs: list) -> dict:
    out = {"completion": float(np.mean([r["complete"] for r in recs])) if recs else None}
    for tag in ("post", "whole"):
        out[f"h_meas_{tag}"] = median_or_none([r.get(f"h_{tag}") for r in recs])
        out[f"roll_rms_{tag}"] = median_or_none([r.get(f"roll_{tag}") for r in recs])
        out[f"pitch_rms_{tag}"] = median_or_none([r.get(f"pitch_{tag}") for r in recs])
    return out


def _falls(recs: list) -> dict:
    alive = [r for r in recs if r["alive_at_onset"]]
    post = sum(r["post_fall"] for r in recs)
    exp_min = sum(r["exposure_steps"] for r in recs) * DT / 60.0
    return {"n": len(recs), "alive_at_onset": len(alive), "post_falls": int(post),
            "pre_falls": int(sum(r["pre_fall"] for r in recs)), "f2": int(sum(r["f2"] for r in recs)),
            "F": (post / len(alive)) if alive else None, "exposure_min": exp_min,
            "hazard_per_min": (post / exp_min) if exp_min > 0 else None,
            "terminations_post": dict(collections.Counter(r["reason"] for r in recs if r["alive_at_onset"]
                                                          and r["fell"]))}


def field_translate_extents(seg: np.ndarray, hold_labels: list) -> list:
    """[(label, start, end)] of field_light's translate holds from a (complete) episode's segment_id."""
    out = []
    s = seg.astype(np.int64)
    b = np.flatnonzero(np.diff(s)) + 1
    starts = np.concatenate([[0], b])
    ends = np.concatenate([b, [len(s)]])
    for a, e in zip(starts, ends):
        k = int(s[a])
        if 0 <= k < len(hold_labels) and hold_labels[k] in TRANSLATE_LABELS:
            out.append((hold_labels[k], int(a), int(e)))
    return out


def r_entries(outs: list, L: Launch, ref_outs: list, onsets, clear, permanent: bool) -> dict:
    """Per translate wave (suite) / per row (field_light): R entries of each episode (both launches)."""
    if L.run_meta.get("suite"):
        ext = L.suite_hold_extent()
        exts = {lab: ext for lab in TRANSLATE_LABELS}
        field = None
    else:
        longest = max(ref_outs, key=lambda o: len(o["seg"]))
        field = field_translate_extents(longest["seg"], L.run_meta.get("hold_labels") or [])
    groups = collections.defaultdict(list)
    for o in outs:
        r = o["rec"]
        on = int(onsets[r["wave"], r["env"]])
        cl = None if permanent else int(clear[r["wave"], r["env"]])
        ws, we = r_window(on, cl, permanent)
        if field is None:
            if r["label"] not in TRANSLATE_LABELS:
                continue
            e = r_episode(o["vx"], o["sp"], r["n"], r["reason"], ws, we, exts[r["label"]])
            groups[f"wave{r['wave']}:{r['label']}"].append(e)
        else:
            ext = next(((a, b) for _lab, a, b in field if a <= ws and we <= b), None)
            e = r_episode(o["vx"], o["sp"], r["n"], r["reason"], ws, we, ext)
            groups["field_light"].append(e)
    return groups


def read_row(reader: Reader, X: Launch, ref: Launch, onsets: np.ndarray, clear, permanent: bool) -> dict:
    """Every §5.2 statistic of one row (faulted launch X vs reference) on the given per-env windows."""
    ox = reader.records(X, onsets)
    oref = reader.records(ref, onsets)
    rx = [o["rec"] for o in ox]
    rr = [o["rec"] for o in oref]
    out = {"row": X.row, "stamp": str(X.stamp), "ref": str(ref.stamp)}
    if X.run_meta.get("suite"):
        waves = sorted({r["wave"] for r in rx})
        out["waves"] = {}
        for w in waves:
            fx, fr = _falls([r for r in rx if r["wave"] == w]), _falls([r for r in rr if r["wave"] == w])
            lab = next(r["label"] for r in rx if r["wave"] == w)
            out["waves"][w] = {"label": lab, "x": fx, "ref": fr,
                               "dF": None if fx["F"] is None or fr["F"] is None else fx["F"] - fr["F"]}
        out["falls_total"] = {"x": _falls(rx), "ref": _falls(rr)}
        out["x"] = _agg_suite(rx)
        out["refagg"] = _agg_suite(rr)
    else:
        fx, fr = _falls(rx), _falls(rr)
        out["falls"] = {"x": fx, "ref": fr, "dF": None if fx["F"] is None or fr["F"] is None else fx["F"] - fr["F"]}
        out["x"] = _agg_field(rx)
        out["refagg"] = _agg_field(rr)
        if X.row in KIND_ROWS:
            sx = [r["complete"] and r["r_max"] >= CROSS_R_M - 1e-9 for r in rx]
            sr = [r["complete"] and r["r_max"] >= CROSS_R_M - 1e-9 for r in rr]
            out["kind"] = {"success_x": float(np.mean(sx)), "success_ref": float(np.mean(sr)),
                           "d_success": float(np.mean(sx) - np.mean(sr)), "n": len(sx),
                           "radius_m": CROSS_R_M, "report_only": True}
    if X.row in ("suite", "field_light"):
        gx = r_entries(ox, X, oref, onsets, clear, permanent)
        gr = r_entries(oref, ref, oref, onsets, clear, permanent)
        out["R"] = {}
        for key in sorted(set(gx) | set(gr)):
            Rx, nx = r_share(gx.get(key, []))
            Rr, nr = r_share(gr.get(key, []))
            excl = sum(1 for e in gx.get(key, []) if not e["inside"])
            out["R"][key] = {"R": Rx, "n": nx, "R_ref": Rr, "n_ref": nr, "excluded_outside": excl,
                             "excluded_dead": sum(1 for e in gx.get(key, []) if e["inside"] and not e["alive"]),
                             "report_only": nr < R_FLOOR_N}
    return out


def lines_of(row_reads: dict, lines: dict, es: int) -> list:
    """The post-onset lines (CHARTER §5.2) of one cell-seed from its suite + field_light reads."""
    out = []

    def add(lid, v, stat, scope, ref=None, report_only=False, extra=None):
        ln = line_for(lines, stat, scope, es)
        if ln is None:
            raise ValueError(f"no threshold for {stat}/{scope}")
        line, direction = ln
        rec = {"id": lid, "value": v, "line": line, "dir": direction, "ref": ref,
               "pass": passes(v, line, direction) if not report_only else None}
        if v is None and not report_only:
            rec["pass"] = None                    # no measurable episode (e.g. every turn excluded): not decisive
            rec["note"] = "no value"
        if extra:
            rec.update(extra)
        out.append(rec)

    s = row_reads.get("suite")
    if s is not None:
        X, Rf = s["x"], s["refagg"]
        cats, rcats, hands, rhands = X["categories"], Rf["categories"], X["hands"], Rf["hands"]
        for c in ("fwd", "side", "back", "diag", "turn", "move_turn"):
            if c in cats:
                add(f"survival.{c}", cats[c]["survival"], "survival", c, (rcats.get(c) or {}).get("survival"))
        for c in ("fwd", "side", "back", "diag"):
            if c in cats:
                v = cats[c]["ratio_post"]
                rv = (rcats.get(c) or {}).get("ratio_post")
                add(f"tracking.{c}", None if v is None else abs(v - 1.0), "tracking", c,
                    None if rv is None else abs(rv - 1.0))
        for h in ("lat_l", "lat_r"):
            if h in hands:
                v, rv = hands[h]["ratio_post"], (rhands.get(h) or {}).get("ratio_post")
                add(f"tracking.{h}", None if v is None else abs(v - 1.0), "tracking", "side",
                    None if rv is None else abs(rv - 1.0))
        if "turn" in cats:
            add("turn_fc.turn", cats["turn"].get("fc_post"), "turn_fc", "turn", rcats["turn"].get("fc_post"),
                extra={"n": cats["turn"].get("fc_post_n"), "excluded": cats["turn"].get("fc_post_excluded")})
        for h in ("turn_l90", "turn_r90"):
            if h in hands:
                add(f"turn_fc.{h}", hands[h].get("fc_post"), "turn_fc", "turn", (rhands.get(h) or {}).get("fc_post"),
                    extra={"n": hands[h].get("fc_post_n"), "excluded": hands[h].get("fc_post_excluded")})
        add("h_meas.suite", X["h_meas_post"], "h_meas", "suite", Rf["h_meas_post"])
        add("roll_rms.suite", X["roll_rms_post"], "roll_rms", "suite", Rf["roll_rms_post"])
        add("pitch_rms.suite", X["pitch_rms_post"], "pitch_rms", "suite", Rf["pitch_rms_post"])
    f = row_reads.get("field_light")
    if f is not None:
        X, Rf = f["x"], f["refagg"]
        add("h_meas.field_light", X["h_meas_post"], "h_meas", "field_light", Rf["h_meas_post"])
        add("roll_rms.field_light", X["roll_rms_post"], "roll_rms", "field_light", Rf["roll_rms_post"])
        add("pitch_rms.field_light", X["pitch_rms_post"], "pitch_rms", "field_light", Rf["pitch_rms_post"])
        ln = lines[("field_light_completion", None, None)]
        out.append({"id": "field_light_completion", "value": X["completion"], "line": ln[0], "dir": ln[1],
                    "ref": Rf["completion"], "pass": passes(X["completion"], *ln)})
    return out


def whole_hold_lines(row_reads: dict) -> dict:
    """Report-only whole-hold values beside the post-onset lines."""
    out = {}
    s = row_reads.get("suite")
    if s is not None:
        X = s["x"]
        for c, e in X["categories"].items():
            out[f"ratio.{c}"] = e.get("ratio_whole")
            if c == "turn":
                out["fc.turn"] = e.get("fc_whole")
        for h, e in X["hands"].items():
            if e.get("ratio_whole") is not None:
                out[f"ratio.{h}"] = e["ratio_whole"]
            if "fc_whole" in e:
                out[f"fc.{h}"] = e["fc_whole"]
        out.update({"h_meas.suite": X["h_meas_whole"], "roll_rms.suite": X["roll_rms_whole"],
                    "pitch_rms.suite": X["pitch_rms_whole"]})
    f = row_reads.get("field_light")
    if f is not None:
        X = f["x"]
        out.update({"h_meas.field_light": X["h_meas_whole"], "roll_rms.field_light": X["roll_rms_whole"],
                    "pitch_rms.field_light": X["pitch_rms_whole"]})
    return out


def cell_seed_verdict(row_reads: dict, lines: dict, es: int, transient: bool) -> dict:
    """§5.3 for one cell-seed against ONE reference: worst ΔF (10 waves + field_light), lines, R."""
    dfs = []
    s = row_reads.get("suite")
    if s is not None:
        for w, e in s["waves"].items():
            if e["dF"] is not None:
                dfs.append((e["dF"], f"wave{w}:{e['label']}", e["x"]["alive_at_onset"], e["x"]["F"]))
    f = row_reads.get("field_light")
    if f is not None and f["falls"]["dF"] is not None:
        dfs.append((f["falls"]["dF"], "field_light", f["falls"]["x"]["alive_at_onset"], f["falls"]["x"]["F"]))
    if not dfs:
        return {"verdict": None, "why": "no ΔF"}
    dF, where, n_worst, _ = max(dfs, key=lambda x: x[0])
    F_worst = max((x[3] for x in dfs if x[3] is not None), default=None)
    lns = lines_of(row_reads, lines, es)
    failed = [x["id"] for x in lns if x["pass"] is False]
    lines_ok = not failed
    Rs = []
    for rr in (s, f):
        if rr is None:
            continue
        for key, e in (rr.get("R") or {}).items():
            if not e["report_only"] and e["R"] is not None:
                Rs.append((e["R"], key))
            elif not e["report_only"] and e["R"] is None:
                Rs.append((0.0, key))              # every qualifying faulted episode fell before the window
    R_required = bool(Rs)
    R_worst = min(Rs) if Rs else None
    v = verdict(dF, lines_ok, None if R_worst is None else R_worst[0], R_required, transient)
    return {"verdict": v, "dF_worst": dF, "dF_where": where, "dF_n": n_worst, "F_worst": F_worst,
            "marginal": marginal(dF, n_worst, transient), "lines_failed": failed, "lines": lns,
            "R_worst": None if R_worst is None else {"R": R_worst[0], "where": R_worst[1]},
            "R_required": R_required, "whole_hold": whole_hold_lines(row_reads)}


# ====================================================================================== the campaign read
class Campaign:
    def __init__(self, root: Path, reader: Reader, launches: list | None = None):
        self.root = Path(root)
        self.reader = reader
        self.cells = reader.cells
        if launches is None:
            launches, self.incomplete = scan(self.root)
        else:
            self.incomplete = []
        self.launches = launches
        self.slots = collections.defaultdict(list)
        for L in launches:
            self.slots[L.slot].append(L)
        self._classes = {}

    def cell_of(self, L: Launch) -> dict | None:
        return self.cells["cells"].get(L.cell) if L.cell else None

    # ---------------------------------------------------------------- onset table of a slot
    def slot_onsets(self, slot) -> tuple[np.ndarray | None, list]:
        """The slot's (waves, N) onset table: every same-slot sidecar must agree; none -> onset_table()."""
        errs, tabs = [], []
        for L in self.slots[slot]:
            if L.sidecar is not None and not L.sidecar.get("synthetic"):
                tabs.append((L, sidecar_onsets(L.sidecar)))
        for L, t in tabs[1:]:
            if not np.array_equal(t, tabs[0][1]):
                errs.append(f"onset table of {L} differs from {tabs[0][0]}")
        if tabs:
            return tabs[0][1], errs
        any_l = self.slots[slot][0]
        lo, hi = (float(x) for x in str(self.cells["rows"][any_l.row]["onset"]).split(":"))
        return onset_table_np(lo, hi, any_l.num_envs, any_l.seed, any_l.waves), errs

    # ---------------------------------------------------------------- classes
    def classes(self, slot) -> dict:
        """{stamp: class label or None (anchor) / 'unmatched'} by bitwise pre-onset equality (CHARTER §5.1)."""
        if slot in self._classes:
            return self._classes[slot]
        fps = {}
        for L in self.slots[slot]:
            cell = self.cell_of(L)
            if cell is not None and is_anchor(cell):
                continue
            fps[str(L.stamp)] = self.reader.class_fingerprint(L)
        groups = collections.defaultdict(list)
        for st, fp in fps.items():
            groups[fp].append(st)
        order = sorted(groups.values(), key=lambda g: min(g))
        out, k = {}, 0
        for g in order:
            if len(g) == 1:
                out[g[0]] = "unmatched"
            else:
                lab = chr(ord("A") + k)
                k += 1
                for st in g:
                    out[st] = lab
        # a singleton nominal still names a class (it is a reference); faulted singletons stay 'unmatched'
        for L in self.slots[slot]:
            if out.get(str(L.stamp)) == "unmatched" and L.cell in NOMINAL_CELLS:
                out[str(L.stamp)] = f"N{k}"
                k += 1
        self._classes[slot] = out
        return out

    def nominals(self, slot) -> list:
        return sorted([L for L in self.slots[slot] if L.cell in NOMINAL_CELLS], key=lambda L: L.stamp.name)

    def references(self, L: Launch) -> tuple[list, bool]:
        """(references, same_class): the earliest same-class nominal; else every nominal (one per class first)."""
        noms = [N for N in self.nominals(L.slot) if N.stamp != L.stamp]
        cell = self.cell_of(L)
        cls = self.classes(L.slot)
        mine = cls.get(str(L.stamp))
        if cell is not None and not is_anchor(cell) and mine not in (None, "unmatched"):
            same = [N for N in noms if cls.get(str(N.stamp)) == mine]
            if same:
                return [same[0]], True
        by_cls, rest = {}, []
        for N in noms:
            c = cls.get(str(N.stamp))
            if c not in by_cls:
                by_cls[c] = N
            else:
                rest.append(N)
        refs = list(by_cls.values()) + rest
        return refs[:2], False

    # ---------------------------------------------------------------- launch checks
    def check_launch(self, L: Launch) -> dict:
        """§6.2 launch checks of one stamp: sidecar, pre-onset identity, npz doses, class."""
        out = {"stamp": str(L.stamp), "cell": L.cell, "row": L.row, "seed": L.seed, "payload": L.payload,
               "policy": L.policy, "errors": [], "warnings": []}
        cell = self.cell_of(L)
        if cell is None:
            out["errors"].append(f"unknown cell (env file {L.env_file!r})")
            return out
        onsets, terr = self.slot_onsets(L.slot)
        out["errors"].extend(terr)
        sc = L.sidecar
        row = self.cells["rows"].get(L.row)
        window = tuple(float(x) for x in str(row["onset"]).split(":")) if row else None
        if is_armed(cell):
            if sc is None:
                out["errors"].append("missing sidecar on an armed cell (launch error)")
            else:
                if cell.get("smoke"):
                    window = tuple(float(x) for x in str(cell["onset"]).split(":"))
                chk = sidecar_checks(L, cell, sc, L.waves, L.num_envs, L.seed, window)
                out["errors"].extend(f"sidecar: {e}" for e in chk["errors"])
                out["sidecar_report"] = chk["report"]
                if cell.get("smoke"):
                    onsets = sidecar_onsets(sc)
        elif sc is not None and not is_armed(cell):
            out["errors"].append("sidecar present on an unarmed cell")
        # class
        if not is_anchor(cell):
            cls = self.classes(L.slot).get(str(L.stamp))
            out["class"] = cls
            if cls == "unmatched":
                n_other = len([M for M in self.slots[L.slot] if M.stamp != L.stamp
                               and not is_anchor(self.cell_of(M) or {})])
                out["class_unmatched"] = n_other > 0
                if n_other > 0:
                    out["warnings"].append(f"pre-onset part matches none of the {n_other} other launches of its slot")
        # identity + doses vs the same-class reference
        refs, same = self.references(L)
        out["reference"] = [str(R.stamp) for R in refs]
        out["same_class_reference"] = same
        diffs = None
        drift_idx = set()                       # physics-drift + batch-coupled envs: out of the pairing checks
        if L.cell not in NOMINAL_CELLS and refs and not is_anchor(cell):
            if same:
                diffs = self.reader.diffs(L, refs[0])
                # armed-but-never row: its own onsets lie beyond the episode, so the identity horizon is the
                # row's latest grid onset (CHARTER 5.1, 2026-11-16 amendment)
                hz = grid_onset_horizon(row) if cell.get("smoke") and row else None
                idr = identity_read(L, diffs, onsets, gated_waves=(0,), horizon=hz)
                out["identity"] = {k: idr[k] for k in ("ok", "n_breaks", "breaks", "n_physics_drift", "drift",
                                                       "n_batch_coupled", "coupled", "first_onsets", "horizon")}
                out["divergence_per_wave"] = idr["per_wave"]
                drift_idx = set(idr["drift_idx"]) | set(idr["coupled_idx"])
                if idr["n_breaks"]:
                    out["errors"].append(f"pre-onset identity broken on {idr['n_breaks']} paired envs vs the "
                                         "same-class reference (pairing premise: stop and amend, never relaunch)")
                if idr["n_physics_drift"] > MAX_PHYSICS_DRIFT:
                    out["errors"].append(f"{idr['n_physics_drift']} physics-drift envs (> {MAX_PHYSICS_DRIFT}: "
                                         "pre-onset divergence with equal actions) vs the same-class reference")
                elif idr["n_physics_drift"]:
                    out["warnings"].append(f"{idr['n_physics_drift']} physics-drift env(s) (pre-onset divergence, "
                                           "actions equal through the first differing step): kept in net counts, "
                                           "excluded from per-episode pairing")
                if idr["n_batch_coupled"]:
                    pw = "; ".join(f"wave {w}: {v['n_coupled']} coupled from step {v['first_coupled_step']} "
                                   f"(wave first onset {v['first_onset']})"
                                   for w, v in sorted(idr["per_wave"].items()) if v["n_coupled"])
                    out["warnings"].append(f"{idr['n_batch_coupled']} batch-coupled env(s) (physics-first divergence "
                                           f"after the wave's first onset, before their own): {pw}; excluded from "
                                           "per-episode pairing")
            else:
                out["warnings"].append("no same-class nominal: identity not checked (read against both nominals)")
        if sc is not None and is_armed(cell) and not cell.get("smoke"):
            dz = npz_dose_checks(self.reader, L, cell, sc, onsets, diffs, exclude=drift_idx)
            out["errors"].extend(f"dose: {e}" for e in dz["errors"])
            out["warnings"].extend(f"dose: {e}" for e in dz.get("warnings", []))
            out["dose_report"] = dz["report"]
        return out

    # ---------------------------------------------------------------- reading a cell-seed
    def clear_of(self, L: Launch, cell: dict, onsets: np.ndarray):
        """(clear table or None, permanent) for the R windows."""
        d = cell_duration_steps(cell)
        if d is None or cell.get("from_start") or is_anchor(cell):
            return None, True
        sc = L.sidecar
        fi = fault_inject()
        if sc is not None and not sc.get("synthetic"):
            return sidecar_clear(sc, str(sc.get("kind")) in fi.DEPTH_KINDS), False
        return onsets + d, False

    def read_cell_seed(self, cell_id: str, policy: str, seed: int, payload: str) -> dict:
        cell = self.cells["cells"][cell_id]
        rows = {}
        for L in self.launches:
            if L.cell == cell_id and L.policy == policy and L.seed == seed and L.payload == payload:
                rows.setdefault(L.row, []).append(L)
        transient = cell_duration_steps(cell) is not None and not cell.get("from_start")
        res = {"cell": cell_id, "policy": policy, "seed": seed, "payload": payload, "family": cell.get("family"),
               "transient": transient, "rows": {}, "checks": {}}
        for row, Ls in rows.items():
            L = sorted(Ls, key=lambda x: x.stamp.name)[-1]            # the latest launch of the row
            if len(Ls) > 1:
                res.setdefault("relaunched_rows", {})[row] = [str(x.stamp) for x in Ls]
            chk = self.check_launch(L)
            res["checks"][row] = chk
            onsets, _ = self.slot_onsets(L.slot)
            clear, perm = self.clear_of(L, cell, onsets)
            refs, same = self.references(L)
            if not refs:
                res["rows"][row] = {"error": "no nominal in the slot"}
                continue
            res["rows"][row] = {"same_class_reference": same, "reads": []}
            for R in refs:
                res["rows"][row]["reads"].append(read_row(self.reader, L, R, onsets, clear, perm))
        main_rows = ("suite", "field_light")
        if all("reads" in (res["rows"].get(r) or {}) for r in main_rows):
            # one verdict per reference (a row with one same-class reference reuses it); the worse decides
            nref = max(len(res["rows"][r]["reads"]) for r in main_rows)
            vs = []
            for k in range(nref):
                rr = {r: res["rows"][r]["reads"][min(k, len(res["rows"][r]["reads"]) - 1)] for r in main_rows}
                vs.append(cell_seed_verdict(rr, self.reader.lines, seed, transient))
            worst = max(vs, key=lambda v: (SEVERITY.get(v["verdict"], -1), v["dF_worst"]))
            res["verdict_reads"] = vs
            res["verdict"] = worst["verdict"]
            res["read"] = worst
            if any(c["errors"] for c in res["checks"].values()):
                res["verdict"] = None                 # a launch error is never a reading (CHARTER §6.2, §8)
                res["withheld"] = "launch error"
        else:
            res["verdict"] = None
            res["pending_rows"] = sorted(set(main_rows) - set(res["rows"]))
        res["launch_errors"] = {r: c["errors"] for r, c in res["checks"].items() if c["errors"]}
        return res

    def read_all(self) -> dict:
        keys = sorted({(L.cell, L.policy, L.seed, L.payload) for L in self.launches
                       if L.cell and L.cell not in NOMINAL_CELLS and L.cell in self.cells["cells"]
                       and not self.cells["cells"][L.cell].get("smoke")})
        unknown = sorted({str(L.stamp) for L in self.launches if not L.cell or L.cell not in self.cells["cells"]})
        seeds = [self.read_cell_seed(*k) for k in keys]
        nominal_checks = [self.check_launch(L) for L in self.launches if L.cell in NOMINAL_CELLS]
        smoke_checks = [self.check_launch(L) for L in self.launches
                        if L.cell in self.cells["cells"] and self.cells["cells"][L.cell].get("smoke")]
        cells = summarize_cells(seeds, self.cells)
        nn = self.nominal_pairs()
        return {"schema": SCHEMA, "root": str(self.root), "n_launches": len(self.launches),
                "incomplete_stamps": self.incomplete, "unknown_cell_stamps": unknown,
                "classes": {f"{s[0]}/{s[1]}/es{s[2]}/{s[3]}": self.classes(s) for s in self.slots},
                "nominal_checks": nominal_checks, "smoke_checks": smoke_checks, "nominal_pairs": nn,
                "cell_seeds": seeds, "cells": cells["cells"], "tolerated_durations": cells["durations"]}

    def nominal_pairs(self) -> dict:
        """Nominal-vs-nominal divergence counts per slot (the baseline beside the faulted waves 1-9 counts)."""
        out = {}
        for slot in self.slots:
            noms = self.nominals(slot)
            if len(noms) < 2:
                continue
            onsets, _ = self.slot_onsets(slot)
            d = self.reader.diffs(noms[0], noms[1])
            idr = identity_read(noms[0], d, onsets, gated_waves=())
            out[f"{slot[0]}/{slot[1]}/es{slot[2]}/{slot[3]}"] = {"a": str(noms[0].stamp), "b": str(noms[1].stamp),
                                                                 "per_wave": idr["per_wave"]}
        return out


def summarize_cells(seeds: list, cells: dict) -> dict:
    by = collections.defaultdict(dict)
    for s in seeds:
        by[(s["cell"], s["policy"], s["payload"])][s["seed"]] = s
    out = {}
    for (cid, pol, pay), ss in sorted(by.items()):
        cell = cells["cells"][cid]
        per = {k: v.get("verdict") for k, v in ss.items()}
        es1 = ss.get(1)
        F1 = (es1.get("read") or {}).get("F_worst") if es1 else None
        comb = combine_seeds(per, F1)
        report_only = bool(cell.get("report_only")) or pol != "C1S"
        out[f"{cid}/{pol}/{pay}"] = {"cell": cid, "policy": pol, "payload": pay, "family": cell.get("family"),
                                     "per_seed": per, **comb, "report_only": report_only,
                                     "marginal": sorted({m for v in ss.values() for m in
                                                         ((v.get("read") or {}).get("marginal") or [])}),
                                     "launch_errors": {k: v.get("launch_errors") for k, v in ss.items()
                                                       if v.get("launch_errors")}}
    fams = collections.defaultdict(list)
    for c in out.values():
        cell = cells["cells"][c["cell"]]
        d = cell_duration_steps(cell)
        if d is not None and not cell.get("from_start") and c["policy"] == "C1S":
            fams[(cell["family"], c["policy"])].append((d, c["verdict"], c["pending"]))
    durations = {f"{f}/{p}": tolerated_duration(lv) for (f, p), lv in sorted(fams.items())}
    return {"cells": out, "durations": durations}


# ====================================================================================== null-fault mode (§6.1.5)
def null_fault(reader: Reader, pairs: list, sidecars: list, cell_id: str | None) -> dict:
    """Read nominal A as if faulted against nominal B on a sidecar's onsets; group by seed into cell-seeds."""
    cells = reader.cells
    by_seed = collections.defaultdict(dict)
    meta = []
    for (a, b), scs in zip(pairs, sidecars):
        A, B = Launch(Path(a), cell="NULL"), Launch(Path(b), cell="NOM")
        if A.row != B.row or A.seed != B.seed or A.payload != B.payload:
            raise SystemExit(f"null-fault pair is not one slot: {A} vs {B}")
        if scs.startswith("synth:"):
            cid = scs.split(":", 1)[1]
            sc = synth_sidecar(cells, cid, A.row, A.num_envs, A.seed, A.waves)
        else:
            sc = json.loads(Path(scs).read_text())
            cid = cell_id or "S1_perm"
        if cid not in cells["cells"]:
            raise SystemExit(f"unknown cell {cid!r}")
        onsets = sidecar_onsets(sc)
        if onsets.shape != (A.waves, A.num_envs):
            onsets = onsets[: A.waves]
        fi = fault_inject()
        clear = sidecar_clear(sc, str(sc.get("kind")) in fi.DEPTH_KINDS)
        perm = clear is None
        rd = read_row(reader, A, B, onsets, clear, perm)
        diffs = reader.diffs(A, B)
        idr = identity_read(A, diffs, onsets, gated_waves=(0,))
        rd["identity"] = {k: idr[k] for k in ("ok", "n_breaks", "breaks", "n_physics_drift", "drift",
                                                  "n_batch_coupled")}
        rd["divergence_per_wave"] = idr["per_wave"]
        rd["as_cell"] = cid
        rd["transient"] = not perm
        by_seed[(A.seed, cid)][A.row] = rd
        meta.append({"a": str(A.stamp), "b": str(B.stamp), "row": A.row, "seed": A.seed, "sidecar": scs,
                     "as_cell": cid, "same_stamp": A.stamp == B.stamp})
    seeds = []
    for (seed, cid), rows in sorted(by_seed.items()):
        transient = any(r["transient"] for r in rows.values())
        v = cell_seed_verdict(rows, reader.lines, seed, transient) if {"suite", "field_light"} <= set(rows) else None
        seeds.append({"seed": seed, "as_cell": cid, "rows": rows, "read": v,
                      "verdict": None if v is None else v["verdict"]})
    return {"schema": SCHEMA, "mode": "null_fault", "pairs": meta, "cell_seeds": seeds}


# ====================================================================================== outputs
def _f(x, nd=3):
    if x is None:
        return "–"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def verdicts_md(doc: dict) -> str:
    out = ["# Fault-injection sweep — verdicts (fault_read.py)", "",
           f"Root `{doc.get('root')}`; {doc.get('n_launches')} launches; schema {doc['schema']}. Every A read: "
           "sim plant, force side not enforced. c1, D1 and kind rows: report only.", ""]
    out += ["## Cells", "", "| cell | policy | payload | es1 | es2 | verdict | split | marginal | report only | launch errors |",
            "|---|---|---|---|---|---|---|---|---|---|"]
    for c in (doc.get("cells") or {}).values():
        out.append(f"| {c['cell']} | {c['policy']} | {c['payload']} | {_f(c['per_seed'].get(1))} | "
                   f"{_f(c['per_seed'].get(2))} | {_f(c['verdict'])}{' (pending)' if c.get('pending') else ''} | "
                   f"{'split' if c.get('split') else ''} | {', '.join(f'{m:g}' for m in c.get('marginal') or [])} | "
                   f"{'yes' if c.get('report_only') else ''} | {'YES' if c.get('launch_errors') else ''} |")
    out += ["", "## Cell-seeds", "",
            "| cell | es | policy | payload | ΔF worst (where, n) | F worst | F2 | pre-onset falls | hazard /min | "
            "R worst (where) | lines failed | verdict |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for s in doc.get("cell_seeds") or []:
        rd = s.get("read") or {}
        f2 = pre = hz = None
        rows = s.get("rows") or {}
        for row in ("suite", "field_light"):
            rr = rows.get(row) or {}
            r0 = (rr.get("reads") or [None])[0] if "reads" in rr else (rr if "x" in rr else None)
            if r0 is None:
                continue
            fx = (r0.get("falls_total") or r0.get("falls") or {}).get("x") or {}
            f2 = (f2 or 0) + fx.get("f2", 0)
            pre = (pre or 0) + fx.get("pre_falls", 0)
            if fx.get("hazard_per_min") is not None:
                hz = max(hz or 0.0, fx["hazard_per_min"])
        Rw = rd.get("R_worst") or {}
        out.append(f"| {s.get('cell', s.get('as_cell'))} | {s['seed']} | {s.get('policy', 'C1S')} | "
                   f"{s.get('payload', '')} | {_f(rd.get('dF_worst'))} ({rd.get('dF_where', '')}, {rd.get('dF_n', '')}) | "
                   f"{_f(rd.get('F_worst'))} | {_f(f2)} | {_f(pre)} | {_f(hz)} | "
                   f"{_f(Rw.get('R'))} ({Rw.get('where', '')}) | {', '.join(rd.get('lines_failed') or []) or '–'} | "
                   f"{_f(s.get('verdict'))} |")
    durs = doc.get("tolerated_durations") or {}
    if durs:
        out += ["", "## Tolerated duration (transient families, C1S)", "", "| family | tolerated d | stopped by |",
                "|---|---|---|"]
        for f, d in durs.items():
            out.append(f"| {f} | {_f(d['tolerated_s'], 2)} s ({_f(d['tolerated_steps'])} steps) | {_f(d['stopped_by'])} |")
    kinds = [(s, row, rd) for s in doc.get("cell_seeds") or [] for row, rr in (s.get("rows") or {}).items()
             if row in KIND_ROWS for rd in (rr.get("reads") or [])[:1]]
    if kinds:
        out += ["", "## Kind rows (report only)", "", "| cell | es | row | success | nominal | Δsuccess | F | F ref |",
                "|---|---|---|---|---|---|---|---|"]
        for s, row, rd in kinds:
            k = rd.get("kind") or {}
            fl = rd.get("falls") or {}
            out.append(f"| {s['cell']} | {s['seed']} | {row} | {_f(k.get('success_x'))} | {_f(k.get('success_ref'))} | "
                       f"{_f(k.get('d_success'))} | {_f((fl.get('x') or {}).get('F'))} | {_f((fl.get('ref') or {}).get('F'))} |")
    errs = [(s["cell"], s["seed"], row, e) for s in doc.get("cell_seeds") or []
            for row, es in (s.get("launch_errors") or {}).items() for e in es]
    errs += [(c.get("cell"), c.get("seed"), c.get("row"), e) for c in (doc.get("nominal_checks") or [])
             + (doc.get("smoke_checks") or []) for e in c.get("errors") or []]
    out += ["", "## Launch errors", ""]
    out += [f"- {c} es{s} {r}: {e}" for c, s, r, e in errs] or ["- none"]
    return "\n".join(out) + "\n"


def null_md(doc: dict) -> str:
    out = ["# Null-fault read (fault_read.py --null-fault)", "",
           "| as cell | es | ΔF worst (where) | R worst (where) | lines failed | identity | verdict |",
           "|---|---|---|---|---|---|---|"]
    for s in doc["cell_seeds"]:
        rd = s.get("read") or {}
        idok = all(r["identity"]["ok"] for r in s["rows"].values())
        Rw = rd.get("R_worst") or {}
        out.append(f"| {s['as_cell']} | {s['seed']} | {_f(rd.get('dF_worst'))} ({rd.get('dF_where', '')}) | "
                   f"{_f(Rw.get('R'))} ({Rw.get('where', '')}) | {', '.join(rd.get('lines_failed') or []) or '–'} | "
                   f"{'ok' if idok else 'BROKEN'} | {_f(s.get('verdict'))} |")
    return "\n".join(out) + "\n"


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    return o


# ====================================================================================== CLI
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--cells", type=Path, default=CELLS_YAML)
    ap.add_argument("--thresholds", type=Path, default=THRESHOLDS_JSON)
    ap.add_argument("--check-only", nargs="+", type=Path, metavar="STAMP")
    ap.add_argument("--null-fault", nargs=2, action="append", metavar=("NOMINAL_A", "NOMINAL_B"))
    ap.add_argument("--sidecar", action="append", default=[], help="fault_onsets.json or synth:<cell>, one per pair")
    ap.add_argument("--cell", default=None, help="null-fault: the cell a real sidecar is read as (its d)")
    ap.add_argument("--no-write", action="store_true")
    a = ap.parse_args(argv)
    reader = Reader(load_cells(a.cells), load_lines(a.thresholds), jobs=a.jobs)
    try:
        if a.null_fault:
            if len(a.sidecar) not in (1, len(a.null_fault)):
                print("one --sidecar per --null-fault pair (or one for all)", file=sys.stderr)
                return 4
            scs = a.sidecar * len(a.null_fault) if len(a.sidecar) == 1 else a.sidecar
            doc = null_fault(reader, a.null_fault, scs, a.cell)
            if not a.no_write:
                a.out_dir.mkdir(parents=True, exist_ok=True)
                (a.out_dir / "null_read.json").write_text(json.dumps(_jsonable(doc), indent=1) + "\n")
                (a.out_dir / "null_verdicts.md").write_text(null_md(doc))
            print(null_md(doc))
            return 0
        if a.check_only:
            camp = Campaign(a.root, reader)
            known = {str(L.stamp.resolve()): L for L in camp.launches}
            rc, outs = 0, []
            for st in a.check_only:
                L = known.get(str(Path(st).resolve()))
                if L is None:
                    L = Launch(Path(st))
                    camp.launches.append(L)
                    camp.slots[L.slot].append(L)
                    camp._classes.pop(L.slot, None)
                chk = camp.check_launch(L)
                outs.append(chk)
                if chk["errors"]:
                    rc = 2
                elif chk.get("class_unmatched") and rc == 0:
                    rc = 3
            print(json.dumps(_jsonable(outs), indent=1))
            for c in outs:
                print(f"FAULTCHECK {c['cell']} {c['row']} es{c['seed']} {c['payload']} errors={len(c['errors'])} "
                      f"class={c.get('class')} physics_drift={(c.get('identity') or {}).get('n_physics_drift', 0)} "
                      f"batch_coupled={(c.get('identity') or {}).get('n_batch_coupled', 0)} "
                      f"stamp={c['stamp']}")
            print(f"FAULTCHECK END rc={rc}")
            return rc
        camp = Campaign(a.root, reader)
        doc = camp.read_all()
        if not a.no_write:
            a.out_dir.mkdir(parents=True, exist_ok=True)
            (a.out_dir / "read.json").write_text(json.dumps(_jsonable(doc), indent=1) + "\n")
            (a.out_dir / "verdicts.md").write_text(verdicts_md(doc))
        print(verdicts_md(doc))
        return 0
    finally:
        reader.close()


if __name__ == "__main__":
    sys.exit(main())
