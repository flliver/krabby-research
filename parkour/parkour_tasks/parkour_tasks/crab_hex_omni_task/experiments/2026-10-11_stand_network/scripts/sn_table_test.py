#!/usr/bin/env python3
"""Stand-network campaign -- TABLE STAND-UP TEST (2026-10-13; the user's "can the stand network stand up from a
position on a table"). A TEST, not a gate: it writes a summary JSON, an npz of env 0's traces and (``--video``) an mp4;
no ledger, no reader, no campaign verdict reads them.

LAUNCH THROUGH ``sn_table_test.sh`` (one Isaac process at a time, MemoryMax 24G, detached; ``--smoke`` / full run).
Kit-free modes (CPU, no Isaac)::

    sn_table_test.py --emit-facts --out DIR            # facts.json: the retracted poses, spawn z, clearances, R at start
    sn_table_test.py --merge DIR                       # DIR/table_test_summary.json from DIR/*_summary.json

The Isaac run (``--headless``; ``--enable_cameras`` with ``--video``)::

    sn_table_test.py --controller {stand_net,r,s1,zero} --pose {rod_min,knee_folded} --num-envs N [--jitter]
                     [--seed 1] [--episode-s 15] [--video] --out DIR [--prefix P] [--smoke] --headless ...

THE SETUP (every number below is computed by ``sn_table_core`` and written into the run's summary under ``facts``)

* ENV = the S1 eval env: ``Isaac-Crab-Hex-Omni-Walk-Play-v0`` with every ``KRABBY_*`` replaced BEFORE Kit boots by the
  REFS_S1_s18 suite es1 stamp's ``run_meta.env_vars`` (``--eval-run``; STAND_OBS 0, CLOCK_HIDE 1, STD_PARAM soft,
  INIT_STD 0.8, ellipse 0.35:0.35:0.25, FACING_OBS_CLIP 0.6, plant A15+B, ring geometry with every tile the flat
  sub-terrain = flat ground with the eval's +-2-3 cm roughness), re-checked after the task import and against
  ``sn_common.env_reasons`` (D5); the harness's own ``_apply_env_overrides`` (ast-extracted, never copied: seed, episode
  length = ``--episode-s`` + 2 s, resampling off, curriculum off, frozen terrain levels, pushes off, spawn yaw pinned 0).
* TABLE: one KINEMATIC cuboid per env (``{ENV_REGEX_NS}/SnTable``), 0.90 m (body x) x 0.70 m (lateral), top at
  0.635 m = 25 in, base sunk to -0.05 m, the terrain's surface material; the legs mount on the 1.22-m-wide body's side
  faces (yaw axes |y| = 0.6096), so the table (|y| <= 0.35; <= 0.45 at a jittered corner) blocks no leg part
  (>= 0.15 m lateral clearance). ``--jitter``: per-env table offset U(+-3 cm) x U(+-3 cm) x U(+-10 deg), seeded by
  ``--seed`` (numpy default_rng, row-major, env 0 always nominal) -- the TABLE moves, the robot stays at its env
  origin at yaw 0, so every env runs the same face-0 stand row.
* START = a reset event appended LAST (``sn_table_core.sn_table_start``): the robot's root at z 0.79875 m (belly
  5 mm above the table top: root = table top 0.635 + body half height 0.15875 + 0.005), yaw 0, zero velocity, joints
  at the RETRACTED pose (``--pose``; Body_Hip / CamShaft 0) with matching joint targets; the action term's reset then
  flags its rod state for a re-sync from these joint angles on the first substep, so the rods START at the retracted
  lengths (no snap) and every later command slews them at the loaded rod speed.
  - ``rod_min`` (default): BOTH lead screws at their minimum length. Hip: the screw's minimum is -48.7 deg, past the
    -45 deg USD hard stop, so the stop binds: hip -44.5 deg (0.5 deg margin; rod 0.4572 m). Knee: the screw's minimum
    at that hip is -41.6 deg (left; right legs +41.6) -- it OPENS the knee: tibias near-horizontal OUTBOARD, toes at
    |y| ~1.9 m, ~0.11 m above the body centre (0.89 m above the floor, above the table top).
  - ``knee_folded``: the hardware's "retracted" words (crab_hex_dimensions: hip 45 deg from up = raised / retracted,
    knee interior 5 deg = retracted / folded): hip at the same stop, knee folded INBOARD to +77.6 deg (interior 12.4
    deg; knee screw 0.572 m, i.e. EXTENDED past mid-stroke) -- the most the sim model allows: its hip plate, femur and
    tibia are coplanar and the tibia would otherwise spawn inside the hip plate (>= 2 cm hull clearance kept). Toes
    hang at |y| 0.64 m, 0.52 m below the body centre: below the table top (pad bottom 0.26 m above the floor), outside
    its footprint.
* STAND COMMAND FROM t = 0: the harness's ``_install_schedule_override_world`` (ast-extracted) drives the command term
  through its own ``schedule_hook`` with a constant row speed 0 / dir 0 / face 0 (= the pinned spawn yaw), applied once
  before the first observation; the eval slot latch is armed at runtime (P2L ``arm_slot_latch``) and the turning bit
  re-derived on reset rows every step (P2L ``P2RMode.rederive_on_reset``), so the latched slot reads STAND from the
  first step for every env (checked, and the per-step slot share is recorded). ``KRABBY_OMNI_STAND_HOLD`` stays 0 (as
  REFS): a yaw drift past the 0.133-rad dead band would put a live yaw-rate command in the obs; the slot stays STAND.
* CONTROLLERS (``sn_table_core.controller_actions``; S1 and its estimator built WITHOUT a PPO runner by
  ``sn_dagger_core.load_strict``, the stand net the same way, both inside a global-RNG save / restore):
  ``stand_net`` = the adopted A-mask s17 net (``policy/stand_net/model_4000.pt``, sha 769cf253..; its
  ``infos.sn_dagger.stand_mask`` must equal ``--stand-mask`` 1, its estimator must equal S1's bitwise, it must have
  been trained beside this S1 sha) on ``sn_common.stand_net_input(h, own estimator, mask=1)``, routed with
  ``sn_common.route_actions`` on the latched stand slot, S1 (``policy/6_p2a_field/model_19994.pt``) elsewhere --
  bitwise ``sn_dagger_core.launcher_actions``; ``r`` = R's target (``sn_common.r_label``, the D3 string) on stand rows;
  ``s1`` = S1 on every row; ``zero`` = zero action. The wrapper clips every action at 1.0 (deployment).
* WHAT WAS CHANGED vs the eval env (each recorded in the summary's ``env_changes``, with the event / termination term
  lists before and after): terminations -> ONE time-out term (``total_terminates`` = ``isaaclab.envs.mdp.time_out``,
  sized 2 s past the episode; ``crab_failure`` (tilt > 0.5 rad, any hip link > 500 N) and the roll / pitch > 1.5 rad /
  root z < -0.25 cutoffs off -- note the belly-on-table contact is NOT one of them: crab_failure watches the hip links
  only); events -> ``reset_robot_joints`` off (0.95-1.05 x default scale; the table event writes the joints),
  ``physics_material`` pinned to ONE friction (``--friction``, default the configured range's midpoint) instead of a
  per-env draw, mass / CoM / camera / push / RSI terms off, ``reset_root_state`` kept (ring spawn, offset 0, yaw
  pinned 0: the parkour bookkeeping) and overridden by the table event; scene -> the table added; viewer (``--video``)
  -> a static env-0 camera.
* NOT CHANGED, AND IT MATTERS: the height scanner ray-casts ``/World/ground`` only -- the table is invisible to it; the
  policy's scan reads the floor ~0.80 m below the root at the start (1.06 m standing): the body looks ~0.26 m LOW.

STOOD (fixed before any run; ``sn_table_core.STOOD_DEFINITION``): an env STOOD iff (1) over the last 3.0 s (150 control
steps): on EVERY step the chassis box's lowest point >= 0.10 m above the table-top plane, AND on >= 95 % of the steps
(>= 143 of 150) >= 5 of 6 footpads loaded on the floor (net contact >= 20 N, footpad centre z <= 0.15 m) -- the
tolerance absorbs post-settle load flickers on a standing robot; (2) whole episode: body tilt <= 0.5 rad (else FALL);
(3) whole episode: the chassis' lowest point never more than 0.05 m below the table-top plane (else FELL OFF THE
TABLE); (4) no termination. Reported, not gated: ``stood_strict`` (>= 5 feet on EVERY window step), the window share of
steps with >= 5 feet and the window's min feet (merge-table columns ``strict`` / ``w5 share`` / ``w min``), lift-off
time (belly gap > 0.02 m held 0.5 s), final root z / gap / feet, max tilt, cams homed at the end (worst |cam| <= 0.10
rad), falls, fell-off-table.

HOW THIS DIFFERS FROM EVERYTHING THE STAND NET SAW (trained: Teacher-v0 2a + the P2__S1_s18 overlay, wall field,
pushes, mass / CoM DR, 40-s episodes, stands entered from S1's walking / spawned standing states; evaluated: suite /
walkstop / field rows from standing or walking states, Walk-Play REFS env): it never started (a) from joint angles
outside the +-0.25-rad action window (here hip ~0.80 rad below default, knee 0.73-1.05 rad off), (b) with the body
SUPPORTED on its belly and no foot load, (c) 0.26 m below standing height, (d) with a support surface its scan cannot
see, (e) with a time-out-only env (falls are recorded, not reset), (f) with one pinned friction. R (its teacher) from
this start: every hip reads far below ``p_down`` -> all six legs SEAT -> R's action is exactly ZERO (the default
targets, cams 0) until a hip comes within 0.05 rad of its default (then HOLD: cam trim only); R never LIFTs here (that
needs a DOWN leg with |cam| > 0.3). And NO controller can hold the retraction: the action window is default +-0.25 rad,
so every action slews every rod toward the window at the rod speed (kinematic zero-action timeline: toes reach the
floor ~1.6 s (rod_min) / ~1.1 s (knee_folded), hips arrive at the default ~3.3 s, standing belly gap ~0.26 m).
"""
from __future__ import annotations

import __future__ as _future
import argparse
import ast
import copy
import hashlib
import importlib.util
import json
import math
import os
import re
import resource
import signal
import sys
import time
import types
from pathlib import Path

_THIS = Path(__file__).resolve()
HERE = _THIS.parent
CAMPAIGN = HERE.parent
EXPERIMENTS = HERE.parents[1]
PKG = HERE.parents[2]                                     # crab_hex_omni_task
REPO = HERE.parents[6]
PARKOUR = REPO / "parkour"
HARNESS = PKG.parent / "crab_hex_forward_task" / "scripts" / "eval_crab_hex_gait.py"
HARNESS_DIR = HARNESS.parent
P2R_LAUNCHER = EXPERIMENTS / "2026-10-07_omni_phase2" / "scripts" / "p2r_mode_eval_crab_hex_gait.py"
TASK_ID = "Isaac-Crab-Hex-Omni-Walk-Play-v0"
DEFAULT_S1 = PKG / "policy" / "6_p2a_field" / "model_19994.pt"
DEFAULT_STAND = PKG / "policy" / "stand_net" / "model_4000.pt"
#: policy/manifest.yaml stand_network.sha256 (the adopted A-mask s17)
STAND_SHA256 = "769cf25331402afdfb23756100ec7c215dacf984a321e90a7b39d2969e4941ff"
DEFAULT_OUT_BASE = PARKOUR / "logs" / "videos" / "sn_table"
REFUSED_RC = 2
_PREFIX_RE = re.compile(r"^[A-Za-z0-9_.-]+$")


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    old = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.dont_write_bytecode = old
    return mod


#: torch-free at import, all four (rule L6): loaded before Kit
CORE = _load(HERE / "sn_table_core.py", "_snt_core")
SN = _load(HERE / "sn_common.py", "_snt_common")
SPEC = _load(HERE / "sn_dagger_spec.py", "_snt_spec")
# NOTE(M18 Task 4 pipeline step 6, 2026-10-16; DESIGN B6 "--s1-sha on the table tests"): with SN_WALKER_SPEC set, the
# default walker is the override's (its sha is SPEC.S1_SHA256, the one the --s1-ckpt refusal checks); unset -> the
# bundled S1 exactly, as before.
if SPEC.WALKER_OVERRIDE is not None:
    DEFAULT_S1 = SPEC.S1_CKPT
P2L = SN.load_by_path(P2R_LAUNCHER, "_snt_p2r")


# ===================================================================================== argv
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(allow_abbrev=False, add_help=False)
    ap.add_argument("--controller", choices=CORE.CONTROLLERS, default=None)
    ap.add_argument("--pose", choices=CORE.POSES, default="rod_min")
    ap.add_argument("--num-envs", dest="num_envs", type=int, default=32)
    ap.add_argument("--jitter", action="store_true", default=False)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--episode-s", dest="episode_s", type=float, default=15.0)
    ap.add_argument("--video", action="store_true", default=False)
    ap.add_argument("--out", default=None)
    ap.add_argument("--prefix", default=None)
    ap.add_argument("--overwrite", action="store_true", default=False)
    ap.add_argument("--smoke", action="store_true", default=False)
    ap.add_argument("--eval-run", dest="eval_run", default=str(SPEC.REFS_ENV_STAMP))
    ap.add_argument("--s1-ckpt", dest="s1_ckpt", default=str(DEFAULT_S1))
    ap.add_argument("--stand-ckpt", dest="stand_ckpt", default=str(DEFAULT_STAND))
    ap.add_argument("--stand-mask", dest="stand_mask", type=int, choices=(0, 1), default=1)
    ap.add_argument("--friction", type=float, default=None)
    ap.add_argument("--cam-eye", dest="cam_eye", default="-4.5,-4.5,1.7")
    ap.add_argument("--cam-lookat", dest="cam_lookat", default="0.0,0.0,0.5")
    ap.add_argument("--quality", type=float, default=8.0)
    ap.add_argument("--rss-abort-gb", dest="rss_abort_gb", type=float, default=20.0)
    ap.add_argument("--emit-facts", dest="emit_facts", action="store_true", default=False)
    ap.add_argument("--merge", default=None)
    return ap


_OWN = ("--controller", "--pose", "--num-envs", "--jitter", "--seed", "--episode-s", "--video", "--out", "--prefix",
        "--overwrite", "--smoke", "--eval-run", "--s1-ckpt", "--stand-ckpt", "--stand-mask", "--friction", "--cam-",
        "--quality", "--rss-", "--emit-facts", "--merge")


def parse(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    o, rest = build_parser().parse_known_args(list(argv))
    stray = [a for a in rest if a.startswith(_OWN)]
    if stray:
        raise SystemExit(f"sn_table_test: malformed argument(s) {stray}")
    o.prefix = o.prefix or (None if o.controller is None else f"{o.controller}_{o.pose}" + ("_video" if o.video
                                                                                           else ""))
    return o, rest


def resolve_run_meta(text: str) -> Path:
    """A run_meta.json, a stamp dir, or a seed dir (newest stamp with run_meta + complete.json)."""
    p = Path(text)
    if p.is_file():
        return p
    if (p / "run_meta.json").is_file():
        return p / "run_meta.json"
    if p.is_dir():
        stamps = sorted(d for d in p.iterdir() if (d / "run_meta.json").is_file() and (d / "complete.json").is_file())
        if stamps:
            return stamps[-1] / "run_meta.json"
    raise SystemExit(f"--eval-run {text!r}: no run_meta.json")


def want_env(run_meta: dict) -> dict:
    return {str(k): str(v) for k, v in (run_meta.get("env_vars") or {}).items()}


def pre_kit_reasons(o: argparse.Namespace, rest: list[str]) -> list[str]:
    """Every refusal before Kit (fail closed)."""
    out = []
    if o.controller is None:
        out.append("--controller is required for an Isaac run")
    if o.out is None:
        out.append("--out is required")
    else:
        out += [f"--out: {r}" for r in SPEC.path_reasons(o.out)]
    if o.num_envs < 1:
        out.append("--num-envs must be >= 1")
    if not o.episode_s > 0:
        out.append("--episode-s must be > 0")
    elif not o.smoke and o.episode_s < CORE.STOOD_WINDOW_S:
        out.append(f"--episode-s {o.episode_s} < the {CORE.STOOD_WINDOW_S}-s STOOD window (allowed with --smoke only)")
    if o.prefix is not None and not _PREFIX_RE.match(o.prefix):
        out.append(f"--prefix {o.prefix!r} must be [A-Za-z0-9_.-]+")
    if o.out is not None and o.prefix is not None and not o.overwrite:
        if (Path(o.out) / f"{o.prefix}_summary.json").exists():
            out.append(f"{Path(o.out) / (o.prefix + '_summary.json')} exists (pass --overwrite or a new --prefix)")
    if o.video and "--enable_cameras" not in rest:
        out.append("--video needs --enable_cameras (AppLauncher)")
    if o.friction is not None and not o.friction > 0:
        out.append("--friction must be > 0")
    s1 = Path(o.s1_ckpt)
    if not s1.is_file():
        out.append(f"--s1-ckpt missing: {s1}")
    elif SN.sha256_file(s1) != SPEC.S1_SHA256:
        out.append(f"--s1-ckpt sha256 is not S1 s18's pinned {SPEC.S1_SHA256[:12]}")
    if o.controller == "stand_net":
        st = Path(o.stand_ckpt)
        if not st.is_file():
            out.append(f"--stand-ckpt missing: {st}")
        elif SN.sha256_file(st) != STAND_SHA256:
            out.append(f"--stand-ckpt sha256 is not the adopted stand net's {STAND_SHA256[:12]}")
    try:
        rm = json.loads(resolve_run_meta(o.eval_run).read_text())
        out += [f"eval env (D5): {r}" for r in SN.env_reasons(want_env(rm))]
        if rm.get("task") != TASK_ID:
            out.append(f"eval stamp task {rm.get('task')!r} != {TASK_ID}")
        if rm.get("plant") != "A15+B":
            out.append(f"eval stamp plant {rm.get('plant')!r} != 'A15+B'")
        if rm.get("checkpoint_sha256") != SPEC.S1_SHA256:
            out.append("eval stamp is not an S1 s18 stamp")
    except SystemExit as exc:
        out.append(str(exc))
    if not SPEC.S1_AGENT_YAML.is_file():
        out.append(f"S1 agent.yaml missing: {SPEC.S1_AGENT_YAML}")
    return out


# ===================================================================================== Kit-free modes
def emit_facts(out_dir: Path) -> dict:
    facts = {p: CORE.build_facts(p, SN=SN) for p in CORE.POSES}
    facts["_sources"] = {"sn_table_core_sha256": SN.sha256_file(HERE / "sn_table_core.py"),
                         "linkage": str(CORE.LINKAGE_PATH), "dimensions": str(CORE.DIMS_PATH),
                         "foot_fk": str(CORE.FOOT_FK_PATH), "usd": str(CORE.USD_PATH),
                         "restance_module_sha256": SN.sha256_file(SN.RESTANCE_MODULE), "r_knob": SN.RESTANCE_D3}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "facts.json").write_text(json.dumps(facts, indent=1, sort_keys=True))
    return facts


def print_facts(facts: dict) -> None:
    for pose in CORE.POSES:
        f = facts[pose]
        p, tl, sl = f["pose"], f["table_leg_clearance"], f["slew_zero_action"]
        print(f"[SNT] {pose}: hip {p['hip_deg']:+.2f} deg ({p['hip_binding']}; screw min = {p['hip_rod_min_angle_deg']:+.2f}) "
              f"rod {p['hip_rod_m']:.4f} m | knee L {p['knee_left_deg']:+.2f} / R {-p['knee_left_deg']:+.2f} deg "
              f"({p['knee_binding']}; interior {p['knee_interior_deg']:.1f}) rod {p['knee_rod_m']:.4f} m | "
              f"hold actions hip {p['hip_hold_action']:+.2f} knee {p['knee_hold_action_left']:+.2f} | "
              f"min link clearance {p['link_clearances']['min_m']:.3f} m", flush=True)
        toe = tl["toes"]["ML"]
        print(f"[SNT] {pose}: spawn root z {f['spawn_root_z_m']:.5f}; ML toe body {[round(x, 3) for x in toe['toe_body']]} "
              f"pad bottom z {toe['pad_bottom_world_z']:.3f} (above table top: {toe['above_table_top']}, in footprint: "
              f"{toe['inside_table_footprint_nominal']}); table-leg lateral clearance {tl['lateral_clearance_m']:.3f} m; "
              f"zero-action slew: toes floor {sl['toe_reaches_floor_s']} s, hip {sl['hip_rod_arrives_s']} s, "
              f"knee {sl['knee_rod_arrives_s']} s, window {sl['hip_enters_action_window_s']} s; "
              f"R at start {sorted(set(f['r_at_start']['branches'].values()))} zero={f['r_at_start']['target_is_zero']}",
              flush=True)


def merge(dir_: Path) -> dict:
    paths = sorted(p for p in dir_.glob("*_summary.json") if p.name != "table_test_summary.json")
    merged = CORE.merge_summaries(paths)
    (dir_ / "table_test_summary.json").write_text(json.dumps(merged, indent=1, sort_keys=True))
    print(CORE.format_table(merged), flush=True)
    for v in merged["videos"]:
        e = v.get("env0") or {}
        print(f"[SNT] video {v['pose']} {v['controller']}: {v['mp4']} -- env0 stood={e.get('stood')} "
              f"lift_off={e.get('lift_off_s')} final_gap={e.get('final_gap_m')}", flush=True)
    return merged


# ===================================================================================== helpers used under Kit
def _rss_gb() -> float:
    try:
        with open("/proc/self/status") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024.0 / 1024.0
    except OSError:
        pass
    return float("nan")


def _vec3(text: str):
    vals = tuple(float(x) for x in str(text).split(","))
    if len(vals) != 3:
        raise SystemExit(f"expected 'x,y,z', got {text!r}")
    return vals


def _dump_terms(cfg_block) -> dict:
    out = {}
    for name, term in vars(cfg_block).items():
        if term is None:
            out[name] = None
            continue
        func = getattr(term, "func", None)
        fname = getattr(func, "__name__", None) or type(func).__name__
        params = getattr(term, "params", None) or {}
        out[name] = {"func": fname, "mode": getattr(term, "mode", None),
                     "time_out": getattr(term, "time_out", None),
                     "params": {k: (repr(v) if len(repr(v)) < 300 else repr(v)[:300] + "...") for k, v in params.items()}}
    return out


def _load_harness_functions(S, torch, UniformParkourCommand) -> dict:
    """Execute the harness's OWN definitions (ast-extracted, never copied), as sn_record_video does."""
    want_funcs = ("_wrap_to_pi", "_observed_command_slots", "_apply_env_overrides", "_install_schedule_override_world")
    want_consts = ("OBS_IDX_CMD_VX", "DEFAULT_OBSERVED_COMMAND_SLOTS")
    tree = ast.parse(HARNESS.read_text(), filename=str(HARNESS))
    body, found = [], set()
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in want_funcs:
            body.append(node)
            found.add(node.name)
        elif isinstance(node, ast.Assign):
            names = {t.id for t in node.targets if isinstance(t, ast.Name)}
            if names & set(want_consts):
                body.append(node)
                found |= names & set(want_consts)
    missing = sorted((set(want_funcs) | set(want_consts)) - found)
    if missing:
        raise SystemExit(f"harness {HARNESS} no longer defines {missing}")
    mod = ast.fix_missing_locations(ast.Module(body=body, type_ignores=[]))
    code = compile(mod, str(HARNESS), "exec", flags=_future.annotations.compiler_flag, dont_inherit=True)
    ns = {"S": S, "torch": torch, "types": types, "UniformParkourCommand": UniformParkourCommand,
          "__name__": "eval_crab_hex_gait_extract"}
    exec(code, ns)  # noqa: S102 - the repo's own harness source
    return ns


class _SlotRule:
    """The eval's stand signal: P2L's ``P2RMode.rederive_on_reset`` on a holder carrying what it reads."""

    rederive_on_reset = P2L.P2RMode.rederive_on_reset

    def __init__(self, cmd_term, unwrapped):
        self.cmd_term = cmd_term
        self.unwrapped = unwrapped
        self.resets_rederived = 0


_CAPTION = None


def _caption(frame, lines):
    global _CAPTION
    import numpy as np  # noqa: PLC0415

    if _CAPTION is None:
        try:
            import cv2  # noqa: F401,PLC0415

            _CAPTION = "cv2"
        except Exception:  # noqa: BLE001
            _CAPTION = "none"
    frame = np.ascontiguousarray(frame[:, :, :3]).copy()
    if _CAPTION != "cv2" or not lines:
        return frame
    import cv2  # noqa: PLC0415

    h, w = frame.shape[:2]
    scale = max(0.5, w / 1280.0 * 0.62)
    line_h = int(round(30 * scale / 0.62))
    thick = 2 if scale >= 0.6 else 1
    box_h = 10 + line_h * len(lines)
    box_w = min(w, 20 + max(cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)[0][0] for t, _c in lines))
    frame[:box_h, :box_w] = (frame[:box_h, :box_w].astype(np.uint16) * 2 // 5).astype(np.uint8)
    for i, (text, color) in enumerate(lines):
        y = 8 + line_h * (i + 1) - int(8 * scale / 0.62)
        cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)
    return frame


WHITE, YELLOW, CYAN, RED, GREEN = (255, 255, 255), (255, 230, 90), (120, 230, 255), (255, 80, 60), (150, 255, 150)
CONTROLLER_LABEL = {"stand_net": "STAND NET (A-mask s17, mask 1) on the stand slot", "r": "R (scripted re-stance, D3)",
                    "s1": "S1 s18 (walker) alone", "zero": "ZERO action"}


# ===================================================================================== the Kit run
def _on_sigterm(signum, _frame):
    """``systemctl --user stop`` / the launcher's ``timeout`` send SIGTERM: unwind through the finally blocks (the mp4
    writer closes, the env and Kit close) instead of dying mid-write."""
    raise SystemExit(128 + int(signum))


def run_kit(o: argparse.Namespace, rest: list[str]) -> int:
    from isaaclab.app import AppLauncher  # noqa: PLC0415

    ap = argparse.ArgumentParser(allow_abbrev=False)
    AppLauncher.add_app_launcher_args(ap)
    app_args = ap.parse_args(rest)
    # every KRABBY_* replaced BEFORE Kit boots (an env write after Kit start flips same-seed evals)
    rm_path = resolve_run_meta(o.eval_run)
    want = want_env(json.loads(rm_path.read_text()))
    dropped = sorted(k for k in os.environ if k.startswith("KRABBY_") and os.environ[k] != want.get(k))
    for k in [k for k in os.environ if k.startswith("KRABBY_")]:
        del os.environ[k]
    os.environ.update(want)
    print(f"[SNT] KRABBY_* replaced by {rm_path} env_vars ({len(want)}); dropped/differing: {dropped or 'none'}",
          flush=True)
    t0 = time.time()
    simulation_app = AppLauncher(app_args).app
    print(f"[SNT] Kit up in {time.time() - t0:.1f}s, RSS {_rss_gb():.2f} GB", flush=True)
    signal.signal(signal.SIGTERM, _on_sigterm)
    try:
        if _rss_gb() > o.rss_abort_gb:
            print(f"[SNT][ABORT] RSS {_rss_gb():.1f} GB right after Kit start (DerivedDataCache?)", flush=True)
            return 3
        return _run(o, app_args, rm_path, want, dropped)
    finally:
        simulation_app.close()


def _compose_env_cfg(env_cfg, o, *, H, facts, joint_names, offsets, friction, sim_utils, RigidObjectCfg,
                     EventTermCfg, TerminationTermCfg, il_time_out) -> list:
    """The test's env changes (module docstring), each recorded; refuses (SystemExit) on anything unexpected."""
    changes = []

    def note(what, old, new, why):
        changes.append({"what": what, "old": old, "new": new, "why": why})

    # ---- terminations: a time-out only
    term = env_cfg.terminations
    for name, t in list(vars(term).items()):
        if t is not None:
            note(f"terminations.{name}", _dump_terms(types.SimpleNamespace(**{name: t}))[name], None,
                 "disabled for the table test (falls are recorded by the STOOD classifier, not reset)")
            setattr(term, name, None)
    term.total_terminates = TerminationTermCfg(func=il_time_out, time_out=True)
    note("terminations.total_terminates", None, "isaaclab.envs.mdp.time_out (time_out=True)",
         f"keep a time-out: episode_length_s = {o.episode_s} + 2 s, never reached inside the {o.episode_s}-s loop")
    # ---- events
    ev = env_cfg.events
    for name, t in list(vars(ev).items()):
        if t is None:
            continue
        mode = getattr(t, "mode", None)
        params = getattr(t, "params", None) or {}
        if name == "reset_root_state":
            yr = tuple(float(x) for x in params.get("yaw_range", (None, None)))
            if yr != (0.0, 0.0) or float(params.get("offset", -1.0)) != 0.0:
                raise SystemExit(f"events.reset_root_state params {params}: want yaw_range (0, 0) and offset 0")
            note("events.reset_root_state", "ring spawn (tile centre, yaw pinned 0 by the harness override)",
                 "kept", "parkour spawn bookkeeping; the table event (last) overrides z and joints")
        elif name == "reset_robot_joints":
            note(f"events.{name}", f"reset_joints_by_scale {params.get('position_range')}", None,
                 "the table event writes the retracted joints")
            setattr(ev, name, None)
        elif name == "physics_material":
            fr = tuple(float(x) for x in params.get("friction_range", (1.0, 1.0)))
            f = float(friction) if friction is not None else 0.5 * (fr[0] + fr[1])
            params["friction_range"] = (f, f)
            note("events.physics_material.friction_range", list(fr), [f, f],
                 "one pinned friction (the range's midpoint unless --friction) instead of a per-env DR draw")
        elif name == "base_external_force_torque":
            fz = all(float(x) == 0.0 for x in tuple(params.get("force_range", (0.0, 0.0))) +
                     tuple(params.get("torque_range", (0.0, 0.0))))
            if fz:
                note("events.base_external_force_torque", "zero force / torque", "kept", "a no-op")
            else:
                note("events.base_external_force_torque", params, None, "external force DR off")
                setattr(ev, name, None)
        elif mode in ("interval",) or name.startswith(("randomize_", "random_", "push", "rsi")):
            note(f"events.{name}", f"{mode} {getattr(getattr(t, 'func', None), '__name__', '?')}", None,
                 "randomization / push / RSI off for the table test")
            setattr(ev, name, None)
        else:
            raise SystemExit(f"events.{name} (mode {mode}) is not a term this test knows how to treat; refused")
    ev.sn_table_start = EventTermCfg(
        func=CORE.sn_table_start, mode="reset",
        params={"joint_pos": [float(facts["pose"]["joint_pos"][n]) for n in joint_names],
                "root_z": float(facts["spawn_root_z_m"]), "table_center_z": float(CORE.TABLE_CENTER_Z_M),
                "table_offsets": [list(map(float, x)) for x in offsets]})
    note("events.sn_table_start", None, "sn_table_core.sn_table_start (reset, LAST)",
         f"robot root z {facts['spawn_root_z_m']:.5f}, yaw 0, pose {o.pose}; table at origin + jitter")
    # ---- scene: the table
    tmat = copy.deepcopy(env_cfg.scene.terrain.physics_material)
    env_cfg.scene.sn_table = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/SnTable",
        spawn=sim_utils.CuboidCfg(
            size=(CORE.TABLE_X_M, CORE.TABLE_Y_M, CORE.TABLE_HEIGHT_M),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=CORE.TABLE_MASS_KG),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            physics_material=tmat,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.55, 0.36, 0.20)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, CORE.TABLE_CENTER_Z_M)),
    )
    note("scene.sn_table", None, f"kinematic cuboid {CORE.TABLE_X_M} x {CORE.TABLE_Y_M} x {CORE.TABLE_HEIGHT_M:.3f} m, "
         f"top {CORE.TABLE_TOP_M} m, terrain material {type(tmat).__name__}", "the table")
    # ---- commands / viewer
    env_cfg.commands.base_velocity.debug_vis = False
    if o.video:
        env_cfg.viewer.origin_type = "env"
        env_cfg.viewer.env_index = 0
        env_cfg.viewer.eye = _vec3(o.cam_eye)
        env_cfg.viewer.lookat = _vec3(o.cam_lookat)
        note("viewer", "asset_root follow-cam", f"static env-0 camera eye {o.cam_eye} lookat {o.cam_lookat}", "video")
    obs_pol = getattr(getattr(env_cfg, "observations", None), "policy", None)
    note("observations.policy.enable_corruption", getattr(obs_pol, "enable_corruption", None), "unchanged",
         "recorded only (the eval env's value)")
    return changes


def _run(o, app_args, rm_path: Path, want: dict, dropped: list) -> int:
    import gymnasium as gym  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415
    import torch  # noqa: PLC0415
    import yaml  # noqa: PLC0415

    parkour_scripts = PARKOUR / "scripts"
    for p in (str(parkour_scripts), str(parkour_scripts / "rsl_rl"), str(PARKOUR / "parkour_tasks")):
        while p in sys.path:
            sys.path.remove(p)
    sys.path.insert(0, str(REPO))
    sys.path.insert(0, str(PARKOUR))
    sys.path.insert(0, str(PARKOUR / "parkour_tasks"))
    if str(HARNESS_DIR) not in sys.path:
        sys.path.insert(0, str(HARNESS_DIR))
    import isaaclab.sim as sim_utils  # noqa: PLC0415
    import parkour_tasks  # noqa: F401,PLC0415
    from gait_eval import schedule as S  # noqa: PLC0415
    from isaaclab.assets import RigidObjectCfg  # noqa: PLC0415
    from isaaclab.envs.mdp import time_out as il_time_out  # noqa: PLC0415
    from isaaclab.managers import EventTermCfg, TerminationTermCfg  # noqa: PLC0415
    from isaaclab_tasks.utils import parse_env_cfg  # noqa: PLC0415
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: PLC0415
    from parkour_isaaclab.envs.mdp.parkour_commands.uniform_parkour_command import (  # noqa: PLC0415
        UniformParkourCommand,
    )
    from scripts.rsl_rl.modules.crab_actor_critic_with_encoder import CrabHexActorCriticRMA  # noqa: PLC0415
    from scripts.rsl_rl.modules.feature_extractors.estimator import DefaultEstimator  # noqa: PLC0415
    from scripts.rsl_rl.runner_factory import agent_cfg_to_train_dict  # noqa: PLC0415
    from scripts.rsl_rl.vecenv_wrapper import ParkourRslRlVecEnvWrapper  # noqa: PLC0415

    dcore = _load(HERE / "sn_dagger_core.py", "_snt_dagger_core")       # imports torch at its top: post-Kit only
    H = _load_harness_functions(S, torch, UniformParkourCommand)
    out = Path(o.out)
    out.mkdir(parents=True, exist_ok=True)
    rm = json.loads(rm_path.read_text())
    krabby_now = {k: v for k, v in sorted(os.environ.items()) if k.startswith("KRABBY_")}
    if krabby_now != want:
        raise SystemExit(f"KRABBY_* after the task import differs from the stamp: {sorted(set(krabby_now.items()) ^ set(want.items()))}")
    facts = CORE.build_facts(o.pose, SN=SN)
    joint_names = list(rm["joint_names"])
    offsets = CORE.jitter_draws(o.num_envs, o.seed, enabled=o.jitter)

    # ---- env cfg: the eval's composition + the test's changes
    env_cfg = parse_env_cfg(TASK_ID, device=app_args.device, num_envs=o.num_envs)
    events_before, terms_before = _dump_terms(env_cfg.events), _dump_terms(env_cfg.terminations)
    scen = types.SimpleNamespace(id="sn_table", env_seed=int(o.seed), episode_length_s=float(o.episode_s) + 2.0,
                                 total_duration_s=float(o.episode_s), freeze_terrain_levels=True,
                                 terrain_generator_seed=None, freeze_friction=False, command_frame="world",
                                 spawn_yaw=0.0, spawn_yaw_rad=lambda: 0.0)
    applied = H["_apply_env_overrides"](env_cfg, scen)
    changes = _compose_env_cfg(env_cfg, o, H=H, facts=facts, joint_names=joint_names, offsets=offsets,
                               friction=o.friction, sim_utils=sim_utils, RigidObjectCfg=RigidObjectCfg,
                               EventTermCfg=EventTermCfg, TerminationTermCfg=TerminationTermCfg,
                               il_time_out=il_time_out)
    events_after, terms_after = _dump_terms(env_cfg.events), _dump_terms(env_cfg.terminations)
    for c in changes:
        print(f"[SNT] change {c['what']}: {c['old']} -> {c['new']} ({c['why']})", flush=True)

    agent_cfg = load_cfg_from_registry(TASK_ID, "rsl_rl_cfg_entry_point")
    train_cfg = agent_cfg_to_train_dict(agent_cfg)
    bad = SPEC.network_cfg_reasons(train_cfg, yaml.safe_load(SPEC.S1_AGENT_YAML.read_text()))
    if bad:
        raise SystemExit("network cfg differs from S1's agent.yaml:\n  " + "\n  ".join(bad))
    env = ParkourRslRlVecEnvWrapper(gym.make(TASK_ID, cfg=env_cfg, render_mode="rgb_array" if o.video else None),
                                    clip_actions=agent_cfg.clip_actions)
    cam = None
    try:
        unwrapped = env.unwrapped
        device = env.device
        dt = float(unwrapped.step_dt)
        N = int(unwrapped.num_envs)
        T = int(round(o.episode_s / dt))
        if int(env.num_obs) != 1149 or int(env.num_actions) != 18 or float(env.clip_actions) != 1.0:
            raise SystemExit(f"env dims / clip {env.num_obs} / {env.num_actions} / {env.clip_actions}: want 1149 / 18 / 1.0")
        if int(unwrapped.max_episode_length) <= T:
            raise SystemExit(f"max_episode_length {unwrapped.max_episode_length} <= the loop's {T} steps")
        robot = unwrapped.scene["robot"]
        table = unwrapped.scene["sn_table"]
        if list(robot.joint_names) != joint_names:
            raise SystemExit(f"joint order {robot.joint_names} != the stamp's {joint_names}")
        usd = str(getattr(getattr(robot.cfg, "spawn", None), "usd_path", ""))
        if Path(usd).name != Path(rm["usd_path"]).name:
            raise SystemExit(f"spawned {usd}, the stamp spawned {rm['usd_path']}")
        if _rss_gb() > o.rss_abort_gb:
            raise SystemExit(f"RSS {_rss_gb():.1f} GB > {o.rss_abort_gb} before the first step")

        # ---- joint limits / defaults / start state (fail closed)
        lim = CORE.joint_limits_rad()
        jl = getattr(robot.data, "joint_pos_limits", None)
        jl = robot.data.default_joint_pos_limits if jl is None else jl
        jl0 = jl[0].detach().cpu().numpy()
        dj0 = robot.data.default_joint_pos[0].detach().cpu().numpy()
        dflt = CORE.defaults_rad()
        lim_rows, mism = {}, []
        for j, n in enumerate(joint_names):
            lo, hi = float(jl0[j, 0]), float(jl0[j, 1])
            lim_rows[n] = [lo, hi]
            leg = n.split("_")[0]
            if "Hip_Femur" in n:
                exp, dexp = lim["hip"], dflt["hip"]
            elif "Femur_Tibia" in n:
                exp = lim["knee_left"] if leg in CORE.LEFT else lim["knee_right"]
                dexp = dflt["knee_left"] if leg in CORE.LEFT else -dflt["knee_left"]
            else:
                exp, dexp = None, 0.0
            if exp is not None and (abs(lo - exp[0]) > 1e-3 or abs(hi - exp[1]) > 1e-3):
                mism.append(f"{n}: sim limits [{lo:.4f}, {hi:.4f}] vs dims [{exp[0]:.4f}, {exp[1]:.4f}]")
            if abs(float(dj0[j]) - dexp) > 1e-4:
                mism.append(f"{n}: default {float(dj0[j]):.5f} vs expected {dexp:.5f}")
            want_q = facts["pose"]["joint_pos"][n]
            if not lo <= want_q <= hi:
                mism.append(f"{n}: start {want_q:.4f} outside the hard limits [{lo:.4f}, {hi:.4f}]")
        if mism:
            raise SystemExit("joint limits / defaults / start pose refused:\n  " + "\n  ".join(mism))
        jp_now = robot.data.joint_pos.detach().cpu().numpy()
        jp_want = np.asarray([facts["pose"]["joint_pos"][n] for n in joint_names])
        rp_now = robot.data.root_pos_w.detach().cpu().numpy()
        rq_now = robot.data.root_quat_w.detach().cpu().numpy()
        org = unwrapped.scene.env_origins.detach().cpu().numpy()
        tp_now = table.data.root_pos_w.detach().cpu().numpy()
        start = {"joint_pos_max_err": float(np.abs(jp_now - jp_want[None]).max()),
                 "root_z_err_max": float(np.abs(rp_now[:, 2] - facts["spawn_root_z_m"]).max()),
                 "root_xy_from_origin_max": float(np.abs(rp_now[:, :2] - org[:, :2]).max()),
                 "root_quat_err_max": float(np.abs(rq_now - np.array([1.0, 0, 0, 0])[None]).max()),
                 "table_xy_err_max": float(np.abs(tp_now[:, :2] - (org[:, :2] + np.asarray(offsets)[:, :2])).max()),
                 "table_z_err_max": float(np.abs(tp_now[:, 2] - CORE.TABLE_CENTER_Z_M).max())}
        print(f"[SNT] start state after the construction reset: {start}", flush=True)
        if (start["joint_pos_max_err"] > 1e-3 or start["root_z_err_max"] > 1e-3 or start["table_xy_err_max"] > 1e-3
                or start["table_z_err_max"] > 1e-3 or start["root_quat_err_max"] > 1e-4):
            raise SystemExit(f"the start state is not the planned one: {start}")

        # ---- networks (no PPO runner; global RNG saved / restored)
        before = dcore.save_rng()
        try:
            s1, s1_est, s1_ck = dcore.load_strict(o.s1_ckpt, CrabHexActorCriticRMA, DefaultEstimator, train_cfg, device)
            st = st_est = None
            stand_check = None
            if o.controller == "stand_net":
                st, st_est, st_ck = dcore.load_strict(o.stand_ckpt, CrabHexActorCriticRMA, DefaultEstimator, train_cfg,
                                                      device)
                ok, why = SN.state_dicts_equal(st_ck["estimator_state_dict"], s1_ck["estimator_state_dict"])
                if not ok:
                    raise SystemExit(f"the stand checkpoint's estimator differs from S1's ({why})")
                sd = (st_ck.get("infos") or {}).get("sn_dagger") or {}
                tm = sd.get("stand_mask")
                if type(tm) is not int or tm != int(o.stand_mask):
                    raise SystemExit(f"stand checkpoint stand_mask {tm!r} != --stand-mask {o.stand_mask}")
                if sd.get("s1_sha256") != SPEC.S1_SHA256:
                    raise SystemExit("the stand net was not trained beside this S1")
                stand_check = {"estimator_equals_s1": True, "stand_mask": tm, "arm": sd.get("arm"),
                               "seed": sd.get("seed"), "iterations": sd.get("iterations"),
                               "trained_beside_s1_sha256": sd.get("s1_sha256"), "r_knob": sd.get("r_knob")}
                del st_ck
            del s1_ck
        finally:
            dcore.restore_rng(before)
        if not dcore.rng_equal(before, dcore.save_rng()):
            raise SystemExit("global RNG not restored after building the networks")
        p0, p1, num_prop, _num_hist = dcore.priv_cols(train_cfg)

        # ---- the stand command from t = 0 (harness override + the eval latch)
        cmd_term = unwrapped.command_manager.get_term("base_velocity")
        if not isinstance(cmd_term, UniformParkourCommand):
            raise SystemExit(f"base_velocity is {type(cmd_term)}")
        latch_pre = P2L.arm_slot_latch(cmd_term, N, cmd_term.slot_id.device)
        slot_rule = _SlotRule(cmd_term, unwrapped)
        compiled = S.CompiledSchedule(
            cmd=np.zeros((T, 1, 3), dtype=np.float32), segment_id=np.zeros((T, 1), dtype=np.int16),
            steady_mask=np.ones((T, 1), dtype=bool), delta_yaw=None, n_steps=T, dt=dt, hold_labels=["stand"],
            world_dir=np.zeros((T, 1)), world_speed=np.zeros((T, 1)), world_face=np.zeros((T, 1)))
        override_state = H["_install_schedule_override_world"](cmd_term, compiled, device, ["stand"])
        obs, _ = env.get_observations()
        override_state["apply"](0)
        obs, _ = env.get_observations()
        observed_slots, _src = H["_observed_command_slots"](unwrapped)
        cmd0 = float(cmd_term.vel_command_b.abs().max().item())
        obs_cmd0 = max(float(obs[:, int(s)].abs().max().item()) for s in observed_slots)
        stand0 = slot_rule.rederive_on_reset()
        if cmd0 != 0.0 or obs_cmd0 != 0.0 or not bool(stand0.all()):
            raise SystemExit(f"the stand command is not in place at t=0: |cmd| {cmd0}, obs cmd slots {obs_cmd0}, "
                             f"stand slot {int(stand0.sum())}/{N}")
        print(f"[SNT] stand row from t=0: |vel_command_b| 0, observed slots {observed_slots} read 0, latched stand slot "
              f"{N}/{N} (latch pre-armed: {latch_pre})", flush=True)

        # ---- sensors / ids
        contact = unwrapped.scene.sensors["contact_forces"]
        body_sid = contact.find_bodies("body")[0]
        if len(body_sid) != 1:
            raise SystemExit(f"contact sensor 'body' ids {body_sid}")
        feet = [f"{leg}_Footpad" for leg in CORE.LEGS]
        foot_sid, fs_names = contact.find_bodies(feet, preserve_order=True)
        foot_bid, fb_names = robot.find_bodies(feet, preserve_order=True)
        if list(fs_names) != feet or list(fb_names) != feet:
            raise SystemExit(f"foot order {fs_names} / {fb_names}")
        cam_cols = torch.tensor([joint_names.index(f"{leg}_Body_CamShaft_RevoluteJoint") for leg in CORE.LEGS],
                                device=device)
        cam_def = robot.data.default_joint_pos[:, cam_cols]
        off_t = torch.as_tensor(offsets, device=device, dtype=torch.float32)
        tcen = unwrapped.scene.env_origins[:, :2] + off_t[:, :2]

        # ---- buffers (device), entry t = the state after control step t
        B = {"root_pos": torch.zeros(T, N, 3, device=device), "root_quat": torch.zeros(T, N, 4, device=device),
             "lowest_z": torch.zeros(T, N, device=device), "cos_tilt": torch.zeros(T, N, device=device),
             "body_force": torch.zeros(T, N, device=device), "foot_z": torch.zeros(T, N, 6, device=device),
             "foot_force": torch.zeros(T, N, 6, device=device), "joint_pos": torch.zeros(T, N, 24, device=device),
             "cam_abs": torch.zeros(T, N, 6, device=device), "actions": torch.zeros(T, N, 18, device=device),
             "stand": torch.zeros(T, N, dtype=torch.bool, device=device),
             "cmd_wz": torch.zeros(T, N, device=device), "done": torch.zeros(T, N, dtype=torch.bool, device=device),
             "table_xy": torch.zeros(T, N, 2, device=device)}
        if o.controller == "r":
            B["r_branch"] = torch.zeros(T, N, 6, dtype=torch.int8, device=device)

        writer = None
        mp4 = out / f"{o.prefix}.mp4"
        if o.video:
            import imageio  # noqa: PLC0415

            cam = getattr(unwrapped, "viewport_camera_controller", None)
            for i in range(60):
                f0 = unwrapped.render()
                if f0 is not None and f0.size and float(f0.mean()) > 1.0:
                    print(f"[SNT] renderer warm after {i + 1} render(s); frame {f0.shape}", flush=True)
                    break
            writer = imageio.get_writer(str(mp4), fps=int(round(1.0 / dt)), codec="libx264", quality=o.quality,
                                        macro_block_size=None, ffmpeg_log_level="error")

        def caption_lines(t_s, i, extra_branch=None):
            rz = float(robot.data.root_pos_w[i, 2].item())
            q = robot.data.root_quat_w[i]
            low, cz = CORE.lowest_z_and_cos_tilt(rz, float(q[0]), float(q[1]), float(q[2]), float(q[3]))
            gap = low - CORE.TABLE_TOP_M
            ff = contact.data.net_forces_w[i, foot_sid].norm(dim=-1)
            fz = robot.data.body_pos_w[i, foot_bid, 2]
            nfeet = int(((ff >= CORE.FOOT_LOADED_N) & (fz <= CORE.FOOT_FLOOR_Z_M)).sum().item())
            camx = float(torch.abs(torch.remainder(robot.data.joint_pos[i, cam_cols] - cam_def[i] + math.pi,
                                                   2 * math.pi) - math.pi).max().item())
            tilt = math.degrees(math.acos(max(-1.0, min(1.0, cz))))
            lines = [(f"TABLE STAND-UP TEST  |  {CONTROLLER_LABEL[o.controller]}  |  start: {o.pose}", WHITE),
                     (f"table top {CORE.TABLE_TOP_M:.3f} m (25 in)   stand command from t=0   seed {o.seed}", CYAN),
                     (f"t = {t_s:5.2f} / {o.episode_s:.1f} s   body (root) height {rz:.3f} m   belly gap {gap * 100:+.1f} cm"
                      f"   {'BELLY ON TABLE' if gap <= CORE.ON_TABLE_GAP_M else 'belly off the table'}",
                      YELLOW if gap <= CORE.ON_TABLE_GAP_M else GREEN),
                     (f"feet on floor (>=20 N, z<=0.15 m): {nfeet}/6   tilt {tilt:4.1f} deg   max|cam| {camx:.2f} rad",
                      GREEN if nfeet >= CORE.STOOD_MIN_FEET else WHITE)]
            if extra_branch is not None:
                lines.append((f"R branches: {extra_branch}", CYAN))
            if tilt > math.degrees(CORE.FALL_TILT_RAD):
                lines.append(("FALL: tilt > 0.5 rad", RED))
            if gap < -CORE.FELL_OFF_DROP_M:
                lines.append(("FELL OFF THE TABLE", RED))
            return lines

        if writer is not None:
            writer.append_data(_caption(unwrapped.render(), caption_lines(0.0, 0)))
        table_check: dict = {}
        trace = hashlib.sha256()
        t_run = time.time()
        branch_names = None
        if o.controller == "r":
            branch_names = SN.load_rst().BRANCH_NAMES
        try:
            with torch.inference_mode():
                for t in range(T):
                    override_state["step"] = t + 1
                    stand = slot_rule.rederive_on_reset()
                    actions, _a_s1, extra = CORE.controller_actions(
                        o.controller, obs, stand, SN=SN, s1=s1, s1_est=s1_est, p0=p0, p1=p1, num_prop=num_prop,
                        st_pol=st, st_est=st_est, mask=int(o.stand_mask))
                    B["actions"][t] = actions.clamp(-1.0, 1.0)
                    B["stand"][t] = stand
                    B["cmd_wz"][t] = cmd_term.vel_command_b[:, 2]
                    if "r_branch" in B:
                        B["r_branch"][t] = extra["r_branch"].to(torch.int8)
                    trace.update(actions.detach().cpu().numpy().tobytes())
                    obs, _, dones, _ = env.step(actions)
                    rp, rq = robot.data.root_pos_w, robot.data.root_quat_w
                    low, cz = CORE.lowest_z_and_cos_tilt(rp[:, 2], rq[:, 0], rq[:, 1], rq[:, 2], rq[:, 3])
                    B["root_pos"][t], B["root_quat"][t] = rp, rq
                    B["lowest_z"][t], B["cos_tilt"][t] = low, cz
                    B["body_force"][t] = contact.data.net_forces_w[:, body_sid[0]].norm(dim=-1)
                    B["foot_z"][t] = robot.data.body_pos_w[:, foot_bid, 2]
                    B["foot_force"][t] = contact.data.net_forces_w[:, foot_sid].norm(dim=-1)
                    jp = robot.data.joint_pos
                    B["joint_pos"][t] = jp
                    B["cam_abs"][t] = torch.abs(torch.remainder(jp[:, cam_cols] - cam_def + math.pi, 2 * math.pi)
                                                - math.pi)
                    rel = rp[:, :2] - tcen
                    c, s = torch.cos(-off_t[:, 2]), torch.sin(-off_t[:, 2])
                    B["table_xy"][t] = torch.stack([c * rel[:, 0] - s * rel[:, 1], s * rel[:, 0] + c * rel[:, 1]], -1)
                    B["done"][t] = dones.reshape(-1).bool()
                    trace.update(rp.detach().cpu().numpy().tobytes())
                    if t == 0:
                        # fail closed: the kinematic table must BE at its pose in PhysX (read back after the step)
                        tp1 = table.data.root_pos_w
                        e_xy = float((tp1[:, :2] - tcen).abs().max().item())
                        e_z = float((tp1[:, 2] - CORE.TABLE_CENTER_Z_M).abs().max().item())
                        table_check["after_step1_xy_err_max"], table_check["after_step1_z_err_max"] = e_xy, e_z
                        if e_xy > 0.01 or e_z > 0.01:
                            raise SystemExit(f"the kinematic table is not at its pose in PhysX after the first step "
                                             f"(xy err {e_xy:.4f}, z err {e_z:.4f} m)")
                    if writer is not None:
                        eb = None
                        if "r_branch" in extra:
                            bc = extra["r_branch"][0].tolist()
                            eb = " ".join(f"{leg}:{branch_names[int(b)]}" for leg, b in zip(CORE.LEGS, bc))
                        writer.append_data(_caption(unwrapped.render(), caption_lines((t + 1) * dt, 0, eb)))
                    if (t + 1) % 250 == 0:
                        rss = _rss_gb()
                        print(f"[SNT] step {t + 1}/{T} ({time.time() - t_run:.0f}s) RSS {rss:.2f} GB; env0 root z "
                              f"{float(rp[0, 2]):.3f} gap {float(low[0]) - CORE.TABLE_TOP_M:+.3f}", flush=True)
                        if rss > o.rss_abort_gb:
                            raise SystemExit(f"RSS {rss:.1f} GB > {o.rss_abort_gb} at step {t + 1}")
        finally:
            if writer is not None:
                writer.close()

        # ---- classify + write
        tp_end = table.data.root_pos_w
        table_check["end_xy_err_max"] = float((tp_end[:, :2] - tcen).abs().max().item())
        table_check["end_z_err_max"] = float((tp_end[:, 2] - CORE.TABLE_CENTER_Z_M).abs().max().item())
        print(f"[SNT] table pose in PhysX: {table_check}", flush=True)
        A = {k: v.detach().cpu().numpy() for k, v in B.items()}
        feet_on = np.stack([CORE.feet_on_floor(A["foot_force"][:, i], A["foot_z"][:, i]) for i in range(N)], axis=1)
        per_env = []
        for i in range(N):
            r = CORE.classify({"lowest_z": A["lowest_z"][:, i], "cos_tilt": A["cos_tilt"][:, i],
                               "feet_on_floor": feet_on[:, i], "root_z": A["root_pos"][:, i, 2],
                               "cam_abs_max": A["cam_abs"][:, i].max(axis=-1), "terminated": A["done"][:, i]}, dt)
            r["env"] = i
            r["table_offset"] = [float(x) for x in offsets[i]]
            r["final_table_frame_xy_m"] = [float(x) for x in A["table_xy"][-1, i]]
            r["stand_slot_share"] = float(A["stand"][:, i].mean())
            if "r_branch" in A:
                r["r_branch_share"] = {n_: float((A["r_branch"][:, i] == k).mean()) for k, n_ in
                                       enumerate(branch_names)}
            per_env.append(r)
        agg = CORE.aggregate(per_env)
        routing = {"stand_slot_share": float(A["stand"].mean()), "non_stand_steps": int((~A["stand"]).sum()),
                   "yaw_cmd_nonzero_steps": int((A["cmd_wz"] != 0).sum()), "resets_rederived":
                   int(slot_rule.resets_rederived), "enter_rule": "latched_stand_slot (P2L rederive_on_reset)",
                   "route": "sn_common.route_actions(stand, a_controller, a_s1)" if o.controller in ("stand_net", "r")
                   else o.controller}
        files = {"script": _THIS, "core": HERE / "sn_table_core.py", "sn_common": HERE / "sn_common.py",
                 "sn_dagger_core": HERE / "sn_dagger_core.py", "harness": HARNESS, "p2l": P2R_LAUNCHER,
                 "restance_module": SN.RESTANCE_MODULE}
        meta = {
            "kind": "sn_table_test", "task": TASK_ID, "controller": o.controller, "pose": o.pose, "num_envs": N,
            "seed": o.seed, "jitter": bool(o.jitter), "episode_s": o.episode_s, "dt": dt, "steps": T,
            "smoke": bool(o.smoke), "video": bool(o.video), "mp4": str(mp4) if o.video else None,
            "s1_ckpt": str(o.s1_ckpt), "s1_sha256": SPEC.S1_SHA256,
            "stand_ckpt": str(o.stand_ckpt) if o.controller == "stand_net" else None,
            "stand_sha256": STAND_SHA256 if o.controller == "stand_net" else None, "stand_mask": int(o.stand_mask),
            "stand_check": stand_check, "r_knob": SN.RESTANCE_D3 if o.controller == "r" else None,
            "forward": SN.FORWARD if o.controller == "stand_net" else None,
            "eval_run_meta": str(rm_path), "env_vars": want, "krabby_dropped": dropped,
            "harness_overrides": {k: (v if isinstance(v, (int, float, str, bool, type(None))) else repr(v))
                                  for k, v in applied.items()},
            "max_episode_length": int(unwrapped.max_episode_length), "clip_actions": float(env.clip_actions),
            "observed_command_slots": {str(k): int(v) for k, v in observed_slots.items()},
            "latch_preexisting": bool(latch_pre), "joint_limits_sim": lim_rows, "start_state": start,
            "table_pose_check": table_check,
            "trace_sha256": trace.hexdigest(), "wall_s": time.time() - t_run, "peak_rss_gb":
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0 / 1024.0,
            "shas": {k: SN.sha256_file(p) for k, p in files.items()}, "app_args": vars(app_args) if hasattr(
                app_args, "__dict__") else None,
        }
        summary = {"meta": meta, "facts": facts, "stood_definition": CORE.STOOD_DEFINITION, "env_changes": changes,
                   "events_before": events_before, "events_after": events_after, "terminations_before": terms_before,
                   "terminations_after": terms_after, "routing": routing, "aggregate": agg, "per_env": per_env}
        (out / f"{o.prefix}_summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True, default=str))
        npz = {f"env0_{k}": v[:, 0] for k, v in A.items()}
        npz.update({"all_lowest_z": A["lowest_z"], "all_cos_tilt": A["cos_tilt"], "all_root_z": A["root_pos"][..., 2],
                    "all_feet_on_floor": feet_on, "all_body_force": A["body_force"],
                    "all_cam_abs_max": A["cam_abs"].max(axis=-1), "table_offsets": np.asarray(offsets),
                    "dt": np.asarray(dt), "table_top_z": np.asarray(CORE.TABLE_TOP_M),
                    "joint_names": np.asarray(joint_names), "legs": np.asarray(CORE.LEGS)})
        np.savez_compressed(out / f"{o.prefix}_traces.npz", **npz)
        print(f"[SNT] {o.controller} / {o.pose}: STOOD {agg['n_stood']}/{agg['n_envs']} (strict every-step feet "
              f"{agg['n_stood_strict']}; lowest window share with >= 5 feet "
              f"{agg['feet_share_last_window']['min']}; lowest window feet {agg['min_feet_last_window']['min']}); "
              f"lift-off median "
              f"{agg['lift_off_s']['median']} s ({agg['n_lifted_off']} lifted); final root z median "
              f"{agg['final_root_z_m']['median']}; final feet median {agg['final_feet_on_floor']['median']}; max tilt "
              f"{agg['max_tilt_rad']['max']}; falls {agg['falls']}; fell off {agg['fell_off_table']}; cams homed "
              f"{agg['cams_homed_end']}; stand-slot share {routing['stand_slot_share']:.3f}", flush=True)
        print(f"[SNT] wrote {out / (o.prefix + '_summary.json')}" + (f" and {mp4}" if o.video else ""), flush=True)
        return 0
    finally:
        if cam is not None:
            cam._viewport_camera_update_handle = None
        env.close()


# ===================================================================================== entry
def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv if argv is None else argv)
    try:
        o, rest = parse(argv[1:])
        if o.merge:
            merge(Path(o.merge))
            return 0
        if o.emit_facts:
            if o.out is None:
                raise SystemExit("--emit-facts needs --out")
            print_facts(emit_facts(Path(o.out)))
            return 0
        bad = pre_kit_reasons(o, rest)
        if bad:
            raise SystemExit("sn_table_test: refused before Kit:\n  " + "\n  ".join(bad))
    except SystemExit as exc:
        if isinstance(exc.code, str):
            print(exc.code, file=sys.stderr, flush=True)
            return REFUSED_RC
        raise
    return run_kit(o, rest)


if __name__ == "__main__":
    sys.exit(main())
