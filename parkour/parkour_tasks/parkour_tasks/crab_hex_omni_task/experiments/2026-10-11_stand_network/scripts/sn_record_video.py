#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""Stand-network campaign -- follow-cam VIDEO of the routed pair (S1 walks, the stand network stands) through one
continuous world-frame schedule row, recorded through the eval harness's own path (VIDEO ONLY; 2026-10-13).

A COPY of ``experiments/tools/record_categories.py`` (never edited) restricted to its SEQUENCE MODE, with a second
network. Launch through ``sn_record_video.sh`` (one Isaac process at a time, memory cap, detached)::

    sn_record_video.py --headless --enable_cameras                       # the defaults below
    sn_record_video.py --headless --enable_cameras --manifest <yaml> --sequence <row> --prefix <p>

Defaults = the deployment the campaign evaluated (REPORT.md section 1-2): S1 s18 (``--checkpoint``,
``crab_hex_omni_teacher/2026-10-07_07-50-10_seed18_X08_p2a_A15pB/model_19994.pt``) walks every command hold and the
stand network A-mask s17 (``--stand-ckpt``, ``stand_network/runs/SN_Amask_s17/model_4000.pt``, ``--stand-mask 1``)
acts on every STAND hold; the KRABBY_* environment is the SN_Amask_s17 suite es1 eval stamp's
(``--eval-run``: ``phase2/lr/SN_Amask_s17/suite__omni/seed001``, newest complete stamp); the row is
``sn_random__omni`` of ``videos/scenarios_sn_video.yaml`` (``sn_video_schedule.py``). This is a VIEWER, not a
scorer: it writes an mp4, stills and a sanity JSON, never a ledger entry, and nothing in the campaign reads them.

WHAT IT KEEPS FROM record_categories.py (see that file for the full argument)

* ENV COMPOSITION = the reference stamp's: every ``KRABBY_*`` is replaced BEFORE Kit boots by exactly its
  ``run_meta["env_vars"]`` (``STAND_OBS=0``, ``CLOCK_HIDE=1``, ``STD_PARAM=soft``, ``INIT_STD=0.8``, the ellipse ...)
  and re-checked after the task imports; ``sn_common.env_reasons`` (the campaign launcher's D5 / L10 refusals) must
  pass on it. The row is built as the harness's single pass builds a schedule row (``S.load_manifest`` validates it,
  ``episodes`` 1, ``_apply_env_overrides``: seed = the row's ``env_seed``, spawn yaw pinned to the first face,
  resampling / curriculum / pushes off, frozen terrain levels); the row's task / terrain_label must equal the
  reference stamp's (``suite__omni``: flat) and a row ``env:`` block is refused. The final row must pass
  ``S.validate_scenario`` (no ``--allow-invalid-schedule`` here: the generator guarantees it).
* COMMAND PIN = the harness's own ``_install_schedule_override_world`` (extracted from ``eval_crab_hex_gait.py`` with
  ``ast`` and executed, never copied; its sha256 is checked against the stamp's ``p2r_mode.harness_sha256``).
* STREAMING imageio writer (never gym ``RecordVideo``: the memory hazard), command arrows (``--no-arrows`` = the
  eval's visual state), captions (``--no-captions``), per-hold stills, RSS abort (at Kit start, before the first
  step and every 500 steps), the rolling ``trace_sha256``. Recording continues after a fall (env auto-reset,
  respawn at the pinned yaw, schedule carries on); the eval's single pass would stop once every env terminated.

WHAT IS NEW: THE SECOND NETWORK, BUILT AND RUN AS THE CAMPAIGN'S EVAL LAUNCHER DOES

* BUILD (``sn_stand_net_eval_crab_hex_gait.py`` -> P2L ``p2r_mode_eval_crab_hex_gait.py`` st2 machinery; P2L is
  loaded BY PATH into a private module and only its functions are called): S1's runner via the ORIGINAL runner factory
  on the real env with its one ``get_observations`` captured (``build_with_template``); the stand runner on P2L's
  ``StandInEnv`` (zero observations WITHOUT an observation-manager call: no history push, no noise draw) with the
  CPU / CUDA / numpy / python RNG states saved and restored around it (``rng_equal`` checked), ``runner.load(...,
  load_optimizer=False)`` (``load_state_dict`` strict); the eval slot latch armed (``arm_slot_latch``); then S1's
  ``runner.load`` and the harness's teacher path. REFUSED before the first step (SystemExit): a distillation runner or
  empirical normalisation; a stand estimator that differs from S1's (``bind_s1``); a trainer checkpoint whose
  ``infos.sn_dagger.stand_mask`` is not ``--stand-mask`` (``trained_arm``; a checkpoint without ``infos.sn_dagger``
  must be S1 itself); a stand actor whose loaded parameters differ from the file's; a trainer checkpoint whose
  ``infos.sn_dagger.s1_sha256`` is not ``--checkpoint``'s.
* STEP (``StandNetMode.__call__``'s order, via ``sn_video_core.route_step``): S1's estimate written into the
  priv-explicit slice (the harness), ``a_s1 = S1(obs, hist_encoding=True)``, ``x = sn_common.stand_net_input(obs,
  stand_estimator, p0, p1, num_prop, mask)`` (the stand ckpt's own estimate on the UNMASKED prop -> a NEW tensor ->
  ``stand_mask_obs`` iff ``--stand-mask 1``), ``a_st = stand(x, hist_encoding=True)``, ``actions =
  sn_common.route_actions(stand, a_st, a_s1)``. Both networks act every step; the harness's obs is never written.
* STAND-NESS (``--stand-rule``, default ``schedule``; ``sn_video_core`` docstring): a hold is STAND iff speed 0 and
  face == the previous hold's face; turn holds (speed 0, facing change) stay S1's for their whole length; the
  prologue takes hold 0's flag; hard switch -- entry on the first step of a stand hold, exit on the step the hold
  changes. The eval's own rule (the command term's LATCHED stand slot, ``P2RMode.rederive_on_reset`` every step) is
  computed and recorded beside it; it differs on EVERY stand hold after the first facing change (the latch re-arms
  only on a fresh facing target or a reset, so it keeps TURN through later same-face moves and stands: the eval never
  ran that sequence; ``sn_video_core.eval_slot_stand_flags`` predicts it, the plan log names the holds) and after a
  fall-respawn at a yaw other than the hold's face. ``--stand-rule slot`` routes on it instead.
* POST-FALL: the env respawns at the pinned yaw while the schedule keeps its absolute face, so a stand hold after a
  fall can carry a live yaw-rate command. The default rule still routes it to the stand net (the user's "stand net
  on every stand"); such holds are MARKED -- per hold ``yaw_cmd`` (``post_fall_turn_cmd``, ``yaw_cmd_at_entry``,
  ``stand_net_on_yaw_cmd_steps``: ``sn_video_core.yaw_cmd_stats``), ``stand_summary.stand_holds_post_fall_turn_cmd``,
  and a red caption line on every such frame.
* CAPTION NAMES come from the LOADED checkpoints unless ``--stand-tag`` / ``--label`` are given: the stand net's
  ``infos.sn_dagger`` arm / seed (``A-mask s17``), the walker's run dir seed and whether it is the S1 the stand net
  was trained beside (``S1 s18``).

OUTPUTS (``--out-dir``, default ``parkour/logs/videos/sn_stand_network``): ``<prefix>_sequence.mp4``,
``stills/<prefix>_sequence_<NN>_<hold>_{mid,end}.png``, ``<prefix>_sequence_sanity.json``: per segment (prologue +
every hold) the record_categories block (commanded vs applied, achieved motion, falls) plus ``kind``, the routed
network and its share, the latched-slot stand share, the routing check (applied == the routed network's action,
bitwise), the swap-live count (stand-net action != S1's) and the ``yaw_cmd`` block; per STAND hold
``stand_net_share`` (1.0), ``cam_max_end_rad`` / ``homed_at_end`` (<= 0.10 rad), the settle time, p2s_stand's
last-5-s ``homed``, falls, ``post_fall_turn_cmd`` / ``yaw_cmd_at_entry``; the run meta (both checkpoints' sha256 and
caption tags, the reference stamp and every sha it was checked against, env, rule, the predicted eval-slot TURN
stands, dt / fps, trace sha256).
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
import resource
import signal
import sys
import time
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent                    # 2026-10-11_stand_network/scripts
CAMPAIGN = HERE.parent
EXPERIMENTS = HERE.parents[1]
PKG = HERE.parents[2]                                      # crab_hex_omni_task
REPO = HERE.parents[6]
PARKOUR = REPO / "parkour"
FWD_PKG = PKG.parent / "crab_hex_forward_task"
HARNESS = FWD_PKG / "scripts/eval_crab_hex_gait.py"      # shared harness: one harness serves all three tasks
HARNESS_DIR = HARNESS.parent
MANIFEST = PKG / "experiments/eval/scenarios_omni.yaml"  # where the reference stamp's scenario_id lives
P2R_LAUNCHER = EXPERIMENTS / "2026-10-07_omni_phase2" / "scripts" / "p2r_mode_eval_crab_hex_gait.py"
SN_LAUNCHER = HERE / "sn_stand_net_eval_crab_hex_gait.py"
COMMON_PATH = HERE / "sn_common.py"
CORE_PATH = HERE / "sn_video_core.py"
LOG_ROOT = PARKOUR / "logs/rsl_rl"
DEFAULT_CKPT = LOG_ROOT / "crab_hex_omni_teacher/2026-10-07_07-50-10_seed18_X08_p2a_A15pB/model_19994.pt"
DEFAULT_STAND_CKPT = PARKOUR / "outputs/omni_plan/stand_network/runs/SN_Amask_s17/model_4000.pt"
DEFAULT_EVAL_RUN = PARKOUR / "outputs/omni_plan/phase2/lr/SN_Amask_s17/suite__omni/seed001"
DEFAULT_MANIFEST = CAMPAIGN / "videos/scenarios_sn_video.yaml"
DEFAULT_ROW = "sn_random__omni"
#: mp4s are bulk artifacts: parkour/logs/, never the campaign tree (record_categories' convention)
DEFAULT_OUT = PARKOUR / "logs/videos/sn_stand_network"


def _load_by_path(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    old = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.dont_write_bytecode = old
    return mod


#: torch-free at import, all three (the campaign launcher's rule L6): loaded before Kit
SN = _load_by_path(COMMON_PATH, "_snv_sn_common")
CORE = _load_by_path(CORE_PATH, "_snv_core")
P2L = SN.load_by_path(P2R_LAUNCHER, "_snv_p2r")             # a PRIVATE instance; only its functions are called

# cli_args exactly as the harness loads it (never put parkour/scripts on sys.path: it shadows pip rsl_rl).
_cli_spec = importlib.util.spec_from_file_location("parkour_rsl_rl_cli_args", PARKOUR / "scripts/rsl_rl/cli_args.py")
cli_args = importlib.util.module_from_spec(_cli_spec)
_cli_spec.loader.exec_module(cli_args)

from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser(description="Routed S1 + stand-net video, recorded through the eval harness's path.")
parser.add_argument("--out-dir", type=str, default=str(DEFAULT_OUT), help=f"Output directory (default {DEFAULT_OUT}).")
parser.add_argument("--prefix", type=str, default="sn_random", help="File prefix: <prefix>_sequence.mp4 / _sanity.json.")
parser.add_argument(
    "--eval-run", type=str, default=str(DEFAULT_EVAL_RUN),
    help="The reference stand_net suite eval: its stamp dir, its run_meta.json, or a seed dir (newest complete stamp).",
)
parser.add_argument("--manifest", type=str, default=str(DEFAULT_MANIFEST), help="Manifest holding the --sequence row.")
parser.add_argument("--sequence", type=str, default=DEFAULT_ROW, metavar="ROW_ID",
                    help=f"The kind:schedule world-frame row to record (default {DEFAULT_ROW}).")
parser.add_argument("--stand-ckpt", type=str, default=str(DEFAULT_STAND_CKPT), help="The stand network checkpoint.")
parser.add_argument("--stand-mask", type=int, choices=(0, 1), default=1,
                    help="The actor-side stand mask the stand net is run with (A-mask 1 / A-full 0).")
parser.add_argument("--stand-rule", choices=CORE.STAND_RULES, default="schedule",
                    help="Route on the schedule's stand holds (default) or on the eval's latched stand slot.")
parser.add_argument("--stand-tag", type=str, default=None,
                    help="Caption name of the stand network (default: from the loaded checkpoint's infos.sn_dagger, "
                         "e.g. 'A-mask s17').")
parser.add_argument("--allow-reference-mismatch", action="store_true",
                    help="Record even if a checkpoint / mask / harness / sn_common / R-module sha differs from the "
                         "reference stamp's (default: refuse; the mismatch is recorded in the sanity JSON).")
parser.add_argument("--label", type=str, default=None,
                    help="Caption title (default '<walker tag> + stand net <stand tag>', both from the loaded "
                         "checkpoints, e.g. 'S1 s18 + stand net A-mask s17').")
# record_categories' camera: a rear-right 3/4 view from ~6.6 m looking at mid-leg height; 'task' keeps the task's.
parser.add_argument("--cam-eye", type=str, default="-5.0,-4.0,1.6",
                    help="Viewer eye offset 'x,y,z' from the robot root (world-aligned), or 'task'.")
parser.add_argument("--cam-lookat", type=str, default="0.0,0.0,-0.45",
                    help="Viewer look-at offset 'x,y,z' from the robot root, or 'task'.")
parser.add_argument("--quality", type=float, default=8.0, help="imageio/libx264 quality 0-10 (repo streamer uses 8).")
parser.add_argument("--arrows", dest="arrows", action="store_true", default=True,
                    help="Draw the command term's own arrows (default): green travel, orange facing, blue achieved.")
parser.add_argument("--no-arrows", dest="arrows", action="store_false",
                    help="Reproduce the eval's visual state exactly (markers off).")
parser.add_argument("--no-captions", action="store_true")
parser.add_argument("--no-stills", action="store_true", help="Skip the per-hold PNG stills.")
parser.add_argument("--trace-npz", action="store_true", help="Also dump every recorded per-step array.")
parser.add_argument("--rss-abort-gb", type=float, default=20.0, help="Abort if RSS exceeds this (DDC balloon).")
parser.add_argument("--disable_fabric", action="store_true", default=False)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.checkpoint is None:
    args_cli.checkpoint = str(DEFAULT_CKPT)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _resolve_eval_run(text: str) -> Path:
    """A run_meta.json, a stamp dir, or a seed dir -> the run_meta.json (newest stamp with run_meta + complete.json)."""
    p = Path(text)
    if p.is_file():
        return p
    if (p / "run_meta.json").is_file():
        return p / "run_meta.json"
    if p.is_dir():
        stamps = sorted(d for d in p.iterdir() if d.is_dir() and (d / "run_meta.json").is_file()
                        and (d / "complete.json").is_file())
        if stamps:
            return stamps[-1] / "run_meta.json"
    raise SystemExit(f"--eval-run {text!r}: no run_meta.json (nor a complete stamp under it)")


# ------------------------------------------------------------------ pre-Kit checks (no torch)
RUN_META_PATH = _resolve_eval_run(args_cli.eval_run)
RUN_META = json.loads(RUN_META_PATH.read_text())
WANT_ENV = {k: str(v) for k, v in RUN_META["env_vars"].items()}
_env_why = SN.env_reasons(WANT_ENV)
if _env_why:
    raise SystemExit("the reference stamp's env fails the campaign launcher's D5 / L10 refusals:\n  " + "\n  ".join(_env_why))
for _p, _what in ((Path(args_cli.checkpoint), "--checkpoint"), (Path(args_cli.stand_ckpt), "--stand-ckpt"),
                  (Path(args_cli.manifest), "--manifest")):
    if not _p.is_file():
        raise SystemExit(f"{_what} not found: {_p}")
S1_SHA = _sha256(Path(args_cli.checkpoint))
STAND_SHA = _sha256(Path(args_cli.stand_ckpt))
HARNESS_SHA = _sha256(HARNESS)
_pm = RUN_META.get("p2r_mode") or {}
_snb = _pm.get(SN.BLOCK_KEY) or {}
REFERENCE_CHECKS = {
    "reference_variant": (_pm.get("variant"), SN.VARIANT),
    "stand_mask": (_pm.get(SN.MASK_KEY), int(args_cli.stand_mask)),
    "stand_ckpt_sha256": ((_pm.get("specialist") or {}).get("sha256"), STAND_SHA),
    "s1_ckpt_sha256": (RUN_META.get("checkpoint_sha256"), S1_SHA),
    "harness_sha256": (_pm.get("harness_sha256"), HARNESS_SHA),
    "sn_common_sha256": (_snb.get("common_sha256"), SN.sha256_file(COMMON_PATH)),
    "restance_module_sha256": (_snb.get("restance_module_sha256"), SN.sha256_file(SN.RESTANCE_MODULE)),
    "forward": (_snb.get("forward"), SN.FORWARD),
    "hist_encoding": (_snb.get("hist_encoding"), True),
    "mask_fn": (_snb.get("mask_fn"), SN.MASK_FN),
}
REFERENCE_MISMATCH = {k: {"reference": a, "here": b} for k, (a, b) in REFERENCE_CHECKS.items() if a != b}
if REFERENCE_MISMATCH and not args_cli.allow_reference_mismatch:
    raise SystemExit(f"this recording would not be the deployment the reference stamp {RUN_META_PATH} evaluated: "
                     f"{REFERENCE_MISMATCH} (pass --allow-reference-mismatch to record anyway)")


def _rss_gb() -> float:
    try:
        with open("/proc/self/status") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024.0 / 1024.0
    except OSError:
        pass
    return float("nan")


def _cgroup_peak_gb() -> float | None:
    try:
        rel = Path("/proc/self/cgroup").read_text().strip().split("::", 1)[1]
        p = Path("/sys/fs/cgroup") / rel.lstrip("/") / "memory.peak"
        return int(p.read_text().strip()) / 1024**3
    except Exception:  # noqa: BLE001
        return None


# ------------------------------------------------------------------ env vars BEFORE Kit boots
# Writing KRABBY_* AFTER Kit has started is a known way to flip a same-seed eval between two
# repeatable outcomes, so the whole replacement happens here, before AppLauncher.
_dropped = sorted(k for k in os.environ if k.startswith("KRABBY_") and os.environ[k] != WANT_ENV.get(k))
for _k in [k for k in os.environ if k.startswith("KRABBY_")]:
    del os.environ[_k]
os.environ.update(WANT_ENV)
print(f"[SNV] reference stamp {RUN_META_PATH}", flush=True)
print(f"[SNV] KRABBY_* env replaced by run_meta env_vars ({len(WANT_ENV)} vars); "
      f"pre-existing differing/extra vars dropped: {_dropped or 'none'}", flush=True)
for _k, _v in sorted(WANT_ENV.items()):
    print(f"[SNV]   {_k}={_v}", flush=True)
print(f"[SNV] reference checks: {'all equal' if not REFERENCE_MISMATCH else f'MISMATCH (allowed): {REFERENCE_MISMATCH}'}",
      flush=True)

_t_boot = time.time()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
print(f"[SNV] Kit up in {time.time() - _t_boot:.1f}s, RSS {_rss_gb():.2f} GB", flush=True)
if _rss_gb() > args_cli.rss_abort_gb:
    print(f"[SNV][ABORT] RSS {_rss_gb():.1f} GB > {args_cli.rss_abort_gb} GB right after Kit start -- "
          "suspect DerivedDataCache corruption; not stepping.", flush=True)
    simulation_app.close()
    sys.exit(3)


def _on_sigterm(signum, _frame):
    # systemctl stop / the launcher's timeout send SIGTERM: unwind through the finally blocks (the mp4 writer is
    # closed, simulation_app.close() runs) instead of dying mid-write.
    raise SystemExit(128 + int(signum))


signal.signal(signal.SIGTERM, _on_sigterm)

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

# The harness's sys.path surgery, verbatim in effect.
_parkour_scripts = PARKOUR / "scripts"
for _p in (str(_parkour_scripts), str(_parkour_scripts / "rsl_rl"), str(PARKOUR / "parkour_tasks")):
    while _p in sys.path:
        sys.path.remove(_p)
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(PARKOUR))
sys.path.insert(0, str(PARKOUR / "parkour_tasks"))
if str(HARNESS_DIR) not in sys.path:
    sys.path.insert(0, str(HARNESS_DIR))

import parkour_tasks  # noqa: F401,E402
from gait_eval import schedule as S  # noqa: E402
from isaaclab.utils.assets import retrieve_file_path  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from parkour_isaaclab.envs.mdp.parkour_commands.uniform_parkour_command import (  # noqa: E402
    UniformParkourCommand,
)
from scripts.rsl_rl.runner_factory import agent_cfg_to_train_dict, make_on_policy_runner  # noqa: E402
from scripts.rsl_rl.vecenv_wrapper import ParkourRslRlVecEnvWrapper  # noqa: E402


def _load_harness_functions() -> dict:
    """Execute the harness's OWN definitions (not copies) of the env-override / command-pin code."""
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
        raise SystemExit(f"harness {HARNESS} no longer defines {missing}; update this recorder")
    mod = ast.fix_missing_locations(ast.Module(body=body, type_ignores=[]))
    code = compile(mod, str(HARNESS), "exec", flags=_future.annotations.compiler_flag, dont_inherit=True)
    ns = {"S": S, "torch": torch, "types": types, "UniformParkourCommand": UniformParkourCommand,
          "__name__": "eval_crab_hex_gait_extract"}
    exec(code, ns)  # noqa: S102 - the repo's own harness source
    return ns


def _wrap(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


def _deg(a: float) -> float:
    return math.degrees(a)


def _circmean(a: np.ndarray) -> float:
    if a.size == 0:
        return float("nan")
    return float(np.arctan2(np.sin(a).mean(), np.cos(a).mean()))


def _parse_vec3(text: str | None):
    if text is None or text.strip().lower() == "task":
        return None
    vals = tuple(float(x) for x in text.split(","))
    if len(vals) != 3:
        raise SystemExit(f"expected 'x,y,z', got {text!r}")
    return vals


def _resolve_sequence_row(scenarios: list, manifest: Path) -> tuple:
    """``--sequence ROW_ID`` -> the schedule row, built as the harness builds a non-suite row (record_categories'
    ``_resolve_sequence_row`` without the subset / retarget options): ``S.load_manifest`` validated it, ``episodes``
    1, ``episode_length_s`` = prologue + schedule + 2 s, then ``S.validate_scenario`` (a failure aborts)."""
    by_id = {s.id: s for s in scenarios}
    row_id = args_cli.sequence
    if row_id not in by_id:
        sched_rows = [s.id for s in scenarios if s.kind == S.SCHEDULE_KIND]
        raise SystemExit(f"--sequence {row_id!r} not in {manifest} (schedule rows: {sched_rows})")
    scenario = by_id[row_id]
    if scenario.kind != S.SCHEDULE_KIND:
        raise SystemExit(f"--sequence {row_id!r} is a {scenario.kind!r} row; this recorder needs a "
                         f"{S.SCHEDULE_KIND!r} row")
    if scenario.command_frame != "world":
        raise SystemExit(f"--sequence {row_id!r} is a body-frame row; this recorder drives the world-frame "
                         "schedule_hook override only")
    # ---- env composition stays the reference SUITE eval's; its suite-only checks do not apply.
    if scenario.task != RUN_META["task"]:
        raise SystemExit(f"sequence row task {scenario.task} != reference run_meta task {RUN_META['task']}")
    ref = by_id.get(RUN_META["scenario_id"])
    ref_src = manifest
    if ref is None and manifest.resolve() != MANIFEST.resolve():
        ref = {s.id: s for s in S.load_manifest(MANIFEST)[0]}.get(RUN_META["scenario_id"])
        ref_src = MANIFEST
    if ref is None:
        raise SystemExit(f"reference row {RUN_META['scenario_id']!r} (run_meta scenario_id) is in neither "
                         f"{manifest} nor {MANIFEST}; cannot check its terrain_label")
    if scenario.terrain_label != ref.terrain_label:
        raise SystemExit(f"sequence row terrain_label {scenario.terrain_label!r} != reference row "
                         f"{ref.id!r} terrain_label {ref.terrain_label!r} ({ref_src})")
    clash = {k: (v, WANT_ENV.get(k)) for k, v in scenario.env_vars.items() if WANT_ENV.get(k) != str(v)}
    if scenario.env_vars:
        raise SystemExit(f"sequence row {row_id!r} carries an env: block {scenario.env_vars} (clashes {clash}); the "
                         "reference stamp ran none, and a row env would be written before Kit here but after Kit by "
                         "the harness (the process-state class) -- refused")
    rm_ell = RUN_META.get("speed_ellipse") or {}
    if (rm_ell and scenario.get_speed_ellipse() != (float(rm_ell["fwd"]), float(rm_ell["lat"]), float(rm_ell["back"]))
            or (RUN_META.get("ang_vel_max") is not None
                and float(scenario.ang_vel_max) != float(RUN_META["ang_vel_max"]))):
        raise SystemExit(f"sequence row speed_ellipse / ang_vel_max {scenario.get_speed_ellipse()} / "
                         f"{scenario.ang_vel_max} != the reference stamp's {rm_ell} / {RUN_META.get('ang_vel_max')}")
    rm_pro = (RUN_META.get("suite") or {}).get("prologue_s")
    if rm_pro is not None and float(scenario.get_default("prologue_s")) != float(rm_pro):
        raise SystemExit(f"sequence row prologue_s {scenario.get_default('prologue_s')} != the reference's {rm_pro}")
    print(f"[SNV] SEQUENCE: row {row_id!r} ({manifest}); checked: task {scenario.task} == run_meta task, "
          f"terrain_label {scenario.terrain_label!r} == reference row {ref.id!r}'s ({ref_src}), no row env: block, "
          "speed_ellipse / ang_vel_max / prologue_s == the reference stamp's.", flush=True)
    if scenario.rotate_holds_by_env:
        raise SystemExit(f"--sequence {row_id!r} sets rotate_holds_by_env; refused (one env, one timeline)")
    manifest_holds = [{"label": h.label, "hold_s": float(h.hold_s)} for h in scenario.schedule]
    scenario.schedule = [dataclasses_replace(h) for h in scenario.schedule]
    scenario.episodes = 1
    scenario.episode_length_s = float(scenario.total_duration_s) + 2.0
    try:
        S.validate_scenario(scenario)
    except S.ManifestError as exc:
        raise SystemExit(f"sequence row {row_id!r} fails the harness validator: {exc}") from exc
    print("[SNV] sequence row passes the harness validator", flush=True)
    info = {
        "row_id": row_id, "manifest": str(manifest), "manifest_sha256": _sha256(manifest), "reference_row": ref.id,
        "reference_row_manifest": str(ref_src), "terrain_label": scenario.terrain_label,
        "manifest_holds": manifest_holds, "validator": "pass",
    }
    return scenario, info


def dataclasses_replace(h):
    import dataclasses  # noqa: PLC0415

    return dataclasses.replace(h)


def _segment_bounds(compiled) -> tuple[np.ndarray, list[int], list[int]]:
    """Env 0's ``segment_id`` plus each hold's ``[start, end)`` step (prologue = steps before start[0])."""
    seg = np.asarray(compiled.segment_id[:, 0]).astype(np.int64)
    n = len(compiled.hold_labels)
    starts = []
    for i in range(n):
        hit = np.nonzero(seg == i)[0]
        if hit.size == 0:
            raise SystemExit(f"compiled schedule has no step for hold {i} {compiled.hold_labels[i]!r}")
        starts.append(int(hit[0]))
    ends = starts[1:] + [int(compiled.n_steps)]
    return seg, starts, ends


# ------------------------------------------------------------------ captions
_CAPTION_BACKEND = None
if not args_cli.no_captions:
    try:
        import cv2  # noqa: E402

        _CAPTION_BACKEND = "cv2"
    except Exception:  # noqa: BLE001
        try:
            from PIL import Image, ImageDraw  # noqa: E402,F401

            _CAPTION_BACKEND = "pil"
        except Exception:  # noqa: BLE001
            _CAPTION_BACKEND = None


def _caption(frame: np.ndarray, lines: list[tuple[str, tuple[int, int, int]]]) -> np.ndarray:
    if _CAPTION_BACKEND is None or not lines:
        return frame
    frame = np.ascontiguousarray(frame[:, :, :3]).copy()
    h, w = frame.shape[:2]
    scale = max(0.5, w / 1280.0 * 0.62)
    line_h = int(round(30 * scale / 0.62))
    box_h = 10 + line_h * len(lines)
    thick = 2 if scale >= 0.6 else 1
    if _CAPTION_BACKEND == "cv2":
        box_w = min(w, 20 + max(cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)[0][0] for t, _c in lines))
    else:
        box_w = min(w, int(900 * scale / 0.62))
    frame[:box_h, :box_w] = (frame[:box_h, :box_w].astype(np.uint16) * 2 // 5).astype(np.uint8)
    if _CAPTION_BACKEND == "cv2":
        for i, (text, color) in enumerate(lines):
            y = 8 + line_h * (i + 1) - int(8 * scale / 0.62)
            cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)
        return frame
    from PIL import Image, ImageDraw

    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    for i, (text, color) in enumerate(lines):
        draw.text((10, 6 + line_h * i), text, fill=color)
    return np.asarray(img)


WHITE, YELLOW, CYAN, RED = (255, 255, 255), (255, 230, 90), (120, 230, 255), (255, 80, 60)
#: the acting-network line (RGB, as every caption colour here)
STANDNET_COLOR, S1_COLOR = (255, 120, 235), (170, 255, 170)
ARROW_LEGEND = "arrows: GREEN = commanded travel   ORANGE = commanded facing   BLUE = achieved velocity"


class _SlotRule:
    """The eval's stand signal: P2L's ``P2RMode.rederive_on_reset`` (the SAME function object, P2L:441-456) on a
    holder carrying only what it reads -- re-derive the turning bit on reset rows, return ``cmd_term.stand_mask()``."""

    rederive_on_reset = P2L.P2RMode.rederive_on_reset

    def __init__(self, cmd_term, unwrapped):
        self.cmd_term = cmd_term
        self.unwrapped = unwrapped
        self.resets_rederived = 0


def _build_networks(env, agent_cfg, device):
    """S1's runner + the stand runner, exactly as the campaign's eval launcher builds them (module docstring)."""
    train_cfg = agent_cfg_to_train_dict(agent_cfg)
    pristine = copy.deepcopy(train_cfg)
    if (pristine.get("algorithm") or {}).get("class_name") == "DistillationWithExtractor":
        raise SystemExit("distillation (student) runners are refused (teacher heads only)")
    if pristine.get("empirical_normalization"):
        raise SystemExit("empirical_normalization is not supported (P2L refuses it too)")
    runner, template = P2L.build_with_template(make_on_policy_runner, env, train_cfg, None, device)
    standin = P2L.StandInEnv(env, template)
    before = P2L.save_rng()
    try:
        st_runner, st_policy, st_est = P2L._load_extra(make_on_policy_runner, standin, pristine,
                                                       str(args_cli.stand_ckpt), device)
    finally:
        P2L.restore_rng(before)
    rng_restored = P2L.rng_equal(before, P2L.save_rng())
    if not rng_restored:
        raise SystemExit("RNG state not restored after building the stand runner")
    unwrapped = env.unwrapped
    cmd_term = unwrapped.command_manager.get_term("base_velocity")
    latch_pre = P2L.arm_slot_latch(cmd_term, int(unwrapped.num_envs), cmd_term.slot_id.device)
    return runner, pristine, st_runner, st_policy, st_est, rng_restored, latch_pre


def _check_stand_ckpt(runner, st_runner, st_est) -> tuple[dict, dict | None]:
    """``bind_s1`` + ``trained_arm`` (the campaign launcher's refusals) + the loaded stand actor == the file's actor.
    Post-Kit, before the first step. Returns the check record and the checkpoint's ``infos.sn_dagger`` (or None)."""
    ok, why = SN.state_dicts_equal(runner.alg.estimator.state_dict(), st_est.state_dict())
    if not ok:
        raise SystemExit(f"the stand checkpoint's estimator differs from S1's ({why}); refused (PLAN D1 / map L5)")
    ck = torch.load(str(args_cli.stand_ckpt), map_location="cpu", weights_only=False)
    infos = ck.get("infos") if isinstance(ck, dict) else None
    sd = infos.get("sn_dagger") if isinstance(infos, dict) else None
    if sd is not None:
        tm = sd.get("stand_mask") if isinstance(sd, dict) else None
        if type(tm) is not int or tm not in (0, 1):
            raise SystemExit(f"the stand checkpoint's infos.sn_dagger.stand_mask is {tm!r}, not an int 0 / 1")
        if tm != int(args_cli.stand_mask):
            raise SystemExit(f"--stand-mask {args_cli.stand_mask} but the checkpoint was trained with stand_mask {tm} "
                             f"(arm {sd.get('arm')!r}); an arm runs with the mask it was trained with")
        trained, kind = int(tm), "sn_dagger"
        s1_trained = sd.get("s1_sha256")
        if s1_trained != S1_SHA and not args_cli.allow_reference_mismatch:
            raise SystemExit(f"the stand net was trained beside S1 sha {str(s1_trained)[:12]}, not --checkpoint's "
                             f"{S1_SHA[:12]} (pass --allow-reference-mismatch to record anyway)")
    else:
        if STAND_SHA != S1_SHA:
            raise SystemExit("the stand checkpoint carries no infos.sn_dagger and is not S1 itself; refused")
        trained, kind, s1_trained = None, "s1_identity", None
    actor_file = {k: v for k, v in ck["model_state_dict"].items() if k.startswith("actor.")}
    actor_live = {k: v for k, v in st_runner.alg.policy.state_dict().items() if k.startswith("actor.")}
    ok, why = SN.state_dicts_equal(actor_file, actor_live)
    if not actor_file or not ok:
        raise SystemExit(f"the loaded stand actor differs from {args_cli.stand_ckpt}'s ({why or 'no actor.* keys'})")
    return ({"estimator_equals_s1": True, "trained_stand_mask": trained, "stand_ckpt_kind": kind,
             "trained_beside_s1_sha256": s1_trained, "actor_params_equal_file": True,
             "sn_dagger_arm": (sd or {}).get("arm") if isinstance(sd, dict) else None,
             "sn_dagger_seed": (sd or {}).get("seed") if isinstance(sd, dict) else None},
            sd if isinstance(sd, dict) else None)


def _caption_tags(stand_check: dict, sn_dagger: dict | None, resume_path) -> dict:
    """The caption names: ``--stand-tag`` / ``--label`` if given, else derived from the LOADED checkpoints
    (``sn_video_core.stand_tag_from_ckpt`` / ``s1_tag_from_ckpt``)."""
    is_s1 = stand_check["stand_ckpt_kind"] == "s1_identity" or stand_check["trained_beside_s1_sha256"] == S1_SHA
    stand_tag = args_cli.stand_tag or CORE.stand_tag_from_ckpt(sn_dagger, args_cli.stand_ckpt)
    s1_tag = CORE.s1_tag_from_ckpt(resume_path, is_stand_nets_s1=is_s1)
    return {"stand_tag": stand_tag, "s1_tag": s1_tag, "label": args_cli.label or f"{s1_tag} + stand net {stand_tag}",
            "stand_tag_source": "--stand-tag" if args_cli.stand_tag else "stand checkpoint infos.sn_dagger",
            "label_source": "--label" if args_cli.label else "derived from the loaded checkpoints"}


def main() -> None:
    out_dir = Path(args_cli.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    H = _load_harness_functions()
    print(f"[SNV] harness functions extracted from {HARNESS} (sha256 {HARNESS_SHA[:16]}...)", flush=True)

    manifest = Path(args_cli.manifest)
    scenarios, _ = S.load_manifest(manifest)
    scenario, seq_info = _resolve_sequence_row(scenarios, manifest)
    kinds = CORE.hold_kinds(scenario.schedule, scenario.spawn_yaw_rad())
    flags = CORE.hold_stand_flags(scenario.schedule, scenario.spawn_yaw_rad())

    # ---- env, composed as the harness composes it
    env_cfg = parse_env_cfg(scenario.task, device=args_cli.device, num_envs=1, use_fabric=not args_cli.disable_fabric)
    applied_overrides = H["_apply_env_overrides"](env_cfg, scenario)
    print(f"[SNV] harness env overrides applied: {applied_overrides} (reference run: {RUN_META['env_overrides']})",
          flush=True)
    env_cfg.commands.base_velocity.debug_vis = bool(args_cli.arrows)
    _parkour_vis = getattr(getattr(env_cfg, "parkours", None), "base_parkour", None)
    print(f"[SNV] command debug arrows: {'ON' if args_cli.arrows else 'OFF'} (parkours.base_parkour.debug_vis stays "
          f"{'n/a' if _parkour_vis is None else _parkour_vis.debug_vis})", flush=True)
    eye, lookat = _parse_vec3(args_cli.cam_eye), _parse_vec3(args_cli.cam_lookat)
    if eye is not None:
        env_cfg.viewer.eye = eye
    if lookat is not None:
        env_cfg.viewer.lookat = lookat
    env_cfg.viewer.env_index = 0
    print(f"[SNV] camera: origin_type={env_cfg.viewer.origin_type} asset={env_cfg.viewer.asset_name} "
          f"eye={tuple(env_cfg.viewer.eye)} lookat={tuple(env_cfg.viewer.lookat)} "
          f"resolution={tuple(env_cfg.viewer.resolution)}", flush=True)

    agent_cfg = cli_args.parse_rsl_rl_cfg(scenario.task, args_cli)
    env = ParkourRslRlVecEnvWrapper(
        gym.make(scenario.task, cfg=env_cfg, render_mode="rgb_array"),
        clip_actions=agent_cfg.clip_actions,
    )
    unwrapped = env.unwrapped
    device = env.device
    dt = float(unwrapped.step_dt)
    print(f"[SNV] env up: dt={dt} max_episode_length={int(unwrapped.max_episode_length)} steps, RSS {_rss_gb():.2f} GB",
          flush=True)
    if _rss_gb() > args_cli.rss_abort_gb:
        raise SystemExit(f"RSS {_rss_gb():.1f} GB > {args_cli.rss_abort_gb} GB before the first step -- aborting")

    krabby_now = {k: v for k, v in sorted(os.environ.items()) if k.startswith("KRABBY_")}
    if krabby_now != WANT_ENV:
        raise SystemExit(f"KRABBY_* env after import differs from run_meta: now={krabby_now} want={WANT_ENV}")
    print("[SNV] KRABBY_* env after task import == reference run_meta env_vars", flush=True)

    robot = unwrapped.scene["robot"]
    usd_spawned = str(getattr(getattr(robot.cfg, "spawn", None), "usd_path", "<unknown>"))
    if Path(usd_spawned).name != Path(RUN_META["usd_path"]).name:
        raise SystemExit(f"spawned {usd_spawned} but the reference run spawned {RUN_META['usd_path']}")
    print(f"[SNV] plant usd {usd_spawned} (reference {RUN_META['usd_path']}, plant {RUN_META['plant']})", flush=True)

    # ---- the two networks, built as the campaign's eval launcher builds them
    runner, pristine, st_runner, st_policy, st_est, rng_restored, latch_pre = _build_networks(env, agent_cfg, device)
    resume_path = retrieve_file_path(str(args_cli.checkpoint))
    runner.load(resume_path)
    estimator_paras = pristine["estimator"]
    num_prop = int(estimator_paras["num_prop"])
    num_scan = int(estimator_paras["num_scan"])
    num_priv_explicit = int(estimator_paras["num_priv_explicit"])
    policy = runner.get_inference_policy(device=device)
    estimator = runner.get_estimator_inference_policy(device=device)
    stand_check, sn_dagger = _check_stand_ckpt(runner, st_runner, st_est)
    tags = _caption_tags(stand_check, sn_dagger, resume_path)
    print(f"[SNV] S1 {resume_path} sha256 {S1_SHA[:16]}...; stand net {args_cli.stand_ckpt} sha256 {STAND_SHA[:16]}... "
          f"mask {args_cli.stand_mask} ({stand_check}); rng_restored={rng_restored} latch_preexisting={latch_pre}; "
          f"num_prop={num_prop} priv=[{num_prop + num_scan}:{num_prop + num_scan + num_priv_explicit}] "
          f"clip_actions={agent_cfg.clip_actions}; stand rule: {args_cli.stand_rule}", flush=True)
    print(f"[SNV] caption names: walker '{tags['s1_tag']}', stand net '{tags['stand_tag']}' "
          f"({tags['stand_tag_source']}), title '{tags['label']}' ({tags['label_source']})", flush=True)

    cmd_term = unwrapped.command_manager.get_term("base_velocity")
    if not isinstance(cmd_term, UniformParkourCommand):
        raise SystemExit(f"base_velocity must be UniformParkourCommand, got {type(cmd_term)}")
    if args_cli.arrows and not hasattr(cmd_term, "travel_cmd_visualizer"):
        raise SystemExit("--arrows: the command term built no visualizers -- debug_vis did not reach it "
                         "(is this the omni command term, and was --enable_cameras passed?)")
    observed_slots, slots_src = H["_observed_command_slots"](unwrapped)
    print(f"[SNV] observed command slots {observed_slots} ({slots_src}); reference {RUN_META['observed_command_slots']}",
          flush=True)
    cam_names = [n for n in robot.joint_names if str(n).endswith(CORE.CAM_SUFFIX)]
    if len(cam_names) != 6:
        raise SystemExit(f"expected 6 cam joints (*{CORE.CAM_SUFFIX}), got {cam_names}")
    cam_cols = torch.tensor([list(robot.joint_names).index(n) for n in cam_names], device=device)

    cam = getattr(unwrapped, "viewport_camera_controller", None)
    if cam is None:
        print("[SNV][WARN] no viewport camera controller (is --enable_cameras set?) -- camera cannot follow", flush=True)

    def _render() -> np.ndarray:
        if cam is not None and env_cfg.viewer.origin_type == "asset_root":
            cam.update_view_to_asset_root(env_cfg.viewer.asset_name)  # no one-render lag after a respawn
        return unwrapped.render()

    import imageio

    fps = int(round(1.0 / dt))
    if abs(fps * dt - 1.0) > 1e-6:
        print(f"[SNV][WARN] 1/dt={1.0 / dt} is not an integer fps; video plays at {fps} fps", flush=True)
    combined_path = out_dir / f"{args_cli.prefix}_sequence.mp4"
    stills_dir = out_dir / "stills"
    if not args_cli.no_stills:
        stills_dir.mkdir(parents=True, exist_ok=True)
    combined = imageio.get_writer(str(combined_path), fps=fps, codec="libx264", quality=args_cli.quality,
                                  macro_block_size=None, ffmpeg_log_level="error")
    print(f"[SNV] captions: {_CAPTION_BACKEND or 'NONE'}; fps {fps}; -> {combined_path}", flush=True)

    for i in range(60):
        f0 = _render()
        if f0 is not None and f0.size and float(f0.mean()) > 1.0:
            print(f"[SNV] renderer warm after {i + 1} render(s); frame {f0.shape}", flush=True)
            break

    ctx = types.SimpleNamespace(
        H=H, env=env, unwrapped=unwrapped, robot=robot, cmd_term=cmd_term, device=device, dt=dt, fps=fps,
        observed_slots=observed_slots, estimator=estimator, policy=policy, num_prop=num_prop,
        priv_lo=num_prop + num_scan, priv_hi=num_prop + num_scan + num_priv_explicit,
        stand_policy=st_policy, stand_estimator=st_est, stand_mask=int(args_cli.stand_mask),
        slot_rule=_SlotRule(cmd_term, unwrapped), cam_cols=cam_cols,
        render=_render, combined=combined, combined_path=combined_path, stills_dir=stills_dir, out_dir=out_dir,
        env_cfg=env_cfg, applied_overrides=applied_overrides, resume_path=resume_path, tags=tags,
        build={"rng_restored": bool(rng_restored), "latch_preexisting": latch_pre, **stand_check},
    )
    try:
        _run_sequence(ctx, scenario, seq_info, kinds, flags)
    finally:
        if cam is not None:
            # Drop the camera's post-update subscription first: otherwise it fires once more during
            # simulation_app.close() against the already-closed scene (a harmless but noisy traceback).
            cam._viewport_camera_update_handle = None
        env.close()


REC_KEYS = ("row_speed", "row_dir", "row_face", "speed_cmd", "travel_dir_w", "heading_target",
            "cmd_bx", "cmd_by", "cmd_wz", "heading_pre", "x_pre", "y_pre",
            "x_post", "y_post", "heading_post", "vx_w", "vy_w", "done", "fail",
            "stand_schedule", "stand_slot", "routed_stand", "route_ok", "st_differs_s1", "cam_abs_max")


def _new_rec(T: int) -> dict:
    return {key: np.zeros(T, dtype=np.float64) for key in REC_KEYS}


def _harness_step(ctx, override_state: dict, obs, t: int, trace, stand_sched: bool, rule: str):
    """ONE control step of the harness's world-frame loop body, with the routed pair in place of the one policy.

    As record_categories: advance ``override_state['step']``, check the schedule state and the observed command
    slots, write S1's estimate into the priv-explicit slice. Then the campaign launcher's step: the eval's stand
    signal (``rederive_on_reset`` -> the latched slot), the routed action (``sn_video_core.route_step``), and the
    checks (applied == the routed network's action, bitwise; stand net != S1). Pre-step cam angles (the harness
    records ``joint_pos`` pre-step). Then ``env.step``."""
    cmd_term, robot, H = ctx.cmd_term, ctx.robot, ctx.H
    override_state["step"] = t + 1
    obs_reflects_cmd = cmd_term.vel_command_b.clone()
    row_t = override_state["row"](t)
    sched_dev = max(
        (cmd_term.travel_dir_w - row_t.dir).abs().max().item(),
        (cmd_term.speed_cmd - row_t.speed).abs().max().item(),
        H["_wrap_to_pi"](cmd_term.heading_target - row_t.face).abs().max().item(),
    )
    cmd_dev = 0.0
    for _slot, _channel in ctx.observed_slots.items():
        cmd_dev = max(cmd_dev, (obs[:, _slot] - obs_reflects_cmd[:, _channel]).abs().max().item())
    obs[:, ctx.priv_lo: ctx.priv_hi] = ctx.estimator.inference(obs[:, :ctx.num_prop])
    slot = ctx.slot_rule.rederive_on_reset()                  # (N,) bool, the eval's latched stand slot
    stand_slot = bool(slot[0].item())
    route = stand_sched if rule == "schedule" else stand_slot
    rows = torch.full((int(obs.shape[0]),), bool(route), dtype=torch.bool, device=obs.device)
    actions, a_s1, a_st = CORE.route_step(SN, obs, ctx.policy, ctx.stand_policy, ctx.stand_estimator, rows,
                                          p0=ctx.priv_lo, p1=ctx.priv_hi, num_prop=ctx.num_prop, mask=ctx.stand_mask)
    route_ok = bool(torch.equal(actions, a_st if route else a_s1))
    st_differs = not bool(torch.equal(a_st, a_s1))
    trace.update(obs.detach().cpu().numpy().tobytes())
    trace.update(actions.detach().cpu().numpy().tobytes())
    cam_abs = torch.abs(torch.remainder(robot.data.joint_pos[0, ctx.cam_cols] + math.pi, 2.0 * math.pi) - math.pi)

    pre = torch.cat([
        cmd_term.speed_cmd[:1], cmd_term.travel_dir_w[:1], cmd_term.heading_target[:1],
        cmd_term.vel_command_b[0, :3], robot.data.heading_w[:1], robot.data.root_pos_w[0, :2],
    ]).double().cpu().numpy()
    obs, _, dones, _extras = ctx.env.step(actions)
    try:
        failed = bool(ctx.unwrapped.termination_manager.get_term("crab_failure")[0].item())
    except (KeyError, AttributeError):
        failed = False
    done = bool(dones.reshape(-1)[0].item())
    post = torch.cat([
        robot.data.root_pos_w[0, :2], robot.data.heading_w[:1], robot.data.root_lin_vel_w[0, :2],
    ]).double().cpu().numpy()
    trace.update(post.tobytes())
    return obs, types.SimpleNamespace(row=row_t, pre=pre, post=post, done=done, failed=failed,
                                      sched_dev=sched_dev, cmd_dev=cmd_dev, stand_sched=bool(stand_sched),
                                      stand_slot=stand_slot, routed_stand=bool(route), route_ok=route_ok,
                                      st_differs=st_differs, cam_abs_max=float(cam_abs.max().item()))


def _store_step(rec: dict, t: int, st) -> None:
    row_t, pre, post = st.row, st.pre, st.post
    rec["row_speed"][t], rec["row_dir"][t], rec["row_face"][t] = row_t.speed, row_t.dir, row_t.face
    (rec["speed_cmd"][t], rec["travel_dir_w"][t], rec["heading_target"][t], rec["cmd_bx"][t],
     rec["cmd_by"][t], rec["cmd_wz"][t], rec["heading_pre"][t], rec["x_pre"][t], rec["y_pre"][t]) = pre
    rec["x_post"][t], rec["y_post"][t], rec["heading_post"][t], rec["vx_w"][t], rec["vy_w"][t] = post
    rec["done"][t], rec["fail"][t] = float(st.done), float(st.failed)
    rec["stand_schedule"][t], rec["stand_slot"][t] = float(st.stand_sched), float(st.stand_slot)
    rec["routed_stand"][t], rec["route_ok"][t] = float(st.routed_stand), float(st.route_ok)
    rec["st_differs_s1"][t], rec["cam_abs_max"][t] = float(st.st_differs), st.cam_abs_max


def _termination_reason(failed: bool, steps_since_reset: int, max_episode_length: int) -> str:
    if failed:
        return "fall (crab_failure)"
    if steps_since_reset >= max_episode_length - 1:
        return "timeout"
    return "terminated_other (roll/pitch/height cutoff)"


def _window_stats(idx, ok_idx, speed, direction, face, dt, rec, until):
    """Commanded-vs-applied over ``idx`` and achieved motion over ``ok_idx`` (record_categories, unchanged)."""
    speed_body = np.hypot(rec["cmd_bx"], rec["cmd_by"])
    moving = ok_idx[speed_body[ok_idx] > 1e-6]
    world_dir_of_body_cmd = rec["heading_pre"] + np.arctan2(rec["cmd_by"], rec["cmd_bx"])
    applied_mean = {
        "speed": float(rec["speed_cmd"][idx].mean()),
        "dir_deg": _deg(_circmean(rec["travel_dir_w"][idx])),
        "face_deg": _deg(_circmean(rec["heading_target"][idx])),
    }
    applied_max_dev = {
        "speed": float(np.abs(rec["speed_cmd"][idx] - speed).max()),
        "dir_rad": float(np.abs(np.arctan2(np.sin(rec["travel_dir_w"][idx] - direction),
                                            np.cos(rec["travel_dir_w"][idx] - direction))).max()),
        "face_rad": float(np.abs(np.arctan2(np.sin(rec["heading_target"][idx] - face),
                                             np.cos(rec["heading_target"][idx] - face))).max()),
    }
    body_cmd_mean = {
        "speed_xy": float(speed_body[idx].mean()),
        "world_dir_deg": _deg(_circmean(world_dir_of_body_cmd[moving])) if moving.size else None,
        "wz": float(rec["cmd_wz"][idx].mean()),
    }
    achieved = None
    if ok_idx.size:
        x0, y0 = rec["x_pre"][ok_idx[0]], rec["y_pre"][ok_idx[0]]
        x1, y1 = rec["x_post"][ok_idx[-1]], rec["y_post"][ok_idx[-1]]
        dx, dy = x1 - x0, y1 - y0
        dur = ok_idx.size * dt
        along = dx * math.cos(direction) + dy * math.sin(direction)
        across = -dx * math.sin(direction) + dy * math.cos(direction)
        achieved = {
            "until": until,
            "duration_s": dur, "dx": dx, "dy": dy, "dist": math.hypot(dx, dy),
            "along_dir": along, "across_dir": across, "mean_speed_along": along / dur,
            "travel_dir_deg": _deg(math.atan2(dy, dx)) if math.hypot(dx, dy) > 0.05 else None,
            "heading_end_deg": _deg(rec["heading_post"][ok_idx[-1]]),
            "heading_err_end_deg": _deg(_wrap(rec["heading_post"][ok_idx[-1]] - face)),
        }
    return applied_mean, applied_max_dev, body_cmd_mean, achieved


def _sequence_segment(i, label, kind, flag, idx, cmd, dt, rec, falls, sched_dev, cmd_dev) -> dict:
    """One segment's summary (``i`` = -1 for the prologue): record_categories' block + the routing / stand reads."""
    speed, direction, face = cmd
    falls_in = [f for f in falls if f["segment"] == i]
    first_fall_step = falls_in[0]["step"] if falls_in else None
    ok_idx = idx if first_fall_step is None else idx[idx < first_fall_step]
    until = "end" if first_fall_step is None else f"first fall t={falls_in[0]['t_s']}"
    applied_mean, applied_max_dev, body_cmd_mean, achieved = _window_stats(
        idx, ok_idx, speed, direction, face, dt, rec, until)
    if achieved is not None:
        h0 = rec["heading_pre"][ok_idx[0]]
        path = np.concatenate([[h0], rec["heading_post"][ok_idx]])
        achieved["heading_start_deg"] = _deg(h0)
        achieved["face_err_start_deg"] = _deg(_wrap(h0 - face))
        achieved["heading_change_deg"] = _deg(float(np.sum(np.arctan2(np.sin(np.diff(path)), np.cos(np.diff(path))))))
    end = int(idx[-1]) if idx.size else None
    routed = rec["routed_stand"][idx] > 0.5
    yaw = CORE.yaw_cmd_stats(rec["routed_stand"], rec["cmd_wz"], idx, [f["step"] for f in falls])
    out = {
        "segment": i, "hold": label, "kind": kind, "stand_rule_flag": bool(flag),
        "start_step": int(idx[0]) if idx.size else None,
        "end_step_exclusive": end + 1 if end is not None else None, "steps": int(idx.size),
        "t0_s": float(idx[0] * dt) if idx.size else None, "t1_s": float((end + 1) * dt) if end is not None else None,
        "commanded": {"speed": float(speed), "dir_deg": _deg(direction), "face_deg": _deg(face)},
        "acting_network": "stand_net" if routed.all() else ("s1" if not routed.any() else "mixed"),
        "stand_net_share": float(routed.mean()) if idx.size else None,
        "eval_slot_stand_share": float((rec["stand_slot"][idx] > 0.5).mean()) if idx.size else None,
        "schedule_vs_slot_disagree_steps": int(((rec["stand_schedule"][idx] > 0.5)
                                                != (rec["stand_slot"][idx] > 0.5)).sum()),
        "route_mismatch_steps": int((rec["route_ok"][idx] < 0.5).sum()),
        "stand_net_differs_from_s1_steps": int((rec["st_differs_s1"][idx] > 0.5).sum()),
        "yaw_cmd": yaw,
        "applied_mean": applied_mean, "applied_max_dev": applied_max_dev,
        "schedule_state_max_dev": sched_dev, "obs_command_slot_max_dev": cmd_dev,
        "body_cmd_mean": body_cmd_mean,
        "falls": falls_in,
        "achieved": achieved,
        "heading_at_end_deg": _deg(rec["heading_post"][end]) if end is not None else None,
        "heading_err_at_end_deg": _deg(_wrap(rec["heading_post"][end] - face)) if end is not None else None,
    }
    if kind == CORE.KIND_STAND and i >= 0:
        out["stand"] = dict(CORE.stand_hold_stats(rec["cam_abs_max"], idx, dt, first_fall_step),
                            falls=len(falls_in), stand_net_share=out["stand_net_share"], **yaw)
    return out


def _run_sequence(ctx, scenario, seq_info: dict, kinds: list, flags: list) -> None:
    """One continuous episode of the schedule row, no forced reset (record_categories' sequence mode)."""
    import imageio

    env, unwrapped, robot, cmd_term, H, dt, fps = (ctx.env, ctx.unwrapped, ctx.robot, ctx.cmd_term, ctx.H,
                                                   ctx.dt, ctx.fps)
    rule = args_cli.stand_rule
    compiled = S.compile_schedule(scenario, dt=dt, num_envs=1)
    if compiled.world_dir is None:
        raise SystemExit(f"{scenario.id!r} compiled without world arrays; this recorder needs a world-frame row")
    T = int(compiled.n_steps)
    labels = list(compiled.hold_labels)
    seg, starts, ends = _segment_bounds(compiled)
    n_holds = len(labels)
    prologue_steps = starts[0]
    stand_sched = CORE.step_stand_mask(seg, flags)
    cmd_of = {}
    for i in [-1] + list(range(n_holds)):
        s0 = 0 if i < 0 else starts[i]
        cmd_of[i] = (float(compiled.world_speed[s0, 0]), float(compiled.world_dir[s0, 0]),
                     float(compiled.world_face[s0, 0]))
    for i, h in enumerate(scenario.schedule):
        if cmd_of[i] != h.as_world():
            raise SystemExit(f"compiled command of hold {i} {h.label!r} {cmd_of[i]} != manifest {h.as_world()}")
    spawn = scenario.spawn_yaw_rad()
    print(f"[SNV] SEQUENCE {scenario.id}: {T} steps = {T * dt:.2f} s at dt {dt}; prologue steps 0-{prologue_steps - 1} "
          f"(holds '{labels[0]}' command); spawn yaw pin {'none' if spawn is None else f'{_deg(spawn):+.1f} deg'}; "
          f"env_cfg.seed {scenario.env_seed}; stand rule {rule}: "
          f"{CORE.STAND_RULE_SCHEDULE if rule == 'schedule' else CORE.STAND_RULE_SLOT}", flush=True)
    eval_flags = CORE.eval_slot_stand_flags(scenario.schedule, spawn)
    predicted_eval_turn = [labels[i] for i in range(n_holds) if flags[i] and not eval_flags[i]]
    for i, h in enumerate(scenario.schedule):
        sp, di, fa = cmd_of[i]
        print(f"[SNV] plan: hold {i + 1:2d}/{n_holds} {h.label:22s} {kinds[i]:9s} -> "
              f"{'STAND NET' if flags[i] else 'S1':9s} speed {sp:.2f} dir {_deg(di):+7.2f} face {_deg(fa):+7.2f}  "
              f"t {starts[i] * dt:6.2f}-{ends[i] * dt:6.2f} s ({h.hold_s:.2f} s)", flush=True)
        if flags[i] and not eval_flags[i]:
            print(f"[SNV]   note: '{h.label}' comes after a facing change with no reset since -- the eval's latched "
                  "slot keeps TURN here (predicted, fall-free); the schedule rule routes it to the stand net",
                  flush=True)

    override_state = H["_install_schedule_override_world"](cmd_term, compiled, ctx.device, compiled.hold_labels)
    obs, _extras = env.get_observations()
    override_state["apply"](0)
    obs, _extras = env.get_observations()
    start_state = {
        "x": float(robot.data.root_pos_w[0, 0].item()), "y": float(robot.data.root_pos_w[0, 1].item()),
        "z": float(robot.data.root_pos_w[0, 2].item()),
        "heading_deg": _deg(float(robot.data.heading_w[0].item())),
        "episode_length_buf": int(unwrapped.episode_length_buf[0].item()),
    }
    print(f"[SNV] start state after the wrapper construction reset: {start_state}", flush=True)

    still_at = {}
    if not args_cli.no_stills:
        for i in range(n_holds):
            still_at[starts[i] + (ends[i] - starts[i]) // 2] = f"{i:02d}_{labels[i]}_mid"
            still_at[ends[i] - 1] = f"{i:02d}_{labels[i]}_end"

    rec = _new_rec(T)
    falls: list[dict] = []
    sched_dev = np.zeros(T)
    cmd_dev = np.zeros(T)
    steps_since_reset = int(unwrapped.episode_length_buf[0].item())
    total_frames = 0
    trace = hashlib.sha256()
    t_run = time.time()
    st_name, s1_name = f"STAND NET ({ctx.tags['stand_tag']})", f"{ctx.tags['s1_tag']} walking"
    try:
        with torch.inference_mode():
            for t in range(T):
                obs, st = _harness_step(ctx, override_state, obs, t, trace, bool(stand_sched[t]), rule)
                sched_dev[t], cmd_dev[t] = st.sched_dev, st.cmd_dev
                _store_step(rec, t, st)
                steps_since_reset += 1
                t_s = (t + 1) * dt
                i = int(seg[t])
                lab = labels[i] if i >= 0 else "prologue"
                if st.done:
                    reason = _termination_reason(st.failed, steps_since_reset, int(unwrapped.max_episode_length))
                    falls.append({"t_s": round(t_s, 3), "step": t, "reason": reason, "segment": i, "hold": lab,
                                  "acting": "stand_net" if st.routed_stand else "s1"})
                    print(f"[SNV] TERMINATION at t={t_s:.2f} s (step {t}, in '{lab}', acting "
                          f"{'stand net' if st.routed_stand else 'S1'}): {reason} -> env auto-reset (respawn at the "
                          "pinned yaw); the schedule continues", flush=True)
                    steps_since_reset = 0

                frame = ctx.render()
                if frame is None:
                    raise SystemExit("render() returned None -- is render_mode rgb_array / --enable_cameras set?")
                if _CAPTION_BACKEND is not None:
                    row_t, post = st.row, st.post
                    k0, k1 = (0, starts[0]) if i < 0 else (starts[i], ends[i])
                    hold_t = (t + 1 - k0) * dt
                    head = (f"{ctx.tags['label']}  |  "
                            + (f"hold {i + 1}/{n_holds} '{lab}' [{kinds[i]}]" if i >= 0
                               else f"PROLOGUE (holds '{labels[0]}')")
                            + f"  {hold_t:4.1f} / {(k1 - k0) * dt:.1f} s", WHITE)
                    acting = (f"ACTING: {st_name if st.routed_stand else s1_name}"
                              + ("" if st.stand_sched == st.stand_slot else
                                 f"   (eval latched slot: {'STAND' if st.stand_slot else 'TURN'})"),
                              STANDNET_COLOR if st.routed_stand else S1_COLOR)
                    cmd_line = (f"CMD  speed {row_t.speed:.2f} m/s   dir {_deg(row_t.dir):+.0f} deg   "
                                f"face {_deg(row_t.face):+.0f} deg   (world frame)   yaw-rate cmd {st.pre[5]:+.2f}",
                                YELLOW)
                    nxt = i + 1
                    if nxt < n_holds:
                        nxt_line = (f"NEXT  '{labels[nxt]}' ({'STAND NET' if flags[nxt] else 'S1'}) at "
                                    f"t={starts[nxt] * dt:.1f} s", CYAN)
                    else:
                        nxt_line = (f"LAST hold -- sequence ends at t={T * dt:.1f} s", CYAN)
                    lines = [head, acting, cmd_line, nxt_line,
                             (f"t = {t_s:6.2f} / {T * dt:.1f} s    heading {_deg(post[2]):+6.1f} deg    "
                              f"|v| {math.hypot(post[3], post[4]):.2f} m/s    max|cam| {st.cam_abs_max:.2f} rad",
                              WHITE)]
                    if args_cli.arrows:
                        lines.append((ARROW_LEGEND, CYAN))
                    if falls:
                        last = falls[-1]
                        lines.append((f"FELL at t={last['t_s']:.2f} s in '{last['hold']}' -> auto-reset, schedule "
                                      "continues" + (f" ({len(falls)} terminations)" if len(falls) > 1 else ""), RED))
                        if st.routed_stand and st.pre[5] != 0.0 and falls[0]["step"] < t:
                            lines.append(("POST-FALL: stand net on a live turn command (respawn yaw != schedule "
                                          "face) -- no gate evaluated this", RED))
                    frame = _caption(frame, lines)
                else:
                    frame = np.ascontiguousarray(frame[:, :, :3])
                ctx.combined.append_data(frame)
                total_frames += 1
                if t in still_at:
                    imageio.imwrite(str(ctx.stills_dir / f"{args_cli.prefix}_sequence_{still_at[t]}.png"), frame)
                if (t + 1) % 500 == 0:
                    el = time.time() - t_run
                    rss = _rss_gb()
                    print(f"[SNV] sequence: {t + 1}/{T} frames ('{lab}') in {el:.0f}s "
                          f"({total_frames / max(el, 1e-9):.1f} fps); RSS {rss:.2f} GB", flush=True)
                    if rss > args_cli.rss_abort_gb:
                        raise SystemExit(f"RSS {rss:.1f} GB > {args_cli.rss_abort_gb} GB at step {t + 1} -- aborting")
    finally:
        ctx.combined.close()

    el = time.time() - t_run
    trace_sha = trace.hexdigest()
    segments = []
    for i in [-1] + list(range(n_holds)):
        idx = np.nonzero(seg == i)[0]
        if idx.size == 0:
            continue
        lab = labels[i] if i >= 0 else "prologue"
        kind = kinds[0] if i < 0 else kinds[i]
        flag = flags[0] if i < 0 else flags[i]
        segments.append(_sequence_segment(i, lab, kind, flag, idx, cmd_of[i], dt, rec, falls,
                                          float(sched_dev[idx].max()), float(cmd_dev[idx].max())))
    ru_peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0 / 1024.0
    cg_peak = _cgroup_peak_gb()
    stands = [s for s in segments if "stand" in s]
    routing = {
        "rule": rule, "rule_text": CORE.STAND_RULE_SCHEDULE if rule == "schedule" else CORE.STAND_RULE_SLOT,
        "stand_steps_routed": int((rec["routed_stand"] > 0.5).sum()), "steps": T,
        "route_mismatch_steps": int((rec["route_ok"] < 0.5).sum()),
        "stand_rows_stand_net_differs_from_s1": int(((rec["routed_stand"] > 0.5) & (rec["st_differs_s1"] > 0.5)).sum()),
        "schedule_vs_slot_disagree_steps": int(((rec["stand_schedule"] > 0.5) != (rec["stand_slot"] > 0.5)).sum()),
        "holds_where_slot_disagrees": [s["hold"] for s in segments if s["schedule_vs_slot_disagree_steps"]],
        "predicted_eval_slot_turn_stands": predicted_eval_turn,
        "eval_slot_prediction": "sn_video_core.eval_slot_stand_flags: fall-free, each face reached before the next",
        "resets_rederived": int(ctx.slot_rule.resets_rederived),
    }
    stand_summary = {
        "n_stand_holds": len(stands),
        "stand_net_share_min": min((s["stand"]["stand_net_share"] for s in stands), default=None),
        "homed_at_end": sum(1 for s in stands if s["stand"]["homed_at_end"]),
        "settled": sum(1 for s in stands if s["stand"]["settled"]),
        "falls_in_stands": sum(s["stand"]["falls"] for s in stands),
        "stand_holds_post_fall_turn_cmd": [s["hold"] for s in stands if s["stand"]["post_fall_turn_cmd"]],
        "stand_holds_yaw_cmd_at_entry": [s["hold"] for s in stands if s["stand"]["yaw_cmd_at_entry"]],
        "stand_net_on_yaw_cmd_steps": sum(s["stand"]["stand_net_on_yaw_cmd_steps"] for s in stands),
        "per_stand": [{"hold": s["hold"], "hold_s": s["steps"] * dt, **s["stand"]} for s in stands],
    }
    print(f"[SNV] wrote {ctx.combined_path} ({total_frames} frames, {total_frames / fps:.1f} s at {fps} fps) "
          f"in {el:.0f}s; trace_sha256 {trace_sha}", flush=True)
    print(f"[SNV] peak RSS (ru_maxrss) {ru_peak:.2f} GB; cgroup memory.peak "
          f"{'n/a' if cg_peak is None else f'{cg_peak:.2f} GB'}", flush=True)
    _print_sequence_table(segments, falls, routing, stand_summary)
    meta = {
        "mode": "sequence", "kind": "sn_stand_network_video",
        "checkpoint": str(ctx.resume_path), "checkpoint_sha256": S1_SHA,
        "stand_checkpoint": str(args_cli.stand_ckpt), "stand_checkpoint_sha256": STAND_SHA,
        "stand_mask": int(args_cli.stand_mask), "caption_tags": ctx.tags, "build": ctx.build,
        "forward": SN.FORWARD, "routing": routing, "stand_summary": stand_summary,
        "reference_run_meta": str(RUN_META_PATH), "reference_scenario_id": RUN_META["scenario_id"],
        "reference_checks": {k: {"reference": a, "here": b} for k, (a, b) in REFERENCE_CHECKS.items()},
        "reference_mismatch": REFERENCE_MISMATCH,
        "task": scenario.task, **seq_info,
        "env_vars": WANT_ENV, "env_overrides": ctx.applied_overrides,
        "harness": str(HARNESS), "harness_sha256": HARNESS_SHA,
        "recorder": str(Path(__file__).resolve()), "recorder_sha256": _sha256(Path(__file__).resolve()),
        "sn_common_sha256": SN.sha256_file(COMMON_PATH), "core_sha256": SN.sha256_file(CORE_PATH),
        "p2r_launcher_sha256": SN.sha256_file(P2R_LAUNCHER),
        "env_seed": scenario.env_seed, "env_seed_note": "env_cfg.seed only (the harness's single pass)",
        "spawn_yaw_pin": ctx.applied_overrides.get("spawn_yaw_pin"),
        "prologue_s": float(scenario.get_default("prologue_s")), "settle_s": float(scenario.get_default("settle_s")),
        "holds": [{"segment": i, "label": h.label, "kind": kinds[i], "stand": bool(flags[i]),
                   "hold_s": float(h.hold_s), "speed": cmd_of[i][0], "dir_deg": _deg(cmd_of[i][1]),
                   "face_deg": _deg(cmd_of[i][2]), "start_step": starts[i], "end_step_exclusive": ends[i],
                   "t0_s": starts[i] * dt, "t1_s": ends[i] * dt}
                  for i, h in enumerate(scenario.schedule)],
        "n_steps": T, "episode_length_s": scenario.episode_length_s,
        "max_episode_length": int(unwrapped.max_episode_length), "dt": dt, "fps": fps,
        "arrows": bool(args_cli.arrows),
        "camera": {"eye": list(ctx.env_cfg.viewer.eye), "lookat": list(ctx.env_cfg.viewer.lookat),
                   "origin_type": ctx.env_cfg.viewer.origin_type, "resolution": list(ctx.env_cfg.viewer.resolution)},
        "captions": _CAPTION_BACKEND, "mp4": str(ctx.combined_path), "frames_total": total_frames,
        "trace_sha256": trace_sha, "start_state": start_state,
        "schedule_state_max_dev": float(sched_dev.max()), "obs_command_slot_max_dev": float(cmd_dev.max()),
        "falls": falls, "peak_rss_gb": ru_peak, "cgroup_peak_gb": cg_peak, "segments": segments,
    }
    if args_cli.trace_npz:
        trace_path = ctx.out_dir / f"{args_cli.prefix}_sequence_trace.npz"
        np.savez(trace_path, segment_id=seg, sched_dev=sched_dev, cmd_dev=cmd_dev, **rec)
        meta["trace_npz"] = str(trace_path)
    sanity_path = ctx.out_dir / f"{args_cli.prefix}_sequence_sanity.json"
    sanity_path.write_text(json.dumps(meta, indent=1))
    print(f"[SNV] sequence sanity -> {sanity_path}", flush=True)
    if routing["route_mismatch_steps"]:
        print(f"[SNV][FAIL] {routing['route_mismatch_steps']} steps applied an action other than the routed "
              "network's -- the video is NOT the routed pair", flush=True)


def _print_sequence_table(segments: list[dict], falls: list[dict], routing: dict, stand_summary: dict) -> None:
    print("[SNV] SEQUENCE SANITY TABLE (net = the network that acted, share; slot = the eval's latched stand-slot "
          "share; cam_end = worst-leg |cam| at the hold's last row; settle = homed held 0.5 s)", flush=True)
    hdr = (f"{'#':>2s} {'hold':22s} {'kind':9s} {'t0-t1 s':>13s} | {'net':>9s} {'share':>5s} {'slot':>5s} | "
           f"{'cmd spd/dir/face':>20s} | {'dx':>6s} {'dy':>6s} {'v_al':>5s} | {'dhead':>7s} {'err':>6s} | "
           f"{'cam_end':>7s} {'settle':>6s} | falls")
    print(hdr, flush=True)
    nan = float("nan")
    for r in segments:
        c = r["commanded"]
        ach = r.get("achieved") or {}
        stn = r.get("stand") or {}
        fl = ",".join(f"{f['t_s']:.2f}s" for f in r["falls"]) or "none"
        num = "P" if r["segment"] < 0 else str(r["segment"] + 1)
        cam_end = stn.get("cam_max_end_rad")
        settle = stn.get("settle_s")
        print(f"{num:>2s} {r['hold']:22s} {r['kind']:9s} {r['t0_s']:6.2f}-{r['t1_s']:6.2f} | "
              f"{r['acting_network']:>9s} {r['stand_net_share']:5.2f} {r['eval_slot_stand_share']:5.2f} | "
              f"{c['speed']:.2f}/{c['dir_deg']:+5.0f}/{c['face_deg']:+5.0f} deg | "
              f"{ach.get('dx', nan):+6.2f} {ach.get('dy', nan):+6.2f} {ach.get('mean_speed_along', nan):+5.2f} | "
              f"{ach.get('heading_change_deg', nan):+7.1f} {ach.get('heading_err_end_deg', nan):+6.1f} | "
              f"{'   --  ' if cam_end is None else f'{cam_end:7.3f}'} "
              f"{'  --  ' if settle is None else f'{settle:6.2f}'} | {fl}", flush=True)
    print(f"[SNV] routing: {routing['stand_steps_routed']}/{routing['steps']} steps to the stand net (rule "
          f"{routing['rule']}); route mismatches {routing['route_mismatch_steps']} (want 0); stand rows where the stand "
          f"net's action == S1's {routing['stand_steps_routed'] - routing['stand_rows_stand_net_differs_from_s1']} "
          f"(want 0); schedule-vs-eval-slot disagreement {routing['schedule_vs_slot_disagree_steps']} steps in "
          f"{routing['holds_where_slot_disagrees']}", flush=True)
    print(f"[SNV] eval latched slot reads TURN on {routing['holds_where_slot_disagrees']} (predicted before the run: "
          f"{routing['predicted_eval_slot_turn_stands']})", flush=True)
    print(f"[SNV] stands: {stand_summary['n_stand_holds']} holds, stand-net share min "
          f"{stand_summary['stand_net_share_min']}, homed at end {stand_summary['homed_at_end']}, settled "
          f"{stand_summary['settled']}, falls {stand_summary['falls_in_stands']}; post-fall turn command under the "
          f"stand net {stand_summary['stand_holds_post_fall_turn_cmd'] or 'none'}; yaw command at entry "
          f"{stand_summary['stand_holds_yaw_cmd_at_entry'] or 'none'}", flush=True)
    print(f"[SNV] terminations in the run: {len(falls)}"
          + ("" if not falls else " -- " + "; ".join(f"t={f['t_s']:.2f}s '{f['hold']}' {f['reason']}" for f in falls)),
          flush=True)


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        import traceback

        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        raise
    finally:
        simulation_app.close()
