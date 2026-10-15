#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""Stand-height campaign -- follow-cam VIDEO of the routed pair with HEIGHT COMMANDS (S1 walks, the stand-height network
stands at each hold's commanded height, raises / lowers inside a stand, the H6 restore hands back at walking height),
recorded through the eval harness's own schedule path (VIDEO ONLY; step 8; read by no gate).

An ADAPTED COPY of the closed predecessor's ``2026-10-11_stand_network/scripts/sn_record_video.py`` (never edited; itself
a copy of ``experiments/tools/record_categories.py``'s sequence mode). Launch through ``sh_record_video.sh`` (one Isaac
process at a time, memory cap, no SIGKILL, detached)::

    sh_record_video.py --headless --enable_cameras --stand-ckpt <model_5000.pt> --stand-mask 1 --deadband 0.25 \\
        --manifest <videos/scenarios_sh_video_raise.yaml> --sequence sh_raise__omni --prefix <p>

DEFAULTS = the final head configuration (CHARTER section 3, the SH_T18 user decisions): S1 s18 (``--checkpoint``,
``crab_hex_omni_teacher/2026-10-07_07-50-10_seed18_X08_p2a_A15pB/model_19994.pt``) walks; the stand-height network
``--stand-ckpt`` (default ``stand_height/runs/SH_T18/model_5000.pt``, ``--stand-mask 1``) stands, with the residual
deadband ``--deadband`` (default 0 = off, the launcher's default; the step-8 videos pass 0.25, the configuration of
record) and the H6 restore ARMED (always: the video is the deployment); the KRABBY_* environment is the reference stamp's
(``--eval-run``, default ``phase2/lr/SH_T18/hraise__omni/seed001``, newest complete stamp); the row is ``sh_raise__omni``
of ``videos/scenarios_sh_video_raise.yaml`` (``sh_video_schedule.py``). This is a VIEWER, not a scorer: it writes an mp4,
stills and a sanity JSON, never a ledger entry, and nothing in the campaign reads them.

WHAT IT KEEPS FROM THE PREDECESSOR (see that file for the full argument)

* ENV COMPOSITION = the reference stamp's ``run_meta["env_vars"]``, replaced BEFORE Kit boots and re-checked after the
  task imports; ``sn_common.env_reasons`` (the D5 / L10 refusals) must pass on it. The row is built as the harness's
  single pass builds a schedule row (``S.load_manifest``, ``episodes`` 1, ``_apply_env_overrides``: seed = the row's
  ``env_seed``, spawn yaw pinned to the first face, resampling / curriculum / pushes off, frozen terrain levels); the
  row's task / terrain_label must equal the reference stamp's row's (``hraise__omni``: flat); a row ``env:`` block is
  refused; the final row must pass ``S.validate_scenario``.
* COMMAND PIN = the harness's own ``_install_schedule_override_world`` (extracted with ``ast`` and executed, never
  copied; its sha256 checked against the stamp's ``p2r_mode.harness_sha256``).
* NETWORK BUILD = P2L's st2 machinery (``build_with_template`` / ``StandInEnv`` / ``_load_extra`` with the RNG states
  saved and restored, ``arm_slot_latch``), then S1's ``runner.load``. Refused before the first step: a distillation
  runner or empirical normalisation; a stand estimator that differs from S1's; a stand checkpoint that is not a
  stand-height trainer checkpoint (``infos.sh_dagger``; the A-mask s17 file and S1 itself are refused here), whose
  ``stand_mask`` is not ``--stand-mask``, or that was trained beside another S1; a loaded actor that differs from the
  file's.
* STREAMING imageio writer (never gym ``RecordVideo``: the memory hazard), command arrows, per-hold stills, RSS abort,
  the rolling ``trace_sha256``, recording through a fall (env auto-reset; the schedule carries on).

WHAT IS NEW: THE HEIGHT PATH, THE LAUNCHER'S (``sh_eval_crab_hex_gait.py --sh-controller stand_net --sh-command
manifest --sh-restore armed [--sh-deadband d]``), COMPOSED FROM ``sh_core`` (``sh_video_core.height_step``, pinned
bitwise against the launcher's ``__call__`` by ``tests/unit/test_sh_video.py``)

* per-hold heights: ``sh_eval_common.height_row`` validates the manifest's ``stand_height`` block BEFORE Kit;
  ``sh_eval_common.height_table(compiled.segment_id, hold_heights)`` maps every step to its command (walk holds 1.064,
  the prologue the first hold's); the compiled hold labels must be the height row's;
* the stand slot: the eval's latched slot (``P2RMode.rederive_on_reset``), as the launcher routes; the schedule's own
  stand flag (speed 0; one facing per row) is recorded beside it;
* ``sh_core.HeightController`` (armed: engage 2 cm) reset as a walk handover at the first command and on every env
  reset; per step ``ctl.step(slot, h_cmd, obs)`` -> limited command + EFFECTIVE stand vector (slot | restoring);
  ``sh_core.stand_net_input`` (column 15 = the limited command, restore rows show a stand) -> the stand net ->
  ``sh_core.stand_action(a_net, h_lim, deadband=d)`` = q(h_lim) + deadband(clamp(a_net, +-1)) -> ``sh_core.route``
  (S1 rows clamp(S1, +-1));
* the wrapper clip raised from S1's 1.0 to the launcher's runtime 4.8 (``sh_core.HARNESS_CLIP``) after the build;
  on every non-reset step the env's applied action is checked to equal the executed tensor (the 4.8 clip realised);
* the reference stamp is checked for the same deployment: variant ``stand_height``, controller ``stand_net``, restore
  ``armed``, mask, stand / S1 / harness shas, the residual deadband, sh_core / sh_kin / sn_common / R-module / facts
  shas, the launcher's and sh_eval_common's shas, the forward string, and (post-Kit) the controller record.

CAPTIONS (per frame; the predecessor's style): mode (walk / stand / restoring, coloured), the scheduled command
``h_cmd``, the limited command, the measured ``h_meas`` (``sh_core.h_meas`` on the harness's raycast), feet loaded
(>= 20 N and <= 0.15 m above the terrain: the trainer's probe rule), homed (every cam within 0.10 rad), the world
command, the next hold, time / heading / speed, the last restore event, falls.

OUTPUTS (``--out-dir``, default ``parkour/logs/videos/sh_stand_height``): ``<prefix>_sequence.mp4``,
``stills/<prefix>_sequence_<NN>_<label>_{mid,end}.png``, ``<prefix>_sequence_sanity.json`` (per segment: the
record_categories block, the routing shares and checks; per stand hold the height read-out -- arrival, slack, the
last-2-s |h_meas - h_cmd|, feet -- and the homing statistic; per walk hold after a stand the restore events; the run
meta: shas, reference checks, controller record, deadband parity, event totals, trace sha256).
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

HERE = Path(__file__).resolve().parent                    # 2026-10-13_stand_height/scripts
CAMPAIGN = HERE.parent
EXPERIMENTS = HERE.parents[1]
PKG = HERE.parents[2]                                      # crab_hex_omni_task
REPO = HERE.parents[6]
PARKOUR = REPO / "parkour"
FWD_PKG = PKG.parent / "crab_hex_forward_task"
HARNESS = FWD_PKG / "scripts/eval_crab_hex_gait.py"      # shared harness: one harness serves all three tasks
HARNESS_DIR = HARNESS.parent
MANIFEST = CAMPAIGN / "scenarios_sh.yaml"                 # where the reference stamp's scenario_id lives
P2R_LAUNCHER = EXPERIMENTS / "2026-10-07_omni_phase2" / "scripts" / "p2r_mode_eval_crab_hex_gait.py"
SH_LAUNCHER = HERE / "sh_eval_crab_hex_gait.py"          # its sha only: never imported here
VC_PATH = HERE / "sh_video_core.py"
EC_PATH = HERE / "sh_eval_common.py"
CORE_PATH = HERE / "sh_core.py"
LOG_ROOT = PARKOUR / "logs/rsl_rl"
DEFAULT_CKPT = LOG_ROOT / "crab_hex_omni_teacher/2026-10-07_07-50-10_seed18_X08_p2a_A15pB/model_19994.pt"
DEFAULT_STAND_CKPT = PARKOUR / "outputs/omni_plan/stand_height/runs/SH_T18/model_5000.pt"
DEFAULT_EVAL_RUN = PARKOUR / "outputs/omni_plan/phase2/lr/SH_T18/hraise__omni/seed001"
DEFAULT_MANIFEST = CAMPAIGN / "videos/scenarios_sh_video_raise.yaml"
DEFAULT_ROW = "sh_raise__omni"
#: mp4s are bulk artifacts: parkour/logs/, never the campaign tree (record_categories' convention)
DEFAULT_OUT = PARKOUR / "logs/videos/sh_stand_height"


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


#: torch-free at import, all of them (the launcher's rule L6): loaded before Kit. ONE core instance (VC's).
VC = _load_by_path(VC_PATH, "_shv_video_core")
EC = VC.EC
CORE = VC.CORE
SN = VC.SN
P2L = SN.load_by_path(P2R_LAUNCHER, "_shv_p2r")             # a PRIVATE instance; only its functions are called

# cli_args exactly as the harness loads it (never put parkour/scripts on sys.path: it shadows pip rsl_rl).
_cli_spec = importlib.util.spec_from_file_location("parkour_rsl_rl_cli_args", PARKOUR / "scripts/rsl_rl/cli_args.py")
cli_args = importlib.util.module_from_spec(_cli_spec)
_cli_spec.loader.exec_module(cli_args)

from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser(description="Routed S1 + stand-height net video with height commands (VIDEO ONLY).")
parser.add_argument("--out-dir", type=str, default=str(DEFAULT_OUT), help=f"Output directory (default {DEFAULT_OUT}).")
parser.add_argument("--prefix", type=str, default="sh_raise", help="File prefix: <prefix>_sequence.mp4 / _sanity.json.")
parser.add_argument(
    "--eval-run", type=str, default=str(DEFAULT_EVAL_RUN),
    help="The reference stand_height eval: its stamp dir, its run_meta.json, or a seed dir (newest complete stamp).",
)
parser.add_argument("--manifest", type=str, default=str(DEFAULT_MANIFEST),
                    help="Manifest holding the --sequence row AND its stand_height block (sh_video_schedule.py).")
parser.add_argument("--sequence", type=str, default=DEFAULT_ROW, metavar="ROW_ID",
                    help=f"The world-frame height row to record (default {DEFAULT_ROW}).")
parser.add_argument("--stand-ckpt", type=str, default=str(DEFAULT_STAND_CKPT),
                    help="The stand-height network checkpoint (a sh_dagger trainer checkpoint).")
parser.add_argument("--stand-mask", type=int, choices=(0, 1), default=1,
                    help="The actor-side stand mask the stand net is run with (its trained arm's).")
parser.add_argument("--deadband", type=float, default=0.0,
                    help="Residual deadband on the hip / knee residual (sh_core.apply_deadband); default 0 = off, the "
                         "launcher's default. The configuration of record (SH_T18 / replicates) uses 0.25.")
parser.add_argument("--stand-tag", type=str, default=None,
                    help="Caption name of the stand network (default: from the loaded checkpoint's infos.sh_dagger, "
                         "e.g. 'SH_T18 s18 db0.25').")
parser.add_argument("--allow-reference-mismatch", action="store_true",
                    help="Record even if a checkpoint / mask / deadband / harness / core sha differs from the reference "
                         "stamp's (default: refuse; the mismatch is recorded in the sanity JSON).")
parser.add_argument("--label", type=str, default=None,
                    help="Caption title (default '<walker tag> + stand-height net <stand tag>', from the loaded "
                         "checkpoints).")
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
    """A run_meta.json, a stamp dir, or a seed dir -> the run_meta.json (newest stamp with run_meta + a completion
    marker: ``complete.json`` or run_arm's ``p2_complete.json``)."""
    p = Path(text)
    if p.is_file():
        return p
    if (p / "run_meta.json").is_file():
        return p / "run_meta.json"
    if p.is_dir():
        stamps = sorted(d for d in p.iterdir() if d.is_dir() and (d / "run_meta.json").is_file()
                        and ((d / "complete.json").is_file() or (d / "p2_complete.json").is_file()))
        if stamps:
            return stamps[-1] / "run_meta.json"
    raise SystemExit(f"--eval-run {text!r}: no run_meta.json (nor a complete stamp under it)")


# ------------------------------------------------------------------ pre-Kit checks (no torch)
try:
    DEADBAND = CORE.check_deadband(args_cli.deadband)
except ValueError as _exc:
    raise SystemExit(f"--deadband: {_exc}") from _exc
RUN_META_PATH = _resolve_eval_run(args_cli.eval_run)
RUN_META = json.loads(RUN_META_PATH.read_text())
WANT_ENV = {k: str(v) for k, v in RUN_META["env_vars"].items()}
_env_why = SN.env_reasons(WANT_ENV)
if _env_why:
    raise SystemExit("the reference stamp's env fails the launcher's D5 / L10 refusals:\n  " + "\n  ".join(_env_why))
for _p, _what in ((Path(args_cli.checkpoint), "--checkpoint"), (Path(args_cli.stand_ckpt), "--stand-ckpt"),
                  (Path(args_cli.manifest), "--manifest")):
    if not _p.is_file():
        raise SystemExit(f"{_what} not found: {_p}")
try:                                                      # the launcher's manifest path: validated BEFORE Kit
    HEIGHT_ROW = EC.height_row(args_cli.manifest, args_cli.sequence)
except (OSError, ValueError) as _exc:
    raise SystemExit(f"--manifest / --sequence: {_exc}") from _exc
S1_SHA = _sha256(Path(args_cli.checkpoint))
STAND_SHA = _sha256(Path(args_cli.stand_ckpt))
HARNESS_SHA = _sha256(HARNESS)
_kind = EC.ckpt_kind(STAND_SHA, S1_SHA)
if _kind is not None:
    raise SystemExit(f"--stand-ckpt is the {_kind} checkpoint; the video records a stand-height trainer checkpoint "
                     "(infos.sh_dagger) only")
PROVENANCE = CORE.provenance(DEADBAND)                    # shas of sh_core / sh_kin / sn_common / R / facts (hashlib)
_pm = RUN_META.get("p2r_mode") or {}
_shb = _pm.get(EC.BLOCK_KEY) or {}
_shc = _shb.get("core") or {}
try:
    _ref_db = CORE.recorded_deadband(_shb) if _shb else None
except ValueError:
    _ref_db = "malformed"
REFERENCE_CHECKS = {
    "reference_variant": (_pm.get("variant"), EC.VARIANT),
    "control": (_shb.get("control"), VC.CONTROLLER),
    "restore": (_shb.get("restore"), VC.RESTORE),
    "stand_mask": (_pm.get(EC.MASK_KEY), int(args_cli.stand_mask)),
    "residual_deadband": (_ref_db, DEADBAND),
    "stand_ckpt_sha256": ((_pm.get("specialist") or {}).get("sha256"), STAND_SHA),
    "s1_ckpt_sha256": (RUN_META.get("checkpoint_sha256"), S1_SHA),
    "harness_sha256": (_pm.get("harness_sha256"), HARNESS_SHA),
    "sh_launcher_sha256": (_shb.get("wrapper_sha256"), _sha256(SH_LAUNCHER)),
    "sh_eval_common_sha256": (_shb.get("common_sha256"), _sha256(EC_PATH)),
    "p2r_launcher_sha256": (_shb.get("p2r_launcher_sha256"), _sha256(P2R_LAUNCHER)),
    **{k: (_shc.get(k), PROVENANCE[k]) for k in ("sh_core_sha256", "sh_kin_sha256", "sn_common_sha256",
                                                 "restance_sha256", "facts_json_sha256")},
    "forward": (_shb.get("forward"), VC.SH_FORWARD),
    "hist_encoding": (_shb.get("hist_encoding"), True),
    "mask_fn": (_shb.get("mask_fn"), SN.MASK_FN),
}
REFERENCE_MISMATCH = {k: {"reference": a, "here": b} for k, (a, b) in REFERENCE_CHECKS.items() if a != b}
if REFERENCE_MISMATCH and not args_cli.allow_reference_mismatch:
    raise SystemExit(f"this recording would not be the deployment the reference stamp {RUN_META_PATH} evaluated: "
                     f"{REFERENCE_MISMATCH} (e.g. pass --deadband {_ref_db} for that stamp's deadband; "
                     "--allow-reference-mismatch records anyway)")


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
print(f"[SHV] reference stamp {RUN_META_PATH}", flush=True)
print(f"[SHV] KRABBY_* env replaced by run_meta env_vars ({len(WANT_ENV)} vars); "
      f"pre-existing differing/extra vars dropped: {_dropped or 'none'}", flush=True)
for _k, _v in sorted(WANT_ENV.items()):
    print(f"[SHV]   {_k}={_v}", flush=True)
print(f"[SHV] reference checks: {'all equal' if not REFERENCE_MISMATCH else f'MISMATCH (allowed): {REFERENCE_MISMATCH}'}",
      flush=True)
print(f"[SHV] height row {HEIGHT_ROW['scenario']} ({HEIGHT_ROW['manifest']}): hold_heights {HEIGHT_ROW['hold_heights']}; "
      f"deadband {DEADBAND}", flush=True)

_t_boot = time.time()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
print(f"[SHV] Kit up in {time.time() - _t_boot:.1f}s, RSS {_rss_gb():.2f} GB", flush=True)
if _rss_gb() > args_cli.rss_abort_gb:
    print(f"[SHV][ABORT] RSS {_rss_gb():.1f} GB > {args_cli.rss_abort_gb} GB right after Kit start -- "
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
from isaaclab.sensors.ray_caster import RayCaster  # noqa: E402
from isaaclab.utils.assets import retrieve_file_path  # noqa: E402
from isaaclab.utils.warp import raycast_mesh  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from parkour_isaaclab.envs.mdp.parkour_commands.uniform_parkour_command import (  # noqa: E402
    UniformParkourCommand,
)
from parkour_tasks.crab_hex_forward_task.mdp.crab_contact_sensors import (  # noqa: E402
    CRAB_HEX_FOOTPAD_BODY_NAMES,
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
    """``--sequence ROW_ID`` -> the schedule row, built as the harness builds a non-suite row (the predecessor's
    ``_resolve_sequence_row``): ``S.load_manifest`` validated it, ``episodes`` 1, ``episode_length_s`` = prologue +
    schedule + 2 s, then ``S.validate_scenario`` (a failure aborts). Its labels must be the height row's."""
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
    # ---- env composition stays the reference eval's; its row-only checks do not apply.
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
    if float(scenario.get_default("prologue_s")) != float(ref.get_default("prologue_s")):
        raise SystemExit(f"sequence row prologue_s {scenario.get_default('prologue_s')} != the reference row's "
                         f"{ref.get_default('prologue_s')}")
    labels = [h.label for h in scenario.schedule]
    if labels != list(HEIGHT_ROW["labels"]):
        raise SystemExit(f"sequence row labels {labels} != the height row's {HEIGHT_ROW['labels']}")
    print(f"[SHV] SEQUENCE: row {row_id!r} ({manifest}); checked: task {scenario.task} == run_meta task, "
          f"terrain_label {scenario.terrain_label!r} == reference row {ref.id!r}'s ({ref_src}), no row env: block, "
          "speed_ellipse / ang_vel_max / prologue_s == the reference's, labels == the height row's.", flush=True)
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
    print("[SHV] sequence row passes the harness validator", flush=True)
    info = {
        "row_id": row_id, "manifest": str(manifest), "manifest_sha256": _sha256(manifest), "reference_row": ref.id,
        "reference_row_manifest": str(ref_src), "terrain_label": scenario.terrain_label,
        "manifest_holds": manifest_holds, "validator": "pass", "height_row": dict(HEIGHT_ROW),
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


# ------------------------------------------------------------------ captions (the predecessor's painter)
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


class _SlotRule:
    """The eval's stand signal: P2L's ``P2RMode.rederive_on_reset`` (the SAME function object) on a holder carrying
    only what it reads -- re-derive the turning bit on reset rows, return ``cmd_term.stand_mask()``."""

    rederive_on_reset = P2L.P2RMode.rederive_on_reset

    def __init__(self, cmd_term, unwrapped):
        self.cmd_term = cmd_term
        self.unwrapped = unwrapped
        self.resets_rederived = 0


def _build_networks(env, agent_cfg, device):
    """S1's runner + the stand runner, exactly as P2L's st2 path builds them (module docstring)."""
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


def _check_stand_ckpt(runner, st_runner, st_est) -> tuple[dict, dict]:
    """The launcher's ``bind_s1`` / ``trained_arm`` refusals for a stand-height trainer checkpoint + S1 pairing + the
    loaded stand actor == the file's actor. Post-Kit, before the first step. Returns the record and
    ``infos.sh_dagger``."""
    ok, why = SN.state_dicts_equal(runner.alg.estimator.state_dict(), st_est.state_dict())
    if not ok:
        raise SystemExit(f"the stand checkpoint's estimator differs from S1's ({why}); refused (map L5)")
    ck = torch.load(str(args_cli.stand_ckpt), map_location="cpu", weights_only=False)
    infos = ck.get("infos") if isinstance(ck, dict) else None
    sd = infos.get(EC.CKPT_INFOS_KEY) if isinstance(infos, dict) else None
    if not isinstance(sd, dict):
        raise SystemExit(f"the stand checkpoint carries no infos.{EC.CKPT_INFOS_KEY} (not a stand-height trainer "
                         "checkpoint); refused")
    tm = sd.get("stand_mask")
    if type(tm) is not int or tm not in (0, 1):
        raise SystemExit(f"infos.{EC.CKPT_INFOS_KEY}.stand_mask is {tm!r}, not an int 0 / 1")
    if tm != int(args_cli.stand_mask):
        raise SystemExit(f"--stand-mask {args_cli.stand_mask} but the checkpoint was trained with stand_mask {tm}; an "
                         "arm runs with the mask it was trained with")
    s1_trained = sd.get("s1_sha256")
    if s1_trained != S1_SHA and not args_cli.allow_reference_mismatch:
        raise SystemExit(f"the stand net was trained beside S1 sha {str(s1_trained)[:12]}, not --checkpoint's "
                         f"{S1_SHA[:12]} (pass --allow-reference-mismatch to record anyway)")
    try:
        trained_db = CORE.recorded_deadband(sd)
    except ValueError as exc:
        raise SystemExit(f"infos.{EC.CKPT_INFOS_KEY} carries a malformed residual deadband ({exc})") from exc
    parity = CORE.deadband_parity({CORE.RESIDUAL_DEADBAND_KEY: DEADBAND, CORE.TRAINED_DEADBAND_KEY: trained_db})
    actor_file = {k: v for k, v in ck["model_state_dict"].items() if k.startswith("actor.")}
    actor_live = {k: v for k, v in st_runner.alg.policy.state_dict().items() if k.startswith("actor.")}
    ok, why = SN.state_dicts_equal(actor_file, actor_live)
    if not actor_file or not ok:
        raise SystemExit(f"the loaded stand actor differs from {args_cli.stand_ckpt}'s ({why or 'no actor.* keys'})")
    return ({"estimator_equals_s1": True, "trained_stand_mask": int(tm), "stand_ckpt_kind": EC.CKPT_KIND_TRAINED,
             "trained_beside_s1_sha256": s1_trained, "actor_params_equal_file": True,
             "trained_residual_deadband": trained_db, "deadband_parity": parity,
             "sh_dagger_arm": sd.get("arm"), "sh_dagger_seed": sd.get("seed"),
             "sh_dagger_iterations": sd.get("iterations"), "trim_branch": sd.get("trim_branch"),
             "trim_min": sd.get("trim_min")}, sd)


def _caption_tags(stand_check: dict, sh_dagger: dict, resume_path) -> dict:
    is_s1 = stand_check["trained_beside_s1_sha256"] == S1_SHA
    stand_tag = args_cli.stand_tag or VC.stand_tag_from_ckpt(sh_dagger, args_cli.stand_ckpt, DEADBAND)
    s1_tag = VC.s1_tag_from_ckpt(resume_path, is_stand_nets_s1=is_s1)
    return {"stand_tag": stand_tag, "s1_tag": s1_tag,
            "label": args_cli.label or f"{s1_tag} + stand-height net {stand_tag}",
            "stand_tag_source": "--stand-tag" if args_cli.stand_tag else "stand checkpoint infos.sh_dagger",
            "label_source": "--label" if args_cli.label else "derived from the loaded checkpoints"}


def main() -> None:
    out_dir = Path(args_cli.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    H = _load_harness_functions()
    print(f"[SHV] harness functions extracted from {HARNESS} (sha256 {HARNESS_SHA[:16]}...)", flush=True)

    manifest = Path(args_cli.manifest)
    scenarios, _ = S.load_manifest(manifest)
    scenario, seq_info = _resolve_sequence_row(scenarios, manifest)
    kinds = VC.hold_kinds(scenario.schedule)

    # ---- env, composed as the harness composes it
    env_cfg = parse_env_cfg(scenario.task, device=args_cli.device, num_envs=1, use_fabric=not args_cli.disable_fabric)
    applied_overrides = H["_apply_env_overrides"](env_cfg, scenario)
    print(f"[SHV] harness env overrides applied: {applied_overrides} (reference run: {RUN_META['env_overrides']})",
          flush=True)
    env_cfg.commands.base_velocity.debug_vis = bool(args_cli.arrows)
    eye, lookat = _parse_vec3(args_cli.cam_eye), _parse_vec3(args_cli.cam_lookat)
    if eye is not None:
        env_cfg.viewer.eye = eye
    if lookat is not None:
        env_cfg.viewer.lookat = lookat
    env_cfg.viewer.env_index = 0
    print(f"[SHV] camera: origin_type={env_cfg.viewer.origin_type} asset={env_cfg.viewer.asset_name} "
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
    print(f"[SHV] env up: dt={dt} max_episode_length={int(unwrapped.max_episode_length)} steps, RSS {_rss_gb():.2f} GB",
          flush=True)
    if _rss_gb() > args_cli.rss_abort_gb:
        raise SystemExit(f"RSS {_rss_gb():.1f} GB > {args_cli.rss_abort_gb} GB before the first step -- aborting")

    krabby_now = {k: v for k, v in sorted(os.environ.items()) if k.startswith("KRABBY_")}
    if krabby_now != WANT_ENV:
        raise SystemExit(f"KRABBY_* env after import differs from run_meta: now={krabby_now} want={WANT_ENV}")
    print("[SHV] KRABBY_* env after task import == reference run_meta env_vars", flush=True)

    robot = unwrapped.scene["robot"]
    usd_spawned = str(getattr(getattr(robot.cfg, "spawn", None), "usd_path", "<unknown>"))
    if Path(usd_spawned).name != Path(RUN_META["usd_path"]).name:
        raise SystemExit(f"spawned {usd_spawned} but the reference run spawned {RUN_META['usd_path']}")
    print(f"[SHV] plant usd {usd_spawned} (reference {RUN_META['usd_path']}, plant {RUN_META['plant']})", flush=True)

    # ---- the two networks, built as P2L's st2 path builds them
    runner, pristine, st_runner, st_policy, st_est, rng_restored, latch_pre = _build_networks(env, agent_cfg, device)
    resume_path = retrieve_file_path(str(args_cli.checkpoint))
    runner.load(resume_path)
    estimator_paras = pristine["estimator"]
    num_prop = int(estimator_paras["num_prop"])
    num_scan = int(estimator_paras["num_scan"])
    num_priv_explicit = int(estimator_paras["num_priv_explicit"])
    policy = runner.get_inference_policy(device=device)
    estimator = runner.get_estimator_inference_policy(device=device)
    stand_check, sh_dagger = _check_stand_ckpt(runner, st_runner, st_est)
    tags = _caption_tags(stand_check, sh_dagger, resume_path)
    if stand_check["deadband_parity"]["note"]:
        print(f"[SHV][NOTE] {stand_check['deadband_parity']['note']}", flush=True)

    # ---- the wrapper clip: S1's 1.0 -> the launcher's runtime 4.8 (S1 rows are clamped to +-1 by sh_core.route)
    clip_before = getattr(env, "clip_actions", None)
    if clip_before is None or isinstance(clip_before, bool) or float(clip_before) != EC.S1_TRAIN_CLIP:
        raise SystemExit(f"the wrapper clip_actions is {clip_before!r}, not S1's {EC.S1_TRAIN_CLIP}; refused (the "
                         "runtime raise to 4.8 is defined from S1's clip)")
    env.clip_actions = CORE.HARNESS_CLIP
    print(f"[SHV] S1 {resume_path} sha256 {S1_SHA[:16]}...; stand net {args_cli.stand_ckpt} sha256 {STAND_SHA[:16]}... "
          f"mask {args_cli.stand_mask} deadband {DEADBAND} ({stand_check}); rng_restored={rng_restored} "
          f"latch_preexisting={latch_pre}; num_prop={num_prop} priv=[{num_prop + num_scan}:"
          f"{num_prop + num_scan + num_priv_explicit}] wrapper clip {clip_before} -> {env.clip_actions}", flush=True)
    print(f"[SHV] caption names: walker '{tags['s1_tag']}', stand net '{tags['stand_tag']}' "
          f"({tags['stand_tag_source']}), title '{tags['label']}' ({tags['label_source']})", flush=True)

    cmd_term = unwrapped.command_manager.get_term("base_velocity")
    if not isinstance(cmd_term, UniformParkourCommand):
        raise SystemExit(f"base_velocity must be UniformParkourCommand, got {type(cmd_term)}")
    if args_cli.arrows and not hasattr(cmd_term, "travel_cmd_visualizer"):
        raise SystemExit("--arrows: the command term built no visualizers -- debug_vis did not reach it "
                         "(is this the omni command term, and was --enable_cameras passed?)")
    observed_slots, slots_src = H["_observed_command_slots"](unwrapped)
    print(f"[SHV] observed command slots {observed_slots} ({slots_src}); reference {RUN_META['observed_command_slots']}",
          flush=True)
    cam_names = [n for n in robot.joint_names if str(n).endswith(VC.CAM_SUFFIX)]
    if len(cam_names) != 6:
        raise SystemExit(f"expected 6 cam joints (*{VC.CAM_SUFFIX}), got {cam_names}")
    cam_cols = torch.tensor([list(robot.joint_names).index(n) for n in cam_names], device=device)

    # ---- the readings: feet (the harness's foot order), contact, the terrain mesh (h_meas, the harness's raycast)
    foot_body_ids, foot_body_names = robot.find_bodies(list(CRAB_HEX_FOOTPAD_BODY_NAMES), preserve_order=True)
    if tuple(foot_body_names) != tuple(CRAB_HEX_FOOTPAD_BODY_NAMES):
        raise SystemExit(f"articulation returned feet in unexpected order: {foot_body_names}")
    contact_sensor = unwrapped.scene.sensors["contact_forces"]
    sensor_foot_ids, sensor_foot_names = contact_sensor.find_bodies(list(CRAB_HEX_FOOTPAD_BODY_NAMES),
                                                                    preserve_order=True)
    if tuple(sensor_foot_names) != tuple(CRAB_HEX_FOOTPAD_BODY_NAMES):
        raise SystemExit(f"contact sensor returned feet in unexpected order: {sensor_foot_names}")
    try:
        scanner = unwrapped.scene.sensors.get("height_scanner")
        mesh_key = scanner.cfg.mesh_prim_paths[0] if scanner is not None else "/World/ground"
        mesh = RayCaster.meshes[mesh_key]
    except (KeyError, AttributeError) as exc:
        raise SystemExit(f"terrain raycast mesh unavailable ({exc}); h_meas needs it (the harness's root-offset "
                         "fallback would make it a constant)") from exc

    cam = getattr(unwrapped, "viewport_camera_controller", None)
    if cam is None:
        print("[SHV][WARN] no viewport camera controller (is --enable_cameras set?) -- camera cannot follow", flush=True)

    def _render() -> np.ndarray:
        if cam is not None and env_cfg.viewer.origin_type == "asset_root":
            cam.update_view_to_asset_root(env_cfg.viewer.asset_name)  # no one-render lag after a respawn
        return unwrapped.render()

    def _readings() -> dict:
        """The state the frame shows: h_meas (sh_core), feet loaded (the trainer probe's rule), cams."""
        d = robot.data
        fpos = d.body_pos_w[:, foot_body_ids, :]
        tz = CORE.terrain_z_under_feet(fpos, mesh, raycast_mesh)
        hm = CORE.h_meas(d.root_pos_w[:, 2], tz)
        f = contact_sensor.data.net_forces_w[:, sensor_foot_ids].norm(dim=-1)
        loaded = VC.feet_loaded_mask(f, fpos[..., 2], tz)
        cam_abs = VC.cam_abs(d.joint_pos[0, cam_cols])
        return {"h_meas": float(hm[0].item()), "feet": int(loaded[0].sum().item()),
                "cam_abs_max": float(cam_abs.max().item())}

    import imageio

    fps = int(round(1.0 / dt))
    if abs(fps * dt - 1.0) > 1e-6:
        print(f"[SHV][WARN] 1/dt={1.0 / dt} is not an integer fps; video plays at {fps} fps", flush=True)
    combined_path = out_dir / f"{args_cli.prefix}_sequence.mp4"
    stills_dir = out_dir / "stills"
    if not args_cli.no_stills:
        stills_dir.mkdir(parents=True, exist_ok=True)
    combined = imageio.get_writer(str(combined_path), fps=fps, codec="libx264", quality=args_cli.quality,
                                  macro_block_size=None, ffmpeg_log_level="error")
    print(f"[SHV] captions: {_CAPTION_BACKEND or 'NONE'}; fps {fps}; -> {combined_path}", flush=True)

    for i in range(60):
        f0 = _render()
        if f0 is not None and f0.size and float(f0.mean()) > 1.0:
            print(f"[SHV] renderer warm after {i + 1} render(s); frame {f0.shape}", flush=True)
            break

    ctx = types.SimpleNamespace(
        H=H, env=env, unwrapped=unwrapped, robot=robot, cmd_term=cmd_term, device=device, dt=dt, fps=fps,
        observed_slots=observed_slots, estimator=estimator, policy=policy, num_prop=num_prop,
        priv_lo=num_prop + num_scan, priv_hi=num_prop + num_scan + num_priv_explicit,
        stand_policy=st_policy, stand_estimator=st_est, stand_mask=int(args_cli.stand_mask),
        slot_rule=_SlotRule(cmd_term, unwrapped), readings=_readings,
        render=_render, combined=combined, combined_path=combined_path, stills_dir=stills_dir, out_dir=out_dir,
        env_cfg=env_cfg, applied_overrides=applied_overrides, resume_path=resume_path, tags=tags,
        build={"rng_restored": bool(rng_restored), "latch_preexisting": latch_pre,
               "wrapper_clip_before": float(clip_before), "wrapper_clip_realised": float(env.clip_actions),
               **stand_check},
    )
    try:
        _run_sequence(ctx, scenario, seq_info, kinds)
    finally:
        if cam is not None:
            # Drop the camera's post-update subscription first: otherwise it fires once more during
            # simulation_app.close() against the already-closed scene (a harmless but noisy traceback).
            cam._viewport_camera_update_handle = None
        env.close()


REC_KEYS = ("row_speed", "row_dir", "row_face", "speed_cmd", "travel_dir_w", "heading_target",
            "cmd_bx", "cmd_by", "cmd_wz", "heading_pre", "x_pre", "y_pre",
            "x_post", "y_post", "heading_post", "vx_w", "vy_w", "done", "fail",
            "stand_schedule", "slot", "restoring", "stand_eff", "mode", "route_ok", "exec_ok", "exec_checked",
            "h_cmd", "h_lim", "h_meas", "feet", "homed", "cam_abs_max",
            "engaged", "completed", "timed_out", "merged", "reset_cut", "db_zeroed", "a_net_abs_max")
MODE_CODE = {VC.MODE_WALK: 0, VC.MODE_STAND: 1, VC.MODE_RESTORING: 2}


def _new_rec(T: int) -> dict:
    return {key: np.zeros(T, dtype=np.float64) for key in REC_KEYS}


def _harness_step(ctx, ctl, override_state: dict, obs, t: int, trace, h_cmd_t, stand_sched: bool):
    """ONE control step of the harness's world-frame loop body with the stand_height launcher's controller in place of
    the one policy: advance ``override_state['step']``, check the schedule state and the observed command slots, write
    S1's estimate into the priv-explicit slice (the harness), the eval's latched slot (``rederive_on_reset``), then
    ``sh_video_core.height_step`` (sh_core: reset / HeightController.step / stand_net_input / stand_action / route),
    the routing checks, ``env.step`` and the post-step readings (the state the frame shows)."""
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
    reset_rows = ctx.unwrapped.episode_length_buf.reshape(-1) == 0
    st = VC.height_step(ctl, obs, slot=slot, reset_rows=reset_rows, h_cmd=h_cmd_t, s1_policy=ctx.policy,
                        stand_policy=ctx.stand_policy, stand_estimator=ctx.stand_estimator, p0=ctx.priv_lo,
                        p1=ctx.priv_hi, num_prop=ctx.num_prop, mask=ctx.stand_mask, deadband=DEADBAND)
    actions, hs = st.actions, st.hs
    eff = bool(hs.stand_eff[0].item())
    route_ok = bool(torch.equal(actions, st.a_stand.to(actions.dtype) if eff else CORE.s1_clamp(st.a_s1)))
    db_zeroed = 0
    if eff and DEADBAND:
        r = torch.clamp(st.a_net, -CORE.RESIDUAL_CLIP, CORE.RESIDUAL_CLIP)[:, CORE.DEADBAND_ACT]
        db_zeroed = int(((r.abs() < DEADBAND) & (r != 0)).sum().item())
    trace.update(obs.detach().cpu().numpy().tobytes())
    trace.update(actions.detach().cpu().numpy().tobytes())

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
    # the env's applied action == the executed tensor (the 4.8 clip realised); a reset step re-zeroes the buffer
    applied = getattr(ctx.unwrapped.action_manager, "action", None)
    exec_checked = (not done) and applied is not None
    exec_ok = bool(torch.equal(applied, actions)) if exec_checked else True
    post = torch.cat([
        robot.data.root_pos_w[0, :2], robot.data.heading_w[:1], robot.data.root_lin_vel_w[0, :2],
    ]).double().cpu().numpy()
    trace.update(post.tobytes())
    rd = ctx.readings()
    return obs, types.SimpleNamespace(
        row=row_t, pre=pre, post=post, done=done, failed=failed, sched_dev=sched_dev, cmd_dev=cmd_dev,
        stand_sched=bool(stand_sched), slot=bool(slot[0].item()), restoring=bool(hs.restoring[0].item()),
        stand_eff=eff, mode=VC.mode_of(bool(slot[0].item()), bool(hs.restoring[0].item())),
        route_ok=route_ok, exec_ok=exec_ok, exec_checked=exec_checked,
        h_cmd=float(CORE.H_SETTLED if h_cmd_t is None else h_cmd_t[0].item()), h_lim=float(hs.h_lim[0].item()),
        engaged=bool(hs.engaged[0].item()), completed=bool(hs.completed[0].item()),
        timed_out=bool(hs.timed_out[0].item()), merged=bool(hs.merged[0].item()), reset_cut=bool(st.cut[0].item()),
        db_zeroed=db_zeroed, a_net_abs_max=float(st.a_net.abs().max().item()), **rd)


def _store_step(rec: dict, t: int, st) -> None:
    row_t, pre, post = st.row, st.pre, st.post
    rec["row_speed"][t], rec["row_dir"][t], rec["row_face"][t] = row_t.speed, row_t.dir, row_t.face
    (rec["speed_cmd"][t], rec["travel_dir_w"][t], rec["heading_target"][t], rec["cmd_bx"][t],
     rec["cmd_by"][t], rec["cmd_wz"][t], rec["heading_pre"][t], rec["x_pre"][t], rec["y_pre"][t]) = pre
    rec["x_post"][t], rec["y_post"][t], rec["heading_post"][t], rec["vx_w"][t], rec["vy_w"][t] = post
    rec["done"][t], rec["fail"][t] = float(st.done), float(st.failed)
    rec["mode"][t] = MODE_CODE[st.mode]
    rec["homed"][t] = float(VC.is_homed(st.cam_abs_max))
    for k in ("stand_sched", "slot", "restoring", "stand_eff", "route_ok", "exec_ok", "exec_checked", "engaged",
              "completed", "timed_out", "merged", "reset_cut"):
        rec["stand_schedule" if k == "stand_sched" else k][t] = float(getattr(st, k))
    for k in ("h_cmd", "h_lim", "h_meas", "feet", "cam_abs_max", "db_zeroed", "a_net_abs_max"):
        rec[k][t] = float(getattr(st, k))


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


def _sequence_segment(i, label, kind, height, idx, cmd, dt, rec, falls, sched_dev, cmd_dev, prev_kind) -> dict:
    """One segment's summary (``i`` = -1 for the prologue): record_categories' block + routing + the height reads."""
    speed, direction, face = cmd
    falls_in = [f for f in falls if f["segment"] == i]
    first_fall_step = falls_in[0]["step"] if falls_in else None
    ok_idx = idx if first_fall_step is None else idx[idx < first_fall_step]
    until = "end" if first_fall_step is None else f"first fall t={falls_in[0]['t_s']}"
    applied_mean, applied_max_dev, body_cmd_mean, achieved = _window_stats(
        idx, ok_idx, speed, direction, face, dt, rec, until)
    end = int(idx[-1]) if idx.size else None
    eff = rec["stand_eff"][idx] > 0.5
    out = {
        "segment": i, "hold": label, "kind": kind, "h_cmd": height,
        "start_step": int(idx[0]) if idx.size else None,
        "end_step_exclusive": end + 1 if end is not None else None, "steps": int(idx.size),
        "t0_s": float(idx[0] * dt) if idx.size else None, "t1_s": float((end + 1) * dt) if end is not None else None,
        "commanded": {"speed": float(speed), "dir_deg": _deg(direction), "face_deg": _deg(face)},
        "acting_network": "stand_net" if eff.all() else ("s1" if not eff.any() else "mixed"),
        "stand_net_share": float(eff.mean()) if idx.size else None,
        "slot_share": float((rec["slot"][idx] > 0.5).mean()) if idx.size else None,
        "restoring_share": float((rec["restoring"][idx] > 0.5).mean()) if idx.size else None,
        "schedule_vs_slot_disagree_steps": int(((rec["stand_schedule"][idx] > 0.5) != (rec["slot"][idx] > 0.5)).sum()),
        "route_mismatch_steps": int((rec["route_ok"][idx] < 0.5).sum()),
        "exec_mismatch_steps": int((rec["exec_ok"][idx] < 0.5).sum()),
        "applied_mean": applied_mean, "applied_max_dev": applied_max_dev,
        "schedule_state_max_dev": sched_dev, "obs_command_slot_max_dev": cmd_dev,
        "body_cmd_mean": body_cmd_mean,
        "falls": falls_in,
        "achieved": achieved,
        "h_meas_end": float(rec["h_meas"][end]) if end is not None else None,
    }
    if kind == VC.KIND_STAND and i >= 0:
        out["stand"] = dict(VC.stand_hold_height_stats(rec["h_cmd"], rec["h_lim"], rec["h_meas"], rec["feet"], idx, dt,
                                                       first_fall_step),
                            homing=VC.SNV.stand_hold_stats(rec["cam_abs_max"], idx, dt, first_fall_step),
                            falls=len(falls_in), deadband_zeroed_entries=int(rec["db_zeroed"][idx].sum()))
    if kind == VC.KIND_WALK and prev_kind == VC.KIND_STAND:
        out["restore"] = VC.restore_hold_stats(rec["restoring"], rec["engaged"], rec["completed"], rec["timed_out"],
                                               idx, dt)
    return out


def _run_sequence(ctx, scenario, seq_info: dict, kinds: list) -> None:
    """One continuous episode of the schedule row, no forced reset (record_categories' sequence mode)."""
    import imageio

    env, unwrapped, H, dt, fps = ctx.env, ctx.unwrapped, ctx.H, ctx.dt, ctx.fps
    robot, cmd_term = ctx.robot, ctx.cmd_term
    compiled = S.compile_schedule(scenario, dt=dt, num_envs=1)
    if compiled.world_dir is None:
        raise SystemExit(f"{scenario.id!r} compiled without world arrays; this recorder needs a world-frame row")
    T = int(compiled.n_steps)
    labels = list(compiled.hold_labels)
    if labels != list(HEIGHT_ROW["labels"]):                     # the launcher's begin_pass check
        raise SystemExit(f"the compiled schedule's hold labels {labels} are not the height row's "
                         f"{HEIGHT_ROW['labels']}")
    heights = list(HEIGHT_ROW["hold_heights"])
    h_np = EC.height_table(compiled.segment_id, heights)        # (T, 1) float64: the launcher's per-step map
    seg, starts, ends = _segment_bounds(compiled)
    n_holds = len(labels)
    prologue_steps = starts[0]
    stand_sched = VC.SNV.step_stand_mask(seg, [k == VC.KIND_STAND for k in kinds])
    cmd_of = {}
    for i in [-1] + list(range(n_holds)):
        s0 = 0 if i < 0 else starts[i]
        cmd_of[i] = (float(compiled.world_speed[s0, 0]), float(compiled.world_dir[s0, 0]),
                     float(compiled.world_face[s0, 0]))
    for i, h in enumerate(scenario.schedule):
        if cmd_of[i] != h.as_world():
            raise SystemExit(f"compiled command of hold {i} {h.label!r} {cmd_of[i]} != manifest {h.as_world()}")
    spawn = scenario.spawn_yaw_rad()
    print(f"[SHV] SEQUENCE {scenario.id}: {T} steps = {T * dt:.2f} s at dt {dt}; prologue steps 0-{prologue_steps - 1} "
          f"(holds '{labels[0]}' command, height {h_np[0, 0]:.3f}); spawn yaw pin "
          f"{'none' if spawn is None else f'{_deg(spawn):+.1f} deg'}; env_cfg.seed {scenario.env_seed}; routing: the "
          "eval's latched stand slot | H6 restore (the launcher's)", flush=True)
    for i, h in enumerate(scenario.schedule):
        sp, di, fa = cmd_of[i]
        what = f"STAND NET at {heights[i]:.3f} m" if kinds[i] == VC.KIND_STAND else "S1 walk"
        print(f"[SHV] plan: hold {i + 1:2d}/{n_holds} {h.label:8s} {kinds[i]:5s} -> {what:22s} speed {sp:.2f} "
              f"dir {_deg(di):+7.2f} face {_deg(fa):+7.2f}  t {starts[i] * dt:6.2f}-{ends[i] * dt:6.2f} s "
              f"({h.hold_s:.2f} s)", flush=True)

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
    print(f"[SHV] start state after the wrapper construction reset: {start_state}", flush=True)

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
    last_event = None
    restore_t0 = None
    ctl = None
    try:
        with torch.inference_mode():
            # the launcher's _start_pass_state (restore armed), inside inference mode like its lazy build
            h_dev = torch.as_tensor(h_np, dtype=torch.float64, device=ctx.device)
            ctl = VC.new_controller(1, ctx.device, dt=dt, h_cmd0=h_dev[0])
            ctl_record = dict(ctl.record(), restore=VC.RESTORE)
            ref_ctl = (((RUN_META.get("p2r_mode") or {}).get(EC.BLOCK_KEY) or {}).get("controller"))
            if ref_ctl != ctl_record:
                msg = f"controller record {ctl_record} != the reference stamp's {ref_ctl}"
                if not args_cli.allow_reference_mismatch:
                    raise SystemExit(msg + " (pass --allow-reference-mismatch to record anyway)")
                REFERENCE_MISMATCH["controller"] = {"reference": ref_ctl, "here": ctl_record}
                print(f"[SHV][WARN] {msg} (allowed)", flush=True)
            for t in range(T):
                obs, st = _harness_step(ctx, ctl, override_state, obs, t, trace, h_dev[t], bool(stand_sched[t]))
                sched_dev[t], cmd_dev[t] = st.sched_dev, st.cmd_dev
                _store_step(rec, t, st)
                steps_since_reset += 1
                t_s = (t + 1) * dt
                i = int(seg[t])
                lab = labels[i] if i >= 0 else "prologue"
                if st.engaged:
                    restore_t0 = t_s
                    last_event = f"restore ENGAGED at t={t_s:.2f} s (limited command {st.h_lim:.3f} m -> 1.064 m)"
                if st.completed:
                    last_event = (f"restore COMPLETED at t={t_s:.2f} s"
                                  + (f" ({t_s - restore_t0:.2f} s)" if restore_t0 is not None else "")
                                  + " -> handed back to S1")
                if st.timed_out:
                    last_event = f"restore TIMED OUT at t={t_s:.2f} s -> handed back to S1 anyway"
                if st.merged:
                    last_event = f"restore MERGED into a new stand at t={t_s:.2f} s"
                if st.done:
                    reason = _termination_reason(st.failed, steps_since_reset, int(unwrapped.max_episode_length))
                    falls.append({"t_s": round(t_s, 3), "step": t, "reason": reason, "segment": i, "hold": lab,
                                  "acting": st.mode, "h_cmd": st.h_cmd, "h_lim": st.h_lim})
                    print(f"[SHV] TERMINATION at t={t_s:.2f} s (step {t}, in '{lab}', mode {st.mode}, h_lim "
                          f"{st.h_lim:.3f}): {reason} -> env auto-reset (respawn at the pinned yaw); the schedule "
                          "continues", flush=True)
                    steps_since_reset = 0

                frame = ctx.render()
                if frame is None:
                    raise SystemExit("render() returned None -- is render_mode rgb_array / --enable_cameras set?")
                if _CAPTION_BACKEND is not None:
                    k0, k1 = (0, starts[0]) if i < 0 else (starts[i], ends[i])
                    hi = 0 if i < 0 else i
                    fall_text = None
                    if falls:
                        last = falls[-1]
                        fall_text = (f"FELL at t={last['t_s']:.2f} s in '{last['hold']}' -> auto-reset, schedule "
                                     "continues" + (f" ({len(falls)} terminations)" if len(falls) > 1 else ""))
                    lines = VC.caption_lines(
                        title=ctx.tags["label"], hold_i=i, n_holds=n_holds, hold_label=labels[hi],
                        hold_kind=kinds[hi], hold_t=(t + 1 - k0) * dt, hold_len=(k1 - k0) * dt, t_s=t_s,
                        total_s=T * dt, mode=st.mode, stand_tag=ctx.tags["stand_tag"], s1_tag=ctx.tags["s1_tag"],
                        deadband=DEADBAND, h_cmd=st.h_cmd, h_lim=st.h_lim, h_meas=st.h_meas, feet_loaded=st.feet,
                        homed=VC.is_homed(st.cam_abs_max), cam_abs_max=st.cam_abs_max, cmd_speed=st.row.speed,
                        cmd_dir_deg=_deg(st.row.dir), cmd_face_deg=_deg(st.row.face),
                        next_text=VC.next_hold_text(hi if i >= 0 else -1, labels, kinds, heights, starts, dt, T * dt),
                        last_event=last_event, heading_deg=_deg(st.post[2]),
                        speed_xy=math.hypot(st.post[3], st.post[4]), arrows=bool(args_cli.arrows),
                        fall_text=fall_text)
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
                    print(f"[SHV] sequence: {t + 1}/{T} frames ('{lab}', {st.mode}, h_lim {st.h_lim:.3f} h_meas "
                          f"{st.h_meas:.3f}) in {el:.0f}s ({total_frames / max(el, 1e-9):.1f} fps); RSS {rss:.2f} GB",
                          flush=True)
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
        hh = heights[0] if i < 0 else heights[i]
        prev_kind = None if i <= 0 else kinds[i - 1]
        segments.append(_sequence_segment(i, lab, kind, hh, idx, cmd_of[i], dt, rec, falls,
                                          float(sched_dev[idx].max()), float(cmd_dev[idx].max()), prev_kind))
    ru_peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0 / 1024.0
    cg_peak = _cgroup_peak_gb()
    stands = [s for s in segments if "stand" in s]
    restores = [s for s in segments if "restore" in s]
    routing = {
        "rule": "the eval's latched stand slot | H6 restore (sh_core.HeightController.step: stand_eff)",
        "stand_eff_steps": int((rec["stand_eff"] > 0.5).sum()), "restoring_steps": int((rec["restoring"] > 0.5).sum()),
        "steps": T, "route_mismatch_steps": int((rec["route_ok"] < 0.5).sum()),
        "exec_checked_steps": int((rec["exec_checked"] > 0.5).sum()),
        "exec_mismatch_steps": int((rec["exec_ok"] < 0.5).sum()),
        "schedule_vs_slot_disagree_steps": int(((rec["stand_schedule"] > 0.5) != (rec["slot"] > 0.5)).sum()),
        "holds_where_slot_disagrees": [s["hold"] for s in segments if s["schedule_vs_slot_disagree_steps"]],
        "resets_rederived": int(ctx.slot_rule.resets_rederived),
    }
    height_summary = {
        "n_stand_holds": len(stands),
        "arrived": sum(1 for s in stands if s["stand"]["arrival_s"] is not None),
        "window_median_abs_err_m": [None if s["stand"]["window"] is None else s["stand"]["window"]["median_abs_err_m"]
                                    for s in stands],
        "homed_at_end": sum(1 for s in stands if s["stand"]["homing"]["homed_at_end"]),
        "falls_in_stands": sum(s["stand"]["falls"] for s in stands),
        "restores": {k: sum(s["restore"][k] for s in restores) for k in ("engaged", "completed", "timed_out")},
        "events_all": ctl.counts() if ctl is not None else None,
        "deadband_zeroed_entries": int(rec["db_zeroed"].sum()),
        "per_stand": [{"hold": s["hold"], "segment": s["segment"], "hold_s": s["steps"] * dt, **s["stand"]}
                      for s in stands],
        "per_restore": [{"hold": s["hold"], "segment": s["segment"], **s["restore"]} for s in restores],
    }
    print(f"[SHV] wrote {ctx.combined_path} ({total_frames} frames, {total_frames / fps:.1f} s at {fps} fps) "
          f"in {el:.0f}s; trace_sha256 {trace_sha}", flush=True)
    print(f"[SHV] peak RSS (ru_maxrss) {ru_peak:.2f} GB; cgroup memory.peak "
          f"{'n/a' if cg_peak is None else f'{cg_peak:.2f} GB'}", flush=True)
    _print_sequence_table(segments, falls, routing, height_summary)
    meta = {
        "mode": "sequence", "kind": "sh_stand_height_video",
        "checkpoint": str(ctx.resume_path), "checkpoint_sha256": S1_SHA,
        "stand_checkpoint": str(args_cli.stand_ckpt), "stand_checkpoint_sha256": STAND_SHA,
        "stand_mask": int(args_cli.stand_mask), CORE.RESIDUAL_DEADBAND_KEY: DEADBAND,
        "controller": VC.CONTROLLER, "restore": VC.RESTORE, "controller_record": ctl_record,
        "clip": CORE.clip_record(DEADBAND), "provenance": PROVENANCE,
        "caption_tags": ctx.tags, "build": ctx.build, "forward": VC.SH_FORWARD,
        "routing": routing, "height_summary": height_summary,
        "reference_run_meta": str(RUN_META_PATH), "reference_scenario_id": RUN_META["scenario_id"],
        "reference_checks": {k: {"reference": a, "here": b} for k, (a, b) in REFERENCE_CHECKS.items()},
        "reference_mismatch": REFERENCE_MISMATCH,
        "task": scenario.task, **seq_info,
        "env_vars": WANT_ENV, "env_overrides": ctx.applied_overrides,
        "harness": str(HARNESS), "harness_sha256": HARNESS_SHA,
        "recorder": str(Path(__file__).resolve()), "recorder_sha256": _sha256(Path(__file__).resolve()),
        "video_core_sha256": _sha256(VC_PATH), "sh_eval_common_sha256": _sha256(EC_PATH),
        "sh_launcher_sha256": _sha256(SH_LAUNCHER), "p2r_launcher_sha256": _sha256(P2R_LAUNCHER),
        "env_seed": scenario.env_seed, "env_seed_note": "env_cfg.seed only (the harness's single pass)",
        "spawn_yaw_pin": ctx.applied_overrides.get("spawn_yaw_pin"),
        "prologue_s": float(scenario.get_default("prologue_s")), "settle_s": float(scenario.get_default("settle_s")),
        "holds": [{"segment": i, "label": h.label, "kind": kinds[i], "h_cmd": heights[i],
                   "hold_s": float(h.hold_s), "speed": cmd_of[i][0], "dir_deg": _deg(cmd_of[i][1]),
                   "face_deg": _deg(cmd_of[i][2]), "start_step": starts[i], "end_step_exclusive": ends[i],
                   "t0_s": starts[i] * dt, "t1_s": ends[i] * dt}
                  for i, h in enumerate(scenario.schedule)],
        "n_steps": T, "episode_length_s": scenario.episode_length_s,
        "max_episode_length": int(unwrapped.max_episode_length), "dt": dt, "fps": fps,
        "arrows": bool(args_cli.arrows),
        "camera": {"eye": list(ctx.env_cfg.viewer.eye), "lookat": list(ctx.env_cfg.viewer.lookat),
                   "origin_type": ctx.env_cfg.viewer.origin_type, "resolution": list(ctx.env_cfg.viewer.resolution)},
        "captions": _CAPTION_BACKEND, "readings": {
            "h_meas": "sh_core.h_meas(root z, sh_core.terrain_z_under_feet(footpads, mesh, raycast_mesh)), post-step",
            "feet_loaded": f"contact >= {VC.FOOT_LOADED_N} N and footpad <= {VC.FOOT_FLOOR_M} m above the terrain "
                           "(sh_dagger_spec; the trainer's probe), post-step",
            "homed": f"every cam |wrap| <= {VC.HOMED_RAD} rad, post-step"},
        "mp4": str(ctx.combined_path), "frames_total": total_frames,
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
    print(f"[SHV] sequence sanity -> {sanity_path}", flush=True)
    if routing["route_mismatch_steps"] or routing["exec_mismatch_steps"]:
        print(f"[SHV][FAIL] {routing['route_mismatch_steps']} route / {routing['exec_mismatch_steps']} applied-action "
              "mismatches -- the video is NOT the launcher's controller path", flush=True)


def _print_sequence_table(segments: list[dict], falls: list[dict], routing: dict, height_summary: dict) -> None:
    print("[SHV] SEQUENCE SANITY TABLE (net = who acted, share; rst = restoring share; arr = command arrival in the "
          "hold; err = median |h_meas - h_cmd| over the last 2 s; feet = loaded at the end; cam_end = worst |cam|)",
          flush=True)
    hdr = (f"{'#':>2s} {'hold':8s} {'kind':5s} {'h_cmd':>6s} {'t0-t1 s':>13s} | {'net':>9s} {'share':>5s} {'rst':>5s} | "
           f"{'dx':>6s} {'dy':>6s} | {'arr':>5s} {'err mm':>6s} {'feet':>4s} {'cam_end':>7s} | restore | falls")
    print(hdr, flush=True)
    nan = float("nan")
    for r in segments:
        ach = r.get("achieved") or {}
        stn = r.get("stand") or {}
        rst = r.get("restore")
        fl = ",".join(f"{f['t_s']:.2f}s" for f in r["falls"]) or "none"
        num = "P" if r["segment"] < 0 else str(r["segment"] + 1)
        win = stn.get("window")
        cam_end = (stn.get("homing") or {}).get("cam_max_end_rad")
        arr = stn.get("arrival_s")
        hc = r.get("h_cmd")
        rtxt = "   --  " if rst is None else (f"e{rst['engaged']} c{rst['completed']} t{rst['timed_out']}"
                                             + (f" @{rst['completed_at_s']:.1f}s" if rst["completed_at_s"] is not None
                                                else ""))
        hc_txt = "  --  " if hc is None else f"{hc:6.3f}"
        arr_txt = "  -- " if arr is None else f"{arr:5.2f}"
        err_txt = "   -- " if win is None else f"{win['median_abs_err_m'] * 1000:6.1f}"
        feet_txt = "  --" if stn.get("feet_end") is None else f"{stn['feet_end']:4d}"
        cam_txt = "   --  " if cam_end is None else f"{cam_end:7.3f}"
        print(f"{num:>2s} {r['hold']:8s} {r['kind']:5s} {hc_txt} "
              f"{r['t0_s']:6.2f}-{r['t1_s']:6.2f} | {r['acting_network']:>9s} {r['stand_net_share']:5.2f} "
              f"{r['restoring_share']:5.2f} | {ach.get('dx', nan):+6.2f} {ach.get('dy', nan):+6.2f} | "
              f"{arr_txt} {err_txt} {feet_txt} {cam_txt} | {rtxt} | {fl}", flush=True)
    print(f"[SHV] routing: {routing['stand_eff_steps']}/{routing['steps']} steps to the stand net "
          f"({routing['restoring_steps']} restoring); route mismatches {routing['route_mismatch_steps']} (want 0); "
          f"applied-action mismatches {routing['exec_mismatch_steps']} of {routing['exec_checked_steps']} checked "
          f"(want 0); schedule-vs-slot disagreement {routing['schedule_vs_slot_disagree_steps']} steps in "
          f"{routing['holds_where_slot_disagrees']}", flush=True)
    print(f"[SHV] heights: {height_summary['n_stand_holds']} stand holds, command arrived in "
          f"{height_summary['arrived']}, homed at end {height_summary['homed_at_end']}, falls in stands "
          f"{height_summary['falls_in_stands']}; restores {height_summary['restores']}; controller events "
          f"{height_summary['events_all']}; deadband-zeroed residual entries {height_summary['deadband_zeroed_entries']}",
          flush=True)
    print(f"[SHV] terminations in the run: {len(falls)}"
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
