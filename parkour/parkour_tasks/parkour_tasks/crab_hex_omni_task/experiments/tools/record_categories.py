#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""Per-category follow-cam video of an omni head -- recorded through the eval harness's own path
(omni task, 2026-09-22).

One episode per COMMAND HOLD of the ``suite__omni`` row on flat ground, reset between episodes,
composed so that what the video shows is what the per-category eval scored. The suite's ten holds
(``stand``, ``fwd``, ``lat_l``, ``lat_r``, ``back``, ``diag``, ``turn_l90``, ``turn_r90``,
``move_turn``, ``move_turn_r``) are recorded in SUITE WAVE ORDER by default -- both hands of each
paired category (``side``, ``turn``, ``move_turn``), each seeded with that hold's own wave seed::

    record_categories.py --headless --enable_cameras
    record_categories.py --headless --enable_cameras --episode-s 6 --categories turn_l90,move_turn_r
    record_categories.py --headless --enable_cameras --checkpoint <pt> --eval-run <gait_eval run dir>

Every default reproduces the 2026-09-22 C1d recording (the C1d head and the suite eval it mirrors);
every path is a default, not a constant. This is a VIEWER, not a scorer: it writes mp4s and a
sanity JSON, never a ledger entry, and nothing in the campaign reads its output.

WHAT IT GUARANTEES, AND HOW

* ENV COMPOSITION = the suite eval's. The ``KRABBY_*`` environment is replaced, BEFORE Kit boots,
  by exactly ``run_meta["env_vars"]`` of the reference suite eval (``run_phases._subenv`` does the
  same: drop every ``KRABBY_*``, set the recorded ones), and re-checked against that record after
  the task package imports. The task is ``run_meta["task"]`` (the Walk-Play task), the scenario row
  is ``run_meta["scenario_id"]`` from the omni manifest (checked equal to the run's recorded suite
  block), and the harness's own ``_apply_env_overrides`` is applied (seed, frozen terrain levels,
  resampling off, curriculum off, push off, spawn yaw pinned to the settle face 0). Differences
  from the eval, all deliberate: ``num_envs`` 1, ``render_mode`` rgb_array (+ ``--enable_cameras``),
  ``episode_length_s`` = episode + 2 s, the category hold lengthened so settle + hold = the
  requested episode length, and the command arrows (below).
* POLICY = the harness's teacher path: ``make_on_policy_runner`` + ``runner.load``,
  ``get_inference_policy`` + ``get_estimator_inference_policy``, the estimator written into the
  priv-explicit slice every step, ``policy(obs, hist_encoding=True)``, and the wrapper's
  ``clip_actions`` from the agent cfg.
* COMMAND PIN = the harness's own mechanism: ``gait_eval.schedule.compile_wave`` builds the
  ``[prologue(settle cmd), settle, hold]`` row for the hold's own wave, and the harness's
  ``_install_schedule_override_world`` replaces the command term's ``compute`` so every step goes
  through ``schedule_hook(row)`` (world -> body conversion + facing P-control) and never
  ``_resample_command``. The harness functions are NOT copied: they are extracted from
  ``eval_crab_hex_gait.py`` with ``ast`` and executed here, so the video follows the harness if it
  changes (its sha256 is logged and recorded in the sanity JSON).

COMMAND ARROWS. The omni command term already draws what the user wants to see -- GREEN commanded
travel (``travel_dir_w``, ~1.7 m of arrow per m/s of the EFFECTIVE commanded speed, so a standing
slot draws nothing), ORANGE commanded facing (``heading_target``, an absolute world yaw at a fixed
~0.7 m) and BLUE achieved base velocity (``root_lin_vel_w``) -- all world-horizontal, stacked over
the base origin (see ``crab_hex_omni_command._debug_vis_callback``). They read the term's live
state, which under the schedule override is the PINNED row, so they cannot show a stale pool draw.
Nothing has to be added here; the one thing that has to be UNDONE is the harness's
``_apply_env_overrides``, which sets ``commands.base_velocity.debug_vis = False`` (the eval runs
100 envs headless and 2026-09-18 measured ~59 GB RSS with the markers on). ``--arrows`` therefore
re-enables that ONE cfg field after the harness override, on 1 env, and touches neither the
environment nor ``parkours.base_parkour.debug_vis`` (the ring anchors/goal markers: clutter here,
and the other half of that OOM). ``--no-arrows`` reproduces the eval's own visual state; the two
are otherwise the same run, and ``trace_sha256`` in the sanity JSON (a rolling sha256 over every
step's policy input, action and post-step root state) is how that was proved -- it is identical
with the arrows on and off, so the arrows are cosmetic.

Per hold: ``env.seed(wave_seed)`` (the suite's own per-wave seed for that hold), a forced
``env.reset()`` (skipped for the first hold, exactly like the harness's wave 0 -- the wrapper's
construction reset stands), then ``round(episode_s / dt)`` control steps, one rendered frame per
step (fps = 1/dt), streamed to the combined mp4 and the hold's own mp4 (CONSTANT MEMORY: gym's
``RecordVideo`` buffers every frame and has twice crashed this machine on a 5-minute clip).
"""

from __future__ import annotations

import __future__ as _future
import argparse
import ast
import contextlib
import hashlib
import importlib.util
import json
import math
import os
import resource
import sys
import time
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[5]  # tools -> experiments -> crab_hex_omni_task -> parkour_tasks -> parkour_tasks -> parkour -> repo
PARKOUR = REPO / "parkour"
PKG = HERE.parents[1]
FWD_PKG = PKG.parent / "crab_hex_forward_task"
HARNESS = FWD_PKG / "scripts/eval_crab_hex_gait.py"  # shared harness: one harness serves all three tasks
HARNESS_DIR = HARNESS.parent
MANIFEST = PKG / "experiments/eval/scenarios_omni.yaml"
LOG_ROOT = PARKOUR / "logs/rsl_rl"
EVAL_ROOT = LOG_ROOT / "gait_eval/omni_phases"
#: Defaults = the 2026-09-22 recording: the C1d head of the facing-tracking ladder and the suite
#: eval that adopted it. Both are overridable (``--checkpoint`` / ``--eval-run``).
DEFAULT_CKPT = LOG_ROOT / "crab_hex_omni_flat_walk/2026-09-22_11-58-07_seed3_X08_c6_A15pB/model_7996.pt"
DEFAULT_EVAL_RUN = (
    EVAL_ROOT / "2026-09-19_facing_tracking_ladder/seed3_X08_c6_A15pB/suite__omni/seed001/2026-09-22_16-48-07"
)
#: mp4s are git-ignored everywhere, so the videos go under ``parkour/logs/`` like every other bulk
#: artifact -- NOT into the campaign tree (a 320 MB combined clip is not a record) and not into a
#: session scratchpad (this is a repo tool; the next caller has a different one).
DEFAULT_OUT = PARKOUR / "logs/videos"
HANDS = {"left": ("l",), "right": ("r",), "both": ("l", "r")}

# cli_args exactly as the harness loads it (never put parkour/scripts on sys.path: it shadows pip rsl_rl).
_cli_spec = importlib.util.spec_from_file_location("parkour_rsl_rl_cli_args", PARKOUR / "scripts/rsl_rl/cli_args.py")
cli_args = importlib.util.module_from_spec(_cli_spec)
_cli_spec.loader.exec_module(cli_args)

from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser(description="Per-hold omni video, recorded through the eval harness's path.")
parser.add_argument("--episode-s", type=float, default=30.0, help="Seconds per episode (settle + hold).")
parser.add_argument(
    "--categories", type=str, default="all",
    help="CSV of category names (stand,fwd,side,back,diag,turn,move_turn) and/or hold labels "
         "(lat_l,turn_r90,move_turn_r,...), or 'all' (default). A category expands to its holds "
         "filtered by --hands; a hold LABEL names one hold and ignores --hands. Recorded in suite "
         "wave order whatever the order given.",
)
parser.add_argument(
    "--hands", choices=tuple(HANDS), default="both",
    help="Which hold(s) of a PAIRED category (side/turn/move_turn) to record; singleton categories "
         "are unaffected (default both -- 'left' is the pre-2026-09-22 behaviour).",
)
parser.add_argument("--out-dir", type=str, default=str(DEFAULT_OUT), help=f"Output directory (default {DEFAULT_OUT}).")
parser.add_argument("--prefix", type=str, default="c1d", help="File prefix: <prefix>_categories.mp4, <prefix>_<hold>.mp4.")
parser.add_argument(
    "--eval-run", type=str, default=str(DEFAULT_EVAL_RUN),
    help="The suite gait_eval run whose env this reproduces: its directory, or its run_meta.json.",
)
parser.add_argument("--label", type=str, default="C1d head (seed3 X08 c6) model_7996", help="Caption title.")
# The task viewer (eye -4,0.5,1.55 / lookat 0,0,0.35) crops the feet: the root rides ~1.08 m above the
# ground and the feet sit ~1.2 m out to each side, so the default here is a rear-right 3/4 view from
# ~6.6 m looking at mid-leg height. 'task' keeps the task's own viewer.
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
parser.add_argument("--trace-npz", action="store_true",
                    help="Also dump every recorded per-step array per hold (the arrows-are-cosmetic diff).")
parser.add_argument("--rss-abort-gb", type=float, default=20.0, help="Abort before stepping if RSS exceeds this (DDC balloon).")
parser.add_argument("--disable_fabric", action="store_true", default=False)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.checkpoint is None:
    args_cli.checkpoint = str(DEFAULT_CKPT)
RUN_META_PATH = Path(args_cli.eval_run)
if RUN_META_PATH.is_dir():
    RUN_META_PATH = RUN_META_PATH / "run_meta.json"
if not RUN_META_PATH.is_file():
    raise SystemExit(f"--eval-run {args_cli.eval_run!r}: no run_meta.json at {RUN_META_PATH}")


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
RUN_META = json.loads(RUN_META_PATH.read_text())
WANT_ENV = {k: str(v) for k, v in RUN_META["env_vars"].items()}
_dropped = sorted(k for k in os.environ if k.startswith("KRABBY_") and os.environ[k] != WANT_ENV.get(k))
for _k in [k for k in os.environ if k.startswith("KRABBY_")]:
    del os.environ[_k]
os.environ.update(WANT_ENV)
print(f"[REC] KRABBY_* env replaced by run_meta env_vars ({len(WANT_ENV)} vars); "
      f"pre-existing differing/extra vars dropped: {_dropped or 'none'}", flush=True)
for _k, _v in sorted(WANT_ENV.items()):
    print(f"[REC]   {_k}={_v}", flush=True)

_t_boot = time.time()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
print(f"[REC] Kit up in {time.time() - _t_boot:.1f}s, RSS {_rss_gb():.2f} GB", flush=True)
if _rss_gb() > args_cli.rss_abort_gb:
    print(f"[REC][ABORT] RSS {_rss_gb():.1f} GB > {args_cli.rss_abort_gb} GB right after Kit start -- "
          "suspect DerivedDataCache corruption; not stepping.", flush=True)
    simulation_app.close()
    sys.exit(3)

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


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


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


def _select_holds(suite, spec: str, hands: tuple[str, ...]) -> list[int]:
    """``--categories`` -> the wave indices to record, in SUITE WAVE ORDER (deduplicated).

    A token is either a category name (expanded to the holds whose ``hand`` is in ``hands``, or to
    its single hold on an unpaired category) or a hold LABEL (that one hold, whatever ``--hands``
    says -- naming ``lat_r`` and then being handed ``lat_l`` would be a trap). The CATEGORY reading
    wins where the two namespaces collide: ``move_turn`` is both a category and its left hold's
    label, and the category is what a caller asking for "move_turn" means (``move_turn`` alone,
    with ``--hands left``, is how to get just that hold). Wave order is the hold's index in
    ``suite.holds()``, which is what ``wave_seed`` and ``compile_wave`` key on.
    """
    all_holds = suite.holds()
    by_label = {h.label: i for i, h in enumerate(all_holds)}
    by_category: dict[str, list[int]] = {}
    for cat in suite.categories:
        by_category[cat.name] = [by_label[h.label] for h in cat.holds]
    wanted: set[int] = set()
    tokens = [t.strip() for t in spec.split(",") if t.strip()]
    if tokens == ["all"]:
        tokens = [c.name for c in suite.categories]
    for tok in tokens:
        if tok in by_category:
            waves = [w for w in by_category[tok] if all_holds[w].hand is None or all_holds[w].hand in hands]
            if not waves:
                raise SystemExit(f"category {tok!r} has no hold for --hands {args_cli.hands}")
            wanted.update(waves)
        elif tok in by_label:
            wanted.add(by_label[tok])
        else:
            raise SystemExit(
                f"unknown category/hold {tok!r}; categories {[c.name for c in suite.categories]}, "
                f"holds {list(by_label)}"
            )
    return sorted(wanted)


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
        # Measure, don't guess: the arrow legend is wider than the old fixed 900 px and its tail
        # rendered on the bare scene, where cyan on pale terrain is unreadable.
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
#: The HUD's key to the three arrows the command term draws (``crab_hex_omni_command``'s
#: TRAVEL / FACING / ACHIEVED_ARROW_COLOR). One line, one colour: cv2's Hershey font has no
#: per-word colouring and a caption is not worth three draw calls per frame.
ARROW_LEGEND = "arrows: GREEN = commanded travel   ORANGE = commanded facing   BLUE = achieved velocity"
HAND_WORD = {"l": "LEFT", "r": "RIGHT"}


def main() -> None:
    out_dir = Path(args_cli.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    H = _load_harness_functions()
    harness_sha = _sha256(HARNESS)
    print(f"[REC] harness functions extracted from {HARNESS} (sha256 {harness_sha[:16]}...)", flush=True)

    # ---- the scenario row, checked against what the reference eval actually ran
    scenarios, _ = S.load_manifest(MANIFEST)
    matches = [s for s in scenarios if s.id == RUN_META["scenario_id"]]
    if not matches:
        raise SystemExit(f"{RUN_META['scenario_id']!r} not in {MANIFEST}")
    scenario = matches[0]
    if scenario.task != RUN_META["task"]:
        raise SystemExit(f"manifest task {scenario.task} != run_meta task {RUN_META['task']}")
    suite = scenario.suite
    if suite is None:
        raise SystemExit(f"scenario {scenario.id!r} is not a category_suite row; this recorder needs one")
    rm_suite = RUN_META["suite"]
    if suite.categories_meta() != rm_suite["categories"]:
        raise SystemExit("manifest suite categories differ from the reference run's recorded suite")
    prologue_s = float(scenario.get_default("prologue_s"))
    if prologue_s != float(rm_suite["prologue_s"]) or suite.settle_hold_s != float(rm_suite["settle_hold_s"]):
        raise SystemExit("manifest prologue/settle differ from the reference run's")
    hands = HANDS[args_cli.hands]
    waves = _select_holds(suite, args_cli.categories, hands)

    episode_s = float(args_cli.episode_s)
    lead_s = prologue_s + float(suite.settle_hold_s)
    hold_s = episode_s - lead_s
    if hold_s <= 0.0:
        raise SystemExit(f"--episode-s {episode_s} leaves no hold after the suite's {lead_s} s settle lead-in")
    manifest_hold_s = float(suite.hold_s)
    suite.hold_s = hold_s
    scenario.schedule = S.suite_schedule(suite)
    scenario.episodes = 1
    scenario.episode_length_s = episode_s + 2.0
    try:
        S.validate_scenario(scenario)
        print("[REC] retargeted suite row passes the harness validator", flush=True)
    except S.ManifestError as exc:
        print(f"[REC][WARN] retargeted row fails the harness validator (expected only for short smoke "
              f"episodes): {exc}", flush=True)
    print(f"[REC] EPISODE SPLIT: {episode_s:.2f} s = prologue {prologue_s:.2f} s (settle cmd: speed 0, face 0) "
          f"+ settle {suite.settle_hold_s:.2f} s (speed 0, face 0) + hold {hold_s:.2f} s "
          f"(the suite scores a {manifest_hold_s:.1f} s hold; lengthened here). episode_length_s = "
          f"{scenario.episode_length_s:.1f} s (no timeout inside an episode).", flush=True)

    all_holds = suite.holds()
    category_of = suite.category_of_hold()
    plan = []
    for wave in waves:
        hold = all_holds[wave]
        plan.append((category_of[hold.label], hold, wave))
        print(f"[REC] plan: {category_of[hold.label]:9s} hold {hold.label:12s} hand {hold.hand or '-'} "
              f"speed {hold.speed:.2f} dir {_deg(hold.dir):+7.2f} deg face {_deg(hold.face):+7.2f} deg  "
              f"wave {wave} seed {S.wave_seed(scenario, wave)}", flush=True)

    # ---- env, composed as the harness composes it
    env_cfg = parse_env_cfg(scenario.task, device=args_cli.device, num_envs=1, use_fabric=not args_cli.disable_fabric)
    applied_overrides = H["_apply_env_overrides"](env_cfg, scenario)
    print(f"[REC] harness env overrides applied: {applied_overrides} (reference run: {RUN_META['env_overrides']})", flush=True)
    # The ONE cfg field this recorder takes back off the harness (see the module docstring): the
    # command term's own three arrows. The parkour term's markers stay off -- they are the ring
    # anchors, not the command, and they are the other half of the 2026-09-18 eval OOM.
    env_cfg.commands.base_velocity.debug_vis = bool(args_cli.arrows)
    _parkour_vis = getattr(getattr(env_cfg, "parkours", None), "base_parkour", None)
    print(f"[REC] command debug arrows: {'ON' if args_cli.arrows else 'OFF'} "
          f"(commands.base_velocity.debug_vis={env_cfg.commands.base_velocity.debug_vis}; "
          f"parkours.base_parkour.debug_vis stays "
          f"{'n/a' if _parkour_vis is None else _parkour_vis.debug_vis})", flush=True)
    eye, lookat = _parse_vec3(args_cli.cam_eye), _parse_vec3(args_cli.cam_lookat)
    if eye is not None:
        env_cfg.viewer.eye = eye
    if lookat is not None:
        env_cfg.viewer.lookat = lookat
    env_cfg.viewer.env_index = 0
    print(f"[REC] camera: origin_type={env_cfg.viewer.origin_type} asset={env_cfg.viewer.asset_name} "
          f"eye={tuple(env_cfg.viewer.eye)} lookat={tuple(env_cfg.viewer.lookat)} (offsets from the robot root, "
          f"world-aligned -- the camera translates with the robot but does not yaw with it) "
          f"resolution={tuple(env_cfg.viewer.resolution)}", flush=True)
    tg = getattr(getattr(env_cfg.scene, "terrain", None), "terrain_generator", None)
    if tg is not None:
        subs = {k: type(v).__name__ for k, v in tg.sub_terrains.items()}
        print(f"[REC] terrain sub_terrains: {subs}", flush=True)

    agent_cfg = cli_args.parse_rsl_rl_cfg(scenario.task, args_cli)
    env = ParkourRslRlVecEnvWrapper(
        gym.make(scenario.task, cfg=env_cfg, render_mode="rgb_array"),
        clip_actions=agent_cfg.clip_actions,
    )
    unwrapped = env.unwrapped
    device = env.device
    dt = float(unwrapped.step_dt)
    print(f"[REC] env up: dt={dt} max_episode_length={int(unwrapped.max_episode_length)} steps, RSS {_rss_gb():.2f} GB",
          flush=True)
    if _rss_gb() > args_cli.rss_abort_gb:
        raise SystemExit(f"RSS {_rss_gb():.1f} GB > {args_cli.rss_abort_gb} GB before the first step -- aborting")

    # KRABBY_* as the config saw it must equal the reference run's record.
    krabby_now = {k: v for k, v in sorted(os.environ.items()) if k.startswith("KRABBY_")}
    if krabby_now != WANT_ENV:
        raise SystemExit(f"KRABBY_* env after import differs from run_meta: now={krabby_now} want={WANT_ENV}")
    print("[REC] KRABBY_* env after task import == reference run_meta env_vars", flush=True)

    robot = unwrapped.scene["robot"]
    usd_spawned = str(getattr(getattr(robot.cfg, "spawn", None), "usd_path", "<unknown>"))
    if Path(usd_spawned).name != Path(RUN_META["usd_path"]).name:
        raise SystemExit(f"spawned {usd_spawned} but the reference run spawned {RUN_META['usd_path']}")
    print(f"[REC] plant usd {usd_spawned} (reference {RUN_META['usd_path']}, plant {RUN_META['plant']})", flush=True)

    # ---- policy, built as the harness builds it (teacher path)
    resume_path = retrieve_file_path(str(args_cli.checkpoint))
    ckpt_sha = _sha256(Path(resume_path))
    if Path(resume_path).resolve() != Path(RUN_META["checkpoint"]).resolve() or ckpt_sha != RUN_META["checkpoint_sha256"]:
        print(f"[REC][WARN] checkpoint differs from the reference eval's ({RUN_META['checkpoint']}, "
              f"sha {RUN_META['checkpoint_sha256'][:16]}...)", flush=True)
    print(f"[REC] checkpoint {resume_path} sha256 {ckpt_sha[:16]}... "
          f"({'==' if ckpt_sha == RUN_META['checkpoint_sha256'] else '!='} reference)", flush=True)
    runner = make_on_policy_runner(env, agent_cfg_to_train_dict(agent_cfg), log_dir=None, device=device)
    runner.load(resume_path)
    estimator_paras = agent_cfg.to_dict()["estimator"]
    num_prop = int(estimator_paras["num_prop"])
    num_scan = int(estimator_paras["num_scan"])
    num_priv_explicit = int(estimator_paras["num_priv_explicit"])
    if agent_cfg.algorithm.class_name == "DistillationWithExtractor":
        raise SystemExit("this recorder runs the teacher path only; the agent cfg is a distillation runner")
    policy = runner.get_inference_policy(device=device)
    estimator = runner.get_estimator_inference_policy(device=device)
    print(f"[REC] policy: teacher, hist_encoding=True, clip_actions={agent_cfg.clip_actions}, "
          f"num_prop={num_prop} num_scan={num_scan} num_priv_explicit={num_priv_explicit} "
          f"(estimator -> obs[{num_prop + num_scan}:{num_prop + num_scan + num_priv_explicit}])", flush=True)

    cmd_term = unwrapped.command_manager.get_term("base_velocity")
    if not isinstance(cmd_term, UniformParkourCommand):
        raise SystemExit(f"base_velocity must be UniformParkourCommand, got {type(cmd_term)}")
    if args_cli.arrows and not hasattr(cmd_term, "travel_cmd_visualizer"):
        raise SystemExit("--arrows: the command term built no visualizers -- debug_vis did not reach it "
                         "(is this the omni command term, and was --enable_cameras passed?)")
    observed_slots, slots_src = H["_observed_command_slots"](unwrapped)
    print(f"[REC] observed command slots {observed_slots} ({slots_src}); reference {RUN_META['observed_command_slots']}",
          flush=True)
    print("[REC] AFTER A TERMINATION the env auto-resets that env INSIDE env.step() (_reset_idx: respawn at the "
          "tile spawn with the pinned yaw 0, joints reset, command term reset -> _resample_command); the SAME "
          "env.step() then runs command_manager.compute() == the harness override -> schedule_hook(row), so the "
          "pinned command is back in force before the next observation. Recording continues to the end of the "
          "episode; the caption carries 'FELL at t=...' from then on.", flush=True)

    cam = getattr(unwrapped, "viewport_camera_controller", None)
    if cam is None:
        print("[REC][WARN] no viewport camera controller (is --enable_cameras set?) -- camera cannot follow", flush=True)

    def _render() -> np.ndarray:
        if cam is not None and env_cfg.viewer.origin_type == "asset_root":
            cam.update_view_to_asset_root(env_cfg.viewer.asset_name)  # no one-render lag after a respawn
        # The arrows are redrawn by the command term's post-update subscription, which Kit pops
        # inside this render -- so every frame carries the command of the step it belongs to.
        frame = unwrapped.render()
        return frame

    # ---- video writers
    import imageio

    fps = int(round(1.0 / dt))
    if abs(fps * dt - 1.0) > 1e-6:
        print(f"[REC][WARN] 1/dt={1.0 / dt} is not an integer fps; video plays at {fps} fps", flush=True)

    def _writer(path: Path):
        return imageio.get_writer(str(path), fps=fps, codec="libx264", quality=args_cli.quality, macro_block_size=None,
                                  ffmpeg_log_level="error")

    combined_path = out_dir / f"{args_cli.prefix}_categories.mp4"
    stills_dir = out_dir / "stills"
    if not args_cli.no_stills:
        stills_dir.mkdir(parents=True, exist_ok=True)
    combined = _writer(combined_path)
    print(f"[REC] captions: {_CAPTION_BACKEND or 'NONE (neither cv2 nor PIL importable)'}; fps {fps}; "
          f"combined -> {combined_path}", flush=True)

    # Warm the renderer up (its first frames come back empty); nothing here steps physics.
    for i in range(60):
        f0 = _render()
        if f0 is not None and f0.size and float(f0.mean()) > 1.0:
            print(f"[REC] renderer warm after {i + 1} render(s); frame {f0.shape}", flush=True)
            break

    per_hold_results = []
    total_frames = 0
    t_run = time.time()
    hold_writer = None
    try:
        for k, (cat_name, hold, wave) in enumerate(plan):
            compiled = S.compile_wave(scenario, wave, dt=dt, num_envs=1)
            T = int(compiled.n_steps)
            hold_seg = wave + 1
            seed = S.wave_seed(scenario, wave)
            env.seed(seed)
            with (torch.inference_mode() if k > 0 else contextlib.nullcontext()):
                if k > 0:
                    env.reset()  # forced reset between holds: respawn at the pinned yaw 0
                override_state = H["_install_schedule_override_world"](cmd_term, compiled, device, compiled.hold_labels)
                obs, extras = env.get_observations()
                override_state["apply"](0)
                obs, extras = env.get_observations()
            seg = compiled.segment_id[:, 0]
            hold_start = int(np.argmax(seg == hold_seg))
            start_state = {
                "x": float(robot.data.root_pos_w[0, 0].item()), "y": float(robot.data.root_pos_w[0, 1].item()),
                "z": float(robot.data.root_pos_w[0, 2].item()),
                "heading_deg": _deg(float(robot.data.heading_w[0].item())),
                "episode_length_buf": int(unwrapped.episode_length_buf[0].item()),
            }
            print(f"[REC] {hold.label}: start state after {'forced env.reset()' if k > 0 else 'the wrapper construction reset'}: "
                  f"root ({start_state['x']:+.3f}, {start_state['y']:+.3f}, {start_state['z']:.3f}) m, heading "
                  f"{start_state['heading_deg']:+.2f} deg, episode_length_buf {start_state['episode_length_buf']}", flush=True)
            print(f"[REC] === hold {k + 1}/{len(plan)} {hold.label} (category {cat_name}, hand {hold.hand or '-'}, "
                  f"wave {wave}, seed {seed}): {T} steps; steps 0-{hold_start - 1} settle "
                  f"(t 0-{hold_start * dt:.2f} s), steps {hold_start}-{T - 1} hold "
                  f"(t {hold_start * dt:.2f}-{T * dt:.2f} s)", flush=True)
            hold_path = out_dir / f"{args_cli.prefix}_{hold.label}.mp4"
            hold_writer = _writer(hold_path)
            still_at = {hold_start - 1: "settle_end", hold_start + (T - hold_start) // 2: "hold_mid", T - 1: "end"}

            rec = {key: np.zeros(T, dtype=np.float64) for key in (
                "row_speed", "row_dir", "row_face", "speed_cmd", "travel_dir_w", "heading_target",
                "cmd_bx", "cmd_by", "cmd_wz", "heading_pre", "x_pre", "y_pre",
                "x_post", "y_post", "heading_post", "vx_w", "vy_w", "done", "fail")}
            falls: list[dict] = []
            steps_since_reset = int(unwrapped.episode_length_buf[0].item())
            max_cmd_dev = 0.0
            max_sched_dev = 0.0
            # The arrows-are-cosmetic witness: a rolling hash over the policy's INPUT, its OUTPUT and
            # the resulting root state, every step. Two runs that differ only in --arrows must land
            # on the same digest (they do -- see the module docstring).
            trace = hashlib.sha256()
            with torch.inference_mode():
                for t in range(T):
                    # --- the harness's world-frame loop body (eval_crab_hex_gait.py main loop)
                    override_state["step"] = t + 1
                    obs_reflects_cmd = cmd_term.vel_command_b.clone()
                    row_t = override_state["row"](t)
                    max_sched_dev = max(
                        max_sched_dev,
                        (cmd_term.travel_dir_w - row_t.dir).abs().max().item(),
                        (cmd_term.speed_cmd - row_t.speed).abs().max().item(),
                        H["_wrap_to_pi"](cmd_term.heading_target - row_t.face).abs().max().item(),
                    )
                    for _slot, _channel in observed_slots.items():
                        max_cmd_dev = max(max_cmd_dev, (obs[:, _slot] - obs_reflects_cmd[:, _channel]).abs().max().item())
                    obs[:, num_prop + num_scan: num_prop + num_scan + num_priv_explicit] = estimator.inference(
                        obs[:, :num_prop]
                    )
                    actions = policy(obs, hist_encoding=True)
                    trace.update(obs.detach().cpu().numpy().tobytes())
                    trace.update(actions.detach().cpu().numpy().tobytes())

                    pre = torch.cat([
                        cmd_term.speed_cmd[:1], cmd_term.travel_dir_w[:1], cmd_term.heading_target[:1],
                        cmd_term.vel_command_b[0, :3], robot.data.heading_w[:1], robot.data.root_pos_w[0, :2],
                    ]).double().cpu().numpy()
                    obs, _, dones, extras = env.step(actions)
                    try:
                        failed = bool(unwrapped.termination_manager.get_term("crab_failure")[0].item())
                    except (KeyError, AttributeError):
                        failed = False
                    done = bool(dones.reshape(-1)[0].item())
                    post = torch.cat([
                        robot.data.root_pos_w[0, :2], robot.data.heading_w[:1], robot.data.root_lin_vel_w[0, :2],
                    ]).double().cpu().numpy()
                    trace.update(post.tobytes())

                    rec["row_speed"][t], rec["row_dir"][t], rec["row_face"][t] = row_t.speed, row_t.dir, row_t.face
                    (rec["speed_cmd"][t], rec["travel_dir_w"][t], rec["heading_target"][t], rec["cmd_bx"][t],
                     rec["cmd_by"][t], rec["cmd_wz"][t], rec["heading_pre"][t], rec["x_pre"][t], rec["y_pre"][t]) = pre
                    rec["x_post"][t], rec["y_post"][t], rec["heading_post"][t], rec["vx_w"][t], rec["vy_w"][t] = post
                    rec["done"][t], rec["fail"][t] = float(done), float(failed)
                    steps_since_reset += 1
                    t_s = (t + 1) * dt
                    if done:
                        if failed:
                            reason = "fall (crab_failure)"
                        elif steps_since_reset >= int(unwrapped.max_episode_length) - 1:
                            reason = "timeout"
                        else:
                            reason = "terminated_other (roll/pitch/height cutoff)"
                        falls.append({"t_s": round(t_s, 3), "step": t, "reason": reason,
                                      "phase": "hold" if seg[t] == hold_seg else "settle"})
                        print(f"[REC] {hold.label}: TERMINATION at t={t_s:.2f} s (step {t}): {reason} -> env auto-reset "
                              "(respawn yaw 0); pinned command re-applied by the override in the same step", flush=True)
                        steps_since_reset = 0

                    # --- frame
                    frame = _render()
                    if frame is None:
                        raise SystemExit("render() returned None -- is render_mode rgb_array / --enable_cameras set?")
                    if _CAPTION_BACKEND is not None:
                        in_hold = seg[t] == hold_seg
                        speed_now = math.hypot(post[3], post[4])
                        hand_txt = f"  [{HAND_WORD[hold.hand]} hand]" if hold.hand else ""
                        lines = [(f"{args_cli.label}  |  {k + 1}/{len(plan)}: {cat_name.upper()} - "
                                  f"'{hold.label}'{hand_txt}", WHITE)]
                        if in_hold:
                            lines.append((f"CMD  speed {row_t.speed:.2f} m/s   dir {_deg(row_t.dir):+.0f} deg   "
                                          f"face {_deg(row_t.face):+.0f} deg   (world frame)", YELLOW))
                        else:
                            lines.append((f"SETTLE  speed 0.00  face +0 deg  -> '{hold.label}' starts at "
                                          f"t={hold_start * dt:.1f} s", CYAN))
                        lines.append((f"t = {t_s:5.2f} / {T * dt:.1f} s    heading {_deg(post[2]):+6.1f} deg    "
                                      f"|v| {speed_now:.2f} m/s", WHITE))
                        if args_cli.arrows:
                            lines.append((ARROW_LEGEND, CYAN))
                        if falls:
                            lines.append((f"FELL at t={falls[0]['t_s']:.2f} s -> auto-reset, command still pinned"
                                          + (f" ({len(falls)} terminations)" if len(falls) > 1 else ""), RED))
                        frame = _caption(frame, lines)
                    else:
                        frame = np.ascontiguousarray(frame[:, :, :3])
                    combined.append_data(frame)
                    hold_writer.append_data(frame)
                    total_frames += 1
                    if not args_cli.no_stills and t in still_at:
                        imageio.imwrite(str(stills_dir / f"{args_cli.prefix}_{hold.label}_{still_at[t]}.png"), frame)
                    if (t + 1) % 500 == 0:
                        el = time.time() - t_run
                        print(f"[REC] {hold.label}: {t + 1}/{T} frames; {total_frames} total in {el:.0f}s "
                              f"({total_frames / max(el, 1e-9):.1f} fps); RSS {_rss_gb():.2f} GB", flush=True)
            hold_writer.close()
            hold_writer = None
            per_hold_results.append(_sanity(cat_name, hold, wave, seed, T, dt, seg, hold_seg, rec, falls,
                                            max_cmd_dev, max_sched_dev, hold_path))
            per_hold_results[-1]["start_state"] = start_state
            per_hold_results[-1]["trace_sha256"] = trace.hexdigest()
            if args_cli.trace_npz:
                trace_path = out_dir / f"{args_cli.prefix}_{hold.label}_trace.npz"
                np.savez(trace_path, **rec)
                per_hold_results[-1]["trace_npz"] = str(trace_path)
            print(f"[REC] {hold.label}: wrote {hold_path} ({T} frames); trace_sha256 "
                  f"{per_hold_results[-1]['trace_sha256']}", flush=True)
    finally:
        if hold_writer is not None:
            hold_writer.close()
        combined.close()

    el = time.time() - t_run
    ru_peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0 / 1024.0
    cg_peak = _cgroup_peak_gb()
    print(f"[REC] wrote {combined_path} ({total_frames} frames, {total_frames / fps:.1f} s at {fps} fps) "
          f"in {el:.0f}s ({total_frames / max(el, 1e-9):.1f} frames/s)", flush=True)
    print(f"[REC] peak RSS (ru_maxrss) {ru_peak:.2f} GB; cgroup memory.peak "
          f"{'n/a' if cg_peak is None else f'{cg_peak:.2f} GB'}", flush=True)
    _print_table(per_hold_results)
    meta = {
        "checkpoint": str(resume_path), "checkpoint_sha256": ckpt_sha,
        "reference_run_meta": str(RUN_META_PATH), "task": scenario.task, "scenario_id": scenario.id,
        "env_vars": WANT_ENV, "env_overrides": applied_overrides, "harness": str(HARNESS), "harness_sha256": harness_sha,
        "episode_s": episode_s, "prologue_s": prologue_s, "settle_s": float(suite.settle_hold_s), "hold_s": hold_s,
        "episode_length_s": scenario.episode_length_s, "dt": dt, "fps": fps,
        "hands": args_cli.hands, "categories_arg": args_cli.categories, "waves": waves,
        "arrows": bool(args_cli.arrows),
        "camera": {"eye": list(env_cfg.viewer.eye), "lookat": list(env_cfg.viewer.lookat),
                   "origin_type": env_cfg.viewer.origin_type, "resolution": list(env_cfg.viewer.resolution)},
        "captions": _CAPTION_BACKEND, "combined_mp4": str(combined_path), "frames_total": total_frames,
        "peak_rss_gb": ru_peak, "cgroup_peak_gb": cg_peak, "holds": per_hold_results,
    }
    sanity_path = out_dir / f"{args_cli.prefix}_sanity.json"
    sanity_path.write_text(json.dumps(meta, indent=1))
    print(f"[REC] sanity -> {sanity_path}", flush=True)
    if cam is not None:
        # Drop the camera's post-update subscription first: otherwise it fires once more during
        # simulation_app.close() against the already-closed scene (a harmless but noisy traceback).
        cam._viewport_camera_update_handle = None
    env.close()


def _sanity(cat_name, hold, wave, seed, T, dt, seg, hold_seg, rec, falls, max_cmd_dev, max_sched_dev, path) -> dict:
    hmask = seg == hold_seg
    idx = np.nonzero(hmask)[0]
    first_fall_step = next((f["step"] for f in falls), None)
    ok_idx = idx if first_fall_step is None else idx[idx < first_fall_step]
    speed_body = np.hypot(rec["cmd_bx"], rec["cmd_by"])
    moving = ok_idx[speed_body[ok_idx] > 1e-6]
    world_dir_of_body_cmd = rec["heading_pre"] + np.arctan2(rec["cmd_by"], rec["cmd_bx"])
    out = {
        "category": cat_name, "hold": hold.label, "hand": hold.hand, "wave": wave, "seed": seed,
        "frames": T, "mp4": str(path),
        "hold_steps": int(idx.size), "hold_t0_s": float(idx[0] * dt) if idx.size else None,
        "commanded": {"speed": hold.speed, "dir_deg": _deg(hold.dir), "face_deg": _deg(hold.face)},
        "applied_mean_hold": {
            "speed": float(rec["speed_cmd"][idx].mean()),
            "dir_deg": _deg(_circmean(rec["travel_dir_w"][idx])),
            "face_deg": _deg(_circmean(rec["heading_target"][idx])),
        },
        "applied_max_dev_hold": {
            "speed": float(np.abs(rec["speed_cmd"][idx] - hold.speed).max()),
            "dir_rad": float(np.abs(np.arctan2(np.sin(rec["travel_dir_w"][idx] - hold.dir),
                                                np.cos(rec["travel_dir_w"][idx] - hold.dir))).max()),
            "face_rad": float(np.abs(np.arctan2(np.sin(rec["heading_target"][idx] - hold.face),
                                                 np.cos(rec["heading_target"][idx] - hold.face))).max()),
        },
        "schedule_state_max_dev_episode": max_sched_dev,
        "obs_command_slot_max_dev_episode": max_cmd_dev,
        "body_cmd_mean_hold": {
            "speed_xy": float(speed_body[idx].mean()),
            "world_dir_deg": _deg(_circmean(world_dir_of_body_cmd[moving])) if moving.size else None,
            "wz": float(rec["cmd_wz"][idx].mean()),
        },
        "falls": falls,
    }
    if ok_idx.size:
        x0, y0 = rec["x_pre"][ok_idx[0]], rec["y_pre"][ok_idx[0]]
        x1, y1 = rec["x_post"][ok_idx[-1]], rec["y_post"][ok_idx[-1]]
        dx, dy = x1 - x0, y1 - y0
        dur = ok_idx.size * dt
        along = dx * math.cos(hold.dir) + dy * math.sin(hold.dir)
        across = -dx * math.sin(hold.dir) + dy * math.cos(hold.dir)
        out["achieved_hold"] = {
            "until": "end" if first_fall_step is None else f"first fall t={falls[0]['t_s']}",
            "duration_s": dur, "dx": dx, "dy": dy, "dist": math.hypot(dx, dy),
            "along_dir": along, "across_dir": across, "mean_speed_along": along / dur,
            "travel_dir_deg": _deg(math.atan2(dy, dx)) if math.hypot(dx, dy) > 0.05 else None,
            "heading_end_deg": _deg(rec["heading_post"][ok_idx[-1]]),
            "heading_err_end_deg": _deg(_wrap(rec["heading_post"][ok_idx[-1]] - hold.face)),
        }
    out["heading_final_deg"] = _deg(rec["heading_post"][T - 1])
    out["heading_err_final_deg"] = _deg(_wrap(rec["heading_post"][T - 1] - hold.face))
    return out


def _print_table(rows: list[dict]) -> None:
    print("[REC] SANITY TABLE (hold = the hold's own command; applied = the command term's world state "
          "while the hold ran; achieved = hold start -> hold end or first fall)", flush=True)
    hdr = (f"{'hold':12s} {'cat':9s} {'hand':4s} | {'cmd spd/dir/face':>20s} | {'applied spd/dir/face':>21s} "
           f"{'maxdev':>7s} | {'body|v| wdir':>12s} | {'dx':>6s} {'dy':>6s} {'along':>6s} {'v_al':>5s} | "
           f"{'head_end':>8s} {'err':>6s} | falls")
    print(hdr, flush=True)
    for r in rows:
        c, a, d, b = r["commanded"], r["applied_mean_hold"], r["applied_max_dev_hold"], r["body_cmd_mean_hold"]
        ach = r.get("achieved_hold") or {}
        maxdev = max(d["speed"], d["dir_rad"], d["face_rad"], r["schedule_state_max_dev_episode"])
        wdir = "  --" if b["world_dir_deg"] is None else f"{b['world_dir_deg']:+5.0f}"
        falls = ",".join(f"{f['t_s']:.2f}s" for f in r["falls"]) or "none"
        print(f"{r['hold']:12s} {r['category']:9s} {r['hand'] or '-':4s} | "
              f"{c['speed']:.2f}/{c['dir_deg']:+5.0f}/{c['face_deg']:+5.0f} deg | "
              f"{a['speed']:.2f}/{a['dir_deg']:+5.0f}/{a['face_deg']:+5.0f} deg {maxdev:7.1e} | "
              f"{b['speed_xy']:.2f} {wdir} | {ach.get('dx', float('nan')):+6.2f} {ach.get('dy', float('nan')):+6.2f} "
              f"{ach.get('along_dir', float('nan')):+6.2f} {ach.get('mean_speed_along', float('nan')):+5.2f} | "
              f"{ach.get('heading_end_deg', float('nan')):+8.1f} {ach.get('heading_err_end_deg', float('nan')):+6.1f} | "
              f"{falls}", flush=True)
    print("[REC] obs-command-slot max dev per episode: "
          + ", ".join(f"{r['hold']} {r['obs_command_slot_max_dev_episode']:.1e}" for r in rows), flush=True)
    print("[REC] trace sha256 per hold: " + ", ".join(f"{r['hold']} {r['trace_sha256'][:16]}" for r in rows), flush=True)


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
