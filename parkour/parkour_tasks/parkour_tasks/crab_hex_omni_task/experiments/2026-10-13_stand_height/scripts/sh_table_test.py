#!/usr/bin/env python3
"""Stand-height campaign -- the TABLE / GROUND STAND-UP TEST (K3, PLAN.md H5 / H7; step 2 build, step-4 calibration,
step-6+ evaluation). Writes one JSON per K3 cell (``<prefix>__<cell>.json``), a job summary, an npz of the traces and
(``--video``) an mp4; ``--merge`` pools the cells, reads K3 against R_h and the step-4 R_h floor.

It extends the closed predecessor's ``sn_table_test.py`` (``../2026-10-11_stand_network/scripts``; loaded BY PATH for
its helpers -- run_meta resolution, the harness extraction, the slot rule, captions -- never edited) from one 25-in
table to the K3 grid of ``sh_table_core`` (read its docstring: cells, poses, commands, start, episode, controllers,
STOOD, the K3 read), with the stand-height semantics of ``sh_core`` (q(h), R_h, the limiter, h_meas, the stand-net
input, the router).

LAUNCH THROUGH ``sh_table_test.sh`` (one Isaac process at a time, MemoryMax, detached). Kit-free modes (CPU)::

    sh_table_test.py --emit-facts --out DIR [--cells all]   # DIR/facts.json (cells, poses, commands, episode, STOOD)
    sh_table_test.py --merge DIR                            # DIR/table_test_summary.json from DIR/*__<cell>.json

The Isaac run (``--headless``; ``--enable_cameras`` with ``--video``)::

    sh_table_test.py --controller {r_h,stand_net,posture_only,s1,amask_old} [--cells all|<surface>|<cell>,...]
                     [--envs-per-cell 100] [--seed 1] [--jitter] [--h-cmd H] [--stand-ckpt P] [--stand-mask 1]
                     [--deadband D] [--label L] [--video] [--smoke [--episode-s S]] --out DIR [--prefix P] --headless ...

``--deadband D`` (``stand_net`` only; default 0 = off, the action path without it, bitwise): the router's residual
deadband (``sh_core.apply_deadband``; CHARTER 2026-10-14 probe), recorded in the job summary and in every cell's meta
(``residual_deadband``); the default label of a deadbanded run carries it (``stand_net_<sha8>_db0p25``), and the merge
refuses a label whose cells mix deadbands.

THE SETUP (what differs from the predecessor test; everything else is its setup, recorded in the summary):

* ENV: the S1 eval env (``Isaac-Crab-Hex-Omni-Walk-Play-v0``) with every ``KRABBY_*`` replaced before Kit by the REFS
  suite es1 stamp's env (D5 checked), the harness's ``_apply_env_overrides`` (seed, episode = the run's + 2 s,
  resampling / curriculum / pushes off, spawn yaw 0), ``sn_common``-style D5 refusals.
* ENVS: ``--envs-per-cell`` (>= 100 outside ``--smoke``) per cell, all cells batched in one process (cell-major env
  order); per env the start / pose / table offset / command of ``sh_table_core.plan_envs`` (depends only on the env
  seed, the cell and the env's index in it).
* EPISODE: exactly ``sh_table_core.k3_episode_s()`` (16.0 s; ``--episode-s`` only with ``--smoke``).
* START: ``sh_table_core.sh_start`` (reset event, LAST): the floor reference = the highest terrain point under the
  robot (raycast); table envs on their own tall kinematic cuboid (top by z placement, above that floor), ground envs
  with the table parked 5 m below the floor. After ``gym.make`` the terrain mesh (the height scanner's
  ``RayCaster.meshes`` entry, the eval harness's lookup) is registered as the spawn probe BEFORE the wrapper's
  construction reset; the SAME mesh gives h_meas every step (``sh_core.terrain_z_under_feet``).
* WRAPPER CLIP: ``env.clip_actions = sh_core.HARNESS_CLIP`` (4.8) at runtime after the wrapper is built (no config
  edit; the agent cfg's 1.0 is recorded beside it); the env action term's clip must be rods +-4.8 / cams +-1 (fail
  closed); S1's rows are clamp(S1, +-1) by the router.
* HEIGHT: one ``sh_core.HeightController`` (dt = the env step), reset at spawn per env with its start kind (table /
  ground) and the MEASURED h_meas (the limiter then starts at clamp(h_meas, h_min, h_max)), stepped every step on the
  latched stand slot with the env's command BEFORE the actions. The slot stays STAND (stand command from t = 0), so no
  restore engages (its counts are recorded and must stay 0).
* TERMINATIONS: a time-out only (2 s past the episode), falls RECORDED (the predecessor's rule); the env's
  ``crab_failure`` parameters (tilt limit, hip-contact threshold, hip bodies) are read before it is disabled and
  applied by the classifier, the hip-contact clause armed per PLAN H5 (``sh_table_core.hip_contact_armed``).
* RECORDED DOSES (summary ``doses``): the realised wrapper clip, max |hip action| on stand rows and the share beyond
  +-1, max |S1 row action|, R_h's residual-label excess, the restore counts, the limiter arrivals.
"""
from __future__ import annotations

import argparse
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
PKG = HERE.parents[2]                                     # crab_hex_omni_task
REPO = HERE.parents[6]
PARKOUR = REPO / "parkour"
PRED_SCRIPTS = PKG / "experiments" / "2026-10-11_stand_network" / "scripts"
DEFAULT_OUT_BASE = PARKOUR / "logs" / "videos" / "sh_table"
REFUSED_RC = 2
_NAME_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
JOB_KIND = "sh_table_job"


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


#: torch-free at import (rule L6): loaded before Kit
CORE = _load(HERE / "sh_table_core.py", "_sht_core")
SH = CORE.SH
SN = SH.SN
#: the predecessor table test (its helpers; its own module-level loads are torch-free too)
SNT = _load(PRED_SCRIPTS / "sn_table_test.py", "_sht_pred_table_test")
SPEC = SNT.SPEC
P2L = SNT.P2L
TASK_ID = SNT.TASK_ID
HARNESS = SNT.HARNESS
DEFAULT_S1 = SNT.DEFAULT_S1


# ===================================================================================== argv
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(allow_abbrev=False, add_help=False)
    ap.add_argument("--controller", choices=CORE.CONTROLLERS, default=None)
    ap.add_argument("--cells", default="all")
    ap.add_argument("--envs-per-cell", dest="envs_per_cell", type=int, default=CORE.ENVS_PER_CELL)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--jitter", action="store_true", default=False)
    ap.add_argument("--episode-s", dest="episode_s", type=float, default=None)
    ap.add_argument("--h-cmd", dest="h_cmd", type=float, default=None)
    ap.add_argument("--label", default=None)
    ap.add_argument("--video", action="store_true", default=False)
    ap.add_argument("--out", default=None)
    ap.add_argument("--prefix", default=None)
    ap.add_argument("--overwrite", action="store_true", default=False)
    ap.add_argument("--smoke", action="store_true", default=False)
    ap.add_argument("--eval-run", dest="eval_run", default=str(SPEC.REFS_ENV_STAMP))
    ap.add_argument("--s1-ckpt", dest="s1_ckpt", default=str(DEFAULT_S1))
    ap.add_argument("--stand-ckpt", dest="stand_ckpt", default=str(SH.STAND_NET_CKPT))
    ap.add_argument("--stand-mask", dest="stand_mask", type=int, choices=(0, 1), default=1)
    ap.add_argument("--deadband", type=float, default=0.0)
    ap.add_argument("--friction", type=float, default=None)
    ap.add_argument("--cam-eye", dest="cam_eye", default="-4.5,-4.5,1.9")
    ap.add_argument("--cam-lookat", dest="cam_lookat", default="0.0,0.0,0.6")
    ap.add_argument("--quality", type=float, default=8.0)
    ap.add_argument("--rss-abort-gb", dest="rss_abort_gb", type=float, default=20.0)
    ap.add_argument("--emit-facts", dest="emit_facts", action="store_true", default=False)
    ap.add_argument("--merge", default=None)
    return ap


_OWN = ("--controller", "--cells", "--envs-per-cell", "--seed", "--jitter", "--episode-s", "--h-cmd", "--label",
        "--video", "--out", "--prefix", "--overwrite", "--smoke", "--eval-run", "--s1-ckpt", "--stand-ckpt",
        "--stand-mask", "--deadband", "--friction", "--cam-", "--quality", "--rss-", "--emit-facts", "--merge")


def deadband_tag(d) -> str:
    """``_db0p25`` for a residual deadband > 0 (label-safe), ``""`` for 0 or a malformed value (refused pre-Kit)."""
    try:
        d = SH.check_deadband(d)
    except ValueError:
        return ""
    return f"_db{d:g}".replace(".", "p") if d else ""


def default_label(o: argparse.Namespace) -> str | None:
    """The merge key: the controller, a stand checkpoint's sha prefix for ``stand_net`` (two checkpoints never
    collide in one stamp dir) and its residual deadband when > 0 (``_db0p25``: a deadbanded run never shares a label
    with the plain one)."""
    if o.controller is None:
        return None
    if o.controller == "stand_net":
        p = Path(o.stand_ckpt)
        return "stand_net_" + (SN.sha256_file(p)[:8] if p.is_file() else "missing") + deadband_tag(
            getattr(o, "deadband", 0.0))
    return o.controller


def parse(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    o, rest = build_parser().parse_known_args(list(argv))
    stray = [a for a in rest if a.startswith(_OWN)]
    if stray:
        raise SystemExit(f"sh_table_test: malformed argument(s) {stray}")
    o.label = o.label or default_label(o)
    if o.prefix is None and o.label is not None:
        o.prefix = f"{o.label}_s{o.seed}" + ("_video_" + o.cells.replace(",", "+") if o.video else "")
    return o, rest


def episode_s(o: argparse.Namespace) -> float:
    return float(o.episode_s) if o.episode_s is not None else CORE.k3_episode_s()


def pre_kit_reasons(o: argparse.Namespace, rest: list[str]) -> list[str]:
    """Every refusal before Kit (fail closed)."""
    out = []
    if o.controller is None:
        out.append("--controller is required for an Isaac run")
    if o.out is None:
        out.append("--out is required")
    else:
        out += [f"--out: {r}" for r in SPEC.path_reasons(o.out)]
    cells = []
    try:
        cells = CORE.parse_cells(o.cells)
    except ValueError as exc:
        out.append(str(exc))
    if o.envs_per_cell < 1:
        out.append("--envs-per-cell must be >= 1")
    if o.video:
        if len(cells) != 1 or o.envs_per_cell != 1:
            out.append("--video runs exactly one cell with --envs-per-cell 1")
        if "--enable_cameras" not in rest:
            out.append("--video needs --enable_cameras (AppLauncher)")
    elif not o.smoke and o.envs_per_cell < CORE.ENVS_PER_CELL:
        out.append(f"--envs-per-cell {o.envs_per_cell} < {CORE.ENVS_PER_CELL} (PLAN K3: n >= 100 per cell; "
                   "allowed with --smoke only)")
    if o.episode_s is not None:
        if not o.smoke:
            out.append(f"--episode-s is the pinned K3 length ({CORE.k3_episode_s()} s) outside --smoke")
        elif not o.episode_s > 0:
            out.append("--episode-s must be > 0")
    for name, val in (("--prefix", o.prefix), ("--label", o.label)):
        if val is not None and not _NAME_RE.match(val):
            out.append(f"{name} {val!r} must be [A-Za-z0-9_.-]+")
    if o.out is not None and o.prefix is not None and not o.overwrite:
        for c in cells:
            p = Path(o.out) / f"{o.prefix}__{c}.json"
            if p.exists():
                out.append(f"{p} exists (pass --overwrite or a new --prefix)")
    if o.friction is not None and not o.friction > 0:
        out.append("--friction must be > 0")
    try:
        db = SH.check_deadband(o.deadband)
    except ValueError as exc:
        out.append(f"--deadband: {exc}")
        db = 0.0
    if db and o.controller != CORE.DEADBAND_CONTROLLER:
        out.append(f"--deadband {db} runs on --controller {CORE.DEADBAND_CONTROLLER} only (a stand network's residual), "
                   f"not {o.controller}")
    if cells and o.envs_per_cell >= 1:
        try:
            plan = CORE.plan_envs(cells, o.envs_per_cell, o.seed, jitter=o.jitter, h_cmd=o.h_cmd)
            need = CORE.plan_episode_s(plan)
            if not o.smoke and episode_s(o) + 1e-9 < need:          # pragma: no cover - k3 covers every cell
                out.append(f"episode {episode_s(o)} s < the plan's need {need} s")
        except (ValueError, RuntimeError) as exc:
            out.append(f"plan: {exc}")
    s1 = Path(o.s1_ckpt)
    if not s1.is_file():
        out.append(f"--s1-ckpt missing: {s1}")
    elif SN.sha256_file(s1) != SPEC.S1_SHA256:
        out.append(f"--s1-ckpt sha256 is not S1 s18's pinned {SPEC.S1_SHA256[:12]}")
    if o.controller in CORE.STAND_CONTROLLERS:
        st = Path(o.stand_ckpt)
        if not st.is_file():
            out.append(f"--stand-ckpt missing: {st}")
        elif o.controller == "amask_old" and SN.sha256_file(st) != SH.STAND_NET_SHA256:
            out.append(f"amask_old runs A-mask s17 only ({SH.STAND_NET_SHA256[:12]})")
        else:
            out += [f"--stand-ckpt: {r}" for r in SPEC.path_reasons(st)]
    try:
        rm = json.loads(SNT.resolve_run_meta(o.eval_run).read_text())
        out += [f"eval env (D5): {r}" for r in SN.env_reasons(SNT.want_env(rm))]
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
def emit_facts(out_dir: Path, cells=None) -> dict:
    facts = CORE.build_facts(cells)
    facts["_sources"] = {"sh_table_core_sha256": SN.sha256_file(HERE / "sh_table_core.py"),
                         "sh_table_test_sha256": SN.sha256_file(_THIS),
                         "sn_table_core_sha256": SN.sha256_file(CORE.STC_PATH),
                         "sn_table_test_sha256": SN.sha256_file(PRED_SCRIPTS / "sn_table_test.py"),
                         "restance_module_sha256": SN.sha256_file(SN.RESTANCE_MODULE), "r_knob": SN.RESTANCE_D3}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "facts.json").write_text(json.dumps(facts, indent=1, sort_keys=True, default=str))
    return facts


def print_facts(facts: dict) -> None:
    print(f"[SHT] K3 cells {len(facts['k3_cells'])} x {facts['envs_per_cell']} envs; episode {facts['k3_episode_s']} s; "
          f"h range [{facts['h_min']}, {facts['h_max']}]; gated tops {facts['k3_tops_m']} (gated cells "
          f"{len(facts['k3_gated_cells'])}), reported tops {facts['k3_reported_tops_m']} + ground (reported cells "
          f"{len(facts['k3_reported_cells'])})", flush=True)
    for p, v in facts["poses"].items():
        if p == "random":
            print(f"[SHT] random: hips {v['hip_grid_deg'][0]:.1f}..{v['hip_grid_deg'][-1]:.1f} deg "
                  f"({len(v['hip_grid_deg'])} nodes)", flush=True)
        else:
            print(f"[SHT] {p}: hip {v['hip_deg']:+.2f} knee L {v['knee_left_deg']:+.2f} deg; ground spawn root "
                  f"{v['ground_spawn_root_rel_m']:.4f} m ({v['first_contact_on_ground']}); table top min "
                  f"{v['table_top_min_m']:.4f} m", flush=True)


def merge(dir_: Path) -> dict:
    paths = sorted(p for p in dir_.glob("*.json") if p.name not in ("table_test_summary.json", "facts.json")
                   and not p.name.endswith("_summary.json"))
    merged = CORE.merge_cells(paths)
    (dir_ / "table_test_summary.json").write_text(json.dumps(merged, indent=1, sort_keys=True, default=str))
    print(CORE.format_table(merged), flush=True)
    for v in merged["videos"]:
        e = v.get("env0") or {}
        print(f"[SHT] video {v['label']} {v['cell']} es{v['seed']}: {v['mp4']} -- stood={e.get('stood')} "
              f"lift_off={e.get('lift_off_s')} h_end={e.get('h_meas_end_m')} h_cmd={e.get('h_cmd_m')}", flush=True)
    return merged


# ===================================================================================== the env composition
CRAB_FAILURE_KEYS = ("limit_angle", "contact_force_threshold")


def crab_failure_params(term_cfg) -> dict:
    """The env's crab_failure clause (tilt limit, hip-contact threshold, hip bodies), read before it is disabled; the
    classifier applies it with the H5 arming. Refuses a cfg without it."""
    t = getattr(term_cfg, "crab_failure", None)
    params = getattr(t, "params", None) or {}
    missing = [k for k in CRAB_FAILURE_KEYS if params.get(k) is None]
    if t is None or missing:
        raise SystemExit(f"terminations.crab_failure missing or without {missing}: the hip-contact / tilt clauses "
                         "cannot be read")
    sc = params.get("hip_contact_sensor_cfg")
    names = list(getattr(sc, "body_names", None) or [".*_Hip"])
    sensor = getattr(sc, "name", "contact_forces")
    return {"limit_angle": float(params["limit_angle"]), "contact_force_threshold": float(
        params["contact_force_threshold"]), "hip_body_names": names, "sensor": sensor}


def _compose_env_cfg(env_cfg, o, *, event_params, friction, sim_utils, RigidObjectCfg, EventTermCfg,
                     TerminationTermCfg, il_time_out, ep_s) -> tuple[list, dict]:
    """The test's env changes (each recorded): terminations -> a time-out only (crab_failure read first); events as the
    predecessor (DR / pushes / RSI off, one friction, ``reset_robot_joints`` off, the ring spawn kept) + ``sh_start``
    LAST; scene: the tall kinematic table; viewer (``--video``)."""
    changes = []

    def note(what, old, new, why):
        changes.append({"what": what, "old": old, "new": new, "why": why})

    fail = crab_failure_params(env_cfg.terminations)
    term = env_cfg.terminations
    for name, t in list(vars(term).items()):
        if t is not None:
            note(f"terminations.{name}", SNT._dump_terms(types.SimpleNamespace(**{name: t}))[name], None,
                 "disabled: falls are RECORDED by the K3 classifier (crab_failure's tilt / hip-contact clauses applied "
                 "there, the hip contact armed per PLAN H5)")
            setattr(term, name, None)
    term.total_terminates = TerminationTermCfg(func=il_time_out, time_out=True)
    note("terminations.total_terminates", None, "isaaclab.envs.mdp.time_out (time_out=True)",
         f"keep a time-out: episode_length_s = {ep_s} + 2 s, never reached inside the {ep_s}-s loop")
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
            note("events.reset_root_state", "ring spawn (tile centre, yaw pinned 0)", "kept",
                 "parkour spawn bookkeeping; sh_start (last) overrides z, joints and the table")
        elif name == "reset_robot_joints":
            note(f"events.{name}", f"reset_joints_by_scale {params.get('position_range')}", None,
                 "sh_start writes the retracted joints")
            setattr(ev, name, None)
        elif name == "physics_material":
            fr = tuple(float(x) for x in params.get("friction_range", (1.0, 1.0)))
            f = float(friction) if friction is not None else 0.5 * (fr[0] + fr[1])
            params["friction_range"] = (f, f)
            note("events.physics_material.friction_range", list(fr), [f, f],
                 "one pinned friction (the range's midpoint unless --friction)")
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
                 "randomization / push / RSI off (pushes suppressed: the robot is never 'up' before the window)")
            setattr(ev, name, None)
        else:
            raise SystemExit(f"events.{name} (mode {mode}) is not a term this test knows how to treat; refused")
    ev.sh_start = EventTermCfg(func=CORE.sh_start, mode="reset", params=event_params)
    note("events.sh_start", None, "sh_table_core.sh_start (reset, LAST)",
         "per env: pose; floor reference = the highest terrain point under the robot; root z = floor + (table: top + "
         "0.16375; ground: the pose's depth + 0.005); table top = floor + top by z placement (ground: parked)")
    tmat = copy.deepcopy(env_cfg.scene.terrain.physics_material)
    env_cfg.scene.sh_table = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/ShTable",
        spawn=sim_utils.CuboidCfg(
            size=(CORE.TABLE_X_M, CORE.TABLE_Y_M, CORE.TABLE_HEIGHT_M),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=CORE.TABLE_MASS_KG),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            physics_material=tmat,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.55, 0.36, 0.20)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, CORE.PARK_CENTER_Z_M)),
    )
    note("scene.sh_table", None, f"kinematic cuboid {CORE.TABLE_X_M} x {CORE.TABLE_Y_M} x {CORE.TABLE_HEIGHT_M} m, "
         f"terrain material {type(tmat).__name__}", "the table (top by z placement per env)")
    env_cfg.commands.base_velocity.debug_vis = False
    if o.video:
        env_cfg.viewer.origin_type = "env"
        env_cfg.viewer.env_index = 0
        env_cfg.viewer.eye = SNT._vec3(o.cam_eye)
        env_cfg.viewer.lookat = SNT._vec3(o.cam_lookat)
        note("viewer", "asset_root follow-cam", f"static env-0 camera eye {o.cam_eye} lookat {o.cam_lookat}", "video")
    obs_pol = getattr(getattr(env_cfg, "observations", None), "policy", None)
    note("observations.policy.enable_corruption", getattr(obs_pol, "enable_corruption", None), "unchanged",
         "recorded only (the eval env's value)")
    return changes, fail


# ===================================================================================== the Kit run
def run_kit(o: argparse.Namespace, rest: list[str]) -> int:
    from isaaclab.app import AppLauncher  # noqa: PLC0415

    ap = argparse.ArgumentParser(allow_abbrev=False)
    AppLauncher.add_app_launcher_args(ap)
    app_args = ap.parse_args(rest)
    rm_path = SNT.resolve_run_meta(o.eval_run)
    want = SNT.want_env(json.loads(rm_path.read_text()))
    dropped = sorted(k for k in os.environ if k.startswith("KRABBY_") and os.environ[k] != want.get(k))
    for k in [k for k in os.environ if k.startswith("KRABBY_")]:
        del os.environ[k]
    os.environ.update(want)
    print(f"[SHT] KRABBY_* replaced by {rm_path} env_vars ({len(want)}); dropped/differing: {dropped or 'none'}",
          flush=True)
    t0 = time.time()
    simulation_app = AppLauncher(app_args).app
    print(f"[SHT] Kit up in {time.time() - t0:.1f}s, RSS {SNT._rss_gb():.2f} GB", flush=True)
    signal.signal(signal.SIGTERM, SNT._on_sigterm)
    try:
        if SNT._rss_gb() > o.rss_abort_gb:
            print(f"[SHT][ABORT] RSS {SNT._rss_gb():.1f} GB right after Kit start (DerivedDataCache?)", flush=True)
            return 3
        return _run(o, app_args, rm_path, want, dropped)
    finally:
        simulation_app.close()


CONTROLLER_LABEL = {"r_h": "R_h (height-aware R, D3)", "stand_net": "STAND NET (q + residual)",
                    "posture_only": "POSTURE ONLY (q, zero residual)", "s1": "S1 s18 (walker) alone",
                    "amask_old": "A-mask s17 on the old +-1 path"}


def _run(o, app_args, rm_path: Path, want: dict, dropped: list) -> int:  # noqa: C901 - one linear Kit script
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
    if str(HARNESS.parent) not in sys.path:
        sys.path.insert(0, str(HARNESS.parent))
    import isaaclab.sim as sim_utils  # noqa: PLC0415
    import parkour_tasks  # noqa: F401,PLC0415
    from gait_eval import schedule as S  # noqa: PLC0415
    from isaaclab.assets import RigidObjectCfg  # noqa: PLC0415
    from isaaclab.envs.mdp import time_out as il_time_out  # noqa: PLC0415
    from isaaclab.managers import EventTermCfg, TerminationTermCfg  # noqa: PLC0415
    from isaaclab.sensors.ray_caster import RayCaster  # noqa: PLC0415
    from isaaclab.utils.warp import raycast_mesh  # noqa: PLC0415
    from isaaclab_tasks.utils import parse_env_cfg  # noqa: PLC0415
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: PLC0415
    from parkour_isaaclab.envs.mdp.parkour_commands.uniform_parkour_command import (  # noqa: PLC0415
        UniformParkourCommand,
    )
    from scripts.rsl_rl.modules.crab_actor_critic_with_encoder import CrabHexActorCriticRMA  # noqa: PLC0415
    from scripts.rsl_rl.modules.feature_extractors.estimator import DefaultEstimator  # noqa: PLC0415
    from scripts.rsl_rl.runner_factory import agent_cfg_to_train_dict  # noqa: PLC0415
    from scripts.rsl_rl.vecenv_wrapper import ParkourRslRlVecEnvWrapper  # noqa: PLC0415

    dcore = _load(PRED_SCRIPTS / "sn_dagger_core.py", "_sht_dagger_core")     # torch at its top: post-Kit only
    H = SNT._load_harness_functions(S, torch, UniformParkourCommand)
    out = Path(o.out)
    out.mkdir(parents=True, exist_ok=True)
    rm = json.loads(rm_path.read_text())
    krabby_now = {k: v for k, v in sorted(os.environ.items()) if k.startswith("KRABBY_")}
    if krabby_now != want:
        raise SystemExit(f"KRABBY_* after the task import differs from the stamp: "
                         f"{sorted(set(krabby_now.items()) ^ set(want.items()))}")
    cells = CORE.parse_cells(o.cells)
    plan = CORE.plan_envs(cells, o.envs_per_cell, o.seed, jitter=o.jitter, h_cmd=o.h_cmd)
    ep_s = episode_s(o)
    joint_names = list(rm["joint_names"])
    ev_params = CORE.event_params(plan, joint_names)
    joint_rows = ev_params["joint_pos"]
    facts = CORE.build_facts(cells)
    N = len(plan)

    # ---- env cfg
    env_cfg = parse_env_cfg(TASK_ID, device=app_args.device, num_envs=N)
    clip_bad = CORE.action_clip_reasons(getattr(env_cfg.actions.joint_pos, "clip", None),
                                        getattr(env_cfg.actions.joint_pos, "scale", None))
    if clip_bad:
        raise SystemExit("the env action term is not the +-4.8 rod / +-1 cam preset the posture offset needs:\n  "
                         + "\n  ".join(clip_bad))
    events_before, terms_before = SNT._dump_terms(env_cfg.events), SNT._dump_terms(env_cfg.terminations)
    scen = types.SimpleNamespace(id="sh_table", env_seed=int(o.seed), episode_length_s=float(ep_s) + 2.0,
                                 total_duration_s=float(ep_s), freeze_terrain_levels=True,
                                 terrain_generator_seed=None, freeze_friction=False, command_frame="world",
                                 spawn_yaw=0.0, spawn_yaw_rad=lambda: 0.0)
    applied = H["_apply_env_overrides"](env_cfg, scen)
    changes, fail = _compose_env_cfg(env_cfg, o, event_params=ev_params, friction=o.friction, sim_utils=sim_utils,
                                     RigidObjectCfg=RigidObjectCfg, EventTermCfg=EventTermCfg,
                                     TerminationTermCfg=TerminationTermCfg, il_time_out=il_time_out, ep_s=ep_s)
    events_after, terms_after = SNT._dump_terms(env_cfg.events), SNT._dump_terms(env_cfg.terminations)
    for c in changes:
        print(f"[SHT] change {c['what']}: {c['old']} -> {c['new']} ({c['why']})", flush=True)
    print(f"[SHT] crab_failure as recorded clauses: {fail}", flush=True)

    agent_cfg = load_cfg_from_registry(TASK_ID, "rsl_rl_cfg_entry_point")
    train_cfg = agent_cfg_to_train_dict(agent_cfg)
    bad = SPEC.network_cfg_reasons(train_cfg, yaml.safe_load(SPEC.S1_AGENT_YAML.read_text()))
    if bad:
        raise SystemExit("network cfg differs from S1's agent.yaml:\n  " + "\n  ".join(bad))
    raw_env = gym.make(TASK_ID, cfg=env_cfg, render_mode="rgb_array" if o.video else None)
    # the terrain mesh (the harness's lookup), registered BEFORE the wrapper's construction reset spawns the envs
    mesh_key = None
    try:
        scanner = raw_env.unwrapped.scene.sensors.get("height_scanner")
        mesh_key = scanner.cfg.mesh_prim_paths[0] if scanner is not None else "/World/ground"
        mesh = RayCaster.meshes[mesh_key]
    except (KeyError, AttributeError) as exc:
        raw_env.close()
        raise SystemExit(f"terrain mesh unavailable ({exc}): h_meas and the ground spawns need it") from exc
    CORE.RUNTIME["terrain_probe"] = CORE.make_terrain_probe(mesh, raycast_mesh)
    CORE.RUNTIME["probe_label"] = f"raycast_mesh:{mesh_key}"
    env = ParkourRslRlVecEnvWrapper(raw_env, clip_actions=agent_cfg.clip_actions)
    cam = None
    try:
        unwrapped = env.unwrapped
        device = env.device
        dt = float(unwrapped.step_dt)
        T = int(round(ep_s / dt))
        agent_clip = float(env.clip_actions)
        env.clip_actions = float(SH.HARNESS_CLIP)                       # runtime, no config edit (PLAN H1)
        if int(env.num_obs) != 1149 or int(env.num_actions) != 18 or float(env.clip_actions) != SH.HARNESS_CLIP:
            raise SystemExit(f"env dims / clip {env.num_obs} / {env.num_actions} / {env.clip_actions}: want 1149 / 18 / "
                             f"{SH.HARNESS_CLIP}")
        if agent_clip != 1.0:
            raise SystemExit(f"the agent cfg's clip_actions is {agent_clip}, not the deployment 1.0 (S1's clamp)")
        if int(unwrapped.max_episode_length) <= T:
            raise SystemExit(f"max_episode_length {unwrapped.max_episode_length} <= the loop's {T} steps")
        robot = unwrapped.scene["robot"]
        table = unwrapped.scene["sh_table"]
        if list(robot.joint_names) != joint_names:
            raise SystemExit(f"joint order {robot.joint_names} != the stamp's {joint_names}")
        usd = str(getattr(getattr(robot.cfg, "spawn", None), "usd_path", ""))
        if Path(usd).name != Path(rm["usd_path"]).name:
            raise SystemExit(f"spawned {usd}, the stamp spawned {rm['usd_path']}")
        if SNT._rss_gb() > o.rss_abort_gb:
            raise SystemExit(f"RSS {SNT._rss_gb():.1f} GB > {o.rss_abort_gb} before the first step")

        # ---- joint limits / defaults / every env's start pose inside the limits (fail closed)
        lim = CORE.STC.joint_limits_rad()
        jl = getattr(robot.data, "joint_pos_limits", None)
        jl = robot.data.default_joint_pos_limits if jl is None else jl
        jl0 = jl[0].detach().cpu().numpy()
        dj0 = robot.data.default_joint_pos[0].detach().cpu().numpy()
        dflt = CORE.STC.defaults_rad()
        jr = np.asarray(joint_rows, dtype=np.float64)
        mism, lim_rows = [], {}
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
            if float(jr[:, j].min()) < lo or float(jr[:, j].max()) > hi:
                mism.append(f"{n}: start range [{jr[:, j].min():.4f}, {jr[:, j].max():.4f}] outside [{lo:.4f}, "
                            f"{hi:.4f}]")
        if mism:
            raise SystemExit("joint limits / defaults / start poses refused:\n  " + "\n  ".join(mism))

        # ---- the start state after the construction reset (fail closed)
        org = unwrapped.scene.env_origins.detach().cpu().numpy()
        jp_now = robot.data.joint_pos.detach().cpu().numpy()
        rp_now = robot.data.root_pos_w.detach().cpu().numpy()
        rq_now = robot.data.root_quat_w.detach().cpu().numpy()
        tp_now = table.data.root_pos_w.detach().cpu().numpy()
        last, floor_t = CORE.RUNTIME.get("last_root_z"), CORE.RUNTIME.get("last_floor_z")
        if last is None or floor_t is None or bool(torch.isnan(last).any()) or bool(torch.isnan(floor_t).any()):
            raise SystemExit("sh_start did not run on every env at the construction reset")
        last_np, floor_np = last.detach().cpu().numpy(), floor_t.detach().double().cpu().numpy()
        is_table = np.asarray([e["start"] == CORE.START_TABLE for e in plan])
        offs = np.asarray([e["table_offset"] for e in plan], dtype=np.float64)
        top_world = np.asarray([floor_np[i] + e["top"] if e["top"] is not None else np.nan for i, e in enumerate(plan)])
        tcz = np.asarray([CORE.table_center_z(e["top"], floor_np[i]) for i, e in enumerate(plan)], dtype=np.float64)
        plan_rel = np.asarray([e["spawn_root_rel"] for e in plan], dtype=np.float64)
        start = {"joint_pos_max_err": float(np.abs(jp_now - jr).max()),
                 "root_z_vs_event_err_max": float(np.abs(rp_now[:, 2] - last_np).max()),
                 "root_z_vs_floor_plus_plan_err_max": float(np.abs(rp_now[:, 2] - floor_np - plan_rel).max()),
                 "floor_reference_m": {k: ([float(floor_np[m].min()), float(np.median(floor_np[m])),
                                            float(floor_np[m].max())] if m.any() else None)
                                       for k, m in (("table", is_table), ("ground", ~is_table))},
                 "root_xy_from_origin_max": float(np.abs(rp_now[:, :2] - org[:, :2]).max()),
                 "root_quat_err_max": float(np.abs(rq_now - np.array([1.0, 0, 0, 0])[None]).max()),
                 "table_xy_err_max": float(np.abs(tp_now[:, :2] - (org[:, :2] + offs[:, :2])).max()),
                 "table_z_err_max": float(np.abs(tp_now[:, 2] - tcz).max()),
                 "probe": CORE.RUNTIME.get("probe_kind")}
        print(f"[SHT] start state after the construction reset: {start}", flush=True)
        if (start["joint_pos_max_err"] > 1e-3 or start["root_z_vs_event_err_max"] > 1e-3
                or start["root_z_vs_floor_plus_plan_err_max"] > 1e-3 or start["table_xy_err_max"] > 1e-3
                or start["table_z_err_max"] > 1e-3 or start["root_quat_err_max"] > 1e-4
                or start["root_xy_from_origin_max"] > 1e-3):
            raise SystemExit(f"the start state is not the planned one: {start}")

        # ---- networks (no PPO runner; global RNG saved / restored)
        before = dcore.save_rng()
        stand_check = None
        try:
            s1, s1_est, s1_ck = dcore.load_strict(o.s1_ckpt, CrabHexActorCriticRMA, DefaultEstimator, train_cfg, device)
            st = st_est = None
            if o.controller in CORE.STAND_CONTROLLERS:
                st, st_est, st_ck = dcore.load_strict(o.stand_ckpt, CrabHexActorCriticRMA, DefaultEstimator, train_cfg,
                                                      device)
                ok, why = SN.state_dicts_equal(st_ck["estimator_state_dict"], s1_ck["estimator_state_dict"])
                if not ok:
                    raise SystemExit(f"the stand checkpoint's estimator differs from S1's ({why}; D1)")
                st_sha = SN.sha256_file(o.stand_ckpt)
                try:
                    rule = CORE.stand_ckpt_rule(o.controller, st_sha, st_ck.get("infos"), mask=int(o.stand_mask),
                                                s1_sha256=SPEC.S1_SHA256)
                except ValueError as exc:
                    raise SystemExit(str(exc)) from exc
                w15_norm = None
                if rule["zero_w15"]:
                    sd, w15_norm = SH.zero_height_column(st_ck["model_state_dict"])
                    st.load_state_dict(sd, strict=True)
                    st.eval()
                stand_check = {**rule, "sha256": st_sha, "estimator_equals_s1": True, "w15_norm_before": w15_norm,
                               "w15_zero_after": None}
                w = dict(st.named_parameters()).get(SH.WARM_ZERO_KEY)
                if w is not None:
                    stand_check["w15_zero_after"] = bool(torch.all(w[:, SH.HEIGHT_COL] == 0).item())
                if rule["zero_w15"] and not stand_check["w15_zero_after"]:
                    raise SystemExit("W[:, 15] is not zero after the warm-start zeroing")
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
        slot_rule = SNT._SlotRule(cmd_term, unwrapped)
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
        hip_sid, hip_names = contact.find_bodies(fail["hip_body_names"])
        if len(hip_sid) != 6:
            raise SystemExit(f"crab_failure hip bodies {fail['hip_body_names']} -> {hip_names} (want 6)")
        cam_cols = torch.tensor([joint_names.index(f"{leg}_Body_CamShaft_RevoluteJoint") for leg in CORE.LEGS],
                                device=device)
        cam_def = robot.data.default_joint_pos[:, cam_cols]
        off_t = torch.as_tensor(offs, device=device, dtype=torch.float32)
        tcen = unwrapped.scene.env_origins[:, :2] + off_t[:, :2]
        tcz_t = torch.as_tensor(tcz, device=device, dtype=torch.float32)

        def terrain_z():
            return SH.terrain_z_under_feet(robot.data.body_pos_w[:, foot_bid], mesh, raycast_mesh)

        def hip_force():
            h = contact.data.net_forces_w_history[:, :, hip_sid].norm(dim=-1)        # (N, hist, 6)
            return h.max(dim=1).values.max(dim=-1).values

        # ---- the height state: limiter at clamp(measured h_meas), one controller step per env step
        h_cmd_t = torch.as_tensor([e["h_cmd"] for e in plan], dtype=torch.float64, device=device)
        codes_t = torch.as_tensor([CORE.START_CODE[e["start"]] for e in plan], dtype=torch.int64, device=device)
        rp0, rq0 = robot.data.root_pos_w, robot.data.root_quat_w
        h0 = SH.h_meas(rp0[:, 2], terrain_z())
        low0, _cz0 = CORE.STC.lowest_z_and_cos_tilt(rp0[:, 2], rq0[:, 0], rq0[:, 1], rq0[:, 2], rq0[:, 3])
        spawn_low = low0.detach().double().cpu().numpy()
        spawn_root = rp0[:, 2].detach().double().cpu().numpy()          # the K3 rise clause's reference (root)
        hc = SH.HeightController(N, device, dt=dt)
        hc.reset(None, codes_t, h_cmd=h_cmd_t, h_meas=h0)
        h_init = hc.h_lim.clone()
        h0_np, h_init_np = h0.detach().cpu().numpy(), h_init.detach().cpu().numpy()
        print(f"[SHT] spawn h_meas {h0_np.min():.3f}..{h0_np.max():.3f} m -> limiter init {h_init_np.min():.3f}.."
              f"{h_init_np.max():.3f} m; commands {h_cmd_t.min().item():.3f}..{h_cmd_t.max().item():.3f} m; "
              f"{N} envs, {T} steps ({ep_s} s), controller {o.controller} ({o.label}), residual deadband "
              f"{float(o.deadband):g}", flush=True)

        # ---- buffers (device), entry t = the state after control step t
        f32 = dict(device=device)
        B = {"root_pos": torch.zeros(T, N, 3, **f32), "root_quat": torch.zeros(T, N, 4, **f32),
             "lowest_z": torch.zeros(T, N, **f32), "cos_tilt": torch.zeros(T, N, **f32),
             "body_force": torch.zeros(T, N, **f32), "foot_z": torch.zeros(T, N, 6, **f32),
             "foot_force": torch.zeros(T, N, 6, **f32), "hip_force": torch.zeros(T, N, **f32),
             "h_meas": torch.zeros(T, N, dtype=torch.float64, device=device),
             "h_lim": torch.zeros(T, N, dtype=torch.float64, device=device),
             "joint_pos": torch.zeros(T, N, 24, **f32), "cam_abs": torch.zeros(T, N, 6, **f32),
             "actions": torch.zeros(T, N, 18, **f32), "stand": torch.zeros(T, N, dtype=torch.bool, device=device),
             "stand_eff": torch.zeros(T, N, dtype=torch.bool, device=device),
             "restoring": torch.zeros(T, N, dtype=torch.bool, device=device),
             "cmd_wz": torch.zeros(T, N, **f32), "done": torch.zeros(T, N, dtype=torch.bool, device=device),
             "table_xy": torch.zeros(T, N, 2, **f32)}
        if o.controller in CORE.STAND_CONTROLLERS:
            B["a_net"] = torch.zeros(T, N, 18, **f32)
        if o.controller == "r_h":
            B["r_branch"] = torch.zeros(T, N, 6, dtype=torch.int8, device=device)
        label_excess = torch.full((), -float("inf"), dtype=torch.float32, device=device)

        writer = None
        mp4 = out / f"{o.prefix}.mp4"
        if o.video:
            import imageio  # noqa: PLC0415

            cam = getattr(unwrapped, "viewport_camera_controller", None)
            for i in range(60):
                f0 = unwrapped.render()
                if f0 is not None and f0.size and float(f0.mean()) > 1.0:
                    print(f"[SHT] renderer warm after {i + 1} render(s); frame {f0.shape}", flush=True)
                    break
            writer = imageio.get_writer(str(mp4), fps=int(round(1.0 / dt)), codec="libx264", quality=o.quality,
                                        macro_block_size=None, ffmpeg_log_level="error")
        e0 = plan[0]
        W, Y, C, R, G = SNT.WHITE, SNT.YELLOW, SNT.CYAN, SNT.RED, SNT.GREEN

        def caption_lines(t_s, hm0, hl0, rise0, nfeet, tilt_deg, armed, branches=None):
            lines = [(f"STAND-UP TEST  |  {CONTROLLER_LABEL[o.controller]}  |  {e0['cell']}", W),
                     (f"start {e0['start']}" + (f" (table top {e0['top']:.3f} m)" if e0["top"] is not None else "")
                      + f"   pose {e0['pose']}   seed {o.seed}", C),
                     (f"t = {t_s:5.2f} / {ep_s:.1f} s   h_cmd {e0['h_cmd']:.3f}   limited {hl0:.3f}   h_meas {hm0:.3f} m"
                      f"   belly rise {rise0 * 100:+.1f} cm", G if abs(hm0 - e0["h_cmd"]) <= CORE.STOOD_H_TOL_M else Y),
                     (f"feet on floor: {nfeet}/6   tilt {tilt_deg:4.1f} deg   hip-contact clause "
                      f"{'ARMED' if armed else 'not armed (ground start, belly < +10 cm)'}", W)]
            if branches:
                lines.append((f"R_h branches: {branches}", C))
            if tilt_deg > math.degrees(fail["limit_angle"]):
                lines.append(("FALL: tilt > limit", R))
            return lines

        if writer is not None:
            writer.append_data(SNT._caption(unwrapped.render(), caption_lines(0.0, float(h0[0]), float(h_init[0]), 0.0,
                                                                                0, 0.0, e0["start"] == "table")))
        table_check: dict = {}
        trace = hashlib.sha256()
        t_run = time.time()
        branch_names = SN.load_rst().BRANCH_NAMES if o.controller == "r_h" else None
        armed0 = e0["start"] == CORE.START_TABLE
        n_resets = 0
        try:
            with torch.inference_mode():
                for t in range(T):
                    override_state["step"] = t + 1
                    stand = slot_rule.rederive_on_reset()
                    hs = hc.step(stand, h_cmd_t, obs)
                    actions, extra = CORE.controller_actions(
                        o.controller, obs, stand, hs, s1=s1, s1_est=s1_est, p0=p0, p1=p1, num_prop=num_prop,
                        st_pol=st, st_est=st_est, mask=int(o.stand_mask), deadband=float(o.deadband))
                    B["actions"][t] = actions.clamp(-SH.HARNESS_CLIP, SH.HARNESS_CLIP)    # the wrapper's clip
                    B["stand"][t], B["stand_eff"][t], B["restoring"][t] = stand, hs.stand_eff, hs.restoring
                    B["h_lim"][t] = hs.h_lim
                    B["cmd_wz"][t] = cmd_term.vel_command_b[:, 2]
                    if "a_net" in B:
                        B["a_net"][t] = extra["a_net"]
                    if "r_branch" in B:
                        B["r_branch"][t] = extra["r_branch"].to(torch.int8)
                        label_excess = torch.maximum(label_excess, extra["r_label_excess"].to(torch.float32))
                    trace.update(actions.detach().cpu().numpy().tobytes())
                    obs, _, dones, _ = env.step(actions)
                    rp, rq = robot.data.root_pos_w, robot.data.root_quat_w
                    low, cz = CORE.STC.lowest_z_and_cos_tilt(rp[:, 2], rq[:, 0], rq[:, 1], rq[:, 2], rq[:, 3])
                    tz = terrain_z()
                    B["root_pos"][t], B["root_quat"][t] = rp, rq
                    B["lowest_z"][t], B["cos_tilt"][t] = low, cz
                    B["h_meas"][t] = SH.h_meas(rp[:, 2], tz)
                    B["body_force"][t] = contact.data.net_forces_w[:, body_sid[0]].norm(dim=-1)
                    B["foot_z"][t] = robot.data.body_pos_w[:, foot_bid, 2]
                    B["foot_force"][t] = contact.data.net_forces_w[:, foot_sid].norm(dim=-1)
                    B["hip_force"][t] = hip_force()
                    jp = robot.data.joint_pos
                    B["joint_pos"][t] = jp
                    B["cam_abs"][t] = torch.abs(torch.remainder(jp[:, cam_cols] - cam_def + math.pi, 2 * math.pi)
                                                - math.pi)
                    rel = rp[:, :2] - tcen
                    c, s = torch.cos(-off_t[:, 2]), torch.sin(-off_t[:, 2])
                    B["table_xy"][t] = torch.stack([c * rel[:, 0] - s * rel[:, 1], s * rel[:, 0] + c * rel[:, 1]], -1)
                    d = dones.reshape(-1).bool()
                    B["done"][t] = d
                    if bool(d.any()):                    # unexpected (time-out only): keep the height state consistent
                        n_resets += int(d.sum())
                        hc.reset(d, codes_t, h_cmd=h_cmd_t, h_meas=SH.h_meas(robot.data.root_pos_w[:, 2], terrain_z()))
                    trace.update(rp.detach().cpu().numpy().tobytes())
                    if t == 0:
                        tp1 = table.data.root_pos_w
                        e_xy = float((tp1[:, :2] - tcen).abs().max().item())
                        e_z = float((tp1[:, 2] - tcz_t).abs().max().item())
                        table_check["after_step1_xy_err_max"], table_check["after_step1_z_err_max"] = e_xy, e_z
                        if e_xy > 0.01 or e_z > 0.01:
                            raise SystemExit(f"a kinematic table is not at its pose in PhysX after the first step "
                                             f"(xy err {e_xy:.4f}, z err {e_z:.4f} m)")
                    if writer is not None:
                        rise0 = float(low[0]) - float(spawn_low[0])
                        armed0 = armed0 or rise0 >= CORE.ARM_RISE_M
                        nfeet = int(((B["foot_force"][t, 0] >= CORE.FOOT_LOADED_N)
                                     & (B["foot_z"][t, 0] <= CORE.FOOT_FLOOR_Z_M)).sum().item())
                        tilt_deg = math.degrees(math.acos(max(-1.0, min(1.0, float(cz[0])))))
                        br = None
                        if "r_branch" in extra:
                            br = " ".join(f"{leg}:{branch_names[int(b)]}" for leg, b in
                                          zip(CORE.LEGS, extra["r_branch"][0].tolist()))
                        writer.append_data(SNT._caption(unwrapped.render(), caption_lines(
                            (t + 1) * dt, float(B["h_meas"][t, 0]), float(hs.h_lim[0]), rise0, nfeet, tilt_deg,
                            armed0, br)))
                    if (t + 1) % 250 == 0:
                        rss = SNT._rss_gb()
                        print(f"[SHT] step {t + 1}/{T} ({time.time() - t_run:.0f}s) RSS {rss:.2f} GB; env0 h_meas "
                              f"{float(B['h_meas'][t, 0]):.3f} (cmd {plan[0]['h_cmd']:.3f}, limited "
                              f"{float(hs.h_lim[0]):.3f})", flush=True)
                        if rss > o.rss_abort_gb:
                            raise SystemExit(f"RSS {rss:.1f} GB > {o.rss_abort_gb} at step {t + 1}")
        finally:
            if writer is not None:
                writer.close()

        # ---- classify + write
        tp_end = table.data.root_pos_w
        table_check["end_xy_err_max"] = float((tp_end[:, :2] - tcen).abs().max().item())
        table_check["end_z_err_max"] = float((tp_end[:, 2] - tcz_t).abs().max().item())
        A = {k: v.detach().cpu().numpy() for k, v in B.items()}
        feet_on = CORE.feet_on_floor(A["foot_force"], A["foot_z"])                     # (T, N)
        per_env = []
        for i, e in enumerate(plan):
            r = CORE.classify({"lowest_z": A["lowest_z"][:, i], "cos_tilt": A["cos_tilt"][:, i],
                               "feet_on_floor": feet_on[:, i], "h_meas": A["h_meas"][:, i],
                               "h_lim": A["h_lim"][:, i], "hip_force": A["hip_force"][:, i],
                               "cam_abs": A["cam_abs"][:, i], "root_z": A["root_pos"][:, i, 2],
                               "terminated": A["done"][:, i]}, dt, start=e["start"],
                              spawn_lowest_z=float(spawn_low[i]), spawn_root_z=float(spawn_root[i]),
                              h_cmd=e["h_cmd"],
                              table_top=None if e["top"] is None else float(top_world[i]),
                              tilt_max=fail["limit_angle"], contact_n=fail["contact_force_threshold"])
            r.update(env=i, cell=e["cell"], k=e["k"], spawn_lowest_z=float(spawn_low[i]),
                     floor_reference_z=float(floor_np[i]),
                     spawn_root_z=float(spawn_root[i]), spawn_h_meas=float(h0_np[i]), h_init=float(h_init_np[i]),
                     final_table_frame_xy_m=[float(x) for x in A["table_xy"][-1, i]],
                     stand_slot_share=float(A["stand"][:, i].mean()))
            if "r_branch" in A:
                r["r_branch_share"] = {n_: float((A["r_branch"][:, i] == k).mean()) for k, n_ in
                                       enumerate(branch_names)}
            per_env.append(r)
        # rows a stand controller executed (s1: none -- S1 acts on every row; amask_old: the latched slot, no restore)
        se = (np.zeros_like(A["stand_eff"]) if o.controller == "s1" else
              A["stand"] if o.controller == "amask_old" else A["stand_eff"])
        hip_a = np.abs(A["actions"][:, :, 6:12])
        doses = {"env_clip_actions": float(env.clip_actions), "agent_clip_actions": agent_clip,
                 "stand_rows": int(se.sum()), "s1_rows": int((~se).sum()),
                 "stand_rows_rule": "rows the stand controller executed (s1: none; amask_old: the latched slot; else "
                                    "the effective stand vector slot | restoring)",
                 "stand_rows_max_abs_hip_action": float(hip_a[se].max()) if se.any() else None,
                 "stand_rows_share_hip_beyond_1": float((hip_a[se] > 1.0).any(axis=-1).mean()) if se.any() else None,
                 "stand_rows_min_hip_action": float(A["actions"][:, :, 6:12][se].min()) if se.any() else None,
                 "stand_rows_max_abs_knee_action": float(np.abs(A["actions"][:, :, 12:18])[se].max())
                 if se.any() else None,
                 "stand_rows_max_abs_cam_action": float(np.abs(A["actions"][:, :, :6])[se].max()) if se.any() else None,
                 "s1_rows_max_abs_action": float(np.abs(A["actions"])[~se].max()) if (~se).any() else None,
                 "all_rows_max_abs_action": float(np.abs(A["actions"]).max()),
                 "r_h_label_excess_max": float(label_excess.item()) if o.controller == "r_h" else None,
                 "restore_counts": hc.counts(), "height_controller": hc.record(),
                 "limiter_arrived_before_window": sum(1 for r in per_env if r["limiter_arrived_before_window"]),
                 "mid_run_resets": n_resets}
        routing = {"stand_slot_share": float(A["stand"].mean()), "stand_eff_share": float(se.mean()),
                   "non_stand_steps": int((~A["stand"]).sum()), "restoring_steps": int(A["restoring"].sum()),
                   "yaw_cmd_nonzero_steps": int((A["cmd_wz"] != 0).sum()),
                   "resets_rederived": int(slot_rule.resets_rederived),
                   "enter_rule": "latched_stand_slot (P2L rederive_on_reset)", "controller": o.controller}
        files = {"script": _THIS, "core": HERE / "sh_table_core.py", "sh_core": CORE.SH_CORE_PATH,
                 "sh_kin": CORE.SH_KIN_PATH, "sn_table_core": CORE.STC_PATH,
                 "sn_table_test": PRED_SCRIPTS / "sn_table_test.py", "sn_common": SH.SN_COMMON_PATH,
                 "sn_dagger_core": PRED_SCRIPTS / "sn_dagger_core.py", "harness": HARNESS, "p2l": SNT.P2R_LAUNCHER,
                 "restance_module": SN.RESTANCE_MODULE, "facts_json": SH.FACTS_JSON}
        meta = {
            "kind": JOB_KIND, "task": TASK_ID, "controller": o.controller, "label": o.label, "cells": cells,
            "envs_per_cell": o.envs_per_cell, "num_envs": N, "seed": o.seed, "jitter": bool(o.jitter),
            "h_cmd_fixed": o.h_cmd, "episode_s": ep_s, "k3_episode_s": CORE.k3_episode_s(), "dt": dt, "steps": T,
            "smoke": bool(o.smoke), "video": bool(o.video), "mp4": str(mp4) if o.video else None,
            "s1_ckpt": str(o.s1_ckpt), "s1_sha256": SPEC.S1_SHA256,
            "stand_ckpt": str(o.stand_ckpt) if o.controller in CORE.STAND_CONTROLLERS else None,
            "stand_sha256": (stand_check or {}).get("sha256"), "stand_mask": int(o.stand_mask),
            SH.RESIDUAL_DEADBAND_KEY: float(o.deadband), "stand_check": stand_check, "r_knob": SN.RESTANCE_D3,
            "crab_failure": fail,
            "eval_run_meta": str(rm_path), "env_vars": want, "krabby_dropped": dropped,
            "harness_overrides": {k: (v if isinstance(v, (int, float, str, bool, type(None))) else repr(v))
                                  for k, v in applied.items()},
            "max_episode_length": int(unwrapped.max_episode_length),
            "clip": {**SH.clip_record(float(o.deadband)), "realised": float(env.clip_actions), "agent_cfg": agent_clip},
            "observed_command_slots": {str(k): int(v) for k, v in observed_slots.items()},
            "latch_preexisting": bool(latch_pre), "joint_limits_sim": lim_rows, "start_state": start,
            "terrain_mesh": mesh_key, "table_pose_check": table_check, "plan_sha256": CORE.plan_digest(plan),
            "trace_sha256": trace.hexdigest(), "wall_s": time.time() - t_run,
            "peak_rss_gb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0 / 1024.0,
            "shas": {k: SN.sha256_file(p) for k, p in files.items()}, "provenance": SH.provenance(float(o.deadband)),
            "app_args": vars(app_args) if hasattr(app_args, "__dict__") else None,
        }
        cell_meta = {k: meta[k] for k in ("controller", "label", "seed", "jitter", "episode_s", "k3_episode_s", "dt",
                                          "steps", "smoke", "video", "mp4", "stand_sha256", "stand_mask", "r_knob",
                                          "crab_failure", "plan_sha256", "trace_sha256", "envs_per_cell",
                                          SH.RESIDUAL_DEADBAND_KEY)}
        cell_meta["kind"] = CORE.CELL_KIND
        cell_files, aggs = {}, {}
        for cell in cells:
            idx = [i for i, e in enumerate(plan) if e["cell"] == cell]
            rec = CORE.cell_record({**cell_meta, "job_summary": str(out / f"{o.prefix}_summary.json")}, cell,
                                   [plan[i] for i in idx], [per_env[i] for i in idx])
            path = out / f"{o.prefix}__{cell}.json"
            path.write_text(json.dumps(rec, indent=1, sort_keys=True, default=str))
            cell_files[cell], aggs[cell] = str(path), rec["aggregate"]
        summary = {"meta": meta, "facts": facts, "stood_definition": CORE.STOOD_DEFINITION, "env_changes": changes,
                   "events_before": events_before, "events_after": events_after, "terminations_before": terms_before,
                   "terminations_after": terms_after, "routing": routing, "doses": doses, "cells": aggs,
                   "cell_files": cell_files}
        (out / f"{o.prefix}_summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True, default=str))
        npz = {f"all_{k}": A[k] for k in ("lowest_z", "cos_tilt", "h_meas", "h_lim", "hip_force", "body_force")}
        npz.update({"all_root_z": A["root_pos"][..., 2], "all_feet_on_floor": feet_on,
                    "all_cam_abs_max": A["cam_abs"].max(axis=-1), "spawn_lowest_z": spawn_low,
                    "h_cmd": np.asarray([e["h_cmd"] for e in plan]), "h_init": h_init_np, "spawn_h_meas": h0_np,
                    "floor_reference_z": floor_np, "table_top_world_z": top_world,
                    "cells": np.asarray([e["cell"] for e in plan]), "dt": np.asarray(dt),
                    "joint_names": np.asarray(joint_names), "legs": np.asarray(CORE.LEGS)})
        for cell in cells:
            i0 = next(i for i, e in enumerate(plan) if e["cell"] == cell)
            for k, v in A.items():
                npz[f"{cell}__k0_{k}"] = v[:, i0]
        np.savez_compressed(out / f"{o.prefix}_traces.npz", **npz)
        for cell in cells:
            a = aggs[cell]
            print(f"[SHT] {o.label} {cell}: STOOD {a['n_stood']}/{a['n_envs']} (5-down {a['n_stood_5down']}, 6-down "
                  f"{a['n_stood_6down']}); falls {a['falls']} (tilt {a['falls_tilt']}, hip {a['falls_hip_contact']}, "
                  f"off {a['fell_off_table']}); homed {a['homed']}; |dh| window median "
                  f"{a['h_err_window_median_m']['median']}", flush=True)
        print(f"[SHT] doses: {json.dumps(doses, default=str)}", flush=True)
        print(f"[SHT] wrote {len(cells)} cell files + {out / (o.prefix + '_summary.json')}"
              + (f" and {mp4}" if o.video else ""), flush=True)
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
            print_facts(emit_facts(Path(o.out), CORE.parse_cells(o.cells)))
            return 0
        bad = pre_kit_reasons(o, rest)
        if bad:
            raise SystemExit("sh_table_test: refused before Kit:\n  " + "\n  ".join(bad))
    except SystemExit as exc:
        if isinstance(exc.code, str):
            print(exc.code, file=sys.stderr, flush=True)
            return REFUSED_RC
        raise
    return run_kit(o, rest)


if __name__ == "__main__":
    sys.exit(main())
