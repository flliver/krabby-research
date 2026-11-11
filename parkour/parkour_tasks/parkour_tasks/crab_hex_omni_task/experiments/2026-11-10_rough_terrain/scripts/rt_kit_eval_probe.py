"""PLAN step 5 (GPU, Kit): the EVAL-env probe of one kind x band row -- the realised eval tiles and mu_eff the
harness's raw npz does not carry. WRITTEN, NOT YET RUN (the queue runs it right after the row's eval-smoke stamp:
``rt_kit_run.sh <row env file> rt_kit_eval_probe.py ...``; one Isaac process at a time, MemoryMax 48G, SIGTERM-only).

The env is the row's: the launch env (= the row's env file, parent KRABBY_* unset) plus the manifest scenario's
``env:`` block, written to ``os.environ`` BEFORE Kit starts (the harness writes it after Kit starts: the eval
process-state note says that moves a DEPTH STUDENT's outcome class, never the terrain or the materials, which are
what this probe reads). The task is the row's ``--task`` (Isaac-Crab-Hex-Omni-Student-Play-v0, cameras on), the env
count the smoke's ``--episodes`` (4), ``env.cfg.seed`` the row's env seed, and the harness's determinism overrides
(``_apply_env_overrides``: seed, episode length, command resampling off, curriculum None, frozen terrain levels,
push event off, the spawn-yaw pin). It resets, takes ``--steps`` (2) zero-action steps, and writes

* ``<out>.npz``: ``site_raster.dump_site_layers`` (the generator's layers, global int arrays, frame, origins);
* ``<out>`` JSON: per env its tile (level row, type column, sub-terrain name, kind, d, centre level), the root right
  after the reset and after each step, the surface under the root, every shape's mu (min / max, static = dynamic),
  ``env._krabby_rt_mu_eff`` (None when the contract is off), the privileged friction input (obs col 210), the
  material event's params, the module origins and PYTHONPATH, the ``build`` (code_sha / eval_sha / dirty computed
  BEFORE Kit starts with rt_machinery.live_build, i.e. run_candidates' own functions; rt_smoke_check.py eval refuses
  a probe whose build is not results/pin.json's); ``complete`` LAST.

rt_smoke_check.py eval joins it with the harness stamp (spawn xy and pad level must match: the probe's tiles are the
harness's tiles) and runs the dose / slope / mu checks on CPU.

    <isaac python> rt_kit_eval_probe.py --headless --manifest ABS --scenario ID --env-seed N [--episodes 4]
        [--task Isaac-Crab-Hex-Omni-Student-Play-v0] [--steps 2] --out JSON
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from isaaclab.app import AppLauncher

HERE = Path(__file__).resolve().parent
W = Path(__file__).resolve().parents[7]
SCRIPTS_FWD = W / "parkour" / "parkour_tasks" / "parkour_tasks" / "crab_hex_forward_task" / "scripts"
parser = argparse.ArgumentParser(description="Rough-terrain PLAN step 5: the eval-env probe of one row.")
parser.add_argument("--manifest", required=True)
parser.add_argument("--scenario", required=True)
parser.add_argument("--env-seed", type=int, required=True)
parser.add_argument("--episodes", type=int, default=4)
parser.add_argument("--task", default="Isaac-Crab-Hex-Omni-Student-Play-v0")
parser.add_argument("--checkpoint", default=None, help="recorded only (the probe runs zero actions)")
parser.add_argument("--steps", type=int, default=2)
parser.add_argument("--out", required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
LAUNCH_KRABBY = {k: v for k, v in sorted(os.environ.items()) if k.startswith("KRABBY_")}
LAUNCH_PYTHONPATH = os.environ.get("PYTHONPATH", "")          # before Kit (it may extend it)


def _probe_build() -> dict:
    """The build this probe runs at (gate_row.sh's code_state.json fields), read before Kit starts; sys.path and
    sys.modules are restored afterwards so nothing of the campaign scripts / experiments/tools shadows Kit's."""
    import importlib.util  # noqa: PLC0415

    path0, mods0 = list(sys.path), set(sys.modules)
    try:
        spec = importlib.util.spec_from_file_location("rt_probe_machinery", HERE / "rt_machinery.py")
        m = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = m          # dataclasses resolve the module by name (popped again below)
        spec.loader.exec_module(m)
        live = m.live_build()
        return {"code_sha": live["code_sha"], "eval_sha": live["eval_sha"], "dirty": str(bool(live["dirty"])),
                "dirty_paths": live["dirty"][:20], "head": live["head"], "writer": "rt_kit_eval_probe.py"}
    except Exception as exc:  # noqa: BLE001 -- recorded; rt_smoke_check then FAILs the row on the build
        return {"error": f"{type(exc).__name__}: {exc}"}
    finally:
        sys.path[:] = path0
        for k in set(sys.modules) - mods0:
            sys.modules.pop(k, None)


PROBE_BUILD = _probe_build()
sys.path.insert(0, str(SCRIPTS_FWD))
from gait_eval import schedule as GS  # noqa: E402  (pure: the harness's own manifest reader)

_scen = [s for s in GS.load_manifest(Path(args.manifest))[0] if s.id == args.scenario]
if not _scen:
    raise SystemExit(f"scenario {args.scenario!r} not in {args.manifest}")
SCEN = _scen[0]
BLOCK = {str(k): str(v) for k, v in SCEN.env_vars.items()}
for _k, _v in BLOCK.items():
    os.environ[_k] = _v                     # BEFORE Kit (see the docstring)
if "Student" in args.task:
    args.enable_cameras = True
app = AppLauncher(args).app

for _p in (str(W / "parkour" / "scripts"), str(W / "parkour" / "scripts" / "rsl_rl")):
    while _p in sys.path:
        sys.path.remove(_p)
for _p in (str(W), str(W / "parkour"), str(W / "parkour" / "parkour_tasks"), str(HERE)):
    if _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import parkour_isaaclab  # noqa: E402
import parkour_tasks  # noqa: E402,F401
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from parkour_tasks.crab_hex_omni_task.mdp import rt_friction, site_raster  # noqa: E402

FRICTION_COL = 89 + 108 + 9 + 4


def harness_overrides(env_cfg) -> dict:
    """eval_crab_hex_gait._apply_env_overrides' determinism knobs, restated (the harness module runs on import)."""
    applied = {"seed": int(args.env_seed)}
    env_cfg.seed = int(args.env_seed)
    env_cfg.episode_length_s = float(SCEN.episode_length_s or (SCEN.total_duration_s + 2.0))
    env_cfg.commands.base_velocity.resampling_time_range = (1.0e9, 1.0e9)
    env_cfg.commands.base_velocity.debug_vis = False
    env_cfg.curriculum = None
    if getattr(env_cfg, "parkours", None) is not None:
        env_cfg.parkours.base_parkour.freeze_terrain_levels = SCEN.freeze_terrain_levels
        env_cfg.parkours.base_parkour.debug_vis = False
        applied["freeze_terrain_levels"] = SCEN.freeze_terrain_levels
    tg = getattr(getattr(env_cfg.scene, "terrain", None), "terrain_generator", None)
    if tg is not None and SCEN.terrain_generator_seed is not None:
        tg.seed = SCEN.terrain_generator_seed
        applied["terrain_generator_seed"] = SCEN.terrain_generator_seed
    ev = getattr(env_cfg, "events", None)
    if ev is not None and getattr(ev, "push_by_setting_velocity", None) is not None:
        ev.push_by_setting_velocity = None
    face0 = SCEN.spawn_yaw_rad() if SCEN.command_frame == "world" else None
    if face0 is not None and ev is not None and getattr(ev, "reset_root_state", None) is not None:
        ev.reset_root_state.params["yaw_range"] = (face0, face0)
        rsi = getattr(ev, "rsi_reference_reset", None)
        if rsi is not None and "yaw_range" in (rsi.params or {}):
            rsi.params["yaw_range"] = (face0, face0)
        applied["spawn_yaw_pin"] = face0
    return applied


def main() -> int:
    t0 = time.time()
    out = {"tool": "rt_kit_eval_probe.py", "manifest": args.manifest, "scenario": args.scenario,
           "env_seed": int(args.env_seed), "num_envs": int(args.episodes), "task": args.task,
           "checkpoint": args.checkpoint, "launch_krabby": LAUNCH_KRABBY, "manifest_block": BLOCK,
           "pythonpath": LAUNCH_PYTHONPATH, "build": PROBE_BUILD,
           "modules": {"parkour_tasks": parkour_tasks.__file__, "parkour_isaaclab": (getattr(parkour_isaaclab, "__file__", None) or next(iter(parkour_isaaclab.__path__), None)),
                       "rt_friction": rt_friction.__file__, "site_raster": site_raster.__file__}}
    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=int(args.episodes))
    out["overrides"] = harness_overrides(env_cfg)
    pe = getattr(env_cfg.events, "physics_material", None)
    params = dict(getattr(pe, "params", {}) or {})
    out["material_event"] = getattr(getattr(pe, "func", None), "__name__", str(getattr(pe, "func", None)))
    out["rt_friction_params"] = {k: (list(v) if isinstance(v, tuple) else v) for k, v in params.items()
                                 if k in ("friction_range", "num_buckets", "gravel_range", "gravel_sub_terrains",
                                          "static_friction_range", "dynamic_friction_range")}
    env = gym.make(args.task, cfg=env_cfg)
    u = env.unwrapped
    obs, _ = env.reset()
    asset = u.scene["robot"]
    n = int(u.num_envs)
    root_reset = asset.data.root_pos_w.detach().cpu().double().numpy().copy()
    s_reset = site_raster.surface_under(u, asset.data.root_pos_w[:, 0], asset.data.root_pos_w[:, 1])
    s_reset = s_reset.detach().cpu().double().numpy()
    zero = torch.zeros(n, u.action_manager.total_action_dim, device=u.device)
    steps, s_steps = [], []
    with torch.inference_mode():
        for _ in range(int(args.steps)):
            obs, _r, _te, _tr, _i = env.step(zero)
            p = asset.data.root_pos_w
            steps.append(p.detach().cpu().double().numpy().copy())
            s_steps.append(site_raster.surface_under(u, p[:, 0], p[:, 1]).detach().cpu().double().numpy())
    pol = (obs["policy"] if isinstance(obs, dict) else obs).detach().cpu().double().numpy()
    npz = str(Path(args.out).with_suffix(".npz"))
    Path(npz).parent.mkdir(parents=True, exist_ok=True)
    site_raster.dump_site_layers(u, npz)
    out["layers_npz"] = npz
    layers = site_raster.site_layers(u)
    terrain = u.scene.terrain
    lv = terrain.terrain_levels.detach().cpu().numpy().astype(int)
    ty = terrain.terrain_types.detach().cpu().numpy().astype(int)
    names = np.asarray(terrain.terrain_generator_class.terrain_names)
    names = names[..., 0] if names.ndim == 3 else names
    mats = asset.root_physx_view.get_material_properties().detach().cpu().double()
    mu_attr = getattr(u, rt_friction.RT_MU_ATTR, None)
    envs = []
    for i in range(n):
        r, c = int(lv[i]), int(ty[i])
        tk = int(layers.tile_kind[r, c]) if layers is not None else -1
        envs.append({
            "index": i, "row": r, "col": c, "tile_name": str(names[r, c]),
            "tile_kind": site_raster.TILE_KINDS[tk] if 0 <= tk < len(site_raster.TILE_KINDS) else None,
            "tile_d": float(layers.tile_d[r, c]) if layers is not None else None,
            "centre_level_m": float(layers.tile_centre_level_m[r, c]) if layers is not None else 0.0,
            "root_reset": root_reset[i].tolist(), "surface_under_root_reset": float(s_reset[i]),
            "root_steps": [s[i].tolist() for s in steps], "surface_under_root_steps": [float(s[i]) for s in s_steps],
            "mu_shape_min": float(mats[i, :, 0].min()), "mu_shape_max": float(mats[i, :, 0].max()),
            "shapes_equal": bool(torch.equal(mats[i, :, 0], mats[i, :1, 0].expand_as(mats[i, :, 0]))),
            "static_eq_dynamic": bool(torch.equal(mats[i, :, 0], mats[i, :, 1])),
            "restitution_max": float(mats[i, :, 2].max()),
            "mu_eff_attr": (float(mu_attr[i]) if mu_attr is not None else None),
            "friction_obs": float(pol[i, FRICTION_COL]),
        })
    out["envs"] = envs
    out["obs_width"] = int(pol.shape[-1])
    out["wall_s"] = time.time() - t0
    out.pop("complete", None)
    out["complete"] = True
    Path(args.out).write_text(json.dumps(out, indent=1, default=float))
    print(f"[rt probe] wrote {args.out} ({n} envs, kinds {sorted({e['tile_kind'] for e in envs}, key=str)})", flush=True)
    env.close()
    return 0


if __name__ == "__main__":
    rc = 1
    try:
        rc = main()
    except Exception:  # noqa: BLE001
        import traceback

        traceback.print_exc()
    finally:
        app.close()
    sys.exit(rc)
