"""PLAN step 8 C4 (GPU, Kit): the per-camera geometry check on the BUILT omni Student-Play env -- the four
RayCasterCameras as they actually cast, not as configured.

Builds ``Isaac-Crab-Hex-Omni-Student-Play-v0`` headless (the process env = the row env file the queue passes, minus
the student runner's refused knobs), resets, holds zero actions for ``--settle-steps`` policy steps (the default joint
pose: the stance), then reads every camera once per reset (``--resets`` resets; the omni reset draws a fresh world
yaw each time) and measures, per camera (STEPC_RULES C4.1 -- the pass rules are applied by t4_stepC_read.py):

* AXIS ERROR (deg): the camera's optical axis in the BODY frame against the spec's ``(cos p cos y, cos p sin y,
  -sin p)`` (omni_student_spec.expected_axes) -- two ways: (a) the sensor's world quaternion (``quat_w_world`` x-axis)
  rotated into the root frame, (b) the mean of the cast ray directions (``ray_hits_w - pos_w``, unit), which is the
  optical axis for a symmetric pinhole grid. Max over envs and resets.
* GROUND BAND (flat mode): the in-range rays (slant distance <= max_distance) that land on the ground (|z - the env's
  ground level| < 0.02 m; the ground level = the median z of that camera's in-range hits) projected on the camera's
  look axis in the body frame (front +x, left +y, right -y, rear -x); per env [min, max] -> the share of envs whose band
  covers cameras.yaml ``target_4b`` [from, to].
* FIELD-BLOCK HITS (field mode): in-range rays more than 0.02 m off the ground level (block tops / walls / pits) --
  per camera the count and the number of envs with >= 5 such rays.
* IMAGE SANITY: share of envs whose depth image has at least one pixel below the clip (the camera sees something).

    <isaac python> t4_stepC_camera_play.py --headless --task Isaac-Crab-Hex-Omni-Student-Play-v0 --mode flat|field
        [--num-envs 64] [--resets 4] [--settle-steps 50] --out JSON
Writes ``{"complete": true, ...}`` last; exit 0 when written (the verdict is the reader's).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys
import time
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Task 4 step 8: omni student camera geometry check.")
parser.add_argument("--task", default="Isaac-Crab-Hex-Omni-Student-Play-v0")
parser.add_argument("--mode", choices=("flat", "field"), required=True)
parser.add_argument("--num-envs", type=int, default=64)
parser.add_argument("--resets", type=int, default=4)
parser.add_argument("--settle-steps", type=int, default=50)
parser.add_argument("--out", required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

R = Path(__file__).resolve().parents[7]
PT = R / "parkour" / "parkour_tasks"
if str(PT) not in sys.path:
    sys.path.insert(0, str(PT))

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

import parkour_tasks  # noqa: E402,F401  (registers the gym ids)
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

OMNI = R / "parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task"
CAMERAS_YAML = OMNI / "experiments/2026-10-16_task4_pipeline/cameras.yaml"
LOOK = {"+x": (0, 1.0), "-x": (0, -1.0), "+y": (1, 1.0), "-y": (1, -1.0)}
GROUND_TOL = 0.02
BLOCK_MIN_RAYS = 5


def _spec():
    p = OMNI / "config/crab_hex/omni_student_spec.py"
    s = importlib.util.spec_from_file_location("t4c_omni_student_spec", str(p))
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


def qrot(q, v):
    """Rotate v (..., 3) by unit quaternions q (..., 4) (w, x, y, z)."""
    w, u = q[..., :1], q[..., 1:]
    t = 2.0 * torch.cross(u, v, dim=-1)
    return v + w * t + torch.cross(u, t, dim=-1)


def qconj(q):
    return torch.cat([q[..., :1], -q[..., 1:]], dim=-1)


def angle_deg(a, b):
    a = a / a.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    b = b / b.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return torch.rad2deg(torch.acos((a * b).sum(-1).clamp(-1.0, 1.0)))


def main() -> int:
    t0 = time.time()
    spec = _spec()
    specs = spec.camera_specs()
    cy = yaml.safe_load(CAMERAS_YAML.read_text())["cameras"]
    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    env = gym.make(args.task, cfg=env_cfg)
    u = env.unwrapped
    robot = u.scene["robot"]
    act = torch.zeros(u.num_envs, u.action_manager.total_action_dim, device=u.device)
    per = {n: {"axis_quat_deg": [], "axis_rays_deg": [], "band_lo": [], "band_hi": [], "covers": [], "block_rays": 0,
               "block_envs": 0, "sees_something": [], "n_in_range": []} for n in spec.CAMERA_NAMES}
    for r in range(args.resets):
        env.reset()
        for _ in range(args.settle_steps):
            env.step(act)
        rq = robot.data.root_quat_w
        rp = robot.data.root_pos_w
        for name in spec.CAMERA_NAMES:
            sp = specs[name]
            cam = u.scene[sp["attr"]]
            d = cam.data                                  # triggers the lazy update
            pos, qw = d.pos_w, d.quat_w_world
            hits = cam.ray_hits_w                          # (N, rays, 3)
            want = torch.tensor(spec.expected_axes(sp["yaw_deg"], sp["pitch_down_deg"])["forward"],
                                device=pos.device, dtype=pos.dtype).expand(pos.shape[0], 3)
            ax_w = qrot(qw, torch.tensor([1.0, 0.0, 0.0], device=pos.device).expand_as(pos))
            ax_b = qrot(qconj(rq), ax_w)
            per[name]["axis_quat_deg"] += angle_deg(ax_b, want).tolist()
            vec = hits - pos[:, None, :]
            fin = torch.isfinite(vec).all(-1)
            dist = torch.where(fin, vec.norm(dim=-1), torch.full_like(vec[..., 0], float("inf")))
            unit = torch.where(fin[..., None], vec / dist[..., None].clamp_min(1e-9), torch.zeros_like(vec))
            mean_dir = unit.sum(1) / fin.sum(1, keepdim=True).clamp_min(1)
            per[name]["axis_rays_deg"] += angle_deg(qrot(qconj(rq), mean_dir), want).tolist()
            inr = dist <= float(spec.MAX_DISTANCE_M)
            per[name]["n_in_range"] += inr.sum(1).tolist()
            img = d.output["distance_to_camera"].reshape(pos.shape[0], -1)
            per[name]["sees_something"] += (img < float(spec.MAX_DISTANCE_M) - 1e-4).any(1).tolist()
            hz = hits[..., 2]
            look_axis = cy[name]["target_4b"]["look_axis"]
            ai, sgn = LOOK[look_axis]
            t_from, t_to = float(cy[name]["target_4b"]["from"]), float(cy[name]["target_4b"]["to"])
            hb = qrot(qconj(rq)[:, None, :].expand(-1, hits.shape[1], -1), hits - rp[:, None, :])
            coord = sgn * hb[..., ai]
            for e in range(pos.shape[0]):
                m = inr[e]
                if int(m.sum()) == 0:
                    per[name]["band_lo"].append(None)
                    per[name]["band_hi"].append(None)
                    per[name]["covers"].append(False)
                    continue
                gz = float(hz[e][m].median())
                gm = m & ((hz[e] - gz).abs() < GROUND_TOL)
                bm = m & ((hz[e] - gz).abs() > GROUND_TOL)
                nb = int(bm.sum())
                per[name]["block_rays"] += nb
                per[name]["block_envs"] += int(nb >= BLOCK_MIN_RAYS)
                if int(gm.sum()) == 0:
                    per[name]["band_lo"].append(None)
                    per[name]["band_hi"].append(None)
                    per[name]["covers"].append(False)
                    continue
                lo, hi = float(coord[e][gm].min()), float(coord[e][gm].max())
                per[name]["band_lo"].append(lo)
                per[name]["band_hi"].append(hi)
                per[name]["covers"].append(lo <= t_from and hi >= t_to)
    out = {"tool": "t4_stepC_camera_play.py", "task": args.task, "mode": args.mode, "num_envs": int(u.num_envs),
           "resets": args.resets, "settle_steps": args.settle_steps, "ground_tol_m": GROUND_TOL,
           "block_min_rays": BLOCK_MIN_RAYS, "max_distance_m": float(spec.MAX_DISTANCE_M),
           "krabby_env": {k: v for k, v in sorted(os.environ.items()) if k.startswith("KRABBY_")}, "cameras": {}}
    for name, v in per.items():
        lo = sorted(x for x in v["band_lo"] if x is not None)
        hi = sorted(x for x in v["band_hi"] if x is not None)
        out["cameras"][name] = {
            "attr": specs[name]["attr"], "yaw_deg": specs[name]["yaw_deg"], "pitch_down_deg": specs[name]["pitch_down_deg"],
            "target_4b": cy[name]["target_4b"], "predicted_band": cy[name].get("ground_band_m"),
            "axis_quat_deg_max": max(v["axis_quat_deg"]), "axis_rays_deg_max": max(v["axis_rays_deg"]),
            "band_lo_median": lo[len(lo) // 2] if lo else None, "band_hi_median": hi[len(hi) // 2] if hi else None,
            "band_lo_max": lo[-1] if lo else None, "band_hi_min": hi[0] if hi else None,
            "cover_share": sum(v["covers"]) / max(1, len(v["covers"])), "n_reads": len(v["covers"]),
            "block_rays": v["block_rays"], "block_envs": v["block_envs"],
            "sees_something_share": sum(v["sees_something"]) / max(1, len(v["sees_something"])),
            "in_range_rays_median": sorted(v["n_in_range"])[len(v["n_in_range"]) // 2]}
    out["wall_s"] = time.time() - t0
    out["complete"] = True
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1, sort_keys=True))
    print(f"[t4c camera] wrote {args.out}", flush=True)
    env.close()
    return 0


if __name__ == "__main__":
    rc = 1
    try:
        rc = main()
    except Exception:  # noqa: BLE001 -- print before the Kit teardown can hang
        import traceback

        traceback.print_exc()
    finally:
        app.close()
    sys.exit(rc)
