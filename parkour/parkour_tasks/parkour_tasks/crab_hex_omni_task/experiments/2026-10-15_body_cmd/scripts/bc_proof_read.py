#!/usr/bin/env python3
"""Step 3 harness proof (CPU): a world head driven on body rows through ``body_hold`` -- 0 route / command / obs
mismatches, step by step, on every episode of the stamp.

Checks per step of every episode's whole schedule (prologue excluded):
  obs      obs_cmd = policy obs cols 5..10: col 5 == cmd wz; cols 8, 9 == cmd vx, vy; col 10 == |cmd xy|;
           cols 6 == 7 == clip(heading_err, +-facing_obs_clip) (the world head's raw encoding)
  command  cmd xy == the hold row's (vx, vy) (the world term converts dir = yaw + atan2(vy, vx) back to the body);
           cmd wz == the world relay of heading_err (clip(k err, +-ang_vel_max), 0 inside the dead band)
  adapter  |wrap(adapter_face - yaw)| <= windup; on unclamped steps face advances by wz_row * dt
  route    run_meta: command_adapter body_hold, env frame world, manifest route term

    bc_proof_read.py --stamp DIR [--json OUT]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

TOL = 1e-5


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def yaw_of(q):
    return np.arctan2(2 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]), 1 - 2 * (q[:, 2] ** 2 + q[:, 3] ** 2))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stamp", required=True)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    S = Path(a.stamp)
    m = json.loads((S / "run_meta.json").read_text())
    dt = float(m["dt"])
    env = m.get("env_vars") or {}
    clip_obs = float(env.get("KRABBY_OMNI_FACING_OBS_CLIP", "0.6"))
    k = float(m.get("heading_control_stiffness", 1.5))
    ang_max = float(m.get("ang_vel_max", 0.3))
    ang_clip = float(m.get("ang_vel_clip", 0.2))
    windup = float((m.get("adapter") or {}).get("windup_rad", 0.6))
    rows = {h["label"]: h for h in m["hold_body_cmds"]}
    labels = m["hold_labels"]
    route = {"command_adapter": m.get("command_adapter"), "env_command_frame": m.get("env_command_frame"),
             "command_route": m.get("command_route")}
    route_ok = route == {"command_adapter": "body_hold", "env_command_frame": "world", "command_route": "term"}
    counts = {"obs_wz": 0, "obs_xy": 0, "obs_speed": 0, "obs_facing": 0, "cmd_xy": 0, "cmd_wz_relay": 0,
              "face_windup": 0, "face_step": 0}
    steps = 0
    worst = {k2: 0.0 for k2 in counts}
    for row in m["env_index_map"]:
        z = np.load(S / "raw" / f"episode_{row['env_index']:02d}.npz")
        seg = z["segment_id"].astype(int)
        live = (seg >= 0) & ~z["done"]
        cmd, obs, he, face = z["cmd_applied"], z["obs_cmd"], z["heading_err"], z["adapter_face"]
        yaw = yaw_of(z["root_quat_w"])
        lab = np.array([labels[s] if s >= 0 else "prologue" for s in seg])
        rx = np.array([rows[l]["vx"] if l in rows else 0.0 for l in lab])
        ry = np.array([rows[l]["vy"] if l in rows else 0.0 for l in lab])
        rw = np.array([rows[l]["wz"] if l in rows else 0.0 for l in lab])
        idx = np.where(live)[0]
        steps += idx.size

        def tally(name, err):
            e = np.abs(err[idx])
            counts[name] += int((e > TOL).sum())
            worst[name] = max(worst[name], float(e.max()) if e.size else 0.0)

        tally("obs_wz", obs[:, 0] - cmd[:, 2])
        tally("obs_xy", np.maximum(np.abs(obs[:, 3] - cmd[:, 0]), np.abs(obs[:, 4] - cmd[:, 1])))
        tally("obs_speed", obs[:, 5] - np.hypot(cmd[:, 0], cmd[:, 1]))
        fe = np.clip(he, -clip_obs, clip_obs)
        tally("obs_facing", np.maximum(np.abs(obs[:, 1] - fe), np.abs(obs[:, 2] - fe)))
        tally("cmd_xy", np.maximum(np.abs(cmd[:, 0] - rx), np.abs(cmd[:, 1] - ry)))
        relay = np.clip(k * he, -ang_max, ang_max)
        relay = np.where(np.abs(relay) < ang_clip, 0.0, relay)
        tally("cmd_wz_relay", cmd[:, 2] - relay)
        tally("face_windup", np.maximum(np.abs(wrap(face - yaw)) - windup, 0.0))
        d = np.zeros_like(face)
        d[1:] = wrap(face[1:] - face[:-1]) - rw[1:] * dt
        unclamped = np.zeros_like(live)
        unclamped[1:] = (np.abs(wrap(face[1:] - yaw[1:])) < windup - 1e-3) & (seg[1:] == seg[:-1]) & (seg[:-1] >= 0)
        e = np.abs(d[unclamped & live])
        counts["face_step"] += int((e > TOL).sum())
        worst["face_step"] = max(worst["face_step"], float(e.max()) if e.size else 0.0)
    total = sum(counts.values())
    res = {"stamp": str(S), "route": route, "route_ok": route_ok, "steps": steps, "mismatches": counts,
           "worst_abs": worst, "tol": TOL, "pass": route_ok and total == 0}
    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=2) + "\n")
    print(json.dumps(res, indent=1))
    return 0 if res["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
