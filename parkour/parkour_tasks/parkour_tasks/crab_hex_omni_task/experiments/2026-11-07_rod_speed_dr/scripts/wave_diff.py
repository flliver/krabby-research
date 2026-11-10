"""Per-wave first-difference table for two stamps of a multi-wave row (suite: 10 waves x 100 envs, run in sequence).

    wave_diff.py X Y [--period 5]

For each wave: episodes that differ, the earliest differing step (episode-local) and its keys, the keys that differ at
step 0, how many episodes differ at step 0 in the action and how many only in the forces (DYNAMIC), and whether every
kinematic state array (STATE) is bit-equal at step 0. Prints one JSON summary line last.

The harness records forces and torque before each env.step and rod speed after it, and env.reset() leaves the applied
torque, the contact force history and the actuator rod speed at the previous wave's end values; so in a later wave a
step-0 difference in DYNAMIC (and, through the current-sense observation, in the action) is carried state, while a
step-0 difference in STATE would be a reset difference.
"""
import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np

STATE = ("root_pos_w", "root_quat_w", "joint_pos", "joint_vel", "body_pos_w", "foot_pos_w", "foot_lin_vel_w",
         "foot_ang_vel_w", "root_lin_vel_b", "root_ang_vel_b", "terrain_z", "cmd_applied", "obs_cmd_vx", "clock_obs",
         "clock_phase", "clock_cmd_speed", "clock_running", "segment_id", "travel_yaw", "heading_err", "rod_speed")
DYNAMIC = ("applied_torque", "foot_force_norm", "leg_contact_force")


def one(job):
    x, y, e = job
    a, b = np.load(x / "raw" / f"episode_{e:02d}.npz"), np.load(y / "raw" / f"episode_{e:02d}.npz")
    first, keys, step0 = None, [], []
    for k in a.files:
        u, v = a[k], b[k]
        if u.ndim == 0:
            continue
        n = min(len(u), len(v))
        neq = np.any(np.ascontiguousarray(u[:n]).view(np.uint8).reshape(n, -1)
                     != np.ascontiguousarray(v[:n]).view(np.uint8).reshape(n, -1), axis=1)
        if not neq.any():
            continue
        t = int(np.argmax(neq))
        if t == 0:
            step0.append(k)
        if first is None or t < first:
            first, keys = t, [k]
        elif t == first:
            keys.append(k)
    return e, first, sorted(keys), sorted(step0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("x", type=Path)
    ap.add_argument("y", type=Path)
    ap.add_argument("--period", type=int, default=5)
    a = ap.parse_args()
    meta = json.loads((a.x / "run_meta.json").read_text())
    wave_of = {int(r["env_index"]): int(r.get("wave", 0)) for r in meta.get("env_index_map", [])}
    eps = sorted(int(r["env_index"]) for r in json.loads((a.x / "scenario_metrics.json").read_text())["episodes"])
    with Pool(8) as pool:
        res = pool.map(one, [(a.x, a.y, e) for e in eps])
    out = {"x": str(a.x), "y": str(a.y), "waves": {}}
    for w in sorted({wave_of.get(e, 0) for e in eps}):
        rw = [r for r in res if wave_of.get(r[0], 0) == w]
        d = [r for r in rw if r[1] is not None]
        s0 = [r for r in d if r[3]]
        pose0 = sorted({k for r in s0 for k in r[3] if k not in DYNAMIC and k != "actions"})
        act0 = sum(1 for r in s0 if "actions" in r[3])
        dyn_only0 = sum(1 for r in s0 if r[3] and set(r[3]) <= set(DYNAMIC))
        kinds = {}
        for r in d:
            kind = ("action-only@depth" if r[2] == ["actions"] and r[1] % a.period == 0 else
                    "step0-dynamic" if r[1] == 0 and set(r[2]) <= set(DYNAMIC) | {"actions"} else "other")
            kinds[kind] = kinds.get(kind, 0) + 1
        out["waves"][w] = {
            "episodes": len(rw), "differ": len(d),
            "earliest_step": min((r[1] for r in d), default=None),
            "earliest_keys": sorted({k for r in d if r[1] == min(q[1] for q in d) for k in r[2]}) if d else [],
            "n_step0_diff": len(s0), "step0_keys": sorted({k for r in s0 for k in r[3]}),
            "n_step0_action_diff": act0, "n_step0_dynamic_only": dyn_only0,
            "step0_state_keys_differ": pose0, "first_diff_kinds": kinds}
        print(f"wave {w}: {len(d)}/{len(rw)} differ; earliest step {out['waves'][w]['earliest_step']} keys "
              f"{out['waves'][w]['earliest_keys']}; step-0 diffs in {len(s0)} eps, keys {out['waves'][w]['step0_keys']}"
              f" (action in {act0}, forces only in {dyn_only0}); non-force state keys differing at step 0: "
              f"{pose0 or 'none'}; kinds {kinds}")
    print(json.dumps(out))


if __name__ == "__main__":
    main()
