"""Offline re-score of saved gait-eval raws with the tracking-ratio metrics (2026-08-21).

The scoring path is pure numpy, so eval runs recorded before tracking became a first-class
aggregate can be re-scored from their raw/ npz without a GPU run. Writes
``scenario_metrics_tracking.json`` + ``summary_tracking.md`` alongside the originals (the
original files are left untouched — they are the record of what the harness printed at the
time).

Usage: <venv-python> rescore_tracking.py <eval_run_dir> [<eval_run_dir> ...]
"""
import json
import os
import sys
from types import SimpleNamespace

import numpy as np

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "..", ".."))
_SCRIPTS = os.path.join(_REPO, "parkour", "parkour_tasks", "parkour_tasks",
                        "crab_hex_forward_task", "scripts")
sys.path.insert(0, _SCRIPTS)

from gait_eval import metrics as M  # noqa: E402
from gait_eval import report as R  # noqa: E402
from gait_eval import schedule as S  # noqa: E402

MANIFEST = os.path.join(_REPO, "parkour", "parkour_tasks", "parkour_tasks",
                        "crab_hex_forward_task", "experiments", "eval", "scenarios_v2.yaml")

RAW_KEYS = (
    "actions", "applied_torque", "cmd_applied", "foot_ang_vel_w", "foot_force_norm",
    "foot_lin_vel_w", "foot_pos_w", "joint_vel", "root_ang_vel_b", "root_lin_vel_b",
    "root_pos_w", "root_quat_w", "terrain_z",
)


def rescore(run_dir: str) -> None:
    meta = json.load(open(os.path.join(run_dir, "run_meta.json")))
    old = json.load(open(os.path.join(run_dir, "scenario_metrics.json")))
    scenarios, _defaults = S.load_manifest(MANIFEST)
    scenario = next(s for s in scenarios if s.id == old["scenario_id"])
    labels = [h.label for h in scenario.schedule]
    term = {e["env_index"]: e["termination_reason"] for e in old["episodes"]}
    dt = float(meta["dt"])

    episodes = []
    for env_idx in sorted(term):
        f = os.path.join(run_dir, "raw", f"episode_{env_idx:02d}.npz")
        d = np.load(f)
        n = d["cmd_applied"].shape[0]
        raw = {k: np.asarray(d[k])[:, None] for k in RAW_KEYS if k in d.files}
        compiled = SimpleNamespace(
            segment_id=np.asarray(d["segment_id"])[:, None],
            steady_mask=np.asarray(d["steady_mask"]).astype(bool)[:, None],
            hold_labels=labels,
        )
        episodes.append(
            R.score_episode(
                raw, env_idx=0, compiled=compiled, scenario=scenario, dt=dt,
                term_reason=term[env_idx], n_steps=n,
            )
        )
        episodes[-1]["env_index"] = env_idx

    agg = R.aggregate(episodes)
    out = {"scenario_id": old["scenario_id"], "aggregate": agg,
           "rescored_from": "raw npz, tracking-metrics rev 2026-08-21",
           "episodes": [
               {"env_index": e["env_index"], "tripod_score": e["tripod_score"],
                "tracking_ratio": e["tracking_ratio"],
                "termination_reason": e["termination_reason"], "n_steps": e["n_steps"]}
               for e in episodes]}
    with open(os.path.join(run_dir, "scenario_metrics_tracking.json"), "w") as fh:
        json.dump(M.json_safe(out), fh, indent=2, sort_keys=True)
    with open(os.path.join(run_dir, "summary_tracking.md"), "w") as fh:
        fh.write("```\n" + R.summary_text(meta, episodes) + "\n```\n")
    t = agg["tracking_ratio"]
    print(f"{run_dir.rstrip('/').split('/')[-1]}: tracking_ratio median "
          f"{t['median']} (n={t['n']})")
    for label, hold in agg["tracking_by_hold"].items():
        print(f"  {label:>8}: cmd {hold['cmd_vx_mean']:.2f} -> achieved "
              f"{hold['achieved_vx']['median']:.3f} m/s (ratio median "
              f"{hold['ratio']['median']})")


if __name__ == "__main__":
    for run in sys.argv[1:]:
        rescore(run)
