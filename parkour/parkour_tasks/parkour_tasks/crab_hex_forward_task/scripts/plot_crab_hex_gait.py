# SPDX-License-Identifier: BSD-3-Clause
"""Render gait diagrams from a saved crab-hex eval run (Milestone 18, Task 0).

Standalone and post-hoc **by design**: matplotlib is not declared in ``parkour/setup.py`` or
``parkour_tasks/setup.py``, so importing it inside the harness would risk an ImportError destroying
a finished multi-minute GPU run. Keeping it separate also means diagrams can be re-rendered from
committed raw arrays without touching the GPU.

The diagram is the *human debug view*; the machine-readable judgment is the tripod score in
``scenario_metrics.json``. Use it to see **why** a score is low -- one lagging leg looks very
different from no phasing at all.

Runs under plain python (no ``isaaclab.sh``):
    python plot_crab_hex_gait.py --run-dir logs/rsl_rl/gait_eval/v1/<scenario>/seed001/<timestamp>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

FOOT_ORDER = ("FL_Footpad", "FR_Footpad", "ML_Footpad", "MR_Footpad", "RL_Footpad", "RR_Footpad")
# Tripod sets: A on top, B below, so a healthy gait reads as a checkerboard.
ROW_ORDER = ("FL_Footpad", "MR_Footpad", "RL_Footpad", "FR_Footpad", "ML_Footpad", "RR_Footpad")


def _plot_episode(npz_path: Path, out_path: Path, threshold: float, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")  # headless: no display on the training box
    import matplotlib.pyplot as plt

    data = np.load(npz_path)
    force = data["foot_force_norm"]  # (T, 6)
    dt = float(data["dt"]) if "dt" in data else 0.02
    contact = force > threshold
    n_steps = contact.shape[0]
    time = np.arange(n_steps) * dt

    fig, (ax, ax_slip) = plt.subplots(
        2, 1, figsize=(14, 6), height_ratios=[3, 1], sharex=True, constrained_layout=True
    )

    for row, name in enumerate(ROW_ORDER):
        col = FOOT_ORDER.index(name)
        y = len(ROW_ORDER) - 1 - row
        colour = "#1f77b4" if name in ("FL_Footpad", "MR_Footpad", "RL_Footpad") else "#d62728"
        in_contact = contact[:, col]
        starts = np.flatnonzero(np.diff(np.concatenate(([0], in_contact.view(np.int8), [0]))) > 0)
        ends = np.flatnonzero(np.diff(np.concatenate(([0], in_contact.view(np.int8), [0]))) < 0)
        for s, e in zip(starts, ends):
            ax.barh(y, (e - s) * dt, left=s * dt, height=0.72, color=colour, edgecolor="none")

    ax.set_yticks(range(len(ROW_ORDER)))
    ax.set_yticklabels([n.replace("_Footpad", "") for n in reversed(ROW_ORDER)])
    ax.set_ylabel("foot (tripod A blue / B red)")
    ax.set_title(title)
    ax.set_ylim(-0.6, len(ROW_ORDER) - 0.4)

    # Shade hold boundaries so a speed-dependent gait change is visible at a glance.
    if "segment_id" in data:
        seg = data["segment_id"]
        edges = np.flatnonzero(np.diff(seg)) + 1
        for e in edges:
            ax.axvline(e * dt, color="0.4", lw=0.8, ls="--")
            ax_slip.axvline(e * dt, color="0.4", lw=0.8, ls="--")

    if "foot_lin_vel_w" in data:
        # Slip is only meaningful while the foot is loaded; masking makes skating obvious.
        speed = np.linalg.norm(data["foot_lin_vel_w"][:, :, :2], axis=-1)
        stance_speed = np.where(contact, speed, np.nan)
        with np.errstate(invalid="ignore"):
            mean_stance_speed = np.nanmean(stance_speed, axis=1)
        ax_slip.plot(time, mean_stance_speed, color="#2ca02c", lw=1.0)
        ax_slip.set_ylabel("stance slip\n(m/s)")
    ax_slip.set_xlabel("time (s)")
    ax_slip.grid(alpha=0.3)

    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Render crab-hex gait diagrams from a saved eval run.")
    parser.add_argument("--run-dir", required=True, help="Run directory containing raw/*.npz.")
    parser.add_argument("--threshold", type=float, default=None, help="Contact force threshold override.")
    parser.add_argument("--max-episodes", type=int, default=4, help="Cap diagrams rendered (default 4).")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    raw_dir = run_dir / "raw"
    if not raw_dir.is_dir():
        print(f"[WARN] no raw/ under {run_dir}; nothing to plot", file=sys.stderr)
        return 0

    threshold = args.threshold
    scenario_id = run_dir.name
    meta_path = run_dir / "run_meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        scenario_id = meta.get("scenario_id", scenario_id)
        if threshold is None:
            for ep in sorted((run_dir / "metrics").glob("episode_*.json")):
                threshold = json.loads(ep.read_text()).get("contact_force_threshold")
                break
    threshold = 1.0 if threshold is None else float(threshold)

    for npz_path in sorted(raw_dir.glob("episode_*.npz"))[: args.max_episodes]:
        out = run_dir / f"gait_diagram_{npz_path.stem}.png"
        try:
            _plot_episode(npz_path, out, threshold, f"{scenario_id} — {npz_path.stem}")
            print(f"[INFO] wrote {out}")
        except Exception as exc:  # noqa: BLE001 - plotting is never fatal to a run
            print(f"[WARN] {npz_path.name}: {exc}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
