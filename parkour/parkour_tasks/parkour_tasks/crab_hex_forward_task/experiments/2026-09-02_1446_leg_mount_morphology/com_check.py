#!/usr/bin/env python3
"""Exact whole-body CoM (body_pos_w + run_meta masses) vs the root-position proxy used by
support_polygon_metrics across rungs (i)-(iv): walking / pre-fall tip-angle deltas on one
transfer run. Usage: com_check.py <run_dir> (a gait_eval run with raw/ episodes)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[6]
sys.path.insert(0, str(REPO / "parkour/parkour_tasks/parkour_tasks/crab_hex_forward_task/scripts"))
from gait_eval import metrics as M  # noqa: E402


def main(run_dir: Path) -> int:
    meta = json.loads((run_dir / "run_meta.json").read_text())
    masses = np.asarray(meta["body_masses_kg"], dtype=np.float64)
    dt = float(meta["dt"])
    rows = []
    for f in sorted((run_dir / "raw").glob("episode_*.npz")):
        z = np.load(f)
        if "body_pos_w" not in z.files:
            print("no body_pos_w in", f.name); return 1
        com = (z["body_pos_w"].astype(np.float64) * masses[None, :, None]).sum(1) / masses.sum()
        walking = z["steady_mask"] & (np.abs(z["cmd_applied"][:, 0]) >= 0.2)
        args = (z["foot_pos_w"], z["foot_force_norm"], z["root_pos_w"], z["root_quat_w"])
        kw = dict(dt=dt, crab_failure=z["crab_failure"], walking_mask=walking)
        proxy = M.support_polygon_metrics(*args, **kw)
        exact = M.support_polygon_metrics(*args, com_pos_w=com, **kw)
        r = {"ep": f.stem}
        for win in ("walking", "prefall"):
            for key in ("tip_angle_fwd_deg", "lead_contact_tip_deg", "fwd_ray_margin_m"):
                a, b = proxy[win][key]["p50"], exact[win][key]["p50"]
                r[f"{win}_{key}_proxy"], r[f"{win}_{key}_exact"] = a, b
            r[f"{win}_neg_proxy"], r[f"{win}_neg_exact"] = proxy[win]["frac_neg_margin"], exact[win]["frac_neg_margin"]
        r["com_minus_root_x_body"] = float(np.median(com[:, 0] - z["root_pos_w"][:, 0]))
        r["com_z"] = float(np.median(com[:, 2])); r["root_z"] = float(np.median(z["root_pos_w"][:, 2]))
        rows.append(r)
    def med(key):
        v = [r[key] for r in rows if r.get(key) is not None]
        return float(np.median(v)) if v else float("nan")
    print(f"{run_dir.name}: {len(rows)} episodes; CoM − root: x {med('com_minus_root_x_body'):+.3f} m (world), z {med('com_z') - med('root_z'):+.3f} m")
    for win in ("walking", "prefall"):
        for key in ("tip_angle_fwd_deg", "lead_contact_tip_deg", "fwd_ray_margin_m"):
            print(f"  {win:8s} {key:22s} proxy {med(f'{win}_{key}_proxy'):7.3f}  exact {med(f'{win}_{key}_exact'):7.3f}  Δ {med(f'{win}_{key}_exact') - med(f'{win}_{key}_proxy'):+.3f}")
        print(f"  {win:8s} frac_neg_margin        proxy {med(f'{win}_neg_proxy'):7.3f}  exact {med(f'{win}_neg_exact'):7.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1])))
