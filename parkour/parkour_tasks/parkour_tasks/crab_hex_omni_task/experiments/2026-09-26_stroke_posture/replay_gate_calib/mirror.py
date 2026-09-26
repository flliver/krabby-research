"""R8 on real rollouts: the term's hand symmetry, separated from the head's gait asymmetry.

Mirror every episode of a stamp through M_y (y -> -y; FL<->FR, ML<->MR, RL<->RR; heading -> -heading;
v_y, w_z -> -v_y, -w_z; command (vx, vy, wz) -> (vx, -vy, -wz)) and replay both. A hand-symmetric term pays
each mirrored episode exactly what it pays the original (float32 tolerance). Any l/r income gap in the
replay is then the HEAD's gait asymmetry (plus eval noise), not the term's.
usage: mirror.py <run name> [k=v ...]
"""
import copy
import dataclasses
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from diag import RSP, RUNS  # noqa: E402

SWAP = [1, 0, 3, 2, 5, 4]   # FL FR ML MR RL RR -> FR FL MR ML RR RL


def mirror_stamp(st):
    arr = {k: np.array(v, copy=True) for k, v in st.arr.items()}
    for k in ("foot_force_norm",):
        arr[k] = arr[k][:, :, SWAP]
    for k in ("foot_pos_w", "foot_lin_vel_w"):
        a = arr[k][:, :, SWAP].copy()
        a[..., 1] *= -1
        arr[k] = a
    arr["root_pos_w"][..., 1] *= -1
    arr["root_lin_vel_b"][..., 1] *= -1
    arr["root_ang_vel_b"][..., 0] *= -1
    arr["root_ang_vel_b"][..., 2] *= -1
    arr["cmd_applied"][..., 1] *= -1
    arr["cmd_applied"][..., 2] *= -1
    return dataclasses.replace(st, arr=arr, psi=-st.psi)


def main(name, params):
    st = RSP.load_stamp(RUNS[name])
    rp = RSP.replay(st, params)
    stm = mirror_stamp(st)
    rpm = RSP.replay(stm, params)
    g = st.own & rp.gate
    gm = stm.own & rpm.gate
    assert (g == gm).all()
    inc = np.array([rp.pay[i].sum(-1)[g[i]].sum() for i in range(st.N)])
    incm = np.array([rpm.pay[i].sum(-1)[gm[i]].sum() for i in range(st.N)])
    cen = np.array([rp.b[i][g[i]].sum() for i in range(st.N)])
    cenm = np.array([rpm.b[i][gm[i]].sum() for i in range(st.N)])
    rel = np.abs(incm - inc) / np.maximum(np.abs(inc), 1e-9)
    out = {"run": name, "episodes": int(st.N), "stroke_total": float(inc.sum()), "stroke_total_mirrored": float(incm.sum()),
           "stroke_max_abs_episode_diff": float(np.abs(incm - inc).max()),
           "stroke_max_rel_episode_diff": float(rel[inc > 1e-3].max()),
           "centring_max_abs_episode_diff": float(np.abs(cenm - cen).max()),
           "centring_total": float(cen.sum()), "centring_total_mirrored": float(cenm.sum())}
    print(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    res = main(sys.argv[1], RSP.parse_params(sys.argv[2:]))
    (Path.cwd() / f"mirror_{sys.argv[1]}.json").write_text(json.dumps(res, indent=1))
