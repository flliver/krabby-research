"""Forward-era (2026-08 tripod campaign, pre-hardware-morphology plant) degenerates through the stroke
term ALONE. The gate tool refuses these stamps (not omni suites, not the sideways head, no joint_pos),
and their plant/economy differ from W's, so only the term-alone per-second income vs the SAME era's
healthy references is meaningful (R5 term-alone clause; R3's "v4 earns <= healthy h2000").

Adapter: category fwd, MOVE slot, never STAND/turning, lat_share 0, own = steady & hold segment >= 0.
Heading from root_quat_w; timing as the tool (state row s+1 with command row s).
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.argv = sys.argv[:1] + sys.argv[1:]
TOOL = Path("/home/nickmagus/krabby/krabby-research/.claude/worktrees/stroke-posture/parkour/parkour_tasks/"
            "parkour_tasks/crab_hex_omni_task/experiments/tools/replay_stroke_posture.py")
import importlib.util

spec = importlib.util.spec_from_file_location("rsp", TOOL)
RSP = importlib.util.module_from_spec(spec)
sys.modules["rsp"] = RSP
spec.loader.exec_module(RSP)

D = Path("/home/nickmagus/krabby/krabby-research/parkour/parkour_tasks/parkour_tasks/crab_hex_forward_task/"
         "experiments/2026-08-10_0058_tripod_stability")
RUNS = {
    "H2000 (healthy)": "healthy_refs/gait_eval_h2000",
    "H3000 (healthy)": "healthy_refs/gait_eval_h3000",
    "H5000 (healthy)": "fromscratch_tripod_v4_swap_short/gait_eval_h5000",
    "v1 lunge": "fromscratch_tripod_reward_w0.15/gait_eval",
    "v2 tip-rock": "fromscratch_tripod_v2_w0.15/gait_eval_midtrain",
    "v3 skate": "fromscratch_tripod_v3_w0.15_short/gait_eval_3000",
    "v3b drag": "fromscratch_tripod_v3b_slide_short/gait_eval_3000",
    "v4@3000 (tippy)": "fromscratch_tripod_v4_swap_short/gait_eval_3000",
    "v4@5000": "fromscratch_tripod_v4_swap_short/gait_eval_5000",
}
KEYS = ("foot_force_norm", "foot_pos_w", "foot_lin_vel_w", "root_pos_w", "root_quat_w", "root_lin_vel_b",
        "root_ang_vel_b", "cmd_applied", "segment_id", "steady_mask", "done")


def load(rel: str) -> "RSP.Stamp":
    run = sorted((D / rel / "flat_walk_forward" / "seed001").iterdir())[0]
    meta = json.loads((run / "run_meta.json").read_text())
    raws = sorted((run / "raw").glob("episode_*.npz"))
    seqs = {k: [] for k in KEYS}
    ents = []
    for i, p in enumerate(raws):
        with np.load(p) as z:
            for k in KEYS:
                seqs[k].append(z[k])
        ents.append({"env_index": i, "env": i, "wave": 0, "category": "fwd", "hand": None, "label": "fwd",
                     "hold_ids": (0, 1, 2)})
    T = max(len(s) for s in seqs["segment_id"])
    for e, seg, done in zip(ents, seqs["segment_id"], seqs["done"]):
        d = np.asarray(done, bool)
        e["T"] = len(seg)
        e["S"] = int(np.argmax(d)) if d.any() else len(seg) - 1
    arr = {k: RSP._pad_stack(v, T) for k, v in seqs.items()}
    N = len(ents)
    S = np.array([e["S"] for e in ents])
    alive = np.arange(T)[None, :] < S[:, None]
    seg = arr["segment_id"].astype(np.int64)
    own = (seg >= 0) & arr["steady_mask"].astype(bool) & alive
    resampled = np.zeros((N, T), bool)
    resampled[:, 1:] = seg[:, 1:] != seg[:, :-1]
    return RSP.Stamp(run_dir=str(run), kind="fwd_era", meta=meta, dt=float(meta.get("dt", 0.02)), turn_radius=1.2,
                     params_source="fwd-era adapter", episodes=ents, arr=arr,
                     psi=RSP.heading_from_quat(arr["root_quat_w"]), S=S, own=own,
                     stand=np.zeros((N, T), bool), turning=np.zeros((N, T), bool),
                     lat_share=np.zeros((N, T)), resampled=resampled, knee_idx=[], json_eps=[None] * N,
                     raw_paths=raws)


def per_s(st, rp, hold=None):
    g = st.own & rp.gate
    if hold is not None:
        g &= st.arr["segment_id"] == hold
    n = g.sum()
    return float(rp.pay.sum(-1)[g].sum() / (n * st.dt)) if n else None, int(n)


def main(params_kv=None, out=None):
    params = RSP.parse_params(params_kv or [])
    rows = {}
    for name, rel in RUNS.items():
        st = load(rel)
        rp = RSP.replay(st, params)
        sd = rp.stances
        k = st.own[sd["env"], sd["step"]] & rp.gate[sd["env"], sd["step"]] if sd["env"].size else np.zeros(0, bool)
        U = sd["ungated"][k].sum()
        kept = float(sd["pay"][k & sd["payable"]].sum() / U) if U > 0 else None
        tot, n = per_s(st, rp)
        rows[name] = {"per_s": tot, "n_steps": n, "kept": kept,
                      "per_hold": {h: per_s(st, rp, h)[0] for h in (0, 1, 2)},
                      "falls": int(sum(1 for e in st.episodes if e["S"] < e["T"] - 1)),
                      "median_B": float(np.median(sd["B"][k & sd["payable"]])) if (k & sd["payable"]).any() else None,
                      "median_rho": float(np.median(sd["rho"][k & sd["payable"]])) if (k & sd["payable"]).any() else None,
                      "slip_gate_mean": float(np.mean(sd["slip_gate"][k & sd["payable"]])) if (k & sd["payable"]).any() else None,
                      "g_prog_mean": float(np.mean(sd["g_prog"][k & sd["payable"]])) if (k & sd["payable"]).any() else None,
                      "scored": int((k & sd["payable"]).sum())}
    hs = [rows[n]["per_s"] for n in rows if "healthy" in n]
    hmin, h2000 = min(hs), rows["H2000 (healthy)"]["per_s"]
    for n, r in rows.items():
        r["ratio_vs_H2000"] = r["per_s"] / h2000 if r["per_s"] is not None else None
        r["ratio_vs_healthy_min"] = r["per_s"] / hmin if r["per_s"] is not None else None
    res = {"params": RSP.dataclasses.asdict(params), "rows": rows}
    if out:
        Path(out).write_text(json.dumps(RSP._jsonable(res), indent=1))
    for n, r in rows.items():
        print(f"{n:18s} S/s {r['per_s']:.4f}  vsH2000 {r['ratio_vs_H2000']:.3f}  vsHmin {r['ratio_vs_healthy_min']:.3f} "
              f"kept {r['kept'] if r['kept'] is None else round(r['kept'],3)} scored {r['scored']} medB {r['median_B']} "
              f"rho {r['median_rho']} slipG {r['slip_gate_mean']} prog {r['g_prog_mean']} falls {r['falls']} "
              f"holds {[None if v is None else round(v,3) for v in r['per_hold'].values()]}")
    return res


if __name__ == "__main__":
    args = sys.argv[1:]
    out = None
    if args and args[0].startswith("--out="):
        out = args[0].split("=", 1)[1]
        args = args[1:]
    main(args, out)
