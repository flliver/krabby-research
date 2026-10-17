"""PLAN step 8 C3 / C4 (GPU, Kit): the payload READBACK -- what PhysX holds after the payload event wrote it.

Builds ``--task`` headless with the process env the queue passes (the training env for ``--mode train``: Teacher-v0
+ p3a.train.env, or the distill env: Student-v0 + p3a.distill env; a level row's env for ``--mode level``), resets
every env (the payload term captures each env's post-startup-DR base lazily at that first reset), then runs
``--rounds`` PARTIAL resets of a random subset (>= 2 envs, at most a quarter) through the env's own ``_reset_idx``.
After every reset it reads ``root_physx_view.get_masses / get_coms / get_inertias`` for the payload body and compares
with an INDEPENDENT float64 composite of the term's record (its captured base, the drawn kg and CoM, the box size):

    m = m0 + mp;  c = (m0 c0 + mp cp) / m;  I = I0 + m0 S(c0 - c) + Ip + mp S(cp - c),  S(d) = |d|^2 E - d d^T,
    Ip = mp / 12 diag(b^2 + h^2, a^2 + h^2, a^2 + b^2)

Recorded per read (STEPC_RULES C3.1 holds the tolerances; the reader applies them): max |mass error| (kg), max
|CoM error| (m, inf-norm), max relative error of the PRINCIPAL moments (sorted eigenvalues -- frame-invariant,
because PhysX may re-diagonalise), the full-tensor relative error in the link frame (report), 0-kg envs bitwise equal
to the captured base, envs NOT in the reset subset bitwise unchanged, every drawn kg inside the range, the transitions
seen (loaded -> 0, 0 -> loaded, loaded -> loaded), the zero-draw share, and for ``--mode level`` every env at exactly
``--level-kg`` with CoM (0, 0, com_z).

    <isaac python> t4_stepC_readback_play.py --headless --task T --mode train|level [--level-kg 90]
        [--num-envs 16] [--rounds 24] [--seed 20261016] --out JSON
Writes ``{"complete": true, ...}`` last; exit 0 when written.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Task 4 step 8: payload PhysX readback after partial resets.")
parser.add_argument("--task", required=True)
parser.add_argument("--mode", choices=("train", "level"), required=True)
parser.add_argument("--level-kg", type=float, default=None)
parser.add_argument("--num-envs", type=int, default=16)
parser.add_argument("--rounds", type=int, default=24)
parser.add_argument("--seed", type=int, default=20261016)
parser.add_argument("--out", required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

R = Path(__file__).resolve().parents[7]
PT = R / "parkour" / "parkour_tasks"
if str(PT) not in sys.path:
    sys.path.insert(0, str(PT))

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import parkour_tasks  # noqa: E402,F401
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from parkour_tasks.crab_hex_omni_task.mdp.omni_events import payload_term  # noqa: E402


def read(term) -> dict:
    v = term.asset.root_physx_view
    b = term.body_id
    return {"m": v.get_masses()[:, b].detach().cpu().double().numpy().copy(),
            "c": v.get_coms()[:, b, :].detach().cpu().double().numpy().copy(),
            "i": v.get_inertias()[:, b, :].detach().cpu().double().numpy().copy()}


def S(d):
    return float(d @ d) * np.eye(3) - np.outer(d, d)


def expected(term, e: int, box) -> dict:
    m0 = float(term.base_mass[e])
    c0 = term.base_com[e, :3].double().numpy()
    i0 = term.base_inertia[e].double().numpy().reshape(3, 3).T
    mp = float(term.payload_kg[e])
    if mp <= 0.0:
        return {"m": m0, "c": c0, "i": i0, "loaded": False}
    cp = term.payload_com[e].double().numpy()
    a, bb, h = (float(x) for x in box)
    ip = np.diag([bb * bb + h * h, a * a + h * h, a * a + bb * bb]) * mp / 12.0
    m = m0 + mp
    c = (m0 * c0 + mp * cp) / m
    i = i0 + m0 * S(c0 - c) + ip + mp * S(cp - c)
    return {"m": m, "c": c, "i": i, "loaded": True}


def compare(term, rd: dict, envs, box) -> dict:
    out = {"mass_err_kg": 0.0, "com_err_m": 0.0, "eig_rel_err": 0.0, "tensor_rel_err": 0.0, "zero_kg_exact": True,
           "n": 0}
    for e in envs:
        x = expected(term, e, box)
        mr, cr, ir = rd["m"][e], rd["c"][e, :3], rd["i"][e].reshape(3, 3)
        out["mass_err_kg"] = max(out["mass_err_kg"], abs(mr - x["m"]))
        out["com_err_m"] = max(out["com_err_m"], float(np.abs(cr - x["c"]).max()))
        ew = np.sort(np.linalg.eigvalsh(0.5 * (x["i"] + x["i"].T)))
        er = np.sort(np.linalg.eigvalsh(0.5 * (ir + ir.T)))
        out["eig_rel_err"] = max(out["eig_rel_err"], float(np.abs(er - ew).max() / max(abs(ew).max(), 1e-12)))
        out["tensor_rel_err"] = max(out["tensor_rel_err"], float(np.abs(ir - x["i"]).max() / max(abs(x["i"]).max(), 1e-12)))
        if not x["loaded"]:
            base = (float(term.base_mass[e]), term.base_com[e].double().numpy(), term.base_inertia[e].double().numpy())
            same = (mr == base[0] and np.array_equal(rd["c"][e], base[1]) and np.array_equal(rd["i"][e], base[2]))
            out["zero_kg_exact"] = out["zero_kg_exact"] and bool(same)
        out["n"] += 1
    return out


def main() -> int:
    t0 = time.time()
    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    env = gym.make(args.task, cfg=env_cfg)
    u = env.unwrapped
    env.reset()
    term = payload_term(env)
    res = {"tool": "t4_stepC_readback_play.py", "task": args.task, "mode": args.mode, "num_envs": int(u.num_envs),
           "krabby_env": {k: v for k, v in sorted(os.environ.items()) if k.startswith("KRABBY_")}}
    if term is None:
        res.update({"term_present": False, "complete": True, "wall_s": time.time() - t0})
        Path(args.out).write_text(json.dumps(res, indent=1, sort_keys=True))
        env.close()
        return 0
    p = term.cfg.params
    box = [float(x) for x in p["box_size"]]
    res.update({"term_present": True, "params": {"kg_range": [float(x) for x in p["kg_range"]],
                                                 "zero_frac": float(p["zero_frac"]), "com_xy": float(p["com_xy"]),
                                                 "com_z": float(p["com_z"]), "box_size": box},
                "captured": term.base_mass is not None,
                "base_mass_spread_kg": (float(term.base_mass.max() - term.base_mass.min())
                                        if term.base_mass is not None else None)})
    N = int(u.num_envs)
    reads = []
    rd = read(term)
    reads.append({"round": 0, "reset": list(range(N)), **compare(term, rd, range(N), box)})
    g = np.random.default_rng(args.seed)
    trans = {"loaded_to_zero": 0, "zero_to_loaded": 0, "loaded_to_loaded": 0, "zero_to_zero": 0}
    draws, zeros, out_of_range = 0, 0, 0
    lo, hi = res["params"]["kg_range"]
    untouched_ok = True
    for r in range(1, args.rounds + 1):
        k = int(g.integers(2, max(3, N // 4 + 1)))
        ids = sorted(int(x) for x in g.choice(N, size=min(k, N), replace=False))
        before_kg = term.payload_kg.clone()
        prev = rd
        u._reset_idx(torch.tensor(ids, device=u.device, dtype=torch.long))
        rd = read(term)
        others = [e for e in range(N) if e not in ids]
        for e in others:
            same = (prev["m"][e] == rd["m"][e] and np.array_equal(prev["c"][e], rd["c"][e])
                    and np.array_equal(prev["i"][e], rd["i"][e]))
            untouched_ok = untouched_ok and bool(same)
        for e in ids:
            a, b = float(before_kg[e]), float(term.payload_kg[e])
            trans[("loaded" if a > 0 else "zero") + "_to_" + ("loaded" if b > 0 else "zero")] += 1
            draws += 1
            zeros += int(b <= 0.0)
            out_of_range += int(b > 0.0 and not (lo - 1e-9 <= b <= hi + 1e-9))
        reads.append({"round": r, "reset": ids, **compare(term, rd, ids, box)})
    worst = {k: max(x[k] for x in reads) for k in ("mass_err_kg", "com_err_m", "eig_rel_err", "tensor_rel_err")}
    res.update({"reads": reads, "worst": worst, "zero_kg_exact": all(x["zero_kg_exact"] for x in reads),
                "untouched_envs_bitwise": untouched_ok, "transitions": trans, "draws": draws,
                "zero_share": (zeros / draws) if draws else None, "out_of_range_draws": out_of_range,
                "distinct_masses_at_end": int(len(set(np.round(rd["m"], 6).tolist()))),
                "term_run_meta": term.run_meta()})
    if args.mode == "level":
        L = float(args.level_kg)
        kg = term.payload_kg.double().numpy()
        com = term.payload_com.double().numpy()
        res["level"] = {"kg": L, "every_env_at_L": bool(np.all(kg == L)),
                        "com_exact": bool(np.all(com[:, 0] == 0.0) and np.all(com[:, 1] == 0.0)
                                          and np.allclose(com[:, 2], res["params"]["com_z"], atol=1e-12))}
    res["wall_s"] = time.time() - t0
    res["complete"] = True
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=1, sort_keys=True, default=float))
    print(f"[t4c readback] wrote {args.out}: worst {worst}", flush=True)
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
