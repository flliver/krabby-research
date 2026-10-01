"""Extract hold-window [4,10) s signals for every side-bisect stamp into a compact cache (CPU, read-only).

Per (stamp, hold label), per valid episode (classifier hold_stats != None AND alive through t0+500):
  act   (300,18) clipped actions  (action_joint_names order)
  jp    (300,18) joint pos: Body_Hip x6, Hip_Femur x6, Femur_Tibia x6 (leg order FL FR ML MR RL RR)
  camv  (300,6)  cam-shaft joint velocity (signed shaft speed, leg order)
  foot  (300,18) foot pos in yaw-body frame rel. root: x x6, y x6, z x6
  body  (300,4)  lean_x_b, lean_y_b (root - feet centroid, yaw-body frame), roll, pitch
  contact (300,6) foot force > 50 N
  ph    (300,)   clock_phase (nan if not logged)
  jpx   (450,18) joint pos over [t0+100, t0+550) (clipped at death) for the intrinsic phase
  stats: feet_ss, sweep, lean_abs (classifier hold_stats verbatim)

ppo_construction copy (2026-10-01) of the side bisect's scripts/rootcause/sym/extract.py. Changes: the stamp registry
is DATA, not code, and the cache is durable (symlib.CACHE, overridable):
  extract.py [--cache DIR] [--jobs N] NAME=STAMP_DIR [NAME:GROUP=STAMP_DIR ...]
Each NAME is cached as <cache>/<NAME>__<label>.npz; <cache>/registry.json is MERGED (a re-extracted NAME replaces
its old entry). The classifier is this campaign's verbatim copy, ../side22_classify.py. one() is verbatim.
"""
import argparse
import importlib.util
import json
import sys
from pathlib import Path
import numpy as np
from concurrent.futures import ProcessPoolExecutor

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import symlib  # noqa: E402

spec = importlib.util.spec_from_file_location("sc", HERE.parent / "side22_classify.py")
sc = importlib.util.module_from_spec(spec); spec.loader.exec_module(sc)

OUT = symlib.CACHE
LEGS = ["FL", "FR", "ML", "MR", "RL", "RR"]


def parse_specs(specs):
    """NAME=STAMP_DIR or NAME:GROUP=STAMP_DIR -> [(name, group, path)] (group defaults to 'run')."""
    items = []
    for spec in specs:
        if "=" not in spec:
            raise SystemExit(f"expected NAME=STAMP_DIR, got {spec!r}")
        name, path = spec.split("=", 1)
        group = "run"
        if ":" in name:
            name, group = name.split(":", 1)
        if not (Path(path) / "run_meta.json").is_file():
            raise SystemExit(f"{name}: {path} has no run_meta.json")
        items.append((name, group, str(Path(path))))
    return items


def quat_rp(q):
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1, 1))
    return roll, pitch


def one(item):
    name, group, d = item
    d = Path(d)
    meta = json.loads((d / "run_meta.json").read_text())
    labels = meta["hold_labels"]
    want = [l for l in labels if l.startswith("lat_")]
    an = meta["action_joint_names"]; jn = meta["joint_names"]
    jidx = {n: i for i, n in enumerate(jn)}
    jsel = [jidx[f"{L}_{t}_RevoluteJoint"] for t in ("Body_Hip", "Hip_Femur", "Femur_Tibia") for L in LEGS]
    csel = [jidx[f"{L}_Body_CamShaft_RevoluteJoint"] for L in LEGS]
    fb = meta["foot_body_names"]; assert [f[:2] for f in fb] == LEGS, fb
    if meta.get("suite"):
        files = [(d / "raw" / f"episode_{e['env_index']:02d}.npz", e["label"]) for e in meta["env_index_map"] if e["label"] in want]
    else:
        files = [(p, None) for p in sorted((d / "raw").glob("episode_*.npz"))]
    rec = {l: {k: [] for k in ("act", "jp", "camv", "foot", "body", "contact", "ph", "jpx", "feet_ss", "sweep", "lean_abs", "canon", "ep")} for l in want}
    nexcl = {l: 0 for l in want}
    for p, lab in files:
        z = np.load(p)
        done = z["done"].astype(bool)
        Sd = int(np.argmax(done)) if done.any() else len(z["segment_id"])
        seg = z["segment_id"]
        for l in ([lab] if lab else want):
            sid = labels.index(l)
            idx = np.where((seg == sid) & (np.arange(len(seg)) < Sd))[0]
            if len(idx) == 0:
                continue
            hs = sc.hold_stats(z, sid, Sd)
            t0 = idx[0]
            if hs is None or idx[-1] < t0 + 500:
                nexcl[l] += 1
                continue
            a4, hi = t0 + 200, t0 + 500
            sl = slice(a4, hi)
            r = rec[l]
            r["act"].append(np.clip(z["actions"][sl].astype(np.float32), -1, 1))
            r["jp"].append(z["joint_pos"][sl][:, jsel].astype(np.float32))
            r["camv"].append(z["joint_vel"][sl][:, csel].astype(np.float32))
            rp = z["root_pos_w"].astype(float); q = z["root_quat_w"].astype(float)
            psi = sc.yaw(q)[sl]; c, s = np.cos(psi), np.sin(psi)
            fp = z["foot_pos_w"][sl].astype(float) - rp[sl][:, None, :]
            fx = c[:, None] * fp[..., 0] + s[:, None] * fp[..., 1]
            fy = -s[:, None] * fp[..., 0] + c[:, None] * fp[..., 1]
            fz = fp[..., 2]
            r["foot"].append(np.concatenate([fx, fy, fz], 1).astype(np.float32))
            lx, ly = -fx.mean(1), -fy.mean(1)   # root minus feet centroid, body frame
            roll, pitch = quat_rp(q[sl])
            r["body"].append(np.stack([lx, ly, roll, pitch], 1).astype(np.float32))
            r["contact"].append((z["foot_force_norm"][sl] > 50).astype(np.int8))
            r["ph"].append(z["clock_phase"][sl].astype(np.float32) if "clock_phase" in z.files else np.full(300, np.nan, np.float32))
            end = min(t0 + 550, idx[-1] + 1, Sd)
            jpx = np.full((450, 18), np.nan, np.float32)
            jpx[: end - (t0 + 100)] = z["joint_pos"][t0 + 100:end][:, jsel]
            r["jpx"].append(jpx)
            r["feet_ss"].append(hs["feet_ss"]); r["sweep"].append(hs["sweep"]); r["lean_abs"].append(hs["lean_abs"])
            r["canon"].append(float(z["canon_flag"][sl].mean()) if "canon_flag" in z.files else np.nan)
            r["ep"].append(str(p.name))
    for l, r in rec.items():
        if not r["act"]:
            continue
        np.savez_compressed(OUT / f"{name}__{l}.npz", **{k: np.array(v) for k, v in r.items()})
    return name, {l: (len(rec[l]["act"]), nexcl[l]) for l in want}, an, jn


def _one_in(arg):
    """Pool entry point: set the cache in the worker (spawn/forkserver-safe), then one()."""
    global OUT
    item, cache = arg
    OUT = Path(cache)
    return one(item)


def main(argv=None):
    global OUT
    ap = argparse.ArgumentParser(description="Extract hold-window signals of eval stamps into the sym cache.")
    ap.add_argument("--cache", default=None, help=f"cache dir (default {symlib.CACHE})")
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("stamps", nargs="+", help="NAME=STAMP_DIR or NAME:GROUP=STAMP_DIR")
    a = ap.parse_args(argv)
    OUT = symlib.set_cache(a.cache) if a.cache else symlib.set_cache(symlib.CACHE)
    items = parse_specs(a.stamps)
    an = jn = None
    with ProcessPoolExecutor(max(1, a.jobs)) as ex:
        for name, counts, an, jn in ex.map(_one_in, [(i, str(OUT)) for i in items]):
            print(name, counts, flush=True)
    reg_path = OUT / "registry.json"
    reg = json.loads(reg_path.read_text()) if reg_path.is_file() else {"stamps": []}
    keep = [s for s in reg.get("stamps", []) if s[0] not in {i[0] for i in items}]
    reg.update({"stamps": keep + [list(i) for i in items], "action_joint_names": an, "joint_names": jn})
    reg_path.write_text(json.dumps(reg, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
