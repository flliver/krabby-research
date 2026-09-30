"""Two-direction side RSI bank from canonical eval stamps (CPU; CHARTER "Mirror canonicalization", step B).

Usage: harvest_canon_bank.py --out BANK.npz [--per-dir 2000] [--seed 0] STAMP_DIR [STAMP_DIR ...]

Each STAMP_DIR is a ``harvest_side_lr__omni`` (or any category_suite) eval stamp run with ``--canon wave`` on a fast
one-direction head: lat_l holds are walked by the head itself, lat_r holds by the mirrored head. Frames are taken
from each hold's own episode with the 2026-09-16 bank's filters (``2026-09-16_1805_omni_formation/RSI_BANK.md`` and
``merge_rsi_banks.py``):

* hold-relative step >= 100 (2 s past onset), every 3rd step, before any termination;
* upright > 0.95 (body z-axis world z), |wz_b| < 0.3;
* the travel angle atan2(vy_b, vx_b) within +-20 deg of the hold's axis (+90 lat_l, -90 lat_r) and ||v_b,xy|| in
  [0.15, 0.5] (so the along-axis speed is > 0.14; the 0.15 along-axis gate is applied too).

Source tags: 1 = +y (lat_l, the old bank's sideways/left convention), 2 = -y (lat_r). ``--per-dir`` frames per tag are
sampled (seeded, without replacement), concatenated and shuffled. The bank carries the six arrays the omni reset
event reads (``crab_hex_omni_rsi.BANK_ARRAYS``) + ``source`` + a JSON ``meta``. Train with KRABBY_OMNI_RSI_SOURCE=all.
"""
import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

AXIS = {"l": 90.0, "r": -90.0}
TAG = {"l": 1, "r": 2}
TOL, SP_LO, SP_HI, ALONG_MIN, WZ_MAX, UP_MIN, MIN_STEP, THIN = 20.0, 0.15, 0.5, 0.15, 0.3, 0.95, 100, 3


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def upright(q):  # world z of the body z-axis, quat (w, x, y, z)
    x, y = q[:, 1], q[:, 2]
    return 1.0 - 2.0 * (x * x + y * y)


def frames_of(stamp: Path):
    meta = json.loads((stamp / "run_meta.json").read_text())
    if not meta.get("canon") or meta["canon"].get("mode") != "wave":
        raise SystemExit(f"{stamp}: not a --canon wave stamp")
    labels = meta["hold_labels"]
    out = {"l": [], "r": []}
    counts = {}
    for e in meta["env_index_map"]:
        lab = e["label"]
        if not lab.startswith("lat_"):
            continue
        hand = lab.split("_")[1]
        z = np.load(stamp / "raw" / f"episode_{e['env_index']:02d}.npz")
        seg = z["segment_id"]
        idx = np.where(seg == labels.index(lab))[0]
        if len(idx) <= MIN_STEP:
            continue
        steps = idx[MIN_STEP::THIN]
        q = z["root_quat_w"][steps].astype(np.float64)
        vb = z["root_lin_vel_b"][steps].astype(np.float64)
        wz = z["root_ang_vel_b"][steps, 2].astype(np.float64)
        sp = np.linalg.norm(vb[:, :2], axis=1)
        ang = np.degrees(np.arctan2(vb[:, 1], vb[:, 0]))
        along = vb[:, 1] * (1.0 if hand == "l" else -1.0)
        ok = (upright(q) > UP_MIN) & (np.abs(wz) < WZ_MAX) & (np.abs(((ang - AXIS[hand]) + 180) % 360 - 180) <= TOL) \
            & (sp >= SP_LO) & (sp <= SP_HI) & (along > ALONG_MIN)
        c = counts.setdefault(lab, [0, 0])
        c[0] += len(steps); c[1] += int(ok.sum())
        s = steps[ok]
        if len(s):
            out[hand].append({
                "joint_pos": z["joint_pos"][s], "joint_vel": z["joint_vel"][s], "root_quat_w": z["root_quat_w"][s],
                "root_z": z["root_pos_w"][s, 2], "root_lin_vel_b": z["root_lin_vel_b"][s], "clock_phase": z["clock_phase"][s],
            })
    return meta, out, counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-dir", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("stamps", nargs="+")
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    pools = {"l": [], "r": []}
    inputs, all_counts = [], {}
    for sd in a.stamps:
        meta, out, counts = frames_of(Path(sd))
        inputs.append({"stamp": sd, "checkpoint": meta.get("checkpoint"), "checkpoint_sha256": meta.get("checkpoint_sha256"),
                       "canon": meta.get("canon", {}).get("mode")})
        all_counts[sd] = counts
        for h in pools:
            pools[h] += out[h]
    arrays, source = {k: [] for k in ("joint_pos", "joint_vel", "root_quat_w", "root_z", "root_lin_vel_b", "clock_phase")}, []
    stats = {}
    for h in ("l", "r"):
        cat = {k: np.concatenate([f[k] for f in pools[h]]) for k in arrays}
        n = len(cat["joint_pos"])
        take = rng.choice(n, size=min(a.per_dir, n), replace=False)
        for k in arrays:
            arrays[k].append(cat[k][take])
        source.append(np.full(len(take), TAG[h], dtype=np.int64))
        vb = cat["root_lin_vel_b"][take]
        sp = np.linalg.norm(vb[:, :2], axis=1); ang = np.degrees(np.arctan2(vb[:, 1], vb[:, 0]))
        stats[h] = {"pool": int(n), "sampled": int(len(take)), "speed_p10_50_90": np.percentile(sp, [10, 50, 90]).round(3).tolist(),
                    "angle_p10_50_90": np.percentile(ang, [10, 50, 90]).round(1).tolist(),
                    "root_z_median": float(np.median(cat["root_z"][take]))}
    bank = {k: np.concatenate(v) for k, v in arrays.items()}
    src = np.concatenate(source)
    perm = rng.permutation(len(src))
    bank = {k: v[perm].astype(np.float32) for k, v in bank.items()}
    src = src[perm]
    # re-check the shuffled result (must be 100 %)
    vb = bank["root_lin_vel_b"].astype(np.float64); sp = np.linalg.norm(vb[:, :2], axis=1)
    ang = np.degrees(np.arctan2(vb[:, 1], vb[:, 0])); axis = np.where(src == 1, 90.0, -90.0)
    assert np.all(np.abs(((ang - axis) + 180) % 360 - 180) <= TOL) and np.all((sp >= SP_LO) & (sp <= SP_HI)), "bank check failed"
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    except Exception:
        head = None
    meta = {"kind": "two-direction side RSI bank (canonical harvest)", "created": datetime.now(timezone.utc).isoformat(),
            "git_head": head, "inputs": inputs, "per_dir": a.per_dir, "seed": a.seed,
            "source_tags": {"1": "+y (lat_l, head as trained)", "2": "-y (lat_r, M_y-canonicalized head)"},
            "filters": {"min_hold_step": MIN_STEP, "thin": THIN, "upright_min": UP_MIN, "wz_max": WZ_MAX, "angle_tol_deg": TOL,
                        "speed": [SP_LO, SP_HI], "along_min": ALONG_MIN},
            "per_hold_counts_candidates_kept": all_counts, "stats": stats, "n_total": int(len(src))}
    np.savez(a.out, **bank, source=src, meta=np.array(json.dumps(meta)))
    print(json.dumps({"out": a.out, "sha256": sha(a.out), "stats": stats, "n_total": int(len(src))}, indent=1))


if __name__ == "__main__":
    main()
