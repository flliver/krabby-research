#!/usr/bin/env python3
"""batched_sfm — photo-spine pipeline: chunk / solve / stitch / merge.

STO-SCN-049 + STO-SCN-050 (EPI-SCN-PHOTO-SPINE-PIPELINE). Positions
arbitrarily large photo pools by solving temporal chunks (≤300 frames —
the measured MASt3R-SfM ceiling on 16 GB) and chaining their gauges
through overlap cameras (gauge_align.umeyama, residual hard gates).

Subcommands (run from anywhere; paths are scene-store absolute/relative):

  chunk   --pool <dir> --out <spine-dir> [--chunk-size 300] [--overlap 50]
          Splits the (sorted) pool into overlapping temporal chunks:
          <spine-dir>/chunk-NN/data/<frames...> + chunks.json manifest.

  solve   --spine <spine-dir> --chunk NN [--image krabby-matcha:...]
          [--snapshot ~/scratch/MAtCha]
          Runs MASt3R-SfM (--sfm_only) on one chunk in the container.
          (One GPU job; farm different chunks to different hosts.)

  stitch  --spine <spine-dir> [--max-residual 0.10]
          Chains every solved chunk into chunk-01's gauge through the
          overlap cameras; writes <spine-dir>/spine_cameras.json
          (schema-5 pool shape: filepaths/focals/cams2world) +
          stitch_report.json. Fails loudly on a residual-gate breach
          or an unsolved chunk.

Frames are matched across chunks by BASENAME (the overlap contract).
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
# numpy + gauge_align imported lazily inside cmd_stitch — chunk/solve must
# run on hosts whose system python has no numpy (the fleet GPU hosts).

IMG_EXT = (".jpg", ".jpeg", ".png")


def now() -> str:
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


# ── chunk ────────────────────────────────────────────────────────────────────

def cmd_chunk(args) -> int:
    pool = Path(args.pool).resolve()
    out = Path(args.out).resolve()
    frames = sorted(f for f in os.listdir(pool) if f.lower().endswith(IMG_EXT))
    n, size, ov = len(frames), args.chunk_size, args.overlap
    if size > 300:
        print(f"WARNING: chunk-size {size} exceeds the measured 300-frame "
              f"solve ceiling (RTX 5080 16 GB) — expect OOM.")
    if not 3 <= ov < size:
        sys.exit(f"ERROR: overlap {ov} must be in [3, chunk-size)")
    step = size - ov
    n_chunks = max(1, -(-(n - size) // step) + 1) if n > size else 1
    chunks = []
    for c in range(n_chunks):
        lo = min(c * step, max(0, n - size))
        hi = min(lo + size, n)
        sel = frames[lo:hi]
        cdir = out / f"chunk-{c+1:02d}"
        (cdir / "data").mkdir(parents=True, exist_ok=True)
        # RELATIVE symlinks (not copies): keeps the store lean (T-016) and
        # resolves inside any container mount that includes both the pool
        # and the spine dir (they share the scene root).
        rel_pool = os.path.relpath(pool, cdir / "data")
        for f in sel:
            dst = cdir / "data" / f
            if not dst.exists():
                os.symlink(os.path.join(rel_pool, f), dst)
        chunks.append({"chunk": c + 1, "lo": lo, "hi": hi, "n": len(sel),
                       "first": sel[0], "last": sel[-1]})
        print(f"chunk-{c+1:02d}: frames [{lo}:{hi}) = {len(sel)}")
    manifest = {
        "schema_version": "1", "kind": "photo-spine",
        "pool": str(pool), "pool_n": n,
        "chunk_size": size, "overlap": ov, "chunks": chunks,
        "created": now(), "story": "STO-SCN-049",
    }
    (out / "chunks.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {out / 'chunks.json'} ({n_chunks} chunks, overlap {ov})")
    return 0


# ── solve ────────────────────────────────────────────────────────────────────

def cmd_solve(args) -> int:
    spine = Path(args.spine).resolve()
    cdir = spine / f"chunk-{args.chunk:02d}"
    if not (cdir / "data").is_dir():
        sys.exit(f"ERROR: {cdir}/data missing — run `chunk` first")
    if (cdir / "out" / "mast3r_sfm" / "cameras.json").is_file() and not args.force:
        print(f"chunk-{args.chunk:02d} already solved — skip (--force to redo)")
        return 0
    (cdir / "out").mkdir(exist_ok=True)
    # Mount the spine's PARENT (the scene input/ dir): chunk frames are
    # relative symlinks into the sibling pool dir, so the mount must
    # contain both or the links dangle inside the container (learned the
    # hard way on the first 005 batch, 2026-06-10).
    work = spine.parent
    mounts = ["-v", f"{work}:/work"]
    snapshot = os.path.expanduser(args.snapshot) if args.snapshot else None
    if snapshot and os.path.isdir(snapshot):
        mounts += ["-v", f"{snapshot}:/opt/MAtCha"]
    rel = f"/work/{spine.name}/chunk-{args.chunk:02d}"
    inner = (
        "source /opt/matcha/bin/activate && "
        "export PYTHONPATH=/opt/MAtCha:/opt/MAtCha/mast3r:/opt/MAtCha/mast3r/dust3r:"
        "/opt/MAtCha/2d-gaussian-splatting:/opt/MAtCha/2d-gaussian-splatting/submodules/simple-knn && "
        "export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && "
        "cd /opt/MAtCha && "
        "python -c 'import torch; assert torch.cuda.is_available()' && "
        f"python train.py -s {rel}/data -o {rel}/out --sfm_config unposed --sfm_only"
    )
    cmd = (["docker", "run", "--rm", "--gpus", "all", "--shm-size=8g",
            "--ipc=host", "--ulimit", "memlock=-1", "--ulimit", "stack=67108864"]
           + mounts + ["--entrypoint", "bash", args.image, "-lc", inner])
    print(f"solving chunk-{args.chunk:02d} ({args.image})...")
    rc = subprocess.run(cmd).returncode
    cams = cdir / "out" / "mast3r_sfm" / "cameras.json"
    if rc != 0 or not cams.is_file():
        sys.exit(f"ERROR: chunk-{args.chunk:02d} solve failed (rc={rc}, "
                 f"cameras.json {'present' if cams.is_file() else 'MISSING'})")
    d = json.loads(cams.read_text())
    print(f"chunk-{args.chunk:02d} solved: {len(d['cams2world'])} poses")
    return 0


# ── stitch ───────────────────────────────────────────────────────────────────

def _np():
    import numpy as np
    return np


def _load_chunk(spine: Path, c: int):
    np = _np()
    p = spine / f"chunk-{c:02d}" / "out" / "mast3r_sfm" / "cameras.json"
    if not p.is_file():
        sys.exit(f"ERROR: chunk-{c:02d} unsolved ({p} missing) — refuse to "
                 f"stitch a spine with holes")
    d = json.loads(p.read_text())
    basenames = [fp.rsplit("/", 1)[-1] for fp in d["filepaths"]]
    return basenames, np.asarray(d["focals"], dtype=np.float64), \
        np.asarray(d["cams2world"], dtype=np.float64)


def cmd_stitch(args) -> int:
    np = _np()
    from gauge_align import align_camera_sets, apply_to_cams2world
    spine = Path(args.spine).resolve()
    manifest = json.loads((spine / "chunks.json").read_text())
    n_chunks = len(manifest["chunks"])

    # Reference gauge = chunk 1.
    names, focals, c2w = _load_chunk(spine, 1)
    pose = {n: (c2w[i], focals[i]) for i, n in enumerate(names)}
    order = list(names)
    report = []
    for c in range(2, n_chunks + 1):
        cn, cf, cc2w = _load_chunk(spine, c)
        shared = [n for n in cn if n in pose]
        if len(shared) < 3:
            sys.exit(f"ERROR: chunk-{c:02d} shares only {len(shared)} frames "
                     f"with the spine — need ≥3 (overlap broken)")
        src = np.stack([cc2w[cn.index(n)][:3, 3] for n in shared])
        dst = np.stack([pose[n][0][:3, 3] for n in shared])
        src_R = np.stack([cc2w[cn.index(n)][:3, :3] for n in shared])
        dst_R = np.stack([pose[n][0][:3, :3] for n in shared])
        try:
            a = align_camera_sets(src, dst, max_residual=args.max_residual,
                                  src_rotations=src_R, dst_rotations=dst_R)
        except RuntimeError as e:
            sys.exit(f"ERROR: stitching chunk-{c:02d}: {e}")
        mapped = apply_to_cams2world(cc2w, a["scale"], a["R"], a["t"])
        added = 0
        for i, n in enumerate(cn):
            if n not in pose:           # overlap frames keep their spine pose
                pose[n] = (mapped[i], cf[i])
                order.append(n)
                added += 1
        report.append({"chunk": c, "shared": len(shared), "added": added,
                       "scale": a["scale"],
                       "max_residual_m": a["max_residual"],
                       "mean_residual_m": a["mean_residual"]})
        print(f"chunk-{c:02d}: {len(shared)} shared | scale {a['scale']:.4f} "
              f"| residual max {a['max_residual']:.4f} m mean "
              f"{a['mean_residual']:.4f} m | +{added} poses")

    out = {
        "filepaths": order,
        "focals": [float(pose[n][1]) for n in order],
        "cams2world": [pose[n][0].tolist() for n in order],
    }
    (spine / "spine_cameras.json").write_text(json.dumps(out) + "\n")
    (spine / "stitch_report.json").write_text(json.dumps({
        "schema_version": "1", "kind": "photo-spine-stitch",
        "n_chunks": n_chunks, "n_poses": len(order),
        "max_residual_gate_m": args.max_residual,
        "stitches": report, "created": now(), "story": "STO-SCN-050",
    }, indent=2) + "\n")
    print(f"\nspine complete: {len(order)} poses across {n_chunks} chunks → "
          f"{spine / 'spine_cameras.json'}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("chunk")
    c.add_argument("--pool", required=True)
    c.add_argument("--out", required=True)
    c.add_argument("--chunk-size", type=int, default=300)
    c.add_argument("--overlap", type=int, default=50)
    s = sub.add_parser("solve")
    s.add_argument("--spine", required=True)
    s.add_argument("--chunk", type=int, required=True)
    s.add_argument("--image", default="krabby-matcha:latest")
    s.add_argument("--snapshot", default="~/scratch/MAtCha",
                   help="host MAtCha source for legacy images; ignored if absent")
    s.add_argument("--force", action="store_true")
    t = sub.add_parser("stitch")
    t.add_argument("--spine", required=True)
    t.add_argument("--max-residual", type=float, default=0.10)
    args = ap.parse_args()
    return {"chunk": cmd_chunk, "solve": cmd_solve, "stitch": cmd_stitch}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
