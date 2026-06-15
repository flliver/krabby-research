"""da3_infer_posed.py — STO-SCN-090: pose-conditioned DA3 inference (da3@1).

The unposed driver (da3_infer_gs.py, baked into krabby-da3) lets DA3
estimate its own camera poses; on 003-firepit that estimate disagrees
with the ingest solve at 60.7% of camera spread (scale 0.58) — the
fuse correctly refuses. DepthAnything3.inference() natively accepts
extrinsics (N,4,4 w2c) + intrinsics (N,3,3, input-image pixel space),
so this driver feeds the store's ingest solve instead — the same
architectural move as the matcha@1 posed weld.

Runs INSIDE the krabby-da3 container; staged to /work by v4exec and
invoked as `python /work/da3_infer_posed.py /work/images /work/out
<res> [nogs]`. Expects /work/cameras/posed.json (see
colmap_posed.solve_to_posed_json). Versioned in research/real2sim
(T-023: fix logic here, never inline divergent copies).
"""
import glob
import json
import os
import sys
import time

import numpy as np
import torch
from depth_anything_3.api import DepthAnything3

src, out = sys.argv[1], sys.argv[2]
process_res = int(sys.argv[3]) if len(sys.argv) > 3 else 504
no_gs = len(sys.argv) > 4 and sys.argv[4] == "nogs"

posed = json.load(open(os.path.join(os.path.dirname(src.rstrip("/")),
                                    "cameras", "posed.json")))
by_name = {e["name"]: e for e in posed}
images = sorted(glob.glob(f"{src}/*.jpeg") + glob.glob(f"{src}/*.jpg")
                + glob.glob(f"{src}/*.png"))
names = [p.rsplit("/", 1)[-1] for p in images]
missing = [n for n in names if n not in by_name]
if missing:
    sys.exit(f"posed REFUSED: images without posed cameras: {missing}")
ext = np.stack([np.asarray(by_name[n]["w2c"], dtype=np.float64) for n in names])
ixt = np.stack([np.asarray(by_name[n]["K"], dtype=np.float64) for n in names])
print(f"{len(images)} images (pose-conditioned: ingest solve)")

# ── STO-SCN-105 fix: recover the gaussian→solve-world similarity ─────────────
# DA3 builds the gs_ply in its OWN normalized frame (recenter-to-camera-0 +
# median-distance rescale), then inside inference() it Umeyama-aligns its
# PREDICTED poses to our input poses but DISCARDS that transform (it only uses
# the scale for depth, and overwrites the exported cameras with our input — so
# the npz/colmap cameras are the echoed input and can't recover the frame).
# The discarded alignment IS the gaussian→world transform (scale + ROTATION +
# translation; the rotation is why the raw splat looks ~125° off). We capture
# DA3's raw predicted poses at that call site, then Umeyama them to our input
# poses ourselves to get the exact similarity. See knowledge/da3-gsply-
# normalized-frame.md. Verified: ByteDance-Seed/Depth-Anything-3 api.py.

def _centers(W):
    """w2c (N,3,4|4,4) -> camera centers C = -R^T t."""
    W = np.asarray(W, dtype=np.float64)
    return np.array([-(w[:3, :3].T @ w[:3, 3]) for w in W])


def _umeyama(A, B):
    """least-squares similarity mapping A->B (B ≈ s R A + t). Returns s,R,t."""
    mA, mB = A.mean(0), B.mean(0)
    A0, B0 = A - mA, B - mB
    C = (B0.T @ A0) / len(A)
    U, D, Vt = np.linalg.svd(C)
    S = np.eye(3)
    if np.linalg.det(U @ Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    s = float(np.trace(np.diag(D) @ S) / ((A0 ** 2).sum() / len(A)))
    t = mB - s * R @ mA
    return s, R, t


def _R_to_quat_xyzw(R):
    """rotation matrix -> three.js quaternion [x,y,z,w]."""
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2
        w = 0.25 * S; x = (R[2, 1]-R[1, 2])/S; y = (R[0, 2]-R[2, 0])/S; z = (R[1, 0]-R[0, 1])/S
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        S = np.sqrt(1.0 + R[0, 0]-R[1, 1]-R[2, 2]) * 2
        w = (R[2, 1]-R[1, 2])/S; x = 0.25*S; y = (R[0, 1]+R[1, 0])/S; z = (R[0, 2]+R[2, 0])/S
    elif R[1, 1] > R[2, 2]:
        S = np.sqrt(1.0 + R[1, 1]-R[0, 0]-R[2, 2]) * 2
        w = (R[0, 2]-R[2, 0])/S; x = (R[0, 1]+R[1, 0])/S; y = 0.25*S; z = (R[1, 2]+R[2, 1])/S
    else:
        S = np.sqrt(1.0 + R[2, 2]-R[0, 0]-R[1, 1]) * 2
        w = (R[1, 0]-R[0, 1])/S; x = (R[0, 2]+R[2, 0])/S; y = (R[1, 2]+R[2, 1])/S; z = 0.25*S
    return [float(x), float(y), float(z), float(w)]


# monkeypatch align_poses_umeyama across DA3 modules to capture raw predicted poses
_cap = {}
def _mk(orig):
    def _w(pred_ext, inp_ext, *a, **k):
        if "pred_ext" not in _cap:
            try:
                _cap["pred_ext"] = np.asarray(pred_ext).copy()
            except Exception:
                pass
        return orig(pred_ext, inp_ext, *a, **k)
    return _w
_patched = []
for _mn, _mod in list(sys.modules.items()):
    if _mn.startswith("depth_anything_3") and hasattr(_mod, "align_poses_umeyama"):
        _o = getattr(_mod, "align_poses_umeyama")
        if callable(_o):
            setattr(_mod, "align_poses_umeyama", _mk(_o))
            _patched.append((_mod, _o))
print(f"[gauge] patched align_poses_umeyama in {len(_patched)} DA3 module(s)")

model = DepthAnything3.from_pretrained("/opt/checkpoints/DA3NESTED-GIANT-LARGE-1.1")
model = model.to("cuda")
t0 = time.time()
pred = model.inference(
    images,
    extrinsics=ext,
    intrinsics=ixt,
    infer_gs=not no_gs,
    export_dir=out,
    export_format="npz-colmap-glb" if no_gs else "glb-npz-gs_ply-colmap",
    process_res=process_res,
)
for _mod, _o in _patched:                      # restore
    setattr(_mod, "align_poses_umeyama", _o)
print(f"process_res: {process_res}")
print(f"inference+export: {time.time()-t0:.1f}s")
print("vram peak:", torch.cuda.max_memory_allocated() // (1 << 20), "MiB")

# Build the gaussian→solve-world similarity. PRIMARY: Umeyama(DA3 predicted
# centers -> input centers). FALLBACK: analytic inverse of DA3's normalization
# (recenter-to-cam0 + median distance) if the capture missed.
Ci = _centers(ext)
transform = None
if "pred_ext" in _cap and len(_cap["pred_ext"]) == len(Ci) and len(Ci) >= 3:
    Cp = _centers(_cap["pred_ext"])                    # gaussian (normalized) frame
    s, R, t = _umeyama(Cp, Ci)                         # world ≈ s R Cp + t
    resid = float(np.sqrt(((Ci - (s * (R @ Cp.T).T + t)) ** 2).sum(1)).mean())
    spread = float(np.linalg.norm(Ci - Ci.mean(0), axis=1).mean()) or 1.0
    transform = {"scale": s, "quat": _R_to_quat_xyzw(R), "translate": t.tolist(),
                 "source": "da3-predicted-umeyama", "n": int(len(Cp)),
                 "residual_frac": round(resid / spread, 4)}
    print(f"[gauge] da3-umeyama: scale={s:.4f} resid={resid/spread*100:.1f}% of spread")
else:
    w0 = np.asarray(ext[0], dtype=np.float64); R0 = w0[:3, :3]; t0v = w0[:3, 3]
    C0 = -R0.T @ t0v
    c2w0 = np.eye(4); c2w0[:3, :3] = R0.T; c2w0[:3, 3] = C0
    dd = []
    for w in ext:
        w4 = np.eye(4); w4[:3, :3] = np.asarray(w)[:3, :3]; w4[:3, 3] = np.asarray(w)[:3, 3]
        dd.append(np.linalg.norm(np.linalg.inv(w4 @ c2w0)[:3, 3]))
    md = float(np.median(dd))
    transform = {"scale": md, "quat": _R_to_quat_xyzw(R0.T), "translate": C0.tolist(),
                 "source": "analytic-cam0-mediandist (capture MISSED — approximate)"}
    print(f"[gauge] FALLBACK analytic: scale(median_dist)={md:.4f} "
          f"(capture missed — verify/refine in match.html)")

scale_factor = getattr(pred, "scale_factor", None)
gauge = {
    "transform": transform,            # gs_ply -> solve world: p = scale * R(quat) * p_g + translate
    "scale_factor": float(scale_factor) if scale_factor is not None else None,
    "is_metric": bool(getattr(pred, "is_metric", False)),
    "note": ("gs_ply lives in DA3's normalized (cam-0-recentered + median-dist) "
             "frame. transform maps it to the solve/world gauge: "
             "p_world = scale * R(quat[xyzw]) * p_gs + translate. STO-SCN-105."),
}
with open(os.path.join(out, "scout_gauge.json"), "w") as fh:
    json.dump(gauge, fh, indent=2)
print(f"scout_gauge: transform={transform['source']} scale={transform['scale']:.4f}")
