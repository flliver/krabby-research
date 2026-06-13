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
print(f"process_res: {process_res}")
print(f"inference+export: {time.time()-t0:.1f}s")
print("vram peak:", torch.cuda.max_memory_allocated() // (1 << 20), "MiB")
