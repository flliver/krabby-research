"""STO-SCN-060: DA3 inference with the gaussian branch enabled.
The auto CLI does not expose infer_gs; this driver calls the API."""
import glob, sys, time
import torch
from depth_anything_3.api import DepthAnything3

src, out = sys.argv[1], sys.argv[2]
process_res = int(sys.argv[3]) if len(sys.argv) > 3 else 504
no_gs = len(sys.argv) > 4 and sys.argv[4] == "nogs"  # depths-only (mesh path); gs head OOMs 16GB above 504
images = sorted(glob.glob(f"{src}/*.jpeg") + glob.glob(f"{src}/*.jpg") + glob.glob(f"{src}/*.png"))
print(f"{len(images)} images")
model = DepthAnything3.from_pretrained("/opt/checkpoints/DA3NESTED-GIANT-LARGE-1.1")
model = model.to("cuda")
t0 = time.time()
pred = model.inference(
    images,
    infer_gs=not no_gs,
    export_dir=out,
    export_format="npz-colmap-glb" if no_gs else "glb-npz-gs_ply-colmap",
    process_res=process_res,
)
print(f"process_res: {process_res}")
print(f"inference+export: {time.time()-t0:.1f}s")
print("vram peak:", torch.cuda.max_memory_allocated() // (1<<20), "MiB")
