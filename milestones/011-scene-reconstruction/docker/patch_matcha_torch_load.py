"""Patch torch.load calls across the MAtCha tree for PyTorch 2.6+.

PyTorch 2.6 changed the default of torch.load from weights_only=False to
weights_only=True for security. Existing checkpoints (MASt3R, DUST3R,
DepthAnythingV2) contain argparse.Namespace and other non-tensor objects
which are rejected by the new default.

Fix: walk the tree and add weights_only=False to every torch.load call.

Same approach as patch_torch_load.py (MASt3R-SLAM); MAtCha has more
checkpoint loaders (~41 sites vs ~12) because it pulls in DUST3R +
MASt3R-SfM + DepthAnythingV2 + 2D-Gaussian-Splatting all in one project.
"""
import pathlib
import re

roots = [pathlib.Path("/opt/MAtCha")]
files = []
for root in roots:
    for p in root.rglob("*.py"):
        try:
            if "torch.load" in p.read_text():
                files.append(p)
        except Exception:
            pass

print(f"Found {len(files)} Python files containing torch.load")

total = 0
for p in files:
    txt = p.read_text()
    # Conservative regex: match torch.load(...) calls without weights_only=
    new_txt = re.sub(
        r"torch\.load\(([^)]*?)\)",
        lambda m: m.group(0) if "weights_only" in m.group(1) else f"torch.load({m.group(1)}, weights_only=False)",
        txt,
    )
    diff = new_txt.count("weights_only=False") - txt.count("weights_only=False")
    if diff > 0:
        p.write_text(new_txt)
        total += diff
        print(f"  patched +{diff}: {p}")

print(f"Total replacements: {total}")
