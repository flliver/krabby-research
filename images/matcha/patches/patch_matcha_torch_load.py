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


def patch_text(txt: str) -> tuple[str, int]:
    """Insert weights_only=False before the MATCHING close paren of each
    torch.load(...) call.

    STO-SCN-038 root-cause note: the original regex (`[^)]*?`) stopped at
    the FIRST `)`, so nested calls — torch.load(os.path.join(a, b), ...) —
    got the kwarg inserted inside the inner call, producing a runtime
    TypeError. That broke matcha/pointmap/depthanythingv2.py and was
    hand-fixed on the production host, creating the snapshot drift this
    story exists to eliminate. This version walks parens to the real
    closing one.
    """
    out = []
    i = 0
    n = 0
    needle = "torch.load("
    while True:
        j = txt.find(needle, i)
        if j == -1:
            out.append(txt[i:])
            break
        start = j + len(needle)
        depth = 1
        k = start
        while k < len(txt) and depth > 0:
            c = txt[k]
            if c == "(":
                depth += 1
            elif c == ")":
                depth -= 1
            k += 1
        # k is now just past the matching close paren
        span = txt[start:k - 1]
        if "weights_only" in span:
            out.append(txt[i:k])
        else:
            out.append(txt[i:k - 1])
            out.append(", weights_only=False)")
            n += 1
        i = k
    return "".join(out), n


total = 0
for p in files:
    txt = p.read_text()
    new_txt, n = patch_text(txt)
    if n > 0:
        p.write_text(new_txt)
        total += n
        print(f"  patched +{n}: {p}")

print(f"Total replacements: {total}")
