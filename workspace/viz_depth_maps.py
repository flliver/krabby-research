"""Visualize MAtCha chart-data depth maps as PNGs.

Reads `<variant>/mast3r_sfm/charts_data.npz` and emits one PNG per (chart,
stage) into `<variant>/mast3r_sfm/depth_viz/`. Two stages:

  - prior_depths — DepthAnythingV2 monodepth output (the "before")
  - depths       — MAtCha-aligned depths (the "after")

Plus a side-by-side per-chart combo PNG with confidence overlay.

Per-image depth normalization (each frame's range maps to [0, 1]). Turbo
colormap. Optional --aligned-only flag to skip the prior renders if all
you want is the final result.

Run:
  python3 viz_depth_maps.py <variant-dir>            # any variant
  python3 viz_depth_maps.py 12-dense-strong          # short form

The short form resolves to data/scenes/004-sky-house-curated-<name>/.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

# Turbo lookup table (256 RGB triples). Source: matplotlib's turbo_colormap
# — tabulated here to avoid the matplotlib dep. Apache-2 licensed.
TURBO_LUT = np.array([
    [48, 18, 59], [50, 21, 67], [51, 24, 74], [52, 27, 81], [53, 30, 88],
    [54, 33, 95], [55, 36, 102], [56, 39, 109], [57, 42, 115], [58, 45, 121],
    [59, 47, 128], [60, 50, 134], [61, 53, 139], [62, 56, 145], [63, 59, 151],
    [63, 62, 156], [64, 64, 162], [65, 67, 167], [65, 70, 172], [66, 73, 177],
    [66, 75, 181], [67, 78, 186], [68, 81, 191], [68, 84, 195], [68, 86, 199],
    [69, 89, 203], [69, 92, 207], [69, 94, 211], [70, 97, 214], [70, 100, 218],
    [70, 102, 221], [70, 105, 224], [70, 107, 227], [71, 110, 230], [71, 113, 233],
    [71, 115, 235], [71, 118, 238], [71, 120, 240], [71, 123, 242], [70, 125, 244],
    [70, 128, 246], [70, 130, 248], [70, 133, 250], [70, 135, 251], [69, 138, 252],
    [69, 140, 253], [68, 143, 254], [67, 145, 254], [66, 148, 255], [65, 150, 255],
    [64, 153, 255], [62, 155, 254], [61, 158, 254], [59, 160, 253], [58, 163, 252],
    [56, 165, 251], [55, 168, 250], [53, 171, 248], [51, 173, 247], [49, 175, 245],
    [47, 178, 244], [46, 180, 242], [44, 183, 240], [42, 185, 238], [40, 188, 235],
    [39, 190, 233], [37, 192, 231], [35, 195, 228], [34, 197, 225], [32, 199, 223],
    [31, 201, 220], [30, 203, 218], [28, 205, 215], [27, 208, 212], [26, 210, 210],
    [26, 212, 207], [25, 213, 205], [24, 215, 202], [24, 217, 200], [24, 219, 197],
    [24, 221, 194], [24, 222, 192], [24, 224, 189], [25, 226, 187], [25, 227, 185],
    [26, 228, 182], [27, 230, 180], [28, 231, 178], [29, 232, 175], [31, 234, 172],
    [32, 235, 170], [34, 236, 167], [37, 237, 164], [39, 238, 161], [42, 239, 158],
    [45, 240, 155], [48, 240, 152], [51, 241, 149], [55, 241, 146], [59, 242, 143],
    [62, 242, 140], [66, 243, 137], [70, 243, 134], [74, 243, 132], [78, 243, 129],
    [82, 243, 127], [86, 243, 124], [90, 243, 121], [94, 243, 119], [98, 243, 116],
    [101, 243, 114], [105, 242, 111], [109, 242, 109], [113, 242, 107], [116, 241, 105],
    [120, 240, 103], [124, 239, 102], [128, 239, 100], [131, 238, 99], [135, 237, 98],
    [138, 236, 97], [142, 235, 95], [146, 233, 94], [149, 232, 93], [153, 231, 92],
    [156, 230, 90], [160, 228, 88], [163, 227, 87], [167, 225, 85], [170, 223, 83],
    [174, 222, 81], [177, 220, 79], [181, 218, 77], [184, 216, 76], [187, 214, 74],
    [191, 212, 72], [194, 210, 70], [197, 208, 68], [200, 205, 67], [203, 203, 65],
    [206, 200, 63], [209, 198, 61], [212, 195, 60], [215, 192, 58], [217, 189, 56],
    [220, 186, 55], [222, 183, 53], [224, 180, 51], [227, 177, 50], [229, 174, 48],
    [231, 170, 47], [233, 167, 45], [235, 164, 43], [237, 160, 42], [239, 157, 40],
    [240, 154, 38], [242, 150, 37], [243, 147, 35], [245, 143, 34], [246, 140, 32],
    [247, 136, 31], [249, 133, 29], [250, 130, 28], [251, 126, 27], [252, 122, 25],
    [252, 119, 24], [253, 115, 22], [253, 112, 21], [254, 108, 20], [254, 105, 18],
    [254, 102, 17], [254, 98, 16], [254, 95, 14], [254, 92, 13], [253, 88, 12],
    [253, 85, 11], [252, 82, 10], [251, 79, 9], [250, 76, 8], [249, 72, 8],
    [248, 69, 7], [247, 66, 6], [245, 63, 6], [244, 60, 5], [242, 58, 4],
    [240, 55, 4], [238, 52, 3], [236, 49, 3], [234, 47, 3], [232, 44, 2],
    [229, 41, 2], [227, 38, 2], [225, 36, 2], [222, 34, 1], [219, 31, 1],
    [216, 29, 1], [213, 27, 1], [210, 25, 1], [207, 23, 1], [204, 21, 0],
    [200, 19, 0], [197, 17, 0], [193, 15, 0], [189, 13, 0], [186, 12, 0],
    [182, 10, 0], [178, 9, 0], [174, 7, 0], [170, 6, 0], [166, 5, 0],
    [162, 4, 0], [158, 3, 0], [154, 2, 0], [150, 1, 0], [146, 1, 0],
    [142, 0, 0], [138, 0, 0], [134, 0, 0], [129, 0, 0], [125, 0, 0],
    [120, 0, 0], [116, 0, 0], [111, 0, 0], [106, 0, 0], [102, 0, 0],
    [97, 0, 0], [93, 0, 0], [88, 0, 0], [83, 0, 0], [78, 0, 0],
    [73, 0, 0], [69, 0, 0], [64, 0, 0], [59, 0, 0], [55, 0, 0],
    [50, 0, 0], [46, 0, 0], [42, 0, 0], [38, 0, 0], [33, 0, 0],
    [29, 0, 0], [25, 0, 0], [22, 0, 0], [18, 0, 0], [14, 0, 0],
    [10, 0, 0], [7, 0, 0], [3, 0, 0], [0, 0, 0],
], dtype=np.uint8)


def colorize(arr01: np.ndarray) -> np.ndarray:
    """Map a (H, W) array in [0, 1] to (H, W, 3) uint8 via Turbo."""
    idx = np.clip((arr01 * 255).astype(np.int32), 0, len(TURBO_LUT) - 1)
    return TURBO_LUT[idx]


def normalize_depth(d: np.ndarray, p_lo: float = 1.0, p_hi: float = 99.0) -> np.ndarray:
    """Per-image normalize to [0, 1] using percentile clipping (robust to outliers)."""
    finite = d[np.isfinite(d)]
    if finite.size == 0:
        return np.zeros_like(d, dtype=np.float32)
    lo = float(np.percentile(finite, p_lo))
    hi = float(np.percentile(finite, p_hi))
    if hi <= lo:
        return np.zeros_like(d, dtype=np.float32)
    out = (d - lo) / (hi - lo)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def stitch_horizontal(*tiles: np.ndarray, gap_px: int = 4, gap_color=(40, 40, 40)) -> np.ndarray:
    """Concatenate (H, W, 3) tiles side-by-side with a thin separator."""
    h = max(t.shape[0] for t in tiles)
    pieces = []
    for i, t in enumerate(tiles):
        if i > 0:
            sep = np.full((h, gap_px, 3), gap_color, dtype=np.uint8)
            pieces.append(sep)
        pieces.append(t)
    return np.concatenate(pieces, axis=1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("variant", help="path to variant dir, or short suffix (e.g. '12-dense-strong')")
    p.add_argument("--aligned-only", action="store_true",
                   help="Skip prior_depths renders; only emit the aligned (post-MAtCha) depth")
    p.add_argument("--no-combo", action="store_true",
                   help="Skip the combo (prior | aligned | confidence) horizontal stitch")
    args = p.parse_args()

    # Resolve short form
    SCENES_ROOT = Path(__file__).resolve().parent.parent / "data" / "scenes"
    if "/" not in args.variant:
        variant_dir = SCENES_ROOT / f"004-sky-house-curated-{args.variant}"
    else:
        variant_dir = Path(args.variant).resolve()

    npz_path = variant_dir / "mast3r_sfm" / "charts_data.npz"
    if not npz_path.is_file():
        sys.exit(
            f"ERROR: {npz_path} not found.\n"
            f"  Pull from tbeeprz:\n"
            f"    rsync tbeeprz:/home/jeremy/outposts/krabby/data/011-scene-reconstruction"
            f"/scenes/{variant_dir.name}/mast3r_sfm/charts_data.npz\\\n"
            f"      {npz_path}"
        )

    out_dir = variant_dir / "mast3r_sfm" / "depth_viz"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Reading: {npz_path}")
    print(f"Writing: {out_dir}/")

    d = np.load(npz_path)
    prior = d["prior_depths"] if "prior_depths" in d.files else None
    aligned = d["depths"] if "depths" in d.files else None
    confs = d["confs"] if "confs" in d.files else None
    scale = float(d["scale_factor"]) if "scale_factor" in d.files else None

    print(f"  prior_depths: {prior.shape if prior is not None else '(missing)'}")
    print(f"  depths:       {aligned.shape if aligned is not None else '(missing)'}")
    print(f"  confs:        {confs.shape if confs is not None else '(missing)'}")
    print(f"  scale_factor: {scale}")

    if aligned is None:
        sys.exit("ERROR: 'depths' array missing from npz")
    n = aligned.shape[0]

    n_emitted = 0
    for i in range(n):
        a_norm = normalize_depth(aligned[i])
        a_rgb = colorize(a_norm)
        Image.fromarray(a_rgb).save(out_dir / f"cam_{i+1:03d}-aligned.png")
        n_emitted += 1

        if not args.aligned_only and prior is not None:
            p_norm = normalize_depth(prior[i])
            p_rgb = colorize(p_norm)
            Image.fromarray(p_rgb).save(out_dir / f"cam_{i+1:03d}-prior.png")
            n_emitted += 1

        if not args.no_combo and prior is not None:
            tiles = [colorize(normalize_depth(prior[i])), a_rgb]
            if confs is not None:
                # Confidence is unbounded; clip to a reasonable visual range.
                c_norm = normalize_depth(confs[i], p_lo=2.0, p_hi=98.0)
                tiles.append(colorize(c_norm))
            combo = stitch_horizontal(*tiles)
            Image.fromarray(combo).save(out_dir / f"cam_{i+1:03d}-combo.png")
            n_emitted += 1

    print(f"\nWrote {n_emitted} PNG(s) to {out_dir}/")
    print(f"Open one to inspect:")
    print(f"  open {out_dir}/cam_001-combo.png")


if __name__ == "__main__":
    main()
