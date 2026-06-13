#!/usr/bin/env python3
"""select_sharp_frames.py — preproc transform: blur-aware frame selection.

Spec-driven, results-emitting (HUG-KRB-002): reads a preproc transform's
`specification.json`, scores every frame in the source pool, selects the
sharpest frame per uniform temporal window, copies the winners into
`data/` (original filenames preserved) and writes a *measured*
`results.json`.

Method (hardens the 001-patio preproc-02 prototype, STO-SCN-045):
  - grayscale, downscale so long edge == parameters.score_edge (480)
  - variance of the 3x3 Laplacian (sharpness score)
  - split the sorted pool into parameters.count uniform windows;
    take the argmax of each window
Windowed-sharpest supersedes uniform stride: uniform stride handed 001's
SfM a degenerate (motion-blurred) frame and produced NaN confidences.

Usage:
    python3 select_sharp_frames.py <store>/<scene>/input/preproc-NN-<slug>

Spec contract (kind):       "preprocessing"
Spec contract (inputs):     [ "input/src" ]   (scene-relative, exactly one dir)
Spec contract (parameters): { "count": 24, "score_edge": 480,
                              "selection_method": "windowed-sharpest-laplacian" }

Selected indices + per-frame scores are emitted in results.json
(measured output — they do NOT belong in the spec, which is the input
contract; 001's prototype baked selected_idx into the spec post-hoc,
this script corrects that).

CPU-only; runs anywhere with PIL + numpy.
"""
from __future__ import annotations

import datetime
import json
import shutil
import socket
import sys
from pathlib import Path

import numpy as np
from PIL import Image

IMAGE_EXTS = {".png", ".jpg", ".jpeg"}


def sharpness_of_gray(gray: "Image.Image", score_edge: int) -> float:
    """Variance of the 3x3 Laplacian on a grayscale PIL image (downscaled to
    score_edge). Split out so callers that already hold an open image can score
    without a second decode (STO-SCN-092 precull)."""
    g = gray.copy()
    g.thumbnail((score_edge, score_edge), Image.BILINEAR)
    a = np.asarray(g, dtype=np.float64)
    lap = (
        a[:-2, 1:-1] + a[2:, 1:-1] + a[1:-1, :-2] + a[1:-1, 2:]
        - 4.0 * a[1:-1, 1:-1]
    )
    return float(lap.var())


def sharpness(path: Path, score_edge: int) -> float:
    """Variance of the 3x3 Laplacian on a grayscale thumbnail."""
    with Image.open(path) as im:
        return sharpness_of_gray(im.convert("L"), score_edge)


def main() -> int:
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    tdir = Path(sys.argv[1]).resolve()
    spec = json.loads((tdir / "specification.json").read_text())
    if spec.get("kind") != "preprocessing":
        sys.exit(f"ERROR: spec kind is {spec.get('kind')!r}, expected 'preprocessing'")
    params = spec.get("parameters", {})
    count = int(params["count"])
    score_edge = int(params.get("score_edge", 480))
    inputs = spec.get("inputs", [])
    if len(inputs) != 1:
        sys.exit(f"ERROR: expected exactly one input dir, got {inputs!r}")

    scene_dir = tdir.parent.parent  # <scene>/input/preproc-NN-<slug> -> <scene>
    src = (scene_dir / inputs[0]).resolve()
    if not src.is_dir():
        sys.exit(f"ERROR: input dir not found: {src}")

    pool = sorted(p for p in src.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    n = len(pool)
    if n < count:
        sys.exit(f"ERROR: pool has {n} frames, fewer than count={count}")

    print(f"Scoring {n} frames (Laplacian variance @ {score_edge}px) ...")
    scores = np.array([sharpness(p, score_edge) for p in pool])

    # count uniform temporal windows; argmax per window
    bounds = np.linspace(0, n, count + 1).astype(int)
    selected = []
    for w in range(count):
        lo, hi = bounds[w], bounds[w + 1]
        idx = lo + int(np.argmax(scores[lo:hi]))
        selected.append(idx)

    data = tdir / "data"
    data.mkdir(exist_ok=True)
    outputs = []
    for idx in selected:
        dst = data / pool[idx].name
        shutil.copy2(pool[idx], dst)
        outputs.append(f"data/{pool[idx].name}")

    results = {
        "schema_version": "1",
        "transform": spec.get("transform", tdir.name),
        "status": "success",
        "provenance": "measured",
        "finished": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "host": socket.gethostname(),
        "outputs": outputs,
        "metrics": {
            "source_pool_n": n,
            "selected": [
                {"idx": int(i), "file": pool[i].name, "score": round(float(scores[i]), 2)}
                for i in selected
            ],
            "pool_score_min": round(float(scores.min()), 2),
            "pool_score_median": round(float(np.median(scores)), 2),
            "pool_score_max": round(float(scores.max()), 2),
        },
    }
    (tdir / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    print(f"Selected {count}/{n}: indices {selected}")
    print(f"Wrote {tdir / 'results.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
