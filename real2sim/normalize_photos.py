#!/usr/bin/env python3
"""normalize_photos.py — preproc transform: photo normalization (STO-SCN-040 family).

Spec-driven, results-emitting (HUG-KRB-002): reads a preproc transform's
`specification.json`, normalizes the source photos, writes `data/` + a
*measured* `results.json`.

What "normalize" means here:
  - decode the PRIMARY image (drops MPO multi-picture payloads / aux depth)
  - apply EXIF orientation, then strip the rotation tag
  - downscale so the long edge == parameters.long_edge (no-op if smaller)
  - re-encode as plain baseline JPEG (parameters.quality, default 95)

Usage (from the scene dir's repo root or anywhere):
    python3 normalize_photos.py <store>/<scene>/input/preproc-NN-<slug>

Spec contract (parameters): { "long_edge": 2048, "quality": 95 }
Spec contract (inputs):     [ "input/src" ]  (scene-relative, exactly one)

CPU-only; runs anywhere. Registry integration into run_transform.py is
STO-SCN-040 scope (local-executor mode); until then this script honors the
same spec/results contract the runner emits.
"""
from __future__ import annotations
import datetime, hashlib, json, platform, socket, sys, time
from pathlib import Path

from PIL import Image, ImageOps


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def main() -> int:
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    tdir = Path(sys.argv[1]).resolve()
    spec = json.loads((tdir / "specification.json").read_text())
    if spec.get("kind") != "preprocessing":
        sys.exit(f"ERROR: spec kind is {spec.get('kind')!r}, expected 'preprocessing'")
    params = spec.get("parameters", {})
    long_edge = int(params.get("long_edge", 2048))
    quality = int(params.get("quality", 95))
    inputs = spec.get("inputs", [])
    if len(inputs) != 1:
        sys.exit(f"ERROR: expected exactly 1 input, got {inputs!r}")
    scene_dir = tdir.parent.parent  # <scene>/input/preproc-NN -> <scene>
    src = scene_dir / inputs[0]
    if not src.is_dir():
        sys.exit(f"ERROR: input dir missing: {src}")

    data = tdir / "data"
    results_path = tdir / "results.json"
    if results_path.exists():
        sys.exit(f"ERROR: {results_path} exists — transforms are immutable; make a new preproc dir.")
    data.mkdir(exist_ok=True)

    started = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
    t0 = time.time()
    outputs, converted = [], []
    for f in sorted(src.iterdir()):
        if f.suffix.lower() not in (".jpg", ".jpeg", ".png", ".heic"):
            continue
        im = Image.open(f)
        fmt_in, size_in = im.format, im.size
        im = ImageOps.exif_transpose(im)  # bake orientation
        if max(im.size) > long_edge:
            r = long_edge / max(im.size)
            im = im.resize((round(im.size[0] * r), round(im.size[1] * r)),
                           Image.Resampling.LANCZOS)
        if im.mode != "RGB":
            im = im.convert("RGB")
        out = data / (f.stem + ".jpg")
        im.save(out, "JPEG", quality=quality)
        outputs.append({"path": f"data/{out.name}", "bytes": out.stat().st_size,
                        "sha256": sha256(out)})
        converted.append(f"{f.name}: {fmt_in} {size_in[0]}x{size_in[1]} -> "
                          f"JPEG {im.size[0]}x{im.size[1]}")
    results = {
        "schema_version": "1", "transform": spec["transform"],
        "status": "success" if outputs else "failed",
        "provenance": "measured", "started": started,
        "finished": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "duration_s": round(time.time() - t0),
        "host": socket.gethostname(),
        "environment": {"os": platform.platform(), "gpu": None, "nvidia_driver": None,
                         "cuda": None, "container": None,
                         "software": {"pillow": Image.__version__ if hasattr(Image, "__version__")
                                      else __import__("PIL").__version__,
                                      "tool": "real2sim/normalize_photos.py"}},
        "outputs": outputs,
    }
    results_path.write_text(json.dumps(results, indent=2) + "\n")
    print("\n".join(converted))
    print(f"results.json written: {len(outputs)} photos normalized in {results['duration_s']}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
