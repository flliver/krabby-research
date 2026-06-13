"""colmap_posed.py — mint a COLMAP sparse/0 from a store ingest solve.

Feeds the ingest-solve cameras INTO the matcha weld (`--sfm_config posed`)
so train.py stops re-solving and minting its own arbitrary gauge
(STO-SCN-089-3 class; 003-firepit is the forcing scene: its re-solve
disagrees with the ingest solve beyond any similarity, 3.1-3.9%).

MAtCha's posed loader (mast3r/run_mast3r.py) expects:
  scene_path/images/                  original-resolution images
  scene_path/sparse/0/cameras.bin     PINHOLE intrinsics, ORIGINAL pixel space
  scene_path/sparse/0/images.bin      w2c extrinsics keyed by image NAME
It re-centers principal points and rescales intrinsics itself, and with
the posed config (fix_focal/pp/rotation/translation + align_camera_locations)
the output gauge IS the input gauge.

The store solve's cameras.json holds focals in mast3r-512 space (long side
resized to 512); originals are staged at full resolution, so the focal is
rescaled by max(W, H) / 512.

No third-party deps: image dims come from a minimal JPEG/PNG header parse.
"""
from __future__ import annotations

import json
import struct
from pathlib import Path

SOLVE_IMAGE_SIZE = 512          # mast3r load_images(size=512): long side
COLMAP_PINHOLE = 1              # model_id; params = fx, fy, cx, cy


def image_dims(path: Path) -> tuple[int, int]:
    """(width, height) from JPEG SOF or PNG IHDR. Pure python, no deps."""
    data = path.read_bytes()
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        w, h = struct.unpack(">II", data[16:24])
        return int(w), int(h)
    if data[:2] == b"\xff\xd8":  # JPEG
        i = 2
        while i < len(data) - 9:
            if data[i] != 0xFF:
                i += 1
                continue
            marker = data[i + 1]
            if marker in (0xD8, 0x01) or 0xD0 <= marker <= 0xD7:
                i += 2
                continue
            seglen = struct.unpack(">H", data[i + 2:i + 4])[0]
            if 0xC0 <= marker <= 0xCF and marker not in (0xC4, 0xC8, 0xCC):  # SOFn
                h, w = struct.unpack(">HH", data[i + 5:i + 9])
                return int(w), int(h)
            i += 2 + seglen
        raise ValueError(f"no SOF marker in {path}")
    raise ValueError(f"unsupported image format: {path}")


def rotmat_to_qvec(R) -> list[float]:
    """3x3 -> COLMAP qvec (w, x, y, z). Shepperd's method, numpy-only."""
    import numpy as np
    R = np.asarray(R, dtype=float)
    K = np.array([
        [R[0, 0] - R[1, 1] - R[2, 2], 0, 0, 0],
        [R[0, 1] + R[1, 0], R[1, 1] - R[0, 0] - R[2, 2], 0, 0],
        [R[0, 2] + R[2, 0], R[1, 2] + R[2, 1], R[2, 2] - R[0, 0] - R[1, 1], 0],
        [R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1],
         R[0, 0] + R[1, 1] + R[2, 2]]]) / 3.0
    w, V = np.linalg.eigh(K)
    q = V[[3, 0, 1, 2], np.argmax(w)]
    if q[0] < 0:
        q = -q
    return [float(v) for v in q]


def write_sparse(sparse_dir: Path, entries: list[dict]) -> None:
    """entries: [{name, width, height, fx, fy, cx, cy, w2c (4x4 list)}].
    Writes cameras.bin / images.bin / points3D.bin (COLMAP binary format)."""
    import numpy as np
    sparse_dir.mkdir(parents=True, exist_ok=True)
    with open(sparse_dir / "cameras.bin", "wb") as f:
        f.write(struct.pack("<Q", len(entries)))
        for i, e in enumerate(entries, start=1):
            f.write(struct.pack("<iiQQ", i, COLMAP_PINHOLE,
                                e["width"], e["height"]))
            f.write(struct.pack("<dddd", e["fx"], e["fy"], e["cx"], e["cy"]))
    with open(sparse_dir / "images.bin", "wb") as f:
        f.write(struct.pack("<Q", len(entries)))
        for i, e in enumerate(entries, start=1):
            w2c = np.asarray(e["w2c"], dtype=float)
            q = rotmat_to_qvec(w2c[:3, :3])
            t = w2c[:3, 3]
            f.write(struct.pack("<i", i))
            f.write(struct.pack("<dddd", *q))
            f.write(struct.pack("<ddd", *[float(v) for v in t]))
            f.write(struct.pack("<i", i))                       # camera_id
            f.write(e["name"].encode() + b"\x00")
            f.write(struct.pack("<Q", 0))                       # num_points2D
    with open(sparse_dir / "points3D.bin", "wb") as f:
        f.write(struct.pack("<Q", 0))


def solve_entries(solve_cameras_json: Path, staged: dict[str, Path]) -> list[dict]:
    """Posed-camera entries for staged images, from a store ingest solve.

    `staged` maps staged-image NAME (as it appears on the host, e.g.
    frame_00.jpg) -> local path to read dims from (the store's image.*).
    Cameras are matched by filename STEM (extension drift between runs
    must not break identity — the 007 lesson). Focal converts from solve
    space (long side = SOLVE_IMAGE_SIZE) to original pixels; principal
    point is the image center (the solve's own convention).
    Refuses on staged images the solve doesn't cover.
    """
    import numpy as np
    cams = json.loads(solve_cameras_json.read_text())
    by_stem = {}
    for fp, f, m in zip(cams["filepaths"], cams["focals"], cams["cams2world"]):
        by_stem[fp.rsplit("/", 1)[-1].rsplit(".", 1)[0]] = (float(f), m)
    missing = [n for n in staged if n.rsplit(".", 1)[0] not in by_stem]
    if missing:
        raise ValueError(f"posed REFUSED: staged images without solve "
                         f"cameras: {sorted(missing)}")
    entries = []
    for name in sorted(staged):
        f_solve, c2w = by_stem[name.rsplit(".", 1)[0]]
        w, h = image_dims(staged[name])
        f_orig = f_solve * max(w, h) / SOLVE_IMAGE_SIZE
        entries.append({"name": name, "width": w, "height": h,
                        "fx": f_orig, "fy": f_orig,
                        "cx": w / 2.0, "cy": h / 2.0,
                        "w2c": np.linalg.inv(np.asarray(c2w, dtype=float)).tolist()})
    return entries


def solve_to_sparse(solve_cameras_json: Path, staged: dict[str, Path],
                    sparse_dir: Path) -> list[str]:
    """Mint COLMAP sparse/0 from a store ingest solve (matcha@1 posed weld)."""
    entries = solve_entries(solve_cameras_json, staged)
    write_sparse(sparse_dir, entries)
    return [e["name"] for e in entries]


def solve_to_posed_json(solve_cameras_json: Path, staged: dict[str, Path],
                        out_json: Path) -> list[str]:
    """Mint posed.json for da3@1 pose-conditioned inference: per staged
    image name, w2c (4x4) + K (3x3, original pixel space)."""
    entries = solve_entries(solve_cameras_json, staged)
    posed = [{"name": e["name"], "w2c": e["w2c"],
              "K": [[e["fx"], 0.0, e["cx"]],
                    [0.0, e["fy"], e["cy"]],
                    [0.0, 0.0, 1.0]]} for e in entries]
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(posed, indent=2) + "\n")
    return [e["name"] for e in entries]
