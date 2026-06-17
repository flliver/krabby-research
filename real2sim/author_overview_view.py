"""Author a pulled-back 'overview' render view for ANY scene so a far-field cull is
visible (the canonical close-in views sit inside the geometry). Computes a high 3/4
look-at pose from the scene's posed-camera extent (gravity-aligned gauge), same opencv
convention as the existing views. Writes views/overview/view.json and adds the slot to
the canonical viewset.

Usage: author_overview_view.py <scene> <cameras.json> <oriented.json>
"""
import json
import os
import sys
from pathlib import Path

# cull_mesh imports open3d — self-bootstrap the recon env via uv exactly like
# v4exec / build_verify, so this runs through system tooling (STO-SCN-151).
try:
    import numpy  # noqa: F401
    import open3d  # noqa: F401
except ImportError:
    if os.environ.get("AUTHOR_OVERVIEW_BOOTSTRAPPED") != "1":
        os.environ["AUTHOR_OVERVIEW_BOOTSTRAPPED"] = "1"
        os.execvp("uv", ["uv", "run", "--quiet", "--python", "3.11",
                         "--with", "open3d", "--with", "numpy",
                         "python3", str(Path(__file__).resolve())] + sys.argv[1:])
    raise

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import cull_mesh  # for load_oriented_cameras (oriented camera centers)

scene, cameras_json, oriented_json = sys.argv[1], sys.argv[2], sys.argv[3]
SCENE = Path("/var/krabby/scenes") / scene

c2w, _, _ = cull_mesh.load_oriented_cameras(cameras_json, oriented_json)
C = c2w[:, :3, 3]
mn, mx = C.min(0), C.max(0)
center = (mn + mx) / 2.0
span = mx - mn
horiz = float(max(span[0], span[1]))

# look at the scene center near floor; pull back along -y and elevate, scaled by extent
T = np.array([center[0], center[1], (mn[2] + center[2]) / 2.0])
P = np.array([center[0], center[1] - 1.6 * horiz, center[2] + 1.3 * horiz])
UP = np.array([0.0, 0.0, 1.0])

f = T - P; f = f / np.linalg.norm(f)              # opencv +z forward
r = np.cross(f, UP); r = r / np.linalg.norm(r)    # opencv +x right
d = np.cross(f, r)                                 # opencv +y down
R = np.column_stack([r, d, f])


def mat_to_quat_wxyz(m):
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2; w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s; y = (m[0, 2] - m[2, 0]) / s; z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2; w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s; y = (m[0, 1] + m[1, 0]) / s; z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2; w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s; y = 0.25 * s; z = (m[1, 2] + m[2, 1]) / s
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2; w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s; y = (m[1, 2] + m[2, 1]) / s; z = 0.25 * s
    q = np.array([w, x, y, z]); return (q / np.linalg.norm(q)).tolist()


quat = mat_to_quat_wxyz(R)
view = {
    "auto_localized": False, "captured_camera_name": "overview_auto",
    "convention": "opencv", "lens_mm": 22.0,
    "localization_method": "computed-lookat", "purpose": "cull-overview",
    "render_engine": "BLENDER_WORKBENCH", "render_resolution": [1920, 1080],
    "sensor_height_mm": 24.0, "sensor_width_mm": 36.0,
    "world_position": P.tolist(), "world_rotation_quat_wxyz": quat,
}
outdir = SCENE / "views" / "overview"
outdir.mkdir(parents=True, exist_ok=True)
(outdir / "view.json").write_text(json.dumps(view, indent=2) + "\n")

vs = SCENE / "viewset" / "canonical" / "views.json"
# Create the canonical viewset if absent — a freshly-imported scene (or one that
# was nuked back to its source) has no viewset until the first view is authored.
d_vs = json.loads(vs.read_text()) if vs.exists() else {"slots": []}
if "overview" not in d_vs.get("slots", []):
    d_vs.setdefault("slots", []).append("overview")
    vs.parent.mkdir(parents=True, exist_ok=True)
    vs.write_text(json.dumps(d_vs, indent=2) + "\n")

print(f"wrote {outdir/'view.json'}; P={np.round(P,2).tolist()} T={np.round(T,2).tolist()}")
print(f"canonical slots -> {d_vs['slots']}")
