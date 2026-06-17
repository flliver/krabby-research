"""STO-SCN-151 — scout verify-surface build/serve orchestration (numpy-free).

The rate_renders server runs under a numpy-LESS python; the verify tools
(`build_verify.py`, datum calibration) need numpy. This module is the seam: it
discovers a numpy-capable python and shells the heavy work out to it, so the
server process stays light. It builds the verify serve dir (scout gaussian +
frustums + de-warped frames + viewer.html/match.html) for a scene and lists /
authors render views.

Reuses `verify_viewer/build_verify.py` (the STO-SCN-095/105 surface) and the
`views/<slot>/view.json` convention (`author_overview_view.py`). No new recon.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCENES_ROOT = Path(os.environ.get("KRABBY_SCENES_ROOT", "/var/krabby/scenes"))


def numpy_python() -> str | None:
    """Path to a python that can import numpy. Env KRABBY_NUMPY_PYTHON wins;
    else this interpreter if it has numpy; else scan common homebrew pythons."""
    env = os.environ.get("KRABBY_NUMPY_PYTHON")
    cands = [env] if env else []
    # The canonical recon venv (real2sim/.venv: py3.11 + numpy + open3d). Prefer it
    # over the bare interpreters so the verify/orient tools have the FULL stack and
    # don't fall back to a numpy-only python (the open3d gap we kept hitting).
    cands.append(str(HERE / ".venv" / "bin" / "python"))
    cands.append(sys.executable)
    cands += ["/opt/homebrew/bin/python3.10", "/opt/homebrew/bin/python3.11",
              "/opt/homebrew/bin/python3.12", "python3.10", "python3.11", "python3"]
    for py in cands:
        if not py:
            continue
        try:
            r = subprocess.run([py, "-c", "import numpy"], capture_output=True, timeout=15)
            if r.returncode == 0:
                return py
        except (OSError, subprocess.SubprocessError):
            continue
    return None


def resolve_scout(scene_dir: Path):
    """Newest (subset, solve, scout) that has a scout.gs.ply, or None.

    Prefers the solve with the most cameras (cameras.json present), then its
    newest scout dir holding scout.gs.ply.
    """
    subs = scene_dir / "images" / "subsets"
    if not subs.is_dir():
        return None
    best = None
    for sd in subs.iterdir():
        if sd.is_symlink() or not sd.is_dir():
            continue
        cams = sd / "cameras"
        if not cams.is_dir():
            continue
        for solve in cams.iterdir():
            sdir = solve / "scout"
            if not sdir.is_dir():
                continue
            scouts = [s for s in sdir.iterdir() if (s / "scout.gs.ply").exists()]
            if not scouts:
                continue
            scout = max(scouts, key=lambda s: s.stat().st_mtime)
            # rank by camera count (members) so the fullest spine wins
            members = 0
            sj = sd / "subset.json"
            if sj.exists():
                try:
                    members = len(json.loads(sj.read_text()).get("members", []))
                except (OSError, ValueError):
                    members = 0
            cand = (members, sd.name, solve.name, scout.name)
            if best is None or cand[0] > best[0]:
                best = cand
    if not best:
        return None
    return {"subset": best[1], "solve": best[2], "scout": best[3]}


def serve_dir(scene_dir: Path) -> Path:
    return scene_dir / "verify-serve"


def build_serve(scene_dir: Path, *, subset=None, solve=None, scout=None,
                progress=None) -> dict:
    """Run build_verify.py (under a numpy python) → scene_dir/verify-serve/.
    Auto-resolves subset/solve/scout if not given. Returns {ok, ...} or {error}."""
    py = numpy_python()
    if not py:
        return {"error": "no numpy-capable python found (set KRABBY_NUMPY_PYTHON)"}
    if not (subset and solve and scout):
        r = resolve_scout(scene_dir)
        if not r:
            return {"error": "no scout gaussian in this scene — run the pipeline (scout phase) first"}
        subset, solve, scout = r["subset"], r["solve"], r["scout"]
    out = serve_dir(scene_dir)
    cmd = [py, str(HERE / "verify_viewer" / "build_verify.py"), scene_dir.name,
           "--subset", subset, "--solve", solve, "--scout", scout,
           "--no-serve", "--serve-dir", str(out)]
    if progress:
        progress("building verify surface (scout gaussian + frustums)…")
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, cwd=str(HERE), timeout=900)
    except subprocess.SubprocessError as e:
        return {"error": f"build_verify failed to launch: {e}"}
    if r.returncode != 0:
        tail = "\n".join((r.stdout + r.stderr).splitlines()[-25:])
        return {"error": "build_verify failed", "log": tail}
    has = (out / "viewer.html").exists() and (out / "scout.gs.ply").exists()
    return {"ok": has, "subset": subset, "solve": solve, "scout": scout,
            "serve_dir": str(out), "log": "\n".join(r.stdout.splitlines()[-8:])}


def list_views(scene_dir: Path) -> list:
    """Named render views = views/<slot>/view.json."""
    vdir = scene_dir / "views"
    if not vdir.is_dir():
        return []
    out = []
    for d in sorted(vdir.iterdir()):
        vj = d / "view.json"
        if vj.is_dir() or not vj.exists():
            continue
        try:
            v = json.loads(vj.read_text())
        except (OSError, ValueError):
            v = {}
        out.append({"name": d.name, "pose": v.get("P") or v.get("position")})
    return out


def author_overview(scene_dir: Path) -> dict:
    """Author the standard pulled-back 'overview' render view (reuse
    author_overview_view.py) from the scene's solve cameras + orientation."""
    py = numpy_python()
    if not py:
        return {"error": "no numpy-capable python found"}
    r = resolve_scout(scene_dir)
    if not r:
        return {"error": "no solve/scout to derive an overview from"}
    cam = scene_dir / "images" / "subsets" / r["subset"] / "cameras" / r["solve"]
    cameras_json = cam / "cameras.json"
    orient = sorted((cam / "orient").glob("*/oriented.json")) if (cam / "orient").is_dir() else []
    if not cameras_json.exists() or not orient:
        return {"error": "missing cameras.json / orient/oriented.json for the solve"}
    cmd = [py, str(HERE / "author_overview_view.py"), scene_dir.name,
           str(cameras_json), str(orient[0])]
    try:
        rr = subprocess.run(cmd, capture_output=True, text=True, cwd=str(HERE), timeout=120)
    except subprocess.SubprocessError as e:
        return {"error": f"author_overview failed: {e}"}
    if rr.returncode != 0:
        return {"error": "author_overview failed", "log": (rr.stdout + rr.stderr)[-400:]}
    return {"ok": True, "views": list_views(scene_dir)}


def normalize(scene_dir: Path, export: dict, *, subset=None, solve=None,
              force=False, dry=False, gate_thresh=1.5) -> dict:
    """STO-SCN-152: run the full metric normalize (normalize_datum.py under a
    numpy python): recompute → build_datum → apply_to_gauge → datum.json."""
    import tempfile
    py = numpy_python()
    if not py:
        return {"error": "no numpy-capable python found (set KRABBY_NUMPY_PYTHON)"}
    if not (subset and solve):
        r = resolve_scout(scene_dir)
        if not r:
            return {"error": "no solve to normalize — run the pipeline first"}
        subset, solve = r["subset"], r["solve"]
    fd, path = tempfile.mkstemp(suffix=".json")
    exp = Path(path)
    try:
        os.close(fd)
        exp.write_text(json.dumps(export))
        cmd = [py, str(HERE / "normalize_datum.py"), scene_dir.name,
               "--subset", subset, "--solve", solve, "--export", str(exp),
               "--store", str(SCENES_ROOT), "--gate-thresh", str(gate_thresh)]
        if force:
            cmd.append("--force")
        if dry:
            cmd.append("--dry")
        r = subprocess.run(cmd, capture_output=True, text=True, cwd=str(HERE), timeout=180)
    except subprocess.SubprocessError as e:
        return {"error": f"normalize failed to launch: {e}"}
    finally:
        exp.unlink(missing_ok=True)
    try:
        return json.loads(r.stdout)
    except (ValueError, json.JSONDecodeError):
        return {"error": "normalize failed", "log": (r.stdout + r.stderr)[-500:]}
