"""STO-SCN-150 — ingest-scene pipeline orchestration (EPI-SCN-SCENE-MANAGER).

Drives a scene from its raw capture to a scouted + meshed state:

    ingest(video → frames + canonicalize) → precull(--set-primary) → solve
      → covis → scout(DA3 gaussian) → reconstruct-da3(DA3 mesh)

Phase 0 (``ingest``) is LOCAL: if the scene has a captured video and an EMPTY
canonical image pool, it extracts frames + canonicalizes — so "Run Pipeline"
covers the full video → mesh path (no separate import step). fps is DEDUCED from
the clip duration to hit a target frame count (clamped to a handheld-overlap
band). It is idempotent: skips when images already exist, or for photo scenes.

The remaining phases are ``v4exec`` subprocesses on the chosen ``--host``. Only
ONE id threads through (the solve), resolved from the STORE (newest
``cameras/*`` under the primary subset) — NOT by parsing stdout. Writes
``pipeline_status.json`` (phase + log tail) for the UI to poll. Phases are
idempotent/NOOP where content exists, so a failed run is safe to re-run.

``dry_run`` builds the plan WITHOUT executing — the pre-flight an operator
eyeballs (it shows the deduced fps + expected frame count). The real run needs
ssh + GPU + docker and is operator-verified (T-020).

select (best-N view selection) is intentionally NOT here — that is
EPI-SCN-AUTO-SUBSET-SELECT, not the default ingest-scene pipeline.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
V4EXEC = HERE / "v4exec.py"

# Frame-extraction deduction (STO-SCN-150): aim for a good SfM pool density, with
# fps clamped to a sane handheld-overlap band so neither very short nor very long
# clips go degenerate. precull later removes blur/dups, so a slight over-target is
# fine; far-too-few frames (the 003-firepit 12-frame failure) is what we prevent.
INGEST_TARGET_FRAMES = 500
INGEST_FPS_MIN = 1.0
INGEST_FPS_MAX = 4.0
INGEST_MAX_LONG_EDGE = 1920   # downscale frames to this long edge (plenty for SfM; DA3 scout=504)
VIDEO_EXT = (".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm")


def resize_target(scene_dir: Path, default_long_edge: int = INGEST_MAX_LONG_EDGE) -> int | None:
    """Long-edge to downscale extracted frames to — UNLESS the scene is declared
    fisheye. The fisheye undistort is pinned to its native-res calibration, so it
    needs full-res input; everything else (rectilinear / unknown) downsizes safely
    (FastMap self-calibrates; SfM ~1600px; DA3 scout 504). A fisheye that ISN'T
    declared can't pass the solve's capture-decl gate anyway, so it never reaches
    undistort — so 'resize unless declared fisheye' is the safe rule."""
    cj = scene_dir / "capture.json"
    if cj.exists():
        try:
            if (json.loads(cj.read_text()).get("mode") or "").lower() == "fisheye":
                return None
        except (OSError, ValueError):
            pass
    return default_long_edge


def gpu_hosts() -> list[str]:
    """Configured GPU hosts. Default ['tbeeprz']; override with env KRABBY_GPU_HOSTS."""
    raw = os.environ.get("KRABBY_GPU_HOSTS", "tbeeprz")
    return [h.strip() for h in raw.split(",") if h.strip()]


def resolve_primary_subset(scene_dir: Path) -> str | None:
    link = scene_dir / "images" / "subsets" / "primary"
    if link.is_symlink():
        return Path(os.readlink(link)).name
    if link.is_dir():
        return "primary"
    return None


def resolve_latest_solve(scene_dir: Path, subset: str | None = None) -> str | None:
    """Newest camera solve (dir with cameras.json) under the primary subset."""
    subset = subset or resolve_primary_subset(scene_dir)
    if not subset:
        return None
    cams = scene_dir / "images" / "subsets" / subset / "cameras"
    if not cams.is_dir():
        return None
    solves = [c for c in cams.iterdir() if c.is_dir() and (c / "cameras.json").exists()]
    if not solves:
        return None
    return max(solves, key=lambda c: c.stat().st_mtime).name


# ---- ingest deduction (phase 0) -------------------------------------------

def video_source(scene_dir: Path) -> Path | None:
    cap = scene_dir / "videos" / "capture"
    if not cap.is_dir():
        return None
    return next((p for p in sorted(cap.iterdir()) if p.suffix.lower() in VIDEO_EXT), None)


def canonical_count(scene_dir: Path) -> int:
    images = scene_dir / "images"
    if not images.is_dir():
        return 0
    return sum(1 for d in images.iterdir()
               if d.is_dir() and d.name not in ("subsets", "ingress"))


def video_duration(path: Path) -> float | None:
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
            capture_output=True, text=True, timeout=30)
        return float(r.stdout.strip())
    except (subprocess.SubprocessError, ValueError):
        return None


def deduce_fps(duration_s: float | None, target_frames: int = INGEST_TARGET_FRAMES,
               fps_min: float = INGEST_FPS_MIN, fps_max: float = INGEST_FPS_MAX):
    """fps to land ~target_frames over the clip, clamped to the overlap band.
    Returns (fps, expected_frames)."""
    raw = (target_frames / duration_s) if duration_s and duration_s > 0 else 2.0
    fps = round(max(fps_min, min(fps_max, raw)), 2)
    expected = round(duration_s * fps) if duration_s and duration_s > 0 else None
    return fps, expected


def ingest_plan(scene_dir: Path) -> dict:
    """What phase 0 will do: extract (with deduced fps) or skip (and why)."""
    video = video_source(scene_dir)
    n = canonical_count(scene_dir)
    if not video:
        return {"action": "skip", "reason": "no captured video (image-import scene)"}
    if n > 0:
        return {"action": "skip", "reason": f"{n} canonical images already present (nuke to re-ingest)"}
    dur = video_duration(video)
    fps, expected = deduce_fps(dur)
    le = resize_target(scene_dir)
    return {"action": "extract", "video": video.name,
            "duration_s": round(dur, 1) if dur else None,
            "fps": fps, "expected_frames": expected,
            "max_long_edge": le, "resize": (f"≤{le}px" if le else "native (fisheye)")}


# ---- host phases (v4exec subprocesses) ------------------------------------

PHASES = [
    {"key": "ingest",  "label": "Ingest video → frames + canonicalize", "local": True},
    {"key": "precull", "label": "Pre-cull → PRIMARY subset",
     "args": ["precull", "--set-primary"], "host": False, "needs_solve": False},
    {"key": "solve",   "label": "Spine solve (FastMap poses)",
     "args": ["solve"], "host": True, "needs_solve": False},
    {"key": "covis",   "label": "Co-visibility validity gate",
     "args": ["covis"], "host": True, "needs_solve": True},
    {"key": "scout",   "label": "DA3 scout gaussian",
     "args": ["scout"], "host": True, "needs_solve": True},
    {"key": "mesh",    "label": "DA3 reconstruct (mesh)",
     "args": ["reconstruct-da3", "--sfm", "posed"], "host": True, "needs_solve": False},
]


def build_command(phase: dict, scene: str, host: str, solve: str | None) -> list[str]:
    """v4exec command line for a host/local-cli phase."""
    cmd = [sys.executable, str(V4EXEC), phase["args"][0], scene]
    cmd += phase["args"][1:]
    if phase.get("host"):
        cmd += ["--host", host]
    if phase.get("needs_solve"):
        cmd += ["--solve", solve or "UNRESOLVED"]
    return cmd


def _ingest_cmd_preview(scene_dir: Path) -> list[str]:
    ip = ingest_plan(scene_dir)
    if ip["action"] == "skip":
        return ["(skip ingest)", ip["reason"]]
    return ["ffmpeg", "-i", ip["video"], "-vf", f"fps={ip['fps']}",
            f"({ip['resize']})", "→", f"~{ip['expected_frames']} frames", "+ canonicalize"]


def plan(scene: str, host: str, scene_dir: Path | None = None) -> list[dict]:
    """Full plan for the dry-run preview (includes the deduced ingest fps/frames)."""
    sd = scene_dir or (Path(os.environ.get("KRABBY_SCENES_ROOT", "/var/krabby/scenes")) / scene)
    solve = resolve_latest_solve(sd) or "<after-solve>"
    out = []
    for p in PHASES:
        if p.get("local"):
            out.append({"key": p["key"], "label": p["label"], "cmd": _ingest_cmd_preview(sd)})
        else:
            out.append({"key": p["key"], "label": p["label"],
                        "cmd": build_command(p, scene, host, solve)})
    return out


def run_pipeline(scene_dir: Path, host: str, *, dry_run: bool = False,
                 status_cb=None, tail_lines: int = 40) -> dict:
    """Run the phases in order, stopping on the first failure. `status_cb(rec)`
    fires after every phase transition with the running status record."""
    scene = scene_dir.name
    phases = [{"key": p["key"], "label": p["label"], "status": "pending"} for p in PHASES]
    rec = {"status": "running", "host": host, "dry_run": dry_run,
           "n_phases": len(PHASES), "phase_idx": 0, "phase": None,
           "phases": phases, "log_tail": "", "solve": None}

    def emit():
        if status_cb:
            status_cb(dict(rec))

    emit()
    solve = resolve_latest_solve(scene_dir)
    for i, p in enumerate(PHASES):
        rec["phase_idx"] = i
        rec["phase"] = p["key"]
        phases[i]["status"] = "running"

        # ---- local ingest phase (video → frames + canonicalize) ----
        if p.get("local"):
            ip = ingest_plan(scene_dir)
            phases[i]["cmd"] = _ingest_cmd_preview(scene_dir)
            if ip["action"] == "skip":
                phases[i]["status"] = "skipped"
                phases[i]["note"] = ip["reason"]
                emit()
                continue
            phases[i]["note"] = (f"{ip['video']} · {ip['duration_s']}s → {ip['fps']}fps "
                                 f"≈ {ip['expected_frames']} frames · {ip['resize']}")
            emit()
            if dry_run:
                phases[i]["status"] = "planned"
                emit()
                continue
            try:
                import scene_ingest as si
                video = video_source(scene_dir)
                frames = si.extract_frames(video, scene_dir / "images" / "ingress",
                                           fps=ip["fps"], max_long_edge=ip["max_long_edge"])
                rec["log_tail"] = f"extracted {len(frames)} frames @ {ip['fps']}fps; canonicalizing…"
                emit()

                def _prog(d, t):
                    if d % 25 == 0 or d == t:
                        rec["log_tail"] = f"canonicalize {d}/{t}"
                        emit()
                res = si.canonicalize_files(
                    scene_dir, frames, f"video:{video.name}", move=True,
                    extra={"source_video": video.name, "fps": ip["fps"]}, progress=_prog)
                phases[i]["status"] = "done"
                phases[i]["note"] = f"{res['n']} frames @ {ip['fps']}fps"
                rec["log_tail"] = f"ingest done: {res['n']} canonical images"
                emit()
            except Exception as e:   # noqa: BLE001
                phases[i]["status"] = "error"
                rec["status"] = "error"
                rec["log_tail"] = f"ingest failed: {type(e).__name__}: {e}"
                emit()
                return rec
            continue

        # ---- host / cli phases (v4exec) ----
        if p.get("needs_solve"):
            solve = solve or resolve_latest_solve(scene_dir)
        cmd = build_command(p, scene, host, solve)
        phases[i]["cmd"] = cmd
        emit()

        if dry_run:
            phases[i]["status"] = "planned"
            emit()
            continue

        try:
            r = subprocess.run(cmd, capture_output=True, text=True, cwd=str(HERE))
        except Exception as e:   # noqa: BLE001
            phases[i]["status"] = "error"
            rec["status"] = "error"
            rec["log_tail"] = f"{type(e).__name__}: {e}"
            emit()
            return rec
        out = (r.stdout or "") + (r.stderr or "")
        rec["log_tail"] = "\n".join(out.splitlines()[-tail_lines:])
        if r.returncode != 0:
            phases[i]["status"] = "error"
            phases[i]["rc"] = r.returncode
            rec["status"] = "error"
            emit()
            return rec
        phases[i]["status"] = "done"
        if p["key"] == "solve":
            solve = resolve_latest_solve(scene_dir)
            rec["solve"] = solve
        emit()

    rec["status"] = "done"
    rec["phase"] = None
    emit()
    return rec
