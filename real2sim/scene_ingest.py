"""STO-SCN-149 — New Scene ingest + canonicalize (EPI-SCN-SCENE-MANAGER).

Create a scene (next 3-digit code + kebab name), land a source (1 video / N
images / a folder), and canonicalize to content-hashed
``images/<hash>/{image.<ext>, metadata.json}``. Reuses ``v4core.file_hash`` for
the canonical identity (same layout the rest of the pipeline reads) + the
ffmpeg frame-extraction kernel from ``extract_frames.sh``.

Pure/stdlib + ffmpeg (no numpy) — the server wires HTTP endpoints around these
functions; they are unit-testable against synthetic trees.
"""
from __future__ import annotations

import datetime
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import v4core as v4  # noqa: E402  (stdlib-only; file_hash + STORE)

IMG_EXT = (".jpg", ".jpeg", ".png")
VIDEO_EXT = (".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm")


def _now() -> str:
    return datetime.datetime.now().isoformat()


def kebab(name: str) -> str:
    """Lower-case, hyphen-separated slug. '' → 'scene'."""
    s = re.sub(r"[^a-z0-9]+", "-", (name or "").strip().lower()).strip("-")
    return s or "scene"


def next_code(store: Path) -> str:
    """Next free 3-digit code = max existing NNN- prefix + 1 (001 if none)."""
    codes = []
    if store.is_dir():
        for d in store.iterdir():
            m = re.match(r"^(\d{3})-", d.name)
            if m:
                codes.append(int(m.group(1)))
    return f"{(max(codes) + 1) if codes else 1:03d}"


def create_scene(store: Path, name: str) -> dict:
    """Make ``store/<NNN>-<kebab>/images/`` and return its identity."""
    code = next_code(store)
    slug = kebab(name)
    scene = f"{code}-{slug}"
    d = store / scene
    if d.exists():
        raise FileExistsError(f"scene already exists: {scene}")
    (d / "images").mkdir(parents=True)
    return {"scene": scene, "code": code, "name": slug, "dir": str(d)}


def canonicalize_file(scene_dir: Path, f: Path, origin: str, *,
                      move: bool = False, extra: dict | None = None) -> str | None:
    """Hash one image → ``images/<hash>/{image.<ext>, metadata.json}``.

    Idempotent: an already-canonical image is skipped (and the duplicate source
    dropped when ``move``). Returns the content hash, or None for a non-image.
    """
    if f.suffix.lower() not in IMG_EXT:
        return None
    h = v4.file_hash(f)
    dst = scene_dir / "images" / h
    if not (dst / "metadata.json").exists():
        dst.mkdir(parents=True, exist_ok=True)
        ext = f.suffix.lower()
        (shutil.move if move else shutil.copy2)(str(f), str(dst / f"image{ext}"))
        md = {"schema": 4, "original_name": f.name, "origin": origin,
              "mechanism": "ingest-ui", "written": _now()}
        if extra:
            md.update(extra)
        (dst / "metadata.json").write_text(json.dumps(md, indent=2) + "\n")
    elif move:
        f.unlink(missing_ok=True)
    return h


def canonicalize_files(scene_dir: Path, files: list[Path], origin: str, *,
                       move: bool = False, extra: dict | None = None,
                       progress=None) -> dict:
    """Canonicalize a list of image files; returns {n, hashes}. `progress(done,total)`."""
    imgs = [f for f in files if f.suffix.lower() in IMG_EXT]
    hashes: list[str] = []
    for i, f in enumerate(imgs):
        h = canonicalize_file(scene_dir, f, origin, move=move, extra=extra)
        if h:
            hashes.append(h)
        if progress:
            progress(i + 1, len(imgs))
    return {"n": len(hashes), "hashes": hashes}


def extract_frames(video: Path, out_dir: Path, fps: float = 2.0) -> list[Path]:
    """ffmpeg frame extraction (kernel of extract_frames.sh): out_dir/frame_%04d.jpg."""
    out_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-i", str(video), "-vf", f"fps={fps}", "-q:v", "1",
         str(out_dir / "frame_%04d.jpg"), "-y"],
        check=True, capture_output=True)
    return sorted(out_dir.glob("frame_*.jpg"))


def ingest_images(scene_dir: Path, sources: list[Path], *,
                  move: bool = False, progress=None) -> dict:
    """Land N images / a folder via images/ingress/ then canonicalize. `sources`
    may be files and/or directories (directories are expanded, non-recursively)."""
    files: list[Path] = []
    for s in sources:
        if s.is_dir():
            files += sorted(p for p in s.iterdir() if p.suffix.lower() in IMG_EXT)
        elif s.suffix.lower() in IMG_EXT:
            files.append(s)
    if not files:
        return {"n": 0, "hashes": [], "error": "no images in source"}
    ingress = scene_dir / "images" / "ingress"
    ingress.mkdir(parents=True, exist_ok=True)
    landed = []
    for f in files:
        dst = ingress / f.name
        (shutil.move if move else shutil.copy2)(str(f), str(dst))
        landed.append(dst)
    # canonicalize FROM ingress (always move out of the staging dir)
    return canonicalize_files(scene_dir, landed, "images/ingress", move=True,
                              progress=progress)


def ingest_video(scene_dir: Path, video: Path, *, fps: float = 2.0,
                 move: bool = False, progress=None) -> dict:
    """Land the video at videos/capture/video.<ext>, extract frames, canonicalize."""
    cap = scene_dir / "videos" / "capture"
    cap.mkdir(parents=True, exist_ok=True)
    dst = cap / f"video{video.suffix.lower()}"
    (shutil.move if move else shutil.copy2)(str(video), str(dst))
    frames_dir = scene_dir / "images" / "ingress"
    frames = extract_frames(dst, frames_dir, fps=fps)
    res = canonicalize_files(scene_dir, frames, f"video:{dst.name}", move=True,
                             extra={"source_video": dst.name, "fps": fps},
                             progress=progress)
    res["frames_extracted"] = len(frames)
    res["video"] = str(dst)
    return res


def ingest_path(scene_dir: Path, source: Path, *, fps: float = 2.0,
                move: bool = False, progress=None) -> dict:
    """Dispatch a single source path: video → ingest_video, file/dir → ingest_images."""
    if source.is_file() and source.suffix.lower() in VIDEO_EXT:
        return ingest_video(scene_dir, source, fps=fps, move=move, progress=progress)
    return ingest_images(scene_dir, [source], move=move, progress=progress)
