#!/usr/bin/env python3
"""v4exec.py — the v4 GRAPH EXECUTOR (STO-SCN-088; HUG-SCN-005 locked #11:
DO NOT MANIPULATE DATA OUTSIDE A GRAPH — this is the only writer).

Materializes graph nodes natively into the v4 store. GPU tasks
dispatch to ONE operator-chosen host (locked decision 3) over SSH +
docker using image-baked tools; CPU tasks run locally. Every artifact
gets per-identity metadata (mechanism: job); every invocation gets a
scene job record (locked #8). Existing identities NOOP (locked #4).

Subcommands (the native proof protocol, locked #11):
    ingest <scene> --host U@H [--raw <dir>]   hash-n-images -> images-subset
                                              -> solve-cameras (host GPU)
    reconstruct-matcha <scene> --host U@H [--dense-regul default|strong]
                       [--sfm unposed|posed]  matcha@0/@1 weld (host GPU) ->
                                              orient bootstrap (local) ->
                                              tetra+tsdf meshes (canonical gauge)
    views-from-blend <scene> <blend>          operator-captured virtual cams ->
                                              views/<slot>/view.json + canonical
    reconstruct-da3 <scene> --host U@H        da3 infer (host GPU) -> fuse (local)
    renders: use v4job.py render-missing (already graph-native)
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

# local mesh/gauge steps need numpy + open3d; self-bootstrap via uv
try:
    import numpy  # noqa: F401
    import open3d  # noqa: F401
except ImportError:
    if os.environ.get("V4EXEC_BOOTSTRAPPED") != "1":
        os.environ["V4EXEC_BOOTSTRAPPED"] = "1"
        os.execvp("uv", ["uv", "run", "--quiet", "--python", "3.11",
                         "--with", "open3d", "--with", "numpy",
                         "python3", str(Path(__file__).resolve())] + sys.argv[1:])
    raise

sys.path.insert(0, str(Path(__file__).parent))
import v4core as v4
import capture_profile as cap  # STO-SCN-091
import precull_frames as pre  # STO-SCN-092

SCRATCH = "/home/jeremy/scratch/v4exec"
MATCHA_IMAGE = "j.pski.org:5000/krabby-matcha:0.2.2-selfcontained"
DA3_IMAGE = "j.pski.org:5000/krabby-da3:0.4"
FASTMAP_IMAGE = "j.pski.org:5000/krabby-fastmap:0.2"  # tools baked (STO-SCN-093 D)
# orient-floor@1: bootstrap-mesh + camera-consensus up prior (STO-SCN-089-2;
# restores the operator-guided 'average camera up' lost in the 082 rejection)
ORIENT_ALGO = "orient-floor@2"
ORIENT_SETTINGS = {"method": "bootstrap-mesh", "ransac_dist": 0.05,
                   "up_prior": "horizon"}
DOCKER_GPU = ["--gpus", "all", "--shm-size", "8g"]

NOW = lambda: datetime.datetime.now().astimezone().isoformat(timespec="seconds")  # noqa: E731


def sh(cmd, **kw):
    r = subprocess.run(cmd, capture_output=True, text=True, **kw)
    if r.returncode != 0:
        raise SystemExit(f"FAILED ({r.returncode}): {' '.join(map(str, cmd))}\n{r.stderr[-2000:]}")
    return r.stdout


def job_record(scene: str, graph: str, nodes: list[dict], bindings: dict):
    jd = v4.Scene(scene).job_dir()
    (jd / "job.json").write_text(json.dumps({
        "schema": 4, "graph": graph, "mechanism": "job", "bindings": bindings,
        "nodes": nodes, "written": NOW()}, indent=2) + "\n")
    print(f"job record: {jd.name}")


def host_digest(host: str, image: str) -> tuple[str, str]:
    dig = sh(["ssh", host, f"docker inspect --format '{{{{index .RepoDigests 0}}}}' {image}"]).strip()
    sha = subprocess.run(["ssh", host,
                          f"docker inspect --format '{{{{index .Config.Labels \"io.krabby.da3.tools_git_sha\"}}}}' {image}"],
                         capture_output=True, text=True).stdout.strip()
    return dig, (sha or None)


def pool_image_paths(scene_dir: Path) -> dict[str, Path]:
    """original_name -> pool image path (from per-image metadata)."""
    out = {}
    for md in scene_dir.glob("images/*/metadata.json"):
        d = json.loads(md.read_text())
        img = next((p for p in md.parent.glob("image.*")), None)
        if img and d.get("original_name"):
            out[d["original_name"]] = img
    return out


def stage_images_on_host(host: str, scene_dir: Path, subset_hash: str, tag: str) -> str:
    """rsync the subset's images (original names) to host scratch; return dir."""
    members = json.loads((scene_dir / "images" / "subsets" / subset_hash / "subset.json")
                         .read_text())["members"]
    by_hash = {p.parent.name: p for p in scene_dir.glob("images/*/image.*")}
    names = {}
    for md in scene_dir.glob("images/*/metadata.json"):
        d = json.loads(md.read_text())
        names[md.parent.name] = d.get("original_name", md.parent.name + ".jpg")
    tmp = Path("/tmp") / f"v4exec-{tag}"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    for h in members:
        shutil.copy2(by_hash[h], tmp / names[h])
    dest = f"{SCRATCH}/{tag}/images"
    sh(["ssh", host, f"rm -rf {SCRATCH}/{tag} && mkdir -p {dest}"])
    sh(["rsync", "-a", f"{tmp}/", f"{host}:{dest}/"])
    shutil.rmtree(tmp)
    return f"{SCRATCH}/{tag}"


def run_in_matcha(host: str, workdir: str, tool_cmd: str, log_to: Path) -> int:
    full = (f"cd /opt/MAtCha && {tool_cmd}")   # train.py lives in the baked source tree
    docker = (f"docker run --rm --gpus all --shm-size 8g -v {workdir}:/work "
              f"--entrypoint bash {MATCHA_IMAGE} -lc {json.dumps(full)} ; rc=$? ; "
              f"docker run --rm -v {workdir}:/work alpine chown -R $(id -u):$(id -g) /work ; exit $rc")
    r = subprocess.run(["ssh", host, docker], capture_output=True, text=True)
    log_to.parent.mkdir(parents=True, exist_ok=True)
    log_to.write_text(r.stdout[-200000:] + "\n--- stderr ---\n" + r.stderr[-50000:])
    return r.returncode


def run_in_fastmap(host: str, workdir: str, tool_cmd: str, log_to: Path) -> int:
    """Run a command in the krabby-fastmap container (baked tools at
    /opt/krabby-tools, colmap, /opt/fastmap) with {workdir}:/work, then chown the
    outputs back to the caller (container writes as root)."""
    docker = (f"docker run --rm --gpus all --shm-size 8g -v {workdir}:/work "
              f"--entrypoint bash {FASTMAP_IMAGE} -lc {json.dumps(tool_cmd)} ; rc=$? ; "
              f"docker run --rm -v {workdir}:/work alpine chown -R $(id -u):$(id -g) /work ; exit $rc")
    r = subprocess.run(["ssh", host, docker], capture_output=True, text=True)
    log_to.parent.mkdir(parents=True, exist_ok=True)
    log_to.write_text(r.stdout[-200000:] + "\n--- stderr ---\n" + r.stderr[-50000:])
    return r.returncode


# ============================================================ ingest

def cmd_ingest(args):
    scene_dir = v4.STORE / args.scene
    sc = v4.Scene(args.scene)
    nodes = []

    # -- node: hash-n-images (operator-draft task; identity = HOH)
    raw = Path(args.raw) if args.raw else scene_dir / "raw"
    hashes = []
    if raw.is_dir():
        for f in sorted(raw.iterdir()):
            if f.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue
            h = v4.file_hash(f)
            hashes.append(h)
            dst = scene_dir / "images" / h
            if not (dst / "metadata.json").exists():
                dst.mkdir(parents=True, exist_ok=True)
                shutil.copy2(f, dst / f"image{f.suffix.lower()}")
                (dst / "metadata.json").write_text(json.dumps({
                    "schema": 4, "original_name": f.name, "origin": str(raw),
                    "mechanism": "job", "written": NOW()}, indent=2) + "\n")
    else:
        hashes = [p.parent.name for p in scene_dir.glob("images/*/metadata.json")]
    if not hashes:
        sys.exit(f"no images found (raw dir {raw} or existing pool)")
    nodes.append({"node": "hash", "task": "hash-n-images", "n": len(hashes), "action": "EXECUTE"})

    # -- node: images-subset (locked #5: identity = HOH only)
    sub = v4.hoh(hashes)
    sdir = scene_dir / "images" / "subsets" / sub
    if not (sdir / "subset.json").exists():
        sdir.mkdir(parents=True, exist_ok=True)
        (sdir / "subset.json").write_text(json.dumps({"schema": 4, "members": sorted(hashes)},
                                                     indent=2) + "\n")
        (sdir / "metadata.json").write_text(json.dumps({
            "schema": 4, "mechanism": "all", "label": "whole-pool", "settings": {},
            "written": NOW()}, indent=2) + "\n")
        nodes.append({"node": "pool", "task": "images-subset", "identity": sub, "action": "EXECUTE"})
    else:
        nodes.append({"node": "pool", "task": "images-subset", "identity": sub, "action": "NOOP"})
    created = sc.set_ref_if_unset("primary", sub)
    nodes.append({"node": "primary-ref", "action": "set" if created else "NOOP", "target": sub})

    # -- node: resolve-capture-profile (STO-SCN-091; camera model from declared
    #    capture mode, NOT pixel inference). Declaration precedence: CLI flags >
    #    <scene>/capture.json. Identity = hash of the declaration, so a camera+mode
    #    resolves once and is reused. ABSENCE of a declaration is a SKIP (today's
    #    mast3r-sfm solve doesn't consume the model; STO-SCN-093 dispatch will) —
    #    only a PRESENT-but-unresolvable declaration fails loud (no guessed model).
    cp_make = getattr(args, "camera_make", None)
    cp_model = getattr(args, "camera_model", None)
    cp_mode = getattr(args, "capture_mode", None)
    decl_path = scene_dir / "capture.json"
    if not (cp_make and cp_model and cp_mode) and decl_path.exists():
        decl = json.loads(decl_path.read_text())
        cp_make = cp_make or decl.get("make")
        cp_model = cp_model or decl.get("model")
        cp_mode = cp_mode or decl.get("mode")
    if not any([cp_make, cp_model, cp_mode]):
        print("[capture-profile] SKIP — no <scene>/capture.json and no --capture-mode; "
              "camera model unresolved (STO-SCN-093 dispatch will require it).")
        nodes.append({"node": "capture-profile", "action": "SKIP", "reason": "no-declaration"})
    else:
        cp_settings = v4.hashable_settings(v4.tasks()["resolve-capture-profile"],
                                           {"make": cp_make, "model": cp_model, "mode": cp_mode})
        cpid = v4.identity_hash({}, cp_settings, "capture-profile@0")
        cpdir = scene_dir / "images" / "capture-profile" / cpid
        if (cpdir / "metadata.json").exists():
            nodes.append({"node": "capture-profile", "identity": cpid, "action": "NOOP"})
        else:
            try:
                prof = cap.resolve(cp_make, cp_model, cp_mode)
            except cap.ProfileError as e:
                sys.exit(f"capture-profile: {e}")
            cpdir.mkdir(parents=True, exist_ok=True)
            (cpdir / "capture-profile.json").write_text(json.dumps(prof, indent=2) + "\n")
            v4.write_metadata(cpdir, task="resolve-capture-profile", algo="capture-profile@0",
                              identity=cpid, resolved_inputs={}, settings=cp_settings,
                              mechanism="job",
                              extra={"camera_model": prof["colmap_camera_model"],
                                     "colmap_compatible": prof["colmap_compatible"],
                                     "dewarp_dead_end": prof["dewarp_dead_end"]})
            print(f"[capture-profile] {cp_make} {cp_model} / {cp_mode} -> "
                  f"{prof['colmap_camera_model']} (colmap_compatible={prof['colmap_compatible']})")
            nodes.append({"node": "capture-profile", "identity": cpid, "action": "EXECUTE",
                          "camera_model": prof["colmap_camera_model"],
                          "colmap_compatible": prof["colmap_compatible"]})

    # -- node: solve-cameras (host GPU; mast3r-sfm@0 via matcha --sfm_only)
    s_settings = v4.hashable_settings(v4.tasks()["solve-cameras"], {})
    sid = v4.identity_hash({"subset": sub}, s_settings, "mast3r-sfm@0")
    sdir_solve = sdir / "cameras" / sid
    if (sdir_solve / "metadata.json").exists():
        nodes.append({"node": "solve", "identity": sid, "action": "NOOP"})
    else:
        tag = f"{args.scene}-solve-{sid}"
        workdir = stage_images_on_host(args.host, scene_dir, sub, tag)
        tool = (f"python train.py -s /work/images -o /work/out --sfm_only "
                f"--sfm_config unposed "
                f"--depthanythingv2_checkpoint_dir /opt/MAtCha/Depth-Anything-V2/checkpoints "
                f"--depthanything_encoder vitl")
        print(f"[solve] {args.host}: {tool}")
        t0 = datetime.datetime.now()
        rc = run_in_matcha(args.host, workdir, tool, sdir_solve / "solve.log")
        dt = int((datetime.datetime.now() - t0).total_seconds())
        # gather: cameras + sparse points (expected-outputs gate)
        sh(["rsync", "-a", "--include=cameras.json", "--include=points.ply",
            "--exclude=*", f"{args.host}:{workdir}/out/mast3r_sfm/", str(sdir_solve) + "/"])
        sh(["ssh", args.host, f"rm -rf {SCRATCH}/{tag}"])
        if rc != 0 or not (sdir_solve / "cameras.json").exists():
            sys.exit(f"solve FAILED (rc={rc}; see {sdir_solve}/solve.log)")
        dig, _ = host_digest(args.host, MATCHA_IMAGE)
        v4.write_metadata(sdir_solve, task="solve-cameras", algo="mast3r-sfm@0",
                          identity=sid, resolved_inputs={"subset": sub},
                          settings=s_settings, mechanism="job",
                          measured={"host": args.host.split("@")[-1], "duration_s": dt,
                                    "image_digest": dig})
        nodes.append({"node": "solve", "identity": sid, "action": "EXECUTE",
                      "host": args.host, "duration_s": dt})
        print(f"[solve] done in {dt}s -> {sdir_solve}")
    job_record(args.scene, "ingest-scene", nodes,
               {"scene": args.scene, "host": args.host, "raw": str(raw)})


# ============================================================ precull (STO-SCN-092)

def cmd_precull(args):
    """Pose-free pre-cull: derive a curated candidate SUBSET from the whole pool
    (sharpness + pHash dedup, no poses). Opt-in side-branch — writes the subset
    set-if-unset; leaves `primary` untouched unless --set-primary (a deliberate
    operator act, locked #1)."""
    scene_dir = v4.STORE / args.scene
    sc = v4.Scene(args.scene)
    # Order by CAPTURE TIME (metadata original_name), NOT store hash — the
    # pre-cull's temporal dedup + gap guard need true frame order. (STO-SCN-093
    # finding: hash order found 4 near-dups on a hyperlapse; capture order found
    # 403.) The id passed downstream stays the content hash.
    entries = []
    for md in scene_dir.glob("images/*/metadata.json"):
        imgs = sorted(md.parent.glob("image.*"))
        if not imgs:
            continue
        name = json.loads(md.read_text()).get("original_name", md.parent.name)
        entries.append((name, md.parent.name, imgs[0]))
    if not entries:
        sys.exit(f"no pooled images for {args.scene} — run ingest first")
    entries.sort(key=lambda e: e[0])              # capture order
    items = [(h, p) for _, h, p in entries]
    hashes = sorted(h for _, h, _ in entries)     # set-stable id for the pool hoh

    ps_settings = v4.hashable_settings(v4.tasks()["precull-subset"], {
        "target": args.target, "phash_thresh": args.phash_thresh, "blur_rel": args.blur_rel,
        "max_gap": args.max_gap, "dup_window": args.dup_window, "score_edge": args.score_edge})
    pool_id = v4.hoh(hashes)
    pid = v4.identity_hash({"pool": pool_id}, ps_settings, "precull@0")
    pdir = scene_dir / "images" / "subsets" / pid

    if (pdir / "subset.json").exists():
        print(f"[precull] NOOP — subset {pid} exists")
    else:
        res = pre.precull(items, target=(args.target or None),
                          phash_thresh=args.phash_thresh, blur_rel=args.blur_rel,
                          max_gap=args.max_gap, dup_window=args.dup_window,
                          score_edge=args.score_edge)
        members = sorted(res.kept)
        pdir.mkdir(parents=True, exist_ok=True)
        (pdir / "subset.json").write_text(json.dumps({"schema": 4, "members": members},
                                                     indent=2) + "\n")
        # subset metadata mirrors the whole-pool subset shape (scan_scene reads
        # label/mechanism) + the measured precull report.
        (pdir / "metadata.json").write_text(json.dumps({
            "schema": 4, "mechanism": "precull", "label": f"precull-{len(members)}",
            "settings": ps_settings, "written": NOW(),
            "resolved_inputs": {"pool": pool_id},
            "source_pool_n": res.report["source_pool_n"], "kept_n": res.report["kept_n"],
            "precull_report": res.report}, indent=2) + "\n")
        r = res.report
        print(f"[precull] {r['source_pool_n']} -> {r['kept_n']} "
              f"(near-dup -{r['dropped_near_dup']}, blur -{r['dropped_blur']}, "
              f"thinned -{r['thinned_to_target']}, gap-fill +{r['gap_filled_inserted']}) "
              f"-> subset {pid}")

    if getattr(args, "set_primary", False):
        created = sc.set_ref_if_unset("primary", pid)
        if created:
            print(f"[precull] primary -> {pid}")
        else:
            cur = sc.resolve("primary")
            print(f"[precull] primary already set ({cur}); ref moves are operator acts "
                  f"(locked #1). Re-point primary deliberately, or pass subset {pid} to "
                  f"the solve, to use the curated set.")
    else:
        print(f"[precull] primary unchanged (opt-in). Curated subset id: {pid}")

    job_record(args.scene, "precull-subset",
               [{"node": "precull", "identity": pid}],
               {"scene": args.scene, "set_primary": bool(getattr(args, "set_primary", False))})


# ============================================================ spine segmentation (STO-SCN-097)

def cmd_spine(args):
    """Spine segmentation (spine@0): partition the ordered pool into M overlapping
    segments, each <= cap, adjacent pairs sharing >= overlap frames, with per-seam
    registrability + cross-segment loop candidates (pHash). Pose-free; runs locally
    on the pool (single decode pass), like precull. Emits spine.json (the per-segment
    boundary_spec + global camera_model) at the scene root. Idempotent NOOP."""
    import spine_segment as sps
    scene_dir = v4.STORE / args.scene
    # Capture-order pool (same ordering the pre-cull uses — true frame order, NOT
    # store-hash order; STO-SCN-093 finding). Downstream id = the store content hash.
    entries = []
    for md in scene_dir.glob("images/*/metadata.json"):
        imgs = sorted(md.parent.glob("image.*"))
        if not imgs:
            continue
        name = json.loads(md.read_text()).get("original_name", md.parent.name)
        entries.append((name, md.parent.name, imgs[0]))
    if not entries:
        sys.exit(f"no pooled images for {args.scene} — run ingest first")
    entries.sort(key=lambda e: e[0])                       # capture order
    ids = [h for _, h, _ in entries]
    paths = [p for _, _, p in entries]
    pool_id = v4.hoh(sorted(ids))                          # set-stable pool id

    settings = v4.hashable_settings(v4.tasks()["spine-segment"], {
        "cap": args.cap, "overlap": args.overlap, "snap": args.snap,
        "reg_thresh": args.reg_thresh, "loop_thresh": args.loop_thresh,
        "loop_min_sep": args.loop_min_sep, "loop_step": args.loop_step})
    sid = v4.identity_hash({"pool": pool_id}, settings, "spine@0")
    sdir = scene_dir / "spine" / sid
    if (sdir / "metadata.json").exists():
        print(f"[spine] NOOP — spine {sid} exists -> {sdir}")
        return

    # global camera model (STO-SCN-091; identical for every segment)
    make, model, mode, modality = _read_capture_decl(scene_dir)
    _, hashes = sps.hashes_for(list(zip(ids, paths)))      # one decode pass
    spec = sps.segment(ids, hashes, cap=args.cap, overlap=args.overlap, snap=args.snap,
                       reg_thresh=args.reg_thresh, loop_thresh=args.loop_thresh,
                       loop_min_sep=args.loop_min_sep, loop_step=args.loop_step)
    spec["camera_model"] = {"make": make, "model": model, "mode": mode, "modality": modality}
    spec["pool"] = pool_id
    sdir.mkdir(parents=True, exist_ok=True)
    (sdir / "spine.json").write_text(json.dumps(spec, indent=2) + "\n")
    v4.write_metadata(sdir, task="spine-segment", algo="spine@0", identity=sid,
                      resolved_inputs={"pool": pool_id}, settings=settings, mechanism="local",
                      measured={"n_frames": spec["n_frames"], "n_segments": spec["n_segments"],
                                "max_segment_n": spec["max_segment_n"],
                                "within_capacity": spec["within_capacity"],
                                "all_seams_registrable": spec["all_seams_registrable"],
                                "n_loop_candidates": spec["n_loop_candidates"]})
    bad = [s for s in spec["seams"] if not s["registrable"]]
    print(f"[spine] {spec['n_frames']} frames -> {spec['n_segments']} segments "
          f"(max {spec['max_segment_n']}/{args.cap}, within-cap={spec['within_capacity']}, "
          f"all-registrable={spec['all_seams_registrable']}, "
          f"{spec['n_loop_candidates']} loop candidate(s)) -> {sdir}")
    if bad:
        print(f"[spine] WARNING: {len(bad)} seam(s) below registrability threshold "
              f"{args.reg_thresh} — widen --overlap or lower --reg-thresh: "
              f"{[(s['seg_a'], s['seg_b'], s['registrability']) for s in bad]}")
    job_record(args.scene, "spine-segment", [{"node": "spine", "identity": sid,
                                              "n_segments": spec["n_segments"]}],
               {"scene": args.scene, "pool": pool_id})


# ============================================================ solve (FastMap) + covis (STO-SCN-093)

def _infer_modality(scene_dir):
    """Auto-heal a missing `modality` from STORE FACTS (not a guess):
      - a captured video (`videos/capture/video.*`)            -> 'video'
      - canonical images that are extracted frames (`frame_*`)  -> 'video'
      - canonical images with discrete camera names            -> 'photos'
    Returns (modality | None, reason). `hyperlapse` is NOT store-detectable
    (no on-disk signal distinguishes it from video) — it must be declared
    explicitly in capture.json; declaration always wins over inference.
    """
    cap = scene_dir / "videos" / "capture"
    if cap.is_dir() and any(cap.glob("video.*")):
        vids = sorted(cap.glob("video.*"))
        return "video", f"{vids[0].name} present in videos/capture/"
    names = []
    for md in scene_dir.glob("images/*/metadata.json"):
        try:
            names.append(json.loads(md.read_text()).get("original_name", "") or "")
        except (OSError, ValueError):
            continue
    if names:
        frames = sum(1 for n in names if n.lower().lstrip().startswith("frame"))
        if frames >= max(1, (len(names) + 1) // 2):
            return "video", f"{frames}/{len(names)} canonical images are extracted frames (frame_*)"
        return "photos", f"{len(names) - frames}/{len(names)} canonical images are discrete photos"
    return None, "no captured video and no canonical images to infer from"


def _read_capture_decl(scene_dir):
    """Resolve (make, model, mode, modality) for a scene, AUTO-HEALING from facts
    so capture.json is optional for EXIF-identifiable, single-mode registered
    cameras (STO-SCN-091/093):
      - make/model : declared in capture.json, else read from a canonical image's EXIF
      - mode       : declared, else the registry's SOLE capture mode for that camera
                     (multi-mode cameras like DJI fisheye/dewarped still need it declared)
      - modality   : declared, else inferred from store facts (`_infer_modality`)
    The camera MODEL is never guessed (the registry owns it, STO-SCN-091) — an
    unknown camera fails loud with an add-a-profile message, not a bad default.
    """
    import capture_profile as cap
    p = scene_dir / "capture.json"
    d = json.loads(p.read_text()) if p.exists() else {}

    make, model = d.get("make"), d.get("model")
    if not (make and model):
        img = next(iter(sorted(scene_dir.glob("images/*/image.*"))), None)
        if img is not None:
            ex = cap.read_exif(img)
            make = make or ex.get("make")
            model = model or ex.get("model")
            if make and model:
                print(f"[capture] make/model auto-healed from EXIF -> {make} / {model}")
    if not (make and model):
        sys.exit(f"{scene_dir}: make/model not declared and not in image EXIF — "
                 f"add capture.json {{make, model, mode}} (STO-SCN-091).")

    mode = d.get("mode")
    if not mode:
        reg = cap.load_registry()
        modes = sorted({pr.get("mode") for pr in reg
                        if cap._norm(pr.get("make")) == cap._norm(make)
                        and cap._norm(pr.get("model")) == cap._norm(model)
                        and pr.get("mode")})
        if len(modes) == 1:
            mode = modes[0]
            print(f"[capture] mode auto-healed -> '{mode}' (the only registry mode for {make} {model})")
        elif len(modes) > 1:
            sys.exit(f"{make} {model} has multiple capture modes {modes} — declare 'mode' in "
                     f"{p} (not derivable from EXIF).")
        else:
            sys.exit(f"no capture profile for {make!r} {model!r} — add one to "
                     f"capture_profiles.json (STO-SCN-091; never guess the camera model).")

    modality = d.get("modality")
    if not modality:
        modality, why = _infer_modality(scene_dir)
        if not modality:
            sys.exit(f"{scene_dir}: 'modality' not declared and not inferable ({why}) — "
                     f"declare hyperlapse|video|photos.")
        print(f"[capture] modality auto-healed -> '{modality}' ({why}); "
              f"declare 'modality' in capture.json to override (e.g. hyperlapse).")
    return make, model, mode, modality


def cmd_solve(args):
    """GPU FastMap solve (fastmap@0) on a subset -> poses + sparse/0. Fisheye is
    undistorted to pinhole first (102 calibration). Settings come from solve_plan
    (091 profile + per-scene modality). Tools are the baked container copies."""
    import capture_profile as cap
    import solve_plan as splan
    scene_dir = v4.STORE / args.scene
    sc = v4.Scene(args.scene)
    subset = args.subset or sc.resolve("primary")
    if not (scene_dir / "images" / "subsets" / subset / "subset.json").exists():
        sys.exit(f"subset {subset} not found — run precull/ingest first")

    make, model, mode, modality = _read_capture_decl(scene_dir)
    profile = cap.resolve(make, model, mode)
    plan = splan.plan_solve(profile, modality)
    for w in plan.get("warnings", []):
        print(f"[solve] WARN: {w}")
    if plan["solver"] != "fastmap":
        sys.exit(f"[solve] solve_plan picked '{plan['solver']}' (not fastmap) for "
                 f"{mode}/{modality} — use the reconstruct-da3 path for that.")

    settings = {"camera_model": plan["solve_camera_model"], "undistort": plan["undistort"],
                "balance": plan["undistort_balance"] or 0.0, "matcher": plan["matcher"]}
    s_settings = v4.hashable_settings(v4.tasks()["solve-fastmap"], settings)
    sid = v4.identity_hash({"subset": subset}, s_settings, "fastmap@0")
    sdir = scene_dir / "images" / "subsets" / subset / "cameras" / sid
    if (sdir / "metadata.json").exists():
        print(f"[solve] NOOP — fastmap@0 solve {sid} exists")
        # STO-SCN-129: self-heal — backfill cameras.json for an older solve that predates
        # solve-time emission (idempotent; only writes if missing).
        cj = sdir / "cameras.json"
        if not cj.exists() and (sdir / "sparse" / "0" / "images.bin").exists():
            n = posed_sparse_to_cameras_json(sdir / "sparse" / "0", cj)
            print(f"  -> backfilled cameras.json ({n} cams, 512-conv)")
        print(f"  -> {sdir}")
        return

    tag = f"{args.scene}-fmsolve-{sid}"
    workdir = stage_images_on_host(args.host, scene_dir, subset, tag)
    cm, matcher, balance = settings["camera_model"], settings["matcher"], settings["balance"]
    # Run the instrumented HOST orchestrator: run_fastmap.sh emits per-phase
    # nanny-progress -> MQTT on the host (tbeeprz) and docker-runs the BAKED
    # container for the compute (undistort/colmap/fastmap). Deploy the script +
    # lib_progress.sh next to the staged images so its `source $HERE/...` resolves.
    here = Path(__file__).parent
    sh(["rsync", "-a", str(here.parent / "images" / "fastmap" / "run_fastmap.sh"),
        str(here / "lib_progress.sh"), str(here / "capture_profiles.json"),
        f"{args.host}:{workdir}/"])            # stage the CURRENT registry (self-deploy; no rebuild)
    env = f"KRABBY_FASTMAP_IMAGE={json.dumps(FASTMAP_IMAGE)}"
    if settings["undistort"]:
        env += (f" UNDISTORT_MODE={json.dumps(mode)} UNDISTORT_MAKE={json.dumps(make)}"
                f" UNDISTORT_MODEL={json.dumps(model)} UNDISTORT_BALANCE={balance}")
    remote = (f"cd {workdir} && {env} bash run_fastmap.sh {workdir}/images {workdir}/out "
              f"{cm} {matcher} 1800 ; rc=$? ; "
              f"docker run --rm -v {workdir}:/work alpine chown -R $(id -u):$(id -g) /work ; exit $rc")
    print(f"[solve] {args.host}: fastmap@0 via run_fastmap.sh "
          f"(undistort={settings['undistort']}, {matcher}, {cm}) — progress -> MQTT")
    t0 = datetime.datetime.now()
    r = subprocess.run(["ssh", args.host, remote], capture_output=True, text=True)
    sdir.mkdir(parents=True, exist_ok=True)
    (sdir / "solve.log").write_text(r.stdout[-200000:] + "\n--- stderr ---\n" + r.stderr[-50000:])
    rc = r.returncode
    dt = int((datetime.datetime.now() - t0).total_seconds())
    # gather sparse/0 + intrinsics
    sh(["rsync", "-a", "--include=sparse/***", "--include=intrinsics.json", "--exclude=*",
        f"{args.host}:{workdir}/out/", str(sdir) + "/"])
    sh(["ssh", args.host, f"rm -rf {SCRATCH}/{tag} {workdir}"])
    if rc != 0 or not (sdir / "sparse" / "0" / "images.bin").exists():
        sys.exit(f"[solve] FAILED (rc={rc}; see {sdir}/solve.log)")
    # STO-SCN-129: emit the renderable+posable cameras.json from sparse/0 at SOLVE time, so every
    # FastMap solve carries it (512-conv focals) — consumers (render, da3-scout, matcha@1 posed)
    # no longer backfill it lazily. Shared helper; idempotent.
    cams_json = sdir / "cameras.json"
    n_cams = posed_sparse_to_cameras_json(sdir / "sparse" / "0", cams_json)
    print(f"[solve] emitted cameras.json ({n_cams} cams, 512-conv) -> {cams_json}")
    import struct
    with open(sdir / "sparse" / "0" / "images.bin", "rb") as f:
        n_reg = struct.unpack("<Q", f.read(8))[0]
    dig, tools_sha = host_digest(args.host, FASTMAP_IMAGE)
    v4.write_metadata(sdir, task="solve-fastmap", algo="fastmap@0", identity=sid,
                      resolved_inputs={"subset": subset}, settings=s_settings, mechanism="job",
                      measured={"host": args.host.split("@")[-1], "duration_s": dt,
                                "registered_images": n_reg, "image_digest": dig,
                                "tools_git_sha": tools_sha},
                      extra={"camera_model": cm, "undistort": settings["undistort"]})
    print(f"[solve] done in {dt}s: {n_reg} images registered -> {sdir}")
    job_record(args.scene, "solve-fastmap", [{"node": "solve", "identity": sid,
               "registered": n_reg}], {"scene": args.scene, "subset": subset, "host": args.host})


def cmd_covis(args):
    """Co-visibility graph + validity gate (covis@0) from a fastmap@0 solve.
    Hard-fails on a nebula so a bad solve never reaches selection (094). CPU; runs
    in the baked container (covis_graph/validity_gate at /opt/krabby-tools)."""
    scene_dir = v4.STORE / args.scene
    sc = v4.Scene(args.scene)
    subset = args.subset or sc.resolve("primary")
    sdir = scene_dir / "images" / "subsets" / subset / "cameras" / args.solve
    if not (sdir / "sparse" / "0" / "images.bin").exists():
        sys.exit(f"no fastmap solve sparse/0 at {sdir} (run `solve` first)")

    settings = v4.hashable_settings(v4.tasks()["covis"], {"min_overlap": args.min_overlap})
    cid = v4.identity_hash({"solve": args.solve}, settings, "covis@0")
    cdir = sdir / "covis" / cid
    if (cdir / "metadata.json").exists():
        print(f"[covis] NOOP — covis {cid} exists -> {cdir}")
        return

    tag = f"{args.scene}-covis-{cid}"
    work = f"{SCRATCH}/{tag}"
    sh(["ssh", args.host, f"rm -rf {work} && mkdir -p {work}/sparse"])
    sh(["rsync", "-a", f"{sdir}/sparse/0", f"{args.host}:{work}/sparse/"])
    cmd = (f"python /opt/krabby-tools/covis_graph.py /work/sparse/0 "
           f"--min-overlap {args.min_overlap} --out /work/covis.json && "
           f"python /opt/krabby-tools/validity_gate.py /work/sparse/0 > /work/validity.txt 2>&1; "
           f"echo VG_RC=$? >> /work/validity.txt")
    rc = run_in_fastmap(args.host, work, cmd, cdir / "covis.log")
    cdir.mkdir(parents=True, exist_ok=True)
    sh(["rsync", "-a", f"{args.host}:{work}/covis.json", f"{args.host}:{work}/validity.txt",
        str(cdir) + "/"])
    sh(["ssh", args.host, f"rm -rf {work}"])
    if rc != 0 or not (cdir / "covis.json").exists():
        sys.exit(f"[covis] FAILED (rc={rc}; see {cdir}/covis.log)")
    g = json.loads((cdir / "covis.json").read_text())
    vtxt = (cdir / "validity.txt").read_text()
    verdict = "FAIL-nebula" if "FAIL" in vtxt else "PASS"
    (cdir / "validity.json").write_text(json.dumps(
        {"verdict": verdict, "raw": vtxt.strip()}, indent=2) + "\n")
    v4.write_metadata(cdir, task="covis", algo="covis@0", identity=cid,
                      resolved_inputs={"solve": args.solve}, settings=settings, mechanism="job",
                      measured={"n_images": g["n_images"], "connected": g["connected"],
                                "n_isolated": g["n_isolated"], "validity": verdict})
    print(f"[covis] {g['n_images']} imgs, connected={g['connected']}, isolated={g['n_isolated']}, "
          f"validity={verdict} -> {cdir}")
    if verdict != "PASS":
        sys.exit(f"[covis] HARD-FAIL: validity {verdict} — solve rejected, not handed to 094.")
    job_record(args.scene, "covis", [{"node": "covis", "identity": cid, "validity": verdict}],
               {"scene": args.scene, "subset": subset, "solve": args.solve})


# ============================================================ best-N selection (STO-SCN-094)

def cmd_select(args):
    """Best-N view selection (select@0) over a fastmap@0 solve, gated behind a PASSing
    covis. Two objectives: `voxel` (STO-SCN-103, default — voxel-face coverage flux,
    rewards angular variety) or `track` (STO-SCN-094 — covisibility). Pure-python; runs
    locally on the store's sparse/0. Emits selection.json + posed.json AND a
    content-addressed FINAL-N subset (the STO-SCN-095 handoff) consumed unchanged by the
    reconstruct graphs via `--subset <final>`. Lives under the solve it selects from."""
    import posed_from_sparse as pfs
    scene_dir = v4.STORE / args.scene
    sc = v4.Scene(args.scene)
    subset = args.subset or sc.resolve("primary")
    sdir = scene_dir / "images" / "subsets" / subset / "cameras" / args.solve
    sparse = sdir / "sparse" / "0"
    if not (sparse / "images.bin").exists():
        sys.exit(f"no solve sparse/0 at {sdir} (run `solve` first)")

    # GATE: selection only proceeds behind a covis that PASSed validity — a nebula
    # solve must never reach the selector (STO-SCN-093 contract).
    cv = sdir / "covis" / args.covis / "validity.json"
    if not cv.exists():
        sys.exit(f"no covis {args.covis} at {sdir}/covis (run `covis` first)")
    verdict = (json.loads(cv.read_text()) or {}).get("verdict")
    if verdict != "PASS":
        sys.exit(f"[select] covis {args.covis} validity={verdict} — refusing to "
                 f"select over a rejected solve.")

    settings = v4.hashable_settings(v4.tasks()["select"],
                                    {"selector": args.selector, "n": args.n, "grid": args.grid,
                                     "min_overlap": args.min_overlap, "div_angle": args.div_angle})
    cid = v4.identity_hash({"covis": args.covis}, settings, "select@0")
    cdir = sdir / "select" / cid
    if (cdir / "metadata.json").exists():
        print(f"[select] NOOP — select {cid} exists -> {cdir}")
        return

    if args.selector == "voxel":                                   # STO-SCN-103
        import voxel_coverage as vc
        _names, rep = vc.select_from_sparse(str(sparse), args.n, grid=args.grid)
    else:                                                          # STO-SCN-094 track
        import select_views as selv
        _names, rep = selv.select_from_sparse(str(sparse), args.n,
                                              min_overlap=args.min_overlap, div_angle=args.div_angle)
    posed = pfs.posed_from_sparse(str(sparse), rep["selected"])

    # FINAL-N subset (STO-SCN-095 handoff): map selected NAMES -> store hashes, write a
    # content-addressed subset the reconstruct graphs consume unchanged (`--subset <final>`).
    name2hash = {}
    for md in scene_dir.glob("images/*/metadata.json"):
        d = json.loads(md.read_text())
        name2hash[d.get("original_name", md.parent.name)] = md.parent.name
    members = sorted({name2hash[n] for n in rep["selected"] if n in name2hash})
    final_id = v4.hoh(members)
    final_sub = scene_dir / "images" / "subsets" / final_id
    if not (final_sub / "subset.json").exists():
        final_sub.mkdir(parents=True, exist_ok=True)
        (final_sub / "subset.json").write_text(json.dumps({"schema": 4, "members": members},
                                                          indent=2) + "\n")
        (final_sub / "metadata.json").write_text(json.dumps({
            "schema": 4, "mechanism": "select", "label": f"final-{len(members)}",
            "resolved_inputs": {"covis": args.covis, "select": cid},
            "selector": args.selector, "kept_n": len(members), "written": NOW()}, indent=2) + "\n")

    cdir.mkdir(parents=True, exist_ok=True)
    (cdir / "selection.json").write_text(json.dumps(rep, indent=2) + "\n")
    (cdir / "posed.json").write_text(json.dumps(posed, indent=2) + "\n")
    (cdir / "final.json").write_text(json.dumps(
        {"schema": 4, "final_subset": final_id, "n": len(members), "members": members,
         "selector": args.selector, "names": rep["selected"]}, indent=2) + "\n")
    cov = rep.get("face_coverage_pct", rep.get("coverage_pct"))
    v4.write_metadata(cdir, task="select", algo="select@0", identity=cid,
                      resolved_inputs={"covis": args.covis}, settings=settings,
                      mechanism="local",
                      measured={"selector": args.selector, "n_selected": rep["n_selected"],
                                "coverage_pct": cov,
                                "median_view_spread_deg": rep.get("median_view_spread_deg"),
                                "final_subset": final_id, "final_n": len(members)})
    print(f"[select] {args.selector}: {rep['n_selected']} views | "
          f"coverage {cov}% | view-spread {rep.get('median_view_spread_deg')}deg")
    print(f"[select] FINAL N -> subset {final_id} ({len(members)} members) — reconstruct "
          f"with: reconstruct-matcha {args.scene} --subset {final_id} --sfm unposed -> {cdir}")
    job_record(args.scene, "select", [{"node": "select", "identity": cid,
                                       "selector": args.selector, "final_subset": final_id}],
               {"scene": args.scene, "subset": subset, "solve": args.solve, "covis": args.covis})


# ============================================================ spine global registration (STO-SCN-098)

def cmd_spine_register(args):
    """Global registration of per-segment submaps into one gauge (spine-register@0).
    Reads each segment's solve (images.bin), builds a SIM(3) pose graph over shared
    boundary cameras + loop correspondences, relaxes into one drift-corrected gauge,
    and emits global.json (per-segment gauges + per-seam residuals + globally
    consistent per-camera poses for fusion). Pure-numpy; runs locally. Idempotent NOOP.

    --solves: comma list `seg=subset/solve` (one per segment); each resolves to
    images/subsets/<subset>/cameras/<solve>/sparse/0/images.bin under the scene."""
    import spine_register as sreg
    scene_dir = v4.STORE / args.scene
    spine_json = scene_dir / "spine" / args.spine / "spine.json"
    if not spine_json.exists():
        sys.exit(f"no spine {args.spine} at {scene_dir}/spine (run `spine` first)")

    seg_solves, manifest = {}, {}
    for tok in args.solves.split(","):
        if "=" not in tok:
            sys.exit(f"--solves entry '{tok}' must be seg=subset/solve")
        seg, loc = tok.split("=", 1)
        sub, _, sol = loc.partition("/")
        if not sol:
            sys.exit(f"--solves entry '{tok}': need subset/solve")
        bin_p = scene_dir / "images" / "subsets" / sub / "cameras" / sol / "sparse" / "0" / "images.bin"
        if not bin_p.exists():
            sys.exit(f"no solve images.bin for segment {seg} at {bin_p}")
        seg_solves[seg] = bin_p
        manifest[seg] = loc

    settings = v4.hashable_settings(v4.tasks()["spine-register"], {"rel_tol": args.rel_tol})
    rid = v4.identity_hash({"spine": args.spine, "solves": manifest}, settings, "spine-register@0")
    rdir = scene_dir / "spine" / args.spine / "register" / rid
    if (rdir / "metadata.json").exists():
        print(f"[spine-register] NOOP — {rid} exists -> {rdir}")
        return

    nodes = sreg.nodes_from_solves(seg_solves)
    out = sreg.register(nodes, rel_tol=args.rel_tol)
    rdir.mkdir(parents=True, exist_ok=True)
    (rdir / "global.json").write_text(json.dumps(out, indent=2) + "\n")
    v4.write_metadata(rdir, task="spine-register", algo="spine-register@0", identity=rid,
                      resolved_inputs={"spine": args.spine, "solves": manifest},
                      settings=settings, mechanism="local",
                      measured={"n_segments": out["n_segments"], "n_edges": out["n_edges"],
                                "n_cameras": out["n_cameras"], "converged": out["converged"],
                                "iters_run": out["iters_run"], "within_tol": out["within_tol"],
                                "max_seam_residual_rel": out["max_seam_residual_rel"]})
    flag = "" if out["within_tol"] else "  ** EXCEEDS TOL **"
    print(f"[spine-register] {out['n_segments']} segments, {out['n_cameras']} cameras -> one gauge "
          f"| max seam residual {out['max_seam_residual_rel']*100:.3f}% of spread "
          f"(within_tol={out['within_tol']}{flag}) | converged={out['converged']} "
          f"({out['iters_run']} it) -> {rdir}")
    if not out["within_tol"]:
        worst = max(out["seams"], key=lambda e: e["residual_max"])
        print(f"[spine-register] WARNING: worst seam {worst['i']}<->{worst['j']} "
              f"({worst['type']}) residual {worst['residual_rel']*100:.2f}% — a segment may be "
              f"mis-solved or under-overlapped; inspect before fusion (STO-SCN-099).")
    job_record(args.scene, "spine-register", [{"node": "spine-register", "identity": rid,
                                              "within_tol": out["within_tol"]}],
               {"scene": args.scene, "spine": args.spine, "solves": manifest})


# ============================================================ spine cohesive fusion (STO-SCN-099)

def cmd_spine_fuse(args):
    """Cohesive fusion of per-segment gaussians into one gauge (spine-fuse@0). Transforms
    each segment's reconstruction .ply by its 098 global gauge, confidence-weights the
    overlaps (camera-coverage feather) so no doubled walls, concatenates into one .ply
    for STO-SCN-013. Pure-numpy + scipy; runs locally. Idempotent NOOP.

    --register: the spine-register@0 identity (its global.json holds gauges + global cams).
    --solves:   seg=subset/solve list (to map each segment's cameras to global centers).
    --gaussians: seg=<ply-path-under-store> list (per-segment reconstruction gaussians)."""
    import spine_fuse as sfuse
    scene_dir = v4.STORE / args.scene
    gj = scene_dir / "spine" / args.spine / "register" / args.register / "global.json"
    if not gj.exists():
        sys.exit(f"no register {args.register} at {gj} (run `spine-register` first)")
    glob = json.loads(gj.read_text())
    gauges = glob["gauges"]
    cam_global = {n: c["center"] for n, c in glob["cameras"].items()}

    def _parse(spec):
        out = {}
        for tok in spec.split(","):
            k, _, v = tok.partition("=")
            if not v:
                sys.exit(f"bad manifest entry '{tok}' (need seg=value)")
            out[k] = v
        return out

    solves = _parse(args.solves)
    plys = _parse(args.gaussians)
    if set(solves) != set(gauges) or set(plys) != set(gauges):
        sys.exit(f"segment keys must match the register gauges {sorted(gauges)}; "
                 f"got solves={sorted(solves)} gaussians={sorted(plys)}")

    import spine_register as sreg
    settings = v4.hashable_settings(v4.tasks()["spine-fuse"], {"radius": args.radius})
    fid = v4.identity_hash({"register": args.register, "gaussians": plys}, settings, "spine-fuse@0")
    fdir = scene_dir / "spine" / args.spine / "fuse" / fid
    if (fdir / "metadata.json").exists():
        print(f"[spine-fuse] NOOP — {fid} exists -> {fdir}")
        return

    import scout_register
    segments = {}
    for k in gauges:
        sub, _, sol = solves[k].partition("/")
        bin_p = scene_dir / "images" / "subsets" / sub / "cameras" / sol / "sparse" / "0" / "images.bin"
        seg_cam_names = sreg.read_solve_poses(bin_p).keys()
        cams = [cam_global[n] for n in seg_cam_names if n in cam_global]
        ply_p = scene_dir / plys[k] if not plys[k].startswith("/") else Path(plys[k])
        # TWO-STAGE gauge (STO-SCN-105 + STO-SCN-098): the DA3 gaussian lives in DA3's
        # NORMALIZED frame, off from its segment solve by a full similarity (scale + ~125°
        # rotation + translation). Register gs->segment-solve via the scout_gauge (105),
        # THEN segment-solve->global via the 098 gauge. Skipping the 105 step leaves the
        # splat mis-oriented vs the frustums — the bug operator-caught 2026-06-14.
        g105 = scout_register.gauge_for(ply_p.parent)
        inner = {"scale": g105["scale"], "R": sfuse.quat_xyzw_to_R(g105.get("quat", [0, 0, 0, 1])),
                 "t": g105.get("translate", [0, 0, 0])}
        outer = {"scale": gauges[k]["scale"], "R": gauges[k]["R"], "t": gauges[k]["t"]}
        composed = sfuse.compose_gauge(outer, inner)
        if not g105.get("registered"):
            print(f"[spine-fuse] WARNING: segment {k} gaussian has no scout_gauge (105) — "
                  f"applying 098 only; splat may be mis-oriented. Re-run scout for {ply_p.parent}.")
        g = sfuse.transform_gaussians(sfuse.read_ply(ply_p), composed)
        segments[k] = {"gaussians": g, "cameras": cams}

    radius = args.radius if args.radius and args.radius > 0 else None
    fused = sfuse.fuse(segments, radius=radius)
    fdir.mkdir(parents=True, exist_ok=True)
    sfuse.write_ply(fdir / "fused.gs.ply", fused)
    n_in = {k: int(len(segments[k]["gaussians"])) for k in segments}
    v4.write_metadata(fdir, task="spine-fuse", algo="spine-fuse@0", identity=fid,
                      resolved_inputs={"register": args.register, "gaussians": plys},
                      settings=settings, mechanism="local",
                      measured={"n_segments": len(segments), "n_in": n_in,
                                "n_total_in": sum(n_in.values()), "n_fused": int(len(fused))})
    print(f"[spine-fuse] {len(segments)} segments, {sum(n_in.values())} gaussians -> "
          f"{len(fused)} fused (overlaps cross-faded) -> {fdir}")
    job_record(args.scene, "spine-fuse", [{"node": "spine-fuse", "identity": fid}],
               {"scene": args.scene, "spine": args.spine, "register": args.register})


# ============================================================ scout gaussian (STO-SCN-095)

def cmd_scout(args):
    """DA3 da3@1 scout gaussian for the verify surface — in the FastMap solve gauge
    (posed.json from the solve's sparse/0) so 094's proposed-N frustums overlay.
    Progress -> MQTT via run_scout.sh. Fisheye undistorted to pinhole first."""
    import select_views as selv
    import posed_from_sparse as pfs
    scene_dir = v4.STORE / args.scene
    sc = v4.Scene(args.scene)
    subset = args.subset or sc.resolve("primary")
    sdir = scene_dir / "images" / "subsets" / subset / "cameras" / args.solve
    sparse = sdir / "sparse" / "0"
    if not (sparse / "images.bin").exists():
        sys.exit(f"no solve sparse/0 at {sdir} (run `solve` first)")
    make, model, mode, _modality = _read_capture_decl(scene_dir)

    # View selection. DEFAULT `track` div_angle=0 = COHERENT/overlapping views: DA3 fuses a
    # clean gaussian only from coherent views; viewpoint-diversity spreads views apart and can
    # nebula. `voxel` (STO-SCN-103) builds the gaussian from the actual coverage-SELECTED N
    # (what the FINAL-N reconstructs) — for verifying the selection, even if less coherent.
    sel = getattr(args, "selector", "track")
    if sel == "voxel":
        import voxel_coverage as vc
        _, rep = vc.select_from_sparse(str(sparse), args.n_scout, grid=args.grid)
    else:
        _, rep = selv.select_from_sparse(str(sparse), args.n_scout, div_angle=0)
    names = rep["selected"]
    posed = pfs.posed_from_sparse(str(sparse), names)
    settings = v4.hashable_settings(v4.tasks()["scout"],
                                    {"n_scout": args.n_scout, "res": args.res,
                                     "selector": sel, "grid": args.grid})
    cid = v4.identity_hash({"solve": args.solve}, settings, "scout@0")
    cdir = sdir / "scout" / cid
    if (cdir / "metadata.json").exists():
        print(f"[scout] NOOP — scout {cid} exists -> {cdir}")
        return

    tag = f"{args.scene}-scout-{cid}"
    work = f"{SCRATCH}/{tag}"
    here = Path(__file__).parent
    name2img = {}
    for md in scene_dir.glob("images/*/metadata.json"):
        d = json.loads(md.read_text())
        img = next((p for p in md.parent.glob("image.*")), None)
        if img and d.get("original_name") in names:
            name2img[d["original_name"]] = img
    tmp = Path("/tmp") / f"v4scout-{tag}"
    shutil.rmtree(tmp, ignore_errors=True)
    (tmp / "images").mkdir(parents=True); (tmp / "cameras").mkdir()
    for nm in names:
        if nm in name2img:
            shutil.copy2(name2img[nm], tmp / "images" / nm)
    (tmp / "cameras" / "posed.json").write_text(json.dumps(posed, indent=2))
    shutil.copy2(here / "da3_infer_posed.py", tmp / "da3_infer_posed.py")
    sh(["ssh", args.host, f"rm -rf {work} && mkdir -p {work}"])
    sh(["rsync", "-a", f"{tmp}/", f"{args.host}:{work}/"])
    sh(["rsync", "-a", str(here.parent / "images" / "fastmap" / "run_scout.sh"),
        str(here / "lib_progress.sh"), str(here / "capture_profiles.json"),
        f"{args.host}:{work}/"])           # stage the CURRENT registry (self-deploy; no rebuild)
    shutil.rmtree(tmp)

    remote = (f"cd {work} && KRABBY_FASTMAP_IMAGE={json.dumps(FASTMAP_IMAGE)} "
              f"KRABBY_DA3_IMAGE={json.dumps(DA3_IMAGE)} bash run_scout.sh "
              f"{work}/images {work}/scout_out {json.dumps(make)} {json.dumps(model)} "
              f"{json.dumps(mode)} {args.res}")
    print(f"[scout] {args.host}: da3@1 scout on {len(names)} views — progress -> MQTT")
    t0 = datetime.datetime.now()
    r = subprocess.run(["ssh", args.host, remote], capture_output=True, text=True)
    cdir.mkdir(parents=True, exist_ok=True)
    (cdir / "scout.log").write_text(r.stdout[-200000:] + "\n--- stderr ---\n" + r.stderr[-50000:])
    dt = int((datetime.datetime.now() - t0).total_seconds())
    # Fault-tolerant: detect a failed run BEFORE the gather, so the operator sees
    # the REAL error (scout.log + stderr tail) instead of an opaque "rsync:
    # change_dir scout_out failed: No such file" when scout_out was never made.
    # Keep the host workdir for diagnosis (do NOT clean up on failure).
    if r.returncode != 0:
        sys.exit(f"[scout] FAILED on {args.host} (rc={r.returncode}) — see {cdir}/scout.log "
                 f"(workdir kept for diagnosis: {args.host}:{work}).\n"
                 f"--- last error ---\n{(r.stderr or r.stdout)[-1500:]}")
    # gather the gs_ply (lands in a gs_ply/ subdir — recurse with */ include),
    # DA3's `scout_gauge.json` (the scale_factor that maps the gaussian back
    # into the solve gauge — STO-SCN-105: this is THE registration; the npz
    # extrinsics are echoed INPUT and align to identity), the colmap export
    # (DA3 output cameras, a cross-check) and the npz (kept for provenance).
    # Without scout_gauge.json the splat sits ~scale_factor× off the frustums
    # (the operator-observed "cameras too high / wrong scale"). do NOT remove
    # the workdir until the gather is confirmed.
    sh(["rsync", "-a", "--include=*/", "--include=*.ply", "--include=*.npz",
        "--include=scout_gauge.json", "--include=*.bin", "--include=*.txt",
        "--exclude=*", f"{args.host}:{work}/scout_out/", str(cdir) + "/"])
    ply = next((p for p in cdir.rglob("*.ply")), None)
    if r.returncode != 0 or ply is None:
        sys.exit(f"[scout] FAILED (rc={r.returncode}; see {cdir}/scout.log; "
                 f"workdir kept for diagnosis: {args.host}:{work})")
    ply.replace(cdir / "scout.gs.ply")        # move up out of the gs_ply/ subdir
    npz = next((p for p in cdir.rglob("results.npz")), None)
    if npz is not None:
        npz.replace(cdir / "da3_poses.npz")   # DA3 output extrinsics (echoed input; provenance)
    gj = next((p for p in cdir.rglob("scout_gauge.json")), None)
    scale_factor = None
    if gj is not None:
        try:
            scale_factor = json.loads(gj.read_text()).get("scale_factor")
        except (ValueError, OSError):
            pass
        if gj != cdir / "scout_gauge.json":
            gj.replace(cdir / "scout_gauge.json")
    else:
        print("[scout] WARNING: no scout_gauge.json gathered — the verify "
              "surface cannot auto-register the splat to the solve gauge "
              "(re-run with the updated da3_infer_posed.py; STO-SCN-105).")
    sh(["ssh", args.host, f"rm -rf {work}"])
    (cdir / "posed.json").write_text(json.dumps(posed, indent=2) + "\n")
    (cdir / "scout_views.json").write_text(json.dumps({"n": len(names), "views": names}, indent=2) + "\n")
    v4.write_metadata(cdir, task="scout", algo="scout@0", identity=cid,
                      resolved_inputs={"solve": args.solve}, settings=settings, mechanism="job",
                      measured={"host": args.host.split("@")[-1], "duration_s": dt,
                                "n_views": len(names), "scale_factor": scale_factor})
    print(f"[scout] done in {dt}s: scout.gs.ply ({len(names)} views) -> {cdir}")
    job_record(args.scene, "scout", [{"node": "scout", "identity": cid}],
               {"scene": args.scene, "subset": subset, "solve": args.solve})


def posed_sparse_to_cameras_json(sparse_dir: Path, out_path: Path) -> int:
    """STO-SCN-129: emit a renderable + posable cameras.json from a FastMap sparse/0.

    `focals` are written in the **SOLVE_IMAGE_SIZE (512) long-side convention** that
    `colmap_posed.solve_entries` (matcha@1 posed weld) AND `build_blender_scene` expect:
    `f_512 = f_native * 512 / solve_long_side`. Writing the *native* solve-pixel focal instead
    (e.g. 1145 in a 3840-px solve) makes `solve_entries` rescale it ~7.5x → wrong intrinsics →
    matcha charts NaN (root-caused on matcha-15, STO-SCN-127/130). Idempotent; returns count."""
    import numpy as np
    import posed_from_sparse as pfs
    from colmap_posed import SOLVE_IMAGE_SIZE
    sp = Path(sparse_dir)
    intr = pfs.read_cameras_intrinsics(sp / "cameras.bin")     # cam_id -> {fx,fy,cx,cy,w,h}
    fps, c2ws, focals = [], [], []
    for im in pfs.read_images_w2c(sp / "images.bin"):          # {name, camera_id, w2c}
        ci = intr.get(im["camera_id"]) or next(iter(intr.values()))
        long_side = max(ci["w"], ci["h"]) or 1
        w2c = np.asarray(im["w2c"], dtype=np.float64)
        if w2c.shape == (3, 4):
            w2c = np.vstack([w2c, [0, 0, 0, 1]])
        fps.append(im["name"])
        c2ws.append(np.linalg.inv(w2c).tolist())
        focals.append(ci["fx"] * SOLVE_IMAGE_SIZE / long_side)
    out_path.write_text(json.dumps(
        {"filepaths": fps, "cams2world": c2ws, "focals": focals}, indent=2) + "\n")
    return len(fps)


def resolve_pose_source(scene_dir: Path, final_subset: str):
    """STO-SCN-130: a FINAL-N selection (member-only subset, no `cameras/`) has no solve of its
    own — find the PARENT (subset, solve) it was selected from, via the `select` node whose
    final.json `final_subset` == this subset. Returns (parent_subset, solve) or (None, None)."""
    for fj in scene_dir.glob("images/subsets/*/cameras/*/select/*/final.json"):
        try:
            if json.loads(fj.read_text()).get("final_subset") == final_subset:
                # .../subsets/<parent_sub>/cameras/<solve>/select/<id>/final.json
                return fj.parents[4].name, fj.parents[2].name
        except (OSError, json.JSONDecodeError):
            continue
    return None, None


# ============================================================ reconstruct-matcha

def cmd_matcha(args):
    import numpy as np
    scene_dir = v4.STORE / args.scene
    sc = v4.Scene(args.scene)
    sub = getattr(args, "subset", None) or sc.resolve("primary")   # STO-SCN-130: explicit subset override
    tdefs = v4.tasks()
    nodes = []
    r_settings = v4.hashable_settings(tdefs["represent-via-matcha"],
                                      {"dense_regul": args.dense_regul})
    solve_dirs = sorted((scene_dir / "images" / "subsets" / sub / "cameras").glob("*/"))
    solve_dirs = [d for d in solve_dirs if (d / "metadata.json").exists()]
    # STO-SCN-130: `sub` is the reconstructed member set (staged + counted + identity);
    # `pose_sub` is the subset that holds the SOLVE we pose from. They differ when primary is a
    # FINAL-N selection (member-only, no cameras/) — then pose its members from the PARENT solve
    # (matcha@1 posed; no re-solve, no new gauge). solve_to_sparse restricts to the members.
    if solve_dirs:
        pose_sub, sid = sub, solve_dirs[0].name
    else:
        pose_sub, sid = resolve_pose_source(scene_dir, sub)
        if sid is None:
            sys.exit(f"no solve for primary {sub} — not a solved subset and no parent select "
                     f"node found (run solve, or select from a solved pool first)")
        print(f"[matcha] primary {sub} is a FINAL-N selection — posing its members "
              f"from parent solve {pose_sub}/{sid}")
    # matcha@0: unposed weld — train.py re-solves cameras, minting its own
    #           gauge (composed out via gauge-sim, STO-SCN-089-3).
    # matcha@1: POSED weld — the ingest solve is fed to train.py as COLMAP
    #           sparse/0 (fix_rotation/translation + align_camera_locations),
    #           so the weld gauge IS the ingest gauge. Kills the gauge-sim
    #           class at the root; the sim becomes a verification gate.
    #           Forced by 003-firepit, whose re-solve disagrees with the
    #           ingest solve beyond any similarity (3.1-3.9%).
    m_algo = "matcha@1" if args.sfm == "posed" else "matcha@0"
    rid = v4.identity_hash({"subset": sub, "cameras": sid}, r_settings, m_algo)
    rdir = scene_dir / "represent" / "matcha" / rid
    o_settings, o_algo = ORIENT_SETTINGS, ORIENT_ALGO
    # the orient reads the bootstrap mesh (z-floor) -> it IS a resolved input
    oid = v4.identity_hash({"solve": sid, "bootstrap_rep": rid}, o_settings, o_algo)
    # gauge is part of the mesh content -> orient is a resolved input of meshify
    mid = v4.identity_hash({"representation": rid, "cameras": sid, "orient": oid},
                           {}, "tetra-extract@1")
    # STO-SCN-133: mesh_res is tunable (default 1024 OOMs on small-radius spine gauges; 512
    # fits a 31 GB host). Override flows into identity so 512/1024 are distinct store nodes.
    ts_overrides = {"mesh_res": args.mesh_res} if getattr(args, "mesh_res", None) else {}
    ts_settings = v4.hashable_settings(tdefs["meshify-via-tsdf"], ts_overrides)
    tid = v4.identity_hash({"representation": rid, "cameras": sid, "orient": oid},
                           ts_settings, "tsdf-extract@1")
    tetra_dir = rdir / "meshify" / "tetra" / mid
    tsdf_dir = rdir / "meshify" / "tsdf" / tid

    if (tsdf_dir / "mesh.ply").exists() and (
            (tetra_dir / "mesh.ply").exists()
            or not (rdir / "out" / "tetra_meshes").is_dir()
            or not list((rdir / "out" / "tetra_meshes").glob("*.ply"))):
        if not (rdir / "metadata.json").exists():
            # crash-recovery: meshes landed but the rep record didn't (008)
            v4.write_metadata(rdir, task="represent-via-matcha", algo=m_algo,
                              identity=rid,
                              resolved_inputs={"subset": sub, "cameras": sid},
                              settings=r_settings, mechanism="job",
                              measured={"recovered": True})
            md = json.loads((rdir / "metadata.json").read_text())
            md["canonical_gauge"] = str(
                (scene_dir / "images" / "subsets" / pose_sub / "cameras" / sid /
                 "orient" / oid / "oriented.json").relative_to(scene_dir))
            (rdir / "metadata.json").write_text(json.dumps(md, indent=2) + "\n")
        print(f"NOOP: {rid} fully materialized")
        return

    # -- the @0 weld: ONE dispatch materializes represent + raw tetra + raw tsdf.
    # Raw outputs are gauge-independent: when they already exist (e.g. an
    # orient revision), the GPU run is skipped and only orient+ground rerun.
    out = rdir / "out"
    tetra_raw = sorted((out / "tetra_meshes").glob("*.ply")) if (out / "tetra_meshes").is_dir() else []
    tsdf_raw = next(iter(sorted((out / "tsdf_meshes").glob("multires_tsdf_post*.ply"))), None)
    if not tetra_raw or tsdf_raw is None:
        tag = f"{args.scene}-matcha-{rid}"
        workdir = stage_images_on_host(args.host, scene_dir, sub, tag)
        n_images = len(json.loads((scene_dir / "images" / "subsets" / sub / "subset.json")
                                  .read_text())["members"])
        if args.sfm == "posed":
            # mint sparse/0 from the ingest solve and stage it next to images/
            from colmap_posed import solve_to_sparse
            tmp_sparse = Path("/tmp") / f"v4exec-{tag}-sparse"
            shutil.rmtree(tmp_sparse, ignore_errors=True)
            members = json.loads((scene_dir / "images" / "subsets" / sub /
                                  "subset.json").read_text())["members"]
            by_hash = {p.parent.name: p for p in scene_dir.glob("images/*/image.*")}
            staged = {}                       # staged NAME -> local store path
            for h in members:
                d = json.loads((scene_dir / "images" / h / "metadata.json").read_text())
                staged[d.get("original_name", h + ".jpg")] = by_hash[h]
            covered = solve_to_sparse(
                scene_dir / "images" / "subsets" / pose_sub / "cameras" / sid / "cameras.json",
                staged, tmp_sparse / "0")
            sh(["ssh", args.host, f"mkdir -p {workdir}/sparse"])
            sh(["rsync", "-a", str(tmp_sparse) + "/", f"{args.host}:{workdir}/sparse/"])
            shutil.rmtree(tmp_sparse)
            print(f"[{m_algo} weld] sparse/0 minted from solve {sid} "
                  f"({len(covered)} posed cameras)")
            src_arg, sfm_cfg = "/work", "posed"
        else:
            src_arg, sfm_cfg = "/work/images", "unposed"
        tool = (f"python train.py -s {src_arg} -o /work/out --sfm_config {sfm_cfg} "
                f"--n_images {n_images} --alignment_config strong")
        if args.dense_regul != "default":
            tool += f" --dense_regul {args.dense_regul}"
        tool += (" --depthanythingv2_checkpoint_dir /opt/MAtCha/Depth-Anything-V2/checkpoints"
                 " --depthanything_encoder vitl")
        # TSDF extract. Default config (mesh_res 1024) via extract_tsdf_mesh.py; a `--mesh-res`
        # override calls render_multires.py directly with the same args (the configs are baked,
        # can't drop a new yaml) — it produces the same multires_tsdf_post.ply (STO-SCN-133).
        if getattr(args, "mesh_res", None):
            tsdf = (f"python 2d-gaussian-splatting/render_multires.py "
                    f"--source_path /work/out/mast3r_sfm --model_path /work/out/free_gaussians "
                    f"--output_dir /work/out/tsdf_meshes --depth_ratio 1.0 --num_cluster 50 "
                    f"--mesh_res {int(args.mesh_res)} --multires_factors 2 8 16 "
                    f"--skip_train --skip_test")
        else:
            tsdf = ("python scripts/extract_tsdf_mesh.py -s /work/out/mast3r_sfm "
                    "-m /work/out/free_gaussians -o /work/out/tsdf_meshes -c default")
        tool += " && PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True " + tsdf
        print(f"[{m_algo} weld] {args.host}: full pipeline ({n_images} images, sfm={args.sfm}, "
              f"dense_regul={args.dense_regul}, mesh_res={getattr(args,'mesh_res',None) or 1024}) "
              f"— ~15-25 min")
        t0 = datetime.datetime.now()
        rc = run_in_matcha(args.host, workdir, tool, rdir / "matcha.log")
        dt = int((datetime.datetime.now() - t0).total_seconds())
        print(f"[{m_algo} weld] rc={rc} in {dt}s; gathering…")
        rdir.mkdir(parents=True, exist_ok=True)
        sh(["rsync", "-a", f"{args.host}:{workdir}/out/", str(rdir / "out") + "/"])
        sh(["ssh", args.host, f"rm -rf {SCRATCH}/{tag}"])
        tetra_raw = sorted((out / "tetra_meshes").glob("*.ply")) if (out / "tetra_meshes").is_dir() else []
        tsdf_raw = next(iter(sorted((out / "tsdf_meshes").glob("multires_tsdf_post*.ply"))), None)
        if rc != 0 or tsdf_raw is None:
            sys.exit(f"matcha weld FAILED (rc={rc}; tetra={bool(tetra_raw)} tsdf={bool(tsdf_raw)}; "
                     f"log {rdir}/matcha.log)")
        if not tetra_raw:
            print(f"[{m_algo} weld] WARNING: tetra stage produced no mesh "
                  "(marching_tetrahedra flake, 010 class) — continuing tsdf-only")
        dig, _ = host_digest(args.host, MATCHA_IMAGE)
        v4.write_metadata(rdir, task="represent-via-matcha", algo=m_algo, identity=rid,
                          resolved_inputs={"subset": sub, "cameras": sid},
                          settings=r_settings, mechanism="job",
                          measured={"host": args.host.split("@")[-1], "duration_s": dt,
                                    "image_digest": dig})
        nodes.append({"node": "represent", "identity": rid, "action": "EXECUTE",
                      "host": args.host, "duration_s": dt})
    else:
        if not (rdir / "metadata.json").exists():
            # recovered weld output (gather completed out-of-band, e.g. host
            # dropped mid-rsync) — mint the rep record now
            v4.write_metadata(rdir, task="represent-via-matcha", algo=m_algo,
                              identity=rid,
                              resolved_inputs={"subset": sub, "cameras": sid},
                              settings=r_settings, mechanism="job",
                              measured={"recovered": True})
        nodes.append({"node": "represent", "identity": rid, "action": "NOOP"})

    # -- gauge-sim: the weld's FULL train.py re-solves cameras internally,
    # minting its OWN arbitrary gauge (measured 136.5 deg / scale 0.9988 vs
    # the ingest solve on 009, camera residual 0.0009 — STO-SCN-089-3).
    # The raw meshes live in the WELD frame; everything else in the store
    # lives in the INGEST-solve frame. Compose the exact similarity
    # (the dtu STO-SCN-041 recipe, now in-graph).
    sys.path.insert(0, str(Path(__file__).parent))
    import open3d as o3d  # noqa: F401  (ensures availability before work)
    import numpy as np
    from gauge_align import align_camera_sets
    sim = weld_to_solve_sim(scene_dir, pose_sub, sid, out)
    print(f"[gauge-sim] weld->solve: scale {sim['s']:.4f} rot {sim['rot_deg']:.1f} deg "
          f"max residual {sim['max_residual']:.4f}")
    if args.sfm == "posed":
        # posed weld: the sim is a VERIFICATION, not a correction — the weld
        # ran in the ingest gauge by construction, so anything beyond numeric
        # noise means the posed path didn't do its job (fail loudly, T-003).
        if sim["residual_frac"] > 0.005 or sim["rot_deg"] > 2.0:
            sys.exit(f"matcha@1 REFUSED: posed weld drifted from ingest gauge "
                     f"(residual {sim['residual_frac']:.2%}, rot "
                     f"{sim['rot_deg']:.2f} deg) — posed sfm not honored")
    elif sim["residual_frac"] > 0.02:
        sys.exit(f"matcha REFUSED: weld->solve camera similarity residual "
                 f"{sim['residual_frac']:.1%} > 2% — solves disagree beyond gauge")

    # -- node: orient-cameras (horizon up prior; z-floor from the SOLVE-framed mesh)
    odir = scene_dir / "images" / "subsets" / pose_sub / "cameras" / sid / "orient" / oid
    if (odir / "oriented.json").exists():
        g = json.loads((odir / "oriented.json").read_text())
        R, z = np.asarray(g["rotation"]), float(g["z_shift"])
        nodes.append({"node": "orient", "identity": oid, "action": "NOOP"})
    else:
        cams = json.loads((scene_dir / "images" / "subsets" / pose_sub / "cameras" / sid /
                           "cameras.json").read_text())
        # STO-SCN-130: orient from the RECONSTRUCTED subset's cameras, not the parent pool.
        # When pose_sub != sub (a FINAL-N selection), `cameras.json` holds the whole pool — the
        # full handheld walk has mixed roll (matcha-15: 27.7° > 15° → orient REFUSED). Restrict
        # to `sub`'s members so the horizon/floor match the mesh's own cameras (== historical
        # behavior, where the subset cameras.json already held only the members).
        member_stems = set()
        for h in json.loads((scene_dir / "images" / "subsets" / sub / "subset.json")
                            .read_text())["members"]:
            d = json.loads((scene_dir / "images" / h / "metadata.json").read_text())
            member_stems.add(d.get("original_name", h).rsplit(".", 1)[0])
        keep = [i for i, fp in enumerate(cams["filepaths"])
                if fp.rsplit("/", 1)[-1].rsplit(".", 1)[0] in member_stems]
        c2w = np.asarray(cams["cams2world"])
        if keep and len(keep) < len(c2w):
            print(f"[orient] restricting to {len(keep)} reconstructed members "
                  f"(of {len(c2w)} pool cameras)")
            c2w = c2w[keep]
        raw = o3d.io.read_triangle_mesh(str(tsdf_raw))
        v_solve = sim["s"] * (np.asarray(raw.vertices) @ np.asarray(sim["R"]).T) + np.asarray(sim["t"])
        R, z = bootstrap_orient(v_solve, cam_R_c2w=c2w[:, :3, :3], cam_C=c2w[:, :3, 3])
        odir.mkdir(parents=True, exist_ok=True)
        (odir / "transform.json").write_text(json.dumps({"rotation": R.tolist(),
                                                         "z_shift": float(z)}, indent=2) + "\n")
        # oriented cameras file (renderer contract: rotation + z_shift + cams)
        (odir / "oriented.json").write_text(json.dumps({"rotation": R.tolist(),
                                                        "z_shift": float(z)}, indent=2) + "\n")
        v4.write_metadata(odir, task="orient-cameras", algo=o_algo, identity=oid,
                          resolved_inputs={"solve": sid, "bootstrap_rep": rid},
                          settings=o_settings, mechanism="job")
        nodes.append({"node": "orient", "identity": oid, "action": "EXECUTE"})

    # -- meshify: weld frame -> solve frame (gauge-sim) -> canonical gauge
    targets = [(tsdf_raw, tsdf_dir, "meshify-via-tsdf", "tsdf-extract@1", ts_settings)]
    if tetra_raw:
        targets.insert(0, (tetra_raw[-1], tetra_dir, "meshify-via-tetra", "tetra-extract@1", {}))
    for src_mesh, mdir, task, algo, msettings in targets:
        mdir.mkdir(parents=True, exist_ok=True)
        ground_mesh(src_mesh, mdir / "mesh.ply", R, z, sim=sim)
        v4.write_metadata(mdir, task=task, algo=algo, identity=mdir.name,
                          resolved_inputs={"representation": rid, "cameras": sid,
                                           "orient": oid},
                          settings=msettings, mechanism="job",
                          extra={"gauge": str(odir.relative_to(scene_dir)),
                                 "gauge_sim": {k: sim[k] for k in
                                               ("s", "R", "t", "rot_deg", "max_residual")}})
        nodes.append({"node": task, "identity": mdir.name, "action": "EXECUTE"})
    # canonical gauge marker for the renderer
    md = json.loads((rdir / "metadata.json").read_text())
    md["canonical_gauge"] = str((odir / "oriented.json").relative_to(scene_dir))
    (rdir / "metadata.json").write_text(json.dumps(md, indent=2) + "\n")
    job_record(args.scene, "reconstruct-matcha", nodes,
               {"scene": args.scene, "host": args.host, "dense_regul": args.dense_regul,
                "sfm": args.sfm})
    print(f"reconstruct-matcha materialized: represent {rid}, tetra {mid}, tsdf {tid}, orient {oid}")


def camera_up_horizon(R_c2w):
    """World-up from the HORIZON constraint (orient-floor@1).

    The operator-guided camera-up prior in its correct form. Photographers
    pitch freely (path shots pitch down 30-40 deg — measured on 009, which
    biases naive mean(-Y) by exactly that much, the @0-era failure) but
    keep the horizon LEVEL: the image-right axis X_i is horizontal in every
    shot (009: all seven X_i within 3.4 deg of horizontal). So true up
    satisfies u . X_i ~ 0 for all i.

      u = eigvec_min( sum X_i X_i^T )   — pitch never enters.

    A walk with any turn makes the solution unique; for a perfectly
    straight walk the null space is 2D and mean(-Y) projected onto it
    supplies the in-plane choice (pitch-corrected). Portrait captures:
    the same constraint lives on Y_i — both axis sets are tried, the one
    with the more consistent horizon (smaller min eigenvalue) wins.
    Returns (u, quality) with quality = max residual |axis . u| in deg."""
    import numpy as np
    R_c2w = np.asarray(R_c2w)
    n = len(R_c2w)
    best = None
    for ax in (0, 1):                       # landscape: X horizon; portrait: Y
        A = R_c2w[:, :, ax]                 # (N,3) the candidate horizon axes
        M = (A[:, :, None] * A[:, None, :]).sum(0) / n
        w, V = np.linalg.eigh(M)
        null = V[:, w < max(0.1, 1.5 * w[0])]   # small-eigval subspace (>=1 dim)
        mean_up = -R_c2w[:, :, 1 - ax].mean(0)  # the other axis ~ up-ish
        u = null @ (null.T @ mean_up)
        if np.linalg.norm(u) < 1e-6:            # mean-up orthogonal: take eigvec
            u = V[:, 0]
        u = u / np.linalg.norm(u)
        if u @ mean_up < 0:
            u = -u
        resid = float(np.degrees(np.arcsin(np.clip(np.abs(A @ u).max(), 0, 1))))
        if best is None or resid < best[1]:
            best = (u, resid)
    return best


def bootstrap_orient(mesh_or_verts, cam_R_c2w=None, cam_C=None):
    """Floor fit on the dense TSDF mesh -> (R 3x3, z_shift), z-up gauge.

    orient-floor@1: multi-plane RANSAC + camera-consensus up prior +
    cameras-above-floor check. @0 (largest plane, no prior) picked the
    gate/hedge wall as 'floor' on 009's corridor capture and rolled the
    whole gauge 90 deg (STO-SCN-089 follow-on, operator-caught)."""
    import numpy as np
    import open3d as o3d
    if isinstance(mesh_or_verts, (str, Path)):
        v = np.asarray(o3d.io.read_triangle_mesh(str(mesh_or_verts)).vertices)
    else:
        v = np.asarray(mesh_or_verts)
    if cam_R_c2w is None:
        sys.exit("orient REFUSED: orient-floor@1 requires camera rotations "
                 "(horizon constraint) — none provided")
    n, resid = camera_up_horizon(cam_R_c2w)
    print(f"[orient@1] horizon up: {np.round(n, 3).tolist()} "
          f"(max horizon residual {resid:.1f} deg)")
    if resid > 15.0:
        sys.exit(f"orient REFUSED: horizon inconsistent across cameras "
                 f"({resid:.1f} deg > 15) — mixed-roll capture? inspect")
    z_axis = np.array([0.0, 0.0, 1.0])
    vv = np.cross(n, z_axis)
    s = np.linalg.norm(vv)
    if s < 1e-9:
        R = np.eye(3)
    else:
        c_ = float(n @ z_axis)
        vx = np.array([[0, -vv[2], vv[1]], [vv[2], 0, -vv[0]], [-vv[1], vv[0], 0]])
        R = np.eye(3) + vx + vx @ vx * ((1 - c_) / (s ** 2))
    vz = (v[::max(1, len(v) // 200_000)] @ R.T)
    if cam_C is not None:
        # floor level = the ground UNDER the camera path (sloped scenes have
        # geometry far below the global percentile, down the hill)
        cz = np.asarray(cam_C) @ R.T
        d2 = ((vz[:, None, :2] - cz[None, :, :2]) ** 2).sum(-1).min(1)
        near = vz[d2 < 2.5 ** 2]
        if len(near) < 200:
            near = vz
        z_floor = float(np.percentile(near[:, 2], 5))
        h = cz[:, 2] - z_floor
        print(f"[orient@1] camera heights above local floor: {np.round(h, 2).tolist()} "
              f"(solve units — SfM scale is arbitrary, no metric gate)")
        rel = float(np.std(h) / max(abs(np.median(h)), 1e-9))
        if np.median(h) <= 0:
            sys.exit(f"orient REFUSED: cameras BELOW the floor (median height "
                     f"{np.median(h):.2f}) — up sign wrong")
        if rel > 0.35:
            # multi-ring orbits (dtu) legitimately vary in height — the
            # horizon residual is the up validator; this is informational
            print(f"[orient@1] note: height band loose (rel spread {rel:.2f}) — "
                  f"multi-elevation capture; horizon residual is the gate")
    else:
        z_floor = float(np.percentile(vz[:, 2], 2))
    return R, -z_floor


def weld_to_solve_sim(scene_dir: Path, sub: str, sid: str, out: Path) -> dict:
    """Exact similarity from the weld's internal sfm gauge to the store's
    ingest-solve gauge, via shared camera identities (STO-SCN-089-3).
    Every mast3r run mints its own arbitrary gauge; the weld's meshes are
    only meaningful in the store after composing this."""
    import numpy as np
    from gauge_align import align_camera_sets
    weld = json.loads((out / "mast3r_sfm" / "cameras.json").read_text())
    store = json.loads((scene_dir / "images" / "subsets" / sub / "cameras" / sid /
                        "cameras.json").read_text())
    def by_name(c):
        # match by STEM: extension drift between runs (jpg/jpeg re-encodes
        # in migrated pools) must not break camera identity (007 lesson)
        return {fp.rsplit("/", 1)[-1].rsplit(".", 1)[0]: np.asarray(m)
                for fp, m in zip(c["filepaths"], c["cams2world"])}
    wm, sm = by_name(weld), by_name(store)
    names = sorted(set(wm) & set(sm))
    if len(names) < 3:
        sys.exit(f"gauge-sim REFUSED: only {len(names)} shared cameras")
    Cw = np.array([wm[n][:3, 3] for n in names])
    Cs = np.array([sm[n][:3, 3] for n in names])
    Rw = np.array([wm[n][:3, :3] for n in names])
    Rs = np.array([sm[n][:3, :3] for n in names])
    res = align_camera_sets(Cw, Cs, src_rotations=Rw, dst_rotations=Rs)
    spread = float(np.linalg.norm(Cs - Cs.mean(0), axis=1).mean())
    R = np.asarray(res["R"])
    return {"s": float(res["scale"]), "R": R.tolist(), "t": np.asarray(res["t"]).tolist(),
            "rot_deg": float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))),
            "max_residual": float(res["max_residual"]),
            "residual_frac": float(res["max_residual"] / spread)}


def ground_mesh(src: Path, dst: Path, R, z, sim: dict | None = None):
    import numpy as np
    import open3d as o3d
    mesh = o3d.io.read_triangle_mesh(str(src))
    vv = np.asarray(mesh.vertices)
    if sim is not None:   # weld frame -> solve frame first (STO-SCN-089-3)
        vv = sim["s"] * (vv @ np.asarray(sim["R"]).T) + np.asarray(sim["t"])
    mesh.vertices = o3d.utility.Vector3dVector(vv @ np.asarray(R).T + np.array([0.0, 0.0, z]))
    mesh.compute_vertex_normals()
    o3d.io.write_triangle_mesh(str(dst), mesh)


# ============================================================ views from blend

def cmd_views(args):
    """Operator-captured virtual cameras (Blender) -> views/<slot>/view.json.
    Operator data entry — allowed input per locked #7/#11."""
    scene_dir = v4.STORE / args.scene
    code = f'''
import bpy, json
out = []
for o in bpy.data.objects:
    if o.type == "CAMERA" and o.get("localization_method") == "viewport-capture":
        q = o.rotation_quaternion if o.rotation_mode == "QUATERNION" else o.rotation_euler.to_quaternion()
        out.append(dict(name=o.name, world_position=list(o.location),
                        world_rotation_quat_wxyz=[q.w, q.x, q.y, q.z],
                        lens_mm=o.data.lens, sensor_width_mm=o.data.sensor_width,
                        sensor_height_mm=o.data.sensor_height,
                        convention="blender",
                        purpose=o.get("view_purpose", "ab-comparison"),
                        render_resolution=[1920, 1080], render_engine="BLENDER_WORKBENCH"))
print("VIEWS_JSON" + json.dumps(out))
'''
    r = subprocess.run(["/Applications/Blender.app/Contents/MacOS/Blender", "--background",
                        args.blend, "--python-expr", code], capture_output=True, text=True)
    views = None
    for line in r.stdout.splitlines():
        if line.startswith("VIEWS_JSON"):
            views = json.loads(line[len("VIEWS_JSON"):])
    if not views:
        sys.exit("no virtual cameras found in blend (collection containing 'virtual' "
                 "or objects with view_purpose)")
    slots = []
    existing = sorted(int(p.name) for p in (scene_dir / "views").glob("[0-9]*")
                      if p.is_dir()) if (scene_dir / "views").is_dir() else []
    # idempotent by captured_name: a camera already in a slot maps back to
    # THAT slot — re-framing updates the existing view.json in place (the
    # expected /camera-save iteration loop), never mints a duplicate slot
    # (the ghost-slot class). New captured_names append a fresh slot.
    by_name = {}
    for p in (scene_dir / "views").glob("[0-9]*") if (scene_dir / "views").is_dir() else []:
        try:
            md = json.loads((p / "metadata.json").read_text())
            by_name[md.get("captured_name")] = p.name
        except (OSError, json.JSONDecodeError):
            continue
    next_slot = (existing[-1] + 1) if existing else 1
    for vw in views:
        if vw["name"] in by_name:
            slot, action = by_name[vw["name"]], "updated"
        else:
            slot, action = f"{next_slot:02d}", "created"
            next_slot += 1
        vdir = scene_dir / "views" / slot
        vdir.mkdir(parents=True, exist_ok=True)
        content = {k: v for k, v in vw.items() if k != "name"}
        (vdir / "view.json").write_text(json.dumps(content, indent=2, sort_keys=True) + "\n")
        (vdir / "metadata.json").write_text(json.dumps({
            "schema": 4, "captured_name": vw["name"], "mechanism": "operator-capture",
            "written": NOW()}, indent=2) + "\n")
        if action == "created":
            slots.append(slot)
        print(f"view {slot}: '{vw['name']}' lens {vw['lens_mm']}mm ({action})")
    cdir = scene_dir / "viewset" / "canonical"
    cdir.mkdir(parents=True, exist_ok=True)
    members = json.loads((cdir / "views.json").read_text())["slots"] \
        if (cdir / "views.json").exists() else []
    members += [s for s in slots if s not in members]
    (cdir / "views.json").write_text(json.dumps({"slots": members}, indent=2) + "\n")
    print(f"canonical viewset: {members}")


# ============================================================ regauge views

def cmd_regauge_views(args):
    """Carry operator-framed views from one orient gauge to another
    (in-graph, job-recorded). The framing is operator data (locked #7);
    when the gauge it was framed IN gets revised, the equivalent framing
    in the new gauge is a deterministic transform — re-capturing by hand
    would only reproduce it with extra operator cost."""
    import numpy as np
    scene_dir = v4.STORE / args.scene
    sc = v4.Scene(args.scene)
    sub = sc.resolve("primary")
    base = scene_dir / "images" / "subsets" / sub / "cameras" / args.solve / "orient"
    g1 = json.loads((base / args.from_orient / "oriented.json").read_text())
    g2 = json.loads((base / args.to_orient / "oriented.json").read_text())
    R1, z1 = np.asarray(g1["rotation"]), float(g1["z_shift"])
    R2, z2 = np.asarray(g2["rotation"]), float(g2["z_shift"])
    # optional weld->solve similarity between the two eras (STO-SCN-089-3):
    # views were framed against the MESH; carry them with the mesh:
    # composite = G_new o Sim o G_old^-1
    if args.sim:
        sim = json.loads(Path(args.sim).read_text()) if Path(args.sim).exists()             else json.loads(args.sim)
        ss, Rs_, ts = float(sim["s"]), np.asarray(sim["R"]), np.asarray(sim["t"])
    else:
        ss, Rs_, ts = 1.0, np.eye(3), np.zeros(3)
    Rd = R2 @ Rs_ @ R1.T
    td = (-ss * (Rd @ np.array([0.0, 0.0, z1])) + R2 @ ts + np.array([0.0, 0.0, z2]))
    sd_scale = ss

    def quat_to_R(w, x, y, z):
        return np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])

    def R_to_quat(R):
        w = np.sqrt(max(0.0, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
        if w > 1e-8:
            return [w, (R[2, 1] - R[1, 2]) / (4 * w), (R[0, 2] - R[2, 0]) / (4 * w),
                    (R[1, 0] - R[0, 1]) / (4 * w)]
        # w ~ 0: use largest diagonal branch
        i = int(np.argmax([R[0, 0], R[1, 1], R[2, 2]]))
        if i == 0:
            x = np.sqrt(max(0.0, 1 + R[0, 0] - R[1, 1] - R[2, 2])) / 2
            return [(R[2, 1] - R[1, 2]) / (4 * x), x, (R[0, 1] + R[1, 0]) / (4 * x),
                    (R[0, 2] + R[2, 0]) / (4 * x)]
        if i == 1:
            y = np.sqrt(max(0.0, 1 - R[0, 0] + R[1, 1] - R[2, 2])) / 2
            return [(R[0, 2] - R[2, 0]) / (4 * y), (R[0, 1] + R[1, 0]) / (4 * y), y,
                    (R[1, 2] + R[2, 1]) / (4 * y)]
        z = np.sqrt(max(0.0, 1 - R[0, 0] - R[1, 1] + R[2, 2])) / 2
        return [(R[1, 0] - R[0, 1]) / (4 * z), (R[0, 2] + R[2, 0]) / (4 * z),
                (R[1, 2] + R[2, 1]) / (4 * z), z]

    nodes = []
    for vdir in sorted(scene_dir.glob("views/[0-9]*/")):
        vj = vdir / "view.json"
        if not vj.exists():
            continue
        view = json.loads(vj.read_text())
        p = np.asarray(view["world_position"])
        q = view["world_rotation_quat_wxyz"]
        Rv = quat_to_R(*q)
        view["world_position"] = [float(x) for x in (sd_scale * (Rd @ p) + td)]
        view["world_rotation_quat_wxyz"] = [float(x) for x in R_to_quat(Rd @ Rv)]
        vj.write_text(json.dumps(view, indent=2, sort_keys=True) + "\n")
        md_f = vdir / "metadata.json"
        md = json.loads(md_f.read_text()) if md_f.exists() else {"schema": 4}
        md.setdefault("regauged", []).append({
            "from": args.from_orient, "to": args.to_orient, "at": NOW()})
        md_f.write_text(json.dumps(md, indent=2) + "\n")
        nodes.append({"node": "regauge-view", "slot": vdir.name, "action": "EXECUTE"})
        print(f"view {vdir.name}: regauged {args.from_orient} -> {args.to_orient}")
    job_record(args.scene, "regauge-views", nodes,
               {"scene": args.scene, "solve": args.solve,
                "from": args.from_orient, "to": args.to_orient})


# ============================================================ reconstruct-da3

def cmd_da3(args):
    import numpy as np
    scene_dir = v4.STORE / args.scene
    sc = v4.Scene(args.scene)
    sub = sc.resolve("primary")
    tdefs = v4.tasks()
    nodes = []
    solve_dirs = [d for d in sorted((scene_dir / "images" / "subsets" / sub / "cameras").glob("*/"))
                  if (d / "metadata.json").exists()]
    sid = solve_dirs[0].name
    ref_id, ref_mesh, oid = matcha_reference(scene_dir, sub, sid, tdefs)
    if ref_mesh is None or oid is None or not (
            scene_dir / "images" / "subsets" / sub / "cameras" / sid / "orient" / oid /
            "oriented.json").exists():
        sys.exit(f"no {ORIENT_ALGO} gauge / matcha reference — "
                 f"run reconstruct-matcha first (bootstrap)")
    r_settings = v4.hashable_settings(tdefs["represent-via-da3"], {})
    # da3@0: unposed — DA3 estimates its own cameras; gauge-independent
    #        (the npz never sees the orient); gauge enters at FUSE
    #        (STO-SCN-089-2).
    # da3@1: POSED — the ingest solve is fed to inference(extrinsics=,
    #        intrinsics=), so the solve IS a resolved input (STO-SCN-090;
    #        forced by 003-firepit: unposed DA3 poses off by 60.7%).
    d_algo = "da3@1" if args.sfm == "posed" else "da3@0"
    d_inputs = {"subset": sub} if d_algo == "da3@0" else {"subset": sub, "cameras": sid}
    rid = v4.identity_hash(d_inputs, r_settings, d_algo)
    rdir = scene_dir / "represent" / "da3" / rid
    if not (rdir / "metadata.json").exists():
        tag = f"{args.scene}-da3-{rid}"
        workdir = stage_images_on_host(args.host, scene_dir, sub, tag)
        if args.sfm == "posed":
            from colmap_posed import solve_to_posed_json
            members = json.loads((scene_dir / "images" / "subsets" / sub /
                                  "subset.json").read_text())["members"]
            by_hash = {p.parent.name: p for p in scene_dir.glob("images/*/image.*")}
            staged = {}
            for h in members:
                d = json.loads((scene_dir / "images" / h / "metadata.json").read_text())
                staged[d.get("original_name", h + ".jpg")] = by_hash[h]
            tmp_posed = Path("/tmp") / f"v4exec-{tag}-posed.json"
            covered = solve_to_posed_json(
                scene_dir / "images" / "subsets" / sub / "cameras" / sid / "cameras.json",
                staged, tmp_posed)
            driver = Path(__file__).parent / "da3_infer_posed.py"
            sh(["ssh", args.host, f"mkdir -p {workdir}/cameras"])
            sh(["rsync", "-a", str(tmp_posed), f"{args.host}:{workdir}/cameras/posed.json"])
            sh(["rsync", "-a", str(driver), f"{args.host}:{workdir}/da3_infer_posed.py"])
            tmp_posed.unlink()
            print(f"[da3@1] posed.json minted from solve {sid} ({len(covered)} cameras)")
            tool = f"python /work/da3_infer_posed.py /work/images /work/out {r_settings['process_res']}"
        else:
            tool = f"python /opt/krabby-tools/da3_infer_gs.py /work/images /work/out {r_settings['process_res']}"
        if r_settings.get("mode") == "nogs":
            tool += " nogs"
        docker = (f"docker run --rm --gpus all -v {workdir}:/work {DA3_IMAGE} {tool} ; rc=$? ; "
                  f"docker run --rm -v {workdir}:/work alpine chown -R $(id -u):$(id -g) /work ; exit $rc")
        print(f"[da3 infer] {args.host}: {tool}")
        t0 = datetime.datetime.now()
        r = subprocess.run(["ssh", args.host, docker], capture_output=True, text=True)
        dt = int((datetime.datetime.now() - t0).total_seconds())
        rdir.mkdir(parents=True, exist_ok=True)
        (rdir / "infer.log").write_text(r.stdout[-100000:] + "\n--- stderr ---\n" + r.stderr[-20000:])
        sh(["rsync", "-a", f"{args.host}:{workdir}/out/", str(rdir) + "/"])
        sh(["ssh", args.host, f"rm -rf {SCRATCH}/{tag}"])
        npz = rdir / "exports" / "npz" / "results.npz"
        if r.returncode != 0 or not npz.exists():
            sys.exit(f"da3 infer FAILED rc={r.returncode} (npz present: {npz.exists()})")
        dig, tools_sha = host_digest(args.host, DA3_IMAGE)
        v4.write_metadata(rdir, task="represent-via-da3", algo=d_algo, identity=rid,
                          resolved_inputs=d_inputs,
                          settings=r_settings, mechanism="job",
                          measured={"host": args.host.split("@")[-1], "duration_s": dt,
                                    "image_digest": dig, "tools_git_sha": tools_sha})
        md = json.loads((rdir / "metadata.json").read_text())
        odir = scene_dir / "images" / "subsets" / sub / "cameras" / sid / "orient" / oid
        md["canonical_gauge"] = str((odir / "oriented.json").relative_to(scene_dir))
        (rdir / "metadata.json").write_text(json.dumps(md, indent=2) + "\n")
        nodes.append({"node": "represent", "identity": rid, "action": "EXECUTE",
                      "host": args.host, "duration_s": dt})
    else:
        nodes.append({"node": "represent", "identity": rid, "action": "NOOP"})

    # -- fuse (local CPU): depths -> mesh, aligned into the orient gauge.
    # da3-fuse@2: ICP-refined onto the matcha reference mesh (STO-SCN-089).
    # DA3 is the evaluation branch — the reference always exists, and the
    # pre-correction magnitude IS the recorded self-alignment score. A
    # refined mesh is no longer independent evidence of DA3's global
    # accuracy (recorded in metadata); as ranked geometry quality it is
    # exactly what the runoff compares.
    fuse_settings = {"voxel_frac": 0.004, "conf_percentile": 40,
                     "icp_schedule": [0.5, 0.25, 0.1]}
    fid = v4.identity_hash({"representation": rid, "cameras": sid, "orient": oid,
                            "reference": ref_id},
                           fuse_settings, "da3-fuse@2")
    fdir = rdir / "meshify" / "tsdf" / fid
    if (fdir / "mesh.ply").exists():
        nodes.append({"node": "fuse", "identity": fid, "action": "NOOP"})
    else:
        measured = fuse_da3(scene_dir, rdir, sub, sid, oid, fdir, ref_mesh)
        sa = measured["self_alignment"]
        applied = sa["icp_applied"]
        extra = {"reference_registered": applied,
                 "note": "placement refined onto matcha reference; "
                         "not independent evidence of DA3 global accuracy"}
        # Gate policy (STO-SCN-089-2): the camera alignment is the PRIMARY
        # placement evidence (verified photo-consistent on 009 where ICP
        # wandered into the reference's near-camera floaters). Flag only
        # when BOTH signals fail: ICP degenerate AND camera residual loose.
        if not applied:
            if sa["camera_residual_frac"] <= 0.03:
                extra["note"] += ("; ICP degenerate (reference floaters) — "
                                  "camera-aligned placement kept, residual "
                                  f"{sa['camera_residual_frac']:.1%} (tight)")
            else:
                extra["rankable"] = False
                extra["rankable_reason"] = (
                    "placement unverifiable: ICP degenerate AND camera residual "
                    f"{sa['camera_residual_frac']:.1%} > 3%")
        v4.write_metadata(fdir, task="meshify-via-tsdf", algo="da3-fuse@2", identity=fid,
                          resolved_inputs={"representation": rid, "cameras": sid,
                                           "orient": oid, "reference": ref_id},
                          settings=fuse_settings, mechanism="job", measured=measured,
                          extra=extra)
        nodes.append({"node": "fuse", "identity": fid, "action": "EXECUTE"})
    job_record(args.scene, "reconstruct-da3", nodes,
               {"scene": args.scene, "host": args.host})
    print(f"reconstruct-da3 materialized: represent {rid}, fused {fid}")


def cmd_da3_scout(args):
    """STO-SCN-127 — matcha-FREE DA3 scene mesh in the SOLVE/spine gauge, from an existing
    scout npz. NO GPU, NO matcha reference: the spine already posed the cameras, so the scout's
    `da3_poses.npz` (depth+conf+echoed-solve extrinsics, already in the solve gauge) TSDF-fuses
    straight into the solve gauge (da3_mesh_from_npz.fuse_npz). Gravity comes from the SOLVE
    cameras (bootstrap_orient — the same orient matcha uses), not from a matcha mesh. Emits a
    content-addressed represent/da3 + meshify node that `v4job render-missing` discovers like
    any other rankable mesh. This is the α path (the spine makes matcha unnecessary for DA3)."""
    import numpy as np
    import open3d as o3d
    from da3_mesh_from_npz import fuse_npz
    scene_dir = v4.STORE / args.scene
    sc = v4.Scene(args.scene)
    sub = args.subset or sc.resolve("primary")
    sid, cid = args.solve, args.scout
    sdir = scene_dir / "images" / "subsets" / sub / "cameras" / sid
    npz = sdir / "scout" / cid / "da3_poses.npz"
    if not npz.exists():
        sys.exit(f"no scout npz at {npz} — run `scout` first (or check --solve/--scout)")
    if not (sdir / "sparse" / "0").exists():
        sys.exit(f"no solve sparse/0 at {sdir} — run `solve` first")
    tdefs = v4.tasks()
    nodes = []

    # FastMap solves emit only sparse/0 (COLMAP bins); the renderer (rep_camera_paths /
    # build_blender_scene) needs a cameras.json {filepaths, cams2world} at the solve dir to
    # place the T2 views. Emit it once from sparse/0 (benefits every rep on this solve).
    cams_json = sdir / "cameras.json"
    if not cams_json.exists():
        n = posed_sparse_to_cameras_json(sdir / "sparse" / "0", cams_json)
        print(f"[da3-scout] emitted solve cameras.json ({n} cams, 512-conv) -> {cams_json}")

    # identities (content-addressed): the represent node is sourced FROM the scout npz.
    r_settings = v4.hashable_settings(tdefs["represent-via-da3"], {})
    d_inputs = {"subset": sub, "cameras": sid, "scout": cid}
    rid = v4.identity_hash(d_inputs, r_settings, "da3@1")
    fuse_settings = {"conf_percentile": args.conf_percentile, "voxel_frac": args.voxel_frac}
    oid = v4.identity_hash({"solve": sid, "bootstrap_rep": rid}, ORIENT_SETTINGS, ORIENT_ALGO)
    tid = v4.identity_hash({"representation": rid, "cameras": sid, "orient": oid},
                           fuse_settings, "da3-mesh@0")
    rdir = scene_dir / "represent" / "da3" / rid
    odir = sdir / "orient" / oid
    tdir = rdir / "meshify" / "tsdf" / tid

    if (tdir / "mesh.ply").exists() and (rdir / "metadata.json").exists():
        print(f"[da3-scout] NOOP — mesh {tid} exists -> {tdir}")
        return

    # 1. TSDF-fuse the scout npz -> raw DA3 mesh in the SOLVE gauge.
    rdir.mkdir(parents=True, exist_ok=True)
    raw_ply = rdir / "da3_scene_raw.ply"
    rec = fuse_npz(str(npz), str(raw_ply), args.conf_percentile, args.voxel_frac)
    if not (rdir / "metadata.json").exists():
        v4.write_metadata(rdir, task="represent-via-da3", algo="da3@1", identity=rid,
                          resolved_inputs=d_inputs, settings=r_settings, mechanism="job",
                          measured={"source": f"scout/{cid}/da3_poses.npz",
                                    "n_views": rec["n_views"], "gpu": False})

    # 2. Gravity orient from the SOLVE cameras (matcha-FREE) + floor fit on the DA3 mesh.
    if (odir / "oriented.json").exists():
        g = json.loads((odir / "oriented.json").read_text())
        R, z = np.asarray(g["rotation"]), float(g["z_shift"])
        nodes.append({"node": "orient", "identity": oid, "action": "NOOP"})
    else:
        # Orient cameras = the exact 24 posed views that built the mesh (npz extrinsics, w2c),
        # inverted to c2w — guarantees the gravity/floor fit matches the mesh's own cameras.
        ext = np.load(str(npz))["extrinsics"].astype(np.float64)   # (N,3,4) w2c
        Rw, tw = ext[:, :3, :3], ext[:, :3, 3]
        Rc2w = np.transpose(Rw, (0, 2, 1))
        C = -np.einsum("nij,nj->ni", Rc2w, tw)
        raw_v = np.asarray(o3d.io.read_triangle_mesh(str(raw_ply)).vertices)
        R, z = bootstrap_orient(raw_v, cam_R_c2w=Rc2w, cam_C=C)
        odir.mkdir(parents=True, exist_ok=True)
        payload = json.dumps({"rotation": R.tolist(), "z_shift": float(z)}, indent=2) + "\n"
        (odir / "oriented.json").write_text(payload)
        (odir / "transform.json").write_text(payload)
        v4.write_metadata(odir, task="orient-cameras", algo=ORIENT_ALGO, identity=oid,
                          resolved_inputs={"solve": sid, "bootstrap_rep": rid},
                          settings=ORIENT_SETTINGS, mechanism="job")
        nodes.append({"node": "orient", "identity": oid, "action": "EXECUTE"})

    # 3. Ground the mesh into the canonical gauge (no weld-sim — already the solve gauge).
    tdir.mkdir(parents=True, exist_ok=True)
    ground_mesh(raw_ply, tdir / "mesh.ply", R, z, sim=None)
    v4.write_metadata(tdir, task="meshify-via-tsdf", algo="da3-mesh@0", identity=tid,
                      resolved_inputs={"representation": rid, "cameras": sid, "orient": oid},
                      settings=fuse_settings, mechanism="job", measured=rec,
                      extra={"gauge": str(odir.relative_to(scene_dir)),
                             "matcha_free": True, "source_gauge": "solve"})
    nodes.append({"node": "da3-mesh", "identity": tid, "action": "EXECUTE"})

    # canonical gauge marker for the renderer (same contract matcha sets).
    md = json.loads((rdir / "metadata.json").read_text())
    md["canonical_gauge"] = str((odir / "oriented.json").relative_to(scene_dir))
    (rdir / "metadata.json").write_text(json.dumps(md, indent=2) + "\n")
    raw_ply.unlink(missing_ok=True)   # the un-grounded intermediate; mesh.ply is canonical

    job_record(args.scene, "reconstruct-da3-scout", nodes,
               {"scene": args.scene, "solve": sid, "scout": cid, "gpu": False})
    print(f"reconstruct-da3-scout materialized: represent {rid}, mesh {tid}, orient {oid}")
    print(f"  mesh: {tdir / 'mesh.ply'}  ({rec['verts']:,} verts / {rec['tris']:,} tris)")
    print(f"  next: python3 real2sim/v4job.py render-missing {args.scene}")


def matcha_reference(scene_dir: Path, sub: str, sid: str, tdefs):
    """Resolve the matcha reference tsdf mesh for this solve.

    Deterministic recompute first (native-run scenes, default settings);
    falls back to a store scan for migrated scenes whose identities were
    minted under other settings. Returns (mesh_identity, mesh_path) or
    (None, None)."""
    r_settings = v4.hashable_settings(tdefs["represent-via-matcha"],
                                      {"dense_regul": "default"})
    ts_settings = v4.hashable_settings(tdefs["meshify-via-tsdf"], {})
    for m_algo in ("matcha@1", "matcha@0"):     # posed weld preferred
        rid_m = v4.identity_hash({"subset": sub, "cameras": sid}, r_settings, m_algo)
        oid = v4.identity_hash({"solve": sid, "bootstrap_rep": rid_m},
                               ORIENT_SETTINGS, ORIENT_ALGO)
        tid = v4.identity_hash({"representation": rid_m, "cameras": sid, "orient": oid},
                               ts_settings, "tsdf-extract@1")
        p = scene_dir / "represent" / "matcha" / rid_m / "meshify" / "tsdf" / tid / "mesh.ply"
        if p.exists():
            return tid, p, oid
    # scan fallback: any matcha tsdf mesh produced against this solve
    for mp in sorted(scene_dir.glob("represent/matcha/*/meshify/tsdf/*/mesh.ply")):
        try:
            md = json.loads((mp.parent / "metadata.json").read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if md.get("resolved_inputs", {}).get("cameras") in (sid, None):
            return (md.get("identity", mp.parent.name), mp,
                    md.get("resolved_inputs", {}).get("orient"))
    return None, None, None


def fuse_da3(scene_dir: Path, rdir: Path, sub: str, sid: str, oid: str, fdir: Path,
             ref_mesh: Path):
    import numpy as np
    import open3d as o3d
    from gauge_align import align_camera_sets
    cams = json.loads((scene_dir / "images" / "subsets" / sub / "cameras" / sid /
                       "cameras.json").read_text())
    gauge = json.loads((scene_dir / "images" / "subsets" / sub / "cameras" / sid /
                        "orient" / oid / "oriented.json").read_text())
    R_o, z = np.asarray(gauge["rotation"]), float(gauge["z_shift"])
    order = np.argsort([fp.rsplit("/", 1)[-1] for fp in cams["filepaths"]])
    c2w = np.asarray(cams["cams2world"])[order]
    C_mat = (R_o @ c2w[:, :3, 3].T).T + np.array([0.0, 0.0, z])
    R_mat = np.einsum("ij,njk->nik", R_o, c2w[:, :3, :3])
    npz = np.load(rdir / "exports" / "npz" / "results.npz")
    depth = npz["depth"].astype(np.float32)
    conf, img = npz["conf"], npz["image"]
    ext = npz["extrinsics"].astype(np.float64)
    K = npz["intrinsics"].astype(np.float64)
    n, H, W = depth.shape
    Rw, tw = ext[:, :3, :3], ext[:, :3, 3]
    res = align_camera_sets(np.einsum("nji,nj->ni", Rw, -tw), C_mat,
                            src_rotations=np.transpose(Rw, (0, 2, 1)), dst_rotations=R_mat)
    spread = np.linalg.norm(C_mat - C_mat.mean(0), axis=1).mean()
    frac = res["max_residual"] / spread
    print(f"[fuse] alignment residual {frac*100:.1f}% scale {res['scale']:.4f}")
    if frac > 0.10:
        sys.exit("fuse REFUSED: alignment residual > 10% of camera spread")

    # ---- da3-fuse@1: DEPTH ANCHORING (STO-SCN-089 fix) ----------------
    # DA3 depths carry per-view scale bias vs its own baselines (measured:
    # 0.158m median error on 006, 0.44m on 009 — cameras align at mm).
    # Anchor each view's depth to the SOLVE's sparse points: transform the
    # matcha-frame sparse cloud into DA3's frame (inverse alignment),
    # project into each view, robust per-view scale = median(z_sparse /
    # z_da3) over confident pixels.
    s_al, R_al, t_al = res["scale"], np.asarray(res["R"]), np.asarray(res["t"])
    pts_path = scene_dir / "images" / "subsets" / sub / "cameras" / sid / "points.ply"
    view_scales = [1.0] * n
    if pts_path.exists():
        sp = o3d.io.read_point_cloud(str(pts_path))
        P_solve = np.asarray(sp.points)
        if len(P_solve) > 100:
            P_or = P_solve @ R_o.T + np.array([0.0, 0.0, z])      # solve -> gauge
            P_da3 = ((P_or - t_al) / s_al) @ R_al                 # gauge -> DA3 frame
            for i in range(n):
                pc = P_da3 @ Rw[i].T + tw[i]                      # w2c
                infront = pc[:, 2] > 0.1
                pc = pc[infront]
                u = pc[:, 0] / pc[:, 2] * K[i][0, 0] + K[i][0, 2]
                v = pc[:, 1] / pc[:, 2] * K[i][1, 1] + K[i][1, 2]
                ok = (u >= 0) & (u < W - 1) & (v >= 0) & (v < H - 1)
                ui, vi = u[ok].astype(int), v[ok].astype(int)
                dz = depth[i][vi, ui]
                cz = conf[i][vi, ui]
                good = (dz > 0.05) & (cz > np.percentile(conf[i], 40))
                if good.sum() >= 30:
                    ratios = pc[ok][good][:, 2] / dz[good]
                    view_scales[i] = float(np.median(ratios))
            print(f"[fuse@1] per-view depth anchors: "
                  f"{[round(s_, 3) for s_ in view_scales]}")
            for i in range(n):
                depth[i] *= view_scales[i]
        else:
            print("[fuse@1] sparse cloud too small — depths unanchored")
    else:
        print("[fuse@1] no points.ply — depths unanchored")
    thr = np.percentile(conf, 40)
    span = float(np.percentile(depth[conf > thr], 95))
    voxel = span * 0.004
    vol = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel, sdf_trunc=4 * voxel,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
    for i in range(n):
        d = depth[i].copy()
        d[conf[i] <= thr] = 0.0
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(np.ascontiguousarray(img[i])), o3d.geometry.Image(d),
            depth_scale=1.0, depth_trunc=float(span * 1.5), convert_rgb_to_intensity=False)
        intr = o3d.camera.PinholeCameraIntrinsic(W, H, K[i][0, 0], K[i][1, 1],
                                                 K[i][0, 2], K[i][1, 2])
        w2c4 = np.eye(4)
        w2c4[:3, :4] = ext[i]
        vol.integrate(rgbd, intr, w2c4)
    mesh = vol.extract_triangle_mesh()
    mesh.remove_degenerate_triangles()
    mesh.remove_unreferenced_vertices()
    s, R_al, t_al = res["scale"], np.asarray(res["R"]), np.asarray(res["t"])
    T = np.eye(4)
    T[:3, :3] = R_al
    T[:3, 3] = t_al / s
    mesh.transform(T)
    mesh.scale(s, center=(0.0, 0.0, 0.0))

    # ---- da3-fuse@2: register onto the matcha reference (STO-SCN-089) ----
    # Camera alignment alone leaves a rigid placement error that varies by
    # capture geometry (measured ICP corrections: 007 2.5°/0.05m "success",
    # 006 10°/0.17m "perfect", 009 16.5°/0.52m "wrong" — same code). The
    # correction magnitude is DA3's self-alignment score; applying it makes
    # the mesh comparable in the shared gauge by construction.
    ref = o3d.io.read_triangle_mesh(str(ref_mesh))
    src = mesh.sample_points_uniformly(60000)
    tgt = ref.sample_points_uniformly(120000)
    tgt.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.3, max_nn=30))
    # Coarse-to-fine with SHRINKING correspondence distance. A single pass
    # at 1.0 m found a degenerate 99-deg basin on 009 (corridor scene: ICP
    # locked the ground plane and spun). Camera alignment already bounds
    # the true correction to well under 30 deg / 1 m, so descent must stay
    # local; anything bigger is a degenerate fit, not a registration.
    Ticp = np.eye(4)
    reg = None
    for max_corr in (0.5, 0.25, 0.1):
        reg = o3d.pipelines.registration.registration_icp(
            src, tgt, max_corr, Ticp,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50))
        Ticp = np.asarray(reg.transformation)
    Ricp = Ticp[:3, :3]
    rot_deg = float(np.degrees(np.arccos(np.clip((np.trace(Ricp) - 1) / 2, -1, 1))))
    trans_m = float(np.linalg.norm(Ticp[:3, 3]))
    print(f"[fuse@2] ICP correction onto reference: {rot_deg:.1f} deg "
          f"{trans_m:.3f} m  fitness {reg.fitness:.2f}  rmse {reg.inlier_rmse:.3f}")
    registered = rot_deg <= 30.0 and trans_m <= 1.0
    if registered:
        mesh.transform(Ticp)
    else:
        print(f"[fuse@2] correction exceeds physical bound (30 deg / 1.0 m) — "
              f"degenerate basin; keeping camera-aligned placement, mesh flagged")

    mesh.compute_vertex_normals()
    fdir.mkdir(parents=True, exist_ok=True)
    o3d.io.write_triangle_mesh(str(fdir / "mesh.ply"), mesh)
    print(f"[fuse] {len(mesh.vertices):,} verts -> {fdir}/mesh.ply")
    return {"self_alignment": {"icp_rot_deg": round(rot_deg, 2),
                               "icp_trans_m": round(trans_m, 4),
                               "icp_fitness": round(float(reg.fitness), 3),
                               "icp_inlier_rmse_m": round(float(reg.inlier_rmse), 4),
                               "icp_applied": bool(registered),
                               "camera_residual_frac": round(float(frac), 4),
                               "depth_anchors": [round(s_, 4) for s_ in view_scales]},
            "reference_mesh": str(ref_mesh.relative_to(scene_dir))}


def cmd_refilter(args):
    """STO-SCN-136 path A: ground a filter-re-extracted matcha tetra into a new v4 node.

    The GPU step (extract_tetra_mesh.py --config filtered, reusing the rep's CACHED gaussians)
    drops the filtered raw tetra at <rep>/out/tetra_filtered/*.ply. This command grounds it into
    the canonical gauge by REUSING the base tetra node's recorded gauge_sim + orient (no recompute,
    so it lands in the exact same frame as the base tetra) and writes a new
    `meshify-via-tetra-filtered` node. NEW task — base tetra identities untouched."""
    scene_dir = v4.STORE / args.scene
    rep_dir = scene_dir / "represent" / "matcha" / args.rep
    if not (rep_dir / "metadata.json").exists():
        sys.exit(f"no matcha rep {args.rep} under {scene_dir}")
    base = [d for d in (rep_dir / "meshify" / "tetra").glob("*/")
            if (d / "metadata.json").exists()]
    if not base:
        sys.exit(f"no base tetra node under {rep_dir} to inherit the gauge from")
    bmd = json.loads((base[0] / "metadata.json").read_text())
    sid = bmd["resolved_inputs"]["cameras"]
    oid = bmd["resolved_inputs"]["orient"]
    gauge = bmd["gauge"]
    sim = bmd["gauge_sim"]
    og = json.loads((scene_dir / gauge / "oriented.json").read_text())
    import numpy as np
    R, z = np.asarray(og["rotation"]), float(og["z_shift"])

    raws = sorted((rep_dir / "out" / "tetra_filtered").glob("*.ply"))
    if not raws:
        sys.exit(f"no filtered raw tetra at {rep_dir}/out/tetra_filtered — run the GPU "
                 f"re-extract first (extract_tetra_mesh.py --config filtered)")
    raw = raws[-1]

    ftask = v4.tasks()["meshify-via-tetra-filtered"]
    fsettings = v4.hashable_settings(ftask, {})
    fid = v4.identity_hash({"representation": args.rep, "cameras": sid, "orient": oid},
                           fsettings, ftask["algo"])
    fdir = rep_dir / "meshify" / "tetra-filtered" / fid
    if (fdir / "mesh.ply").exists():
        print(f"[refilter] NOOP — {fid} exists ({fdir.relative_to(scene_dir)})")
        return
    fdir.mkdir(parents=True, exist_ok=True)
    print(f"[refilter] grounding {raw.name} -> meshify/tetra-filtered/{fid} "
          f"(reusing base gauge_sim rot {sim.get('rot_deg')}° + orient {oid})")
    ground_mesh(raw, fdir / "mesh.ply", R, z, sim=sim)
    v4.write_metadata(fdir, task="meshify-via-tetra-filtered", algo=ftask["algo"], identity=fid,
                      resolved_inputs={"representation": args.rep, "cameras": sid, "orient": oid},
                      settings=fsettings, mechanism="job",
                      extra={"gauge": gauge, "gauge_sim": sim,
                             "source": "extract_tetra_mesh.py --config filtered (cached gaussians)"})
    job_record(args.scene, "refilter-tetra",
               [{"node": "meshify-via-tetra-filtered", "identity": fid, "action": "EXECUTE"}],
               {"scene": args.scene, "rep": args.rep})
    print(f"[refilter] wrote {fdir.relative_to(scene_dir)}/mesh.ply")


def cmd_mergefill(args):
    """STO-SCN-142: (A) merge & gap-fill via screened Poisson — additive condition node.

    Consumes a materialized mesh (a meshify node OR a condition node — so it can run on a culled
    mesh; chaining), runs merge_gapfill.py (Open3D Poisson + density-trim + largest-component +
    colour transfer) on the GATHER HOST (CPU, no GPU), and writes `<meshify>/condition/<cid>/mesh.ply`.
    Poisson is gauge-preserving (the mesh stays in its canonical gauge). NOOP when the identity
    exists. NEW `merge-gapfill@0` task — existing identities untouched."""
    import re
    scene_dir = v4.STORE / args.scene
    variant = args.variant
    # input may be a meshify node OR a condition node (chain Poisson onto a culled mesh)
    cand = (list(scene_dir.glob(f"represent/*/*/meshify/*/{variant}"))
            + list(scene_dir.glob(f"represent/*/*/meshify/*/*/condition/{variant}")))
    matches = [d for d in cand if (d / "mesh.ply").exists()]
    if not matches:
        sys.exit(f"no materialized mesh node '{variant}' under {scene_dir} "
                 f"(looked for meshify and condition nodes)")
    if len(matches) > 1:
        sys.exit(f"ambiguous variant '{variant}': {[str(m) for m in matches]}")
    src_node = matches[0]
    # the output condition node lives under the parent MESHIFY dir (so v4job.mesh_targets,
    # which globs meshify/*/*/condition/*/, discovers it — one condition level)
    meshify_dir = src_node.parent.parent if src_node.parent.name == "condition" else src_node
    md = json.loads((src_node / "metadata.json").read_text())
    gauge = md.get("gauge")

    overrides = {}
    if getattr(args, "method", None) is not None:
        overrides["method"] = args.method
    if getattr(args, "hole_size", None) is not None:
        overrides["hole_size"] = args.hole_size
    if args.poisson_depth is not None:
        overrides["poisson_depth"] = args.poisson_depth
    if args.density_quantile is not None:
        overrides["density_quantile"] = args.density_quantile
    if args.samples is not None:
        overrides["samples"] = args.samples
    task = v4.tasks()["merge-gapfill"]
    algo = task["algo"]
    settings = v4.hashable_settings(task, overrides)
    cid = v4.identity_hash({"mesh": variant}, settings, algo)
    cdir = meshify_dir / "condition" / cid

    if (cdir / "mesh.ply").exists():
        print(f"[mergefill] NOOP — {cid} exists ({cdir.relative_to(scene_dir)})")
        return
    cdir.mkdir(parents=True, exist_ok=True)
    tool = str(Path(__file__).parent / "merge_gapfill.py")
    cmd = ["uv", "run", "--quiet", "--python", "3.11", "--with", "numpy", "--with", "open3d",
           "python3", tool,
           "--mesh", str(src_node / "mesh.ply"),
           "--output", str(cdir / "mesh.ply"),
           "--method", str(settings["method"]),
           "--hole-size", str(settings["hole_size"]),
           "--poisson-depth", str(settings["poisson_depth"]),
           "--density-quantile", str(settings["density_quantile"]),
           "--samples", str(settings["samples"])]
    print(f"[mergefill] {variant} -> condition/{cid} "
          f"(method={settings['method']} hole_size={settings['hole_size']}) — CPU")
    t0 = datetime.datetime.now()
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    dt = int((datetime.datetime.now() - t0).total_seconds())
    sys.stdout.write(r.stdout)
    if r.returncode != 0 or not (cdir / "mesh.ply").exists():
        shutil.rmtree(cdir, ignore_errors=True)
        sys.exit(f"[mergefill] FAILED (rc={r.returncode})\n{r.stderr[-2000:]}")
    measured = {"duration_s": dt}
    mv = re.search(r"final:\s*([\d,]+)\s*verts\s*/\s*([\d,]+)\s*tris\s*watertight=(\w+)", r.stdout)
    if mv:
        measured["verts"] = int(mv.group(1).replace(",", ""))
        measured["tris"] = int(mv.group(2).replace(",", ""))
        measured["watertight"] = mv.group(3) == "True"
    v4.write_metadata(cdir, task="merge-gapfill", algo=algo, identity=cid,
                      resolved_inputs={"mesh": variant}, settings=settings, mechanism="job",
                      measured=measured, extra={"gauge": gauge,
                                                "source_mesh": str(src_node.relative_to(scene_dir))})
    job_record(args.scene, "merge-gapfill",
               [{"node": "merge-gapfill", "identity": cid, "action": "EXECUTE", "duration_s": dt}],
               {"scene": args.scene, "variant": variant})
    print(f"[mergefill] wrote {cdir.relative_to(scene_dir)}/mesh.ply "
          f"(watertight={measured.get('watertight','?')}) in {dt}s")


def cmd_cull(args):
    """STO-SCN-136: post-meshify CPU cull as an additive content-addressed condition node.

    Consumes an already-materialized meshify mesh (the grounded mesh.ply, canonical gauge) +
    its solve cameras + oriented gauge, runs cull_mesh.py (drop few-view / distant / sub-floor
    verts) on the GATHER HOST (no GPU), and writes `<meshify>/condition/<cid>/mesh.ply`. The
    cull settings flow into the identity, so a culled mesh is a DISTINCT store node from the raw
    one (raw stays for comparison) and a re-run is a NOOP. New `cull-mesh@0` task — touches no
    existing meshify taskdef, so historical mesh identities are unchanged (STO-SCN-136
    backwards-compat)."""
    import re
    scene_dir = v4.STORE / args.scene
    variant = args.variant
    # locate the meshify node: represent/<kind>/<rid>/meshify/<tetra|tsdf>/<variant>
    matches = [d for d in scene_dir.glob(f"represent/*/*/meshify/*/{variant}")
               if (d / "mesh.ply").exists()]
    if not matches:
        sys.exit(f"no materialized meshify node '{variant}' under {scene_dir} "
                 f"(looked for represent/*/*/meshify/*/{variant}/mesh.ply)")
    if len(matches) > 1:
        sys.exit(f"ambiguous variant '{variant}': {[str(m) for m in matches]}")
    mdir = matches[0]
    md = json.loads((mdir / "metadata.json").read_text())
    gauge = md.get("gauge")                       # …/cameras/<sid>/orient/<oid>  (relpath)
    if not gauge:
        # da3-reference nodes (cmd_da3) record cameras+orient in resolved_inputs, not a `gauge`
        # relpath — derive the gauge dir by globbing for that solve+orient (works for any node).
        ri = md.get("resolved_inputs", {})
        cam_id, ori_id = ri.get("cameras"), ri.get("orient")
        if cam_id and ori_id:
            hits = list(scene_dir.glob(f"images/subsets/*/cameras/{cam_id}/orient/{ori_id}"))
            if hits:
                gauge = str(hits[0].relative_to(scene_dir))
    if not gauge:
        sys.exit(f"meshify node {variant} has no resolvable gauge — cannot resolve cameras")
    oriented_json = scene_dir / gauge / "oriented.json"
    cameras_json = scene_dir / Path(gauge).parent.parent / "cameras.json"
    for p in (oriented_json, cameras_json):
        if not p.exists():
            sys.exit(f"missing gauge input for cull: {p}")
    sid = md.get("resolved_inputs", {}).get("cameras")
    oid = md.get("resolved_inputs", {}).get("orient")

    # settings: explicit overrides only; hashable_settings injects task defaults (identity-stable)
    overrides = {}
    if args.min_views is not None:
        overrides["min_views"] = args.min_views
    if args.max_dist is not None:
        overrides["max_dist_from_cluster"] = args.max_dist
    if getattr(args, "cambox_expand", None) is not None:
        overrides["cambox_expand"] = args.cambox_expand
    if args.floor_z_min is not None:
        overrides["floor_z_min"] = args.floor_z_min
    if args.image_size is not None:
        overrides["image_size"] = args.image_size
    if getattr(args, "primitives", None) is not None:
        with open(args.primitives) as pf:
            overrides["primitives"] = json.load(pf)   # STO-SCN-145: inline spec content -> identity
    cull_task = v4.tasks()["cull-mesh"]
    cull_algo = cull_task["algo"]            # version-safe (STO-SCN-137 bumped @0 -> @1)
    settings = v4.hashable_settings(cull_task, overrides)
    cid = v4.identity_hash({"mesh": variant}, settings, cull_algo)
    cdir = mdir / "condition" / cid

    if (cdir / "mesh.ply").exists():
        print(f"[cull] NOOP — {cid} exists ({cdir.relative_to(scene_dir)})")
        job_record(args.scene, "cull-mesh",
                   [{"node": "cull-mesh", "identity": cid, "action": "NOOP"}],
                   {"scene": args.scene, "variant": variant})
        return

    cdir.mkdir(parents=True, exist_ok=True)
    # STO-SCN-145: materialize the primitive spec next to the node for cull_mesh.py to consume
    prim_path = None
    if settings.get("primitives") is not None:
        prim_path = cdir / "primitives.json"
        prim_path.write_text(json.dumps(settings["primitives"]))
    cull_py = str(Path(__file__).parent / "cull_mesh.py")
    cmd = ["uv", "run", "--quiet", "--python", "3.11", "--with", "numpy", "--with", "open3d",
           "python3", cull_py,
           "--mesh", str(mdir / "mesh.ply"),
           "--cameras", str(cameras_json),
           "--oriented-cameras", str(oriented_json),
           "--output", str(cdir / "mesh.ply"),
           "--min-views", str(settings["min_views"]),
           "--floor-z-min", str(settings["floor_z_min"]),
           "--max-dist-from-cluster", str(settings["max_dist_from_cluster"]),
           "--cambox-expand", str(settings["cambox_expand"]),
           "--image-size", str(settings["image_size"])]
    if prim_path is not None:
        cmd += ["--primitives", str(prim_path)]
    print(f"[cull] {variant} -> condition/{cid}  "
          f"(min_views={settings['min_views']} max_dist={settings['max_dist_from_cluster']} "
          f"floor_z_min={settings['floor_z_min']}) — CPU, no GPU")
    t0 = datetime.datetime.now()
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    dt = int((datetime.datetime.now() - t0).total_seconds())
    sys.stdout.write(r.stdout)
    if r.returncode != 0 or not (cdir / "mesh.ply").exists():
        shutil.rmtree(cdir, ignore_errors=True)        # don't leave a half-node
        sys.exit(f"[cull] FAILED (rc={r.returncode})\n{r.stderr[-2000:]}")
    # parse final counts for the metadata measured block
    measured = {"duration_s": dt}
    m = re.search(r"final:\s*([\d,]+)\s*verts\s*/\s*([\d,]+)\s*tris", r.stdout)
    if m:
        measured["verts"] = int(m.group(1).replace(",", ""))
        measured["tris"] = int(m.group(2).replace(",", ""))
    v4.write_metadata(cdir, task="cull-mesh", algo=cull_algo, identity=cid,
                      resolved_inputs={"mesh": variant, "cameras": sid, "orient": oid},
                      settings=settings, mechanism="job", measured=measured,
                      extra={"gauge": gauge,
                             "source_mesh": str(mdir.relative_to(scene_dir))})
    print(f"[cull] wrote {cdir.relative_to(scene_dir)}/mesh.ply "
          f"({measured.get('verts','?')} verts / {measured.get('tris','?')} tris) in {dt}s")
    job_record(args.scene, "cull-mesh",
               [{"node": "cull-mesh", "identity": cid, "action": "EXECUTE",
                 "duration_s": dt}],
               {"scene": args.scene, "variant": variant})


def cmd_verify_frame(args):
    """Verification task (STO-SCN-089 DoD): does a fused mesh COINCIDE with
    the matcha reference in the shared gauge? Camera-residual gates can't
    see this failure class — geometry can disagree while cameras fit.
    Writes the measured verdict into the artifact's metadata (measurement
    annotation, locked #11-legitimate) + rankable:false on failure."""
    import numpy as np
    import open3d as o3d
    scene_dir = v4.STORE / args.scene
    ref = None
    for m in sorted(scene_dir.glob("represent/matcha/*/meshify/tsdf/*/mesh.ply")):
        ref = m
        break
    if ref is None:
        sys.exit("no matcha tsdf reference in scene")
    pt = o3d.geometry.PointCloud()
    pt.points = o3d.io.read_triangle_mesh(str(ref)).vertices
    kd = o3d.geometry.KDTreeFlann(pt)
    nodes = []
    for mdir in sorted(scene_dir.glob("represent/da3/*/meshify/*/*/")):
        mp = mdir / "mesh.ply"
        if not mp.exists():
            continue
        v = np.asarray(o3d.io.read_triangle_mesh(str(mp)).vertices)[::40]
        d = np.asarray([kd.search_knn_vector_3d(p_, 1)[2][0] ** 0.5 for p_ in v])
        med = float(np.median(d))
        ok = med <= 0.15
        md = json.loads((mdir / "metadata.json").read_text())
        md.setdefault("measured", {})["frame_check"] = {
            "vs": str(ref.relative_to(scene_dir)), "median_m": round(med, 3),
            "gate_m": 0.15, "pass": ok, "checked": NOW()}
        if not ok:
            md["rankable"] = False
            md["rankable_reason"] = (f"frame check FAILED: median {med:.2f}m from the matcha "
                                     f"reference in the shared gauge (STO-SCN-089 class — "
                                     f"cameras align, geometry does not)")
        (mdir / "metadata.json").write_text(json.dumps(md, indent=2) + "\n")
        nodes.append({"node": "verify-frame", "mesh": mdir.name,
                      "median_m": round(med, 3), "pass": ok})
        print(f"{mdir.parent.parent.parent.name}/{mdir.name}: median {med:.3f}m -> "
              f"{'PASS' if ok else 'FAIL (rankable:false)'}")
    job_record(args.scene, "verify-frame", nodes, {"scene": args.scene})


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("ingest")
    p.add_argument("scene")
    p.add_argument("--host", required=True)
    p.add_argument("--raw", default=None)
    p.add_argument("--capture-mode", default=None,
                   help="camera capture mode (fisheye|dewarped) — not in EXIF; "
                        "overrides <scene>/capture.json (STO-SCN-091)")
    p.add_argument("--camera-make", default=None, help="camera make (capture declaration)")
    p.add_argument("--camera-model", default=None, help="camera model (capture declaration)")
    p.set_defaults(fn=cmd_ingest)
    p = sp.add_parser("precull", help="pose-free pre-cull -> curated subset (STO-SCN-092)")
    p.add_argument("scene")
    p.add_argument("--target", type=int, default=pre.DEFAULTS["target"],
                   help="candidate ceiling (0 = no thin); default 300 = solve ceiling")
    p.add_argument("--phash-thresh", type=int, default=pre.DEFAULTS["phash_thresh"])
    p.add_argument("--blur-rel", type=float, default=pre.DEFAULTS["blur_rel"])
    p.add_argument("--max-gap", type=int, default=pre.DEFAULTS["max_gap"])
    p.add_argument("--dup-window", type=int, default=pre.DEFAULTS["dup_window"])
    p.add_argument("--score-edge", type=int, default=pre.DEFAULTS["score_edge"])
    p.add_argument("--set-primary", action="store_true",
                   help="set the curated subset as primary (deliberate operator act; "
                        "no-op if primary already set)")
    p.set_defaults(fn=cmd_precull)
    p = sp.add_parser("spine", help="spine segmentation -> M overlapping segments (STO-SCN-097)")
    p.add_argument("scene")
    p.add_argument("--cap", type=int, default=300, help="max frames per segment (solver capacity)")
    p.add_argument("--overlap", type=int, default=30, help="min shared frames per seam (budget)")
    p.add_argument("--snap", type=int, default=10, help="boundary snap search window")
    p.add_argument("--reg-thresh", type=int, default=12, help="max mean-pHash dist for a registrable seam")
    p.add_argument("--loop-thresh", type=int, default=8, help="max pHash dist for a loop candidate")
    p.add_argument("--loop-min-sep", type=int, default=2, help="min segment separation for a loop")
    p.add_argument("--loop-step", type=int, default=5, help="frame subsample stride for the loop scan")
    p.set_defaults(fn=cmd_spine)
    p = sp.add_parser("spine-register", help="register per-segment submaps into one gauge (STO-SCN-098)")
    p.add_argument("scene")
    p.add_argument("--spine", required=True, help="the spine@0 identity")
    p.add_argument("--solves", required=True, help="comma list seg=subset/solve (one per segment)")
    p.add_argument("--rel-tol", type=float, default=0.02, help="max seam residual frac of spread (gate)")
    p.set_defaults(fn=cmd_spine_register)
    p = sp.add_parser("spine-fuse", help="fuse per-segment gaussians into one gauge (STO-SCN-099)")
    p.add_argument("scene")
    p.add_argument("--spine", required=True, help="the spine@0 identity")
    p.add_argument("--register", required=True, help="the spine-register@0 identity")
    p.add_argument("--solves", required=True, help="comma list seg=subset/solve (segment cameras)")
    p.add_argument("--gaussians", required=True, help="comma list seg=ply-path (per-segment reconstruction)")
    p.add_argument("--radius", type=float, default=0.0, help="coverage falloff radius (0 = auto)")
    p.set_defaults(fn=cmd_spine_fuse)
    p = sp.add_parser("solve", help="GPU FastMap solve -> poses + sparse/0 (STO-SCN-093)")
    p.add_argument("scene")
    p.add_argument("--host", required=True)
    p.add_argument("--subset", default=None, help="subset id (default: primary)")
    p.set_defaults(fn=cmd_solve)
    p = sp.add_parser("covis", help="covis graph + validity gate from a fastmap solve (STO-SCN-093)")
    p.add_argument("scene")
    p.add_argument("--host", required=True)
    p.add_argument("--solve", required=True, help="the fastmap@0 solve identity")
    p.add_argument("--subset", default=None)
    p.add_argument("--min-overlap", type=int, default=15)
    p.set_defaults(fn=cmd_covis)
    p = sp.add_parser("select", help="best-N selection over a solve, gated by covis (STO-SCN-094/103)")
    p.add_argument("scene")
    p.add_argument("--solve", required=True, help="the fastmap@0 solve identity")
    p.add_argument("--covis", required=True, help="the covis@0 identity (must have PASSed validity)")
    p.add_argument("--subset", default=None)
    p.add_argument("--selector", choices=["voxel", "track"], default="voxel",
                   help="voxel = STO-SCN-103 coverage flux (default); track = STO-SCN-094 covisibility")
    p.add_argument("--n", type=int, default=24, help="target view count (downstream sweet spot)")
    p.add_argument("--grid", type=int, default=64, help="voxel grid resolution (voxel selector)")
    p.add_argument("--min-overlap", type=int, default=10, help="connectivity: shared pts vs selected set (track)")
    p.add_argument("--div-angle", type=float, default=25.0, help="viewpoint-diversity angle, track (0 = off)")
    p.set_defaults(fn=cmd_select)
    p = sp.add_parser("scout", help="DA3 scout gaussian for the verify surface (STO-SCN-095)")
    p.add_argument("scene")
    p.add_argument("--host", required=True)
    p.add_argument("--solve", required=True, help="the fastmap@0 solve identity")
    p.add_argument("--subset", default=None)
    p.add_argument("--n-scout", type=int, default=32, help="scout views (~DA3 ceiling)")
    p.add_argument("--res", type=int, default=504)
    p.add_argument("--selector", choices=["track", "voxel"], default="track",
                   help="track = coherent/overlap (default, clean DA3); voxel = the STO-SCN-103 coverage-selected N")
    p.add_argument("--grid", type=int, default=64, help="voxel grid resolution (voxel selector)")
    p.set_defaults(fn=cmd_scout)
    p = sp.add_parser("reconstruct-matcha")
    p.add_argument("scene")
    p.add_argument("--host", required=True)
    p.add_argument("--subset", default=None,
                   help="subset to reconstruct (default: primary). A FINAL-N selection is posed "
                        "from its parent solve — no re-solve (STO-SCN-130).")
    p.add_argument("--dense-regul", default="default", choices=["default", "strong"])
    p.add_argument("--mesh-res", dest="mesh_res", type=int, default=None,
                   help="TSDF mesh resolution (default config = 1024). Lower (e.g. 512) for "
                        "small-radius spine gauges where 1024 OOMs the host (STO-SCN-133).")
    p.add_argument("--sfm", default="unposed", choices=["unposed", "posed"],
                   help="posed = matcha@1: feed the ingest solve into train.py "
                        "as COLMAP sparse/0 (no re-solve, no arbitrary gauge)")
    p.set_defaults(fn=cmd_matcha)
    p = sp.add_parser("views-from-blend")
    p.add_argument("scene")
    p.add_argument("blend")
    p.set_defaults(fn=cmd_views)
    p = sp.add_parser("reconstruct-da3")
    p.add_argument("scene")
    p.add_argument("--host", required=True)
    p.add_argument("--sfm", default="unposed", choices=["unposed", "posed"],
                   help="posed = da3@1: feed the ingest solve into DA3 "
                        "inference(extrinsics=, intrinsics=) instead of "
                        "letting DA3 estimate its own cameras")
    p.set_defaults(fn=cmd_da3)
    p = sp.add_parser("reconstruct-da3-scout",
                      help="STO-SCN-127: matcha-FREE DA3 scene mesh in the solve/spine gauge "
                           "from an existing scout npz (no GPU, no matcha reference)")
    p.add_argument("scene")
    p.add_argument("--solve", required=True, help="the spine solve id")
    p.add_argument("--scout", required=True, help="the scout id whose da3_poses.npz to fuse")
    p.add_argument("--subset", default=None, help="subset id (default: primary)")
    p.add_argument("--conf-percentile", dest="conf_percentile", type=float, default=40.0)
    p.add_argument("--voxel-frac", dest="voxel_frac", type=float, default=0.004)
    p.set_defaults(fn=cmd_da3_scout)
    p = sp.add_parser("regauge-views")
    p.add_argument("scene")
    p.add_argument("--solve", required=True)
    p.add_argument("--from-orient", dest="from_orient", required=True)
    p.add_argument("--to-orient", dest="to_orient", required=True)
    p.add_argument("--sim", default=None,
                   help="optional weld->solve similarity json (file or inline)")
    p.set_defaults(fn=cmd_regauge_views)
    p = sp.add_parser("refilter",
                      help="STO-SCN-136 path A: ground a filter-re-extracted matcha tetra "
                           "(out/tetra_filtered/*.ply) into a new meshify-via-tetra-filtered node")
    p.add_argument("scene")
    p.add_argument("--rep", required=True, help="the matcha represent id whose tetra to re-ground")
    p.set_defaults(fn=cmd_refilter)
    p = sp.add_parser("mergefill",
                      help="STO-SCN-142: (A) merge & gap-fill via screened Poisson -> watertight "
                           "manifold condition node (CPU; runs on a meshify OR condition mesh)")
    p.add_argument("scene")
    p.add_argument("--variant", required=True, help="mesh id to fill (meshify or condition node)")
    p.add_argument("--method", choices=["fill-holes", "poisson"], default=None,
                   help="fill-holes = local gap-fill, preserves open scene (default); poisson = global seal")
    p.add_argument("--hole-size", dest="hole_size", type=float, default=None,
                   help="fill-holes: max hole boundary size to fill (default 0.3)")
    p.add_argument("--poisson-depth", dest="poisson_depth", type=int, default=None,
                   help="Poisson octree depth (task default 9)")
    p.add_argument("--density-quantile", dest="density_quantile", type=float, default=None,
                   help="drop verts below this density quantile (default 0.05; 0=off)")
    p.add_argument("--samples", type=int, default=None, help="surface sample points (default 1e6)")
    p.set_defaults(fn=cmd_mergefill)
    p = sp.add_parser("cull",
                      help="STO-SCN-136: post-meshify CPU cull (drop few-view / distant / "
                           "sub-floor verts) -> additive condition node; reuses the materialized "
                           "mesh, no GPU, NOOP on re-run")
    p.add_argument("scene")
    p.add_argument("--variant", required=True,
                   help="the meshify mesh identity to cull (tetra/tsdf/da3 mesh id)")
    p.add_argument("--min-views", dest="min_views", type=int, default=None,
                   help="drop verts seen by < N cameras (task default 2)")
    p.add_argument("--max-dist-from-cluster", dest="max_dist", type=float, default=None,
                   help="drop verts > D m from the camera centroid — sky/far killer (default 0=off)")
    p.add_argument("--cambox-expand", dest="cambox_expand", type=float, default=None,
                   help="STO-SCN-137: keep verts inside the posed-camera AABB +this/side, cull "
                        "outside (gravity-aligned); <0 = off (default)")
    p.add_argument("--floor-z-min", dest="floor_z_min", type=float, default=None,
                   help="drop verts with z < this in the oriented gauge (default -0.5)")
    p.add_argument("--image-size", dest="image_size", default=None,
                   help="WxH for the in-bounds view-count projection (default 1024,576)")
    p.add_argument("--primitives", dest="primitives", default=None,
                   help="STO-SCN-145: JSON file of boolean cull primitives (datum frame, meters); "
                        "keep verts inside the resulting solid (flows into content identity)")
    p.set_defaults(fn=cmd_cull)
    p = sp.add_parser("verify-frame")
    p.add_argument("scene")
    p.set_defaults(fn=cmd_verify_frame)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
