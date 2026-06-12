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
                                              matcha@0 weld (host GPU) ->
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

SCRATCH = "/home/jeremy/scratch/v4exec"
MATCHA_IMAGE = "j.pski.org:5000/krabby-matcha:0.2.2-selfcontained"
DA3_IMAGE = "j.pski.org:5000/krabby-da3:0.4"
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


# ============================================================ reconstruct-matcha

def cmd_matcha(args):
    import numpy as np
    scene_dir = v4.STORE / args.scene
    sc = v4.Scene(args.scene)
    sub = sc.resolve("primary")
    tdefs = v4.tasks()
    nodes = []
    r_settings = v4.hashable_settings(tdefs["represent-via-matcha"],
                                      {"dense_regul": args.dense_regul})
    solve_dirs = sorted((scene_dir / "images" / "subsets" / sub / "cameras").glob("*/"))
    solve_dirs = [d for d in solve_dirs if (d / "metadata.json").exists()]
    if not solve_dirs:
        sys.exit("no solve for primary — run ingest first")
    sid = solve_dirs[0].name
    rid = v4.identity_hash({"subset": sub, "cameras": sid}, r_settings, "matcha@0")
    rdir = scene_dir / "represent" / "matcha" / rid
    o_settings, o_algo = ORIENT_SETTINGS, ORIENT_ALGO
    # the orient reads the bootstrap mesh (z-floor) -> it IS a resolved input
    oid = v4.identity_hash({"solve": sid, "bootstrap_rep": rid}, o_settings, o_algo)
    # gauge is part of the mesh content -> orient is a resolved input of meshify
    mid = v4.identity_hash({"representation": rid, "cameras": sid, "orient": oid},
                           {}, "tetra-extract@1")
    ts_settings = v4.hashable_settings(tdefs["meshify-via-tsdf"], {})
    tid = v4.identity_hash({"representation": rid, "cameras": sid, "orient": oid},
                           ts_settings, "tsdf-extract@1")
    tetra_dir = rdir / "meshify" / "tetra" / mid
    tsdf_dir = rdir / "meshify" / "tsdf" / tid

    if (tsdf_dir / "mesh.ply").exists() and (
            (tetra_dir / "mesh.ply").exists()
            or not (rdir / "out" / "tetra_meshes").is_dir()
            or not list((rdir / "out" / "tetra_meshes").glob("*.ply"))):
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
        tool = (f"python train.py -s /work/images -o /work/out --sfm_config unposed "
                f"--n_images {n_images} --alignment_config strong")
        if args.dense_regul != "default":
            tool += f" --dense_regul {args.dense_regul}"
        tool += (" --depthanythingv2_checkpoint_dir /opt/MAtCha/Depth-Anything-V2/checkpoints"
                 " --depthanything_encoder vitl")
        tool += (" && PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "
                 "python scripts/extract_tsdf_mesh.py -s /work/out/mast3r_sfm "
                 "-m /work/out/free_gaussians -o /work/out/tsdf_meshes -c default")
        print(f"[matcha@0 weld] {args.host}: full pipeline ({n_images} images, "
              f"dense_regul={args.dense_regul}) — ~15-25 min")
        t0 = datetime.datetime.now()
        rc = run_in_matcha(args.host, workdir, tool, rdir / "matcha.log")
        dt = int((datetime.datetime.now() - t0).total_seconds())
        print(f"[matcha@0 weld] rc={rc} in {dt}s; gathering…")
        rdir.mkdir(parents=True, exist_ok=True)
        sh(["rsync", "-a", f"{args.host}:{workdir}/out/", str(rdir / "out") + "/"])
        sh(["ssh", args.host, f"rm -rf {SCRATCH}/{tag}"])
        tetra_raw = sorted((out / "tetra_meshes").glob("*.ply")) if (out / "tetra_meshes").is_dir() else []
        tsdf_raw = next(iter(sorted((out / "tsdf_meshes").glob("multires_tsdf_post*.ply"))), None)
        if rc != 0 or tsdf_raw is None:
            sys.exit(f"matcha weld FAILED (rc={rc}; tetra={bool(tetra_raw)} tsdf={bool(tsdf_raw)}; "
                     f"log {rdir}/matcha.log)")
        if not tetra_raw:
            print("[matcha@0 weld] WARNING: tetra stage produced no mesh "
                  "(marching_tetrahedra flake, 010 class) — continuing tsdf-only")
        dig, _ = host_digest(args.host, MATCHA_IMAGE)
        v4.write_metadata(rdir, task="represent-via-matcha", algo="matcha@0", identity=rid,
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
            v4.write_metadata(rdir, task="represent-via-matcha", algo="matcha@0",
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
    sim = weld_to_solve_sim(scene_dir, sub, sid, out)
    print(f"[gauge-sim] weld->solve: scale {sim['s']:.4f} rot {sim['rot_deg']:.1f} deg "
          f"max residual {sim['max_residual']:.4f}")
    if sim["residual_frac"] > 0.02:
        sys.exit(f"matcha REFUSED: weld->solve camera similarity residual "
                 f"{sim['residual_frac']:.1%} > 2% — solves disagree beyond gauge")

    # -- node: orient-cameras (horizon up prior; z-floor from the SOLVE-framed mesh)
    odir = scene_dir / "images" / "subsets" / sub / "cameras" / sid / "orient" / oid
    if (odir / "oriented.json").exists():
        g = json.loads((odir / "oriented.json").read_text())
        R, z = np.asarray(g["rotation"]), float(g["z_shift"])
        nodes.append({"node": "orient", "identity": oid, "action": "NOOP"})
    else:
        cams = json.loads((scene_dir / "images" / "subsets" / sub / "cameras" / sid /
                           "cameras.json").read_text())
        c2w = np.asarray(cams["cams2world"])
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
               {"scene": args.scene, "host": args.host, "dense_regul": args.dense_regul})
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
    if o.type == "CAMERA" and (o.users_collection and any("virtual" in c.name for c in o.users_collection) or o.get("view_purpose")):
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
    next_slot = (existing[-1] + 1) if existing else 1
    for vw in views:
        slot = f"{next_slot:02d}"
        next_slot += 1
        vdir = scene_dir / "views" / slot
        vdir.mkdir(parents=True, exist_ok=True)
        content = {k: v for k, v in vw.items() if k != "name"}
        (vdir / "view.json").write_text(json.dumps(content, indent=2, sort_keys=True) + "\n")
        (vdir / "metadata.json").write_text(json.dumps({
            "schema": 4, "captured_name": vw["name"], "mechanism": "operator-capture",
            "written": NOW()}, indent=2) + "\n")
        slots.append(slot)
        print(f"view {slot}: '{vw['name']}' lens {vw['lens_mm']}mm")
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
    # inference is gauge-independent (the npz never sees the orient); the
    # gauge enters at the FUSE step (STO-SCN-089-2)
    rid = v4.identity_hash({"subset": sub}, r_settings, "da3@0")
    rdir = scene_dir / "represent" / "da3" / rid
    if not (rdir / "metadata.json").exists():
        tag = f"{args.scene}-da3-{rid}"
        workdir = stage_images_on_host(args.host, scene_dir, sub, tag)
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
        v4.write_metadata(rdir, task="represent-via-da3", algo="da3@0", identity=rid,
                          resolved_inputs={"subset": sub},
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


def matcha_reference(scene_dir: Path, sub: str, sid: str, tdefs):
    """Resolve the matcha reference tsdf mesh for this solve.

    Deterministic recompute first (native-run scenes, default settings);
    falls back to a store scan for migrated scenes whose identities were
    minted under other settings. Returns (mesh_identity, mesh_path) or
    (None, None)."""
    r_settings = v4.hashable_settings(tdefs["represent-via-matcha"],
                                      {"dense_regul": "default"})
    rid_m = v4.identity_hash({"subset": sub, "cameras": sid}, r_settings, "matcha@0")
    ts_settings = v4.hashable_settings(tdefs["meshify-via-tsdf"], {})
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
    p.set_defaults(fn=cmd_ingest)
    p = sp.add_parser("reconstruct-matcha")
    p.add_argument("scene")
    p.add_argument("--host", required=True)
    p.add_argument("--dense-regul", default="default", choices=["default", "strong"])
    p.set_defaults(fn=cmd_matcha)
    p = sp.add_parser("views-from-blend")
    p.add_argument("scene")
    p.add_argument("blend")
    p.set_defaults(fn=cmd_views)
    p = sp.add_parser("reconstruct-da3")
    p.add_argument("scene")
    p.add_argument("--host", required=True)
    p.set_defaults(fn=cmd_da3)
    p = sp.add_parser("regauge-views")
    p.add_argument("scene")
    p.add_argument("--solve", required=True)
    p.add_argument("--from-orient", dest="from_orient", required=True)
    p.add_argument("--to-orient", dest="to_orient", required=True)
    p.add_argument("--sim", default=None,
                   help="optional weld->solve similarity json (file or inline)")
    p.set_defaults(fn=cmd_regauge_views)
    p = sp.add_parser("verify-frame")
    p.add_argument("scene")
    p.set_defaults(fn=cmd_verify_frame)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
