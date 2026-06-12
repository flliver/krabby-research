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
    mid = v4.identity_hash({"representation": rid, "cameras": sid}, {}, "tetra-extract@0")
    ts_settings = v4.hashable_settings(tdefs["meshify-via-tsdf"], {})
    tid = v4.identity_hash({"representation": rid, "cameras": sid}, ts_settings, "tsdf-extract@0")
    tetra_dir = rdir / "meshify" / "tetra" / mid
    tsdf_dir = rdir / "meshify" / "tsdf" / tid

    if (tetra_dir / "mesh.ply").exists() and (tsdf_dir / "mesh.ply").exists():
        print(f"NOOP: {rid} fully materialized")
        return

    # -- the @0 weld: ONE dispatch materializes represent + raw tetra + raw tsdf
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
    out = rdir / "out"
    tetra_raw = sorted((out / "tetra_meshes").glob("*.ply")) if (out / "tetra_meshes").is_dir() else []
    tsdf_raw = next(iter(sorted((out / "tsdf_meshes").glob("multires_tsdf_post*.ply"))), None)
    if rc != 0 or not tetra_raw or tsdf_raw is None:
        sys.exit(f"matcha weld FAILED (rc={rc}; tetra={bool(tetra_raw)} tsdf={bool(tsdf_raw)}; "
                 f"log {rdir}/matcha.log)")
    dig, _ = host_digest(args.host, MATCHA_IMAGE)
    v4.write_metadata(rdir, task="represent-via-matcha", algo="matcha@0", identity=rid,
                      resolved_inputs={"subset": sub, "cameras": sid},
                      settings=r_settings, mechanism="job",
                      measured={"host": args.host.split("@")[-1], "duration_s": dt,
                                "image_digest": dig})
    nodes.append({"node": "represent", "identity": rid, "action": "EXECUTE",
                  "host": args.host, "duration_s": dt})

    # -- node: orient-cameras (bootstrap-mesh — IN-GRAPH; method per STO-SCN-082)
    sys.path.insert(0, str(Path(__file__).parent))
    import open3d as o3d  # noqa: F401  (ensures availability before work)
    o_settings = {"method": "bootstrap-mesh", "ransac_dist": 0.05}
    oid = v4.identity_hash({"solve": sid}, o_settings, "orient-floor@0")
    odir = scene_dir / "images" / "subsets" / sub / "cameras" / sid / "orient" / oid
    R, z = bootstrap_orient(tsdf_raw)
    odir.mkdir(parents=True, exist_ok=True)
    (odir / "transform.json").write_text(json.dumps({"rotation": R.tolist(),
                                                     "z_shift": float(z)}, indent=2) + "\n")
    # oriented cameras file (renderer contract: rotation + z_shift + cams)
    (odir / "oriented.json").write_text(json.dumps({"rotation": R.tolist(),
                                                    "z_shift": float(z)}, indent=2) + "\n")
    v4.write_metadata(odir, task="orient-cameras", algo="orient-floor@0", identity=oid,
                      resolved_inputs={"solve": sid}, settings=o_settings, mechanism="job")
    nodes.append({"node": "orient", "identity": oid, "action": "EXECUTE"})

    # -- meshify: apply gauge -> canonical mesh.ply (tetra + tsdf)
    for src_mesh, mdir, task, algo, msettings in (
            (tetra_raw[-1], tetra_dir, "meshify-via-tetra", "tetra-extract@0", {}),
            (tsdf_raw, tsdf_dir, "meshify-via-tsdf", "tsdf-extract@0", ts_settings)):
        mdir.mkdir(parents=True, exist_ok=True)
        ground_mesh(src_mesh, mdir / "mesh.ply", R, z)
        v4.write_metadata(mdir, task=task, algo=algo, identity=mdir.name,
                          resolved_inputs={"representation": rid, "cameras": sid},
                          settings=msettings, mechanism="job",
                          extra={"gauge": str(odir.relative_to(scene_dir))})
        nodes.append({"node": task, "identity": mdir.name, "action": "EXECUTE"})
    # canonical gauge marker for the renderer
    md = json.loads((rdir / "metadata.json").read_text())
    md["canonical_gauge"] = str((odir / "oriented.json").relative_to(scene_dir))
    (rdir / "metadata.json").write_text(json.dumps(md, indent=2) + "\n")
    job_record(args.scene, "reconstruct-matcha", nodes,
               {"scene": args.scene, "host": args.host, "dense_regul": args.dense_regul})
    print(f"reconstruct-matcha materialized: represent {rid}, tetra {mid}, tsdf {tid}, orient {oid}")


def bootstrap_orient(mesh_path: Path):
    """RANSAC floor fit on the dense TSDF mesh (the validated STO-SCN-004
    method, adopted by STO-SCN-082). Returns (R 3x3, z_shift)."""
    import numpy as np
    import open3d as o3d
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    pcd = o3d.geometry.PointCloud()
    v = np.asarray(mesh.vertices)
    step = max(1, len(v) // 200_000)
    pcd.points = o3d.utility.Vector3dVector(v[::step])
    plane, _ = pcd.segment_plane(distance_threshold=0.05, ransac_n=3, num_iterations=1000)
    n = np.asarray(plane[:3], dtype=float)
    n /= np.linalg.norm(n)
    # mesh bulk should be ABOVE the floor: flip normal toward the centroid side
    c = v.mean(0)
    if np.dot(n, c) + plane[3] < 0:
        n, plane = -n, [-x for x in plane]
    z_axis = np.array([0.0, 0.0, 1.0])
    vv = np.cross(n, z_axis)
    s = np.linalg.norm(vv)
    if s < 1e-9:
        R = np.eye(3)
    else:
        c_ = float(np.dot(n, z_axis))
        vx = np.array([[0, -vv[2], vv[1]], [vv[2], 0, -vv[0]], [-vv[1], vv[0], 0]])
        R = np.eye(3) + vx + vx @ vx * ((1 - c_) / (s ** 2))
    z_floor = float(np.percentile((v @ R.T)[:, 2], 2))
    return R, -z_floor


def ground_mesh(src: Path, dst: Path, R, z):
    import numpy as np
    import open3d as o3d
    mesh = o3d.io.read_triangle_mesh(str(src))
    vv = np.asarray(mesh.vertices)
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
    odirs = sorted((scene_dir / "images" / "subsets" / sub / "cameras" / sid / "orient").glob("*/"))
    if not odirs:
        sys.exit("no orient gauge — run reconstruct-matcha first (bootstrap)")
    oid = odirs[0].name
    r_settings = v4.hashable_settings(tdefs["represent-via-da3"], {})
    rid = v4.identity_hash({"subset": sub, "orient": oid}, r_settings, "da3@0")
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
                          resolved_inputs={"subset": sub, "orient": oid},
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

    # -- fuse (local CPU): depths -> mesh, aligned into the orient gauge
    ts = v4.hashable_settings(tdefs["meshify-via-tsdf"], {})
    fid = v4.identity_hash({"representation": rid, "cameras": sid},
                           {"voxel_frac": 0.004, "conf_percentile": 40}, "tsdf-extract@0")
    fdir = rdir / "meshify" / "tsdf" / fid
    if (fdir / "mesh.ply").exists():
        nodes.append({"node": "fuse", "identity": fid, "action": "NOOP"})
    else:
        fuse_da3(scene_dir, rdir, sub, sid, oid, fdir)
        v4.write_metadata(fdir, task="meshify-via-tsdf", algo="tsdf-extract@0", identity=fid,
                          resolved_inputs={"representation": rid, "cameras": sid},
                          settings={"voxel_frac": 0.004, "conf_percentile": 40},
                          mechanism="job")
        nodes.append({"node": "fuse", "identity": fid, "action": "EXECUTE"})
    job_record(args.scene, "reconstruct-da3", nodes,
               {"scene": args.scene, "host": args.host})
    print(f"reconstruct-da3 materialized: represent {rid}, fused {fid}")


def fuse_da3(scene_dir: Path, rdir: Path, sub: str, sid: str, oid: str, fdir: Path):
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
    mesh.compute_vertex_normals()
    fdir.mkdir(parents=True, exist_ok=True)
    o3d.io.write_triangle_mesh(str(fdir / "mesh.ply"), mesh)
    print(f"[fuse] {len(mesh.vertices):,} verts -> {fdir}/mesh.ply")


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
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
