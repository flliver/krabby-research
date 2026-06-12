#!/usr/bin/env python3
"""migrate_v4.py — full-restructure migration to store-shape v4 (STO-SCN-080).

HUG-SCN-005 locked #9: restructure the existing store into the
content-addressed layout; identities COMPUTED (not invented) from the
inputs + spec settings that already exist; legacy executions get
retroactive algo@0; compute work (solves, representations, meshes,
renders) MOVES — nothing recomputed, nothing left in legacy shape.

    python3 real2sim/migrate_v4.py <scene> [--apply]     # default: dry-run
    python3 real2sim/migrate_v4.py all [--apply]

Dry-run prints the action plan. --apply moves files on disk (git add -A
afterwards records renames; LFS OIDs unchanged so no re-upload),
writes per-identity metadata.json (migrated: true, origin recorded,
unknowables explicit — T-002), and logs ONE migration job per scene
(mechanism: migrate, locked #8).
"""
from __future__ import annotations

import datetime
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import v4core as v4

STORE = v4.STORE
IMG_EXTS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
VID_EXTS = {".mp4", ".mkv", ".mov"}

NOW = datetime.datetime.now().astimezone().isoformat(timespec="seconds")


class Mig:
    """Collects actions; executes them on --apply."""

    def __init__(self, scene: str, apply: bool):
        self.scene, self.apply = scene, apply
        self.dir = STORE / scene
        self.actions: list[str] = []
        self.pool: dict[str, str] = {}        # image content hash -> new rel path
        self.by_origin: dict[str, str] = {}   # old rel path -> image hash
        self.moved: set[str] = set()          # sources already consumed (dry-run truth)

    def log(self, s: str):
        self.actions.append(s)

    def is_moved(self, p: Path) -> bool:
        rp = str(p)
        return any(rp == m or rp.startswith(m + "/") for m in self.moved)

    def move(self, src: Path, dst: Path, kind="mv"):
        if self.is_moved(src):
            return
        self.log(f"{kind}: {src.relative_to(self.dir)} -> {dst.relative_to(self.dir)}")
        self.moved.add(str(src))
        if self.apply:
            dst.parent.mkdir(parents=True, exist_ok=True)
            if not dst.exists():
                shutil.move(str(src), str(dst))
            elif src.exists():
                # identical-identity duplicate (e.g. anchor copies) — keep as origin residue
                shutil.move(str(src), str(dst.parent / f"origin-dup-{src.name}"))

    def copy(self, src: Path, dst: Path):
        self.log(f"cp: {src.relative_to(self.dir)} -> {dst.relative_to(self.dir)}")
        if self.apply:
            dst.parent.mkdir(parents=True, exist_ok=True)
            if not dst.exists():
                shutil.copy2(str(src), str(dst))

    def write_json(self, path: Path, obj):
        self.log(f"write: {path.relative_to(self.dir)}")
        if self.apply:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")

    def metadata(self, out_dir: Path, **kw):
        self.log(f"metadata: {out_dir.relative_to(self.dir)}/metadata.json")
        if self.apply:
            v4.write_metadata(out_dir, migrated=True, **kw)

    # ---------------- phase 1: videos + image pool ----------------

    def ingest_image(self, f: Path, origin_note: str) -> str:
        h = v4.file_hash(f)
        rel = str(f.relative_to(self.dir))
        self.by_origin[rel] = h
        if h in self.pool:
            self.log(f"dup: {rel} == images/{h} (selection copy, not re-pooled)")
            if self.apply and f.exists():
                f.unlink()
            return h
        ext = f.suffix.lower().lstrip(".") or "img"
        dst = self.dir / "images" / h / f"image.{ext}"
        self.move(f, dst)
        self.write_json(self.dir / "images" / h / "metadata.json",
                        {"schema": 4, "original_name": f.name, "origin": origin_note,
                         "migrated": True, "written": NOW})
        self.pool[h] = f"images/{h}"
        return h

    def phase_pool(self):
        # re-runnability: preload pool + origin map from already-migrated images
        for md in self.dir.glob("images/*/metadata.json"):
            d = json.loads(md.read_text())
            h = md.parent.name
            self.pool[h] = f"images/{h}"
            if d.get("origin") and d.get("original_name"):
                self.by_origin[f"{d['origin'].rstrip('/')}/{d['original_name']}"] = h
        inp = self.dir / "input"
        if not inp.is_dir():
            return
        for f in sorted(inp.iterdir()):
            if f.suffix.lower() in VID_EXTS:
                self.move(f, self.dir / "videos" / f.stem / f"video{f.suffix.lower()}")
        for pdir in sorted(inp.iterdir()):
            # pool sources: src/, images/, and capture-session dirs (e.g.
            # 005's 5_10_2023). preproc-*/spine-* are handled elsewhere.
            if not pdir.is_dir() or pdir.name.startswith(("preproc-", "spine-")):
                continue
            for f in sorted(pdir.iterdir()):
                if f.suffix in IMG_EXTS:
                    self.ingest_image(f, f"input/{pdir.name}")

    # ---------------- phase 2: preprocs -> subsets / pool additions / solves ----

    def subset_for(self, hashes: list[str], mechanism: str, label: str,
                   origin: str, settings: dict | None = None) -> str:
        sh = v4.hoh(hashes)
        sdir = self.dir / "images" / "subsets" / sh
        if not (sdir / "subset.json").exists() or not self.apply:
            self.write_json(sdir / "subset.json",
                            {"schema": 4, "members": sorted(hashes)})
            self.write_json(sdir / "metadata.json",
                            {"schema": 4, "mechanism": mechanism, "label": label,
                             "settings": settings or {}, "origin": origin,
                             "migrated": True, "written": NOW})
        return sh

    def dir_image_hashes(self, d: Path, origin: str) -> list[str]:
        out = []
        for f in sorted(d.iterdir()):
            if f.suffix in IMG_EXTS:
                out.append(self.ingest_image(f, origin))
        return out

    def phase_preproc(self) -> dict[str, str]:
        """Returns map: old preproc rel-path -> subset hash (for run input mapping)."""
        sub_by_path = {}
        inp = self.dir / "input"
        if not inp.is_dir():
            return sub_by_path
        for pp in sorted(inp.glob("preproc-*")):
            spec = {}
            if (pp / "specification.json").exists():
                spec = json.loads((pp / "specification.json").read_text())
            params = spec.get("parameters", {})
            data = pp / "data"
            label = pp.name
            if "pool-sfm" in pp.name or "sfm" in str(spec.get("kind", "")):
                # a real pool solve over the PRECEDING pool subset: migrate its
                # cameras into that subset (compute preserved, locked #9)
                pool_sub = None
                for prior in sorted(inp.glob("preproc-*")):
                    pk = str(prior.relative_to(self.dir))
                    if prior.name < pp.name and sub_by_path.get(pk) not in (None, "POOL_SOLVE"):
                        pool_sub = sub_by_path[pk]
                if pool_sub is None:
                    # re-run path: prior preprocs already migrated -> look up
                    # the subset whose recorded origin is the preceding preproc
                    best_n = -1
                    for md in self.dir.glob("images/subsets/*/metadata.json"):
                        d = json.loads(md.read_text())
                        org = str(d.get("origin", ""))
                        if org.startswith("input/preproc-") and org.split("/")[1] < pp.name:
                            n = len(json.loads((md.parent / "subset.json").read_text())["members"])
                            if n > best_n:
                                best_n, pool_sub = n, md.parent.name
                cams = sorted(pp.rglob("mast3r_sfm/cameras.json"))
                if pool_sub and cams:
                    s_settings = {"sfm_config": "unposed", "chunk_size": 300, "overlap": 50}
                    sid = v4.identity_hash({"subset": pool_sub}, s_settings, "mast3r-sfm@0")
                    sdir = self.dir / "images" / "subsets" / pool_sub / "cameras" / sid
                    self.move(cams[0], sdir / "cameras.json")
                    pts = sorted(pp.rglob("mast3r_sfm/points.ply"))
                    if pts:
                        self.move(pts[0], sdir / "points.ply")
                    self.metadata(sdir, task="solve-cameras", algo="mast3r-sfm@0",
                                  identity=sid, resolved_inputs={"subset": pool_sub},
                                  settings=s_settings, mechanism="migrate",
                                  origin=str(pp.relative_to(self.dir)))
                    self.log(f"pool-solve {pp.name} -> subsets/{pool_sub}/cameras/{sid}")
                else:
                    self.log(f"WARN: pool solve {pp.name}: no preceding pool subset or no cameras.json")
                sub_by_path[str(pp.relative_to(self.dir))] = "POOL_SOLVE"
                continue
            if data.is_dir():
                hashes = self.dir_image_hashes(data, str(pp.relative_to(self.dir)))
                if hashes:
                    mech = params.get("selection_method", spec.get("kind", "preproc"))
                    sh = self.subset_for(hashes, str(mech), label,
                                         str(pp.relative_to(self.dir)), params)
                    sub_by_path[str(pp.relative_to(self.dir)) + "/data"] = sh
                    sub_by_path[str(pp.relative_to(self.dir))] = sh
            # retire the preproc dir; keep its spec/results as subset provenance
            for j in ("specification.json", "results.json"):
                if (pp / j).exists() and sub_by_path.get(str(pp.relative_to(self.dir))) not in (None, "POOL_SOLVE"):
                    sh = sub_by_path[str(pp.relative_to(self.dir))]
                    self.copy(pp / j, self.dir / "images" / "subsets" / sh / f"origin-{j}")
        return sub_by_path

    # ---------------- phase 3: runs -> cameras / represent / meshify / renders ----

    def run_subset(self, spec: dict, sub_by_path: dict) -> str | None:
        inputs = spec.get("inputs", [])
        for i in inputs:
            i = i.rstrip("/")
            if i in sub_by_path and sub_by_path[i] != "POOL_SOLVE":
                return sub_by_path[i]
        # whole-pool runs (input/src, input/images)
        if any(i.rstrip("/").endswith(("input/src", "input/images", "src")) for i in inputs):
            src_hashes = [h for o, h in self.by_origin.items()
                          if o.startswith(("input/src/", "input/images/"))]
            if src_hashes:
                return self.subset_for(sorted(set(src_hashes)), "all", "whole-pool",
                                       "input pool")
        return None

    def migrate_run(self, rdir: Path, sub_by_path: dict, views: dict):
        pl = rdir.parent.name.removeprefix("pipeline-")
        rn = rdir.name.removeprefix("run-")
        variant = f"{pl}--{rn}"
        run_meta = json.loads((rdir / "run.json").read_text()) if (rdir / "run.json").exists() else {}
        tdirs = sorted(rdir.glob("transform-*"))
        if not tdirs:   # render-variant: renders attach to source run's mesh later
            self.log(f"render-variant {variant}: deferred to render phase (source {run_meta.get('source_run')})")
            return {"variant": variant, "kind": "render-variant",
                    "source_run": run_meta.get("source_run"), "rdir": rdir}
        t = tdirs[0]
        spec = json.loads((t / "specification.json").read_text()) if (t / "specification.json").exists() else {}
        params = dict(spec.get("parameters", {}))
        # v3 studio runs: provenance lives in run_record.json, not a spec
        if not spec and (rdir / "run_record.json").exists():
            rr = json.loads((rdir / "run_record.json").read_text())
            for node_settings in rr.get("instance", {}).get("expanded_settings", {}).values():
                if isinstance(node_settings, dict):
                    params.update(node_settings)
            spec = {"inputs": ["input/src"], "parameters": params,
                    "_from": "run_record.json (v3 studio trigger)"}
        subset = self.run_subset(spec, sub_by_path)
        if subset is None:
            subset = "UNKNOWN-" + v4.content_hash(variant.encode())[:8]
            self.log(f"WARN {variant}: consumed images unresolvable -> placeholder subset {subset} (T-002)")
        data = t / "data"

        # ---- cameras (solve + orient) from the run's own solve
        solve_id = orient_id = None
        cams = data / "mast3r_sfm" / "cameras.json"
        if cams.exists():
            s_settings = {"sfm_config": params.get("sfm_config", "unposed"),
                          "chunk_size": 300, "overlap": 50}
            solve_id = v4.identity_hash({"subset": subset}, s_settings, "mast3r-sfm@0")
            sdir = self.dir / "images" / "subsets" / subset / "cameras" / solve_id
            self.move(cams, sdir / "cameras.json")
            if (data / "mast3r_sfm" / "points.ply").exists():
                self.move(data / "mast3r_sfm" / "points.ply", sdir / "points.ply")
            self.metadata(sdir, task="solve-cameras", algo="mast3r-sfm@0",
                          identity=solve_id, resolved_inputs={"subset": subset},
                          settings=s_settings, mechanism="migrate",
                          origin=str(cams.relative_to(self.dir)))
            ori = data / "oriented" / "oriented_cameras.json"
            if ori.exists():
                o_settings = {"method": "bootstrap-mesh", "ransac_dist": 0.05}
                orient_id = v4.identity_hash({"solve": solve_id}, o_settings, "orient-floor@0")
                odir = sdir / "orient" / orient_id
                self.copy(ori, odir / "oriented.json")
                if self.apply:
                    od = json.loads(ori.read_text())
                    (odir / "transform.json").parent.mkdir(parents=True, exist_ok=True)
                    (odir / "transform.json").write_text(json.dumps(
                        {"rotation": od.get("rotation"), "z_shift": od.get("z_shift")},
                        indent=2) + "\n")
                else:
                    self.log(f"write: {odir.relative_to(self.dir)}/transform.json")
                self.metadata(odir, task="orient-cameras", algo="orient-floor@0",
                              identity=orient_id, resolved_inputs={"solve": solve_id},
                              settings=o_settings, mechanism="migrate",
                              origin=str(ori.relative_to(self.dir)))

        # ---- represent
        algo = {"matcha": "matcha@0", "da3": "da3@0"}.get(pl, f"{pl}@legacy")
        known = {"dense_regul", "n_iters", "encoder", "alignment_config", "dense_pruning",
                 "process_res", "mode"}
        r_settings = {k: v for k, v in params.items() if k in known}
        if pl == "da3":
            r_settings.setdefault("mode", "nogs" if params.get("infer_gs") is False else "gs")
            r_settings.setdefault("process_res", params.get("process_res", 504))
        rin = {"subset": subset}
        if pl == "matcha" and solve_id:
            rin["cameras"] = solve_id
        if pl == "da3" and orient_id:
            rin["orient"] = orient_id
        rid = v4.identity_hash(rin, r_settings, algo)
        rdir_new = self.dir / "represent" / pl / rid
        moved_rep = False
        for payload in ("free_gaussians", "exports", "gs_ply", "gs_video", "depth_vis",
                        "charts", "run_logs"):
            if (data / payload).is_dir():
                self.move(data / payload, rdir_new / payload)
                moved_rep = True
        for j in ("specification.json", "results.json"):
            if (t / j).exists():
                self.copy(t / j, rdir_new / f"origin-{j}")
        self.metadata(rdir_new, task=f"represent-via-{pl}", algo=algo, identity=rid,
                      resolved_inputs=rin, settings=r_settings, mechanism="migrate",
                      origin=str(t.relative_to(self.dir)),
                      extra={"legacy_variant": variant,
                             "provenance_note": "migrated; image digest in origin-results.json; "
                                                "fields not recorded at run time are absent (T-002)"})

        # ---- meshify branches
        mesh_dirs = {}
        tetra_src = None
        for cand in sorted((data / "oriented").glob("oriented_tetra.ply")) if (data / "oriented").is_dir() else []:
            tetra_src = cand
        tetra_raw = sorted((data / "tetra_meshes").glob("*.ply")) if (data / "tetra_meshes").is_dir() else []
        if tetra_src or tetra_raw:
            mid = v4.identity_hash({"representation": rid, "cameras": solve_id or "unknown"},
                                   {}, "tetra-extract@0")
            mdir = rdir_new / "meshify" / "tetra" / mid
            if tetra_src:
                self.move(tetra_src, mdir / "mesh.ply")
            for raw in tetra_raw:
                self.move(raw, mdir / raw.name)
            self.metadata(mdir, task="meshify-via-tetra", algo="tetra-extract@0",
                          identity=mid, resolved_inputs={"representation": rid},
                          settings={}, mechanism="migrate", origin=variant)
            mesh_dirs["oriented"] = mdir
            # conditioned variants nest under tetra
            if (data / "oriented").is_dir():
                for cond in sorted((data / "oriented").glob("oriented_tetra_conditioned_*.ply")):
                    rec = data / "oriented" / f"tetra_condition_record_{cond.stem.rsplit('_',1)[-1]}.json"
                    c_settings = {"target_tris": 1000000, "taubin_iters": 10}
                    if rec.exists():
                        c_settings = json.loads(rec.read_text()).get("parameters", c_settings)
                        c_settings = {k: v for k, v in c_settings.items()
                                      if k in ("target_tris", "taubin_iters")}
                    cid = v4.identity_hash({"mesh": mid}, c_settings, "tetra-condition@0")
                    cdir = mdir / "condition" / cid
                    self.move(cond, cdir / "mesh.ply")
                    if rec.exists():
                        self.move(rec, cdir / "origin-record.json")
                    self.metadata(cdir, task="condition", algo="tetra-condition@0",
                                  identity=cid, resolved_inputs={"mesh": mid},
                                  settings=c_settings, mechanism="migrate", origin=variant)
                    mesh_dirs["conditioned"] = cdir
        tsdf = data / "tsdf_meshes" / "multires_tsdf_post_oriented.ply"
        if tsdf.exists():
            ts_settings = {"mesh_res": 1024, "config": "default"}
            tid = v4.identity_hash({"representation": rid, "cameras": solve_id or "unknown"},
                                   ts_settings, "tsdf-extract@0")
            tdir2 = rdir_new / "meshify" / "tsdf" / tid
            self.move(tsdf, tdir2 / "mesh.ply")
            if (data / "tsdf_meshes" / "fusion_record.json").exists():
                self.move(data / "tsdf_meshes" / "fusion_record.json", tdir2 / "origin-record.json")
            self.metadata(tdir2, task="meshify-via-tsdf", algo="tsdf-extract@0",
                          identity=tid, resolved_inputs={"representation": rid},
                          settings=ts_settings, mechanism="migrate", origin=variant)
            mesh_dirs["tsdf"] = mesh_dirs.get("tsdf", tdir2)

        # remaining transient payloads (oriented leftovers, logs, misc) -> represent dir
        if data.is_dir():
            for leftover in sorted(data.iterdir()):
                if self.is_moved(leftover):
                    continue
                if leftover.name in ("mast3r_sfm", "oriented", "tetra_meshes", "tsdf_meshes"):
                    for f in sorted(leftover.rglob("*")):
                        if f.is_file() and not self.is_moved(f):
                            self.move(f, rdir_new / "origin-data" / leftover.name / f.name)
                elif leftover.is_dir():
                    self.move(leftover, rdir_new / "origin-data" / leftover.name)
                elif leftover.is_file():
                    self.move(leftover, rdir_new / "origin-data" / leftover.name)

        # residue sweep: whatever remains in the old run dir (run.json,
        # run_record.json, blends, logs, spec/results originals) -> origin-run/
        for f in sorted(rdir.rglob("*")):
            if f.is_file() and not self.is_moved(f) and "renders" not in f.parts:
                self.move(f, rdir_new / "origin-run" / f.relative_to(rdir))

        return {"variant": variant, "kind": "full", "rid": rid, "rdir_new": rdir_new,
                "mesh_dirs": mesh_dirs, "old_rdir": rdir}

    # ---------------- phase 4: views ----------------

    def phase_views(self) -> dict[str, dict]:
        """old view name -> {slot, content_hash}. From scene cameras.json (schema 5)."""
        out = {}
        cj = self.dir / "cameras.json"
        if not cj.exists():
            return out
        doc = json.loads(cj.read_text())
        views = doc.get("views", doc.get("cameras", []))
        if isinstance(views, dict):
            views = [dict(v, name=k) for k, v in views.items()]
        slots = []
        for i, vw in enumerate(sorted(views, key=lambda v: v.get("name", "")), 1):
            slot = f"{i:02d}"
            content = {k: v for k, v in vw.items() if k != "name"}
            vdir = self.dir / "views" / slot
            self.write_json(vdir / "view.json", content)
            ch = v4.content_hash(json.dumps(content, indent=2, sort_keys=True).encode() + b"\n")
            self.write_json(vdir / "metadata.json",
                            {"schema": 4, "legacy_name": vw.get("name", f"view-{i}"),
                             "content_hash": ch, "migrated": True, "written": NOW})
            out[vw.get("name", f"view-{i}")] = {"slot": slot, "hash": ch}
            slots.append(slot)
        self.write_json(self.dir / "viewset" / "canonical" / "views.json", {"slots": slots})
        self.move(cj, self.dir / "views" / "origin-cameras.json", kind="mv(origin)")
        return out

    # ---------------- phase 5: renders + scores ----------------

    def phase_renders(self, run_results: list[dict], views: dict) -> dict[str, str]:
        """variant -> representative identity (mesh) for score mapping."""
        variant_identity = {}
        by_variant = {r["variant"]: r for r in run_results if r}
        for r in run_results:
            if not r:
                continue
            src = by_variant.get(r.get("source_run", "").replace("pipeline-", "").replace("/run-", "--")) \
                if r["kind"] == "render-variant" else r
            target = src if src and src.get("kind") == "full" else r if r["kind"] == "full" else None
            rdir_old = r.get("rdir") or r.get("old_rdir")
            renders = sorted((rdir_old / "renders").glob("*.png")) if (rdir_old / "renders").is_dir() else []
            if not renders:
                if r["kind"] == "full":
                    variant_identity[r["variant"]] = r["rid"]
                if r["kind"] == "render-variant" and rdir_old.is_dir():
                    for f in sorted(rdir_old.rglob("*")):
                        if f.is_file() and not self.is_moved(f):
                            self.move(f, self.dir / "_migration-orphans" / r["variant"] / f.name)
                continue
            # pick mesh dir: sidecar mesh_source / variant suffix
            for png in renders:
                sidecar = png.with_suffix(".json")
                sc = json.loads(sidecar.read_text()) if sidecar.exists() else {}
                msrc = sc.get("mesh_source", "oriented")
                vname = png.stem
                vinfo = views.get(vname)
                mesh_dirs = (target or {}).get("mesh_dirs", {})
                # variant suffix overrides (tetra/tetra1m render-variants)
                if r["variant"].endswith("-tetra"):
                    mdir = mesh_dirs.get("oriented")
                elif r["variant"].endswith("tetra1m"):
                    mdir = mesh_dirs.get("conditioned") or mesh_dirs.get("oriented")
                else:
                    mdir = mesh_dirs.get("tsdf" if msrc == "tsdf" else "oriented") \
                        or mesh_dirs.get("tsdf") or mesh_dirs.get("oriented")
                if mdir is None or vinfo is None:
                    self.log(f"WARN render {r['variant']}/{vname}: no mesh dir or view -> origin-kept")
                    self.move(png, (self.dir / "_migration-orphans" / r["variant"] / png.name))
                    if sidecar.exists():
                        self.move(sidecar, (self.dir / "_migration-orphans" / r["variant"] / sidecar.name))
                    continue
                mid = json.loads((mdir / "metadata.json").read_text())["identity"] if self.apply and (mdir / "metadata.json").exists() else "PENDING"
                rsettings = {"engine": "BLENDER_WORKBENCH", "resolution": [1920, 1080]}
                rend_id = v4.identity_hash({"mesh": mid, "view_content": vinfo["hash"]},
                                           rsettings, "render-workbench@0")
                rdest = mdir / "renders" / rend_id
                self.move(png, rdest / "render.png")
                if sidecar.exists():
                    self.move(sidecar, rdest / "origin-sidecar.json")
                self.metadata(rdest, task="render", algo="render-workbench@0",
                              identity=rend_id,
                              resolved_inputs={"mesh": mid, "view_content": vinfo["hash"]},
                              settings=rsettings, mechanism="migrate",
                              origin=f"{r['variant']}/{vname}",
                              extra={"view_slot": vinfo["slot"], "legacy_view": vname})
                variant_identity[r["variant"]] = mid
            # render-variant residue (run.json etc.)
            if r["kind"] == "render-variant":
                for f in sorted(rdir_old.rglob("*")):
                    if f.is_file() and not self.is_moved(f):
                        self.move(f, self.dir / "_migration-orphans" / r["variant"] / f.name)
        return variant_identity

    def phase_scores(self, variant_identity: dict, views: dict):
        rk = self.dir / "rankings.jsonl"
        if not rk.exists():
            return
        entries = [json.loads(l) for l in rk.read_text().splitlines() if l.strip()]
        out = []
        for e in entries:
            vinfo = views.get(e.get("view", ""), {})
            for variant, rank in e.get("ranks", {}).items():
                out.append({"schema": 4, "at": variant_identity.get(variant, f"UNMAPPED:{variant}"),
                            "legacy_variant": variant, "view": vinfo.get("hash"),
                            "slot": vinfo.get("slot"), "rank": rank,
                            "rater": e.get("rater"), "ts": e.get("submitted_at"),
                            "migrated": True})
        self.log(f"scores.jsonl: {len(out)} entries from {len(entries)} ranking submissions")
        if self.apply:
            with (self.dir / "scores.jsonl").open("a") as f:
                for o in out:
                    f.write(json.dumps(o, sort_keys=True) + "\n")
        self.move(rk, self.dir / "views" / "origin-rankings.jsonl", kind="mv(origin)")

    # ---------------- driver ----------------

    def run(self):
        self.phase_pool()
        sub_by_path = self.phase_preproc()
        views = self.phase_views()
        run_results = []
        for rdir in sorted(self.dir.glob("pipeline-*/run-*")):
            run_results.append(self.migrate_run(rdir, sub_by_path, views))
        variant_identity = self.phase_renders(run_results, views)
        self.phase_scores(variant_identity, views)
        # primary ref: largest subset that has cameras
        if self.apply:
            best = None
            for sd in (self.dir / "images" / "subsets").glob("*/"):
                if sd.name == "primary" or not (sd / "subset.json").exists():
                    continue
                n = len(json.loads((sd / "subset.json").read_text())["members"])
                has_cams = (sd / "cameras").is_dir()
                key = (has_cams, n)
                if best is None or key > best[0]:
                    best = (key, sd.name)
            if best:
                sc = v4.Scene(self.scene)
                if sc.set_ref_if_unset("primary", best[1]):
                    self.log(f"ref: primary -> {best[1]} (set-if-unset)")
        # final input sweep: anything left (preproc spec/results, solver
        # byproducts, capture archives) -> _migration-orphans/input/ —
        # bytes preserved (Mac archive), legacy structure retired.
        # spine-* stays: parked epic with its own documented location.
        inp = self.dir / "input"
        if inp.is_dir():
            for f in sorted(inp.rglob("*")):
                if f.is_file() and "spine-" not in f.parts[len(self.dir.parts) + 1] \
                        and not self.is_moved(f):
                    self.move(f, self.dir / "_migration-orphans" / "input" / f.relative_to(inp))
        # cleanup empty legacy dirs
        if self.apply:
            for legacy in sorted(self.dir.glob("pipeline-*")) + [self.dir / "input"]:
                if legacy.is_dir():
                    for d in sorted((p for p in legacy.rglob("*") if p.is_dir()),
                                    key=lambda p: -len(p.parts)):
                        try:
                            d.rmdir()
                        except OSError:
                            pass
                    try:
                        legacy.rmdir()
                    except OSError:
                        self.log(f"WARN: {legacy.name} not empty after migration — residue to inspect")
            for rb in self.dir.glob("rank_board_*.png"):
                rb.unlink()
                self.log(f"rm: {rb.name} (regenerable rank board)")
        # job record (locked #8)
        if self.apply:
            jd = v4.Scene(self.scene).job_dir()
            (jd / "job.json").write_text(json.dumps({
                "schema": 4, "graph": "migrate-v4", "mechanism": "migrate",
                "bindings": {"scene": self.scene}, "story": "STO-SCN-080",
                "actions": len(self.actions), "written": NOW}, indent=2) + "\n")
        return self.actions


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    apply = "--apply" in sys.argv
    if not args:
        print(__doc__)
        return 2
    scenes = sorted(d.name for d in STORE.iterdir()
                    if d.is_dir() and not d.name.startswith((".", "_"))) \
        if args[0] == "all" else [args[0]]
    for s in scenes:
        m = Mig(s, apply)
        actions = m.run()
        print(f"=== {s}: {'APPLIED' if apply else 'DRY-RUN'} {len(actions)} actions")
        if not apply:
            for a in actions:
                print("  " + a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
