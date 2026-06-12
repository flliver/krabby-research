"""HTTP server for the render-ranking app.

Pure stdlib http.server — no FastAPI, no node_modules, no build step. The
frontend is one index.html with vanilla JS + native HTML5 drag-and-drop.

Routes:
    GET  /                            → index.html
    GET  /static/<file>               → static asset (css, js)
    GET  /api/scenes                  → list of scenes available
    GET  /api/scene/<scene>           → views + variants + manifests
    GET  /api/render/<scene>/<view>/<variant>.png  → PNG bytes
    GET  /api/rankings/<scene>        → existing rankings
    POST /api/rankings/<scene>        → append a ranking row
    GET  /api/aggregate/<scene>       → Borda-count aggregate

Data sources are read-only EXCEPT rankings.jsonl, which we append-only-write.

Run:
    python3 server.py [--port 8090]
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

# Reuse manifest_lib + aggregation logic from the parent workspace dir
HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parent
sys.path.insert(0, str(WORKSPACE))
# STO-SCN-045: the server now reads the scene store
# (scenes/<scene>/pipeline-<p>/run-<r>/) instead of the deleted legacy
# milestone layout. Variant labels are "<pipeline>--<run>" (prefixes
# stripped), matching render_comparison_matrix.sh. Legacy manifest_lib
# is no longer used — per-run metadata comes from the run's run.json.
import os  # noqa: E402

SCENES_ROOT = Path(os.environ.get("KRABBY_SCENES_ROOT", "/var/krabby/scenes"))
STATIC_DIR = HERE / "static"
SCENE_PREFIX_DEFAULT = "004-sky-house"  # scene 'family' for variant discovery


# ---------------------------------------------------------------------------
# Borda-count aggregation (kept identical to the previous viser app's math
# so existing rankings.jsonl files keep producing the same numbers).
# ---------------------------------------------------------------------------

def aggregate(rankings: list[dict]) -> dict:
    """Compute per-view + overall Borda scores. Returns a structured dict
    the frontend renders as a leaderboard.

    Tied ranks: items at the same rank value share the average of the
    point-slots they occupy (standard Borda-with-ties).
    """
    if not rankings:
        return {"per_view": {}, "overall": [], "n_submissions": 0}

    by_view: dict[str, list[dict]] = {}
    for r in rankings:
        by_view.setdefault(r["view"], []).append(r)

    per_view = {}
    overall: dict[str, list[float]] = {}

    for view in sorted(by_view.keys()):
        subs = by_view[view]
        per_variant: dict[str, list[float]] = {}
        for sub in subs:
            ranks = sub["ranks"]
            n = len(ranks)
            by_rank: dict[int, list[str]] = {}
            for v, rk in ranks.items():
                by_rank.setdefault(int(rk), []).append(v)
            sorted_ranks = sorted(by_rank.keys())
            slot = 1
            for rk in sorted_ranks:
                tied = by_rank[rk]
                slots = list(range(slot, slot + len(tied)))
                pts_each = sum(n - s + 1 for s in slots) / len(tied)
                for v in tied:
                    per_variant.setdefault(v, []).append(pts_each)
                slot += len(tied)
        avg = {v: sum(pts) / len(pts) for v, pts in per_variant.items()}
        per_view[view] = {
            "n_submissions": len(subs),
            "leaderboard": sorted(
                [{"variant": v, "score": s} for v, s in avg.items()],
                key=lambda x: -float(x["score"]),
            ),
        }
        n_variants = len(avg) if avg else 1
        for v, score in avg.items():
            overall.setdefault(v, []).append(score / n_variants)

    overall_avg = {v: sum(s) / len(s) for v, s in overall.items()}
    return {
        "n_submissions": len(rankings),
        "per_view": per_view,
        "overall": sorted(
            [{"variant": v, "score": s} for v, s in overall_avg.items()],
            key=lambda x: -float(x["score"]),
        ),
    }


# ---------------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------------

class Handler(BaseHTTPRequestHandler):
    # ---- helpers --------------------------------------------------------

    def _send_bytes(self, data: bytes, content_type: str, status: int = 200):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        # Disable caching during dev so reloads always hit fresh files
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _send_json(self, payload, status: int = 200):
        body = json.dumps(payload).encode("utf-8")
        self._send_bytes(body, "application/json", status)

    def _send_text(self, msg: str, status: int = 200):
        self._send_bytes(msg.encode("utf-8"), "text/plain", status)

    def _not_found(self, msg: str = "Not Found"):
        self._send_text(msg, 404)

    def _bad_request(self, msg: str = "Bad Request"):
        self._send_text(msg, 400)

    def log_message(self, format, *args):  # noqa: A002 (override sig)
        # Quieter than default; one line per request, no "127.0.0.1 - -"
        sys.stderr.write(f"  [{self.command} {self.path}] {format % args}\n")

    # ---- routing --------------------------------------------------------

    def do_GET(self):
        url = urlparse(self.path)
        p = url.path

        if p == "/" or p == "/index.html":
            return self._serve_static("index.html")
        if p.startswith("/static/"):
            return self._serve_static(p[len("/static/"):])
        if p == "/api/scenes":
            return self._send_json(self._list_scenes())
        if p.startswith("/api/scene/"):
            return self._send_json(self._scene_payload(p[len("/api/scene/"):]))
        if p.startswith("/api/render/"):
            return self._serve_render(p[len("/api/render/"):])
        if p.startswith("/api/materialize/"):
            return self._send_json(self._materialize_status(p[len("/api/materialize/"):]))
        if p.startswith("/api/rankings/"):
            return self._send_json(self._read_rankings(p[len("/api/rankings/"):]))
        if p.startswith("/api/aggregate/"):
            scene = p[len("/api/aggregate/"):]
            return self._send_json(aggregate(self._read_rankings(scene)))
        return self._not_found()

    def do_POST(self):
        url = urlparse(self.path)
        p = url.path
        if p.startswith("/api/rankings/"):
            return self._handle_post_ranking(p[len("/api/rankings/"):])
        if p.startswith("/api/materialize/"):
            return self._handle_materialize(p[len("/api/materialize/"):])
        return self._not_found()

    # ---- materialize (STO-SCN-086: missing tiles trigger render jobs) ----

    @staticmethod
    def _v4job_running() -> bool:
        import subprocess
        r = subprocess.run(["pgrep", "-f", "v4job.py render-missing"],
                           capture_output=True, text=True)
        return bool(r.stdout.strip())

    def _materialize_status(self, scene: str) -> dict:
        scene_dir = SCENES_ROOT / scene
        running = self._v4job_running()
        last = None
        jobs = sorted(scene_dir.glob("jobs/*/job.json"), reverse=True)
        for j in jobs:
            try:
                d = json.loads(j.read_text())
            except ValueError:
                continue
            if d.get("graph") == "render-missing":
                last = {"job": j.parent.name, "outcome": d.get("outcome")}
                break
        return {"running": running, "last": last}

    def _handle_materialize(self, scene: str):
        scene_dir = SCENES_ROOT / scene
        if not self._is_v4(scene_dir):
            return self._bad_request("materialize is v4-only")
        # Concurrency guard: ONE materialize at a time, store-wide — local
        # Blender renders are serial work and NOOP semantics make a queued
        # re-click pointless. Second click gets the running status back.
        if self._v4job_running():
            return self._send_json({"ok": True, "already_running": True,
                                    **self._materialize_status(scene)})
        import subprocess
        import sys as _sys
        repo = Path(__file__).resolve().parent.parent
        subprocess.Popen([_sys.executable, str(repo / "v4job.py"),
                          "render-missing", scene],
                         stdout=open("/tmp/v4job-materialize.log", "ab"),
                         stderr=subprocess.STDOUT,
                         start_new_session=True)
        return self._send_json({"ok": True, "started": True, "scene": scene})

    # ---- static ---------------------------------------------------------

    def _serve_static(self, rel: str):
        # Security: clamp path inside static dir
        target = (STATIC_DIR / rel).resolve()
        if not str(target).startswith(str(STATIC_DIR)):
            return self._bad_request("Invalid path")
        if not target.is_file():
            return self._not_found(f"static file: {rel}")
        # Pick a content-type by extension
        ct_map = {
            ".html": "text/html; charset=utf-8",
            ".css": "text/css; charset=utf-8",
            ".js": "application/javascript; charset=utf-8",
            ".json": "application/json",
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".svg": "image/svg+xml",
            ".ico": "image/x-icon",
        }
        ct = ct_map.get(target.suffix.lower(), "application/octet-stream")
        self._send_bytes(target.read_bytes(), ct)

    # ---- scene + variants ----------------------------------------------

    def _list_scenes(self) -> list[str]:
        """List scene roots: v4 = has viewset/canonical (HUG-SCN-005);
        v2 legacy = unified cameras.json."""
        out = []
        if not SCENES_ROOT.is_dir():
            return out
        for d in sorted(SCENES_ROOT.iterdir()):
            if not d.is_dir():
                continue
            if (d / "viewset" / "canonical" / "views.json").exists() \
                    or (d / "cameras.json").exists():
                out.append(d.name)
        return out

    # ---- store-shape v4 (HUG-SCN-005, STO-SCN-080) -----------------------
    # URL contract preserved: /api/render/<scene>/<view>/<variant>.png —
    # in v4, <view> = slot ("01") and <variant> = mesh identity.

    @staticmethod
    def _is_v4(scene_dir: Path) -> bool:
        return (scene_dir / "viewset" / "canonical" / "views.json").exists()

    @staticmethod
    def _ply_stats(ply: Path) -> dict:
        """Header-only PLY stats (STO-SCN-084): verts/faces + size."""
        counts = {}
        try:
            with ply.open("rb") as f:
                for raw in f:
                    line = raw.decode("ascii", "ignore").strip()
                    if line.startswith("element"):
                        _, name, n = line.split()
                        counts[name] = int(n)
                    if line == "end_header" or f.tell() > 65536:
                        break
            return {"verts": counts.get("vertex", 0), "faces": counts.get("face", 0),
                    "size_mb": round(ply.stat().st_size / 2**20, 1)}
        except OSError:
            return {}

    @staticmethod
    def _v4_render_index(scene_dir: Path) -> dict:
        """(slot, mesh_identity) -> render.png path; plus labels + slots."""
        import sys as _sys
        _sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        import v4core as v4
        sc = v4.scan_scene(scene_dir.name)
        hash_to_slot = {h: slot for slot, h in sc["views"].items()}
        idx, labels = {}, {}
        def add_renders(mdir: Path, identity: str, label: str):
            labels[identity] = label
            for rmd in mdir.glob("renders/*/metadata.json"):
                md = json.loads(rmd.read_text())
                slot = md.get("view_slot") or hash_to_slot.get(
                    md.get("resolved_inputs", {}).get("view_content"), None)
                if slot and (rmd.parent / "render.png").exists():
                    idx[(slot, identity)] = rmd.parent / "render.png"
        for rep in sc["representations"]:
            base = rep.get("legacy_variant") or rep["identity"]
            rdir = scene_dir / "represent" / rep["kind"] / rep["identity"]
            for m in rep["meshes"]:
                mdir = rdir / "meshify" / m["method"] / m["identity"]
                warn = "" if m.get("rankable", True) else " ⚠ mis-aligned"
                add_renders(mdir, m["identity"], f"{base} [{m['method']}]{warn}")
                for c in m["conditioned"]:
                    add_renders(mdir / "condition" / c["identity"], c["identity"],
                                f"{base} [{m['method']}+conditioned]")
        slots = sorted(json.loads((scene_dir / "viewset" / "canonical" / "views.json")
                                  .read_text())["slots"])
        return {"index": idx, "labels": labels, "slots": slots,
                "views": sc["views"], "scan": sc}

    @staticmethod
    def _variant_label(pipeline_dir: Path, run_dir: Path) -> str:
        """pipeline-matcha / run-12-strong → 'matcha--12-strong'."""
        p = pipeline_dir.name.removeprefix("pipeline-")
        r = run_dir.name.removeprefix("run-")
        return f"{p}--{r}"

    def _scene_payload(self, scene: str) -> dict:
        """Return everything the frontend needs to render a scene's UI."""
        scene_dir = SCENES_ROOT / scene
        if not scene_dir.is_dir():
            return {"error": f"scene not found: {scene}"}
        if self._is_v4(scene_dir):
            ix = self._v4_render_index(scene_dir)
            rendered: dict[str, list[str]] = {}
            for (slot, identity) in sorted(ix["index"]):
                rendered.setdefault(slot, []).append(identity)
            variants = sorted({i for vs in rendered.values() for i in vs})
            manifests = {}
            for rep in ix["scan"]["representations"]:
                rdir = scene_dir / "represent" / rep["kind"] / rep["identity"]
                mesh_paths = {}
                for mm in rep["meshes"]:
                    mdir = rdir / "meshify" / mm["method"] / mm["identity"]
                    mesh_paths[mm["identity"]] = mdir / "mesh.ply"
                    for c in mm["conditioned"]:
                        mesh_paths[c["identity"]] = mdir / "condition" / c["identity"] / "mesh.ply"
                for m in rep["meshes"] + [c for mm in rep["meshes"] for c in mm["conditioned"]]:
                    if m["identity"] in variants:
                        mp = mesh_paths.get(m["identity"])
                        notes = []
                        if not m.get("rankable", True):
                            sa = m.get("self_alignment") or {}
                            notes.append("MIS-ALIGNED: " + (m.get("quality_flag") or "flagged")
                                         + (f" (ICP fitness {sa['icp_fitness']})"
                                            if sa.get("icp_fitness") is not None else ""))
                        if not rep["deliverable_eligible"]:
                            notes.append("NOT DELIVERABLE: " + "; ".join(rep["license_flags"]))
                        manifests[m["identity"]] = {
                            "variant_name": ix["labels"].get(m["identity"], m["identity"]),
                            "pipeline": rep["kind"],
                            "run": m["identity"],
                            "notes": "; ".join(notes),
                            "mesh": self._ply_stats(mp) if mp and mp.exists() else {},
                            "transforms": {rep["algo"] or rep["kind"]: {
                                "parameters": {**rep["settings"], **m.get("settings", {})}}}}
            # STO-SCN-085: expected = every mesh artifact × every canonical
            # slot (we KNOW what renders should exist); missing = expected
            # minus the render index. Surfaced so the UI can show gaps.
            expected_ids = sorted(ix["labels"])
            missing = {slot: [i for i in expected_ids if (slot, i) not in ix["index"]]
                       for slot in ix["slots"]}
            # STO-SCN-087: task-tier gaps from the GRAPHS (planner view),
            # not just absent renders on existing artifacts
            import v4core as _v4
            task_gaps = _v4.expected_task_gaps(scene)
            return {"scene": scene, "views": ix["slots"], "rendered": rendered,
                    "variants": variants, "manifests": manifests,
                    "labels": ix["labels"], "missing": missing,
                    "task_gaps": task_gaps, "store": "v4"}
        # 1) views — unified cameras.json (schema 5) only
        views: list = []
        views_path = scene_dir / "cameras.json"
        if views_path.exists():
            with open(views_path) as f:
                cv = json.load(f)
            views = [v["name"] for v in cv.get("views", [])]
        # 2) which (view, variant) PNGs exist. In the pipelines/runs store
        #    a scene has many runs that are NOT comparison candidates
        #    (legacy imports, repro verification runs, runner pilots) — a
        #    run is rankable iff render_comparison_matrix.sh produced a
        #    PNG for it. Renders therefore DEFINE the variant set; the
        #    full run inventory is not the ranking pool.
        # STO-SCN-058: renders live IN the run that produced them
        # (pipeline-<p>/run-<r>/renders/<view>.png + settings sidecar).
        # The per-view aggregation happens here, at read time.
        rendered: dict[str, list[str]] = {}
        for pipeline_dir in sorted(scene_dir.glob("pipeline-*")):
            for run_dir in sorted(pipeline_dir.glob("run-*")):
                rdir = run_dir / "renders"
                if not rdir.is_dir():
                    continue
                label = self._variant_label(pipeline_dir, run_dir)
                for png in rdir.glob("*.png"):
                    rendered.setdefault(png.stem, []).append(label)
        rendered = {view: sorted(vs) for view, vs in sorted(rendered.items())}
        variants = sorted({v for vs in rendered.values() for v in vs})
        # 3) manifests — SETTINGS-FIRST. What the runoff compares is the
        #    settings of each transformation (operator, 2026-06-09): each
        #    run is one parameterization. Per rendered run, assemble every
        #    transform's specification.json `parameters` (the settings
        #    being ranked) + the measured stats from its results.json.
        manifests = {}
        for v in variants:
            p, sep, r = v.partition("--")
            if not sep:
                manifests[v] = {"variant_name": v,
                                "notes": "(unrecognized variant label)"}
                continue
            run_dir = scene_dir / f"pipeline-{p}" / f"run-{r}"
            entry: dict = {"variant_name": v, "pipeline": p, "run": r,
                           "transforms": {}}
            run_json = run_dir / "run.json"
            if run_json.exists():
                try:
                    with open(run_json) as f:
                        entry["notes"] = json.load(f).get("notes", "")
                except (OSError, ValueError):
                    pass
            for tdir in sorted(run_dir.glob("transform-*")):
                if not tdir.is_dir():
                    continue
                tinfo: dict = {}
                spec_p = tdir / "specification.json"
                if spec_p.exists():
                    try:
                        with open(spec_p) as f:
                            spec = json.load(f)
                        tinfo["kind"] = spec.get("kind")
                        tinfo["description"] = spec.get("description")
                        tinfo["parameters"] = spec.get("parameters", {})
                    except (OSError, ValueError):
                        tinfo["parameters"] = {"error": "spec unreadable"}
                res_p = tdir / "results.json"
                if res_p.exists():
                    try:
                        with open(res_p) as f:
                            res = json.load(f)
                        tinfo["measured"] = {
                            k: res.get(k) for k in
                            ("status", "provenance", "duration_s",
                             "host", "peak_vram_mib")
                            if res.get(k) is not None
                        }
                    except (OSError, ValueError):
                        pass
                entry["transforms"][tdir.name] = tinfo
            manifests[v] = entry
        # 5) raters who've submitted on this scene (alphabetical, unique)
        raters = sorted({
            r.get("rater", "").strip()
            for r in self._read_rankings(scene)
            if r.get("rater")
        })
        return {
            "scene": scene,
            "views": views,
            "variants": variants,
            "manifests": manifests,
            "rendered": rendered,
            "raters": raters,
        }

    def _serve_render(self, rel: str):
        """rel is 'scene/view/variant.png' (URL contract unchanged);
        resolves into the producing run's renders/ dir (STO-SCN-058)."""
        # Path-shape sanity check + security clamp
        try:
            scene, view, fname = rel.split("/")
        except ValueError:
            return self._bad_request("expected scene/view/variant.png")
        variant = fname.removesuffix(".png")
        scene_dir = SCENES_ROOT / scene
        if self._is_v4(scene_dir):
            ix = self._v4_render_index(scene_dir)
            target = ix["index"].get((view, variant))
            if target is None:
                return self._not_found(f"render: {rel}")
            return self._send_bytes(target.read_bytes(), "image/png")
        pipeline, sep, run = variant.partition("--")
        if not sep:
            return self._bad_request(f"unrecognized variant label: {variant}")
        target = (
            SCENES_ROOT / scene / f"pipeline-{pipeline}" / f"run-{run}"
            / "renders" / f"{view}.png"
        ).resolve()
        if not str(target).startswith(str(SCENES_ROOT.resolve())):
            return self._bad_request("Invalid path")
        if not target.is_file():
            return self._not_found(f"render: {rel}")
        self._send_bytes(target.read_bytes(), "image/png")

    # ---- rankings -------------------------------------------------------

    def _rankings_path(self, scene: str) -> Path:
        return SCENES_ROOT / scene / "rankings.jsonl"

    def _read_rankings(self, scene: str) -> list[dict]:
        # Scene-root rankings.jsonl only. Legacy rows under _unsorted/ used
        # the old variant labels (e.g. "12-strong") which don't exist in the
        # pipelines/runs world — they are provenance, not live data.
        scene_dir = SCENES_ROOT / scene
        if self._is_v4(scene_dir):
            # synthesize ranking rows from scores.jsonl: one row per
            # (ts, rater, slot) submission group
            sj = scene_dir / "scores.jsonl"
            if not sj.exists():
                return []
            groups: dict = {}
            for line in sj.read_text().splitlines():
                if not line.strip():
                    continue
                s = json.loads(line)
                key = (s.get("ts"), s.get("rater"), s.get("slot"))
                g = groups.setdefault(key, {"schema_version": 1, "scene": scene,
                                            "view": s.get("slot"), "rater": s.get("rater"),
                                            "submitted_at": s.get("ts"), "ranks": {}})
                g["ranks"][s["at"]] = s["rank"]
            # retired identities (scores history outlives artifacts) are
            # dropped from LIVE results (operator-reported: outdated items)
            import sys as _sys
            _sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
            import v4core as _v4
            sc = _v4.scan_scene(scene)
            live = {m["identity"] for rep in sc["representations"] for m in rep["meshes"]}
            live |= {c["identity"] for rep in sc["representations"]
                     for m in rep["meshes"] for c in m["conditioned"]}
            out = []
            for k in sorted(groups):
                g = groups[k]
                g["ranks"] = {i: r for i, r in g["ranks"].items() if i in live}
                if g["ranks"]:
                    out.append(g)
            return out
        p = self._rankings_path(scene)
        if not p.exists():
            return []
        out = []
        with open(p) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue  # skip corrupt lines
        return out

    def _handle_post_ranking(self, scene: str):
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0 or length > 1_000_000:
            return self._bad_request("missing/oversized body")
        body = self.rfile.read(length)
        try:
            payload = json.loads(body)
        except json.JSONDecodeError as e:
            return self._bad_request(f"bad JSON: {e}")
        # Required fields
        for k in ("rater", "view", "ranks"):
            if k not in payload:
                return self._bad_request(f"missing field: {k}")
        if not isinstance(payload["ranks"], dict) or not payload["ranks"]:
            return self._bad_request("ranks must be non-empty {variant: int}")
        # Stamp + persist
        row = {
            "schema_version": 1,
            "scene": scene,
            "view": str(payload["view"]),
            "rater": str(payload["rater"]).strip(),
            "ranks": {str(k): int(v) for k, v in payload["ranks"].items()},
            "submitted_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        }
        if not row["rater"]:
            return self._bad_request("rater name required")
        scene_dir = SCENES_ROOT / scene
        if self._is_v4(scene_dir):
            # v4: scores attached to identities (HUG-SCN-005 locked #7c).
            # view = slot; ranks keyed by mesh identity. Append per-identity
            # rows to scenes/<scene>/scores.jsonl.
            slot = row["view"]
            vh = None
            vj = scene_dir / "views" / slot / "view.json"
            if vj.exists():
                import sys as _sys
                _sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
                import v4core as v4
                vh = v4.content_hash(vj.read_bytes())
            rows = [{"schema": 4, "at": ident, "view": vh, "slot": slot,
                     "rank": rank, "rater": row["rater"],
                     "ts": row["submitted_at"]}
                    for ident, rank in row["ranks"].items()]
            with open(scene_dir / "scores.jsonl", "a") as f:
                for r in rows:
                    f.write(json.dumps(r, sort_keys=True) + "\n")
            # `row` kept for the frontend contract (STO-SCN-083: the v4
            # branch returned only rows[] and the submit handler crashed
            # reading d.row.submitted_at)
            return self._send_json({"ok": True, "row": row, "rows": rows,
                                    "store": "v4"})
        # v2 legacy: append-only rankings.jsonl
        path = self._rankings_path(scene)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a") as f:
            f.write(json.dumps(row, sort_keys=True) + "\n")
        self._send_json({"ok": True, "row": row})


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=8090,
                   help="port to bind on (default 8090)")
    p.add_argument("--host", default="0.0.0.0",
                   help="bind address (default 0.0.0.0 = LAN-accessible; '127.0.0.1' for localhost-only)")
    args = p.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"rate-renders → http://{args.host}:{args.port}")
    print(f"  scenes:    {SCENES_ROOT}")
    print(f"  static:    {STATIC_DIR}")
    print(f"  Ctrl+C to stop")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping.")
        server.shutdown()


if __name__ == "__main__":
    main()
