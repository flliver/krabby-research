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
import re
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
# Render description (STO-SCN-106) — ultra-succinct narrative of how a render
# was built, DERIVED from its manifest (v4 algo+settings, or the legacy
# transform chain). Read-only synthesis: every visible render gets one (the
# manifest is rebuilt each request), so "backfill" is automatic.
# ---------------------------------------------------------------------------

_SALIENT = ("sfm", "dense_regul", "selector", "n_images", "n_scout", "n", "grid",
            "res", "process_res", "conf_percentile", "voxel_frac", "alignment_config",
            # STO-SCN-138: cull / condition knobs (cull-mesh@0)
            "max_dist_from_cluster", "min_views", "cambox_expand", "floor_z_min")


def _human_count(n: int) -> str:
    n = int(n)
    if n >= 1_000_000:
        return f"{n / 1e6:.1f}M"
    if n >= 1_000:
        return f"{n / 1e3:.0f}k"
    return str(n)


def _setting_tok(k: str, v) -> str | None:
    """One telegraphic token for a salient setting (skip defaults/empties)."""
    if v in (None, "", "default"):
        return None
    if k == "sfm":
        return str(v)                       # posed / unposed
    if k == "dense_regul":
        return f"dense-{v}"
    if k == "selector":
        return f"{v}-select"
    if k in ("n_images", "n_scout", "n"):
        return f"{v}v"
    if k in ("res", "process_res"):
        return f"{v}px"
    if k == "conf_percentile":
        return f"conf{v}"
    if k == "grid":
        return f"grid{v}"
    if k == "alignment_config":
        return f"align-{v}"
    # STO-SCN-138: cull knobs — omit the disabled sentinels so the line stays clean
    if k == "max_dist_from_cluster":
        try:
            return f"≤{float(v):g}m" if float(v) > 0 else None
        except (TypeError, ValueError):
            return None
    if k == "min_views":
        try:
            return f"≥{int(v)}views" if int(v) > 0 else None
        except (TypeError, ValueError):
            return None
    if k == "cambox_expand":
        try:
            return f"cambox+{float(v):g}" if float(v) >= 0 else None
        except (TypeError, ValueError):
            return None
    if k == "floor_z_min":
        return f"floor≥{v}"
    return f"{k}={v}"


def describe_render(m: dict) -> str:
    """Ultra-succinct narrative of how a render was built, from its manifest
    (v4 `{algo: {parameters}}` or the legacy `{transform-NN: {kind, parameters}}`
    chain). Telegraphic, dot-joined; degrades to the variant name, never raises."""
    try:
        tf = m.get("transforms") or {}
        toks: list[str] = []
        legacy = any(str(k).startswith("transform-") for k in tf)
        if legacy:
            if m.get("pipeline"):
                toks.append(str(m["pipeline"]))
            for _name, t in sorted(tf.items()):
                t = t or {}
                if t.get("kind"):
                    toks.append(str(t["kind"]))
                p = t.get("parameters") or {}
                for k in _SALIENT:
                    tok = _setting_tok(k, p.get(k))
                    if tok:
                        toks.append(tok)
        else:
            algo = next(iter(tf), None) or m.get("pipeline")
            if algo:
                toks.append(str(algo))
            p = (tf.get(algo) or {}).get("parameters", {}) if algo else {}
            for k in _SALIENT:
                tok = _setting_tok(k, p.get(k))
                if tok:
                    toks.append(tok)
            mm = re.search(r"\[([^\]]+)\]", m.get("variant_name", ""))   # mesh method
            if mm:
                toks.append(mm.group(1))
        mesh = m.get("mesh") or {}
        if mesh.get("faces"):
            toks.append(f"{_human_count(mesh['faces'])} tris")
        elif mesh.get("verts"):
            toks.append(f"{_human_count(mesh['verts'])} verts")
        notes = m.get("notes") or ""
        if "MIS-ALIGNED" in notes:
            toks.append("⚠ mis-aligned")
        if "NOT DELIVERABLE" in notes:
            toks.append("⚠ non-deliverable")
        out = " · ".join(dict.fromkeys(t for t in toks if t))   # dedup, keep order
        return out or m.get("variant_name", "(unknown)")
    except Exception:
        return m.get("variant_name", "(unknown)")


def _find_node_meta(scene_dir, mesh_id):
    """Locate a mesh node (meshify OR condition) by identity; return its metadata dict."""
    for pat in (f"represent/*/*/meshify/*/{mesh_id}/metadata.json",
                f"represent/*/*/meshify/*/*/condition/{mesh_id}/metadata.json"):
        for p in scene_dir.glob(pat):
            try:
                return json.loads(p.read_text())
            except (OSError, ValueError):
                pass
    return None


def chain_transforms(scene_dir, node_md, rep_algo, rep_settings):
    """STO-SCN-138 / operator 2026-06-16: RETAIN HISTORICAL SETTINGS in the manifest. Walk the
    condition node's `resolved_inputs.mesh` lineage so every upstream conditioning step's settings
    are shown — e.g. tetra-filter → cambox cull → poisson — newest first (so describe_render names
    the node's own op), then the base meshify + representation. Each step keyed by its algo,
    de-duplicated if an algo recurs in the chain."""
    steps = []            # (algo, settings) newest -> oldest
    md, seen = node_md, set()
    while md and md.get("resolved_inputs", {}).get("mesh"):
        steps.append((md.get("algo") or "condition", md.get("settings", {})))
        up = md["resolved_inputs"]["mesh"]
        if up in seen:
            break
        seen.add(up)
        md = _find_node_meta(scene_dir, up)
    if md and md.get("algo"):                       # the base meshify node
        steps.append((md["algo"], md.get("settings", {})))
    steps.append((rep_algo, dict(rep_settings)))    # the representation
    transforms = {}
    for algo, st in steps:
        key, i = algo, 2
        while key in transforms:
            key = f"{algo} #{i}"; i += 1
        transforms[key] = {"parameters": st}
    return transforms


def scene_meta(scene_dir: Path) -> dict:
    """Read-only metadata for a scene (STO-SCN-153).

    Pure function of a scene directory — no server/network state, so it is
    unit-testable against a synthetic tree. Returns identity + capture mode +
    counts + scale/datum status + pipeline state for the Metadata view.
    """
    name = scene_dir.name
    code, _, rest = name.partition("-")
    images_dir = scene_dir / "images"

    # canonical content-addressed images: images/<HASH>/ (exclude subsets/ingress)
    canonical = 0
    if images_dir.is_dir():
        canonical = sum(
            1 for d in images_dir.iterdir()
            if d.is_dir() and d.name not in ("subsets", "ingress"))

    subsets_dir = images_dir / "subsets"
    subsets = [d for d in subsets_dir.iterdir() if d.is_dir()] if subsets_dir.is_dir() else []
    # solves = camera solutions across all subsets (images/subsets/*/cameras/*/)
    solves = []
    for s in subsets:
        cdir = s / "cameras"
        if cdir.is_dir():
            solves += [c for c in cdir.iterdir() if c.is_dir()]

    # capture mode — a captured video vs. ingested images
    has_video = (scene_dir / "videos" / "capture").is_dir() and any(
        (scene_dir / "videos" / "capture").glob("*"))
    capture_mode = "video" if has_video else ("images" if canonical else "empty")

    # render views — unified cameras.json (schema 5) names, else the views/ dir
    render_views = 0
    cj = scene_dir / "cameras.json"
    if cj.exists():
        try:
            render_views = len(json.loads(cj.read_text()).get("views", []))
        except (OSError, ValueError):
            render_views = 0
    if not render_views and (scene_dir / "views").is_dir():
        render_views = sum(1 for d in (scene_dir / "views").iterdir() if d.is_dir())

    # scale / datum — additive datum.json sidecar next to a solve gauge
    datum = {"calibrated": False}
    datum_files = sorted(scene_dir.glob("images/subsets/*/cameras/*/datum.json"))
    if datum_files:
        try:
            dj = json.loads(datum_files[0].read_text())
            prov = dj.get("provenance", {}) if isinstance(dj.get("provenance"), dict) else {}
            datum = {
                "calibrated": True,
                "scale_m_per_unit": dj.get("scale_m_per_unit"),
                "method": prov.get("method"),
                "status": prov.get("status"),
                "scene_extent_m": prov.get("scene_extent_m"),
                "path": str(datum_files[0].relative_to(scene_dir)),
            }
        except (OSError, ValueError):
            datum = {"calibrated": False, "error": "datum.json unreadable"}

    # pipeline state — coarse stage flags
    meshed = any(scene_dir.glob("**/mesh.ply")) or any(scene_dir.glob("**/*.ply"))
    scouted = any(scene_dir.glob("images/subsets/*/cameras/*/gs_ply/**/*.ply")) \
        or any(scene_dir.glob("images/subsets/*/cameras/*/scout/**"))
    state = {
        "ingested": bool(canonical or has_video),
        "solved": bool(solves),
        "scouted": bool(scouted),
        "meshed": bool(meshed),
        "calibrated": datum["calibrated"],
    }

    return {
        "scene": name,
        "code": code,
        "name": rest or name,
        "capture_mode": capture_mode,
        "counts": {
            "images": canonical,
            "subsets": len(subsets),
            "solves": len(solves),
            "render_views": render_views,
        },
        "datum": datum,
        "state": state,
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
        if p.startswith("/api/scene/") and p.endswith("/meta"):   # STO-SCN-153
            scene = p[len("/api/scene/"):-len("/meta")]
            scene_dir = SCENES_ROOT / scene
            if not scene_dir.is_dir():
                return self._send_json({"error": f"scene not found: {scene}"})
            return self._send_json(scene_meta(scene_dir))
        if p.startswith("/api/scene/"):
            return self._send_json(self._scene_payload(p[len("/api/scene/"):]))
        if p.startswith("/api/render/"):
            return self._serve_render(p[len("/api/render/"):])
        if p.startswith("/api/materialize/"):
            return self._send_json(self._materialize_status(p[len("/api/materialize/"):]))
        if p.startswith("/api/jobs/"):
            return self._send_json(self._jobs_status(p[len("/api/jobs/"):]))
        if p.startswith("/api/rankings/"):
            return self._send_json(self._read_rankings(p[len("/api/rankings/"):]))
        if p.startswith("/api/aggregate/"):
            scene = p[len("/api/aggregate/"):]
            return self._send_json(aggregate(self._read_rankings(scene)))
        if p == "/api/profiles":                       # STO-SCN-108
            return self._send_json({"profiles": self._read_profiles()})
        return self._not_found()

    def do_POST(self):
        url = urlparse(self.path)
        p = url.path
        if p.startswith("/api/rankings/"):
            return self._handle_post_ranking(p[len("/api/rankings/"):])
        if p.startswith("/api/materialize/"):
            return self._handle_materialize(p[len("/api/materialize/"):])
        if p == "/api/profiles":                       # STO-SCN-108
            try:
                length = int(self.headers.get("Content-Length", 0))
                name = json.loads(self.rfile.read(length) or b"{}").get("name", "")
            except (ValueError, OSError):
                return self._bad_request("invalid profile body")
            return self._send_json({"profiles": self._add_profile(name)})
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

    # ---- job feedback channel (STO-SCN-088) -----------------------------
    # Source of truth = the per-invocation job.json records (locked #8).
    # The retained-MQTT state (krabby/jobs/<scene>/<job_id>) is an optional
    # FAST PATH overlay: when a broker is configured (KRABBY_MQTT_HOST) we
    # read the retained heartbeat for live per-node progress, but a missing
    # broker/client degrades silently to file truth — the tile never lies.

    @staticmethod
    def _jobs_files(scene_dir: Path) -> list:
        out = []
        for j in sorted(scene_dir.glob("jobs/*/job.json"), reverse=True):
            try:
                d = json.loads(j.read_text())
            except (ValueError, OSError):
                continue
            d["job"] = j.parent.name
            out.append(d)
        return out

    @staticmethod
    def _jobs_live(scene: str) -> dict:
        """Best-effort retained-MQTT overlay keyed by job_id. Returns {} on
        any failure (no broker, no client, timeout) — file truth stands."""
        host = os.environ.get("KRABBY_MQTT_HOST")
        if not host:
            return {}
        import shutil
        import subprocess
        if not shutil.which("mosquitto_sub"):
            return {}
        try:
            r = subprocess.run(
                ["mosquitto_sub", "-h", host,
                 "-p", os.environ.get("KRABBY_MQTT_PORT", "1883"),
                 "-t", f"krabby/jobs/{scene}/#", "-v",
                 "-W", "1", "--retained-only"],
                capture_output=True, text=True, timeout=3)
        except (subprocess.SubprocessError, OSError):
            return {}
        live = {}
        for line in r.stdout.splitlines():
            topic, _, payload = line.partition(" ")
            job_id = topic.rsplit("/", 1)[-1]
            try:
                live[job_id] = json.loads(payload)
            except ValueError:
                continue
        return live

    def _jobs_status(self, scene: str) -> dict:
        scene_dir = SCENES_ROOT / scene
        records = self._jobs_files(scene_dir)
        live = self._jobs_live(scene)
        for rec in records:
            if rec["job"] in live:
                rec["live"] = live[rec["job"]]
        return {"scene": scene, "running": self._v4job_running(),
                "jobs": records, "live_source": "mqtt" if live else "file"}

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

    def _list_scenes(self) -> list[dict]:
        """List scene roots with a representative thumbnail.
        Representative image = a render of the #1-ranked variant (current
        aggregate); fallback = first render in the index; null when the
        scene has no renders. v4 = has viewset/canonical; legacy = cameras.json."""
        out = []
        if not SCENES_ROOT.is_dir():
            return out
        for d in sorted(SCENES_ROOT.iterdir()):
            if not d.is_dir():
                continue
            if not ((d / "viewset" / "canonical" / "views.json").exists()
                    or (d / "cameras.json").exists()):
                continue
            thumb = None
            if self._is_v4(d):
                try:
                    ix = self._v4_render_index(d)
                    index = ix["index"]   # (slot, identity) -> path
                    if index:
                        top = None
                        try:
                            agg = aggregate(self._read_rankings(d.name))
                            for row in agg.get("overall", []):
                                if any(i == row["variant"] for (_s, i) in index):
                                    top = row["variant"]
                                    break
                        except Exception:
                            pass
                        key = None
                        if top is not None:
                            key = next((k for k in sorted(index) if k[1] == top), None)
                        if key is None:
                            key = sorted(index)[0]
                        thumb = f"/api/render/{d.name}/{key[0]}/{key[1]}.png"
                except Exception:
                    thumb = None
            out.append({"name": d.name, "thumb": thumb})
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

    @staticmethod
    def _camera_subset(scene_dir: Path, rep_md: dict) -> dict:
        """STO-SCN-134: the cameras/frames a reconstruction was actually built from.
        da3-scout reps → the scout's selected views; otherwise the subset's members."""
        ri = (rep_md or {}).get("resolved_inputs", {})
        sub, sid, scout = ri.get("subset"), ri.get("cameras"), ri.get("scout")
        if scout and sub and sid:
            sv = (scene_dir / "images" / "subsets" / sub / "cameras" / sid /
                  "scout" / scout / "scout_views.json")
            if sv.exists():
                try:
                    sd = json.loads(sv.read_text())
                    names = sd.get("views") or sd.get("selected") or sd.get("names") or []
                    return {"n": len(names), "frames": sorted(names), "source": f"scout {scout[:8]}"}
                except (OSError, json.JSONDecodeError):
                    pass
        if sub:
            sj = scene_dir / "images" / "subsets" / sub / "subset.json"
            if sj.exists():
                try:
                    members = json.loads(sj.read_text()).get("members", [])
                except (OSError, json.JSONDecodeError):
                    members = []
                frames = []
                for h in members:
                    mp = scene_dir / "images" / h / "metadata.json"
                    try:
                        frames.append(json.loads(mp.read_text()).get("original_name", h)
                                      if mp.exists() else h)
                    except (OSError, json.JSONDecodeError):
                        frames.append(h)
                return {"n": len(members), "frames": sorted(frames), "source": f"subset {sub[:8]}"}
        return {}

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
                rep_md = (json.loads((rdir / "metadata.json").read_text())
                          if (rdir / "metadata.json").exists() else {})
                cam_subset = self._camera_subset(scene_dir, rep_md)   # STO-SCN-134
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
                        # STO-SCN-138: a condition node (cull) carries its OWN algo+settings —
                        # chain the cull transform FIRST (drives describe_render) + the upstream
                        # represent transform, so the cull knobs are labelled + surfaced. Base
                        # meshes keep the represent algo (unchanged — no regression).
                        rep_algo = rep["algo"] or rep["kind"]
                        cull_algo = m.get("algo")          # present only for condition nodes
                        node_md_path = (mp.parent / "metadata.json") if mp else None
                        if cull_algo and node_md_path and node_md_path.exists():
                            # RETAIN HISTORICAL SETTINGS: walk the full conditioning lineage
                            # (tetra-filter → cull → poisson → …) so the manifest shows every step.
                            try:
                                node_md = json.loads(node_md_path.read_text())
                                transforms = chain_transforms(
                                    scene_dir, node_md, rep_algo, rep["settings"])
                            except (OSError, ValueError):
                                transforms = {cull_algo: {"parameters": m.get("settings", {})},
                                              rep_algo: {"parameters": dict(rep["settings"])}}
                        elif cull_algo:
                            transforms = {cull_algo: {"parameters": m.get("settings", {})},
                                          rep_algo: {"parameters": dict(rep["settings"])}}
                        else:
                            transforms = {rep_algo: {
                                "parameters": {**rep["settings"], **m.get("settings", {})}}}
                        manifests[m["identity"]] = {
                            "variant_name": ix["labels"].get(m["identity"], m["identity"]),
                            "pipeline": rep["kind"],
                            "run": m["identity"],
                            "camera_subset": cam_subset,   # STO-SCN-134
                            "notes": "; ".join(notes),
                            "mesh": self._ply_stats(mp) if mp and mp.exists() else {},
                            "transforms": transforms}
                        manifests[m["identity"]]["description"] = describe_render(
                            manifests[m["identity"]])
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
                manifests[v] = {"variant_name": v, "description": v,
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
            entry["description"] = describe_render(entry)
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

    # ---- profiles (STO-SCN-108) -----------------------------------------
    # Rudimentary, passwordless, store-level rater identities so the ranker is
    # a pick-from-a-list (origin-independent) instead of free text the user must
    # remember + retype per browser origin.

    @staticmethod
    def _profiles_path() -> Path:
        return SCENES_ROOT / "profiles.json"

    @staticmethod
    def _scores_raters() -> set:
        """Every rater that has submitted on any scene (from scores.jsonl)."""
        out = set()
        for sj in SCENES_ROOT.glob("*/scores.jsonl"):
            try:
                for line in sj.read_text().splitlines():
                    if line.strip():
                        r = (json.loads(line).get("rater") or "").strip()
                        if r and r != "__diag__":
                            out.add(r)
            except (OSError, ValueError):
                continue
        return out

    def _read_profiles(self) -> list:
        """Server-side profile list = explicitly-added profiles ∪ raters seen in
        submissions, deduped + case-insensitively sorted. Origin-independent (the
        store, not the browser)."""
        explicit = []
        p = self._profiles_path()
        if p.exists():
            try:
                explicit = json.loads(p.read_text()) or []
            except (OSError, ValueError):
                explicit = []
        names = {str(n).strip() for n in explicit if str(n).strip()}
        names |= self._scores_raters()
        return sorted(names, key=str.lower)

    def _add_profile(self, name: str) -> list:
        """Append a profile (passwordless; any user). Dedup; persists to
        <store>/profiles.json. Returns the refreshed list."""
        name = (name or "").strip()
        if name:
            p = self._profiles_path()
            try:
                cur = json.loads(p.read_text()) if p.exists() else []
            except (OSError, ValueError):
                cur = []
            if name not in cur:
                cur.append(name)
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(json.dumps(sorted(set(cur), key=str.lower), indent=2) + "\n")
        return self._read_profiles()

    # ---- one-submission-per-ranker (STO-SCN-109) ------------------------
    @staticmethod
    def _latest_score_rows(rows: list) -> list:
        """Keep only the rows of the LATEST submission per (rater, slot). A
        submission = the set of per-variant rows sharing a ts; re-ranking
        replaces. ISO8601 ts sorts correctly under a single tz offset."""
        latest: dict = {}
        for r in rows:
            k = (r.get("rater"), r.get("slot"))
            ts = r.get("ts") or ""
            if ts >= latest.get(k, ""):
                latest[k] = ts
        return [r for r in rows
                if (r.get("ts") or "") == latest.get((r.get("rater"), r.get("slot")))]

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
            # one-submission-per-ranker (STO-SCN-109): collapse to the latest
            # submission per (rater, slot) before grouping, so each person counts
            # once per view even if stray older rows linger in the file.
            all_rows = [json.loads(line) for line in sj.read_text().splitlines() if line.strip()]
            groups: dict = {}
            for s in self._latest_score_rows(all_rows):
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
            # one submission per (rater, slot) (STO-SCN-109): a re-submit OVERWRITES.
            # Drop this rater's prior rows for this slot, then append the new set.
            sj = scene_dir / "scores.jsonl"
            kept = []
            if sj.exists():
                for line in sj.read_text().splitlines():
                    if not line.strip():
                        continue
                    s = json.loads(line)
                    if not (s.get("rater") == row["rater"] and s.get("slot") == slot):
                        kept.append(s)
            scene_dir.mkdir(parents=True, exist_ok=True)
            with open(sj, "w") as f:
                for r in kept + rows:
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
