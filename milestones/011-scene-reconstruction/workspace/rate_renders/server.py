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
from manifest_lib import read_manifest, list_variants  # type: ignore[import-not-found]  # noqa: E402

SCENES_ROOT = WORKSPACE.parent / "data" / "scenes"
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
        return self._not_found()

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
        """List scene roots — every dir under data/scenes/ that contains a
        comparison_views.json (and isn't itself a -curated-* variant).

        A scene is identifiable purely by having that file. The variant
        prefix (used to discover sibling variant directories) is stored
        inside comparison_views.json under the `variant_prefix` field;
        we don't infer it from the directory name.
        """
        out = []
        if not SCENES_ROOT.is_dir():
            return out
        for d in sorted(SCENES_ROOT.iterdir()):
            if not d.is_dir():
                continue
            if "-curated-" in d.name:
                continue   # variants aren't scene roots
            if (d / "comparison_views.json").exists():
                out.append(d.name)
        return out

    def _scene_payload(self, scene: str) -> dict:
        """Return everything the frontend needs to render a scene's UI."""
        scene_dir = SCENES_ROOT / scene
        if not scene_dir.is_dir():
            return {"error": f"scene not found: {scene}"}
        # 1) views + variant prefix (from comparison_views.json)
        views_path = scene_dir / "comparison_views.json"
        views: list = []
        # Default variant prefix to the scene-root name (works for bicycle,
        # which uses dtu-bicycle/ → dtu-bicycle-curated-*). Existing scenes
        # like 004-sky-house-dining override via the explicit field because
        # their variants share a different stem (004-sky-house-curated-*).
        variant_prefix = scene
        if views_path.exists():
            with open(views_path) as f:
                cv = json.load(f)
            views = [v["name"] for v in cv.get("views", [])]
            variant_prefix = cv.get("variant_prefix", variant_prefix)
        # 2) variants — siblings whose names start with `<variant_prefix>-curated-`
        variants = list_variants(str(SCENES_ROOT), variant_prefix)
        # 3) manifests for each variant. Always pass the FULL directory name to
        #    read_manifest (variant_prefix + "-curated-" + suffix) — passing the
        #    bare suffix collides with siblings sharing the same suffix in
        #    other scenes (e.g., 004-sky-house-curated-12-dense-strong vs
        #    dtu-bicycle-curated-12-dense-strong).
        manifests = {}
        for v in variants:
            full_name = f"{variant_prefix}-curated-{v}"
            try:
                manifests[v] = read_manifest(str(SCENES_ROOT), full_name)
            except (FileNotFoundError, OSError, ValueError):
                manifests[v] = {
                    "variant_name": v,
                    "notes": "(no manifest captured yet)",
                }
        # 4) which (view, variant) PNGs exist
        renders_dir = scene_dir / "comparison_renders"
        rendered = {}
        if renders_dir.is_dir():
            for view_dir in renders_dir.iterdir():
                if view_dir.is_dir():
                    rendered[view_dir.name] = sorted(
                        f.stem for f in view_dir.glob("*.png")
                    )
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
        """rel is like '004-sky-house-dining/compare_01/12-strong.png'."""
        # Path-shape sanity check + security clamp
        try:
            scene, view, fname = rel.split("/")
        except ValueError:
            return self._bad_request("expected scene/view/variant.png")
        target = (
            SCENES_ROOT / scene / "comparison_renders" / view / fname
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
        # Append-only
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
