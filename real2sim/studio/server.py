"""Pipeline Studio server (STO-SCN-072, EPI-SCN-PIPELINE-STUDIO).

Pure stdlib http.server, same mold as rate_renders — no build step.
Browse A–F, edit pipeline_instances via catalog-constrained forms,
diff two instances' settings. Read-only against the scene store;
writes ONLY repo-side `real2sim/instances/*.json` (E objects).

Routes:
    GET  /                       → index.html
    GET  /api/tasks              → task catalog (A)
    GET  /api/pipelines          → pipelines (D)
    GET  /api/instances          → pipeline_instances (E)
    GET  /api/runs[?scene=<s>]   → pipeline_runs incl. task_runs (C/F)
    GET  /api/leaderboard/<scene>→ scores joined from rankings.jsonl
    POST /api/instances          → save instance (validated vs catalog)

Run:  python3 real2sim/studio/server.py [--port 8091]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "rate_renders"))
import studio_model as sm  # noqa: E402
# Ranking absorbed (STO-SCN-074, T-023): ONE ranking implementation —
# Studio's handler subclasses the rate_renders handler, inheriting the
# render-resolve, rankings append, and Borda aggregate routes verbatim.
# rate_renders remains runnable standalone until the operator confirms
# Studio covers the flow (T-020).
import importlib.util as _ilu  # noqa: E402

_spec = _ilu.spec_from_file_location("rate_renders_server",
                                     REPO / "rate_renders" / "server.py")
rr = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(rr)

try:
    import jsonschema
except ImportError:
    jsonschema = None  # stdlib fallback below still enforces type/range/enum


def _fallback_errors(taskdef: dict, settings: dict) -> list[str]:
    """Minimal validator (type / minimum / maximum / enum) so the form
    server NEVER accepts out-of-range settings, even without the
    jsonschema package (T-003 — no silent validation skip)."""
    errs = []
    props = taskdef.get("properties", {})
    for k, v in settings.items():
        p = props.get(k)
        if p is None:
            errs.append(f"unknown setting: {k}")
            continue
        t = p.get("type")
        if t == "integer" and not isinstance(v, int) or \
           t == "number" and not isinstance(v, (int, float)) or \
           t == "string" and not isinstance(v, str) or \
           t == "array" and not isinstance(v, list):
            errs.append(f"{k}: expected {t}, got {type(v).__name__}")
            continue
        if "minimum" in p and isinstance(v, (int, float)) and v < p["minimum"]:
            errs.append(f"{k}: {v} is less than the minimum of {p['minimum']}")
        if "maximum" in p and isinstance(v, (int, float)) and v > p["maximum"]:
            errs.append(f"{k}: {v} is greater than the maximum of {p['maximum']}")
        if "enum" in p and v not in p["enum"]:
            errs.append(f"{k}: {v!r} not in {p['enum']}")
    # conditional ceilings (allOf if/then with required+const, e.g. da3-infer)
    for cond in taskdef.get("allOf", []):
        if_props = cond.get("if", {}).get("properties", {})
        if all(settings.get(k) == s.get("const") for k, s in if_props.items()) and \
           all(k in settings for k in cond.get("if", {}).get("required", [])):
            for k, s in cond.get("then", {}).get("properties", {}).items():
                v = settings.get(k)
                if "maximum" in s and isinstance(v, (int, float)) and v > s["maximum"]:
                    errs.append(f"{k}: {v} exceeds {s['maximum']} when "
                                f"{dict((kk, ss.get('const')) for kk, ss in if_props.items())}")
    return errs


def validate_instance(inst: dict) -> list[str]:
    errs = []
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]*", inst.get("name", "")):
        errs.append("name must be kebab-case")
    pipes = sm.pipelines()
    pipe = pipes.get(inst.get("pipeline"))
    if not pipe:
        return errs + [f"unknown pipeline: {inst.get('pipeline')}"]
    cat = sm.tasks()
    node_task = {n["id"]: n["task"] for n in pipe["nodes"]}
    variables = inst.get("variables", {})
    for node_id, settings in inst.get("settings", {}).items():
        if node_id not in node_task:
            errs.append(f"settings for unknown node: {node_id}")
            continue
        taskdef = cat[node_task[node_id]]
        # expand $var refs before range validation
        expanded = {}
        for k, v in settings.items():
            if isinstance(v, str) and v.startswith("$"):
                if v[1:] not in variables:
                    errs.append(f"{node_id}.{k}: undeclared variable {v}")
                    continue
                v = variables[v[1:]]
            expanded[k] = v
        expanded = {k: v for k, v in expanded.items()
                    if not (isinstance(v, str) and "(operator)" in v)}
        if jsonschema:
            sub = {**taskdef, "required": []}
            msgs = [e.message for e in
                    jsonschema.Draft202012Validator(sub).iter_errors(expanded)]
        else:
            msgs = _fallback_errors(taskdef, expanded)
        errs += [f"{node_id}: {m}" for m in msgs]
    return errs


class Handler(rr.Handler):
    def _send(self, code: int, body: bytes, ctype="application/json"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code=200):
        self._send(code, json.dumps(obj).encode())

    def do_GET(self):  # noqa: N802
        url = urlparse(self.path)
        parts = [p for p in url.path.split("/") if p]
        if url.path == "/":
            self._send(200, (HERE / "index.html").read_bytes(), "text/html")
        elif url.path == "/api/tasks":
            self._json(sm.tasks())
        elif url.path == "/api/pipelines":
            self._json(sm.pipelines())
        elif url.path == "/api/instances":
            self._json(sm.instances())
        elif url.path == "/api/runs":
            q = parse_qs(url.query)
            self._json(sm.runs(q["scene"][0] if "scene" in q else None))
        elif len(parts) == 3 and parts[:2] == ["api", "leaderboard"]:
            self._json(sm.leaderboard(parts[2]))
        elif url.path == "/rank":
            # the absorbed rate_renders app, embedded under Studio
            self._send(200, (REPO / "rate_renders" / "static" / "index.html").read_bytes(),
                       "text/html")
        elif len(parts) == 4 and parts[:2] == ["api", "scout"]:
            self._json(self._scout_build(parts[2], parts[3]))   # STO-SCN-135
        elif len(parts) >= 3 and parts[0] == "scout":
            self._serve_scout_file(parts[1], parts[2], "/".join(parts[3:]) or "viewer.html")
        else:
            # rankings / renders / scenes / aggregate / static — inherited
            return super().do_GET()

    # STO-SCN-135: "Open in Scout" — build a mesh+spine+highlighted-subset viewer for a variant
    # (via a uv subprocess; the studio's python has no numpy) and serve it.
    @staticmethod
    def _scout_dir(scene: str, variant: str):
        import pathlib
        return pathlib.Path(f"/tmp/scout-mesh-{scene}-{variant}")

    def _scout_build(self, scene: str, variant: str) -> dict:
        import subprocess
        d = self._scout_dir(scene, variant)
        if not (d / "viewer.html").exists():
            r = subprocess.run(
                ["uv", "run", "--quiet", "--python", "3.11", "--with", "numpy",
                 "python3", str(REPO / "verify_viewer" / "build_scout_mesh.py"), scene,
                 "--mesh", variant, "--no-serve", "--serve-dir", str(d)],
                capture_output=True, text=True, cwd=str(REPO))
            if r.returncode != 0 or not (d / "viewer.html").exists():
                return {"error": "scout build failed", "detail": (r.stderr or r.stdout)[-1500:]}
        return {"url": f"/scout/{scene}/{variant}/viewer.html"}

    def _serve_scout_file(self, scene: str, variant: str, rel: str):
        import shutil as _sh
        fp = self._scout_dir(scene, variant) / rel
        if not fp.exists():
            return self._send(404, b"not found", "text/plain")
        if fp.suffix == ".ply":                 # stream (meshes are 100s of MB; follows symlink)
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(fp.stat().st_size))
            self.end_headers()
            with fp.open("rb") as f:
                _sh.copyfileobj(f, self.wfile)
            return
        ctype = "text/html" if fp.suffix == ".html" else \
                "application/json" if fp.suffix == ".json" else "text/plain"
        self._send(200, fp.read_bytes(), ctype)

    def do_POST(self):  # noqa: N802
        if self.path.startswith("/api/rankings/") or self.path == "/api/profiles":
            return super().do_POST()        # rankings.jsonl / profiles.json, inherited
        if self.path != "/api/instances":
            return self._json({"error": "not found"}, 404)
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        inst = json.loads(body)
        inst.setdefault("schema", 3)
        errs = validate_instance(inst)
        if errs:
            return self._json({"ok": False, "errors": errs}, 422)
        out = REPO / "instances" / f"{inst['name']}.json"
        out.write_text(json.dumps(inst, indent=2) + "\n")
        self._json({"ok": True, "path": str(out)})

    def log_message(self, fmt, *args):  # quiet
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8091)
    args = ap.parse_args()
    print(f"Pipeline Studio: http://localhost:{args.port}/")
    ThreadingHTTPServer(("0.0.0.0", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
