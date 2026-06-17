#!/usr/bin/env python3
"""v4core.py — DAG-of-DAGs model core (STO-SCN-079, HUG-SCN-005).

Implements the locked decisions:

  #1  refs (primary, canonical) resolve to concrete hashes before use
  #3  IDENTITY_HASH = hash(resolved inputs + tunable + frozen settings
      + algo@version); digest is audit metadata, never keyed
  #4  vocabulary task/graph/job; jobs MATERIALIZE (NOOP when the
      identity already exists)
  #5  subset identity = HOH of member image hashes, content-only
  #7  view identity = content hash of view.json; render keyed on the
      view, never the set
  #8  graph defs in repo (real2sim/graphs/), job records in
      scenes/<scene>/jobs/
  #10 license eligibility derived by walking ancestry

Identity format (operator draft): uppercase [0-9A-Z]+ — first 12 chars
of uppercase base32(sha256), no padding.
"""
from __future__ import annotations

import base64
import datetime
import hashlib
import json
import os
from pathlib import Path

REPO = Path(__file__).parent
STORE = Path(os.environ.get("KRABBY_SCENES_ROOT", "/var/krabby/scenes"))
TASKS_DIR = REPO / "tasks"
GRAPHS_DIR = REPO / "graphs"

ID_LEN = 12


# ---------------------------------------------------------------- identity

def _canon(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()


def content_hash(data: bytes) -> str:
    """[0-9A-Z]+ identity: uppercase base32(sha256), first ID_LEN chars."""
    return base64.b32encode(hashlib.sha256(data).digest()).decode().rstrip("=")[:ID_LEN]


def identity_hash(resolved_inputs: dict, settings: dict, algo: str) -> str:
    """Locked #3. resolved_inputs: name -> identity string (refs already
    resolved); settings: tunable+frozen only (pins live in algo@version);
    algo: 'name@version'.

    STO-SCN-155 dev-loop guardrail: when KRABBY_DEV_TOOLS is set, the engine
    containers run with live `real2sim/` tools bind-mounted over the baked
    /opt/krabby-tools — code that is NOT captured by the identity (identity keys
    on inputs+settings+algo, never the tool bytes). Salting the algo with `+dev`
    routes those runs to a distinct identity namespace so an unprovenanced dev
    result can never overwrite a canonical store node (honors STO-SCN-093 D).
    Stages invoked with explicit canonical upstream refs (e.g. `covis --solve
    <id>`) still reuse that upstream — only the recomputed stage is isolated."""
    if os.environ.get("KRABBY_DEV_TOOLS"):
        algo = f"{algo}+dev"
    return content_hash(_canon({"in": resolved_inputs, "set": settings, "algo": algo}))


def hoh(hashes: list[str]) -> str:
    """Locked #5: hash-of-hashes — sorted, then hashed."""
    return content_hash(_canon(sorted(hashes)))


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return base64.b32encode(h.digest()).decode().rstrip("=")[:ID_LEN]


# ---------------------------------------------------------------- defs

def tasks() -> dict[str, dict]:
    return {d["name"]: d for d in
            (json.loads(p.read_text()) for p in sorted(TASKS_DIR.glob("*.json")))}


def graphs() -> dict[str, dict]:
    return {d["name"]: d for d in
            (json.loads(p.read_text()) for p in sorted(GRAPHS_DIR.glob("*.json")))}


def hashable_settings(taskdef: dict, settings: dict) -> dict:
    """Tunable + frozen enter the hash; anything classified 'pin' is
    refused (pins live in algo@version — locked #3)."""
    cls = taskdef.get("settings", {})
    out = {}
    for k, v in settings.items():
        kind = cls.get(k, {}).get("class")
        if kind == "pin":
            raise ValueError(f"{taskdef['name']}: '{k}' is a pin — not a hashable setting")
        out[k] = v
    # frozen defaults participate even when unspecified (constant today)
    for k, spec in cls.items():
        if k not in out and spec.get("class") in ("tunable", "frozen") and "default" in spec:
            out[k] = spec["default"]
    return out


# ---------------------------------------------------------------- scene + refs

class Scene:
    def __init__(self, name: str, root: Path | None = None):
        self.name = name
        self.dir = (root or STORE) / name

    # ---- refs (locked #1 + #7: set-if-unset; moves are operator acts)
    def _ref_path(self, ref: str) -> Path:
        return {"primary": self.dir / "images" / "subsets" / "primary",
                "canonical": self.dir / "viewset" / "canonical"}[ref]

    def resolve(self, ref: str) -> str:
        p = self._ref_path(ref)
        if ref == "canonical":
            # canonical is a directory holding the mutable members file
            members = json.loads((p / "views.json").read_text())["slots"]
            return members  # list of slots; views resolve per-slot
        if not p.is_symlink():
            raise FileNotFoundError(f"{self.name}: ref '{ref}' not set")
        return os.readlink(p)

    def set_ref_if_unset(self, ref: str, target: str) -> bool:
        p = self._ref_path(ref)
        if p.exists() or p.is_symlink():
            return False                      # never move (operator act)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.symlink_to(target)
        return True

    # ---- views (locked #7)
    def view_content_hash(self, slot: str) -> str:
        return content_hash((self.dir / "views" / slot / "view.json").read_bytes())

    def canonical_views(self) -> dict[str, str]:
        """slot -> content hash, resolved NOW (recorded by render jobs)."""
        return {slot: self.view_content_hash(slot) for slot in self.resolve("canonical")}

    # ---- jobs (locked #8)
    def job_dir(self) -> Path:
        ts = datetime.datetime.now().astimezone().strftime("%Y%m%dT%H%M%S")
        sid = content_hash(os.urandom(8))[:4]
        d = self.dir / "jobs" / f"{ts}-{sid}"
        d.mkdir(parents=True, exist_ok=True)
        return d


# ---------------------------------------------------------------- metadata

def write_metadata(out_dir: Path, *, task: str, algo: str, identity: str,
                   resolved_inputs: dict, settings: dict, measured: dict | None = None,
                   mechanism: str | None = None, migrated: bool = False,
                   origin: str | None = None, extra: dict | None = None) -> None:
    """Per-identity metadata: what this artifact IS (locked #8)."""
    md = {
        "schema": 4, "task": task, "algo": algo, "identity": identity,
        "resolved_inputs": resolved_inputs, "settings": settings,
        "measured": measured or {},
        "mechanism": mechanism,
        "migrated": migrated,
        "origin": origin,
        "written": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    if extra:
        md.update(extra)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metadata.json").write_text(json.dumps(md, indent=2) + "\n")


# ---------------------------------------------------------------- planner

def plan(graph_name: str, scene: Scene, bindings: dict) -> list[dict]:
    """Materialize-check pass (locked #4): for each node in topo order,
    compute identity + output dir; mark NOOP if it exists.
    Returns the plan; execution backends consume it."""
    g = graphs()[graph_name]
    tdefs = tasks()
    node_by_id = {n["id"]: n for n in g["nodes"]}
    # topo order
    indeg = {n["id"]: 0 for n in g["nodes"]}
    adj = {n["id"]: [] for n in g["nodes"]}
    for a, b in g["edges"]:
        adj[a].append(b)
        indeg[b] += 1
    order, q = [], [i for i, d in indeg.items() if d == 0]
    while q:
        i = q.pop(0)
        order.append(i)
        for j in adj[i]:
            indeg[j] -= 1
            if indeg[j] == 0:
                q.append(j)
    plan_rows = []
    produced: dict[str, dict] = {}   # node id -> {identity, out_dir}
    for nid in order:
        node = node_by_id[nid]
        tdef = tdefs[node["task"]]
        algo = tdef["algo"]
        settings = hashable_settings(tdef, bindings.get("settings", {}).get(nid, {}))
        # resolve inputs: from bindings, refs, or upstream nodes
        resolved = {}
        for inp in tdef.get("inputs", []):
            name = inp["name"]
            if name in bindings.get("inputs", {}).get(nid, {}):
                resolved[name] = bindings["inputs"][nid][name]
            elif inp.get("from") in produced:
                resolved[name] = produced[inp["from"]]["identity"]
            elif inp.get("ref"):
                resolved[name] = scene.resolve(inp["ref"])
        identity = identity_hash(resolved, settings, algo)
        out_dir = scene.dir / tdef["placement"].format(
            **{**{k: v for k, v in resolved.items()}, "identity": identity,
               **{f"up_{k}": v["identity"] for k, v in produced.items()}})
        exists = (out_dir / "metadata.json").exists()
        plan_rows.append({"node": nid, "task": node["task"], "algo": algo,
                          "identity": identity, "settings": settings,
                          "resolved_inputs": resolved,
                          "out_dir": str(out_dir),
                          "action": "NOOP" if exists else "EXECUTE"})
        produced[nid] = {"identity": identity, "out_dir": str(out_dir)}
    return plan_rows


# ---------------------------------------------------------------- store scan (read-side)

def scan_scene(scene: str) -> dict:
    """Read-side view of a v4 scene: subsets, cameras, representations,
    meshes, renders, views, scores. Powers Studio + rate_renders."""
    sdir = STORE / scene
    out = {"scene": scene, "subsets": [], "representations": [], "views": {},
           "scores": [], "primary": None}
    pr = sdir / "images" / "subsets" / "primary"
    if pr.is_symlink():
        out["primary"] = os.readlink(pr)
    for sub in sorted(sdir.glob("images/subsets/*/")):
        if sub.name == "primary" or not (sub / "subset.json").exists():
            continue
        md = json.loads((sub / "metadata.json").read_text()) if (sub / "metadata.json").exists() else {}
        out["subsets"].append({
            "hash": sub.name,
            "n": len(json.loads((sub / "subset.json").read_text())["members"]),
            "label": md.get("label"), "mechanism": md.get("mechanism"),
            "solves": [c.name for c in sorted((sub / "cameras").glob("*/"))
                       if (c / "metadata.json").exists()] if (sub / "cameras").is_dir() else []})
    for rep in sorted(sdir.glob("represent/*/*/")):
        if not (rep / "metadata.json").exists():
            continue
        rmd = json.loads((rep / "metadata.json").read_text())
        meshes = []
        for mdir in sorted(rep.glob("meshify/*/*/")) :
            if not (mdir / "metadata.json").exists():
                continue
            mmd = json.loads((mdir / "metadata.json").read_text())
            # rankable:false artifacts stay VISIBLE, annotated (operator
            # call, 2026-06-12): mis-alignment is a quality detractor the
            # ranker should see and score down — not a reason to hide.
            entry = {"method": mdir.parent.name, "identity": mdir.name,
                     "settings": mmd.get("settings", {}),
                     "rankable": mmd.get("rankable", True) is not False,
                     "quality_flag": (mmd.get("rankable_reason") or "flagged")
                                     if mmd.get("rankable") is False else None,
                     "self_alignment": mmd.get("measured", {}).get("self_alignment"),
                     "renders": sorted(r.parent.name for r in mdir.glob("renders/*/render.png")),
                     "conditioned": []}
            for cdir in sorted(mdir.glob("condition/*/")):
                if (cdir / "metadata.json").exists():
                    cmd_ = json.loads((cdir / "metadata.json").read_text())
                    entry["conditioned"].append({
                        "identity": cdir.name, "settings": cmd_.get("settings", {}),
                        "algo": cmd_.get("algo"),     # STO-SCN-138: label the cull/condition transform
                        "renders": sorted(r.parent.name for r in cdir.glob("renders/*/render.png"))})
            meshes.append(entry)
        ok, flags = deliverable_eligible(rep)
        out["representations"].append({
            "kind": rep.parent.name, "identity": rep.name,
            "algo": rmd.get("algo"), "settings": rmd.get("settings", {}),
            "legacy_variant": rmd.get("legacy_variant"),
            "migrated": rmd.get("migrated", False),
            "deliverable_eligible": ok, "license_flags": flags,
            "meshes": meshes})
    for vdir in sorted(sdir.glob("views/*/")):
        if (vdir / "view.json").exists():
            out["views"][vdir.name] = content_hash((vdir / "view.json").read_bytes())
    sj = sdir / "scores.jsonl"
    if sj.exists():
        out["scores"] = [json.loads(l) for l in sj.read_text().splitlines() if l.strip()]
    return out


def expected_task_gaps(scene: str) -> list[dict]:
    """STO-SCN-087: the expected set comes from the GRAPHS, not from
    what happens to exist. Returns non-render gaps (task tier):
      - graph branches never run on this scene (e.g. no da3 rep at all)
      - mesh-branch gaps on existing representations
    Render gaps stay where they are (render index vs meshes). These
    gaps need the GPU executor (STO-SCN-088) + an operator host choice
    (HUG-SCN-005 decision 3) to materialize."""
    sc = scan_scene(scene)
    tdefs = tasks()
    gaps = []
    kinds = {r["kind"] for r in sc["representations"]}
    # whole-branch gap: reconstruct-da3 never ran here
    if "da3" not in kinds and sc.get("primary"):
        d = tdefs.get("represent-via-da3", {})
        defaults = {k: s.get("default") for k, s in d.get("settings", {}).items()
                    if s.get("class") in ("tunable", "frozen") and "default" in s}
        gaps.append({"task": "represent-via-da3", "graph": "reconstruct-da3",
                     "label": f"da3 (defaults: {defaults.get('process_res')}/{defaults.get('mode')})",
                     "settings": defaults, "gpu": True,
                     "license_flags": [d.get("license_flag")] if d.get("license_flag") else []})
    # mesh-branch gaps on existing representations
    methods_by_kind = {"matcha": ["tetra", "tsdf"], "da3": ["tsdf"]}
    for r in sc["representations"]:
        have = {m["method"] for m in r["meshes"]}
        for miss in [m for m in methods_by_kind.get(r["kind"], []) if m not in have]:
            gaps.append({"task": f"meshify-via-{miss}", "rep": r["identity"],
                         "label": f"{r.get('legacy_variant') or r['identity']} [{miss}]",
                         "gpu": True,
                         "license_flags": r.get("license_flags", [])})
    return gaps


def leaderboard(scene: str) -> dict:
    """Mean rank per scored identity (lower = better), labels resolved."""
    sc = scan_scene(scene)
    label = {}
    for rep in sc["representations"]:
        for m in rep["meshes"]:
            label[m["identity"]] = (rep.get("legacy_variant") or rep["identity"]) + f" [{m['method']}]"
            for c in m["conditioned"]:
                label[c["identity"]] = (rep.get("legacy_variant") or rep["identity"]) + " [conditioned]"
        label[rep["identity"]] = rep.get("legacy_variant") or rep["identity"]
    agg: dict[str, list] = {}
    for s in sc["scores"]:
        agg.setdefault(s["at"], []).append(s["rank"])
    flagged = {m["identity"]: m["quality_flag"]
               for rep in sc["representations"] for m in rep["meshes"]
               if not m.get("rankable", True)}
    live = {m["identity"] for rep in sc["representations"] for m in rep["meshes"]}
    live |= {c["identity"] for rep in sc["representations"] for m in rep["meshes"]
             for c in m["conditioned"]}
    # retired identities (scores history outlives artifacts) stay OUT of
    # the live leaderboard (operator-reported: outdated items in list)
    rows = sorted(({"identity": k, "label": label.get(k, k),
                    "mean_rank": round(sum(v) / len(v), 2), "n": len(v),
                    "quality_flag": flagged.get(k)}
                   for k, v in agg.items() if k in live),
                  key=lambda r: r["mean_rank"])
    return {"scene": scene, "rows": rows}


# ---------------------------------------------------------------- license ancestry (locked #10)

def deliverable_eligible(out_dir: Path) -> tuple[bool, list[str]]:
    """NC anywhere in ancestry => not deliverable. License facts live on
    task defs (keyed by algo). Nested placement (locked #6) means every
    ancestor artifact lies ON the path — walk up collecting metadata."""
    lic_by_algo = {d["algo"]: d.get("license_flag") for d in tasks().values()}
    flags = []
    d = out_dir.resolve()
    while len(d.parts) > 2:
        md_file = d / "metadata.json"
        if md_file.exists():
            md = json.loads(md_file.read_text())
            flag = lic_by_algo.get(md.get("algo"))
            if flag:
                flags.append(f"{md['algo']}: {flag}")
        d = d.parent
    return (not flags, flags)
