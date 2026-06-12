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
TASKS_DIR = REPO / "tasks-v4"
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
    algo: 'name@version'."""
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
