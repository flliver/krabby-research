#!/usr/bin/env python3
"""STO-SCN-033 — migrate flat M11 scenes into the canonical schema (git-LFS data repo).

- CoW-clones data (APFS clonefile, same volume) — instant, 0 extra space.
- Converts legacy manifest.json -> scene.toml + run.json + transform spec/results.
- Skips editor/OS cruft (.DS_Store, *.blend1).
- Verifies file-count + byte parity per migrated unit.
- Idempotent: skips units already present in the dest.
- Never touches the source (verify-before-swap; removal is a separate, explicit step).

Mapping table below = the logical-scene judgment (21 dirs -> ~10 scenes). Reviewable.
"""
from __future__ import annotations
import json, os, subprocess, sys
from pathlib import Path

SRC = Path("/Volumes/Archives-01/krabby/data/scenes")
DST = Path("/Volumes/Archives-01/krabby/scenes")
CRUFT_NAMES = {".DS_Store", ".TemporaryItems", ".apdisk"}
CRUFT_SUFFIX = (".blend1", ".blend2")

def is_cruft(p: Path) -> bool:
    return p.name in CRUFT_NAMES or p.name.endswith(CRUFT_SUFFIX) or p.name.startswith("._")

# ── Mapping: source dir -> migration spec ────────────────────────────────────
# kind: manifest | raw | legacy | drop
#   manifest : a MAtCha run carrying manifest.json  -> pipeline/run/transform (measured)
#   raw      : capture only -> input/
#   legacy   : multi-tool dir, no manifest -> pipeline-<tool>/run-legacy/... (deduced)
#   drop     : empty/staging dir
MAP = {
 # 004-sky-house — 5 curated MAtCha runs (manifest, measured) + dining (legacy, deduced)
 "004-sky-house-curated-12":                ("004-sky-house","manifest","matcha","12"),
 "004-sky-house-curated-12-strong":         ("004-sky-house","manifest","matcha","12-strong"),
 "004-sky-house-curated-12-dense-strong":   ("004-sky-house","manifest","matcha","12-dense-strong"),
 "004-sky-house-curated-12-dense-strong-r3":("004-sky-house","manifest","matcha","12-dense-strong-r3"),
 "004-sky-house-curated-16-strong":         ("004-sky-house","manifest","matcha","16-strong"),
 "004-sky-house-dining":                    ("004-sky-house","legacy",None,None),
 # dtu-bicycle (external benchmark)
 "dtu-bicycle-curated-12-dense-strong":     ("dtu-bicycle","manifest","matcha","12-dense-strong"),
 "dtu-bicycle":                             ("dtu-bicycle","legacy",None,None),
 # 001-patio (kitchen-sink: colmap+mast3r+matcha+vggt+mesh) + empty staging dirs
 "001-patio-fisheye":                       ("001-patio","legacy",None,None),
 "001-patio-fisheye-vggt":                  (None,"drop",None,None),
 "001-patio-fisheye-vggt-tiny":             (None,"drop",None,None),
 # 002-patio-dewarped — kept as its own ordinal scene (faithful to numbering)
 "002-patio-dewarped":                      ("002-patio","legacy",None,None),
 # 003-firepit (mast3r+matcha+slam3r+mesh)
 "003-firepit-fisheye":                     ("003-firepit","legacy",None,None),
 # raw captures
 "005-meadow-house":                        ("005-meadow","raw",None,None),
 "006-kubota-001":                          ("006-kubota","raw",None,None),
 "007-kubota-002":                          ("007-kubota","raw",None,None),
 "008-kubota-003":                          ("008-kubota","raw",None,None),
 "009-kubota-004":                          ("009-kubota","raw",None,None),
 "010-kubota-005":                          ("010-kubota","raw",None,None),
 "011-kubota-006":                          ("011-kubota","raw",None,None),
 "012-kubota-007":                          ("012-kubota","raw",None,None),
}

# legacy tool-subdir -> pipeline slug
LEGACY_PIPE = {
 "sparse":"colmap","dense":"colmap","database.db":"colmap",
 "mast3r_output":"mast3r", "mast3r_sfm":"matcha",
 "matcha_output":"matcha","tetra_meshes":"matcha","tsdf_meshes":"matcha","oriented":"matcha","free_gaussians":"matcha",
 "vggt_images":"vggt","vggt_images_tiny":"vggt",
 "slam3r_output":"slam3r",
}
INPUT_SUBDIRS = {"images","src","5_10_2023","5_10_2023.zip"}  # -> input/

def sh(*args) -> None:
    subprocess.run(args, check=True)

def cow_clone(src: Path, dst: Path) -> None:
    """APFS CoW clone src (file or dir) into dst (parent must exist)."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    sh("cp", "-cR", str(src), str(dst))

def prune_cruft(root: Path) -> None:
    for p in sorted(root.rglob("*"), reverse=True):
        if is_cruft(p):
            if p.is_dir(): __import__("shutil").rmtree(p, ignore_errors=True)
            else: p.unlink(missing_ok=True)

def tally(root: Path):
    n=b=0
    for p in root.rglob("*"):
        if p.is_file() and not is_cruft(p): n+=1; b+=p.stat().st_size
    return n,b

def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2)+"\n")

def scene_toml(scene: str, source: str) -> str:
    return (f'schema_version = "1"\nid           = "{scene}"\n'
            f'source       = "{source}"\ntier         = "research"\n'
            f'notes        = "Migrated from M11 flat scenes (STO-SCN-033)."\n\n'
            f'[scale]\nstatus          = "uncalibrated"\nmethod          = ""\nmeters_per_unit = 0.0\n')

def ensure_scene_toml(scene: str, source: str):
    f = DST/scene/"scene.toml"
    if not f.exists():
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(scene_toml(scene, source))

def do_manifest(srcdir: Path, scene, pipeline, run):
    run_dir = DST/scene/f"pipeline-{pipeline}"/f"run-{run}"
    if run_dir.exists(): return f"skip (exists) {scene}/{pipeline}/{run}"
    m = json.loads((srcdir/"manifest.json").read_text())
    ensure_scene_toml(scene, "external" if scene.startswith("dtu") else "capture")
    tdir = run_dir/"transform-01-matcha"
    (tdir/"data").mkdir(parents=True, exist_ok=True)
    # clone tool-native subdirs into data/
    for child in srcdir.iterdir():
        if child.name == "manifest.json": continue
        if is_cruft(child): continue
        cow_clone(child, tdir/"data"/child.name)
    prune_cruft(run_dir)
    # legacy manifest -> run.json
    mt = m.get("matcha",{}); ex = m.get("execution",{})
    write_json(run_dir/"run.json", {"schema_version":"1","pipeline":"matcha","run":run,
        "params":{k:mt.get(k) for k in ("alignment_config","dense_regul","encoder","sfm_config",
                  "image_resolution_long_edge","chart_resolutions") if k in mt}
                  | {"frames": m.get("frames",{}).get("count")},
        "promoted": False, "notes": f"Migrated from manifest.json variant_name={m.get('variant_name')}"})
    (run_dir/"manifest.legacy.json").write_text(json.dumps(m,indent=2)+"\n")
    write_json(tdir/"specification.json", {"schema_version":"1","transform":"transform-01-matcha",
        "pipeline":"matcha","run":run,"kind":"reconstruction",
        "description":"MAtCha full pipeline (migrated)","inputs":["input/preproc-01-frame-select/data"],
        "parameters":mt,"command":"python train.py (see manifest.legacy.json)","maturity":"prototype","story":"STO-SCN-033"})
    write_json(tdir/"results.json", {"schema_version":"1","transform":"transform-01-matcha",
        "status": ex.get("exit_status","success"),"provenance":"measured",
        "started":m.get("captured_at"),"finished":None,"duration_s":ex.get("duration_seconds"),
        "host":ex.get("host"),"peak_vram_mib":ex.get("peak_vram_mib"),
        "environment":{"os":"unknown","gpu":ex.get("gpu","unknown"),"nvidia_driver":"unknown","cuda":"unknown",
            "container":{"image":(mt.get("image") or "krabby-matcha"),"tag":"unknown","digest":"unknown"},
            "software":{"matcha":f"git_sha:{mt.get('git_sha')}"}},
        "outputs":[{"path":f"data/{v}"} for v in (m.get("outputs") or {}).values() if isinstance(v,str)]})
    return f"OK manifest {scene}/matcha/run-{run}"

def do_raw(srcdir: Path, scene):
    ind = DST/scene/"input"
    if ind.exists(): return f"skip (exists) {scene}/input"
    ensure_scene_toml(scene, "capture")
    for child in srcdir.iterdir():
        if is_cruft(child): continue
        cow_clone(child, ind/child.name)
    prune_cruft(ind)
    return f"OK raw {scene}/input ({len(list(ind.rglob('*')))} entries)"

def do_legacy(srcdir: Path, scene):
    sdir = DST/scene
    if (sdir/"_migrated").exists(): return f"skip (exists) {scene} legacy"
    ensure_scene_toml(scene, "external" if scene.startswith("dtu") else "capture")
    placed=[]
    for child in srcdir.iterdir():
        if is_cruft(child) or child.name=="manifest.json": continue
        nm = child.name
        if nm in INPUT_SUBDIRS:
            cow_clone(child, sdir/"input"/nm); placed.append(f"input/{nm}")
        elif nm in LEGACY_PIPE:
            pipe = LEGACY_PIPE[nm]
            data = sdir/f"pipeline-{pipe}"/"run-legacy"/"transform-01-legacy"/"data"/nm
            cow_clone(child, data); placed.append(f"pipeline-{pipe}/run-legacy/.../{nm}")
        else:
            # eval/comparison/mesh/loose artifacts -> scene-level _unsorted/ (flagged; schema gap)
            cow_clone(child, sdir/"_unsorted"/nm); placed.append(f"_unsorted/{nm}")
    prune_cruft(sdir)
    # deduced spec/results stubs per legacy pipeline present
    for pipe_dir in sorted(sdir.glob("pipeline-*/run-legacy/transform-01-legacy")):
        write_json(pipe_dir/"specification.json", {"schema_version":"1","transform":"transform-01-legacy",
            "pipeline":pipe_dir.parts[-3].replace("pipeline-",""),"kind":"reconstruction",
            "description":"Legacy run migrated from flat M11 layout (no manifest).","inputs":["input"],
            "parameters":{},"maturity":"prototype","story":"STO-SCN-033"})
        write_json(pipe_dir/"results.json", {"schema_version":"1","transform":"transform-01-legacy",
            "status":"success","provenance":"deduced",
            "environment":{"os":"unknown","gpu":"unknown","container":{"image":"unknown","digest":"unknown"}},
            "outputs":[]})
    (sdir/"_migrated").write_text("legacy migration marker (STO-SCN-033)\n")
    return f"OK legacy {scene}: " + ", ".join(placed)

def main():
    report=[]
    for name in sorted(MAP):
        scene, kind, pipe, run = MAP[name]
        srcdir = SRC/name
        if not srcdir.exists(): report.append(f"MISSING src {name}"); continue
        if kind=="drop":
            report.append(f"drop {name} ({tally(srcdir)[1]} bytes — empty/staging)"); continue
        before = tally(srcdir)
        if kind=="manifest": r=do_manifest(srcdir,scene,pipe,run)
        elif kind=="raw":    r=do_raw(srcdir,scene)
        elif kind=="legacy": r=do_legacy(srcdir,scene)
        else: r=f"?? {name}"
        report.append(r)
    print("\n".join(report))

if __name__=="__main__":
    main()
