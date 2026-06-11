#!/usr/bin/env python3
"""run_pipeline.py — central run trigger (STO-SCN-073, EPI-SCN-PIPELINE-STUDIO).

Launch a pipeline_instance on ONE operator-chosen host (decision 3:
host is a parameter, never a scheduler choice) and capture a complete
v3 run_record. Reproducible by construction:

  - tasks run the image BAKED tools only (`/opt/krabby-tools`) — no
    /tools dev mounts (tooling-provenance policy, RECIPES.md)
  - image digest + tools_git_sha read from the HOST's docker (measured,
    not assumed), input content hashes computed from the store
  - expanded settings (variables resolved) snapshotted into run_record

Dispatch path = the existing SSH + docker flow. No git operations on
the host clone (specs are rsynced over, outputs rsynced back, host run
dir deleted — gather hygiene without touching the host's checkout).

MVP executes the auto-executable nodes it has arg-builders for
(da3-infer, da3-tsdf-mesh, tetra-condition). matcha-reconstruction
keeps its hardened runner (run_transform.py, T-025) — composing that
under this trigger is follow-on work, recorded in the story.

Usage:
    python3 real2sim/run_pipeline.py --instance <name> --scene <scene> \
        --run <run-name> --host <user@host> [--nodes infer,fuse] [--dry-run]
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import studio_model as sm  # noqa: E402

STORE = sm.STORE
HOST_STORE = "/home/jeremy/krabby/scenes"
STORE_PIPELINE = {"matcha-trunk": "matcha", "da3-eval": "da3"}


def sh(cmd: list[str], **kw) -> str:
    return subprocess.run(cmd, check=True, capture_output=True, text=True, **kw).stdout.strip()


# ---------- per-task arg builders: (scene, run_rel, tdir_rel, settings) -> container cmd ----------

def _args_da3_infer(scene, run_rel, tdir_rel, s):
    # NB: out arg is the DATA ROOT — the tool creates exports/, gs_ply/,
    # depth_vis/, gs_video/ underneath it (verified against run-8-giant layout)
    cmd = ["python", "/opt/krabby-tools/da3_infer_gs.py",
           f"/scenes/{scene}/input/src",
           f"/scenes/{scene}/{run_rel}/{tdir_rel}/data",
           str(s["process_res"])]
    if s.get("mode") == "nogs":
        cmd.append("nogs")
    return cmd


def _args_da3_tsdf(scene, run_rel, tdir_rel, s):
    return ["python", "/opt/krabby-tools/da3_tsdf_mesh.py",
            "--scene", f"/scenes/{scene}",
            "--matcha-run", s["matcha_run"],          # anchor cameras (tracked metadata)
            "--da3-run", run_rel,
            "--voxel-frac", str(s.get("voxel_frac", 0.004)),
            "--conf-percentile", str(s.get("conf_percentile", 40))]


def _args_tetra_condition(scene, run_rel, tdir_rel, s):
    return ["python", "/opt/krabby-tools/tetra_condition.py",
            "--in-mesh", s["in_mesh"], "--out-dir", s["out_dir"],
            "--target-tris", str(s["target_tris"]),
            "--taubin-iters", str(s["taubin_iters"])]


BUILDERS = {"da3-infer": _args_da3_infer, "da3-tsdf-mesh": _args_da3_tsdf,
            "tetra-condition": _args_tetra_condition}
EXTRA_SETTINGS = {"da3-tsdf-mesh": ["matcha_run"],  # wiring inputs, not catalog tunables
                  "tetra-condition": ["in_mesh", "out_dir"]}


def expand(settings: dict, variables: dict) -> dict:
    out = {}
    for k, v in settings.items():
        if isinstance(v, str) and v.startswith("$"):
            if v[1:] not in variables:
                sys.exit(f"undeclared variable {v}")
            v = variables[v[1:]]
        out[k] = v
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance", required=True)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--host", required=True, help="operator-chosen execution host (user@host)")
    ap.add_argument("--nodes", default=None, help="comma list; default = all buildable nodes")
    ap.add_argument("--set", action="append", default=[],
                    help="node.key=value overrides for wiring settings (e.g. fuse.matcha_run=...)")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    inst = sm.instances()[a.instance]
    pipe = sm.pipelines()[inst["pipeline"]]
    cat = sm.tasks()
    p_label = STORE_PIPELINE.get(pipe["name"], pipe["name"])
    run_rel = f"pipeline-{p_label}/run-{a.run}"
    run_dir = STORE / a.scene / f"pipeline-{p_label}" / f"run-{a.run}"
    if run_dir.exists():
        sys.exit(f"REFUSING: {run_dir} already exists")

    # expand variables -> full settings snapshot (every node, not just executed
    # ones — the snapshot IS the reproducibility record). Range validation is
    # enforced at instance-save time by the studio server (072); the trigger
    # re-validates when jsonschema is importable, and skips otherwise.
    expanded = {nid: expand(s, inst.get("variables", {}))
                for nid, s in inst["settings"].items()}
    try:
        from studio.server import validate_instance
        errs = validate_instance(inst)
        if errs:
            sys.exit("instance invalid:\n  " + "\n  ".join(errs))
    except ImportError:
        pass

    for ov in a.set:
        node_key, val = ov.split("=", 1)
        nid, key = node_key.split(".", 1)
        expanded.setdefault(nid, {})[key] = val

    node_order = [n for n in pipe["nodes"]]
    todo = []
    for i, n in enumerate(node_order):
        task = n["task"]
        x = cat[task]["x-task"]
        if a.nodes and n["id"] not in a.nodes.split(","):
            continue
        if x.get("operator") or task not in BUILDERS:
            continue
        todo.append((n["id"], task, x))
    if not todo:
        sys.exit("no executable nodes (operator tasks and builder-less tasks are skipped)")

    print(f"trigger: {a.instance} -> {a.scene}/{run_rel} on {a.host}")
    print(f"executing nodes: {[t[0] for t in todo]}")
    if a.dry_run:
        for nid, task, x in todo:
            print(f"  {nid}: {' '.join(BUILDERS[task](a.scene, run_rel, 'transform-01-' + p_label, expanded[nid]))}")
        return 0

    started = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
    tdir_rel = f"transform-01-{p_label}"
    host_run = f"{HOST_STORE}/{a.scene}/{run_rel}"

    # LFS-pointer input guard (inherited lesson, STO-SCN-041 failure chain):
    # an aborted pull leaves pointers masquerading as inputs on the host
    host_src = f"{HOST_STORE}/{a.scene}/input/src"
    pointers = sh(["ssh", a.host,
                   f"grep -l 'git-lfs.github.com' {host_src}/* 2>/dev/null | head -3; true"])
    if pointers:
        sys.exit(f"REFUSING: LFS pointers masquerading as inputs on {a.host}:\n{pointers}")

    # input content hashes from the (git-tracked) store — the reproducibility anchor
    src = STORE / a.scene / "input" / "src"
    input_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()[:16]
                    for p in sorted(src.iterdir()) if p.is_file()}

    provenance, statuses = {}, {}
    for nid, task, x in todo:
        image = x["image"]
        digest = sh(["ssh", a.host,
                     f"docker inspect --format '{{{{index .RepoDigests 0}}}}' {image} 2>/dev/null || "
                     f"docker images --no-trunc --format '{{{{.ID}}}}' {image} | head -1"])
        tools_sha = sh(["ssh", a.host,
                        f"docker inspect --format '{{{{index .Config.Labels \"io.krabby.da3.tools_git_sha\"}}}}' {image}"]) or None
        cmd = BUILDERS[task](a.scene, run_rel, tdir_rel, expanded[nid])
        # gather hygiene (RECIPES.md): docker writes land root-owned and wedge
        # later rm/pull — chown back to the invoking user in the same dispatch
        docker = (f"mkdir -p {host_run}/{tdir_rel}/data && "
                  f"docker run --rm --gpus all -v {HOST_STORE}:/scenes {image} " + " ".join(cmd) +
                  f" ; rc=$? ; docker run --rm -v {HOST_STORE}:/scenes alpine "
                  f"chown -R $(id -u):$(id -g) /scenes/{a.scene}/{run_rel} ; exit $rc")
        print(f"[{nid}] {a.host}: {' '.join(cmd)}")
        t0 = datetime.datetime.now()
        r = subprocess.run(["ssh", a.host, docker], capture_output=True, text=True)
        dt = int((datetime.datetime.now() - t0).total_seconds())
        log_dir = run_dir / tdir_rel / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        (log_dir / f"{nid}.log").write_text(r.stdout + ("\n--- stderr ---\n" + r.stderr if r.stderr else ""))
        statuses[nid] = "success" if r.returncode == 0 else "failure"
        provenance[nid] = {"image": image, "image_digest": digest,
                           "tools_git_sha": tools_sha,
                           "code_ref": x.get("code_ref"),
                           "input_hashes": input_hashes if nid == todo[0][0] else None}
        print(f"[{nid}] rc={r.returncode} ({dt}s)")
        if r.returncode != 0:
            print(r.stderr[-2000:])
            break

    # gather: rsync host outputs back (data/ -> Mac transient archive role), then rm host copy
    run_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(["rsync", "-a", f"{a.host}:{host_run}/", str(run_dir) + "/"], check=True)
    subprocess.run(["ssh", a.host, f"rm -rf {host_run} && rmdir {HOST_STORE}/{a.scene}/pipeline-{p_label} 2>/dev/null; true"], check=True)

    # expected-outputs hard gate (tool rc=0 lies — STO-SCN-041 lesson):
    # every executed node's catalog-declared outputs must exist post-gather
    import fnmatch
    gathered = [str(p.relative_to(run_dir)) for p in run_dir.rglob("*") if p.is_file()]
    for nid, task, x in todo:
        if statuses.get(nid) != "success":
            continue
        for out_decl in cat[task]["x-task"].get("outputs", []):
            pat = out_decl["pattern"].split("#")[0]
            if not any(fnmatch.fnmatch(g, pat) or fnmatch.fnmatch(g, pat.replace("**/", "*"))
                       for g in gathered):
                statuses[nid] = "failure"
                print(f"[{nid}] EXPECTED-OUTPUT MISSING (rc=0 lied): {pat}")

    finished = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
    overall = ("success" if all(v == "success" for v in statuses.values())
               else "failure" if any(v == "failure" for v in statuses.values()) else "partial")
    flags = [f"{expanded[nid].get('model', task)}: {lic}"
             for nid, task, x in todo
             if (lic := x.get("license_flag"))]
    record = {
        "schema": 3, "scene": a.scene, "pipeline": pipe["name"], "run": a.run,
        "variant": f"{p_label}--{a.run}", "source_run": None,
        "instance": {"name": a.instance, "expanded_settings": expanded},
        "execution": {"host": a.host.split("@")[-1], "trigger": "studio",
                      "started": started, "finished": finished, "status": overall},
        "provenance": provenance,
        "reproducibility": {
            "by_record": all(p["image_digest"] and "sha256:" in p["image_digest"]
                             for p in provenance.values()),
            "license_flags": flags,
            "notes": f"transients at mac:{run_dir}/{tdir_rel}/data (store-shape v2); "
                     f"logs in {tdir_rel}/logs/",
        },
        "backfilled": False, "backfill_notes": None,
    }
    (run_dir / "run_record.json").write_text(json.dumps(record, indent=2) + "\n")
    print(f"run_record: {run_dir / 'run_record.json'} (status: {overall}, "
          f"by_record: {record['reproducibility']['by_record']})")
    return 0 if overall == "success" else 1


if __name__ == "__main__":
    sys.exit(main())
