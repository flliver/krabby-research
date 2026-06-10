#!/usr/bin/env python3
"""run_transform.py — pipeline runner v1 (STO-SCN-039).

Execute ONE transform of a scene-store run from its specification.json and
emit a *measured* results.json: HUG-KRB-002's "configuration-driven data
pipelines" execution layer, v1.

    spec in  -> container run (canonical flags) -> data/ artifacts
             -> data/run_logs/{train.log, nvidia-smi.csv}
             -> results.json  (provenance: measured; sha256'd outputs)

Usage (on the host that holds the store clone + GPU, e.g. tbeeprz):

    # re-execute an existing transform IN PLACE (refuses if data/ has results)
    python3 run_transform.py <scene>/pipeline-<p>/run-<r>/transform-NN-<slug> \
        [--store ~/krabby/scenes] [--force]

    # clone an existing transform's spec into a NEW run, then execute it
    python3 run_transform.py <...>/transform-NN-<slug> --as <new-run-slug>

Design notes (v1 scope — see STO-SCN-040 for the registry generalization):
  - transform registry is the in-file TRANSFORMS dict; `matcha` only.
  - MAtCha source is bind-mounted from the host snapshot until STO-SCN-038
    ships a self-contained image; recorded honestly in results.environment.
  - fast-fail CUDA preamble; fast-forward of partial state is never attempted.
  - parameters the registry can't express raise instead of being dropped
    (T-002: never silently run something other than the spec).
"""
from __future__ import annotations
import argparse, datetime, hashlib, json, os, re, shutil, signal, socket
import subprocess, sys, time
from pathlib import Path

CANONICAL_DOCKER_FLAGS = [
    "--gpus", "all", "--shm-size=8g", "--ipc=host",
    "--ulimit", "memlock=-1", "--ulimit", "stack=67108864",
]

# ── transform registry (v1: in-file; STO-SCN-040 externalizes) ──────────────
TRANSFORMS = {
    "matcha": {
        "image": "krabby-matcha:latest",
        # host path bind-mounted over the image's (empty) /opt/MAtCha until
        # STO-SCN-038 makes the image self-contained.
        "source_mount": ("~/scratch/MAtCha", "/opt/MAtCha"),
        "container_setup": (
            "source /opt/matcha/bin/activate && "
            "export PYTHONPATH=/opt/MAtCha:/opt/MAtCha/mast3r:/opt/MAtCha/mast3r/dust3r:"
            "/opt/MAtCha/2d-gaussian-splatting:/opt/MAtCha/2d-gaussian-splatting/submodules/simple-knn && "
            "export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && cd /opt/MAtCha"
        ),
        # parameters the v1 template knows how to express:
        "supported_params": {
            "alignment_config", "encoder", "sfm_config", "n_images",
            # informational / not flags:
            "image", "image_id", "git_sha", "source_snapshot", "tool_args_raw",
            "image_resolution_long_edge", "chart_resolutions_active",
            "extra_flags", "dense_regul", "dense_pruning",
        },
        # values that demand flags v1 has NOT verified → hard error (T-002).
        # dense_regul verified 2026-06-09 against ~/scratch/MAtCha/train.py:24
        # ('default'|'strong'|'weak'|'none') and moved to build_command.
        "unverified_nondefaults": {"dense_pruning": "default"},
    },
}


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def now_iso() -> str:
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


def run(cmd, **kw):
    return subprocess.run(cmd, text=True, capture_output=True, **kw)


def measure_environment(image: str) -> dict:
    env = {"os": "unknown", "gpu": "unknown", "nvidia_driver": "unknown",
           "cuda": "unknown", "container": {"image": image, "tag": "unknown", "digest": "unknown"},
           "software": {}}
    r = run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
             "--format=csv,noheader"])
    if r.returncode == 0 and r.stdout.strip():
        name, drv, mem = [x.strip() for x in r.stdout.strip().split(",")[:3]]
        gib = round(int(re.sub(r"\D", "", mem)) / 1024)
        env["gpu"] = f"{name} / {gib} GB"
        env["nvidia_driver"] = drv
    r = run(["bash", "-c", ". /etc/os-release 2>/dev/null && echo $PRETTY_NAME"])
    if r.stdout.strip():
        env["os"] = r.stdout.strip()
    r = run(["docker", "images", "--digests",
             "--format", "{{.Repository}}:{{.Tag}} {{.Digest}} {{.ID}}"])
    for line in r.stdout.splitlines():
        parts = line.split()
        if parts and parts[0] == image:
            env["container"]["digest"] = (
                parts[1] if parts[1] != "<none>" else f"none (local id {parts[2]})")
            env["container"]["tag"] = image.rsplit(":", 1)[-1]
    # cuda version as reported by the image itself
    r = run(["docker", "run", "--rm", "--entrypoint", "bash", image, "-c",
             "cat /usr/local/cuda/version.json 2>/dev/null | python3 -c \"import sys,json;print(json.load(sys.stdin)['cuda']['version'])\" 2>/dev/null || nvcc --version 2>/dev/null | grep -oE 'release [0-9.]+' | cut -d' ' -f2"])
    if r.returncode == 0 and r.stdout.strip():
        env["cuda"] = r.stdout.strip().splitlines()[-1]
    return env


def build_command(pipeline: str, params: dict, frames_ct: str, out_ct: str) -> str:
    reg = TRANSFORMS[pipeline]
    unknown = {k for k in params
               if k not in reg["supported_params"] and not k.startswith("gs_")}
    if unknown:
        raise SystemExit(f"ERROR: spec parameters not supported by runner v1: {sorted(unknown)} "
                         f"(extend the registry — STO-SCN-040 — rather than dropping them)")
    for k, dflt in reg["unverified_nondefaults"].items():
        if params.get(k, dflt) != dflt:
            raise SystemExit(f"ERROR: {k}={params[k]!r} requires a flag mapping v1 has not "
                             f"verified against the tool. Verify + extend TRANSFORMS first (T-002).")
    if params.get("extra_flags"):
        raise SystemExit("ERROR: extra_flags non-empty; v1 refuses unvetted passthrough.")
    cmd = (f"python train.py -s {frames_ct} -o {out_ct}"
           f" --sfm_config {params.get('sfm_config', 'unposed')}")
    if params.get("n_images"):
        cmd += f" --n_images {params['n_images']}"
    if params.get("alignment_config", "default") != "default":
        cmd += f" --alignment_config {params['alignment_config']}"
    if params.get("dense_regul", "default") != "default":
        if params["dense_regul"] not in ("strong", "weak", "none"):
            raise SystemExit(f"ERROR: dense_regul={params['dense_regul']!r} not a "
                             f"verified train.py choice (default|strong|weak|none)")
        cmd += f" --dense_regul {params['dense_regul']}"
    cmd += (" --depthanythingv2_checkpoint_dir /opt/MAtCha/Depth-Anything-V2/checkpoints"
            f" --depthanything_encoder {params.get('encoder', 'vitl')}")
    return cmd


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("transform", help="<scene>/pipeline-<p>/run-<r>/transform-NN-<slug> (store-relative)")
    ap.add_argument("--store", default="~/krabby/scenes")
    ap.add_argument("--as", dest="as_run", default=None,
                    help="clone spec into a new run-<slug> and execute there")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    store = Path(os.path.expanduser(args.store)).resolve()
    tdir = store / args.transform
    spec_path = tdir / "specification.json"
    if not spec_path.is_file():
        sys.exit(f"ERROR: no specification.json at {spec_path}")

    if args.as_run:
        run_dir = tdir.parent
        new_run = run_dir.parent / f"run-{args.as_run}"
        new_t = new_run / tdir.name
        if new_run.exists():
            sys.exit(f"ERROR: {new_run} already exists")
        new_t.mkdir(parents=True)
        spec = json.loads(spec_path.read_text())
        spec["run"] = args.as_run
        (new_t / "specification.json").write_text(json.dumps(spec, indent=2) + "\n")
        rj = run_dir / "run.json"
        run_doc = json.loads(rj.read_text()) if rj.is_file() else {
            "schema_version": "1", "pipeline": spec.get("pipeline"), "params": {}}
        run_doc["run"] = args.as_run
        run_doc["promoted"] = False
        run_doc["notes"] = (f"Created by run_transform.py --as {args.as_run} from "
                            f"{args.transform} on {now_iso()}")
        (new_run / "run.json").write_text(json.dumps(run_doc, indent=2) + "\n")
        tdir, spec_path = new_t, new_t / "specification.json"
        print(f"new run scaffolded: {new_run.relative_to(store)}")

    spec = json.loads(spec_path.read_text())
    pipeline = spec["pipeline"]
    if pipeline not in TRANSFORMS:
        sys.exit(f"ERROR: pipeline {pipeline!r} not in runner v1 registry {list(TRANSFORMS)}")
    reg = TRANSFORMS[pipeline]
    params = spec.get("parameters", {})
    image = params.get("image", reg["image"])

    data = tdir / "data"
    results_path = tdir / "results.json"
    if results_path.is_file() and not args.force:
        sys.exit(f"ERROR: {results_path} exists — runs are immutable; use --as <new-run> (or --force).")
    data.mkdir(exist_ok=True)
    logs = data / "run_logs"
    logs.mkdir(exist_ok=True)

    # inputs: scene-relative → container paths under /scene
    inputs = [i for i in spec.get("inputs", []) if not i.startswith("(")]
    if len(inputs) != 1:
        sys.exit(f"ERROR: v1 expects exactly 1 machine-readable input, got {inputs!r}")
    scene_dir = tdir.parents[2]
    frames_host = scene_dir / inputs[0]
    if not frames_host.is_dir():
        sys.exit(f"ERROR: input dir missing: {frames_host}")
    # LFS-pointer guard: a 131-byte "version https://git-lfs..." file is not an
    # image (observed 2026-06-09 — un-smudged clone). Refuse before burning GPU.
    for f in sorted(frames_host.iterdir()):
        if f.is_file() and f.stat().st_size < 1024:
            head = f.read_bytes()[:40]
            if head.startswith(b"version https://git-lfs"):
                sys.exit(f"ERROR: input {f.name} is an un-smudged LFS pointer — "
                         f"run `git lfs pull` in the store first.")
    frames_ct = f"/scene/{inputs[0]}"
    out_ct = f"/scene/{tdir.relative_to(scene_dir)}/data"

    tool_cmd = build_command(pipeline, params, frames_ct, out_ct)
    full = (f"{reg['container_setup']} && "
            f"python -c \"import torch; assert torch.cuda.is_available(), 'CUDA unavailable'\" && "
            f"{tool_cmd}")
    src_host, src_ct = reg["source_mount"]
    src_host = os.path.expanduser(src_host)

    docker_cmd = (["docker", "run", "--rm", "--name", f"runner-{tdir.parent.name}"]
                  + CANONICAL_DOCKER_FLAGS
                  + ["-v", f"{scene_dir}:/scene", "-v", f"{src_host}:{src_ct}",
                     "--entrypoint", "bash", image, "-lc", full])

    print(f"runner: {pipeline} | image {image}\n  cmd: {tool_cmd}")
    env = measure_environment(image)
    env["software"][pipeline] = (f"host snapshot {src_host} (pre-STO-SCN-038; "
                                 f"see archives + fingerprint in STO-SCN-038)")

    # VRAM sampler (5s, schema matches the historical nvidia-smi.csv)
    csv = logs / "nvidia-smi.csv"
    sampler = subprocess.Popen(
        ["bash", "-c",
         f"echo ts,used_mib,total_mib > {csv}; while true; do "
         f"echo \"$(date +%Y-%m-%dT%H:%M:%S%z),$(nvidia-smi --query-gpu=memory.used,memory.total "
         f"--format=csv,noheader,nounits | tr -d ' ' | tr ',' ',')\" >> {csv}; sleep 5; done"])
    started = now_iso()
    t0 = time.time()
    nanny = shutil.which("nanny-progress") or "/usr/local/bin/nanny-progress"
    have_nanny = os.path.exists(nanny)
    if have_nanny:
        subprocess.run([nanny, "set", "1/1", "0"], check=False)
    try:
        with open(logs / "train.log", "w") as lf:
            rc = subprocess.run(docker_cmd, stdout=lf, stderr=subprocess.STDOUT).returncode
    finally:
        sampler.send_signal(signal.SIGTERM)
        if have_nanny:
            subprocess.run([nanny, "clear"], check=False)
    duration = round(time.time() - t0)
    finished = now_iso()

    peak = 0
    for line in csv.read_text().splitlines()[1:]:
        try:
            peak = max(peak, int(line.split(",")[1]))
        except (IndexError, ValueError):
            pass

    # canonical outputs: what the registry expects the tool to produce.
    # HARD GATE: the tool (MAtCha train.py) chains stages via os.system and
    # exits 0 even when a stage crashes (observed 2026-06-09: dtu repro "ran"
    # 11s on LFS-pointer inputs, rc=0, no outputs). Missing expected outputs
    # => failed, regardless of rc.
    expected = ["data/tetra_meshes/tetra_mesh_binary_search_7.ply",
                "data/mast3r_sfm/cameras.json", "data/mast3r_sfm/points.ply"]
    missing = [e for e in expected if not (tdir / e).is_file()]
    if missing and rc == 0:
        print(f"WARNING: tool rc=0 but expected outputs missing: {missing} — marking failed")
        rc = 97  # synthetic: expected-outputs gate
    outputs = []
    for rel in expected + ["data/run_logs/train.log", "data/run_logs/nvidia-smi.csv",
                           "data/free_gaussians/cfg_args"]:
        p = tdir / rel
        if p.is_file():
            o = {"path": rel, "bytes": p.stat().st_size}
            if p.stat().st_size < (1 << 31):
                o["sha256"] = sha256(p)
            outputs.append(o)

    results = {
        "schema_version": "1", "transform": spec["transform"],
        "status": "success" if rc == 0 else "failed",
        "provenance": "measured", "started": started, "finished": finished,
        "duration_s": duration, "host": socket.gethostname(),
        "peak_vram_mib": peak, "environment": env, "outputs": outputs,
        "runner": {"name": "run_transform.py", "version": "v1 (STO-SCN-039)",
                    "command": tool_cmd},
    }
    results_path.write_text(json.dumps(results, indent=2) + "\n")
    print(f"results.json written: status={results['status']} duration={duration}s "
          f"peak_vram={peak}MiB outputs={len(outputs)}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
