#!/usr/bin/env python3
"""Config dumps (Kit, no simulation stepped) of every target in a dump-targets file, one Kit process at a time.

A wrapped copy of the body-cmd campaign's ``scripts/bc_dump_cfgs.py`` (that file is not edited), for the M1a payload
inertness proof (PLAN step 3: "13 knob-off dumps byte-identical"). Differences from the original:

* ``--root`` picks the checkout the dump runs from (default: the main checkout). The dump script of THAT checkout is
  run with ``cwd=<root>/parkour`` and ``PYTHONPATH=<root>/parkour/parkour_tasks:<root>/parkour``, so a detached
  worktree at HEAD imports its own code, not the editable install's main checkout. Before/after dumps taken in the
  SAME worktree differ only by the patch applied between them (paths in the dumps are identical).
* Before every launch it waits while any Isaac python of the campaign's busy pattern is running (one Kit process at a
  time, machine-wide), then launches under ``systemd-run --user --scope -p MemoryMax=48G -p MemorySwapMax=0`` and
  ``timeout --signal=TERM`` (never SIGKILL a Kit process).
* The default targets file is the body-cmd campaign's ``dump_targets.json`` (read only).

    t4_dump_cfgs.py --out DIR [--root CHECKOUT] [--only NAME ...] [--set KEY=VALUE ...] [--targets JSON]
                    [--task-override NAME=TASK ...]

``--task-override`` dumps a target through another gym id (e.g. the Play cfg of a teacher target) without editing the
targets file. Each target is ``{phase, task, overlay}``: ``KRABBY_OMNI_PHASE=<phase>`` + ``KRABBY_PLANT=A15+B`` + the
overlay env, exactly the env the candidates driver composes for a stage; every ``--set`` pair is added on top.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

R = Path("/home/nickmagus/krabby/krabby-research")
PY = "/home/nickmagus/krabby/isaac_venv/bin/python"
BC_TARGETS = R / "parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task/experiments/2026-10-15_body_cmd/dump_targets.json"
MEM_MAX = "48G"
TIMEOUT_S = 900
#: The campaign's machine-wide busy pattern (PLAN step 3 brief): any of these Isaac pythons running = wait.
BUSY_RE = re.compile(
    r"^[^ ]*[i]saac_venv/bin/python3?( -[^ ]+)* [^ ]*(run_phases|run_candidates|eval_crab_hex_gait|train|play|"
    r"play_stream_video|record_categories|crab_hex_phase_cfg_dump|sh_[A-Za-z0-9_]+|sn_[A-Za-z0-9_]+|bc_dump_cfgs)\.py"
)


def busy() -> list[str]:
    out = subprocess.run(["ps", "-eo", "args"], capture_output=True, text=True).stdout.splitlines()
    return [line for line in out if BUSY_RE.search(line)]


def wait_idle(poll_s: float = 20.0) -> None:
    first = True
    while True:
        hits = busy()
        if not hits:
            return
        if first:
            print(f"[dump] waiting: {len(hits)} Isaac process(es) running, e.g. {hits[0][:160]!r}", flush=True)
            first = False
        time.sleep(poll_s)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--root", default=str(R))
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--targets", default=str(BC_TARGETS))
    ap.add_argument("--task-override", action="append", default=[])
    a = ap.parse_args()
    root = Path(a.root).resolve()
    dump = root / "tests/integration/crab_hex_phase_cfg_dump.py"
    if not dump.is_file():
        print(f"[dump] no dump script at {dump}", flush=True)
        return 2
    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    targets = json.loads(Path(a.targets).read_text())
    extra = dict(kv.split("=", 1) for kv in a.set)
    task_over = dict(kv.split("=", 1) for kv in a.task_override)
    base_env = {k: v for k, v in os.environ.items() if not k.startswith("KRABBY_")}
    base_env.update({
        "OMNI_KIT_ACCEPT_EULA": "yes",
        "PYTHONUNBUFFERED": "1",
        "PYTHONPATH": f"{root}/parkour/parkour_tasks:{root}/parkour",
    })
    fails = []
    for name, t in sorted(targets.items()):
        if a.only and name not in a.only:
            continue
        task = task_over.get(name, t["task"])
        env = dict(base_env)
        env.update({"KRABBY_OMNI_PHASE": t["phase"], "KRABBY_PLANT": "A15+B"})
        env.update({k: str(v) for k, v in t["overlay"].items()})
        env.update(extra)
        js, log = out / f"{name}.json", out / f"{name}.log"
        cmd = ["systemd-run", "--user", "--scope", "-q", "-p", f"MemoryMax={MEM_MAX}", "-p", "MemorySwapMax=0",
               "timeout", "--signal=TERM", str(TIMEOUT_S), PY, "-u", str(dump),
               "--headless", "--task", task, "--out", str(js)]
        wait_idle()
        print(f"[dump] {name}: {t['phase']} {task} + {len(t['overlay'])} overlay keys + {len(extra)} extra "
              f"(root {root})", flush=True)
        t0 = time.time()
        with open(log, "w") as fh:
            rc = subprocess.run(cmd, cwd=str(root / "parkour"), env=env, stdout=fh, stderr=subprocess.STDOUT).returncode
        if rc != 0 or not js.is_file():
            fails.append(name)
            print(f"[dump] {name}: FAILED rc={rc} ({time.time() - t0:.0f} s)", flush=True)
        else:
            print(f"[dump] {name}: ok ({time.time() - t0:.0f} s)", flush=True)
    print(f"T4DUMP END fails={fails}", flush=True)
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
