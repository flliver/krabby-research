#!/usr/bin/env python3
"""Config dumps (Kit, no simulation stepped) of every target in ``dump_targets.json``, one process at a time.

The step-1 inertness fixture: run once at HEAD before the body-command code lands (``--out .../before``) and once
after (``--out .../after``), then ``bc_dump_compare.py`` requires the knob-off dumps to be identical. Each target is
``{phase, task, overlay}``: ``KRABBY_OMNI_PHASE=<phase>`` + ``KRABBY_PLANT=A15+B`` + the overlay env, exactly the env
the candidates driver composes for a stage. Extra ``--set KEY=VALUE`` pairs are added to every target (used for the
knob-ON dumps of the ``_b`` presets).

    bc_dump_cfgs.py --out DIR [--only NAME ...] [--set KEY=VALUE ...] [--targets JSON]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

R = Path("/home/nickmagus/krabby/krabby-research")
PY = "/home/nickmagus/krabby/isaac_venv/bin/python"
DUMP = R / "tests/integration/crab_hex_phase_cfg_dump.py"
HERE = Path(__file__).resolve().parent
TARGETS = HERE.parent / "dump_targets.json"
MEM_MAX = "24G"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--targets", default=str(TARGETS))
    a = ap.parse_args()
    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    targets = json.loads(Path(a.targets).read_text())
    extra = dict(kv.split("=", 1) for kv in a.set)
    base_env = {k: v for k, v in os.environ.items() if not k.startswith("KRABBY_")}
    base_env.update({"OMNI_KIT_ACCEPT_EULA": "yes", "PYTHONUNBUFFERED": "1"})
    fails = []
    for name, t in sorted(targets.items()):
        if a.only and name not in a.only:
            continue
        env = dict(base_env)
        env.update({"KRABBY_OMNI_PHASE": t["phase"], "KRABBY_PLANT": "A15+B"})
        env.update({k: str(v) for k, v in t["overlay"].items()})
        env.update(extra)
        js, log = out / f"{name}.json", out / f"{name}.log"
        cmd = ["systemd-run", "--user", "--scope", "-q", "-p", f"MemoryMax={MEM_MAX}", PY, "-u", str(DUMP),
               "--headless", "--task", t["task"], "--out", str(js)]
        print(f"[dump] {name}: {t['phase']} + {len(t['overlay'])} overlay keys + {len(extra)} extra", flush=True)
        with open(log, "w") as fh:
            rc = subprocess.run(cmd, cwd=str(R / "parkour"), env=env, stdout=fh, stderr=subprocess.STDOUT).returncode
        if rc != 0 or not js.is_file():
            fails.append(name)
            print(f"[dump] {name}: FAILED rc={rc}", flush=True)
    print(f"BCDUMP END fails={fails}", flush=True)
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
