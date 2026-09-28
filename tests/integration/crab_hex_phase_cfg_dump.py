"""Dump crab-hex env / runner configs to JSON inside Isaac Sim (config-identity fixture).

Run from ``parkour/`` with the Isaac venv python, headless::

    OMNI_KIT_ACCEPT_EULA=yes KRABBY_PHASE=2c \\   # KRABBY_PLANT optional: the main asset is A15+B
        python tests/integration/crab_hex_phase_cfg_dump.py --headless \\
        --task Isaac-Crab-Hex-Teacher-v0 --task Isaac-Crab-Hex-Flat-Walk-v0 --out /tmp/cfgs.json

The process environment is the fixture: ``KRABBY_PHASE`` / ``KRABBY_PLANT`` (or a raw ``KRABBY_*``
stack) must be set before launch because the scene cfg reads the USD path at import time and
``activate_phase()`` runs on the first import of the crab-hex config package. Output::

    {"<task>": {"env": <class_to_dict(env_cfg)>, "agent": <class_to_dict(agent_cfg)>}, ...,
     "_environ": {<KRABBY_* seen by the process>}}

Used by ``tests/integration/test_crab_hex_phase_configs.py``; no simulation is stepped.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Dump crab-hex configs inside Isaac Sim.")
parser.add_argument("--task", action="append", required=True, help="Gym task id (repeatable).")
parser.add_argument("--out", required=True, help="Output JSON path.")
parser.add_argument("--set-after-import", action="append", default=[], metavar="KEY=VALUE",
                    help="Environment variables to set AFTER the task package import but before the cfgs are built (mimics a manifest env block).")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

PARKOUR_TASKS_SRC = Path(__file__).resolve().parents[2] / "parkour" / "parkour_tasks"
if str(PARKOUR_TASKS_SRC) not in sys.path:
    sys.path.insert(0, str(PARKOUR_TASKS_SRC))

import gymnasium as gym  # noqa: E402
from isaaclab.utils.dict import class_to_dict  # noqa: E402

import parkour_tasks.crab_hex_forward_task  # noqa: E402,F401  (registers the tasks; runs activate_phase)


def main() -> int:
    for kv in args.set_after_import:
        k, _, v = kv.partition("=")
        os.environ[k] = v
    out: dict = {"_environ": {k: v for k, v in os.environ.items() if k.startswith("KRABBY_")}}
    for task in args.task:
        spec = gym.spec(task)
        env_cfg = spec.kwargs["env_cfg_entry_point"]()
        agent_cfg = spec.kwargs["rsl_rl_cfg_entry_point"]()
        out[task] = {"env": class_to_dict(env_cfg), "agent": class_to_dict(agent_cfg)}
    Path(args.out).write_text(json.dumps(out, indent=1, sort_keys=True, default=str))
    print(f"[dump] wrote {args.out}: {', '.join(args.task)}")
    return 0


if __name__ == "__main__":
    rc = 1
    try:
        rc = main()
    except Exception:  # noqa: BLE001 -- print before the Kit teardown can hang
        import traceback

        traceback.print_exc()
    finally:
        simulation_app.close()
    sys.exit(rc)
