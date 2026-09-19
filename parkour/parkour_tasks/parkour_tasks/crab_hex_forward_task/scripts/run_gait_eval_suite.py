#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""Run the crab-hex gait eval scenario set, one Isaac process per scenario.

stdlib only -- runs under plain ``python3``, no Isaac import. Each scenario gets its own
``isaaclab.sh -p eval_crab_hex_gait.py`` invocation, **serialized**: only one Isaac Sim process fits
on a 16 GB GPU (teacher ~4-5 GB, student depth rendering peaks near 10 GB), and warp's mesh cache is
process-global, so re-creating envs in one process is not worth the risk.

    python3 run_gait_eval_suite.py --list
    python3 run_gait_eval_suite.py --dry-run
    python3 run_gait_eval_suite.py --scenario teacher_2b2_forward
    python3 run_gait_eval_suite.py --repeat 2      # determinism / noise-floor check
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_TASK_DIR = _SCRIPT_DIR.parent
_PARKOUR_ROOT = _SCRIPT_DIR.parents[3]
_DEFAULT_MANIFEST = _TASK_DIR / "experiments" / "eval" / "scenarios_v1.yaml"


def _scenario_ids(manifest: Path) -> list[str]:
    """Scenario ids without requiring PyYAML (the driver must stay stdlib-only)."""
    text = manifest.read_text()
    return re.findall(r"^\s*-\s+id:\s*(\S+)\s*$", text, flags=re.MULTILINE)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the crab-hex gait eval scenario set.")
    parser.add_argument("--manifest", default=str(_DEFAULT_MANIFEST))
    parser.add_argument("--scenario", action="append", default=None, help="Repeatable; default all.")
    parser.add_argument("--isaaclab", default=os.environ.get("ISAACLAB_PATH", str(Path.home() / "krabby" / "IsaacLab")))
    parser.add_argument("--repeat", type=int, default=1, help="Runs per scenario (use 2 for a noise-floor check).")
    parser.add_argument("--seed-offset", type=int, default=0, help="Added to each repeat index for --env-seed.")
    parser.add_argument("--extra", default="", help="Extra args appended verbatim to the harness.")
    parser.add_argument(
        "--plant",
        default=None,
        help="Named plant passed to the harness (--plant); e.g. legacy_golden for pre-A15+B heads trained "
        "after the 2026-08-20 rebuild (with --manifest scenarios_v2.yaml or scenarios_morph.yaml). "
        "The v1 baselines predate that rebuild and no longer load.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args()

    manifest = Path(args.manifest)
    if not manifest.exists():
        print(f"manifest not found: {manifest}", file=sys.stderr)
        return 2
    ids = _scenario_ids(manifest)
    if args.list:
        for scenario_id in ids:
            print(scenario_id)
        return 0

    selected = args.scenario or ids
    unknown = [s for s in selected if s not in ids]
    if unknown:
        print(f"unknown scenario(s): {unknown}\navailable: {ids}", file=sys.stderr)
        return 2

    isaaclab_sh = Path(args.isaaclab) / "isaaclab.sh"
    if not isaaclab_sh.exists() and not args.dry_run:
        print(f"isaaclab.sh not found at {isaaclab_sh} (set --isaaclab or ISAACLAB_PATH)", file=sys.stderr)
        return 2

    harness = _SCRIPT_DIR / "eval_crab_hex_gait.py"
    failures: list[str] = []
    for scenario_id in selected:
        for rep in range(args.repeat):
            cmd = [
                str(isaaclab_sh),
                "-p",
                str(harness),
                "--headless",
                "--manifest",
                str(manifest),
                "--scenario",
                scenario_id,
            ]
            if args.repeat > 1:
                # Same scenario, different seed -> the spread across repeats is the gate's
                # resolution limit. A threshold tighter than that spread is not a gate.
                cmd += ["--env-seed", str(1 + args.seed_offset + rep)]
            if args.plant:
                cmd += ["--plant", args.plant]
            if args.extra:
                cmd += args.extra.split()

            label = f"{scenario_id} (run {rep + 1}/{args.repeat})"
            print(f"\n=== {label} ===\n$ {' '.join(cmd)}", flush=True)
            if args.dry_run:
                continue
            started = time.time()
            result = subprocess.run(cmd, cwd=str(_PARKOUR_ROOT), check=False)
            elapsed = time.time() - started
            if result.returncode != 0:
                print(f"[FAIL] {label} exited {result.returncode} after {elapsed:.0f}s", file=sys.stderr)
                failures.append(label)
                if not args.continue_on_error:
                    return result.returncode
            else:
                print(f"[OK] {label} in {elapsed:.0f}s", flush=True)

    if failures:
        print(f"\n{len(failures)} scenario run(s) failed: {failures}", file=sys.stderr)
        return 1
    print("\nAll scenario runs completed.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
