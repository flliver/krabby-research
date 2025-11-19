#!/usr/bin/env python3

import argparse
import glob
import json
import os
import shutil
import sys
from datetime import datetime


def _latest_path_in(dir_path: str, pattern: str) -> str | None:
    matches = glob.glob(os.path.join(dir_path, pattern))
    if not matches:
        return None
    return max(matches, key=os.path.getmtime)


def _find_latest_run_dir(experiment_dir: str) -> str | None:
    # run directories look like YYYY-MM-DD_HH-MM-SS
    return _latest_path_in(experiment_dir, "20*_*")


def _select_checkpoint(run_dir: str) -> str | None:
    # Prefer latest training checkpoint (model_*.pt)
    model_ckpt = _latest_path_in(run_dir, "model_*.pt")
    if model_ckpt:
        return model_ckpt
    # Fallback to any exported policy (e.g., exported_*/policy.pt or policy.pt)
    exported_policy = _latest_path_in(run_dir, "exported*/policy.pt")
    if exported_policy:
        return exported_policy
    policy_pt = os.path.join(run_dir, "policy.pt")
    if os.path.isfile(policy_pt):
        return policy_pt
    return None


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _write_metadata(dst_json: str, src_path: str, experiment_name: str, run_dir: str) -> None:
    meta = {
        "experiment": experiment_name,
        "source_path": os.path.abspath(src_path),
        "run_dir": os.path.abspath(run_dir),
        "published_at": datetime.utcnow().isoformat() + "Z",
        "size_bytes": os.path.getsize(src_path) if os.path.exists(src_path) else None,
    }
    with open(dst_json, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def publish_latest(weights_root: str, assets_dir: str, experiments: list[str] | None, dry_run: bool, force: bool) -> list[tuple[str, str]]:
    """Publish latest checkpoints for given experiments.

    Returns list of (experiment_name, published_path)
    """
    published: list[tuple[str, str]] = []

    # Logs root like: <project>/logs/rsl_rl/<experiment_name>/<run_timestamp>/model_*.pt
    logs_root = os.path.abspath(weights_root)
    if not os.path.isdir(logs_root):
        raise FileNotFoundError(f"Logs root not found: {logs_root}")

    exp_dirs = []
    if experiments:
        for exp in experiments:
            exp_dir = os.path.join(logs_root, exp)
            if os.path.isdir(exp_dir):
                exp_dirs.append(exp_dir)
            else:
                print(f"[WARN] Experiment directory not found: {exp_dir}")
    else:
        for name in os.listdir(logs_root):
            path = os.path.join(logs_root, name)
            if os.path.isdir(path):
                exp_dirs.append(path)

    if not exp_dirs:
        print(f"[INFO] No experiments found in {logs_root}")
        return published

    _ensure_dir(assets_dir)

    for exp_dir in sorted(exp_dirs):
        experiment_name = os.path.basename(exp_dir)
        run_dir = _find_latest_run_dir(exp_dir)
        if not run_dir:
            print(f"[INFO] No runs found for {experiment_name}")
            continue
        ckpt = _select_checkpoint(run_dir)
        if not ckpt:
            print(f"[INFO] No checkpoints found in run {run_dir}")
            continue

        # Destination filename: <experiment_name>.pt
        dst_path = os.path.join(assets_dir, f"{experiment_name}.pt")
        dst_meta = os.path.join(assets_dir, f"{experiment_name}.json")

        if os.path.exists(dst_path) and not force:
            print(f"[SKIP] {dst_path} exists. Use --force to overwrite.")
            continue

        print(f"[PUBLISH] {experiment_name}: {ckpt} -> {dst_path}")
        if not dry_run:
            shutil.copy2(ckpt, dst_path)
            _write_metadata(dst_meta, ckpt, experiment_name, run_dir)
        published.append((experiment_name, os.path.abspath(dst_path)))

    return published


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Publish latest training weights into assets for version control")
    parser.add_argument(
        "--workflow",
        default="rsl_rl",
        help="Workflow name under logs/ (default: rsl_rl)",
    )
    parser.add_argument(
        "--experiments",
        nargs="*",
        help="Optional list of experiment names to publish (defaults to all under logs/<workflow>/)",
    )
    parser.add_argument(
        "--assets-dir",
        default=None,
        help="Target directory to place published weights (default: parkour/assets/weights)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done without copying files")
    parser.add_argument("--force", action="store_true", help="Overwrite existing files in assets directory")

    args = parser.parse_args(argv)

    # Resolve project root as the parent of this scripts directory
    scripts_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(scripts_dir, ".."))

    logs_root = os.path.join(project_root, "logs", args.workflow)
    assets_dir = (
        os.path.abspath(args.assets_dir)
        if args.assets_dir
        else os.path.join(project_root, "assets", "weights")
    )

    try:
        published = publish_latest(logs_root, assets_dir, args.experiments, args.dry_run, args.force)
    except FileNotFoundError as e:
        print(str(e))
        return 1

    if not published:
        print("No weights published.")
        return 2

    print("Published:")
    for name, path in published:
        print(f"- {name}: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
