"""krabby-bench CLI entry point."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from krabby_bench._config import CONFIG_PATH, EcrConfig, SmokeConfig, load_config
from krabby_bench.watchdog import run


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="krabby-bench",
        description="Bench watchdog: polls ECR for new digests and runs smoke tests.",
    )
    parser.add_argument(
        "--config", metavar="PATH",
        default=str(CONFIG_PATH),
        help=f"Path to config.toml (default: {CONFIG_PATH})",
    )
    subparsers = parser.add_subparsers(dest="command")

    install_p = subparsers.add_parser(
        "install",
        help="Bootstrap the systemd service (must run as root). "
             "Reads credentials from BENCH_SMTP_* and BENCH_GITHUB_TOKEN env vars.",
    )
    install_p.add_argument("--ecr-tag", default=EcrConfig.tag, metavar="TAG")
    install_p.add_argument("--firmware-channel", default=SmokeConfig.firmware_channel, metavar="CHANNEL")
    install_p.add_argument("--mode", default="both", choices=["email", "github", "both"],
                           help="Alert mode (default: both)")
    install_p.add_argument("--github-repo", default=os.environ.get("BENCH_GITHUB_REPO", ""),
                           metavar="OWNER/REPO")

    args = parser.parse_args()

    if args.command == "install":
        from krabby_bench._install import install
        install(
            ecr_tag=args.ecr_tag,
            firmware_channel=args.firmware_channel,
            mode=args.mode,
            github_repo=args.github_repo,
        )
        return

    config = load_config(Path(args.config))
    run(config)


if __name__ == "__main__":
    main()
