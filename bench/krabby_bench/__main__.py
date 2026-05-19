"""krabby-bench CLI entry point."""
from __future__ import annotations

import argparse
from pathlib import Path

from krabby_bench._config import load_config
from krabby_bench.watchdog import run


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="krabby-bench",
        description="Bench watchdog: polls ECR for new digests and runs smoke tests.",
    )
    parser.add_argument(
        "--config", metavar="PATH",
        default="/etc/krabby-bench/config.toml",
        help="Path to config.toml (default: /etc/krabby-bench/config.toml)",
    )
    args = parser.parse_args()
    config = load_config(Path(args.config))
    run(config)


if __name__ == "__main__":
    main()
