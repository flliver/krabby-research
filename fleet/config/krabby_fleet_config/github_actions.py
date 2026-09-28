"""Export committed fleet.toml values into GitHub Actions GITHUB_ENV."""
from __future__ import annotations

import os
import sys

from krabby_fleet_config.loader import load_fleet_config


def main() -> None:
    cfg = load_fleet_config()
    if not cfg.ci_github_actions_role_arn:
        print(
            "[ci].github_actions_role_arn is required in fleet.toml "
            "(FleetServiceStack FleetGitHubActionsRoleArn output)",
            file=sys.stderr,
        )
        sys.exit(1)
    github_env = os.environ.get("GITHUB_ENV")
    if not github_env:
        print("GITHUB_ENV is not set", file=sys.stderr)
        sys.exit(1)
    with open(github_env, "a", encoding="utf-8") as handle:
        for key, value in cfg.as_env().items():
            handle.write(f"{key}={value}\n")


if __name__ == "__main__":
    main()
