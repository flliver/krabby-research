"""krabby-fleet CLI entry point."""
from __future__ import annotations

import argparse
import sys


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="krabby-fleet",
        description="Operator CLI for the Krabby fleet.",
    )
    sub = parser.add_subparsers(dest="command", metavar="<command>")

    p_ssh = sub.add_parser("ssh", help="Open a Secure Tunnel and SSH into a robot")
    p_ssh.add_argument("robot", metavar="ROBOT", help="Thing name of the robot to SSH into")
    p_ssh.add_argument("--user", "-u", metavar="USER", help="SSH user (default: from config)")

    p_teleop = sub.add_parser("teleop", help="Open the portal teleop view for a robot")
    p_teleop.add_argument("robot", metavar="ROBOT", help="Thing name of the robot to teleop")

    sub.add_parser("list", help="List enrolled robots with online/last-seen + telemetry summary")
    sub.add_parser("devices", help="Alias for `list`")

    args = parser.parse_args()

    if args.command == "ssh":
        from krabby_fleet_cli.ssh import cmd_ssh
        cmd_ssh(args.robot, user=args.user)
    elif args.command == "teleop":
        from krabby_fleet_cli.teleop import cmd_teleop
        cmd_teleop(args.robot)
    elif args.command in ("list", "devices"):
        from krabby_fleet_cli.list import cmd_list
        cmd_list()
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
