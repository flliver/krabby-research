"""Interactive M16 Task 4 bench validation."""
import argparse
import sys
from firmware.bench_tests import task4

DEFAULT_RESULTS = "bench-results-{suite}.json"

def _cmd_run(args: argparse.Namespace) -> int:
    results_path = args.results or DEFAULT_RESULTS.format(suite=args.suite)
    if args.suite == "task4":
        if args.only:
            print("task4 is one safety-ordered stateful session; --only is not supported",
                  file=sys.stderr)
            return 2
        if args.yes:
            print("task4 requires physical confirmations; --yes is not supported",
                  file=sys.stderr)
            return 2
        if not args.remote:
            print("--remote USER@HOST is required for the production Orin phase",
                  file=sys.stderr)
            return 2
        return task4.run(args.port, args.remote, results_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    listing = commands.add_parser("list")
    listing.add_argument("--suite", choices=["task4"], default="task4")
    run = commands.add_parser("run")
    run.add_argument("--suite", choices=["task4"], default="task4")
    run.add_argument("--port", required=True)
    run.add_argument("--remote")
    run.add_argument("--results")
    run.add_argument("--only", nargs="*")
    run.add_argument("--yes", action="store_true")
    args = parser.parse_args()
    if args.command == "list":
        for key, title in task4.PHASES:
            print(f"{key}: {title}")
        return 0
    return _cmd_run(args)

if __name__ == "__main__":
    raise SystemExit(main())
