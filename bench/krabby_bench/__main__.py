"""krabby-bench CLI entry point."""
from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import NoReturn
from pathlib import Path

from krabby_bench._config import CONFIG_PATH, EcrConfig, PID_PATH, SmokeConfig, load_config
from krabby_bench._harness import DEFAULT_JOINTS
from krabby_bench.watchdog import run


class _Parser(argparse.ArgumentParser):
    def error(self, message: str) -> NoReturn:
        print(f"krabby-bench: {message}\n", file=sys.stderr)
        self.print_help(sys.stderr)
        sys.exit(2)


def main() -> None:
    parser = _Parser(
        prog="krabby-bench",
        description="Bench watchdog + four-stage harness (install/flash/bringup/motion).",
    )
    parser.add_argument(
        "--config", metavar="PATH",
        default=str(CONFIG_PATH),
        help=f"Path to config.toml (default: {CONFIG_PATH})",
    )
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("help", help="Show this help message and exit.")

    subparsers.add_parser(
        "force-recheck",
        help="Signal the running watchdog to re-run the update + smoke test immediately, "
             "regardless of whether the ECR digest has changed.",
    )

    install_p = subparsers.add_parser(
        "install",
        help="Bootstrap the systemd service (must run as root). "
             "SSM mode: set BENCH_AWS_KEY_ID + BENCH_AWS_SECRET_KEY and pass --ssm-prefix. "
             "Legacy mode: set BENCH_SMTP_* and BENCH_GITHUB_TOKEN env vars.",
    )
    install_p.add_argument("--ecr-tag", default=EcrConfig.tag, metavar="TAG")
    install_p.add_argument("--firmware-channel", default=SmokeConfig.firmware_channel, metavar="CHANNEL")
    install_p.add_argument("--error-alert-type", default="both", choices=["email", "github", "both"],
                           dest="error_alert_type",
                           help="Alert delivery type (default: both)")
    install_p.add_argument("--github-repo", default=os.environ.get("BENCH_GITHUB_REPO", ""),
                           metavar="OWNER/REPO")
    install_p.add_argument("--ssm-prefix", default="/krabby/bench", metavar="PREFIX",
                           help="SSM parameter path prefix (default: /krabby/bench)")

    harness_p = subparsers.add_parser(
        "harness",
        help="Run the four-stage bench harness once: install → flash → bringup → motion. "
             "Uses ~/.venv-krabby-bench and scripts/jetson/bench-reset.sh (dual-use Orin).",
    )
    harness_p.add_argument(
        "--bench-venv",
        default=str(Path.home() / ".venv-krabby-bench"),
        help="Bench Install venv (default: ~/.venv-krabby-bench). Never uses ~/.venv-krabby.",
    )
    harness_p.add_argument(
        "--firmware-channel",
        default="release/0.2.15",
        help="Firmware channel for the flash stage (default: release/0.2.15)",
    )
    harness_p.add_argument("--no-reset", action="store_true", help="Skip bench-reset.sh before Install")
    harness_p.add_argument("--rmi", action="store_true", help="Pass --rmi to bench-reset.sh")
    harness_p.add_argument("--skip-install", action="store_true")
    harness_p.add_argument("--skip-flash", action="store_true")
    harness_p.add_argument("--skip-bringup", action="store_true")
    harness_p.add_argument("--skip-motion", action="store_true")
    harness_p.add_argument(
        "--joint",
        nargs="+",
        default=list(DEFAULT_JOINTS),
        help=f"Joints to jog in motion stage (default: {' '.join(DEFAULT_JOINTS)})",
    )
    harness_p.add_argument("--jog-pwm", type=int, default=180)
    harness_p.add_argument("--jog-seconds", type=float, default=1.5)
    harness_p.add_argument("--bringup-timeout", type=float, default=90.0)
    harness_p.add_argument(
        "--repo-root",
        default="",
        help="krabby-research root (for bench-reset.sh / pip install ./bench). Default: inferred.",
    )
    harness_p.add_argument(
        "--run-url",
        default="",
        help="CI run URL for Discord (default: derived from GITHUB_* env when set).",
    )
    harness_p.add_argument(
        "--commit",
        default="",
        help="Commit SHA for Discord (default: GITHUB_SHA).",
    )
    harness_p.add_argument(
        "--commit-subject",
        default="",
        help="Short commit subject for Discord.",
    )
    harness_p.add_argument(
        "--no-discord",
        action="store_true",
        help="Skip Discord notify even if DISCORD_WEBHOOK_URL is set.",
    )

    args = parser.parse_args()

    if args.command == "help":
        parser.print_help()
        return

    if args.command == "force-recheck":
        import signal as _signal
        if not PID_PATH.exists():
            print("error: krabby-bench is not running (no PID file at %s)" % PID_PATH, file=sys.stderr)
            sys.exit(1)
        try:
            pid = int(PID_PATH.read_text().strip())
        except ValueError:
            print("error: PID file is corrupt", file=sys.stderr)
            sys.exit(1)
        try:
            os.kill(pid, _signal.SIGUSR1)
            print(f"Force recheck signal sent to krabby-bench (pid {pid})")
        except ProcessLookupError:
            print(f"error: no process with pid {pid} — service may have stopped", file=sys.stderr)
            sys.exit(1)
        return

    if args.command == "install":
        from krabby_bench._install import install
        install(
            ecr_tag=args.ecr_tag,
            firmware_channel=args.firmware_channel,
            mode=args.error_alert_type,
            github_repo=args.github_repo,
            ssm_prefix=args.ssm_prefix,
        )
        return

    if args.command == "harness":
        from krabby_bench._harness import HarnessConfig, run_harness

        logging.basicConfig(
            level=getattr(logging, args.log_level),
            format="%(asctime)s %(levelname)s %(message)s",
        )
        cfg = HarnessConfig(
            bench_venv=Path(args.bench_venv).expanduser(),
            firmware_channel=args.firmware_channel,
            reset=not args.no_reset,
            reset_rmi=args.rmi,
            skip_install=args.skip_install,
            skip_flash=args.skip_flash,
            skip_bringup=args.skip_bringup,
            skip_motion=args.skip_motion,
            joints=list(args.joint),
            jog_pwm=args.jog_pwm,
            jog_seconds=args.jog_seconds,
            bringup_timeout=args.bringup_timeout,
            repo_root=Path(args.repo_root).resolve() if args.repo_root else None,
        )
        result = run_harness(cfg)
        print("")
        print("=== harness summary ===")
        for line in result.summary_lines():
            print(line)
        detail = "\n".join(result.summary_lines())
        if not args.no_discord:
            from krabby_bench._discord import notify_harness_result

            notify_harness_result(
                ok=result.ok,
                failed_stage=result.failed_stage,
                detail=detail,
                run_url=args.run_url,
                commit=args.commit,
                subject=args.commit_subject,
            )
        if result.ok:
            print("PASS")
            sys.exit(0)
        print(f"FAIL at stage: {result.failed_stage}")
        sys.exit(1)

    config = load_config(Path(args.config))
    run(config, log_level=args.log_level)


if __name__ == "__main__":
    main()
