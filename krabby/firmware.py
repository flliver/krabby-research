"""krabby firmware — pass-through to krabby-firmware inside the locomotion container.

Distinct from the standalone `krabby-firmware` host CLI (the `firmware` package's
flash tool): this wrapper runs that same CLI *inside the installed image*, so a kit
owner who only `pip install krabby-launcher` can flash MCUs without installing flash
tools (avrdude/arduino-cli) on the host. Args are forwarded verbatim (except
``--channel``, which is translated to the matching S3 branch), and a TTY is
allocated when the host shell is interactive so the menu and paged output behave
the same as running `krabby-firmware` directly.

``--channel`` shapes supported here:

- ``krabby firmware --channel main update``
- ``krabby firmware update --channel main``  (flag inside remainder; peeled below)

Image selection for ``--channel`` uses ``resolve_image_ref`` in ``_state``.
"""
from __future__ import annotations

import subprocess
import sys
from typing import Optional

from krabby._docker import firmware_cmd
from krabby._state import CHANNELS, firmware_branch_for_channel, installed_image, resolve_image_ref


def _is_port(s: str) -> bool:
    """True if *s* looks like a serial device path, not an S3 branch name."""
    return s.startswith("/dev/") or s.startswith("COM") or s.upper().startswith("COM")


def pop_channel_flag(args: list[str]) -> tuple[Optional[str], list[str]]:
    """Remove ``--channel`` / ``--channel=`` from *args*; return ``(channel, rest)``.

    ``krabby firmware`` puts most tokens in argparse ``REMAINDER``, so
    ``krabby firmware update --channel main`` never hits the subparser's
    ``--channel`` and must be peeled off before forwarding into the container.
    """
    if args and args[0] == "--":
        args = args[1:]
    out: list[str] = []
    channel: Optional[str] = None
    i = 0
    while i < len(args):
        a = args[i]
        if a == "--channel":
            if i + 1 >= len(args):
                sys.exit("--channel requires main or release")
            channel = args[i + 1]
            i += 2
            continue
        if a.startswith("--channel="):
            channel = a.split("=", 1)[1]
            i += 1
            continue
        out.append(a)
        i += 1
    if channel is not None and channel not in CHANNELS:
        sys.exit(f"unknown channel {channel!r}; choose {' or '.join(CHANNELS)}")
    return channel, out


def merge_channel(
    parser_channel: Optional[str], args_channel: Optional[str]
) -> Optional[str]:
    """Combine subparser ``--channel`` with a flag found in remainder args."""
    if parser_channel and args_channel and parser_channel != args_channel:
        sys.exit(
            f"conflicting --channel values: {parser_channel!r} vs {args_channel!r}"
        )
    return parser_channel or args_channel


def apply_channel_to_firmware_args(args: list[str], channel: str) -> list[str]:
    """Rewrite ``update …`` args so the S3 branch matches *channel*.

    - ``main`` → ensure branch ``main`` (insert before any port).
    - ``release`` → omit branch so ``krabby-firmware update`` picks newest
      ``release/*``; a specific ``release/x.y.z`` positional is kept.
    - Non-``update`` commands are returned unchanged.
    """
    if not args or args[0] != "update":
        return args
    try:
        want = firmware_branch_for_channel(channel)
    except ValueError as exc:
        sys.exit(str(exc))

    rest = args[1:]
    existing: Optional[str] = None
    ports: list[str]
    if rest and not _is_port(rest[0]):
        existing = rest[0]
        ports = rest[1:]
    else:
        ports = rest

    if want is None:
        # release → newest release/* unless a specific release/* branch was given
        if existing == "main":
            sys.exit("channel 'release' conflicts with firmware branch 'main'")
        if existing is None:
            return ["update", *ports]
        return ["update", existing, *ports]

    if existing is not None and existing != want:
        sys.exit(
            f"channel {channel!r} conflicts with firmware branch {existing!r}"
        )
    return ["update", want, *ports]


def cmd_firmware(
    firmware_args: list[str],
    image_ref: Optional[str] = None,
    channel: Optional[str] = None,
) -> None:
    args_channel, firmware_args = pop_channel_flag(list(firmware_args or []))
    channel = merge_channel(channel, args_channel)

    if channel is not None:
        firmware_args = apply_channel_to_firmware_args(firmware_args, channel)

    if image_ref is None and channel is None:
        image_ref = installed_image()
    try:
        ref = resolve_image_ref(image_ref, channel=channel)
    except ValueError as exc:
        sys.exit(str(exc))

    interactive = sys.stdin.isatty() and sys.stdout.isatty()
    cmd = firmware_cmd(ref, firmware_args, interactive=interactive)
    result = subprocess.run(cmd)
    sys.exit(result.returncode)
