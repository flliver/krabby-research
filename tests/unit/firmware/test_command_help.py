"""Unit tests for the `krabby-firmware help` curated command summary.

The summary is hand-maintained (firmware/__main__.py:_COMMAND_HELP) so it can show
each subcommand's required arguments inline. These tests guard it from drifting
out of sync with the argparse subparsers it documents."""
import argparse
import io
from contextlib import redirect_stdout
from unittest.mock import patch

from firmware.__main__ import _COMMAND_HELP, _command_help, main


def _registered_subcommands() -> set[str]:
    """The subcommand names main()'s argparse actually registers."""
    captured: dict[str, set[str]] = {}
    real_add_subparsers = argparse.ArgumentParser.add_subparsers

    def spy_add_subparsers(self, *a, **k):
        sub = real_add_subparsers(self, *a, **k)
        real_add_parser = sub.add_parser
        names: set[str] = set()

        def spy_add_parser(name, *aa, **kk):
            names.add(name)
            return real_add_parser(name, *aa, **kk)

        sub.add_parser = spy_add_parser
        captured["names"] = names
        return sub

    # SystemExit because no args are supplied; we only want the parser built.
    with patch.object(argparse.ArgumentParser, "add_subparsers", spy_add_subparsers):
        with patch("sys.argv", ["krabby-firmware", "help"]):
            with redirect_stdout(io.StringIO()):
                main()
    return captured["names"]


def _help_command_words() -> set[str]:
    """First whitespace-delimited token of each usage row = the subcommand name."""
    return {usage.split()[0] for usage, _ in _COMMAND_HELP}


def test_help_documents_every_registered_subcommand():
    assert _help_command_words() == _registered_subcommands()


def test_help_lists_required_args_for_value_commands():
    text = _command_help()
    # Commands with required positionals/flags must surface them in the summary.
    assert "KEY=VAL" in text          # set
    assert "JOINT" in text            # calibrate-joint / get-calibration / jog
    assert "--joint" in text and "--pwm" in text  # jog required flags


def test_help_mentions_default_interactive_mode():
    text = _command_help()
    assert "no command" in text.lower()


def test_help_command_prints_summary(capsys):
    with patch("sys.argv", ["krabby-firmware", "help"]):
        main()
    out = capsys.readouterr().out
    assert out.strip() == _command_help().strip()
