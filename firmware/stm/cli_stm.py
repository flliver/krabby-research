"""Subcommand handlers for krabby-firmware --stm."""

from __future__ import annotations


def not_implemented(subcommand: str) -> int:
    raise NotImplementedError(f"krabby-firmware --stm {subcommand}")
