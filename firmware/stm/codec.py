"""cantools load/singleton for krabby_leg.dbc."""

from __future__ import annotations

from pathlib import Path

_DBC_PATH = Path(__file__).resolve().parent / "dbc" / "krabby_leg.dbc"


def dbc_path() -> Path:
    return _DBC_PATH


def load_database():
    import cantools

    return cantools.database.load_file(_DBC_PATH)
