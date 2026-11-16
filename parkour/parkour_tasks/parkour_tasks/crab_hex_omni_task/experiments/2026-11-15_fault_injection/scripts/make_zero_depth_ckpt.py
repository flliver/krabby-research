#!/usr/bin/env python3
"""Fault-injection sweep, anchor T12: the zero-depth copy of C1S (the 2026-11-06 sweep's mechanism, run_sweep.py
``prepare()``). Run by hand / by the queue, never on import.

Writes ``<MAIN>/parkour/outputs/fault_sweep/ckpt_zero_depth/`` (cells.yaml ``checkpoints.C1S_zero_depth``): the C1S
.pt copied byte for byte (sha checked), ``params/env.yaml`` copied, and ``params/agent.yaml`` with its single
``zero_depth: false`` flipped to ``zero_depth: true`` (the student runner then feeds the encoder a normalised-zero
image). An existing copy that already matches is left alone (``--check`` only verifies; exit 1 when it does not).

    python3 scripts/make_zero_depth_ckpt.py [--check]
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
CAMPAIGN = HERE.parent
CELLS_YAML = CAMPAIGN / "cells.yaml"
#: The only place this helper may write (MAIN outputs, untracked).
ALLOWED_ROOT = Path("/home/nickmagus/krabby/krabby-research/parkour/outputs/fault_sweep")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def paths() -> tuple[Path, Path, str]:
    doc = yaml.safe_load(CELLS_YAML.read_text())
    ck = doc["checkpoints"]
    src, dst = Path(ck["C1S"]["path"]), Path(ck["C1S_zero_depth"]["path"])
    if not (dst.is_absolute() and ALLOWED_ROOT in dst.parents):
        raise SystemExit(f"refusing to write {dst}: not under {ALLOWED_ROOT}")
    if src.name != dst.name:
        raise SystemExit(f"the copy must keep the checkpoint name ({src.name} vs {dst.name})")
    return src, dst, str(ck["C1S"]["sha256_prefix"])


def want_agent(src: Path) -> str:
    text = (src.parent / "params" / "agent.yaml").read_text()
    if text.count("zero_depth: false") != 1:
        raise SystemExit(f"{src.parent}/params/agent.yaml: expected exactly one 'zero_depth: false' to flip")
    return text.replace("zero_depth: false", "zero_depth: true")


def matches(src: Path, dst: Path) -> bool:
    zd = dst.parent
    return (dst.is_file() and not dst.is_symlink() and _sha256(dst) == _sha256(src)
            and (zd / "params" / "env.yaml").is_file()
            and (zd / "params" / "env.yaml").read_bytes() == (src.parent / "params" / "env.yaml").read_bytes()
            and (zd / "params" / "agent.yaml").is_file()
            and (zd / "params" / "agent.yaml").read_text() == want_agent(src))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--check", action="store_true", help="verify only; exit 1 when the copy is missing or stale")
    args = ap.parse_args(argv)
    src, dst, prefix = paths()
    if not _sha256(src).startswith(prefix):
        raise SystemExit(f"{src}: sha256 does not start with {prefix}")
    ok = matches(src, dst)
    if args.check or ok:
        print(f"zero-depth copy {'OK' if ok else 'MISSING / STALE'}: {dst}")
        return 0 if ok else 1
    zd = dst.parent
    (zd / "params").mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    shutil.copy2(src, dst)
    shutil.copy2(src.parent / "params" / "env.yaml", zd / "params" / "env.yaml")
    (zd / "params" / "agent.yaml").write_text(want_agent(src))
    if not matches(src, dst):
        raise SystemExit(f"{dst}: the copy does not verify")
    print(f"wrote zero-depth copy {dst} (sha {_sha256(dst)[:8]})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
