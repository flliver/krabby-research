#!/usr/bin/env python3
"""The eval env of a body reference chain's head, composed EXACTLY as the driver composes it (CPU, pure).

``run_phases.eval_env_for(PHASES[stage.base], plant, carry_overlay=stage.overlay)`` (+ ``KRABBY_PLANT``), the call
``run_candidates`` makes for a stage eval -- for heads whose stage the driver did not evaluate (the smoke chains) and
for the rows the driver does not run (turnlevels, field rows). ``--row-env`` merges a manifest row's ``env:`` block
into the PROCESS env, the way the REFS / anchor field stamps were launched (their env_vars = the suite env + the
row block), so a body head's field eval is launched like its anchors'.

    bc_head_env.py --chain BC_x_s18 --out FILE [--row-env MANIFEST:ROW_ID]
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
C = HERE.parent
TOOLS = C.parent / "tools"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def head_env(chain: str, row_env: str | None = None) -> dict:
    rp = _load(TOOLS / "run_phases.py", "run_phases_bc_head_env")
    cands = yaml.safe_load((C / "body_cmd_train/candidates.yaml").read_text())
    st = cands["chains"][chain]["stages"][0]
    spec = rp.ph.PHASES[st["base"]]
    # keep_source FORMATION + CANARY_KEYS: the suite (eval_morph) call the driver makes, run_phases.py:1856
    ev = dict(rp.eval_env_for(spec, "A15+B", keep=rp.CANARY_KEYS, keep_source=rp.ph.FORMATION,
                              carry_overlay=st["overlay"]))
    ev["KRABBY_PLANT"] = "A15+B"
    if row_env:
        man, rid = row_env.split(":", 1)
        rows = {r["id"]: r for r in yaml.safe_load(Path(man).read_text())["scenarios"]}
        ev.update({k: str(v) for k, v in (rows[rid].get("env") or {}).items()})
    return ev


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chain", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--row-env", default=None)
    a = ap.parse_args()
    ev = head_env(a.chain, a.row_env)
    Path(a.out).write_text("".join(f"{k}={v}\n" for k, v in sorted(ev.items())))
    print(f"{a.out}: {len(ev)} keys; frame={ev.get('KRABBY_OMNI_COMMAND_FRAME')} "
          f"encoding={ev.get('KRABBY_OMNI_FACING_OBS_ENCODING')} envelope={ev.get('KRABBY_OMNI_BODY_ENVELOPE')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
