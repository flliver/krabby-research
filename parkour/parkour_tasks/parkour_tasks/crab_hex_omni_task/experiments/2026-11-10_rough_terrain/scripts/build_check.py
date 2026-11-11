#!/usr/bin/env python3
"""Rough-terrain campaign: the BUILD PIN (PLAN step 5: "the code and eval sha are frozen from here through step 9").

    build_check.py show                print the live build (code_sha / eval_sha / dirty / HEAD); exit 0
    build_check.py pin                 write results/pin.json ONCE (refused when it exists or the tree is dirty)
    build_check.py check [--json P]    "BUILD_CHECK OK ..." exit 0, or "BUILD_CHECK BAD ..." exit 1

code_sha / eval_sha are computed with experiments/tools/run_candidates.py's own functions (``code_sha()`` /
``eval_sha()``, HEAD's tree over CODE_PATHSPECS / EVAL_PATHSPECS) and the clean-tree check is gate_row.sh's
(``git status --porcelain`` over both pathspecs minus ``code_drop``) -- the values every stamp's code_state.json
records, so a check that passes here is a stamp that pairs (readout.build_problems). The pin also freezes the
row-defining campaign files (manifests/, envs/: a change refuses) and records the run-machinery / read-out scripts
(a change is reported, not refused). Every stage of the queue (rt_queue.py) runs ``check`` first and stops on BAD.
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import argparse  # noqa: E402
from pathlib import Path  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rt_machinery as M  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=("show", "pin", "check"))
    ap.add_argument("--json", type=Path, default=None, help="check: also write the result here")
    ap.add_argument("--pin-path", type=Path, default=M.PIN_PATH)
    a = ap.parse_args(argv)
    live = M.live_build()
    if a.cmd == "show":
        print(f"BUILD code_sha {live['code_sha']} eval_sha {live['eval_sha']} head {live['head']} "
              f"dirty {len(live['dirty'])} {live['dirty'][:8]}")
        return 0
    if a.cmd == "pin":
        if a.pin_path.exists():
            print(f"BUILD_PIN REFUSED {a.pin_path} exists (written once; pinned_at "
                  f"{(M.read_pin(a.pin_path) or {}).get('pinned_at')})")
            return 1
        if live["dirty"]:
            print(f"BUILD_PIN REFUSED dirty code/eval paths {live['dirty'][:8]} (commit first)")
            return 1
        doc = M.pin_doc(live)
        M.write_json(a.pin_path, doc)
        print(f"BUILD_PIN WROTE {a.pin_path} code_sha {doc['code_sha'][:12]} eval_sha {doc['eval_sha'][:12]} "
              f"head {doc['head'][:10]} frozen_files {len(doc['frozen_files'])}")
        return 0
    pin = M.read_pin(a.pin_path)
    bad, warn = M.pin_problems(pin, live, M.campaign_files(M.FROZEN_GLOBS), M.campaign_files(M.REPORTED_GLOBS))
    for w in warn:
        print(f"BUILD_CHECK WARN {w}")
    print(f"BUILD_CHECK {'OK' if not bad else 'BAD'} code_sha {live['code_sha'][:12]} eval_sha "
          f"{live['eval_sha'][:12]} dirty {bool(live['dirty'])}" + (f" -- {'; '.join(bad)}" if bad else ""))
    if a.json:
        M.write_json(a.json, {"ok": not bad, "refusals": bad, "warnings": warn, "live": live, "pin": pin,
                              "at": M.now_iso()})
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
