#!/usr/bin/env python3
"""STO-SCN-109 — one-time: collapse every scene's scores.jsonl to the latest
submission per (rater, slot), the one-true-submission. Also drops any `__diag__`
test rows. The store is git-tracked, so the pre-dedup state is recoverable (T-018).

Usage:  python3 rate_renders/dedup_scores.py [--dry-run]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from server import Handler, SCENES_ROOT  # noqa: E402  (reuse _latest_score_rows + root)


def main(dry_run: bool = False) -> int:
    before = after = 0
    for sj in sorted(SCENES_ROOT.glob("*/scores.jsonl")):
        orig = [json.loads(line) for line in sj.read_text().splitlines() if line.strip()]
        rows = [r for r in orig if r.get("rater") != "__diag__"]      # drop test pollution
        kept = Handler._latest_score_rows(rows)
        before += len(orig)
        after += len(kept)
        if len(kept) != len(orig):                      # changed: test rows dropped or deduped
            print(f"  {sj.parent.name}: {len(orig)} -> {len(kept)} rows"
                  + ("  (dry-run)" if dry_run else ""))
            if not dry_run:
                sj.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in kept))
    print(f"total: {before} -> {after} rows" + ("  (dry-run, nothing written)" if dry_run else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main("--dry-run" in sys.argv))
