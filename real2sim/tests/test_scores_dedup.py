#!/usr/bin/env python3
"""STO-SCN-109 — one-submission-per-ranker dedup tests.

`_latest_score_rows` keeps only the rows of the latest submission per (rater, slot);
distinct raters / slots are preserved.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "rate_renders"))
from server import Handler  # noqa: E402

L = Handler._latest_score_rows


def _row(rater, slot, at, rank, ts):
    return {"schema": 4, "at": at, "slot": slot, "rank": rank, "rater": rater, "ts": ts}


def test_keeps_only_latest_submission_for_a_rater_slot():
    rows = [
        _row("Jeremy", "01", "A", 1, "2026-06-12T00:00:00-07:00"),
        _row("Jeremy", "01", "B", 2, "2026-06-12T00:00:00-07:00"),   # submission 1 (2 rows)
        _row("Jeremy", "01", "A", 2, "2026-06-15T00:00:00-07:00"),
        _row("Jeremy", "01", "B", 1, "2026-06-15T00:00:00-07:00"),   # submission 2 (latest)
    ]
    out = L(rows)
    assert len(out) == 2
    assert all(r["ts"].startswith("2026-06-15") for r in out)
    assert {(r["at"], r["rank"]) for r in out} == {("A", 2), ("B", 1)}   # the latest ranks


def test_distinct_raters_and_slots_preserved():
    rows = [
        _row("Jeremy", "01", "A", 1, "2026-06-15T01:00:00-07:00"),
        _row("Alice", "01", "A", 1, "2026-06-15T02:00:00-07:00"),     # different rater
        _row("Jeremy", "02", "A", 1, "2026-06-15T03:00:00-07:00"),    # different slot
    ]
    out = L(rows)
    assert len(out) == 3                                              # none collapsed
    keys = {(r["rater"], r["slot"]) for r in out}
    assert keys == {("Jeremy", "01"), ("Alice", "01"), ("Jeremy", "02")}


def test_empty_and_single():
    assert L([]) == []
    one = [_row("Jeremy", "01", "A", 1, "2026-06-15T00:00:00-07:00")]
    assert L(one) == one


if __name__ == "__main__":
    import traceback
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn(); print(f"  PASS {fn.__name__}")
        except Exception:
            failed += 1; print(f"  FAIL {fn.__name__}"); traceback.print_exc()
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
