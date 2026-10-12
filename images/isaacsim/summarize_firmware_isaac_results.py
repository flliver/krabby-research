#!/usr/bin/env python3
"""Print firmware Isaac pytest verdict from mounted result artifacts (host-side).

``run_firmware_isaac_pytest.sh`` writes under ``build/firmware-isaac-pytest/`` inside the
container and re-evaluates pass/fail here — Kit shutdown often yields container exit 0 even
when pytest failed or did not finish reporting.
"""

from __future__ import annotations

import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


def _load_outcomes(results_dir: Path) -> list[dict]:
    path = results_dir / "outcomes.jsonl"
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _load_collected(results_dir: Path) -> int | None:
    path = results_dir / "collected.json"
    if not path.is_file():
        return None
    return int(json.loads(path.read_text(encoding="utf-8")).get("collected", 0))


def _junit_counts(results_dir: Path) -> tuple[int, int, int] | None:
    path = results_dir / "junit.xml"
    if not path.is_file():
        return None
    root = ET.parse(path).getroot()
    suites = root.findall("testsuite") if root.tag == "testsuites" else [root]
    tests = failures = errors = 0
    for suite in suites:
        tests += int(suite.get("tests", 0))
        failures += int(suite.get("failures", 0))
        errors += int(suite.get("errors", 0))
    return tests, failures, errors


def main(argv: list[str]) -> int:
    results_dir = Path(argv[1] if len(argv) > 1 else "build/firmware-isaac-pytest").resolve()
    if not results_dir.is_dir():
        print(f"ERROR: results directory missing: {results_dir}", file=sys.stderr)
        return 2

    outcomes = _load_outcomes(results_dir)
    expected = _load_collected(results_dir)
    junit = _junit_counts(results_dir)
    rc_path = results_dir / "pytest.rc"
    pytest_rc = int(rc_path.read_text(encoding="utf-8").strip()) if rc_path.is_file() else None

    passed = {o["nodeid"] for o in outcomes if o.get("outcome") == "passed"}
    failed = [o for o in outcomes if o.get("outcome") in ("failed", "error")]
    skipped = [o for o in outcomes if o.get("outcome") == "skipped"]

    print(f"Results directory: {results_dir}")
    if expected is not None:
        print(f"Collected (pytest): {expected}")
    print(f"Recorded outcomes: {len(outcomes)}")
    for row in outcomes:
        print(f"  [{str(row.get('outcome', '?')).upper()}] {row.get('nodeid')}")
        if row.get("longrepr"):
            print(f"         {row['longrepr']}")
    if junit is not None:
        print(f"JUnit: tests={junit[0]} failures={junit[1]} errors={junit[2]}")
    if pytest_rc is not None:
        print(f"In-container pytest exit code (informational): {pytest_rc}")

    # Hook rows are the primary source; JUnit catches setup/fixture errors that never reach
    # the call phase (outcomes.jsonl stays empty in that case).
    if failed:
        print(f"\nVERDICT: FAIL — {len(failed)} failed/errored")
        return 1
    if junit is not None and junit[1] + junit[2] > 0:
        print(f"\nVERDICT: FAIL — JUnit reports {junit[1]} failure(s), {junit[2]} error(s)")
        return 1
    if not outcomes:
        print("\nVERDICT: FAIL — no test outcomes on disk (Sim may have exited before pytest recorded results)")
        return 1
    if expected is not None and len(passed) + len(skipped) < expected:
        print(
            f"\nVERDICT: FAIL — only {len(passed) + len(skipped)}/{expected} tests reported "
            "(Kit may have torn down before pytest finished)"
        )
        return 1
    if not passed:
        print(f"\nVERDICT: FAIL — 0 passed, {len(skipped)} skipped")
        return 1

    print(f"\nVERDICT: PASS — {len(passed)} passed, {len(skipped)} skipped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
