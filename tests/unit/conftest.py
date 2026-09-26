"""Make the unit suite import THIS checkout's ``parkour`` code, and fail if anything leaks in from another.

The isaac venv resolves ``parkour_tasks`` through a setuptools editable finder and ``parkour_isaaclab``
through a ``.pth`` entry, and both point at the MAIN checkout. A test run from a git worktree
(``.claude/worktrees/<name>``) would therefore quietly exercise the main checkout's code for any
module it imports by package name. Two measures close that:

1. This conftest prepends ``<root>/parkour/parkour_tasks``, ``<root>/parkour`` and the gait-eval
   ``scripts`` dir to ``sys.path`` and to ``PYTHONPATH``, so subprocess tests inherit them too. Plain
   ``sys.path`` entries are consulted before the editable meta-path finder and before ``.pth`` entries.
2. After collection and again at session end, every loaded module whose file lies under ANOTHER
   checkout's ``parkour/`` tree fails the run. So does every ``parkour_tasks.*`` or
   ``parkour_isaaclab.*`` module outside this root.

``parkour_isaaclab`` has no ``__init__.py``, so it is a namespace package: Python merges every checkout's portion,
this one first. A submodule missing here would silently load from the main checkout; measure 2 catches that.

In the main checkout ``<root>`` IS the editable target, so nothing is changed there: no ``sys.path`` or
``PYTHONPATH`` edit, and the guard finds nothing. From a worktree, the exported ``PYTHONPATH`` is visible to
``test_crab_hex_omni_gate.py::test_train_without_the_new_kwargs_names_no_run_and_carries_no_overlay``, which
asserts the exact ``PYTHON*`` env of a driver child and so fails there. Its fixture logs
(``tests/unit/fixtures/omni_track_ratio/*.log``) are also git-ignored, so its 7 fixture tests fail in any
fresh worktree.

For hand-run tools from a worktree, export the same paths::

    export PYTHONPATH="$PWD/parkour/parkour_tasks:$PWD/parkour:$PWD/parkour/parkour_tasks/parkour_tasks/crab_hex_forward_task/scripts${PYTHONPATH:+:$PYTHONPATH}"
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
_PARKOUR = ROOT / "parkour"
_IMPORT_ROOTS = (
    _PARKOUR / "parkour_tasks",
    _PARKOUR,
    _PARKOUR / "parkour_tasks" / "parkour_tasks" / "crab_hex_forward_task" / "scripts",
)
_PACKAGES = ("parkour_tasks", "parkour_isaaclab")


def _main_checkout(root: Path) -> Path:
    """The checkout this worktree hangs off (``<main>/.claude/worktrees/<name>``), else ``root``."""
    parts = root.parts
    for i in range(len(parts) - 2):
        if parts[i] == ".claude" and parts[i + 1] == "worktrees":
            return Path(*parts[:i])
    return root


MAIN = _main_checkout(ROOT)

if MAIN != ROOT:  # a worktree: redirect imports here. The main checkout is left exactly as it was.
    for _p in reversed(_IMPORT_ROOTS):
        if str(_p) not in sys.path:
            sys.path.insert(0, str(_p))
    os.environ["PYTHONPATH"] = os.pathsep.join(
        [str(p) for p in _IMPORT_ROOTS] + [p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p]
    )


def _foreign_modules() -> list[str]:
    bad = []
    for name, mod in list(sys.modules.items()):
        path = getattr(mod, "__file__", None)
        if not path:
            continue
        resolved = Path(path).resolve()
        top = name.split(".", 1)[0]
        if top in _PACKAGES and not resolved.is_relative_to(ROOT):
            bad.append(f"{name} -> {resolved}")
        elif MAIN != ROOT and resolved.is_relative_to(MAIN / "parkour"):
            bad.append(f"{name} -> {resolved}")
    return sorted(bad)


def _check(when: str) -> None:
    bad = _foreign_modules()
    if bad:
        raise pytest.UsageError(
            f"modules imported from outside this checkout ({ROOT}) {when}:\n  " + "\n  ".join(bad)
        )


def pytest_collection_finish(session):
    _check("during collection")


@pytest.fixture(autouse=True, scope="session")
def _checkout_import_guard():
    yield
    bad = _foreign_modules()
    if bad:
        pytest.fail(
            f"modules imported from outside this checkout ({ROOT}) during the run:\n  " + "\n  ".join(bad),
            pytrace=False,
        )
