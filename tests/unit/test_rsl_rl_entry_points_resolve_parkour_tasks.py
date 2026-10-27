"""The RSL-RL entry scripts must import the REAL ``parkour_tasks`` package, not the namespace shadow.

``parkour/`` sits on ``sys.path`` (the isaaclab_parkour editable ``.pth``), and ``parkour/parkour_tasks``
-- the outer directory that holds ``setup.py`` and the real ``parkour_tasks/`` package -- has no
``__init__.py``. Python's path finder therefore resolves the bare name ``parkour_tasks`` to that outer
directory as a PEP 420 namespace package before the editable meta-path finder is ever consulted, and
``parkour_tasks/__init__.py`` (``import_packages`` -> every task package's ``gym.register`` calls) never
runs. Sub-packages still resolve through the editable finder, which is why the forward task kept
registering: ``parkour_isaaclab.envs.mdp.rewards`` imports it by name. Nothing imports the sideways
package by name, so ``train.py`` raised ``NameNotFound: Isaac-Crab-Hex-Side-Walk`` (2026-09-11).

The fix (``train.py`` / ``play.py``): put the ``parkour_tasks`` source dir first on ``sys.path`` before
any ``parkour_tasks`` import, as ``eval_crab_hex_gait.py`` and ``zero_agent.py`` already did. Pure Python,
no Isaac: the resolution is checked with ``importlib.util.find_spec`` in an isolated interpreter (the
top-level spec lookup does not execute the package).
"""
import ast
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
PARKOUR = REPO / "parkour"
SCRIPTS = PARKOUR / "scripts" / "rsl_rl"
TASKS_SRC = PARKOUR / "parkour_tasks"
ENTRY_SCRIPTS = ("train.py", "play.py")


def _top_level_statements(path: Path) -> list[ast.stmt]:
    return ast.parse(path.read_text(), filename=str(path)).body


def _imports_parkour_tasks(node: ast.stmt) -> bool:
    if isinstance(node, ast.Import):
        return any(a.name == "parkour_tasks" or a.name.startswith("parkour_tasks.") for a in node.names)
    if isinstance(node, ast.ImportFrom):
        mod = node.module or ""
        return mod == "parkour_tasks" or mod.startswith("parkour_tasks.")
    return False


def _is_src_insert(node: ast.stmt) -> bool:
    """``sys.path.insert(0, _PARKOUR_TASKS_SRC)`` (possibly wrapped in ``if ... not in sys.path``)."""
    stmts = node.body if isinstance(node, ast.If) else [node]
    for s in stmts:
        if not (isinstance(s, ast.Expr) and isinstance(s.value, ast.Call)):
            continue
        call = s.value
        if ast.unparse(call.func) == "sys.path.insert" and len(call.args) == 2:
            if ast.unparse(call.args[0]) == "0" and ast.unparse(call.args[1]) == "_PARKOUR_TASKS_SRC":
                return True
    return False


@pytest.mark.parametrize("script", ENTRY_SCRIPTS)
def test_source_dir_is_inserted_before_any_parkour_tasks_import(script):
    body = _top_level_statements(SCRIPTS / script)
    inserts = [i for i, n in enumerate(body) if _is_src_insert(n)]
    imports = [i for i, n in enumerate(body) if _imports_parkour_tasks(n)]
    assert inserts, f"{script}: no `sys.path.insert(0, _PARKOUR_TASKS_SRC)` at module level"
    assert imports, f"{script}: no `import parkour_tasks` (the registration import)"
    assert inserts[0] < imports[0], f"{script}: the sys.path insert must precede the first parkour_tasks import"
    assert any(isinstance(n, ast.Import) and any(a.name == "parkour_tasks" for a in n.names) for n in body), \
        f"{script}: needs a bare `import parkour_tasks` so import_packages runs even if no sub-package is named"


def _resolve(path_entries: list[str]) -> dict:
    code = (
        "import sys, json, importlib.util as u\n"
        f"sys.path[:0] = {json.dumps(path_entries)}\n"
        "s = u.find_spec('parkour_tasks')\n"
        "print(json.dumps({'loader': None if s is None or s.loader is None else type(s.loader).__name__,"
        " 'origin': None if s is None else s.origin}))\n"
    )
    out = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True, check=True)
    return json.loads(out.stdout.strip().splitlines()[-1])


def test_entry_script_path_setup_resolves_the_real_package():
    """With ``parkour/`` on the path (as the .pth puts it) the bare name is a namespace package; with the
    entry scripts' insert first it is the real package whose ``__init__`` runs ``import_packages``."""
    shadowed = _resolve([str(SCRIPTS), str(PARKOUR)])
    assert shadowed["loader"] is None, f"expected the namespace shadow without the insert, got {shadowed}"

    fixed = _resolve([str(TASKS_SRC), str(SCRIPTS), str(PARKOUR)])
    assert fixed["loader"] == "SourceFileLoader", fixed
    assert Path(fixed["origin"]) == TASKS_SRC / "parkour_tasks" / "__init__.py", fixed


def test_entry_scripts_compute_the_same_source_dir():
    """``_PARKOUR_TASKS_SRC`` in both scripts is ``<script dir>/../../parkour_tasks`` == ``parkour/parkour_tasks``."""
    for script in ENTRY_SCRIPTS:
        src = (SCRIPTS / script).read_text()
        assert '_PARKOUR_TASKS_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "parkour_tasks"))' in src, script
    assert (SCRIPTS / ".." / ".." / "parkour_tasks").resolve() == TASKS_SRC
    assert (TASKS_SRC / "parkour_tasks" / "__init__.py").is_file()
    assert not (TASKS_SRC / "__init__.py").exists(), "the outer dir must stay a plain directory (setup.py lives there)"
