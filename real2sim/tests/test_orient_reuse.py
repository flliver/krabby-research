"""STO-SCN-161 — find_any_orient: solve-scoped, model-agnostic orient reuse.

The first reconstructor (DA3 or matcha) computes the orient-floor gauge under the
solve; every subsequent one REUSES it, so each model stands alone AND all land in
the identical frame. `find_any_orient` is the lookup that makes that work.

Pulled out of v4exec.py by AST so the test doesn't import v4exec (importing it
runs its argparse main()). The function uses only pathlib.
"""
import ast
import time
from pathlib import Path


_V4 = Path(__file__).resolve().parents[1] / "v4exec.py"


def _load(name):
    for node in ast.parse(_V4.read_text()).body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            ns = {"Path": Path}
            exec(compile(ast.Module([node], []), "v4exec.py", "exec"), ns)
            return ns[name]
    raise AssertionError(f"{name} not found")


find_any_orient = _load("find_any_orient")


def _orient(solve_dir, oid):
    d = solve_dir / "orient" / oid
    d.mkdir(parents=True)
    (d / "oriented.json").write_text("{}")


def test_none_when_absent(tmp_path):
    assert find_any_orient(tmp_path) is None
    (tmp_path / "orient").mkdir()
    assert find_any_orient(tmp_path) is None          # empty orient/


def test_returns_existing_regardless_of_model(tmp_path):
    _orient(tmp_path, "MATCHAOID")
    assert find_any_orient(tmp_path) == "MATCHAOID"   # matcha-first → DA3 would reuse this


def test_newest_wins_so_both_reuse_one(tmp_path):
    _orient(tmp_path, "FIRST")
    time.sleep(0.02)
    _orient(tmp_path, "SECOND")
    # with two, the lookup converges on one (newest) → both reconstructors share it
    assert find_any_orient(tmp_path) == "SECOND"


def test_ignores_dir_without_oriented_json(tmp_path):
    (tmp_path / "orient" / "PARTIAL").mkdir(parents=True)   # no oriented.json
    assert find_any_orient(tmp_path) is None
    _orient(tmp_path, "REAL")
    assert find_any_orient(tmp_path) == "REAL"
