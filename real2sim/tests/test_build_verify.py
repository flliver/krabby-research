"""STO-SCN-095 — verify-surface frustum math. Pure stdlib."""
import importlib.util
from pathlib import Path

_MOD = Path(__file__).resolve().parents[1] / "verify_viewer" / "build_verify.py"
_spec = importlib.util.spec_from_file_location("build_verify", _MOD)
bv = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bv)


def test_frustum_identity_pose():
    # identity rotation, t=(1,2,3): center = -R^T t = (-1,-2,-3); c2w = I
    w2c = [[1, 0, 0, 1], [0, 1, 0, 2], [0, 0, 1, 3], [0, 0, 0, 1]]
    R, c = bv.frustum_from_w2c(w2c)
    assert c == [-1.0, -2.0, -3.0]
    assert R == [1, 0, 0, 0, 1, 0, 0, 0, 1]


def test_frustum_rotation_transposes():
    # 90deg about Z (w2c): R_w2c = [[0,-1,0],[1,0,0],[0,0,1]]; c2w = transpose
    w2c = [[0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    R, c = bv.frustum_from_w2c(w2c)
    assert c == [0.0, 0.0, 0.0]
    # c2w = R_w2c^T = [[0,1,0],[-1,0,0],[0,0,1]] flattened row-major
    assert R == [0, 1, 0, -1, 0, 0, 0, 0, 1]
