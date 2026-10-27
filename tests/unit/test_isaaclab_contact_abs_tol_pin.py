"""Pin: Isaac Lab's ``ContactSensor.compute_first_contact`` / ``compute_first_air`` still default
``abs_tol`` to ``1e-8`` (stride/posture plan, plan step 5).

WHY THIS MATTERS. The old ``reward_stride_length`` (``compute_first_contact(step_dt)``) and
``reward_feet_air_time_positive`` (``compute_first_air`` timing) are dead in every task that
carries them: the contact sensor's float32 ``_timestamp`` drifts about 2 s into an episode, so the
per-refresh elapsed time stops satisfying ``current_contact_time < dt + abs_tol`` at the shipped
``abs_tol`` (see ``crab_hex_omni_stroke_math.py`` and
``crab_hex_omni_task/experiments/tools/replay_motion_terms.py:sensor_elapsed_f32``). Both baked
heads and the whole clock campaign were trained against that dead arithmetic, bit for bit. If an
upstream Isaac Lab release loosens ``abs_tol`` (e.g. to something that tolerates the observed
~1e-6 float32 rounding), ``reward_stride_length`` (weight 0.5 where it is still wired) and
``reward_feet_air_time_positive`` (weight 0.8) would silently come back alive in every task that
carries them, changing the training economy of every affected run without a single line of THIS
repo's code changing. This test is the tripwire: it pins the two defaults so a venv/Isaac Lab
upgrade that moves them fails loudly here, with an explanation, instead of quietly re-arming a
reward term nobody re-baselined for.

TEXT-ONLY: this never imports ``isaaclab`` (which drags in ``pxr``/Kit). The source file is
located via ``importlib.util.find_spec``, which does not import the package for a plain
``sys.path`` egg or an editable install (verified in this venv: it lands on the on-disk
``__init__.py`` via the package's own meta-path finder without adding ``isaaclab`` itself to
``sys.modules``). If some other collected test happens to make ``find_spec`` behave otherwise in
this venv (or it raises), this test falls back to the known Isaac Lab checkout path rather than
import anything. The file is then only ``ast``-parsed, never executed.
"""

from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path

import pytest

#: Known-good fallback if ``find_spec`` can't locate the package without importing it (see
#: ``_isaaclab_contact_sensor_py``'s docstring).
FALLBACK_ISAACLAB_INIT_DIR = Path("/home/nickmagus/krabby/IsaacLab/source/isaaclab/isaaclab")

TARGET_FUNCS = ("compute_first_contact", "compute_first_air")
EXPECTED_ABS_TOL = 1e-8


def _isaaclab_contact_sensor_py() -> Path | None:
    """Locate ``isaaclab/sensors/contact_sensor/contact_sensor.py`` on disk, without importing it.

    ``importlib.util.find_spec`` resolves an editable/egg install's ``submodule_search_locations``
    by walking ``sys.meta_path`` and does not need to execute the package's ``__init__.py`` to do
    it (Python only imports the module once something asks for it via ``import`` /
    ``importlib.import_module`` / ``exec_module`` -- none of which happen here). As a defence
    against a future venv where some finder DOES import as a side effect, or where
    ``find_spec`` raises (e.g. a stub already sitting in ``sys.modules`` with ``__spec__ is
    None`` -- observed elsewhere in this suite, see ``test_crab_hex_omni_rewards.py``), both
    outcomes fall back to the known checkout path instead of trusting an import that may have
    happened.
    """
    before = "isaaclab" in sys.modules
    try:
        spec = importlib.util.find_spec("isaaclab")
    except Exception:  # noqa: BLE001 -- a hostile/broken finder must not break collection
        spec = None
    imported_as_side_effect = ("isaaclab" in sys.modules) and not before
    if spec is not None and spec.submodule_search_locations and not imported_as_side_effect:
        for loc in spec.submodule_search_locations:
            candidate = Path(loc) / "sensors" / "contact_sensor" / "contact_sensor.py"
            if candidate.exists():
                return candidate
    fallback = FALLBACK_ISAACLAB_INIT_DIR / "sensors" / "contact_sensor" / "contact_sensor.py"
    return fallback if fallback.exists() else None


def _default_value(node: ast.FunctionDef, arg_name: str):
    """The AST default (a literal) bound to ``arg_name`` in ``node``'s signature, or ``_MISSING``."""
    args = node.args
    positional = args.posonlyargs + args.args
    # Defaults right-align to the end of the positional args (Python's own binding rule).
    defaulted = positional[len(positional) - len(args.defaults):]
    for name, default in zip(defaulted, args.defaults):
        if name.arg == arg_name:
            if isinstance(default, ast.Constant):
                return default.value
            return _MISSING  # a non-literal default (unexpected here) -- don't guess its value
    for kw_name, kw_default in zip(args.kwonlyargs, args.kw_defaults):
        if kw_name.arg == arg_name and kw_default is not None and isinstance(kw_default, ast.Constant):
            return kw_default.value
    return _MISSING


class _Missing:
    def __repr__(self) -> str:
        return "<no such parameter / default>"


_MISSING = _Missing()


def _find_method_defaults(source: str, class_name: str, func_names: tuple[str, ...], arg_name: str) -> dict:
    """``{func_name: default_value}`` for methods of ``class_name`` in ``source`` (never executed)."""
    tree = ast.parse(source)
    out: dict = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name in func_names:
                    out[item.name] = _default_value(item, arg_name)
    return out


CONTACT_SENSOR_PY = _isaaclab_contact_sensor_py()


@pytest.mark.skipif(CONTACT_SENSOR_PY is None, reason="Isaac Lab source not found in this venv/checkout")
def test_compute_first_contact_and_air_still_default_abs_tol_to_1e_minus_8():
    source = CONTACT_SENSOR_PY.read_text()
    defaults = _find_method_defaults(source, "ContactSensor", TARGET_FUNCS, "abs_tol")
    missing = [name for name in TARGET_FUNCS if name not in defaults]
    assert not missing, (
        f"{CONTACT_SENSOR_PY} no longer defines {missing} on ContactSensor -- Isaac Lab's contact "
        "sensor API moved out from under this pin; re-check compute_first_contact/compute_first_air "
        "by hand before touching anything here."
    )
    for name in TARGET_FUNCS:
        assert defaults[name] == pytest.approx(EXPECTED_ABS_TOL), (
            f"ContactSensor.{name}'s abs_tol default moved from {EXPECTED_ABS_TOL} to "
            f"{defaults[name]} in {CONTACT_SENSOR_PY}. This default being small and strict is WHY "
            "reward_stride_length (weight 0.5 where it is still wired) and "
            "reward_feet_air_time_positive (weight 0.8) are dead in training once the contact "
            "sensor's float32 timestamp drifts (~2 s into an episode): loosening abs_tol would let "
            "compute_first_contact / compute_first_air start matching again on rows they have "
            "never paid on, silently reviving both terms and changing the training economy of "
            "every task that carries them. Re-baseline (or explicitly re-pin this test to the new "
            "value with that consequence understood) before bumping the Isaac Lab version."
        )
