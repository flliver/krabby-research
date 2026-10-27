"""The gait_eval scoring path reproduces the committed golden fixture EXACTLY.

The omni task turns ``travel_yaw`` from a constant into a per-step array throughout
``gait_eval.metrics`` / ``gait_eval.report`` (plan D10 C2). Two lineages of record are scored by
that code -- the forward task at ``travel_yaw = 0.0`` and the sideways task at ``+pi/2`` -- so the
refactor is only acceptable if both produce bit-identical numbers, not merely close ones.

``tests/unit/fixtures/gait_eval_travel_yaw_golden.json`` was generated on the clean base commit
(plan Verification 0 / step 1) by ``gen_gait_eval_travel_yaw_golden.py``. This test re-runs that
generator's ``build_raw`` / ``score_both`` against the CURRENT code and asserts float equality with
the stored JSON. Never regenerate the fixture to make this pass: a diff here means the scoring
changed for the forward or sideways rows, which is the thing the fixture exists to forbid.

ONE deliberate regeneration, 2026-10-26 -- the user's decision "Make scoring BLAS-free" (Task 4 CHARTER).
The fixture had been written on an AVX-512 machine, whose OpenBLAS kernel computes the 2-element ``ddot``
behind ``np.linalg.norm`` as ``fma(x1, x1, x0 * x0)``; GitHub's non-AVX-512 runners compute the plain
``x0 * x0 + x1 * x1``, so CI scored 44 stride / slip_net_disp / slip_ratio leaves 1-2 ulp away from it and
passed only when the job happened to land on an AVX-512 runner. ``gait_eval.metrics`` now does its vector
reductions without BLAS (``_dot`` / ``_norm``, and ``pearson``), and the fixture was regenerated once, from
the unchanged generator. The new JSON is byte-identical under every OpenBLAS kernel that runs on the dev
machine and on numpy 1.26 and 2.5; it is exactly what the OLD code scores on a Haswell / Zen kernel (the
values the failing CI job reported); and it differs from the previous fixture only in those 44 leaves
(<= 2 ulp, |rel| <= 3.1e-16). That fixed a CPU dependence, it did not change what the rows score. It licenses
nothing else: the rule above stands. :func:`test_scoring_is_identical_under_every_openblas_kernel` keeps the
dependence from coming back.
"""

import json
import os
import platform
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
if str(FIXTURES_DIR) not in sys.path:
    sys.path.insert(0, str(FIXTURES_DIR))

import gen_gait_eval_travel_yaw_golden as GEN  # noqa: E402


@pytest.fixture(scope="module")
def golden() -> dict:
    return json.loads(GEN.FIXTURE_PATH.read_text())


@pytest.fixture(scope="module")
def scored() -> dict:
    return GEN.score_both(GEN.build_raw())


def test_generator_parameters_are_the_ones_the_fixture_was_written_with(golden):
    """A changed generator would make the comparison below vacuous rather than wrong."""
    gen = golden["generator"]
    assert gen["seed"] == GEN.SEED
    assert gen["dt"] == GEN.DT
    assert gen["n_envs"] == GEN.N_ENVS
    assert gen["half_period"] == GEN.HALF_PERIOD
    assert gen["fall_env"] == GEN.FALL_ENV
    assert gen["fall_frac"] == GEN.FALL_FRAC
    assert gen["fall_grace_steps"] == GEN.FALL_GRACE_STEPS
    assert gen["travel_yaws"] == GEN.TRAVEL_YAWS
    assert gen["schedule"] == [
        {"hold_s": h.hold_s, "vx": h.vx, "vy": h.vy, "wz": h.wz, "label": h.label}
        for h in GEN._schedule()
    ]


def test_forward_and_sideways_scoring_is_exactly_the_golden_fixture(scored, golden):
    """Float EQUALITY, not approx: the refactor must not move a single ulp."""
    assert scored == golden["results"]


@pytest.mark.parametrize("key", ["yaw_0", "yaw_half_pi"])
def test_each_episode_matches_key_by_key(scored, golden, key):
    """Same assertion, reported per episode/field so a regression names itself."""
    want, got = golden["results"][key], scored[key]
    assert len(got) == len(want)
    for i, (g, w) in enumerate(zip(got, want)):
        assert set(g) == set(w), f"{key}[{i}]: metric keys changed"
        for field in w:
            assert g[field] == w[field], f"{key}[{i}][{field}]"


def test_the_fixture_actually_exercises_the_paths_under_test(golden):
    """Guard against a fixture that would match trivially."""
    for key in ("yaw_0", "yaw_half_pi"):
        eps = golden["results"][key]
        assert len(eps) == GEN.N_ENVS
        assert eps[0]["tripod_score"] is not None and eps[0]["tripod_score"] > 0.9
        assert eps[0]["holds"]["fwd"]["tracking"]["vx"]["ratio"] is not None
        assert eps[GEN.FALL_ENV]["fall_direction"]["fall_class"] != "none"
        assert eps[GEN.FALL_ENV]["support_polygon"]["prefall"]["n_frames"] > 0
    # The two conventions must not be the same numbers, or the pi/2 half proves nothing.
    assert golden["results"]["yaw_0"][0]["holds"]["fwd"]["tracking"]["vx"]["ratio"] != (
        golden["results"]["yaw_half_pi"][0]["holds"]["fwd"]["tracking"]["vx"]["ratio"]
    )


def test_body_rows_carry_no_omni_fields(scored):
    """The world-row additions (facing; the per-category suite's direction / motion and episode
    wave / hold / category / hand) must be absent -- not null -- on body rows."""
    for key in ("yaw_0", "yaw_half_pi"):
        for ep in scored[key]:
            for field in ("facing", "facing_error_rms", "wave", "hold", "category", "hand"):
                assert field not in ep, field
            for hold in ep["holds"].values():
                for field in ("facing", "direction", "motion"):
                    assert field not in hold, field


# ---------------------------------------------------------------------------------------------------------
# CPU independence (2026-10-26, "Make scoring BLAS-free"): the scoring must not depend on which OpenBLAS
# kernel numpy loaded -- the reason the fixture above had to be regenerated once. OpenBLAS picks its kernel
# once, when it loads, so each kernel is forced (``OPENBLAS_CORETYPE``) in a fresh interpreter.
# ---------------------------------------------------------------------------------------------------------
#: Kernels forced beside the CPU's own, with the ``/proc/cpuinfo`` flags each needs to run. Haswell (the
#: kernel Zen CPUs load), Sandybridge and Nehalem are three ``ddot`` / ``gemm`` families that run on any
#: x86-64 CPU with AVX2 -- GitHub's runners included; SkylakeX (FMA ``ddot``) adds the AVX-512 family where the
#: CPU has it.
OPENBLAS_GUARD_KERNELS = (
    ("Haswell", ("avx2", "fma")),
    ("Sandybridge", ("avx",)),
    ("Nehalem", ("sse4_2",)),
    ("SkylakeX", ("avx512f", "avx512cd", "avx512bw", "avx512dq", "avx512vl")),
)

#: The child: the golden generator's raw data through ``score_run`` at both travel yaws (the stride / slip /
#: distance / support-polygon paths), plus three noisy tripod windows whose Pearson correlation is NOT exact
#: in floating point (the generator's perfect tripod makes every product in it exact, so it cannot see the
#: reduction order inside ``pearson``). It refuses to score any ``gait_eval`` but this checkout's. One JSON
#: document on stdout.
_GUARD_CHILD = """
import json, sys
from pathlib import Path
sys.dont_write_bytecode = True
sys.path.insert(0, sys.argv[1])
import numpy as np
import gen_gait_eval_travel_yaw_golden as GEN
assert Path(GEN.M.__file__).resolve().is_relative_to(Path(sys.argv[2]).resolve()), GEN.M.__file__
rng = np.random.default_rng(GEN.SEED)
noisy = []
for n in (97, 300, 601):
    contact = GEN._tripod_contact(n, GEN.HALF_PERIOD) ^ (rng.random((n, 6)) < 0.1)
    noisy.append(GEN.M.json_safe(GEN.M.tripod_window_metrics(contact, dt=GEN.DT)))
doc = {"golden": GEN.score_both(GEN.build_raw()), "noisy_tripod": noisy}
sys.stdout.write(json.dumps(doc, sort_keys=True, allow_nan=False))
"""


def _cpu_flags() -> set[str] | None:
    try:
        text = Path("/proc/cpuinfo").read_text()
    except OSError:
        return None
    for line in text.splitlines():
        if line.startswith("flags"):
            return set(line.split(":", 1)[1].split())
    return None


def _first_difference(a, b, path="$"):
    """The first leaf path at which two JSON documents differ (for the failure message)."""
    if type(a) is not type(b):
        return f"{path}: {a!r} != {b!r}"
    if isinstance(a, dict):
        if set(a) != set(b):
            return f"{path}: keys {sorted(set(a) ^ set(b))}"
        return next((d for k in sorted(a) if (d := _first_difference(a[k], b[k], f"{path}.{k}"))), None)
    if isinstance(a, list):
        if len(a) != len(b):
            return f"{path}: lengths {len(a)} != {len(b)}"
        return next((d for i, (u, v) in enumerate(zip(a, b)) if (d := _first_difference(u, v, f"{path}[{i}]"))), None)
    return None if a == b else f"{path}: {a!r} != {b!r}"


def _score_under(kernel: str | None) -> tuple[str, str | None]:
    """``(stdout, the core OpenBLAS reports)`` of the child under ``kernel`` (``None`` = the CPU's own).

    Hermetic: this interpreter with ``-I`` (no PYTHON* variables, no user site, no cwd on sys.path) and
    ``-B``, and an explicit environment: nothing from the test process's environment except ``PATH``.
    """
    env = {"PATH": os.environ.get("PATH", os.defpath), "LC_ALL": "C.UTF-8", "OPENBLAS_VERBOSE": "2"}
    if kernel is not None:
        env["OPENBLAS_CORETYPE"] = kernel
    r = subprocess.run([sys.executable, "-I", "-B", "-c", _GUARD_CHILD, str(FIXTURES_DIR), str(FIXTURES_DIR.parents[2])],
                       env=env, capture_output=True, text=True, timeout=20)
    assert r.returncode == 0, f"{kernel or 'native'}: rc {r.returncode}\n{r.stderr[-2000:]}"
    cores = re.findall(r"^Core: (\S+)", r.stderr, re.M)
    return r.stdout, (cores[-1] if cores else None)


def test_scoring_is_identical_under_every_openblas_kernel():
    """The same raw arrays score to the same bytes whichever OpenBLAS kernel numpy loaded.

    Before 2026-10-26 they did not: ``np.linalg.norm`` of a 2-vector (stride, slip, distance) and the
    support polygon's ``np.dot`` went to OpenBLAS ``ddot`` (FMA on the AVX-512 kernels, a plain multiply-add
    elsewhere), and ``pearson``'s ``np.corrcoef`` to a ``gemm`` whose reduction order differs between kernel
    families. Skipped -- never passed vacuously -- where the kernel cannot be chosen: not x86-64, numpy not on
    OpenBLAS, an OpenBLAS without DYNAMIC_ARCH, or fewer than two forced kernels this CPU can run.
    """
    if platform.machine().lower() not in ("x86_64", "amd64"):
        pytest.skip(f"the forced kernels are OpenBLAS x86-64 kernels; this machine is {platform.machine()!r}")
    try:
        blas = np.show_config(mode="dicts")["Build Dependencies"]["blas"]
    except (TypeError, KeyError) as exc:  # show_config(mode=) is numpy >= 1.25
        pytest.skip(f"cannot read numpy's BLAS from show_config: {exc!r}")
    if "openblas" not in str(blas.get("name", "")).lower():
        pytest.skip(f"numpy's BLAS is {blas.get('name')!r}, not OpenBLAS: OPENBLAS_CORETYPE selects nothing")
    flags = _cpu_flags()
    if flags is None:
        pytest.skip("no /proc/cpuinfo flags line: cannot tell which OpenBLAS kernels this CPU can run")
    runnable = [k for k, need in OPENBLAS_GUARD_KERNELS if set(need) <= flags]

    native, native_core = _score_under(None)
    if native_core is None:
        pytest.skip("OPENBLAS_VERBOSE=2 printed no 'Core:' line: numpy's OpenBLAS is not a DYNAMIC_ARCH build, "
                    "so OPENBLAS_CORETYPE cannot select a kernel")
    got = {f"native ({native_core})": native}
    not_built = []
    for kernel in runnable:
        out, core = _score_under(kernel)
        if core is None or core.lower() != kernel.lower():
            not_built.append(f"{kernel} (loaded {core})")
            continue
        got[kernel] = out
    if len(got) - 1 < 2:
        pytest.skip(f"fewer than two forced OpenBLAS kernels ran: the CPU can run {runnable}; "
                    f"not in this OpenBLAS build: {not_built}")

    (ref_name, ref), *rest = got.items()
    for name, out in rest:
        same = out == ref  # a bool, so a failure names the first differing leaf instead of diffing 600-kB strings
        assert same, (f"scoring depends on the OpenBLAS kernel: {name} != {ref_name} at "
                      f"{_first_difference(json.loads(out), json.loads(ref))}")


def test_blas_free_helpers_keep_numpy_dtype_and_shape_semantics():
    """``metrics._norm`` / ``metrics._dot`` are ``np.linalg.norm`` / ``np.dot`` without BLAS: the same result type
    (a numpy scalar; float32 in, float32 out; non-float input cast to float64), and bit-equal to the BLAS-free
    form numpy itself uses for ``np.linalg.norm(x, axis=k)``."""
    M = GEN.M
    rng = np.random.default_rng(GEN.SEED)
    for dtype in (np.float64, np.float32):
        for shape in ((2,), (3,), (17,), (2, 3)):
            x = rng.normal(size=shape).astype(dtype)
            got, ref = M._norm(x), np.linalg.norm(x)
            assert type(got) is type(ref) and np.ndim(got) == 0, (dtype, shape)
            assert got == np.linalg.norm(x.ravel(), axis=0), (dtype, shape)
            assert np.isclose(got, ref, rtol=1e-6 if dtype == np.float32 else 1e-15, atol=0)
            if len(shape) == 1:
                y = rng.normal(size=shape).astype(dtype)
                d, dref = M._dot(x, y), np.dot(x, y)
                assert type(d) is type(dref) and d == np.add.reduce(x * y), (dtype, shape)
                assert np.isclose(d, dref, rtol=1e-5 if dtype == np.float32 else 1e-12, atol=1e-12)
    for x in (np.array([3, 4]), [True, True]):
        assert type(M._norm(x)) is type(np.linalg.norm(x)) is np.float64
    assert M._norm(np.array([3, 4])) == 5.0
    assert type(M._dot(np.array([1, 2]), np.array([3, 4]))) is type(np.dot(np.array([1, 2]), np.array([3, 4])))
    with pytest.raises(ValueError):
        M._dot(np.ones(2), np.ones(3))
    with pytest.raises(ValueError):
        M._dot(np.ones((2, 2)), np.ones((2, 2)))
    with pytest.raises(TypeError):
        M._norm(np.ones(2, dtype=complex))
