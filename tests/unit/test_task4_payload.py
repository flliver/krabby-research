"""M18 Task 4 pipeline, step 3 ("M1a", 2026-10-16): the payload knobs and the ``payload_reset`` event, on CPU.

``config/crab_hex/payload_knobs.py`` (``KRABBY_PAYLOAD_KG / _ZERO_FRAC / _COM_XY / _COM_Z``, default off) arms
``events.payload_reset`` (``mdp/omni_events.py``): a deck-footprint box load re-drawn at every reset and written into
PhysX as an ABSOLUTE composite (total mass, mass-weighted CoM, parallel-axis inertia) on each env's lazily captured
post-startup-DR base. Every failure mode here is silent in training -- a CoM that walks a little every reset, a partial
reset that also moves the envs that did not reset, a base captured before the startup DR ran, a 0-kg row that is
re-written through a float round trip, a payload stream that shifts every other random draw -- so each is pinned:

* knobs: parsing and refusals; unset (or blank) writes nothing; armed writes exactly the term the event expects;
  ``omni_knobs`` re-exports them for the driver whitelist and applies them before the command-frame swap;
* math: the composite against a hand calculation and against an independent (about-the-origin) route; the box
  inertia against a numerical integration; the box size against ``crab_hex_dimensions``;
* the event, on a fake PhysX view that refills cached buffers like the real one: partial resets never compound and
  never touch other envs, lazy capture sees the startup DR, 0 kg is a no-op (``0:0`` never writes), a loaded env
  drawn 0 kg is restored bit for bit, level rows are exact, the private generator is deterministic and leaves the
  global stream alone, the run_meta record.

Nothing real is registered under ``isaaclab.*`` / ``parkour_tasks.*``: ``omni_events`` is path-loaded under a private
package name with the same ``isaaclab.managers`` stand-in the RSI suites use.
"""

from __future__ import annotations

import copy
import importlib
import importlib.machinery
import importlib.util
import inspect
import math
import sys
import types
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[2]
OMNI = REPO / "parkour" / "parkour_tasks" / "parkour_tasks" / "crab_hex_omni_task"
OMNI_MDP = OMNI / "mdp"
PAYLOAD_KNOBS_PY = OMNI / "config" / "crab_hex" / "payload_knobs.py"
OMNI_KNOBS_PY = OMNI / "config" / "crab_hex" / "omni_knobs.py"
DIMS_PY = REPO / "parkour/parkour_tasks/parkour_tasks/crab_hex_forward_task/mdp/crab_hex_dimensions.py"
PARKOURS_DIR = REPO / "parkour" / "parkour_isaaclab" / "envs" / "mdp" / "parkours"
PKG = "_omni_mdp_payload_under_test"


def _load(path: Path, name: str):
    mod = sys.modules.get(name)
    if mod is None:
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
    return mod


def _load_real_module(dotted: str, path: Path) -> types.ModuleType:
    """Register the real file at ``path`` under ``dotted`` (``test_crab_hex_omni_rsi.py``'s helper, same rules)."""
    if dotted in sys.modules:
        return sys.modules[dotted]
    parts = dotted.split(".")
    directory = path.parent
    for i in range(len(parts) - 1, 0, -1):
        name = ".".join(parts[:i])
        pkg = sys.modules.get(name)
        if pkg is None:
            pkg = types.ModuleType(name)
            pkg.__path__ = [str(directory)]
            sys.modules[name] = pkg
        elif str(directory) not in getattr(pkg, "__path__", []):
            pkg.__path__ = list(getattr(pkg, "__path__", [])) + [str(directory)]
        directory = directory.parent
    spec = importlib.util.spec_from_file_location(dotted, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[dotted] = mod
    spec.loader.exec_module(mod)
    setattr(sys.modules[".".join(parts[:-1])], parts[-1], mod)
    return mod


# the real (pure-torch) exposure ledger the omni exposure helpers import
_load_real_module("parkour_isaaclab.envs.mdp.parkours.exposure_stats", PARKOURS_DIR / "exposure_stats.py")
# isaaclab.managers: the RSI suites' stand-in (SceneEntityCfg only) -- omni_events must import under it
_isaaclab = sys.modules.setdefault("isaaclab", types.ModuleType("isaaclab"))
_managers = sys.modules.get("isaaclab.managers")
if _managers is None:
    _managers = types.ModuleType("isaaclab.managers")
    sys.modules["isaaclab.managers"] = _managers
if not hasattr(_managers, "SceneEntityCfg"):
    class SceneEntityCfg:  # noqa: D401 - stand-in for the cfg dataclass
        def __init__(self, name="robot", **kw):
            self.name = name
    _managers.SceneEntityCfg = SceneEntityCfg
_isaaclab.managers = _managers
if PKG not in sys.modules:
    _spec = importlib.machinery.ModuleSpec(PKG, loader=None, is_package=True)
    _spec.submodule_search_locations = [str(OMNI_MDP)]
    sys.modules[PKG] = importlib.util.module_from_spec(_spec)

oe = importlib.import_module(f"{PKG}.omni_events")
pk = _load(PAYLOAD_KNOBS_PY, "payload_knobs__task4_test")

BOX = pk.PAYLOAD_BOX_SIZE_M
BODY_DIMS = (0.7112, 1.2192, 0.3175)
BODY_MASS = 158.7573295
D5 = {"kg_range": (0.0, 90.0), "zero_frac": 0.25, "com_xy": 0.10, "com_z": 0.309}


def _level(kg: float) -> dict:
    return {"kg_range": (kg, kg), "zero_frac": 0.0, "com_xy": 0.0, "com_z": 0.309}


# =============================================================================== knobs
@pytest.mark.parametrize("text, want", [
    ("0:90", (0.0, 90.0)), ("90:90", (90.0, 90.0)), ("0:0", (0.0, 0.0)), (" 45 : 135 ", (45.0, 135.0)),
    ("-0:0", (0.0, 0.0)), ("", None), (None, None), ("  ", None),
])
def test_kg_parses(text, want):
    got = pk.parse_payload_kg(text)
    assert got == want
    if got is not None:
        assert not any(math.copysign(1.0, v) < 0 for v in got)          # -0.0 normalised


@pytest.mark.parametrize("text", ["90", "a:b", "nan:1", "inf:inf", "-1:5", "5:1", "0:600", "1:2:3", "0:-0.5"])
def test_kg_refuses(text):
    with pytest.raises(ValueError, match="KRABBY_PAYLOAD_KG"):
        pk.parse_payload_kg(text)


@pytest.mark.parametrize("parser, good, bad", [
    (pk.parse_payload_zero_frac, ["0", "0.25", "1", "1.0"], ["1.5", "-0.1", "x", "", "nan"]),
    (pk.parse_payload_com_xy, ["0", "0.10", "0.3556"], ["0.36", "-0.1", "x", "", "inf"]),
    (pk.parse_payload_com_z, ["0", "0.309", "1.0"], ["-0.1", "1.5", "x", "", "nan"]),
])
def test_companion_parsers(parser, good, bad):
    for text in good:
        assert parser(text) == float(text)
    for text in bad:
        with pytest.raises(ValueError, match="KRABBY_PAYLOAD_"):
            parser(text)


def _env(**kv):
    names = {"kg": pk.PAYLOAD_KG_VAR, "zero": pk.PAYLOAD_ZERO_FRAC_VAR, "xy": pk.PAYLOAD_COM_XY_VAR,
             "z": pk.PAYLOAD_COM_Z_VAR}
    return {names[k]: v for k, v in kv.items()}


def test_parse_knobs_off_armed_and_refusals():
    assert pk.parse_payload_knobs({}) is None
    assert pk.parse_payload_knobs(_env(kg="", zero="", xy=" ", z="")) is None
    assert pk.parse_payload_knobs(_env(kg="0:90", zero="0.25", xy="0.10", z="0.309")) == D5
    assert pk.parse_payload_knobs(_env(kg="90:90", zero="0", xy="0", z="0.309")) == _level(90.0)
    with pytest.raises(ValueError, match="set but KRABBY_PAYLOAD_KG is not"):
        pk.parse_payload_knobs(_env(zero="0.25"))
    with pytest.raises(ValueError, match="KRABBY_PAYLOAD_COM_XY, KRABBY_PAYLOAD_COM_Z"):
        pk.parse_payload_knobs(_env(kg="0:90", zero="0.25"))
    with pytest.raises(ValueError, match="KRABBY_PAYLOAD_ZERO_FRAC"):
        pk.parse_payload_knobs(_env(kg="0:90", zero="", xy="0.1", z="0.309"))
    with pytest.raises(ValueError):
        pk.parse_payload_knobs(_env(kg="0:90", zero="2", xy="0.1", z="0.309"))


class _Events:
    """A cfg.events stand-in with two existing terms."""

    def __init__(self):
        self.reset_root_state = "spawn"
        self.randomize_rigid_body_mass = None


def test_unset_or_blank_writes_nothing():
    for environ in ({}, _env(kg="", zero="", xy="", z=""), {"KRABBY_OMNI_CLOCK_HIDE": "1"}):
        events = _Events()
        cfg = types.SimpleNamespace(events=events, other=1)
        before = dict(vars(cfg)), copy.deepcopy(vars(events))
        pk.apply_payload_knobs(cfg, environ)
        assert vars(cfg) == before[0] and cfg.events is events and vars(events) == before[1]
        assert not hasattr(cfg.events, pk.PAYLOAD_EVENT)
    # off never needs an events block (nothing is written)
    pk.apply_payload_knobs(types.SimpleNamespace(), {})


class _TermCfg:
    def __init__(self, func, mode, params):
        self.func, self.mode, self.params = func, mode, params


class _Entity:
    def __init__(self, name, **kw):
        self.name, self.kw = name, kw

    def __eq__(self, other):
        return isinstance(other, _Entity) and (self.name, self.kw) == (other.name, other.kw)


def _stub_isaac(monkeypatch):
    monkeypatch.setitem(sys.modules, "isaaclab.managers",
                        types.SimpleNamespace(EventTermCfg=_TermCfg, SceneEntityCfg=_Entity))
    monkeypatch.setitem(sys.modules, "parkour_tasks.crab_hex_omni_task.mdp.omni_events", oe)


def test_armed_writes_the_event_term(monkeypatch):
    _stub_isaac(monkeypatch)
    cfg = types.SimpleNamespace(events=_Events())
    pk.apply_payload_knobs(cfg, _env(kg="0:90", zero="0.25", xy="0.10", z="0.309"))
    term = getattr(cfg.events, pk.PAYLOAD_EVENT)
    assert term.func is oe.payload_reset and term.mode == "reset"
    assert term.params == {"asset_cfg": _Entity("robot", body_names="body"), "kg_range": (0.0, 90.0),
                           "zero_frac": 0.25, "com_xy": 0.10, "com_z": 0.309, "box_size": BOX}
    assert cfg.events.reset_root_state == "spawn"                         # nothing else touched
    # IsaacLab's static check (ManagerBase._resolve_common_term_cfg): params == the class __call__'s
    # arguments after (self, env, env_ids), none of them defaulted
    sig = inspect.signature(oe.payload_reset.__call__).parameters
    assert list(sig)[:3] == ["self", "env", "env_ids"]
    assert set(list(sig)[3:]) == set(term.params)
    assert all(p.default is inspect.Parameter.empty for p in sig.values())
    # idempotent: a second application replaces the term
    pk.apply_payload_knobs(cfg, _env(kg="90:90", zero="0", xy="0", z="0.309"))
    assert getattr(cfg.events, pk.PAYLOAD_EVENT).params["kg_range"] == (90.0, 90.0)
    with pytest.raises(RuntimeError, match="no events block"):
        pk.apply_payload_knobs(types.SimpleNamespace(events=None), _env(kg="0:0", zero="0", xy="0", z="0.309"))
    assert pk.PAYLOAD_EVENT == oe.PAYLOAD_EVENT_NAME


def test_omni_knobs_reexports_and_applies_before_the_command_frame():
    knobs = _load(OMNI_KNOBS_PY, "omni_knobs__task4_payload_test")
    names = {getattr(knobs, a) for a in dir(knobs) if a.endswith("_VAR")}   # run_candidates.knob_whitelist
    for var in pk.PAYLOAD_VARS:
        assert var in names, var
    for attr in ("PAYLOAD_KG_VAR", "PAYLOAD_ZERO_FRAC_VAR", "PAYLOAD_COM_XY_VAR", "PAYLOAD_COM_Z_VAR",
                 "PAYLOAD_KNOB_PARSERS"):
        assert attr in knobs.__all__, attr
    assert set(knobs.PAYLOAD_KNOB_PARSERS) == set(pk.PAYLOAD_VARS)
    src = inspect.getsource(knobs.apply_omni_knobs)
    assert src.count("apply_payload_knobs") == 1
    # after the ring/field knobs, before the motion knobs; the last three calls stay motion -> stroke -> frame
    hook = src.index("_pk.apply_payload_knobs(cfg, env)")
    assert src.index("apply_omni_ring_knobs(cfg, env)") < hook < src.index("_mk.apply_motion_knobs(cfg, env)")
    assert hook < src.index("_apply_command_frame(cfg, env)")


def test_box_constant_matches_the_measured_deck():
    dims = _load(DIMS_PY, "crab_hex_dimensions__task4_test")
    assert BOX[0] == pytest.approx(dims.BODY_SIZE_M[0], abs=1e-9)
    assert BOX[1] == pytest.approx(dims.BODY_SIZE_M[1], abs=1e-9)
    assert BOX[2] == 0.30
    # D5's CoM z: half the body height + half the load box (a 0.30 m load resting on the deck)
    assert dims.BODY_SIZE_M[2] / 2 + BOX[2] / 2 == pytest.approx(D5["com_z"], abs=5e-4)
    assert pk.PAYLOAD_COM_XY_MAX == pytest.approx(dims.BODY_SIZE_M[0] / 2, abs=1e-9)


# =============================================================================== math
def _body_box_inertia(m=BODY_MASS, size=BODY_DIMS):
    a, b, h = size
    return (m / 12 * (b * b + h * h), m / 12 * (a * a + h * h), m / 12 * (a * a + b * b))


def test_composite_matches_a_hand_calculation():
    mb, mp, z = BODY_MASS, 90.0, 0.309
    ib = _body_box_inertia()
    ip = _body_box_inertia(mp, BOX)
    M = mb + mp
    cz = mp * z / M
    want_xx = ib[0] + mb * cz ** 2 + ip[0] + mp * (z - cz) ** 2
    want_yy = ib[1] + mb * cz ** 2 + ip[1] + mp * (z - cz) ** 2
    want_zz = ib[2] + ip[2]
    m, c, i = oe.composite_mass_properties(
        torch.tensor([mb], dtype=torch.float64), torch.zeros(1, 3, dtype=torch.float64),
        torch.diag(torch.tensor(ib, dtype=torch.float64)).unsqueeze(0),
        torch.tensor([mp], dtype=torch.float64), torch.tensor([[0.0, 0.0, z]], dtype=torch.float64),
        torch.diag_embed(oe.box_inertia_diag(torch.tensor([mp]), BOX)))
    assert m.dtype == torch.float64 and float(m[0]) == pytest.approx(M, rel=1e-15)
    assert c[0].tolist() == pytest.approx([0.0, 0.0, cz], rel=1e-13, abs=1e-15)
    assert torch.diagonal(i[0]).tolist() == pytest.approx([want_xx, want_yy, want_zz], rel=1e-13)
    off = i[0] - torch.diag(torch.diagonal(i[0]))
    assert float(off.abs().max()) == 0.0
    # magnitudes, for the record: 248.76 kg, CoM up 0.1118 m, Ixx 21.0 -> 38.3 kg m^2 (Iyy 8.0 -> 18.0, Izz 26.4 -> 41.3)
    assert M == pytest.approx(248.757, abs=1e-3) and cz == pytest.approx(0.11180, abs=1e-4)
    assert (want_xx, want_yy, want_zz) == pytest.approx((38.31, 17.98, 41.30), abs=0.01)


def _spd(rng: torch.Generator, n: int) -> torch.Tensor:
    q, _ = torch.linalg.qr(torch.randn(n, 3, 3, generator=rng, dtype=torch.float64))
    d = torch.rand(n, 3, generator=rng, dtype=torch.float64) * 30 + 2
    return q @ torch.diag_embed(d) @ q.transpose(1, 2)


def _S(d):
    return (d * d).sum(-1)[:, None, None] * torch.eye(3, dtype=torch.float64) - d[:, :, None] * d[:, None, :]


def test_composite_matches_the_about_the_origin_route():
    """General case (off-centre base CoM, a rotated base inertia, an off-axis payload): Huygens-Steiner taken the
    other way round -- every part to the ORIGIN, summed, then the total moved to the composite CoM."""
    rng = torch.Generator().manual_seed(7)
    n = 64
    mb = torch.rand(n, generator=rng, dtype=torch.float64) * 5 + 156
    ma = torch.rand(n, generator=rng, dtype=torch.float64) * 270
    cb = (torch.rand(n, 3, generator=rng, dtype=torch.float64) - 0.5) * 0.02
    ca = (torch.rand(n, 3, generator=rng, dtype=torch.float64) - 0.5) * 0.2 + torch.tensor([0, 0, 0.309])
    ib, ia = _spd(rng, n), _spd(rng, n)
    m, c, i = oe.composite_mass_properties(mb, cb, ib, ma, ca, ia)
    i_origin = ib + mb[:, None, None] * _S(cb) + ia + ma[:, None, None] * _S(ca)
    want = i_origin - m[:, None, None] * _S(c)
    assert torch.allclose(m, mb + ma, rtol=0, atol=1e-12)
    assert torch.allclose(c, (mb[:, None] * cb + ma[:, None] * ca) / m[:, None], rtol=0, atol=1e-15)
    assert torch.allclose(i, want, rtol=1e-12, atol=1e-11)
    assert torch.allclose(i, i.transpose(1, 2), rtol=0, atol=1e-12)          # symmetric
    assert bool((torch.linalg.eigvalsh(i) > 0).all())                         # positive definite
    # adding nothing changes nothing (up to the float64 CoM division)
    m0, c0, i0 = oe.composite_mass_properties(mb, cb, ib, torch.zeros(n), ca, torch.zeros(n, 3, 3))
    assert torch.equal(m0, mb) and torch.allclose(c0, cb, atol=1e-17) and torch.allclose(i0, ib, atol=1e-12)


def test_box_inertia_matches_a_numerical_integration():
    """Pins which edge goes with which axis: a midpoint-rule integral of a uniform 0.7112 x 1.2192 x 0.30 box."""
    k = 80                                   # midpoint rule: relative error 1/k^2 = 1.6e-4
    axes = [((torch.arange(k, dtype=torch.float64) + 0.5) / k - 0.5) * s for s in BOX]
    x, y, z = torch.meshgrid(*axes, indexing="ij")
    m = 90.0
    w = m / x.numel()
    num = torch.tensor([(w * (y * y + z * z)).sum(), (w * (x * x + z * z)).sum(), (w * (x * x + y * y)).sum()])
    got = oe.box_inertia_diag(torch.tensor([m]), BOX)[0]
    assert torch.allclose(got, num, rtol=3e-4)
    assert float(got[2]) > float(got[0]) > float(got[1])        # z (a^2+b^2) > x (b^2+h^2) > y (a^2+h^2)


# =============================================================================== the event on a fake PhysX view
N_LINKS = 3                 # link 0 = body (the payload link), 1-2 stand in for legs (never touched)


class FakeView:
    """``ArticulationView`` mass properties: ``get_*`` REFILL and return one cached buffer each (as the real
    tensor API does), so an un-cloned capture would silently alias the live state; ``set_*`` copy only the rows
    ``indices`` names. Every call is logged."""

    def __init__(self, masses, coms, inertias):
        self.m, self.c, self.i = masses.clone(), coms.clone(), inertias.clone()
        self._bm, self._bc, self._bi = torch.empty_like(self.m), torch.empty_like(self.c), torch.empty_like(self.i)
        self.calls: list = []

    def get_masses(self):
        self.calls.append("get_masses")
        self._bm.copy_(self.m)
        return self._bm

    def get_coms(self):
        self.calls.append("get_coms")
        self._bc.copy_(self.c)
        return self._bc

    def get_inertias(self):
        self.calls.append("get_inertias")
        self._bi.copy_(self.i)
        return self._bi

    def _set(self, dst, data, indices, name):
        idx = torch.as_tensor(indices, dtype=torch.long).reshape(-1)
        assert data.dtype == torch.float32 and data.shape == dst.shape, name
        dst[idx] = data[idx]
        self.calls.append((name, idx.tolist()))

    def set_masses(self, data, indices):
        self._set(self.m, data, indices, "set_masses")

    def set_coms(self, data, indices):
        self._set(self.c, data, indices, "set_coms")

    def set_inertias(self, data, indices):
        self._set(self.i, data, indices, "set_inertias")

    def sets(self):
        return [c for c in self.calls if isinstance(c, tuple)]

    def state(self):
        return self.m.clone(), self.c.clone(), self.i.clone()


def _plant(n: int, seed: int = 3):
    """The A15+B body (158.757 kg box, CoM at its centre) with per-env 'pre-DR' noise, plus two leg links."""
    rng = torch.Generator().manual_seed(seed)
    masses = torch.zeros(n, N_LINKS)
    masses[:, 0] = BODY_MASS
    masses[:, 1:] = torch.tensor([7.413, 2.066])
    coms = torch.zeros(n, N_LINKS, 7)
    coms[:, :, 6] = 1.0                                        # identity principal axes (x, y, z, w)
    coms[:, 1, :3] = torch.tensor([0.0, -0.016, -0.014])
    inertias = torch.zeros(n, N_LINKS, 9)
    ib = torch.tensor(_body_box_inertia(), dtype=torch.float32)
    inertias[:, 0, [0, 4, 8]] = ib
    inertias[:, 1, [0, 4, 8]] = torch.tensor([0.20, 0.19, 0.007])
    inertias[:, 2, [0, 4, 8]] = torch.tensor([0.09, 0.002, 0.09])
    masses[:, 0] += torch.rand(n, generator=rng) * 1e-3          # distinct per env
    return FakeView(masses, coms, inertias)


def _startup_dr(view: FakeView, seed: int = 11):
    """What p2a's startup events do before the first reset: mass add U(-0.5, 1.5) with the inertia scaled by the
    mass ratio (IsaacLab ``randomize_rigid_body_mass``), CoM add U(+-0.01) (parkour ``randomize_rigid_body_com``)."""
    rng = torch.Generator().manual_seed(seed)
    n = view.m.shape[0]
    add = torch.rand(n, generator=rng) * 2.0 - 0.5
    ratio = (view.m[:, 0] + add) / view.m[:, 0]
    view.m[:, 0] += add
    view.i[:, 0] *= ratio[:, None]
    view.c[:, 0, :3] += (torch.rand(n, 3, generator=rng) * 2 - 1) * 0.01


def _make(n=6, spec=None, seed=18, body_ids=(0,), view=None):
    spec = dict(D5 if spec is None else spec)
    view = _plant(n) if view is None else view
    asset = types.SimpleNamespace(root_physx_view=view)
    env = types.SimpleNamespace(num_envs=n, device="cpu", cfg=types.SimpleNamespace(seed=seed),
                                scene={"robot": asset}, extras={"log": {}})
    params = {"asset_cfg": types.SimpleNamespace(name="robot", body_ids=list(body_ids)),
              **{k: spec[k] for k in ("kg_range", "zero_frac", "com_xy", "com_z")}, "box_size": BOX}
    cfg = types.SimpleNamespace(params=params)
    term = oe.payload_reset(cfg, env)
    return term, env, view, params


def _call(term, env, params, ids):
    term(env, ids, **params)


def _expected(term, ids):
    """float32 composite the term must have written for ``ids`` (from its captured base + current payload)."""
    return term._composite(torch.as_tensor(ids), term.payload_kg[ids], term.payload_com[ids], BOX)


def test_init_reads_nothing_and_capture_is_lazy_after_startup_dr():
    term, env, view, params = _make(n=5)
    assert view.calls == [] and term.base_mass is None                     # __init__ never touches PhysX
    pre = view.state()
    _startup_dr(view)                                                       # startup DR runs AFTER term init
    post = view.state()
    assert not torch.equal(pre[0], post[0])
    _call(term, env, params, torch.arange(5))
    assert view.calls[:3] == ["get_masses", "get_coms", "get_inertias"]
    assert torch.equal(term.base_mass, post[0][:, 0])                       # the POST-DR base, exactly
    assert torch.equal(term.base_com, post[1][:, 0])
    assert torch.equal(term.base_inertia, post[2][:, 0])
    # the capture is a copy: later buffer refills do not move it
    snap = term.base_mass.clone()
    view.get_masses().fill_(-1.0)
    assert torch.equal(term.base_mass, snap)
    # and the written mass is post-DR base + payload
    loaded = term.payload_kg > 0
    assert loaded.any()
    want = (post[0][:, 0].double() + term.payload_kg).float()
    assert torch.equal(view.m[loaded, 0], want[loaded])


def test_write_order_and_only_the_body_link():
    term, env, view, params = _make(n=4, spec=_level(90.0))
    before = view.state()
    _call(term, env, params, torch.arange(4))
    names = [c if isinstance(c, str) else c[0] for c in view.calls]
    assert names == ["get_masses", "get_coms", "get_inertias",               # the lazy capture
                     "get_masses", "get_coms", "get_inertias", "set_masses", "set_coms", "set_inertias"]
    for got, ref in zip(view.state(), before):
        assert torch.equal(got[:, 1:], ref[:, 1:])                          # the legs are never written
    assert torch.equal(view.c[:, 0, 3:], before[1][:, 0, 3:])               # principal-axes rotation re-sent as captured


def test_partial_resets_never_compound_and_never_touch_other_envs():
    term, env, view, params = _make(n=6)
    _startup_dr(view)
    _call(term, env, params, torch.arange(6))
    after_full = view.state()
    others = torch.tensor([0, 2, 4, 5])
    sub = torch.tensor([1, 3])
    for _ in range(10):
        _call(term, env, params, sub)
        m, c, i = _expected(term, sub)
        assert torch.equal(view.m[sub, 0], m) and torch.equal(view.c[sub, 0, :3], c) and torch.equal(view.i[sub, 0], i)
        for got, ref in zip(view.state(), after_full):
            assert torch.equal(got[others], ref[others])                    # untouched, bitwise
    # composite is always from the BASE: mass = base + this episode's payload (never base + sum of payloads)
    assert torch.allclose(view.m[sub, 0].double(), term.base_mass[sub].double() + term.payload_kg[sub], atol=2e-5)
    assert term.resets.tolist() == [1, 11, 1, 11, 1, 1]
    # a level: the 10th reset of an env lands bitwise where the 1st did
    term, env, view, params = _make(n=4, spec=_level(135.0))
    _call(term, env, params, torch.arange(4))
    first = view.state()
    for _ in range(10):
        _call(term, env, params, torch.tensor([2]))
    for got, ref in zip(view.state(), first):
        assert torch.equal(got, ref)


def test_zero_zero_is_a_true_no_op():
    for spec in (_level(0.0), {**D5, "zero_frac": 1.0}, {**_level(0.0), "com_xy": 0.1}):
        term, env, view, params = _make(n=5, spec=spec)
        _startup_dr(view)
        ref = view.state()
        for ids in (torch.arange(5), torch.tensor([0, 3]), torch.tensor([4]), None, slice(None), [1, 2]):
            _call(term, env, params, ids)
        assert view.sets() == [] and term.writes == 0                       # no PhysX write, ever
        assert [c for c in view.calls if c in ("get_masses", "get_coms", "get_inertias")] == \
            ["get_masses", "get_coms", "get_inertias"]                     # only the one capture read
        for got, want in zip(view.state(), ref):
            assert torch.equal(got, want)
        assert torch.equal(term.payload_kg, torch.zeros(5, dtype=torch.float64))
        assert torch.equal(term.payload_com, torch.zeros(5, 3, dtype=torch.float64))
        assert not term.loaded.any()


def test_a_loaded_env_drawn_zero_is_restored_bit_for_bit():
    term, env, view, params = _make(n=4, spec=_level(60.0))
    _startup_dr(view)
    base = view.state()
    _call(term, env, params, torch.arange(4))
    assert not torch.equal(view.m, base[0])
    zero = {**params, "kg_range": (0.0, 0.0)}
    term(env, torch.tensor([1, 2]), **zero)                                 # envs 1, 2 unload
    assert view.sets()[-3:] == [("set_masses", [1, 2]), ("set_coms", [1, 2]), ("set_inertias", [1, 2])]
    for got, ref in zip(view.state(), base):
        assert torch.equal(got[[1, 2]], ref[[1, 2]])                        # the captured base, exactly
    assert term.loaded.tolist() == [True, False, False, True]
    n_sets = len(view.sets())
    term(env, torch.tensor([1, 2]), **zero)                                 # already unloaded: no write
    assert len(view.sets()) == n_sets


def test_level_rows_are_exact_and_deterministic():
    for kg in (45.0, 90.0, 270.0):
        term, env, view, params = _make(n=8, spec=_level(kg), seed=1)
        _startup_dr(view)
        for ids in (torch.arange(8), torch.tensor([0, 5, 7]), torch.tensor([3])):
            _call(term, env, params, ids)
        assert torch.equal(term.payload_kg, torch.full((8,), kg, dtype=torch.float64))
        want = torch.tensor([0.0, 0.0, 0.309], dtype=torch.float64).expand(8, 3)
        assert torch.equal(term.payload_com, want)
        assert not torch.signbit(term.payload_com).any()                    # +0.0, not -0.0
        m, _, _ = _expected(term, torch.arange(8))
        assert torch.equal(view.m[:, 0], m)
        assert torch.equal(m, (term.base_mass.double() + kg).float())
        # the generator state cannot matter: another seed gives the same level row
        t2, e2, v2, p2 = _make(n=8, spec=_level(kg), seed=999)
        _startup_dr(v2)
        _call(t2, e2, p2, torch.arange(8))
        assert torch.equal(v2.m, view.m) and torch.equal(v2.c, view.c) and torch.equal(v2.i, view.i)


def test_level_composite_against_the_hand_calculation():
    term, env, view, params = _make(n=1, spec=_level(90.0))
    _call(term, env, params, torch.arange(1))
    mb = float(term.base_mass[0])
    M = mb + 90.0
    cz = 90.0 * 0.309 / M
    ib = term.base_inertia[0, [0, 4, 8]].double().tolist()
    ip = _body_box_inertia(90.0, BOX)
    want = [ib[0] + mb * cz ** 2 + ip[0] + 90 * (0.309 - cz) ** 2,
            ib[1] + mb * cz ** 2 + ip[1] + 90 * (0.309 - cz) ** 2, ib[2] + ip[2]]
    assert float(view.m[0, 0]) == pytest.approx(M, rel=1e-7)
    assert view.c[0, 0, :3].tolist() == pytest.approx([0.0, 0.0, cz], rel=1e-6, abs=1e-9)
    assert view.i[0, 0, [0, 4, 8]].tolist() == pytest.approx(want, rel=1e-6)
    assert view.i[0, 0, [1, 2, 3, 5, 6, 7]].tolist() == [0.0] * 6


def test_draws_follow_d5():
    gen = torch.Generator().manual_seed(oe.payload_seed(18))
    kg, com = oe.draw_payload(40000, gen, **D5)
    zero = kg == 0
    assert float(zero.double().mean()) == pytest.approx(0.25, abs=0.01)
    assert float(kg[~zero].min()) > 0 and float(kg[~zero].max()) < 90.0
    assert float(kg[~zero].mean()) == pytest.approx(45.0, abs=0.6)
    assert float(com[~zero, :2].abs().max()) <= 0.10
    assert float(com[~zero, :2].std()) == pytest.approx(0.10 / math.sqrt(3), rel=0.03)
    assert torch.equal(com[~zero, 2], torch.full((int((~zero).sum()),), 0.309, dtype=torch.float64))
    assert torch.equal(com[zero], torch.zeros(int(zero.sum()), 3, dtype=torch.float64))


def test_private_generator_is_deterministic_and_leaves_the_global_stream_alone():
    def run(seed):
        term, env, view, params = _make(n=7, seed=seed)
        out = []
        for ids in (torch.arange(7), torch.tensor([2, 4]), torch.tensor([0, 1, 6])):
            _call(term, env, params, ids)
            out.append(term.payload_kg.clone())
        return torch.stack(out), view.state()
    torch.manual_seed(1234)
    state = torch.get_rng_state()
    a, sa = run(18)
    assert torch.equal(torch.get_rng_state(), state)                        # the global CPU stream is untouched
    b, sb = run(18)
    assert torch.equal(a, b) and all(torch.equal(x, y) for x, y in zip(sa, sb))
    c, _ = run(19)
    assert not torch.equal(a, c)
    assert oe.payload_seed(18) != 18 and oe.payload_seed(None) == oe.payload_seed(0)
    # the salt: the payload stream is not the stream torch.manual_seed(env_seed) gives
    g = torch.Generator().manual_seed(18)
    h = torch.Generator().manual_seed(oe.payload_seed(18))
    assert not torch.equal(torch.rand(8, generator=g), torch.rand(8, generator=h))


def test_env_id_forms_and_empty_resets():
    outs = []
    for ids in (None, slice(None), list(range(4)), torch.arange(4), torch.arange(4, dtype=torch.int32)):
        term, env, view, params = _make(n=4)
        _call(term, env, params, ids)
        outs.append((term.payload_kg.clone(), view.state()))
    for kg, st in outs[1:]:
        assert torch.equal(kg, outs[0][0]) and all(torch.equal(x, y) for x, y in zip(st, outs[0][1]))
    term, env, view, params = _make(n=4)
    _call(term, env, params, torch.tensor([], dtype=torch.long))
    assert term.base_mass is not None and view.sets() == [] and term.resets.sum() == 0


def test_extras_log_and_refusals():
    term, env, view, params = _make(n=4, spec=_level(90.0))
    _call(term, env, params, torch.tensor([0, 1]))
    assert env.extras["log"] == {"Payload/kg_mean": 45.0, "Payload/loaded_frac": 0.5}
    env.extras = {}                                                         # no "log" dict: nothing is created
    _call(term, env, params, torch.tensor([2]))
    assert env.extras == {}
    with pytest.raises(ValueError, match="exactly one body"):
        _make(body_ids=(0, 1))
    with pytest.raises(ValueError, match="kg_range"):
        _make(spec={**D5, "kg_range": (5.0, 1.0)})
    with pytest.raises(ValueError, match="zero_frac"):
        _make(spec={**D5, "zero_frac": 1.5})
    with pytest.raises(ValueError, match="non-finite"):
        _make(spec={**D5, "com_z": float("nan")})
    with pytest.raises(ValueError, match="box_size"):
        oe.check_payload_params((0, 1), 0, 0, 0.3, (0.7, 1.2))


class _EventManager:
    def __init__(self, terms):
        self._terms = terms

    @property
    def active_terms(self):
        return {"startup": [], "reset": list(self._terms)}

    def get_term_cfg(self, name):
        return types.SimpleNamespace(func=self._terms[name])


def test_run_meta_record_and_lookup():
    term, env, view, params = _make(n=3, spec=_level(90.0), seed=5)
    meta = term.run_meta()
    assert meta["captured"] is False and meta["base_mass_kg"] is None and meta["total_mass_kg"] is None
    _call(term, env, params, torch.arange(3))
    meta = term.run_meta()
    assert meta["event"] == "payload_reset" and meta["captured"] is True
    assert meta["kg_range"] == [90.0, 90.0] and meta["zero_frac"] == 0.0 and meta["com_xy"] == 0.0
    assert meta["com_z"] == 0.309 and meta["box_size_m"] == list(BOX)
    assert meta["env_seed"] == 5 and meta["generator_seed"] == oe.payload_seed(5)
    assert meta["payload_kg"] == [90.0] * 3 and meta["payload_com_m"] == [[0.0, 0.0, 0.309]] * 3
    assert meta["resets_per_env"] == [1, 1, 1] and meta["writes"] == 1
    assert meta["total_mass_kg"] == [float(view.m[k, 0]) for k in range(3)]          # what PhysX holds, exactly
    assert meta["total_com_m"] == view.c[:, 0, :3].double().tolist()
    assert [t - b for t, b in zip(meta["total_mass_kg"], meta["base_mass_kg"])] == pytest.approx([90.0] * 3, abs=2e-5)
    # the harness hook: found through the event manager (also through .unwrapped); None when off / unresolved
    wrapped = types.SimpleNamespace(unwrapped=types.SimpleNamespace(event_manager=_EventManager({"payload_reset": term})))
    assert oe.payload_term(wrapped) is term and oe.payload_run_meta(wrapped) == meta
    assert oe.payload_run_meta(types.SimpleNamespace(event_manager=_EventManager({"reset_root_state": object()}))) is None
    assert oe.payload_term(types.SimpleNamespace(event_manager=_EventManager({"payload_reset": oe.payload_reset}))) is None
    assert oe.payload_run_meta(types.SimpleNamespace()) is None
