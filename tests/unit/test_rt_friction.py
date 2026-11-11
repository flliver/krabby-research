# SPDX-License-Identifier: BSD-3-Clause
"""The rough-terrain FRICTION CONTRACT (``KRABBY_RT_FRICTION=gravel:LO:HI``;
``crab_hex_omni_task/experiments/2026-11-10_rough_terrain/PLAN.md`` Design "Friction contract", step 2, "Left for
implementation" items 5 and 13).

No Isaac Sim: ``mdp/rt_friction.py`` (numpy / torch only) and ``mdp/omni_terrain_knobs.py`` are path-loaded; the event
runs on a stand-in env (a PhysX-view double holding an ``(N, S, 3)`` material tensor, the terrain importer's
``terrain_names`` / ``terrain_types`` built exactly as Isaac Lab's curriculum origins build them for the composed
20-column training mix); the privileged friction input is the REAL ``ExtremeParkourObservations._get_priv_latent``
(exec'd from ``parkour_isaaclab/envs/mdp/observations.py``) on a stand-in term.

Pinned: the realised per-env, all-shape mu (gravel ``U(LO, HI)``, others ``(b + 1) / 2`` of the event's 64 buckets
``U(0.6, 2.0)``, one bucket per env, uniform), static = dynamic, restitution 0, the observation mapping ``2 mu_eff - 1``
(TDR's ``b`` on non-gravel envs), pairing determinism across two builds with the same seed (and independence from the
env count and the global RNG), the ground side (multiply at 1.0), and the knob's refusals / default-off.
"""
from __future__ import annotations

import ast
import copy
import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[2]
OMNI = REPO / "parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task"
RT_FRICTION_PY = OMNI / "mdp/rt_friction.py"
TERRAIN_KNOBS_PY = OMNI / "mdp/omni_terrain_knobs.py"
OMNI_KNOBS_PY = OMNI / "config/crab_hex/omni_knobs.py"
OBS_PY = REPO / "parkour/parkour_isaaclab/envs/mdp/observations.py"
RUN_CANDIDATES_PY = OMNI / "experiments/tools/run_candidates.py"
STUDENT_SPEC_PY = OMNI / "config/crab_hex/omni_student_spec.py"
TRAIN_ENV = OMNI / "experiments/2026-11-10_rough_terrain/envs/RT_train.env"


def _load(path: Path, name: str) -> types.ModuleType:
    mod = sys.modules.get(name)
    if mod is None:
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
    return mod


rf = _load(RT_FRICTION_PY, "rt_friction__rt_friction_test")
tk = _load(TERRAIN_KNOBS_PY, "crab_hex_omni_terrain_knobs")

FR, NB, GR = (0.6, 2.0), 64, (0.2, 0.5)
SEED = 19
N_SHAPES = 37
#: the composed training columns (``test_rt_terrain.py`` pins this map against the real cfg and env file)
TRAIN_COLUMNS = (["parkour_flat"] * 3 + ["rt_gravel"] * 2 + ["site_field_sparse"] * 2 + ["site_field_dense"]
                 + ["rt_slopes"] * 2 + ["rt_hills"] * 2 + ["site_ledge_up"] * 2 + ["site_ledge_down"] * 2
                 + ["site_pits"] * 2 + ["site_rubble"] * 2)


# ================================================================================ stand-in env
class _View:
    def __init__(self, n, s):
        self.mat = torch.zeros(n, s, 3)
        self.mat[..., 0] = 1.0
        self.mat[..., 1] = 1.0
        self.set_calls = []

    def get_material_properties(self):
        return self.mat.clone()

    def set_material_properties(self, materials, ids):
        self.mat = materials.clone()
        self.set_calls.append(torch.as_tensor(ids).clone())


def make_env(n_envs=4096, columns=TRAIN_COLUMNS, rows=10, seed=SEED):
    """An env whose terrain importer places envs on columns exactly as ``_compute_env_origins_curriculum`` does."""
    names = np.array([[[c] for c in columns]] * rows).astype(str)                 # (R, C, 1) like the generator's
    types_ = torch.div(torch.arange(n_envs), n_envs / len(columns), rounding_mode="floor").long()
    view = _View(n_envs, N_SHAPES)
    robot = types.SimpleNamespace(root_physx_view=view)
    terrain = types.SimpleNamespace(terrain_generator_class=types.SimpleNamespace(terrain_names=names),
                                    terrain_types=types_)
    env = types.SimpleNamespace(scene=_Scene(n_envs, terrain, robot), cfg=types.SimpleNamespace(seed=seed))
    return env, view, types_


class _Scene:
    def __init__(self, n, terrain, robot):
        self.num_envs, self.terrain, self._robot = n, terrain, robot

    def __getitem__(self, name):
        assert name == "robot"
        return self._robot


ASSET = types.SimpleNamespace(name="robot")


def run_contract(env):
    rf.rt_friction_contract(env, None, ASSET, FR, NB, GR, ("rt_gravel",))


# ================================================================================ realised mu
def test_every_shape_gets_one_mu_static_equals_dynamic_and_the_distributions():
    env, view, types_ = make_env()
    run_contract(env)
    mat = view.mat
    assert len(view.set_calls) == 1 and torch.equal(view.set_calls[0], torch.arange(4096))
    mu = mat[:, 0, 0].double()
    assert torch.equal(mat[..., 0], mat[:, :1, 0].expand_as(mat[..., 0]))           # every shape, one mu
    assert torch.equal(mat[..., 0], mat[..., 1])                                     # static = dynamic
    assert (mat[..., 2] == 0).all()                                                  # restitution 0 (TDR's event's)
    gravel = np.isin(np.array(TRAIN_COLUMNS)[types_.numpy()], ["rt_gravel"])
    assert torch.equal(getattr(env, rf.RT_GRAVEL_ATTR), torch.as_tensor(gravel))
    assert gravel.sum() == int((torch.isin(types_, torch.tensor([3, 4]))).sum()) > 0       # columns 3-4 only
    g, o = mu[gravel].numpy(), mu[~gravel].numpy()
    assert g.min() >= GR[0] and g.max() <= GR[1]
    assert abs(g.mean() - 0.35) < 0.01 and abs(np.quantile(g, 0.25) - 0.275) < 0.015       # U(0.2, 0.5)
    buckets = rf.bucket_values(SEED, FR, NB)
    assert buckets.size == 64 and buckets.min() >= 0.6 and buckets.max() <= 2.0
    allowed = 0.5 * (buckets + 1.0)
    idx = np.abs(o[:, None] - allowed[None, :].astype(np.float32)).argmin(1)
    assert np.abs(o - allowed[idx].astype(np.float32)).max() < 1e-6                 # every other env: a (b + 1) / 2
    counts = np.bincount(idx, minlength=64)
    assert counts.min() > 0 and counts.max() < 3 * counts.mean()                     # uniform bucket per env
    assert o.min() >= 0.8 - 1e-6 and o.max() <= 1.5 + 1e-6                          # TDR's effective range
    assert torch.allclose(getattr(env, rf.RT_MU_ATTR), mu)


def test_bucket_picks_are_uniform_and_the_buckets_are_tdrs_distribution():
    idx = np.arange(64000)
    mu = rf.contract_mu(SEED, idx, np.zeros(idx.size, bool), FR, NB, GR)
    allowed = 0.5 * (rf.bucket_values(SEED, FR, NB) + 1.0)
    pick = np.abs(mu[:, None] - allowed[None, :]).argmin(1)
    counts = np.bincount(pick, minlength=64)
    chi2 = float(((counts - 1000.0) ** 2 / 1000.0).sum())
    assert chi2 < 120.0                                                              # df 63: p ~ 1e-5 at 120
    many = rf.bucket_values(7, FR, 200000)
    assert abs(many.mean() - 1.3) < 0.005 and many.min() >= 0.6 and many.max() <= 2.0  # U(0.6, 2.0)


def test_pairing_same_seed_same_mu_per_env_index_whatever_the_build():
    env_a, view_a, _ = make_env()
    torch.manual_seed(123)
    np.random.seed(5)
    run_contract(env_a)
    env_b, view_b, _ = make_env()
    torch.manual_seed(999)                                                           # other global streams
    np.random.seed(77)
    _ = torch.rand(1000), np.random.rand(1000)
    run_contract(env_b)
    assert torch.equal(view_a.mat, view_b.mat)
    # the env count does not enter an env's draw: index i on the same column kind gets the same mu
    small_env, small_view, small_types = make_env(n_envs=100, columns=["rt_gravel"] * 20)
    run_contract(small_env)
    big_gravel = getattr(env_a, rf.RT_GRAVEL_ATTR).numpy()
    both = np.nonzero(big_gravel[:100])[0]
    if both.size:
        assert torch.equal(small_view.mat[both, 0, 0], view_a.mat[both, 0, 0])
    alone = rf.contract_mu(SEED, [17, 3], [True, False], FR, NB, GR)
    assert alone[0] == rf.contract_mu(SEED, [17], [True], FR, NB, GR)[0]
    assert alone[1] == rf.contract_mu(SEED, [3], [False], FR, NB, GR)[0]
    other, other_view, _ = make_env(seed=SEED + 1)
    run_contract(other)
    assert not torch.equal(other_view.mat, view_a.mat)


def test_a_mixed_gravel_column_is_refused():
    env, _view, _ = make_env()
    names = env.scene.terrain.terrain_generator_class.terrain_names
    names[0, 3, 0] = "parkour_flat"                                                  # a gravel column, one flat tile
    with pytest.raises(RuntimeError, match="mix"):
        run_contract(env)


# ================================================================================ the privileged friction input
def _priv_latent_fn():
    tree = ast.parse(OBS_PY.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "ExtremeParkourObservations")
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_get_priv_latent")
    ns = {"torch": torch}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(OBS_PY), "exec"), ns)  # noqa: S102
    return ns["_get_priv_latent"], cls


def _obs_term(view, armed):
    n = view.mat.shape[0]
    data = types.SimpleNamespace(com_pos_b=torch.zeros(n, 2, 3), joint_stiffness=torch.ones(n, 4),
                                 default_joint_stiffness=torch.ones(n, 4), joint_damping=torch.ones(n, 4),
                                 default_joint_damping=torch.ones(n, 4))
    view.get_masses = lambda: torch.ones(n, 2)
    asset = types.SimpleNamespace(root_physx_view=view, data=data)
    term = types.SimpleNamespace(asset=asset, body_id=0, device="cpu")
    if armed is not None:
        term._rt_friction_obs = armed
    return term


def test_the_friction_input_reads_2mu_minus_1_when_armed_and_the_raw_mu_otherwise():
    fn, cls = _priv_latent_fn()
    env, view, types_ = make_env(n_envs=200)
    run_contract(env)
    mu = view.mat[:, 0, 0]
    col = 1 + 3                                                                      # mass (1) + com (3), then mu
    armed = fn(_obs_term(view, True))[:, col]
    assert torch.allclose(armed, 2.0 * mu - 1.0)
    gravel = getattr(env, rf.RT_GRAVEL_ATTR)
    # non-gravel envs read TDR's own input: the bucket value b
    b = torch.as_tensor(rf.bucket_values(SEED, FR, NB), dtype=torch.float32)
    assert torch.min(torch.abs(armed[~gravel][:, None] - b[None, :]), dim=1).values.max() < 1e-5
    assert armed[gravel].min() >= 2 * GR[0] - 1 - 1e-6 and armed[gravel].max() <= 2 * GR[1] - 1 + 1e-6
    off = fn(_obs_term(view, False))[:, col]
    assert torch.equal(off, mu)
    legacy = fn(_obs_term(view, None))[:, col]                                       # a term built before the flag
    assert torch.equal(legacy, mu)
    init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
    src = ast.unparse(init)
    assert "self._rt_friction_obs = getattr(env.cfg, 'rt_friction', None) is not None" in src
    assert rf.RT_FRICTION_ATTR == tk.RT_FRICTION_ATTR == "rt_friction"


# ================================================================================ the knob
def material(**kw):
    m = types.SimpleNamespace(static_friction=1.0, dynamic_friction=1.0, restitution=0.0,
                              friction_combine_mode="average", restitution_combine_mode="average")
    m.__dict__.update(kw)
    return m


def knob_cfg():
    return types.SimpleNamespace(
        scene=types.SimpleNamespace(terrain=types.SimpleNamespace(physics_material=material())),
        sim=types.SimpleNamespace(physics_material=material()),
        events=types.SimpleNamespace(physics_material=types.SimpleNamespace(
            func=types.SimpleNamespace(__name__="randomize_rigid_body_material"), mode="startup",
            params={"asset_cfg": ASSET, "friction_range": FR, "num_buckets": NB})))


RT_ENV = {"KRABBY_FLAT_TERRAIN_GEOM": "site", "KRABBY_SITE_LADDER": "rt", "KRABBY_RT_FRICTION": "gravel:0.2:0.5"}


def test_parse_rt_friction():
    assert tk.parse_rt_friction(None) is None and tk.parse_rt_friction("  ") is None
    assert tk.parse_rt_friction(" gravel:0.2:0.5 ") == {"kind": "gravel", "lo": 0.2, "hi": 0.5}
    assert tk.RT_KNOB_PARSERS == {"KRABBY_RT_FRICTION": tk.parse_rt_friction}
    for bad in ("0.2:0.5", "pits:0.2:0.5", "gravel:0.5:0.2", "gravel:0.0:0.5", "gravel:0.2:2.5", "gravel:x:0.5",
                "gravel:0.2", "gravel:0.2:0.5:1", "gravel:nan:0.5"):
        with pytest.raises(ValueError, match="KRABBY_RT_FRICTION"):
            tk.parse_rt_friction(bad)


def test_the_knob_arms_the_contract_on_both_sides():
    cfg = knob_cfg()
    asset_cfg = cfg.events.physics_material.params["asset_cfg"]
    spec = tk.apply_rt_friction_knobs(cfg, RT_ENV, exclusive_with=("KRABBY_STRESS_MU",),
                                      contract_func=rf.rt_friction_contract)
    assert spec == {"kind": "gravel", "lo": 0.2, "hi": 0.5}
    ev = cfg.events.physics_material
    assert ev.func is rf.rt_friction_contract and ev.mode == "startup"
    assert ev.params == {"asset_cfg": asset_cfg, "friction_range": FR, "num_buckets": NB, "gravel_range": GR,
                         "gravel_sub_terrains": ("rt_gravel",)}
    for mat in (cfg.scene.terrain.physics_material, cfg.sim.physics_material):
        assert mat.friction_combine_mode == "multiply" and mat.static_friction == mat.dynamic_friction == 1.0
        assert mat.restitution == 0.0 and mat.restitution_combine_mode == "average"          # restitution untouched
    assert getattr(cfg, tk.RT_FRICTION_ATTR) == {"gravel_range": [0.2, 0.5], "friction_range": [0.6, 2.0],
                                                 "num_buckets": 64, "obs": "2*mu_eff-1"}
    # the event params are exactly the contract function's signature
    env, view, _ = make_env(n_envs=40)
    ev.func(env, None, **ev.params)
    assert view.set_calls


def test_the_knob_off_writes_nothing_and_prints_nothing(capsys):
    for env in ({}, {"KRABBY_RT_FRICTION": ""}, {"KRABBY_FLAT_TERRAIN_GEOM": "field"},
                {"KRABBY_FLAT_TERRAIN_GEOM": "site", "KRABBY_SITE_LADDER": "rt"}):
        cfg = knob_cfg()
        before = copy.deepcopy(cfg)
        assert tk.apply_rt_friction_knobs(cfg, env, exclusive_with=("KRABBY_STRESS_MU",)) is None
        assert repr(vars(cfg)) == repr(vars(before)) and not hasattr(cfg, tk.RT_FRICTION_ATTR)
    assert capsys.readouterr().out == ""


def test_the_knob_refusals():
    for env in ({"KRABBY_RT_FRICTION": "gravel:0.2:0.5", "KRABBY_FLAT_TERRAIN_GEOM": "field"},
                {"KRABBY_RT_FRICTION": "gravel:0.2:0.5", "KRABBY_FLAT_TERRAIN_GEOM": "site"},
                {"KRABBY_RT_FRICTION": "gravel:0.2:0.5", "KRABBY_FLAT_TERRAIN_GEOM": "site",
                 "KRABBY_SITE_LADDER": "b4"}):
        with pytest.raises(RuntimeError, match="needs KRABBY_FLAT_TERRAIN_GEOM=site"):
            tk.apply_rt_friction_knobs(knob_cfg(), env)
    with pytest.raises(ValueError, match="exclusive"):
        tk.apply_rt_friction_knobs(knob_cfg(), dict(RT_ENV, KRABBY_STRESS_MU="1.2"), exclusive_with=("KRABBY_STRESS_MU",))
    cfg = knob_cfg()
    cfg.events.physics_material = None
    with pytest.raises(RuntimeError, match="is off"):
        tk.apply_rt_friction_knobs(cfg, RT_ENV, contract_func=rf.rt_friction_contract)
    cfg = knob_cfg()
    tk.apply_rt_friction_knobs(cfg, RT_ENV, contract_func=rf.rt_friction_contract)
    with pytest.raises(RuntimeError, match="not TDR's"):                             # applied twice
        tk.apply_rt_friction_knobs(cfg, RT_ENV, contract_func=rf.rt_friction_contract)


def test_omni_knobs_applies_the_contract_once_before_the_stress_knobs_and_reexports_it():
    fn = next(n for n in ast.parse(OMNI_KNOBS_PY.read_text()).body
              if isinstance(n, ast.FunctionDef) and n.name == "apply_omni_knobs")
    calls = [ast.unparse(s) for s in fn.body if isinstance(s, ast.Expr)]
    hook = "_tk.apply_rt_friction_knobs(cfg, env, exclusive_with=(_srk.MU_VAR,))"
    assert calls.count(hook) == 1
    assert calls.index("_pk.apply_payload_knobs(cfg, env)") < calls.index(hook) \
        < calls.index("_srk.apply_stress_knobs(cfg, env)")
    src = OMNI_KNOBS_PY.read_text()
    assert "RT_FRICTION_VAR = _tk.RT_FRICTION_VAR" in src and "RT_KNOB_PARSERS = _tk.RT_KNOB_PARSERS" in src


def test_the_knob_is_whitelisted_parsed_dump_checked_and_never_stripped():
    rc = _load(RUN_CANDIDATES_PY, "run_candidates__rt_friction_test")
    var = "KRABBY_RT_FRICTION"
    assert var in rc.knob_whitelist() and var not in rc.FORBIDDEN_OVERLAY_KEYS
    assert rc.knob_parsers()[var]("gravel:0.2:0.5") == {"kind": "gravel", "lo": 0.2, "hi": 0.5}
    assert var in rc.OVERLAY_DUMP_CHECKS and var not in rc.AGENT_DUMP_KEYS
    assert var not in tk.SITE_UNPORTED_VARS
    assert var not in _student_refused()
    assert "KRABBY_RT_FRICTION=gravel:0.2:0.5" in TRAIN_ENV.read_text()


def _student_refused() -> tuple:
    tree = ast.parse(STUDENT_SPEC_PY.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.Assign)
                and getattr(n.targets[0], "id", None) == "STUDENT_RUNNER_KNOBS_REFUSED")
    return tuple(ast.literal_eval(node.value))


def test_the_dump_check_reads_the_contract():
    rc = _load(RUN_CANDIDATES_PY, "run_candidates__rt_friction_test")
    on = ("scene:\n  terrain:\n    physics_material:\n      static_friction: 1.0\n      dynamic_friction: 1.0\n"
          "      friction_combine_mode: multiply\n"
          "events:\n  physics_material:\n    func: parkour_tasks.crab_hex_omni_task.mdp.rt_friction:rt_friction_contract\n"
          "    params:\n      gravel_range: !!python/tuple\n      - 0.2\n      - 0.5\n"
          "rt_friction:\n  gravel_range:\n  - 0.2\n  - 0.5\n")
    off = ("scene:\n  terrain:\n    physics_material:\n      friction_combine_mode: average\n"
           "events:\n  physics_material:\n    func: parkour_isaaclab.envs.mdp.events:randomize_rigid_body_material\n")
    check = rc.OVERLAY_DUMP_CHECKS["KRABBY_RT_FRICTION"]
    assert check(on, "gravel:0.2:0.5") and not check(on, "gravel:0.2:0.4") and not check(on, "")
    assert check(off, "") and not check(off, "gravel:0.2:0.5")
