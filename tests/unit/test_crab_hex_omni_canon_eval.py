"""Eval-time mirror canonicalization (``eval_crab_hex_gait.py --canon``; side-bisect ANALYSIS step 1).

When an env's flag is on the teacher is fed ``M_y(obs)`` and its action is mapped back with ``M_y``.
The harness imports Isaac at module level, so its pure canon helpers, the ``--canon`` argparse
lines and the untouched teacher branch are lifted out of the file with ``ast`` (the harness-test
pattern ``test_crab_hex_gait_schedule.py`` uses) and exercised on CPU:

* the omni ``build_mirrors`` pair is an involution on the shipped obs (1149) / action (18) shapes;
* ``_canon_hyst_update`` / ``_canon_wave_flags`` behave as specified;
* ``_canon_policy_step`` with every flag off is bit-identical to the harness's own teacher branch,
  mirrors exactly per row when on, and never writes the raw observation;
* with an exactly equivariant policy, canonicalization changes nothing -- the map is self-consistent;
* ``--canon`` / ``--canon-hyst`` accept and refuse as specified; canon-only state is guarded;
* the reversal manifest loads, validates and compiles.

Every module is loaded by FILE PATH under a private name (collection order must not matter).
"""

from __future__ import annotations

import argparse
import ast
import importlib.util
import math
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[2]
TASKS = REPO / "parkour/parkour_tasks/parkour_tasks"
SCRIPTS_DIR = TASKS / "crab_hex_forward_task/scripts"
EVAL_SCRIPT = SCRIPTS_DIR / "eval_crab_hex_gait.py"
OMNI_MIRROR_PY = TASKS / "crab_hex_omni_task/mdp/crab_hex_omni_mirror.py"
REVERSAL_MANIFEST = TASKS / "crab_hex_omni_task/experiments/2026-09-28_side_bisect/scenarios_side_reversal.yaml"
SIDE_LR_MANIFEST = TASKS / "crab_hex_omni_task/experiments/2026-09-28_side_bisect/scenarios_side_lr.yaml"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from gait_eval import schedule as S  # noqa: E402


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


omni = _load(OMNI_MIRROR_PY, "crab_hex_omni_mirror__canon_test")

# ------------------------------------------------------------------ the harness, lifted with ast
TREE = ast.parse(EVAL_SCRIPT.read_text())
CANON_FUNCS = (
    "_canon_check_args",
    "_canon_hyst_update",
    "_canon_lateral",
    "_canon_wave_flags",
    "_canon_policy_step",
    "_canon_flips",
)


def _lift_helpers() -> types.SimpleNamespace:
    """The CANON_* constants and the canon helper functions, exec'd with only torch/numpy in scope."""
    nodes = []
    for node in TREE.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id.startswith("CANON_") for t in node.targets
        ):
            nodes.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name in CANON_FUNCS:
            nodes.append(node)
    assert {n.name for n in nodes if isinstance(n, ast.FunctionDef)} == set(CANON_FUNCS)
    ns: dict = {"torch": torch, "np": np}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(EVAL_SCRIPT), "exec"), ns)  # noqa: S102
    return types.SimpleNamespace(**{k: v for k, v in ns.items() if not k.startswith("__")})


H = _lift_helpers()


def _policy_if_chain() -> ast.If:
    """The ``if args_cli.zero_actions: ... elif is_student: ... elif canon_on: ... else:`` chain."""
    for node in ast.walk(TREE):
        if isinstance(node, ast.If) and ast.unparse(node.test) == "args_cli.zero_actions":
            return node
    raise AssertionError("policy branch not found in the harness")


def _teacher_branch_source() -> tuple[str, str]:
    """(canon branch source, untouched teacher ``else`` source)."""
    top = _policy_if_chain()
    student = top.orelse[0]
    assert isinstance(student, ast.If) and ast.unparse(student.test) == "is_student"
    canon = student.orelse[0]
    assert isinstance(canon, ast.If) and ast.unparse(canon.test) == "canon_on"
    teacher = canon.orelse
    return (
        "\n".join(ast.unparse(s) for s in canon.body),
        "\n".join(ast.unparse(s) for s in teacher),
    )


# ------------------------------------------------------------------ real-shaped fake env
LEGS = ["FL", "FR", "ML", "MR", "RL", "RR"]
JOINT_NAMES = [leg + jt for jt in ("_Body_CamShaft_RevoluteJoint", "_Body_Hip_RevoluteJoint",
                                   "_Hip_Femur_RevoluteJoint", "_Femur_Tibia_RevoluteJoint")
               for leg in LEGS]
ACTION_NAMES = [leg + jt for jt in ("_Body_CamShaft_RevoluteJoint", "_Hip_Femur_RevoluteJoint",
                                    "_Femur_Tibia_RevoluteJoint") for leg in LEGS]
CONTACT_NAMES = [leg + "_Footpad" for leg in LEGS]
N_PROP = 17 + 24 + 24 + 18 + 6            # 89
N_SCAN = 9 * 12                           # 108
N_PRIV_EXPLICIT = 9
OBS_DIM = N_PROP + N_SCAN + 9 + 5 + 48 + 10 * N_PROP   # 1149
N_ACT = 18


def _fake_env(encoding: str = "raw"):
    pattern = types.SimpleNamespace(size=(2.4, 3.4), resolution=0.3, ordering="xy")
    scanner = types.SimpleNamespace(cfg=types.SimpleNamespace(pattern_cfg=pattern))
    contacts = types.SimpleNamespace(body_names=CONTACT_NAMES + ["body"])
    robot = types.SimpleNamespace(joint_names=JOINT_NAMES)

    class _Scene:
        sensors = {"contact_forces": contacts, "height_scanner": scanner}

        def __getitem__(self, key):
            assert key == "robot"
            return robot

    action_term = types.SimpleNamespace(_joint_names=ACTION_NAMES)
    obs_term = types.SimpleNamespace(params={"history_length": 10, "facing_obs_encoding": encoding})
    return types.SimpleNamespace(
        scene=_Scene(),
        action_manager=types.SimpleNamespace(get_term=lambda name: action_term),
        device="cpu",
        cfg=types.SimpleNamespace(
            observations=types.SimpleNamespace(
                policy=types.SimpleNamespace(extreme_parkour_observations=obs_term)
            )
        ),
    )


@pytest.fixture(scope="module")
def lr():
    lr_map, _ = omni.build_mirrors(_fake_env(), OBS_DIM)
    assert lr_map.obs_dim == OBS_DIM
    return lr_map


# ------------------------------------------------------------------ 1. the maps are involutions
@pytest.mark.parametrize("encoding", ["raw", "sincos"])
def test_the_omni_mirror_pair_is_an_involution_on_the_shipped_shapes(encoding):
    lr_map, fa_map = omni.build_mirrors(_fake_env(encoding), OBS_DIM)
    g = torch.Generator().manual_seed(0)
    obs = torch.randn(7, OBS_DIM, generator=g)
    act = torch.randn(7, N_ACT, generator=g)
    for m in (lr_map, fa_map):
        assert m.obs_dim == OBS_DIM
        assert torch.equal(m.mirror_obs(m.mirror_obs(obs)), obs)
        assert torch.equal(m.mirror_actions(m.mirror_actions(act)), act)
        # a genuine reflection, not the identity
        assert not torch.equal(m.mirror_obs(obs), obs)
        assert not torch.equal(m.mirror_actions(act), act)
    # the canon map is M_y: the command slots 8/9 (vx_b, vy_b) keep vx and negate vy
    x = torch.zeros(1, OBS_DIM)
    x[0, 8], x[0, 9] = 0.1, 0.22
    y = lr_map.mirror_obs(x)
    assert (y[0, 8].item(), y[0, 9].item()) == pytest.approx((0.1, -0.22))


# ------------------------------------------------------------------ 2. hysteresis
def test_hysteresis_initializes_from_the_first_command_sign():
    vy = torch.tensor([-0.22, 0.22, 0.0, -0.01, 0.05])
    assert H._canon_hyst_update(None, vy, 0.08).tolist() == [True, False, False, True, False]


def test_hysteresis_flips_only_outside_the_band_and_holds_inside():
    h = 0.08
    prev = torch.tensor([True, True, False, False, True, False])
    vy = torch.tensor([0.07, 0.09, -0.07, -0.09, h, -h])
    # inside (or exactly on) the band: hold; beyond +h: off; beyond -h: on
    assert H._canon_hyst_update(prev, vy, h).tolist() == [True, False, False, True, True, False]


def test_hysteresis_over_a_reversal_sequence():
    h = 0.08
    seq = [0.0, 0.0, 0.1, 0.22, 0.05, -0.05, -0.079, -0.081, -0.22, -0.02, 0.02, 0.0801, 0.22]
    want = [False, False, False, False, False, False, False, True, True, True, True, False, False]
    flag, got = None, []
    for v in seq:
        flag = H._canon_hyst_update(flag, torch.tensor([v]), h)
        got.append(bool(flag[0]))
    assert got == want
    assert flag.dtype == torch.bool


# ------------------------------------------------------------------ 3. wave flags
def test_wave_flags_fill_undecided_steps_from_the_next_lateral_hold():
    lat = np.array([[0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.22, -0.22, 0.0],
                    [-0.22, 0.22, -0.22, 0.0],
                    [0.0, 0.0, 1e-17, 0.0],
                    [0.22, 0.0, 0.0, 0.0]])
    got = H._canon_wave_flags(lat)
    assert got.dtype == bool and got.shape == lat.shape
    # env 0: settle takes the next hold (-y -> on) until the +y hold, whose flag then trails
    assert got[:, 0].tolist() == [True, True, True, False, False]
    assert got[:, 1].tolist() == [False] * 5          # +y only
    assert got[:, 2].tolist() == [True] * 5            # -y only; 1e-17 (sin(pi) noise) is undecided
    assert got[:, 3].tolist() == [False] * 5           # never lateral -> off


def test_wave_flags_on_the_real_suite_waves_are_constant_per_episode():
    scenario = [s for s in S.load_manifest(SIDE_LR_MANIFEST)[0] if s.id == "side22_25lr__omni"][0]
    want = {"stand": False, "lat_l_022": False, "lat_r_022": True, "lat_l_025": False, "lat_r_025": True}
    for wave in range(scenario.waves):
        comp = S.compile_wave(scenario, wave, dt=0.02, num_envs=3)
        flags = H._canon_wave_flags(H._canon_lateral(comp))
        label = comp.hold_labels[wave + 1]
        assert flags.shape == (comp.n_steps, 3)
        # the settle (and prologue) is mirrored exactly like the hold it precedes
        assert (flags == want[label]).all(), label


def test_lateral_reads_body_vy_on_body_rows():
    comp = types.SimpleNamespace(world_dir=None, cmd=np.array([[[0.3, -0.2, 0.0]], [[0.3, 0.2, 0.0]]]))
    assert H._canon_lateral(comp)[:, 0].tolist() == pytest.approx([-0.2, 0.2])


def test_flips_count_only_recorded_steps():
    f = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [1, 0]], dtype=bool)
    assert H._canon_flips(f, [5, 3]) == [3, 1]
    assert H._canon_flips(f, [1, 0]) == [0, 0]


# ------------------------------------------------------------------ 4. the per-step hook
class _Estimator:
    def __init__(self, seed=1, zero=False):
        g = torch.Generator().manual_seed(seed)
        self.w = torch.zeros(N_PROP, N_PRIV_EXPLICIT) if zero else torch.randn(N_PROP, N_PRIV_EXPLICIT, generator=g)
        self.seen = []

    def inference(self, prop):
        self.seen.append(prop.clone())
        return prop @ self.w


class _Policy:
    def __init__(self, seed=2):
        g = torch.Generator().manual_seed(seed)
        self.w = torch.randn(OBS_DIM, N_ACT, generator=g) / 30.0
        self.seen = []

    def __call__(self, obs, hist_encoding=True):
        assert hist_encoding is True
        self.seen.append(obs.clone())
        return torch.tanh(obs @ self.w)


def _obs(n=6, seed=3):
    return torch.randn(n, OBS_DIM, generator=torch.Generator().manual_seed(seed))


PRIV = (N_PROP + N_SCAN, N_PROP + N_SCAN + N_PRIV_EXPLICIT)


def _run_teacher_branch(obs, estimator, policy):
    """The harness's own (untouched) teacher ``else`` branch, exec'd verbatim."""
    _, src = _teacher_branch_source()
    ns = {"obs": obs, "estimator": estimator, "policy": policy,
          "num_prop": N_PROP, "num_scan": N_SCAN, "num_priv_explicit": N_PRIV_EXPLICIT}
    exec(src, ns)  # noqa: S102
    return ns["actions"]


def test_the_teacher_branch_is_textually_unchanged():
    _, src = _teacher_branch_source()
    assert src == (
        "obs[:, num_prop + num_scan:num_prop + num_scan + num_priv_explicit] = "
        "estimator.inference(obs[:, :num_prop])\n"
        "actions = policy(obs, hist_encoding=True)"
    )


def test_canon_with_every_flag_off_is_bit_identical_to_the_teacher_branch(lr):
    obs = _obs()
    raw = obs.clone()
    est, pol = _Estimator(), _Policy()
    want = _run_teacher_branch(obs.clone(), est, pol)
    got = H._canon_policy_step(obs, torch.zeros(obs.shape[0], dtype=torch.bool), lr, est, pol, *PRIV, N_PROP)
    assert torch.equal(got, want)
    assert torch.equal(obs, raw), "the raw observation is never written"


def test_canon_on_feeds_the_mirrored_obs_and_mirrors_the_action_back(lr):
    obs = _obs()
    flag = torch.tensor([True, False, True, True, False, False])
    est, pol = _Estimator(), _Policy()
    got = H._canon_policy_step(obs, flag, lr, est, pol, *PRIV, N_PROP)
    fed = pol.seen[-1]
    m_obs = lr.mirror_obs(obs)
    for i, on in enumerate(flag.tolist()):
        src = m_obs[i] if on else obs[i]
        # everything but the estimator slice is the (mirrored) observation ...
        keep = torch.ones(OBS_DIM, dtype=torch.bool)
        keep[PRIV[0]:PRIV[1]] = False
        assert torch.equal(fed[i][keep], src[keep])
        # ... and the estimator ran on the (mirrored) proprio
        assert torch.equal(est.seen[-1][i], src[:N_PROP])
        assert torch.allclose(fed[i][PRIV[0]:PRIV[1]], src[:N_PROP] @ est.w)
        # each row equals the single-env computation, mirrored back when on
        solo = _run_teacher_branch(src.unsqueeze(0).clone(), _Estimator(), _Policy())[0]
        expect = lr.mirror_actions(solo) if on else solo
        assert torch.allclose(got[i], expect, atol=1e-5)  # batch vs single-row GEMM rounding


def test_an_equivariant_policy_is_unchanged_by_canonicalization(lr):
    """pi(o) = f(o) + M f(M o) satisfies pi(M o) = M pi(o), so the canon flag must not matter."""
    base = _Policy()

    def equivariant(o, hist_encoding=True):
        return base(o) + lr.mirror_actions(base(lr.mirror_obs(o)))

    obs = _obs(8)
    est = _Estimator(zero=True)     # zero estimator: the priv slice is 0 in both frames
    off = H._canon_policy_step(obs, torch.zeros(8, dtype=torch.bool), lr, est, equivariant, *PRIV, N_PROP)
    on = H._canon_policy_step(obs, torch.ones(8, dtype=torch.bool), lr, est, equivariant, *PRIV, N_PROP)
    assert torch.allclose(on, off, atol=1e-6)


def test_the_canon_branch_calls_the_hook_and_never_reads_the_flag_from_obs():
    src, _ = _teacher_branch_source()
    assert "actions = _canon_policy_step(obs, canon_flag, canon_lr, estimator, policy," in src
    assert "cmd_term.vel_command_b[:, 1]" in src
    assert "obs[" not in src, "the flag must come from the command term, never from the observation"
    assert "buf['canon_flag'][t] = canon_flag" in src


def test_canon_only_state_is_guarded_so_canon_off_keys_are_unchanged():
    """Every write of buf['canon_flag'] / run_meta['canon'] sits under an ``if`` on ``canon_on``."""
    parents = {child: node for node in ast.walk(TREE) for child in ast.iter_child_nodes(node)}
    sites = []
    for node in ast.walk(TREE):
        if not isinstance(node, ast.Subscript) or not isinstance(node.ctx, ast.Store):
            continue
        text = ast.unparse(node)
        if not (text.startswith("buf['canon_flag']") or text == "run_meta['canon']"):
            continue
        guarded, cur = False, node
        while cur in parents:
            cur = parents[cur]
            if isinstance(cur, ast.If) and ast.unparse(cur.test) == "canon_on":
                guarded = True
                break
        sites.append((text, guarded))
    assert {t for t, _ in sites} == {"buf['canon_flag']", "buf['canon_flag'][t]", "run_meta['canon']"}
    assert all(ok for _, ok in sites), sites


# ------------------------------------------------------------------ 5. CLI
def _canon_parser() -> argparse.ArgumentParser:
    """A fresh parser carrying exactly the harness's --canon / --canon-hyst add_argument calls."""
    calls = [
        node for node in TREE.body
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
        and ast.unparse(node.value.func) == "parser.add_argument"
        and isinstance(node.value.args[0], ast.Constant) and str(node.value.args[0].value).startswith("--canon")
    ]
    assert [c.value.args[0].value for c in calls] == ["--canon", "--canon-hyst"]
    ns = {"parser": argparse.ArgumentParser(exit_on_error=False)}
    exec(compile(ast.Module(body=calls, type_ignores=[]), str(EVAL_SCRIPT), "exec"), ns)  # noqa: S102
    return ns["parser"]


def test_argparse_defaults_and_accepted_values():
    p = _canon_parser()
    a = p.parse_args([])
    assert (a.canon, a.canon_hyst) == ("off", 0.08)
    for mode in ("off", "wave", "cmd"):
        assert p.parse_args(["--canon", mode]).canon == mode
    assert p.parse_args(["--canon", "cmd", "--canon-hyst", "0.05"]).canon_hyst == 0.05


@pytest.mark.parametrize("bad", [["--canon", "on"], ["--canon", "both"], ["--canon-hyst", "x"]])
def test_argparse_refuses_bad_values(bad):
    p = _canon_parser()
    with pytest.raises((SystemExit, argparse.ArgumentError)):
        p.parse_args(bad)


def test_check_args_refuses_non_omni_tasks_and_bad_combinations():
    omni_task = "Isaac-Crab-Hex-Omni-Walk-Play-v0"
    # off is a no-op on any task
    H._canon_check_args("off", 0.08, "Isaac-Crab-Hex-Walk-Play-v0", zero_actions=True)
    for mode in ("wave", "cmd"):
        H._canon_check_args(mode, 0.08, omni_task, zero_actions=False)
        for task in ("Isaac-Crab-Hex-Walk-Play-v0", "Isaac-Crab-Hex-Side-Walk-Play-v0", ""):
            with pytest.raises(SystemExit, match="omni task"):
                H._canon_check_args(mode, 0.08, task, zero_actions=False)
        with pytest.raises(SystemExit, match="zero-actions"):
            H._canon_check_args(mode, 0.08, omni_task, zero_actions=True)
        with pytest.raises(SystemExit, match="canon-hyst"):
            H._canon_check_args(mode, -0.01, omni_task, zero_actions=False)
    with pytest.raises(SystemExit):
        H._canon_check_args("on", 0.08, omni_task, zero_actions=False)
    # main() runs the check before building anything
    assert "_canon_check_args(canon_mode, args_cli.canon_hyst, scenario.task" in EVAL_SCRIPT.read_text()


# ------------------------------------------------------------------ 6. the reversal manifest
def test_reversal_manifest_loads_validates_and_compiles():
    scenarios, _ = S.load_manifest(REVERSAL_MANIFEST)
    by_id = {s.id: s for s in scenarios}
    assert set(by_id) == {"rev_direct__omni", "rev_pause__omni"}
    half = math.pi / 2
    shapes = {
        "rev_direct__omni": [("stand", 0.0, 0.0, 2.0), ("lat_l_1", 0.22, half, 8.0),
                             ("lat_r_2", 0.22, -half, 8.0), ("lat_l_3", 0.22, half, 8.0)],
        "rev_pause__omni": [("stand", 0.0, 0.0, 2.0), ("lat_l_1", 0.22, half, 8.0),
                            ("pause_1", 0.0, 0.0, 1.0), ("lat_r_2", 0.22, -half, 8.0),
                            ("pause_2", 0.0, 0.0, 1.0), ("lat_l_3", 0.22, half, 8.0)],
    }
    for sid, want in shapes.items():
        sc = by_id[sid]
        S.validate_scenario(sc)
        assert not S.uses_waves(sc) and sc.kind == "schedule" and sc.episodes == 100
        assert sc.command_frame == "world" and sc.spawn_yaw_rad() == 0.0
        assert sc.task == "Isaac-Crab-Hex-Omni-Walk-Play-v0" and not sc.env_vars
        got = [(h.label, h.speed, h.dir, h.face, h.hold_s) for h in sc.schedule]
        assert [(g[0], g[4]) for g in got] == [(w[0], w[3]) for w in want]
        for g, w in zip(got, want):
            assert g[1] == pytest.approx(w[1]) and g[2] == pytest.approx(w[2], abs=1e-7) and g[3] == 0.0
        assert sc.episode_length_s >= sc.total_duration_s
        comp = S.compile_schedule(sc, dt=0.02, num_envs=2)
        assert comp.n_steps == round(sc.total_duration_s / 0.02)
        # wave-mode flags: off through the first stand and both lat_l holds, on through lat_r; a
        # pause takes the NEXT lateral hold's flag
        flags = H._canon_wave_flags(H._canon_lateral(comp))[:, 0]
        by_label = {lab: set(flags[comp.segment_id[:, 0] == i].tolist())
                    for i, lab in enumerate(comp.hold_labels)}
        assert by_label["stand"] == {False} and by_label["lat_l_1"] == {False}
        assert by_label["lat_r_2"] == {True} and by_label["lat_l_3"] == {False}
        if sid == "rev_pause__omni":
            assert by_label["pause_1"] == {True} and by_label["pause_2"] == {False}
