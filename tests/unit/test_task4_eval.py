"""Task 4 pipeline step 5 ("E1", 2026-10-16): the eval commit, on CPU, without Isaac.

``crab_hex_omni_task/experiments/2026-10-16_task4_pipeline`` (PLAN step 5, DESIGN §4 E1 / B8 / B9). Pinned here,
each a quiet failure otherwise:

* **the omni student branch** of ``eval_crab_hex_gait.py`` -- it calls the ONE shared step
  (``OmniStudentStep.from_runner`` / ``.act`` / ``.reset``, the trainer's and play.py's), never a second forward
  pass; the routing (role student AND the task's runner cfg flag, cross-checked against the checkpoint's
  ``params/agent.yaml``, whose ``zero_depth`` is honoured); ``--policy-role teacher`` on Student-Play is untouched;
  the forward yaw-probe route is refused; and **every pre-E1 statement of ``main()`` is AST-identical to the
  baseline commit** once the E1 statements are removed (forward student, teacher, canon, adapters);
* **the student runner's knob policy** -- a Student(-Play) row whose environment sets a refused runner knob is
  gate-0 INVALID before the agent cfg is built, with the remedy named; none of the refused knobs is an eval-carry
  key, and the recorded S1 eval env sets none;
* **``run_meta["payload"]``** -- the exact key, ``null`` when the event is off, the real ``payload_reset`` record
  read AFTER the reset that applied it (a pre-reset read is refused), ``by_wave`` on suite rows, the read placed
  inside the wave loop after the reset and before step 0;
* **opt-in ``metric_class``** -- a ``metric_class: translate`` category gets exactly the metrics a translation
  category gets; without it nothing changes: every committed manifest parses, compiles, records and SCORES
  bit-identically to the baseline commit's ``gait_eval`` (run side by side in subprocesses);
* **the sweep row file** ``rows/scenarios_pl_body.yaml`` -- 16 translate bins d000 ... d338 at 22.5 deg, |v| 0.22,
  wz 0, metric_class translate, the envelope block verbatim, and a schedule that builds and scores.
"""

from __future__ import annotations

import ast
import copy
import dataclasses
import hashlib
import importlib.util
import json
import math
import subprocess
import sys
import textwrap
import types
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

REPO = Path(__file__).resolve().parents[2]
UNIT = Path(__file__).resolve().parent
TASKS = REPO / "parkour/parkour_tasks/parkour_tasks"
OMNI = TASKS / "crab_hex_omni_task"
SCRIPTS_DIR = TASKS / "crab_hex_forward_task/scripts"
EVAL_SCRIPT = SCRIPTS_DIR / "eval_crab_hex_gait.py"
GE_DIR = SCRIPTS_DIR / "gait_eval"
PL = OMNI / "experiments/2026-10-16_task4_pipeline"
SWEEP = PL / "rows/scenarios_pl_body.yaml"
BODY_MANIFEST = OMNI / "experiments/eval/scenarios_omni_body.yaml"
OMNI_MANIFEST = OMNI / "experiments/eval/scenarios_omni.yaml"
FWD_MANIFEST = TASKS / "crab_hex_forward_task/experiments/eval/scenarios_v1.yaml"   # the harness default
SPEC_PY = OMNI / "config/crab_hex/omni_student_spec.py"
INIT_PY = OMNI / "config/crab_hex/__init__.py"
RUN_PHASES = OMNI / "experiments/tools/run_phases.py"
OMNI_MATH_PY = OMNI / "mdp/crab_hex_omni_command_math.py"
#: The last commit before E1 (step 4, M1b): the harness / gait_eval identity baseline.
BASELINE = "dfae3c27"
DT = 0.02

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from gait_eval import metrics as M  # noqa: E402
from gait_eval import report as R  # noqa: E402
from gait_eval import schedule as S  # noqa: E402


def _load(path: Path, name: str):
    mod = sys.modules.get(name)
    if mod is None:
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
    return mod


def _git_show(rel: str) -> str:
    try:
        return subprocess.run(["git", "-C", str(REPO), "show", f"{BASELINE}:{rel}"], check=True,
                              capture_output=True, text=True).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        pytest.skip(f"baseline commit {BASELINE} unavailable: {exc}")


# ================================================================ the harness, lifted with ast
SRC = EVAL_SCRIPT.read_text()
TREE = ast.parse(SRC)
LIFT_FUNCS = ("_checkpoint_agent_yaml", "_load_agent_params", "_omni_student_route", "_student_runner_knob_check",
              "_payload_check_read", "_payload_run_meta_record")
LIFT_CONSTS = ("PAYLOAD_PER_DRAW_KEYS", "PAYLOAD_READ_AT")


def _lift():
    nodes = [n for n in TREE.body
             if (isinstance(n, ast.FunctionDef) and n.name in LIFT_FUNCS)
             or (isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id in LIFT_CONSTS for t in n.targets))]
    assert {n.name for n in nodes if isinstance(n, ast.FunctionDef)} == set(LIFT_FUNCS)
    ns: dict = {"Path": Path}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(EVAL_SCRIPT), "exec"), ns)  # noqa: S102
    return types.SimpleNamespace(**{k: v for k, v in ns.items() if not k.startswith("__")})


HG = _lift()


def _main(tree) -> ast.FunctionDef:
    return [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main"][0]


def _calls(tree, attr: str) -> list[ast.Call]:
    return [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
            and n.func.attr == attr]


# ------------------------------------------------------------------ 1. forward / legacy identity of main()
#: Every name E1 introduces into main(); a statement that mentions one is an E1 statement.
E1_NAMES = {"omni_step", "omni_student_settings", "_role_student", "_ckpt_yaml", "payload_reads", "payload_armed",
            "_student_runner_knob_check", "_step_mod", "_payload_check_read", "_payload_run_meta_record"}


def _mentions_e1(node) -> bool:
    return any(isinstance(n, ast.Name) and n.id in E1_NAMES for n in ast.walk(node))


class _StripE1(ast.NodeTransformer):
    """main() with every E1 statement removed: ``if omni_step is not None: <omni> else: <pre-E1>`` -> <pre-E1>;
    an ``if`` whose TEST is E1-only, and every simple statement mentioning an E1 name, dropped; the run_meta
    dict's ``"payload"`` entry dropped. Compound pre-E1 statements are recursed into, never dropped."""

    def generic_visit(self, node):
        for field_name in ("body", "orelse", "finalbody"):
            stmts = getattr(node, field_name, None)
            if isinstance(stmts, list) and stmts and isinstance(stmts[0], ast.stmt):
                setattr(node, field_name, self._stmts(stmts))
        for child in ast.iter_child_nodes(node):
            if not isinstance(child, ast.stmt):
                super().visit(child)
        return node

    def visit_Dict(self, node):
        keep = [(k, v) for k, v in zip(node.keys, node.values)
                if not (isinstance(k, ast.Constant) and k.value == "payload")]
        node.keys, node.values = [k for k, _ in keep], [v for _, v in keep]
        return node

    def _stmts(self, stmts):
        out = []
        for s in stmts:
            if isinstance(s, ast.If) and ast.unparse(s.test) == "omni_step is not None" and s.orelse:
                out.extend(self._stmts(s.orelse))
                continue
            if isinstance(s, ast.If) and _mentions_e1(s.test):
                continue
            compound = isinstance(s, (ast.If, ast.For, ast.While, ast.With, ast.Try, ast.FunctionDef))
            if not compound:
                self.visit(s)  # the run_meta dict first
                if _mentions_e1(s):
                    continue
                out.append(s)
                continue
            out.append(self.visit(s))
        return out


def test_main_is_the_baseline_main_plus_the_e1_statements_only():
    """Forward student, teacher, canon, adapters, recorders, scoring: AST-identical to the baseline commit once
    the E1 statements are removed -- the existing rows run exactly the code they ran."""
    rel = str(EVAL_SCRIPT.relative_to(REPO))
    base = _main(ast.parse(_git_show(rel)))
    cur = _StripE1().visit(copy.deepcopy(_main(TREE)))
    assert ast.dump(cur) == ast.dump(base)
    # and the stripping removed something real (the transformer is not vacuous)
    assert ast.dump(_main(TREE)) != ast.dump(base)


def test_module_level_is_the_baseline_plus_e1_imports_and_helpers():
    base = ast.parse(_git_show(str(EVAL_SCRIPT.relative_to(REPO))))
    new_funcs = set(LIFT_FUNCS)
    new_consts = set(LIFT_CONSTS)

    def _keep(n):
        if isinstance(n, ast.FunctionDef) and n.name in new_funcs | {"main"}:
            return False
        if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id in new_consts for t in n.targets):
            return False
        if isinstance(n, ast.ImportFrom) and n.module in (
                "scripts.rsl_rl.modules.omni_student_step", "parkour_tasks.crab_hex_omni_task.config.crab_hex",
                "parkour_tasks.crab_hex_omni_task.config.crab_hex.agents.rsl_rl_ppo_cfg",
                "parkour_tasks.crab_hex_omni_task.mdp.omni_events"):
            return False
        return not (isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant) and isinstance(n.value.value, str))

    cur_rest = [ast.dump(n) for n in TREE.body if _keep(n)]
    base_rest = [ast.dump(n) for n in base.body if _keep(n)]
    assert cur_rest == base_rest
    imported = {(n.module, a.name) for n in TREE.body if isinstance(n, ast.ImportFrom) for a in n.names}
    assert {("scripts.rsl_rl.modules.omni_student_step", "OmniStudentStep"),
            ("scripts.rsl_rl.modules.omni_student_step", "is_omni_student"),
            ("parkour_tasks.crab_hex_omni_task.config.crab_hex", "omni_student_spec"),
            ("parkour_tasks.crab_hex_omni_task.config.crab_hex.agents.rsl_rl_ppo_cfg", "CrabHexOmniStudentPPORunnerCfg"),
            ("parkour_tasks.crab_hex_omni_task.mdp.omni_events", "payload_run_meta"),
            ("parkour_tasks.crab_hex_omni_task.mdp.omni_events", "PAYLOAD_EVENT_NAME")} <= imported


# ------------------------------------------------------------------ 2. the student branch calls the shared step
def _policy_if_chain() -> ast.If:
    for node in ast.walk(TREE):
        if isinstance(node, ast.If) and ast.unparse(node.test) == "args_cli.zero_actions":
            return node
    raise AssertionError("policy branch not found")


def test_the_student_branch_calls_the_shared_step_and_nothing_else():
    top = _policy_if_chain()
    student = top.orelse[0]
    assert ast.unparse(student.test) == "is_student"
    omni = student.body[0]
    assert isinstance(omni, ast.If) and ast.unparse(omni.test) == "omni_step is not None" and len(student.body) == 1
    assert [ast.unparse(s) for s in omni.body] == [
        "actions = omni_step.act(obs, extras['observations'], unwrapped.common_step_counter)"]
    # the arguments of play.py's call, in the harness's names
    (act,) = [c for c in _calls(TREE, "act") if ast.unparse(c.func.value) == "omni_step"]
    assert ast.unparse(act) == "omni_step.act(obs, extras['observations'], unwrapped.common_step_counter)"
    (fr,) = [c for c in _calls(TREE, "from_runner") if ast.unparse(c.func.value) == "OmniStudentStep"]
    assert ast.unparse(fr) == "OmniStudentStep.from_runner(runner, device=device)"
    resets = sorted(ast.unparse(c) for c in _calls(TREE, "reset") if ast.unparse(c.func.value) == "omni_step")
    assert resets == ["omni_step.reset()", "omni_step.reset(dones)"]
    # the omni branch writes nothing into the observation (the forward branch's 6:8 write stays its own)
    for n in ast.walk(ast.Module(body=omni.body, type_ignores=[])):
        assert not (isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Subscript))
    assert "obs[:, 6:8] = 1.5 * yaw_pred" in "\n".join(ast.unparse(s) for s in omni.orelse)   # the forward branch


def test_no_second_student_forward_pass_in_the_harness_or_gait_eval():
    """The depth networks are reached only through the shared step on the omni path: the harness's only direct
    depth-encoder / depth-actor calls are the forward branch's (one each), gait_eval touches none."""
    main = _main(TREE)
    direct = sorted(ast.unparse(c) for c in ast.walk(main) if isinstance(c, ast.Call) and isinstance(c.func, ast.Name)
                    and c.func.id in ("depth_encoder", "policy"))
    assert direct == ["depth_encoder(depth_camera, obs_student)", "policy(obs, hist_encoding=True)",
                      "policy(obs, hist_encoding=True, scandots_latent=depth_latent)"]
    for path in GE_DIR.glob("*.py"):
        text = path.read_text()
        for word in ("depth_encoder", "depth_actor", "scandots_latent", "OmniStudentStep", "omni_student_step"):
            assert word not in text, f"{path.name} mentions {word}"


def _wave_loop() -> ast.For:
    for node in ast.walk(_main(TREE)):
        if isinstance(node, ast.For) and ast.unparse(node.target) == "wave":
            return node
    raise AssertionError("wave loop not found")


def test_gru_reset_after_every_env_step_and_at_every_wave_start():
    wave = _wave_loop()
    # reset() at the top of every wave: right after the per-wave bookkeeping block, before the prologue
    kinds = [ast.unparse(s.test) if isinstance(s, ast.If) else type(s).__name__ for s in wave.body]
    i_suite, i_reset = kinds.index("suite_mode"), kinds.index("omni_step is not None")
    assert i_reset == i_suite + 1 and ast.unparse(wave.body[i_reset].body[0]) == "omni_step.reset()"
    # reset(dones) is the statement right after env.step in the step loop
    (step_loop,) = [n for n in ast.walk(wave) if isinstance(n, ast.For) and ast.unparse(n.target) == "t"]
    body = step_loop.body
    i_step = [i for i, s in enumerate(body) if isinstance(s, ast.Assign) and "env.step(actions)" in ast.unparse(s)][0]
    assert ast.unparse(body[i_step].targets[0]) == "(obs, _, dones, extras)"
    nxt = body[i_step + 1]
    assert isinstance(nxt, ast.If) and ast.unparse(nxt.test) == "omni_step is not None"
    assert [ast.unparse(s) for s in nxt.body] == ["omni_step.reset(dones)"]


def test_the_omni_student_route_is_decided_before_the_runner_and_the_delta_yaw_route_is_refused():
    main_src = ast.unparse(_main(TREE))
    i_route = main_src.index("omni_student_settings = _omni_student_route(")
    i_zero = main_src.index("agent_cfg.depth_encoder.zero_depth = True")
    i_runner = main_src.index("runner = make_on_policy_runner(")
    assert i_route < i_zero < i_runner
    # the route's role rule is is_student's rule
    assert ("_role_student = args_cli.policy_role == 'student' if args_cli.policy_role != 'auto' else "
            "agent_cfg.algorithm.class_name == 'DistillationWithExtractor'") in main_src
    assert "is_student = agent_cfg.algorithm.class_name == 'DistillationWithExtractor'" in main_src
    assert "is_student = args_cli.policy_role == 'student'" in main_src
    assert "if omni_step is not None and delta_yaw_tensor is not None:\n        raise SystemExit(" in main_src
    # the estimator is put in eval mode for the step's write, exactly where the step is built
    assert ("if omni_student_settings is not None:\n                runner.get_estimator_inference_policy(device=device)"
            "\n                omni_step = OmniStudentStep.from_runner(runner, device=device)") in main_src


# ------------------------------------------------------------------ 3. the route helper (runtime)
OMNI_YAML = {"depth_encoder": {"omni_student": True, "zero_depth": False, "num_prop": 89}}
FWD_YAML = {"depth_encoder": {"num_prop": 75}}
TEACHER_YAML = {"algorithm": {"class_name": "PPOWithExtractor"}}


def test_route_teacher_role_never_reaches_the_student_path():
    for task_flag in (True, False):
        for params in (None, OMNI_YAML, TEACHER_YAML):
            assert HG._omni_student_route(role_student=False, task_flag=task_flag, ckpt_params=params) is None


def test_route_omni_student_and_its_zero_depth():
    got = HG._omni_student_route(role_student=True, task_flag=True, ckpt_params=OMNI_YAML, ckpt_yaml=Path("/r/a.yaml"))
    assert got == {"zero_depth": False, "zero_depth_source": "checkpoint params/agent.yaml depth_encoder.zero_depth "
                   "(/r/a.yaml)", "checkpoint_agent_yaml": "/r/a.yaml"}
    zero = {"depth_encoder": {**OMNI_YAML["depth_encoder"], "zero_depth": True}}
    assert HG._omni_student_route(role_student=True, task_flag=True, ckpt_params=zero)["zero_depth"] is True
    none = HG._omni_student_route(role_student=True, task_flag=True, ckpt_params=None)
    assert none["zero_depth"] is False and none["checkpoint_agent_yaml"] is None and "default False" in none["zero_depth_source"]


def test_route_forward_student_is_unchanged_and_mismatches_are_refused():
    assert HG._omni_student_route(role_student=True, task_flag=False, ckpt_params=FWD_YAML) is None
    assert HG._omni_student_route(role_student=True, task_flag=False, ckpt_params=None) is None
    with pytest.raises(SystemExit, match="INVALID EVAL"):        # omni student checkpoint on the forward task
        HG._omni_student_route(role_student=True, task_flag=False, ckpt_params=OMNI_YAML)
    for params in (FWD_YAML, TEACHER_YAML, {}):                  # a non-omni checkpoint on the omni student task
        with pytest.raises(SystemExit, match="INVALID EVAL"):
            HG._omni_student_route(role_student=True, task_flag=True, ckpt_params=params)


def test_checkpoint_agent_yaml_lookup_and_the_tag_tolerant_loader(tmp_path):
    ck = tmp_path / "run" / "model_100.pt"
    ck.parent.mkdir()
    ck.write_bytes(b"x")
    assert HG._checkpoint_agent_yaml(ck) is None
    (ck.parent / "params").mkdir()
    y = ck.parent / "params" / "agent.yaml"
    y.write_text(textwrap.dedent("""\
        algorithm:
          class_name: DistillationWithExtractor
        depth_encoder:
          depth_shape: !!python/tuple
          - 58
          - 87
          omni_student: true
          zero_depth: true
          weird: !!python/object/apply:builtins.slice
          - 0
          - 3
          - null
        """))
    assert HG._checkpoint_agent_yaml(ck) == y.resolve()
    params = HG._load_agent_params(y)
    assert params["depth_encoder"]["depth_shape"] == [58, 87] and params["depth_encoder"]["weird"] is None
    got = HG._omni_student_route(role_student=True, task_flag=True, ckpt_params=params, ckpt_yaml=y)
    assert got["zero_depth"] is True and got["checkpoint_agent_yaml"] == str(y)


def test_real_trainer_dump_parses_when_present():
    dumps = sorted((REPO / "parkour/logs/rsl_rl/crab_hex_student").glob("*/params/agent.yaml"))
    if not dumps:
        pytest.skip("no forward student run dir in this checkout")
    params = HG._load_agent_params(dumps[-1])
    assert params["algorithm"]["class_name"] == "DistillationWithExtractor"
    # a forward student dump: no omni flag -> the forward branch, untouched
    assert HG._omni_student_route(role_student=True, task_flag=False, ckpt_params=params) is None


# ------------------------------------------------------------------ 4. the student runner's knob policy
spec = _load(SPEC_PY, "omni_student_spec__t4_eval_test")


class _StudentRunner:
    pass


class _StudentRunnerSub(_StudentRunner):
    pass


class _ForwardStudentRunner:
    pass


def test_knob_check_refuses_a_student_row_with_runner_knobs_and_names_the_remedy():
    env = {"KRABBY_STD_PARAM": "soft", "KRABBY_INIT_STD": "0.8", "KRABBY_SYM_LOSS_COEF": "0.5",
           "KRABBY_ACTION_STD_MAX": "1.0"}
    ref = spec.student_runner_knob_refusals(env)
    assert len(ref) == 2
    for cls in (_StudentRunner, _StudentRunnerSub):
        with pytest.raises(SystemExit) as exc:
            HG._student_runner_knob_check(cls, _StudentRunner, ref)
        msg = str(exc.value)
        assert "INVALID EVAL (gate 0)" in msg and "KRABBY_SYM_LOSS_COEF" in msg and "KRABBY_ACTION_STD_MAX" in msg
        assert "env -u" in msg and "STUDENT_RUNNER_KNOBS_REFUSED" in msg
    # every other runner (forward student, teacher), a string entry point, or a clean env: no refusal
    HG._student_runner_knob_check(_ForwardStudentRunner, _StudentRunner, ref)
    HG._student_runner_knob_check("pkg.mod:_StudentRunner", _StudentRunner, ref)
    HG._student_runner_knob_check(_StudentRunner, _StudentRunner, [])


#: S1 s18's recorded eval env (REFS_S1_s18 suite__omni es1 run_meta.env_vars): what a T* / student row carries.
REFS_S1_ENV = {
    "KRABBY_ACTION_LPF_HZ": "0", "KRABBY_ACTION_LPF_OBS": "0", "KRABBY_APEX_W": "1.0", "KRABBY_CLOCK_W": "1.0",
    "KRABBY_FLAT_TERRAIN_GEOM": "ring", "KRABBY_INIT_STD": "0.8", "KRABBY_LIN_VEL_X": "0.0:0.35",
    "KRABBY_OMNI_ANG_VEL_CLIP": "0.2", "KRABBY_OMNI_ANG_VEL_MAX": "0.3", "KRABBY_OMNI_CLOCK_HIDE": "1",
    "KRABBY_OMNI_CLOCK_TURN_R": "1.2", "KRABBY_OMNI_CMD_OBS": "0", "KRABBY_OMNI_FACING_OBS_CLIP": "0.6",
    "KRABBY_OMNI_FACING_OBS_ENCODING": "raw", "KRABBY_OMNI_RELAY_HYST": "0", "KRABBY_OMNI_SPEED_ELLIPSE": "0.35:0.35:0.25",
    "KRABBY_OMNI_STAND_HOLD": "0", "KRABBY_OMNI_STAND_OBS": "0", "KRABBY_OMNI_STAND_PARK": "0",
    "KRABBY_OMNI_YAW_RANGE": "-3.14159:3.14159", "KRABBY_PLANT": "A15+B", "KRABBY_PLAY_DEBUG_VIS": "0",
    "KRABBY_RSI_FRAC": "0", "KRABBY_STD_PARAM": "soft", "KRABBY_TRACK_L1_W": "-1.0", "KRABBY_TRACK_SIGMA2": "0.1",
}


def _eval_carry() -> tuple:
    """``run_phases.EVAL_CARRY`` read from the source (never executed)."""
    for node in ast.parse(RUN_PHASES.read_text()).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "EVAL_CARRY" for t in node.targets):
            return tuple(ast.literal_eval(node.value))
    raise AssertionError("EVAL_CARRY not found")


def test_refused_runner_knobs_are_never_eval_carry_keys_and_the_s1_eval_env_sets_none():
    carry = set(_eval_carry())
    refused = set(spec.STUDENT_RUNNER_KNOBS_REFUSED)
    assert not refused & carry, sorted(refused & carry)
    assert set(spec.STUDENT_RUNNER_KNOBS_APPLIED) <= carry          # STD_PARAM / INIT_STD are carried (and applied)
    assert not [k for k in carry if k.startswith("KRABBY_PAYLOAD_")]  # levels come only from the row's env file
    assert spec.student_runner_knob_refusals(REFS_S1_ENV) == []
    stamp = REPO / ("parkour/outputs/omni_plan/phase2/lr/REFS_S1_s18/suite__omni/seed001/2026-10-09_03-20-24/"
                    "run_meta.json")
    if stamp.is_file():
        assert json.loads(stamp.read_text())["env_vars"] == REFS_S1_ENV


def test_the_check_runs_before_the_agent_cfg_and_targets_the_student_gym_ids():
    main_src = ast.unparse(_main(TREE))
    i_chk = main_src.index("_student_runner_knob_check(gym.spec(scenario.task).kwargs.get('rsl_rl_cfg_entry_point'), "
                           "CrabHexOmniStudentPPORunnerCfg, _OMNI_STUDENT_SPEC.student_runner_knob_refusals(_ENVIRON))")
    i_cfg = main_src.index("agent_cfg = cli_args.parse_rsl_rl_cfg(scenario.task, args_cli)")
    i_envblock = main_src.index("os.environ[key] = value")
    assert i_envblock < i_chk < i_cfg      # after the manifest env block merge, before the cfg's __post_init__
    regs = {}
    for c in _calls(ast.parse(INIT_PY.read_text()), "register"):
        kw = {k.arg: k.value for k in c.keywords}
        regs[ast.literal_eval(kw["id"])] = {ast.literal_eval(k): ast.unparse(v)
                                            for k, v in zip(kw["kwargs"].keys, kw["kwargs"].values)}
    for gid in ("Isaac-Crab-Hex-Omni-Student-v0", "Isaac-Crab-Hex-Omni-Student-Play-v0"):
        assert regs[gid]["rsl_rl_cfg_entry_point"] == "agents.rsl_rl_ppo_cfg.CrabHexOmniStudentPPORunnerCfg"
    assert all(v["rsl_rl_cfg_entry_point"] != "agents.rsl_rl_ppo_cfg.CrabHexOmniStudentPPORunnerCfg"
               for k, v in regs.items() if "Student" not in k)
    # the harness never writes the environment for it (the process-state-sensitivity rule)
    assert "os.environ.pop" not in SRC and "del os.environ" not in SRC and "_ENVIRON.pop" not in SRC


# ------------------------------------------------------------------ 5. run_meta["payload"]
PH = _load(UNIT / "test_task4_payload.py", "task4_payload_helpers__t4_eval")
oe = PH.oe


def _wrapped(term):
    return types.SimpleNamespace(unwrapped=types.SimpleNamespace(event_manager=PH._EventManager({"payload_reset": term})))


def test_payload_record_is_null_when_off():
    assert HG._payload_run_meta_record([], False) is None
    assert HG._payload_run_meta_record([None], False) is None
    assert HG._payload_run_meta_record([None] * 10, True) is None
    HG._payload_check_read(None, 0, False)                         # off and not armed: fine
    off_env = types.SimpleNamespace(event_manager=PH._EventManager({"reset_root_state": object()}))
    assert oe.payload_run_meta(off_env) is None


def test_payload_record_read_after_the_reset_that_applied_it():
    term, env, view, params = PH._make(n=3, spec=PH._level(90.0), seed=5)
    early = oe.payload_run_meta(_wrapped(term))
    with pytest.raises(SystemExit, match="captured=False"):        # read before the first reset: refused
        HG._payload_check_read(early, 0, True)
    PH._call(term, env, params, torch.arange(3))                   # the wrapper's initial reset
    meta = oe.payload_run_meta(_wrapped(term))
    HG._payload_check_read(meta, 0, True)
    rec = HG._payload_run_meta_record([meta], False)
    assert rec == {**meta, "read_at": HG.PAYLOAD_READ_AT} and "by_wave" not in rec
    assert rec["captured"] is True and rec["payload_kg"] == [90.0] * 3
    assert rec["payload_com_m"] == [[0.0, 0.0, 0.309]] * 3
    assert rec["kg_range"] == [90.0, 90.0] and rec["zero_frac"] == 0.0 and rec["com_xy"] == 0.0 and rec["com_z"] == 0.309
    assert rec["total_mass_kg"] == [float(view.m[k, 0]) for k in range(3)]
    json.dumps(rec)                                                # JSON-ready as written into run_meta.json
    with pytest.raises(SystemExit, match="no instantiated term"):  # armed in the cfg, no live term
        HG._payload_check_read(None, 0, True)
    with pytest.raises(SystemExit, match="does not arm"):
        HG._payload_check_read(meta, 0, False)


def test_payload_record_by_wave_on_suite_rows():
    term, env, view, params = PH._make(n=4, seed=7)                # D5: U(0, 90) kg, 25 % zero, CoM shift
    reads = []
    for _wave in range(3):
        PH._call(term, env, params, torch.arange(4))               # each wave's env.reset() re-draws
        reads.append(oe.payload_run_meta(_wrapped(term)))
    rec = HG._payload_run_meta_record(reads, True)
    assert rec["payload_kg"] == reads[0]["payload_kg"] and rec["resets_per_env"] == [1] * 4
    assert [w["wave"] for w in rec["by_wave"]] == [0, 1, 2]
    for w, r in zip(rec["by_wave"], reads):
        assert set(w) == {"wave", *HG.PAYLOAD_PER_DRAW_KEYS} and all(w[k] == r[k] for k in HG.PAYLOAD_PER_DRAW_KEYS)
    assert rec["by_wave"][0]["payload_kg"] != rec["by_wave"][1]["payload_kg"]
    with pytest.raises(SystemExit, match="some waves"):
        HG._payload_run_meta_record([reads[0], None], True)


def test_payload_key_is_written_on_every_row_and_read_inside_the_wave_loop_after_the_reset():
    main = _main(TREE)
    (rm,) = [n for n in ast.walk(main) if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == "run_meta"
             and isinstance(n.value, ast.Dict)]
    entries = {k.value: ast.unparse(v) for k, v in zip(rm.value.keys, rm.value.values) if isinstance(k, ast.Constant)}
    assert entries["payload"] == "_payload_run_meta_record(payload_reads, suite_mode)"
    # the read: one per wave, inside the prologue block -- after the wave's env.reset() and both get_observations,
    # before the canon setup and the step loop
    wave = _wave_loop()
    (prologue,) = [s for s in wave.body if isinstance(s, ast.With) and "get_observations" in ast.unparse(s)]
    src = [ast.unparse(s) for s in prologue.body]
    i_read = src.index("payload_reads.append(payload_run_meta(unwrapped))")
    assert src[i_read + 1] == "_payload_check_read(payload_reads[-1], wave, payload_armed)"
    assert "env.reset()" in src[0] and [i for i, t in enumerate(src) if "env.get_observations()" in t][-1] < i_read
    i_pro = wave.body.index(prologue)
    assert any(isinstance(s, ast.With) and "for t in range(T)" in ast.unparse(s) for s in wave.body[i_pro + 1:])
    main_src = ast.unparse(main)
    assert ("payload_armed = getattr(getattr(env_cfg, 'events', None), PAYLOAD_EVENT_NAME, None) is not None"
            in main_src)
    assert oe.PAYLOAD_EVENT_NAME == "payload_reset"


def test_student_run_meta_block_only_on_omni_student_runs():
    main_src = ast.unparse(_main(TREE))
    assert main_src.count("run_meta['student'] =") == 1
    i = main_src.index("if omni_step is not None:\n        _step_mod")
    assert main_src.index("run_meta['student'] =") > i


# ================================================================ 6. opt-in metric_class
def test_applicability_category_and_the_representative_rule():
    assert M.applicability_category("d022") == "d022"
    assert M.applicability_category("d022", "translate") == "fwd"
    assert M.applicability_category("x", "turn") == "turn" and M.applicability_category("x", "stand") == "stand"
    assert M.applicability_category("x", "move_turn") == "move_turn"
    with pytest.raises(ValueError, match="metric_class"):
        M.applicability_category("x", "diag")
    assert set(M.METRIC_CLASS_CATEGORY) == set(S.METRIC_CLASSES) == set(S.CATEGORY_KINDS)
    for key, cats in {**M.MOTION_QUALITY_APPLIES, **M.RIGIDITY_APPLIES}.items():
        assert len({c in cats for c in ("fwd", "side", "back", "diag")}) == 1, key


def _hold_arrays(T=600, seed=0, speed=0.22, yaw_rate=0.0):
    rng = np.random.default_rng(seed)
    t = np.arange(T) * DT
    phase = 2 * np.pi * 1.2 * t
    a = np.sin(phase) > 0
    loaded = np.stack([a, ~a, a, ~a, a, ~a], axis=-1)
    force = np.where(loaded, 400.0, 5.0)
    pos = np.zeros((T, 3))
    pos[:, 0] = speed * t
    yaw = yaw_rate * t
    quat = np.stack([np.cos(yaw / 2), np.zeros(T), np.zeros(T), np.sin(yaw / 2)], axis=-1)
    foot = rng.normal(0, 0.01, (T, 6, 3))
    foot[..., 2] = np.where(loaded, 0.03, 0.08)
    return dict(dt=DT, foot_force_norm=force, root_pos_w=pos, root_quat_w=quat, foot_pos_w=foot,
                terrain_z=np.zeros((T, 6)), clock_phase=np.remainder(phase, 2 * np.pi))


def _strip_names(block):
    return {k: v for k, v in block.items() if k not in ("category", "metric_class")}


@pytest.mark.parametrize("cls,rep", [("translate", "fwd"), ("turn", "turn"), ("stand", "stand"), ("move_turn", "move_turn")])
def test_metric_class_scores_exactly_as_its_representative(cls, rep):
    arr = _hold_arrays(yaw_rate=(0.15 if cls in ("turn", "move_turn") else 0.0),
                       speed=(0.0 if cls in ("turn", "stand") else 0.22))
    by_name = M.rigidity_metrics(rep, **arr)
    by_class = M.rigidity_metrics("d022", metric_class=cls, **arr)
    unscored = M.rigidity_metrics("d022", **arr)
    assert _strip_names(by_class) == _strip_names(by_name)
    assert by_class["category"] == "d022" and by_class["metric_class"] == cls and "metric_class" not in by_name
    assert all(unscored[k] is None for k in M.RIGIDITY_EPISODE_KEYS)     # the by-name rule on an unknown name
    q = dict(dt=DT, actions=np.zeros((600, 18)), foot_force_norm=arr["foot_force_norm"], foot_pos_w=arr["foot_pos_w"],
             foot_lin_vel_w=np.zeros((600, 6, 3)), root_pos_w=arr["root_pos_w"], root_quat_w=arr["root_quat_w"],
             cmd_wz=np.zeros(600), hold_s_nominal=12.0)
    qa, qb = M.motion_quality_metrics(rep, **q), M.motion_quality_metrics("d022", metric_class=cls, **q)
    assert _strip_names(qb) == _strip_names(qa) and qb["metric_class"] == cls and "metric_class" not in qa
    blocks = [M.rigidity_metrics(rep, **_hold_arrays(seed=s, speed=0.2 + 0.01 * s)) for s in range(4)]
    assert M.rigidity_pooled_metrics("d022", blocks, cls) == M.rigidity_pooled_metrics(rep, blocks)


def test_the_translate_class_gives_the_50n_gait_set():
    b = M.rigidity_metrics("d022", metric_class="translate", **_hold_arrays())
    for key in ("steps_per_m", "swing_apex", "support_feet", "taps_per_m", "step_rate_hz"):
        assert b[key] is not None, key
    assert M.rigidity_metrics("d022", **_hold_arrays())["steps_per_m"] is None


V3_HEAD = {
    "manifest_version": 3, "command_route": "term",
    "envelope": {"w_min": 0.05, "ceiling": 0.168, "w_max": 0.21, "cap": [[0.0, 0.21], [0.22, 0.105], [0.30, 0.065]]},
    "speed_ellipse": {"fwd": 0.35, "lat": 0.35, "back": 0.25},
    "defaults": {"prologue_s": 1.5, "settle_s": 1.0, "lin_vel_clip": 0.2},
}


def _v3(tmp_path, categories, **row):
    n = sum(len(c["holds"]) for c in categories)
    doc = {**json.loads(json.dumps(V3_HEAD)), "scenarios": [{
        "id": "su", "kind": "category_suite", "task": "Isaac-Crab-Hex-Omni-Walk-Play-v0", "checkpoint": "x",
        "episodes": 2, "waves": n, "episode_length_s": 18.0, "settle": {"hold_s": 2.0, "label": "settle"},
        "hold_s": 12.0, "categories": categories, **row}]}
    path = tmp_path / "m.json"
    path.write_text(json.dumps(doc))
    return path


def _cat(name, vx, vy, wz=0.0, kind="translate", **extra):
    return {"name": name, "kind": kind, "holds": [{"label": name, "vx": vx, "vy": vy, "wz": wz}], **extra}


def test_metric_class_parse_rules(tmp_path):
    sc = S.load_manifest(_v3(tmp_path, [_cat("d022", 0.2032535, 0.0841904, metric_class="translate"),
                                        _cat("d045", 0.1555635, 0.1555635)]))[0][0]
    assert [c.metric_class for c in sc.suite.categories] == ["translate", None]
    meta = sc.suite.categories_meta()
    assert meta[0]["metric_class"] == "translate" and "metric_class" not in meta[1]
    assert sc.suite.metric_class_of_hold() == {"d022": "translate", "d045": None}
    with pytest.raises(S.ManifestError, match="must be one of"):
        S.load_manifest(_v3(tmp_path, [_cat("d022", 0.2032535, 0.0841904, metric_class="turn")]))   # != kind
    with pytest.raises(S.ManifestError, match="must be one of"):
        S.load_manifest(_v3(tmp_path, [_cat("d022", 0.2032535, 0.0841904, metric_class="walk")]))
    with pytest.raises(S.ManifestError, match="remove"):
        S.load_manifest(_v3(tmp_path, [_cat("d022", 0.2032535, 0.0841904, watch_only=True)]))      # still strict


def test_metric_class_is_refused_on_v1_v2_suites(tmp_path):
    doc = {"manifest_version": 2, "speed_ellipse": {"fwd": 0.35, "lat": 0.25, "back": 0.25}, "ang_vel_max": 0.3,
           "scenarios": [{"id": "s", "kind": "category_suite", "task": "Isaac-Crab-Hex-Omni-Walk-Play-v0",
                          "checkpoint": "x", "episodes": 2, "waves": 1, "episode_length_s": 18.0,
                          "spawn_yaw": "first_face", "settle": {"hold_s": 2.0, "label": "settle"}, "hold_s": 12.0,
                          "categories": [{"name": "d000", "kind": "translate", "metric_class": "translate",
                                          "holds": [{"label": "d000", "speed": 0.22, "dir": 0.0, "face": 0.0}]}]}]}
    path = tmp_path / "v2.json"
    path.write_text(json.dumps(doc))
    with pytest.raises(S.ManifestError, match="remove"):
        S.load_manifest(path)
    del doc["scenarios"][0]["categories"][0]["metric_class"]
    path.write_text(json.dumps(doc))
    sc = S.load_manifest(path)[0][0]
    sc.suite.categories[0].metric_class = "translate"             # a code-built v2 suite carrying it
    with pytest.raises(S.ManifestError, match="manifest-v3 category key"):
        S.validate_scenario(sc)


def _synthetic_suite_raw(compiled, seed=3):
    """Walking contact patterns, root travel along the body command, per env (time first)."""
    rng = np.random.default_rng(seed)
    T, N = compiled.n_steps, compiled.segment_id.shape[1]
    cmd = np.stack([compiled.body_vx, compiled.body_vy, compiled.body_wz], axis=-1)
    t = np.arange(T) * DT
    phase = 2 * np.pi * 1.1 * t[:, None] + rng.uniform(0, 2 * np.pi, N)[None, :]
    a = np.sin(phase) > 0
    loaded = np.stack([a, ~a, a, ~a, a, ~a], axis=-1)
    lin = np.concatenate([cmd[..., :2] * 0.8, np.zeros((T, N, 1))], axis=-1)
    pos = np.zeros((T, N, 3))
    pos[..., :2] = np.cumsum(lin[..., :2], axis=0) * DT
    quat = np.zeros((T, N, 4))
    quat[..., 0] = 1.0
    foot = rng.normal(0, 0.01, (T, N, 6, 3))
    foot[..., 2] = np.where(loaded, 0.03, 0.08)
    return {
        "foot_force_norm": np.where(loaded, 400.0, 5.0), "cmd_applied": cmd, "root_lin_vel_b": lin,
        "root_ang_vel_b": np.zeros((T, N, 3)), "root_quat_w": quat, "root_pos_w": pos, "foot_pos_w": foot,
        "foot_lin_vel_w": np.zeros((T, N, 6, 3)), "foot_ang_vel_w": np.zeros((T, N, 6, 3)),
        "terrain_z": np.zeros((T, N, 6)), "actions": np.zeros((T, N, 18)), "joint_vel": np.zeros((T, N, 24)),
        "crab_failure": np.zeros((T, N), dtype=bool), "travel_yaw": np.arctan2(cmd[..., 1], cmd[..., 0]),
        "heading_err": np.zeros((T, N)), "clock_phase": np.remainder(phase, 2 * np.pi),
    }


def _score_suite(sc, n_envs=2):
    sc = dataclasses.replace(sc, episodes=n_envs)
    comp = S.stack_waves([S.compile_wave(sc, k, dt=DT, num_envs=n_envs) for k in range(sc.waves)])
    raw = _synthetic_suite_raw(comp)
    N = comp.segment_id.shape[1]
    eps = R.score_run(raw, compiled=comp, scenario=sc, dt=DT, term_reason=["schedule_complete"] * N,
                      n_steps_env=[comp.n_steps] * N, travel_yaw=raw["travel_yaw"], terrain_mesh=True,
                      body_cmd=np.stack([comp.body_vx, comp.body_vy, comp.body_wz], axis=-1))
    run_meta = {"scenario_id": sc.id, "task": sc.task, "command_override_max_deviation": 0.0,
                **S.suite_run_meta(sc, dt=DT, num_envs=n_envs, wave_seeds=[S.wave_seed(sc, k) for k in range(sc.waves)],
                                   trained_categories=None)}
    agg = R.aggregate(eps, suite=R.suite_aggregate_arg(run_meta))
    return eps, agg, R.summary_text(run_meta, eps)


def test_report_plumbs_metric_class_into_the_own_hold_and_the_aggregate(tmp_path):
    sc = S.load_manifest(_v3(tmp_path, [_cat("fwd", 0.22, 0.0), _cat("d022", 0.2032535, 0.0841904, metric_class="translate"),
                                        _cat("d045", 0.1555635, 0.1555635)]))[0][0]
    eps, agg, txt = _score_suite(sc)
    by = agg["suite"]["by_category"]
    for key in ("steps_per_m", "swing_apex", "support_feet", "taps_per_m", "step_rate_hz", "swings_per_m"):
        assert by["d022"][key]["median"] is not None and by["fwd"][key]["median"] is not None, key
    # without metric_class an unknown name reads nothing class-specific (the by-name rule, unchanged)
    assert by["d045"]["steps_per_m"]["n"] == 0 and by["d045"]["support_feet"]["n"] == 0
    assert by["d022"]["hz_speed_slope"]["n"] >= 0 and "hz_speed_slope" in by["d045"]
    assert agg["suite"]["metric_class_of_category"] == {"d022": "translate"}
    own = [e for e in eps if e["category"] == "d022"]
    assert own and all(e["holds"]["d022"]["rigidity"]["metric_class"] == "translate" for e in own)
    assert all("metric_class" not in e["holds"][e["hold"]]["rigidity"] for e in eps if e["category"] != "d022")
    rig_lines = txt.split("rigidity by category")[1].split("\n")
    assert any(line.strip().startswith("d022:") and "steps_per_m=" in line for line in rig_lines)
    assert any(line.strip().startswith("d045:") and "steps_per_m=" not in line for line in rig_lines)


COMMITTED = [FWD_MANIFEST, OMNI_MANIFEST, BODY_MANIFEST,
             TASKS / "crab_hex_forward_task/experiments/eval/scenarios_v2.yaml",
             TASKS / "crab_hex_sideways_task/experiments/eval/scenarios_side.yaml",
             OMNI / "experiments/eval/scenarios_gpu0_sidesweep.yaml",
             OMNI / "experiments/2026-10-07_omni_phase2/scenarios_p2.yaml",
             OMNI / "experiments/2026-10-13_stand_height/scenarios_sh.yaml"]


@pytest.mark.parametrize("path", COMMITTED, ids=lambda p: p.name)
def test_committed_manifests_carry_no_metric_class(path):
    for sc in S.load_manifest(path)[0]:
        if sc.suite is None:
            continue
        assert all(c.metric_class is None for c in sc.suite.categories)
        assert all(set(c) == {"name", "kind", "holds"} for c in sc.suite.categories_meta())
        assert all(v is None for v in sc.suite.metric_class_of_hold().values())


#: Run once against the baseline commit's gait_eval and once against this tree's, in separate processes (the
#: package imports itself as ``gait_eval``): every committed row's parse, compiled schedule, category maps,
#: suite run_meta and SYNTHETIC SCORING (episodes + aggregate + summary text) as one sha256 per row.
_FINGERPRINT = r'''
import dataclasses, hashlib, json, sys
from pathlib import Path
import numpy as np
root, out, omni_math, *manifests = sys.argv[1:]
sys.path.insert(0, root)
from gait_eval import metrics as M, report as R, schedule as S
S._OMNI_MATH_PY = Path(omni_math)
DT = 0.02
def h(x): return hashlib.sha256(json.dumps(M.json_safe(x), sort_keys=True, default=repr).encode()).hexdigest()
def arrs(c): return {f.name: (hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest() if isinstance(v, np.ndarray) else v)
                     for f in dataclasses.fields(c) for v in [getattr(c, f.name)]}
def raw_for(comp, rng):
    T, N = comp.n_steps, comp.segment_id.shape[1]
    if comp.body_vx is not None: cmd = np.stack([comp.body_vx, comp.body_vy, comp.body_wz], -1)
    elif comp.world_dir is not None:
        cmd = np.stack([comp.world_speed * np.cos(comp.world_dir - comp.world_face),
                        comp.world_speed * np.sin(comp.world_dir - comp.world_face), np.zeros((T, N))], -1)
    else: cmd = np.asarray(comp.cmd, np.float64)
    ph = 2 * np.pi * 1.1 * (np.arange(T) * DT)[:, None] + rng.uniform(0, 6.28, N)[None]
    a = np.sin(ph) > 0
    ld = np.stack([a, ~a, a, ~a, a, ~a], -1) | (rng.uniform(size=(T, N, 6)) < 0.05)
    lin = np.concatenate([cmd[..., :2] * 0.7, np.zeros((T, N, 1))], -1)
    pos = np.zeros((T, N, 3)); pos[..., :2] = np.cumsum(lin[..., :2], 0) * DT
    q = np.zeros((T, N, 4)); q[..., 0] = 1.0
    ang = np.zeros((T, N, 3)); ang[..., 2] = cmd[..., 2] * 0.9
    fp = rng.normal(0, 0.01, (T, N, 6, 3)); fp[..., 2] = np.where(ld, 0.03, 0.08)
    raw = {"foot_force_norm": np.where(ld, 400.0, 5.0), "cmd_applied": cmd, "root_lin_vel_b": lin,
           "root_ang_vel_b": ang, "root_quat_w": q, "root_pos_w": pos, "foot_pos_w": fp,
           "foot_lin_vel_w": rng.normal(0, 0.02, (T, N, 6, 3)), "foot_ang_vel_w": np.zeros((T, N, 6, 3)),
           "terrain_z": np.zeros((T, N, 6)), "actions": rng.normal(0, 0.5, (T, N, 18)),
           "joint_vel": rng.normal(0, 1, (T, N, 24)), "applied_torque": rng.normal(0, 5, (T, N, 24)),
           "crab_failure": np.zeros((T, N), bool), "clock_phase": np.remainder(ph, 6.283185307179586),
           "clock_running": np.ones((T, N), bool), "rod_speed": rng.normal(0, 0.01, (T, N, 12)),
           "travel_yaw": np.arctan2(cmd[..., 1], cmd[..., 0])}
    if comp.world_dir is not None or comp.body_vx is not None: raw["heading_err"] = rng.normal(0, 0.1, (T, N))
    return raw
JN = [l + j for j in ("_Body_CamShaft_RevoluteJoint", "_Body_Hip_RevoluteJoint", "_Hip_Femur_RevoluteJoint",
                      "_Femur_Tibia_RevoluteJoint") for l in ("FL", "FR", "ML", "MR", "RL", "RR")]
res = {}
for mp in manifests:
    for sc in S.load_manifest(mp)[0]:
        key = Path(mp).name + "::" + sc.id
        rec = {}
        if S.uses_waves(sc):
            comp = S.stack_waves([S.compile_wave(sc, k, dt=DT, num_envs=sc.episodes) for k in range(sc.waves)])
            rec["suite"] = h({"meta": sc.suite.categories_meta(), "coh": sc.suite.category_of_hold(),
                              "rm": S.suite_run_meta(sc, dt=DT, num_envs=sc.episodes,
                                                     wave_seeds=[S.wave_seed(sc, k) for k in range(sc.waves)],
                                                     trained_categories=None)})
            small = dataclasses.replace(sc, episodes=2)
            sc_comp = S.stack_waves([S.compile_wave(small, k, dt=DT, num_envs=2) for k in range(small.waves)])
        else:
            comp = S.compile_schedule(sc, dt=DT, num_envs=sc.episodes)
            small = dataclasses.replace(sc, episodes=2)
            sc_comp = S.compile_schedule(small, dt=DT, num_envs=2)
        rec["compiled"] = h(arrs(comp))
        raw = raw_for(sc_comp, np.random.default_rng(5))
        N = sc_comp.segment_id.shape[1]
        hook = sc.command_frame == "world" or getattr(sc, "command_route", None) == "term"
        kw = dict(compiled=sc_comp, scenario=small, dt=DT, term_reason=["schedule_complete"] * N,
                  n_steps_env=[sc_comp.n_steps] * N, travel_yaw=(raw["travel_yaw"] if hook else 0.0),
                  motion_groups=M.motion_joint_groups(JN), action_clip=1.0, terrain_mesh=True)
        if sc_comp.body_vx is not None:
            kw["body_cmd"] = np.stack([sc_comp.body_vx, sc_comp.body_vy, sc_comp.body_wz], -1)
        eps = R.score_run(raw, **kw)
        rm = {"scenario_id": sc.id, "task": sc.task, "command_override_max_deviation": 0.0}
        if S.uses_waves(small):
            rm.update(S.suite_run_meta(small, dt=DT, num_envs=2, wave_seeds=[S.wave_seed(small, k) for k in range(small.waves)],
                                       trained_categories=None))
        rec["scored"] = h({"eps": eps, "agg": R.aggregate(eps, suite=R.suite_aggregate_arg(rm))})
        rec["summary"] = hashlib.sha256(R.summary_text(rm, eps).encode()).hexdigest()
        res[key] = rec
Path(out).write_text(json.dumps(res, sort_keys=True))
'''


def test_committed_rows_parse_compile_and_score_bit_identically_to_the_baseline(tmp_path):
    base_root = tmp_path / "baseline"
    (base_root / "gait_eval").mkdir(parents=True)
    for name in ("__init__.py", "metrics.py", "report.py", "schedule.py"):
        (base_root / "gait_eval" / name).write_text(_git_show(str((GE_DIR / name).relative_to(REPO))))
    script = tmp_path / "fp.py"
    script.write_text(_FINGERPRINT)
    manifests = [str(FWD_MANIFEST), str(OMNI_MANIFEST), str(BODY_MANIFEST)]
    outs = {}
    for tag, root in (("base", base_root), ("cur", SCRIPTS_DIR)):
        out = tmp_path / f"{tag}.json"
        subprocess.run([sys.executable, "-I", str(script), str(root), str(out), str(OMNI_MATH_PY), *manifests],
                       check=True, cwd=str(tmp_path), timeout=600)
        outs[tag] = json.loads(out.read_text())
    assert outs["cur"] == outs["base"]
    assert len(outs["cur"]) >= 15 and sum("suite" in r for r in outs["cur"].values()) >= 4


# ================================================================ 7. the sweep row file
def test_the_sweep_manifest_parses_into_16_translate_bins():
    scs, defaults = S.load_manifest(SWEEP)
    (sc,) = scs
    assert sc.id == "dirsweep__omni_body" and sc.kind == S.SUITE_KIND and sc.command_route == S.COMMAND_ROUTE_TERM
    assert sc.task == "Isaac-Crab-Hex-Omni-Walk-Play-v0" and sc.episodes == 100 and sc.waves == 16
    assert sc.env_seed == 1 and sc.spawn_yaw == "first_face" and sc.spawn_yaw_rad() == 0.0
    assert sc.suite.hold_s == 12.0 and sc.suite.settle_hold_s == 2.0 and sc.env_vars == {}
    cats = sc.suite.categories
    want = ["d000", "d022", "d045", "d068", "d090", "d112", "d135", "d158",
            "d180", "d202", "d225", "d248", "d270", "d292", "d315", "d338"]
    assert [c.name for c in cats] == want == [f"d{round(22.5 * k):03d}" for k in range(16)]
    for k, c in enumerate(cats):
        (h,) = c.holds
        assert c.kind == "translate" and c.metric_class == "translate" and h.label == c.name and h.hand is None
        assert h.wz == 0.0 and h.wz_symbol is None
        assert math.hypot(h.vx, h.vy) == pytest.approx(0.22, abs=1e-6)
        heading = math.degrees(math.atan2(h.vy, h.vx)) % 360.0
        assert min(abs(heading - 22.5 * k), 360.0 - abs(heading - 22.5 * k)) < 1e-4, (c.name, heading)
    # the top block (envelope, ellipse, defaults) is scenarios_omni_body.yaml's, verbatim
    def _top(text):
        lines = text.split("\n")
        return "\n".join(lines[lines.index("manifest_version: 3"):lines.index("scenarios:")])
    assert _top(SWEEP.read_text()) == _top(BODY_MANIFEST.read_text())
    body_sc = S.load_manifest(BODY_MANIFEST)[0][0]
    assert sc.envelope_block == body_sc.envelope_block and sc.speed_ellipse == body_sc.speed_ellipse
    assert yaml.safe_load(SWEEP.read_text())["envelope"] == yaml.safe_load(BODY_MANIFEST.read_text())["envelope"]
    assert S.body_hold_commands(sc)[1:] == [{"label": c.name, "vx": c.holds[0].vx, "vy": c.holds[0].vy, "wz": 0.0,
                                             "wz_symbol": None, "hold_s": 12.0} for c in cats]


def test_the_sweep_schedule_builds_and_every_bin_gets_the_50n_gait_set():
    sc = S.load_manifest(SWEEP)[0][0]
    parts = [S.compile_wave(sc, k, dt=DT, num_envs=100) for k in range(sc.waves)]
    comp = S.stack_waves(parts)
    assert comp.n_steps == round((1.5 + 2.0 + 12.0) / DT) and comp.segment_id.shape == (comp.n_steps, 1600)
    assert comp.hold_labels == ["settle"] + [c.name for c in sc.suite.categories]
    assert sorted(set(comp.hold_of_env.tolist())) == list(range(1, 17))
    meta = S.suite_run_meta(sc, dt=DT, num_envs=100, wave_seeds=[S.wave_seed(sc, k) for k in range(16)],
                            trained_categories=None)
    assert all(c["metric_class"] == "translate" for c in meta["suite"]["categories"])
    with pytest.raises(S.ManifestError, match="not categories"):           # the gate rows' list never applies
        S.suite_cli_args(sc, trained_categories="fwd,side")
    eps, agg, _txt = _score_suite(sc)
    by = agg["suite"]["by_category"]
    assert set(by) == {c.name for c in sc.suite.categories}
    for name, cat in by.items():
        for key in ("steps_per_m", "swing_apex", "support_feet", "taps_per_m", "step_rate_hz"):
            assert cat[key]["median"] is not None, (name, key)
        assert cat["net_yaw_ratio"]["n"] >= 0                               # the v3 body block joins every bin
    assert agg["suite"]["metric_class_of_category"] == {c.name: "translate" for c in sc.suite.categories}
