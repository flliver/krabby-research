"""Stage p2r step 1 (P2R_PLAN.md step 1 / CHARTER 12): ``scripts/p2r_mode_eval_crab_hex_gait.py`` and the
``run_arm.sh`` hooks. CPU only: nothing here starts Isaac/Kit or a GPU.

* argv: the launcher strips its own flags (both spellings, env fallback) and leaves the harness's argv in order;
  every inconsistent mode / checkpoint / probe combination is refused before Kit would start;
* ``runpy.run_path`` returns a COPY of the globals: patching ``g`` is a silent no-op and patching
  ``g["main"].__globals__`` takes effect (proved on a tiny fake harness);
* the patched factory: S1 built on the real env exactly once, the extra runners on the stand-in env (no
  observation computed), the train cfg deep-copied (the runner pops from it), and the CPU / numpy / python RNG
  restored around the extras;
* the mode wrapper on a stub env + stub omni command term: the latched slot (a settled turn stays turn), the
  reset re-derivation (the suite's cross-wave latch carry-over), hold = zeros on stand rows only, st2 selection with
  the specialist's own estimator on a clone, the probe's anchor-path dmu formula on the TRUE priv columns, the
  reservoir bank;
* end to end through a fake harness that mirrors the real loop: the probe run reproduces the unpatched run
  bitwise (actions), hold zeroes exactly the stand-mode steps and matches before the first stand entry, the record
  block keys, the per-episode raw series, the bank file, the settle share per wave, ``simulation_app.close()``;
* ``run_arm.sh``: ``P2_EVAL`` replaces the eval script and ``P2_ENV_EXTRA`` reaches the child env in row_eval and
  lr_eval.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import random
import runpy
import subprocess
import sys
import textwrap
import types
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[2]
CAMP = REPO / "parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task/experiments/2026-10-07_omni_phase2"
SCRIPTS = CAMP / "scripts"
LAUNCHER = SCRIPTS / "p2r_mode_eval_crab_hex_gait.py"
RUN_ARM = SCRIPTS / "run_arm.sh"
REAL_HARNESS = REPO / "parkour/parkour_tasks/parkour_tasks/crab_hex_forward_task/scripts/eval_crab_hex_gait.py"

CONTRACT_KEYS = {"variant", "enter_rule", "launcher_sha256", "harness_sha256", "s1_checkpoint_sha256", "specialist",
                 "rng_restored", "stand_mode_share", "switches_per_episode_mean", "settle_mode_share_by_wave", "bank"}


@pytest.fixture(scope="module")
def L():
    spec = importlib.util.spec_from_file_location("_p2r_launcher", LAUNCHER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ===================================================================================== stubs (written to tmp)
STUBS = textwrap.dedent('''
    """Stubs for the p2r launcher tests: a tiny omni-like env, command term, runner and report writer."""
    import contextlib, hashlib, json, math, types
    from pathlib import Path
    import numpy as np
    import torch
    from torch import nn

    NUM_PROP, NUM_SCAN, NUM_PRIV = 6, 4, 3
    P0, P1 = NUM_PROP + NUM_SCAN, NUM_PROP + NUM_SCAN + NUM_PRIV
    OBS_W, N_ACT = 20, 4
    DEAD, WZ, TURN_GAIN = 0.1, 0.3, 0.5
    CATS = ("stand", "fwd", "side", "back", "diag", "turn", "move_turn")


    def cfg_dict():
        return {"policy": {"class_name": "StubActor"}, "algorithm": {"class_name": "PPOWithExtractor"},
                "estimator": {"class_name": "StubEst", "num_prop": NUM_PROP, "num_scan": NUM_SCAN,
                              "num_priv_explicit": NUM_PRIV},
                "empirical_normalization": False}


    class StubActor(nn.Module):
        def __init__(self):
            super().__init__()
            self.w_hist = nn.Parameter(torch.randn(OBS_W, N_ACT))   # init draws the global RNG, like the real nets
            self.w_priv = nn.Parameter(torch.randn(OBS_W, N_ACT))
            self.std_raw = nn.Parameter(torch.full((N_ACT,), 0.5))
            self.calls = []

        def act_inference(self, obs, hist_encoding=False, **kw):
            self.calls.append((bool(hist_encoding), obs.detach().clone()))
            return torch.tanh(obs @ (self.w_hist if hist_encoding else self.w_priv))

        def effective_std(self):
            return self.std_raw.abs() + 0.1


    class StubEst(nn.Module):
        def __init__(self):
            super().__init__()
            self.w = nn.Parameter(torch.randn(NUM_PROP, NUM_PRIV))

        def inference(self, x):
            with torch.no_grad():
                return x @ self.w


    class StubRunner:
        builds = []

        def __init__(self, env, cfg, log_dir=None, device="cpu"):
            cfg["policy"].pop("class_name")      # the real runner pops class_name: a reused dict would KeyError
            cfg["estimator"].pop("class_name")
            obs, extras = env.get_observations()
            assert obs.shape[1] == OBS_W and "critic" in extras["observations"]
            StubRunner.builds.append(type(env).__name__)
            self.env = env
            self.alg = types.SimpleNamespace(policy=StubActor(), estimator=StubEst())
            self.loaded = None

        def load(self, path, load_optimizer=True):
            sd = torch.load(path)
            self.alg.policy.load_state_dict(sd["policy"])
            self.alg.estimator.load_state_dict(sd["est"])
            self.loaded = (str(path), load_optimizer)

        def get_inference_policy(self, device=None):
            return self.alg.policy.act_inference

        def get_estimator_inference_policy(self, device=None):
            return self.alg.estimator


    def make_runner(env, cfg, log_dir=None, device="cpu"):
        return StubRunner(env, cfg, log_dir=log_dir, device=device)


    def save_ckpt(path, seed):
        g = torch.Generator().manual_seed(seed)
        pol = {"w_hist": torch.randn(OBS_W, N_ACT, generator=g), "w_priv": torch.randn(OBS_W, N_ACT, generator=g),
               "std_raw": torch.rand(N_ACT, generator=g) + 0.2}
        torch.save({"policy": pol, "est": {"w": torch.randn(NUM_PROP, NUM_PRIV, generator=g)}}, path)


    def wrap(x):
        return torch.atan2(torch.sin(x), torch.cos(x))


    class StubCmd:
        """The omni term's slot / latch semantics (``schedule_hook`` :889-950), on a toy relay."""
        TELEMETRY_CATEGORIES = CATS

        def __init__(self, env):
            n = env.num_envs
            self.env = env
            self.slot_id = torch.zeros(n, dtype=torch.long)
            self.speed_cmd = torch.zeros(n)
            self.travel_dir_w = torch.zeros(n)
            self.heading_target = torch.zeros(n)
            self.vel_command_b = torch.zeros(n, 3)
            self._slot_latch = False
            self.compute = lambda: None

        def stand_mask(self):
            return self.slot_id == 0

        def telemetry_category(self):
            s = self.slot_id
            return torch.where(s == 1, 1, torch.where(s == 2, 5, torch.where(s == 3, 6, 0)))

        def _write(self):
            err = wrap(self.heading_target - self.env.heading)
            wz = torch.where(err.abs() > DEAD, torch.sign(err) * WZ, torch.zeros_like(err))
            self.vel_command_b = torch.stack([self.speed_cmd.clone(), torch.zeros_like(err), wz], dim=1)

        def schedule_hook(self, row):
            fresh = self.heading_target != row["face"]
            self.travel_dir_w[:] = row["dir"]
            self.speed_cmd[:] = row["speed"]
            self.heading_target[:] = row["face"]
            self._write()
            moving = self.speed_cmd > 0.0
            turning = self.vel_command_b[:, 2] != 0.0
            if self._slot_latch:
                self._hook_turning = torch.where(fresh, turning, self._hook_turning)
                turning = self._hook_turning
            self.slot_id[:] = moving.long() + 2 * turning.long()

        def resample(self, ids):
            # a NON-turning draw: the facing target is the current heading (the carry-over's cause)
            self.heading_target[ids] = self.env.heading[ids]
            self.speed_cmd[ids] = 0.1


    class StubUnwrapped:
        def __init__(self, n, cmd):
            self.num_envs = n
            self.episode_length_buf = torch.zeros(n, dtype=torch.long)
            self.command_manager = types.SimpleNamespace(get_term=lambda name: cmd)


    class StubEnv:
        def __init__(self, n, falls=None):
            self.num_envs, self.num_actions = n, N_ACT
            self.heading = torch.zeros(n)
            self.cmd = StubCmd(self)
            self.unwrapped = StubUnwrapped(n, self.cmd)
            self.state = torch.linspace(-0.5, 0.5, n * OBS_W).reshape(n, OBS_W)
            self.falls = dict(falls or {})      # {global step: env index}
            self.t = 0
            self.get_calls = 0

        def _obs(self):
            o = self.state.clone()
            return o, {"observations": {"policy": o, "critic": o[:, :8].clone()}}

        def get_observations(self):
            self.get_calls += 1
            return self._obs()

        def step(self, actions):
            n = self.num_envs
            actions = actions.clamp(-1.0, 1.0)
            noise = torch.rand(n, OBS_W) * 0.01                      # the global RNG, as physics noise
            self.state = torch.tanh(0.9 * self.state + 0.1 * actions.sum(1, keepdim=True) + noise + 0.05)
            self.heading = self.heading + self.cmd.vel_command_b[:, 2] * TURN_GAIN
            self.unwrapped.episode_length_buf = self.unwrapped.episode_length_buf + 1
            self.t += 1
            dones = torch.zeros(n, dtype=torch.long)
            e = self.falls.get(self.t)
            if e is not None:
                dones[e] = 1
                self.heading = torch.where(torch.arange(n) == e, torch.zeros_like(self.heading), self.heading)
                self.unwrapped.episode_length_buf = torch.where(
                    torch.arange(n) == e, torch.zeros_like(self.unwrapped.episode_length_buf),
                    self.unwrapped.episode_length_buf)
                self.cmd.resample([e])
            self.cmd.compute()
            o, ex = self._obs()
            return o, torch.zeros(n), dones, ex

        def reset(self):
            self.heading = torch.zeros(self.num_envs)
            self.state = torch.linspace(-0.5, 0.5, self.num_envs * OBS_W).reshape(self.num_envs, OBS_W)
            self.unwrapped.episode_length_buf = torch.zeros(self.num_envs, dtype=torch.long)
            self.cmd.resample(list(range(self.num_envs)))
            return self._obs()

        def seed(self, s):
            return s


    def compile_holds(holds, n, prologue=1, wave=None, all_labels=None):
        """holds: [(label, speed, face, steps)] -> a CompiledSchedule-like namespace (prologue rows = hold 0's)."""
        labels = all_labels or [h[0] for h in holds]
        rows, seg = [], []
        for i, (lab, sp, fa, k) in enumerate(holds):
            if i == 0:
                rows += [{"speed": sp, "dir": 0.0, "face": fa}] * prologue
                seg += [-1] * prologue
            rows += [{"speed": sp, "dir": 0.0, "face": fa}] * k
            seg += [labels.index(lab)] * k
        segment = np.repeat(np.asarray(seg, dtype=np.int16)[:, None], n, axis=1)
        return types.SimpleNamespace(segment_id=segment, hold_labels=list(labels), n_steps=len(rows), rows=rows,
                                     wave_of_env=(None if wave is None else np.full(n, wave)))


    def stack(parts):
        return types.SimpleNamespace(segment_id=np.concatenate([p.segment_id for p in parts], axis=1),
                                     hold_labels=parts[0].hold_labels, n_steps=parts[0].n_steps, rows=None,
                                     wave_of_env=np.concatenate([p.wave_of_env for p in parts]))


    SINGLE = [("fwd", 0.2, 0.0, 4), ("stand", 0.0, 0.0, 4), ("turn_l", 0.0, 1.0, 12), ("fwd2", 0.2, 1.0, 4)]
    SUITE_LABELS = ["settle", "turn_l", "move_turn"]
    SUITE_WAVES = [[("settle", 0.0, 0.0, 3), ("turn_l", 0.0, 1.0, 12)],
                   [("settle", 0.0, 0.0, 3), ("move_turn", 0.2, 1.0, 12)]]


    def _write_run(run_dir, *, run_meta, episode_metrics, raw, scenario, compiled, history_root):
        run_dir = Path(run_dir)
        (run_dir / "raw").mkdir(parents=True, exist_ok=True)
        (run_dir / "run_meta.json").write_text(json.dumps(run_meta, indent=1, sort_keys=True))
        for e, n in enumerate(run_meta["n_steps_per_env"]):
            np.savez_compressed(run_dir / "raw" / f"episode_{e:02d}.npz",
                                **{k: np.asarray(v)[:n, e] for k, v in raw.items()})
        (run_dir / "complete.json").write_text("{}")


    REPORT = types.SimpleNamespace(write_run=_write_run, summary_text=lambda *a: "")
''')

FAKE_HARNESS = textwrap.dedent('''
    """A fake harness mirroring eval_crab_hex_gait.py's flow: argparse at import, module-level runner factory,
    _resolve_scenario / _install_schedule_override_world / R.write_run globals, main() without app close."""
    import argparse, contextlib, hashlib, sys
    from pathlib import Path
    import numpy as np
    import torch
    import p2r_stubs as SB
    from p2r_stubs import make_runner as make_on_policy_runner

    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--suite", action="store_true")
    ap.add_argument("--fall", type=str, default="")
    ap.add_argument("--save-raw", dest="save_raw", action="store_true", default=True)
    args_cli = ap.parse_args()
    N = 3


    class _App:
        def close(self):
            Path(args_cli.out).mkdir(parents=True, exist_ok=True)
            (Path(args_cli.out) / "app_closed").write_text("1")


    simulation_app = _App()
    R = SB.REPORT


    def _resolve_scenario():
        suite = type("Suite", (), {"settle_label": "settle"})() if args_cli.suite else None
        return type("Scen", (), {"suite": suite, "id": "fake"})()


    def _install_schedule_override(*a, **k):
        raise AssertionError("body rows are not used here")


    def _install_schedule_override_world(cmd_term, compiled, device, hold_labels):
        state = {"step": 0}

        def _row(step):
            return compiled.rows[min(max(int(step), 0), compiled.n_steps - 1)]

        cmd_term.compute = lambda: cmd_term.schedule_hook(_row(state["step"]))
        cmd_term.schedule_hook(_row(0))
        state["row"] = _row
        return state


    def main():
        scenario = _resolve_scenario()
        falls = {int(k): int(v) for k, v in (p.split(":") for p in args_cli.fall.split(",") if p)}
        env = SB.StubEnv(N, falls=falls)
        runner = make_on_policy_runner(env, SB.cfg_dict(), log_dir=None, device="cpu")
        runner.load(args_cli.checkpoint)
        policy = runner.get_inference_policy(device="cpu")
        estimator = runner.get_estimator_inference_policy(device="cpu")
        if args_cli.suite:
            waves = [SB.compile_holds(h, N, wave=w, all_labels=SB.SUITE_LABELS) for w, h in enumerate(SB.SUITE_WAVES)]
        else:
            waves = [SB.compile_holds(SB.SINGLE, N)]
        cmd_term = env.unwrapped.command_manager.get_term("base_velocity")
        state = _install_schedule_override_world(cmd_term, waves[0], "cpu", waves[0].hold_labels)
        parts, n_all = [], []
        for w, comp in enumerate(waves):
            with (torch.inference_mode() if w > 0 else contextlib.nullcontext()):
                if w > 0:
                    env.reset()
                    state = _install_schedule_override_world(cmd_term, comp, "cpu", comp.hold_labels)
                obs, extras = env.get_observations()
            T = comp.n_steps
            acts = torch.zeros(T, N, SB.N_ACT)
            alive = torch.ones(N, dtype=torch.bool)
            n = torch.zeros(N, dtype=torch.long)
            with torch.inference_mode():
                for t in range(T):
                    state["step"] = t + 1
                    obs[:, SB.P0:SB.P1] = estimator.inference(obs[:, :SB.NUM_PROP])
                    actions = policy(obs, hist_encoding=True)
                    acts[t] = actions
                    obs, _, dones, extras = env.step(actions)
                    n = torch.where(alive, n + 1, n)
                    alive = alive & ~dones.bool()
            parts.append(acts.numpy().copy())
            n_all += [int(v) for v in n]
        host = {"actions": np.concatenate(parts, axis=1)}
        compiled = SB.stack(waves) if args_cli.suite else waves[0]
        sha = hashlib.sha256(Path(args_cli.checkpoint).read_bytes()).hexdigest()
        run_meta = {"checkpoint_sha256": sha, "n_steps_per_env": n_all, "dt": 0.02}
        R.write_run(Path(args_cli.out) / "stamp", run_meta=run_meta, episode_metrics=[], raw=host,
                    scenario=scenario, compiled=compiled, history_root=None)
''')


@pytest.fixture()
def SB(tmp_path, monkeypatch):
    (tmp_path / "p2r_stubs.py").write_text(STUBS)
    (tmp_path / "fake_harness.py").write_text(FAKE_HARNESS)
    monkeypatch.syspath_prepend(str(tmp_path))
    sys.modules.pop("p2r_stubs", None)
    import p2r_stubs  # noqa: PLC0415

    p2r_stubs.StubRunner.builds.clear()
    yield p2r_stubs
    sys.modules.pop("p2r_stubs", None)


def _opts(L, argv, environ=None, check_files=False):
    o, rest = L.strip_p2r_args(argv, environ or {})
    return L.validate_opts(o, check_files=check_files), rest


# ===================================================================================== argv
def test_strip_keeps_harness_argv_in_order(L):
    argv = ["--headless", "--manifest", "m.yaml", "--p2r-mode", "probe", "--scenario", "suite__omni",
            "--p2r-probe", "ST2_m20500=/a.pt", "--env-seed", "1", "--p2r-probe=C0R_m24993=/b.pt",
            "--checkpoint", "/s1.pt", "--p2r-bank-max", "500", "--save-raw", "--no-plot"]
    o, rest = _opts(L, argv)
    assert rest == ["--headless", "--manifest", "m.yaml", "--scenario", "suite__omni", "--env-seed", "1",
                    "--checkpoint", "/s1.pt", "--save-raw", "--no-plot"]
    assert o.mode == "probe" and o.bank_max == 500
    assert list(o.probes.items()) == [("ST2_m20500", "/a.pt"), ("C0R_m24993", "/b.pt")]


def test_strip_without_p2r_args_leaves_argv_untouched_and_defaults(L):
    argv = ["--headless", "--scenario", "x", "--checkpoint", "/c.pt"]
    o, rest = L.strip_p2r_args(argv, {})
    assert rest == argv and o.mode is None and o.bank_max == L.DEFAULT_BANK_MAX


def test_env_fallback_and_cli_precedence(L):
    env = {"P2R_MODE": "st2", "P2R_STAND_CKPT": "/st2.pt", "P2R_BANK_MAX": "7"}
    o, rest = _opts(L, ["--scenario", "x"], env)
    assert (o.mode, o.stand_ckpt, rest) == ("st2", "/st2.pt", ["--scenario", "x"])
    o, _ = _opts(L, ["--p2r-mode", "hold"], {"P2R_MODE": "probe"})
    assert o.mode == "hold"
    o, _ = _opts(L, [], {"P2R_MODE": "probe", "P2R_PROBES": "A=/a.pt,B=/b.pt"})
    assert list(o.probes) == ["A", "B"]


@pytest.mark.parametrize("argv", [
    [],                                                            # no mode
    ["--p2r-mode", "st2"],                                         # st2 without its specialist
    ["--p2r-mode", "hold", "--p2r-stand-ckpt", "/x.pt"],           # a specialist on hold
    ["--p2r-mode", "probe"],                                       # probe without probes
    ["--p2r-mode", "hold", "--p2r-probe", "A=/a.pt"],              # probes off the probe variant
    ["--p2r-mode", "probe", "--p2r-probe", "A=/a.pt", "--p2r-probe", "A=/b.pt"],   # duplicate name
    ["--p2r-mode", "probe", "--p2r-probe", "bad-name=/a.pt"],      # not an npz-key name
    ["--p2r-mode", "probe", "--p2r-probe", "/a.pt"],               # no NAME=
    ["--p2r-mode", "probe", "--p2r-probe", "A=/a.pt", "--p2r-bank-max", "0"],
    ["--p2r-mode", "probe", "--p2r-probe", "A=/a.pt", "--p2r-bankmax", "3"],      # a typo'd p2r flag
    ["--p2r-mode", "statue"],
])
def test_inconsistent_args_refused(L, argv):
    with pytest.raises(SystemExit):
        _opts(L, argv)


def test_missing_checkpoint_refused(L, tmp_path):
    with pytest.raises(SystemExit, match="not found"):
        _opts(L, ["--p2r-mode", "st2", "--p2r-stand-ckpt", str(tmp_path / "nope.pt")], check_files=True)


def test_launcher_name_matches_run_arm_idle_check():
    """run_arm.sh's drivers_idle greps the process table for ...eval_crab_hex_gait.py."""
    assert LAUNCHER.name.endswith("eval_crab_hex_gait.py")
    line = f"/home/nickmagus/krabby/isaac_venv/bin/python {LAUNCHER} --headless"
    pat = r"^[^ ]*isaac_venv/bin/python [^ ]*(run_phases|run_candidates|eval_crab_hex_gait)\.py"
    assert subprocess.run(["grep", "-qE", pat], input=line, text=True).returncode == 0


def test_default_harness_is_the_locked_harness(L):
    assert L.HARNESS == REAL_HARNESS and REAL_HARNESS.is_file()


def test_real_harness_patch_points_present(L):
    """The globals the launcher patches exist in the locked harness, at module level, with the call shapes used."""
    src = REAL_HARNESS.read_text()
    for needle in ("from scripts.rsl_rl.runner_factory import agent_cfg_to_train_dict, make_on_policy_runner",
                   "runner = make_on_policy_runner(env, agent_cfg_to_train_dict(agent_cfg), log_dir=None, device=device)",
                   "policy = runner.get_inference_policy(device=device)",
                   "estimator = runner.get_estimator_inference_policy(device=device)",
                   "def _install_schedule_override_world(", "def _resolve_scenario()", "    R.write_run(",
                   "actions = policy(obs, hist_encoding=True)", "obs, _, dones, extras = env.step(actions)",
                   "simulation_app = app_launcher.app", "args_cli = parser.parse_args()"):
        assert needle in src, needle
    # main() itself never closes the app (only the __main__ block does)
    main_src = src.split("def main() -> None:")[1].split('if __name__ == "__main__":')[0]
    assert "simulation_app.close()" not in main_src


# ===================================================================================== runpy globals
def test_runpy_returns_a_copy_and_only_main_globals_patch_takes(tmp_path):
    fake = tmp_path / "tiny_harness.py"
    fake.write_text("def make_on_policy_runner(*a, **k):\n    return 'ORIGINAL'\n\n"
                    "def main():\n    return make_on_policy_runner(None, {})\n")
    g = runpy.run_path(str(fake), run_name="p2r_harness")
    assert g["main"].__globals__ is not g
    g["make_on_policy_runner"] = lambda *a, **k: "PATCHED"
    assert g["main"]() == "ORIGINAL"               # patching the returned copy: silent no-op
    g["main"].__globals__["make_on_policy_runner"] = lambda *a, **k: "PATCHED"
    assert g["main"]() == "PATCHED"


# ===================================================================================== factory / RNG / stand-in
def _ctx(L, mode, **kw):
    o = types.SimpleNamespace(mode=mode, stand_ckpt=kw.get("stand_ckpt"), probes=kw.get("probes", {}),
                              bank_max=kw.get("bank_max", 50))
    return L._Ctx(o, REAL_HARNESS)


def test_factory_builds_extras_on_stand_in_restores_rng_and_deepcopies(L, SB, tmp_path):
    for i, name in enumerate(("a", "b")):
        SB.save_ckpt(tmp_path / f"{name}.pt", 10 + i)
    # reference: S1 alone
    torch.manual_seed(0); np.random.seed(0); random.seed(0)
    env0 = SB.StubEnv(3)
    SB.make_runner(env0, SB.cfg_dict())
    ref = (torch.rand(4), np.random.rand(2), random.random())
    # patched factory with two probes
    torch.manual_seed(0); np.random.seed(0); random.seed(0)
    env = SB.StubEnv(3)
    ctx = _ctx(L, "probe", probes={"A": str(tmp_path / "a.pt"), "B": str(tmp_path / "b.pt")})
    ctx.scenario = types.SimpleNamespace(suite=None)
    SB.StubRunner.builds.clear()
    runner = L.make_factory(ctx, SB.make_runner)(env, SB.cfg_dict(), log_dir=None, device="cpu")
    got = (torch.rand(4), np.random.rand(2), random.random())
    assert torch.equal(ref[0], got[0]) and np.array_equal(ref[1], got[1]) and ref[2] == got[2]
    assert ctx.rng_restored is True
    # S1 on the real env (its one get_observations), the probes on the stand-in env (none computed)
    assert SB.StubRunner.builds == ["StubEnv", "StandInEnv", "StandInEnv"]
    assert env.get_calls == 1
    assert set(ctx.mode.probes) == {"A", "B"} and ctx.probe_meta["A"]["sha256"] == L.sha256_file(tmp_path / "a.pt")
    # the probes carry their checkpoints' weights
    sd = torch.load(tmp_path / "b.pt")
    assert torch.equal(ctx.mode.probes["B"].w_priv, sd["policy"]["w_priv"])
    # the latch armed at runtime, the env wrapped, get_inference_policy -> the mode wrapper
    ct = env.unwrapped.command_manager.get_term("base_velocity")
    assert ct._slot_latch is True and ct._hook_turning.dtype == torch.bool and not ct._hook_turning.any()
    assert "step" in vars(env) and "get_observations" in vars(env)
    s1 = tmp_path / "s1.pt"
    SB.save_ckpt(s1, 1)
    runner.load(str(s1))
    assert ctx.s1_sha == L.sha256_file(s1) and runner.loaded == (str(s1), True)
    assert runner.get_inference_policy(device="cpu") is ctx.mode
    assert torch.allclose(ctx.mode.s1_sigma, runner.alg.policy.effective_std())


def test_extra_loads_skip_the_optimizer(L, SB, tmp_path):
    SB.save_ckpt(tmp_path / "st.pt", 3)
    env = SB.StubEnv(2)
    seen = []
    orig = SB.StubRunner.load

    def spy(self, path, load_optimizer=True):
        seen.append(load_optimizer)
        return orig(self, path, load_optimizer=load_optimizer)

    SB.StubRunner.load = spy
    try:
        ctx = _ctx(L, "st2", stand_ckpt=str(tmp_path / "st.pt"))
        L.make_factory(ctx, SB.make_runner)(env, SB.cfg_dict(), log_dir=None, device="cpu")
    finally:
        SB.StubRunner.load = orig
    assert seen == [False]
    assert ctx.specialist == {"path": str(tmp_path / "st.pt"), "sha256": L.sha256_file(tmp_path / "st.pt")}


def test_stand_in_env_never_computes_or_steps(L, SB):
    env = SB.StubEnv(2)
    tmpl = {"obs": torch.zeros(2, SB.OBS_W), "groups": {"policy": torch.zeros(2, SB.OBS_W), "critic": torch.ones(2, 8)}}
    si = L.StandInEnv(env, tmpl)
    o, ex = si.get_observations()
    assert env.get_calls == 0 and not o.any() and not ex["observations"]["critic"].any()
    assert si.num_actions == SB.N_ACT and si.unwrapped is env.unwrapped
    with pytest.raises(RuntimeError):
        si.step(torch.zeros(2, SB.N_ACT))


# ===================================================================================== mode wrapper (unit)
def _mode(L, SB, variant, n=4, **kw):
    env = SB.StubEnv(n)
    ct = env.cmd
    L.arm_slot_latch(ct, n, "cpu")
    s1 = SB.StubActor()
    m = L.P2RMode(variant, ct, env.unwrapped, priv=(SB.P0, SB.P1), num_prop=SB.NUM_PROP, **kw)
    m.bind_s1(s1.act_inference, s1)
    return env, ct, s1, m


def _begin(m, SB, n, labels=("fwd", "stand", "turn_l"), seg=None, T=5):
    seg = np.zeros((T, n), np.int16) if seg is None else seg
    m.begin_pass(types.SimpleNamespace(segment_id=seg, hold_labels=list(labels), n_steps=T))


def test_latched_slot_settled_turn_stays_turn(L, SB):
    env, ct, s1, m = _mode(L, SB, "hold", n=2)
    env.unwrapped.episode_length_buf[:] = 5                   # mid-episode: no re-derivation
    ct.schedule_hook({"speed": 0.0, "dir": 0.0, "face": 1.0})  # a fresh turn target, error 1.0 > dead band
    assert (ct.slot_id == 2).all()
    env.heading[:] = 1.0                                       # the turn has settled: raw relay is 0
    ct.schedule_hook({"speed": 0.0, "dir": 0.0, "face": 1.0})
    assert (ct.vel_command_b[:, 2] == 0).all() and (ct.slot_id == 2).all()   # latched: still turn, not stand
    _begin(m, SB, 2)
    obs, _ = env.get_observations()
    m.observe(obs)
    with torch.inference_mode():
        a = m(obs, hist_encoding=True)
    assert torch.equal(a, s1.act_inference(obs, hist_encoding=True))        # not a stand row: S1's action
    env.heading[:] = 0.0
    ct.schedule_hook({"speed": 0.0, "dir": 0.0, "face": 0.0})               # a fresh stand target, no facing error
    assert ct.stand_mask().all()


def test_reset_rederives_the_carried_over_latch(L, SB):
    env, ct, s1, m = _mode(L, SB, "hold", n=3)
    ct._hook_turning = torch.tensor([True, True, False])       # the previous wave's turn latch
    ct.slot_id = torch.tensor([2, 2, 0])                       # a settle hold that inherited it
    env.unwrapped.episode_length_buf = torch.tensor([0, 7, 0])  # env 0 just reset; env 1 mid-episode
    _begin(m, SB, 3)
    obs, _ = env.get_observations()
    m.observe(obs)
    with torch.inference_mode():
        a = m(obs, hist_encoding=True)
    assert ct._hook_turning.tolist() == [False, True, False]   # re-derived from the raw relay (0 at face 0)
    assert ct.slot_id.tolist() == [0, 2, 0]
    ref = s1.act_inference(obs, hist_encoding=True)
    assert torch.equal(a[1], ref[1]) and not a[0].any() and not a[2].any()
    assert m.resets_rederived == 2


def test_reset_rederive_keeps_a_reopened_turn(L, SB):
    """An env that falls in a turn hold respawns with the hold's facing error re-opened: re-derivation says turn."""
    env, ct, s1, m = _mode(L, SB, "hold", n=2)
    ct.heading_target[:] = 1.0
    ct._write()                                                # heading 0, target 1.0 -> raw relay non-zero
    ct._hook_turning = torch.tensor([False, False])
    env.unwrapped.episode_length_buf = torch.tensor([0, 3])
    _begin(m, SB, 2)
    obs, _ = env.get_observations()
    m.observe(obs)
    with torch.inference_mode():
        m(obs, hist_encoding=True)
    assert ct.slot_id.tolist() == [2, 0]


def test_hold_zeroes_only_stand_rows(L, SB):
    env, ct, s1, m = _mode(L, SB, "hold", n=4)
    env.unwrapped.episode_length_buf[:] = 3
    ct.slot_id = torch.tensor([0, 1, 2, 3])
    _begin(m, SB, 4)
    obs, _ = env.get_observations()
    m.observe(obs)
    with torch.inference_mode():
        a = m(obs, hist_encoding=True)
    ref = s1.act_inference(obs, hist_encoding=True)
    assert torch.equal(a[0], torch.zeros(SB.N_ACT))
    assert torch.equal(a[1:], ref[1:])
    assert m._cur["stand"][0].tolist() == [True, False, False, False] and m._cur["t"] == 1


def test_st2_selects_specialist_on_stand_rows_with_its_own_estimate(L, SB):
    st_actor, st_est = SB.StubActor(), SB.StubEst()
    env, ct, s1, m = _mode(L, SB, "st2", n=3, stand=(st_actor.act_inference, st_est))
    env.unwrapped.episode_length_buf[:] = 3
    ct.slot_id = torch.tensor([0, 1, 0])
    _begin(m, SB, 3)
    obs, _ = env.get_observations()
    m.observe(obs)
    s1_est = SB.StubEst()
    obs[:, SB.P0:SB.P1] = s1_est.inference(obs[:, :SB.NUM_PROP])   # the harness's in-place S1 estimate
    before = obs.clone()
    with torch.inference_mode():
        a = m(obs, hist_encoding=True)
    assert torch.equal(obs, before)                                # the harness's obs is never written
    own = before.clone()
    own[:, SB.P0:SB.P1] = st_est.inference(before[:, :SB.NUM_PROP])
    a_st = torch.tanh(own @ st_actor.w_hist)
    a_s1 = torch.tanh(before @ s1.w_hist)
    assert torch.equal(a[0], a_st[0]) and torch.equal(a[2], a_st[2]) and torch.equal(a[1], a_s1[1])
    assert st_actor.calls[-1][0] is True                           # hist_encoding=True, like the harness


def test_probe_applies_s1_and_logs_anchor_dmu_on_true_priv(L, SB):
    p1, p2 = SB.StubActor(), SB.StubActor()
    env, ct, s1, m = _mode(L, SB, "probe", n=3, probes={"A": p1, "B": p2}, bank_max=100)
    env.unwrapped.episode_length_buf[:] = 3
    ct.slot_id = torch.tensor([0, 1, 1])
    _begin(m, SB, 3, labels=("stand", "fwd"), seg=np.array([[0, 1, 1]] * 5, np.int16))
    obs, _ = env.get_observations()
    m.observe(obs)
    true = obs.clone()
    obs[:, SB.P0:SB.P1] = SB.StubEst().inference(obs[:, :SB.NUM_PROP])   # overwritten in place, as the harness does
    with torch.inference_mode():
        a = m(obs, hist_encoding=True)
    assert torch.equal(a, torch.tanh(obs @ s1.w_hist))           # S1 acts, on the harness's obs
    sigma = s1.effective_std().detach()
    mu_s1 = torch.tanh(true @ s1.w_priv)                         # the anchor path: hist_encoding=False, TRUE priv
    for name, p in (("A", p1), ("B", p2)):
        want = 0.5 * (((torch.tanh(true @ p.w_priv) - mu_s1) / sigma) ** 2).sum(-1)
        assert torch.allclose(m._cur["dmu"][name][0], want.float(), atol=1e-6)
        assert p.calls[-1][0] is False and torch.equal(p.calls[-1][1], true)
    assert not torch.allclose(true, obs)                         # the snapshot really differs from the harness obs


def test_probe_refuses_an_obs_it_did_not_see(L, SB):
    env, ct, s1, m = _mode(L, SB, "probe", n=2, probes={"A": SB.StubActor()})
    _begin(m, SB, 2)
    obs, _ = env.get_observations()
    m.observe(obs)
    with pytest.raises(RuntimeError, match="snapshot"):
        m(obs.clone(), hist_encoding=True)


def test_dmu_formula(L):
    mu_p = torch.tensor([[1.0, 2.0], [0.0, 0.0]])
    mu_s = torch.tensor([[0.0, 0.0], [0.0, 1.0]])
    sig = torch.tensor([0.5, 2.0])
    assert torch.allclose(L.P2RMode.dmu(mu_p, mu_s, sig), torch.tensor([0.5 * (4.0 + 1.0), 0.5 * 0.25]))


def test_bank_reservoir_non_stand_alive_only(L, SB):
    n, T, K = 4, 30, 25
    env, ct, s1, m = _mode(L, SB, "probe", n=n, probes={"A": SB.StubActor()}, bank_max=K,
                           stand_labels=("stand", "settle"))
    seg = np.tile(np.array([[-1, 0, 1, 2]], np.int16), (T, 1))  # env0 prologue, env1 settle, env2 stand, env3 turn
    m.begin_pass(types.SimpleNamespace(segment_id=seg, hold_labels=["settle", "stand", "turn_l"], n_steps=T))
    env.unwrapped.episode_length_buf[:] = 3
    ct.slot_id = torch.tensor([0, 0, 0, 2])
    obs, _ = env.get_observations()
    for t in range(T):
        m.observe(obs)
        with torch.inference_mode():
            m(obs, hist_encoding=True)
        obs = obs + 1.0
    assert m.bank_seen == T                                       # only env 3 (turn_l) is non-stand
    b = m.bank_arrays()
    assert b["obs"].shape == (K, SB.OBS_W) and set(b["hold_label"]) == {"turn_l"} and set(b["category"]) == {"turn"}
    assert b["s1_sigma"].shape == (SB.N_ACT,)
    # every banked row is a real env-3 obs (true priv) from some step, no duplicates
    base = env.state[3]
    steps = sorted(int(round(float(r[0] - base[0]))) for r in b["obs"])
    assert len(set(steps)) == K and all(0 <= s < T for s in steps)
    # a dead env is never banked
    m.alive = torch.tensor([True, True, True, False])
    seen = m.bank_seen
    with torch.inference_mode():
        m._bank_add(0, obs)
    assert m.bank_seen == seen


def test_bank_reservoir_is_deterministic_and_does_not_touch_global_rng(L, SB):
    outs = []
    for _ in range(2):
        torch.manual_seed(5); np.random.seed(5)
        env, ct, s1, m = _mode(L, SB, "probe", n=2, probes={"A": SB.StubActor()}, bank_max=3)
        m.begin_pass(types.SimpleNamespace(segment_id=np.ones((20, 2), np.int16), hold_labels=["stand", "fwd"],
                                           n_steps=20))
        env.unwrapped.episode_length_buf[:] = 3
        obs, _ = env.get_observations()
        g0 = np.random.get_state()[1].copy()
        for t in range(20):
            m.observe(obs)
            with torch.inference_mode():
                m(obs, hist_encoding=True)
            obs = obs + 1.0
        assert np.array_equal(np.random.get_state()[1], g0)
        outs.append(m.bank_arrays()["obs"].copy())
    assert np.array_equal(outs[0], outs[1])


def test_summarize_shares_switches_and_settle_by_wave(L):
    st = np.zeros((6, 4), bool)
    st[:2, :] = True               # settle steps 0-1 in stand mode everywhere
    st[4, 0] = True                # env 0: one more entry -> switches 3
    st[1, 3] = False               # wave-1 env 3 left the settle early
    series = {L.RAW_STAND: st}
    seg = np.array([[0] * 4] * 2 + [[1] * 4] * 4, np.int16)
    comp = types.SimpleNamespace(segment_id=seg, hold_labels=["settle", "turn_l"], wave_of_env=np.array([0, 0, 1, 1]))
    out = L.summarize(series, [6, 6, 6, 3], compiled=comp, settle_label="settle")
    rec = np.arange(6)[:, None] < np.array([6, 6, 6, 3])[None, :]
    assert out["stand_mode_share"] == pytest.approx(st[rec].mean())
    assert out["switches_per_episode_mean"] == pytest.approx((3 + 1 + 1 + 1) / 4)
    assert out["settle_mode_share_by_wave"] == [1.0, 0.75]
    assert out["stand_mode_share_by_hold"]["settle"] == pytest.approx(7 / 8)
    assert L.summarize(series, [6, 6, 6, 3])["settle_mode_share_by_wave"] is None


# ===================================================================================== end to end (fake harness)
def _run_ref(tmp_path, args, seed=0):
    torch.manual_seed(seed)
    old = sys.argv
    sys.argv = [str(tmp_path / "fake_harness.py")] + args
    try:
        g = runpy.run_path(str(tmp_path / "fake_harness.py"), run_name="p2r_harness")
        g["main"]()
    finally:
        sys.argv = old


def _run_launcher(L, tmp_path, args, seed=0):
    torch.manual_seed(seed)
    old = sys.argv
    try:
        L.main([str(LAUNCHER)] + args, harness=tmp_path / "fake_harness.py", environ={})
    finally:
        sys.argv = old


def _episodes(stamp: Path):
    return [dict(np.load(p)) for p in sorted((stamp / "raw").glob("episode_*.npz"))]


@pytest.fixture()
def ckpts(SB, tmp_path):
    for name, seed in (("s1", 1), ("st2", 2), ("pa", 3), ("pb", 4)):
        SB.save_ckpt(tmp_path / f"{name}.pt", seed)
    return {k: str(tmp_path / f"{k}.pt") for k in ("s1", "st2", "pa", "pb")}


@pytest.mark.parametrize("suite", [False, True])
def test_e2e_probe_reproduces_the_unpatched_run_bitwise(L, SB, tmp_path, ckpts, suite):
    base = ["--checkpoint", ckpts["s1"], "--fall", "9:1"] + (["--suite"] if suite else [])
    _run_ref(tmp_path, base + ["--out", str(tmp_path / "ref")])
    _run_launcher(L, tmp_path, ["--p2r-mode", "probe", "--p2r-probe", f"ST2_m20500={ckpts['pa']}",
                                "--p2r-probe", f"C0R_m24993={ckpts['pb']}", "--p2r-bank-max", "10"]
                  + base + ["--out", str(tmp_path / "pr")])
    ref, pr = _episodes(tmp_path / "ref" / "stamp"), _episodes(tmp_path / "pr" / "stamp")
    assert len(ref) == len(pr) == (6 if suite else 3)
    for a, b in zip(ref, pr):
        assert np.array_equal(a["actions"], b["actions"])          # the launcher is inert under probe
        assert b["p2r_stand_mode"].dtype == bool and b["p2r_stand_mode"].shape == b["actions"].shape[:1]
        for name in ("ST2_m20500", "C0R_m24993"):
            v = b[f"p2r_dmu_{name}"]
            assert v.dtype == np.float32 and v.shape == b["actions"].shape[:1] and (v > 0).all()
    meta = json.loads((tmp_path / "pr" / "stamp" / "run_meta.json").read_text())
    blk = meta["p2r_mode"]
    assert CONTRACT_KEYS | {"probes"} <= set(blk)
    assert blk["variant"] == "probe" and blk["enter_rule"] == "latched_stand_slot" and blk["rng_restored"] is True
    assert blk["launcher_sha256"] == L.sha256_file(LAUNCHER)
    assert blk["harness_sha256"] == L.sha256_file(tmp_path / "fake_harness.py")
    assert blk["s1_checkpoint_sha256"] == hashlib.sha256(Path(ckpts["s1"]).read_bytes()).hexdigest()
    assert blk["specialist"] is None and blk["bank"] == "p2r_bank.npz"
    assert set(blk["probes"]) == {"ST2_m20500", "C0R_m24993"}
    assert blk["probes"]["C0R_m24993"]["sha256"] == L.sha256_file(ckpts["pb"])
    bank = np.load(tmp_path / "pr" / "stamp" / "p2r_bank.npz")
    assert set(bank.files) == {"obs", "hold_label", "category", "s1_sigma"}
    assert 0 < bank["obs"].shape[0] <= 10 and bank["obs"].dtype == np.float32
    assert not set(bank["hold_label"]) & {"stand", "settle", "prologue"}
    assert (tmp_path / "pr" / "app_closed").exists()
    # the bank precedes complete.json (the writer's last file)
    assert (tmp_path / "pr" / "stamp" / "p2r_bank.npz").stat().st_mtime_ns <= \
        (tmp_path / "pr" / "stamp" / "complete.json").stat().st_mtime_ns
    if suite:
        # the move_turn wave's settle would inherit the turn wave's latch without the reset re-derivation
        assert blk["settle_mode_share_by_wave"] == [1.0, 1.0]
    else:
        assert blk["settle_mode_share_by_wave"] is None
        assert blk["stand_mode_share_by_hold"]["stand"] == 1.0
        assert blk["stand_mode_share_by_hold"]["turn_l"] == 0.0     # a settled turn stays turn
        assert blk["stand_mode_share_by_hold"]["fwd"] == 0.0


def test_e2e_hold_zeroes_stand_steps_and_matches_before_first_entry(L, SB, tmp_path, ckpts):
    base = ["--checkpoint", ckpts["s1"]]
    _run_ref(tmp_path, base + ["--out", str(tmp_path / "ref")])
    _run_launcher(L, tmp_path, ["--p2r-mode", "hold"] + base + ["--out", str(tmp_path / "hold")])
    ref, hold = _episodes(tmp_path / "ref" / "stamp"), _episodes(tmp_path / "hold" / "stamp")
    for a, b in zip(ref, hold):
        st = b["p2r_stand_mode"]
        assert st.any() and (b["actions"][st] == 0).all()
        assert not (b["actions"][~st] == 0).all(axis=1).any()
        first = int(np.argmax(st))
        assert np.array_equal(a["actions"][:first], b["actions"][:first])
    blk = json.loads((tmp_path / "hold" / "stamp" / "run_meta.json").read_text())["p2r_mode"]
    assert CONTRACT_KEYS <= set(blk) and "probes" not in blk
    assert blk["variant"] == "hold" and blk["bank"] is None and blk["specialist"] is None
    assert not (tmp_path / "hold" / "stamp" / "p2r_bank.npz").exists()
    assert blk["switches_per_episode_mean"] == 2.0                 # enter at the stand hold, exit at the turn


def test_e2e_st2_records_specialist(L, SB, tmp_path, ckpts):
    _run_launcher(L, tmp_path, ["--p2r-mode", "st2", "--p2r-stand-ckpt", ckpts["st2"], "--checkpoint", ckpts["s1"],
                                "--out", str(tmp_path / "st2")])
    blk = json.loads((tmp_path / "st2" / "stamp" / "run_meta.json").read_text())["p2r_mode"]
    assert blk["specialist"] == {"path": ckpts["st2"], "sha256": L.sha256_file(ckpts["st2"])}
    assert blk["rng_restored"] is True and blk["variant"] == "st2"
    ep = _episodes(tmp_path / "st2" / "stamp")[0]
    assert ep["p2r_stand_mode"].any() and not (ep["actions"][ep["p2r_stand_mode"]] == 0).all()


def test_e2e_app_closed_even_when_main_raises(L, SB, tmp_path, ckpts):
    # env 9 does not exist: the stub env raises IndexError inside main(), after the app started
    with pytest.raises(IndexError):
        _run_launcher(L, tmp_path, ["--p2r-mode", "hold", "--checkpoint", ckpts["s1"], "--out", str(tmp_path / "x"),
                                    "--fall", "3:9"])
    assert (tmp_path / "x" / "app_closed").exists()


def test_e2e_unstripped_args_would_break_the_harness_argparse(SB, tmp_path, ckpts):
    """Control: the fake harness's import-time argparse rejects a p2r flag, so stripping is load-bearing."""
    with pytest.raises(SystemExit):
        _run_ref(tmp_path, ["--checkpoint", ckpts["s1"], "--out", str(tmp_path / "y"), "--p2r-mode", "hold"])


def test_refused_harness_flags(L):
    for ns in (dict(zero_actions=True), dict(sample_actions=True), dict(canon="wave"), dict(policy_role="student"),
               dict(save_raw=False)):
        with pytest.raises(SystemExit):
            L._refuse_harness_flags(types.SimpleNamespace(**ns))
    L._refuse_harness_flags(types.SimpleNamespace(zero_actions=False, sample_actions=False, canon="off",
                                                  policy_role="auto", save_raw=True))


# ===================================================================================== run_arm.sh hooks
def _bash(script: str, env: dict) -> subprocess.CompletedProcess:
    return subprocess.run(["bash", "-c", script], env=env, capture_output=True, text=True, timeout=120)


def _env(tmp: Path, **extra) -> dict:
    env = {k: v for k, v in os.environ.items() if not k.startswith(("KRABBY_", "P2_", "P2R_"))}
    camp = tmp / "camp"
    (camp / "logs").mkdir(parents=True, exist_ok=True)
    (tmp / "code" / "parkour").mkdir(parents=True, exist_ok=True)
    env.update(P2_OUT=str(tmp / "out"), CAMPAIGN_DIR=str(camp), P2_CAP="", CODE_ROOT=str(tmp / "code"))
    env.update({k: str(v) for k, v in extra.items()})
    return env


EVAL_STUB = textwrap.dedent("""\
    import json, os, sys, pathlib
    a = sys.argv
    sc = a[a.index('--scenario') + 1]
    out = pathlib.Path(a[a.index('--output-root') + 1]) / sc / ('seed%03d' % int(a[a.index('--env-seed') + 1])) / 'ts'
    out.mkdir(parents=True)
    (out / 'complete.json').write_text('{}')
    (out / 'argv.json').write_text(json.dumps({'argv': a, 'script': a[0],
        'env': {k: v for k, v in os.environ.items() if k.startswith(('P2R_', 'KRABBY_STAMP'))}}))
    """)


def test_run_arm_hooks_default_unchanged(tmp_path):
    env = _env(tmp_path, P2_RUN_ARM_LIB=1)
    r = _bash(f'source {RUN_ARM}\necho "EVAL=$EVAL"', env)
    assert r.stdout.strip() == f"EVAL={tmp_path}/code/parkour/parkour_tasks/parkour_tasks/crab_hex_forward_task/" \
                               "scripts/eval_crab_hex_gait.py", r.stdout + r.stderr


@pytest.mark.parametrize("which", ["row", "lr"])
def test_run_arm_p2_eval_and_env_extra_reach_the_eval(tmp_path, which):
    ev = tmp_path / "p2r_stub_eval_crab_hex_gait.py"
    ev.write_text(EVAL_STUB)
    env = _env(tmp_path, P2_PY=sys.executable, P2_RUN_ARM_LIB=1, P2_EVAL=ev,
               P2_ENV_EXTRA=f"P2R_MODE=st2 P2R_STAND_CKPT={tmp_path}/st2.pt")
    st = tmp_path / "stamp"
    st.mkdir()
    (st / "run_meta.json").write_text(json.dumps({"env_vars": {"KRABBY_STAMP_A": "1"}}))
    call = (f"row_eval N {tmp_path}/m.pt {st} {tmp_path}/man.yaml field_hard__omni 1; echo \"rc=$? dir=$ROW_DIR\""
            if which == "row" else f"lr_eval N {tmp_path}/m.pt {st}; echo \"rc=$? dir=$LR_DIR\"")
    script = f"""
        source {RUN_ARM}
        code_guard() {{ return 0; }}
        drivers_idle() {{ :; }}
        write_sidecar() {{ echo '{{}}' > "$1/code_state.json"; }}
        {call}
    """
    r = _bash(script, env)
    assert "rc=0" in r.stdout, r.stdout + r.stderr
    sc = "field_hard__omni" if which == "row" else "side22_25lr__omni"
    got = json.loads((tmp_path / "out/lr/N" / sc / "seed001/ts/argv.json").read_text())
    assert got["script"] == str(ev)
    assert got["env"] == {"P2R_MODE": "st2", "P2R_STAND_CKPT": f"{tmp_path}/st2.pt", "KRABBY_STAMP_A": "1"}


def test_run_arm_diff_is_only_the_two_hooks():
    """The two hooks (P2_EVAL / P2_ENV_EXTRA) this step added are still exactly where / how many they were: one
    EVAL= assignment, two env lines. (Later campaign steps -- the p2r step-2 graft branch of verify_head_sha /
    fork_source_rd, CHARTER 12 -- legitimately add OTHER lines elsewhere in the file, so this no longer pins an
    exact line NUMBER, only the hooks' own shape.)"""
    s = RUN_ARM.read_text()
    assert s.count("${P2_EVAL:-") == 1 and s.count("${P2_ENV_EXTRA:-}") == 2
    assert ("EVAL=${P2_EVAL:-$CODE_ROOT/parkour/parkour_tasks/parkour_tasks/crab_hex_forward_task/"
           "scripts/eval_crab_hex_gait.py}") in s.splitlines()


# ===================================================================================== controls (the guards bite)
def test_control_without_reset_rederive_the_settle_inherits_the_latch(L, SB, tmp_path, ckpts, monkeypatch):
    monkeypatch.setattr(L.P2RMode, "rederive_on_reset", lambda self: self.cmd_term.stand_mask().reshape(-1).bool())
    _run_launcher(L, tmp_path, ["--p2r-mode", "hold", "--checkpoint", ckpts["s1"], "--suite",
                                "--out", str(tmp_path / "c")])
    blk = json.loads((tmp_path / "c" / "stamp" / "run_meta.json").read_text())["p2r_mode"]
    assert blk["settle_mode_share_by_wave"][0] == 1.0 and blk["settle_mode_share_by_wave"][1] == 0.0


def test_control_without_rng_restore_the_probe_run_diverges(L, SB, tmp_path, ckpts, monkeypatch):
    monkeypatch.setattr(L, "restore_rng", lambda st: None)
    monkeypatch.setattr(L, "rng_equal", lambda a, b: True)
    base = ["--checkpoint", ckpts["s1"]]
    _run_ref(tmp_path, base + ["--out", str(tmp_path / "ref")])
    _run_launcher(L, tmp_path, ["--p2r-mode", "probe", "--p2r-probe", f"A={ckpts['pa']}"] + base
                  + ["--out", str(tmp_path / "pr")])
    ref, pr = _episodes(tmp_path / "ref" / "stamp"), _episodes(tmp_path / "pr" / "stamp")
    assert not all(np.array_equal(a["actions"], b["actions"]) for a, b in zip(ref, pr))
