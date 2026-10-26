"""Stand-height campaign (``2026-10-13_stand_height``): the step-8 RAISE / LOWER VIDEO tools on CPU (no Isaac / Kit /
GPU; VIDEO ONLY, read by no gate).

Pinned here:

* ``scripts/sh_video_schedule.py`` (the generator): every plan passes the launcher's manifest rules
  (``sh_eval_common.height_row_reasons``: heights in the frozen [h_min, h_max], "stand" labels, one facing, walk first
  and last, ``|dh| / 0.08 + 1 + 2`` s per stand, ``|h - 1.064| / 0.08 + 1.5 + 2`` s per walk after a stand -- re-derived
  here independently) and the harness validator; the raise plan spans h_min..h_max with in-stand raises AND lowers and
  restored hand-backs (~150 s); the smoke is ~20 s; the random plan is deterministic per seed, differs across seeds,
  stays within ``--total-s``; the written YAML re-loads through ``gait_eval.schedule.load_manifest`` AND
  ``sh_eval_common.height_row`` and its per-step height table (the launcher's manifest path) gives every step its
  hold's height; ``--dry-run`` writes nothing;
* ``scripts/sh_video_core.py``: ``height_step`` is BITWISE the stand_height launcher's own ``__call__`` (the executed
  tensor it swaps in at ``env.step``) over raise / lower / restore / reset, deadband 0 and 0.25; it hands the deadband
  to ``sh_core.stand_action`` and composes sh_core only; the readings (feet loaded, homed, mode), the per-hold
  read-outs and the caption text (mode, h_cmd, limited h, h_meas, feet loaded, homed);
* ``scripts/sh_record_video.py`` compiles (never imported: it boots Kit), routes through ``sh_video_core.height_step``
  with ``--deadband`` (no control logic of its own), raises the wrapper clip to 4.8, reads per-hold heights through
  ``sh_eval_common.height_row`` / ``height_table``; its defaults name the final head configuration;
* ``scripts/sh_record_video.sh``: ``bash -n``; its busy pattern matches every Isaac command line and not its CPU
  helpers; ``--dry-run`` prints a ``timeout --signal=TERM`` WITHOUT ``-k``, ``MemoryMax=24G`` / ``MemorySwapMax=0``,
  launches and writes nothing; a real launch refuses (exit 3) while a (fake) ``ps`` shows an Isaac process, before
  generating or launching anything.
"""
from __future__ import annotations

import importlib.util
import math
import os
import py_compile
import re
import subprocess
import sys
import types
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
yaml = pytest.importorskip("yaml")
np = pytest.importorskip("numpy")

REPO = Path(__file__).resolve().parents[2]
TASK = REPO / "parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task"
CAMP = TASK / "experiments/2026-10-13_stand_height"
SCRIPTS = CAMP / "scripts"
VIDEOS = CAMP / "videos"
GEN_PATH = SCRIPTS / "sh_video_schedule.py"
VC_PATH = SCRIPTS / "sh_video_core.py"
REC_PATH = SCRIPTS / "sh_record_video.py"
SH_PATH = SCRIPTS / "sh_record_video.sh"
LAUNCHER = SCRIPTS / "sh_eval_crab_hex_gait.py"
P2L_PATH = TASK / "experiments/2026-10-07_omni_phase2/scripts/p2r_mode_eval_crab_hex_gait.py"
PRED_SCRIPTS = TASK / "experiments/2026-10-11_stand_network/scripts"
W, P0, P1, NP = 1149, 207, 216, 75
SEEDS = (20261014, 0, 1, 2, 3, 7, 42, 99, 123, 2024, 31337, 777)


def _load(path: Path, name: str):
    """By path, never writing ``__pycache__`` next to the campaign / phase-2 / predecessor scripts."""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    old = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.dont_write_bytecode = old
    return mod


@pytest.fixture(scope="module")
def GEN():
    return _load(GEN_PATH, "_t_shv_gen")


@pytest.fixture(scope="module")
def VC(GEN):
    return GEN.VC


@pytest.fixture(scope="module")
def CORE(VC):
    return VC.CORE


@pytest.fixture(scope="module")
def EC(VC):
    return VC.EC


@pytest.fixture(scope="module")
def S(GEN):
    return GEN.schedule_module()


@pytest.fixture(scope="module")
def HDR(GEN):
    return GEN.reference_header()


@pytest.fixture(scope="module")
def gens(GEN, HDR):
    out = {"raise": GEN.generate("raise", hdr=HDR), "smoke": GEN.generate("smoke", hdr=HDR)}
    for s in SEEDS:
        out[f"random{s}"] = GEN.generate("random", s, 150.0, hdr=HDR)
    return out


# ===================================================================================== the schedule
def _independent_rule_check(gen, CORE):
    """The scenarios_sh.yaml rules, re-derived here (not via sh_eval_common)."""
    holds, heights = gen["holds"], gen["heights"]
    assert holds[0].speed > 0 and holds[-1].speed > 0                               # walk first and last
    level, prev = 1.064, None
    for h, hh in zip(holds, heights):
        assert h.face == 0.0
        if h.speed == 0.0:
            assert h.label == "stand" and hh is not None
            assert 0.719 - 1e-12 <= hh <= 1.346 + 1e-12 and CORE.H_MIN <= hh <= CORE.H_MAX
            assert h.hold_s + 1e-9 >= abs(hh - level) / 0.08 + 1.0 + 2.0, (h, hh, level)
            level, prev = hh, hh
        else:
            assert h.label != "stand" and hh is None
            if prev is not None:
                assert h.hold_s + 1e-9 >= abs(prev - 1.064) / 0.08 + 1.5 + 2.0, (h, prev)
            level, prev = 1.064, None
        assert abs(h.hold_s * 10 - round(h.hold_s * 10)) < 1e-9                    # whole deciseconds


def test_every_plan_passes_the_launcher_rules_and_the_harness_validator(gens, GEN, EC, CORE, S, HDR):
    for name, g in gens.items():
        _independent_rule_check(g, CORE)
        assert EC.height_row_reasons(GEN.hold_dicts(g["holds"]), g["heights"]) == [], name
        S.validate_scenario(GEN._scenario(S, HDR, g["holds"], g["row_id"]))
        assert g["summary"]["min_slack_s"] >= 0.0


def test_raise_plan_spans_the_range_with_in_stand_raises_lowers_and_restores(gens, CORE):
    g = gens["raise"]
    sm = g["summary"]
    hs = [h for h in g["heights"] if h is not None]
    assert min(hs) == CORE.H_MIN and max(hs) == CORE.H_MAX                         # h_min 0.719 .. h_max 1.346
    assert 1.064 in hs and 1.300 in hs and 0.750 in hs
    seq = g["heights"][1:7]
    assert seq == [1.064, 1.3, 0.75, 1.15, 0.85, 1.064]                            # the brief's example, one stand
    assert sm["in_stand_raises"] >= 3 and sm["in_stand_lowers"] >= 3 and sm["restores"] >= 2
    assert sm["n_walks"] >= 4                                                       # walks between the stands
    assert 140.0 <= sm["video_s"] <= 165.0                                          # ~150 s
    assert g["row_id"] == "sh_raise__omni"


def test_smoke_plan_is_short_and_covers_raise_lower_restore(gens):
    g = gens["smoke"]
    sm = g["summary"]
    assert 15.0 <= sm["video_s"] <= 25.0
    assert g["heights"] == [None, 1.2, 1.0, None]
    assert sm["in_stand_lowers"] == 1 and sm["restores"] == 1
    assert g["row_id"] == "sh_smoke__omni"


def test_random_plan_heights_lengths_and_total(gens, CORE, S, HDR):
    fwd, lat, back = (HDR["speed_ellipse"][k] for k in ("fwd", "lat", "back"))
    stand_lens = set()
    for s in SEEDS:
        g = gens[f"random{s}"]
        sm = g["summary"]
        assert sm["video_s"] <= 150.0 + 1e-9 and sm["video_s"] >= 110.0, (s, sm["video_s"])
        assert sm["n_stands"] >= 3 and sm["n_walks"] >= 3
        for h, hh in zip(g["holds"], g["heights"]):
            if hh is None:
                cap = S.ellipse_cap(S.wrap_to_pi(h.dir - h.face), fwd, lat, back)
                assert 0.12 <= h.speed <= cap + 1e-12
            else:
                stand_lens.add(h.hold_s)
                assert CORE.H_MIN <= hh <= CORE.H_MAX
    assert len(stand_lens) > 10                                                     # lengths are random


def test_random_plan_is_deterministic_per_seed(GEN, HDR):
    def key(g):
        return [(h.label, h.hold_s, h.as_world()) for h in g["holds"]], g["heights"]

    for s in (0, 7, 20261014):
        assert key(GEN.generate("random", s, 150.0, hdr=HDR)) == key(GEN.generate("random", s, 150.0, hdr=HDR))
    assert key(GEN.generate("random", 1, 150.0, hdr=HDR)) != key(GEN.generate("random", 2, 150.0, hdr=HDR))
    # the fixed plans ignore the seed
    assert key(GEN.generate("raise", 1, hdr=HDR)) == key(GEN.generate("raise", 2, hdr=HDR))


def test_rules_refuse_a_short_hold(GEN, S, HDR, gens):
    g = gens["raise"]
    holds = list(g["holds"])
    i = 3                                                                            # a 0.750 stand after 1.300
    holds[i] = GEN._hold(S, "stand", 0.0, 0.0, int(round((abs(0.75 - 1.3) / 0.08 + 2.5) * 10)))
    with pytest.raises(ValueError, match="hold_s"):
        GEN.check_rules(S, HDR, holds, g["heights"], g["row_id"])
    with pytest.raises(ValueError, match="frozen range"):
        GEN.check_rules(S, HDR, g["holds"], [None if h is None else 1.40 for h in g["heights"]], g["row_id"])


@pytest.mark.parametrize("plan,extra", [("raise", []), ("smoke", []), ("random", ["--seed", "7", "--total-s", "90"])])
def test_cli_writes_and_the_launcher_path_reads_the_heights(GEN, EC, CORE, S, tmp_path, plan, extra):
    assert GEN.main(["--plan", plan, *extra, "--out-dir", str(tmp_path)]) == 0
    yml = tmp_path / f"scenarios_sh_video_{plan}.yaml"
    assert (tmp_path / f"sh_video_schedule_{plan}.json").is_file()
    gen = GEN.generate(plan, *( [7, 90.0] if plan == "random" else []))
    row = EC.height_row(yml, f"sh_{plan}__omni")                                    # the launcher's manifest loader
    assert row["hold_heights"] == gen["heights"] and row["labels"] == [h.label for h in gen["holds"]]
    sc = {s.id: s for s in S.load_manifest(yml)[0]}[f"sh_{plan}__omni"]
    comp = S.compile_schedule(sc, dt=0.02, num_envs=1)
    tab = EC.height_table(comp.segment_id, row["hold_heights"])                     # the launcher's per-step map
    seg = np.asarray(comp.segment_id)[:, 0]
    for i, hh in enumerate(row["hold_heights"]):
        vals = tab[seg == i, 0]
        assert vals.size and np.all(vals == (CORE.H_SETTLED if hh is None else hh)), (i, hh)
    first = row["hold_heights"][0]
    assert np.all(tab[seg < 0, 0] == (CORE.H_SETTLED if first is None else first))  # the prologue: hold 0's command


def test_dry_run_writes_nothing(GEN, tmp_path, capsys):
    out = tmp_path / "v"
    assert GEN.main(["--plan", "raise", "--out-dir", str(out), "--dry-run"]) == 0
    assert not out.exists()
    assert "would write" in capsys.readouterr().out


# ===================================================================================== the controller path
class _Est:
    """A deterministic estimator double: ``inference(prop) -> (N, k)``."""

    def __init__(self, k: int, scale: float):
        self.k, self.scale = k, scale

    def inference(self, prop):
        return (prop.sum(dim=1, keepdim=True) * self.scale + torch.arange(self.k, dtype=prop.dtype)).to(prop.dtype)


def _s1(x, hist_encoding=False):
    assert hist_encoding is True
    return 1.6 * torch.tanh(x[:, 40:58] * 0.7)                                       # saturates beyond +-1


def _stand(x, hist_encoding=False):
    assert hist_encoding is True
    # reads the height column (15) and a restore-zeroed column (8): small AND large residuals (the deadband acts)
    return 0.6 * torch.tanh(x[:, 100:118] * 0.5 + 2.0 * x[:, 15:16] - x[:, 8:9])


def _scenario_arrays(n=2):
    """(T,) segment ids: walk 20 / stand 1.2 150 / stand 0.9 200 (reached) / walk 200 / stand 1.3 120 (not reached) /
    walk 160; env 0 resets at t = 420 (50 steps into the restore from 0.9: a cut), env 1 never."""
    seg = np.array([0] * 20 + [1] * 150 + [2] * 200 + [3] * 200 + [4] * 120 + [5] * 160)
    heights = [None, 1.2, 0.9, None, 1.3, None]
    T = seg.size
    segn = np.repeat(seg[:, None], n, axis=1)
    ep = np.zeros((T, n), dtype=np.int64)
    for e in range(n):
        last = 0
        for t in range(T):
            if e == 0 and t == 420:
                last = t
            ep[t, e] = t - last
    slot = np.isin(seg, (1, 2, 4))
    return seg, segn, heights, ep, slot


def _obs_seq(T, n, seed=0):
    g = torch.Generator().manual_seed(seed)
    obs = torch.randn(T, n, W, generator=g)
    obs[:, :, 29:35] *= 0.0005                                                      # hips near default: restores complete
    return obs


def _launcher_mode(SHL, deadband, h_np, n, est, slot_box, ep_box):
    """The launcher's ``_HeightMixin`` (its ``__call__`` and ``_start_pass_state`` / ``_alloc``, unmodified) on a bare
    instance carrying exactly what they read (P2L / SNL wiring replaced by these attributes)."""
    class _M(SHL._HeightMixin):
        pass

    m = object.__new__(_M)
    T = int(h_np.shape[0])
    m.__dict__.update(
        s1_policy=_s1, sh_bound=True, _cur={"stand": None, "rst": None, "T": T, "t": 0}, _ret=None, _exec=None,
        alive=None, ctl=None, _h_np=h_np, _h_dev=None, _compiled_dt=0.02, net=True, old_path=False,
        stand=(_stand, est), p0=P0, p1=P1, num_prop=NP, stand_mask=1, deadband=float(deadband),
        controller="stand_net", restore="armed", acc=None, calls=0, ctl_record=None,
        unwrapped=types.SimpleNamespace(step_dt=0.02, episode_length_buf=None),
        rederive_on_reset=lambda: slot_box[0])
    m.unwrapped.episode_length_buf = ep_box
    return m


@pytest.mark.parametrize("deadband", [0.0, 0.25])
def test_height_step_is_bitwise_the_launchers_call(VC, EC, deadband):
    SHL = _load(LAUNCHER, f"_t_shv_launcher_{int(deadband * 100)}")
    n = 2
    seg, segn, heights, ep, slot = _scenario_arrays(n)
    T = seg.size
    h_np = EC.height_table(segn, heights)
    obs_all = _obs_seq(T, n)
    est = _Est(P1 - P0, 0.01)
    slot_box = [None]
    ep_box = torch.zeros(n, dtype=torch.long)
    m = _launcher_mode(SHL, deadband, h_np, n, est, slot_box, ep_box)
    h_dev = torch.as_tensor(h_np, dtype=torch.float64)
    ctl = VC.new_controller(n, "cpu", dt=0.02, h_cmd0=h_dev[0])
    seen = {"restoring": 0, "completed": 0, "cut": 0, "stand_rows": 0, "beyond1": 0, "zeroed": 0}
    with torch.inference_mode():
        for t in range(T):
            obs = obs_all[t].clone()
            slot_t = torch.full((n,), bool(slot[t]), dtype=torch.bool)
            ep_box.copy_(torch.as_tensor(ep[t]))
            slot_box[0] = slot_t
            m(obs.clone())
            want = m._exec
            m._ret = m._exec = None
            obs0 = obs.clone()
            st = VC.height_step(ctl, obs, slot=slot_t, reset_rows=torch.as_tensor(ep[t] == 0), h_cmd=h_dev[t],
                                s1_policy=_s1, stand_policy=_stand, stand_estimator=est, p0=P0, p1=P1, num_prop=NP,
                                mask=1, deadband=deadband)
            assert torch.equal(obs, obs0)                                            # the harness's obs never written
            assert torch.equal(st.actions, want), t
            assert torch.equal(st.hs.h_lim, m.ctl.h_lim) and torch.equal(st.hs.restoring, m.ctl.restoring)
            seen["restoring"] += int(st.hs.restoring.sum())
            seen["completed"] += int(st.hs.completed.sum())
            seen["cut"] += int(st.cut.sum())
            seen["stand_rows"] += int(st.hs.stand_eff.sum())
            seen["beyond1"] += int((st.actions[st.hs.stand_eff][:, 6:].abs() > 1.0).sum())
            r = torch.clamp(st.a_net, -1, 1)[:, 6:18]
            seen["zeroed"] += int(((r.abs() < 0.25) & (r != 0)).sum())
    # the sequence exercised what it claims: restores (completed + one cut by the reset), q beyond +-1, small residuals
    assert seen["restoring"] > 0 and seen["completed"] >= 2 and seen["cut"] >= 1, seen
    assert seen["beyond1"] > 0 and seen["zeroed"] > 0, seen
    assert ctl.counts() == m.ctl.counts()


def test_height_step_composes_sh_core_with_the_deadband(VC, monkeypatch):
    calls = {}
    real = {k: getattr(VC.CORE, k) for k in ("stand_action", "stand_net_input", "route")}

    def spy(name):
        def f(*a, **k):
            calls.setdefault(name, []).append(k)
            return real[name](*a, **k)
        return f

    for k in real:
        monkeypatch.setattr(VC.CORE, k, spy(k))
    n = 3
    ctl = VC.new_controller(n, "cpu", dt=0.02, h_cmd0=torch.full((n,), 1.2, dtype=torch.float64))
    obs = _obs_seq(1, n)[0]
    est = _Est(P1 - P0, 0.01)
    st = VC.height_step(ctl, obs, slot=torch.tensor([True, False, True]), reset_rows=torch.zeros(n, dtype=torch.bool),
                        h_cmd=torch.full((n,), 1.2, dtype=torch.float64), s1_policy=_s1, stand_policy=_stand,
                        stand_estimator=est, p0=P0, p1=P1, num_prop=NP, mask=1, deadband=0.25)
    assert calls["stand_action"] == [{"deadband": 0.25}]
    (k_in,) = calls["stand_net_input"]
    assert k_in["mask"] is True and torch.equal(k_in["h_cmd_limited"], st.hs.h_lim)
    assert torch.equal(k_in["restoring"], st.hs.restoring)
    assert len(calls["route"]) == 1
    # S1 rows are clamp(S1, +-1); stand rows q(h_lim) + deadband(clamp(a_net))
    assert torch.equal(st.actions[1], torch.clamp(st.a_s1[1], -1, 1))
    want, _ = real["stand_action"](st.a_net, st.hs.h_lim, deadband=0.25)
    assert torch.equal(st.actions[[0, 2]], want[[0, 2]])
    with pytest.raises(ValueError):
        VC.height_step(ctl, obs, slot=torch.ones(n, dtype=torch.bool), reset_rows=torch.zeros(n, dtype=torch.bool),
                       h_cmd=None, s1_policy=_s1, stand_policy=_stand, stand_estimator=est, p0=P0, p1=P1,
                       num_prop=NP, mask=1, deadband=1.5)


def test_video_core_has_no_control_logic_of_its_own():
    src = VC_PATH.read_text()
    body = src.split("def height_step", 1)[1].split("\ndef ", 1)[0]
    for needle in ("ctl.reset(reset_rows, \"walk\", h_cmd=h_cmd)", "ctl.step(slot, h_cmd, obs)",
                   "CORE.stand_net_input(", "CORE.stand_action(a_net, hs.h_lim, deadband=d)",
                   "CORE.route(hs.stand_eff, a_stand, a_s1)"):
        assert needle in body, needle
    for banned in ("torch.where", "torch.clamp", "clamp(", "+ q", "q18("):
        assert banned not in body, banned


def test_readings_and_modes(VC):
    f = np.array([[25.0, 19.9, 100.0, 20.0, 0.0, 30.0]])
    z = np.array([[0.10, 0.10, 0.16, 0.15, 0.0, 0.05]])
    tz = np.zeros((1, 6))
    assert VC.feet_loaded_mask(f, z, tz).tolist() == [[True, False, False, True, False, True]]
    assert VC.feet_loaded_mask(torch.tensor(f), torch.tensor(z) + 2.0, torch.full((1, 6), 2.0)).tolist() == \
        VC.feet_loaded_mask(f, z + 2.0, tz + 2.0).tolist()                           # the terrain is subtracted
    c = VC.cam_abs(np.array([0.05, -0.2, 2 * math.pi - 0.03]))
    assert np.allclose(c, [0.05, 0.2, 0.03])
    assert torch.allclose(VC.cam_abs(torch.tensor([0.05, -0.2, 2 * math.pi - 0.03])), torch.tensor(c).float())
    assert VC.is_homed(0.10) and not VC.is_homed(0.1001)
    assert VC.mode_of(False, False) == "walk" and VC.mode_of(True, False) == "stand"
    assert VC.mode_of(False, True) == "restoring" and VC.mode_of(True, True) == "restoring"


def test_caption_lines_carry_mode_heights_feet_homed(VC):
    base = dict(title="S1 s18 + stand-height net SH_T18 s18 db0.25", hold_i=2, n_holds=6, hold_label="stand",
                hold_kind="stand", hold_t=3.2, hold_len=8.0, t_s=20.0, total_s=40.0, stand_tag="SH_T18 s18 db0.25",
                s1_tag="S1 s18", deadband=0.25, h_cmd=1.3, h_lim=1.172, h_meas=1.1684, feet_loaded=6, homed=True,
                cam_abs_max=0.031, cmd_speed=0.0, cmd_dir_deg=0.0, cmd_face_deg=0.0, next_text="NEXT x",
                last_event=None, heading_deg=1.0, speed_xy=0.01)
    for mode, word in (("walk", "MODE: WALK"), ("stand", "MODE: STAND"), ("restoring", "MODE: RESTORING")):
        lines = VC.caption_lines(mode=mode, **base)
        text = "\n".join(t for t, _c in lines)
        assert word in text
        assert "cmd 1.300 m" in text and "limited 1.172 m" in text and "measured 1.168 m" in text
        assert "FEET loaded 6/6" in text and "HOMED yes" in text
        assert all(isinstance(c, tuple) and len(c) == 3 for _t, c in lines)
    lines = VC.caption_lines(mode="stand", **dict(base, homed=False, feet_loaded=3, last_event="restore TIMED OUT x"))
    text = "\n".join(t for t, _c in lines)
    assert "HOMED NO" in text and "FEET loaded 3/6" in text and "deadband 0.25" in text
    assert lines[-1][1] == VC.RED
    with pytest.raises(ValueError):
        VC.caption_lines(mode="hover", **base)
    nt = VC.next_hold_text(0, ["fwd", "stand", "lat_l"], ["walk", "stand", "walk"], [None, 1.3, None], [75, 475, 875],
                           0.02, 25.0)
    assert "stand 1.300 m" in nt and "t=9.5 s" in nt
    assert VC.next_hold_text(2, ["a", "b", "c"], ["walk"] * 3, [None] * 3, [0, 1, 2], 0.02, 25.0).startswith("LAST")


def test_per_hold_readouts(VC):
    dt = 0.02
    T = 500
    h_cmd = np.full(T, 1.2)
    h_lim = np.minimum(1.064 + 0.08 * dt * np.arange(1, T + 1), 1.2)
    h_meas = h_lim - 0.002
    feet = np.full(T, 6)
    idx = np.arange(T)
    s = VC.stand_hold_height_stats(h_cmd, h_lim, h_meas, feet, idx, dt)
    k = int(np.nonzero(h_lim == 1.2)[0][0])
    assert s["h_cmd"] == 1.2 and s["arrival_s"] == pytest.approx(k * dt) and s["slack_s"] == pytest.approx((T - k) * dt)
    assert s["window"]["median_abs_err_m"] == pytest.approx(0.002) and s["window"]["feet_min"] == 6
    short = VC.stand_hold_height_stats(h_cmd, h_lim, h_meas, feet, idx[:k + 60], dt)
    assert short["window"] is None                                                   # window opens before arrival + 1 s
    assert VC.stand_hold_height_stats(h_cmd, h_lim, h_meas, feet, idx, dt, first_fall_step=0)["rows"] == 0
    r = np.zeros(T)
    r[10:130] = 1
    e = np.zeros(T)
    e[10] = 1
    c = np.zeros(T)
    c[130] = 1
    rs = VC.restore_hold_stats(r, e, c, np.zeros(T), idx, dt)
    assert rs["engaged"] == 1 and rs["completed"] == 1 and rs["timed_out"] == 0
    assert rs["restoring_s"] == pytest.approx(120 * dt) and rs["completed_at_s"] == pytest.approx(130 * dt)


def test_caption_tags_come_from_the_checkpoint(VC):
    sd = {"seed": 18, "iterations": 5000, "smoke": False}
    assert VC.stand_tag_from_ckpt(sd, "/x/runs/SH_T18/model_5000.pt", 0.25) == "SH_T18 s18 db0.25"
    assert VC.stand_tag_from_ckpt(sd, "/x/runs/SH_T18/model_2500.pt", 0.0) == "SH_T18 s18 (model_2500) no-db"
    assert VC.stand_tag_from_ckpt(dict(sd, smoke=True), "/x/runs/SH_T18_SMOKE/model_5000.pt", 0.25).endswith(
        "SMOKE db0.25")
    assert VC.s1_tag_from_ckpt("/l/2026-10-07_07-50-10_seed18_X08_p2a_A15pB/model_19994.pt",
                               is_stand_nets_s1=True) == "S1 s18"


# ===================================================================================== recorder (static)
def test_recorder_compiles_routes_through_the_core_and_names_the_head(tmp_path):
    py_compile.compile(str(REC_PATH), cfile=str(tmp_path / "rec.pyc"), doraise=True)
    src = REC_PATH.read_text()
    for needle in ("crab_hex_omni_teacher/2026-10-07_07-50-10_seed18_X08_p2a_A15pB/model_19994.pt",
                   "outputs/omni_plan/stand_height/runs/SH_T18/model_5000.pt",
                   "phase2/lr/SH_T18/hraise__omni/seed001",
                   'parser.add_argument("--stand-mask", type=int, choices=(0, 1), default=1',
                   'parser.add_argument("--deadband", type=float, default=0.0',
                   "DEADBAND = CORE.check_deadband(args_cli.deadband)",
                   "VC.height_step(ctl, obs, slot=slot, reset_rows=reset_rows, h_cmd=h_cmd_t, s1_policy=ctx.policy",
                   "num_prop=ctx.num_prop, mask=ctx.stand_mask, deadband=DEADBAND)",
                   "ctl = VC.new_controller(1, ctx.device, dt=dt, h_cmd0=h_dev[0])",
                   "HEIGHT_ROW = EC.height_row(args_cli.manifest, args_cli.sequence)",
                   "h_np = EC.height_table(compiled.segment_id, heights)",
                   "reset_rows = ctx.unwrapped.episode_length_buf.reshape(-1) == 0",
                   "slot = ctx.slot_rule.rederive_on_reset()",
                   "env.clip_actions = CORE.HARNESS_CLIP", "EC.S1_TRAIN_CLIP",
                   "P2L.build_with_template(make_on_policy_runner", "P2L.StandInEnv(env, template)",
                   "P2L.save_rng()", "P2L.restore_rng(before)", "P2L.arm_slot_latch(", "SN.env_reasons(WANT_ENV)",
                   "CORE.terrain_z_under_feet(fpos, mesh, raycast_mesh)", "CORE.h_meas(", "VC.feet_loaded_mask(",
                   "VC.caption_lines(", "imageio.get_writer(", '"residual_deadband": (_ref_db, DEADBAND)',
                   "EC.ckpt_kind(STAND_SHA, S1_SHA)", "CORE.deadband_parity("):
        assert needle in src, needle
    assert "RecordVideo" not in src.replace("gym ``RecordVideo``", "")
    # no control logic of its own: the limiter / restore / input / router live in sh_core, called via height_step
    for banned in ("HeightController(", "stand_net_input(", "CORE.stand_action(", "CORE.route(", "q18(",
                   "apply_deadband("):
        assert banned not in src, banned
    assert re.search(r'"--stand-tag", type=str, default=None', src) and re.search(r'"--label", type=str, default=None',
                                                                                   src)


def test_p2l_and_harness_pieces_the_recorder_calls_exist():
    p2l = _load(P2L_PATH, "_t_shv_p2l")
    for name in ("build_with_template", "StandInEnv", "_load_extra", "save_rng", "restore_rng", "rng_equal",
                 "arm_slot_latch"):
        assert callable(getattr(p2l, name)), name
    assert callable(getattr(p2l.P2RMode, "rederive_on_reset", None))
    harness = (TASK.parent / "crab_hex_forward_task/scripts/eval_crab_hex_gait.py").read_text()
    for fn in ("_wrap_to_pi", "_observed_command_slots", "_apply_env_overrides", "_install_schedule_override_world"):
        assert re.search(rf"^def {fn}\(", harness, flags=re.M), fn


def test_reference_stamp_defaults_match_the_launcher_record(VC, EC, CORE):
    """The recorder's reference checks read these run_meta names (skipped where the stamp is not on this machine)."""
    seed_dir = REPO / "parkour/outputs/omni_plan/phase2/lr/SH_T18/hraise__omni/seed001"
    stamps = sorted(d for d in seed_dir.iterdir() if (d / "run_meta.json").is_file()) if seed_dir.is_dir() else []
    if not stamps:
        pytest.skip("the SH_T18 hraise es1 stamp is not on this machine")
    import json  # noqa: PLC0415

    rm = json.loads((stamps[-1] / "run_meta.json").read_text())
    pm = rm["p2r_mode"]
    blk = pm[EC.BLOCK_KEY]
    assert pm["variant"] == EC.VARIANT and blk["control"] == VC.CONTROLLER and blk["restore"] == VC.RESTORE
    assert blk["forward"] == VC.SH_FORWARD and CORE.recorded_deadband(blk) == 0.25
    assert set(blk["core"]) >= {"sh_core_sha256", "sh_kin_sha256", "sn_common_sha256", "restance_sha256",
                                "facts_json_sha256"}
    assert {"wrapper_sha256", "common_sha256", "p2r_launcher_sha256", "controller"} <= set(blk)
    ctl = VC.new_controller(1, "cpu", dt=float(rm["dt"]))
    assert dict(ctl.record(), restore=VC.RESTORE) == blk["controller"]


# ===================================================================================== the .sh launcher
def _sh_pat(name: str) -> str:
    m = re.search(rf"^{name}='([^']*)'$", SH_PATH.read_text(), flags=re.M)
    assert m, f"{name} not found"
    return m.group(1)


def test_launcher_busy_pattern():
    pat, nonkit = re.compile(_sh_pat("ISAAC_PAT")), re.compile(_sh_pat("NONKIT_PAT"))
    py = "/home/nickmagus/krabby/isaac_venv/bin/python"
    sh = "/x/2026-10-13_stand_height/scripts"
    busy = [f"{py} {sh}/sh_record_video.py --headless --prefix a", f"{py} {sh}/sh_dagger.py --head x --mode dagger",
            f"{py} {sh}/sh_table_test.py --headless", f"{py} /x/sh_eval_crab_hex_gait.py --sh-controller r_h",
            f"{py} -u /x/scripts/rsl_rl/train.py --task X", f"{py} /x/sn_record_video.py --headless",
            f"{py} /x/eval_crab_hex_gait.py", f"{py} /x/play_stream_video.py", f"{py} /x/run_phases.py",
            f"{py} sh_record_video.py --headless --prefix a", f"{py} sh_table_test.py --headless"]
    idle = [f"{py} -B {sh}/sh_video_schedule.py --plan raise", f"{py} -B {sh}/sh_video_core.py",
            f"{py} {sh}/sh_dagger.py --read RUN --premise 1:2", f"{py} -m pytest -q", f"{py} {sh}/sh_read.py --head x",
            f"{py} sh_read.py --spec x", f"{py} -B sh_video_schedule.py --plan raise", f"{py} sh_core.py",
            f"bash -c {py} {sh}/sh_record_video.py", f"timeout --signal=TERM 5400 {py} {sh}/sh_record_video.py",
            f"systemd-run --user --scope -q timeout 1 {py} {sh}/sh_record_video.py",
            "grep -E ^[^ ]*[i]saac_venv/bin/python"]
    for line in busy:
        assert pat.search(line) and not nonkit.search(line), line
    for line in idle:
        assert not pat.search(line) or nonkit.search(line), line


def test_launcher_never_sigkills_kit():
    src = SH_PATH.read_text()
    code = "\n".join(ln for ln in src.splitlines() if not ln.lstrip().startswith("#"))
    assert "timeout --signal=TERM \"$TMO\"" in code
    assert not re.search(r"timeout[^\n]*\s-k\b", code) and "--kill-after" not in code
    assert "KILL_AFTER" not in code and "kill -9" not in code and "SIGKILL" not in code
    assert "pgrep" not in code
    for needle in ('-p "MemoryMax=$MEM" -p MemorySwapMax=0', "setsid nohup", 'MEM=${SH_VIDEO_MEMORY_MAX:-24G}',
                   "du -s --block-size=1G \"$DDC\""):
        assert needle in code, needle


def _fake_bin(tmp_path, ps_line: str | None):
    fake = tmp_path / "bin"
    fake.mkdir(exist_ok=True)
    marker = tmp_path / "LAUNCHED"
    for tool in ("systemd-run", "setsid", "nohup"):
        (fake / tool).write_text(f"#!/bin/sh\ntouch {marker}\nexit 1\n")
        (fake / tool).chmod(0o755)
    if ps_line is not None:
        (fake / "ps").write_text(
            "#!/bin/sh\ncase \"$*\" in *pid,args*) printf '  4242 %s\\n' \"$PS_LINE\" ;; *) printf '%s\\n' \"$PS_LINE\" "
            ";; esac\n")
        (fake / "ps").chmod(0o755)
    env = dict(os.environ, PATH=f"{fake}:{os.environ.get('PATH', '')}", SH_DDC=str(tmp_path / "ddc"),
               SH_VIDEO_OUT=str(tmp_path / "out"), CUDA_VISIBLE_DEVICES="", PS_LINE=ps_line or "",
               SH_PY=sys.executable)
    return env, marker


ISAAC_LINE = ("/home/nickmagus/krabby/isaac_venv/bin/python /x/2026-10-13_stand_height/scripts/sh_dagger.py "
              "--head SH_T16 --mode dagger")
HEAD = str(REPO / "parkour/outputs/omni_plan/stand_height/runs/SH_T18/model_5000.pt")


@pytest.mark.parametrize("busy", [True, False])
def test_launcher_dry_run_command_line_and_busy_report(tmp_path, busy):
    subprocess.run(["bash", "-n", str(SH_PATH)], check=True)
    env, marker = _fake_bin(tmp_path, ISAAC_LINE if busy else "/usr/bin/bash idle")
    before = {p.name: p.read_bytes() for p in VIDEOS.iterdir()} if VIDEOS.is_dir() else {}
    for extra, row, tmo in ((["--smoke"], "sh_smoke__omni", "1800"), ([], "sh_raise__omni", "5400"),
                            (["--seed", "7", "--total-s", "90"], "sh_random__omni", "5400")):
        r = subprocess.run(["bash", str(SH_PATH), "--dry-run", *extra, "--", "--stand-ckpt", HEAD, "--stand-mask", "1",
                            "--deadband", "0.25"], env=env, capture_output=True, text=True, timeout=300)
        assert r.returncode == 0, r.stderr
        out = r.stdout
        for needle in ("OMNI_KIT_ACCEPT_EULA=yes", "PYTHONUNBUFFERED=1", "MemoryMax=24G", "MemorySwapMax=0",
                       "systemd-run --user --scope", "--headless", "--enable_cameras", "sh_record_video.py",
                       f"timeout --signal=TERM {tmo} ", f"--sequence {row}", "--deadband 0.25", "--stand-mask 1",
                       f"scenarios_sh_video_{row[3:-6]}.yaml"):
            assert needle in out, (needle, out)
        cmd = [ln for ln in out.splitlines() if ln.startswith("systemd-run")]
        assert len(cmd) == 1 and " -k " not in cmd[0] and "--kill-after" not in cmd[0]
        assert ("isaac busy: YES" in out) == busy and ("isaac busy: no" in out) == (not busy)
    assert not marker.exists()
    assert not (tmp_path / "out").exists()
    after = {p.name: p.read_bytes() for p in VIDEOS.iterdir()} if VIDEOS.is_dir() else {}
    assert after == before                                                          # a dry run writes nothing


def test_launcher_refuses_while_busy_before_anything_runs(tmp_path):
    env, marker = _fake_bin(tmp_path, ISAAC_LINE)
    gen_marker = tmp_path / "GEN_RAN"
    fake_py = tmp_path / "bin" / "python"
    fake_py.write_text(f"#!/bin/sh\ntouch {gen_marker}\nexit 1\n")
    fake_py.chmod(0o755)
    env["SH_PY"] = str(fake_py)
    r = subprocess.run(["bash", str(SH_PATH), "--smoke", "--", "--stand-ckpt", HEAD, "--deadband", "0.25"], env=env,
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 3, (r.returncode, r.stderr)
    assert "another Isaac process is running" in r.stderr and "4242" in r.stderr
    assert not marker.exists() and not gen_marker.exists() and not (tmp_path / "out").exists()


def test_launcher_usage_refusals(tmp_path):
    env, _marker = _fake_bin(tmp_path, "/usr/bin/bash idle")
    for args, needle in ((["--dry-run", "--", "--manifest", "x"], "set by this launcher"),
                         (["--dry-run", "--smoke", "--seed", "1"], "exclusive"),
                         (["--dry-run", "--total-s", "90"], "random plan only"),
                         (["--dry-run", "--seed", "x1"], "non-negative integer"),
                         (["--dry-run", "--prefix", "a b"], "--prefix")):
        r = subprocess.run(["bash", str(SH_PATH), *args], env=env, capture_output=True, text=True, timeout=60)
        assert r.returncode == 5 and needle in r.stderr, (args, r.returncode, r.stderr)
    r = subprocess.run(["bash", str(SH_PATH), "--dry-run"], env=dict(env, SH_VIDEO_TIMEOUT="0"), capture_output=True,
                       text=True, timeout=60)
    assert r.returncode == 5 and "SH_VIDEO_TIMEOUT" in r.stderr


def test_no_bytecode_next_to_the_scripts():
    for d in (SCRIPTS, PRED_SCRIPTS):
        assert not list(d.glob("__pycache__/sh_video*")) and not list(d.glob("__pycache__/sh_record*"))
