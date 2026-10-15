"""``experiments/tools/resolution.py`` -- the D00 eval-resolution record (plan decision 6).

Pure: the tool is loaded by path, every synthetic eval run dir is built in ``tmp_path`` by a
mini-harness that writes its aggregate through ``gait_eval.report.aggregate`` and its per-hold facing
through ``gait_eval.metrics.facing_metrics``, so a fixture is self-consistent by construction and the
tool's own replication checks run against it for real. No Isaac, no GPU, no network.

The one test that wants the recorded 1c-004 row (decision 6's null construction names it) skips when
that run dir is absent -- eval output is untracked, so a fresh checkout has none. Its property is
pinned unconditionally on synthetic columns by the test above it.

Rule version 2 (per-category eval plan 2026-09-21, D5/D9/D10): ``make_run(suite=True)`` builds a
``suite__omni`` stamp the way the harness writes one (the real manifest row, row-global segments,
``aggregate.suite`` from ``report.aggregate`` itself, ``complete.json`` last), and the tests below it
pin ``D00s``: the env-clustered draw, the expanded rule's six blocks and ten guards, the ratio-system
null with floored survival guards, the acceptance, the freshness rule and the recorded stamps. The
v1 tests keep ``RULE`` (hold-keyed) and the frozen v1 ``candidates.yaml`` fixture.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
TOOLS = (REPO / "parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task/experiments/tools")
RES_PY = TOOLS / "resolution.py"
GAIT_SCRIPTS = REPO / "parkour/parkour_tasks/parkour_tasks/crab_hex_forward_task/scripts"
RECORDED_1C = (REPO / "parkour/logs/rsl_rl/gait_eval/omni_phases/2026-09-16_1805_omni_formation"
               / "seed3_1c_004_A15pB/slow__omni/seed001/2026-09-18_08-27-40")


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


res = _load(RES_PY, "crab_hex_omni_resolution_under_test")


@pytest.fixture(scope="module")
def ge():
    """``gait_eval``'s ``report`` and ``metrics`` -- the harness the fixtures imitate."""
    if str(GAIT_SCRIPTS) not in sys.path:
        sys.path.insert(0, str(GAIT_SCRIPTS))
    from gait_eval import metrics as M
    from gait_eval import report as RPT
    return RPT, M


# --------------------------------------------------------------------------------- the mini harness
FIVE_CARRY = {"KRABBY_OMNI_FACING_OBS_CLIP": "0.6", "KRABBY_OMNI_SPEED_ELLIPSE": "0.35:0.25:0.25",
              "KRABBY_OMNI_ANG_VEL_MAX": "0.3", "KRABBY_OMNI_CLOCK_TURN_R": "1.2",
              "KRABBY_OMNI_YAW_RANGE": "-3.14159:3.14159"}
DT = 0.02


def _schedule(sid):
    sc = res._manifest_scenario(sid)
    return sc["hold_labels"], sc["hold_s"], sc["prologue_s"], sc["fall_exclusion_s"]


def _raw_episode(sid, rng, *, closes: float, fall: bool):
    """Per-step arrays for one episode, on the manifest's real hold boundaries.

    ``closes`` scales how much of each hold's initial heading error is closed by its end, so a
    fixture can make one head better at facing than another; ``fall`` truncates the episode inside
    the last turn hold, which is what makes the survival guards and the fall trim non-trivial.
    """
    labels, hold_s, prologue_s, _ = _schedule(sid)
    n_pro = int(round(prologue_s / DT))
    seg = [np.full(n_pro, -1, dtype=np.int16)]
    steady = [np.zeros(n_pro, dtype=bool)]
    err, wz, cmd = [np.zeros(n_pro)], [np.zeros(n_pro)], [np.zeros((n_pro, 3))]
    for hi, label in enumerate(labels):
        k = int(round(hold_s[label] / DT))
        seg.append(np.full(k, hi, dtype=np.int16))
        st = np.ones(k, dtype=bool)
        st[:int(round(1.0 / DT))] = False              # settle_s 1.0, as the manifest defaults say
        steady.append(st)
        turning = label.startswith("turn") or "turn" in label
        e0 = (1.4 + 0.2 * rng.standard_normal()) if turning else (0.05 + 0.02 * rng.random())
        t = np.arange(k) / max(1, k - 1)
        e = e0 * (1.0 - closes * t) + 0.01 * rng.standard_normal(k)
        err.append(e)
        wz.append(np.gradient(-e, DT) * 0.5 + 0.01 * rng.standard_normal(k))
        c = np.zeros((k, 3))
        c[:, 0] = 0.25 if label in ("fwd", "diag") else (0.22 if label not in ("stand",) and
                                                        not label.startswith("turn") else 0.0)
        c[:, 2] = 0.3 * np.sign(e0) if turning else 0.0
        cmd.append(c)
    out = {"segment_id": np.concatenate(seg), "steady_mask": np.concatenate(steady),
           "heading_err": np.concatenate(err).astype(np.float32),
           "root_ang_vel_b": np.stack([np.zeros(sum(len(x) for x in wz)),
                                       np.zeros(sum(len(x) for x in wz)),
                                       np.concatenate(wz)], axis=-1).astype(np.float32),
           "cmd_applied": np.concatenate(cmd).astype(np.float32)}
    total = out["segment_id"].size
    n_steps = total
    reason = "schedule_complete"
    if fall:
        n_steps = int(total * 0.8)
        reason = "fall"
    for k, v in out.items():
        out[k] = v[:n_steps]
    return out, n_steps, reason


def make_run(tmp: Path, sid, *, head, rng, n_eps=12, closes=0.6, ratio=0.6, tripod=0.5,
             fall_every=4, env_vars=None, with_raws=True, ge_mods=None, suite=False, **suite_kw):
    """One synthetic eval run dir: run_meta, metrics/episode_XX.json, scenario_metrics, raw/*.npz.

    The per-hold ``facing`` block is computed from the raws through the harness's own
    ``facing_metrics`` over the harness's own whole-hold window, so ``resolution.py``'s window
    replication check passes iff the tool cuts the same window the harness does -- which is the point
    of building the fixture this way rather than hand-writing numbers.

    ``suite=True`` builds a ``category_suite`` stamp instead (:func:`make_suite_run`).
    """
    if suite:
        return make_suite_run(tmp, head=head, rng=rng, closes=closes, ratio=ratio, tripod=tripod,
                              env_vars=env_vars, with_raws=with_raws, ge_mods=ge_mods, **suite_kw)
    RPT, M = ge_mods
    labels, hold_s, prologue_s, fall_excl = _schedule(sid)
    fall_trim = int(round(fall_excl / DT))
    run = tmp
    (run / "metrics").mkdir(parents=True, exist_ok=True)
    if with_raws:
        (run / "raw").mkdir(parents=True, exist_ok=True)
    episodes, n_steps_env = [], []
    for i in range(n_eps):
        raw, n_steps, reason = _raw_episode(sid, rng, closes=closes,
                                            fall=(fall_every and i % fall_every == 0))
        n_steps_env.append(n_steps)
        holds = {}
        for hi, label in enumerate(labels):
            widx = res.whole_hold_index(raw["segment_id"], hi, n_steps,
                                        ended_in_fall=reason in ("fall", "hard_fall"),
                                        fall_trim_steps=fall_trim)
            fac = M.facing_metrics(raw["heading_err"][widx], raw["root_ang_vel_b"][widx, 2], DT,
                                   wz_cmd_t=raw["cmd_applied"][widx, 2],
                                   min_initial=res.FACING_CLOSED_MIN_INITIAL_RAD)
            fac.pop("facing_closed")                 # a pre-2026-09-18 aggregate has neither key
            fac.pop("heading_error_initial_abs")
            fac.pop("n_uninformative")
            cmd0 = float(abs(raw["cmd_applied"][widx, 0]).mean()) if widx.size else 0.0
            speed = 0.25 if label in ("fwd", "diag") else (0.22 if cmd0 > 0.05 else 0.0)
            entry = {"label": label, "n_steps": int(widx.size), "facing": fac,
                     "tripod": {"tripod_score": float(tripod + 0.05 * rng.standard_normal()),
                                "low_confidence": False, "valid": True, "discard_reason": None}}
            if speed > 0.2 and widx.size:
                got = float(speed * (ratio + 0.05 * rng.standard_normal()))
                entry["tracking"] = {"vx": {"cmd_mean": speed, "actual_mean": got,
                                            "ratio": got / speed, "mean_abs": abs(got),
                                            "rms": abs(got), "signed_mean": got},
                                     "vy": {"actual_mean": 0.0, "cmd_mean": 0.0},
                                     "wz": {"actual_mean": 0.0, "cmd_mean": 0.0}}
            holds[label] = entry
        ep = {"env_index": i, "n_steps": int(n_steps), "duration_s": n_steps * DT,
              "termination_reason": reason, "completed_schedule": reason == "schedule_complete",
              "tripod_score": float(tripod), "tippy_tap_fraction": 0.1, "slip_ratio_mean": 0.1,
              "facing_error_rms": 0.5, "holds": holds, "discarded_windows": []}
        ep["tracking_ratio"] = float(np.mean([h["tracking"]["vx"]["ratio"] for h in holds.values()
                                              if h.get("tracking")]))
        episodes.append(ep)
        (run / "metrics" / f"episode_{i:02d}.json").write_text(json.dumps(ep, sort_keys=True))
        if with_raws:
            payload = dict(raw)
            payload["dt"] = np.asarray(DT, dtype=np.float64)
            np.savez_compressed(run / "raw" / f"episode_{i:02d}.npz", **payload)
    agg = RPT.aggregate(episodes)
    meta = {"scenario_id": sid, "hold_labels": list(labels), "dt": DT, "num_envs": n_eps,
            "n_steps_per_env": n_steps_env, "plant": "A15+B",
            "checkpoint": f"/logs/{head}/model_999.pt",
            "env_vars": dict(FIVE_CARRY if env_vars is None else env_vars)}
    (run / "run_meta.json").write_text(json.dumps(meta, sort_keys=True))
    (run / "scenario_metrics.json").write_text(json.dumps(
        {"scenario_id": sid, "aggregate": agg, "run_meta": meta,
         "episodes": [{k: e[k] for k in ("env_index", "tripod_score", "tracking_ratio",
                                         "tippy_tap_fraction", "slip_ratio_mean",
                                         "termination_reason", "n_steps", "facing_error_rms")}
                      for e in episodes]}, sort_keys=True))
    return run


SUITE = "suite__omni"


def _suite_scenario():
    """The committed ``suite__omni`` row, parsed by the harness's own loader."""
    S = res._gait_schedule()
    return next(sc for sc in S.load_manifest(res.rp.OMNI_MANIFEST)[0] if sc.id == SUITE)


def ckpt_of(run: Path) -> Path:
    """Where :func:`make_suite_run` put the checkpoint file the stamp records."""
    return run.parent / f"{run.name}_ckpt" / "model_999.pt"


def make_suite_run(run: Path, *, head, rng, ge_mods, n_env=8, closes=0.6, ratio=0.6, tripod=0.5,
                   fall_every=5, fall_offset=0, env_vars=None, with_raws=True, complete=True,
                   checkpoint=None, ckpt_bytes=None):
    """One synthetic ``suite__omni`` stamp, written the way the harness writes a suite row.

    The real manifest row (ten waves, 2 s settle, 12 s holds); ``n_env`` envs per wave, episode
    ``env_index = wave * n_env + env`` running ``[prologue, settle, its own hold]`` with row-global
    ``segment_id`` into the 11-label union. Every union label carries an entry (unvisited ones with
    ``n_steps: 0``), facing from the harness's ``facing_metrics`` on the harness's window (the new
    harness keeps ``facing_closed``), and the aggregate -- incl. ``aggregate.suite`` -- comes from
    ``report.aggregate(..., suite=suite_aggregate_arg(run_meta))`` itself. ``complete.json`` is
    written LAST, as ``write_run`` does. Env ``j`` falls halfway through its own hold in wave ``w``
    when ``(j + w + fall_offset) % fall_every == 0`` (heads with different offsets fall on different
    episodes, so their paired survival difference is not degenerate); otherwise env ``n_env - 1`` of
    wave 0 falls ON the wave's last step (which counts as survived).
    """
    RPT, M = ge_mods
    S = res._gait_schedule()
    sc = _suite_scenario()
    suite = sc.suite
    labels = suite.hold_labels()
    settle = labels[0]
    geo = res._manifest_scenario(SUITE)
    n_pro = int(round(geo["prologue_s"] / DT))
    n_set = int(round(geo["settle_hold_s"] / DT))
    fall_trim = int(round(geo["fall_exclusion_s"] / DT))
    cat_of, hand_of = suite.category_of_hold(), suite.hand_of_hold()
    hold_def = {h.label: h for h in suite.holds()}
    ckpt = Path(checkpoint) if checkpoint is not None else ckpt_of(run)
    if checkpoint is None:
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        ckpt.write_bytes(ckpt_bytes if ckpt_bytes is not None else f"weights of {head}".encode())
    sha = hashlib.sha256(ckpt.read_bytes()).hexdigest()
    (run / "metrics").mkdir(parents=True, exist_ok=True)
    if with_raws:
        (run / "raw").mkdir(parents=True, exist_ok=True)
    episodes, n_steps_env, raws = [], [], []
    for w, own in enumerate(labels[1:]):
        hd = hold_def[own]
        n_hold = int(round(float(geo["hold_s"][own]) / DT))
        turning = abs(hd.face) > 0.0
        for j in range(n_env):
            i = w * n_env + j
            seg = np.concatenate([np.full(n_pro, -1), np.zeros(n_set), np.full(n_hold, w + 1)])
            seg = seg.astype(np.int16)
            t = np.arange(n_hold) / max(1, n_hold - 1)
            e0 = (np.sign(hd.face) * (1.5708 + 0.05 * rng.standard_normal()) if turning
                  else 0.03 * rng.standard_normal())
            e_hold = e0 * (1.0 - closes * t) + 0.01 * rng.standard_normal(n_hold)
            err = np.concatenate([0.01 * rng.standard_normal(n_pro + n_set), e_hold])
            wz = np.gradient(-err, DT) * 0.5 + 0.01 * rng.standard_normal(err.size)
            cmd = np.zeros((err.size, 3))
            cmd[n_pro + n_set:, 0] = hd.speed
            if turning:
                cmd[n_pro + n_set:, 2] = 0.3 * np.sign(hd.face)
            raw = {"segment_id": seg, "steady_mask": np.ones(seg.size, dtype=bool),
                   "heading_err": err.astype(np.float32),
                   "root_ang_vel_b": np.stack([np.zeros(err.size), np.zeros(err.size), wz],
                                              axis=-1).astype(np.float32),
                   "cmd_applied": cmd.astype(np.float32)}
            total = seg.size
            n_steps, reason = total, "schedule_complete"
            if fall_every and (j + w + fall_offset) % fall_every == 0:
                n_steps, reason = n_pro + n_set + n_hold // 2, "fall"
            elif w == 0 and j == n_env - 1:
                reason = "fall"                          # ON the last step: survived
            raw = {k: v[:n_steps] for k, v in raw.items()}
            holds = {}
            for hi, label in enumerate(labels):
                widx = res.whole_hold_index(raw["segment_id"], hi, n_steps,
                                            ended_in_fall=reason in ("fall", "hard_fall"),
                                            fall_trim_steps=fall_trim)
                fac = M.facing_metrics(raw["heading_err"][widx], raw["root_ang_vel_b"][widx, 2], DT,
                                       wz_cmd_t=raw["cmd_applied"][widx, 2],
                                       min_initial=res.FACING_CLOSED_MIN_INITIAL_RAD)
                entry = {"label": label, "n_steps": int(widx.size), "facing": fac,
                         "tripod": {"tripod_score": None, "low_confidence": False, "valid": False,
                                    "discard_reason": "short"}}
                if widx.size and label in (own, settle):
                    entry["motion"] = {"speed_abs_mean_m_s": float(abs(0.02 + 0.01 * rng.standard_normal())),
                                       "pos_drift_m": 0.1, "path_len_m": 0.2, "yaw_drift_rad": 0.0,
                                       "foot_lift_count": 3, "contact_churn_per_s": 1.0,
                                       "n_steps": int(widx.size)}
                if widx.size and label == own:
                    # motion quality (gait-motion-quality step 1): the keys defined on the category,
                    # None elsewhere -- exactly the harness's N/A convention
                    entry["quality"] = {
                        key: (float(abs(1.0 + 0.2 * rng.standard_normal()))
                              if cat_of[own] in M.MOTION_QUALITY_APPLIES[key] else None)
                        for key in M.MOTION_QUALITY_KEYS}
                    # rigidity (clock-deprecation plan step 3), same N/A convention. Deterministic
                    # per (episode, key) -- it draws nothing from ``rng``, so every number the
                    # stream below feeds (and every pinned D00s value) is unchanged.
                    entry["rigidity"] = {
                        key: (1.0 + 0.01 * ((7 * i + 3 * k) % 17)
                              if cat_of[own] in M.RIGIDITY_APPLIES[key] else None)
                        for k, key in enumerate(M.RIGIDITY_EPISODE_KEYS)}
                    entry["tripod"] = {"tripod_score": float(tripod + 0.05 * rng.standard_normal()),
                                       "low_confidence": bool(rng.random() < 0.1), "valid": True,
                                       "discard_reason": None}
                    if hd.speed > 0.0:
                        got = float(hd.speed * (ratio + 0.05 * rng.standard_normal()))
                        entry["tracking"] = {"vx": {"cmd_mean": hd.speed, "actual_mean": got,
                                                    "ratio": got / hd.speed, "mean_abs": abs(got),
                                                    "rms": abs(got), "signed_mean": got},
                                             "vy": {"actual_mean": 0.0, "cmd_mean": 0.0},
                                             "wz": {"actual_mean": 0.0, "cmd_mean": 0.0}}
                        entry["direction"] = {
                            "direction_error_rad": float(abs(0.1 * rng.standard_normal())),
                            "cross_track_ratio": float(abs(0.1 * rng.standard_normal())),
                            "travel_speed_m_s": got}
                holds[label] = entry
            own_entry = holds[own]
            ep = {"env_index": i, "n_steps": int(n_steps), "duration_s": n_steps * DT,
                  "termination_reason": reason, "completed_schedule": reason == "schedule_complete",
                  "tripod_score": own_entry["tripod"]["tripod_score"], "tippy_tap_fraction": 0.1,
                  "slip_ratio_mean": 0.1, "facing_error_rms": 0.5, "holds": holds,
                  "discarded_windows": [],
                  "tracking_ratio": (own_entry["tracking"]["vx"]["ratio"]
                                     if own_entry.get("tracking") else None),
                  "wave": w, "hold": own, "category": cat_of[own], "hand": hand_of[own]}
            episodes.append(ep)
            n_steps_env.append(int(n_steps))
            raws.append(raw)
    meta = {"scenario_id": SUITE, "hold_labels": list(labels), "dt": DT, "num_envs": n_env,
            "n_steps_per_env": n_steps_env, "plant": "A15+B", "checkpoint": str(ckpt),
            "checkpoint_sha256": sha,
            "env_vars": dict(FIVE_CARRY if env_vars is None else env_vars)}
    meta.update(S.suite_run_meta(sc, dt=DT, num_envs=n_env,
                                 wave_seeds=[S.wave_seed(sc, k) for k in range(sc.waves)],
                                 trained_categories=list(res._ec().CATEGORIES)))
    agg = RPT.aggregate(episodes, suite=RPT.suite_aggregate_arg(meta))
    (run / "run_meta.json").write_text(json.dumps(meta, sort_keys=True))
    for ep in episodes:
        (run / "metrics" / f"episode_{ep['env_index']:02d}.json").write_text(json.dumps(ep, sort_keys=True))
    (run / "scenario_metrics.json").write_text(json.dumps(
        {"scenario_id": SUITE, "aggregate": agg, "run_meta": meta,
         "episodes": [{k: e[k] for k in ("env_index", "tripod_score", "tracking_ratio",
                                         "tippy_tap_fraction", "slip_ratio_mean",
                                         "termination_reason", "n_steps", "facing_error_rms")}
                      for e in episodes]}, sort_keys=True))
    if with_raws:
        for ep, raw in zip(episodes, raws):
            payload = dict(raw)
            payload["dt"] = np.asarray(DT, dtype=np.float64)
            np.savez_compressed(run / "raw" / f"episode_{ep['env_index']:02d}.npz", **payload)
    if complete:
        (run / "complete.json").write_text(json.dumps(
            {"episodes_total": len(episodes), "checkpoint_sha256": sha, "raw_written": with_raws},
            sort_keys=True))
    return run


RULE = {
    "blocks": {
        "F": {"level": 0.95, "metrics": ["facing_closed_by_hold.turn_l90",
                                         "facing_closed_by_hold.turn_r90",
                                         "facing_closed_by_hold.move_turn",
                                         "yaw_progress_by_hold.turn_l90",
                                         "yaw_progress_by_hold.turn_r90",
                                         "yaw_progress_by_hold.move_turn"]},
        "T": {"level": 0.95, "metrics": ["ratio_by_hold.fwd", "ratio_by_hold.lat_l",
                                         "ratio_by_hold.diag", "ratio_by_hold.back"]},
        "F_abs": {"level": 0.95, "metrics": ["facing_closed_by_hold.turn_l90",
                                            "net_yaw_by_hold.turn_l90"]},
        "T_abs": {"level": 0.95, "metrics": ["achieved_speed_by_hold.fwd",
                                             "achieved_speed_by_hold.back"]},
    },
    "guards": {"level": 0.99, "metrics": ["completion", "tripod_walking_dir",
                                          "turn_survival_l90", "turn_survival_r90"]},
    "watch": ["ratio_by_hold.move_turn", "ratio_by_hold.lat_r", "track_ratio_tail"],
    "charter_floors": {"completion": 0.10, "tripod_walking_dir": 0.056},
    "bootstrap": {"B": 200, "paired_on": "env_index", "rng_seed": 7},
}


def _cands(*rows, rule=None):
    return {"resolution_inputs": list(rows), "decision_rule": dict(RULE if rule is None else rule)}


def _entry(run_dir, **kw):
    return {"run_dir": str(run_dir), **kw}


@pytest.fixture
def three_heads(tmp_path, ge):
    """1c-004 / 2b / 2a-reeval on ``slow__omni``, the D00 shape, with 2b worse at facing."""
    rng = np.random.default_rng(1)
    a = make_run(tmp_path / "a", "slow__omni", head="1c-004", rng=rng, closes=0.7, ge_mods=ge)
    b = make_run(tmp_path / "b", "slow__omni", head="2b", rng=rng, closes=0.4, ratio=0.5,
                 tripod=0.4, ge_mods=ge)
    c = make_run(tmp_path / "c", "slow__omni", head="2a-reeval", rng=rng, closes=0.6, ge_mods=ge)
    return a, b, c


# ------------------------------------------------------------------------------------- pure maths
def test_half_width_is_the_centred_spread_and_ignores_the_offset():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(20000)
    base = res.half_width(x, 0.975)
    assert base == pytest.approx(1.96, abs=0.06)
    assert res.half_width(x + 17.0, 0.975) == pytest.approx(base, abs=1e-12)
    assert res.half_width(x, 0.99) == pytest.approx(2.326, abs=0.09)
    assert res.half_width([1.0], 0.975) is None                     # below MIN_BOOTSTRAP_N
    assert res.half_width([np.nan, np.nan, 1.0, 2.0], 0.975) is not None


def test_quantile_p_is_two_sided_95_and_one_sided_99():
    assert res.quantile_p(0.95, one_sided=False) == pytest.approx(0.975)
    assert res.quantile_p(0.99, one_sided=True) == pytest.approx(0.99)


def test_classify_is_decision_5s_rule_including_the_boundary():
    assert res.classify(1.8, 0.9) == "BETTER"
    assert res.classify(-1.8, 0.9) == "WORSE"
    assert res.classify(0.2, 0.9) == "TIE"
    assert res.classify(0.9, 0.9) == "TIE"          # boundary is a TIE on both sides
    assert res.classify(-0.9, 0.9) == "TIE"
    assert res.classify(1.8, 0.9, one_sided=True) == "TIE"   # a guard never says BETTER
    assert res.classify(-1.8, 0.9, one_sided=True) == "WORSE"
    assert res.classify(None, 0.9) == "MISSING"
    assert res.classify(1.0, None) == "MISSING"
    assert res.classify(float("nan"), 0.9) == "MISSING"


def test_single_and_pooled_R_are_decision_6s_arithmetic():
    assert res.single_block_R(0.5, 0.7) == pytest.approx(0.7)
    assert res.single_block_R(0.5, None) == pytest.approx(0.5)
    assert res.single_block_R(None, None) is None
    assert res.single_guard_R(0.08, 0.03) == pytest.approx(0.08)
    assert res.pooled_block_R(0.6, 0.2) == pytest.approx(0.6 / math.sqrt(2))
    assert res.pooled_block_R(0.6, 0.5) == pytest.approx(0.5)
    assert res.pooled_guard_R(0.08, 0.03) == pytest.approx(math.sqrt((0.08 / math.sqrt(2)) ** 2
                                                                     + 0.03 ** 2))
    assert res.pooled_guard_R(None, None) is None


def test_block_seed_term_cannot_cancel_across_alternating_signs():
    r = {f"m{i}": 1.0 for i in range(6)}
    alternating = {f"m{i}": (1.0 if i % 2 else -1.0) for i in range(6)}
    term = res.block_seed_term(alternating, r)
    assert term == pytest.approx(1.0 / math.sqrt(6), abs=1e-12)   # rms 1 / sqrt(k)
    assert term > 0.0
    # and it is never below the signed pair statistic
    aligned = {f"m{i}": 1.0 for i in range(6)}
    z_pair = abs(sum(aligned.values()) / 6)
    assert res.block_seed_term(aligned, r) >= z_pair - 1e-12
    assert res.block_seed_term({"a": None}, {"a": 1.0}) is None


def test_block_mean_drops_a_missing_member_not_the_whole_draw():
    std = np.array([[1.0, 2.0, np.nan], [3.0, np.nan, np.nan]])
    got = res.block_mean(std)
    assert got[0] == pytest.approx(2.0)     # both members
    assert got[1] == pytest.approx(2.0)     # one member left the mean
    assert math.isnan(got[2])               # every member missing -> the draw drops out


def test_required_carry_keys_are_the_five_without_a_default():
    req = res.required_carry_keys()
    # 2026-10-15: the body-only envelope keys (no default, refused in a world eval) are never required.
    assert set(req) == set(res.rp.EVAL_CARRY) - set(res.rp.EVAL_CARRY_DEFAULTS) - set(res.rp.EVAL_CARRY_BODY_ONLY)
    assert len(req) == 5
    assert "KRABBY_OMNI_FACING_OBS_ENCODING" not in req    # defaulted -> not required
    assert "KRABBY_OMNI_ANG_VEL_CLIP" not in req


# ----------------------------------------------------------------------------- the provenance rule
def test_a_defaulted_carry_key_is_taken_at_its_default_and_recorded(tmp_path, ge):
    run = make_run(tmp_path / "r", "slow__omni", head="h", rng=np.random.default_rng(2), ge_mods=ge)
    row = res.load_row(_entry(run), ring_proxy=False, rule=RULE)
    assert row.provenance_ok and row.accepted and row.reason == "ok"
    assert set(row.env_vars_defaulted) == set(res.rp.EVAL_CARRY_DEFAULTS)


def test_a_missing_non_defaulted_carry_key_is_refused_and_a_waiver_makes_it_informational(tmp_path, ge):
    thin = {k: v for k, v in FIVE_CARRY.items() if k != "KRABBY_OMNI_ANG_VEL_MAX"}
    run = make_run(tmp_path / "r", "slow__omni", head="h", rng=np.random.default_rng(3),
                   env_vars=thin, ge_mods=ge)
    refused = res.load_row(_entry(run), ring_proxy=False, rule=RULE)
    assert refused.provenance_ok is False and refused.accepted is False
    assert "refused" in refused.reason and "KRABBY_OMNI_ANG_VEL_MAX" in refused.reason
    waived = res.load_row(_entry(run, provenance_ok=True), ring_proxy=False, rule=RULE)
    assert waived.provenance_ok is False, "the measured verdict does not change with a waiver"
    assert waived.accepted is False, "a waived row is informational, never a candidate-R source"
    assert "informational" in waived.reason and "waived" in waived.reason


def test_role_informational_is_reported_but_never_pooled(tmp_path, ge):
    run = make_run(tmp_path / "r", "slow__omni", head="h", rng=np.random.default_rng(4), ge_mods=ge)
    row = res.load_row(_entry(run, role="informational"), ring_proxy=False, rule=RULE)
    assert row.provenance_ok is True and row.accepted is False
    assert "informational by role" in row.reason


# ----------------------------------------------------------------------------------- cross-head pairing
def test_cross_head_pairs_are_the_control_against_each_other_head(three_heads, tmp_path):
    a, b, c = three_heads
    out = res.resolve(_cands(_entry(a, head="1c-004"), _entry(b, head="2b"),
                             _entry(c, head="2a-reeval")))
    names = {p["name"] for p in out["pairs"]}
    assert names == {"1c-004 vs 2b @slow__omni", "1c-004 vs 2a-reeval @slow__omni"}, names
    assert all(p["kind"] == "cross_head" for p in out["pairs"])
    assert out["by_row"]["slow__omni"]["control_head"] == "1c-004"
    # every scored metric names the pairs its R came from, and R is the MAX over them
    for mid, m in out["metrics"].items():
        assert m["pairs_used"], mid
        assert m["R_eval"] == pytest.approx(max(v for v in m["per_pair"].values() if v is not None))
    # the control is the FIRST accepted entry listed; reordering moves it
    flipped = res.resolve(_cands(_entry(b, head="2b"), _entry(a, head="1c-004"),
                                 _entry(c, head="2a-reeval")))
    assert flipped["by_row"]["slow__omni"]["control_head"] == "2b"
    forced = res.resolve(_cands(_entry(b, head="2b"), _entry(a, head="1c-004")),
                         control_head="1c-004")
    assert forced["by_row"]["slow__omni"]["control_head"] == "1c-004"


def test_an_informational_row_forms_no_pair(three_heads):
    a, b, c = three_heads
    out = res.resolve(_cands(_entry(a, head="1c-004"), _entry(b, head="2b"),
                             _entry(c, head="corrected", role="informational")))
    assert {p["name"] for p in out["pairs"]} == {"1c-004 vs 2b @slow__omni"}
    assert [i["head"] for i in out["inputs"] if not i["accepted"]] == ["corrected"]


def test_two_rows_of_one_head_are_the_process_floor_not_a_candidate_pair(tmp_path, ge):
    rng = np.random.default_rng(5)
    a = make_run(tmp_path / "a", "slow__omni", head="2b", rng=rng, ge_mods=ge)
    a2 = make_run(tmp_path / "a2", "slow__omni", head="2b", rng=rng, ge_mods=ge)
    b = make_run(tmp_path / "b", "slow__omni", head="1c-004", rng=rng, closes=0.7, ge_mods=ge)
    out = res.resolve(_cands(_entry(b, head="1c-004"), _entry(a, head="2b"),
                             _entry(a2, head="2b", process_floor_pair=True)))
    kinds = {p["name"]: p["kind"] for p in out["pairs"]}
    assert kinds["2b vs 2b @slow__omni"] == "process_floor"
    assert kinds["1c-004 vs 2b @slow__omni"] == "cross_head"
    assert out["process_floor"], "the same-head pair must be reported as the process-noise floor"
    assert "completion" in out["process_floor"]
    # and it never contributes to the candidate R
    for m in out["metrics"].values():
        assert "2b vs 2b @slow__omni" not in m["per_pair"]


def test_a_flagged_floor_row_with_its_own_label_is_a_floor_pair_not_a_third_cross_arm(tmp_path, ge,
                                                                                       three_heads):
    """``process_floor_pair`` is a FLAG, not a coincidence of labels.

    The campaign lists the 2b re-eval as ``head: 2b_floor`` (so the record reads in campaign names),
    and decision 6 says the same-head pair is "reported separately as the process-noise floor, NEVER
    as the candidate R". Grouping by head label alone made that row a third cross-head arm whose
    half-widths were folded into the candidate R by the max -- silently, since `process_floor` then
    stayed {} and the note still asked for the row that had just been added.
    """
    a, b, c = three_heads
    floor_run = make_run(tmp_path / "bf", "slow__omni", head="2b", rng=np.random.default_rng(31),
                         closes=0.4, ratio=0.5, tripod=0.4, ge_mods=ge)
    base = res.resolve(_cands(_entry(a, head="1c-004"), _entry(b, head="2b"),
                              _entry(c, head="2a-reeval")))
    out = res.resolve(_cands(_entry(a, head="1c-004"), _entry(b, head="2b"),
                             _entry(c, head="2a-reeval"),
                             _entry(floor_run, head="2b_floor", floor_of="2b",
                                    role="process_floor", process_floor_pair=True)))
    kinds = {p["name"]: p["kind"] for p in out["pairs"]}
    assert kinds["2b vs 2b_floor @slow__omni"] == "process_floor"
    assert set(kinds) - {"2b vs 2b_floor @slow__omni"} == {"1c-004 vs 2b @slow__omni",
                                                           "1c-004 vs 2a-reeval @slow__omni"}
    assert out["process_floor"], "the flagged row is the process floor"
    for mid, m in base["metrics"].items():                       # and the candidate R never moved
        assert out["metrics"][mid]["R_eval"] == m["R_eval"], mid
        assert all("2b_floor" not in name for name in m["per_pair"]), mid
    for block, bl in base["blocks"].items():
        assert out["blocks"][block]["R_eval"] == bl["R_eval"], block


def test_a_flagged_floor_row_that_matches_nothing_is_refused(tmp_path, ge, three_heads):
    """Better a refusal than a silent extra arm: a flag that found no target is a spec mistake."""
    a, b, _ = three_heads
    orphan = make_run(tmp_path / "orphan", "slow__omni", head="nobody",
                      rng=np.random.default_rng(32), ge_mods=ge)
    with pytest.raises(res.ResolutionError) as exc:
        res.resolve(_cands(_entry(a, head="1c-004"), _entry(b, head="2b"),
                           _entry(orphan, head="ghost", floor_of="not-a-head",
                                  process_floor_pair=True)))
    assert "process_floor_pair" in str(exc.value) and "floor_of" in str(exc.value)


def test_the_block_half_width_is_standardised_by_the_MERGED_metric_R(three_heads):
    """Decision 5: "one standardiser everywhere: R_i = R_eval,i" -- and R_eval,i is the PUBLISHED
    half-width (the max over the pairs), which is what ``decide()`` divides by.

    So ``R_F^eval``, "the paired-bootstrap 95 % half-width of z_F itself", has to be bootstrapped in
    those same units. Standardising each pair by its own per-metric spread measures a slightly
    different statistic (5-7 % wide on the recorded inputs) and is only conservative by accident.
    """
    a, b, c = three_heads
    rows = [res.load_row(_entry(run, head=head), ring_proxy=False, rule=RULE)
            for run, head in ((a, "1c-004"), (b, "2b"), (c, "2a-reeval"))]
    specs, _ = res.specs_from_rule(RULE, rows[0].hold_labels)
    pairs = [res.bootstrap_pair(rows[0], other, B=200, rng_seed=7, specs=specs)
             for other in rows[1:]]
    metrics, blocks, _ = res._merge_pairs(pairs)
    for block, slot in blocks.items():
        recomputed = {}
        for pr in pairs:
            use = slot["members_used"][pr.name]
            z = res.block_mean(np.stack([pr.deltas[m] / float(metrics[m]["R_eval"]) for m in use],
                                        axis=0))
            want = res.half_width(z, res.quantile_p(0.95, one_sided=False))
            assert slot["per_pair"][pr.name] == pytest.approx(want, abs=1e-12), (block, pr.name)
            recomputed[pr.name] = want
        assert slot["R_eval"] == pytest.approx(max(recomputed.values()), abs=1e-12)
    # and it really differs from the one-pass, pair-local version on at least one pair
    local = [pr.blocks[b_]["R_eval"] for pr in pairs for b_ in pr.blocks]
    published = [blocks[b_]["per_pair"][pr.name] for pr in pairs for b_ in pr.blocks]
    assert any(abs(x - y) > 1e-12 for x, y in zip(local, published)), \
        "the merged R must differ from at least one pair's own, or this pins nothing"


def test_process_floor_is_empty_without_a_same_head_pair(three_heads):
    a, b, c = three_heads
    out = res.resolve(_cands(_entry(a, head="1c-004"), _entry(b, head="2b"), _entry(c, head="2a")))
    assert out["process_floor"] == {}
    assert any("process_floor empty" in n for n in out["notes"])


# ------------------------------------------------------------------------------- facing_closed sourcing
def test_the_rescore_reproduces_every_recorded_facing_key(tmp_path, ge):
    """The window replication is exact, so ``facing_closed`` off the raws is trustworthy."""
    run = make_run(tmp_path / "r", "slow__omni", head="h", rng=np.random.default_rng(6), ge_mods=ge)
    row = res.load_row(_entry(run), ring_proxy=False, rule=RULE)
    assert row.mask_deviation == 0.0
    assert row.facing_closed_source == "metric"
    closed, source = res.facing_closed_by_hold_from_raws(run)
    assert source == "metric"
    assert closed["turn_l90"] is not None and closed["stand"] is None


def test_a_wrong_window_is_refused_not_absorbed(tmp_path, ge):
    run = make_run(tmp_path / "r", "slow__omni", head="h", rng=np.random.default_rng(7), ge_mods=ge)
    ep_path = run / "metrics" / "episode_00.json"
    ep = json.loads(ep_path.read_text())
    ep["holds"]["turn_l90"]["facing"]["heading_error_final_abs"] += 0.25
    ep_path.write_text(json.dumps(ep, sort_keys=True))
    with pytest.raises(res.ResolutionError, match="window replication"):
        res.load_row(_entry(run), ring_proxy=False, rule=RULE)


def test_a_desynchronised_aggregate_is_refused(tmp_path, ge):
    run = make_run(tmp_path / "r", "slow__omni", head="h", rng=np.random.default_rng(8), ge_mods=ge)
    sm_path = run / "scenario_metrics.json"
    sm = json.loads(sm_path.read_text())
    sm["aggregate"]["schedule_completion_rate"] = 0.123
    sm_path.write_text(json.dumps(sm, sort_keys=True))
    with pytest.raises(res.ResolutionError, match="does not reproduce summarize"):
        res.load_row(_entry(run), ring_proxy=False, rule=RULE)


def test_ring_proxy_scores_the_ring_row_off_heading_error_max_abs(tmp_path, ge, three_heads):
    slow_a, slow_b, _ = three_heads
    rng = np.random.default_rng(9)
    a = make_run(tmp_path / "ra", "ring__omni", head="1c-004", rng=rng, closes=0.7, ge_mods=ge)
    b = make_run(tmp_path / "rb", "ring__omni", head="2b", rng=rng, closes=0.4, ge_mods=ge)
    rows = (_entry(slow_a, head="1c-004"), _entry(slow_b, head="2b"),
            _entry(a, head="1c-004"), _entry(b, head="2b"))
    default = res.resolve(_cands(*rows))
    assert {i["facing_closed_source"] for i in default["inputs"]} == {"metric"}
    proxied = res.resolve(_cands(*rows), ring_proxy=True)
    by_row = {(i["head"], i["row"]): i["facing_closed_source"] for i in proxied["inputs"]}
    assert by_row[("1c-004", "ring__omni")] == "proxy"
    assert by_row[("1c-004", "slow__omni")] == "metric", "only the ring rows take the proxy"
    row = res.load_row(_entry(a), ring_proxy=True, rule=RULE)
    assert row.facing_closed_source == "proxy"
    # the proxy is exactly the plan's documented formula, per episode
    ep = json.loads((a / "metrics" / "episode_01.json").read_text())
    fac = ep["holds"]["turn_l90"]["facing"]
    want = 1.0 - fac["heading_error_final_abs"] / max(fac["heading_error_max_abs"],
                                                      res.FACING_CLOSED_MIN_INITIAL_RAD)
    assert row.cols["closed:turn_l90"][1] == pytest.approx(max(-1.0, min(1.0, want)))
    assert proxied["by_row"]["ring__omni"]["metrics"][
        "facing_closed_by_hold.turn_l90"]["source"] == "proxy"
    assert default["by_row"]["ring__omni"]["metrics"][
        "facing_closed_by_hold.turn_l90"]["source"] == "raw"


def test_a_row_with_no_raws_falls_back_to_the_proxy(tmp_path, ge):
    run = make_run(tmp_path / "r", "slow__omni", head="h", rng=np.random.default_rng(10),
                   with_raws=False, ge_mods=ge)
    row = res.load_row(_entry(run), ring_proxy=False, rule=RULE)
    assert row.facing_closed_source == "proxy"
    assert row.mask_deviation is None


def test_a_proxy_row_is_never_paired_against_a_rescored_row(tmp_path, ge):
    rng = np.random.default_rng(11)
    a = make_run(tmp_path / "a", "slow__omni", head="1c-004", rng=rng, ge_mods=ge)
    b = make_run(tmp_path / "b", "slow__omni", head="2b", rng=rng, with_raws=False, ge_mods=ge)
    with pytest.raises(res.ResolutionError, match="facing_closed sources differ"):
        res.resolve(_cands(_entry(a, head="1c-004"), _entry(b, head="2b")))


# -------------------------------------------------------------------------------- blocks and record
def test_the_record_carries_the_schema_v1_interface(three_heads):
    a, b, c = three_heads
    out = res.resolve(_cands(_entry(a, head="1c-004"), _entry(b, head="2b"),
                             _entry(c, head="2a-reeval")))
    assert out["schema"] == 1
    assert out["B"] == RULE["bootstrap"]["B"], "the yaml's bootstrap block is honoured"
    assert out["rng_seed"] == RULE["bootstrap"]["rng_seed"]
    assert out["decision_rule"] == RULE, "frozen verbatim"
    assert out["seed_terms"] == {}, "D00 is eval-only; D00b/D00c are the driver's"
    for key in ("created", "inputs", "pairs", "metrics", "blocks", "process_floor", "notes"):
        assert key in out, key
    for i in out["inputs"]:
        assert set(("run_dir", "row", "head", "provenance_ok", "accepted", "reason",
                    "env_vars_defaulted")) <= set(i)
        assert Path(i["run_dir"]).is_absolute()
    for p in out["pairs"]:
        assert set(("a", "b", "row", "n")) <= set(p)
    for mid, m in out["metrics"].items():
        assert m["block"] in ("F", "T", "guard", "F_abs", "T_abs"), (mid, m["block"])
        assert m["level"] in (0.95, 0.99)
        assert m["source"] in ("raw", "proxy", "json")
        assert isinstance(m["n"], int)
        assert (m["hold"] is None) or isinstance(m["hold"], str)
    for block, bl in out["blocks"].items():
        assert block in ("F", "T", "F_abs", "T_abs")
        assert bl["members"] and set(bl["members"]) <= set(out["metrics"])
    # guards are one-sided at 99 %, block members two-sided at 95 %
    guards = {mid for mid, m in out["metrics"].items() if m["block"] == "guard"}
    assert guards == set(RULE["guards"]["metrics"])
    assert all(out["metrics"][g]["one_sided"] and out["metrics"][g]["level"] == 0.99 for g in guards)
    assert all(not out["metrics"][m]["one_sided"] for m in out["blocks"]["F"]["members"])


def test_watch_metrics_are_out_of_the_scored_dict_and_the_right_handed_holds_are_borrowed(three_heads):
    a, b, c = three_heads
    out = res.resolve(_cands(_entry(a, head="1c-004"), _entry(b, head="2b"),
                             _entry(c, head="2a-reeval")))
    assert "ratio_by_hold.move_turn" not in out["metrics"], "WATCH is never a scored member"
    assert "ratio_by_hold.move_turn" in out["watch_metrics"]
    assert "track_ratio_tail" not in out["metrics"] and "track_ratio_tail" not in out["watch_metrics"]
    assert any("track_ratio_tail" in n for n in out["notes"])
    borrowed = out["watch_metrics"]["ratio_by_hold.lat_r"]
    assert borrowed["borrowed_from"] == "ratio_by_hold.lat_l"
    assert borrowed["R_eval"] == pytest.approx(out["metrics"]["ratio_by_hold.lat_l"]["R_eval"])
    assert out["watch_metrics"]["ratio_by_hold.move_turn_r"]["borrowed_from"] == \
        "ratio_by_hold.move_turn"
    assert any("borrow" in n for n in out["notes"])
    assert out["aliases"]["speed_ratio_by_hold.fwd"] == "ratio_by_hold.fwd"


def test_the_block_half_width_is_bootstrapped_on_z_not_propagated(three_heads):
    a, b, _ = three_heads
    out = res.resolve(_cands(_entry(a, head="1c-004"), _entry(b, head="2b")))
    R_F = out["blocks"]["F"]["R_eval"]
    members = out["blocks"]["F"]["members"]
    # each standardised member has half-width ~1 by construction, so a propagated sum would be ~1
    # (perfect correlation) and an independence model 1/sqrt(k); a directly bootstrapped z lands
    # strictly between, never above 1.
    assert 1.0 / math.sqrt(len(members)) * 0.8 < R_F < 1.05
    mde = out["blocks"]["F"]["mde"]
    for mid in members:
        assert mde[mid] == pytest.approx(R_F * out["metrics"][mid]["R_eval"])


def test_a_metric_with_too_few_usable_episodes_reports_no_half_width(tmp_path, ge):
    """A spread estimated off one or two episodes is meaningless; MISSING is the loud answer."""
    rng = np.random.default_rng(12)
    rule = json.loads(json.dumps(RULE))
    rule["watch"] = ["facing_by_hold.turn_l90.settle"]
    a = make_run(tmp_path / "a", "slow__omni", head="x", rng=rng, closes=0.0, ge_mods=ge)
    b = make_run(tmp_path / "b", "slow__omni", head="y", rng=rng, closes=0.0, ge_mods=ge)
    out = res.resolve(_cands(_entry(a, head="x"), _entry(b, head="y"), rule=rule))
    # closes=0.0 means the turn holds' heading error never comes inside the settle tolerance, so
    # settle_time_s is None on every episode of those holds.
    entry = out["watch_metrics"]["facing_by_hold.turn_l90.settle"]
    assert entry["n_valid"] < res.MIN_R_EVAL_N, entry
    assert entry["R_eval"] is None
    assert any("MISSING" in n and "settle" in n for n in out["notes"])
    # a hold that DOES settle keeps its half-width
    assert out["watch_metrics"]["facing_by_hold.turn_l90.settle"]["block"] == "watch"


def test_every_registered_id_reproduces_summarizes_own_number(tmp_path, ge):
    """The reducer check is the contract: a per-episode reconstruction that does not reproduce the
    recorded aggregate is resampling a different estimator than the decision rule reads."""
    rule = json.loads(json.dumps(RULE))
    rule["guards"]["metrics"] = ["completion", "tripod_walking_dir", "turn_survival_l90",
                                 "turn_survival_r90", "tripod", "tracking", "facing_error_rms",
                                 "yaw_progress_turn_min"]
    rule["watch"] = ["tripod_by_hold.move_turn", "survival_by_hold.back",
                     "facing_by_hold.turn_r90.settle", "tripod_walking", "track_ratio_tail"]
    run = make_run(tmp_path / "r", "slow__omni", head="h", rng=np.random.default_rng(16), ge_mods=ge)
    row = res.load_row(_entry(run), ring_proxy=False, rule=rule)
    specs, unresolved = res.specs_from_rule(rule, row.hold_labels)
    assert set(unresolved) == {"tripod_walking", "track_ratio_tail"}, unresolved
    assert row.reducer_check, "the check must actually have run"
    for mid, dev in row.reducer_check.items():
        assert dev == pytest.approx(0.0, abs=1e-9), (mid, dev)
    covered = {s.mid for s in specs} - {"facing_closed_by_hold.turn_l90",
                                        "facing_closed_by_hold.turn_r90",
                                        "facing_closed_by_hold.move_turn"}
    assert covered <= set(row.reducer_check), covered - set(row.reducer_check)
    assert res.why_unresolved("tripod_walking", row.hold_labels).startswith("the harness decides")
    assert "training-log" in res.why_unresolved("track_ratio_tail", row.hold_labels)


def test_the_screen_row_is_required(tmp_path, ge):
    rng = np.random.default_rng(13)
    a = make_run(tmp_path / "a", "ring__omni", head="p", rng=rng, ge_mods=ge)
    b = make_run(tmp_path / "b", "ring__omni", head="q", rng=rng, ge_mods=ge)
    with pytest.raises(res.ResolutionError, match="screen row"):
        res.resolve(_cands(_entry(a, head="p"), _entry(b, head="q")))


def test_charter_floors_are_cited_and_superseded_never_applied(three_heads):
    a, b, c = three_heads
    out = res.resolve(_cands(_entry(a, head="1c-004"), _entry(b, head="2b"), _entry(c, head="2a")))
    assert out["charter_floors"] == RULE["charter_floors"]
    for mid, floor in RULE["charter_floors"].items():
        R = out["metrics"][mid]["R_eval"]
        assert R != pytest.approx(floor), "the floor must not become the R"
        assert any(mid in n and "charter floor cited" in n for n in out["notes"])


# ------------------------------------------------------------------------------------ determinism
def test_the_record_is_deterministic_under_its_rng_seed(three_heads, tmp_path):
    a, b, c = three_heads
    cands = _cands(_entry(a, head="1c-004"), _entry(b, head="2b"), _entry(c, head="2a-reeval"))
    one, two = res.resolve(dict(cands)), res.resolve(dict(cands))
    one.pop("created"), two.pop("created")
    assert json.dumps(one, sort_keys=True) == json.dumps(two, sort_keys=True)
    other = res.resolve(dict(cands), rng_seed=RULE["bootstrap"]["rng_seed"] + 1)
    assert other["rng_seed"] != one["rng_seed"]
    assert any(other["metrics"][m]["R_eval"] != one["metrics"][m]["R_eval"] for m in one["metrics"])


def test_adding_a_later_input_leaves_every_existing_pair_bit_identical(tmp_path, ge, three_heads):
    """The process-noise floor row lands weeks after D00; it must not move a recorded R.

    Each pair's bootstrap is seeded from ``(rng_seed, pair name)``, not drawn from one shared stream,
    so a new input adds a pair without re-rolling the others. Decision 6 says R is never re-fitted --
    a shared stream would silently re-fit every one of them the next time the tool ran.
    """
    a, b, c = three_heads
    before = res.resolve(_cands(_entry(a, head="1c-004"), _entry(b, head="2b"),
                                _entry(c, head="2a-reeval")))
    a2 = make_run(tmp_path / "a2", "slow__omni", head="1c-004", rng=np.random.default_rng(99),
                  closes=0.7, ge_mods=ge)
    after = res.resolve(_cands(_entry(a, head="1c-004"), _entry(b, head="2b"),
                               _entry(c, head="2a-reeval"),
                               _entry(a2, head="1c-004", process_floor_pair=True)))
    assert after["process_floor"], "the new row must produce a process floor"
    for mid, m in before["metrics"].items():
        assert after["metrics"][mid]["R_eval"] == m["R_eval"], mid
    for block, bl in before["blocks"].items():
        assert after["blocks"][block]["R_eval"] == bl["R_eval"], block


def test_an_eval_paired_against_itself_has_a_process_floor_of_exactly_zero(tmp_path, ge, three_heads):
    """The paired construction cancels identical inputs exactly, so any non-zero floor from the real
    same-head re-eval is genuine process noise rather than bootstrap slack."""
    a, b, _ = three_heads
    out = res.resolve(_cands(_entry(a, head="1c-004"), _entry(b, head="2b"),
                             _entry(a, head="1c-004", process_floor_pair=True)))
    assert out["process_floor"]
    assert set(out["process_floor"].values()) == {0.0}


def test_write_resolution_round_trips_and_is_sorted(three_heads, tmp_path):
    a, b, c = three_heads
    out = res.resolve(_cands(_entry(a, head="1c-004"), _entry(b, head="2b"), _entry(c, head="2a")))
    path = res.write_resolution(out, tmp_path / "camp" / "resolution.json")
    assert path.exists() and not list(path.parent.glob("*.tmp"))
    assert json.loads(path.read_text()) == json.loads(json.dumps(out))


def test_a_missing_run_dir_is_a_refusal_not_a_traceback(tmp_path):
    """The driver reports "resolution input <dir> is not on this machine (resolution.py refuses it)",
    so a missing dir must come back as a ResolutionError rather than a FileNotFoundError."""
    with pytest.raises(res.ResolutionError, match="unusable"):
        res.load_row(_entry(tmp_path / "nope"), ring_proxy=False, rule=RULE)
    half = tmp_path / "half"
    (half / "metrics").mkdir(parents=True)
    (half / "run_meta.json").write_text("{}")
    with pytest.raises(res.ResolutionError, match="scenario_metrics.json is missing"):
        res.load_row(_entry(half), ring_proxy=False, rule=RULE)


def test_the_resolution_line_quotes_the_bootstrap_actually_used(three_heads):
    a, b, c = three_heads
    cands = _cands(_entry(a, head="1c-004"), _entry(b, head="2b"), _entry(c, head="2a"))
    out = res.resolve(dict(cands), bootstrap=311)
    assert out["B"] == 311
    assert "B=311" in out["resolution_line"]
    assert "guards @0.99 one-sided" in out["resolution_line"]
    yaml_default = res.resolve(dict(cands))
    assert yaml_default["B"] == RULE["bootstrap"]["B"]
    assert f"B={RULE['bootstrap']['B']}" in yaml_default["resolution_line"]


def test_main_writes_the_default_paths(three_heads, tmp_path):
    a, b, c = three_heads
    camp = tmp_path / "camp"
    camp.mkdir()
    import yaml
    (camp / "candidates.yaml").write_text(yaml.safe_dump(
        _cands(_entry(a, head="1c-004"), _entry(b, head="2b"), _entry(c, head="2a"))))
    assert res.main(["--campaign-dir", str(camp)]) == 0
    assert (camp / "resolution.json").exists()
    assert res.main(["--campaign-dir", str(tmp_path / "missing")]) == 1


# ---------------------------------------------------------------------------- the curriculum hook
def test_curriculum_runs_rederives_the_right_handed_holds(tmp_path, ge, three_heads):
    a, b, c = three_heads
    rng = np.random.default_rng(14)
    cur_a = make_run(tmp_path / "ca", "curriculum__omni", head="X01", rng=rng, n_eps=10, ge_mods=ge)
    cur_b = make_run(tmp_path / "cb", "curriculum__omni", head="X05", rng=rng, n_eps=10, ratio=0.5,
                     ge_mods=ge)
    cands = _cands(_entry(a, head="1c-004"), _entry(b, head="2b"), _entry(c, head="2a"))
    borrowed = res.resolve(dict(cands))["watch_metrics"]["ratio_by_hold.lat_r"]
    assert "borrowed_from" in borrowed
    out = res.resolve(dict(cands), curriculum_runs=[cur_a, cur_b])
    got = out["watch_metrics"]["ratio_by_hold.lat_r"]
    assert "borrowed_from" not in got
    assert got["rederived_from"], "the re-derived entry names the curriculum pair it came from"
    assert got["R_eval"] != pytest.approx(borrowed["R_eval"])
    assert "curriculum__omni" in out["by_row"]
    assert any("re-derived ratio_by_hold.lat_r" in n for n in out["notes"])


def test_curriculum_runs_refuses_a_non_curriculum_row(three_heads, tmp_path):
    a, b, c = three_heads
    with pytest.raises(res.ResolutionError, match="curriculum"):
        res.resolve(_cands(_entry(a, head="1c-004"), _entry(b, head="2b"), _entry(c, head="2a")),
                    curriculum_runs=[a])


def test_one_curriculum_run_keeps_the_borrowed_R(tmp_path, ge, three_heads):
    a, b, c = three_heads
    cur = make_run(tmp_path / "ca", "curriculum__omni", head="X01",
                   rng=np.random.default_rng(15), n_eps=8, ge_mods=ge)
    out = res.resolve(_cands(_entry(a, head="1c-004"), _entry(b, head="2b"), _entry(c, head="2a")),
                      curriculum_runs=[cur])
    assert out["watch_metrics"]["ratio_by_hold.lat_r"]["borrowed_from"] == "ratio_by_hold.lat_l"
    assert any("fewer than two heads" in n for n in out["notes"])


# ------------------------------------------------------------------ decision 6's null construction
def _null_columns(n, rng, *, k_f=6, k_t=4):
    """Per-episode columns with the real shapes: a correlated latent plus per-metric noise, holes
    where an episode produced nothing, and a Bernoulli completion / survival pair."""
    latent = rng.standard_normal(n)
    cols, specs = {}, []
    for i, hold in enumerate(("turn_l90", "turn_r90", "move_turn")):
        for key, stem, block in (("closed", "facing_closed_by_hold", ("F", "F_abs")),
                                 ("progress", "yaw_progress_by_hold", ("F",))):
            v = 0.5 * latent + rng.standard_normal(n)
            v[rng.random(n) < 0.35] = np.nan          # only ~65 % of episodes reach a turn hold
            cols[f"{key}:{hold}"] = v
            specs.append(res._spec(f"{stem}.{hold}", block[0], block, hold, (f"{key}:{hold}",)))
    for hold in ("fwd", "lat_l", "diag", "back"):
        v = 0.4 * latent + rng.standard_normal(n)
        v[rng.random(n) < 0.2] = np.nan
        cols[f"ratio:{hold}"] = v
        specs.append(res._spec(f"ratio_by_hold.{hold}", "T", ("T",), hold, (f"ratio:{hold}",)))
        cols[f"tripod:{hold}"] = 0.5 + 0.1 * latent + 0.1 * rng.standard_normal(n)
    cols["completed"] = (latent + rng.standard_normal(n) > 0).astype(np.float64)
    cols["surv:back"] = (latent + rng.standard_normal(n) > -1.0).astype(np.float64)
    cols["surv:turn_l90"] = np.minimum(cols["surv:back"],
                                       (latent + rng.standard_normal(n) > -0.6).astype(np.float64))
    cols["surv:turn_r90"] = np.minimum(cols["surv:turn_l90"],
                                       (latent + rng.standard_normal(n) > -0.4).astype(np.float64))
    specs.append(res._spec("completion", "guard", (), None, ("completed",), "mean",
                           level=0.99, one_sided=True))
    specs.append(res._spec("tripod_walking_dir", "guard", (), None,
                           tuple(f"tripod:{h}" for h in ("fwd", "lat_l", "diag", "back")),
                           "min_of_medians", level=0.99, one_sided=True))
    specs.append(res._spec("turn_survival_l90", "guard", (), "turn_l90",
                           ("surv:turn_l90", "surv:back"), "ratio_of_means",
                           level=0.99, one_sided=True))
    specs.append(res._spec("turn_survival_r90", "guard", (), "turn_r90",
                           ("surv:turn_r90", "surv:turn_l90"), "ratio_of_means",
                           level=0.99, one_sided=True))
    del k_f, k_t
    return cols, specs


def test_the_null_candidate_is_calibrated_at_its_declared_levels():
    """Decision 6's pinned test: a run submitted against itself must almost always read all-TIE.

    Two independent episode resamples of one run, 1000 trials, thresholds re-derived by the same
    bootstrap on the same resample distribution. The per-block marginal then pins the quantile
    arithmetic itself (a 95 % spread is exceeded 5 % of the time), and all-TIE lands in the plan's
    band -- above the 0.95**2 * 0.99**4 ~ 0.87 independence bound, because blocks and guards share
    the fall events that make an episode's holds missing together.
    """
    rng = np.random.default_rng(20260919)
    cols, specs = _null_columns(100, rng)
    out = res.null_construction(cols, specs, n=100, B=2000, N=1000, rng=rng)
    assert 0.85 <= out["all_tie"] <= 0.98, out
    for block in ("F", "T"):
        assert out["blocks"][block] == pytest.approx(0.95, abs=0.03), (block, out["blocks"])
    for guard, tie in out["guards"].items():
        assert tie >= 0.96, (guard, tie)
    # widening every threshold can only add TIEs
    wide = res.null_construction(cols, specs, n=100, N=1000, rng=np.random.default_rng(1),
                                 R_metrics={k: (None if v is None else 3 * v)
                                            for k, v in out["R_metrics"].items()},
                                 R_blocks={k: (None if v is None else 3 * v)
                                           for k, v in out["R_blocks"].items()})
    assert wide["all_tie"] >= out["all_tie"]


@pytest.mark.skipif(not RECORDED_1C.exists(), reason="the recorded 1c-004 slow row is untracked")
def test_the_null_candidate_on_the_recorded_1c_004_row():
    """The plan's own dataset: the 1c-004 ``slow__omni`` row as candidate and control."""
    row = res.load_row({"run_dir": str(RECORDED_1C)}, ring_proxy=False, rule=RULE)
    assert row.mask_deviation == 0.0 and row.facing_closed_source == "metric"
    specs, _ = res.specs_from_rule(RULE, row.hold_labels)
    out = res.null_construction(row.cols, specs, n=len(row.env_index), B=2000, N=1000,
                                rng=np.random.default_rng(20260919))
    assert 0.85 <= out["all_tie"] <= 0.98, out
    for block in ("F", "T"):
        assert out["blocks"][block] == pytest.approx(0.95, abs=0.03), (block, out["blocks"])


# ---------------------------------------------------------------------------------- layout / self-pins
def test_repo_root_is_self_relative_and_the_source_carries_no_dated_path():
    assert Path(res.REPO).resolve() == REPO
    source = RES_PY.read_text()
    assert "experiments/2026-" not in source
    assert "rx." not in source
    assert "rf." not in source
    assert "STUDENT_TASK" not in source
    assert not (TOOLS / "__init__.py").exists()


def test_summary_text_names_every_input_and_the_resolution_line(three_heads):
    a, b, c = three_heads
    out = res.resolve(_cands(_entry(a, head="1c-004"), _entry(b, head="2b"),
                             _entry(c, head="2a-reeval")))
    text = res.summary_text(out)
    for head in ("1c-004", "2b", "2a-reeval"):
        assert head in text
    assert out["resolution_line"] in text
    assert "guards @0.99 one-sided" in out["resolution_line"]
    assert "B=%d" % out["B"] in out["resolution_line"]


# ------------------------------------------------------------ the null construction under D00's R
LADDER = (REPO / "parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task/experiments"
          / "2026-09-19_facing_tracking_ladder")
SHIPPED_D00 = LADDER / "resolution.json"
#: The pre-suite ``candidates.yaml`` (``git show 464be672:<ladder>/candidates.yaml``), frozen byte for
#: byte: the live yaml moves to rule version 2, and D00 is the record of THIS spec.
CANDIDATES_V1 = REPO / "tests/unit/fixtures/candidates_v1_2026-09-19.yaml"


def test_the_null_construction_is_run_under_D00s_own_cross_head_R():
    """Decision 6 asks for the all-TIE band with the eval-only R AND adds "under D00's cross-head R
    the marginal is asserted only >= 0.92".

    That second construction is the one the campaign actually decides in: every candidate is judged
    against the SHIPPED half-widths, not against a threshold re-derived on its own resample
    distribution. ``resolution.py`` computes it into ``diagnostics.null_under_d00_R`` at D00 time,
    but nothing asserted it, so a later re-measure of D00 (a new input, a different B or rng_seed,
    the curriculum rows landing) could drop the null candidate's all-TIE rate under the plan's 0.85
    with the suite still green.
    """
    if not (RECORDED_1C.exists() and SHIPPED_D00.exists()):
        pytest.skip("the recorded 1c-004 row or the shipped resolution.json is untracked")
    shipped = json.loads(SHIPPED_D00.read_text())
    row = res.load_row({"run_dir": str(RECORDED_1C)}, ring_proxy=False, rule=RULE)
    specs, _ = res.specs_from_rule(RULE, row.hold_labels)
    scored = shipped["by_row"][res.LEGACY_SCREEN_ROW]     # D00 is the v1 record: slow__omni
    r_metrics = {mid: m["R_eval"] for mid, m in scored["metrics"].items()}
    r_blocks = {name: b["R_eval"] for name, b in scored["blocks"].items()}
    assert r_metrics and r_blocks, "the shipped record carries the screen row's own table"
    out = res.null_construction(row.cols, specs, n=len(row.env_index), B=2000, N=1000,
                                rng=np.random.default_rng(20260919),
                                R_metrics=r_metrics, R_blocks=r_blocks)
    assert 0.85 <= out["all_tie"] <= 0.98, out["all_tie"]
    for block in ("F", "T"):
        assert out["blocks"][block] >= 0.92, (block, out["blocks"])
    # the record's own diagnostic is the same construction, so it must agree qualitatively
    diag = (shipped.get("diagnostics") or {}).get("null_under_d00_R") or {}
    assert 0.85 <= diag["all_tie"] <= 0.98, diag
    assert all(v >= 0.92 for v in diag["blocks"].values()), diag["blocks"]
    # ... and it names the degenerate guard rather than hiding it (the CHARTER caveat)
    assert min(diag["guards"].values()) >= 0.90, diag["guards"]


def test_the_shipped_record_pins_its_control_head_rather_than_the_input_order():
    """``R_eval`` is the MAX over the pairs a control forms, so the control is load-bearing. With
    nothing in the yaml naming it, a reorder of ``resolution_inputs`` would silently re-fit D00."""
    if not SHIPPED_D00.exists():
        pytest.skip("the shipped resolution.json is untracked")
    import yaml
    # The FROZEN v1 spec D00 was measured against (the live yaml is rule v2 since the suite).
    doc = yaml.safe_load(CANDIDATES_V1.read_text())
    assert doc["decision_rule"]["control_head"] == "1c-004"
    shipped = json.loads(SHIPPED_D00.read_text())
    for sid, table in shipped["by_row"].items():
        assert table["control_head"] == "1c-004", sid


def test_a_metric_in_two_blocks_takes_the_wider_of_their_levels():
    """``facing_closed_by_hold.*`` is in F and F_abs. It gets ONE half-width, so a rule that
    levelled the absolute-units blocks differently used to mis-level the shared members on the order
    of the yaml's keys."""
    rule = json.loads(json.dumps(RULE))
    rule["blocks"]["F"]["level"] = 0.95
    rule["blocks"]["F_abs"]["level"] = 0.99
    specs, _ = res.specs_from_rule(rule, ("turn_l90", "turn_r90", "move_turn", "fwd", "lat_l",
                                          "diag", "back"))
    shared = [s for s in specs if s.mid == "facing_closed_by_hold.turn_l90"]
    assert len(shared) == 1 and shared[0].level == 0.99
    # and the other direction: swapping which block is tighter gives the same answer
    rule["blocks"]["F"]["level"], rule["blocks"]["F_abs"]["level"] = 0.99, 0.95
    specs2, _ = res.specs_from_rule(rule, ("turn_l90", "turn_r90", "move_turn", "fwd", "lat_l",
                                           "diag", "back"))
    shared2 = [s for s in specs2 if s.mid == "facing_closed_by_hold.turn_l90"]
    assert len(shared2) == 1 and shared2[0].level == 0.99


# =================================================================================================
# Rule version 2: D00s on the category suite (per-category eval plan 2026-09-21, D5 / D9 / D10)
# =================================================================================================
EC = res._ec()
ALL7 = list(EC.CATEGORIES)
RULE_V2 = EC.default_rule_v2(bootstrap={"B": 200, "paired_on": "env_index", "rng_seed": 7},
                             control_head="1c-004",
                             charter_floors={"completion": 0.10, "tripod_walking_dir": 0.056})
EVAL_SHA = "e" * 64
STAMP = "2026-09-22_01-00-00"
NormalDistZ99 = 2.3263478740408408          # z_0.99, the one-sided guard level


def _union(decisive=None):
    dec = ALL7 if decisive is None else decisive
    return EC.expand_rule_union(RULE_V2, dec, dec)


def _stamp_dir(root: Path, stamp: str = STAMP) -> Path:
    """``<eval_dir>/suite__omni/seed001/<stamp>`` -- the harness's stamp layout."""
    return root / SUITE / "seed001" / stamp


def _pair_entry(root: Path, head: str, run: Path, **kw) -> dict:
    """A checkpoint-shaped ``role: pair`` input (plan D9 item 2) with its recorded stamp."""
    return {"head": head, "role": "pair", "preset": "1c", "eval_dir": str(root),
            "checkpoint": str(ckpt_of(run)), "stamp": run.name, "row": SUITE, **kw}


def _fake_d00():
    """A D00-shaped record: light/ring by_row tables and the frozen v1 decision_rule."""
    table = {"metrics": {"completion": {"R_eval": 0.11, "block": "guard"}},
             "blocks": {"F": {"R_eval": 0.8}, "T": {"R_eval": 0.7}},
             "watch_metrics": {}, "control_head": "1c-004", "heads": ["1c-004", "2b"]}
    return {"by_row": {"light__omni": dict(table), "ring__omni": dict(table, heads=["x"]),
                       "slow__omni": dict(table, heads=["slow"])},
            "decision_rule": json.loads(json.dumps(RULE))}


@pytest.fixture(scope="module")
def suite_heads(tmp_path_factory, ge):
    """1c-004 / 2b / 2a-reeval on ``suite__omni``, each a stamp under its own reeval root."""
    base = tmp_path_factory.mktemp("suite_heads")
    rng = np.random.default_rng(21)
    out = {}
    for head, kw in (("1c-004", dict(closes=0.7, fall_offset=0)),
                     ("2b", dict(closes=0.4, ratio=0.5, tripod=0.4, fall_offset=1)),
                     ("2a-reeval", dict(closes=0.6, fall_offset=2))):
        root = base / f"reeval_{head}"
        run = _stamp_dir(root)
        run.parent.mkdir(parents=True)
        make_run(run, SUITE, head=head, rng=rng, ge_mods=ge, suite=True, **kw)
        out[head] = (root, run)
    return out


def _v2_cands(suite_heads, *extra, rule=None):
    return {"resolution_inputs": [_pair_entry(root, head, run)
                                  for head, (root, run) in suite_heads.items()] + list(extra),
            "decision_rule": json.loads(json.dumps(RULE_V2 if rule is None else rule))}


@pytest.fixture(scope="module")
def d00s(suite_heads):
    """The D00s payload of the three suite heads -- resolve() exactly as the driver calls it."""
    return res.resolve(_v2_cands(suite_heads), eval_sha=EVAL_SHA, decisive_at=sorted(ALL7),
                       d00=_fake_d00(), supersedes="D00", candidates_sha256="c" * 64)


def _copy_run(src: Path, dst: Path) -> Path:
    shutil.copytree(src, dst)
    ck = ckpt_of(src)
    if ck.exists() and not ckpt_of(dst).exists():
        shutil.copytree(ck.parent, ckpt_of(dst).parent)
    return dst


# ---------------------------------------------------------------------------- the clustered draw
def test_clustered_draw_keeps_every_hold_count_and_hand_split(suite_heads):
    W, E, B = 10, 7, 50
    draw = res.cluster_draw(np.random.default_rng(3), E, W, B)
    assert draw.shape == (B, W * E)
    # every drawn position stays inside its own wave -> every hold keeps exactly E episodes
    assert np.array_equal(draw // E, np.broadcast_to(np.repeat(np.arange(W), E), draw.shape))
    # ... and the SAME env draw is taken in every wave (the robot is resampled, not the episode)
    env = draw % E
    for w in range(1, W):
        assert np.array_equal(env[:, w * E:(w + 1) * E], env[:, :E])
    assert len(np.unique(env)) > 1, "a resample, not the identity"
    # on a real suite row: per replicate, every hold has its n and each paired category its l/r split
    _, run = suite_heads["1c-004"]
    row = res.load_row(_entry(run, head="1c-004"), ring_proxy=False, rule=_union())
    assert row.suite and row.waves == 10 and row.episodes_per_wave == 8
    idx = res.cluster_draw(np.random.default_rng(4), row.episodes_per_wave, row.waves, 40)
    holds = np.asarray(row.hold_index)[idx]
    hands = np.asarray(row.hand_index, dtype=object)[idx]
    cats = np.asarray(row.category_index)[idx]
    for label in row.hold_of_wave:
        assert np.all(np.sum(holds == label, axis=1) == 8), label
    for cat in ("side", "turn", "move_turn"):
        in_cat = cats == cat
        assert np.all(np.sum(in_cat & (hands == "l"), axis=1) == 8), cat
        assert np.all(np.sum(in_cat & (hands == "r"), axis=1) == 8), cat
    # one wave is the legacy episode draw, bit for bit
    assert np.array_equal(res.cluster_draw(np.random.default_rng(11), 13, 1, 40),
                          np.random.default_rng(11).integers(0, 13, size=(40, 13)))


def test_suite_pairing_on_env_index_pairs_within_category_and_hand(suite_heads, tmp_path, ge):
    rows = {h: res.load_row(_entry(run, head=h), ring_proxy=False, rule=_union())
            for h, (_, run) in suite_heads.items()}
    a, b = rows["1c-004"], rows["2b"]
    assert a.env_index == b.env_index and a.hold_index == b.hold_index
    assert a.hand_index == b.hand_index and a.pairing_key == b.pairing_key
    specs, _ = res.specs_from_rule(_union(), a.hold_labels, suite=True)
    pr = res.bootstrap_pair(a, b, B=200, rng_seed=7, specs=specs)
    # the pair-local block diagnostics iterate the rule's blocks (F_mt included), never a literal
    assert list(pr.blocks) == ["F", "F_mt", "T", "F_abs", "F_mt_abs", "T_abs"]
    assert set(pr.blocks["F_mt"]["members"]) == {"cat.move_turn.facing_closed",
                                                 "cat.move_turn.yaw_progress"}
    # the SAME clustered draw is applied to both heads, and a hand's column is NaN outside that hand
    draw = res.cluster_draw(res.pair_rng(7, pr.name), a.episodes_per_wave, a.waves, 200)
    for mid, col in (("cat.side.by_hand.l.ratio", "cat:side:hand:l:ratio"),
                     ("cat.move_turn.facing_closed", "cat:move_turn:facing_closed"),
                     ("cat.turn.survival", "cat:turn:survival")):
        red = "mean" if mid.endswith("survival") else "median"
        want = res._reduce(red, [a.cols[col][draw]]) - res._reduce(red, [b.cols[col][draw]])
        np.testing.assert_array_equal(pr.deltas[mid], want)
    lat_l = np.asarray(a.hold_index) == "lat_l"
    assert np.all(np.isfinite(a.cols["cat:side:hand:l:survival"][lat_l]))
    assert np.all(np.isnan(a.cols["cat:side:hand:l:survival"][~lat_l]))
    # a row of another geometry is refused, never paired on env_index
    odd = make_run(tmp_path / "odd", SUITE, head="odd", rng=np.random.default_rng(5), ge_mods=ge,
                   suite=True, n_env=6)
    other = res.load_row(_entry(odd, head="odd"), ring_proxy=False, rule=_union())
    with pytest.raises(res.ResolutionError, match="pairing precondition"):
        res.bootstrap_pair(a, other, B=50, rng_seed=7, specs=specs)


# ----------------------------------------------------------------------- the v2 record (D00s)
def test_resolve_expands_the_v2_rule_and_publishes_F_mt_R(d00s):
    out = d00s
    assert out["rule_version"] == 2 and out["screen_row"] == SUITE
    assert out["eval_sha"] == EVAL_SHA and out["supersedes"] == "D00"
    assert out["decisive_at"] == sorted(ALL7), "recorded exactly as the driver passed it"
    six = ("F", "F_mt", "T", "F_abs", "F_mt_abs", "T_abs")
    assert set(out["blocks"]) == set(six)
    for block in six:
        assert out["blocks"][block]["R_eval"] > 0.0, block
        for mid in out["blocks"][block]["members"]:
            assert out["metrics"][mid]["R_eval"] > 0.0, (block, mid)
    assert set(out["blocks"]["F_mt"]["members"]) == {"cat.move_turn.facing_closed",
                                                     "cat.move_turn.yaw_progress"}
    assert set(out["blocks"]["F_mt_abs"]["members"]) == {"cat.move_turn.facing_closed",
                                                         "cat.move_turn.net_yaw"}
    assert set(out["blocks"]["F"]["members"]) == {"cat.turn.facing_closed", "cat.turn.yaw_progress"}
    # both move_turn guards (ratio for ratio-units candidates, achieved_speed for absolute_units)
    for g in ("cat.move_turn.ratio", "cat.move_turn.achieved_speed", "cat_tripod_min"):
        assert out["metrics"][g]["block"] == "guard" and out["metrics"][g]["R_eval"] > 0.0, g
        assert out["metrics"][g]["one_sided"] and out["metrics"][g]["level"] == 0.99
    guards = {m for m, v in out["metrics"].items() if v["block"] == "guard"}
    assert guards == set(_union()["guards"]["metrics"]) and len(guards) == 10
    assert {f"cat.{c}.survival" for c in ALL7} <= guards
    sys_ = out["scored_systems"]
    assert sys_["ratio"]["blocks"] == ["F", "F_mt", "T"] and len(sys_["ratio"]["guards"]) == 9
    assert sys_["absolute"]["blocks"] == ["F_abs", "F_mt_abs", "T_abs"]
    assert out["acceptance"] == {"ok": True, "failures": []}
    assert out["resolution_line"].startswith("RESOLUTION D00s | env-clustered paired bootstrap")
    assert "F_mt R=" in out["resolution_line"] and "F_mt_abs R=" in out["resolution_line"]
    assert out["by_category_n"]["side"] == {"n": 16, "n_env": 8, "holds": {"lat_l": 8, "lat_r": 8}}
    assert out["by_category_n"]["fwd"]["n"] == 8
    assert out["decision_rule"] == RULE_V2, "the raw v2 rule is frozen verbatim"
    assert out["watch_metrics"] and all(v["block"] == "watch" for v in out["watch_metrics"].values())
    assert not any("borrowed_from" in v for v in out["watch_metrics"].values())
    for rec in out["inputs"]:
        assert rec["role"] == "pair" and rec["suite"] and rec["stamp"] == STAMP
    json.dumps(out)                                         # JSON-safe as written
    text = res.summary_text(out)
    assert "rule_version 2 on suite__omni" in text and "acceptance ok" in text
    assert out["resolution_line"] in text
    # the legacy charter floors are cited against the WATCH R the suite measured for them
    assert any("charter floor cited, not applied: completion" in n for n in out["notes"])
    # the raw v2 rule never reaches specs_from_rule unexpanded (it would score the legacy surface)
    with pytest.raises(res.ResolutionError, match="not expanded"):
        res.specs_from_rule(RULE_V2, res._manifest_scenario(SUITE)["hold_labels"], suite=True)


def test_null_all_tie_scores_exactly_the_ratio_systems_blocks_and_guards(d00s, suite_heads):
    out = d00s
    ratio = EC.expand_rule(RULE_V2, ALL7, ALL7, absolute_units=False)
    assert out["diagnostics"]["null_scored"] == {"blocks": ["F", "F_mt", "T"],
                                                 "guards": sorted(ratio["guards"]["metrics"])}
    assert len(out["diagnostics"]["null_scored"]["guards"]) == 9
    assert out["null_nominal"] == pytest.approx(0.95 ** 3 * 0.99 ** 9)
    # the absolute-units system is measured and reported, never scored into all_tie
    cal = out["diagnostics"]["null_calibrated"]
    assert "cat.move_turn.achieved_speed" in cal["guards"] and "F_abs" in cal["blocks"]
    # pinned directly: all_tie is the conjunction of exactly those TIE indicators
    _, run = suite_heads["1c-004"]
    row = res.load_row(_entry(run, head="1c-004"), ring_proxy=False, rule=_union())
    specs, _ = res.specs_from_rule(_union(), row.hold_labels, suite=True)
    kw = dict(n=len(row.env_index), B=200, N=400, waves=row.waves,
              survival_guards=[f"cat.{c}.survival" for c in ALL7])
    base = res.null_construction(row.cols, specs, rng=np.random.default_rng(9),
                                 scored_blocks=["F", "F_mt", "T"],
                                 scored_guards=ratio["guards"]["metrics"], **kw)
    wider = res.null_construction(row.cols, specs, rng=np.random.default_rng(9),
                                  scored_blocks=["F", "F_mt", "T", "F_abs", "F_mt_abs", "T_abs"],
                                  scored_guards=_union()["guards"]["metrics"], **kw)
    assert base["scored"]["blocks"] == ["F", "F_mt", "T"]
    assert wider["all_tie"] <= base["all_tie"], "more conjuncts can only remove TIEs"
    assert base["blocks"] == wider["blocks"] and base["guards"] == wider["guards"]


def test_d00s_copies_light_and_ring_by_row_from_d00(d00s, suite_heads):
    d00 = _fake_d00()
    for row in ("light__omni", "ring__omni"):
        want = dict(d00["by_row"][row], by_row_source="D00")
        assert d00s["by_row"][row] == want, row
    assert "slow__omni" not in d00s["by_row"], "only light/ring are copied"
    assert d00s["by_row_rule"] == d00["decision_rule"] and d00s["by_row_rule"] == RULE
    assert d00s["by_row_source"] == {"light__omni": "D00", "ring__omni": "D00"}
    assert "by_row_source" not in d00s["by_row"][SUITE], "the suite table is D00s' own"
    broken = _fake_d00()
    del broken["by_row"]["ring__omni"]
    with pytest.raises(res.ResolutionError, match="ring__omni"):
        res.resolve(_v2_cands(suite_heads), eval_sha=EVAL_SHA, decisive_at=ALL7, d00=broken)
    with pytest.raises(res.ResolutionError, match="decision_rule"):
        res.resolve(_v2_cands(suite_heads), eval_sha=EVAL_SHA, decisive_at=ALL7,
                    d00={"by_row": _fake_d00()["by_row"]})
    if SHIPPED_D00.exists():                  # the real record has both tables and its rule
        shipped = json.loads(SHIPPED_D00.read_text())
        real = res.resolve(_v2_cands(suite_heads), eval_sha=EVAL_SHA, decisive_at=ALL7, d00=shipped)
        assert real["by_row"]["ring__omni"]["metrics"] == shipped["by_row"]["ring__omni"]["metrics"]
        assert real["by_row_rule"] == shipped["decision_rule"]


def test_one_member_block_R_eval_is_exactly_one(suite_heads):
    """A one-member block standardises its member by that member's own merged R: z = delta / R, and
    the 95 % half-width of z is then exactly 1 on the pair that set R (<= 1 on the others)."""
    out = res.resolve(_v2_cands(suite_heads), eval_sha=EVAL_SHA, decisive_at=["stand", "fwd"])
    assert out["blocks"]["T"]["members"] == ["cat.fwd.ratio"]
    assert out["blocks"]["T_abs"]["members"] == ["cat.fwd.achieved_speed"]
    for block in ("T", "T_abs"):
        assert out["blocks"][block]["R_eval"] == pytest.approx(1.0, abs=1e-12), block
        assert all(v <= 1.0 + 1e-12 for v in out["blocks"][block]["per_pair"].values())
    # F / F_mt have no decisive member: N/A -- absent from the record, never a TIE-ing R
    assert not {"F", "F_mt", "F_abs", "F_mt_abs"} & set(out["blocks"])
    assert out["scored_systems"]["ratio"]["blocks"] == ["T"]
    assert set(out["scored_systems"]["ratio"]["na_blocks"]) == {"F", "F_mt"}
    assert out["null_nominal"] == pytest.approx(0.95 * 0.99 ** 3)   # stand/fwd survival + tripod_min


def test_the_screen_row_is_the_suite(three_heads, suite_heads):
    assert res.SCREEN_ROW == "suite__omni" and res.LEGACY_SCREEN_ROW == "slow__omni"
    a, b, c = three_heads
    legacy = _cands(_entry(a, head="1c-004"), _entry(b, head="2b"))
    assert res.resolve(legacy)["screen_row"] == "slow__omni", "a v1 rule stays on the legacy row"
    with pytest.raises(res.ResolutionError, match="suite__omni"):
        res.resolve({"resolution_inputs": legacy["resolution_inputs"],
                     "decision_rule": json.loads(json.dumps(RULE_V2))}, decisive_at=ALL7)
    # informational legacy rows ride along under v2 without pairs or category-id noise
    out = res.resolve(_v2_cands(suite_heads, _entry(a, head="1c-004", role="informational")),
                      decisive_at=ALL7)
    assert out["screen_row"] == SUITE and "slow__omni" not in out["by_row"]
    assert not any("slow__omni" in n and "category id" in n for n in out["notes"])
    with pytest.raises(res.ResolutionError, match="decisive_at"):
        res.resolve(_v2_cands(suite_heads))
    with pytest.raises(res.ResolutionError, match="rule_version 2 arguments"):
        res.resolve(legacy, decisive_at=ALL7)


def test_borrow_and_curriculum_runs_are_retired_under_rule_v2(suite_heads, three_heads, d00s):
    assert any("retired" in n for n in d00s["notes"])
    assert not any("borrow lat_l" in n for n in d00s["notes"])
    assert "ratio_by_hold.lat_r" not in d00s["watch_metrics"]
    a, _, _ = three_heads
    with pytest.raises(res.ResolutionError, match="retired under rule_version 2"):
        res.resolve(_v2_cands(suite_heads), decisive_at=ALL7, curriculum_runs=[a])


# --------------------------------------------------------------------- the inputs and their stamps
def test_resolve_eval_dir_requires_the_completion_marker_and_matches_checkpoints_by_sha(
        tmp_path, ge):
    root = tmp_path / "reeval_x"
    run = _stamp_dir(root)
    run.parent.mkdir(parents=True)
    make_run(run, SUITE, head="x", rng=np.random.default_rng(6), ge_mods=ge, suite=True, n_env=4)
    ck = ckpt_of(run)
    assert res.resolve_eval_dir(root, SUITE) == run
    # the harness records an ABSOLUTE path; candidates.yaml a repo-relative one -- still the same file
    rel = os.path.relpath(ck, res.REPO)
    assert not Path(rel).is_absolute()
    assert res.resolve_eval_dir(str(root), SUITE, checkpoint=rel) == run
    # the same bytes somewhere else match by sha256 ...
    twin = tmp_path / "elsewhere" / "model_999.pt"
    twin.parent.mkdir()
    twin.write_bytes(ck.read_bytes())
    assert res.resolve_eval_dir(root, SUITE, checkpoint=twin) == run
    assert res.checkpoint_matches(twin, json.loads((run / "run_meta.json").read_text()))
    # ... and another checkpoint's bytes are refused
    other = tmp_path / "other" / "model_999.pt"
    other.parent.mkdir()
    other.write_bytes(b"another head")
    with pytest.raises(res.ResolutionError, match="not the entry's checkpoint"):
        res.check_suite_stamp(run, checkpoint=other)
    # a missing raw -> a refusal naming the stamp, never a FileNotFoundError
    gone = next((run / "raw").glob("episode_*.npz"))
    gone.unlink()
    with pytest.raises(res.ResolutionError, match="partial raw set") as exc:
        res.check_suite_stamp(run)
    assert str(run) in str(exc.value)
    with pytest.raises(res.ResolutionError):
        res.load_row(_entry(run), ring_proxy=False, rule=_union())
    # no completion marker -> partial: never read, whatever else is there
    part = _stamp_dir(tmp_path / "reeval_y")
    part.parent.mkdir(parents=True)
    make_run(part, SUITE, head="y", rng=np.random.default_rng(7), ge_mods=ge, suite=True, n_env=4,
             complete=False)
    with pytest.raises(res.ResolutionError, match="complete.json"):
        res.resolve_eval_dir(tmp_path / "reeval_y", SUITE)
    with pytest.raises(res.ResolutionError, match="complete.json"):
        res.load_row(_entry(part), ring_proxy=False, rule=_union())


def test_resolve_eval_dir_takes_the_newest_stamp_or_refuses(tmp_path, ge):
    root = tmp_path / "reeval_z"
    old = _stamp_dir(root, "2026-09-22_01-00-00")
    new = _stamp_dir(root, "2026-09-22_02-00-00")
    old.parent.mkdir(parents=True)
    ck = tmp_path / "ck" / "model_999.pt"
    ck.parent.mkdir()
    ck.write_bytes(b"the head")
    for run in (old, new):
        make_run(run, SUITE, head="z", rng=np.random.default_rng(8), ge_mods=ge, suite=True,
                 n_env=4, checkpoint=ck)
    assert res.resolve_eval_dir(root, SUITE, checkpoint=ck) == new
    (new / "complete.json").unlink()                        # the newer one is partial
    assert res.resolve_eval_dir(root, SUITE, checkpoint=ck) == old
    (old / "complete.json").unlink()
    with pytest.raises(res.ResolutionError, match="passes the freshness rule") as exc:
        res.resolve_eval_dir(root, SUITE)
    assert "2026-09-22_01-00-00" in str(exc.value) and "2026-09-22_02-00-00" in str(exc.value)
    with pytest.raises(res.ResolutionError, match="no eval stamps"):
        res.resolve_eval_dir(tmp_path / "nothing", SUITE)


def test_a_pair_input_reads_its_recorded_stamp_and_refuses_without_one(suite_heads, tmp_path, ge):
    root, run = suite_heads["2b"]
    entry = _pair_entry(root, "2b", run)
    row = res.load_row(entry, ring_proxy=False, rule=_union())
    assert row.run_dir == run and row.stamp == STAMP and row.entry_checkpoint == str(ckpt_of(run))
    # a NEWER complete stamp of the same checkpoint is never picked over the recorded one
    newer = _copy_run(run, _stamp_dir(root, "2026-09-30_00-00-00"))
    try:
        assert res.load_row(entry, ring_proxy=False, rule=_union()).run_dir == run
    finally:
        shutil.rmtree(newer)
        shutil.rmtree(ckpt_of(newer).parent, ignore_errors=True)
    missing = {k: v for k, v in entry.items() if k != "stamp"}
    with pytest.raises(res.ResolutionError, match=r"2b \(role: pair\): no reeval stamp"):
        res.load_row(missing, ring_proxy=False, rule=_union())
    with pytest.raises(res.ResolutionError, match="both run_dir and checkpoint"):
        res.load_row(dict(entry, run_dir=str(run)), ring_proxy=False, rule=_union())
    with pytest.raises(res.ResolutionError, match="neither run_dir nor"):
        res.load_row({"head": "ghost", "role": "pair"}, ring_proxy=False, rule=_union())
    with pytest.raises(res.ResolutionError, match="needs both checkpoint and eval_dir"):
        res.load_row({"head": "half", "checkpoint": "x.pt"}, ring_proxy=False, rule=_union())
    # resolve() refuses the whole record when one pair input has no stamp (the driver halts D00s)
    cands = _v2_cands(suite_heads)
    cands["resolution_inputs"][1] = missing
    with pytest.raises(res.ResolutionError, match="2b"):
        res.resolve(cands, decisive_at=ALL7)


def test_seed_pair_role_is_excluded_from_cross_head_pairs(suite_heads, tmp_path, ge):
    root = tmp_path / "X02"
    run = _stamp_dir(root)
    run.parent.mkdir(parents=True)
    make_run(run, SUITE, head="B0-s2", rng=np.random.default_rng(12), ge_mods=ge, suite=True,
             closes=0.5, fall_offset=3)
    seed_pair = {"head": "B0-s2", "role": "seed_pair", "preset": "1c", "eval_dir": str(root),
                 "checkpoint": str(ckpt_of(run)), "row": SUITE}
    base = res.resolve(_v2_cands(suite_heads), decisive_at=ALL7)
    out = res.resolve(_v2_cands(suite_heads, seed_pair), decisive_at=ALL7)
    assert {p["name"] for p in out["pairs"]} == {"1c-004 vs 2b @suite__omni",
                                                 "1c-004 vs 2a-reeval @suite__omni"}
    rec = next(i for i in out["inputs"] if i["head"] == "B0-s2")
    assert rec["accepted"] is False and rec["role"] == "seed_pair" and "seed_pair" in rec["reason"]
    assert rec["run_dir"] == str(run), "resolved through resolve_eval_dir (newest fresh stamp)"
    for mid, m in out["metrics"].items():
        assert m["R_eval"] == base["metrics"][mid]["R_eval"], mid
        assert not any("B0-s2" in k for k in m["per_pair"]), mid


# ------------------------------------------------------------------------------ load_row on a suite
def test_load_row_verifies_a_suite_row_under_the_full_v2_rule_and_rule_less(suite_heads):
    _, run = suite_heads["1c-004"]
    row = res.load_row(_entry(run, head="1c-004"), ring_proxy=False, rule=_union())
    assert row.suite and row.facing_closed_source == "metric" and row.mask_deviation == 0.0
    assert row.reducer_check and all(v <= 1e-9 for v in row.reducer_check.values())
    assert {"cat.turn.facing_closed", "cat.move_turn.yaw_progress", "cat.side.ratio",
            "cat.stand.survival", "cat_tripod_min"} <= set(row.reducer_check)
    labels = row.hold_labels
    specs, unresolved = res.specs_from_rule(_union(), labels, suite=True)
    assert {"turn_survival_l90", "turn_survival_r90"} <= set(unresolved)
    assert "not nested" in res.why_unresolved("turn_survival_l90", labels, suite=True)
    assert not any(s.mid.startswith("turn_survival") for s in res.metric_specs(labels, suite=True))
    assert any(s.mid.startswith("turn_survival") for s in res.metric_specs(labels)), "legacy kept"
    # surv:<h> is each hold's OWN survival: NaN outside that hold's episodes
    agg = json.loads((run / "scenario_metrics.json").read_text())["aggregate"]
    own = np.asarray(row.hold_index)
    for label in row.hold_of_wave:
        col = row.cols[f"surv:{label}"]
        assert np.all(np.isnan(col[own != label])) and np.all(np.isfinite(col[own == label]))
        assert float(np.mean(col[own == label])) == pytest.approx(
            agg["suite"]["survival_by_hold"][label], abs=1e-12)
    assert np.all(np.isnan(row.cols["surv:settle"]))
    # a fall ON the wave's last step survived (env 7 of wave 0 -- the fixture's boundary case)
    assert row.cols["surv:stand"][7] == 1.0
    # rule-less: the fallback both shared helpers use verifies too, and is never refused
    bare = res.load_row(_entry(run), ring_proxy=False)
    assert bare.reducer_check and all(v <= 1e-9 for v in bare.reducer_check.values())
    summary = res.summarize_run_dir(run)
    assert summary["facing_closed_source"] == "metric"
    assert summary["facing_closed_by_hold"]["turn_l90"] is not None
    assert summary["facing_closed_by_hold"]["fwd"] is None, "a travel hold has nothing to close"
    assert summary["cat"]["side"]["n"] == 16 and summary["cat"]["side"]["n_env"] == 8
    closed, source = res.facing_closed_by_hold_from_raws(run)
    assert source == "metric" and closed["move_turn_r"] is not None


def test_load_row_refuses_a_populated_hold_that_is_not_the_episodes_own(suite_heads, tmp_path):
    _, src = suite_heads["2a-reeval"]
    run = _copy_run(src, tmp_path / "copy")
    path = run / "metrics" / "episode_32.json"               # wave 4 = back
    ep = json.loads(path.read_text())
    assert ep["hold"] == "back" and ep["holds"]["fwd"]["n_steps"] == 0
    ep["holds"]["fwd"]["n_steps"] = 5
    path.write_text(json.dumps(ep, sort_keys=True))
    with pytest.raises(res.ResolutionError, match="has 5 steps in hold 'fwd'"):
        res.load_row(_entry(run), ring_proxy=False, rule=_union())
    # an episode whose own-hold key disagrees with aggregate.suite is refused too
    ep["holds"]["fwd"]["n_steps"] = 0
    ep["hold"] = "diag"
    path.write_text(json.dumps(ep, sort_keys=True))
    with pytest.raises(res.ResolutionError, match="own hold is 'diag'"):
        res.load_row(_entry(run), ring_proxy=False, rule=_union())


def test_summary_with_survival_uses_the_suite_block(suite_heads, monkeypatch):
    _, run = suite_heads["2b"]

    def _no_walk(sid):                       # the cumulative walk must never run on a suite row
        raise AssertionError(f"_manifest_hold_s({sid!r}) called on a suite row")

    monkeypatch.setattr(res.rp, "_manifest_hold_s", _no_walk)
    agg = json.loads((run / "scenario_metrics.json").read_text())["aggregate"]
    summary = res._summary_with_survival(run, SUITE)
    assert summary["survival_by_hold"] == agg["suite"]["survival_by_hold"]
    assert "settle" not in summary["survival_by_hold"]
    # ... which is NOT what a cumulative walk over the union labels would give
    eps = json.loads((run / "scenario_metrics.json").read_text())["episodes"]
    geo = res._manifest_scenario(SUITE)
    walked = res.rp.survival_by_hold(eps, geo["hold_labels"], {"settle": 2.0, **geo["hold_s"]},
                                     dt=DT)
    assert walked["fwd"] != summary["survival_by_hold"]["fwd"]
    # the own-hold ends are round((prologue + settle + hold) / dt) for every label
    ends = res._survival_end_steps(SUITE, DT)
    assert set(ends) == set(geo["hold_s"]) and set(ends.values()) == {round(15.5 / DT)}
    assert res._manifest_scenario(SUITE)["kind"] == "category_suite"
    assert res._manifest_scenario("slow__omni")["kind"] == "schedule"


# ------------------------------------------------------------------------- the registry on a suite
def test_cat_ids_reproduce_summarize_cat_keys(suite_heads):
    """Every per-episode ``cat`` key, pooled and per hand, reproduces ``summarize()['cat']``."""
    _, run = suite_heads["1c-004"]
    labels = res._manifest_scenario(SUITE)["hold_labels"]
    ids = [f"cat.{c}.{k}" for c in ALL7 for k in res.CAT_COLUMN_KEYS]
    ids += [f"cat.{c}.by_hand.{h}.{k}" for c in EC.PAIRED_CATEGORIES for h in "lr"
            for k in res.CAT_COLUMN_KEYS]
    rule = {"rule_version": 2, "expanded": True, "blocks": {},
            "guards": {"level": 0.99, "metrics": ["cat_tripod_min"]},
            "composite": {"cat_tripod_min": ["fwd", "side", "back", "diag"]}, "watch": ids}
    specs, unresolved = res.specs_from_rule(rule, labels, suite=True)
    assert unresolved == [] and len(specs) == len(ids) + 1
    row = res.load_row(_entry(run), ring_proxy=False, rule=rule)
    checked = {m for m, v in row.reducer_check.items() if v == 0.0 or v <= 1e-9}
    assert set(ids) | {"cat_tripod_min"} <= set(row.reducer_check)
    assert set(row.reducer_check) == checked
    # ... against the actual summary values (not a vacuous both-missing pass) where there is one
    summary = res._summary_with_survival(run, SUITE)
    live = [m for m in ids if res._summary_value(summary, m) is not None]
    assert len(live) > len(ids) // 2
    # the counts are no per-episode quantity; category ids never resolve off a suite row
    for mid in ("cat.fwd.n", "cat.side.n_env"):
        assert res.spec_for(mid, labels, block="watch", blocks_all=(), level=0.95, one_sided=False,
                            suite=True) is None
        assert "count" in res.why_unresolved(mid, labels, suite=True)
    legacy_labels = res._manifest_scenario("slow__omni")["hold_labels"]
    for mid in ("cat.fwd.ratio", "cat_tripod_min"):
        assert res.spec_for(mid, legacy_labels, block="watch", blocks_all=(), level=0.95,
                            one_sided=False) is None
        assert "category_suite row only" in res.why_unresolved(mid, legacy_labels)


def test_reducer_check_covers_cat_tripod_min(suite_heads, tmp_path):
    _, src = suite_heads["1c-004"]
    row = res.load_row(_entry(src), ring_proxy=False, rule=_union())
    assert row.reducer_check["cat_tripod_min"] <= 1e-9
    spec = next(s for s in res.specs_from_rule(_union(), row.hold_labels, suite=True)[0]
                if s.mid == "cat_tripod_min")
    assert spec.needs == tuple(f"cat:{c}:tripod" for c in ("fwd", "side", "back", "diag"))
    assert spec.reducer == "min_of_medians" and spec.block == "guard"
    # its reference is the SHARED composite over the summary: move the fwd tripod and it disagrees
    run = _copy_run(src, tmp_path / "moved")
    sm_path = run / "scenario_metrics.json"
    sm = json.loads(sm_path.read_text())
    sm["aggregate"]["suite"]["by_category"]["fwd"]["tripod"]["median"] = -1.0
    sm_path.write_text(json.dumps(sm, sort_keys=True))
    with pytest.raises(res.ResolutionError, match="cat_tripod_min"):
        res.load_row(_entry(run), ring_proxy=False, rule=_union())
    # ... and a summary with no value for a member is a NAMED refusal, not a NaN
    sm["aggregate"]["suite"]["by_category"]["fwd"]["tripod"]["median"] = None
    sm_path.write_text(json.dumps(sm, sort_keys=True))
    with pytest.raises(res.ResolutionError, match="composite_guards gives no value"):
        res.load_row(_entry(run), ring_proxy=False, rule=_union())
    # a narrower decisive set narrows the composite (the expanded rule's own members)
    narrow = EC.expand_rule_union(RULE_V2, ["stand", "fwd", "back"], ["stand", "fwd", "back"])
    spec2 = next(s for s in res.specs_from_rule(narrow, row.hold_labels, suite=True)[0]
                 if s.mid == "cat_tripod_min")
    assert spec2.needs == ("cat:fwd:tripod", "cat:back:tripod")


# ------------------------------------------------------------------------------ guards and nulls
def test_survival_guard_R_has_the_binomial_floor(d00s):
    assert res.binomial_guard_floor(1.0, 100) == pytest.approx(0.0717, abs=5e-4)
    assert res.binomial_guard_floor(1.0, 200) == pytest.approx(0.0507, abs=5e-4)
    assert res.binomial_guard_floor(0.0, 100) == pytest.approx(res.binomial_guard_floor(1.0, 100))
    assert res.binomial_guard_floor(0.5, 100) == pytest.approx(
        NormalDistZ99 * math.sqrt(2 * 0.25 / 100))
    assert res.binomial_guard_floor(None, 100) is None and res.binomial_guard_floor(0.5, 0) is None
    # the pooled form: max(today's, R_binom / sqrt2); None keeps today's rule exactly
    assert res.pooled_guard_R(0.08, 0.03) == res.pooled_guard_R(0.08, 0.03, None)
    assert res.pooled_guard_R(0.0, None, r_binom=0.0717) == pytest.approx(0.0717 / math.sqrt(2))
    assert res.pooled_guard_R(None, None, r_binom=0.0717) == pytest.approx(0.0717 / math.sqrt(2))
    assert res.pooled_guard_R(0.3, 0.2, r_binom=0.01) == pytest.approx(res.pooled_guard_R(0.3, 0.2))
    assert res.pooled_guard_R(0.02, 0.01) < 0.0717 / math.sqrt(2)          # the floor binds here
    assert res.pooled_guard_R(0.02, 0.01, r_binom=0.0717) == pytest.approx(0.0717 / math.sqrt(2))
    # D00s names each survival guard's floor per cross-head pair
    for c in ALL7:
        m = d00s["metrics"][f"cat.{c}.survival"]
        for name, pp in m["binomial_floor"]["per_pair"].items():
            pt = m["points"][name]
            assert pp["p_bar"] == pytest.approx((pt["a"] + pt["b"]) / 2)
            assert pp["R_binom"] == pytest.approx(res.binomial_guard_floor(pp["p_bar"], pp["n_c"]))
            assert pp["n_c"] == (8 if c in ("stand", "fwd", "back", "diag") else 16)
        assert m["binomial_floor"]["R_binom"] == max(
            v["R_binom"] for v in m["binomial_floor"]["per_pair"].values())
    assert any(n.startswith("binomial floors") for n in d00s["notes"])
    assert set(d00s["null_survival_tie"]) == {f"cat.{c}.survival" for c in ALL7}



def test_null_band_uses_the_clustered_draw_and_the_floored_guard_R():
    W, E = 10, 12
    n = W * E
    wave = np.repeat(np.arange(W), E)
    # a quantity that exists only in wave 3: a clustered resample always has exactly E of it, so its
    # mean indicator over all episodes never moves -- an unclustered one does
    ind = res._spec("x", "guard", (), None, ("x",), "mean", level=0.99, one_sided=True)
    cols = {"x": (wave == 3).astype(np.float64)}
    clustered = res.null_construction(cols, [ind], n=n, B=300, N=200, waves=W,
                                      rng=np.random.default_rng(1))
    flat = res.null_construction(cols, [ind], n=n, B=300, N=200, rng=np.random.default_rng(1))
    assert clustered["R_metrics"]["x"] == 0.0 and flat["R_metrics"]["x"] > 0.0
    # the survival floor: against a tiny R a Bernoulli(0.5) survival trips often; floored at R_binom
    # (z_0.99 sqrt(2 p(1-p) / n_c), n_c = the category's episode count) it almost never does
    rng = np.random.default_rng(2)
    surv = np.full(n, np.nan)
    surv[wave == 5] = (rng.random(E) < 0.5).astype(np.float64)
    g = res._spec("cat.fwd.survival", "guard", (), "fwd", ("cat:fwd:survival",), "mean",
                  level=0.99, one_sided=True)
    c2 = {"cat:fwd:survival": surv}
    tiny = {"R_metrics": {"cat.fwd.survival": 1e-6}, "R_blocks": {}}
    raw = res.null_construction(c2, [g], n=n, N=400, waves=W, rng=np.random.default_rng(3), **tiny)
    floored = res.null_construction(c2, [g], n=n, N=400, waves=W, rng=np.random.default_rng(3),
                                    survival_guards=["cat.fwd.survival"], **tiny)
    assert raw["guards"]["cat.fwd.survival"] < 0.8
    assert floored["guards"]["cat.fwd.survival"] >= 0.99
    p = float(np.mean(surv[wave == 5]))
    assert floored["binomial_floor"]["cat.fwd.survival"] == pytest.approx(
        res.binomial_guard_floor(p, E), rel=0.35)
    # a survival guard with no R at all is scored on the floor alone, never dropped
    none_r = res.null_construction(c2, [g], n=n, N=100, waves=W, rng=np.random.default_rng(4),
                                   R_metrics={"cat.fwd.survival": None}, R_blocks={},
                                   survival_guards=["cat.fwd.survival"])
    assert "cat.fwd.survival" in none_r["guards"]


def test_d00s_acceptance_is_the_null_band_and_completeness():
    systems = {"ratio": {"blocks": ["F", "F_mt", "T"], "na_blocks": [],
                         "members": {"F": ["m.f"], "F_mt": ["m.fmt"], "T": ["m.t"]},
                         "guards": ["cat.fwd.survival", "cat.move_turn.ratio", "cat_tripod_min"]},
               "absolute": {"blocks": ["T_abs"], "na_blocks": [], "members": {"T_abs": ["m.ta"]},
                            "guards": ["cat.fwd.survival", "cat.move_turn.achieved_speed"]}}
    good_m = {k: {"R_eval": 0.1} for k in ("m.f", "m.fmt", "m.t", "m.ta", "cat.move_turn.ratio",
                                            "cat_tripod_min", "cat.move_turn.achieved_speed")}
    good_m["cat.fwd.survival"] = {"R_eval": 0.0}           # exempt: scored on the binomial floor
    good_b = {b: {"R_eval": 0.9} for b in ("F", "F_mt", "T", "T_abs")}
    nominal = 0.78
    ok = res.acceptance_d00s(good_b, good_m, systems, {"all_tie": 0.80}, nominal)
    assert ok == {"ok": True, "failures": []}
    assert not res.acceptance_d00s(good_b, good_m, systems, {"all_tie": 0.74}, nominal)["ok"]
    assert not res.acceptance_d00s(good_b, good_m, systems, {"all_tie": 0.99}, nominal)["ok"]
    assert not res.acceptance_d00s(good_b, good_m, systems, None, nominal)["ok"]
    for mutate, needle in (
            (lambda b, m: b.pop("F_mt"), "block F_mt"),                      # F_mt with no R
            (lambda b, m: m["m.f"].update(R_eval=0.0), "F member m.f"),       # a member at 0
            (lambda b, m: m["cat_tripod_min"].update(R_eval=None), "guard cat_tripod_min"),
            (lambda b, m: m.pop("cat.move_turn.achieved_speed"), "cat.move_turn.achieved_speed")):
        b, m = json.loads(json.dumps(good_b)), json.loads(json.dumps(good_m))
        mutate(b, m)
        got = res.acceptance_d00s(b, m, systems, {"all_tie": 0.80}, nominal)
        assert not got["ok"] and any(needle in f for f in got["failures"]), (needle, got)


# ------------------------------------------------------------------------------ legacy bit-identity
def test_legacy_pairs_are_bit_identical_under_the_clustered_path(three_heads):
    a, b, _ = three_heads
    rows = [res.load_row(_entry(r, head=h), ring_proxy=False, rule=RULE)
            for r, h in ((a, "1c-004"), (b, "2b"))]
    assert not rows[0].suite and rows[0].waves == 1
    specs, _ = res.specs_from_rule(RULE, rows[0].hold_labels)
    pr = res.bootstrap_pair(rows[0], rows[1], B=200, rng_seed=7, specs=specs)
    n = len(rows[0].env_index)
    draw = res.pair_rng(7, pr.name).integers(0, n, size=(200, n))     # the pre-suite draw
    for spec in specs:
        want = (res._reduce(spec.reducer, [rows[0].cols[c][draw] for c in spec.needs])
                - res._reduce(spec.reducer, [rows[1].cols[c][draw] for c in spec.needs]))
        np.testing.assert_array_equal(pr.deltas[spec.mid], want, err_msg=spec.mid)
    # the null construction's legacy draw: two plain episode resamples per call, in that order
    rng = np.random.default_rng(5)
    got = res.null_construction(rows[0].cols, specs, n=n, B=100, N=50, rng=rng)
    rng = np.random.default_rng(5)
    ia, ib = rng.integers(0, n, size=(100, n)), rng.integers(0, n, size=(100, n))
    for spec in specs:
        ref = (res._reduce(spec.reducer, [rows[0].cols[c][ia] for c in spec.needs])
               - res._reduce(spec.reducer, [rows[0].cols[c][ib] for c in spec.needs]))
        want = res.half_width(ref, res.quantile_p(spec.level, one_sided=spec.one_sided))
        assert got["R_metrics"][spec.mid] == res._jf(want), spec.mid
    assert got["scored"]["blocks"] == ["F", "T"]
    assert "binomial_floor" not in got, "no guard of a legacy null is floored"


#: The six EVAL_CARRY_DEFAULTS keys added with the motion-quality knobs (all at their OFF value).
MOTION_CARRY_DEFAULTS = ("KRABBY_OMNI_RELAY_HYST", "KRABBY_OMNI_STAND_HOLD", "KRABBY_OMNI_STAND_PARK",
                         "KRABBY_ACTION_LPF_HZ", "KRABBY_ACTION_LPF_OBS", "KRABBY_OMNI_STAND_OBS")
#: ... and the clock-hide knob's (clock-deprecation plan step 2/3, 2026-09-23; OFF value "0").
CLOCK_CARRY_DEFAULTS = ("KRABBY_OMNI_CLOCK_HIDE",)


@pytest.mark.skipif(not (SHIPPED_D00.exists() and CANDIDATES_V1.exists()),
                    reason="the shipped D00 record or its recorded inputs are untracked")
def test_the_shipped_d00_is_reproduced_bit_for_bit_from_the_frozen_v1_spec():
    """The strongest legacy pin: D00 recomputed from the frozen v1 candidates.yaml under today's
    code equals the shipped record in every number (``created`` aside)."""
    cands = res.load_candidates(CANDIDATES_V1)
    if not all(Path(e["run_dir"]).exists() or (res.REPO / e["run_dir"]).exists()
               for e in cands["resolution_inputs"]):
        pytest.skip("the D00 input run dirs are untracked")
    sha = hashlib.sha256(CANDIDATES_V1.read_bytes()).hexdigest()
    out = res.resolve(cands, candidates_sha256=sha)
    shipped = json.loads(SHIPPED_D00.read_text())
    out.pop("created"), shipped.pop("created")
    # 2026-09-22 (motion-quality plan step 2): EVAL_CARRY_DEFAULTS grew six knob-OFF defaults, which
    # the recorded rows (evaluated before the knobs existed) take at their default -- so they appear
    # in each input's ``env_vars_defaulted`` provenance list and nowhere else. No number moves; the
    # names are dropped here so the pin still covers every number and every other field. The
    # clock-hide default (2026-09-23) is the same case.
    # 2026-10-15 (body-command campaign step 2): the frame default KRABBY_OMNI_COMMAND_FRAME=world, the same case.
    added = set(MOTION_CARRY_DEFAULTS) | set(CLOCK_CARRY_DEFAULTS) | {"KRABBY_OMNI_COMMAND_FRAME"}
    for inp in out.get("inputs") or []:
        if "env_vars_defaulted" in inp:
            inp["env_vars_defaulted"] = [k for k in inp["env_vars_defaulted"] if k not in added]
    assert json.loads(json.dumps(out)) == shipped


# ---------------------------------------------------------------------------------- the CLI mirror
def test_main_mirrors_the_in_process_d00s_call(suite_heads, tmp_path, monkeypatch):
    import yaml
    camp = tmp_path / "camp"
    camp.mkdir()
    cands = _v2_cands(suite_heads)
    stamps = {}
    for e in cands["resolution_inputs"]:
        stamps[e["head"]] = e.pop("stamp")          # the yaml never carries the stamps
    (camp / "candidates.yaml").write_text(yaml.safe_dump(cands))
    d00_path = tmp_path / "resolution.json"
    d00_path.write_text(json.dumps(_fake_d00()))
    reevals = [{"head": h, "eval_sha": EVAL_SHA, "stamp": s} for h, s in stamps.items()]
    reevals.append({"head": "2b", "eval_sha": "f" * 64, "stamp": "an-older-eval"})
    before = sorted(p.name for p in camp.iterdir())
    argv = ["--campaign-dir", str(camp), "--eval-sha", EVAL_SHA, "--reeval-stamps",
            json.dumps(reevals), "--supersedes", "D00", "--d00", str(d00_path)]
    assert res.main(argv) == 0
    out_path = camp / f"resolution.{EVAL_SHA[:12]}.json"
    assert sorted(p.name for p in camp.iterdir()) == sorted(before + [out_path.name])
    written = json.loads(out_path.read_text())
    sha = hashlib.sha256((camp / "candidates.yaml").read_bytes()).hexdigest()
    in_process = res.resolve(dict(cands, resolution_inputs=res.apply_reeval_stamps(
        cands["resolution_inputs"], stamps)), eval_sha=EVAL_SHA, decisive_at=ALL7,
        d00=_fake_d00(), supersedes="D00", candidates_sha256=sha)
    written.pop("created"), in_process.pop("created")
    assert written == json.loads(json.dumps(in_process))
    assert written["eval_sha"] == EVAL_SHA and written["supersedes"] == "D00"
    # the object form works too; a missing stamp refuses (rc 1) and writes nothing
    out_path.unlink()
    assert res.main(argv[:4] + ["--reeval-stamps", json.dumps(stamps)]) == 0 and out_path.exists()
    out_path.unlink()
    assert res.main(argv[:4] + ["--reeval-stamps", json.dumps({"1c-004": STAMP})]) == 1
    assert not out_path.exists()
    assert res.main(["--campaign-dir", str(camp)]) == 1, "rule v2 needs --eval-sha or --out"
    # a failing acceptance writes only the rejected diagnostic, never the record
    monkeypatch.setattr(res, "acceptance_d00s",
                        lambda *a, **k: {"ok": False, "failures": ["forced"]})
    assert res.main(argv) == 2
    assert not out_path.exists()
    assert (camp / f"resolution.{EVAL_SHA[:12]}.D00s.rejected.json").exists()
    # resolve() itself never writes
    snapshot = sorted(p.name for p in camp.iterdir())
    res.resolve(dict(cands, resolution_inputs=res.apply_reeval_stamps(
        cands["resolution_inputs"], stamps)), eval_sha=EVAL_SHA, decisive_at=ALL7)
    assert sorted(p.name for p in camp.iterdir()) == snapshot
