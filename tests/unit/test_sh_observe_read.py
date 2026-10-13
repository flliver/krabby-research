"""Stand-height campaign (``2026-10-13_stand_height``): the observe-pass wrapper ``scripts/sh_observe_read.py`` (PLAN H7
"observe pass", review fix 2026-10-13) on synthetic stand_height stamps (CPU, no Isaac).

Pinned: the residual frame (applied = clip(a_net), R = R_h - q) reproduces the predecessor's read bit for bit at
1.064 and keeps its self-checks at other heights; stand rows = p2r_stand_mode (who acted); validity fails closed
(variant, network controller, command per row, routing / execution record, one checkpoint per head); PENDING rows;
the CLI writes only its two paths; no bytecode.
"""
from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

REPO = Path(__file__).resolve().parents[2]
TASK = REPO / "parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task"
SCRIPTS = TASK / "experiments/2026-10-13_stand_height/scripts"
PRED = TASK / "experiments/2026-10-11_stand_network/scripts"
LEGS = ("FL", "FR", "ML", "MR", "RL", "RR")
JOINTS = ([f"{l}_Body_Hip_RevoluteJoint" for l in LEGS] + [f"{l}_Hip_Femur_RevoluteJoint" for l in LEGS]
          + [f"{l}_Body_CamShaft_RevoluteJoint" for l in LEGS] + [f"{l}_Femur_Tibia_RevoluteJoint" for l in LEGS])
CAM_J = [JOINTS.index(f"{l}_Body_CamShaft_RevoluteJoint") for l in LEGS]
DT = 0.02
TS = "2026-10-13_12-00-00"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    old = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.dont_write_bytecode = old
    return mod


def _pyc():
    return {str(d): (sorted(p.name for p in (d / "__pycache__").iterdir()) if (d / "__pycache__").is_dir() else None)
            for d in (SCRIPTS, PRED)}


_PYC_BEFORE = _pyc()


@pytest.fixture(scope="module")
def O():
    return _load(SCRIPTS / "sh_observe_read.py", "_t_sh_observe_read")


def episode(O, *, h_stand=(0.90, 1.064), seed=0, T=400):
    """One synthetic stand_height episode: two stand runs (the first at ``h_stand[0]``, the second at
    ``h_stand[1]``), R_h's HOLD cam law and LIFT hip (= q + lift) consistent with the recorded cams, a stand net that
    tracks R_h's residual with noise (some outputs beyond +-1), S1 rows elsewhere."""
    rng = np.random.default_rng(seed)
    SN, EC = O.SN, O.EC
    C = EC.CORE
    rst, p = SN.load_rst(), SN.r_params()
    stand = np.zeros(T, bool)
    runs = [(50, 200, h_stand[0]), (250, 380, h_stand[1])]
    h = np.full(T, C.H_SETTLED)
    for a, b, hh in runs:
        stand[a:b] = True
        h[a:b] = hh
    q = C.q18(torch.as_tensor(h, dtype=torch.float64), dtype=torch.float32).numpy()
    q = np.where(stand[:, None], q, 0.0).astype(np.float32)
    c = rng.uniform(-0.25, 0.25, size=(T, 6))
    br = np.full((T, 6), rst.BRANCH_HOLD, dtype=np.int8)
    for a, b, _ in runs:                                                 # each run: a LIFT burst on legs 0 / 3
        br[a + 20:a + 35, [0, 3]] = rst.BRANCH_LIFT
        br[a + 60:a + 70, [1, 4]] = rst.BRANCH_LIFT
    br = np.where(stand[:, None], br, -1).astype(np.int8)
    tgt = q.astype(np.float64).copy()
    hold, lift = br == rst.BRANCH_HOLD, br == rst.BRANCH_LIFT
    tgt[:, 0:6] = np.where(hold, -np.clip(p.k_h * c, -p.trim, p.trim), 0.0)
    tgt[:, 6:12] = np.where(lift, q[:, 6:12].astype(np.float64) + p.lift, q[:, 6:12])
    tgt = np.where(stand[:, None], tgt, 0.0).astype(np.float32)
    a_net = ((tgt - q) + rng.normal(0.0, 0.2, size=(T, 18))).astype(np.float32)
    a_net[stand & (np.arange(T) % 37 == 0), 7] = 1.6                     # beyond +-1: the clip is the applied action
    a_net = np.where(stand[:, None], a_net, 0.0).astype(np.float32)
    s1_raw = rng.normal(0.0, 1.0, size=(T, 18)).astype(np.float32)
    actions = np.where(stand[:, None], np.where(q != 0, a_net + q, a_net), s1_raw).astype(np.float32)
    ex = q.astype(np.float64) + np.clip(a_net.astype(np.float64), -1, 1)
    ex[:, 0:6] = np.clip(ex[:, 0:6], -1, 1)
    ex[:, 6:] = np.clip(ex[:, 6:], -4.8, 4.8)
    ex = np.where(stand[:, None], ex, 0.0).astype(np.float32)
    jp = np.zeros((T, 24), np.float32)
    jp[:, CAM_J] = c
    return {EC.RAW_STAND_MODE: stand, "actions": actions, "joint_pos": jp, EC.RAW_A_NET: a_net, EC.RAW_Q: q,
            EC.RAW_RH_TARGET: tgt, EC.RAW_RH_BRANCH: br, EC.RAW_STAND_ACTION: ex,
            EC.RAW_S1_ACTION: np.where(stand[:, None], np.clip(s1_raw, -1, 1), 0.0).astype(np.float32)}


def meta(O, command="manifest", *, sha="a" * 64, control="stand_net", variant=None):
    SN, EC = O.SN, O.EC
    return {"p2r_mode": {"variant": variant or EC.VARIANT, EC.MASK_KEY: 1,
                         "specialist": {"path": "/synthetic.pt", "sha256": sha}, "s1_checkpoint_sha256": "b" * 64,
                         EC.BLOCK_KEY: {"control": control, "r_knob": SN.RESTANCE_D3, "command": command,
                                        "stand_ckpt_kind": "sh_dagger"},
                         "raw_series": [EC.RAW_STAND_MODE, *EC.RAW_KEYS]},
            "env_vars": {SN.STAND_OBS_VAR: "0", **SN.ENV4}, "joint_names": JOINTS, "dt": DT}


def write(lr: Path, head: str, row: str, seed: int, eps: list, m: dict, *, complete=True) -> Path:
    d = lr / head / f"{row}__omni" / f"seed{seed:03d}" / TS
    (d / "raw").mkdir(parents=True, exist_ok=True)
    (d / "run_meta.json").write_text(json.dumps(m))
    if complete:
        (d / "p2_complete.json").write_text("{}")
    for i, ep in enumerate(eps):
        np.savez_compressed(d / "raw" / f"episode_{i:02d}.npz", **ep)
    return d


def head(O, lr: Path, name: str = "HEAD", **kw):
    for row, cmd, hs in (("walkstop", "zero", (1.064, 1.064)), ("hwalkstop", "manifest", (0.90, 1.25))):
        write(lr, name, row, 1, [episode(O, h_stand=hs, seed=s) for s in (0, 1)], meta(O, cmd, **kw))


def test_a_clean_head_reads_in_the_residual_frame(O, tmp_path):
    lr = tmp_path / "lr"
    head(O, lr)
    out = O.observe("HEAD", ["walkstop@1", "hwalkstop@1"], lr)
    assert out["status"] == "complete" and not out["invalid"] and not out["pending"], out["invalid"]
    assert set(out["pooled"]) == {"zero", "height", "all"}
    for k in ("walkstop@1", "hwalkstop@1"):
        r = out["rows"][k]
        assert r["status"] == O.READ and r["counts"]["stands"] == 4 and r["counts"]["stand_rows"] == 2 * 280
        ck = r["checks"]
        assert ck["routing_max_abs"] == 0.0 and ck["branch_rows_ok"] is True
        assert ck["c_align_max_abs"] <= O.OBS.ALIGN_TOL and ck["lift_target_max_abs"] <= 1e-6
        lift = r["lift_hip"]["lift"]
        assert lift["leg_rows"] == 4 * (2 * 15 + 2 * 10) and -0.75 < lift["mu_hip"] < -0.25   # ~ R's lift -0.5
        hold = r["by_branch"]["hold"]["hip"]
        assert hold["mean_abs"] < 0.25                                     # |clip(a_net) - (R_h - q)| ~ the noise
    # the height row's stand rows sit at q != 0: the ABSOLUTE frame would read |q| on every HOLD hip
    assert out["rows"]["hwalkstop@1"]["by_branch"]["hold"]["hip"]["mean_abs"] < 0.25
    md = O.markdown(out)
    assert md.startswith("# Stand-height observe pass — `HEAD`") and "RESIDUAL FRAME" in md


def test_at_walking_height_it_is_the_predecessors_read_bit_for_bit(O):
    """q = 0 everywhere (a zero-command stamp): the residual frame IS the predecessor's frame."""
    OBS, SN = O.OBS, O.SN
    rst, p = SN.load_rst(), SN.r_params()
    ep = episode(O, h_stand=(1.064, 1.064), seed=3)
    dd, why = O.residual_episode(ep)
    assert why == []
    stand = ep[O.EC.RAW_STAND_MODE]
    pred = {O.RAW_STAND: stand, SN.RAW_STAND_ACTION: ep[O.EC.RAW_A_NET], SN.RAW_R_TARGET: ep[O.EC.RAW_RH_TARGET],
            SN.RAW_R_BRANCH: ep[O.EC.RAW_RH_BRANCH], SN.RAW_S1_ACTION: ep[O.EC.RAW_S1_ACTION],
            "actions": ep["actions"], "joint_pos": ep["joint_pos"]}
    a, b = OBS.Acc(), OBS.Acc()
    w = int(round(OBS.LAUNCH_WINDOW_S / DT))
    a.add_episode(dd, CAM_J, w, p, rst.BRANCH_LIFT, rst.BRANCH_HOLD)
    b.add_episode(pred, CAM_J, w, p, rst.BRANCH_LIFT, rst.BRANCH_HOLD)
    assert json.dumps(OBS.summarize(a, DT, p), sort_keys=True) == json.dumps(OBS.summarize(b, DT, p), sort_keys=True)


@pytest.mark.parametrize("change, needle", [
    (dict(variant="stand_net"), "variant"),
    (dict(control="r_h"), "network controller"),
    (dict(command="zero"), "command"),
])
def test_meta_validity_fails_closed(O, tmp_path, change, needle):
    lr = tmp_path / "lr"
    write(lr, "H", "hwalkstop", 1, [episode(O)], meta(O, **{"command": "manifest", **change}))
    out = O.observe("H", ["hwalkstop@1"], lr)
    assert out["rows"]["hwalkstop@1"]["status"] == O.INVALID and out["status"] == "refused"
    assert any(needle in x for x in out["invalid"]["hwalkstop@1"]), out["invalid"]


def test_routing_and_execution_records_fail_closed(O, tmp_path):
    lr = tmp_path / "lr"
    ep = episode(O)
    stand_row = int(np.flatnonzero(ep[O.EC.RAW_STAND_MODE])[5])
    bad = dict(ep, actions=ep["actions"].copy())
    bad["actions"][stand_row, 8] += 0.5                                  # the harness record is not the network's
    write(lr, "H", "hwalkstop", 1, [bad], meta(O))
    bad2 = dict(ep, **{O.EC.RAW_STAND_ACTION: ep[O.EC.RAW_STAND_ACTION].copy()})
    bad2[O.EC.RAW_STAND_ACTION][stand_row, 8] += 0.5                     # the executed action is not q + clip(a_net)
    write(lr, "H2", "hwalkstop", 1, [bad2], meta(O))
    r1 = O.observe("H", ["hwalkstop@1"], lr)["invalid"]["hwalkstop@1"]
    r2 = O.observe("H2", ["hwalkstop@1"], lr)["invalid"]["hwalkstop@1"]
    assert any("recorded actions differ" in x for x in r1) and any(O.EC.RAW_STAND_ACTION in x for x in r2)
    ep3 = {k: v for k, v in ep.items() if k != O.EC.RAW_RH_BRANCH}
    write(lr, "H3", "hwalkstop", 1, [ep3], meta(O))
    assert any("lacks raw key" in x for x in O.observe("H3", ["hwalkstop@1"], lr)["invalid"]["hwalkstop@1"])


def test_one_checkpoint_per_head_and_pending_rows(O, tmp_path):
    lr = tmp_path / "lr"
    write(lr, "H", "walkstop", 1, [episode(O, h_stand=(1.064, 1.064))], meta(O, "zero", sha="a" * 64))
    write(lr, "H", "hwalkstop", 1, [episode(O)], meta(O, "manifest", sha="c" * 64))
    out = O.observe("H", ["walkstop@1", "hwalkstop@1"], lr)
    assert all(any("mix specialist_sha256" in x for x in v) for v in out["invalid"].values())
    write(lr, "P", "hraise", 1, [episode(O)], meta(O), complete=False)
    out = O.observe("P", ["hraise@1", "suite@2"], lr)
    assert out["status"] == "partial" and len(out["pending"]) == 2 and "in progress" in out["pending"][0]
    with pytest.raises(ValueError):
        O.parse_row("field_hard@1")
    assert O.parse_row("hraise__omni@2") == ("hraise", 2)


def test_cli_writes_only_its_two_paths_and_exit_codes(O, tmp_path, capsys):
    lr = tmp_path / "lr"
    head(O, lr)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    before = sorted(p for p in tmp_path.rglob("*"))
    rc = O.main(["--head", "HEAD", "--rows", "walkstop@1", "hwalkstop@1", "--lr", str(lr), "--json",
                 str(out_dir / "o.json"), "--md", str(out_dir / "o.md")])
    assert rc == O.RC_OK
    after = sorted(p for p in tmp_path.rglob("*"))
    assert set(after) - set(before) == {out_dir / "o.json", out_dir / "o.md"}
    js = json.loads((out_dir / "o.json").read_text())
    assert js["tool_sha256"] and js["sn_observe_read_sha256"] and js["frame"].startswith("residual")
    rc = O.main(["--head", "HEAD", "--rows", "hraise@1", "--lr", str(lr), "--json", str(out_dir / "p.json"), "--md",
                 str(out_dir / "p.md")])
    assert rc == O.RC_PENDING
    assert O.main(["--head", "HEAD", "--rows", "bogus@1", "--json", str(out_dir / "x.json"), "--md",
                   str(out_dir / "x.md")]) == O.RC_USAGE
    capsys.readouterr()


def test_no_bytecode_next_to_campaign_or_predecessor_files(O):
    assert _pyc() == _PYC_BEFORE
    assert math.isfinite(O.EXEC_TOL)
