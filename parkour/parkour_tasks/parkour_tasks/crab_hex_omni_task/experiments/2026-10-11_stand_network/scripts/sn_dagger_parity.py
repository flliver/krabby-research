#!/usr/bin/env python3
"""Stand-network campaign, PLAN.md step 6 -- the OFFLINE leg of the D1 forward-path parity (CPU only, no Kit, no GPU).

PLAN step 6: "a trainer-captured obs row fed offline to the stand_net launcher's policy object returns the trainer's
action bitwise (pins the D1 forward path)". A DAgger smoke writes ``<run>/smoke_capture.pt`` (a full rollout batch of
raw obs with stand rows, the stand mask, the trainer's routed actions computed on the GPU, the saved checkpoint's path
and sha). This tool, on CPU:

1. builds S1 (``model_19994.pt``, sha pinned) and the stand net from the checkpoint through the launcher's load path
   (``sn_dagger_core.load_strict``: ``torch.load`` without map_location, strict loads);
2. (a) the TRAINER's path on CPU: ``sn_dagger_core.trainer_actions`` (``sn_common.stand_net_input`` + ``act_inference
   (hist_encoding=True)`` + ``route_actions``);
   (b) the launcher's REAL policy object: ``StandNetMode`` from ``sn_stand_net_eval_crab_hex_gait.py`` (a private P2L
   instance; a minimal stub command term / env whose stand mask is the captured one; ``bind_s1`` runs the estimator
   identity check and the trained-arm check -- the checkpoint's ``infos.sn_dagger.stand_mask`` must be the captured
   mask), fed the harness's obs (S1's estimate written in place, exactly as H:1027-1029);
3. PASS iff (a) == (b) bitwise (``torch.equal``, same device / batch) AND (a) is within the CHARTER's GPU atol 1e-5 of
   the GPU-captured trainer actions (cross-device: bitwise is not expected).

    sn_dagger_parity.py <run>/smoke_capture.pt [--ckpt PATH] [--json OUT]       exit 0 PASS, 1 FAIL, 2 refused
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
GPU_ATOL = 1e-5


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    old = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.dont_write_bytecode = old
    return mod


def rsl_classes(repo: Path):
    """``CrabHexActorCriticRMA`` / ``DefaultEstimator`` from ``parkour/scripts/rsl_rl/modules`` as a private package
    (the CPU tests' loader; no Isaac import)."""
    import importlib  # noqa: PLC0415

    name = "_sn_parity_rslmods"
    if name not in sys.modules:
        pkg = types.ModuleType(name)
        pkg.__path__ = [str(repo / "parkour/scripts/rsl_rl/modules")]
        sys.modules[name] = pkg
    crab = importlib.import_module(f"{name}.crab_actor_critic_with_encoder")
    est = importlib.import_module(f"{name}.feature_extractors.estimator")
    return crab.CrabHexActorCriticRMA, est.DefaultEstimator


class _Cmd:
    """The few command-term attributes ``StandNetMode`` touches (its stand latch = the captured stand mask)."""

    def __init__(self, stand):
        import torch  # noqa: PLC0415

        n = int(stand.shape[0])
        self._stand = stand.clone()
        self.slot_id = torch.where(stand, 0, 1).long()
        self.speed_cmd = torch.zeros(n)
        self.vel_command_b = torch.zeros(n, 3)
        self._hook_turning = torch.zeros(n, dtype=torch.bool)
        self._slot_latch = True

    def stand_mask(self):
        return self._stand


def launcher_mode_actions(obs_raw, stand, s1, s1_est, st_pol, st_est, *, p0, p1, num_prop, mask: int, ckpt,
                          s1_sha=None):
    """The launcher's ``StandNetMode`` (real class, private P2L instance) on one step of ``obs_raw``. ``ckpt`` = the
    stand checkpoint file (its ``infos.sn_dagger.stand_mask`` is checked against ``mask`` by ``bind_s1``, exactly
    as in the eval); ``s1_sha`` = S1's sha256 (only used for a checkpoint without ``infos.sn_dagger``)."""
    import numpy as np  # noqa: PLC0415
    import torch  # noqa: PLC0415

    snl = _load(HERE / "sn_stand_net_eval_crab_hex_gait.py", "_sn_parity_launcher")
    p2r = snl.load_p2r()
    cls = snl.make_mode_class(p2r, int(mask), str(ckpt))
    n = int(obs_raw.shape[0])
    cmd = _Cmd(stand)
    unwrapped = types.SimpleNamespace(num_envs=n, episode_length_buf=torch.ones(n, dtype=torch.long))
    m = cls("st2", cmd, unwrapped, priv=(p0, p1), num_prop=num_prop, stand=(st_pol.act_inference, st_est))
    m.s1_runner = types.SimpleNamespace(alg=types.SimpleNamespace(estimator=s1_est))
    m.sn_ctx = types.SimpleNamespace(s1_sha=s1_sha, specialist={"sha256": snl.SN.sha256_file(ckpt)})
    m.bind_s1(s1.act_inference, s1)
    m.begin_pass(types.SimpleNamespace(segment_id=np.zeros((1, n), np.int16), hold_labels=["stand"], n_steps=1))
    obs = obs_raw.clone()
    m.observe(obs)
    with torch.inference_mode():
        obs[:, p0:p1] = s1_est.inference(obs[:, :num_prop])           # the harness's in-place write (H:1027-1029)
        return m(obs, hist_encoding=True)


def replay(capture_path, ckpt=None) -> dict:
    import torch  # noqa: PLC0415
    import yaml  # noqa: PLC0415

    core = _load(HERE / "sn_dagger_core.py", "_sn_parity_core")
    spec = core.SPEC
    cap = torch.load(str(capture_path), map_location="cpu", weights_only=False)
    ckpt = str(ckpt or cap["ckpt"])
    if core.sha256_file(ckpt) != cap["ckpt_sha256"]:
        return {"verdict": "REFUSED", "why": f"{ckpt} is not the captured checkpoint (sha differs)"}
    if core.sha256_file(spec.S1_CKPT) != spec.S1_SHA256:
        return {"verdict": "REFUSED", "why": "S1's sha256 is not the pinned one"}
    policy_cls, est_cls = rsl_classes(spec.REPO)
    cfg = yaml.safe_load(spec.S1_AGENT_YAML.read_text())
    nets = core.build_networks(policy_cls, est_cls, cfg, spec.S1_CKPT, "cpu")
    pol, est, _ = core.load_strict(ckpt, policy_cls, est_cls, cfg, "cpu")
    nets.student = pol
    obs, stand, mask = cap["obs"].float(), cap["stand"].bool(), int(cap["stand_mask"])
    with torch.inference_mode():
        a_tr = core.trainer_actions(obs, stand, nets, mask)
    a_mode = launcher_mode_actions(obs, stand, nets.s1, nets.est, pol, est, p0=nets.p0, p1=nets.p1,
                                   num_prop=nets.num_prop, mask=mask, ckpt=ckpt, s1_sha=nets.s1_sha)
    gpu = cap["actions_trainer"].float()
    bitwise = bool(torch.equal(a_tr, a_mode))
    d_gpu = float((a_tr - gpu).abs().max())
    ok = bitwise and d_gpu <= GPU_ATOL
    return {"verdict": "PASS" if ok else "FAIL", "cpu_trainer_vs_standnetmode_bitwise": bitwise,
            "cpu_trainer_vs_standnetmode_max_abs": float((a_tr - a_mode).abs().max()),
            "cpu_vs_captured_max_abs": d_gpu, "gpu_atol": GPU_ATOL, "rows": int(obs.shape[0]),
            "stand_rows": int(stand.sum()), "stand_mask": mask, "ckpt": ckpt, "capture": str(capture_path),
            "captured_on": cap.get("device")}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("capture")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--json", default=None)
    a = ap.parse_args(argv)
    res = replay(a.capture, a.ckpt)
    txt = json.dumps(res, indent=1, sort_keys=True)
    print(txt)
    if a.json:
        Path(a.json).write_text(txt)
    return {"PASS": 0, "FAIL": 1}.get(res["verdict"], 2)


if __name__ == "__main__":
    sys.exit(main())
