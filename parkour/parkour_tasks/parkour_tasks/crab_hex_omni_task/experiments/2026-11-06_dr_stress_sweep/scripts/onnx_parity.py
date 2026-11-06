#!/usr/bin/env python3
"""DR stress sweep, step 4 (CPU, no Kit): the bundled ONNX student computes what the evaluated ``.pt`` computes.

The sweep evaluates the student checkpoint (``model_28000.pt``) through ``OmniStudentStep`` -- the harness cannot load
ONNX -- so this check ties the deployable export (``policy/f19_9_student/{depth_latest,policy}.onnx``) to it:

1. rebuild the student's three networks on CPU from the checkpoint's own ``params/agent.yaml`` (depth actor from the
   policy class, estimator, ``RecurrentMultiDepthBackbone``) and load the checkpoint's state dicts into them;
2. replay env 0's recorded policy observations + four depth images (``stress_smoke.py --group G0 --record``, a real
   Student-Play rollout) through (a) ``OmniStudentStep.act`` with the recorded step counters (encode every 5 steps,
   latent held in between) and (b) the two ONNX graphs with the runtime contract of ``omni_student_io.json`` (the GRU
   state fed back at each encode, the latent held in between);
3. compare the actions step by step (and the latent at each encode).

ONNX runs on onnxruntime when installed, else on the ONNX reference evaluator (exact float32 numpy). Writes a JSON
record ending in ``"complete": true``; exit 0 when written, the verdict is ``"parity_ok"``.

    onnx_parity.py --record G0_record.npz --out parity.json [--checkpoint PT] [--onnx-dir DIR] [--tol 1e-4]
"""
from __future__ import annotations

import argparse
import copy
import importlib
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

R = Path(__file__).resolve().parents[7]
MODULES = R / "parkour" / "scripts" / "rsl_rl" / "modules"
MAIN = Path("/home/nickmagus/krabby/krabby-research")
CKPT = MAIN / "parkour/logs/rsl_rl/crab_hex_omni_student/2026-10-22_23-41-17_T4F19_st_a2/model_28000.pt"
ONNX_DIR = R / "parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task/policy/f19_9_student"
PKG = "_drs_rsl_modules"
if str(R / "parkour") not in sys.path:
    sys.path.insert(0, str(R / "parkour"))


def _load_pkg(name: str, directory: Path):
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, directory / "__init__.py",
                                                      submodule_search_locations=[str(directory)])
        pkg = importlib.util.module_from_spec(spec)
        sys.modules[name] = pkg
        spec.loader.exec_module(pkg)
    return sys.modules[name]


def onnx_runner(path: Path):
    try:
        import onnxruntime as ort  # noqa: PLC0415
        sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        return (lambda feeds: sess.run(None, feeds)), "onnxruntime"
    except ImportError:
        import onnx  # noqa: PLC0415
        from onnx.reference import ReferenceEvaluator  # noqa: PLC0415
        ev = ReferenceEvaluator(onnx.load(str(path)))
        return (lambda feeds: ev.run(None, feeds)), "onnx.reference"


def build_step(ckpt: Path):
    _load_pkg(PKG, MODULES)
    fe = importlib.import_module(f"{PKG}.feature_extractors")
    acm = importlib.import_module(f"{PKG}.crab_actor_critic_with_encoder")
    step_mod = importlib.import_module(f"{PKG}.omni_student_step")
    cfg = yaml.load((ckpt.parent / "params" / "agent.yaml").read_text(), Loader=yaml.FullLoader)
    pol = copy.deepcopy(cfg["policy"])
    pol.pop("class_name")
    for k in ("state_dependent_std", "actor_obs_normalization", "critic_obs_normalization"):
        pol.pop(k, None)
    est, de = cfg["estimator"], cfg["depth_encoder"]
    sd = torch.load(str(ckpt), map_location="cpu", weights_only=False)
    policy = acm.CrabHexActorCriticRMA(step_mod.OBS_WIDTH, 18, **pol)
    depth_actor = copy.deepcopy(policy.actor)
    depth_actor.load_state_dict(sd["depth_actor_state_dict"])
    estimator = fe.DefaultEstimator(num_prop=int(est["num_prop"]), num_priv_explicit=int(est["num_priv_explicit"]),
                                    hidden_dims=list(est["hidden_dims"]))
    estimator.load_state_dict(sd["estimator_state_dict"])
    encoder = fe.RecurrentMultiDepthBackbone(
        fe.DepthOnlyFCBackbone58x87(int(de["latent_dim"]), int(de["hidden_dims"])),
        dict(num_prop=int(de["num_prop"]), num_cameras=int(de["num_cameras"]), latent_dim=int(de["latent_dim"]),
             grad_checkpoint=bool(de.get("grad_checkpoint", True))))
    encoder.load_state_dict(sd["depth_encoder_state_dict"])
    for m in (depth_actor, estimator, encoder):
        m.eval()
    step = step_mod.OmniStudentStep(encoder, depth_actor, estimator, num_prop=int(est["num_prop"]),
                                    num_scan=int(est["num_scan"]), num_priv_explicit=int(est["num_priv_explicit"]),
                                    depth_num_prop=int(de["num_prop"]))
    return step, step_mod, int(de["num_prop"]), int(sd.get("iter", -1))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--checkpoint", default=str(CKPT))
    ap.add_argument("--onnx-dir", default=str(ONNX_DIR))
    ap.add_argument("--tol", type=float, default=1e-4)
    a = ap.parse_args()
    t0 = time.time()
    rec = np.load(a.record)
    obs, depth, counter, dones = rec["obs"], rec["depth"], rec["counter"], rec["dones"]
    step, step_mod, depth_prop, it = build_step(Path(a.checkpoint))
    run_e, label = onnx_runner(Path(a.onnx_dir) / "depth_latest.onnx")
    run_p, _ = onnx_runner(Path(a.onnx_dir) / "policy.onnx")
    io = json.loads((Path(a.onnx_dir) / "omni_student_io.json").read_text())
    period = int(io["encode_every_policy_steps"])
    h = np.zeros(tuple(io["gru_hidden"]), dtype=np.float32)
    latent = None
    step.reset()
    act_err, raw_err, lat_err, n, encodes = 0.0, 0.0, 0.0, 0, 0
    for t in range(len(obs)):
        if t > 0 and bool(dones[t - 1]):
            break  # an episode boundary: the parity read stops at the first reset of env 0
        o = torch.as_tensor(obs[t][None], dtype=torch.float32)
        d = {g: torch.as_tensor(depth[t][i][None], dtype=torch.float32) for i, g in enumerate(step_mod.DEPTH_GROUPS)}
        with torch.inference_mode():
            a_t = step.act(o, d, int(counter[t])).numpy()
        if latent is None or int(counter[t]) % period == 0:
            lat, h = run_e({"depth_image": depth[t][None].astype(np.float32),
                            "proprioception": obs[t][None, :depth_prop].astype(np.float32), "h_in": h})
            latent = lat
            encodes += 1
            lat_err = max(lat_err, float(np.abs(latent - step.depth_latent.numpy()).max()))
        (a_o,) = run_p({"obs": obs[t][None].astype(np.float32), "depth_latent": latent.astype(np.float32)})
        # the deploy contract clips raw actions to +-1 before the action term, on both paths
        act_err = max(act_err, float(np.abs(np.clip(a_o, -1, 1) - np.clip(a_t, -1, 1)).max()))
        raw_err = max(raw_err, float(np.abs(a_o - a_t).max()))
        n += 1
    res = {"tool": "onnx_parity.py", "checkpoint": a.checkpoint, "checkpoint_iter": it, "onnx_dir": a.onnx_dir,
           "runtime": label, "steps_compared": n, "encodes": encodes, "max_abs_action_err": act_err,
           "max_abs_raw_action_err": raw_err,
           "max_abs_latent_err": lat_err, "tol": a.tol,
           "parity_ok": bool(n >= 20 and encodes >= 4 and act_err <= a.tol and lat_err <= a.tol),
           "wall_s": time.time() - t0, "complete": True}
    Path(a.out).write_text(json.dumps(res, indent=1, sort_keys=True))
    print(f"ONNX_PARITY END ok={res['parity_ok']} steps={n} encodes={encodes} act_err={act_err:.3g} "
          f"lat_err={lat_err:.3g} ({label})", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
