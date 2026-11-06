"""DR stress sweep, step 4 (GPU, Kit): the DOSE SMOKE -- every stress knob's REALIZED dose, read from the sim.

Builds ``Isaac-Crab-Hex-Omni-Student-Play-v0`` headless with the process env the launcher passes (``smoke.sh``: the
F19 ``p3b.env`` keys plus one knob group), drives the robot with a deterministic action pattern, and checks what the
sim actually does, not what the cfg says:

``--group G0`` (no knob)   every stress hook is ABSENT (obs term ``_stress_obs`` None, action scales None, depth lag 1,
                           no ``stress_*`` event); records env 0's policy obs + 4 depth images per step for the CPU
                           ONNX parity check (``onnx_parity.py``)
``--group G1`` (plant)     rod slew ceilings (hip / knee rod speed), cam acceleration slope and speed cap, cam
                           effort / saturation, PhysX max-force / stiffness / damping of the rod joints, leg-link
                           masses and inertias, robot shape friction, the ground material; vxy slot zeroed
``--group G2`` (sensor)    action delay (raw action lag), IMU / encoder noise bounds and spread, current-proxy gain and
                           noise against the clean proxy recomputed at the same state, vxy noise spread, depth frame
                           index
``--group G3`` (push)      the additive push (the action term's ``stress_push`` hook) on a STANDING robot (zero
                           actions: a reset restarts the per-env timer): count, first-push time and |dv|

Each check is ``{"want", "got", "tol", "ok"}``; the JSON ends with ``"all_ok"`` and ``"complete": true`` (written last).

    <isaac python> stress_smoke.py --headless --group G0|G1|G2|G3 --out JSON [--record NPZ] [--num-envs 4]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="DR stress sweep dose smoke.")
parser.add_argument("--group", choices=("G0", "G1", "G2", "G3"), required=True)
parser.add_argument("--out", required=True)
parser.add_argument("--record", default=None, help="G0 only: npz of env 0's obs + depth images per step")
parser.add_argument("--num-envs", type=int, default=4)
parser.add_argument("--steps", type=int, default=560)
parser.add_argument("--seed", type=int, default=7)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

R = Path(__file__).resolve().parents[7]
PT = R / "parkour" / "parkour_tasks"
if str(PT) not in sys.path:
    sys.path.insert(0, str(PT))

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import parkour_tasks  # noqa: E402,F401
from isaaclab.utils.math import euler_xyz_from_quat, wrap_to_pi  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from parkour_tasks.crab_hex_forward_task.mdp import crab_hex_linkage as linkage  # noqa: E402
from parkour_tasks.crab_hex_forward_task.mdp.stress_obs import OBS_NOISE_REF  # noqa: E402

TASK = "Isaac-Crab-Hex-Omni-Student-Play-v0"
DEPTH_GROUPS = ("depth_front", "depth_left", "depth_right", "depth_rear")
# proprio layout (forward ``CrabHexParkourObservations``; the omni head adds no channel with STAND_OBS / LPF_OBS off)
ANG, RP, VXY, JP, JV, ACT, CUR = slice(0, 3), slice(3, 5), slice(13, 15), slice(17, 41), slice(41, 65), \
    slice(65, 83), slice(83, 89)


def knob(name, cast=float):
    v = os.environ.get(f"KRABBY_STRESS_{name}")
    return None if v is None or not v.strip() else cast(v)


class Checks:
    def __init__(self):
        self.rows = {}

    def close(self, name, got, want, tol, rel=False):
        got, want = float(got), float(want)
        err = abs(got - want) / max(abs(want), 1e-12) if rel else abs(got - want)
        self.rows[name] = {"want": want, "got": got, "tol": tol, "rel": rel, "ok": bool(err <= tol)}

    def true(self, name, cond, detail=None):
        self.rows[name] = {"want": True, "got": bool(cond), "detail": detail, "ok": bool(cond)}

    @property
    def ok(self):
        return all(r["ok"] for r in self.rows.values())


def obs_term(u, group="policy"):
    return u.observation_manager._group_obs_term_cfgs[group][0].func


def action_pattern(step: int, n: int, n_act: int, term, device) -> torch.Tensor:
    """Pitch / knee channels swing at +-1 with a 2 s period (rods driven, the proxy live); cams at a constant +1."""
    a = torch.zeros(n, n_act, device=device)
    s = 1.0 if (step // 50) % 2 == 0 else -1.0
    a[:, term._pitch_action_cols] = s
    a[:, term._knee_action_cols] = -s
    a[:, term._cam_action_cols] = 1.0
    return a


def main() -> int:
    t0 = time.time()
    env_cfg = parse_env_cfg(TASK, device=args.device, num_envs=args.num_envs)
    env_cfg.seed = args.seed
    env = gym.make(TASK, cfg=env_cfg)
    u = env.unwrapped
    obs, _ = env.reset()
    robot = u.scene["robot"]
    term = u.action_manager.get_term("joint_pos")
    pterm = obs_term(u)
    n, dev = u.num_envs, u.device
    n_act = term.action_dim
    c = Checks()
    res = {"tool": "stress_smoke.py", "group": args.group, "task": TASK, "num_envs": n, "seed": args.seed,
           "parkour_tasks_file": parkour_tasks.__file__,
           "krabby_env": {k: v for k, v in sorted(os.environ.items()) if k.startswith("KRABBY_")}}
    c.true("imports this checkout", str(Path(parkour_tasks.__file__).resolve()).startswith(str(R)), parkour_tasks.__file__)
    view = robot.root_physx_view
    pitch_ids, _ = robot.find_joints(".*_Hip_Femur_RevoluteJoint", preserve_order=True)
    knee_ids, _ = robot.find_joints(".*_Femur_Tibia_RevoluteJoint", preserve_order=True)

    if args.group == "G0":
        c.true("obs stress absent", getattr(pterm, "_stress_obs", "missing") is None)
        c.true("rod scale absent", term._rod_speed_scale is None)
        c.true("cam scale absent", term._cam_scale is None and term._cam_speed_cap is None)
        c.true("push hook absent", getattr(term, "_stress_push", "missing") is None)
        for g in DEPTH_GROUPS:
            d = obs_term(u, g)
            c.true(f"{g} lag 1 / buffer 2", d.lag == 1 and d.buffer_len == 2, (d.lag, d.buffer_len))
        names = list(u.event_manager.active_terms.values()) if isinstance(u.event_manager.active_terms, dict) else []
        flat = [x for v in names for x in v]
        c.true("no stress events", not any(x.startswith("stress_") for x in flat), flat)
        c.true("no stress_obs attr", getattr(u.cfg, "stress_obs", None) is None)
        rec = {"obs": [], "depth": [], "counter": [], "dones": []}
        for t in range(120):
            a = action_pattern(t, n, n_act, term, dev)
            obs, _, term_, trunc, _ = env.step(a)
            rec["obs"].append(obs["policy"][0].detach().cpu().numpy().copy())
            rec["depth"].append(np.stack([obs[g][0].detach().cpu().numpy() for g in DEPTH_GROUPS]))
            rec["counter"].append(int(u.common_step_counter))
            rec["dones"].append(bool((term_ | trunc)[0]))
        if args.record:
            np.savez_compressed(args.record, obs=np.stack(rec["obs"]), depth=np.stack(rec["depth"]),
                                counter=np.array(rec["counter"]), dones=np.array(rec["dones"]))
            res["record"] = args.record

    elif args.group == "G1":
        rod = knob("ROD_SPEED")
        cam = knob("CAM")
        eff = knob("ROD_EFFORT")
        kpkd = knob("KPKD")
        leg = knob("LEG_MASS")
        mu = knob("MU")
        # static PhysX reads
        mf = view.get_dof_max_forces().detach().cpu()
        stf = view.get_dof_stiffnesses().detach().cpu()
        dmp = view.get_dof_dampings().detach().cpu()
        if eff is not None:
            c.close("hip max force", mf[:, pitch_ids].mean(), 1000.0 * eff, 1e-4, rel=True)
            c.close("knee max force", mf[:, knee_ids].mean(), 400.0 * eff, 1e-4, rel=True)
        if kpkd is not None:
            c.close("hip stiffness", stf[:, pitch_ids].mean(), 16000.0 * kpkd, 1e-4, rel=True)
            c.close("hip damping", dmp[:, pitch_ids].mean(), 600.0 * kpkd, 1e-4, rel=True)
            c.close("knee stiffness", stf[:, knee_ids].mean(), 6000.0 * kpkd, 1e-4, rel=True)
            c.close("knee damping", dmp[:, knee_ids].mean(), 200.0 * kpkd, 1e-4, rel=True)
        if cam is not None:
            act = robot.actuators["body_hip_yaw"]
            c.close("cam effort", act.effort_limit.mean(), 20.0 * cam, 1e-5, rel=True)
            sat = act._saturation_effort
            c.close("cam saturation", sat.mean() if torch.is_tensor(sat) else sat, 24.0 * cam, 1e-5, rel=True)
        if leg is not None:
            m = view.get_masses().detach().cpu()
            m0 = robot.data.default_mass.detach().cpu()
            leg_ids, leg_names = robot.find_bodies("(FL|FR|ML|MR|RL|RR)_(Hip|Femur|Tibia|Footpad)")
            other = [i for i in range(m.shape[1]) if i not in set(leg_ids)]
            c.close("leg mass ratio", (m[:, leg_ids] / m0[:, leg_ids]).mean(), leg, 1e-5, rel=True)
            c.close("leg mass ratio spread", (m[:, leg_ids] / m0[:, leg_ids]).std(), 0.0, 1e-5)
            c.close("other mass ratio", (m[:, other] / m0[:, other]).mean(), 1.0, 1e-6, rel=True)
            c.close("leg links found", len(leg_ids), 24, 0)
            inn = view.get_inertias().detach().cpu()
            i0 = robot.data.default_inertia.detach().cpu()
            ratio = (inn[:, leg_ids, 0] / i0[:, leg_ids, 0])
            c.close("leg inertia ratio", ratio.mean(), leg, 1e-4, rel=True)
            res["leg_kg_per_leg_default"] = float(m0[0, leg_ids].sum() / 6)
        if mu is not None:
            mat = view.get_material_properties().detach().cpu()
            c.close("robot static friction", mat[..., 0].mean(), mu, 1e-6, rel=True)
            c.close("robot dynamic friction", mat[..., 1].mean(), mu, 1e-6, rel=True)
            c.close("robot friction spread", mat[..., :2].std(), 0.0, 1e-6)
            c.close("ground cfg static", u.cfg.scene.terrain.physics_material.static_friction, mu, 1e-9)
            c.close("ground cfg dynamic", u.cfg.scene.terrain.physics_material.dynamic_friction, mu, 1e-9)
            try:  # the authored ground material on the stage
                import omni.usd  # noqa: PLC0415
                from pxr import UsdPhysics  # noqa: PLC0415
                stage = omni.usd.get_context().get_stage()
                vals = []
                for prim in stage.Traverse():
                    p = str(prim.GetPath())
                    if p.startswith("/World/ground") and prim.HasAPI(UsdPhysics.MaterialAPI):
                        api = UsdPhysics.MaterialAPI(prim)
                        vals.append((p, api.GetStaticFrictionAttr().Get(), api.GetDynamicFrictionAttr().Get()))
                res["ground_materials"] = vals
                if vals:
                    c.true("ground stage friction", all(abs(s - mu) < 1e-6 and abs(d - mu) < 1e-6 for _, s, d in vals),
                           vals)
            except Exception as exc:  # noqa: BLE001 - report only
                res["ground_material_read_error"] = repr(exc)
        # dynamic: rod slew ceilings, cam slope / cap, vxy zero
        hip_max = knee_max = 0.0
        cam_hist = []
        cam_sat = 0.0
        vxy_absmax = 0.0
        for t in range(args.steps if (rod or cam) else 60):
            a = action_pattern(t, n, n_act, term, dev)
            obs, *_ = env.step(a)
            hip_max = max(hip_max, float(term.hip_rod_speed.abs().max()))
            knee_max = max(knee_max, float(term.knee_rod_speed.abs().max()))
            if len(cam_hist) < 3:
                cam_hist.append(float(term._cam_vel_applied.abs().max()))
            cam_sat = max(cam_sat, float(term._cam_vel_applied.abs().max()))
            vxy_absmax = max(vxy_absmax, float(obs["policy"][:, VXY].abs().max()))
        if rod is not None:
            c.close("hip rod speed ceiling", hip_max, linkage.HIP_SPEED_M_S * rod, 1e-4, rel=True)
            c.close("knee rod speed ceiling", knee_max, linkage.KNEE_SPEED_M_S * rod, 1e-4, rel=True)
        if cam is not None:
            dt = u.physics_dt
            sub = u.cfg.decimation
            # the 1-step action delay: the +1 cam command first acts on step 1 (step 0 applies the zero history)
            res["cam_applied_first_steps"] = cam_hist
            c.close("cam accel slope (one step)", cam_hist[2] - cam_hist[1],
                    linkage.dims.CAM_ACCEL_LIMIT_RAD_S2 * cam * dt * sub, 1e-4, rel=True)
            c.close("cam speed cap", cam_sat, math.pi * min(cam, 1.0), 1e-4, rel=True)
        if knob("VXY_ZERO", str) is not None:
            c.close("vxy slot zeroed", vxy_absmax, 0.0, 0.0)

    else:  # G2 (sensor) / G3 (push, standing still: a fall would restart the per-env push timer)
        delay = knob("ACT_DELAY", int)
        k = knob("OBS_NOISE")
        cs, cn = knob("CUR_SCALE"), knob("CUR_NOISE")
        vn = knob("VXY_NOISE")
        lag = knob("DEPTH_LAG", int)
        dv = knob("PUSH_DV")
        res_ang, res_rp, res_jp, res_jv, res_vxy, res_cur = [], [], [], [], [], []
        cur_clean = []
        raw_hits = []
        t_push = None
        pushes_prev = 0
        push_dv = []
        push_term = term._stress_push if dv is not None else None  # the action term's hook (stress_push.py)
        col = term._pitch_action_cols[0]
        resets = 0
        for t in range(args.steps):
            a = (torch.zeros(n, n_act, device=dev) if args.group == "G3"
                 else action_pattern(t, n, n_act, term, dev))
            if delay is not None:
                a[:, col] = 0.5 if t == 40 else 0.0
            v_before = robot.data.root_lin_vel_w[:, :2].clone()
            obs, *_ = env.step(a)
            resets += int((u.episode_length_buf == 0).sum())
            if delay is not None and t >= 40 and not raw_hits:
                if bool(torch.isclose(term._raw_actions[0, col], torch.tensor(0.5, device=dev))):
                    raw_hits.append(t)
            p = obs["policy"]
            if k is not None:
                res_ang.append((p[:, ANG] / 0.25 - robot.data.root_ang_vel_b).cpu())
                roll, pitch, _ = euler_xyz_from_quat(robot.data.root_quat_w)
                rp = torch.stack((wrap_to_pi(roll), wrap_to_pi(pitch)), dim=1)
                res_rp.append((p[:, RP] - rp).cpu())
                jpd = robot.data.joint_pos - robot.data.default_joint_pos
                if pterm._cam_shaft_obs_ids:
                    jpd[:, pterm._cam_shaft_obs_ids] = wrap_to_pi(jpd[:, pterm._cam_shaft_obs_ids])
                res_jp.append(wrap_to_pi(p[:, JP] - jpd).cpu())
                res_jv.append((p[:, JV] / 0.05 - robot.data.joint_vel).cpu())
            if vn is not None:
                res_vxy.append((p[:, VXY] / 2.0 - robot.data.root_lin_vel_b[:, :2]).cpu())
            if cs is not None or cn is not None:
                clean = pterm._get_contact_fill()
                cur_clean.append(clean.cpu())
                res_cur.append(p[:, CUR].cpu())
            if push_term is not None:
                np_ = int(push_term.pushes.sum())
                if np_ > pushes_prev:
                    if t_push is None:
                        t_push = t
                    push_dv.append(float((robot.data.root_lin_vel_w[:, :2] - v_before).norm(dim=-1).mean()))
                    pushes_prev = np_
        if delay is not None:
            c.close("action delay (steps)", (raw_hits[0] - 40) if raw_hits else -1, delay, 0)
        if k is not None:
            for name, lst, ref in (("ang_vel", res_ang, OBS_NOISE_REF["ang_vel"]), ("roll_pitch", res_rp,
                                   OBS_NOISE_REF["roll_pitch"]), ("joint_pos", res_jp, OBS_NOISE_REF["joint_pos"]),
                                   ("joint_vel", res_jv, OBS_NOISE_REF["joint_vel"])):
                x = torch.cat(lst)
                hw = k * ref
                c.true(f"{name} noise within +-k*ref", float(x.abs().max()) <= hw * (1 + 1e-3) + 1e-5,
                       float(x.abs().max()))
                c.close(f"{name} noise std", x.std(), hw / math.sqrt(3.0), 0.1, rel=True)
        if vn is not None:
            x = torch.cat(res_vxy)
            c.close("vxy noise std", x.std(), vn, 0.1, rel=True)
            c.close("vxy noise mean", x.mean(), 0.0, 0.2 * vn)
        if cs is not None or cn is not None:
            clean = torch.cat(cur_clean)
            got = torch.cat(res_cur)
            active = (clean > -0.3) & (clean < 0.6)
            c.true("current proxy driven (active samples)", int(active.sum()) > 100, int(active.sum()))
            want = ((clean + 0.5) * (cs if cs is not None else 1.0))
            resid = (got + 0.5) - want
            if cn is None:
                c.close("current gain residual (no noise)", resid[active].abs().max(), 0.0, 1e-4)
            else:
                c.close("current noise std (active)", resid[active].std(), cn, 0.15, rel=True)
            c.true("current proxy range", float(got.min()) >= -0.5 - 1e-6 and float(got.max()) <= 1.0 + 1e-6,
                   (float(got.min()), float(got.max())))
        if lag is not None:
            for g in DEPTH_GROUPS:
                d = obs_term(u, g)
                c.true(f"{g} lag/buffer", d.lag == lag and d.buffer_len == lag + 1, (d.lag, d.buffer_len))
                c.true(f"{g} returns buffer[-1-lag]", bool(torch.equal(obs[g], d.depth_buffer[:, -1 - lag])))
        res["resets_during_run"] = resets
        if dv is not None:
            step_s = u.step_dt
            c.true("pushes happened", pushes_prev >= n, pushes_prev)
            c.close("first push time (s)", (t_push + 1) * step_s if t_push is not None else -1.0, 10.0, 2 * step_s)
            c.close("push |dv| (m/s)", float(np.mean(push_dv)) if push_dv else -1.0, dv, 0.15 * dv + 0.02)

    res["checks"] = c.rows
    res["all_ok"] = c.ok
    res["wall_s"] = time.time() - t0
    res["complete"] = True
    Path(args.out).write_text(json.dumps(res, indent=1, sort_keys=True, default=str))
    print(f"STRESS_SMOKE END group={args.group} all_ok={c.ok} checks={len(c.rows)}", flush=True)
    env.close()
    return 0


if __name__ == "__main__":
    rc = main()
    app.close()
    sys.exit(rc)
