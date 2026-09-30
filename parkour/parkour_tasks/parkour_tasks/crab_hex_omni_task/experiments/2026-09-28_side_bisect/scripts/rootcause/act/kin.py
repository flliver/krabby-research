"""Q1/Q2: where does travel come from? stance-foot kinematic decomposition (rod = radial, cam = yaw) + slip."""
import sys, numpy as np
from lib import *
DT = 0.02; FTH = 50.0
def analyse(key, labels, axis):
    m, jn, an, E = episodes(key, labels)
    ji = {n:i for i,n in enumerate(jn)}
    res = {}
    for lab in labels:
        eps = [e for e in E if e["label"] == lab]
        if not eps: continue
        acc = {k:[] for k in ["vbody","vfeet","rod","cam","slip","rot","sweep","sweep_rod","sweep_cam","lean","z","nstance",
                              "camvel","yawamp","radamp","ctr_lead","ctr_trail","air","prog_air","duty","camabs"]}
        for e in eps:
            c = e["cmd"][:, :2].mean(0)
            d = np.zeros(3); 
            if axis == "y": d[1] = np.sign(c[1])
            else: d[0] = np.sign(c[0])
            fb, hb = e["fb"], e["hb"]
            u = fb[..., :2] - hb[..., :2]; rho = np.linalg.norm(u, axis=-1); er = u / rho[..., None]
            et = np.stack([-er[..., 1], er[..., 0]], -1)
            vrel = np.gradient(fb, DT, axis=0)[..., :2]
            vr = (vrel * er).sum(-1); vt = (vrel * et).sum(-1)
            dd = d[:2]
            rod_d = vr * (er @ dd); cam_d = vt * (et @ dd)
            st = e["F"] > FTH
            w = e["F"] * st
            vbody = e["vb"] @ d
            # stance-foot identity: v_body = v_foot_w(body frame) - v_rel - w x r
            fvb = np.einsum('tji,tkj->tki', e["R"], e["fv"])  # world->body: R^T v ; R is body->world so use transpose
            fvb = np.einsum('tij,tkj->tki', np.transpose(e["R"], (0,2,1)), e["fv"])
            wx = np.cross(e["wb"][:, None, :], fb)
            slip_d = fvb @ d; rot_d = wx @ d
            W = w.sum(1); ok = W > 0
            def wavg(x): return float(((x * w).sum(1)[ok] / W[ok]).mean())
            acc["vbody"].append(vbody.mean()); acc["rod"].append(-wavg(rod_d)); acc["cam"].append(-wavg(cam_d))
            acc["slip"].append(wavg(slip_d)); acc["rot"].append(-wavg(rot_d))
            # feet centroid speed (world), projected on command direction in world
            dw = np.array([c[0], c[1], 0.]) ; dw = np.einsum('ij,j->i', e["R"][0], d)
            cen = e["fw"].mean(1); acc["vfeet"].append(float((cen[-1] - cen[0]) @ dw / (len(cen) * DT)))
            # lean: body over feet centroid along d (body frame)
            acc["lean"].append(float(-(fb.mean(1) @ d).mean())); acc["z"].append(float((e["p"][:, 2] - e["tz"].mean(1)).mean()))
            acc["nstance"].append(float(st.sum(1).mean())); acc["air"].append(float((st.sum(1) <= 2).mean()))
            acc["duty"].append(float(st.mean()))
            # progress split: body displacement during frames with <=2 vs >=3 loaded feet
            acc["prog_air"].append(float(vbody[st.sum(1) <= 2].sum() / max(vbody.sum(), 1e-6)))
            # stance sweep per bout (body frame along d), split rod/cam
            sw, swr, swc = [], [], []
            for k in range(6):
                s = st[:, k].astype(int); ed = np.diff(np.r_[0, s, 0]); on = np.where(ed == 1)[0]; off = np.where(ed == -1)[0]
                for a, b in zip(on, off):
                    if b - a < 5: continue
                    sw.append(abs((fb[b-1, k] - fb[a, k]) @ d)); swr.append(-(rod_d[a:b, k].sum() * DT)); swc.append(-(cam_d[a:b, k].sum() * DT))
            acc["sweep"].append(np.median(sw) if sw else np.nan); acc["sweep_rod"].append(np.median(swr) if swr else np.nan); acc["sweep_cam"].append(np.median(swc) if swc else np.nan)
            cv = np.stack([e["qd"][:, ji[l+"_Body_CamShaft_RevoluteJoint"]] for l in LEGS], 1)
            acc["camvel"].append(cv.mean(0)); acc["camabs"].append(np.abs(cv).mean(0))
            yaw = np.stack([e["q"][:, ji[l+"_Body_Hip_RevoluteJoint"]] for l in LEGS], 1)
            acc["yawamp"].append(np.ptp(yaw, 0)); acc["radamp"].append(np.ptp(rho, 0))
            # per-leg foot position along d relative to its hip (leading row = hip on +d side)
            side = np.sign(hb[..., :2].mean(0) @ dd)
            rel = ((fb[..., :2] - hb[..., :2]) @ dd).mean(0) * side  # outboard(+)/inboard(-) of hip
            acc["ctr_lead"].append(rel[side > 0].mean()); acc["ctr_trail"].append(rel[side < 0].mean())
        r = {}
        for k, v in acc.items():
            v = np.array(v, dtype=float)
            r[k] = np.nanmedian(v, 0) if v.ndim > 1 else float(np.nanmedian(v))
        r["n"] = len(eps); res[lab] = r
        print(f"{key:14s} {lab:10s} n{len(eps):3d} vbody {r['vbody']:.3f} feet {r['vfeet']:.3f} | stance: rod {r['rod']:+.3f} cam {r['cam']:+.3f} slip {r['slip']:+.3f} rot {r['rot']:+.3f} "
              f"| sweep {r['sweep']:.3f} (rod {r['sweep_rod']:+.3f} cam {r['sweep_cam']:+.3f}) lean {r['lean']:+.3f} z {r['z']:.3f} nst {r['nstance']:.1f} duty {r['duty']:.2f} le2 {r['air']:.2f} progle2 {r['prog_air']:.2f} "
              f"| lead_out {r['ctr_lead']:+.3f} trail_out {r['ctr_trail']:+.3f}")
        print(f"{'':25s} camvel {np.round(r['camvel'],2)} |cam| {np.round(r['camabs'],2)} yawptp {np.round(r['yawamp'],2)} radptp {np.round(r['radamp'],3)}")
    return res
if __name__ == "__main__":
    import pickle
    allr = {}
    for key, labs, ax in [("W", ["fwd", "back"], "x"), ("W", ["lat_l", "lat_r"], "y"),
                          ("canon:C", ["lat_l_022", "lat_r_022"], "y"), ("canon:C_mo_s2", ["lat_l_022", "lat_r_022"], "y"),
                          ("C_mo_s1", ["lat_l_022"], "y"), ("O", ["lat_l_022"], "y"), ("C_rsi", ["lat_l_022"], "y"),
                          ("lr:LR4", ["lat_l_022", "lat_r_022"], "y"), ("lr:LRfa", ["lat_l_022", "lat_r_022"], "y"),
                          ("lr:C_lr", ["lat_l_022", "lat_r_022"], "y"), ("lr:C_lr_10", ["lat_l_022", "lat_r_022"], "y"),
                          ("lr:C_lr_rsi", ["lat_l_022", "lat_r_022"], "y")]:
        allr[(key, ax)] = analyse(key, labs, ax)
    pickle.dump(allr, open("kin.pkl", "wb"))
