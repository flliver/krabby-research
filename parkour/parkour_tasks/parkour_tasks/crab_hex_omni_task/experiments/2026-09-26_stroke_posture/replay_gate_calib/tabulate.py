import json, sys, glob, os
S = None
root = sys.argv[1]
order = ["base","p15","noff3","p15_noff3","rho04","rho06","drho02","drho04","cmax10","cmax125","cmax175","cmax20","fh15","fh25","fo40","fo60","ml4","ml6","plo010","plo020","pw010","pw020","ptau3","ptau5","sig015","sig025","phisat020","phisat015"]
tags = [t for t in order if os.path.exists(f"{root}/{t}/replay_gate.json")] + sorted(set(os.path.basename(os.path.dirname(p)) for p in glob.glob(f"{root}/*/replay_gate.json")) - set(order))
rows = []
for t in tags:
    d = json.load(open(f"{root}/{t}/replay_gate.json")); c = d["criteria"]
    r = json.load(open(f"{root}/{t}/restated.json"))
    lit = [k for k in ("R1","R2","R3","R4","R5","R6","R7","R8","P1","P2","P3","P4") if c[k].get("pass") is False]
    r1r = min(v["kept_planted"] for v in r["R1r"]["rows"].values())
    r1 = min(v for v in c["R1"]["kept"].values() if v is not None)
    sk = max(c["R2"]["side_kept"].values())
    r2c = max(v for v in c["R2"]["side_vs_sideways"].values() if v is not None)
    r3a = max(v for v in c["R3"]["short_share"].values() if v is not None)
    ws = r["R3r"]["rows"]["W.side"]
    r3lo = max(v["ci95"][0] for v in r["R3r"]["rows"].values() if v["ci95"][0] is not None)
    ta = c["R5"]["term_alone"]
    ta_tr = max(v["ratio"] for v in ta.values() if v["gated"] and v["ratio"] is not None)
    ta_rot = max(v["ratio"] for v in ta.values() if not v["gated"] and v["ratio"] is not None)
    r6rows = {k: v for k, v in c["R6"]["rows"].items() if v.get("gated", True)}
    r6i = max(abs(v["income_shift"]) for v in r6rows.values())
    r6s = max((v["income_dup"]/v["income"]-1) for v in r6rows.values())
    r6l = max(v["liftoff_shift"] for v in r6rows.values() if v.get("liftoff_shift") is not None)
    r7 = max(abs(v["shift_pp"]) for v in c["R7"]["rows"].values()); r7a = max(v["absorbed"] or 0 for v in c["R7"]["rows"].values())
    r8 = max(v["gap"] for v in c["R8"]["rows"].values() if v["gap"] is not None)
    r8ls = max(v["gap"] for k, v in c["R8"]["rows"].items() if v["gap"] is not None and not k.endswith("move_turn"))
    p1 = c["P1"]; p3 = c["P3"]
    fwdsh = c["R4"]["shares_at_w_S"]["W"].get("fwd")
    r5nf = c["R5"]["clauses"]["no_flip"]
    rows.append((t, d["w_S"], d["w_P"], ",".join(lit), r1, r1r, sk, r2c, r3a, ws["rho"], r3lo, r5nf, ta_tr, ta_rot, r6i, r6s, r6l, r7, r7a, r8, r8ls, p1["sideways_side"]["side_slow"], p1["sideways_side"]["side_step"], p1["healthy_side"]["W"], (p3.get("creeper") or {}).get("ratio"), p3["clauses"]["no_flip"]))
hdr = ["tag","w_S","w_P","literal FAIL","R1min","R1r min","R2a max","R2c max","R3a max","R3b W.side","R3b maxCIlo","R5 noflip","R5 alone tr","R5 alone rot","R6 |inc|","R6 inc+","R6 lift","R7 pp","R7 abs","R8 max","R8 side/turn","P1 flat","P1 step","P1 W","P3 creeper","P3 noflip"]
fmt = lambda x: (f"{x:.3f}" if isinstance(x, float) else str(x))
print(" | ".join(hdr))
for r_ in rows:
    print(" | ".join(fmt(x) for x in r_))
json.dump({"hdr": hdr, "rows": rows}, open(f"{root}/table.json", "w"), indent=0)
