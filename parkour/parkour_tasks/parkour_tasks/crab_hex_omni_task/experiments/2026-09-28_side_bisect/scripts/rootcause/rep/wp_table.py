import json, numpy as np
D = json.load(open("weight_probe.json"))
ITS = ["0", "100", "300", "600", "900", "1200", "1500", "2000", "2500", "2999"]
def tab(title, f, fmt="%6.2f"):
    print("\n==", title); print("%-9s %-3s %-13s " % ("arm", "cl", "cfg") + " ".join("%6s" % i for i in ITS))
    for lab, v in D.items():
        r = v["rows"]; print("%-9s %-3s %-13s " % (lab, v["cls"], v["desc"]) + " ".join(fmt % f(r[i]) if i in r else "   n/a" for i in ITS))
tab("first-layer col-norm: cmd_vy (slot 9) / median prop col", lambda r: r["col"]["cmd_vy9"] / r["ref"])
tab("first-layer col-norm: cmd_vx (slot 8) / median prop col", lambda r: r["col"]["cmd_vx8"] / r["ref"])
tab("first-layer col-norm: clock (15-16) / median prop col", lambda r: r["col"]["clock15_16"] / r["ref"])
tab("first-layer col-norm: facing (6-7) / median prop col", lambda r: r["col"]["facing6_7"] / r["ref"])
tab("first-layer col-norm: speed |cmd| (10) / median prop col", lambda r: r["col"]["speed10"] / r["ref"])
tab("first-layer col-norm: lin vel (13-14) / median prop col", lambda r: r["col"]["linv13_14"] / r["ref"])
tab("first-layer col-norm: wz cmd (5) / median prop col", lambda r: r["col"]["wz_cmd5"] / r["ref"])
tab("median prop col norm (abs)", lambda r: r["ref"], "%6.3f")
tab("INPUT-SCALED: |W col 9| * 0.28 / (|W col clock| * 0.707)   [direction drive vs clock drive into layer 1]",
    lambda r: r["col"]["cmd_vy9"] * 0.28 / (r["col"]["clock15_16"] * 0.707))
tab("path-norm proxy to rods: cmd_vy9 / clock", lambda r: r["path_rod"]["cmd_vy9"] / r["path_rod"]["clock15_16"])
tab("output layer |W4| (Frobenius)", lambda r: r["W4n"], "%6.2f")
tab("|W1|,|W2|,|W3| product (1e3)", lambda r: r["W1n"] * r["W2n"] * r["W3n"] / 1e3, "%6.2f")
