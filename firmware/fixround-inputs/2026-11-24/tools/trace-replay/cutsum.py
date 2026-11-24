import sys, collections
res = collections.OrderedDict()
for line in open(sys.argv[1]):
    p = line.split(); name, tag, off, tr, pwm, joint, tot = p[0].split("|")
    d = {"kind": p[1]}
    for kv in p[2:]:
        k, v = kv.split("="); d[k] = int(v)
    d["tot"] = int(tot)
    res.setdefault((name, tr, pwm, joint), {}).setdefault(tag[:2], {}).setdefault(tag, {})[off] = d
grand = collections.Counter()
for (name, tr, pwm, joint), meths in res.items():
    for meth, tags in meths.items():
        c = collections.Counter(); rng = collections.defaultdict(list)
        for tag, offs in tags.items():
            for off, d in offs.items():
                key = d["kind"] + ("*" if d["kind"] == "Still" and d["kArm"] >= 0 else "")
                c[key] += 1; rng[key].append(d["tot"])
                grand[(tr, pwm, meth, key)] += 1
        desc = " ".join(f"{k}:{v}[{min(rng[k])}-{max(rng[k])}]" for k, v in sorted(c.items()))
        print(f"{name:36s} {joint} pwm={pwm} truth={tr:6s} {meth} cuts={len(tags):3d}x3  {desc}")
print()
for k, v in sorted(grand.items()): print(k, v)
