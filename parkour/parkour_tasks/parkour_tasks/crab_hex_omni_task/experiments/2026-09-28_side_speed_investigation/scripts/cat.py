import sys,re,collections,math
# usage: cat.py log [lastN]
f=sys.argv[1]; N=int(sys.argv[2]) if len(sys.argv)>2 else 50
blocks=[];cur=None
for line in open(f,errors='ignore'):
    if 'Learning iteration' in line:
        if cur is not None: blocks.append(cur)
        cur={}
        continue
    if cur is None: continue
    m=re.match(r'\s*Episode_RewardCat/(\w+)/(\w+):\s*([-\d.eE+naif]+)',line)
    if m:
        try: cur[(m.group(1),m.group(2))]=float(m.group(3))
        except: pass
if cur: blocks.append(cur)
blocks=[b for b in blocks if b][-N:]
acc=collections.defaultdict(list)
for b in blocks:
    for k,v in b.items():
        if not math.isnan(v): acc[k].append(v)
mean={k:sum(v)/len(v) for k,v in acc.items()}
cats=['fwd','side','diag','back','turn','move_turn','stand']
terms=sorted({k[1] for k in mean})
print(f"{len(blocks)} iters averaged; {f.split('/')[-1]}")
print(f"{'term':42s}"+''.join(f"{c:>10s}" for c in cats))
for t in terms:
    vals=[mean.get((c,t),float('nan')) for c in cats]
    if all(abs(v)<1e-5 or math.isnan(v) for v in vals): continue
    print(f"{t:42s}"+''.join(f"{v:10.4f}" for v in vals))
