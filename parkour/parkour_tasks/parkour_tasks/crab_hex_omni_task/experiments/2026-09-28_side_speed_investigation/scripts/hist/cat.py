import sys,re,collections
f=sys.argv[1]; N=int(sys.argv[2]) if len(sys.argv)>2 else 50
# collect per-iteration blocks
blocks=[];cur=None
for line in open(f,errors='ignore'):
    if 'Learning iteration' in line:
        if cur: blocks.append(cur)
        cur={}
    elif cur is not None:
        m=re.search(r'Episode_RewardCat/(\w+)/(\w+):\s*([-0-9.eE+]+)',line)
        if m: cur[(m.group(1),m.group(2))]=float(m.group(3))
if cur: blocks.append(cur)
blocks=[b for b in blocks if b][-N:]
acc=collections.defaultdict(list)
for b in blocks:
    for k,v in b.items(): acc[k].append(v)
cats=sorted({k[0] for k in acc}); terms=sorted({k[1] for k in acc})
show=[c for c in ['stand','fwd','side','back','diag','turn','move_turn'] if c in cats]+[c for c in cats if c not in ['stand','fwd','side','back','diag','turn','move_turn']]
print('last',len(blocks),'iters; cats',show)
print(f"{'term':42s}"+''.join(f"{c:>10s}" for c in show))
for t in terms:
    row=[sum(acc[(c,t)])/len(acc[(c,t)]) if acc.get((c,t)) else float('nan') for c in show]
    if all(abs(x)<0.005 for x in row if x==x): continue
    print(f"{t:42s}"+''.join(f"{x:10.3f}" for x in row))
