import sys, os
FIX = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'ced-fix')
sys.path.insert(0, FIX)
import evaluate as ev
moves = ev.load_all()
by = {m[0]: m for m in moves}
out = []
def emit(name, s, retract, leaving):
    out.append(f"{name} {int(retract)} {int(leaving)} {len(s)}")
    for e, i, r in s:
        out.append(f"{e} {i} {r}")
f1 = by['clearance-FR-115750#7']
s = f1[3]
print('F1 len', len(s), 'max e', max(x[0] for x in s), file=sys.stderr)
emit('F1full', s, True, False)
emit('F1fullpad', ev.pad(s), True, False)
for cut in range(0, 200):
    v = ev.cut_variant(s, cut)
    if v is None or max(x[0] for x in v) < 12: break
    emit(f'F1cut{cut}_tot{max(x[0] for x in v)}', v, True, False)
open('f1.txt','w').write('\n'.join(out)+'\n')
# dump the F1 sample stream for inspection
with open('f1_samples.txt','w') as f:
    for k,(e,i,r) in enumerate(s): f.write(f"{k} {e} {i} {r}\n")
