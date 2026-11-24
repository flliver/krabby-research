import sys; sys.path.insert(0,'.')
import detector as det
class Noise:
    def __init__(s, seed): s.state=(seed*2654435761+1)&0xFFFFFFFF
    def next(s, a):
        s.state=(s.state*1103515245+12345)&0xFFFFFFFF
        if a<=0: return 0
        return ((s.state>>16)%(2*a+1))-a
def run(out, e, count, per, i, noise=None, amp=0):
    for n in range(count):
        e+=per; out.append((e, i+(noise.next(amp) if noise else 0)))
    return e
def feed(retract, leaving, samples, rampK=0):
    d=det.EndDetector(1, retract, not leaving, det.Params())
    for n,(e,i) in enumerate(samples):
        x=d.step(e,i,1 if n>=rampK else 0)
        if x: return x.kind, n, e, d
    return 'none', -1, 0, d
# ripple
for so in (0,1):
    res=set()
    for seed in range(1,51):
        nz=Noise(seed); s=[]; e=0
        e=run(s,e,40,5,160,nz,10)
        for n in range(20):
            if n%2==1: e+=1
            s.append((e,250+nz.next(50)))
        for n in range(60): s.append((e, (130 if so else 170) if (n//5)%2==0 else 310))
        k=feed(True,False,s); res.add((k[0], getattr(k[3],'normal_n',0)>=12))
    print('ripple', so, res)
# short curl
for lr in (0,1):
    s=[]; e=0
    e=run(s,e,10,1,60)
    if lr: e=run(s,e,40,5,160)
    cur=160 if lr else 100
    for st in [5,4,4,4,3,3,3,3,2,2,2,2,1,1,1,1,1,0,1,0]:
        e+=st; cur+= 10 if cur<170 else 0; s.append((e,cur))
    nz=Noise(7); e=run(s,e,40,0,170,nz,15)
    k=feed(False,False,s,10); print('shortcurl', lr, k[:3], k[3].armed, getattr(k[3],'normal_n',0), k[3].i_ref)
    if not lr: print(' uncurl', feed(False,True,s,10)[:3])
# short retract
for rise in (0,1):
    s=[]; e=0; e=run(s,e,10,1,60)
    for st in [5,5,4,4,3,3,3,3,2,2,2,2,2,2]:
        e+=st; s.append((e,100))
    e=run(s,e,40,0,101 if rise else 95)
    k=feed(True,False,s,10); print('shortretract', rise, k[:3], k[3].armed, getattr(k[3],'normal_n',0), k[3].i_ref)
s=[]; e=0; e=run(s,e,10,1,60); e=run(s,e,10,5,100)
for n in range(50):
    if n%3==0: e+=1
    s.append((e,100))
e=run(s,e,40,0,100)
k=feed(False,False,s,10); print('crawl', k[:3], k[3].armed, getattr(k[3],'normal_n',0), k[3].arm_k, k[3].last_edge_k)
# slow curl pause
s=[]; e=0
for n in range(40):
    if n%4!=3: e+=1
    s.append((e,120))
print('e after slow', e)
e=run(s,e,12,0,120)
for n in range(20):
    if n%4!=3: e+=1
    s.append((e,120))
L=len(s)-1
while s[L][0]==s[L-1][0]: L-=1
e=run(s,e,40,0,120)
k=feed(True,False,s,4); print('slowpause', k[:3], k[3].armed, 'lastedge', L)
