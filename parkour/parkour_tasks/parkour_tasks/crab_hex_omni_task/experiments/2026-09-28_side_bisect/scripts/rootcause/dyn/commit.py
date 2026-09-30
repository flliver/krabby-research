import sys; sys.path.insert(0,'.')
from common import *
T='Metrics/base_parkour/track_ratio_mean'
allx=FAST+SLOW
res={}
for x in allx:
    s,v=ser(x,T); sm=smooth(v,50)
    last=s.max()
    fin=np.nanmean(v[s>=last-100])
    def first(frac):
        idx=np.where(sm>=frac*fin)[0]; return int(s[idx[0]]) if len(idx) else None
    # plateau: max of smoothed before 1000 vs final
    peak=np.nanmax(sm); pk_it=int(s[np.nanargmax(sm)])
    res[x]=dict(fin=fin,c90=first(.90),c95=first(.95),c98=first(.98),peak=peak,pk_it=pk_it,last=int(last))
    at=lambda i: float(np.nanmean(v[(s>=i-25)&(s<i+25)]))
    print(f"{ARM[x]:12s} {'F' if x in FAST else 'S'} fin={fin:.3f} c90={res[x]['c90']} c95={res[x]['c95']} c98={res[x]['c98']} peak={peak:.3f}@{pk_it} last={int(last)} | tr@200 {at(200):.3f} @300 {at(300):.3f} @400 {at(400):.3f} @500 {at(500):.3f} @600 {at(600):.3f} @800 {at(800):.3f} @1000 {at(1000):.3f} @1500 {at(1500):.3f}")
# separation: per iteration smoothed min fast vs max slow (1D slow and all slow)
S={x:smooth(ser(x,T)[1],50) for x in allx}
n=min(len(S[x]) for x in FAST+[y for y in SLOW if y!='X19'])
for grp,name in [([y for y in SLOW if y!='X19'],'all slow excl X19'),(['X01','X03','X05','X06'],'1D slow'),(SLOW,'all slow incl X19 (to 1500)')]:
    m=min(n, min(len(S[y]) for y in grp))
    f=np.nanmin(np.stack([S[x][:m] for x in FAST]),0); sl=np.nanmax(np.stack([S[y][:m] for y in grp]),0)
    gap=f-sl
    # last iteration where gap<=0
    bad=np.where(~(gap>0))[0]
    print(name, 'separated permanently from it', bad[-1]+1 if len(bad) else 0, 'gap@600', gap[600], 'gap@1000',gap[1000] if m>1000 else None,'gap@end', gap[m-1])
    fe=np.nanmean(np.stack([S[x][:m] for x in FAST]),0); se=np.nanmean(np.stack([S[y][:m] for y in grp]),0)
    print('   mean F-S at', {i: round(float(fe[i]-se[i]),3) for i in [100,200,300,400,500,600,800,1000,1400]})
import json; json.dump(res,open('commit.json','w'),indent=1)
