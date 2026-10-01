import pickle,glob,os,numpy as np,collections,csv
T='/tmp/claude-1000/-home-nickmagus-krabby-krabby-research/f09fa754-dd17-413f-8553-6b22fdb41cf7/scratchpad/ppo_review/telemetry/'
rows={r['run_id']:r for r in csv.DictReader(open(T+'runs.csv'))}
conc=pickle.load(open(T+'conc.pkl','rb'))
S={}
for p in glob.glob(T+'series/*.pkl'):
    d=pickle.load(open(p,'rb')); rid=os.path.basename(p)[:-4]
    s=d['series']
    if 'Perf/collection time' not in s: continue
    S[rid]={k:np.array(s[k]) for k in ['Perf/collection time','Perf/learning_time','Perf/total_fps','Train/mean_episode_length'] if k in s}
# spans for concurrency per-iteration
spans={k:(v['Perf/collection time'][0,1],v['Perf/collection time'][-1,1]) for k,v in S.items()}
def busy(k):
    a=S[k]['Perf/collection time']; w=a[:,1]; b=np.zeros(len(w),bool)
    for kk,s in spans.items():
        if kk!=k: b|=(w>=s[0])&(w<=s[1])
    return b
print('== 64-env runs')
for k,r in sorted(rows.items(), key=lambda x:x[1]['run']):
    if k not in S or r['num_envs']!='64': continue
    a=S[k]['Perf/collection time']; l=S[k]['Perf/learning_time']; b=busy(k)
    el=S[k].get('Train/mean_episode_length')
    print(r['experiment'][9:],r['run'][:40],'its',len(a),'resume',r['resume'],'coll med %.3f (it>=2 %.3f) learn %.3f busyfrac %.2f eplen %.0f->%.0f'%(np.median(a[:,2]),np.median(a[2:,2]) if len(a)>2 else np.nan,np.median(l[:,2]),b.mean(), el[0,2] if el is not None else -1, el[-1,2] if el is not None else -1))
print('== matched window: iterations 0-19, fresh (resume False), not concurrent')
res=collections.defaultdict(lambda: collections.defaultdict(list))
for k,r in rows.items():
    if k not in S or r['resume']!='False': continue
    a=S[k]['Perf/collection time']; l=S[k]['Perf/learning_time']; b=busy(k)
    m=(a[:,0]<20)&(a[:,0]>=1)&(~b)
    if m.sum()<10: continue
    e=r['experiment'].replace('crab_hex_','')
    res[e][r['num_envs']].append((np.median(a[m,2]),np.median(l[m,2])))
for e in sorted(res):
    for n in sorted(res[e]):
        v=np.array(res[e][n]); print(e,n,'runs',len(v),'coll p50 %.3f [p10 %.3f p90 %.3f] learn p50 %.3f'%(np.median(v[:,0]),np.percentile(v[:,0],10),np.percentile(v[:,0],90),np.median(v[:,1])))
print('== whole-run medians, 256, non-concurrent iterations only, fresh vs resume')
res2=collections.defaultdict(list)
for k,r in rows.items():
    if k not in S: continue
    a=S[k]['Perf/collection time']; b=busy(k)
    if (~b).sum()<10: continue
    res2[(r['experiment'].replace('crab_hex_',''),r['num_envs'])].append(np.median(a[~b,2]))
for k in sorted(res2): v=res2[k]; print(k,len(v),'p50 %.3f p10 %.3f p90 %.3f'%(np.median(v),np.percentile(v,10),np.percentile(v,90)))
print('== date-matched: 256 runs of same experiment starting within the 64-run date range (+-2 days), non-concurrent its, iterations 1-19 and whole-run')
import datetime
for e in ['flat_walk','omni_flat_walk','side_flat_walk','side_teacher','teacher']:
    r64=[r for k,r in rows.items() if k in S and r['num_envs']=='64' and r['experiment']=='crab_hex_'+e]
    d64=sorted(r['start_time'][:10] for r in r64)
    lo=datetime.date.fromisoformat(d64[0])-datetime.timedelta(days=2); hi=datetime.date.fromisoformat(d64[-1])+datetime.timedelta(days=2)
    out=[]
    for k,r in rows.items():
        if k not in S or r['num_envs']!='256' or r['experiment']!='crab_hex_'+e: continue
        d=datetime.date.fromisoformat(r['start_time'][:10])
        if not (lo<=d<=hi): continue
        a=S[k]['Perf/collection time']; b=busy(k); m=(~b)&(a[:,0]>=1)&(a[:,0]<20)
        out.append((r['run'][:30],r['resume'],len(a),np.median(a[~b,2]) if (~b).sum() else np.nan, np.median(a[m,2]) if m.sum() else np.nan))
    print(e,'64-run dates',d64[0],d64[-1],'n256 in window',len(out))
    for o in sorted(out): print('   ',o[0],'resume',o[1],'its',o[2],'whole %.3f first20 %.3f'%(o[3],o[4]))
