import pickle,numpy as np,glob,os
T="Metrics/base_parkour/track_ratio_mean"
for p in sorted(glob.glob('tb/X*.pkl')):
    d=pickle.load(open(p,'rb')); s=np.array(d[T]); st,v=s[:,0],s[:,1]
    er=v[(st>=800)&(st<1000)].mean(); fin=v[(st>=2900)&(st<3000)].mean()
    print(os.path.basename(p)[:3], int(st.max()), len(st), f"early {er:.3f} (n {((st>=800)&(st<1000)).sum()})  final {fin:.3f} (n {((st>=2900)&(st<3000)).sum()})")
