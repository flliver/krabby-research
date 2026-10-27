from symlib import *
for name,lab in [("O","lat_l_022"),("C_rsi","lat_l_022"),("R","lat_l_022"),("W","lat_l_022"),("C","lat_l_022"),("LR4_dwell","lat_l_022"),("side_s3_4999","lat_l_022")]:
    d=load(name,lab)
    for rep in ("jp","act","foot"):
        X=d[rep].astype(float); X=X-X.mean(1,keepdims=True)
        nfft=4096
        F=(np.abs(np.fft.rfft(X*np.hanning(300)[None,:,None],n=nfft,axis=1))**2).sum(2).mean(0)
        f=np.fft.rfftfreq(nfft,DT)
        # report power in bands
        tot=F[(f>0.05)&(f<5)].sum()
        def bp(a,b): return F[(f>=a)&(f<b)].sum()/tot
        pk=[(f[i],F[i]) for i in range(1,len(F)-1) if F[i]>F[i-1] and F[i]>F[i+1] and f[i]<2.5]
        pk=sorted(pk,key=lambda x:-x[1])[:4]
        print(f"{name:12s} {rep:4s} peaks {[round(p[0],3) for p in pk]}  frac<0.25Hz {bp(0.05,0.25):.2f} 0.25-0.4 {bp(0.25,0.4):.2f} 0.4-0.75 {bp(0.4,0.75):.2f} 0.75-1.1 {bp(0.75,1.1):.2f} >1.1 {bp(1.1,5):.2f}")
