"""Deterministic paired autocorrelation HNR and spectral-band energy proxies."""
import numpy as np

def hnr_frames(wav,sr=24000):
    # 25ms rectangular frames, 10ms stride, subtract frame mean. For each
    # frame maximize normalized overlapping-segment autocorrelation over
    # 60–400Hz lags. HNR=10log10(r/(1-r)); no pitch-dependent retuning.
    x=np.asarray(wav,np.float64).reshape(-1);n=int(.025*sr);hop=int(.010*sr)
    frames=np.lib.stride_tricks.sliding_window_view(x,n)[::hop].copy()
    frames-=frames.mean(axis=1,keepdims=True)
    corrs=[]
    for lag in range(int(sr/400),int(sr/60)+1):
        a,b=frames[:,:-lag],frames[:,lag:]
        denom=np.sqrt(np.sum(a*a,axis=1)*np.sum(b*b,axis=1))
        corrs.append(np.sum(a*b,axis=1)/np.maximum(denom,1e-20))
    rho=np.clip(np.max(corrs,axis=0),1e-8,1-1e-8)
    return 10*np.log10(rho/(1-rho))

def highband_db(wav,sr=24000):
    # Whole valid raw waveform, mean removed, unwindowed FFT. Disjoint bins:
    # low [0,4kHz), high [4kHz,12kHz], both in the same real-signal spectrum.
    x=np.asarray(wav,np.float64).reshape(-1);x=x-x.mean()
    z=np.fft.rfft(x);power=np.abs(z)**2;frequency=np.fft.rfftfreq(x.size,1/sr)
    # Parseval weights for real signal: DC and Nyquist single, interior double.
    weights=np.full(power.shape,2.0);weights[0]=1
    if x.size%2==0:weights[-1]=1
    power*=weights
    low=power[frequency<4000].sum();high=power[(frequency>=4000)&(frequency<=12000)].sum()
    return float(10*np.log10(max(high,1e-30)/max(low,1e-30)))

def paired_proxies(reference,actual):
    assert np.shape(reference)==np.shape(actual)
    a,b=hnr_frames(reference),hnr_frames(actual);voiced=a>0
    assert voiced.any(),'No voiced baseline frames; supervisor policy required'
    ra,rb=float(a[voiced].mean()),float(b[voiced].mean())
    ha,hb=highband_db(reference),highband_db(actual)
    return {'hnr':{'reference_mean_db':ra,'actual_mean_db':rb,'difference_db':rb-ra,'voiced_frames':int(voiced.sum()),'all_frames':a.size,'limit_db':1.0,'pass':bool(abs(rb-ra)<=1.0)},
            'highband_ratio':{'reference_db':ha,'actual_db':hb,'difference_db':hb-ha,'limit_db':1.5,'pass':bool(abs(hb-ha)<=1.5)},
            'pass':bool(abs(rb-ra)<=1 and abs(hb-ha)<=1.5)}
