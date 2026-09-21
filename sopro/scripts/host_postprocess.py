"""NumPy mirror of sopro 2.2.0 single-segment synthesis post-processing.

All indexes follow the source's integer frame arithmetic. Quantiles use linear
interpolation and source comparisons are strict (>), preserving trim points.
"""
import numpy as np

SAMPLE_RATE = 24000
ONSET_THRESHOLD_DB = -45.0
ONSET_OVER_FLOOR_DB = 15.0
ONSET_WINDOW_FRAMES = 6
ONSET_MIN_FRAMES = 5


def output_gain(prompt_level_db=-19.8):
    return 10.0 ** ((-23.0-float(prompt_level_db))/20.0)


def _short_rms(wav, sample_rate=SAMPLE_RATE):
    x = np.asarray(wav,dtype=np.float32).reshape(-1)
    win = int(sample_rate*.010)
    frames = x.size//win
    y = x[:frames*win].reshape(frames,win)
    return np.sqrt(np.mean(y*y,axis=-1,dtype=np.float32)),win


def _onset_threshold(rms):
    threshold = 10.0**(ONSET_THRESHOLD_DB/20.0)
    if rms.size>=30:
        threshold=max(threshold,float(np.quantile(rms,.1,method='linear')) * 10.0**(ONSET_OVER_FLOOR_DB/20.0))
    return threshold


def speech_onset(wav,sample_rate=SAMPLE_RATE):
    rms,win=_short_rms(wav,sample_rate)
    if np.size(wav)<win*ONSET_WINDOW_FRAMES:
        return None
    above=(rms>_onset_threshold(rms)).astype(np.int32)
    hits=np.flatnonzero(np.lib.stride_tricks.sliding_window_view(above,ONSET_WINDOW_FRAMES).sum(axis=-1)>=ONSET_MIN_FRAMES)
    return int(hits[0])*win if hits.size else None


def trim_lead(wav,sample_rate=SAMPLE_RATE,lead=.08,skip=0.0):
    x=np.asarray(wav,dtype=np.float32)
    onset=speech_onset(x,sample_rate)
    cut=0
    if onset is not None:
        cut=max(onset-int(lead*sample_rate),int(skip*sample_rate))
        cut=min(cut,max(0,onset-int(.02*sample_rate)))
    return x[...,cut:],cut,onset


def trim_trail(wav,sample_rate=SAMPLE_RATE,trail=.30):
    x=np.asarray(wav,dtype=np.float32)
    rms,win=_short_rms(x,sample_rate)
    end=x.size
    if x.size>=win:
        above=np.flatnonzero(rms>_onset_threshold(rms))
        if above.size:
            end=min(x.size,(int(above[-1])+1)*win+int(trail*sample_rate))
    return x[..., :end],end


def soft_limit(wav,knee=.9):
    x=np.asarray(wav,dtype=np.float32)
    mag=np.abs(x)
    over=np.float32(knee)+np.float32(1-knee)*np.tanh((mag-np.float32(knee))/np.float32(1-knee))
    return np.where(mag>np.float32(knee),np.sign(x)*over,x)


def fade_edges(wav,sample_rate=SAMPLE_RATE,fade_in=False,fade_out=True,fade_seconds=.08):
    x=np.asarray(wav,dtype=np.float32)
    fade=int(fade_seconds*sample_rate)
    if x.shape[-1]<=2*fade:
        return x
    out=x.copy()
    ramp=np.linspace(0,1,fade,dtype=np.float32)
    if fade_in:out[..., :fade]*=ramp
    if fade_out:out[..., -fade:]*=ramp[::-1]
    return out


def postprocess_segment(raw_wav,reference_level_db,sample_rate=SAMPLE_RATE):
    x=np.asarray(raw_wav,dtype=np.float32).reshape(-1)
    gain=output_gain(reference_level_db)
    gained=x*np.float32(gain)
    leading,cut,onset=trim_lead(gained,sample_rate)
    trimmed,end=trim_trail(leading,sample_rate)
    # join_segments on one element applies neither join fade.
    final=fade_edges(soft_limit(trimmed),sample_rate,False,True,.08)
    return final,{'gain':gain,'speech_onset_samples':onset,'lead_cut_samples':cut,
                  'trail_end_after_lead_samples':end,'final_samples':int(final.size),
                  'input_samples':int(x.size),'sample_rate_hz':sample_rate}
