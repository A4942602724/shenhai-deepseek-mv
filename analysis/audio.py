import numpy as np, sys
S = sys.argv[1]
sr = 22050
x = np.fromfile(S + "/music.f32", dtype=np.float32)
r = np.fromfile(S + "/ref.f32", dtype=np.float32)
# --- alignment check (first 60s, downsampled envelope xcorr) ---
def env(a, hop=220):
    n = len(a)//hop
    return np.sqrt((a[:n*hop].reshape(n,hop)**2).mean(1))
ex, er = env(x[:sr*60]), env(r[:sr*60])
ex = (ex-ex.mean())/ex.std(); er=(er-er.mean())/er.std()
c = np.correlate(er, ex, mode="full"); lag = c.argmax()-(len(ex)-1)
print(f"ref-vs-music offset: {lag*220/sr*1000:.0f} ms  (corr {c.max()/len(ex):.3f})")
# --- frame features ---
hop, win = 512, 2048
nfr = 1 + (len(x)-win)//hop
idx = np.arange(win)[None,:] + hop*np.arange(nfr)[:,None]
w = np.hanning(win).astype(np.float32)
mag = np.abs(np.fft.rfft(x[idx]*w, axis=1))
rms = np.sqrt((x[idx]**2).mean(1))
logm = np.log1p(mag*10)
flux = np.maximum(0, np.diff(logm, axis=0)).sum(1); flux = np.r_[0, flux]
fps = sr/hop
# low band (kick) flux
freqs = np.fft.rfftfreq(win, 1/sr)
lowb = logm[:, freqs<150]
lflux = np.r_[0, np.maximum(0, np.diff(lowb,axis=0)).sum(1)]
# --- tempo via autocorrelation of onset ---
o = flux - np.convolve(flux, np.ones(16)/16, mode="same"); o = np.maximum(o,0)
ac = np.correlate(o, o, mode="full")[len(o)-1:]
lags = np.arange(len(ac)); bpm = 60*fps/np.maximum(lags,1)
m = (bpm>70)&(bpm<200)
best = lags[m][ac[m].argmax()]
print(f"tempo estimate: {60*fps/best:.1f} BPM")
cands = sorted([(ac[l], 60*fps/l) for l in lags[m]], reverse=True)[:8]
print("top lags:", [f"{b:.1f}" for _,b in cands])
# --- energy per 2s block, printed as bar ---
blk = int(2*fps)
print("\n t(s)   RMS(dB)  brightness  bar")
cent = (mag*freqs[None,:]).sum(1)/(mag.sum(1)+1e-9)
for i in range(0, nfr, blk):
    e = 20*np.log10(rms[i:i+blk].mean()+1e-9); b = cent[i:i+blk].mean()
    print(f"{i/fps:6.1f}  {e:6.1f}  {b:7.0f}  " + "#"*max(0,int((e+40)*1.2)))
np.save(S+"/rms.npy", rms); np.save(S+"/flux.npy", flux); np.save(S+"/lflux.npy", lflux)
