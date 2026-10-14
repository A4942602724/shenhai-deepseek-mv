import numpy as np, sys
S = sys.argv[1]; sr = 22050
x = np.fromfile(S + "/music.f32", dtype=np.float32)
hop, win = 128, 1024
nfr = 1 + (len(x)-win)//hop
idx = np.arange(win)[None,:] + hop*np.arange(nfr)[:,None]
mag = np.log1p(10*np.abs(np.fft.rfft(x[idx]*np.hanning(win), axis=1)))
o = np.r_[0, np.maximum(0, np.diff(mag, axis=0)).sum(1)]
o = o - np.convolve(o, np.ones(64)/64, "same"); o = np.maximum(o, 0)
fps = sr/hop; t = np.arange(nfr)/fps
def score(bpm, seg):
    a, b = seg; m = (t>=a)&(t<b); p = 60/bpm
    ph = np.exp(2j*np.pi*t[m]/p); return np.abs((o[m]*ph).sum())/o[m].sum(), np.angle((o[m]*ph).sum())
segs = {"intro 1-30":(1,30), "drop1 42-62":(42,62), "v2 68-87":(68,87), "drop2 102-119":(102,119), "final 124-180":(124,180)}
bpms = np.arange(80, 200, 0.05)
for name, seg in segs.items():
    sc = np.array([score(b, seg)[0] for b in bpms]); top = bpms[np.argsort(sc)[::-1]]
    picks = []; 
    for b in top:
        if all(abs(b-p)>2 for p in picks): picks.append(b)
        if len(picks)==3: break
    print(f"{name:14s}", "  ".join(f"{p:.2f}({sc[np.abs(bpms-p).argmin()]:.3f})" for p in picks))
# global fit on loud sections
allm = [(42,62),(88,97),(102,119),(124,180)]
sc = np.array([np.mean([score(b,s)[0] for s in allm]) for b in bpms]); B = bpms[sc.argmax()]
print(f"\nglobal best: {B:.2f} BPM")
ph = score(B, (42,180))[1]; p = 60/B; t0 = (-ph/(2*np.pi))*p % p
print(f"beat period {p*1000:.1f} ms, first-beat phase {t0:.3f}s, bar(4 beats) {4*p:.3f}s")
for s in [30.0, 40.0, 42.0, 63.5, 67.8, 87.8, 97.5, 101.5, 119.5, 123.6, 163.5, 181.5]:
    n = round((s-t0)/p); print(f"  boundary ~{s:6.1f}s -> nearest beat #{n} at {t0+n*p:7.3f}s (bar {n//4}, beat {n%4+1})")
