"""Corrosion: rust eating an old iron plate from its edges and weak spots, with specks and run-down streaks.
Used sparingly (an accent, not a theme). Never emissive: decay sits darker than the neon.

  R = Plate(seed, reach=260)      # once: the plate's weak places (frame edges, corners, an optional bias map)
  L = R.layers(amount)            # amount 0..1 = how far the rust has got (time)
  paint(fr, L, amt)               # stain, rust body, crust, front, flecks, pits, streaks
"""
import cv2
import numpy as np

from lib import H, W, hexc, value_noise

RUST = dict(core=hexc("#120806"), deep=hexc("#2b1309"), body=hexc("#5e2810"), mid=hexc("#7d3a17"),
            hot=hexc("#9e521e"), fleck=hexc("#c9773a"), stain=hexc("#3a200e"), verd=hexc("#2a6a62"))
_C = {}


def _noise(seed, cell, octaves, h, w):
    key = (seed, cell, octaves, h, w)
    if key not in _C:
        n = value_noise(h, w, cell, seed=seed, octaves=octaves)
        _C[key] = ((n - n.min()) / (n.max() - n.min() + 1e-6)).astype(np.float32)
    return _C[key]


def _warped(seed, cell, h, w, amp=70.0):
    """fbm seen through another fbm: fractal, organic patch edges."""
    key = ("w", seed, cell, h, w, amp)
    if key not in _C:
        base = _noise(seed, cell, 5, h, w)
        dx = (_noise(seed + 1, cell, 3, h, w) - 0.5) * 2 * amp
        dy = (_noise(seed + 2, cell, 3, h, w) - 0.5) * 2 * amp
        ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
        _C[key] = cv2.remap(base, xs + dx, ys + dy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return _C[key]


def _grain(seed, h, w):
    key = ("g", seed, h, w)
    if key not in _C:
        r = np.random.default_rng(seed)
        g = r.random((h // 2, w // 2)).astype(np.float32)
        g = cv2.resize(cv2.GaussianBlur(g, (0, 0), 0.7), (w, h), interpolation=cv2.INTER_LINEAR)
        _C[key] = (g - g.min()) / (g.max() - g.min() + 1e-6)
    return _C[key]


def border_exposure(h, w, reach=260.0, corner=1.6):
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    ex = np.clip(1 - np.minimum(xs, w - 1 - xs) / reach, 0, 1)
    ey = np.clip(1 - np.minimum(ys, h - 1 - ys) / reach, 0, 1)
    return np.clip(np.maximum(ex, ey) ** 1.6 + corner * (ex * ey) ** 1.2, 0, 1.4).astype(np.float32)


class Plate:
    def __init__(self, seed=0, reach=260.0, h=H, w=W, bias=None, specks=900, cell=150):
        self.h, self.w, self.seed = h, w, seed
        E = border_exposure(h, w, reach)
        if bias is not None:
            E = np.maximum(E, bias)
        self.E = E
        self.n = _warped(seed, cell, h, w)
        r = np.random.default_rng(seed + 17)
        # specks: tiny pits that pepper the plate, denser where it is exposed; each has a birth time
        sp = []
        while len(sp) < specks:
            x, y = int(r.uniform(0, w)), int(r.uniform(0, h))
            if r.random() < 0.08 + 0.9 * min(1.0, float(E[y, x])):
                sp.append((x, y, int(r.choice([1, 1, 1, 2, 2, 3, 4, 6])), float(r.uniform(0, 1))))
        self.specks = sp
        # rust runs down only in a few narrow channels
        cols = np.zeros(w, np.float32)
        for _ in range(int(w * 0.022)):
            x = int(r.uniform(0, w))
            cols[x:x + int(r.choice([1, 1, 2, 3]))] = r.uniform(0.35, 1.0)
        self.cols = cols

    def layers(self, amount, drip=True, verd=0.35):
        h, w = self.h, self.w
        a = float(np.clip(amount, 0, 1))
        f = self.n * 0.55 + self.E * 0.62
        thr = 1.12 - 0.62 * a
        g = _grain(self.seed, h, w)
        fg_ = f + (g - 0.5) * 0.05
        stain = np.clip((fg_ - (thr - 0.10)) / 0.08, 0, 1)
        body = np.clip((fg_ - thr) / 0.02, 0, 1)
        inner = np.clip((fg_ - thr - 0.035) / 0.03, 0, 1)
        core = np.clip((fg_ - thr - 0.13) / 0.05, 0, 1)
        sk = np.zeros((h, w), np.float32)
        for (x, y, s, b) in self.specks:
            if b < a:
                grow = min(1.0, (a - b) / 0.25)
                cv2.circle(sk, (x, y), max(1, int(round(s * (0.4 + 0.6 * grow)))), 1.0, -1, cv2.LINE_AA)
        body = np.maximum(body, sk)
        front = np.clip(body - inner, 0, 1) * (1 - sk)
        mottle = (g > 0.58).astype(np.float32) * inner * (1 - core)
        fleck = (g > 0.90).astype(np.float32) * inner * (1 - core)
        pits = (g < 0.12).astype(np.float32) * core
        out = dict(stain=np.clip(stain - body, 0, 1), body=body, front=front, mottle=mottle, core=core,
                   fleck=fleck, pits=pits)
        if verd:
            v = _noise(self.seed + 9, 50, 2, h, w)
            out["verd"] = np.clip((v - 0.76) / 0.05, 0, 1) * inner * (1 - core) * verd
        if drip:
            # rust runs down the plate: streaks under the body, of uneven length per column
            k = 161
            src = cv2.erode(body, np.ones((1, 5), np.uint8))
            st = cv2.blur(src, (2, k), anchor=(1, k - 1))          # value at y = mean of the k pixels above it
            out["drip"] = np.clip(st * 2.0 * self.cols[None, :] - body, 0, 1) * 0.6
        return out


def paint(fr, L, amt=1.0, mask=None):
    """Print the corrosion masks onto a cyber Frame. `mask` limits it (e.g. to a picture's footprint)."""
    if amt <= 0:
        return fr

    def m(k):
        x = L[k]
        return x * mask if mask is not None else x
    fr.fill(m("stain"), RUST["stain"], 0.35 * amt)
    if "drip" in L:
        fr.fill(m("drip"), RUST["deep"], 0.75 * amt)
    fr.fill(m("body"), RUST["body"], 0.96 * amt)
    fr.fill(m("mottle"), RUST["mid"], 0.8 * amt)
    if "verd" in L:
        fr.fill(m("verd"), RUST["verd"], 0.75 * amt)
    fr.fill(m("front"), RUST["hot"], 0.65 * amt)
    fr.fill(m("fleck"), RUST["fleck"], 0.7 * amt)
    fr.fill(m("core"), RUST["deep"], 0.95 * amt)
    fr.fill(m("pits"), RUST["core"], 0.95 * amt)
    return fr
