"""A school of small fish streaming along a sinuous ribbon that slowly changes shape (the S-shaped flow of the mood
board). Each fish is a sardine-proportioned silhouette: dark back, silver belly; it flashes when it turns into the
light, so the school shimmers as it bends."""
import math

import cv2
import numpy as np

from cyber import C
from lib import H, W, hexc


def fish_poly(L, flick):
    """Side-view sardine outline, head at +x, in local units. flick: tail beat -1..1 (the fork narrows/widens)."""
    u = np.linspace(0, 0.84, 14)
    hh = 0.105 * np.sin(np.pi * np.clip(u / 0.84, 0, 1) ** 0.62) ** 0.9 + 0.012
    top = np.stack([0.5 - u, -hh], 1)
    bot = np.stack([0.5 - u, hh * 0.9], 1)[::-1]
    sp = 0.10 + 0.04 * flick
    tail = np.array([[-0.34, -0.018], [-0.5, -sp], [-0.43, 0.0], [-0.5, sp], [-0.34, 0.018]])
    body = np.concatenate([top, tail, bot]) * L
    belly = np.concatenate([np.stack([0.5 - u, np.zeros_like(u) + 0.01], 1), bot]) * L
    return body, belly


class School:
    def __init__(self, n=420, seed=0, size=(12, 26), path=None, width=70.0, speed=(0.035, 0.06), dim=1.0):
        g = np.random.default_rng(seed)
        self.n = n
        self.s0 = g.uniform(0, 1, n)
        self.v = g.uniform(*speed, n)
        self.o = g.normal(0, 1, n).astype(np.float32)
        self.o2 = g.normal(0, 1, n).astype(np.float32) * 0.4
        self.L = g.uniform(*size, n)
        self.ph = g.uniform(0, 6.28, n)
        self.fr = g.uniform(5, 9, n)
        self.width = width
        self.dim = dim
        self.path = path or self.default_path

    @staticmethod
    def default_path(s, t):
        """Ribbon centre: enters low left, S-bends, leaves high right; the bends travel and breathe."""
        x = -500 + s * (W + 1000)
        y = 640 - s * 260 + 150 * np.sin(2 * np.pi * (s * 1.3 - 0.11 * t)) + 70 * np.sin(2 * np.pi * (s * 0.55 + 0.05 * t) + 1)
        return x, y

    def draw(self, fr, t, light=(0.3, -1.0), oy=0.0):
        s = (self.s0 + self.v * t) % 1.0
        ds = 0.002
        x, y = self.path(s, t)
        x2, y2 = self.path(s + ds, t)
        tx, ty = x2 - x, y2 - y
        nrm = np.sqrt(tx * tx + ty * ty) + 1e-6
        tx, ty = tx / nrm, ty / nrm
        wid = self.width * (1 + 0.35 * np.sin(2 * np.pi * (s * 2 - 0.2 * t)))
        px = x - ty * self.o * wid + tx * self.o2 * wid
        py = y + tx * self.o * wid + ty * self.o2 * wid + oy
        # each fish's own wobble in heading
        ang = np.arctan2(ty, tx) + 0.18 * np.sin(t * 1.7 + self.ph)
        lx, ly = light
        ll = math.hypot(lx, ly)
        dark = np.zeros((H, W), np.float32)
        silver = np.zeros((H, W), np.float32)
        flash = np.zeros((H, W), np.float32)
        for i in np.argsort(self.L):
            if not (-40 < px[i] < W + 40 and -40 < py[i] < H + 40):
                continue
            body, belly = fish_poly(self.L[i], math.sin(t * self.fr[i] + self.ph[i]))
            c, sn = math.cos(ang[i]), math.sin(ang[i])
            R = np.array([[c, -sn], [sn, c]], np.float32)
            b = (body @ R.T + (px[i], py[i])) * 16
            be = (belly @ R.T + (px[i], py[i])) * 16
            cv2.fillPoly(dark, [b.astype(np.int32)], 1.0, cv2.LINE_AA, shift=4)
            # a flank turned to the light flashes
            k = abs(math.cos(ang[i] - math.atan2(ly, lx) + 0.6 * math.sin(t * 0.9 + self.ph[i])))
            g = k ** 12
            cv2.fillPoly(silver, [be.astype(np.int32)], 0.55 + 0.45 * g, cv2.LINE_AA, shift=4)
            if g > 0.25:
                cv2.fillPoly(flash, [b.astype(np.int32)], g, cv2.LINE_AA, shift=4)
        a = self.dim
        fr.fill(dark, hexc("#0e2448"), 0.95 * a)
        fr.fill(silver * dark, hexc("#a9cbe6"), 0.9 * a, glow=0.1 * a)
        fr.fill(flash * dark, C["ice"], 0.9 * a, glow=0.8 * a)
        return fr
