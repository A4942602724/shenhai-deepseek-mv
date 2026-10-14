"""Dream particles for the sea: bokeh, twinkles, bubbles, light shafts. All pure functions of t (seeded, no state)."""
import math

import cv2
import numpy as np

from lib import H, W, blur, value_noise


class Bokeh:
    """Out-of-focus motes: soft discs with a brighter rim, drifting slowly, breathing in and out of focus."""

    def __init__(self, n=40, seed=0, r=(6, 34), vy=(-18, -4), vx=(-6, 6), w=W, h=H):
        g = np.random.default_rng(seed)
        self.n, self.w, self.h = n, w, h
        self.x, self.y = g.uniform(0, w, n), g.uniform(0, h, n)
        self.vx, self.vy = g.uniform(*vx, n), g.uniform(*vy, n)
        self.r = g.uniform(*r, n)
        self.ph = g.uniform(0, 6.28, n)
        self.br = g.uniform(0.3, 1.0, n)

    def mask(self, t, oy=0.0):
        m = np.zeros((self.h, self.w), np.float32)
        for i in range(self.n):
            x = (self.x[i] + self.vx[i] * t + 14 * math.sin(t * 0.3 + self.ph[i])) % (self.w + 80) - 40
            y = (self.y[i] + self.vy[i] * t + oy) % (self.h + 80) - 40
            r = self.r[i] * (0.85 + 0.15 * math.sin(t * 0.7 + self.ph[i]))
            b = self.br[i] * (0.6 + 0.4 * math.sin(t * 0.9 + self.ph[i] * 2))
            p = (int(x * 16), int(y * 16))
            cv2.circle(m, p, int(r * 16), 0.35 * b, -1, cv2.LINE_AA, shift=4)
            cv2.circle(m, p, int(r * 16), 0.55 * b, 1, cv2.LINE_AA, shift=4)
        return blur(m, 1.4)


class Twinkles:
    """Tiny four-pointed glints that flare and die, each on its own clock."""

    def __init__(self, n=60, seed=1, region=(0, 0, W, H), size=(4, 16), rate=(0.4, 1.2)):
        g = np.random.default_rng(seed)
        x0, y0, x1, y1 = region
        self.n = n
        self.x, self.y = g.uniform(x0, x1, n), g.uniform(y0, y1, n)
        self.s = g.uniform(*size, n)
        self.rate = g.uniform(*rate, n)
        self.ph = g.uniform(0, 1, n)

    def mask(self, t, pos=None):
        m = np.zeros((H, W), np.float32)
        for i in range(self.n):
            u = (t * self.rate[i] + self.ph[i]) % 1.0
            k = math.sin(math.pi * min(1.0, u / 0.35)) ** 2 if u < 0.35 else 0.0
            if k < 0.02:
                continue
            x, y = (self.x[i], self.y[i]) if pos is None else pos(i, t)
            L = self.s[i] * k
            for (dx, dy, f) in ((1, 0, 1.0), (0, 1, 1.0), (0.7, 0.7, 0.4), (0.7, -0.7, 0.4)):
                cv2.line(m, (int((x - dx * L * f) * 16), int((y - dy * L * f) * 16)),
                         (int((x + dx * L * f) * 16), int((y + dy * L * f) * 16)), float(k), 1, cv2.LINE_AA, shift=4)
            cv2.circle(m, (int(x * 16), int(y * 16)), int(max(1.0, 1.6 * k) * 16), float(k), -1, cv2.LINE_AA, shift=4)
        return m


class Bubbles:
    """Bubbles rising with a wobble; rims and a highlight, like the node-01 ones."""

    def __init__(self, n=50, seed=2, r=(1.5, 7), vy=(-120, -40)):
        g = np.random.default_rng(seed)
        self.n = n
        self.x, self.y = g.uniform(0, W, n), g.uniform(0, H, n)
        self.r = g.uniform(*r, n) ** 1.0
        self.vy = g.uniform(*vy, n)
        self.ph = g.uniform(0, 6.28, n)

    def mask(self, t, oy=0.0):
        rim = np.zeros((H, W), np.float32)
        for i in range(self.n):
            x = self.x[i] + 6 * math.sin(t * 3.0 + self.ph[i]) * (self.r[i] / 4)
            y = (self.y[i] + self.vy[i] * t + oy) % (H + 40) - 20
            r = self.r[i]
            p = (int(x * 16), int(y * 16))
            cv2.circle(rim, p, int(r * 16), 0.8, 1, cv2.LINE_AA, shift=4)
            cv2.circle(rim, (int((x - r * 0.35) * 16), int((y - r * 0.35) * 16)), int(max(0.6, r * 0.28) * 16), 1.0, -1,
                       cv2.LINE_AA, shift=4)
        return rim


_RAYS = {}


def rays(t, angle=14.0, n=7, seed=3, top=0.0, length=1.0, speed=0.06):
    """Light shafts slanting down from the surface: a few bright bands in a noise field, fading with depth.
    `top`: screen y where the shafts start (the surface); `length`: how far down they reach (fraction of H)."""
    sw, sh = W // 4, H // 4
    key = ("g", sw, sh)
    if key not in _RAYS:
        ys, xs = np.mgrid[0:sh, 0:sw].astype(np.float32)
        _RAYS[key] = (xs, ys)
    xs, ys = _RAYS[key]
    tan = math.tan(math.radians(angle))
    u = (xs + (ys - top / 4) * tan) / sw                         # position across the shafts
    g = np.random.default_rng(seed)
    acc = np.zeros((sh, sw), np.float32)
    for k in range(n):
        c = g.uniform(-0.2, 1.2) + speed * math.sin(t * g.uniform(0.15, 0.4) + k) * 2
        w = g.uniform(0.012, 0.05)
        b = g.uniform(0.4, 1.0) * (0.7 + 0.3 * math.sin(t * g.uniform(0.6, 1.3) + k * 1.7))
        acc += b * np.exp(-((u - c) / w) ** 2)
    depth = np.clip((ys - top / 4) / (sh * length), 0, 1)
    acc *= (ys >= top / 4) * (1 - depth) ** 1.6
    return cv2.resize(acc, (W, H), interpolation=cv2.INTER_LINEAR)
