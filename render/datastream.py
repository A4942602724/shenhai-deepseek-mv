"""Information flow for the descent, kept faint so it never takes the eye off her:
  Rain     columns of hex glyphs rising (the data goes up; she goes down)
  Streams  thin fibres slanting up toward the surface, packets of light running along them
All draw into glyphs.Layers; the caller flushes them with its colours."""
import math
import zlib

import cv2
import numpy as np

from glyphs import glyph, stamp
from lib import H, W

SYMS = "0123456789abcdef0123456789abcdef{}[]<>=/:;.#"
def _h(*a):
    return zlib.crc32(repr(a).encode())


class Rain:
    def __init__(self, n=36, seed=0, avoid=(700, 1220), size=12, lh=15):
        g = np.random.default_rng(seed)
        cols = []
        while len(cols) < n:
            x = g.uniform(30, W - 30)
            behind = avoid[0] < x < avoid[1]
            if behind and g.random() < 0.75:
                continue
            cols.append((x, g.uniform(30, 110), int(g.integers(8, 24)), g.uniform(0, H * 2), g.uniform(0.25, 1.0) * (0.4 if behind else 1.0),
                         g.uniform(3, 9)))
        self.cols, self.size, self.lh = cols, size, lh

    def draw(self, L, t, oy=0.0, amt=1.0):
        for ci, (x, v, n, y0, b, rate) in enumerate(self.cols):
            span = H + n * self.lh + 60
            head = (y0 - v * t + oy) % span - 30
            for i in range(n):
                y = head + i * self.lh
                if y < -20 or y > H + 20:
                    continue
                ch = SYMS[_h(ci, i, int(t * rate) if i else int(t * rate * 3)) % len(SYMS)]
                a = b * amt * (1 - i / n) ** 1.6
                stamp(L["head" if i == 0 else "rain"], glyph(ch, self.size, "mono"), x, y, a)


class Streams:
    def __init__(self, n=7, seed=1):
        g = np.random.default_rng(seed)
        self.s = []
        for k in range(n):
            x0 = g.uniform(-200, W + 200)
            self.s.append(dict(x0=x0, slant=g.uniform(-0.45, 0.45), amp=g.uniform(30, 110), wl=g.uniform(500, 1100),
                               ph=g.uniform(0, 6.28), v=g.uniform(0.10, 0.22), n=int(g.integers(5, 12)),
                               off=g.uniform(0, 1, 16), b=g.uniform(0.35, 1.0)))

    def curve(self, st, s, t):
        """s 0 (below the frame) .. 1 (above it)."""
        y = H + 120 - s * (H + 240)
        x = st["x0"] + st["slant"] * (H - y) + st["amp"] * np.sin((H - y) / st["wl"] * 6.28 + st["ph"] + 0.3 * t)
        return x, y

    def draw(self, L, t, amt=1.0):
        ss = np.linspace(0, 1, 90)
        for st in self.s:
            x, y = self.curve(st, ss, t)
            pts = (np.stack([x, y], 1) * 16).astype(np.int32)
            fib = np.zeros((H, W), np.float32)
            cv2.polylines(fib, [pts], False, 0.5 * st["b"] * amt, 1, cv2.LINE_AA, shift=4)
            np.maximum(L["fiber"], fib, out=L["fiber"])
            for k in range(st["n"]):
                s0 = (st["off"][k] + st["v"] * t) % 1.0
                seg_s = np.linspace(max(0, s0 - 0.035), s0, 6)
                px, py = self.curve(st, seg_s, t)
                p = (np.stack([px, py], 1) * 16).astype(np.int32)
                cv2.polylines(L["pkt"], [p], False, float(st["b"] * amt), 2, cv2.LINE_AA, shift=4)
