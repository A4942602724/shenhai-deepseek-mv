"""Glyph sprites for typographic pictures (concrete poetry) and glyph particles.

  glyph(ch, size, role, rot)  -> float mask, centred, cached
  Layers                      -> named accumulation masks, flushed into a cyber.Frame with one fill per colour
"""
from functools import lru_cache

import cv2
import numpy as np
from PIL import Image, ImageDraw

from cyber import _font
from lib import H, W


@lru_cache(maxsize=8192)
def glyph(ch, size, role="serif", rot=0):
    size = int(size)
    s = int(size * 1.6) + 4
    im = Image.new("L", (s, s), 0)
    ImageDraw.Draw(im).text((s / 2, s / 2), ch, fill=255, font=_font(role, size), anchor="mm")
    m = np.asarray(im, np.float32) / 255
    if rot:
        M = cv2.getRotationMatrix2D((s / 2, s / 2), rot, 1.0)
        m = cv2.warpAffine(m, M, (s, s), flags=cv2.INTER_LINEAR)
    return m


def stamp(acc, m, cx, cy, a=1.0):
    """max-composite mask m (centred on cx, cy) into acc, scaled by a."""
    h, w = m.shape
    x0, y0 = int(round(cx - w / 2)), int(round(cy - h / 2))
    xa, ya, xb, yb = max(0, x0), max(0, y0), min(acc.shape[1], x0 + w), min(acc.shape[0], y0 + h)
    if xa >= xb or ya >= yb or a <= 0:
        return
    sub = m[ya - y0:yb - y0, xa - x0:xb - x0]
    if a != 1.0:
        sub = sub * a
    np.maximum(acc[ya:yb, xa:xb], sub, out=acc[ya:yb, xa:xb])


class Layers:
    """Named accumulation masks. flush(fr, {name: (colour, amount, glow)}) fills each one into the frame."""

    def __init__(self):
        self.m = {}

    def __getitem__(self, k):
        if k not in self.m:
            self.m[k] = np.zeros((H, W), np.float32)
        return self.m[k]

    def flush(self, fr, spec, order=None):
        for k in (order or spec.keys()):
            if k in self.m:
                col, amt, glow = spec[k]
                fr.fill(np.clip(self.m[k], 0, 1), col, amt, glow)
        return fr
