"""Close-up prints: an illustration reduced to a few flat tones plus its own line art, printed in the cyber palette.
For faces seen close (node 02's singing profile), where the three-ink split (plates.Inks) loses the mouth.

  T = Tones(path)                    # once
  P = T.warp(anchor, pos, sc, rot)   # (H, W, n+2): tone plates dark->light, lines, fg
  print_tones(fr, P, PALETTE)
"""
import cv2
import numpy as np
from PIL import Image

from lib import H, W, affine, hexc
from cyber import C

# dark -> light. The cyanotype ramp: ink, deep blue, her blue, pale, ice.
RAMP = (C["ink"], hexc("#10245e"), C["blue"], hexc("#8cc4ff"), C["ice"])
LINE = hexc("#04070f")


class Tones:
    def __init__(self, path, cuts=(0.26, 0.44, 0.64, 0.82), scale=1.0, line_k=0.10, line_open=0):
        im = np.asarray(Image.open(path).convert("RGBA")).astype(np.float32) / 255
        if scale != 1:
            im = cv2.resize(im, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        a = im[..., 3]
        rgb = im[..., :3]
        lum = (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]).astype(np.float32)
        sm = cv2.bilateralFilter(lum, 9, 0.10, 5)
        # drawn lines: darker than their surroundings by line_k, and thin
        bg = cv2.GaussianBlur(lum, (0, 0), 3.5)
        lines = ((bg - lum) > line_k).astype(np.float32)
        lines = cv2.morphologyEx(lines, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8)) if line_open else lines
        lines = cv2.GaussianBlur(lines, (0, 0), 0.6)
        fg = np.clip((a - 0.1) / 0.8, 0, 1)
        edges = (0,) + tuple(cuts) + (2,)
        plates = []
        for k in range(len(edges) - 1):
            m = ((sm >= edges[k]) & (sm < edges[k + 1])).astype(np.float32)
            m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
            plates.append(cv2.GaussianBlur(m, (0, 0), 0.7) * fg)
        self.n = len(plates)
        self.plates = np.stack(plates + [np.clip(lines, 0, 1) * fg, fg], axis=2).astype(np.float32)
        self.size = (im.shape[1], im.shape[0])

    def warp(self, anchor, pos, sc, rot=0.0, size=(W, H)):
        M = affine(anchor, pos, sc, rot)
        return cv2.warpAffine(self.plates, M, size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def print_tones(fr, P, ramp=RAMP, line=LINE, line_amt=0.9, alpha=1.0, rim=True, rim_col=None, under=None):
    """Print warped tone plates: flat ramp colours, then the line art, then a thin neon rim on the outer edge."""
    n = P.shape[2] - 2
    fg = P[..., -1]
    if under is not None:
        fr.fill(fg * alpha, under, 1.0)
    for k in range(n):
        fr.fill(P[..., k] * alpha, ramp[k])
    fr.fill(P[..., n] * alpha, line, line_amt)
    if rim:
        e = np.clip(fg - cv2.warpAffine(fg, np.float32([[1, 0, -4], [0, 1, 0]]), (fg.shape[1], fg.shape[0])), 0, 1)
        e = cv2.GaussianBlur(e, (0, 0), 0.9)
        fr.fill(e * alpha, rim_col if rim_col is not None else C["cyan"], 0.9, glow=0.8)
    return fr
