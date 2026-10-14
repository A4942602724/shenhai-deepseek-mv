"""Specimens traced from public-domain natural-history plates (assets/ref/, see SOURCES.md).
A plate's ink is lifted off its paper and becomes light: luminous line drawings on the dark sea.

  ink_crop(key)                      -> float ink mask of one specimen (cached, working resolution)
  draw(fr, key, t, x, y, scale, ...) -> print it: drawn in from the bell to the tentacle tips, swaying, glowing
"""
import math

import cv2
import numpy as np
from PIL import Image

from cyber import C
from lib import H, ROOT, W

REF = ROOT / "assets" / "ref"
# (plate file, crop box x0, y0, x1, y1 in plate pixels, anchor = top of the bell in crop-relative fraction)
SPEC = {
    "lepto5": ("Haeckel_Leptomedusae.jpg", (60, 560, 1180, 2200)),
    "lepto4": ("Haeckel_Leptomedusae.jpg", (1260, 680, 2300, 1950)),
    "lepto6": ("Haeckel_Leptomedusae.jpg", (1090, 2010, 2090, 3300)),
    "lepto2": ("Haeckel_Leptomedusae.jpg", (770, 70, 1980, 560)),
    "lepto1": ("Haeckel_Leptomedusae.jpg", (50, 90, 660, 660)),
    "lepto3": ("Haeckel_Leptomedusae.jpg", (40, 2140, 970, 3270)),
    "disco4": ("Haeckel_Discomedusae_8.jpg", (1460, 1420, 2290, 3000)),
    "disco3": ("Haeckel_Discomedusae_8.jpg", (30, 15, 930, 1160)),
    "disco2": ("Haeckel_Discomedusae_8.jpg", (610, 2010, 1660, 3080)),
    "siph": ("Haeckel_Siphonophorae_7.jpg", (0, 0, 2318, 3258)),
}
_PLATES, _CROPS = {}, {}


def _plate_ink(fname):
    """Ink = how much darker than the local paper a pixel is (colour plates: also how saturated)."""
    if fname not in _PLATES:
        im = np.asarray(Image.open(REF / fname).convert("RGB"), np.float32) / 255
        lum = im @ np.array([0.299, 0.587, 0.114], np.float32)
        paper = cv2.dilate(lum, np.ones((31, 31), np.uint8))
        paper = cv2.GaussianBlur(paper, (0, 0), 25)
        sat = im.max(2) - im.min(2)
        ink = np.clip((paper - lum) / np.maximum(paper * 0.55, 0.05), 0, 1)
        ink = np.maximum(ink, np.clip((sat - 0.10) * 1.6, 0, 1) * 0.8)
        _PLATES[fname] = ink.astype(np.float32)
    return _PLATES[fname]


def ink_crop(key, max_dim=760):
    if key not in _CROPS:
        fname, (x0, y0, x1, y1) = SPEC[key]
        ink = _plate_ink(fname)[y0:y1, x0:x1]
        s = max_dim / max(ink.shape)
        ink = cv2.resize(ink, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        ink = np.clip((ink - 0.14) / 0.86, 0, 1)                     # lift the paper grain off: no box behind it
        # keep the specimen, drop neighbours that strayed into the crop: the biggest blob after bridging gaps
        solid = cv2.dilate((ink > 0.2).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)))
        n, lab, st, _ = cv2.connectedComponentsWithStats(solid, connectivity=8)
        if n > 1:
            mass = np.bincount(lab.ravel(), weights=ink.ravel(), minlength=n)
            mass[0] = 0
            keep = cv2.dilate((lab == int(mass.argmax())).astype(np.uint8), np.ones((5, 5), np.uint8))
            ink = ink * cv2.GaussianBlur(keep.astype(np.float32), (0, 0), 2)
        h, w = ink.shape
        # feather the crop edge so neighbours on the plate fade out instead of being cut
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        d = np.minimum.reduce([xx, yy, w - 1 - xx, h - 1 - yy]) / (0.08 * min(h, w))
        ink = ink * np.clip(d, 0, 1) ** 1.5
        ink = np.clip(ink * 1.25, 0, 1) ** 0.85
        # reveal order: from the top of the bell down the tentacles, with grain so it reads as being drawn
        rng = np.random.default_rng(len(key))
        order = yy / h * 0.85 + cv2.GaussianBlur(rng.random((h, w)).astype(np.float32), (0, 0), 6) * 0.3
        order = (order - order.min()) / (order.max() - order.min())
        _CROPS[key] = (ink.astype(np.float32), order.astype(np.float32))
    return _CROPS[key]


def draw(fr, key, t, x, y, scale, t_in=None, build=0.6, rot=0.0, alpha=1.0, color=None, glow=0.55, sway=1.0,
         seed=0, edge_col=None, pulse=0.0):
    """Draw specimen `key` with its top-centre at (x, y). t_in: when it starts being drawn (None = already there).
    pulse 0..1: the bell contracting (narrower, shorter), the tentacles following less."""
    ink, order = ink_crop(key)
    h, w = ink.shape
    m = ink
    front = None
    if t_in is not None:
        if t < t_in:
            return fr
        p = (t - t_in) / build
        if p < 1.15:
            m = ink * np.clip((p - order) / 0.06, 0, 1)
            front = ink * np.exp(-((order - p) / 0.025) ** 2)
    if sway:
        # swimming: the bell breathes, tentacles trail in a slow wave that grows toward the tips
        yy = np.arange(h, dtype=np.float32)[:, None]
        xx = np.arange(w, dtype=np.float32)[None, :]
        amp = sway * (yy / h) ** 1.6 * w * 0.035
        bw = np.clip(1 - yy / (0.42 * h), 0, 1) ** 0.5 + 0.25 * np.clip(1 - yy / h, 0, 1)
        cx = w / 2
        mx = (cx + (xx - cx) * (1 + 0.17 * pulse * bw) + amp * np.sin(yy / h * 6.0 - t * 1.4 + seed)).astype(np.float32)
        my = np.broadcast_to(yy * (1 + 0.025 * math.sin(t * 1.6 + seed) + 0.06 * pulse * bw), (h, w)).astype(np.float32)
        m = cv2.remap(m, mx, my, cv2.INTER_LINEAR)
        if front is not None:
            front = cv2.remap(front, mx, my, cv2.INTER_LINEAR)
    c, s = math.cos(math.radians(-rot)) * scale, math.sin(math.radians(-rot)) * scale
    M = np.float32([[c, -s, 0], [s, c, 0]])
    M[:, 2] = np.float32([x, y]) - M[:, :2] @ np.float32([w / 2, 0])
    mf = cv2.warpAffine(m, M, (W, H), flags=cv2.INTER_LINEAR)
    fr.fill(mf, C["cyan"] if color is None else color, alpha, glow=glow * alpha)
    if front is not None:
        ff = cv2.warpAffine(front, M, (W, H), flags=cv2.INTER_LINEAR)
        fr.fill(ff * 0.7, C["ice"] if edge_col is None else edge_col, alpha, glow=0.7 * alpha)
    return fr


def sample_points(key, n, seed=0):
    """Points distributed like the specimen's ink (crop pixel coords, origin top-left) - for turning it into particles."""
    ink, _ = ink_crop(key)
    p = ink.ravel() ** 1.5
    p = p / p.sum()
    r = np.random.default_rng(seed)
    idx = r.choice(len(p), size=n, p=p)
    h, w = ink.shape
    return np.stack([idx % w, idx // w], 1).astype(np.float32), (w, h)
