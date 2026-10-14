"""The sea's surface as scanlines (from above and from below) and the water column under it.

  scan_sea(fr, t, cam_h)     a perspective sea drawn as scanlines (the LiDAR look of node 01), waves rolling in.
  scan_ceiling(fr, t, ...)   the same sea seen from just under it: a ceiling of light above the horizon line.
  water(t, depth_px, ...)    the colour of the water column at a given depth, light shafts from the surface.
"""
import math

import cv2
import numpy as np

from cyber import C
from lib import H, W, hexc, lerp

# water by depth (world px below the surface): sunlit teal-blue -> twilight -> midnight -> abyss
DEPTH = [(0, hexc("#2a86d6")), (350, hexc("#1d62b0")), (900, hexc("#123f86")), (2600, hexc("#0a2459")), (5200, hexc("#06123a")),
         (8500, hexc("#030820")), (12000, hexc("#01030c"))]


def depth_col(d):
    d = np.asarray(d, np.float32)
    out = np.zeros(d.shape + (3,), np.float32)
    out[...] = DEPTH[0][1]
    for (a, ca), (b, cb) in zip(DEPTH[:-1], DEPTH[1:]):
        k = np.clip((d - a) / (b - a), 0, 1)[..., None]
        out = np.where((d >= a)[..., None], ca * (1 - k) + cb * k, out)
    return out


DEPTH_SCALE = 1.0                  # how fast the water darkens per px of descent (a scene may slow its descent)


def water(t, surf_y, depth0):
    """Background of the water column: screen row y is (y - surf_y) px below the surface, plus depth0."""
    ys = np.arange(H, dtype=np.float32)
    d = (np.maximum(0, ys - surf_y) + depth0) * DEPTH_SCALE
    col = depth_col(d)[:, None, :]
    return np.broadcast_to(col, (H, W, 3)).astype(np.float32).copy()


def scan_sea(fr, t, cam_h=6.0, horizon=360, f=900.0, nlines=76, z0=3.0, z1=140.0, extra=None):
    """The sea as scanlines in perspective, drawn far to near; each line hides what is behind it.
    extra(wx, z) -> added height (e.g. the rings from something falling in)."""
    xs = np.arange(-40, W + 40, 8, dtype=np.float32)
    zs = z0 * (z1 / z0) ** (1 - np.arange(nlines) / (nlines - 1))           # far -> near
    for j, z in enumerate(zs):
        wx = (xs - W / 2) / f * z
        h = (0.55 * np.sin(wx * 0.35 + z * 0.55 - t * 1.6) + 0.30 * np.sin(wx * 0.8 - z * 0.3 - t * 2.3 + 1.0)
             + 0.15 * np.sin(wx * 1.9 + z * 1.2 - t * 3.1)) * (0.6 + 0.4 * math.sin(z * 0.05))
        if extra is not None:
            h = h + extra(wx, z)
        y = horizon + f * (cam_h - h) / z
        pts = np.stack([xs, y], 1)
        poly = np.concatenate([pts, [[W + 40, H + 10], [-40, H + 10]]])
        m = np.zeros((H, W), np.float32)
        cv2.fillPoly(m, [(poly * 16).astype(np.int32)], 1.0, cv2.LINE_AA, shift=4)
        near = 1 - j / nlines
        fr.img = fr.img * (1 - m[..., None]) + (hexc("#030814") * 0.6 + hexc("#0a1c44") * 0.4 * (1 - near))[None, None] * m[..., None]
        ln = np.zeros((H, W), np.float32)
        cv2.polylines(ln, [(pts * 16).astype(np.int32)], False, 1.0, 1 if z > 12 else 2, cv2.LINE_AA, shift=4)
        a = min(1.0, 0.2 + 6.0 / z)
        fr.fill(ln, C["cyan"], float(min(1.0, a)), glow=0.45 * float(min(1.0, a)))
    return fr


def _scan_h(wx, z, t):
    return (0.55 * np.sin(wx * 0.35 + z * 0.55 - t * 1.6) + 0.30 * np.sin(wx * 0.8 - z * 0.3 - t * 2.3 + 1.0)
            + 0.15 * np.sin(wx * 1.9 + z * 1.2 - t * 3.1)) * (0.6 + 0.4 * np.sin(z * 0.05))


def scan_ceiling(fr, t, cam_h, horizon, f=900.0, nlines=64, z0=2.0, z1=140.0, alpha=1.0):
    """B seen from below (cam_h < 0): the same scanline sea, now a ceiling of light above the horizon line.
    Near the camera it is overhead and bright (the sky shows through); far off it sinks into the haze at the horizon."""
    xs = np.arange(-40, W + 40, 8, dtype=np.float32)
    zs = z0 * (z1 / z0) ** (1 - np.arange(nlines) / (nlines - 1))           # far -> near
    lo, hi = hexc("#1f6fc0"), hexc("#9adcff")
    covered = np.zeros((H, W), np.float32)
    for j, z in enumerate(zs):
        wx = (xs - W / 2) / f * z
        y = horizon + f * (cam_h - _scan_h(wx, z, t)) / z
        if y.max() < -20:
            continue
        pts = np.stack([xs, y], 1)
        poly = np.concatenate([pts, [[W + 40, -20], [-40, -20]]])
        m = np.zeros((H, W), np.float32)
        cv2.fillPoly(m, [(poly * 16).astype(np.int32)], 1.0, cv2.LINE_AA, shift=4)
        near = (j / (nlines - 1)) ** 1.6
        col = lo * (1 - near) + hi * near
        fr.img = fr.img * (1 - m[..., None] * alpha) + col[None, None] * m[..., None] * alpha
        covered = np.maximum(covered, m)
        ln = np.zeros((H, W), np.float32)
        cv2.polylines(ln, [(pts * 16).astype(np.int32)], False, 1.0, 1 if z > 10 else 2, cv2.LINE_AA, shift=4)
        a = float(min(1.0, 0.25 + 5.0 / z)) * alpha
        fr.fill(ln, C["ice"], a, glow=0.5 * a)
    # the haze where the surface meets the horizon
    ys = np.arange(H, dtype=np.float32)[:, None]
    haze = np.exp(-((ys - horizon) / 40.0) ** 2) * np.ones((1, W), np.float32)
    fr.add(haze, C["cyan"], 0.25 * alpha, glow=0.2 * alpha)
    return covered
