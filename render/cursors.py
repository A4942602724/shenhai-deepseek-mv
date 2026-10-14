"""The users, as they exist in her world: pixel-art pointers. '#' outline (amber), 'o' fill (dark amber), '.' empty.
Designed here (not copied from any OS cursor set)."""
import math
from functools import lru_cache

import cv2
import numpy as np

from cyber import C, label
from lib import H, W, hexc

ARROW = [
    "#...........",
    "##..........",
    "#o#.........",
    "#oo#........",
    "#ooo#.......",
    "#oooo#......",
    "#ooooo#.....",
    "#oooooo#....",
    "#ooooooo#...",
    "#oooooooo#..",
    "#ooooo#####.",
    "#oo#oo#.....",
    "#o#.#oo#....",
    "##..#oo#....",
    "#....#oo#...",
    ".....#oo#...",
    "......##....",
]

HAND = [
    ".....##..........",
    "....#oo#.........",
    "....#oo#.........",
    "....#oo#.........",
    "....#oo#.........",
    "....#oo###.......",
    "....#oo#oo###....",
    "....#oo#oo#oo##..",
    ".##.#oo#oo#oo#o#.",
    "#oo##oooooooooo#.",
    "#ooo#oooooooooo#.",
    ".#oo#oooooooooo#.",
    "..#oooooooooooo#.",
    "..#ooooooooooo#..",
    "...#oooooooooo#..",
    "...#ooooooooo#...",
    "....#oooooooo#...",
    "....#oooooooo#...",
    "....##########...",
]

HOT = {"arrow": (0.5, 0.5), "hand": (6.0, 0.5)}      # hotspot (col, row) in bitmap cells
FILL = hexc("#1c1408")
_P = {}


@lru_cache(maxsize=8)
def bitmap(kind):
    rows = ARROW if kind == "arrow" else HAND
    w = max(len(r) for r in rows)
    o = np.array([[ch == "#" for ch in r.ljust(w, ".")] for r in rows])
    f = np.array([[ch == "o" for ch in r.ljust(w, ".")] for r in rows])
    return o, f


def appear_times(kind, seed, span):
    key = (kind, seed, span)
    if key not in _P:
        o, f = bitmap(kind)
        r = np.random.default_rng(seed)
        _P[key] = r.uniform(0, span, o.shape).astype(np.float32)
    return _P[key]


def patch(kind, p, t, t_in, seed=0, rez=0.22, flip=False, press=0.0):
    """RGBA + glow patch of a cursor at pixel size p. Pixels rez in one by one (each flashes ice when it lands).
    Returns (rgb, alpha, glow, hotspot_xy) in patch pixels."""
    o, f = bitmap(kind)
    ap = appear_times(kind, seed, rez)
    age = t - t_in - ap
    vis = age >= 0
    flash = np.where(vis, np.exp(-np.maximum(age, 0) * 22), 0).astype(np.float32)
    if flip:
        o, f, vis, flash = o[::-1, ::-1], f[::-1, ::-1], vis[::-1, ::-1], flash[::-1, ::-1]
    def up(a):
        return np.repeat(np.repeat(a.astype(np.float32), p, 0), p, 1)
    O, F, V, FL = up(o), up(f), up(vis), up(flash)
    h, w = O.shape
    rgb = np.zeros((h, w, 3), np.float32)
    a = np.clip((O + F) * V, 0, 1)
    amber = C["amber"] * (1 + 0.35 * press)
    rgb += (F * V)[..., None] * FILL + (O * V)[..., None] * amber
    rgb = rgb * (1 - FL[..., None] * 0.85) + FL[..., None] * 0.85 * C["ice"]
    glow = (O * V)[..., None] * amber * (0.7 + 0.8 * press) + (FL * a)[..., None] * C["ice"] * 0.8
    hx, hy = HOT[kind]
    if flip:
        hx, hy = o.shape[1] - hx, o.shape[0] - hy
    return rgb, a, glow, (hx * p, hy * p)


def paste(fr, pt, x, y, alpha=1.0, shadow=True, flipv=False, wobble=0.0, t=0.0, tint=None):
    """Paste a cursor patch so its hotspot lands on (x, y). flipv+wobble: its reflection under the water."""
    rgb, a, glow, (hx, hy) = pt
    if flipv:
        rgb, a, glow = rgb[::-1], a[::-1], glow[::-1]
        hy = a.shape[0] - hy
    h, w = a.shape
    x0, y0 = int(round(x - hx)), int(round(y - hy))
    if wobble:
        rows = np.arange(h, dtype=np.float32)
        dx = (wobble * np.sin(rows * 0.12 + t * 4.0)).astype(np.float32)
        mx = (np.arange(w, dtype=np.float32)[None, :] - dx[:, None]).astype(np.float32)
        my = np.broadcast_to(rows[:, None], (h, w)).astype(np.float32)
        rgb = cv2.remap(rgb, mx, my, cv2.INTER_LINEAR)
        a = cv2.remap(a, mx, my, cv2.INTER_LINEAR)
        glow = cv2.remap(glow, mx, my, cv2.INTER_LINEAR)
    if tint is not None:
        rgb = rgb * 0.4 + np.asarray(tint, np.float32) * a[..., None] * 0.6
    xa, ya, xb, yb = max(0, x0), max(0, y0), min(W, x0 + w), min(H, y0 + h)
    if xa >= xb or ya >= yb:
        return fr
    sl = (slice(ya - y0, yb - y0), slice(xa - x0, xb - x0))
    if shadow:
        sx, sy = 6, 6
        xs0, ys0 = xa + sx, ya + sy
        xs1, ys1 = min(W, xb + sx), min(H, yb + sy)
        if xs0 < xs1 and ys0 < ys1:
            sa = a[sl][: ys1 - ys0, : xs1 - xs0][..., None] * 0.5 * alpha
            fr.img[ys0:ys1, xs0:xs1] = fr.img[ys0:ys1, xs0:xs1] * (1 - sa) + sa * C["bg0"]
    aa = a[sl][..., None] * alpha
    fr.img[ya:yb, xa:xb] = fr.img[ya:yb, xa:xb] * (1 - aa) + rgb[sl] * alpha
    fr.glow[ya:yb, xa:xb] += glow[sl] * alpha
    return fr


def tag(fr, s, x, y, alpha=1.0):
    """Figma-style name tag next to a cursor: an amber pill with dark mono text."""
    from cyber import mtext
    if not s or alpha <= 0:
        return fr
    m, _ = mtext(s, 15, "mono", "sans")
    w = m.shape[1] + 14
    pill = np.zeros((H, W), np.float32)
    cv2.rectangle(pill, (int(x), int(y)), (int(x + w), int(y + 22)), 1.0, -1)
    fr.fill(pill, C["amber"], 0.9 * alpha, glow=0.25 * alpha)
    label(fr, s, x + 7, y + 11, C["bg0"], 15, alpha, "lc")
    return fr


def click_ring(fr, x, y, t, t0, color=None):
    if t < t0 or t > t0 + 0.35:
        return fr
    k = (t - t0) / 0.35
    m = np.zeros((H, W), np.float32)
    cv2.circle(m, (int(x), int(y)), int(6 + 44 * (1 - (1 - k) ** 3)), 1.0, 2, cv2.LINE_AA)
    fr.fill(m, C["amber"] if color is None else color, 1 - k, glow=1 - k)
    return fr
